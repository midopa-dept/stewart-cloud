"""양자화 결과에 빠진 텐서(보통 MTP)를 원본(0 채운 BF16)에서 옮겨 넣는다.

  python graft_missing.py --src <0 채운 BF16 폴더> --dst <양자화 결과 폴더> [--prefix mtp.] [--prefix model.visual.]

- transformers 의 Qwen3_5 모델에는 MTP 모듈이 없어, AutoRound 결과에서 mtp.* 가 빠진다(dbirks·ababaka 판은 MTP 를 BF16 으로 따로 둔다).
- 빠진 것만 model_extra_tensors.safetensors 하나에 바이트 그대로 담고, 색인에 그 파일로 적는다.
- auto-round 0.14.2 는 저장할 때 빠진 텐서를 스스로 model_extra_tensors.safetensors 에 옮긴다(utils/missing_tensors.py,
  compressed-tensors 형식이면 다시 양자화하지 않는다). 그 파일이 있으면 키마다 원본과 dtype·모양·바이트(sha256)를 대조하고,
  색인과 파일의 키가 같은지 본다. 하나라도 다르면 거부한다. 그 뒤에도 빠진 것이 있으면 그 파일에 합쳐 다시 쓴다
  (HyperQwen prepare/quant_mtp.py 는 MTP 선형층이 한 파일에 있어야 한다 — assert len(shards) == 1).
- 같은 키가 두 파일에 있으면 거부한다(HyperQwen third-party 문서의 SPEC=mtp 모양 불일치 사고).
- 끝에 원본의 모든 모듈이 결과에 있는지(양자화로 weight → weight_packed·weight_scale 로 바뀐 것 포함) 센다.
  종료 코드: 0 다 있음, 1 거부(대조 실패·색인 불일치), 3 빠진 모듈이 남음.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_io  # noqa: E402

EXTRA = "model_extra_tensors.safetensors"


class Refuse(Exception):
    pass


def _sha(path: Path, start: int, meta: dict) -> str:
    b, e = meta["data_offsets"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start + b)
        left = e - b
        while left:
            chunk = f.read(min(st_io.CHUNK, left))
            if not chunk:
                raise Refuse(f"{path.name} 가 머리보다 짧다")
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()


def verify_existing(src: Path, sidx: dict, dst: Path, didx: dict) -> int:
    """dst 의 EXTRA 가 원본에서 바이트 그대로 옮긴 것인지. 맞으면 키 수, 아니면 Refuse."""
    if not (dst / EXTRA).exists():
        raise Refuse(f"색인은 {EXTRA} 를 가리키는데 파일이 없다")
    st_io.check_intact(dst / EXTRA)
    h, start = st_io.read_header(dst / EXTRA)
    on_disk = {n: m for n, m in st_io.tensors(h)}
    listed = {k for k, v in didx.items() if v == EXTRA}
    if set(on_disk) != listed:
        raise Refuse(f"{EXTRA} 의 키와 색인이 다르다(파일에만 {len(set(on_disk) - listed)}, 색인에만 {len(listed - set(on_disk))})")
    heads: dict = {}
    bad = {"not_in_src": 0, "dtype_or_shape": 0, "bytes": 0}
    for k, m in sorted(on_disk.items()):
        if k not in sidx:
            bad["not_in_src"] += 1
            continue
        shard = sidx[k]
        if shard not in heads:
            heads[shard] = st_io.read_header(src / shard)
        sh, sstart = heads[shard]
        sm = sh[k]
        if (sm["dtype"], sm["shape"]) != (m["dtype"], m["shape"]):
            bad["dtype_or_shape"] += 1
            continue
        if _sha(src / shard, sstart, sm) != _sha(dst / EXTRA, start, m):
            bad["bytes"] += 1
    if any(bad.values()):
        raise Refuse(f"{EXTRA}(auto-round 가 쓴 것)가 원본과 다르다: {json.dumps(bad)}")
    return len(on_disk)


def module_of(key: str) -> str:
    return key.rsplit(".", 1)[0]


def write_graft(src: Path, sidx: dict, dst: Path, name: str, take: list, keep: bool = False) -> None:
    """take 를 원본에서 바이트 그대로 dst/name 에 쓴다. keep 이면 dst/name 에 이미 있는 텐서를 앞에 그대로 두고 합쳐 다시 쓴다
    (write_file 은 임시 파일에 다 쓴 뒤 이름을 바꾸므로 읽는 중인 원래 파일은 끝까지 온전하다)."""
    by_shard: dict[str, list[str]] = {}
    for k in take:
        by_shard.setdefault(sidx[k], []).append(k)
    plan, handles = [], []
    try:
        if keep:
            h, start = st_io.read_header(dst / name)
            f = open(dst / name, "rb")
            handles.append(f)
            for k, meta in st_io.tensors(h):
                plan.append((k, meta, (lambda m, fh, s: (lambda out: st_io.copy_bytes(fh, s, m, out)))(meta, f, start)))
        for shard, keys in sorted(by_shard.items()):
            h, start = st_io.read_header(src / shard)
            f = open(src / shard, "rb")
            handles.append(f)
            for k in keys:
                meta = h[k]
                plan.append((k, meta, (lambda m, fh, s: (lambda out: st_io.copy_bytes(fh, s, m, out)))(meta, f, start)))
        st_io.write_file(dst / name, plan, {"format": "pt"})
    finally:
        for f in handles:
            f.close()
    st_io.check_intact(dst / name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--prefix", action="append", default=None)
    a = ap.parse_args(argv)
    src, dst = Path(a.src), Path(a.dst)
    prefixes = tuple(a.prefix or ["mtp."])
    sidx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    didx_all = json.loads((dst / "model.safetensors.index.json").read_text())
    didx = didx_all["weight_map"]
    try:
        verified = None
        if EXTRA in set(didx.values()) or (dst / EXTRA).exists():
            verified = verify_existing(src, sidx, dst, didx)
    except (Refuse, ValueError) as e:
        print(f"graft: {e}", file=sys.stderr)
        return 1
    dmods = {module_of(k) for k in didx}
    take = sorted(k for k in sidx if k.startswith(prefixes) and module_of(k) not in dmods and k not in didx)
    if take:
        write_graft(src, sidx, dst, EXTRA, take, keep=verified is not None)
        for k in take:
            didx[k] = EXTRA
    total = 0
    for shard in sorted(set(didx.values())):
        h, _ = st_io.read_header(dst / shard)
        total += sum(st_io.nbytes(m) for _, m in st_io.tensors(h))
        on_disk = {n for n, _ in st_io.tensors(h)}
        listed = {k for k, v in didx.items() if v == shard}
        if on_disk != listed:
            print(f"graft: {shard} 의 키와 색인이 다르다(파일에만 {len(on_disk - listed)}, 색인에만 {len(listed - on_disk)})",
                  file=sys.stderr)
            return 1
    didx_all.setdefault("metadata", {})["total_size"] = total
    (dst / "model.safetensors.index.json").write_text(json.dumps(didx_all, ensure_ascii=False, indent=2) + "\n")
    missing = sorted({module_of(k) for k in sidx} - {module_of(k) for k in didx})
    print(json.dumps({"grafted": len(take), "file": EXTRA if take else None, "verified_existing": verified,
                      "src_modules_missing_in_dst": len(missing), "missing_examples": missing[:5]}, ensure_ascii=False))
    return 0 if not missing else 3


if __name__ == "__main__":
    sys.exit(main())
