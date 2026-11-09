"""양자화 결과에 빠진 텐서(보통 MTP)를 원본(0 채운 BF16)에서 옮겨 넣는다.

  python graft_missing.py --src <0 채운 BF16 폴더> --dst <양자화 결과 폴더> [--prefix mtp.] [--prefix model.visual.]

- transformers 의 Qwen3_5 모델에는 MTP 모듈이 없어, AutoRound 결과에서 mtp.* 가 빠진다(dbirks·ababaka 판은 MTP 를 BF16 으로 따로 둔다).
- 빠진 것만 model_extra_tensors.safetensors 하나에 바이트 그대로 담고, 색인에 그 파일로 적는다.
- 같은 키가 두 파일에 있으면 거부한다(HyperQwen third-party 문서의 SPEC=mtp 모양 불일치 사고).
- 끝에 원본의 모든 모듈이 결과에 있는지(양자화로 weight → weight_packed·weight_scale 로 바뀐 것 포함) 센다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_io  # noqa: E402

EXTRA = "model_extra_tensors.safetensors"


def module_of(key: str) -> str:
    return key.rsplit(".", 1)[0]


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
    if EXTRA in set(didx.values()) or (dst / EXTRA).exists():
        print(f"graft: {EXTRA} 가 이미 있다", file=sys.stderr)
        return 1
    dmods = {module_of(k) for k in didx}
    take = sorted(k for k in sidx if k.startswith(prefixes) and module_of(k) not in dmods and k not in didx)
    clash = sorted(k for k in sidx if k.startswith(prefixes) and k in didx)
    if not take:
        print(json.dumps({"grafted": 0, "already_present": len(clash)}))
        return 0
    by_shard: dict[str, list[str]] = {}
    for k in take:
        by_shard.setdefault(sidx[k], []).append(k)
    plan, handles = [], []
    try:
        for shard, keys in sorted(by_shard.items()):
            h, start = st_io.read_header(src / shard)
            f = open(src / shard, "rb")
            handles.append(f)
            for k in keys:
                meta = h[k]
                plan.append((k, meta, (lambda m, fh, s: (lambda out: st_io.copy_bytes(fh, s, m, out)))(meta, f, start)))
        st_io.write_file(dst / EXTRA, plan, {"format": "pt"})
    finally:
        for f in handles:
            f.close()
    st_io.check_intact(dst / EXTRA)
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
    print(json.dumps({"grafted": len(take), "file": EXTRA, "src_modules_missing_in_dst": len(missing),
                      "missing_examples": missing[:5]}, ensure_ascii=False))
    return 0 if not missing else 3


if __name__ == "__main__":
    sys.exit(main())
