"""MLP 중간 크기를 0 으로 채워 늘린다. demo2 는 층마다 뉴런 하나를 더해 17409(MTP 는 17408)라 4비트 그룹(128)·Marlin 에 맞지 않는다.

  python pad_intermediate.py --src <BF16 폴더> --dst <새 폴더> --to 17536

- 바꾸는 텐서: model.language_model.layers.N.mlp.{gate,up,down}_proj.weight, mtp.layers.N.mlp.{gate,up,down}_proj.weight
  - gate·up [I, H] → [T, H]: 새 행은 0
  - down    [H, I] → [H, T]: 새 열은 0
  - 새 뉴런은 gate=0 → SiLU(0)=0, up=0 → 곱 0, down 열 0 → 더하는 값 0. 그래서 출력이 원래와 같다(수학적으로; 행렬 크기가 달라 부동소수 순서 차이만 남는다).
- config.json 의 text_config.intermediate_size(와 맨 위 intermediate_size 가 있으면 그것)를 T 로.
- 그 밖 텐서·파일은 바이트 그대로. 색인의 total_size 만 다시 센다.
- 결과 요약: <dst>/pad_report.json
torch 없이 바이트만 다룬다(dtype 무관).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_io  # noqa: E402

MLP_RE = re.compile(r"^(model\.language_model\.layers\.\d+|mtp\.layers\.\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
MAX_PAD = 256      # 이보다 많이 늘리는 것은 실수로 본다


def plan_tensor(name: str, meta: dict, to: int) -> tuple[dict, str | None, int]:
    """(새 meta, 종류 'rows'|'cols'|None, 원래 I)."""
    m = MLP_RE.match(name)
    if not m:
        return meta, None, 0
    kind = m.group(2)
    shape = list(meta["shape"])
    if len(shape) != 2:
        raise ValueError(f"{name}: 2차원이 아니다 {shape}")
    i = shape[0] if kind in ("gate_proj", "up_proj") else shape[1]
    if i > to or to - i > MAX_PAD:
        raise ValueError(f"{name}: 중간 크기 {i} 를 {to} 로 늘릴 수 없다")
    if i == to:
        return meta, None, i
    new = dict(meta)
    if kind in ("gate_proj", "up_proj"):
        new["shape"] = [to, shape[1]]
        return new, "rows", i
    new["shape"] = [shape[0], to]
    return new, "cols", i


def pad_writer(src, data_start: int, meta: dict, kind: str, to: int):
    es = st_io.DTYPE_SIZE[meta["dtype"]]
    r, c = meta["shape"]

    def rows(out):
        st_io.copy_bytes(src, data_start, meta, out)
        out.write(b"\0" * ((to - r) * c * es))

    def cols(out):
        b, _ = meta["data_offsets"]
        src.seek(data_start + b)
        pad = b"\0" * ((to - c) * es)
        row = c * es
        for _ in range(r):
            chunk = src.read(row)
            if len(chunk) != row:
                raise ValueError("원본이 짧다")
            out.write(chunk)
            out.write(pad)

    return rows if kind == "rows" else cols


def pad_shard(src_path: Path, dst_path: Path, to: int, report: dict) -> int:
    h, start = st_io.read_header(src_path)
    plan = []
    with open(src_path, "rb") as src:
        for name, meta in st_io.tensors(h):
            new, kind, i = plan_tensor(name, meta, to)
            if kind:
                fn = pad_writer(src, start, meta, kind, to)
                report["padded"].append({"name": name, "from": meta["shape"], "to": new["shape"]})
                report["intermediate_seen"][str(i)] = report["intermediate_seen"].get(str(i), 0) + 1
            else:
                fn = (lambda m: (lambda out: st_io.copy_bytes(src, start, m, out)))(meta)
            plan.append((name, new, fn))
        st_io.write_file(dst_path, plan, h.get("__metadata__"))
    st_io.check_intact(dst_path)
    return sum(st_io.nbytes(m) for _, m, _ in plan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--to", type=int, default=17536)
    a = ap.parse_args(argv)
    src, dst, to = Path(a.src), Path(a.dst), a.to
    if to % 128:
        print("pad: --to 는 128 의 배수", file=sys.stderr)
        return 2
    if dst.exists() and any(dst.iterdir()):
        print(f"pad: 출력 폴더가 비어 있지 않다: {dst}", file=sys.stderr)
        return 2
    dst.mkdir(parents=True, exist_ok=True)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    report = {"src": str(src), "to": to, "padded": [], "intermediate_seen": {}, "shards": len(shards)}
    total = 0
    for s in shards:
        total += pad_shard(src / s, dst / s, to, report)
        print(f"pad: {s} 끝", flush=True)
    cfg = json.loads((src / "config.json").read_text())
    before = {}
    for holder in (cfg, cfg.get("text_config") or {}):
        if "intermediate_size" in holder:
            before[id(holder)] = holder["intermediate_size"]
            holder["intermediate_size"] = to
    if not before:
        print("pad: config 에 intermediate_size 가 없다", file=sys.stderr)
        return 1
    report["config_intermediate_before"] = sorted(set(before.values()))
    (dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    idx.setdefault("metadata", {})["total_size"] = total
    (dst / "model.safetensors.index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2) + "\n")
    for f in src.iterdir():
        if f.is_file() and f.name not in shards and f.name not in ("config.json", "model.safetensors.index.json") \
                and not f.name.startswith(".stewart"):
            shutil.copy2(f, dst / f.name)
    kinds = {}
    for p in report["padded"]:
        k = p["name"].split(".mlp.")[1].split(".")[0] + (" (mtp)" if p["name"].startswith("mtp.") else "")
        kinds[k] = kinds.get(k, 0) + 1
    report["counts"] = kinds
    (dst / "pad_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"to": to, "counts": kinds, "intermediate_seen": report["intermediate_seen"],
                      "config_before": report["config_intermediate_before"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
