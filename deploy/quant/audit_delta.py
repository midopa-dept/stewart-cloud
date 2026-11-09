"""두 BF16 체크포인트(같은 구조: 원본 → 편집본)의 층별 ΔW 요약(CPU, numpy).

  python audit_delta.py --base <원본 폴더> --edited <편집본 폴더> --out audit-delta.json [--tsv audit-delta.tsv]

출력은 요약 수치 표뿐이다. ΔW·특이 벡터·방향 같은 텐서 값은 어디에도 쓰지 않는다.
- tensors: 전체·같음(바이트)·바뀜·한쪽에만·모양 다름 개수
- changed_layers: 바뀐 텐서가 있는 본체 층 번호(구간으로 줄임), 처음·끝
- towers: 본체(layers)·MTP(mtp.)·비전(visual)·임베딩·lm_head·그 밖 — 텐서 수와 바뀐 수
- by_kind: 행렬 종류(층 번호를 뺀 이름)마다 바뀐 수 / 전체
- rows: 바뀐 텐서마다 층·종류·모양·상대 크기 ‖ΔW‖/‖W‖·최대 절댓값, 2차원이면 특잇값 에너지 비율 e1·e2·e4·e8·e16·e32
  (상위 k 특잇값 제곱합 / ‖ΔW‖², 무작위 저차 SVD 근삿값)과 r90·r99(그 비율을 넘는 가장 작은 k, 32 안에 없으면 null)
바이트가 같은 텐서는 계산하지 않는다. 텐서는 하나씩 읽고 버린다(메모리 = 가장 큰 텐서 float32 의 약 3배).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from pathlib import Path

KS = (1, 2, 4, 8, 16, 32)
SEED = 0
_LAYER = re.compile(r"(?:^|\.)layers\.([0-9]+)\.")


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def index(model_dir: Path) -> dict:
    """텐서 이름 → (파일, 자료 시작, meta)."""
    wm = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out, heads = {}, {}
    for name, shard in wm.items():
        if shard not in heads:
            heads[shard] = read_header(model_dir / shard)
        h, start = heads[shard]
        if name not in h:
            raise SystemExit(f"색인의 텐서 {name} 가 {shard} 에 없다")
        out[name] = (model_dir / shard, start, h[name])
    return out


def raw(entry) -> bytes:
    path, start, meta = entry
    b, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(start + b)
        return f.read(e - b)


def tower(name: str) -> str:
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if "visual" in name or "vision" in name:
        return "visual"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if _LAYER.search(name):
        return "layers"
    return "other"


def kind(name: str) -> str:
    """층 번호를 뺀 이름(행렬 종류)."""
    return _LAYER.sub(".layers.N.", "." + name).lstrip(".")


def layer_of(name: str):
    m = _LAYER.search(name)
    return int(m.group(1)) if m and tower(name) == "layers" else None


def to_f32(b: bytes, meta: dict):
    import numpy as np
    dt = meta["dtype"]
    if dt == "BF16":
        u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
        a = u.view(np.float32)
    elif dt in ("F16", "F32", "F64"):
        a = np.frombuffer(b, dtype={"F16": np.float16, "F32": np.float32, "F64": np.float64}[dt]).astype(np.float32)
    else:
        return None
    return a.reshape(meta["shape"]) if meta["shape"] else a


def top_singular(d, k: int = max(KS), oversample: int = 8, power: int = 2):
    """무작위 저차 SVD 로 상위 k 특잇값(근삿값, 내림차순). 고정 씨앗."""
    import numpy as np
    m, n = d.shape
    r = min(k + oversample, m, n)
    rng = np.random.default_rng(SEED)
    q, _ = np.linalg.qr(d @ rng.standard_normal((n, r)).astype(np.float32))
    for _ in range(power):
        q, _ = np.linalg.qr(d.T @ q)
        q, _ = np.linalg.qr(d @ q)
    s = np.linalg.svd(q.T @ d, compute_uv=False)
    return s[:k]


def matrix_stats(base, edited) -> dict:
    import numpy as np
    d = edited - base
    fro_d = float(np.linalg.norm(d))
    fro_b = float(np.linalg.norm(base))
    out = {"rel": round(fro_d / fro_b, 6) if fro_b else None, "max_abs": float(np.max(np.abs(d))) if d.size else 0.0}
    if d.ndim == 2 and fro_d > 0:
        s = top_singular(d.astype(np.float32))
        e = np.cumsum(s.astype(np.float64) ** 2) / (fro_d ** 2)
        for k in KS:
            out[f"e{k}"] = round(float(e[min(k, len(e)) - 1]), 6)
        for tag, lim in (("r90", 0.90), ("r99", 0.99)):
            hit = [i + 1 for i, v in enumerate(e) if v >= lim]
            out[tag] = hit[0] if hit else None
    return out


def ranges(nums: list[int]) -> str:
    if not nums:
        return ""
    nums, out, a = sorted(nums), [], None
    for i, x in enumerate(nums):
        if a is None:
            a = x
        if i + 1 == len(nums) or nums[i + 1] != x + 1:
            out.append(f"{a}" if a == x else f"{a}-{x}")
            a = None
    return ",".join(out)


def audit(base_dir: Path, edited_dir: Path, stats_fn=matrix_stats, log=lambda m: None) -> dict:
    bi, ei = index(base_dir), index(edited_dir)
    names = sorted(set(bi) | set(ei))
    counts = {"total": len(names), "same": 0, "changed": 0, "only_base": 0, "only_edited": 0, "shape_diff": 0}
    towers: dict = {}
    kinds: dict = {}
    rows, layers = [], set()
    t0 = time.time()
    for i, n in enumerate(names):
        tw = towers.setdefault(tower(n), {"tensors": 0, "changed": 0})
        tw["tensors"] += 1
        kd = kinds.setdefault(kind(n), {"changed": 0, "of": 0})
        kd["of"] += 1
        if n not in bi or n not in ei:
            counts["only_base" if n in bi else "only_edited"] += 1
            tw["changed"] += 1
            continue
        mb, me = bi[n][2], ei[n][2]
        if mb["shape"] != me["shape"] or mb["dtype"] != me["dtype"]:
            counts["shape_diff"] += 1
            tw["changed"] += 1
            continue
        rb, re_ = raw(bi[n]), raw(ei[n])
        if rb == re_:
            counts["same"] += 1
            continue
        counts["changed"] += 1
        tw["changed"] += 1
        kd["changed"] += 1
        if layer_of(n) is not None:
            layers.add(layer_of(n))
        row = {"name": n, "layer": layer_of(n), "tower": tower(n), "kind": kind(n), "shape": mb["shape"], "dtype": mb["dtype"]}
        if stats_fn is not None:                 # None 이면 바이트 비교·분류만(수치 없이)
            b, e = to_f32(rb, mb), to_f32(re_, me)
            del rb, re_
            if b is not None:
                row |= stats_fn(b, e)
        rows.append(row)
        if i % 50 == 0:
            log(f"{i + 1}/{len(names)} 텐서, 바뀜 {counts['changed']}, {time.time() - t0:.0f}s")
    return {"version": 1, "base": base_dir.name, "edited": edited_dir.name, "tensors": counts,
            "changed_layers": {"ranges": ranges(sorted(layers)), "first": min(layers) if layers else None,
                               "last": max(layers) if layers else None, "count": len(layers)},
            "towers": towers, "by_kind": {k: v for k, v in sorted(kinds.items()) if v["changed"]},
            "energy_k": list(KS), "method": f"무작위 저차 SVD(상위 {max(KS)}, 덧붙임 8, 거듭 2, 씨앗 {SEED})", "rows": rows}


TSV_COLS = ("layer", "tower", "kind", "shape", "rel", "max_abs") + tuple(f"e{k}" for k in KS) + ("r90", "r99")


def write_tsv(res: dict, path: Path) -> None:
    lines = ["\t".join(TSV_COLS)]
    for r in res["rows"]:
        lines.append("\t".join("x".join(map(str, r[c])) if c == "shape" else ("" if r.get(c) is None else str(r.get(c)))
                               for c in TSV_COLS))
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--edited", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsv")
    a = ap.parse_args(argv)
    res = audit(Path(a.base), Path(a.edited), log=lambda m: print(m, file=sys.stderr, flush=True))
    Path(a.out + ".tmp").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    Path(a.out + ".tmp").replace(a.out)
    if a.tsv:
        write_tsv(res, Path(a.tsv))
    c, cl = res["tensors"], res["changed_layers"]
    print(json.dumps({"changed": c["changed"], "same": c["same"], "layers": cl["ranges"],
                      "mtp_changed": res["towers"].get("mtp", {}).get("changed", 0),
                      "visual_changed": res["towers"].get("visual", {}).get("changed", 0),
                      "sha": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:16]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
