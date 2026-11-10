"""0 채우기 바이트 증명: pad_intermediate.py 가 만든 판이 원본과 "MLP 의 늘어난 0 말고는" 바이트로 같은지.

  python pad_check.py --src <원본 BF16 폴더> --dst <0 채운 폴더> --to 17536 --out pad-check.json

- MLP 텐서(model.language_model.layers.N.mlp.{gate,up,down}_proj.weight, mtp.layers.N.mlp.*): gate·up 은 앞 I 행이 원본
  바이트와 같고 늘어난 행은 모두 0, down 은 행마다 앞 I 열이 원본과 같고 늘어난 열은 모두 0. dtype 은 같아야 한다.
- 그 밖 텐서: 이름·dtype·모양·바이트가 모두 같다. 색인의 텐서 집합도 같다.
- 그 밖 파일(토크나이저 등, safetensors·색인·config 말고): 이름과 바이트가 같다.
- config.json: 원본과 견줘 바뀐 칸이 intermediate_size(맨 위·text_config)뿐이고 값은 --to.
- 새 뉴런은 gate=0·up=0 → SiLU(0)·0 = 0, down 열 0 → 더하는 값 0 이다. 그래서 위가 맞으면 출력은 수학적으로 원본과 같다.
  BF16 GPU 전체 KL 은 행렬 모양이 달라 반올림 차례가 바뀌므로 0 이 아니다 — 그 숫자는 kl_check 가 따로 남긴다(문턱 없음).
통과면 0, 어긋남이 하나라도 있으면 1. 결과(어긋난 이름은 앞 20개까지)를 --out 에 쓴다. 텐서는 하나씩 읽고 버린다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_io  # noqa: E402

MLP_RE = re.compile(r"^(model\.language_model\.layers\.\d+|mtp\.layers\.\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
SKIP_FILES = {"config.json", "model.safetensors.index.json", "pad_report.json"}


def index(d: Path) -> dict:
    """텐서 이름 → (파일, 자료 시작, meta)."""
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    heads, out = {}, {}
    for name, shard in wm.items():
        if shard not in heads:
            heads[shard] = st_io.read_header(d / shard)
        h, start = heads[shard]
        if name not in h:
            raise SystemExit(f"{d.name}: 색인의 {name} 가 {shard} 에 없다")
        out[name] = (d / shard, start, h[name])
    return out


def raw(entry) -> bytes:
    path, start, meta = entry
    b, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(start + b)
        return f.read(e - b)


def check_mlp(name: str, s, d, to: int) -> str | None:
    sm, dm = s[2], d[2]
    if sm["dtype"] != dm["dtype"]:
        return f"{name}: dtype {sm['dtype']} → {dm['dtype']}"
    es = st_io.DTYPE_SIZE[sm["dtype"]]
    rs, cs = sm["shape"]
    rd, cd = dm["shape"]
    kind = MLP_RE.match(name).group(2)
    sb, db = raw(s), raw(d)
    if kind in ("gate_proj", "up_proj"):              # [I, H] → [T, H]
        if cd != cs or rd != to or rs > to:
            return f"{name}: 모양 {sm['shape']} → {dm['shape']}(기대 [{to}, {cs}])"
        n = rs * cs * es
        if db[:n] != sb:
            return f"{name}: 앞 {rs} 행이 원본과 다르다"
        if db[n:].count(0) != len(db) - n:
            return f"{name}: 늘어난 행이 0 이 아니다"
        return None
    if rd != rs or cd != to or cs > to:               # down [H, I] → [H, T]
        return f"{name}: 모양 {sm['shape']} → {dm['shape']}(기대 [{rs}, {to}])"
    srow, drow = cs * es, cd * es
    mv_s, mv_d = memoryview(sb), memoryview(db)
    for r in range(rs):
        a = mv_d[r * drow:(r + 1) * drow]
        if a[:srow] != mv_s[r * srow:(r + 1) * srow]:
            return f"{name}: {r} 번 행의 앞 {cs} 열이 원본과 다르다"
        if bytes(a[srow:]).count(0) != drow - srow:
            return f"{name}: {r} 번 행의 늘어난 열이 0 이 아니다"
    return None


def config_diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += config_diff(a.get(k, "<없음>"), b.get(k, "<없음>"), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [path]


def check(src: Path, dst: Path, to: int) -> dict:
    t0 = time.time()
    bad: list[str] = []
    si, di = index(src), index(dst)
    only = sorted(set(si) ^ set(di))
    if only:
        bad.append(f"색인의 텐서 집합이 다르다({len(only)}개, 예: {only[:3]})")
    mlp = same = 0
    for n in sorted(set(si) & set(di)):
        if MLP_RE.match(n):
            mlp += 1
            why = check_mlp(n, si[n], di[n], to)
        else:
            sm, dm = si[n][2], di[n][2]
            why = None
            if (sm["dtype"], sm["shape"]) != (dm["dtype"], dm["shape"]):
                why = f"{n}: dtype·모양 {sm['dtype']}{sm['shape']} → {dm['dtype']}{dm['shape']}"
            elif raw(si[n]) != raw(di[n]):
                why = f"{n}: 바이트가 다르다"
            else:
                same += 1
        if why:
            bad.append(why)
    shards_s = set(json.loads((src / "model.safetensors.index.json").read_text())["weight_map"].values())
    other_s = sorted(p.name for p in src.iterdir() if p.is_file() and p.name not in SKIP_FILES and p.name not in shards_s)
    other_d = {p.name for p in dst.iterdir() if p.is_file()}
    files_same = 0
    for n in other_s:
        if n not in other_d:
            bad.append(f"파일 {n} 이 0 채운 판에 없다")
        elif (src / n).read_bytes() != (dst / n).read_bytes():
            bad.append(f"파일 {n} 의 바이트가 다르다")
        else:
            files_same += 1
    diff = config_diff(json.loads((src / "config.json").read_text()), json.loads((dst / "config.json").read_text()))
    cfg = json.loads((dst / "config.json").read_text())
    allowed = {"intermediate_size", "text_config.intermediate_size"}
    if set(diff) - allowed:
        bad.append(f"config 에서 intermediate_size 말고 바뀐 칸: {sorted(set(diff) - allowed)}")
    for k in diff:
        holder = cfg if k == "intermediate_size" else cfg.get("text_config", {})
        if holder.get("intermediate_size") != to:
            bad.append(f"config {k} 가 {to} 가 아니다")
    if mlp == 0:
        bad.append("MLP 텐서가 하나도 없다(이름 꼴이 다르다)")
    return {"ok": not bad, "to": to, "tensors": len(set(si) & set(di)), "mlp_checked": mlp, "other_tensors_same": same,
            "files_same": files_same, "config_changed": diff, "bad_count": len(bad), "bad": bad[:20],
            "elapsed_s": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--to", type=int, required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = check(Path(a.src), Path(a.dst), a.to)
    body = json.dumps(res, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(body + "\n")
    print(body)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
