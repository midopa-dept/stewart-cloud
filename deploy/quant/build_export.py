"""내보낼 목록과 보고(export.json)를 만든다. run-convert.sh 11단계.
  python build_export.py --q /workspace/quant --targets targets.json --pile-rev … [--draft-ids /app/prepare/draft_vocab_ids.json]
- 대상(targets.json 의 별칭)마다 세트 둘(별칭 = 보관 서버 catalog 이름):
  - <별칭>-bf16h: t/<별칭>/out-bf16heads(BF16 head, 공식 vLLM 그대로)
  - <별칭>-int8h: t/<별칭>/out-int8heads(HyperQwen prepare int8 head + draft head, 첫 서빙 판)
  끝난 대상(done/<별칭>/11-built)만 싣는다. 세트마다 그 대상의 report 를 단다(맨 위 report 는 첫 대상 — 옛 꼴 읽는 쪽).
- 파일: 폴더 바로 아래, 숨은 것·.bak 빼고. 이름은 한 조각(폴더 없음)이어야 한다.
- 집은 pickle 을 받지 않는다(promote 가 .pt 를 거부).
  - int8h 의 mtp_draft_vocab_ids.pt 는 torch.load(weights_only=True)로 읽어 정수 목록 mtp_draft_vocab_ids.json 으로 바꿔 내보낸다. .pt 는 목록에서 뺀다.
  - 그 목록이 HyperQwen 의 draft_vocab_ids.json 과 같은지 본다. 파드가 JSON 에서 .pt 를 다시 만든다(serving.sh).
- report: pad_to·0 채우기 KL·양자화 KL·verify·quality·추측 디코딩 수락·조리법·0 채우기 보고·바뀐 패키지(promote 의 문턱이 읽는다)
- audit: 감사 요약(audit-delta.json, 수치 표만)이 있으면 맨 위에 싣는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SET_DIRS = (("bf16h", "out-bf16heads"), ("int8h", "out-int8heads"))
DRAFT_PT, DRAFT_JSON = "mtp_draft_vocab_ids.pt", "mtp_draft_vocab_ids.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def files_of(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.iterdir(), key=lambda p: p.name.encode()):
        if p.name.startswith(".") or ".bak" in p.name or p.name == DRAFT_PT:   # v13.1: prepare 의 .bak-mtp·.bak_embed 꼴도
            continue
        if p.is_dir():
            raise SystemExit(f"하위 폴더는 내보내지 않는다: {p}")
        if not NAME_RE.match(p.name):
            raise SystemExit(f"이름 꼴이 아니다: {p.name}")
        out.append({"name": p.name, "sha256": sha(p), "size": p.stat().st_size})
    return out


def draft_json(d: Path, shipped: Path | None) -> int | None:
    """.pt → .json(정수 목록). 만든 개수. .pt 가 없으면 None."""
    pt = d / DRAFT_PT
    if not pt.exists():
        return None
    import torch
    t = torch.load(pt, map_location="cpu", weights_only=True)
    ids = [int(x) for x in t.tolist()]
    if shipped is not None and shipped.exists():
        ref = json.loads(shipped.read_text())
        if ids != ref:
            raise SystemExit("draft 어휘가 HyperQwen draft_vocab_ids.json 과 다르다")
    body = json.dumps(ids)
    out = d / DRAFT_JSON
    if not out.exists() or out.read_text() != body:   # 다시 만들 때(대상별 내보내기) 받는 중인 파일을 덮지 않게: 다를 때만, 임시 뒤 이름 바꾸기
        (d / (".tmp-" + DRAFT_JSON)).write_text(body)
        (d / (".tmp-" + DRAFT_JSON)).replace(out)
    return len(ids)


def _j(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def report(q: Path, t: Path, target: dict, pile_rev: str) -> dict:
    """대상 하나의 보고. q = 작업 뿌리(공통: pip-changed), t = 대상 폴더(t/<별칭>)."""
    vlog = (t / "verify.log").read_text(errors="replace").splitlines() if (t / "verify.log").exists() else []
    qres = sorted(t.glob("result_*.json"))
    qlog = (t / "quality.log").read_text(errors="replace").splitlines() if (t / "quality.log").exists() else []
    pip_changed = (q / "pip-changed.txt").read_text().splitlines() if (q / "pip-changed.txt").exists() else []
    ok = (q / "done" / target["alias"] / "9-serve-quality").exists()
    return {"src": {"repo": target["repo"], "rev": target["rev"]}, "calibration": {"dataset": "NeelNanda/pile-10k", "rev": pile_rev},
            "pad_to": target["pad_to"], "recipe": _j(t / "recipe.json"),
            "pad": _j(t / "out-bf16heads" / "stewart_pad_report.json") if target["pad_to"] else None,
            "kl_pad": _j(t / "kl-pad.json") if target["pad_to"] else None,
            "pad_check": _j(t / "pad-check.json") if target["pad_to"] else None,
            "kl_quant_bf16heads": _j(t / "kl-quant-bf16heads.json"),
            "verify": {"ok": ok, "last": vlog[-1][:300] if vlog else None},
            "quality": {"ok": ok, "result": _j(qres[-1]) if qres else None, "tail": [ln[:300] for ln in qlog[-12:]]},
            "spec_decode": (_j(t / "spec.json") or {}).get("spec_decode"),
            "pip_added": [ln for ln in pip_changed if ln.startswith("added:")]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--q", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--pile-rev", required=True)
    ap.add_argument("--draft-ids", default="/app/prepare/draft_vocab_ids.json")
    a = ap.parse_args(argv)
    q = Path(a.q)
    targets = json.loads(Path(a.targets).read_text())["targets"]
    sets, first = [], None
    for tg in targets:
        if not NAME_RE.match(tg["alias"]) or not (q / "done" / tg["alias"] / "11-built").exists():
            continue
        t = q / "t" / tg["alias"]
        rep = report(q, t, tg, a.pile_rev)
        first = first or rep
        for suffix, d in SET_DIRS:
            n = draft_json(t / d, Path(a.draft_ids) if a.draft_ids else None)
            sets.append({"alias": f"{tg['alias']}-{suffix}", "dir": f"t/{tg['alias']}/{d}", "draft_ids": n,
                         "files": files_of(t / d), "report": rep})
    if not sets:
        raise SystemExit("끝난 대상이 없다(done/<별칭>/11-built)")
    out = {"version": 1, "sets": sets, "report": first, "audit": _j(q / "audit-delta.json")}
    body = json.dumps(out, ensure_ascii=False, indent=1)
    if len(body.encode()) > 900_000:
        raise SystemExit("export.json 이 너무 크다(1 MB 상한 아래로)")
    (q / "export.json.tmp").write_text(body + "\n")
    (q / "export.json.tmp").replace(q / "export.json")
    print(json.dumps({s["alias"]: len(s["files"]) for s in sets}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
