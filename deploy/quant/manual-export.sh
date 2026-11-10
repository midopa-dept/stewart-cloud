#!/bin/bash
# 손 내보내기(v13.1): 복사가 들어왔는데 판정이 끝내 안 지날 때(/workspace/import/done 이 ESTALE 로 깨짐) 셸 비상구로
# 들어와 한 줄로 친다:  bash /workspace/quant/files/manual-export.sh
# 하는 일(하나라도 실패하면 그 자리에서 멈춤 — set -e):
#   1. 받은 대상마다 세트 둘을 10단계 sha 목록으로 대조(VERIFY-OK)  — 대조를 빼지 않는다
#   2. $IMQ/t/<대상> → $Q/t/<대상> 으로 옮김(MOVED). $Q 에 이미 있으면 멈춘다
#   3. $Q/done/<대상>/ 단계 표시(2-src … 11-built)를 만든다 — 대조를 지난 세트가 11-built 까지 간 것은 원래 인스턴스 로그로 확인한 일
#   4. 감사 결과가 있으면 함께 옮기고 done/audit
#   5. build_export.py → export.json → EXPORT-OK(앞이 다 지났을 때만)
# 지우는 것 없음. 대상은 묶음의 resume.json "first"(인자로 바꿀 수 있다).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q=${Q:-/workspace/quant} APP=${QUANT_APP:-/app}
PY=$APP/venv/bin/python
IMQ=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("import", "/workspace/import"))' "$HERE/resume.json")
A=${*:-$($PY -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["first"]))' "$HERE/resume.json")}
for a in $A; do
  for d in out-bf16heads out-int8heads; do
    ( cd "$IMQ/t/$a/$d" && sha256sum -c --strict --quiet "../$d.sha256" )
    echo "VERIFY-OK $a $d"
  done
done
mkdir -p "$Q/t" "$Q/done"
for a in $A; do
  test ! -e "$Q/t/$a"
  mv "$IMQ/t/$a" "$Q/t/$a"
  mkdir -p "$Q/done/$a"
  for m in 2-src 6-quant 7-kl-quant 8-prepare 9-serve-quality 10-pins 11-built; do touch "$Q/done/$a/$m"; done
  echo "MOVED $a"
done
if [ -s "$IMQ/audit-delta.json" ]; then
  mv "$IMQ/audit-delta.json" "$Q/audit-delta.json"
  if [ -e "$IMQ/audit-delta.tsv" ]; then mv "$IMQ/audit-delta.tsv" "$Q/audit-delta.tsv"; fi
  touch "$Q/done/audit"
  echo "AUDIT-MOVED"
fi
cd "$HERE"
"$PY" build_export.py --q "$Q" --targets targets.json --pile-rev 127bfedcd5047750df5ccf3a12979a47bfa0bafa \
  --draft-ids "$APP/prepare/draft_vocab_ids.json"
test -s "$Q/export.json"
echo EXPORT-OK
