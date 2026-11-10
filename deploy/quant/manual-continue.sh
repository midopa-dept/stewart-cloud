#!/bin/bash
# 손으로 이어 가기(v13.1): 지금 도는 run-convert.sh 를 멈춰 두고(끄지 않음 — 끄면 부모 boot 가 900초 뒤 장비를 끈다) 같은 컨테이너에서
# 새 판을 띄운다. 셸 비상구로 들어와 한 줄로 친다:  bash /workspace/quant/files/manual-continue.sh
# - 멈출 대상은 pgrep 으로 먼저 뽑고 이 스크립트·그 부모는 뺀다(pkill -f 가 그 글자를 품은 ssh 셸까지 멈춘 일이 있었다).
# - 새 판은 테일넷·quantsrv 가 이미 돌면 그대로 쓰고, 키 파일은 비우지 않는다(run-convert v13). 출력은 컨테이너 로그(PID 1 의 stdout)로.
# - 성공 표지: STOPPED <pid…> → STARTED <pid> → 새 판 로그 끝 몇 줄 → CONTINUE-OK
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q=${Q:-/workspace/quant}
OUT=${QUANT_OUT:-/proc/1/fd/1}
me=$$ parent=$PPID
export QUANT_MANUAL_SELF=$me QUANT_MANUAL_PARENT=$parent       # 시험의 pgrep 대역이 읽는다(뺄 것을 내놓아 빼는지 본다)
stopped=""
for p in $(${QUANT_PGREP:-pgrep} -f 'bash .*run-convert\.sh' || true); do   # QUANT_PGREP 는 시험만(맥 샌드박스는 pgrep 을 막는다)
  if [ "$p" = "$me" ] || [ "$p" = "$parent" ]; then continue; fi
  if kill -STOP "$p" 2>/dev/null; then stopped="$stopped $p"; fi   # 그새 끝난 pid 는 건너뛴다
done
echo "STOPPED${stopped:-(없음)}"
setsid nohup bash "$HERE/run-convert.sh" >> "$OUT" 2>&1 < /dev/null &
echo "STARTED $!"
sleep "${CONTINUE_WAIT_S:-8}"
tail -n 5 "$Q/convert-log.tsv"
echo CONTINUE-OK
