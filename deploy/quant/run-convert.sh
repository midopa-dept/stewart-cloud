#!/bin/bash
# 우리 첫 4비트 변환: demo2 BF16 → 0 채우기(17536) → AutoRound W4A16 → MTP 붙이기 → HyperQwen prepare(int8 head·draft).
# 변환 전용 Vast 인스턴스, 이미지 = HyperQwen 8cf642e(모델 A 와 같은 torch 2.13·transformers 5.15.0·compressed-tensors 0.17.0).
# 묶음(quant-files)을 받아 sha 를 대조한 뒤 이 스크립트를 돌린다. 단계마다 표시 파일을 남겨 다시 돌리면 이어서 한다.
# 결과는 HF 가 아니라 보관 서버로 간다. 이 인스턴스는 테일넷(stewart-quant, tag:gpu)에 들어와 :8001 상태·:8090 내보내기를 열고,
# 집이 먼저 연결해 끌어간다(pull). 이 인스턴스 → 집 연결은 없다. 쓰기 토큰·사람 입력도 없다.
set -euo pipefail
# 키(Vast 인스턴스 키·Tailscale 인증 키)는 다른 프로세스를 띄우기 전에 /dev/shm 파일로 옮기고 환경에서 지운다(pod-start 와 같게)
# SHM·APP 는 시험(tests/test_quant.py TRunConvert)만 바꾼다 — 파드에서는 /dev/shm·/app(HyperQwen 이미지) 그대로
SHM=${QUANT_SHM:-/dev/shm} APP=${QUANT_APP:-/app}
# 값이 있을 때만 쓴다 — 셸 비상구의 rerun(같은 컨테이너에서 다시 돎)이 빈 값으로 덮지 않게
umask 077; VKEY=$SHM/vast-key KEYF=$SHM/ts-authkey
[ -n "${CONTAINER_API_KEY:-}" ] && printf '%s' "$CONTAINER_API_KEY" > "$VKEY"; unset CONTAINER_API_KEY
[ -n "${TS_AUTHKEY:-}" ] && printf '%s' "$TS_AUTHKEY" > "$KEYF"; unset TS_AUTHKEY; umask 022
Q=${Q:-/workspace/quant}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$APP/venv/bin/python HF=$APP/venv/bin/hf
# 이어 하기(묶음에 resume.json 이 있을 때): 다른(정지한) 변환 인스턴스의 /workspace/quant 를 콘솔 복사로 $IMQ(기본 /workspace/import)
# 에 받아, 끝난 대상($RESUME_FIRST)의 세트를 10단계 sha 와 대조한 뒤에만 이 인스턴스($Q)로 옮겨 곧바로 내보낸다. 나머지 대상은
# 이 인스턴스에서 이어 한다(복사된 원본이 pins 와 맞으면 그것을 쓴다). 복사는 실행 중인 $Q 의 묶음·테일넷 상태를 덮지 않는다.
RESUME_FIRST="" IMQ=/workspace/import IMPORT_WAIT_MIN=0
if [ -f "$HERE/resume.json" ]; then
  RESUME_FIRST=$($PY -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["first"]))' "$HERE/resume.json")
  IMQ=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("import", "/workspace/import"))' "$HERE/resume.json")
  IMPORT_WAIT_MIN=$($PY -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("wait_min", 180)))' "$HERE/resume.json")
  mkdir -p "$IMQ"
fi
mkdir -p "$Q/done" "$Q/cache"
export HOME="$Q/cache" HF_HOME="$Q/cache/huggingface" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
log() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$Q/convert-log.tsv"; }
stop_instance() {   # 돈 지키기: 정지(디스크와 단계 표시는 남는다. 다시 켜면 이어서 한다)
  if [ -n "${CONTAINER_ID:-}" ] && [ -s "$VKEY" ]; then
    (umask 077; printf 'Authorization: Bearer %s\n' "$(cat "$VKEY")" > "$SHM/vast-hdr")
    code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT -H @"$SHM/vast-hdr" -H "Content-Type: application/json" \
      -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" || true)
    rm -f "$SHM/vast-hdr"
    log "stop api HTTP $code"
  else
    log "stop api 건너뜀(CONTAINER_ID·CONTAINER_API_KEY 없음)"
  fi
}
step() { [ -e "$Q/done/$1" ]; }
mark() { mkdir -p "$(dirname "$Q/done/$1")"; touch "$Q/done/$1"; log "done $1"; }
# ---- 내보내기 함수(시험이 떼어 돈다) ----
SETDIRS="out-bf16heads out-int8heads"
# 복사 판정(v13.1): 대상마다 11-built 표시, 세트 둘의 폴더와 10단계 sha 목록이 있고, 숨은 임시 파일(rsync 가 받는 중)이 없을 때만 ok.
# 목록(readdir)만 읽고 없는 이름을 찾지 않는다 — 실제로 v13 이 복사 중에 없는 done/… 을 이름으로 찾아 둔 사이 호스트 쪽
# rsync(copy_direct)가 done 을 만들자 겹친 파일 시스템이 "없음"을 기억해 done 이 ? 로 보이고 mkdir 가 ESTALE 이었다(짐작 — 목록으로
# 읽은 t 는 멀쩡했다). 첫 이유 한 줄을 찍는다(10분 진행 줄에 붙는다). ESTALE 이면 컨테이너를 다시 켜야 풀린다.
import_check() {
  $PY - "$IMQ" $SETDIRS -- $RESUME_FIRST <<'PYEOF'
import errno, os, sys
a = sys.argv[1:]
root, sets, first = a[0], a[1:a.index("--")], a[a.index("--") + 1:]
j = os.path.join
def ls(*p):
    return set(os.listdir(j(root, *p)))
def why():
    top = ls()
    for need in ("done", "t"):
        if need not in top:
            return f"없음: {need}/"
    dl, tl = ls("done"), ls("t")
    for al in first:
        if al not in dl:
            return f"없음: done/{al}/"
        if "11-built" not in ls("done", al):
            return f"없음: done/{al}/11-built"
        if al not in tl:
            return f"없음: t/{al}/"
        have = ls("t", al)
        for d in sets:
            if d not in have:
                return f"없음: t/{al}/{d}/"
            if d + ".sha256" not in have:
                return f"없음: t/{al}/{d}.sha256"
            if os.path.getsize(j(root, "t", al, d + ".sha256")) == 0:
                return f"빈 목록: t/{al}/{d}.sha256"
            hid = sorted(n for n in ls("t", al, d) if n.startswith("."))
            if hid:
                return f"받는 중: t/{al}/{d}/{hid[0]}"
    return "ok"
try:
    print(why())
except OSError as e:
    if e.errno == errno.ESTALE:
        print(f"ESTALE {e.filename} — 겹친 파일 시스템이 낡은 항목을 쥐었다. 컨테이너를 다시 켜면 풀린다(stop → start, 디스크 남음)")
    else:
        print(f"읽기 오류 {e.filename}: {e.strerror}")
PYEOF
}
import_ready() { [ "$(import_check)" = ok ]; }
# 지문: 폴더들의 파일 이름·크기(1분 사이 바뀌지 않아야 sha 를 센다 — 받는 중인 파일을 세지 않게)
dir_sizes() {
  $PY - "$@" <<'PYEOF'
import os, sys
for p in sys.argv[1:]:
    for n in sorted(os.listdir(p)) if os.path.isdir(p) else []:
        print(p, n, os.path.getsize(os.path.join(p, n)))
PYEOF
}
import_sizes() {
  local a d args=()
  for a in $RESUME_FIRST; do for d in $SETDIRS; do args+=("$IMQ/t/$a/$d"); done; done
  dir_sizes "${args[@]}"
}
# 대조: 세트마다 10단계에서 원래 인스턴스가 남긴 sha 목록과 파일이 모두 같아야 0(빠진·짧은·다른 파일이면 1)
import_verify() {
  local a d
  for a in $RESUME_FIRST; do
    for d in $SETDIRS; do
      ( cd "$IMQ/t/$a/$d" && sha256sum -c --strict --quiet "../$d.sha256" ) || return 1
    done
  done
}
# 옮기기: 대조를 지난 대상의 폴더·단계 표시를 $Q 로(같은 디스크 — mv 는 바로 끝난다). 감사 결과도 있으면
import_move() {
  local a
  mkdir -p "$Q/t" "$Q/done"
  for a in $RESUME_FIRST; do
    if [ -e "$Q/t/$a" ]; then log "이어 하기: $Q/t/$a 가 이미 있다 — 옮기지 않는다"; continue; fi
    mv "$IMQ/t/$a" "$Q/t/$a"
    rm -rf "$Q/done/$a"
    mv "$IMQ/done/$a" "$Q/done/$a"
  done
  if [ -e "$IMQ/done/audit" ] && [ -s "$IMQ/audit-delta.json" ]; then
    mv "$IMQ/audit-delta.json" "$Q/audit-delta.json"
    [ -e "$IMQ/audit-delta.tsv" ] && mv "$IMQ/audit-delta.tsv" "$Q/audit-delta.tsv"
    touch "$Q/done/audit"
  fi
}
# 받지 않은 세트: export.json 의 세트 가운데 집이 받았다고 알리지 않은 것(done/exported-sets.json — quantsrv 가 쓴다)
pending_sets() {
  $PY - "$1" <<'PYEOF'
import json, os, sys
root = sys.argv[1]
try:
    idx = json.load(open(os.path.join(root, "export.json")))
except (OSError, ValueError):
    sys.exit(0)
try:
    got = set(json.load(open(os.path.join(root, "done", "exported-sets.json"))))
except (OSError, ValueError):
    got = set()
print(" ".join(s["alias"] for s in idx.get("sets", []) if s["alias"] not in got))
PYEOF
}
# 받기 대기: 받지 않은 세트가 없으면 끝. 받는 활동($1/export-active, quantsrv 가 보내는 동안 15초마다 갱신)이 5분 안에 있으면
# 기다림을 다시 센다 — 받는 중에는 정지하지 않는다. 활동 없이 EXPORT_WAIT_MIN 분(기본 120)이면 끝.
wait_export() {
  local root=$1 idle=0 w=${EXPORT_WAIT_MIN:-120} age
  log "보관 서버가 끌어가기를 기다린다(:8090 /export, 받지 않은 세트: $(pending_sets "$root")). 받는 활동 없이 ${w}분이면 정지"
  while [ -n "$(pending_sets "$root")" ]; do
    age=999999
    if [ -e "$root/export-active" ]; then age=$(( $(date +%s) - $(stat -c %Y "$root/export-active" 2>/dev/null || echo 0) )); fi
    if [ "$age" -lt 300 ]; then idle=0; else idle=$(( idle + 1 )); fi
    [ "$idle" -ge "$w" ] && break
    sleep 60
  done
  log "받기 대기 끝(받지 않은 세트: $(pending_sets "$root"))"
}
# 보정 자료 고정본(3단계가 받고 export_now 가 보고에 적는다)
PILE_REV=127bfedcd5047750df5ccf3a12979a47bfa0bafa         # NeelNanda/pile-10k
PILE_SHA=a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31
PILE_FILE=data/train-00000-of-00001-4746b8785c874cc7.parquet
# 들어온 양: $IMQ 전체의 "파일 수 바이트"(숨은 임시 파일 포함, 겉보기 크기) — 바뀌는 동안은 복사가 들어오는 중이다
import_total() {
  $PY - "$IMQ" <<'PYEOF'
import os, sys
n = b = 0
for d, _, fs in os.walk(sys.argv[1]):
    for f in fs:
        try:
            b += os.lstat(os.path.join(d, f)).st_size
            n += 1
        except OSError:
            pass
print(n, b)
PYEOF
}
# 이미 옮겼나: 먼저 내보낼 대상이 모두 $Q 에 11-built 로 있다(옮긴 뒤 다시 켠 인스턴스·손으로 옮긴 인스턴스)
import_done_here() {
  local a
  for a in $RESUME_FIRST; do
    [ -e "$Q/done/$a/11-built" ] && [ -d "$Q/t/$a" ] || return 1      # $IMQ 는 이름으로 찾지 않는다(v13.1)
  done
}
# 복사 기다림(v13.1): 들어온 양(목록으로 센 파일 수·바이트)이 IMPORT_CALM_MIN(3)분 그대로일 때만 판정(import_check, 목록만)을
# 본다 — 받는 중에는 받는 폴더 안을 들여다보지 않는다. 판정이 ok 면 대조(import_verify), 대조는 들어온 양이 바뀐 뒤에만 다시 한다.
# 들어온 양이 그대로인 분만 센다(받는 중에는 한도를 세지 않는다). 그대로인 채 IMPORT_WAIT_MIN 분이면 1. 10분마다 진행·판정 한 줄.
import_wait() {
  local cur last="" idle=0 calm=0 n=0 why="" tried=""
  while [ "$idle" -lt "$IMPORT_WAIT_MIN" ]; do
    cur=$(import_total)
    if [ -n "$last" ] && [ "$cur" = "$last" ]; then idle=$(( idle + 1 )) calm=$(( calm + 1 )); else idle=0 calm=0; fi
    last=$cur n=$(( n + 1 ))
    if [ "$calm" -ge "${IMPORT_CALM_MIN:-3}" ] && [ "$tried" != "$cur" ]; then
      why=$(import_check || true)
      if [ "$why" = ok ]; then
        tried=$cur
        if import_verify; then return 0; fi
        why="10단계 sha 와 다르다(받는 중일 수 있다 — 들어온 양이 바뀌면 다시 대조한다)"
        log "이어 하기: $why"
      fi
    fi
    if [ $(( n % 10 )) = 0 ]; then
      log "이어 하기: 들어온 파일 ${cur% *}개 $(( ${cur#* } / 1048576 )) MiB, 그대로인 지 ${idle}분${why:+ — 판정: $why}"
    fi
    sleep 60
  done
  log "이어 하기: 마지막 판정 — ${why:-판정 전(들어온 양이 ${IMPORT_CALM_MIN:-3}분 그대로인 적이 없다)}"
  return 1
}
# 내보내기 갱신: 끝난 대상(done/<별칭>/11-built)으로 export.json 을 다시 만든다. 실패해도 멈추지 않는다(있던 export.json 은 그대로)
export_now() {
  ls "$Q"/done/*/11-built >/dev/null 2>&1 || return 0
  if $PY "$HERE/build_export.py" --q "$Q" --targets "$HERE/targets.json" --pile-rev "$PILE_REV" --draft-ids "$APP/prepare/draft_vocab_ids.json" | tee -a "$Q/convert-log.tsv"; then
    log "내보내기 갱신: 받지 않은 세트 $(pending_sets "$Q")"
  else
    log "내보내기 갱신 실패 — 있던 export.json 그대로"
  fi
}
# ---- 내보내기 함수 끝 ----
# HALT(v13): 이유 한 줄(stdout·convert-log.tsv). 열린 세트(export.json)가 있으면 받기를 기다린 뒤 정지.
# 없으면 — 셸 비상구(테일넷 SSH, 집 → 이 인스턴스)가 열려 있을 때 — 켜진 채 HALT_HOLD_MIN 분(기본 60) 사람을 기다린다:
#   touch $Q/hold     지울 때까지 더 기다린다(HALT 부터 HALT_HOLD_MAX_MIN 분 상한, 기본 360 — A100 1.7 $/hr 로 최대 약 $10)
#   touch $Q/rerun    이 스크립트를 처음부터 다시 돈다(단계 표시로 이어서, 고친 $HERE 파일로). 테일넷·quantsrv 는 그대로 쓴다
#   touch $Q/stop-now 바로 정지
#   손으로 export.json 을 만들면(build_export.py) 받기를 기다린 뒤 정지
# 비상구가 아직 없으면(테일넷 전 HALT) 예전처럼 HALT_WAIT_S(900초) 뒤 정지.
HALT_WAIT_S=900 HALT_HOLD_MIN=${HALT_HOLD_MIN:-60} HALT_HOLD_MAX_MIN=${HALT_HOLD_MAX_MIN:-360} TS_UP=0 TS_NAME=?
hold_door() {
  local i=0
  rm -f "$Q/rerun" "$Q/stop-now"
  log "셸 비상구: 집에서 tailscale ssh root@$TS_NAME 로 들어온다. ${HALT_HOLD_MIN}분 뒤 정지(touch $Q/hold 면 지울 때까지·상한 ${HALT_HOLD_MAX_MIN}분, touch $Q/rerun 이면 처음부터 다시, touch $Q/stop-now 면 바로 정지)"
  while [ "$i" -lt "$HALT_HOLD_MAX_MIN" ]; do
    if [ -e "$Q/stop-now" ]; then rm -f "$Q/stop-now"; log "셸 비상구: stop-now — 정지"; return 0; fi
    if [ -e "$Q/rerun" ]; then rm -f "$Q/rerun"; log "셸 비상구: rerun — 처음부터 다시 돈다"; exec bash "$HERE/run-convert.sh"; fi
    if [ -s "$Q/export.json" ]; then log "셸 비상구: export.json 이 생겼다 — 받기로 넘어간다"; return 0; fi
    if [ "$i" -ge "$HALT_HOLD_MIN" ] && [ ! -e "$Q/hold" ]; then break; fi
    sleep 60
    i=$(( i + 1 ))
  done
  log "셸 비상구: ${i}분 기다림 끝 — 정지"
}
halt() {
  trap - ERR
  log "HALT: $*"
  if [ ! -s "$Q/export.json" ] && [ "$TS_UP" = 1 ]; then hold_door; fi
  if [ -s "$Q/export.json" ]; then
    log "열린 세트가 있다 — 받기를 기다린 뒤 정지"; wait_export "$Q"
  elif [ "$TS_UP" != 1 ]; then
    log "HALT ${HALT_WAIT_S}초 뒤 스스로 정지"; sleep "$HALT_WAIT_S" || true
  fi
  stop_instance
  exec sleep infinity
}
trap 'halt "명령 실패(줄 $LINENO)"' ERR
# 최대 켜짐 감시: 어느 단계가 멈춰 버려도 MAX_HOURS(기본 6) 뒤에는 정지한다. 다시 켜면 표시 파일로 이어서 하고 감시도 새로 돈다
NT=$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["targets"]))' "$(dirname "${BASH_SOURCE[0]}")/targets.json")
MAXH=${MAX_HOURS:-$(( 4 * NT + 2 ))}                      # 대상마다 4시간 + 받기·감사 2시간
( sleep "$(( MAXH * 3600 ))"; log "최대 켜짐 ${MAXH} h — 정지"; stop_instance ) &

# ---- 테일넷(stewart-quant) + 창구 둘: 진행 중에도 래퍼가 :8001/status 를 본다 ----
SERVER_TS_IP=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["server_ip"])' "$HERE/home.json")
[[ $SERVER_TS_IP =~ ^100\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || halt "home.json 의 server_ip 가 테일넷 주소가 아니다"
TS_VER=1.104.5                                                           # pod-start 와 같은 판·sha(11/6 stable)
TS_SHA=720cbbccbfc03d8b55df72afc19e5a72e31b0f65934700aa6fa38074d4ad7bf7
mkdir -p "$Q/bin" "$Q/ts-state"
if [ "$(cat "$Q/bin/.ts-version" 2>/dev/null)" != "$TS_VER" ]; then
  T=tailscale_${TS_VER}_amd64.tgz
  curl -fsSL "https://pkgs.tailscale.com/stable/$T" -o "/tmp/$T"
  got=$(sha256sum "/tmp/$T" | cut -d' ' -f1)
  [ "$got" = "$TS_SHA" ] || halt "tailscale sha256 불일치 got=$got"
  tar -xzf "/tmp/$T" -C /tmp
  cp "/tmp/tailscale_${TS_VER}_amd64/tailscale" "/tmp/tailscale_${TS_VER}_amd64/tailscaled" "$Q/bin/"
  echo "$TS_VER" > "$Q/bin/.ts-version"
fi
TS="$Q/bin/tailscale --socket=/tmp/tailscaled.sock"
QS_PORT=${QUANT_STATUS_PORT:-8001} QE_PORT=${QUANT_EXPORT_PORT:-8090}      # 시험만 바꾼다
# 셸 비상구(v13): tailscale up --ssh — 집(테일넷 ACL 의 ssh 규칙이 허락한 쪽)에서 이 인스턴스로 root 셸.
# 이 인스턴스에서 집으로 여는 연결은 그대로 없다(ACL 에 tag:gpu → 집 허락이 없다). --ssh 가 안 되면 비상구 없이 붙는다(멈추지 않는다)
ts_up() {
  if $TS up "$@" --ssh; then log "tailscale ssh 켬(셸 비상구)"; return 0; fi
  log "tailscale up --ssh 실패 — 셸 비상구 없이 붙는다"
  $TS up "$@"
}
if $TS status >/dev/null 2>&1; then
  log "tailscale 이미 붙어 있다(rerun) — 그대로 쓴다"
else
  # 고정 UDP 포트(v13.1): 템플릿이 -p 70000:70000/udp 를 열면 Vast 가 바깥 포트를 VAST_UDP_PORT_70000(안팎 같은 번호)으로,
  # 공인 IP 를 PUBLIC_IPADDR 로 준다. 그 포트로 tailscaled 를 띄우고 그 끝점을 광고한다(장비는 Docker 뒤 대칭 NAT 라 STUN 이 본
  # 포트가 틀린다 — 실측). 직결이 중계보다 느릴 수 있어(먼 지역 사이에서 22 KB/s 를 본 적이 있다) $Q/ts-relay 가 있으면 쓰지 않는다
  TS_PORT=${VAST_UDP_PORT_70000:-} TS_ADV=""
  if [ -n "$TS_PORT" ] && [ -n "${PUBLIC_IPADDR:-}" ] && [ ! -e "$Q/ts-relay" ]; then TS_ADV="$PUBLIC_IPADDR:$TS_PORT"; fi
  if [ -n "$TS_ADV" ]; then
    TS_DEBUG_PRETENDPOINT="$TS_ADV" "$Q/bin/tailscaled" --tun=userspace-networking --port="$TS_PORT" --statedir="$Q/ts-state" \
      --socket=/tmp/tailscaled.sock --no-logs-no-support >>"$Q/tailscaled.log" 2>&1 &
    log "tailscale 고정 UDP $TS_ADV(템플릿 -p 70000:70000/udp)"
  else
    "$Q/bin/tailscaled" --tun=userspace-networking --statedir="$Q/ts-state" --socket=/tmp/tailscaled.sock \
      --no-logs-no-support >>"$Q/tailscaled.log" 2>&1 &
    log "tailscale UDP 무작위 포트(VAST_UDP_PORT_70000=${TS_PORT:-없음} PUBLIC_IPADDR=${PUBLIC_IPADDR:-없음}$([ -e "$Q/ts-relay" ] && echo ' ts-relay 있음'))"
  fi
  sleep 2
  if [ -s "$KEYF" ]; then
    ts_up --auth-key="file:$KEYF" --hostname=stewart-quant --accept-dns=false --timeout=60s || halt "테일넷에 못 들어갔다"
  else
    ts_up --hostname=stewart-quant --accept-dns=false --timeout=60s || halt "TS_AUTHKEY 없음·상태도 없음"
  fi
  rm -f "$KEYF"
  log "tailscale up stewart-quant"
fi
TS_UP=1
TS_NAME=$($TS status --json 2>/dev/null | $PY -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo '?')
log "tailscale name $TS_NAME"
# 보관 서버와의 길(direct/relay)을 1분 뒤 한 줄(받기 속도를 가늠한다 — relay 면 DERP 중계)
( sleep 60; log "tailscale 길(보관 서버 $SERVER_TS_IP): $($TS status 2>/dev/null | grep -F "$SERVER_TS_IP" | tr -s ' ' | cut -c1-160)" ) &
if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$QS_PORT/status"; then
  log "quantsrv 이미 돈다(rerun) — 그대로 쓴다"
else
  $PY "$HERE/quantsrv.py" --q "$Q" --server-ip "$SERVER_TS_IP" --ts-cmd "$TS" --status-port "$QS_PORT" --export-port "$QE_PORT" \
    >>"$Q/quantsrv.log" 2>&1 &
fi

# 변환 대상은 묶음의 targets.json(서빙 표의 quant_src, order 차례). 대상마다 $Q/t/<별칭>/ 아래에서 단계 표시를 따로 남긴다
QD="$Q/quality-data"
TARGETS=$($PY -c 'import json,sys; [print(t["alias"], t["repo"], t["rev"], t["pins"], t["pad_to"], t["chat_template"]) for t in json.load(open(sys.argv[1]))["targets"]]' "$HERE/targets.json")
AUDIT=$($PY -c 'import json,sys; a = json.load(open(sys.argv[1])).get("audit") or []; print(" ".join(a))' "$HERE/targets.json")
log "targets: $(printf '%s' "$TARGETS" | awk '{print $1}' | tr '\n' ' ')audit: ${AUDIT:-없음}"

# ---- 0. 이미지·묶음 대조 ----
if ! step 0-image; then
  ( cd / && sha256sum -c --strict --quiet "$HERE/image-8cf642e.sha256" ) || halt "HyperQwen 이미지 파일이 8cf642e 와 다르다"
  ( cd "$HERE" && sha256sum -c --strict --quiet SHA256SUMS ) || halt "변환 묶음 sha 불일치"
  log "gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) ram $(free -g | awk '/Mem:/{print $2}')G disk $(df -BG --output=avail "$Q" | tail -1)"
  mark 0-image
fi

# ---- 이어 하기: 복사 대조(대조 뒤에만) → 끝난 대상 옮기기 → 곧바로 내보내기 → 남은 대상은 아래 보통 길로 ----
if [ -n "$RESUME_FIRST" ]; then
  if import_done_here; then
    log "이어 하기: 대상 $RESUME_FIRST 는 이미 $Q 로 옮겨 있다 — 기다리지 않고 내보낸다"
  else
    log "이어 하기: $IMQ 로 복사가 들어오기를 기다린다(먼저 내보낼 대상 $RESUME_FIRST, 들어오는 것 없이 ${IMPORT_WAIT_MIN}분이면 HALT)"
    import_wait || halt "이어 하기: 들어오는 것 없이 ${IMPORT_WAIT_MIN}분 — 복사가 끝나지 않았다(또는 10단계 sha 와 다르다) — $IMQ"
    log "이어 하기: 대상 $RESUME_FIRST 의 세트가 10단계 sha 와 모두 같다 — 옮기고 바로 내보낸다"
    import_move
  fi
  export_now
  df -BG --output=avail "$Q" | tail -1 | sed 's/^/이어 하기: 남은 디스크 /' | tee -a "$Q/convert-log.tsv"
fi

# ---- 1. 도구: auto-round 0.14.2 와 그 의존성만 더한다. 이미지 venv 의 판은 constraints 로 묶어 바뀌지 않게 ----
if ! step 1-tools; then
  "$APP"/venv/bin/pip freeze --exclude-editable | grep -E '^[A-Za-z0-9._-]+==' > "$Q/constraints-image.txt"
  AR_SHA=4265c11d33a6f688c211bfed87e4698b685473efe1d6faf95d744a3f7c19ca24        # auto_round-0.14.2-py3-none-any.whl(PyPI)
  "$APP"/venv/bin/pip download -q --no-deps "auto-round==0.14.2" -d "$Q/wheels"
  echo "$AR_SHA  $Q/wheels/auto_round-0.14.2-py3-none-any.whl" | sha256sum -c --strict --quiet || halt "auto-round 휠 sha 불일치"
  # 이미지에 이미 있는 꾸러미는 이미지 판을 쓴다(요구 목록에서 뺀다). 이미지 판은 constraints 로 묶되, UNPIN 의 꾸러미만 풀어 준다:
  # datasets(조리법 판)는 pyarrow>=21 을 바라는데 이미지는 pyarrow 20.0.0 이다. 두 꾸러미는 변환에서 parquet 읽기에만 쓰인다.
  # 그 밖 이미지 판이 바뀌면 아래 검사가 멈춘다. 판을 푼 결과(바뀐 판)는 pip-changed.txt 의 allowed 줄에 남는다
  UNPIN="pyarrow fsspec"
  $PY - "$Q/constraints-image.txt" "$HERE/requirements-quant.txt" "$Q/req-pinned.txt" "$Q/constraints-solve.txt" "$UNPIN" <<'PYEOF' | tee -a "$Q/convert-log.tsv"
import re, sys
norm = lambda n: n.strip().lower().replace("_", "-")
lines = [l for l in open(sys.argv[1]) if "==" in l]
img = {norm(l.split("==")[0]) for l in lines}
unpin = {norm(x) for x in sys.argv[5].split()}
pinned, used = [], []
for line in open(sys.argv[2]):
    req = line.split("#")[0].strip()
    if not req:
        continue
    name = norm(re.split(r"[=<>~!\[ ;]", req, maxsplit=1)[0])
    if name in img:
        used.append(name)
        continue
    pinned.append(req)
open(sys.argv[3], "w").write("".join(x + "\n" for x in pinned))
open(sys.argv[4], "w").write("".join(l for l in lines if norm(l.split("==")[0]) not in unpin))
print("tools: image keeps " + (",".join(used) or "-") + "; add " + (",".join(pinned) or "-") + "; unpin " + ",".join(sorted(unpin)))
PYEOF
  if ! "$APP"/venv/bin/pip install -q -c "$Q/constraints-solve.txt" -r "$Q/req-pinned.txt" \
       "$Q/wheels/auto_round-0.14.2-py3-none-any.whl" > "$Q/pip-install.log" 2>&1; then
    grep -v '^\[notice\]' "$Q/pip-install.log" | tail -n 30 | tee -a "$Q/convert-log.tsv"
    halt "도구 설치 실패(위 pip 줄 — 조리법 판이 이미지 판과 부딪친다)"
  fi
  "$APP"/venv/bin/pip freeze > "$Q/pip-freeze-after.txt"
  # 이미지에 있던 패키지는 판이 하나도 바뀌지 않아야 한다(더해진 것만 허용)
  $PY - "$Q/constraints-image.txt" "$Q/pip-freeze-after.txt" "$UNPIN" > "$Q/pip-changed.txt" <<'PYEOF' || halt "이미지 패키지 판이 바뀌었다(pip-changed.txt)"
import sys
def rd(p):
    return {l.split("==")[0].strip().lower().replace("_", "-"): l.split("==")[1].strip() for l in open(p) if "==" in l}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
unpin = set(sys.argv[3].split())
bad = {k: (v, b.get(k)) for k, v in a.items() if b.get(k) != v and k not in unpin}
for k, v in sorted(bad.items()):
    print(k, v)
print("allowed:", {k: (v, b.get(k)) for k, v in sorted(a.items()) if k in unpin and b.get(k) != v})
print("added:", sorted(set(b) - set(a)))
sys.exit(1 if bad else 0)
PYEOF
  tail -n 2 "$Q/pip-changed.txt" | tee -a "$Q/convert-log.tsv"
  mark 1-tools
fi

# ---- 3. 보정 자료(pile-10k)·품질 자료: 고정 리비전을 HF 캐시에 받고 refs/main 을 그 커밋으로. 그 뒤로는 오프라인 ----
if ! step 3-data; then
  $HF download NeelNanda/pile-10k "$PILE_FILE" README.md dataset_infos.json --repo-type dataset --revision "$PILE_REV" >/dev/null
  H="$HF_HOME/hub/datasets--NeelNanda--pile-10k"
  echo "$PILE_SHA  $PILE_FILE" | ( cd "$H/snapshots/$PILE_REV" && sha256sum -c --strict --quiet ) || halt "pile-10k sha 불일치"
  mkdir -p "$H/refs" && printf '%s' "$PILE_REV" > "$H/refs/main"
  $HF download Salesforce/wikitext wikitext-2-raw-v1/test-00000-of-00001.parquet --repo-type dataset \
    --revision b08601e04326c79dfdd32d625aee71d232d685c3 --local-dir "$QD/wikitext" >/dev/null
  $HF download openai/gsm8k main/test-00000-of-00001.parquet --repo-type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 --local-dir "$QD/gsm8k" >/dev/null
  $HF download HuggingFaceFW/fineweb-2 data/dan_Latn/test/000_00000.parquet --repo-type dataset \
    --revision af9c13333eb981300149d5ca60a8e9d659b276b9 --local-dir "$QD/fineweb2" >/dev/null
  ( cd "$QD" && grep -v '^#' "$HERE/pins/quality-data.sha256" | sha256sum -c --strict --quiet ) || halt "품질 자료 sha 불일치"
  # datasets 의 오프라인 읽기는 허브 캐시가 아니라 datasets 자체 캐시를 본다. 온라인일 때 고정 리비전으로 한 번 읽어 그 캐시를 만든다
  # (파일은 위에서 sha 를 대조한 허브 캐시의 같은 스냅숏에서 온다)
  out=$($PY -c 'import sys, datasets; d = datasets.load_dataset("NeelNanda/pile-10k", revision=sys.argv[1], split="train"); print("pile-10k cached", len(d))' "$PILE_REV" 2>&1 | tail -n3) || true
  printf '%s\n' "$out" | tee -a "$Q/convert-log.tsv"
  case "$out" in *"pile-10k cached 10000"*) ;; *) halt "pile-10k 를 datasets 캐시에 못 넣었다: ${out##*$'\n'}" ;; esac
  mark 3-data
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if ! step 3b-offline; then   # 긴 일 전에: auto-round 가 부를 load_dataset 이 오프라인으로 고정본을 읽는지
  out=$($PY -c 'import datasets; print("datasets", datasets.__version__); d = datasets.load_dataset("NeelNanda/pile-10k", split="train"); assert len(d) == 10000, len(d); print("pile-10k offline ok", len(d))' 2>&1 | tail -n3) || true
  printf '%s\n' "$out" | tee -a "$Q/convert-log.tsv"
  case "$out" in *"offline ok"*) ;; *) halt "pile-10k 를 오프라인으로 못 읽는다: ${out##*$'\n'}" ;; esac
  mark 3b-offline
fi

# ---- 대상마다 2·4~10 단계 ----
src_of() { echo "$Q/t/$1/src"; }
# 이어 하기: 복사된 원본($IMQ/t/<별칭>/src)이 다 들어와(크기가 1분 그대로) pins 와 맞으면 옮긴다. 60분 안에 안 되면 그냥 둔다(받는다)
import_src() {
  local alias=$1 pins=$2 prev="" cur i
  for i in $(seq 60); do
    cur=$(dir_sizes "$IMQ/t/$alias/src")
    if [ -n "$prev" ] && [ "$cur" = "$prev" ] && [ -z "$(find "$IMQ/t/$alias/src" -maxdepth 1 -name '.*' -type f | head -n1)" ]; then
      if ( cd "$IMQ/t/$alias/src" && grep -v '^#' "$HERE/$pins" | sha256sum -c --strict --quiet ); then
        mkdir -p "$Q/t/$alias"
        rm -rf "$Q/t/$alias/src"
        mv "$IMQ/t/$alias/src" "$Q/t/$alias/src"
        mark "$alias/2-src"
        log "[$alias] 이어 하기: 복사된 원본이 pins 와 같다 — 받지 않고 쓴다"
        return 0
      fi
    fi
    prev=$cur
    sleep 60
  done
  log "[$alias] 이어 하기: 복사된 원본을 쓰지 못했다 — HF 에서 받는다"
}
convert_target() {   # 별칭 저장소 리비전 pins 0채우기(0=안 함) 템플릿
  local alias=$1 repo=$2 rev=$3 pins=$4 pad=$5 tpl=$6 rc
  local T="$Q/t/$1" SRC PAD OUTB OUTI
  SRC="$T/src" OUTB="$T/out-bf16heads" OUTI="$T/out-int8heads"
  if [ "$pad" = 0 ]; then PAD="$SRC"; else PAD="$T/padded"; fi
  mkdir -p "$T"
  step "$alias/11-built" && { log "[$alias] 이미 끝남"; return 0; }
  # 2. 원본 BF16(고정 리비전 + 파일 sha)
  if ! step "$alias/2-src" && [ -n "$RESUME_FIRST" ] && [ -e "$IMQ/done/$alias/2-src" ] && [ -d "$IMQ/t/$alias/src" ]; then
    import_src "$alias" "$pins"                 # 이어 하기: 복사된 원본이 pins 와 맞으면 옮기고 2-src(아니면 아래에서 받는다)
  fi
  [ -n "$RESUME_FIRST" ] && rm -rf "$IMQ/t/$alias"   # 복사된 나머지(옛 0 채운 판 등)는 지운다 — 새 0 채우기 전에 디스크를 비운다
  if ! step "$alias/2-src"; then
    # 원본 받기만 온라인(3b 뒤로는 HF_HUB_OFFLINE=1 — 보정 자료·양자화는 오프라인 그대로). 받은 파일은 바로 pins 로 대조한다
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 $HF download "$repo" --revision "$rev" --local-dir "$SRC" >/dev/null
    ( cd "$SRC" && grep -v '^#' "$HERE/$pins" | sha256sum -c --strict --quiet ) || halt "[$alias] 원본 sha 불일치"
    mark "$alias/2-src"
  fi
  run_audit                                  # 감사 쌍의 원본이 둘 다 있으면 지금(다음 원본을 받기 전 — 원본 둘이 함께 있는 때를 줄인다)
  [ -s "$T/ref.pt" ] || $PY "$HERE/kl_check.py" dump --model "$SRC" --text "$WIKI" --out "$T/ref.pt" | tee -a "$Q/convert-log.tsv"
  # 4·5. 0 채우기(pad_to 가 있을 때만)와 전후 KL
  if [ "$pad" != 0 ] && ! step "$alias/5-kl-pad"; then
    rm -rf "$PAD"
    $PY "$HERE/pad_intermediate.py" --src "$SRC" --dst "$PAD" --to "$pad" | tail -n1 | tee -a "$Q/convert-log.tsv"
    # 문턱은 바이트 증명: MLP 는 앞 I 행·열이 원본 바이트와 같고 늘어난 곳은 모두 0, 그 밖 텐서·파일은 바이트 같음, config 는 intermediate_size 만
    $PY "$HERE/pad_check.py" --src "$SRC" --dst "$PAD" --to "$pad" --out "$T/pad-check.json" | tail -n1 | tee -a "$Q/convert-log.tsv" \
      || halt "[$alias] 0 채우기 바이트 증명 실패(pad-check.json)"
    # BF16 전체 KL 은 숫자만 남긴다(GPU BF16 은 행렬 모양이 다르면 반올림 차례가 달라 0 이 아니다)
    $PY "$HERE/kl_check.py" compare --model "$PAD" --ref "$T/ref.pt" --out "$T/kl-pad.json" | tee -a "$Q/convert-log.tsv" \
      || log "[$alias] KL(0 채우기) 를 못 쟀다 — 바이트 증명은 지났다, 계속"
    mark "$alias/5-kl-pad"
  fi
  [ "$pad" != 0 ] && ! in_audit "$alias" && rm -rf "$SRC"     # 0 채운 판이 있으면 원본은 끝(ref.pt 는 남김)
  # 6. AutoRound W4A16(조리법 고정) + 빠진 MTP·비전을 BF16 으로 + 채팅 템플릿 고정본
  if ! step "$alias/6-quant"; then
    rm -rf "$OUTB"
    $PY "$HERE/quantize_autoround.py" --src "$PAD" --out "$OUTB" --recipe "$T/recipe.json" 2>&1 | tee "$T/quantize.log" | tail -n1
    [ -s "$OUTB/model.safetensors.index.json" ] || halt "[$alias] 양자화 결과가 없다(quantize.log)"
    rc=0; $PY "$HERE/graft_missing.py" --src "$PAD" --dst "$OUTB" --prefix mtp. --prefix model.visual. | tee -a "$Q/convert-log.tsv" || rc=$?
    [ "$rc" = 0 ] || halt "[$alias] MTP 붙이기 뒤에도 빠진 모듈이 있다(rc=$rc)"
    [ -f "$PAD/pad_report.json" ] && cp "$PAD/pad_report.json" "$OUTB/stewart_pad_report.json"
    if [ -n "$tpl" ]; then
      ( cd "$HERE" && grep -F "  $tpl" SHA256SUMS | sha256sum -c --strict --quiet ) || halt "[$alias] 채팅 템플릿 sha 불일치"
      cp "$HERE/$tpl" "$OUTB/chat_template.jinja"
      $PY - "$OUTB/tokenizer_config.json" <<'PYEOF' || halt "[$alias] tokenizer_config.json 손질 실패"
import json, os, sys
p = sys.argv[1]
if os.path.exists(p):
    with open(p, encoding="utf-8") as f:
        j = json.load(f)
    if j.pop("chat_template", None) is not None:
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        os.replace(p + ".tmp", p)
PYEOF
    fi
    mark "$alias/6-quant"
  fi
  # 원본·0 채운 판은 여기서 끝(감사 쌍의 원본은 감사 뒤에 지운다)
  [ "$pad" != 0 ] && rm -rf "$PAD"
  in_audit "$alias" || rm -rf "$SRC"
  # 7. 양자화 전후 KL(문턱 없이 숫자만)
  if ! step "$alias/7-kl-quant"; then
    $PY "$HERE/kl_check.py" compare --model "$OUTB" --ref "$T/ref.pt" --out "$T/kl-quant-bf16heads.json" | tee -a "$Q/convert-log.tsv" \
      || log "[$alias] KL(양자화) 를 못 쟀다 — 결과에 적고 계속"
    mark "$alias/7-kl-quant"
  fi
  # 8. int8 head 판: HyperQwen prepare 4단계(lm_head·임베딩·MTP int8, draft head 40,960)
  if ! step "$alias/8-prepare"; then
    rm -rf "$OUTI" && cp -a "$OUTB" "$OUTI"
    ( cd "$APP" && BASE_MODEL_DIR="$OUTI" HF_REPO="$repo" FAST_VARIANT=0 DFLASH2=0 HARDEN_TEMPLATES=0 TRANSLATE_EFFORT=0 \
        bash docker/prepare.sh ) 2>&1 | tee "$T/prepare.log" | tail -n3
    find "$OUTI" -maxdepth 1 -name '*.bak*' -delete     # v13.1: prepare 백업(.bak-mtp·.bak_embed·.bak-draft…) — 서빙에 안 쓴다, promote 가 거부한다
    mark "$alias/8-prepare"
  fi
  # 9. HyperQwen verify + 서빙(CTX=long SPEC=mtp PREFIX_CACHE=1) + quality_battery(GSM8K — mtp 에서 prompt_logprobs 를 보내지 않는다.
  #    PPL 쪽은 7단계 KL 이 맡는다) + 추측 디코딩 수락 누계
  if ! step "$alias/9-serve-quality"; then
    ( cd "$APP" && MODEL="$OUTI" bash verify.sh --no-server ) > "$T/verify.log" 2>&1 || halt "[$alias] verify FAIL(verify.log)"
    tail -n1 "$T/verify.log" | tee -a "$Q/convert-log.tsv"
    ( cd "$APP" && MODEL="$OUTI" PREPARE=0 CTX=long SPEC=mtp PREFIX_CACHE=1 HOST=127.0.0.1 PORT=8000 \
        exec bash single-user/start_qwen.sh ) > "$T/vllm.log" 2>&1 &
    VPID=$!
    for i in $(seq 360); do curl -sf -o /dev/null http://127.0.0.1:8000/health && break; sleep 10; done
    curl -sf -o /dev/null http://127.0.0.1:8000/health || { kill $VPID || true; halt "[$alias] 서버가 서지 않았다(vllm.log)"; }
    rc=0; VLLM_API=http://127.0.0.1:8000/v1 QUALITY_DATA="$QD" QUALITY_MODEL=qwen3.8-27b \
      $PY "$HERE/quality_battery.py" "convert-$alias" --gsm-only > "$T/quality.log" 2>&1 || rc=$?
    curl -sf http://127.0.0.1:8000/metrics | $PY "$HERE/spec_metrics.py" > "$T/spec.json" || true
    kill $VPID || true; wait $VPID 2>/dev/null || true
    [ "$rc" = 0 ] || halt "[$alias] quality_battery 실패(quality.log)"
    mv "$QD"/result_convert-"$alias"*.json "$T/" 2>/dev/null || true
    mark "$alias/9-serve-quality"
  fi
  # 10. 결과 sha(기록용)
  if ! step "$alias/10-pins"; then
    for d in "$OUTB" "$OUTI"; do
      ( cd "$d" && find . -maxdepth 1 -type f -not -name '.*' -not -name '*.bak*' -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum ) > "$d.sha256"
    done
    mark "$alias/10-pins"
  fi
  mark "$alias/11-built"
}
in_audit() { case " $AUDIT " in *" $1 "*) ! step audit ;; *) return 1 ;; esac; }

# ---- 감사: 원본 → 편집본 BF16 의 층별 ΔW(요약 수치만, CPU). 쌍의 원본이 다 있을 때 한 번 ----
run_audit() {
  [ -n "$AUDIT" ] && ! step audit || return 0
  set -- $AUDIT
  [ -s "$(src_of "$1")/model.safetensors.index.json" ] && [ -s "$(src_of "$2")/model.safetensors.index.json" ] || return 0
  $PY "$HERE/audit_delta.py" --base "$(src_of "$1")" --edited "$(src_of "$2")" --out "$Q/audit-delta.json" \
    --tsv "$Q/audit-delta.tsv" 2>&1 | tail -n3 | tee -a "$Q/convert-log.tsv" || halt "감사 실패(audit-delta)"
  mark audit
  for x in "$1" "$2"; do step "$x/6-quant" && rm -rf "$(src_of "$x")"; done   # 양자화가 끝난 쪽만(남은 쪽은 제 변환 뒤에 지운다)
  return 0
}

WIKI="$QD/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
while read -r alias repo rev pins pad tpl; do
  [ -n "$alias" ] || continue
  step "$alias/failed" && { log "[$alias] 앞서 실패로 표시됨 — 건너뛴다"; continue; }
  log "[$alias] 시작 $repo@${rev:0:12} pad=$pad"
  # 대상마다 가둔다: 이 대상의 HALT·명령 실패는 이 대상만 멈춘다(끝난 대상·열린 세트는 그대로). 하위 셸 안에서 set -e 를 다시 켠다
  # (`( … ) || rc=$?` 꼴이면 bash 가 안쪽의 set -e 를 끈다 — 그래서 set +e 로 부르고 $? 를 본다)
  trap - ERR
  set +e
  ( set -eE; trap 'log "[$alias] 명령 실패(줄 $LINENO) — 위 줄이 이유다"' ERR; halt() { trap - ERR; log "HALT: $*"; exit 3; }
    convert_target "$alias" "$repo" "$rev" "$pins" "$pad" "$tpl"; run_audit )
  rc=$?
  set -e
  trap 'halt "명령 실패(줄 $LINENO)"' ERR
  if [ "$rc" != 0 ]; then
    mark "$alias/failed"
    log "[$alias] 실패(rc=$rc) — 이 대상만 멈춘다. 중간 산출물은 $Q/t/$alias 에 남는다"
  fi
  export_now
done <<< "$TARGETS"

# ---- 11. 내보낼 목록(export.json): 대상마다 세트 둘(bf16h·int8h)과 대상별 보고, 감사 요약 ----
export_now
[ -s "$Q/export.json" ] && ! step 11-export && mark 11-export
trap - ERR
log "변환 끝."
# 집이 POST /export/done 을 보내면 정지. 받는 활동 없이 EXPORT_WAIT_MIN 분(기본 120)이면 정지(디스크는 남아 다시 켜면 이어서 기다린다)
wait_export "$Q"
stop_instance
exec sleep infinity
