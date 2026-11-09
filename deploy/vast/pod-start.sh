#!/bin/bash
# Vast.ai 인스턴스 시작 스크립트(stewart 클라우드 시범). HyperQwen 이미지(sha-8cf642e)의 진입 스크립트는 모르는 첫 인자를
# 그대로 실행하므로(exec "$cmd" "$@"), 템플릿(Entrypoint 모드)의 짧은 Args 가 gist($PU)에서 이 묶음을 받아 sha256($PS)을
# 대조한 뒤 이 스크립트를 돌리고, 끝에 vLLM으로 넘긴다(make_template_env.py 참고).
# 볼륨 없이 컨테이너 디스크(serve 60GB·eval 80GB)에 받는다. destroy 하면 다음에 다시 받는다.
#   1) 원격 측정 끄기 1a) 역할(serve|eval, 래퍼가 만들 때 넣는 env)·차선 계획 2) Tailscale 사용자 공간 모드(이름은 역할 표에서:
#   stewart-gpu·stewart-gpu-eval, 태그는 인증 키에 붙어 있음) 3) 자동 종료 감시기·차선 상태
#   4) 가중치·품질 자료를 HF 커밋과 sha256 으로 고정해 받기(모델마다 한 번, 차선이 함께 씀)
#   5) 오프라인으로 vLLM: serve = 127.0.0.1:8000 하나(v6 그대로), eval = 차선 k 마다 127.0.0.1:80k0(카드는 계획 표) + 품질 점검
# 환경변수: TS_AUTHKEY(Vast 계정 환경 변수), PU·PS·CF·CTX·SPEC(템플릿 Docker Options) (선택: PREFIX_CACHE, FAST_VARIANT, QUALITY, MODEL, ADAPTER, PLAN)
#   STEWART_ROLE·STEWART_GPUS: stewart-cloud up 이 만들 때(PUT asks env) 넣는다. 템플릿 env 와 키별로 합쳐진다(Vast 문서).
# 모델·어댑터는 serving/models.toml 에서 고른다(MODEL=<별칭>, ADAPTER=auto|hyperqwen-8cf642e|vllm-patched|vllm-stock).
#   주소·리비전·sha 는 묶음 안 표에서만 온다. 이미지는 HyperQwen 8cf642e 또는 공식 vllm/vllm-openai v0.30.0(serving_detect 가 가린다).
# CF: 설정 gist(최신 raw 주소). 허용 목록의 KEY=VALUE 만 읽어 Docker Options 의 기본값을 덮는다(코드·주소·키는 못 바꾼다).
# Vast 가 넣는 것: CONTAINER_ID(인스턴스 번호), CONTAINER_API_KEY(그 인스턴스만 시작·정지·삭제할 수 있는 키)
# vLLM 키는 두지 않는다(경계는 테일넷 ACL).
set -euo pipefail

# ---- 0. 인증 키: 다른 프로세스를 띄우기 전에 /dev/shm 의 600 파일로 옮기고 환경에서 지운다 ----
# (컨테이너 PID 1 의 시작 환경에 남는 것은 막을 수 없다 — 알려진 위험)
umask 077
KEYF=/dev/shm/ts-authkey
printf '%s' "${TS_AUTHKEY:-}" > "$KEYF"
unset TS_AUTHKEY
VKEY=/dev/shm/vast-key
printf '%s' "${CONTAINER_API_KEY:-}" > "$VKEY"
unset CONTAINER_API_KEY
umask 022

# ---- 1. 원격 측정·업데이트 확인 끄기(제3자 이미지·가중치, 비공개 코드가 흐름) ----
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 TS_NO_LOGS_NO_SUPPORT=true
unset VLLM_API_KEY OPENAI_API_KEY

S=/workspace/stewart                 # 컨테이너 디스크(볼륨 없음)
mkdir -p "$S/ts-state" "$S/bin" "$S/models" "$S/cache"
log() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$S/pod-log.tsv"; }
# 치명적 오류: 그냥 끝내면 컨테이너가 재시작을 되풀이하며 과금될 수 있다 → 이유 한 줄(stdout·pod-log), 키 파일 지움,
# HALT_WAIT_S(900초) 기다린 뒤 스스로 정지(래퍼 logs 로 이유를 읽을 틈 — T23), 대기. destroy 는 사람·래퍼.
HALT_WAIT_S=900
halt() {
  rm -f "$KEYF"
  log "HALT: $*"
  log "HALT ${HALT_WAIT_S}초 뒤 스스로 정지(stewart-cloud logs <id> 로 이유를 본다)"
  sleep "$HALT_WAIT_S" || true
  if [ -n "${CONTAINER_ID:-}" ] && [ -s "$VKEY" ]; then
    (umask 077; printf 'Authorization: Bearer %s\n' "$(cat "$VKEY")" > /dev/shm/vast-hdr)   # 키가 명령 줄(ps)에 보이지 않게
    code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT -H @/dev/shm/vast-hdr \
      -H "Content-Type: application/json" -d '{"state":"stopped"}' \
      "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" || true)
    log "HALT stop api HTTP $code"
  fi
  exec sleep infinity
}
trap 'halt "명령 실패(줄 $LINENO)"' ERR
[ -s "$KEYF" ] || halt "TS_AUTHKEY 없음(템플릿 Secret 확인)"
log "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
# 전력·클럭(호스트 전력 상한 확인용). 새 드라이버는 clocks_event_reasons, 옛 드라이버는 clocks_throttle_reasons
PQ=power.limit,power.default_limit,power.max_limit,power.draw,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem
pw=$(nvidia-smi --query-gpu=$PQ,clocks_event_reasons.active --format=csv,noheader 2>/dev/null) \
  || pw=$(nvidia-smi --query-gpu=$PQ,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null) || pw="?"
log "power limit,default,max,draw,sm,mem,max.sm,max.mem,reasons=${pw%%$'\n'*}"
log "start instance=${CONTAINER_ID:-?} label=${VAST_CONTAINERLABEL:-?} vastkey=$([ -s "$VKEY" ] && echo yes || echo no) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
# 이미지 종류·파이썬·받기 도구(serving_detect), 이미지 대조 한 줄(vllm 판·HyperQwen series sha·nvcc)
. "$S/serving/serving.sh"
serving_detect

# ---- 1a. 역할(T20): 래퍼가 만들 때 넘긴 STEWART_ROLE·STEWART_GPUS. 테일넷 이름·차선 수는 이 표에서만 고른다 ----
# 설정 파일(CF) 허용 목록에는 넣지 않는다(설정 파일로 이름을 못 바꾼다). 없거나 모르면 HALT — 이름을 잘못 잡지 않게(결정 5).
ROLE=${STEWART_ROLE:-}
WANT_GPUS=${STEWART_GPUS:-}
REQ_PLAN=${STEWART_PLAN:-}           # 래퍼 up 의 계획 인자(T23): CF 의 PLAN 보다 먼저. serve 는 무시
unset STEWART_ROLE STEWART_GPUS STEWART_PLAN
[[ -z $REQ_PLAN || $REQ_PLAN =~ ^[a-z0-9][a-z0-9.-]{0,47}$ ]] || halt "STEWART_PLAN 꼴이 다르다"
ROLE_NAME=$ROLE                      # 래퍼 역할 그대로(기록용). eval2 는 이름만 다르고 그 밖은 "eval"(T26)
case "$ROLE" in
  serve) TS_HOST=stewart-gpu ;;
  eval)  TS_HOST=stewart-gpu-eval ;;
  eval2) TS_HOST=stewart-gpu-eval2; ROLE=eval ;;   # 4장 둘째 인스턴스(T26). 아래는 ROLE=eval 로 같은 차선·감시기
  *) halt "STEWART_ROLE 없음·모름(serve|eval|eval2) — stewart-cloud up 이 만들 때 넣는다" ;;
esac
case "$WANT_GPUS" in [1-8]) ;; *) halt "STEWART_GPUS 없음·형식 다름(1~8)" ;; esac
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
[ "$NGPU" = "$WANT_GPUS" ] || halt "GPU 수 다름: 보임 $NGPU, 래퍼 $WANT_GPUS"
if [ "$ROLE" = serve ]; then
  [ "$NGPU" = 1 ] || halt "serve 는 1장"
fi
log "role=$ROLE ts=$TS_HOST gpus=$NGPU$([ "$ROLE_NAME" = "$ROLE" ] || echo " as=$ROLE_NAME")"

# ---- 1b. 설정 파일(CF): 허용 목록의 KEY=VALUE 만. 받기 실패·없음이면 Docker Options 기본값 ----
CTX=${CTX:-fast} SPEC=${SPEC:-mtp} PREFIX_CACHE=${PREFIX_CACHE:-0} FAST_VARIANT=${FAST_VARIANT:-1} QUALITY=${QUALITY:-auto}
MODEL_ALIAS=${MODEL:-default} ADAPTER=${ADAPTER:-auto} PLAN=${PLAN:-}   # PLAN: 평가 역할의 차선 계획(pod v7 이 쓴다)
unset MODEL                          # HyperQwen 의 MODEL(폴더 경로)과 겹치지 않게. 폴더는 serving.py 가 정한다
CFG_SRC=default CFG_IGN=""
if [ -n "${CF:-}" ] && cfg=$(curl -fsS --max-time 20 --retry 2 "$CF?t=$(date +%s)" | head -c 4096); then
  CFG_SRC=cf
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line//[[:space:]]/}
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in
      CTX=fast|CTX=long) CTX=${line#*=} ;;
      SPEC=mtp) SPEC=${line#*=} ;;
      PREFIX_CACHE=0|PREFIX_CACHE=1) PREFIX_CACHE=${line#*=} ;;
      FAST_VARIANT=0|FAST_VARIANT=1) FAST_VARIANT=${line#*=} ;;
      QUALITY=auto|QUALITY=0|QUALITY=1) QUALITY=${line#*=} ;;
      ADAPTER=auto|ADAPTER=hyperqwen-8cf642e|ADAPTER=vllm-patched|ADAPTER=vllm-stock) ADAPTER=${line#*=} ;;
      MODEL=*) v=${line#*=}; if [[ $v =~ ^[a-z0-9][a-z0-9.-]{0,47}$ ]]; then MODEL_ALIAS=$v; else CFG_IGN="$CFG_IGN MODEL"; fi ;;
      PLAN=*) v=${line#*=}; if [[ $v =~ ^[a-z0-9][a-z0-9.-]{0,47}$ ]]; then PLAN=$v; else CFG_IGN="$CFG_IGN PLAN"; fi ;;
      *) k=${line%%=*}; k=${k//[^A-Za-z0-9_]/}; CFG_IGN="$CFG_IGN ${k:0:24}" ;;   # 값은 남기지 않는다
    esac
  done <<< "$cfg"
elif [ -n "${CF:-}" ]; then
  CFG_SRC=default-cf-failed
fi
case "$CTX" in fast|long) ;; *) CTX=fast ;; esac       # Docker Options 값도 같은 허용 목록으로
case "$SPEC" in mtp) ;; *) SPEC=mtp ;; esac
case "$PREFIX_CACHE" in 0|1) ;; *) PREFIX_CACHE=0 ;; esac
case "$FAST_VARIANT" in 0|1) ;; *) FAST_VARIANT=1 ;; esac
case "$QUALITY" in auto|0|1) ;; *) QUALITY=auto ;; esac
case "$ADAPTER" in auto|hyperqwen-8cf642e|vllm-patched|vllm-stock) ;; *) ADAPTER=auto ;; esac
[[ $MODEL_ALIAS =~ ^[a-z0-9][a-z0-9.-]{0,47}$ ]] || MODEL_ALIAS=default
[[ -z $PLAN || $PLAN =~ ^[a-z0-9][a-z0-9.-]{0,47}$ ]] || PLAN=""
# 차선 계획은 eval 역할만 읽는다(T21b): serve 는 PLAN 을 버린다 → CF 를 같이 읽어도 serve 에는 계획·계획 조건이 가지 않는다.
# 만들 때 넘긴 STEWART_PLAN(T23)이 있으면 CF 의 PLAN 보다 먼저다(표에 없는 이름이면 1c 의 계획 단계가 HALT).
PLAN_SRC=cf
if [ "${ROLE:-}" = "eval" ] && [ -n "${REQ_PLAN:-}" ]; then PLAN=$REQ_PLAN; PLAN_SRC=up; fi
if [ "${ROLE:-}" != "eval" ] && [ -n "$PLAN" ]; then CFG_IGN="$CFG_IGN PLAN"; PLAN=""; fi
if [ "${ROLE:-}" != "eval" ] && [ -n "${REQ_PLAN:-}" ]; then CFG_IGN="$CFG_IGN STEWART_PLAN"; fi
COND_SRC=cf                          # 비교 조건(CTX·SPEC·PREFIX_CACHE·FAST_VARIANT)을 어디서 받았나: cf | plan(1c 에서 덮음)
write_run_config() {
  export CTX SPEC PREFIX_CACHE FAST_VARIANT
  SIG="CTX=$CTX SPEC=$SPEC PREFIX_CACHE=$PREFIX_CACHE FAST_VARIANT=$FAST_VARIANT MODEL=$MODEL_ALIAS ADAPTER=$ADAPTER PLAN=${PLAN:--}"
  printf '{"src":"%s","CTX":"%s","SPEC":"%s","PREFIX_CACHE":"%s","FAST_VARIANT":"%s","QUALITY":"%s","MODEL":"%s","ADAPTER":"%s","PLAN":"%s","plan_src":"%s","cond":"%s","ignored":"%s"}\n' \
    "$CFG_SRC" "$CTX" "$SPEC" "$PREFIX_CACHE" "$FAST_VARIANT" "$QUALITY" "$MODEL_ALIAS" "$ADAPTER" "$PLAN" "$PLAN_SRC" "$COND_SRC" "${CFG_IGN# }" > "$S/run-config.json"
  log "config src=$CFG_SRC $SIG QUALITY=$QUALITY cond=$COND_SRC ignored=[${CFG_IGN# }]"
}
write_run_config

# ---- 1c. 차선 계획(eval): 테일넷·받기 전에 정한다(감시기가 차선 수·차선 env 를 읽고, 계획이 틀리면 이름을 잡기 전에 HALT) ----
# $S/lanes/plan.txt 줄 = "k 포트 카드 별칭 어댑터 quality", 차선 env = $S/lanes/lane-k.env(serving.py 가 표에서만 쓴다).
#   PLAN 있음: serving.py plan(카드 합 = 계획 gpus = 인스턴스 장 수(T25, 남는 카드도 HALT), fill 계획은 남는 카드를 채움,
#   hyperqwen 어댑터는 차선 0 이 같은 모델의 hyperqwen 일 때만 — 표 검사). PLAN 없음: 카드마다 같은 MODEL·ADAPTER(T20 그대로).
# serve 는 차선 하나(model.env, 4절의 serving_select 가 쓴다).
LANES=1
if [ "$ROLE" = "eval" ]; then
  rm -rf "$S/lanes"; mkdir -p "$S/lanes"
  if [ -n "$PLAN" ]; then
    out=$($PY "$SV/serving.py" plan "$PLAN" "$IMAGE_KIND" "$NGPU" "$S/lanes" 2>&1) || halt "차선 계획 실패: ${out:0:600}"
    printf '%s\n' "$out" > "$S/lanes/plan.txt"
    # 계획의 비교 조건으로 CF 값을 덮는다(같은 허용 목록으로 다시 본다). 품질 점검 SIG·run-config 도 이 값으로.
    while IFS= read -r line || [ -n "$line" ]; do
      kv=${line//\'/}
      case "$kv" in
        '') ;;
        CTX=fast|CTX=long|SPEC=mtp|PREFIX_CACHE=0|PREFIX_CACHE=1|FAST_VARIANT=0|FAST_VARIANT=1) printf -v "${kv%%=*}" '%s' "${kv#*=}" ;;
        *) halt "계획 조건을 모른다: ${kv:0:40}" ;;
      esac
    done < "$S/lanes/plan.env"
    COND_SRC=plan
    write_run_config
  else
    out=$($PY "$SV/serving.py" select "$MODEL_ALIAS" "$ADAPTER" "$IMAGE_KIND" "$S/lanes/lane-0.env" 2>&1) || halt "모델 고르기 실패: $out"
    ( . "$S/lanes/lane-0.env"
      for k in $(seq 0 $((NGPU - 1))); do
        [ "$k" = 0 ] || cp "$S/lanes/lane-0.env" "$S/lanes/lane-$k.env"
        echo "$k $((8000 + 10 * k)) $k $M_ALIAS $M_ADAPTER $([ "$k" = 0 ] && echo 1 || echo 0)"
      done ) > "$S/lanes/plan.txt"
  fi
  LANES=$(grep -c . "$S/lanes/plan.txt" || true)
  case "$LANES" in [1-8]) ;; *) halt "차선 수가 이상하다: $LANES" ;; esac
  log "lanes plan=${PLAN:--} n=$LANES: $(tr '\n' ';' < "$S/lanes/plan.txt")"
fi

# ---- 2. Tailscale: 버전 고정 + sha256 대조(pkgs 의 .sha256 과 여기 고정값 둘 다). 볼륨에 캐시 ----
TS_VER=1.104.5                                                           # 11/6 stable
TS_SHA=720cbbccbfc03d8b55df72afc19e5a72e31b0f65934700aa6fa38074d4ad7bf7  # tailscale_1.104.5_amd64.tgz.sha256 (11/6)
if [ "$(cat "$S/bin/.ts-version" 2>/dev/null)" != "$TS_VER" ]; then
  T=tailscale_${TS_VER}_amd64.tgz
  curl -fsSL "https://pkgs.tailscale.com/stable/$T" -o "/tmp/$T"
  pub=$(curl -fsSL "https://pkgs.tailscale.com/stable/$T.sha256" | tr -d '[:space:]')
  got=$(sha256sum "/tmp/$T" | cut -d' ' -f1)
  [ "$got" = "$pub" ] && [ "$got" = "$TS_SHA" ] || halt "tailscale sha256 불일치 got=$got pub=$pub pinned=$TS_SHA"
  tar -xzf "/tmp/$T" -C /tmp
  cp "/tmp/tailscale_${TS_VER}_amd64/tailscale" "/tmp/tailscale_${TS_VER}_amd64/tailscaled" "$S/bin/"
  echo "$TS_VER" > "$S/bin/.ts-version"
  log "tailscale $TS_VER installed sha256=$got"
fi
TS="$S/bin/tailscale --socket=/tmp/tailscaled.sock"
"$S/bin/tailscaled" --tun=userspace-networking --statedir="$S/ts-state" --socket=/tmp/tailscaled.sock \
  --no-logs-no-support >>"$S/tailscaled.log" 2>&1 &
sleep 2
# 상태가 볼륨에 남아 있으면 같은 노드로 다시 붙는다(이름 유지). 없으면 키로 새로 붙는다.
# --accept-dns=false: 테일넷 DNS(서버 AdGuard)에는 tag:gpu 허가가 없다. 파드 DNS는 컨테이너 기본값.
$TS up --auth-key="file:$KEYF" --hostname="$TS_HOST" --accept-dns=false --timeout=60s
rm -f "$KEYF"
log "tailscale up as $($TS status --json | $PY -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"])')"

# ---- 3. 자동 종료 감시기(유휴 30분·기동 45분·8시간, 모든 차선의 /metrics 합) + 상태(차선 k: 127.0.0.1:80k1/status) ----
$PY "$S/idle_stop.py" --state-dir "$S" --role "$ROLE" --lanes "$LANES" >>"$S/idle_stop.log" 2>&1 &

# ---- 4. 가중치·품질 자료: HF 커밋으로 고정해 받고, 받은 파일마다 sha256 을 여기 적은 값과 대조(다르면 HALT) ----
export HOME="$S/cache" HF_HOME="$S/cache/huggingface"
Q="$S/quality-data"
Q_SHA='5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91  wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet
ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59  gsm8k/main/test-00000-of-00001.parquet
808a5126898c11f5da89bfb35f66dbee67d69544fe5a7d4f9903370d7af5259d  fineweb2/data/dan_Latn/test/000_00000.parquet'
pin_check() {  # $1 폴더, $2 "sha256  상대경로" 줄들, $3 이름
  ( cd "$1" && printf '%s\n' "$2" | sha256sum -c --quiet --strict ) || halt "$3 sha256 불일치(위 줄 참고)"
  log "$3 sha256 일치 ($(printf '%s\n' "$2" | wc -l | tr -d ' ')개)"
}
# 모델 표(serving/models.toml)에서 고르고, 고정 리비전으로 받아 pins 와 대조한다(대조를 통과해야만 표시를 남긴다).
if [ "$ROLE" = serve ]; then
  serving_select "$MODEL_ALIAS" "$ADAPTER"
  serving_fetch
else
  # 차선 env 마다 받기(같은 모델은 표시를 보고 건너뜀). 준비(overlay·채팅 틀)는 품질 자료 뒤 serving_prepare 에서
  while read -r k port cards alias adapter q; do
    serving_use "$S/lanes/lane-$k.env"
    serving_fetch
  done < "$S/lanes/plan.txt"
fi
# 품질 자료(bench/quality_battery.py 가 읽는 3개 parquet, 약 47 MB)
if [ "$QUALITY" != "0" ] && [ ! -e "$Q/.stewart-ok" ]; then
  $HF download Salesforce/wikitext wikitext-2-raw-v1/test-00000-of-00001.parquet --repo-type dataset \
    --revision b08601e04326c79dfdd32d625aee71d232d685c3 --local-dir "$Q/wikitext" >/dev/null
  $HF download openai/gsm8k main/test-00000-of-00001.parquet --repo-type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 --local-dir "$Q/gsm8k" >/dev/null
  $HF download HuggingFaceFW/fineweb-2 data/dan_Latn/test/000_00000.parquet --repo-type dataset \
    --revision af9c13333eb981300149d5ca60a8e9d659b276b9 --local-dir "$Q/fineweb2" >/dev/null
  pin_check "$Q" "$Q_SHA" quality-data
  touch "$Q/.stewart-ok"
fi
export DFLASH2=0                      # DFlash2 드래프터는 받지 않는다(시범 범위 밖, sha 고정 대상 아님)
if [ "$ROLE" = serve ]; then
  serving_prepare                     # vllm-patched: overlay 에 고정 패치 묶음(원본·덧댄 뒤 sha 대조, 실패면 HALT)
  log "weights pinned model=$M_ALIAS rev=$M_REV adapter=$M_ADAPTER fast=$FAST_VARIANT:${M_FAST_REV:-none}"
else
  while read -r k port cards alias adapter q; do
    serving_use "$S/lanes/lane-$k.env"
    serving_prepare                   # 같은 패치 묶음의 overlay 는 앞 차선이 만든 것을 다시 대조만(patchset build 가 같은 결과)
    log "weights pinned lane=$k model=$M_ALIAS rev=$M_REV adapter=$M_ADAPTER fast=$FAST_VARIANT:${M_FAST_REV:-none}"
  done < "$S/lanes/plan.txt"
fi
export HF_HUB_OFFLINE=1               # 이제부터 HF 로 나가지 않는다(prepare·vLLM 모두 로컬 사본)

# 품질 점검의 PPL(prompt_logprobs)은 SPEC=mtp 와 같이 쓰면 엔진이 죽거나 값이 깨진다(실측: serve 파드의 EngineDeadError 때
# 첫 요청이 PPL 이었다, quality_battery.py 머리말 gotcha 46). 지금 차선 env(M_*)의 추측 디코딩이 mtp 면 GSM8K 만 돌린다.
quality_flags() {
  local spec=${M_SPEC:-}
  [ "$M_ADAPTER" = hyperqwen-8cf642e ] && spec=${SPEC:-${M_HQ_SPEC:-}}
  [ -n "${M_LANE_SPEC:-}" ] && spec=$M_LANE_SPEC        # 계획이 차선 하나만 바꾼 값이 이긴다
  if [ "$spec" = mtp ]; then echo "--gsm-only"; fi
}

# ---- 5. vLLM(어댑터: serving_select 가 고름) ----
trap - ERR                            # 여기부터는 서버가 맡는다(실패하면 감시기의 기동 유예 45분이 끈다)
log "vllm start model=$M_ALIAS adapter=$M_ADAPTER image=$IMAGE_KIND CTX=$CTX SPEC=$SPEC PREFIX_CACHE=$PREFIX_CACHE"
# 품질 점검(첫 파드 검사): 서버가 /health 를 내면 bench/quality_battery.py(ppl en·da·code + GSM8K 200)를 한 번 돌린다.
# 결과는 $S/quality-<태그>.log·$Q/result_<태그>.json, 요약은 /status 의 quality.
# QUALITY=auto(기본): 이 설정(SIG)으로 끝까지 돈 적이 없을 때만. 1: 매번. 0: 안 함. 끝난 설정은 $S/quality-done 에 남긴다.
if [ "$ROLE" = "eval" ]; then
  # ---- 5e. eval: 계획의 차선마다 serving.sh 의 serve_lane(카드 = 계획 표, 포트 80k0, 차선 env). 차례는 T20 그대로 ----
  # 차선 0 을 먼저 띄우고 /health 가 나면(torch compile 캐시가 찬 뒤) 나머지를 띄운다. 무게 올리기만 차례로(앞 차선 로그에
  # WEIGHTS_DONE 이 나면 다음 차선, 최대 20분) — 호스트 RAM 고점·디스크 읽기를 겹치지 않게. 차선 0 이 안 뜨면 나머지도 안 띄운다.
  # quality=1 차선은 /health 뒤 품질 점검(VLLM_API=그 포트, QUALITY_MODEL=차선 env 의 M_SERVED). 통과하면 $S/lanes/quality-ok-k.
  WEIGHTS_DONE='model weights took|Model loading took'
  lane_log() { if [ "$1" = 0 ]; then echo "$S/vllm.log"; else echo "$S/vllm-$1.log"; fi; }
  lane_quality() {   # $1 k $2 port — 차선 env 를 읽어 품질 점검(배경)
    local k=$1 port=$2 tag lsig
    ( . "$S/lanes/lane-$k.env"
      lsig="$SIG LANE=$k M=$M_ALIAS A=$M_ADAPTER"
      if [ "$QUALITY" = "0" ] || { [ "$QUALITY" = "auto" ] && grep -qxF "$lsig" "$S/quality-done" 2>/dev/null; }; then
        log "quality skip lane=$k(QUALITY=$QUALITY)"; [ "$QUALITY" = 0 ] || touch "$S/lanes/quality-ok-$k"; exit 0
      fi
      tag="L$k-$(date -u +%Y%m%dT%H%MZ)-$M_ALIAS-$M_ADAPTER-fast$FAST_VARIANT-$CTX-pc$PREFIX_CACHE"
      qf=$(quality_flags)
      log "quality run lane=$k tag=$tag ppl=${qf:+off(SPEC=mtp)}${qf:-on}"
      for i in $(seq 540); do              # lane_gone 은 아래 정의(부를 때 찾는다)
        curl -sf -o /dev/null "http://127.0.0.1:$port/health" && break
        lane_gone "$k" && { log "quality skip lane=$k(차선이 끝났거나 실패)"; exit 0; }
        sleep 10
      done
      sleep 30; rc=0
      # shellcheck disable=SC2086
      VLLM_API="http://127.0.0.1:$port/v1" QUALITY_DATA="$Q" QUALITY_MODEL="$M_SERVED" PYTHONPATH="$S/pylib" \
        $PY "$S/serving/bench/quality_battery.py" "$tag" $qf >"$S/quality-$tag.log" 2>&1 || rc=$?
      echo "exit=$rc" >>"$S/quality-$tag.log"
      if [ "$rc" = 0 ]; then echo "$lsig" >> "$S/quality-done"; touch "$S/lanes/quality-ok-$k"; fi ) &
  }
  # T28: 차선마다 표시 파일 — start-k(띄운 시각 epoch), pid-k(서버 pid, exec 라 python 그대로), rc-k(서버가 끝난 종료 코드).
  # 감시기가 이것으로 차선 기동 상한(LANE_BOOT_MIN)·프로세스 끝남을 보고 state=failed·why(차선 로그 끝 줄)를 단다(failed-k).
  start_lane() {     # $1 k $2 port $3 cards
    rm -f "$S/lanes/rc-$1" "$S/lanes/failed-$1" "$S/lanes/pid-$1"; date +%s > "$S/lanes/start-$1"
    ( ( export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$3; serve_lane "$1" "$2" "$S/lanes/lane-$1.env" ) \
        >>"$(lane_log "$1")" 2>&1 &
      echo "$!" > "$S/lanes/pid-$1"; rc=0; wait "$!" || rc=$?; echo "$rc" > "$S/lanes/rc-$1" ) &
    PIDS="${PIDS:+$PIDS }$!"
    log "lane $1 start port=$2 cards=$3"
  }
  lane_gone() { [ -e "$S/lanes/rc-$1" ] || [ -e "$S/lanes/failed-$1" ]; }   # 차선 k 가 끝났거나 감시기가 실패로 정함
  PIDS=""
  read -r k0 p0 c0 a0 ad0 q0 < "$S/lanes/plan.txt"
  start_lane "$k0" "$p0" "$c0"
  [ "$q0" = 1 ] && lane_quality "$k0" "$p0"
  for i in $(seq 540); do
    curl -sf -o /dev/null "http://127.0.0.1:$p0/health" && break
    kill -0 "${PIDS%% *}" 2>/dev/null || break
    lane_gone "$k0" && break
    sleep 10
  done
  if curl -sf -o /dev/null "http://127.0.0.1:$p0/health"; then
    log "lane 0 health ok"
    prev=$(lane_log 0) pk=0
    while read -r k port cards alias adapter q; do
      [ "$k" = 0 ] && continue
      for i in $(seq 120); do                       # 앞 차선의 무게 올림 끝(최대 20분). 앞 차선이 끝났거나 실패면 바로(T28)
        grep -qiE "$WEIGHTS_DONE" "$prev" 2>/dev/null && break
        lane_gone "$pk" && { log "lane $pk 이 끝났거나 실패 — 기다리지 않고 lane $k 를 띄운다"; break; }
        sleep 10
      done
      start_lane "$k" "$port" "$cards"
      [ "$q" = 1 ] && lane_quality "$k" "$port"
      prev=$(lane_log "$k") pk=$k
    done < "$S/lanes/plan.txt"
  else
    log "lane 0 이 뜨지 않았다 — 다른 차선은 띄우지 않는다(감시기의 기동 유예가 끈다)"
  fi
  # shellcheck disable=SC2086
  wait $PIDS || true
  log "모든 차선이 끝났다 — 감시기가 유휴로 멈춘다"
  exec sleep infinity
fi
# ---- 5s. serve: 차선 하나(:8000), 그대로 ----
QDONE="$S/quality-done"
if [ "$QUALITY" = "1" ] || { [ "$QUALITY" = "auto" ] && ! grep -qxF "$SIG" "$QDONE" 2>/dev/null; }; then
  TAG="$(date -u +%Y%m%dT%H%MZ)-$M_ALIAS-$M_ADAPTER-fast$FAST_VARIANT-$CTX-pc$PREFIX_CACHE"
  QF=$(quality_flags)
  log "quality run tag=$TAG ppl=${QF:+off(SPEC=mtp)}${QF:-on}"
  ( for i in $(seq 540); do curl -sf -o /dev/null http://127.0.0.1:8000/health && break; sleep 10; done
    sleep 30; rc=0
    # shellcheck disable=SC2086
    VLLM_API=http://127.0.0.1:8000/v1 QUALITY_DATA="$Q" QUALITY_MODEL="$M_SERVED" PYTHONPATH="$S/pylib" \
      $PY "$S/serving/bench/quality_battery.py" "$TAG" $QF >"$S/quality-$TAG.log" 2>&1 || rc=$?
    echo "exit=$rc" >>"$S/quality-$TAG.log"
    if [ "$rc" = 0 ]; then echo "$SIG" >> "$QDONE"; mkdir -p "$S/lanes"; touch "$S/lanes/quality-ok-0"; fi ) &
else
  log "quality skip(QUALITY=$QUALITY, 이 설정은 이미 끝남)"
fi
# vLLM 로그를 볼륨에도 남긴다(KV 캐시 크기·verify 판정 줄을 상태 응답이 읽는다). 컨테이너 로그에도 그대로 나간다.
exec > >(tee -a "$S/vllm.log") 2>&1
[ -n "$STEWART_OVERLAY" ] && log "image-overlay $(PYTHONPATH="$STEWART_OVERLAY" $PY "$SV/image_info.py" --overlay "$STEWART_OVERLAY" 2>&1 | head -c 1500)"
serve_lane 0 8000                     # 차선 0, :8000(v7 은 차선마다 CUDA_VISIBLE_DEVICES 를 정해 serve_lane k 80k0)
