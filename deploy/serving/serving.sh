# deploy/serving/serving.sh — pod-start.sh 가 source 한다. 쓰는 것: $S, log, halt, pin_check(모두 pod-start 의 것).
# 차선 하나(serve 역할·지금): serving_detect → serving_select <별칭> <어댑터> → serving_fetch → serving_prepare → (HF_HUB_OFFLINE=1) → serve_lane 0 8000
# 차선 계획(eval 역할, pod v7): serving_detect → serving_plan <계획> → (HF_HUB_OFFLINE=1) → 차선마다
#   ( CUDA_VISIBLE_DEVICES=<카드> serve_lane k 80k0 "$S/lanes/lane-k.env" ) &   — 줄 목록은 $S/lanes/plan.txt("k 포트 카드 별칭 어댑터 quality")
SV="$S/serving"

# 이미지 종류와 쓸 파이썬·받기 도구. PY 는 감시기·도구용, STEWART_VLLM_PY 는 vllm 이 깔린 파이썬(지금은 같은 것).
serving_detect() {
  if [ -f /app/docker/entrypoint.sh ] && [ -x /app/venv/bin/python ]; then
    IMAGE_KIND=hyperqwen PY=/app/venv/bin/python HF=/app/venv/bin/hf
  else
    IMAGE_KIND=vllm-openai PY=$(command -v python3 || true) HF=$(command -v hf || true)
  fi
  [ -n "$PY" ] || halt "python 없음(이미지 확인)"
  [ -n "$HF" ] || HF="$PY -m huggingface_hub.cli.hf"      # hf 명령이 PATH 에 없을 때(공식 이미지 대비)
  STEWART_VLLM_PY=$PY STEWART_MODELS="$S/models"
  export IMAGE_KIND STEWART_VLLM_PY STEWART_MODELS
  log "image $($PY "$SV/image_info.py" 2>&1 | head -c 2000)"     # 결정 1: vllm 판·패치 series sha·nvcc
}

# 표에서 한 줄 고르기. 별칭·어댑터는 CF 허용 목록을 지난 값이고, 표에 있는지는 serving.py 가 다시 본다.
serving_select() {
  local out
  out=$($PY "$SV/serving.py" select "${1:-default}" "${2:-auto}" "$IMAGE_KIND" "$S/model.env" 2>&1) || halt "모델 고르기 실패: $out"
  serving_use "$S/model.env"
  log "model $out repo=$M_REPO rev=$M_REV dir=$MDIR"
}

# 차선 env 하나를 지금 값으로(받기·준비 함수가 M_*·MDIR 을 쓴다)
serving_use() {
  # shellcheck disable=SC1090
  . "$1"
  MDIR="$S/models/$M_DIR"
}

# 차선 계획: 차선마다 $S/lanes/lane-k.env 를 쓰고, 모델마다 받기·준비를 한다(같은 모델·overlay 는 표시를 보고 건너뛴다).
serving_plan() {
  local out k port cards alias adapter q
  mkdir -p "$S/lanes"
  out=$($PY "$SV/serving.py" plan "$1" "$IMAGE_KIND" "${STEWART_GPUS:-1}" "$S/lanes" 2>&1) || halt "차선 계획 실패: $out"
  printf '%s\n' "$out" > "$S/lanes/plan.txt"
  log "plan $1 gpus=${STEWART_GPUS:-1}: $(tr '\n' ';' < "$S/lanes/plan.txt")"
  while read -r k port cards alias adapter q; do
    serving_use "$S/lanes/lane-$k.env"
    serving_fetch
    serving_prepare
  done < "$S/lanes/plan.txt"
}

# 가중치: 표의 rev 로 받고 pins 와 대조. 대조를 통과해야만 표시(.stewart-rev)를 남긴다.
# 표시가 있으면 다시 받지도 대조하지도 않는다(채팅 틀 손질이 chat_template.jinja 를 제자리에서 고치기 때문).
serving_fetch() {
  # home 별칭: 받는 창구(modelrecv.py)를 127.0.0.1:<recv_port> 에 열고 보관 서버가 밀어 넣기를 기다린다.
  # 사용자 공간 Tailscale 이 테일넷 :<recv_port> 를 그리로 넘긴다. 파드 → 집 연결은 없다. 못 받으면 HALT(→ 인스턴스 정지).
  if [ "$M_SOURCE" = home ]; then
    if [ "$(cat "$MDIR/.stewart-rev" 2>/dev/null)" != "$M_REV" ]; then
      local hip hport htmo rc=0
      read -r hip hport htmo < <($PY "$SV/serving.py" home)
      log "model-fetch wait alias=$M_ALIAS port=$hport timeout=${htmo}m from=$hip"
      $PY "$SV/modelrecv.py" --alias "$M_ALIAS" --dir "$MDIR" --server-ip "$hip" --port "$hport" --timeout-min "$htmo" \
        --status "$S/model-fetch.json" 2>>"$S/modelrecv.log" || rc=$?
      case $rc in
        0) ;;
        2) halt "모델 받기 시간 초과(${htmo}분, $M_ALIAS) — 보관 서버 push 데몬·ACL 확인" ;;
        3) halt "모델 받기 sha 불일치($M_ALIAS)" ;;
        *) halt "모델 받기 실패 rc=$rc($M_ALIAS, modelrecv.log)" ;;
      esac
      pin_check "$MDIR" "$($PY "$SV/serving.py" pins "$M_ALIAS")" "$M_ALIAS"   # 받은 뒤 한 번 더(목록 전체)
      echo "$M_REV" > "$MDIR/.stewart-rev"
    fi
    return 0
  fi
  if [ "$(cat "$MDIR/.stewart-rev" 2>/dev/null)" != "$M_REV" ]; then
    log "download $M_REPO@$M_REV"
    $HF download "$M_REPO" --revision "$M_REV" --local-dir "$MDIR" >/dev/null
    pin_check "$MDIR" "$($PY "$SV/serving.py" pins "$M_ALIAS")" "$M_ALIAS"
    echo "$M_REV" > "$MDIR/.stewart-rev"
  fi
  # HyperQwen fast variant: prepare 의 fetch_fast_variant.py 가 리비전 없이 부르므로 고정 커밋을 HF 캐시에 받고 refs/main 을 적는다(v6 그대로)
  if [ "$M_ADAPTER" = hyperqwen-8cf642e ] && [ "${FAST_VARIANT:-1}" != "0" ] && [ -n "${M_FAST_REV:-}" ] \
     && [ "$(cat "$S/models/.fast-rev" 2>/dev/null)" != "$M_FAST_REV" ]; then
    # shellcheck disable=SC2086
    $HF download "$M_FAST_REPO" $M_FAST_FILES --revision "$M_FAST_REV" >/dev/null
    local H="$HF_HOME/hub/models--${M_FAST_REPO/\//--}"
    pin_check "$H/snapshots/$M_FAST_REV" "$($PY "$SV/serving.py" pins "$M_ALIAS" --fast)" fast
    mkdir -p "$H/refs"
    printf '%s' "$M_FAST_REV" > "$H/refs/main"
    echo "$M_FAST_REV" > "$S/models/.fast-rev"
  fi
}

# 어댑터 준비. vllm-patched: 이미지의 vllm 을 overlay 로 복사해 고정 묶음을 덧댄다(원본 대조·덧댄 뒤 sha 대조, 실패면 HALT).
serving_prepare() {
  STEWART_OVERLAY=${STEWART_OVERLAY:-}     # 계획에서 앞 차선이 만든 overlay 는 그대로 둔다
  if [ "$M_ADAPTER" = vllm-patched ]; then
    local src extra="" out
    src=$($PY -c 'import os, vllm; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
    [ -d "$src" ] || halt "이미지의 vllm 을 못 찾음"
    [ "$IMAGE_KIND" = hyperqwen ] && extra="--unapply-hyperqwen /app"
    STEWART_OVERLAY="$S/overlay/$M_PATCHSET"
    # shellcheck disable=SC2086
    out=$($PY "$SV/patchset.py" build --src "$src" --dst "$STEWART_OVERLAY" --set "$SV/patches/$M_PATCHSET" $extra 2>&1) \
      || halt "패치 묶음 실패: ${out:0:600}"
    log "overlay ${out:0:400}"
    out=$($PY "$SV/patchset.py" check --pkg "$STEWART_OVERLAY/vllm" --set "$SV/patches/$M_PATCHSET" 2>&1) || halt "줄인 verify 실패: $out"
    log "verify-lite $out"
    out=$(PYTHONPATH="$STEWART_OVERLAY" $PY -c 'import vllm, sys; print(vllm.__version__, vllm.__file__)' 2>&1 | tail -n1)
    case "$out" in "0.30.0 $STEWART_OVERLAY/"*) log "overlay import ok: $out" ;; *) halt "overlay 의 vllm 을 못 씀: ${out:0:300}" ;; esac
  fi
  export STEWART_OVERLAY
  # draft 어휘: 집은 pickle(.pt)을 나르지 않는다. 목록 JSON 이 오면 여기서 우리 손으로 .pt 를 만든다(HyperQwen build_draft_vocab 과 같은 int64 텐서)
  if [ -f "$MDIR/mtp_draft_vocab_ids.json" ] && [ ! -f "$MDIR/mtp_draft_vocab_ids.pt" ]; then
    $PY - "$MDIR" <<'PYEOF' || halt "draft 어휘 .pt 만들기 실패"
import json, sys, torch
d = sys.argv[1]
ids = json.load(open(d + "/mtp_draft_vocab_ids.json"))
vocab = json.load(open(d + "/config.json")).get("text_config", {}).get("vocab_size") or 10**7
assert isinstance(ids, list) and ids and all(isinstance(i, int) and 0 <= i < vocab for i in ids) and len(set(ids)) == len(ids)
torch.save(torch.tensor(ids, dtype=torch.int64), d + "/mtp_draft_vocab_ids.pt")
print(len(ids))
PYEOF
    log "draft vocab .pt from json"
  fi
  # 채팅 템플릿 고정: 표의 chat_template(묶음 templates/, sha 대조)을 모델 폴더의 chat_template.jinja 로 쓰고
  # tokenizer_config.json 안의 chat_template 칸은 뺀다(출처를 파일 하나로). 손질(translate 등)은 늘 이 원본 위에 한다.
  if [ -n "${M_CHAT_TPL:-}" ]; then
    echo "$M_CHAT_TPL_SHA  $SV/$M_CHAT_TPL" | sha256sum -c --quiet --strict || halt "채팅 템플릿 sha 불일치($M_CHAT_TPL)"
    cp "$SV/$M_CHAT_TPL" "$MDIR/chat_template.jinja"
    $PY - "$MDIR/tokenizer_config.json" <<'PYEOF' || halt "tokenizer_config.json 의 chat_template 칸을 못 뺐다"
import json, os, sys
p = sys.argv[1]
if os.path.exists(p):
    with open(p, encoding="utf-8") as f:
        j = json.load(f)
    if "chat_template" in j:
        del j["chat_template"]
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        os.replace(p + ".tmp", p)
PYEOF
    log "chat-template pinned $M_CHAT_TPL sha=${M_CHAT_TPL_SHA:0:12}"
  fi
  # 채팅 틀 손질(hyperqwen 어댑터는 자기 prepare 가 한다). 표의 chat_template_fix 만
  if [ "$M_ADAPTER" != hyperqwen-8cf642e ]; then
    for f in $M_CHAT_FIX; do
      case "$f" in
        # 실패(가져오기 오류 등)를 로그 한 줄로 흘려보내지 않는다 — 틀을 못 고친 채 서빙하지 않고 멈춘다
        translate) out=$($PY "$SV/prepare/translate_chat_template.py" "$MDIR" 2>&1) \
                     || halt "chat-template translate 실패($M_ALIAS): ${out: -400}"
                   log "chat-template ${out##*$'\n'}" ;;
        harden) out=$($PY "$SV/prepare/harden_chat_template.py" "$MDIR/chat_template.jinja" 2>&1) \
                     || halt "chat-template harden 실패($M_ALIAS): ${out: -400}"
                log "chat-template ${out##*$'\n'}" ;;
      esac
    done
  fi
  # 품질 점검의 parquet 읽기(pyarrow). HyperQwen venv 에는 있다. 없으면 휠 sha 고정으로 따로 깐다(--no-deps, numpy 는 이미지 것)
  if [ "${QUALITY:-auto}" != "0" ] && ! $PY -c 'import pyarrow' 2>/dev/null; then
    $PY -m pip install -q --no-deps --require-hashes --target "$S/pylib" -r "$SV/requirements-quality.txt" \
      && log "quality deps pyarrow -> $S/pylib" || log "quality deps 설치 실패(품질 점검만 못 함)"
  fi
}

# 차선 k 를 포트 port 로 띄운다(exec — 이 셸이 서버가 된다). 카드는 부르는 쪽의 CUDA_VISIBLE_DEVICES. $3 = 차선 env(없으면 model.env)
serve_lane() {
  local envf=${3:-$S/model.env}
  serving_use "$envf"
  log "serve lane=$1 port=$2 adapter=$M_ADAPTER model=$M_ALIAS tp=$M_TP gpus=${CUDA_VISIBLE_DEVICES:-all}"
  exec $PY "$SV/serving.py" serve "$envf" "$1" "$2"
}
