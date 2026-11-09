#!/bin/bash
# 우리 첫 4비트 변환: demo2 BF16 → 0 채우기(17536) → AutoRound W4A16 → MTP 붙이기 → HyperQwen prepare(int8 head·draft).
# 변환 전용 Vast 인스턴스, 이미지 = HyperQwen 8cf642e(모델 A 와 같은 torch 2.13·transformers 5.15.0·compressed-tensors 0.17.0).
# 묶음(quant-files)을 받아 sha 를 대조한 뒤 이 스크립트를 돌린다. 단계마다 표시 파일을 남겨 다시 돌리면 이어서 한다.
# 결과는 HF 가 아니라 보관 서버로 간다. 이 인스턴스는 테일넷(stewart-quant, tag:gpu)에 들어와 :8001 상태·:8090 내보내기를 열고,
# 집이 먼저 연결해 끌어간다(pull). 이 인스턴스 → 집 연결은 없다. 쓰기 토큰·사람 입력도 없다.
set -euo pipefail
# 키(Vast 인스턴스 키·Tailscale 인증 키)는 다른 프로세스를 띄우기 전에 /dev/shm 파일로 옮기고 환경에서 지운다(pod-start 와 같게)
umask 077; VKEY=/dev/shm/vast-key; printf '%s' "${CONTAINER_API_KEY:-}" > "$VKEY"; unset CONTAINER_API_KEY
KEYF=/dev/shm/ts-authkey; printf '%s' "${TS_AUTHKEY:-}" > "$KEYF"; unset TS_AUTHKEY; umask 022
Q=${Q:-/workspace/quant}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/app/venv/bin/python HF=/app/venv/bin/hf
mkdir -p "$Q/done" "$Q/cache"
export HOME="$Q/cache" HF_HOME="$Q/cache/huggingface" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
log() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$Q/convert-log.tsv"; }
stop_instance() {   # 돈 지키기: 정지(디스크와 단계 표시는 남는다. 다시 켜면 이어서 한다)
  if [ -n "${CONTAINER_ID:-}" ] && [ -s "$VKEY" ]; then
    (umask 077; printf 'Authorization: Bearer %s\n' "$(cat "$VKEY")" > /dev/shm/vast-hdr)
    code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT -H @/dev/shm/vast-hdr -H "Content-Type: application/json" \
      -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" || true)
    rm -f /dev/shm/vast-hdr
    log "stop api HTTP $code"
  else
    log "stop api 건너뜀(CONTAINER_ID·CONTAINER_API_KEY 없음)"
  fi
}
# HALT: 이유 한 줄(stdout·convert-log.tsv) → 900초(그동안 :8001 /status 의 halted·log_tail 과 logs 로 이유를 본다) → 스스로 정지
HALT_WAIT_S=900
halt() { trap - ERR; log "HALT: $*"; log "HALT ${HALT_WAIT_S}초 뒤 스스로 정지"; sleep "$HALT_WAIT_S" || true; stop_instance; exec sleep infinity; }
trap 'halt "명령 실패(줄 $LINENO)"' ERR
# 최대 켜짐 감시: 어느 단계가 멈춰 버려도 MAX_HOURS(기본 6) 뒤에는 정지한다. 다시 켜면 표시 파일로 이어서 하고 감시도 새로 돈다
NT=$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["targets"]))' "$(dirname "${BASH_SOURCE[0]}")/targets.json")
MAXH=${MAX_HOURS:-$(( 4 * NT + 2 ))}                      # 대상마다 4시간 + 받기·감사 2시간
( sleep "$(( MAXH * 3600 ))"; log "최대 켜짐 ${MAXH} h — 정지"; stop_instance ) &
step() { [ -e "$Q/done/$1" ]; }
mark() { mkdir -p "$(dirname "$Q/done/$1")"; touch "$Q/done/$1"; log "done $1"; }

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
"$Q/bin/tailscaled" --tun=userspace-networking --statedir="$Q/ts-state" --socket=/tmp/tailscaled.sock \
  --no-logs-no-support >>"$Q/tailscaled.log" 2>&1 &
sleep 2
if [ -s "$KEYF" ]; then
  $TS up --auth-key="file:$KEYF" --hostname=stewart-quant --accept-dns=false --timeout=60s || halt "테일넷에 못 들어갔다"
else
  $TS up --hostname=stewart-quant --accept-dns=false --timeout=60s || halt "TS_AUTHKEY 없음·상태도 없음"
fi
rm -f "$KEYF"
log "tailscale up stewart-quant"
$PY "$HERE/quantsrv.py" --q "$Q" --server-ip "$SERVER_TS_IP" --ts-cmd "$TS" >>"$Q/quantsrv.log" 2>&1 &

# 변환 대상은 묶음의 targets.json(서빙 표의 quant_src, order 차례). 대상마다 $Q/t/<별칭>/ 아래에서 단계 표시를 따로 남긴다
PILE_REV=127bfedcd5047750df5ccf3a12979a47bfa0bafa         # NeelNanda/pile-10k
PILE_SHA=a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31
PILE_FILE=data/train-00000-of-00001-4746b8785c874cc7.parquet
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

# ---- 1. 도구: auto-round 0.14.2 와 그 의존성만 더한다. 이미지 venv 의 판은 constraints 로 묶어 바뀌지 않게 ----
if ! step 1-tools; then
  /app/venv/bin/pip freeze --exclude-editable | grep -E '^[A-Za-z0-9._-]+==' > "$Q/constraints-image.txt"
  AR_SHA=4265c11d33a6f688c211bfed87e4698b685473efe1d6faf95d744a3f7c19ca24        # auto_round-0.14.2-py3-none-any.whl(PyPI)
  /app/venv/bin/pip download -q --no-deps "auto-round==0.14.2" -d "$Q/wheels"
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
  if ! /app/venv/bin/pip install -q -c "$Q/constraints-solve.txt" -r "$Q/req-pinned.txt" \
       "$Q/wheels/auto_round-0.14.2-py3-none-any.whl" > "$Q/pip-install.log" 2>&1; then
    grep -v '^\[notice\]' "$Q/pip-install.log" | tail -n 30 | tee -a "$Q/convert-log.tsv"
    halt "도구 설치 실패(위 pip 줄 — 조리법 판이 이미지 판과 부딪친다)"
  fi
  /app/venv/bin/pip freeze > "$Q/pip-freeze-after.txt"
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
convert_target() {   # 별칭 저장소 리비전 pins 0채우기(0=안 함) 템플릿
  local alias=$1 repo=$2 rev=$3 pins=$4 pad=$5 tpl=$6 rc
  local T="$Q/t/$1" SRC PAD OUTB OUTI
  SRC="$T/src" OUTB="$T/out-bf16heads" OUTI="$T/out-int8heads"
  if [ "$pad" = 0 ]; then PAD="$SRC"; else PAD="$T/padded"; fi
  mkdir -p "$T"
  step "$alias/11-built" && { log "[$alias] 이미 끝남"; return 0; }
  # 2. 원본 BF16(고정 리비전 + 파일 sha)
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
    $PY "$HERE/kl_check.py" compare --model "$PAD" --ref "$T/ref.pt" --out "$T/kl-pad.json" --max-mean-kl 1e-4 --min-top1 0.999 \
      | tee -a "$Q/convert-log.tsv" || halt "[$alias] 0 채우기 전후가 다르다(kl-pad.json)"
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
    ( cd /app && BASE_MODEL_DIR="$OUTI" HF_REPO="$repo" FAST_VARIANT=0 DFLASH2=0 HARDEN_TEMPLATES=0 TRANSLATE_EFFORT=0 \
        bash docker/prepare.sh ) 2>&1 | tee "$T/prepare.log" | tail -n3
    find "$OUTI" -name '*.bak' -delete
    mark "$alias/8-prepare"
  fi
  # 9. HyperQwen verify + 서빙(CTX=long SPEC=mtp PREFIX_CACHE=1) + quality_battery(GSM8K — mtp 에서 prompt_logprobs 를 보내지 않는다.
  #    PPL 쪽은 7단계 KL 이 맡는다) + 추측 디코딩 수락 누계
  if ! step "$alias/9-serve-quality"; then
    ( cd /app && MODEL="$OUTI" bash verify.sh --no-server ) > "$T/verify.log" 2>&1 || halt "[$alias] verify FAIL(verify.log)"
    tail -n1 "$T/verify.log" | tee -a "$Q/convert-log.tsv"
    ( cd /app && MODEL="$OUTI" PREPARE=0 CTX=long SPEC=mtp PREFIX_CACHE=1 HOST=127.0.0.1 PORT=8000 \
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
      ( cd "$d" && find . -maxdepth 1 -type f -not -name '.*' -not -name '*.bak' -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum ) > "$d.sha256"
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
  log "[$alias] 시작 $repo@${rev:0:12} pad=$pad"
  convert_target "$alias" "$repo" "$rev" "$pins" "$pad" "$tpl"
  run_audit
done <<< "$TARGETS"
run_audit

# ---- 11. 내보낼 목록(export.json): 대상마다 세트 둘(bf16h·int8h)과 대상별 보고, 감사 요약 ----
if ! step 11-export; then
  $PY "$HERE/build_export.py" --q "$Q" --targets "$HERE/targets.json" --pile-rev "$PILE_REV" | tee -a "$Q/convert-log.tsv"
  mark 11-export
fi
trap - ERR
log "변환 끝. 보관 서버가 끌어가기를 기다린다(:8090 /export, 다 받으면 POST /export/done)"
# 집이 POST /export/done 을 보내면 정지. EXPORT_WAIT_MIN 분(기본 120) 안에 안 오면 정지(디스크는 남아 다시 켜면 이어서 기다린다)
for i in $(seq "${EXPORT_WAIT_MIN:-120}"); do step exported && break; sleep 60; done
log "정지(exported=$(step exported && echo yes || echo no))"
stop_instance
exec sleep infinity
