#!/bin/bash
# 변환(quant) 인스턴스 기동 스크립트. 파드 boot.sh(T23, ② 177a55f)와 같은 꼴: 템플릿 Docker Options 의 SB(이 파일의
# base64)로 들어오고, Args 가 sha256 을 대조한 뒤에만 돌린다. 다른 점은 작업 폴더(/workspace/quant/files)와 넘길 곳(run-convert.sh)뿐.
# 실패하면 이유를 stdout 에 한 줄 남기고 900초 뒤 스스로 정지한다(인스턴스 범위 키 CONTAINER_API_KEY, 헤더 파일 — 명령 줄에 키 없음).
# run-convert.sh 는 exec 하지 않고 부른다: 그 스크립트는 HALT·끝 모두 스스로 정지한 뒤 잠들어 돌아오지 않는다. 돌아왔다면(죽었거나
# 예상 밖으로 끝남) 여기서 이유 한 줄 → 900초 → 정지한다. 비밀값은 들어 있지 않다(키는 Vast 가 컨테이너에 넣는 env).
set -uo pipefail
W=/workspace/quant/files
HALT_WAIT_S=900
stop_later() {
  echo "HALT: $*"
  echo "HALT ${HALT_WAIT_S}초 뒤 스스로 정지(stewart-cloud logs <id> 로 이유를 본다)"
  sleep "$HALT_WAIT_S"
  if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
    (umask 077; printf 'Authorization: Bearer %s\n' "$CONTAINER_API_KEY" > /dev/shm/vast-hdr)   # 키가 명령 줄(ps)에 보이지 않게
    code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT -H @/dev/shm/vast-hdr \
      -H "Content-Type: application/json" -d '{"state":"stopped"}' \
      "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" || true)
    rm -f /dev/shm/vast-hdr
    echo "HALT stop api HTTP $code"
  else
    echo "HALT stop api 건너뜀(CONTAINER_ID·CONTAINER_API_KEY 없음)"
  fi
  exec sleep infinity
}
mkdir -p "$W" && cd "$W" || stop_later "작업 폴더 $W 를 만들 수 없다"
[ -n "${PU:-}" ] || stop_later "PU 없음(템플릿 Docker Options 의 env 가 빠졌다 — 만들 때 env 합치기를 본다)"
[[ ${PS:-} =~ ^[0-9a-f]{64}$ ]] || stop_later "PS 없음·꼴이 다르다(64자 hex)"
curl -fsSL --retry 3 --max-time 120 "$PU" | tr -d '\r\n' > p || stop_later "quant files 받기 실패(gist 주소·망)"
[ -s p ] || stop_later "quant files 가 비었다"
echo "$PS  p" | sha256sum -c --quiet --strict || stop_later "quant files sha256 불일치(gist 리비전과 PS 를 맞춘다)"
base64 -d p | tar xz || stop_later "quant files 풀기 실패"
[ -f run-convert.sh ] || stop_later "run-convert.sh 없음"
bash run-convert.sh
stop_later "run-convert.sh 가 돌아왔다(rc=$?) — convert-log.tsv 를 본다"
