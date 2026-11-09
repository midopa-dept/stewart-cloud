#!/bin/bash
# stewart pod 기동 스크립트(T23). 템플릿 Docker Options 의 SB(이 파일의 base64)로 들어오고, Args 가 sha256 을 대조한 뒤에만 돌린다.
# 하는 일은 v6~v8 의 Args 와 같다: gist($PU)에서 묶음을 받아 줄바꿈을 지우고 sha256($PS)을 대조한 뒤 풀어 pod-start.sh 로 넘긴다.
# 다른 점: 실패하면 이유를 stdout 에 한 줄 남기고 900초 뒤 스스로 정지한다(인스턴스 범위 키 CONTAINER_API_KEY, idle_stop 과 같은 길).
# 256자 Args 에는 정지 요청이 들어가지 않아 이 파일로 옮겼다. 비밀값은 들어 있지 않다(키는 Vast 가 컨테이너에 넣는 env).
set -uo pipefail
S=/workspace/stewart
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
mkdir -p "$S" && cd "$S" || stop_later "작업 폴더 $S 를 만들 수 없다"
[ -n "${PU:-}" ] || stop_later "PU 없음(템플릿 Docker Options 의 env 가 빠졌다 — 만들 때 env 합치기를 본다)"
[[ ${PS:-} =~ ^[0-9a-f]{64}$ ]] || stop_later "PS 없음·꼴이 다르다(64자 hex)"
curl -fsSL --retry 3 --max-time 120 "$PU" | tr -d '\r\n' > p || stop_later "pod files 받기 실패(gist 주소·망)"
[ -s p ] || stop_later "pod files 가 비었다"
echo "$PS  p" | sha256sum -c --quiet --strict || stop_later "pod files sha256 불일치(gist 리비전과 PS 를 맞춘다)"
base64 -d p | tar xz || stop_later "pod files 풀기 실패"
[ -f pod-start.sh ] || stop_later "pod-start.sh 없음"
exec bash pod-start.sh
