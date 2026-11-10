"""서빙(role=serve) 파드의 vLLM·HyperQwen 출력에서 사용자 대화 글을 지운다(운영자·도구가 실사용 대화를 보지 않게).

  <vLLM 출력> | python logscrub.py | tee -a vllm.log      (pod-start.sh, ROLE=serve 일 때만)

남기는 것: 수치·토큰 수·시간·상태·오류 종류(예외 이름), 우리 완료 줄(engine-completion-log: 요청 id·끝난 까닭·토큰 수·시간).
지우는 것(줄마다):
- vLLM 요청 기록 꼴(Received request …: prompt: …, params …) — 기본은 꺼져 있다(--enable-log-requests 없음). 켜져도 글은 지운다.
- pydantic 검증 오류의 input_value=… — 요청 본문이 그대로 들어간다.
- 예외 이름: 메시지 꼴의 줄은 이름과 글자 수만(메시지에 요청 조각이 들어갈 수 있다).
- 따옴표 안 24자 넘는 문자열 — 설정 값 몇은 짧아 남고, 요청 글은 지워진다.
- 400자 넘는 줄은 앞 200자만(덤프 꼴).
이것은 마지막 울타리다. 첫 울타리는 vLLM 이 요청 글을 쓰지 않게 하는 설정(요청 기록 꺼짐, VLLM_ENABLE_RESPONSES_API_STORE=0)이다.
"""

from __future__ import annotations

import re
import sys

REQ = re.compile(r"(Received request|Added request|Generated response|request_logger)\b.*", re.I)
INPUT_VALUE = re.compile(r"input_value=(.*?)(, input_type=|$)")
EXC = re.compile(r"^(\s*(?:[\w.]+\.)?[A-Z]\w*(?:Error|Exception|Warning)): (.+)$")
QUOTED = re.compile(r"""(['"])((?:\\.|(?!\1).){24,}?)\1""")
KEYS = re.compile(r"""(\b(?:prompt|messages|content|text|input|completion|delta)\b\s*[:=]\s*)(['"\[{].*)$""", re.I)
LONG = 400


def scrub(line: str) -> str:
    s = line.rstrip("\n")
    m = REQ.search(s)
    if m:
        return s[:m.start()] + m.group(1) + " <요청 글 가림>"
    s = INPUT_VALUE.sub(lambda m: f"input_value=<가림 {len(m.group(1))}자>{m.group(2)}", s)
    m = EXC.match(s)
    if m and "<가림" not in m.group(2):
        return f"{m.group(1)}: <메시지 가림 {len(m.group(2))}자>"
    s = KEYS.sub(lambda m: f"{m.group(1)}<가림 {len(m.group(2))}자>", s)
    s = QUOTED.sub(lambda m: f"{m.group(1)}<가림 {len(m.group(2))}자>{m.group(1)}", s)
    if len(s) > LONG:
        s = s[:200] + f" …<가림 {len(s) - 200}자>"
    return s


def main() -> int:
    for line in sys.stdin:
        sys.stdout.write(scrub(line) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
