"""vLLM /metrics(표준 입력)에서 추측 디코딩 누계를 JSON 한 줄로(표준 라이브러리만).

  curl -s http://127.0.0.1:8000/metrics | python spec_metrics.py

acceptance = 수락 토큰 / 초안 토큰, mean_accept_len = 1 + 수락 토큰 / 초안 수, per_pos[i] = i 번째 초안 자리의 수락 비율.
추측 디코딩 카운터가 없으면 {"spec_decode": null}.
"""

from __future__ import annotations

import json
import re
import sys

POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
_POS = re.compile(r'position="([0-9]{1,2})"')


def parse(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[name] = out.get(name, 0.0) + val
        if name == POS and (m := _POS.search(line)):
            key = f"{name}[{int(m.group(1))}]"
            out[key] = out.get(key, 0.0) + val
    return out


def view(m: dict) -> dict | None:
    if "vllm:spec_decode_num_drafts_total" not in m:
        return None
    drafts = m.get("vllm:spec_decode_num_drafts_total", 0.0)
    dtok = m.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    acc = m.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    pos = sorted((int(k[len(POS) + 1:-1]), v) for k, v in m.items() if k.startswith(POS + "["))
    return {"drafts": int(drafts), "draft_tokens": int(dtok), "accepted_tokens": int(acc),
            "acceptance": round(acc / dtok, 4) if dtok else None,
            "mean_accept_len": round(1 + acc / drafts, 3) if drafts else None,
            "per_pos": [round(v / drafts, 4) if drafts else None for _, v in pos]}


def main() -> int:
    print(json.dumps({"spec_decode": view(parse(sys.stdin.read()))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
