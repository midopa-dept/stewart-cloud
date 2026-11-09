"""같은 글에서 두 모델의 다음 토큰 분포를 비교한다. 전체 어휘 KL(근사 없음), transformers 로.

  python kl_check.py dump    --model <기준 폴더> --text <parquet> --out ref.pt [--windows 8 --tokens 256]
  python kl_check.py compare --model <비교 폴더> --ref ref.pt --out kl.json [--max-mean-kl 1e-4 --min-top1 0.999]

- 글: wikitext-2 test parquet(pod 의 품질 자료와 같은 고정 파일)의 문단을 이어 붙여 앞에서부터 windows × tokens 를 자른다.
- dump: 위치마다 log_softmax(전체 어휘)를 fp16 으로 저장(8×256×248320×2바이트 ≈ 1 GB).
- compare: KL(기준 ‖ 비교) 평균·p99·최대, top-1 일치율, 기준 쪽 ppl·비교 쪽 ppl. 문턱을 주면 넘을 때 exit 1.
- 0 채우기 전후(같은 수학)는 KL ≈ 0, top-1 ≈ 100% 여야 한다. 양자화 전후는 문턱 없이 숫자만 남긴다.
- compressed-tensors 체크포인트는 transformers 가 BF16 로 풀어 읽는다(같은 코드 경로로 비교).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def load_text(parquet: Path, need_chars: int) -> str:
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).column("text").to_pylist()
    out, n = [], 0
    for r in rows:
        r = (r or "").strip()
        if len(r) < 200:            # 머리글·빈 줄은 건너뛴다
            continue
        out.append(r)
        n += len(r)
        if n >= need_chars:
            break
    return "\n\n".join(out)


def load_model(path: str):
    """config 의 architectures 로 클래스를 고른다(…ForConditionalGeneration 은 비전 포함 클래스로 — 텍스트 전용으로 읽으면
    키 이름이 달라 무게가 비고 KL 이 엉터리가 된다). 빠진·남는 무게가 있으면 멈춘다(mtp.* 만 허용: transformers 에 MTP 가 없다)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    arch = (json.loads((Path(path) / "config.json").read_text()).get("architectures") or [""])[0]
    cls = AutoModelForImageTextToText if arch.endswith("ForConditionalGeneration") else AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    m, info = cls.from_pretrained(path, dtype=torch.bfloat16, device_map="auto", output_loading_info=True)
    missing = [k for k in info.get("missing_keys", [])]
    extra = [k for k in info.get("unexpected_keys", []) if not k.startswith("mtp.")]
    if missing or extra:
        raise SystemExit(f"무게 읽기가 맞지 않다: missing {missing[:5]}({len(missing)}) unexpected {extra[:5]}({len(extra)})")
    m.eval()
    return tok, m


def windows_of(tok, text: str, windows: int, tokens: int):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < windows * tokens:
        raise SystemExit(f"글이 짧다: {len(ids)} 토큰 < {windows * tokens}")
    return [ids[i * tokens:(i + 1) * tokens] for i in range(windows)]


def logprobs(m, win):
    import torch
    with torch.no_grad():
        x = torch.tensor([win], device=next(m.parameters()).device)
        lg = m(input_ids=x).logits[0].float()
        return torch.log_softmax(lg, dim=-1)


def kl_stats(ref_lp, test_lp, ids) -> dict:
    """ref_lp·test_lp: [T, V] log 확률(같은 위치). ids: 그 창의 토큰(다음 토큰 ppl 용)."""
    import torch
    p = ref_lp.exp()
    kl = (p * (ref_lp - test_lp)).sum(-1)                    # [T]
    top1 = (ref_lp.argmax(-1) == test_lp.argmax(-1)).float()
    nxt = torch.tensor(ids[1:], device=ref_lp.device)
    nll_ref = -ref_lp[:-1].gather(1, nxt[:, None]).squeeze(1)
    nll_test = -test_lp[:-1].gather(1, nxt[:, None]).squeeze(1)
    return {"kl": kl.clamp_min(0).cpu(), "top1": top1.cpu(), "nll_ref": nll_ref.cpu(), "nll_test": nll_test.cpu()}


def summarize(parts: list[dict]) -> dict:
    import torch
    kl = torch.cat([p["kl"] for p in parts])
    top1 = torch.cat([p["top1"] for p in parts])
    nr = torch.cat([p["nll_ref"] for p in parts])
    nt = torch.cat([p["nll_test"] for p in parts])
    return {"positions": int(kl.numel()), "mean_kl": float(kl.mean()), "p99_kl": float(kl.quantile(0.99)),
            "max_kl": float(kl.max()), "top1_agree": float(top1.mean()),
            "ppl_ref": math.exp(float(nr.mean())), "ppl_test": math.exp(float(nt.mean()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--model", required=True)
    d.add_argument("--text", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--windows", type=int, default=8)
    d.add_argument("--tokens", type=int, default=256)
    c = sub.add_parser("compare")
    c.add_argument("--model", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-mean-kl", type=float)
    c.add_argument("--min-top1", type=float)
    a = ap.parse_args(argv)
    import torch
    t0 = time.time()
    tok, m = load_model(a.model)
    if a.cmd == "dump":
        wins = windows_of(tok, load_text(Path(a.text), a.windows * a.tokens * 8), a.windows, a.tokens)
        lps = [logprobs(m, w).to(torch.float16).cpu() for w in wins]
        torch.save({"model": a.model, "windows": wins, "logprobs": lps}, a.out)
        print(json.dumps({"dumped": len(wins), "tokens": a.tokens, "elapsed_s": round(time.time() - t0, 1)}))
        return 0
    ref = torch.load(a.ref)
    parts = []
    for w, rl in zip(ref["windows"], ref["logprobs"]):
        tl = logprobs(m, w)
        parts.append(kl_stats(rl.to(tl.device).float(), tl, w))
    s = summarize(parts) | {"ref_model": ref["model"], "model": a.model, "elapsed_s": round(time.time() - t0, 1)}
    ok = True
    if a.max_mean_kl is not None and s["mean_kl"] > a.max_mean_kl:
        ok = False
    if a.min_top1 is not None and s["top1_agree"] < a.min_top1:
        ok = False
    s["pass"] = ok if (a.max_mean_kl is not None or a.min_top1 is not None) else None
    Path(a.out).write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(s, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
