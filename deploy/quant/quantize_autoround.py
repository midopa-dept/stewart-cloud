"""W4A16 양자화: dbirks / ababaka 조리법 그대로(모델 A 와 같은 판).

  python quantize_autoround.py --src <0 채운 BF16 폴더> --out <결과 폴더> --recipe <recipe.json>

조리법(바꾸지 않는다)
- auto-round 0.14.2, scheme W4A16(int4, group 128, 대칭), 결과 형식 llm_compressor(compressed-tensors pack-quantized)
- 보정 자료 NeelNanda/pile-10k(리비전은 run-convert.sh 가 HF 캐시에 고정), nsamples 128, seqlen 2048, batch_size 4, iters 200, seed 42
- device_map=0, trust_remote_code=True, quant_nontext_module=False(비전은 BF16)
- BF16 로 남김(layer_config bits=16): linear_attn.in_proj_a / in_proj_b. lm_head 는 기본으로 양자화하지 않는다. mtp 는 transformers 모델에 없어 결과에 빠진다(graft_missing.py 가 BF16 으로 넣는다)
- 시작 전에 intermediate_size 가 128 의 배수인지 본다(아니면 pad_intermediate.py 를 먼저).
recipe.json 에 실제로 쓴 인자·판·시각·GPU 를 남긴다(HF 저장소 README 의 근거).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

RECIPE = {"scheme": "W4A16", "dataset": "NeelNanda/pile-10k", "nsamples": 128, "seqlen": 2048, "batch_size": 4,
          "iters": 200, "seed": 42, "device_map": 0, "trust_remote_code": True, "quant_nontext_module": False}
FORMAT = "llm_compressor"
KEEP_BF16 = ("linear_attn.in_proj_a", "linear_attn.in_proj_b")
WANT = {"auto-round": "0.14.2", "transformers": "5.15.0", "compressed-tensors": "0.17.0"}


def bf16_layer_config(weight_keys) -> dict:
    """색인의 키에서 모듈 이름을 만들어 in_proj_a/b 를 bits=16 으로(dbirks: 'built from module names')."""
    out = {}
    for k in weight_keys:
        if k.endswith(".weight") and k[: -len(".weight")].endswith(KEEP_BF16):
            out[k[: -len(".weight")]] = {"bits": 16}
    return out


def check_intermediate(cfg: dict) -> int:
    t = cfg.get("text_config") or cfg
    i = int(t["intermediate_size"])
    if i % 128:
        raise SystemExit(f"intermediate_size {i} 가 128 의 배수가 아니다 — pad_intermediate.py 를 먼저")
    return i


def versions() -> dict:
    import importlib.metadata as md
    out = {}
    for d in ("auto-round", "torch", "transformers", "compressed-tensors", "accelerate", "datasets", "safetensors",
              "huggingface-hub", "numpy"):
        try:
            out[d] = md.version(d)
        except md.PackageNotFoundError:
            out[d] = None
    return out


def settle_output(out: Path, folders) -> str | None:
    """auto-round 0.14.2 의 quantize_and_save 는 받은 폴더 아래 '<모델 폴더 이름>-w4g128' 같은 하위 폴더에 쓴다.
    그 폴더(돌려준 folders, 없으면 색인을 가진 하위 폴더 하나)의 파일을 out 바로 아래로 옮기고 빈 폴더를 지운다.
    옮긴 하위 폴더 이름(이미 out 이면 None)."""
    import shutil
    cands = [Path(f) for f in (folders if isinstance(folders, (list, tuple)) else [folders]) if f]
    cands = [f for f in cands if (f / "model.safetensors.index.json").exists() or (f / "config.json").exists()]
    if not cands:
        cands = [d for d in out.iterdir() if d.is_dir() and (d / "model.safetensors.index.json").exists()]
    if len(cands) != 1:
        if (out / "model.safetensors.index.json").exists():
            return None
        raise SystemExit(f"양자화 결과 폴더를 하나로 못 정했다: {[str(c) for c in cands]}")
    sub = cands[0].resolve()
    if sub == out.resolve():
        return None
    if sub.parent != out.resolve():
        raise SystemExit(f"양자화 결과가 출력 폴더 밖이다: {sub}")
    for p in sub.iterdir():
        dst = out / p.name
        if dst.exists():
            raise SystemExit(f"옮길 곳에 이미 있다: {dst}")
        shutil.move(str(p), str(dst))
    sub.rmdir()
    return sub.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", required=True)
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    cfg = json.loads((src / "config.json").read_text())
    inter = check_intermediate(cfg)
    keys = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    layer_config = bf16_layer_config(keys)
    if len(layer_config) == 0:
        raise SystemExit("in_proj_a/b 를 못 찾았다(모델 구조 확인)")
    ver = versions()
    bad = {k: (ver.get(k), v) for k, v in WANT.items() if ver.get(k) != v}
    if bad:
        raise SystemExit(f"도구 판이 조리법과 다르다: {bad}")
    import torch
    from auto_round import AutoRound
    rec = {"recipe": RECIPE, "format": FORMAT, "layer_config_bf16": len(layer_config), "keep_bf16": list(KEEP_BF16),
           "intermediate_size": inter, "src": str(src), "versions": ver, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(a.recipe).write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    t0 = time.time()
    ar = AutoRound(str(src), layer_config=layer_config, **RECIPE)
    res = ar.quantize_and_save(output_dir=str(out), format=FORMAT)
    rec["moved_from"] = settle_output(out, res[1] if isinstance(res, tuple) and len(res) > 1 else None)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if torch.cuda.is_available():
        rec["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    Path(a.recipe).write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(out), "elapsed_s": rec["elapsed_s"], "peak_vram_gib": rec.get("peak_vram_gib")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
