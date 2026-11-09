"""이미지 대조 한 줄: 어느 이미지·vLLM·패치 위에서 도는지 JSON 한 줄로 남긴다. 모델·네트워크 호출 없음.

  python image_info.py [--overlay <overlay 루트>] [--verify-log <entrypoint 로그>]

- vllm.__version__·경로, torch·transformers·flashinfer·compressed-tensors 판, python 판
- nvcc 위치·판, CUDA_HOME(FlashInfer fp8 KV JIT 가 nvcc 를 쓴다)
- /app/patches/series 의 sha256 과 8cf642e 고정값 일치 여부(HyperQwen 이미지일 때)
- overlay 가 있으면 그 표시(stewart-overlay.json)의 묶음 이름·sha
- --verify-log: HyperQwen verify.sh 의 PASS·WARN·FAIL 줄 수와 FAIL 줄(앞 5개)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _ver(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def nvcc_info() -> dict:
    exe = shutil.which("nvcc") or next((str(p) for p in (Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "nvcc",)
                                        if p.is_file()), None)
    out = {"path": exe, "version": None, "cuda_home": os.environ.get("CUDA_HOME")}
    if exe:
        try:
            r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
            m = re.search(r"release ([0-9.]+)", r.stdout)
            out["version"] = m.group(1) if m else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    return out


def verify_summary(log: Path) -> dict:
    if not log.is_file():
        return {"log": str(log), "found": False}
    txt = log.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = [ln.strip() for ln in txt if re.match(r"^\s+(PASS|WARN|FAIL)\s", ln)]
    fails = [r for r in rows if r.startswith("FAIL")]
    return {"found": True, "pass": sum(r.startswith("PASS") for r in rows), "warn": sum(r.startswith("WARN") for r in rows),
            "fail": len(fails), "fail_lines": [f[:160] for f in fails[:5]]}


def info(overlay: str | None = None, verify_log: str | None = None) -> dict:
    d: dict = {"python": sys.version.split()[0], "executable": sys.executable,
               "image_kind": "hyperqwen" if Path("/app/docker/entrypoint.sh").is_file() else "vllm-openai"}
    try:
        import vllm  # noqa: PLC0415
        d["vllm"] = getattr(vllm, "__version__", None)
        d["vllm_path"] = str(Path(vllm.__file__).parent)
    except Exception as e:  # noqa: BLE001 — 깨진 이미지도 한 줄은 남긴다
        d["vllm"] = None
        d["vllm_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    for k, dist in (("torch", "torch"), ("transformers", "transformers"), ("flashinfer", "flashinfer-python"),
                    ("compressed_tensors", "compressed-tensors"), ("huggingface_hub", "huggingface-hub"),
                    ("pyarrow", "pyarrow")):
        d[k] = _ver(dist)
    d["nvcc"] = nvcc_info()
    series = Path("/app/patches/series")
    if series.is_file():
        pinned = None
        man = HERE / "patches" / "hyperqwen-8cf642e.sha256"
        for ln in man.read_text().splitlines() if man.is_file() else []:
            if ln.endswith("  patches/series"):
                pinned = ln.split()[0]
        d["hyperqwen_series_sha256"] = _sha(series)
        d["hyperqwen_series_matches_8cf642e"] = d["hyperqwen_series_sha256"] == pinned
    if overlay:
        mk = Path(overlay) / "stewart-overlay.json"
        if mk.is_file():
            o = json.loads(mk.read_text())
            d["overlay"] = {k: o.get(k) for k in ("set", "set_sha", "hyperqwen_unapplied", "reused")} | {"applied": len(o.get("applied", []))}
    if verify_log:
        d["verify"] = verify_summary(Path(verify_log))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay")
    ap.add_argument("--verify-log")
    a = ap.parse_args()
    print(json.dumps(info(a.overlay, a.verify_log), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
