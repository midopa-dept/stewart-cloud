"""서빙 모델 표(models.toml)를 읽고, 이미지에 맞는 어댑터로 서버를 띄운다. 표준 라이브러리만(python 3.11+).

  python3 serving.py validate                         # 표 형식 검사
  python3 serving.py names                            # 별칭 목록
  python3 serving.py select <별칭|default> <어댑터|auto> <이미지 종류> <출력 env 파일>
  python3 serving.py pins <별칭> [--fast]             # sha256sum -c 꼴 줄(home 은 files 에서)
  python3 serving.py need <별칭>                      # home 별칭의 받을 목록 JSON(modelrecv.py 가 읽는다)
  python3 serving.py home                             # "server_ip recv_port recv_timeout_min"
  python3 serving.py argv <별칭> <모델 폴더> <포트>   # vllm serve 인자(JSON, 시험·보기용)
  python3 serving.py serve <env 파일> <차선 k> <포트> # 서버로 바뀐다(exec). 카드는 부르는 쪽의 CUDA_VISIBLE_DEVICES
  python3 serving.py plan <계획> <이미지 종류> <카드 수> <출력 폴더>   # 차선마다 lane-<k>.env, 줄마다 "k 포트 카드 별칭 어댑터 quality"
  (T25) 계획의 카드 합 = gpus. 카드 수는 인스턴스 장 수와 같아야 한다(남는 카드는 돈만 든다). fill = true 인 계획은
  gpus 가 최소 장 수이고, 남는 카드를 마지막 차선과 같은 차선(quality false)으로 채운다(예: a-all — 카드마다 A·hyperqwen).

이미지 종류: hyperqwen(/app/docker/entrypoint.sh 와 /app/venv 가 있음) 또는 vllm-openai(그 밖, python3 에 vllm 이 있음).
어댑터
- hyperqwen-8cf642e: HyperQwen 이미지에서만. 차선 0 은 entrypoint.sh single(prepare·verify 포함), 그 밖 차선은 PREPARE=0 VERIFY=0 start_qwen.sh.
- vllm-patched: 어느 이미지든. 이미지의 vllm 을 overlay 로 복사해 고정 패치 묶음을 덧댄 사본으로 띄운다(patchset.py).
  HyperQwen 이미지에서는 이미지의 패치를 거꾸로 풀어 원본을 되살린 뒤 덧댄다(같은 torch·transformers 로 비교).
- vllm-stock: 공식 이미지에서만. 고치지 않은 vllm 그대로.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "models.toml"
LOCAL_NAME = "models.local.toml"         # 공개하지 않는 줄(있으면 합쳐 읽는다, 없어도 돈다)
ADAPTERS = ("hyperqwen-8cf642e", "vllm-patched", "vllm-stock")
IMAGE_KINDS = ("hyperqwen", "vllm-openai")
# 계획이 스스로 갖는 비교 조건(T21b, 메인 11/9 12:0x): CF 에는 PLAN 만 두고 조건은 계획이 정한다. pod-start 가 같은 허용 목록으로 다시 본다.
PLAN_ENV = {"CTX": ("fast", "long"), "SPEC": ("mtp",), "PREFIX_CACHE": ("0", "1"), "FAST_VARIANT": ("0", "1")}
ADAPTER_IMAGES = {"hyperqwen-8cf642e": ("hyperqwen",), "vllm-patched": ("hyperqwen", "vllm-openai"),
                  "vllm-stock": ("vllm-openai",)}
ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,47}$")
PLAN_KEYS = {"note", "image", "gpus", "env", "lanes", "fill"}
MAX_CARDS = 8                                            # 차선 포트 80k0·80k1 은 k = 0..7(프록시 허용 목록·ACL 16포트, T20)
SAFE_RE = re.compile(r"^[A-Za-z0-9._/:@=+-]*$")        # env 파일에 쓰는 값(셸이 그대로 읽어도 안전한 글자만)
REV_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA_LINE_RE = re.compile(r"^[0-9a-f]{64}  [A-Za-z0-9._/-]+$")
SERVE_KEYS = {"max_model_len": int, "gpu_memory_utilization": float, "max_num_seqs": int, "max_num_batched_tokens": int,
              "kv_cache_dtype": str, "mamba_ssm_cache_dtype": str, "spec_method": str, "spec_tokens": int,
              "prefix_cache": bool, "cudagraph_max": int, "language_model_only": bool, "tool_parser": str,
              "reasoning_parser": str, "sse_keep_alive": int, "attention_backend": str}
MODEL_KEYS = {"source", "files", "repo", "rev", "dir", "pins", "arch", "quant", "quant_note", "served_name", "adapters", "patchset", "tp",
              "vram_min_gb", "chat_template_fix", "backend_precision", "fast", "serve", "hyperqwen",
              "pending", "quant_src", "chat_template", "prepared", "mirror"}
QUANT_SRC_KEYS = {"repo", "rev", "pins", "pad_to", "order"}
LANE_KEYS = {"model", "adapter", "quality", "spec"}
LANE_SPECS = ("mtp", "off")          # 차선 하나의 추측 디코딩(없으면 계획·표 값)
TOP_KEYS = {"schema", "default", "home", "images", "models", "plans", "quant"}
CHAT_FIXES = ("translate", "harden")
SOURCES = ("hf", "home")
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")      # home 파일 이름: 폴더 없이 한 조각, 점으로 시작 안 함


def home_digest(files: list[dict]) -> str:
    """home 별칭의 파일 목록 지문(받기 끝 표시·평가 기록용). 이름순 'sha256  크기  이름' 줄의 sha256."""
    import hashlib
    rows = "".join(f"{f['sha256']}  {f['size']}  {f['name']}\n" for f in sorted(files, key=lambda f: f["name"]))
    return hashlib.sha256(rows.encode()).hexdigest()


def _ipv4_in_cgnat(ip: str) -> bool:
    import ipaddress
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.version == 4 and a in ipaddress.ip_network("100.64.0.0/10")


class TableError(Exception):
    pass


def merge(pub: dict, loc: dict | None) -> dict:
    """공개 표 + local 표. 별칭·계획·이미지는 겹치면 안 된다. default 는 local 이 이기고, home·quant 칸은 local 이 더한다."""
    if not loc:
        return pub
    if loc.get("schema", pub.get("schema")) != pub.get("schema"):
        raise TableError("models.local.toml 의 schema 가 공개 표와 다르다")
    t = {k: v for k, v in pub.items()}
    for sec in ("models", "plans", "images"):
        a, b = dict(pub.get(sec) or {}), loc.get(sec) or {}
        both = sorted(set(a) & set(b))
        if both:
            raise TableError(f"models.local.toml 의 {sec} 이 공개 표와 겹친다: {both}")
        a.update(b)
        t[sec] = a
    for sec in ("home", "quant"):
        a, b = dict(pub.get(sec) or {}), loc.get(sec) or {}
        both = sorted(set(a) & set(b))
        if both:
            raise TableError(f"models.local.toml 의 {sec} 칸이 공개 표와 겹친다: {both}")
        a.update(b)
        t[sec] = a
    if "default" in loc:
        t["default"] = loc["default"]
    bad = set(loc) - TOP_KEYS
    if bad:
        raise TableError(f"models.local.toml 에 모르는 칸 {sorted(bad)}")
    return t


def read_raw(path: Path = TABLE, local: Path | None = None) -> dict:
    """검사 전 합친 표(dict). local 을 안 주면 같은 폴더의 models.local.toml(없으면 공개 표만)."""
    pub = tomllib.loads(path.read_text(encoding="utf-8"))
    lp = local if local is not None else path.with_name(LOCAL_NAME)
    loc = tomllib.loads(lp.read_text(encoding="utf-8")) if lp.is_file() else None
    return merge(pub, loc)


def load(path: Path = TABLE, local: Path | None = None) -> dict:
    t = read_raw(path, local)
    errs = validate(t, path.parent)
    if errs:
        raise TableError("; ".join(errs))
    return t


def _pins_ok(base: Path, rel: str, errs: list[str], name: str) -> None:
    p = base / rel
    if not p.is_file():
        errs.append(f"{name}: pins 파일이 없다: {rel}")
        return
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        errs.append(f"{name}: pins 가 비었다")
    for ln in lines:
        if not SHA_LINE_RE.match(ln) or ".." in ln.split("  ", 1)[1]:
            errs.append(f"{name}: pins 줄 형식이 아니다: {ln[:80]}")


def validate(t: dict, base: Path = HERE) -> list[str]:
    errs: list[str] = []
    if t.get("schema") != 1:
        errs.append("schema = 1 이 아니다")
    if set(t) - TOP_KEYS:
        errs.append(f"모르는 칸 {sorted(set(t) - TOP_KEYS)}")
    models = t.get("models") or {}
    if t.get("default") not in models:
        errs.append("default 가 models 에 없다")
    for name, m in models.items():
        if not ALIAS_RE.match(name):
            errs.append(f"별칭 형식이 아니다: {name}")
        bad = set(m) - MODEL_KEYS
        if bad:
            errs.append(f"{name}: 모르는 칸 {sorted(bad)}")
        src = m.get("source", "hf")
        if src not in SOURCES:
            errs.append(f"{name}: source 는 {SOURCES}")
        if src == "hf":
            if not REPO_RE.match(str(m.get("repo", ""))):
                errs.append(f"{name}: repo 형식")
            if not REV_RE.match(str(m.get("rev", ""))):
                errs.append(f"{name}: rev 는 커밋 sha 40자")
            if m.get("files"):
                errs.append(f"{name}: hf 별칭은 files 대신 pins")
        else:
            for k in ("repo", "rev", "pins", "fast"):
                if k in m:
                    errs.append(f"{name}: home 별칭에는 {k} 를 두지 않는다(files 만)")
            fl = m.get("files") or []
            if m.get("pending"):
                if fl:
                    errs.append(f"{name}: pending 별칭에는 files 를 두지 않는다(promote 뒤 pending 을 뺀다)")
            elif not fl:
                errs.append(f"{name}: home 별칭은 files 가 있어야 한다")
            seen = set()
            for f in fl:
                if set(f) != {"name", "sha256", "size"}:
                    errs.append(f"{name}.files: 칸은 name·sha256·size")
                    continue
                if not FILE_NAME_RE.match(str(f["name"])) or f["name"] in seen:
                    errs.append(f"{name}.files: 이름 형식·중복 {f['name']!r}")
                seen.add(f["name"])
                if not re.fullmatch(r"[0-9a-f]{64}", str(f["sha256"])):
                    errs.append(f"{name}.files.{f['name']}: sha256 64자")
                if not isinstance(f["size"], int) or isinstance(f["size"], bool) or f["size"] <= 0:
                    errs.append(f"{name}.files.{f['name']}: size 는 양의 정수")
            if fl and not {"config.json", "model.safetensors.index.json"} <= seen:
                errs.append(f"{name}.files: config.json·model.safetensors.index.json 이 있어야 한다")
        if not SAFE_RE.match(str(m.get("dir", ""))) or "/" in str(m.get("dir", "")) or not m.get("dir"):
            errs.append(f"{name}: dir 은 폴더 이름 하나")
        if not SAFE_RE.match(str(m.get("served_name", ""))) or not m.get("served_name"):
            errs.append(f"{name}: served_name")
        ad = m.get("adapters") or []
        if not ad or any(a not in ADAPTERS for a in ad):
            errs.append(f"{name}: adapters 는 {ADAPTERS} 중에서")
        if any(a != "hyperqwen-8cf642e" for a in ad):
            ps = m.get("patchset")
            if not ps or not (base / "patches" / str(ps) / "series").is_file():
                errs.append(f"{name}: patchset 폴더가 없다: {ps}")
            sv = m.get("serve") or {}
            for k, v in sv.items():
                if k not in SERVE_KEYS:
                    errs.append(f"{name}.serve: 모르는 칸 {k}")
                elif not isinstance(v, SERVE_KEYS[k]) or (SERVE_KEYS[k] is int and isinstance(v, bool)):
                    errs.append(f"{name}.serve.{k}: {SERVE_KEYS[k].__name__} 이어야 한다")
                elif isinstance(v, str) and not SAFE_RE.match(v):
                    errs.append(f"{name}.serve.{k}: 글자")
            for k in ("max_model_len", "gpu_memory_utilization", "max_num_seqs", "kv_cache_dtype"):
                if k not in sv:
                    errs.append(f"{name}.serve: {k} 가 없다")
        if "hyperqwen-8cf642e" in ad:
            hq = m.get("hyperqwen") or {}
            if hq.get("CTX") not in ("fast", "long") or hq.get("SPEC") not in ("mtp",):
                errs.append(f"{name}.hyperqwen: CTX fast|long, SPEC mtp")
        tp = m.get("tp", 1)
        if not isinstance(tp, int) or isinstance(tp, bool) or tp not in (1, 2, 4, 8):
            errs.append(f"{name}: tp 는 1·2·4·8")
        if tp > 1 and "hyperqwen-8cf642e" in ad:
            errs.append(f"{name}: hyperqwen 어댑터는 tp 1 만(single 보정이 한 장 기준)")
        if any(f not in CHAT_FIXES for f in m.get("chat_template_fix", [])):
            errs.append(f"{name}: chat_template_fix 는 {CHAT_FIXES} 중에서")
        if not isinstance(m.get("pending", False), bool) or (m.get("pending") and src != "home"):
            errs.append(f"{name}: pending 은 home 별칭의 true|false")
        # prepared = 이미 HyperQwen prepare(int8 head·draft)를 거친 세트(우리 변환의 int8h). hyperqwen 어댑터가
        # prepare·verify 를 다시 돌리지 않고(PREPARE=0 VERIFY=0 start_qwen.sh) 어느 차선에서나 띄운다
        if not isinstance(m.get("prepared", False), bool) or (m.get("prepared") and "hyperqwen-8cf642e" not in ad):
            errs.append(f"{name}: prepared 는 hyperqwen 어댑터가 있는 별칭의 true|false")
        # T35: mirror = home 세트의 HF 비공개 사본(파드가 읽기 전용 토큰으로 받는다 — 집 → 파드 테일넷이 느릴 때).
        # 받은 뒤 대조는 files(home 과 같은 세기). 저장소 안 자리는 <별칭>/
        if "mirror" in m:
            mr = m["mirror"] if isinstance(m["mirror"], dict) else {}
            if src != "home" or set(mr) != {"repo", "rev"} or not REPO_RE.match(str(mr.get("repo", ""))) \
                    or not REV_RE.match(str(mr.get("rev", ""))):
                errs.append(f"{name}: mirror 는 home 별칭의 {{repo, rev(커밋 40자)}}")
        _chat_template_ok(base, m.get("chat_template"), errs, name)
        _quant_src_ok(m.get("quant_src"), errs, name)
        if src == "hf":
            _pins_ok(base, str(m.get("pins", "")), errs, name)
        if m.get("fast"):
            f = m["fast"]
            if not REPO_RE.match(str(f.get("repo", ""))) or not REV_RE.match(str(f.get("rev", ""))):
                errs.append(f"{name}.fast: repo·rev")
            if any(not SAFE_RE.match(x) for x in f.get("files", [])) or not f.get("files"):
                errs.append(f"{name}.fast: files")
            _pins_ok(base, str(f.get("pins", "")), errs, name + ".fast")
    for iname, im in (t.get("images") or {}).items():
        ref = str(im.get("ref", ""))
        if not re.fullmatch(r"[a-z0-9./_-]+:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}", ref):
            errs.append(f"이미지 {iname}: ref 는 이름:태그@sha256:<64자>")
    hm = t.get("home") or {}
    if any((m.get("source", "hf") == "home" and not m.get("pending")) for m in models.values()):
        if not _ipv4_in_cgnat(str(hm.get("server_ip", ""))):
            errs.append("home.server_ip 가 테일넷 IPv4(100.64.0.0/10)가 아니다 — home 별칭을 쓰기 전에 채운다")
    for k in ("recv_port", "recv_timeout_min"):
        if k in hm and (not isinstance(hm[k], int) or isinstance(hm[k], bool) or hm[k] <= 0):
            errs.append(f"home.{k} 는 양의 정수")
    plans = t.get("plans") or {}
    for pn, pl in plans.items():
        if not ALIAS_RE.match(pn):
            errs.append(f"계획 이름 형식: {pn}")
        if pl.get("image") not in IMAGE_KINDS:
            errs.append(f"계획 {pn}: image 는 {IMAGE_KINDS}")
        pe = pl.get("env", {})
        if not isinstance(pe, dict) or any(k not in PLAN_ENV or str(v) not in PLAN_ENV[k] for k, v in pe.items()):
            allowed = ", ".join(k + "=" + "|".join(v) for k, v in PLAN_ENV.items())
            errs.append(f"계획 {pn}: env 는 {allowed} 만")
        if set(pl) - PLAN_KEYS:
            errs.append(f"계획 {pn}: 모르는 칸 {sorted(set(pl) - PLAN_KEYS)}")
        if not isinstance(pl.get("fill", False), bool):
            errs.append(f"계획 {pn}: fill 은 true|false")
        lanes = pl.get("lanes") or []
        if not lanes or len(lanes) > MAX_CARDS:
            errs.append(f"계획 {pn}: 차선 1~{MAX_CARDS}개")
        need = 0
        for i, ln in enumerate(lanes):
            m = models.get(ln.get("model"))
            if m is None:
                errs.append(f"계획 {pn} 차선 {i}: 표에 없는 모델 {ln.get('model')}")
                continue
            if m.get("pending"):
                errs.append(f"계획 {pn} 차선 {i}: 변환·promote 전(pending) 별칭 {ln.get('model')}")
            if "spec" in ln and ln["spec"] not in LANE_SPECS:
                errs.append(f"계획 {pn} 차선 {i}: spec 은 {LANE_SPECS}")
            if ln.get("adapter") not in (m.get("adapters") or []):
                errs.append(f"계획 {pn} 차선 {i}: 그 모델이 허락하지 않는 어댑터 {ln.get('adapter')}")
            elif pl.get("image") not in ADAPTER_IMAGES[ln["adapter"]]:
                errs.append(f"계획 {pn} 차선 {i}: 어댑터 {ln['adapter']} 는 이미지 {pl.get('image')} 에서 못 돈다")
            lm = models.get(ln.get("model")) or {}
            if ln.get("adapter") == "hyperqwen-8cf642e" and i != 0 and not lm.get("prepared") and (
                    lanes[0].get("adapter") != "hyperqwen-8cf642e" or lanes[0].get("model") != ln.get("model")):
                # prepare·verify 는 차선 0 만 돈다(k≥1 hyperqwen 은 PREPARE=0 VERIFY=0 start_qwen.sh — serve_cmd). 그래서
                # hyperqwen 차선이 있으면 차선 0 이 같은 모델의 hyperqwen 이어야 한다(T23 a-x4: 네 차선 모두 hyperqwen)
                errs.append(f"계획 {pn} 차선 {i}: hyperqwen 어댑터는 차선 0 만이거나, 차선 0 이 같은 모델의 hyperqwen 일 때만"
                            "(prepare·verify 를 차선 0 이 돈다 — prepared 별칭은 예외)")
            if set(ln) - LANE_KEYS:
                errs.append(f"계획 {pn} 차선 {i}: 모르는 칸 {sorted(set(ln) - LANE_KEYS)}")
            need += m.get("tp", 1)
        g = pl.get("gpus")
        if not isinstance(g, int) or isinstance(g, bool) or not 1 <= g <= MAX_CARDS or need != g:
            # T25: 카드 합 = gpus(남는 카드를 두지 않는다). fill 계획은 gpus 가 최소 장 수(나열한 차선의 카드 합)
            errs.append(f"계획 {pn}: gpus({g}) 가 차선 카드 합({need})과 같아야 한다(1~{MAX_CARDS})")
    qa = (t.get("quant") or {}).get("audit")
    if qa is not None and (not isinstance(qa, list) or len(qa) != 2 or any(
            a not in models or not (models[a].get("quant_src")) for a in qa)):
        errs.append("quant.audit 는 quant_src 가 있는 별칭 두 개 [원본, 편집본]")
    orders = [m["quant_src"].get("order") for m in models.values() if isinstance(m.get("quant_src"), dict)]
    if len(orders) != len(set(orders)):
        errs.append("quant_src.order 가 겹친다")
    return errs


def _chat_template_ok(base: Path, ct, errs: list[str], name: str) -> None:
    """chat_template = {file, sha256}: 서빙 폴더 기준 파일(templates/ 아래)이 있고 sha 가 같아야 한다."""
    if ct is None:
        return
    import hashlib
    if not isinstance(ct, dict) or set(ct) != {"file", "sha256"}:
        errs.append(f"{name}.chat_template: 칸은 file·sha256")
        return
    f, want = str(ct["file"]), str(ct["sha256"])
    if not re.fullmatch(r"templates/[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.jinja", f) or not re.fullmatch(r"[0-9a-f]{64}", want):
        errs.append(f"{name}.chat_template: file 은 templates/<이름>.jinja, sha256 64자")
        return
    p = base / f
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != want:
        errs.append(f"{name}.chat_template: {f} 가 없거나 sha256 이 다르다")


def _quant_src_ok(qs, errs: list[str], name: str) -> None:
    """quant_src = 변환 대상(deploy/quant). pins 파일은 deploy/quant 기준이라 여기서는 꼴만 본다(서빙 묶음에 없다)."""
    if qs is None:
        return
    if not isinstance(qs, dict) or set(qs) != QUANT_SRC_KEYS:
        errs.append(f"{name}.quant_src: 칸은 {sorted(QUANT_SRC_KEYS)}")
        return
    if not REPO_RE.match(str(qs["repo"])) or not REV_RE.match(str(qs["rev"])):
        errs.append(f"{name}.quant_src: repo·rev(40자)")
    if not re.fullmatch(r"pins/[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.sha256", str(qs["pins"])):
        errs.append(f"{name}.quant_src: pins 는 pins/<이름>.sha256")
    pt = qs["pad_to"]
    if not isinstance(pt, int) or isinstance(pt, bool) or (pt != 0 and (pt % 128 or not 1024 <= pt <= 65536)):
        errs.append(f"{name}.quant_src: pad_to 는 0(채우지 않음) 또는 128 의 배수")
    o = qs["order"]
    if not isinstance(o, int) or isinstance(o, bool) or not 1 <= o <= 99:
        errs.append(f"{name}.quant_src: order 는 1~99")


def quant_targets(t: dict) -> list[dict]:
    """변환 대상(order 차례): {alias, repo, rev, pins, pad_to}. 변환 묶음(make_quant_bundle)이 targets.json 으로 싣는다."""
    out = [{"alias": a, **{k: m["quant_src"][k] for k in ("repo", "rev", "pins", "pad_to")}, "order": m["quant_src"]["order"]}
           for a, m in (t.get("models") or {}).items() if m.get("quant_src")]
    return sorted(out, key=lambda x: x["order"])


def plan_lanes(t: dict, name: str, image_kind: str, gpus: int) -> list[dict]:
    """차선마다 {k, port, cards, alias, adapter, quality, env}. 카드는 앞 차선부터 tp 장씩, 포트는 80k0."""
    pl = (t.get("plans") or {}).get(name)
    if pl is None:
        raise TableError(f"표에 없는 계획: {name}(있는 것: {', '.join(t.get('plans') or {}) or '없음'})")
    if pl["image"] != image_kind:
        raise TableError(f"계획 {name} 은 이미지 {pl['image']} 용이다(지금 {image_kind})")
    rows = [dict(ln) for ln in pl["lanes"]]
    if not 1 <= gpus <= MAX_CARDS:
        raise TableError(f"카드 수 {gpus} 는 1~{MAX_CARDS}")
    if pl.get("fill"):
        # 남는 카드를 마지막 차선과 같은 모델·어댑터 차선(quality false)으로 채운다. tp 장씩 나눠떨어져야 한다
        last = rows[-1]
        tp = int(select_env(t, last["model"], last["adapter"], image_kind)["M_TP"])
        extra = gpus - pl["gpus"]
        if extra < 0 or extra % tp:
            raise TableError(f"계획 {name} 은 카드 {pl['gpus']} 장 이상, 마지막 차선 tp {tp} 장씩 채운다(지금 {gpus})")
        rows += [{**{k: v for k, v in last.items() if k != "quality"}, "quality": False}] * (extra // tp)
    elif gpus != pl["gpus"]:
        have = ", ".join(n + "=" + str(p["gpus"]) + ("+" if p.get("fill") else "") for n, p in t["plans"].items())
        raise TableError(f"계획 {name} 은 카드 {pl['gpus']} 장용이다(지금 {gpus}). 남는·모자라는 카드 없이 장 수에 맞는 계획을 고른다"
                         f"(있는 것: {have})")
    out, card = [], 0
    for k, ln in enumerate(rows):
        env = select_env(t, ln["model"], ln["adapter"], image_kind)
        if ln.get("spec"):                  # 차선 하나만 추측 디코딩을 바꾼다(serve_cmd·품질 점검이 읽는다)
            env |= {"M_LANE_SPEC": ln["spec"], "M_SPEC": ln["spec"]}
        tp = int(env["M_TP"])
        out.append({"k": k, "port": 8000 + 10 * k, "cards": ",".join(str(c) for c in range(card, card + tp)),
                    "alias": env["M_ALIAS"], "adapter": env["M_ADAPTER"], "quality": bool(ln.get("quality")), "env": env})
        card += tp
    return out


def plan_env(t: dict, name: str) -> dict[str, str]:
    """계획의 비교 조건(CTX·SPEC·PREFIX_CACHE·FAST_VARIANT 중 정한 것). 없으면 {}."""
    pl = (t.get("plans") or {}).get(name)
    if pl is None:
        raise TableError(f"표에 없는 계획: {name}")
    return {k: str(v) for k, v in sorted((pl.get("env") or {}).items())}


def model(t: dict, alias: str) -> dict:
    alias = t["default"] if alias in ("", "default") else alias
    if alias not in t["models"]:
        raise TableError(f"표에 없는 별칭: {alias}(있는 것: {', '.join(t['models'])})")
    if t["models"][alias].get("pending"):
        raise TableError(f"{alias} 은 아직 변환·promote 전이다(pending) — 다른 별칭을 고른다")
    return {"alias": alias, **t["models"][alias]}


def choose_adapter(m: dict, want: str, image_kind: str) -> str:
    if image_kind not in IMAGE_KINDS:
        raise TableError(f"모르는 이미지 종류: {image_kind}")
    if want in ("", "auto"):
        for a in m["adapters"]:
            if image_kind in ADAPTER_IMAGES[a]:
                return a
        raise TableError(f"{m['alias']}: 이 이미지({image_kind})에서 되는 어댑터가 없다({m['adapters']})")
    if want not in ADAPTERS:
        raise TableError(f"모르는 어댑터: {want}")
    if want not in m["adapters"]:
        raise TableError(f"{m['alias']}: 표가 허락하지 않는 어댑터 {want}({m['adapters']})")
    if image_kind not in ADAPTER_IMAGES[want]:
        raise TableError(f"어댑터 {want} 는 이미지 {image_kind} 에서 못 돈다(되는 이미지: {ADAPTER_IMAGES[want]})")
    return want


def select_env(t: dict, alias: str, want: str, image_kind: str) -> dict[str, str]:
    m = model(t, alias)
    a = choose_adapter(m, want, image_kind)
    home = m.get("source", "hf") == "home"
    env = {"M_ALIAS": m["alias"], "M_SOURCE": "home" if home else "hf", "M_REPO": "" if home else m["repo"],
           "M_REV": ("home-" + home_digest(m["files"])) if home else m["rev"], "M_DIR": m["dir"], "M_SERVED": m["served_name"],
           "M_ADAPTER": a, "M_IMAGE": image_kind, "M_TP": str(m.get("tp", 1)), "M_PATCHSET": str(m.get("patchset", "")),
           "M_PINS": "" if home else m["pins"], "M_CHAT_FIX": " ".join(m.get("chat_template_fix", [])),
           "M_PRECISION": str(m.get("backend_precision", "")),
           # T28: 추측 디코딩 방법(품질 점검이 SPEC=mtp 면 PPL 을 뺀다). hyperqwen 은 표 값, 실제로는 CF·계획의 SPEC 이 이긴다(pod-start)
           "M_SPEC": str(m["hyperqwen"]["SPEC"] if a == "hyperqwen-8cf642e" else (m.get("serve") or {}).get("spec_method", "")),
           "M_CHAT_TPL": str((m.get("chat_template") or {}).get("file", "")),
           "M_CHAT_TPL_SHA": str((m.get("chat_template") or {}).get("sha256", ""))}
    mr = m.get("mirror") or {}
    env |= {"M_MIRROR_REPO": str(mr.get("repo", "")), "M_MIRROR_REV": str(mr.get("rev", ""))}
    if a == "hyperqwen-8cf642e":
        env |= {"M_HQ_CTX": m["hyperqwen"]["CTX"], "M_HQ_SPEC": m["hyperqwen"]["SPEC"], "M_PREPARED": "1" if m.get("prepared") else "0"}
        if m.get("fast"):
            env |= {"M_FAST_REPO": m["fast"]["repo"], "M_FAST_REV": m["fast"]["rev"], "M_FAST_PINS": m["fast"]["pins"],
                    "M_FAST_FILES": " ".join(m["fast"]["files"])}
    for k, v in env.items():
        if not SAFE_RE.match(v.replace(" ", "")):
            raise TableError(f"env 값 글자: {k}")
    return env


def write_env(env: dict[str, str], out: Path) -> None:
    out.write_text("".join(f"{k}='{v}'\n" for k, v in env.items()), encoding="utf-8")


def read_env(p: Path) -> dict[str, str]:
    env = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        k, _, v = ln.partition("=")
        env[k] = v.strip("'")
    return env


def pins(t: dict, alias: str, fast: bool = False) -> str:
    m = model(t, alias)
    if m.get("source", "hf") == "home" and not fast:
        return "".join(f"{f['sha256']}  {f['name']}\n" for f in sorted(m["files"], key=lambda f: f["name"]))
    rel = m["fast"]["pins"] if fast else m["pins"]
    return "".join(ln + "\n" for ln in (HERE / rel).read_text(encoding="utf-8").splitlines()
                   if ln.strip() and not ln.startswith("#"))


def vllm_argv(m: dict, model_dir: str, port: int, host: str = "127.0.0.1", spec: bool = True) -> list[str]:
    """HyperQwen single-user/start_qwen.sh(SPEC=mtp CTX=long PREFIX_CACHE=1) 가 푸는 인자와 같은 순서·값."""
    s = m["serve"]
    a = ["serve", model_dir, "--served-model-name", m["served_name"], "--host", host, "--port", str(port),
         "--gpu-memory-utilization", str(s["gpu_memory_utilization"]), "--max-model-len", str(s["max_model_len"]),
         "--max-num-seqs", str(s["max_num_seqs"]), "--api-server-count", "1"]
    if s.get("language_model_only", True):
        a.append("--language-model-only")
    if s.get("attention_backend"):
        a += ["--attention-backend", s["attention_backend"]]
    a += ["--kv-cache-dtype", s["kv_cache_dtype"]]
    if s.get("mamba_ssm_cache_dtype"):
        a += ["--mamba-ssm-cache-dtype", s["mamba_ssm_cache_dtype"]]
    a += ["--async-scheduling", "--max-num-batched-tokens", str(s.get("max_num_batched_tokens", 2048))]
    if s.get("spec_method") and spec:
        a += ["--speculative-config", json.dumps({"method": s["spec_method"], "num_speculative_tokens": s["spec_tokens"],
                                                  "draft_sample_method": "probabilistic"}, separators=(",", ":"))]
    a += ["--compilation-config", json.dumps({"max_cudagraph_capture_size": s.get("cudagraph_max", 32),
                                              "custom_ops": ["+rms_norm", "+silu_and_mul"]}, separators=(",", ":")),
          "--reasoning-parser", s.get("reasoning_parser", "qwen3"), "--enable-prompt-tokens-details",
          "--enable-auto-tool-choice", "--tool-call-parser", s.get("tool_parser", "qwen3_coder")]
    if s.get("prefix_cache"):
        a += ["--prefix-cache-retention-interval", "None", "--enable-prefix-caching", "--mamba-cache-mode", "align"]
    tp = m.get("tp", 1)
    if tp > 1:
        a += ["--tensor-parallel-size", str(tp)]
    if s.get("sse_keep_alive") is not None:
        a += ["--sse-keep-alive-interval", str(s["sse_keep_alive"])]
    return a


VLLM_MAIN = "import sys; from vllm.entrypoints.cli.main import main; sys.argv = ['vllm'] + sys.argv[1:]; sys.exit(main())"


def serve_cmd(t: dict, env_file: Path, lane: int, port: int, base_env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """(argv, env). 부르는 쪽이 exec 한다. base_env 는 지금 프로세스의 환경."""
    e = read_env(env_file)
    m = model(t, e["M_ALIAS"])
    models_root = base_env.get("STEWART_MODELS", "/workspace/stewart/models")
    mdir = f"{models_root}/{m['dir']}"
    env = dict(base_env)
    # 지난 요청 글을 서버가 쥐지 않게 — Responses API 저장(GET /v1/responses/{id} 로 되읽힘)을 끈다(기본도 꺼짐, 드러내 둔다)
    env |= {"PORT": str(port), "HOST": "127.0.0.1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
            "VLLM_ENABLE_RESPONSES_API_STORE": "0"}
    a = e["M_ADAPTER"]
    lane_spec = e.get("M_LANE_SPEC", "")
    if a == "hyperqwen-8cf642e":
        fast = base_env.get("FAST_VARIANT", "1") != "0" and bool(m.get("fast"))   # fast 판이 없는 별칭은 늘 기본 판
        env |= {"MODEL": mdir + ("-fast" if fast else ""), "BASE_MODEL_DIR": mdir, "HF_REPO": m.get("repo", ""),
                "CTX": base_env.get("CTX", e["M_HQ_CTX"]), "SPEC": lane_spec or base_env.get("SPEC", e["M_HQ_SPEC"]),
                "DFLASH2": "0", "FAST_VARIANT": "1" if fast else "0"}
        if lane == 0 and not m.get("prepared"):
            return ["bash", "/app/docker/entrypoint.sh", "single"], env
        env |= {"PREPARE": "0", "VERIFY": "0"}
        return ["bash", "/app/single-user/start_qwen.sh"], env
    py = base_env.get("STEWART_VLLM_PY", "python3")
    if a == "vllm-patched":
        ov = base_env.get("STEWART_OVERLAY", "")
        if not ov or not Path(ov, "vllm").is_dir() or not Path(ov, "stewart-overlay.json").is_file():
            raise TableError(f"overlay 가 없다: {ov!r}(serving.sh 의 serving_prepare 가 먼저 만든다)")
        env["PYTHONPATH"] = ov + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    tp = m.get("tp", 1)
    env |= {"VLLM_USE_FLASHINFER_SAMPLER": "0", "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF") or
            ("expandable_segments:False" if tp > 1 else "expandable_segments:True")}
    return [py, "-c", VLLM_MAIN, *vllm_argv(m, mdir, port, spec=lane_spec != "off")], env


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if not a:
        print(__doc__)
        return 2
    cmd, rest = a[0], a[1:]
    try:
        t = load()
        if cmd == "validate":
            print(f"ok: 모델 {len(t['models'])}개, default {t['default']}")
        elif cmd == "names":
            print("\n".join(t["models"]))
        elif cmd == "select":
            alias, want, kind, out = rest
            env = select_env(t, alias, want, kind)
            write_env(env, Path(out))
            print(" ".join(f"{k}={v}" for k, v in env.items() if k in ("M_ALIAS", "M_ADAPTER", "M_IMAGE", "M_TP")))
        elif cmd == "need":
            m = model(t, rest[0])
            if m.get("source", "hf") != "home":
                raise TableError(f"{m['alias']} 은 home 별칭이 아니다")
            print(json.dumps({"alias": m["alias"], "digest": home_digest(m["files"]), "files": m["files"]}, ensure_ascii=False))
        elif cmd == "quant-targets":
            print(json.dumps({"targets": quant_targets(t), "audit": (t.get("quant") or {}).get("audit")}, ensure_ascii=False))
        elif cmd == "home":
            h = t.get("home") or {}
            print(h.get("server_ip", ""), h.get("recv_port", 8090), h.get("recv_timeout_min", 60))
        elif cmd == "pins":
            sys.stdout.write(pins(t, rest[0], fast="--fast" in rest))
        elif cmd == "argv":
            print(json.dumps(vllm_argv(model(t, rest[0]), rest[1], int(rest[2])), ensure_ascii=False))
        elif cmd == "plan":
            name, kind, gpus, outdir = rest[0], rest[1], int(rest[2]), Path(rest[3])
            outdir.mkdir(parents=True, exist_ok=True)
            lanes = plan_lanes(t, name, kind, gpus)
            write_env(plan_env(t, name), outdir / "plan.env")       # 계획의 비교 조건(pod-start 1c 가 읽는다)
            for ln in lanes:
                write_env(ln["env"], outdir / f"lane-{ln['k']}.env")
                print(ln["k"], ln["port"], ln["cards"], ln["alias"], ln["adapter"], int(ln["quality"]))
        elif cmd == "serve":
            env_file, lane, port = Path(rest[0]), int(rest[1]), int(rest[2])
            args, env = serve_cmd(t, env_file, lane, port, dict(os.environ))
            sys.stdout.flush()
            os.execvpe(args[0], args, env)
        else:
            print(f"모르는 명령: {cmd}", file=sys.stderr)
            return 2
    except (TableError, ValueError, KeyError, OSError) as e:
        print(f"serving: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
