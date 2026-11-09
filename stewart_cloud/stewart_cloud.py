"""stewart-cloud 본체 — Vast.ai 에서 stewart GPU 인스턴스를 빌리고(up)·보고(status)·지운다(down).

설치: PREFIX/stewart_cloud.py (root 소유 644, PREFIX 기본 /opt/stewart-cloud — 설치 절은 README)
부르는 곳: PREFIX/bin/stewart-cloud 만(env -i 뒤, /usr/bin/python3 -I -B). 직접 부르지 않는다.
  stewart-cloud offers [serve|eval|quant] [4090]      (offers = serve. 4090 은 eval 참고 보기만)
  stewart-cloud offers eval [4090] <장 수>              (eval 을 그 장 수로만, 예: offers eval 8. up 은 offer 의 장 수를 STEWART_GPUS 로)
  stewart-cloud offers eval2 … · up eval2 <offer_id> <max_dph> [계획]   (둘째 eval, eval 과 같은 조건·템플릿·계획)
  stewart-cloud up [serve|eval|quant] <offer_id> <max_dph>   (up <offer> <상한> = serve)
  stewart-cloud up eval <offer_id> <max_dph> <계획>           (계획 = serving models.toml plans 의 이름, env STEWART_PLAN)
  stewart-cloud status
  stewart-cloud down <instance_id>
  stewart-cloud logs <instance_id> [daemon]                   (컨테이너 또는 데몬 로그 끝 200줄, 비밀 꼴은 가림)
  stewart-cloud verify <묶음> <커밋 40자> <PS 64자>            (커밋에서 다시 만들어 묶음 폴더·PS 와 대조만. 키·망 없음)
  stewart-cloud publish <묶음> <커밋 40자> <PS 64자>           (verify + gist 올리기 + Vast 템플릿 만들기 + 지금 템플릿 전환)
  stewart-cloud templates                                       (기록한 템플릿을 API 로 읽어 기록과 같은지)
  묶음 = vN(serve·eval·eval2 공용 파드) 또는 quant-vN. 템플릿은 상태 파일 PREFIX/state/templates.json 이 정한다
  (config 의 template_hash_id 는 쓰지 않는다). up 은 그 템플릿을 API 로 읽어 PS·SB·Args 가 publish 기록과 같을 때만 빌린다.

역할: 역할마다 인스턴스 하나, 합쳐 최대 넷.
  serve  1장, 테일넷 이름 stewart-gpu, 이름표 stewart-serve-…(이름표 stewart-YYYYMMDD-HHMM 도 serve)
  eval   여러 장, 테일넷 이름 stewart-gpu-eval, 이름표 stewart-eval-…
  eval2  eval 과 같은 조건(config.roles.eval)의 둘째 인스턴스, 테일넷 이름 stewart-gpu-eval2(config.roles.eval2.ts_name), 이름표 stewart-eval2-…
         eval + eval2 의 켜진 값 합은 config.eval_total_max_dph(없으면 6.00) 이하. config 에 roles.eval2 가 없으면 eval2 만 78
  quant  A100 80GB(H100 은 값 상한 안이면) 1장 + RAM·CPU·디스크·망 하한, 테일넷 이름 stewart-quant, 이름표 stewart-quant-…,
         전용 템플릿(변환 묶음). HF 토큰은 래퍼·템플릿·env 어디에도 없다.
  만들 때 env 로 역할을 넘긴다(-e STEWART_ROLE=… -e STEWART_GPUS=N, 정해진 낱말·정수만). 파드가 이름을 표에서 고른다.

키:
- 로그인 키체인 generic password(서비스 stewart-cloud.vast-api, 계정 = 이 프로그램을 부른 사용자)를 /usr/bin/security … -w 로 읽는다.
  출력은 파이프로만 받는다.
- 키를 환경 변수·명령줄로 넘기지 않는다(같은 사용자 프로세스의 시작 환경은 KERN_PROCARGS2 로 읽힌다). 자식 프로세스는 security 하나뿐이다.
- HTTP 는 이 프로세스 안 urllib 로 하고 Authorization 헤더에만 키를 넣는다. 응답 원문·헤더는 출력하지 않는다(허용 목록 필드만).

키 권한(Vast 범주): misc(offer 찾기) + instance_read(목록) + instance_write(만들기·지우기). billing·machine·team 은 쓰지 않는다.
템플릿 찾기(GET /v0/template/, user_read)·만들기(POST, user_write)는 키체인 stewart-cloud.vast-template 이 있으면 그 키로, 없으면
stewart-cloud.vast-api 로 부른다. user_write 는 Vast 에서 API 키·계정 env 만들기까지 같은 범주라 따로 둔 키를 권한다(README).
GitHub gist 토큰(stewart-cloud.github-gist, fine-grained Gists read/write 만)은 publish 에서만, GitHub API Authorization 헤더에만 넣는다.
지출 상한은 선불 크레딧으로 둔다(자동 충전 끔). 이 프로그램은 잔액을 보지 않는다. 겹 방어는 역할별 값 상한(config.roles.*.max_dph,
코드 상한 ABS_MAX_DPH)과 역할마다 1개뿐이다.
정지(stop)는 하지 않는다 — 늘 Destroy. 파드의 유휴 자동 정지 뒤에도 디스크 요금이 나가므로 down 이 필요하다.

이름 겹침(Tailscale): down 은 Destroy 전에 파드의 POST :8001/logout 을 불러 테일넷에서 바로 빠지게 한다.
up 은 그 역할의 이름(stewart-gpu·stewart-gpu-eval)이 아직 풀리면 거부한다(이전 노드가 남아 있으면 새 노드가 stewart-gpu-1 이 된다).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from datetime import datetime, timezone

API = "https://console.vast.ai/api"   # 경로에 판(v0·v1)을 붙인다. 목록만 v1(v0 목록은 410 — 11/8 v12.10 실측), 나머지는 v0
PREFIX = "/opt/stewart-cloud"   # 설치 폴더(root 소유). 사람만 고친다 — 세션·에이전트가 쓸 수 없는 곳에 둔다


def _site(path: str = PREFIX + "/site.json") -> dict:
    """설치 폴더의 site.json(선택): 계정·홈·기록·저장소·묶음 폴더. 없으면 부른 사용자와 그 홈에서 정한다."""
    import pwd
    pw = pwd.getpwuid(os.getuid())
    s = {"user": pw.pw_name, "home": pw.pw_dir}
    try:
        with open(path, encoding="utf-8") as f:
            got = json.load(f)
    except FileNotFoundError:
        got = {}
    if not isinstance(got, dict):
        raise SystemExit(f"{path} 는 JSON 객체여야 한다")
    for k in ("user", "home", "record_dir", "repo_git", "bundle_dir"):
        if k in got:
            if not isinstance(got[k], str) or not got[k] or (k != "user" and not got[k].startswith("/")):
                raise SystemExit(f"{path} 의 {k} 는 절대 경로(user 는 이름) 글이어야 한다")
            s[k] = got[k]
    s.setdefault("record_dir", s["home"] + "/.cache/stewart-cloud")
    s.setdefault("repo_git", s["home"] + "/src/stewart-cloud/.git")
    s.setdefault("bundle_dir", s["home"] + "/.local/share/stewart-cloud/bundles")
    return s


SITE = _site()
SERVICE = "stewart-cloud.vast-api"
KC_ACCOUNT = SITE["user"]
HOME_DIR = SITE["home"]
KEYCHAIN = HOME_DIR + "/Library/Keychains/login.keychain-db"
CONFIG = PREFIX + "/config.json"
RECORD_DIR = SITE["record_dir"]
GUIDE = "README.md 의 '5. 키'"
PROG = "stewart-cloud"

# T27 publish(아래 '묶음 올리기' 절)
REPO_GIT = SITE["repo_git"]          # 묶음을 다시 만들 저장소(이 저장소)의 .git
LABS_VAST = SITE["bundle_dir"]       # 빌더가 낸 묶음 폴더(<묶음>/args.txt …)
STATE_DIR = PREFIX + "/state"        # 사용자 700, 부모 root — 세션·에이전트는 쓸 수 없게 둔다
STATE_FILE = "templates.json"
BUILD_PY = PREFIX + "/py313/bin/python3.13"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
GH_API = "https://api.github.com"
GH_SERVICE = "stewart-cloud.github-gist"        # fine-grained, Account permissions Gists read/write 만
TPL_SERVICE = "stewart-cloud.vast-template"   # 있으면 템플릿 읽기·만들기에 쓴다(user_read·user_write). 없으면 SERVICE
KINDS = ("pod", "quant")
BUNDLE_OUT = {   # 종류 → (빌더, 묶음 파일, gist 파일 이름 앞, 빈칸 → env 이름)
    "pod": ("deploy/vast/make_template_env.py", "pod-files.b64", "stewart-pod-",
            {"<gist raw 주소>": "PU", "<설정 gist 최신 raw 주소>": "CF"}),
    "quant": ("deploy/quant/make_quant_bundle.py", "quant-files.b64", "stewart-quant-", {"<gist raw 주소>": "PU"}),
}
BUILD_FILES = ("args.txt", "boot.sb.txt", "docker-options.txt")   # + 묶음 파일. 넷 다 빌더가 내고 labs 에도 같아야 한다
_BUNDLE = re.compile(r"^(quant-)?v[1-9][0-9]{0,2}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THASH = re.compile(r"^[0-9a-f]{16,64}$")
_HOLE = re.compile(r"<[^<>\s][^<>]*>")
_GIST_RAW = re.compile(r"^https://gist\.githubusercontent\.com/([A-Za-z0-9-]{1,39})/([0-9a-f]{20,40})/raw/(?:[0-9a-f]{40}/)?[A-Za-z0-9._-]{1,100}$")
TPL_COPY = ("image", "tag", "runtype", "recommended_disk_space", "onstart", "ssh_direct", "jup_direct", "use_ssh",
            "use_jupyter_lab", "jupyter_dir", "extra_filters")
TPL_SAME = ("runtype", "recommended_disk_space")    # 만든 뒤 다시 읽어 바탕과 같아야 하는 칸(이미지는 image_ref 로 — tag 는 Vast 가 채울 수 있다)
GIT_MAX_FILES, GIT_MAX_BYTES = 3000, 64_000_000
BUILD_TIMEOUT = 180
OPTIONS_MAX = 4096
B64_MAX = 900_000                 # gist API 가 내용을 자르지 않는 크기 안(1 MB)
BUILD_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny mach-lookup)
(deny process-fork)
(deny file-read* (subpath "/Volumes") (subpath "/Users") (subpath "/private/tmp") (subpath "/private/var/folders"))
(allow file-read* (subpath (param "TMP")))
(deny file-write*)
(allow file-write* (subpath (param "OUT")) (literal "/dev/null"))
"""

EXIT_USAGE, EXIT_KEY, EXIT_AUTH, EXIT_NET, EXIT_REFUSE, EXIT_OTHER = 64, 78, 77, 75, 69, 70

CONFIG_KEYS = {"gpu_name", "cuda_max_good_min", "reliability_min", "datacenter", "verified",
               "label_prefix", "logout_port", "band_dph", "offers_shown", "roles"}
CONFIG_OPT_KEYS = {"eval_total_max_dph", "template_hash_id"}   # T26: eval + eval2 켜진 값 합 상한(없으면 EVAL_TOTAL_DEFAULT).
# template_hash_id(·roles.quant.template_hash_id)는 T27 부터 쓰지 않는다 — 템플릿은 상태 파일(STATE_DIR/templates.json)이 정한다. 옛 설정을 받으려고 남김
EVAL_TOTAL_DEFAULT = 6.0
ROLES = ("serve", "eval", "eval2", "quant")
EVAL_LIKE = ("eval", "eval2")                      # 같은 조건·계획·템플릿(eval2 는 이름만 다르다 — pod 1a 가 ROLE=eval 로 돈다)
OPT_ROLES = {"eval2"}                              # config.roles 에 없어도 된다(없으면 그 역할만 78)
ROLE_KEYS = {"num_gpus_min", "num_gpus_max", "max_dph", "disk", "ts_name"}
QUANT_KEYS = {"gpu_names", "gpu_ram_gb_min", "cpu_ram_gb_min", "cpu_cores_min", "inet_down_min"}
ROLE_KEYS_BY = {"serve": ROLE_KEYS, "eval": ROLE_KEYS, "eval2": {"ts_name"}, "quant": ROLE_KEYS | QUANT_KEYS}
# 있어도 없어도 되는 칸(T25). eval 의 cpu_ram_gb_per_gpu_min: offer 의 RAM(MB) ≥ 이 값 × 장 수 × 1000. 없으면 RAM 조건 없음(T24 까지와 같다)
ROLE_OPT_KEYS = {"serve": set(), "eval": {"cpu_ram_gb_per_gpu_min"}, "eval2": set(), "quant": {"template_hash_id", "datacenter"}}
# T28: roles.quant.datacenter(참·거짓, 없으면 맨 위 datacenter). false 면 quant 검색·판정에서만 datacenter 조건을 뺀다
# (verified·reliability 는 그대로). 변환은 공개 모델만, GPU→집은 ACL 로 막힘, 가중치 바꿔치기는 KL 검사가 잡는다(메인 T28).
ROLE_NAMES = {"serve": "stewart-gpu", "eval": "stewart-gpu-eval", "eval2": "stewart-gpu-eval2",
              "quant": "stewart-quant"}   # 테일넷 이름 앞부분(pod 1a 의 표와 같다)
ABS_MAX_DPH = 8.0          # 설정이 틀려도 넘지 못하는 값 상한($/hr, 인스턴스 하나)
MAX_GPUS = 8
MAX_TOTAL = 4              # 역할마다 1개, 합쳐 4개(T22 3 → T26 eval2 로 4)
REF_GPUS = {"4090": "RTX 4090"}   # offers eval 4090 — 참고 보기만(up 은 config.gpu_name 만)
OFFER_FIELDS = ("id", "machine_id", "dph_total", "dph_per_gpu", "num_gpus", "gpu_name", "gpu_ram", "reliability", "cuda_max_good",
                "geolocation", "inet_down", "cpu_cores_effective", "cpu_ram", "disk_space",
                "verification", "hosting_type")   # verified·datacenter 칸은 응답에 없다(문서)
INSTANCE_FIELDS = ("id", "machine_id", "label", "actual_status", "cur_state", "intended_status", "dph_total",
                   "start_date", "gpu_name")
_ID = re.compile(r"^[1-9][0-9]{0,11}$")
_NGPU = re.compile(r"^[1-8]$")
_PLAN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,47}$")   # serving models.toml 의 계획 이름 꼴(pod-start 1b 와 같다)
_DPH = re.compile(r"^[0-9]\.[0-9]{1,3}$")
_KEYLIKE = re.compile(r"[A-Za-z0-9_\-]{40,}")


class Stop(Exception):
    def __init__(self, msg: str, code: int):
        super().__init__(msg)
        self.code = code


def say(msg: str) -> None:
    sys.stdout.write(f"{PROG}: {msg}\n")
    sys.stdout.flush()


def emit(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def scrub(s: str) -> str:
    return _KEYLIKE.sub("[가림]", s)


# ── 설정·인자 ────────────────────────────────────────────────────

def load_config(path: str = CONFIG) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
    except (OSError, ValueError) as e:
        raise Stop(f"설정 파일을 읽을 수 없다: {path} ({type(e).__name__})", EXIT_OTHER)
    if not isinstance(c, dict) or not CONFIG_KEYS <= set(c) <= CONFIG_KEYS | CONFIG_OPT_KEYS:
        got = set(c) if isinstance(c, dict) else set()
        raise Stop(f"설정 키가 다르다: 빠짐 {sorted(CONFIG_KEYS - got)}, 모름 {sorted(got - CONFIG_KEYS - CONFIG_OPT_KEYS)}", EXIT_OTHER)
    tot = c.get("eval_total_max_dph", EVAL_TOTAL_DEFAULT)
    if isinstance(tot, bool) or not isinstance(tot, (int, float)) or not 0 < tot <= 2 * ABS_MAX_DPH:
        raise Stop(f"config.eval_total_max_dph 는 0 초과 {2 * ABS_MAX_DPH} 이하", EXIT_OTHER)
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,20}-", c["label_prefix"]):
        raise Stop("config.label_prefix 형식이 다르다(예: stewart-)", EXIT_OTHER)
    roles = c["roles"]
    need = set(ROLES) - OPT_ROLES
    if not isinstance(roles, dict) or not need <= set(roles) <= set(ROLES):
        raise Stop(f"config.roles 는 {sorted(need)} 칸(+ 있어도 되는 {sorted(OPT_ROLES)})이어야 한다", EXIT_OTHER)
    names = set()
    for r in [x for x in ROLES if x in roles]:
        rc = roles[r]
        want, opt = ROLE_KEYS_BY[r], ROLE_OPT_KEYS[r]
        if not isinstance(rc, dict) or not want <= set(rc) <= want | opt:
            got = set(rc) if isinstance(rc, dict) else set()
            raise Stop(f"config.roles.{r} 키가 다르다: 빠짐 {sorted(want - got)}, 모름 {sorted(got - want - opt)}", EXIT_OTHER)
        if r == "eval2":     # ts_name 만 — 장 수·값 상한·디스크·RAM 은 roles.eval 것을 쓴다(아래 끝에서 합친다)
            if not re.fullmatch(re.escape(ROLE_NAMES[r]) + r"\.[a-z0-9-]{1,63}\.ts\.net", str(rc["ts_name"])):
                raise Stop(f"config.roles.{r}.ts_name 은 {ROLE_NAMES[r]}.<테일넷>.ts.net 꼴", EXIT_OTHER)
            names.add(rc["ts_name"])
            continue
        if "cpu_ram_gb_per_gpu_min" in rc and not (isinstance(rc["cpu_ram_gb_per_gpu_min"], int)
                                                   and not isinstance(rc["cpu_ram_gb_per_gpu_min"], bool)
                                                   and 1 <= rc["cpu_ram_gb_per_gpu_min"] <= 256):
            raise Stop(f"config.roles.{r}.cpu_ram_gb_per_gpu_min 은 1~256 정수", EXIT_OTHER)
        lo, hi = rc["num_gpus_min"], rc["num_gpus_max"]
        if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi <= MAX_GPUS):
            raise Stop(f"config.roles.{r}.num_gpus_min·max 는 1 ≤ min ≤ max ≤ {MAX_GPUS} 정수", EXIT_OTHER)
        if r in ("serve", "quant") and hi != 1:
            raise Stop(f"config.roles.{r} 는 1장만(num_gpus_max 1)", EXIT_OTHER)
        if r == "quant":
            if "datacenter" in rc and not isinstance(rc["datacenter"], bool):
                raise Stop("config.roles.quant.datacenter 는 true·false", EXIT_OTHER)
            if not (isinstance(rc["gpu_names"], list) and 1 <= len(rc["gpu_names"]) <= 12
                    and all(isinstance(g, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{1,31}", g) for g in rc["gpu_names"])):
                raise Stop("config.roles.quant.gpu_names 는 GPU 이름 1~12개", EXIT_OTHER)
            for key, lo_, hi_ in (("gpu_ram_gb_min", 16, 200), ("cpu_ram_gb_min", 8, 4096), ("cpu_cores_min", 1, 512),
                                  ("inet_down_min", 10, 100000)):
                if not (isinstance(rc[key], int) and not isinstance(rc[key], bool) and lo_ <= rc[key] <= hi_):
                    raise Stop(f"config.roles.quant.{key} 는 {lo_}~{hi_} 정수", EXIT_OTHER)
        if not (0 < float(rc["max_dph"]) <= ABS_MAX_DPH):
            raise Stop(f"config.roles.{r}.max_dph 는 0 초과 {ABS_MAX_DPH} 이하", EXIT_OTHER)
        if not (isinstance(rc["disk"], int) and 20 <= rc["disk"] <= 500):
            raise Stop(f"config.roles.{r}.disk 는 20~500(GB) 정수", EXIT_OTHER)
        if not re.fullmatch(re.escape(ROLE_NAMES[r]) + r"\.[a-z0-9-]{1,63}\.ts\.net", str(rc["ts_name"])):
            raise Stop(f"config.roles.{r}.ts_name 은 {ROLE_NAMES[r]}.<테일넷>.ts.net 꼴", EXIT_OTHER)
        names.add(rc["ts_name"])
    if len(names) != len([x for x in ROLES if x in roles]):
        raise Stop("config.roles 의 ts_name 이 겹친다", EXIT_OTHER)
    if "eval2" in roles:
        roles["eval2"] = {**roles["eval"], "ts_name": roles["eval2"]["ts_name"]}   # 메모리에서만 합친다(파일은 그대로)
    return c


def role_cfg(cfg: dict, role: str) -> dict:
    rc = (cfg.get("roles") or {}).get(role)
    if rc is None:
        raise Stop(f"config.roles.{role} 가 없다 — {role} 는 쓸 수 없다(README 의 eval2)", EXIT_KEY)
    return rc


def roles_on(cfg: dict) -> list:
    """설정에 있는 역할(eval2 는 있을 때만)."""
    return [r for r in ROLES if r in (cfg.get("roles") or {})]


def eval_total(cfg: dict) -> float:
    return float(cfg.get("eval_total_max_dph", EVAL_TOTAL_DEFAULT))


def role_label(cfg: dict, role: str) -> str:
    return f"{cfg['label_prefix']}{role}-"


def role_of(cfg: dict, label) -> str | None:
    """이름표로 역할을 안다. 옛 이름표 stewart-YYYYMMDD-HHMM(역할 나누기 전)은 serve. 그 밖의 stewart- 이름표는 None."""
    lab = str(label or "")
    for r in ROLES:
        if lab.startswith(role_label(cfg, r)):
            return r
    if re.fullmatch(re.escape(cfg["label_prefix"]) + r"[0-9]{8}-[0-9]{4}", lab):
        return "serve"
    return None


def parse_args(argv: list, cfg: dict | None = None) -> tuple:
    """(명령, 인자들). 형태가 다르면 Stop(64). cfg 가 있으면 up 의 max_dph 를 그 역할의 상한과 견준다."""
    usage = ("형태는 offers [serve|eval|eval2|quant] [4090] | offers eval|eval2 [4090] <장 수> | status | "
             "up [serve|eval|eval2|quant] <offer_id> <max_dph> | up eval|eval2 <offer_id> <max_dph> <계획> | down <instance_id> | "
             "logs <instance_id> [daemon] | verify|publish <묶음 vN|quant-vN> <커밋 40자> <PS 64자> | templates 뿐이다")
    if not argv:
        raise Stop(usage, EXIT_USAGE)
    cmd, rest = argv[0], argv[1:]
    if cmd in ("status", "templates"):
        if rest:
            raise Stop(usage, EXIT_USAGE)
        return cmd, ()
    if cmd in ("verify", "publish"):           # T27: 승인 창에 묶음·커밋·PS 가 그대로 보이게 셋 다 인자로 받는다
        if len(rest) != 3 or not _BUNDLE.match(rest[0]) or not _SHA1.match(rest[1]) or not _SHA256.match(rest[2]):
            raise Stop(usage + " (묶음 v12·quant-v4 꼴, 커밋은 40자 소문자 hex, PS 는 64자 소문자 hex)", EXIT_USAGE)
        return cmd, tuple(rest)
    if cmd == "offers":
        if rest == []:
            return cmd, ("serve", None, None)         # 옛 형태
        if len(rest) == 1 and rest[0] in ROLES:
            return cmd, (rest[0], None, None)
        if rest and rest[0] in EVAL_LIKE and len(rest) <= 3:
            ref, n = None, None
            tail = rest[1:]
            if tail and tail[0] in REF_GPUS:          # offers eval 4090 [장 수] — 참고 보기
                ref, tail = REF_GPUS[tail[0]], tail[1:]
            if len(tail) == 1 and _NGPU.match(tail[0]):   # offers eval [4090] <장 수>(T25)
                n, tail = int(tail[0]), []
                lo, hi = (role_cfg(cfg, "eval")["num_gpus_min"], role_cfg(cfg, "eval")["num_gpus_max"]) if cfg else (1, MAX_GPUS)
                if not lo <= n <= hi:
                    raise Stop(f"eval 장 수 {n} 은 설정 범위 {lo}~{hi} 밖이다", EXIT_USAGE)
            if not tail and (ref or n):
                return cmd, (rest[0], ref, n)
        raise Stop(usage, EXIT_USAGE)
    if cmd == "up":
        role, plan = "serve", None                    # 옛 형태 up <offer> <상한> = serve(결정 2)
        if len(rest) == 4 and rest[0] in EVAL_LIKE:   # 계획은 eval·eval2 만(T23·T26)
            if not _PLAN.match(rest[3]):
                raise Stop(usage + " (계획 이름은 소문자·숫자·.·- 48자까지)", EXIT_USAGE)
            role, plan, rest = rest[0], rest[3], rest[1:3]
        elif len(rest) == 3 and rest[0] in ROLES:
            role, rest = rest[0], rest[1:]
        if len(rest) != 2 or not _ID.match(rest[0]) or not _DPH.match(rest[1]):
            raise Stop(usage + " (offer_id 는 숫자, max_dph 는 x.xxx)", EXIT_USAGE)
        v = float(rest[1])
        lim = float(role_cfg(cfg, role)["max_dph"]) if cfg is not None else ABS_MAX_DPH
        if v <= 0 or v > lim:
            raise Stop(f"max_dph {rest[1]} 이 {role} 설정 상한 {lim} 를 넘거나 0 이다", EXIT_USAGE)
        return cmd, (role, int(rest[0]), v, plan)
    if cmd == "down":
        if len(rest) != 1 or not _ID.match(rest[0]):
            raise Stop(usage + " (instance_id 는 숫자)", EXIT_USAGE)
        return cmd, (int(rest[0]),)
    if cmd == "logs":
        if len(rest) not in (1, 2) or not _ID.match(rest[0]) or (len(rest) == 2 and rest[1] != "daemon"):
            raise Stop(usage + " (logs <instance_id> [daemon])", EXIT_USAGE)
        return cmd, (int(rest[0]), len(rest) == 2)
    raise Stop(usage, EXIT_USAGE)


# ── 키 ───────────────────────────────────────────────────────────

# 키체인이 잠겨 있을 때의 security 종료 코드(OSStatus 아래 바이트): 36 = errSecInteractionNotAllowed(-25308, SSH 셸 등
# GUI 로그인의 풀림이 넘어오지 않는 곳), 51 = errSecAuthFailed(-25293). 키를 다른 곳에 두지 않는다 — 풀고 다시 부른다.
KC_LOCKED = {36, 51}
UNLOCK_HINT = ("SSH 셸이면 먼저: security unlock-keychain ~/Library/Keychains/login.keychain-db "
               "(이 Mac 의 로그인 비밀번호 입력, 화면에 안 보임) 뒤 같은 명령을 다시 친다")


def read_secret(service: str, pattern: str, what: str, missing_ok: bool = False):
    """로그인 키체인 generic password(서비스 service, 계정 KC_ACCOUNT)를 security -w 로 읽는다. 출력은 파이프로만.
    missing_ok 면 항목이 없을 때 None(템플릿 키처럼 없어도 되는 것)."""
    if not os.access(KEYCHAIN, os.R_OK):
        raise Stop("로그인 키체인 파일을 읽을 수 없다 — 샌드박스 안에서 불린 것이다. 정해진 형태 그대로"
                   "(리다이렉트·파이프·앞붙인 환경변수 없이) 다시 부른다", EXIT_KEY)
    try:
        r = subprocess.run(["/usr/bin/security", "find-generic-password", "-s", service, "-a", KC_ACCOUNT, "-w", KEYCHAIN],
                           env={"PATH": "/usr/bin:/bin", "HOME": HOME_DIR}, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Stop(f"security 를 부를 수 없다({type(e).__name__})", EXIT_KEY)
    if r.returncode == 44:
        if missing_ok:
            return None
        raise Stop(f"키체인에 {what} 가 없다(서비스 {service}, 계정 {KC_ACCOUNT}). 사람이 이 Mac 에서 등록한다: {GUIDE}", EXIT_KEY)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip()
        locked = r.returncode in KC_LOCKED or "interaction is not allowed" in err.lower()
        raise Stop(f"키체인에서 {what} 를 읽지 못했다(security exit {r.returncode}: {scrub(err[:200])}). "
                   + (UNLOCK_HINT if locked else "잠겨 있으면 이 Mac 에 로그인해 푼다"), EXIT_KEY)
    key = r.stdout.decode("utf-8", "replace").rstrip("\r\n")
    if not re.fullmatch(pattern, key):
        raise Stop(f"키체인 값({service})이 {what} 형식이 아니다. 다시 등록한다: {GUIDE}", EXIT_KEY)
    return key


def read_key() -> str:
    return read_secret(SERVICE, r"[A-Za-z0-9_\-]{20,256}", "Vast API 키")


def read_template_key():
    """템플릿 읽기·만들기용 Vast 키(user_read·user_write). 키체인에 없으면 None → SERVICE 를 쓴다."""
    return read_secret(TPL_SERVICE, r"[A-Za-z0-9_\-]{20,256}", "Vast 템플릿 키", missing_ok=True)


# ── HTTP(이 프로세스 안) ─────────────────────────────────────────

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 환경의 프록시 설정을 쓰지 않는다


_PATHS = {   # (방법, 경로 꼴) 일곱만
    ("POST", re.compile(r"/v0/bundles/")),                      # offer 찾기
    ("PUT", re.compile(r"/v0/asks/[1-9][0-9]{0,11}/")),          # 만들기
    ("DELETE", re.compile(r"/v0/instances/[1-9][0-9]{0,11}/")),  # 지우기
    ("GET", re.compile(r"/v1/instances/")),                     # 목록(페이지 나눔)
    ("PUT", re.compile(r"/v0/instances/request_logs/[1-9][0-9]{0,11}/")),   # 로그 요청(T23, vast-cli logs 와 같다)
    ("GET", re.compile(r"/v0/template/")),                      # 템플릿 찾기(hash_id 로, T27 — user_read)
    ("POST", re.compile(r"/v0/template/")),                     # 템플릿 만들기(publish 만, T27 — user_write)
}
_QUERY_OK = {"/v1/instances/": {"limit", "order_by", "select_filters", "after_token"}, "/v0/template/": {"select_filters"}}
_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{1,2048}")


def vast(key: str, method: str, path: str, body: dict | None = None, timeout: float = 30, query: dict | None = None) -> dict:
    """Vast API 를 부른다. 방법·경로는 _PATHS 만, 쿼리는 v1 목록·템플릿 찾기에만(값 꼴도 본다)."""
    if not any(m == method and rx.fullmatch(path) for m, rx in _PATHS):
        raise Stop(f"허용하지 않은 경로: {method} {path}", EXIT_OTHER)
    url = API + path
    if query:
        if path not in _QUERY_OK or set(query) - _QUERY_OK[path]:
            raise Stop(f"허용하지 않은 쿼리: {sorted(query)}", EXIT_OTHER)
        if "after_token" in query and not _TOKEN.fullmatch(str(query["after_token"])):
            raise Stop("Vast 가 준 페이지 표지(next_token) 꼴이 이상하다", EXIT_OTHER)
        url += "?" + "&".join(f"{k}={urllib.parse.quote_plus(v if isinstance(v, str) else json.dumps(v))}"
                              for k, v in query.items())
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Accept": "application/json",
                                          "Content-Type": "application/json", "User-Agent": "stewart-cloud/1"})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            raw = r.read(8_000_000)
    except urllib.error.HTTPError as e:
        msg = ""
        try:
            j = json.loads(e.read(200_000) or b"{}")
            msg = str(j.get("msg") or j.get("error") or "")[:300]
        except Exception:
            pass
        code = EXIT_AUTH if e.code in (401, 403) else EXIT_OTHER
        raise Stop(f"Vast {method} {re.sub(r'/[0-9]+/', '/<id>/', path)} → HTTP {e.code} {scrub(msg)}"
                   + (" (키 권한·만료를 본다. 템플릿 쓰기에 user_read 가 필요하면 그것만 더한다)" if e.code == 403 else ""), code)
    except (urllib.error.URLError, OSError) as e:
        raise Stop(f"Vast 에 연결하지 못했다({type(e).__name__})", EXIT_NET)
    try:
        return json.loads(raw)
    except ValueError:
        raise Stop("Vast 응답이 JSON 이 아니다", EXIT_OTHER)


# ── 이름(Tailscale) ─────────────────────────────────────────────

def name_present(name: str) -> bool:
    """MagicDNS 이름이 풀리는지(테일넷에 그 이름의 기기 기록이 있는지)."""
    try:
        socket.getaddrinfo(name, None)
        return True
    except (socket.gaierror, UnicodeError, OSError):
        return False


_START_LINE = re.compile(r"\bstart instance=([0-9]+)\b")


def pod_status(name: str, port: int, timeout: float = 10) -> dict | None:
    """이름을 쥔 pod 의 /status(읽기 전용 JSON). 못 읽으면 None."""
    req = urllib.request.Request(f"http://{name}:{port}/status", headers={"User-Agent": "stewart-cloud/1"})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            st = json.loads(r.read(2_000_000) or b"{}")
    except Exception:
        return None
    return st if isinstance(st, dict) else None


def pod_instance(name: str, port: int, timeout: float = 10) -> str | None:
    """이름을 쥔 pod 가 어느 인스턴스인지. /status 의 instance_id(v6)를 먼저 보고, 없으면 pod_log 의 "start instance=N" 줄.
    못 읽거나 형식이 다르면 None."""
    st = pod_status(name, port, timeout)
    if st is None:
        return None
    iid = st.get("instance_id")
    if iid is not None:
        return str(iid) if re.fullmatch(r"[0-9]{1,12}", str(iid)) else None
    for line in reversed(st.get("pod_log") or []):
        m = _START_LINE.search(str(line))
        if m:
            return m.group(1)
    return None


def pod_logout(name: str, port: int, timeout: float = 20) -> str:
    req = urllib.request.Request(f"http://{name}:{port}/logout", data=b"", method="POST",
                                 headers={"User-Agent": "stewart-cloud/1"})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            j = json.loads(r.read(10_000) or b"{}")
        return str(j.get("result", "ok"))[:80]
    except Exception as e:  # pod 이 없거나 이미 빠짐 — Destroy 는 그래도 한다
        return f"실패({type(e).__name__})"


# ── 동작 ─────────────────────────────────────────────────────────

def gpu_range(rc: dict, n: int | None = None) -> tuple:
    """장 수 범위(lo, hi). n 이 오면(offers eval 8 — T25) 그 장 수만."""
    return (n, n) if n else (int(rc["num_gpus_min"]), int(rc["num_gpus_max"]))


def num_gpus_query(rc: dict, n: int | None = None) -> dict:
    lo, hi = gpu_range(rc, n)
    return {"eq": lo} if lo == hi else {"gte": lo, "lte": hi}


def offer_query(cfg: dict, role: str = "serve", gpu_name: str | None = None, n: int | None = None) -> dict:
    """offers 와 up 이 같이 쓰는 검색식 하나(up 도 이 결과에서 id 로 고른다). 역할마다 장 수·값 상한만 다르다.
    eval(T25): 디스크(disk_space ≥ 요청 disk)와, 설정에 있으면 RAM(장당 하한 × 가장 적은 장 수)을 서버 검색에도 건다."""
    rc = role_cfg(cfg, role)
    names = gpu_names(cfg, role, gpu_name)
    q = {"limit": 64, "type": "ondemand", "rentable": {"eq": True}, "rented": {"eq": False},
         "verified": {"eq": bool(cfg["verified"])},
         "gpu_name": {"in": sorted({x for g in names for x in (g, g.replace(" ", "_"))})},  # 문서 예시는 밑줄, 응답은 빈칸
         "num_gpus": num_gpus_query(rc, n),
         "cuda_max_good": {"gte": float(cfg["cuda_max_good_min"])}, "reliability": {"gte": float(cfg["reliability_min"])},
         "dph_total": {"lte": float(rc["max_dph"])}, "order": [["dph_total", "asc"]]}
    if want_datacenter(cfg, role):
        q["datacenter"] = {"eq": True}
    if role == "quant":   # 변환 기계 조건. gpu_ram·cpu_ram 은 MB(vast-cli offers_mult 1000), disk_space GB, inet_down Mb/s
        q |= {"gpu_ram": {"gte": int(rc["gpu_ram_gb_min"]) * 1000}, "cpu_ram": {"gte": int(rc["cpu_ram_gb_min"]) * 1000},
              "cpu_cores_effective": {"gte": int(rc["cpu_cores_min"])}, "disk_space": {"gte": int(rc["disk"])},
              "inet_down": {"gte": int(rc["inet_down_min"])}}
    if role in EVAL_LIKE:
        q["disk_space"] = {"gte": int(rc["disk"])}
        if rc.get("cpu_ram_gb_per_gpu_min"):
            q["cpu_ram"] = {"gte": int(rc["cpu_ram_gb_per_gpu_min"]) * gpu_range(rc, n)[0] * 1000}
    return q


def want_datacenter(cfg: dict, role: str) -> bool:
    """datacenter 조건을 걸지(T28: quant 만 roles.quant.datacenter 로 끌 수 있다)."""
    rc = (cfg.get("roles") or {}).get(role) or {}
    return bool(rc["datacenter"]) if role == "quant" and "datacenter" in rc else bool(cfg["datacenter"])


def gpu_names(cfg: dict, role: str, gpu_name: str | None = None) -> list:
    if gpu_name:
        return [gpu_name]
    return list(role_cfg(cfg, role).get("gpu_names") or [cfg["gpu_name"]])


def per_gpu(o: dict) -> float:
    n = int(o.get("num_gpus") or 1) or 1
    return float(o.get("dph_total") or 99) / n


def offer_fails(cfg: dict, o: dict, role: str = "serve", gpu_name: str | None = None, n: int | None = None) -> list:
    """돌아온 offer 가 고정 조건에 맞는지 다시 본다(서버 필터를 믿기만 하지 않는다). eval 도 serve 와 같은 조건(결정 8)
    + 디스크·RAM(T25, RAM 은 그 offer 의 장 수 × 장당 하한)."""
    rc = role_cfg(cfg, role)
    names = {g.replace("_", " ") for g in gpu_names(cfg, role, gpu_name)}
    lo, hi = gpu_range(rc, n)
    checks = [
        (str(o.get("gpu_name") or "").replace("_", " ") in names, "gpu_name"),
        (lo <= int(o.get("num_gpus") or 0) <= hi, "num_gpus"),
        (float(o.get("cuda_max_good") or 0) >= float(cfg["cuda_max_good_min"]), "cuda_max_good"),
        (float(o.get("reliability") or o.get("reliability2") or 0) >= float(cfg["reliability_min"]), "reliability"),
        (float(o.get("dph_total") or 99) <= float(rc["max_dph"]), "dph_total"),
    ]
    if role == "quant":
        def num(k):
            try:
                return float(o.get(k) or 0)
            except (TypeError, ValueError):
                return 0.0
        checks += [(num("gpu_ram") >= int(rc["gpu_ram_gb_min"]) * 1000, "gpu_ram"),
                   (num("cpu_ram") >= int(rc["cpu_ram_gb_min"]) * 1000, "cpu_ram"),
                   (num("cpu_cores_effective") >= int(rc["cpu_cores_min"]), "cpu_cores_effective"),
                   (num("disk_space") >= int(rc["disk"]), "disk_space"),
                   (num("inet_down") >= int(rc["inet_down_min"]), "inet_down")]
    if role in EVAL_LIKE:
        def num_e(k):
            try:
                return float(o.get(k) or 0)
            except (TypeError, ValueError):
                return 0.0
        checks.append((num_e("disk_space") >= int(rc["disk"]), "disk_space"))
        if rc.get("cpu_ram_gb_per_gpu_min"):
            checks.append((num_e("cpu_ram") >= int(rc["cpu_ram_gb_per_gpu_min"]) * int(o.get("num_gpus") or 0) * 1000, "cpu_ram"))
    if want_datacenter(cfg, role):
        # 응답에 datacenter 칸은 없다(검색 조건일 뿐, 문서). v12.10 실측의 datacenter 검색 결과는 hosting_type 1 이었다.
        # 칸이 오면 그 값을, 없으면 hosting_type == 1 을 본다(서버의 datacenter 검색 조건에 기대는 몫이 남음, README)
        dc = o.get("datacenter")
        checks.append((bool(dc) if dc is not None else o.get("hosting_type") == 1, "datacenter"))
    if cfg["verified"]:
        # 응답의 verification 문자열(verified·deverified·unverified)을 본다. 칸이 없을 때만 verified 불(옛 꼴)·서버 검색 조건에 기댄다
        v = o.get("verification")
        checks.append((v == "verified" if v is not None else o.get("verified") in (True, None), "verified"))
    return [n for ok, n in checks if not ok]


def offer_ok(cfg: dict, o: dict, role: str = "serve", gpu_name: str | None = None, n: int | None = None) -> str | None:
    bad = offer_fails(cfg, o, role, gpu_name, n)
    return ("조건 밖: " + ", ".join(bad)) if bad else None


def history_machines(record_dir: str = RECORD_DIR) -> list:
    p = os.path.join(record_dir, "history.jsonl")
    out = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("event") == "up" and j.get("machine_id") and j["machine_id"] not in out:
                    out.append(j["machine_id"])
    except OSError:
        pass
    return out


def rank_offers(offers: list, prev_machines: list, band: float) -> list:
    """장당 값 싼 순(1장이면 그냥 값 싼 순). 가장 싼 장당 값 + band 안에서는 전에 쓴 기계를 앞에, 그다음 장 수 많은 것을 앞에 둔다.
    band 는 이 정렬에만 쓴다 — up 은 band 를 검사하지 않는다(결정 T19b)."""
    offers = sorted(offers, key=lambda o: (per_gpu(o), o.get("id", 0)))
    if not offers:
        return offers
    lim = per_gpu(offers[0]) + band
    head = [o for o in offers if per_gpu(o) <= lim]
    tail = [o for o in offers if o not in head]
    head.sort(key=lambda o: (0 if o.get("machine_id") in prev_machines else 1, -int(o.get("num_gpus") or 1), per_gpu(o)))
    return head + tail


def pick(o: dict, fields) -> dict:
    return {k: o.get(k) for k in fields}


LIST_PAGES_MAX = 20   # 25개씩 500개까지(우리 것은 많아야 1개)


def list_instances(key: str) -> list:
    """내 인스턴스 전부(GET /api/v1/instances/, next_token 으로 다음 쪽). 응답 칸은 v0 목록과 같은 instance 행이다."""
    rows, q = [], {"limit": 25, "order_by": [{"col": "id", "dir": "asc"}], "select_filters": {}}
    for _ in range(LIST_PAGES_MAX):
        j = vast(key, "GET", "/v1/instances/", query=q)
        got = j.get("instances")
        if not isinstance(got, list):
            raise Stop("Vast 목록 응답에 instances 목록이 없다", EXIT_OTHER)
        rows += [i for i in got if isinstance(i, dict)]
        tok = j.get("next_token")
        if not tok:
            return rows
        q = dict(q, after_token=tok)
    raise Stop(f"인스턴스 목록이 {LIST_PAGES_MAX} 쪽을 넘는다", EXIT_OTHER)


def ours(cfg: dict, inst: list, role: str | None = None) -> list:
    """stewart- 이름표 인스턴스. role 을 주면 그 역할만(이름표로 안다)."""
    mine = [i for i in inst if str(i.get("label") or "").startswith(cfg["label_prefix"])]
    return mine if role is None else [i for i in mine if role_of(cfg, i.get("label")) == role]


def current_file(role: str) -> str:
    """serve 는 current.json(사용자 서버·stewart cloud wait 가 읽는 그대로), eval 은 current-eval.json."""
    return "current.json" if role == "serve" else f"current-{role}.json"


def record(event: dict, record_dir: str = RECORD_DIR, current: dict | None = None, current_name: str = "current.json") -> None:
    os.makedirs(record_dir, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), **event}
    with open(os.path.join(record_dir, "history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    if current is not None:
        tmp = os.path.join(record_dir, f".{current_name}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(record_dir, current_name))


def do_offers(key: str, cfg: dict, record_dir: str = RECORD_DIR, role: str = "serve", ref_gpu: str | None = None,
              n: int | None = None) -> int:
    rc = role_cfg(cfg, role)
    r = vast(key, "POST", "/v0/bundles/", offer_query(cfg, role, ref_gpu, n))
    offers = [o for o in (r.get("offers") or []) if not offer_ok(cfg, o, role, ref_gpu, n)]
    ranked = rank_offers(offers, history_machines(record_dir), float(cfg["band_dph"]))
    present = name_present(rc["ts_name"])
    say(f"[{role}] 조건에 맞는 offer {len(ranked)}개({cond_view(cfg, role, ref_gpu, n)}). 이름 {rc['ts_name']}: "
        + ("아직 테일넷에 있다 — up 이 거부된다(최대 60분 뒤 자동으로 지워짐)" if present else "비어 있음"))
    prev = set(history_machines(record_dir))
    for o in ranked[: int(cfg["offers_shown"])]:
        emit({**pick({**o, "dph_per_gpu": round(per_gpu(o), 4)}, OFFER_FIELDS), "used_before": o.get("machine_id") in prev})
    if ref_gpu:
        say(f"참고 보기다 — up 은 {cfg['gpu_name']} 만 빌린다")
    elif ranked:
        o = ranked[0]
        word = "" if role == "serve" else f"{role} "
        plan = " <계획>" if role in EVAL_LIKE else ""
        say(f"빌리려면: {PREFIX}/bin/stewart-cloud up {word}{o['id']} {float(o['dph_total']) + 0.005:.3f}{plan}")
        if role in EVAL_LIKE:
            say(f"계획은 그 offer 의 장 수({o.get('num_gpus')})에 맞는 것(serving models.toml plans 의 gpus — 다르면 파드가 HALT, "
                "fill 계획 a-all 은 몇 장이든). 계획 없이 빌리면 카드마다 CF 의 MODEL")
    if not ranked:
        near_misses(key, cfg, role, ref_gpu, n)
    return 0 if ranked else EXIT_REFUSE


def cond_view(cfg: dict, role: str, ref_gpu: str | None = None, n: int | None = None) -> str:
    """offers 머리줄의 조건 글. GPU 이름은 검색식과 같은 목록(T26: quant 가 RTX 3090 으로 찍히던 것을 고침)."""
    rc = role_cfg(cfg, role)
    lo, hi = gpu_range(rc, n)
    names = gpu_names(cfg, role, ref_gpu)
    out = f"{'·'.join(names)} ×{lo}" + (f"~{hi}" if hi != lo else "") + f", 값 상한 {rc['max_dph']}$/hr"
    if role in EVAL_LIKE:
        out += f", 디스크 ≥{rc['disk']}GB" + (f", RAM ≥장당 {rc['cpu_ram_gb_per_gpu_min']}GB" if rc.get("cpu_ram_gb_per_gpu_min") else "")
    if role == "quant":
        out += (f", gpu_ram ≥{rc['gpu_ram_gb_min']}GB, RAM ≥{rc['cpu_ram_gb_min']}GB, CPU ≥{rc['cpu_cores_min']}, "
                f"디스크 ≥{rc['disk']}GB, 망 ≥{rc['inet_down_min']}Mb/s")
    return out + f", 신뢰도 ≥{cfg['reliability_min']}, cuda ≥{cfg['cuda_max_good_min']}" + \
        (", verified" if cfg["verified"] else "") + (", datacenter" if want_datacenter(cfg, role) else ", datacenter 조건 없음")


NEAR_SHOWN = 3


def near_misses(key: str, cfg: dict, role: str, ref_gpu: str | None = None, n: int | None = None) -> None:
    """0개일 때(T26): 같은 GPU 이름·장 수로 그 밖 조건을 풀어 한 번 더 찾고, 조건별로 몇 개가 빠졌는지와 가장 가까운 몇 개를 보인다.
    무료 조회(offer 검색) 하나 더. 빌리는 데는 쓰지 않는다 — up 은 늘 offer_query 의 전체 조건이다."""
    rc = role_cfg(cfg, role)
    q = {"limit": 64, "type": "ondemand", "rentable": {"eq": True}, "rented": {"eq": False},
         "gpu_name": {"in": sorted({x for g in gpu_names(cfg, role, ref_gpu) for x in (g, g.replace(" ", "_"))})},
         "num_gpus": num_gpus_query(rc, n), "dph_total": {"lte": ABS_MAX_DPH}, "order": [["dph_total", "asc"]]}
    got = vast(key, "POST", "/v0/bundles/", q).get("offers") or []
    rows = [(offer_fails(cfg, o, role, ref_gpu, n), o) for o in got]
    count: dict = {}
    for fails, _ in rows:
        for f in fails:
            count[f] = count.get(f, 0) + 1
    why = ", ".join(f"{k} {v}" for k, v in sorted(count.items(), key=lambda kv: -kv[1])) or "없음"
    say(f"조건을 풀어 본 같은 GPU·장 수 offer {len(got)}개(값 ≤{ABS_MAX_DPH}$/hr) — 조건별로 빠진 수: {why}")
    for fails, o in sorted(rows, key=lambda r: (len(r[0]), float(r[1].get("dph_total") or 99)))[:NEAR_SHOWN]:
        # stdout 의 JSON 줄(빌릴 수 있는 offer)과 섞이지 않게 stderr 글 줄로만 — 이 id 로는 up 이 늘 거부된다
        v = pick({**o, "dph_per_gpu": round(per_gpu(o), 4)}, OFFER_FIELDS)
        say("가까운 것(빌릴 수 없음): " + json.dumps({"near_miss": fails, **v}, ensure_ascii=False))


def elapsed(i: dict, now: float | None = None) -> dict:
    """인스턴스 목록 행의 start_date(epoch 초)와 dph_total 로 켜진 시간·지금까지 값(추정: 시간 × 지금 값, 디스크·망 요금 뺌)."""
    try:
        t0, dph = float(i.get("start_date") or 0), float(i.get("dph_total") or 0)
    except (TypeError, ValueError):
        return {"hours": None, "usd": None}
    if t0 <= 0:
        return {"hours": None, "usd": None}
    h = max(0.0, ((now if now is not None else time.time()) - t0) / 3600)
    return {"hours": round(h, 2), "usd": round(h * dph, 2)}


STALL_MIN = 10      # running 인데 이 분이 지나도 테일넷 이름이 없으면 부팅 멈춤 의심(T23, E12: 1장 serve 는 6분)
FETCH_KEYS = ("alias", "state", "files_done", "files_total", "bytes_done", "bytes_total", "rate_Bps", "current", "updated")
# ↑ modelrecv 의 진행 파일($S/model-fetch.json) 칸 중 보일 것(T24 — T22 의 짐작 칸 이름을 버림). 감시기 v9 가 /status model_fetch 로 싣는다


def idle_left_view(st: dict | None) -> str | None:
    """감시기 v9 /status 의 idle_left_s·limits.idle_s → "남은 N분(쉬다 정지 M분)". 없으면 None(v8 이하 파드)."""
    try:
        left = (st or {}).get("idle_left_s")
        if left is None:
            return None
        lim = ((st or {}).get("limits") or {}).get("idle_s")
        return f"{round(float(left) / 60)}분 남음" + (f"(쉬다 정지 {round(float(lim) / 60)}분)" if lim else "")
    except (TypeError, ValueError):
        return None


def model_fetch_view(st: dict | None):
    """/status 의 model_fetch에서 짧은 칸만. 없으면 None."""
    mf = (st or {}).get("model_fetch")
    if not isinstance(mf, dict):
        return None
    out = {k: mf[k] for k in FETCH_KEYS if k in mf and isinstance(mf[k], (str, int, float, bool, type(None)))}
    out = {k: (scrub(v)[:120] if isinstance(v, str) else v) for k, v in out.items()}
    try:                                     # 사람이 읽기 쉽게: 받은 GB·퍼센트·MB/s
        done, total = float(out.get("bytes_done") or 0), float(out.get("bytes_total") or 0)
        if total > 0:
            out["pct"] = round(100 * done / total, 1)
        if out.get("rate_Bps") is not None:
            out["MBps"] = round(float(out["rate_Bps"]) / 1e6, 1)
    except (TypeError, ValueError):
        pass
    return out or None


def do_status(key: str, cfg: dict) -> int:
    inst = ours(cfg, list_instances(key))
    say(f"{cfg['label_prefix']} 인스턴스 {len(inst)}개(역할마다 1개, 합쳐 최대 {MAX_TOTAL}개)")
    present = {r: name_present(role_cfg(cfg, r)["ts_name"]) for r in roles_on(cfg)}
    for i in inst:
        role = role_of(cfg, i.get("label"))
        el = elapsed(i)
        emit({"role": role, **pick(i, INSTANCE_FIELDS), "num_gpus": i.get("num_gpus"), **el})
        if is_stopped(i):
            say(f"경고: {i.get('id')} 는 정지 상태다 — Destroy 하지 않아 디스크 요금이 나간다(down {i.get('id')}, "
                "같은 역할의 up 은 먼저 지운다)")
        elif (role in present and str(i.get("actual_status")) == "running" and not present[role]
              and el["hours"] is not None and el["hours"] * 60 >= STALL_MIN):
            say(f"경고: {i.get('id')}({role}) 는 running 인데 켜진 지 {round(el['hours'] * 60)}분째 테일넷 이름 "
                f"{role_cfg(cfg, role)['ts_name']} 이 비었다 — 부팅 멈춤 의심 — logs {i.get('id')}")
    for r in roles_on(cfg):
        n = role_cfg(cfg, r)["ts_name"]
        if not present[r]:
            say(f"[{r}] 이름 {n}: 비어 있음")
            continue
        st = pod_status(n, int(cfg["logout_port"]), timeout=5)
        mf = model_fetch_view(st)
        idle = idle_left_view(st)
        say(f"[{r}] 이름 {n}: 테일넷에 있다" + (f" — 쉬다 정지까지 {idle}" if idle else "")
            + (f" — model_fetch {json.dumps(mf, ensure_ascii=False)}" if mf else ""))
        for line in lane_fail_lines(st):
            say(f"[{r}] {line}")
    return 0


def lane_fail_lines(st: dict | None) -> list:
    """T28: 파드 v12 /status 의 lanes[] 중 failed 차선과 HALT 줄(pod_log 끝). 글은 비밀 꼴을 가려 300자까지."""
    out = []
    lanes = (st or {}).get("lanes")
    for ln in lanes if isinstance(lanes, list) else []:
        if isinstance(ln, dict) and ln.get("state") == "failed":
            out.append(f"차선 {ln.get('k')} failed({ln.get('alias') or '?'}): {scrub_log(str(ln.get('why') or '-'))[:300]}")
    for line in ((st or {}).get("pod_log") or [])[-20:]:
        if "HALT" in str(line):
            out.append("파드 " + scrub_log(str(line))[:300])
            break
    return out


def is_stopped(i: dict) -> bool:
    """정지(쉬다 정지·HALT 정지·사람 정지) — GPU 는 안 쓰고 디스크 요금만 나가는 상태."""
    return str(i.get("actual_status")) in ("exited", "stopped", "offline") or i.get("intended_status") == "stopped"


def role_env(role: str, n: int, plan: str | None = None) -> dict:
    """만들 때 넘기는 env. 정해진 낱말·정수·계획 이름 꼴만 — 설정의 글자를 넣지 않는다.
    꼴(T23): vast-cli create instance 와 같은 **dict**({"KEY": "값"}). vast-cli 는 "-e K=V" 글을 parse_env 로 dict 로 바꿔 보낸다.
    Vast 문서(create instance): "When using a template, request env is merged with template env"(키별, 겹치면 요청 값).
    T20~T22 는 문서 스키마의 글 꼴("-e K=V …")을 보냈다 — E12(v7 eval 이 테일넷에 한 번도 안 붙음)의 의심 원인이라 dict 로 바꿨다."""
    if role not in ROLES or not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= MAX_GPUS):
        raise Stop(f"역할 env 를 만들 수 없다({role!r}, {n!r})", EXIT_OTHER)
    env = {"STEWART_ROLE": role, "STEWART_GPUS": str(n)}
    if plan is not None:
        if role not in EVAL_LIKE or not _PLAN.match(str(plan)):
            raise Stop(f"계획은 eval·eval2 만, 이름 꼴 {_PLAN.pattern}", EXIT_USAGE)
        env["STEWART_PLAN"] = plan
    return env


def do_up(key: str, cfg: dict, role: str, offer_id: int, max_dph: float, record_dir: str = RECORD_DIR,
          plan: str | None = None, tkey: str | None = None, state_dir: str = STATE_DIR) -> int:
    # T27: 지금 템플릿(상태 파일)을 API 로 읽어 PS·SB·Args 가 publish 기록과 같을 때만 — 다르면 아무것도 지우거나 빌리지 않는다
    tpl = check_up_template(tkey or key, load_state(state_dir), role)
    rc = role_cfg(cfg, role)
    if max_dph > float(rc["max_dph"]):
        raise Stop(f"max_dph 가 {role} 설정 상한 {rc['max_dph']} 를 넘는다", EXIT_USAGE)
    mine = ours(cfg, list_instances(key))
    same = [i for i in mine if role_of(cfg, i.get("label")) == role]
    if same and all(is_stopped(i) for i in same):
        # T24: 같은 역할의 정지(쉬다 정지·HALT 정지) 인스턴스는 디스크 요금만 새므로 먼저 지우고 빌린다(사람이 손으로 down 하던 것)
        for i in same:
            say(f"같은 역할({role})의 정지 인스턴스 {i.get('id')}({i.get('actual_status')}) 를 먼저 지운다")
            if do_down(key, cfg, int(i.get("id")), record_dir) != 0:
                raise Stop(f"정지 인스턴스 {i.get('id')} 를 지우지 못했다 — 빌리지 않는다(status 로 본다)", EXIT_OTHER)
        mine = ours(cfg, list_instances(key))
        same = [i for i in mine if role_of(cfg, i.get("label")) == role]
    if same:
        raise Stop(f"이미 {role} 인스턴스가 있다: {', '.join(str(i.get('id')) for i in same)} (역할마다 1개, 돌고 있음). "
                   "먼저 down 한다", EXIT_REFUSE)
    if len(mine) >= MAX_TOTAL:
        raise Stop(f"이미 {cfg['label_prefix']} 인스턴스가 {len(mine)}개다: {', '.join(str(i.get('id')) for i in mine)} "
                   f"(합쳐 최대 {MAX_TOTAL}개). 먼저 down 한다", EXIT_REFUSE)
    if name_present(rc["ts_name"]):
        raise Stop(f"이름 {rc['ts_name']} 이 아직 테일넷에 있다(이전 노드). 새 노드가 다른 이름(-1)이 되므로 빌리지 않는다. "
                   "최대 60분 기다리거나 Tailscale 관리 화면에서 그 기기를 지운다", EXIT_REFUSE)
    # offers 와 같은 검색식(같은 조건·같은 상한)으로 다시 받아 그 안에서 id 로 고른다.
    # 검색식에 id 조건을 넣으면 Vast 가 받지 않거나 다르게 풀어 늘 빈 목록이었다(11/8 22:4x 실측, T19).
    r = vast(key, "POST", "/v0/bundles/", offer_query(cfg, role))
    found = [o for o in (r.get("offers") or []) if int(o.get("id") or 0) == offer_id]
    if not found:
        raise Stop(f"offer {offer_id} 가 지금 {role} 조건(값 상한 {rc['max_dph']} 포함)에 맞는 목록에 없다(빌려 갔거나 조건 밖). "
                   f"offers {role} 를 다시 본다", EXIT_REFUSE)
    o = found[0]
    if (why := offer_ok(cfg, o, role)):
        raise Stop(f"offer {offer_id} {why}", EXIT_REFUSE)
    if float(o["dph_total"]) > max_dph:
        raise Stop(f"offer {offer_id} 값 {float(o['dph_total']):.4f}$/hr 이 승인한 상한 {max_dph} 를 넘는다", EXIT_REFUSE)
    if role in EVAL_LIKE:
        # T26: eval + eval2 의 켜진 값 합(정지 인스턴스는 디스크 요금만이라 빼고, 역할 이름표로 센다) + 이 offer ≤ eval_total_max_dph
        on = [i for i in mine if role_of(cfg, i.get("label")) in EVAL_LIKE and not is_stopped(i)]
        used = sum(float(i.get("dph_total") or 0) for i in on)
        if used + float(o["dph_total"]) > eval_total(cfg) + 1e-9:
            raise Stop(f"eval·eval2 켜진 값 합 {used:.3f} + 이 offer {float(o['dph_total']):.3f} 이 합계 상한 {eval_total(cfg)}$/hr "
                       f"(config.eval_total_max_dph)을 넘는다: {', '.join(str(i.get('id')) for i in on) or '-'}", EXIT_REFUSE)
    n = int(o["num_gpus"])
    label = role_label(cfg, role) + datetime.now().strftime("%Y%m%d-%H%M")
    body = {"template_hash_id": tpl, "label": label, "disk": int(rc["disk"]), "env": role_env(role, n, plan)}
    c = vast(key, "PUT", f"/v0/asks/{offer_id}/", body)
    iid = c.get("new_contract")
    if not c.get("success") or not iid:
        raise Stop(f"만들기 응답이 성공이 아니다: {scrub(str(c.get('msg') or c.get('error') or '')[:200])}", EXIT_OTHER)
    machine = o.get("machine_id")
    for _ in range(5):                        # 목록에 나타날 때까지 잠깐(machine_id 를 목록에서도 확인)
        got = [i for i in list_instances(key) if int(i.get("id") or 0) == int(iid)]
        if got:
            machine = got[0].get("machine_id") or machine
            break
        time.sleep(2)
    cur = {"role": role, "plan": plan, "template_hash_id": tpl, "instance_id": int(iid), "machine_id": machine, "offer_id": offer_id, "num_gpus": n,
           "dph_total": o.get("dph_total"), "label": label, "ts_name": rc["ts_name"], "geolocation": o.get("geolocation"),
           "created": datetime.now().astimezone().isoformat(timespec="seconds"),
           "state": "created", "used_before": machine in history_machines(record_dir)}
    record({"event": "up", **cur}, record_dir, current=cur, current_name=current_file(role))
    wait = "uv run stewart cloud wait" if role == "serve" else f"/status 확인: http://{rc['ts_name']}:8001/status"
    say(f"빌림[{role}]: instance {iid}, machine {machine}, GPU {n}장, {float(o['dph_total']):.4f}$/hr, label {label}. 기다리기: {wait}")
    emit(cur)
    return 0


def do_down(key: str, cfg: dict, instance_id: int, record_dir: str = RECORD_DIR) -> int:
    inst = list_instances(key)
    hit = [i for i in inst if int(i.get("id") or 0) == instance_id]
    if not hit:
        raise Stop(f"instance {instance_id} 가 목록에 없다(이미 지워졌거나 남의 것)", EXIT_REFUSE)
    if not str(hit[0].get("label") or "").startswith(cfg["label_prefix"]):
        raise Stop(f"instance {instance_id} 는 라벨이 {cfg['label_prefix']} 가 아니다 — 지우지 않는다", EXIT_REFUSE)
    role = role_of(cfg, hit[0].get("label"))
    # 역할을 모르는 stewart- 이름표(손으로 붙인 것)는 이름을 쥐었는지 알 수 없다 — logout 없이 Destroy 만(아래 holder None 과 같다)
    ts_name = role_cfg(cfg, role)["ts_name"] if role else None
    # 이름을 쥔 pod 가 지우려는 그 인스턴스일 때만 logout 한다.
    # (손으로 빌린 다른 인스턴스가 이름을 쥐고 있으면 그쪽이 테일넷에서 빠져 그쪽 일이 끊긴다 — 메인 검토 11/8 14:0x)
    holder = pod_instance(ts_name, int(cfg["logout_port"])) if ts_name else None
    gone = False
    if holder == str(instance_id):
        lo = pod_logout(ts_name, int(cfg["logout_port"]))
        for _ in range(15):                   # 이름이 풀리지 않을 때까지 최대 30초
            if not name_present(ts_name):
                gone = True
                break
            time.sleep(2)
        say(f"pod logout: {lo}; 이름 {ts_name}: " + ("비었다" if gone else "아직 있다(30~60분 뒤 자동으로 지워짐)"))
    elif holder is None:
        lo = "건너뜀(이름을 쥔 pod 를 읽지 못함)"
        say(f"pod logout 건너뜀 — {ts_name or '(역할을 모르는 이름표)'} 의 /status 를 읽지 못해 어느 인스턴스인지 모른다. "
            "Destroy 는 한다(이 인스턴스가 이름을 쥐고 있었다면 30~60분 뒤 자동으로 지워짐)")
    else:
        lo = f"건너뜀(이름을 쥔 pod 가 다른 인스턴스 {holder})"
        say(f"pod logout 건너뜀 — 이름 {ts_name} 을 쥔 pod 가 다른 인스턴스({holder})다. "
            f"그쪽은 건드리지 않고 {instance_id} 만 Destroy 한다")
    d = vast(key, "DELETE", f"/v0/instances/{instance_id}/")
    if d.get("success") is False:
        raise Stop(f"Destroy 응답이 실패다: {scrub(str(d.get('msg') or '')[:200])}", EXIT_OTHER)
    still = True
    for _ in range(15):
        if not any(int(i.get("id") or 0) == instance_id for i in list_instances(key)):
            still = False
            break
        time.sleep(2)
    ev = {"event": "down", "role": role, "instance_id": instance_id, "machine_id": hit[0].get("machine_id"), "label": hit[0].get("label"),
          "logout": lo, "name_cleared": gone, "listed_after": still}
    cur = None
    try:
        with open(os.path.join(record_dir, current_file(role or "serve")), encoding="utf-8") as f:
            c0 = json.load(f)
        if int(c0.get("instance_id") or 0) == instance_id:
            cur = {**c0, "state": "destroyed", "destroyed": datetime.now().astimezone().isoformat(timespec="seconds")}
    except (OSError, ValueError):
        pass
    record(ev, record_dir, current=cur, current_name=current_file(role or "serve"))
    say(f"Destroy: instance {instance_id}" + (" — 목록에서 사라짐" if not still else " — 아직 목록에 있다(잠시 뒤 status 로 본다)"))
    emit(ev)
    return 0 if not still else EXIT_OTHER


LOG_TAIL = 200
_RESULT_URL = re.compile(r"https://([a-z0-9-]{1,63}\.)+[a-z]{2,24}/[^\s\\\"'<>]{0,2000}")
_LOG_SECRETS = [
    (re.compile(r"tskey-[A-Za-z0-9_\-]+"), "<가림>"),                                         # Tailscale 인증 키
    (re.compile(r"https?://gist\.(?:githubusercontent|github)\.com/\S+"), "<가림>"),           # PU·CF(gist raw 주소)
    (re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?\S+"), r"\1<가림>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer <가림>"),
    (re.compile(r"(?i)\b((?:[a-z_]*_)?(?:api_?key|token|secret|password))(\s*[=:]\s*)\S+"), r"\1\2<가림>"),
]


def scrub_log(text: str, key: str | None = None) -> str:
    """로그를 보이기 전에 비밀 꼴을 가린다. 64자 hex(sha256)는 남긴다(PS 대조 실패 원인을 봐야 하므로 일반 긴 글 가림은 쓰지 않음)."""
    if key:
        text = text.replace(key, "<가림>")
    for rx, sub in _LOG_SECRETS:
        text = rx.sub(sub, text)
    return text


def fetch_result(url: str, tries: int = 30, wait_s: float = 1.0) -> str:
    """Vast 가 준 result_url(S3 등, 서명된 주소)에서 로그 글을 받는다. 키는 보내지 않는다. 생길 때까지 몇 초 기다린다."""
    if not _RESULT_URL.fullmatch(url or "") or re.match(r"https://[^/]*\.ts\.net/", url):
        raise Stop("Vast 가 준 로그 주소 꼴이 이상하다(https 공개 주소가 아님)", EXIT_OTHER)
    last = ""
    for _ in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "stewart-cloud/1"})
        try:
            with _OPENER.open(req, timeout=20) as r:
                return r.read(8_000_000).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError) as e:
            last = type(e).__name__
        time.sleep(wait_s)
    raise Stop(f"로그가 아직 안 나왔다({last}, {tries}번). 잠시 뒤 다시 부른다", EXIT_OTHER)


def do_logs(key: str, cfg: dict, instance_id: int, daemon: bool = False) -> int:
    hit = [i for i in list_instances(key) if int(i.get("id") or 0) == instance_id]
    if not hit:
        raise Stop(f"instance {instance_id} 가 목록에 없다(이미 지워졌거나 남의 것)", EXIT_REFUSE)
    if not str(hit[0].get("label") or "").startswith(cfg["label_prefix"]):
        raise Stop(f"instance {instance_id} 는 라벨이 {cfg['label_prefix']} 가 아니다 — 로그를 보지 않는다", EXIT_REFUSE)
    body = {"tail": str(LOG_TAIL)}
    if daemon:
        body["daemon_logs"] = "true"
    r = vast(key, "PUT", f"/v0/instances/request_logs/{instance_id}/", body)
    url = r.get("result_url")
    if not url:
        raise Stop(f"로그 주소를 못 받았다: {scrub(str(r.get('msg') or r.get('error') or '')[:200])}", EXIT_OTHER)
    text = fetch_result(str(url))
    lines = [l for l in text.splitlines() if l.strip()][-LOG_TAIL:]
    say(f"{'데몬' if daemon else '컨테이너'} 로그 끝 {len(lines)}줄 — instance {instance_id}({role_of(cfg, hit[0].get('label')) or '?'}, "
        f"{hit[0].get('actual_status')}). 비밀 꼴은 <가림>")
    for l in lines:
        sys.stdout.write(scrub_log(l, key)[:2000] + "\n")
    sys.stdout.flush()
    return 0


# ── 묶음 올리기·템플릿 만들기(publish, T27) ─────────────────────────
# 흐름: verify(무료) = 저장소 커밋에서 깨끗한 사본으로 묶음을 다시 만들어 PS·SB·Args·Docker Options 가 labs 의 묶음과 같은지 본다.
#       publish(ask)  = verify + gist 에 새 파일 → 리비전 고정 raw 주소 대조 → Vast 템플릿 만들기 → 다시 읽어 대조 → 상태 파일 기록·전환.
#       up            = 상태 파일의 지금 템플릿을 API 로 읽어 PS·SB·Args 가 기록(검증 통과한 묶음)과 같을 때만 빌린다.
# 저장소 코드는 이 프로세스에서 돌리지 않는다:
#   - git 실행 파일·저장소 설정(.git/config 의 filter·fsmonitor·promisor 등)을 거치지 않고 .git/objects 를 직접 읽고, 객체마다 SHA-1 을 다시 계산해
#     커밋 sha 에서 파일까지 이어지는지 본다(replace·grafts·필터가 끼어들 수 없다).
#   - 빌더(deploy/vast/make_template_env.py · deploy/quant/make_quant_bundle.py)는 root 소유 Python 3.13(세션이 쓰는 uv 3.13.16 의 사본)으로
#     sandbox-exec 안에서 돌린다: 망 없음, mach 없음, fork 없음, /Volumes·/Users·/private/tmp·/private/var/folders 읽기 없음
#     (이번 임시 폴더만 읽음), 쓰기는 출력 폴더만.
# 올리는 것은 다시 만든 바이트다(labs 파일이 아니다 — labs 는 같은지 대조만).

def kind_of(bundle: str) -> str:
    return "quant" if bundle.startswith("quant-") else "pod"


def kind_of_role(role: str) -> str:
    return "quant" if role == "quant" else "pod"


def sha256s(s) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


# ── git 객체 직접 읽기 ──

class GitStore:
    """.git/objects 를 읽는다(낱 객체 + pack idx v2·델타). 돌려주는 객체는 늘 이름(SHA-1)과 대조한 것이다."""
    TYPES = {1: "commit", 2: "tree", 3: "blob", 4: "tag"}

    def __init__(self, gitdir: str):
        self.obj = os.path.join(gitdir, "objects")
        if not os.path.isdir(self.obj):
            raise Stop(f"저장소 객체 폴더가 없다: {self.obj}", EXIT_OTHER)
        if os.path.exists(os.path.join(self.obj, "info", "alternates")):
            raise Stop("저장소가 alternates 를 쓴다 — 다른 객체 폴더는 읽지 않는다", EXIT_REFUSE)
        self._packs = None

    def _packs_load(self) -> list:
        if self._packs is None:
            self._packs = []
            pdir = os.path.join(self.obj, "pack")
            for f in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
                if not f.endswith(".idx"):
                    continue
                with open(os.path.join(pdir, f), "rb") as fh:
                    idx = fh.read()
                if idx[:8] != b"\xfftOc\x00\x00\x00\x02":
                    raise Stop(f"pack idx 판이 2 가 아니다: {f}", EXIT_OTHER)
                n = int.from_bytes(idx[8 + 255 * 4:8 + 256 * 4], "big")
                names = 8 + 256 * 4
                offs = names + n * 20 + n * 4
                large = offs + n * 4
                self._packs.append((os.path.join(pdir, f[:-4] + ".pack"), idx, n, names, offs, large))
        return self._packs

    def _pack_find(self, h: bytes):
        for path, idx, n, names, offs, large in self._packs_load():
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                cur = idx[names + mid * 20:names + mid * 20 + 20]
                if cur < h:
                    lo = mid + 1
                elif cur > h:
                    hi = mid
                else:
                    o = int.from_bytes(idx[offs + mid * 4:offs + mid * 4 + 4], "big")
                    if o & 0x80000000:
                        k = o & 0x7FFFFFFF
                        o = int.from_bytes(idx[large + k * 8:large + k * 8 + 8], "big")
                    return path, o
        return None

    @staticmethod
    def _inflate(fh, pos: int, size: int) -> bytes:
        fh.seek(pos)
        d, out = zlib.decompressobj(), b""
        while not d.eof:
            chunk = fh.read(65536)
            if not chunk:
                raise Stop("pack 객체가 잘렸다", EXIT_OTHER)
            out += d.decompress(chunk)
            if len(out) > GIT_MAX_BYTES:
                raise Stop("git 객체가 너무 크다", EXIT_OTHER)
        if len(out) != size:
            raise Stop("pack 객체 크기가 머리와 다르다", EXIT_OTHER)
        return out

    @staticmethod
    def _delta(base: bytes, d: bytes) -> bytes:
        i = 0

        def varint():
            nonlocal i
            v = shift = 0
            while True:
                c = d[i]
                i += 1
                v |= (c & 0x7F) << shift
                shift += 7
                if not c & 0x80:
                    return v
        if varint() != len(base):
            raise Stop("델타 바탕 크기가 다르다", EXIT_OTHER)
        want, out = varint(), bytearray()
        while i < len(d):
            c = d[i]
            i += 1
            if c & 0x80:
                off = n = 0
                for k in range(4):
                    if c & (1 << k):
                        off |= d[i] << (8 * k)
                        i += 1
                for k in range(3):
                    if c & (0x10 << k):
                        n |= d[i] << (8 * k)
                        i += 1
                n = n or 0x10000
                if off + n > len(base):
                    raise Stop("델타 복사가 바탕 밖이다", EXIT_OTHER)
                out += base[off:off + n]
            elif c:
                out += d[i:i + c]
                i += c
            else:
                raise Stop("델타 명령 0", EXIT_OTHER)
        if len(out) != want:
            raise Stop("델타 결과 크기가 다르다", EXIT_OTHER)
        return bytes(out)

    def _pack_obj(self, path: str, off: int, depth: int = 0):
        if depth > 60:
            raise Stop("델타 사슬이 너무 길다", EXIT_OTHER)
        with open(path, "rb") as fh:
            fh.seek(off)
            c = fh.read(1)[0]
            typ, size, shift, p = (c >> 4) & 7, c & 15, 4, off + 1
            while c & 0x80:
                c = fh.read(1)[0]
                p += 1
                size |= (c & 0x7F) << shift
                shift += 7
            if typ in self.TYPES:
                return self.TYPES[typ], self._inflate(fh, p, size)
            if typ == 6:                                   # OFS_DELTA
                c = fh.read(1)[0]
                p += 1
                neg = c & 0x7F
                while c & 0x80:
                    c = fh.read(1)[0]
                    p += 1
                    neg = ((neg + 1) << 7) | (c & 0x7F)
                d = self._inflate(fh, p, size)
                bt, base = self._pack_obj(path, off - neg, depth + 1)
                return bt, self._delta(base, d)
            if typ == 7:                                   # REF_DELTA
                bh = fh.read(20)
                d = self._inflate(fh, p + 20, size)
                bt, base = self._raw(bh.hex(), depth + 1)
                return bt, self._delta(base, d)
        raise Stop(f"모르는 pack 객체 종류 {typ}", EXIT_OTHER)

    def _raw(self, h: str, depth: int = 0):
        p = os.path.join(self.obj, h[:2], h[2:])
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                raw = zlib.decompress(fh.read())
            hdr, _, body = raw.partition(b"\0")
            typ, _, size = hdr.decode("ascii", "replace").partition(" ")
            if not size.isdigit() or int(size) != len(body):
                raise Stop(f"낱 객체 머리가 이상하다: {h}", EXIT_OTHER)
            return typ, body
        hit = self._pack_find(bytes.fromhex(h))
        if hit is None:
            raise Stop(f"저장소에 객체가 없다: {h}", EXIT_REFUSE)
        return self._pack_obj(hit[0], hit[1], depth)

    def get(self, h: str, want: str) -> bytes:
        if not _SHA1.match(h):
            raise Stop(f"객체 이름 꼴이 다르다: {h!r}", EXIT_OTHER)
        typ, body = self._raw(h)
        if hashlib.sha1(f"{typ} {len(body)}".encode() + b"\0" + body).hexdigest() != h:
            raise Stop(f"객체 {h} 의 내용이 이름(SHA-1)과 다르다 — 객체 폴더가 바뀌었다", EXIT_REFUSE)
        if typ != want:
            raise Stop(f"객체 {h} 는 {typ} 다({want} 가 아님)", EXIT_REFUSE)
        return body

    def tree(self, h: str) -> list:
        body, out, i = self.get(h, "tree"), [], 0
        while i < len(body):
            sp = body.index(b" ", i)
            nul = body.index(b"\0", sp)
            out.append((body[i:sp].decode("ascii"), body[sp + 1:nul].decode("utf-8"), body[nul + 1:nul + 21].hex()))
            i = nul + 21
        return out

    def commit_tree(self, commit: str) -> str:
        first = self.get(commit, "commit").split(b"\n", 1)[0].decode("ascii", "replace")
        m = re.fullmatch(r"tree ([0-9a-f]{40})", first)
        if not m:
            raise Stop(f"커밋 {commit} 의 첫 줄이 tree 가 아니다", EXIT_OTHER)
        return m.group(1)

    def export(self, commit: str, sub: str, dest: str) -> int:
        """커밋의 sub(예: deploy) 아래 파일을 dest/sub 에 쓴다(일반 파일 644·755 만, 링크·하위 모듈이면 멈춤). 파일 수를 돌려준다."""
        t = self.commit_tree(commit)
        for part in sub.split("/"):
            hit = [e for e in self.tree(t) if e[1] == part]
            if not hit or hit[0][0] != "40000":
                raise Stop(f"커밋 {commit[:12]} 에 {sub}/ 가 없다", EXIT_REFUSE)
            t = hit[0][2]
        count = [0, 0]

        def walk(th: str, rel: str):
            for mode, name, h in self.tree(th):
                if not re.fullmatch(r"[A-Za-z0-9._+@-][A-Za-z0-9 ._+@-]{0,200}", name) or name in (".", ".."):
                    raise Stop(f"커밋 안 파일 이름 꼴이 이상하다: {rel}/{name!r}", EXIT_REFUSE)
                p = os.path.join(dest, rel, name)
                if mode == "40000":
                    os.makedirs(p, exist_ok=True)
                    walk(h, f"{rel}/{name}")
                elif mode in ("100644", "100755"):
                    data = self.get(h, "blob")
                    count[0] += 1
                    count[1] += len(data)
                    if count[0] > GIT_MAX_FILES or count[1] > GIT_MAX_BYTES:
                        raise Stop(f"{sub}/ 가 너무 크다(파일 {count[0]}, {count[1]} 바이트)", EXIT_REFUSE)
                    with open(p, "xb") as fh:
                        fh.write(data)
                    os.chmod(p, 0o755 if mode == "100755" else 0o644)
                else:
                    raise Stop(f"커밋 안 {rel}/{name} 은 일반 파일이 아니다(mode {mode}) — 링크·하위 모듈은 받지 않는다", EXIT_REFUSE)
        os.makedirs(os.path.join(dest, sub), exist_ok=True)
        walk(t, sub)
        return count[0]


# ── 다시 만들기(verify) ──

def build_prefix(out: str) -> list:
    """빌더를 부르는 명령 앞부분: root 소유 Python 3.13 을 sandbox-exec 안에서."""
    for p in (BUILD_PY, os.path.dirname(os.path.dirname(BUILD_PY))):
        try:
            st = os.stat(p)
        except OSError:
            raise Stop(f"빌드용 Python 이 없다: {p} — 설치 스크립트(v12.11)가 uv 의 cpython-3.13.16 을 root 소유로 복사한다", EXIT_KEY)
        if st.st_uid != 0 or st.st_mode & 0o022:
            raise Stop(f"빌드용 Python 이 root 소유·쓰기 막힘이 아니다: {p}", EXIT_KEY)
    return [SANDBOX_EXEC, "-D", f"OUT={out}", "-D", f"TMP={os.path.dirname(out)}", "-p", BUILD_PROFILE, BUILD_PY, "-I", "-B"]


def rebuild(bundle: str, commit: str, git_dir: str = REPO_GIT) -> dict:
    """커밋에서 깨끗한 사본을 만들어 빌더를 돌린다. {파일 이름: 바이트}(묶음 + BUILD_FILES)."""
    kind = kind_of(bundle)
    builder, b64name = BUNDLE_OUT[kind][0], BUNDLE_OUT[kind][1]
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="stewart-publish-"))
    try:
        src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
        os.makedirs(out)
        n = GitStore(git_dir).export(commit, "deploy", src)
        if not os.path.isfile(os.path.join(src, builder)):
            raise Stop(f"커밋 {commit[:12]} 에 {builder} 가 없다", EXIT_REFUSE)
        cmd = build_prefix(out) + [os.path.join(src, builder), "--out", out]
        try:
            r = subprocess.run(cmd, cwd=src, env={"PATH": "/usr/bin:/bin", "HOME": tmp, "LANG": "en_US.UTF-8"},
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=BUILD_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise Stop(f"빌더를 돌리지 못했다({type(e).__name__})", EXIT_OTHER)
        if r.returncode != 0:
            tail = scrub(r.stderr.decode("utf-8", "replace").strip()[-600:])
            raise Stop(f"빌더가 실패했다(exit {r.returncode}): {tail}", EXIT_REFUSE)
        got = {}
        for f in (b64name,) + BUILD_FILES:
            p = os.path.join(out, f)
            if not os.path.isfile(p) or os.path.islink(p):
                raise Stop(f"빌더가 {f} 를 내지 않았다(이 커밋의 묶음 꼴이 SB·docker-options 를 내는 판인지 본다)", EXIT_REFUSE)
            with open(p, "rb") as fh:
                got[f] = fh.read(B64_MAX + 1)
        if len(got[b64name]) > B64_MAX:
            raise Stop(f"묶음이 {B64_MAX}자를 넘는다", EXIT_REFUSE)
        say(f"다시 만듦: 커밋 {commit} 의 deploy/ 파일 {n}개(객체 SHA-1 대조) → {builder}(sandbox-exec, 망 없음)")
        return got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bundle_facts(bundle: str, files: dict) -> dict:
    b64 = files[BUNDLE_OUT[kind_of(bundle)][1]].decode("ascii")
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", b64):
        raise Stop("묶음이 줄바꿈 없는 base64 한 줄이 아니다", EXIT_REFUSE)
    args = files["args.txt"].decode("utf-8")
    sb = files["boot.sb.txt"].decode("ascii").strip()
    opts = files["docker-options.txt"].decode("utf-8").strip()
    return {"ps": sha256s(b64), "sb_sha": sha256s(sb), "args_sha": sha256s(args), "b64": b64, "args": args, "sb": sb,
            "options": opts}


def do_verify(bundle: str, commit: str, ps: str, git_dir: str = REPO_GIT, labs: str = LABS_VAST) -> dict:
    """다시 만든 것과 인자의 PS, labs/<묶음>/ 의 파일이 모두 같을 때만 사실(facts)을 돌려준다. 다르면 Stop(69)."""
    files = rebuild(bundle, commit, git_dir)
    facts = bundle_facts(bundle, files)
    bad = []
    if facts["ps"] != ps:
        bad.append(f"PS: 다시 만든 것 {facts['ps']} ≠ 인자 {ps}")
    ldir = os.path.join(labs, bundle)
    for f, data in files.items():
        p = os.path.join(ldir, f)
        try:
            with open(p, "rb") as fh:
                have = fh.read(B64_MAX + 1)
        except OSError:
            bad.append(f"labs/{bundle}/{f} 가 없다")
            continue
        if have != data:
            bad.append(f"labs/{bundle}/{f} 가 다시 만든 것과 다르다(sha {sha256s(have)[:12]}… ≠ {sha256s(data)[:12]}…)")
    if (kind_of(bundle) == "pod") != ("CF=" in facts["options"]):
        bad.append("docker-options 의 CF 칸이 종류와 맞지 않는다")
    for hole in _HOLE.findall(facts["options"]):
        if hole not in BUNDLE_OUT[kind_of(bundle)][3]:
            bad.append(f"docker-options 에 모르는 빈칸 {hole}")
    if bad:
        raise Stop("재현 대조 실패 — 올리지 않는다: " + "; ".join(bad), EXIT_REFUSE)
    say(f"재현 대조 통과[{bundle}]: PS {facts['ps']} · SB sha {facts['sb_sha']} · Args sha {facts['args_sha']} (labs/{bundle} 와 같음)")
    return facts


# ── 상태 파일(래퍼만 쓴다) ──

REC_KEYS = {"kind", "bundle", "commit", "ps", "sb_sha", "args_sha", "image", "source", "ts", "base"}


def state_check_dir(state_dir: str, strict: bool) -> None:
    try:
        st = os.lstat(state_dir)
    except OSError:
        raise Stop(f"상태 폴더가 없다: {state_dir} — 설치 스크립트가 만든다(README 의 publish)", EXIT_KEY)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise Stop(f"상태 폴더 {state_dir} 는 링크가 아닌 이 사용자 700 폴더여야 한다", EXIT_KEY)
    if strict:
        par = os.lstat(os.path.dirname(state_dir))
        if par.st_uid != 0 or par.st_mode & 0o022 or not stat.S_ISDIR(par.st_mode):
            raise Stop(f"상태 폴더의 부모가 root 소유·쓰기 막힘이 아니다: {os.path.dirname(state_dir)}", EXIT_KEY)


def load_state(state_dir: str = STATE_DIR) -> dict:
    state_check_dir(state_dir, state_dir == STATE_DIR)
    p = os.path.join(state_dir, STATE_FILE)
    try:
        if os.path.islink(p):
            raise OSError("link")
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
    except (OSError, ValueError) as e:
        raise Stop(f"상태 파일을 읽을 수 없다: {p} ({type(e).__name__}) — 설치 때 기존 템플릿을 넣어 만든다", EXIT_KEY)
    if not isinstance(s, dict) or set(s) != {"v", "current", "templates"} or s["v"] != 1:
        raise Stop("상태 파일 꼴이 다르다({v:1, current, templates})", EXIT_KEY)
    tpls, cur = s["templates"], s["current"]
    if not isinstance(tpls, dict) or not isinstance(cur, dict) or not set(cur) <= set(KINDS):
        raise Stop("상태 파일의 current·templates 꼴이 다르다", EXIT_KEY)
    for h, r in tpls.items():
        if not (_THASH.match(h) and isinstance(r, dict) and set(r) == REC_KEYS and r["kind"] in KINDS
                and _BUNDLE.match(str(r["bundle"])) and kind_of(r["bundle"]) == r["kind"]
                and all(_SHA256.match(str(r[k])) for k in ("ps", "sb_sha", "args_sha"))
                and (r["commit"] is None or _SHA1.match(str(r["commit"]))) and r["source"] in ("console", "publish")
                and (r["image"] is None or isinstance(r["image"], str)) and (r["base"] is None or _THASH.match(str(r["base"])))):
            raise Stop(f"상태 파일의 템플릿 기록 {h[:16]} 꼴이 다르다", EXIT_KEY)
    for k, h in cur.items():
        if h not in tpls or tpls[h]["kind"] != k:
            raise Stop(f"상태 파일 current.{k} = {h} 가 기록에 없거나 종류가 다르다", EXIT_KEY)
    return s


def save_state(s: dict, state_dir: str = STATE_DIR) -> None:
    state_check_dir(state_dir, state_dir == STATE_DIR)
    tmp = os.path.join(state_dir, f".{STATE_FILE}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, os.path.join(state_dir, STATE_FILE))


def current_template(s: dict, role: str) -> str:
    k = kind_of_role(role)
    h = s["current"].get(k)
    if not h:
        raise Stop(f"상태 파일에 지금 {k} 템플릿이 없다 — publish 하거나 설치 때 넣는다", EXIT_KEY)
    return h


# ── Vast 템플릿 읽기·대조 ──

def env_values(env: str, name: str) -> list:
    return re.findall(r"(?:^|\s)-e\s+" + name + r"=(\S*)", env or "")


def image_ref(t: dict) -> str:
    img, tag = str(t.get("image") or ""), str(t.get("tag") or "")
    if "@sha256:" in img or not tag:
        return img
    return img + ("@" if tag.startswith("sha256:") else ":") + tag


def get_template(tkey: str, h: str) -> dict:
    j = vast(tkey, "GET", "/v0/template/", query={"select_filters": {"hash_id": {"eq": h}}})
    rows = [t for t in (j.get("templates") or []) if isinstance(t, dict) and t.get("hash_id") == h]
    if len(rows) != 1:
        raise Stop(f"템플릿 {h} 를 API 로 {len(rows)}개 읽었다(1개여야 한다)", EXIT_REFUSE)
    return rows[0]


def template_facts(t: dict) -> dict:
    env = str(t.get("env") or "")
    one = {}
    for n in ("PS", "SB", "PU", "CF"):
        v = env_values(env, n)
        one[n] = v[0] if len(v) == 1 else (None if not v else "<여럿>")
    args = t.get("args_str")
    return {"env": env, "ps": one["PS"], "sb": one["SB"], "sb_sha": sha256s(one["SB"]) if one["SB"] and one["SB"] != "<여럿>" else None,
            "args_sha": sha256s(args) if isinstance(args, str) else None, "pu": one["PU"], "cf": one["CF"],
            "image": image_ref(t), "holes": _HOLE.findall(env)}


# 파드·quant 묶음 빌더가 만드는 Args 꼴(boot 의 sha256 만 다르다). 콘솔로 만든 템플릿의 Args 를 SB 로부터 다시 만들어 견준다
ARGS_FMT = ("bash -c 'echo \"$SB\"|base64 -d>/dev/shm/b&&echo \"{sha}  /dev/shm/b\"|sha256sum -c"
            "&&exec bash /dev/shm/b;echo HALT: boot SB;sleep 1d'")


def expected_args(sb) -> str | None:
    """env SB(boot 스크립트 base64)로 빌더가 만들었을 Args 글. SB 가 없거나 깨졌으면 None."""
    import base64
    import binascii
    if not isinstance(sb, str) or not sb or sb == "<여럿>":
        return None
    try:
        boot = base64.b64decode(sb, validate=True)
    except (binascii.Error, ValueError):
        return None
    return ARGS_FMT.format(sha=hashlib.sha256(boot).hexdigest())


def args_argv(a) -> list | None:
    """Args 를 실행 인자 목록으로: JSON 글 목록이면 그대로, 글이면 POSIX 셸 낱말 나누기. 못 나누면 None."""
    if isinstance(a, list):
        return a if all(isinstance(x, str) for x in a) else None
    if not isinstance(a, str):
        return None
    st = a.strip()
    if st.startswith("["):
        try:
            j = json.loads(st)
        except ValueError:
            j = None
        if isinstance(j, list) and all(isinstance(x, str) for x in j):
            return j
    try:
        return shlex.split(a, posix=True)
    except ValueError:
        return None


_ARGS_SQ = re.compile(r"[ \t\r\n]*bash[ \t]+-c[ \t]+'([^']*)'[ \t\r\n]*")


def args_canonical(a) -> str | None:
    """빌더 꼴 "bash -c '<글>'" 로 맞춘 Args, 아니면 None. 받는 표기 차이는 셋뿐(실행 인자 목록이 같은 것):
    바깥 공백·탭·CR·LF, bash·-c·글 사이 공백 수, JSON 목록 ["bash", "-c", 글]. 작은따옴표 안의 글은 한 글자도 바꾸지 않는다.
    큰따옴표·역빗금 표기는 셸마다 펼침이 달라 받지 않는다."""
    if isinstance(a, str):
        st = a.strip(" \t\r\n")
        if st.startswith("["):
            a = args_argv(st)
        else:
            m = _ARGS_SQ.fullmatch(a)
            return f"bash -c '{m.group(1)}'" if m else None
    if isinstance(a, list) and len(a) == 3 and a[:2] == ["bash", "-c"] and isinstance(a[2], str) and "'" not in a[2]:
        return f"bash -c '{a[2]}'"
    return None


def _cls(text: str) -> str:
    """글자 무리를 내용 없이 꼴로: 공백·탭·CR·LF·따옴표·역빗금 개수와 그 밖 글자 수."""
    parts = []
    for name, ch in (("공백", " "), ("탭", "\t"), ("CR", "\r"), ("LF", "\n"), ('큰따옴표', '"'), ("작은따옴표", "'"), ("역빗금", "\\")):
        n = text.count(ch)
        if n:
            parts.append(f"{name}{n}")
    other = len(text) - sum(text.count(c) for c in (" ", "\t", "\r", "\n", '"', "'", "\\"))
    if other:
        parts.append(f"그밖{other}")
    return "+".join(parts) or "없음"


def args_shape(api, want: str | None) -> dict:
    """API Args 와 기대 Args 의 차이 꼴(본문 없이): 형, 길이, 바뀐 자리(앞·중간·끝)와 들어간·빠진 글자의 꼴, 인자 목록이 같은지."""
    import difflib
    out = {"api_type": type(api).__name__, "api_len": len(api) if isinstance(api, (str, list)) else None,
           "want_len": len(want) if want else None}
    if want is None:
        out["note"] = "SB 로 기대 Args 를 만들 수 없다"
        return out
    if not isinstance(api, str):
        out["same_argv"] = args_argv(api) == args_argv(want)
        return out
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, want, api, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        where = "앞" if i1 == 0 else ("끝" if i2 == len(want) else "중간")
        ops.append({"op": {"replace": "바뀜", "delete": "빠짐", "insert": "더해짐"}[tag], "where": where,
                    "removed": _cls(want[i1:i2]), "added": _cls(api[j1:j2])})
    out |= {"diffs": ops[:12], "same_argv": args_argv(api) == args_argv(want), "canonical_equal": args_canonical(api) == want}
    return out


def template_mismatch(rec: dict, t: dict) -> list:
    """기록(검증 통과한 묶음)과 API 로 읽은 템플릿이 다른 칸. 키가 없으면 그 칸 이름과 받은 키 목록(값은 아님)."""
    f, bad = template_facts(t), []
    if "env" not in t or "args_str" not in t:
        bad.append(f"API 응답에 env·args_str 칸이 없다(받은 키: {sorted(t)[:40]})")
    for k in ("ps", "sb_sha", "args_sha"):
        if f[k] != rec[k]:
            # 콘솔로 만든 기록만: Args 는 표기만 다르고 실행 인자 목록이 빌더 꼴과 같으면 같다고 본다(args_canonical).
            # publish 로 만든 템플릿은 래퍼가 보낸 글 그대로여야 한다(바이트 대조 그대로)
            if k == "args_sha" and rec.get("source") == "console":
                c = args_canonical(t.get("args_str"))
                if c is not None and sha256s(c) == rec["args_sha"]:
                    continue
            bad.append(k)
    if rec.get("image") and f["image"] != rec["image"]:
        bad.append("image")
    if f["holes"]:
        bad.append(f"빈칸 {f['holes'][:3]}")
    if not f["pu"] or not _GIST_RAW.match(f["pu"]):
        bad.append("PU(gist raw 주소 꼴)")
    return bad


def check_up_template(tkey: str, s: dict, role: str) -> str:
    """up 직전: 지금 템플릿을 API 로 읽어 기록과 같을 때만 그 해시를 돌려준다. 다르면 69(빌리지 않는다)."""
    h = current_template(s, role)
    bad = template_mismatch(s["templates"][h], get_template(tkey, h))
    if bad:
        raise Stop(f"템플릿 {h} 가 publish 기록과 다르다({', '.join(bad)}) — 빌리지 않는다. templates 로 보고 publish 로 다시 만든다",
                   EXIT_REFUSE)
    return h


# ── gist ──

def read_gh_token() -> str:
    return read_secret(GH_SERVICE, r"(github_pat_|ghp_)[A-Za-z0-9_]{20,250}", "GitHub gist 토큰")


def gh(token: str, method: str, gist_id: str, body: dict | None = None) -> dict:
    """GitHub gist API(GET·PATCH /gists/<id> 만). 토큰은 이 프로세스의 Authorization 헤더에만 넣는다."""
    if method not in ("GET", "PATCH") or not re.fullmatch(r"[0-9a-f]{20,40}", gist_id):
        raise Stop(f"허용하지 않은 gist 요청: {method}", EXIT_OTHER)
    req = urllib.request.Request(f"{GH_API}/gists/{gist_id}", data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                                         "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "stewart-cloud/1",
                                                         "Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=60) as r:
            return json.loads(r.read(8_000_000))
    except urllib.error.HTTPError as e:
        msg = ""
        try:
            msg = str(json.loads(e.read(200_000) or b"{}").get("message") or "")[:200]
        except Exception:
            pass
        raise Stop(f"GitHub {method} /gists/<id> → HTTP {e.code} {scrub(msg.replace(token, '<가림>'))}"
                   + (" (토큰 권한: Account permissions Gists read/write)" if e.code in (401, 403, 404) else ""),
                   EXIT_AUTH if e.code in (401, 403) else EXIT_OTHER)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise Stop(f"GitHub 에 연결하지 못했다({type(e).__name__})", EXIT_NET)


def fetch_gist_raw(url: str, tries: int = 10, wait_s: float = 3.0) -> str:
    """gist raw(토큰 없이 — 파드가 받는 것과 같은 길). 리비전 고정 주소는 처음 몇 초 404 일 수 있다."""
    if not re.match(r"^https://gist\.githubusercontent\.com/", url or ""):
        raise Stop("gist raw 주소 꼴이 아니다", EXIT_OTHER)
    last = ""
    for _ in range(tries):
        try:
            with _OPENER.open(urllib.request.Request(url, headers={"User-Agent": "stewart-cloud/1"}), timeout=30) as r:
                return r.read(B64_MAX + 100_000).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError) as e:
            last = type(e).__name__
        time.sleep(wait_s)
    raise Stop(f"gist raw 를 받지 못했다({last})", EXIT_NET)


def gist_put(token: str, owner: str, gist_id: str, fname: str, b64: str, ps: str) -> tuple:
    """묶음 gist 에 fname 을 더한다(같은 이름이 이미 있고 내용이 같으면 그대로 쓴다, 다르면 멈춤).
    (리비전 고정 raw 주소, 리비전) — 토큰 없이 받아 줄바꿈을 지운 sha256 이 PS 와 같을 때만."""
    g = gh(token, "GET", gist_id)
    if str((g.get("owner") or {}).get("login") or "").lower() != owner.lower():
        raise Stop("묶음 gist 의 주인이 바탕 템플릿 PU 의 사용자와 다르다", EXIT_REFUSE)
    have = (g.get("files") or {}).get(fname)
    if have is not None:
        body = have.get("content") if not have.get("truncated") else None
        body = body if isinstance(body, str) else fetch_gist_raw(str(have.get("raw_url") or ""))
        if sha256s(body.replace("\r", "").replace("\n", "")) != ps:
            raise Stop(f"gist 에 {fname} 가 이미 있고 내용이 다르다 — 덮지 않는다(묶음 번호를 올린다)", EXIT_REFUSE)
        say(f"gist 에 {fname} 가 이미 같은 내용으로 있다 — 그대로 쓴다")
    else:
        g = gh(token, "PATCH", gist_id, {"files": {fname: {"content": b64}}})
    rev = str(((g.get("history") or [{}])[0] or {}).get("version") or "")
    raw = str(((g.get("files") or {}).get(fname) or {}).get("raw_url") or "")
    want = re.compile(r"^https://gist\.githubusercontent\.com/" + re.escape(owner) + "/" + gist_id
                      + r"/raw/[0-9a-f]{40}/" + re.escape(fname) + "$", re.I)
    if not want.match(raw):
        raise Stop("gist 응답의 raw 주소가 리비전 고정 꼴이 아니다", EXIT_OTHER)
    got = fetch_gist_raw(raw)
    if sha256s(got.replace("\r", "").replace("\n", "")) != ps:
        raise Stop("gist 에서 받은 묶음의 sha256 이 PS 와 다르다 — 템플릿을 만들지 않는다", EXIT_REFUSE)
    say(f"gist 대조 통과: {fname}(리비전 {rev[:12] or '?'}) 를 토큰 없이 받아 sha256 = PS")
    return raw, rev


# ── 명령 ──

def do_publish(tkey: str, token: str, bundle: str, commit: str, ps: str, facts: dict, record_dir: str = RECORD_DIR,
               state_dir: str = STATE_DIR, labs: str = LABS_VAST) -> int:
    """facts = do_verify 결과(키를 읽기 전에 main 이 만든다). 여기서부터는 자식 프로세스 없이 HTTP 만."""
    kind = kind_of(bundle)
    if not facts or facts.get("ps") != ps:
        raise Stop("재현 대조 결과가 없다 — publish 는 verify 를 먼저 거친다", EXIT_OTHER)
    s = load_state(state_dir)
    base_h = s["current"].get(kind)
    if not base_h:
        raise Stop(f"바탕 템플릿(지금 {kind})이 상태 파일에 없다", EXIT_KEY)
    base = get_template(tkey, base_h)
    bad = template_mismatch(s["templates"][base_h], base)
    if "args_sha" in bad:
        # 바탕에서는 이미지·runtype·디스크·onstart·접속 칸·PU 의 gist·CF 만 베낀다. Args 는 다시 만든 것을 쓰고 만든 뒤 바이트로 대조한다
        bad.remove("args_sha")
        say(f"바탕 템플릿 {base_h} 의 Args 가 기록과 다르다 — publish 는 Args 를 베끼지 않으므로 계속한다(templates 의 args_diff 로 꼴을 본다)")
    if bad:
        raise Stop(f"바탕 템플릿 {base_h} 가 기록과 다르다({', '.join(bad)}) — 그 칸을 베끼지 않는다", EXIT_REFUSE)
    bf = template_facts(base)
    img = bf["image"]
    if not re.search(r"@sha256:[0-9a-f]{64}$", img):
        raise Stop("바탕 템플릿 이미지가 digest(@sha256:…) 고정이 아니다", EXIT_REFUSE)
    try:
        with open(os.path.join(labs, bundle, "image.txt"), encoding="utf-8") as f:
            want_img = f.read().strip()
        if want_img != img:
            raise Stop(f"labs/{bundle}/image.txt 가 바탕 템플릿 이미지와 다르다 — 이미지 바꾸기는 publish 가 하지 않는다", EXIT_REFUSE)
    except FileNotFoundError:
        pass
    m = _GIST_RAW.match(bf["pu"] or "")
    owner, gist_id = m.group(1), m.group(2)
    fill = {}
    if kind == "pod":                       # 설정 gist(CF)는 바탕 템플릿 것을 그대로(최신 raw). gist 에 올리기 전에 본다
        cf = bf["cf"]
        if not cf or cf == "<여럿>" or not re.match(r"^https://gist\.githubusercontent\.com/" + re.escape(owner) + r"/[0-9a-f]{20,40}/raw/", cf):
            raise Stop("바탕 템플릿의 CF(설정 gist 최신 raw 주소)가 없거나 꼴이 다르다", EXIT_REFUSE)
        if not re.search(r"(?m)^CTX=\S+", fetch_gist_raw(cf).replace("\r", "")):
            raise Stop("설정 gist(CF)에 CTX= 줄이 없다", EXIT_REFUSE)
        fill["CF"] = cf
    fname = f"{BUNDLE_OUT[kind][2]}{bundle.removeprefix('quant-')}.b64"
    pu, rev = gist_put(token, owner, gist_id, fname, facts["b64"], ps)
    fill["PU"] = pu
    env = facts["options"]
    for hole, name in BUNDLE_OUT[kind][3].items():
        if env.count(f"-e {name}={hole}") != 1:
            raise Stop(f"docker-options 의 {name} 빈칸이 하나가 아니다", EXIT_REFUSE)
        env = env.replace(f"-e {name}={hole}", f"-e {name}={fill[name]}")
    if _HOLE.search(env) or len(env) > OPTIONS_MAX:
        raise Stop(f"채운 Docker Options 에 빈칸이 남았거나 {OPTIONS_MAX}자를 넘는다({len(env)}자)", EXIT_REFUSE)
    body = {k: base[k] for k in TPL_COPY if base.get(k) is not None}
    body |= {"name": f"stewart {bundle}", "env": env, "args_str": facts["args"], "private": True,
             "desc": f"stewart-cloud publish {bundle} commit {commit[:12]} PS {ps[:12]}"}
    c = vast(tkey, "POST", "/v0/template/", body)
    h = str((c.get("template") or {}).get("hash_id") or "")
    if not c.get("success", True) or not _THASH.match(h):
        raise Stop(f"템플릿 만들기 응답이 성공이 아니다: {scrub(str(c.get('msg') or c.get('error') or '')[:200])}", EXIT_OTHER)
    t = get_template(tkey, h)
    tf, bad = template_facts(t), []
    for k, want in (("ps", ps), ("sb_sha", facts["sb_sha"]), ("args_sha", facts["args_sha"]), ("pu", pu),
                    ("cf", fill.get("CF")), ("image", img)):
        if tf[k] != want:
            bad.append(k)
    bad += [k for k in TPL_SAME if t.get(k) != base.get(k)]
    if tf["holes"]:
        bad.append(f"빈칸 {tf['holes'][:3]}")
    if bad:
        raise Stop(f"만든 템플릿 {h} 를 다시 읽으니 다르다({', '.join(bad)}) — 상태 파일을 바꾸지 않는다(지금 템플릿 그대로)", EXIT_REFUSE)
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    rec = {"kind": kind, "bundle": bundle, "commit": commit, "ps": ps, "sb_sha": facts["sb_sha"], "args_sha": facts["args_sha"],
           "image": img, "source": "publish", "ts": ts, "base": base_h}
    s = load_state(state_dir)                                          # 그사이 바뀐 것이 있으면 그 위에
    old = s["current"].get(kind)
    s["templates"][h] = rec
    s["current"][kind] = h
    save_state(s, state_dir)
    ev = {"event": "publish", "template_hash_id": h, "previous": old, **{k: rec[k] for k in ("kind", "bundle", "commit", "ps")},
          "gist_file": fname, "gist_revision": rev}
    record(ev, record_dir)
    say(f"템플릿 만듦·대조 통과: {h}(stewart {bundle}). 지금 {kind} 템플릿 {old} → {h}"
        + (" (serve·eval·eval2 공용)" if kind == "pod" else ""))
    emit(ev)
    return 0


def do_templates(tkey: str, state_dir: str = STATE_DIR) -> int:
    """기록한 템플릿마다 API 로 읽어 기록과 같은지(무료). 설치 뒤 확인·up 거부 원인 보기."""
    s = load_state(state_dir)
    bad_any = False
    for h, rec in s["templates"].items():
        cur = [k for k, v in s["current"].items() if v == h]
        diff = None
        try:
            t = get_template(tkey, h)
            bad = template_mismatch(rec, t)
            api = t.get("args_str")
            if not (isinstance(api, str) and sha256s(api) == rec["args_sha"]):   # 바이트가 다르면(같다고 본 것도) 꼴을 보인다
                want = expected_args(template_facts(t).get("sb"))
                diff = args_shape(api, want if want and sha256s(want) == rec["args_sha"] else None)
        except Stop as e:
            bad = [str(e)]
        bad_any |= bool(bad and cur)
        emit({"template_hash_id": h, "current": cur, "bundle": rec["bundle"], "source": rec["source"], "ps": rec["ps"],
              "ok": not bad, "mismatch": bad, **({"args_diff": diff} if diff else {})})
    say("지금 템플릿: " + ", ".join(f"{k} {v}" for k, v in s["current"].items()) + (" — 지금 템플릿이 기록과 달라 up 이 거부된다"
                                                                                    if bad_any else ""))
    return EXIT_REFUSE if bad_any else 0


CLEAN_ENV = {"PATH": "/usr/bin:/bin", "HOME": HOME_DIR}


def clean_environ() -> None:
    """부른 쪽 환경 변수를 모두 버린다(ssl·urllib 를 쓰기 전에). 진입 스크립트의 env -i 를 --clean-env 로 비켜 불러도
    SSL_CERT_FILE·SSL_CERT_DIR(가짜 CA)·HTTPS_PROXY 같은 값이 남지 않게 한다. -I 는 PYTHON* 만 무시한다(메인 검토 11/8 14:1x)."""
    os.environ.clear()
    os.environ.update(CLEAN_ENV)


def main(argv: list | None = None, key_fn=read_key, config_path: str = CONFIG, record_dir: str = RECORD_DIR,
         tkey_fn=read_template_key, gh_fn=read_gh_token, state_dir: str = STATE_DIR, git_dir: str = REPO_GIT,
         labs: str = LABS_VAST) -> int:
    clean_environ()
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        cmd, args = parse_args(argv)
        if cmd == "verify":                 # 키·망·설정 없이(무료, 승인 없음)
            do_verify(args[0], args[1], args[2], git_dir, labs)
            return 0
        cfg = load_config(config_path)
        cmd, args = parse_args(argv, cfg)
        if cmd == "up":                     # 상태 파일이 틀렸으면 키를 읽거나 API 를 부르기 전에 멈춘다
            current_template(load_state(state_dir), args[0])
        elif cmd in ("publish", "templates"):
            load_state(state_dir)
        facts = None
        if cmd == "publish":                # 다시 만들기(자식 프로세스)는 키를 읽기 전에 끝낸다 — 키가 메모리에 있는 동안 자식 없음
            facts = do_verify(args[0], args[1], args[2], git_dir, labs)
        key = key_fn()
        if cmd == "offers":
            return do_offers(key, cfg, record_dir, args[0], args[1], args[2])
        if cmd == "status":
            return do_status(key, cfg)
        if cmd == "logs":
            return do_logs(key, cfg, args[0], args[1])
        if cmd == "down":
            return do_down(key, cfg, args[0], record_dir)
        tkey = tkey_fn() or key
        if cmd == "up":
            return do_up(key, cfg, args[0], args[1], args[2], record_dir, plan=args[3], tkey=tkey, state_dir=state_dir)
        if cmd == "templates":
            return do_templates(tkey, state_dir)
        return do_publish(tkey, gh_fn(), args[0], args[1], args[2], facts, record_dir, state_dir, labs)
    except Stop as e:
        sys.stderr.write(f"{PROG}: 중단 — {e}\n")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
