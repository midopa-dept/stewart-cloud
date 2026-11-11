"""Vast.ai 인스턴스 안 자동 종료 감시기(표준 라이브러리만). 인스턴스 범위 키(CONTAINER_API_KEY)로 자기 인스턴스만 멈춘다.
계정 전체 API 키는 두지 않는다. 키는 pod-start.sh 가 /dev/shm/vast-key(600)로 옮겨 환경에서 지운 것을 읽는다.

- 유휴: vLLM /metrics 의 요청·토큰 카운터가 IDLE_MIN(기본 30)분 동안 그대로이고 running·waiting 이 0
- 기동 유예: /metrics 가 아직 안 뜨면 BOOT_GRACE_MIN(기본 45)분까지 기다린다(첫 부팅 다운로드·재양자화)
- 하루 상한: 켜진 지 MAX_HOURS(기본 8)시간이면 멈춘다
- 정지(PUT console.vast.ai/api/v0/instances/<CONTAINER_ID>/ {"state":"stopped"})가 성공했을 때만 곧바로 tailscale logout(이름 비우기).
  stop 은 디스크를 남긴다(디스크 요금 계속). destroy 는 사람이 한다.
  실패하면 logout하지 않고 5분마다 다시 시도하며 /status 의 stop_failed 에 사유·횟수를 싣는다. 기록은 <state-dir>/pod-log.tsv
- 1분마다 nvidia-smi 표본을 뜨고(/status 의 gpu·gpu_ts·gpu_age_s), 5분마다 <state-dir>/gpu.tsv 에 남긴다(T24: 5분 표본이 낡은 채
  /status 에 실려 E13 14:04:46 사본에서 막 뜬 차선 1~3 의 memory.used 가 0 으로 보였다)
- v9(T24): 쉬다 정지 기본은 역할별(serve 90분 — 사람 작업, eval 30분). /status 에 idle_left_s. home 출처 모델을 받는 동안
  ($S/model-fetch.json 의 state 가 waiting·receiving)은 기동 유예를 세지 않는다(받기 끝부터 센다). /status 의 model_fetch = 그 파일.
- 읽기 전용 상태 응답 127.0.0.1:8001 /status (JSON: 켜진 시간, 유휴 시간, GPU 표본, pod-log 끝, vLLM 로그의 KV 캐시 줄)
- v7(T20) 차선: --role serve|eval, --lanes N. 차선 k 는 vLLM 127.0.0.1:80k0, 상태 127.0.0.1:80k1(k=0 이 8001).
  유휴는 모든 차선 /metrics 의 합으로 본다(한 차선이라도 일하면 유휴 아님).
  차선마다 /health 가 나면 짧은 응답 확인(/v1/models 의 id·root, 짧은 생성 하나)을 하고,
  통과한 차선만 ready(차선 0 은 품질 점검 exit=0 까지, k≥1 은 모델 root·served id·설정 sha 가 차선 0 과 같아야).
  8001 은 v6 꼴(차선 0 보기) + role·lanes 요약, 80k1 은 차선 k 보기(gpu = 카드 k, vllm_kv·verify = 차선 k 로그, config 에 lane·port).
- POST 127.0.0.1:8001/logout: 테일넷에서 바로 빠진다(v6, stewart-cloud down 이 Destroy 전에 부름). 한 번만, 본문 무시, 인증 없음.
  테일넷 ACL·내 Mac 샌드박스에서 8001을 열어야 내 Mac가 읽는다. 비밀값은 넣지 않는다.
- v12(T28) 차선 기동 상한: pod-start 가 차선마다 lanes/start-k(띄운 시각)·pid-k·rc-k(서버가 끝난 종료 코드)를 남긴다.
  health 없이 차선 k 의 서버가 끝났거나(rc-k), 띄운 뒤 LANE_BOOT_MIN(기본 30, 차선 0 은 기동 유예와 같은 45)분이 지나도 health 가
  없으면 그 차선만 state=failed, why = 이유 + 차선 로그 끝 3줄(lanes/failed-k 에도 적는다 — pod-start 가 다음 차선을 바로 띄운다).
  기동 상한을 넘긴 서버는 SIGTERM(자식까지). home 받기 동안(hold)은 세지 않는다. 다른 차선은 계속.
  차선 0 이 실패하면 띄우지 않은 차선도 failed. 모든 차선이 failed 면 HALT: 이유 한 줄 → 900초 → 정지(boot.sh 와 같은 꼴).
  python idle_stop.py --state-dir /workspace/stewart [--role serve|eval] [--lanes N] [--dry-run] [--once]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request
from datetime import datetime, timezone

ACTIVITY = ("vllm:request_success_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total")
BUSY = ("vllm:num_requests_running", "vllm:num_requests_waiting")


SPEC_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
_POS = re.compile(r'position="([0-9]{1,2})"')


class QuickHTTPServer(ThreadingHTTPServer):
    """server_bind 가 부르는 socket.getfqdn(역방향 DNS)을 건너뛴다 — 느린 기계(macOS CI)에서 서버 하나에 수십 초가 걸려
    bind 와 listen 사이에 붙은 요청이 기다렸다. server_name 은 주소 그대로다(이 서버들은 그 값을 쓰지 않는다)."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), port


def parse_metrics(text: str) -> dict[str, float]:
    """이름별 합(라벨이 다른 줄은 더한다). 추측 디코딩의 자리별 수락 수는 "<이름>[자리]" 로도 따로 둔다."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[name] = out.get(name, 0.0) + val
        if name == SPEC_POS:
            m = _POS.search(line)
            if m:
                key = f"{name}[{int(m.group(1))}]"
                out[key] = out.get(key, 0.0) + val
    return out


def spec_view(m: dict | None) -> dict | None:
    """차선 /metrics 의 추측 디코딩 누계(서버가 뜬 뒤부터). 평가 한 번의 값은 시작·끝 /status 사본의 차로 낸다.
    acceptance = 수락 토큰 / 초안 토큰, mean_accept_len = 1 + 수락 토큰 / 초안 수(검증 한 번에 나오는 토큰, 보너스 포함),
    per_pos[i] = i 번째 초안 자리의 수락 비율. 추측 디코딩이 꺼진 차선(카운터 없음)은 None."""
    if not m or "vllm:spec_decode_num_drafts_total" not in m:
        return None
    drafts = m.get("vllm:spec_decode_num_drafts_total", 0.0)
    dtok = m.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    acc = m.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    pos = sorted((int(k[len(SPEC_POS) + 1:-1]), v) for k, v in m.items() if k.startswith(SPEC_POS + "["))
    return {"drafts": int(drafts), "draft_tokens": int(dtok), "accepted_tokens": int(acc),
            "acceptance": round(acc / dtok, 4) if dtok else None,
            "mean_accept_len": round(1 + acc / drafts, 3) if drafts else None,
            "per_pos": [round(v / drafts, 4) if drafts else None for _, v in pos]}


class Watch:
    def __init__(self, idle_s: float, boot_grace_s: float, max_s: float, now: float):
        self.idle_s, self.boot_grace_s, self.max_s = idle_s, boot_grace_s, max_s
        self.t_start = now
        self.t_boot = now           # 기동 유예를 세는 시작(home 받기 동안 hold 로 미룬다 — T24)
        self.last_change = now
        self.last_sig: tuple | None = None
        self.seen_metrics = False

    def observe(self, metrics: dict[str, float] | None, now: float) -> str | None:
        """멈출 이유(문자열) 또는 None."""
        if now - self.t_start >= self.max_s:
            return f"max_uptime {self.max_s / 3600:.1f}h"
        if metrics is None:
            if not self.seen_metrics and now - self.t_boot >= self.boot_grace_s:
                return f"boot_timeout {self.boot_grace_s / 60:.0f}m"
            if self.seen_metrics and now - self.last_change >= self.idle_s:
                return f"metrics_down_idle {self.idle_s / 60:.0f}m"
            return None
        if not self.seen_metrics:
            self.seen_metrics = True
            self.last_change = now
        sig = tuple(metrics.get(k, 0.0) for k in ACTIVITY)
        busy = sum(metrics.get(k, 0.0) for k in BUSY)
        if sig != self.last_sig or busy > 0:
            self.last_sig = sig
            self.last_change = now
            return None
        if now - self.last_change >= self.idle_s:
            return f"idle {self.idle_s / 60:.0f}m"
        return None


    def hold(self, now: float) -> None:
        """모델을 받는 중: 기동 유예를 지금부터 다시 센다(받기 자체의 상한 60분은 modelrecv 가 HALT 로 지킨다)."""
        self.t_boot = now

    def idle_left(self, now: float) -> float | None:
        """쉬다 정지까지 남은 초(서버를 한 번도 못 봤으면 None)."""
        if not self.seen_metrics:
            return None
        return max(0.0, self.idle_s - (now - self.last_change))


IDLE_MIN_BY_ROLE = {"serve": 90.0, "eval": 30.0}   # T24: serve 는 사람 작업(사용자 private 정리)이라 쉬는 틈이 길다
FETCH_KEYS = ("alias", "digest", "state", "files_total", "files_done", "bytes_total", "bytes_done", "rate_Bps",
              "current", "started", "updated")       # modelrecv 의 진행 파일 칸
FETCH_HOLD = ("waiting", "receiving")


def model_fetch(state_dir: str) -> dict | None:
    """$S/model-fetch.json에서 정해진 칸만. 없거나 깨졌으면 None."""
    try:
        with open(os.path.join(state_dir, "model-fetch.json"), encoding="utf-8") as f:
            j = json.loads(f.read(65536))
    except (OSError, ValueError):
        return None
    if not isinstance(j, dict):
        return None
    out = {}
    for k in FETCH_KEYS:
        v = j.get(k)
        if isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:200]
    return out


def fetch(url: str, timeout: float = 10) -> dict[str, float] | None:
    try:
        with LOCAL.open(url, timeout=timeout) as r:
            return parse_metrics(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def log(state_dir: str, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t{msg}"
    print(line, flush=True)
    with open(os.path.join(state_dir, "pod-log.tsv"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _logout(state_dir: str) -> None:
    ts = os.path.join(state_dir, "bin", "tailscale")
    if os.path.exists(ts):
        try:
            subprocess.run([ts, "--socket=/tmp/tailscaled.sock", "logout"], timeout=30)
            log(state_dir, "tailscale logout")
        except (OSError, subprocess.TimeoutExpired) as e:
            log(state_dir, f"tailscale logout 실패: {type(e).__name__}")


KEY_FILE = "/dev/shm/vast-key"
# 파드 안 127.0.0.1(vLLM·차선 상태)로 가는 요청은 환경·시스템 프록시를 타지 않는다(macOS urllib 은 시스템 프록시도 따른다).
# 밖으로 가는 Vast API 요청은 urlopen 그대로.
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))
API = "https://console.vast.ai/api/v0/instances/{}/"


def _key() -> str | None:
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return os.environ.get("CONTAINER_API_KEY") or None


def request_stop(state_dir: str) -> tuple[bool, str]:
    """Vast 정지 요청(인스턴스 범위 키). (성공 여부, 설명). 키 값은 남기지 않는다."""
    cid, key = os.environ.get("CONTAINER_ID"), _key()
    if not (cid and key):
        return False, f"api 건너뜀(CONTAINER_ID {'있음' if cid else '없음'}, 키 {'있음' if key else '없음'})"
    req = urllib.request.Request(API.format(cid), data=json.dumps({"state": "stopped"}).encode(), method="PUT",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read(2000)
        try:
            ok = json.loads(body).get("success") is True
        except ValueError:
            ok = False
        return ok, f"api HTTP {r.status} success={ok}"
    except Exception as e:
        return False, f"api {type(e).__name__} {getattr(e, 'code', '')}".strip()


def stop_pod(state_dir: str, reason: str, dry_run: bool, failed: dict | None = None) -> bool:
    """정지를 먼저 요청하고, 성공했을 때만 곧바로 tailscale logout(이름 비우기).
    실패하면 logout하지 않는다(테일넷에 남아 /status 로 실패가 보이게). failed 에 사유·횟수를 쌓는다."""
    log(state_dir, f"stop reason={reason} instance={os.environ.get('CONTAINER_ID') or '?'} dry_run={dry_run}")
    if dry_run:
        return True
    ok, detail = request_stop(state_dir)
    log(state_dir, f"stop {'ok' if ok else 'FAILED'}: {detail}")
    if ok:
        _logout(state_dir)
        return True
    if failed is not None:
        failed["reason"] = reason
        failed["count"] = failed.get("count", 0) + 1
        failed["last_error"] = detail
        failed["last_ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return False


KV_LINE = re.compile(r"KV cache|Maximum concurrency|max_model_len|model weights took|Available KV|CUDA out of memory", re.I)


# 전력·클럭 기록(호스트 전력 상한 확인용). 새 드라이버는 clocks_event_reasons, 옛 드라이버는 clocks_throttle_reasons.
GPU_FIELDS = ["memory.used", "memory.total", "utilization.gpu", "power.draw", "power.limit",
              "clocks.sm", "clocks.mem", "clocks_event_reasons.active"]


GPU_SAMPLE_S, GPU_TSV_S = 60, 300   # /status 표본은 1분마다(평가기 GpuSampler 도 1분), 파일 기록은 5분마다
SPEC_TSV_S = 60                     # T35(H5): 차선별 추측 디코딩 누계를 1분마다(바뀐 차선만) spec-log.tsv 에


def plan_rows(state_dir: str) -> dict[int, tuple[str, str]]:
    """$S/lanes/plan.txt("k 포트 카드 별칭 어댑터 quality") → {k: (별칭, 어댑터)}. 없으면 빈 것."""
    out = {}
    try:
        for ln in open(os.path.join(state_dir, "lanes", "plan.txt"), encoding="utf-8"):
            f = ln.split()
            if len(f) >= 5 and f[0].isdigit():
                out[int(f[0])] = (f[3], f[4])
    except OSError:
        pass
    return out


def spec_tsv(state_dir: str, st: list[dict], rows: dict, last: dict, ts: str) -> int:
    """차선 = vLLM 프로세스 하나. 그 차선의 누계(서버가 뜬 뒤부터)가 바뀌었으면 한 줄:
    시각 차선 별칭 어댑터 drafts draft_tokens accepted_tokens per_pos(쉼표, 자리별 수락 비율). 조각(칸) 하나의 값은 그 조각의
    시작·끝 시각 앞뒤 줄의 차로 낸다(평가 runner.log 의 시각). 서버가 다시 뜨면 누계가 0 부터 — 줄어든 줄이 그 경계다."""
    n = 0
    with open(os.path.join(state_dir, "spec-log.tsv"), "a", encoding="utf-8") as f:
        for s in st:
            sd = s.get("spec_decode")
            if not sd:
                continue
            key = (sd["drafts"], sd["draft_tokens"], sd["accepted_tokens"])
            if last.get(s["k"]) == key:
                continue
            last[s["k"]] = key
            al, ad = rows.get(s["k"], ("?", "?"))
            pp = ",".join("" if v is None else str(v) for v in sd.get("per_pos") or [])
            f.write(f"{ts}\t{s['k']}\t{al}\t{ad}\t{key[0]}\t{key[1]}\t{key[2]}\t{pp}\n")
            n += 1
    return n


def gpu_samples() -> list[str] | None:
    """카드마다 한 줄(nvidia-smi 번호 순 = CUDA_DEVICE_ORDER=PCI_BUS_ID 의 CUDA 번호 순)."""
    for fields in (GPU_FIELDS, GPU_FIELDS[:-1] + ["clocks_throttle_reasons.active"]):
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()
    return None


def gpu_sample() -> str | None:
    """v6 꼴(첫 카드 한 줄)."""
    g = gpu_samples()
    return g[0] if g else None


# ── 차선(v7) ─────────────────────────────────────────────────────

MAX_LANES = 8


def lane_port(k: int) -> int:
    return 8000 + 10 * k


def lane_status_port(k: int) -> int:
    return 8001 + 10 * k


def lane_log(state_dir: str, k: int) -> str:
    return os.path.join(state_dir, "vllm.log" if k == 0 else f"vllm-{k}.log")


def combine(ms: list) -> dict[str, float] | None:
    """차선들의 /metrics 를 합친다. 하나도 못 읽으면 None(기동 중이거나 다 죽음)."""
    got = [m for m in ms if m is not None]
    if not got:
        return None
    out: dict[str, float] = {}
    for m in got:
        for k, v in m.items():
            out[k] = out.get(k, 0.0) + v
    return out


def _http(url: str, body: dict | None = None, timeout: float = 30) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with LOCAL.open(req, timeout=timeout) as r:
            raw = r.read(2_000_000)
            code = r.status
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return 0, type(e).__name__
    try:
        return code, json.loads(raw or b"null")
    except ValueError:
        return code, raw[:200].decode("utf-8", "replace")


SMOKE_PROMPT = "Reply with exactly one word: ok"


def smoke(port: int, timeout: float = 60) -> dict:
    """짧은 응답 확인: /v1/models 의 첫 모델(id·root) + 짧은 생성 하나(온도 0, 생각 끔). ok 는 응답이 비지 않았는지."""
    base = f"http://127.0.0.1:{port}"
    code, j = _http(base + "/v1/models", timeout=timeout)
    if code != 200 or not isinstance(j, dict) or not j.get("data"):
        return {"ok": False, "error": f"models HTTP {code}"}
    m0 = j["data"][0]
    mid, root = m0.get("id"), m0.get("root")
    code, j = _http(base + "/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": SMOKE_PROMPT}],
                                                    "max_tokens": 16, "temperature": 0,
                                                    "chat_template_kwargs": {"enable_thinking": False}}, timeout=timeout)
    try:
        text = (j["choices"][0]["message"].get("content") or "").strip()
    except (TypeError, KeyError, IndexError):
        return {"ok": False, "id": mid, "root": root, "error": f"chat HTTP {code}"}
    return {"ok": bool(text), "id": mid, "root": root, "text": text[:40], "error": None if text else "빈 응답"}


def config_sha(cfg: dict | None, lane_env) -> str:
    """같은 모델·같은 설정인지 견줄 값: 이번 부팅 설정(run-config.json) + 그 차선의 env(모델·어댑터·tp) sha256 앞 16자."""
    return hashlib.sha256(json.dumps([cfg, lane_env], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def lane_env(state_dir: str, k: int) -> dict | None:
    """차선 k 의 env(serving.py 가 쓴 KEY='값' 줄). eval = lanes/lane-k.env, serve 차선 0 = model.env. 없으면 None."""
    for p in [os.path.join(state_dir, "lanes", f"lane-{k}.env")] + ([os.path.join(state_dir, "model.env")] if k == 0 else []):
        try:
            with open(p, encoding="utf-8") as f:
                env = {}
                for ln in f.read().splitlines():
                    key, _, v = ln.partition("=")
                    if re.fullmatch(r"M_[A-Z_]{1,24}", key):
                        env[key] = v.strip("'")[:300]
                return env
        except OSError:
            continue
    return None


def plan_lines(state_dir: str) -> dict:
    """lanes/plan.txt 의 "k 포트 카드 별칭 어댑터 quality" → {k: {cards, quality}}. 없으면 {}."""
    out = {}
    try:
        with open(os.path.join(state_dir, "lanes", "plan.txt"), encoding="utf-8") as f:
            for ln in f.read().splitlines():
                p = ln.split()
                if len(p) == 6 and p[0].isdigit():
                    out[int(p[0])] = {"cards": p[2], "quality": p[5] == "1"}
    except OSError:
        pass
    return out


CONFIG_FIELDS = ("lane", "port", "cards", "model", "adapter", "served_name", "model_rev", "tp", "image", "patchset", "plan", "spec")


def lane_meta(state_dir: str, k: int, role: str | None) -> dict:
    """차선 표시: lane·port·cards·model(표 별칭)·adapter·served_name·model_rev·tp·image·patchset·plan
    + quality(품질 차선인지)·samples(평가기 GpuSampler 의 gpu 표본이 어느 카드인지 — tp>1 이면 첫 카드뿐, 결정 T21b 4).
    evalq(v12.11d)는 lanes[k].alias(= model) 또는 lanes[k].config.model 로 차선을 고른다."""
    e = lane_env(state_dir, k) or {}
    pl = plan_lines(state_dir).get(k, {})
    cards = pl.get("cards") or ("0" if role == "serve" and k == 0 else None)
    try:
        tp = int(e["M_TP"]) if e.get("M_TP") else None
    except ValueError:
        tp = None
    first = str(cards).split(",")[0] if cards else str(k)
    return {"lane": k, "port": lane_port(k), "cards": cards, "model": e.get("M_ALIAS"), "adapter": e.get("M_ADAPTER"),
            "served_name": e.get("M_SERVED"), "model_rev": e.get("M_REV"), "tp": tp, "image": e.get("M_IMAGE"),
            "patchset": e.get("M_PATCHSET") if e.get("M_ADAPTER") == "vllm-patched" else ("" if e else None),
            "plan": (_config(state_dir) or {}).get("PLAN") or "", "spec": lane_spec(e, _config(state_dir)),
            "quality": pl.get("quality", k == 0), "samples": f"card{first}"}


def lane_spec(e: dict, cfg: dict | None) -> str | None:
    """차선이 실제로 쓰는 추측 디코딩: 계획의 차선 값 → (hyperqwen) 설정 SPEC → 표 값. 차선 env 가 없으면 None."""
    if not e:
        return None
    if e.get("M_LANE_SPEC"):
        return e["M_LANE_SPEC"]
    if e.get("M_ADAPTER") == "hyperqwen-8cf642e":
        return str((cfg or {}).get("SPEC") or e.get("M_HQ_SPEC") or "") or None
    return e.get("M_SPEC") or "off"


def _quality_logs(state_dir: str, k: int | None) -> list[str]:
    """품질 로그 이름(오래된 것부터). k 를 주면 그 차선 것(quality-L<k>-…), None 이면 차선 표시가 없는 것(serve·v6 꼴)."""
    try:
        names = sorted(n for n in os.listdir(state_dir) if n.startswith("quality-") and n.endswith(".log"))
    except OSError:
        return []
    if k is None:
        return [n for n in names if not re.match(r"quality-L[0-9]-", n)]
    return [n for n in names if n.startswith(f"quality-L{k}-")]


def quality_state(state_dir: str, cfg: dict | None, k: int | None = None) -> str:
    """품질 점검: ok | failed | pending. 이번 부팅의 마지막 로그에 exit= 가 있으면 그 결과, 로그가 돌고 있으면 pending,
    로그가 없으면 표시 파일(lanes/quality-ok-<k>, 앞 부팅에서 통과)이면 ok. QUALITY=0 이면 ok."""
    if cfg and str(cfg.get("QUALITY")) == "0":
        return "ok"
    logs = _quality_logs(state_dir, k)
    if k == 0 and not logs:
        logs = _quality_logs(state_dir, None)          # serve 차선 0 은 차선 표시 없는 이름
    if logs:
        ex = [l for l in _tail(os.path.join(state_dir, logs[-1]), 6, re.compile(r"^exit=")) if l.startswith("exit=")]
        if ex:
            return "ok" if ex[-1] == "exit=0" else "failed"
        return "pending"
    if os.path.exists(os.path.join(state_dir, "lanes", f"quality-ok-{k if k is not None else 0}")):
        return "ok"
    return "pending"


SMOKE_TRIES = 5
LANE_BOOT_MIN = 30.0            # T28: 차선 k≥1 기동 상한(띄운 뒤 health 까지). 차선 0 은 --boot-grace-min(45)
HALT_WAIT_S = 900               # 모든 차선 실패 → 이유를 남기고 이만큼 뒤 정지(boot.sh·pod-start halt 와 같다)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _read(path: str, n: int = 64) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(n).strip()
    except OSError:
        return None


def lane_tail(state_dir: str, k: int, n: int = 3) -> str:
    """차선 로그 끝 n 줄(빈 줄·색 코드 뺌, 줄마다 200자) — 기동이 어디서 멈췄는지 why 에 싣는다(T28 E14)."""
    try:
        with open(lane_log(state_dir, k), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return "(차선 로그 없음)"
    lines = [_ANSI.sub("", l).strip() for l in text.replace("\r", "\n").splitlines()]
    lines = [l[:200] for l in lines if l]
    return " ⏎ ".join(lines[-n:]) or "(차선 로그 비었음)"


def kill_tree(pid: int) -> list[int]:
    """pid 와 그 자손에 SIGTERM(자손 먼저). /proc/<pid>/task/<pid>/children 로 찾는다. 보낸 pid 목록."""
    def kids(p):
        out = []
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{t}/children") as f:
                    out += [int(x) for x in f.read().split()]
        except (OSError, ValueError):
            pass
        return out
    order, todo = [], [pid]
    while todo and len(order) < 256:
        p = todo.pop()
        order.append(p)
        todo += kids(p)
    sent = []
    for p in reversed(order):
        try:
            os.kill(p, signal.SIGTERM)
            sent.append(p)
        except OSError:
            pass
    return sent


def lane_proc_ok(pid: int) -> bool:
    """pid 가 아직 차선 서버(vllm·serving.py·HyperQwen 스크립트)인지 — 다른 프로세스가 같은 번호를 받았으면 건드리지 않는다."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read()
    except OSError:
        return False
    return any(w in cmd for w in (b"vllm", b"serving.py", b"/app/docker/entrypoint.sh", b"start_qwen.sh"))


class Lanes:
    """차선 상태. check() 는 감시 고리에서 부른다(이 프로세스 하나만 쓴다, 읽기는 상태 응답 쓰레드).
    ready(v8) = 지금 health + 짧은 응답 확인 통과 + 서빙 이름이 차선 env 의 M_SERVED 와 같음 + 품질 ok.
    품질은 quality 차선이면 제 것, 아니면 같은 별칭·어댑터의 첫 quality 차선 것(없으면 같은 별칭, 그것도 없으면 짧은 응답 확인만).
    차선 env 를 못 읽으면(v7 꼴) 차선 0 과 모델 id·root·설정 sha 가 같아야 하고 차선 0 품질을 본다."""

    def __init__(self, state_dir: str, n: int, role: str, smoke_fn=smoke, health_fn=None,
                 lane_boot_s: float = LANE_BOOT_MIN * 60, lane0_boot_s: float = 45 * 60, now_fn=time.time,
                 kill_fn=None):
        if not 1 <= n <= MAX_LANES:
            raise ValueError("lanes 1~8")
        self.state_dir, self.n, self.role = state_dir, n, role
        self.lane_boot_s, self.lane0_boot_s, self.now_fn = lane_boot_s, lane0_boot_s, now_fn
        self.kill_fn = kill_fn or (lambda pid: kill_tree(pid) if lane_proc_ok(pid) else [])
        self.hold_t = 0.0                # home 받기가 끝난 시각(이 뒤부터 센다)
        self.smoke_fn = smoke_fn
        self.health_fn = health_fn or (lambda port: _http(f"http://127.0.0.1:{port}/health", timeout=10)[0] == 200)
        self.lock = threading.Lock()
        self.st = [{"k": k, "port": lane_port(k), "status_port": lane_status_port(k), "state": "starting", "health": False,
                    "ready": False, "smoke": None, "tries": 0, "config_sha": None, "running": None, "waiting": None,
                    "why": None, "meta": {}, "spec_decode": None} for k in range(n)]

    def observe_metrics(self, ms: list) -> None:
        with self.lock:
            for s, m in zip(self.st, ms):
                s["running"] = None if m is None else m.get("vllm:num_requests_running", 0.0)
                s["waiting"] = None if m is None else m.get("vllm:num_requests_waiting", 0.0)
                if m is not None:                    # 서버가 잠깐 안 보여도 마지막 누계는 남긴다
                    s["spec_decode"] = spec_view(m)

    def quality_ref(self, k: int, metas: list) -> int | None:
        me = metas[k]
        if me.get("quality"):
            return k
        for same in (lambda m: m.get("model") == me.get("model") and m.get("adapter") == me.get("adapter")
                     and m.get("spec") == me.get("spec"),
                     lambda m: m.get("model") == me.get("model") and m.get("adapter") == me.get("adapter"),
                     lambda m: m.get("model") == me.get("model")):
            for j, m in enumerate(metas):
                if m.get("quality") and m.get("model") and same(m):
                    return j
        return None

    def hold(self, now: float | None = None) -> None:
        """home 받기 동안: 차선 기동 상한을 지금부터 다시 센다(벽시계 초)."""
        self.hold_t = self.now_fn() if now is None else now

    def _fail(self, k: int, msg: str) -> None:
        """차선 k 를 failed 로(잠금 안에서 부른다). why 에 차선 로그 끝 줄을 붙이고 lanes/failed-k 에도 적는다."""
        s = self.st[k]
        s["state"], s["ready"] = "failed", False
        s["why"] = f"{msg} — 로그 끝: {lane_tail(self.state_dir, k)}"[:900]
        try:
            with open(os.path.join(self.state_dir, "lanes", f"failed-{k}"), "w", encoding="utf-8") as f:
                f.write(s["why"] + "\n")
        except OSError:
            pass
        log(self.state_dir, f"lane {k} failed: {s['why']}")

    def check_boot(self) -> None:
        """T28: 차선마다 기동 상한·서버 끝남. health 가 없는 차선만 본다(뜬 뒤 죽은 것도 rc-k 로 failed)."""
        now = self.now_fn()
        ld = os.path.join(self.state_dir, "lanes")
        with self.lock:
            for k, s in enumerate(self.st):
                if s["state"] == "failed":
                    continue
                rc = _read(os.path.join(ld, f"rc-{k}"))
                start = _read(os.path.join(ld, f"start-{k}"))
                if rc is not None:
                    self._fail(k, f"차선 서버가 끝났다(rc={rc[:8]})")
                    continue
                if s["health"]:
                    continue
                if start is None:
                    if k > 0 and self.st[0]["state"] == "failed":
                        self._fail(k, "차선 0 이 실패해 띄우지 않았다")
                    continue
                try:
                    t0 = float(start)
                except ValueError:
                    continue
                lim = self.lane0_boot_s if k == 0 else self.lane_boot_s
                if now - max(t0, self.hold_t) >= lim:
                    pid = _read(os.path.join(ld, f"pid-{k}"))
                    sent = self.kill_fn(int(pid)) if pid and pid.isdigit() else []
                    self._fail(k, f"기동 {lim / 60:.0f}분 넘음(health 없음, 서버 {'SIGTERM ' + str(len(sent)) + '개' if sent else '정리 못 함'})")

    def all_failed(self) -> bool:
        with self.lock:
            return all(s["state"] == "failed" for s in self.st)

    def check(self) -> None:
        cfg = _config(self.state_dir)
        metas = [lane_meta(self.state_dir, k, self.role) for k in range(self.n)]
        envs = [lane_env(self.state_dir, k) for k in range(self.n)]
        for k in range(self.n):
            s = self.st[k]
            h = bool(self.health_fn(lane_port(k)))
            with self.lock:
                s["health"], s["meta"] = h, metas[k]
            if s["state"] in ("starting", "checking") and h and s["tries"] < SMOKE_TRIES:
                r = self.smoke_fn(lane_port(k))
                with self.lock:
                    s["tries"] += 1
                    s["smoke"] = r
                    s["config_sha"] = config_sha(cfg, envs[k] if envs[k] is not None else r.get("root"))
                    if r.get("ok"):
                        s["state"] = "checked"
                    elif s["tries"] >= SMOKE_TRIES:
                        s["state"], s["why"] = "failed", f"짧은 응답 확인 {SMOKE_TRIES}번 실패: {r.get('error')}"
                        log(self.state_dir, f"lane {k} failed: {s['why']}")
                    else:
                        s["state"] = "checking"
        qs = {}

        def q(j):
            if j not in qs:
                qs[j] = quality_state(self.state_dir, cfg, j)
            return qs[j]
        with self.lock:
            ref0 = self.st[0]
            for s in self.st:
                k, was = s["k"], s["ready"]
                if s["state"] not in ("checked", "ready"):
                    s["ready"] = False
                    continue
                if envs[k] is not None:
                    want = envs[k].get("M_SERVED")
                    same = bool(s["smoke"]) and (not want or s["smoke"].get("id") == want)
                    j = self.quality_ref(k, metas)
                    qk = "ok" if j is None else q(j)
                    ok = same and qk == "ok"
                    why = None if ok else (f"서빙 이름 {s['smoke'].get('id') if s['smoke'] else None} ≠ 차선 env {want}" if not same
                                           else f"품질 점검(차선 {j}) {qk}")
                else:                                                  # 차선 env 없음(v7 꼴): 차선 0 이 기준
                    qk = q(0)
                    if k == 0:
                        ok, why = qk == "ok", (None if qk == "ok" else f"품질 점검 {qk}")
                    else:
                        same = (ref0["state"] in ("checked", "ready") and ref0["smoke"] and s["smoke"]
                                and s["smoke"].get("id") == ref0["smoke"].get("id")
                                and s["smoke"].get("root") == ref0["smoke"].get("root")
                                and s["config_sha"] == ref0["config_sha"])
                        ok = bool(same) and qk == "ok"
                        why = None if ok else ("차선 0 과 모델·설정이 다르거나 차선 0 확인 전" if not same else f"차선 0 품질 점검 {qk}")
                s["why"] = why
                s["ready"] = bool(ok and s["health"])
                if ok and s["state"] == "checked":
                    s["state"] = "ready"
                if s["ready"] and not was:
                    log(self.state_dir, f"lane {k} ready port={s['port']} model={metas[k].get('model')} "
                                        f"adapter={metas[k].get('adapter')} cards={metas[k].get('cards')} sha={s['config_sha']}")
        self.check_boot()

    def summary(self, gpus: list | None) -> list:
        with self.lock:
            out = []
            for s in self.st:
                d = {k: s[k] for k in ("k", "port", "status_port", "state", "ready", "health", "running", "waiting",
                                       "config_sha", "why", "spec_decode")}
                meta = s["meta"] or lane_meta(self.state_dir, s["k"], self.role)
                d["alias"], d["quality"] = meta.get("model"), meta.get("quality")   # evalq(v12.11d)가 alias 를 읽는다
                d["config"] = {x: meta.get(x) for x in CONFIG_FIELDS + ("samples",)}
                d["smoke"] = {k: (s["smoke"] or {}).get(k) for k in ("ok", "id", "root", "text", "error")} if s["smoke"] else None
                d["gpu"], d["gpu_cards"] = lane_gpus(gpus, meta.get("cards"), s["k"])
                out.append(d)
            return out


def lane_gpus(gpus: list | None, cards, k: int) -> tuple:
    """(첫 카드 한 줄, 카드별 줄). 카드를 모르면 차선 번호 k 의 카드."""
    if not gpus:
        return None, None
    idx = [int(c) for c in str(cards).split(",") if c.isdigit()] if cards else [k]
    lines = [gpus[i] for i in idx if i < len(gpus)]
    return (lines[0] if lines else None), (lines or None)


def _config(state_dir: str) -> dict | None:
    """pod-start.sh 가 남긴 이번 부팅의 설정(허용 목록 값만)."""
    try:
        with open(os.path.join(state_dir, "run-config.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _tail(path: str, n: int, pat: re.Pattern | None = None) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    if pat:
        lines = [l for l in lines if pat.search(l)]
    return [l[:300] for l in lines[-n:]]


def status(state_dir: str, w: "Watch", now: float, last_gpu, stop_failed: dict | None = None,
           lanes: "Lanes | None" = None, k: int = 0, role: str | None = None, gpu_ts: float | None = None) -> dict:
    """차선 k 의 보기. last_gpu 는 v6 꼴 한 줄(str) 또는 카드별 줄 목록. k=0(8001)에는 role·lanes 요약을 더한다."""
    gpus = last_gpu if isinstance(last_gpu, list) else ([last_gpu] if last_gpu else None)
    cfg = _config(state_dir)
    meta = lane_meta(state_dir, k, role) if lanes is not None else {}
    if lanes is not None:   # 평가기 machine_info 가 config 를 meta 에 남긴다(차선·모델·어댑터·카드·표본 카드가 결과 폴더에 남음)
        # run-config 의 MODEL·PLAN 같은 이름과 겹치는 칸은 차선 값이 이긴다
        cfg = dict(cfg or {}, role=role, lanes=lanes.n, gpus=len(gpus) if gpus else None,
                   **{x: meta.get(x) for x in CONFIG_FIELDS + ("samples",)})
    lg = lane_log(state_dir, k)
    st = {"instance_id": os.environ.get("CONTAINER_ID") or None,   # stewart-cloud down 이 logout 전에 대조(v6)
          "stop_failed": dict(stop_failed) if stop_failed else None, "uptime_s": round(now - w.t_start), "idle_s": round(now - w.last_change), "seen_metrics": w.seen_metrics,
          "limits": {"idle_s": w.idle_s, "boot_grace_s": w.boot_grace_s, "max_s": w.max_s},
          "config": cfg,
          "gpu": (lane_gpus(gpus, meta.get("cards"), k)[0] if lanes is not None else (gpus[0] if gpus else None)), "gpu_fields": GPU_FIELDS,
          "gpu_ts": (datetime.fromtimestamp(gpu_ts, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if gpu_ts else None),
          "gpu_age_s": (round(time.time() - gpu_ts) if gpu_ts else None),
          "idle_left_s": (None if w.idle_left(now) is None else round(w.idle_left(now))),
          "model_fetch": model_fetch(state_dir),
          "pod_log": _tail(os.path.join(state_dir, "pod-log.tsv"), 20),
          "vllm_kv": _tail(lg, 15, KV_LINE),
          "verify": _tail(lg, 40, VERIFY_LINE),
          "quality": _quality(state_dir)}
    if lanes is not None:
        summ = lanes.summary(gpus)
        st["lane"] = summ[k]
        if k == 0:
            st["role"] = role
            st["lanes"] = summ
    return st


# entrypoint 가 돌리는 verify.sh --no-server 의 판정 줄
VERIFY_LINE = re.compile(r"^\s*(PASS|WARN|FAIL)\s|entrypoint: verify|^== ")


def _quality(state_dir: str) -> dict:
    """quality-<tag>.log(파드 안 bench/quality_battery.py)마다 PPL·GSM8K·exit 줄만."""
    out = {}
    try:
        names = sorted(n for n in os.listdir(state_dir) if n.startswith("quality-") and n.endswith(".log"))
    except OSError:
        return out
    for n in names[-8:]:
        out[n[8:-4]] = _tail(os.path.join(state_dir, n), 6, re.compile(r" PPL | GSM8K |^exit=|Error|error"))
    return out


class LogoutOnce:
    """POST /logout: 테일넷에서 바로 빠진다(stewart-cloud down 이 Destroy 전에 부른다). 인증 없음 — 할 수 있는 일이
    "테일넷에서 빠지기" 뿐이라 정보·권한이 늘지 않는다(결정). 한 번만 하고, 그 뒤는 already 를 돌려준다.
    응답을 먼저 보낸 뒤 조금 있다 logout 한다(logout 하면 이 연결이 끊기므로). 인증 키 파일이 남아 있으면 지워
    이 컨테이너가 다시 붙지 않게 한다."""

    def __init__(self, state_dir: str, delay_s: float = 1.0, logout=None):
        self.state_dir, self.delay_s = state_dir, delay_s
        self.logout = logout or (lambda: _logout(state_dir))
        self.done = False
        self.lock = threading.Lock()

    def __call__(self, client: str) -> dict:
        with self.lock:
            if self.done:
                log(self.state_dir, f"logout 요청(이미 함) from {client}")
                return {"result": "already_logged_out"}
            self.done = True
        log(self.state_dir, f"logout 요청 from {client}")
        for f in (TS_KEY_FILE,):
            try:
                os.unlink(f)
            except OSError:
                pass

        def later():
            time.sleep(self.delay_s)
            self.logout()

        threading.Thread(target=later, daemon=True).start()
        return {"result": "logging_out"}


TS_KEY_FILE = "/dev/shm/ts-authkey"


def serve_status(port: int, get, on_logout=None) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, obj) -> None:
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_POST(self):
            if self.path.rstrip("/") != "/logout" or on_logout is None:
                self._json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if 0 < n <= 65536:
                self.rfile.read(n)                 # 본문은 읽고 버린다
            self._json(200, on_logout(self.client_address[0]))

        def do_GET(self):
            if self.path.rstrip("/") != "/status":
                self.send_response(404)
                self.end_headers()
                return
            b = json.dumps(get(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = QuickHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="/workspace/stewart")
    ap.add_argument("--role", choices=("serve", "eval"), default="serve")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--idle-min", type=float, default=None, help="없으면 IDLE_MIN 환경 변수, 그것도 없으면 역할별(serve 90·eval 30)")
    ap.add_argument("--boot-grace-min", type=float, default=float(os.environ.get("BOOT_GRACE_MIN", 45)))
    ap.add_argument("--max-hours", type=float, default=float(os.environ.get("MAX_HOURS", 8)))
    ap.add_argument("--lane-boot-min", type=float, default=float(os.environ.get("LANE_BOOT_MIN") or LANE_BOOT_MIN),
                    help="차선 k≥1 기동 상한(분, T28). 차선 0 은 --boot-grace-min")
    ap.add_argument("--status-port", type=int, default=8001, help="0이면 상태 응답을 띄우지 않는다(차선 k 는 이 값 + 10k)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true", help="한 번만 보고 판정을 출력(시험용)")
    a = ap.parse_args()
    if not 1 <= a.lanes <= MAX_LANES or (a.role == "serve" and a.lanes != 1):
        ap.error("lanes 는 1~8, serve 는 1")
    if a.idle_min is None:
        a.idle_min = float(os.environ.get("IDLE_MIN") or IDLE_MIN_BY_ROLE[a.role])
    w = Watch(a.idle_min * 60, a.boot_grace_min * 60, a.max_hours * 3600, time.monotonic())
    lanes = Lanes(a.state_dir, a.lanes, a.role, lane_boot_s=a.lane_boot_min * 60, lane0_boot_s=a.boot_grace_min * 60)
    log(a.state_dir, f"watch start role={a.role} lanes={a.lanes} idle={a.idle_min}m boot_grace={a.boot_grace_min}m "
                     f"lane_boot={a.lane_boot_min}m max={a.max_hours}h")
    halt_at = None
    gpus: dict = {"v": None, "t": None}
    t_gpu, t_tsv, t_spec, spec_last = -1e9, -1e9, -1e9, {}
    stop_failed: dict = {}
    if a.status_port and not a.once:
        for k in range(a.lanes):
            serve_status(a.status_port + 10 * k,
                         (lambda k=k: status(a.state_dir, w, time.monotonic(), gpus["v"], stop_failed, lanes, k, a.role, gpus["t"])),
                         on_logout=LogoutOnce(a.state_dir) if k == 0 else None)
    while True:
        if time.monotonic() - t_gpu >= GPU_SAMPLE_S:
            g, t_gpu = gpu_samples(), time.monotonic()
            if g:
                gpus["v"], gpus["t"] = g, time.time()
            if g and time.monotonic() - t_tsv >= GPU_TSV_S:
                t_tsv = time.monotonic()
                ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                with open(os.path.join(a.state_dir, "gpu.tsv"), "a", encoding="utf-8") as f:
                    for n, line in enumerate(gpus["v"]):
                        f.write(f"{ts}\t{line}\n" if n == 0 else f"{ts}\tgpu{n}\t{line}\n")   # 첫 카드는 v6 꼴 그대로
        mf = model_fetch(a.state_dir)
        if mf and mf.get("state") in FETCH_HOLD:
            w.hold(time.monotonic())                        # home 받기 동안은 기동 유예를 세지 않는다(T24)
            lanes.hold()                                    # 차선 기동 상한도(T28)
        ms = [fetch(f"http://127.0.0.1:{lane_port(k)}/metrics") for k in range(a.lanes)]
        lanes.observe_metrics(ms)
        if time.monotonic() - t_spec >= SPEC_TSV_S:
            t_spec = time.monotonic()
            try:
                spec_tsv(a.state_dir, lanes.st, plan_rows(a.state_dir), spec_last,
                         datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            except Exception as e:      # 기록이 감시(자동 정지)를 멈추게 하지 않는다
                log(a.state_dir, f"spec-log 오류: {type(e).__name__}")
        try:
            lanes.check()
        except Exception as e:      # 차선 확인이 감시(자동 정지)를 멈추게 하지 않는다
            log(a.state_dir, f"lane check 오류: {type(e).__name__}")
        reason = w.observe(combine(ms), time.monotonic())
        if lanes.all_failed():                              # T28: 모든 차선 실패 → HALT(이유 한 줄 → 900초 → 정지)
            if halt_at is None:
                halt_at = time.monotonic() + HALT_WAIT_S
                whys = "; ".join("%s: %s" % (x["k"], (x["why"] or "-")[:120]) for x in lanes.summary(None))
                log(a.state_dir, f"HALT: 모든 차선 실패({whys}) — {HALT_WAIT_S}초 뒤 정지")
            if time.monotonic() >= halt_at:
                reason = reason or "all_lanes_failed"
        if a.once:
            print(json.dumps({"metrics": any(m is not None for m in ms), "reason": reason,
                              "lanes": [s["state"] for s in lanes.summary(None)]}))
            return 0
        if reason:
            if stop_pod(a.state_dir, reason, a.dry_run, stop_failed):
                return 0
            time.sleep(300)   # 멈추지 못했으면 테일넷에 남은 채 5분 뒤 다시(/status stop_failed 로 보인다)
            continue
        all_ready = all(s["ready"] or s["state"] == "failed" for s in lanes.summary(None))
        time.sleep(a.interval if all_ready else min(a.interval, 15))   # 차선이 다 뜨기 전에는 15초마다(ready 를 빨리)


if __name__ == "__main__":
    raise SystemExit(main())
