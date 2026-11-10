"""파드 묶음 시험: 쉬다 정지(idle_stop)·파드 설정·역할·HALT·차선·템플릿 환경."""

import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tests import FIX_ROOT, LOCAL
from unittest import mock


class TIdleStop(unittest.TestCase):
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("idle_stop", FIX_ROOT / "deploy/vast/idle_stop.py")
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_local_requests_skip_proxies(self):
        """127.0.0.1 요청(fetch·_http)은 환경 프록시·시스템 프록시가 있어도 직접 간다(macOS urllib 은 시스템 프록시를 따른다)."""
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"vllm:generation_tokens_total 7\n" if self.path == "/metrics" else b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        port = srv.server_address[1]
        dead = "http://127.0.0.1:9"                                              # 닫힌 포트: 프록시로 가면 실패한다
        env = {"HTTP_PROXY": dead, "http_proxy": dead, "HTTPS_PROXY": dead, "https_proxy": dead, "NO_PROXY": "", "no_proxy": ""}
        with mock.patch.dict(os.environ, env), \
                mock.patch("urllib.request.getproxies", lambda: {"http": dead, "https": dead}):   # 시스템 프록시 흉내
            self.assertEqual(self.m.fetch(f"http://127.0.0.1:{port}/metrics")["vllm:generation_tokens_total"], 7.0)
            self.assertEqual(self.m._http(f"http://127.0.0.1:{port}/health")[0], 200)
            with self.assertRaises(OSError):                                      # 같은 조건에서 기본 urlopen 은 프록시로 간다
                urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)

    def test_parse(self):
        t = '# HELP x\nvllm:num_requests_running{model_name="q"} 0.0\nvllm:generation_tokens_total{a="1"} 10\n' \
            'vllm:generation_tokens_total{a="2"} 5\n'
        self.assertEqual(self.m.parse_metrics(t)["vllm:generation_tokens_total"], 15.0)

    def test_watch(self):
        W = self.m.Watch
        idle = {"vllm:generation_tokens_total": 5.0, "vllm:num_requests_running": 0.0}
        w = W(1800, 2700, 8 * 3600, 0)
        self.assertIsNone(w.observe(None, 600))                 # 기동 중
        self.assertIsNone(w.observe(idle, 1000))                # 처음 봄
        self.assertIsNone(w.observe(idle, 2700))                # 29분 유휴
        self.assertEqual(w.observe(idle, 2801), "idle 30m")     # 30분 유휴
        w = W(1800, 2700, 8 * 3600, 0)
        w.observe(idle, 100)
        busy = dict(idle, **{"vllm:num_requests_running": 1.0})
        self.assertIsNone(w.observe(busy, 5000))                 # 긴 요청이 돌고 있으면 유휴 아님
        self.assertIsNone(w.observe(dict(idle, **{"vllm:generation_tokens_total": 9.0}), 6000))
        self.assertIsNone(w.observe(dict(idle, **{"vllm:generation_tokens_total": 9.0}), 7700))
        self.assertEqual(W(1800, 2700, 3600, 0).observe(None, 2800), "boot_timeout 45m")
        self.assertEqual(W(1800, 2700, 3600, 0).observe(idle, 3600), "max_uptime 1.0h")

    def test_status_endpoint(self):
        d = tempfile.mkdtemp()
        Path(d, "vllm.log").write_text("INFO x\nINFO GPU KV cache size: 123,456 tokens\nINFO y\n"
                                       "== model\n  PASS  lm_head int8\n  FAIL  draft head missing\n")
        Path(d, "quality-T1-fast1-fast.log").write_text("noise\nT1 PPL {'en': (10.7, 9000)} 80s\nT1 GSM8K n=200 acc=0.960 mean_tokens=250 90s\nexit=0\n")
        w = self.m.Watch(1800, 2700, 3600, 0)
        srv = self.m.serve_status(0, lambda: self.m.status(d, w, 100, "20000, 24564, 0, 30"))
        try:
            import urllib.request
            with LOCAL.open(f"http://127.0.0.1:{srv.server_address[1]}/status", timeout=5) as r:
                st = json.loads(r.read())
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(st["vllm_kv"], ["INFO GPU KV cache size: 123,456 tokens"])
        self.assertEqual(st["uptime_s"], 100)
        self.assertIsNone(st["config"])
        self.assertEqual(st["gpu"], "20000, 24564, 0, 30")
        self.assertIn("power.limit", st["gpu_fields"])
        self.assertEqual(st["verify"], ["== model", "  PASS  lm_head int8", "  FAIL  draft head missing"])
        self.assertEqual(st["quality"]["T1-fast1-fast"][-1], "exit=0")
        self.assertEqual(len(st["quality"]["T1-fast1-fast"]), 3)

    def test_stop_order(self):
        d = tempfile.mkdtemp()
        failed = {}
        with mock.patch("sys.stdout", io.StringIO()), mock.patch.object(self.m, "_logout") as lo, \
                mock.patch.object(self.m, "request_stop", return_value=(False, "api HTTPError 401")):
            self.assertFalse(self.m.stop_pod(d, "idle 30m", False, failed))
            self.assertFalse(self.m.stop_pod(d, "idle 30m", False, failed))
            lo.assert_not_called()                                   # 실패하면 logout 하지 않는다
        self.assertEqual((failed["count"], failed["reason"]), (2, "idle 30m"))
        w = self.m.Watch(1, 1, 1, 0)
        self.assertEqual(self.m.status(d, w, 5, None, failed)["stop_failed"]["count"], 2)
        with mock.patch("sys.stdout", io.StringIO()), mock.patch.object(self.m, "_logout") as lo, \
                mock.patch.object(self.m, "request_stop", return_value=(True, "api HTTP 200")):
            self.assertTrue(self.m.stop_pod(d, "idle 30m", False, failed))
            lo.assert_called_once()                                  # 성공한 뒤에만 logout
        log = (Path(d) / "pod-log.tsv").read_text()
        self.assertLess(log.index("stop ok"), len(log))
        self.assertNotIn("Bearer", log)

    def test_vast_stop_request(self):
        seen = {}

        class R:
            status = 200
            def read(self, n=-1): return b'{"success": true}'
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake(req, timeout=0):
            seen.update(url=req.full_url, method=req.get_method(), body=req.data, auth=req.get_header("Authorization"))
            return R()
        kf = Path(tempfile.mkdtemp(), "k")
        kf.write_text("KEY123\n")
        with mock.patch.dict(os.environ, {"CONTAINER_ID": "38250"}, clear=False), \
                mock.patch.object(self.m, "KEY_FILE", str(kf)), mock.patch.object(self.m.urllib.request, "urlopen", fake):
            ok, detail = self.m.request_stop("/nonexistent")
        self.assertTrue(ok)
        self.assertEqual((seen["url"], seen["method"]), ("https://console.vast.ai/api/v0/instances/38250/", "PUT"))
        self.assertEqual(json.loads(seen["body"]), {"state": "stopped"})
        self.assertEqual(seen["auth"], "Bearer KEY123")
        self.assertNotIn("KEY123", detail)                            # 키 값은 설명에 남기지 않는다
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(self.m, "KEY_FILE", "/nonexistent"):
            ok, detail = self.m.request_stop("/nonexistent")
        self.assertFalse(ok)
        self.assertIn("CONTAINER_ID 없음", detail)

    def test_stop_dry_run_logs(self):
        d = tempfile.mkdtemp()
        with mock.patch("sys.stdout", io.StringIO()):
            self.assertTrue(self.m.stop_pod(d, "idle 30m", dry_run=True))
        self.assertIn("stop reason=idle 30m", (Path(d) / "pod-log.tsv").read_text())


class TPodConfig(unittest.TestCase):
    """pod-start.sh 의 설정 파일(CF) 구간만 떼어 bash 로 돌린다. curl 은 가짜(파일을 내보낸다)."""
    SH = FIX_ROOT / "deploy/vast/pod-start.sh"

    def run_cfg(self, content: str | None, env: dict | None = None) -> tuple[dict, str, Path]:
        import subprocess
        src = self.SH.read_text(encoding="utf-8")
        part = src[src.index("# ---- 1b."):src.index("# ---- 1c.")]
        d = Path(tempfile.mkdtemp())
        (d / "bin").mkdir()
        fake = d / "bin" / "curl"
        if content is None:
            fake.write_text("#!/bin/sh\nexit 22\n")
        else:
            (d / "cf.txt").write_text(content)
            fake.write_text(f"#!/bin/sh\ncat '{d}/cf.txt'\n")
        fake.chmod(0o755)
        script = "set -euo pipefail\nS=%s\nlog() { printf '%%s\\n' \"$*\" >> \"$S/log\"; }\n%s\n" % (d, part)
        e = {"PATH": f"{d}/bin:/usr/bin:/bin", "CF": "https://example.invalid/cf"}
        e.update(env or {})
        r = subprocess.run(["bash", "-c", script], env=e, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads((d / "run-config.json").read_text()), (d / "log").read_text(), d

    def test_allowlist_and_ignore(self):
        cfg, log, d = self.run_cfg("CTX=long\r\n SPEC = mtp\nPREFIX_CACHE=1\nFAST_VARIANT=1\n# 주석\n"
                                   "CTX=huge\nPU=https://evil/x\nPS=0000\nHF_REPO=evil/model\n"
                                   "QUALITY=$(touch pwned)\nCTX=long;touch pwned2\nfoo\n")
        self.assertEqual((cfg["src"], cfg["CTX"], cfg["SPEC"], cfg["PREFIX_CACHE"], cfg["FAST_VARIANT"], cfg["QUALITY"]),
                         ("cf", "long", "mtp", "1", "1", "auto"))
        for k in ("CTX", "PU", "PS", "HF_REPO", "QUALITY", "foo"):
            self.assertIn(k, cfg["ignored"])
        self.assertNotIn("evil", log + json.dumps(cfg))           # 무시한 값은 남기지 않는다
        self.assertFalse(any(d.glob("pwned*")))                    # 실행되지 않는다
        self.assertIn("config src=cf CTX=long SPEC=mtp PREFIX_CACHE=1 FAST_VARIANT=1", log)

    def test_fetch_failure_uses_docker_defaults(self):
        cfg, log, _ = self.run_cfg(None, {"CTX": "fast", "PREFIX_CACHE": "1"})
        self.assertEqual((cfg["src"], cfg["CTX"], cfg["PREFIX_CACHE"], cfg["FAST_VARIANT"]), ("default-cf-failed", "fast", "1", "1"))

    def test_model_adapter_allowlist(self):
        cfg, log, d = self.run_cfg("MODEL=qwen36-35b-a3b-demo-w4a16\nADAPTER=vllm-patched\n")
        self.assertEqual((cfg["MODEL"], cfg["ADAPTER"]), ("qwen36-35b-a3b-demo-w4a16", "vllm-patched"))
        self.assertIn("MODEL=qwen36-35b-a3b-demo-w4a16 ADAPTER=vllm-patched", log)   # SIG 에 든다(품질 점검 단위)
        cfg, log, d = self.run_cfg("MODEL=../evil\nMODEL=$(touch pwned)\nADAPTER=evil\nMODEL=Big/Repo\n")
        self.assertEqual((cfg["MODEL"], cfg["ADAPTER"]), ("default", "auto"))
        self.assertIn("MODEL", cfg["ignored"])
        self.assertIn("ADAPTER", cfg["ignored"])
        self.assertNotIn("evil", log + json.dumps(cfg))
        self.assertFalse(any(d.glob("pwned*")))
        cfg, _, _ = self.run_cfg("", {"MODEL": "x;rm", "ADAPTER": "bad", "PLAN": "../p"})   # Docker Options 값도 같은 검사
        self.assertEqual((cfg["MODEL"], cfg["ADAPTER"], cfg["PLAN"]), ("default", "auto", ""))
        cfg, _, _ = self.run_cfg("PLAN=cmp-a-b1\n", {"ROLE": "eval"})
        self.assertEqual(cfg["PLAN"], "cmp-a-b1")

    def test_up_plan_beats_cf(self):
        """T23: 만들 때 넘긴 STEWART_PLAN 이 CF 의 PLAN 보다 먼저(eval). serve 는 무시."""
        cfg, log, _ = self.run_cfg("PLAN=cmp-a-b1\n", {"ROLE": "eval", "REQ_PLAN": "a-x4"})
        self.assertEqual((cfg["PLAN"], cfg["plan_src"]), ("a-x4", "up"))
        cfg, log, _ = self.run_cfg("", {"ROLE": "eval", "REQ_PLAN": "a-x4"})     # CF 에 PLAN 이 없어도
        self.assertEqual((cfg["PLAN"], cfg["plan_src"]), ("a-x4", "up"))
        cfg, log, _ = self.run_cfg("PLAN=cmp-a-b1\n", {"ROLE": "eval"})
        self.assertEqual((cfg["PLAN"], cfg["plan_src"]), ("cmp-a-b1", "cf"))
        base, _, _ = self.run_cfg("", {"ROLE": "serve"})
        cfg, log, _ = self.run_cfg("PLAN=cmp-a-b1\n", {"ROLE": "serve", "REQ_PLAN": "a-x4"})
        self.assertEqual(cfg["PLAN"], "")
        self.assertIn("STEWART_PLAN", cfg["ignored"])
        for k in ("CTX", "SPEC", "PREFIX_CACHE", "FAST_VARIANT", "MODEL", "ADAPTER", "cond"):
            self.assertEqual(cfg[k], base[k], k)

    def test_serve_ignores_plan(self):
        """T21b: CF 를 serve 와 eval 이 같이 읽는다. serve 는 PLAN 을 버리므로 PLAN 만 더한 CF 로도 v6 와 같은 값."""
        base, blog, _ = self.run_cfg("", {"ROLE": "serve"})
        got, glog, _ = self.run_cfg("PLAN=cmp-a-b1\n", {"ROLE": "serve"})
        self.assertEqual(got["PLAN"], "")
        self.assertIn("PLAN", got["ignored"])
        for k in ("CTX", "SPEC", "PREFIX_CACHE", "FAST_VARIANT", "QUALITY", "MODEL", "ADAPTER", "cond"):
            self.assertEqual(got[k], base[k], k)
        sig = lambda log: log.split("config src=")[1].split(" QUALITY=")[0].split(" ", 1)[1]
        self.assertEqual(sig(glog), sig(blog))                                # 품질 점검 단위(SIG)도 같다
        self.assertIn("PLAN=-", glog)
        self.assertEqual(got["cond"], "cf")

    def test_docker_values_also_allowlisted(self):
        cfg, _, _ = self.run_cfg("", {"CTX": "huge", "SPEC": "dflash2", "FAST_VARIANT": "x"})
        self.assertEqual((cfg["CTX"], cfg["SPEC"], cfg["FAST_VARIANT"]), ("fast", "mtp", "1"))


class TPodRole(unittest.TestCase):
    """pod-start.sh 의 역할 구간(1a)만 떼어 bash 로 돌린다. nvidia-smi 는 가짜(GPU 줄 N개), halt 는 기록 후 끝."""
    SH = FIX_ROOT / "deploy/vast/pod-start.sh"

    def run_role(self, env: dict, ngpu: int) -> tuple[int, str]:
        import subprocess
        src = self.SH.read_text(encoding="utf-8")
        part = src[src.index("# ---- 1a."):src.index("# ---- 1b.")]
        d = Path(tempfile.mkdtemp())
        (d / "bin").mkdir()
        fake = d / "bin" / "nvidia-smi"
        fake.write_text("#!/bin/sh\n" + "".join(f"echo 'GPU {i}: NVIDIA GeForce RTX 3090 (UUID: GPU-{i})'\n" for i in range(ngpu)))
        fake.chmod(0o755)
        script = ("set -euo pipefail\nS=%s\nlog() { printf '%%s\\n' \"$*\" >> \"$S/log\"; }\n"
                  "halt() { printf 'HALT %%s\\n' \"$*\" >> \"$S/log\"; exit 3; }\n%s\n"
                  "printf '%%s %%s %%s %%s %%s\\n' \"$TS_HOST\" \"$NGPU\" \"${STEWART_ROLE:-unset}\" \"${REQ_PLAN:-none}\" \"${STEWART_PLAN:-unset}\" >> \"$S/log\"\n") % (d, part)
        e = {"PATH": f"{d}/bin:/usr/bin:/bin"}
        e.update(env)
        r = subprocess.run(["bash", "-c", script], env=e, capture_output=True, text=True, timeout=30)
        return r.returncode, (d / "log").read_text() if (d / "log").exists() else r.stderr

    def test_roles(self):
        rc, log = self.run_role({"STEWART_ROLE": "serve", "STEWART_GPUS": "1"}, 1)
        self.assertEqual(rc, 0, log)
        self.assertIn("stewart-gpu 1 unset none unset", log)                   # 환경에서 지운다
        rc, log = self.run_role({"STEWART_ROLE": "eval", "STEWART_GPUS": "4"}, 4)
        self.assertEqual(rc, 0, log)
        self.assertIn("role=eval ts=stewart-gpu-eval gpus=4", log)
        self.assertIn("stewart-gpu-eval 4 unset none unset", log)
        rc, log = self.run_role({"STEWART_ROLE": "eval", "STEWART_GPUS": "4", "STEWART_PLAN": "a-x4"}, 4)
        self.assertEqual(rc, 0, log)
        self.assertIn("stewart-gpu-eval 4 unset a-x4 unset", log)               # 계획은 REQ_PLAN 으로 옮기고 env 에서 지운다
        rc, log = self.run_role({"STEWART_ROLE": "eval2", "STEWART_GPUS": "4", "STEWART_PLAN": "a-all"}, 4)   # T26 둘째 eval
        self.assertEqual(rc, 0, log)
        self.assertIn("role=eval ts=stewart-gpu-eval2 gpus=4 as=eval2", log)   # 이름만 다르고 ROLE=eval(같은 차선·계획 길)
        self.assertIn("stewart-gpu-eval2 4 unset a-all unset", log)
        for bad in ("A-X4", "a;rm", "../x", "x" * 49):
            rc, log = self.run_role({"STEWART_ROLE": "eval", "STEWART_GPUS": "4", "STEWART_PLAN": bad}, 4)
            self.assertEqual(rc, 3, bad)
            self.assertIn("HALT STEWART_PLAN", log)

    def test_halts(self):
        for env, n in (({}, 1), ({"STEWART_GPUS": "1"}, 1), ({"STEWART_ROLE": "Eval", "STEWART_GPUS": "4"}, 4),
                       ({"STEWART_ROLE": "eval;touch x", "STEWART_GPUS": "4"}, 4), ({"STEWART_ROLE": "box", "STEWART_GPUS": "1"}, 1),
                       ({"STEWART_ROLE": "eval", "STEWART_GPUS": "8"}, 4), ({"STEWART_ROLE": "eval", "STEWART_GPUS": "9"}, 9),
                       ({"STEWART_ROLE": "eval", "STEWART_GPUS": "4x"}, 4), ({"STEWART_ROLE": "serve", "STEWART_GPUS": "2"}, 2),
                       ({"STEWART_ROLE": "eval"}, 4), ({"STEWART_ROLE": "eval3", "STEWART_GPUS": "4"}, 4),
                       ({"STEWART_ROLE": "Eval2", "STEWART_GPUS": "4"}, 4), ({"STEWART_ROLE": "eval2", "STEWART_GPUS": "8"}, 4)):
            rc, log = self.run_role(env, n)
            self.assertEqual(rc, 3, (env, n, log))
            self.assertIn("HALT", log)
            self.assertNotIn("role=", log)

    def test_cf_cannot_set_role(self):
        cfg, log, _ = TPodConfig.run_cfg(TPodConfig(), "STEWART_ROLE=eval\nSTEWART_GPUS=8\nTS_HOST=evil\nCTX=long\n")
        for k in ("STEWART_ROLE", "STEWART_GPUS", "TS_HOST"):
            self.assertIn(k, cfg["ignored"])
        self.assertNotIn("evil", log + json.dumps(cfg))


class TPodHalt(unittest.TestCase):
    """T23: HALT 는 이유 한 줄 → 900초 뒤 스스로 정지(인스턴스 범위 키). pod-start 의 halt, boot.sh(Args 가 SB 로 돌림)."""
    ROOT = FIX_ROOT / "deploy/vast"

    def fakes(self, d: Path, curl_body: str = "exit 22"):
        (d / "bin").mkdir(exist_ok=True)
        (d / "bin" / "sleep").write_text(f"#!/bin/sh\necho \"sleep $*\" >> {d}/calls\n")
        (d / "bin" / "curl").write_text(
            "#!/bin/sh\n"
            f"echo \"curl $*\" >> {d}/calls\n"
            "for a in \"$@\"; do case \"$a\" in @*) cat \"${a#@}\" >> " + f"{d}/hdr" + ";; esac; done\n"
            "case \"$*\" in *PUT*) printf 200; exit 0;; esac\n" + curl_body + "\n")
        # 맥에는 sha256sum 이 없다(pod 은 리눅스) — shasum 으로 대신(-c·--quiet·--strict 만)
        (d / "bin" / "sha256sum").write_text("#!/bin/sh\nargs=\"\"\nfor a in \"$@\"; do case \"$a\" in --quiet|--strict) ;; "
                                             "*) args=\"$args $a\";; esac; done\nexec /usr/bin/shasum -a 256 $args\n")
        # 맥 base64 는 "-d 파일" 꼴을 받지 않는다(GNU 는 받음) — 그 꼴만 -i 로 바꿔 넘긴다
        (d / "bin" / "base64").write_text("#!/bin/sh\nif [ \"$1\" = -d ] && [ $# -eq 2 ]; then exec /usr/bin/base64 -d -i \"$2\"; fi\n"
                                          "exec /usr/bin/base64 \"$@\"\n")
        for f in ("sleep", "curl", "sha256sum", "base64"):
            (d / "bin" / f).chmod(0o755)

    def test_pod_start_halt_waits_then_stops(self):
        import subprocess
        src = (self.ROOT / "pod-start.sh").read_text(encoding="utf-8")
        part = src[src.index("HALT_WAIT_S=900"):src.index("trap 'halt")]
        d = Path(tempfile.mkdtemp())
        self.fakes(d)
        (d / "vkey").write_text("INSTKEY123")
        (d / "tskey").write_text("tskey-x")
        part = part.replace("/dev/shm/vast-hdr", f"{d}/vast-hdr")
        script = (f"set -euo pipefail\nS={d}\nKEYF={d}/tskey\nVKEY={d}/vkey\n"
                  "log() { printf '%s\\n' \"$*\" | tee -a \"$S/pod-log.tsv\"; }\n" + part + "\nhalt \"가짜 이유\"\n")
        r = subprocess.run(["bash", "-c", script], env={"PATH": f"{d}/bin:/usr/bin:/bin", "CONTAINER_ID": "12345678"},
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("HALT: 가짜 이유", r.stdout)                           # 이유 한 줄이 stdout 에(래퍼 logs 로 보인다)
        calls = (d / "calls").read_text().splitlines()
        self.assertEqual(calls[0], "sleep 900")                              # 먼저 900초
        self.assertIn("-X PUT", calls[1])
        self.assertIn("https://console.vast.ai/api/v0/instances/12345678/", calls[1])
        self.assertNotIn("INSTKEY123", calls[1])                             # 키는 명령 줄이 아니라 헤더 파일로
        self.assertEqual((d / "hdr").read_text(), "Authorization: Bearer INSTKEY123\n")
        self.assertEqual(calls[2], "sleep infinity")
        self.assertFalse((d / "tskey").exists())
        self.assertIn("HALT stop api HTTP 200", (d / "pod-log.tsv").read_text())

    def test_pod_start_halt_holds_after_tailnet(self):
        """v14: 테일넷이 붙은 뒤의 HALT 는 셸 비상구로 HALT_HOLD_MIN 분 기다린 뒤 정지(hold 면 더, stop-now 면 바로)."""
        import subprocess
        src = (self.ROOT / "pod-start.sh").read_text(encoding="utf-8")
        part = src[src.index("HALT_WAIT_S=900"):src.index("trap 'halt")]
        d = Path(tempfile.mkdtemp())
        self.fakes(d)
        (d / "vkey").write_text("INSTKEY123")
        part = part.replace("/dev/shm/vast-hdr", f"{d}/vast-hdr")
        script = (f"set -euo pipefail\nS={d}\nKEYF={d}/tskey\nVKEY={d}/vkey\n"
                  "log() { printf '%s\\n' \"$*\" | tee -a \"$S/pod-log.tsv\"; }\n" + part
                  + "\nTS_UP=1 TS_NAME=stewart-gpu-eval.x\nhalt \"가짜 이유\"\n")
        r = subprocess.run(["bash", "-c", script], env={"PATH": f"{d}/bin:/usr/bin:/bin", "CONTAINER_ID": "1", "HALT_HOLD_MIN": "3"},
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("셸 비상구: 집에서 tailscale ssh root@stewart-gpu-eval.x 로 들어온다", r.stdout)
        self.assertIn("셸 비상구: 3분 기다림 끝 — 정지", r.stdout)
        calls = (d / "calls").read_text().splitlines()
        self.assertEqual(calls[:3], ["sleep 60"] * 3)                           # 900초가 아니라 1분씩 세 번
        self.assertIn("-X PUT", calls[3])

    def run_boot(self, env: dict, curl_body: str = "exit 22"):
        import subprocess
        d = Path(tempfile.mkdtemp())
        self.fakes(d, curl_body)
        boot = (self.ROOT / "boot.sh").read_text(encoding="utf-8").replace("/workspace/stewart", f"{d}/ws").replace("/dev/shm", str(d))
        (d / "boot.sh").write_text(boot)
        e = {"PATH": f"{d}/bin:/usr/bin:/bin", "CONTAINER_ID": "777", "CONTAINER_API_KEY": "INSTKEY9"}
        e.update(env)
        r = subprocess.run(["bash", str(d / "boot.sh")], env=e, capture_output=True, text=True, timeout=30)
        return r, d, ((d / "calls").read_text().splitlines() if (d / "calls").exists() else [])

    def test_boot_failures_stop_after_900(self):
        good_ps = "0" * 64
        cases = [({}, "PU 없음"), ({"PU": "https://gist.example/x"}, "PS 없음"),
                 ({"PU": "https://gist.example/x", "PS": "xyz"}, "PS 없음"),
                 ({"PU": "https://gist.example/x", "PS": good_ps}, "pod files")]
        for env, why in cases:
            r, d, calls = self.run_boot(env)
            self.assertIn("HALT: " + why, r.stdout, (env, r.stdout, r.stderr))
            self.assertEqual(calls[-3], "sleep 900", calls)
            self.assertIn("-X PUT", calls[-2])
            self.assertIn("console.vast.ai/api/v0/instances/777/", calls[-2])
            self.assertNotIn("INSTKEY9", " ".join(calls))
            self.assertEqual((d / "hdr").read_text(), "Authorization: Bearer INSTKEY9\n")
            self.assertEqual(calls[-1], "sleep infinity")
            self.assertIn("HALT stop api HTTP 200", r.stdout)
        r, d, calls = self.run_boot({"PU": "https://gist.example/x", "PS": good_ps}, "printf 'QUJD'")   # 받았지만 sha 다름
        self.assertIn("HALT: pod files sha256 불일치", r.stdout)
        self.assertNotIn("pod-start", " ".join(calls))

    def test_boot_success_runs_pod_start(self):
        import io as _io, tarfile as _tf
        raw = _io.BytesIO()
        with _tf.open(fileobj=raw, mode="w:gz") as t:
            data = b"echo POD-START-RAN $PWD\n"
            ti = _tf.TarInfo("pod-start.sh"); ti.size = len(data)
            t.addfile(ti, _io.BytesIO(data))
        b64 = base64.b64encode(raw.getvalue()).decode()
        sha = hashlib.sha256(b64.encode()).hexdigest()
        r, d, calls = self.run_boot({"PU": "https://gist.example/x", "PS": sha}, f"printf '%s\\n' '{b64[:40]}'; printf '%s' '{b64[40:]}'")
        self.assertIn("POD-START-RAN", r.stdout, r.stdout + r.stderr)        # 줄바꿈을 지운 뒤 대조 → 풀고 pod-start
        self.assertFalse([c for c in calls if "PUT" in c])

    def test_args_checks_sb(self):
        import importlib.util, subprocess
        spec = importlib.util.spec_from_file_location("mte", self.ROOT / "make_template_env.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        d = Path(tempfile.mkdtemp())
        self.fakes(d)
        inner = m.ARGS[len("bash -c '"):-1].replace("/dev/shm", str(d))
        e = {"PATH": f"{d}/bin:/usr/bin:/bin", "CONTAINER_ID": "5"}
        r = subprocess.run(["bash", "-c", inner], env=dict(e, SB=base64.b64encode(b"echo not-our-boot").decode()),
                           capture_output=True, text=True, timeout=30)
        self.assertIn("HALT: boot SB", r.stdout)                             # sha 가 다른 SB 는 돌리지 않는다
        self.assertNotIn("not-our-boot", r.stdout)
        r = subprocess.run(["bash", "-c", inner], env=dict(e, SB=m.boot_b64()), capture_output=True, text=True, timeout=30)
        self.assertIn(f"{d}/b: OK", r.stdout)                                # 우리 boot.sh 면 대조 통과 → 실행(작업 폴더에서 HALT)
        self.assertIn("HALT: ", r.stdout)
        self.assertNotIn("HALT: boot SB", r.stdout)


class TPodLanes(unittest.TestCase):
    """pod-start.sh 의 1c(차선 계획)와 5e(eval 차선 띄우기)를 떼어 bash 로 돌린다. 계획은 진짜 serving.py(표 deploy/serving),
    serve_lane 은 가짜: serving.py 의 serve_cmd 로 부를 명령만 기록하고(실행하지 않음) PORT 에 /health 서버를 띄운다."""
    SH = FIX_ROOT / "deploy/vast/pod-start.sh"
    SV = FIX_ROOT / "deploy/serving"
    SERVER = (
        "import http.server, json, os, sys, threading, time\n"
        "sys.path.insert(0, os.environ['SV'])\n"
        "import serving as SVM\n"
        "envf, lane, port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])\n"
        "args, env = SVM.serve_cmd(SVM.load(), SVM.Path(envf), lane, port, dict(os.environ))\n"
        "rec = {'t': time.time(), 'lane': lane, 'port': port, 'cards': os.environ.get('CUDA_VISIBLE_DEVICES'),"
        " 'order': os.environ.get('CUDA_DEVICE_ORDER'), 'argv0': args[:2], 'tp': ('--tensor-parallel-size' in args),"
        " 'prepare': env.get('PREPARE'), 'verify': env.get('VERIFY'), 'model': env.get('MODEL'),"
        " 'pythonpath': env.get('PYTHONPATH', '')}\n"
        "open(os.environ['REC'], 'a').write(json.dumps(rec) + '\\n')\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def log_message(self, *a): pass\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
        "srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), H)\n"
        "threading.Thread(target=srv.serve_forever, daemon=True).start()\n"
        "time.sleep(0.3); print('INFO Model loading took 15.2 GiB memory', flush=True)\n"
        "time.sleep(float(os.environ.get('LIFE', '5')))\n")
    QB = ("import os, sys\nopen(os.environ['QREC'], 'a').write('%s %s %s%s\\n' % (sys.argv[1].split('-')[0], os.environ['VLLM_API'],"
          " os.environ['QUALITY_MODEL'], ' gsm-only' if '--gsm-only' in sys.argv else ''))\nprint('T PPL x')\n")

    def run_plan(self, plan: str, ngpu: int, quality: str = "1", lane0_dies: bool = False, expect_rc: int = 0,
                 die_lane: int | None = None):
        import random, subprocess, sys as _sys
        src = self.SH.read_text(encoding="utf-8")
        p1c = src[src.index("# ---- 1c."):src.index("# ---- 2.")]
        p5e = src[src.index('if [ "$ROLE" = "eval" ]; then\n  # ---- 5e.'):src.index("# ---- 5s.")]
        p5e = p5e.replace("exec sleep infinity", "exit 0")
        p5e = src[src.index("quality_flags() {"):src.index("# ---- 5. vLLM")] + p5e     # T28: 품질 점검 PPL 끄기 함수
        d = Path(tempfile.mkdtemp())
        (d / "serving" / "bench").mkdir(parents=True)
        (d / "serving" / "bench" / "quality_battery.py").write_text(self.QB)
        ov = d / "overlay" / "vllm-0.30.0"
        (ov / "vllm").mkdir(parents=True)
        (ov / "stewart-overlay.json").write_text("{}")
        (d / "srv.py").write_text(self.SERVER)
        bp = random.randrange(20000, 60000, 100)
        fake = (f'serve_lane() {{ {_sys.executable} {d}/srv.py "$3" "$1" "$2"; }}\n' if not lane0_dies else
                f'serve_lane() {{ if [ "$1" = 0 ]; then echo "entrypoint: verify.sh FAILED"; return 1; fi; {_sys.executable} {d}/srv.py "$3" "$1" "$2"; }}\n')
        if die_lane is not None:                                   # T28: 그 차선 서버가 바로 끝난다(E14 꼴)
            fake = (f'serve_lane() {{ if [ "$1" = {die_lane} ]; then echo "ImportError: boom at engine core"; return 7; fi; '
                    f'{_sys.executable} {d}/srv.py "$3" "$1" "$2"; }}\n')
        script = ("set -euo pipefail\nS=%s\nSV=%s\nPY=%s\nIMAGE_KIND=hyperqwen ROLE=eval NGPU=%d PLAN=%s MODEL_ALIAS=default ADAPTER=auto\n"
                  "QUALITY=%s SIG=sig FAST_VARIANT=1 CTX=fast PREFIX_CACHE=0 SPEC=mtp Q=$S/q COND_SRC=cf\nexport CTX SPEC PREFIX_CACHE FAST_VARIANT\n"
                  "write_run_config() { export CTX SPEC PREFIX_CACHE FAST_VARIANT; printf '%%s\\n' \"CTX=$CTX SPEC=$SPEC PREFIX_CACHE=$PREFIX_CACHE FAST_VARIANT=$FAST_VARIANT cond=$COND_SRC\" >> \"$S/cond.txt\"; }\n"
                  "log() { printf '%%s\\n' \"$*\" >> \"$S/pod-log.tsv\"; }\nhalt() { printf 'HALT %%s\\n' \"$*\" >> \"$S/pod-log.tsv\"; exit 3; }\n"
                  "seq() { if [ \"$1\" = 540 ] || [ \"$1\" = 120 ]; then /usr/bin/seq 40; else /usr/bin/seq \"$@\"; fi; }\n"
                  "sleep() { /bin/sleep 0.2; }\n%s\n"
                  "%s\n"
                  "awk -v bp=%d '{ $2 = $2 - 8000 + bp; print }' \"$S/lanes/plan.txt\" > \"$S/lanes/p2\" && mv \"$S/lanes/p2\" \"$S/lanes/plan.txt\"\n"
                  "export STEWART_OVERLAY=%s STEWART_MODELS=$S/models\n%s\n") % (d, self.SV, _sys.executable, ngpu, plan, quality, fake, p1c, bp, ov, p5e)
        e = {"PATH": "/usr/bin:/bin", "REC": str(d / "rec.jsonl"), "QREC": str(d / "q.txt"), "SV": str(self.SV), "LIFE": "4"}
        r = subprocess.run(["bash", "-c", script], env=e, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, expect_rc, r.stderr[-3000:] + (d / "pod-log.tsv").read_text() if (d / "pod-log.tsv").exists() else r.stderr)
        rec = [json.loads(l) for l in (d / "rec.jsonl").read_text().splitlines()] if (d / "rec.jsonl").exists() else []
        return d, bp, (d / "pod-log.tsv").read_text(), rec

    def test_cmp_a_b1_three_lanes(self):
        d, bp, log, rec = self.run_plan("cmp-a-b1", 4)
        plan = (d / "lanes" / "plan.txt").read_text().split("\n")
        self.assertIn(f"0 {bp} 0 qwen38-27b-demo-w4a16 hyperqwen-8cf642e 1", plan)
        self.assertIn(f"1 {bp + 10} 1 qwen38-27b-demo-w4a16 vllm-patched 1", plan)
        self.assertIn(f"2 {bp + 20} 2,3 qwen36-35b-a3b-demo-w4a16 vllm-patched 1", plan)
        self.assertEqual([(r["lane"], r["port"], r["cards"]) for r in rec], [(0, bp, "0"), (1, bp + 10, "1"), (2, bp + 20, "2,3")])
        self.assertTrue(all(r["order"] == "PCI_BUS_ID" for r in rec))
        self.assertEqual(rec[0]["argv0"], ["bash", "/app/docker/entrypoint.sh"])   # 차선 0 만 hyperqwen entrypoint(prepare·verify)
        self.assertTrue(rec[0]["model"].endswith("/Demo-Qwen3.8-27B-W4A16-AutoRound"))   # 계획의 FAST_VARIANT=0(CF 는 1)
        self.assertEqual((d / "cond.txt").read_text(), "CTX=long SPEC=mtp PREFIX_CACHE=1 FAST_VARIANT=0 cond=plan\n")   # 계획 조건이 CF 를 덮음
        self.assertEqual((d / "lanes" / "plan.env").read_text(), "CTX='long'\nFAST_VARIANT='0'\nPREFIX_CACHE='1'\nSPEC='mtp'\n")
        for r in rec[1:]:
            self.assertEqual(r["argv0"][1], "-c")                                      # vllm serve(파이썬)
            self.assertIn("/overlay/vllm-0.30.0", r["pythonpath"])                    # vllm-patched overlay
        self.assertEqual([r["tp"] for r in rec], [False, False, True])                # B1 은 tp 2
        t = [r["t"] for r in rec]
        self.assertEqual(t, sorted(t))
        self.assertIn("lane 0 health ok", log)
        self.assertLess(log.index("lane 1 start"), log.index("lane 2 start"))
        self.assertIn("lanes plan=cmp-a-b1 n=3", log)
        q = sorted((d / "q.txt").read_text().split("\n")[:-1])                       # 품질 점검: 차선마다 그 포트·그 서빙 이름
        # T28: SPEC=mtp(hyperqwen 차선은 CF SPEC, vllm-patched 차선은 표 serve.spec_method)면 PPL 을 빼고 GSM8K 만
        self.assertEqual(q, [f"L0 http://127.0.0.1:{bp}/v1 qwen3.8-27b gsm-only", f"L1 http://127.0.0.1:{bp + 10}/v1 qwen3.8-27b gsm-only",
                             f"L2 http://127.0.0.1:{bp + 20}/v1 qwen3.6-35b-a3b gsm-only"])
        self.assertIn("ppl=off(SPEC=mtp)", log)
        for k in range(3):
            self.assertTrue((d / "lanes" / f"quality-ok-{k}").exists(), k)
            self.assertTrue((d / "lanes" / f"lane-{k}.env").exists())
        self.assertTrue((d / "vllm.log").exists() and (d / "vllm-1.log").exists() and (d / "vllm-2.log").exists())

    def test_cards_must_fit(self):
        d, bp, log, rec = self.run_plan("cmp-a-b1", 3, expect_rc=3)                    # 카드 4장 계획에 3장 → HALT(띄우지 않음)
        self.assertIn("HALT 차선 계획 실패", log)
        self.assertEqual(rec, [])
        d, bp, log, rec = self.run_plan("cmp-a-b1", 5, expect_rc=3)                    # T25: 남는 카드도 HALT(카드 합 = 장 수)
        self.assertIn("4 장용이다(지금 5)", log)
        self.assertEqual(rec, [])
        d, bp, log, rec = self.run_plan("a-x4", 8, expect_rc=3)                        # 8장에 4장 계획 → HALT(a-all·cmp-a-b1-x2 를 고른다)
        self.assertIn("a-all=1+", log)
        self.assertEqual(rec, [])
        d, bp, log, rec = self.run_plan("off-a-b1", 4, expect_rc=3)                    # 공식 이미지 계획을 HyperQwen 이미지에서 → HALT
        self.assertIn("이미지", log)

    def test_no_plan_same_model_every_card(self):
        d, bp, log, rec = self.run_plan("", 3, quality="0")
        self.assertEqual([(r["lane"], r["cards"]) for r in rec], [(0, "0"), (1, "1"), (2, "2")])
        self.assertEqual(rec[0]["argv0"], ["bash", "/app/docker/entrypoint.sh"])
        self.assertEqual([r["argv0"] for r in rec[1:]], [["bash", "/app/single-user/start_qwen.sh"]] * 2)   # hyperqwen k≥1
        self.assertEqual([(r["prepare"], r["verify"]) for r in rec], [(None, None), ("0", "0"), ("0", "0")])
        self.assertTrue((d / "lanes" / "plan.txt").read_text().startswith(f"0 {bp} 0 qwen38-27b-demo-w4a16 hyperqwen-8cf642e 1"))

    def test_a_x4_same_as_v7(self):
        """T23 계획 a-x4 = v7 eval(PLAN 없음, 카드마다 A·hyperqwen)과 같은 차선·명령. 조건은 계획(CTX long·FAST_VARIANT 1)."""
        d, bp, log, rec = self.run_plan("a-x4", 4)
        self.assertEqual([(r["lane"], r["cards"]) for r in rec], [(0, "0"), (1, "1"), (2, "2"), (3, "3")])
        self.assertEqual(rec[0]["argv0"], ["bash", "/app/docker/entrypoint.sh"])
        self.assertEqual([r["argv0"] for r in rec[1:]], [["bash", "/app/single-user/start_qwen.sh"]] * 3)
        self.assertEqual([(r["prepare"], r["verify"]) for r in rec], [(None, None)] + [("0", "0")] * 3)
        self.assertTrue(all(r["model"].endswith("-fast") for r in rec))                 # FAST_VARIANT=1(E5~E11 기준)
        self.assertEqual((d / "cond.txt").read_text(), "CTX=long SPEC=mtp PREFIX_CACHE=1 FAST_VARIANT=1 cond=plan\n")
        self.assertEqual((d / "q.txt").read_text().split("\n")[:-1], [f"L0 http://127.0.0.1:{bp}/v1 qwen3.8-27b gsm-only"])   # 품질은 차선 0 만
        d2, bp2, log2, rec2 = self.run_plan("", 4, quality="0")                           # v7(PLAN 없음)
        key = lambda r: (r["lane"], r["cards"], tuple(r["argv0"]), r["prepare"], r["verify"], r["model"].rsplit("/", 1)[1], r["tp"])
        self.assertEqual([key(r) for r in rec], [key(r) for r in rec2])

    def test_a_all_fills_every_card(self):
        """T25 a-all(fill): 4장이면 a-x4 와 같은 차선·명령·조건, 8장이면 카드마다 차선 8개(포트 80k0, k=0..7)."""
        d, bp, log, rec = self.run_plan("a-all", 4)
        d2, bp2, log2, rec2 = self.run_plan("a-x4", 4)
        key = lambda r: (r["lane"], r["cards"], tuple(r["argv0"]), r["prepare"], r["verify"], r["model"].rsplit("/", 1)[1], r["tp"])
        self.assertEqual([key(r) for r in rec], [key(r) for r in rec2])
        self.assertEqual((d / "cond.txt").read_text(), (d2 / "cond.txt").read_text())
        d, bp, log, rec = self.run_plan("a-all", 8, quality="0")
        self.assertEqual([(r["lane"], r["port"], r["cards"]) for r in rec], [(k, bp + 10 * k, str(k)) for k in range(8)])
        self.assertEqual([r["argv0"] for r in rec[1:]], [["bash", "/app/single-user/start_qwen.sh"]] * 7)   # prepare·verify 는 차선 0 만
        self.assertIn("lanes plan=a-all n=8", log)
        self.assertTrue(all((d / "lanes" / f"lane-{k}.env").exists() for k in range(8)))

    def test_cmp_a_b1_x2_eight_cards(self):
        """T25 cmp-a-b1-x2: A·hyperqwen 2, A·vllm-patched 2, B1 tp2 2 = 8장. 품질은 갈래마다 첫 차선."""
        d, bp, log, rec = self.run_plan("cmp-a-b1-x2", 8)
        self.assertEqual([(r["lane"], r["cards"], r["tp"]) for r in rec],
                         [(0, "0", False), (1, "1", False), (2, "2", False), (3, "3", False), (4, "4,5", True), (5, "6,7", True)])
        self.assertEqual(rec[0]["argv0"], ["bash", "/app/docker/entrypoint.sh"])
        self.assertEqual(rec[1]["argv0"], ["bash", "/app/single-user/start_qwen.sh"])
        self.assertTrue(all(r["argv0"][1] == "-c" for r in rec[2:]))                   # vllm-patched
        q = sorted(x.split()[0] for x in (d / "q.txt").read_text().split("\n")[:-1])
        self.assertEqual(q, ["L0", "L2", "L4"])
        self.assertEqual((d / "cond.txt").read_text(), "CTX=long SPEC=mtp PREFIX_CACHE=1 FAST_VARIANT=0 cond=plan\n")
        d, bp, log, rec = self.run_plan("cmp-a-b1-x2", 4, expect_rc=3)
        self.assertIn("8 장용이다(지금 4)", log)

    def test_lane_dies_rc_and_next_lane_not_waiting(self):
        """T28: 차선 1 서버가 바로 끝나면 rc-1 이 남고, 차선 2 는 무게 올림 끝(20분)을 기다리지 않고 뜨고, 차선 1 품질은 건너뛴다."""
        d, bp, log, rec = self.run_plan("cmp-a-b1", 4, die_lane=1)
        self.assertEqual([r["lane"] for r in rec], [0, 2])
        self.assertEqual((d / "lanes" / "rc-1").read_text().strip(), "7")
        self.assertIn("lane 1 이 끝났거나 실패 — 기다리지 않고 lane 2 를 띄운다", log)
        self.assertIn("quality skip lane=1(차선이 끝났거나 실패)", log)
        self.assertTrue((d / "lanes" / "start-1").read_text().strip().isdigit())
        self.assertTrue((d / "lanes" / "pid-2").read_text().strip().isdigit())
        self.assertIn("ImportError: boom", (d / "vllm-1.log").read_text())
        q = sorted(x.split()[0] for x in (d / "q.txt").read_text().split("\n")[:-1])
        self.assertEqual(q, ["L0", "L2"])

    def test_quality_flags(self):
        """T28: PPL 은 추측 디코딩이 mtp 가 아닐 때만. hyperqwen 은 CF·계획 SPEC(없으면 표 M_HQ_SPEC), 그 밖은 표 M_SPEC."""
        import subprocess
        src = self.SH.read_text(encoding="utf-8")
        fn = src[src.index("quality_flags() {"):src.index("# ---- 5. vLLM")]
        for env, want in ((dict(M_ADAPTER="hyperqwen-8cf642e", SPEC="mtp"), "--gsm-only"),
                          (dict(M_ADAPTER="hyperqwen-8cf642e", SPEC="off", M_HQ_SPEC="mtp"), ""),
                          (dict(M_ADAPTER="hyperqwen-8cf642e", M_HQ_SPEC="mtp"), "--gsm-only"),
                          (dict(M_ADAPTER="hyperqwen-8cf642e", SPEC="dflash2"), ""),
                          (dict(M_ADAPTER="vllm-patched", M_SPEC="mtp", SPEC="off"), "--gsm-only"),
                          (dict(M_ADAPTER="vllm-patched", M_SPEC=""), ""), (dict(M_ADAPTER="vllm-stock"), ""),
                          (dict(M_ADAPTER="hyperqwen-8cf642e", SPEC="mtp", M_LANE_SPEC="off"), ""),     # 계획의 차선 값이 이긴다
                          (dict(M_ADAPTER="vllm-patched", M_SPEC="off", M_LANE_SPEC="off"), "")):
            r = subprocess.run(["bash", "-c", "set -u\n" + fn + "\nquality_flags"], env={"PATH": "/usr/bin:/bin", **env},
                               capture_output=True, text=True)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, want), env)

    def test_lane0_failure_starts_no_others(self):
        d, bp, log, rec = self.run_plan("cmp-a-b1", 4, quality="0", lane0_dies=True)
        self.assertEqual(rec, [])
        self.assertIn("lane 0 이 뜨지 않았다", log)
        self.assertNotIn("lane 1 start", log)


class TIdleStopLanes(unittest.TestCase):
    """v7 차선: /metrics 합, 짧은 응답 확인, ready 판정, 차선별 상태 응답."""
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("idle_stop", FIX_ROOT / "deploy/vast/idle_stop.py")
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.d = tempfile.mkdtemp()
        Path(self.d, "run-config.json").write_text(json.dumps({"src": "cf", "CTX": "long", "SPEC": "mtp", "PREFIX_CACHE": "1",
                                                               "FAST_VARIANT": "1", "QUALITY": "auto", "ignored": ""}))

    def test_ports_and_combine(self):
        self.assertEqual([self.m.lane_port(k) for k in range(4)], [8000, 8010, 8020, 8030])
        self.assertEqual([self.m.lane_status_port(k) for k in range(4)], [8001, 8011, 8021, 8031])
        self.assertTrue(self.m.lane_log(self.d, 0).endswith("/vllm.log"))
        self.assertTrue(self.m.lane_log(self.d, 3).endswith("/vllm-3.log"))
        self.assertIsNone(self.m.combine([None, None]))
        c = self.m.combine([{"vllm:num_requests_running": 1.0, "vllm:generation_tokens_total": 5.0}, None,
                            {"vllm:generation_tokens_total": 7.0}])
        self.assertEqual(c, {"vllm:num_requests_running": 1.0, "vllm:generation_tokens_total": 12.0})
        w = self.m.Watch(1800, 2700, 8 * 3600, 0)                  # 한 차선이 일하면 유휴가 아니다
        idle = {"vllm:generation_tokens_total": 5.0}
        w.observe(self.m.combine([idle, idle]), 100)
        self.assertIsNone(w.observe(self.m.combine([idle, dict(idle, **{"vllm:num_requests_running": 1.0})]), 5000))
        self.assertEqual(w.observe(self.m.combine([idle, idle]), 7000), "idle 30m")

    def lanes(self, n, health, smoke):
        return self.m.Lanes(self.d, n, "eval", smoke_fn=smoke, health_fn=lambda port: health.get(port, False))

    def test_lane_boot_limit_and_why(self):
        """T28: 띄운 뒤 30분(차선 0 은 45분) health 가 없으면 그 차선만 failed, why = 이유 + 로그 끝 3줄, failed-k, SIGTERM."""
        ld = Path(self.d, "lanes")
        ld.mkdir()
        now = [10_000.0]
        killed = []
        lanes = self.m.Lanes(self.d, 4, "eval", smoke_fn=lambda p: {"ok": True, "id": "x", "root": "/m", "text": "ok", "error": None},
                             health_fn=lambda port: port == 8000, now_fn=lambda: now[0], kill_fn=lambda pid: killed.append(pid) or [pid])
        for k in range(3):
            (ld / f"start-{k}").write_text(str(int(now[0])))
        (ld / "pid-1").write_text("4242\n")
        Path(self.m.lane_log(self.d, 1)).write_text("INFO a\n\x1b[32mINFO\x1b[0m chat-template ok\n\nINFO Waiting for core engine\nINFO still waiting\n")
        lanes.check()
        self.assertEqual([s["state"] for s in lanes.summary(None)][1:], ["starting", "starting", "starting"])
        now[0] += 29 * 60
        lanes.check()
        self.assertNotEqual(lanes.summary(None)[1]["state"], "failed")
        now[0] += 2 * 60                                              # 31분
        lanes.check()
        st = lanes.summary(None)
        self.assertEqual([s["state"] for s in st], ["checked", "failed", "failed", "starting"])   # 차선 0 은 health, 3 은 아직 안 띄움
        self.assertIn("기동 30분 넘음", st[1]["why"])
        self.assertIn("chat-template ok ⏎ INFO Waiting for core engine ⏎ INFO still waiting", st[1]["why"])
        self.assertNotIn("\x1b", st[1]["why"])
        self.assertIn("차선 로그 없음", st[2]["why"])
        self.assertEqual(killed, [4242])                                 # pid 파일이 있는 차선만
        self.assertTrue((ld / "failed-1").read_text().startswith("기동 30분 넘음"))
        self.assertIn("lane 1 failed", Path(self.d, "pod-log.tsv").read_text())
        self.assertFalse(lanes.all_failed())
        lanes.check()                                                    # 한 번만(다시 죽이지 않음)
        self.assertEqual(killed, [4242])

    def test_lane_rc_hold_and_all_failed(self):
        ld = Path(self.d, "lanes")
        ld.mkdir()
        now = [5_000.0]
        lanes = self.m.Lanes(self.d, 3, "eval", smoke_fn=lambda p: {}, health_fn=lambda port: False, now_fn=lambda: now[0],
                             kill_fn=lambda pid: [])
        (ld / "start-0").write_text(str(int(now[0])))
        now[0] += 40 * 60
        lanes.hold()                                                     # home 받기가 막 끝남 → 지금부터 센다
        now[0] += 10 * 60                                                # 띄운 지 50분이지만 hold 뒤 10분
        lanes.check()
        self.assertEqual(lanes.summary(None)[0]["state"], "starting")
        (ld / "rc-0").write_text("1\n")                                 # 서버가 끝났다 → 바로 failed
        Path(self.m.lane_log(self.d, 0)).write_text("Traceback\nModuleNotFoundError: No module named 'x'\n")
        lanes.check()
        st = lanes.summary(None)
        self.assertEqual([s["state"] for s in st], ["failed", "failed", "failed"])   # 차선 0 실패 → 안 띄운 차선도
        self.assertIn("차선 서버가 끝났다(rc=1)", st[0]["why"])
        self.assertIn("ModuleNotFoundError", st[0]["why"])
        self.assertIn("차선 0 이 실패해 띄우지 않았다", st[2]["why"])
        self.assertTrue(lanes.all_failed())

    def test_lane0_uses_boot_grace_and_ready_lane_dying(self):
        ld = Path(self.d, "lanes")
        ld.mkdir()
        now = [0.0]
        h = {8000: False}
        lanes = self.m.Lanes(self.d, 1, "eval", smoke_fn=lambda p: {"ok": True, "id": "x", "root": "/m", "text": "ok", "error": None},
                             health_fn=lambda port: h[port], now_fn=lambda: now[0], kill_fn=lambda pid: [])
        (ld / "start-0").write_text("0")
        now[0] = 44 * 60
        lanes.check()
        self.assertEqual(lanes.summary(None)[0]["state"], "starting")   # 차선 0 은 45분
        h[8000] = True
        lanes.check()
        self.assertNotEqual(lanes.summary(None)[0]["state"], "failed")
        h[8000] = False
        (ld / "rc-0").write_text("137")                                 # 뜬 뒤 죽음(OOM 등)도 failed
        lanes.check()
        self.assertEqual(lanes.summary(None)[0]["state"], "failed")
        self.assertIn("rc=137", lanes.summary(None)[0]["why"])

    def test_spec_log_tsv(self):
        """차선별 추측 디코딩 누계를 바뀐 차선만 한 줄씩(시각·차선·별칭·어댑터·초안 수·초안 토큰·수락 토큰·자리별)."""
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "lanes"))
        with open(os.path.join(d, "lanes", "plan.txt"), "w") as f:
            f.write("0 8000 0 qwen38-27b-demo-w4a16 hyperqwen-8cf642e 1\n1 8010 1 qwen38-27b-orig-w4a16q-int8h hyperqwen-8cf642e 1\n")
        rows = self.m.plan_rows(d)
        self.assertEqual(rows[1], ("qwen38-27b-orig-w4a16q-int8h", "hyperqwen-8cf642e"))
        st = [{"k": 0, "spec_decode": {"drafts": 10, "draft_tokens": 30, "accepted_tokens": 21, "per_pos": [0.9, 0.7, 0.5]}},
              {"k": 1, "spec_decode": None}]
        last = {}
        self.assertEqual(self.m.spec_tsv(d, st, rows, last, "T1"), 1)
        self.assertEqual(self.m.spec_tsv(d, st, rows, last, "T2"), 0)                 # 그대로면 안 쓴다
        st[0]["spec_decode"] = {"drafts": 12, "draft_tokens": 36, "accepted_tokens": 25, "per_pos": [0.9, 0.7, None]}
        self.assertEqual(self.m.spec_tsv(d, st, rows, last, "T3"), 1)
        lines = open(os.path.join(d, "spec-log.tsv")).read().splitlines()
        self.assertEqual(lines, ["T1\t0\tqwen38-27b-demo-w4a16\thyperqwen-8cf642e\t10\t30\t21\t0.9,0.7,0.5",
                                 "T3\t0\tqwen38-27b-demo-w4a16\thyperqwen-8cf642e\t12\t36\t25\t0.9,0.7,"])

    def test_spec_decode_metrics(self):
        text = ("# HELP x\nvllm:spec_decode_num_drafts_total{engine=\"0\",model_name=\"m\"} 100\n"
                "vllm:spec_decode_num_draft_tokens_total{engine=\"0\",model_name=\"m\"} 300\n"
                "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\",model_name=\"m\"} 210\n"
                "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"m\",position=\"0\"} 90\n"
                "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"m\",position=\"1\"} 70\n"
                "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"m\",position=\"2\"} 50\n"
                "vllm:num_requests_running 0\n")
        m = self.m.parse_metrics(text)
        self.assertEqual(m["vllm:spec_decode_num_accepted_tokens_per_pos_total"], 210)
        self.assertEqual(self.m.spec_view(m), {"drafts": 100, "draft_tokens": 300, "accepted_tokens": 210, "acceptance": 0.7,
                                               "mean_accept_len": 3.1, "per_pos": [0.9, 0.7, 0.5]})
        self.assertIsNone(self.m.spec_view({"vllm:num_requests_running": 0.0}))   # 추측 디코딩 끈 차선
        self.assertIsNone(self.m.spec_view(None))
        lanes = self.lanes(2, {}, lambda p: {})
        lanes.observe_metrics([m, None])
        st = lanes.summary(None)
        self.assertEqual((st[0]["spec_decode"]["acceptance"], st[1]["spec_decode"]), (0.7, None))
        lanes.observe_metrics([None, None])                             # 잠깐 안 보여도 마지막 누계는 남긴다
        self.assertEqual(lanes.summary(None)[0]["spec_decode"]["mean_accept_len"], 3.1)
        import importlib.util                                           # 변환 인스턴스 쪽 같은 계산(deploy/quant/spec_metrics.py)
        sp = importlib.util.spec_from_file_location("spec_metrics", FIX_ROOT / "deploy/quant/spec_metrics.py")
        sm = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(sm)
        self.assertEqual(sm.view(sm.parse(text)), self.m.spec_view(m))

    def test_lane_spec_meta(self):
        f = self.m.lane_spec
        self.assertEqual(f({"M_ADAPTER": "hyperqwen-8cf642e", "M_HQ_SPEC": "mtp"}, {"SPEC": "mtp"}), "mtp")
        self.assertEqual(f({"M_ADAPTER": "hyperqwen-8cf642e", "M_HQ_SPEC": "mtp", "M_LANE_SPEC": "off"}, {"SPEC": "mtp"}), "off")
        self.assertEqual(f({"M_ADAPTER": "vllm-patched", "M_SPEC": "mtp"}, None), "mtp")
        self.assertEqual(f({"M_ADAPTER": "vllm-stock", "M_SPEC": ""}, None), "off")
        self.assertIsNone(f({}, None))
        self.assertIn("spec", self.m.CONFIG_FIELDS)

    def test_kill_guard(self):
        self.assertFalse(self.m.lane_proc_ok(999999))                  # 없는 pid·/proc 없음(맥) → 건드리지 않음
        self.assertEqual(self.m.kill_tree(999999), [])

    def test_ready_rules(self):
        good = lambda port: {"ok": True, "id": "qwen3.8-27b", "root": "/workspace/stewart/models/X-fast", "text": "ok", "error": None}
        health = {8000: True, 8010: True, 8020: False}
        L = self.lanes(3, health, good)
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        st = L.summary(None)
        self.assertEqual([s["ready"] for s in st], [False, False, False])     # 차선 0 품질 점검 전이면 아무 차선도 ready 아님
        self.assertEqual(st[0]["why"], "품질 점검 pending")
        self.assertEqual((st[1]["state"], st[1]["why"]), ("checked", "차선 0 품질 점검 pending"))   # 1 은 짧은 응답 확인은 마침
        self.assertEqual(st[2]["state"], "starting")                          # /health 전
        self.assertTrue(st[1]["health"])
        Path(self.d, "quality-T1-fast1-long-pc1.log").write_text("T1 PPL x\nexit=0\n")
        health[8020] = True
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        st = L.summary(["a, 1", "b, 2", "c, 3"])
        self.assertEqual([s["ready"] for s in st], [True, True, True])
        self.assertEqual([s["gpu"] for s in st], ["a, 1", "b, 2", "c, 3"])
        self.assertEqual(len({s["config_sha"] for s in st}), 1)
        self.assertIn("lane 2 ready port=8020", Path(self.d, "pod-log.tsv").read_text())
        health[8010] = False                                                  # 나중에 죽으면 ready 가 내려간다
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        self.assertEqual([s["ready"] for s in L.summary(None)], [True, False, True])

    def test_mismatch_and_failures(self):
        def smoke(port):
            if port == 8010:
                return {"ok": True, "id": "qwen3.8-27b", "root": "/other/model", "text": "ok", "error": None}
            if port == 8020:
                return {"ok": False, "id": "qwen3.8-27b", "root": "/m", "text": "", "error": "빈 응답"}
            return {"ok": True, "id": "qwen3.8-27b", "root": "/m", "text": "ok", "error": None}
        Path(self.d, "quality-T1.log").write_text("exit=0\n")
        L = self.lanes(3, {8000: True, 8010: True, 8020: True}, smoke)
        with mock.patch("sys.stdout", io.StringIO()):
            for _ in range(self.m.SMOKE_TRIES + 1):
                L.check()
        st = L.summary(None)
        self.assertEqual([s["ready"] for s in st], [True, False, False])
        self.assertIn("모델·설정이 다르", st[1]["why"])                        # 다른 모델 root
        self.assertEqual(st[2]["state"], "failed")                            # 빈 응답 5번
        self.assertEqual(st[2]["smoke"]["error"], "빈 응답")
        Path(self.d, "quality-T2.log").write_text("exit=1\n")                 # 차선 0 품질 실패면 0 도 ready 아님
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        self.assertFalse(L.summary(None)[0]["ready"])
        self.assertEqual(L.summary(None)[0]["why"], "품질 점검 failed")

    def write_plan(self, lanes):
        """lanes = [(k, cards, alias, adapter, served, tp, quality)] → lanes/plan.txt·lane-k.env(serving.py 꼴)."""
        L = Path(self.d, "lanes")
        L.mkdir(exist_ok=True)
        rows = []
        for k, cards, alias, adapter, served, tp, q in lanes:
            Path(L, f"lane-{k}.env").write_text(f"M_ALIAS='{alias}'\nM_ADAPTER='{adapter}'\nM_SERVED='{served}'\nM_TP='{tp}'\nM_DIR='D-{alias}'\n"
                                                f"M_REV='{'c' * 40}'\nM_IMAGE='hyperqwen'\nM_PATCHSET='vllm-0.30.0'\n")
            rows.append(f"{k} {8000 + 10 * k} {cards} {alias} {adapter} {int(q)}")
        Path(L, "plan.txt").write_text("\n".join(rows) + "\n")

    def test_v8_plan_lanes_ready_by_own_plan(self):
        A, B1 = "qwen38-27b-demo-w4a16", "qwen36-35b-a3b-demo-w4a16"
        self.write_plan([(0, "0", A, "hyperqwen-8cf642e", "qwen3.8-27b", 1, True), (1, "1", A, "vllm-patched", "qwen3.8-27b", 1, True),
                         (2, "2,3", B1, "vllm-patched", "qwen3.6-35b-a3b", 2, True)])
        served = {8000: "qwen3.8-27b", 8010: "qwen3.8-27b", 8020: "qwen3.6-35b-a3b"}
        sm = lambda port: {"ok": True, "id": served[port], "root": f"/r{port}", "text": "ok", "error": None}
        L = self.lanes(3, {8000: True, 8010: True, 8020: True}, sm)
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        self.assertEqual([s["ready"] for s in L.summary(None)], [False, False, False])   # 품질 전
        Path(self.d, "quality-L2-T-b1.log").write_text("T PPL\nexit=0\n")                  # B1 차선만 먼저 통과
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        st = L.summary(["g0", "g1", "g2", "g3"])
        self.assertEqual([s["ready"] for s in st], [False, False, True])                 # 차선마다 제 품질(차선 0 을 기다리지 않음)
        self.assertEqual(st[1]["why"], "품질 점검(차선 1) pending")
        self.assertEqual(st[2]["alias"], B1)                                            # evalq(v12.11d)가 읽는 칸
        self.assertEqual(st[2]["config"], {"lane": 2, "port": 8020, "cards": "2,3", "model": B1, "adapter": "vllm-patched",
                                           "served_name": "qwen3.6-35b-a3b", "model_rev": "c" * 40, "tp": 2, "image": "hyperqwen",
                                           "patchset": "vllm-0.30.0", "plan": "", "spec": "off", "samples": "card2"})   # 칸 이름
        self.assertEqual(st[0]["config"]["patchset"], "")                               # hyperqwen 어댑터는 패치 묶음 칸 비움
        self.assertTrue(st[2]["quality"])
        self.assertEqual((st[2]["gpu"], st[2]["gpu_cards"]), ("g2", ["g2", "g3"]))
        self.assertEqual(len({s["config_sha"] for s in st}), 3)                         # 차선마다 모델·어댑터가 달라 sha 도 다름
        Path(self.d, "quality-L0-T-a.log").write_text("exit=0\n")
        Path(self.d, "quality-L1-T-a.log").write_text("exit=1\n")                       # 차선 1 품질 실패
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        st = L.summary(None)
        self.assertEqual([s["ready"] for s in st], [True, False, True])
        self.assertEqual(st[1]["why"], "품질 점검(차선 1) failed")

    def test_v8_served_name_and_quality_ref(self):
        A = "qwen38-27b-demo-w4a16"
        self.write_plan([(0, "0", A, "hyperqwen-8cf642e", "qwen3.8-27b", 1, True), (1, "1", A, "hyperqwen-8cf642e", "qwen3.8-27b", 1, False),
                         (2, "2", A, "hyperqwen-8cf642e", "qwen3.8-27b", 1, False)])
        sm = lambda port: {"ok": True, "id": "wrong-name" if port == 8020 else "qwen3.8-27b", "root": "/m", "text": "ok", "error": None}
        L = self.lanes(3, {8000: True, 8010: True, 8020: True}, sm)
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        st = L.summary(None)
        self.assertEqual(st[1]["why"], "품질 점검(차선 0) pending")                     # 품질 차선이 아니면 같은 별칭·어댑터의 품질 차선을 본다
        Path(self.d, "quality-L0-T.log").write_text("exit=0\n")
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        st = L.summary(None)
        self.assertEqual([s["ready"] for s in st], [True, True, False])
        self.assertIn("서빙 이름 wrong-name", st[2]["why"])

    def test_v8_status_view_has_lane_meta(self):
        self.write_plan([(0, "0", "a", "hyperqwen-8cf642e", "qa", 1, True), (1, "1,2", "b", "vllm-patched", "qb", 2, True)])
        for k in (0, 1):
            Path(self.d, f"quality-L{k}-T.log").write_text("exit=0\n")
        sm = lambda port: {"ok": True, "id": {8000: "qa", 8010: "qb"}[port], "root": "/m", "text": "ok", "error": None}
        L = self.lanes(2, {8000: True, 8010: True}, sm)
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        w = self.m.Watch(1800, 2700, 3600, 0)
        g = ["c0", "c1", "c2"]
        s0 = self.m.status(self.d, w, 5, g, None, L, 0, "eval")
        s1 = self.m.status(self.d, w, 5, g, None, L, 1, "eval")
        self.assertEqual([l["alias"] for l in s0["lanes"]], ["a", "b"])
        self.assertTrue(all(l["ready"] for l in s0["lanes"]))
        self.assertEqual((s1["gpu"], s1["config"]["model"], s1["config"]["adapter"], s1["config"]["tp"], s1["config"]["cards"],
                          s1["config"]["gpus"], s1["config"]["lanes"], s1["config"]["lane"], s1["config"]["served_name"],
                          s1["config"]["samples"]), ("c1", "b", "vllm-patched", 2, "1,2", 3, 2, 1, "qb", "card1"))   # tp 2: 평가기 표본은 첫 카드
        self.assertEqual(s1["lane"]["gpu_cards"], ["c1", "c2"])

    # ── T24 ──
    def test_v9_idle_by_role_and_left(self):
        import subprocess, sys as _sys
        script = str(FIX_ROOT / "deploy/vast/idle_stop.py")
        for role, lanes, env, want in (("serve", "1", {}, "idle=90.0m"), ("eval", "4", {}, "idle=30.0m"), ("serve", "1", {"IDLE_MIN": "45"}, "idle=45.0m")):
            d = tempfile.mkdtemp()
            r = subprocess.run([_sys.executable, script, "--state-dir", d, "--role", role, "--lanes", lanes, "--once", "--status-port", "0"],
                               capture_output=True, text=True, timeout=60, env=dict(os.environ, **env))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"role={role}", Path(d, "pod-log.tsv").read_text())
            self.assertIn(want, Path(d, "pod-log.tsv").read_text(), (role, env))
        w = self.m.Watch(90 * 60, 2700, 8 * 3600, 0)
        self.assertIsNone(w.idle_left(10))                                     # 서버를 못 봤으면 없음
        w.observe({"vllm:generation_tokens_total": 1.0}, 100)
        self.assertEqual(w.idle_left(100 + 600), 90 * 60 - 600)
        st = self.m.status(self.d, w, 700, None)
        self.assertEqual(st["idle_left_s"], 90 * 60 - 600)
        self.assertEqual(st["limits"]["idle_s"], 90 * 60)

    def test_v9_boot_grace_waits_for_home_fetch(self):
        W = self.m.Watch
        w = W(1800, 2700, 8 * 3600, 0)
        for t in range(0, 3601, 60):                                           # 60분 동안 받는 중(hold) — 유예가 지나도 멈추지 않는다
            w.hold(t)
            self.assertIsNone(w.observe(None, t), t)
        self.assertIsNone(w.observe(None, 3600 + 2699))                        # 받기 끝 뒤 45분까지는 기다림
        self.assertEqual(w.observe(None, 3600 + 2700), "boot_timeout 45m")
        self.assertEqual(W(1800, 2700, 8 * 3600, 0).observe(None, 2700), "boot_timeout 45m")   # hold 가 없으면 그대로
        w = W(1800, 2700, 3600, 0)
        w.hold(3599)
        self.assertEqual(w.observe(None, 3600), "max_uptime 1.0h")             # 켜짐 상한은 hold 와 상관없다

    def test_v9_model_fetch_in_status(self):
        self.assertIsNone(self.m.model_fetch(self.d))
        Path(self.d, "model-fetch.json").write_text(json.dumps({
            "alias": "qwen38-27b-demo2-w4a16-int8h", "digest": "sha256:" + "a" * 64, "state": "receiving", "files_total": 12,
            "files_done": 3, "bytes_total": 19_000_000_000, "bytes_done": 4_200_000_000, "rate_Bps": 52_000_000.5,
            "current": "model-00004-of-00007.safetensors", "started": "2026-11-09T14:00:00Z", "updated": "2026-11-09T14:01:21Z",
            "secret": "x", "nested": {"a": 1}}))
        mf = self.m.model_fetch(self.d)
        self.assertEqual(set(mf), set(self.m.FETCH_KEYS))
        self.assertEqual((mf["state"], mf["bytes_done"], mf["files_done"]), ("receiving", 4_200_000_000, 3))
        w = self.m.Watch(1800, 2700, 3600, 0)
        st = self.m.status(self.d, w, 5, ["g0"], None, None, 0, "serve", gpu_ts=time.time() - 42)
        self.assertEqual(st["model_fetch"], mf)
        self.assertTrue(40 <= st["gpu_age_s"] <= 44)                           # 표본 나이(E13: 낡은 표본이 표시 없이 실림)
        self.assertRegex(st["gpu_ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        Path(self.d, "model-fetch.json").write_text("{not json")
        self.assertIsNone(self.m.model_fetch(self.d))
        self.assertEqual(self.m.GPU_SAMPLE_S, 60)

    def test_quality_state(self):
        cfg = json.loads(Path(self.d, "run-config.json").read_text())
        self.assertEqual(self.m.quality_state(self.d, cfg), "pending")
        Path(self.d, "lanes").mkdir()
        Path(self.d, "lanes", "quality-ok-0").write_text("")                  # 재시작: 앞 부팅에서 통과(표시 파일)
        self.assertEqual(self.m.quality_state(self.d, cfg), "ok")
        self.assertEqual(self.m.quality_state(self.d, cfg, 1), "pending")    # 차선마다 따로
        self.assertEqual(self.m.quality_state(self.d, dict(cfg, QUALITY="0")), "ok")
        Path(self.d, "quality-T9.log").write_text("T9 PPL\n")
        self.assertEqual(self.m.quality_state(self.d, cfg), "pending")      # 이번 부팅 점검이 돌고 있으면 그 결과를 기다린다

    def test_lane_status_views(self):
        Path(self.d, "vllm.log").write_text("INFO GPU KV cache size: 100 tokens\n")
        Path(self.d, "vllm-1.log").write_text("INFO GPU KV cache size: 222 tokens\n  PASS  x\n")
        Path(self.d, "quality-T1.log").write_text("exit=0\n")
        ok = lambda port: {"ok": True, "id": "q", "root": "/m", "text": "ok", "error": None}
        L = self.lanes(2, {8000: True, 8010: True}, ok)
        with mock.patch("sys.stdout", io.StringIO()):
            L.check()
        w = self.m.Watch(1800, 2700, 3600, 0)
        gpus = ["0, 24564, 90, 300, 350, 1900, 9751, 0x0", "1, 24564, 10, 120, 350, 1900, 9751, 0x0"]
        srvs = [self.m.serve_status(0, (lambda k=k: self.m.status(self.d, w, 50, gpus, None, L, k, "eval"))) for k in range(2)]
        try:
            import urllib.request
            got = []
            for srv in srvs:
                with LOCAL.open(f"http://127.0.0.1:{srv.server_address[1]}/status", timeout=5) as r:
                    got.append(json.loads(r.read()))
        finally:
            for srv in srvs:
                srv.shutdown()
                srv.server_close()
        s0, s1 = got
        self.assertEqual((s0["role"], len(s0["lanes"]), s0["gpu"]), ("eval", 2, gpus[0]))
        self.assertTrue(all(l["ready"] for l in s0["lanes"]))
        self.assertEqual(s0["config"]["lane"], 0)
        self.assertNotIn("lanes", s1)                                        # 차선 보기에는 요약 없음
        self.assertEqual((s1["gpu"], s1["config"]["lane"], s1["config"]["port"], s1["config"]["gpus"]), (gpus[1], 1, 8010, 2))
        self.assertEqual(s1["vllm_kv"], ["INFO GPU KV cache size: 222 tokens"])
        self.assertEqual(s1["verify"], ["  PASS  x"])
        self.assertTrue(s1["lane"]["ready"])
        self.assertEqual(s1["config"]["CTX"], "long")                        # 평가기 machine_info 가 남기는 칸

    def test_smoke_against_fake_server(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading as th
        seen = {}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _j(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            def do_GET(self):
                self._j({"data": [{"id": "qwen3.8-27b", "root": "/workspace/stewart/models/M-fast"}]})
            def do_POST(self):
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._j({"choices": [{"message": {"content": " ok "}}]})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        th.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = self.m.smoke(srv.server_address[1], timeout=5)
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(r, {"ok": True, "id": "qwen3.8-27b", "root": "/workspace/stewart/models/M-fast", "text": "ok", "error": None})
        self.assertEqual((seen["body"]["temperature"], seen["body"]["max_tokens"]), (0, 16))
        self.assertEqual(seen["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(self.m.smoke(1, timeout=1)["ok"])                   # 닫힌 포트

    def test_once_cli(self):
        import subprocess, sys as _sys
        r = subprocess.run([_sys.executable, str(FIX_ROOT / "deploy/vast/idle_stop.py"), "--state-dir", self.d,
                            "--role", "eval", "--lanes", "3", "--once", "--status-port", "0"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        j = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual((j["metrics"], j["lanes"]), (False, ["starting"] * 3))
        Path(self.d, "lanes").mkdir()                                       # T28: 차선 셋이 모두 끝났다 → HALT 한 줄(900초 뒤 정지)
        for k in range(3):
            Path(self.d, "lanes", f"rc-{k}").write_text("1")
        r = subprocess.run([_sys.executable, str(FIX_ROOT / "deploy/vast/idle_stop.py"), "--state-dir", self.d,
                            "--role", "eval", "--lanes", "3", "--once", "--status-port", "0", "--dry-run"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1])["lanes"], ["failed"] * 3)
        pl = Path(self.d, "pod-log.tsv").read_text()
        self.assertIn("HALT: 모든 차선 실패(0: 차선 서버가 끝났다(rc=1)", pl)
        self.assertIn("— 900초 뒤 정지", pl)
        self.assertIn("lane_boot=30.0m", pl)
        for bad in (["--role", "serve", "--lanes", "2"], ["--role", "eval", "--lanes", "9"], ["--role", "box"]):
            r = subprocess.run([_sys.executable, str(FIX_ROOT / "deploy/vast/idle_stop.py"), "--state-dir", self.d,
                                "--once"] + bad, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0, bad)


class TTemplateEnv(unittest.TestCase):
    def test_bundle_roundtrip(self):
        import gzip as gz
        import importlib.util
        import tarfile
        spec = importlib.util.spec_from_file_location("mte", FIX_ROOT / "deploy/vast/make_template_env.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with tarfile.open(fileobj=io.BytesIO(gz.decompress(m.bundle()))) as t:
            names = t.getnames()
            py = t.extractfile("idle_stop.py").read().decode()
            sh = t.extractfile("pod-start.sh").read().decode()
            sv = {n: t.extractfile(n).read() for n in names if n.startswith("serving/")}
        self.assertEqual(sorted(n for n in names if not n.startswith("serving/")), ["idle_stop.py", "pod-start.sh"])
        root = FIX_ROOT / "deploy/serving"
        for n, data in sv.items():                                   # serving/ 은 바이트 그대로(패치 sha 가 맞아야 한다)
            self.assertEqual(data, (root / n[len("serving/"):]).read_bytes(), n)
        for must in ("serving/serving.sh", "serving/serving.py", "serving/patchset.py", "serving/models.toml",
                     "serving/patches/vllm-0.30.0/series", "serving/patches/vllm-0.30.0/SHA256SUMS",
                     "serving/patches/hyperqwen-8cf642e.sha256", "serving/pins/qwen38-27b-demo-w4a16.sha256",
                     "serving/models.local.toml", "serving/templates/qwen38-27b-chat.jinja"):
            self.assertIn(must, sv)
        table = sv["serving/models.local.toml"].decode()                  # 모델 A 줄은 local 표에
        svsh = sv["serving/serving.sh"].decode()
        compile(py, "idle_stop.py", "exec")
        self.assertTrue(sh.startswith("#!/bin/bash"))
        for must in ("HF_HUB_OFFLINE=1", "--accept-dns=false", "VLLM_NO_USAGE_STATS=1", "TS_VER=1.104.5", '--auth-key="file:$KEYF"',
                     "720cbbccbfc03d8b55df72afc19e5a72e31b0f65934700aa6fa38074d4ad7bf7", "serve_lane 0 8000"):
            self.assertIn(must, sh)
        for must in ('repo = "example-org/Demo-Qwen3.8-27B-W4A16-AutoRound"',
                     'rev = "c20530baefe3e77ccfc6891c2b50cce7ea28bf1e"', 'rev = "124c14e7e8c7d2f5402933b9af368e772a9fcf0c"'):
            self.assertIn(must, table)
        self.assertIn('"HOST": "127.0.0.1"', sv["serving/serving.py"].decode())
        self.assertNotIn("mode=json", sh)                            # 버전 조회 안 함
        self.assertNotIn("dbirks", sh)                               # 옛 BASE 를 받지 않는다
        import re as _re
        def shas(name):                                              # 고정값 줄: "sha256  경로"
            if name == "Q_SHA":
                blk = sh.split(name + "='", 1)[1].split("'", 1)[0].splitlines()
            else:
                blk = [l for l in sv[name].decode().splitlines() if l and not l.startswith("#")]
            self.assertTrue(all(_re.fullmatch(r"[0-9a-f]{64}  \S+", l) for l in blk), blk)
            return [l.split("  ", 1)[1] for l in blk]
        base = shas("serving/pins/qwen38-27b-demo-w4a16.sha256")
        self.assertEqual(len(base), 20)
        for f in [f"model-0000{i}-of-00007.safetensors" for i in range(1, 8)] + [
                "model_extra_tensors.safetensors", "model.safetensors.index.json", "mtp_draft_vocab_ids.pt",
                "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
            self.assertIn(f, base)
        self.assertEqual(sorted(shas("serving/pins/qwen38-27b-demo-w4a16.fast.sha256")), ["config.json", "model-00007-of-00007.safetensors",
                         "model.safetensors.index.json", "model_extra_tensors.safetensors", "mtp_draft_vocab_ids.pt"])
        self.assertEqual(len(shas("Q_SHA")), 3)
        # 대조가 표시 파일보다 먼저(대조에 실패하면 HALT 해 표시가 남지 않는다)
        self.assertLess(svsh.index('pin_check "$MDIR"'), svsh.index('echo "$M_REV" > "$MDIR/.stewart-rev"'))
        self.assertLess(sh.index("serving_fetch"), sh.index("export HF_HUB_OFFLINE=1"))
        self.assertLess(sh.index("serving_prepare"), sh.rindex("trap - ERR"))      # 패치 실패는 HALT 구간 안(v14: halt 안에도 trap - ERR)
        self.assertIn('|| halt "$3 sha256', sh)
        self.assertIn('echo "$M_FAST_REV" > "$S/models/.fast-rev"', svsh)       # 재시작 때 다시 받지 않는다
        self.assertIn('grep -qxF "$SIG" "$QDONE"', sh)                       # 품질은 설정이 바뀔 때만
        import re as _re2
        self.assertIsNone(_re2.search(r"(^|[\s;&|(`$])eval\s", sh))          # 셸 eval 명령은 없다(역할 낱말 eval 은 있음)
        self.assertIn('--hostname="$TS_HOST"', sh)                          # 이름은 역할 표에서(v7)
        self.assertNotIn("--hostname=stewart-gpu ", sh)
        self.assertIn('--role "$ROLE" --lanes "$LANES"', sh)
        self.assertLess(sh.index('halt "STEWART_ROLE'), sh.index('"$S/bin/tailscaled"'))   # 역할을 모르면 이름을 잡기 전에 멈춘다
        self.assertIn('serve_lane "$1" "$2" "$S/lanes/lane-$1.env"', sh)    # 서빙 어댑터를 차선마다
        self.assertNotIn("runpod", sh.lower())
        self.assertIn("console.vast.ai/api/v0/instances/$CONTAINER_ID/", sh)
        self.assertNotIn('Bearer $(cat', sh)                         # 키를 curl 명령 줄에 넣지 않는다
        _ls = [l.strip() for l in sh.splitlines()]
        self.assertLess(_ls.index("unset CONTAINER_API_KEY"), next(i for i, l in enumerate(_ls) if l.startswith('"$S/bin/tailscaled"')))
        self.assertLessEqual(len(m.ARGS), m.ARGS_MAX)                # Vast Args 256자
        self.assertTrue(m.ARGS.startswith("bash -c '") and m.ARGS.endswith(";sleep 1d'"))
        self.assertIn(f'echo "{m.boot_sha()}  /dev/shm/b"|sha256sum -c&&exec bash /dev/shm/b', m.ARGS)   # SB 를 대조한 뒤에만 돌린다(T23)
        self.assertEqual(hashlib.sha256(base64.b64decode(m.boot_b64())).hexdigest(), m.boot_sha())
        self.assertIn(" -e SB=" + m.boot_b64(), m.options("u", "0" * 64, "c"))
        self.assertLessEqual(len(m.options("https://gist.githubusercontent.com/" + "x" * 200, "0" * 64,   # SB 포함
                                            "https://gist.githubusercontent.com/" + "y" * 200)), m.OPTIONS_MAX)
        self.assertIn(" -e CF=", m.options("u", "0" * 64, "c"))
        lines = [l.strip() for l in sh.splitlines()]
        i_unset = lines.index("unset TS_AUTHKEY")
        cmds = [l for l in lines[1:i_unset] if l]
        # 키를 환경에서 지우기 전에는 셸 내장 동작만(파일로 옮기기) — 자식 프로세스 없음
        self.assertTrue(all(l.split()[0] in ("set", "umask", "KEYF=/dev/shm/ts-authkey", "printf") for l in cmds), cmds)
        self.assertLess(i_unset, next(i for i, l in enumerate(lines) if "tailscaled\" --tun" in l or l.startswith('"$S/bin/tailscaled"')))
        self.assertNotIn("TS_AUTHKEY=", sh.replace('"${TS_AUTHKEY:?', ""))  # 키 값은 템플릿 비밀에서만
        self.assertNotIn("STEWART_POD_FILES", m.ARGS + sh)


if __name__ == "__main__":
    unittest.main()
