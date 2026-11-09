"""변환 인스턴스의 창구 둘. 표준 라이브러리만. run-convert.sh 가 테일넷에 들어온 뒤 띄운다.

  python quantsrv.py --q /workspace/quant --server-ip <SERVER_TS_IP> [--status-port 8001] [--export-port 8090] [--ts-cmd "<tailscale …>"]

- :8001 상태(serve·eval 의 감시기와 같은 자리, 래퍼 ② 가 본다)
  - GET  /status : {"role":"quant","instance_id","label","stage","done":[…],"halted","log_tail":[…],"export_ready","exported","model_fetch":null}
  - POST /logout : 한 번만 테일넷에서 나간다(래퍼가 destroy 전에 부른다)
- :8090 내보내기(보관 서버 SERVER_TS_IP 에게만 — 아래)
  - GET  /export               : export.json(세트별 files[name, sha256, size] + report). 아직이면 409
  - GET  /export/<별칭>/<이름> : 그 파일 그대로(목록 안 이름만)
  - POST /export/done          : 집이 다 받았다고 알림 → done/exported 를 남긴다(run-convert 가 보고 정지)
- 둘 다 127.0.0.1 에서만 듣는다. 사용자 공간 Tailscale 이 테일넷 포트를 그리로 넘긴다.
  - 그때 출발 주소는 127.0.0.1 로 보인다. 그래서 내보내기는 "출발 주소가 보이면 SERVER_TS_IP 만" 이다.
  - 실제 막음은 Tailscale ACL(tag:server → tag:gpu tcp:8090)이 한다. modelrecv.py 와 같다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,47}$")
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
STAGES = ["0-image", "1-tools", "2-src", "3-data", "3b-offline", "4-pad", "5-kl-pad", "6-quant", "7-kl-quant",
          "8-prepare", "9-serve-quality", "10-pins", "11-built", "audit", "11-export"]
CHUNK = 1 << 20


class Ctx:
    def __init__(self, q: Path, server_ip: str, ts_cmd: list[str] | None):
        self.q, self.server_ip, self.ts_cmd = q, server_ip, ts_cmd
        self.logged_out = False
        self.lock = threading.Lock()

    def status(self) -> dict:
        root = self.q / "done"
        done = sorted((p.name for p in root.glob("*") if p.is_file()), key=lambda n: STAGES.index(n) if n in STAGES else 99)
        stage = next((s for s in reversed(STAGES) if s in done), None)
        # 대상마다(done/<별칭>/<단계>) 마지막 단계. 대상 하나만 돌던 옛 꼴은 위 done·stage 그대로
        targets = {}
        for d in sorted(p for p in root.glob("*") if p.is_dir()):
            ds = sorted((p.name for p in d.glob("*")), key=lambda n: STAGES.index(n) if n in STAGES else 99)
            targets[d.name] = next((s for s in reversed(STAGES) if s in ds), ds[-1] if ds else None)
        log = self.q / "convert-log.tsv"
        tail = log.read_text(errors="replace").splitlines()[-10:] if log.exists() else []
        return {"role": "quant", "instance_id": os.environ.get("CONTAINER_ID"), "label": os.environ.get("VAST_CONTAINERLABEL"),
                "stage": stage, "done": done, "targets": targets, "halted": any("HALT:" in ln for ln in tail),
                "log_tail": [ln[:300] for ln in tail], "export_ready": (self.q / "export.json").exists(),
                "exported": (self.q / "done" / "exported").exists(), "model_fetch": None, "updated": round(time.time())}

    def export_index(self) -> dict | None:
        p = self.q / "export.json"
        return json.loads(p.read_text()) if p.exists() else None


class Base(BaseHTTPRequestHandler):
    ctx: Ctx
    server_version = "stewart-quantsrv"
    sys_version = ""

    def log_message(self, fmt, *args):
        sys.stderr.write("quantsrv: %s %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StatusH(Base):
    def do_GET(self):
        if self.path == "/status":
            return self._json(200, self.ctx.status())
        self._json(404, {"error": "없는 경로"})

    def do_POST(self):
        if self.path != "/logout":
            return self._json(404, {"error": "없는 경로"})
        with self.ctx.lock:
            if self.ctx.logged_out:
                return self._json(200, {"ok": True, "already": True})
            rc = None
            if self.ctx.ts_cmd:
                rc = subprocess.run(self.ctx.ts_cmd + ["logout"], capture_output=True, timeout=30).returncode
            self.ctx.logged_out = rc in (None, 0)
            return self._json(200 if self.ctx.logged_out else 500, {"ok": self.ctx.logged_out, "rc": rc})


class ExportH(Base):
    def _peer_ok(self) -> bool:
        peer = self.client_address[0]
        if peer == self.ctx.server_ip or peer in LOOPBACK:
            return True
        self._json(403, {"error": "보낸 쪽 주소가 보관 서버가 아니다"})
        return False

    def do_GET(self):
        if not self._peer_ok():
            return
        idx = self.ctx.export_index()
        if self.path == "/export":
            return self._json(200, idx) if idx else self._json(409, {"error": "아직 내보낼 것이 없다", "stage": self.ctx.status()["stage"]})
        m = re.fullmatch(r"/export/([^/]+)/([^/]+)", self.path)
        if not m or not idx:
            return self._json(404, {"error": "없는 경로"})
        alias, name = m.group(1), m.group(2)
        st = next((s for s in idx["sets"] if s["alias"] == alias), None)
        f = next((x for x in (st or {}).get("files", []) if x["name"] == name), None)
        if st is None or f is None or not NAME_RE.match(name) or not ALIAS_RE.match(alias):
            return self._json(404, {"error": "목록에 없다"})
        p = self.ctx.q / st["dir"] / name
        if not p.is_file() or p.stat().st_size != f["size"]:
            return self._json(500, {"error": "파일이 목록과 다르다"})
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(f["size"]))
        self.end_headers()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(CHUNK), b""):
                self.wfile.write(b)

    def do_POST(self):
        if not self._peer_ok():
            return
        if self.path != "/export/done":
            return self._json(404, {"error": "없는 경로"})
        if not self.ctx.export_index():
            return self._json(409, {"error": "아직 내보낼 것이 없다"})
        (self.ctx.q / "done" / "exported").touch()
        self._json(200, {"ok": True})


def serve(ctx: Ctx, listen: str, status_port: int, export_port: int) -> list[ThreadingHTTPServer]:
    out = []
    for port, h in ((status_port, StatusH), (export_port, ExportH)):
        cls = type(h.__name__, (h,), {"ctx": ctx})
        srv = ThreadingHTTPServer((listen, port), cls)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        out.append(srv)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--q", required=True)
    ap.add_argument("--server-ip", required=True)
    ap.add_argument("--status-port", type=int, default=8001)
    ap.add_argument("--export-port", type=int, default=8090)
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--ts-cmd", default="")
    a = ap.parse_args(argv)
    ctx = Ctx(Path(a.q), a.server_ip, shlex.split(a.ts_cmd) if a.ts_cmd else None)
    serve(ctx, a.listen, a.status_port, a.export_port)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
