"""파드의 모델 받는 창구: 보관 서버가 먼저 연결해 home 별칭의 파일을 밀어 넣는다(push). 파드 → 집 연결은 없다.

  python modelrecv.py --alias <별칭> --dir <모델 폴더> --server-ip <SERVER_TS_IP> [--port 8090] [--timeout-min 60]
                      [--status <model-fetch.json>] [--listen 127.0.0.1]

- 받을 목록은 묶음 안 models.toml 의 files(이름·sha256·크기)에서만 온다. 보내는 쪽은 목록을 바꿀 수 없다.
- GET  /need          → {"alias", "digest", "remaining", "files": [{name, sha256, size, have}]}
- PUT  /files/<이름>  → 목록 안 이름만, Content-Length = 목록 크기만. 숨은 임시 파일에 쓰며 sha256 을 센 뒤 맞으면 이름을 바꾼다.
  - 끊기면(덜 받음) 임시 파일을 지우고 400(다시 보내면 된다).
  - 다 받았는데 sha 가 다르면 422 를 주고 창구를 닫으며 exit 3(부르는 쪽이 HALT + 정지).
- 다 받으면 창구를 닫고 exit 0. --timeout-min 안에 못 받으면 exit 2. 그 밖 오류는 exit 1.
- 이미 폴더에 있는 파일은 시작할 때 sha 를 세어 맞으면 받은 것으로 친다(재시작에 이어 받기).
- 보낸 쪽 주소: 출발 주소가 보이면 --server-ip 만 받는다.
  - 파드의 Tailscale 은 사용자 공간 모드라, 테일넷으로 들어온 TCP 를 tailscaled 가 127.0.0.1:<포트> 로 넘긴다. 이때 출발 주소는 127.0.0.1 이다.
  - 그래서 127.0.0.1 도 받는다(--listen 127.0.0.1 이라 파드 밖에서는 테일넷으로만 닿는다).
  - 누가 보낼 수 있는지는 Tailscale ACL(tag:server → tag:gpu tcp:8090 만)이, 무엇을 받는지는 sha 목록이 지킨다.
- --status 파일: 진행(받은 바이트·속도) JSON. 감시기 /status 의 model_fetch 가 읽는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serving as SV  # noqa: E402

CHUNK = 1 << 20
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class State:
    def __init__(self, alias: str, files: list[dict], d: Path, status: Path | None):
        self.alias, self.dir, self.status_path = alias, d, status
        self.files = {f["name"]: dict(f, have=False) for f in files}
        self.digest = SV.home_digest(files)
        self.lock = threading.Lock()
        self.busy: set[str] = set()
        self.bytes_done = 0
        self.first_byte: float | None = None
        self.started = time.time()
        self.state = "waiting"
        self.exit_code: int | None = None
        self.current = ""
        self._last_status = 0.0
        d.mkdir(parents=True, exist_ok=True)
        for name, f in self.files.items():
            p = d / name
            if p.is_file() and p.stat().st_size == f["size"] and file_sha(p) == f["sha256"]:
                f["have"] = True
                self.bytes_done += f["size"]
        self.write_status(force=True)

    @property
    def bytes_total(self) -> int:
        return sum(f["size"] for f in self.files.values())

    def remaining(self) -> int:
        return sum(1 for f in self.files.values() if not f["have"])

    def write_status(self, force: bool = False, extra: int = 0) -> None:
        if not self.status_path:
            return
        now = time.time()
        if not force and now - self._last_status < 1.0:
            return
        self._last_status = now
        done = self.bytes_done + extra
        rate = (done / (now - self.first_byte)) if self.first_byte and now > self.first_byte else None
        d = {"alias": self.alias, "digest": self.digest, "state": self.state,
             "files_total": len(self.files), "files_done": len(self.files) - self.remaining(),
             "bytes_total": self.bytes_total, "bytes_done": done, "rate_Bps": round(rate) if rate else None,
             "current": self.current, "started": round(self.started), "updated": round(now)}
        tmp = self.status_path.with_name(self.status_path.name + ".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, self.status_path)


def make_handler(st: State, server_ip: str, stop: threading.Event):
    class H(BaseHTTPRequestHandler):
        server_version = "stewart-modelrecv"
        sys_version = ""

        def log_message(self, fmt, *args):   # 요청 줄만 짧게
            sys.stderr.write("modelrecv: %s %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _peer_ok(self) -> bool:
            peer = self.client_address[0]
            if peer == server_ip or peer in LOOPBACK:
                return True
            self._send(403, {"error": "보낸 쪽 주소가 보관 서버가 아니다"})
            return False

        def do_GET(self):
            if not self._peer_ok():
                return
            if self.path != "/need":
                return self._send(404, {"error": "없는 경로"})
            with st.lock:
                files = [{k: f[k] for k in ("name", "sha256", "size", "have")} for f in st.files.values()]
                return self._send(200, {"alias": st.alias, "digest": st.digest, "remaining": st.remaining(), "files": files})

        def do_PUT(self):
            if not self._peer_ok():
                return
            if not self.path.startswith("/files/"):
                return self._send(404, {"error": "없는 경로"})
            name = self.path[len("/files/"):]
            f = st.files.get(name)
            if f is None or not SV.FILE_NAME_RE.match(name):
                return self._send(404, {"error": "목록에 없는 이름"})
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._send(411, {"error": "Content-Length 가 필요하다"})
            if n != f["size"]:
                return self._send(413 if n > f["size"] else 400, {"error": f"크기는 목록과 같아야 한다({f['size']})"})
            with st.lock:
                if f["have"]:
                    return self._send(409, {"error": "이미 받았다"})
                if name in st.busy:
                    return self._send(409, {"error": "받는 중"})
                st.busy.add(name)
                st.state, st.current = "receiving", name
                if st.first_byte is None:
                    st.first_byte = time.time()
            part = st.dir / f".recv-{name}.part"
            h = hashlib.sha256()
            got = 0
            try:
                with open(part, "wb") as out:
                    while got < n:
                        b = self.rfile.read(min(CHUNK, n - got))
                        if not b:
                            break
                        out.write(b)
                        h.update(b)
                        got += len(b)
                        with st.lock:
                            st.write_status(extra=got)
                if got < n:
                    part.unlink(missing_ok=True)
                    with st.lock:
                        st.busy.discard(name)
                    return self._send(400, {"error": f"덜 받았다({got}/{n}) — 다시 보낸다"})
                if h.hexdigest() != f["sha256"]:
                    part.unlink(missing_ok=True)
                    with st.lock:
                        st.busy.discard(name)
                        st.state, st.exit_code = "failed", 3
                        st.write_status(force=True)
                    self._send(422, {"error": "sha256 가 목록과 다르다 — 창구를 닫는다"})
                    stop.set()
                    return
                os.replace(part, st.dir / name)
                with st.lock:
                    f["have"] = True
                    st.busy.discard(name)
                    st.bytes_done += n
                    left = st.remaining()
                    if left == 0:
                        st.state, st.exit_code, st.current = "done", 0, ""
                    st.write_status(force=True)
                self._send(200, {"ok": name, "remaining": left})
                if left == 0:
                    stop.set()
            except OSError as e:
                part.unlink(missing_ok=True)
                with st.lock:
                    st.busy.discard(name)
                return self._send(500, {"error": str(e)[:200]})

        def do_POST(self):
            if self._peer_ok():
                self._send(405, {"error": "없는 방법"})

        do_DELETE = do_POST

    return H


def run(alias: str, files: list[dict], d: Path, server_ip: str, port: int, timeout_s: float,
        status: Path | None, listen: str = "127.0.0.1", ready: threading.Event | None = None) -> int:
    st = State(alias, files, d, status)
    if st.remaining() == 0:
        st.state, st.exit_code = "done", 0
        st.write_status(force=True)
        return 0
    stop = threading.Event()
    srv = ThreadingHTTPServer((listen, port), make_handler(st, server_ip, stop))
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    t.start()
    if ready is not None:
        ready.set()
    deadline = time.time() + timeout_s
    while not stop.is_set() and time.time() < deadline:
        stop.wait(1.0)
        with st.lock:
            st.write_status()
    srv.shutdown()
    srv.server_close()
    with st.lock:
        if st.exit_code is None:
            st.state, st.exit_code = "timeout", 2
        st.write_status(force=True)
        return st.exit_code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alias", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--server-ip", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--timeout-min", type=float, default=60)
    ap.add_argument("--status")
    ap.add_argument("--listen", default="127.0.0.1")
    a = ap.parse_args(argv)
    try:
        t = SV.load()
        m = SV.model(t, a.alias)
        if m.get("source", "hf") != "home":
            print(f"modelrecv: {m['alias']} 은 home 별칭이 아니다", file=sys.stderr)
            return 1
        return run(m["alias"], m["files"], Path(a.dir), a.server_ip, a.port, a.timeout_min * 60,
                   Path(a.status) if a.status else None, a.listen)
    except (SV.TableError, OSError) as e:
        print(f"modelrecv: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
