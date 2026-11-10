"""pod POST /logout(v6): 한 번만, 본문 무시, GET·다른 경로는 404, 응답 뒤 logout, 키 파일 지움."""

import importlib.util
import os
from unittest import mock
import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tests import FIX_ROOT, LOCAL


def load():
    spec = importlib.util.spec_from_file_location("idle_stop", FIX_ROOT / "deploy/vast/idle_stop.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Logout(unittest.TestCase):
    def setUp(self):
        self.m = load()
        self.d = tempfile.mkdtemp()
        self.key = Path(self.d) / "ts-authkey"
        self.key.write_text("tskey-fake")
        self.m.TS_KEY_FILE = str(self.key)
        self.calls = []
        self.lo = self.m.LogoutOnce(self.d, delay_s=0.05, logout=lambda: self.calls.append(time.time()))
        w = self.m.Watch(1800, 2700, 3600, 0)
        self.srv = self.m.serve_status(0, lambda: self.m.status(self.d, w, 1, None), on_logout=self.lo)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def post(self, path, body=b"ignored"):
        req = urllib.request.Request(self.base + path, data=body, method="POST")
        with LOCAL.open(req, timeout=5) as r:
            return json.loads(r.read())

    def test_once(self):
        self.assertEqual(self.post("/logout", b'{"anything": 1}'), {"result": "logging_out"})
        self.assertFalse(self.key.exists())                       # 다시 붙지 않게 키 파일을 지운다
        self.assertEqual(self.post("/logout/"), {"result": "already_logged_out"})
        time.sleep(0.3)
        self.assertEqual(len(self.calls), 1)                     # logout 은 한 번만, 응답 뒤에
        log = Path(self.d, "pod-log.tsv").read_text()
        self.assertEqual(log.count("logout 요청"), 2)
        self.assertIn("from 127.0.0.1", log)

    def test_get_and_other_paths(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            LOCAL.open(self.base + "/logout", timeout=5)
        self.assertEqual(cm.exception.code, 404)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/status")
        self.assertEqual(cm.exception.code, 404)
        self.assertEqual(self.calls, [])
        with mock.patch.dict(os.environ, {"CONTAINER_ID": "5555"}):
            with LOCAL.open(self.base + "/status", timeout=5) as r:     # 읽기 전용 상태는 그대로 + instance_id
                st = json.loads(r.read())
        self.assertIn("uptime_s", st)
        self.assertEqual(st["instance_id"], "5555")

    def test_no_logout_handler(self):
        w = self.m.Watch(1800, 2700, 3600, 0)
        srv = self.m.serve_status(0, lambda: {}, on_logout=None)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/logout", data=b"", method="POST")
            with self.assertRaises(urllib.error.HTTPError):
                LOCAL.open(req, timeout=5)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
