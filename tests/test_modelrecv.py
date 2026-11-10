"""home 별칭(models.toml source=home)과 파드의 받는 창구(modelrecv.py)."""

import copy
import hashlib
import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
import tomllib
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tests import FIX_ROOT, LOCAL

ROOT = FIX_ROOT / "deploy" / "serving"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SV = _load("serving")
MR = _load("modelrecv")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def files_of(contents: dict[str, bytes]) -> list[dict]:
    return [{"name": n, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)} for n, b in contents.items()]


class THomeTable(unittest.TestCase):
    def setUp(self):
        self.raw = SV.read_raw()

    def home_model(self, ip="100.101.102.103"):
        t = copy.deepcopy(self.raw)
        t["home"]["server_ip"] = ip
        t["models"]["qwen38-27b-demo2-w4a16-int8h"] = {
            "source": "home", "dir": "demo2-int8h", "served_name": "qwen3.8-27b", "arch": "Qwen3_5ForConditionalGeneration",
            "quant": "compressed-tensors", "adapters": ["hyperqwen-8cf642e", "vllm-patched"], "patchset": "vllm-0.30.0",
            "tp": 1, "hyperqwen": {"CTX": "long", "SPEC": "mtp"},
            "serve": dict(self.raw["models"]["qwen38-27b-demo-w4a16"]["serve"]),
            "files": files_of({"config.json": b"{}", "model.safetensors.index.json": b"{}", "model-1.safetensors": b"x" * 10})}
        return t

    def test_valid_and_select(self):
        t = self.home_model()
        self.assertEqual(SV.validate(t, ROOT), [])
        e = SV.select_env(t, "qwen38-27b-demo2-w4a16-int8h", "auto", "hyperqwen")
        self.assertEqual((e["M_SOURCE"], e["M_REPO"], e["M_PINS"]), ("home", "", ""))
        self.assertTrue(e["M_REV"].startswith("home-") and len(e["M_REV"]) == 5 + 64)
        pins = SV.pins(t, "qwen38-27b-demo2-w4a16-int8h").splitlines()
        self.assertEqual([p.split("  ")[1] for p in pins], ["config.json", "model-1.safetensors", "model.safetensors.index.json"])

    def test_refusals(self):
        self.assertTrue(any("server_ip" in e for e in SV.validate(self.home_model("CHANGE-ME"), ROOT)))
        self.assertTrue(any("server_ip" in e for e in SV.validate(self.home_model("192.168.0.2"), ROOT)))   # 테일넷 주소만
        for bad in ({"name": "../x", "sha256": "0" * 64, "size": 1}, {"name": ".hidden", "sha256": "0" * 64, "size": 1},
                    {"name": "a", "sha256": "zz", "size": 1}, {"name": "a", "sha256": "0" * 64, "size": 0},
                    {"name": "a", "sha256": "0" * 64}):
            t = self.home_model()
            t["models"]["qwen38-27b-demo2-w4a16-int8h"]["files"].append(bad)
            self.assertTrue(SV.validate(t, ROOT), bad)
        t = self.home_model()
        t["models"]["qwen38-27b-demo2-w4a16-int8h"]["repo"] = "a/b"            # home 에 repo 는 안 된다
        self.assertTrue(SV.validate(t, ROOT))
        self.assertEqual(SV.validate(self.raw, ROOT), [])                      # home 별칭이 없으면 CHANGE-ME 그대로 괜찮다


class TRecv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.contents = {"config.json": b'{"a": 1}', "model.safetensors.index.json": b"{}", "model-1.safetensors": b"w" * 300000}
        self.files = files_of(self.contents)
        self.port = free_port()
        self.result = {}

    def start(self, files=None, timeout_s=10.0):
        ready = threading.Event()

        def go():
            self.result["rc"] = MR.run("alias-x", files or self.files, self.tmp / "m", "100.64.0.9", self.port, timeout_s,
                                       self.tmp / "fetch.json", ready=ready)
        self.th = threading.Thread(target=go, daemon=True)
        self.th.start()
        ready.wait(5)

    def url(self, p):
        return f"http://127.0.0.1:{self.port}{p}"

    def put(self, name, data, length=None):
        req = urllib.request.Request(self.url(f"/files/{name}"), data=data, method="PUT",
                                     headers={"Content-Length": str(len(data) if length is None else length)})
        try:
            with LOCAL.open(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def need(self):
        with LOCAL.open(self.url("/need"), timeout=5) as r:
            return json.loads(r.read())

    def test_full_receive(self):
        self.start()
        n = self.need()
        self.assertEqual((n["alias"], n["remaining"]), ("alias-x", 3))
        self.assertEqual(self.put("evil.py", b"x")[0], 404)                         # 목록 밖 이름
        self.assertEqual(self.put("config.json", self.contents["config.json"] + b"!")[0], 413)   # 목록보다 크다
        st, _ = self.put("config.json", self.contents["config.json"])
        self.assertEqual(st, 200)
        self.assertEqual(self.put("config.json", self.contents["config.json"])[0], 409)   # 이미 받음
        self.assertEqual(self.need()["remaining"], 2)
        for name in ("model.safetensors.index.json", "model-1.safetensors"):
            self.assertEqual(self.put(name, self.contents[name])[0], 200)
        self.th.join(5)
        self.assertEqual(self.result["rc"], 0)
        for name, b in self.contents.items():
            self.assertEqual((self.tmp / "m" / name).read_bytes(), b)
        self.assertFalse(list((self.tmp / "m").glob(".recv-*")))                    # 임시 파일 없음
        st = json.loads((self.tmp / "fetch.json").read_text())
        self.assertEqual((st["state"], st["files_done"], st["bytes_done"], st["bytes_total"]),
                         ("done", 3, sum(map(len, self.contents.values())), sum(map(len, self.contents.values()))))
        with self.assertRaises(OSError):                                            # 다 받으면 창구를 닫는다
            LOCAL.open(self.url("/need"), timeout=2)

    def test_sha_mismatch_halts(self):
        self.start()
        bad = b"W" * 300000                                                         # 크기는 같고 내용이 다르다
        st, body = self.put("model-1.safetensors", bad)
        self.assertEqual(st, 422)
        self.th.join(5)
        self.assertEqual(self.result["rc"], 3)
        self.assertFalse((self.tmp / "m" / "model-1.safetensors").exists())

    def test_timeout(self):
        self.start(timeout_s=1.5)
        self.th.join(5)
        self.assertEqual(self.result["rc"], 2)
        self.assertEqual(json.loads((self.tmp / "fetch.json").read_text())["state"], "timeout")

    def test_resume_existing(self):
        d = self.tmp / "m"
        d.mkdir()
        (d / "config.json").write_bytes(self.contents["config.json"])
        (d / "model-1.safetensors").write_bytes(b"partial")                        # 틀린 것은 받은 것으로 치지 않는다
        self.start()
        n = self.need()
        self.assertEqual({f["name"]: f["have"] for f in n["files"]},
                         {"config.json": True, "model.safetensors.index.json": False, "model-1.safetensors": False})
        for name in ("model.safetensors.index.json", "model-1.safetensors"):
            self.put(name, self.contents[name])
        self.th.join(5)
        self.assertEqual(self.result["rc"], 0)

    def test_short_body_can_retry(self):
        self.start()
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.putrequest("PUT", "/files/model-1.safetensors")
        c.putheader("Content-Length", "300000")
        c.endheaders()
        c.send(b"w" * 1000)
        c.close()                                                                   # 끊김
        time.sleep(0.5)
        st, _ = self.put("model-1.safetensors", self.contents["model-1.safetensors"])
        self.assertEqual(st, 200)                                                   # 다시 보내면 된다

    def test_already_complete(self):
        d = self.tmp / "m"
        d.mkdir()
        for n, b in self.contents.items():
            (d / n).write_bytes(b)
        rc = MR.run("alias-x", self.files, d, "100.64.0.9", self.port, 5, None)
        self.assertEqual(rc, 0)                                                     # 창구를 열지 않고 끝


if __name__ == "__main__":
    unittest.main()
