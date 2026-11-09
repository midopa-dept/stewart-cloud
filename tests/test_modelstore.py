"""보관 서버 보관소(deploy/modelstore/modelpush.py): push(→ 파드 modelrecv), pull(← 변환 quantsrv), promote 검사, list."""

import hashlib
import importlib.util
import json
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from tests import FIX_ROOT

REPO = FIX_ROOT


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SV = _load("serving", "deploy/serving/serving.py")
MR = _load("modelrecv", "deploy/serving/modelrecv.py")
QS = _load("quantsrv", "deploy/quant/quantsrv.py")
MP = _load("modelpush", "deploy/modelstore/modelpush.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def safetensors_bytes(tensors: dict[str, list[float]]) -> bytes:
    hdr, pos, data = {}, 0, b""
    for k, v in tensors.items():
        b = struct.pack(f"<{len(v)}f", *v)
        hdr[k] = {"dtype": "F32", "shape": [len(v)], "data_offsets": [pos, pos + len(b)]}
        pos += len(b)
        data += b
    hb = json.dumps(hdr).encode()
    hb += b" " * ((8 - (8 + len(hb)) % 8) % 8)
    return struct.pack("<Q", len(hb)) + hb + data


def model_files() -> dict[str, bytes]:
    st = safetensors_bytes({"a.weight": [1.0, 2.0], "b.weight": [3.0]})
    return {"config.json": b'{"architectures": ["Qwen3_5ForConditionalGeneration"]}',
            "model.safetensors.index.json": json.dumps({"weight_map": {"a.weight": "model-1.safetensors",
                                                                       "b.weight": "model-1.safetensors"}}).encode(),
            "model-1.safetensors": st, "tokenizer.json": b"{}", "chat_template.jinja": b"{{ x }}",
            "mtp_draft_vocab_ids.json": b"[0, 1, 2]"}


def files_list(contents):
    return [{"name": n, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)} for n, b in sorted(contents.items())]


GOOD_REPORT = {"kl_pad": {"mean_kl": 1e-7, "top1_agree": 1.0}, "kl_quant_bf16heads": {"mean_kl": 0.02, "top1_agree": 0.95},
               "verify": {"ok": True}, "quality": {"ok": True, "result": {"gsm8k": 0.95}}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "srv-models"
        (self.root / "catalog").mkdir(parents=True)
        (self.root / "incoming").mkdir()
        self.cfg = MP.load_config(None) | {"root": str(self.root), "http_timeout_s": 5}


class TDigest(unittest.TestCase):
    def test_same_as_serving(self):
        fl = files_list(model_files())
        self.assertEqual(MP.home_digest(fl), SV.home_digest(fl))     # 파드 M_REV = home-<digest> 와 같은 계산


class TPush(Base):
    def put_catalog(self, alias, contents):
        d = self.root / "catalog" / alias
        d.mkdir()
        for n, b in contents.items():
            (d / n).write_bytes(b)
        (d / "manifest.json").write_text(json.dumps({"alias": alias, "digest": MP.home_digest(files_list(contents)),
                                                      "files": files_list(contents)}))

    def run_pod(self, alias, contents, files=None):
        port = free_port()
        ready = threading.Event()
        res = {}
        th = threading.Thread(target=lambda: res.setdefault("rc", MR.run(alias, files or files_list(contents), self.tmp / "pod",
                                                                          "100.64.0.9", port, 10, None, ready=ready)), daemon=True)
        th.start()
        ready.wait(5)
        return port, th, res

    def test_push_into_pod(self):
        c = model_files()
        self.put_catalog("demo2-int8h", c)
        port, th, res = self.run_pod("demo2-int8h", c)
        self.cfg["hosts"] = [f"127.0.0.1:{port}"]
        msgs = MP.push_once(self.cfg)
        th.join(5)
        self.assertEqual(res["rc"], 0, msgs)
        for n, b in c.items():
            self.assertEqual((self.tmp / "pod" / n).read_bytes(), b)

    def test_refuse_when_lists_differ(self):
        c = model_files()
        self.put_catalog("demo2-int8h", c)
        other = dict(c, **{"config.json": b'{"x": 2}'})                   # 파드가 다른 목록을 말한다
        port, th, res = self.run_pod("demo2-int8h", other)
        self.cfg["hosts"] = [f"127.0.0.1:{port}"]
        msgs = MP.push_once(self.cfg)
        self.assertIn("목록이 catalog manifest 와 다르다", msgs[0])
        self.assertFalse((self.tmp / "pod" / "config.json").exists())

    def test_unknown_alias_and_dead_host(self):
        port, th, res = self.run_pod("not-in-catalog", model_files())
        self.cfg["hosts"] = [f"127.0.0.1:{port}", f"127.0.0.1:{free_port()}"]
        msgs = MP.push_once(self.cfg)
        self.assertIn("catalog 에 not-in-catalog 가 없다", msgs[0])
        self.assertIn("/need 없음(파드는 있음, ConnectionRefusedError)", msgs[1])   # 이름은 풀림 = 파드는 있다

    def test_no_pod_name_vs_closed_port(self):
        import socket
        from unittest import mock
        self.cfg["hosts"] = ["stewart-serve.tail-example.ts.net", "bad host"]
        with mock.patch.object(MP.http.client.HTTPConnection, "request", side_effect=socket.gaierror(-2, "Name or service not known")):
            msgs = MP.push_once(self.cfg)
        self.assertEqual(msgs[0], "stewart-serve.tail-example.ts.net: 파드 없음(이름 못 풂)")
        self.assertIn("/need 오류(호스트 꼴", msgs[1])


class TPullPromote(Base):
    def make_quant(self, contents=None, report=None, extra_file=None):
        q = self.tmp / "q"
        (q / "done").mkdir(parents=True)
        (q / "out-int8heads").mkdir()
        c = contents or model_files()
        for n, b in c.items():
            (q / "out-int8heads" / n).write_bytes(b)
        files = files_list(c)
        if extra_file:
            (q / "out-int8heads" / extra_file[0]).write_bytes(extra_file[1])
            files = files + files_list({extra_file[0]: extra_file[1]})
        (q / "export.json").write_text(json.dumps({"version": 1, "sets": [
            {"alias": "qwen38-27b-demo2-w4a16-int8h", "dir": "out-int8heads", "files": files}], "report": report or GOOD_REPORT}))
        sp, ep = free_port(), free_port()
        srvs = QS.serve(QS.Ctx(q, "100.64.0.9", None), "127.0.0.1", sp, ep)
        self.addCleanup(lambda: [s.shutdown() for s in srvs])
        return q, ep

    def test_pull_and_promote(self):
        q, ep = self.make_quant()
        dest = MP.pull(self.cfg, "127.0.0.1", port=ep)
        self.assertEqual(dest.parent, self.root / "incoming")
        self.assertTrue((q / "done" / "exported").exists())                # POST /export/done
        man = MP.promote(self.cfg, dest, "qwen38-27b-demo2-w4a16-int8h")
        cat = self.root / "catalog" / "qwen38-27b-demo2-w4a16-int8h"
        self.assertEqual(json.loads((cat / "manifest.json").read_text())["digest"], man["digest"])
        self.assertEqual((cat / "model-1.safetensors").read_bytes(), model_files()["model-1.safetensors"])
        snip = MP.toml_snippet(man)
        self.assertIn('source = "home"', snip)
        self.assertEqual(len(MP.list_catalog(self.cfg)), 1)
        with self.assertRaises(MP.StoreError):
            MP.promote(self.cfg, dest, "qwen38-27b-demo2-w4a16-int8h")      # 같은 별칭 두 번은 안 된다
        # 파드 쪽 표에 넣으면 그대로 검사를 통과하는 꼴인지(serving.validate)
        import tomllib
        t = SV.read_raw()
        t["home"]["server_ip"] = "100.64.0.9"
        row = tomllib.loads(snip)["models"]["qwen38-27b-demo2-w4a16-int8h"]
        row |= {k: t["models"]["qwen38-27b-demo-w4a16"][k] for k in ("served_name", "adapters", "patchset", "tp", "hyperqwen", "serve",
                                                                       "chat_template_fix", "arch", "quant")}
        t["models"]["qwen38-27b-demo2-w4a16-int8h"] = row
        self.assertEqual(SV.validate(t, REPO / "deploy/serving"), [])

    def test_pull_rejects_bad_sha(self):
        q, ep = self.make_quant()
        idx = json.loads((q / "export.json").read_text())
        idx["sets"][0]["files"][0]["sha256"] = "0" * 64
        (q / "export.json").write_text(json.dumps(idx))
        with self.assertRaises(MP.StoreError):
            MP.pull(self.cfg, "127.0.0.1", port=ep)
        self.assertFalse((q / "done" / "exported").exists())               # 실패면 done 을 안 보낸다

    def test_promote_rejects(self):
        cases = [("pickle", None, ("mtp_draft_vocab_ids.pt", b"\x80\x04pickle")),
                 ("code", None, ("evil.py", b"import os")),
                 ("kl", dict(GOOD_REPORT, kl_pad={"mean_kl": 0.01, "top1_agree": 0.9}), None),
                 ("verify", dict(GOOD_REPORT, verify={"ok": False}), None)]
        for i, (why, rep, extra) in enumerate(cases):
            self.tmp_case = Path(tempfile.mkdtemp())
            q, ep = self.make_quant(report=rep, extra_file=extra)
            dest = MP.pull(self.cfg, "127.0.0.1", into=self.root / "incoming" / f"case{i}", send_done=False, port=ep)
            with self.assertRaises(MP.StoreError, msg=why):
                MP.promote(self.cfg, dest, "qwen38-27b-demo2-w4a16-int8h")
            import shutil
            shutil.rmtree(q)
        self.assertEqual(MP.list_catalog(self.cfg), [])

    def test_promote_bad_safetensors_and_index(self):
        bad = dict(model_files(), **{"model-1.safetensors": b"\x10\x00\x00\x00\x00\x00\x00\x00{not json}xxxxxxxx"})
        q, ep = self.make_quant(contents=bad)
        dest = MP.pull(self.cfg, "127.0.0.1", send_done=False, port=ep)
        with self.assertRaises(MP.StoreError):
            MP.promote(self.cfg, dest, "qwen38-27b-demo2-w4a16-int8h")
        import shutil
        shutil.rmtree(q)
        miss = dict(model_files())
        miss["model.safetensors.index.json"] = json.dumps({"weight_map": {"a.weight": "model-1.safetensors",
                                                                          "zz.weight": "model-1.safetensors"}}).encode()
        q, ep = self.make_quant(contents=miss)
        dest = MP.pull(self.cfg, "127.0.0.1", into=self.root / "incoming" / "miss", send_done=False, port=ep)
        with self.assertRaisesRegex(MP.StoreError, "zz.weight"):
            MP.promote(self.cfg, dest, "qwen38-27b-demo2-w4a16-int8h")

    def test_promote_only_from_incoming(self):
        with self.assertRaises(MP.StoreError):
            MP.promote(self.cfg, self.tmp, "qwen38-27b-demo2-w4a16-int8h")
        with self.assertRaises(MP.StoreError):
            MP.promote(self.cfg, self.root / "incoming" / "x", "../evil")


class TReport(unittest.TestCase):
    def test_check_report_pad_to(self):
        th = MP.DEFAULTS["thresholds"] if hasattr(MP, "DEFAULTS") else None
        th = th or {"pad_mean_kl_max": 1e-4, "pad_top1_min": 0.999, "quant_mean_kl_max": 0.1, "quant_top1_min": 0.85,
                    "require_verify": True, "require_quality": True}
        ok = {"verify": {"ok": True}, "quality": {"ok": True}, "kl_quant_bf16heads": {"mean_kl": 0.01, "top1_agree": 0.95}}
        self.assertEqual(MP.check_report(dict(ok, pad_to=0, kl_pad=None), th), [])            # 0 채우기 없는 대상
        self.assertTrue(MP.check_report(dict(ok, kl_pad=None), th))                           # pad_to 없는 옛 보고 = 0 채움 문턱
        self.assertTrue(MP.check_report(dict(ok, pad_to=17536, kl_pad={"mean_kl": 1e-3, "top1_agree": 1.0}), th))
        self.assertTrue(MP.check_report(dict(ok, pad_to=0, verify={"ok": False}), th))        # 그 밖 문턱은 그대로


class TFiles(unittest.TestCase):
    def test_shell_and_unit(self):
        d = REPO / "deploy/modelstore"
        for f in ("modelstore", "install.sh"):
            r = subprocess.run(["bash", "-n", str(d / f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        unit = (d / "modelpush.service").read_text()
        for must in ("User=modelpush", "ProtectSystem=strict", "ProtectHome=yes", "NoNewPrivileges=yes", "PrivateTmp=yes",
                     "ReadOnlyPaths=/srv/models/catalog", "ReadWritePaths=/srv/models/incoming", "IPAddressDeny=any",
                     "IPAddressAllow=100.64.0.0/10", "RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX"):
            self.assertIn(must, unit)
        head = unit.split("[Service]")[0]
        self.assertIn("RequiresMountsFor=/srv/models", head)                  # 마운트가 빠지면 뜨지 않는다([Unit] 절)
        inst = (d / "install.sh").read_text()
        self.assertIn("mountpoint -q /srv/models", inst)
        self.assertLess(inst.index("mountpoint -q /srv/models"), inst.index("install -d"))   # 빈 루트 디스크에 만들기 전에 멈춘다
        self.assertIn("systemctl restart modelpush.service", inst)                           # 다시 돌리면 새 판이 먹는다
        w = (d / "modelstore").read_text()
        self.assertIn("-p RequiresMountsFor=$ROOT", w.split("base=(")[1].split(")")[0])      # 일회 유닛 모두
        promote = w.split("promote)")[1].split(";;")[0]
        self.assertIn("ReadWritePaths=$ROOT/catalog", promote)                # catalog 쓰기는 promote 만
        self.assertIn("IPAddressDeny=any", promote)
        self.assertNotIn("IPAddressAllow", promote)
        pull = w.split("pull)")[1].split(";;")[0]
        self.assertIn("ReadOnlyPaths=$ROOT/catalog", pull)
        self.assertEqual(json.loads((d / "config.example.json").read_text())["port"], 8090)


if __name__ == "__main__":
    unittest.main()
