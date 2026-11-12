"""stewart-cloud 단위 시험 — 네트워크·키체인 없이(가짜 opener·가짜 키). 설치 전 원본으로 돌린다.

  python3 -m unittest tests.test_stewart_cloud      (저장소 뿌리에서)
"""

import base64
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

# 시험용 지어낸 Tailscale 키 꼴(가림 규칙 시험). 비밀 스캐너가 진짜 키로 잡지 않게 조각으로 잇는다(11-12 GitHub secret scanning 오탐)
FAKE_TSKEY = "tskey" + "-auth-" + "kQ1-abcdef"

HERE = Path(__file__).resolve().parents[1] / "stewart_cloud"
spec = importlib.util.spec_from_file_location("stewart_cloud", HERE / "stewart_cloud.py")
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)

KEY = "k" * 64
HASH = "0123456789abcdef0123456789abcdef"
QHASH = "fedcba9876543210fedcba9876543210"
GHTOKEN = "github_pat_" + "T" * 82
GH_OWNER, GID, CFGID = "someone", "a" * 32, "c" * 32
IMG = "ghcr.io/syv-ai/hyperqwen@sha256:" + "6" * 64
QIMG = "ghcr.io/syv-ai/hyperqwen@sha256:" + "8" * 64
CF_URL = f"https://gist.githubusercontent.com/{GH_OWNER}/{CFGID}/raw/stewart-cf.env"


def sha(t):
    return SC.sha256s(t)


def tpl(h, ps, sb, args, img, cf=True, disk=60, pu_file="stewart-pod-v11.b64"):
    """콘솔에서 만든 꼴의 템플릿(API 응답 행)."""
    env = (("-e CTX=fast -e SPEC=mtp " if cf else "") + f"-e PU=https://gist.githubusercontent.com/{GH_OWNER}/{GID}/raw/{'e' * 40}/{pu_file}"
           f" -e PS={ps}" + (f" -e CF={CF_URL}" if cf else "") + f" -e SB={sb}")
    return {"id": 1, "hash_id": h, "name": "stewart x", "image": img, "tag": None, "runtype": "args",
            "recommended_disk_space": disk, "env": env, "args_str": args, "private": True, "onstart": None}


def seed_state(d, templates):
    """설치 때 넣는 꼴의 상태 파일(T27)."""
    d.mkdir(mode=0o700, exist_ok=True)
    s = {"v": 1, "current": {}, "templates": {}}
    for kind, bundle, t in templates:
        f = SC.template_facts(t)
        s["templates"][t["hash_id"]] = {"kind": kind, "bundle": bundle, "commit": None, "ps": f["ps"], "sb_sha": f["sb_sha"],
                                        "args_sha": f["args_sha"], "image": f["image"], "source": "console", "ts": "2026-11-09T19:00:00+09:00",
                                        "base": None}
        s["current"][kind] = t["hash_id"]
    (d / "templates.json").write_text(json.dumps(s))
    return s


def offer(i, dph, machine, **kw):
    o = {"id": i, "machine_id": machine, "dph_total": dph, "reliability": 0.995, "cuda_max_good": 13.2, "geolocation": "CZ",
         "inet_down": 900.0, "gpu_name": "RTX 3090", "num_gpus": 1, "verification": "verified", "hosting_type": 1}
    o.update(kw)
    return o


class FakeRaw:
    def __init__(self, raw):
        self.body = raw

    def read(self, n=-1):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeResp:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self, n=-1):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeVast:
    """(method, path) → 응답. 요청을 모두 적어 둔다."""

    def __init__(self):
        self.offers = [offer(101, 0.40, 9001), offer(102, 0.35, 9002), offer(103, 0.36, 10001)]
        self.instances = [{"id": 777, "label": "someone-else", "machine_id": 5, "actual_status": "running"}]
        self.calls = []
        self.logout_ok = True
        self.pod_status = {"instance_id": "5555", "pod_log": []}       # 이름을 쥔 pod 의 /status(stewart-gpu)
        self.pod_status_eval = None                                    # stewart-gpu-eval 의 /status(None = 없음)
        self.pod_status_quant = None                                   # stewart-quant 의 /status
        self.pod_status_eval2 = None                                   # stewart-gpu-eval2 의 /status(T26)
        self.next_id = 5555
        self.log_text, self.log_gets, self.log_404s = "", 0, 2
        self.fail = {}
        self.list_queries = []
        self.bundle_bodies = []
        sb0 = base64.b64encode(b"#!/bin/bash\necho boot\n").decode()
        self.templates = {HASH: tpl(HASH, "1" * 64, sb0, "bash -c 'boot v11'", IMG),
                          QHASH: tpl(QHASH, "2" * 64, sb0, "bash -c 'boot quant'", QIMG, cf=False, disk=200,
                                     pu_file="stewart-quant-v3.b64")}
        self.tpl_posts = []
        self.tpl_mutate = None                                         # 만든 템플릿을 다시 읽을 때 바꿔 보이기(시험)
        self.gist = {GID: {"owner": GH_OWNER, "files": {}, "rev": 0}}
        self.gh_calls = []
        self.raw_override = {}                                         # raw 주소 → 다른 내용(시험)
        self.cf_text = "CTX=long\nSPEC=mtp\n"
        self.retpl_new_id = None
        self.queue_ids = set()                                         # 켜기를 대기열에 두는 인스턴스(T32)
        self.copies = []
        self.user = {"credit": 41.2345, "balance": -0.5, "api_key": "k" * 64, "email": "who@example.invalid",
                     "fullname": "Some Body", "billaddress_line1": "1 Road", "id": 99}

    def open(self, req, timeout=None):
        url, method = req.full_url, req.get_method()
        body = json.loads(req.data) if req.data else None
        self.calls.append({"method": method, "url": url, "body": body, "headers": dict(req.header_items())})
        if url.startswith("https://logs.example-bucket.com/"):        # result_url(S3 꼴): 처음 몇 번은 아직 없음
            self.log_gets += 1
            if self.log_gets <= self.log_404s:
                raise SC.urllib.error.HTTPError(url, 404, "x", {}, io.BytesIO(b""))
            self.calls[-1]["auth"] = req.get_header("Authorization")
            return FakeRaw(self.log_text.encode())
        if url.startswith("https://api.github.com/gists/"):
            return self.github(req, method, url, body)
        if url.startswith("https://gist.githubusercontent.com/"):
            self.calls[-1]["auth"] = req.get_header("Authorization")
            if url in self.raw_override:
                return FakeRaw(self.raw_override[url].encode())
            if url == CF_URL:
                return FakeRaw(self.cf_text.encode())
            m = re.fullmatch(r"https://gist\.githubusercontent\.com/([^/]+)/([0-9a-f]+)/raw/([0-9a-f]{40})/(.+)", url)
            g = self.gist.get(m.group(2)) if m else None
            if not g or m.group(4) not in g["files"]:
                raise SC.urllib.error.HTTPError(url, 404, "x", {}, io.BytesIO(b""))
            return FakeRaw(g["files"][m.group(4)].encode())
        if (method, url) in self.fail:
            code = self.fail[(method, url)]
            raise SC.urllib.error.HTTPError(url, code, "x", {}, io.BytesIO(json.dumps({"msg": "nope " + KEY}).encode()))
        if url.endswith(":8001/status"):
            st = (self.pod_status_eval if "stewart-gpu-eval." in url else self.pod_status_eval2 if "stewart-gpu-eval2." in url else
                  self.pod_status_quant if "stewart-quant." in url else self.pod_status)
            if st is None:
                raise SC.urllib.error.URLError("no pod")
            return FakeResp(st)
        if url.endswith(":8001/logout"):
            if not self.logout_ok:
                raise SC.urllib.error.URLError("down")
            return FakeResp({"result": "logged_out"})
        path, _, qs = url[len(SC.API):].partition("?")
        q = {k: v[0] for k, v in SC.urllib.parse.parse_qs(qs).items()}
        if path == "/v0/instances/" and method == "GET":     # v0 목록은 지원 종료(11/8 v12.10 실측)
            raise SC.urllib.error.HTTPError(url, 410, "x", {}, io.BytesIO(b'{"msg": "/api/v0/instances/ is deprecated."}'))
        if path == "/v0/users/current" and method == "GET":
            self.calls[-1]["auth"] = req.get_header("Authorization")
            return FakeResp(self.user)
        if path == "/v0/bundles/" and method == "POST":
            self.bundle_bodies.append(body)
            if "id" in body:                                  # 11/8 22:4x 실측: id 조건을 넣으면 offers 에 있던 것도 안 나온다
                return FakeResp({"offers": []})
            # 서버는 값 상한·싼 순·limit 만 적용한다고 본다(그 밖 조건은 래퍼의 offer_ok 가 다시 본다)
            lte = float(body["dph_total"]["lte"])
            if "datacenter" in body:                          # T36: 서버 datacenter 검색 = hosting_type 1(datacenter 칸이 있으면 그 값)
                pool = [o for o in self.offers if (o["datacenter"] if "datacenter" in o else o.get("hosting_type") == 1)]
            else:
                pool = self.offers
            ng = body["num_gpus"]
            lo, hi = ng.get("eq", ng.get("gte")), ng.get("eq", ng.get("lte"))
            os_ = sorted((o for o in pool if o["dph_total"] <= lte and lo <= o["num_gpus"] <= hi), key=lambda o: o["dph_total"])
            return FakeResp({"offers": os_[: int(body["limit"])]})
        if path == "/v1/instances/" and method == "GET":    # 문서의 v1 꼴: 25개씩, next_token
            self.list_queries.append(q)
            lim = int(q.get("limit", 25))
            start = int(q["after_token"].removeprefix("tok")) if "after_token" in q else 0
            page = sorted(self.instances, key=lambda i: i["id"])[start:start + lim]
            nxt = f"tok{start + lim}" if start + lim < len(self.instances) else None
            return FakeResp({"success": True, "instances_found": len(page), "total_instances": len(self.instances),
                             "label_counts": {}, "next_token": nxt, "instances": page})
        if path == "/v0/template/" and method == "GET":
            h = json.loads(q["select_filters"])["hash_id"]["eq"]
            t = self.templates.get(h)
            if t is not None and self.tpl_mutate and h in [x["hash_id"] for x in self.tpl_posts]:
                t = self.tpl_mutate(dict(t))
            return FakeResp({"success": True, "templates": [t] if t else []})
        if path == "/v0/template/" and method == "POST":
            h = SC.sha256s(json.dumps(body, sort_keys=True))[:32]
            self.templates[h] = {**body, "id": 2, "hash_id": h}
            self.tpl_posts.append(self.templates[h])
            return FakeResp({"success": True, "msg": "Template Created", "template": {"id": 2, "hash_id": h, "name": body["name"]}})
        m = re.fullmatch(r"/v0/asks/(\d+)/", path)
        if m and method == "PUT":
            o = [x for x in self.offers if x["id"] == int(m.group(1))][0]
            iid, self.next_id = self.next_id, self.next_id + 1
            self.instances.append({"id": iid, "label": body["label"], "machine_id": o["machine_id"], "actual_status": "loading",
                                   "num_gpus": o["num_gpus"], "dph_total": o["dph_total"], "start_date": 1_700_000_000.0})
            return FakeResp({"success": True, "new_contract": iid})
        m = re.fullmatch(r"/v0/instances/request_logs/(\d+)/", path)
        if m and method == "PUT":
            return FakeResp({"success": True, "result_url": f"https://logs.example-bucket.com/{m.group(1)}.log?X-Amz-Signature=abc"})
        m = re.fullmatch(r"/v0/instances/update_template/(\d+)/", path)
        if m and method == "PUT":                             # T30-1R: 같은 id 로 다시 만든다(새 id 를 주게 하려면 retpl_new_id)
            for i in self.instances:
                if i["id"] == int(m.group(1)):
                    i["template_hash_id"] = body["template_hash_id"]
                    if self.retpl_new_id:
                        i["id"] = self.retpl_new_id
            return FakeResp({"success": True, "updated_instance": {"id": int(m.group(1)), "template_hash_id": body["template_hash_id"],
                                                                   "api_key": "k" * 64}})
        m = re.fullmatch(r"/v0/instances/(\d+)/", path)
        if m and method == "PUT":
            if body == {"state": "running"} and int(m.group(1)) in self.queue_ids:   # 실측: GPU 가 남에게 가 있으면
                return FakeResp({"success": False, "msg": "Required resources are currently unavailable, state change queued."})
            for i in self.instances:
                if i["id"] == int(m.group(1)):
                    st = "running" if body["state"] == "running" else "stopped"
                    i.update(actual_status=st if st == "running" else "exited", intended_status=st, cur_state=st)
            return FakeResp({"success": True})
        if path == "/v0/commands/copy_direct/" and method == "PUT":
            self.copies.append(body)
            return FakeResp({"success": True, "msg": ""})
        if m and method == "DELETE":
            self.instances = [i for i in self.instances if i["id"] != int(m.group(1))]
            return FakeResp({"success": True})
        raise AssertionError(f"모르는 요청 {method} {url}")

    def github(self, req, method, url, body):
        self.gh_calls.append({"method": method, "url": url, "auth": req.get_header("Authorization"), "body": body})
        gid = url.rsplit("/", 1)[1]
        g = self.gist.get(gid)
        if g is None:
            raise SC.urllib.error.HTTPError(url, 404, "x", {}, io.BytesIO(json.dumps({"message": "Not Found " + GHTOKEN}).encode()))
        if method == "PATCH":
            for n, f in body["files"].items():
                g["files"][n] = f["content"]
            g["rev"] += 1
        rev = SC.sha256s(f"rev{g['rev']}")[:40]
        files = {n: {"filename": n, "raw_url": f"https://gist.githubusercontent.com/{g['owner']}/{gid}/raw/{SC.sha256s(c)[:40]}/{n}",
                     "content": c, "truncated": False, "size": len(c)} for n, c in g["files"].items()}
        return FakeResp({"id": gid, "owner": {"login": g["owner"]}, "files": files, "history": [{"version": rev}]})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfgp = self.tmp / "config.json"
        cfg = json.loads((HERE / "config.example.json").read_text())
        cfg["template_hash_id"] = HASH
        self.cfgp.write_text(json.dumps(cfg))
        self.rec = self.tmp / "rec"
        self.fake = FakeVast()
        self.state = self.tmp / "state"
        seed_state(self.state, [("pod", "v11", self.fake.templates[HASH]), ("quant", "quant-v3", self.fake.templates[QHASH])])
        self.names = {"present": False}
        patches = [mock.patch.object(SC, "_OPENER", self.fake),
                   mock.patch.object(SC, "name_present", lambda n: self.names["present"]),
                   mock.patch.object(SC.time, "sleep", lambda s: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, *argv, key_fn=None, tkey_fn=None, **kw):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.dict(os.environ):   # main 이 비운 환경을 시험 뒤 되돌린다
            rc = SC.main(list(argv), key_fn=key_fn or (lambda: KEY), config_path=str(self.cfgp), record_dir=str(self.rec),
                         tkey_fn=tkey_fn or (lambda: None), gh_fn=lambda: GHTOKEN, state_dir=str(self.state), **kw)
        self.assertNotIn(KEY, out.getvalue() + err.getvalue())           # 키는 어디에도 나오지 않는다
        self.assertNotIn(GHTOKEN, out.getvalue() + err.getvalue())       # gist 토큰도(T27)
        return rc, out.getvalue(), err.getvalue()


class Args(Base):
    def test_good_and_bad_forms(self):
        cfg = SC.load_config(str(self.cfgp))
        good = {("offers",): ("offers", ("serve", None, None)), ("offers", "serve"): ("offers", ("serve", None, None)),
                ("offers", "eval"): ("offers", ("eval", None, None)), ("offers", "eval", "4090"): ("offers", ("eval", "RTX 4090", None)),
                ("offers", "eval", "8"): ("offers", ("eval", None, 8)), ("offers", "eval", "4"): ("offers", ("eval", None, 4)),
                ("offers", "eval", "4090", "8"): ("offers", ("eval", "RTX 4090", 8)),
                ("status",): ("status", ()), ("up", "123", "0.4"): ("up", ("serve", 123, 0.4, None)),
                ("up", "serve", "123", "0.4"): ("up", ("serve", 123, 0.4, None)), ("up", "eval", "123", "1.85"): ("up", ("eval", 123, 1.85, None)),
                ("up", "eval", "123", "6.0"): ("up", ("eval", 123, 6.0, None)), ("down", "5555"): ("down", (5555,)),
                ("up", "eval", "123", "1.85", "a-x4"): ("up", ("eval", 123, 1.85, "a-x4")),
                ("up", "eval", "123", "1.85", "cmp-a-b1"): ("up", ("eval", 123, 1.85, "cmp-a-b1")),
                ("logs", "5555"): ("logs", (5555, False)), ("logs", "5555", "daemon"): ("logs", (5555, True)),
                ("offers", "eval2"): ("offers", ("eval2", None, None)), ("offers", "eval2", "4"): ("offers", ("eval2", None, 4)),
                ("offers", "eval2", "4090", "4"): ("offers", ("eval2", "RTX 4090", 4)),
                ("up", "eval2", "123", "1.85"): ("up", ("eval2", 123, 1.85, None)),
                ("up", "eval2", "123", "1.85", "a-all"): ("up", ("eval2", 123, 1.85, "a-all"))}
        for g, want in good.items():
            self.assertEqual(SC.parse_args(list(g), cfg), want, g)
        bad = [[], ["offer"], ["offers", "x"], ["offers", "serve", "4090"], ["offers", "eval", "4080"], ["offers", "eval", "eval"],
               ["offers", "eval", "3"], ["offers", "eval", "9"], ["offers", "eval", "08"], ["offers", "eval", "8", "4090"],
               ["offers", "eval", "4090", "8", "x"], ["offers", "eval", "4090", "4090"], ["offers", "serve", "1"], ["offers", "quant", "1"],
               ["offers", "eval", "8", "8"], ["offers", "eval", " 8"], ["offers", "eval3"], ["up", "eval3", "123", "1.0"],
               ["up", "eval2", "123", "6.01"], ["up", "Eval2", "123", "1.0"], ["up", "eval2", "123", "1.0", "A"], ["offers", "eval2", "9"],
               ["status", "--all"], ["up"], ["up", "123"], ["up", "123", "0.4", "x"],
               ["up", "-1", "0.4"], ["up", "0123", "0.4"], ["up", "12a", "0.4"], ["up", "123", "1"], ["up", "123", "0.61"],
               ["up", "123", "0.0"], ["up", "123", ".4"], ["up", "123", "0.4000"], ["up", "--offer=1", "0.4"],
               ["up", "eval", "123", "6.01"], ["up", "serve", "123", "1.85"], ["up", "Eval", "123", "0.4"], ["up", "box", "123", "0.4"],
               ["up", "eval", "123", "10.0"], ["up", "eval", "123"], ["up", "eval", "serve", "123", "0.4"],
               ["down"], ["down", "abc"], ["down", "1", "2"], ["down", "1" * 13], ["destroy", "1"], ["up", "123", "0.4;rm"],
               ["up", "serve", "123", "0.4", "a-x4"], ["up", "quant", "123", "0.4", "a-x4"], ["up", "eval", "123", "1.85", "A-X4"],
               ["up", "eval", "123", "1.85", "../p"], ["up", "eval", "123", "1.85", "a;rm"], ["up", "eval", "123", "1.85", "x" * 49],
               ["up", "eval", "123", "1.85", "a-x4", "z"], ["logs"], ["logs", "x"], ["logs", "1", "all"], ["logs", "1", "daemon", "x"]]
        for b in bad:
            with self.assertRaises(SC.Stop, msg=b) as cm:
                SC.parse_args(b, cfg)
            self.assertEqual(cm.exception.code, 64, b)

    def test_shell_entry_rejects(self):
        sh = str(HERE / "stewart-cloud")
        for argv in ([], ["offers", "a", "b", "c", "d"], ["up", "1"], ["up", "a", "b", "c", "d", "e"], ["down"], ["rm", "-rf"],
                     ["status", "1"], ["logs"], ["logs", "1", "daemon", "x"], ["publish", "v12", "a" * 40],
                     ["publish", "v12", "a" * 40, "b" * 64, "x"], ["verify"], ["templates", "x"], ["adopt", "x"]):
            r = subprocess.run(["/bin/sh", sh] + argv, capture_output=True, text=True)
            self.assertEqual(r.returncode, 64, argv)

    def test_config_checks(self):
        cfg = json.loads(self.cfgp.read_text())
        SC.load_config(str(self.cfgp))
        role_bad = [("eval", {"max_dph": 8.5}), ("eval", {"max_dph": 0}), ("serve", {"num_gpus_max": 2}),
                    ("eval", {"num_gpus_min": 0}), ("eval", {"num_gpus_min": 5, "num_gpus_max": 4}), ("eval", {"num_gpus_max": 9}),
                    ("eval", {"num_gpus_min": 4.0}), ("eval", {"disk": 10}), ("eval", {"disk": "80"}),
                    ("eval", {"ts_name": "stewart-gpu.tail-example.ts.net"}), ("serve", {"ts_name": "evil.example.com"}),
                    ("eval", {"extra": 1})]
        top_bad = [{"label_prefix": "x"}, {"extra": 1}, {"max_dph": 0.6}, {"roles": {"serve": cfg["roles"]["serve"]}}]
        cases = [dict(cfg, **b) for b in top_bad]
        for r, b in role_bad:
            c = json.loads(json.dumps(cfg))
            c["roles"][r].update(b)
            cases.append(c)
        for c in cases:
            p = self.tmp / "c2.json"
            p.write_text(json.dumps(c))
            with self.assertRaises(SC.Stop, msg=json.dumps(c)[:200]):
                SC.load_config(str(p))


class Key(Base):
    def test_sandbox_and_missing(self):
        with mock.patch.object(SC.os, "access", lambda p, m: False):
            with self.assertRaises(SC.Stop) as cm:
                SC.read_key()
        self.assertEqual(cm.exception.code, 78)
        self.assertIn("샌드박스", str(cm.exception))
        calls = []

        def fake_run(args, **kw):
            calls.append((args, kw))
            return subprocess.CompletedProcess(args, 44, b"", b"not found")

        with mock.patch.object(SC.os, "access", lambda p, m: True), mock.patch.object(SC.subprocess, "run", fake_run):
            with self.assertRaises(SC.Stop) as cm:
                SC.read_key()
        self.assertEqual(cm.exception.code, 78)
        args, kw = calls[0]
        self.assertEqual(args[:2], ["/usr/bin/security", "find-generic-password"])
        self.assertEqual(kw["env"], {"PATH": "/usr/bin:/bin", "HOME": SC.HOME_DIR})   # 키를 넘기는 환경이 아니다
        self.assertEqual(kw["stdout"], subprocess.PIPE)

    def test_locked_keychain_hint(self):
        for code, err in ((36, b""), (51, b""), (1, b"User interaction is not allowed.")):
            with mock.patch.object(SC.os, "access", lambda p, m: True), \
                    mock.patch.object(SC.subprocess, "run", lambda a, **kw: subprocess.CompletedProcess(a, code, b"", err)):
                with self.assertRaises(SC.Stop) as cm:
                    SC.read_key()
            self.assertEqual(cm.exception.code, 78)
            self.assertIn("security unlock-keychain ~/Library/Keychains/login.keychain-db", str(cm.exception))
        with mock.patch.object(SC.os, "access", lambda p, m: True), \
                mock.patch.object(SC.subprocess, "run", lambda a, **kw: subprocess.CompletedProcess(a, 1, b"", b"other")):
            with self.assertRaises(SC.Stop) as cm:
                SC.read_key()
        self.assertNotIn("unlock-keychain", str(cm.exception))      # 다른 실패에는 붙이지 않는다

    def test_key_format(self):
        for raw, ok in ((KEY.encode() + b"\n", True), (b"short\n", False), (b"has space " + b"x" * 30, False)):
            with mock.patch.object(SC.os, "access", lambda p, m: True), \
                    mock.patch.object(SC.subprocess, "run", lambda a, **k: subprocess.CompletedProcess(a, 0, raw, b"")):
                if ok:
                    self.assertEqual(SC.read_key(), KEY)
                else:
                    with self.assertRaises(SC.Stop):
                        SC.read_key()

    def test_no_child_gets_key(self):
        """키를 읽은 뒤에는 자식 프로세스를 띄우지 않는다(security 하나뿐)."""
        spawned = []
        with mock.patch.object(SC.subprocess, "run", lambda *a, **k: spawned.append(a)), \
                mock.patch.object(SC.subprocess, "Popen", lambda *a, **k: spawned.append(a)):
            rc, out, err = self.run_main("up", "102", "0.36")
        self.assertEqual(rc, 0, err)
        self.assertEqual(spawned, [])
        src = (HERE / "stewart_cloud.py").read_text()
        self.assertEqual(src.count("subprocess.run("), 2)                   # security + 묶음 다시 만들기(T27, 키 읽기 전 — Publish 시험)
        self.assertNotIn("Popen", src)
        body = src.split("def clean_environ", 1)[0] + src.split("def clean_environ", 1)[1].split("\ndef ", 1)[1]
        self.assertNotIn("os.environ", body)                               # 환경은 clean_environ 에서 비우기만 한다
        self.assertNotIn("os.system", src)

    def test_environ_cleaned_first(self):
        seen, cafile = {}, None

        def key_fn():
            seen.update(os.environ)                                         # 키를 읽는 때(=ssl·urllib 전)의 환경
            import ssl
            nonlocal cafile
            cafile = ssl.get_default_verify_paths()
            return KEY

        bad = {"SSL_CERT_FILE": "/tmp/evil-ca.pem", "SSL_CERT_DIR": "/tmp/evil", "HTTPS_PROXY": "http://evil:8080",
               "https_proxy": "http://evil:8080", "PYTHONPATH": "/tmp/evil", "VAST_API_KEY": "x" * 40}
        with mock.patch.dict(os.environ, bad):
            rc, out, err = self.run_main("status", key_fn=key_fn)
        self.assertEqual(rc, 0, err)
        self.assertEqual(seen, {"PATH": "/usr/bin:/bin", "HOME": SC.HOME_DIR})
        self.assertNotIn("SSL_CERT_FILE", os.environ)                       # (run_main 안의 patch 가 되돌린 뒤 바깥 patch 도 되돌림)
        self.assertNotIn("/tmp/evil", str(cafile))                          # ssl 이 보는 CA 경로도 기본값


class Flows(Base):
    def test_offers_order_and_query(self):
        os.makedirs(self.rec)
        (self.rec / "history.jsonl").write_text(json.dumps({"event": "up", "machine_id": 10001}) + "\n")
        rc, out, err = self.run_main("offers")
        self.assertEqual(rc, 0, err)
        rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
        self.assertEqual([r["id"] for r in rows], [103, 102, 101])     # 0.35~0.37 안에서는 전에 쓴 기계(10001) 먼저
        self.assertTrue(rows[0]["used_before"])
        q = self.fake.calls[0]["body"]
        self.assertEqual(q["num_gpus"], {"eq": 1})
        self.assertEqual(q["cuda_max_good"], {"gte": 13.0})
        self.assertEqual(q["reliability"], {"gte": 0.95})
        self.assertEqual(q["dph_total"], {"lte": 0.6})
        self.assertEqual(q["datacenter"], {"eq": True})
        self.assertEqual(self.fake.calls[0]["headers"]["Authorization"], f"Bearer {KEY}")
        self.assertIn("stewart-cloud up 103 0.365", out)

    def test_offers_filters_bad_server_rows(self):
        self.fake.offers = [offer(1, 0.3, 1, cuda_max_good=12.2), offer(2, 0.3, 2, num_gpus=2), offer(3, 0.3, 3, reliability=0.9),
                            offer(4, 0.7, 4), offer(5, 0.3, 5, gpu_name="RTX 4090"), offer(6, 0.3, 6, hosting_type=0)]
        rc, out, err = self.run_main("offers")
        self.assertEqual(rc, 69)
        self.assertEqual(self.lines(out), [])                                 # 빌릴 수 있는 offer 줄({…})은 없다
        # T26: 0개면 조건을 풀어 한 번 더 찾고 조건별로 빠진 수와 가까운 것을 글 줄로만 보인다
        relaxed = self.fake.bundle_bodies[-1]
        self.assertNotIn("reliability", relaxed)
        self.assertNotIn("verified", relaxed)
        self.assertEqual(relaxed["dph_total"], {"lte": SC.ABS_MAX_DPH})
        self.assertIn("조건별로 빠진 수:", out)
        near = [l for l in out.splitlines() if "가까운 것(빌릴 수 없음)" in l]
        self.assertTrue(near and all(l.startswith("stewart-cloud: ") for l in near))

    def test_template_checked_only_for_up(self):
        """T27: 템플릿은 상태 파일이 정한다. 상태가 틀렸으면 up 만 78(키·API 안 건드림), offers·status·down 은 돈다."""
        good = (self.state / "templates.json").read_text()
        bads = ["", "{}", json.dumps({**json.loads(good), "current": {"pod": "f" * 32}}),
                json.dumps({**json.loads(good), "v": 2}), None]
        for bad in bads:
            if bad is None:
                (self.state / "templates.json").unlink()
            else:
                (self.state / "templates.json").write_text(bad)
            self.fake.calls.clear()
            keys = []
            rc, out, err = self.run_main("up", "102", "0.36", key_fn=lambda: keys.append(1) or KEY)
            self.assertEqual(rc, 78, (bad, err))
            self.assertIn("상태", err)
            self.assertEqual((keys, self.fake.calls), ([], []))       # 키도 API 도 건드리지 않는다
            for argv in (("offers",), ("status",), ("down", "5555")):
                rc, out, err = self.run_main(*argv)
                self.assertNotIn("상태", err, (bad, argv))
                self.assertNotEqual(rc, 70, (bad, argv, err))
            (self.state / "templates.json").write_text(good)
        with self.assertRaises(SC.Stop) as e:
            SC.do_up(KEY, SC.load_config(str(self.cfgp)), "serve", 102, 0.36, str(self.rec), state_dir=str(self.tmp / "none"))
        self.assertEqual(e.exception.code, 78)                          # do_up 을 바로 불러도 막힌다

    def test_down_works_with_empty_template(self):
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)      # 켜진 GPU 가 있는데
        (self.state / "templates.json").unlink()                         # 상태 파일이 사라졌다
        self.assertEqual(self.run_main("status")[0], 0)
        rc, out, err = self.run_main("down", "5555")
        self.assertEqual(rc, 0, err)                                      # 그래도 끌 수 있다
        self.assertEqual([i["id"] for i in self.fake.instances], [777])     # 우리 것만 지움(남의 777 그대로)

    def test_up_records_and_body(self):
        rc, out, err = self.run_main("up", "102", "0.36")
        self.assertEqual(rc, 0, err)
        put = [c for c in self.fake.calls if c["method"] == "PUT"][0]
        self.assertTrue(put["url"].endswith("/api/v0/asks/102/"))
        self.assertEqual(set(put["body"]), {"template_hash_id", "label", "disk", "env"})  # image·args·onstart 없음
        self.assertEqual(put["body"]["env"], {"STEWART_ROLE": "serve", "STEWART_GPUS": "1"})   # vast-cli 와 같은 dict(T23)
        self.assertEqual(put["body"]["disk"], 60)
        self.assertEqual(put["body"]["template_hash_id"], HASH)
        self.assertTrue(put["body"]["label"].startswith("stewart-serve-"))
        cur = json.loads((self.rec / "current.json").read_text())
        self.assertEqual((cur["instance_id"], cur["machine_id"], cur["offer_id"], cur["state"]), (5555, 9002, 102, "created"))
        hist = [json.loads(l) for l in (self.rec / "history.jsonl").read_text().splitlines()]
        self.assertEqual(hist[-1]["event"], "up")

    def test_up_refusals(self):
        self.assertEqual(self.run_main("up", "102", "0.34")[0], 69)          # 값이 승인한 상한보다 비쌈
        self.assertEqual(self.run_main("up", "999", "0.5")[0], 69)           # 목록에 없음
        self.assertEqual(self.run_main("up", "102", "0.7")[0], 64)           # 설정 상한 넘음
        self.names["present"] = True
        rc, out, err = self.run_main("up", "102", "0.36")
        self.assertEqual(rc, 69)
        self.assertIn("아직 테일넷에 있다", err)
        self.names["present"] = False
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)
        rc, out, err = self.run_main("up", "103", "0.4")
        self.assertEqual(rc, 69)                                              # 한 번에 1개
        self.assertIn("이미 serve 인스턴스", err)
        self.assertEqual(len([c for c in self.fake.calls if c["method"] == "PUT"]), 1)

    def test_up_picks_from_offers_search(self):
        """T19: up 은 offers 와 같은 검색식으로 받아 그 안에서 id 로 고른다(id 조건은 넣지 않는다)."""
        self.assertEqual(self.run_main("offers")[0], 0)
        rc, out, err = self.run_main("up", "103", "0.37")                    # 목록에 있다(둘째로 싼 것) → 통과
        self.assertEqual(rc, 0, err)
        offers_body, up_body = self.fake.bundle_bodies
        self.assertNotIn("id", up_body)
        self.assertEqual(up_body, offers_body)                                # 같은 조건·같은 상한
        put = [c for c in self.fake.calls if c["method"] == "PUT"]
        self.assertEqual([c["url"].rsplit("/", 2)[-2] for c in put], ["103"])

    def test_up_not_in_list_or_out_of_bounds(self):
        self.fake.offers.append(offer(104, 0.65, 9004))                       # Vast 에는 있으나 설정 값 상한(0.60) 밖 → 서버가 빼서 목록에 없음
        rc, out, err = self.run_main("up", "104", "0.6")
        self.assertEqual(rc, 69)
        self.assertIn("목록에 없다", err)
        self.fake.offers += [offer(200 + n, 0.10 + n / 1000, 9100 + n) for n in range(64)]   # 더 싼 64개에 밀려 limit 밖
        rc, out, err = self.run_main("up", "102", "0.36")
        self.assertEqual(rc, 69)
        self.assertIn("목록에 없다", err)
        self.assertFalse([c for c in self.fake.calls if c["method"] == "PUT"])

    def test_up_rejects_out_of_condition_from_list(self):
        """서버가 돌려준 목록에 있어도 고정 조건 밖이면 거부한다(verified·datacenter·3090·reliability_min·max_dph)."""
        cfg = json.loads(self.cfgp.read_text())
        bad = {111: {"verification": "deverified"}, 112: {"hosting_type": 0}, 113: {"gpu_name": "RTX 4090"},
               114: {"reliability": float(cfg["reliability_min"]) - 0.01}, 115: {"num_gpus": 2}}
        for i, kw in bad.items():
            self.fake.offers.append(offer(i, 0.30, 9200 + i, **kw))
        for i in bad:
            rc, out, err = self.run_main("up", str(i), "0.31")
            self.assertEqual(rc, 69, i)
            self.assertIn("조건 밖", err)
        self.assertFalse([c for c in self.fake.calls if c["method"] == "PUT"])
        self.assertTrue(all("id" not in b for b in self.fake.bundle_bodies))

    # ── T20 역할(serve·eval) ──
    def add_eval_offers(self):
        r = self.eval_rich()                                    # T25: eval 은 디스크·RAM 도 본다
        self.fake.offers += [offer(401, 1.60, 7001, num_gpus=4, **r), offer(402, 1.44, 7002, num_gpus=4, **r),
                             offer(801, 2.96, 7008, num_gpus=8, **r), offer(201, 0.70, 7020, num_gpus=2, **r),
                             offer(491, 2.40, 7091, num_gpus=4, gpu_name="RTX 4090", **r)]

    def lines(self, out):
        return [json.loads(l) for l in out.splitlines() if l.startswith("{")]

    def test_offers_eval_ranks_per_gpu(self):
        self.add_eval_offers()
        rc, out, err = self.run_main("offers", "eval")
        self.assertEqual(rc, 0, err)
        rows = self.lines(out)
        self.assertEqual([r["id"] for r in rows], [801, 402, 401])          # 장당 0.37 · 0.36 · 0.40 — band 0.02 안에서 장 수 많은 것 먼저
        self.assertEqual(rows[0]["dph_per_gpu"], 0.37)
        self.assertIn("up eval 801 2.965", out)
        q = self.fake.bundle_bodies[-1]
        self.assertEqual(q["num_gpus"], {"gte": 4, "lte": 8})
        self.assertEqual(q["dph_total"], {"lte": 6.0})                    # 견본 = 설치값(v12.11h)
        self.assertEqual(q["gpu_name"], {"in": ["RTX 3090", "RTX_3090"]})
        rc, out, err = self.run_main("offers")                              # 옛 형태 = serve: 1장만
        self.assertEqual({r["num_gpus"] for r in self.lines(out)}, {1})
        self.assertEqual(self.fake.bundle_bodies[-1]["num_gpus"], {"eq": 1})
        rc, out, err = self.run_main("offers", "eval", "4090")              # 참고 보기
        self.assertEqual([r["id"] for r in self.lines(out)], [491])
        self.assertIn("참고 보기", out)
        self.assertNotIn("빌리려면", out)

    def eval_rich(self, **kw):
        base = dict(disk_space=500, cpu_ram=257000)
        base.update(kw)
        return base

    def test_offers_eval_count_disk_ram(self):
        """T25: offers eval 8 은 8장만, eval 은 디스크 ≥ disk·RAM ≥ 장당 하한 × 장 수(설정에 있을 때). up 은 offer 의 장 수를 env 로."""
        cfg = json.loads(self.cfgp.read_text())
        cfg["roles"]["eval"]["cpu_ram_gb_per_gpu_min"] = 12
        self.cfgp.write_text(json.dumps(cfg))
        self.fake.offers += [offer(811, 2.40, 7811, num_gpus=8, **self.eval_rich()),
                             offer(812, 2.30, 7812, num_gpus=8, **self.eval_rich(cpu_ram=64000)),     # 8장에 64 GB < 96
                             offer(813, 2.20, 7813, num_gpus=8, **self.eval_rich(disk_space=60)),     # 디스크 60 < 80
                             offer(411, 1.30, 7411, num_gpus=4, **self.eval_rich(cpu_ram=64000)),     # 4장에 64 ≥ 48
                             offer(891, 3.60, 7891, num_gpus=8, gpu_name="RTX 4090", **self.eval_rich())]
        rc, out, err = self.run_main("offers", "eval", "8")
        self.assertEqual(rc, 0, err)
        self.assertEqual([r["id"] for r in self.lines(out)], [811])
        q = self.fake.bundle_bodies[-1]
        self.assertEqual((q["num_gpus"], q["disk_space"], q["cpu_ram"]), ({"eq": 8}, {"gte": 80}, {"gte": 96000}))
        self.assertIn("RTX 3090 ×8,", err + out)
        self.assertIn("up eval 811 2.405 <계획>", out + err)
        rc, out, err = self.run_main("offers", "eval")                               # 4~8
        self.assertEqual([r["id"] for r in self.lines(out)], [811, 411])               # 장당 0.30 · 0.325
        self.assertEqual(self.fake.bundle_bodies[-1]["cpu_ram"], {"gte": 48000})     # 서버 검색은 가장 적은 장 수 기준
        rc, out, err = self.run_main("offers", "eval", "4090", "8")                  # 참고 보기
        self.assertEqual([r["id"] for r in self.lines(out)], [891])
        self.assertNotIn("빌리려면", out + err)
        rc, out, err = self.run_main("up", "eval", "811", "2.405", "a-all")
        self.assertEqual(rc, 0, err)
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertEqual(put["body"]["env"], {"STEWART_ROLE": "eval", "STEWART_GPUS": "8", "STEWART_PLAN": "a-all"})
        self.assertEqual(self.run_main("down", "5555")[0], 0)
        self.assertEqual(self.run_main("up", "eval", "812", "2.305")[0], 69)          # RAM 모자람은 up 도 거부
        self.assertEqual(self.run_main("up", "eval", "813", "2.205")[0], 69)          # 디스크 모자람

    def test_eval_ram_key_optional(self):
        """cpu_ram_gb_per_gpu_min 이 없는 설치본(T24 config)도 읽히고 RAM 조건만 빠진다. 틀린 값·모르는 칸은 거부."""
        cfg = json.loads(self.cfgp.read_text())
        cfg["roles"]["eval"].pop("cpu_ram_gb_per_gpu_min", None)
        self.cfgp.write_text(json.dumps(cfg))
        self.fake.offers.append(offer(812, 2.30, 7812, num_gpus=8, **self.eval_rich(cpu_ram=64000)))
        rc, out, err = self.run_main("offers", "eval", "8")
        self.assertEqual([r["id"] for r in self.lines(out)], [812])
        self.assertNotIn("cpu_ram", self.fake.bundle_bodies[-1])
        for bad in (0, 300, "12", True, 1.5):
            cfg["roles"]["eval"]["cpu_ram_gb_per_gpu_min"] = bad
            self.cfgp.write_text(json.dumps(cfg))
            with self.assertRaises(SC.Stop, msg=bad):
                SC.load_config(str(self.cfgp))
        cfg["roles"]["eval"]["cpu_ram_gb_per_gpu_min"] = 12
        cfg["roles"]["serve"]["cpu_ram_gb_per_gpu_min"] = 12                       # serve 에는 없는 칸
        self.cfgp.write_text(json.dumps(cfg))
        with self.assertRaises(SC.Stop):
            SC.load_config(str(self.cfgp))

    # ── T26 eval2 ──
    def eval2_cfg(self, total=None, eval2=True):
        cfg = json.loads(self.cfgp.read_text())
        if not eval2:
            cfg["roles"].pop("eval2", None)
        if total is not None:
            cfg["eval_total_max_dph"] = total
        self.cfgp.write_text(json.dumps(cfg))

    def test_eval2_up_body_and_records(self):
        self.add_eval_offers()
        rc, out, err = self.run_main("up", "eval2", "402", "1.445", "a-all")
        self.assertEqual(rc, 0, out + err)
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertEqual(put["body"]["env"], {"STEWART_ROLE": "eval2", "STEWART_GPUS": "4", "STEWART_PLAN": "a-all"})
        self.assertEqual((put["body"]["disk"], put["body"]["template_hash_id"]), (80, HASH))   # eval 과 같은 디스크·템플릿
        self.assertTrue(put["body"]["label"].startswith("stewart-eval2-"))
        cur = json.loads((self.rec / "current-eval2.json").read_text())
        self.assertEqual((cur["role"], cur["ts_name"], cur["plan"]), ("eval2", "stewart-gpu-eval2.tail-example.ts.net", "a-all"))
        self.assertFalse((self.rec / "current-eval.json").exists())
        rc, out, err = self.run_main("up", "eval", "401", "1.605", "a-all")       # eval 첫째도 따로 된다
        self.assertEqual(rc, 0, out + err)
        rc, out, err = self.run_main("up", "eval2", "801", "2.965")               # eval2 둘째는 거부(역할마다 1개)
        self.assertEqual(rc, 69)
        self.assertIn("이미 eval2 인스턴스", out + err)
        rc, out, err = self.run_main("status")
        self.assertEqual({r["id"]: r["role"] for r in self.lines(out)}, {5555: "eval2", 5556: "eval"})
        self.assertIn("[eval2] 이름 stewart-gpu-eval2.tail-example.ts.net", out)
        self.assertIn("최대 4개", out)

    def test_eval_total_cap(self):
        """eval + eval2 켜진 값 합 + 새 offer ≤ eval_total_max_dph(기본 6.00). 정지 인스턴스는 세지 않는다."""
        self.add_eval_offers()
        self.eval2_cfg(total=4.0)
        self.assertEqual(self.run_main("up", "eval", "402", "1.445")[0], 0)          # 1.44
        rc, out, err = self.run_main("up", "eval2", "801", "2.965")                   # 1.44 + 2.96 = 4.40 > 4.0
        self.assertEqual(rc, 69)
        self.assertIn("합계 상한 4.0", out + err)
        self.assertEqual(self.run_main("up", "eval2", "401", "1.605")[0], 0)          # 1.44 + 1.60 = 3.04
        self.assertEqual(len([c for c in self.fake.calls if c["method"] == "PUT"]), 2)
        for i in self.fake.instances:                                                 # 둘 다 정지면 합에 안 든다
            if i["id"] in (5555, 5556):
                i["actual_status"] = "exited"
        self.fake.instances = [i for i in self.fake.instances if i["id"] != 5556]     # eval2 는 지운 셈
        self.fake.offers = [o for o in self.fake.offers if o["id"] != 401]
        self.fake.offers.append(offer(811, 3.90, 7811, num_gpus=8, **self.eval_rich()))
        rc, out, err = self.run_main("up", "eval2", "811", "3.905")                   # 켜진 것 0 + 3.90 ≤ 4.0
        self.assertEqual(rc, 0, out + err)
        self.eval2_cfg(total=6.0)
        cfg = SC.load_config(str(self.cfgp))
        self.assertEqual(SC.eval_total(cfg), 6.0)
        c = json.loads(self.cfgp.read_text())
        c.pop("eval_total_max_dph")
        self.cfgp.write_text(json.dumps(c))
        self.assertEqual(SC.eval_total(SC.load_config(str(self.cfgp))), SC.EVAL_TOTAL_DEFAULT)   # 칸이 없으면 6.00
        for bad in (0, -1, 17, True, "6"):
            c["eval_total_max_dph"] = bad
            self.cfgp.write_text(json.dumps(c))
            with self.assertRaises(SC.Stop, msg=bad):
                SC.load_config(str(self.cfgp))

    def test_eval2_absent_or_bad_config(self):
        """roles.eval2 가 없는 설치 config(d38ade64 꼴)는 그대로 읽히고 eval2 만 78. status 는 eval2 를 건너뛴다."""
        self.add_eval_offers()
        self.eval2_cfg(eval2=False)
        rc, out, err = self.run_main("up", "eval2", "402", "1.445")
        self.assertEqual(rc, 78)
        self.assertIn("config.roles.eval2 가 없다", out + err)
        self.assertEqual(self.run_main("offers", "eval2")[0], 78)
        rc, out, err = self.run_main("status")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("[eval2]", out)
        self.assertEqual(self.run_main("up", "eval", "402", "1.445")[0], 0)
        for bad in ({"ts_name": "stewart-gpu-eval.tail-example.ts.net"}, {"ts_name": "stewart-gpu-eval3.tail-example.ts.net"},
                    {"ts_name": "stewart-gpu-eval2.tail-example.ts.net", "max_dph": 9}, {}):
            c = json.loads(self.cfgp.read_text())
            c["roles"]["eval2"] = bad
            self.cfgp.write_text(json.dumps(c))
            with self.assertRaises(SC.Stop, msg=bad):
                SC.load_config(str(self.cfgp))

    def test_down_eval2_uses_eval2_name(self):
        self.add_eval_offers()
        self.assertEqual(self.run_main("up", "eval", "402", "1.445")[0], 0)        # 5555
        self.assertEqual(self.run_main("up", "eval2", "401", "1.605")[0], 0)       # 5556
        self.fake.pod_status_eval = {"instance_id": "5555", "pod_log": []}
        self.fake.pod_status_eval2 = {"instance_id": "5556", "pod_log": []}
        rc, out, err = self.run_main("down", "5556")
        self.assertEqual(rc, 0, out + err)
        lo = [c["url"] for c in self.fake.calls if "logout" in c["url"]]
        self.assertEqual(len(lo), 1)
        self.assertIn("stewart-gpu-eval2.", lo[0])
        self.assertEqual(json.loads((self.rec / "current-eval2.json").read_text())["state"], "destroyed")
        self.assertEqual(json.loads((self.rec / "current-eval.json").read_text())["state"], "created")

    def test_quant_offers_header_names(self):
        """T26 1): quant 머리줄이 RTX 3090 이 아니라 검색식의 A100·H100 이름과 quant 조건을 보인다."""
        self.quant_cfg()
        rc, out, err = self.run_main("offers", "quant")
        head = out.splitlines()[0]
        self.assertIn("A100 SXM4·A100 PCIE·A100X·H100 SXM·H100 PCIE·H100 NVL ×1", head)
        self.assertNotIn("RTX 3090", head)
        self.assertIn("gpu_ram ≥79GB, RAM ≥128GB, CPU ≥16, 디스크 ≥250GB, 망 ≥1000Mb/s", head)

    def test_up_eval_body_and_records(self):
        self.add_eval_offers()
        rc, out, err = self.run_main("up", "eval", "402", "1.445")
        self.assertEqual(rc, 0, err)
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertTrue(put["url"].endswith("/api/v0/asks/402/"))
        self.assertEqual(put["body"]["env"], {"STEWART_ROLE": "eval", "STEWART_GPUS": "4"})
        self.assertEqual(put["body"]["disk"], 80)
        self.assertTrue(put["body"]["label"].startswith("stewart-eval-"))
        cur = json.loads((self.rec / "current-eval.json").read_text())
        self.assertEqual((cur["role"], cur["num_gpus"], cur["ts_name"]), ("eval", 4, "stewart-gpu-eval.tail-example.ts.net"))
        self.assertFalse((self.rec / "current.json").exists())             # serve 의 current.json 은 건드리지 않음
        hist = [json.loads(l) for l in (self.rec / "history.jsonl").read_text().splitlines()]
        self.assertEqual(hist[-1]["role"], "eval")

    def test_role_limits(self):
        self.add_eval_offers()
        self.fake.instances.append({"id": 6001, "label": "stewart-20261109-0903", "machine_id": 10002, "actual_status": "running"})
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 69)          # 옛 이름표 = serve 가 이미 있음
        self.assertEqual(self.run_main("up", "serve", "102", "0.36")[0], 69)
        rc, out, err = self.run_main("up", "eval", "402", "1.445")           # 다른 역할은 된다(합쳐 2)
        self.assertEqual(rc, 0, err)
        rc, out, err = self.run_main("up", "eval", "401", "1.605")           # eval 둘째는 거부
        self.assertEqual(rc, 69)
        self.assertIn("이미 eval 인스턴스", err)
        rc, out, err = self.run_main("status")
        self.assertEqual(rc, 0, err)
        roles = {r["id"]: r["role"] for r in self.lines(out)}
        self.assertEqual(roles, {6001: "serve", 5555: "eval"})
        # 역할을 모르는 stewart- 이름표도 합계에는 센다
        self.fake.instances = [i for i in self.fake.instances if i["id"] != 6001]
        self.fake.instances += [{"id": 6002, "label": "stewart-manual", "machine_id": 1, "actual_status": "running"},
                                {"id": 6003, "label": "stewart-manual2", "machine_id": 2, "actual_status": "running"},
                                {"id": 6004, "label": "stewart-manual3", "machine_id": 3, "actual_status": "running"}]
        rc, out, err = self.run_main("up", "serve", "102", "0.36")
        self.assertEqual(rc, 69)
        self.assertIn("합쳐 최대 4개", err)
        self.assertEqual(len([c for c in self.fake.calls if c["method"] == "PUT"]), 1)

    def test_eval_name_present_only_blocks_eval(self):
        self.add_eval_offers()
        present = {"stewart-gpu-eval.tail-example.ts.net"}
        with mock.patch.object(SC, "name_present", lambda n: n in present):
            rc, out, err = self.run_main("up", "eval", "402", "1.445")
            self.assertEqual(rc, 69)
            self.assertIn("stewart-gpu-eval", err)
            self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)

    def test_eval_out_of_condition(self):
        self.add_eval_offers()
        self.assertEqual(self.run_main("up", "eval", "201", "0.705")[0], 69)   # 2장은 eval 범위 밖(서버가 뺌)
        self.assertEqual(self.run_main("up", "eval", "491", "2.405")[0], 69)   # 4090 은 up 안 됨
        self.assertEqual(self.run_main("up", "eval", "402", "1.43")[0], 69)    # 승인 값보다 비쌈
        self.assertEqual(self.run_main("up", "eval", "102", "0.36")[0], 69)    # 1장은 eval 아님
        self.fake.offers.append(offer(403, 1.40, 7003, num_gpus=4, verification="deverified"))
        rc, out, err = self.run_main("up", "eval", "403", "1.405")
        self.assertEqual(rc, 69)
        self.assertIn("조건 밖", err)
        self.assertFalse([c for c in self.fake.calls if c["method"] == "PUT"])

    def test_down_eval_uses_eval_name(self):
        self.add_eval_offers()
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)            # serve 5555
        self.assertEqual(self.run_main("up", "eval", "402", "1.445")[0], 0)   # eval 5556
        self.fake.pod_status = {"instance_id": "5555", "pod_log": []}
        self.fake.pod_status_eval = {"instance_id": "5556", "pod_log": []}
        rc, out, err = self.run_main("down", "5556")
        self.assertEqual(rc, 0, err)
        lo = [c["url"] for c in self.fake.calls if "logout" in c["url"]]
        self.assertEqual(len(lo), 1)
        self.assertIn("stewart-gpu-eval.", lo[0])                            # serve 이름은 건드리지 않음
        self.assertEqual(json.loads((self.rec / "current-eval.json").read_text())["state"], "destroyed")
        self.assertEqual(json.loads((self.rec / "current.json").read_text())["state"], "created")
        self.assertEqual(sorted(i["id"] for i in self.fake.instances), [777, 5555])
        hist = [json.loads(l) for l in (self.rec / "history.jsonl").read_text().splitlines()]
        self.assertEqual((hist[-1]["event"], hist[-1]["role"]), ("down", "eval"))

    # ── T22 quant ──
    QHASH = QHASH

    def quant_cfg(self, **kw):
        cfg = json.loads(self.cfgp.read_text())
        cfg["roles"]["quant"]["template_hash_id"] = self.QHASH
        cfg["roles"]["quant"].update(kw)
        self.cfgp.write_text(json.dumps(cfg))

    def add_quant_offers(self):
        big = dict(cpu_ram=257000, cpu_cores_effective=32, disk_space=500, inet_down=2500)
        self.fake.offers += [offer(901, 1.20, 8001, gpu_name="A100 SXM4", gpu_ram=81920, **big),
                             offer(902, 1.05, 8002, gpu_name="A100_PCIE", gpu_ram=81920, **big),
                             offer(903, 0.70, 8003, gpu_name="A100 PCIE", gpu_ram=40960, **big),          # 40GB
                             offer(904, 1.55, 8004, gpu_name="H100 SXM", gpu_ram=81559, **big),
                             offer(905, 1.80, 8005, gpu_name="H100 SXM", gpu_ram=81559, **big),           # 1.6 넘음
                             offer(906, 0.90, 8006, gpu_name="A100 SXM4", gpu_ram=81920, **dict(big, cpu_ram=64000)),
                             offer(907, 0.95, 8007, gpu_name="A100 SXM4", gpu_ram=81920, **dict(big, inet_down=500)),
                             offer(908, 0.99, 8008, gpu_name="A100 SXM4", gpu_ram=81920, **dict(big, cpu_cores_effective=8)),
                             offer(909, 0.98, 8009, gpu_name="A100 SXM4", gpu_ram=81920, **dict(big, disk_space=150))]

    def test_quant_offers_query_and_filter(self):
        self.quant_cfg()
        self.add_quant_offers()
        rc, out, err = self.run_main("offers", "quant")
        self.assertEqual(rc, 0, err)
        self.assertEqual([r["id"] for r in self.lines(out)], [902, 901, 904])          # 40GB·RAM·망·CPU·디스크 모자람·1.6 넘음은 빠짐
        q = self.fake.bundle_bodies[-1]
        self.assertIn("A100 SXM4", q["gpu_name"]["in"])
        self.assertIn("H100_SXM", q["gpu_name"]["in"])
        self.assertNotIn("RTX 3090", q["gpu_name"]["in"])
        self.assertEqual((q["gpu_ram"], q["cpu_ram"], q["cpu_cores_effective"], q["disk_space"], q["inet_down"]),
                         ({"gte": 79000}, {"gte": 128000}, {"gte": 16}, {"gte": 250}, {"gte": 1000}))
        self.assertEqual((q["num_gpus"], q["dph_total"]), ({"eq": 1}, {"lte": 1.6}))
        self.assertIn("up quant 902 1.055", out)

    def test_quant_up_body_and_limits(self):
        self.quant_cfg()
        self.add_quant_offers()
        for bad in ("903", "906", "907", "908", "909"):                              # 서버가 걸러 주지 않아도 래퍼가 거부
            rc, out, err = self.run_main("up", "quant", bad, "1.5")
            self.assertEqual(rc, 69, bad)
            self.assertIn("조건 밖", err)
        self.assertEqual(self.run_main("up", "quant", "905", "1.6")[0], 69)            # 값 상한 밖(서버가 뺌)
        self.assertEqual(self.run_main("up", "quant", "905", "1.85")[0], 64)           # 승인 값이 설정 상한 1.60 을 넘음
        rc, out, err = self.run_main("up", "quant", "902", "1.055")
        self.assertEqual(rc, 0, err)
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertEqual(put["body"]["template_hash_id"], self.QHASH)                 # quant 전용 템플릿
        self.assertEqual((put["body"]["disk"], put["body"]["env"]), (250, {"STEWART_ROLE": "quant", "STEWART_GPUS": "1"}))
        self.assertTrue(put["body"]["label"].startswith("stewart-quant-"))
        self.assertNotIn("HF", json.dumps(put["body"]).upper().replace("STEWART", ""))   # 토큰 같은 것은 넣지 않는다
        cur = json.loads((self.rec / "current-quant.json").read_text())
        self.assertEqual((cur["role"], cur["ts_name"]), ("quant", "stewart-quant.tail-example.ts.net"))
        self.assertEqual(self.run_main("up", "quant", "901", "1.205")[0], 69)          # 역할마다 1개
        self.assertEqual(self.run_main("up", "eval", "402", "1.445")[0], 69)           # (eval offer 없음 — 목록에 없음)
        self.add_eval_offers()
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)                     # serve·eval·quant 셋까지
        self.assertEqual(self.run_main("up", "eval", "402", "1.445")[0], 0)
        self.assertEqual(sorted(i["label"].split("-")[1] for i in self.fake.instances if i["label"].startswith("stewart-")),
                         ["eval", "quant", "serve"])

    def test_quant_template_separate(self):
        st = json.loads((self.state / "templates.json").read_text())
        del st["current"]["quant"]                                        # quant 템플릿이 정해지지 않음 → 78(serve 해시가 있어도)
        (self.state / "templates.json").write_text(json.dumps(st))
        rc, out, err = self.run_main("up", "quant", "902", "1.055")
        self.assertEqual(rc, 78)
        self.assertIn("quant 템플릿", err)
        self.assertFalse([c for c in self.fake.calls if "vast.ai" in c["url"]])
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)                     # serve 는 그대로
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertEqual(put["body"]["template_hash_id"], HASH)

    def test_quant_config_checks(self):
        cfg = json.loads(self.cfgp.read_text())
        for b in ({"num_gpus_max": 2}, {"gpu_names": []}, {"gpu_names": ["A100;rm"]}, {"gpu_ram_gb_min": "79"},
                  {"cpu_ram_gb_min": True}, {"inet_down_min": 0}, {"ts_name": "stewart-gpu.tail-example.ts.net"},
                  {"max_dph": 9}):
            c = json.loads(json.dumps(cfg))
            c["roles"]["quant"].update(b)
            p = self.tmp / "c3.json"
            p.write_text(json.dumps(c))
            with self.assertRaises(SC.Stop, msg=b):
                SC.load_config(str(p))
        c = json.loads(json.dumps(cfg))
        del c["roles"]["quant"]["gpu_names"]
        p = self.tmp / "c3.json"
        p.write_text(json.dumps(c))
        with self.assertRaises(SC.Stop):
            SC.load_config(str(p))

    def test_status_elapsed_and_model_fetch(self):
        self.quant_cfg()
        self.add_quant_offers()
        self.assertEqual(self.run_main("up", "quant", "902", "1.055")[0], 0)
        self.fake.pod_status_quant = {"instance_id": "5555", "idle_left_s": 1500, "limits": {"idle_s": 1800},
                                      "model_fetch": {"alias": "qwen38-27b-demo2-w4a16-int8h", "state": "receiving", "files_done": 3,
                                                      "files_total": 12, "bytes_done": 1_200_000_000, "bytes_total": 5_000_000_000,
                                                      "rate_Bps": 52_340_000, "digest": "sha256:" + "a" * 64, "secret": "x" * 50}}
        present = {"stewart-quant.tail-example.ts.net"}
        with mock.patch.object(SC, "name_present", lambda n: n in present), \
                mock.patch.object(SC.time, "time", lambda: 1_700_000_000.0 + 2.5 * 3600):
            rc, out, err = self.run_main("status")
        self.assertEqual(rc, 0, err)
        row = [r for r in self.lines(out) if r["role"] == "quant"][0]
        self.assertEqual((row["hours"], row["usd"]), (2.5, 2.62))                       # 2.5 h × 1.05 $/h
        self.assertIn('model_fetch {"alias": "qwen38-27b-demo2-w4a16-int8h", "state": "receiving", "files_done": 3, "files_total": 12, '
                      '"bytes_done": 1200000000, "bytes_total": 5000000000, "rate_Bps": 52340000, "pct": 24.0, "MBps": 52.3}', out)   # 칸
        self.assertIn("쉬다 정지까지 25분 남음(쉬다 정지 30분)", out)
        self.assertNotIn("secret", out)
        self.assertNotIn("digest", out)
        self.assertEqual(SC.elapsed({"start_date": None, "dph_total": 1}), {"hours": None, "usd": None})

    def test_down_quant_uses_quant_name(self):
        self.quant_cfg()
        self.add_quant_offers()
        self.assertEqual(self.run_main("up", "quant", "902", "1.055")[0], 0)
        self.fake.pod_status_quant = {"instance_id": "5555"}
        self.fake.pod_status = {"instance_id": "1"}
        rc, out, err = self.run_main("down", "5555")
        self.assertEqual(rc, 0, err)
        lo = [c["url"] for c in self.fake.calls if "logout" in c["url"]]
        self.assertEqual(len(lo), 1)
        self.assertIn("stewart-quant.", lo[0])
        self.assertEqual(json.loads((self.rec / "current-quant.json").read_text())["state"], "destroyed")

    # ── T23 ──
    def test_up_eval_with_plan(self):
        self.add_eval_offers()
        rc, out, err = self.run_main("up", "eval", "402", "1.445", "a-x4")
        self.assertEqual(rc, 0, err)
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertEqual(put["body"]["env"], {"STEWART_ROLE": "eval", "STEWART_GPUS": "4", "STEWART_PLAN": "a-x4"})
        self.assertEqual(json.loads((self.rec / "current-eval.json").read_text())["plan"], "a-x4")

    def test_logs_serve_shows_stage_lines_only(self):
        """실사용 서빙(serve)의 컨테이너 로그는 시작 단계 줄(시각<TAB>글)·boot HALT 줄만 — vLLM 출력은 내지 않는다. 데몬 로그는 그대로."""
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)
        self.fake.log_text = ("/dev/shm/b: OK\n2026-11-11T00:00:01Z\tstart instance=5555\n2026-11-11T00:00:09Z\tmodel-fetch hf-mirror alias=x\n"
                              "(APIServer pid=1) INFO 11-11 Received request <요청 글 가림>\nINFO:  127.0.0.1 - \"POST /v1/chat/completions\" 200\n"
                              "사용자 대화 글이 섞인 줄\nHALT: 모델 받기 실패\n")
        with mock.patch.object(SC.time, "sleep", lambda s: None):
            rc, out, err = self.run_main("logs", "5555")
        self.assertEqual(rc, 0, err)
        self.assertIn("start instance=5555", out)
        self.assertIn("model-fetch hf-mirror", out)
        self.assertIn("HALT: 모델 받기 실패", out)
        for hidden in ("Received request", "POST /v1/chat", "대화 글이 섞인"):
            self.assertNotIn(hidden, out)
        self.assertIn("serve: 시작 단계 줄만, 그 밖 3줄은 내지 않음", out)
        self.fake.log_gets = 0
        self.fake.log_text = "daemon: container started\n"
        with mock.patch.object(SC.time, "sleep", lambda s: None):
            rc, out, err = self.run_main("logs", "5555", "daemon")
        self.assertIn("daemon: container started", out)                       # 데몬 로그는 그대로

    def test_status_serve_state_line(self):
        st = {"seen_metrics": True, "pod_log": ["2026-11-11T00:00:01Z\tstart", "2026-11-11T00:03:00Z\tweights pinned lane=0"]}
        self.assertEqual(SC.serve_state(st), "ready — 마지막 단계: weights pinned lane=0")
        self.assertTrue(SC.serve_state({"seen_metrics": False, "pod_log": []}).startswith("loading"))
        self.assertEqual(SC.serve_state({"seen_metrics": False, "pod_log": ["t\tHALT: 모델 받기 실패"]}), "HALT — HALT: 모델 받기 실패")

    def test_logs(self):
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)
        [i for i in self.fake.instances if i["id"] == 5555][0]["label"] = "stewart-eval-20261111-0000"   # serve 는 아래 따로
        self.fake.log_text = ("boot\n\nHALT: pod files\nTS_AUTHKEY=" + FAKE_TSKEY + " curl https://gist.githubusercontent.com/u/x/raw/1/p\n"
                              "Authorization: Bearer " + KEY + "\nsha 048d3b859fc68b3c746f38246e968370cc745230cb91b5c03fcd060e3c5f9c58  p: FAILED\n"
                              + "".join(f"line {n}\n" for n in range(300)))
        with mock.patch.object(SC.time, "sleep", lambda s: None):
            rc, out, err = self.run_main("logs", "5555")
        self.assertEqual(rc, 0, err)
        req = [c for c in self.fake.calls if "request_logs" in c["url"]][0]
        self.assertEqual((req["method"], req["body"]), ("PUT", {"tail": "200"}))
        self.assertTrue(req["url"].endswith("/api/v0/instances/request_logs/5555/"))
        get = [c for c in self.fake.calls if "example-bucket" in c["url"]]
        self.assertEqual(len(get), 3)                                         # 404 두 번 뒤 받음
        self.assertIsNone(get[-1].get("auth"))                               # 로그 주소에는 키를 보내지 않는다
        self.assertEqual(len([l for l in out.splitlines() if l.startswith("line ")]), 200 - 0)   # 끝 200줄
        self.assertIn("line 299", out)
        self.fake.log_text = "HALT: pod files\nTS_AUTHKEY=" + FAKE_TSKEY + "\ncurl https://gist.githubusercontent.com/u/x/raw/1/p\n" \
                             "Authorization: Bearer " + KEY + "\napi_key=zzz token=yyy\nsha 048d3b859fc68b3c746f38246e968370cc745230cb91b5c03fcd060e3c5f9c58  p: FAILED\n"
        self.fake.log_gets = 0
        with mock.patch.object(SC.time, "sleep", lambda s: None):
            rc, out, err = self.run_main("logs", "5555", "daemon")
        self.assertEqual(rc, 0, err)
        self.assertEqual([c for c in self.fake.calls if "request_logs" in c["url"]][-1]["body"], {"tail": "200", "daemon_logs": "true"})
        for leak in ("tskey-", "gist.githubusercontent", KEY, "zzz", "yyy"):
            self.assertNotIn(leak, out)
        self.assertIn("HALT: pod files", out)
        self.assertIn("048d3b859fc68b3c746f38246e968370cc745230cb91b5c03fcd060e3c5f9c58  p: FAILED", out)   # sha 는 남긴다
        self.assertEqual(self.run_main("logs", "777")[0], 69)                # 남의 이름표
        self.assertEqual(self.run_main("logs", "4242")[0], 69)               # 목록에 없음
        self.fake.log_404s = 99
        self.fake.log_gets = 0
        with mock.patch.object(SC.time, "sleep", lambda s: None):
            rc, out, err = self.run_main("logs", "5555")
        self.assertEqual(rc, 70)
        self.assertIn("아직 안 나왔다", err)

    def test_result_url_shape(self):
        for bad in ("http://logs.example-bucket.com/x", "https://10.0.0.1/x", "https://stewart-gpu.tail-example.ts.net/x",
                    "https://localhost/x", "file:///etc/passwd", ""):
            with self.assertRaises(SC.Stop, msg=bad):
                SC.fetch_result(bad, tries=1, wait_s=0)

    def test_status_stall_warning(self):
        self.add_eval_offers()
        self.assertEqual(self.run_main("up", "eval", "402", "1.445")[0], 0)
        for i in self.fake.instances:
            if i["id"] == 5555:
                i["actual_status"] = "running"
        with mock.patch.object(SC.time, "time", lambda: 1_700_000_000.0 + 79 * 60):
            rc, out, err = self.run_main("status")
        self.assertIn("5555(eval) 는 running 인데 켜진 지 79분째 테일넷 이름 stewart-gpu-eval.tail-example.ts.net 이 비었다 — 부팅 멈춤 의심 — logs 5555", out)
        with mock.patch.object(SC.time, "time", lambda: 1_700_000_000.0 + 9 * 60):
            rc, out, err = self.run_main("status")
        self.assertNotIn("부팅 멈춤", out)                                    # 10분 전
        with mock.patch.object(SC, "name_present", lambda n: True), \
                mock.patch.object(SC.time, "time", lambda: 1_700_000_000.0 + 79 * 60):
            rc, out, err = self.run_main("status")
        self.assertNotIn("부팅 멈춤", out)                                    # 이름이 붙었으면 아님

    # ── T24 ──
    def test_up_replaces_stopped_same_role(self):
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)                     # serve 5555
        for i in self.fake.instances:
            if i["id"] == 5555:
                i["actual_status"], i["intended_status"] = "exited", "stopped"        # 쉬다 정지
        self.fake.pod_status = None                                                   # 정지한 pod 은 /status 를 못 읽는다(logout 건너뜀)
        rc, out, err = self.run_main("up", "103", "0.37")
        self.assertEqual(rc, 0, err)
        self.assertIn("같은 역할(serve)의 정지 인스턴스 5555(exited) 를 먼저 지운다", out)
        dels = [c for c in self.fake.calls if c["method"] == "DELETE"]
        puts = [c for c in self.fake.calls if c["method"] == "PUT" and "/asks/" in c["url"]]
        self.assertEqual(len(dels), 1)
        self.assertTrue(dels[0]["url"].endswith("/instances/5555/"))
        self.assertLess(self.fake.calls.index(dels[0]), self.fake.calls.index(puts[-1]))   # 지운 뒤 빌린다
        self.assertEqual(sorted(i["id"] for i in self.fake.instances), [777, 5556])
        hist = [json.loads(l) for l in (self.rec / "history.jsonl").read_text().splitlines()]
        self.assertEqual([h["event"] for h in hist], ["up", "down", "up"])

    def test_up_running_same_role_still_refused(self):
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)
        for i in self.fake.instances:
            if i["id"] == 5555:
                i["actual_status"] = "running"
        rc, out, err = self.run_main("up", "103", "0.37")
        self.assertEqual(rc, 69)
        self.assertIn("돌고 있음", err)
        self.assertFalse([c for c in self.fake.calls if c["method"] == "DELETE"])

    def test_up_stopped_other_role_not_touched(self):
        self.add_eval_offers()
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)                     # serve
        for i in self.fake.instances:
            if i["id"] == 5555:
                i["actual_status"] = "exited"
        self.assertEqual(self.run_main("up", "eval", "402", "1.445")[0], 0)            # 다른 역할은 지우지 않는다
        self.assertFalse([c for c in self.fake.calls if c["method"] == "DELETE"])

    def test_up_replace_fails_stops(self):
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)
        for i in self.fake.instances:
            if i["id"] == 5555:
                i["actual_status"] = "stopped"
        self.fake.fail[("DELETE", SC.API + "/v0/instances/5555/")] = 500
        rc, out, err = self.run_main("up", "103", "0.37")
        self.assertNotEqual(rc, 0)
        self.assertEqual(len([c for c in self.fake.calls if c["method"] == "PUT" and "/asks/" in c["url"]]), 1)   # 새로 빌리지 않음

    def test_role_env_only_fixed_words(self):
        self.assertEqual(SC.role_env("eval", 8), {"STEWART_ROLE": "eval", "STEWART_GPUS": "8"})
        self.assertEqual(SC.role_env("eval", 4, "a-x4"), {"STEWART_ROLE": "eval", "STEWART_GPUS": "4", "STEWART_PLAN": "a-x4"})
        for bad in (("serve", 1, "a-x4"), ("quant", 1, "a-x4"), ("eval", 4, "A B"), ("eval", 4, "")):
            with self.assertRaises(SC.Stop):
                SC.role_env(*bad)
        for bad in (("eval", 9), ("eval", 0), ("eval", "4"), ("box", 1), ("eval -e X=1", 4)):
            with self.assertRaises(SC.Stop):
                SC.role_env(*bad)

    def test_down(self):
        self.run_main("up", "102", "0.36")
        self.assertEqual(self.run_main("down", "777")[0], 69)                # 라벨이 stewart- 가 아니다
        self.assertEqual(self.run_main("down", "1234")[0], 69)               # 목록에 없다
        self.assertFalse([c for c in self.fake.calls if c["method"] == "DELETE"])
        rc, out, err = self.run_main("down", "5555")
        self.assertEqual(rc, 0, err)
        order = [c["method"] + " " + c["url"].rsplit("/", 2)[-2] for c in self.fake.calls[-4:]]
        self.assertTrue(any("logout" in c["url"] for c in self.fake.calls))
        i_lo = [n for n, c in enumerate(self.fake.calls) if "logout" in c["url"]][-1]
        i_del = [n for n, c in enumerate(self.fake.calls) if c["method"] == "DELETE"][-1]
        self.assertLess(i_lo, i_del)                                          # logout 이 Destroy 보다 먼저
        cur = json.loads((self.rec / "current.json").read_text())
        self.assertEqual(cur["state"], "destroyed")
        self.assertEqual(self.fake.instances[0]["id"], 777)                   # 남의 것은 그대로

    def test_down_logout_only_if_same_instance(self):
        cases = [  # (이름을 쥔 pod 의 /status, logout 호출 수, 출력에 있어야 할 말)
            ({"instance_id": "5555"}, 1, "비었다"),                                          # 같음 → logout
            ({"pod_log": ["t\tstart instance=5555 label=x"]}, 1, "pod logout:"),             # v5 pod: 줄로 대조
            ({"instance_id": "4444", "pod_log": ["start instance=5555"]}, 0, "다른 인스턴스(4444)"),   # 다름 → 건너뜀(칸 우선)
            (None, 0, "읽지 못해"),                                                            # 못 읽음 → 건너뜀
            ({"pod_log": ["start instance=9999"]}, 0, "다른 인스턴스(9999)"),
            ({"instance_id": "x;1"}, 0, "읽지 못해"),                                          # 형식이 다름 → 모른다
            ({"pod_log": []}, 0, "읽지 못해"),
        ]
        for st, n_logout, word in cases:
            self.fake = FakeVast()
            self.fake.pod_status = st
            with mock.patch.object(SC, "_OPENER", self.fake):
                self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)
                rc, out, err = self.run_main("down", "5555")
            self.assertEqual(rc, 0, (st, err))
            self.assertEqual(len([c for c in self.fake.calls if c["url"].endswith("/logout")]), n_logout, st)
            self.assertIn(word, out, st)
            self.assertFalse(any(i["id"] == 5555 for i in self.fake.instances), st)          # Destroy 는 늘 한다
            os.remove(self.rec / "current.json")

    def test_down_logout_fails_still_destroys(self):
        self.run_main("up", "102", "0.36")
        self.fake.logout_ok = False
        self.names["present"] = True
        rc, out, err = self.run_main("down", "5555")
        self.assertEqual(rc, 0, err)
        self.assertIn("실패", out)
        self.assertIn("30~60분", out)
        self.assertFalse(any(i["id"] == 5555 for i in self.fake.instances))

    def test_status_and_errors(self):
        self.run_main("up", "102", "0.36")
        self.fake.instances[-1]["actual_status"] = "exited"
        rc, out, err = self.run_main("status")
        self.assertEqual(rc, 0)
        self.assertIn("디스크 요금", out)
        self.assertNotIn("someone-else", out)                                 # 남의 인스턴스는 보이지 않는다
        self.fake.fail[("GET", SC.API + "/v1/instances/?limit=25&order_by=%5B%7B%22col%22%3A+%22id%22%2C+%22dir%22%3A+%22asc%22%7D%5D&select_filters=%7B%7D")] = 403
        rc, out, err = self.run_main("status")
        self.assertEqual(rc, 77)
        self.assertIn("user_read", err)
        self.assertNotIn(KEY, err)
        self.assertIn("[가림]", err)                                          # 오류 msg 안 키 모양 문자열

    def test_list_v1_pages(self):
        self.fake.instances += [{"id": 1000 + n, "label": "x", "machine_id": n, "actual_status": "running"} for n in range(30)]
        self.fake.instances.append({"id": 9999, "label": "stewart-20261108-1900", "machine_id": 9, "actual_status": "running"})
        rc, out, err = self.run_main("status")
        self.assertEqual(rc, 0, err)
        self.assertIn("9999", out)                            # 둘째 쪽에 있는 우리 인스턴스도 본다
        self.assertEqual(len(self.fake.list_queries), 2)
        self.assertNotIn("after_token", self.fake.list_queries[0])
        self.assertEqual(self.fake.list_queries[1]["after_token"], "tok25")
        self.assertEqual(json.loads(self.fake.list_queries[0]["order_by"]), [{"col": "id", "dir": "asc"}])
        self.assertEqual(self.fake.list_queries[0]["limit"], "25")
        self.assertFalse(any("/v0/instances/" in c["url"] and c["method"] == "GET" for c in self.fake.calls))

    def test_list_v1_bad_shape(self):
        orig = self.fake.open

        def bad(req, timeout=None):
            if "/v1/instances/" in req.full_url:
                return FakeResp({"success": True, "instances": {"id": 1}})
            return orig(req, timeout)
        self.fake.open = bad
        rc, out, err = self.run_main("status")
        self.assertEqual(rc, 70)
        self.assertIn("instances 목록이 없다", err)

    def test_offer_verification_string(self):
        cfg = json.loads(self.cfgp.read_text())
        self.assertIsNone(SC.offer_ok(cfg, offer(1, 0.3, 1)))
        for bad in ("deverified", "unverified"):
            self.assertIn("verified", SC.offer_ok(cfg, offer(1, 0.3, 1, verification=bad)))
        legacy = offer(1, 0.3, 1)
        del legacy["verification"]
        self.assertIsNone(SC.offer_ok(cfg, legacy))           # 칸이 없으면 서버 검색 조건에 기댄다
        self.assertIn("datacenter", SC.offer_ok(cfg, offer(1, 0.3, 1, hosting_type=0)))
        self.assertIsNone(SC.offer_ok(cfg, offer(1, 0.3, 1, hosting_type=0, datacenter=True)))   # 칸이 오면 그 값

    def test_paths_fixed(self):
        with self.assertRaises(SC.Stop):
            SC.vast(KEY, "GET", "/users/current/")
        with self.assertRaises(SC.Stop):
            SC.vast(KEY, "PUT", "/v0/instances/1/../../users/")
        for method, path in (("GET", "/v0/instances/"), ("GET", "/instances/"), ("GET", "/v0/instances/1/"),
                             ("PUT", "/v0/instances/1/"), ("DELETE", "/v1/instances/"), ("POST", "/v1/bundles/")):
            with self.assertRaises(SC.Stop):
                SC.vast(KEY, method, path)            # 방법·판까지 맞아야 한다(정지 PUT 은 없다)
        with self.assertRaises(SC.Stop):
            SC.vast(KEY, "GET", "/v1/instances/", query={"owner": "all"})
        with self.assertRaises(SC.Stop):
            SC.vast(KEY, "GET", "/v1/instances/", query={"after_token": "../x y"})
        with self.assertRaises(SC.Stop):
            SC.vast(KEY, "POST", "/v0/bundles/", query={"limit": 1})
        src = (HERE / "stewart_cloud.py").read_text()
        code = re.sub(r'"""[\s\S]*?"""|#.*', "", src)
        for word in ("billing", "api_key", "templates/", "target_state", "invoices"):
            self.assertFalse(word in code, word)          # 결제 API 를 부르지 않는다
        # T32: 정지는 stop 명령(do_state) 하나 — 본문은 vast() 가 {"state": running|stopped} 만 받는다
        self.assertEqual(code.count('{"state": "stopped"}'), 1)                         # vast() 의 허용 본문 한 곳
        self.assertEqual(len(re.findall(r'do_state\(', code)), 2)                       # 정의 + main 한 곳
        # 계정 API 는 사용자 정보 읽기(GET /v0/users/current, balance 의 잔액 두 칸) 하나만. 그 밖의 /users 경로는 없다
        self.assertEqual(re.findall(r'"/v0/users/[^"]*"', code), ['"/v0/users/current"'] * 2)   # _PATHS 와 do_balance
        self.assertEqual(len(re.findall(r'/users', code)), 2)                                 # _PATHS 한 줄 + do_balance 한 줄
        for method, path in (("PUT", "/v0/users/current"), ("POST", "/v0/users/current"), ("GET", "/v0/users/current/reset-apikey/"),
                             ("GET", "/v0/users/me/invoices")):
            with self.assertRaises(SC.Stop):
                SC.vast(KEY, method, path)
        for method in ("PUT", "DELETE"):                   # 템플릿은 찾기(GET)·만들기(POST)만(T27) — 고치기·지우기 없음
            with self.assertRaises(SC.Stop):
                SC.vast(KEY, method, "/v0/template/")
        with self.assertRaises(SC.Stop):
            SC.vast(KEY, "GET", "/v0/template/", query={"limit": 1})


# ── T27 publish ──────────────────────────────────────────────────

FAKE_BUILDER = """import base64, hashlib, os, sys
out = sys.argv[sys.argv.index("--out") + 1]
here = os.path.dirname(os.path.abspath(__file__))
data = b"".join(open(os.path.join(here, n), "rb").read() for n in sorted(os.listdir(here)) if n.endswith(".sh"))
if not os.path.basename(os.environ.get("HOME", "")).startswith("stewart-publish-"):
    raise SystemExit("HOME 이 깨끗한 임시 폴더가 아니다")
s = base64.b64encode(data).decode()
sha = hashlib.sha256(s.encode()).hexdigest()
sb = base64.b64encode(b"#!/bin/bash\\necho boot " + KIND.encode() + b"\\n").decode()
args = "bash -c 'boot " + hashlib.sha256(sb.encode()).hexdigest() + "'"
cf = " -e CF=<설정 gist 최신 raw 주소>" if KIND == "pod" else ""
def w(n, t):
    with open(os.path.join(out, n), "w", encoding="utf-8") as f:
        f.write(t)
w(NAME, s); w(NAME[:-4] + ".sha256", sha + "\\n"); w("args.txt", args); w("boot.sb.txt", sb)
w("docker-options.txt", "-e CTX=fast -e SPEC=mtp -e PU=<gist raw 주소> -e PS=" + sha + cf + " -e SB=" + sb + "\\n")
"""

GIT_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_AUTHOR_DATE": "2026-11-09T19:00:00+0900",
           "GIT_COMMITTER_DATE": "2026-11-09T19:00:00+0900"}


class Publish(Base):
    def setUp(self):
        super().setUp()
        self.repo = self.tmp / "repo"
        self.git("init", "-q", "-b", "main", str(self.repo), cwd=self.tmp)
        (self.repo / "deploy" / "vast").mkdir(parents=True)
        (self.repo / "deploy" / "quant").mkdir(parents=True)
        (self.repo / "deploy" / "serving" / "sub").mkdir(parents=True)
        (self.repo / "deploy" / "vast" / "make_template_env.py").write_text(
            'KIND, NAME = "pod", "pod-files.b64"\n' + FAKE_BUILDER)
        (self.repo / "deploy" / "quant" / "make_quant_bundle.py").write_text(
            'KIND, NAME = "quant", "quant-files.b64"\n' + FAKE_BUILDER)
        (self.repo / "deploy" / "vast" / "pod-start.sh").write_text("#!/bin/bash\necho v12\n" * 50)
        (self.repo / "deploy" / "quant" / "run-convert.sh").write_text("#!/bin/bash\necho quant\n")
        (self.repo / "deploy" / "serving" / "sub" / "x.toml").write_text("a = 1\n")
        os.chmod(self.repo / "deploy" / "vast" / "pod-start.sh", 0o755)
        (self.repo / "README").write_text("밖\n")
        self.commit = self.commit_all("v12")
        self.labs = self.tmp / "labs"
        self.labs.mkdir()
        self.order = []
        self.py = [sys.executable, "-I", "-B"]
        self.real_prefix = SC.build_prefix
        p = mock.patch.object(SC, "build_prefix", lambda out: self.order.append("build") or list(self.py))
        p.start()
        self.addCleanup(p.stop)

    def git(self, *a, cwd=None):
        r = subprocess.run(["git", *a], cwd=cwd or self.repo, env=GIT_ENV, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def commit_all(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def labs_build(self, bundle, commit=None):
        """세션이 하던 대로 labs/<묶음>/ 을 만든다(같은 커밋에서)."""
        d = self.labs / bundle
        d.mkdir()
        files = SC.rebuild(bundle, commit or self.commit, str(self.repo / ".git"))
        for n, b in files.items():
            (d / n).write_bytes(b)
        return SC.bundle_facts(bundle, files)["ps"]

    def pub(self, *argv, **kw):
        return self.run_main(*argv, git_dir=str(self.repo / ".git"), labs=str(self.labs),
                             key_fn=kw.pop("key_fn", lambda: self.order.append("key") or KEY), **kw)

    # ── git 객체 ──
    def test_git_export_loose_and_packed(self):
        want = {str(p.relative_to(self.repo)): p.read_bytes() for p in (self.repo / "deploy").rglob("*") if p.is_file()}
        for packed in (False, True):
            if packed:
                for i in range(3):                                  # 델타가 생기게 비슷한 판을 몇 개
                    (self.repo / "deploy" / "vast" / "pod-start.sh").write_text("#!/bin/bash\necho v12\n" * 50 + f"# {i}\n")
                    self.commit_all(f"d{i}")
                self.git("gc", "-q", "--aggressive", "--prune=now")
                self.assertFalse([p for p in (self.repo / ".git" / "objects").glob("??/*")])     # 낱 객체 없음
            dest = self.tmp / f"x{packed}"
            n = SC.GitStore(str(self.repo / ".git")).export(self.commit, "deploy", str(dest))
            got = {str(p.relative_to(dest)): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
            self.assertEqual((n, got), (len(want), want), packed)
            self.assertEqual(os.stat(dest / "deploy" / "vast" / "pod-start.sh").st_mode & 0o777, 0o755)
            self.assertFalse((dest / "README").exists())             # deploy/ 밖은 꺼내지 않는다

    def test_git_tampered_object_refused(self):
        blob = self.git("rev-parse", f"{self.commit}:deploy/vast/pod-start.sh")
        p = self.repo / ".git" / "objects" / blob[:2] / blob[2:]
        os.chmod(p, 0o644)
        p.write_bytes(SC.zlib.compress(b"blob 9\0evil code"[:7] + b"9\0evilcode"))      # 이름과 다른 내용
        p.write_bytes(SC.zlib.compress(b"blob 8\0evilcode"))
        with self.assertRaises(SC.Stop) as e:
            SC.GitStore(str(self.repo / ".git")).export(self.commit, "deploy", str(self.tmp / "y"))
        self.assertEqual(e.exception.code, 69)
        self.assertIn("SHA-1", str(e.exception))

    def test_git_symlink_and_alternates_refused(self):
        os.symlink("/etc/passwd", self.repo / "deploy" / "vast" / "link")
        c = self.commit_all("link")
        with self.assertRaises(SC.Stop) as e:
            SC.GitStore(str(self.repo / ".git")).export(c, "deploy", str(self.tmp / "z"))
        self.assertIn("120000", str(e.exception))
        (self.repo / ".git" / "objects" / "info").mkdir(exist_ok=True)
        (self.repo / ".git" / "objects" / "info" / "alternates").write_text("/tmp/other\n")
        with self.assertRaises(SC.Stop):
            SC.GitStore(str(self.repo / ".git"))
        with self.assertRaises(SC.Stop):
            SC.GitStore(str(self.repo / ".git" / "nope"))

    def test_git_filters_and_replace_ignored(self):
        """저장소 설정의 filter·replace 는 래퍼가 읽는 내용을 바꾸지 못한다(git 실행 파일을 거치지 않음)."""
        (self.repo / ".gitattributes").write_text("*.sh filter=evil\n")
        self.git("config", "filter.evil.smudge", "echo PWNED")
        c = self.commit_all("attr")
        fake = self.git("hash-object", "-w", "--stdin", cwd=self.repo) if False else None
        blob = self.git("rev-parse", f"{c}:deploy/vast/pod-start.sh")
        other = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=self.repo, env=GIT_ENV, input="evil\n",
                               capture_output=True, text=True).stdout.strip()
        self.git("replace", blob, other)
        dest = self.tmp / "w"
        SC.GitStore(str(self.repo / ".git")).export(c, "deploy", str(dest))
        self.assertEqual((dest / "deploy" / "vast" / "pod-start.sh").read_text(), "#!/bin/bash\necho v12\n" * 50)
        self.assertIsNone(fake)

    # ── 재현 대조(verify) ──
    def test_verify_ok_and_failures(self):
        ps = self.labs_build("v12")
        self.order.clear()
        rc, out, err = self.pub("verify", "v12", self.commit, ps)
        self.assertEqual(rc, 0, err)
        self.assertIn("재현 대조 통과[v12]", out)
        self.assertEqual(self.order, ["build"])                                  # 키를 읽지 않는다
        self.assertEqual(self.fake.calls, [])                                    # 망도 쓰지 않는다
        self.assertEqual(self.pub("verify", "v12", self.commit, "0" * 64)[0], 69)      # 인자 PS 와 다름
        (self.repo / "deploy" / "vast" / "pod-start.sh").write_text("#!/bin/bash\necho other\n")
        c2 = self.commit_all("other")
        rc, out, err = self.pub("verify", "v12", c2, ps)                         # 다른 커밋 → 다른 묶음
        self.assertEqual(rc, 69)
        self.assertIn("재현 대조 실패", err)
        (self.labs / "v12" / "boot.sb.txt").write_text("x")                      # labs 가 바뀜(SB)
        rc, out, err = self.pub("verify", "v12", self.commit, ps)
        self.assertEqual(rc, 69)
        self.assertIn("boot.sb.txt", err)
        (self.labs / "v12" / "boot.sb.txt").unlink()
        self.assertIn("없다", self.pub("verify", "v12", self.commit, ps)[2])
        self.assertEqual(self.pub("verify", "v13", self.commit, ps)[0], 69)     # labs/v13 없음
        self.assertEqual(self.pub("verify", "v12", "f" * 40, ps)[0], 69)        # 저장소에 없는 커밋
        self.py = [sys.executable, "-I", "-B", "-c", "raise SystemExit(3)"]    # 빌더 실패
        self.assertEqual(self.pub("verify", "v12", self.commit, ps)[0], 69)

    def test_verify_quant_and_forms(self):
        ps = self.labs_build("quant-v4")
        self.assertEqual(self.pub("verify", "quant-v4", self.commit, ps)[0], 0)
        for bad in (["verify", "v12", self.commit[:12], ps], ["verify", "v0", self.commit, ps], ["verify", "V12", self.commit, ps],
                    ["verify", "v12", self.commit, ps.upper()], ["publish", "v12", self.commit], ["publish", "pod-v12", self.commit, ps],
                    ["templates", "x"], ["verify", "v12", self.commit, ps, "x"]):
            self.assertEqual(self.pub(*bad)[0], 64, bad)

    def test_build_prefix_needs_root_python(self):
        with mock.patch.object(SC, "BUILD_PY", str(self.tmp / "py" / "bin" / "python3.13")):
            with self.assertRaises(SC.Stop) as e:
                self.real_prefix("/x")
            self.assertEqual(e.exception.code, 78)
            (self.tmp / "py" / "bin").mkdir(parents=True)
            (self.tmp / "py" / "bin" / "python3.13").write_text("")
            with self.assertRaises(SC.Stop):                                    # 사용자 소유 → 거부
                self.real_prefix("/x")
        with mock.patch.object(SC, "BUILD_PY", "/usr/bin/python3"):            # root 소유면 sandbox-exec 명령
            cmd = self.real_prefix("/private/tmp/x/out")
        self.assertEqual(cmd[:7], ["/usr/bin/sandbox-exec", "-D", "OUT=/private/tmp/x/out", "-D", "TMP=/private/tmp/x", "-p", SC.BUILD_PROFILE])
        for rule in ("(deny network*)", "(deny mach-lookup)", "(deny process-fork)",
                     '(deny file-read* (subpath "/Volumes") (subpath "/Users") (subpath "/private/tmp") (subpath "/private/var/folders"))',
                     '(allow file-read* (subpath (param "TMP")))',
                     "(deny file-write*)", '(allow file-write* (subpath (param "OUT"))'):
            self.assertIn(rule, SC.BUILD_PROFILE)

    # ── publish ──
    def test_publish_end_to_end(self):
        ps = self.labs_build("v12")
        (self.labs / "v12" / "image.txt").write_text(IMG + "\n")
        self.order.clear()
        rc, out, err = self.pub("publish", "v12", self.commit, ps)
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.order, ["build", "key"])                         # 다시 만들기는 키를 읽기 전에 끝난다
        new = self.fake.tpl_posts[0]
        h = new["hash_id"]
        pu = f"https://gist.githubusercontent.com/{GH_OWNER}/{GID}/raw/{sha(self.fake.gist[GID]['files']['stewart-pod-v12.b64'])[:40]}/stewart-pod-v12.b64"
        self.assertEqual(self.fake.gist[GID]["files"]["stewart-pod-v12.b64"], (self.labs / "v12" / "pod-files.b64").read_text())
        self.assertEqual(new["env"], (self.labs / "v12" / "docker-options.txt").read_text().strip()
                         .replace("<gist raw 주소>", pu).replace("<설정 gist 최신 raw 주소>", CF_URL))
        self.assertNotRegex(new["env"], r"<[^>]+>")
        self.assertEqual(new["args_str"], (self.labs / "v12" / "args.txt").read_text())
        base = self.fake.templates[HASH]
        self.assertEqual([new[k] for k in ("image", "runtype", "recommended_disk_space")],
                         [base[k] for k in ("image", "runtype", "recommended_disk_space")])
        self.assertTrue(new["private"])
        self.assertNotIn("tag", new)                                          # 바탕에 없던(None) 칸은 넣지 않는다
        st = json.loads((self.state / "templates.json").read_text())
        self.assertEqual(st["current"], {"pod": h, "quant": QHASH})
        self.assertEqual({k: st["templates"][h][k] for k in ("bundle", "commit", "ps", "source", "base", "image")},
                         {"bundle": "v12", "commit": self.commit, "ps": ps, "source": "publish", "base": HASH, "image": IMG})
        self.assertEqual(oct(os.stat(self.state / "templates.json").st_mode & 0o777), "0o600")
        hist = [json.loads(l) for l in (self.rec / "history.jsonl").read_text().splitlines()]
        self.assertEqual((hist[-1]["event"], hist[-1]["template_hash_id"], hist[-1]["previous"]), ("publish", h, HASH))
        self.assertNotIn("gist.githubusercontent", out + err + json.dumps(hist))   # gist 주소는 찍지 않는다
        # 토큰은 GitHub API 헤더에만, raw 받기에는 없다. Vast 템플릿 호출에는 Vast 키
        self.assertTrue(self.fake.gh_calls and all(c["auth"] == f"Bearer {GHTOKEN}" for c in self.fake.gh_calls))
        raws = [c for c in self.fake.calls if c["url"].startswith("https://gist.githubusercontent.com/")]
        self.assertTrue(raws and all(c.get("auth") is None for c in raws))
        self.assertFalse([c for c in self.fake.calls if GHTOKEN in json.dumps(c) and "api.github.com" not in c["url"]])
        # 그 뒤 up 은 새 템플릿으로
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertEqual(put["body"]["template_hash_id"], h)
        # 같은 묶음을 다시 publish 해도 gist 를 덮지 않고 그대로 쓴다(되돌리기에도 같은 길)
        n_patch = len([c for c in self.fake.gh_calls if c["method"] == "PATCH"])
        self.assertEqual(self.pub("publish", "v12", self.commit, ps)[0], 0)
        self.assertEqual(len([c for c in self.fake.gh_calls if c["method"] == "PATCH"]), n_patch)

    def test_publish_quant(self):
        ps = self.labs_build("quant-v4")
        rc, out, err = self.pub("publish", "quant-v4", self.commit, ps)
        self.assertEqual(rc, 0, err)
        new = self.fake.tpl_posts[0]
        self.assertNotIn("CF=", new["env"])
        self.assertIn("stewart-quant-v4.b64", new["env"])
        self.assertEqual(new["recommended_disk_space"], 200)
        st = json.loads((self.state / "templates.json").read_text())
        self.assertEqual(st["current"], {"pod": HASH, "quant": new["hash_id"]})

    def test_publish_gist_mismatch_stops(self):
        ps = self.labs_build("v12")
        before = (self.state / "templates.json").read_text()
        # gist 가 다른 내용을 내준다(리비전 고정 주소를 받아 대조)
        content = (self.labs / "v12" / "pod-files.b64").read_text()
        url = f"https://gist.githubusercontent.com/{GH_OWNER}/{GID}/raw/{sha(content)[:40]}/stewart-pod-v12.b64"
        self.fake.raw_override[url] = content[:-4] + "AAAA"
        rc, out, err = self.pub("publish", "v12", self.commit, ps)
        self.assertEqual(rc, 69)
        self.assertIn("PS 와 다르다", err)
        self.assertEqual(self.fake.tpl_posts, [])                                    # 템플릿을 만들지 않는다
        self.assertEqual((self.state / "templates.json").read_text(), before)
        # 같은 이름의 파일이 이미 다른 내용으로 있으면 덮지 않는다
        self.fake.raw_override.clear()
        self.fake.gist[GID]["files"]["stewart-pod-v12.b64"] = "QUJD"
        rc, out, err = self.pub("publish", "v12", self.commit, ps)
        self.assertEqual(rc, 69)
        self.assertIn("덮지 않는다", err)
        self.assertEqual(self.fake.gist[GID]["files"]["stewart-pod-v12.b64"], "QUJD")

    def test_publish_template_reread_mismatch_stops(self):
        ps = self.labs_build("v12")
        before = (self.state / "templates.json").read_text()
        for mutate, word in ((lambda t: {**t, "env": t["env"].replace("-e PS=", "-e PS=0")}, "ps"),
                             (lambda t: {**t, "env": t["env"] + " -e X=<빈칸>"}, "빈칸"),
                             (lambda t: {**t, "args_str": t["args_str"] + " "}, "args_sha"),
                             (lambda t: {**t, "image": "ghcr.io/other@sha256:" + "0" * 64}, "image"),
                             (lambda t: {k: v for k, v in t.items() if k != "env"}, "ps")):
            self.fake.tpl_mutate = mutate
            self.fake.tpl_posts.clear()
            rc, out, err = self.pub("publish", "v12", self.commit, ps)
            self.assertEqual(rc, 69, word)
            self.assertIn("다시 읽으니 다르다", err)
            self.assertIn(word, err)
            self.assertEqual((self.state / "templates.json").read_text(), before)    # 상태(지금 템플릿) 그대로

    def test_publish_base_and_cf_checks(self):
        ps = self.labs_build("v12")
        self.fake.cf_text = "SPEC=mtp\n"                                             # CF 에 CTX= 줄 없음
        rc, out, err = self.pub("publish", "v12", self.commit, ps)
        self.assertEqual(rc, 69)
        self.assertIn("CTX=", err)
        self.assertEqual((self.fake.tpl_posts, [c for c in self.fake.gh_calls if c["method"] == "PATCH"]), ([], []))
        self.fake.cf_text = "CTX=long\n"
        (self.labs / "v12" / "image.txt").write_text("ghcr.io/other@sha256:" + "0" * 64)
        rc, out, err = self.pub("publish", "v12", self.commit, ps)
        self.assertEqual(rc, 69)
        self.assertIn("image.txt", err)
        (self.labs / "v12" / "image.txt").unlink()
        self.fake.gh_calls.clear()
        self.fake.templates[HASH] = {**self.fake.templates[HASH], "args_str": "bash -c 'other'"}   # 바탕의 Args 만 다름
        rc, out, err = self.pub("publish", "v12", self.commit, ps)
        self.assertEqual(rc, 0, err)                                                   # Args 는 베끼지 않으므로 계속
        self.assertIn("Args 가 기록과 다르다", out)
        self.assertEqual(self.fake.tpl_posts[-1]["args_str"], (self.labs / "v12" / "args.txt").read_text())
        st = json.loads((self.state / "templates.json").read_text())
        self.fake.templates[HASH] = {**self.fake.templates[HASH], "args_str": self.fake.templates[HASH]["args_str"],
                                     "env": self.fake.templates[HASH]["env"].replace("-e PS=", "-e PS=0")}
        st["current"]["pod"] = HASH                                                    # 바탕의 PS 가 다르면 여전히 멈춘다
        (self.state / "templates.json").write_text(json.dumps(st))
        self.fake.gh_calls.clear()
        rc, out, err = self.pub("publish", "v12", self.commit, ps)
        self.assertEqual(rc, 69)
        self.assertIn("바탕 템플릿", err)
        self.assertEqual(self.fake.gh_calls, [])                                       # gist 도 건드리지 않음

    def test_publish_token_never_printed(self):
        ps = self.labs_build("v12")
        del self.fake.gist[GID]                                                       # 404 응답 본문에 토큰이 섞여 온다
        rc, out, err = self.pub("publish", "v12", self.commit, ps)                    # run_main 이 토큰이 없는지 본다
        self.assertEqual(rc, 70)
        self.assertIn("HTTP 404", err)
        self.assertIn("<가림>", err)

    # ── up 확인 ──
    def test_up_refuses_unrecorded_template(self):
        for change in ({"env": self.fake.templates[HASH]["env"].replace("-e PS=" + "1" * 64, "-e PS=" + "9" * 64)},
                       {"env": self.fake.templates[HASH]["env"] + " -e SB=x"},
                       {"args_str": "bash -c 'evil'"}, {"image": "ghcr.io/evil@sha256:" + "0" * 64},
                       {"env": self.fake.templates[HASH]["env"].replace("https://gist.githubusercontent.com", "https://evil.example")}):
            keep = dict(self.fake.templates[HASH])
            self.fake.templates[HASH].update(change)
            self.fake.calls.clear()
            rc, out, err = self.run_main("up", "102", "0.36")
            self.assertEqual(rc, 69, change)
            self.assertIn("publish 기록과 다르다", err)
            self.assertFalse([c for c in self.fake.calls if c["method"] in ("PUT", "DELETE")])   # 지우지도 빌리지도 않는다
            self.fake.templates[HASH] = keep
        del self.fake.templates[HASH]                                                   # API 가 못 찾음
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 69)

    def test_up_uses_template_key_if_present(self):
        tk = "t" * 40
        rc, out, err = self.run_main("up", "102", "0.36", tkey_fn=lambda: tk)
        self.assertEqual(rc, 0, err)
        tget = [c for c in self.fake.calls if "/v0/template/" in c["url"]]
        self.assertEqual({c["headers"].get("Authorization") for c in tget}, {f"Bearer {tk}"})
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertEqual(put["headers"].get("Authorization"), f"Bearer {KEY}")       # 빌리기는 원래 키
        self.assertNotIn(tk, out + err)

    def test_templates_command(self):
        rc, out, err = self.run_main("templates")
        self.assertEqual(rc, 0, err)
        rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
        self.assertEqual({r["template_hash_id"]: r["ok"] for r in rows}, {HASH: True, QHASH: True})
        self.fake.templates[QHASH]["args_str"] = "x"
        rc, out, err = self.run_main("templates")
        self.assertEqual(rc, 69)
        self.assertIn("up 이 거부된다", out)

    def test_state_dir_checks(self):
        os.chmod(self.state, 0o755)
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 78)                  # 다른 사람이 읽을 수 있는 폴더
        os.chmod(self.state, 0o700)
        link = self.tmp / "state-link"
        os.symlink(self.state, link)
        with self.assertRaises(SC.Stop):
            SC.load_state(str(link))                                                   # 링크 거부
        with self.assertRaises(SC.Stop):
            SC.state_check_dir(str(self.state), True)                                  # 부모가 root 가 아님(설치본 조건)
        self.assertEqual(SC.STATE_DIR, SC.PREFIX + "/state")


class T28(Base):
    """T28: quant 만 datacenter 조건 끄기, status 의 차선 failed 줄."""

    def test_quant_datacenter_off(self):
        cfg = SC.load_config(str(self.cfgp))
        self.assertFalse(SC.want_datacenter(cfg, "quant"))                     # 견본: roles.quant.datacenter false
        self.assertTrue(all(SC.want_datacenter(cfg, r) for r in ("serve", "eval", "eval2")))
        self.assertNotIn("datacenter", SC.offer_query(cfg, "quant"))
        self.assertEqual(SC.offer_query(cfg, "eval")["datacenter"], {"eq": True})
        self.assertEqual(SC.offer_query(cfg, "quant")["verified"], {"eq": True})          # verified·신뢰도는 그대로
        self.assertEqual(SC.offer_query(cfg, "quant")["reliability"], {"gte": 0.95})
        big = dict(cpu_ram=257000, cpu_cores_effective=32, disk_space=500, inet_down=2500, gpu_ram=81920)
        home = offer(950, 1.1, 8050, gpu_name="A100 SXM4", hosting_type=0, **big)
        self.assertEqual(SC.offer_fails(cfg, home, "quant"), [])
        self.assertEqual(SC.offer_fails(cfg, dict(home, verification="unverified"), "quant"), ["verified"])
        self.assertEqual(SC.offer_fails(cfg, dict(home, reliability=0.9), "quant"), ["reliability"])
        self.assertEqual(SC.offer_fails(cfg, dict(home, gpu_name="RTX 3090", num_gpus=1, dph_total=0.4), "serve"), ["datacenter"])
        self.assertIn("datacenter 조건 없음", SC.cond_view(cfg, "quant"))
        c = json.loads(self.cfgp.read_text())
        del c["roles"]["quant"]["datacenter"]                                      # 칸이 없으면 맨 위 값(true)
        self.cfgp.write_text(json.dumps(c))
        cfg2 = SC.load_config(str(self.cfgp))
        self.assertTrue(SC.want_datacenter(cfg2, "quant"))
        self.assertEqual(SC.offer_fails(cfg2, home, "quant"), ["datacenter"])
        c["roles"]["quant"]["datacenter"] = "no"
        self.cfgp.write_text(json.dumps(c))
        with self.assertRaises(SC.Stop):
            SC.load_config(str(self.cfgp))

    def test_quant_up_non_datacenter(self):
        big = dict(cpu_ram=257000, cpu_cores_effective=32, disk_space=500, inet_down=2500, gpu_ram=81920)
        self.fake.offers.append(offer(950, 1.1, 8050, gpu_name="A100 SXM4", hosting_type=0, **big))
        rc, out, err = self.run_main("offers", "quant")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("datacenter", self.fake.bundle_bodies[-1])
        self.assertEqual(self.run_main("up", "quant", "950", "1.105")[0], 0)

    def test_status_lane_failed_lines(self):
        self.names["present"] = True
        why = "기동 30분 넘음(health 없음) — 로그 끝: INFO Waiting for core engine ⏎ token=" + "s" * 30
        self.fake.pod_status_eval = {"instance_id": "5556", "pod_log": ["t\tlane 1 failed", "t\tHALT: 모든 차선 실패(…) — 900초 뒤 정지"],
                                     "lanes": [{"k": 0, "state": "ready", "alias": "a", "why": None},
                                               {"k": 1, "state": "failed", "alias": "qwen38", "why": why}]}
        rc, out, err = self.run_main("status")
        self.assertEqual(rc, 0, err)
        self.assertIn("[eval] 차선 1 failed(qwen38): 기동 30분 넘음(health 없음) — 로그 끝: INFO Waiting for core engine", out)
        self.assertNotIn("s" * 30, out)                                             # 비밀 꼴 가림
        self.assertIn("[eval] 파드 t\tHALT: 모든 차선 실패", out)
        self.assertNotIn("차선 0 failed", out)
        self.assertEqual(SC.lane_fail_lines(None), [])
        self.assertEqual(SC.lane_fail_lines({"lanes": "x", "pod_log": None}), [])


class ArgsShape(Base):
    """콘솔로 만든 템플릿의 Args 표기 차이(실행 인자 목록이 같으면 같다)와 차이 꼴 진단. publish 기록은 바이트 그대로."""

    def setUp(self):
        super().setUp()
        sb = base64.b64encode(b"#!/bin/bash\necho boot\n").decode()
        self.want = SC.expected_args(sb)
        self.inner = self.want[len("bash -c '"):-1]
        self.fake.templates[HASH] = tpl(HASH, "1" * 64, sb, self.want, IMG)
        seed_state(self.state, [("pod", "v11", self.fake.templates[HASH]), ("quant", "quant-v3", self.fake.templates[QHASH])])

    def with_args(self, a):
        self.fake.templates[HASH] = {**self.fake.templates[HASH], "args_str": a}

    def test_equivalent_forms_pass_for_console(self):
        for a in (self.want + "\n", "  " + self.want + " \r\n", "\n" + self.want, "bash  -c\t'" + self.inner + "'",
                  json.dumps(["bash", "-c", self.inner]), ["bash", "-c", self.inner]):
            self.with_args(a)
            self.assertEqual(SC.args_canonical(a), self.want, repr(a)[:60])
            self.assertEqual(SC.template_mismatch(json.loads((self.state / "templates.json").read_text())["templates"][HASH],
                                                  self.fake.templates[HASH]), [], repr(a)[:60])
        self.with_args(self.want + "\n")
        self.assertEqual(self.run_main("up", "102", "0.36")[0], 0)            # 콘솔 템플릿은 표기 차이로 막히지 않는다

    def test_meaning_changes_fail(self):
        rec = json.loads((self.state / "templates.json").read_text())["templates"][HASH]
        bad = [self.want.replace("sleep 1d", "sleep 2d"), self.want.replace("&&exec", "&& exec"),
               self.want.replace('"$SB"', '\\"$SB\\"'),                     # 작은따옴표 안 역빗금은 글자 그대로라 뜻이 다르다
               self.want + " x", "bash -c '" + self.inner + "' extra", "sh -c '" + self.inner + "'",
               "bash -c '" + self.inner + "\r'", None, 42, "", "bash -c 'unterminated",
               "bash -c \"" + self.inner.replace('"', '\\"') + "\"",                   # 큰따옴표 꼴은 받지 않는다($SB 가 바깥에서 펼쳐질 수 있다)
               ["bash", "-c", self.inner, "x"], json.dumps(["bash", "-c", self.inner + " "])]
        for a in bad:
            self.with_args(a)
            self.assertEqual(SC.template_mismatch(rec, self.fake.templates[HASH]), ["args_sha"], repr(a)[:60])
        self.with_args(self.want + "\n")
        self.assertEqual(SC.template_mismatch(dict(rec, source="publish"), self.fake.templates[HASH]), ["args_sha"])   # publish 는 바이트
        self.with_args(self.want.replace("sleep 1d", "sleep 2d"))
        rc, out, err = self.run_main("up", "102", "0.36")
        self.assertEqual(rc, 69)
        self.assertFalse([c for c in self.fake.calls if c["method"] in ("PUT", "DELETE")])

    def test_templates_shows_shape_not_content(self):
        self.with_args(self.want + "\r\n")
        rc, out, err = self.run_main("templates")
        self.assertEqual(rc, 0, err)
        row = [json.loads(l) for l in out.splitlines() if l.startswith("{") and HASH in l][0]
        self.assertTrue(row["ok"])
        d = row["args_diff"]
        self.assertEqual((d["api_type"], d["api_len"], d["want_len"], d["same_argv"], d["canonical_equal"]),
                         ("str", len(self.want) + 2, len(self.want), True, True))
        self.assertEqual(d["diffs"], [{"op": "더해짐", "where": "끝", "removed": "없음", "added": "CR1+LF1"}])
        for leak in ("/dev/shm/b", "base64", "sha256sum", "HALT: boot", self.inner[:20], "gist."):
            self.assertNotIn(leak, out + err)                                  # Args 본문·주소는 보이지 않는다
        self.with_args(self.want.replace('"$SB"', '\\"$SB\\"'))
        rc, out, err = self.run_main("templates")
        self.assertEqual(rc, 69)
        d = [json.loads(l) for l in out.splitlines() if l.startswith("{") and HASH in l][0]["args_diff"]
        self.assertEqual((d["same_argv"], d["canonical_equal"]), (False, False))
        self.assertEqual([x["added"] for x in d["diffs"]], ["역빗금1", "역빗금1"])
        self.with_args(["bash", "-c", self.inner])                            # 목록 꼴
        rc, out, err = self.run_main("templates")
        d = [json.loads(l) for l in out.splitlines() if l.startswith("{") and HASH in l][0]["args_diff"]
        self.assertEqual((rc, d["api_type"], d["same_argv"]), (0, "list", True))

    def test_expected_args_matches_builder(self):
        mte = Path(SC.LABS_VAST)
        for b in ("v10", "v11", "v12", "quant-v3"):
            if (mte / b / "args.txt").exists():
                self.assertEqual(SC.expected_args((mte / b / "boot.sb.txt").read_text().strip()), (mte / b / "args.txt").read_text(), b)
        self.assertIsNone(SC.expected_args("not base64!"))
        self.assertIsNone(SC.expected_args(None))



class TBalance(Base):
    """balance: 잔액 두 칸만 보이고 개인 칸(api_key·이메일·이름·주소)은 어디에도 없다. 켜진 $/hr 합과 바닥까지 어림."""

    def setUp(self):
        super().setUp()
        self.fake.instances += [
            {"id": 6001, "label": "stewart-quant-20261110-0906", "machine_id": 1, "actual_status": "running", "dph_total": 1.5},
            {"id": 6002, "label": "stewart-eval-20261110-0624", "machine_id": 2, "actual_status": "exited",
             "intended_status": "stopped", "dph_total": 2.0},
            {"id": 6003, "label": "other-job", "machine_id": 3, "actual_status": "running", "dph_total": 0.5}]

    def test_balance_fields_only(self):
        rc, out, err = self.run_main("balance", tkey_fn=lambda: "t" * 64)
        self.assertEqual(rc, 0, err)
        j = [json.loads(l) for l in out.splitlines() if l.startswith("{")][0]
        self.assertEqual((j["credit"], j["balance"]), (41.2345, -0.5))
        self.assertEqual((j["running"], j["stopped"], j["running_dph"]), (3, 1, 2.0))     # 이름표와 상관없이 켜진 것 전부(777 은 값 없음)
        self.assertEqual(j["hours_left_at_running_dph"], round(41.2345 / 2.0, 1))
        self.assertEqual({m["id"] for m in j["mine"]}, {6001, 6002})
        for leak in ("k" * 64, "who@example.invalid", "Some Body", "1 Road", "api_key", "email", "fullname"):
            self.assertNotIn(leak, out + err)
        self.assertFalse(list(self.rec.glob("*")) if self.rec.exists() else [])          # 기록 파일을 쓰지 않는다
        call = [c for c in self.fake.calls if c["url"].endswith("/v0/users/current")][0]
        self.assertEqual(call["auth"], "Bearer " + "t" * 64)                                  # user_read 가 있는 템플릿 키로

    def test_balance_missing_fields_and_no_running(self):
        self.fake.user = {"email": "who@example.invalid", "credit": "12"}                    # 숫자가 아니면 None
        self.fake.instances = [i for i in self.fake.instances if i["id"] == 6002]
        rc, out, err = self.run_main("balance")
        self.assertEqual(rc, 0, err)
        j = [json.loads(l) for l in out.splitlines() if l.startswith("{")][0]
        self.assertEqual((j["credit"], j["balance"], j["running_dph"], j["hours_left_at_running_dph"]), (None, None, 0.0, None))
        self.assertNotIn("who@example.invalid", out + err)

    def test_balance_403_and_forms(self):
        self.fake.fail[("GET", SC.API + "/v0/users/current")] = 403
        rc, out, err = self.run_main("balance")
        self.assertEqual(rc, SC.EXIT_AUTH)
        self.assertNotIn(KEY, out + err)
        with self.assertRaises(SC.Stop) as cm:
            SC.parse_args(["balance", "x"])
        self.assertEqual(cm.exception.code, 64)
        r = subprocess.run(["/bin/sh", str(HERE / "stewart-cloud"), "balance", "x"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 64)

class Retemplate(Base):
    """T30-1R: 정지한 stewart- 인스턴스를 그 역할의 지금 템플릿으로 다시 만들고 켠다."""

    def stopped_quant(self, iid=7101, label="stewart-quant-20261110-0906"):
        self.fake.instances.append({"id": iid, "label": label, "machine_id": 9101, "actual_status": "exited",
                                    "cur_state": "stopped", "intended_status": "stopped", "template_hash_id": "old"})

    def test_retemplate_stopped_quant(self):
        self.stopped_quant()
        rc, out, err = self.run_main("retemplate", "7101")
        self.assertEqual(rc, 0, err)
        puts = [c for c in self.fake.calls if c["method"] == "PUT"]
        self.assertEqual(len(puts), 2)
        self.assertTrue(puts[0]["url"].endswith("/api/v0/instances/update_template/7101/"))
        self.assertEqual(puts[0]["body"], {"id": 7101, "template_hash_id": QHASH})       # 그 역할(quant)의 지금 템플릿
        self.assertTrue(puts[1]["url"].endswith("/api/v0/instances/7101/"))
        self.assertEqual(puts[1]["body"], {"state": "running"})
        self.assertIn("id 그대로", out)
        self.assertIn("정지해 있어 켰다", out)
        self.assertNotIn("k" * 64, out + err)                                # 응답의 허용 밖 칸(api_key 등)은 찍지 않는다
        ev = json.loads(out.strip().splitlines()[-1])
        self.assertEqual((ev["event"], ev["role"], ev["now_id"], ev["started"]), ("retemplate", "quant", 7101, True))
        self.assertEqual(ev["after"]["actual_status"], "running")
        self.assertIn('"event": "retemplate"', (self.rec / "history.jsonl").read_text())

    def test_retemplate_reports_new_id(self):
        self.stopped_quant()
        self.fake.retpl_new_id = 7103
        rc, out, err = self.run_main("retemplate", "7101")
        self.assertEqual(rc, 0, err)
        self.assertIn("id 가 바뀌었다(7101 → 7103)", out)
        put = [c for c in self.fake.calls if c["method"] == "PUT"][-1]
        self.assertTrue(put["url"].endswith("/api/v0/instances/7103/"))

    def test_retemplate_refuses_others(self):
        rc, out, err = self.run_main("retemplate", "777")                    # 남의 이름표
        self.assertEqual(rc, SC.EXIT_REFUSE)
        self.stopped_quant(iid=888, label="stewart-handmade")              # 역할을 모르는 stewart- 이름표
        rc, out, err = self.run_main("retemplate", "888")
        self.assertEqual(rc, SC.EXIT_REFUSE)
        self.assertIn("역할을 모른다", err)
        self.assertFalse([c for c in self.fake.calls if c["method"] == "PUT"])

    def test_retemplate_refuses_changed_template(self):
        self.stopped_quant()
        self.fake.templates[QHASH] = {**self.fake.templates[QHASH], "args_str": "bash -c 'other'"}
        rc, out, err = self.run_main("retemplate", "7101")
        self.assertEqual(rc, SC.EXIT_REFUSE, err)
        self.assertFalse([c for c in self.fake.calls if c["method"] == "PUT"])

    def test_retemplate_args(self):
        for bad in ((), ("x",), ("1", "2")):
            rc, out, err = self.run_main("retemplate", *bad)
            self.assertEqual(rc, SC.EXIT_USAGE)

    def test_instance_put_only_running(self):
        for bad in ({"state": "destroyed"}, {"state": "running", "label": "x"}, None):
            with self.assertRaises(SC.Stop):
                SC.vast(KEY, "PUT", "/v0/instances/5/", bad)
        with self.assertRaises(SC.Stop):
            SC.vast(KEY, "PUT", "/v0/instances/5/", {"label": "x"})


class KeepCopy(Base):
    """T32: 정지 인스턴스를 지우지 않고 하나 더(up-keep)·인스턴스 → 인스턴스 복사(copy)·켜기(start)·정지/대기 취소(stop).
    c 길을 처음부터 끝까지 main 으로 탄다(Vast API 는 FakeVast)."""
    QHASH = QHASH

    def setUp(self):
        super().setUp()
        cfg = json.loads(self.cfgp.read_text())
        cfg["roles"]["quant"]["template_hash_id"] = QHASH
        self.cfgp.write_text(json.dumps(cfg))
        Flows.add_quant_offers(self)
        for iid, label, m in ((7101, "stewart-quant-20261110-0906", 9101), (7102, "stewart-quant-import-20261110", 9102)):
            self.fake.instances.append({"id": iid, "label": label, "machine_id": m, "actual_status": "exited", "cur_state": "stopped",
                                        "intended_status": "stopped", "dph_total": 1.68, "start_date": 1_700_000_000.0})
        self.fake.queue_ids = {7101}

    def test_tonight_c_path_end_to_end(self):
        # 1) 7101 켜기 → 대기열(실제 응답 꼴) → 대기 취소
        rc, out, err = self.run_main("start", "7101")
        self.assertEqual(rc, 0, err)
        self.assertIn("대기열에 들어갔다", out)
        self.assertTrue(json.loads(out.strip().splitlines()[-1])["queued"])
        rc, out, err = self.run_main("stop", "7101")
        self.assertEqual(rc, 0, err)
        self.assertEqual([c["body"] for c in self.fake.calls if c["method"] == "PUT"][-1], {"state": "stopped"})
        # 2) up-keep: 정지한 quant 둘은 그대로, 새 quant 하나(디스크 300)
        rc, out, err = self.run_main("up-keep", "quant", "902", "1.055", "300")
        self.assertEqual(rc, 0, err)
        self.assertIn("지우지 않고 둔다(up-keep)", out)
        self.assertFalse([c for c in self.fake.calls if c["method"] == "DELETE"])
        put = [c for c in self.fake.calls if c["method"] == "PUT" and "/asks/" in c["url"]][-1]
        self.assertEqual((put["body"]["disk"], put["body"]["template_hash_id"]), (300, QHASH))
        new = json.loads(out.strip().splitlines()[-1])
        self.assertTrue(new["label"].startswith("stewart-quant-"))
        self.assertEqual(sorted(i["id"] for i in self.fake.instances if str(i["label"]).startswith("stewart-quant")),
                         sorted([7101, 7102, new["instance_id"]]))
        # 3) copy: 7102 /workspace/quant/ → 새 인스턴스 /workspace/import/
        rc, out, err = self.run_main("copy", "7102:/workspace/quant/", f"{new['instance_id']}:/workspace/import/")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.fake.copies, [{"client_id": "me", "src_id": "7102", "dst_id": str(new["instance_id"]),
                                            "src_path": "/workspace/quant/", "dst_path": "/workspace/import/"}])
        ev = [json.loads(l) for l in (self.rec / "history.jsonl").read_text().splitlines()]
        self.assertEqual([e["event"] for e in ev], ["start", "stop", "up-keep", "copy"])

    def test_up_keep_refuses_running_same_role(self):
        self.fake.instances[-1].update(actual_status="running", intended_status="running")
        rc, out, err = self.run_main("up-keep", "quant", "902", "1.055", "300")
        self.assertEqual(rc, SC.EXIT_REFUSE)
        self.assertIn("켜진 quant 인스턴스가 있다", err)

    def test_up_keep_args(self):
        for bad in (("quant", "902", "1.055"), ("quant", "902", "1.055", "100"), ("quant", "902", "9.999", "300"),
                    ("nope", "902", "1.055", "300"), ("quant", "902", "1.055", "99999")):
            self.assertEqual(self.run_main("up-keep", *bad)[0], SC.EXIT_USAGE, bad)

    def test_up_keep_counts_total(self):
        self.fake.instances += [{"id": 70 + k, "label": f"stewart-eval-{k}", "machine_id": 1, "actual_status": "exited",
                                 "intended_status": "stopped"} for k in range(2)]
        rc, out, err = self.run_main("up-keep", "quant", "902", "1.055", "300")
        self.assertEqual(rc, SC.EXIT_REFUSE)                               # 정지 넷 = 합쳐 최대 4
        self.assertIn("합쳐 최대", err)

    def test_copy_refuses(self):
        for a, b in (("777:/x/", "7102:/y/"),                         # 남의 이름표
                     ("7102:/workspace/quant/", "7102:/workspace/import/")):   # 같은 인스턴스
            rc, out, err = self.run_main("copy", a, b)
            self.assertIn(rc, (SC.EXIT_REFUSE, SC.EXIT_USAGE), (a, b, err))
        for bad in (("7102:/workspace/../etc/", "7101:/x/"), ("7102:workspace/", "7101:/x/"),
                    ("7102:/a b/", "7101:/x/"), ("7102", "7101:/x/")):
            self.assertEqual(self.run_main("copy", *bad)[0], SC.EXIT_USAGE, bad)
        self.assertEqual(self.fake.copies, [])

    def test_start_stop_only_ours(self):
        self.assertEqual(self.run_main("start", "777")[0], SC.EXIT_REFUSE)
        self.assertEqual(self.run_main("stop", "777")[0], SC.EXIT_REFUSE)
        self.assertFalse([c for c in self.fake.calls if c["method"] == "PUT"])

    def test_start_runs_when_free(self):
        rc, out, err = self.run_main("start", "7102")
        self.assertEqual(rc, 0, err)
        ev = json.loads(out.strip().splitlines()[-1])
        self.assertEqual((ev["queued"], ev["after"]["actual_status"]), (False, "running"))


class T36(Base):
    """T36: roles.eval.nondc_reliability_min(선택 칸, 견본에는 없다)을 0.98 로 두면 eval·eval2 는 datacenter 면 신뢰도 ≥0.95,
    아니면 verified·신뢰도 ≥0.98. serve·quant 그대로."""
    R = dict(disk_space=500, cpu_ram=257000, num_gpus=4)

    def setUp(self):
        super().setUp()
        c = json.loads(self.cfgp.read_text())
        c["roles"]["eval"]["nondc_reliability_min"] = 0.98
        self.cfgp.write_text(json.dumps(c))

    def eval_offers(self):
        r = self.R
        self.fake.offers += [offer(601, 1.30, 6001, reliability=0.96, **r),                        # datacenter, 0.96 → 됨
                             offer(602, 0.88, 6002, reliability=0.995, hosting_type=0, geolocation="TW", **r),   # 집 기계 0.995 → 됨
                             offer(603, 0.80, 6003, reliability=0.97, hosting_type=0, **r),        # 집 기계 0.97 → 안 됨
                             offer(604, 0.70, 6004, reliability=0.999, hosting_type=0, verification="unverified", **r)]

    def test_config_and_judgement(self):
        cfg = SC.load_config(str(self.cfgp))
        self.assertTrue(SC.eval_relaxed(cfg, "eval") and SC.eval_relaxed(cfg, "eval2"))       # eval2 는 eval 것을 따른다
        self.assertFalse(SC.eval_relaxed(cfg, "serve") or SC.eval_relaxed(cfg, "quant"))
        home = offer(1, 0.9, 1, reliability=0.99, hosting_type=0, **self.R)
        self.assertEqual(SC.offer_fails(cfg, home, "eval"), [])
        self.assertEqual(SC.offer_fails(cfg, dict(home, reliability=0.97), "eval"), ["reliability_nondc"])
        self.assertEqual(SC.offer_fails(cfg, dict(home, reliability=0.94), "eval"), ["reliability", "reliability_nondc"])
        self.assertEqual(SC.offer_fails(cfg, dict(home, verification=None), "eval"), ["verified_nondc"])   # 칸 없음도 안 됨
        self.assertIn("verified_nondc", SC.offer_fails(cfg, dict(home, verification="deverified"), "eval2"))
        dc = dict(home, hosting_type=1, reliability=0.96)
        self.assertEqual(SC.offer_fails(cfg, dc, "eval"), [])
        self.assertEqual(SC.offer_fails(cfg, dict(dc, dc_search=False), "eval"), ["reliability_nondc"])  # 서버 datacenter 검색에 없으면 아님
        self.assertEqual(SC.offer_fails(cfg, dict(home, num_gpus=1, dph_total=0.4), "serve"), ["datacenter"])   # serve 그대로
        self.assertNotIn("datacenter", SC.offer_query(cfg, "eval"))
        self.assertEqual(SC.offer_query(cfg, "eval")["reliability"], {"gte": 0.95})
        self.assertIn("datacenter 아니면 verified·신뢰도 ≥0.98", SC.cond_view(cfg, "eval2"))
        self.assertIn(", datacenter", SC.cond_view(cfg, "serve"))
        c = json.loads(self.cfgp.read_text())
        for bad in (0.9, 1.5, True, "0.98"):
            c["roles"]["eval"]["nondc_reliability_min"] = bad
            self.cfgp.write_text(json.dumps(c))
            with self.assertRaises(SC.Stop):
                SC.load_config(str(self.cfgp))
        del c["roles"]["eval"]["nondc_reliability_min"]                                 # 칸이 없으면 옛 조건(datacenter 필수)
        self.cfgp.write_text(json.dumps(c))
        cfg2 = SC.load_config(str(self.cfgp))
        self.assertFalse(SC.eval_relaxed(cfg2, "eval"))
        self.assertEqual(SC.offer_fails(cfg2, home, "eval"), ["datacenter"])
        c["roles"]["eval2"]["nondc_reliability_min"] = 0.98                             # eval2 에 따로 두지 않는다
        self.cfgp.write_text(json.dumps(c))
        with self.assertRaises(SC.Stop):
            SC.load_config(str(self.cfgp))

    def test_offers_two_searches_and_fields(self):
        self.eval_offers()
        rc, out, err = self.run_main("offers", "eval", "4")
        self.assertEqual(rc, 0, out + err)
        rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
        self.assertEqual([r["id"] for r in rows], [602, 601])
        self.assertEqual([(r["hosting_type"], r["is_datacenter"]) for r in rows], [(0, False), (1, True)])
        self.assertEqual(rows[0]["reliability"], 0.995)
        bodies = self.fake.bundle_bodies[-2:]
        self.assertNotIn("datacenter", bodies[0])
        self.assertEqual(bodies[1]["datacenter"], {"eq": True})
        self.assertIn("datacenter 아니면 verified·신뢰도 ≥0.98", out)
        self.assertIn("stewart-cloud up eval 602 0.885 <계획>", out)

    def test_up_non_datacenter_recorded(self):
        self.eval_offers()
        rc, out, err = self.run_main("up", "eval", "602", "0.885", "a-all")
        self.assertEqual(rc, 0, out + err)
        cur = json.loads((self.rec / "current-eval.json").read_text())
        self.assertEqual((cur["machine_id"], cur["hosting_type"], cur["is_datacenter"], cur["reliability"], cur["verification"],
                          cur["geolocation"], cur["gpu_name"]), (6002, 0, False, 0.995, "verified", "TW", "RTX 3090"))
        ev = [json.loads(l) for l in (self.rec / "history.jsonl").read_text().splitlines()][-1]
        self.assertEqual((ev["event"], ev["hosting_type"], ev["is_datacenter"]), ("up", 0, False))
        self.assertEqual(self.run_main("up", "eval2", "603", "0.805", "a-all")[0], 69)       # 집 기계 0.97 은 거부
        self.assertEqual(self.run_main("up", "eval2", "604", "0.705", "a-all")[0], 69)       # 집 기계 unverified 거부
        rc, out, err = self.run_main("up", "eval2", "601", "1.305", "a-all")                 # datacenter 0.96 은 된다
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(json.loads((self.rec / "current-eval2.json").read_text())["is_datacenter"])

    def test_serve_still_datacenter(self):
        self.fake.offers = [offer(701, 0.30, 7101, hosting_type=0, reliability=0.999)]
        rc, out, err = self.run_main("offers")
        self.assertEqual(rc, 69)
        self.assertEqual(self.fake.bundle_bodies[0]["datacenter"], {"eq": True})
        self.assertEqual(self.run_main("up", "701", "0.305")[0], 69)


if __name__ == "__main__":
    unittest.main()
