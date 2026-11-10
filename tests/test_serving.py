"""서빙 묶음: 모델 표(models.toml)·어댑터 고르기·vllm 인자·패치 적용기(patchset.py)·이미지 대조 줄.

실제 vLLM 0.30.0 원본으로 하는 시험은 STEWART_VLLM_SRC(원본 vllm 패키지 폴더)가 있을 때만 돈다.
"""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from tests import FIX_ROOT

ROOT = FIX_ROOT / "deploy" / "serving"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SV = _load("serving")
PS = _load("patchset")
A, B1 = "qwen38-27b-demo-w4a16", "qwen36-35b-a3b-demo-w4a16"
ORIGQ, HHQ, DEMO2 = "qwen38-27b-orig-w4a16q", "qwen38-27b-demo-w4a16q", "qwen38-27b-demo2-w4a16"

# HyperQwen 8cf642e single-user/start_qwen.sh 가 SPEC=mtp CTX=long PREFIX_CACHE=1 (FAST 와 무관)에서 vllm serve 에 넘기는 인자(손으로 푼 값)
HQ_ARGV_A = ["serve", "/m", "--served-model-name", "qwen3.8-27b", "--host", "127.0.0.1", "--port", "8000",
             "--gpu-memory-utilization", "0.93", "--max-model-len", "150000", "--max-num-seqs", "8", "--api-server-count", "1",
             "--language-model-only", "--kv-cache-dtype", "fp8", "--mamba-ssm-cache-dtype", "float16", "--async-scheduling",
             "--max-num-batched-tokens", "2048",
             "--speculative-config", '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}',
             "--compilation-config", '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}',
             "--reasoning-parser", "qwen3", "--enable-prompt-tokens-details", "--enable-auto-tool-choice",
             "--tool-call-parser", "qwen3_coder", "--prefix-cache-retention-interval", "None", "--enable-prefix-caching",
             "--mamba-cache-mode", "align", "--sse-keep-alive-interval", "30"]


class TTable(unittest.TestCase):
    def setUp(self):
        self.t = SV.load()

    def test_valid_and_two_models(self):
        self.assertEqual(set(self.t["models"]), {A, B1, ORIGQ, HHQ, DEMO2} | {f"{x}-{k}" for x in (ORIGQ, HHQ, DEMO2) for k in ("bf16h", "int8h")})
        self.assertEqual(self.t["default"], A)                                  # local 이 default 를 정한다
        self.assertEqual([x["alias"] for x in SV.quant_targets(self.t)], [ORIGQ, HHQ, DEMO2])

    # ---- 공개 표·local 표 ----
    def test_public_table_alone(self):
        """local 이 없어도 돈다: 공개 표 = 원본 W4A16q 하나(pending), 비공개 줄·보관 서버 주소 없음."""
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(ROOT, Path(d) / "s", ignore=shutil.ignore_patterns("models.local.toml", "__pycache__"))
            t = SV.load(Path(d) / "s" / "models.toml")
        self.assertEqual((t["default"], set(t["models"]), t.get("plans", {})), (ORIGQ, {ORIGQ}, {}))
        self.assertNotIn("server_ip", t["home"])
        with self.assertRaises(SV.TableError):
            SV.model(t, "default")                                              # 변환 전에는 고를 수 없다
        text = (ROOT / "models.toml").read_text(encoding="utf-8").lower()
        for word in ("demo", "edited", "demo2", "server_ip ="):
            self.assertNotIn(word, text, word)
        self.assertNotRegex(text, r"[0-9]+\uCC28|\uBA54\uC778 \uACB0\uC815|record[s]/")   # 운영 꼬리표(회차·결정·기록 경로)가 없다

    def test_merge_rules(self):
        pub, loc = tomllib.loads((ROOT / "models.toml").read_text()), tomllib.loads((ROOT / "models.local.toml").read_text())
        for sec, key in (("models", ORIGQ), ("home", "recv_port")):
            l2 = copy.deepcopy(loc)
            l2.setdefault(sec, {})[key] = copy.deepcopy(pub[sec][key])
            with self.assertRaises(SV.TableError):
                SV.merge(pub, l2)                                              # 겹치면 안 된다
        with self.assertRaises(SV.TableError):
            SV.merge(pub, dict(loc, schema=2))
        with self.assertRaises(SV.TableError):
            SV.merge(pub, dict(loc, extra=1))
        m = SV.merge(pub, loc)
        self.assertEqual((m["default"], m["home"]["recv_port"], m["home"]["server_ip"]), (A, 8090, "100.64.0.9"))

    def test_pending_and_quant_src(self):
        raw = SV.read_raw()
        for bad in ({"pending": "yes"}, {"files": [{"name": "config.json", "sha256": "a" * 64, "size": 1}]},
                    {"quant_src": {"repo": "x/y", "rev": "a" * 40, "pins": "pins/a.sha256", "pad_to": 100, "order": 1}},
                    {"quant_src": {"repo": "x/y", "rev": "a" * 39, "pins": "pins/a.sha256", "pad_to": 0, "order": 1}},
                    {"quant_src": {"repo": "x/y", "rev": "a" * 40, "pins": "../a", "pad_to": 0, "order": 1}},
                    {"quant_src": {"repo": "x/y", "rev": "a" * 40, "pins": "pins/a.sha256", "pad_to": 0, "order": 2}}):   # order 겹침
            t = copy.deepcopy(raw)
            t["models"][ORIGQ].update(bad)
            self.assertTrue(SV.validate(t, ROOT), bad)
        t = copy.deepcopy(raw)
        t["plans"]["a-x4"]["lanes"][1]["model"] = HHQ                          # pending 은 계획에 못 넣는다(장 수는 같음)
        self.assertTrue(any("pending" in e for e in SV.validate(t, ROOT)))
        t = copy.deepcopy(raw)
        t["quant"]["audit"] = [ORIGQ, A]                                        # A 는 변환 대상이 아님
        self.assertTrue(any("quant.audit" in e for e in SV.validate(t, ROOT)))
        t = copy.deepcopy(raw)
        del t["home"]["server_ip"]                                              # 고를 수 있는 home 별칭(…-bf16h·…-int8h)이 있으니
        self.assertTrue(any("home.server_ip" in e for e in SV.validate(t, ROOT)))   # 주소가 있어야 한다
        for pins in {x["pins"] for x in SV.quant_targets(raw)}:                 # 변환 pins 는 deploy/quant 에 있다
            lines = (ROOT.parent / "quant" / pins).read_text().split("\n")
            self.assertTrue(any(l.endswith("  model.safetensors.index.json") for l in lines), pins)

    def test_chat_template_pinned(self):
        """세 별칭(A·원본 q·demo q)과 C 가 같은 템플릿 파일 하나(sha c3cf9e34…)를 쓴다. 원본 11-07 판(generation 꼬리표)이 아니다."""
        raw = SV.read_raw()
        want = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
        for a in (A, ORIGQ, HHQ, DEMO2):
            self.assertEqual(raw["models"][a]["chat_template"], {"file": "templates/qwen38-27b-chat.jinja", "sha256": want}, a)
        body = (ROOT / "templates" / "qwen38-27b-chat.jinja").read_bytes()
        self.assertEqual(hashlib.sha256(body).hexdigest(), want)
        self.assertNotIn(b"generation %}", body)
        env = SV.select_env(self.t, A, "auto", "hyperqwen")
        self.assertEqual((env["M_CHAT_TPL"], env["M_CHAT_TPL_SHA"]), ("templates/qwen38-27b-chat.jinja", want))
        t = copy.deepcopy(raw)
        t["models"][A]["chat_template"] = {"file": "templates/qwen38-27b-chat.jinja", "sha256": "0" * 64}
        self.assertTrue(any("chat_template" in e for e in SV.validate(t, ROOT)))
        t["models"][A]["chat_template"] = {"file": "../models.toml", "sha256": want}
        self.assertTrue(any("chat_template" in e for e in SV.validate(t, ROOT)))
        sh = (ROOT / "serving.sh").read_text()
        self.assertIn('cp "$SV/$M_CHAT_TPL" "$MDIR/chat_template.jinja"', sh)
        self.assertLess(sh.index('cp "$SV/$M_CHAT_TPL"'), sh.index("for f in $M_CHAT_FIX"))     # 손질은 고정본 위에

    def test_lane_spec_and_fast(self):
        lanes = SV.plan_lanes(self.t, "a-spec-ab", "hyperqwen", 4)
        self.assertEqual([ln["env"].get("M_LANE_SPEC") for ln in lanes], [None, None, "off", "off"])
        self.assertEqual([ln["env"]["M_SPEC"] for ln in lanes], ["mtp", "mtp", "off", "off"])
        with tempfile.TemporaryDirectory() as d:
            for k in (0, 2):
                SV.write_env(lanes[k]["env"], Path(d) / f"l{k}.env")
            base = {"SPEC": "mtp", "FAST_VARIANT": "1", "STEWART_MODELS": "/m"}
            argv0, env0 = SV.serve_cmd(self.t, Path(d) / "l0.env", 1, 8010, base)
            argv2, env2 = SV.serve_cmd(self.t, Path(d) / "l2.env", 2, 8020, base)
        self.assertEqual((env0["SPEC"], env2["SPEC"]), ("mtp", "off"))
        self.assertEqual((env0["FAST_VARIANT"], env0["MODEL"].endswith("-fast")), ("1", True))   # A 는 fast 판이 있다
        t = copy.deepcopy(SV.read_raw())                                       # fast 판이 없는 별칭은 FAST_VARIANT=1 이어도 기본 판
        del t["models"][A]["fast"]
        with tempfile.TemporaryDirectory() as d:
            SV.write_env(SV.select_env(t, A, "hyperqwen-8cf642e", "hyperqwen"), Path(d) / "e")
            _, env = SV.serve_cmd(t, Path(d) / "e", 1, 8010, {"FAST_VARIANT": "1", "STEWART_MODELS": "/m"})
        self.assertEqual((env["FAST_VARIANT"], env["MODEL"]), ("0", "/m/" + t["models"][A]["dir"]))
        a = SV.vllm_argv(SV.model(self.t, A), "/m", 8000)
        self.assertIn("--speculative-config", a)
        self.assertNotIn("--speculative-config", SV.vllm_argv(SV.model(self.t, A), "/m", 8000, spec=False))
        t = copy.deepcopy(SV.read_raw())
        t["plans"]["a-spec-ab"]["lanes"][2]["spec"] = "dflash2"
        self.assertTrue(any("spec" in e for e in SV.validate(t, ROOT)))

    def test_validate_catches(self):
        raw = SV.read_raw()
        cases = [("rev", "c20530b"), ("repo", "evil"), ("dir", "../x"), ("adapters", ["podman"]), ("tp", 3),
                 ("served_name", "a b"), ("chat_template_fix", ["rm"]), ("pins", "pins/없음.sha256"), ("extra_key", 1)]
        for k, v in cases:
            t = copy.deepcopy(raw)
            t["models"][A][k] = v
            self.assertTrue(SV.validate(t, ROOT), k)
        t = copy.deepcopy(raw)
        t["models"][A]["tp"] = 2                      # hyperqwen 어댑터는 한 장만
        self.assertTrue(any("tp 1" in e for e in SV.validate(t, ROOT)))
        t = copy.deepcopy(raw)
        t["models"][A]["serve"]["max_model_len"] = "150000"
        self.assertTrue(any("int" in e for e in SV.validate(t, ROOT)))
        t = copy.deepcopy(raw)
        t["models"]["Bad_Name"] = t["models"].pop(B1)
        self.assertTrue(any("별칭" in e for e in SV.validate(t, ROOT)))

    def test_choose_adapter(self):
        a, b = SV.model(self.t, A), SV.model(self.t, B1)
        self.assertEqual(SV.choose_adapter(a, "auto", "hyperqwen"), "hyperqwen-8cf642e")
        self.assertEqual(SV.choose_adapter(a, "auto", "vllm-openai"), "vllm-patched")
        self.assertEqual(SV.choose_adapter(a, "vllm-patched", "hyperqwen"), "vllm-patched")   # 같은 이미지에서 비교
        self.assertEqual(SV.choose_adapter(b, "auto", "hyperqwen"), "vllm-patched")
        self.assertEqual(SV.choose_adapter(b, "vllm-stock", "vllm-openai"), "vllm-stock")
        for m, want, kind in ((a, "vllm-stock", "vllm-openai"), (b, "hyperqwen-8cf642e", "hyperqwen"),
                              (b, "vllm-stock", "hyperqwen"), (a, "hyperqwen-8cf642e", "vllm-openai"), (a, "x", "hyperqwen")):
            with self.assertRaises(SV.TableError):
                SV.choose_adapter(m, want, kind)
        with self.assertRaises(SV.TableError):
            SV.model(self.t, "없는-별칭")

    def test_argv_matches_hyperqwen(self):
        self.assertEqual(SV.vllm_argv(SV.model(self.t, A), "/m", 8000), HQ_ARGV_A)
        b = SV.vllm_argv(SV.model(self.t, B1), "/m", 8010)
        self.assertEqual(b[b.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(b[b.index("--served-model-name") + 1], "qwen3.6-35b-a3b")

    def test_pins(self):
        a = SV.pins(self.t, A).splitlines()
        self.assertEqual(len(a), 20)
        self.assertEqual(len(SV.pins(self.t, A, fast=True).splitlines()), 5)
        b = [ln.split("  ", 1)[1] for ln in SV.pins(self.t, B1).splitlines()]
        for f in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model_mtp.safetensors",
                  "model.safetensors.index.json", "config.json", "tokenizer.json", "chat_template.jinja"):
            self.assertIn(f, b)
        self.assertEqual(len(b), 14)                                         # 저장소 파일 전부

    def test_select_env_and_serve_cmd(self):
        d = Path(tempfile.mkdtemp())
        e = SV.select_env(self.t, "default", "auto", "hyperqwen")
        self.assertEqual((e["M_ALIAS"], e["M_ADAPTER"], e["M_HQ_CTX"]), (A, "hyperqwen-8cf642e", "long"))
        SV.write_env(e, d / "model.env")
        self.assertEqual(SV.read_env(d / "model.env"), e)
        base = {"STEWART_MODELS": "/w/models", "PATH": "/bin", "FAST_VARIANT": "1", "CTX": "long", "SPEC": "mtp"}
        argv, env = SV.serve_cmd(self.t, d / "model.env", 0, 8000, base)
        self.assertEqual(argv, ["bash", "/app/docker/entrypoint.sh", "single"])
        self.assertEqual((env["MODEL"], env["BASE_MODEL_DIR"], env["HOST"], env["PORT"]),
                         (f"/w/models/{self.t['models'][A]['dir']}-fast", f"/w/models/{self.t['models'][A]['dir']}", "127.0.0.1", "8000"))
        argv, env = SV.serve_cmd(self.t, d / "model.env", 1, 8010, base | {"FAST_VARIANT": "0"})
        self.assertEqual((argv[-1], env["PREPARE"], env["VERIFY"], env["PORT"]), ("/app/single-user/start_qwen.sh", "0", "0", "8010"))
        self.assertFalse(env["MODEL"].endswith("-fast"))
        # vllm-patched: overlay 가 있어야 하고, PYTHONPATH 맨 앞이 overlay
        SV.write_env(SV.select_env(self.t, B1, "auto", "vllm-openai"), d / "b1.env")
        with self.assertRaises(SV.TableError):
            SV.serve_cmd(self.t, d / "b1.env", 0, 8000, base)
        ov = d / "ov"
        (ov / "vllm").mkdir(parents=True)
        (ov / "stewart-overlay.json").write_text("{}")
        argv, env = SV.serve_cmd(self.t, d / "b1.env", 0, 8000, base | {"STEWART_OVERLAY": str(ov), "PYTHONPATH": "/x"})
        self.assertEqual(env["PYTHONPATH"], f"{ov}:/x")
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:False")    # tp>1(HyperQwen #163)
        self.assertEqual((env["VLLM_USE_FLASHINFER_SAMPLER"], argv[1]), ("0", "-c"))
        self.assertIn("--tensor-parallel-size", argv)

    def test_prepared_int8h_mixed_plan(self):
        """T35: prepared(int8h) 별칭은 어느 차선에서나 hyperqwen — prepare·verify 를 다시 돌리지 않는다(start_qwen.sh PREPARE=0)."""
        oi, hi = ORIGQ + "-int8h", HHQ + "-int8h"
        self.assertTrue(self.t["models"][oi].get("prepared") and self.t["models"][hi].get("prepared"))
        self.assertEqual(SV.validate(SV.read_raw(), ROOT), [])
        for name, want in (("x1-01-eval", [A, oi, hi, oi]), ("x1-01-eval2", [oi, hi, hi, oi])):
            self.assertEqual([x["alias"] for x in SV.plan_lanes(self.t, name, "hyperqwen", 4)], want)
            self.assertEqual(self.t["plans"][name]["env"]["FAST_VARIANT"], "0")   # A 도 int8h 와 같은 머리 처리
        raw = SV.read_raw()                                                     # prepared 가 아니면 예전 규칙 그대로 막힌다
        raw["models"][oi]["prepared"] = False
        self.assertTrue(any("hyperqwen 어댑터는 차선 0 만" in e for e in SV.validate(raw, ROOT)))
        raw = SV.read_raw()
        raw["models"][ORIGQ + "-bf16h"]["prepared"] = True                      # hyperqwen 이 없는 별칭에는 못 단다
        self.assertTrue(any("prepared 는 hyperqwen" in e for e in SV.validate(raw, ROOT)))
        d = Path(tempfile.mkdtemp())
        e = SV.select_env(self.t, oi, "hyperqwen-8cf642e", "hyperqwen")
        self.assertEqual(e["M_PREPARED"], "1")
        SV.write_env(e, d / "l.env")
        base = {"STEWART_MODELS": "/w/models", "FAST_VARIANT": "0", "CTX": "long", "SPEC": "mtp"}
        for lane in (0, 2):                                                     # 차선 0 이어도 prepare 없이
            argv, env = SV.serve_cmd(self.t, d / "l.env", lane, 8000 + 10 * lane, base)
            self.assertEqual(argv, ["bash", "/app/single-user/start_qwen.sh"])
            self.assertEqual((env["PREPARE"], env["VERIFY"], env["MODEL"], env["FAST_VARIANT"]), ("0", "0", f"/w/models/{oi}", "0"))
        e = SV.select_env(self.t, A, "hyperqwen-8cf642e", "hyperqwen")            # 모델 A 는 그대로 차선 0 에서 prepare
        self.assertEqual(e["M_PREPARED"], "0")

    def test_fetch_hf_mirror(self):
        """T35: home 별칭에 mirror 가 있고 읽기 토큰 파일이 있으면 HF 사본에서 받아 files 로 대조한다. 토큰은 파일 경로(HF_TOKEN_PATH)로만.
        토큰이 없거나 받기가 실패하면 집에서 밀어 넣기(modelrecv)로 간다. 대조가 어긋나면 HALT."""
        d = Path(tempfile.mkdtemp())
        S, sv = d / "S", d / "S" / "serving"
        sv.mkdir(parents=True)
        shutil.copy(ROOT / "serving.sh", sv / "serving.sh")
        good = {"config.json": b'{"a": 1}', "model.safetensors.index.json": b"{}", "m-1.safetensors": b"w" * 3000}
        (d / "fix" / "al-int8h").mkdir(parents=True)
        for n, b in good.items():
            (d / "fix" / "al-int8h" / n).write_bytes(b)
        (d / "fix" / "other").mkdir()
        (d / "fix" / "other" / "x.json").write_text("{}")
        pins = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(good.items()))
        (sv / "serving.py").write_text(f"import sys\nif sys.argv[1] == 'pins': print({pins!r}, end='')\nelif sys.argv[1] == 'home': print('100.64.0.1 8090 1')\n")
        (d / "hf").write_text("#!/bin/sh\n" f'echo "hf $* TOKEN_PATH=$HF_TOKEN_PATH TOKEN=${{HF_TOKEN:-}}" >> "{d}/calls"\n'
                              f'[ -e "{d}/hf-fail" ] && exit 1\n'
                              'while [ $# -gt 0 ]; do case "$1" in --local-dir) shift; LD=$1;; esac; shift; done\n'
                              f'mkdir -p "$LD" && cp -R "{d}/fix/al-int8h" "{d}/fix/other" "$LD/"\n')
        (d / "hf").chmod(0o755)
        sha = "sha256sum" if shutil.which("sha256sum", path="/usr/bin:/bin") else "shasum -a 256"
        def run(token: bool, extra: str = ""):
            tok = d / "tok"
            tok.write_text("hf_ro_xxx") if token else tok.unlink(missing_ok=True)
            sh = (f"set -euo pipefail\nS={S}\nPY={sys.executable}\nHF={d}/hf\nHF_READ_FILE={tok}\n"
                  f'log() {{ echo "LOG $*"; }}\nhalt() {{ echo "HALT $*"; exit 7; }}\n'
                  f'pin_check() {{ ( cd "$1" && printf "%s\\n" "$2" | {sha} -c >/dev/null ) || halt "$3 sha256 불일치"; log "$3 sha256 일치"; }}\n'
                  f". {sv}/serving.sh\n"
                  f"M_SOURCE=home M_ALIAS=al-int8h M_REV=home-abc M_MIRROR_REPO=o/r M_MIRROR_REV={'a' * 40} MDIR={d}/models/al-int8h\n"
                  f"{extra}\nserving_fetch\necho FETCH-END\n")
            return subprocess.run(["bash", "-c", sh], capture_output=True, text=True, timeout=30)
        r = run(True)
        self.assertIn("LOG al-int8h(hf-mirror) sha256 일치", r.stdout, r.stdout + r.stderr)
        self.assertIn("FETCH-END", r.stdout)
        self.assertEqual(sorted(p.name for p in (d / "models" / "al-int8h").iterdir()), sorted([*good, ".stewart-rev"]))
        call = (d / "calls").read_text()
        self.assertIn("--include al-int8h/*", call)
        self.assertIn(f"TOKEN_PATH={d}/tok TOKEN=\n", call)                       # 값이 아니라 파일 경로만
        self.assertNotIn("modelrecv", r.stdout)
        shutil.rmtree(d / "models")                                              # 토큰이 없으면 사본을 건너뛴다 → 집에서 받기(여기서는 가짜로 실패)
        r = run(False, "modelrecv_called=1")
        self.assertIn("model-fetch wait alias=al-int8h", r.stdout)
        (d / "hf-fail").touch()                                                  # 받기 실패 → 집에서 받기로 넘어간다
        r = run(True)
        self.assertIn("hf-mirror 실패 rc=1", r.stdout)
        self.assertIn("model-fetch wait alias=al-int8h", r.stdout)
        (d / "hf-fail").unlink()
        (d / "fix" / "al-int8h" / "m-1.safetensors").write_bytes(b"x" * 3000)   # 바이트가 다르면 HALT(표시를 남기지 않는다)
        r = run(True)
        self.assertIn("HALT al-int8h(hf-mirror) sha256 불일치", r.stdout)
        self.assertFalse((d / "models" / "al-int8h" / ".stewart-rev").exists())

    def test_logscrub_serve(self):
        """serve 파드의 vLLM 출력에서 사용자 글을 지우고 수치·상태는 남긴다."""
        L = _load("logscrub")
        secret = "내 비밀 이야기를 길게 적은 사용자 글 0123456789"
        cases = [f"(APIServer pid=1) INFO [logger.py:41] Received request chatcmpl-1: prompt: '{secret}', params: SamplingParams(n=1)",
                 f"  Value error, [type=value_error, input_value={{'messages': [{{'role': 'user', 'content': '{secret}'}}]}}, input_type=dict]",
                 f"jinja2.exceptions.TemplateError: {secret}",
                 f'content: "{secret}"',
                 "x" * 50 + f" '{secret}'",
                 "y" * 500]
        for c in cases:
            out = L.scrub(c)
            self.assertNotIn("비밀", out, c)
            self.assertLessEqual(len(out), 260)
        keep = ["INFO 11-11 [scheduler.py] request finished id=chatcmpl-1 reason=stop prompt_tokens=120 generated_tokens=400 elapsed_s=12.3",
                "INFO 11-11 [gpu_model_runner.py] Model loading took 16.8 GiB and 40.2 seconds",
                "INFO 11-11 [kv_cache_utils.py] GPU KV cache size: 120,000 tokens"]
        for c in keep:
            self.assertEqual(L.scrub(c), c)                                     # 수치·상태 줄은 그대로
        sh = (ROOT.parent / "vast" / "pod-start.sh").read_text()
        self.assertIn('exec > >($PY "$SV/logscrub.py" | tee -a "$S/vllm.log") 2>&1', sh)   # serve 갈래의 출력은 거른 뒤에만
        self.assertLess(sh.index('if [ "$ROLE" = "eval" ]; then'), sh.index("logscrub.py"))   # eval 갈래 뒤(serve 만)
        e = SV.serve_cmd(self.t, self._env_file(), 0, 8000, {"STEWART_MODELS": "/m", "FAST_VARIANT": "0"})[1]
        self.assertEqual(e["VLLM_ENABLE_RESPONSES_API_STORE"], "0")              # 지난 응답을 서버가 쥐지 않는다
        self.assertNotIn("--enable-log-requests", " ".join(SV.vllm_argv(self.t["models"][A], "/m", 8000)))

    def _env_file(self):
        d = Path(tempfile.mkdtemp())
        SV.write_env(SV.select_env(self.t, "default", "auto", "hyperqwen"), d / "e")
        return d / "e"

    def test_plan(self):
        lanes = SV.plan_lanes(self.t, "cmp-a-b1", "hyperqwen", 4)
        self.assertEqual([(x["k"], x["port"], x["cards"], x["alias"], x["adapter"]) for x in lanes],
                         [(0, 8000, "0", A, "hyperqwen-8cf642e"), (1, 8010, "1", A, "vllm-patched"),
                          (2, 8020, "2,3", B1, "vllm-patched")])          # 포트 80k0(T20), B1 은 카드 두 장
        for kind, gpus in (("vllm-openai", 4), ("hyperqwen", 3)):
            with self.assertRaises(SV.TableError):
                SV.plan_lanes(self.t, "cmp-a-b1", kind, gpus)
        raw = SV.read_raw()
        bad = copy.deepcopy(raw)
        bad["plans"]["cmp-a-b1"]["lanes"].reverse()                   # hyperqwen 차선이 0 이 아니다
        self.assertTrue(any("차선 0 만" in e for e in SV.validate(bad, ROOT)))
        for g in (3, 5):                                               # T25: 카드 합 = gpus(모자람도 남음도 안 됨)
            bad = copy.deepcopy(raw)
            bad["plans"]["cmp-a-b1"]["gpus"] = g
            self.assertTrue(any("카드 합" in e for e in SV.validate(bad, ROOT)), g)
        bad = copy.deepcopy(raw)
        bad["plans"]["a-all"]["fill"] = "yes"
        self.assertTrue(any("fill" in e for e in SV.validate(bad, ROOT)))
        bad = copy.deepcopy(raw)
        bad["plans"]["a-all"]["cards"] = 8
        self.assertTrue(any("모르는 칸" in e for e in SV.validate(bad, ROOT)))
        with self.assertRaises(SV.TableError):
            SV.plan_lanes(self.t, "cmp-a-b1", "hyperqwen", 8)          # 4장 계획을 8장에 → 남는 카드
        self.assertEqual(len(SV.plan_lanes(self.t, "a-all", "hyperqwen", 8)), 8)
        self.assertEqual([x["quality"] for x in SV.plan_lanes(self.t, "a-all", "hyperqwen", 4)], [True, False, False, False])
        with self.assertRaises(SV.TableError):
            SV.plan_lanes(self.t, "a-all", "hyperqwen", 9)
        fill = copy.deepcopy(raw)                                      # fill 의 마지막 차선이 tp 2 면 2장씩만 채운다
        fill["plans"]["b-all"] = {"image": "hyperqwen", "gpus": 2, "fill": True,
                                  "lanes": [{"model": B1, "adapter": "vllm-patched", "quality": True}]}
        self.assertEqual(SV.validate(fill, ROOT), [])
        self.assertEqual([x["cards"] for x in SV.plan_lanes(fill, "b-all", "hyperqwen", 6)], ["0,1", "2,3", "4,5"])
        with self.assertRaises(SV.TableError):
            SV.plan_lanes(fill, "b-all", "hyperqwen", 5)
        bad = copy.deepcopy(raw)
        bad["plans"]["cmp-a-b1"]["lanes"][2]["adapter"] = "hyperqwen-8cf642e"
        self.assertTrue(any("허락하지 않는" in e for e in SV.validate(bad, ROOT)))
        self.assertEqual(SV.plan_env(self.t, "cmp-a-b1"), {"CTX": "long", "FAST_VARIANT": "0", "PREFIX_CACHE": "1", "SPEC": "mtp"})
        for env in ({"FAST_VARIANT": "2"}, {"MODEL": "x"}, {"PU": "https://evil"}, {"CTX": "huge"}, "CTX=long"):
            bad = copy.deepcopy(raw)
            bad["plans"]["cmp-a-b1"]["env"] = env                       # 계획 조건은 허용 목록(CTX·SPEC·PREFIX_CACHE·FAST_VARIANT)만
            self.assertTrue(any("env 는" in e for e in SV.validate(bad, ROOT)), env)

    def test_cli(self):
        r = subprocess.run([sys.executable, str(ROOT / "serving.py"), "validate"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([sys.executable, str(ROOT / "serving.py"), "select", "nope", "auto", "hyperqwen", os.devnull],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("표에 없는 별칭", r.stderr)


def _diff(old: str, new: str, path="x.py") -> str:
    import difflib
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}", n=3))


class TPatch(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.base = "".join(f"line {i}\n" for i in range(40))

    def test_apply_offset_and_reverse(self):
        new = self.base.replace("line 10\n", "line 10 changed\n").replace("line 30\n", "")
        p = _diff(self.base, new)
        (self.d / "x.py").write_text("extra 0\nextra 1\n" + self.base)          # 위치만 2줄 어긋남(offset 은 받는다)
        PS.apply_patch(self.d, p)
        self.assertEqual((self.d / "x.py").read_text(), "extra 0\nextra 1\n" + new)
        self.assertTrue(PS.is_applied(self.d, p))
        PS.apply_patch(self.d, p, reverse=True)
        self.assertEqual((self.d / "x.py").read_text(), "extra 0\nextra 1\n" + self.base)

    def test_fuzz0_refuses_and_writes_nothing(self):
        new = self.base.replace("line 10\n", "line 10 changed\n").replace("line 30\n", "line 30 changed\n")
        p = _diff(self.base, new)
        bad = self.base.replace("line 31\n", "line 31 (다름)\n")               # 둘째 hunk 의 문맥 한 줄이 다르다
        (self.d / "x.py").write_text(bad)
        with self.assertRaises(PS.PatchError):
            PS.apply_patch(self.d, p)
        self.assertEqual((self.d / "x.py").read_text(), bad)                    # 첫 hunk 도 쓰지 않았다

    def test_new_file_and_no_newline(self):
        p = "--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1,2 @@\n+a\n+b\n\\ No newline at end of file\n"
        PS.apply_patch(self.d, p)
        self.assertEqual((self.d / "pkg/new.py").read_text(), "a\nb")
        with self.assertRaises(PS.PatchError):
            PS.apply_patch(self.d, p)                                          # 이미 있다
        PS.apply_patch(self.d, p, reverse=True)
        self.assertFalse((self.d / "pkg/new.py").exists())

    def test_preamble_text_ignored(self):
        p = "설명 글\n--- 이것은 머리가 아니다\n본문\n\n" + _diff(self.base, self.base.replace("line 5\n", "five\n"))
        (self.d / "x.py").write_text(self.base)
        self.assertEqual(PS.apply_patch(self.d, p), ["x.py"])

    def test_build_with_tables(self):
        pkg = self.d / "src" / "vllm"
        pkg.mkdir(parents=True)
        (pkg / "x.py").write_text(self.base)
        sd = self.d / "set"
        sd.mkdir()
        new = self.base.replace("line 3\n", "three\n")
        (sd / "one.patch").write_text(_diff(self.base, new))
        (sd / "series").write_text("one.patch\n")
        (sd / "pristine.sha256").write_text(f"{PS.sha256(pkg / 'x.py')}  x.py\n")
        (pkg.parent / "after.py").write_text(new)
        (sd / "applied.sha256").write_text(f"{PS.sha256(pkg.parent / 'after.py')}  x.py\n")
        sums = lambda: (sd / "SHA256SUMS").write_text("".join(f"{PS.sha256(sd / n)}  {n}\n" for n in
                                                             ("series", "pristine.sha256", "applied.sha256", "one.patch")))
        sums()
        info = PS.build(pkg, self.d / "ov", sd, None, None)
        self.assertEqual((self.d / "ov/vllm/x.py").read_text(), new)
        self.assertEqual((pkg / "x.py").read_text(), self.base)                 # 원래 것은 그대로
        self.assertFalse(info["reused"])
        self.assertTrue(PS.build(pkg, self.d / "ov", sd, None, None)["reused"])
        (sd / "one.patch").write_text(_diff(self.base, self.base.replace("line 4\n", "four\n")))   # 묶음 파일을 바꿈
        with self.assertRaises(PS.PatchError):
            PS.build(pkg, self.d / "ov2", sd, None, None)                      # SHA256SUMS 불일치
        sums()
        (pkg / "x.py").write_text(self.base + "local edit\n")
        with self.assertRaises(PS.PatchError):
            PS.build(pkg, self.d / "ov3", sd, None, None)                      # 원본(pristine)이 아니다
        self.assertFalse((self.d / "ov3").exists())

    def test_real_set_files(self):
        sd = ROOT / "patches" / "vllm-0.30.0"
        PS.check_set_files(sd)
        self.assertEqual(len(PS.series(sd)), 12)
        self.assertEqual(PS.series(sd)[0], "qwen3_5-embed-quant.patch")
        self.assertIn("speed-knobs-envs.patch", PS.series(sd))                 # 묶음 패치가 쓰는 노브를 envs.py 에 등록
        man = PS.read_sums(ROOT / "patches" / "hyperqwen-8cf642e.sha256")
        self.assertEqual(len([n for _, n in man if n.startswith("patches/") and n != "patches/series"]), 44)


@unittest.skipUnless(os.environ.get("STEWART_VLLM_SRC"), "STEWART_VLLM_SRC(vLLM 0.30.0 원본 vllm 폴더)가 없다")
class TRealVllm(unittest.TestCase):
    def test_overlay_from_pristine(self):
        src = Path(os.environ["STEWART_VLLM_SRC"])
        sd = ROOT / "patches" / "vllm-0.30.0"
        self.assertEqual(PS.check_pristine(src, sd), [])
        d = Path(tempfile.mkdtemp())
        info = PS.build(src, d / "ov", sd, None, None)
        self.assertEqual(info["applied"], PS.series(sd))
        self.assertEqual(PS.check_pristine(d / "ov" / "vllm", sd, "applied.sha256"), [])
        shutil.rmtree(d)

    @unittest.skipUnless(os.environ.get("STEWART_HQ_APP") and os.environ.get("STEWART_HQ_PKG"),
                         "STEWART_HQ_APP(HyperQwen 8cf642e /app 사본)·STEWART_HQ_PKG(패치 덧댄 vllm)가 없다")
    def test_overlay_from_hyperqwen_image(self):
        sd = ROOT / "patches" / "vllm-0.30.0"
        d = Path(tempfile.mkdtemp())
        info = PS.build(Path(os.environ["STEWART_HQ_PKG"]), d / "ov", sd, Path(os.environ["STEWART_HQ_APP"]),
                        ROOT / "patches" / "hyperqwen-8cf642e.sha256")
        self.assertEqual(info["hyperqwen_unapplied"], 47)                       # series 44 + KVarN 3
        self.assertEqual(PS.check_pristine(d / "ov" / "vllm", sd, "applied.sha256"), [])
        shutil.rmtree(d)


class TImageInfo(unittest.TestCase):
    def test_runs_without_vllm(self):
        r = subprocess.run([sys.executable, str(ROOT / "image_info.py"), "--verify-log", os.devnull], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        for k in ("python", "vllm", "nvcc", "image_kind", "verify"):
            self.assertIn(k, d)

    def test_verify_summary(self):
        _load("image_info")
        import image_info as II
        p = Path(tempfile.mkdtemp()) / "log"
        p.write_text("x\n  PASS  a applied\n  WARN  b\n  FAIL  c NOT applied\n  PASS  d\n")
        s = II.verify_summary(p)
        self.assertEqual((s["pass"], s["warn"], s["fail"]), (2, 1, 1))
        self.assertEqual(s["fail_lines"], ["FAIL  c NOT applied"])


if __name__ == "__main__":
    unittest.main()
