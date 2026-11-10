"""변환 조리법(deploy/quant): safetensors 바이트 다루기, 0 채우기 전후 같은 MLP 출력, MTP 붙이기, 묶음·순서."""

import importlib.util
import json
import math
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import FIX_ROOT, LOCAL

ROOT = FIX_ROOT / "deploy" / "quant"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ST = _load("st_io")
PAD = _load("pad_intermediate")
GR = _load("graft_missing")
QA = _load("quantize_autoround")
BE = _load("build_export")
QS = _load("quantsrv")
MB = _load("make_quant_bundle")

H, I_LANG, I_MTP, T = 4, 127, 126, 128


def f32(vals):
    return struct.pack(f"<{len(vals)}f", *vals)


def unf32(b):
    return list(struct.unpack(f"<{len(b) // 4}f", b))


def write_st(path: Path, tensors: dict[str, tuple[list[int], list[float]]]):
    plan = [(k, {"dtype": "F32", "shape": shp}, (lambda v: (lambda out: out.write(f32(v))))(vals))
            for k, (shp, vals) in tensors.items()]
    ST.write_file(path, plan, {"format": "pt"})


def read_tensor(path: Path, name: str):
    h, start = ST.read_header(path)
    b, e = h[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(start + b)
        return h[name]["shape"], unf32(f.read(e - b))


def mlp(gate, up, down, x, inter):
    """y = down(silu(gate x) * up x), 행 우선 행렬, 순수 파이썬 float32 흉내 없이 float64."""
    hid = []
    for j in range(inter):
        g = sum(gate[j * H + k] * x[k] for k in range(H))
        u = sum(up[j * H + k] * x[k] for k in range(H))
        hid.append(g / (1 + math.exp(-g)) * u)
    return [sum(down[r * inter + j] * hid[j] for j in range(inter)) for r in range(H)]


def make_model(d: Path, rnd: random.Random):
    d.mkdir(parents=True)
    t = {}
    for name, inter in (("model.language_model.layers.0", I_LANG), ("model.language_model.layers.1", I_LANG),
                        ("mtp.layers.0", I_MTP)):
        t[f"{name}.mlp.gate_proj.weight"] = ([inter, H], [rnd.uniform(-1, 1) for _ in range(inter * H)])
        t[f"{name}.mlp.up_proj.weight"] = ([inter, H], [rnd.uniform(-1, 1) for _ in range(inter * H)])
        t[f"{name}.mlp.down_proj.weight"] = ([H, inter], [rnd.uniform(-1, 1) for _ in range(inter * H)])
    t = {k: (shp, unf32(f32(v))) for k, (shp, v) in t.items()}          # float32 로 저장되는 값 그대로 비교한다
    t["model.language_model.embed_tokens.weight"] = ([3, H], [0.5] * (3 * H))
    t["model.visual.blocks.0.mlp.linear_fc1.weight"] = ([I_LANG, H], [0.25] * (I_LANG * H))   # 비전은 건드리지 않는다
    keys = sorted(t)
    s1 = {k: t[k] for k in keys if not k.startswith("mtp.")}
    s2 = {k: t[k] for k in keys if k.startswith("mtp.")}
    write_st(d / "model-00001-of-00002.safetensors", s1)
    write_st(d / "model-00002-of-00002.safetensors", s2)
    wm = {k: "model-00001-of-00002.safetensors" for k in s1} | {k: "model-00002-of-00002.safetensors" for k in s2}
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                               "text_config": {"intermediate_size": I_LANG, "hidden_size": H}}))
    (d / "tokenizer.json").write_text("{}")
    return t


class TPad(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rnd = random.Random(7)
        self.t = make_model(self.tmp / "src", self.rnd)

    def test_pad_same_output(self):
        rc = PAD.main(["--src", str(self.tmp / "src"), "--dst", str(self.tmp / "pad"), "--to", str(T)])
        self.assertEqual(rc, 0)
        d = self.tmp / "pad"
        for name, inter in (("model.language_model.layers.0", I_LANG), ("model.language_model.layers.1", I_LANG),
                            ("mtp.layers.0", I_MTP)):
            shard = d / ("model-00002-of-00002.safetensors" if name.startswith("mtp") else "model-00001-of-00002.safetensors")
            gs, g = read_tensor(shard, f"{name}.mlp.gate_proj.weight")
            us, u = read_tensor(shard, f"{name}.mlp.up_proj.weight")
            ds, dn = read_tensor(shard, f"{name}.mlp.down_proj.weight")
            self.assertEqual((gs, us, ds), ([T, H], [T, H], [H, T]))
            self.assertEqual(g[:inter * H], self.t[f"{name}.mlp.gate_proj.weight"][1])     # 원래 행 그대로(바이트)
            self.assertEqual(set(g[inter * H:]) | set(u[inter * H:]), {0.0})              # 새 행은 0
            self.assertEqual({dn[r * T + j] for r in range(H) for j in range(inter, T)}, {0.0})   # 새 열은 0
            for _ in range(5):
                x = [self.rnd.uniform(-2, 2) for _ in range(H)]
                a = mlp(self.t[f"{name}.mlp.gate_proj.weight"][1], self.t[f"{name}.mlp.up_proj.weight"][1],
                        self.t[f"{name}.mlp.down_proj.weight"][1], x, inter)
                b = mlp(g, u, dn, x, T)
                self.assertEqual(a, b)                                                     # 0 을 더하므로 정확히 같다
        _, vis = read_tensor(d / "model-00001-of-00002.safetensors", "model.visual.blocks.0.mlp.linear_fc1.weight")
        self.assertEqual(len(vis), I_LANG * H)                                             # 비전 MLP 는 그대로
        cfg = json.loads((d / "config.json").read_text())
        self.assertEqual(cfg["text_config"]["intermediate_size"], T)
        rep = json.loads((d / "pad_report.json").read_text())
        self.assertEqual(rep["intermediate_seen"], {str(I_LANG): 6, str(I_MTP): 3})
        self.assertEqual(rep["counts"], {"gate_proj": 2, "up_proj": 2, "down_proj": 2, "gate_proj (mtp)": 1,
                                         "up_proj (mtp)": 1, "down_proj (mtp)": 1})
        self.assertTrue((d / "tokenizer.json").exists())
        for s in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            ST.check_intact(d / s)

    def test_refuse(self):
        self.assertEqual(PAD.main(["--src", str(self.tmp / "src"), "--dst", str(self.tmp / "x"), "--to", "100"]), 2)
        with self.assertRaises(ValueError):
            PAD.plan_tensor("model.language_model.layers.0.mlp.down_proj.weight", {"dtype": "BF16", "shape": [4, 17409]}, 17408)
        with self.assertRaises(ValueError):
            PAD.plan_tensor("mtp.layers.0.mlp.up_proj.weight", {"dtype": "BF16", "shape": [1000, 4]}, 17536)
        self.assertIsNone(PAD.plan_tensor("model.language_model.layers.0.self_attn.q_proj.weight",
                                          {"dtype": "BF16", "shape": [4, 17409]}, 17536)[1])

    def test_graft(self):
        PAD.main(["--src", str(self.tmp / "src"), "--dst", str(self.tmp / "pad"), "--to", str(T)])
        out = self.tmp / "out"
        out.mkdir()
        # 양자화 결과 흉내: mtp 가 없고 언어 MLP 는 weight_packed 로 바뀜
        tens = {"model.language_model.layers.0.mlp.gate_proj.weight_packed": ([1], [1.0]),
                "model.language_model.layers.0.mlp.up_proj.weight_packed": ([1], [1.0]),
                "model.language_model.layers.0.mlp.down_proj.weight_packed": ([1], [1.0]),
                "model.language_model.layers.1.mlp.gate_proj.weight_packed": ([1], [1.0]),
                "model.language_model.layers.1.mlp.up_proj.weight_packed": ([1], [1.0]),
                "model.language_model.layers.1.mlp.down_proj.weight_packed": ([1], [1.0]),
                "model.language_model.embed_tokens.weight": ([1], [1.0]),
                "model.visual.blocks.0.mlp.linear_fc1.weight": ([1], [1.0])}
        write_st(out / "model.safetensors", tens)
        (out / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tens}}))
        r = subprocess.run([sys.executable, str(ROOT / "graft_missing.py"), "--src", str(self.tmp / "pad"), "--dst", str(out)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)
        self.assertEqual((res["grafted"], res["src_modules_missing_in_dst"]), (3, 0))
        idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(idx["mtp.layers.0.mlp.down_proj.weight"], GR.EXTRA)
        shp, v = read_tensor(out / GR.EXTRA, "mtp.layers.0.mlp.down_proj.weight")
        self.assertEqual(shp, [H, T])
        _, pv = read_tensor(self.tmp / "pad" / "model-00002-of-00002.safetensors", "mtp.layers.0.mlp.down_proj.weight")
        self.assertEqual(v, pv)                                                           # 바이트 그대로
        r = subprocess.run([sys.executable, str(ROOT / "graft_missing.py"), "--src", str(self.tmp / "pad"), "--dst", str(out)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)                                                 # 두 번은 안 된다
        # 빠진 모듈이 남으면 3
        out2 = self.tmp / "out2"
        out2.mkdir()
        write_st(out2 / "model.safetensors", {"model.language_model.embed_tokens.weight": ([1], [1.0])})
        (out2 / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.language_model.embed_tokens.weight": "model.safetensors"}}))
        r = subprocess.run([sys.executable, str(ROOT / "graft_missing.py"), "--src", str(self.tmp / "pad"), "--dst", str(out2)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)


class TRecipe(unittest.TestCase):
    def test_layer_config_and_intermediate(self):
        keys = ["model.language_model.layers.0.linear_attn.in_proj_a.weight", "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "model.language_model.layers.3.self_attn.q_proj.weight",
                "mtp.fc.weight"]
        self.assertEqual(QA.bf16_layer_config(keys), {"model.language_model.layers.0.linear_attn.in_proj_a": {"bits": 16},
                                                      "model.language_model.layers.0.linear_attn.in_proj_b": {"bits": 16}})
        with self.assertRaises(SystemExit):
            QA.check_intermediate({"text_config": {"intermediate_size": 17409}})
        self.assertEqual(QA.check_intermediate({"text_config": {"intermediate_size": 17536}}), 17536)
        self.assertEqual(QA.RECIPE | {}, {"scheme": "W4A16", "dataset": "NeelNanda/pile-10k", "nsamples": 128, "seqlen": 2048,
                                         "batch_size": 4, "iters": 200, "seed": 42, "device_map": 0,
                                         "trust_remote_code": True, "quant_nontext_module": False})
        self.assertEqual((QA.FORMAT, QA.WANT["auto-round"], QA.WANT["transformers"]), ("llm_compressor", "0.14.2", "5.15.0"))

    def test_settle_output(self):
        """auto-round 0.14.2 는 out/<모델 폴더 이름>-w4g128/ 에 쓴다 → 파일을 out 바로 아래로 옮긴다."""
        out = Path(tempfile.mkdtemp())
        sub = out / "src-w4g128"
        sub.mkdir()
        for n in ("config.json", "model.safetensors.index.json", "model-00001-of-00004.safetensors"):
            (sub / n).write_text(n)
        self.assertEqual(QA.settle_output(out, [str(sub)]), "src-w4g128")
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["config.json", "model-00001-of-00004.safetensors",
                                                                  "model.safetensors.index.json"])
        self.assertIsNone(QA.settle_output(out, [str(out)]))                     # 이미 바로 아래면 그대로
        out2 = Path(tempfile.mkdtemp())                                            # folders 를 못 받아도 색인 가진 하위 폴더 하나면
        (out2 / "x-w4g128").mkdir()
        (out2 / "x-w4g128" / "model.safetensors.index.json").write_text("{}")
        self.assertEqual(QA.settle_output(out2, None), "x-w4g128")
        out3 = Path(tempfile.mkdtemp())                                            # 둘 이상이면 멈춘다
        for d in ("a", "b"):
            (out3 / d).mkdir()
            (out3 / d / "model.safetensors.index.json").write_text("{}")
        with self.assertRaises(SystemExit):
            QA.settle_output(out3, None)
        with self.assertRaises(SystemExit):                                        # 출력 폴더 밖은 받지 않는다
            other = Path(tempfile.mkdtemp())
            (other / "config.json").write_text("{}")
            QA.settle_output(Path(tempfile.mkdtemp()), [str(other)])

    def test_pins(self):
        lines = [ln for ln in (ROOT / "pins/demo2-bf16.sha256").read_text().splitlines() if ln and not ln.startswith("#")]
        names = [ln.split("  ", 1)[1] for ln in lines]
        self.assertEqual(len(names), 24)
        self.assertEqual(sum(n.startswith("model-000") for n in names), 11)
        for n in ("config.json", "model.safetensors.index.json", "demo2_report.json", "tokenizer.json", "chat_template.jinja"):
            self.assertIn(n, names)


class TBundle(unittest.TestCase):
    def test_bundle(self):
        import base64, gzip, hashlib, io, tarfile
        b = MB.bundle()
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(b))) as t:
            files = {n: t.extractfile(n).read() for n in t.getnames()}
        sums = dict((ln.split("  ")[1], ln.split("  ")[0]) for ln in files["SHA256SUMS"].decode().splitlines())
        self.assertEqual(set(sums), set(files) - {"SHA256SUMS"})
        for n, h in sums.items():
            self.assertEqual(hashlib.sha256(files[n]).hexdigest(), h)
        self.assertEqual(files["quality_battery.py"], (ROOT.parent / "serving/bench/quality_battery.py").read_bytes())
        self.assertLessEqual(len(MB.ARGS), 256)
        self.assertLess(MB.ARGS.index("sha256sum -c"), MB.ARGS.index("exec bash /dev/shm/b"))          # boot 대조 뒤에만
        boot = (ROOT / "boot-quant.sh").read_text()
        self.assertLess(boot.index("sha256sum -c"), boot.index("base64 -d p"))                    # 묶음도 대조 뒤에만 푼다
        self.assertTrue(base64.b64encode(b))
        self.assertEqual(json.loads(files["home.json"])["server_ip"], MB.server_ip())            # 서빙 표 하나에서
        for gone in ("upload.py", "make_readme.py"):
            self.assertNotIn(gone, files)
        for must in ("quantsrv.py", "build_export.py", "audit_delta.py", "spec_metrics.py", "targets.json"):
            self.assertIn(must, files)
        tg = json.loads(files["targets.json"])
        self.assertEqual([t["alias"] for t in tg["targets"]], ["qwen38-27b-orig-w4a16q", "qwen38-27b-demo-w4a16q", "qwen38-27b-demo2-w4a16"])
        self.assertEqual([t["pad_to"] for t in tg["targets"]], [0, 0, 17536])
        self.assertEqual(tg["audit"], ["qwen38-27b-orig-w4a16q", "qwen38-27b-demo-w4a16q"])
        for t in tg["targets"]:
            self.assertIn(t["pins"], files)                               # 원본 pins·템플릿 고정본도 묶음에(SHA256SUMS 대조)
            self.assertIn(t["chat_template"], files)
            self.assertTrue(any(l.endswith("  model.safetensors.index.json") for l in files[t["pins"]].decode().splitlines()))
        self.assertEqual(hashlib.sha256(files["templates/qwen38-27b-chat.jinja"]).hexdigest(),
                         "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041")
        self.assertEqual(tg["targets"][0]["rev"], "37561816e66b5b9153b1b64c10968c18c1e6f4a0")

    def test_run_convert_order_and_no_token(self):
        sh = (ROOT / "run-convert.sh").read_text()
        r = subprocess.run(["bash", "-n", str(ROOT / "run-convert.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        order = ["mark 0-image", "mark 1-tools", "mark 3-data", "mark 3b-offline", 'mark "$alias/2-src"', 'mark "$alias/5-kl-pad"',
                 'mark "$alias/6-quant"', 'mark "$alias/7-kl-quant"', 'mark "$alias/8-prepare"', 'mark "$alias/9-serve-quality"',
                 'mark "$alias/10-pins"', 'mark "$alias/11-built"', "mark audit", "mark 11-export"]
        pos = [sh.index(s) for s in order]
        self.assertEqual(pos, sorted(pos))
        self.assertLess(sh.index("export HF_HUB_OFFLINE=1"), sh.index("quantize_autoround.py"))   # 양자화는 오프라인
        self.assertLess(sh.index("--max-mean-kl 1e-4"), sh.index("quantize_autoround.py"))        # 0 채우기 검사 통과 뒤에만
        loop = sh.index("done <<< \"$TARGETS\"")
        self.assertLess(sh.index("convert_target \"$alias\""), loop)                                # 대상마다 차례로, 감사는 쌍이 갖춰지면
        self.assertLess(sh.index("run_audit\n"), loop)
        self.assertIn('in_audit "$alias" || rm -rf "$SRC"', sh)                                    # 감사 쌍의 원본은 감사 뒤에 지운다
        self.assertLess(sh.index('mark audit'), sh.index('step "$x/6-quant" && rm -rf "$(src_of "$x")"'))
        body = sh[sh.index("convert_target() {"):sh.index("in_audit() {")]
        self.assertLess(body.index('mark "$alias/2-src"'), body.index("run_audit"))                # 원본을 받자마자 감사(쌍이 갖춰지면)
        self.assertLess(body.index("run_audit"), body.index("quantize_autoround.py"))
        self.assertIn('"convert-$alias" --gsm-only', sh)
        # 도구: 이미지에 있는 꾸러미는 이미지 판(요구 목록에서 뺌), 이미지 판은 묶되 pyarrow·fsspec 만 풀고, 실패하면 바로 멈춘다
        tools = sh[sh.index("if ! step 1-tools"):sh.index("mark 1-tools")]
        self.assertIn('UNPIN="pyarrow fsspec"', tools)
        data = sh[sh.index("if ! step 3-data"):sh.index("mark 3-data")]                         # 온라인일 때 datasets 캐시를 만든다
        self.assertLess(data.index("pins/quality-data.sha256"), data.index('load_dataset("NeelNanda/pile-10k", revision=sys.argv[1]'))
        self.assertLess(sh.index("mark 3-data"), sh.index("export HF_HUB_OFFLINE=1"))
        # 대상 원본 받기는 오프라인 전환 뒤에 있으므로 그 명령만 온라인으로(그 밖 HF 호출은 오프라인 그대로)
        dl = [ln for ln in sh.splitlines() if '$HF download "$repo"' in ln]
        self.assertEqual(len(dl), 1)
        self.assertIn("HF_HUB_OFFLINE=0", dl[0])
        self.assertLess(sh.index("export HF_HUB_OFFLINE=1"), sh.index('$HF download "$repo"'))
        self.assertFalse([ln for ln in sh.splitlines() if "HF_HUB_OFFLINE=0" in ln and "download" not in ln and not ln.lstrip().startswith("#")])
        self.assertNotIn("req-loose", tools)                                                      # 판을 다 푸는 대체 길 없음
        self.assertLess(tools.index('-c "$Q/constraints-solve.txt"'), tools.index("이미지 패키지 판이 바뀌었다"))
        self.assertIn("auto_round-0.14.2-py3-none-any.whl\" | sha256sum -c", tools.replace('"\n', '"'))
        blocks = tools.split("<<'PYEOF'")
        sel = blocks[1][blocks[1].index("\n") + 1:blocks[1].index("\nPYEOF\n") + 1]
        chk = blocks[2][blocks[2].index("\n") + 1:blocks[2].index("\nPYEOF\n") + 1]
        d = Path(tempfile.mkdtemp())
        (d / "c.txt").write_text("pydantic==2.12.5\ntorch==2.13.0\nPy_CpuInfo==8.0.0\npyarrow==20.0.0\nfsspec==2026.9.0\n")
        r = subprocess.run([sys.executable, "-", str(d / "c.txt"), str(ROOT / "requirements-quant.txt"), str(d / "p.txt"),
                            str(d / "s.txt"), "pyarrow fsspec"], input=sel, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((d / "p.txt").read_text().split(), ["accelerate==1.14.0", "datasets==5.0.0"])   # 이미지의 pydantic·py-cpuinfo 는 뺌
        self.assertEqual((d / "s.txt").read_text().split(), ["pydantic==2.12.5", "torch==2.13.0", "Py_CpuInfo==8.0.0"])
        self.assertIn("image keeps py-cpuinfo,pydantic", r.stdout)
        (d / "after.txt").write_text("pydantic==2.12.5\ntorch==2.13.0\nPy_CpuInfo==8.0.0\npyarrow==21.0.0\nfsspec==2026.4.0\ndatasets==5.0.0\n")
        r = subprocess.run([sys.executable, "-", str(d / "c.txt"), str(d / "after.txt"), "pyarrow fsspec"], input=chk,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)                                             # 푼 둘만 바뀌면 통과
        self.assertIn("allowed: {'fsspec': ('2026.9.0', '2026.4.0'), 'pyarrow': ('20.0.0', '21.0.0')}", r.stdout)
        (d / "after.txt").write_text("pydantic==2.13.4\ntorch==2.13.0\npy-cpuinfo==8.0.0\npyarrow==21.0.0\nfsspec==2026.9.0\n")
        r = subprocess.run([sys.executable, "-", str(d / "c.txt"), str(d / "after.txt"), "pyarrow fsspec"], input=chk,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)                                                       # 그 밖 이미지 판이 바뀌면 멈춘다
        self.assertIn("pydantic", r.stdout)                                            # mtp 서빙에서 prompt_logprobs 없음
        self.assertIn('cp "$HERE/$tpl" "$OUTB/chat_template.jinja"', sh)                           # 템플릿 고정본(묶음 sha 대조 뒤)
        self.assertLess(sh.index('grep -F "  $tpl" SHA256SUMS'), sh.index('cp "$HERE/$tpl"'))
        code = [ln for ln in sh.splitlines() if not ln.lstrip().startswith("#")]
        self.assertFalse([ln for ln in code if "HF_TOKEN" in ln and not ln.lstrip().startswith("log ")])  # 토큰을 다루지 않는다
        self.assertIn("unset CONTAINER_API_KEY", sh)
        self.assertLess(sh.index("unset CONTAINER_API_KEY"), sh.index("pip install"))
        self.assertIn("stop_instance", sh.split("halt()")[1].split("\n")[0])                         # HALT 면 정지
        self.assertIn('MAXH=${MAX_HOURS:-$(( 4 * NT + 2 ))}', sh)                                   # 최대 켜짐 감시(대상 수만큼)
        self.assertIn('sleep "$(( MAXH * 3600 ))"', sh)
        self.assertLess(sh.index("MAX_HOURS"), sh.index("mark 0-image"))
        for gone in ("upload.py", "make_readme", "HF_TOKEN", "--license"):                          # HF·토큰·라이선스 없음
            self.assertNotIn(gone, sh)
        self.assertLess(sh.index("unset TS_AUTHKEY"), sh.index('"$Q/bin/tailscaled"'))             # 키를 환경에서 지운 뒤에 데몬
        self.assertIn("--hostname=stewart-quant", sh)
        self.assertLess(sh.index("quantsrv.py"), sh.index("mark 0-image"))                         # 진행 중에도 /status
        self.assertIn("step exported", sh)
        self.assertLess(sh.index("mark 11-export"), sh.index("step exported"))


class TExport(unittest.TestCase):
    """11단계 export.json(대상마다 세트 둘·보고)과 quantsrv(:8001 상태·:8090 내보내기)."""
    TG = {"targets": [{"alias": "qwen38-27b-orig-w4a16q", "repo": "Qwen/Qwen3.8-27B", "rev": "a" * 40, "pins": "pins/o.sha256",
                       "pad_to": 0, "chat_template": "templates/t.jinja"},
                      {"alias": "qwen38-27b-demo2-w4a16", "repo": "example-org/Demo2-Qwen3.8-27B", "rev": "b" * 40,
                       "pins": "pins/demo2-bf16.sha256", "pad_to": 17536, "chat_template": ""},
                      {"alias": "qwen38-27b-demo-w4a16q", "repo": "x/y", "rev": "c" * 40, "pins": "pins/h.sha256", "pad_to": 0,
                       "chat_template": ""}],
          "audit": None}

    def setUp(self):
        self.q = Path(tempfile.mkdtemp())
        (self.q / "done").mkdir()
        for alias in ("qwen38-27b-orig-w4a16q", "qwen38-27b-demo2-w4a16"):          # demo 는 아직 안 끝남
            t = self.q / "t" / alias
            for d in ("out-bf16heads", "out-int8heads"):
                (t / d).mkdir(parents=True)
                (t / d / "config.json").write_text('{"a": 1}')
                (t / d / "model.safetensors.index.json").write_text("{}")
                (t / d / "model-00001.safetensors").write_bytes(b"w" * 5000)
                (t / d / ".cache").mkdir()
                (t / d / "x.bak").write_text("b")
            (self.q / "done" / alias).mkdir()
            for m in ("2-src", "9-serve-quality", "10-pins", "11-built"):
                (self.q / "done" / alias / m).touch()
            (t / "kl-quant-bf16heads.json").write_text(json.dumps({"mean_kl": 0.01, "top1_agree": 0.95}))
            (t / "spec.json").write_text(json.dumps({"spec_decode": {"acceptance": 0.7}}))
        (self.q / "t" / "qwen38-27b-demo2-w4a16" / "kl-pad.json").write_text(json.dumps({"mean_kl": 1e-7, "top1_agree": 1.0, "pass": True}))
        (self.q / "audit-delta.json").write_text(json.dumps({"version": 1, "tensors": {"changed": 3}}))
        (self.q / "convert-log.tsv").write_text("t\tdone 0-image\nt\tdone 10-pins\n")
        (self.q / "done" / "0-image").touch()
        (self.q / "targets.json").write_text(json.dumps(self.TG))

    def build(self):
        rc = BE.main(["--q", str(self.q), "--targets", str(self.q / "targets.json"), "--pile-rev", "127bfed", "--draft-ids", ""])
        self.assertEqual(rc, 0)
        return json.loads((self.q / "export.json").read_text())

    def test_build_export(self):
        e = self.build()
        self.assertEqual([s["alias"] for s in e["sets"]], ["qwen38-27b-orig-w4a16q-bf16h", "qwen38-27b-orig-w4a16q-int8h",
                                                           "qwen38-27b-demo2-w4a16-bf16h", "qwen38-27b-demo2-w4a16-int8h"])
        self.assertEqual(e["sets"][1]["dir"], "t/qwen38-27b-orig-w4a16q/out-int8heads")
        names = [f["name"] for f in e["sets"][1]["files"]]
        self.assertEqual(names, ["config.json", "model-00001.safetensors", "model.safetensors.index.json"])   # 숨은 것·.bak 빠짐
        r0, r2 = e["sets"][0]["report"], e["sets"][2]["report"]
        self.assertEqual((r0["pad_to"], r0["kl_pad"], r0["src"]["repo"]), (0, None, "Qwen/Qwen3.8-27B"))
        self.assertEqual((r2["pad_to"], r2["kl_pad"]["mean_kl"], r2["src"]["rev"]), (17536, 1e-7, "b" * 40))
        self.assertEqual(r0["spec_decode"], {"acceptance": 0.7})
        self.assertTrue(r0["verify"]["ok"])
        self.assertEqual(e["report"], r0)                                   # 맨 위 = 첫 대상(옛 꼴 읽는 쪽)
        self.assertEqual(e["audit"]["tensors"]["changed"], 3)
        (self.q / "t" / "qwen38-27b-orig-w4a16q" / "out-int8heads" / "sub").mkdir()
        with self.assertRaises(SystemExit):
            self.build()                                                    # 하위 폴더는 거부

    def test_quantsrv(self):
        import socket
        import urllib.error
        import urllib.request

        def free():
            with socket.socket() as so:
                so.bind(("127.0.0.1", 0))
                return so.getsockname()[1]
        sp, ep = free(), free()
        ctx = QS.Ctx(self.q, "100.64.0.9", None)
        srvs = QS.serve(ctx, "127.0.0.1", sp, ep)
        self.addCleanup(lambda: [x.shutdown() for x in srvs])
        get = lambda port, path: json.loads(LOCAL.open(f"http://127.0.0.1:{port}{path}", timeout=5).read())
        st = get(sp, "/status")
        self.assertEqual((st["role"], st["stage"], st["export_ready"], st["exported"]), ("quant", "0-image", False, False))
        self.assertEqual(st["targets"], {"qwen38-27b-demo2-w4a16": "11-built", "qwen38-27b-orig-w4a16q": "11-built"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            LOCAL.open(f"http://127.0.0.1:{ep}/export", timeout=5)
        self.assertEqual(cm.exception.code, 409)                            # 아직
        self.build()
        idx = get(ep, "/export")
        alias = idx["sets"][1]["alias"]
        body = LOCAL.open(f"http://127.0.0.1:{ep}/export/{alias}/model-00001.safetensors", timeout=5).read()
        self.assertEqual(body, b"w" * 5000)
        for bad in (f"/export/{alias}/x.bak", f"/export/{alias}/..%2Fspec.json", "/export/nope/config.json"):
            with self.assertRaises(urllib.error.HTTPError, msg=bad) as cm:
                LOCAL.open(f"http://127.0.0.1:{ep}{bad}", timeout=5)
            self.assertEqual(cm.exception.code, 404)
        r = LOCAL.open(urllib.request.Request(f"http://127.0.0.1:{ep}/export/done", data=b"", method="POST"), timeout=5)
        self.assertEqual(r.status, 200)
        self.assertTrue((self.q / "done" / "exported").exists())
        self.assertTrue(get(sp, "/status")["exported"])
        r = LOCAL.open(urllib.request.Request(f"http://127.0.0.1:{sp}/logout", data=b"", method="POST"), timeout=5)
        self.assertEqual(json.loads(r.read())["ok"], True)


if __name__ == "__main__":
    unittest.main()
