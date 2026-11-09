"""층별 ΔW 감사(deploy/quant/audit_delta.py): 작은 safetensors 두 벌로 분류·요약 꼴과 특잇값 에너지 비율을 본다."""

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from tests import FIX_ROOT

ROOT = FIX_ROOT / "deploy" / "quant"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AD = _load("audit_delta")
try:
    import numpy as np
except ImportError:                                   # 저장소 venv 에는 없다. `uv run --with numpy` 로 돌리면 수치 시험까지
    np = None


def bf16(vals: list[float]) -> bytes:
    """float → BF16 바이트(반올림 없이 위 16비트)."""
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16) for v in vals)


def write_model(d: Path, tensors: dict[str, tuple[list, list]]) -> None:
    """tensors: 이름 → (모양, 값 목록). 샤드 둘로 나눠 쓴다(색인 포함)."""
    d.mkdir(parents=True)
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    wm = {}
    for shard, ns in shards.items():
        header, blobs, pos = {}, [], 0
        for n in ns:
            shape, vals = tensors[n]
            b = bf16(vals)
            header[n] = {"dtype": "BF16", "shape": shape, "data_offsets": [pos, pos + len(b)]}
            blobs.append(b)
            pos += len(b)
            wm[n] = shard
        hb = json.dumps(header).encode()
        hb += b" " * ((8 - (8 + len(hb)) % 8) % 8)
        (d / shard).write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))


def mat(r, c, f):
    return [float(f(i, j)) for i in range(r) for j in range(c)]


class TAudit(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        base = {
            "model.language_model.layers.0.mlp.down_proj.weight": ([8, 16], mat(8, 16, lambda i, j: ((i * 7 + j * 3) % 11) / 8 - 0.5)),
            "model.language_model.layers.1.mlp.down_proj.weight": ([8, 16], mat(8, 16, lambda i, j: ((i * 5 + j) % 13) / 8 - 0.7)),
            "model.language_model.layers.2.self_attn.o_proj.weight": ([8, 8], mat(8, 8, lambda i, j: ((i + 2 * j) % 7) / 4 - 0.75)),
            "model.language_model.layers.3.self_attn.o_proj.weight": ([8, 8], mat(8, 8, lambda i, j: ((3 * i + j) % 5) / 4 - 0.5)),
            "model.language_model.layers.3.input_layernorm.weight": ([8], [1.0] * 8),
            "mtp.layers.0.mlp.up_proj.weight": ([4, 8], mat(4, 8, lambda i, j: (i - j) / 8)),
            "model.visual.blocks.0.attn.qkv.weight": ([4, 4], mat(4, 4, lambda i, j: (i * j) / 16)),
            "model.language_model.embed_tokens.weight": ([16, 8], mat(16, 8, lambda i, j: ((i + j) % 3) / 2)),
        }
        ed = json.loads(json.dumps(base))
        # 층 1·3 의 행렬에 계수 1 변화(u vᵀ), 층 3 의 노름 하나, MTP 한 칸. 2 의 형은 0.5 배(계수 8 변화가 아니라 전체 비례)
        u, v = [1, -1, 0.5, 0, 0, 0.5, -0.5, 1], [0.5] * 16
        ed["model.language_model.layers.1.mlp.down_proj.weight"][1] = [a + u[i] * v[j] for (i, j), a in
                                                                       zip(((i, j) for i in range(8) for j in range(16)),
                                                                           base["model.language_model.layers.1.mlp.down_proj.weight"][1])]
        ed["model.language_model.layers.3.self_attn.o_proj.weight"][1] = [a + u[i] * 0.25 for (i, j), a in
                                                                          zip(((i, j) for i in range(8) for j in range(8)),
                                                                              base["model.language_model.layers.3.self_attn.o_proj.weight"][1])]
        ed["model.language_model.layers.3.input_layernorm.weight"][1][2] = 1.5
        ed["mtp.layers.0.mlp.up_proj.weight"][1][0] = 2.0
        write_model(self.d / "base", base)
        write_model(self.d / "edited", ed)

    def test_structure_without_numbers(self):
        r = AD.audit(self.d / "base", self.d / "edited", stats_fn=None)
        self.assertEqual(r["tensors"], {"total": 8, "same": 4, "changed": 4, "only_base": 0, "only_edited": 0, "shape_diff": 0})
        self.assertEqual(r["changed_layers"], {"ranges": "1,3", "first": 1, "last": 3, "count": 2})
        self.assertEqual(r["towers"]["mtp"], {"tensors": 1, "changed": 1})
        self.assertEqual(r["towers"]["visual"], {"tensors": 1, "changed": 0})
        self.assertEqual(r["towers"]["layers"], {"tensors": 5, "changed": 3})
        self.assertEqual(r["by_kind"]["model.language_model.layers.N.self_attn.o_proj.weight"], {"changed": 1, "of": 2})
        self.assertEqual({row["layer"] for row in r["rows"]}, {1, 3, None})
        self.assertEqual(AD.ranges([1, 2, 3, 5, 7, 8]), "1-3,5,7-8")

    @unittest.skipIf(np is None, "numpy 없음(uv run --with numpy 로 돌린다)")
    def test_energy_and_no_tensor_values(self):
        out = self.d / "a.json"
        tsv = self.d / "a.tsv"
        rc = AD.main(["--base", str(self.d / "base"), "--edited", str(self.d / "edited"), "--out", str(out), "--tsv", str(tsv)])
        self.assertEqual(rc, 0)
        r = json.loads(out.read_text())
        rows = {row["name"]: row for row in r["rows"]}
        d1 = rows["model.language_model.layers.1.mlp.down_proj.weight"]
        self.assertAlmostEqual(d1["e1"], 1.0, places=3)                   # 계수 1 변화 → 첫 특잇값이 에너지 전부
        self.assertEqual((d1["r90"], d1["r99"]), (1, 1))
        self.assertGreater(d1["rel"], 0)
        ln = rows["model.language_model.layers.3.input_layernorm.weight"]
        self.assertNotIn("e1", ln)                                          # 1차원은 크기만
        self.assertAlmostEqual(ln["max_abs"], 0.5, places=3)
        allowed = {"name", "layer", "tower", "kind", "shape", "dtype", "rel", "max_abs", "r90", "r99"} | {f"e{k}" for k in AD.KS}
        for row in r["rows"]:
            self.assertLessEqual(set(row), allowed)                         # 요약 수치뿐(텐서 값·벡터 없음)
            for v in row.values():
                self.assertFalse(isinstance(v, list) and len(v) > 2, row["name"])   # 모양([행, 열]) 말고 목록 없음
        self.assertEqual(tsv.read_text().splitlines()[0].split("\t"), list(AD.TSV_COLS))
        self.assertEqual(len(tsv.read_text().splitlines()), 1 + r["tensors"]["changed"])


if __name__ == "__main__":
    unittest.main()
