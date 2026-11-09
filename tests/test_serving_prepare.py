"""서빙 묶음의 채팅 틀 손질 스크립트가 파드에서처럼(묶음 폴더에서 바로) 돈다.

E14 차선 1·2(vllm-patched): "chat-template ModuleNotFoundError: No module named 'atomic_publish'" — HyperQwen 의 보조 모듈을
묶음에 넣지 않았다. 같은 꼴(묶음 밖 모듈 가져오기)이 다시 나오면 여기서 잡힌다.
"""

import ast
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import FIX_ROOT

REPO = FIX_ROOT
SV = REPO / "deploy" / "serving"
PREP = SV / "prepare"

# 묶음 파이썬이 가져다 써도 되는 바깥 꾸러미(파드 이미지 venv 에 있는 것). 이 밖의 이름은 표준 라이브러리이거나 같은 폴더 파일이어야 한다.
THIRD_PARTY = {"torch", "safetensors", "transformers", "vllm", "numpy", "pyarrow", "huggingface_hub", "tomllib", "yaml"}


def run(script: str, *args: str):
    # serving.sh 와 같은 꼴: $PY "$SV/prepare/<스크립트>" <인자>(-I 없이, 작업 폴더는 묶음 밖)
    return subprocess.run([sys.executable, str(PREP / script), *args], capture_output=True, text=True,
                          cwd=tempfile.gettempdir(), timeout=60)


class TPrepareScripts(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_harden_runs_and_rewrites(self):
        sys.path.insert(0, str(PREP))
        try:
            import harden_chat_template as H
        finally:
            sys.path.remove(str(PREP))
        t = self.tmp / "chat_template.jinja"
        t.write_text("{# head #}\n" + H.OLD + "\n{# tail #}\n", encoding="utf-8")
        r = run("harden_chat_template.py", str(t))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("ModuleNotFoundError", r.stdout + r.stderr)
        self.assertIn(H.MARKER, t.read_text(encoding="utf-8"))
        self.assertFalse((self.tmp / "chat_template.jinja.tmp").exists())   # 임시 파일 없이 바꿔 끼움

    def test_translate_runs_and_rewrites(self):
        sys.path.insert(0, str(PREP))
        try:
            import translate_chat_template as TR
        finally:
            sys.path.remove(str(PREP))
        d = self.tmp / "model"
        d.mkdir()
        body = ["{#- head #}\n", "    {%- " + TR.ANCHOR + "'medium') %}\n",
                "    {%- " + TR.SHAPES[0] + " ['low', 'medium', 'xhigh'] %}\n",
                "        {{- " + TR.SHAPES[1] + "'bad effort') }}\n", "    {%- " + TR.SHAPES[2] + " %}\n", "{#- tail #}\n"]
        (d / "chat_template.jinja").write_text("".join(body), encoding="utf-8")
        r = run("translate_chat_template.py", str(d))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(": translated", r.stdout.strip().splitlines()[-1])
        self.assertIn(TR.MARKER_V2, (d / "chat_template.jinja").read_text(encoding="utf-8"))

    def test_vendored_helper_is_pinned_upstream(self):
        pin = {ln.split("  ")[1]: ln.split("  ")[0] for ln in (REPO / "deploy/quant/image-8cf642e.sha256").read_text().splitlines()
               if "  " in ln and not ln.startswith("#")}
        self.assertEqual(hashlib.sha256((PREP / "atomic_publish.py").read_bytes()).hexdigest(), pin["app/prepare/atomic_publish.py"])
        self.assertIn("prepare/atomic_publish.py", (SV / "NOTICE").read_text(encoding="utf-8"))

    def test_bundle_python_imports_resolve(self):
        """deploy/serving 의 .py 가 가져오는 이름은 표준 라이브러리·같은 폴더 파일·허용 꾸러미뿐이다."""
        bad = []
        for f in sorted(SV.rglob("*.py")):
            siblings = {p.stem for p in f.parent.glob("*.py")}
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for n in names:
                    if n not in sys.stdlib_module_names and n not in siblings and n not in THIRD_PARTY:
                        bad.append(f"{f.relative_to(REPO)}: {n}")
        self.assertEqual(bad, [])

    def test_serving_sh_halts_on_template_failure(self):
        sh = (SV / "serving.sh").read_text(encoding="utf-8")
        for step in ("translate", "harden"):
            part = sh.split(f"        {step}) ")[1].split(";;")[0]
            self.assertIn(f'|| halt "chat-template {step} 실패', part)
            self.assertNotIn("| tail -n1", part)                 # 실패를 마지막 줄 로그로 흘려보내던 꼴


if __name__ == "__main__":
    unittest.main()
