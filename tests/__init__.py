"""시험 패키지.

- 가짜 백엔드 주소로 /v1/models(서버 문맥 길이)를 묻지 않게 끈다(시험이 직접 켜서 본다).
- FIX_ROOT: deploy/ 를 임시 폴더로 복사하고 tests/fixtures/deploy/(시험용 local 표·pins)를 얹은 뿌리.
  공개 표(models.toml)만으로는 고를 수 있는 모델이 없어서, 차선·계획·감사 시험은 이 사본을 읽는다. 저장소 파일은 바꾸지 않는다.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("STEWART_SERVED_CONTEXT", "0")

_REPO = Path(__file__).resolve().parents[1]
FIX_ROOT = Path(tempfile.mkdtemp(prefix="stewart-cloud-tests-"))
shutil.copytree(_REPO / "deploy", FIX_ROOT / "deploy", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(_REPO / "tests" / "fixtures" / "deploy", FIX_ROOT / "deploy", dirs_exist_ok=True)
atexit.register(shutil.rmtree, FIX_ROOT, True)
