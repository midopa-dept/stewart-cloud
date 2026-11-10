"""Vast.ai 인스턴스에 넘길 값 만들기: pod-start.sh·idle_stop.py 를 줄이고(주석·문서 문자열 제거) tar.gz+base64 한 줄로 묶는다.
비밀값은 들어 있지 않다.

  uv run python deploy/vast/make_template_env.py --out 폴더

전달 방식(메인 11/7 18:5x 확정). Vast 한도: Docker Options(환경 변수 포함) 4096자, Args 256자.
- 묶음 한 줄(pod-files.b64)을 사용자의 비밀 gist 에 올리고, 리비전을 고정한 raw 주소를 쓴다.
- 설정 파일(CF): 허용 목록의 KEY=VALUE 만(CTX fast|long, SPEC mtp, PREFIX_CACHE 0|1, FAST_VARIANT 0|1, QUALITY auto|0|1).
- Args(ARGS, T23): Docker Options 의 SB(boot.sh 의 base64)를 /dev/shm/b 로 풀어 sha256 을 대조한 뒤에만 돌린다.
  boot.sh 가 gist 를 받아 줄바꿈을 지우고 sha256(PS)을 대조한 뒤 풀어 pod-start.sh 로 넘긴다. 실패하면 이유 한 줄 → 900초 → 스스로 정지.
  (정지 요청까지 Args 256자에 넣을 수 없어 boot.sh 로 옮겼다. SB 가 없거나 sha 가 다르면 Args 는 "HALT: boot SB" 를 남기고 잔다.)
- Docker Options: -e CTX=fast -e SPEC=mtp -e PU=<gist raw 주소> -e PS=<묶음 sha256 64자> -e CF=<설정 gist 최신 raw 주소> -e SB=<boot.sh base64>
- v7(T20): 역할(STEWART_ROLE serve|eval)·장 수(STEWART_GPUS)는 템플릿이 아니라 stewart-cloud up 이 만들 때(PUT asks env) 넣는다.
  템플릿은 serve·eval 이 하나를 같이 쓴다. Docker Options·설정 파일(CF)에는 역할을 넣지 않는다(없으면 pod 이 HALT).
"""

import argparse
import ast
import base64
import gzip
import hashlib
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARGS_MAX, OPTIONS_MAX = 256, 4096
BOOT = HERE / "boot.sh"


def boot_bytes() -> bytes:
    """boot.sh 를 주석만 뺀 판(Docker Options 4096자 한도 — pod-start 와 같은 slim_sh)."""
    return slim_sh(BOOT.read_text(encoding="utf-8")).encode("utf-8")


def boot_b64() -> str:
    """Docker Options 의 SB 값(줄바꿈 없는 base64)."""
    return base64.b64encode(boot_bytes()).decode()


def boot_sha() -> str:
    return hashlib.sha256(boot_bytes()).hexdigest()


def make_args(sha: str) -> str:
    # Vast Entrypoint 모드의 Args 칸. 작은따옴표: $SB 는 컨테이너 안 bash 가 펼친다. sha 는 boot.sh 의 sha256(여기 고정).
    return (f"bash -c 'echo \"$SB\"|base64 -d>/dev/shm/b&&echo \"{sha}  /dev/shm/b\"|sha256sum -c"
            "&&exec bash /dev/shm/b;echo HALT: boot SB;sleep 1d'")



def options(gist_raw_url: str, sha: str, cf: str = "<설정 gist 최신 raw 주소>", ctx: str = "fast", spec: str = "mtp",
            sb: str | None = None) -> str:
    # CTX·SPEC 는 설정 파일(CF)을 못 받았을 때의 기본값이다. SB = boot.sh(T23).
    # v14: -p 70000:70000/udp — tailscaled 고정 UDP 포트(Vast 가 안팎 같은 번호를 VAST_UDP_PORT_70000 으로 준다)
    return (f"-e CTX={ctx} -e SPEC={spec} -e PU={gist_raw_url} -e PS={sha} -e CF={cf} -e SB={sb if sb is not None else boot_b64()}"
            " -p 70000:70000/udp")


def slim_sh(text: str) -> str:
    out = []
    for i, line in enumerate(text.splitlines()):
        s = line.strip()
        if (i == 0 and s.startswith("#!")) or (s and not s.startswith("#")):
            out.append(line)
    return "\n".join(out) + "\n"


def slim_py(text: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.unparse(tree) + "\n"


SERVING = HERE.parent / "serving"
SERVING_GLOBS = ("*.py", "*.sh", "*.toml", "*.txt", "LICENSE.HyperQwen", "NOTICE", "pins/*.sha256", "prepare/*.py",
                 "bench/*.py", "patches/*.sha256", "patches/*/*", "templates/*.jinja")


def serving_files() -> dict[str, bytes]:
    """deploy/serving 을 serving/ 아래에 바이트 그대로 넣는다. 패치·표·sha 목록은 줄이면 대조가 깨진다."""
    out: dict[str, bytes] = {}
    for g in SERVING_GLOBS:
        for f in sorted(SERVING.glob(g)):
            if f.is_file() and "__pycache__" not in f.parts:
                out["serving/" + f.relative_to(SERVING).as_posix()] = f.read_bytes()
    return out


def bundle() -> bytes:
    files = {"pod-start.sh": slim_sh((HERE / "pod-start.sh").read_text(encoding="utf-8")).encode("utf-8"),
             "idle_stop.py": slim_py((HERE / "idle_stop.py").read_text(encoding="utf-8")).encode("utf-8"),
             **serving_files()}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o755 if name.endswith(".sh") else 0o644, 0
            t.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)


ARGS = make_args(boot_sha())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="값 파일을 쓸 폴더(없으면 화면에만)")
    ap.add_argument("--gist", default="<gist raw 주소>", help="Docker Options 보기용 raw 주소")
    ap.add_argument("--cf", default="<설정 gist 최신 raw 주소>", help="Docker Options 보기용 설정 주소")
    a = ap.parse_args()
    s = base64.b64encode(bundle()).decode()
    sha = hashlib.sha256(s.encode()).hexdigest()
    opts = options(a.gist, sha, a.cf)
    assert len(ARGS) <= ARGS_MAX, len(ARGS)
    print(f"# 묶음 {len(s)}자, sha256 {sha}")
    print(f"# Args({len(ARGS)}/{ARGS_MAX}자): {ARGS}")
    assert len(opts) <= OPTIONS_MAX, len(opts)
    print(f"# Docker Options({len(opts)}/{OPTIONS_MAX}자, SB {len(boot_b64())}자 포함): {opts[:300]}…")
    print(f"# boot.sh sha256 {boot_sha()}")
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "pod-files.b64").write_text(s, encoding="utf-8")
        (o / "pod-files.sha256").write_text(sha + "\n", encoding="utf-8")
        (o / "args.txt").write_text(ARGS, encoding="utf-8")
        (o / "boot.sb.txt").write_text(boot_b64(), encoding="utf-8")                      # Docker Options 의 -e SB= 값
        (o / "docker-options.txt").write_text(options("<gist raw 주소>", sha, a.cf) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
