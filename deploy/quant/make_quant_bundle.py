"""변환 인스턴스에 넘길 묶음(quant-files) 만들기. pod-files 와 같은 방식: 묶음 한 줄을 gist 에, Args 가 sha 를 대조한 뒤에만 푼다.

  uv run python deploy/quant/make_quant_bundle.py --out <폴더>

- 넣는 것: deploy/quant 의 스크립트·sha 목록, deploy/serving/bench/quality_battery.py. 바이트 그대로, SHA256SUMS 를 함께.
- 비밀값은 없다. Vast 템플릿: 이미지 HyperQwen 8cf642e(digest 고정), Docker Options `-e PU=<gist raw> -e PS=<묶음 sha> -e SB=<boot>`,
  디스크 ≥ 200 GB.
- 기동은 파드 v8/v9 와 같은 꼴. Args 는 SB(boot-quant.sh 를 주석만 뺀 판의 base64)를 /dev/shm/b 로 풀어 sha256 을 대조한 뒤에만
  돌린다. boot-quant.sh 가 PU·PS·받기·대조·풀기를 하고 run-convert.sh 를 부른다. 실패는 모두 이유 한 줄 → 900초 → 스스로 정지.
  SB 가 없거나 sha 가 다르면(boot 를 못 돌림) Args 는 "HALT: boot SB" 를 남기고 잔다 — 정지 요청까지 256자에 넣을 수 없다(파드와 같다).
"""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import gzip
import hashlib
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["run-convert.sh", "st_io.py", "pad_intermediate.py", "graft_missing.py", "quantize_autoround.py", "kl_check.py",
         "quantsrv.py", "build_export.py", "audit_delta.py", "spec_metrics.py", "requirements-quant.txt", "image-8cf642e.sha256",
         "pins/quality-data.sha256"]          # 대상별 원본 pins 는 서빙 표의 quant_src 에서(targets.json 과 함께)
EXTRA = {"quality_battery.py": HERE.parent / "serving" / "bench" / "quality_battery.py"}
ARGS_MAX, OPTIONS_MAX = 256, 4096
BOOT = HERE / "boot-quant.sh"


def slim_sh(text: str) -> str:
    """주석·빈 줄만 뺀다(첫 줄 #! 는 둔다). 파드 make_template_env.slim_sh 와 같은 규칙 — Docker Options 4096자 한도."""
    out = []
    for i, line in enumerate(text.splitlines()):
        st = line.strip()
        if (i == 0 and st.startswith("#!")) or (st and not st.startswith("#")):
            out.append(line)
    return "\n".join(out) + "\n"


def boot_bytes() -> bytes:
    return slim_sh(BOOT.read_text(encoding="utf-8")).encode("utf-8")


def boot_b64() -> str:
    """Docker Options 의 SB 값(줄바꿈 없는 base64)."""
    return base64.b64encode(boot_bytes()).decode()


def boot_sha() -> str:
    return hashlib.sha256(boot_bytes()).hexdigest()


def make_args(sha: str) -> str:
    # Vast Entrypoint 모드의 Args 칸(파드 v8/v9 와 같은 글). 작은따옴표: $SB 는 컨테이너 안 bash 가 펼친다. sha 는 boot-quant.sh 의 것.
    return (f"bash -c 'echo \"$SB\"|base64 -d>/dev/shm/b&&echo \"{sha}  /dev/shm/b\"|sha256sum -c"
            "&&exec bash /dev/shm/b;echo HALT: boot SB;sleep 1d'")


def options(gist_raw_url: str, sha: str) -> str:
    return f"-e PU={gist_raw_url} -e PS={sha} -e SB={boot_b64()}"


ARGS = make_args(boot_sha())


def table() -> dict:
    """서빙 표(공개 + local, 검사한 것). serving.py 를 같은 저장소에서 읽는다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("serving", HERE.parent / "serving" / "serving.py")
    sv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sv)
    return sv.load(), sv


def server_ip() -> str:
    """보관 서버 테일넷 주소는 서빙 표의 [home].server_ip(local 표) 하나에서 온다."""
    return str((table()[0].get("home") or {}).get("server_ip", ""))


def targets() -> dict:
    """변환 대상(서빙 표의 quant_src, order 차례)과 감사 쌍. 대상마다 채팅 템플릿 고정본 경로(묶음 안 templates/)."""
    t, sv = table()
    out = []
    for x in sv.quant_targets(t):
        ct = t["models"][x["alias"]].get("chat_template") or {}
        out.append({k: x[k] for k in ("alias", "repo", "rev", "pins", "pad_to")} | {"chat_template": ct.get("file", "")})
    return {"targets": out, "audit": (t.get("quant") or {}).get("audit")}


def contents() -> dict[str, bytes]:
    out = {f: (HERE / f).read_bytes() for f in FILES}
    out |= {k: v.read_bytes() for k, v in EXTRA.items()}
    tg = targets()
    for x in tg["targets"]:
        out[x["pins"]] = (HERE / x["pins"]).read_bytes()
        if x["chat_template"]:
            out[x["chat_template"]] = (HERE.parent / "serving" / x["chat_template"]).read_bytes()
    out["targets.json"] = (json.dumps(tg, ensure_ascii=False, indent=1) + "\n").encode()
    out["home.json"] = (json.dumps({"server_ip": server_ip()}) + "\n").encode()
    out["SHA256SUMS"] = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(out.items())).encode()
    return out


def bundle() -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as t:
        for name, data in sorted(contents().items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o755 if name.endswith(".sh") else 0o644, 0
            t.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    ip = server_ip()
    try:
        ok = ipaddress.ip_address(ip) in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        ok = False
    if a.out and not ok:
        raise SystemExit(f"서빙 표의 home.server_ip 가 테일넷 주소가 아니다({ip!r}) — 채운 뒤 묶음을 만든다")
    s = base64.b64encode(bundle()).decode()
    sha = hashlib.sha256(s.encode()).hexdigest()
    opts = options("<gist raw 주소>", sha)
    assert len(ARGS) <= ARGS_MAX, len(ARGS)
    assert len(opts) <= OPTIONS_MAX, len(opts)
    print(f"# 묶음 {len(s)}자, sha256 {sha}")
    print(f"# Args({len(ARGS)}/{ARGS_MAX}자): {ARGS}")
    print(f"# boot-quant.sh(주석 뺀 판) sha256 {boot_sha()}, SB {len(boot_b64())}자")
    print(f"# Docker Options({len(opts)}/{OPTIONS_MAX}자, PU 칸만 사람이 넣는다): {opts[:200]}…")
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "quant-files.b64").write_text(s)
        (o / "quant-files.sha256").write_text(sha + "\n")
        (o / "args.txt").write_text(ARGS)
        (o / "boot.sb.txt").write_text(boot_b64())
        (o / "boot.sha256").write_text(boot_sha() + "\n")
        (o / "docker-options.txt").write_text(opts + "\n")

if __name__ == "__main__":
    main()
