"""run-convert.sh 실물을 처음부터 끝까지 돈다(v13 전체 검토).

떼어 낸 함수 블록이 아니라 묶음을 풀어 놓은 꼴 그대로($Q/files/run-convert.sh, targets.json·pins·SHA256SUMS·resume.json)를
bash 로 돌린다. 갈래·차례·단계 표시·내보내기·HALT 는 실물이다. 바꾼 것은 망·GPU 가 드는 호출뿐이다(아래 FAKES).
집 쪽은 deploy/modelstore/modelpush.py 의 pull·promote 실물로 받는다(127.0.0.1 — 테일넷 대신).

대역(실물 아님)
- $APP(/app 대신): venv/bin/python = 실제 파이썬으로 넘기되 kl_check·quantize_autoround·graft_missing·audit_delta·
  quality_battery·spec_metrics 만 가짜(fake_tools.py), venv/bin/hf = 고정 작은 모델 복사, venv/bin/pip = 부르면 실패,
  docker/prepare.sh·verify.sh·single-user/start_qwen.sh = 가짜. $SHM(/dev/shm 대신) = 임시 폴더.
- PATH: sleep(60초 → 한 틱, 3600초 넘으면 진짜로 오래), curl(127.0.0.1:8000 만 가짜 성공, 127.0.0.1 의 다른 포트는 진짜, 밖은 막힘),
  nvidia-smi·free·df(GNU 꼴 출력). 맥에 없는 GNU 도구(sha256sum·find -printf·xargs -d·stat -c)는 같은 뜻의 대역.
- tailscale: $Q/bin 에 판 표시(.ts-version)와 가짜 tailscale·tailscaled 를 미리 둔다(받기·sha 대조 갈래는 안 탄다).
- 1-tools·3-data·3b-offline 은 미리 표시한다(고정 sha 의 망 파일이라 대역으로 못 탄다).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from contextlib import redirect_stdout
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.test_quant import ROOT, MB, make_model

MP_PATH = ROOT.parent / "modelstore" / "modelpush.py"
REAL = sys.executable
ORIG, DEMO, DEMO2 = "qwen38-27b-orig-w4a16q", "qwen38-27b-demo-w4a16q", "qwen38-27b-demo2-w4a16"
PILE_REV = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"


def _mp():
    import importlib.util
    spec = importlib.util.spec_from_file_location("modelpush_rc", MP_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MP = _mp()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gnu(tool: str, arg: str = "--version") -> bool:
    """스크립트가 도는 PATH(/usr/bin:/bin)에 GNU 꼴 도구가 있나."""
    exe = shutil.which(tool, path="/usr/bin:/bin")
    if not exe:
        return False
    try:
        return subprocess.run([exe, arg], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


FAKE_TOOLS = r'''
import json, os, shutil, sys, time
from pathlib import Path
tool = os.path.basename(sys.argv[1]); a = sys.argv[2:]
rig = Path(os.environ["RIG"])
with open(rig / "calls.log", "a") as f:
    f.write(tool + " " + " ".join(a) + "\n")
def opt(k, d=None):
    return a[a.index(k) + 1] if k in a else d
fail = os.environ.get("RIG_FAIL", "")
def alias_of(p):
    parts = Path(p).parts
    return parts[parts.index("t") + 1] if "t" in parts else "?"
if tool == "kl_check.py":
    out = Path(opt("--out"))
    if a[0] == "dump":
        out.write_bytes(b"ref"); print('{"dump": 8}')
    else:
        j = {"mean_kl": 6.57e-4, "top1_agree": 0.98975} if "kl-pad" in out.name else {"mean_kl": 0.0375, "top1_agree": 0.917}
        out.write_text(json.dumps(j)); print(json.dumps(j))
elif tool == "quantize_autoround.py":
    src, out = Path(opt("--src")), Path(opt("--out"))
    al = alias_of(out)
    if fail == al + ":quantize":
        print("quantize 가짜 실패"); sys.exit(1)
    want = [x for x in os.environ.get("RIG_QUANT_WAIT", "").split(",") if x.startswith(al + ":")]
    for w in want:                                   # 이 대상을 양자화하는 동안 집이 앞 세트를 받는다(끝난 대상부터)
        s = w.split(":", 1)[1]
        q = out.parents[2]
        t0 = time.time()
        while time.time() - t0 < 90:
            try:
                if s in json.loads((q / "done" / "exported-sets.json").read_text()):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        with open(rig / "calls.log", "a") as f:
            f.write(f"quant-wait {al} {s} {round(time.time() - t0, 1)}s\n")
    shutil.copytree(src, out)
    for x in ("pad_report.json",):
        (out / x).unlink(missing_ok=True)
    Path(opt("--recipe")).write_text(json.dumps({"scheme": "W4A16", "fake": True}))
    print("quantize ok(가짜)")
elif tool == "graft_missing.py":
    print(json.dumps({"grafted": 0, "verified_existing": 0, "missing_after": 0}))
elif tool == "audit_delta.py":
    j = {"changed": 70, "same": 1129, "changed_layers": "17-51", "mtp_changed": 0, "visual_changed": 0}
    Path(opt("--out")).write_text(json.dumps(j)); Path(opt("--tsv")).write_text("k\tv\n"); print(json.dumps(j))
elif tool == "quality_battery.py":
    qd = Path(os.environ["QUALITY_DATA"]); qd.mkdir(parents=True, exist_ok=True)
    (qd / f"result_{a[0]}.json").write_text(json.dumps({"gsm8k": {"acc": 0.9}})); print("quality ok(가짜)")
elif tool == "spec_metrics.py":
    sys.stdin.read(); print(json.dumps({"spec_decode": {"accept_rate": 0.6}}))
else:
    print("모르는 도구", tool); sys.exit(2)
'''


class Rig:
    """빈 인스턴스 하나. root/ws 가 /workspace 노릇을 한다."""

    def __init__(self, script: Path | None = None, resume: dict | None = None, wait_min: int = 3):
        self.d = Path(tempfile.mkdtemp(prefix="rc-"))
        self.ws, self.app, self.shm, self.bin = self.d / "ws", self.d / "app", self.d / "shm", self.d / "bin"
        self.q, self.imq, self.here = self.ws / "quant", self.ws / "import", self.ws / "quant" / "files"
        self.home = self.d / "home"
        for p in (self.app, self.shm, self.bin, self.here, self.home):
            p.mkdir(parents=True)
        self.sp, self.ep = free_port(), free_port()
        self.rnd = random.Random(11)
        self.fixtures = {}
        self._apps()
        self._shims()
        self._bundle(script or ROOT / "run-convert.sh", resume, wait_min)
        self._pod_state()

    # ---- 고정 작은 모델(HF 대역) ----
    def model(self, repo: str) -> Path:
        if repo not in self.fixtures:
            p = self.d / "hf" / repo.replace("/", "--")
            make_model(p, self.rnd)
            self.fixtures[repo] = p
        return self.fixtures[repo]

    def pins_for(self, repo: str) -> bytes:
        p = self.model(repo)
        return "".join(f"{sha(f)}  {f.name}\n" for f in sorted(p.iterdir())).encode()

    def _apps(self):
        v = self.app / "venv" / "bin"
        v.mkdir(parents=True)
        (self.d / "fake_tools.py").write_text(FAKE_TOOLS)
        (v / "python").write_text(
            "#!/bin/bash\n"
            'case "$(basename -- "${1:-}")" in\n'
            "  kl_check.py|quantize_autoround.py|graft_missing.py|audit_delta.py|quality_battery.py|spec_metrics.py)\n"
            f'    exec {REAL} "$RIG/fake_tools.py" "$@" ;;\n'
            "esac\n"
            f'exec {REAL} "$@"\n')
        (v / "hf").write_text(
            f"#!{REAL}\n"
            "import os, shutil, sys\n"
            "a = sys.argv[1:]\n"
            "open(os.path.join(os.environ['RIG'], 'calls.log'), 'a').write('hf ' + ' '.join(a) + '\\n')\n"
            "assert a[0] == 'download', a\n"
            "src = os.path.join(os.environ['RIG'], 'hf', a[1].replace('/', '--'))\n"
            "dst = a[a.index('--local-dir') + 1]\n"
            "shutil.copytree(src, dst, dirs_exist_ok=True)\n")
        (v / "pip").write_text("#!/bin/sh\necho \"pip 는 시험에서 부르지 않는다: $*\" >&2\nexit 1\n")
        (self.app / "docker").mkdir()
        (self.app / "docker" / "prepare.sh").write_text(
            '#!/bin/bash\necho "prepare: int8 head(가짜)"\n'
            'echo \'{"draft": "fake"}\' > "$BASE_MODEL_DIR/stewart_prepared.json"\n'
            'echo "{}" > "$BASE_MODEL_DIR/config.json.bak-mtp"\necho e > "$BASE_MODEL_DIR/x.safetensors.bak_embed"\n'
            'echo "prepare: model ready"\n')
        (self.app / "verify.sh").write_text('#!/bin/bash\necho "verify: OK (0 failures)"\n')
        (self.app / "single-user").mkdir()
        (self.app / "single-user" / "start_qwen.sh").write_text("#!/bin/bash\nexec /bin/sleep 300\n")
        (self.app / "prepare").mkdir()
        (self.app / "prepare" / "draft_vocab_ids.json").write_text("[1, 2, 3]")
        for p in (*v.iterdir(), self.app / "docker" / "prepare.sh", self.app / "verify.sh",
                  self.app / "single-user" / "start_qwen.sh"):
            p.chmod(0o755)

    def _shims(self):
        b = self.bin
        sh = {
            "sleep": '#!/bin/bash\ncase "$1" in infinity) exit 0;; esac\nn=${1%%.*}\n'
                     'if [ "${n:-0}" -ge 3600 ]; then exec /bin/sleep 3600; fi\n'
                     'echo tick >> "$RIG/ticks"\n[ -x "$RIG/hook" ] && "$RIG/hook"\nexec /bin/sleep "${RIG_TICK:-0.05}"\n',
            "curl": '#!/bin/bash\ncase "$*" in\n  *127.0.0.1:8000/health*) exit 0;;\n  *127.0.0.1:8000/metrics*) echo "# fake"; exit 0;;\n'
                    '  *http://127.0.0.1:*) exec /usr/bin/curl --noproxy "*" "$@";;\nesac\n'
                    'echo "curl 막힘(시험): $*" >> "$RIG/blocked.log"\nexit 7\n',
            "nvidia-smi": '#!/bin/sh\necho "NVIDIA A100-SXM4-80GB(가짜), 81920 MiB"\n',
            "free": '#!/bin/sh\necho "              total        used"\necho "Mem:           503         1"\n',
            "df": '#!/bin/sh\necho " Avail"\necho "  300G"\n',
            "setsid": '#!/bin/sh\nexec "$@"\n',
        }
        if not gnu("sha256sum"):
            sh["sha256sum"] = ('#!/bin/sh\nargs=""\nfor a in "$@"; do case "$a" in --quiet|--strict) ;; *) args="$args $a";; esac; done\n'
                               'exec /usr/bin/shasum -a 256 $args\n')
        if not gnu("find"):
            sh["find"] = (f"#!{REAL}\n"
                          "import os, sys\n"
                          "a = sys.argv[1:]\n"
                          "if '-printf' not in a:\n"
                          "    os.execv('/usr/bin/find', ['find'] + a)\n"
                          "assert a[:3] == ['.', '-maxdepth', '1'] and a[-2:] == ['-printf', '%P\\\\n'], a\n"
                          "for n in sorted(os.listdir('.')):\n"
                          "    if os.path.isfile(n) and not n.startswith('.') and '.bak' not in n:\n"
                          "        print(n)\n")
        if not gnu("xargs", "--help"):
            sh["xargs"] = (f"#!{REAL}\n"
                           "import os, sys\n"
                           "a = sys.argv[1:]\n"
                           "if a[:2] != ['-d', '\\\\n']:\n"
                           "    os.execv('/usr/bin/xargs', ['xargs'] + a)\n"
                           "items = [x for x in sys.stdin.read().split('\\n') if x]\n"
                           "os.execvp(a[2], a[2:] + items)\n")
        if not gnu("stat"):
            sh["stat"] = (f"#!{REAL}\n"
                          "import os, sys\n"
                          "a = sys.argv[1:]\n"
                          "assert a[:2] == ['-c', '%Y'], a\n"
                          "print(int(os.stat(a[2]).st_mtime))\n")
        for n, body in sh.items():
            (b / n).write_text(body)
            (b / n).chmod(0o755)

    def _bundle(self, script: Path, resume: dict | None, wait_min: int):
        h = self.here
        files = {f: (ROOT / f).read_bytes() for f in MB.FILES if f != "image-8cf642e.sha256"}
        files["run-convert.sh"] = script.read_bytes()
        files |= {k: v.read_bytes() for k, v in MB.EXTRA.items()}
        tpl = "templates/fake-chat.jinja"
        files[tpl] = b"{# fake template #}\n"
        targets = []
        for alias, repo, pad in ((ORIG, "Qwen/Qwen3.8-27B", 0), (DEMO, "example-org/Demo-Qwen3.8-27B", 0),
                                 (DEMO2, "example-org/Demo2-Qwen3.8-27B", 128)):
            pins = f"pins/{alias}.sha256"
            files[pins] = b"# fake pins\n" + self.pins_for(repo)
            targets.append({"alias": alias, "repo": repo, "rev": "0" * 40, "pins": pins, "pad_to": pad,
                            "chat_template": tpl if alias != DEMO else ""})
        files["targets.json"] = json.dumps({"targets": targets, "audit": [ORIG, DEMO]}).encode()
        files["home.json"] = json.dumps({"server_ip": "100.64.0.1"}).encode()
        if resume is not None:
            files["resume.json"] = json.dumps({**resume, "import": str(self.imq), "wait_min": wait_min}).encode()
        img = self.app / "verify.sh"           # 이미지 대조: 절대 경로 목록(실물은 / 아래 상대 경로)
        files["image-8cf642e.sha256"] = f"{sha(img)}  {img}\n".encode()
        files["SHA256SUMS"] = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(files.items())).encode()
        for n, b in files.items():
            (h / n).parent.mkdir(parents=True, exist_ok=True)
            (h / n).write_bytes(b)
        (h / "run-convert.sh").chmod(0o755)

    def _pod_state(self):
        qb = self.q / "bin"
        qb.mkdir(parents=True)
        (qb / ".ts-version").write_text("1.104.5\n")
        (qb / "tailscale").write_text(
            '#!/bin/bash\nfor a in "$@"; do case "$a" in --socket=*) shift;; esac; done\n'
            'echo "tailscale $*" >> "$RIG/calls.log"\n'
            'case "$1" in\n'
            '  status) [ -e "$RIG/ts-up" ] || exit 1; case "$*" in *--json*) echo \'{"Self": {"DNSName": "stewart-quant-1.test."}}\';; esac;;\n'
            '  up) [ -n "${RIG_TS_NO_SSH:-}" ] && [[ "$*" == *--ssh* ]] && exit 1; touch "$RIG/ts-up";;\n'
            'esac\nexit 0\n')
        (qb / "tailscaled").write_text('#!/bin/sh\necho "tailscaled $* PRETEND=${TS_DEBUG_PRETENDPOINT:-}" >> "$RIG/calls.log"\nexit 0\n')
        for p in (qb / "tailscale", qb / "tailscaled"):
            p.chmod(0o755)
        for m in ("1-tools", "3-data", "3b-offline"):
            (self.q / "done").mkdir(parents=True, exist_ok=True)
            (self.q / "done" / m).touch()

    # ---- 다른 인스턴스의 /workspace/quant 꼴(끝난 대상의 세트·10단계 sha·단계 표시·감사) ----
    def source_tree(self, root: Path, aliases=(ORIG, DEMO), junk: bool = True):
        for al in aliases:
            t = root / "t" / al
            make_model(t / "out-bf16heads", self.rnd)
            shutil.copytree(t / "out-bf16heads", t / "out-int8heads")
            (t / "out-int8heads" / "stewart_prepared.json").write_text('{"draft": "fake"}')
            (t / "out-int8heads" / "config.json.bak-mtp").write_text("{}")            # 실물: prepare 백업이 세트·sha 목록에 남았다
            (t / "out-int8heads" / "model-00001-of-00002.safetensors.bak_embed").write_bytes(b"e" * 2000)
            for d in ("out-bf16heads", "out-int8heads"):
                (t / f"{d}.sha256").write_text("".join(f"{sha(f)}  {f.name}\n" for f in sorted((t / d).iterdir())))
            (t / "verify.log").write_text("verify: OK (0 failures)\n")
            (t / "quality.log").write_text("quality ok\n")
            (t / "kl-quant-bf16heads.json").write_text(json.dumps({"mean_kl": 0.0375, "top1_agree": 0.917}))
            (t / "result_convert-x.json").write_text("{}")
            (t / "recipe.json").write_text("{}")
            for m in ("2-src", "6-quant", "7-kl-quant", "8-prepare", "9-serve-quality", "10-pins", "11-built"):
                (root / "done" / al).mkdir(parents=True, exist_ok=True)
                (root / "done" / al / m).touch()
        (root / "done" / "audit").touch()
        (root / "audit-delta.json").write_text(json.dumps({"changed": 70}))
        (root / "audit-delta.tsv").write_text("k\tv\n")
        for j in ("files/run-convert.sh", "ts-state/tailscaled.state", "bin/.ts-version", "convert-log.tsv") if junk else ():
            (root / j).parent.mkdir(parents=True, exist_ok=True)
            (root / j).write_text("옛 인스턴스의 것\n")

    def copy_hook(self, stage: Path, per_tick: int = 3):
        """콘솔 복사 흉내: 틱(=1분)마다 rsync 이름 차례로 몇 개씩. 큰 파일은 숨은 임시 이름으로 쓴 뒤 다음 틱에 이름을 바꾼다."""
        order = [p.relative_to(stage) for p in sorted(stage.rglob("*")) if p.is_file()]
        (self.d / "copy-order.json").write_text(json.dumps([str(p) for p in order]))
        hook = self.d / "hook"
        hook.write_text(
            f"#!{REAL}\n"
            "import json, os, shutil\n"
            "from pathlib import Path\n"
            f"stage, dst, rig = Path({str(stage)!r}), Path({str(self.imq)!r}), Path({str(self.d)!r})\n"
            "order = json.loads((rig / 'copy-order.json').read_text())\n"
            "st = rig / 'copy-state.json'\n"
            "i, pend = json.loads(st.read_text()) if st.exists() else (0, None)\n"
            "if pend:\n"
            "    tmp = dst / pend[1]\n"
            "    tmp.rename(dst / pend[0]); pend = None\n"
            f"n = 0\n"
            f"while i < len(order) and n < {per_tick} and not pend:\n"
            "    rel = order[i]; i += 1; n += 1\n"
            "    s, d = stage / rel, dst / rel\n"
            "    d.parent.mkdir(parents=True, exist_ok=True)\n"
            "    if s.stat().st_size > 1000:\n"
            "        tmp = d.parent / ('.' + d.name + '.Xy12')\n"
            "        shutil.copyfile(s, tmp); pend = [rel, str(tmp.relative_to(dst))]\n"
            "    else:\n"
            "        shutil.copyfile(s, d)\n"
            "st.write_text(json.dumps([i, pend]))\n"
            "if i >= len(order) and not pend:\n"
            "    (rig / 'copy-done').touch()\n")
        hook.chmod(0o755)

    # ---- 돌리기 + 집(pull) ----
    def run(self, extra_env: dict | None = None, timeout: int = 240, pull: bool = True):
        env = {"PATH": f"{self.bin}:/usr/bin:/bin", "RIG": str(self.d), "Q": str(self.q), "QUANT_APP": str(self.app),
               "QUANT_SHM": str(self.shm), "QUANT_STATUS_PORT": str(self.sp), "QUANT_EXPORT_PORT": str(self.ep),
               "TMPDIR": str(self.d), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
               "TS_AUTHKEY": "tskey-auth-시험값", "EXPORT_WAIT_MIN": "40", **(extra_env or {})}
        self.out = self.d / "stdout.log"
        self.pulls: list[dict] = []
        stop = threading.Event()
        cfg = MP.load_config(None)
        cfg.update({"root": str(self.home), "min_rate_Bps": 1, "http_timeout_s": 5})

        def home():
            k = 0
            while not stop.is_set():
                try:
                    code, idx = MP.get_json("127.0.0.1", self.ep, "/export", 3)
                except Exception:      # 아직 안 떴다
                    code = None
                if code == 200:
                    k += 1
                    dest = Path(cfg["root"]) / "incoming" / f"pull{k}"
                    try:
                        MP.pull(cfg, "127.0.0.1", into=dest, port=self.ep)
                        self.pulls.append({"dest": dest, "sets": [s["alias"] for s in json.loads((dest / "export.json").read_text())["sets"]]})
                    except Exception as e:  # noqa: BLE001
                        self.pulls.append({"error": repr(e)})
                stop.wait(0.2)

        th = threading.Thread(target=home, daemon=True)
        if pull:
            th.start()
        with open(self.out, "w") as fo:
            pr = subprocess.Popen(["bash", str(self.here / "run-convert.sh")], cwd=self.here, env=env, stdout=fo,
                                  stderr=subprocess.STDOUT, start_new_session=True)
            try:
                self.rc = pr.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.rc = "timeout"
            finally:
                stop.set()
                try:
                    os.killpg(pr.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        th.join(timeout=10) if pull else None
        self.log = self.out.read_text()
        self.cfg = cfg
        return self

    def env(self, extra: dict | None = None) -> dict:
        return {"PATH": f"{self.bin}:/usr/bin:/bin", "RIG": str(self.d), "Q": str(self.q), "QUANT_APP": str(self.app),
                "QUANT_SHM": str(self.shm), "QUANT_STATUS_PORT": str(self.sp), "QUANT_EXPORT_PORT": str(self.ep),
                "TMPDIR": str(self.d), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "EXPORT_WAIT_MIN": "40", **(extra or {})}

    def start_bg(self, extra_env: dict | None = None):
        """run-convert 를 띄우고 돌려준다(끝을 기다리지 않음). 끝에 kill_all."""
        self.bg_out = open(self.d / "bg.log", "w")
        self.bg = subprocess.Popen(["bash", str(self.here / "run-convert.sh")], cwd=self.here,
                                   env=self.env({"TS_AUTHKEY": "tskey-auth-시험값", **(extra_env or {})}), stdout=self.bg_out,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        return self.bg

    def wait_log(self, needle: str, timeout: float = 120) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            p = self.q / "convert-log.tsv"
            if p.exists() and needle in p.read_text():
                return True
            time.sleep(0.2)
        return False

    def kill_all(self):
        try:
            os.killpg(self.bg.pid, signal.SIGKILL)
        except (ProcessLookupError, AttributeError):
            pass
        subprocess.run(["pkill", "-KILL", "-f", str(self.here)], capture_output=True)   # setsid 로 떠난 새 판·자식들

    def script(self, name: str, extra_env: dict | None = None, timeout: int = 120):
        return subprocess.run(["bash", str(self.here / name)], cwd=self.here, env=self.env(extra_env), capture_output=True,
                              text=True, timeout=timeout)

    def promote(self, dest: Path, alias: str):
        return MP.promote(self.cfg, dest, alias)


def tail(text: str, n: int = 60) -> str:
    return "\n".join(text.splitlines()[-n:])


class TRunConvert(unittest.TestCase):
    maxDiff = None

    def assertInLog(self, s, r: Rig):
        self.assertIn(s, r.log, "\n--- 로그 끝 ---\n" + tail(r.log))

    def test_resume_copy_then_export_then_demo2(self):
        """c 길: 새 인스턴스 → 콘솔 복사가 몇 분에 걸쳐 들어옴 → 대조 → 옮김 → 바로 내보냄(집이 받음) → demo2 → 둘째 내보냄 → 정지."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        stage = r.d / "src-instance"
        r.source_tree(stage)
        r.copy_hook(stage, per_tick=3)
        r.run({"RIG_QUANT_WAIT": f"{DEMO2}:{ORIG}-bf16h"})
        log = r.log
        ticks = len((r.d / "ticks").read_text().splitlines())
        self.assertTrue((r.d / "copy-done").exists())
        self.assertGreater(ticks, 3 + 3)                       # 복사가 기다림 한도(3분)보다 오래 걸렸다 — 그래도 HALT 가 아니다
        for s in ("done 0-image",
                  "tailscale ssh 켬(셸 비상구)",
                  "tailscale name stewart-quant-1.test",
                  f"이어 하기: {r.imq} 로 복사가 들어오기를 기다린다",
                  f"이어 하기: 대상 {ORIG} {DEMO} 의 세트가 10단계 sha 와 모두 같다 — 옮기고 바로 내보낸다",
                  "내보내기 갱신: 받지 않은 세트",
                  f"[{DEMO2}] 시작",
                  f'"ok": true',                                  # pad_check 실물(바이트 증명)
                  f"done {DEMO2}/5-kl-pad",
                  f"done {DEMO2}/11-built",
                  "변환 끝.",
                  "받기 대기 끝(받지 않은 세트: )",
                  "stop api 건너뜀"):
            self.assertInLog(s, r)
        self.assertNotIn("HALT:", log)
        self.assertNotIn("command not found", log)
        self.assertEqual(r.rc, 0, tail(log))
        ok = [p for p in r.pulls if "sets" in p]
        self.assertEqual([p.get("error") for p in r.pulls if "error" in p], [])
        self.assertEqual(ok[0]["sets"], [f"{ORIG}-bf16h", f"{ORIG}-int8h", f"{DEMO}-bf16h", f"{DEMO}-int8h"])
        self.assertEqual(ok[-1]["sets"], [f"{DEMO2}-bf16h", f"{DEMO2}-int8h"])
        # 집의 promote 실물: 대조 통과본은 들어가고, demo2 는 0 채우기 KL 문턱(1e-4)에 걸린다(열린 결정 a)
        for s in ok[0]["sets"]:
            r.promote(ok[0]["dest"], s)
        with self.assertRaises(MP.StoreError) as e:
            r.promote(ok[-1]["dest"], f"{DEMO2}-bf16h")
        self.assertIn("0 채우기 KL", str(e.exception))
        # 옮긴 세트가 원본 인스턴스의 것과 바이트로 같다
        for al in (ORIG, DEMO):
            for d in ("out-bf16heads", "out-int8heads"):
                for f in (stage / "t" / al / d).iterdir():
                    self.assertEqual(sha(f), sha(r.q / "t" / al / d / f.name))
        self.assertIn(f"hf download example-org/Demo2-Qwen3.8-27B", (r.d / "calls.log").read_text())
        self.assertNotIn("hf download Qwen/", (r.d / "calls.log").read_text())     # 끝난 대상은 다시 받지 않는다
        self.assertFalse((r.d / "blocked.log").exists())
        print("\n--- c 길 로그 끝 ---\n" + tail(log, 45))

    def test_resume_no_copy_halts_and_holds(self):
        """복사가 오지 않으면: 들어오는 것 없이 3분 → HALT → 셸 비상구로 60분(여기서는 2틱) 기다림 → 정지."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        r.run({"HALT_HOLD_MIN": "2", "HALT_HOLD_MAX_MIN": "5"}, pull=False)
        self.assertInLog("HALT: 이어 하기: 들어오는 것 없이 3분", r)
        self.assertInLog("셸 비상구: 집에서 tailscale ssh root@stewart-quant-1.test 로 들어온다", r)
        self.assertInLog("셸 비상구: 2분 기다림 끝 — 정지", r)
        self.assertInLog("stop api 건너뜀", r)
        self.assertFalse((r.q / "export.json").exists())
        self.assertEqual(r.rc, 0)

    def test_halt_hold_then_manual_export(self):
        """이번 일의 손 절차: HALT 뒤 비상구로 들어와 export.json 을 손으로 만든다 → 받기로 넘어가 집이 받고 정지."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        r.source_tree(r.q, junk=False)                        # 옮긴 뒤 상태 — 그러나 이어 하기 대상 하나를 일부러 빼서 HALT 시킨다
        shutil.rmtree(r.q / "done" / DEMO)
        hook = r.d / "hook"
        hook.write_text(
            "#!/bin/bash\n"
            f'if grep -q "셸 비상구: 집에서" "{r.q}/convert-log.tsv" && [ ! -e "{r.d}/manual" ]; then\n'
            f'  touch "{r.d}/manual"\n'
            f'  cd "{r.here}" && "{r.app}/venv/bin/python" build_export.py --q "{r.q}" --targets targets.json '
            f'--pile-rev {PILE_REV} --draft-ids "{r.app}/prepare/draft_vocab_ids.json" >> "{r.d}/manual.log" 2>&1\n'
            "fi\n")
        hook.chmod(0o755)
        r.run({"HALT_HOLD_MIN": "30", "HALT_HOLD_MAX_MIN": "60"})
        self.assertInLog("셸 비상구: export.json 이 생겼다 — 받기로 넘어간다", r)
        self.assertInLog("받기 대기 끝(받지 않은 세트: )", r)
        ok = [p for p in r.pulls if "sets" in p]
        self.assertEqual(ok[0]["sets"], [f"{ORIG}-bf16h", f"{ORIG}-int8h"])
        self.assertEqual(r.rc, 0)

    def test_halt_hold_rerun(self):
        """비상구로 들어와 고친 뒤 rerun: 같은 컨테이너에서 처음부터 다시 — 테일넷·quantsrv 는 그대로, 단계 표시로 이어서."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        hook = r.d / "hook"
        hook.write_text(
            "#!/bin/bash\n"
            f'if grep -q "셸 비상구: 집에서" "{r.q}/convert-log.tsv" && [ ! -e "{r.d}/fixed" ]; then\n'
            f'  touch "{r.d}/fixed"\n'
            f'  {REAL} -c "import json,sys; p=sys.argv[1]; j=json.load(open(p)); j.pop(\'first\'); open(p,\'w\').write(json.dumps(j))" '
            f'"{r.here}/resume.json"\n'
            f'  echo \'{{"first": []}}\' > /dev/null; rm -f "{r.here}/resume.json"; touch "{r.q}/rerun"\n'
            "fi\n")
        hook.chmod(0o755)
        r.run({"HALT_HOLD_MIN": "30", "HALT_HOLD_MAX_MIN": "60", "RIG_QUANT_WAIT": f"{DEMO}:{ORIG}-bf16h"})
        self.assertInLog("셸 비상구: rerun — 처음부터 다시 돈다", r)
        self.assertInLog("tailscale 이미 붙어 있다(rerun) — 그대로 쓴다", r)
        self.assertInLog("quantsrv 이미 돈다(rerun) — 그대로 쓴다", r)
        self.assertInLog(f"done {DEMO2}/11-built", r)
        self.assertInLog("받기 대기 끝(받지 않은 세트: )", r)
        self.assertEqual(r.rc, 0, tail(r.log))
        self.assertTrue((r.shm / "ts-authkey").exists() is False or (r.shm / "ts-authkey").stat().st_size == 0)

    def test_fresh_all_targets(self):
        """b 길: resume.json 없는 보통 갈래 — 셋을 처음부터. 끝난 대상부터 집이 받는다(orig → demo → demo2). 감사도 탄다."""
        r = Rig(resume=None)
        r.run({"RIG_QUANT_WAIT": f"{DEMO}:{ORIG}-bf16h,{DEMO2}:{DEMO}-bf16h"})
        for s in (f"done {ORIG}/11-built", "done audit", f"done {DEMO}/11-built", f"done {DEMO2}/11-built",
                  "done 11-export", "받기 대기 끝(받지 않은 세트: )"):
            self.assertInLog(s, r)
        self.assertNotIn("이어 하기", r.log)
        self.assertNotIn("HALT:", r.log)
        ok = [p for p in r.pulls if "sets" in p]
        self.assertEqual([p.get("error") for p in r.pulls if "error" in p], [])
        self.assertEqual(ok[0]["sets"][:2], [f"{ORIG}-bf16h", f"{ORIG}-int8h"])
        got = [s for p in ok for s in p["sets"]]
        self.assertEqual(sorted(got), sorted(f"{a}-{k}" for a in (ORIG, DEMO, DEMO2) for k in ("bf16h", "int8h")))
        self.assertEqual(r.rc, 0, tail(r.log))
        print("\n--- b 길 로그 끝 ---\n" + tail(r.log, 30))

    def test_target_failure_is_contained(self):
        """demo2 양자화가 실패해도: 그 대상만 failed, 끝난 둘은 내보내고 받은 뒤 정지."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        r.source_tree(r.imq)                                  # 복사가 이미 다 와 있다
        r.run({"RIG_FAIL": f"{DEMO2}:quantize"})
        self.assertInLog(f"[{DEMO2}] 명령 실패(줄 ", r)                     # 이유 줄(하위 셸의 ERR trap)
        self.assertInLog(f"[{DEMO2}] 실패(rc=1) — 이 대상만 멈춘다", r)
        ok = [p for p in r.pulls if "sets" in p]
        self.assertEqual(ok[0]["sets"], [f"{ORIG}-bf16h", f"{ORIG}-int8h", f"{DEMO}-bf16h", f"{DEMO}-int8h"])
        self.assertInLog("받기 대기 끝(받지 않은 세트: )", r)
        self.assertTrue((r.q / "done" / DEMO2 / "failed").exists())
        self.assertEqual(r.rc, 0, tail(r.log))

    def test_already_moved_restart(self):
        """옮긴 뒤 다시 켠 인스턴스: 기다리지 않고 바로 내보낸다."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        r.source_tree(r.q, junk=False)
        r.run({"RIG_FAIL": f"{DEMO2}:quantize"})
        self.assertInLog(f"이어 하기: 대상 {ORIG} {DEMO} 는 이미 {r.q} 로 옮겨 있다 — 기다리지 않고 내보낸다", r)
        self.assertNotIn("복사가 들어오기를 기다린다", r.log)
        ok = [p for p in r.pulls if "sets" in p]
        self.assertEqual(ok[0]["sets"], [f"{ORIG}-bf16h", f"{ORIG}-int8h", f"{DEMO}-bf16h", f"{DEMO}-int8h"])

    def test_no_ssh_still_runs(self):
        """tailscale up --ssh 가 안 되면 비상구 없이 붙어 계속 간다(멈추지 않는다)."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        r.source_tree(r.imq)
        r.run({"RIG_TS_NO_SSH": "1", "RIG_FAIL": f"{DEMO2}:quantize"})
        self.assertInLog("tailscale up --ssh 실패 — 셸 비상구 없이 붙는다", r)
        self.assertInLog("tailscale up stewart-quant", r)
        self.assertInLog("받기 대기 끝(받지 않은 세트: )", r)


class TV131(unittest.TestCase):
    """v13.1: 판정 이유 줄·손 내보내기·손으로 이어 가기·고정 UDP 포트."""

    def test_reason_line_when_done_missing(self):
        """복사에 done 이 빠지면 판정이 그 이유를 10분 줄과 마지막 줄에 찍고, 들어오는 것 없이 한도 → HALT."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=14)
        stage = r.d / "src-instance"
        r.source_tree(stage)
        shutil.rmtree(stage / "done")
        r.copy_hook(stage, per_tick=6)
        r.run({"HALT_HOLD_MIN": "1", "HALT_HOLD_MAX_MIN": "2"}, pull=False)
        self.assertIn("판정: 없음: done/", r.log, tail(r.log))
        self.assertIn("이어 하기: 마지막 판정 — 없음: done/", r.log)
        self.assertIn("HALT: 이어 하기: 들어오는 것 없이 14분", r.log)
        self.assertNotIn("10단계 sha 와 모두 같다", r.log)

    def test_manual_export(self):
        """손 내보내기: done 이 없는 받은 폴더 → 대조 → 옮김 → 표시 → export.json(EXPORT-OK). 대조가 어긋나면 멈추고 아무것도 안 옮긴다."""
        r = Rig(resume={"first": [ORIG, DEMO]})
        r.source_tree(r.imq)
        shutil.rmtree(r.imq / "done")
        bad = r.imq / "t" / DEMO / "out-int8heads" / "config.json"
        good = bad.read_bytes()
        bad.write_bytes(b"{}x")                                              # 한 파일이 다르면
        out = r.script("manual-export.sh")
        self.assertNotEqual(out.returncode, 0)
        self.assertNotIn("EXPORT-OK", out.stdout)
        self.assertFalse((r.q / "t").exists() and any((r.q / "t").iterdir()))   # 아무것도 안 옮겼다
        bad.write_bytes(good)
        out = r.script("manual-export.sh")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        lines = out.stdout.splitlines()
        self.assertEqual([l for l in lines if l.startswith("VERIFY-OK")],
                         [f"VERIFY-OK {a} {d}" for a in (ORIG, DEMO) for d in ("out-bf16heads", "out-int8heads")])
        self.assertEqual([l for l in lines if l.startswith("MOVED")], [f"MOVED {ORIG}", f"MOVED {DEMO}"])
        self.assertEqual(lines[-1], "EXPORT-OK")
        idx = json.loads((r.q / "export.json").read_text())
        self.assertEqual([x["alias"] for x in idx["sets"]], [f"{a}-{k}" for a in (ORIG, DEMO) for k in ("bf16h", "int8h")])
        self.assertTrue((r.q / "done" / ORIG / "9-serve-quality").exists())
        self.assertTrue((r.q / "done" / "audit").exists())
        out = r.script("manual-export.sh")                                   # 두 번 치면 옮길 것이 없어 멈춘다(덮지 않음)
        self.assertNotEqual(out.returncode, 0)

    def test_manual_continue(self):
        """손으로 이어 가기: 판정에서 못 나오는 옛 판(복사가 깨짐)을 멈춰 두고, 손 내보내기 뒤 새 판을 띄우면 이미 옮겨 있다 → demo2 → 정지.
        스크립트는 자기·부모를 멈추지 않는다(pkill -f 가 ssh 셸까지 멈춘 일이 있었다)."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=500)
        r.start_bg({"RIG_TICK": "0.2"})
        self.addCleanup(r.kill_all)
        self.assertTrue(r.wait_log("복사가 들어오기를 기다린다", 60), "옛 판이 기다림에 안 들어갔다")
        r.source_tree(r.imq)
        shutil.rmtree(r.imq / "done")                                         # 깨진 done 흉내 — 판정은 끝내 못 지난다
        self.assertEqual(r.script("manual-export.sh").returncode, 0)
        # pgrep 대역: 옛 판 pid 와 부른 쪽(스크립트 자신)의 pid 를 함께 내놓는다 — 스크립트가 자기는 빼야 한다
        (r.bin / "pgrep-fake").write_text(f"#!/bin/sh\necho {r.bg.pid}\necho $QUANT_MANUAL_SELF\necho $QUANT_MANUAL_PARENT\n")
        (r.bin / "pgrep-fake").chmod(0o755)
        out = r.script("manual-continue.sh", {"QUANT_OUT": str(r.d / "new.log"), "CONTINUE_WAIT_S": "3",
                                              "QUANT_PGREP": str(r.bin / "pgrep-fake")})
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(f"STOPPED {r.bg.pid}\n", out.stdout)                   # 자기 pid 는 빠졌다
        self.assertIn("CONTINUE-OK", out.stdout)
        pid, status = os.waitpid(r.bg.pid, os.WUNTRACED | os.WNOHANG)
        self.assertTrue(pid == r.bg.pid and os.WIFSTOPPED(status), (pid, status))   # 옛 판은 멈춤(끄지 않음)
        self.assertTrue(r.wait_log(f"이어 하기: 대상 {ORIG} {DEMO} 는 이미", 60), tail((r.q / "convert-log.tsv").read_text()))
        self.assertTrue(r.wait_log("tailscale 이미 붙어 있다(rerun)", 5))
        self.assertTrue(r.wait_log(f"done {DEMO2}/11-built", 120), tail((r.q / "convert-log.tsv").read_text()))

    def test_fixed_udp_port(self):
        """VAST_UDP_PORT_70000·PUBLIC_IPADDR 가 있으면 그 포트로 tailscaled 를 띄우고 끝점을 광고한다. ts-relay 면 쓰지 않는다."""
        r = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        r.source_tree(r.q, junk=False)
        r.run({"RIG_FAIL": f"{DEMO2}:quantize", "VAST_UDP_PORT_70000": "40123", "PUBLIC_IPADDR": "203.0.113.5"})
        self.assertIn("tailscale 고정 UDP 203.0.113.5:40123(템플릿 -p 70000:70000/udp)", r.log)
        calls = (r.d / "calls.log").read_text()
        self.assertIn("--port=40123", calls)
        self.assertIn("PRETEND=203.0.113.5:40123", calls)
        r2 = Rig(resume={"first": [ORIG, DEMO]}, wait_min=3)
        r2.source_tree(r2.q, junk=False)
        (r2.q / "ts-relay").touch()
        r2.run({"RIG_FAIL": f"{DEMO2}:quantize", "VAST_UDP_PORT_70000": "40123", "PUBLIC_IPADDR": "203.0.113.5"})
        self.assertIn("tailscale UDP 무작위 포트(VAST_UDP_PORT_70000=40123 PUBLIC_IPADDR=203.0.113.5 ts-relay 있음)", r2.log)
        self.assertNotIn("--port=", (r2.d / "calls.log").read_text())


class TRangeResume(unittest.TestCase):
    """v13.1: quantsrv 의 Range(이어 보내기)와 modelpush pull 의 .part 이어 받기·처음 30초 속도 줄."""

    def setUp(self):
        from tests.test_quant import TExport, QS, BE
        self.QS, self.BE, self.TG = QS, BE, TExport.TG
        TExport.setUp(self)
        TExport.build(self)

    def serve(self):
        sp, ep = free_port(), free_port()
        srvs = self.QS.serve(self.QS.Ctx(self.q, "100.64.0.9", None), "127.0.0.1", sp, ep)
        self.addCleanup(lambda: [x.shutdown() for x in srvs])
        return ep

    def test_range(self):
        import http.client
        ep = self.serve()
        alias = json.loads((self.q / "export.json").read_text())["sets"][0]["alias"]
        c = http.client.HTTPConnection("127.0.0.1", ep, timeout=5)
        c.request("GET", f"/export/{alias}/model-00001.safetensors", headers={"Range": "bytes=4000-"})
        r = c.getresponse()
        self.assertEqual((r.status, r.getheader("Content-Range"), r.read()), (206, "bytes 4000-4999/5000", b"w" * 1000))
        for bad in ("bytes=0-", "bytes=5000-", "bytes=1-2", "items=3-"):          # 그 밖의 꼴은 통째(200)
            c = http.client.HTTPConnection("127.0.0.1", ep, timeout=5)
            c.request("GET", f"/export/{alias}/model-00001.safetensors", headers={"Range": bad})
            r = c.getresponse()
            self.assertEqual((r.status, len(r.read())), (200, 5000), bad)

    def test_pull_resumes_part(self):
        ep = self.serve()
        home = Path(tempfile.mkdtemp())
        cfg = MP.load_config(None)
        cfg.update({"root": str(home), "http_timeout_s": 5})
        self.assertEqual((cfg["min_rate_Bps"], cfg["stall_s"]), (0, 300))       # 기본: 평균 문턱 없음, 멈춤 300초
        dest = home / "incoming" / "p1"
        alias = json.loads((self.q / "export.json").read_text())["sets"][0]["alias"]
        (dest / alias).mkdir(parents=True)
        (dest / alias / ".model-00001.safetensors.part").write_bytes(b"w" * 3000)  # 앞서 끊긴 반쪽
        buf = io.StringIO()
        with redirect_stdout(buf):
            MP.pull(cfg, "127.0.0.1", into=dest, port=ep)
        self.assertIn(f"{alias} model-00001.safetensors 이어 받기 3,000B 부터", buf.getvalue())
        self.assertEqual((dest / alias / "model-00001.safetensors").read_bytes(), b"w" * 5000)
        self.assertFalse((dest / alias / ".model-00001.safetensors.part").exists())
        (self.q / "done" / "exported-sets.json").unlink()                       # 다시 내보내게(받았다는 표시를 지움)
        dest2 = home / "incoming" / "p2"                                         # 반쪽이 틀린 바이트면 sha 에서 막고 .part 를 지운다
        (dest2 / alias).mkdir(parents=True)
        (dest2 / alias / ".model-00001.safetensors.part").write_bytes(b"x" * 3000)
        with redirect_stdout(io.StringIO()), self.assertRaises(MP.StoreError) as e:
            MP.pull(cfg, "127.0.0.1", into=dest2, port=ep)
        self.assertIn("sha 가 목록과 다르다", str(e.exception))
        self.assertFalse((dest2 / alias / ".model-00001.safetensors.part").exists())


class TDefOrder(unittest.TestCase):
    """함수·전역 변수가 불리는 줄보다 앞에서 정의되는지(부른 함수가 다시 부르는 것까지, trap 문자열 포함)."""

    def test_run_convert(self):
        from tests import shell_order
        text = (ROOT / "run-convert.sh").read_text()
        self.assertEqual(shell_order.problems(text), [])
        self.assertEqual(shell_order.var_problems(text), [])

    def test_catches_v12_bug(self):
        from tests import shell_order
        v12 = ("set -e\nlog() { echo \"$*\"; }\n"
               "halt() { log x; wait_export; }\ntrap 'halt y' ERR\n"
               "if true; then export_now; fi\n"
               "wait_export() { :; }\nexport_now() { log z; }\n")
        got = shell_order.problems(v12)
        self.assertTrue(any("halt → wait_export" in g for g in got), got)
        self.assertTrue(any("export_now → export_now" in g for g in got), got)


if __name__ == "__main__":
    unittest.main()
