"""변환(quant) 인스턴스 기동도 스스로 정지. boot-quant.sh(SB 로 들어와 Args 가 sha 대조 뒤 돌림)와 run-convert.sh 의 HALT.

파드 boot 시험(② tests/test_cloud.py TPodHalt)과 같은 방식: sleep·curl 은 가짜(부른 것만 적음), 맥에 없는 GNU 꼴은 얇은 감싸개.
"""

import base64
import gzip
import hashlib
import importlib.util
import io
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from tests import FIX_ROOT

ROOT = FIX_ROOT / "deploy" / "quant"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MB = _load("make_quant_bundle")


def fakes(d: Path, curl_body: str = "exit 22"):
    (d / "bin").mkdir(exist_ok=True)
    (d / "bin" / "sleep").write_text(f"#!/bin/sh\necho \"sleep $*\" >> {d}/calls\n")
    (d / "bin" / "curl").write_text(
        "#!/bin/sh\n"
        f"echo \"curl $*\" >> {d}/calls\n"
        "for a in \"$@\"; do case \"$a\" in @*) cat \"${a#@}\" >> " + f"{d}/hdr" + ";; esac; done\n"
        "case \"$*\" in *PUT*) printf 200; exit 0;; esac\n" + curl_body + "\n")
    # 맥에는 sha256sum 이 없다(인스턴스는 리눅스) — shasum 으로 대신(-c·--quiet·--strict 만)
    (d / "bin" / "sha256sum").write_text("#!/bin/sh\nargs=\"\"\nfor a in \"$@\"; do case \"$a\" in --quiet|--strict) ;; "
                                         "*) args=\"$args $a\";; esac; done\nexec /usr/bin/shasum -a 256 $args\n")
    (d / "bin" / "base64").write_text("#!/bin/sh\nif [ \"$1\" = -d ] && [ $# -eq 2 ]; then exec /usr/bin/base64 -d -i \"$2\"; fi\n"
                                      "exec /usr/bin/base64 \"$@\"\n")
    for f in ("sleep", "curl", "sha256sum", "base64"):
        (d / "bin" / f).chmod(0o755)


def tgz_b64(files: dict[str, bytes]) -> str:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as t:
        for n, b in files.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    return base64.b64encode(gzip.compress(raw.getvalue())).decode()


class TQuantBoot(unittest.TestCase):
    def run_boot(self, env: dict, curl_body: str = "exit 22"):
        d = Path(tempfile.mkdtemp())
        fakes(d, curl_body)
        boot = (ROOT / "boot-quant.sh").read_text(encoding="utf-8") \
            .replace("/workspace/quant/files", f"{d}/ws").replace("/dev/shm", str(d))
        (d / "boot.sh").write_text(boot)
        e = {"PATH": f"{d}/bin:/usr/bin:/bin", "CONTAINER_ID": "777", "CONTAINER_API_KEY": "INSTKEY9"}
        e.update(env)
        r = subprocess.run(["bash", str(d / "boot.sh")], env=e, capture_output=True, text=True, timeout=30)
        return r, d, ((d / "calls").read_text().splitlines() if (d / "calls").exists() else [])

    def assert_stopped(self, r, d, calls, why):
        self.assertIn("HALT: " + why, r.stdout, (why, r.stdout, r.stderr))
        self.assertEqual(calls[-3], "sleep 900", calls)                     # 이유 한 줄 → 900초
        self.assertIn("-X PUT", calls[-2])
        self.assertIn("console.vast.ai/api/v0/instances/777/", calls[-2])
        self.assertIn('{"state":"stopped"}', calls[-2])
        self.assertNotIn("INSTKEY9", " ".join(calls))                        # 키는 명령 줄이 아니라 헤더 파일로
        self.assertEqual((d / "hdr").read_text(), "Authorization: Bearer INSTKEY9\n")
        self.assertFalse((d / "vast-hdr").exists())
        self.assertEqual(calls[-1], "sleep infinity")
        self.assertIn("HALT stop api HTTP 200", r.stdout)

    def test_five_failures_stop_after_900(self):
        url = "https://gist.example/x"
        cases = [({}, "PU 없음", "exit 22"),                                   # 1 PU 없음
                 ({"PU": url}, "PS 없음", "exit 22"),                          # 2 PS 없음
                 ({"PU": url, "PS": "xyz"}, "PS 없음", "exit 22"),             #   PS 꼴 틀림
                 ({"PU": url, "PS": "0" * 64}, "quant files 받기 실패", "exit 22"),     # 3 받기 실패
                 ({"PU": url, "PS": "0" * 64}, "quant files sha256 불일치", "printf 'QUJD'")]   # 4 sha 불일치
        for env, why, body in cases:
            r, d, calls = self.run_boot(env, body)
            self.assert_stopped(r, d, calls, why)
            self.assertFalse((d / "ws" / "run-convert.sh").exists())
        # 5 풀기 실패·run-convert.sh 없음(대조는 맞지만 묶음이 틀림)
        bad = base64.b64encode(b"not a tgz").decode()
        r, d, calls = self.run_boot({"PU": url, "PS": hashlib.sha256(bad.encode()).hexdigest()}, f"printf '%s' '{bad}'")
        self.assert_stopped(r, d, calls, "quant files 풀기 실패")
        other = tgz_b64({"README.md": b"x"})
        r, d, calls = self.run_boot({"PU": url, "PS": hashlib.sha256(other.encode()).hexdigest()}, f"printf '%s' '{other}'")
        self.assert_stopped(r, d, calls, "run-convert.sh 없음")

    def test_success_runs_run_convert_in_workdir(self):
        # 진짜 run-convert.sh 는 돌아오지 않는다(끝·HALT 모두 스스로 정지 뒤 sleep infinity). 여기 가짜는 돌아오므로 그 뒤 정지까지 본다.
        b64 = tgz_b64({"run-convert.sh": b"echo RUN-CONVERT-RAN $PWD\n"})
        sha = hashlib.sha256(b64.encode()).hexdigest()
        r, d, calls = self.run_boot({"PU": "https://gist.example/x", "PS": sha},
                                    f"printf '%s\\n' '{b64[:40]}'; printf '%s\\r\\n' '{b64[40:]}'")   # 줄바꿈·CR 은 지운다
        self.assertIn(f"RUN-CONVERT-RAN {d}/ws", r.stdout, r.stdout + r.stderr)
        self.assertLess(r.stdout.index("RUN-CONVERT-RAN"), r.stdout.index("HALT"))   # 앞 단계는 모두 통과
        self.assert_stopped(r, d, calls, "run-convert.sh 가 돌아왔다(rc=0)")

    def test_run_convert_returning_stops(self):
        b64 = tgz_b64({"run-convert.sh": b"echo died; exit 7\n"})
        r, d, calls = self.run_boot({"PU": "https://gist.example/x", "PS": hashlib.sha256(b64.encode()).hexdigest()},
                                    f"printf '%s' '{b64}'")
        self.assert_stopped(r, d, calls, "run-convert.sh 가 돌아왔다(rc=7)")

    def test_no_key_skips_api_but_still_sleeps(self):
        r, d, calls = self.run_boot({"CONTAINER_API_KEY": ""})
        self.assertIn("HALT: PU 없음", r.stdout)
        self.assertEqual(calls, ["sleep 900", "sleep infinity"])
        self.assertIn("건너뜀", r.stdout)

    def test_run_convert_halt_waits_then_stops(self):
        """변환 단계 HALT(run-convert.sh 의 halt·ERR trap): 이유 한 줄 → 900초 → 정지. 키 옮기기부터 trap 까지 떼어 돌린다."""
        src = (ROOT / "run-convert.sh").read_text(encoding="utf-8")
        part = src[src.index("umask 077; VKEY="):src.index("# 최대 켜짐 감시")]
        d = Path(tempfile.mkdtemp())
        fakes(d)
        part = part.replace("/dev/shm", str(d))
        script = ("set -euo pipefail\n" + part.replace('Q=${Q:-/workspace/quant}', f"Q={d}/q")
                  .replace("HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"", f"HERE={d}")
                  + "\nfalse\necho NOT-REACHED\n")                                  # 아무 명령 실패 → ERR trap → halt
        r = subprocess.run(["bash", "-c", script], env={"PATH": f"{d}/bin:/usr/bin:/bin", "CONTAINER_ID": "4242",
                                                        "CONTAINER_API_KEY": "QKEY77", "TS_AUTHKEY": "tskey-x"},
                           capture_output=True, text=True, timeout=30)
        self.assertNotIn("NOT-REACHED", r.stdout)
        self.assertIn("HALT: 명령 실패", r.stdout)                                  # 이유 한 줄이 stdout 에(logs 로 보인다)
        calls = (d / "calls").read_text().splitlines()
        self.assertEqual(calls[0], "sleep 900", calls)
        self.assertIn("-X PUT", calls[1])
        self.assertIn("console.vast.ai/api/v0/instances/4242/", calls[1])
        self.assertNotIn("QKEY77", " ".join(calls))
        self.assertEqual((d / "hdr").read_text(), "Authorization: Bearer QKEY77\n")
        self.assertFalse((d / "vast-hdr").exists())
        self.assertEqual(calls[2], "sleep infinity")
        self.assertEqual(len(calls), 3)                                              # halt 안에서 trap 이 다시 돌지 않는다
        log = (d / "q" / "convert-log.tsv").read_text()
        self.assertIn("HALT: 명령 실패", log)
        self.assertIn("stop api HTTP 200", log)

    def test_args_checks_sb(self):
        d = Path(tempfile.mkdtemp())
        fakes(d)
        inner = MB.ARGS[len("bash -c '"):-1].replace("/dev/shm", str(d))
        e = {"PATH": f"{d}/bin:/usr/bin:/bin", "CONTAINER_ID": "5"}
        r = subprocess.run(["bash", "-c", inner], env=dict(e, SB=base64.b64encode(b"echo not-our-boot").decode()),
                           capture_output=True, text=True, timeout=30)
        self.assertIn("HALT: boot SB", r.stdout)                             # sha 가 다른 SB 는 돌리지 않는다
        self.assertNotIn("not-our-boot", r.stdout)
        r = subprocess.run(["bash", "-c", inner], env=e, capture_output=True, text=True, timeout=30)
        self.assertIn("HALT: boot SB", r.stdout)                             # SB 없음
        r = subprocess.run(["bash", "-c", inner], env=dict(e, SB=MB.boot_b64()), capture_output=True, text=True, timeout=30)
        self.assertIn(f"{d}/b: OK", r.stdout)                                # 우리 boot 면 대조 통과 → 실행(작업 폴더에서 HALT)
        self.assertIn("HALT: ", r.stdout)
        self.assertNotIn("HALT: boot SB", r.stdout)

    def test_template_shape(self):
        self.assertLessEqual(len(MB.ARGS), MB.ARGS_MAX)
        self.assertIn(MB.boot_sha(), MB.ARGS)
        self.assertNotIn("sleep 1800", MB.ARGS)
        opts = MB.options("<gist raw 주소>", "a" * 64)
        self.assertLessEqual(len(opts), MB.OPTIONS_MAX)
        self.assertTrue(opts.startswith("-e PU=<gist raw 주소> -e PS=" + "a" * 64 + " -e SB="))
        sb = base64.b64decode(opts.split("-e SB=")[1])
        self.assertEqual(hashlib.sha256(sb).hexdigest(), MB.boot_sha())
        self.assertNotIn(b"\n#", sb)                                          # 주석은 빠졌다
        txt = (ROOT / "boot-quant.sh").read_text()
        for key in ("CONTAINER_API_KEY", "TS_AUTHKEY"):                       # 키 값을 넣는 칸이 없다
            self.assertNotIn(f"-e {key}", opts)
        self.assertNotIn("Bearer $CONTAINER_API_KEY\" ", txt)


if __name__ == "__main__":
    unittest.main()
