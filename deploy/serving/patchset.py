"""vLLM 파이썬 패키지 사본(overlay)에 고정 패치를 덧대는 도구. 표준 라이브러리만 쓴다(이미지에 patch 가 없어도 된다).

규칙: GNU patch -p1 --fuzz 0 과 같다. 문맥이 한 글자라도 다르면 실패하고, 줄 위치만 어긋난 것(offset)은 가장 가까운 곳을 받는다.
실패하면 그 패치는 하나도 쓰지 않는다. overlay 는 버리는 사본이라, 묶음 중간에 실패하면 부르는 쪽이 사본을 지운다.

  python3 patchset.py build --src <원래 vllm 폴더> --dst <overlay 루트> --set <패치 폴더> [--unapply-hyperqwen <이미지 /app>]
  python3 patchset.py check --pkg <vllm 폴더> --set <패치 폴더>        # 덧댄 뒤 파일이 applied.sha256 과 같은지(줄인 verify)
  python3 patchset.py pristine --pkg <vllm 폴더> --set <패치 폴더>     # 원본 대조표(pristine.sha256)와 같은지

--unapply-hyperqwen: HyperQwen 8cf642e 이미지 안에서 같은 torch·transformers 로 비교하려고, 이미지의 패치(series 44개 + KVarN)를
거꾸로 풀어 vLLM 0.30.0 원본을 되살린 뒤 우리 묶음을 덧댄다. 되살린 결과는 pristine.sha256 으로 대조하고, 이미지의 패치 파일들은
hyperqwen-8cf642e.sha256 으로 먼저 대조한다(다르면 거부).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    pass


@dataclass
class Hunk:
    old_start: int
    new_start: int
    old: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)


@dataclass
class FilePatch:
    old_path: str | None      # None = 새 파일(/dev/null)
    new_path: str | None      # None = 지우는 파일
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


def _strip(p: str) -> str | None:
    p = p.split("\t")[0].strip()
    if p == "/dev/null":
        return None
    if "/" not in p:
        raise PatchError(f"-p1 로 자를 수 없는 경로: {p}")
    return p.split("/", 1)[1]


def parse(text: str) -> list[FilePatch]:
    """unified diff 를 읽는다. '--- ' 바로 다음 줄이 '+++ ' 일 때만 파일 머리로 본다(앞 설명 글은 건너뜀)."""
    lines = text.splitlines(keepends=True)
    out: list[FilePatch] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            fp = FilePatch(_strip(ln[4:]), _strip(lines[i + 1][4:]))
            i += 2
            while i < len(lines):
                m = HUNK_RE.match(lines[i])
                if not m:
                    break
                old_n = int(m.group(2)) if m.group(2) is not None else 1
                new_n = int(m.group(4)) if m.group(4) is not None else 1
                h = Hunk(int(m.group(1)), int(m.group(3)))
                i += 1
                o = n = 0
                last = None
                while i < len(lines) and (o < old_n or n < new_n or lines[i].startswith("\\")):
                    s = lines[i]
                    if s.startswith("\\"):            # \ No newline at end of file — 앞 줄의 줄바꿈을 지운다
                        for side in last or ():
                            if side and side[-1].endswith("\n"):
                                side[-1] = side[-1][:-1]
                        i += 1
                        continue
                    tag, body = s[:1], s[1:]
                    if tag == " " or s in ("\n", "\r\n"):  # 빈 문맥 줄이 공백 없이 온 경우
                        body = body if tag == " " else s
                        h.old.append(body); h.new.append(body); o += 1; n += 1; last = (h.old, h.new)
                    elif tag == "-":
                        h.old.append(body); o += 1; last = (h.old,)
                    elif tag == "+":
                        h.new.append(body); n += 1; last = (h.new,)
                    else:
                        raise PatchError(f"hunk 줄 형식이 아니다(줄 {i + 1}): {s[:60]!r}")
                    i += 1
                if o != old_n or n != new_n:
                    raise PatchError(f"hunk 길이가 머리와 다르다: {fp.path} @@ -{h.old_start}")
                fp.hunks.append(h)
            out.append(fp)
            continue
        i += 1
    if not out:
        raise PatchError("파일 머리(--- / +++)가 없다")
    return out


def _find(cur: list[str], block: list[str], want: int) -> int | None:
    """block 이 cur 에 정확히 있는 위치 중 want 에 가장 가까운 것(GNU patch 의 offset 탐색)."""
    if not block:
        return max(0, min(want, len(cur)))
    n, L = len(block), len(cur)
    for d in range(0, L + 1):
        for pos in ((want + d, want - d) if d else (want,)):
            if 0 <= pos <= L - n and cur[pos:pos + n] == block:
                return pos
        if want - d < 0 and want + d > L - n:
            break
    return None


def _apply_file(lines: list[str], fp: FilePatch, reverse: bool) -> list[str]:
    cur = list(lines)
    delta = 0
    for h in fp.hunks:
        old, new = (h.new, h.old) if reverse else (h.old, h.new)
        start = max(0, (h.new_start if reverse else h.old_start) - 1)
        want = max(0, start + delta)
        pos = _find(cur, old, want)
        if pos is None:
            raise PatchError(f"{fp.path}: hunk @@ -{h.old_start} 의 문맥이 맞지 않는다(fuzz 0)")
        cur[pos:pos + len(old)] = new
        delta = pos - start + (len(new) - len(old))   # 다음 hunk 는 이번에 찾은 어긋남을 이어받는다
    return cur


def apply_patch(pkg: Path, text: str, reverse: bool = False, dry: bool = False) -> list[str]:
    """패치 하나를 pkg(vllm 폴더) 에 덧댄다. 모든 파일·hunk 가 맞을 때만 쓴다. 바꾼 경로 목록을 돌려준다."""
    plan: list[tuple[Path, list[str] | None]] = []
    for fp in parse(text):
        creates = (fp.old_path is None) if not reverse else (fp.new_path is None)
        deletes = (fp.new_path is None) if not reverse else (fp.old_path is None)
        target = pkg / fp.path
        if creates:
            if target.exists():
                raise PatchError(f"{fp.path}: 새로 만들 파일이 이미 있다")
            lines: list[str] = []
        else:
            if not target.is_file():
                raise PatchError(f"{fp.path}: 고칠 파일이 없다")
            lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        res = _apply_file(lines, fp, reverse)
        if deletes:
            if res:
                raise PatchError(f"{fp.path}: 지울 파일의 내용이 패치와 다르다")
            plan.append((target, None))
        else:
            plan.append((target, res))
    if not dry:
        for target, res in plan:
            if res is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(res), encoding="utf-8")
    return [str(t.relative_to(pkg)) for t, _ in plan]


def is_applied(pkg: Path, text: str) -> bool:
    """거꾸로 덧대기가 맞으면 덧대진 것이다(GNU patch -R --dry-run 과 같은 판정)."""
    try:
        apply_patch(pkg, text, reverse=True, dry=True)
        return True
    except PatchError:
        return False


# ---- 고정 목록·대조 ----

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(p: Path) -> list[tuple[str, str]]:
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        h, name = ln.split(None, 1)
        out.append((h, name.strip()))
    return out


def series(setdir: Path) -> list[str]:
    return [s for s in (re.sub(r"#.*", "", ln).strip() for ln in (setdir / "series").read_text().splitlines()) if s]


def check_set_files(setdir: Path) -> str:
    """패치 폴더의 SHA256SUMS 대조(series·패치·대조표). 묶음 sha(SHA256SUMS 의 sha) 를 돌려준다."""
    sums = setdir / "SHA256SUMS"
    want = dict((n, h) for h, n in read_sums(sums))
    for name in ["series", "pristine.sha256", "applied.sha256", *series(setdir)]:
        if want.get(name) != sha256(setdir / name):
            raise PatchError(f"패치 묶음 파일 sha 불일치 또는 목록에 없음: {name}")
    return sha256(sums)


def check_pristine(pkg: Path, setdir: Path, table: str = "pristine.sha256") -> list[str]:
    """표의 파일마다 sha 대조. pristine.sha256 = 덧대기 전 0.30.0 원본, applied.sha256 = 우리 묶음을 덧댄 뒤."""
    bad = []
    for h, rel in read_sums(setdir / table):
        p = pkg / rel
        got = sha256(p) if p.is_file() else "absent"
        if got != h:
            bad.append(rel)
    return bad


HQ_SERIES_SKIP = {"dflash2-backport.patch"}   # HyperQwen Dockerfile 이 건너뛰는 것
HQ_KVARN = ["kvarn-0.30.0.patch", "kvarn-v2-runner-0.30.0.patch", "kvarn-recycled-pages-0.30.0.patch"]


def unapply_hyperqwen(pkg: Path, app: Path, manifest: Path) -> int:
    """이미지 /app 의 패치 파일을 고정 목록과 대조한 뒤, KVarN → series 순으로 거꾸로 푼다. 푼 패치 수."""
    for h, rel in read_sums(manifest):
        p = app / rel
        if not p.is_file() or sha256(p) != h:
            raise PatchError(f"HyperQwen 이미지 파일이 8cf642e 고정 목록과 다르다: {rel}")
    n = 0
    for k in reversed(HQ_KVARN):
        apply_patch(pkg, (app / "kvarn" / k).read_text(encoding="utf-8"), reverse=True); n += 1
    kfiles = app / "kvarn" / "files" / "vllm"
    for f in sorted(kfiles.rglob("*")):
        if f.is_file():
            t = pkg / f.relative_to(kfiles)
            if t.is_file() and sha256(t) == sha256(f):
                t.unlink()
            else:
                raise PatchError(f"KVarN 파일이 설치본과 다르다: {t}")
    for d in sorted({p.parent for p in kfiles.rglob("*") if p.is_file()}, key=lambda p: -len(p.parts)):
        t = pkg / d.relative_to(kfiles)
        if t.is_dir() and not [x for x in t.iterdir() if x.name != "__pycache__"]:
            shutil.rmtree(t)
    names = [s for s in (re.sub(r"#.*", "", ln).strip() for ln in (app / "patches" / "series").read_text().splitlines()) if s]
    for name in reversed([s for s in names if s not in HQ_SERIES_SKIP]):
        apply_patch(pkg, (app / "patches" / name).read_text(encoding="utf-8"), reverse=True); n += 1
    return n


def build(src: Path, dst_root: Path, setdir: Path, hq_app: Path | None, manifest: Path | None) -> dict:
    """dst_root/vllm 에 사본을 만들고 묶음을 덧댄다. 끝 표시(stewart-overlay.json)가 같은 묶음이면 다시 하지 않는다."""
    set_sha = check_set_files(setdir)
    mark = dst_root / "stewart-overlay.json"
    if mark.is_file():
        old = json.loads(mark.read_text())
        if old.get("set_sha") == set_sha and old.get("src") == str(src) and (dst_root / "vllm").is_dir():
            return {**old, "reused": True}
    tmp = dst_root.with_name(dst_root.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    pkg = tmp / "vllm"
    shutil.copytree(src, pkg, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    unapplied = 0
    if hq_app is not None:
        unapplied = unapply_hyperqwen(pkg, hq_app, manifest)
    bad = check_pristine(pkg, setdir)
    if bad:
        raise PatchError(f"vLLM 원본(0.30.0)과 다른 파일 {len(bad)}개: {', '.join(bad[:5])}")
    applied = []
    for name in series(setdir):
        apply_patch(pkg, (setdir / name).read_text(encoding="utf-8"))
        applied.append(name)
    after = check_pristine(pkg, setdir, "applied.sha256")
    if after:
        raise PatchError(f"덧댄 뒤 sha 가 applied.sha256 과 다른 파일 {len(after)}개: {', '.join(after[:5])}")
    info = {"set": setdir.name, "set_sha": set_sha, "src": str(src), "hyperqwen_unapplied": unapplied,
            "applied": applied, "pristine_files": len(read_sums(setdir / "pristine.sha256")), "reused": False}
    (tmp / "stewart-overlay.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    if dst_root.exists():
        shutil.rmtree(dst_root)
    os.replace(tmp, dst_root)
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--dst", required=True)
    b.add_argument("--set", required=True)
    b.add_argument("--unapply-hyperqwen", metavar="APP")
    for name in ("check", "pristine"):
        c = sub.add_parser(name)
        c.add_argument("--pkg", required=True)
        c.add_argument("--set", required=True)
    a = ap.parse_args(argv)
    setdir = Path(a.set)
    try:
        if a.cmd == "build":
            hq = Path(a.unapply_hyperqwen) if a.unapply_hyperqwen else None
            info = build(Path(a.src), Path(a.dst), setdir, hq, setdir.parent / "hyperqwen-8cf642e.sha256" if hq else None)
            print(json.dumps(info, ensure_ascii=False))
        elif a.cmd == "check":
            check_set_files(setdir)
            pkg = Path(a.pkg)
            bad = check_pristine(pkg, setdir, "applied.sha256")
            n = len(read_sums(setdir / "applied.sha256"))
            print(f"  {'PASS' if not bad else 'FAIL'}  {setdir.name}: 덧댄 뒤 파일 {n - len(bad)}/{n} 이 applied.sha256 과 같다"
                  + (f" (다름: {', '.join(bad[:5])})" if bad else ""))
            return 0 if not bad else 1
        else:
            bad = check_pristine(Path(a.pkg), setdir)
            print(json.dumps({"pristine": not bad, "different": bad}, ensure_ascii=False))
            return 0 if not bad else 1
    except (PatchError, OSError) as e:
        print(f"patchset: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
