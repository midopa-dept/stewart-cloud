"""보관 서버 모델 보관소: 집이 먼저 연결해 파드에 밀어 넣고(push) 변환 결과를 끌어온다(pull). 파이썬 표준 라이브러리만.

  modelpush.py --config /etc/modelstore/config.json daemon                       # 30초마다 파드들의 /need → catalog 에서 PUT
  modelpush.py --config … push-once                                             # 한 바퀴만(시험·손 확인)
  modelpush.py --config … pull <호스트> [--into <incoming 폴더>] [--no-done]      # 변환 인스턴스의 /export → incoming/<시각>-<호스트>/
  modelpush.py --config … promote <incoming 폴더> <별칭>                         # 검사 통과 → catalog/<별칭>/ + manifest.json
  modelpush.py --config … list

- 빌린 기계 → 집 연결은 없다. 이 프로그램만 파드(tag:gpu)의 :8090 으로 연결한다(Tailscale ACL tag:server → tag:gpu tcp:8090).
- 파드가 고를 수 있는 것은 별칭뿐이다. 보낼 파일은 catalog/<별칭>/manifest.json 의 목록뿐이고, 파드가 말한 목록(이름·sha256·크기)이 manifest 와 같을 때만 보낸다.
- promote 검사
  - 허용 꼴: *.safetensors, *.json, tokenizer.model, tokenizer.json, *.txt, *.md, chat_template.jinja
  - 거부: .py .bin .pt .pth .pkl .pickle 그리고 그 밖 전부
  - safetensors 머리 파싱·크기, json 파싱, 색인과 샤드 맞추기, sha256·크기 다시 대조, 보고 숫자 문턱
- 목록 지문(digest)은 deploy/serving/serving.py home_digest 와 같은 계산이다. 파드의 M_REV = "home-<digest>" 와 맞춰 볼 수 있다.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import struct
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,47}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
JSON_MAX = 1 << 20                  # /need·/export 응답 상한 1 MB
CHUNK = 1 << 20
ALLOWED_EXACT = {"tokenizer.model", "tokenizer.json", "chat_template.jinja"}
ALLOWED_SUFFIX = (".safetensors", ".json", ".txt", ".md")
DENY_SUFFIX = (".py", ".bin", ".pt", ".pth", ".pkl", ".pickle")
ST_DTYPES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
             "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "U32": 4, "U64": 8}
DEFAULTS = {"root": "/srv/models", "hosts": [], "port": 8090, "interval_s": 30, "http_timeout_s": 15,
            "min_rate_Bps": 5_000_000, "max_total_bytes": 200 * 10**9,
            "thresholds": {"pad_mean_kl_max": 1e-4, "pad_top1_min": 0.999, "quant_mean_kl_max": 0.1, "quant_top1_min": 0.85,
                           "require_verify": True, "require_quality": True}}


class StoreError(Exception):
    pass


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t{msg}", flush=True)


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in user.items():
            if k == "thresholds" and isinstance(v, dict):
                cfg["thresholds"].update(v)
            else:
                cfg[k] = v
    for h in cfg["hosts"]:
        if not HOST_RE.match(str(h)):
            raise StoreError(f"hosts 꼴이 아니다: {h!r}")
    return cfg


def home_digest(files: list[dict]) -> str:
    """serving.home_digest 와 같다: 이름순 'sha256  크기  이름' 줄의 sha256."""
    rows = "".join(f"{f['sha256']}  {f['size']}  {f['name']}\n" for f in sorted(files, key=lambda f: f["name"]))
    return hashlib.sha256(rows.encode()).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_files_list(files) -> list[dict]:
    if not isinstance(files, list) or not files:
        raise StoreError("files 가 비었다")
    out, seen = [], set()
    for f in files:
        if not isinstance(f, dict) or not NAME_RE.match(str(f.get("name", ""))) or f["name"] in seen:
            raise StoreError(f"파일 이름 꼴·중복: {f!r}"[:200])
        if not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))):
            raise StoreError(f"sha256 꼴: {f['name']}")
        if not isinstance(f.get("size"), int) or isinstance(f["size"], bool) or f["size"] <= 0:
            raise StoreError(f"size 꼴: {f['name']}")
        seen.add(f["name"])
        out.append({"name": f["name"], "sha256": f["sha256"], "size": f["size"]})
    return out


# ---- HTTP(파드 쪽 창구) ----

def _conn(host: str, port: int, timeout: float) -> http.client.HTTPConnection:
    if not HOST_RE.match(host):
        raise StoreError(f"호스트 꼴: {host!r}")
    return http.client.HTTPConnection(host, port, timeout=timeout)


def get_json(host: str, port: int, path: str, timeout: float) -> tuple[int, object]:
    c = _conn(host, port, timeout)
    try:
        c.request("GET", path)
        r = c.getresponse()
        body = r.read(JSON_MAX + 1)
        if len(body) > JSON_MAX:
            raise StoreError(f"{host}{path}: 응답이 1 MB 를 넘는다")
        try:
            return r.status, json.loads(body or b"null")
        except ValueError:
            raise StoreError(f"{host}{path}: JSON 이 아니다") from None
    finally:
        c.close()


def put_file(host: str, port: int, name: str, path: Path, size: int, timeout: float) -> tuple[int, object]:
    c = _conn(host, port, timeout)
    try:
        c.putrequest("PUT", "/files/" + urllib.parse.quote(name))
        c.putheader("Content-Length", str(size))
        c.putheader("Content-Type", "application/octet-stream")
        c.endheaders()
        sent = 0
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(CHUNK), b""):
                c.send(b)
                sent += len(b)
        if sent != size:
            raise StoreError(f"{name}: 보낸 바이트 {sent} ≠ {size}")
        r = c.getresponse()
        body = r.read(JSON_MAX)
        try:
            return r.status, json.loads(body or b"null")
        except ValueError:
            return r.status, None
    finally:
        c.close()


# ---- push ----

def catalog(cfg: dict) -> Path:
    return Path(cfg["root"]) / "catalog"


def manifest_of(cfg: dict, alias: str) -> dict | None:
    if not ALIAS_RE.match(alias):
        return None
    p = catalog(cfg) / alias / "manifest.json"
    return json.loads(p.read_text()) if p.is_file() else None


def push_host(cfg: dict, host: str, port: int | None = None, state: dict | None = None) -> str:
    """한 호스트에 한 번. 돌려주는 글은 기록용 상태(같은 상태는 다시 찍지 않는다)."""
    port = port or cfg["port"]
    t = cfg["http_timeout_s"]
    try:
        code, need = get_json(host, port, "/need", t)
    except socket.gaierror:   # 테일넷에 그 이름이 없다(파드를 빌리지 않았거나 이미 내렸다)
        return f"{host}: 파드 없음(이름 못 풂)"
    except StoreError as e:
        return f"{host}: /need 오류({e})"
    except OSError as e:      # 이름은 풀렸다 = 파드(테일넷 기기)는 있다. 받는 창구(:{port})가 아직·이미 닫혀 있다
        return f"{host}: /need 없음(파드는 있음, {type(e).__name__})"
    if code != 200 or not isinstance(need, dict):
        return f"{host}: /need HTTP {code}"
    alias = str(need.get("alias", ""))
    if not ALIAS_RE.match(alias):
        return f"{host}: 별칭 꼴이 아니다"
    try:
        want = check_files_list(need.get("files"))
    except StoreError as e:
        return f"{host}: 목록 꼴 오류 {e}"
    man = manifest_of(cfg, alias)
    if man is None:
        return f"{host}: catalog 에 {alias} 가 없다"
    have = {(f["name"], f["sha256"], f["size"]) for f in man["files"]}
    if have != {(f["name"], f["sha256"], f["size"]) for f in want}:
        return f"{host}: {alias} 목록이 catalog manifest 와 다르다(보내지 않음)"
    todo = [f for f, raw in zip(want, need["files"]) if not raw.get("have")]
    if not todo:
        return f"{host}: {alias} 다 받음"
    for f in todo:
        src = catalog(cfg) / alias / f["name"]
        if not src.is_file() or src.stat().st_size != f["size"]:
            return f"{host}: catalog 파일이 manifest 와 다르다 {f['name']}"
        rate = cfg["min_rate_Bps"]
        t0 = time.monotonic()
        try:
            code, body = put_file(host, port, f["name"], src, f["size"], max(t, f["size"] / rate))
        except (OSError, StoreError) as e:
            return f"{host}: PUT {f['name']} 실패({type(e).__name__}: {str(e)[:120]})"
        dt = time.monotonic() - t0
        if code == 200:
            log(f"push {host} {alias} {f['name']} {f['size']:,}B {f['size'] / max(dt, 1e-6) / 1e6:.1f} MB/s")
        elif code == 409:
            continue
        else:
            return f"{host}: PUT {f['name']} HTTP {code} {json.dumps(body, ensure_ascii=False)[:160]}"
    return f"{host}: {alias} 보냄"


def push_once(cfg: dict, state: dict | None = None) -> list[str]:
    out = []
    state = {} if state is None else state
    for h in cfg["hosts"]:
        host, _, port = str(h).partition(":")
        msg = push_host(cfg, host, int(port) if port else None)
        if state.get(h) != msg:
            log(msg)
            state[h] = msg
        out.append(msg)
    return out


def daemon(cfg: dict) -> None:
    state: dict = {}
    log(f"modelpush daemon hosts={cfg['hosts']} every {cfg['interval_s']}s")
    while True:
        push_once(cfg, state)
        time.sleep(cfg["interval_s"])


# ---- pull ----

def check_export(idx) -> list[dict]:
    if not isinstance(idx, dict) or idx.get("version") != 1 or not isinstance(idx.get("sets"), list) or not idx["sets"]:
        raise StoreError("export.json 꼴이 아니다")
    sets = []
    for s in idx["sets"]:
        if not isinstance(s, dict) or not ALIAS_RE.match(str(s.get("alias", ""))):
            raise StoreError("세트 별칭 꼴")
        sets.append({"alias": s["alias"], "files": check_files_list(s.get("files"))})
    if len({s["alias"] for s in sets}) != len(sets):
        raise StoreError("세트 별칭 중복")
    return sets


def pull(cfg: dict, host: str, into: Path | None = None, send_done: bool = True, port: int | None = None) -> Path:
    port = port or cfg["port"]
    t = cfg["http_timeout_s"]
    code, idx = get_json(host, port, "/export", t)
    if code != 200:
        raise StoreError(f"{host}: /export HTTP {code}(아직 변환 중일 수 있다)")
    sets = check_export(idx)
    total = sum(f["size"] for s in sets for f in s["files"])
    if total > cfg["max_total_bytes"]:
        raise StoreError(f"합 {total:,}B 가 상한을 넘는다")
    inc = Path(cfg["root"]) / "incoming"
    dest = into or inc / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + host.split(".")[0])
    if dest.resolve().parent != inc.resolve():
        raise StoreError(f"받을 곳은 {inc} 바로 아래여야 한다")
    for s in sets:
        d = dest / s["alias"]
        d.mkdir(parents=True, exist_ok=True)
        for f in s["files"]:
            out = d / f["name"]
            if out.is_file() and out.stat().st_size == f["size"] and file_sha(out) == f["sha256"]:
                continue                                     # 이어 받기
            part = d / f".{f['name']}.part"
            c = _conn(host, port, max(t, 120))
            deadline = time.monotonic() + f["size"] / cfg["min_rate_Bps"] + 600
            try:
                c.request("GET", f"/export/{s['alias']}/{urllib.parse.quote(f['name'])}")
                r = c.getresponse()
                if r.status != 200:
                    raise StoreError(f"{f['name']}: HTTP {r.status}")
                if int(r.getheader("Content-Length", "-1")) != f["size"]:
                    raise StoreError(f"{f['name']}: Content-Length 가 목록 크기와 다르다")
                h, got = hashlib.sha256(), 0
                with open(part, "wb") as fh:
                    while got < f["size"]:
                        b = r.read(min(CHUNK, f["size"] - got))
                        if not b:
                            break
                        fh.write(b)
                        h.update(b)
                        got += len(b)
                        if time.monotonic() > deadline:
                            raise StoreError(f"{f['name']}: 시간 제한을 넘었다")
                if got != f["size"] or h.hexdigest() != f["sha256"]:
                    raise StoreError(f"{f['name']}: 크기·sha 가 목록과 다르다({got}/{f['size']})")
                os.replace(part, out)
                log(f"pull {host} {s['alias']} {f['name']} {got:,}B")
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            finally:
                c.close()
    (dest / "export.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1) + "\n")
    if send_done:
        c = _conn(host, port, t)
        try:
            c.request("POST", "/export/done", body=b"")
            st = c.getresponse().status
        finally:
            c.close()
        log(f"pull {host}: POST /export/done HTTP {st}")
    return dest


# ---- promote ----

def check_safetensors(p: Path) -> dict:
    size = p.stat().st_size
    with open(p, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise StoreError(f"{p.name}: safetensors 가 아니다")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 100 * 1024 * 1024 or 8 + n > size:
            raise StoreError(f"{p.name}: 머리 길이가 이상하다")
        try:
            h = json.loads(f.read(n))
        except ValueError:
            raise StoreError(f"{p.name}: 머리가 JSON 이 아니다") from None
    if not isinstance(h, dict):
        raise StoreError(f"{p.name}: 머리 꼴")
    meta = h.pop("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        raise StoreError(f"{p.name}: __metadata__ 는 글→글")
    pos = 0
    for k, v in sorted(h.items(), key=lambda kv: kv[1].get("data_offsets", [0])[0] if isinstance(kv[1], dict) else 0):
        if not isinstance(v, dict) or v.get("dtype") not in ST_DTYPES or not isinstance(v.get("shape"), list):
            raise StoreError(f"{p.name}: 텐서 {k} 꼴")
        b, e = v.get("data_offsets", [None, None])
        want = ST_DTYPES[v["dtype"]]
        for d in v["shape"]:
            if not isinstance(d, int) or d < 0:
                raise StoreError(f"{p.name}: 텐서 {k} 모양")
            want *= d
        if b != pos or e - b != want:
            raise StoreError(f"{p.name}: 텐서 {k} 오프셋·크기")
        pos = e
    if size != 8 + n + pos:
        raise StoreError(f"{p.name}: 파일 크기가 머리와 다르다")
    return h


def name_allowed(name: str) -> bool:
    low = name.lower()
    if low.endswith(DENY_SUFFIX):
        return False
    return name in ALLOWED_EXACT or low.endswith(ALLOWED_SUFFIX)


def check_report(rep: dict, th: dict) -> list[str]:
    """보고 숫자 문턱. pad_to 가 0(0 채우기를 하지 않은 대상)이면 0 채우기 문턱은 건너뛴다. pad_to 가 없는 옛 보고는 0 채운 것으로 본다."""
    bad = []
    if rep.get("pad_to") != 0:
        kp = rep.get("kl_pad") or {}
        if not (isinstance(kp.get("mean_kl"), (int, float)) and kp["mean_kl"] <= th["pad_mean_kl_max"]):
            bad.append(f"0 채우기 KL 평균 {kp.get('mean_kl')} > {th['pad_mean_kl_max']}")
        if not (isinstance(kp.get("top1_agree"), (int, float)) and kp["top1_agree"] >= th["pad_top1_min"]):
            bad.append(f"0 채우기 top-1 {kp.get('top1_agree')} < {th['pad_top1_min']}")
    kq = rep.get("kl_quant_bf16heads")
    if kq:
        if isinstance(kq.get("mean_kl"), (int, float)) and kq["mean_kl"] > th["quant_mean_kl_max"]:
            bad.append(f"양자화 KL 평균 {kq['mean_kl']:.4g} > {th['quant_mean_kl_max']}")
        if isinstance(kq.get("top1_agree"), (int, float)) and kq["top1_agree"] < th["quant_top1_min"]:
            bad.append(f"양자화 top-1 {kq['top1_agree']:.4g} < {th['quant_top1_min']}")
    if th["require_verify"] and not (rep.get("verify") or {}).get("ok"):
        bad.append("verify 가 통과하지 않았다")
    if th["require_quality"] and not (rep.get("quality") or {}).get("ok"):
        bad.append("quality_battery 가 통과하지 않았다")
    return bad


def promote(cfg: dict, src: Path, alias: str) -> dict:
    if not ALIAS_RE.match(alias):
        raise StoreError(f"별칭 꼴: {alias!r}")
    inc = (Path(cfg["root"]) / "incoming").resolve()
    src = src.resolve()
    if src.parent != inc:
        raise StoreError(f"incoming 바로 아래 폴더만 받는다: {src}")
    idx = json.loads((src / "export.json").read_text())
    sets = check_export(idx)
    st = next((s for s in sets if s["alias"] == alias), None)
    if st is None:
        raise StoreError(f"{alias} 가 export.json 에 없다(있는 것: {', '.join(s['alias'] for s in sets)})")
    d = src / alias
    on_disk = sorted(p.name for p in d.iterdir() if not p.name.startswith("."))
    listed = sorted(f["name"] for f in st["files"])
    if on_disk != listed:
        raise StoreError(f"폴더와 목록이 다르다(남는 것 {sorted(set(on_disk) - set(listed))}, 빠진 것 {sorted(set(listed) - set(on_disk))})")
    bad_names = [n for n in listed if not name_allowed(n)]
    if bad_names:
        raise StoreError(f"허용하지 않는 파일 꼴: {bad_names}")
    headers = {}
    for f in st["files"]:
        p = d / f["name"]
        if p.stat().st_size != f["size"] or file_sha(p) != f["sha256"]:
            raise StoreError(f"{f['name']}: 크기·sha 가 목록과 다르다")
        if f["name"].endswith(".safetensors"):
            headers[f["name"]] = check_safetensors(p)
        elif f["name"].endswith(".json"):
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                raise StoreError(f"{f['name']}: JSON 이 아니다") from None
    for need in ("config.json", "model.safetensors.index.json"):
        if need not in listed:
            raise StoreError(f"{need} 가 없다")
    wm = json.loads((d / "model.safetensors.index.json").read_text()).get("weight_map") or {}
    shards = {n for n in listed if n.endswith(".safetensors")}
    if set(wm.values()) != shards:
        raise StoreError(f"색인의 샤드와 파일이 다르다(색인에만 {sorted(set(wm.values()) - shards)}, 파일만 {sorted(shards - set(wm.values()))})")
    for k, shard in wm.items():
        if k not in headers[shard]:
            raise StoreError(f"색인의 텐서 {k} 가 {shard} 에 없다")
    raw = next(s for s in idx["sets"] if s.get("alias") == alias)
    rep = raw.get("report") if isinstance(raw.get("report"), dict) else (idx.get("report") or {})   # 세트마다 보고(대상 여럿), 옛 꼴은 맨 위
    bad = check_report(rep, cfg["thresholds"])
    if bad:
        raise StoreError("보고 숫자 문턱: " + "; ".join(bad))
    cat = catalog(cfg)
    final, tmp = cat / alias, cat / f".{alias}.tmp"
    if final.exists():
        raise StoreError(f"catalog 에 {alias} 가 이미 있다(바꾸려면 새 별칭으로)")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in st["files"]:
        shutil.copyfile(d / f["name"], tmp / f["name"])
        if file_sha(tmp / f["name"]) != f["sha256"]:
            raise StoreError(f"{f['name']}: 복사 뒤 sha 가 다르다")
    man = {"alias": alias, "digest": home_digest(st["files"]), "files": st["files"], "from": str(src),
           "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "report": {k: rep.get(k) for k in ("src", "calibration", "pad_to", "kl_pad", "kl_quant_bf16heads", "verify", "spec_decode")},
           "quality": (rep.get("quality") or {}).get("result")}
    (tmp / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, final)
    return man


def toml_snippet(man: dict) -> str:
    rows = ",\n".join(f'  {{ name = "{f["name"]}", sha256 = "{f["sha256"]}", size = {f["size"]} }}' for f in man["files"])
    return (f'# promote {man["promoted_at"]} digest {man["digest"]} — deploy/serving/models.toml 에 붙인다(나머지 칸은 모델 A 를 본뜬다)\n'
            f'[models.{man["alias"]}]\nsource = "home"\ndir = "{man["alias"]}"\nfiles = [\n{rows},\n]\n')


def list_catalog(cfg: dict) -> list[dict]:
    out = []
    for p in sorted(catalog(cfg).glob("*/manifest.json")):
        m = json.loads(p.read_text())
        out.append({"alias": m["alias"], "files": len(m["files"]), "bytes": sum(f["size"] for f in m["files"]),
                    "digest": m["digest"], "promoted_at": m.get("promoted_at")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daemon")
    sub.add_parser("push-once")
    pl = sub.add_parser("pull")
    pl.add_argument("host")
    pl.add_argument("--into")
    pl.add_argument("--no-done", action="store_true")
    pr = sub.add_parser("promote")
    pr.add_argument("incoming")
    pr.add_argument("alias")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config)
        if a.cmd == "daemon":
            daemon(cfg)
        elif a.cmd == "push-once":
            for m in push_once(cfg):
                print(m)
        elif a.cmd == "pull":
            host, _, port = a.host.partition(":")
            print(pull(cfg, host, Path(a.into) if a.into else None, not a.no_done, int(port) if port else None))
        elif a.cmd == "promote":
            man = promote(cfg, Path(a.incoming), a.alias)
            print(json.dumps({"alias": man["alias"], "digest": man["digest"], "files": len(man["files"])}, ensure_ascii=False))
            print(toml_snippet(man))
        else:
            for m in list_catalog(cfg):
                print(json.dumps(m, ensure_ascii=False))
    except (StoreError, OSError, ValueError) as e:
        print(f"modelpush: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
