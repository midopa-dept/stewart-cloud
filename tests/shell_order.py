"""셸 스크립트 정의 차례 검사(v13): 함수·대문자 전역 변수가, 그것을 (부른 함수를 거쳐서라도) 쓰는 맨 위 줄보다 앞에서
정의되는지. trap 문자열도 부르는 자리로 본다. heredoc 본문·주석은 뺀다. 그 자리에서 실제로 불리는지는 보지 않으므로 더 엄하다."""
import re, sys
from pathlib import Path

def analyze(text: str):
    lines = text.splitlines()
    defs, bodies, top = {}, {}, []          # 이름 → 줄, 이름 → 본문 줄들, 맨 위 (줄, 글)
    i, heredoc, cur = 0, None, None
    for n, ln in enumerate(lines, 1):
        if heredoc:
            if ln.strip() == heredoc:
                heredoc = None
            continue
        code = ln.split(" #", 1)[0] if not ln.lstrip().startswith("#") else ""
        m = re.search(r"<<-?'?(\w+)'?", code)
        if m and "<<<" not in code:
            heredoc = m.group(1)
        if cur is None:
            d = re.match(r"^(\w+)\(\) \{(.*)$", code)
            if d:
                name, rest = d.group(1), d.group(2)
                defs.setdefault(name, n)
                if rest.rstrip().endswith("}"):
                    bodies[name] = [rest]
                else:
                    cur = name
                    bodies[name] = [rest]
                continue
            top.append((n, code))
        else:
            if ln.startswith("}"):
                cur = None
            else:
                bodies[cur].append(code)
    return defs, bodies, top

def calls(code: str, names):
    code = re.sub(r"\$\w+|\$\{[^}]*\}", " ", code)
    out = set()
    for w in re.findall(r"(?<![\w./$-])([a-z_][a-z0-9_]*)(?![\w.-])", code):
        if w in names:
            out.add(w)
    return out

def problems(text: str):
    defs, bodies, top = analyze(text)
    names = set(defs)
    graph = {f: set().union(*(calls(c, names) for c in b)) - {f} if b else set() for f, b in bodies.items()}
    bad = []
    for n, code in top:
        for f in calls(code, names):
            seen, stack = set(), [f]
            while stack:
                g = stack.pop()
                if g in seen:
                    continue
                seen.add(g)
                if defs[g] > n:
                    bad.append(f"{n}줄: {f} → {g}(정의 {defs[g]}줄)")
                stack += graph.get(g, ())
    return bad

def var_problems(text: str):
    """함수가 쓰는 대문자 전역 변수가, 그 함수를 맨 위에서 처음 부르는 줄보다 앞에서 정해지는지(${X:-…} 꼴은 기본값이 있어 뺀다)."""
    defs, bodies, top = analyze(text)
    names = set(defs)
    graph = {f: set().union(*(calls(c, names) for c in b)) - {f} if b else set() for f, b in bodies.items()}
    assigned = {}
    for n, code in top:
        for v in re.findall(r"(?:^|[\s;(])(?:export\s+)?([A-Z][A-Z0-9_]*)=", code):
            assigned.setdefault(v, n)
        for v in re.findall(r"\bfor ([A-Z][A-Z0-9_]*) in", code):
            assigned.setdefault(v, n)
        m = re.match(r"\s*while read -r (.*?);", code)
        if m:
            for v in m.group(1).split():
                assigned.setdefault(v, n)
    env_ok = {"CONTAINER_ID", "CONTAINER_API_KEY", "TS_AUTHKEY", "MAX_HOURS", "EXPORT_WAIT_MIN", "LINENO", "HOME", "PATH"}
    def used(f):
        out, mine = set(), set()
        for c in bodies.get(f, []):
            out |= set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)(?![A-Z0-9_]*:-)", c))
            mine |= set(re.findall(r"(?:^|[\s;(])([A-Z][A-Z0-9_]*)=", c))
        out -= mine
        loc = set()
        for c in bodies.get(f, []):
            m = re.search(r"\blocal\s+(.*)", c)
            if m:
                loc |= {x.split("=")[0] for x in m.group(1).split()}
        return out - loc
    bad = []
    for n, code in top:
        for f in calls(code, names):
            seen, stack = set(), [f]
            while stack:
                g = stack.pop()
                if g in seen:
                    continue
                seen.add(g)
                for v in used(g):
                    if v in env_ok:
                        continue
                    if v not in assigned or assigned[v] > n:
                        bad.append(f"{n}줄: {f} → {g} 가 쓰는 ${v}(정함 {assigned.get(v, '없음')}줄)")
                stack += graph.get(g, ())
    return sorted(set(bad))

