"""safetensors 를 표준 라이브러리만으로 읽고 쓴다. torch 없이 텐서 바이트를 그대로 옮기거나 0 을 채울 때 쓴다.

형식: 앞 8바이트 = 머리 길이 N(little-endian u64), 다음 N바이트 = JSON 머리
{이름: {"dtype", "shape", "data_offsets": [시작, 끝]}, "__metadata__": {...}}, 그 뒤가 자료(오프셋은 자료 시작 기준).
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
              "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "U32": 4, "U64": 8}
CHUNK = 1 << 24


def read_header(path: Path) -> tuple[dict, int]:
    """(머리 dict, 자료 시작 바이트)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def tensors(h: dict) -> list[tuple[str, dict]]:
    """머리의 텐서를 자료 순서(오프셋)대로."""
    return sorted(((k, v) for k, v in h.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0])


def nbytes(meta: dict) -> int:
    n = DTYPE_SIZE[meta["dtype"]]
    for d in meta["shape"]:
        n *= d
    return n


def check_intact(path: Path) -> None:
    """머리가 말하는 바이트가 다 있는지, 오프셋이 겹치거나 비지 않는지."""
    h, start = read_header(path)
    pos = 0
    for k, v in tensors(h):
        b, e = v["data_offsets"]
        if b != pos or e - b != nbytes(v):
            raise ValueError(f"{path.name}: {k} 오프셋·크기가 맞지 않다")
        pos = e
    if os.path.getsize(path) != start + pos:
        raise ValueError(f"{path.name}: 파일 크기가 머리와 다르다")


# 텐서 하나를 쓰는 함수: (원본 파일, 원본 자료 시작, 원본 meta, 출력 파일) → None. 출력 바이트 수는 새 meta 와 같아야 한다.
Writer = Callable[[BinaryIO, int, dict, BinaryIO], None]


def copy_bytes(src: BinaryIO, data_start: int, meta: dict, out: BinaryIO) -> None:
    b, e = meta["data_offsets"]
    src.seek(data_start + b)
    left = e - b
    while left:
        chunk = src.read(min(CHUNK, left))
        if not chunk:
            raise ValueError("원본이 짧다")
        out.write(chunk)
        left -= len(chunk)


def write_file(out_path: Path, plan: list[tuple[str, dict, Callable[[BinaryIO], None]]], metadata: dict | None) -> None:
    """plan: [(이름, 새 meta(dtype·shape), 바이트를 쓰는 함수)]. 머리는 8바이트 정렬로 공백을 채운다. 다 쓰고 나서 이름을 바꾼다."""
    header: dict = {}
    pos = 0
    for name, meta, _ in plan:
        n = nbytes(meta)
        header[name] = {"dtype": meta["dtype"], "shape": list(meta["shape"]), "data_offsets": [pos, pos + n]}
        pos += n
    if metadata is not None:
        header["__metadata__"] = metadata
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - (8 + len(hb)) % 8) % 8)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for name, meta, fn in plan:
            before = out.tell()
            fn(out)
            if out.tell() - before != nbytes(meta):
                raise ValueError(f"{name}: 쓴 바이트 수가 모양과 다르다")
    os.replace(tmp, out_path)


def iter_index(model_dir: Path) -> Iterator[tuple[str, str]]:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    yield from idx["weight_map"].items()
