"""Length-prefixed JSON RPC over local Unix sockets."""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path
from typing import Any, Protocol

MAX_RPC_BYTES = 2 * 1024 * 1024


class RpcStream(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def write(self, data: bytes) -> int | None: ...

    def flush(self) -> None: ...


def read_message(stream: RpcStream) -> dict[str, Any]:
    header = stream.read(4)
    if len(header) != 4:
        raise EOFError("research RPC header is incomplete")
    size = struct.unpack("!I", header)[0]
    if size <= 0 or size > MAX_RPC_BYTES:
        raise ValueError("research RPC message size is invalid")
    raw = stream.read(size)
    if len(raw) != size:
        raise EOFError("research RPC message is incomplete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("research RPC message must be an object")
    return value


def write_message(stream: RpcStream, value: dict[str, Any]) -> None:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_RPC_BYTES:
        raise ValueError("research RPC response is too large")
    stream.write(struct.pack("!I", len(raw)))
    stream.write(raw)
    stream.flush()


def call(
    socket_path: str | Path,
    value: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    af_unix = getattr(socket, "AF_UNIX", None)
    if af_unix is None:
        raise RuntimeError("research RPC requires AF_UNIX")
    with socket.socket(af_unix, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(socket_path))
        with connection.makefile("rwb", buffering=0) as stream:
            write_message(stream, value)
            return read_message(stream)
