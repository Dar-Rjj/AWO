"""Length-prefixed JSON protocol shared by the remote sandbox client and service."""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Mapping
from typing import Any

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 2_097_152
_HEADER = struct.Struct("!I")


class ProtocolError(ValueError):
    """Raised when a sandbox protocol frame is invalid."""


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ProtocolError("sandbox connection closed before the frame completed")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_message(
    connection: socket.socket,
    *,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> dict[str, Any]:
    """Receive and validate one length-prefixed JSON object."""

    header = _recv_exact(connection, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    if size <= 0 or size > max_frame_bytes:
        raise ProtocolError(f"invalid sandbox frame size: {size}")
    try:
        value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("sandbox frame is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError("sandbox frame must contain a JSON object")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported sandbox protocol version")
    return value


def send_message(
    connection: socket.socket,
    value: Mapping[str, Any],
    *,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> None:
    """Encode and send one length-prefixed JSON object."""

    message = {"protocol_version": PROTOCOL_VERSION, **dict(value)}
    payload = json.dumps(
        message,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if not payload or len(payload) > max_frame_bytes:
        raise ProtocolError(f"sandbox frame exceeds {max_frame_bytes} bytes")
    connection.sendall(_HEADER.pack(len(payload)) + payload)
