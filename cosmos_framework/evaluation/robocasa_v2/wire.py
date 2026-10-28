"""Small bounded inherited-socket protocol, never TCP or pickle."""

import json
import struct

LIMIT = 1024 * 1024


def send(sock, value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > LIMIT:
        raise ValueError("oversized actor message")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


def receive(sock):
    def exact(size):
        chunks = []
        while size:
            chunk = sock.recv(size)
            if not chunk:
                raise EOFError("actor control socket closed")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    length = struct.unpack("!I", exact(4))[0]
    if length > LIMIT:
        raise ValueError("oversized actor frame")
    value = json.loads(exact(length))
    if not isinstance(value, dict):
        raise ValueError("actor frame must be object")
    return value
