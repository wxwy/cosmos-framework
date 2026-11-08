"""Bounded local Unix IPC for the standalone shared VAE Worker.

No launcher, model loading, pickle, or Action Path integration lives here. The
owner control capability is distinct from per-session submission capabilities.
"""

from __future__ import annotations

import os
import secrets
import socket
import stat
import struct
import threading
import time
from pathlib import Path

from cosmos_framework.inference.robocasa_vae_protocol import (
    MAX_FRAME_WIRE_BYTES,
    Frame,
    canonical_json,
    strict_json,
)

_CONTROL_BYTES = 65536


class VAEIPCError(RuntimeError):
    def __init__(self, code, remote_type):
        self.code, self.remote_type = code, remote_type
        super().__init__("VAE IPC request rejected: " + code)


def _json(value):
    return canonical_json(value)


def _strict_json(raw):
    return strict_json(raw, max_bytes=MAX_FRAME_WIRE_BYTES)


def _ns(value):
    if type(value) is not int or not 0 < value < 10**20:
        raise ValueError("positive absolute monotonic nanoseconds required")
    return value


def _wire_ns(value):
    if type(value) is not str or not value or len(value) > 20 or any(c not in "0123456789" for c in value):
        raise ValueError("decimal nanosecond string required")
    return _ns(int(value))


def _remaining(deadline_ns):
    remaining = (_ns(deadline_ns) - time.monotonic_ns()) / 1e9
    if remaining <= 0:
        raise TimeoutError("original IPC deadline exhausted")
    return remaining


def _drain_deadline(deadline_ns):
    # A negative drain receipt also needs bounded time to reach its caller.
    remaining = max(0, deadline_ns - time.monotonic_ns())
    return deadline_ns - min(10_000_000, remaining // 4)


def _exact(connection, length, deadline_ns):
    result = bytearray()
    while len(result) < length:
        connection.settimeout(_remaining(deadline_ns))
        chunk = connection.recv(length - len(result))
        if not chunk:
            raise ConnectionError("incomplete IPC frame")
        result.extend(chunk)
    return bytes(result)


def _receive(connection, deadline_ns, limit):
    length = struct.unpack("!I", _exact(connection, 4, deadline_ns))[0]
    if not 0 < length <= limit:
        raise ValueError("IPC frame exceeds bounded size")
    return _strict_json(_exact(connection, length, deadline_ns))


def _send(connection, raw, deadline_ns, limit):
    if not 0 < len(raw) <= limit:
        raise ValueError("IPC frame exceeds bounded size")
    connection.settimeout(_remaining(deadline_ns))
    connection.sendall(struct.pack("!I", len(raw)) + raw)


def _control_path(path):
    return Path(str(path) + ".ctl")


def _socket_path(path):
    path = Path(path)
    if not path.is_absolute() or len(os.fsencode(_control_path(path))) >= 108:
        raise ValueError("bounded absolute Unix socket path required")
    return path


class UnixVAEServer:
    """Independent bounded data/control lanes, one request per connection."""

    def __init__(
        self,
        worker,
        socket_path,
        control_capability,
        *,
        max_data_connections=4,
        max_control_connections=2,
        connection_timeout_s=1.0,
        max_frame_bytes=MAX_FRAME_WIRE_BYTES,
    ):
        if type(control_capability) is not bytes or not 32 <= len(control_capability) <= 256:
            raise ValueError("independent local control capability of at least 32 bytes required")
        if any(type(n) is not int or not 1 <= n <= 32 for n in (max_data_connections, max_control_connections)):
            raise ValueError("bounded IPC connection counts required")
        if type(connection_timeout_s) not in (int, float) or not 0 < connection_timeout_s <= 30:
            raise ValueError("bounded IPC connection deadline required")
        if type(max_frame_bytes) is not int or not 1024 <= max_frame_bytes <= MAX_FRAME_WIRE_BYTES:
            raise ValueError("bounded IPC frame size required")
        self.worker, self.path = worker, _socket_path(socket_path)
        self._control_capability = control_capability.hex()
        self._connection_ns = int(connection_timeout_s * 1e9)
        self._max_frame_bytes = max_frame_bytes
        self._limits = {"data": max_data_connections, "control": max_control_connections}
        self._slots = {name: threading.BoundedSemaphore(n) for name, n in self._limits.items()}
        self._changed = threading.Lock()
        self._stop = threading.Event()
        self._listeners, self._acceptors, self._handlers, self._connections, self._inodes = {}, [], {}, set(), {}
        self._started, self._parent_created, self._parent_inode = False, False, None
        self._first_error = None

    def _prepare_parent(self):
        parent = self.path.parent
        for ancestor in (parent, *parent.parents):
            try:
                mode = ancestor.lstat().st_mode
            except FileNotFoundError:
                if ancestor != parent:
                    raise ValueError("socket parent ancestors must exist") from None
                continue
            if not stat.S_ISDIR(mode):
                raise ValueError("socket ancestors must be real directories")
        try:
            parent.mkdir(mode=0o700)
            self._parent_created = True
        except FileExistsError:
            pass
        record = parent.lstat()
        if not stat.S_ISDIR(record.st_mode) or stat.S_IMODE(record.st_mode) != 0o700 or record.st_uid != os.geteuid():
            raise PermissionError("socket parent must be owned mode 0700 directory")
        self._parent_inode = record.st_dev, record.st_ino

    def start(self):
        if self._started or self._stop.is_set():
            raise RuntimeError("IPC server cannot restart")
        self._prepare_parent()
        try:
            for lane, path in (("data", self.path), ("control", _control_path(self.path))):
                if path.exists() or path.is_symlink():
                    raise FileExistsError("socket path already exists")
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._listeners[lane] = listener
                listener.bind(str(path))
                record = path.lstat()
                self._inodes[path] = record.st_dev, record.st_ino
                os.chmod(path, 0o600, follow_symlinks=False)
                listener.listen(self._limits[lane])
                listener.settimeout(0.1)
            self._started = True
            for lane, listener in self._listeners.items():
                thread = threading.Thread(
                    target=self._accept, args=(lane, listener), name="vae-ipc-" + lane, daemon=True
                )
                thread.start()
                self._acceptors.append(thread)
            return self
        except BaseException:
            self._stop_listeners()
            self._unlink_owned()
            raise

    def _accept(self, lane, listener):
        while not self._stop.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            deadline_ns = time.monotonic_ns() + self._connection_ns
            self._reap_handlers()
            if not self._slots[lane].acquire(blocking=False):
                connection.close()
                continue
            with self._changed:
                if self._stop.is_set():
                    connection.close()
                    self._slots[lane].release()
                    return
                thread = threading.Thread(
                    target=self._handle,
                    args=(lane, connection, deadline_ns),
                    name="vae-ipc-request-" + lane,
                    daemon=True,
                )
                self._connections.add(connection)
                self._handlers[thread] = lane
                try:
                    thread.start()
                except Exception as error:
                    del self._handlers[thread]
                    self._connections.discard(connection)
                    connection.close()
                    self._slots[lane].release()
                    if self._first_error is None:
                        self._first_error = error
                    self._stop_listeners()
                    return

    def _reap_handlers(self):
        # A different thread observes termination before returning its slot.
        # Self-removal or release in the handler's finally can falsely certify
        # quiescence while its remaining socket cleanup is still executing.
        with self._changed:
            for thread, lane in tuple(self._handlers.items()):
                if not thread.is_alive():
                    del self._handlers[thread]
                    self._slots[lane].release()

    def _dispatch(self, lane, request, accepted_deadline):
        op = request.get("op")
        if type(op) is not str:
            raise ValueError("IPC operation name required")
        common = {"op", "deadline_ns"}
        owner_fields = {
            "owner_register": {"lease", "session_capability", "episode_deadline_ns"},
            "owner_issue": {"session_id", "chunk_id", "request_key", "source_start", "count"},
            "owner_complete": {"session_id", "chunk_id", "completed_through"},
            "owner_boundary_skip": {
                "lease",
                "chunk_id",
                "request_key",
                "source_step",
                "source_rgb_sha256",
                "raw15_sha256",
                "policy_boundary_sha256",
                "post_step_receipt",
            },
            "shutdown": set(),
        }
        normal_fields = {
            "submit": {"frame", "capability"},
            "poll": {"session_id", "job_key", "capability"},
            "ack": {"session_id", "job_key", "capability"},
            "end": {"session_id", "capability"},
            "info": set(),
        }
        if op in owner_fields:
            if lane != "control" or set(request) != common | {"control_capability"} | owner_fields[op]:
                raise PermissionError("owner operation requires control lane")
            cap = request["control_capability"]
            if type(cap) is not str or not secrets.compare_digest(cap, self._control_capability):
                raise PermissionError("owner control capability mismatch")
        elif op in normal_fields:
            if set(request) != common | normal_fields[op] or (lane == "data") != (op == "submit"):
                raise ValueError("exact IPC fields and operation lane required")
        else:
            raise ValueError("unknown IPC operation")
        deadline = min(accepted_deadline, _wire_ns(request["deadline_ns"]))
        _remaining(deadline)
        if op == "submit":
            frame = Frame.from_wire(request["frame"])
            return {"job_key": self.worker.submit(frame, request["capability"])}, False
        if op == "poll":
            return self.worker.poll(request["session_id"], request["job_key"], request["capability"]), False
        if op == "ack":
            return self.worker.ack(request["session_id"], request["job_key"], request["capability"]), False
        if op == "end":
            return self.worker.end(request["session_id"], request["capability"], _drain_deadline(deadline)), False
        if op == "info":
            return self.worker.info(), False
        if op == "owner_register":
            return self.worker.owner_register(
                request["lease"], request["session_capability"], _wire_ns(request["episode_deadline_ns"])
            ), False
        if op == "owner_issue":
            return self.worker.owner_issue(
                request["session_id"],
                request["chunk_id"],
                request["request_key"],
                request["source_start"],
                request["count"],
            ), False
        if op == "owner_boundary_skip":
            return self.worker.owner_boundary_skip(
                request["lease"],
                request["chunk_id"],
                request["request_key"],
                request["source_step"],
                request["source_rgb_sha256"],
                request["raw15_sha256"],
                request["policy_boundary_sha256"],
                request["post_step_receipt"],
            ), False
        if op == "owner_complete":
            return self.worker.owner_complete(
                request["session_id"], request["chunk_id"], request["completed_through"]
            ), False
        result = self.worker.close(_drain_deadline(deadline))
        return {**result, "ipc_quiescent": False}, result.get("closed") is True and result.get("quiescent") is True

    def _handle(self, lane, connection, deadline_ns):
        shutdown = False
        limit = self._max_frame_bytes if lane == "data" else _CONTROL_BYTES
        try:
            try:
                request = _receive(connection, deadline_ns, limit)
                if "deadline_ns" in request:
                    deadline_ns = min(deadline_ns, _wire_ns(request["deadline_ns"]))
                if request.get("op") == "submit":
                    frame = request.get("frame")
                    if type(frame) is not dict or type(frame.get("header")) is not dict:
                        raise ValueError("exact Frame object required")
                    header = frame["header"]
                    for field in ("episode_deadline_ns", "job_deadline_ns", "replan_deadline_ns"):
                        if header.get(field) is not None:
                            deadline_ns = min(deadline_ns, _wire_ns(header[field]))
                result, shutdown = self._dispatch(lane, request, deadline_ns)
                response = {"ok": True, "result": result}
            except Exception as error:
                code = (
                    "FORBIDDEN"
                    if isinstance(error, PermissionError)
                    else "TIMEOUT"
                    if isinstance(error, TimeoutError)
                    else "INVALID"
                    if isinstance(error, ValueError)
                    else "UNAVAILABLE"
                )
                response = {"ok": False, "error": {"code": code, "type": type(error).__name__}}
            _send(connection, _json(response), deadline_ns, self._max_frame_bytes)
        except (OSError, ValueError, TimeoutError):
            pass
        finally:
            connection.close()
            with self._changed:
                self._connections.discard(connection)
            if shutdown:
                self._stop_listeners()
                self._unlink_owned()

    def _stop_listeners(self):
        self._stop.set()
        for listener in self._listeners.values():
            listener.close()

    def _unlink_owned(self):
        for path, identity in self._inodes.items():
            try:
                current = path.lstat()
                if stat.S_ISSOCK(current.st_mode) and (current.st_dev, current.st_ino) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        if self._parent_created:
            try:
                current = self.path.parent.lstat()
                if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == self._parent_inode:
                    self.path.parent.rmdir()
            except OSError:
                pass

    def wait_closed(self, timeout=None):
        return self._stop.wait(timeout)

    def close(self, deadline_ns):
        _ns(deadline_ns)
        self._stop_listeners()
        with self._changed:
            connections = tuple(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            result = self.worker.close(deadline_ns)
        finally:
            threads = tuple(self._acceptors)
            with self._changed:
                threads += tuple(self._handlers)
            for thread in threads:
                if thread is not threading.current_thread():
                    thread.join(max(0, (deadline_ns - time.monotonic_ns()) / 1e9))
            self._unlink_owned()
        ipc_quiescent = not any(thread.is_alive() for thread in threads if thread is not threading.current_thread())
        self._reap_handlers()
        if self._first_error is not None:
            raise self._first_error
        return {
            **result,
            "ipc_quiescent": ipc_quiescent,
            "acknowledged": result.get("closed") is True and ipc_quiescent,
            "quiescent": result.get("quiescent") is True and ipc_quiescent,
        }


class _Client:
    def __init__(self, socket_path, *, retry_interval_s=0.01):
        self.path = _socket_path(socket_path)
        if type(retry_interval_s) not in (int, float) or not 0 <= retry_interval_s <= 1:
            raise ValueError("bounded IPC retry interval required")
        self.retry_interval_s = retry_interval_s

    def _call(self, op, fields, deadline_ns, *, data=False):
        _ns(deadline_ns)
        raw = _json({"op": op, "deadline_ns": str(deadline_ns), **fields})
        limit = MAX_FRAME_WIRE_BYTES if data else _CONTROL_BYTES
        if len(raw) > limit:
            raise ValueError("IPC frame exceeds bounded size")
        path, first_error = self.path if data else _control_path(self.path), None
        while True:
            try:
                remaining = _remaining(deadline_ns)
            except TimeoutError as error:
                if first_error is not None:
                    raise error from first_error
                raise
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(remaining)
                    connection.connect(str(path))
                    _send(connection, raw, deadline_ns, limit)
                    reply = _receive(connection, deadline_ns, MAX_FRAME_WIRE_BYTES)
                _remaining(deadline_ns)
                if type(reply.get("ok")) is not bool:
                    raise ValueError("exact IPC response required")
                if reply["ok"]:
                    if set(reply) != {"ok", "result"}:
                        raise ValueError("exact IPC response required")
                    return reply["result"]
                if (
                    set(reply) != {"ok", "error"}
                    or type(reply["error"]) is not dict
                    or set(reply["error"]) != {"code", "type"}
                ):
                    raise ValueError("exact IPC error response required")
                error = VAEIPCError(reply["error"]["code"], reply["error"]["type"])
                if first_error is not None:
                    raise error from first_error
                raise error
            except (OSError, ConnectionError) as error:
                if first_error is None:
                    first_error = error
                time.sleep(min(self.retry_interval_s, max(0, (deadline_ns - time.monotonic_ns()) / 1e9)))


class LocalVAEClient(_Client):
    """Session-only client: no registration or completed-frontier authority."""

    def __init__(self, socket_path, session_id, capability, **kwargs):
        super().__init__(socket_path, **kwargs)
        self.session_id, self._capability = session_id, capability

    def submit(self, frame, deadline_ns):
        if not isinstance(frame, Frame):
            raise ValueError("immutable validated Frame required")
        header = frame.header
        bounds = [deadline_ns, _wire_ns(header["episode_deadline_ns"]), _wire_ns(header["job_deadline_ns"])]
        if header["replan_deadline_ns"] is not None:
            bounds.append(_wire_ns(header["replan_deadline_ns"]))
        if header["lease"]["session_id"] != self.session_id:
            raise ValueError("foreign Frame Session")
        return self._call("submit", {"frame": frame.to_wire(), "capability": self._capability}, min(bounds), data=True)[
            "job_key"
        ]

    def poll(self, job_key, deadline_ns):
        return self._call(
            "poll", {"session_id": self.session_id, "job_key": job_key, "capability": self._capability}, deadline_ns
        )

    def ack(self, job_key, deadline_ns):
        return self._call(
            "ack", {"session_id": self.session_id, "job_key": job_key, "capability": self._capability}, deadline_ns
        )

    def end(self, deadline_ns):
        return self._call("end", {"session_id": self.session_id, "capability": self._capability}, deadline_ns)

    def info(self, deadline_ns):
        return self._call("info", {}, deadline_ns)


class OwnerVAEClient(_Client):
    def __init__(self, socket_path, control_capability, **kwargs):
        super().__init__(socket_path, **kwargs)
        if type(control_capability) is not bytes or not 32 <= len(control_capability) <= 256:
            raise ValueError("independent local control capability required")
        self._control_capability = control_capability.hex()

    def _owner(self, op, fields, deadline_ns):
        return self._call(op, {"control_capability": self._control_capability, **fields}, deadline_ns)

    def register(self, lease, session_capability, episode_deadline_ns, deadline_ns):
        return self._owner(
            "owner_register",
            {
                "lease": lease,
                "session_capability": session_capability,
                "episode_deadline_ns": str(_ns(episode_deadline_ns)),
            },
            deadline_ns,
        )

    def issue(self, session_id, chunk_id, request_key, source_start, count, deadline_ns):
        return self._owner(
            "owner_issue",
            dict(
                session_id=session_id,
                chunk_id=chunk_id,
                request_key=request_key,
                source_start=source_start,
                count=count,
            ),
            deadline_ns,
        )

    def complete(self, session_id, chunk_id, completed_through, deadline_ns):
        return self._owner(
            "owner_complete",
            dict(session_id=session_id, chunk_id=chunk_id, completed_through=completed_through),
            deadline_ns,
        )

    def boundary_skip(
        self,
        lease,
        chunk_id,
        request_key,
        source_step,
        source_rgb_sha256,
        raw15_sha256,
        policy_boundary_sha256,
        post_step_receipt,
        deadline_ns,
    ):
        """Owner-only sparse advance; never returns an encoded Worker Feature."""
        return self._owner(
            "owner_boundary_skip",
            dict(
                lease=lease,
                chunk_id=chunk_id,
                request_key=request_key,
                source_step=source_step,
                source_rgb_sha256=source_rgb_sha256,
                raw15_sha256=raw15_sha256,
                policy_boundary_sha256=policy_boundary_sha256,
                post_step_receipt=post_step_receipt,
            ),
            deadline_ns,
        )

    def shutdown(self, deadline_ns):
        return self._owner("shutdown", {}, deadline_ns)
