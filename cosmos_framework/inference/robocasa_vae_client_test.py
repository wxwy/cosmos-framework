"""Actual local sockets/subprocesses; fake encoder bytes, no model or GPU."""

from __future__ import annotations

import json
import os
import secrets
import select
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from cosmos_framework.inference.robocasa_vae_client import (
    LocalVAEClient,
    OwnerVAEClient,
    UnixVAEServer,
    VAEIPCError,
    _json,
    _receive,
    _send,
)
from cosmos_framework.inference.robocasa_vae_protocol import (
    MAX_FRAME_WIRE_BYTES,
    RGB_BYTES,
    Frame,
    ProtocolError,
    verify_feature,
)


def _deadline(seconds=5):
    return time.monotonic_ns() + int(seconds * 1e9)


def _lease(session="session"):
    return dict(
        server_epoch="policy-epoch",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_id="slot",
        generation=1,
        task="CloseFridge",
        episode_index=0,
        attempt_id="attempt",
        session_id=session,
        episode_id="episode-" + session,
    )


def _identity():
    return dict(
        vae_weights_sha256="b" * 64,
        vae_config_sha256="c" * 64,
        preprocess_profile_sha256="d" * 64,
        visual96_contract_sha256="e" * 64,
        encoder_backend="eager",
        encoder_dtype="bfloat16",
    )


def _frame(episode_deadline, *, step=0, value=1, session="session"):
    return Frame.create(
        _lease(session),
        "chunk",
        "request",
        step,
        bytes([value]) * RGB_BYTES,
        struct.pack("<15f", *([0.1] * 15)),
        _identity(),
        episode_deadline,
        episode_deadline,
        "test-clock",
    )


class _Worker:
    def __init__(self):
        self.closed = False
        self.calls = []

    def info(self):
        return {"standalone_fake": True, "closed": self.closed}

    def owner_complete(self, *args):
        self.calls.append(args)
        return {"completed": True}

    def close(self, deadline_ns):
        self.closed = True
        return {"closed": True, "quiescent": True}


@pytest.fixture
def socket_root():
    with tempfile.TemporaryDirectory(prefix="psm23-ipc-") as directory:
        yield Path(directory)


@pytest.fixture
def server(socket_root):
    worker, secret = _Worker(), secrets.token_bytes(32)
    service = UnixVAEServer(
        worker, socket_root / "ipc" / "vae.sock", secret, max_data_connections=2, connection_timeout_s=0.2
    ).start()
    yield service, worker, secret
    service.close(_deadline())


@pytest.mark.parametrize("failed_lane", ["data", "control"])
def test_acceptor_start_failure_can_be_closed_without_joining_unstarted_thread(socket_root, monkeypatch, failed_lane):
    worker = _Worker()
    service = UnixVAEServer(worker, socket_root / "ipc" / "vae.sock", b"x" * 32)
    original = threading.Thread.start
    sentinel = RuntimeError("acceptor allocation sentinel")

    def fail_acceptor(thread):
        if thread.name == "vae-ipc-" + failed_lane:
            raise sentinel
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_acceptor)
    with pytest.raises(RuntimeError) as raised:
        service.start()
    assert raised.value is sentinel
    assert service.close(_deadline())["ipc_quiescent"]
    assert worker.closed
    assert not service.path.exists() and not Path(str(service.path) + ".ctl").exists()


def _raw(path, raw, *, deadline=None, claimed_size=None):
    deadline = _deadline() if deadline is None else deadline
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2)
        connection.connect(str(path))
        if claimed_size is None:
            _send(connection, raw, deadline, MAX_FRAME_WIRE_BYTES)
        else:
            connection.sendall(struct.pack("!I", claimed_size) + raw)
        return _receive(connection, deadline, MAX_FRAME_WIRE_BYTES)


def test_socket_permissions_and_explicit_owner_control_are_independent(server):
    service, worker, secret = server
    assert stat.S_IMODE(service.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(service.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(str(service.path) + ".ctl").stat().st_mode) == 0o600
    owner = OwnerVAEClient(service.path, secret)
    assert owner.complete("session", "chunk", 1, _deadline()) == {"completed": True}
    assert worker.calls == [("session", "chunk", 1)]
    sim = LocalVAEClient(service.path, "session", "sim-only-capability")
    assert sim.info(_deadline())["standalone_fake"]
    assert not hasattr(sim, "register") and not hasattr(sim, "issue") and not hasattr(sim, "complete")
    with pytest.raises(VAEIPCError) as result:
        OwnerVAEClient(service.path, b"x" * 32).complete("session", "chunk", 2, _deadline())
    assert result.value.code == "FORBIDDEN" and worker.calls == [("session", "chunk", 1)]


def test_data_lane_cannot_run_owner_operation_even_with_correct_secret(server):
    service, worker, secret = server
    request = {
        "op": "owner_complete",
        "deadline_ns": str(_deadline()),
        "control_capability": secret.hex(),
        "session_id": "session",
        "chunk_id": "chunk",
        "completed_through": 1,
    }
    reply = _raw(service.path, _json(request))
    assert reply["ok"] is False and reply["error"]["code"] == "FORBIDDEN"
    assert not worker.calls and secret.hex() not in json.dumps(reply)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"op":"info","op":"shutdown","deadline_ns":"1"}',
        b'{"op":"info","deadline_ns":"1","unexpected":true}',
        b'{"op":"info","deadline_ns":"1","bad":{"x":1,"x":2}}',
        b'{"op":"info","deadline_ns":1.5}',
        b'{"op":"info","deadline_ns":NaN}',
        b"[]",
    ],
)
def test_duplicate_extra_and_noncanonical_JSON_fields_are_rejected(server, raw):
    service, _, _ = server
    # Expired wire deadlines produce a transport timeout; make the valid deadline
    # current so malformed schema, rather than expiry, determines this test.
    raw = raw.replace(b'"1"', ('"' + str(_deadline()) + '"').encode())
    reply = _raw(Path(str(service.path) + ".ctl"), raw)
    assert reply["ok"] is False and reply["error"]["code"] == "INVALID"


@pytest.mark.parametrize("length", [0, MAX_FRAME_WIRE_BYTES + 1, 0xFFFFFFFF])
def test_declared_frame_size_is_checked_before_body_allocation(server, length):
    service, _, _ = server
    reply = _raw(service.path, b"", claimed_size=length)
    assert reply["ok"] is False and reply["error"]["code"] == "INVALID"


def test_full_partial_data_connections_do_not_block_control(server):
    service, _, _ = server
    partial = []
    try:
        for _ in range(2):
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(str(service.path))
            connection.sendall(struct.pack("!I", 1000))
            partial.append(connection)
        result = LocalVAEClient(service.path, "session", "sim").info(_deadline(1))
        assert result["standalone_fake"]
        with service._changed:
            assert len(service._handlers) <= 4
    finally:
        for connection in partial:
            connection.close()


def test_slow_trickle_does_not_refresh_accepted_connection_deadline(server):
    service, _, _ = server
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        connection.connect(str(service.path))
        connection.sendall(struct.pack("!I", 1000))
        stopped = threading.Event()

        def trickle():
            while not stopped.wait(0.02):
                try:
                    connection.sendall(b" ")
                except OSError:
                    return

        thread = threading.Thread(target=trickle)
        started = time.monotonic()
        thread.start()
        try:
            assert connection.recv(1) == b""
            assert time.monotonic() - started < 0.8
        finally:
            stopped.set()
            thread.join(1)
        assert not thread.is_alive()


def test_transport_retry_preserves_original_deadline_and_first_error(socket_root):
    client = LocalVAEClient(socket_root / "missing.sock", "session", "sim", retry_interval_s=0.001)
    deadline = _deadline(0.1)
    with pytest.raises(TimeoutError, match="original IPC deadline") as result:
        client.info(deadline)
    assert isinstance(result.value.__cause__, FileNotFoundError)
    assert time.monotonic_ns() >= deadline


def test_cleanup_preserves_replacement_socket_and_unrelated_file(server):
    service, _, _ = server
    old = service.path.with_name("old-owned.sock")
    service.path.rename(old)
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    marker = service.path.parent / "external.txt"
    marker.write_text("external")
    try:
        replacement.bind(str(service.path))
        inode = service.path.stat().st_ino
        assert service.close(_deadline())["acknowledged"]
        assert service.path.stat().st_ino == inode and marker.read_text() == "external"
    finally:
        replacement.close()
        service.path.unlink(missing_ok=True)
        old.unlink(missing_ok=True)


def test_shutdown_handler_is_joined_before_full_IPC_quiescence(server, monkeypatch):
    service, _, secret = server
    entered, release = threading.Event(), threading.Event()
    original = service._unlink_owned

    def held_cleanup():
        if threading.current_thread().name == "vae-ipc-request-control":
            entered.set()
            assert release.wait(3)
        original()

    monkeypatch.setattr(service, "_unlink_owned", held_cleanup)
    try:
        result = OwnerVAEClient(service.path, secret).shutdown(_deadline())
        assert result["closed"] and not result["ipc_quiescent"]
        assert entered.wait(3) and service.wait_closed(0)
        unproven = service.close(_deadline(0.05))
        assert not unproven["ipc_quiescent"] and not unproven["acknowledged"]
        release.set()
        proven = service.close(_deadline())
        assert proven["ipc_quiescent"] and proven["acknowledged"]
    finally:
        release.set()


def test_request_thread_start_failure_rolls_back_owned_resources_and_preserves_first_error(socket_root, monkeypatch):
    worker = _Worker()
    service = UnixVAEServer(worker, socket_root / "ipc" / "vae.sock", b"x" * 32).start()
    original = threading.Thread.start
    sentinel = RuntimeError("request thread allocation sentinel")

    def fail_handler_start(thread):
        if thread.name.startswith("vae-ipc-request-"):
            raise sentinel
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_handler_start)
    try:
        with pytest.raises((OSError, ConnectionError)):
            _raw(Path(str(service.path) + ".ctl"), _json({"op": "info", "deadline_ns": str(_deadline())}))
        assert service.wait_closed(1)
        with service._changed:
            assert not service._handlers and not service._connections
        with pytest.raises(RuntimeError) as result:
            service.close(_deadline())
        assert result.value is sentinel and worker.closed
        assert not service.path.exists() and not Path(str(service.path) + ".ctl").exists()
    finally:
        try:
            service.close(_deadline())
        except RuntimeError as error:
            assert error is sentinel


@pytest.mark.parametrize("kind", ["symlink", "public_directory", "existing_socket"])
def test_start_rejects_unsafe_or_foreign_paths(socket_root, kind):
    parent, foreign = socket_root / "ipc", socket_root / "foreign"
    foreign.mkdir(mode=0o700)
    existing = None
    if kind == "symlink":
        parent.symlink_to(foreign, target_is_directory=True)
    else:
        parent.mkdir(mode=0o755 if kind == "public_directory" else 0o700)
        if kind == "public_directory":
            # mkdir mode is filtered by the process umask (0077 in CPU CI).
            parent.chmod(0o755)
            assert stat.S_IMODE(parent.stat().st_mode) == 0o755
        if kind == "existing_socket":
            existing = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            existing.bind(str(parent / "vae.sock"))
    try:
        with pytest.raises((ValueError, PermissionError, FileExistsError)):
            UnixVAEServer(_Worker(), parent / "vae.sock", b"x" * 32).start()
        if existing is not None:
            assert (parent / "vae.sock").is_socket()
    finally:
        if existing is not None:
            existing.close()


_CHILD = r"""
import json, os, struct, sys, time
from cosmos_framework.inference.robocasa_vae_client import UnixVAEServer
from cosmos_framework.inference.robocasa_vae_worker import SharedVAEWorker, WorkerConfig

secret_fd, event_fd, release_fd = map(int, sys.argv[1:4])
path, mode = sys.argv[4:6]
raw = bytearray()
while True:
    chunk = os.read(secret_fd, 4096)
    if not chunk:
        break
    raw.extend(chunk)
    if len(raw) > 16384:
        raise ValueError("bounded inherited secret input required")
os.close(secret_fd)
credentials = json.loads(raw)
def event(value):
    os.write(event_fd, json.dumps(value, sort_keys=True).encode() + b"\n")
def encoder(frames):
    event({"event":"encoder_entered", "size":len(frames)})
    if mode == "blocked":
        os.read(release_fd, 1)
    if mode == "crash":
        os._exit(23)
    if mode == "failure":
        raise RuntimeError("first synthetic encoder failure")
    return [struct.pack("<96f", *([frame.rgb[0] / 255] * 96)) for frame in frames]
worker = SharedVAEWorker(encoder, credentials["identity"], bytes.fromhex(credentials["signing_key"]),
                         "test-key", "test-worker", clock_domain="test-clock",
                         config=WorkerConfig(bmax=1, wait_ms=0, max_items=1, max_bytes=1024*1024))
server = UnixVAEServer(worker, path, bytes.fromhex(credentials["control"]), connection_timeout_s=0.2).start()
with open("/proc/self/stat") as handle:
    process_stat = handle.read()
start_ticks = process_stat[process_stat.rfind(")") + 2:].split()[19]
event({"event":"ready", "pid":os.getpid(), "pgid":os.getpgrp(), "start_ticks":start_ticks, "info":worker.info()})
server.wait_closed()
receipt = server.close(time.monotonic_ns() + 2_000_000_000)
event({"event":"closed", "receipt":receipt, "info":worker.info()})
os.close(event_fd)
os.close(release_fd)
"""


def _event(fd, timeout=5):
    deadline, raw = time.monotonic() + timeout, bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], max(0, remaining))[0]:
            raise TimeoutError("bounded subprocess event missing")
        byte = os.read(fd, 1)
        if not byte:
            raise RuntimeError("subprocess event channel closed")
        if byte == b"\n":
            return json.loads(raw)
        raw.extend(byte)
        if len(raw) > 65536:
            raise ValueError("bounded subprocess event exceeded")


class _Process:
    def __init__(self, tmp_path, mode="normal", name="ipc"):
        self._cleaned = False
        self.path = tmp_path / name / "vae.sock"
        self.control, self.signing_key = secrets.token_bytes(32), secrets.token_bytes(32)
        secret_read, secret_write = os.pipe()
        event_read, event_write = os.pipe()
        release_read, release_write = os.pipe()
        self.events, self.release = event_read, release_write
        self.process = subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(secret_read), str(event_write), str(release_read), str(self.path), mode],
            pass_fds=(secret_read, event_write, release_read),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[2],
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1"},
        )
        for fd in (secret_read, event_write, release_read):
            os.close(fd)
        try:
            credentials = dict(identity=_identity(), control=self.control.hex(), signing_key=self.signing_key.hex())
            os.write(secret_write, _json(credentials))
        finally:
            os.close(secret_write)
        try:
            self.ready = _event(self.events)
            assert self.ready["event"] == "ready" and self.ready["pid"] == self.process.pid
            self.inodes = {
                path: (path.stat().st_dev, path.stat().st_ino) for path in (self.path, Path(str(self.path) + ".ctl"))
            }
            self.owner = OwnerVAEClient(self.path, self.control)
            self.sim = LocalVAEClient(self.path, "session", "sim-capability-test-only-32bytes")
        except BaseException:
            self.cleanup()
            raise

    def register(self, *, count=1):
        deadline = _deadline()
        self.owner.register(_lease(), "sim-capability-test-only-32bytes", deadline, deadline)
        self.owner.issue("session", "chunk", "request", 0, count, deadline)
        self.owner.complete("session", "chunk", count, deadline)
        return deadline

    def ready_result(self, key, deadline):
        while True:
            result = self.sim.poll(key, deadline)
            if result["status"] != "pending":
                return result
            if time.monotonic_ns() >= deadline:
                raise TimeoutError("bounded fake encoder result missing")
            time.sleep(0.001)

    def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        if self.process.poll() is None:
            try:
                os.write(self.release, b"x")
                if hasattr(self, "owner"):
                    self.owner.shutdown(_deadline(1))
                self.process.wait(timeout=2)
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=3)
        for fd in (self.events, self.release):
            try:
                os.close(fd)
            except OSError:
                pass
        for path, identity in getattr(self, "inodes", {}).items():
            try:
                current = path.lstat()
                if stat.S_ISSOCK(current.st_mode) and (current.st_dev, current.st_ino) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        self.process.communicate(timeout=1)


def test_real_subprocess_signed_receipt_retry_ACK_End_and_owned_cleanup(tmp_path, socket_root):
    child = _Process(socket_root)
    external = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
    try:
        deadline = child.register()
        frame = _frame(deadline)
        key = child.sim.submit(frame, deadline)
        assert child.sim.submit(frame, deadline) == key
        result = child.ready_result(key, deadline)
        assert result["status"] == "ready"
        receipt = result["feature"]
        assert verify_feature(receipt, child.signing_key)["header"]["lease"] == _lease()
        assert child.sim.ack(key, deadline)["acknowledged"]
        ended = child.sim.end(deadline)
        assert ended["acknowledged"] and ended["quiescent"]
        shutdown = child.owner.shutdown(deadline)
        assert shutdown["closed"] and shutdown["quiescent"] and not shutdown["ipc_quiescent"]
        child.process.wait(timeout=3)
        assert _event(child.events)["event"] == "encoder_entered"
        closed = _event(child.events)
        assert closed["event"] == "closed" and closed["receipt"]["ipc_quiescent"]
        assert closed["receipt"]["acknowledged"]
        assert child.process.returncode == 0 and external.poll() is None
        assert not child.path.exists() and not Path(str(child.path) + ".ctl").exists()
        assert child.control.hex() not in json.dumps(child.process.args)
        assert child.signing_key.hex() not in json.dumps(child.process.args)
        (tmp_path / "lifecycle.json").write_text(
            json.dumps(
                {
                    "pid": child.process.pid,
                    "pgid": child.ready["pgid"],
                    "start_ticks": child.ready["start_ticks"],
                    "returncode": child.process.returncode,
                    "external_pid": external.pid,
                    "external_alive": external.poll() is None,
                    "secret_delivery": "inherited_fd_only",
                    "end": ended,
                    "ipc_close": closed,
                    "ready_info": child.ready["info"],
                },
                sort_keys=True,
            )
        )
    finally:
        child.cleanup()
        external.terminate()
        external.wait(timeout=3)


def test_real_subprocess_full_encoder_does_not_block_control_and_unquiescent_End(socket_root):
    child = _Process(socket_root, "blocked")
    try:
        deadline = child.register(count=2)
        key = child.sim.submit(_frame(deadline), deadline)
        assert _event(child.events)["event"] == "encoder_entered"
        with pytest.raises(VAEIPCError):
            child.sim.submit(_frame(deadline, step=1), deadline)
        assert isinstance(child.sim.info(deadline), dict)
        assert child.owner.complete("session", "chunk", 2, deadline)
        ended = child.sim.end(_deadline(0.05))
        assert not ended["acknowledged"] and not ended["quiescent"]
        assert child.process.poll() is None and child.path.exists()
        os.write(child.release, b"x")
        assert child.sim.poll(key, deadline)["status"] in ("pending", "error")
    finally:
        child.cleanup()


def test_real_subprocess_first_encoder_failure_remains_error_not_feature(socket_root):
    child = _Process(socket_root, "failure")
    try:
        deadline = child.register()
        key = child.sim.submit(_frame(deadline), deadline)
        result = child.ready_result(key, deadline)
        assert result["status"] == "error" and "feature" not in result
        assert result == child.sim.poll(key, deadline)
        assert child.sim.end(deadline)["quiescent"]
    finally:
        child.cleanup()


def test_real_subprocess_null_session_capability_cannot_read_ack_end_or_submit(socket_root):
    child = _Process(socket_root)
    try:
        deadline = child.register()
        frame = _frame(deadline)
        key = child.sim.submit(frame, deadline)
        result = child.ready_result(key, deadline)
        assert result["status"] == "ready"
        for op in ("poll", "ack", "end", "submit"):
            fields = {"capability": None}
            if op == "submit":
                fields["frame"] = frame.to_wire()
                path = child.path
            else:
                fields["session_id"] = "session"
                if op != "end":
                    fields["job_key"] = key
                path = Path(str(child.path) + ".ctl")
            reply = _raw(path, _json({"op": op, "deadline_ns": str(deadline), **fields}))
            assert reply["ok"] is False and reply["error"]["code"] in ("FORBIDDEN", "INVALID")
        assert child.sim.poll(key, deadline) == result
        assert child.sim.ack(key, deadline)["acknowledged"]
        assert child.sim.end(deadline)["acknowledged"]
    finally:
        child.cleanup()


def test_real_subprocess_crash_restart_rejects_old_epoch_and_preserves_external(socket_root):
    old = _Process(socket_root, "crash")
    external = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
    replacement = None
    try:
        deadline = old.register()
        try:
            old.sim.submit(_frame(deadline), _deadline(0.3))
        except TimeoutError:
            # Process death can race delivery of the admission response. The
            # original caller budget remains fixed even when that ACK is lost.
            pass
        old.process.wait(timeout=3)
        assert old.process.returncode == 23 and external.poll() is None
        old.cleanup()
        replacement = _Process(socket_root, name="ipc")
        with pytest.raises(VAEIPCError):
            old.sim.poll("old-job", _deadline())
        deadline = replacement.register()
        key = replacement.sim.submit(_frame(deadline), deadline)
        receipt = replacement.ready_result(key, deadline)["feature"]
        with pytest.raises(ProtocolError):
            verify_feature(receipt, old.signing_key)
        assert receipt["header"]["worker_epoch"] != old.ready["info"]["worker_epoch"]
        assert external.poll() is None
    finally:
        old.cleanup()
        if replacement is not None:
            replacement.cleanup()
        external.terminate()
        external.wait(timeout=3)
