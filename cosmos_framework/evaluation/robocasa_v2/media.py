"""Bounded diagnostic encoder; all-mode durable completion is independent of SR."""

from __future__ import annotations

import errno
import json
import multiprocessing as mp
import os
import socket
import threading
import time
from pathlib import Path
from queue import Full

from cosmos_framework.evaluation.robocasa_v2.generated_video import canonical, read_receipt, sha, write_generated


def _diagnostic(errors, error):
    if len(errors) < 16:
        errors.append(repr(error)[:2048] if not isinstance(error, str) else error[:2048])


class _ReaderClose:
    """Parent-only close shared with QueueFeederThread, installed after spawn.

    Connection.close() is idempotent but not thread-safe: its _handle test and
    os.close can interleave. Serializing both callers prevents the sentinel path
    from raising EBADF and then incorrectly releasing Queue's semaphore.
    """

    def __init__(self, close, errors, grace):
        self.close, self.errors, self.grace = close, errors, grace
        self.lock = threading.Lock()
        self.done = self.failed = False

    def __call__(self):
        if not self.lock.acquire(timeout=self.grace):
            self.failed = True
            _diagnostic(self.errors, "encoder reader close lock did not quiesce")
            return
        try:
            if self.done:
                return
            try:
                self.close()
            except OSError as error:
                # EBADF is a descriptor diagnostic, never evidence that a
                # process died. Actual encoder/feeder liveness is checked later.
                self.failed = self.failed or error.errno != errno.EBADF
                _diagnostic(self.errors, error)
            except Exception as error:
                self.failed = True
                _diagnostic(self.errors, error)
            finally:
                self.done = True
        finally:
            self.lock.release()


def _guarded_encoder(target, args, parent):
    import ctypes
    import signal

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG")
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)
    target(*args)


def _encode(queue, directory, fps, rollout, acknowledgements=None):
    # Encoder gets neither CUDA visibility nor a Local session. Parent waits only
    # AFTER native Local cleanup, and forcibly reaps stalled encoders.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import imageio_ffmpeg

    executable = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PSM_EVAL_REAL_FFMPEG"] = executable
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(Path(__file__).with_name("ffmpeg_guard.py"))
    import imageio.v2 as imageio

    from cosmos_framework.simulation.robocasa.eval_utils import decode_pred_video

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        while True:
            item = queue.get()
            if item is None:
                break
            kind, data = item
            if kind == "frame" and rollout:
                if writer is None:
                    writer = imageio.get_writer(directory / "rollout.mp4", fps=fps)
                writer.append_data(data)
            elif kind == "strict_prediction":
                index, binding = data["index"], data["binding"]
                try:
                    receipt = write_generated(directory, data, fps)
                    ack = {"status": "VIDEO_SAVED", "index": index, "binding_sha256": receipt["binding_sha256"]}
                except Exception as error:
                    ack = {
                        "status": "VIDEO_FAILED",
                        "index": index,
                        "binding_sha256": sha(binding),
                        "error": repr(error)[:1024],
                    }
                # One atomic, bounded local datagram; never an unbounded feeder.
                raw = canonical(ack)
                if len(raw) > 4096:
                    raise ValueError("bounded Encoder completion datagram required")
                acknowledgements.send(raw)
            elif kind == "prediction":
                index, video = data
                target = directory / "generated"
                target.mkdir(exist_ok=True)
                with imageio.get_writer(target / f"replan_{index:04d}.mp4", fps=fps) as generated:
                    for encoded in video:
                        import base64
                        import io

                        from PIL import Image

                        raw = base64.b64decode(encoded.split(",", 1)[-1])
                        with Image.open(io.BytesIO(raw)) as header:
                            if header.width * header.height > 2 * 1024 * 1024:
                                raise ValueError("oversized diagnostic frame")
                        for frame in decode_pred_video([encoded]):
                            generated.append_data(frame)
    finally:
        if writer is not None:
            writer.close()


class MediaSink:
    def __init__(
        self,
        directory,
        *,
        rollout=True,
        generated="none",
        fps=20,
        grace_s=2.0,
        target=None,
        drain_timeout_s=300.0,
        identity=None,
    ):
        self.generated, self.rollout, self.grace_s = generated, rollout, grace_s
        self.dropped = 0
        self.errors = []
        self.queue = None
        self.process = None
        self.forced = False
        self.helpers = []
        self.directory = Path(directory)
        self.strict = generated == "all"
        self.drain_timeout_s = drain_timeout_s
        if type(drain_timeout_s) not in (int, float) or not 0 < drain_timeout_s <= 3600:
            raise ValueError("finite bounded media drain timeout required")
        self.identity = dict(identity or {})
        self.saved, self.pending = {}, {}
        self.incomplete = False
        self._ack = self._ack_sender = None
        self._drain_deadline = None
        self._closed = False
        self._result = None
        self._reader_guard = None
        if rollout or generated != "none":
            ctx = mp.get_context("spawn")
            self.queue = ctx.Queue(maxsize=4)
            self.queue._ignore_epipe = True
            if self.strict:
                self._ack, self._ack_sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
                self._ack_sender.settimeout(min(grace_s, drain_timeout_s))
            from multiprocessing import resource_tracker

            if resource_tracker._resource_tracker._pid is not None:
                self.helpers.append(resource_tracker._resource_tracker._pid)
            arguments = (self.queue, str(directory), fps, rollout)
            if self.strict:
                arguments += (self._ack_sender,)
            self.process = ctx.Process(
                target=_guarded_encoder,
                args=(target or _encode, arguments, os.getpid()),
                daemon=True,
            )
            try:
                self.process.start()
                if self._ack_sender is not None:
                    self._ack_sender.close()
                # Install only after Queue was pickled into the child, before
                # the first parent put() can start its feeder. No stdlib patch.
                self._reader_guard = _ReaderClose(self.queue._reader.close, self.errors, self.grace_s)
                self.queue._reader.close = self._reader_guard
            except Exception as e:
                _diagnostic(self.errors, e)
                if self.process.pid is None:
                    self.process = None
                if self._reader_guard is None:
                    self._reader_guard = _ReaderClose(self.queue._reader.close, self.errors, self.grace_s)
                    self.queue._reader.close = self._reader_guard
                self.close()

    def _submit(self, kind, payload):
        if self._closed:
            self.dropped += 1
            return
        if self.queue is None:
            return
        try:
            if self.process is None or not self.process.is_alive():
                raise RuntimeError("encoder exited")
            self.queue.put_nowait((kind, payload))
        except (Full, RuntimeError, OSError, ValueError) as e:
            self.dropped += 1
            if len(self.errors) < 16:
                self.errors.append(type(e).__name__ + ": " + str(e))

    def frame(self, frame):
        if self.rollout:
            if getattr(frame, "nbytes", 10**9) > 2 * 1024 * 1024:
                self.dropped += 1
            else:
                self._submit("frame", frame)

    def prediction(self, index, result):
        if self.generated == "all" or (self.generated == "first" and index == 0):
            video = result.get("video")
            if (
                not isinstance(video, list)
                or not 1 <= len(video) <= 65
                or any(not isinstance(x, str) for x in video)
                or sum(len(x) for x in video) > 32 * 1024 * 1024
            ):
                if len(self.errors) < 16:
                    self.errors.append("missing/oversized optional generated video")
                if self.strict:
                    self.incomplete = True
                    raise RuntimeError("MEDIA_INCOMPLETE: missing/oversized generated video")
                return
            if self.strict:
                return self._save_prediction(index, video, result.get("_media_binding"), result.get("_media_deadline"))
            self._submit("prediction", (index, video))

    def _save_prediction(self, index, video, transfer, upstream_deadline=None):
        binding = {"episode": self.identity, "index": index, "transfer": transfer, "video_sha256": sha(video)}
        if index in self.saved:
            try:
                return read_receipt(self.directory, index, binding)
            except Exception as error:
                self.incomplete = True
                _diagnostic(self.errors, "MEDIA_INCOMPLETE: replay: " + repr(error))
                raise RuntimeError("MEDIA_INCOMPLETE: receipt replay mismatch") from error
        if self._closed or self.incomplete or type(index) is not int or not 0 <= index <= 100000:
            self.incomplete = True
            raise RuntimeError("MEDIA_INCOMPLETE: Encoder closed/failed or invalid replan")
        deadline = time.monotonic() + self.drain_timeout_s
        if upstream_deadline is not None:
            deadline = min(deadline, upstream_deadline)
        self._drain_deadline = deadline
        self.pending[index] = binding
        try:
            while True:
                if self.process is None or not self.process.is_alive():
                    raise RuntimeError("Encoder exited before VIDEO_SAVED")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Encoder queue deadline exhausted")
                try:
                    self.queue.put(
                        ("strict_prediction", {"index": index, "video": video, "binding": binding}),
                        timeout=min(0.05, remaining),
                    )
                    break
                except Full:
                    continue
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Encoder VIDEO_SAVED deadline exhausted")
                self._ack.settimeout(min(0.05, remaining))
                try:
                    raw = self._ack.recv(4097)
                except socket.timeout:
                    if self.process is None or not self.process.is_alive():
                        raise RuntimeError("Encoder exited before VIDEO_SAVED")
                    continue
                if len(raw) > 4096:
                    raise ValueError("oversized Encoder acknowledgement")
                ack = json.loads(raw)
                if ack.get("index") != index or ack.get("binding_sha256") != sha(binding):
                    raise ValueError("foreign/late Encoder acknowledgement")
                if ack.get("status") != "VIDEO_SAVED":
                    raise RuntimeError("Encoder VIDEO_FAILED: " + str(ack.get("error")))
                receipt = read_receipt(self.directory, index, binding)
                self.saved[index] = receipt
                del self.pending[index]
                return receipt
        except Exception as error:
            self.incomplete = True
            _diagnostic(self.errors, "MEDIA_INCOMPLETE: " + repr(error))
            raise RuntimeError("MEDIA_INCOMPLETE: " + str(error)) from error

    def close(self):
        if self._result is not None:
            return dict(self._result, errors=list(self._result["errors"]), helper_pids=list(self.helpers))
        self._closed = True
        encoder_alive = feeder_alive = fd_failed = False
        if self.process is not None:
            try:
                if self.strict:
                    deadline = self._drain_deadline or (time.monotonic() + self.drain_timeout_s)
                    self.queue.put(None, timeout=max(0, deadline - time.monotonic()))
                else:
                    self.queue.put_nowait(None)
            except (Full, OSError, ValueError) as error:
                self.dropped += 1
                _diagnostic(self.errors, error)
            self.process.join(max(0, deadline - time.monotonic()) if self.strict else self.grace_s)
            if self.process.is_alive():
                self.forced = True
                if self.strict:
                    self.incomplete = True
                    _diagnostic(self.errors, "MEDIA_INCOMPLETE: Encoder drain deadline exhausted")
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
                self.process.join(self.grace_s)
            if self.process.is_alive():
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                self.process.join(self.grace_s)
            encoder_alive = self.process.is_alive()
            if encoder_alive:
                _diagnostic(self.errors, "owned encoder survived SIGKILL")
            elif self.process.exitcode != 0:
                _diagnostic(self.errors, f"encoder exit {self.process.exitcode}")
        if self.queue is not None:
            try:
                self.queue.cancel_join_thread()  # Never an unbounded join at interpreter exit.
                self.queue.close()
            except (OSError, ValueError) as error:
                fd_failed = True
                _diagnostic(self.errors, error)
            feeder = self.queue._thread
            if feeder is not None:
                feeder.join(self.grace_s)  # Normal path: feeder owns reader_close().
            if feeder is not None and feeder.is_alive():
                # A dead consumer can leave send_bytes blocked on a full pipe.
                # Use the SAME serialized close as the feeder, then wait again.
                self.forced = True
                self._reader_guard()
                feeder.join(self.grace_s)
            feeder_alive = feeder is not None and feeder.is_alive()
            if feeder_alive:
                _diagnostic(self.errors, "encoder feeder thread did not stop")
            else:
                try:
                    self.queue._reader.close()
                    self.queue._writer.close()
                except OSError as error:
                    fd_failed = fd_failed or error.errno != errno.EBADF
                    _diagnostic(self.errors, error)
                if self._reader_guard is not None:
                    fd_failed = fd_failed or self._reader_guard.failed
                    # Drop the instance override only after the feeder is gone;
                    # this breaks its Connection/guard reference cycle.
                    del self.queue._reader.close
        for channel in (self._ack, self._ack_sender):
            if channel is not None:
                channel.close()
        self._result = {
            "dropped": self.dropped,
            "errors": list(self.errors),
            "independent_of_SR": True,
            "forced_encoder_cleanup": self.forced,
            "process_cleanup_failed": encoder_alive or feeder_alive or fd_failed,
            "encoder_alive": encoder_alive,
            "feeder_alive": feeder_alive,
            "helper_pids": list(self.helpers),
        }
        if self.strict:
            self._result["generated_media"] = {
                "status": "MEDIA_INCOMPLETE" if self.incomplete or self.pending else "COMPLETE",
                "saved": sorted(self.saved),
                "pending": sorted(self.pending),
                "encoder_done_receipts": len(self.saved),
            }
        return dict(self._result, errors=list(self._result["errors"]))
