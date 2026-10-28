"""Bounded optional encoder subprocess. Never decodes media in the SR path."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from queue import Full


def _guarded_encoder(target, args, parent):
    import ctypes
    import signal

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG")
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)
    target(*args)


def _encode(queue, directory, fps, rollout):
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
    def __init__(self, directory, *, rollout=True, generated="none", fps=20, grace_s=2.0, target=None):
        self.generated, self.rollout, self.grace_s = generated, rollout, grace_s
        self.dropped = 0
        self.errors = []
        self.queue = None
        self.process = None
        self.forced = False
        self.helpers = []
        if rollout or generated != "none":
            ctx = mp.get_context("spawn")
            self.queue = ctx.Queue(maxsize=4)
            self.queue._ignore_epipe = True
            from multiprocessing import resource_tracker

            if resource_tracker._resource_tracker._pid is not None:
                self.helpers.append(resource_tracker._resource_tracker._pid)
            self.process = ctx.Process(
                target=_guarded_encoder,
                args=(target or _encode, (self.queue, str(directory), fps, rollout), os.getpid()),
                daemon=True,
            )
            try:
                self.process.start()
            except Exception as e:
                self.errors.append(repr(e))
                self.process = None
                self.queue.close()
                self.queue = None

    def _submit(self, kind, payload):
        if self.queue is None:
            return
        try:
            if self.process is None or not self.process.is_alive():
                raise RuntimeError("encoder exited")
            self.queue.put_nowait((kind, payload))
        except (Full, RuntimeError, OSError) as e:
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
                return
            self._submit("prediction", (index, video))

    def close(self):
        if self.process is not None:
            try:
                self.queue.put_nowait(None)
            except (Full, OSError):
                self.dropped += 1
            self.process.join(self.grace_s)
            if self.process.is_alive():
                self.forced = True
                self.process.terminate()
                self.process.join(self.grace_s)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(self.grace_s)
            if self.process.is_alive():
                raise RuntimeError("owned encoder survived SIGKILL")
            if self.process.exitcode != 0:
                self.errors.append(f"encoder exit {self.process.exitcode}")
            self.queue.cancel_join_thread()
            self.queue.close()
            self.queue._reader.close()
            if self.queue._thread is not None:
                self.queue._thread.join(self.grace_s)
                if self.queue._thread.is_alive():
                    self.forced = True
                    self.errors.append("encoder feeder thread did not stop")
        return {
            "dropped": self.dropped,
            "errors": self.errors,
            "independent_of_SR": True,
            "forced_encoder_cleanup": self.forced,
            "helper_pids": self.helpers,
        }
