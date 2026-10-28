"""Native V2 adapter: original policy/action/Local loop with strict V2 boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from cosmos_framework.evaluation.robocasa_v2.admission import boolean
from cosmos_framework.evaluation.robocasa_v2.contract import EnvironmentOutcome
from cosmos_framework.evaluation.robocasa_v2.media import MediaSink
from cosmos_framework.simulation.robocasa import closed_loop_eval as native
from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient
from cosmos_framework.utils.rollout_evidence import atomic_json


class InvalidInitialSuccess(ValueError):
    pass


class NativeEpisode:
    def __init__(self, env, settings, *, policy=native.run_policy, media_factory=MediaSink):
        self.env, self.settings, self.policy, self.media_factory = env, settings, policy, media_factory
        self.observation = None
        self.admitted = False
        native.CAMERA_SET = "left_wrist"
        native.USE_BASE_ACTION = True
        native.BASE_ENCODING = "raw"

    def initialize(self):
        stage = "reset"
        try:
            self.observation = self.env.reset()
            stage = "initial_success"
            try:
                initial = boolean(self.env._check_success())
            except ValueError as e:
                raise InvalidInitialSuccess(str(e)) from e
            self.admitted = not initial
            return {"initial_success": initial, "stage": stage, "error": None}
        except Exception as error:
            self.admitted = False
            # Only known pure sampling/typed-precheck errors with a positive
            # physics probe may retry. EGL/device/unknown exceptions safe-stop.
            pure = isinstance(error, InvalidInitialSuccess) or (
                type(error).__name__ == "RandomizationError" and type(error).__module__.startswith("robosuite.")
            )
            proof = {"physics_finite": False, "forward_ok": False, "sampling_error_class": pure}
            if pure:
                try:
                    sim = self.env.sim
                    proof["physics_finite"] = bool(
                        np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()
                    )
                    sim.forward()
                    proof["forward_ok"] = True
                except Exception:
                    pass
            return {
                "initial_success": None,
                "stage": stage,
                "error": {"type": type(error).__name__, "message": str(error)},
                "safe_to_retry": pure and proof["physics_finite"] and proof["forward_ok"],
                "resource_proof": proof,
            }

    def execute(self, attempt, *, reset_index, directory):
        if not self.admitted or self.observation is None:
            raise ValueError("scene must be admitted before policy/Local")
        self.admitted = False
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        scene = {k: attempt[k] for k in ("task", "episode", "attempt_id", "stream_id", "seed", "manifest_digest")}
        scene.update(initial_success=False, reset_index=reset_index)
        metadata = self.env.get_ep_meta()
        instruction = metadata.get("lang")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("actual Episode language instruction is required")
        try:
            raw = json.dumps(metadata, default=str, allow_nan=False, ensure_ascii=False).encode()
            scene["episode_metadata"] = (
                json.loads(raw) if len(raw) <= 65536 else {"sha256": hashlib.sha256(raw).hexdigest(), "truncated": True}
            )
        except (ValueError, TypeError) as error:
            scene["metadata_diagnostic_error"] = repr(error)
        atomic_json(directory / "scene_provenance.json", scene)
        outcome = EnvironmentOutcome(self.settings["horizon"])
        replans = 0
        cleanup = None
        media_errors = []
        try:
            sink = self.media_factory(
                directory / "media",
                rollout=self.settings.get("save_rollout_mp4", True),
                generated=self.settings.get("save_generated_video", "none"),
                grace_s=self.settings.get("encoder_grace_s", 2.0),
            )
        except Exception as error:
            media_errors.append(repr(error))
            sink = None

        def media_call(method, *args):
            try:
                if sink is not None:
                    getattr(sink, method)(*args)
            except Exception as error:
                if len(media_errors) < 16:
                    media_errors.append(repr(error))

        client = RoboCasaLocalMemoryClient(enabled=True)
        http_client = requests.Session()
        http_client.trust_env = False
        start = time.monotonic()
        with (
            (directory / "actions.jsonl").open("x") as actions,
            (directory / "predictions.jsonl").open("x") as predictions,
        ):

            def action(row):
                actions.write(json.dumps(row, allow_nan=False) + "\n")
                actions.flush()

            def prediction(result, latency):
                nonlocal replans
                if not isinstance(result.get("action"), list) or len(result["action"]) != 16:
                    raise ValueError("Action Server must return exactly16 raw15 actions")
                value = {
                    "replan": replans,
                    "source_step": len(outcome.checks),
                    "action": result["action"][:16],
                    "local_memory": result.get("local_memory"),
                    "latency_ms": latency,
                }
                predictions.write(json.dumps(value, allow_nan=False) + "\n")
                predictions.flush()
                media_call("prediction", replans, result)
                replans += 1

            def cleaned(value):
                nonlocal cleanup
                cleanup = value
                atomic_json(directory / "local_memory.json", value)

            def progress(value):
                # Rate-limit non-authoritative UI writes; Core actions/predictions
                # are flushed independently and never inferred from UI state.
                now = time.monotonic()
                if now - getattr(progress, "last", 0) >= 1 or value.get("event") != "stage":
                    atomic_json(directory / "progress.json", value)
                    progress.last = now

            try:
                success, steps, prompt = self.policy(
                    self.env,
                    server_url=self.settings["server_url"],
                    image_size=256,
                    action_horizon=16,
                    max_steps=self.settings["horizon"],
                    latch=1,
                    timeout=self.settings["predict_timeout_s"],
                    use_state=True,
                    local_memory_client=client,
                    initial_observation=self.observation,
                    success_check=outcome.check_after_step,
                    on_prediction=prediction,
                    on_action=action,
                    on_cleanup=cleaned,
                    on_progress=progress,
                    language_instruction=instruction,
                    http_client=http_client,
                    on_frame=lambda frame: media_call("frame", frame),
                    video_fps=20,
                )
                if steps != len(outcome.checks) or success != any(outcome.checks):
                    raise ValueError("native result contradicts strict environment checks")
                if cleanup is None or cleanup.get("reset_ok") is not True:
                    raise RuntimeError("Local cleanup did not acknowledge reset")
                observation = outcome.finish(normal=True, cleanup_ok=True)
            finally:
                http_client.close()
                # Native loop has already attempted end+reset. Media cannot delay
                # that cleanup or replace a valid scientific outcome.
                for f in (actions, predictions):
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    media = sink.close() if sink is not None else {"errors": media_errors}
                except Exception as e:
                    media = {"errors": [repr(e)], "forced_encoder_cleanup": True}
                media["parent_errors"] = media_errors
                try:
                    atomic_json(directory / "media_diagnostics.json", media)
                except OSError:
                    pass
        return {
            "observation": observation,
            "instruction": prompt,
            "replans": replans,
            "elapsed_seconds": time.monotonic() - start,
            "media_process_fault": media.get("forced_encoder_cleanup", False),
            "media_helper_pids": media.get("helper_pids", []),
        }
