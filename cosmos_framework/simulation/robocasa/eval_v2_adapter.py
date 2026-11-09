"""Native V2 adapter: original policy/action/Local loop with strict V2 boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import requests

from cosmos_framework.evaluation.robocasa_v2.admission import boolean
from cosmos_framework.evaluation.robocasa_v2.contract import EnvironmentOutcome
from cosmos_framework.evaluation.robocasa_v2.media import MediaSink
from cosmos_framework.inference.robocasa_feature_trace import TraceRecorder, use_trace
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
        arm = settings.get("single_latency_arm")
        self.trace = None
        if arm is not None:
            if arm not in ("A", "B", "C"):
                raise ValueError("single latency profiling requires explicit A/B/C arm")
            self.trace = TraceRecorder(
                max_events=settings.get("feature_trace_max_events", 8192),
                max_bytes=settings.get("feature_trace_max_bytes", 4 * 1024**2),
            )
            self.trace.event("single_latency_profile", arm=arm, egl_separate="unavailable")
        native.CAMERA_SET = "left_wrist"
        native.USE_BASE_ACTION = True
        native.BASE_ENCODING = "raw"

    def initialize(self):
        stage = "reset"
        try:
            with nullcontext() if self.trace is None else self.trace.span("initial_reset", stratum="load"):
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

    def execute(
        self,
        attempt,
        *,
        reset_index,
        directory,
        sampling=None,
        phase_callback=None,
        transaction_assignment=None,
        episode_deadline=None,
    ):
        if not self.admitted or self.observation is None:
            raise ValueError("scene must be admitted before policy/Local")
        self.admitted = False
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        trace_metadata = None
        if self.trace is not None and isinstance(transaction_assignment, dict):
            self.trace.clock_domain = transaction_assignment.get("prediction_clock_domain", self.trace.clock_domain)
        scene = {k: attempt[k] for k in ("task", "episode", "attempt_id", "stream_id", "seed", "manifest_digest")}
        scene.update(initial_success=False, reset_index=reset_index)
        if sampling is not None:
            scene["sampling"] = sampling
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
        first_local = None
        media_errors = []
        limit = self.settings.get("rollout_mp4_first_n_per_task")
        rollout_enabled = self.settings.get("save_rollout_mp4", True) and (limit is None or attempt["episode"] < limit)
        transactional = None
        if self.settings.get("execution_profile") == "phase_b":
            from cosmos_framework.simulation.robocasa.eval_v2_transactions import PolicyTransport

            transactional = PolicyTransport(self.settings, transaction_assignment, episode_deadline)
            client, http_client = transactional.local, transactional
        else:
            if transaction_assignment is not None:
                raise ValueError("legacy path cannot accept transactional assignment")
            client = RoboCasaLocalMemoryClient(enabled=True)
            http_client = requests.Session()
            http_client.trust_env = False
        try:
            strict = self.settings.get("save_generated_video") == "all"
            media_options = {}
            if strict:
                media_options = {
                    "drain_timeout_s": self.settings.get("media_drain_timeout_s", 300.0),
                    "identity": {k: attempt[k] for k in ("task", "episode", "attempt_id", "generation")},
                }
            sink = self.media_factory(
                directory / "media",
                rollout=rollout_enabled,
                generated=self.settings.get("save_generated_video", "none"),
                grace_s=self.settings.get("encoder_grace_s", 2.0),
                **media_options,
            )
        except Exception as error:
            media_errors.append(repr(error))
            sink = None

        def media_call(method, *args):
            try:
                if sink is not None:
                    return getattr(sink, method)(*args)
            except Exception as error:
                if len(media_errors) < 16:
                    media_errors.append(repr(error))

        def generated(index, result):
            if strict:
                if sink is None:
                    raise RuntimeError("MEDIA_INCOMPLETE: Encoder unavailable")
                return sink.prediction(index, result)
            return media_call("prediction", index, result)

        start = time.monotonic()
        policy_error = None
        with (
            (directory / "actions.jsonl").open("x") as actions,
            (directory / "predictions.jsonl").open("x") as predictions,
        ):

            def action(row):
                actions.write(json.dumps(row, allow_nan=False) + "\n")
                actions.flush()

            def prediction(result, latency):
                nonlocal replans, first_local
                if not isinstance(result.get("action"), list) or len(result["action"]) != 16:
                    raise ValueError("Action Server must return exactly16 raw15 actions")
                value = {
                    "replan": replans,
                    "source_step": len(outcome.checks),
                    "action": result["action"][:16],
                    "local_memory": result.get("local_memory"),
                    "latency_ms": latency,
                }
                if transactional is not None:
                    value["transaction"] = {
                        k: result[k]
                        for k in (
                            "protocol",
                            "request_key",
                            "payload_sha256",
                            "assignment",
                            "frontier",
                            "native_batch",
                            "media",
                        )
                    }
                if transactional is not None and "timing" in result:
                    value["transaction"]["timing"] = result["timing"]
                if first_local is None:
                    first_local = value["local_memory"]
                predictions.write(json.dumps(value, allow_nan=False) + "\n")
                predictions.flush()
                if transactional is None:
                    media_call("prediction", replans, result)
                else:
                    transactional.media(result, replans, generated)
                replans += 1

            def cleaned(value):
                nonlocal cleanup
                if transactional is not None and client is None:
                    # Common Session End is mandatory even when no Local client
                    # exists. Never turn a failed End into an episode success.
                    transactional.end()
                    value = {
                        **value,
                        "session_id": transactional.lease["session_id"],
                        "reset_ok": True,
                        "local_memory_mode": "off",
                        "local_state": "not_instantiated",
                    }
                cleanup = value
                atomic_json(directory / "local_memory.json", value)

            def progress(value):
                if phase_callback is not None:
                    phase_callback(value.get("stage"))
                # Rate-limit non-authoritative UI writes; Core actions/predictions
                # are flushed independently and never inferred from UI state.
                now = time.monotonic()
                if now - getattr(progress, "last", 0) >= 1 or value.get("event") != "stage":
                    atomic_json(directory / "progress.json", value)
                    progress.last = now

            try:
                with nullcontext() if self.trace is None else use_trace(self.trace):
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
                        on_frame=(lambda frame: media_call("frame", frame)) if rollout_enabled else None,
                        video_fps=20,
                        **({"trace": self.trace} if self.trace is not None else {}),
                    )
                if steps != len(outcome.checks) or success != any(outcome.checks):
                    raise ValueError("native result contradicts strict environment checks")
                if cleanup is None or cleanup.get("reset_ok") is not True:
                    raise RuntimeError("Local cleanup did not acknowledge reset")
                observation = outcome.finish(normal=True, cleanup_ok=True)
            except BaseException as error:
                policy_error = error
                if self.trace is not None:
                    self.trace.mark_incomplete("episode_execution_error")
                    self.trace.event("episode_error", error_type=type(error).__name__)
                raise
            finally:
                close_error = None
                try:
                    http_client.close()
                except BaseException as error:
                    close_error = error
                finally:
                    if transactional is not None:
                        media_errors.extend(transactional.media_diagnostics()["errors"])
                # Native loop has already attempted end+reset. Media cannot delay
                # that cleanup or replace a valid scientific outcome.
                for f in (actions, predictions):
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    media = sink.close() if sink is not None else {"errors": media_errors}
                except Exception as e:
                    media = {"errors": [repr(e)], "forced_encoder_cleanup": True, "process_cleanup_failed": True}
                media["parent_errors"] = media_errors
                media["rollout_requested"] = rollout_enabled
                media["rollout_mp4_first_n_per_task"] = limit
                if transactional is not None:
                    media["transaction_fetch"] = transactional.media_diagnostics()
                completion = None
                if strict:
                    saved = media.get("generated_media", {}).get("saved", [])
                    acknowledged = (
                        media["transaction_fetch"].get("completion", {}).get("acknowledged", 0)
                        if transactional is not None
                        else len(saved)
                    )
                    missing = sorted(set(range(replans)) - set(saved))
                    complete = (
                        not missing
                        and len(saved) == replans
                        and acknowledged == replans
                        and media.get("generated_media", {}).get("status") == "COMPLETE"
                        and (
                            transactional is None
                            or media["transaction_fetch"].get("completion", {}).get("status") == "COMPLETE"
                        )
                    )
                    completion = {
                        "schema": "psm_generated_media_completion_v1",
                        "status": "COMPLETE" if complete else "MEDIA_INCOMPLETE",
                        "task": attempt["task"],
                        "episode": attempt["episode"],
                        "attempt_id": attempt["attempt_id"],
                        "expected": replans,
                        "saved": len(saved),
                        "acknowledged": acknowledged,
                        "server_ack_required": transactional is not None,
                        "missing": missing,
                    }
                    media["generated_media_completion"] = completion
                    try:
                        atomic_json(directory / "generated_media_completion.json", completion)
                    except OSError as error:
                        media_errors.append("MEDIA_INCOMPLETE: completion record: " + repr(error))
                try:
                    atomic_json(directory / "media_diagnostics.json", media)
                except OSError:
                    pass
                if self.trace is not None:
                    if close_error is not None:
                        self.trace.mark_incomplete("episode_cleanup_error")
                    try:
                        try:
                            full_metadata = self.trace.dump(directory / "feature_trace.json")
                        except Exception as error:
                            full_metadata = {
                                "complete": False,
                                "error_type": type(error).__name__,
                                "count": self.trace.snapshot()["count"],
                            }
                        atomic_json(directory / "feature_trace_metadata.json", full_metadata)
                        if "sha256" in full_metadata:
                            trace_metadata = {key: full_metadata[key] for key in ("path", "sha256", "count")}
                        else:
                            trace_metadata = {
                                "path": "feature_trace_metadata.json",
                                "count": full_metadata["count"],
                                "sha256": hashlib.sha256(
                                    (directory / "feature_trace_metadata.json").read_bytes()
                                ).hexdigest(),
                            }
                    except Exception as error:
                        primary = policy_error if policy_error is not None else close_error
                        if primary is None:
                            raise
                        primary.profiling_error = {"type": type(error).__name__}
                if close_error is not None:
                    if policy_error is not None:
                        policy_error.cleanup_error = repr(close_error)
                        # The original policy exception is already propagating.
                    else:
                        raise close_error
        return {
            "local_session": {
                "session_id": cleanup["session_id"],
                "episode_id": first_local["episode_id"] if client is not None else transactional.lease["episode_id"],
                "start_frontier": first_local["consumer_step"] if client is not None else 0,
                "reset_ack": cleanup["reset_ok"],
            },
            "observation": observation,
            "instruction": prompt,
            "replans": replans,
            "elapsed_seconds": time.monotonic() - start,
            **({"single_latency_trace": trace_metadata} if self.trace is not None else {}),
            "media_process_fault": media.get("process_cleanup_failed", media.get("forced_encoder_cleanup", False))
            or (transactional is not None and transactional.media_process_fault),
            "media_helper_pids": media.get("helper_pids", []),
            **({"generated_media_completion": completion} if strict else {}),
        }
