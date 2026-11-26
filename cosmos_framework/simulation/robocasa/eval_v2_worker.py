"""Persistent one-Task native environment. Started only by the V2 coordinator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from cosmos_framework.evaluation.robocasa_v2.wire import receive, send


def serve(channel, cfg, *, factory=None):
    started = time.perf_counter()
    timings = {}
    # Factory is a Python CPU test boundary, never selectable via production CLI.
    from cosmos_framework.evaluation.robocasa_v2 import generated_video, media
    from cosmos_framework.simulation.robocasa import closed_loop_eval as native
    from cosmos_framework.simulation.robocasa import egl_lifecycle, render_devices
    from cosmos_framework.simulation.robocasa.eval_v2_adapter import NativeEpisode
    from cosmos_framework.simulation.robocasa.eval_v2_slot import SimSlot

    timings["imports"] = (time.perf_counter() - started) * 1000

    slot = None
    lifecycle = None
    cleanup_attempted = False
    cleanup_response = None
    primary_error = None

    def cleanup():
        nonlocal cleanup_attempted, cleanup_response
        if cleanup_attempted:
            return cleanup_response
        # Never retry an uncertain native free, including when slot.close raises.
        cleanup_attempted = True
        try:
            cleanup_response = slot.close() if slot is not None else {"closed": True, "clean": True}
            if lifecycle is not None:
                cleanup_response["egl_lifecycle"] = lifecycle.finish()
            return cleanup_response
        except BaseException as error:
            if lifecycle is not None:
                try:
                    lifecycle.failure(error)
                except BaseException:
                    pass  # Preserve the real cleanup error and nonzero process exit.
            raise

    os.set_inheritable(channel.fileno(), False)
    try:
        if factory is None:
            import importlib.metadata

            if {name: importlib.metadata.version(name) for name in cfg["simulation_libraries"]} != cfg[
                "simulation_libraries"
            ]:
                raise ValueError("simulation interpreter libraries changed after preflight")
            modules = {
                "worker": Path(__file__),
                "native": Path(native.__file__),
                "adapter": Path(sys.modules[NativeEpisode.__module__].__file__),
                "slot": Path(sys.modules[SimSlot.__module__].__file__),
                "rendering": Path(render_devices.__file__),
                "egl_lifecycle": Path(egl_lifecycle.__file__),
                "egl_lifecycle_core": egl_lifecycle.CORE_PATH,
                "media": Path(media.__file__),
                "generated_video": Path(generated_video.__file__),
            }
            if cfg["settings"].get("execution_profile") == "phase_b":
                from cosmos_framework.simulation.robocasa import eval_v2_transactions

                modules["transactions"] = Path(eval_v2_transactions.__file__)
            g2_profile = cfg["settings"].get("g2_profile", "legacy")
            if g2_profile != "legacy":
                if (
                    g2_profile not in ("g2_r1", "g2_r2")
                    or cfg["settings"].get("execution_profile") != "phase_b"
                    or cfg["settings"].get("local_memory_mode") != "required"
                    or cfg["settings"].get("local_evidence_profile", "server_vae") != "server_vae"
                ):
                    raise ValueError("G2 actor requires an assigned Phase B Local-TTT session with server VAE")
                from importlib.util import find_spec

                runtime_spec = find_spec("g2.online.runtime")
                if runtime_spec is None or runtime_spec.origin is None:
                    raise ValueError("missing Root G2 runtime origin")
                modules["g2_runtime"] = Path(runtime_spec.origin)
            if cfg["settings"].get("environment_runtime") is True:
                from importlib.util import find_spec
                from inspect import signature

                if g2_profile != "g2_r2" or "on_env_step" not in signature(native.run_policy).parameters:
                    raise ValueError("W1 Actor requires R2 and the W2-owned env.step seam")
                for name, module in {
                    "w1_environment___init__": "environment",
                    "w1_environment_contracts": "environment.contracts",
                    "w1_environment_robocasa": "environment.robocasa",
                    "w1_environment_runtime": "environment.runtime",
                    "w1_environment_spatial": "environment.spatial",
                    "w1_nav_session": "g2.online.nav_session",
                }.items():
                    spec = find_spec(module)
                    if spec is None or spec.origin is None:
                        raise ValueError(f"missing Root W1 source origin: {module}")
                    modules[name] = Path(spec.origin)
            evidence_profile = cfg["settings"].get("local_evidence_profile", "server_vae")
            if evidence_profile not in ("server_vae", "sim_shared_vae"):
                raise ValueError("unknown Actor source profile")
            if evidence_profile == "sim_shared_vae":
                if (
                    cfg["settings"].get("execution_profile") != "phase_b"
                    or cfg["settings"].get("local_memory_mode") != "required"
                ):
                    raise ValueError("C Actor source profile requires phase_b and required Local-TTT")
                from importlib import import_module

                # The allowlist is source-owned, never inferred from supplied labels.
                for name, module in {
                    "shared_client": "cosmos_framework.simulation.robocasa.sim_shared_vae_client",
                    "local_client": "cosmos_framework.simulation.robocasa.local_memory_client",
                    "transaction_client": "cosmos_framework.inference.robocasa_transaction_client",
                    "feature_trace": "cosmos_framework.inference.robocasa_feature_trace",
                    "feature_evidence": "cosmos_framework.inference.robocasa_feature_evidence",
                    "session_transactions": "cosmos_framework.inference.robocasa_session_transactions",
                    "batch_transactions": "cosmos_framework.inference.robocasa_batch_transactions",
                    "vae_protocol": "cosmos_framework.inference.robocasa_vae_protocol",
                    "vae_worker": "cosmos_framework.inference.robocasa_vae_worker",
                    "vae_client": "cosmos_framework.inference.robocasa_vae_client",
                }.items():
                    modules[name] = Path(import_module(module).__file__)
            if set(cfg["source_files"]) != set(modules):
                raise ValueError("incomplete native source witnesses")
            for name, file in modules.items():
                expected = cfg["source_files"][name]
                if (
                    str(file.resolve()) != expected["path"]
                    or hashlib.sha256(file.read_bytes()).hexdigest() != expected["sha256"]
                ):
                    raise ValueError("native import origin/source mismatch")
        stage = time.perf_counter()
        rendering = render_devices.configure(cfg["rendering"])
        timings["render_proof"] = (time.perf_counter() - stage) * 1000
        cfg["rendering"] = rendering
        if factory is None:
            stage = time.perf_counter()
            runtime = native.verify_runtime(SimpleNamespace(**cfg["runtime_args"]), next(iter(cfg["tasks"])))
            if runtime is None or any(
                runtime["protocol"]["official_horizons"][task] != spec["horizon"] for task, spec in cfg["tasks"].items()
            ):
                raise ValueError("missing/incorrect authoritative runtime horizons")
            timings["vendor_attestation"] = (time.perf_counter() - stage) * 1000
        # All native G1 No-TTT/Local-TTT and G2 workers share this admission.
        # Native vendor/device checks precede V5; no environment exists yet.
        stage = time.perf_counter()
        lifecycle = egl_lifecycle.start(cfg, test_factory=factory is not None)
        timings["egl_lifecycle"] = (time.perf_counter() - stage) * 1000
        slot = SimSlot(cfg, factory=factory)
        send(
            channel,
            {
                "kind": "READY",
                "startup_timing_ms": timings,
                "nonce": cfg["nonce"],
                "pid": os.getpid(),
                "physical_slot_id": cfg["physical_slot_id"],
                "state": slot.state,
                "local_memory_mode": cfg["settings"].get("local_memory_mode", "required"),
                **({"g2_profile": g2_profile} if factory is None and g2_profile != "legacy" else {}),
                "runtime_digest": runtime["digest"] if factory is None else "CPU_TEST_FACTORY",
                "rendering": rendering,
                "egl_lifecycle": lifecycle.startup_receipt,
            },
        )
        while True:
            message = receive(channel)
            if message.get("nonce") != cfg["nonce"]:
                raise ValueError("actor request identity mismatch")
            op = message.get("op")
            response = {"request_id": message["request_id"], "nonce": cfg["nonce"]}
            stage = time.perf_counter()
            if op == "start_episode":
                response.update(
                    slot.start_episode(
                        message["attempt"], grant=message.get("grant"), health_only=message.get("health_only", False)
                    )
                )
            elif op == "initialize":
                response.update(slot.initialize())
            elif op == "finish_without_session":
                response.update(slot.finish_without_session())
            elif op == "execute":
                response.update(
                    slot.execute(
                        message["attempt"],
                        reset_index=message["reset_index"],
                        directory=message["directory"],
                        transaction_assignment=message.get("transaction_assignment"),
                        episode_deadline=message.get("episode_deadline"),
                    )
                )
            elif op == "close":
                response.update(cleanup())
                send(channel, response)
                return
            else:
                raise ValueError("unknown actor operation")
            response["operation_ms"] = (time.perf_counter() - stage) * 1000
            send(channel, response)
    except EOFError:
        pass
    except BaseException as error:
        primary_error = error
        if lifecycle is not None:
            try:
                lifecycle.failure(error)
            except BaseException:
                pass  # A failed evidence write must not hide the original error.
        try:
            send(channel, {"kind": "FATAL", "nonce": cfg["nonce"], "error": f"{type(error).__name__}: {error}"})
        except OSError:
            pass
        raise
    finally:
        try:
            if not cleanup_attempted:
                try:
                    cleanup()
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    # The cleanup helper records its own failure. Preserve the
                    # initiating exception as the worker's propagated error.
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(f"V5 cleanup also failed: {cleanup_error!r}")
        finally:
            channel.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control-fd", type=int, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    a = p.parse_args()
    with a.config.open("rb") as handle:
        raw = handle.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != a.config_sha256:
        raise ValueError("worker config identity mismatch")
    serve(socket.socket(fileno=a.control_fd), json.loads(raw))


if __name__ == "__main__":
    main()
