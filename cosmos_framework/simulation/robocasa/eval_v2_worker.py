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
    from cosmos_framework.simulation.robocasa import render_devices
    from cosmos_framework.simulation.robocasa.eval_v2_adapter import NativeEpisode
    from cosmos_framework.simulation.robocasa.eval_v2_slot import SimSlot

    timings["imports"] = (time.perf_counter() - started) * 1000

    slot = None
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
                "media": Path(media.__file__),
                "generated_video": Path(generated_video.__file__),
            }
            if cfg["settings"].get("execution_profile") == "phase_b":
                from cosmos_framework.simulation.robocasa import eval_v2_transactions

                modules["transactions"] = Path(eval_v2_transactions.__file__)
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
                "runtime_digest": runtime["digest"] if factory is None else "CPU_TEST_FACTORY",
                "rendering": rendering,
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
                response.update(slot.close())
                send(channel, response)
                return
            else:
                raise ValueError("unknown actor operation")
            response["operation_ms"] = (time.perf_counter() - stage) * 1000
            send(channel, response)
    except EOFError:
        pass
    except BaseException as error:
        try:
            send(channel, {"kind": "FATAL", "nonce": cfg["nonce"], "error": f"{type(error).__name__}: {error}"})
        except OSError:
            pass
        raise
    finally:
        if slot is not None:
            slot.close()
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
