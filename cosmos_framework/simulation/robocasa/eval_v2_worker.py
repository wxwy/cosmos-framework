"""Persistent one-Task native environment. Started only by the V2 coordinator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

from cosmos_framework.evaluation.robocasa_v2.wire import receive, send


def serve(channel, cfg, *, factory=None):
    # Optional factory is for CPU tests; production CLI cannot select/import one.
    from cosmos_framework.simulation.robocasa import closed_loop_eval as native
    from cosmos_framework.simulation.robocasa.eval_v2_adapter import NativeEpisode

    env = None
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
            }
            expected = cfg["source_files"]
            if set(expected) != set(modules):
                raise ValueError("incomplete native source witnesses")
            for name, file in modules.items():
                if (
                    str(file.resolve()) != expected[name]["path"]
                    or hashlib.sha256(file.read_bytes()).hexdigest() != expected[name]["sha256"]
                ):
                    raise ValueError("native import origin/source mismatch")
            runtime = native.verify_runtime(SimpleNamespace(**cfg["runtime_args"]), cfg["task"])
            if runtime is None or runtime["protocol"]["official_horizons"][cfg["task"]] != cfg["settings"]["horizon"]:
                raise ValueError("missing/incorrect authoritative runtime horizon")
            if (
                hashlib.sha256((Path(cfg["dataset_dir"]) / "extras/dataset_meta.json").read_bytes()).hexdigest()
                != cfg["metadata_sha256"]
            ):
                raise ValueError("Task metadata changed before environment construction")
            env = native.make_env(
                cfg["dataset_dir"],
                256,
                obj_split="target",
                layout_style=native.ATOMIC_SEEN_LAYOUT_STYLE,
                seed=cfg["seed"],
            )
        else:
            env = factory()
        adapter = NativeEpisode(env, cfg["settings"])
        send(
            channel,
            {
                "kind": "READY",
                "nonce": cfg["nonce"],
                "pid": os.getpid(),
                "task": cfg["task"],
                "stream_id": cfg["stream_id"],
                "runtime_digest": runtime["digest"] if factory is None else "CPU_TEST_FACTORY",
            },
        )
        while True:
            message = receive(channel)
            if message.get("nonce") != cfg["nonce"]:
                raise ValueError("actor request identity mismatch")
            op = message.get("op")
            response = {"request_id": message["request_id"], "nonce": cfg["nonce"]}
            if op == "initialize":
                response.update(adapter.initialize())
            elif op == "execute":
                attempt = message["attempt"]
                if (
                    attempt["task"] != cfg["task"]
                    or attempt["stream_id"] != cfg["stream_id"]
                    or attempt["seed"] != cfg["seed"]
                ):
                    raise ValueError("foreign Episode attempt")
                directory = Path(message["directory"]).resolve()
                expected = (
                    Path(cfg["work_root"]).resolve() / f"ep_{attempt['episode']:03d}" / "work" / attempt["attempt_id"]
                )
                if directory != expected:
                    raise ValueError("output outside owned work area")
                response.update(adapter.execute(attempt, reset_index=message["reset_index"], directory=directory))
            elif op == "close":
                env.close()
                env = None
                send(channel, dict(response, closed=True))
                return
            else:
                raise ValueError("unknown actor operation")
            send(channel, response)
    except EOFError:
        pass  # Coordinator died; no new work. Native env is closed below.
    except BaseException as error:
        try:
            send(channel, {"kind": "FATAL", "nonce": cfg["nonce"], "error": f"{type(error).__name__}: {error}"})
        except OSError:
            pass
        raise
    finally:
        if env is not None:
            env.close()
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
