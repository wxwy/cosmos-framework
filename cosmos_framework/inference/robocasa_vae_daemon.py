"""Registry-owned Wan-only daemon and actual loaded-encoder attestation.

The command-line surface contains FD numbers only. Credentials arrive through a
private inherited pipe; READY contains public identity and owned socket metadata.
Importing this module does not import Torch or load a model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import struct
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from cosmos_framework.evaluation.robocasa_v2.contract import ACTION_CHUNK_SIZE
from cosmos_framework.inference.robocasa_vae_protocol import canonical_json, strict_json


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _tensor_hash(values):
    import torch

    result = hashlib.sha256(b"PSM24-LOADED-TENSORS-v1\0")
    for name, tensor in sorted(values.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("loaded VAE requires concrete dense tensors")
        metadata = canonical_json({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        result.update(struct.pack(">I", len(metadata)))
        result.update(metadata)
        result.update(struct.pack(">Q", len(raw)))
        result.update(raw)
    return result.hexdigest()


def _plain_config_container(value):
    # Resolve in the original OmegaConf tree, before copying away its parent.
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    if isinstance(value, (DictConfig, ListConfig)):
        try:
            return OmegaConf.to_container(value, resolve=True, throw_on_missing=True, enum_to_str=False)
        except OmegaConfBaseException as error:
            raise ValueError("unresolved loaded Wan configuration") from error
    return value


def worker_constructor_config(tokenizer, *, vae_path=None):
    """Effective nonsecret Wan constructor settings, with a local weights path.

    Location/credentials are not encoder semantics. Actual loaded tensors and
    scale are independently hashed; the daemon additionally hashes its input file.
    """
    frames = _plain_config_container(tokenizer.encode_chunk_frames)
    durations = _plain_config_container(tokenizer.encode_exact_durations)
    if (
        not isinstance(frames, Mapping)
        or not frames
        or any(
            type(key) is not str or not key or type(value) is not int or value <= 0 or value % 4
            for key, value in frames.items()
        )
    ):
        raise ValueError("loaded Wan chunk frames require string keys and positive integer multiples of four")
    if durations is not None and (
        type(durations) not in (list, tuple) or any(type(value) is not int or value <= 0 for value in durations)
    ):
        raise ValueError("loaded Wan exact durations require a list of positive integers or None")
    config = {
        "chunk_duration": tokenizer.chunk_duration,
        "keep_decoder_cache": False,
        "use_streaming_encode": False,
        "encode_chunk_frames": dict(frames),
        "encode_exact_durations": None if durations is None else list(durations),
        "spatial_compression_factor": tokenizer.spatial_compression_factor,
        "temporal_compression_factor": tokenizer.temporal_compression_factor,
        "causal": tokenizer.is_causal,
    }
    if (
        any(
            type(config[name]) is not int or config[name] <= 0
            for name in ("chunk_duration", "spatial_compression_factor", "temporal_compression_factor")
        )
        or type(config["causal"]) is not bool
    ):
        raise ValueError("loaded Wan constructor requires integer dimensions and a boolean causal flag")
    if vae_path is not None:
        config.update(bucket_name="", object_store_credential_path_pretrained="", vae_path=str(vae_path))
    # Freeze mutable lists/mappings rather than return tokenizer-owned objects.
    return strict_json(canonical_json(config), max_bytes=65536)


def loaded_vae_identity(tokenizer, tokenizer_config=None):
    """Hash actual loaded parameters, scale and effective encoding configuration.

    Both Policy and Worker call this independently. No supplied file hash or
    model name is treated as evidence that those weights were actually loaded.
    C currently requires eager BF16; installed AOT artifacts fail closed.
    """
    import torch

    from cosmos_framework.inference.robocasa_vae_encoder import preprocess_profile_digest, visual96_contract_digest

    loaded = getattr(tokenizer, "model", None)
    module = getattr(loaded, "model", None)
    if (
        module is None
        or module.training
        or loaded.dtype != torch.bfloat16
        or tokenizer.is_causal is not True
        or tokenizer.use_streaming_encode is not False
        or tokenizer._keep_encoder_cache is not False
        or getattr(module, "_aot_chunk_fns", None)
        or tokenizer.spatial_compression_factor != 16
        or tokenizer.temporal_compression_factor != 4
    ):
        raise ValueError("Stage3A requires actual frozen causal eager BF16 Wan")
    parameters = tuple(module.parameters())
    if not parameters or any(parameter.requires_grad or parameter.dtype != torch.bfloat16 for parameter in parameters):
        raise ValueError("actual Wan parameters are not frozen BF16")
    effective = worker_constructor_config(tokenizer)
    # The optional resolved config is a consistency input, never the identity
    # authority. A caller cannot supply different bytes to replace loaded state.
    if tokenizer_config is not None:
        supplied = dict(tokenizer_config)
        for name in ("bucket_name", "vae_path", "object_store_credential_path_pretrained", "_target_"):
            supplied.pop(name, None)
        for name, value in supplied.items():
            if name in effective and _plain_config_container(value) != effective[name]:
                raise ValueError("resolved tokenizer config differs from actual loaded configuration")
    scale = getattr(loaded, "scale", None)
    if not isinstance(scale, (tuple, list)) or len(scale) != 2:
        raise ValueError("actual loaded Wan normalization scale is missing")
    window = _plain_config_container(getattr(module, "temporal_window", effective["encode_chunk_frames"]))
    if isinstance(window, Mapping):
        window = dict(window)
    configuration = {
        "version": "PSM24-LOADED-WAN-v1",
        "class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
        "effective": effective,
        "actual_temporal_window": window,
        "actual_exact_durations": sorted(
            getattr(module, "_encode_exact_durations", effective["encode_exact_durations"] or [])
        ),
        "scale_sha256": _tensor_hash({str(index): tensor for index, tensor in enumerate(scale)}),
        "dtype": "bfloat16",
        "backend": "eager",
    }
    return {
        "vae_weights_sha256": _tensor_hash(module.state_dict()),
        "vae_config_sha256": hashlib.sha256(canonical_json(configuration)).hexdigest(),
        "preprocess_profile_sha256": preprocess_profile_digest(256),
        "visual96_contract_sha256": visual96_contract_digest(),
        "encoder_backend": "eager",
        "encoder_dtype": "bfloat16",
    }


def load_real_encoder(config):
    """Load only a local Wan VAE, never the Policy model, dataset or checkpoint."""
    from cosmos_framework.inference.robocasa_vae_encoder import LoadedWanEncode1
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    path = Path(config["vae_weights"])
    expected = config["vae_file_sha256"]
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError("Worker VAE input file differs from frozen identity")
    kwargs = dict(config["constructor_config"])
    if any(name in kwargs for name in ("vae_path", "bucket_name", "object_store_credential_path_pretrained")):
        raise ValueError("Worker constructor location must come from the frozen local file")
    tokenizer = Wan2pt2VAEInterface(
        bucket_name="", object_store_credential_path_pretrained="", vae_path=str(path), **kwargs
    )
    actual = loaded_vae_identity(tokenizer)
    if file_sha256(path) != expected or actual != config["identity"]:
        raise ValueError("actual Worker VAE does not match Policy loaded weights/config")
    return LoadedWanEncode1(tokenizer, identity=actual), {
        "identity": actual,
        "vae_file_sha256": expected,
        "constructor_config_sha256": hashlib.sha256(canonical_json(kwargs)).hexdigest(),
        "backend": "actual_loaded_wan",
        "device_type": "cuda",
    }


def _read_private(fd):
    raw = bytearray()
    try:
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > 131072:
                raise ValueError("bounded private Worker bootstrap required")
    finally:
        os.close(fd)
    return strict_json(bytes(raw), max_bytes=131072)


def _event(fd, value):
    raw = canonical_json(value) + b"\n"
    if len(raw) > 65536:
        raise ValueError("bounded Worker READY proof required")
    while raw:
        count = os.write(fd, raw)
        raw = raw[count:]


def _clock_domain():
    from cosmos_framework.inference.robocasa_session_transactions import monotonic_clock_domain

    return monotonic_clock_domain()


def _physical_device(expected, encoder):
    import torch

    from cosmos_framework.simulation.robocasa.render_devices import normalize_pci

    parameters = tuple(encoder.tokenizer.model.model.parameters())
    device = parameters[0].device if parameters else None
    if (
        device is None
        or device.type != "cuda"
        or any(parameter.device != device for parameter in parameters)
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Worker requires exactly one visible CUDA device")
    uid = str(torch.cuda.get_device_properties(device).uuid)
    if re.fullmatch(r"(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uid) is None:
        raise ValueError("actual Worker CUDA UUID has invalid full 128-bit format")
    if not uid.startswith("GPU-"):
        uid = "GPU-" + uid
    if uid != expected["uuid"]:
        raise ValueError("actual Worker CUDA UUID differs from assigned physical GPU")
    output = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                uid,
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        .strip()
        .splitlines()
    )
    if len(output) != 1:
        raise ValueError("actual Worker CUDA physical identity unavailable")
    index, actual_uid, pci = [part.strip() for part in output[0].split(",")]
    actual = {"gpu_id": int(index), "uuid": actual_uid, "pci_bus_id": normalize_pci(pci)}
    if actual != expected:
        raise ValueError("actual Worker CUDA PCI/UUID differs from assigned device")
    return actual


def run_daemon(
    bootstrap_fd,
    ready_fd,
    *,
    encoder_factory=load_real_encoder,
    clock_domain_fn=_clock_domain,
    device_attestor=_physical_device,
):
    """Injecting a fake encoder is a Python-only CPU-test seam, not a CLI mode."""
    from cosmos_framework.inference.robocasa_vae_client import UnixVAEServer
    from cosmos_framework.inference.robocasa_vae_worker import SharedVAEWorker, WorkerConfig

    worker = server = None
    primary = None
    try:
        private = _read_private(bootstrap_fd)
        config = private["public"]
        if config["bmax"] != 1 or config["wait_ms"] != 0:
            raise ValueError("Stage3A Action Path supports Worker B1/wait0 only")
        if clock_domain_fn() != config["clock_domain"]:
            raise ValueError("Worker actual monotonic clock domain mismatch")
        encoder, attestation = encoder_factory(config)
        if attestation["identity"] != config["identity"]:
            raise ValueError("loaded Worker attestation mismatch")
        physical = device_attestor(config["physical_device"], encoder)
        worker = SharedVAEWorker(
            encoder,
            attestation["identity"],
            bytes.fromhex(private["signing_key_hex"]),
            config["key_id"],
            config["worker_id"],
            worker_epoch=config["worker_epoch"],
            clock_domain=config["clock_domain"],
            config=WorkerConfig(
                bmax=1,
                wait_ms=0,
                max_sessions=config.get("max_sessions", 16),
                max_policy_epochs=config.get("max_policy_epochs", 1),
                history_k=config.get("local_ttt_history_k"),
                action_horizon=config.get("action_chunk_size", ACTION_CHUNK_SIZE),
            ),
        )
        server = UnixVAEServer(
            worker,
            config["socket_path"],
            bytes.fromhex(private["control_capability_hex"]),
            connection_timeout_s=10,
        ).start()
        socket_proof = {}
        for name in (config["socket_path"], config["socket_path"] + ".ctl"):
            value = Path(name).lstat()
            if not stat.S_ISSOCK(value.st_mode):
                raise ValueError("Worker READY requires actual owned Unix sockets")
            socket_proof[name] = {"device": value.st_dev, "inode": value.st_ino, "mode": stat.S_IMODE(value.st_mode)}
        _event(
            ready_fd,
            {
                "kind": "READY",
                "pid": os.getpid(),
                "worker_id": config["worker_id"],
                "worker_epoch": config["worker_epoch"],
                "key_id": config["key_id"],
                "clock_domain": config["clock_domain"],
                "bmax": 1,
                "wait_ms": 0,
                "attestation": attestation,
                "sockets": socket_proof,
                "physical_device": physical,
                "limits": {
                    "max_sessions": worker.config.max_sessions,
                    "max_policy_epochs": worker.config.max_policy_epochs,
                },
                **(
                    {"local_ttt_history_k": worker.config.history_k, "action_chunk_size": worker.config.action_horizon}
                    if worker.config.history_k != worker.config.action_horizon
                    else {}
                ),
            },
        )
        server.wait_closed()
        receipt = server.close(time.monotonic_ns() + int(config["drain_timeout_s"] * 1e9))
        _event(ready_fd, {"kind": "CLOSED", "receipt": receipt})
        if not receipt.get("quiescent") or not receipt.get("ipc_quiescent"):
            raise RuntimeError("Worker close has no positive drain proof")
    except BaseException as error:
        primary = error
        try:
            _event(ready_fd, {"kind": "ERROR", "error_type": type(error).__name__})
        except OSError:
            pass
        raise
    finally:
        try:
            if server is not None:
                server.close(time.monotonic_ns() + 1_000_000_000)
            elif worker is not None:
                worker.close(time.monotonic_ns() + 1_000_000_000)
        except BaseException as cleanup:
            if primary is None:
                raise
            if callable(getattr(primary, "add_note", None)):
                primary.add_note("Worker cleanup also failed: " + type(cleanup).__name__)
        finally:
            os.close(ready_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-fd", type=int, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    args = parser.parse_args()
    run_daemon(args.bootstrap_fd, args.ready_fd)


if __name__ == "__main__":
    main()
