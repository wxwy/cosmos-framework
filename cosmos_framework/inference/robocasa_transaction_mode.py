"""Bind explicit EvalV2 mode to loaded configuration and actual DCP inventory."""

from pathlib import Path


def validate_model_mode(model, config, checkpoint, mode, *, use_ema_weights=False):
    from torch.distributed.checkpoint import FileSystemReader

    from cosmos_framework.inference.robocasa_session_transactions import protocol_for_mode

    protocol_for_mode(mode)
    if type(use_ema_weights) is not bool:
        raise ValueError("explicit selected checkpoint weight tree required")
    expected = mode == "required"
    configured = config.get("model", {}).get("config", {}).get("local_memory_enabled")
    net = model.net
    if configured is not expected or getattr(net.config, "local_memory_enabled", None) is not expected:
        raise ValueError("explicit transaction mode differs from checkpoint training/model config")
    runtime = getattr(net, "local_memory_runtime", None)
    if (runtime is not None) != expected:
        raise ValueError("transaction mode differs from loaded Local-TTT capability")
    directory = Path(checkpoint)
    if (directory / "model").is_dir():
        directory = directory / "model"
    metadata = FileSystemReader(directory).read_metadata().state_dict_metadata
    local = {key: value for key, value in metadata.items() if "local_memory_runtime." in key}
    if bool(local) != expected:
        raise ValueError("transaction mode differs from actual checkpoint Local-TTT inventory")
    if expected:
        prefix = ("net_ema." if use_ema_weights else "net.") + "local_memory_runtime."
        for name, parameter in runtime.named_parameters():
            value = local.get(prefix + name)
            if value is None or tuple(value.size) != tuple(parameter.shape):
                raise ValueError("checkpoint Local-TTT inventory is incomplete or has incompatible shapes")
