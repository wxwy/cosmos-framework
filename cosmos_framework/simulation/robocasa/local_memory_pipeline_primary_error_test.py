"""Exercise the actual native adapter's newly reachable double-error cleanup."""

from types import SimpleNamespace

import pytest

from cosmos_framework.simulation.robocasa import eval_v2_transactions
from cosmos_framework.simulation.robocasa.eval_v2_adapter import NativeEpisode


class ExceptionWithoutNotes(RuntimeError):
    add_note = None


@pytest.mark.parametrize(
    "primary",
    [
        ArithmeticError("Sim first error"),
        KeyboardInterrupt("interrupted Sim"),
        ExceptionWithoutNotes("pre-3.11 Sim error"),
    ],
)
def test_pipeline_cleanup_failure_does_not_replace_original_rollout_exception(tmp_path, monkeypatch, primary):
    cleanup = RuntimeError("completed-evidence background failure")
    closed = []

    class Transport:
        def __init__(self, *args):
            self.local = SimpleNamespace(streaming_enabled=True, pipeline_stats=lambda: {"enabled": True})

        def close(self):
            closed.append(True)
            raise cleanup

        def media_diagnostics(self):
            return {"errors": []}

    class Environment:
        def reset(self):
            return {}

        def _check_success(self):
            return False

        def get_ep_meta(self):
            return {"lang": "Actual CPU instruction"}

    def policy(*args, **kwargs):
        raise primary

    monkeypatch.setattr(eval_v2_transactions, "PolicyTransport", Transport)
    native = NativeEpisode(
        Environment(),
        {
            "execution_profile": "phase_b",
            "server_url": "http://unused",
            "horizon": 1,
            "predict_timeout_s": 1,
            "save_rollout_mp4": False,
            "save_generated_video": "none",
        },
        policy=policy,
        media_factory=lambda *args, **kwargs: SimpleNamespace(close=lambda: {"errors": []}),
    )
    assert native.initialize()["initial_success"] is False
    attempt = {
        "task": "CloseFridge",
        "episode": 0,
        "attempt_id": "attempt",
        "stream_id": "stream",
        "seed": 0,
        "manifest_digest": "a" * 64,
    }
    with pytest.raises(type(primary)) as observed:
        native.execute(attempt, reset_index=0, directory=tmp_path / "episode")
    assert observed.value is primary and closed == [True]
    if callable(getattr(primary, "add_note", None)):
        assert any("completed-evidence background failure" in note for note in primary.__notes__)
    else:
        assert "completed-evidence background failure" in primary.local_pipeline_cleanup_error
