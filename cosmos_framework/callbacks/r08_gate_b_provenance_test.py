import json

from cosmos_framework.callbacks.r08_gate_b_provenance import R08GateBProvenanceCallback


def test_provenance_is_cwd_independent(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PSM_R08_HISTORY_MODE", "normal")
    monkeypatch.setenv("PSM_R08_GATE_B_CAPTURE_ONLY", "1")
    monkeypatch.setenv("PSM_R08_GATE_B_CHECKPOINT_PATH", "/checkpoint")
    monkeypatch.setattr(
        "cosmos_framework.callbacks.r08_gate_b_provenance.subprocess.check_output", lambda *args, **kwargs: "abc\n"
    )
    output = tmp_path / "provenance.json"
    R08GateBProvenanceCallback(str(output)).on_train_start(None, iteration=0)
    payload = json.loads(output.read_text())
    assert payload["root_revision"] == payload["submodule_revision"] == payload["gitlink_revision"] == "abc"
    assert payload["history_mode"] == "normal" and payload["capture_only"] is True
