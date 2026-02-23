from __future__ import annotations

from pathlib import Path

from fisheye.utils import model_resolution_provenance as mod


def test_build_model_resolution_payload_includes_canonical_sections(monkeypatch) -> None:
    def _fake_invocation_record(*, tool, args, argv):  # noqa: ANN001
        return {
            "tool": tool,
            "command": "scripts/py -m fake.tool --flag",
            "args": {"flag": True, "top_k": 5},
            "git": {
                "commit_hash": "abcdef0123456789",
                "short_hash": "abcdef01",
                "branch": "main",
                "is_dirty": False,
                "remote_url": "git@github.com:example/repo.git",
            },
            "environment": {"environment_type": "conda", "environment_name": "palette-py311"},
            "platform": {"hostname": "ws1", "system": "Linux"},
        }

    monkeypatch.setattr(mod, "build_invocation_record", _fake_invocation_record)

    payload = mod.build_model_resolution_payload(
        tool="fisheye.utils.run_detect_with_registry_model",
        args={"flag": True},
        argv=["--recording-dir", "/tmp/rec"],
        task="detect",
        registry_path=Path("/tmp/palette_registry.sqlite"),
        recording_id="2026-01-28T19-22-28Z_arena_1",
        target={"recording_id": "2026-01-28T19-22-28Z_arena_1"},
        selected={"run_id": "run_001", "model_path": "/tmp/model.pt"},
        candidates=[{"run_id": "run_001", "model_path": "/tmp/model.pt"}],
        parameters={"require_unique": True},
        inputs={"recording_dir": "/tmp/rec"},
        artifacts={"selected_model": {"run_id": "run_001"}},
    )

    assert payload["contract"]["name"] == mod.MODEL_RESOLUTION_CONTRACT_NAME
    assert payload["contract"]["version"] == mod.MODEL_RESOLUTION_CONTRACT_VERSION
    assert payload["task"] == "detect"
    assert payload["mode"] == "registry"
    assert payload["registry_path"] == "/tmp/palette_registry.sqlite"
    assert payload["recording_id"] == "2026-01-28T19-22-28Z_arena_1"
    assert payload["command"] == "scripts/py -m fake.tool --flag"
    assert payload["git"]["commit"] == "abcdef0123456789"
    assert payload["git"]["short"] == "abcdef01"
    assert payload["git"]["branch"] == "main"
    assert payload["git"]["remote"] == "git@github.com:example/repo.git"
    assert payload["environment"]["environment_name"] == "palette-py311"
    assert payload["platform"]["hostname"] == "ws1"
    assert payload["parameters"]["flag"] is True
    assert payload["parameters"]["require_unique"] is True
    assert payload["inputs"]["recording_dir"] == "/tmp/rec"
    assert payload["artifacts"]["selected_model"]["run_id"] == "run_001"
