from __future__ import annotations

import json

from fisheye.utils import sync_refined_subject_mask_metadata as cli_mod


def test_main_dispatches_check_source_updates(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def _fake_check_source_updates(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"status": "updated", "auto_synced_roi_count": 1}

    monkeypatch.setattr(cli_mod, "check_refined_subject_source_updates", _fake_check_source_updates)

    rc = cli_mod.main(
        [
            "--zarr-path",
            "/tmp/example.zarr",
            "--refined-run",
            "refined_subject_masks_001",
            "--component-name",
            "swim_bladder",
            "--roi-indices",
            "1,3",
            "--check-source-updates",
            "--assume-source-changed-untracked",
            "--force-source-changed",
        ]
    )

    assert rc == 0
    assert captured["refined_run"] == "refined_subject_masks_001"
    assert captured["component_name"] == "swim_bladder"
    assert captured["roi_indices"] == [1, 3]
    assert captured["assume_source_changed_untracked"] is True
    assert captured["force_source_changed"] is True
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "updated"
    assert payload["auto_synced_roi_count"] == 1
