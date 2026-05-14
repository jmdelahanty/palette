from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fisheye.utils import review_detect_batch as mod


def _write_zarr_json(path: Path, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"attributes": attrs}), encoding="utf-8")


def _make_review_zarr(
    root: Path,
    *,
    refined_run: str = "refined_detect_1",
    state: str = "pending",
    intended_use: str = "training",
) -> Path:
    zarr_path = root / "sample_training.zarr"
    _write_zarr_json(zarr_path / "zarr.json", {"zarr_purpose": "training"})
    _write_zarr_json(zarr_path / "refined_detect_runs" / "zarr.json", {"latest": refined_run})
    _write_zarr_json(
        zarr_path / "refined_detect_runs" / refined_run / "zarr.json",
        {
            "detect_review_status": {
                "state": state,
                "intended_use": intended_use,
                "resolved_group": "refined",
            }
        },
    )
    return zarr_path


def test_read_detect_review_status_from_metadata_files(tmp_path: Path) -> None:
    zarr_path = _make_review_zarr(tmp_path, state="needs_review")

    status = mod._read_detect_review_status(zarr_path, refined_run=None)

    assert status == {
        "refined_run": "refined_detect_1",
        "review_state": "needs_review",
        "review_intended_use": "training",
        "review_resolved_group": "refined",
    }


def test_dry_run_writes_resumable_state(tmp_path: Path) -> None:
    zarr_path = _make_review_zarr(tmp_path)
    queue_file = tmp_path / "queue.txt"
    queue_file.write_text(str(zarr_path) + "\n", encoding="utf-8")
    state_file = tmp_path / "state.json"

    rc = mod.main(
        [
            "--queue-file",
            str(queue_file),
            "--state-file",
            str(state_file),
            "--dry-run",
            "--all",
            "--reviewer",
            "tester",
        ]
    )

    assert rc == 0
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    entry = payload["entries"][str(zarr_path)]
    assert entry["last_outcome"] == "dry_run"
    assert "--all" in entry["command"]
    assert "--reviewer" in entry["command"]
    assert payload["summary"]["queue_count"] == 1


def test_resume_skips_previously_reviewed_states(tmp_path: Path) -> None:
    first = _make_review_zarr(tmp_path / "first", state="approved")
    second = _make_review_zarr(tmp_path / "second", state="pending")
    queue_file = tmp_path / "queue.txt"
    queue_file.write_text(f"{first}\n{second}\n", encoding="utf-8")
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_name": mod.SCHEMA_NAME,
                "schema_version": mod.SCHEMA_VERSION,
                "entries": {
                    str(first): {
                        "zarr_path": str(first),
                        "last_review_state": "approved",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--queue-file",
            str(queue_file),
            "--state-file",
            str(state_file),
            "--resume",
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["summary"]["queue_count"] == 1
    assert str(second) in payload["entries"]


def test_subprocess_run_updates_final_review_state(tmp_path: Path, monkeypatch) -> None:
    zarr_path = _make_review_zarr(tmp_path, state="pending")
    queue_file = tmp_path / "queue.txt"
    queue_file.write_text(str(zarr_path) + "\n", encoding="utf-8")
    state_file = tmp_path / "state.json"

    def fake_run(cmd, check=False):
        assert "-m" in cmd
        assert "fisheye.tune.detect_review" in cmd
        _write_zarr_json(
            zarr_path / "refined_detect_runs" / "refined_detect_1" / "zarr.json",
            {
                "detect_review_status": {
                    "state": "approved",
                    "intended_use": "training",
                    "resolved_group": "refined",
                }
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rc = mod.main(["--queue-file", str(queue_file), "--state-file", str(state_file)])

    assert rc == 0
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    entry = payload["entries"][str(zarr_path)]
    assert entry["returncode"] == 0
    assert entry["last_review_state"] == "approved"
    assert entry["last_outcome"] == "approved"
