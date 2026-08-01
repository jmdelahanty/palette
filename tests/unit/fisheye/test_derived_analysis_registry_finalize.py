from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.analysis_workflows import registry_finalize as mod
from fisheye.registry.db import Registry


def _selected_derived_runs(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    eye_parent = root.require_group("analysis/eye_angle_runs")
    eye_parent.attrs.update(
        {
            "palette_completion_epoch": 1,
            "latest": "eye_1",
            "latest_complete": "eye_1",
        }
    )
    eye = eye_parent.require_group("eye_1")
    eye.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    track_parent = root.require_group("analysis/track_kinematics_runs")
    track_parent.attrs.update(
        {
            "palette_completion_epoch": 1,
            "latest": "offline/track_1",
            "latest_complete": "offline/track_1",
        }
    )
    track = track_parent.require_group("offline/track_1")
    track.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )


def _execution_receipt(path: Path, zarr_path: Path) -> None:
    payload = {
        "schema_id": "palette.analysis_workflow_execution",
        "schema_version": 1,
        "mode": "apply",
        "status": "complete",
        "registry_write_mode": "deferred_to_serial_finalizer",
        "zarr_path": str(zarr_path),
        "execution_plan": {
            "commands": [
                {
                    "node_id": "track_kinematics",
                    "stage_id": "track_kinematics",
                    "output_run": "track_1",
                },
                {
                    "node_id": "eye_angles",
                    "stage_id": "eye_angles",
                    "output_run": "eye_1",
                },
            ]
        },
        "node_results": [
            {
                "node_id": "track_kinematics",
                "status": "complete",
                "run_name": "track_1",
            },
            {
                "node_id": "eye_angles",
                "status": "complete",
                "run_name": "eye_1",
            },
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_execution_receipt_finalizer_uses_exact_selected_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _selected_derived_runs(zarr_path)
    receipt = tmp_path / "execution.json"
    _execution_receipt(receipt, zarr_path)
    requests = mod._execution_report_requests(receipt)
    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()
    events: list[tuple[str, str]] = []

    def emit_eye(*_args, **kwargs):
        events.append(("eye_angles", kwargs["run_name"]))
        return True

    def emit_track(*_args, **kwargs):
        events.append(
            ("track_kinematics", f"{kwargs['run_type']}/{kwargs['run_name']}")
        )
        return True

    monkeypatch.setattr(mod, "emit_eye_angle_stage_completion", emit_eye)
    monkeypatch.setattr(mod, "emit_track_kinematics_stage_completion", emit_track)

    report = mod.finalize_registry(
        requests,
        registry_path=registry_path,
        apply=True,
    )

    assert report["status"] == "complete"
    assert report["registry_integrity"] == "ok"
    assert report["publication_count"] == 2
    assert events == [("eye_angles", "eye_1"), ("track_kinematics", "offline/track_1")]
    assert {row["receipt_sha256"] for row in report["publications"]} == {
        mod.hashlib.sha256(receipt.read_bytes()).hexdigest()
    }


def test_finalizer_rejects_receipt_for_nonselected_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _selected_derived_runs(zarr_path)
    receipt = tmp_path / "execution.json"
    _execution_receipt(receipt, zarr_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["execution_plan"]["commands"][1]["output_run"] = "eye_stale"
    payload["node_results"][1]["run_name"] = "eye_stale"
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    requests = mod._execution_report_requests(receipt)
    try:
        mod.finalize_registry(
            requests,
            registry_path=tmp_path / "unused.sqlite",
            apply=False,
        )
    except RuntimeError as exc:
        assert "selected canonical run" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("nonselected receipt unexpectedly validated")


def test_target_receipts_must_exactly_cover_manifest(tmp_path: Path) -> None:
    first = (tmp_path / "first.zarr").resolve()
    second = (tmp_path / "second.zarr").resolve()
    target_list = tmp_path / "targets.txt"
    target_list.write_text(f"{first}\n{second}\n", encoding="utf-8")
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    (status_dir / "first.json").write_text(
        json.dumps(
            {
                "schema_id": mod.TARGET_RECEIPT_SCHEMA,
                "schema_version": 1,
                "status": "complete",
                "registry_write_mode": "deferred_to_serial_finalizer",
                "zarr_path": str(first),
                "requested_publications": {
                    "eye_angles": {"requested": True}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        mod._target_receipt_requests(
            target_list,
            status_dir,
            ("eye_angles=latest",),
        )
    except RuntimeError as exc:
        assert str(second) in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("incomplete target receipt set unexpectedly validated")
