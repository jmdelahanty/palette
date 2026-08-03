from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.analysis_workflows import registry_finalize as mod
from fisheye.registry.db import Registry


GENERIC_STAGE_RUNS = {
    "swim_bouts": ("analysis/swim_bout_runs", "bouts_1"),
    "bout_kinematics": ("analysis/bout_kinematics_runs", "bout_kin_1"),
    "subject_shape": ("analysis/subject_shape_runs", "shape_1"),
    "tail_kinematics": ("analysis/tail_kinematics_runs", "tail_kin_1"),
    "tail_posture_view": ("analysis/tail_posture_view_runs", "tail_view_1"),
    "bout_classification": (
        "analysis/bout_classification_runs",
        "classify_1",
    ),
    "stimulus_response": ("analysis/stimulus_response_runs", "stimulus_1"),
}


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
    for parent_path, run_name in GENERIC_STAGE_RUNS.values():
        parent = root.require_group(parent_path)
        parent.attrs.update(
            {
                "palette_completion_epoch": 1,
                "latest": run_name,
                "latest_complete": run_name,
            }
        )
        run = parent.require_group(run_name)
        run.attrs.update(
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


def test_finalizer_dispatches_every_generic_derived_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _selected_derived_runs(zarr_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    requests = [
        mod.RequestedPublication(
            zarr_path=zarr_path,
            stage_id=stage_id,
            requested_run=run_name,
            receipt_path=receipt_path,
            receipt_sha256="a" * 64,
        )
        for stage_id, (_parent_path, run_name) in GENERIC_STAGE_RUNS.items()
    ]
    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()
    events: list[tuple[str, str]] = []

    def emit_generic(*_args, **kwargs):  # type: ignore[no-untyped-def]
        events.append((kwargs["stage_id"], kwargs["run_name"]))
        return True

    monkeypatch.setattr(
        mod,
        "emit_derived_analysis_stage_completion",
        emit_generic,
    )

    report = mod.finalize_registry(
        requests,
        registry_path=registry_path,
        apply=True,
    )

    assert report["publication_count"] == len(GENERIC_STAGE_RUNS)
    assert set(events) == {
        (stage_id, run_name)
        for stage_id, (_parent_path, run_name) in GENERIC_STAGE_RUNS.items()
    }


def test_generic_finalizer_rejects_selector_ineligible_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _selected_derived_runs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r+")
    root["analysis/swim_bout_runs/bouts_1"].attrs["stage_selector_eligible"] = False
    request = mod.RequestedPublication(
        zarr_path=zarr_path,
        stage_id="swim_bouts",
        requested_run="bouts_1",
        receipt_path=tmp_path / "receipt.json",
        receipt_sha256="b" * 64,
    )

    try:
        mod.finalize_registry(
            [request],
            registry_path=tmp_path / "unused.sqlite",
            apply=False,
        )
    except RuntimeError as exc:
        assert "selected canonical run" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("selector-ineligible run unexpectedly validated")


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
                "requested_publications": {"eye_angles": {"requested": True}},
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


def test_target_receipt_latest_binds_exact_eye_output_run(tmp_path: Path) -> None:
    target = (tmp_path / "analysis.zarr").resolve()
    target_list = tmp_path / "targets.txt"
    target_list.write_text(f"{target}\n", encoding="utf-8")
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    receipt_path = status_dir / "target.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_id": mod.TARGET_RECEIPT_SCHEMA,
                "schema_version": 1,
                "status": "complete",
                "registry_write_mode": "deferred_to_serial_finalizer",
                "zarr_path": str(target),
                "requested_publications": {
                    "eye_angles": {
                        "requested": True,
                        "output_run_name": "eye_exact_1",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    requests = mod._target_receipt_requests(
        target_list,
        status_dir,
        ("eye_angles=latest",),
    )

    assert len(requests) == 1
    assert requests[0].requested_run == "eye_exact_1"


def test_target_receipt_rejects_generic_selector_run_mismatch(tmp_path: Path) -> None:
    target = (tmp_path / "analysis.zarr").resolve()
    target_list = tmp_path / "targets.txt"
    target_list.write_text(f"{target}\n", encoding="utf-8")
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    (status_dir / "target.json").write_text(
        json.dumps(
            {
                "schema_id": mod.TARGET_RECEIPT_SCHEMA,
                "schema_version": 1,
                "status": "complete",
                "registry_write_mode": "deferred_to_serial_finalizer",
                "zarr_path": str(target),
                "requested_publications": {
                    "swim_bouts": {
                        "requested": True,
                        "run_name": "bouts_receipt",
                    }
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
            ("swim_bouts=bouts_other",),
        )
    except RuntimeError as exc:
        assert "selector differs from the target receipt" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("mismatched target run unexpectedly validated")
