from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.native_canonical_detection_publication import (
    publish_native_canonical_detection_candidate,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_schema import SOURCE_KIND_CODE_MAP
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    require_runs_parent,
)
from fisheye.utils.finalize_recording_refined_detection_v1 import (
    _validate_gate_binding,
    _validate_gate_policy_binding,
    finalize_recording_refined_detection_v1,
)
from tests.unit.fisheye.test_native_canonical_detection_publication import (
    RECORDING_IDENTITY,
    RUN_ID,
    _archive,
    _candidate,
)


class _Run:
    def __init__(self, attrs):
        self.attrs = attrs


def _run(requirement: str, *, status: str, applied: bool, gate_run: str | None):
    return _Run(
        {
            "registered_detection_gate_requirement": requirement,
            "registered_detection_gate": {
                "requirement": requirement,
                "status": status,
                "applied": applied,
                "gate_run": gate_run,
                "selection_run": "selection_001",
                "selection_digest": "a" * 64,
                "selection_record_schema_version": 2,
                "comparison_policy_id": "manual_review_only_v1",
            },
        }
    )


def test_required_finalization_preserves_exact_applied_gate_binding() -> None:
    evidence = _validate_gate_binding(
        _run("required", status="applied", applied=True, gate_run="gate_001"),
        requirement="required",
        expected_gate_run="gate_001",
    )
    assert evidence["gate_run"] == "gate_001"
    assert evidence["selection_digest"] == "a" * 64


def test_finalization_accepts_comparison_bound_v2_policy() -> None:
    evidence = _validate_gate_binding(
        _run("required", status="applied", applied=True, gate_run="gate_001"),
        requirement="required",
        expected_gate_run="gate_001",
    )

    _validate_gate_policy_binding(
        evidence,
        configured_policy_id="manual_review_only_v1",
    )


def test_finalization_maps_reviewed_palette_v3_to_manual_review_policy() -> None:
    evidence = _validate_gate_binding(
        _run("required", status="applied", applied=True, gate_run="gate_001"),
        requirement="required",
        expected_gate_run="gate_001",
    )
    evidence.update(
        {
            "selection_record_schema_version": 3,
            "selection_policy": "manual_reviewed_palette_candidate_exact_binding_v3",
            "selection_decision_source": "manual_review",
            "comparison_run": None,
            "comparison_policy_id": None,
        }
    )

    _validate_gate_policy_binding(
        evidence,
        configured_policy_id="manual_review_only_v1",
    )


def test_finalization_rejects_reviewed_palette_v3_as_corroborated_policy() -> None:
    evidence = _validate_gate_binding(
        _run("required", status="applied", applied=True, gate_run="gate_001"),
        requirement="required",
        expected_gate_run="gate_001",
    )
    evidence.update(
        {
            "selection_record_schema_version": 3,
            "selection_policy": "manual_reviewed_palette_candidate_exact_binding_v3",
            "selection_decision_source": "manual_review",
            "comparison_run": None,
            "comparison_policy_id": None,
        }
    )

    with pytest.raises(ValueError, match="differs from the policy bound"):
        _validate_gate_policy_binding(
            evidence,
            configured_policy_id="corroborated_acquisition_v1",
        )


def test_finalization_rejects_unrecognized_comparison_free_gate_policy() -> None:
    evidence = _validate_gate_binding(
        _run("required", status="applied", applied=True, gate_run="gate_001"),
        requirement="required",
        expected_gate_run="gate_001",
    )
    evidence.update(
        {
            "selection_record_schema_version": 3,
            "selection_policy": "unreviewed_or_unknown",
            "selection_decision_source": "manual_review",
            "comparison_run": None,
            "comparison_policy_id": None,
        }
    )

    with pytest.raises(ValueError, match="lacks a supported selection-policy"):
        _validate_gate_policy_binding(
            evidence,
            configured_policy_id="manual_review_only_v1",
        )


@pytest.mark.parametrize(
    ("run", "requirement", "expected", "message"),
    (
        (
            _run("required", status="unavailable", applied=False, gate_run="gate_001"),
            "required",
            "gate_001",
            "needs an applied exact gate",
        ),
        (
            _run("required", status="applied", applied=True, gate_run="gate_other"),
            "required",
            "gate_001",
            "expected 'gate_001'",
        ),
        (
            _run("off", status="applied", applied=True, gate_run="gate_001"),
            "off",
            None,
            "inconsistent gate evidence",
        ),
    ),
)
def test_finalization_rejects_inconsistent_gate_binding(
    run, requirement: str, expected: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_gate_binding(
            run,
            requirement=requirement,
            expected_gate_run=expected,
        )


def test_if_available_finalization_preserves_explicit_unavailable_state() -> None:
    evidence = _validate_gate_binding(
        _run("if_available", status="unavailable", applied=False, gate_run=None),
        requirement="if_available",
        expected_gate_run=None,
    )
    assert evidence["status"] == "unavailable"
    assert evidence["applied"] is False


def _reviewed_palette_v3_gate_evidence() -> dict[str, object]:
    return {
        "requirement": "required",
        "status": "applied",
        "applied": True,
        "gate_run": "gate_001",
        "selection_run": "selection_001",
        "selection_digest": "b" * 64,
        "selection_record_schema_version": 3,
        "selection_policy": "manual_reviewed_palette_candidate_exact_binding_v3",
        "selection_decision_source": "manual_review",
        "comparison_run": None,
        "comparison_policy_id": None,
    }


def _write_working_refined(
    archive: Path,
    *,
    gate_evidence: dict[str, object] | None = None,
) -> None:
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    source = root["detect_runs"][RUN_ID]["instances"]
    parent = require_runs_parent(
        root,
        "refined_detect_runs",
        completion_epoch=COMPLETION_EPOCH_STRICT,
    )
    run = parent.create_group("refined_working")
    run.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "source_detect_run": RUN_ID,
            "registered_detection_gate_requirement": "required",
            "registered_detection_gate": gate_evidence
            or {
                "requirement": "required",
                "status": "applied",
                "applied": True,
                "gate_run": "gate_001",
                "selection_run": "selection_001",
                "selection_digest": "a" * 64,
                "selection_record_schema_version": 2,
                "comparison_policy_id": "manual_review_only_v1",
            },
        }
    )
    frame_indices = np.asarray(source["frame_indices"][:], dtype=np.int32)
    bbox_norm = np.asarray(source["bbox_norm_coords"][:], dtype=np.float64)
    scores = np.asarray(source["scores"][:], dtype=np.float32)
    class_ids = np.asarray(source["class_ids"][:], dtype=np.int32)
    instance_key = np.asarray(source["instance_key"][:], dtype=np.uint64)
    rows = np.arange(frame_indices.size, dtype=np.int32)
    instances = run.create_group("instances")
    instance_arrays = {
        "frame_indices": frame_indices,
        "refined_row_ids": rows.astype(np.int64),
        "bbox_norm_coords": bbox_norm,
        "source_kind_codes": np.full(
            rows.size,
            SOURCE_KIND_CODE_MAP["raw_detect"],
            dtype=np.int8,
        ),
        "manual_edit_flags": np.zeros(rows.size, dtype=np.bool_),
        "source_detect_row_index": rows,
        "confidence_scores": scores,
        "class_ids": class_ids,
        "instance_key": instance_key,
        "frame_counts": np.asarray([1, 0, 1], dtype=np.int32),
        "frame_offsets": np.asarray([0, 1, 1, 2], dtype=np.int64),
    }
    for name, values in instance_arrays.items():
        instances.create_array(name, data=values)
    source_rows = run.create_group("source_detections")
    source_arrays = {
        "source_detect_row_index": rows,
        "frame_indices": frame_indices,
        "bbox_norm_coords": bbox_norm,
        "decision_codes": np.zeros(rows.size, dtype=np.int8),
        "resolved_refined_row_id": rows.astype(np.int64),
        "confidence_scores": scores,
        "class_ids": class_ids,
        "instance_key": instance_key,
    }
    for name, values in source_arrays.items():
        source_rows.create_array(name, data=values)


def test_finalizer_publishes_crop_bindable_immutable_authority(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    frame, pixel = _archive(archive)
    candidate = _candidate(tmp_path, frame, pixel)
    publish_native_canonical_detection_candidate(
        analysis_zarr=archive,
        candidate_zarr=candidate.output_path,
        run_id=RUN_ID,
        recording_identity=RECORDING_IDENTITY,
    )
    _write_working_refined(archive)
    scratch = tmp_path / "scratch" / "job"
    scratch.mkdir(parents=True)

    result = finalize_recording_refined_detection_v1(
        analysis_zarr=archive,
        canonical_detect_run=RUN_ID,
        working_refined_run="refined_working",
        output_run="refined_final",
        recording_identity=RECORDING_IDENTITY,
        registered_gate_requirement="required",
        registered_gate_run="gate_001",
        selection_policy_id="manual_review_only_v1",
        scratch_root=scratch,
    )

    assert result["status"] == "complete"
    assert result["raw_detection_unchanged"] is True
    bound = bind_refined_detection_crop_source(
        archive,
        run_id="refined_final",
        allow_selector_ineligible_benchmark=True,
    )
    assert bound.run_group.attrs["finalized_recording_authority"] is True
    assert bound.run_group.attrs["registered_detection_gate"]["gate_run"] == (
        "gate_001"
    )
    assert bound.manifest["payload_digest"] == result["run_manifest_digest"]


def test_finalizer_publishes_from_reviewed_palette_v3_gate(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    frame, pixel = _archive(archive)
    candidate = _candidate(tmp_path, frame, pixel)
    publish_native_canonical_detection_candidate(
        analysis_zarr=archive,
        candidate_zarr=candidate.output_path,
        run_id=RUN_ID,
        recording_identity=RECORDING_IDENTITY,
    )
    _write_working_refined(
        archive,
        gate_evidence=_reviewed_palette_v3_gate_evidence(),
    )
    scratch = tmp_path / "scratch" / "job"
    scratch.mkdir(parents=True)

    result = finalize_recording_refined_detection_v1(
        analysis_zarr=archive,
        canonical_detect_run=RUN_ID,
        working_refined_run="refined_working",
        output_run="refined_final",
        recording_identity=RECORDING_IDENTITY,
        registered_gate_requirement="required",
        registered_gate_run="gate_001",
        selection_policy_id="manual_review_only_v1",
        scratch_root=scratch,
    )

    assert result["status"] == "complete"
    assert result["registered_detection_gate"]["selection_record_schema_version"] == 3
    assert result["selection_policy_id"] == "manual_review_only_v1"
