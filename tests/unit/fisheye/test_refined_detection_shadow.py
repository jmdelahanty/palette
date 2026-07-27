from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.instance_keys import mint_manual_curation_instance_keys
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceIdentity,
    validate_refined_detection_run_manifest,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
)
from fisheye.shared.zarr.refined_detection_shadow import (
    publish_refined_detection_shadow,
    require_safe_refined_detection_shadow_destination,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_refined_detection_transition,
)


RECORDING_IDENTITY = "shadow_multi_subject_recording"


def _transition():
    source_bbox = np.asarray(
        [[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]],
        dtype=np.float64,
    )
    manual_bbox = np.asarray([[0.55, 0.5, 0.1, 0.2]], dtype=np.float32)
    manual_key = mint_manual_curation_instance_keys(
        recording_identity=RECORDING_IDENTITY,
        refined_row_ids=np.asarray([1], dtype=np.int64),
        frame_indices=np.asarray([1], dtype=np.int32),
        bbox_norm_coords=manual_bbox,
        class_ids=np.asarray([2], dtype=np.int32),
    )[0]
    current = {
        "instances": {
            "frame_indices": np.asarray([1, 1], dtype=np.int32),
            "refined_row_ids": np.asarray([0, 1], dtype=np.int64),
            "bbox_norm_coords": np.asarray(
                [source_bbox[0], manual_bbox[0]],
                dtype=np.float64,
            ),
            "source_kind_codes": np.asarray(
                [
                    SOURCE_KIND_CODE_MAP["raw_detect"],
                    SOURCE_KIND_CODE_MAP["manual"],
                ],
                dtype=np.int8,
            ),
            "manual_edit_flags": np.asarray([False, True], dtype=np.bool_),
            "source_detect_row_index": np.asarray([0, -1], dtype=np.int32),
            "confidence_scores": np.asarray([0.9, 0.0], dtype=np.float32),
            "class_ids": np.asarray([1, 2], dtype=np.int32),
            "instance_key": np.asarray([100, manual_key], dtype=np.uint64),
            "reason": np.asarray(["clean", "manual_addition"], dtype=object),
        },
        "source_detections": {
            "source_detect_row_index": np.asarray([0, 1], dtype=np.int32),
            "frame_indices": np.asarray([1, 3], dtype=np.int32),
            "bbox_norm_coords": source_bbox,
            "decision_codes": np.asarray(
                [
                    SOURCE_DECISION_CODE_MAP["accepted"],
                    SOURCE_DECISION_CODE_MAP["filtered"],
                ],
                dtype=np.int8,
            ),
            "resolved_refined_row_id": np.asarray([0, -1], dtype=np.int64),
            "confidence_scores": np.asarray([0.9, 0.8], dtype=np.float32),
            "class_ids": np.asarray([1, 3], dtype=np.int32),
            "instance_key": np.asarray([100, 101], dtype=np.uint64),
            "reason": np.asarray(["clean", "filtered_blip"], dtype=object),
        },
    }
    return build_refined_detection_transition(
        current,
        n_frames=4,
        source_width=640,
        source_height=480,
        recording_identity=RECORDING_IDENTITY,
    )


def _lineage() -> RefinedDetectionSnapshotLineage:
    return RefinedDetectionSnapshotLineage(
        lineage_id="11111111-1111-4111-8111-111111111111",
        snapshot_id="22222222-2222-4222-8222-222222222222",
        recording_identity=RECORDING_IDENTITY,
        next_refined_row_id=2,
    )


def _source() -> RefinedDetectionSourceIdentity:
    return RefinedDetectionSourceIdentity(
        run_id="detect_shadow_1",
        run_manifest_digest="a" * 64,
        logical_content_digest="b" * 64,
    )


def test_shadow_publisher_is_standalone_consolidated_and_selector_ineligible(
    tmp_path: Path,
) -> None:
    shadow_root = tmp_path / "palette-refined-detection-shadows"
    destination = shadow_root / "multi_subject.zarr"

    result = publish_refined_detection_shadow(
        _transition(),
        destination=destination,
        run_id="refined_shadow_1",
        lineage=_lineage(),
        source=_source(),
        shadow_root=shadow_root,
    )

    assert result.output_path == destination.resolve()
    assert result.receipt["status"] == "complete"
    assert result.receipt["production_state_changes"] == []
    assert validate_refined_detection_run_manifest(result.manifest) == ()
    receipt = json.loads(
        (destination / "shadow_publication_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt == result.receipt

    root = zarr.open_group(
        str(destination),
        mode="r",
        use_consolidated=True,
    )
    family = root["refined_detect_runs"]
    run = family["refined_shadow_1"]
    assert root.attrs["benchmark_only"] is True
    assert root.attrs["selector_eligible"] is False
    assert family.attrs["selection_contract"] == "none_shadow_direct_path_only"
    assert "latest" not in family.attrs
    assert "latest_complete" not in family.attrs
    assert "authoritative_run" not in family.attrs
    assert run.attrs["status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert set(run["instances"].array_keys()) == {
        binding.path.split("/", 1)[1]
        for binding in REFINED_DETECTION_SCHEMA_V1.core_bindings
        if binding.path.startswith("instances/")
    }
    assert np.asarray(run["instances/frame_row_offsets"][:]).tolist() == [
        0,
        0,
        2,
        2,
        2,
    ]
    assert np.unique(np.asarray(run["instances/instance_key"][:])).size == 2


def test_shadow_destination_rejects_unsafe_existing_or_canonical_paths(
    tmp_path: Path,
) -> None:
    shadow_root = tmp_path / "safe"
    shadow_root.mkdir()

    with pytest.raises(ValueError, match="child"):
        require_safe_refined_detection_shadow_destination(
            shadow_root,
            shadow_root=shadow_root,
        )
    with pytest.raises(ValueError, match="suffix"):
        require_safe_refined_detection_shadow_destination(
            shadow_root / "candidate",
            shadow_root=shadow_root,
        )
    existing = shadow_root / "existing.zarr"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        require_safe_refined_detection_shadow_destination(
            existing,
            shadow_root=shadow_root,
        )
    canonical_parent = shadow_root / "recording_analysis.zarr"
    canonical_parent.mkdir()
    with pytest.raises(ValueError, match="recording archive"):
        require_safe_refined_detection_shadow_destination(
            canonical_parent / "shadow.zarr",
            shadow_root=shadow_root,
        )


def test_shadow_root_must_be_explicitly_safe(tmp_path: Path) -> None:
    unsafe_root = Path("/var/lib/palette-shadows")
    with pytest.raises(ValueError, match="Shadow roots"):
        require_safe_refined_detection_shadow_destination(
            unsafe_root / "candidate.zarr",
            shadow_root=unsafe_root,
        )
