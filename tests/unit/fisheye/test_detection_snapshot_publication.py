from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.shared.detect_reason_codec import encode_reason_bytes
from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.detection_snapshot_publication import (
    inspect_canonical_detection_successor_source,
    publish_canonical_detection_successor,
    publish_detection_snapshot_pair,
)
from fisheye.shared.zarr.canonical_detection_crop_preflight import (
    inspect_canonical_detection_crop_preflight,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
)
from fisheye.shared.zarr.refined_detection_schema import (
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
)


RECORDING_IDENTITY = "snapshot_publication_recording"


def _build_sources(archive: Path) -> None:
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.attrs["recording_id"] = RECORDING_IDENTITY
    raw_family = root.create_group("detect_runs")
    raw_family.attrs.update(
        {"latest": "detect_source", "latest_complete": "detect_source"}
    )
    raw = raw_family.create_group("detect_source")
    raw.attrs.update({"source_video_width": 640, "source_video_height": 480})
    frames = np.asarray([1, 3], dtype=np.int32)
    bbox = np.asarray(
        [[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]],
        dtype=np.float64,
    )
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    classes = np.asarray([1, 3], dtype=np.int32)
    keys = mint_detection_instance_keys(
        recording_identity=RECORDING_IDENTITY,
        frame_indices=frames,
        bbox_norm_coords=bbox.astype(np.float32),
        class_ids=classes,
    )
    raw.create_array("frame_indices", data=frames)
    raw.create_array("bbox_norm_coords", data=bbox)
    raw.create_array("scores", data=scores)
    raw.create_array("class_ids", data=classes)
    raw.create_array("instance_key", data=keys)
    raw.create_array("frame_counts", data=np.asarray([0, 1, 0, 1], dtype=np.int32))

    refined_family = root.create_group("refined_detect_runs")
    refined_family.attrs["authoritative_run"] = "refined_source"
    refined = refined_family.create_group("refined_source")
    instances = refined.create_group("instances")
    instances.create_array("frame_indices", data=frames[:1])
    instances.create_array("refined_row_ids", data=np.asarray([0], dtype=np.int64))
    instances.create_array("bbox_norm_coords", data=bbox[:1])
    instances.create_array(
        "source_kind_codes",
        data=np.asarray([SOURCE_KIND_CODE_MAP["raw_detect"]], dtype=np.int8),
    )
    instances.create_array("manual_edit_flags", data=np.asarray([False]))
    instances.create_array(
        "source_detect_row_index", data=np.asarray([0], dtype=np.int32)
    )
    instances.create_array("confidence_scores", data=scores[:1])
    instances.create_array("class_ids", data=classes[:1])
    instances.create_array("instance_key", data=keys[:1])
    instances.create_array(
        "reason_bytes",
        data=encode_reason_bytes(np.asarray(["clean"], dtype=object)),
    )

    source = refined.create_group("source_detections")
    source.create_array("source_detect_row_index", data=np.arange(2, dtype=np.int32))
    source.create_array("frame_indices", data=frames)
    source.create_array("bbox_norm_coords", data=bbox)
    source.create_array(
        "decision_codes",
        data=np.asarray(
            [
                SOURCE_DECISION_CODE_MAP["accepted"],
                SOURCE_DECISION_CODE_MAP["filtered"],
            ],
            dtype=np.int8,
        ),
    )
    source.create_array(
        "resolved_refined_row_id", data=np.asarray([0, -1], dtype=np.int64)
    )
    source.create_array("confidence_scores", data=scores)
    source.create_array("class_ids", data=classes)
    source.create_array("instance_key", data=keys)
    source.create_array(
        "reason_bytes",
        data=encode_reason_bytes(np.asarray(["clean", "filtered_blip"], dtype=object)),
    )


def test_pair_is_atomically_placed_but_not_selected(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _build_sources(archive)

    result = publish_detection_snapshot_pair(
        analysis_zarr=archive,
        source_detect_group_path="detect_runs/detect_source",
        source_refined_group_path="refined_detect_runs/refined_source",
        recording_identity=RECORDING_IDENTITY,
        canonical_run_id="detect_snapshot_v1",
        refined_run_id="refined_snapshot_v1",
        scratch_root=scratch,
        coordinate_catalog=True,
    )

    assert result["status"] == "complete"
    assert result["selector_eligible"] is False
    assert result["registry_updated"] is False
    assert result["storage_profiles"] == {
        "canonical": "detection_published_access_aware_v1",
        "refined": "detection_published_access_aware_v1",
    }
    assert result["validation"] == {
        "canonical_errors": [],
        "refined_errors": [],
        "direct_consolidated_metadata_equal": True,
    }
    assert list(scratch.iterdir()) == []

    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    raw_family = root["detect_runs"]
    refined_family = root["refined_detect_runs"]
    assert raw_family.attrs["latest"] == "detect_source"
    assert raw_family.attrs["latest_complete"] == "detect_source"
    assert refined_family.attrs["authoritative_run"] == "refined_source"

    canonical = raw_family["detect_snapshot_v1"]
    refined = refined_family["refined_snapshot_v1"]
    assert canonical.attrs["run_manifest"]["schema_version"] == (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert refined.attrs["run_manifest"]["schema_version"] == (
        REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert "coordinate_contract" in canonical.attrs["run_manifest"]["payload"]
    assert "coordinate_contract" in refined.attrs["run_manifest"]["payload"]
    assert canonical.attrs["stage_selector_eligible"] is False
    assert canonical.attrs["immutable_snapshot"] is True
    assert canonical.attrs["production_selector_activation"] == "deferred"
    assert "shadow_only" not in canonical.attrs
    assert refined.attrs["stage_selector_eligible"] is False
    assert refined.attrs["immutable_snapshot"] is True
    assert refined.attrs["production_selector_activation"] == "deferred"
    assert "shadow_only" not in refined.attrs
    assert np.asarray(canonical["instances/frame_row_offsets"][:]).tolist() == [
        0,
        0,
        1,
        1,
        2,
    ]
    assert np.asarray(refined["instances/frame_row_offsets"][:]).tolist() == [
        0,
        0,
        1,
        1,
        1,
    ]
    assert np.asarray(refined["source_detections/frame_row_offsets"][:]).tolist() == [
        0,
        0,
        1,
        1,
        2,
    ]


def test_raw_successor_dry_run_preserves_source_identity_without_writes(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _build_sources(archive)

    result = inspect_canonical_detection_successor_source(
        analysis_zarr=archive,
        source_detect_group_path="detect_runs/detect_source",
        recording_identity=RECORDING_IDENTITY,
        successor_run_id="detect_canonical_v3",
    )

    assert result["status"] == "ready"
    assert result["successor_group_path"] == ("detect_runs/detect_canonical_v3")
    assert result["instance_key_policy"] == "preserved_from_source"
    assert result["storage_profile_id"] == ("detection_published_access_aware_v1")
    assert result["coordinate_catalog"] is True
    assert result["selector_eligible"] is False
    assert not (archive / result["successor_group_path"]).exists()


def test_raw_successor_is_canonical_v3_atomic_and_selector_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _build_sources(archive)
    source = zarr.open_group(
        str(archive / "detect_runs/detect_source"),
        mode="a",
        use_consolidated=False,
    )
    custom_keys = np.asarray([111, 222], dtype=np.uint64)
    source["instance_key"][:] = custom_keys

    result = publish_canonical_detection_successor(
        analysis_zarr=archive,
        source_detect_group_path="detect_runs/detect_source",
        recording_identity=RECORDING_IDENTITY,
        successor_run_id="detect_canonical_v3",
        scratch_root=scratch,
    )

    assert result["status"] == "complete"
    assert result["coordinate_catalog"] is True
    assert result["instance_key_policy"] == "preserved_from_source"
    assert result["selector_eligible"] is False
    assert result["registry_updated"] is False
    assert result["selectors_before"] == result["selectors_after"]
    assert result["validation"]["canonical_errors"] == []
    assert result["validation"]["direct_consolidated_metadata_equal"] is True
    assert list(scratch.iterdir()) == []

    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    family = root["detect_runs"]
    assert family.attrs["latest"] == "detect_source"
    assert family.attrs["latest_complete"] == "detect_source"
    successor = family["detect_canonical_v3"]
    assert successor.attrs["run_manifest"]["schema_version"] == (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert successor.attrs["stage_selector_eligible"] is False
    assert successor.attrs["immutable_snapshot"] is True
    assert successor.attrs["production_selector_activation"] == "deferred"
    assert "shadow_only" not in successor.attrs
    np.testing.assert_array_equal(
        successor["instances/instance_key"][:],
        custom_keys,
    )
    np.testing.assert_array_equal(source["instance_key"][:], custom_keys)
    assert np.asarray(successor["instances/frame_row_offsets"][:]).tolist() == [
        0,
        0,
        1,
        1,
        2,
    ]

    from fisheye.shared.zarr import canonical_detection_crop_preflight as preflight

    monkeypatch.setattr(
        preflight,
        "load_persisted_acquisition_camera_authority",
        lambda _root: (
            object(),
            SimpleNamespace(
                record=SimpleNamespace(
                    source_total_frames=4,
                    width_px=640,
                    height_px=480,
                )
            ),
        ),
    )
    policy = CropGeometryPolicy(
        purpose="ordinary_zebrafish_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(348, 348),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )
    with pytest.raises(ValueError, match="candidate-preflight allowance"):
        inspect_canonical_detection_crop_preflight(
            analysis_zarr=archive,
            detection_run_id="detect_canonical_v3",
            policy=policy,
        )
    crop_preflight = inspect_canonical_detection_crop_preflight(
        analysis_zarr=archive,
        detection_run_id="detect_canonical_v3",
        policy=policy,
        allow_selector_ineligible_candidate=True,
    )
    assert crop_preflight["status"] == "ready"
    assert crop_preflight["padding"]["padded_row_count"] == 2
    assert crop_preflight["padding"]["examples"][0]["padding_ltrb"] == [
        14,
        0,
        0,
        0,
    ]
    assert crop_preflight["crop_zarr_writes"] is False


def test_raw_successor_requires_existing_instance_keys_before_writing(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _build_sources(archive)
    source = zarr.open_group(
        str(archive / "detect_runs/detect_source"),
        mode="a",
        use_consolidated=False,
    )
    del source["instance_key"]

    with pytest.raises(ValueError, match="lacks persisted instance_key"):
        publish_canonical_detection_successor(
            analysis_zarr=archive,
            source_detect_group_path="detect_runs/detect_source",
            recording_identity=RECORDING_IDENTITY,
            successor_run_id="detect_canonical_v3",
            scratch_root=scratch,
        )

    assert not (archive / "detect_runs/detect_canonical_v3").exists()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert root["detect_runs"].attrs["latest"] == "detect_source"


def test_raw_successor_tombstones_postcopy_metadata_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _build_sources(archive)

    from fisheye.shared.zarr import detection_snapshot_publication as module

    original = module.consolidate_metadata_capture_expected_warnings
    calls = 0

    def fail_once(path):  # noqa: ANN001, ANN202
        nonlocal calls
        if Path(path).resolve() == archive.resolve():
            calls += 1
        if Path(path).resolve() == archive.resolve() and calls == 1:
            raise RuntimeError("injected consolidation failure")
        return original(path)

    monkeypatch.setattr(
        module,
        "consolidate_metadata_capture_expected_warnings",
        fail_once,
    )

    with pytest.raises(RuntimeError, match="injected consolidation failure"):
        publish_canonical_detection_successor(
            analysis_zarr=archive,
            source_detect_group_path="detect_runs/detect_source",
            recording_identity=RECORDING_IDENTITY,
            successor_run_id="detect_canonical_v3",
            scratch_root=scratch,
        )

    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    family = root["detect_runs"]
    failed = family["detect_canonical_v3"]
    assert family.attrs["latest"] == "detect_source"
    assert family.attrs["latest_complete"] == "detect_source"
    # The immutable scientific payload remains sealed as complete, while the
    # lifecycle completion marker is the fail-closed publication authority.
    assert failed.attrs["status"] == "complete"
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert "atomic_publication_tombstone" in failed.attrs


def test_clipped_compatibility_source_fails_before_archive_publication(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _build_sources(archive)
    refined = zarr.open_group(
        str(archive / "refined_detect_runs/refined_source"),
        mode="a",
        use_consolidated=False,
    )
    refined["instances"].create_array(
        "source_recording_frame_ids",
        data=np.asarray([2], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="refuses clipped lineage arrays"):
        publish_detection_snapshot_pair(
            analysis_zarr=archive,
            source_detect_group_path="detect_runs/detect_source",
            source_refined_group_path="refined_detect_runs/refined_source",
            recording_identity=RECORDING_IDENTITY,
            canonical_run_id="detect_snapshot_v1",
            refined_run_id="refined_snapshot_v1",
            scratch_root=scratch,
        )

    assert not (archive / "detect_runs/detect_snapshot_v1").exists()
    assert not (archive / "refined_detect_runs/refined_snapshot_v1").exists()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert root["detect_runs"].attrs["latest"] == "detect_source"
    assert root["refined_detect_runs"].attrs["authoritative_run"] == ("refined_source")
