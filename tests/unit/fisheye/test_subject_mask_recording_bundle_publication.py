from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest
import zarr

from fisheye.cluster.subject_masks.publish_recording_bundle import (
    publish_recording_subject_mask_bundle,
)
from fisheye.shared.subject_mask_attempt import (
    build_subject_mask_attempt,
    build_subject_mask_scientific_identity,
)
from fisheye.shared.subject_mask_worker_receipt import (
    RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    build_subject_mask_worker_semantic_receipt,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR,
)
from fisheye.shared.zarr.subject_mask_schema import (
    derive_subject_mask_frame_row_offsets,
    derive_subject_mask_metrics,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    subject_mask_array_unit_document,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.utils.import_refined_subject_mask_clip_packages import (
    import_refined_subject_mask_clip_packages,
)
from tests.unit.fisheye.test_import_refined_subject_mask_clip_packages import (
    _write_package,
)


def _create_array(group: zarr.Group, path: str, values: np.ndarray) -> None:
    parts = path.split("/")
    target = group
    for part in parts[:-1]:
        target = target.require_group(part)
    target.create_array(parts[-1], data=values, overwrite=True)


def _surfaces() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    masks = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[:, 1, 2, 2] = 1
    masks[:, 2, 2, 5] = 1
    masks[:, 3, 5, 3] = 1
    metrics = derive_subject_mask_metrics(masks)
    probabilities = masks * np.uint8(255)
    raw = {
        "mask_probs_roi": probabilities,
        "available_channels": np.ones((4,), dtype=bool),
        "metrics/prob_max": (
            np.max(probabilities, axis=(2, 3)).astype(np.float32) / np.float32(255.0)
        ),
        **{f"metrics/{name}": value for name, value in metrics.items()},
    }
    refined = {
        "masks_roi": masks,
        "available_channels": np.ones((4,), dtype=bool),
        **{f"metrics/{name}": value for name, value in metrics.items()},
    }
    return raw, refined


def _seal_worker(
    draft_path: Path,
    run: zarr.Group,
    *,
    stage_kind: str,
    paths: tuple[str, ...],
) -> None:
    row_count = int(run["source_crop_row_ids"].shape[0])
    run_path = str(run.path).strip("/")
    science = build_subject_mask_scientific_identity(
        stage_kind=stage_kind,
        model={"artifact": "pytest"},
        crop={"run_id": "crop_001"},
        pixels={"digest": "a" * 64},
        row_identity={"rows": row_count, "run_path": run_path},
        inference_contract={"components": list(run.attrs["mask_labels"])},
    )
    attempt = build_subject_mask_attempt(
        scientific_identity=science,
        run_path=run_path,
        attempt_id=str(uuid5(NAMESPACE_URL, f"pytest:{stage_kind}:{run_path}")),
    )
    arrays = {path: run[path] for path in paths}
    receipt = build_subject_mask_worker_semantic_receipt(
        stage_kind=stage_kind,
        run_path=str(run.path).strip("/"),
        scientific_identity=science,
        attempt=attempt,
        scope={"recording": "recording_001"},
        row_count=row_count,
        array_document=subject_mask_array_unit_document(arrays, paths, unit_rows=2),
        required_paths=paths,
        roi_aligned_paths=tuple(path for path in paths if path != "available_channels"),
    )
    receipt_bytes = canonical_json_bytes(receipt)
    relative = f"{str(run.path).strip('/')}/worker_semantic_receipt.json"
    receipt_path = draft_path / relative
    receipt_path.write_bytes(receipt_bytes)
    run.attrs.update(
        {
            "subject_mask_scientific_identity": science,
            "subject_mask_attempt": attempt,
            "subject_mask_worker_semantic_receipt_binding": {
                "schema_id": receipt["schema_id"],
                "schema_version": receipt["schema_version"],
                "payload_digest": receipt["payload_digest"],
                "relative_path": relative,
                "document_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "storage": "strict_json_sidecar_v1",
            },
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
        }
    )


def _draft(
    tmp_path: Path,
    *,
    raw_parent: str,
    raw_slices: dict[str, slice] | None = None,
    refined_slices: dict[str, slice] | None = None,
    split_eye_registry: bool = False,
) -> Path:
    path = tmp_path / "draft.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    crop = root.require_group("crop_runs").create_group("crop_001")
    for name, values in {
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
        "source_crop_xywh": np.asarray(
            [[0, 0, 8, 8], [1, 0, 8, 8], [0, 1, 8, 8], [1, 1, 8, 8]],
            dtype=np.float32,
        ),
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(frames, n_frames=4),
    }.items():
        _create_array(crop, name, values)
    raw_values, refined_values = _surfaces()
    if split_eye_registry:
        refined_masks = refined_values["masks_roi"]
        raw_masks = np.stack(
            (
                refined_masks[:, 0],
                np.maximum(refined_masks[:, 1], refined_masks[:, 2]),
                refined_masks[:, 3],
            ),
            axis=1,
        )
        raw_metrics = derive_subject_mask_metrics(raw_masks)
        raw_probabilities = raw_masks * np.uint8(255)
        raw_values = {
            "mask_probs_roi": raw_probabilities,
            "available_channels": np.ones((3,), dtype=bool),
            "metrics/prob_max": (
                np.max(raw_probabilities, axis=(2, 3)).astype(np.float32)
                / np.float32(255.0)
            ),
            **{
                f"metrics/{name}": value
                for name, value in raw_metrics.items()
            },
        }
    row_values = {
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
    }
    raw_parent_group = root.require_group(raw_parent)
    refined_parent = root.require_group("refined_subject_masks_runs")
    refined_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    raw_labels = (
        ["subject_body", "eyes_union", "swim_bladder"]
        if split_eye_registry
        else refined_labels
    )
    for raw_name, row_slice in (
        raw_slices or {"raw_draft": slice(0, len(frames))}
    ).items():
        raw = raw_parent_group.create_group(raw_name)
        raw.attrs.update(
            {"mask_labels": raw_labels, "mask_probability_threshold": 0.5}
        )
        for name, values in row_values.items():
            _create_array(raw, name, values[row_slice])
        for name, values in raw_values.items():
            _create_array(
                raw,
                name,
                values if name == "available_channels" else values[row_slice],
            )
        _seal_worker(
            path,
            raw,
            stage_kind="raw_subject_mask",
            paths=RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
        )
    for refined_name, row_slice in (
        refined_slices or {"refined_draft": slice(0, len(frames))}
    ).items():
        refined = refined_parent.create_group(refined_name)
        refined.attrs["mask_labels"] = refined_labels
        for name, values in {
            "source_crop_row_ids": row_values["source_crop_row_ids"],
            **refined_values,
        }.items():
            _create_array(
                refined,
                name,
                values if name == "available_channels" else values[row_slice],
            )
        _seal_worker(
            path,
            refined,
            stage_kind="refined_subject_mask",
            paths=REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
        )
    return path


@pytest.mark.parametrize("raw_parent", ("subject_mask_runs", "subject_mask_shard_runs"))
def test_recording_bundle_publication_is_proof_bound_and_inactive(
    tmp_path: Path,
    raw_parent: str,
) -> None:
    draft = _draft(tmp_path, raw_parent=raw_parent)
    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent=raw_parent,
        raw_draft_run="raw_draft",
        refined_draft_run="refined_draft",
        raw_run="raw_001",
        refined_run="refined_001",
        quality_run="quality_001",
        bundle_id="bundle_001",
        local_output_root=tmp_path / "local_outputs",
        quality_scratch_root=tmp_path / "quality_scratch",
    )

    assert result["status"] == "complete"
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs
    assert (
        published["subject_mask_bundle_runs/bundle_001"].attrs[
            "palette_run_completion_status"
        ]
        == "complete"
    )
    for path in (
        "subject_mask_runs/raw_001",
        "refined_subject_masks_runs/refined_001",
        "subject_mask_quality_runs/quality_001",
    ):
        assert published[path].attrs["stage_selector_eligible"] is False


def test_recording_bundle_composes_multiple_raw_clip_shards_without_reordering(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
    )
    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_b",
        # Deliberately reverse command order: canonical crop-row intervals,
        # rather than caller order, define the recording assembly.
        raw_draft_runs=("raw_clip_b", "raw_clip_a"),
        refined_draft_run="refined_draft",
        raw_run="raw_multi_001",
        refined_run="refined_multi_001",
        quality_run="quality_multi_001",
        bundle_id="bundle_multi_001",
        local_output_root=tmp_path / "local_multi_outputs",
        quality_scratch_root=tmp_path / "quality_multi_scratch",
    )

    assert result["status"] == "complete"
    assert result["n_rois"] == 4
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    raw = published["subject_mask_runs/raw_multi_001"]
    np.testing.assert_array_equal(
        raw["source_crop_row_ids"][:], np.arange(4, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        raw["instance_key"][:], np.asarray([101, 102, 201, 301], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        raw["mask_probs_roi"][:, 0, 1, 1], np.full(4, 255, dtype=np.uint8)
    )
    assert raw.attrs["stage_selector_eligible"] is False
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs


def test_recording_bundle_composes_multiple_refined_clip_shards_without_reordering(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
        refined_slices={
            "refined_clip_a": slice(0, 2),
            "refined_clip_b": slice(2, 4),
        },
    )
    analysis = tmp_path / "analysis_refined_shards.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_b",
        raw_draft_runs=("raw_clip_b", "raw_clip_a"),
        refined_draft_run="refined_clip_b",
        refined_draft_runs=("refined_clip_b", "refined_clip_a"),
        raw_run="raw_multi_refined_001",
        refined_run="refined_multi_refined_001",
        quality_run="quality_multi_refined_001",
        bundle_id="bundle_multi_refined_001",
        local_output_root=tmp_path / "local_multi_refined_outputs",
        quality_scratch_root=tmp_path / "quality_multi_refined_scratch",
    )

    assert result["status"] == "complete"
    assert result["n_rois"] == 4
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    refined = published["refined_subject_masks_runs/refined_multi_refined_001"]
    np.testing.assert_array_equal(
        refined["source_crop_row_ids"][:], np.arange(4, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        refined["instance_key"][:],
        np.asarray([101, 102, 201, 301], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        refined["masks_roi"][:, 0, 1, 1], np.ones(4, dtype=np.uint8)
    )
    assert refined.attrs["stage_selector_eligible"] is False
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs


def test_recording_bundle_binds_distinct_raw_and_refined_component_registries(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
        refined_slices={
            "refined_clip_a": slice(0, 2),
            "refined_clip_b": slice(2, 4),
        },
        split_eye_registry=True,
    )
    analysis = tmp_path / "analysis_split_eye_registry.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_a",
        raw_draft_runs=("raw_clip_a", "raw_clip_b"),
        refined_draft_run="refined_clip_a",
        refined_draft_runs=("refined_clip_a", "refined_clip_b"),
        raw_run="raw_split_eye_001",
        refined_run="refined_split_eye_001",
        quality_run="quality_split_eye_001",
        bundle_id="bundle_split_eye_001",
        local_output_root=tmp_path / "local_split_eye_outputs",
        quality_scratch_root=tmp_path / "quality_split_eye_scratch",
    )

    assert result["status"] == "complete"
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    raw = published["subject_mask_runs/raw_split_eye_001"]
    refined = published[
        "refined_subject_masks_runs/refined_split_eye_001"
    ]
    quality = published[
        "subject_mask_quality_runs/quality_split_eye_001"
    ]
    assert raw.attrs["mask_labels"] == [
        "subject_body",
        "eyes_union",
        "swim_bladder",
    ]
    assert refined.attrs["mask_labels"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert raw["available_channels"].shape == (3,)
    assert refined["available_channels"].shape == (4,)
    quality_schema = quality.attrs["run_manifest"]["payload"]["logical_schema"]
    assert quality_schema["dimensions"]["n_channels"] == 4
    assert quality_schema["components"]["labels"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    bundle = published["subject_mask_bundle_runs/bundle_split_eye_001"]
    bundle_manifest = bundle.attrs["run_manifest"]
    assert bundle_manifest["schema_version"] == 2
    cross_binding = bundle_manifest["payload"]["cross_binding"]
    assert (
        cross_binding["component_registry_policy"]
        == "raw_and_refined_bound_independently_v1"
    )
    assert cross_binding["raw_components"]["labels"] == [
        "subject_body",
        "eyes_union",
        "swim_bladder",
    ]
    assert cross_binding["components"]["labels"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert "available_channels" not in cross_binding[
        "raw_refined_identity_array_values_sha256"
    ]
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs


def test_two_clip_proof_import_flows_into_atomic_recording_bundle(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
    )
    mutable = zarr.open_group(str(draft), mode="a", use_consolidated=False)
    mutable.attrs["recording_frame_index_row_count"] = 4
    labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    package_a = _write_package(
        tmp_path,
        package_name="bundle_clip_a",
        run_name="refined_bundle_a",
        crop_row_ids=[0, 1],
        source_crop_run="crop_001",
        labels=labels,
        frame_indices=[0, 0],
        production_proof=True,
        roi_shape=(8, 8),
    )
    package_b = _write_package(
        tmp_path,
        package_name="bundle_clip_b",
        run_name="refined_bundle_b",
        crop_row_ids=[2, 3],
        source_crop_run="crop_001",
        labels=labels,
        frame_indices=[2, 3],
        production_proof=True,
        roi_shape=(8, 8),
    )
    imported = import_refined_subject_mask_clip_packages(
        zarr_path=draft,
        package_paths=(package_b, package_a),
        output_run="refined_recording_import",
        expected_target_crop_run="crop_001",
        require_production_proof=True,
        array_copy_workers=2,
    )
    assert imported["selector_eligible"] is False
    assert imported["source_validation_receipt_payload_digest"]

    analysis = tmp_path / "analysis_from_clips.zarr"
    analysis_root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    analysis_root.attrs["recording_id"] = "recording_001"
    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_a",
        raw_draft_runs=("raw_clip_b", "raw_clip_a"),
        refined_draft_run="refined_recording_import",
        raw_run="raw_from_clips",
        refined_run="refined_from_clips",
        quality_run="quality_from_clips",
        bundle_id="bundle_from_clips",
        local_output_root=tmp_path / "local_clipped_outputs",
        quality_scratch_root=tmp_path / "quality_clipped_scratch",
    )

    assert result["status"] == "complete"
    assert result["n_rois"] == 4
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs
    for path in (
        "subject_mask_runs/raw_from_clips",
        "refined_subject_masks_runs/refined_from_clips",
        "subject_mask_quality_runs/quality_from_clips",
    ):
        assert published[path].attrs["stage_selector_eligible"] is False
    np.testing.assert_array_equal(
        published["refined_subject_masks_runs/refined_from_clips"][
            "source_crop_row_ids"
        ][:],
        np.arange(4, dtype=np.int64),
    )
