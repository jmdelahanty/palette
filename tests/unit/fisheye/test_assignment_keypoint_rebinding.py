from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from fisheye.shared.zarr.assignment_keypoint_rebinding import (
    ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
    ASSIGNMENT_KEYPOINT_REBINDING_POLICY,
    ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID,
    ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION,
    ASSIGNMENT_KEYPOINT_SOURCE_DIRECT_PROFILE,
    ASSIGNMENT_KEYPOINT_SOURCE_REBINDING_PROFILE,
    _assignment_collection_source,
    _assignment_collection_source_run,
    _chunked_equivalence,
    _refined_historical_labels,
    inspect_assignment_keypoint_rebinding,
    load_assignment_keypoint_source,
    load_assignment_keypoint_rebinding_manifest,
    validate_assignment_keypoint_rebinding_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _collection() -> dict[str, object]:
    return {
        "schema_id": "palette.subject_mask.assignment_keypoint_collection",
        "schema_version": 1,
        "mode": "exact_worker_partition",
        "row_policy": "ordered_contiguous_recording_crop_rows_v1",
        "n_rois": 5,
        "workers": [
            {
                "global_row_interval": {"start_row": 0, "stop_row": 2},
                "assignment": {
                    "assignment_keypoint_group": "keypoints_runs",
                    "assignment_keypoints_run": "historical",
                    "assignment_keypoint_success_dataset": "detection_success",
                },
            },
            {
                "global_row_interval": {"start_row": 2, "stop_row": 5},
                "assignment": {
                    "assignment_keypoint_group": "keypoints_runs",
                    "assignment_keypoints_run": "historical",
                    "assignment_keypoint_success_dataset": "detection_success",
                },
            },
        ],
    }


def _manifest() -> dict[str, object]:
    payload = {
        "rebinding_run_id": "rebind_001",
        "policy": ASSIGNMENT_KEYPOINT_REBINDING_POLICY,
        "recording_identity": "recording",
        "camera_identity": "camera",
        "row_count": 5,
        "assignment_state": "used",
        "subject_mask_source": {
            "bundle_id": "bundle_001",
            "bundle_manifest_payload_digest": "1" * 64,
            "bundle_coordinate_authority_digest": "2" * 64,
            "refined_run_path": "refined_subject_masks_runs/refined_001",
            "assignment_collection_digest": "3" * 64,
            "historical_keypoint_run_path": "keypoints_runs/historical",
        },
        "canonical_keypoint_source": {
            "authority_profile": ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
            "run_path": "keypoints_runs/canonical",
            "run_manifest_payload_digest": "a" * 64,
            "run_manifest_document_digest": "b" * 64,
            "keypoint_bundle_authority_generation": 1,
            "keypoint_bundle_authority_digest": "c" * 64,
            "coordinate_successor_authority_digest": "d" * 64,
            "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
            "eye_keypoint_indices": {"eye_left": 1, "eye_right": 2},
            "keypoints_dataset": "keypoints_roi",
            "success_dataset": "pose_success",
        },
        "equivalence": {
            name: {
                "shape": [5, 3, 2] if name.startswith("keypoints_roi") else [5],
                "historical_dtype": (
                    "float64" if name.startswith("keypoints_roi") else "uint64"
                ),
                "canonical_dtype": (
                    "float32" if name.startswith("keypoints_roi") else "uint64"
                ),
                "normalization": (
                    "numpy_astype_float32_c_order_v1"
                    if name.startswith("keypoints_roi")
                    else "identity"
                ),
                "digest_algorithm": "sha256_c_contiguous_bytes_v1",
                "normalized_sha256": "e" * 64,
            }
            for name in (
                "source_crop_row_ids_to_source_crop_row_ids",
                "instance_key_to_instance_key",
                ("source_acquisition_frame_index_to_" "source_acquisition_frame_index"),
                "keypoints_roi_to_keypoints_roi",
                "detection_success_to_pose_success",
            )
        },
        "selection_policy": "explicit_bundle_and_keypoint_run_no_fallback_v1",
        "stage_selector_eligible": False,
        "production_state_changes": [],
    }
    return {
        "schema_id": ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID,
        "schema_version": ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def test_assignment_collection_requires_one_gapless_recording_run() -> None:
    assert _assignment_collection_source_run(_collection()) == "historical"

    refined = _collection()
    for worker in refined["workers"]:
        worker["assignment"]["assignment_keypoint_group"] = "refined_keypoints_runs"
        worker["assignment"]["assignment_keypoint_success_dataset"] = (
            "usable_keypoints"
        )
    assert _assignment_collection_source(refined) == (
        "refined_keypoints_runs",
        "historical",
        "usable_keypoints",
    )

    direct = _collection()
    for worker in direct["workers"]:
        worker["assignment"]["assignment_keypoint_success_dataset"] = "pose_success"
    assert _assignment_collection_source(direct) == (
        "keypoints_runs",
        "historical",
        "pose_success",
    )

    gap = _collection()
    gap["workers"][1]["global_row_interval"]["start_row"] = 3
    try:
        _assignment_collection_source_run(gap)
    except ValueError as exc:
        assert "partition" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("gapped assignment collection was accepted")

    mixed = _collection()
    mixed["workers"][1]["assignment"]["assignment_keypoints_run"] = "other"
    try:
        _assignment_collection_source_run(mixed)
    except ValueError as exc:
        assert "one recording-wide" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("mixed assignment collection was accepted")

    mixed_success = direct
    mixed_success["workers"][1]["assignment"][
        "assignment_keypoint_success_dataset"
    ] = "detection_success"
    with pytest.raises(ValueError, match="one recording-wide"):
        _assignment_collection_source(mixed_success)


def test_refined_assignment_labels_require_exact_active_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = ["swim_bladder", "eye_left", "eye_right"]
    source_bindings = {"schema_id": "fixture.refined.source"}
    manifest = {
        "payload_digest": "1" * 64,
        "payload": {
            "logical_content": {"digest": "2" * 64},
            "source_bindings": source_bindings,
        },
    }
    historical = SimpleNamespace(
        attrs={"run_manifest": manifest, "source_bindings": source_bindings}
    )
    source = SimpleNamespace(
        recording_identity="recording",
        raw_run_id="raw",
        raw_manifest_digest="3" * 64,
        raw_logical_content_digest="4" * 64,
        skeleton_semantics={"keypoint_labels": labels},
    )
    authority = {
        "members": {
            "refined_keypoints": {
                "run_path": "refined_keypoints_runs/historical",
                "manifest_payload_digest": manifest["payload_digest"],
                "manifest_document_digest": canonical_json_sha256(manifest),
                "logical_content_digest": "2" * 64,
            },
            "raw_keypoints": {
                "run_path": "keypoints_runs/raw",
                "manifest_document_digest": "3" * 64,
                "logical_content_digest": "4" * 64,
            },
        }
    }
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "validate_refined_keypoint_run_manifest",
        lambda _manifest: (),
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "refined_keypoint_source_bindings_from_manifest",
        lambda _bindings: source,
    )

    assert _refined_historical_labels(
        historical,
        historical_path="refined_keypoints_runs/historical",
        active_authority=authority,
        recording_identity="recording",
    ) == labels

    changed = copy.deepcopy(authority)
    changed["members"]["refined_keypoints"]["run_path"] = (
        "refined_keypoints_runs/other"
    )
    with pytest.raises(ValueError, match="exact active"):
        _refined_historical_labels(
            historical,
            historical_path="refined_keypoints_runs/historical",
            active_authority=changed,
            recording_identity="recording",
        )


def test_chunked_equivalence_seals_declared_float32_normalization() -> None:
    values64 = np.asarray(
        [[[1.25, 2.5]], [[np.nan, np.nan]], [[3.75, 4.0]]],
        dtype=np.float64,
    )
    values32 = values64.astype(np.float32)
    record = _chunked_equivalence(
        values64,
        values32,
        normalized_dtype=np.dtype("float32"),
        block_rows=2,
    )
    assert record["normalization"] == "numpy_astype_float32_c_order_v1"
    assert record["historical_dtype"] == "float64"
    assert record["canonical_dtype"] == "float32"

    changed = values32.copy()
    changed[-1, 0, 0] += 1
    try:
        _chunked_equivalence(
            values64,
            changed,
            normalized_dtype=np.dtype("float32"),
            block_rows=2,
        )
    except ValueError as exc:
        assert "values differ" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("changed keypoints were accepted")


def test_rebinding_manifest_is_closed_and_digest_sealed() -> None:
    manifest = _manifest()
    assert validate_assignment_keypoint_rebinding_manifest(manifest) == ()

    refined = copy.deepcopy(manifest)
    refined["payload"]["subject_mask_source"]["historical_keypoint_run_path"] = (
        "refined_keypoints_runs/historical"
    )
    success = refined["payload"]["equivalence"].pop(
        "detection_success_to_pose_success"
    )
    refined["payload"]["equivalence"]["usable_keypoints_to_pose_success"] = success
    refined["payload_digest"] = canonical_json_sha256(refined["payload"])
    assert validate_assignment_keypoint_rebinding_manifest(refined) == ()

    tampered = copy.deepcopy(manifest)
    tampered["payload"]["row_count"] = 6
    assert any(
        "payload digest differs" in error
        for error in validate_assignment_keypoint_rebinding_manifest(tampered)
    )

    expanded = copy.deepcopy(manifest)
    expanded["payload"]["unexpected"] = True
    expanded["payload_digest"] = canonical_json_sha256(expanded["payload"])
    assert any(
        "payload fields are not exact" in error
        for error in validate_assignment_keypoint_rebinding_manifest(expanded)
    )


def test_rebinding_loader_reuses_provided_bound_mask_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    collection = _collection()
    manifest["payload"]["subject_mask_source"]["assignment_collection_digest"] = (
        canonical_json_sha256(collection)
    )
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    run = SimpleNamespace(
        attrs={
            "run_manifest": manifest,
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "production_candidate": True,
        }
    )
    root = {"subject_mask_assignment_keypoint_rebinding_runs/rebind_001": run}
    bundle = SimpleNamespace(
        archive_path=Path("/archive"),
        recording_identity="recording",
        camera_identity="camera",
        n_rois=5,
        bundle_id="bundle_001",
        bundle_manifest={"payload_digest": "1" * 64},
        authority_digest="2" * 64,
        refined_run_path="refined_subject_masks_runs/refined_001",
        assignment_keypoint_collection=collection,
        admission_receipt={"status": "admitted"},
    )
    keypoints = manifest["payload"]["canonical_keypoint_source"]
    source = SimpleNamespace(
        active_keypoint_bundle_authority_digest=keypoints[
            "keypoint_bundle_authority_digest"
        ],
        active_keypoint_bundle_authority={
            "generation": keypoints["keypoint_bundle_authority_generation"]
        },
        run_path=keypoints["run_path"],
        successor_authority_digest=keypoints["coordinate_successor_authority_digest"],
        manifest={"payload_digest": keypoints["run_manifest_payload_digest"]},
        manifest_digest=keypoints["run_manifest_document_digest"],
    )

    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding.open_zarr_root",
        lambda *_args, **_kwargs: root,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding.zarr.open_group",
        lambda *_args, **_kwargs: root,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "require_bound_recording_subject_mask_coordinate_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_recording_subject_mask_coordinate_authority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provided authority must prevent a second bundle load")
        ),
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_keypoint_coordinate_successor_admission",
        lambda *_args, **_kwargs: source,
    )

    loaded = load_assignment_keypoint_rebinding_manifest(
        Path("/archive"),
        rebinding_run_id="rebind_001",
        subject_mask_authority=bundle,
    )

    assert loaded == manifest


def test_direct_assignment_source_resolves_one_canonical_successor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = (tmp_path / "analysis.zarr").resolve()
    archive.mkdir()

    def digest(values: np.ndarray) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(values).tobytes(order="C")
        ).hexdigest()

    identity = {
        "source_crop_row_ids": np.asarray([0, 1], dtype=np.int64),
        "instance_key": np.asarray([10, 11], dtype=np.uint64),
        "source_acquisition_frame_index": np.asarray([20, 21], dtype=np.int64),
    }
    keypoints = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    pose_success = np.asarray([True, False], dtype=np.bool_)

    class Group(dict[str, object]):
        def __init__(self, values: dict[str, object], *, path: str) -> None:
            super().__init__(values)
            self.path = path

    run_path = "keypoints_runs/canonical"
    run = Group(
        {**identity, "keypoints_roi": keypoints, "pose_success": pose_success},
        path=run_path,
    )
    labels = ("swim_bladder", "eye_left", "eye_right")
    declarations = {
        name: {
            "shape": [int(value) for value in values.shape],
            "dtype": str(values.dtype),
            "digest_algorithm": "sha256_c_contiguous_bytes_v1",
            "sha256": digest(values),
        }
        for name, values in run.items()
    }
    manifest = {
        "payload_digest": "1" * 64,
        "payload": {
            "source_crop_snapshot": {"run_path": "crop_runs/canonical"},
            "logical_content": {"document": {"arrays": declarations}},
            "pose_model_schema_binding": {
                "pose_schema": {"keypoint_labels": list(labels)}
            },
        },
    }
    context = SimpleNamespace(
        run_path=run_path,
        source=SimpleNamespace(crop_path="crop_runs/canonical"),
        row_identity=SimpleNamespace(leading_dimension=2),
        temporal_authority=SimpleNamespace(
            record=SimpleNamespace(
                source_total_frames=30,
                recording_id="recording",
                camera_id="camera",
            )
        ),
        keypoint_labels=labels,
    )
    source = SimpleNamespace(
        run_path=run_path,
        run_group=run,
        manifest=manifest,
        surfaces=SimpleNamespace(context=context),
    )
    collection = {
        "schema_id": "palette.subject_mask.assignment_keypoint_collection",
        "schema_version": 1,
        "mode": "exact_worker_partition",
        "row_policy": "ordered_contiguous_recording_crop_rows_v1",
        "n_rois": 2,
        "workers": [
            {
                "global_row_interval": {"start_row": start, "stop_row": stop},
                "assignment": {
                    "assignment_keypoint_group": "keypoints_runs",
                    "assignment_keypoints_run": "canonical",
                    "assignment_keypoint_success_dataset": "pose_success",
                },
            }
            for start, stop in ((0, 1), (1, 2))
        ],
    }
    bundle = SimpleNamespace(
        archive_path=archive,
        bundle_id="bundle",
        authority_digest="2" * 64,
        assignment_keypoint_collection=collection,
        crop_run_path="crop_runs/canonical",
        n_rois=2,
        source_total_frames=30,
        recording_identity="recording",
        camera_identity="camera",
        bundle_manifest={
            "payload": {
                "cross_binding": {
                    "raw_refined_identity_array_values_sha256": {
                        name: digest(values) for name, values in identity.items()
                    }
                }
            }
        },
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "require_bound_recording_subject_mask_coordinate_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_keypoint_coordinate_successor_source",
        lambda _archive, *, run_path: (
            source
            if run_path == "keypoints_runs/canonical"
            else pytest.fail(f"unexpected keypoint path: {run_path}")
        ),
    )

    resolved = load_assignment_keypoint_source(
        archive,
        subject_mask_authority=bundle,
    )

    assert resolved.evidence_profile == ASSIGNMENT_KEYPOINT_SOURCE_DIRECT_PROFILE
    assert resolved.coordinate_source is source
    assert resolved.keypoint_run_path == "keypoints_runs/canonical"
    assert resolved.success_dataset == "pose_success"
    assert resolved.eye_keypoint_indices == {"eye_left": 1, "eye_right": 2}

    with pytest.raises(ValueError, match="cannot also select a rebinding"):
        load_assignment_keypoint_source(
            archive,
            subject_mask_authority=bundle,
            rebinding_run_id="ambiguous",
        )

    bundle.bundle_manifest["payload"]["cross_binding"][
        "raw_refined_identity_array_values_sha256"
    ]["instance_key"] = "0" * 64
    with pytest.raises(ValueError, match="instance_key differs"):
        load_assignment_keypoint_source(
            archive,
            subject_mask_authority=bundle,
        )


def test_historical_assignment_source_resolves_only_through_exact_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = (tmp_path / "analysis.zarr").resolve()
    archive.mkdir()
    keypoints = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
    pose_success = np.asarray([True], dtype=np.bool_)

    def digest(values: np.ndarray) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(values).tobytes(order="C")
        ).hexdigest()

    run_path = "keypoints_runs/canonical"
    run = {"keypoints_roi": keypoints, "pose_success": pose_success}
    source = SimpleNamespace(
        run_path=run_path,
        run_group=run,
        manifest={"payload_digest": "1" * 64},
        manifest_digest="2" * 64,
        successor_authority_digest="3" * 64,
        active_keypoint_bundle_authority_digest="4" * 64,
    )
    collection = {
        "schema_id": "palette.subject_mask.assignment_keypoint_collection",
        "schema_version": 1,
        "mode": "exact_worker_partition",
        "row_policy": "ordered_contiguous_recording_crop_rows_v1",
        "n_rois": 1,
        "workers": [
            {
                "global_row_interval": {"start_row": 0, "stop_row": 1},
                "assignment": {
                    "assignment_keypoint_group": "keypoints_runs",
                    "assignment_keypoints_run": "historical",
                    "assignment_keypoint_success_dataset": "detection_success",
                },
            }
        ],
    }
    bundle = SimpleNamespace(
        archive_path=archive,
        bundle_id="bundle",
        authority_digest="5" * 64,
        assignment_keypoint_collection=collection,
        n_rois=1,
    )
    labels = ("swim_bladder", "eye_left", "eye_right")
    arrays = {
        "keypoints_roi": {"sha256": digest(keypoints)},
        "pose_success": {"sha256": digest(pose_success)},
    }
    payload = {
        "subject_mask_source": {
            "historical_keypoint_run_path": "keypoints_runs/historical"
        },
        "canonical_keypoint_source": {
            "authority_profile": ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
            "run_path": run_path,
            "run_manifest_payload_digest": "1" * 64,
            "run_manifest_document_digest": "2" * 64,
            "coordinate_successor_authority_digest": "3" * 64,
            "keypoint_bundle_authority_digest": "4" * 64,
            "keypoints_dataset": "keypoints_roi",
            "success_dataset": "pose_success",
            "keypoint_labels": list(labels),
            "eye_keypoint_indices": {"eye_left": 1, "eye_right": 2},
        },
        "equivalence": {
            "keypoints_roi_to_keypoints_roi": {
                "digest_algorithm": "sha256_c_contiguous_bytes_v1",
                "normalized_sha256": digest(keypoints),
            },
            "detection_success_to_pose_success": {
                "digest_algorithm": "sha256_c_contiguous_bytes_v1",
                "normalized_sha256": digest(pose_success),
            },
        },
    }
    rebinding = {"payload": payload}
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "require_bound_recording_subject_mask_coordinate_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_assignment_keypoint_rebinding_manifest",
        lambda *_args, **_kwargs: rebinding,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_keypoint_coordinate_successor_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "_canonical_coordinate_successor_binding",
        lambda *_args, **_kwargs: (
            labels,
            {"eye_left": 1, "eye_right": 2},
            arrays,
        ),
    )

    resolved = load_assignment_keypoint_source(
        archive,
        subject_mask_authority=bundle,
        rebinding_run_id="rebind",
        expected_rebinding_manifest=rebinding,
    )

    assert resolved.evidence_profile == ASSIGNMENT_KEYPOINT_SOURCE_REBINDING_PROFILE
    assert resolved.coordinate_source is source
    assert resolved.rebinding_run_id == "rebind"

    with pytest.raises(ValueError, match="Attached assignment rebinding differs"):
        load_assignment_keypoint_source(
            archive,
            subject_mask_authority=bundle,
            rebinding_run_id="rebind",
            expected_rebinding_manifest={"payload": {"changed": True}},
        )


@pytest.mark.parametrize(
    ("historical_group", "historical_success_dataset", "selector_eligible"),
    (
        ("keypoints_runs", "detection_success", True),
        ("refined_keypoints_runs", "usable_keypoints", False),
    ),
)
def test_inspection_uses_resolver_digests_for_immutable_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    historical_group: str,
    historical_success_dataset: str,
    selector_eligible: bool,
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()

    def digest(values: np.ndarray) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(values).tobytes(order="C")
        ).hexdigest()

    identity = {
        "source_crop_row_ids": np.asarray([0, 1], dtype=np.int64),
        "instance_key": np.asarray([10, 11], dtype=np.uint64),
        "source_acquisition_frame_index": np.asarray([20, 21], dtype=np.int64),
    }
    keypoints = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    pose_success = np.asarray([True, False], dtype=np.bool_)
    canonical_arrays = {
        **identity,
        "keypoints_roi": keypoints,
        "pose_success": pose_success,
    }
    historical_arrays = {
        **identity,
        "keypoints_roi": keypoints.astype(np.float64),
        historical_success_dataset: pose_success,
    }

    class Group(dict[str, object]):
        def __init__(
            self,
            values: dict[str, object],
            *,
            path: str,
            attrs: dict[str, object] | None = None,
        ) -> None:
            super().__init__(values)
            self.path = path
            self.attrs = attrs or {}

    labels = ["swim_bladder", "eye_left", "eye_right"]
    historical = Group(
        historical_arrays,
        path=f"{historical_group}/historical",
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": selector_eligible,
            "keypoint_labels": labels,
        },
    )
    canonical = Group(
        canonical_arrays,
        path="keypoints_runs/canonical",
    )
    array_declarations = {
        name: {
            "digest_algorithm": "sha256_c_contiguous_bytes_v1",
            "sha256": digest(values),
        }
        for name, values in canonical_arrays.items()
    }
    manifest_document = {
        "payload_digest": "a" * 64,
        "payload": MappingProxyType(
            {
                "logical_content": MappingProxyType(
                    {
                        "document": MappingProxyType(
                            {"arrays": MappingProxyType(array_declarations)}
                        )
                    }
                ),
                "source_crop_snapshot": MappingProxyType(
                    {"run_path": "crop_runs/canonical"}
                ),
                "pose_model_schema_binding": MappingProxyType(
                    {
                        "pose_schema": MappingProxyType(
                            {"keypoint_labels": tuple(labels)}
                        )
                    }
                ),
            }
        ),
    }
    manifest_digest = "b" * 64
    authority_digest = "c" * 64
    successor_digest = "d" * 64
    source = SimpleNamespace(
        run_group=canonical,
        manifest=MappingProxyType(manifest_document),
        manifest_digest=manifest_digest,
        active_keypoint_bundle_authority=MappingProxyType({"generation": 7}),
        active_keypoint_bundle_authority_digest=authority_digest,
        successor_authority_digest=successor_digest,
    )
    collection = MappingProxyType(
        {
            "schema_id": "palette.subject_mask.assignment_keypoint_collection",
            "schema_version": 1,
            "mode": "exact_worker_partition",
            "row_policy": "ordered_contiguous_recording_crop_rows_v1",
            "n_rois": 2,
            "workers": [
                {
                    "global_row_interval": {"start_row": 0, "stop_row": 2},
                    "assignment": {
                        "assignment_keypoint_group": historical_group,
                        "assignment_keypoints_run": "historical",
                        "assignment_keypoint_success_dataset": (
                            historical_success_dataset
                        ),
                    },
                }
            ],
        }
    )
    bundle = SimpleNamespace(
        assignment_keypoint_collection=collection,
        crop_run_path="crop_runs/canonical",
        bundle_manifest={
            "payload_digest": "e" * 64,
            "payload": {
                "cross_binding": {
                    "raw_refined_identity_array_values_sha256": {
                        name: digest(values) for name, values in identity.items()
                    }
                }
            },
        },
        recording_identity="recording",
        camera_identity="camera",
        n_rois=2,
        bundle_id="bundle",
        authority_digest="f" * 64,
        refined_run_path="refined_subject_masks_runs/refined",
    )

    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_recording_subject_mask_coordinate_authority",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_keypoint_coordinate_successor_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding.open_zarr_root",
        lambda *_args, **_kwargs: {
            f"{historical_group}/historical": historical
        },
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "_refined_historical_labels",
        lambda *_args, **_kwargs: labels,
    )

    result = inspect_assignment_keypoint_rebinding(
        analysis_zarr=archive,
        subject_mask_bundle_id="bundle",
        keypoint_run_id="canonical",
        rebinding_run_id="rebind_001",
        block_rows=1,
    )

    keypoint_source = result["payload"]["canonical_keypoint_source"]
    assert keypoint_source["run_manifest_document_digest"] == manifest_digest
    assert keypoint_source["keypoint_bundle_authority_digest"] == authority_digest
    assert keypoint_source["coordinate_successor_authority_digest"] == successor_digest
    subject_source = result["payload"]["subject_mask_source"]
    assert subject_source["historical_keypoint_run_path"] == (
        f"{historical_group}/historical"
    )
    assert (
        f"{historical_success_dataset}_to_pose_success"
        in result["payload"]["equivalence"]
    )
    json.dumps(result, allow_nan=False)
