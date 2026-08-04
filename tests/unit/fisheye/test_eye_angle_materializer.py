from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis.eye_angle_storage import (
    EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
)
from fisheye.analysis_workflows.materializers import atomic_run_publisher as atomic_mod
from fisheye.analysis_workflows.materializers import eye_angles as mod
from fisheye.shared import eye_geometry_source as eye_geometry_source_mod
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    COORDINATE_DESCRIPTOR_ATTR,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    build_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_frame_record import (
    array_payload_sha256,
    array_values_sha256,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
)
from fisheye.shared.coordinate_record import coordinate_record_sha256
from fisheye.shared.keypoint_coordinate_publication import (
    KEYPOINT_LABEL_AUTHORITY_ATTR,
    KEYPOINT_LABEL_AUTHORITY_SCHEMA_ID,
    KEYPOINT_LABEL_AUTHORITY_SCHEMA_VERSION,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_MANIFEST_ATTR,
    SUBJECT_SHAPE_SCALAR_SURFACE_ATTR,
)


_REAL_SUBJECT_SHAPE_PUBLICATION_LOADER = (
    eye_geometry_source_mod.load_persisted_subject_shape_coordinate_publication
)
_SHAPE_RUN_PATH = "analysis/subject_shape_runs/shape_1"
_KEYPOINT_RUN_PATH = "keypoints_runs/kp_raw_1"
_KEYPOINT_LABELS = ("swim_bladder", "eye_left", "eye_right")
_SHAPE_ARRAY_PATHS = (
    "components/eye_left/ellipse_params",
    "components/eye_left/ellipse_success",
    "components/eye_right/ellipse_params",
    "components/eye_right/ellipse_success",
    "relations/eye_pair/separation_px",
)


@pytest.fixture(autouse=True)
def _disable_registry_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "emit_eye_angle_stage_completion",
        lambda *args, **kwargs: False,
    )


def _fixture_keypoint_label_authority_record(
    *,
    row_identity_ref: str,
    row_identity_sha256: str,
    rows: int,
    labels: tuple[str, ...] = _KEYPOINT_LABELS,
) -> dict[str, object]:
    return {
        "schema_id": KEYPOINT_LABEL_AUTHORITY_SCHEMA_ID,
        "schema_version": KEYPOINT_LABEL_AUTHORITY_SCHEMA_VERSION,
        "axis0": {
            "role": "observation_instance",
            "row_identity_ref": row_identity_ref,
            "row_identity_sha256": row_identity_sha256,
        },
        "axis1": {
            "role": "keypoint",
            "cardinality": len(labels),
            "labels": list(labels),
        },
        "coordinate_component_axis": {
            "axis": 2,
            "components": ["x", "y"],
        },
        "arrays": {
            "keypoints_roi": {
                "array_ref": f"/{_KEYPOINT_RUN_PATH}/keypoints_roi",
                "shape": [int(rows), len(labels), 2],
                "dtype": np.dtype(np.float32).str,
                "keypoint_axis": 1,
            },
        },
    }


def _build_source(
    path: Path,
    *,
    rows: int = 4,
    keypoint_labels: tuple[str, ...] = _KEYPOINT_LABELS,
) -> None:
    if len(keypoint_labels) != 3:
        raise ValueError("Synthetic eye-angle fixtures require three keypoint labels.")
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "eye-angle-materializer-fixture"
    analysis = root.create_group("analysis")
    shape_parent = analysis.create_group("subject_shape_runs")
    shape_parent.attrs["latest"] = "shape_1"
    shape = shape_parent.create_group("shape_1")
    shape.attrs.update(
        {
            "schema_id": "analysis.subject_shape_runs",
            "schema_version": 3,
            "method": "subject_shape",
            "method_version": "subject_shape.v3",
            "palette_run_completion_status": "complete",
            "source_refined_subject_masks_run": "refined_masks_1",
            "source_keypoints_run": "kp_raw_1",
            "source_fingerprint": "shape-fixture-fingerprint",
        }
    )
    for component, center_y, angle in (
        ("eye_left", 1.0, 90.0),
        ("eye_right", -1.0, 90.0),
    ):
        group = shape.create_group(f"components/{component}")
        group.attrs.update(
            {
                "ellipse_method": "cv2.fitEllipse_component_contour_v1",
                "geometry_schema_id": "subject_shape.eye_ellipse.v1",
            }
        )
        ellipse = np.tile(
            np.asarray([1.0, center_y, 4.0, 2.0, angle], dtype=np.float32),
            (rows, 1),
        )
        group.create_array("ellipse_params", data=ellipse, chunks=(2, 5))
        group.create_array(
            "ellipse_success",
            data=np.ones(rows, dtype=bool),
            chunks=(2,),
        )
    pair = shape.create_group("relations/eye_pair")
    pair.create_array(
        "separation_px",
        data=np.full(rows, 2.0, dtype=np.float32),
        chunks=(2,),
    )

    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "kp_refined_1"
    refined = refined_parent.create_group("kp_refined_1")
    refined.attrs.update(
        {
            "schema_id": "refined_keypoints",
            "schema_version": 4,
            "method": "manual_plus_model_refinement",
            "method_version": "refined_keypoints.v4",
            "palette_run_completion_status": "complete",
            "source_keypoints_run": "kp_raw_1",
            "source_lineage_hash": "refined-keypoint-lineage",
            "keypoint_labels": list(keypoint_labels),
            "stage_selector_eligible": False,
            "coordinate_contract": (
                "palette.refined_keypoints.legacy_unverified_nonselector.v1"
            ),
            "legacy_unverified_diagnostic_output": True,
            "publication_scope": "historical_diagnostic_only",
        }
    )
    keypoints = np.tile(
        np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
            dtype=np.float32,
        ),
        (rows, 1, 1),
    )
    refined.create_array("keypoints_roi", data=keypoints, chunks=(2, 3, 2))
    refined.create_array(
        "heading",
        data=np.zeros(rows, dtype=np.float32),
        chunks=(2,),
    )
    refined.create_array(
        "refined_success",
        data=np.ones(rows, dtype=bool),
        chunks=(2,),
    )
    refined.create_array(
        "instance_key",
        data=np.arange(1, rows + 1, dtype=np.uint64),
        chunks=(2,),
    )
    refined.create_array(
        "frame_indices",
        data=np.arange(rows, dtype=np.int64),
        chunks=(2,),
    )

    raw_parent = root.create_group("keypoints_runs")
    raw_parent.attrs["latest"] = "kp_raw_1"
    raw = raw_parent.create_group("kp_raw_1")
    raw.attrs.update(
        {
            "schema_id": "keypoints",
            "schema_version": 2,
            "method": "yolo_pose",
            "method_version": "detector.v2",
            "palette_run_completion_status": "complete",
            "lineage_hash": "raw-keypoint-lineage",
            "keypoint_labels": list(keypoint_labels),
        }
    )
    raw.create_array("keypoints_roi", data=keypoints, chunks=(2, 3, 2))
    raw.create_array(
        "detection_success",
        data=np.ones(rows, dtype=bool),
        chunks=(2,),
    )
    raw.create_array(
        "instance_key",
        data=np.arange(1, rows + 1, dtype=np.uint64),
        chunks=(2,),
    )
    raw.create_array(
        "source_acquisition_frame_index",
        data=np.arange(rows, dtype=np.int64),
        chunks=(2,),
    )
    raw_identity = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=np.arange(1, rows + 1, dtype=np.uint64),
    )
    label_authority = _fixture_keypoint_label_authority_record(
        row_identity_ref=f"/{_KEYPOINT_RUN_PATH}@row_identity_contract",
        row_identity_sha256=raw_identity.digest(),
        rows=rows,
        labels=keypoint_labels,
    )
    raw.attrs.update(
        {
            KEYPOINT_LABEL_AUTHORITY_ATTR: label_authority,
            f"{KEYPOINT_LABEL_AUTHORITY_ATTR}_sha256": (
                coordinate_record_sha256(label_authority)
            ),
        }
    )


def _fixture_ellipse_descriptor(*, rows: int):
    identity = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=np.arange(1, rows + 1, dtype=np.uint64),
    )
    frame_ref = "/coordinate_frames/source_camera@pixel_frame_authority"
    frame_sha256 = "1" * 64
    return build_canonical_coordinate_descriptor(
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="ellipse_cxcy_wh_angle",
        components=("center_x", "center_y", "width", "height", "angle"),
        component_units=("px", "px", "px", "px", "deg"),
        reference_width=100,
        reference_height=80,
        reference_authority=DigestBoundCoordinateRecordRef(
            record_ref=frame_ref,
            record_sha256=frame_sha256,
        ),
        reference_selector="record",
        pixel_convention="continuous",
        row_identity_contract=identity,
        row_identity_record_ref=f"/{_SHAPE_RUN_PATH}@row_identity_contract",
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame_ref,
            record_sha256=frame_sha256,
        ),
    )


def _fixture_bound_row_identity(
    values: np.ndarray,
    *,
    record_ref: str,
) -> SimpleNamespace:
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=np.asarray(values, dtype=np.uint64),
    )
    return SimpleNamespace(
        contract=contract,
        leading_dimension=contract.leading_dimension,
        record_ref=record_ref,
        record_sha256=contract.digest(),
    )


def _fixture_temporal_authority(
    frame_indices: np.ndarray,
    *,
    row_identity: SimpleNamespace,
    record_ref: str,
) -> SimpleNamespace:
    values = np.asarray(frame_indices, dtype=np.int64)
    frame_record = SimpleNamespace(
        dtype=np.dtype(values.dtype).str,
        shape=tuple(int(value) for value in values.shape),
        content_sha256=array_values_sha256(values),
    )
    record = SimpleNamespace(
        recording_id="eye-angle-materializer-fixture",
        camera_id="camera_0",
        source_total_frames=(
            max(1, int(values.max()) + 1) if values.size else 1
        ),
        source_identity_domain=row_identity.contract.domain,
        source_identity_mode=row_identity.contract.mode,
        source_leading_dimension=row_identity.contract.leading_dimension,
        source_acquisition_frame_index=frame_record,
    )
    digest_payload = {
        "recording_id": record.recording_id,
        "camera_id": record.camera_id,
        "source_total_frames": record.source_total_frames,
        "source_identity_domain": record.source_identity_domain,
        "source_identity_mode": record.source_identity_mode,
        "source_leading_dimension": record.source_leading_dimension,
        "frame_index_dtype": frame_record.dtype,
        "frame_index_shape": list(frame_record.shape),
        "frame_index_content_sha256": frame_record.content_sha256,
    }
    return SimpleNamespace(
        record=record,
        record_ref=record_ref,
        record_sha256=coordinate_record_sha256(digest_payload),
    )


def _fixture_bound_record(*, record_ref: str, seed: str) -> SimpleNamespace:
    return SimpleNamespace(
        record_ref=record_ref,
        record_sha256=hashlib.sha256(seed.encode("utf-8")).hexdigest(),
    )


def _fake_keypoint_coordinate_surfaces(
    root: zarr.Group,
    path: str,
) -> SimpleNamespace:
    assert path == _KEYPOINT_RUN_PATH
    group = root[path]
    instance_key = np.asarray(group["instance_key"][:], dtype=np.uint64)
    frame_indices = np.asarray(
        group["source_acquisition_frame_index"][:],
        dtype=np.int64,
    )
    row_identity = _fixture_bound_row_identity(
        instance_key,
        record_ref=f"/{path}@row_identity_contract",
    )
    temporal_authority = _fixture_temporal_authority(
        frame_indices,
        row_identity=row_identity,
        record_ref=f"/{path}@source_row_temporal_authority",
    )
    descriptor_sha256 = hashlib.sha256(
        (
            "fixture-keypoints-roi-descriptor:"
            + row_identity.record_sha256
        ).encode("utf-8")
    ).hexdigest()
    label_authority_record = dict(group.attrs[KEYPOINT_LABEL_AUTHORITY_ATTR])
    label_authority_sha256 = str(
        group.attrs[f"{KEYPOINT_LABEL_AUTHORITY_ATTR}_sha256"]
    )
    exact_labels = tuple(group.attrs["keypoint_labels"])
    assert label_authority_sha256 == coordinate_record_sha256(
        label_authority_record
    )
    assert tuple(label_authority_record["axis1"]["labels"]) == exact_labels
    return SimpleNamespace(
        context=SimpleNamespace(
            run_path=path,
            _run_group=group,
            row_identity=row_identity,
            temporal_authority=temporal_authority,
            keypoint_labels=exact_labels,
            context_record=_fixture_bound_record(
                record_ref=f"/{path}@coordinate_context",
                seed=f"{path}:coordinate-context:{row_identity.record_sha256}",
            ),
            keypoint_label_authority=SimpleNamespace(
                record=label_authority_record,
                record_ref=f"/{path}@keypoint_label_authority",
                record_sha256=label_authority_sha256,
            ),
        ),
        derivation=_fixture_bound_record(
            record_ref=f"/{path}@coordinate_derivation",
            seed=f"{path}:coordinate-derivation:{descriptor_sha256}",
        ),
        keypoints_roi=SimpleNamespace(
            coordinate_node=group["keypoints_roi"],
            descriptor=SimpleNamespace(
                digest=lambda digest=descriptor_sha256: digest,
            ),
        ),
    )


def _fake_assignment_keypoint_authority(
    surfaces: SimpleNamespace,
) -> SimpleNamespace:
    group = surfaces.context._run_group
    record = {
        "schema_id": "palette.refined_subject_mask_assignment_keypoint_authority",
        "schema_version": 1,
        "status": "used",
        "selection_policy": "exact_full_raw_keypoint_rowset_no_fallback_v1",
        "keypoint_run_path": surfaces.context.run_path,
        "keypoint_labels": list(surfaces.context.keypoint_labels),
        "keypoints_roi": {
            "payload": {
                "array_values_sha256": array_values_sha256(
                    group["keypoints_roi"]
                ),
            },
        },
        "success": {
            "dataset": "detection_success",
            "payload": {
                "array_values_sha256": array_values_sha256(
                    group["detection_success"]
                ),
            },
        },
        "row_identity": {
            "record_ref": surfaces.context.row_identity.record_ref,
            "record_sha256": surfaces.context.row_identity.record_sha256,
        },
    }
    return SimpleNamespace(
        record=record,
        record_ref=(
            "/refined_subject_masks_runs/refined_masks_1"
            "@refined_subject_mask_assignment_keypoint_authority"
        ),
        record_sha256=coordinate_record_sha256(record),
    )


def _fake_coordinate_publication(
    root: zarr.Group,
    shape: zarr.Group,
    path: str,
    *,
    stamp_metadata: bool = False,
):
    assert path == _SHAPE_RUN_PATH
    rows = int(shape["components/eye_left/ellipse_params"].shape[0])
    descriptor = _fixture_ellipse_descriptor(rows=rows)
    subject_instance_key = np.arange(1, rows + 1, dtype=np.uint64)
    subject_frame_indices = np.arange(rows, dtype=np.int64)
    row_identity = _fixture_bound_row_identity(
        subject_instance_key,
        record_ref=f"/{path}@row_identity_contract",
    )
    temporal_authority = _fixture_temporal_authority(
        subject_frame_indices,
        row_identity=row_identity,
        record_ref=f"/{path}@source_row_temporal_authority",
    )
    assignment_surfaces = _fake_keypoint_coordinate_surfaces(
        root,
        _KEYPOINT_RUN_PATH,
    )
    assignment_authority = _fake_assignment_keypoint_authority(
        assignment_surfaces
    )
    descriptors: dict[str, object] = {}
    arrays: dict[str, dict[str, object]] = {}
    for relative_ref in _SHAPE_ARRAY_PATHS:
        node = shape[relative_ref]
        arrays[relative_ref] = {
            "array_ref": f"/{path}/{relative_ref}",
            "relative_ref": relative_ref,
            "dtype": np.dtype(node.dtype).str,
            "shape": [int(value) for value in node.shape],
            "content_sha256": array_payload_sha256(node),
            "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        }
        if relative_ref.endswith("ellipse_params"):
            if stamp_metadata:
                node.attrs.update(descriptor.to_attrs())
                node.attrs[f"{COORDINATE_DESCRIPTOR_ATTR}_owner_dtype"] = (
                    np.dtype(node.dtype).str
                )
            descriptors[relative_ref] = SimpleNamespace(
                coordinate_node=node,
                descriptor=descriptor,
            )

    separation = shape["relations/eye_pair/separation_px"]
    separation_record = {
        "schema_id": "palette.subject_shape_scalar_surface",
        "schema_version": 1,
        "relative_ref": "relations/eye_pair/separation_px",
        "quantity": "eye_centroid_separation",
        "units": "px",
        "surface_kind": "row_scalar",
        "row_identity": {
            "record_ref": f"/{path}@row_identity_contract",
            "record_sha256": descriptor.row_identity.record_sha256,
        },
    }
    separation_sha256 = coordinate_record_sha256(separation_record)
    if stamp_metadata:
        separation.attrs[SUBJECT_SHAPE_SCALAR_SURFACE_ATTR] = separation_record
        separation.attrs[f"{SUBJECT_SHAPE_SCALAR_SURFACE_ATTR}_sha256"] = (
            separation_sha256
        )

    manifest_record = {
        "schema_id": "palette.subject_shape_coordinate_publication_manifest",
        "schema_version": 1,
        "run_ref": f"/{path}",
        "arrays": arrays,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest_record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if stamp_metadata:
        shape.attrs.update(
            {
                "stage_selector_eligible": True,
                "coordinate_contract": "canonical_v2",
                "coordinate_binding_status": "bound_canonical_v2",
                "subject_shape_publication_owner_uuid": "f" * 32,
                "publication_manifest_sha256": manifest_sha256,
                "component_names": ["eye_left", "eye_right"],
                "relation_names": ["eye_pair"],
            }
        )
    semantics = SimpleNamespace(
        record_ref=(
            f"/{path}/relations/eye_pair/separation_px"
            f"@{SUBJECT_SHAPE_SCALAR_SURFACE_ATTR}"
        ),
        record_sha256=separation_sha256,
    )

    def require_scalar_surface(relative_ref, *, units=None, surface_kind=None):
        assert relative_ref == "relations/eye_pair/separation_px"
        assert units == "px"
        assert surface_kind == "row_scalar"
        return SimpleNamespace(array_node=separation, semantics=semantics)

    return SimpleNamespace(
        run_path=path,
        manifest=SimpleNamespace(
            record=manifest_record,
            record_ref=f"/{path}@{SUBJECT_SHAPE_MANIFEST_ATTR}",
            record_sha256=manifest_sha256,
        ),
        source=SimpleNamespace(
            context=SimpleNamespace(
                assignment_keypoint_surfaces=assignment_surfaces,
                assignment_keypoint_authority=assignment_authority,
            ),
        ),
        row_identity=row_identity,
        temporal_authority=temporal_authority,
        descriptors=descriptors,
        require_scalar_surface=require_scalar_surface,
    )


def _accept_synthetic_subject_shape_publication(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
) -> None:
    root = zarr.open_group(str(source), mode="a", use_consolidated=False)
    _fake_coordinate_publication(
        root,
        root[_SHAPE_RUN_PATH],
        _SHAPE_RUN_PATH,
        stamp_metadata=True,
    )

    def _load(root: zarr.Group, path: str) -> object:
        return _fake_coordinate_publication(root, root[path], path)

    monkeypatch.setattr(
        eye_geometry_source_mod,
        "load_persisted_subject_shape_coordinate_publication",
        _load,
    )
    monkeypatch.setattr(
        mod.eye_writer,
        "load_persisted_keypoint_coordinate_surfaces",
        _fake_keypoint_coordinate_surfaces,
    )


def _stage_synthetic_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    source_name: str = "source.zarr",
    scratch_name: str = "scratch",
    run_name: str = "eye_1",
    chunk_rows: int = 2,
    keypoint_labels: tuple[str, ...] = _KEYPOINT_LABELS,
):
    source = tmp_path / source_name
    scratch = tmp_path / scratch_name
    _build_source(source, keypoint_labels=keypoint_labels)
    _accept_synthetic_subject_shape_publication(monkeypatch, source)
    plan = mod.build_eye_angle_materialization_plan(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name=run_name,
        chunk_rows=chunk_rows,
        fps=100.0,
    )
    staging = mod.stage_eye_angle_sources(
        plan,
        copy_backend="python",
        check_capacity=False,
    )
    return source, plan, staging


def _nested_subject_shape_authority(
    receipt: dict[str, object],
) -> dict[str, object]:
    return mod.eye_writer._staged_subject_shape_authority_from_input_receipt(
        receipt
    )


def _nested_keypoint_authority(
    receipt: dict[str, object],
) -> dict[str, object]:
    return mod.eye_writer._staged_keypoint_authority_from_input_receipt(
        receipt
    )


def _resign_integrity_record(record: dict[str, object]) -> dict[str, object]:
    body = copy.deepcopy(record)
    body.pop("record_sha256", None)
    return {
        **body,
        "record_sha256": mod.eye_writer._canonical_json_sha256(body),
    }


def _resign_receipt_with_keypoint_authority(
    plan: mod.EyeAngleMaterializationPlan,
    receipt: dict[str, object],
    authority: dict[str, object],
) -> dict[str, object]:
    updated = copy.deepcopy(receipt)
    updated["canonical_keypoint_authority"] = copy.deepcopy(authority)
    updated["canonical_keypoint_authority_sha256"] = authority[
        "record_sha256"
    ]
    source_contracts = copy.deepcopy(plan.source_contracts)
    source_contracts["keypoints"]["canonical_keypoint_authority"] = (
        copy.deepcopy(authority)
    )
    updated["source_contract_sha256"] = (
        mod.eye_writer._canonical_json_sha256(source_contracts)
    )
    return _resign_integrity_record(updated)


def _resolve_staged_context_from_receipt(
    plan: mod.EyeAngleMaterializationPlan,
    receipt: dict[str, object],
):
    canonical = mod.eye_writer._canonical_staged_input_integrity_receipt(
        receipt
    )
    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="r",
        use_consolidated=False,
    )
    return mod.eye_writer._resolve_eye_angle_inputs(
        staged_root,
        subject_shape_run="shape_1",
        refined_subject_run=None,
        keypoint_run="kp_raw_1",
        _staged_subject_shape_authority=(
            mod.eye_writer._staged_subject_shape_authority_from_input_receipt(
                canonical
            )
        ),
        _staged_keypoint_authority=(
            mod.eye_writer._staged_keypoint_authority_from_input_receipt(
                canonical
            )
        ),
        _verify_staged_payload=False,
    )


def test_plan_rejects_unsealed_subject_shape_before_scratch_creation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)

    with pytest.raises(ValueError, match="not a canonical publication"):
        mod.materialize_eye_angles(
            source,
            scratch_root=scratch,
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_1",
            fps=100.0,
            apply=False,
        )

    assert not scratch.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "eye_angle_runs" not in root["analysis"]


@pytest.mark.parametrize(
    ("relation", "through_symlink"),
    (
        pytest.param("equal", False, id="equal-direct"),
        pytest.param("equal", True, id="equal-symlink"),
        pytest.param("scratch_inside_source", False, id="scratch-child-direct"),
        pytest.param("scratch_inside_source", True, id="scratch-child-symlink"),
        pytest.param("source_inside_scratch", False, id="source-child-direct"),
        pytest.param("source_inside_scratch", True, id="source-child-symlink"),
    ),
)
def test_plan_rejects_all_resolved_source_scratch_overlap_before_open(
    tmp_path: Path,
    relation: str,
    through_symlink: bool,
) -> None:
    if relation == "equal":
        source = tmp_path / "source.zarr"
        source.mkdir()
        scratch_target = source
    elif relation == "scratch_inside_source":
        source = tmp_path / "source.zarr"
        scratch_target = source / "scratch"
        scratch_target.mkdir(parents=True)
    else:
        scratch_target = tmp_path / "scratch"
        source = scratch_target / "source.zarr"
        source.mkdir(parents=True)

    scratch_argument = scratch_target
    if through_symlink:
        scratch_argument = tmp_path / "scratch-alias"
        try:
            scratch_argument.symlink_to(scratch_target, target_is_directory=True)
        except OSError as exc:  # pragma: no cover - platform policy
            pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(
        ValueError,
        match="must be disjoint after resolving symlinks",
    ):
        mod.build_eye_angle_materialization_plan(
            source,
            scratch_root=scratch_argument,
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_1",
            fps=100.0,
        )


def test_plan_rejects_reordered_same_count_instance_keys_before_scratch_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    root = zarr.open_group(str(source), mode="a", use_consolidated=False)
    root["keypoints_runs/kp_raw_1/instance_key"][:] = np.asarray(
        [2, 1, 3, 4],
        dtype=np.uint64,
    )
    _accept_synthetic_subject_shape_publication(monkeypatch, source)

    with pytest.raises(ValueError, match="exact ordered instance_key identity"):
        mod.materialize_eye_angles(
            source,
            scratch_root=scratch,
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_1",
            fps=100.0,
            apply=True,
            copy_backend="python",
            check_capacity=False,
        )

    assert not scratch.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "eye_angle_runs" not in root["analysis"]


def test_plan_rejects_refined_keypoint_assertion_before_scratch_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _accept_synthetic_subject_shape_publication(monkeypatch, source)

    with pytest.raises(
        ValueError,
        match="--keypoint-run differs from the exact base keypoint run sealed",
    ):
        mod.materialize_eye_angles(
            source,
            scratch_root=scratch,
            subject_shape_run="shape_1",
            # This argument is a canonical-base assertion, not a legacy-group
            # selector, even though a refined run with this name exists.
            keypoint_run="kp_refined_1",
            run_name="eye_1",
            fps=100.0,
            apply=True,
            copy_backend="python",
            check_capacity=False,
        )

    assert not scratch.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "eye_angle_runs" not in root["analysis"]


def test_plan_is_read_only_and_selects_only_resolved_geometry_and_keypoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _accept_synthetic_subject_shape_publication(monkeypatch, source)

    result = mod.materialize_eye_angles(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run=None,
        run_name="eye_1",
        chunk_rows=2,
        fps=100.0,
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["mutates_archive"] is False
    assert not scratch.exists()
    plan = result["plan"]
    assert plan["row_count"] == 4
    assert plan["frame_count"] == 4
    assert plan["angle_chunk_rows"] == 4096
    assert plan["angle_chunk_columns"] == 16
    assert plan["output_shard_rows"] == 131072
    assert plan["angle_shard_columns"] == 32
    assert plan["fps_source"] == "cli_override"
    assert plan["keypoint_run"] == "kp_raw_1"
    assert plan["source_keypoint_run"] == "kp_raw_1"
    receipt = plan["staged_input_integrity_receipt"]
    assert receipt["schema_id"] == (
        mod.eye_writer.EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCHEMA_ID
    )
    assert receipt["integrity_scope"] == (
        mod.eye_writer.EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCOPE
    )
    assert receipt["closed_logical_input_inventory"] is True
    assert receipt["normal_reader_authority"] is False
    assert receipt["coordinate_authority"] is False
    assert receipt["scientific_parameters"] == {
        "fps": 100.0,
        "fps_source": "cli_override",
    }
    assert len(receipt["record_sha256"]) == 64
    assert plan["staged_input_integrity_receipt_sha256"] == (
        receipt["record_sha256"]
    )

    authority = _nested_subject_shape_authority(receipt)
    assert authority["schema_id"] == (
        eye_geometry_source_mod.EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_ID
    )
    assert authority["closed_array_inventory"] is True
    assert authority["normal_reader_authority"] is False
    assert len(authority["record_sha256"]) == 64
    assert set(authority["allowed_arrays"]) == set(_SHAPE_ARRAY_PATHS)
    assert receipt["subject_shape_authority_sha256"] == (
        authority["record_sha256"]
    )
    assert plan["source_contracts"]["eye_geometry"]["source_authority"] == (
        authority
    )
    keypoint_authority = _nested_keypoint_authority(receipt)
    assert keypoint_authority["schema_id"] == (
        mod.eye_writer.EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCHEMA_ID
    )
    assert keypoint_authority["keypoint_run_path"] == _KEYPOINT_RUN_PATH
    assert keypoint_authority["keypoint_labels"] == list(_KEYPOINT_LABELS)
    assert keypoint_authority["closed_array_inventory"] is True
    assert keypoint_authority["normal_reader_authority"] is False
    assert set(keypoint_authority["arrays"]) == {
        "keypoints_roi",
        "detection_success",
        "instance_key",
        "source_acquisition_frame_index",
    }
    assert keypoint_authority["ordered_row_alignment"]["policy"] == (
        "same_ordered_observation_instance_and_acquisition_time_v1"
    )
    assert receipt["canonical_keypoint_authority_sha256"] == (
        keypoint_authority["record_sha256"]
    )
    assert plan["source_contracts"]["keypoints"]["source_mode"] == (
        mod.eye_writer.EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
    )
    assert plan["source_contracts"]["keypoints"][
        "canonical_keypoint_authority"
    ] == keypoint_authority
    expected_source_refs = {
        "ellipse_params": [
            "analysis/subject_shape_runs/shape_1/components/eye_left/ellipse_params",
            "analysis/subject_shape_runs/shape_1/components/eye_right/ellipse_params",
        ],
        "ellipse_success": [
            "analysis/subject_shape_runs/shape_1/components/eye_left/ellipse_success",
            "analysis/subject_shape_runs/shape_1/components/eye_right/ellipse_success",
        ],
        "keypoints_roi": ["keypoints_runs/kp_raw_1/keypoints_roi"],
        "detection_success": [
            "keypoints_runs/kp_raw_1/detection_success"
        ],
        "instance_key": ["keypoints_runs/kp_raw_1/instance_key"],
        "source_acquisition_frame_index": [
            "keypoints_runs/kp_raw_1/source_acquisition_frame_index"
        ],
    }
    assert set(receipt["logical_inputs"]) == set(expected_source_refs)
    for role, refs in expected_source_refs.items():
        logical_input = receipt["logical_inputs"][role]
        assert logical_input["source_array_refs"] == refs
        assert logical_input["snapshot_shape"][0] == 4

    assert [
        (chunk["chunk_index"], chunk["start_row"], chunk["stop_row"])
        for chunk in receipt["chunks"]
    ] == [(0, 0, 2), (1, 2, 4)]
    for chunk in receipt["chunks"]:
        assert set(chunk["logical_inputs"]) == set(expected_source_refs)
        assert len(chunk["record_sha256"]) == 64
        for payload in chunk["logical_inputs"].values():
            assert payload["shape"][0] == (
                chunk["stop_row"] - chunk["start_row"]
            )
            assert len(payload["content_sha256"]) == 64
    assert all("masks_roi" not in path for path in plan["selected_arrays"])
    assert all(not path.endswith("/heading") for path in plan["selected_arrays"])
    assert plan["selected_arrays"] == sorted(
        [
            "analysis/subject_shape_runs/shape_1/components/eye_left/ellipse_params",
            "analysis/subject_shape_runs/shape_1/components/eye_left/ellipse_success",
            "analysis/subject_shape_runs/shape_1/components/eye_right/ellipse_params",
            "analysis/subject_shape_runs/shape_1/components/eye_right/ellipse_success",
            "analysis/subject_shape_runs/shape_1/relations/eye_pair/separation_px",
            "keypoints_runs/kp_raw_1/detection_success",
            "keypoints_runs/kp_raw_1/instance_key",
            "keypoints_runs/kp_raw_1/keypoints_roi",
            "keypoints_runs/kp_raw_1/source_acquisition_frame_index",
        ]
    )
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "eye_angle_runs" not in root["analysis"]


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is unavailable")
def test_rsync_staging_preserves_the_planned_physical_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _accept_synthetic_subject_shape_publication(monkeypatch, source)
    plan = mod.build_eye_angle_materialization_plan(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name="eye_1",
        fps=100.0,
    )

    staging = mod.stage_eye_angle_sources(
        plan,
        copy_backend="rsync",
        check_capacity=False,
    )

    assert staging["status"] == "complete"
    assert staging["inventory"]["valid"] is True
    assert staging["inventory"]["mtime_mismatches"] == []
    assert staging["source_revision_audit"]["status"] == "current"
    assert staging["source_authority_mode"] == "digest_bound_staged_subset"
    assert staging["staged_input_integrity_receipt_sha256"] == (
        plan.staged_input_integrity_receipt["record_sha256"]
    )
    assert staging["staged_input_integrity_receipt"] == (
        plan.staged_input_integrity_receipt
    )


def test_staged_receipt_is_private_and_normal_reader_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="r",
        use_consolidated=False,
    )
    authority = _nested_subject_shape_authority(
        plan.staged_input_integrity_receipt
    )

    staged = eye_geometry_source_mod.resolve_eye_geometry_source(
        staged_root,
        subject_shape_run="shape_1",
        _staged_subject_shape_authority=authority,
        _verify_staged_payload=True,
    )

    assert staged.source_authority_mode == "digest_bound_staged_subset"
    assert staged.subject_shape_coordinate_publication is None
    assert staged.source_authority == authority

    monkeypatch.setattr(
        eye_geometry_source_mod,
        "load_persisted_subject_shape_coordinate_publication",
        _REAL_SUBJECT_SHAPE_PUBLICATION_LOADER,
    )
    with pytest.raises(ValueError, match="not a canonical publication"):
        eye_geometry_source_mod.resolve_eye_geometry_source(
            staged_root,
            subject_shape_run="shape_1",
        )
    with pytest.raises(ValueError, match="private to digest-bound staged"):
        eye_geometry_source_mod.resolve_eye_geometry_source(
            staged_root,
            subject_shape_run="shape_1",
            _verify_staged_payload=False,
        )


def test_staged_context_accepts_exact_alias_labels_and_normalizes_anatomy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alias_labels = ("bladder", "left_eye", "right_eye")
    _source, plan, _staging = _stage_synthetic_source(
        monkeypatch,
        tmp_path,
        keypoint_labels=alias_labels,
    )
    receipt = plan.staged_input_integrity_receipt
    authority = receipt["canonical_keypoint_authority"]
    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="r",
        use_consolidated=False,
    )
    staged_group = staged_root[_KEYPOINT_RUN_PATH]

    assert tuple(staged_group.attrs["keypoint_labels"]) == alias_labels
    assert tuple(
        staged_group.attrs[KEYPOINT_LABEL_AUTHORITY_ATTR]["axis1"]["labels"]
    ) == alias_labels
    assert tuple(authority["keypoint_labels"]) == alias_labels
    assert tuple(receipt["keypoint_axis"]["resolved_labels"]) == alias_labels
    assert tuple(
        plan.source_contracts["keypoints"][
            "canonical_keypoint_authority"
        ]["keypoint_labels"]
    ) == alias_labels

    context = _resolve_staged_context_from_receipt(plan, receipt)

    assert context.keypoint_labels == alias_labels
    assert context.keypoint_indices == {
        "swim_bladder": 0,
        "eye_left": 1,
        "eye_right": 2,
    }


def test_staged_receipt_rejects_stale_digest_and_payload_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="a",
        use_consolidated=False,
    )
    authority = _nested_subject_shape_authority(
        plan.staged_input_integrity_receipt
    )
    stale = copy.deepcopy(authority)
    stale["row_count"] = int(stale["row_count"]) + 1
    with pytest.raises(ValueError, match="digest is missing or stale"):
        eye_geometry_source_mod.resolve_eye_geometry_source(
            staged_root,
            subject_shape_run="shape_1",
            _staged_subject_shape_authority=stale,
        )

    ellipse = staged_root[
        "analysis/subject_shape_runs/shape_1/"
        "components/eye_left/ellipse_params"
    ]
    changed = np.asarray(ellipse[:])
    changed[0, 0] += np.float32(5.0)
    ellipse[:] = changed

    lightweight = eye_geometry_source_mod.resolve_eye_geometry_source(
        staged_root,
        subject_shape_run="shape_1",
        _staged_subject_shape_authority=authority,
        _verify_staged_payload=False,
    )
    assert lightweight.source_authority_mode == "digest_bound_staged_subset"
    with pytest.raises(ValueError, match="differs from its canonical payload"):
        eye_geometry_source_mod.resolve_eye_geometry_source(
            staged_root,
            subject_shape_run="shape_1",
            _staged_subject_shape_authority=authority,
            _verify_staged_payload=True,
        )


def test_staged_receipt_rejects_another_source_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source_a, plan_a, _staging = _stage_synthetic_source(
        monkeypatch,
        tmp_path,
        source_name="source-a.zarr",
        scratch_name="scratch-a",
        run_name="eye_a",
    )
    source_b = tmp_path / "source-b.zarr"
    _build_source(source_b)
    root_b = zarr.open_group(str(source_b), mode="a", use_consolidated=False)
    ellipse_b = root_b[
        "analysis/subject_shape_runs/shape_1/"
        "components/eye_left/ellipse_params"
    ]
    changed_b = np.asarray(ellipse_b[:])
    changed_b[0, 0] += np.float32(9.0)
    ellipse_b[:] = changed_b
    _accept_synthetic_subject_shape_publication(monkeypatch, source_b)
    plan_b = mod.build_eye_angle_materialization_plan(
        source_b,
        scratch_root=tmp_path / "scratch-b",
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name="eye_b",
        fps=100.0,
    )
    staged_root_a = zarr.open_group(
        str(plan_a.staged_zarr),
        mode="r",
        use_consolidated=False,
    )

    with pytest.raises(
        ValueError,
        match="source contract attrs differ|canonical payload",
    ):
        eye_geometry_source_mod.resolve_eye_geometry_source(
            staged_root_a,
            subject_shape_run="shape_1",
            _staged_subject_shape_authority=(
                _nested_subject_shape_authority(
                    plan_b.staged_input_integrity_receipt
                )
            ),
            _verify_staged_payload=True,
        )


def test_combined_receipt_rejects_stale_root_chunk_and_chunk_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="r",
        use_consolidated=False,
    )
    authority = _nested_subject_shape_authority(
        plan.staged_input_integrity_receipt
    )
    keypoint_authority = _nested_keypoint_authority(
        plan.staged_input_integrity_receipt
    )
    context = mod.eye_writer._resolve_eye_angle_inputs(
        staged_root,
        subject_shape_run="shape_1",
        refined_subject_run=None,
        keypoint_run="kp_raw_1",
        _staged_subject_shape_authority=authority,
        _staged_keypoint_authority=keypoint_authority,
        _verify_staged_payload=False,
    )

    stale_root = copy.deepcopy(plan.staged_input_integrity_receipt)
    stale_root["row_count"] = int(stale_root["row_count"]) + 1
    with pytest.raises(ValueError, match="unsupported or stale"):
        mod.eye_writer._validate_staged_eye_angle_input_integrity_receipt(
            context,
            stale_root,
            verify_payload=False,
        )

    stale_chunk = copy.deepcopy(plan.staged_input_integrity_receipt)
    stale_chunk["chunks"][0]["logical_inputs"]["keypoints_roi"][
        "content_sha256"
    ] = "0" * 64
    stale_chunk = _resign_integrity_record(stale_chunk)
    with pytest.raises(ValueError, match="chunk receipt is unsupported or stale"):
        mod.eye_writer._validate_staged_eye_angle_input_integrity_receipt(
            context,
            stale_chunk,
            verify_payload=False,
        )

    gap = copy.deepcopy(plan.staged_input_integrity_receipt)
    gap_chunk = gap["chunks"][1]
    gap_chunk["start_row"] = 3
    for payload in gap_chunk["logical_inputs"].values():
        payload["shape"][0] = 1
    gap["chunks"][1] = _resign_integrity_record(gap_chunk)
    gap = _resign_integrity_record(gap)
    with pytest.raises(ValueError, match="gap, overlap, or wrong order"):
        mod.eye_writer._validate_staged_eye_angle_input_integrity_receipt(
            context,
            gap,
            verify_payload=False,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "top_level_source_total_frames",
        "keypoint_temporal_source_total_frames",
    ],
)
def test_resigned_staged_keypoint_authority_rejects_temporal_cross_field_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    receipt = copy.deepcopy(plan.staged_input_integrity_receipt)
    authority = copy.deepcopy(receipt["canonical_keypoint_authority"])

    if mutation == "top_level_source_total_frames":
        authority["source_total_frames"] = (
            int(authority["source_total_frames"]) + 1
        )
    else:
        temporal = authority["ordered_row_alignment"][
            "keypoint_temporal_authority"
        ]
        temporal["source_total_frames"] = int(temporal["source_total_frames"]) + 1

    authority = _resign_integrity_record(authority)
    receipt["canonical_keypoint_authority"] = authority
    receipt["canonical_keypoint_authority_sha256"] = authority["record_sha256"]
    receipt = _resign_integrity_record(receipt)

    with pytest.raises(ValueError):
        mod.eye_writer._canonical_staged_input_integrity_receipt(receipt)


def test_resigned_staged_keypoint_authority_rejects_swapped_ordered_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    receipt = copy.deepcopy(plan.staged_input_integrity_receipt)
    authority = copy.deepcopy(receipt["canonical_keypoint_authority"])

    swapped_labels = list(_KEYPOINT_LABELS)
    left_index = swapped_labels.index("eye_left")
    right_index = swapped_labels.index("eye_right")
    swapped_labels[left_index], swapped_labels[right_index] = (
        swapped_labels[right_index],
        swapped_labels[left_index],
    )
    authority["keypoint_labels"] = swapped_labels
    authority = _resign_integrity_record(authority)
    receipt["keypoint_axis"] = {
        "resolved_labels": swapped_labels,
        "resolved_head_keypoint_indices": {
            label: swapped_labels.index(label)
            for label in _KEYPOINT_LABELS
        },
    }
    receipt = _resign_receipt_with_keypoint_authority(
        plan,
        receipt,
        authority,
    )

    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="r",
        use_consolidated=False,
    )
    staged_group = staged_root[_KEYPOINT_RUN_PATH]
    assert tuple(staged_group.attrs["keypoint_labels"]) == _KEYPOINT_LABELS
    assert tuple(
        staged_group.attrs[KEYPOINT_LABEL_AUTHORITY_ATTR]["axis1"]["labels"]
    ) == _KEYPOINT_LABELS

    with pytest.raises(ValueError, match="keypoint labels differ"):
        _resolve_staged_context_from_receipt(plan, receipt)


def test_resigned_staged_keypoint_authority_rejects_noncanonical_identity_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    receipt = copy.deepcopy(plan.staged_input_integrity_receipt)
    authority = copy.deepcopy(receipt["canonical_keypoint_authority"])
    alignment = authority["ordered_row_alignment"]

    for name in ("subject_shape_row_identity", "keypoint_row_identity"):
        identity = alignment[name]
        identity["domain"] = "legacy_observation_row"
        identity["mode"] = "legacy_row_key"
        identity["components"] = ["legacy_row_key"]
    for name in (
        "subject_shape_temporal_authority",
        "keypoint_temporal_authority",
    ):
        temporal = alignment[name]
        temporal["source_identity_domain"] = "legacy_observation_row"
        temporal["source_identity_mode"] = "legacy_row_key"

    authority = _resign_integrity_record(authority)
    receipt = _resign_receipt_with_keypoint_authority(
        plan,
        receipt,
        authority,
    )

    with pytest.raises(ValueError, match="row-identity evidence is invalid"):
        _resolve_staged_context_from_receipt(plan, receipt)


@pytest.mark.parametrize(
    ("relative_path", "index"),
    [
        ("keypoints_runs/kp_raw_1/keypoints_roi", (0, 0, 0)),
        ("keypoints_runs/kp_raw_1/detection_success", (0,)),
        ("keypoints_runs/kp_raw_1/instance_key", (0,)),
        (
            "keypoints_runs/kp_raw_1/source_acquisition_frame_index",
            (0,),
        ),
    ],
)
def test_combined_receipt_rejects_staged_keypoint_input_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
    index: tuple[int, ...],
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="a",
        use_consolidated=False,
    )
    node = staged_root[relative_path]
    changed = np.asarray(node[:])
    if np.issubdtype(changed.dtype, np.bool_):
        changed[index] = not bool(changed[index])
    elif np.issubdtype(changed.dtype, np.integer):
        changed[index] += 10
    else:
        changed[index] += np.asarray(5, dtype=changed.dtype)
    node[:] = changed

    array_name = relative_path.rsplit("/", 1)[-1]
    with pytest.raises(
        ValueError,
        match=(
            "Staged canonical keypoint array "
            f"keypoints_runs/kp_raw_1/{array_name} differs from its authority"
        ),
    ):
        mod._resolve_source_plan(
            plan.staged_zarr,
            subject_shape_run=plan.subject_shape_run,
            keypoint_run=plan.keypoint_run,
            staged_input_integrity_receipt=(
                plan.staged_input_integrity_receipt
            ),
            staged_subject_shape_subset=True,
            verify_staged_payload=True,
        )


def test_combined_receipt_rejects_another_keypoint_source_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source_a, plan_a, _staging = _stage_synthetic_source(
        monkeypatch,
        tmp_path,
        source_name="source-a.zarr",
        scratch_name="scratch-a",
        run_name="eye_a",
    )
    source_b = tmp_path / "source-b.zarr"
    _build_source(source_b)
    root_b = zarr.open_group(str(source_b), mode="a", use_consolidated=False)
    keypoints_b = root_b["keypoints_runs/kp_raw_1/keypoints_roi"]
    changed_b = np.asarray(keypoints_b[:])
    changed_b[0, 0, 0] += np.float32(9.0)
    keypoints_b[:] = changed_b
    _accept_synthetic_subject_shape_publication(monkeypatch, source_b)
    plan_b = mod.build_eye_angle_materialization_plan(
        source_b,
        scratch_root=tmp_path / "scratch-b",
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name="eye_b",
        chunk_rows=2,
        fps=100.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Staged canonical keypoint array "
            "keypoints_runs/kp_raw_1/keypoints_roi differs from its authority"
        ),
    ):
        mod._resolve_source_plan(
            plan_a.staged_zarr,
            subject_shape_run=plan_a.subject_shape_run,
            keypoint_run=plan_a.keypoint_run,
            staged_input_integrity_receipt=(
                plan_b.staged_input_integrity_receipt
            ),
            staged_subject_shape_subset=True,
            verify_staged_payload=True,
        )


def test_worker_rejects_transient_snapshot_tamper_after_source_restore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="a",
        use_consolidated=False,
    )
    authority = _nested_subject_shape_authority(
        plan.staged_input_integrity_receipt
    )
    keypoint_authority = _nested_keypoint_authority(
        plan.staged_input_integrity_receipt
    )
    context = mod.eye_writer._resolve_eye_angle_inputs(
        staged_root,
        subject_shape_run=plan.subject_shape_run,
        refined_subject_run=None,
        keypoint_run=plan.keypoint_run,
        _staged_subject_shape_authority=authority,
        _staged_keypoint_authority=keypoint_authority,
        _verify_staged_payload=False,
    )
    run_group = staged_root["analysis"].require_group(
        "eye_angle_runs"
    ).create_group("transient_snapshot_probe")
    mod.eye_writer._prepare_base_output_arrays(
        run_group,
        total_detections=plan.row_count,
        chunk_len=plan.chunk_rows,
    )
    keypoints_node = context.kp_group["keypoints_roi"]
    original = np.asarray(keypoints_node[:])
    real_loader = mod.eye_writer._load_eye_angle_chunk_input_snapshot

    def load_while_source_is_transiently_changed(
        snapshot_context,
        *,
        start_row: int,
        stop_row: int,
    ):
        changed = original.copy()
        changed[start_row, 0, 0] += np.float32(7.0)
        keypoints_node[:] = changed
        try:
            return real_loader(
                snapshot_context,
                start_row=start_row,
                stop_row=stop_row,
            )
        finally:
            keypoints_node[:] = original

    monkeypatch.setattr(
        mod.eye_writer,
        "_load_eye_angle_chunk_input_snapshot",
        load_while_source_is_transiently_changed,
    )

    with pytest.raises(
        ValueError,
        match="worker input differs.*keypoints_roi: payload changed",
    ):
        mod.eye_writer._process_and_write_eye_angle_chunk_groups(
            context,
            run_group,
            start_row=0,
            stop_row=2,
            chunk_index=0,
            fps=100.0,
            execution_backend="serial_driver",
            _staged_input_integrity_chunk=(
                plan.staged_input_integrity_receipt["chunks"][0]
            ),
        )

    assert np.array_equal(np.asarray(keypoints_node[:]), original)


def test_subject_shape_receipt_alone_cannot_enter_writer_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    authority = _nested_subject_shape_authority(
        plan.staged_input_integrity_receipt
    )
    writer_argv = [
        str(plan.staged_zarr),
        "--subject-shape-run",
        plan.subject_shape_run,
        "--keypoint-run",
        plan.keypoint_run,
        "--run-name",
        "subject_only",
        "--chunk-size",
        str(plan.chunk_rows),
        "--execution-backend",
        "serial_driver",
        "--scheduler",
        "single-threaded",
        "--num-workers",
        "1",
        "--fps",
        "100.0",
        "--quiet",
    ]

    with pytest.raises(
        ValueError,
        match="input integrity receipt fields are not exact|unsupported",
    ):
        mod.eye_writer.main(
            writer_argv,
            _staged_input_integrity_receipt=authority,
        )

    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="r",
        use_consolidated=False,
    )
    assert staged_root["analysis"].get("eye_angle_runs") is None


def test_combined_receipt_rejects_a_different_runtime_fps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, plan, _staging = _stage_synthetic_source(monkeypatch, tmp_path)
    writer_argv = [
        str(plan.staged_zarr),
        "--subject-shape-run",
        plan.subject_shape_run,
        "--keypoint-run",
        plan.keypoint_run,
        "--run-name",
        "fps_mismatch",
        "--chunk-size",
        str(plan.chunk_rows),
        "--execution-backend",
        "serial_driver",
        "--scheduler",
        "single-threaded",
        "--num-workers",
        "1",
        "--fps",
        "99.0",
        "--quiet",
    ]

    with pytest.raises(
        ValueError,
        match="FPS differs from its sealed materialization plan",
    ):
        mod.eye_writer.main(
            writer_argv,
            _staged_input_integrity_receipt=(
                plan.staged_input_integrity_receipt
            ),
        )

    staged_root = zarr.open_group(
        str(plan.staged_zarr),
        mode="r",
        use_consolidated=False,
    )
    assert staged_root["analysis"].get("eye_angle_runs") is None


def test_materializer_stages_computes_shards_and_publishes_with_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _accept_synthetic_subject_shape_publication(monkeypatch, source)
    registry_events: list[dict[str, object]] = []

    def capture_registry(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        run = kwargs["run_group"]
        parent = root["analysis/eye_angle_runs"]
        assert run.attrs["palette_run_completion_status"] == "complete"
        assert run.attrs["stage_selector_eligible"] is True
        assert parent.attrs["latest"] == "eye_1"
        registry_events.append({"zarr_path": zarr_path, **kwargs})
        return True

    monkeypatch.setattr(
        mod,
        "emit_eye_angle_stage_completion",
        capture_registry,
    )

    result = mod.materialize_eye_angles(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name="eye_1",
        chunk_rows=2,
        angle_chunk_rows=2,
        angle_chunk_columns=4,
        output_shard_rows=3,
        angle_shard_columns=4,
        execution_backend="dask_worker_chunks",
        scheduler="single-threaded",
        num_workers=1,
        shard_workers=1,
        fps=100.0,
        smoothing_window=3,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-eye-materializer",
    )

    assert result["status"] == "complete"
    assert result["publish"]["pre_pointer_validation"]["valid"] is True
    assert result["publish"]["final_validation"]["valid"] is True
    assert result["local_materialization"]["regular_validation"][
        "exact_compact_v7_valid"
    ] is True
    assert result["local_materialization"]["sharded_validation"][
        "exact_compact_v7_valid"
    ] is True
    for validation_name in (
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
    ):
        assert result["publish"][validation_name]["exact_compact_v7_valid"] is True
    assert result["publish"]["source_revision_audit"]["status"] == "current"
    assert result["publish"]["registry_updated"] is True
    assert len(registry_events) == 1
    assert registry_events[0]["run_name"] == "eye_1"
    assert registry_events[0]["source"] == "eye_angle_atomic_materializer"

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/eye_angle_runs"]
    run = parent["eye_1"]
    from fisheye.analysis.eye_angle_io import load_eye_angle_run_tables

    strict_tables = load_eye_angle_run_tables(root, run_name="eye_1")
    assert strict_tables.schema_version == 7
    assert parent.attrs["latest"] == "eye_1"
    assert parent.attrs["latest_complete"] == "eye_1"
    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["atomic_publication_owner_uuid"]
    assert "atomic_publication_tombstone" not in run.attrs
    assert run.attrs["schema_version"] == 7
    assert run.attrs["eye_angle_output_schema"]["schema_version"] == 9
    assert run.attrs["eye_angle_algorithm_contract"]["schema_version"] == 1
    assert run.attrs["keypoint_source_mode"] == (
        mod.eye_writer.EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
    )
    assert run.attrs["source_keypoints_run"] == "kp_raw_1"
    assert tuple(run["frame_angles"].chunks) == (2, 4)
    assert tuple(run["roi_angles"].chunks) == (2, 4)
    assert tuple(run["frame_angles"].shards) == (4, 4)
    assert tuple(run["roi_angles"].shards) == (4, 4)
    assert np.array_equal(
        np.asarray(run["support/instance_key"][:]),
        np.arange(1, 5, dtype=np.uint64),
    )
    assert np.array_equal(
        np.asarray(run["support/source_acquisition_frame_index"][:]),
        np.arange(4, dtype=np.int64),
    )
    assert np.array_equal(
        np.asarray(run["support/frame_indices"][:]),
        np.asarray(run["support/source_acquisition_frame_index"][:]),
    )
    assert run["support/instance_key"].attrs["identity_mode"] == "instance_key"
    assert run["support/frame_indices"].attrs["compatibility_alias_of"] == (
        "support/source_acquisition_frame_index"
    )
    assert "heading_deg" in run["support/body_frame"]

    local = run.attrs["node_local_materialization"]
    assert local["authoritative_source_zarr"] == str(source.resolve())
    assert local["source_staging"]["node_local_staged_zarr"] == str(
        (scratch / "eye-inputs-and-output.zarr").resolve()
    )
    assert local["source_staging"]["source_revision_audit"]["status"] == "current"
    assert local["source_staging"]["source_authority_mode"] == (
        "digest_bound_staged_subset"
    )
    receipt_sha256 = local["staged_input_integrity_receipt_sha256"]
    assert receipt_sha256 == (
        local["source_staging"]["staged_input_integrity_receipt_sha256"]
    )
    assert local["staged_input_integrity_receipt"] == (
        local["source_staging"]["staged_input_integrity_receipt"]
    )
    assert run.attrs["staged_input_integrity_receipt_sha256"] == receipt_sha256
    assert local["compute"]["writer"] == "fisheye.analysis.eye_angle_analysis"
    assert local["compute"]["angle_chunk_rows"] == 2
    assert local["compute"]["angle_chunk_columns"] == 4
    assert local["compute"]["angle_column_order_profile"] == (
        "semantic_bundles_v1"
    )
    assert local["compute"]["stage_command"] == "unit-test-eye-materializer"
    assert local["algorithm_contract"]["sha256"]
    assert local["output_contract"]["sha256"]
    assert local["sharding"]["exact_decoded_validation"] is True
    layouts = {
        item["path"]: item for item in local["sharding"]["angle_array_layouts"]
    }
    assert tuple(layouts["frame_angles"]["inner_chunks"]) == (2, 4)
    assert tuple(layouts["frame_angles"]["outer_shards"]) == (4, 4)

    provenance = run.attrs["provenance"]
    assert provenance["materialization"]["authoritative_source_zarr"] == str(
        source.resolve()
    )
    assert provenance["materialization"]["selected_arrays"]
    assert all(
        not path.startswith("refined_keypoints_runs/")
        and not path.endswith("/heading")
        for path in provenance["materialization"]["selected_arrays"]
    )
    assert provenance["execution"][
        "staged_input_integrity_receipt_sha256"
    ] == receipt_sha256
    assert provenance["materialization"][
        "staged_input_integrity_receipt_sha256"
    ] == receipt_sha256
    assert run.attrs["source_eye_geometry_authority_mode"] == (
        "digest_bound_staged_subset"
    )
    assert run.attrs["eye_angle_source_contracts"]["eye_geometry"][
        "source_authority"
    ]["record_sha256"] == local["staged_input_integrity_receipt"][
        "subject_shape_authority_sha256"
    ]
    publication = run.attrs["cluster_output_staging"]
    assert publication["publisher_contract"] == {
        "schema_id": "palette.atomic_run_group_publisher",
        "schema_version": 1,
    }
    assert publication["promotion_policy"] == (
        "complete_ineligible_then_pointers_then_eligibility_final"
    )
    assert publication["physical_copy"]["verification"] == (
        "sha256_all_physical_files"
    )


def _seed_eye_angle_selectors(source: Path) -> None:
    root = zarr.open_group(str(source), mode="a", use_consolidated=False)
    parent = root["analysis"].require_group("eye_angle_runs")
    parent.attrs["latest"] = "established_eye"
    parent.attrs["latest_complete"] = "established_eye"


def _materialize_storage_candidate(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    scratch: Path,
    *,
    run_name: str = "eye_candidate",
) -> dict[str, object]:
    _accept_synthetic_subject_shape_publication(monkeypatch, source)
    return mod.materialize_eye_angles(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name=run_name,
        storage_profile=EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        chunk_rows=2,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_workers=1,
        fps=100.0,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-eye-candidate",
    )


def test_storage_candidate_is_atomic_ineligible_pointer_preserving_and_consolidated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _seed_eye_angle_selectors(source)
    # Deliberately create a generation that cannot yet contain the candidate.
    # Publication must refresh it only after all publisher metadata is final.
    zarr.consolidate_metadata(str(source))

    result = _materialize_storage_candidate(monkeypatch, source, scratch)

    assert result["status"] == "complete"
    publication = result["publish"]
    assert publication["registry_updated"] is False
    assert publication["archive_direct_consolidated_array_count"] == 41
    assert publication["node_local_sharded_run"] is None
    assert publication["node_local_publication_run"].endswith(
        "eye-inputs-and-output.zarr/analysis/eye_angle_runs/eye_candidate"
    )
    assert publication["promotion_policy"] == (
        "immutable_named_candidate_no_pointer_or_registry_activation"
    )
    assert publication["metadata_visibility_policy"] == {
        "authoritative_root_consolidation": "after_final_publisher_metadata_write",
        "direct_consolidated_group_attrs_required": True,
        "direct_consolidated_array_declarations_required": 41,
        "consolidated_parent_selectors_must_match_publication_snapshot": True,
    }
    assert result["local_materialization"][
        "local_direct_consolidated_array_count"
    ] == 41
    assert result["local_materialization"]["final_physical_validation"][
        "candidate_storage_valid"
    ] is True
    assert "sharded_validation" not in result["local_materialization"]
    assert "sharding" not in result["local_materialization"]
    assert not (scratch / "eye-angle-sharded-run").exists()

    direct = zarr.open_group(str(source), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(source), mode="r", use_consolidated=True)
    for root in (direct, consolidated):
        parent = root["analysis/eye_angle_runs"]
        assert parent.attrs["latest"] == "established_eye"
        assert parent.attrs["latest_complete"] == "established_eye"
        candidate = parent["eye_candidate"]
        assert candidate.attrs["palette_run_completion_status"] == "complete"
        assert candidate.attrs["stage_selector_eligible"] is False
        assert candidate.attrs["eye_angle_storage_candidate"][
            "activation_allowed"
        ] is False
        assert candidate.attrs["cluster_output_staging"]["promotion_policy"] == (
            "immutable_named_candidate_no_pointer_or_registry_activation"
        )
    assert direct["analysis/eye_angle_runs/eye_candidate"].attrs[
        "cluster_output_staging"
    ] == consolidated["analysis/eye_angle_runs/eye_candidate"].attrs[
        "cluster_output_staging"
    ]

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        mod.materialize_eye_angles(
            source,
            scratch_root=tmp_path / "other-scratch",
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_candidate",
            storage_profile=EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
            execution_backend="serial_driver",
            fps=100.0,
            apply=False,
        )


def test_storage_candidate_recovers_same_name_after_pre_rename_copy_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _seed_eye_angle_selectors(source)
    real_copy = atomic_mod._copy_tree
    copy_attempts = 0

    def interrupt_first_copy(source_path, target_path, *, backend):  # type: ignore[no-untyped-def]
        nonlocal copy_attempts
        copy_attempts += 1
        if copy_attempts == 1:
            raise RuntimeError("injected pre-rename copy interruption")
        return real_copy(source_path, target_path, backend=backend)

    monkeypatch.setattr(atomic_mod, "_copy_tree", interrupt_first_copy)
    with pytest.raises(RuntimeError, match="pre-rename copy interruption"):
        _materialize_storage_candidate(monkeypatch, source, scratch)

    direct = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = direct["analysis/eye_angle_runs"]
    assert parent.attrs["latest"] == "established_eye"
    assert parent.attrs["latest_complete"] == "established_eye"
    assert parent.get("eye_candidate") is None
    assert not tuple(
        (source / "analysis" / "eye_angle_runs").glob(
            ".eye_candidate.publish_tmp.*"
        )
    )

    plan = mod.build_eye_angle_materialization_plan(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name="eye_candidate",
        storage_profile=EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        chunk_rows=2,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_workers=1,
        fps=100.0,
    )
    local = zarr.open_group(
        str(plan.local_run_path), mode="r", use_consolidated=False
    )
    recovered = mod.publish_eye_angle_run(
        plan,
        materialization_payload=dict(local.attrs["node_local_materialization"]),
        copy_backend="python",
    )
    assert recovered["archive_direct_consolidated_array_count"] == 41
    direct = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert direct["analysis/eye_angle_runs/eye_candidate"].attrs[
        "palette_run_completion_status"
    ] == "complete"

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        mod.publish_eye_angle_run(
            plan,
            materialization_payload=dict(local.attrs["node_local_materialization"]),
            copy_backend="python",
        )


def test_storage_candidate_post_consolidation_failure_repairs_both_metadata_views(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _seed_eye_angle_selectors(source)
    zarr.consolidate_metadata(str(source))
    real_equivalence = mod._require_candidate_direct_consolidated_equivalence
    equivalence_labels: list[str] = []

    def fail_after_authoritative_consolidation(
        direct_run,  # type: ignore[no-untyped-def]
        consolidated_run,  # type: ignore[no-untyped-def]
        *,
        dimensions,  # type: ignore[no-untyped-def]
        label: str,
    ) -> int:
        result = real_equivalence(
            direct_run,
            consolidated_run,
            dimensions=dimensions,
            label=label,
        )
        equivalence_labels.append(label)
        if label == "Authoritative":
            raise RuntimeError(
                "injected post-consolidation authoritative equivalence failure"
            )
        return result

    monkeypatch.setattr(
        mod,
        "_require_candidate_direct_consolidated_equivalence",
        fail_after_authoritative_consolidation,
    )
    with pytest.raises(
        RuntimeError,
        match="post-consolidation authoritative equivalence failure",
    ):
        _materialize_storage_candidate(monkeypatch, source, scratch)
    assert equivalence_labels == ["Node-local", "Authoritative"]

    direct = zarr.open_group(str(source), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(source), mode="r", use_consolidated=True)
    direct_parent = direct["analysis/eye_angle_runs"]
    consolidated_parent = consolidated["analysis/eye_angle_runs"]
    for parent in (direct_parent, consolidated_parent):
        assert parent.attrs["latest"] == "established_eye"
        assert parent.attrs["latest_complete"] == "established_eye"
        failed = parent["eye_candidate"]
        assert failed.attrs["palette_run_completion_status"] == "failed"
        assert failed.attrs["stage_selector_eligible"] is False
        assert "palette_run_completed_at_utc" not in failed.attrs
        tombstone = failed.attrs["atomic_publication_tombstone"]
        assert tombstone["schema_id"] == "palette.atomic_publication_tombstone"
        assert tombstone["schema_version"] == 1
        assert set(tombstone) == {
            "schema_id",
            "schema_version",
            "failed_at_utc",
            "publication_owner_attr",
            "publication_owner_uuid",
            "run_name",
            "run_path",
            "public_path_retained",
            "selector_eligible",
            "retry_policy",
            "failure_type",
            "failure",
        }
        assert failed.attrs[tombstone["publication_owner_attr"]] == tombstone[
            "publication_owner_uuid"
        ]
        assert failed.attrs["palette_run_error"] == tombstone["failure"]
        assert tombstone["public_path_retained"] is True
        assert tombstone["selector_eligible"] is False
        assert tombstone["retry_policy"] == "new_immutable_run_name_required"
        assert (
            "post-consolidation authoritative equivalence failure"
            in tombstone["failure"]
        )
    assert dict(direct_parent["eye_candidate"].attrs) == dict(
        consolidated_parent["eye_candidate"].attrs
    )


@pytest.mark.parametrize(
    ("tamper_kind", "expected_error"),
    [
        pytest.param(
            "instance_key",
            "support/instance_key differs from sealed canonical source",
            id="instance-key-values",
        ),
        pytest.param(
            "coordinated_acquisition_frames",
            (
                "support/source_acquisition_frame_index differs from sealed "
                "canonical source"
            ),
            id="coordinated-acquisition-frame-values",
        ),
        pytest.param(
            "angle_index_metadata",
            "channel_index_content_mismatch:angle_channel_index/representation",
            id="exact-angle-index-metadata",
        ),
        pytest.param(
            "column_order_envelope",
            "column_order_contract_mismatch:angle_column_order_contract",
            id="exact-column-order-envelope",
        ),
        pytest.param(
            "output_schema_nested",
            "eye_angle_output_schema must exactly equal its executable contract",
            id="exact-output-schema-nested",
        ),
        pytest.param(
            "algorithm_contract_nested",
            (
                "eye_angle_algorithm_contract must exactly equal the reconstructed "
                "executable contract"
            ),
            id="exact-algorithm-contract-nested",
        ),
        pytest.param(
            "heading_alias",
            "heading_alias_mismatch:roi_angles/heading_deg",
            id="heading-alias-values",
        ),
        pytest.param(
            "frame_alias",
            "frame_alias_mismatch:support/frame_indices",
            id="frame-alias-values",
        ),
    ],
)
def test_publication_rejects_output_identity_that_differs_from_sealed_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper_kind: str,
    expected_error: str,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _accept_synthetic_subject_shape_publication(monkeypatch, source)
    real_publish = mod.publish_eye_angle_run
    tampered = False

    def tamper_then_publish(plan, *, materialization_payload, copy_backend):
        nonlocal tampered
        sharded = zarr.open_group(
            str(plan.sharded_run),
            mode="a",
            use_consolidated=False,
        )
        if tamper_kind == "instance_key":
            instance_key = np.asarray(sharded["support/instance_key"][:])
            instance_key[0] += np.uint64(100)
            sharded["support/instance_key"][:] = instance_key
        elif tamper_kind == "coordinated_acquisition_frames":
            acquisition = np.asarray(
                sharded["support/source_acquisition_frame_index"][:],
                dtype=np.int64,
            )
            acquisition[0] += np.int64(100)
            sharded["support/source_acquisition_frame_index"][:] = acquisition
            sharded["support/frame_indices"][:] = acquisition
            assert np.array_equal(
                np.asarray(
                    sharded["support/source_acquisition_frame_index"][:],
                    dtype=np.int64,
                ),
                np.asarray(
                    sharded["support/frame_indices"][:],
                    dtype=np.int64,
                ),
            )
        elif tamper_kind == "angle_index_metadata":
            representation = np.asarray(
                sharded["angle_channel_index/representation"][:],
                dtype=np.uint8,
            )
            representation[0, :] = 0
            sharded["angle_channel_index/representation"][:] = representation
        elif tamper_kind == "column_order_envelope":
            column_order = dict(sharded.attrs["angle_column_order_contract"])
            column_order["unexpected"] = True
            sharded.attrs["angle_column_order_contract"] = column_order
        elif tamper_kind == "output_schema_nested":
            output_schema = dict(sharded.attrs["eye_angle_output_schema"])
            output_schema["row_axes"] = {
                **output_schema["row_axes"],
                "roi": "tampered_rows",
            }
            sharded.attrs["eye_angle_output_schema"] = output_schema
        elif tamper_kind == "algorithm_contract_nested":
            algorithm = dict(sharded.attrs["eye_angle_algorithm_contract"])
            algorithm["delta"] = {
                **algorithm["delta"],
                "time_normalized": True,
            }
            sharded.attrs["eye_angle_algorithm_contract"] = algorithm
        elif tamper_kind == "heading_alias":
            roi_angles = np.asarray(sharded["roi_angles"][:])
            names = mod._decode_text_index(sharded["angle_channel_index/name"])
            roi_angles[0, names.index("heading_deg")] += np.float32(1.0)
            sharded["roi_angles"][:] = roi_angles
        else:
            frame_alias = np.asarray(sharded["support/frame_indices"][:])
            frame_alias[0] += np.int64(1)
            sharded["support/frame_indices"][:] = frame_alias
        tampered = True
        return real_publish(
            plan,
            materialization_payload=materialization_payload,
            copy_backend=copy_backend,
        )

    monkeypatch.setattr(mod, "publish_eye_angle_run", tamper_then_publish)

    with pytest.raises(
        RuntimeError,
        match=rf"^Local run validation failed: .*{expected_error}",
    ):
        mod.materialize_eye_angles(
            source,
            scratch_root=scratch,
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_1",
            chunk_rows=2,
            output_shard_rows=3,
            execution_backend="serial_driver",
            scheduler="single-threaded",
            num_workers=1,
            shard_workers=1,
            fps=100.0,
            copy_backend="python",
            apply=True,
            check_capacity=False,
        )

    assert tampered is True
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "eye_angle_runs" not in root["analysis"]


def test_post_read_staged_payload_tamper_fails_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _accept_synthetic_subject_shape_publication(monkeypatch, source)
    real_process = mod.eye_writer._process_and_write_eye_angle_chunk_groups
    tampered = False

    def process_then_tamper(context, run_group, **kwargs):
        nonlocal tampered
        result = real_process(context, run_group, **kwargs)
        if not tampered:
            ellipse = context.eye_geometry.group[
                "components/eye_left/ellipse_params"
            ]
            changed = np.asarray(ellipse[:])
            changed[0, 0] += np.float32(3.0)
            ellipse[:] = changed
            tampered = True
        return result

    monkeypatch.setattr(
        mod.eye_writer,
        "_process_and_write_eye_angle_chunk_groups",
        process_then_tamper,
    )

    with pytest.raises(ValueError, match="differs from its canonical payload"):
        mod.materialize_eye_angles(
            source,
            scratch_root=scratch,
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_1",
            chunk_rows=2,
            output_shard_rows=3,
            execution_backend="serial_driver",
            scheduler="single-threaded",
            num_workers=1,
            shard_workers=1,
            fps=100.0,
            copy_backend="python",
            apply=True,
            check_capacity=False,
        )

    assert tampered is True
    assert scratch.exists()
    staged = zarr.open_group(
        str(scratch / "eye-inputs-and-output.zarr"),
        mode="r",
        use_consolidated=False,
    )
    local_run = staged["analysis/eye_angle_runs/eye_1"]
    assert local_run.attrs.get("palette_run_completion_status") != "complete"
    assert local_run.attrs.get("stage_selector_eligible") is not True
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "eye_angle_runs" not in root["analysis"]


def test_publication_rolls_back_when_source_revision_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _accept_synthetic_subject_shape_publication(monkeypatch, source)
    real_audit = mod.audit_eye_angle_source_revision
    call_count = 0

    def changing_audit(plan):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return real_audit(plan)
        return {
            "schema_id": mod.SOURCE_REVISION_AUDIT_SCHEMA_ID,
            "status": "changed",
            "errors": ["injected source revision change"],
        }

    monkeypatch.setattr(mod, "audit_eye_angle_source_revision", changing_audit)

    with pytest.raises(RuntimeError, match="inputs changed during materialization"):
        mod.materialize_eye_angles(
            source,
            scratch_root=scratch,
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_1",
            chunk_rows=2,
            output_shard_rows=3,
            execution_backend="serial_driver",
            scheduler="single-threaded",
            num_workers=1,
            shard_workers=1,
            fps=100.0,
            copy_backend="python",
            apply=True,
            check_capacity=False,
        )

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/eye_angle_runs"]
    failed = parent["eye_1"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert "palette_run_completed_at_utc" not in failed.attrs
    tombstone = failed.attrs["atomic_publication_tombstone"]
    assert tombstone["schema_id"] == "palette.atomic_publication_tombstone"
    assert tombstone["public_path_retained"] is True
    assert tombstone["selector_eligible"] is False
    assert tombstone["retry_policy"] == "new_immutable_run_name_required"
    assert "injected source revision change" in tombstone["failure"]
    assert parent.attrs.get("latest") != "eye_1"
    assert parent.attrs.get("latest_complete") != "eye_1"


def _materialize_established_eye_source(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    scratch: Path,
) -> dict[str, object]:
    _accept_synthetic_subject_shape_publication(monkeypatch, source)
    return mod.materialize_eye_angles(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name="eye_source",
        chunk_rows=2,
        output_shard_rows=3,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_workers=1,
        fps=100.0,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-eye-source",
    )


def test_candidate_execution_binding_acceptance_and_post_return_tombstone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)
    _materialize_established_eye_source(
        monkeypatch,
        source,
        tmp_path / "source-scratch",
    )
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    source_hashes = mod.compute_eye_angle_logical_hashes(
        root["analysis/eye_angle_runs/eye_source"]
    )
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": "eye-unit",
        "request_payload_digest": "a" * 64,
        "candidate_run_path": "analysis/eye_angle_runs/eye_candidate",
    }
    accepted: list[str] = []

    def accept(_root, _parent, candidate):  # type: ignore[no-untyped-def]
        assert candidate.attrs[mod.EXECUTION_BINDING_ATTR] == binding
        accepted.append("called")
        return {"accepted": True, "execution_binding": binding}

    result = mod.materialize_eye_angles(
        source,
        scratch_root=tmp_path / "candidate-scratch",
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        run_name="eye_candidate",
        storage_profile=EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        chunk_rows=2,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_workers=1,
        native_threads=1,
        fps=100.0,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        execution_binding=binding,
        expected_source_logical_hashes=source_hashes,
        publication_acceptance_validator=accept,
    )

    assert accepted == ["called"]
    assert result["caller_acceptance"] == {
        "accepted": True,
        "execution_binding": binding,
    }
    assert result["source_logical_manifest_sha256"] == result[
        "published_logical_manifest_sha256"
    ]
    assert [
        phase["name"] for phase in result["runtime_telemetry"]["phases"]
    ] == list(mod.EYE_ANGLE_EXECUTION_PHASE_ORDER)
    direct = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = direct["analysis/eye_angle_runs"]
    assert parent.attrs["latest"] == "eye_source"
    assert parent.attrs["latest_complete"] == "eye_source"
    assert parent["eye_candidate"].attrs["stage_selector_eligible"] is False

    tombstone = mod.tombstone_eye_angle_execution_candidate(
        source,
        run_name="eye_candidate",
        expected_execution_binding=binding,
        failure_phase="runner_receipt_assembly",
        error_type="RuntimeError",
        error_message="injected post-return failure",
    )
    assert tombstone["tombstoned"] is True
    for consolidated in (False, True):
        view = zarr.open_group(
            str(source),
            mode="r",
            use_consolidated=consolidated,
        )
        failed = view["analysis/eye_angle_runs/eye_candidate"]
        assert failed.attrs["palette_run_completion_status"] == "failed"
        assert failed.attrs["stage_selector_eligible"] is False
        assert failed.attrs[mod.EXECUTION_FAILURE_TOMBSTONE_ATTR][
            "failure_phase"
        ] == "runner_receipt_assembly"


def test_candidate_atomic_acceptance_failure_is_failed_and_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)
    _materialize_established_eye_source(
        monkeypatch,
        source,
        tmp_path / "source-scratch",
    )
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    source_hashes = mod.compute_eye_angle_logical_hashes(
        root["analysis/eye_angle_runs/eye_source"]
    )
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": "eye-fail",
        "request_payload_digest": "b" * 64,
        "candidate_run_path": "analysis/eye_angle_runs/eye_candidate",
    }

    def reject(*_args):  # type: ignore[no-untyped-def]
        raise RuntimeError("injected atomic acceptance rejection")

    with pytest.raises(RuntimeError, match="atomic acceptance rejection") as error:
        mod.materialize_eye_angles(
            source,
            scratch_root=tmp_path / "candidate-scratch",
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_candidate",
            storage_profile=EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
            chunk_rows=2,
            execution_backend="serial_driver",
            scheduler="single-threaded",
            num_workers=1,
            shard_workers=1,
            native_threads=1,
            fps=100.0,
            copy_backend="python",
            apply=True,
            keep_scratch=True,
            check_capacity=False,
            execution_binding=binding,
            expected_source_logical_hashes=source_hashes,
            publication_acceptance_validator=reject,
        )
    assert isinstance(
        getattr(error.value, "palette_runtime_telemetry", None),
        dict,
    )
    direct = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = direct["analysis/eye_angle_runs"]
    failed = parent["eye_candidate"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert parent.attrs["latest"] == "eye_source"
    assert parent.attrs["latest_complete"] == "eye_source"
