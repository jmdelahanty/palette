from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.shared.canonical_coordinate_publication import (
    build_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_NOT_SUITABLE,
    CanonicalCollectionAxis,
    DigestBoundCoordinateRecordRef,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
    stamp_and_bind_row_identity_contract,
)
from fisheye.shared.coordinate_record import stamp_and_bind_persisted_coordinate_record
from fisheye.shared.pixel_frame_authority import (
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
    stamp_source_camera_pixel_frame_authority,
)
import fisheye.shared.tail_coordinate_publication as mod


def _identity_source(*, direct: bool = True) -> tuple[zarr.Group, SimpleNamespace]:
    root = zarr.group()
    source = root.require_group("analysis/subject_shape_runs/shape")
    if direct:
        source.create_array(
            "instance_key",
            data=np.asarray([11, 22], dtype=np.uint64),
        )
        source.create_array(
            "source_crop_row_ids",
            data=np.asarray([5, 2], dtype=np.int64),
        )
        source.create_array(
            "source_acquisition_frame_index",
            data=np.asarray([101, 105], dtype=np.int64),
        )
    else:
        aliases = source.require_group("row_index")
        aliases.create_array(
            "instance_key",
            data=np.asarray([11, 22], dtype=np.uint64),
        )
        aliases.create_array(
            "frame_indices",
            data=np.asarray([101, 105], dtype=np.int64),
        )
    publication = SimpleNamespace(
        _run=source,
        row_identity=SimpleNamespace(leading_dimension=2),
    )
    return root, publication


def test_tail_identity_requires_direct_canonical_arrays_not_aliases() -> None:
    root, publication = _identity_source(direct=False)
    run = root.require_group("analysis/tail_kinematics_runs/tail")
    run.create_array("instance_key", data=np.asarray([11, 22], dtype=np.uint64))
    run.create_array("source_crop_row_ids", data=np.asarray([5, 2], dtype=np.int64))
    run.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([101, 105], dtype=np.int64),
    )

    with pytest.raises(
        mod.TailCoordinatePublicationError,
        match="required direct array 'instance_key'",
    ):
        mod._validate_exact_identity_copies(run, publication)


def test_tail_identity_rejects_reordered_or_substituted_instance_keys() -> None:
    root, publication = _identity_source()
    run = root.require_group("analysis/tail_kinematics_runs/tail")
    run.create_array("instance_key", data=np.asarray([22, 11], dtype=np.uint64))
    run.create_array("source_crop_row_ids", data=np.asarray([5, 2], dtype=np.int64))
    run.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([101, 105], dtype=np.int64),
    )

    with pytest.raises(
        mod.TailCoordinatePublicationError,
        match="not an exact dtype-preserving source copy",
    ):
        mod._validate_exact_identity_copies(run, publication)


def _minimal_posture_run() -> zarr.Group:
    root = zarr.group()
    run = root.require_group("analysis/tail_posture_view_runs/posture")
    arrays = {
        "instance_key": np.asarray([11, 22], dtype=np.uint64),
        "source_crop_row_ids": np.asarray([5, 2], dtype=np.int64),
        "source_acquisition_frame_index": np.asarray([101, 105], dtype=np.int64),
        "valid": np.asarray([True, True], dtype=bool),
        "failure_reason_bytes": np.zeros((2, 64), dtype=np.uint8),
        "head_xy": np.zeros((2, 2), dtype=np.float32),
        "head_yaw_rad": np.zeros((2,), dtype=np.float32),
        "tail_keypoints_xy": np.zeros((2, 3, 2), dtype=np.float32),
        "tail_angle_rad": np.zeros((2, 2), dtype=np.float32),
        "tail_angle_deg": np.zeros((2, 2), dtype=np.float32),
    }
    for name, values in arrays.items():
        run.create_array(name, data=values)
    records = run.require_group("coordinate_records")
    records.require_group("source_subject_shape")
    records.require_group("point_collection_axis")
    records.require_group("measurement_collection_axis")
    records.require_group("derivation")
    records.require_group("measurement_authority")
    return run


def test_tail_schema_inventory_rejects_unknown_arrays_and_groups() -> None:
    run = _minimal_posture_run()
    arrays, groups = mod._validate_closed_schema(run, "tail_posture_view")
    assert arrays == tuple(sorted(mod._POSTURE_ARRAYS))
    assert groups == tuple(sorted(mod._RECORD_GROUPS))

    run.create_array("rogue_geometry", data=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(mod.TailCoordinatePublicationError, match="array inventory"):
        mod._validate_closed_schema(run, "tail_posture_view")
    del run["rogue_geometry"]
    run.require_group("rogue_namespace")
    with pytest.raises(mod.TailCoordinatePublicationError, match="group inventory"):
        mod._validate_closed_schema(run, "tail_posture_view")


def test_tail_collection_axis_rejects_mismatched_declared_cardinality() -> None:
    run = _minimal_posture_run()
    run.attrs["keypoint_count"] = 4

    with pytest.raises(
        mod.TailCoordinatePublicationError,
        match="keypoint_count.*cardinality",
    ):
        mod._collection_axis_record(run, "tail_posture_view")


def test_tail_schema_rejects_partial_source_revision_identity() -> None:
    run = _minimal_posture_run()
    revisions = run.require_group("source_refined_subject_masks")
    revisions.create_array(
        "row_revision",
        data=np.asarray([[1], [2]], dtype=np.int64),
    )

    with pytest.raises(
        mod.TailCoordinatePublicationError,
        match="complete controlled revision pair",
    ):
        mod._validate_closed_schema(run, "tail_posture_view")


def test_tail_schema_accepts_complete_nested_revision_identity() -> None:
    run = _minimal_posture_run()
    revisions = run.require_group("source_refined_subject_masks")
    revisions.create_array(
        "row_revision",
        data=np.asarray([[1], [2]], dtype=np.int64),
    )
    revisions.create_array(
        "row_revision_available",
        data=np.asarray([[True], [True]], dtype=bool),
    )

    arrays, groups = mod._validate_closed_schema(run, "tail_posture_view")

    assert arrays == tuple(
        sorted((*mod._POSTURE_ARRAYS, *mod._OPTIONAL_REVISION_ARRAYS))
    )
    assert groups == tuple(
        sorted((*mod._RECORD_GROUPS, "source_refined_subject_masks"))
    )


def test_tail_reader_freshly_rechecks_sealed_payload_and_attrs(monkeypatch) -> None:
    root = zarr.group()
    run = root.require_group("analysis/tail_kinematics_runs/tail")
    run.attrs.update(
        {
            "tail_coordinate_publication_kind": "tail_kinematics",
            "coordinate_contract": "canonical_v2",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_subject_shape_path": "analysis/subject_shape_runs/shape",
            "scientific_setting": "v1",
        }
    )
    run.create_array("valid", data=np.asarray([True, False], dtype=bool))

    sentinel = SimpleNamespace()
    monkeypatch.setattr(mod, "_source_publication", lambda *_args: sentinel)
    monkeypatch.setattr(mod, "_identity", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(
        mod,
        "_stamp_or_load_source_authority",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(
        mod,
        "_stamp_or_load_collection_axis",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(
        mod,
        "_stamp_or_load_measurement_collection_axis",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(
        mod,
        "_stamp_or_load_derivation",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(mod, "_descriptor_bindings", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        mod,
        "_source_measurement_inputs",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        mod,
        "_stamp_or_load_measurement_authority",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(mod, "_measurement_bindings", lambda *_args, **_kwargs: {})

    def live_manifest(*_args, **_kwargs):
        return {
            "schema_id": "fixture.tail_manifest",
            "valid_sha256": array_payload_sha256(run["valid"]),
            "scientific_setting": run.attrs["scientific_setting"],
        }

    monkeypatch.setattr(mod, "_manifest_record", live_manifest)
    manifest = stamp_and_bind_persisted_coordinate_record(
        run,
        live_manifest(),
        attr_name=mod.TAIL_PUBLICATION_MANIFEST_ATTR,
    )
    run.attrs[mod.TAIL_PUBLICATION_MANIFEST_ALIAS_ATTR] = manifest.record_sha256

    mod.load_tail_kinematics_coordinate_publication(
        root,
        "analysis/tail_kinematics_runs/tail",
    )

    run["valid"][0] = False
    with pytest.raises(mod.TailCoordinatePublicationError, match="manifest differs"):
        mod.load_tail_kinematics_coordinate_publication(
            root,
            "analysis/tail_kinematics_runs/tail",
        )

    run["valid"][:] = np.asarray([True, False], dtype=bool)
    run.attrs["scientific_setting"] = "v2"
    with pytest.raises(mod.TailCoordinatePublicationError, match="manifest differs"):
        mod.load_tail_kinematics_coordinate_publication(
            root,
            "analysis/tail_kinematics_runs/tail",
        )


def _fixture_record(source: zarr.Group, name: str):
    node = source.require_group(f"fixture_records/{name}")
    return stamp_and_bind_persisted_coordinate_record(
        node,
        {
            "schema_id": f"fixture.{name}",
            "schema_version": 1,
            "name": name,
        },
        attr_name=f"fixture_{name}",
    )


def _canonical_source_publication(root: zarr.Group) -> SimpleNamespace:
    root.attrs.update(
        {
            "recording_id": "recording-1",
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "camera_id": "camera-a",
                "source_path": "/recording/cams/camera-a.mp4",
                "width": 100,
                "height": 80,
                "total_frames": 2,
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "cams/camera-a.mp4",
                },
                "file_fingerprint": {
                    "strategy": "size_mtime_sha256_v1",
                    "value": "a" * 64,
                    "size_bytes": 1234,
                    "mtime_ns": 5678,
                    "relocation_stable": False,
                },
            },
        }
    )
    acquisition_node = root.require_group("analysis/acquisition_camera_frames/camera-a")
    ownership = stamp_acquisition_import_ownership(root, acquisition_node)
    acquisition = stamp_acquisition_camera_frame(
        root,
        acquisition_node,
        import_ownership=ownership,
    )
    frame_node = root.require_group(
        "analysis/coordinate_frames/source_camera/camera-a/continuous"
    )
    frame = stamp_source_camera_pixel_frame_authority(
        frame_node,
        frame_id="camera-a_source_camera_continuous",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )

    source = root.require_group("analysis/subject_shape_runs/shape")
    instance_key = source.create_array(
        "instance_key",
        data=np.asarray([11, 22], dtype=np.uint64),
    )
    source.create_array(
        "source_crop_row_ids",
        data=np.asarray([5, 2], dtype=np.int64),
    )
    source.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([101, 105], dtype=np.int64),
    )
    identity = stamp_and_bind_row_identity_contract(
        source,
        instance_key,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=np.asarray(instance_key[:]),
        ),
    )
    tail_sample_axis = _fixture_record(source, "tail_sample_axis")
    collection = CanonicalCollectionAxis(
        axis=1,
        role="keypoint",
        cardinality=5,
        label_authority=DigestBoundCoordinateRecordRef(
            record_ref=tail_sample_axis.record_ref,
            record_sha256=tail_sample_axis.record_sha256,
        ),
    )
    body = source.require_group("components/subject_body")
    body_frame = source.require_group("body_frame")
    coordinate_arrays = {
        "components/subject_body/tail_sample_xy": body.create_array(
            "tail_sample_xy",
            data=np.zeros((2, 5, 2), dtype=np.float32),
        ),
        "components/subject_body/tail_tangent_xy": body.create_array(
            "tail_tangent_xy",
            data=np.zeros((2, 5, 2), dtype=np.float32),
        ),
        "components/subject_body/tail_base_xy": body.create_array(
            "tail_base_xy",
            data=np.zeros((2, 2), dtype=np.float32),
        ),
        "components/subject_body/centroid_xy": body.create_array(
            "centroid_xy",
            data=np.zeros((2, 2), dtype=np.float32),
        ),
        "body_frame/forward_axis_xy": body_frame.create_array(
            "forward_axis_xy",
            data=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        ),
        "body_frame/left_axis_xy": body_frame.create_array(
            "left_axis_xy",
            data=np.asarray([[0.0, -1.0], [0.0, -1.0]], dtype=np.float32),
        ),
    }
    descriptors = {}
    for path, node in coordinate_arrays.items():
        is_tail_sample = path.endswith("tail_sample_xy")
        is_tail_tangent = path.endswith("tail_tangent_xy")
        is_vector = is_tail_tangent or "axis_xy" in path
        has_tail_sample_axis = is_tail_sample or is_tail_tangent
        descriptors[path] = build_bound_canonical_coordinate_descriptor(
            node,
            profile_id=(
                "source_camera_image_px.unit_vector_y_down.v1"
                if is_vector
                else "source_camera_image_px.top_left_y_down.v1"
            ),
            geometry_type=(
                "vector_sequence_xy"
                if is_tail_tangent
                else "vector_xy" if is_vector else "point_xy"
            ),
            components=("x", "y"),
            component_units=("unitless", "unitless") if is_vector else ("px", "px"),
            pixel_convention="not_applicable" if is_vector else "continuous",
            row_identity=identity,
            reference_frame_authority=frame,
            source_camera_overlay_status=(
                CANONICAL_OVERLAY_NOT_SUITABLE
                if is_vector
                else CANONICAL_OVERLAY_DIRECT
            ),
            lineage_records=(tail_sample_axis,) if has_tail_sample_axis else (),
            collection_axis=collection if is_tail_sample else None,
        )
    stamp_bound_canonical_coordinate_descriptors(descriptors.values())
    curvature_semantics = _fixture_record(source, "tail_curvature_semantics")

    def require_scalar_surface(relative_ref, *, units=None, surface_kind=None):
        assert relative_ref == "components/subject_body/tail_curvature_px_inv"
        assert units == "px^-1"
        assert surface_kind == "row_profile"
        return SimpleNamespace(semantics=curvature_semantics)

    return SimpleNamespace(
        _run=source,
        run_path="analysis/subject_shape_runs/shape",
        row_identity=identity,
        manifest=_fixture_record(source, "manifest"),
        temporal_authority=_fixture_record(source, "temporal_authority"),
        scientific_configuration=_fixture_record(
            source,
            "scientific_configuration",
        ),
        tail_sample_axis=tail_sample_axis,
        derivation=_fixture_record(source, "derivation"),
        body_frame=_fixture_record(source, "body_frame"),
        descriptors=descriptors,
        require_scalar_surface=require_scalar_surface,
    )


def _base_tail_attrs(source: SimpleNamespace) -> dict[str, object]:
    return {
        "source_subject_shape_path": source.run_path,
        "source_subject_shape_publication_manifest_sha256": (
            source.manifest.record_sha256
        ),
        "palette_run_completion_status": "complete",
        "stage_selector_eligible": False,
        mod.TAIL_PUBLICATION_OWNER_ATTR: "11111111-1111-4111-8111-111111111111",
    }


def _kinematics_run(root: zarr.Group, source: SimpleNamespace) -> zarr.Group:
    run = root.require_group("analysis/tail_kinematics_runs/tail_001")
    run.attrs.update(
        {
            **_base_tail_attrs(source),
            "method": "tail_metrics_from_subject_shape",
            "method_version": 1,
            "tail_angle_sample_count": 5,
            "tail_angle_reference_axis": "caudal_axis=-forward_axis",
            "tail_angle_positive_direction": "anatomical_left",
        }
    )
    rows = 2
    samples = 5
    arrays = {
        "instance_key": np.asarray([11, 22], dtype=np.uint64),
        "source_crop_row_ids": np.asarray([5, 2], dtype=np.int64),
        "source_acquisition_frame_index": np.asarray([101, 105], dtype=np.int64),
        "valid": np.asarray([True, True], dtype=bool),
        "failure_reason_bytes": np.zeros((rows, 64), dtype=np.uint8),
        "tail_angle_sample_s": np.linspace(0.0, 1.0, samples, dtype=np.float32),
        "tail_angle_sample_xy": np.zeros((rows, samples, 2), dtype=np.float32),
        "tail_angle_rad": np.zeros((rows, samples), dtype=np.float32),
        "tail_angle_deg": np.zeros((rows, samples), dtype=np.float32),
        "tail_tip_angle_rad": np.zeros((rows,), dtype=np.float32),
        "tail_tip_angle_deg": np.zeros((rows,), dtype=np.float32),
        "tail_lateral_deflection_px": np.zeros((rows, samples), dtype=np.float32),
        "tail_tip_lateral_deflection_px": np.zeros((rows,), dtype=np.float32),
        "max_abs_tail_angle_rad": np.zeros((rows,), dtype=np.float32),
        "max_abs_tail_angle_deg": np.zeros((rows,), dtype=np.float32),
        "tail_angle_rms_rad": np.zeros((rows,), dtype=np.float32),
        "tail_angle_rms_deg": np.zeros((rows,), dtype=np.float32),
        "integrated_abs_tail_angle_rad": np.zeros((rows,), dtype=np.float32),
        "tail_curvature_px_inv": np.zeros((rows, samples), dtype=np.float32),
        "max_abs_tail_curvature_px_inv": np.zeros((rows,), dtype=np.float32),
        "integrated_abs_tail_curvature": np.zeros((rows,), dtype=np.float32),
    }
    for name, values in arrays.items():
        run.create_array(name, data=values)
    return run


def _posture_run(
    root: zarr.Group,
    source: SimpleNamespace,
    *,
    run_name: str = "posture_001",
    owner: str = "11111111-1111-4111-8111-111111111111",
    view_family: str = "canonical_tail_posture",
) -> zarr.Group:
    run = root.require_group(f"analysis/tail_posture_view_runs/{run_name}")
    run.attrs.update(
        {
            **_base_tail_attrs(source),
            mod.TAIL_PUBLICATION_OWNER_ATTR: owner,
            "method": "tail_posture_view_from_subject_shape",
            "method_version": 1,
            "view_family": view_family,
            "head_source": "centroid_xy",
            "keypoint_count": 5,
            "angle_convention": "megabouts_cumulative_segment_angle",
        }
    )
    arrays = {
        "instance_key": np.asarray([11, 22], dtype=np.uint64),
        "source_crop_row_ids": np.asarray([5, 2], dtype=np.int64),
        "source_acquisition_frame_index": np.asarray([101, 105], dtype=np.int64),
        "valid": np.asarray([True, True], dtype=bool),
        "failure_reason_bytes": np.zeros((2, 64), dtype=np.uint8),
        "head_xy": np.zeros((2, 2), dtype=np.float32),
        "head_yaw_rad": np.zeros((2,), dtype=np.float32),
        "tail_keypoints_xy": np.zeros((2, 5, 2), dtype=np.float32),
        "tail_angle_rad": np.zeros((2, 4), dtype=np.float32),
        "tail_angle_deg": np.zeros((2, 4), dtype=np.float32),
    }
    for name, values in arrays.items():
        run.create_array(name, data=values)
    return run


def test_real_tail_publication_binds_descriptors_identity_and_fresh_seals(
    monkeypatch,
) -> None:
    root = zarr.group()
    source = _canonical_source_publication(root)
    monkeypatch.setattr(mod, "_source_publication", lambda *_args: source)

    tail = _kinematics_run(root, source)
    mod.publish_tail_kinematics_coordinate_surfaces(root, tail)
    with pytest.raises(
        mod.TailCoordinatePublicationError,
        match="selector-eligible",
    ):
        mod.load_tail_kinematics_coordinate_publication(
            root,
            "analysis/tail_kinematics_runs/tail_001",
        )
    mod.activate_tail_coordinate_publication(
        root,
        root["analysis/tail_kinematics_runs"],
        tail,
        run_name="tail_001",
        expected_publication_owner_uuid=tail.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR],
    )
    publication = mod.load_tail_kinematics_coordinate_publication(
        root,
        "analysis/tail_kinematics_runs/tail_001",
    )
    assert publication.descriptors["tail_angle_sample_xy"].descriptor.space_id == (
        "source_camera_image_px"
    )
    assert (
        publication.descriptors["tail_angle_sample_xy"].descriptor.collection_axis.role
        == "keypoint"
    )
    assert publication.measurements["tail_angle_rad"].record["units"] == "rad"
    assert publication.measurements["tail_curvature_px_inv"].record["units"] == "px^-1"
    angle_source_refs = {
        item["record_ref"]
        for item in publication.measurements["tail_angle_rad"].record[
            "source_coordinate_descriptors"
        ]
    }
    assert angle_source_refs == {
        f"/{source.run_path}/components/subject_body/tail_tangent_xy@coordinate_descriptor",
        f"/{source.run_path}/body_frame/forward_axis_xy@coordinate_descriptor",
        f"/{source.run_path}/body_frame/left_axis_xy@coordinate_descriptor",
    }
    assert publication.measurements["tail_curvature_px_inv"].record[
        "source_measurement_descriptors"
    ] == [
        {
            "record_ref": source.require_scalar_surface(
                "components/subject_body/tail_curvature_px_inv",
                units="px^-1",
                surface_kind="row_profile",
            ).semantics.record_ref,
            "record_sha256": source.require_scalar_surface(
                "components/subject_body/tail_curvature_px_inv",
                units="px^-1",
                surface_kind="row_profile",
            ).semantics.record_sha256,
        }
    ]
    tail_parent = root["analysis/tail_kinematics_runs"]
    assert tail_parent.attrs[mod.TAIL_PUBLICATION_POLICY_ATTR] == (
        mod.TAIL_PUBLICATION_POLICY
    )
    assert tail_parent.attrs[mod.TAIL_PUBLICATION_GENERATION_ATTR] == 1
    assert (
        tail_parent.attrs[mod.TAIL_PARENT_PUBLICATION_LEASE_ATTR]["owner_uuid"]
        == tail.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR]
    )
    assert (
        publication.manifest.record["publication_owner_uuid"]
        == tail.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR]
    )
    sample_derivation = publication.derivation.record["coordinate_outputs"][
        "tail_angle_sample_xy"
    ]
    assert sample_derivation["operation"] == (
        "linear_interpolation_over_normalized_tail_arclength_v1"
    )
    assert sample_derivation["source_coordinate_descriptor"]["record_ref"] == (
        f"/{source.run_path}/components/subject_body/tail_sample_xy@coordinate_descriptor"
    )
    assert sample_derivation["source_collection_axis"] == {
        "record_ref": source.tail_sample_axis.record_ref,
        "record_sha256": source.tail_sample_axis.record_sha256,
    }
    assert (
        tail["tail_angle_rad"].attrs[mod.ARRAY_MEASUREMENT_DESCRIPTOR_ATTR][
            "collection_axis"
        ]["role"]
        == "keypoint"
    )
    assert np.array_equal(
        tail["instance_key"][:],
        source._run["instance_key"][:],
    )
    assert np.array_equal(
        tail["source_acquisition_frame_index"][:],
        source._run["source_acquisition_frame_index"][:],
    )

    posture_run = _posture_run(root, source)
    mod.publish_tail_posture_coordinate_surfaces(root, posture_run)
    mod.activate_tail_coordinate_publication(
        root,
        root["analysis/tail_posture_view_runs"],
        posture_run,
        run_name="posture_001",
        expected_publication_owner_uuid=posture_run.attrs[
            mod.TAIL_PUBLICATION_OWNER_ATTR
        ],
    )
    posture = mod.load_tail_posture_coordinate_publication(
        root,
        "analysis/tail_posture_view_runs/posture_001",
    )
    assert posture.descriptors["head_xy"].descriptor.geometry_type == "point_xy"
    assert (
        posture.descriptors["tail_keypoints_xy"].descriptor.geometry_type == "point_xy"
    )
    assert (
        posture.descriptors["tail_keypoints_xy"].descriptor.collection_axis.cardinality
        == 5
    )
    assert posture.measurement_collection_axis.record["role"] == "tail_segment"
    assert posture.measurement_collection_axis.record["cardinality"] == 4
    assert (
        posture.derivation.record["coordinate_outputs"]["head_xy"]["operation"]
        == "exact_rowwise_coordinate_copy_v1"
    )
    assert (
        posture.derivation.record["coordinate_outputs"]["tail_keypoints_xy"][
            "operation"
        ]
        == "linear_interpolation_over_normalized_tail_arclength_v1"
    )
    assert posture.measurements["tail_angle_rad"].record["collection_axis"] == {
        "axis": 1,
        "role": "tail_segment",
        "cardinality": 4,
        "label_authority": {
            "record_ref": posture.measurement_collection_axis.record_ref,
            "record_sha256": posture.measurement_collection_axis.record_sha256,
        },
    }
    tampered_measurement = deepcopy(
        posture_run["tail_angle_rad"].attrs[mod.ARRAY_MEASUREMENT_DESCRIPTOR_ATTR]
    )
    tampered_measurement["units"] = "px"
    posture_run["tail_angle_rad"].attrs[
        mod.ARRAY_MEASUREMENT_DESCRIPTOR_ATTR
    ] = tampered_measurement
    with pytest.raises(
        mod.TailCoordinatePublicationError,
        match="Measurement descriptor|invalid",
    ):
        mod.load_tail_posture_coordinate_publication(
            root,
            "analysis/tail_posture_view_runs/posture_001",
        )

    changed = np.asarray(tail["tail_angle_sample_xy"][:], dtype=np.float32)
    changed[0, 0, 0] += 1.0
    tail["tail_angle_sample_xy"][:] = changed
    with pytest.raises(mod.TailCoordinatePublicationError, match="invalid|differs"):
        mod.load_tail_kinematics_coordinate_publication(
            root,
            "analysis/tail_kinematics_runs/tail_001",
        )


def test_tail_activation_preserves_foreign_lease_and_restores_own_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    source = _canonical_source_publication(root)
    monkeypatch.setattr(mod, "_source_publication", lambda *_args: source)
    tail = _kinematics_run(root, source)
    mod.publish_tail_kinematics_coordinate_surfaces(root, tail)
    parent = root["analysis/tail_kinematics_runs"]
    parent.attrs["latest_complete"] = "prior"
    parent.attrs["latest"] = "prior"
    original_write = mod._write_tail_activation_attr
    alien_lease = {
        "schema_id": "alien.lease",
        "schema_version": 1,
        "owner_uuid": "22222222-2222-4222-8222-222222222222",
    }

    def hostile_write(attrs, name, value):
        original_write(attrs, name, value)
        if name == "latest_complete":
            parent.attrs[mod.TAIL_PARENT_PUBLICATION_LEASE_ATTR] = alien_lease

    monkeypatch.setattr(mod, "_write_tail_activation_attr", hostile_write)
    with pytest.raises(
        mod.TailCoordinatePublicationError,
        match="lost exact ownership",
    ):
        mod.activate_tail_coordinate_publication(
            root,
            parent,
            tail,
            run_name="tail_001",
            expected_publication_owner_uuid=tail.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR],
        )

    assert parent.attrs["latest_complete"] == "prior"
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs[mod.TAIL_PARENT_PUBLICATION_LEASE_ATTR] == alien_lease
    assert tail.attrs["stage_selector_eligible"] is False


def test_tail_publication_rejects_contradictory_angle_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    source = _canonical_source_publication(root)
    monkeypatch.setattr(mod, "_source_publication", lambda *_args: source)
    tail = _kinematics_run(root, source)
    tail.attrs["tail_angle_positive_direction"] = "clockwise_y_down"

    with pytest.raises(
        mod.TailCoordinatePublicationError,
        match="contradict.*controlled operation",
    ):
        mod.publish_tail_kinematics_coordinate_surfaces(root, tail)


def test_posture_family_selectors_advance_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    source = _canonical_source_publication(root)
    monkeypatch.setattr(mod, "_source_publication", lambda *_args: source)
    parent = root.require_group("analysis/tail_posture_view_runs")

    first = _posture_run(
        root,
        source,
        run_name="posture_a",
        owner="22222222-2222-4222-8222-222222222222",
        view_family="family_a",
    )
    mod.publish_tail_posture_coordinate_surfaces(root, first)
    mod.activate_tail_coordinate_publication(
        root,
        parent,
        first,
        run_name="posture_a",
        expected_publication_owner_uuid=first.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR],
        additional_selector_attrs=("latest_family_a",),
    )

    second = _posture_run(
        root,
        source,
        run_name="posture_b",
        owner="33333333-3333-4333-8333-333333333333",
        view_family="family_b",
    )
    mod.publish_tail_posture_coordinate_surfaces(root, second)
    parent = root["analysis/tail_posture_view_runs"]
    mod.activate_tail_coordinate_publication(
        root,
        parent,
        second,
        run_name="posture_b",
        expected_publication_owner_uuid=second.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR],
        additional_selector_attrs=("latest_family_b",),
    )

    parent = root["analysis/tail_posture_view_runs"]
    assert parent.attrs["latest"] == "posture_b"
    assert parent.attrs["latest_complete"] == "posture_b"
    assert parent.attrs["latest_family_a"] == "posture_a"
    assert parent.attrs["latest_family_b"] == "posture_b"
    assert parent.attrs[mod.TAIL_PUBLICATION_GENERATION_ATTR] == 2
