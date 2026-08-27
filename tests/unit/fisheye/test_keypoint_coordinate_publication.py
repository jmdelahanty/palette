from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.detection import detect_keypoints_yolo as keypoint_writer_module
import fisheye.shared.keypoint_coordinate_publication as publication_module
import fisheye.shared.zarr.coordinate_successor_authority as successor_authority_module
import fisheye.shared.zarr.sealed_geometry_crop_profile as sealed_crop_module
from fisheye.shared.keypoint_terminal_pixel_evidence import (
    DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE,
)
from fisheye.shared.proof_verification import proof_verification_scope
from fisheye.detection.detect_yolo import (
    _publish_detection_acquisition_mapping,
    _publish_detection_frame_evidence,
)
from fisheye.shared.coordinate_descriptor import COORDINATE_DESCRIPTOR_ATTR
from fisheye.shared.keypoint_coordinate_publication import (
    KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR,
    KEYPOINT_COORDINATE_DERIVATION_ATTR,
    KEYPOINT_LABEL_AUTHORITY_ATTR,
    KEYPOINT_PUBLICATION_GENERATION_ATTR,
    KEYPOINT_PUBLICATION_OWNER_ATTR,
    KEYPOINT_PUBLICATION_POLICY_ATTR,
    KeypointCoordinatePublicationError,
    capture_keypoint_coordinate_publication_checkpoint,
    derive_keypoint_coordinate_batch,
    load_persisted_keypoint_coordinate_context,
    load_persisted_keypoint_coordinate_surfaces,
    load_persisted_keypoint_crop_source,
    model_input_batch_to_roi,
    prepare_keypoint_coordinate_context,
    publish_keypoint_coordinate_surfaces,
    revalidate_keypoint_coordinate_batch_context,
    rollback_keypoint_coordinate_publication,
)
from fisheye.shared.model_input_transform import (
    ModelInputTransform,
    resolve_model_input_transform,
)
from fisheye.shared.pose_model_schema_binding import (
    build_explicit_pose_model_schema_binding,
)
from fisheye.shared.immutable_yolo_storage import validate_immutable_yolo_storage
from fisheye.shared.observation_coordinate_publication import (
    OBSERVATION_ROW_COUNT_ATTR,
    derive_detection_source_camera_geometry,
    publish_crop_observation_geometry,
    publish_crop_roi_bbox_edge_reference_extent,
    publish_detection_backend_result_projection,
    publish_detection_instance_key_derivation,
    publish_detection_observation_cardinality,
    publish_crop_roi_geometry,
    publish_detection_observation_geometry,
)
from fisheye.shared.instance_keys import (
    instance_key_attrs,
    mint_detection_instance_keys,
)
from fisheye.shared.pixel_frame_authority import (
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    stamp_crop_placement_ownership,
    stamp_roi_pixel_frame_authority,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.coordinate_reference import bind_array_reference_extent
from fisheye.shared.directed_transform_chain import resolve_bound_directed_transform_chain
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    stamp_directed_transform_v2,
)
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    stamp_crop_placement_transform_authority,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)
from fisheye.shared.zarr.keypoint_manifest import KeypointPreprocessingReference
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_directed_transform_chain import FakeArray, FakeGroup
from tests.unit.fisheye.test_observation_coordinate_publication import (
    _crop_copy,
    _crop_roi,
    _published_detection,
)


class _RootRegistry:
    def __init__(self, token: object) -> None:
        self.path = "archive_root"
        self.attrs: dict[str, Any] = {}
        self._coordinate_archive_token = token
        self.nodes: dict[str, Any] = {}

    def register(self, node: Any) -> Any:
        self.nodes[node.path] = node
        return node

    def __getitem__(self, path: str) -> Any:
        return self.nodes[path]

    def __delitem__(self, path: str) -> None:
        node = self.nodes.pop(path)
        parent_path, _, name = path.rpartition("/")
        parent = self.nodes.get(parent_path)
        if parent is not None and getattr(parent, "children", {}).get(name) is node:
            del parent.children[name]


def test_complete_coordinate_successor_validates_evidence_then_uses_resolved_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = object()
    root = _RootRegistry(token)
    run = root.register(
        FakeGroup(
            path="keypoints_runs/successor",
            archive_token=token,
        )
    )
    run.attrs["coordinate_successor_historical_crop_adapter"] = {
        "schema_id": "test-adapter"
    }
    resolved_source = object()
    binding = type("Binding", (), {"source": resolved_source})()
    result = object()
    calls: list[tuple[object, ...]] = []

    def rebind(root_node: Any, run_node: Any, *, run_path: str) -> object:
        calls.append(("rebind", root_node, run_node, run_path))
        return binding

    def load_impl(
        root_node: Any,
        run_path: str,
        *,
        require_complete: bool,
        expected_selector_eligible: bool,
        resolved_crop_source: Any | None = None,
    ) -> object:
        calls.append(
            (
                "load",
                root_node,
                run_path,
                require_complete,
                expected_selector_eligible,
                resolved_crop_source,
            )
        )
        return result

    monkeypatch.setattr(
        publication_module,
        "_load_persisted_sealed_crop_successor_binding",
        rebind,
    )
    monkeypatch.setattr(
        publication_module,
        "_load_persisted_keypoint_coordinate_surfaces_impl",
        load_impl,
    )
    loaded = publication_module._load_persisted_keypoint_coordinate_surfaces(
        root,
        "keypoints_runs/successor",
        require_complete=True,
        expected_selector_eligible=False,
    )

    assert loaded is result
    assert calls == [
        ("rebind", root, run, "keypoints_runs/successor"),
        (
            "load",
            root,
            "keypoints_runs/successor",
            True,
            False,
            resolved_source,
        ),
    ]


def test_persisted_successor_reader_resolves_direct_hybrid_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persisted reader must consume the same profile grammar as its writer."""

    token = object()
    root = _RootRegistry(token)
    successor = root.register(
        FakeGroup(path="keypoints_runs/successor", archive_token=token)
    )
    source = root.register(FakeGroup(path="keypoints_runs/source", archive_token=token))
    for name, values in {
        "source_crop_row_ids": np.asarray([0], dtype="<i8"),
        "instance_key": np.asarray([10], dtype="<u8"),
        "source_acquisition_frame_index": np.asarray([0], dtype="<i8"),
        "source_crop_row_signature": np.zeros((1, 32), dtype="uint8"),
    }.items():
        source.children[name] = FakeArray(
            values,
            path=f"keypoints_runs/source/{name}",
            archive_token=token,
        )

    transform_attrs = ModelInputTransform(
        name="identity",
        native_height=384,
        native_width=384,
        model_height=384,
        model_width=384,
    ).to_attrs()
    preprocessing = KeypointPreprocessingReference(
        profile_id=DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE,
        profile_version=1,
        input_mode="numpy_list",
        document={
            "evidence_semantics": "observed_completed_inference_runtime_v1",
            "coordinate_contract_mode": "legacy_noncanonical",
            "observed_input_mode_effective": "numpy-list",
            "observed_runtime": {
                "input_mode_effective": "numpy-list",
                "model_input_transform": transform_attrs,
                "model_input_shape_hw": [384, 384],
                "model_network_input_shape_hw": [384, 384],
                "native_roi_shape_hw": [384, 384],
            },
        },
    )
    source_manifest = {
        "payload_digest": "1" * 64,
        "payload": {
            "logical_content": {"digest": "2" * 64},
            "preprocessing": preprocessing.as_manifest(),
            "source_crop_snapshot": {"run_path": "crop_runs/crop"},
        },
    }
    source.attrs["run_manifest"] = source_manifest

    crop_evidence = {"source_run_path": "keypoints_runs/source"}
    padded = SimpleNamespace(
        record_ref="/successor@padded",
        record_sha256="3" * 64,
        record={"source_crop_adapter": crop_evidence},
    )
    crop = SimpleNamespace(record=crop_evidence)
    bbox = SimpleNamespace(
        record_ref="/successor@bbox",
        record_sha256="4" * 64,
    )
    records = {
        "coordinate_successor_padded_crop_lineage": padded,
        "coordinate_successor_historical_crop_adapter": crop,
        sealed_crop_module.SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR: bbox,
    }
    authority = {
        "payload": {
            "coordinate_records": {
                "padded_crop_lineage": {
                    "record_ref": padded.record_ref,
                    "record_sha256": padded.record_sha256,
                },
                "historical_bbox_normalization": {
                    "record_ref": bbox.record_ref,
                    "record_sha256": bbox.record_sha256,
                },
            },
            "source": {
                "family": "keypoints_runs",
                "run_path": "keypoints_runs/source",
                "manifest_payload_digest": source_manifest["payload_digest"],
                "manifest_document_digest": canonical_json_sha256(source_manifest),
                "logical_content_digest": "2" * 64,
            },
        }
    }
    bound = SimpleNamespace(as_record=lambda: crop_evidence)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        successor_authority_module,
        "load_coordinate_successor_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(
        publication_module,
        "bind_persisted_coordinate_record",
        lambda _run, *, attr_name: records[attr_name],
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.keypoint_manifest.validate_keypoint_run_manifest",
        lambda _manifest: (),
    )
    monkeypatch.setattr(
        publication_module,
        "archive_identity",
        lambda _root: SimpleNamespace(kind="local_store_root", key=("/archive",)),
    )

    def bind_source(**kwargs: object) -> object:
        observed["transform"] = kwargs["model_input_transform"]
        return bound

    monkeypatch.setattr(
        sealed_crop_module,
        "bind_sealed_geometry_crop_successor_source",
        bind_source,
    )
    monkeypatch.setattr(
        sealed_crop_module,
        "load_sealed_geometry_bbox_normalization_from_successor",
        lambda binding, **_kwargs: binding,
    )

    assert (
        publication_module._load_persisted_sealed_crop_successor_binding(
            root,
            successor,
            run_path="keypoints_runs/successor",
        )
        is bound
    )
    assert observed["transform"].to_attrs() == transform_attrs


def test_keypoint_crop_loader_reuses_only_within_one_proof_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = object()
    root = _RootRegistry(token)
    bound = object()
    loads: list[tuple[Any, str]] = []
    checks: list[object] = []

    def load_fresh(root_node: Any, crop_path: str) -> object:
        loads.append((root_node, crop_path))
        return bound

    monkeypatch.setattr(
        publication_module,
        "_load_persisted_keypoint_crop_source_fresh",
        load_fresh,
    )
    monkeypatch.setattr(
        publication_module,
        "_assert_keypoint_crop_source_unchanged",
        checks.append,
    )

    with proof_verification_scope():
        first = publication_module.load_persisted_keypoint_crop_source(
            root,
            "crop_runs/c1",
        )
        second = publication_module.load_persisted_keypoint_crop_source(
            root,
            "crop_runs/c1",
        )
        assert first is second is bound
        assert len(loads) == 1

    assert checks == [bound]
    publication_module.load_persisted_keypoint_crop_source(root, "crop_runs/c1")
    assert len(loads) == 2


class _MutableGroup(FakeGroup):
    def __init__(self, *, path: str, root: _RootRegistry, token: object) -> None:
        super().__init__(path=path, archive_token=token)
        self._root = root
        root.register(self)

    def __contains__(self, name: str) -> bool:
        return name in self.children

    def create_group(self, name: str) -> _MutableGroup:
        if name in self.children:
            raise ValueError(f"duplicate child {name!r}")
        child = _MutableGroup(
            path=f"{self.path}/{name}",
            root=self._root,
            token=self._coordinate_archive_token,
        )
        self.children[name] = child
        return child

    def create_array(
        self,
        name: str,
        *,
        data: Any,
        chunks: tuple[int, ...] | None = None,
        overwrite: bool = False,
        **_kwargs: Any,
    ) -> FakeArray:
        if name in self.children and not overwrite:
            raise ValueError(f"duplicate child {name!r}")
        child = FakeArray(
            data,
            path=f"{self.path}/{name}",
            archive_token=self._coordinate_archive_token,
            chunks=chunks,
        )
        self.children[name] = child
        self._root.register(child)
        return child


def _attach(group: FakeGroup, node: Any) -> None:
    group[node.path.rsplit("/", 1)[-1]] = node


def _artifact(
    *,
    keypoint_labels: tuple[str, ...] = ("swim_bladder", "eye_left"),
) -> dict[str, Any]:
    labels = list(keypoint_labels)
    binding = build_explicit_pose_model_schema_binding(
        model_sha256="a" * 64,
        assertion_id="fixture-reviewed-pose-model",
        skeleton_id="pose_skel_fixture_v1",
        model_kpt_shape=[len(labels), 3],
        keypoint_labels=labels,
        edges=[[index, index + 1] for index in range(len(labels) - 1)],
    )
    return {
        "role": "keypoint_model",
        "path": "/models/pose.pt",
        "fingerprint_scheme": "content_v1",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "mtime_ns": 456,
        "source": "computed",
        "pose_schema_binding": binding,
    }


def _external_world(token: object) -> dict[str, Any]:
    root = FakeGroup(
        path="archive_root",
        archive_token=token,
        attrs={
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
        },
    )
    acquisition_node = FakeGroup(
        path="analysis/acquisition_camera_frames/camera-a",
        archive_token=token,
    )
    ownership = stamp_acquisition_import_ownership(root, acquisition_node)
    acquisition = stamp_acquisition_camera_frame(
        root,
        acquisition_node,
        import_ownership=ownership,
    )
    camera_frame = stamp_source_camera_pixel_frame_authority(
        FakeGroup(
            path="analysis/coordinate_frames/source_camera/camera-a/continuous",
            archive_token=token,
        ),
        frame_id="camera_a_native",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )
    return {
        "archive_token": token,
        "acquisition_frame": acquisition,
        "camera_frame": camera_frame,
    }


def _fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[_RootRegistry, _MutableGroup, Any, FakeArray]:
    token = object()
    world = _external_world(token)
    source = _published_detection(world)
    crop_nodes = _crop_copy(world, source)
    crop = publish_crop_observation_geometry(*crop_nodes, source_geometry=source)
    placements, bbox_roi, ownership, roi_frame, chain, _point_ownership = _crop_roi(
        world,
        crop,
    )
    publish_crop_roi_geometry(
        placements,
        bbox_roi,
        crop_geometry=crop,
        crop_placement_ownership=ownership,
        roi_frame=roi_frame,
        roi_to_source_camera=chain,
    )
    crop_rowset = crop._rowset_node
    for node in crop_nodes[1:]:
        _attach(crop_rowset, node)
    roi_images = crop_rowset["roi_images"]
    bbox_frame_node = roi_frame._authority_node
    for node in (placements, bbox_roi, roi_images):
        _attach(crop_rowset, node)
    crop_rowset.attrs["coordinate_contract"] = "canonical_v2"
    crop_rowset.attrs["crop_storage_mode"] = "materialized"
    crop_rowset.attrs["stage_selector_eligible"] = True
    mark_run_started(crop_rowset, run_name="c1", stage="crop")
    mark_run_complete(crop_rowset)

    root = _RootRegistry(token)
    root.register(crop_rowset)
    root.register(roi_images)
    root.register(bbox_frame_node)
    run_parent = _MutableGroup(path="keypoints_runs", root=root, token=token)
    run = run_parent.create_group("k1")
    run.attrs[KEYPOINT_PUBLICATION_OWNER_ATTR] = "a" * 32
    run.attrs["stage_selector_eligible"] = False
    mark_run_started(run, run_name="k1", stage="keypoints")
    labels = ["swim_bladder", "eye_left"]
    pose_schema = _artifact()["pose_schema_binding"]["pose_schema"]
    run.attrs.update(
        {
            "keypoint_labels": list(labels),
            "keypoint_confidence_labels": list(labels),
            "skeleton_id": "pose_skel_fixture_v1",
            "kpt_shape": [2, 2],
            "model_kpt_shape": [2, 3],
            "pose_schema": pose_schema,
        }
    )

    selected_rows = np.asarray([1, 0], dtype="<i8")
    crop_keys = np.asarray(crop._key_node[:])
    crop_frames = np.asarray(crop._source_frame_index_node[:])
    crop_placements = np.asarray(placements[:])
    source_values = {
        "source_crop_row_ids": selected_rows,
        "instance_key": crop_keys[selected_rows],
        "source_acquisition_frame_index": crop_frames[selected_rows],
        "source_crop_xywh": crop_placements[selected_rows],
    }
    for name, values in source_values.items():
        run.create_array(name, data=values, chunks=(2,) if values.ndim == 1 else (2, 4))

    keypoints_roi = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [np.nan, np.nan]],
        ],
        dtype="<f8",
    )
    bbox_roi_values = np.asarray(
        [[1.0, 2.0, 11.0, 12.0], [5.0, 6.0, 15.0, 16.0]],
        dtype="<f4",
    )
    offsets = source_values["source_crop_xywh"][:, :2]
    keypoints_img = keypoints_roi + offsets[:, None, :]
    bbox_offsets = np.column_stack((offsets, offsets)).astype("<f4")
    bbox_img = bbox_roi_values + bbox_offsets
    normalization = np.asarray([100.0, 80.0], dtype="<f8")
    coordinate_values = {
        "keypoints_roi": keypoints_roi,
        "keypoints_img": keypoints_img,
        "keypoints_norm": keypoints_img / normalization,
        "pose_bbox_xyxy_roi": bbox_roi_values,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_bbox_xyxy_norm": np.asarray(
            bbox_img / np.tile(normalization, 2),
            dtype="<f4",
        ),
    }
    for name, values in coordinate_values.items():
        run.create_array(name, data=values, chunks=values.shape)
    run.create_array(
        "keypoint_confidences",
        data=np.full((2, 2), 0.5, dtype="<f8"),
        chunks=(2, 2),
    )

    monkeypatch.setattr(
        publication_module,
        "load_persisted_crop_observation_geometry",
        lambda root_node, path: crop,
    )
    return root, run, crop, roi_images


def test_canonical_keypoint_publication_binds_exact_crop_preprocessing_and_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, roi_images = _fixture(monkeypatch)

    source = load_persisted_keypoint_crop_source(root, "crop_runs/c1")
    assert source.roi_frame.endpoint.width == 40
    assert source.roi_frame.endpoint.height == 40
    assert source.roi_frame.pixel_convention == "continuous"
    assert source.roi_frame.record_ref == "/crop_runs/c1/roi_images@pixel_frame_authority"
    assert source.bbox_roi_frame.pixel_convention == "pixel_edge_half_open"
    assert source.bbox_roi_frame.record_ref == (
        "/crop_runs/c1/coordinate_frames/roi_bbox_edge@pixel_frame_authority"
    )
    assert roi_images.read_count == 0

    transform = resolve_model_input_transform(
        (40, 40),
        mode="pad_to_size",
        model_hw=(48, 52),
    )
    context = prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=transform,
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )

    model_point = np.asarray([[7.0, 6.0]], dtype="<f8")
    np.testing.assert_array_equal(
        model_input_batch_to_roi(
            model_point,
            context=context,
            output_dtype=np.float64,
        ),
        np.asarray([[1.0, 2.0]], dtype="<f8"),
    )
    with pytest.raises(KeypointCoordinatePublicationError, match="status='complete'"):
        load_persisted_keypoint_coordinate_context(root, "keypoints_runs/k1")
    assert (
        revalidate_keypoint_coordinate_batch_context(
            context,
            row_start=0,
            row_stop=1,
        )
        is context
    )

    pending_surfaces = publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    with pytest.raises(KeypointCoordinatePublicationError, match="status='complete'"):
        load_persisted_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    parent = root["keypoints_runs"]
    parent_snapshot = keypoint_writer_module._snapshot_selected_attrs(
        parent,
        keypoint_writer_module._KEYPOINT_PARENT_SELECTOR_ATTRS,
    )
    root_snapshot = keypoint_writer_module._snapshot_selected_attrs(
        root,
        ("current_keypoint_group_path",),
    )
    parent.attrs["latest_pending"] = "k1"
    mark_run_complete(run)
    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="selector eligibility True",
    ):
        load_persisted_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    fresh_surfaces = (
        publication_module._load_completed_ineligible_keypoint_coordinate_surfaces(
            root,
            "keypoints_runs/k1",
        )
    )
    publication_module._activate_validated_keypoint_coordinate_surfaces(
        root,
        root["keypoints_runs"],
        fresh_surfaces,
        run_name="k1",
        publication_owner_token="a" * 32,
        parent_selector_snapshot=parent_snapshot,
        root_pointer_snapshot=root_snapshot,
    )
    loaded_context = load_persisted_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
    )
    surfaces = load_persisted_keypoint_coordinate_surfaces(
        root,
        "keypoints_runs/k1",
    )

    assert pending_surfaces.derivation.record_sha256 == surfaces.derivation.record_sha256
    assert loaded_context.model_input_transform == transform
    assert loaded_context.model_artifact == _artifact()
    assert run.attrs["coordinate_contract"] == "canonical_v2"
    assert parent.attrs[KEYPOINT_PUBLICATION_GENERATION_ATTR] == 1
    assert parent.attrs[KEYPOINT_PUBLICATION_POLICY_ATTR] == (
        "owner_generation_guarded_selectors_then_eligibility_v1"
    )
    assert parent.attrs[KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR][
        "publication_owner"
    ] == "a" * 32
    assert surfaces.keypoints_roi.descriptor.space_id == "roi_local_px"
    assert surfaces.keypoints_roi.descriptor.geometry_type == "point_xy"
    assert surfaces.keypoints_roi.descriptor.collection_axis is not None
    assert surfaces.keypoints_roi.descriptor.collection_axis.role == "keypoint"
    assert surfaces.keypoints_roi.descriptor.collection_axis.cardinality == 2
    assert loaded_context.keypoint_labels == ("swim_bladder", "eye_left")
    assert (
        surfaces.keypoints_roi.descriptor.collection_axis.label_authority.record_ref
        == loaded_context.keypoint_label_authority.record_ref
    )
    assert run.attrs[KEYPOINT_LABEL_AUTHORITY_ATTR]["axis0"]["role"] == (
        "observation_instance"
    )
    assert surfaces.keypoints_roi.descriptor.source_camera_overlay.status == (
        "requires_transform"
    )
    assert surfaces.keypoints_img.descriptor.space_id == "source_camera_image_px"
    assert surfaces.keypoints_img.descriptor.source_camera_overlay.status == "direct"
    assert surfaces.keypoints_norm.descriptor.space_id == (
        "source_camera_normalized_xy"
    )
    assert surfaces.pose_bbox_xyxy_img.descriptor.geometry_type == "bbox_xyxy"
    assert surfaces.keypoints_roi.descriptor.pixel_convention == "continuous"
    assert surfaces.keypoints_img.descriptor.pixel_convention == "continuous"
    assert (
        surfaces.pose_bbox_xyxy_roi.descriptor.pixel_convention
        == "pixel_edge_half_open"
    )
    assert (
        surfaces.pose_bbox_xyxy_img.descriptor.pixel_convention
        == "pixel_edge_half_open"
    )
    assert (
        surfaces.pose_bbox_xyxy_norm.descriptor.pixel_convention
        == "continuous"
    )
    assert (
        surfaces.keypoints_norm.descriptor.frame_record.record_ref
        == loaded_context.point_normalized_frame.record_ref
    )
    assert (
        surfaces.pose_bbox_xyxy_norm.descriptor.frame_record.record_ref
        != loaded_context.point_normalized_frame.record_ref
    )
    assert surfaces.keypoints_img.descriptor.row_identity.record_ref == (
        surfaces.context.row_identity.record_ref
    )
    assert run["keypoints_img"].attrs[COORDINATE_DESCRIPTOR_ATTR]["schema_version"] == 2


@pytest.mark.parametrize("marker", [True, None])
def test_keypoint_staging_requires_explicit_selector_ineligibility(
    monkeypatch: pytest.MonkeyPatch,
    marker: bool | None,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    if marker is None:
        del run.attrs["stage_selector_eligible"]
    else:
        run.attrs["stage_selector_eligible"] = marker

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="selector eligibility False",
    ):
        prepare_keypoint_coordinate_context(
            root,
            "keypoints_runs/k1",
            crop_path="crop_runs/c1",
            model_input_transform=resolve_model_input_transform(
                (40, 40),
                mode="pad_to_size",
                model_hw=(48, 52),
            ),
            preprocessing_input_mode="numpy-list",
            model_artifact=_artifact(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "keypoint_confidence_labels",
            ["eye_left", "swim_bladder"],
            "keypoint_confidence_labels",
        ),
        ("skeleton_id", "pose_skel_decoy", "pose_schema"),
        ("kpt_shape", [3, 2], "kpt_shape"),
        ("model_kpt_shape", [3, 2], "model_kpt_shape"),
        ("model_kpt_shape", [2, 2, 1], "model_kpt_shape"),
    ],
)
def test_keypoint_preflight_rejects_inconsistent_collection_axis_metadata(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    run.attrs[field] = value

    with pytest.raises(KeypointCoordinatePublicationError, match=message):
        prepare_keypoint_coordinate_context(
            root,
            "keypoints_runs/k1",
            crop_path="crop_runs/c1",
            model_input_transform=resolve_model_input_transform((40, 40)),
            preprocessing_input_mode="numpy-list",
            model_artifact=_artifact(),
        )

    assert KEYPOINT_LABEL_AUTHORITY_ATTR not in run.attrs


def test_keypoint_preflight_rejects_conflicting_nested_skeleton_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    pose_schema = dict(run.attrs["pose_schema"])
    pose_schema["metadata"] = {
        **dict(pose_schema["metadata"]),
        "skeleton_id": "pose_skel_decoy",
    }
    run.attrs["pose_schema"] = pose_schema

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="metadata skeleton_id",
    ):
        prepare_keypoint_coordinate_context(
            root,
            "keypoints_runs/k1",
            crop_path="crop_runs/c1",
            model_input_transform=resolve_model_input_transform((40, 40)),
            preprocessing_input_mode="numpy-list",
            model_artifact=_artifact(),
        )

    assert KEYPOINT_LABEL_AUTHORITY_ATTR not in run.attrs


def test_keypoint_batch_derivation_requires_exact_label_axis_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _run, _crop, _roi_images = _fixture(monkeypatch)
    context = prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="geometry shapes",
    ):
        derive_keypoint_coordinate_batch(
            context=context,
            row_start=0,
            row_stop=1,
            keypoints_roi=np.zeros((1, 3, 2), dtype=np.float64),
            pose_bbox_xyxy_roi=np.zeros((1, 4), dtype=np.float32),
        )


def test_keypoint_publication_rejects_pose_labels_changed_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    run.attrs["keypoint_labels"] = ["eye_left", "swim_bladder"]

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="pose_schema|keypoint_confidence_labels",
    ):
        publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")


def test_keypoint_surface_publication_rejects_roi_image_mixing_transactionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    run["keypoints_img"].data[0, 0, 0] += 1.0

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="exact dtype-preserving declared ROI/image/normalization derivation",
    ):
        publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")

    assert KEYPOINT_COORDINATE_DERIVATION_ATTR not in run.attrs
    assert "coordinate_contract" not in run.attrs
    for name in publication_module.KEYPOINT_ARRAY_NAMES:
        assert COORDINATE_DESCRIPTOR_ATTR not in run[name].attrs


def test_fresh_keypoint_loader_rejects_persisted_payload_or_transform_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    context = prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform(
            (40, 40), mode="pad_to_size", model_hw=(48, 52)
        ),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    run["source_crop_xywh"].data[0, 0] += 1.0
    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="batch source_crop_xywh changed",
    ):
        revalidate_keypoint_coordinate_batch_context(
            context,
            row_start=0,
            row_stop=1,
        )
    run["source_crop_xywh"].data[0, 0] -= 1.0
    publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    mark_run_complete(run)
    run.attrs["stage_selector_eligible"] = True

    run["keypoints_img"].data[0, 0, 0] += 1.0
    with pytest.raises(KeypointCoordinatePublicationError):
        load_persisted_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    run["keypoints_img"].data[0, 0, 0] -= 1.0

    matrix = root["keypoints_runs/k1/coordinate_transforms/model_input_to_roi"]
    matrix.data[0, 2] += 1.0
    with pytest.raises(Exception, match="transform|matrix|record|stale"):
        load_persisted_keypoint_coordinate_context(root, "keypoints_runs/k1")


def test_keypoint_context_rejects_mixed_crop_identity_and_reference_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    run["instance_key"].data[[0, 1]] = run["instance_key"].data[[1, 0]]

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="exact dtype-preserving crop subset/reorder",
    ):
        prepare_keypoint_coordinate_context(
            root,
            "keypoints_runs/k1",
            crop_path="crop_runs/c1",
            model_input_transform=resolve_model_input_transform((40, 40)),
            preprocessing_input_mode="numpy-list",
            model_artifact=_artifact(),
        )

    run["instance_key"].data[[0, 1]] = run["instance_key"].data[[1, 0]]
    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="native extent differs from the exact selected crop ROI frame",
    ):
        prepare_keypoint_coordinate_context(
            root,
            "keypoints_runs/k1",
            crop_path="crop_runs/c1",
            model_input_transform=resolve_model_input_transform((39, 40)),
            preprocessing_input_mode="numpy-list",
            model_artifact=_artifact(),
        )


def test_keypoint_eligibility_flip_is_literal_final_activation_action() -> None:
    source = textwrap.dedent(
        inspect.getsource(
            publication_module._activate_validated_keypoint_coordinate_surfaces
        )
    )
    function = ast.parse(source).body[0]
    final_action = function.body[-1]

    assert isinstance(final_action, ast.Assign)
    target = final_action.targets[0]
    assert isinstance(target, ast.Subscript)
    assert ast.unparse(target.value) == "activation_run.attrs"
    assert ast.literal_eval(target.slice) == "stage_selector_eligible"
    assert isinstance(final_action.value, ast.Constant)
    assert final_action.value.value is True


def test_keypoint_activation_rechecks_lease_before_selector_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    parent = root["keypoints_runs"]
    parent_snapshot = keypoint_writer_module._snapshot_selected_attrs(
        parent,
        keypoint_writer_module._KEYPOINT_PARENT_SELECTOR_ATTRS,
    )
    root_snapshot = keypoint_writer_module._snapshot_selected_attrs(
        root,
        ("current_keypoint_group_path",),
    )
    parent.attrs["latest_pending"] = "k1"
    mark_run_complete(run)
    surfaces = (
        publication_module._load_completed_ineligible_keypoint_coordinate_surfaces(
            root,
            "keypoints_runs/k1",
        )
    )
    acquire = publication_module._acquire_keypoint_parent_publication_lease

    def replace_after_acquire(*args: Any, **kwargs: Any) -> dict[str, Any]:
        lease = acquire(*args, **kwargs)
        parent.attrs[KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR] = {
            **lease,
            "publication_owner": "e" * 32,
        }
        return lease

    monkeypatch.setattr(
        publication_module,
        "_acquire_keypoint_parent_publication_lease",
        replace_after_acquire,
    )

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="lease was replaced",
    ):
        publication_module._activate_validated_keypoint_coordinate_surfaces(
            root,
            parent,
            surfaces,
            run_name="k1",
            publication_owner_token="a" * 32,
            parent_selector_snapshot=parent_snapshot,
            root_pointer_snapshot=root_snapshot,
        )

    assert "latest_complete" not in parent.attrs
    assert "latest" not in parent.attrs
    assert "current_keypoint_group_path" not in root.attrs
    assert run.attrs["stage_selector_eligible"] is False


@pytest.mark.parametrize("takeover", ["foreign_lease", "advanced_generation"])
def test_keypoint_attempt_rollback_preserves_concurrent_selector_takeover(
    monkeypatch: pytest.MonkeyPatch,
    takeover: str,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    parent = root["keypoints_runs"]
    parent.attrs.update({"latest": "prior", "latest_complete": "prior"})
    root.attrs["current_keypoint_group_path"] = "keypoints_runs/prior"
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    boundary = keypoint_writer_module._KeypointAttemptFailureBoundary()
    boundary.prepare(root=root, parent=parent)
    assert boundary.owner_token is not None
    run.attrs[KEYPOINT_PUBLICATION_OWNER_ATTR] = boundary.owner_token
    boundary.bind_run(run, "k1")
    parent.attrs["latest_pending"] = "k1"
    checkpoint = capture_keypoint_coordinate_publication_checkpoint(
        root,
        "keypoints_runs/k1",
        expected_publication_owner=boundary.owner_token,
    )
    boundary.bind_coordinate_checkpoint(checkpoint)
    publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")

    lease = publication_module._keypoint_publication_lease_record(
        run_path="keypoints_runs/k1",
        publication_owner=boundary.owner_token,
        base_generation=0,
    )
    if takeover == "foreign_lease":
        lease["run_path"] = "keypoints_runs/winner"
        lease["publication_owner"] = "e" * 32
        generation = 1
    else:
        generation = 2
    parent.attrs.update(
        {
            KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR: lease,
            KEYPOINT_PUBLICATION_POLICY_ATTR: (
                "owner_generation_guarded_selectors_then_eligibility_v1"
            ),
            KEYPOINT_PUBLICATION_GENERATION_ATTR: generation,
            "latest": "winner",
            "latest_complete": "winner",
            "latest_pending": "winner",
        }
    )
    root.attrs["current_keypoint_group_path"] = "keypoints_runs/winner"

    boundary.fail(RuntimeError("synthetic concurrent takeover"))

    assert parent.attrs["latest"] == "winner"
    assert parent.attrs["latest_complete"] == "winner"
    assert parent.attrs["latest_pending"] == "winner"
    assert root.attrs["current_keypoint_group_path"] == "keypoints_runs/winner"
    assert parent.attrs[KEYPOINT_PUBLICATION_GENERATION_ATTR] == generation
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "failed"
    assert run.attrs["stage_selector_eligible"] is False


def test_keypoint_loaders_reject_incomplete_or_failed_coordinate_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    crop = root["crop_runs/c1"]
    crop.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_RUNNING
    with pytest.raises(KeypointCoordinatePublicationError, match="status='complete'"):
        load_persisted_keypoint_crop_source(root, "crop_runs/c1")

    crop.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    mark_run_failed(run, error="synthetic failure")
    with pytest.raises(KeypointCoordinatePublicationError, match="status='complete'"):
        load_persisted_keypoint_coordinate_context(root, "keypoints_runs/k1")
    with pytest.raises(KeypointCoordinatePublicationError, match="status='complete'"):
        load_persisted_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")


def test_keypoint_context_creation_rolls_back_all_new_evidence_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)

    def _fail_matrix(*_args: Any, **_kwargs: Any) -> tuple[Any, bool]:
        raise RuntimeError("synthetic matrix creation failure")

    monkeypatch.setattr(publication_module, "_create_matrix", _fail_matrix)
    with pytest.raises(RuntimeError, match="synthetic matrix creation failure"):
        prepare_keypoint_coordinate_context(
            root,
            "keypoints_runs/k1",
            crop_path="crop_runs/c1",
            model_input_transform=resolve_model_input_transform((40, 40)),
            preprocessing_input_mode="numpy-list",
            model_artifact=_artifact(),
        )

    assert "coordinate_frames" not in run
    assert "coordinate_transforms" not in run
    assert not any(path.startswith("keypoints_runs/k1/coordinate_") for path in root.nodes)


def test_keypoint_context_keyboard_interrupt_rolls_back_all_new_evidence_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)

    def _interrupt_matrix(*_args: Any, **_kwargs: Any) -> tuple[Any, bool]:
        raise KeyboardInterrupt("synthetic context interrupt")

    monkeypatch.setattr(publication_module, "_create_matrix", _interrupt_matrix)
    with pytest.raises(KeyboardInterrupt, match="synthetic context interrupt"):
        prepare_keypoint_coordinate_context(
            root,
            "keypoints_runs/k1",
            crop_path="crop_runs/c1",
            model_input_transform=resolve_model_input_transform((40, 40)),
            preprocessing_input_mode="numpy-list",
            model_artifact=_artifact(),
        )

    assert "coordinate_frames" not in run
    assert "coordinate_transforms" not in run
    assert not any(path.startswith("keypoints_runs/k1/coordinate_") for path in root.nodes)


def test_keypoint_publication_checkpoint_removes_partial_or_successful_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    checkpoint = capture_keypoint_coordinate_publication_checkpoint(
        root,
        "keypoints_runs/k1",
    )
    publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    rollback_keypoint_coordinate_publication(checkpoint)

    assert KEYPOINT_COORDINATE_DERIVATION_ATTR not in run.attrs
    assert "coordinate_contract" not in run.attrs
    for name in publication_module.KEYPOINT_ARRAY_NAMES:
        assert COORDINATE_DESCRIPTOR_ATTR not in run[name].attrs


def test_keypoint_attempt_boundary_rolls_back_late_descriptor_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    parent = root["keypoints_runs"]
    parent.attrs.update(
        {
            "latest": "prior",
            "latest_complete": "prior",
            "latest_pending": "prior_pending",
        }
    )
    root.attrs["current_keypoint_group_path"] = "keypoints_runs/prior"
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    boundary = keypoint_writer_module._KeypointAttemptFailureBoundary()
    boundary.prepare(root=root, parent=parent)
    run.attrs[KEYPOINT_PUBLICATION_OWNER_ATTR] = boundary.owner_token
    boundary.bind_run(run, "k1")
    parent.attrs["latest_pending"] = "k1"
    boundary.bind_coordinate_checkpoint(
        capture_keypoint_coordinate_publication_checkpoint(
            root,
            "keypoints_runs/k1",
        )
    )
    publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")

    boundary.fail(RuntimeError("synthetic post-publication failure"))

    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "failed"
    assert KEYPOINT_COORDINATE_DERIVATION_ATTR not in run.attrs
    assert "coordinate_contract" not in run.attrs
    for name in publication_module.KEYPOINT_ARRAY_NAMES:
        assert COORDINATE_DESCRIPTOR_ATTR not in run[name].attrs
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs["latest_complete"] == "prior"
    assert parent.attrs["latest_pending"] == "prior_pending"
    assert root.attrs["current_keypoint_group_path"] == "keypoints_runs/prior"


def test_keypoint_surface_publication_rolls_back_a_mid_stamp_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    original_stamp = publication_module.stamp_bound_canonical_coordinate_descriptors

    def _partially_stamp_then_fail(bindings: Any) -> None:
        descriptors = tuple(bindings)
        original_stamp(descriptors[:1])
        raise RuntimeError("synthetic descriptor stamp failure")

    monkeypatch.setattr(
        publication_module,
        "stamp_bound_canonical_coordinate_descriptors",
        _partially_stamp_then_fail,
    )
    with pytest.raises(RuntimeError, match="synthetic descriptor stamp failure"):
        publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")

    assert KEYPOINT_COORDINATE_DERIVATION_ATTR not in run.attrs
    assert "coordinate_contract" not in run.attrs
    for name in publication_module.KEYPOINT_ARRAY_NAMES:
        assert COORDINATE_DESCRIPTOR_ATTR not in run[name].attrs


def test_keypoint_surface_publication_keyboard_interrupt_rolls_back_partial_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    original_stamp = publication_module.stamp_bound_canonical_coordinate_descriptors

    def _partially_stamp_then_interrupt(bindings: Any) -> None:
        descriptors = tuple(bindings)
        original_stamp(descriptors[:1])
        raise KeyboardInterrupt("synthetic publication interrupt")

    monkeypatch.setattr(
        publication_module,
        "stamp_bound_canonical_coordinate_descriptors",
        _partially_stamp_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt, match="synthetic publication interrupt"):
        publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")

    assert KEYPOINT_COORDINATE_DERIVATION_ATTR not in run.attrs
    assert "coordinate_contract" not in run.attrs
    for name in publication_module.KEYPOINT_ARRAY_NAMES:
        assert COORDINATE_DESCRIPTOR_ATTR not in run[name].attrs


@pytest.mark.parametrize(
    "bbox",
    (
        np.asarray([5.0, 2.0, 4.0, 12.0], dtype="<f4"),
        np.asarray([1.0, 2.0, 41.0, 12.0], dtype="<f4"),
    ),
)
def test_keypoint_publication_rejects_invalid_half_open_bbox_edges(
    monkeypatch: pytest.MonkeyPatch,
    bbox: np.ndarray,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    run["pose_bbox_xyxy_roi"].data[0] = bbox
    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="positive half-open edge boxes",
    ):
        publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")


@pytest.mark.parametrize("invalid_x", (-0.01, 40.0))
def test_keypoint_publication_rejects_points_outside_continuous_roi_domain(
    monkeypatch: pytest.MonkeyPatch,
    invalid_x: float,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    run["keypoints_roi"].data[0, 0] = np.asarray(
        [invalid_x, 2.0],
        dtype="<f8",
    )
    placement = np.asarray(run["source_crop_xywh"].data[0, :2], dtype="<f8")
    image_point = run["keypoints_roi"].data[0, 0] + placement
    run["keypoints_img"].data[0, 0] = image_point
    run["keypoints_norm"].data[0, 0] = image_point / np.asarray(
        [100.0, 80.0],
        dtype="<f8",
    )

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="continuous ROI point domain",
    ):
        publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")


def test_keypoint_publication_accepts_exact_half_open_bbox_extent_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _crop, _roi_images = _fixture(monkeypatch)
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    bbox_roi = np.asarray([0.0, 0.0, 40.0, 40.0], dtype="<f4")
    offset = np.asarray(run["source_crop_xywh"].data[0, :2], dtype="<f4")
    bbox_img = bbox_roi + np.tile(offset, 2)
    run["pose_bbox_xyxy_roi"].data[0] = bbox_roi
    run["pose_bbox_xyxy_img"].data[0] = bbox_img
    run["pose_bbox_xyxy_norm"].data[0] = np.asarray(
        bbox_img / np.asarray([100.0, 80.0, 100.0, 80.0], dtype="<f4"),
        dtype="<f4",
    )

    surfaces = publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")

    assert (
        surfaces.pose_bbox_xyxy_roi.descriptor.pixel_convention
        == "pixel_edge_half_open"
    )


def _real_canonical_archive(
    tmp_path: Any,
    *,
    include_bilateral_eyes: bool = False,
    selected_crop_rows: np.ndarray | None = None,
) -> tuple[Any, Any]:
    root = zarr.open_group(str(tmp_path / "canonical.zarr"), mode="w")
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
    acquisition_node = root.require_group(
        "analysis/acquisition_camera_frames/camera-a"
    )
    acquisition_ownership = stamp_acquisition_import_ownership(
        root,
        acquisition_node,
    )
    acquisition = stamp_acquisition_camera_frame(
        root,
        acquisition_node,
        import_ownership=acquisition_ownership,
    )

    detect = root.require_group("detect_runs").create_group("d1")
    evidence_result = _publish_detection_frame_evidence(
        root,
        detect,
        acquisition_frame=acquisition,
    )
    evidence = (
        evidence_result[0]
        if isinstance(evidence_result, tuple)
        else evidence_result
    )
    frame_indices = np.asarray([0, 1], dtype="<i4")
    detect.create_array(
        "frame_indices", data=frame_indices
    )
    source_frames = detect.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([0, 1], dtype="<i8"),
    )
    normalized_values = np.asarray(
        [[0.25, 0.50, 0.20, 0.25], [0.75, 0.25, 0.10, 0.20]],
        dtype="<f8",
    )
    bbox_values, center_values = derive_detection_source_camera_geometry(
        normalized_values,
        frame_evidence=evidence,
    )
    class_ids_values = np.zeros((2,), dtype="<i4")
    instance_key = detect.create_array(
        "instance_key",
        data=mint_detection_instance_keys(
            recording_identity=acquisition.record.recording_id,
            frame_indices=np.asarray(source_frames[:], dtype="<i8"),
            bbox_norm_coords=normalized_values,
            class_ids=class_ids_values,
        ),
    )
    bbox_norm = detect.create_array("bbox_norm_coords", data=normalized_values)
    bbox_img = detect.create_array("bbox_img_xyxy", data=bbox_values)
    centers = detect.create_array("centers_img_xy", data=center_values)
    class_ids = detect.create_array("class_ids", data=class_ids_values)
    detect.create_array("scores", data=np.ones((2,), dtype="<f4"))
    frame_counts = np.ones((2,), dtype="<i4")
    detect.create_array("frame_counts", data=frame_counts)
    detect.create_array("n_detections", data=frame_counts)
    mapping = _publish_detection_acquisition_mapping(
        detect,
        acquisition_frame=acquisition,
    )
    dense_mapping = np.arange(2, dtype="<i8")
    detect.attrs.update(
        {
            **instance_key_attrs(
                acquisition.record.recording_id,
                frame_domain="recording_parent_frame_index",
                frame_mapping_source=(
                    f"{acquisition.record_ref}#"
                    "full_untrimmed_video_decode_identity_v1"
                ),
                frame_mapping_sha256=hashlib.sha256(
                    np.ascontiguousarray(dense_mapping).view(np.uint8)
                ).hexdigest(),
            ),
            OBSERVATION_ROW_COUNT_ATTR: 2,
            "summary_statistics": {
                "total_detections": 2,
                "frames_with_detections": 2,
                "frames_with_zero_detections": 0,
                "frames_with_multiple_detections": 0,
            },
            "detect_storage_layout": "regular_chunks_v1",
            "detect_storage_policy": "explicit_regular_chunks_override",
            "detect_row_shard_rows": None,
            "detect_frame_shard_rows": None,
            "detect_shard_write": None,
            "model_path": "/models/detect.pt",
            "model_name": "detect.pt",
            "inference_height": 80,
            "inference_width": 100,
            "validated_backend_result_count": 2,
            "validated_backend_result_orig_shape_hw": [80, 100],
            "decode_backend_effective": "opencv",
            "video_reader_type": "opencv",
            "parameters": {
                "decode_backend_effective": "opencv",
                "resize_dims": [80, 100],
                "pre_resize_dims": [80, 100],
                "effective_input_resize_dims": [80, 100],
                "tensor_resize_dims": None,
                "imgsz_applied": [80, 100],
            },
        }
    )
    validate_immutable_yolo_storage(
        detect,
        stage="detect",
        row_shard_rows=None,
        frame_shard_rows=None,
    )
    instance_key_derivation = publish_detection_instance_key_derivation(
        detect,
        instance_key,
        source_frames,
        bbox_norm,
        class_ids,
        acquisition_frame=acquisition,
        acquisition_mapping=mapping,
    )
    backend_projection = publish_detection_backend_result_projection(
        detect,
        bbox_norm,
        frame_evidence=evidence,
        model_artifact={
            "role": "detect_model",
            "path": "/models/detect.pt",
            "fingerprint_scheme": "content_v1",
            "sha256": "b" * 64,
            "size_bytes": 321,
            "mtime_ns": 654,
            "source": "computed",
        },
    )
    publish_detection_observation_cardinality(
        detect,
        acquisition_frame=acquisition,
    )
    detection = publish_detection_observation_geometry(
        detect,
        instance_key,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
        frame_evidence=evidence,
        source_lineage_records=(
            mapping,
            backend_projection,
            instance_key_derivation,
        ),
    )
    detect.attrs["coordinate_contract"] = "canonical_v2"
    detect.attrs["stage_selector_eligible"] = True
    mark_run_started(detect, run_name="d1", stage="detect")
    mark_run_complete(detect)

    crop = root.require_group("crop_runs").create_group("c1")
    crop_rows_values = np.asarray([1, 0], dtype="<i8")
    crop_rows = crop.create_array("detection_indices", data=crop_rows_values)
    crop_key = crop.create_array(
        "instance_key",
        data=np.asarray(instance_key[:])[crop_rows_values],
    )
    crop_time = crop.create_array(
        "source_acquisition_frame_index",
        data=np.asarray(source_frames[:])[crop_rows_values],
    )
    crop_norm = crop.create_array(
        "bbox_norm_coords",
        data=np.asarray(bbox_norm[:])[crop_rows_values],
    )
    crop_img = crop.create_array(
        "bbox_img_xyxy",
        data=np.asarray(bbox_img[:])[crop_rows_values],
    )
    crop_centers = crop.create_array(
        "centers_img_xy",
        data=np.asarray(centers[:])[crop_rows_values],
    )
    crop_geometry = publish_crop_observation_geometry(
        crop,
        crop_key,
        crop_rows,
        crop_time,
        crop_norm,
        crop_img,
        crop_centers,
        source_geometry=detection,
    )
    placement_values = np.asarray(
        [[60.0, 0.0, 40.0, 40.0], [0.0, 20.0, 40.0, 40.0]],
        dtype="<f8",
    )
    placement = crop.create_array("source_crop_xywh", data=placement_values)
    crop.create_array(
        "roi_coordinates_full",
        data=np.asarray(placement_values[:, :2], dtype="<i4"),
    )
    crop.create_array(
        "frame_indices",
        data=np.asarray(crop_time[:], dtype="<i8"),
    )
    roi_images = crop.create_array(
        "roi_images",
        data=np.zeros((2, 40, 40), dtype="u1"),
    )
    point_ownership = stamp_crop_placement_ownership(
        placement,
        row_identity=crop_geometry.row_identity,
        source_camera_frame=evidence.source_camera_frame,
    )
    point_roi_frame = stamp_roi_pixel_frame_authority(
        bind_array_reference_extent(roi_images, units="px"),
        frame_id="c1_roi_continuous",
        pixel_convention="continuous",
        crop_placement_ownership=point_ownership,
    )
    point_authority = stamp_crop_placement_transform_authority(
        placement,
        authority_id="c1_roi_continuous_to_source_camera",
        source_frame=point_roi_frame,
        target_frame=evidence.source_camera_frame,
    )
    stamp_directed_transform_v2(
        placement,
        transform_id="c1_roi_continuous_to_source_camera",
        authority=point_authority,
        source_frame=point_roi_frame,
        target_frame=evidence.source_camera_frame,
        row_identity=crop_geometry.row_identity,
    )
    ownership = stamp_crop_placement_ownership(
        placement,
        row_identity=crop_geometry.row_identity,
        source_camera_frame=evidence.bbox_source_camera_frame,
        attr_name=CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    )
    bbox_frame_node = crop.require_group("coordinate_frames").require_group(
        "roi_bbox_edge"
    )
    roi_frame = stamp_roi_pixel_frame_authority(
        publish_crop_roi_bbox_edge_reference_extent(
            bbox_frame_node,
            roi_images,
        ),
        frame_id="c1_roi_bbox_edge",
        pixel_convention="pixel_edge_half_open",
        crop_placement_ownership=ownership,
    )
    authority = stamp_crop_placement_transform_authority(
        placement,
        authority_id="c1_roi_to_source_camera",
        source_frame=roi_frame,
        target_frame=evidence.bbox_source_camera_frame,
        attr_name=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    )
    link = stamp_directed_transform_v2(
        placement,
        transform_id="c1_roi_to_source_camera",
        authority=authority,
        source_frame=roi_frame,
        target_frame=evidence.bbox_source_camera_frame,
        row_identity=crop_geometry.row_identity,
        attr_name=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    )
    offsets = np.column_stack(
        (
            placement_values[:, 0],
            placement_values[:, 1],
            placement_values[:, 0],
            placement_values[:, 1],
        )
    )
    bbox_roi = crop.create_array(
        "bbox_roi_xyxy",
        data=np.asarray(crop_img[:]) - offsets,
    )
    publish_crop_roi_geometry(
        placement,
        bbox_roi,
        crop_geometry=crop_geometry,
        crop_placement_ownership=ownership,
        roi_frame=roi_frame,
        roi_to_source_camera=resolve_bound_directed_transform_chain((link,)),
    )
    crop.attrs["coordinate_contract"] = "canonical_v2"
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["source_detect_run"] = "d1"
    crop.attrs["roi_size"] = [40, 40]
    crop.attrs["stage_selector_eligible"] = True
    root["crop_runs"].attrs["latest"] = "c1"
    root["crop_runs"].attrs["latest_complete"] = "c1"
    mark_run_started(crop, run_name="c1", stage="crop")
    mark_run_complete(crop)

    run_parent = root.require_group("keypoints_runs")
    run = run_parent.create_group("k1")
    run.attrs[KEYPOINT_PUBLICATION_OWNER_ATTR] = "a" * 32
    run.attrs["stage_selector_eligible"] = False
    mark_run_started(run, run_name="k1", stage="keypoints")
    labels = ["swim_bladder", "eye_left"]
    if include_bilateral_eyes:
        labels.append("eye_right")
    artifact = _artifact(keypoint_labels=tuple(labels))
    pose_schema = artifact["pose_schema_binding"]["pose_schema"]
    run.attrs.update(
        {
            "keypoint_labels": list(labels),
            "keypoint_confidence_labels": list(labels),
            "skeleton_id": "pose_skel_fixture_v1",
            "kpt_shape": [len(labels), 2],
            "model_kpt_shape": [len(labels), 3],
            "pose_schema": pose_schema,
        }
    )
    selected = np.asarray(
        [1, 0] if selected_crop_rows is None else selected_crop_rows,
        dtype="<i8",
    )
    run.create_array("source_crop_row_ids", data=selected)
    run.create_array("instance_key", data=np.asarray(crop_key[:])[selected])
    run.create_array(
        "source_acquisition_frame_index",
        data=np.asarray(crop_time[:])[selected],
    )
    output_placement = np.asarray(placement[:])[selected]
    run.create_array("source_crop_xywh", data=output_placement)
    if include_bilateral_eyes:
        keypoints_roi_values = [
            [[1.0, 2.0], [3.0, 4.0], [8.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0], [10.0, 8.0]],
        ]
    else:
        keypoints_roi_values = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    keypoints_roi = np.asarray(
        keypoints_roi_values,
        dtype="<f8",
    )
    keypoints_img = keypoints_roi + output_placement[:, None, :2]
    norm = np.asarray([100.0, 80.0], dtype="<f8")
    bbox_roi_values = np.asarray(
        [[1.0, 2.0, 11.0, 12.0], [5.0, 6.0, 15.0, 16.0]],
        dtype="<f4",
    )
    bbox_offsets = np.column_stack(
        (
            output_placement[:, 0],
            output_placement[:, 1],
            output_placement[:, 0],
            output_placement[:, 1],
        )
    ).astype("<f4")
    bbox_img_values = bbox_roi_values + bbox_offsets
    values = {
        "keypoints_roi": keypoints_roi,
        "keypoints_img": keypoints_img,
        "keypoints_norm": keypoints_img / norm,
        "pose_bbox_xyxy_roi": bbox_roi_values,
        "pose_bbox_xyxy_img": bbox_img_values,
        "pose_bbox_xyxy_norm": np.asarray(
            bbox_img_values / np.tile(norm, 2), dtype="<f4"
        ),
    }
    for name, value in values.items():
        run.create_array(name, data=value)
    run.create_array(
        "keypoint_confidences",
        data=np.full((2, len(labels)), 0.5, dtype="<f8"),
    )
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=artifact,
    )
    publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    run = root["keypoints_runs/k1"]
    parent_snapshot = keypoint_writer_module._snapshot_selected_attrs(
        run_parent,
        keypoint_writer_module._KEYPOINT_PARENT_SELECTOR_ATTRS,
    )
    root_snapshot = keypoint_writer_module._snapshot_selected_attrs(
        root,
        ("current_keypoint_group_path",),
    )
    run_parent.attrs["latest_pending"] = "k1"
    mark_run_complete(run)
    fresh = publication_module._load_completed_ineligible_keypoint_coordinate_surfaces(
        root,
        "keypoints_runs/k1",
    )
    publication_module._activate_validated_keypoint_coordinate_surfaces(
        root,
        run_parent,
        fresh,
        run_name="k1",
        publication_owner_token="a" * 32,
        parent_selector_snapshot=parent_snapshot,
        root_pointer_snapshot=root_snapshot,
    )
    return root, run


def test_real_zarr_fresh_loader_rejects_persisted_keypoint_tampering(tmp_path: Any) -> None:
    root, run = _real_canonical_archive(tmp_path)
    run["keypoints_img"][0, 0, 0] += 1.0

    reopened = zarr.open_group(str(tmp_path / "canonical.zarr"), mode="r+")
    with pytest.raises(KeypointCoordinatePublicationError):
        load_persisted_keypoint_coordinate_surfaces(reopened, "keypoints_runs/k1")
