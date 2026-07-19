from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.detection import detect_keypoints_yolo as keypoint_writer_module
import fisheye.shared.keypoint_coordinate_publication as publication_module
from fisheye.detection.detect_yolo import (
    _publish_detection_acquisition_mapping,
    _publish_detection_frame_evidence,
)
from fisheye.shared.coordinate_descriptor import COORDINATE_DESCRIPTOR_ATTR
from fisheye.shared.keypoint_coordinate_publication import (
    KEYPOINT_COORDINATE_DERIVATION_ATTR,
    KeypointCoordinatePublicationError,
    capture_keypoint_coordinate_publication_checkpoint,
    load_persisted_keypoint_coordinate_context,
    load_persisted_keypoint_coordinate_surfaces,
    load_persisted_keypoint_crop_source,
    model_input_batch_to_roi,
    prepare_keypoint_coordinate_context,
    publish_keypoint_coordinate_surfaces,
    revalidate_keypoint_coordinate_batch_context,
    rollback_keypoint_coordinate_publication,
)
from fisheye.shared.model_input_transform import resolve_model_input_transform
from fisheye.shared.observation_coordinate_publication import (
    derive_detection_source_camera_geometry,
    publish_crop_observation_geometry,
    publish_crop_roi_geometry,
    publish_detection_observation_geometry,
)
from fisheye.shared.pixel_frame_authority import (
    stamp_crop_placement_ownership,
    stamp_roi_pixel_frame_authority,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.coordinate_reference import bind_array_reference_extent
from fisheye.shared.directed_transform_chain import resolve_bound_directed_transform_chain
from fisheye.shared.directed_transform_v2 import stamp_directed_transform_v2
from fisheye.shared.transform_authority import (
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


def _artifact() -> dict[str, Any]:
    return {
        "role": "keypoint_model",
        "path": "/models/pose.pt",
        "fingerprint_scheme": "content_v1",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "mtime_ns": 456,
        "source": "computed",
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
    return {"archive_token": token, "camera_frame": camera_frame}


def _fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[_RootRegistry, _MutableGroup, Any, FakeArray]:
    token = object()
    world = _external_world(token)
    source = _published_detection(world)
    crop_nodes = _crop_copy(world, source)
    crop = publish_crop_observation_geometry(*crop_nodes, source_geometry=source)
    placements, bbox_roi, ownership, roi_frame, chain = _crop_roi(world, crop)
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
    roi_images = roi_frame._authority_node
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
    run = _MutableGroup(path="keypoints_runs/k1", root=root, token=token)
    run.attrs["stage_selector_eligible"] = True
    mark_run_started(run, run_name="k1", stage="keypoints")

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
    mark_run_complete(run)
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
    assert surfaces.keypoints_roi.descriptor.space_id == "roi_local_px"
    assert surfaces.keypoints_roi.descriptor.source_camera_overlay.status == (
        "requires_transform"
    )
    assert surfaces.keypoints_img.descriptor.space_id == "source_camera_image_px"
    assert surfaces.keypoints_img.descriptor.source_camera_overlay.status == "direct"
    assert surfaces.keypoints_norm.descriptor.space_id == (
        "source_camera_normalized_xy"
    )
    assert surfaces.pose_bbox_xyxy_img.descriptor.geometry_type == "bbox_xyxy"
    assert surfaces.keypoints_img.descriptor.row_identity.record_ref == (
        surfaces.context.row_identity.record_ref
    )
    assert run["keypoints_img"].attrs[COORDINATE_DESCRIPTOR_ATTR]["schema_version"] == 2


@pytest.mark.parametrize("marker", [False, None])
def test_keypoint_context_requires_explicit_normal_selector_eligibility(
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
        match="explicit normal-selector eligibility",
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
    parent = FakeGroup(
        path="keypoints_runs",
        archive_token=run._coordinate_archive_token,
        attrs={
            "latest": "prior",
            "latest_complete": "prior",
            "latest_pending": "prior_pending",
        },
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
def test_keypoint_publication_rejects_invalid_continuous_bbox_edges(
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
        match="positive continuous edge boxes",
    ):
        publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")


def test_keypoint_publication_accepts_exact_continuous_bbox_extent_edges(
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

    assert surfaces.pose_bbox_xyxy_roi.descriptor.pixel_convention == "continuous"


def _real_canonical_archive(tmp_path: Any) -> tuple[Any, Any]:
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
    instance_key = detect.create_array(
        "instance_key", data=np.asarray([101, 202], dtype="<u8")
    )
    detect.create_array(
        "frame_indices", data=np.asarray([0, 1], dtype="<i8")
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
    bbox_norm = detect.create_array("bbox_norm_coords", data=normalized_values)
    bbox_img = detect.create_array("bbox_img_xyxy", data=bbox_values)
    centers = detect.create_array("centers_img_xy", data=center_values)
    mapping = _publish_detection_acquisition_mapping(
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
        source_lineage_records=(mapping,),
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
    ownership = stamp_crop_placement_ownership(
        placement,
        row_identity=crop_geometry.row_identity,
        source_camera_frame=evidence.source_camera_frame,
    )
    roi_frame = stamp_roi_pixel_frame_authority(
        bind_array_reference_extent(roi_images, units="px"),
        frame_id="c1_roi",
        pixel_convention="continuous",
        crop_placement_ownership=ownership,
    )
    authority = stamp_crop_placement_transform_authority(
        placement,
        authority_id="c1_roi_to_source_camera",
        source_frame=roi_frame,
        target_frame=evidence.source_camera_frame,
    )
    link = stamp_directed_transform_v2(
        placement,
        transform_id="c1_roi_to_source_camera",
        authority=authority,
        source_frame=roi_frame,
        target_frame=evidence.source_camera_frame,
        row_identity=crop_geometry.row_identity,
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

    run = root.require_group("keypoints_runs").create_group("k1")
    run.attrs["stage_selector_eligible"] = True
    mark_run_started(run, run_name="k1", stage="keypoints")
    selected = np.asarray([1, 0], dtype="<i8")
    run.create_array("source_crop_row_ids", data=selected)
    run.create_array("instance_key", data=np.asarray(crop_key[:])[selected])
    run.create_array(
        "source_acquisition_frame_index",
        data=np.asarray(crop_time[:])[selected],
    )
    output_placement = np.asarray(placement[:])[selected]
    run.create_array("source_crop_xywh", data=output_placement)
    keypoints_roi = np.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
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
    prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=resolve_model_input_transform((40, 40)),
        preprocessing_input_mode="numpy-list",
        model_artifact=_artifact(),
    )
    publish_keypoint_coordinate_surfaces(root, "keypoints_runs/k1")
    mark_run_complete(run)
    return root, run


def test_real_zarr_fresh_loader_rejects_persisted_keypoint_tampering(tmp_path: Any) -> None:
    root, run = _real_canonical_archive(tmp_path)
    run["keypoints_img"][0, 0, 0] += 1.0

    reopened = zarr.open_group(str(tmp_path / "canonical.zarr"), mode="r+")
    with pytest.raises(KeypointCoordinatePublicationError):
        load_persisted_keypoint_coordinate_surfaces(reopened, "keypoints_runs/k1")
