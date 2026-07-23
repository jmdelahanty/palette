from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import fisheye.shared.keypoint_coordinate_publication as keypoint_publication
from fisheye.shared.keypoint_coordinate_publication import (
    KeypointCoordinatePublicationError,
    load_persisted_keypoint_crop_source,
)
from fisheye.shared.observation_coordinate_publication import (
    publish_crop_observation_geometry,
    publish_crop_roi_geometry,
)
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started
from tests.unit.fisheye.test_observation_coordinate_publication import (
    _crop_copy,
    _crop_roi,
    _published_detection,
    _world,
)


class _ExactRoot:
    def __init__(self, token: object) -> None:
        self.path = "archive_root"
        self.attrs: dict[str, Any] = {}
        self._coordinate_archive_token = token
        self.nodes: dict[str, Any] = {}

    def register(self, node: Any) -> None:
        self.nodes[node.path] = node

    def __getitem__(self, path: str) -> Any:
        return self.nodes[path]


def _attach(rowset: Any, node: Any) -> None:
    rowset[node.path.rsplit("/", 1)[-1]] = node


def _canonical_crop(monkeypatch: pytest.MonkeyPatch) -> tuple[_ExactRoot, Any]:
    token = object()
    world = _world(convention="continuous", archive_token=token)
    detection = _published_detection(world)
    crop_nodes = _crop_copy(world, detection)
    crop = publish_crop_observation_geometry(
        *crop_nodes,
        source_geometry=detection,
    )
    placements, bbox_roi, ownership, bbox_frame, chain, _point_ownership = (
        _crop_roi(world, crop)
    )
    publish_crop_roi_geometry(
        placements,
        bbox_roi,
        crop_geometry=crop,
        crop_placement_ownership=ownership,
        roi_frame=bbox_frame,
        roi_to_source_camera=chain,
    )
    rowset = crop._rowset_node
    for node in (*crop_nodes[1:], placements, bbox_roi):
        _attach(rowset, node)
    rowset.attrs.update(
        {
            "coordinate_contract": "canonical_v2",
            "crop_storage_mode": "materialized",
            "stage_selector_eligible": True,
        }
    )
    mark_run_started(rowset, run_name="c1", stage="crop")
    mark_run_complete(rowset)

    root = _ExactRoot(token)
    root.register(rowset)
    root.register(rowset["roi_images"])
    root.register(bbox_frame._authority_node)
    monkeypatch.setattr(
        keypoint_publication,
        "load_persisted_crop_observation_geometry",
        lambda root_node, path: crop,
    )
    return root, bbox_frame._authority_node


def test_keypoint_crop_loader_preserves_point_frame_and_loads_run_local_bbox_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, bbox_frame_node = _canonical_crop(monkeypatch)

    source = load_persisted_keypoint_crop_source(root, "crop_runs/c1")

    assert source.roi_frame.pixel_convention == "continuous"
    assert source.roi_frame.record_ref == (
        "/crop_runs/c1/roi_images@pixel_frame_authority"
    )
    assert source.bbox_roi_frame.pixel_convention == "pixel_edge_half_open"
    assert source.bbox_roi_frame.record_ref == (
        "/crop_runs/c1/coordinate_frames/roi_bbox_edge@pixel_frame_authority"
    )
    assert source.bbox_roi_frame._authority_node is bbox_frame_node


def test_keypoint_crop_loader_fails_closed_without_run_local_bbox_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, bbox_frame_node = _canonical_crop(monkeypatch)
    del root.nodes[bbox_frame_node.path]

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="bbox-edge frame",
    ):
        load_persisted_keypoint_crop_source(root, "crop_runs/c1")


def test_bbox_normalization_uses_bbox_specific_source_camera_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point_camera = SimpleNamespace(
        pixel_convention="continuous",
        endpoint=SimpleNamespace(width=100, height=80),
    )
    bbox_camera = SimpleNamespace(
        pixel_convention="pixel_edge_half_open",
        endpoint=SimpleNamespace(width=200, height=160),
    )
    context = SimpleNamespace(
        source=SimpleNamespace(
            crop_geometry=SimpleNamespace(
                source_geometry=SimpleNamespace(
                    frame_evidence=SimpleNamespace(
                        source_camera_frame=point_camera,
                        bbox_source_camera_frame=bbox_camera,
                    )
                )
            )
        )
    )
    monkeypatch.setattr(
        keypoint_publication,
        "require_bound_keypoint_coordinate_context",
        lambda value: value,
    )

    actual = keypoint_publication._image_to_normalized(
        np.asarray([[20.0, 16.0, 100.0, 80.0]], dtype="<f4"),
        context=context,
    )

    np.testing.assert_allclose(
        actual,
        np.asarray([[0.1, 0.1, 0.5, 0.5]], dtype="<f4"),
    )


def test_bbox_normalization_rejects_point_convention_crosswire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point_camera = SimpleNamespace(
        pixel_convention="continuous",
        endpoint=SimpleNamespace(width=100, height=80),
    )
    context = SimpleNamespace(
        source=SimpleNamespace(
            crop_geometry=SimpleNamespace(
                source_geometry=SimpleNamespace(
                    frame_evidence=SimpleNamespace(
                        source_camera_frame=point_camera,
                        bbox_source_camera_frame=point_camera,
                    )
                )
            )
        )
    )
    monkeypatch.setattr(
        keypoint_publication,
        "require_bound_keypoint_coordinate_context",
        lambda value: value,
    )

    with pytest.raises(
        KeypointCoordinatePublicationError,
        match="cross-wired.*pixel_edge_half_open",
    ):
        keypoint_publication._image_to_normalized(
            np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype="<f4"),
            context=context,
        )
