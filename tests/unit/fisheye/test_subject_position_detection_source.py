"""Fake-bound tests for the strict detection position source adapter."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.shared.subject_position_detection_source as detection_source
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    evaluate_estimator_profile,
)


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


@dataclass(frozen=True)
class _Authority:
    record_ref: str
    record_sha256: str


@dataclass(frozen=True)
class _Overlay:
    status: str = "direct"


@dataclass(frozen=True)
class _Descriptor:
    profile_id: str = "source_camera_image_px.top_left_y_down.v1"
    geometry_type: str = "bbox_xyxy"
    components: tuple[str, ...] = ("x_min", "y_min", "x_max", "y_max")
    component_units: tuple[str, ...] = ("px", "px", "px", "px")
    pixel_convention: str = "pixel_edge_half_open"
    source_camera_overlay: _Overlay = _Overlay()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": "palette.canonical_coordinate_descriptor",
            "schema_version": 2,
            "profile_id": self.profile_id,
            "space_id": "source_camera_image_px",
            "geometry_type": self.geometry_type,
            "components": list(self.components),
            "component_units": list(self.component_units),
            "origin": "top_left",
            "positive_directions": {"x": "right", "y": "down"},
            "reference_extent": {"width": 100, "height": 80, "units": "px"},
            "pixel_convention": self.pixel_convention,
            "row_identity": {"mode": "sibling", "array_ref": "instance_key"},
            "source_camera_overlay": {"status": self.source_camera_overlay.status},
            "lineage_refs": [],
        }

    def digest(self) -> str:
        return _DIGEST_C


@dataclass(frozen=True)
class _Surface:
    descriptor: _Descriptor
    reference_frame_authority: _Authority


@dataclass(frozen=True)
class _FrameEvidence:
    source_camera_frame: _Authority
    bbox_source_camera_frame: _Authority


@dataclass(frozen=True)
class _IdentityKey:
    content_sha256: str
    leading_dimension: int


@dataclass(frozen=True)
class _IdentityContract:
    key_array: _IdentityKey


@dataclass(frozen=True)
class _Identity:
    rowset_path: str
    key_array_path: str
    record_ref: str
    record_sha256: str
    contract: _IdentityContract


@dataclass(frozen=True)
class _Geometry:
    row_identity: _Identity
    bbox_image: _Surface
    frame_evidence: _FrameEvidence
    bbox_projection: _Authority
    temporal_authority: _Authority


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 3,
        "payload_digest": _DIGEST_A,
        "payload": {
            "run_id": "canonical_v1",
            "source_evidence_kind": "native_detection",
            "publication": {
                "completion_status": "complete",
                "stage_selector_eligible": True,
                "metadata_state": "direct_and_consolidated_validated",
                "metadata_declarations_digest": _DIGEST_B,
            },
            "logical_content": {"digest": _DIGEST_C},
        },
    }


def _fixture(
    *,
    keys: np.ndarray | None = None,
    descriptor: _Descriptor | None = None,
) -> tuple[object, _Geometry, dict[str, np.ndarray]]:
    keys = np.asarray(
        np.array([101, 202], dtype="<u8") if keys is None else keys,
        dtype="<u8",
    )
    frames = np.array([4, 9], dtype="<i8")
    bbox = np.array([[0.0, 0.0, 2.0, 4.0], [1.0, 1.0, 3.0, 5.0]], dtype="<f4")
    identity = _Identity(
        rowset_path="detect_runs/canonical_v1",
        key_array_path="detect_runs/canonical_v1/instance_key",
        record_ref="/detect_runs/canonical_v1@row_identity_contract",
        record_sha256=_DIGEST_A,
        contract=_IdentityContract(
            key_array=_IdentityKey(
                content_sha256=array_values_sha256(keys),
                leading_dimension=keys.shape[0],
            )
        ),
    )
    bbox_frame = _Authority("/frames/bbox@pixel_frame_authority", _DIGEST_B)
    geometry = _Geometry(
        row_identity=identity,
        bbox_image=_Surface(descriptor or _Descriptor(), bbox_frame),
        frame_evidence=_FrameEvidence(
            source_camera_frame=_Authority(
                "/frames/point@pixel_frame_authority", _DIGEST_C
            ),
            bbox_source_camera_frame=bbox_frame,
        ),
        bbox_projection=_Authority("/detect_runs/canonical_v1@bbox_projection", _DIGEST_A),
        temporal_authority=_Authority(
            "/detect_runs/canonical_v1@source_row_temporal_authority", _DIGEST_B
        ),
    )
    values = {
        "instance_key": keys,
        "source_acquisition_frame_index": frames,
        "bbox_norm_coords": np.array(
            [[0.01, 0.025, 0.02, 0.05], [0.02, 0.0375, 0.02, 0.05]],
            dtype="<f4",
        ),
        "bbox_img_xyxy": bbox,
    }
    return SimpleNamespace(), geometry, values


def _install(monkeypatch, geometry: _Geometry, values: dict[str, np.ndarray]):
    monkeypatch.setattr(
        detection_source,
        "require_active_coordinate_canonical_detection",
        lambda root, group_path: _manifest(),
    )
    monkeypatch.setattr(
        detection_source,
        "load_persisted_detection_observation_geometry",
        lambda root, path: geometry,
    )
    monkeypatch.setattr(
        detection_source,
        "require_bound_detection_observation_geometry",
        lambda value: value,
    )
    monkeypatch.setattr(
        detection_source,
        "detection_observation_geometry_values",
        lambda value: values,
    )


def test_caller_cannot_override_canonical_detection_validity(monkeypatch):
    root, geometry, values = _fixture()
    _install(monkeypatch, geometry, values)

    source = detection_source.load_persisted_detection_position_source(
        root,
        "detect_runs/canonical_v1",
    )
    np.testing.assert_array_equal(source.observation_validity, [True, True])
    with pytest.raises(TypeError):
        detection_source.load_persisted_detection_position_source(
            root,
            "detect_runs/canonical_v1",
            upstream_validity={"values": np.array([True, True], dtype=bool)},
        )


def test_schema_invalid_detection_row_fails_closed(monkeypatch):
    root, geometry, values = _fixture()
    values["bbox_norm_coords"][1, 2] = 0.0
    _install(monkeypatch, geometry, values)

    with pytest.raises(
        detection_source.DetectionPositionSourceError,
        match="schema-v1",
    ):
        detection_source.load_persisted_detection_position_source(
            root,
            "detect_runs/canonical_v1",
        )


def test_invalid_or_stale_selector_is_not_bypassed(monkeypatch):
    root, geometry, values = _fixture()
    _install(monkeypatch, geometry, values)
    monkeypatch.setattr(
        detection_source,
        "require_active_coordinate_canonical_detection",
        lambda root, group_path: (_ for _ in ()).throw(
            ValueError("selected run is stale")
        ),
    )

    with pytest.raises(ValueError, match="selected run is stale"):
        detection_source.load_persisted_detection_position_source(
            root,
            "detect_runs/canonical_v1",
        )


def test_reordered_identity_fails_even_when_rows_have_equal_length(monkeypatch):
    root, geometry, values = _fixture()
    reordered = dict(values)
    reordered["instance_key"] = values["instance_key"][::-1]
    _install(monkeypatch, geometry, reordered)

    with pytest.raises(detection_source.DetectionPositionSourceError, match="reordered"):
        detection_source.load_persisted_detection_position_source(
            root,
            "detect_runs/canonical_v1",
        )


def test_coordinate_mismatch_fails_closed(monkeypatch):
    root, geometry, values = _fixture(
        descriptor=_Descriptor(source_camera_overlay=_Overlay("requires_transform"))
    )
    _install(monkeypatch, geometry, values)

    with pytest.raises(detection_source.DetectionPositionSourceError, match="half-open"):
        detection_source.load_persisted_detection_position_source(
            root,
            "detect_runs/canonical_v1",
        )


def test_bbox_binding_is_half_open_and_evaluates_midpoint(monkeypatch):
    root, geometry, values = _fixture()
    _install(monkeypatch, geometry, values)
    source = detection_source.load_persisted_detection_position_source(
        root,
        "detect_runs/canonical_v1",
    )

    assert source.source_modality == "detection"
    assert source.source_kind == "native_detection"
    assert source.source_row_index.dtype == np.dtype("<i8")
    np.testing.assert_array_equal(source.source_row_index, [0, 1])
    assert source.bbox_descriptor.descriptor.pixel_convention == "pixel_edge_half_open"
    assert source.point_expression_bindings.bboxes["bbox_img_xyxy"].valid.dtype == bool

    result = evaluate_estimator_profile(
        DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
        source.point_expression_bindings,
    )
    np.testing.assert_allclose(result.position_xy[0], [1.0, 2.0])
    assert bool(result.valid[0])
    assert bool(result.valid[1])
    assert source.source_binding_record["metadata_evidence"]["metadata_state"] == (
        "direct_and_consolidated_validated"
    )
    assert source.source_binding_record["observation_validity"]["record_sha256"] == (
        _DIGEST_A
    )


def test_path_loader_requires_equal_direct_and_consolidated_evidence(
    monkeypatch,
    tmp_path,
):
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    bound = SimpleNamespace(source_binding_digest=_DIGEST_A, _analysis_zarr=None)
    monkeypatch.setattr(
        detection_source,
        "validate_direct_consolidated_subtree",
        lambda *args, **kwargs: SimpleNamespace(
            to_json=lambda: {"status": "equivalent"}
        ),
    )
    roots = iter((SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(detection_source, "open_zarr_root", lambda *args, **kwargs: next(roots))
    monkeypatch.setattr(detection_source, "_build_source", lambda *args, **kwargs: bound)

    result = detection_source.load_persisted_detection_position_source(
        archive,
        "detect_runs/canonical_v1",
    )
    assert result._analysis_zarr == archive.resolve()


def test_path_loader_rejects_direct_consolidated_source_disagreement(
    monkeypatch,
    tmp_path,
):
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    monkeypatch.setattr(
        detection_source,
        "validate_direct_consolidated_subtree",
        lambda *args, **kwargs: SimpleNamespace(
            to_json=lambda: {"status": "equivalent"}
        ),
    )
    monkeypatch.setattr(
        detection_source,
        "open_zarr_root",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    bindings = iter(
        (
            SimpleNamespace(source_binding_digest=_DIGEST_A),
            SimpleNamespace(source_binding_digest=_DIGEST_B),
        )
    )
    monkeypatch.setattr(
        detection_source,
        "_build_source",
        lambda *args, **kwargs: next(bindings),
    )

    with pytest.raises(
        detection_source.DetectionPositionSourceError,
        match="Direct and consolidated",
    ):
        detection_source.load_persisted_detection_position_source(
            archive,
            "detect_runs/canonical_v1",
        )
