from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import pytest

import fisheye.shared.coordinate_frame_record as coordinate_frame_records
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_NOT_SUITABLE,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    CanonicalCollectionAxis,
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    build_canonical_coordinate_descriptor,
    canonical_coordinate_descriptor_v2_attrs,
)
from fisheye.shared.canonical_coordinate_publication import (
    build_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_frame_record import (
    BODY_FRAME_CONTRACT_ATTR,
    BODY_FRAME_ESTIMATOR_ATTR,
    BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    FISH_ANATOMICAL_BODY_FRAME_ATTR,
    FISH_ANATOMICAL_BODY_FRAME_KIND,
    PHYSICAL_FRAME_CALIBRATION_ATTR,
    PHYSICAL_FRAME_CALIBRATION_KIND,
    PHYSICAL_FRAME_COMPATIBLE_PROFILE_IDS,
    REFERENCE_EXTENT_FINITE,
    REFERENCE_EXTENT_NOT_APPLICABLE,
    REFERENCE_EXTENT_UNBOUNDED,
    SELECTED_CAMERA_FRAME_EVIDENCE_ATTR,
    SOURCE_CAMERA_PROFILE_ID,
    BoundPhysicalFrameCalibration,
    CoordinateFrameRecordError,
    bind_body_spline_with_anchor_polarity_source,
    bind_keypoint_head_axis_source,
    bind_mask_component_axis_source,
    bind_body_frame_geometry,
    bind_body_source_coordinate_descriptor,
    build_body_frame_contract_record,
    build_body_frame_estimator_record,
    build_body_estimator_source_manifest_record,
    build_fish_anatomical_body_frame_record,
    build_physical_frame_calibration_record,
    load_bound_physical_frame_calibration,
    parse_body_frame_contract_record,
    parse_fish_anatomical_body_frame_record,
    parse_physical_frame_calibration_record,
    parse_selected_camera_frame_evidence_record,
    stamp_body_frame_contract,
    stamp_body_frame_estimator,
    stamp_fish_anatomical_body_frame_record,
    stamp_physical_frame_calibration_record,
    stamp_selected_camera_frame_evidence,
    verify_bound_coordinate_frame,
    verify_bound_body_estimator_source,
    verify_bound_selected_camera_frame_evidence,
    verify_bound_body_source_coordinate_descriptor,
    array_payload_sha256,
)
from fisheye.shared.proof_verification import proof_verification_scope
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    STIMULUS_STATE_DOMAIN,
    TRACK_SAMPLE_DOMAIN,
    TRACK_SAMPLE_INTERPOLATION_DTYPE,
    build_row_identity_contract,
    build_track_sample_key,
    derive_track_source_instance_values,
    resolve_source_acquisition_frame_indices,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
    stamp_track_sample_time_lineage,
)
from fisheye.shared.coordinate_record import (
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.selected_calibration import (
    build_selected_camera_source_evidence_from_h5_values,
)
from fisheye.shared.pixel_frame_authority import (
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
    stamp_source_camera_pixel_frame_authority,
)


class _Node:
    def __init__(
        self,
        path: str,
        *,
        token: object,
        data: Any | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: str = "u1",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self._coordinate_archive_token = token
        if data is None:
            if shape is None:
                shape = ()
            self._data = np.zeros(shape, dtype=dtype)
        else:
            self._data = np.asarray(data)
        self.shape = tuple(int(item) for item in self._data.shape)
        self.dtype = self._data.dtype
        self.attrs = {} if attrs is None else attrs

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data[key] = value


def _stamp_external_acquisition(root: Any, authority_node: Any) -> Any:
    ownership = stamp_acquisition_import_ownership(root, authority_node)
    return stamp_acquisition_camera_frame(
        root,
        authority_node,
        import_ownership=ownership,
    )


class _MutatingArrayNode(_Node):
    def __init__(self, *args: Any, mutate_on_read: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._read_count = 0
        self._mutate_on_read = mutate_on_read

    def __getitem__(self, key: Any) -> Any:
        self._read_count += 1
        if self._read_count == self._mutate_on_read:
            self._data = self._data.copy()
            self._data.flat[0] = self._data.flat[0] + 1
        return self._data[key]


class _CountingArrayNode(_Node):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read_count = 0

    def __getitem__(self, key: Any) -> Any:
        self.read_count += 1
        return self._data[key]


def test_array_payload_hash_reuses_only_one_operation_scoped_proof() -> None:
    node = _CountingArrayNode(
        "analysis/run/values",
        token=object(),
        data=np.arange(12, dtype=np.float32),
    )

    with proof_verification_scope():
        first = array_payload_sha256(node)
        second = array_payload_sha256(node)
        assert first == second
        # One stable initial proof performs the traditional two reads.
        assert node.read_count == 2

    # The outer scope closes with one fresh read and comparison.
    assert node.read_count == 3
    array_payload_sha256(node)
    # Outside a scope, each call retains the traditional two-read proof.
    assert node.read_count == 5


class _FailOnceAttrs(dict[str, Any]):
    def __init__(self, value: dict[str, Any], *, trigger: str) -> None:
        super().__init__(value)
        self.trigger = trigger
        self.fail = True

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        incoming = dict(other, **kwargs)
        if self.fail and self.trigger in incoming:
            self.fail = False
            first = next(iter(incoming))
            self[first] = incoming[first]
            raise RuntimeError("injected partial update")
        super().update(incoming)


class _MutatingAttrs(dict[str, Any]):
    def __init__(self, value: dict[str, Any], *, victim: _Node) -> None:
        super().__init__(value)
        self.victim = victim
        self.mutate = True

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        incoming = dict(other, **kwargs)
        super().update(incoming)
        if self.mutate and FISH_ANATOMICAL_BODY_FRAME_ATTR in incoming:
            self.mutate = False
            self.victim.attrs["coordinate_descriptor_sha256"] = "f" * 64


class _CoercingAttrs(dict[str, Any]):
    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        incoming = copy.deepcopy(dict(other, **kwargs))
        if PHYSICAL_FRAME_CALIBRATION_ATTR in incoming:
            incoming[PHYSICAL_FRAME_CALIBRATION_ATTR]["schema_version"] = 1.0
        super().update(incoming)


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_camera(
    *, camera_id: str = "2010093", width: int = 160, height: int = 120
) -> Any:
    record = {
        "camera_id": camera_id,
        "native_width_px": width,
        "native_height_px": height,
        "pixels_per_mm_camera": 50.0,
        "pixels_per_mm_projector": 4.0,
        "real_world_ref_mm": 10.0,
    }
    arena = {
        "active_camera_id": camera_id,
        "camera_calibrations": [record],
    }
    return build_selected_camera_source_evidence_from_h5_values(
        source_h5_path="/data/recording.h5",
        arena_config_raw=json.dumps(arena, separators=(",", ":")),
        camera_group_path=f"/calibration_snapshot/{camera_id}",
        camera_group_attrs={
            "pixels_per_mm_camera": 50.0,
            "pixels_per_mm_projector": 4.0,
            "real_world_ref_mm": 10.0,
        },
        expected_camera_id=camera_id,
    )


@pytest.fixture
def physical_inputs() -> dict[str, Any]:
    token = object()
    root = _Node(
        "archive_root",
        token=token,
        attrs={
            "recording_id": "recording-1",
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "camera_id": "2010093",
                "width": 160,
                "height": 120,
                "total_frames": 3,
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "camera.mp4",
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
    selected_node = _Node("analysis/calibration/selected", token=token)
    selected = stamp_selected_camera_frame_evidence(
        selected_node,
        source_camera=_verified_camera(),
    )
    source_node = _Node(
        "analysis/acquisition_camera_frames/2010093",
        token=token,
    )
    acquisition = _stamp_external_acquisition(root, source_node)
    source_frame_node = _Node(
        "analysis/coordinate_frames/source_camera/2010093/continuous",
        token=token,
    )
    source = stamp_source_camera_pixel_frame_authority(
        source_frame_node,
        frame_id="camera_2010093_native",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )
    record = build_physical_frame_calibration_record(
        frame_id="camera_2010093_mm",
        source_camera_pixels=source,
        selected_camera_evidence=selected,
    )
    return {
        "token": token,
        "root": root,
        "selected_node": selected_node,
        "selected": selected,
        "source_node": source_node,
        "source_frame_node": source_frame_node,
        "acquisition": acquisition,
        "source": source,
        "record": record,
    }


def _stamp_physical(
    inputs: dict[str, Any], *, mode: str = REFERENCE_EXTENT_FINITE
) -> tuple[_Node, Any]:
    node = _Node(f"analysis/coordinate_frames/physical_{mode}", token=inputs["token"])
    record = build_physical_frame_calibration_record(
        frame_id=f"camera_2010093_{mode}",
        source_camera_pixels=inputs["source"],
        selected_camera_evidence=inputs["selected"],
        physical_extent_mode=mode,
    )
    bound = stamp_physical_frame_calibration_record(
        node,
        record,
        expected_record_ref=f"/{node.path}@{PHYSICAL_FRAME_CALIBRATION_ATTR}",
        source_camera_pixels=inputs["source"],
        selected_camera_evidence=inputs["selected"],
    )
    return node, bound


def test_physical_round_trip_binds_native_camera_pixels_and_camera_ppm(
    physical_inputs: dict[str, Any],
) -> None:
    node, bound = _stamp_physical(physical_inputs)
    assert bound.kind == PHYSICAL_FRAME_CALIBRATION_KIND
    assert bound.coordinate_units == "mm"
    assert (bound.positive_x, bound.positive_y) == ("right", "down")
    assert bound.extent_mode == REFERENCE_EXTENT_FINITE
    assert bound.reference_width == 160.0 / 50.0
    assert bound.reference_height == 120.0 / 50.0
    assert bound.reference_units == "mm"
    assert bound.origin == bound.record.origin == "physical_frame_origin"
    assert bound.source_origin_relation == "coincident_with_source_camera_top_left"
    assert (bound.record.positive_x, bound.record.positive_y) == ("right", "down")
    assert bound.compatible_profile_ids == PHYSICAL_FRAME_COMPATIBLE_PROFILE_IDS
    assert bound.archive_identity == physical_inputs["source"].archive_identity
    assert (
        verify_bound_coordinate_frame(
            bound, expected_kind=PHYSICAL_FRAME_CALIBRATION_KIND
        )
        is bound
    )
    loaded = load_bound_physical_frame_calibration(
        node,
        expected_record_ref=bound.record_ref,
        expected_record_sha256=bound.record_sha256,
        expected_camera_id="2010093",
        source_camera_pixels=physical_inputs["source"],
        selected_camera_evidence=physical_inputs["selected"],
    )
    assert loaded.record == bound.record


def test_physical_frame_rejects_implicit_y_up_profile_compatibility(
    physical_inputs: dict[str, Any],
) -> None:
    payload = physical_inputs["record"].to_dict()
    payload["compatible_profile_ids"].append("physical_mm.cartesian_y_up.v1")
    with pytest.raises(
        CoordinateFrameRecordError,
        match="physical_profile_compatibility_invalid",
    ):
        parse_physical_frame_calibration_record(payload)


@pytest.mark.parametrize(
    ("mode", "units", "width"),
    (
        (REFERENCE_EXTENT_FINITE, "mm", 3.2),
        (REFERENCE_EXTENT_UNBOUNDED, "mm", None),
        (REFERENCE_EXTENT_NOT_APPLICABLE, "not_applicable", None),
    ),
)
def test_physical_extent_modes_remain_distinct(
    physical_inputs: dict[str, Any], mode: str, units: str, width: float | None
) -> None:
    _, bound = _stamp_physical(physical_inputs, mode=mode)
    assert bound.extent_mode == mode
    assert bound.reference_units == units
    assert bound.reference_width == width


def test_physical_rejects_projector_scale_swap(
    physical_inputs: dict[str, Any],
) -> None:
    payload = physical_inputs["record"].to_dict()
    payload["scale"]["pixels_per_mm_camera"] = 4.0
    payload["scale"]["mm_per_pixel"] = 0.25
    forged = parse_physical_frame_calibration_record(payload)
    node = _Node(
        "analysis/coordinate_frames/projector_swap", token=physical_inputs["token"]
    )
    with pytest.raises(
        CoordinateFrameRecordError, match="physical_scale_binding_mismatch"
    ):
        stamp_physical_frame_calibration_record(
            node,
            forged,
            expected_record_ref=f"/{node.path}@{PHYSICAL_FRAME_CALIBRATION_ATTR}",
            source_camera_pixels=physical_inputs["source"],
            selected_camera_evidence=physical_inputs["selected"],
        )


def test_physical_rejects_wrong_camera_and_unsupported_source_spaces(
    physical_inputs: dict[str, Any],
) -> None:
    wrong_camera = physical_inputs["record"].to_dict()
    wrong_camera["camera_id"] = "other_camera"
    node = _Node(
        "analysis/coordinate_frames/wrong_camera", token=physical_inputs["token"]
    )
    with pytest.raises(CoordinateFrameRecordError, match="camera_identity_mismatch"):
        stamp_physical_frame_calibration_record(
            node,
            parse_physical_frame_calibration_record(wrong_camera),
            expected_record_ref=f"/{node.path}@{PHYSICAL_FRAME_CALIBRATION_ATTR}",
            source_camera_pixels=physical_inputs["source"],
            selected_camera_evidence=physical_inputs["selected"],
        )

    for space_id in ("roi_local_px", "detector_model_input_px", "projector_px"):
        payload = physical_inputs["record"].to_dict()
        payload["source_space_id"] = space_id
        with pytest.raises(
            CoordinateFrameRecordError, match="source_space_unsupported"
        ):
            parse_physical_frame_calibration_record(payload)


def test_selected_camera_rejects_arbitrary_schema_and_stale_evidence(
    physical_inputs: dict[str, Any],
) -> None:
    arbitrary = physical_inputs["selected"].record.to_dict()
    arbitrary["schema_id"] = "caller.invented"
    with pytest.raises(CoordinateFrameRecordError, match="schema_invalid"):
        parse_selected_camera_frame_evidence_record(arbitrary)

    physical_inputs["selected_node"].attrs[SELECTED_CAMERA_FRAME_EVIDENCE_ATTR][
        "camera_id"
    ] = "different"
    with pytest.raises(
        CoordinateFrameRecordError, match="selected_camera_identity_mismatch"
    ):
        verify_bound_selected_camera_frame_evidence(physical_inputs["selected"])


def test_selected_camera_rejects_projector_ppm_selector(
    physical_inputs: dict[str, Any],
) -> None:
    payload = physical_inputs["selected"].record.to_dict()
    payload["pixels_per_mm_camera_selector"] = (
        "/selected_camera_record/pixels_per_mm_projector"
    )
    with pytest.raises(
        CoordinateFrameRecordError, match="selected_camera_scale_selector_invalid"
    ):
        parse_selected_camera_frame_evidence_record(payload)


def test_selected_camera_rejects_deepcopied_unsealed_source_evidence(
    physical_inputs: dict[str, Any],
) -> None:
    node = _Node(
        "analysis/calibration/deepcopy_rejected",
        token=physical_inputs["token"],
    )
    with pytest.raises(
        CoordinateFrameRecordError,
        match="selected_camera_source_unverified",
    ):
        stamp_selected_camera_frame_evidence(
            node,
            source_camera=copy.deepcopy(_verified_camera()),
        )


def test_source_camera_rejects_same_paths_from_different_archives(
    physical_inputs: dict[str, Any],
) -> None:
    other_token = object()
    other_root = _Node(
        "archive_root",
        token=other_token,
        attrs=copy.deepcopy(dict(physical_inputs["root"].attrs)),
    )
    other_source = _Node(
        "analysis/acquisition_camera_frames/2010093", token=other_token
    )
    acquisition = _stamp_external_acquisition(other_root, other_source)
    other_frame_node = _Node(
        "analysis/coordinate_frames/source_camera/2010093/continuous",
        token=other_token,
    )
    other_frame = stamp_source_camera_pixel_frame_authority(
        other_frame_node,
        frame_id="camera_2010093_native",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )
    with pytest.raises(CoordinateFrameRecordError, match="archive_mismatch"):
        build_physical_frame_calibration_record(
            frame_id="cross_archive",
            source_camera_pixels=other_frame,
            selected_camera_evidence=physical_inputs["selected"],
        )


def test_source_camera_rejects_camera_native_dimension_mismatch(
    physical_inputs: dict[str, Any],
) -> None:
    wrong_root = _Node(
        "wrong_archive_root",
        token=physical_inputs["token"],
        attrs=copy.deepcopy(dict(physical_inputs["root"].attrs)),
    )
    wrong_root.attrs["source_video_metadata"]["height"] = 100
    wrong_authority = _Node(
        "analysis/acquisition_camera_frames/2010093",
        token=physical_inputs["token"],
    )
    wrong_acquisition = _stamp_external_acquisition(
        wrong_root,
        wrong_authority,
    )
    wrong_frame = stamp_source_camera_pixel_frame_authority(
        _Node(
            "analysis/coordinate_frames/source_camera/2010093/continuous",
            token=physical_inputs["token"],
        ),
        frame_id="wrong_native",
        pixel_convention="continuous",
        acquisition_frame=wrong_acquisition,
    )
    with pytest.raises(
        CoordinateFrameRecordError, match="camera_native_extent_mismatch"
    ):
        build_physical_frame_calibration_record(
            frame_id="wrong_dimensions",
            source_camera_pixels=wrong_frame,
            selected_camera_evidence=physical_inputs["selected"],
        )


def test_source_camera_rejects_deepcopied_unsealed_authority(
    physical_inputs: dict[str, Any],
) -> None:
    with pytest.raises(
        CoordinateFrameRecordError, match="source_camera_frame_unverified"
    ):
        build_physical_frame_calibration_record(
            frame_id="deepcopy_rejected",
            source_camera_pixels=copy.deepcopy(physical_inputs["source"]),
            selected_camera_evidence=physical_inputs["selected"],
        )


def test_physical_detects_stale_source_pixel_extent(
    physical_inputs: dict[str, Any],
) -> None:
    _, bound = _stamp_physical(physical_inputs)
    physical_inputs["root"].attrs["source_video_metadata"]["width"] = 159
    with pytest.raises(CoordinateFrameRecordError, match="source_camera_frame_unverified"):
        verify_bound_coordinate_frame(
            bound, expected_kind=PHYSICAL_FRAME_CALIBRATION_KIND
        )


def test_physical_stamp_rejects_hostile_attrs_without_any_mutation(
    physical_inputs: dict[str, Any],
) -> None:
    attrs = _FailOnceAttrs(
        {"keep": {"nested": [1, 2]}}, trigger=PHYSICAL_FRAME_CALIBRATION_ATTR
    )
    before = copy.deepcopy(dict(attrs))
    node = _Node(
        "analysis/coordinate_frames/failing",
        token=physical_inputs["token"],
        attrs=attrs,
    )
    encoded_before = json.dumps(before, sort_keys=True, separators=(",", ":"))
    with pytest.raises(CoordinateFrameRecordError, match="stamp_preflight_failed"):
        stamp_physical_frame_calibration_record(
            node,
            physical_inputs["record"],
            expected_record_ref=f"/{node.path}@{PHYSICAL_FRAME_CALIBRATION_ATTR}",
            source_camera_pixels=physical_inputs["source"],
            selected_camera_evidence=physical_inputs["selected"],
        )
    assert dict(attrs) == before
    assert json.dumps(dict(attrs), sort_keys=True, separators=(",", ":")) == encoded_before


def test_frame_transaction_fully_rolls_back_after_post_write_failure(
    physical_inputs: dict[str, Any],
) -> None:
    node = _Node(
        "analysis/coordinate_frames/rollback_after_write",
        token=physical_inputs["token"],
        attrs={"keep": {"nested": [1, 2]}},
    )
    before = copy.deepcopy(dict(node.attrs))

    def fail_reload() -> None:
        raise RuntimeError("injected post-write verification failure")

    with pytest.raises(CoordinateFrameRecordError, match="stamp_failed"):
        coordinate_frame_records._transactional_stamp(
            node,
            attr_name="test_coordinate_frame_record",
            payload={"schema_id": "test", "schema_version": 1},
            reload_and_verify=fail_reload,
        )
    assert dict(node.attrs) == before
    assert json.dumps(dict(node.attrs), sort_keys=True, separators=(",", ":")) == (
        json.dumps(before, sort_keys=True, separators=(",", ":"))
    )


def test_physical_stamp_rejects_custom_coercing_attrs_without_write(
    physical_inputs: dict[str, Any],
) -> None:
    attrs = _CoercingAttrs({"keep": {"nested": [1, 2]}})
    before = copy.deepcopy(dict(attrs))
    node = _Node(
        "analysis/coordinate_frames/coercing",
        token=physical_inputs["token"],
        attrs=attrs,
    )
    with pytest.raises(CoordinateFrameRecordError, match="stamp_preflight_failed"):
        stamp_physical_frame_calibration_record(
            node,
            physical_inputs["record"],
            expected_record_ref=f"/{node.path}@{PHYSICAL_FRAME_CALIBRATION_ATTR}",
            source_camera_pixels=physical_inputs["source"],
            selected_camera_evidence=physical_inputs["selected"],
        )
    assert dict(attrs) == before


def test_physical_bound_construction_is_sealed() -> None:
    with pytest.raises(CoordinateFrameRecordError, match="frame_unsealed"):
        BoundPhysicalFrameCalibration(
            record_ref="/analysis/f@physical_frame_calibration",
            record_sha256="0" * 64,
            record=None,  # type: ignore[arg-type]
            source_camera_pixels=None,  # type: ignore[arg-type]
            selected_camera_evidence=None,  # type: ignore[arg-type]
            archive_identity=None,  # type: ignore[arg-type]
            node=None,
        )


def test_physical_parser_rejects_recursive_duplicate_json_key(
    physical_inputs: dict[str, Any],
) -> None:
    raw = physical_inputs["record"].canonical_json()
    duplicate = raw.replace(
        '"scale":{', '"scale":{"quantity":"pixels_per_mm_camera",', 1
    )
    with pytest.raises(CoordinateFrameRecordError, match="duplicate JSON key"):
        parse_physical_frame_calibration_record(duplicate)


def test_frame_record_schema_versions_require_exact_integer_raw_values(
    physical_inputs: dict[str, Any],
) -> None:
    payload = physical_inputs["record"].to_dict()
    payload["schema_version"] = 1.0
    with pytest.raises(CoordinateFrameRecordError, match="schema_invalid"):
        parse_physical_frame_calibration_record(payload)


def _identity(
    *, token: object, rowset_path: str = "analysis/body_source"
) -> tuple[_Node, _Node, Any]:
    rowset = _Node(rowset_path, token=token)
    key = _Node(
        f"{rowset_path}/instance_key",
        token=token,
        data=np.array([101, 102, 103], dtype=np.uint64),
    )
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=key[:],
    )
    bound = stamp_and_bind_row_identity_contract(
        rowset,
        key,
        contract=contract,
    )
    return rowset, key, bound


def _source_camera_descriptor(
    physical_inputs: dict[str, Any],
    *,
    rowset_path: str = "analysis/body_source",
) -> dict[str, Any]:
    rowset, key, identity = _identity(
        token=physical_inputs["token"], rowset_path=rowset_path
    )
    coordinates = _Node(
        f"{rowset_path}/positions_px",
        token=physical_inputs["token"],
        data=np.array(
            [
                [[10.0, 9.0], [10.0, 11.0], [8.0, 10.0]],
                [[21.0, 20.0], [19.0, 20.0], [20.0, 18.0]],
                [[30.0, 29.0], [30.0, 31.0], [28.0, 30.0]],
            ],
            dtype=np.float32,
        ),
    )
    source = physical_inputs["source"]
    descriptor = build_bound_canonical_coordinate_descriptor(
        coordinates,
        profile_id=SOURCE_CAMERA_PROFILE_ID,
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="continuous",
        row_identity=identity,
        reference_frame_authority=source,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )
    stamp_bound_canonical_coordinate_descriptor(descriptor)
    bound_source = bind_body_source_coordinate_descriptor(
        coordinates,
        row_identity=identity,
        source_camera_pixels=source,
    )
    return {
        "rowset": rowset,
        "key": key,
        "identity": identity,
        "coordinates": coordinates,
        "source": bound_source,
    }


def _typed_source_descriptor(
    physical_inputs: dict[str, Any],
    *,
    rowset_path: str,
    values: np.ndarray,
    geometry_type: str,
    collection_axis: CanonicalCollectionAxis | None = None,
    lineage_records: tuple[Any, ...] = (),
) -> dict[str, Any]:
    rowset, key, identity = _identity(
        token=physical_inputs["token"],
        rowset_path=rowset_path,
    )
    coordinates = _Node(
        f"{rowset_path}/source_xy",
        token=physical_inputs["token"],
        data=values,
    )
    publication = build_bound_canonical_coordinate_descriptor(
        coordinates,
        profile_id=SOURCE_CAMERA_PROFILE_ID,
        geometry_type=geometry_type,
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="continuous",
        row_identity=identity,
        reference_frame_authority=physical_inputs["source"],
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        collection_axis=collection_axis,
        lineage_records=lineage_records,
    )
    stamp_bound_canonical_coordinate_descriptor(publication)
    source = bind_body_source_coordinate_descriptor(
        coordinates,
        row_identity=identity,
        source_camera_pixels=physical_inputs["source"],
        lineage_records=lineage_records,
    )
    return {
        "rowset": rowset,
        "key": key,
        "identity": identity,
        "coordinates": coordinates,
        "source": source,
    }


def _source_descriptor_for_identity(
    physical_inputs: dict[str, Any],
    *,
    rowset_path: str,
    domain: str,
) -> tuple[Any, Any]:
    rowset = _Node(rowset_path, token=physical_inputs["token"])
    time_lineage = None
    if domain == STIMULUS_STATE_DOMAIN:
        values = np.asarray([0, 1, 2], dtype=np.int64)
        key_name = "stimulus_state_key"
        contract = build_row_identity_contract(
            domain=domain,
            values=values,
            components=("stimulus_frame_num",),
        )
    else:
        values = build_track_sample_key(
            np.asarray([7, 7, 8], dtype=np.int64),
            np.asarray([0, 1, 2], dtype=np.int64),
        )
        key_name = "track_sample_key"
    key = _Node(f"{rowset_path}/{key_name}", token=physical_inputs["token"], data=values)
    if domain == TRACK_SAMPLE_DOMAIN:
        source_rowset = _Node(
            f"{rowset_path}_immediate_source",
            token=physical_inputs["token"],
        )
        source_keys = np.asarray([41, 73, 109], dtype=np.uint64)
        source_key = _Node(
            f"{source_rowset.path}/instance_key",
            token=physical_inputs["token"],
            data=source_keys,
        )
        source_identity = stamp_and_bind_row_identity_contract(
            source_rowset,
            source_key,
            contract=build_row_identity_contract(
                domain=OBSERVATION_INSTANCE_DOMAIN,
                values=source_keys,
            ),
        )
        source_row_frames = _Node(
            f"{source_rowset.path}/source_acquisition_frame_index",
            token=physical_inputs["token"],
            data=values[:, 1].copy(),
        )
        source_temporal = stamp_source_row_temporal_authority(
            source_rowset,
            source_row_frames,
            source_row_identity=source_identity,
            acquisition_frame=physical_inputs["acquisition"],
        )
        source_row_index = _Node(
            f"{rowset_path}/source_row_index",
            token=physical_inputs["token"],
            data=np.arange(values.shape[0], dtype=np.int64),
        )
        source_frame = _Node(
            f"{rowset_path}/source_acquisition_frame_index",
            token=physical_inputs["token"],
            data=resolve_source_acquisition_frame_indices(
                source_temporal,
                source_row_index[:],
            ),
        )
        interpolation_values = np.zeros(
            (values.shape[0],),
            dtype=TRACK_SAMPLE_INTERPOLATION_DTYPE,
        )
        interpolation_values["left_source_frame_index"] = values[:, 1]
        interpolation_values["right_source_frame_index"] = values[:, 1]
        interpolation = _Node(
            f"{rowset_path}/source_frame_interpolation",
            token=physical_inputs["token"],
            data=interpolation_values,
        )
        source_instance = _Node(
            f"{rowset_path}/source_instance_key",
            token=physical_inputs["token"],
            data=derive_track_source_instance_values(
                source_temporal,
                source_row_index[:],
            ),
        )
        time_lineage = stamp_track_sample_time_lineage(
            rowset,
            key,
            source_row_index,
            source_frame,
            interpolation,
            source_instance,
            source_temporal_authority=source_temporal,
        )
        contract = build_row_identity_contract(
            domain=domain,
            values=values,
            track_time_lineage=time_lineage,
        )
    identity = stamp_and_bind_row_identity_contract(
        rowset,
        key,
        contract=contract,
        track_time_lineage=time_lineage,
    )
    coordinates = _Node(
        f"{rowset_path}/positions_px",
        token=physical_inputs["token"],
        data=np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
    )
    source_camera = physical_inputs["source"]
    descriptor = build_bound_canonical_coordinate_descriptor(
        coordinates,
        profile_id=SOURCE_CAMERA_PROFILE_ID,
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="continuous",
        row_identity=identity,
        reference_frame_authority=source_camera,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )
    stamp_bound_canonical_coordinate_descriptor(descriptor)
    return identity, coordinates


def test_body_identity_allowlist_accepts_track_samples_and_rejects_stimulus(
    physical_inputs: dict[str, Any],
) -> None:
    track_identity, track_coordinates = _source_descriptor_for_identity(
        physical_inputs,
        rowset_path="analysis/body_track_source",
        domain=TRACK_SAMPLE_DOMAIN,
    )
    track_source = bind_body_source_coordinate_descriptor(
        track_coordinates,
        row_identity=track_identity,
        source_camera_pixels=physical_inputs["source"],
    )
    assert track_source.row_identity.contract.domain == TRACK_SAMPLE_DOMAIN

    stimulus_identity, stimulus_coordinates = _source_descriptor_for_identity(
        physical_inputs,
        rowset_path="analysis/body_stimulus_source",
        domain=STIMULUS_STATE_DOMAIN,
    )
    with pytest.raises(
        CoordinateFrameRecordError,
        match="body_row_identity_unsupported",
    ):
        bind_body_source_coordinate_descriptor(
            stimulus_coordinates,
            row_identity=stimulus_identity,
            source_camera_pixels=physical_inputs["source"],
        )


@pytest.fixture
def body_inputs(physical_inputs: dict[str, Any]) -> dict[str, Any]:
    source_parts = _source_camera_descriptor(physical_inputs)
    token = physical_inputs["token"]
    contract_node = _Node("analysis/body_contracts/canonical_v1", token=token)
    contract = stamp_body_frame_contract(
        contract_node,
        record=build_body_frame_contract_record(),
    )
    estimator_node = _Node("analysis/body_estimators/keypoint_v3", token=token)
    estimator = stamp_body_frame_estimator(
        estimator_node,
        record=build_body_frame_estimator_record(
            method="keypoint_head_axis",
            implementation_version="3.0.0",
            configuration_schema_id="palette.keypoint_head_axis_parameters",
            configuration={
                "eye_left": "eye_left",
                "eye_right": "eye_right",
                "posterior_anchor": "swim_bladder",
            },
        ),
    )
    source_schema_node = _Node(
        "analysis/body_source_schemas/keypoints_v1",
        token=token,
    )
    source_schema = stamp_and_bind_persisted_coordinate_record(
        source_schema_node,
        {
            "schema_id": "palette.keypoint_label_schema",
            "schema_version": 1,
            "labels": ["eye_left", "eye_right", "swim_bladder"],
        },
        attr_name="keypoint_schema",
    )
    source_validity = _Node(
        f"{source_parts['rowset'].path}/keypoints_valid",
        token=token,
        data=np.array(
            [
                [True, True, True],
                [True, True, True],
                [False, False, False],
            ],
            dtype=np.bool_,
        ),
    )
    source_manifest = stamp_and_bind_persisted_coordinate_record(
        source_parts["rowset"],
        build_body_estimator_source_manifest_record(
            method="keypoint_head_axis",
            source_descriptor=source_parts["source"],
            estimator=estimator,
            source_schema=source_schema,
            support_nodes={"validity": source_validity},
        ),
        attr_name=BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    )
    estimator_source = bind_keypoint_head_axis_source(
        source_descriptor=source_parts["source"],
        estimator=estimator,
        keypoint_schema=source_schema,
        validity_node=source_validity,
        producer_manifest=source_manifest,
    )
    frame_node = _Node("analysis/body_frames/body_v1", token=token)
    origin = _Node(
        f"{frame_node.path}/origin_xy",
        token=token,
        data=np.array([[10.0, 10.0], [20.0, 20.0], [np.nan, np.nan]], dtype=np.float32),
    )
    forward = _Node(
        f"{frame_node.path}/forward_axis_xy",
        token=token,
        data=np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]], dtype=np.float32),
    )
    left = _Node(
        f"{frame_node.path}/left_axis_xy",
        token=token,
        data=np.array([[0.0, -1.0], [1.0, 0.0], [np.nan, np.nan]], dtype=np.float32),
    )
    valid = _Node(
        f"{frame_node.path}/axis_valid",
        token=token,
        data=np.array([True, True, False], dtype=np.bool_),
    )
    geometry = bind_body_frame_geometry(
        frame_node,
        origin_xy_node=origin,
        forward_axis_xy_node=forward,
        left_axis_xy_node=left,
        axis_valid_node=valid,
        row_identity=source_parts["identity"],
        estimator_source=estimator_source,
    )
    record = build_fish_anatomical_body_frame_record(
        frame_id="fish_body_v1",
        origin_definition="eye_pair_midpoint",
        body_frame_contract=contract,
        estimator_source=estimator_source,
        geometry=geometry,
        row_identity=source_parts["identity"],
    )
    return {
        **physical_inputs,
        **source_parts,
        "contract_node": contract_node,
        "contract": contract,
        "estimator_node": estimator_node,
        "estimator": estimator,
        "source_schema_node": source_schema_node,
        "source_schema": source_schema,
        "source_validity": source_validity,
        "source_manifest": source_manifest,
        "estimator_source": estimator_source,
        "frame_node": frame_node,
        "origin": origin,
        "forward": forward,
        "left": left,
        "valid": valid,
        "geometry": geometry,
        "record": record,
    }


def _stamp_body(inputs: dict[str, Any]) -> Any:
    return stamp_fish_anatomical_body_frame_record(
        inputs["frame_node"],
        inputs["record"],
        expected_record_ref=f"/{inputs['frame_node'].path}@{FISH_ANATOMICAL_BODY_FRAME_ATTR}",
        body_frame_contract=inputs["contract"],
        estimator_source=inputs["estimator_source"],
        geometry=inputs["geometry"],
        row_identity=inputs["identity"],
    )


def test_body_round_trip_binds_descriptor_identity_contract_estimator_and_arrays(
    body_inputs: dict[str, Any],
) -> None:
    bound = _stamp_body(body_inputs)
    assert bound.kind == FISH_ANATOMICAL_BODY_FRAME_KIND
    assert bound.coordinate_units == "px"
    assert bound.origin == "body_frame_origin"
    assert bound.origin_definition == "eye_pair_midpoint"
    assert (bound.positive_x, bound.positive_y) == ("anterior", "anatomical_left")
    assert bound.extent_mode == REFERENCE_EXTENT_NOT_APPLICABLE
    assert bound.reference_units == "not_applicable"
    assert bound.record.geometry.forward_axis_xy.content_sha256
    assert bound.record.row_identity.record_ref == body_inputs["identity"].record_ref
    assert bound.archive_identity == body_inputs["identity"].archive_identity
    assert (
        verify_bound_coordinate_frame(
            bound, expected_kind=FISH_ANATOMICAL_BODY_FRAME_KIND
        )
        is bound
    )


def test_keypoint_estimator_source_binds_exact_schema_validity_and_payload(
    body_inputs: dict[str, Any],
) -> None:
    bundle = body_inputs["estimator_source"]
    assert bundle.record.method == "keypoint_head_axis"
    assert bundle.record.labels == ("eye_left", "eye_right", "swim_bladder")
    assert bundle.record.source_payload == body_inputs["source"].source_payload
    assert set(bundle.record.support_arrays) == {"validity"}
    assert verify_bound_body_estimator_source(bundle) is bundle

    body_inputs["source_validity"]._data[0, 0] = False
    with pytest.raises(
        CoordinateFrameRecordError,
        match="estimator_source_manifest_mismatch",
    ):
        verify_bound_body_estimator_source(bundle)


def test_body_geometry_is_exactly_rederived_not_only_orthonormal(
    body_inputs: dict[str, Any],
) -> None:
    body_inputs["forward"]._data[0] = [0.0, 1.0]
    body_inputs["left"]._data[0] = [1.0, 0.0]
    with pytest.raises(
        CoordinateFrameRecordError,
        match="estimator_formula_output_mismatch",
    ):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )


def test_body_geometry_cannot_suppress_a_derivable_valid_axis(
    body_inputs: dict[str, Any],
) -> None:
    body_inputs["valid"]._data[0] = False
    body_inputs["origin"]._data[0] = np.nan
    body_inputs["forward"]._data[0] = np.nan
    body_inputs["left"]._data[0] = np.nan
    with pytest.raises(
        CoordinateFrameRecordError,
        match="estimator_source_validity_mismatch",
    ):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )


def test_keypoint_estimator_rejects_unlabeled_point_substitution(
    physical_inputs: dict[str, Any],
    body_inputs: dict[str, Any],
) -> None:
    point = _typed_source_descriptor(
        physical_inputs,
        rowset_path="analysis/body_point_substitution",
        values=np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        geometry_type="point_xy",
    )
    validity = _Node(
        f"{point['rowset'].path}/validity",
        token=physical_inputs["token"],
        data=np.ones((3, 2), dtype=np.bool_),
    )
    with pytest.raises(
        CoordinateFrameRecordError,
        match="estimator_source_geometry_invalid",
    ):
        bind_keypoint_head_axis_source(
            source_descriptor=point["source"],
            estimator=body_inputs["estimator"],
            keypoint_schema=body_inputs["source_schema"],
            validity_node=validity,
            producer_manifest=body_inputs["source_manifest"],
        )


def test_body_axis_valid_cannot_exceed_exact_estimator_source_validity(
    body_inputs: dict[str, Any],
) -> None:
    body_inputs["origin"]._data[2] = [30.0, 30.0]
    body_inputs["forward"]._data[2] = [1.0, 0.0]
    body_inputs["left"]._data[2] = [0.0, -1.0]
    body_inputs["valid"]._data[2] = True
    with pytest.raises(
        CoordinateFrameRecordError,
        match="estimator_source_validity_mismatch",
    ):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )


def test_mask_and_spline_estimators_require_their_exact_typed_source_bundles(
    physical_inputs: dict[str, Any],
) -> None:
    token = physical_inputs["token"]
    mask_schema = stamp_and_bind_persisted_coordinate_record(
        _Node("analysis/body_source_schemas/mask_v1", token=token),
        {
            "schema_id": "palette.mask_component_geometry_schema",
            "schema_version": 1,
            "components": ["eye_left", "eye_right", "swim_bladder"],
        },
        attr_name="component_schema",
    )
    mask_source = _typed_source_descriptor(
        physical_inputs,
        rowset_path="analysis/body_mask_source",
        values=np.asarray(
            [
                [[1, 2], [3, 4], [2, 6]],
                [[5, 6], [7, 8], [6, 10]],
                [[9, 10], [11, 12], [10, 14]],
            ],
            dtype=np.float32,
        ),
        geometry_type="point_xy",
        collection_axis=CanonicalCollectionAxis(
            axis=1,
            role="subject_component",
            cardinality=3,
            label_authority=DigestBoundCoordinateRecordRef(
                record_ref=mask_schema.record_ref,
                record_sha256=mask_schema.record_sha256,
            ),
        ),
        lineage_records=(mask_schema,),
    )
    mask_estimator = stamp_body_frame_estimator(
        _Node("analysis/body_estimators/mask_v1", token=token),
        record=build_body_frame_estimator_record(
            method="mask_component_axis",
            implementation_version="1.0.0",
            configuration_schema_id="palette.mask_component_axis_parameters",
            configuration={
                "eye_left": "eye_left",
                "eye_right": "eye_right",
                "posterior_anchor": "swim_bladder",
            },
        ),
    )
    mask_validity = _Node(
        f"{mask_source['rowset'].path}/component_valid",
        token=token,
        data=np.ones((3, 3), dtype=np.bool_),
    )
    mask_manifest = stamp_and_bind_persisted_coordinate_record(
        mask_source["rowset"],
        build_body_estimator_source_manifest_record(
            method="mask_component_axis",
            source_descriptor=mask_source["source"],
            estimator=mask_estimator,
            source_schema=mask_schema,
            support_nodes={"validity": mask_validity},
        ),
        attr_name=BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    )
    mask_bundle = bind_mask_component_axis_source(
        source_descriptor=mask_source["source"],
        estimator=mask_estimator,
        component_schema=mask_schema,
        validity_node=mask_validity,
        producer_manifest=mask_manifest,
    )
    assert mask_bundle.record.method == "mask_component_axis"
    assert mask_source["source"].descriptor.geometry_type == "point_xy"
    assert mask_source["source"].descriptor.collection_axis is not None
    assert (
        mask_source["source"].descriptor.collection_axis.label_authority.record_ref
        == mask_schema.record_ref
    )

    uncollected_mask_source = _typed_source_descriptor(
        physical_inputs,
        rowset_path="analysis/body_mask_source_uncollected",
        values=np.asarray(mask_source["coordinates"][:], dtype=np.float32),
        geometry_type="points_xy",
    )
    uncollected_validity = _Node(
        f"{uncollected_mask_source['rowset'].path}/component_valid",
        token=token,
        data=np.ones((3, 3), dtype=np.bool_),
    )
    uncollected_manifest = stamp_and_bind_persisted_coordinate_record(
        uncollected_mask_source["rowset"],
        build_body_estimator_source_manifest_record(
            method="mask_component_axis",
            source_descriptor=uncollected_mask_source["source"],
            estimator=mask_estimator,
            source_schema=mask_schema,
            support_nodes={"validity": uncollected_validity},
        ),
        attr_name=BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    )
    with pytest.raises(
        CoordinateFrameRecordError,
        match="mask-component sources require collected point_xy",
    ):
        bind_mask_component_axis_source(
            source_descriptor=uncollected_mask_source["source"],
            estimator=mask_estimator,
            component_schema=mask_schema,
            validity_node=uncollected_validity,
            producer_manifest=uncollected_manifest,
        )
    mask_frame = _Node("analysis/body_frames/mask_v1", token=token)
    mask_geometry = bind_body_frame_geometry(
        mask_frame,
        origin_xy_node=_Node(
            f"{mask_frame.path}/origin_xy",
            token=token,
            data=np.asarray([[2, 3], [6, 7], [10, 11]], dtype=np.float32),
        ),
        forward_axis_xy_node=_Node(
            f"{mask_frame.path}/forward_axis_xy",
            token=token,
            data=np.asarray([[0, -1], [0, -1], [0, -1]], dtype=np.float32),
        ),
        left_axis_xy_node=_Node(
            f"{mask_frame.path}/left_axis_xy",
            token=token,
            data=np.asarray([[-1, 0], [-1, 0], [-1, 0]], dtype=np.float32),
        ),
        axis_valid_node=_Node(
            f"{mask_frame.path}/axis_valid",
            token=token,
            data=np.ones((3,), dtype=np.bool_),
        ),
        row_identity=mask_source["identity"],
        estimator_source=mask_bundle,
    )
    assert mask_geometry.record.origin_xy.content_sha256

    spline_source = _typed_source_descriptor(
        physical_inputs,
        rowset_path="analysis/body_spline_source",
        values=np.asarray(
            [
                [[1, 1], [2, 1], [3, 1]],
                [[1, 2], [2, 2], [3, 2]],
                [[1, 3], [2, 3], [3, 3]],
            ],
            dtype=np.float32,
        ),
        geometry_type="polyline_xy",
    )
    spline_estimator = stamp_body_frame_estimator(
        _Node("analysis/body_estimators/spline_v1", token=token),
        record=build_body_frame_estimator_record(
            method="body_spline_with_anchor_polarity",
            implementation_version="1.0.0",
            configuration_schema_id="palette.body_spline_anchor_parameters",
            configuration={
                "eye_left": "eye_left",
                "eye_right": "eye_right",
                "posterior_anchor": "swim_bladder",
            },
        ),
    )
    spline_schema = stamp_and_bind_persisted_coordinate_record(
        _Node("analysis/body_source_schemas/spline_v1", token=token),
        {
            "schema_id": "palette.body_spline_polarity_schema",
            "schema_version": 1,
            "anchors": ["eye_left", "eye_right", "swim_bladder"],
        },
        attr_name="polarity_schema",
    )
    anchors = _Node(
        f"{spline_source['rowset'].path}/polarity_anchors",
        token=token,
        data=np.asarray(
            [
                [[3, 0], [3, 2], [1, 1]],
                [[3, 1], [3, 3], [1, 2]],
                [[3, 2], [3, 4], [1, 3]],
            ],
            dtype=np.float32,
        ),
    )
    polarity_valid = _Node(
        f"{spline_source['rowset'].path}/polarity_valid",
        token=token,
        data=np.ones((3, 3), dtype=np.bool_),
    )
    spline_manifest = stamp_and_bind_persisted_coordinate_record(
        spline_source["rowset"],
        build_body_estimator_source_manifest_record(
            method="body_spline_with_anchor_polarity",
            source_descriptor=spline_source["source"],
            estimator=spline_estimator,
            source_schema=spline_schema,
            support_nodes={
                "polarity_anchors": anchors,
                "polarity_valid": polarity_valid,
            },
        ),
        attr_name=BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    )
    spline_bundle = bind_body_spline_with_anchor_polarity_source(
        source_descriptor=spline_source["source"],
        estimator=spline_estimator,
        polarity_schema=spline_schema,
        polarity_anchors_node=anchors,
        polarity_valid_node=polarity_valid,
        producer_manifest=spline_manifest,
    )
    assert spline_bundle.record.method == "body_spline_with_anchor_polarity"
    spline_frame = _Node("analysis/body_frames/spline_v1", token=token)
    spline_geometry = bind_body_frame_geometry(
        spline_frame,
        origin_xy_node=_Node(
            f"{spline_frame.path}/origin_xy",
            token=token,
            data=np.asarray([[3, 1], [3, 2], [3, 3]], dtype=np.float32),
        ),
        forward_axis_xy_node=_Node(
            f"{spline_frame.path}/forward_axis_xy",
            token=token,
            data=np.asarray([[1, 0], [1, 0], [1, 0]], dtype=np.float32),
        ),
        left_axis_xy_node=_Node(
            f"{spline_frame.path}/left_axis_xy",
            token=token,
            data=np.asarray([[0, -1], [0, -1], [0, -1]], dtype=np.float32),
        ),
        axis_valid_node=_Node(
            f"{spline_frame.path}/axis_valid",
            token=token,
            data=np.ones((3,), dtype=np.bool_),
        ),
        row_identity=spline_source["identity"],
        estimator_source=spline_bundle,
    )
    assert spline_geometry.record.forward_axis_xy.content_sha256

    with pytest.raises(
        CoordinateFrameRecordError,
        match="estimator_source_method_mismatch",
    ):
        bind_mask_component_axis_source(
            source_descriptor=mask_source["source"],
            estimator=spline_estimator,
            component_schema=mask_schema,
            validity_node=mask_validity,
            producer_manifest=mask_manifest,
        )


def test_body_source_rejects_nonnumeric_coordinate_payload(
    physical_inputs: dict[str, Any],
) -> None:
    parts = _source_camera_descriptor(
        physical_inputs,
        rowset_path="analysis/body_nonnumeric_source",
    )
    parts["coordinates"]._data = np.full((3, 2, 2), "not-a-number", dtype="U12")
    parts["coordinates"].shape = parts["coordinates"]._data.shape
    parts["coordinates"].dtype = parts["coordinates"]._data.dtype
    with pytest.raises(
        CoordinateFrameRecordError,
        match="body_source_dtype_invalid",
    ):
        bind_body_source_coordinate_descriptor(
            parts["coordinates"],
            row_identity=parts["identity"],
            source_camera_pixels=physical_inputs["source"],
        )


def test_body_contract_rejects_axes_mismatch() -> None:
    payload = build_body_frame_contract_record().to_dict()
    payload["axes"]["positive_y"] = "right"
    with pytest.raises(CoordinateFrameRecordError, match="body_axes_invalid"):
        parse_body_frame_contract_record(payload)


def test_body_rejects_unrelated_same_length_identity(
    body_inputs: dict[str, Any],
) -> None:
    _, _, unrelated = _identity(
        token=body_inputs["token"], rowset_path="analysis/unrelated_rows"
    )
    with pytest.raises(
        CoordinateFrameRecordError, match="row_identity_binding_mismatch"
    ):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=unrelated,
            estimator_source=body_inputs["estimator_source"],
        )


def test_body_source_descriptor_fails_when_descriptor_digest_drifts(
    body_inputs: dict[str, Any],
) -> None:
    body_inputs["coordinates"].attrs["coordinate_descriptor_sha256"] = "f" * 64
    with pytest.raises(CoordinateFrameRecordError, match="source_descriptor_invalid"):
        _stamp_body(body_inputs)


def test_body_source_binds_payload_content_and_detects_mutation(
    body_inputs: dict[str, Any],
) -> None:
    bound = _stamp_body(body_inputs)
    assert bound.record.source_coordinate_payload == body_inputs["source"].source_payload

    body_inputs["coordinates"]._data[0, 0] += 1.0
    with pytest.raises(
        CoordinateFrameRecordError,
        match="body_source_descriptor_stale|source_coordinate_payload",
    ):
        verify_bound_coordinate_frame(
            bound,
            expected_kind=FISH_ANATOMICAL_BODY_FRAME_KIND,
        )


def test_body_source_requires_every_descriptor_lineage_record_to_be_sealed(
    physical_inputs: dict[str, Any],
) -> None:
    parts = _source_camera_descriptor(
        physical_inputs,
        rowset_path="analysis/body_source_with_lineage",
    )
    lineage_node = _Node(
        "analysis/source_lineage/detect_v1",
        token=physical_inputs["token"],
    )
    lineage = stamp_and_bind_persisted_coordinate_record(
        lineage_node,
        {
            "schema_id": "palette.test_source_lineage",
            "schema_version": 1,
            "source_run_path": "/analysis/detect_runs/d1",
        },
        attr_name="source_lineage",
    )
    descriptor = build_bound_canonical_coordinate_descriptor(
        parts["coordinates"],
        profile_id=SOURCE_CAMERA_PROFILE_ID,
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="continuous",
        row_identity=parts["identity"],
        reference_frame_authority=physical_inputs["source"],
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        lineage_records=(lineage,),
    )
    stamp_bound_canonical_coordinate_descriptor(descriptor)

    with pytest.raises(
        CoordinateFrameRecordError,
        match="source_descriptor_lineage_unverified",
    ):
        bind_body_source_coordinate_descriptor(
            parts["coordinates"],
            row_identity=parts["identity"],
            source_camera_pixels=physical_inputs["source"],
        )

    bound = bind_body_source_coordinate_descriptor(
        parts["coordinates"],
        row_identity=parts["identity"],
        source_camera_pixels=physical_inputs["source"],
        lineage_records=(lineage,),
    )
    assert any(item.record_ref == lineage.record_ref for item in bound.lineage_records)
    lineage_node.attrs["source_lineage"]["source_run_path"] = "/analysis/other"
    with pytest.raises(
        CoordinateFrameRecordError,
        match="source_descriptor_lineage_unverified",
    ):
        verify_bound_body_source_coordinate_descriptor(bound)


def test_body_source_rejects_caller_invented_lineage_ref(
    physical_inputs: dict[str, Any],
) -> None:
    parts = _source_camera_descriptor(
        physical_inputs,
        rowset_path="analysis/body_source_forged_lineage",
    )
    descriptor = build_canonical_coordinate_descriptor(
        profile_id=SOURCE_CAMERA_PROFILE_ID,
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        reference_width=physical_inputs["source"].endpoint.width,
        reference_height=physical_inputs["source"].endpoint.height,
        reference_authority=DigestBoundCoordinateRecordRef(
            physical_inputs["source"].record_ref,
            physical_inputs["source"].record_sha256,
        ),
        reference_selector="record",
        pixel_convention="continuous",
        row_identity_contract=parts["identity"].contract,
        row_identity_record_ref=parts["identity"].record_ref,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=physical_inputs["source"].record_ref,
            record_sha256=physical_inputs["source"].record_sha256,
        ),
        lineage_refs=(
            DigestBoundCoordinateRecordRef(
                "/analysis/missing@source_lineage",
                "0" * 64,
            ),
        ),
    )
    parts["coordinates"].attrs.update(
        canonical_coordinate_descriptor_v2_attrs(descriptor)
    )
    with pytest.raises(
        CoordinateFrameRecordError,
        match="source_descriptor_lineage_unverified",
    ):
        bind_body_source_coordinate_descriptor(
            parts["coordinates"],
            row_identity=parts["identity"],
            source_camera_pixels=physical_inputs["source"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("forward", np.array([2.0, 0.0], dtype=np.float32), "axis_norm_invalid"),
        ("left", np.array([1.0, 0.0], dtype=np.float32), "axis_orthogonality_invalid"),
        ("left", np.array([0.0, 1.0], dtype=np.float32), "axis_polarity_invalid"),
    ),
)
def test_body_rejects_invalid_vector_geometry(
    body_inputs: dict[str, Any], field: str, value: np.ndarray, message: str
) -> None:
    body_inputs[field]._data[0] = value
    with pytest.raises(CoordinateFrameRecordError, match=message):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )


def test_body_rejects_nonfinite_valid_and_non_nan_invalid_rows(
    body_inputs: dict[str, Any],
) -> None:
    body_inputs["origin"]._data[0, 0] = np.nan
    with pytest.raises(CoordinateFrameRecordError, match="geometry_nonfinite_valid"):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )

    body_inputs["origin"]._data[0, 0] = 10.0
    body_inputs["forward"]._data[2] = [0.0, 0.0]
    with pytest.raises(
        CoordinateFrameRecordError, match="geometry_invalid_row_encoding"
    ):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )


def test_body_detects_axis_payload_and_dtype_drift(
    body_inputs: dict[str, Any],
) -> None:
    bound = _stamp_body(body_inputs)
    body_inputs["forward"]._data[0] = [0.0, 1.0]
    with pytest.raises(CoordinateFrameRecordError):
        verify_bound_coordinate_frame(
            bound, expected_kind=FISH_ANATOMICAL_BODY_FRAME_KIND
        )

    # Rebuild a clean fixture through direct restoration, then drift declared dtype.
    body_inputs["forward"]._data[0] = [1.0, 0.0]
    body_inputs["forward"].dtype = np.dtype("f8")
    with pytest.raises(
        CoordinateFrameRecordError, match="array_metadata_value_mismatch"
    ):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )


def test_body_rejects_axis_valid_dtype_drift(
    body_inputs: dict[str, Any],
) -> None:
    body_inputs["valid"]._data = body_inputs["valid"]._data.astype(np.uint8)
    body_inputs["valid"].dtype = body_inputs["valid"]._data.dtype
    with pytest.raises(CoordinateFrameRecordError, match="axis_valid_dtype_invalid"):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=body_inputs["forward"],
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )


def test_body_geometry_rejects_payload_change_between_validation_and_hash(
    body_inputs: dict[str, Any],
) -> None:
    mutating_forward = _MutatingArrayNode(
        body_inputs["forward"].path,
        token=body_inputs["token"],
        data=body_inputs["forward"][:].copy(),
        mutate_on_read=2,
    )
    with pytest.raises(
        CoordinateFrameRecordError,
        match="array_changed_during_binding",
    ):
        bind_body_frame_geometry(
            body_inputs["frame_node"],
            origin_xy_node=body_inputs["origin"],
            forward_axis_xy_node=mutating_forward,
            left_axis_xy_node=body_inputs["left"],
            axis_valid_node=body_inputs["valid"],
            row_identity=body_inputs["identity"],
            estimator_source=body_inputs["estimator_source"],
        )


def test_body_supports_exact_physical_mm_source(
    physical_inputs: dict[str, Any],
) -> None:
    _, physical = _stamp_physical(physical_inputs)
    rowset, key, identity = _identity(
        token=physical_inputs["token"], rowset_path="analysis/body_physical_source"
    )
    coordinates = _Node(
        f"{rowset.path}/positions_mm",
        token=physical_inputs["token"],
        data=np.array([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2]], dtype=np.float32),
    )
    descriptor = build_bound_canonical_coordinate_descriptor(
        coordinates,
        profile_id="physical_mm.source_camera_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("mm", "mm"),
        pixel_convention="not_applicable",
        row_identity=identity,
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        frame_record=physical,
    )
    stamp_bound_canonical_coordinate_descriptor(descriptor)
    source = bind_body_source_coordinate_descriptor(
        coordinates,
        row_identity=identity,
        physical_frame=physical,
    )
    assert source.coordinate_units == "mm"
    assert source.positive_y == "down"
    assert source.physical_frame is physical
    assert source.archive_identity == identity.archive_identity
    assert key.path.endswith("/instance_key")


@pytest.mark.parametrize(
    "profile_id",
    (
        "roi_local_px.top_left_y_down.v1",
        "detector_model_input_px.top_left_y_down.v1",
    ),
)
def test_body_rejects_roi_and_model_sources_without_typed_directed_lineage(
    physical_inputs: dict[str, Any], profile_id: str
) -> None:
    rowset, _, identity = _identity(
        token=physical_inputs["token"],
        rowset_path=f"analysis/unsupported_{profile_id.split('.')[0]}",
    )
    coordinates = _Node(
        f"{rowset.path}/positions_px",
        token=physical_inputs["token"],
        data=np.zeros((3, 2), dtype=np.float32),
    )
    source_frame = physical_inputs["source"]
    descriptor = build_canonical_coordinate_descriptor(
        profile_id=profile_id,
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        reference_width=source_frame.endpoint.width,
        reference_height=source_frame.endpoint.height,
        reference_authority=DigestBoundCoordinateRecordRef(
            source_frame.record_ref, source_frame.record_sha256
        ),
        reference_selector="record",
        pixel_convention="continuous",
        row_identity_contract=identity.contract,
        row_identity_record_ref=identity.record_ref,
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=source_frame.record_ref,
            record_sha256=source_frame.record_sha256,
        ),
    )
    coordinates.attrs.update(canonical_coordinate_descriptor_v2_attrs(descriptor))
    with pytest.raises(
        CoordinateFrameRecordError, match="body_source_profile_unsupported"
    ):
        bind_body_source_coordinate_descriptor(
            coordinates,
            row_identity=identity,
        )


def test_body_record_rejects_source_descriptor_mismatch(
    body_inputs: dict[str, Any],
) -> None:
    payload = body_inputs["record"].to_dict()
    payload["source_descriptor"]["record_ref"] = "/analysis/other@coordinate_descriptor"
    forged = parse_fish_anatomical_body_frame_record(payload)
    with pytest.raises(CoordinateFrameRecordError, match="body_authority_mismatch"):
        stamp_fish_anatomical_body_frame_record(
            body_inputs["frame_node"],
            forged,
            expected_record_ref=f"/{body_inputs['frame_node'].path}@{FISH_ANATOMICAL_BODY_FRAME_ATTR}",
            body_frame_contract=body_inputs["contract"],
            estimator_source=body_inputs["estimator_source"],
            geometry=body_inputs["geometry"],
            row_identity=body_inputs["identity"],
        )


def test_body_record_rejects_arena_physical_profile_without_arena_authority(
    body_inputs: dict[str, Any],
) -> None:
    payload = body_inputs["record"].to_dict()
    payload["source_profile_id"] = "physical_mm.arena_y_down.v1"
    with pytest.raises(
        CoordinateFrameRecordError,
        match="body_source_profile_unsupported",
    ):
        parse_fish_anatomical_body_frame_record(payload)


def test_body_rejects_cross_archive_contract_even_with_same_paths(
    body_inputs: dict[str, Any],
) -> None:
    other_contract_node = _Node(body_inputs["contract_node"].path, token=object())
    other_contract = stamp_body_frame_contract(
        other_contract_node,
        record=build_body_frame_contract_record(),
    )
    with pytest.raises(CoordinateFrameRecordError, match="archive_mismatch"):
        build_fish_anatomical_body_frame_record(
            frame_id="cross_archive",
            origin_definition="eye_pair_midpoint",
            body_frame_contract=other_contract,
            estimator_source=body_inputs["estimator_source"],
            geometry=body_inputs["geometry"],
            row_identity=body_inputs["identity"],
        )


def test_body_rejects_self_overwrite_and_dependency_alias(
    body_inputs: dict[str, Any],
) -> None:
    frame_contract = stamp_body_frame_contract(
        body_inputs["frame_node"],
        record=build_body_frame_contract_record(),
    )
    before = copy.deepcopy(dict(body_inputs["frame_node"].attrs))
    with pytest.raises(CoordinateFrameRecordError, match="dependency_cycle"):
        build_fish_anatomical_body_frame_record(
            frame_id="self_overwrite",
            origin_definition="eye_pair_midpoint",
            body_frame_contract=frame_contract,
            estimator_source=body_inputs["estimator_source"],
            geometry=body_inputs["geometry"],
            row_identity=body_inputs["identity"],
        )
    assert dict(body_inputs["frame_node"].attrs) == before

    aliased_estimator = stamp_body_frame_estimator(
        body_inputs["contract_node"],
        record=body_inputs["estimator"].record,
    )
    aliased_manifest = stamp_and_bind_persisted_coordinate_record(
        body_inputs["rowset"],
        build_body_estimator_source_manifest_record(
            method="keypoint_head_axis",
            source_descriptor=body_inputs["source"],
            estimator=aliased_estimator,
            source_schema=body_inputs["source_schema"],
            support_nodes={"validity": body_inputs["source_validity"]},
        ),
        attr_name=BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    )
    aliased_estimator_source = bind_keypoint_head_axis_source(
        source_descriptor=body_inputs["source"],
        estimator=aliased_estimator,
        keypoint_schema=body_inputs["source_schema"],
        validity_node=body_inputs["source_validity"],
        producer_manifest=aliased_manifest,
    )
    aliased_geometry = bind_body_frame_geometry(
        body_inputs["frame_node"],
        origin_xy_node=body_inputs["origin"],
        forward_axis_xy_node=body_inputs["forward"],
        left_axis_xy_node=body_inputs["left"],
        axis_valid_node=body_inputs["valid"],
        row_identity=body_inputs["identity"],
        estimator_source=aliased_estimator_source,
    )
    with pytest.raises(CoordinateFrameRecordError, match="dependency_alias"):
        build_fish_anatomical_body_frame_record(
            frame_id="dependency_alias",
            origin_definition="eye_pair_midpoint",
            body_frame_contract=body_inputs["contract"],
            estimator_source=aliased_estimator_source,
            geometry=aliased_geometry,
            row_identity=body_inputs["identity"],
        )


def test_body_dependency_graph_rejects_source_and_geometry_role_aliases(
    body_inputs: dict[str, Any],
) -> None:
    # The output frame itself cannot also be the exact source-coordinate array.
    source_frame = body_inputs["coordinates"]
    origin = _Node(
        f"{source_frame.path}/origin_xy",
        token=body_inputs["token"],
        data=body_inputs["origin"][:].copy(),
    )
    forward = _Node(
        f"{source_frame.path}/forward_axis_xy",
        token=body_inputs["token"],
        data=body_inputs["forward"][:].copy(),
    )
    left = _Node(
        f"{source_frame.path}/left_axis_xy",
        token=body_inputs["token"],
        data=body_inputs["left"][:].copy(),
    )
    valid = _Node(
        f"{source_frame.path}/axis_valid",
        token=body_inputs["token"],
        data=body_inputs["valid"][:].copy(),
    )
    self_source_geometry = bind_body_frame_geometry(
        source_frame,
        origin_xy_node=origin,
        forward_axis_xy_node=forward,
        left_axis_xy_node=left,
        axis_valid_node=valid,
        row_identity=body_inputs["identity"],
        estimator_source=body_inputs["estimator_source"],
    )
    with pytest.raises(CoordinateFrameRecordError, match="dependency_cycle"):
        build_fish_anatomical_body_frame_record(
            frame_id="self_source",
            origin_definition="eye_pair_midpoint",
            body_frame_contract=body_inputs["contract"],
            estimator_source=body_inputs["estimator_source"],
            geometry=self_source_geometry,
            row_identity=body_inputs["identity"],
        )

    geometry_aliased_contract = stamp_body_frame_contract(
        body_inputs["origin"],
        record=build_body_frame_contract_record(),
    )
    with pytest.raises(CoordinateFrameRecordError, match="dependency_alias"):
        build_fish_anatomical_body_frame_record(
            frame_id="geometry_alias",
            origin_definition="eye_pair_midpoint",
            body_frame_contract=geometry_aliased_contract,
            estimator_source=body_inputs["estimator_source"],
            geometry=body_inputs["geometry"],
            row_identity=body_inputs["identity"],
        )

    source_dependency_contract = stamp_body_frame_contract(
        body_inputs["source_frame_node"],
        record=build_body_frame_contract_record(),
    )
    with pytest.raises(CoordinateFrameRecordError, match="dependency_alias"):
        build_fish_anatomical_body_frame_record(
            frame_id="source_dependency_alias",
            origin_definition="eye_pair_midpoint",
            body_frame_contract=source_dependency_contract,
            estimator_source=body_inputs["estimator_source"],
            geometry=body_inputs["geometry"],
            row_identity=body_inputs["identity"],
        )


def test_body_stamp_reloads_all_evidence_and_rolls_back_on_cycle_or_stale_source(
    body_inputs: dict[str, Any],
) -> None:
    attrs = _MutatingAttrs({}, victim=body_inputs["coordinates"])
    body_inputs["frame_node"].attrs = attrs
    before = copy.deepcopy(dict(attrs))
    with pytest.raises(CoordinateFrameRecordError, match="stamp_preflight_failed"):
        _stamp_body(body_inputs)
    assert dict(attrs) == before


def test_body_bound_detects_stale_contract_and_estimator_evidence(
    body_inputs: dict[str, Any],
) -> None:
    bound = _stamp_body(body_inputs)
    body_inputs["contract_node"].attrs[BODY_FRAME_CONTRACT_ATTR]["angle_convention"] = (
        "wrong"
    )
    with pytest.raises(CoordinateFrameRecordError):
        verify_bound_coordinate_frame(
            bound, expected_kind=FISH_ANATOMICAL_BODY_FRAME_KIND
        )

    body_inputs["contract_node"].attrs[BODY_FRAME_CONTRACT_ATTR] = body_inputs[
        "contract"
    ].record.to_dict()
    body_inputs["contract_node"].attrs[f"{BODY_FRAME_CONTRACT_ATTR}_sha256"] = (
        body_inputs["contract"].record_sha256
    )
    body_inputs["estimator_node"].attrs[BODY_FRAME_ESTIMATOR_ATTR][
        "implementation_version"
    ] = "9.9.9"
    with pytest.raises(CoordinateFrameRecordError):
        verify_bound_coordinate_frame(
            bound, expected_kind=FISH_ANATOMICAL_BODY_FRAME_KIND
        )


def test_generic_caller_forgeable_evidence_api_is_not_exposed() -> None:
    import fisheye.shared.coordinate_frame_record as frame_records

    assert not hasattr(frame_records, "BoundFrameEvidence")
    assert not hasattr(frame_records, "load_bound_frame_evidence")


def test_real_zarr_store_identity_rejects_cross_root_composition() -> None:
    import zarr

    root = zarr.group()
    selected_node = (
        root.require_group("analysis")
        .require_group("calibration")
        .require_group("selected")
    )
    selected = stamp_selected_camera_frame_evidence(
        selected_node,
        source_camera=_verified_camera(),
    )
    metadata = {
        "schema_id": "palette.source_video_metadata.v2",
        "layout": "single_video",
        "camera_id": "2010093",
        "width": 160,
        "height": 120,
        "total_frames": 3,
        "locator": {
            "kind": "recording_relative",
            "relative_path": "camera.mp4",
        },
        "file_fingerprint": {
            "strategy": "size_mtime_sha256_v1",
            "value": "a" * 64,
            "size_bytes": 1234,
            "mtime_ns": 5678,
            "relocation_stable": False,
        },
    }
    root.attrs.update(
        {"recording_id": "recording-1", "source_video_metadata": metadata}
    )
    source_node = (
        root.require_group("analysis")
        .require_group("acquisition_camera_frames")
        .require_group("2010093")
    )
    acquisition = _stamp_external_acquisition(root, source_node)
    source_frame_node = (
        root.require_group("analysis")
        .require_group("coordinate_frames")
        .require_group("source_camera")
        .require_group("2010093")
        .require_group("continuous")
    )
    source = stamp_source_camera_pixel_frame_authority(
        source_frame_node,
        frame_id="camera_2010093_native",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )
    assert source.archive_identity == selected.archive_identity

    other_root = zarr.group()
    other_root.attrs.update(
        {
            "recording_id": "recording-1",
            "source_video_metadata": copy.deepcopy(metadata),
        }
    )
    other_source = (
        other_root.require_group("analysis")
        .require_group("acquisition_camera_frames")
        .require_group("2010093")
    )
    other_acquisition = _stamp_external_acquisition(other_root, other_source)
    other_frame_node = (
        other_root.require_group("analysis")
        .require_group("coordinate_frames")
        .require_group("source_camera")
        .require_group("2010093")
        .require_group("continuous")
    )
    other_frame = stamp_source_camera_pixel_frame_authority(
        other_frame_node,
        frame_id="camera_2010093_native",
        pixel_convention="continuous",
        acquisition_frame=other_acquisition,
    )
    with pytest.raises(CoordinateFrameRecordError, match="archive_mismatch"):
        build_physical_frame_calibration_record(
            frame_id="cross_archive",
            source_camera_pixels=other_frame,
            selected_camera_evidence=selected,
        )
