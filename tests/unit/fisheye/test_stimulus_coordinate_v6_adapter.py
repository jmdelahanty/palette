"""Standalone, pre-materialization Citrus v6 validator fixture.

The fixture intentionally has no historical v5 calibration layout.  It proves
the separate validator only; normal v5 preflight remains fail-closed for v6.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from fisheye.shared.stimulus_coordinate_contract import (
    StimulusCoordinateContractError,
    preflight_stimulus_coordinate_contract,
)


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: bytes) -> str:
    return f"sha256:{sha256(value).hexdigest()}"


def _array_digest(values: np.ndarray, *, descriptor: object | None = None) -> str:
    array = np.ascontiguousarray(values)
    header = _canonical({
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "dtype": np.lib.format.dtype_to_descr(array.dtype) if descriptor is None else descriptor,
        "shape": [int(value) for value in array.shape],
    }).encode("utf-8")
    return _sha(header + b"\x00" + array.tobytes(order="C"))


def _packed_rows_digest(values: np.ndarray, fields: list[str], dtypes: dict[str, str]) -> str:
    header = _canonical({"canonicalization": "packed_row_fields_little_endian_v1", "dtype": dtypes, "shape": [int(values.shape[0])]})
    payload = bytearray(header.encode("utf-8") + b"\x00")
    for row in values:
        for field in fields:
            payload.extend(np.asarray(row[field], dtype=np.dtype(dtypes[field])).tobytes())
    return _sha(bytes(payload))


def _attr_json(node: h5py.Group | h5py.Dataset, name: str, value: object) -> None:
    node.attrs[name] = _canonical(value)


def _write_v6_fixture(path: Path, *, gap: bool = False, orphan: bool = False) -> None:
    fixture = json.loads(
        (Path(__file__).parents[2] / "fixtures" / "stimulus_coordinate_v6_golden_oracle.json").read_text()
    )
    orange = fixture["orange_mapping"]
    oracle = fixture["oracle"]
    if gap:
        orange["camera_streams"]["CAM-42"]["coverage"]["gap_count"] = 1
    source_text = _canonical(orange)
    source_digest = _sha(source_text.encode("utf-8"))
    normalized = {
        "canonicalization": "canonical_json_utf8_sort_keys_compact_v1",
        "orange_source_record": orange,
        "orange_source_record_sha256": source_digest,
        "schema_id": "citrus.stimulus_coordinate_v6.mapping_normalized",
        "schema_version": 1,
    }
    normalized_text = _canonical(normalized)
    normalized_digest = _sha(normalized_text.encode("utf-8"))
    with h5py.File(path, "w") as h5:
        receipt = h5.create_group("stimulus_coordinate_v6")
        receipt.attrs.update({
            "schema_id": "citrus.stimulus_coordinate_v6.receipt", "schema_version": 1,
            "status": "sealed", "reason_code": "sealed", "recording_id": "recording-42",
            "camera_serial": "CAM-42", "source_record_observed_sha256": source_digest,
            "source_record_declared_sha256": source_digest,
            "normalized_record_sha256": normalized_digest,
        })
        strings = h5py.string_dtype(encoding="utf-8")
        source = receipt.create_dataset("source_semantic_record_json", data=source_text, dtype=strings)
        source.attrs.update({"observed_checksum_sha256": source_digest, "declared_checksum_sha256": source_digest, "serialization": "exact_source_bytes"})
        normalized_node = receipt.create_dataset("normalized_semantic_record_json", data=normalized_text, dtype=strings)
        normalized_node.attrs.update({"checksum_sha256": normalized_digest, "canonicalization": "canonical_json_utf8_sort_keys_compact_v1"})

        renderer = h5.create_group("stimulus_renderer_snapshot")
        renderer.attrs.update({"schema_id": "citrus.stimulus_renderer_snapshot", "schema_version": 1, "capture_phase": "experiment_start_after_arena_initialization"})
        arena_renderer = renderer.create_group("arena_1")
        arena_renderer.attrs.update({"active_stimulus_mode": "chaser", "texture_width_px": 348, "texture_height_px": 348, "texture_origin": "top_left"})
        custom = arena_renderer.create_group("custom_coordinates")
        custom.attrs.update({"texture_center_x": 174.0, "texture_center_y": 174.0})

        calibration = h5.create_group("calibration_snapshot")
        arena = calibration.create_group("arena_geometry")
        authority = {
            "arena_id": "arena_1", "camera_serial": "CAM-42",
            "coordinate_frame": "arena_relative_canvas_px", "origin": "top_left_of_active_arena",
            "units": "px", "x_axis": "right", "y_axis": "down",
            "native_source_extent_px": [1920, 1080], "final_display_extent_px": [348, 348],
            "runtime_geometry_contract_ref": "/runtime_geometry_contract", "runtime_geometry_contract_sha256": "sha256:" + "3" * 64,
            "calibration_authority_ref": "/calibration_snapshot", "calibration_authority_sha256": "sha256:" + "4" * 64,
        }
        arena.attrs.update({"arena_region_width_px": 348, "arena_region_height_px": 348, "arena_origin_in_canvas_x_px": 0.0, "arena_origin_in_canvas_y_px": 0.0, "coordinate_frame_authority": _canonical(authority), "coordinate_frame_authority_sha256": _sha(_canonical(authority).encode("utf-8"))})
        for name, direction, source_space, destination_space in (
            ("source_camera_to_final_display_homography", "source_camera_to_final_display", "source_camera_px", "final_display_canvas_px"),
            ("final_display_to_source_camera_homography", "final_display_to_source_camera", "final_display_canvas_px", "source_camera_px"),
        ):
            values = np.eye(3, dtype="<f8")
            node = arena.create_dataset(name, data=values, dtype="<f8")
            node.attrs.update({"content_sha256": _array_digest(values), "transform_direction": direction, "source_space": source_space, "destination_space": destination_space})

        frames_dtype = np.dtype([("stimulus_frame_num", "<u8"), ("source_recording_frame_id", "<u8"), ("source_acquisition_frame_index", "<i8")])
        frames = np.asarray([(0, 1, 0), (1, 2, 1)], dtype=frames_dtype)
        video = h5.create_group("video_metadata")
        frame_node = video.create_dataset("frame_metadata", data=frames)
        frame_dtype_map = {"stimulus_frame_num": "<u8", "source_recording_frame_id": "<u8", "source_acquisition_frame_index": "<i8"}
        frame_order = list(frame_dtype_map)
        frame_contract = {"schema_id": "citrus.stimulus_frame_metadata.contract", "schema_version": 1, "dtype": frame_dtype_map, "field_order": frame_order, "shape": [2], "rows_packed_le_sha256": _packed_rows_digest(frames, frame_order, frame_dtype_map)}
        frame_node.attrs.update({"schema_id": "citrus.stimulus_frame_metadata", "schema_version": 2, "source_acquisition_mapping_ref": "/stimulus_coordinate_v6/source_semantic_record_json", "dataset_contract": _canonical(frame_contract), "dataset_contract_sha256": _sha(_canonical(frame_contract).encode("utf-8"))})

        tracking = h5.create_group("tracking_data")
        states_dtype = np.dtype([
            ("stimulus_frame_num", "<u8"), ("chaser_index", "<i8"),
            ("chaser_pos_x", "<f4"), ("chaser_pos_y", "<f4"),
            ("target_pos_x", "<f4"), ("target_pos_y", "<f4"),
            ("target_clamped_pos_x", "<f4"), ("target_clamped_pos_y", "<f4"),
        ])
        states = np.asarray([(0, 0, 10, 11, 50, 51, 50, 51), (0, 1, 20, 21, 50, 51, 50, 51), (1, 0, 12, 13, 50, 51, 50, 51), (1, 1, 22, 23, 0, 0, 0, 0)], dtype=states_dtype)
        state_node = tracking.create_dataset("chaser_states", data=states)
        keys = np.asarray(oracle["chaser_state_keys"], dtype="<i8")
        if orphan:
            keys[2, 1] = 2
        key_node = tracking.create_dataset("stimulus_state_key", data=keys, dtype="<i8")
        key_digest = _array_digest(keys)
        row_contract = {"schema_id": "citrus.tracking.chaser_states.row_identity", "schema_version": 1, "components": ["chaser_index", "stimulus_frame_num"], "key_array_ref": "/tracking_data/stimulus_state_key", "key_array_sha256": key_digest}
        key_node.attrs.update({"row_identity_contract": _canonical(row_contract), "row_identity_contract_sha256": _sha(_canonical(row_contract).encode("utf-8")), "content_sha256": key_digest})
        source_indices = np.asarray(oracle["source_acquisition_frame_index"], dtype="<i8")
        target_indices = np.asarray(oracle["target_source_acquisition_frame_index"], dtype="<i8")
        valid = np.asarray(oracle["target_source_acquisition_frame_valid"], dtype="u1")
        source_node = tracking.create_dataset("source_acquisition_frame_index", data=source_indices, dtype="<i8")
        target_node = tracking.create_dataset("target_source_acquisition_frame_index", data=target_indices, dtype="<i8")
        valid_node = tracking.create_dataset("target_source_acquisition_frame_valid", data=valid, dtype="u1")
        source_array_digest = _array_digest(source_indices)
        target_array_digest = _array_digest(target_indices)
        valid_digest = _array_digest(valid, descriptor="|u1")
        source_node.attrs.update({"content_sha256": source_array_digest, "logical_name": "current_orange_recording_acquisition"})
        target_node.attrs.update({"content_sha256": target_array_digest, "logical_name": "held_target_orange_recording_acquisition"})
        valid_node.attrs.update({"content_sha256": valid_digest, "logical_type": "bool", "invalid_sentinel": -1})
        descriptor = {"schema_id": "citrus.tracking.chaser_states.coordinate_descriptor", "schema_version": 1, "canonicalization": "canonical_json_utf8_sort_keys_compact_v1", "coordinate_space": "arena_relative_canvas_px", "origin": "top_left_of_active_arena", "units": "px", "x_axis": "right", "y_axis": "down"}
        manifest = {"schema_id": "citrus.tracking.chaser_states.surface_manifest", "schema_version": 1, "canonicalization": "canonical_json_utf8_sort_keys_compact_v1", "row_identity": ["chaser_index", "stimulus_frame_num"], "surfaces": {"chaser_position_xy": ["chaser_pos_x", "chaser_pos_y"], "target_position_xy": ["target_pos_x", "target_pos_y"], "target_clamped_position_xy": ["target_clamped_pos_x", "target_clamped_pos_y"]}}
        source_mapping = {"schema_id": "citrus.stimulus_source_acquisition_mapping", "schema_version": 2, "array_ref": "/tracking_data/source_acquisition_frame_index", "array_sha256": source_array_digest, "camera_serial": "CAM-42", "orange_source_record_sha256": source_digest, "recording_id": "recording-42", "row_identity_ref": "/tracking_data/stimulus_state_key", "source_total_frames": 2}
        target_mapping = {"schema_id": "citrus.stimulus_target_source_acquisition_mapping", "schema_version": 2, "array_ref": "/tracking_data/target_source_acquisition_frame_index", "array_sha256": target_array_digest, "validity_array_ref": "/tracking_data/target_source_acquisition_frame_valid", "validity_array_sha256": valid_digest, "invalid_sentinel": -1, "recording_id": "recording-42", "camera_serial": "CAM-42", "source_total_frames": 2, "row_identity_ref": "/tracking_data/stimulus_state_key", "orange_source_record_sha256": source_digest}
        state_dtype_map = {"stimulus_frame_num": "<u8", "chaser_index": "<i8", "chaser_pos_x": "<f4", "chaser_pos_y": "<f4", "target_pos_x": "<f4", "target_pos_y": "<f4", "target_clamped_pos_x": "<f4", "target_clamped_pos_y": "<f4"}
        state_order = list(state_dtype_map)
        state_contract = {"schema_id": "citrus.tracking.chaser_states.dataset_contract", "schema_version": 1, "dtype": state_dtype_map, "field_order": state_order, "shape": [4], "rows_packed_le_sha256": _packed_rows_digest(states, state_order, state_dtype_map)}
        state_node.attrs.update({"schema_id": "citrus.tracking.chaser_states", "schema_version": 6, "coordinate_descriptor": _canonical(descriptor), "coordinate_descriptor_sha256": _sha(_canonical(descriptor).encode("utf-8")), "coordinate_surface_manifest": _canonical(manifest), "coordinate_surface_manifest_sha256": _sha(_canonical(manifest).encode("utf-8")), "source_acquisition_mapping_record": _canonical(source_mapping), "source_acquisition_mapping_record_sha256": _sha(_canonical(source_mapping).encode("utf-8")), "target_source_acquisition_mapping_record": _canonical(target_mapping), "target_source_acquisition_mapping_record_sha256": _sha(_canonical(target_mapping).encode("utf-8")), "dataset_contract": _canonical(state_contract), "dataset_contract_sha256": _sha(_canonical(state_contract).encode("utf-8"))})


def test_v6_validator_accepts_closed_citrus_two_frame_two_chaser_fixture(tmp_path: Path) -> None:
    path = tmp_path / "citrus_v6.h5"
    _write_v6_fixture(path)
    with h5py.File(path, "r") as h5:
        from fisheye.shared.stimulus_coordinate_contract import validate_citrus_stimulus_coordinate_v6_artifact
        artifact = validate_citrus_stimulus_coordinate_v6_artifact(h5, source_h5=path)
    assert artifact.stimulus_state_key.tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert artifact.source_acquisition_frame_index.tolist() == [0, 0, 1, 1]
    assert artifact.target_source_acquisition_frame_index.tolist() == [0, 0, 0, -1]
    assert artifact.target_source_acquisition_frame_valid.tolist() == [True, True, True, False]
    assert artifact.source_semantic_record_sha256.startswith("sha256:")
    assert artifact.status == "pre_materialization_only"

    # v6 is not silently passed through the immutable v5 materialization
    # path.  Normal import remains fail-closed until its end-to-end
    # normalization bridge (calibration, transform, and destination authority)
    # is implemented.
    with h5py.File(path, "r") as h5, pytest.raises(StimulusCoordinateContractError):
        preflight_stimulus_coordinate_contract(h5, source_h5=path)


def test_v6_validator_consumes_citrus_generated_golden_h5() -> None:
    """Cross-repository seam: no hand-written H5 is substituted here."""
    path = Path(os.environ.get("CITRUS_V6_GOLDEN_H5", "/tmp/citrus_v6_review_golden.h5"))
    if not path.exists():
        pytest.skip("Citrus v6 golden H5 has not been generated in /tmp")
    from fisheye.shared.stimulus_coordinate_contract import (
        validate_citrus_stimulus_coordinate_v6_artifact,
    )
    with h5py.File(path, "r") as h5:
        artifact = validate_citrus_stimulus_coordinate_v6_artifact(h5, source_h5=path)
    assert artifact.status == "pre_materialization_only"
    assert artifact.stimulus_state_key.tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert artifact.source_acquisition_frame_index.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("kwargs, match", [
    ({"gap": True}, "coverage"),
    ({"orphan": True}, "composite stimulus-state key|terminal orphan"),
])
def test_preflight_rejects_unsealed_or_orphan_v6_evidence(tmp_path: Path, kwargs: dict[str, bool], match: str) -> None:
    path = tmp_path / "bad_citrus_v6.h5"
    _write_v6_fixture(path, **kwargs)
    with h5py.File(path, "r") as h5, pytest.raises(StimulusCoordinateContractError, match=match):
        from fisheye.shared.stimulus_coordinate_contract import validate_citrus_stimulus_coordinate_v6_artifact
        validate_citrus_stimulus_coordinate_v6_artifact(h5, source_h5=path)
