from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fisheye.shared import stimulus_coordinate_contract as contract


def _write_mapping(
    path: Path,
    *,
    values: np.ndarray = np.asarray([4, 7, 11], dtype="<i8"),
    row_identity_sha256: str = "1" * 64,
    row_contract_sha256: str = "2" * 64,
) -> None:
    values = np.asarray(values, dtype="<i8")
    with h5py.File(path, "w") as h5:
        tracking = h5.create_group("tracking_data")
        node = tracking.create_dataset(
            contract.SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=values,
            dtype="<i8",
        )
        record = {
            "schema_id": contract.SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
            "schema_version": contract.SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
            "mapping_method": "explicit_per_stimulus_state_v1",
            "source_rowset_ref": "/tracking_data/chaser_states",
            "source_row_identity_ref": "/tracking_data/stimulus_state_key",
            "source_row_identity_sha256": row_identity_sha256,
            "source_row_identity_contract_sha256": row_contract_sha256,
            "acquisition_recording_id": "recording-1",
            "acquisition_camera_id": "camera-1",
            "source_total_frames": 20,
            "target_domain": "acquisition_frame_index",
            "array_ref": contract.SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
            "array_dtype": np.dtype("<i8").str,
            "array_shape": [int(values.shape[0])],
            "array_content_sha256": contract.numpy_content_digest(values),
            "canonicalization": "canonical_json_sort_keys_v1",
        }
        node.attrs[contract.SOURCE_ACQUISITION_MAPPING_RECORD_ATTR] = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        node.attrs[
            contract.SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
        ] = contract.canonical_mapping_digest(record)


def test_load_source_acquisition_mapping_binds_exact_values_and_lineage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stimulus.h5"
    _write_mapping(path)

    with h5py.File(path, "r") as h5:
        values, record, digest = contract._load_source_acquisition_mapping(
            h5,
            row_count=3,
            row_identity_sha256="1" * 64,
            row_identity_contract_sha256="2" * 64,
        )

    np.testing.assert_array_equal(values, [4, 7, 11])
    assert values.flags.writeable is False
    assert record["acquisition_recording_id"] == "recording-1"
    assert record["acquisition_camera_id"] == "camera-1"
    assert digest == contract.canonical_mapping_digest(record)


def test_missing_source_acquisition_mapping_never_reinterprets_camera_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stimulus.h5"
    with h5py.File(path, "w") as h5:
        tracking = h5.create_group("tracking_data")
        tracking.create_dataset(
            "triggering_camera_frame_id",
            data=np.asarray([100, 101, 102], dtype="<i8"),
        )

    with h5py.File(path, "r") as h5, pytest.raises(
        contract.StimulusCoordinateContractError,
        match="never reinterpreted as acquisition_frame_index",
    ):
        contract._load_source_acquisition_mapping(
            h5,
            row_count=3,
            row_identity_sha256="1" * 64,
            row_identity_contract_sha256="2" * 64,
        )


def _write_target_mapping(path: Path) -> None:
    indices = np.asarray([4, 4, -1], dtype="<i8")
    valid = np.asarray([True, True, False], dtype=bool)
    with h5py.File(path, "w") as h5:
        tracking = h5.create_group("tracking_data")
        node = tracking.create_dataset(
            contract.TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=indices,
            dtype="<i8",
        )
        valid_node = tracking.create_dataset(
            contract.TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY,
            data=valid,
            dtype="bool",
        )
        record = {
            "schema_id": contract.TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
            "schema_version": (
                contract.TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION
            ),
            "mapping_method": (
                "explicit_per_stimulus_state_target_provenance_v1"
            ),
            "source_rowset_ref": "/tracking_data/chaser_states",
            "source_row_identity_ref": "/tracking_data/stimulus_state_key",
            "source_row_identity_sha256": "1" * 64,
            "source_row_identity_contract_sha256": "2" * 64,
            "source_target_frame_field": (
                "/tracking_data/chaser_states#target_source_frame_id"
            ),
            "source_target_camera_field": (
                "/tracking_data/chaser_states#target_source_camera_id"
            ),
            "acquisition_recording_id": "recording-1",
            "acquisition_camera_id": "camera-1",
            "source_total_frames": 20,
            "target_domain": "acquisition_frame_index",
            "array_ref": contract.TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
            "array_dtype": np.dtype("<i8").str,
            "array_shape": [3],
            "array_content_sha256": contract.numpy_content_digest(indices),
            "validity_array_ref": (
                contract.TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH
            ),
            "validity_array_dtype": np.dtype("bool").str,
            "validity_array_shape": [3],
            "validity_array_content_sha256": contract.numpy_content_digest(
                valid
            ),
            "invalid_index_sentinel": -1,
            "canonicalization": "canonical_json_sort_keys_v1",
        }
        digest = contract.canonical_mapping_digest(record)
        node.attrs[
            contract.TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR
        ] = json.dumps(record, separators=(",", ":"), sort_keys=True)
        node.attrs[
            contract.TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
        ] = digest
        valid_node.attrs[
            contract.TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_REF_ATTR
        ] = (
            f"{contract.TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH}@"
            f"{contract.TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR}"
        )
        valid_node.attrs[
            contract.TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
        ] = digest


def test_target_source_mapping_is_separate_and_allows_explicit_invalid_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "target_mapping.h5"
    _write_target_mapping(path)

    with h5py.File(path, "r") as h5:
        indices, valid, record, digest = (
            contract._load_target_source_acquisition_mapping(
                h5,
                row_count=3,
                row_identity_sha256="1" * 64,
                row_identity_contract_sha256="2" * 64,
            )
        )

    assert indices.tolist() == [4, 4, -1]
    assert valid.tolist() == [True, True, False]
    assert record["mapping_method"] == (
        "explicit_per_stimulus_state_target_provenance_v1"
    )
    assert digest == contract.canonical_mapping_digest(record)


def test_target_source_mapping_rejects_non_sentinel_invalid_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "target_mapping_bad_invalid.h5"
    _write_target_mapping(path)
    with h5py.File(path, "r+") as h5:
        h5[contract.TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH][2] = 0

    with h5py.File(path, "r") as h5, pytest.raises(
        contract.StimulusCoordinateContractError,
        match="exactly -1 when invalid",
    ):
        contract._load_target_source_acquisition_mapping(
            h5,
            row_count=3,
            row_identity_sha256="1" * 64,
            row_identity_contract_sha256="2" * 64,
        )


@pytest.mark.parametrize("mutation", ["digest", "row_identity", "range"])
def test_source_acquisition_mapping_fails_closed_on_stale_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / f"stimulus_{mutation}.h5"
    _write_mapping(path)
    with h5py.File(path, "r+") as h5:
        node = h5[contract.SOURCE_ACQUISITION_MAPPING_ARRAY_PATH]
        if mutation == "digest":
            node.attrs[
                contract.SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
            ] = "0" * 64
        elif mutation == "row_identity":
            record = json.loads(
                str(node.attrs[contract.SOURCE_ACQUISITION_MAPPING_RECORD_ATTR])
            )
            record["source_row_identity_sha256"] = "9" * 64
            node.attrs[contract.SOURCE_ACQUISITION_MAPPING_RECORD_ATTR] = (
                json.dumps(
                    record,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            node.attrs[
                contract.SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
            ] = contract.canonical_mapping_digest(record)
        else:
            node[2] = 99

    with h5py.File(path, "r") as h5, pytest.raises(
        contract.StimulusCoordinateContractError
    ):
        contract._load_source_acquisition_mapping(
            h5,
            row_count=3,
            row_identity_sha256="1" * 64,
            row_identity_contract_sha256="2" * 64,
        )


def test_destination_authority_must_match_recording_camera_extent_and_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition = SimpleNamespace(
        record=SimpleNamespace(
            recording_id="recording-1",
            camera_id="camera-1",
            source_total_frames=20,
            width_px=100,
            height_px=80,
        )
    )
    monkeypatch.setattr(
        contract,
        "load_persisted_acquisition_camera_authority",
        lambda root, *, expected_camera_id: (object(), acquisition),
    )
    preflight = SimpleNamespace(
        has_chaser_states=True,
        source_acquisition_mapping_record={
            "acquisition_recording_id": "recording-1",
            "acquisition_camera_id": "camera-1",
            "source_total_frames": 20,
        },
        selected_calibration=SimpleNamespace(
            active_camera_id="camera-1",
            source_camera=SimpleNamespace(native_width_px=100, native_height_px=80),
        ),
    )

    assert (
        contract.validate_stimulus_destination_acquisition_authority(
            object(),
            preflight=preflight,
        )
        is acquisition
    )

    preflight.source_acquisition_mapping_record["source_total_frames"] = 21
    with pytest.raises(
        contract.StimulusCoordinateContractError,
        match="disagree",
    ):
        contract.validate_stimulus_destination_acquisition_authority(
            object(),
            preflight=preflight,
        )
