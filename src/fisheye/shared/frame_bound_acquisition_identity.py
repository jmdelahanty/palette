"""Strict loader-minted evidence for Citrus frame-bound acquisition identity v6.

The v6 companion seals the current-controller-input row conversion, the exact
Orange recording identity carried by every Shaman-v2 slot, the acquisition
camera tuple, and the exact raw Citrus H5 artifact.  It does not prove physical
display presentation; that scientific limit remains literal in every binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np

from fisheye.shared.stimulus_coordinate_contract import (
    STIMULUS_COORDINATE_V6_FRAME_METADATA_PATH,
    StimulusCoordinateContractError,
    canonical_mapping_digest,
    numpy_content_digest,
    validate_citrus_stimulus_coordinate_v6_artifact,
)


FRAME_BOUND_ACQUISITION_IDENTITY_SCHEMA_ID = (
    "palette.frame_bound_acquisition_identity"
)
FRAME_BOUND_ACQUISITION_IDENTITY_SCHEMA_VERSION = 1
FRAME_BOUND_ACQUISITION_MAPPING_POLICY = (
    "orange_recording_frame_id_minus_one_finalized_dense_v1"
)
FRAME_BOUND_RECORDING_MEMBERSHIP_STATUS = (
    "per_slot_recording_identity_token_verified"
)
FRAME_BOUND_TEMPORAL_ALIGNMENT_CLASS = "controller_input_provenance_proxy"
FRAME_BOUND_SCIENTIFIC_USE_CLASS = "exploratory_proxy"
FRAME_BOUND_PROMOTION_BLOCKER = (
    "controlled_four_camera_hardware_validation_pending"
)
PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_ID = (
    "palette.paired_frame_bound_chaser_source"
)
PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_VERSION = 1


def _readonly(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _required_text(value: Any, *, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StimulusCoordinateContractError(
                f"{label} must be UTF-8 text."
            ) from exc
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StimulusCoordinateContractError(
            f"{label} must be a non-empty trimmed string."
        )
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


@dataclass(frozen=True)
class FrameBoundAcquisitionIdentityEvidence:
    """Immutable v6 current-input identity evidence and its scientific limits."""

    source_h5: Path
    source_file_identity: Mapping[str, Any]
    source_snapshot_sha256: str
    source_contract_sha256: str
    recording_id: str
    camera_serial: str
    acquisition_camera_id: str
    shaman_numeric_camera_id: int
    source_total_frames: int
    orange_source_record_sha256: str
    normalized_source_record_sha256: str
    raw_citrus_h5_binding: Mapping[str, Any]
    raw_citrus_h5_binding_sha256: str
    acquisition_camera_binding: Mapping[str, Any]
    acquisition_camera_binding_sha256: str
    orange_recording_identity: Mapping[str, Any]
    orange_recording_identity_sha256: str
    recording_identity_token: str
    stimulus_frame_num: np.ndarray = field(repr=False, compare=False)
    source_recording_frame_id: np.ndarray = field(repr=False, compare=False)
    frame_source_acquisition_frame_index: np.ndarray = field(
        repr=False, compare=False
    )
    stimulus_state_key: np.ndarray = field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = field(
        repr=False, compare=False
    )
    target_source_acquisition_frame_index: np.ndarray = field(
        repr=False, compare=False
    )
    target_source_acquisition_frame_valid: np.ndarray = field(
        repr=False, compare=False
    )
    chaser_position_xy: np.ndarray = field(repr=False, compare=False)
    target_position_xy: np.ndarray = field(repr=False, compare=False)
    target_clamped_position_xy: np.ndarray = field(repr=False, compare=False)
    source_camera_to_final_display_homography: np.ndarray = field(
        repr=False, compare=False
    )
    final_display_to_source_camera_homography: np.ndarray = field(
        repr=False, compare=False
    )

    @property
    def selector_eligible(self) -> bool:
        return False

    @property
    def physical_presentation_verified(self) -> bool:
        return False

    @property
    def presentation_timestamp_available(self) -> bool:
        return False

    @property
    def per_slot_recording_membership_verified(self) -> bool:
        return True

    def source_binding(self) -> dict[str, Any]:
        """Return the bounded, array-free binding safe for manifests/receipts."""

        return {
            "schema_id": FRAME_BOUND_ACQUISITION_IDENTITY_SCHEMA_ID,
            "schema_version": FRAME_BOUND_ACQUISITION_IDENTITY_SCHEMA_VERSION,
            "source_h5": str(self.source_h5),
            "source_file_identity": _plain(self.source_file_identity),
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "recording_id": self.recording_id,
            "camera_serial": self.camera_serial,
            "acquisition_camera_id": self.acquisition_camera_id,
            "shaman_numeric_camera_id": self.shaman_numeric_camera_id,
            "source_total_frames": self.source_total_frames,
            "orange_source_record_sha256": self.orange_source_record_sha256,
            "normalized_source_record_sha256": (
                self.normalized_source_record_sha256
            ),
            "raw_citrus_h5_binding": _plain(self.raw_citrus_h5_binding),
            "raw_citrus_h5_binding_sha256": self.raw_citrus_h5_binding_sha256,
            "acquisition_camera_binding": _plain(
                self.acquisition_camera_binding
            ),
            "acquisition_camera_binding_sha256": (
                self.acquisition_camera_binding_sha256
            ),
            "orange_recording_identity": _plain(
                self.orange_recording_identity
            ),
            "orange_recording_identity_sha256": (
                self.orange_recording_identity_sha256
            ),
            "recording_identity_token": self.recording_identity_token,
            "mapping_policy": FRAME_BOUND_ACQUISITION_MAPPING_POLICY,
            "finalized_mapping_record_verified": True,
            "row_number_conversion_verified": True,
            "recording_scoped_acquisition_row_identity_verified": True,
            "raw_h5_exact_binding_verified": True,
            "recording_membership_status": (
                FRAME_BOUND_RECORDING_MEMBERSHIP_STATUS
            ),
            "per_slot_recording_membership_verified": True,
            "temporal_alignment_class": FRAME_BOUND_TEMPORAL_ALIGNMENT_CLASS,
            "scientific_use_class": FRAME_BOUND_SCIENTIFIC_USE_CLASS,
            "physical_presentation_verified": False,
            "presentation_timestamp_available": False,
            "camera_presentation_clock_transform_available": False,
            "selector_eligible": False,
            "promotion_blocker": FRAME_BOUND_PROMOTION_BLOCKER,
            "arrays": {
                "stimulus_frame_num_sha256": numpy_content_digest(
                    self.stimulus_frame_num
                ),
                "source_recording_frame_id_sha256": numpy_content_digest(
                    self.source_recording_frame_id
                ),
                "frame_source_acquisition_frame_index_sha256": (
                    numpy_content_digest(
                        self.frame_source_acquisition_frame_index
                    )
                ),
                "stimulus_state_key_sha256": numpy_content_digest(
                    self.stimulus_state_key
                ),
                "source_acquisition_frame_index_sha256": numpy_content_digest(
                    self.source_acquisition_frame_index
                ),
                "target_source_acquisition_frame_index_sha256": (
                    numpy_content_digest(
                        self.target_source_acquisition_frame_index
                    )
                ),
                "target_source_acquisition_frame_valid_sha256": (
                    numpy_content_digest(
                        self.target_source_acquisition_frame_valid
                    )
                ),
                "chaser_position_xy_sha256": numpy_content_digest(
                    self.chaser_position_xy
                ),
                "target_position_xy_sha256": numpy_content_digest(
                    self.target_position_xy
                ),
                "target_clamped_position_xy_sha256": numpy_content_digest(
                    self.target_clamped_position_xy
                ),
                "source_camera_to_final_display_homography_sha256": (
                    numpy_content_digest(
                        self.source_camera_to_final_display_homography
                    )
                ),
                "final_display_to_source_camera_homography_sha256": (
                    numpy_content_digest(
                        self.final_display_to_source_camera_homography
                    )
                ),
            },
        }

    def assert_current(self) -> "FrameBoundAcquisitionIdentityEvidence":
        """Reopen and revalidate the exact companion before scientific use."""

        current = load_frame_bound_acquisition_identity(
            self.source_h5,
            expected_recording_id=self.recording_id,
            expected_camera_serial=self.camera_serial,
            expected_acquisition_camera_id=self.acquisition_camera_id,
            expected_shaman_numeric_camera_id=self.shaman_numeric_camera_id,
            expected_source_total_frames=self.source_total_frames,
        )
        if (
            current.source_file_identity != self.source_file_identity
            or current.source_snapshot_sha256 != self.source_snapshot_sha256
            or current.source_contract_sha256 != self.source_contract_sha256
        ):
            raise StimulusCoordinateContractError(
                "Frame-bound acquisition identity source changed after loading."
            )
        return current


def validate_frame_bound_acquisition_identity(
    h5: h5py.File,
    *,
    source_h5: Path,
    expected_recording_id: str,
    expected_camera_serial: str,
    expected_acquisition_camera_id: str,
    expected_shaman_numeric_camera_id: int,
    expected_source_total_frames: int,
) -> FrameBoundAcquisitionIdentityEvidence:
    """Validate v6 and bind it to explicit destination recording expectations."""

    expected_recording = _required_text(
        expected_recording_id, label="expected_recording_id"
    )
    expected_camera = _required_text(
        expected_camera_serial, label="expected_camera_serial"
    )
    expected_acquisition_camera = _required_text(
        expected_acquisition_camera_id,
        label="expected_acquisition_camera_id",
    )
    if (
        isinstance(expected_shaman_numeric_camera_id, bool)
        or not isinstance(expected_shaman_numeric_camera_id, int)
        or expected_shaman_numeric_camera_id < 0
        or expected_shaman_numeric_camera_id > (2**32 - 1)
    ):
        raise StimulusCoordinateContractError(
            "expected_shaman_numeric_camera_id must be a uint32 integer."
        )
    if (
        isinstance(expected_source_total_frames, bool)
        or not isinstance(expected_source_total_frames, int)
        or expected_source_total_frames <= 0
    ):
        raise StimulusCoordinateContractError(
            "expected_source_total_frames must be a positive integer."
        )

    artifact = validate_citrus_stimulus_coordinate_v6_artifact(
        h5, source_h5=source_h5
    )
    recording_id = _required_text(
        artifact.orange_recording_identity.get("recording_id"),
        label="v6 Orange recording identity recording_id",
    )
    camera_serial = _required_text(
        artifact.acquisition_camera_binding.get("camera_serial"),
        label="v6 acquisition-camera binding camera_serial",
    )
    acquisition_camera_id = _required_text(
        artifact.acquisition_camera_binding.get("acquisition_camera_id"),
        label="v6 acquisition_camera_id",
    )
    shaman_numeric_camera_id = artifact.acquisition_camera_binding.get(
        "shaman_numeric_camera_id"
    )
    streams = artifact.source_semantic_record.get("camera_streams")
    if not isinstance(streams, Mapping) or camera_serial not in streams:
        raise StimulusCoordinateContractError(
            "v6 source record lacks the selected camera stream."
        )
    stream = streams[camera_serial]
    if not isinstance(stream, Mapping) or not isinstance(
        stream.get("coverage"), Mapping
    ):
        raise StimulusCoordinateContractError(
            "v6 selected camera coverage is malformed."
        )
    source_total_frames = stream["coverage"].get("total_acquisitions")
    if (
        recording_id != expected_recording
        or camera_serial != expected_camera
        or acquisition_camera_id != expected_acquisition_camera
        or shaman_numeric_camera_id != expected_shaman_numeric_camera_id
        or source_total_frames != expected_source_total_frames
    ):
        raise StimulusCoordinateContractError(
            "v6 recording, camera, or acquisition population differs from "
            "the explicit destination authority."
        )

    frame_values = np.asarray(h5[STIMULUS_COORDINATE_V6_FRAME_METADATA_PATH][:])
    states = np.asarray(h5["/tracking_data/chaser_states"][:])
    arena = h5["/calibration_snapshot/arena_geometry"]
    source_to_display = np.asarray(
        arena["source_camera_to_final_display_homography"][:], dtype="<f8"
    )
    display_to_source = np.asarray(
        arena["final_display_to_source_camera_homography"][:], dtype="<f8"
    )
    arrays = {
        "stimulus_frame_num": _readonly(
            frame_values["stimulus_frame_num"], dtype="<u8"
        ),
        "source_recording_frame_id": _readonly(
            frame_values["source_recording_frame_id"], dtype="<u8"
        ),
        "frame_source_acquisition_frame_index": _readonly(
            frame_values["source_acquisition_frame_index"], dtype="<i8"
        ),
        "stimulus_state_key": _readonly(
            artifact.stimulus_state_key, dtype="<i8"
        ),
        "source_acquisition_frame_index": _readonly(
            artifact.source_acquisition_frame_index, dtype="<i8"
        ),
        "target_source_acquisition_frame_index": _readonly(
            artifact.target_source_acquisition_frame_index, dtype="<i8"
        ),
        "target_source_acquisition_frame_valid": _readonly(
            artifact.target_source_acquisition_frame_valid, dtype=bool
        ),
        "chaser_position_xy": _readonly(
            np.column_stack(
                (states["chaser_pos_x"], states["chaser_pos_y"])
            ),
            dtype="<f4",
        ),
        "target_position_xy": _readonly(
            np.column_stack(
                (states["target_pos_x"], states["target_pos_y"])
            ),
            dtype="<f4",
        ),
        "target_clamped_position_xy": _readonly(
            np.column_stack(
                (
                    states["target_clamped_pos_x"],
                    states["target_clamped_pos_y"],
                )
            ),
            dtype="<f4",
        ),
        "source_camera_to_final_display_homography": _readonly(
            source_to_display, dtype="<f8"
        ),
        "final_display_to_source_camera_homography": _readonly(
            display_to_source, dtype="<f8"
        ),
    }
    snapshot_record = {
        "schema_id": FRAME_BOUND_ACQUISITION_IDENTITY_SCHEMA_ID,
        "schema_version": FRAME_BOUND_ACQUISITION_IDENTITY_SCHEMA_VERSION,
        "source_file_identity": dict(artifact.source_file_identity),
        "source_contract_sha256": artifact.source_contract_sha256,
        "recording_id": recording_id,
        "camera_serial": camera_serial,
        "acquisition_camera_id": acquisition_camera_id,
        "shaman_numeric_camera_id": shaman_numeric_camera_id,
        "source_total_frames": source_total_frames,
        "orange_source_record_sha256": artifact.source_semantic_record_sha256,
        "normalized_source_record_sha256": (
            artifact.normalized_semantic_record_sha256
        ),
        "raw_citrus_h5_binding": dict(artifact.raw_citrus_h5_binding),
        "raw_citrus_h5_binding_sha256": (
            artifact.raw_citrus_h5_binding_sha256
        ),
        "acquisition_camera_binding": dict(
            artifact.acquisition_camera_binding
        ),
        "acquisition_camera_binding_sha256": (
            artifact.acquisition_camera_binding_sha256
        ),
        "orange_recording_identity": dict(artifact.orange_recording_identity),
        "orange_recording_identity_sha256": (
            artifact.orange_recording_identity_sha256
        ),
        "recording_identity_token": artifact.recording_identity_token,
        "array_sha256": {
            name: numpy_content_digest(values) for name, values in arrays.items()
        },
        "mapping_policy": FRAME_BOUND_ACQUISITION_MAPPING_POLICY,
        "finalized_mapping_record_verified": True,
        "row_number_conversion_verified": True,
        "recording_scoped_acquisition_row_identity_verified": True,
        "raw_h5_exact_binding_verified": True,
        "recording_membership_status": FRAME_BOUND_RECORDING_MEMBERSHIP_STATUS,
        "per_slot_recording_membership_verified": True,
        "temporal_alignment_class": FRAME_BOUND_TEMPORAL_ALIGNMENT_CLASS,
        "physical_presentation_verified": False,
        "selector_eligible": False,
        "promotion_blocker": FRAME_BOUND_PROMOTION_BLOCKER,
    }
    return FrameBoundAcquisitionIdentityEvidence(
        source_h5=Path(source_h5).expanduser().resolve(),
        source_file_identity=_freeze(artifact.source_file_identity),
        source_snapshot_sha256=canonical_mapping_digest(snapshot_record),
        source_contract_sha256=artifact.source_contract_sha256,
        recording_id=recording_id,
        camera_serial=camera_serial,
        acquisition_camera_id=acquisition_camera_id,
        shaman_numeric_camera_id=shaman_numeric_camera_id,
        source_total_frames=source_total_frames,
        orange_source_record_sha256=artifact.source_semantic_record_sha256,
        normalized_source_record_sha256=(
            artifact.normalized_semantic_record_sha256
        ),
        raw_citrus_h5_binding=_freeze(artifact.raw_citrus_h5_binding),
        raw_citrus_h5_binding_sha256=(
            artifact.raw_citrus_h5_binding_sha256
        ),
        acquisition_camera_binding=_freeze(artifact.acquisition_camera_binding),
        acquisition_camera_binding_sha256=(
            artifact.acquisition_camera_binding_sha256
        ),
        orange_recording_identity=_freeze(artifact.orange_recording_identity),
        orange_recording_identity_sha256=(
            artifact.orange_recording_identity_sha256
        ),
        recording_identity_token=artifact.recording_identity_token,
        **arrays,
    )


def load_frame_bound_acquisition_identity(
    source_h5: Path,
    *,
    expected_recording_id: str,
    expected_camera_serial: str,
    expected_acquisition_camera_id: str,
    expected_shaman_numeric_camera_id: int,
    expected_source_total_frames: int,
) -> FrameBoundAcquisitionIdentityEvidence:
    """Open one immutable v6 companion and return a strict source handle."""

    path = Path(source_h5).expanduser().resolve()
    with h5py.File(path, "r") as h5:
        return validate_frame_bound_acquisition_identity(
            h5,
            source_h5=path,
            expected_recording_id=expected_recording_id,
            expected_camera_serial=expected_camera_serial,
            expected_acquisition_camera_id=expected_acquisition_camera_id,
            expected_shaman_numeric_camera_id=(
                expected_shaman_numeric_camera_id
            ),
            expected_source_total_frames=expected_source_total_frames,
        )


@dataclass(frozen=True)
class FrameBoundChaserSourceDimensions:
    """Native sample dimensions exposed to the pure proxy selector."""

    total_frames: int
    stimulus_sample_count: int
    chaser_count: int


@dataclass(frozen=True)
class PairedFrameBoundChaserSource:
    """Exact raw-H5/companion pair, shaped as a native chaser source handle."""

    companion: FrameBoundAcquisitionIdentityEvidence = field(
        repr=False, compare=False
    )
    recording_bundle_root: Path
    raw_h5: Path
    raw_file_identity: Mapping[str, Any]
    raw_h5_sha256: str
    recording_id: str
    dimensions: FrameBoundChaserSourceDimensions
    source_authority: Mapping[str, Any] = field(repr=False)
    source_authority_id: str
    source_authority_digest: str
    manifest_sha256: str
    verification_digest: str
    run_path: str
    stimulus_frame_num: np.ndarray = field(repr=False, compare=False)
    timestamp_ns_session: np.ndarray = field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = field(
        repr=False, compare=False
    )
    source_sample_row_index: np.ndarray = field(repr=False, compare=False)
    source_stimulus_run_row_index: np.ndarray = field(
        repr=False, compare=False
    )
    source_stimulus_source_row_index: np.ndarray = field(
        repr=False, compare=False
    )
    chaser_index: np.ndarray = field(repr=False, compare=False)
    chaser_position_xy: np.ndarray = field(repr=False, compare=False)
    chaser_valid: np.ndarray = field(repr=False, compare=False)

    @property
    def selector_eligible(self) -> bool:
        return False

    def assert_current(self) -> "PairedFrameBoundChaserSource":
        current = load_paired_frame_bound_chaser_source(
            self.companion.source_h5,
            recording_bundle_root=self.recording_bundle_root,
            expected_recording_id=self.companion.recording_id,
            expected_camera_serial=self.companion.camera_serial,
            expected_acquisition_camera_id=(
                self.companion.acquisition_camera_id
            ),
            expected_shaman_numeric_camera_id=(
                self.companion.shaman_numeric_camera_id
            ),
            expected_source_total_frames=self.companion.source_total_frames,
        )
        if current.verification_digest != self.verification_digest:
            raise StimulusCoordinateContractError(
                "Paired raw-H5/companion source changed after loading."
            )
        return current

    def assert_verified(self) -> None:
        self.assert_current()

    def reload_binding(self) -> dict[str, Any]:
        """Return the exact JSON-safe arguments needed to reverify this pair.

        Scientific publications persist this bounded record rather than the
        in-memory arrays.  Reloaders must still reproduce both sealed source
        digests, so a moved or substituted companion/raw pair fails closed.
        """

        return {
            "schema_id": (
                f"{PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_ID}.reload_binding"
            ),
            "schema_version": 1,
            "companion_h5": str(self.companion.source_h5),
            "recording_bundle_root": str(self.recording_bundle_root),
            "expected_recording_id": self.companion.recording_id,
            "expected_camera_serial": self.companion.camera_serial,
            "expected_acquisition_camera_id": (
                self.companion.acquisition_camera_id
            ),
            "expected_shaman_numeric_camera_id": (
                self.companion.shaman_numeric_camera_id
            ),
            "expected_source_total_frames": self.companion.source_total_frames,
            "companion_source_snapshot_sha256": (
                self.companion.source_snapshot_sha256
            ),
            "source_authority_digest": self.source_authority_digest,
            "manifest_sha256": self.manifest_sha256,
            "verification_digest": self.verification_digest,
            "raw_h5": str(self.raw_h5),
            "raw_h5_sha256": self.raw_h5_sha256,
        }


def _open_h5_file_identity(
    h5: h5py.File,
    *,
    expected_path: Path,
) -> tuple[dict[str, Any], int]:
    actual = Path(str(h5.filename)).expanduser().resolve()
    if actual != expected_path:
        raise StimulusCoordinateContractError(
            "Open raw Citrus H5 filename differs from the sealed path."
        )
    try:
        handle = h5.id.get_vfd_handle()
        if isinstance(handle, tuple):
            handle = handle[0]
        file_descriptor = int(handle)
        open_stat = os.fstat(file_descriptor)
        path_stat = os.stat(expected_path)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise StimulusCoordinateContractError(
            "Unable to bind the open raw Citrus H5 to an exact file identity."
        ) from exc
    if (open_stat.st_dev, open_stat.st_ino) != (
        path_stat.st_dev,
        path_stat.st_ino,
    ):
        raise StimulusCoordinateContractError(
            "The sealed raw Citrus H5 path no longer identifies the open file."
        )
    return (
        {
            "resolved_path": str(expected_path),
            "device": int(open_stat.st_dev),
            "inode": int(open_stat.st_ino),
            "size_bytes": int(open_stat.st_size),
            "mtime_unix_ns": int(open_stat.st_mtime_ns),
        },
        file_descriptor,
    )


def _sha256_open_file(
    file_descriptor: int,
    *,
    size_bytes: int,
) -> str:
    digest = sha256()
    offset = 0
    while offset < size_bytes:
        chunk = os.pread(
            file_descriptor,
            min(8 * 1024 * 1024, size_bytes - offset),
            offset,
        )
        if not chunk:
            raise StimulusCoordinateContractError(
                "Raw Citrus H5 ended while computing its sealed SHA-256."
            )
        digest.update(chunk)
        offset += len(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_raw_field(
    node: h5py.Dataset,
    *,
    name: str,
    dtype: np.dtype[Any],
) -> None:
    fields = node.dtype.fields
    if fields is None or name not in fields or fields[name][0] != dtype:
        raise StimulusCoordinateContractError(
            f"Raw Citrus chaser_states field {name!r} must use {dtype.str}."
        )


def _validate_raw_chaser_rows(
    raw_h5: h5py.File,
    *,
    companion: FrameBoundAcquisitionIdentityEvidence,
) -> tuple[np.ndarray, np.ndarray]:
    path = "/tracking_data/chaser_states"
    if path not in raw_h5 or not isinstance(raw_h5[path], h5py.Dataset):
        raise StimulusCoordinateContractError(
            "Sealed raw Citrus H5 lacks /tracking_data/chaser_states."
        )
    node = raw_h5[path]
    if node.ndim != 1 or int(node.shape[0]) <= 0:
        raise StimulusCoordinateContractError(
            "Raw Citrus chaser_states must be a non-empty row table."
        )
    expected_fields = {
        "stimulus_frame_num": np.dtype("<u8"),
        "timestamp_ns_session": np.dtype("<i8"),
        "chaser_index": np.dtype("u1"),
        "chaser_pos_x": np.dtype("<f4"),
        "chaser_pos_y": np.dtype("<f4"),
        "target_pos_x": np.dtype("<f4"),
        "target_pos_y": np.dtype("<f4"),
        "target_clamped_pos_x": np.dtype("<f4"),
        "target_clamped_pos_y": np.dtype("<f4"),
    }
    for name, dtype in expected_fields.items():
        _require_raw_field(node, name=name, dtype=dtype)
    attrs = {str(key): value for key, value in node.attrs.items()}
    if (
        attrs.get("schema_version") != 4
        or _required_text(
            attrs.get("camera_id"), label="raw chaser_states camera_id"
        )
        != companion.acquisition_camera_id
        or attrs.get("coordinate_frame") != "arena_relative_canvas_px"
        or attrs.get("coordinate_origin") != "top_left_of_active_arena"
        or attrs.get("x_axis_direction") != "right"
        or attrs.get("y_axis_direction") != "down"
        or attrs.get("units") != "px"
    ):
        raise StimulusCoordinateContractError(
            "Raw Citrus chaser_states schema, camera, or coordinate authority differs from v6."
        )
    rows = np.asarray(node[:])
    raw_key = np.column_stack(
        (
            rows["chaser_index"].astype("<i8"),
            rows["stimulus_frame_num"].astype("<i8"),
        )
    )
    if not np.array_equal(raw_key, companion.stimulus_state_key):
        raise StimulusCoordinateContractError(
            "Raw Citrus and v6 companion chaser keys do not match exactly."
        )
    raw_positions = np.column_stack(
        (rows["chaser_pos_x"], rows["chaser_pos_y"])
    ).astype("<f4", copy=False)
    raw_target = np.column_stack(
        (rows["target_pos_x"], rows["target_pos_y"])
    ).astype("<f4", copy=False)
    raw_clamped = np.column_stack(
        (
            rows["target_clamped_pos_x"],
            rows["target_clamped_pos_y"],
        )
    ).astype("<f4", copy=False)
    if (
        not np.array_equal(raw_positions, companion.chaser_position_xy)
        or not np.array_equal(raw_target, companion.target_position_xy)
        or not np.array_equal(
            raw_clamped, companion.target_clamped_position_xy
        )
    ):
        raise StimulusCoordinateContractError(
            "Raw Citrus and v6 companion coordinate rows differ."
        )
    timestamps = rows["timestamp_ns_session"].astype("<i8", copy=True)
    if np.any(timestamps < 0):
        raise StimulusCoordinateContractError(
            "Raw Citrus chaser timestamps must be nonnegative session time."
        )
    return rows, timestamps


def _native_sample_arrays(
    *,
    companion: FrameBoundAcquisitionIdentityEvidence,
    raw_rows: np.ndarray,
    raw_timestamps: np.ndarray,
) -> dict[str, np.ndarray]:
    keys = np.asarray(companion.stimulus_state_key, dtype="<i8")
    stimulus_frames = np.unique(keys[:, 1])
    chaser_indices_i64 = np.unique(keys[:, 0])
    if (
        np.any(chaser_indices_i64 > np.iinfo(np.int16).max)
        or stimulus_frames.size * chaser_indices_i64.size != keys.shape[0]
    ):
        raise StimulusCoordinateContractError(
            "v6 chaser rows do not form a complete sample-by-chaser grid."
        )
    row_lookup = {
        (int(frame), int(chaser)): row
        for row, (chaser, frame) in enumerate(keys.tolist())
    }
    row_matrix = np.empty(
        (stimulus_frames.size, chaser_indices_i64.size), dtype="<i8"
    )
    for sample, frame in enumerate(stimulus_frames.tolist()):
        for chaser_offset, chaser in enumerate(chaser_indices_i64.tolist()):
            try:
                row_matrix[sample, chaser_offset] = row_lookup[
                    (int(frame), int(chaser))
                ]
            except KeyError as exc:
                raise StimulusCoordinateContractError(
                    "v6 chaser rows lack one declared chaser in a native sample."
                ) from exc

    sample_timestamps = np.empty(stimulus_frames.size, dtype="<i8")
    sample_acquisition = np.empty(stimulus_frames.size, dtype="<i8")
    positions = np.empty(
        (stimulus_frames.size, chaser_indices_i64.size, 2), dtype="<f4"
    )
    for sample, rows in enumerate(row_matrix):
        timestamp_values = raw_timestamps[rows]
        acquisition_values = companion.source_acquisition_frame_index[rows]
        if (
            np.unique(timestamp_values).size != 1
            or np.unique(acquisition_values).size != 1
        ):
            raise StimulusCoordinateContractError(
                "One native stimulus sample has mixed timestamps or acquisition identity."
            )
        sample_timestamps[sample] = timestamp_values[0]
        sample_acquisition[sample] = acquisition_values[0]
        positions[sample, :, 0] = raw_rows["chaser_pos_x"][rows]
        positions[sample, :, 1] = raw_rows["chaser_pos_y"][rows]
    if (
        np.any(np.diff(stimulus_frames) <= 0)
        or np.any(np.diff(sample_timestamps) < 0)
        or np.any(np.diff(sample_acquisition) < 0)
    ):
        raise StimulusCoordinateContractError(
            "Paired native stimulus samples are not monotonically ordered."
        )
    arrays = {
        "stimulus_frame_num": stimulus_frames.astype("<i8", copy=False),
        "timestamp_ns_session": sample_timestamps,
        "source_acquisition_frame_index": sample_acquisition,
        "source_sample_row_index": np.arange(
            stimulus_frames.size, dtype="<i8"
        ),
        "source_stimulus_run_row_index": row_matrix,
        "source_stimulus_source_row_index": row_matrix.copy(),
        "chaser_index": chaser_indices_i64.astype("<i2"),
        "chaser_position_xy": positions,
        "chaser_valid": np.ones(row_matrix.shape, dtype=bool),
    }
    return {name: _readonly(values) for name, values in arrays.items()}


def load_paired_frame_bound_chaser_source(
    companion_h5: Path,
    *,
    recording_bundle_root: Path,
    expected_recording_id: str,
    expected_camera_serial: str,
    expected_acquisition_camera_id: str,
    expected_shaman_numeric_camera_id: int,
    expected_source_total_frames: int,
) -> PairedFrameBoundChaserSource:
    """Verify a sealed raw H5 and v6 companion, then expose native samples."""

    companion = load_frame_bound_acquisition_identity(
        companion_h5,
        expected_recording_id=expected_recording_id,
        expected_camera_serial=expected_camera_serial,
        expected_acquisition_camera_id=expected_acquisition_camera_id,
        expected_shaman_numeric_camera_id=expected_shaman_numeric_camera_id,
        expected_source_total_frames=expected_source_total_frames,
    )
    root = Path(recording_bundle_root).expanduser().resolve()
    artifact = companion.raw_citrus_h5_binding["h5_artifact"]
    relative_path = artifact["relative_path"]
    raw_path = (root / relative_path).resolve()
    try:
        raw_path.relative_to(root)
    except ValueError as exc:
        raise StimulusCoordinateContractError(
            "Sealed raw Citrus H5 path escapes the recording bundle root."
        ) from exc
    try:
        with h5py.File(raw_path, "r") as raw_h5:
            raw_identity, file_descriptor = _open_h5_file_identity(
                raw_h5, expected_path=raw_path
            )
            if raw_identity["size_bytes"] != artifact["size_bytes"]:
                raise StimulusCoordinateContractError(
                    "Raw Citrus H5 size differs from the sealed v6 binding."
                )
            raw_digest = _sha256_open_file(
                file_descriptor, size_bytes=raw_identity["size_bytes"]
            )
            if raw_digest != artifact["sha256"]:
                raise StimulusCoordinateContractError(
                    "Raw Citrus H5 SHA-256 differs from the sealed v6 binding."
                )
            raw_rows, raw_timestamps = _validate_raw_chaser_rows(
                raw_h5, companion=companion
            )
            arrays = _native_sample_arrays(
                companion=companion,
                raw_rows=raw_rows,
                raw_timestamps=raw_timestamps,
            )
    except StimulusCoordinateContractError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StimulusCoordinateContractError(
            f"Unable to verify sealed raw Citrus H5: {exc}"
        ) from exc

    authority = {
        "schema_id": PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_ID,
        "schema_version": PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_VERSION,
        "recording_id": companion.recording_id,
        "camera_serial": companion.camera_serial,
        "acquisition_camera_id": companion.acquisition_camera_id,
        "shaman_numeric_camera_id": companion.shaman_numeric_camera_id,
        "recording_identity_token": companion.recording_identity_token,
        "companion_h5": str(companion.source_h5),
        "companion_source_snapshot_sha256": (
            companion.source_snapshot_sha256
        ),
        "raw_h5": str(raw_path),
        "raw_file_identity": raw_identity,
        "raw_h5_sha256": raw_digest,
        "raw_h5_binding_sha256": companion.raw_citrus_h5_binding_sha256,
        "join_key": ["chaser_index", "stimulus_frame_num"],
        "timestamp_field": "timestamp_ns_session",
        "temporal_alignment_class": FRAME_BOUND_TEMPORAL_ALIGNMENT_CLASS,
        "physical_presentation_verified": False,
        "selector_eligible": False,
        "promotion_blocker": FRAME_BOUND_PROMOTION_BLOCKER,
    }
    authority_digest = canonical_mapping_digest(authority)
    manifest = {
        "schema_id": f"{PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_ID}.manifest",
        "schema_version": 1,
        "source_authority_digest": authority_digest,
        "dimensions": {
            "total_frames": companion.source_total_frames,
            "stimulus_samples": int(arrays["stimulus_frame_num"].size),
            "chasers": int(arrays["chaser_index"].size),
        },
        "arrays": {
            name: numpy_content_digest(values)
            for name, values in arrays.items()
        },
    }
    manifest_digest = canonical_mapping_digest(manifest)
    verification = {
        "schema_id": f"{PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_ID}.verification",
        "schema_version": 1,
        "source_authority_digest": authority_digest,
        "manifest_sha256": manifest_digest,
        "companion_source_file_identity": dict(
            companion.source_file_identity
        ),
        "raw_file_identity": raw_identity,
    }
    return PairedFrameBoundChaserSource(
        companion=companion,
        recording_bundle_root=root,
        raw_h5=raw_path,
        raw_file_identity=_freeze(raw_identity),
        raw_h5_sha256=raw_digest,
        recording_id=companion.recording_id,
        dimensions=FrameBoundChaserSourceDimensions(
            total_frames=companion.source_total_frames,
            stimulus_sample_count=int(arrays["stimulus_frame_num"].size),
            chaser_count=int(arrays["chaser_index"].size),
        ),
        source_authority=_freeze(authority),
        source_authority_id=PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_ID,
        source_authority_digest=authority_digest,
        manifest_sha256=manifest_digest,
        verification_digest=canonical_mapping_digest(verification),
        run_path=f"{raw_path}#/tracking_data/chaser_states",
        **arrays,
    )


def load_paired_frame_bound_chaser_source_from_binding(
    binding: Mapping[str, Any],
) -> PairedFrameBoundChaserSource:
    """Reverify one paired source from an exact persisted reload binding."""

    if not isinstance(binding, Mapping):
        raise StimulusCoordinateContractError(
            "Paired frame-bound reload binding must be one mapping."
        )
    expected_keys = {
        "schema_id",
        "schema_version",
        "companion_h5",
        "recording_bundle_root",
        "expected_recording_id",
        "expected_camera_serial",
        "expected_acquisition_camera_id",
        "expected_shaman_numeric_camera_id",
        "expected_source_total_frames",
        "companion_source_snapshot_sha256",
        "source_authority_digest",
        "manifest_sha256",
        "verification_digest",
        "raw_h5",
        "raw_h5_sha256",
    }
    if set(binding) != expected_keys:
        raise StimulusCoordinateContractError(
            "Paired frame-bound reload binding has a non-canonical key set."
        )
    if (
        binding.get("schema_id")
        != f"{PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_ID}.reload_binding"
        or binding.get("schema_version") != 1
    ):
        raise StimulusCoordinateContractError(
            "Paired frame-bound reload binding schema is unsupported."
        )
    source = load_paired_frame_bound_chaser_source(
        Path(_required_text(binding.get("companion_h5"), label="companion_h5")),
        recording_bundle_root=Path(
            _required_text(
                binding.get("recording_bundle_root"),
                label="recording_bundle_root",
            )
        ),
        expected_recording_id=_required_text(
            binding.get("expected_recording_id"),
            label="expected_recording_id",
        ),
        expected_camera_serial=_required_text(
            binding.get("expected_camera_serial"),
            label="expected_camera_serial",
        ),
        expected_acquisition_camera_id=_required_text(
            binding.get("expected_acquisition_camera_id"),
            label="expected_acquisition_camera_id",
        ),
        expected_shaman_numeric_camera_id=binding.get(
            "expected_shaman_numeric_camera_id"
        ),
        expected_source_total_frames=binding.get(
            "expected_source_total_frames"
        ),
    )
    observed = source.reload_binding()
    if observed != dict(binding):
        raise StimulusCoordinateContractError(
            "Paired frame-bound source differs from its persisted reload binding."
        )
    return source


__all__ = [
    "FRAME_BOUND_ACQUISITION_IDENTITY_SCHEMA_ID",
    "FRAME_BOUND_ACQUISITION_IDENTITY_SCHEMA_VERSION",
    "FRAME_BOUND_ACQUISITION_MAPPING_POLICY",
    "FRAME_BOUND_PROMOTION_BLOCKER",
    "FRAME_BOUND_RECORDING_MEMBERSHIP_STATUS",
    "FRAME_BOUND_SCIENTIFIC_USE_CLASS",
    "FRAME_BOUND_TEMPORAL_ALIGNMENT_CLASS",
    "PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_ID",
    "PAIRED_FRAME_BOUND_CHASER_SOURCE_SCHEMA_VERSION",
    "FrameBoundAcquisitionIdentityEvidence",
    "FrameBoundChaserSourceDimensions",
    "PairedFrameBoundChaserSource",
    "load_frame_bound_acquisition_identity",
    "load_paired_frame_bound_chaser_source",
    "load_paired_frame_bound_chaser_source_from_binding",
    "validate_frame_bound_acquisition_identity",
]
