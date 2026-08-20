"""Shared scientific contract for chaser input-provenance proxy results.

The analysis selector and the Zarr schema both depend on these immutable
semantics and result fields.  Keeping the contract in the shared layer avoids
making storage schemas import higher-level workflow implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


PROXY_POLICY_ID = "latest_logged_cpu_state_per_input_acquisition_proxy_v1"
TEMPORAL_ALIGNMENT_REQUIREMENT = "input_provenance_proxy_allowed"
TEMPORAL_ALIGNMENT_CLASS = "controller_input_provenance_proxy"
PHYSICAL_PRESENTATION_VERIFIED = False
PRESENTATION_TIMESTAMP_AVAILABLE = False
CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE = False
CAMERA_EXPOSURE_REFERENCE = "unknown"
SCIENTIFIC_USE_CLASS = "exploratory_proxy"
BEHAVIORAL_DENOMINATOR = "unique_input_acquisition_frames"
PROJECTION_RECORD_SCHEMA_ID = "palette.chaser_input_provenance_proxy"
PROJECTION_RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ChaserInputProvenanceProxyResult:
    """Read-only acquisition-frame proxy plus complete native candidate lineage."""

    source_handle: object = field(repr=False, compare=False)
    recording_id: str
    source_authority_id: str
    source_authority_digest: str
    source_manifest_sha256: str
    source_verification_digest: str
    source_run_path: str
    acquisition_frame_index: np.ndarray
    candidate_offsets: np.ndarray
    candidate_sample_count: np.ndarray
    candidate_native_sample_row_index: np.ndarray
    candidate_stimulus_frame_num: np.ndarray
    candidate_timestamp_ns_session: np.ndarray
    candidate_source_acquisition_frame_index: np.ndarray
    candidate_complete: np.ndarray
    candidate_reason_code: np.ndarray
    candidate_source_stimulus_run_row_index: np.ndarray
    candidate_source_stimulus_source_row_index: np.ndarray
    selected: np.ndarray
    selection_reason_code: np.ndarray
    selected_native_sample_row_index: np.ndarray
    selected_stimulus_frame_num: np.ndarray
    selected_timestamp_ns_session: np.ndarray
    selected_source_stimulus_run_row_index: np.ndarray
    selected_source_stimulus_source_row_index: np.ndarray
    selected_chaser_index: np.ndarray
    selected_chaser_position_xy: np.ndarray
    selected_chaser_valid: np.ndarray
    acquisition_projection_record: Mapping[str, Any]
    acquisition_projection_record_sha256: str
    provenance: Mapping[str, Any]

    @property
    def unique_acquisition_frame_count(self) -> int:
        return int(self.acquisition_frame_index.size)

    @property
    def behavioral_denominator(self) -> str:
        return BEHAVIORAL_DENOMINATOR

    @property
    def policy_id(self) -> str:
        return PROXY_POLICY_ID

    @property
    def acquisition_projection_digest(self) -> str:
        """Digest of the compact readable record for publication context."""

        return self.acquisition_projection_record_sha256

    @property
    def compact_record(self) -> Mapping[str, Any]:
        """Alias used when binding this result to a publication context."""

        return self.acquisition_projection_record

    def candidate_rows_for_frame(self, frame_offset: int) -> np.ndarray:
        """Return candidate native sample row indices for one output frame."""

        if (
            type(frame_offset) is not int
            or not 0 <= frame_offset < self.unique_acquisition_frame_count
        ):
            raise IndexError("frame_offset is outside the proxy result.")
        start = int(self.candidate_offsets[frame_offset])
        end = int(self.candidate_offsets[frame_offset + 1])
        return self.candidate_native_sample_row_index[start:end]


__all__ = [
    "BEHAVIORAL_DENOMINATOR",
    "CAMERA_EXPOSURE_REFERENCE",
    "CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE",
    "ChaserInputProvenanceProxyResult",
    "PHYSICAL_PRESENTATION_VERIFIED",
    "PRESENTATION_TIMESTAMP_AVAILABLE",
    "PROJECTION_RECORD_SCHEMA_ID",
    "PROJECTION_RECORD_SCHEMA_VERSION",
    "PROXY_POLICY_ID",
    "SCIENTIFIC_USE_CLASS",
    "TEMPORAL_ALIGNMENT_CLASS",
    "TEMPORAL_ALIGNMENT_REQUIREMENT",
]
