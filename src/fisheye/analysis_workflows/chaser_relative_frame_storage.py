"""Prepare a chaser-relative result for compact, typed publication.

This module is deliberately pure: it performs no filesystem or Zarr writes.
It is the fail-closed boundary between the in-memory scientific computation
and a later immutable materializer.  Row evidence stays in typed arrays;
metadata contains only bounded, readable authority records, controlled code
registries, policies, and compact array declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    BEHAVIORAL_DENOMINATOR as INPUT_PROVENANCE_BEHAVIORAL_DENOMINATOR,
    CAMERA_EXPOSURE_REFERENCE as INPUT_PROVENANCE_CAMERA_EXPOSURE_REFERENCE,
    PROJECTION_RECORD_SCHEMA_ID as INPUT_PROVENANCE_PROJECTION_SCHEMA_ID,
    PROJECTION_RECORD_SCHEMA_VERSION as INPUT_PROVENANCE_PROJECTION_SCHEMA_VERSION,
    PROXY_POLICY_ID as INPUT_PROVENANCE_PROXY_POLICY_ID,
    SCIENTIFIC_USE_CLASS as INPUT_PROVENANCE_SCIENTIFIC_USE_CLASS,
    TEMPORAL_ALIGNMENT_CLASS as INPUT_PROVENANCE_TEMPORAL_ALIGNMENT_CLASS,
    TEMPORAL_ALIGNMENT_REQUIREMENT as INPUT_PROVENANCE_TEMPORAL_ALIGNMENT_REQUIREMENT,
)
from fisheye.analysis_workflows.chaser_relative_frame import (
    ChaserRelativeFrameResult,
    ProviderSourceAuthority,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.chaser_relative_frame_schema import (
    CHASER_RELATIVE_FRAME_LAYOUT,
    CHASER_RELATIVE_FRAME_REASON_CODES,
    CHASER_RELATIVE_FRAME_SCHEMA_ID,
    CHASER_RELATIVE_FRAME_SCHEMA_V1,
    CHASER_RELATIVE_FRAME_SCHEMA_VERSION,
    ChaserRelativeFrameDimensions,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


PREPARED_CHASER_RELATIVE_FRAME_SCHEMA_ID = (
    "palette.analysis.chaser_relative_frame.prepared_candidate"
)
PREPARED_CHASER_RELATIVE_FRAME_SCHEMA_VERSION = 1
FLATTEN_POLICY_ID = "acquisition_frame_major_chaser_axis_minor_v1"
COMPUTATION_ID = "compute_chaser_relative_frame_v2_activity_orthogonal"
MAX_CONTEXT_RECORD_BYTES = 65_536

_INPUT_PROVENANCE_PROJECTION_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "policy_id",
        "temporal_alignment_requirement",
        "temporal_alignment_class",
        "physical_presentation_verified",
        "presentation_timestamp_available",
        "camera_presentation_clock_transform_available",
        "camera_exposure_reference",
        "scientific_use_class",
        "behavioral_denominator",
        "native_sample_axis",
        "native_sample_rows_preserved",
        "source_acquisition_frame_field",
        "selection_order",
        "complete_sample_rule",
        "missing_frame_rule",
        "native_sample_count",
        "unique_acquisition_frame_count",
        "selected_acquisition_frame_count",
        "chaser_count",
        "candidate_sample_row_index_is_zero_based",
        "source_authority_id",
        "source_authority_digest",
        "source_manifest_sha256",
        "source_verification_digest",
        "source_run_path",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_PROVENANCE_PUBLICATION_BINDING_SCHEMA_ID = (
    "palette.chaser_input_provenance_proxy_publication_binding"
)
_INPUT_PROVENANCE_PUBLICATION_BINDING_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "run_path",
        "manifest_sha256",
        "acquisition_projection_record_sha256",
        "policy_id",
        "temporal_alignment_class",
        "source_run_path",
        "source_manifest_sha256",
        "source_verification_digest",
        "n_frames",
        "n_candidates",
        "n_chasers",
        "selector_eligible",
        "selection",
    }
)

_REASON_TO_CODE = {
    reason: np.uint16(code) for code, reason in CHASER_RELATIVE_FRAME_REASON_CODES.items()
}


class ChaserRelativeFrameStorageError(ValueError):
    """Raised when a result lacks exact, publishable storage evidence."""


def _fail(message: str) -> None:
    raise ChaserRelativeFrameStorageError(message)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _strict_json_record(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{field} must be one non-empty JSON object.")
    if any(type(key) is not str for key in value):
        _fail(f"{field} keys must be strings.")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"{field} must be strict JSON: {exc}")
    if len(encoded) > MAX_CONTEXT_RECORD_BYTES:
        _fail(
            f"{field} exceeds the bounded metadata limit of "
            f"{MAX_CONTEXT_RECORD_BYTES} bytes."
        )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        _fail(f"{field} did not canonicalize to an object.")
    if type(decoded.get("schema_id")) is not str:
        _fail(f"{field}.schema_id must be explicit.")
    if type(decoded.get("schema_version")) is not int:
        _fail(f"{field}.schema_version must be an exact integer.")
    return decoded


def _validate_input_provenance_projection_record(
    record: Mapping[str, Any],
) -> None:
    if set(record) != _INPUT_PROVENANCE_PROJECTION_FIELDS:
        _fail("Input-provenance projection record has an inexact field set.")
    expected = {
        "schema_id": INPUT_PROVENANCE_PROJECTION_SCHEMA_ID,
        "schema_version": INPUT_PROVENANCE_PROJECTION_SCHEMA_VERSION,
        "policy_id": INPUT_PROVENANCE_PROXY_POLICY_ID,
        "temporal_alignment_requirement": (
            INPUT_PROVENANCE_TEMPORAL_ALIGNMENT_REQUIREMENT
        ),
        "temporal_alignment_class": INPUT_PROVENANCE_TEMPORAL_ALIGNMENT_CLASS,
        "physical_presentation_verified": False,
        "presentation_timestamp_available": False,
        "camera_presentation_clock_transform_available": False,
        "camera_exposure_reference": INPUT_PROVENANCE_CAMERA_EXPOSURE_REFERENCE,
        "scientific_use_class": INPUT_PROVENANCE_SCIENTIFIC_USE_CLASS,
        "behavioral_denominator": INPUT_PROVENANCE_BEHAVIORAL_DENOMINATOR,
        "native_sample_axis": "stimulus_samples",
        "native_sample_rows_preserved": True,
        "source_acquisition_frame_field": "source_acquisition_frame_index",
        "selection_order": [
            "timestamp_ns_session",
            "stimulus_frame_num",
            "source_stimulus_run_row_index",
            "source_stimulus_source_row_index",
            "source_sample_row_index",
        ],
        "complete_sample_rule": (
            "all_declared_chasers_valid_and_finite_in_one_native_sample"
        ),
        "missing_frame_rule": "no_carry_forward",
        "candidate_sample_row_index_is_zero_based": True,
    }
    for name, expected_value in expected.items():
        if record.get(name) != expected_value:
            _fail(f"Input-provenance projection record has invalid {name}.")
    for name in (
        "source_authority_digest",
        "source_manifest_sha256",
        "source_verification_digest",
    ):
        value = record.get(name)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            _fail(f"Input-provenance projection record has invalid {name}.")
    for name in ("source_authority_id", "source_run_path"):
        value = record.get(name)
        if type(value) is not str or not value or value != value.strip():
            _fail(f"Input-provenance projection record has invalid {name}.")
    counts = tuple(
        record.get(name)
        for name in (
            "native_sample_count",
            "unique_acquisition_frame_count",
            "selected_acquisition_frame_count",
        )
    )
    if (
        any(type(value) is not int or value < 0 for value in counts)
        or counts[0] <= 0
        or counts[1] <= 0
        or counts[2] > counts[1]
        or counts[1] > counts[0]
    ):
        _fail("Input-provenance projection counts are contradictory.")
    if type(record.get("chaser_count")) is not int or record["chaser_count"] <= 0:
        _fail("Input-provenance projection chaser_count is invalid.")


def _validate_input_provenance_publication_binding(
    binding: Mapping[str, Any],
    *,
    projection: Mapping[str, Any],
) -> None:
    if set(binding) != _INPUT_PROVENANCE_PUBLICATION_BINDING_FIELDS:
        _fail("Input-provenance publication binding has an inexact field set.")
    expected = {
        "schema_id": _INPUT_PROVENANCE_PUBLICATION_BINDING_SCHEMA_ID,
        "schema_version": 1,
        "recording_id": projection["recording_id"],
        "acquisition_projection_record_sha256": canonical_json_sha256(
            dict(projection)
        ),
        "policy_id": projection["policy_id"],
        "temporal_alignment_class": projection["temporal_alignment_class"],
        "source_run_path": projection["source_run_path"],
        "source_manifest_sha256": projection["source_manifest_sha256"],
        "source_verification_digest": projection["source_verification_digest"],
        "n_frames": projection["unique_acquisition_frame_count"],
        "n_candidates": projection["native_sample_count"],
        "n_chasers": projection["chaser_count"],
        "selector_eligible": False,
        "selection": "none",
    }
    for name, expected_value in expected.items():
        if binding.get(name) != expected_value:
            _fail(f"Input-provenance publication binding has invalid {name}.")
    for name in ("manifest_sha256", "acquisition_projection_record_sha256"):
        value = binding.get(name)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            _fail(f"Input-provenance publication binding has invalid {name}.")
    for name in ("run_path", "source_run_path"):
        value = binding.get(name)
        if type(value) is not str or not value or value != value.strip():
            _fail(f"Input-provenance publication binding has invalid {name}.")


def validate_chaser_input_provenance_projection_binding(
    *,
    projection: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> None:
    """Validate one proxy projection and its exact immutable publication."""

    _validate_input_provenance_projection_record(projection)
    _validate_input_provenance_publication_binding(
        publication,
        projection=projection,
    )


@dataclass(frozen=True, slots=True)
class ChaserRelativeFramePublicationContext:
    """Readable external authorities required before preparation.

    The records remain schema-owned by their producer.  This boundary only
    enforces their exact recording/subject identity and bounds their metadata
    size; it never substitutes a digest for the readable record.
    """

    fish_identity: str
    subject_identity_record: Mapping[str, Any]
    temporal_selection_record: Mapping[str, Any]
    chaser_occurrence_record: Mapping[str, Any]
    acquisition_projection_record: Mapping[str, Any]
    analysis_profile_record: Mapping[str, Any]
    acquisition_projection_publication_record: Mapping[str, Any] | None = None
    controller_state_record: Mapping[str, Any] | None = None
    body_frame_projection_record: Mapping[str, Any] | None = None
    core_authority_record: Mapping[str, Any] | None = None
    arena_geometry_record: Mapping[str, Any] | None = None
    arena_to_source_camera_transform_record: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fish_identity", _text(self.fish_identity, field="fish_identity")
        )
        for name in (
            "subject_identity_record",
            "temporal_selection_record",
            "chaser_occurrence_record",
            "acquisition_projection_record",
            "analysis_profile_record",
        ):
            record = _strict_json_record(getattr(self, name), field=name)
            object.__setattr__(self, name, MappingProxyType(record))
        if (self.arena_geometry_record is None) != (
            self.arena_to_source_camera_transform_record is None
        ):
            _fail(
                "arena geometry and arena-to-source-camera transform records must be "
                "supplied together."
            )
        for name in (
            "arena_geometry_record",
            "arena_to_source_camera_transform_record",
        ):
            value = getattr(self, name)
            if value is not None:
                record = _strict_json_record(value, field=name)
                object.__setattr__(self, name, MappingProxyType(record))

        publication = self.acquisition_projection_publication_record
        if publication is not None:
            publication_record = _strict_json_record(
                publication,
                field="acquisition_projection_publication_record",
            )
            object.__setattr__(
                self,
                "acquisition_projection_publication_record",
                MappingProxyType(publication_record),
            )
        controller_state = self.controller_state_record
        if controller_state is not None:
            controller_record = _strict_json_record(
                controller_state,
                field="controller_state_record",
            )
            object.__setattr__(
                self,
                "controller_state_record",
                MappingProxyType(controller_record),
            )
        body_projection = self.body_frame_projection_record
        if body_projection is not None:
            body_projection_record = _strict_json_record(
                body_projection,
                field="body_frame_projection_record",
            )
            object.__setattr__(
                self,
                "body_frame_projection_record",
                MappingProxyType(body_projection_record),
            )
        core_authority = self.core_authority_record
        if core_authority is not None:
            core_authority_record = _strict_json_record(
                core_authority,
                field="core_authority_record",
            )
            object.__setattr__(
                self,
                "core_authority_record",
                MappingProxyType(core_authority_record),
            )

        subject = self.subject_identity_record
        if subject.get("subject_id") != self.fish_identity:
            _fail("subject_identity_record.subject_id does not match fish_identity.")
        if type(self.temporal_selection_record.get("selection_id")) is not str:
            _fail("temporal_selection_record.selection_id must be explicit.")
        if type(self.chaser_occurrence_record.get("occurrence_policy_id")) is not str:
            _fail("chaser_occurrence_record.occurrence_policy_id must be explicit.")
        if type(self.acquisition_projection_record.get("policy_id")) is not str:
            _fail("acquisition_projection_record.policy_id must be explicit.")
        if (
            self.acquisition_projection_record.get("policy_id")
            == INPUT_PROVENANCE_PROXY_POLICY_ID
        ):
            _validate_input_provenance_projection_record(
                self.acquisition_projection_record
            )
            if self.acquisition_projection_publication_record is None:
                _fail(
                    "Input-provenance relative frames require an exact published "
                    "proxy binding."
                )
            _validate_input_provenance_publication_binding(
                self.acquisition_projection_publication_record,
                projection=self.acquisition_projection_record,
            )
        elif self.acquisition_projection_publication_record is not None:
            _fail(
                "An acquisition projection publication binding is only valid for "
                "the input-provenance proxy policy."
            )
        if type(self.analysis_profile_record.get("profile_id")) is not str:
            _fail("analysis_profile_record.profile_id must be explicit.")

    def require_recording(self, recording_id: str) -> None:
        for name in (
            "subject_identity_record",
            "temporal_selection_record",
            "chaser_occurrence_record",
            "acquisition_projection_record",
            "acquisition_projection_publication_record",
            "controller_state_record",
            "body_frame_projection_record",
            "core_authority_record",
            "arena_geometry_record",
            "arena_to_source_camera_transform_record",
        ):
            record = getattr(self, name)
            if record is not None and record.get("recording_id") != recording_id:
                _fail(f"{name}.recording_id does not match the computed recording.")

    @staticmethod
    def _envelope(record: Mapping[str, Any]) -> dict[str, Any]:
        plain = dict(record)
        return {"record": plain, "sha256": canonical_json_sha256(plain)}

    def as_manifest(self) -> dict[str, Any]:
        manifest = {
            "fish_identity": self.fish_identity,
            "subject_identity": self._envelope(self.subject_identity_record),
            "temporal_selection": self._envelope(self.temporal_selection_record),
            "chaser_occurrence": self._envelope(self.chaser_occurrence_record),
            "acquisition_projection": self._envelope(
                self.acquisition_projection_record
            ),
            "acquisition_projection_publication": (
                None
                if self.acquisition_projection_publication_record is None
                else self._envelope(
                    self.acquisition_projection_publication_record
                )
            ),
            "analysis_profile": self._envelope(self.analysis_profile_record),
            "arena_geometry": (
                None
                if self.arena_geometry_record is None
                else self._envelope(self.arena_geometry_record)
            ),
            "arena_to_source_camera_transform": (
                None
                if self.arena_to_source_camera_transform_record is None
                else self._envelope(self.arena_to_source_camera_transform_record)
            ),
        }
        if self.controller_state_record is not None:
            manifest["controller_state"] = self._envelope(
                self.controller_state_record
            )
        if self.body_frame_projection_record is not None:
            manifest["body_frame_projection"] = self._envelope(
                self.body_frame_projection_record
            )
        if self.core_authority_record is not None:
            manifest["core_authority"] = self._envelope(
                self.core_authority_record
            )
        return manifest


@dataclass(frozen=True, slots=True)
class PreparedChaserRelativeFrame:
    """Validated base/body arrays and their compact immutable manifest."""

    dimensions: ChaserRelativeFrameDimensions
    base_arrays: Mapping[str, np.ndarray]
    body_arrays: Mapping[str, np.ndarray] | None
    manifest: Mapping[str, Any]

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _require_binding_envelope(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"record", "sha256"}:
        _fail(f"{field} must be one exact record-plus-sha256 envelope.")
    record = _strict_json_record(value["record"], field=f"{field}.record")
    if value["sha256"] != canonical_json_sha256(record):
        _fail(f"{field} digest does not match its readable record.")
    return record


def validate_prepared_chaser_relative_frame(
    prepared: PreparedChaserRelativeFrame,
) -> dict[str, Any]:
    """Recheck schema, logical array hashes, and every compact binding."""

    if not isinstance(prepared, PreparedChaserRelativeFrame):
        _fail("prepared must be one PreparedChaserRelativeFrame.")
    CHASER_RELATIVE_FRAME_SCHEMA_V1.require(
        prepared.base_arrays,
        dimensions=prepared.dimensions,
        body_arrays=prepared.body_arrays,
    )
    manifest = dict(prepared.manifest)
    payload_digest = manifest.pop("payload_digest", None)
    if type(payload_digest) is not str or payload_digest != canonical_json_sha256(
        manifest
    ):
        _fail("Prepared manifest payload_digest is missing or stale.")
    if (
        manifest.get("schema_id") != PREPARED_CHASER_RELATIVE_FRAME_SCHEMA_ID
        or manifest.get("schema_version")
        != PREPARED_CHASER_RELATIVE_FRAME_SCHEMA_VERSION
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        _fail("Prepared manifest identity or selector-ineligible state is invalid.")
    dimensions = manifest.get("dimensions")
    if not isinstance(dimensions, Mapping) or dimensions.get(
        "n_rows"
    ) != prepared.dimensions.n_rows:
        _fail("Prepared manifest dimensions do not match the typed arrays.")
    n_frames = dimensions.get("n_frames")
    n_chasers = dimensions.get("n_chasers")
    if (
        type(n_frames) is not int
        or type(n_chasers) is not int
        or n_chasers <= 0
        or n_frames < 0
        or n_frames * n_chasers != prepared.dimensions.n_rows
    ):
        _fail("Prepared frame/chaser dimensions are inconsistent.")
    schema = manifest.get("schema_binding")
    if not isinstance(schema, Mapping) or schema != {
        "schema_id": CHASER_RELATIVE_FRAME_SCHEMA_ID,
        "schema_version": CHASER_RELATIVE_FRAME_SCHEMA_VERSION,
        "layout": CHASER_RELATIVE_FRAME_LAYOUT,
        "body_extension_present": prepared.body_arrays is not None,
    }:
        _fail("Prepared schema binding is missing, stale, or contradictory.")
    expected_declarations = _array_declarations(
        prepared.base_arrays, prepared.body_arrays
    )
    if manifest.get("array_declarations") != expected_declarations:
        _fail("Prepared array declarations do not match logical array content.")
    context = manifest.get("context")
    if not isinstance(context, Mapping):
        _fail("Prepared context binding is missing.")
    records: dict[str, dict[str, Any]] = {}
    for name in (
        "subject_identity",
        "temporal_selection",
        "chaser_occurrence",
        "acquisition_projection",
        "analysis_profile",
    ):
        records[name] = _require_binding_envelope(
            context.get(name), field=f"context.{name}"
        )
    if context.get("controller_state") is not None:
        records["controller_state"] = _require_binding_envelope(
            context.get("controller_state"), field="context.controller_state"
        )
    if context.get("body_frame_projection") is not None:
        if prepared.body_arrays is None:
            _fail(
                "Position-only prepared output has an unexpected body-frame "
                "projection binding."
            )
        records["body_frame_projection"] = _require_binding_envelope(
            context.get("body_frame_projection"),
            field="context.body_frame_projection",
        )
    if context.get("core_authority") is not None:
        records["core_authority"] = _require_binding_envelope(
            context.get("core_authority"), field="context.core_authority"
        )
    publication = context.get("acquisition_projection_publication")
    if records["acquisition_projection"].get("policy_id") == (
        INPUT_PROVENANCE_PROXY_POLICY_ID
    ):
        publication_record = _require_binding_envelope(
            publication,
            field="context.acquisition_projection_publication",
        )
        validate_chaser_input_provenance_projection_binding(
            projection=records["acquisition_projection"],
            publication=publication_record,
        )
    elif publication is not None:
        _fail(
            "Prepared context has an unexpected acquisition projection "
            "publication binding."
        )
    geometry = context.get("arena_geometry")
    transform = context.get("arena_to_source_camera_transform")
    if (geometry is None) != (transform is None):
        _fail("Prepared context has a partial arena geometry/transform binding.")
    if geometry is not None:
        _require_binding_envelope(geometry, field="context.arena_geometry")
        _require_binding_envelope(
            transform, field="context.arena_to_source_camera_transform"
        )
    return {
        "schema_id": "palette.analysis.chaser_relative_frame.validation_receipt",
        "schema_version": 1,
        "payload_digest": payload_digest,
        "n_frames": n_frames,
        "n_chasers": n_chasers,
        "n_rows": prepared.dimensions.n_rows,
        "body_extension_present": prepared.body_arrays is not None,
        "selector_eligible": False,
    }


def _readonly(values: object, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _frame_rows(values: np.ndarray, chaser_count: int) -> np.ndarray:
    values = np.asarray(values)
    expanded = np.repeat(values[:, None, ...], chaser_count, axis=1)
    return expanded.reshape((-1,) + values.shape[1:])


def _pair_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values.reshape((-1,) + values.shape[2:])


def _encode_reasons(values: np.ndarray, *, field: str) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    encoded = np.empty(flat.shape, dtype=np.uint16)
    for index, raw in enumerate(flat):
        reason = str(raw)
        if reason not in _REASON_TO_CODE:
            _fail(f"{field} contains unregistered reason {reason!r}.")
        encoded[index] = _REASON_TO_CODE[reason]
    return encoded.reshape(np.asarray(values).shape)


def _valid_float_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return np.isfinite(values)
    return np.isfinite(values).all(axis=tuple(range(1, values.ndim)))


def _float32_with_nan(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[~np.asarray(valid, dtype=bool)] = np.nan
    return result


def _reason_codes_for_position(
    declared_valid: np.ndarray,
    finite: np.ndarray,
    *,
    invalid_reason: str,
) -> np.ndarray:
    reasons = np.full(
        declared_valid.shape,
        invalid_reason,
        dtype=np.dtypes.StringDType(),
    )
    reasons[declared_valid & ~finite] = "nonfinite_coordinate"
    reasons[declared_valid & finite] = "valid"
    return _encode_reasons(reasons, field=f"{invalid_reason}_position_reason")


def _authority_record(authority: ProviderSourceAuthority) -> dict[str, Any]:
    return {
        "recording_id": authority.recording_id,
        "source_authority_id": authority.source_authority_id,
        "source_digest": authority.source_digest,
        "provider_id": authority.provider_id,
        "provider_digest": authority.provider_digest,
        "coordinate_authority_id": authority.coordinate_authority_id,
        "scale_authority_id": authority.scale_authority_id,
        "timing_authority_id": authority.timing_authority_id,
        "row_axis_authority_id": authority.row_axis_authority_id,
        "row_axis_authority_digest": authority.row_axis_authority_digest,
    }


def _array_declarations(
    base_arrays: Mapping[str, np.ndarray],
    body_arrays: Mapping[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for prefix, arrays in (("base", base_arrays), ("body", body_arrays or {})):
        for path in sorted(arrays):
            array = np.asarray(arrays[path])
            declarations.append(
                {
                    "path": f"{prefix}/{path}",
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "content_sha256": array_values_sha256(array),
                }
            )
    return declarations


def prepare_chaser_relative_frame(
    result: ChaserRelativeFrameResult,
    *,
    context: ChaserRelativeFramePublicationContext,
) -> PreparedChaserRelativeFrame:
    """Flatten, encode, validate, and bind one computed relative frame.

    The row order is exactly acquisition-frame major, then stable chaser-axis
    minor.  A recording with no chaser axis is an inapplicable capability and
    cannot masquerade as a zero-row successful relative-frame publication.
    """

    if not isinstance(result, ChaserRelativeFrameResult):
        _fail("result must be one ChaserRelativeFrameResult.")
    recording_id = result.frame_keys.recording_id
    context.require_recording(recording_id)
    n_frames = result.frame_keys.row_count
    n_chasers = len(result.chaser_identities)
    if n_chasers == 0:
        _fail("A chaser-relative publication requires a non-empty chaser axis.")
    projection = context.acquisition_projection_record
    if projection.get("policy_id") == INPUT_PROVENANCE_PROXY_POLICY_ID:
        if projection["unique_acquisition_frame_count"] != n_frames:
            _fail(
                "Input-provenance projection frame count does not match the "
                "relative-frame acquisition axis."
            )
        if projection["chaser_count"] != n_chasers:
            _fail(
                "Input-provenance projection chaser count does not match the "
                "relative-frame chaser axis."
            )
        complete_selected_rows = np.all(result.chaser_valid, axis=1)
        if int(np.count_nonzero(complete_selected_rows)) != projection[
            "selected_acquisition_frame_count"
        ]:
            _fail(
                "Input-provenance selected count does not match complete "
                "relative-frame chaser rows."
            )
        publication = context.acquisition_projection_publication_record
        assert publication is not None  # validated by the context constructor
        expected_chaser_authority = {
            "source_authority_id": publication["run_path"],
            "source_digest": publication["manifest_sha256"],
            "provider_id": projection["policy_id"],
            "provider_digest": publication[
                "acquisition_projection_record_sha256"
            ],
        }
        for name, expected_value in expected_chaser_authority.items():
            if getattr(result.chaser_authority, name) != expected_value:
                _fail(
                    "Input-provenance proxy does not match the relative-frame "
                    f"chaser authority field {name}."
                )
    n_rows = n_frames * n_chasers
    dimensions = ChaserRelativeFrameDimensions(n_rows=n_rows)

    def frame(values: object) -> np.ndarray:
        return _frame_rows(np.asarray(values), n_chasers)

    def pair(values: object) -> np.ndarray:
        return _pair_rows(np.asarray(values))

    timestamp_present = result.frame_keys.timestamp_ns is not None
    if timestamp_present:
        timestamp_ns = frame(result.frame_keys.timestamp_ns)
        timestamp_valid = np.ones(n_rows, dtype=bool)
        timestamp_reason = np.zeros(n_rows, dtype=np.uint16)
    else:
        timestamp_ns = np.full(n_rows, -1, dtype=np.int64)
        timestamp_valid = np.zeros(n_rows, dtype=bool)
        timestamp_reason = np.full(
            n_rows, _REASON_TO_CODE["timestamp_unavailable"], dtype=np.uint16
        )

    fish_xy = frame(result.fish_xy)
    fish_declared = frame(result.fish_valid).astype(bool, copy=False)
    fish_finite = _valid_float_rows(fish_xy)
    fish_valid = fish_declared & fish_finite
    fish_reason = _reason_codes_for_position(
        fish_declared, fish_finite, invalid_reason="fish_invalid"
    )
    fish_rows = frame(result.fish_source_row_index)
    fish_rows_valid = fish_rows >= 0
    if np.any(fish_valid & ~fish_rows_valid):
        _fail("A valid fish position lacks a nonnegative source-row identity.")

    chaser_xy = pair(result.chaser_xy)
    chaser_declared = pair(result.chaser_valid).astype(bool, copy=False)
    chaser_finite = _valid_float_rows(chaser_xy)
    chaser_valid = chaser_declared & chaser_finite
    chaser_reason = _reason_codes_for_position(
        chaser_declared, chaser_finite, invalid_reason="chaser_invalid"
    )
    chaser_rows = pair(result.chaser_source_row_index)
    chaser_rows_valid = chaser_rows >= 0
    if np.any(chaser_valid & ~chaser_rows_valid):
        _fail("A valid chaser position lacks a nonnegative source-row identity.")

    if n_chasers > np.iinfo(np.uint16).max:
        _fail("Chaser identity cardinality exceeds uint16 storage.")
    identity_codes = np.arange(1, n_chasers + 1, dtype=np.uint16)
    chaser_identity_code = np.tile(identity_codes, n_frames)
    role_rows = pair(result.chaser_behavior_roles)
    role_values = tuple(sorted({str(value) for value in role_rows.tolist()}))
    if len(role_values) > np.iinfo(np.uint8).max:
        _fail("Chaser behavior-role cardinality exceeds uint8 storage.")
    role_to_code = {role: index + 1 for index, role in enumerate(role_values)}
    role_codes = np.asarray(
        [role_to_code[str(value)] for value in role_rows], dtype=np.uint8
    )
    relative_valid = pair(result.relative_valid).astype(bool, copy=False)
    relative_reason = _encode_reasons(
        pair(result.relative_reason_code), field="relative_reason_code"
    )
    relative_px = _float32_with_nan(pair(result.relative_xy), relative_valid)
    distance_px = _float32_with_nan(pair(result.distance_px), relative_valid)
    relative_physical = _float32_with_nan(
        pair(result.relative_xy_physical), relative_valid
    )
    distance_physical = _float32_with_nan(
        pair(result.distance_physical), relative_valid
    )

    nearest_valid_frame = np.asarray(result.nearest_valid, dtype=bool)
    nearest_source_frame = np.full(n_frames, -1, dtype=np.int64)
    nearest_identity_frame = np.zeros(n_frames, dtype=np.uint16)
    for row, nearest_index in enumerate(result.nearest_chaser_index):
        if not nearest_valid_frame[row]:
            continue
        index = int(nearest_index)
        if index < 0 or index >= n_chasers:
            _fail("A valid nearest-chaser row has an invalid chaser-axis index.")
        source_row = int(result.chaser_source_row_index[row, index])
        if source_row < 0:
            _fail("A valid nearest chaser lacks a nonnegative source-row identity.")
        nearest_source_frame[row] = source_row
        nearest_identity_frame[row] = identity_codes[index]
    nearest_member = (
        np.arange(n_chasers, dtype=np.int64)[None, :]
        == np.asarray(result.nearest_chaser_index, dtype=np.int64)[:, None]
    ) & nearest_valid_frame[:, None]

    base: dict[str, np.ndarray] = {
        "acquisition_frame_id": frame(result.frame_keys.acquisition_frame_id),
        "track_sample_id": frame(result.frame_keys.track_sample_id),
        "timestamp_ns": timestamp_ns,
        "timestamp_valid": timestamp_valid,
        "timestamp_reason_code": timestamp_reason,
        "fish_source_row_id": fish_rows,
        "fish_source_row_valid": fish_rows_valid,
        "fish_source_row_reason_code": np.where(
            fish_rows_valid, 0, _REASON_TO_CODE["source_row_unavailable"]
        ).astype(np.uint16),
        "chaser_source_row_id": chaser_rows,
        "chaser_source_row_valid": chaser_rows_valid,
        "chaser_source_row_reason_code": np.where(
            chaser_rows_valid, 0, _REASON_TO_CODE["source_row_unavailable"]
        ).astype(np.uint16),
        "fish_position_xy_px": _float32_with_nan(fish_xy, fish_valid),
        "fish_position_valid": fish_valid,
        "fish_position_reason_code": fish_reason,
        "chaser_position_xy_px": _float32_with_nan(chaser_xy, chaser_valid),
        "chaser_position_valid": chaser_valid,
        "chaser_position_reason_code": chaser_reason,
        "fish_identity_code": np.ones(n_rows, dtype=np.uint16),
        "chaser_identity_code": chaser_identity_code,
        "chaser_behavior_role_code": role_codes,
        "chaser_behavior_role_valid": np.ones(n_rows, dtype=bool),
        "chaser_behavior_role_reason_code": np.zeros(n_rows, dtype=np.uint16),
        "selection_member": frame(result.selection_membership),
        "chaser_occurrence_member": pair(result.occurrence_membership),
        "row_valid": relative_valid,
        "row_reason_code": relative_reason,
        "acquisition_frame_delta": frame(result.acquisition_frame_delta),
        "timestamp_delta_ns": frame(result.timestamp_delta_ns),
        "fish_transition_valid": frame(result.fish_transition_valid),
        "fish_transition_reason_code": _encode_reasons(
            frame(result.fish_transition_reason_code),
            field="fish_transition_reason_code",
        ),
        "relative_transition_valid": pair(result.relative_transition_valid),
        "relative_transition_reason_code": _encode_reasons(
            pair(result.relative_transition_reason_code),
            field="relative_transition_reason_code",
        ),
        "relative_vector_px_xy": relative_px,
        "relative_distance_px": distance_px,
        "relative_px_valid": relative_valid,
        "relative_px_reason_code": relative_reason,
        "relative_vector_physical_xy": relative_physical,
        "relative_distance_physical": distance_physical,
        "relative_physical_valid": relative_valid,
        "relative_physical_reason_code": relative_reason,
        "nearest_chaser_member": nearest_member.reshape(-1),
        "nearest_chaser_identity_code": frame(nearest_identity_frame),
        "nearest_chaser_source_row_id": frame(nearest_source_frame),
        "nearest_chaser_distance_px": _float32_with_nan(
            frame(result.nearest_distance_px), frame(nearest_valid_frame)
        ),
        "nearest_chaser_distance_physical": _float32_with_nan(
            frame(result.nearest_distance_physical), frame(nearest_valid_frame)
        ),
        "nearest_chaser_valid": frame(nearest_valid_frame),
        "nearest_chaser_reason_code": _encode_reasons(
            frame(result.nearest_reason_code), field="nearest_reason_code"
        ),
    }

    if result.chaser_trial_ids is not None:
        trial = pair(result.chaser_trial_ids)
        trial_valid = trial >= 0
        base.update(
            {
                "trial_id": trial,
                "trial_valid": trial_valid,
                "trial_reason_code": np.where(
                    trial_valid, 0, _REASON_TO_CODE["trial_unavailable"]
                ).astype(np.uint16),
            }
        )
    if result.chaser_active is not None:
        active = pair(result.chaser_active).astype(bool, copy=False)
        base.update(
            {
                "active_state_code": active.astype(np.uint8),
                "active_state_valid": np.ones(n_rows, dtype=bool),
                "active_state_reason_code": np.zeros(n_rows, dtype=np.uint16),
            }
        )

    body_arrays: dict[str, np.ndarray] | None = None
    if result.body_frame_present:
        body_valid_frame = np.asarray(result.body_frame_valid, dtype=bool)
        body_valid = frame(body_valid_frame)
        body_reason = _encode_reasons(
            frame(result.body_frame_reason_code), field="body_frame_reason_code"
        )
        body_source = frame(result.body_frame_source_row_index)
        body_source_valid = body_source >= 0
        if np.any(body_valid & ~body_source_valid):
            _fail("A valid body frame lacks a nonnegative source-row identity.")
        body_relative_values = pair(result.body_relative_xy)
        body_relative_valid = _valid_float_rows(body_relative_values)
        body_pair_reason_text = pair(result.egocentric_reason_code).astype(
            np.dtypes.StringDType()
        )
        body_pair_reason_text[body_relative_valid] = "valid"
        body_pair_reason = _encode_reasons(
            body_pair_reason_text, field="body_relative_reason_code"
        )
        bearing_valid = pair(result.egocentric_valid).astype(bool, copy=False)
        bearing_reason = _encode_reasons(
            pair(result.egocentric_reason_code), field="body_bearing_reason_code"
        )
        body_arrays = {
            "body_source_row_id": body_source,
            "body_source_row_valid": body_source_valid,
            "body_source_row_reason_code": np.where(
                body_source_valid, 0, _REASON_TO_CODE["source_row_unavailable"]
            ).astype(np.uint16),
            "body_origin_xy_px": _float32_with_nan(
                frame(result.body_frame_origin_xy), body_valid
            ),
            "body_forward_axis_xy": _float32_with_nan(
                frame(result.body_frame_forward_axis_xy), body_valid
            ),
            "body_left_axis_xy": _float32_with_nan(
                frame(result.body_frame_left_axis_xy), body_valid
            ),
            "body_origin_valid": body_valid,
            "body_origin_reason_code": body_reason,
            "body_axes_valid": body_valid,
            "body_axes_reason_code": body_reason,
            "body_relative_vector_px_xy": _float32_with_nan(
                body_relative_values, body_relative_valid
            ),
            "body_relative_px_valid": body_relative_valid,
            "body_relative_px_reason_code": body_pair_reason,
            "body_relative_vector_physical_xy": _float32_with_nan(
                pair(result.body_relative_xy_physical), body_relative_valid
            ),
            "body_relative_physical_valid": body_relative_valid,
            "body_relative_physical_reason_code": body_pair_reason,
            "body_heading_deg": _float32_with_nan(
                frame(result.body_frame_heading_deg), body_valid
            ),
            "body_heading_valid": body_valid,
            "body_heading_reason_code": body_reason,
            "body_heading_transition_valid": frame(
                result.heading_transition_valid
            ),
            "body_heading_transition_reason_code": _encode_reasons(
                frame(result.heading_transition_reason_code),
                field="body_heading_transition_reason_code",
            ),
            "body_forward_coordinate_px": _float32_with_nan(
                pair(result.forward_coordinate_px), body_relative_valid
            ),
            "body_left_coordinate_px": _float32_with_nan(
                pair(result.left_coordinate_px), body_relative_valid
            ),
            "body_coordinates_px_valid": body_relative_valid,
            "body_coordinates_px_reason_code": body_pair_reason,
            "body_forward_coordinate_physical": _float32_with_nan(
                pair(result.forward_coordinate_physical), body_relative_valid
            ),
            "body_left_coordinate_physical": _float32_with_nan(
                pair(result.left_coordinate_physical), body_relative_valid
            ),
            "body_coordinates_physical_valid": body_relative_valid,
            "body_coordinates_physical_reason_code": body_pair_reason,
            "body_bearing_deg": _float32_with_nan(
                pair(result.egocentric_bearing_deg), bearing_valid
            ),
            "body_bearing_valid": bearing_valid,
            "body_bearing_reason_code": bearing_reason,
            "body_valid": body_relative_valid,
            "body_reason_code": body_pair_reason,
        }

    base = {name: _readonly(value, np.asarray(value).dtype) for name, value in base.items()}
    if body_arrays is not None:
        body_arrays = {
            name: _readonly(value, np.asarray(value).dtype)
            for name, value in body_arrays.items()
        }
    CHASER_RELATIVE_FRAME_SCHEMA_V1.require(
        base,
        dimensions=dimensions,
        body_arrays=body_arrays,
    )

    schema_binding = {
        "schema_id": CHASER_RELATIVE_FRAME_SCHEMA_ID,
        "schema_version": CHASER_RELATIVE_FRAME_SCHEMA_VERSION,
        "layout": CHASER_RELATIVE_FRAME_LAYOUT,
        "body_extension_present": body_arrays is not None,
    }
    context_manifest = context.as_manifest()
    payload: dict[str, Any] = {
        "schema_id": PREPARED_CHASER_RELATIVE_FRAME_SCHEMA_ID,
        "schema_version": PREPARED_CHASER_RELATIVE_FRAME_SCHEMA_VERSION,
        "recording_id": recording_id,
        "selector_eligible": False,
        "selection": "none",
        "candidate_state": "validated_prepared_selector_ineligible",
        "computation_id": COMPUTATION_ID,
        "flatten_policy_id": FLATTEN_POLICY_ID,
        "dimensions": {
            "n_frames": n_frames,
            "n_chasers": n_chasers,
            "n_rows": n_rows,
        },
        "schema_binding": schema_binding,
        "identity_registries": {
            "fish": {"1": context.fish_identity},
            "chaser": {
                str(index + 1): identity
                for index, identity in enumerate(result.chaser_identities)
            },
            "behavior_role": {
                str(code): role for role, code in role_to_code.items()
            },
            "active_state": {"0": "inactive", "1": "active"},
        },
        "reason_codes": {
            str(code): reason
            for code, reason in CHASER_RELATIVE_FRAME_REASON_CODES.items()
        },
        "context": context_manifest,
        "source_authorities": {
            "fish_position": _authority_record(result.fish_authority),
            "chaser_position": _authority_record(result.chaser_authority),
            "body_frame": (
                None
                if result.body_frame_authority is None
                else _authority_record(result.body_frame_authority)
            ),
        },
        "coordinate_policy": {
            "policy_id": result.coordinate_policy.policy_id,
            "coordinate_authority_id": result.coordinate_policy.coordinate_authority_id,
            "coordinate_frame": result.coordinate_policy.coordinate_frame,
            "origin": result.coordinate_policy.origin,
            "x_axis_direction": result.coordinate_policy.x_axis_direction,
            "y_axis_direction": result.coordinate_policy.y_axis_direction,
        },
        "scale_policy": {
            "policy_id": result.scale_policy.policy_id,
            "scale_authority_id": result.scale_policy.scale_authority_id,
            "scale_digest": result.scale_policy.scale_digest,
            "pixels_per_unit": result.scale_policy.pixels_per_unit,
            "unit": result.scale_policy.unit,
        },
        "timing_policy": {
            "policy_id": result.timing_policy.policy_id,
            "timing_authority_id": result.timing_policy.timing_authority_id,
            "timing_digest": result.timing_policy.timing_digest,
            "frame_key_name": result.timing_policy.frame_key_name,
            "track_sample_key_name": result.timing_policy.track_sample_key_name,
            "timestamp_field": result.timing_policy.timestamp_field,
        },
        "active_position_validity_policy": {
            "policy_id": result.active_position_validity_policy,
            "active_state_present": result.chaser_active is not None,
            "active_state_surface": (
                "base/active_state_code"
                if result.chaser_active is not None
                else None
            ),
            "position_validity_semantics": (
                "controller activity is preserved as evidence and does not "
                "invalidate otherwise finite selected occurring fish/chaser geometry"
            ),
        },
        "array_declarations": _array_declarations(base, body_arrays),
        "metadata_policy": {
            "row_evidence": "typed_arrays_only",
            "authority_records": "readable_bounded_records_plus_sha256",
            "maximum_context_record_bytes": MAX_CONTEXT_RECORD_BYTES,
        },
    }
    manifest = {**payload, "payload_digest": canonical_json_sha256(payload)}
    prepared = PreparedChaserRelativeFrame(
        dimensions=dimensions,
        base_arrays=MappingProxyType(base),
        body_arrays=(
            None if body_arrays is None else MappingProxyType(body_arrays)
        ),
        manifest=MappingProxyType(manifest),
    )
    validate_prepared_chaser_relative_frame(prepared)
    return prepared


__all__ = [
    "COMPUTATION_ID",
    "FLATTEN_POLICY_ID",
    "MAX_CONTEXT_RECORD_BYTES",
    "PREPARED_CHASER_RELATIVE_FRAME_SCHEMA_ID",
    "PREPARED_CHASER_RELATIVE_FRAME_SCHEMA_VERSION",
    "ChaserRelativeFramePublicationContext",
    "ChaserRelativeFrameStorageError",
    "PreparedChaserRelativeFrame",
    "prepare_chaser_relative_frame",
    "validate_chaser_input_provenance_projection_binding",
    "validate_prepared_chaser_relative_frame",
]
