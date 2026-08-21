"""Pure preparation of a provider-aware chaser-distance successor.

The successor is an in-memory, selector-ineligible analysis candidate.  It
consumes one exact verified chaser-relative-frame handle through
``load_chaser_relative_distance_view`` and changes only the vocabulary and
physical-distance presentation of that already sealed relation table.

It deliberately does not select a provider, resolve a selector, interpolate,
match timestamps, reconstruct display presentation, or write Zarr data.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    BEHAVIORAL_DENOMINATOR,
    CAMERA_EXPOSURE_REFERENCE,
    CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE,
    PHYSICAL_PRESENTATION_VERIFIED,
    PRESENTATION_TIMESTAMP_AVAILABLE,
    PROXY_POLICY_ID,
    SCIENTIFIC_USE_CLASS,
    TEMPORAL_ALIGNMENT_CLASS,
    TEMPORAL_ALIGNMENT_REQUIREMENT,
)
from fisheye.analysis_workflows.chaser_relative_distance_view import (
    ChaserRelativeDistanceView,
    load_chaser_relative_distance_view,
)
from fisheye.analysis_workflows.chaser_relative_frame_storage import (
    validate_chaser_input_provenance_projection_binding,
)
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_ID,
    CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_VERSION,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.provider_chaser_distance_schema import (
    PROVIDER_CHASER_DISTANCE_LAYOUT,
    PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
    PROVIDER_CHASER_DISTANCE_SCHEMA_V1,
    PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
    ProviderChaserDistanceDimensions,
)


PREPARED_PROVIDER_CHASER_DISTANCE_SCHEMA_ID = (
    "palette.analysis.provider_chaser_distance.prepared_successor"
)
PREPARED_PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION = 1
COMPUTATION_ID = "provider_chaser_distance_successor_v1"
SOURCE_POSITION_ROW_ID_SEMANTICS = (
    "original_provider_row_identity_in_bound_source_position_provider;"
    "not_output_row_number_or_inferred_instance_key"
)


class ProviderChaserDistanceSuccessorError(ValueError):
    """Raised when an exact provider-aware successor cannot be prepared."""


def _fail(message: str) -> None:
    raise ProviderChaserDistanceSuccessorError(message)


def _copy_readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _frame_rows(view: ChaserRelativeDistanceView, name: str) -> np.ndarray:
    values = np.asarray(view.frame_array(name))
    return np.repeat(values, view.n_chasers, axis=0)


def _pair_rows(view: ChaserRelativeDistanceView, name: str) -> np.ndarray:
    return np.asarray(view.pair_array(name)).reshape(
        (view.n_rows,) + np.asarray(view.pair_array(name)).shape[2:]
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _authority(value: object, *, field: str, recording_id: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one verified provider authority record.")
    expected = {
        "recording_id",
        "source_authority_id",
        "source_digest",
        "provider_id",
        "provider_digest",
        "coordinate_authority_id",
        "scale_authority_id",
        "timing_authority_id",
        "row_axis_authority_id",
        "row_axis_authority_digest",
    }
    if set(value) != expected:
        _fail(f"{field} has missing or extra authority fields.")
    result: dict[str, str] = {}
    for name in sorted(expected):
        result[name] = _text(value[name], field=f"{field}.{name}")
    if result["recording_id"] != recording_id:
        _fail(f"{field}.recording_id does not match the source recording.")
    return result


def _record_from_context(
    handle: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    context = getattr(handle, "context", None)
    if not isinstance(context, Mapping):
        _fail("Verified source handle lacks its typed context mapping.")
    envelope = context.get(name)
    if not isinstance(envelope, Mapping) or set(envelope) != {"record", "sha256"}:
        _fail(f"Verified source context {name!r} is not an exact record envelope.")
    record = envelope["record"]
    if not isinstance(record, Mapping):
        _fail(f"Verified source context {name!r} record is not an object.")
    if envelope["sha256"] != canonical_json_sha256(dict(record)):
        _fail(f"Verified source context {name!r} digest does not match its record.")
    return record


def _proxy_record(handle: Any, *, recording_id: str, n_frames: int, n_chasers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    projection = dict(_record_from_context(handle, name="acquisition_projection"))
    publication = dict(_record_from_context(handle, name="acquisition_projection_publication"))
    try:
        validate_chaser_input_provenance_projection_binding(
            projection=projection,
            publication=publication,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Verified source proxy binding is invalid: {exc}")
    if projection.get("recording_id") != recording_id:
        _fail("Source proxy projection belongs to another recording.")
    if projection.get("policy_id") != PROXY_POLICY_ID:
        _fail("Provider chaser-distance successor requires the explicit proxy policy.")
    expected = {
        "temporal_alignment_requirement": TEMPORAL_ALIGNMENT_REQUIREMENT,
        "temporal_alignment_class": TEMPORAL_ALIGNMENT_CLASS,
        "physical_presentation_verified": PHYSICAL_PRESENTATION_VERIFIED,
        "presentation_timestamp_available": PRESENTATION_TIMESTAMP_AVAILABLE,
        "camera_presentation_clock_transform_available": CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE,
        "camera_exposure_reference": CAMERA_EXPOSURE_REFERENCE,
        "scientific_use_class": SCIENTIFIC_USE_CLASS,
        "behavioral_denominator": BEHAVIORAL_DENOMINATOR,
    }
    for name, expected_value in expected.items():
        if projection.get(name) != expected_value:
            _fail(f"Source proxy projection has invalid {name!r}.")
    if projection["unique_acquisition_frame_count"] != n_frames:
        _fail("Source proxy frame count does not match the relative-frame view.")
    if projection["chaser_count"] != n_chasers:
        _fail("Source proxy chaser count does not match the relative-frame view.")
    return projection, publication


def _scale_policy(handle: Any) -> dict[str, Any]:
    manifest = getattr(handle, "run_manifest", None)
    if not isinstance(manifest, Mapping):
        _fail("Verified source handle lacks its run manifest.")
    policy = manifest.get("scale_policy")
    if not isinstance(policy, Mapping):
        _fail("Provider chaser-distance successor requires a scale policy.")
    result = dict(policy)
    if result.get("unit") != "mm":
        _fail("Provider chaser-distance successor requires an explicit millimetre scale.")
    pixels_per_unit = result.get("pixels_per_unit")
    if (
        isinstance(pixels_per_unit, bool)
        or not isinstance(pixels_per_unit, (int, float))
        or not np.isfinite(float(pixels_per_unit))
        or float(pixels_per_unit) <= 0
    ):
        _fail("Scale policy must provide positive finite pixels_per_unit in mm.")
    for name in ("policy_id", "scale_authority_id", "scale_digest", "unit"):
        _text(result.get(name), field=f"scale_policy.{name}")
    result["pixels_per_unit"] = float(pixels_per_unit)
    return result


def _coordinate_policy(handle: Any) -> dict[str, Any]:
    manifest = getattr(handle, "run_manifest", None)
    policy = manifest.get("coordinate_policy") if isinstance(manifest, Mapping) else None
    if not isinstance(policy, Mapping):
        _fail("Verified source handle lacks its coordinate policy.")
    result = dict(policy)
    expected = {
        "coordinate_frame": "source_camera_pixels",
        "origin": "top_left",
        "x_axis_direction": "right",
        "y_axis_direction": "down",
    }
    for name, value in expected.items():
        if result.get(name) != value:
            _fail(f"Source coordinate policy has invalid {name!r}.")
    return result


def _declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "dtype": array.dtype.str,
            "shape": [int(value) for value in array.shape],
        }
        for name, array in sorted(arrays.items())
    ]


@dataclass(frozen=True, slots=True)
class PreparedProviderChaserDistance:
    """Read-only successor values and bounded provenance for later storage."""

    dimensions: ProviderChaserDistanceDimensions
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider chaser-distance array {name!r}.") from exc

    def to_json(self) -> dict[str, Any]:
        return _plain(self.manifest)


def build_provider_chaser_distance_successor(
    source_handle: Any,
) -> PreparedProviderChaserDistance:
    """Build one pure successor from an exact verified source handle.

    The source view is always obtained through the strict loader.  No output
    selection or publication state is changed here.
    """

    distance_view = load_chaser_relative_distance_view(source_handle)
    if not isinstance(distance_view, ChaserRelativeDistanceView):
        _fail("Source loader did not return a ChaserRelativeDistanceView.")
    dimensions = ProviderChaserDistanceDimensions(
        n_frames=distance_view.n_frames,
        n_chasers=distance_view.n_chasers,
    )
    recording_id = _text(distance_view.recording_id, field="recording_id")
    projection, publication = _proxy_record(
        source_handle,
        recording_id=recording_id,
        n_frames=dimensions.n_frames,
        n_chasers=dimensions.n_chasers,
    )
    scale_policy = _scale_policy(source_handle)
    coordinate_policy = _coordinate_policy(source_handle)
    timing_policy = dict(getattr(source_handle, "run_manifest", {}).get("timing_policy", {}))
    source_authorities = getattr(source_handle, "source_authorities", None)
    if not isinstance(source_authorities, Mapping):
        _fail("Verified source handle lacks source provider authorities.")
    source_authority = _authority(
        source_authorities.get("fish_position"),
        field="source_authorities.fish_position",
        recording_id=recording_id,
    )
    chaser_authority = _authority(
        source_authorities.get("chaser_position"),
        field="source_authorities.chaser_position",
        recording_id=recording_id,
    )

    source_position = _frame_rows(distance_view, "fish_position_xy_px")
    source_position_valid = _frame_rows(distance_view, "fish_position_valid")
    chaser_position = _pair_rows(distance_view, "chaser_position_xy_px")
    chaser_position_valid = _pair_rows(distance_view, "chaser_position_valid")
    distance_px_valid = _pair_rows(distance_view, "relative_px_valid")
    distance_px = _pair_rows(distance_view, "relative_distance_px")
    if np.any(distance_px_valid):
        expected_distance = np.linalg.norm(
            chaser_position[distance_px_valid].astype(np.float64)
            - source_position[distance_px_valid].astype(np.float64),
            axis=1,
        )
        if not np.allclose(
            distance_px[distance_px_valid], expected_distance, atol=5e-4, rtol=0.0
        ):
            _fail("Source relative distance does not equal source-to-chaser pixel geometry.")
    if np.any(distance_px_valid & (~source_position_valid | ~chaser_position_valid)):
        _fail("Source relative distance marks a row valid without valid positions.")
    distance_mm = np.full(distance_px.shape, np.nan, dtype=np.float32)
    distance_mm[distance_px_valid] = (
        distance_px[distance_px_valid].astype(np.float64)
        / scale_policy["pixels_per_unit"]
    ).astype(np.float32)

    arrays: dict[str, np.ndarray] = {
        "acquisition_frame_id": _copy_readonly(_frame_rows(distance_view, "acquisition_frame_id")),
        "track_sample_id": _copy_readonly(_frame_rows(distance_view, "track_sample_id")),
        "timestamp_ns": _copy_readonly(_frame_rows(distance_view, "timestamp_ns")),
        "timestamp_valid": _copy_readonly(_frame_rows(distance_view, "timestamp_valid")),
        "timestamp_reason_code": _copy_readonly(_frame_rows(distance_view, "timestamp_reason_code")),
        "source_position_row_id": _copy_readonly(_frame_rows(distance_view, "fish_source_row_id")),
        "source_position_row_valid": _copy_readonly(_frame_rows(distance_view, "fish_source_row_valid")),
        "source_position_row_reason_code": _copy_readonly(_frame_rows(distance_view, "fish_source_row_reason_code")),
        "source_position_xy_px": _copy_readonly(source_position),
        "source_position_valid": _copy_readonly(source_position_valid),
        "source_position_reason_code": _copy_readonly(_frame_rows(distance_view, "fish_position_reason_code")),
        "fish_identity_code": _copy_readonly(_frame_rows(distance_view, "fish_identity_code")),
        "selection_member": _copy_readonly(_frame_rows(distance_view, "selection_member")),
        "acquisition_frame_delta": _copy_readonly(_frame_rows(distance_view, "acquisition_frame_delta")),
        "timestamp_delta_ns": _copy_readonly(_frame_rows(distance_view, "timestamp_delta_ns")),
        "chaser_position_row_id": _copy_readonly(_pair_rows(distance_view, "chaser_source_row_id")),
        "chaser_position_row_valid": _copy_readonly(_pair_rows(distance_view, "chaser_source_row_valid")),
        "chaser_position_row_reason_code": _copy_readonly(_pair_rows(distance_view, "chaser_source_row_reason_code")),
        "chaser_position_xy_px": _copy_readonly(chaser_position),
        "chaser_position_valid": _copy_readonly(chaser_position_valid),
        "chaser_position_reason_code": _copy_readonly(_pair_rows(distance_view, "chaser_position_reason_code")),
        "chaser_identity_code": _copy_readonly(_pair_rows(distance_view, "chaser_identity_code")),
        "chaser_behavior_role_code": _copy_readonly(_pair_rows(distance_view, "chaser_behavior_role_code")),
        "chaser_behavior_role_valid": _copy_readonly(_pair_rows(distance_view, "chaser_behavior_role_valid")),
        "chaser_behavior_role_reason_code": _copy_readonly(_pair_rows(distance_view, "chaser_behavior_role_reason_code")),
        "chaser_occurrence_member": _copy_readonly(_pair_rows(distance_view, "chaser_occurrence_member")),
        "nearest_chaser_member": _copy_readonly(_pair_rows(distance_view, "nearest_chaser_member")),
        "row_valid": _copy_readonly(_pair_rows(distance_view, "row_valid")),
        "row_reason_code": _copy_readonly(_pair_rows(distance_view, "row_reason_code")),
        "relative_transition_valid": _copy_readonly(_pair_rows(distance_view, "relative_transition_valid")),
        "relative_transition_reason_code": _copy_readonly(_pair_rows(distance_view, "relative_transition_reason_code")),
        "relative_vector_px_xy": _copy_readonly(_pair_rows(distance_view, "relative_vector_px_xy")),
        "distance_px": _copy_readonly(distance_px),
        "distance_px_valid": _copy_readonly(distance_px_valid),
        "distance_px_reason_code": _copy_readonly(_pair_rows(distance_view, "relative_px_reason_code")),
        "distance_mm": _copy_readonly(distance_mm),
        "distance_mm_valid": _copy_readonly(distance_px_valid),
        "distance_mm_reason_code": _copy_readonly(_pair_rows(distance_view, "relative_px_reason_code")),
    }
    for name in ("trial_id", "trial_valid", "trial_reason_code"):
        if name in distance_view.pair_arrays:
            arrays[name] = _copy_readonly(_pair_rows(distance_view, name))

    PROVIDER_CHASER_DISTANCE_SCHEMA_V1.require(arrays, dimensions=dimensions)

    native_sample_count = int(projection["native_sample_count"])
    selected_input_frame_count = int(projection["selected_acquisition_frame_count"])
    denominators = {
        "unique_acquisition_frame_count": dimensions.n_frames,
        "frame_x_chaser_relation_row_count": dimensions.n_rows,
        "valid_source_position_frame_count": int(np.count_nonzero(distance_view.frame_array("fish_position_valid"))),
        "valid_distance_relation_row_count": int(np.count_nonzero(distance_px_valid)),
        "native_stimulus_sample_count": native_sample_count,
        "selected_input_acquisition_frame_count": selected_input_frame_count,
    }
    source_manifest_sha256 = _text(
        getattr(source_handle, "manifest_sha256", None),
        field="source_handle.manifest_sha256",
    )
    source_payload_digest = _text(
        getattr(source_handle, "payload_digest", None),
        field="source_handle.payload_digest",
    )
    source_manifest = {
        "schema_id": CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_ID,
        "schema_version": CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_VERSION,
        "run_path": distance_view.source_run_path,
        "manifest_sha256": source_manifest_sha256,
        "payload_digest": source_payload_digest,
        "verification_digest": distance_view.source_run_digest,
    }
    payload: dict[str, Any] = {
        "schema_id": PREPARED_PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
        "schema_version": PREPARED_PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
        "recording_id": recording_id,
        "status": "prepared_selector_ineligible",
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "production_selector_activation": False,
        "computation_id": COMPUTATION_ID,
        "schema_binding": {
            "schema_id": PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
            "schema_version": PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
            "layout": PROVIDER_CHASER_DISTANCE_LAYOUT,
        },
        "dimensions": dimensions.as_manifest(),
        "source": source_manifest,
        "source_position_row_id_semantics": SOURCE_POSITION_ROW_ID_SEMANTICS,
        "provider_authorities": {
            "source_position": source_authority,
            "chaser_position": chaser_authority,
        },
        "coordinate_policy": coordinate_policy,
        "scale_policy": scale_policy,
        "timing_policy": timing_policy,
        "temporal_alignment": {
            "policy_id": projection["policy_id"],
            "temporal_alignment_requirement": projection["temporal_alignment_requirement"],
            "temporal_alignment_class": projection["temporal_alignment_class"],
            "physical_presentation_verified": projection["physical_presentation_verified"],
            "presentation_timestamp_available": projection["presentation_timestamp_available"],
            "camera_presentation_clock_transform_available": projection["camera_presentation_clock_transform_available"],
            "camera_exposure_reference": projection["camera_exposure_reference"],
            "scientific_use_class": projection["scientific_use_class"],
            "source_projection_record_sha256": canonical_json_sha256(projection),
            "source_projection_publication_record_sha256": canonical_json_sha256(publication),
            "timestamp_matching_performed": False,
        },
        "denominators": denominators,
        "denominator_policy": {
            "behavioral_frame_denominator": BEHAVIORAL_DENOMINATOR,
            "relation_denominator": "valid_distance_relation_row_count",
            "native_stimulus_denominator": "native_stimulus_sample_count",
            "native_sample_rows_preserved_in_source": True,
        },
        "optional_fields": {
            "trial_triple_present": all(name in arrays for name in ("trial_id", "trial_valid", "trial_reason_code")),
        },
        "array_declarations": _declarations(arrays),
        "invariants": PROVIDER_CHASER_DISTANCE_SCHEMA_V1.as_manifest(dimensions=dimensions)["invariants"],
    }
    payload["payload_digest"] = canonical_json_sha256(payload)
    return PreparedProviderChaserDistance(
        dimensions=dimensions,
        arrays=MappingProxyType(dict(arrays)),
        manifest=_freeze(payload),
    )


def prepare_provider_chaser_distance_successor(
    source_handle: Any,
) -> PreparedProviderChaserDistance:
    """Load the exact verified view and prepare its pure successor."""

    return build_provider_chaser_distance_successor(source_handle)


__all__ = [
    "COMPUTATION_ID",
    "PREPARED_PROVIDER_CHASER_DISTANCE_SCHEMA_ID",
    "PREPARED_PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION",
    "PreparedProviderChaserDistance",
    "ProviderChaserDistanceSuccessorError",
    "SOURCE_POSITION_ROW_ID_SEMANTICS",
    "build_provider_chaser_distance_successor",
    "prepare_provider_chaser_distance_successor",
]
