"""Exact provider-track source binding for provider spatial analytics.

This module is the canonical source adapter between two loader-minted
publications and :class:`ProviderTrackSamples`.  It deliberately has no
selector lookup, no Zarr writes, and no cardinality-based alignment.  The
complete subject-position and tracking rowsets are first proven to describe
the same keyed observation domain; only then are explicitly unassigned
tracking rows excluded from the single-subject analytical sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping
from pathlib import Path

import numpy as np

from fisheye.analysis.provider_spatial_trajectory import ProviderTrackSamples
from fisheye.analysis_workflows.subject_position_source_handle import (
    SubjectPositionSourceHandle,
    SubjectPositionSourceHandleError,
    load_subject_position_source_handle,
    require_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    TrackingSourceHandle,
    TrackingSourceHandleError,
    load_tracking_source_handle,
    require_tracking_source_handle,
)
from fisheye.shared.subject_position_types import (
    POSITION_FAILURE_REASON_TAGS,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root


PROVIDER_TRACK_SOURCE_SCHEMA_ID = "palette.provider_spatial_track_source"
PROVIDER_TRACK_SOURCE_SCHEMA_VERSION = 1
TRACK_SAMPLE_POLICY_ID = "single_subject_instance_key_one_sample_per_frame_v1"

_SHA256_LENGTH = 64
_MUTABLE_ID_ALIASES = frozenset(
    {
        "active",
        "authoritative",
        "current",
        "default",
        "latest",
        "latest_complete",
        "selected",
        "stale",
        "unknown",
    }
)
_SUPPORTED_MODALITIES = frozenset({"detection", "keypoint", "subject_mask"})


class ProviderSpatialTrackSourceError(ValueError):
    """Raised when exact provider and tracking authorities cannot be joined."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _text(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or value.lower() in _MUTABLE_ID_ALIASES
    ):
        raise ProviderSpatialTrackSourceError(
            f"{name} must be one immutable nonempty identity."
        )
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProviderSpatialTrackSourceError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _canonical_record(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ProviderSpatialTrackSourceError(f"{name} must be one nonempty object.")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ProviderSpatialTrackSourceError(
            f"{name} must be strict canonical JSON."
        ) from exc
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        raise ProviderSpatialTrackSourceError(f"{name} must be one object.")
    return decoded


def _read_array(node: Any, *, dtype: np.dtype[Any], name: str) -> np.ndarray:
    try:
        value = np.asarray(node[:])
    except (TypeError, ValueError, KeyError, OSError, RuntimeError) as exc:
        raise ProviderSpatialTrackSourceError(
            f"Unable to read exact {name} source array."
        ) from exc
    if value.dtype != dtype or value.ndim != 1:
        raise ProviderSpatialTrackSourceError(
            f"{name} must be one exact {dtype.str} vector."
        )
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _read_position_xy(node: Any, *, row_count: int) -> np.ndarray:
    value = np.asarray(node[:])
    if value.ndim != 2 or value.shape != (row_count, 2):
        raise ProviderSpatialTrackSourceError(
            "subject-position position_xy must have exact shape (N, 2)."
        )
    if value.dtype.kind not in "iu f".replace(" ", ""):
        raise ProviderSpatialTrackSourceError(
            "subject-position position_xy must be numeric."
        )
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ProviderTrackSourceAuthorities:
    """Explicit recording, timeline, and one-subject identity authorities."""

    recording_id: str
    timeline_authority_id: str
    subject_identity: str

    def __post_init__(self) -> None:
        for name in (
            "recording_id",
            "timeline_authority_id",
            "subject_identity",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name=name),
            )

    def as_record(self) -> dict[str, str]:
        return {
            "recording_id": self.recording_id,
            "timeline_authority_id": self.timeline_authority_id,
            "subject_identity": self.subject_identity,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.as_record())


# This spelling is useful at call sites that describe the three authorities as
# identity authorities rather than as a source bundle.
ProviderTrackIdentityAuthorities = ProviderTrackSourceAuthorities


@dataclass(frozen=True, init=False)
class ProviderTrackSourceEvidence:
    """Immutable evidence binding the two exact source publications."""

    record: Mapping[str, Any]
    sha256: str
    source_id: str
    track_sample_policy_id: str

    def __init__(
        self,
        *,
        record: Mapping[str, Any],
        sha256: str | None = None,
        source_id: str | None = None,
        track_sample_policy_id: str = TRACK_SAMPLE_POLICY_ID,
    ) -> None:
        canonical = _canonical_record(record, name="source binding evidence")
        expected = canonical_json_sha256(canonical)
        actual = expected if sha256 is None else _digest(sha256, name="sha256")
        if actual != expected:
            raise ProviderSpatialTrackSourceError(
                "source binding evidence digest is stale."
            )
        policy = _text(track_sample_policy_id, name="track_sample_policy_id")
        source = actual if source_id is None else _text(source_id, name="source_id")
        if source != actual:
            raise ProviderSpatialTrackSourceError(
                "source_id must equal the source binding evidence digest."
            )
        object.__setattr__(self, "record", _freeze(canonical))
        object.__setattr__(self, "sha256", actual)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "track_sample_policy_id", policy)

    @property
    def record_sha256(self) -> str:
        return self.sha256

    @property
    def authority_record(self) -> Mapping[str, Any]:
        return self.record


def _require_exact_revalidated_sources(
    position: object,
    tracking: object,
) -> tuple[SubjectPositionSourceHandle, TrackingSourceHandle]:
    try:
        supplied_position = require_subject_position_source_handle(position)
        supplied_tracking = require_tracking_source_handle(tracking)
        if supplied_position.analysis_zarr_path != supplied_tracking.analysis_zarr_path:
            raise ProviderSpatialTrackSourceError(
                "subject-position and tracking authorities belong to different analysis archives."
            )
        refreshed_position = load_subject_position_source_handle(
            supplied_position.analysis_zarr_path,
            supplied_position.run_path,
            expected_selector_eligible=supplied_position.selector_eligible,
            use_consolidated=False,
            expected_manifest_sha256=supplied_position.manifest_sha256,
        )
        refreshed_tracking = load_tracking_source_handle(
            supplied_tracking.analysis_zarr_path,
            supplied_tracking.run_path,
            expected_selector_eligible=supplied_tracking.selector_eligible,
            use_consolidated=False,
            expected_manifest_sha256=supplied_tracking.manifest_sha256,
        )
    except (
        SubjectPositionSourceHandleError,
        TrackingSourceHandleError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        if isinstance(exc, ProviderSpatialTrackSourceError):
            raise
        raise ProviderSpatialTrackSourceError(
            f"exact provider-track source revalidation failed: {exc}"
        ) from exc
    if refreshed_position.analysis_zarr_path != refreshed_tracking.analysis_zarr_path:
        raise ProviderSpatialTrackSourceError(
            "revalidated provider and tracking sources belong to different archives."
        )
    if (
        refreshed_position.decoded_content_sha256
        != supplied_position.decoded_content_sha256
        or refreshed_tracking.verification_digest != supplied_tracking.verification_digest
    ):
        raise ProviderSpatialTrackSourceError(
            "a source changed after its loader-minted handle was sealed."
        )
    return refreshed_position, refreshed_tracking


def _check_authority_mirrors(
    archive: Path,
    authorities: ProviderTrackSourceAuthorities,
    position: SubjectPositionSourceHandle,
    tracking: TrackingSourceHandle,
) -> None:
    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    attrs = root.attrs
    if attrs.get("recording_id") != authorities.recording_id:
        raise ProviderSpatialTrackSourceError(
            "recording authority does not match the analysis archive recording_id."
        )
    optional_root_mirrors = {
        "timeline_authority_id": authorities.timeline_authority_id,
        "subject_identity": authorities.subject_identity,
    }
    for name, expected in optional_root_mirrors.items():
        if name in attrs and attrs[name] != expected:
            raise ProviderSpatialTrackSourceError(
                f"analysis archive {name} differs from the explicit authority."
            )

    records: list[tuple[str, Mapping[str, Any]]] = [
        ("subject-position source", position.source_record),
    ]
    payload = tracking.manifest.get("payload")
    if isinstance(payload, Mapping) and isinstance(payload.get("source"), Mapping):
        records.append(("tracking source", payload["source"]))
    for label, record in records:
        for field, expected in (
            ("recording_id", authorities.recording_id),
            ("recording_identity", authorities.recording_id),
            ("timeline_authority_id", authorities.timeline_authority_id),
            ("timeline_id", authorities.timeline_authority_id),
            ("subject_identity", authorities.subject_identity),
            ("subject_id", authorities.subject_identity),
        ):
            if field in record and record[field] != expected:
                raise ProviderSpatialTrackSourceError(
                    f"{label} {field} differs from the explicit authority."
                )


def _build_source_binding(
    *,
    authorities: ProviderTrackSourceAuthorities,
    position: SubjectPositionSourceHandle,
    tracking: TrackingSourceHandle,
    modality: str,
    estimator_id: str,
    source_keys: np.ndarray,
    source_frames: np.ndarray,
    source_reasons: np.ndarray,
    tracking_keys: np.ndarray,
    tracking_frames: np.ndarray,
    tracking_ids: np.ndarray,
    assigned_rows: np.ndarray,
    position_rows: np.ndarray,
    output_order: np.ndarray,
    provider_reason_tags: tuple[str, ...],
    unassigned_track_id: int,
) -> ProviderTrackSourceEvidence:
    record = {
        "schema_id": PROVIDER_TRACK_SOURCE_SCHEMA_ID,
        "schema_version": PROVIDER_TRACK_SOURCE_SCHEMA_VERSION,
        "track_sample_policy_id": TRACK_SAMPLE_POLICY_ID,
        "authorities": authorities.as_record(),
        "archive": {
            "analysis_zarr_path": str(position.analysis_zarr_path),
            "recording_id": authorities.recording_id,
        },
        "provider": {
            "source_modality": modality,
            "estimator_id": estimator_id,
            "estimator_sha256": position.estimator_sha256,
            "coordinate_sha256": position.coordinate_sha256,
        },
        "subject_position_source": {
            "run_path": position.run_path,
            "manifest_sha256": position.manifest_sha256,
            "decoded_content_sha256": position.decoded_content_sha256,
            "row_count": int(source_keys.size),
            "instance_key_sha256": array_values_sha256(source_keys),
            "frame_index_sha256": array_values_sha256(source_frames),
            "failure_reason_code_sha256": array_values_sha256(source_reasons),
            "failure_reason_codes": [int(value) for value in source_reasons.tolist()],
            "failure_reason_tags": list(
                POSITION_FAILURE_REASON_TAGS[int(value)] for value in source_reasons
            ),
        },
        "tracking_source": {
            "run_path": tracking.run_path,
            "manifest_sha256": tracking.manifest_sha256,
            "verification_digest": tracking.verification_digest,
            "row_count": int(tracking_keys.size),
            "instance_key_sha256": array_values_sha256(tracking_keys),
            "frame_index_sha256": array_values_sha256(tracking_frames),
            "track_id_sha256": array_values_sha256(tracking_ids),
            "unassigned_track_id": int(unassigned_track_id),
        },
        "keyed_join": {
            "mode": "exact_instance_key_set_join_v1",
            "source_row_count": int(source_keys.size),
            "tracking_row_count": int(tracking_keys.size),
            "assigned_row_count": int(assigned_rows.size),
            "unassigned_row_count": int(tracking_keys.size - assigned_rows.size),
            "assigned_tracking_row_index_sha256": array_values_sha256(
                assigned_rows
            ),
            "matched_subject_position_row_index_sha256": array_values_sha256(
                position_rows
            ),
            "output_order_sha256": array_values_sha256(output_order),
            "provider_reason_tags": list(provider_reason_tags),
        },
    }
    return ProviderTrackSourceEvidence(record=record)


def build_provider_track_source(
    position_source: SubjectPositionSourceHandle,
    tracking_source: TrackingSourceHandle,
    *,
    authorities: ProviderTrackSourceAuthorities,
) -> tuple[ProviderTrackSamples, ProviderTrackSourceEvidence]:
    """Build exact provider track samples and immutable source evidence.

    The complete source rowsets must have exactly the same ``instance_key`` set.
    Tracking rows carrying the manifest's unassigned track ID are retained in
    the evidence but excluded from the returned single-subject sample.  The
    remaining rows must have one assigned track and at most one sample per
    acquisition frame.  Reordered source rows are accepted because the join is
    keyed; the returned rows are canonical track/frame order for the trajectory
    contract.
    """

    if type(authorities) is not ProviderTrackSourceAuthorities:
        raise ProviderSpatialTrackSourceError(
            "explicit ProviderTrackSourceAuthorities are required."
        )
    position, tracking = _require_exact_revalidated_sources(
        position_source,
        tracking_source,
    )
    if position.analysis_zarr_path != tracking.analysis_zarr_path:
        raise ProviderSpatialTrackSourceError(
            "provider and tracking sources must share one analysis archive."
        )
    _check_authority_mirrors(
        position.analysis_zarr_path,
        authorities,
        position,
        tracking,
    )

    estimator_id = _text(
        position.estimator_record.get("estimator_id"),
        name="subject-position estimator_id",
    )
    modality = position.estimator_record.get("source_modality")
    if modality not in _SUPPORTED_MODALITIES:
        raise ProviderSpatialTrackSourceError(
            f"unsupported subject-position source modality: {modality!r}."
        )
    modality = str(modality)

    source_keys = _read_array(
        position.instance_key_node,
        dtype=np.dtype("uint64"),
        name="subject-position instance_key",
    )
    source_frames = _read_array(
        position.source_acquisition_frame_index_node,
        dtype=np.dtype("int64"),
        name="subject-position source_acquisition_frame_index",
    )
    source_valid = _read_array(
        position.valid_node,
        dtype=np.dtype("bool"),
        name="subject-position valid",
    )
    source_reasons = _read_array(
        position.failure_reason_codes_node,
        dtype=np.dtype("uint16"),
        name="subject-position failure_reason_codes",
    )
    source_positions = _read_position_xy(
        position.position_xy_node,
        row_count=int(source_keys.size),
    )
    if any(
        values.size != source_keys.size
        for values in (source_frames, source_valid, source_reasons)
    ):
        raise ProviderSpatialTrackSourceError(
            "subject-position source arrays have mismatched cardinality."
        )
    if np.unique(source_keys).size != source_keys.size:
        raise ProviderSpatialTrackSourceError(
            "subject-position instance_key contains duplicates."
        )
    if np.any(source_valid & ~np.isfinite(source_positions).all(axis=1)):
        raise ProviderSpatialTrackSourceError(
            "subject-position marks a nonfinite position valid."
        )
    reason_tags: list[str] = []
    for row, code in enumerate(source_reasons.tolist()):
        tag = POSITION_FAILURE_REASON_TAGS.get(int(code))
        if tag is None:
            raise ProviderSpatialTrackSourceError(
                f"subject-position failure reason code is unknown at row {row}."
            )
        if bool(source_valid[row]) != (tag == "ok"):
            raise ProviderSpatialTrackSourceError(
                "subject-position valid and failure reason code disagree."
            )
        reason_tags.append(str(tag))

    tracking_keys = np.asarray(tracking.instance_key)
    tracking_ids = np.asarray(tracking.track_ids)
    tracking_frames = np.asarray(tracking.frame_indices)
    if (
        tracking_keys.dtype != np.dtype("uint64")
        or tracking_keys.ndim != 1
        or tracking_ids.dtype != np.dtype("int64")
        or tracking_ids.ndim != 1
        or tracking_frames.dtype != np.dtype("int64")
        or tracking_frames.ndim != 1
    ):
        raise ProviderSpatialTrackSourceError(
            "tracking source must expose exact uint64 instance keys and int64 track/frame arrays."
        )
    if np.unique(tracking_keys).size != tracking_keys.size:
        raise ProviderSpatialTrackSourceError("tracking instance_key contains duplicates.")
    if not (
        tracking_ids.size == tracking_keys.size
        and tracking_frames.size == tracking_keys.size
    ):
        raise ProviderSpatialTrackSourceError(
            "tracking source arrays have mismatched cardinality."
        )

    source_key_set = set(int(value) for value in source_keys.tolist())
    tracking_key_set = set(int(value) for value in tracking_keys.tolist())
    if source_key_set != tracking_key_set:
        missing = sorted(tracking_key_set - source_key_set)
        extra = sorted(source_key_set - tracking_key_set)
        raise ProviderSpatialTrackSourceError(
            "subject-position and tracking sources identify different keyed rowsets "
            f"(missing_position={missing[:5]!r}, extra_position={extra[:5]!r})."
        )
    source_rows_by_key = {
        int(key): row for row, key in enumerate(source_keys.tolist())
    }
    matched_position_rows = np.asarray(
        [source_rows_by_key[int(key)] for key in tracking_keys.tolist()],
        dtype=np.int64,
    )
    matched_position_rows.setflags(write=False)
    matched_frames = source_frames[matched_position_rows]
    if not np.array_equal(matched_frames, tracking_frames):
        raise ProviderSpatialTrackSourceError(
            "subject-position and tracking sources disagree on acquisition frame identity."
        )

    payload = tracking.manifest.get("payload")
    if not isinstance(payload, Mapping) or type(payload.get("unassigned_track_id")) is not int:
        raise ProviderSpatialTrackSourceError(
            "tracking source lacks one exact unassigned_track_id authority."
        )
    unassigned_track_id = int(payload["unassigned_track_id"])
    assigned_rows = np.flatnonzero(tracking_ids != unassigned_track_id).astype(
        np.int64,
        copy=False,
    )
    assigned_rows.setflags(write=False)
    if assigned_rows.size == 0:
        raise ProviderSpatialTrackSourceError(
            "tracking source contains no assigned rows after explicit unassigned filtering."
        )
    assigned_track_ids = tracking_ids[assigned_rows]
    assigned_frames = tracking_frames[assigned_rows]
    if np.unique(assigned_track_ids).size != 1:
        raise ProviderSpatialTrackSourceError(
            "single-subject profile requires exactly one assigned track identity."
        )
    if np.unique(assigned_frames).size != assigned_frames.size:
        raise ProviderSpatialTrackSourceError(
            "single-subject profile has more than one assigned sample per acquisition frame."
        )
    output_order = np.lexsort((assigned_frames, assigned_track_ids)).astype(
        np.int64,
        copy=False,
    )
    output_order.setflags(write=False)
    output_position_rows = matched_position_rows[assigned_rows][output_order]
    output_frames = assigned_frames[output_order]
    output_track_ids = assigned_track_ids[output_order]
    output_valid = source_valid[output_position_rows]
    output_positions = source_positions[output_position_rows]
    output_reason_tags = tuple(
        "ok" if bool(value) else "provider_invalid" for value in output_valid
    )
    samples = ProviderTrackSamples(
        track_sample_key=np.column_stack(
            (output_track_ids, output_frames)
        ).astype(np.int64, copy=False),
        acquisition_frame=output_frames,
        subject_identity=(authorities.subject_identity,) * output_frames.size,
        track_identity=tuple(
            f"track:{int(value)}" for value in output_track_ids.tolist()
        ),
        source_position_xy=output_positions,
        provider_present=np.ones(output_frames.size, dtype=bool),
        provider_valid=output_valid,
        provider_reason_code=output_reason_tags,
        recording_ids=(authorities.recording_id,) * output_frames.size,
        timeline_authority_ids=(authorities.timeline_authority_id,) * output_frames.size,
    )
    evidence = _build_source_binding(
        authorities=authorities,
        position=position,
        tracking=tracking,
        modality=modality,
        estimator_id=estimator_id,
        source_keys=source_keys,
        source_frames=source_frames,
        source_reasons=source_reasons,
        tracking_keys=tracking_keys,
        tracking_frames=tracking_frames,
        tracking_ids=tracking_ids,
        assigned_rows=assigned_rows,
        position_rows=matched_position_rows,
        output_order=output_order,
        provider_reason_tags=tuple(reason_tags),
        unassigned_track_id=unassigned_track_id,
    )
    return samples, evidence


def bind_provider_track_source(
    position_source: SubjectPositionSourceHandle,
    tracking_source: TrackingSourceHandle,
    *,
    authorities: ProviderTrackSourceAuthorities,
) -> tuple[ProviderTrackSamples, ProviderTrackSourceEvidence]:
    """Alias emphasizing that this function binds, rather than discovers, sources."""

    return build_provider_track_source(
        position_source,
        tracking_source,
        authorities=authorities,
    )


__all__ = [
    "PROVIDER_TRACK_SOURCE_SCHEMA_ID",
    "PROVIDER_TRACK_SOURCE_SCHEMA_VERSION",
    "TRACK_SAMPLE_POLICY_ID",
    "ProviderSpatialTrackSourceError",
    "ProviderTrackIdentityAuthorities",
    "ProviderTrackSourceAuthorities",
    "ProviderTrackSourceEvidence",
    "bind_provider_track_source",
    "build_provider_track_source",
]
