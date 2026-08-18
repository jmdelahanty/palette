"""Publish one exact provider spatial trajectory as an immutable canary.

This materializer is deliberately a narrow storage boundary.  It accepts one
already prepared :class:`ProviderSpatialTrajectory`; it does not resolve a
provider, consult a selector, fill missing rows, or change any production
pointer.  The result is a selector-ineligible child under
``analysis/provider_spatial_trajectory_runs``.

The selected-frame membership is stored independently of source rows.  That
is important: a selected frame with no provider row remains part of the exact
denominator instead of disappearing during a row join.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import re
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import zarr

from fisheye.analysis.provider_spatial_trajectory import (
    ProviderSpatialTrajectory,
    SelectedFrameMembership,
    SourceCameraToArenaMMTransform,
    TrajectoryAuthorityIdentities,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


PARENT_PATH = "analysis/provider_spatial_trajectory_runs"
MATERIALIZATION_SCHEMA_ID = "palette.provider_spatial_trajectory_materialization"
MATERIALIZATION_SCHEMA_VERSION = 1
PUBLISH_SCHEMA_ID = "palette.provider_spatial_trajectory_publish"
RUN_SCHEMA_ID = "palette.provider_spatial_trajectory_run"
RUN_SCHEMA_VERSION = 1
ARRAY_MANIFEST_SCHEMA_ID = "palette.provider_spatial_trajectory_array_manifest"
ARRAY_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_ATTR = "provider_spatial_trajectory_manifest"
RUN_MANIFEST_SHA256_ATTR = "provider_spatial_trajectory_manifest_sha256"
ARRAY_MANIFEST_ATTR = "provider_spatial_trajectory_array_manifest"
ARRAY_MANIFEST_SHA256_ATTR = "provider_spatial_trajectory_array_manifest_sha256"
POLICY_ID = "exact_provider_trajectory_selector_ineligible_v1"
RETRY_POLICY_ID = "immutable_named_run_no_overwrite_v1"
STRING_ENCODING = "utf8_uint8_buffer_int64_offsets_v1"

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SELECTOR_ALIASES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_provider",
        "latest_any",
        "latest_materialized",
        "latest_composite",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
        "active_run",
        "active",
        "current_run",
        "current",
        "default_run",
        "default",
        "selected_run",
        "selected",
        "publication_generation",
        "publication_policy",
    }
)
_SELECTOR_NAME_PREFIXES = (
    "latest_",
    "authoritative_",
    "active_",
    "current_",
    "default_",
    "selected_",
    "publication_",
)
_SELECTOR_ATTRS = _SELECTOR_ALIASES
_CONTROLLED_REASON_CODES = frozenset(
    {
        "ok",
        "not_in_selection",
        "provider_missing",
        "provider_invalid",
        "source_position_nonfinite",
        "source_position_out_of_extent",
        "transform_invalid",
        "out_of_grid",
    }
)
_PRIMARY_REASON_CODES = frozenset(_CONTROLLED_REASON_CODES - {"ok", "not_in_selection"})


class ProviderSpatialTrajectoryMaterializationError(ValueError):
    """Raised when a trajectory cannot be published without guessing."""


def _selector_snapshot(parent: Any) -> dict[str, Any]:
    attrs = getattr(parent, "attrs", {})
    return {
        name: copy.deepcopy(json_attr_safe(attrs[name]))
        for name in _SELECTOR_ATTRS
        if name in attrs
    }


def _safe_run_name(value: object) -> str:
    if type(value) is not str or _RUN_NAME_RE.fullmatch(value) is None:
        raise ProviderSpatialTrajectoryMaterializationError(
            "run_name must be one explicit immutable path-safe name."
        )
    if _selector_like_name(value):
        raise ProviderSpatialTrajectoryMaterializationError(
            "run_name cannot be a selector alias or selector-like prefix."
        )
    return value


def _selector_like_name(value: object) -> bool:
    if type(value) is not str:
        return False
    lowered = value.lower()
    return lowered in _SELECTOR_ALIASES or lowered.startswith(_SELECTOR_NAME_PREFIXES)


def _require_provenance_parent(root: Any) -> Any:
    parent = require_runs_parent(
        root.require_group("analysis"),
        "provider_spatial_trajectory_runs",
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    if parent.attrs.get(COMPLETION_EPOCH_ATTR) != COMPLETION_EPOCH_REQUIRE_PROVENANCE:
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory parent must require completion provenance"
        )
    return parent


def _json_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ProviderSpatialTrajectoryMaterializationError(
            f"{label} must be one nonempty mapping."
        )
    try:
        result = json_attr_safe(dict(value))
    except Exception as exc:  # pragma: no cover - defensive serialization gate
        raise ProviderSpatialTrajectoryMaterializationError(
            f"{label} is not JSON-safe."
        ) from exc
    if not isinstance(result, dict):
        raise ProviderSpatialTrajectoryMaterializationError(
            f"{label} must serialize to one JSON object."
        )
    return result


def _encode_utf8(
    values: Sequence[str],
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode text as portable offsets plus one stable uint8 payload."""

    offsets = [0]
    payload = bytearray()
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{label}[{index}] is not a nonempty NUL-free string."
            )
        try:
            item = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{label}[{index}] is not valid UTF-8."
            ) from exc
        payload.extend(item)
        offsets.append(len(payload))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.frombuffer(bytes(payload), dtype=np.uint8).copy(),
    )


def _validate_offsets(
    offsets: np.ndarray,
    *,
    item_count: int,
    outer_count: int,
    label: str,
    allow_empty: bool,
) -> None:
    if offsets.dtype != np.dtype(np.int64) or offsets.shape != (outer_count + 1,):
        raise ProviderSpatialTrajectoryMaterializationError(
            f"{label} must be exactly int64[{outer_count + 1}]."
        )
    if offsets[0] != 0 or offsets[-1] != item_count or np.any(np.diff(offsets) < 0):
        raise ProviderSpatialTrajectoryMaterializationError(
            f"{label} does not delimit the complete payload."
        )
    if not allow_empty and np.any(np.diff(offsets) == 0):
        raise ProviderSpatialTrajectoryMaterializationError(
            f"{label} contains an empty required item."
        )


def _decode_utf8(
    payload: np.ndarray,
    offsets: np.ndarray,
    *,
    expected_count: int,
    label: str,
) -> tuple[str, ...]:
    if payload.dtype != np.dtype(np.uint8) or payload.ndim != 1:
        raise ProviderSpatialTrajectoryMaterializationError(
            f"{label} payload must be a one-dimensional uint8 array."
        )
    _validate_offsets(
        offsets,
        item_count=int(payload.size),
        outer_count=expected_count,
        label=f"{label} offsets",
        allow_empty=False,
    )
    result: list[str] = []
    raw = payload.tobytes(order="C")
    for index in range(expected_count):
        try:
            decoded = raw[int(offsets[index]) : int(offsets[index + 1])].decode(
                "utf-8"
            )
        except UnicodeDecodeError as exc:
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{label}[{index}] is not valid UTF-8."
            ) from exc
        if not decoded or "\x00" in decoded:
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{label}[{index}] is empty or contains NUL."
            )
        result.append(decoded)
    return tuple(result)


def _flatten_text_rows(
    rows: Sequence[Sequence[str]],
    *,
    label: str,
    allow_empty: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    flattened: list[str] = []
    for row_index, row in enumerate(rows):
        values = tuple(row)
        if not allow_empty and not values:
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{label}[{row_index}] must not be empty."
            )
        if len(set(values)) != len(values):
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{label}[{row_index}] contains duplicate identities."
            )
        for value in values:
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ProviderSpatialTrajectoryMaterializationError(
                    f"{label}[{row_index}] contains an invalid identity."
                )
        flattened.extend(values)
        offsets.append(len(flattened))
    text_offsets, payload = _encode_utf8(flattened, label=f"{label}.flat")
    return np.asarray(offsets, dtype=np.int64), text_offsets, payload


def _readonly(values: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _iter_array_paths(group: Any, prefix: str = ""):
    for name, array in sorted(group.arrays(), key=lambda item: str(item[0])):
        yield f"{prefix}/{name}" if prefix else str(name), array
    for name, child in sorted(group.groups(), key=lambda item: str(item[0])):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_array_paths(child, child_prefix)


def _trajectory_digest(trajectory: ProviderSpatialTrajectory) -> str:
    """Digest the complete in-memory object, including arrays omitted by as_record."""

    arrays = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": array_values_sha256(value),
        }
        for name, value in (
            ("source_row_index", trajectory.source_row_index),
            ("track_sample_key", trajectory.track_sample_key),
            ("acquisition_frame", trajectory.acquisition_frame),
            ("source_position_xy", trajectory.source_position_xy),
            ("arena_position_xy", trajectory.arena_position_xy),
            ("provider_present", trajectory.provider_present),
            ("provider_valid", trajectory.provider_valid),
            ("source_position_valid", trajectory.source_position_valid),
            ("source_extent_valid", trajectory.source_extent_valid),
            ("in_selection", trajectory.in_selection),
            ("transform_valid", trajectory.transform_valid),
            ("in_grid", trajectory.in_grid),
        )
    }
    payload = {
        "record": trajectory.as_record(),
        "subject_identity": list(trajectory.subject_identity),
        "track_identity": list(trajectory.track_identity),
        "selection_membership_key": [
            list(row) for row in trajectory.selection_membership_key
        ],
        "selection_occurrence_id": [
            list(row) for row in trajectory.selection_occurrence_id
        ],
        "selection_role": [list(row) for row in trajectory.selection_role],
        "reason_codes": [list(row) for row in trajectory.reason_codes],
        "reason_counts": dict(trajectory.reason_counts),
        "selected_reason_counts": dict(trajectory.selected_reason_counts),
        "arrays": arrays,
    }
    return canonical_json_sha256(payload)


def _expected_counts(trajectory: ProviderSpatialTrajectory) -> dict[str, int]:
    frames = np.asarray(trajectory.acquisition_frame, dtype=np.int64)
    selected_frames = np.asarray(trajectory.selection.acquisition_frames, dtype=np.int64)
    selected = np.asarray(trajectory.in_selection, dtype=bool)
    present = np.asarray(trajectory.provider_present, dtype=bool)
    valid = np.asarray(trajectory.provider_valid, dtype=bool)
    source_position_valid = np.asarray(trajectory.source_position_valid, dtype=bool)
    source_extent_valid = np.asarray(trajectory.source_extent_valid, dtype=bool)
    transform_valid = np.asarray(trajectory.transform_valid, dtype=bool)
    in_grid = np.asarray(trajectory.in_grid, dtype=bool)
    finite = np.all(np.isfinite(trajectory.source_position_xy), axis=1)
    source_frames = set(int(value) for value in frames.tolist())

    def count(values: np.ndarray) -> int:
        return int(np.count_nonzero(values))

    return {
        "expected_selected_frames": int(selected_frames.size),
        "source_rows": int(frames.size),
        "selected_source_rows": count(selected),
        "missing_selected_frames": sum(
            int(value) not in source_frames for value in selected_frames.tolist()
        ),
        "provider_present_rows": count(present),
        "provider_valid_rows": count(valid),
        "valid_position_rows": count(source_position_valid),
        "source_extent_valid_rows": count(source_extent_valid),
        "transform_valid_rows": count(transform_valid),
        "in_grid_rows": count(in_grid),
        "missing_provider_rows": count(~present),
        "invalid_provider_rows": count(present & ~valid),
        "nonfinite_position_rows": count(present & valid & ~finite),
        "source_position_out_of_extent_rows": count(
            source_position_valid & ~source_extent_valid
        ),
        "transform_invalid_rows": count(source_position_valid & ~transform_valid),
        "out_of_grid_rows": count(transform_valid & ~in_grid),
        "selected_provider_present_rows": count(selected & present),
        "selected_provider_valid_rows": count(selected & valid),
        "selected_valid_position_rows": count(selected & source_position_valid),
        "selected_source_extent_valid_rows": count(selected & source_extent_valid),
        "selected_transform_valid_rows": count(selected & transform_valid),
        "selected_in_grid_rows": count(selected & in_grid),
        "selected_missing_provider_rows": count(selected & ~present),
        "selected_invalid_provider_rows": count(selected & present & ~valid),
        "selected_nonfinite_position_rows": count(
            selected & present & valid & ~finite
        ),
        "selected_source_position_out_of_extent_rows": count(
            selected & source_position_valid & ~source_extent_valid
        ),
        "selected_transform_invalid_rows": count(
            selected & source_position_valid & ~transform_valid
        ),
        "selected_out_of_grid_rows": count(selected & transform_valid & ~in_grid),
    }


def _primary_reason(row: Sequence[str]) -> str:
    for reason in row:
        if reason in _PRIMARY_REASON_CODES:
            return reason
    return "ok"


def _expected_reason_counts(
    trajectory: ProviderSpatialTrajectory,
) -> tuple[dict[str, int], dict[str, int]]:
    all_counts = {reason: 0 for reason in sorted(_CONTROLLED_REASON_CODES)}
    selected_counts = {reason: 0 for reason in sorted(_CONTROLLED_REASON_CODES)}
    for index, row in enumerate(trajectory.reason_codes):
        primary = _primary_reason(row)
        all_counts[primary] += 1
        if bool(trajectory.in_selection[index]):
            selected_counts[primary] += 1
    return all_counts, selected_counts


def _validate_trajectory(trajectory: object) -> ProviderSpatialTrajectory:
    if type(trajectory) is not ProviderSpatialTrajectory:
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory must be one exact ProviderSpatialTrajectory object."
        )
    value = trajectory
    if type(value.authorities) is not TrajectoryAuthorityIdentities:
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory.authorities is not the exact authority record."
        )
    if type(value.selection) is not SelectedFrameMembership:
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory.selection is not the exact selected-frame record."
        )
    if type(value.transform) is not SourceCameraToArenaMMTransform:
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory.transform is not the exact transform record."
        )
    selection_frames = np.asarray(value.selection.acquisition_frames)
    if (
        selection_frames.dtype != np.dtype(np.int64)
        or selection_frames.ndim != 1
        or np.unique(selection_frames).size != selection_frames.size
        or (
            selection_frames.size
            and not np.all(np.diff(selection_frames) > 0)
        )
    ):
        raise ProviderSpatialTrajectoryMaterializationError(
            "selected acquisition frames are malformed or reordered."
        )
    membership_vectors = (
        value.selection.membership_keys,
        value.selection.occurrence_ids,
        value.selection.roles,
    )
    if any(len(rows) != selection_frames.size for rows in membership_vectors):
        raise ProviderSpatialTrajectoryMaterializationError(
            "selected-frame membership vectors have mismatched cardinality."
        )
    flattened_membership_keys: list[str] = []
    for index, rows in enumerate(zip(*membership_vectors, strict=True)):
        keys, occurrences, roles = (tuple(row) for row in rows)
        if not keys or len(keys) != len(occurrences) or len(keys) != len(roles):
            raise ProviderSpatialTrajectoryMaterializationError(
                f"selected-frame memberships are malformed at row {index}."
            )
        if any(
            not isinstance(item, str) or not item
            for group in (keys, occurrences, roles)
            for item in group
        ):
            raise ProviderSpatialTrajectoryMaterializationError(
                f"selected-frame memberships contain an invalid identity at row {index}."
            )
        if len(set(keys)) != len(keys):
            raise ProviderSpatialTrajectoryMaterializationError(
                f"selected-frame membership keys are duplicated at row {index}."
            )
        flattened_membership_keys.extend(keys)
    if len(set(flattened_membership_keys)) != len(flattened_membership_keys):
        raise ProviderSpatialTrajectoryMaterializationError(
            "selected-frame membership keys are duplicated across rows."
        )
    matrix = np.asarray(value.transform.matrix)
    if (
        matrix.dtype != np.dtype(np.float64)
        or matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
    ):
        raise ProviderSpatialTrajectoryMaterializationError(
            "transform matrix is malformed."
        )
    try:
        n_rows = int(value.track_sample_key.shape[0])
    except Exception as exc:
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory track_sample_key is malformed."
        ) from exc
    if value.track_sample_key.dtype != np.dtype(np.int64) or value.track_sample_key.shape != (
        n_rows,
        2,
    ):
        raise ProviderSpatialTrajectoryMaterializationError(
            "track_sample_key must be exactly int64[N, 2]."
        )
    if np.any(value.track_sample_key < 0):
        raise ProviderSpatialTrajectoryMaterializationError(
            "track_sample_key cannot contain negative values."
        )
    canonical_order = np.lexsort(
        (value.track_sample_key[:, 1], value.track_sample_key[:, 0])
    ).tolist()
    if canonical_order != list(range(n_rows)):
        raise ProviderSpatialTrajectoryMaterializationError(
            "track_sample_key rows are not in canonical order."
        )
    if np.unique(value.track_sample_key, axis=0).shape[0] != n_rows:
        raise ProviderSpatialTrajectoryMaterializationError(
            "track_sample_key contains duplicate rows."
        )

    one_d = (
        ("source_row_index", value.source_row_index, np.dtype(np.int64)),
        ("acquisition_frame", value.acquisition_frame, np.dtype(np.int64)),
        ("provider_present", value.provider_present, np.dtype(bool)),
        ("provider_valid", value.provider_valid, np.dtype(bool)),
        ("source_position_valid", value.source_position_valid, np.dtype(bool)),
        ("source_extent_valid", value.source_extent_valid, np.dtype(bool)),
        ("in_selection", value.in_selection, np.dtype(bool)),
        ("transform_valid", value.transform_valid, np.dtype(bool)),
        ("in_grid", value.in_grid, np.dtype(bool)),
    )
    for name, array, dtype in one_d:
        if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != (n_rows,):
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{name} must be exactly {dtype}[N]."
            )
    if not np.array_equal(value.source_row_index, np.arange(n_rows, dtype=np.int64)):
        raise ProviderSpatialTrajectoryMaterializationError(
            "source_row_index must be dense and canonical."
        )
    if not np.array_equal(value.acquisition_frame, value.track_sample_key[:, 1]):
        raise ProviderSpatialTrajectoryMaterializationError(
            "acquisition_frame disagrees with track_sample_key."
        )
    for name, rows in (
        ("subject_identity", value.subject_identity),
        ("track_identity", value.track_identity),
    ):
        if len(rows) != n_rows or any(not isinstance(item, str) or not item for item in rows):
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{name} has malformed cardinality or identity values."
            )
    for name, array in (
        ("source_position_xy", value.source_position_xy),
        ("arena_position_xy", value.arena_position_xy),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.dtype(np.float64)
            or array.shape != (n_rows, 2)
        ):
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{name} must be exactly float64[N, 2]."
            )
    if np.any(value.provider_valid & ~value.provider_present):
        raise ProviderSpatialTrajectoryMaterializationError(
            "provider_valid cannot be true when provider_present is false."
        )
    finite = np.all(np.isfinite(value.source_position_xy), axis=1)
    if not np.array_equal(
        value.source_position_valid,
        value.provider_present & value.provider_valid & finite,
    ):
        raise ProviderSpatialTrajectoryMaterializationError(
            "source_position_valid is not independently consistent with source data."
        )
    if value.transform.source_camera_extent_px is None:
        raise ProviderSpatialTrajectoryMaterializationError(
            "source camera extent is required for trajectory materialization."
        )
    source_xmin, source_xmax, source_ymin, source_ymax = (
        value.transform.source_camera_extent_px
    )
    expected_source_extent_valid = finite & (
        (value.source_position_xy[:, 0] >= source_xmin)
        & (value.source_position_xy[:, 0] < source_xmax)
        & (value.source_position_xy[:, 1] >= source_ymin)
        & (value.source_position_xy[:, 1] < source_ymax)
    )
    if not np.array_equal(value.source_extent_valid, expected_source_extent_valid):
        raise ProviderSpatialTrajectoryMaterializationError(
            "source_extent_valid is not consistent with the authoritative source extent."
        )
    if np.any(value.in_grid & ~value.transform_valid):
        raise ProviderSpatialTrajectoryMaterializationError(
            "in_grid cannot be true when transform_valid is false."
        )
    for name, rows in (
        ("selection_membership_key", value.selection_membership_key),
        ("selection_occurrence_id", value.selection_occurrence_id),
        ("selection_role", value.selection_role),
    ):
        if len(rows) != n_rows:
            raise ProviderSpatialTrajectoryMaterializationError(
                f"{name} has malformed row cardinality."
            )
        for row_index, row in enumerate(rows):
            if len(set(row)) != len(row) or any(
                not isinstance(item, str) or not item for item in row
            ):
                raise ProviderSpatialTrajectoryMaterializationError(
                    f"{name}[{row_index}] contains malformed membership identities."
                )
            if bool(value.in_selection[row_index]) != bool(row):
                raise ProviderSpatialTrajectoryMaterializationError(
                    f"{name}[{row_index}] disagrees with in_selection."
                )
    if not (
        len(value.reason_codes) == n_rows
        and all(
            row and all(reason in _CONTROLLED_REASON_CODES for reason in row)
            for row in value.reason_codes
        )
    ):
        raise ProviderSpatialTrajectoryMaterializationError(
            "reason_codes contain unsupported or empty evidence."
        )
    expected_counts = _expected_counts(value)
    if value.counts.as_record() != expected_counts:
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory coverage counts are stale or malformed."
        )
    expected_reasons, expected_selected_reasons = _expected_reason_counts(value)
    if dict(value.reason_counts) != expected_reasons or dict(value.selected_reason_counts) != expected_selected_reasons:
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory reason counts are stale or malformed."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(value.source_rows_sha256)):
        raise ProviderSpatialTrajectoryMaterializationError(
            "source_rows_sha256 is not a canonical SHA-256 digest."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(value.trajectory_sha256)):
        raise ProviderSpatialTrajectoryMaterializationError(
            "trajectory_sha256 is not a canonical SHA-256 digest."
        )
    return value


def _build_arrays(
    trajectory: ProviderSpatialTrajectory,
) -> dict[str, np.ndarray]:
    subject_offsets, subject_utf8 = _encode_utf8(
        trajectory.subject_identity,
        label="subject_identity",
    )
    track_offsets, track_utf8 = _encode_utf8(
        trajectory.track_identity,
        label="track_identity",
    )
    (
        source_membership_offsets,
        source_membership_key_offsets,
        source_membership_key_utf8,
    ) = _flatten_text_rows(
        trajectory.selection_membership_key,
        label="selection_membership_key",
        allow_empty=True,
    )
    (
        source_occurrence_outer_offsets,
        source_occurrence_id_offsets,
        source_occurrence_id_utf8,
    ) = _flatten_text_rows(
        trajectory.selection_occurrence_id,
        label="selection_occurrence_id",
        allow_empty=True,
    )
    source_role_outer_offsets, source_role_offsets, source_role_utf8 = _flatten_text_rows(
        trajectory.selection_role,
        label="selection_role",
        allow_empty=True,
    )
    if not (
        np.array_equal(source_membership_offsets, source_occurrence_outer_offsets)
        and np.array_equal(source_membership_offsets, source_role_outer_offsets)
    ):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Source-row membership keys, occurrences, and roles are not aligned."
        )
    (
        selected_membership_offsets,
        selected_membership_key_offsets,
        selected_membership_key_utf8,
    ) = _flatten_text_rows(
        trajectory.selection.membership_keys,
        label="selected.membership_keys",
        allow_empty=False,
    )
    (
        selected_occurrence_outer_offsets,
        selected_occurrence_id_offsets,
        selected_occurrence_id_utf8,
    ) = _flatten_text_rows(
        trajectory.selection.occurrence_ids,
        label="selected.occurrence_ids",
        allow_empty=False,
    )
    selected_role_outer_offsets, selected_role_offsets, selected_role_utf8 = _flatten_text_rows(
        trajectory.selection.roles,
        label="selected.roles",
        allow_empty=False,
    )
    if not (
        np.array_equal(selected_membership_offsets, selected_occurrence_outer_offsets)
        and np.array_equal(selected_membership_offsets, selected_role_outer_offsets)
    ):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Selected membership keys, occurrences, and roles are not aligned."
        )
    reason_offsets, reason_code_offsets, reason_code_utf8 = _flatten_text_rows(
        trajectory.reason_codes,
        label="reason_codes",
        allow_empty=False,
    )
    arrays: dict[str, np.ndarray] = {
        "source_row_index": _readonly(trajectory.source_row_index, dtype=np.int64),
        "track_sample_key": _readonly(trajectory.track_sample_key, dtype=np.int64),
        "acquisition_frame": _readonly(trajectory.acquisition_frame, dtype=np.int64),
        "subject_identity_offsets": subject_offsets,
        "subject_identity_utf8": subject_utf8,
        "track_identity_offsets": track_offsets,
        "track_identity_utf8": track_utf8,
        "source_position_xy": _readonly(trajectory.source_position_xy, dtype=np.float64),
        "arena_position_xy": _readonly(trajectory.arena_position_xy, dtype=np.float64),
        "provider_present": _readonly(trajectory.provider_present, dtype=bool),
        "provider_valid": _readonly(trajectory.provider_valid, dtype=bool),
        "source_position_valid": _readonly(trajectory.source_position_valid, dtype=bool),
        "source_extent_valid": _readonly(trajectory.source_extent_valid, dtype=bool),
        "in_selection": _readonly(trajectory.in_selection, dtype=bool),
        "transform_valid": _readonly(trajectory.transform_valid, dtype=bool),
        "in_grid": _readonly(trajectory.in_grid, dtype=bool),
        "selection/source_row_membership_offsets": source_membership_offsets,
        "selection/source_row_membership_key_offsets": source_membership_key_offsets,
        "selection/source_row_membership_key_utf8": source_membership_key_utf8,
        "selection/source_row_occurrence_id_offsets": source_occurrence_id_offsets,
        "selection/source_row_occurrence_id_utf8": source_occurrence_id_utf8,
        "selection/source_row_role_offsets": source_role_offsets,
        "selection/source_row_role_utf8": source_role_utf8,
        "selection/acquisition_frame": _readonly(
            trajectory.selection.acquisition_frames, dtype=np.int64
        ),
        "selection/membership_offsets": selected_membership_offsets,
        "selection/membership_key_offsets": selected_membership_key_offsets,
        "selection/membership_key_utf8": selected_membership_key_utf8,
        "selection/occurrence_id_offsets": selected_occurrence_id_offsets,
        "selection/occurrence_id_utf8": selected_occurrence_id_utf8,
        "selection/role_offsets": selected_role_offsets,
        "selection/role_utf8": selected_role_utf8,
        "reasons/offsets": reason_offsets,
        "reasons/code_offsets": reason_code_offsets,
        "reasons/code_utf8": reason_code_utf8,
    }
    return arrays


def _array_declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    def axes_and_units(path: str, ndim: int) -> tuple[list[str], list[str], str]:
        if path == "track_sample_key":
            return ["track_sample", "key_component"], ["1", "1"], "canonical_track_sample_key"
        if path.endswith("position_xy"):
            units = ["px", "px"] if path == "source_position_xy" else ["mm", "mm"]
            return ["track_sample", "xy"], units, (
                "source_camera_position" if path == "source_position_xy" else "arena_position"
            )
        if path in {"subject_identity_offsets", "track_identity_offsets"}:
            identity = path.removesuffix("_offsets")
            return ["track_sample_plus_one"], ["byte_offset"], f"{identity}_offsets"
        if path in {"subject_identity_utf8", "track_identity_utf8"}:
            identity = path.removesuffix("_utf8")
            return [f"{identity}_utf8_byte"], ["byte"], f"{identity}_utf8_payload"
        if path.startswith("selection/"):
            if path == "selection/acquisition_frame":
                return ["selected_frame"], ["acquisition_frame"], "selected_frame_denominator"
            if path in {
                "selection/membership_offsets",
                "selection/source_row_membership_offsets",
            }:
                axis = (
                    "selected_frame_plus_one"
                    if path == "selection/membership_offsets"
                    else "track_sample_plus_one"
                )
                return [axis], ["item_offset"], "ragged_membership_outer_offsets"
            if path.endswith("_offsets"):
                axis = (
                    "source_membership_plus_one"
                    if "source_row" in path
                    else "membership_plus_one"
                )
                return [axis], ["byte_offset"], "ragged_text_offsets"
            if path.endswith("_utf8"):
                axis = (
                    "source_membership_utf8_byte"
                    if "source_row" in path
                    else "membership_utf8_byte"
                )
                return [axis], ["byte"], "ragged_text_utf8_payload"
        if path.startswith("reasons/"):
            if path == "reasons/offsets":
                return ["track_sample_plus_one"], ["item_offset"], "reason_outer_offsets"
            if path == "reasons/code_offsets":
                return ["reason_evidence_plus_one"], ["byte_offset"], "reason_text_offsets"
            if path == "reasons/code_utf8":
                return ["reason_evidence_utf8_byte"], ["byte"], "reason_text_utf8_payload"
        state_roles = {
            "provider_present": "provider_presence_state",
            "provider_valid": "provider_validity_state",
            "source_position_valid": "source_position_validity_state",
            "source_extent_valid": "source_extent_validity_state",
            "in_selection": "selection_membership_state",
            "transform_valid": "transform_validity_state",
            "in_grid": "grid_membership_state",
        }
        if path in state_roles:
            return ["track_sample"], ["1"], state_roles[path]
        if path in {"source_row_index", "acquisition_frame"}:
            return ["track_sample"], ["1"], path
        raise ProviderSpatialTrajectoryMaterializationError(
            f"No logical array contract exists for {path!r}."
        )

    declarations: list[dict[str, Any]] = []
    for path, values in sorted(arrays.items()):
        axes, units, authority_role = axes_and_units(path, values.ndim)
        declarations.append(
            {
                "path": path,
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "axes": axes,
                "units": units,
                "authority_role": authority_role,
                "content_sha256": array_values_sha256(values),
            }
        )
    return declarations


@dataclass(frozen=True)
class ProviderSpatialTrajectoryMaterializationPlan:
    source_zarr: Path
    run_name: str
    run_path: str
    target_run_path: Path
    scratch_root: Path
    local_zarr: Path
    local_run_path: Path
    trajectory: ProviderSpatialTrajectory
    trajectory_digest: str
    arrays: Mapping[str, np.ndarray]
    array_manifest: Mapping[str, Any]
    run_manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    parent_selector_attrs: Mapping[str, Any]
    parent_exists: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "source_zarr": str(self.source_zarr),
            "run_name": self.run_name,
            "run_path": self.run_path,
            "target_run_path": str(self.target_run_path),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "trajectory_digest": self.trajectory_digest,
            "array_manifest_sha256": canonical_json_sha256(self.array_manifest),
            "run_manifest_sha256": canonical_json_sha256(self.run_manifest),
            "parent_selector_attrs": dict(self.parent_selector_attrs),
            "parent_exists": bool(self.parent_exists),
            "policy_id": POLICY_ID,
            "retry_policy": RETRY_POLICY_ID,
        }


def _build_manifests(
    trajectory: ProviderSpatialTrajectory,
    arrays: Mapping[str, np.ndarray],
    *,
    run_name: str,
    run_path: str,
    software_record: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    declarations = _array_declarations(arrays)
    selection_frame_array = arrays["selection/acquisition_frame"]
    array_manifest = {
        "schema_id": ARRAY_MANIFEST_SCHEMA_ID,
        "schema_version": ARRAY_MANIFEST_SCHEMA_VERSION,
        "array_digest_algorithm": "numpy_dtype_shape_c_order_bytes_sha256_v1",
        "string_encoding": STRING_ENCODING,
        "arrays": declarations,
        "selected_frame_denominator": {
            "array_path": "selection/acquisition_frame",
            "count": int(selection_frame_array.size),
            "content_sha256": array_values_sha256(selection_frame_array),
            "selection_sha256": trajectory.selection.sha256,
        },
    }
    array_manifest_sha256 = canonical_json_sha256(array_manifest)
    selected_denominator = dict(array_manifest["selected_frame_denominator"])
    run_manifest = {
        "schema_id": RUN_SCHEMA_ID,
        "schema_version": RUN_SCHEMA_VERSION,
        "run_name": run_name,
        "run_path": run_path,
        "row_axis": "track_samples",
        "policy_id": POLICY_ID,
        "stage_selector_eligible": False,
        "authorities": trajectory.authorities.as_record(),
        "authorities_sha256": canonical_json_sha256(trajectory.authorities.as_record()),
        "selection": trajectory.selection.as_record(),
        "selection_sha256": trajectory.selection.sha256,
        "transform": trajectory.transform.as_record(),
        "transform_sha256": trajectory.transform.sha256,
        "source_rows_sha256": trajectory.source_rows_sha256,
        "trajectory_sha256": trajectory.trajectory_sha256,
        "trajectory_object_digest": _trajectory_digest(trajectory),
        "counts": trajectory.counts.as_record(),
        "reason_counts": dict(trajectory.reason_counts),
        "selected_reason_counts": dict(trajectory.selected_reason_counts),
        "selected_frame_denominator": selected_denominator,
        "array_manifest_sha256": array_manifest_sha256,
        "software": dict(software_record),
        "provenance_sha256": canonical_json_sha256(provenance),
        "retry_policy": {
            "policy_id": RETRY_POLICY_ID,
            "existing_target": "fail_closed",
            "source_mutation_since_plan": "fail_closed",
            "retry_requires": "new_explicit_run_name",
        },
    }
    return array_manifest, run_manifest


def plan_provider_spatial_trajectory_run(
    analysis_zarr: str | Path,
    trajectory: ProviderSpatialTrajectory,
    *,
    run_name: str,
    scratch_root: str | Path,
    software_record: Mapping[str, Any] | None = None,
) -> ProviderSpatialTrajectoryMaterializationPlan:
    """Plan a publication without creating scratch or archive state."""

    value = _validate_trajectory(trajectory)
    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    name = _safe_run_name(run_name)
    scratch = Path(scratch_root).expanduser().resolve()
    try:
        scratch.relative_to(archive)
    except ValueError:
        pass
    else:
        raise ProviderSpatialTrajectoryMaterializationError(
            "scratch_root must not be inside the authoritative analysis Zarr."
        )
    run_path = f"{PARENT_PATH}/{name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        raise FileExistsError(f"Refusing existing immutable trajectory run: {target}")

    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    parent_path = archive.joinpath(*PARENT_PATH.split("/"))
    parent_exists = parent_path.is_dir()
    if parent_exists:
        parent = root[PARENT_PATH]
        parent_selectors = _selector_snapshot(parent)
    else:
        parent_selectors = {}

    software = _json_mapping(
        software_record
        or {
            "materializer": "provider_spatial_trajectory",
            "contract": MATERIALIZATION_SCHEMA_ID,
        },
        label="software_record",
    )
    provenance = build_writer_run_provenance(
        command="provider_spatial_trajectory_materializer",
        params={
            "run_name": name,
            "run_path": run_path,
            "policy_id": POLICY_ID,
            "retry_policy": RETRY_POLICY_ID,
        },
        input_run_ids={
            "recording_id": value.authorities.recording_id,
            "provider_id": value.authorities.provider_id,
            "source_id": value.authorities.source_id,
            "selection_sha256": value.selection.sha256,
            "transform_sha256": value.transform.sha256,
            "trajectory_sha256": value.trajectory_sha256,
        },
        cwd=Path(__file__).resolve().parents[4],
        include_system_context=False,
    )
    arrays = {name_: _readonly(array) for name_, array in _build_arrays(value).items()}
    array_manifest, run_manifest = _build_manifests(
        value,
        arrays,
        run_name=name,
        run_path=run_path,
        software_record=software,
        provenance=provenance,
    )
    local_zarr = scratch / f"provider_spatial_trajectory_{name}.zarr"
    return ProviderSpatialTrajectoryMaterializationPlan(
        source_zarr=archive,
        run_name=name,
        run_path=run_path,
        target_run_path=target,
        scratch_root=scratch,
        local_zarr=local_zarr,
        local_run_path=local_zarr.joinpath(*run_path.split("/")),
        trajectory=value,
        trajectory_digest=_trajectory_digest(value),
        arrays=arrays,
        array_manifest=array_manifest,
        run_manifest=run_manifest,
        provenance=provenance,
        parent_selector_attrs=parent_selectors,
        parent_exists=parent_exists,
    )


def _ensure_plan_current(plan: ProviderSpatialTrajectoryMaterializationPlan) -> None:
    if type(plan) is not ProviderSpatialTrajectoryMaterializationPlan:
        raise ProviderSpatialTrajectoryMaterializationError(
            "plan must be one exact ProviderSpatialTrajectoryMaterializationPlan."
        )
    _validate_trajectory(plan.trajectory)
    observed_digest = _trajectory_digest(plan.trajectory)
    if observed_digest != plan.trajectory_digest:
        raise ProviderSpatialTrajectoryMaterializationError(
            "ProviderSpatialTrajectory changed after planning."
        )
    if plan.target_run_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing immutable trajectory run: {plan.target_run_path}"
        )
    root = open_zarr_root(plan.source_zarr, mode="r", use_consolidated=False)
    parent_path = plan.source_zarr.joinpath(*PARENT_PATH.split("/"))
    if parent_path.is_dir() != plan.parent_exists:
        raise ProviderSpatialTrajectoryMaterializationError(
            "Trajectory parent existence changed after planning."
        )
    if plan.parent_exists:
        observed_selectors = _selector_snapshot(root[PARENT_PATH])
        if observed_selectors != dict(plan.parent_selector_attrs):
            raise ProviderSpatialTrajectoryMaterializationError(
                "Trajectory parent selectors changed after planning."
            )


def _write_local(plan: ProviderSpatialTrajectoryMaterializationPlan) -> None:
    if plan.local_zarr.exists():
        raise FileExistsError(f"Refusing existing local trajectory candidate: {plan.local_zarr}")
    plan.scratch_root.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(
        str(plan.local_zarr), mode="w-", zarr_format=3, use_consolidated=False
    )
    parent = root.require_group("analysis").require_group(
        "provider_spatial_trajectory_runs"
    )
    run = parent.create_group(plan.run_name)
    mark_run_started(run, run_name=plan.run_name, stage="provider_spatial_trajectory")
    for path, values in sorted(plan.arrays.items()):
        if values.dtype.hasobject:
            raise ProviderSpatialTrajectoryMaterializationError(
                f"Object array is forbidden: {path}."
            )
        if values.ndim == 0:
            chunks = ()
        elif values.ndim == 1:
            chunks = (max(1, min(int(values.shape[0]), 16_384)),)
        else:
            chunks = (max(1, min(int(values.shape[0]), 16_384)), *values.shape[1:])
        run.create_array(path, data=values, chunks=chunks)

    run.attrs.update(
        {
            "schema_id": RUN_SCHEMA_ID,
            "schema_version": RUN_SCHEMA_VERSION,
            "materialization_schema_id": MATERIALIZATION_SCHEMA_ID,
            "materialization_schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "policy_id": POLICY_ID,
            "stage_selector_eligible": False,
            "storage_candidate_profile_promoted": False,
            "row_axis": "track_samples",
            RUN_NAME_ATTR: plan.run_name,
            RUN_MANIFEST_ATTR: json_attr_safe(dict(plan.run_manifest)),
            RUN_MANIFEST_SHA256_ATTR: canonical_json_sha256(plan.run_manifest),
            ARRAY_MANIFEST_ATTR: json_attr_safe(dict(plan.array_manifest)),
            ARRAY_MANIFEST_SHA256_ATTR: canonical_json_sha256(plan.array_manifest),
            "run_provenance": json_attr_safe(dict(plan.provenance)),
            "completion_provenance_policy": "run_provenance_v1_required",
            "retry_policy": json_attr_safe(
                dict(plan.run_manifest["retry_policy"])
            ),
        }
    )
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=plan.run_name,
        run_provenance=plan.provenance,
    )


def _validate_run(
    path: Path,
    *,
    plan: ProviderSpatialTrajectoryMaterializationPlan,
) -> dict[str, Any]:
    run = open_zarr_root(path, mode="r", use_consolidated=False)
    if run.attrs.get(RUN_MANIFEST_ATTR) != dict(plan.run_manifest):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Trajectory run manifest is missing or stale."
        )
    if run.attrs.get(RUN_MANIFEST_SHA256_ATTR) != canonical_json_sha256(plan.run_manifest):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Trajectory run manifest digest is stale."
        )
    if run.attrs.get(ARRAY_MANIFEST_ATTR) != dict(plan.array_manifest):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Trajectory array manifest is missing or stale."
        )
    if run.attrs.get(ARRAY_MANIFEST_SHA256_ATTR) != canonical_json_sha256(plan.array_manifest):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Trajectory array manifest digest is stale."
        )
    if run.attrs.get("stage_selector_eligible") is not False:
        raise ProviderSpatialTrajectoryMaterializationError(
            "Trajectory candidate became selector eligible."
        )
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ProviderSpatialTrajectoryMaterializationError(
            "Trajectory candidate is not complete."
        )
    if any(_selector_like_name(name) for name in run.attrs):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Trajectory candidate contains selector attributes."
        )
    declarations = plan.array_manifest["arrays"]
    observed_paths = {path_ for path_, _ in _iter_array_paths(run)}
    expected_paths = {str(item["path"]) for item in declarations}
    if observed_paths != expected_paths:
        raise ProviderSpatialTrajectoryMaterializationError(
            f"Trajectory array paths differ: observed={sorted(observed_paths)!r}, "
            f"expected={sorted(expected_paths)!r}."
        )
    for declaration in declarations:
        values = np.asarray(run[declaration["path"]][:])
        if values.dtype.kind not in "iufb":
            raise ProviderSpatialTrajectoryMaterializationError(
                f"Trajectory array uses a non-portable dtype: {declaration['path']}."
            )
        observed = {
            "path": declaration["path"],
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "axes": declaration["axes"],
            "units": declaration["units"],
            "authority_role": declaration["authority_role"],
            "content_sha256": array_values_sha256(values),
        }
        if observed != declaration:
            raise ProviderSpatialTrajectoryMaterializationError(
                f"Trajectory array drifted: {declaration['path']}."
            )
    key = np.asarray(run["track_sample_key"][:])
    n_rows = int(key.shape[0])
    if key.dtype != np.dtype(np.int64) or key.shape != (n_rows, 2):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Published track_sample_key is not int64[N, 2]."
        )
    source_index = np.asarray(run["source_row_index"][:])
    frame = np.asarray(run["acquisition_frame"][:])
    if not np.array_equal(source_index, np.arange(n_rows, dtype=np.int64)):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Published source_row_index is not canonical."
        )
    if not np.array_equal(key[:, 1], frame):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Published acquisition frame disagrees with canonical key."
        )
    for name in ("subject_identity", "track_identity"):
        _decode_utf8(
            np.asarray(run[f"{name}_utf8"][:]),
            np.asarray(run[f"{name}_offsets"][:]),
            expected_count=n_rows,
            label=name,
        )

    source_outer = np.asarray(run["selection/source_row_membership_offsets"][:])
    source_text_offsets = {
        "membership_key": np.asarray(
            run["selection/source_row_membership_key_offsets"][:]
        ),
        "occurrence_id": np.asarray(
            run["selection/source_row_occurrence_id_offsets"][:]
        ),
        "role": np.asarray(run["selection/source_row_role_offsets"][:]),
    }
    source_item_counts = {
        name: int(offsets.size - 1)
        for name, offsets in source_text_offsets.items()
    }
    if len(set(source_item_counts.values())) != 1:
        raise ProviderSpatialTrajectoryMaterializationError(
            "Source-row membership text columns have mismatched cardinality."
        )
    source_item_count = next(iter(source_item_counts.values()))
    _validate_offsets(
        source_outer,
        item_count=source_item_count,
        outer_count=n_rows,
        label="source-row membership outer offsets",
        allow_empty=True,
    )
    for name, offsets in source_text_offsets.items():
        _decode_utf8(
            np.asarray(run[f"selection/source_row_{name}_utf8"][:]),
            offsets,
            expected_count=source_item_count,
            label=f"source-row {name}",
        )

    selected_frames = np.asarray(run["selection/acquisition_frame"][:])
    selected_count = int(selected_frames.size)
    selected_offsets = np.asarray(run["selection/membership_offsets"][:])
    selected_text_offsets = {
        "membership_key": np.asarray(run["selection/membership_key_offsets"][:]),
        "occurrence_id": np.asarray(run["selection/occurrence_id_offsets"][:]),
        "role": np.asarray(run["selection/role_offsets"][:]),
    }
    selected_item_counts = {
        name: int(offsets.size - 1)
        for name, offsets in selected_text_offsets.items()
    }
    if len(set(selected_item_counts.values())) != 1:
        raise ProviderSpatialTrajectoryMaterializationError(
            "Selected-frame membership text columns have mismatched cardinality."
        )
    selected_item_count = next(iter(selected_item_counts.values()))
    _validate_offsets(
        selected_offsets,
        item_count=selected_item_count,
        outer_count=selected_count,
        label="selected-frame membership outer offsets",
        allow_empty=False,
    )
    for name, offsets in selected_text_offsets.items():
        _decode_utf8(
            np.asarray(run[f"selection/{name}_utf8"][:]),
            offsets,
            expected_count=selected_item_count,
            label=f"selected-frame {name}",
        )

    reason_outer = np.asarray(run["reasons/offsets"][:])
    reason_text_offsets = np.asarray(run["reasons/code_offsets"][:])
    reason_count = int(reason_text_offsets.size - 1)
    _validate_offsets(
        reason_outer,
        item_count=reason_count,
        outer_count=n_rows,
        label="reason outer offsets",
        allow_empty=False,
    )
    decoded_reasons = _decode_utf8(
        np.asarray(run["reasons/code_utf8"][:]),
        reason_text_offsets,
        expected_count=reason_count,
        label="reason code",
    )
    if any(reason not in _CONTROLLED_REASON_CODES for reason in decoded_reasons):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Published reason evidence contains an unsupported code."
        )

    denominator = plan.array_manifest["selected_frame_denominator"]
    if denominator["count"] != selected_count or denominator["content_sha256"] != array_values_sha256(selected_frames):
        raise ProviderSpatialTrajectoryMaterializationError(
            "Selected-frame denominator is not bound by the array manifest."
        )
    return {
        "valid": True,
        "run_path": plan.run_path,
        "row_count": n_rows,
        "selected_frame_count": selected_count,
        "run_manifest_sha256": canonical_json_sha256(plan.run_manifest),
        "array_manifest_sha256": canonical_json_sha256(plan.array_manifest),
    }


def publish_provider_spatial_trajectory_run(
    plan: ProviderSpatialTrajectoryMaterializationPlan,
    *,
    copy_backend: str = "python",
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Materialize and atomically publish a selector-ineligible trajectory."""

    _ensure_plan_current(plan)
    _write_local(plan)
    # The object is retained by the plan precisely so a concurrent or
    # post-planning mutation cannot be hidden by the copied local arrays.
    _ensure_plan_current(plan)
    _validate_run(plan.local_run_path, plan=plan)
    consolidate_metadata_capture_expected_warnings(plan.local_zarr)
    local_metadata = validate_direct_consolidated_subtree(
        plan.local_zarr,
        subtree_path=plan.run_path,
    ).to_json()
    _validate_run(plan.local_run_path, plan=plan)
    acceptance: dict[str, Any] = {"local_direct_consolidated": local_metadata}

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_run(path, plan=plan)

    def prepare(root: Any) -> tuple[Any]:
        return (_require_provenance_parent(root),)

    def complete(_root: Any, parent: Any, run: Any) -> None:
        run.attrs["stage_selector_eligible"] = False
        run.attrs["storage_candidate_profile_promoted"] = False
        mark_run_complete(
            run,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=plan.provenance,
        )

    def verify(root: Any) -> None:
        parent = root[PARENT_PATH]
        if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
            raise ProviderSpatialTrajectoryMaterializationError(
                "Trajectory publication changed parent selectors."
            )
        _validate_run(plan.target_run_path, plan=plan)

    def finalize(root: Any, parent: Any, run: Any) -> None:
        if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
            raise ProviderSpatialTrajectoryMaterializationError(
                "Trajectory finalization observed changed parent selectors."
            )
        if run.attrs.get("stage_selector_eligible") is not False:
            raise ProviderSpatialTrajectoryMaterializationError(
                "Trajectory finalization observed selector eligibility."
            )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        acceptance["published_direct_consolidated"] = validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=plan.run_path,
        ).to_json()
        consolidated_root = open_zarr_root(
            plan.source_zarr,
            mode="r",
            use_consolidated=True,
        )
        if plan.run_path not in consolidated_root:
            raise ProviderSpatialTrajectoryMaterializationError(
                "Consolidated metadata omitted the published trajectory run."
            )
        acceptance["published_validation"] = _validate_run(
            plan.target_run_path,
            plan=plan,
        )

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="provider-spatial-trajectory",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy=POLICY_ID,
            rollback_policy="retain_failed_tombstone_leave_parent_selectors_untouched",
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        payload_metadata={
            "policy_id": POLICY_ID,
            "selector_eligible": False,
            "run_manifest_sha256": canonical_json_sha256(plan.run_manifest),
            "array_manifest_sha256": canonical_json_sha256(plan.array_manifest),
            "selected_frame_denominator": plan.run_manifest[
                "selected_frame_denominator"
            ],
        },
        activate_run=finalize,
        repair_failed_publication_visibility=lambda _target: (
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        ),
        accept_persisted_activation_on_callback_error=False,
    )
    result = {
        "status": "complete",
        "run_path": plan.run_path,
        "selector_eligible": False,
        "policy_id": POLICY_ID,
        "run_manifest_sha256": canonical_json_sha256(plan.run_manifest),
        "array_manifest_sha256": canonical_json_sha256(plan.array_manifest),
        "validation": acceptance,
        "publication": publication,
    }
    if not keep_scratch and plan.local_zarr.exists():
        shutil.rmtree(plan.local_zarr)
    return json_attr_safe(result)


def materialize_provider_spatial_trajectory_run(
    analysis_zarr: str | Path,
    trajectory: ProviderSpatialTrajectory,
    *,
    run_name: str,
    scratch_root: str | Path,
    software_record: Mapping[str, Any] | None = None,
    apply: bool = True,
    copy_backend: str = "python",
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Plan, validate, and optionally publish one exact trajectory."""

    plan = plan_provider_spatial_trajectory_run(
        analysis_zarr,
        trajectory,
        run_name=run_name,
        scratch_root=scratch_root,
        software_record=software_record,
    )
    if not apply:
        return json_attr_safe({"status": "planned", "plan": plan.as_dict()})
    return publish_provider_spatial_trajectory_run(
        plan,
        copy_backend=copy_backend,
        keep_scratch=keep_scratch,
    )


# Explicit aliases make the storage boundary discoverable without introducing
# a second implementation or a selector-aware convenience path.
plan_provider_spatial_trajectory_materialization = plan_provider_spatial_trajectory_run
publish_provider_spatial_trajectory_materialization = publish_provider_spatial_trajectory_run
materialize_provider_spatial_trajectory = materialize_provider_spatial_trajectory_run


__all__ = [
    "ARRAY_MANIFEST_ATTR",
    "ARRAY_MANIFEST_SCHEMA_ID",
    "MATERIALIZATION_SCHEMA_ID",
    "PARENT_PATH",
    "POLICY_ID",
    "ProviderSpatialTrajectoryMaterializationError",
    "ProviderSpatialTrajectoryMaterializationPlan",
    "RUN_MANIFEST_ATTR",
    "RUN_SCHEMA_ID",
    "materialize_provider_spatial_trajectory",
    "materialize_provider_spatial_trajectory_run",
    "plan_provider_spatial_trajectory_materialization",
    "plan_provider_spatial_trajectory_run",
    "publish_provider_spatial_trajectory_materialization",
    "publish_provider_spatial_trajectory_run",
]
