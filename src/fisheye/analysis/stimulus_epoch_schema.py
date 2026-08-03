"""Exact logical and semantic contract for immutable stimulus-epoch v2 runs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import AnalysisAuthorityRole
from fisheye.shared.zarr.storage_intent import AccessPattern

from ._exact_tabular_run_schema import (
    ColumnSpec,
    MANIFEST_ATTRIBUTE,
    build_exact_array_declarations,
    build_exact_manifest,
    collect_run_arrays,
    validate_exact_manifest,
)


LEGACY_STIMULUS_EPOCH_SCHEMA_ID = "palette.stimulus_epoch_windows.v1"
LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION = 1
STIMULUS_EPOCH_RUN_SCHEMA_ID = "palette.stimulus_epoch_windows.v2"
STIMULUS_EPOCH_RUN_SCHEMA_VERSION = 2
STIMULUS_EPOCH_LAYOUT = "exact_columnar_v1"
STIMULUS_EPOCH_ARRAY_MANIFEST_SCHEMA_ID = (
    "palette.stimulus_epoch_windows.array_schema_manifest"
)
WINDOWS_PATH = "windows"
WINDOW_LABEL_WIDTH = 96
EVENT_NAME_WIDTH = 96
SOURCE_POLICY_WIDTH = 160


def _specs() -> dict[str, ColumnSpec]:
    eager = AccessPattern.EAGER
    scientific = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY
    lineage = AnalysisAuthorityRole.LINEAGE_INDEX
    semantic = AnalysisAuthorityRole.SEMANTIC_METADATA
    return {
        "windows/window_id": ColumnSpec(
            path="windows/window_id",
            dtype=np.dtype("int32").str,
            axes=("row",),
            authority=lineage,
            access=eager,
            fill="every row stores one nonnegative stable window identifier",
            null="no null sentinel",
        ),
        "windows/label_bytes": ColumnSpec(
            path="windows/label_bytes",
            dtype=np.dtype("uint8").str,
            axes=("row", "label_byte"),
            authority=semantic,
            access=eager,
            fill="NUL-padded UTF-8 in exactly 96 bytes",
            null="empty labels are forbidden",
        ),
        "windows/start_frame": ColumnSpec(
            path="windows/start_frame",
            dtype=np.dtype("int64").str,
            axes=("row",),
            units="acquisition_frame_index",
            authority=scientific,
            access=eager,
            fill="every row stores the inclusive first camera frame",
            null="negative frame indices are forbidden",
        ),
        "windows/end_frame": ColumnSpec(
            path="windows/end_frame",
            dtype=np.dtype("int64").str,
            axes=("row",),
            units="acquisition_frame_index",
            authority=scientific,
            access=eager,
            fill="every row stores the inclusive last camera frame",
            null="negative frame indices are forbidden",
        ),
        "windows/start_time_s": ColumnSpec(
            path="windows/start_time_s",
            dtype=np.dtype("float64").str,
            axes=("row",),
            units="s",
            authority=scientific,
            access=eager,
            fill="start_frame divided by the exact run FPS",
            null="NaN and infinity are forbidden",
        ),
        "windows/end_time_s": ColumnSpec(
            path="windows/end_time_s",
            dtype=np.dtype("float64").str,
            axes=("row",),
            units="s",
            authority=scientific,
            access=eager,
            fill="exclusive end time: (end_frame + 1) divided by FPS",
            null="NaN and infinity are forbidden",
        ),
        "windows/duration_s": ColumnSpec(
            path="windows/duration_s",
            dtype=np.dtype("float64").str,
            axes=("row",),
            units="s",
            authority=scientific,
            access=eager,
            fill="inclusive frame count divided by FPS",
            null="NaN, infinity, and nonpositive durations are forbidden",
        ),
        "windows/source_start_event_name_bytes": ColumnSpec(
            path="windows/source_start_event_name_bytes",
            dtype=np.dtype("uint8").str,
            axes=("row", "event_name_byte"),
            authority=lineage,
            access=eager,
            fill="NUL-padded UTF-8 in exactly 96 bytes",
            null="empty source event names are forbidden",
        ),
        "windows/source_end_event_name_bytes": ColumnSpec(
            path="windows/source_end_event_name_bytes",
            dtype=np.dtype("uint8").str,
            axes=("row", "event_name_byte"),
            authority=lineage,
            access=eager,
            fill="NUL-padded UTF-8 in exactly 96 bytes",
            null="empty source event names are forbidden",
        ),
        "windows/source_start_event_frame": ColumnSpec(
            path="windows/source_start_event_frame",
            dtype=np.dtype("int64").str,
            axes=("row",),
            units="acquisition_frame_boundary",
            authority=lineage,
            access=eager,
            fill="resolved inclusive source-event boundary",
            null="negative frame boundaries are forbidden",
        ),
        "windows/source_end_event_frame": ColumnSpec(
            path="windows/source_end_event_frame",
            dtype=np.dtype("int64").str,
            axes=("row",),
            units="acquisition_frame_boundary",
            authority=lineage,
            access=eager,
            fill="resolved exclusive source-event boundary",
            null="negative frame boundaries are forbidden",
        ),
        "windows/source_policy_bytes": ColumnSpec(
            path="windows/source_policy_bytes",
            dtype=np.dtype("uint8").str,
            axes=("row", "source_policy_byte"),
            authority=semantic,
            access=eager,
            fill="NUL-padded UTF-8 in exactly 160 bytes",
            null="empty source policies are forbidden",
        ),
    }


STIMULUS_EPOCH_FIELD_NAMES = tuple(
    path.split("/", 1)[1] for path in _specs()
)


def _array(run_group: Any, path: str) -> Any:
    node = run_group
    for component in path.split("/"):
        node = node[component]
    return node


def _decode_fixed_utf8(values: np.ndarray, *, label: str) -> tuple[str, ...]:
    rows = np.asarray(values)
    if rows.ndim != 2 or rows.dtype != np.dtype("uint8"):
        raise ValueError(f"{label} must be a rank-2 uint8 array.")
    decoded: list[str] = []
    for row_index, row in enumerate(rows):
        payload = bytes(np.asarray(row, dtype=np.uint8).tolist())
        terminator = payload.find(b"\0")
        if terminator < 0:
            raise ValueError(f"{label}[{row_index}] lacks a NUL terminator.")
        encoded = payload[:terminator]
        trailing = payload[terminator:]
        if any(trailing):
            raise ValueError(
                f"{label}[{row_index}] has nonzero bytes after its NUL terminator."
            )
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label}[{row_index}] is not valid UTF-8.") from exc
        if not text:
            raise ValueError(f"{label}[{row_index}] is empty.")
        decoded.append(text)
    return tuple(decoded)


def validate_stimulus_epoch_semantics(run_group: Any) -> tuple[str, ...]:
    """Validate row ordering, interval, fixed-text, and time semantics."""

    errors: list[str] = []
    try:
        arrays = collect_run_arrays(run_group)
        specs = _specs()
        if set(arrays) != set(specs):
            missing = sorted(set(specs) - set(arrays))
            unexpected = sorted(set(arrays) - set(specs))
            if missing:
                errors.append(f"missing required arrays: {missing!r}")
            if unexpected:
                errors.append(f"unexpected scientific arrays: {unexpected!r}")
            return tuple(errors)
        declarations = build_exact_array_declarations(
            arrays,
            schema_prefix="palette.array.stimulus_epoch",
            required=specs,
            optional_bundles={},
        )
        dimensions: dict[str, int] = {}
        for declaration in declarations:
            observed = arrays[declaration.path]
            contract_errors = declaration.contract.validate_observation(
                observed,
                dimensions=dimensions,
            )
            errors.extend(
                f"{declaration.path}: {error}" for error in contract_errors
            )
        if errors:
            return tuple(errors)

        n_rows = int(arrays["windows/window_id"].shape[0])
        if n_rows <= 0:
            errors.append("stimulus-epoch v2 requires at least one window")
        for path, array in arrays.items():
            if int(array.shape[0]) != n_rows:
                errors.append(
                    f"{path}: row count {int(array.shape[0])} differs from {n_rows}"
                )
        for path, expected_width in (
            ("windows/label_bytes", WINDOW_LABEL_WIDTH),
            ("windows/source_start_event_name_bytes", EVENT_NAME_WIDTH),
            ("windows/source_end_event_name_bytes", EVENT_NAME_WIDTH),
            ("windows/source_policy_bytes", SOURCE_POLICY_WIDTH),
        ):
            if tuple(arrays[path].shape) != (n_rows, expected_width):
                errors.append(
                    f"{path}: expected shape {(n_rows, expected_width)!r}, "
                    f"got {tuple(arrays[path].shape)!r}"
                )
        if errors:
            return tuple(errors)

        labels = _decode_fixed_utf8(
            np.asarray(arrays["windows/label_bytes"][:]), label="label_bytes"
        )
        _decode_fixed_utf8(
            np.asarray(arrays["windows/source_start_event_name_bytes"][:]),
            label="source_start_event_name_bytes",
        )
        _decode_fixed_utf8(
            np.asarray(arrays["windows/source_end_event_name_bytes"][:]),
            label="source_end_event_name_bytes",
        )
        _decode_fixed_utf8(
            np.asarray(arrays["windows/source_policy_bytes"][:]),
            label="source_policy_bytes",
        )
        if len(set(labels)) != len(labels):
            errors.append("window labels must be unique")

        ids = np.asarray(arrays["windows/window_id"][:], dtype=np.int32)
        starts = np.asarray(arrays["windows/start_frame"][:], dtype=np.int64)
        ends = np.asarray(arrays["windows/end_frame"][:], dtype=np.int64)
        source_starts = np.asarray(
            arrays["windows/source_start_event_frame"][:], dtype=np.int64
        )
        source_ends = np.asarray(
            arrays["windows/source_end_event_frame"][:], dtype=np.int64
        )
        start_time = np.asarray(arrays["windows/start_time_s"][:], dtype=np.float64)
        end_time = np.asarray(arrays["windows/end_time_s"][:], dtype=np.float64)
        duration = np.asarray(arrays["windows/duration_s"][:], dtype=np.float64)

        if np.any(ids < 0) or np.any(np.diff(ids.astype(np.int64)) <= 0):
            errors.append("window_id must be nonnegative, unique, and strictly increasing")
        total_frames = run_group.attrs.get("total_frames")
        fps = run_group.attrs.get("fps")
        if run_group.attrs.get("window_count") != n_rows:
            errors.append("window_count must equal the exact persisted row count")
        for attr_name in (
            "recording_id",
            "source_stimulus_run",
            "source_stimulus_path",
            "method",
            "method_version",
            "epoch_policy",
        ):
            value = run_group.attrs.get(attr_name)
            if type(value) is not str or not value.strip():
                errors.append(f"{attr_name} must be one nonempty exact string")
        if type(total_frames) is not int or total_frames <= 0:
            errors.append("total_frames must be a positive exact integer")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) <= 0:
            errors.append("fps must be a positive finite number")
        elif not np.isfinite(float(fps)):
            errors.append("fps must be a positive finite number")
        if errors:
            return tuple(errors)

        assert type(total_frames) is int
        fps_value = float(fps)
        if np.any(starts < 0) or np.any(ends < starts) or np.any(ends >= total_frames):
            errors.append("window frame intervals must be nonempty and inside total_frames")
        if n_rows > 1 and np.any(starts[1:] <= ends[:-1]):
            errors.append("window rows must be chronological and non-overlapping")
        if (
            np.any(source_starts < 0)
            or np.any(source_starts > total_frames)
            or np.any(source_ends < 0)
            or np.any(source_ends > total_frames)
            or np.any(source_ends <= source_starts)
        ):
            errors.append("source event boundaries must be ordered inside [0, total_frames]")
        else:
            expected_starts = source_starts
            expected_ends = np.minimum(
                total_frames - 1,
                np.maximum(expected_starts, source_ends - 1),
            )
            if not np.array_equal(starts, expected_starts):
                errors.append("start_frame differs from its resolved source boundary")
            if not np.array_equal(ends, expected_ends):
                errors.append("end_frame differs from its resolved exclusive boundary")
        if not (
            np.all(np.isfinite(start_time))
            and np.all(np.isfinite(end_time))
            and np.all(np.isfinite(duration))
        ):
            errors.append("window time columns must be finite")
        else:
            expected_start = starts.astype(np.float64) / fps_value
            expected_end = (ends.astype(np.float64) + 1.0) / fps_value
            expected_duration = (ends - starts + 1).astype(np.float64) / fps_value
            tolerance = max(np.finfo(np.float64).eps * max(total_frames, 1), 1e-12)
            if not np.allclose(start_time, expected_start, rtol=0.0, atol=tolerance):
                errors.append("start_time_s differs from start_frame / fps")
            if not np.allclose(end_time, expected_end, rtol=0.0, atol=tolerance):
                errors.append("end_time_s differs from (end_frame + 1) / fps")
            if not np.allclose(duration, expected_duration, rtol=0.0, atol=tolerance):
                errors.append("duration_s differs from inclusive frame count / fps")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return tuple(errors)


def validate_legacy_stimulus_epoch_source(run_group: Any) -> tuple[str, ...]:
    """Validate an explicit v1 input without treating it as a v2 authority."""

    attrs = dict(run_group.attrs)
    errors: list[str] = []
    if attrs.get("schema_id") != LEGACY_STIMULUS_EPOCH_SCHEMA_ID:
        errors.append("legacy stimulus-epoch schema_id mismatch")
    if attrs.get("schema_version") != LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION:
        errors.append("legacy stimulus-epoch schema_version mismatch")
    windows = run_group.get(WINDOWS_PATH)
    if windows is None:
        errors.append("legacy stimulus-epoch windows group is missing")
        return tuple(errors)
    if windows.attrs.get("storage_layout") != "columnar":
        errors.append("legacy windows storage_layout must be columnar")
    if list(windows.attrs.get("field_names", [])) != list(
        STIMULUS_EPOCH_FIELD_NAMES
    ):
        errors.append("legacy windows field_names differ from the frozen inventory")
    errors.extend(validate_stimulus_epoch_semantics(run_group))
    return tuple(errors)


def build_stimulus_epoch_array_declarations(
    run_group: Any,
    *,
    byte_planner_adopted: bool,
) -> tuple[Any, ...]:
    return build_exact_array_declarations(
        collect_run_arrays(run_group),
        schema_prefix="palette.array.stimulus_epoch",
        required=_specs(),
        optional_bundles={},
        byte_planner_adopted=byte_planner_adopted,
    )


def build_stimulus_epoch_array_manifest(
    run_group: Any,
    *,
    byte_planner_adopted: bool,
) -> dict[str, Any]:
    return build_exact_manifest(
        run_group,
        collect_run_arrays(run_group),
        manifest_schema_id=STIMULUS_EPOCH_ARRAY_MANIFEST_SCHEMA_ID,
        run_schema_id=STIMULUS_EPOCH_RUN_SCHEMA_ID,
        run_schema_version=STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
        layout=STIMULUS_EPOCH_LAYOUT,
        schema_prefix="palette.array.stimulus_epoch",
        required=_specs(),
        optional_bundles={},
        columnar_table_paths=(WINDOWS_PATH,),
        byte_planner_adopted=byte_planner_adopted,
    )


def validate_stimulus_epoch_array_manifest(
    run_group: Any,
    *,
    byte_planner_adopted: bool,
) -> tuple[str, ...]:
    attrs = dict(run_group.attrs)
    errors: list[str] = []
    if attrs.get("schema_id") != STIMULUS_EPOCH_RUN_SCHEMA_ID:
        errors.append("stimulus-epoch v2 schema_id mismatch")
    if attrs.get("schema_version") != STIMULUS_EPOCH_RUN_SCHEMA_VERSION:
        errors.append("stimulus-epoch v2 schema_version mismatch")
    if attrs.get("layout") != STIMULUS_EPOCH_LAYOUT:
        errors.append("stimulus-epoch v2 layout mismatch")
    if attrs.get("row_axis") != "epoch_windows":
        errors.append("stimulus-epoch v2 row_axis mismatch")
    errors.extend(validate_stimulus_epoch_semantics(run_group))
    errors.extend(
        validate_exact_manifest(
            run_group,
            collect_run_arrays(run_group),
            attrs.get(MANIFEST_ATTRIBUTE),
            manifest_schema_id=STIMULUS_EPOCH_ARRAY_MANIFEST_SCHEMA_ID,
            run_schema_id=STIMULUS_EPOCH_RUN_SCHEMA_ID,
            run_schema_version=STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
            layout=STIMULUS_EPOCH_LAYOUT,
            schema_prefix="palette.array.stimulus_epoch",
            required=_specs(),
            optional_bundles={},
            columnar_table_paths=(WINDOWS_PATH,),
            byte_planner_adopted=byte_planner_adopted,
        )
    )
    return tuple(errors)


def write_stimulus_epoch_array_manifest(
    run_group: Any,
    *,
    byte_planner_adopted: bool,
) -> Mapping[str, Any]:
    manifest = build_stimulus_epoch_array_manifest(
        run_group,
        byte_planner_adopted=byte_planner_adopted,
    )
    run_group.attrs[MANIFEST_ATTRIBUTE] = manifest
    errors = validate_stimulus_epoch_array_manifest(
        run_group,
        byte_planner_adopted=byte_planner_adopted,
    )
    if errors:
        raise ValueError("Invalid stimulus-epoch v2 run: " + "; ".join(errors))
    return manifest


__all__ = [
    "EVENT_NAME_WIDTH",
    "LEGACY_STIMULUS_EPOCH_SCHEMA_ID",
    "LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION",
    "SOURCE_POLICY_WIDTH",
    "STIMULUS_EPOCH_ARRAY_MANIFEST_SCHEMA_ID",
    "STIMULUS_EPOCH_FIELD_NAMES",
    "STIMULUS_EPOCH_LAYOUT",
    "STIMULUS_EPOCH_RUN_SCHEMA_ID",
    "STIMULUS_EPOCH_RUN_SCHEMA_VERSION",
    "WINDOW_LABEL_WIDTH",
    "build_stimulus_epoch_array_declarations",
    "build_stimulus_epoch_array_manifest",
    "validate_legacy_stimulus_epoch_source",
    "validate_stimulus_epoch_array_manifest",
    "validate_stimulus_epoch_semantics",
    "write_stimulus_epoch_array_manifest",
]
