"""Exact logical and semantic contract for immutable stimulus-epoch v2 runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import AnalysisAuthorityRole
from fisheye.shared.run_lineage_fingerprint import (
    LINEAGE_ATTR_SCHEMA_ID,
    LINEAGE_ATTR_SCHEMA_VERSION,
    LINEAGE_CANONICALIZATION,
    build_run_lineage_payload,
    canonical_lineage_json,
    compute_run_lineage_hash,
    normalize_lineage_value,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.storage_intent import AccessPattern
from fisheye.shared.zarr_run_completion import RUN_NAME_ATTR

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
STIMULUS_EPOCH_RUN_MANIFEST_ATTR = "stimulus_epoch_run_manifest"
STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_ID = "palette.stimulus_epoch_run_manifest"
STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_VERSION = 1
STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_ID = "palette.stimulus_epoch_logical_content"
STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_VERSION = 1
STIMULUS_SOURCE_FINGERPRINT_ALGORITHM = (
    "sha256_canonical_stimulus_group_logical_tree_v1"
)
_SOURCE_EPOCH_LIFECYCLE_FIELDS = {
    "completion_status",
    "stage_selector_eligible",
    "stage_selector_marker",
    "selection_policy",
    "allow_selector_ineligible_source",
}


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


STIMULUS_EPOCH_FIELD_NAMES = tuple(path.split("/", 1)[1] for path in _specs())


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
            errors.extend(f"{declaration.path}: {error}" for error in contract_errors)
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
            errors.append(
                "window_id must be nonnegative, unique, and strictly increasing"
            )
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
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or float(fps) <= 0
        ):
            errors.append("fps must be a positive finite number")
        elif not np.isfinite(float(fps)):
            errors.append("fps must be a positive finite number")
        if errors:
            return tuple(errors)

        assert type(total_frames) is int
        fps_value = float(fps)
        if np.any(starts < 0) or np.any(ends < starts) or np.any(ends >= total_frames):
            errors.append(
                "window frame intervals must be nonempty and inside total_frames"
            )
        if n_rows > 1 and np.any(starts[1:] <= ends[:-1]):
            errors.append("window rows must be chronological and non-overlapping")
        if (
            np.any(source_starts < 0)
            or np.any(source_starts > total_frames)
            or np.any(source_ends < 0)
            or np.any(source_ends > total_frames)
            or np.any(source_ends <= source_starts)
        ):
            errors.append(
                "source event boundaries must be ordered inside [0, total_frames]"
            )
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
    if list(windows.attrs.get("field_names", [])) != list(STIMULUS_EPOCH_FIELD_NAMES):
        errors.append("legacy windows field_names differ from the frozen inventory")
    errors.extend(validate_stimulus_epoch_semantics(run_group))
    return tuple(errors)


def stimulus_epoch_logical_content_document(run_group: Any) -> dict[str, object]:
    """Return an exact decoded-content document for all twelve arrays."""

    errors = validate_stimulus_epoch_semantics(run_group)
    if errors:
        raise ValueError(
            "Cannot describe invalid stimulus-epoch content: " + "; ".join(errors)
        )
    arrays: dict[str, object] = {}
    for path in sorted(_specs()):
        array = _array(run_group, path)
        values = np.ascontiguousarray(array[...])
        arrays[path] = {
            "dtype": str(np.dtype(array.dtype)),
            "shape": [int(value) for value in array.shape],
            "digest_algorithm": "sha256_c_order_decoded_bytes_v1",
            "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        }
    document: dict[str, object] = {
        "schema_id": STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_VERSION,
        "array_count": len(arrays),
        "arrays": arrays,
    }
    canonical_json_bytes(document)
    return document


def stimulus_epoch_logical_content_sha256(run_group: Any) -> str:
    return canonical_json_sha256(stimulus_epoch_logical_content_document(run_group))


def _iter_group_tree(group: Any, prefix: str = ""):
    yield prefix, group
    for name, child in sorted(group.groups(), key=lambda item: item[0]):
        child_path = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_group_tree(child, child_path)


def stimulus_group_logical_fingerprint(group: Any) -> str:
    """Hash one complete stimulus-group logical tree and its attributes."""

    groups: dict[str, object] = {}
    arrays: dict[str, object] = {}
    for group_path, node in _iter_group_tree(group):
        groups[group_path] = normalize_lineage_value(dict(node.attrs))
        for name, array in sorted(node.arrays(), key=lambda item: item[0]):
            path = f"{group_path}/{name}" if group_path else str(name)
            values = np.ascontiguousarray(array[...])
            arrays[path] = {
                "dtype": str(np.dtype(array.dtype)),
                "shape": [int(value) for value in array.shape],
                "attributes": normalize_lineage_value(dict(array.attrs)),
                "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            }
    document = {
        "schema_id": "palette.stimulus_group_logical_tree",
        "schema_version": 1,
        "groups": groups,
        "arrays": arrays,
    }
    canonical_json_bytes(document)
    return canonical_json_sha256(document)


def _required_text_attr(run_group: Any, name: str) -> str:
    value = run_group.attrs.get(name)
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be one canonical nonempty exact string.")
    return value


def _required_exact_int_attr(run_group: Any, name: str, *, positive: bool) -> int:
    value = run_group.attrs.get(name)
    if type(value) is not int or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be one {qualifier}exact integer.")
    return value


def _required_mapping_attr(
    run_group: Any,
    name: str,
    *,
    exact_fields: set[str],
) -> dict[str, Any]:
    value = run_group.attrs.get(name)
    if type(value) is not dict or set(value) != exact_fields:
        raise ValueError(f"{name} must have the exact canonical field set.")
    canonical_json_bytes(value)
    return dict(value)


def _optional_source_epoch_lifecycle(run_group: Any) -> dict[str, Any] | None:
    """Read the lifecycle binding while retaining old v2 manifest compatibility."""

    if "source_stimulus_epoch_lifecycle" not in run_group.attrs:
        return None
    value = _required_mapping_attr(
        run_group,
        "source_stimulus_epoch_lifecycle",
        exact_fields=_SOURCE_EPOCH_LIFECYCLE_FIELDS,
    )
    if value["completion_status"] != "complete":
        raise ValueError("source stimulus-epoch lifecycle is not complete.")
    if value["stage_selector_eligible"] not in (True, False, None):
        raise ValueError("source stimulus-epoch lifecycle selector state is not exact.")
    if type(value["allow_selector_ineligible_source"]) is not bool:
        raise ValueError(
            "source stimulus-epoch lifecycle opt-in marker is not an exact bool."
        )
    if (
        value["allow_selector_ineligible_source"]
        and value["stage_selector_eligible"] is not False
    ):
        raise ValueError(
            "selector-ineligible source lifecycle does not bind literal false."
        )
    return value


def _require_protocol_profile_identity(value: Mapping[str, Any]) -> None:
    for name in (
        "profile_id",
        "profile_sha256",
        "profile_source",
        "source_adapter_id",
        "role_resolver_id",
    ):
        item = value.get(name)
        if type(item) is not str or not item.strip() or item != item.strip():
            raise ValueError(f"protocol_profile.{name} must be canonical text.")
    for name in (
        "profile_version",
        "source_adapter_version",
        "role_resolver_version",
    ):
        item = value.get(name)
        if type(item) is not int or item <= 0:
            raise ValueError(
                f"protocol_profile.{name} must be a positive exact integer."
            )
    digest = value["profile_sha256"]
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("protocol_profile.profile_sha256 must be lowercase SHA-256.")


def build_stimulus_epoch_candidate_lineage_payload(run_group: Any) -> dict[str, Any]:
    """Reconstruct the candidate-owned lineage payload from bound run attrs."""

    protocol_profile = _required_mapping_attr(
        run_group,
        "protocol_profile",
        exact_fields={
            "profile_id",
            "profile_version",
            "profile_sha256",
            "profile_source",
            "source_adapter_id",
            "source_adapter_version",
            "role_resolver_id",
            "role_resolver_version",
        },
    )
    _require_protocol_profile_identity(protocol_profile)
    source_stimulus_run = _required_text_attr(run_group, "source_stimulus_run")
    source_stimulus_path = _required_text_attr(run_group, "source_stimulus_path")
    if source_stimulus_path != f"analysis/stimulus_runs/{source_stimulus_run}":
        raise ValueError("source_stimulus_path does not bind source_stimulus_run.")
    source_epoch_run = _required_text_attr(run_group, "source_stimulus_epoch_run")
    source_epoch_path = _required_text_attr(run_group, "source_stimulus_epoch_path")
    if source_epoch_path != f"analysis/stimulus_epoch_runs/{source_epoch_run}":
        raise ValueError(
            "source_stimulus_epoch_path does not bind source_stimulus_epoch_run."
        )
    fingerprint_algorithm = _required_text_attr(
        run_group, "source_stimulus_fingerprint_algorithm"
    )
    if fingerprint_algorithm != STIMULUS_SOURCE_FINGERPRINT_ALGORITHM:
        raise ValueError("source stimulus fingerprint algorithm mismatch.")
    source_stimulus_fingerprint = _required_text_attr(
        run_group, "source_stimulus_fingerprint"
    )
    source_epoch_lineage_hash = _required_text_attr(
        run_group, "source_stimulus_epoch_lineage_hash"
    )
    source_epoch_lineage_payload_sha256 = _required_text_attr(
        run_group, "source_stimulus_epoch_lineage_payload_sha256"
    )
    source_epoch_content_sha256 = _required_text_attr(
        run_group, "source_stimulus_epoch_logical_content_sha256"
    )
    source_epoch_lifecycle = _optional_source_epoch_lifecycle(run_group)
    for name, value in (
        ("source_stimulus_fingerprint", source_stimulus_fingerprint),
        ("source_stimulus_epoch_lineage_hash", source_epoch_lineage_hash),
        (
            "source_stimulus_epoch_lineage_payload_sha256",
            source_epoch_lineage_payload_sha256,
        ),
        (
            "source_stimulus_epoch_logical_content_sha256",
            source_epoch_content_sha256,
        ),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    epoch_policy = _required_text_attr(run_group, "epoch_policy")
    epoch_policy_version = _required_exact_int_attr(
        run_group, "epoch_policy_version", positive=True
    )
    method = _required_text_attr(run_group, "method")
    method_version = _required_text_attr(run_group, "method_version")
    total_frames = _required_exact_int_attr(run_group, "total_frames", positive=True)
    fps = run_group.attrs.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise ValueError("fps must be one positive finite number.")
    materializer_commit = _required_text_attr(
        run_group, "candidate_materializer_git_commit"
    )
    materializer_dirty = run_group.attrs.get("candidate_materializer_git_dirty")
    if type(materializer_dirty) is not bool:
        raise ValueError("candidate_materializer_git_dirty must be an exact bool.")
    lineage_parameters: dict[str, Any] = {
        "recording_id": _required_text_attr(run_group, "recording_id"),
        "fps": float(fps),
        "total_frames": total_frames,
        "epoch_policy": epoch_policy,
        "epoch_policy_version": epoch_policy_version,
        "protocol_profile": protocol_profile,
    }
    if source_epoch_lifecycle is not None:
        lineage_parameters["source_stimulus_epoch_lifecycle"] = source_epoch_lifecycle
    return build_run_lineage_payload(
        run_family="analysis/stimulus_epoch_runs",
        analysis_schema={
            "schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
            "schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
            "layout": STIMULUS_EPOCH_LAYOUT,
            "row_axis": "epoch_windows",
        },
        method=method,
        method_version=method_version,
        source_refs={
            "source_stimulus_run": source_stimulus_run,
            "source_stimulus_path": source_stimulus_path,
            "source_stimulus_epoch_run": source_epoch_run,
            "source_stimulus_epoch_path": source_epoch_path,
        },
        source_fingerprints={
            "source_stimulus_fingerprint_algorithm": fingerprint_algorithm,
            "source_stimulus_fingerprint": source_stimulus_fingerprint,
            "source_stimulus_epoch_lineage_hash": source_epoch_lineage_hash,
            "source_stimulus_epoch_lineage_payload_sha256": (
                source_epoch_lineage_payload_sha256
            ),
            "source_stimulus_epoch_logical_content_sha256": (
                source_epoch_content_sha256
            ),
        },
        parameters=lineage_parameters,
        code={
            "git_commit": materializer_commit,
            "git_dirty": materializer_dirty,
        },
    )


def validate_stimulus_epoch_candidate_lineage(run_group: Any) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        lineage_json = run_group.attrs.get("lineage_payload_json")
        if type(lineage_json) is not str:
            raise ValueError("lineage_payload_json must be a canonical JSON string.")
        payload = json.loads(lineage_json)
        if type(payload) is not dict:
            raise ValueError("lineage_payload_json must decode to an exact object.")
        if lineage_json != canonical_lineage_json(payload):
            errors.append("candidate lineage payload JSON is not canonical")
        expected = build_stimulus_epoch_candidate_lineage_payload(run_group)
        if canonical_lineage_json(payload) != canonical_lineage_json(expected):
            errors.append("candidate lineage payload differs from executable binding")
        lineage_hash = compute_run_lineage_hash(payload)
        for attr_name in ("source_fingerprint", "source_lineage_hash", "lineage_hash"):
            if run_group.attrs.get(attr_name) != lineage_hash:
                errors.append(f"{attr_name} differs from candidate lineage payload")
        if run_group.attrs.get("fingerprint_status") != "complete":
            errors.append("candidate fingerprint_status must be complete")
        if (
            run_group.attrs.get("lineage_fingerprint_schema_id")
            != LINEAGE_ATTR_SCHEMA_ID
        ):
            errors.append("candidate lineage attribute schema_id mismatch")
        if (
            run_group.attrs.get("lineage_fingerprint_schema_version")
            != LINEAGE_ATTR_SCHEMA_VERSION
        ):
            errors.append("candidate lineage attribute schema_version mismatch")
        if (
            run_group.attrs.get("lineage_fingerprint_canonicalization")
            != LINEAGE_CANONICALIZATION
        ):
            errors.append("candidate lineage canonicalization mismatch")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return tuple(errors)


def _authoritative_child_group_document(run_group: Any) -> dict[str, object]:
    if set(run_group.group_keys()) != {WINDOWS_PATH}:
        raise ValueError("stimulus-epoch run must contain only the windows group.")
    windows = run_group[WINDOWS_PATH]
    if list(windows.group_keys()):
        raise ValueError("windows cannot contain nested groups.")
    attrs = dict(windows.attrs)
    if set(attrs) != {"storage_layout", "field_names"}:
        raise ValueError("windows attributes must have the exact v2 field set.")
    if attrs.get("storage_layout") != "columnar":
        raise ValueError("windows storage_layout must be columnar.")
    if attrs.get("field_names") != list(STIMULUS_EPOCH_FIELD_NAMES):
        raise ValueError("windows field_names differ from the exact v2 order.")
    return {"windows": attrs}


def build_stimulus_epoch_run_manifest(run_group: Any) -> dict[str, object]:
    """Build the canonical complete scientific/lineage envelope for v2."""

    if run_group.attrs.get("storage_candidate_profile_promoted") is not False:
        raise ValueError("storage_candidate_profile_promoted must be exact false.")
    if run_group.attrs.get("stage_selector_eligible") is not False:
        raise ValueError("stage_selector_eligible must be exact false.")
    if run_group.attrs.get("schema_id") != STIMULUS_EPOCH_RUN_SCHEMA_ID:
        raise ValueError("stimulus-epoch run schema_id mismatch.")
    if run_group.attrs.get("schema_version") != STIMULUS_EPOCH_RUN_SCHEMA_VERSION:
        raise ValueError("stimulus-epoch run schema_version mismatch.")
    lineage_errors = validate_stimulus_epoch_candidate_lineage(run_group)
    if lineage_errors:
        raise ValueError("Invalid candidate lineage: " + "; ".join(lineage_errors))
    array_errors = validate_stimulus_epoch_array_manifest(
        run_group,
        byte_planner_adopted=True,
    )
    if array_errors:
        raise ValueError("Invalid exact array manifest: " + "; ".join(array_errors))
    from fisheye.analysis.exact_tabular_storage import (
        validate_exact_tabular_storage_receipt,
    )

    storage_errors = validate_exact_tabular_storage_receipt(
        run_group,
        declarations=build_stimulus_epoch_array_declarations(
            run_group,
            byte_planner_adopted=True,
        ),
    )
    if storage_errors:
        raise ValueError(
            "Invalid executable storage receipt: " + "; ".join(storage_errors)
        )
    array_manifest = run_group.attrs.get(MANIFEST_ATTRIBUTE)
    storage_receipt = run_group.attrs.get("analysis_storage_plan_receipt")
    if type(array_manifest) is not dict or type(storage_receipt) is not dict:
        raise ValueError("Exact array manifest and storage receipt are required.")
    source_event_schema = _required_mapping_attr(
        run_group,
        "source_event_schema",
        exact_fields={"events_path", "event_name_fields", "frame_fields"},
    )
    source_refs = _required_mapping_attr(
        run_group,
        "source_refs",
        exact_fields={"source_stimulus_run", "source_stimulus_path"},
    )
    protocol_profile = _required_mapping_attr(
        run_group,
        "protocol_profile",
        exact_fields={
            "profile_id",
            "profile_version",
            "profile_sha256",
            "profile_source",
            "source_adapter_id",
            "source_adapter_version",
            "role_resolver_id",
            "role_resolver_version",
        },
    )
    _require_protocol_profile_identity(protocol_profile)
    lineage_json = _required_text_attr(run_group, "lineage_payload_json")
    logical_content = stimulus_epoch_logical_content_document(run_group)
    source_content_sha256 = _required_text_attr(
        run_group, "source_stimulus_epoch_logical_content_sha256"
    )
    candidate_content_sha256 = canonical_json_sha256(logical_content)
    if source_content_sha256 != candidate_content_sha256:
        raise ValueError("Candidate logical content differs from source epoch content.")
    source_stimulus_run = _required_text_attr(run_group, "source_stimulus_run")
    source_stimulus_path = _required_text_attr(run_group, "source_stimulus_path")
    source_epoch_run = _required_text_attr(run_group, "source_stimulus_epoch_run")
    source_epoch_path = _required_text_attr(run_group, "source_stimulus_epoch_path")
    if source_stimulus_path != f"analysis/stimulus_runs/{source_stimulus_run}":
        raise ValueError("run manifest source stimulus run/path binding mismatch.")
    if source_epoch_path != f"analysis/stimulus_epoch_runs/{source_epoch_run}":
        raise ValueError("run manifest source epoch run/path binding mismatch.")
    source_epoch_lifecycle = _optional_source_epoch_lifecycle(run_group)
    if source_refs != {
        "source_stimulus_run": source_stimulus_run,
        "source_stimulus_path": source_stimulus_path,
    }:
        raise ValueError("source_refs differs from exact source stimulus identity.")
    if source_event_schema.get("events_path") != f"{source_stimulus_path}/events":
        raise ValueError("source_event_schema events_path binding mismatch.")
    if source_event_schema.get("event_name_fields") != [
        "event_name",
        "event_type_name",
        "name",
        "event_type_id",
    ]:
        raise ValueError("source_event_schema event-name fields mismatch.")
    if source_event_schema.get("frame_fields") != [
        "camera_frame_id",
        "camera_frame_num",
        "triggering_camera_frame_id",
    ]:
        raise ValueError("source_event_schema frame fields mismatch.")
    if (
        _required_text_attr(run_group, "storage_candidate_source_run")
        != source_epoch_run
        or _required_text_attr(run_group, "storage_candidate_source_run_path")
        != source_epoch_path
    ):
        raise ValueError("storage candidate source identity differs from source epoch.")
    source_epoch_payload: dict[str, object] = {
        "run": source_epoch_run,
        "path": source_epoch_path,
        "schema_id": LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
        "schema_version": LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
        "lineage_hash": _required_text_attr(
            run_group, "source_stimulus_epoch_lineage_hash"
        ),
        "lineage_payload_sha256": _required_text_attr(
            run_group, "source_stimulus_epoch_lineage_payload_sha256"
        ),
        "logical_content_sha256": source_content_sha256,
    }
    if source_epoch_lifecycle is not None:
        source_epoch_payload["lifecycle"] = source_epoch_lifecycle
    payload: dict[str, object] = {
        "run_identity": {
            "recording_id": _required_text_attr(run_group, "recording_id"),
            "run_name": _required_text_attr(run_group, RUN_NAME_ATTR),
            "run_schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
            "run_schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
            "layout": STIMULUS_EPOCH_LAYOUT,
            "row_axis": "epoch_windows",
        },
        "dimensions": {
            "window_count": _required_exact_int_attr(
                run_group, "window_count", positive=True
            ),
            "total_frames": _required_exact_int_attr(
                run_group, "total_frames", positive=True
            ),
            "fps": float(run_group.attrs["fps"]),
        },
        "source_stimulus": {
            "run": source_stimulus_run,
            "path": source_stimulus_path,
            "fingerprint_algorithm": _required_text_attr(
                run_group, "source_stimulus_fingerprint_algorithm"
            ),
            "fingerprint": _required_text_attr(
                run_group, "source_stimulus_fingerprint"
            ),
            "event_schema": source_event_schema,
        },
        "source_epoch": source_epoch_payload,
        "protocol": {
            "method": _required_text_attr(run_group, "method"),
            "method_version": _required_text_attr(run_group, "method_version"),
            "epoch_policy": _required_text_attr(run_group, "epoch_policy"),
            "epoch_policy_version": _required_exact_int_attr(
                run_group, "epoch_policy_version", positive=True
            ),
            "profile": protocol_profile,
            "source_refs": source_refs,
        },
        "candidate_lineage": {
            "lineage_hash": _required_text_attr(run_group, "lineage_hash"),
            "lineage_payload_sha256": hashlib.sha256(
                lineage_json.encode("utf-8")
            ).hexdigest(),
            "fingerprint_status": "complete",
        },
        "logical_content": logical_content,
        "schema_bindings": {
            "array_manifest_schema_id": array_manifest.get("schema_id"),
            "array_manifest_payload_digest": array_manifest.get("payload_digest"),
            "storage_receipt_schema_id": storage_receipt.get("schema_id"),
            "storage_receipt_payload_digest": storage_receipt.get("payload_digest"),
        },
        "authoritative_child_groups": _authoritative_child_group_document(run_group),
        "publication_state": {
            "stage_selector_eligible": False,
            "storage_candidate_profile_promoted": False,
            "storage_profile_id": _required_text_attr(
                run_group, "analysis_storage_profile_id"
            ),
            "source_candidate_run": _required_text_attr(
                run_group, "storage_candidate_source_run"
            ),
            "source_candidate_path": _required_text_attr(
                run_group, "storage_candidate_source_run_path"
            ),
        },
    }
    canonical_json_bytes(payload)
    return {
        "schema_id": STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": STIMULUS_EPOCH_RUN_MANIFEST_ATTR,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def validate_stimulus_epoch_run_manifest(run_group: Any) -> tuple[str, ...]:
    value = run_group.attrs.get(STIMULUS_EPOCH_RUN_MANIFEST_ATTR)
    if type(value) is not dict or set(value) != {
        "schema_id",
        "schema_version",
        "persisted_attribute",
        "digest_algorithm",
        "payload",
        "payload_digest",
    }:
        return ("stimulus-epoch run manifest is absent or not exact",)
    errors: list[str] = []
    if value.get("schema_id") != STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_ID:
        errors.append("stimulus-epoch run manifest schema_id mismatch")
    if value.get("schema_version") != STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_VERSION:
        errors.append("stimulus-epoch run manifest schema_version mismatch")
    if value.get("persisted_attribute") != STIMULUS_EPOCH_RUN_MANIFEST_ATTR:
        errors.append("stimulus-epoch run manifest attribute binding mismatch")
    if value.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("stimulus-epoch run manifest digest algorithm mismatch")
    payload = value.get("payload")
    if type(payload) is not dict:
        return (*errors, "stimulus-epoch run manifest payload is not exact")
    try:
        if value.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("stimulus-epoch run manifest payload digest mismatch")
        expected = build_stimulus_epoch_run_manifest(run_group)
        if canonical_json_bytes(value) != canonical_json_bytes(expected):
            errors.append("stimulus-epoch run manifest differs from executable binding")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return tuple(errors)


def write_stimulus_epoch_run_manifest(run_group: Any) -> Mapping[str, Any]:
    manifest = build_stimulus_epoch_run_manifest(run_group)
    run_group.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR] = manifest
    errors = validate_stimulus_epoch_run_manifest(run_group)
    if errors:
        raise ValueError("Invalid stimulus-epoch run manifest: " + "; ".join(errors))
    return manifest


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
    "STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_ID",
    "STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_VERSION",
    "STIMULUS_EPOCH_RUN_MANIFEST_ATTR",
    "STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_ID",
    "STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_VERSION",
    "STIMULUS_EPOCH_RUN_SCHEMA_ID",
    "STIMULUS_EPOCH_RUN_SCHEMA_VERSION",
    "STIMULUS_SOURCE_FINGERPRINT_ALGORITHM",
    "WINDOW_LABEL_WIDTH",
    "build_stimulus_epoch_array_declarations",
    "build_stimulus_epoch_array_manifest",
    "build_stimulus_epoch_candidate_lineage_payload",
    "build_stimulus_epoch_run_manifest",
    "stimulus_epoch_logical_content_document",
    "stimulus_epoch_logical_content_sha256",
    "stimulus_group_logical_fingerprint",
    "validate_legacy_stimulus_epoch_source",
    "validate_stimulus_epoch_array_manifest",
    "validate_stimulus_epoch_candidate_lineage",
    "validate_stimulus_epoch_run_manifest",
    "validate_stimulus_epoch_semantics",
    "write_stimulus_epoch_array_manifest",
    "write_stimulus_epoch_run_manifest",
]
