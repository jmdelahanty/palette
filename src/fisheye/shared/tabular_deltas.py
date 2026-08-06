"""Sparse, partition-owned edit deltas for immutable tabular snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

import numpy as np


DELTA_RUN_SCHEMA = "palette.tabular_delta_run.v1"
DELTA_GENERATION_SCHEMA = "palette.tabular_delta_generation.v1"
DELTA_PARTITION_SCHEMA = "palette.tabular_delta_partition.v1"
DELTA_PARENT = "edit_delta_runs"

KEYPOINT_OPERATION_CODE_MAP: dict[str, int] = {
    "replace_xy": 1,
    "set_valid": 2,
    "clear_keypoint": 3,
}
KEYPOINT_DELTA_REASON_CODE_MAP: dict[str, int] = {
    "manual_correction": 1,
    "fish_present_no_keypoints": 2,
    "detection_issue": 3,
    "cleared_failure_label": 4,
}
DETECTION_OPERATION_CODE_MAP: dict[str, int] = {
    "replace_bbox": 1,
    "delete_instance": 2,
    "restore_instance": 3,
    "add_instance": 4,
}
OPERATION_CODE_SCHEMA = "palette.tabular_delta_operation_codes.v1"

DeltaKind = Literal["keypoints", "detections"]
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ResolvedKeypointDelta:
    """Newest verified edit for one immutable observation landmark."""

    instance_key: int
    row_index: int
    keypoint_index: int
    operation_code: int
    revision: int
    timestamp_ns: int
    partition: str
    partition_row_index: int
    reason_code: int
    new_xy: tuple[float, float]
    valid: bool


@dataclass(frozen=True)
class ResolvedKeypointDeltaOverlay:
    """Verified, deterministically resolved view of one delta generation."""

    delta_run: str
    generation: str
    generation_status: str
    base_run_path: str
    base_instance_key_sha256: str
    partition_count: int
    event_count: int
    max_revision: int
    overlay_sha256: str
    reason_code_map: Mapping[str, int]
    edits: tuple[ResolvedKeypointDelta, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text or not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(f"{name} must match {_SAFE_COMPONENT.pattern!r}; got {value!r}.")
    return text


def _resolve_group(root: Any, path: str) -> Any:
    node = root
    for part in [value for value in str(path).strip("/").split("/") if value]:
        node = node[part]
    return node


def _instance_key_array(base_run: Any, target_kind: DeltaKind) -> Any:
    if target_kind == "detections" and "instances" in base_run:
        return base_run["instances/instance_key"]
    return base_run["instance_key"]


def instance_key_digest(values: np.ndarray) -> str:
    keys = np.asarray(values, dtype=np.uint64).reshape(-1)
    return hashlib.sha256(np.ascontiguousarray(keys).view(np.uint8)).hexdigest()


def bind_base_run(root: Any, *, base_run_path: str, target_kind: DeltaKind) -> dict[str, Any]:
    base = _resolve_group(root, base_run_path)
    keys = np.asarray(_instance_key_array(base, target_kind)[:], dtype=np.uint64).reshape(-1)
    if keys.size and np.unique(keys).shape[0] != keys.shape[0]:
        raise ValueError(f"Base run {base_run_path} instance_key values are not unique.")
    if str(base.attrs.get("artifact_mutability") or "") not in {
        "immutable_snapshot",
        "raw_immutable",
    }:
        raise ValueError(
            f"Base run {base_run_path} is not stamped as an immutable snapshot."
        )
    return {
        "base_run_path": str(base_run_path).strip("/"),
        "base_instance_key_count": int(keys.shape[0]),
        "base_instance_key_sha256": instance_key_digest(keys),
        "base_instance_keys": keys,
    }


def create_delta_generation(
    root: Any,
    *,
    delta_run: str,
    generation: str,
    generation_ordinal: int,
    target_kind: DeltaKind,
    base_run_path: str,
    created_by: str,
) -> dict[str, Any]:
    """Create one open generation bound to an immutable base rowset."""

    run_name = _safe_component(delta_run, name="delta_run")
    generation_name = _safe_component(generation, name="generation")
    if target_kind not in {"keypoints", "detections"}:
        raise ValueError("target_kind must be 'keypoints' or 'detections'.")
    if int(generation_ordinal) < 0:
        raise ValueError("generation_ordinal must be non-negative.")
    binding = bind_base_run(root, base_run_path=base_run_path, target_kind=target_kind)

    parent = root.require_group(DELTA_PARENT)
    if run_name in parent:
        delta = parent[run_name]
        existing = {
            "target_kind": delta.attrs.get("target_kind"),
            "base_run_path": delta.attrs.get("base_run_path"),
            "base_instance_key_sha256": delta.attrs.get("base_instance_key_sha256"),
        }
        expected = {
            "target_kind": target_kind,
            "base_run_path": binding["base_run_path"],
            "base_instance_key_sha256": binding["base_instance_key_sha256"],
        }
        if existing != expected:
            raise ValueError(f"Existing delta run binding mismatch: {existing!r} != {expected!r}.")
    else:
        delta = parent.create_group(run_name)
        delta.attrs.update(
            {
                "schema": DELTA_RUN_SCHEMA,
                "delta_run": run_name,
                "target_kind": target_kind,
                "base_run_path": binding["base_run_path"],
                "base_instance_key_count": binding["base_instance_key_count"],
                "base_instance_key_sha256": binding["base_instance_key_sha256"],
                "operation_code_schema": OPERATION_CODE_SCHEMA,
                "created_at_utc": _utc_now(),
                "created_by": str(created_by),
                "status": "open",
            }
        )
    generations = delta.require_group("generations")
    if generation_name in generations:
        raise ValueError(f"Delta generation already exists: {run_name}/{generation_name}.")
    generation_group = generations.create_group(generation_name)
    generation_group.require_group("partitions")
    generation_group.attrs.update(
        {
            "schema": DELTA_GENERATION_SCHEMA,
            "generation": generation_name,
            "generation_ordinal": int(generation_ordinal),
            "target_kind": target_kind,
            "base_run_path": binding["base_run_path"],
            "base_instance_key_sha256": binding["base_instance_key_sha256"],
            "created_at_utc": _utc_now(),
            "created_by": str(created_by),
            "status": "open",
        }
    )
    delta.attrs["active_generation"] = generation_name
    delta.attrs["active_generation_ordinal"] = int(generation_ordinal)
    return {
        key: value for key, value in binding.items() if key != "base_instance_keys"
    } | {
        "delta_run": run_name,
        "generation": generation_name,
        "generation_ordinal": int(generation_ordinal),
        "target_kind": target_kind,
    }


def _normalize_common(
    *,
    instance_keys: Sequence[int] | np.ndarray,
    row_index_hints: Sequence[int] | np.ndarray,
    operation_codes: Sequence[int] | np.ndarray,
    revisions: Sequence[int] | np.ndarray,
    timestamp_ns: Sequence[int] | np.ndarray,
    reason_codes: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        "instance_key": np.asarray(instance_keys, dtype=np.uint64).reshape(-1),
        "row_index_hint": np.asarray(row_index_hints, dtype=np.int64).reshape(-1),
        "operation_codes": np.asarray(operation_codes, dtype=np.uint8).reshape(-1),
        "revision": np.asarray(revisions, dtype=np.uint64).reshape(-1),
        "timestamp_ns": np.asarray(timestamp_ns, dtype=np.int64).reshape(-1),
        "reason_codes": np.asarray(reason_codes, dtype=np.uint16).reshape(-1),
    }
    lengths = {name: int(values.shape[0]) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Delta common-array lengths differ: {lengths}.")
    if not lengths or next(iter(lengths.values())) <= 0:
        raise ValueError("Delta partitions must contain at least one row.")
    if np.any(arrays["revision"] == 0):
        raise ValueError("Delta revisions must be positive.")
    return arrays


def _write_array(group: Any, name: str, values: np.ndarray) -> None:
    data = np.asarray(values)
    chunks = (max(1, int(data.shape[0])), *tuple(int(value) for value in data.shape[1:]))
    group.create_array(name, data=data, chunks=chunks, overwrite=False)


def _partition_digest(arrays: Mapping[str, np.ndarray], attrs: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(attrs), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(arrays):
        values = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).view(np.uint8))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def write_delta_partition(
    root: Any,
    *,
    delta_run: str,
    generation: str,
    partition: str,
    editor: str,
    instance_keys: Sequence[int] | np.ndarray,
    row_index_hints: Sequence[int] | np.ndarray,
    operation_codes: Sequence[int] | np.ndarray,
    revisions: Sequence[int] | np.ndarray,
    timestamp_ns: Sequence[int] | np.ndarray,
    reason_codes: Sequence[int] | np.ndarray,
    keypoint_index: Sequence[int] | np.ndarray | None = None,
    new_xy: Sequence[Sequence[float]] | np.ndarray | None = None,
    valid: Sequence[bool] | np.ndarray | None = None,
    new_bbox_norm_coords: Sequence[Sequence[float]] | np.ndarray | None = None,
    reason_code_map: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Write one immutable delta partition; no shared array is appended."""

    run_name = _safe_component(delta_run, name="delta_run")
    generation_name = _safe_component(generation, name="generation")
    partition_name = _safe_component(partition, name="partition")
    delta = root[f"{DELTA_PARENT}/{run_name}"]
    generation_group = delta[f"generations/{generation_name}"]
    if str(generation_group.attrs.get("status")) != "open":
        raise ValueError(f"Delta generation {generation_name} is not open.")
    target_kind = str(delta.attrs.get("target_kind"))
    arrays = _normalize_common(
        instance_keys=instance_keys,
        row_index_hints=row_index_hints,
        operation_codes=operation_codes,
        revisions=revisions,
        timestamp_ns=timestamp_ns,
        reason_codes=reason_codes,
    )
    row_count = int(arrays["instance_key"].shape[0])
    base = bind_base_run(
        root,
        base_run_path=str(delta.attrs["base_run_path"]),
        target_kind=target_kind,  # type: ignore[arg-type]
    )
    if base["base_instance_key_sha256"] != delta.attrs.get("base_instance_key_sha256"):
        raise ValueError("Delta base instance-key fingerprint changed.")
    base_keys = np.asarray(base["base_instance_keys"], dtype=np.uint64)
    base_lookup = {int(value): idx for idx, value in enumerate(base_keys.tolist())}
    add_code = DETECTION_OPERATION_CODE_MAP["add_instance"]
    for idx, (key, hint, operation) in enumerate(
        zip(
            arrays["instance_key"].tolist(),
            arrays["row_index_hint"].tolist(),
            arrays["operation_codes"].tolist(),
            strict=True,
        )
    ):
        resolved = base_lookup.get(int(key))
        if target_kind == "detections" and int(operation) == add_code:
            if resolved is not None or int(hint) != -1:
                raise ValueError("add_instance requires a new instance_key and row_index_hint=-1.")
            continue
        if resolved is None:
            raise ValueError(f"Delta row {idx} instance_key is absent from the base run.")
        if int(hint) >= 0 and int(hint) != int(resolved):
            raise ValueError(
                f"Delta row {idx} row_index_hint={hint} does not resolve to its instance_key."
            )

    if target_kind == "keypoints":
        if keypoint_index is None or new_xy is None or valid is None:
            raise ValueError("Keypoint delta partitions require keypoint_index, new_xy, and valid.")
        arrays["keypoint_index"] = np.asarray(keypoint_index, dtype=np.int16).reshape(-1)
        arrays["new_xy"] = np.asarray(new_xy, dtype=np.float64).reshape(-1, 2)
        arrays["valid"] = np.asarray(valid, dtype=bool).reshape(-1)
        allowed = set(KEYPOINT_OPERATION_CODE_MAP.values())
    else:
        if new_bbox_norm_coords is None or valid is None:
            raise ValueError("Detection delta partitions require new_bbox_norm_coords and valid.")
        arrays["new_bbox_norm_coords"] = np.asarray(
            new_bbox_norm_coords, dtype=np.float64
        ).reshape(-1, 4)
        arrays["valid"] = np.asarray(valid, dtype=bool).reshape(-1)
        allowed = set(DETECTION_OPERATION_CODE_MAP.values())
    for name, values in arrays.items():
        if int(values.shape[0]) != row_count:
            raise ValueError(f"Delta array {name} has {values.shape[0]} rows; expected {row_count}.")
    unknown = sorted(set(int(value) for value in arrays["operation_codes"].tolist()) - allowed)
    if unknown:
        raise ValueError(f"Unsupported {target_kind} operation code(s): {unknown}.")
    if target_kind == "keypoints":
        replace_code = KEYPOINT_OPERATION_CODE_MAP["replace_xy"]
        set_valid_code = KEYPOINT_OPERATION_CODE_MAP["set_valid"]
        clear_code = KEYPOINT_OPERATION_CODE_MAP["clear_keypoint"]
        for row, (operation, xy, is_valid, index) in enumerate(
            zip(
                arrays["operation_codes"].tolist(),
                arrays["new_xy"],
                arrays["valid"].tolist(),
                arrays["keypoint_index"].tolist(),
                strict=True,
            )
        ):
            if int(index) < 0:
                raise ValueError(f"Keypoint delta row {row} has a negative keypoint_index.")
            finite_xy = bool(np.all(np.isfinite(xy)))
            nan_xy = bool(np.all(np.isnan(xy)))
            if int(operation) == replace_code and (not finite_xy or not is_valid):
                raise ValueError(f"replace_xy requires finite new_xy and valid=true; row {row} violates that contract.")
            if int(operation) == set_valid_code and not nan_xy:
                raise ValueError(f"set_valid requires NaN placeholder new_xy at row {row}.")
            if int(operation) == clear_code and (not nan_xy or is_valid):
                raise ValueError(f"clear_keypoint requires NaN placeholder new_xy and valid=false; row {row} violates that contract.")

    normalized_reason_map = {str(label): int(code) for label, code in dict(reason_code_map or {}).items()}
    if any(not label for label in normalized_reason_map):
        raise ValueError("Delta reason labels must be non-empty strings.")
    if any(code < 0 or code > 65535 for code in normalized_reason_map.values()):
        raise ValueError("Delta reason codes must fit uint16.")
    if len(set(normalized_reason_map.values())) != len(normalized_reason_map):
        raise ValueError("Delta reason-code values must be unique within a partition.")
    declared_reason_codes = set(normalized_reason_map.values()) | {0}
    unknown_reasons = sorted(set(int(value) for value in arrays["reason_codes"].tolist()) - declared_reason_codes)
    if unknown_reasons:
        raise ValueError(f"Delta reason code(s) are absent from reason_code_map: {unknown_reasons}.")

    partitions = generation_group["partitions"]
    if partition_name in partitions:
        raise ValueError(f"Delta partition already exists: {partition_name}.")
    attrs = {
        "schema": DELTA_PARTITION_SCHEMA,
        "delta_run": run_name,
        "generation": generation_name,
        "generation_ordinal": int(generation_group.attrs["generation_ordinal"]),
        "partition": partition_name,
        "target_kind": target_kind,
        "base_run_path": str(delta.attrs["base_run_path"]),
        "base_instance_key_sha256": str(delta.attrs["base_instance_key_sha256"]),
        "editor": str(editor),
        "created_at_utc": _utc_now(),
        "row_count": row_count,
        "operation_code_schema": OPERATION_CODE_SCHEMA,
        "operation_code_map": (KEYPOINT_OPERATION_CODE_MAP if target_kind == "keypoints" else DETECTION_OPERATION_CODE_MAP),
        "reason_code_map": normalized_reason_map,
        "merge_order": ["revision", "timestamp_ns", "partition", "partition_row_index"],
    }
    attrs["partition_sha256"] = _partition_digest(arrays, attrs)
    partition_group = partitions.create_group(partition_name)
    try:
        for name, values in arrays.items():
            _write_array(partition_group, name, values)
        partition_group.attrs.update(attrs)
    except Exception:
        # A partition becomes visible by name before its arrays are complete.
        # Remove that incomplete name so readers never mistake a torn write for
        # immutable review evidence.
        del partitions[partition_name]
        raise
    return dict(attrs)


_KEYPOINT_PARTITION_ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "instance_key": np.dtype(np.uint64),
    "row_index_hint": np.dtype(np.int64),
    "operation_codes": np.dtype(np.uint8),
    "revision": np.dtype(np.uint64),
    "timestamp_ns": np.dtype(np.int64),
    "reason_codes": np.dtype(np.uint16),
    "keypoint_index": np.dtype(np.int16),
    "new_xy": np.dtype(np.float64),
    "valid": np.dtype(bool),
}

_DELTA_PARTITION_ATTRIBUTE_NAMES = {
    "schema",
    "delta_run",
    "generation",
    "generation_ordinal",
    "partition",
    "target_kind",
    "base_run_path",
    "base_instance_key_sha256",
    "editor",
    "created_at_utc",
    "row_count",
    "operation_code_schema",
    "operation_code_map",
    "reason_code_map",
    "merge_order",
    "partition_sha256",
}


def _read_exact_keypoint_partition(
    partition_group: Any,
    *,
    partition_name: str,
    delta_run: str,
    generation: str,
    generation_ordinal: int,
    base_run_path: str,
    base_instance_key_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    attrs = dict(partition_group.attrs)
    unexpected_attrs = sorted(set(attrs) - _DELTA_PARTITION_ATTRIBUTE_NAMES)
    missing_attrs = sorted(_DELTA_PARTITION_ATTRIBUTE_NAMES - set(attrs))
    if unexpected_attrs or missing_attrs:
        raise ValueError(
            f"Delta partition {partition_name} attribute envelope differs from v1: missing={missing_attrs}, unexpected={unexpected_attrs}."
        )
    expected_attrs = {
        "schema": DELTA_PARTITION_SCHEMA,
        "delta_run": delta_run,
        "generation": generation,
        "generation_ordinal": int(generation_ordinal),
        "partition": partition_name,
        "target_kind": "keypoints",
        "base_run_path": base_run_path,
        "base_instance_key_sha256": base_instance_key_sha256,
        "operation_code_schema": OPERATION_CODE_SCHEMA,
        "operation_code_map": KEYPOINT_OPERATION_CODE_MAP,
        "merge_order": [
            "revision",
            "timestamp_ns",
            "partition",
            "partition_row_index",
        ],
    }
    for name, expected in expected_attrs.items():
        if attrs.get(name) != expected:
            raise ValueError(f"Delta partition {partition_name} has invalid {name}: {attrs.get(name)!r} != {expected!r}.")
    if not str(attrs.get("editor") or "").strip():
        raise ValueError(f"Delta partition {partition_name} has no editor identity.")

    array_names = {str(name) for name, _array in partition_group.arrays()}
    expected_array_names = set(_KEYPOINT_PARTITION_ARRAY_DTYPES)
    if array_names != expected_array_names:
        raise ValueError(
            f"Delta partition {partition_name} arrays differ from v1: missing={sorted(expected_array_names - array_names)}, unexpected={sorted(array_names - expected_array_names)}."
        )
    arrays: dict[str, np.ndarray] = {}
    row_count = int(attrs["row_count"])
    if row_count <= 0:
        raise ValueError(f"Delta partition {partition_name} must contain rows.")
    for name, expected_dtype in _KEYPOINT_PARTITION_ARRAY_DTYPES.items():
        array = partition_group[name]
        if np.dtype(array.dtype) != expected_dtype:
            raise ValueError(f"Delta partition {partition_name}/{name} dtype is {np.dtype(array.dtype)}, expected {expected_dtype}.")
        values = np.asarray(array[:])
        expected_shape = (row_count, 2) if name == "new_xy" else (row_count,)
        if values.shape != expected_shape:
            raise ValueError(f"Delta partition {partition_name}/{name} shape is {values.shape}, expected {expected_shape}.")
        arrays[name] = values

    digest_attrs = dict(attrs)
    stored_digest = str(digest_attrs.pop("partition_sha256") or "")
    computed_digest = _partition_digest(arrays, digest_attrs)
    if stored_digest != computed_digest:
        raise ValueError(f"Delta partition {partition_name} digest mismatch: {stored_digest!r} != {computed_digest!r}.")
    reason_map = attrs.get("reason_code_map")
    if not isinstance(reason_map, Mapping):
        raise ValueError(f"Delta partition {partition_name} reason_code_map is invalid.")
    normalized_reason_map = {str(label): int(code) for label, code in reason_map.items()}
    if any(not label for label in normalized_reason_map):
        raise ValueError(f"Delta partition {partition_name} has an empty reason label.")
    if len(set(normalized_reason_map.values())) != len(normalized_reason_map):
        raise ValueError(f"Delta partition {partition_name} reuses a reason code.")
    unknown_reasons = sorted(set(int(value) for value in arrays["reason_codes"].tolist()) - (set(normalized_reason_map.values()) | {0}))
    if unknown_reasons:
        raise ValueError(f"Delta partition {partition_name} uses undeclared reason codes {unknown_reasons}.")
    return arrays, attrs


def resolve_keypoint_delta_overlay(
    root: Any,
    *,
    delta_run: str,
    generation: str,
    n_keypoints: int,
) -> ResolvedKeypointDeltaOverlay:
    """Verify and resolve one keypoint delta generation against its base.

    Resolution is deterministic and uses the frozen v1 merge order.  The
    returned row indices are derived from ``instance_key``; persisted row hints
    are checked but never treated as authority.
    """

    run_name = _safe_component(delta_run, name="delta_run")
    generation_name = _safe_component(generation, name="generation")
    if type(n_keypoints) is not int or n_keypoints <= 0:
        raise ValueError("n_keypoints must be a positive exact integer.")
    delta = root[f"{DELTA_PARENT}/{run_name}"]
    if delta.attrs.get("schema") != DELTA_RUN_SCHEMA:
        raise ValueError(f"Delta run {run_name} has an unsupported schema.")
    if delta.attrs.get("target_kind") != "keypoints":
        raise ValueError(f"Delta run {run_name} does not target keypoints.")
    base_run_path = str(delta.attrs.get("base_run_path") or "")
    binding = bind_base_run(root, base_run_path=base_run_path, target_kind="keypoints")
    base_digest = str(delta.attrs.get("base_instance_key_sha256") or "")
    if binding["base_instance_key_sha256"] != base_digest:
        raise ValueError("Delta run base instance-key fingerprint changed.")
    base_keys = np.asarray(binding["base_instance_keys"], dtype=np.uint64)
    row_by_key = {int(value): row for row, value in enumerate(base_keys.tolist())}

    generation_group = delta[f"generations/{generation_name}"]
    if generation_group.attrs.get("schema") != DELTA_GENERATION_SCHEMA:
        raise ValueError(f"Delta generation {generation_name} has an unsupported schema.")
    status = str(generation_group.attrs.get("status") or "")
    if status not in {"open", "frozen", "compacted"}:
        raise ValueError(f"Delta generation {generation_name} has invalid status {status!r}.")
    generation_ordinal = int(generation_group.attrs.get("generation_ordinal", -1))
    if (
        generation_group.attrs.get("target_kind") != "keypoints"
        or generation_group.attrs.get("base_run_path") != base_run_path
        or generation_group.attrs.get("base_instance_key_sha256") != base_digest
    ):
        raise ValueError(f"Delta generation {generation_name} lost its base binding.")

    partitions = generation_group["partitions"]
    partition_names = sorted(str(name) for name, _group in partitions.groups())
    if status in {"frozen", "compacted"}:
        if generation_group.attrs.get("partition_names") != partition_names:
            raise ValueError("Frozen delta generation partition list changed.")
        generation_digest = hashlib.sha256()
        for partition_name in partition_names:
            generation_digest.update(partition_name.encode("utf-8"))
            generation_digest.update(str(partitions[partition_name].attrs.get("partition_sha256") or "").encode("ascii"))
        if generation_digest.hexdigest() != generation_group.attrs.get("generation_sha256"):
            raise ValueError("Frozen delta generation digest changed.")

    resolved: dict[tuple[int, int], tuple[tuple[int, int, str, int], ResolvedKeypointDelta]] = {}
    combined_reason_map: dict[str, int] = {}
    label_by_reason_code: dict[int, str] = {}
    event_count = 0
    max_revision = 0
    overlay_digest = hashlib.sha256()
    overlay_digest.update(run_name.encode("utf-8"))
    overlay_digest.update(generation_name.encode("utf-8"))
    overlay_digest.update(base_digest.encode("ascii"))

    for partition_name in partition_names:
        arrays, attrs = _read_exact_keypoint_partition(
            partitions[partition_name],
            partition_name=partition_name,
            delta_run=run_name,
            generation=generation_name,
            generation_ordinal=generation_ordinal,
            base_run_path=base_run_path,
            base_instance_key_sha256=base_digest,
        )
        overlay_digest.update(partition_name.encode("utf-8"))
        overlay_digest.update(str(attrs["partition_sha256"]).encode("ascii"))
        for label, code_value in dict(attrs["reason_code_map"]).items():
            label_text = str(label)
            code = int(code_value)
            if label_text in combined_reason_map and combined_reason_map[label_text] != code:
                raise ValueError(f"Reason label {label_text!r} changes code across partitions.")
            if code in label_by_reason_code and label_by_reason_code[code] != label_text:
                raise ValueError(f"Reason code {code} changes label across partitions.")
            combined_reason_map[label_text] = code
            label_by_reason_code[code] = label_text

        for partition_row in range(int(attrs["row_count"])):
            instance_key = int(arrays["instance_key"][partition_row])
            row_index = row_by_key.get(instance_key)
            if row_index is None:
                raise ValueError(f"Delta partition {partition_name} targets an unknown instance_key.")
            hint = int(arrays["row_index_hint"][partition_row])
            if hint >= 0 and hint != row_index:
                raise ValueError(f"Delta partition {partition_name} row hint no longer resolves by key.")
            keypoint_index = int(arrays["keypoint_index"][partition_row])
            if not 0 <= keypoint_index < n_keypoints:
                raise ValueError(f"Delta partition {partition_name} keypoint_index {keypoint_index} exceeds skeleton size {n_keypoints}.")
            operation = int(arrays["operation_codes"][partition_row])
            xy = np.asarray(arrays["new_xy"][partition_row], dtype=np.float64)
            valid = bool(arrays["valid"][partition_row])
            replace_code = KEYPOINT_OPERATION_CODE_MAP["replace_xy"]
            set_valid_code = KEYPOINT_OPERATION_CODE_MAP["set_valid"]
            clear_code = KEYPOINT_OPERATION_CODE_MAP["clear_keypoint"]
            if operation == replace_code and (not np.all(np.isfinite(xy)) or not valid):
                raise ValueError("Verified replace_xy payload is not finite and valid.")
            if operation == set_valid_code and not np.all(np.isnan(xy)):
                raise ValueError("Verified set_valid payload has non-placeholder coordinates.")
            if operation == clear_code and (not np.all(np.isnan(xy)) or valid):
                raise ValueError("Verified clear_keypoint payload is not clear/invalid.")
            revision = int(arrays["revision"][partition_row])
            timestamp = int(arrays["timestamp_ns"][partition_row])
            order = (revision, timestamp, partition_name, partition_row)
            edit = ResolvedKeypointDelta(
                instance_key=instance_key,
                row_index=row_index,
                keypoint_index=keypoint_index,
                operation_code=operation,
                revision=revision,
                timestamp_ns=timestamp,
                partition=partition_name,
                partition_row_index=partition_row,
                reason_code=int(arrays["reason_codes"][partition_row]),
                new_xy=(float(xy[0]), float(xy[1])),
                valid=valid,
            )
            target = (instance_key, keypoint_index)
            previous = resolved.get(target)
            if previous is None or order > previous[0]:
                resolved[target] = (order, edit)
            event_count += 1
            max_revision = max(max_revision, revision)

    edits = tuple(item[1] for _target, item in sorted(resolved.items(), key=lambda value: (value[0][0], value[0][1])))
    return ResolvedKeypointDeltaOverlay(
        delta_run=run_name,
        generation=generation_name,
        generation_status=status,
        base_run_path=base_run_path,
        base_instance_key_sha256=base_digest,
        partition_count=len(partition_names),
        event_count=event_count,
        max_revision=max_revision,
        overlay_sha256=overlay_digest.hexdigest(),
        reason_code_map=dict(sorted(combined_reason_map.items())),
        edits=edits,
    )


def apply_keypoint_delta_overlay(
    base_keypoints_roi: Any,
    *,
    instance_keys: Any,
    overlay: ResolvedKeypointDeltaOverlay,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize verified landmark edits without mutating the base array."""

    points = np.asarray(base_keypoints_roi[:]).copy()
    keys = np.asarray(instance_keys[:], dtype=np.uint64).reshape(-1)
    if points.ndim != 3 or points.shape[0] != keys.shape[0] or points.shape[2] != 2:
        raise ValueError("Base keypoint coordinates and instance keys are incompatible.")
    valid = np.all(np.isfinite(points), axis=2)
    replace_code = KEYPOINT_OPERATION_CODE_MAP["replace_xy"]
    set_valid_code = KEYPOINT_OPERATION_CODE_MAP["set_valid"]
    clear_code = KEYPOINT_OPERATION_CODE_MAP["clear_keypoint"]
    for edit in overlay.edits:
        if int(keys[edit.row_index]) != edit.instance_key:
            raise ValueError("Resolved keypoint edit row no longer matches instance_key.")
        target = (edit.row_index, edit.keypoint_index)
        if edit.operation_code == replace_code:
            points[target] = np.asarray(edit.new_xy, dtype=points.dtype)
            valid[target] = True
        elif edit.operation_code == set_valid_code:
            if edit.valid and not np.all(np.isfinite(points[target])):
                raise ValueError("set_valid cannot restore a landmark without coordinates.")
            if not edit.valid:
                points[target] = np.asarray([np.nan, np.nan], dtype=points.dtype)
            valid[target] = edit.valid
        elif edit.operation_code == clear_code:
            points[target] = np.asarray([np.nan, np.nan], dtype=points.dtype)
            valid[target] = False
        else:  # pragma: no cover - verified above
            raise ValueError(f"Unsupported resolved operation {edit.operation_code}.")
    return points, valid


def freeze_delta_generation(
    root: Any,
    *,
    delta_run: str,
    generation: str,
    frozen_by: str,
) -> dict[str, Any]:
    """Freeze one generation as the fixed input snapshot for compaction."""

    delta = root[f"{DELTA_PARENT}/{_safe_component(delta_run, name='delta_run')}"]
    generation_name = _safe_component(generation, name="generation")
    group = delta[f"generations/{generation_name}"]
    if str(group.attrs.get("status")) != "open":
        raise ValueError(f"Generation {generation_name} is not open.")
    partition_names = sorted(str(name) for name, _child in group["partitions"].groups())
    if not partition_names:
        raise ValueError("Cannot freeze an empty delta generation.")
    digest = hashlib.sha256()
    for name in partition_names:
        partition = group[f"partitions/{name}"]
        digest.update(name.encode("utf-8"))
        digest.update(str(partition.attrs.get("partition_sha256") or "").encode("ascii"))
    frozen_at = _utc_now()
    group.attrs.update(
        {
            "status": "frozen",
            "frozen_at_utc": frozen_at,
            "frozen_by": str(frozen_by),
            "partition_count": len(partition_names),
            "partition_names": partition_names,
            "generation_sha256": digest.hexdigest(),
        }
    )
    if delta.attrs.get("active_generation") == generation_name:
        delta.attrs["active_generation"] = None
    return {
        "delta_run": str(delta_run),
        "generation": generation_name,
        "status": "frozen",
        "partition_count": len(partition_names),
        "generation_sha256": digest.hexdigest(),
        "frozen_at_utc": frozen_at,
    }


__all__ = [
    "DELTA_GENERATION_SCHEMA",
    "DELTA_PARENT",
    "DELTA_PARTITION_SCHEMA",
    "DELTA_RUN_SCHEMA",
    "DETECTION_OPERATION_CODE_MAP",
    "KEYPOINT_OPERATION_CODE_MAP",
    "KEYPOINT_DELTA_REASON_CODE_MAP",
    "OPERATION_CODE_SCHEMA",
    "ResolvedKeypointDelta",
    "ResolvedKeypointDeltaOverlay",
    "apply_keypoint_delta_overlay",
    "bind_base_run",
    "create_delta_generation",
    "freeze_delta_generation",
    "instance_key_digest",
    "resolve_keypoint_delta_overlay",
    "write_delta_partition",
]
