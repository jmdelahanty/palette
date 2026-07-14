"""Sparse, partition-owned edit deltas for immutable tabular snapshots."""

from __future__ import annotations

import hashlib
import json
import re
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
DETECTION_OPERATION_CODE_MAP: dict[str, int] = {
    "replace_bbox": 1,
    "delete_instance": 2,
    "restore_instance": 3,
    "add_instance": 4,
}
OPERATION_CODE_SCHEMA = "palette.tabular_delta_operation_codes.v1"

DeltaKind = Literal["keypoints", "detections"]
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


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
        "operation_code_map": (
            KEYPOINT_OPERATION_CODE_MAP
            if target_kind == "keypoints"
            else DETECTION_OPERATION_CODE_MAP
        ),
        "reason_code_map": dict(reason_code_map or {}),
        "merge_order": ["revision", "timestamp_ns", "partition", "partition_row_index"],
    }
    attrs["partition_sha256"] = _partition_digest(arrays, attrs)
    partition_group = partitions.create_group(partition_name)
    for name, values in arrays.items():
        _write_array(partition_group, name, values)
    partition_group.attrs.update(attrs)
    return dict(attrs)


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
    "OPERATION_CODE_SCHEMA",
    "bind_base_run",
    "create_delta_generation",
    "freeze_delta_generation",
    "instance_key_digest",
    "write_delta_partition",
]
