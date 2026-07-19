#!/usr/bin/env python3
"""Publish a depth-one raw subject-mask snapshot from keyed replacements."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from fisheye.refinement.finalize_subject_masks import _SUBJECT_MASK_SHARD_COMPAT_ATTRS
from fisheye.shared.crop_snapshot_identity import (
    CropSignatureSnapshot,
    CropSnapshotIdentityError,
    crop_detection_source,
    crop_frame_counts,
    require_crop_snapshot,
    resolve_crop_source_signatures,
)
from fisheye.shared.composite_subject_mask import (
    COMPOSITE_SUBJECT_MASK_PAYLOAD_GROUP,
    COMPOSITE_SUBJECT_MASK_SCHEMA_ID,
    COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION,
    COMPOSITE_SUBJECT_MASK_SOURCE_BASE,
    COMPOSITE_SUBJECT_MASK_SOURCE_DELTA,
    COMPOSITE_SUBJECT_MASK_STORAGE_MODE,
    validate_composite_subject_mask_run,
)
from fisheye.shared.keyed_replacement import (
    REPLACEMENT_SOURCE_BASE,
    KeyedReplacementPlan,
    build_keyed_replacement_plan,
)
from fisheye.shared.row_lineage import ROW_IDENTITY_MODE_SCHEMA
from fisheye.shared.row_source_signature import (
    ROW_SOURCE_SIGNATURE_ARRAY,
    load_row_source_signature_spec,
    validate_row_source_signature_array,
)
from fisheye.shared.run_provenance import build_writer_run_provenance, json_ready, stable_json
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array
from fisheye.shared.zarr.columnar import pick_shards, store_array
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_LATEST_COMPLETE_ATTR,
    RUN_PROVENANCE_ATTR,
    is_run_complete_in_parent,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)


SUBJECT_MASK_PARENT = "subject_mask_runs"
SUBJECT_MASK_SHARD_PARENT = "subject_mask_shard_runs"
REFINED_SUBJECT_MASK_PARENT = "refined_subject_masks_runs"
SUBJECT_MASK_COMPACTION_METHOD = "fisheye.utils.compact_subject_mask_deltas"
SUBJECT_MASK_COMPACTION_SCHEMA_ID = "palette.subject_mask_keyed_compaction"
SUBJECT_MASK_COMPACTION_SCHEMA_VERSION = 1
DEFAULT_TABULAR_SHARD_ROWS = 131_072

_SEMANTIC_ATTRS = tuple(
    dict.fromkeys(
        (
            *_SUBJECT_MASK_SHARD_COMPAT_ATTRS,
            "probabilities_dtype",
            "probability_semantics",
            "output_semantics",
            "overlap_policy",
            "model_input_transform",
            "model_input_transform_name",
            "model_input_shape_hw",
            "native_roi_shape_hw",
            "source_keypoints_run",
            "source_keypoint_group",
            "assignment_keypoints_run",
            "assignment_keypoint_group",
        )
    )
)
_REQUIRED_SEMANTIC_ATTRS = (
    "mask_labels",
    "label_schema_id",
    "probabilities_encoding",
    "mask_probability_threshold",
)
_TARGET_LINEAGE = (
    "frame_indices",
    "source_frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
    "detection_indices",
    "detection_source",
    "instance_key",
)
_BASE_MANIFEST_LINEAGE = (
    "frame_indices",
    "source_frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
    "detection_source",
)


class SubjectMaskCompactionError(RuntimeError):
    """Raised when a composite mask snapshot cannot be proven exact."""


@dataclass(frozen=True)
class _MaskBaseCollection:
    """One logical immutable base backed by one or more physical raw runs."""

    manifest_path: str
    manifest_group: Any
    run_paths: tuple[str, ...]
    groups: tuple[Any, ...]
    instance_keys: np.ndarray
    source_signatures: np.ndarray
    signature_spec_digest: str
    signature_modes: tuple[str, ...]
    instance_key_modes: tuple[str, ...]
    flat_run_indices: np.ndarray
    flat_local_rows: np.ndarray

    @property
    def primary(self) -> Any:
        return self.groups[0]


def _require_complete(parent: Any, name: str, *, label: str) -> Any:
    if name not in parent:
        raise SubjectMaskCompactionError(f"{label} {name!r} does not exist.")
    group = parent[name]
    if not is_run_complete_in_parent(parent, group):
        raise SubjectMaskCompactionError(f"{label} {name!r} is not complete.")
    return group


def _keys(group: Any, *, label: str, rows: int) -> np.ndarray:
    if "instance_key" not in group:
        raise SubjectMaskCompactionError(f"{label} lacks instance_key.")
    array = group["instance_key"]
    if np.dtype(array.dtype) != np.dtype(np.uint64) or tuple(array.shape) != (rows,):
        raise SubjectMaskCompactionError(f"{label}/instance_key must be uint64[{rows}].")
    values = np.asarray(array[:], dtype=np.uint64).reshape(-1)
    if np.unique(values).shape[0] != values.shape[0]:
        raise SubjectMaskCompactionError(f"{label}/instance_key is not unique.")
    return values


def _bound_instance_keys(
    root: Any,
    run: Any,
    *,
    label: str,
    rows: int,
) -> tuple[np.ndarray, str]:
    """Load modern keys or derive historical raw-shard keys from its crop."""

    crop_name = str(run.attrs.get("source_crop_run") or "").strip()
    crop_parent = root.get("crop_runs")
    if not crop_name or crop_parent is None:
        raise SubjectMaskCompactionError(f"{label} has no source_crop_run.")
    try:
        crop = require_crop_snapshot(crop_parent, crop_name, label="Source crop run")
    except CropSnapshotIdentityError as exc:
        raise SubjectMaskCompactionError(str(exc)) from exc
    crop_rows = int(crop["frame_indices"].shape[0]) if "frame_indices" in crop else -1
    if "source_crop_row_ids" not in run:
        raise SubjectMaskCompactionError(f"{label} lacks source_crop_row_ids.")
    source_rows = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64).reshape(-1)
    if source_rows.shape != (rows,) or (
        source_rows.size
        and (source_rows.min() < 0 or source_rows.max() >= crop_rows)
    ):
        raise SubjectMaskCompactionError(f"{label} references invalid source crop rows.")
    actual = _keys(run, label=label, rows=rows) if "instance_key" in run else None
    if "instance_key" in crop:
        crop_keys = _keys(crop, label=f"crop_runs/{crop_name}", rows=crop_rows)
    elif actual is not None and np.array_equal(
        np.sort(source_rows),
        np.arange(crop_rows, dtype=np.int64),
    ):
        crop_keys = np.empty(crop_rows, dtype=np.uint64)
        crop_keys[source_rows] = actual
    else:
        raise SubjectMaskCompactionError(
            f"{label} cannot establish keys for keyless crop_runs/{crop_name}."
        )
    expected = crop_keys[source_rows]
    if actual is None:
        return expected, "derived_from_bound_crop_rows_v1"
    if not np.array_equal(actual, expected):
        raise SubjectMaskCompactionError(
            f"{label} instance_key does not match its bound crop rows."
        )
    return actual, "persisted"


def _source_signatures(
    root: Any,
    run: Any,
    *,
    label: str,
) -> tuple[np.ndarray, str, str]:
    crop_name = str(run.attrs.get("source_crop_run") or "").strip()
    crop_parent = root.get("crop_runs")
    if not crop_name or crop_parent is None:
        raise SubjectMaskCompactionError(f"{label} has no source_crop_run.")
    try:
        crop = require_crop_snapshot(crop_parent, crop_name, label="Source crop run")
    except CropSnapshotIdentityError as exc:
        raise SubjectMaskCompactionError(str(exc)) from exc
    if "source_crop_row_ids" not in run:
        raise SubjectMaskCompactionError(f"{label} lacks source_crop_row_ids.")
    source_rows = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64).reshape(-1)
    run_keys, _ = _bound_instance_keys(
        root,
        run,
        label=label,
        rows=source_rows.shape[0],
    )
    crop_rows = int(crop["frame_indices"].shape[0]) if "frame_indices" in crop else -1
    if "instance_key" in crop:
        crop_keys = _keys(crop, label=f"crop_runs/{crop_name}", rows=crop_rows)
    else:
        if not np.array_equal(
            np.sort(source_rows),
            np.arange(crop_rows, dtype=np.int64),
        ):
            raise SubjectMaskCompactionError(
                f"{label} cannot supply complete keys for crop_runs/{crop_name}."
            )
        crop_keys = np.empty(crop_rows, dtype=np.uint64)
        crop_keys[source_rows] = run_keys
    try:
        snapshot = resolve_crop_source_signatures(
            root,
            crop,
            label=f"crop_runs/{crop_name}",
            instance_keys=crop_keys,
        )
    except CropSnapshotIdentityError as exc:
        raise SubjectMaskCompactionError(str(exc)) from exc
    if source_rows.size and (source_rows.min() < 0 or source_rows.max() >= crop_rows):
        raise SubjectMaskCompactionError(f"{label} references an out-of-range crop row.")
    if not np.array_equal(run_keys, crop_keys[source_rows]):
        raise SubjectMaskCompactionError(f"{label} keys do not match its source crop rows.")
    signatures = snapshot.signatures[source_rows]
    return signatures, snapshot.spec.spec_digest, snapshot.mode


def _input_artifact_sha256(group: Any, *, role: str, label: str) -> str:
    fingerprints: set[str] = set()
    for attr_name in ("run_provenance", "cli_provenance"):
        provenance = group.attrs.get(attr_name)
        if not isinstance(provenance, dict):
            continue
        artifacts = provenance.get("input_artifacts")
        if not isinstance(artifacts, (list, tuple)):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict) or str(artifact.get("role") or "") != role:
                continue
            sha256 = str(artifact.get("sha256") or "").strip().lower()
            if len(sha256) == 64:
                fingerprints.add(sha256)
    if len(fingerprints) != 1:
        raise SubjectMaskCompactionError(
            f"{label} must expose exactly one {role!r} SHA-256 fingerprint; found {len(fingerprints)}."
        )
    return next(iter(fingerprints))


def _resolve_raw_run_path(root: Any, path: str, *, label: str) -> Any:
    parts = [part for part in str(path).strip().strip("/").split("/") if part]
    if len(parts) != 2 or parts[0] not in {SUBJECT_MASK_PARENT, SUBJECT_MASK_SHARD_PARENT}:
        raise SubjectMaskCompactionError(
            f"{label} path {path!r} must name one raw subject-mask run."
        )
    parent = root.get(parts[0])
    if parent is None:
        raise SubjectMaskCompactionError(f"Missing raw subject-mask parent {parts[0]!r}.")
    return _require_complete(parent, parts[1], label=label)


def _resolve_base_collection(root: Any, base_run: str) -> _MaskBaseCollection:
    mask_parent = root.get(SUBJECT_MASK_PARENT)
    refined_parent = root.get(REFINED_SUBJECT_MASK_PARENT)
    manifest_path: str
    manifest: Any
    run_paths: list[str]
    if mask_parent is not None and base_run in mask_parent:
        manifest = _require_complete(
            mask_parent,
            base_run,
            label="Base subject-mask run",
        )
        manifest_path = f"{SUBJECT_MASK_PARENT}/{base_run}"
        run_paths = [manifest_path]
    elif refined_parent is not None and base_run in refined_parent:
        manifest = _require_complete(
            refined_parent,
            base_run,
            label="Base refined subject-mask collection manifest",
        )
        manifest_path = f"{REFINED_SUBJECT_MASK_PARENT}/{base_run}"
        raw_paths = manifest.attrs.get("source_subject_mask_shard_run_paths")
        if isinstance(raw_paths, (list, tuple)) and raw_paths:
            run_paths = [str(value).strip().strip("/") for value in raw_paths]
        else:
            raw_names = manifest.attrs.get("source_subject_mask_shard_runs")
            if not isinstance(raw_names, (list, tuple)) or not raw_names:
                raise SubjectMaskCompactionError(
                    "Refined base manifest lacks source subject-mask shard paths."
                )
            run_paths = [f"{SUBJECT_MASK_SHARD_PARENT}/{value}" for value in raw_names]
    else:
        raise SubjectMaskCompactionError(
            f"Base run {base_run!r} is absent from both raw and refined subject-mask parents."
        )
    if len(set(run_paths)) != len(run_paths):
        raise SubjectMaskCompactionError("Base collection declares duplicate raw run paths.")
    groups = tuple(
        _resolve_raw_run_path(root, path, label="Base subject-mask source")
        for path in run_paths
    )
    if not groups:
        raise SubjectMaskCompactionError("Base subject-mask collection is empty.")

    refined_manifest_identity: tuple[np.ndarray, np.ndarray, str, str] | None = None
    if manifest_path.startswith(f"{REFINED_SUBJECT_MASK_PARENT}/"):
        if "instance_key" not in manifest:
            raise SubjectMaskCompactionError(
                "Refined base manifest lacks instance_key."
            )
        manifest_rows = int(manifest["instance_key"].shape[0])
        manifest_keys = _keys(
            manifest,
            label=manifest_path,
            rows=manifest_rows,
        )
        manifest_signatures, manifest_digest, manifest_signature_mode = (
            _source_signatures(root, manifest, label=manifest_path)
        )
        source_rows_total = sum(
            int(group["mask_probs_roi"].shape[0]) for group in groups
        )
        if source_rows_total != manifest_rows:
            raise SubjectMaskCompactionError(
                "Refined base manifest row count differs from its declared raw shards."
            )
        refined_manifest_identity = (
            manifest_keys,
            manifest_signatures,
            manifest_digest,
            manifest_signature_mode,
        )

    key_parts: list[np.ndarray] = []
    signature_parts: list[np.ndarray] = []
    signature_digests: set[str] = set()
    signature_modes: list[str] = []
    instance_key_modes: list[str] = []
    flat_run_indices: list[np.ndarray] = []
    flat_local_rows: list[np.ndarray] = []
    manifest_start = 0
    for run_index, (path, group) in enumerate(zip(run_paths, groups, strict=True)):
        if str(group.attrs.get("subject_mask_storage_mode") or "") == COMPOSITE_SUBJECT_MASK_STORAGE_MODE:
            raise SubjectMaskCompactionError(
                f"Base source {path!r} is composite; reference chains are forbidden."
            )
        if "mask_probs_roi" not in group:
            raise SubjectMaskCompactionError(f"Base source {path!r} lacks mask_probs_roi.")
        rows = int(group["mask_probs_roi"].shape[0])
        if refined_manifest_identity is None:
            keys, key_mode = _bound_instance_keys(
                root,
                group,
                label=path,
                rows=rows,
            )
            signatures, digest, mode = _source_signatures(
                root,
                group,
                label=path,
            )
        else:
            manifest_stop = manifest_start + rows
            keys = refined_manifest_identity[0][manifest_start:manifest_stop]
            signatures = refined_manifest_identity[1][manifest_start:manifest_stop]
            digest = refined_manifest_identity[2]
            mode = (
                "exact_refined_collection_manifest_v1:"
                f"{refined_manifest_identity[3]}"
            )
            key_mode = "derived_from_exact_refined_collection_manifest_v1"
            compared = 0
            for lineage_name in _BASE_MANIFEST_LINEAGE:
                if lineage_name not in manifest or lineage_name not in group:
                    continue
                manifest_values = np.asarray(
                    manifest[lineage_name][manifest_start:manifest_stop]
                )
                source_values = np.asarray(group[lineage_name][:])
                if not np.array_equal(manifest_values, source_values):
                    raise SubjectMaskCompactionError(
                        f"Base source {path!r} does not match the refined manifest "
                        f"for lineage {lineage_name!r}."
                    )
                compared += 1
            if compared < 4:
                raise SubjectMaskCompactionError(
                    f"Base source {path!r} has insufficient shared lineage with its manifest."
                )
            manifest_start = manifest_stop
        key_parts.append(keys)
        signature_parts.append(signatures)
        signature_digests.add(digest)
        signature_modes.append(mode)
        instance_key_modes.append(key_mode)
        flat_run_indices.append(np.full(rows, run_index, dtype=np.int32))
        flat_local_rows.append(np.arange(rows, dtype=np.int64))
    if len(signature_digests) != 1:
        raise SubjectMaskCompactionError(
            "Base subject-mask shards use incompatible crop signature specifications."
        )
    keys = np.concatenate(key_parts) if key_parts else np.empty(0, dtype=np.uint64)
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise SubjectMaskCompactionError(
            "Base subject-mask collection contains duplicate instance_key values."
        )
    return _MaskBaseCollection(
        manifest_path=manifest_path,
        manifest_group=manifest,
        run_paths=tuple(run_paths),
        groups=groups,
        instance_keys=keys,
        source_signatures=(
            np.concatenate(signature_parts, axis=0)
            if signature_parts
            else np.empty((0, 32), dtype=np.uint8)
        ),
        signature_spec_digest=next(iter(signature_digests)),
        signature_modes=tuple(signature_modes),
        instance_key_modes=tuple(instance_key_modes),
        flat_run_indices=np.concatenate(flat_run_indices),
        flat_local_rows=np.concatenate(flat_local_rows),
    )


def _validate_semantics(base: _MaskBaseCollection, replacements: Sequence[Any]) -> str:
    primary = base.primary
    for name in _REQUIRED_SEMANTIC_ATTRS:
        if name not in primary.attrs:
            raise SubjectMaskCompactionError(f"Base subject-mask run lacks attr {name!r}.")
    if "available_channels" not in primary or "mask_probs_roi" not in primary:
        raise SubjectMaskCompactionError("Base subject-mask run lacks probability/channel payload.")
    expected_available = np.asarray(primary["available_channels"][:], dtype=bool)
    expected_surface = primary["mask_probs_roi"]
    model_fingerprints: set[str] = set()
    if len(expected_surface.shape) != 4:
        raise SubjectMaskCompactionError("Base mask_probs_roi must have shape (N,C,H,W).")
    for index, source in enumerate(base.groups):
        for name in _REQUIRED_SEMANTIC_ATTRS:
            if name not in source.attrs:
                raise SubjectMaskCompactionError(
                    f"Base subject-mask source {index} lacks attr {name!r}."
                )
        for name in _SEMANTIC_ATTRS:
            if stable_json(primary.attrs.get(name)) != stable_json(source.attrs.get(name)):
                raise SubjectMaskCompactionError(
                    f"Base subject-mask sources differ in semantic attr {name!r}."
                )
        if "available_channels" not in source or not np.array_equal(
            np.asarray(source["available_channels"][:], dtype=bool),
            expected_available,
        ):
            raise SubjectMaskCompactionError(
                f"Base subject-mask source {index} available channels differ."
            )
        surface = source["mask_probs_roi"]
        if (
            tuple(surface.shape[1:]) != tuple(expected_surface.shape[1:])
            or np.dtype(surface.dtype) != np.dtype(expected_surface.dtype)
        ):
            raise SubjectMaskCompactionError(
                f"Base subject-mask source {index} probability shape or dtype differs."
            )
        model_fingerprints.add(
            _input_artifact_sha256(
                source,
                role="subject_mask_unet_checkpoint",
                label=f"Base subject-mask source {index}",
            )
        )
    if len(model_fingerprints) != 1:
        raise SubjectMaskCompactionError(
            "Base subject-mask sources used different model fingerprints."
        )
    model_sha256 = next(iter(model_fingerprints))
    for index, replacement in enumerate(replacements):
        for name in _SEMANTIC_ATTRS:
            if name in _REQUIRED_SEMANTIC_ATTRS and name not in replacement.attrs:
                raise SubjectMaskCompactionError(f"Replacement {index} lacks attr {name!r}.")
            if stable_json(primary.attrs.get(name)) != stable_json(
                replacement.attrs.get(name)
            ):
                raise SubjectMaskCompactionError(
                    f"Subject-mask semantic attr {name!r} differs for replacement {index}."
                )
        if "available_channels" not in replacement or not np.array_equal(
            np.asarray(replacement["available_channels"][:], dtype=bool), expected_available
        ):
            raise SubjectMaskCompactionError(f"Replacement {index} available channels differ.")
        if "mask_probs_roi" not in replacement:
            raise SubjectMaskCompactionError(f"Replacement {index} lacks mask_probs_roi.")
        surface = replacement["mask_probs_roi"]
        if tuple(surface.shape[1:]) != tuple(expected_surface.shape[1:]) or np.dtype(surface.dtype) != np.dtype(expected_surface.dtype):
            raise SubjectMaskCompactionError(f"Replacement {index} probability shape or dtype differs.")
        if _input_artifact_sha256(
            replacement,
            role="subject_mask_unet_checkpoint",
            label=f"Replacement {index}",
        ) != model_sha256:
            raise SubjectMaskCompactionError(
                f"Subject-mask model fingerprint differs for replacement {index}."
            )
    return model_sha256


def _resolve_replacements(
    root: Any,
    names: Sequence[str],
    target: Any,
    target_name: str,
    target_keys: np.ndarray,
    target_signature_snapshot: CropSignatureSnapshot,
) -> list[Any]:
    parent = root.get(SUBJECT_MASK_SHARD_PARENT)
    if names and parent is None:
        raise SubjectMaskCompactionError(f"Missing {SUBJECT_MASK_SHARD_PARENT}.")
    if len(set(names)) != len(names):
        raise SubjectMaskCompactionError("Duplicate replacement run names were supplied.")
    target_keys = np.asarray(target_keys, dtype=np.uint64).reshape(-1)
    groups: list[Any] = []
    for name in names:
        group = _require_complete(parent, str(name), label="Replacement subject-mask run")
        if normalize_attr(group.attrs.get("incremental_materialization_role")) != "delta_replacement_rows":
            raise SubjectMaskCompactionError(
                f"{SUBJECT_MASK_SHARD_PARENT}/{name} is not a delta replacement run."
            )
        if str(group.attrs.get("source_crop_run") or "") != target_name:
            raise SubjectMaskCompactionError(
                f"{SUBJECT_MASK_SHARD_PARENT}/{name} is not bound to crop_runs/{target_name}."
            )
        rows = int(group["instance_key"].shape[0]) if "instance_key" in group else -1
        keys = _keys(group, label=f"{SUBJECT_MASK_SHARD_PARENT}/{name}", rows=rows)
        if "source_crop_row_ids" not in group:
            raise SubjectMaskCompactionError(f"{SUBJECT_MASK_SHARD_PARENT}/{name} lacks crop rows.")
        target_rows = np.asarray(group["source_crop_row_ids"][:], dtype=np.int64).reshape(-1)
        if target_rows.shape != (rows,) or (target_rows.size and (target_rows.min() < 0 or target_rows.max() >= target_keys.shape[0])):
            raise SubjectMaskCompactionError(f"{SUBJECT_MASK_SHARD_PARENT}/{name} crop rows are invalid.")
        if not np.array_equal(keys, target_keys[target_rows]):
            raise SubjectMaskCompactionError(
                f"{SUBJECT_MASK_SHARD_PARENT}/{name} keys do not match target crop rows."
            )
        if ROW_SOURCE_SIGNATURE_ARRAY not in group:
            raise SubjectMaskCompactionError(
                f"{SUBJECT_MASK_SHARD_PARENT}/{name} lacks the signed crop rows used for inference."
            )
        validate_row_source_signature_array(
            group[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=rows
        )
        replacement_spec = load_row_source_signature_spec(group.attrs)
        replacement_signatures = np.asarray(
            group[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8
        )
        current_signatures = target_signature_snapshot.signatures[target_rows]
        if (
            replacement_spec.spec_digest
            != target_signature_snapshot.spec.spec_digest
            or not np.array_equal(replacement_signatures, current_signatures)
        ):
            raise SubjectMaskCompactionError(
                f"{SUBJECT_MASK_SHARD_PARENT}/{name} was inferred from stale or incompatible crop rows."
            )
        groups.append(group)
    return groups


def _read_rows_into(array: Any, source_rows: np.ndarray, output: np.ndarray, output_rows: np.ndarray) -> None:
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    destinations = np.asarray(output_rows, dtype=np.int64).reshape(-1)
    if not rows.size:
        return
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    destinations = destinations[order]
    start = 0
    while start < rows.shape[0]:
        stop = start + 1
        while stop < rows.shape[0] and rows[stop] == rows[stop - 1] + 1:
            stop += 1
        first = int(rows[start])
        last = int(rows[stop - 1]) + 1
        values = np.asarray(array[first:last])
        output[destinations[start:stop]] = values[rows[start:stop] - first]
        start = stop


def _gather_replacement_rows(
    name: str,
    *,
    replacements: Sequence[Any],
    plan: KeyedReplacementPlan,
    target_rows: np.ndarray,
    empty_template: Any | None = None,
) -> np.ndarray:
    target_rows = np.asarray(target_rows, dtype=np.int64).reshape(-1)
    if not target_rows.size:
        source = replacements[0][name] if replacements else empty_template
        if source is None:
            raise SubjectMaskCompactionError(
                "Cannot determine the empty replacement payload shape."
            )
        return np.empty((0, *tuple(source.shape[1:])), dtype=source.dtype)
    first_run = int(plan.source_run_indices[target_rows[0]])
    if first_run < 0:
        raise SubjectMaskCompactionError("Delta payload unexpectedly references the base.")
    source = replacements[first_run][name]
    output = np.empty((target_rows.shape[0], *tuple(int(v) for v in source.shape[1:])), dtype=source.dtype)
    run_indices = plan.source_run_indices[target_rows]
    source_rows = plan.source_row_indices[target_rows]
    for run_index, replacement in enumerate(replacements):
        positions = np.flatnonzero(run_indices == run_index)
        _read_rows_into(replacement[name], source_rows[positions], output, positions)
    return output


def _gather_all_rows(
    name: str,
    *,
    base: _MaskBaseCollection,
    replacements: Sequence[Any],
    plan: KeyedReplacementPlan,
    start: int,
    stop: int,
) -> np.ndarray:
    source = base.primary[name]
    output = np.empty((stop - start, *tuple(int(v) for v in source.shape[1:])), dtype=source.dtype)
    run_indices = plan.source_run_indices[start:stop]
    source_rows = plan.source_row_indices[start:stop]
    positions = np.flatnonzero(run_indices == REPLACEMENT_SOURCE_BASE)
    flat_rows = source_rows[positions]
    for base_run_index, group in enumerate(base.groups):
        within = np.flatnonzero(base.flat_run_indices[flat_rows] == base_run_index)
        if not within.size:
            continue
        _read_rows_into(
            group[name],
            base.flat_local_rows[flat_rows[within]],
            output,
            positions[within],
        )
    for run_index, replacement in enumerate(replacements):
        positions = np.flatnonzero(run_indices == run_index)
        _read_rows_into(replacement[name], source_rows[positions], output, positions)
    return output


def _surface_layout(source: Any, row_count: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    chunks = tuple(int(value) for value in source.chunks)
    raw_shards = getattr(source, "shards", None)
    shards = tuple(int(value) for value in raw_shards) if raw_shards else None
    if shards is not None:
        logical_chunks = max(1, math.ceil(max(1, row_count) / chunks[0]))
        requested_chunks = max(1, shards[0] // chunks[0])
        outer_rows = min(logical_chunks, requested_chunks) * chunks[0]
        shards = (outer_rows, *shards[1:]) if logical_chunks > 1 else None
    return chunks, shards


def _create_tabular_like(group: Any, name: str, source: Any, row_count: int, shard_rows: int) -> Any:
    shape = (int(row_count), *tuple(int(value) for value in source.shape[1:]))
    chunks = (max(1, min(16_384, max(1, row_count))), *shape[1:])
    shards = pick_shards(shape, chunks, shard_rows=int(shard_rows))
    kwargs: dict[str, Any] = {"shape": shape, "dtype": source.dtype, "chunks": chunks, "overwrite": True}
    if shards is not None:
        kwargs["shards"] = shards
    return create_geometry_preload_array(group, name, **kwargs)


def _write_readback(array: Any, start: int, stop: int, values: np.ndarray, digest: Any) -> None:
    contiguous = np.ascontiguousarray(values)
    array[start:stop] = contiguous
    persisted = np.ascontiguousarray(array[start:stop])
    if not np.array_equal(contiguous, persisted, equal_nan=True):
        raise SubjectMaskCompactionError("Subject-mask compaction readback differs from its write.")
    digest.update(persisted.view(np.uint8))


def _snapshot_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _target_crop_keys(
    target: Any,
    *,
    target_name: str,
    base: _MaskBaseCollection,
) -> tuple[np.ndarray, str]:
    rows = int(target["frame_indices"].shape[0]) if "frame_indices" in target else -1
    if "instance_key" in target:
        return _keys(target, label=f"crop_runs/{target_name}", rows=rows), "persisted"
    manifest = base.manifest_group
    if (
        not base.manifest_path.startswith(f"{REFINED_SUBJECT_MASK_PARENT}/")
        or str(manifest.attrs.get("source_crop_run") or "") != target_name
        or "instance_key" not in manifest
        or "source_crop_row_ids" not in manifest
    ):
        raise SubjectMaskCompactionError(
            f"crop_runs/{target_name} lacks instance_key and has no exact refined manifest binding."
        )
    manifest_rows = int(manifest["instance_key"].shape[0])
    manifest_keys = _keys(
        manifest,
        label=base.manifest_path,
        rows=manifest_rows,
    )
    target_rows = np.asarray(
        manifest["source_crop_row_ids"][:], dtype=np.int64
    ).reshape(-1)
    if (
        target_rows.shape != (manifest_rows,)
        or not np.array_equal(
            np.sort(target_rows),
            np.arange(rows, dtype=np.int64),
        )
    ):
        raise SubjectMaskCompactionError(
            "Refined base manifest does not cover every target crop row exactly once."
        )
    keys = np.empty(rows, dtype=np.uint64)
    keys[target_rows] = manifest_keys
    compared = 0
    for name in _BASE_MANIFEST_LINEAGE:
        if name not in manifest or name not in target:
            continue
        if not np.array_equal(
            np.asarray(manifest[name][:]),
            np.asarray(target[name][:])[target_rows],
        ):
            raise SubjectMaskCompactionError(
                f"Refined base manifest does not match target crop lineage {name!r}."
            )
        compared += 1
    if compared < 4:
        raise SubjectMaskCompactionError(
            "Refined base manifest has insufficient shared target-crop lineage."
        )
    return keys, "derived_from_exact_refined_collection_manifest_v1"


def compact_subject_mask_deltas(
    *,
    zarr_path: str | Path,
    base_run: str,
    target_crop_run: str,
    replacement_runs: Sequence[str] = (),
    output_run: str | None = None,
    dry_run: bool = False,
    tabular_shard_rows: int = DEFAULT_TABULAR_SHARD_ROWS,
    before_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Create a complete logical raw-mask run without rewriting unchanged probabilities."""

    archive = Path(zarr_path).expanduser().resolve()
    root = open_zarr_root(archive, mode="r" if dry_run else "a")
    if getattr(getattr(root, "metadata", None), "zarr_format", None) != 3:
        raise SubjectMaskCompactionError("Composite subject-mask compaction requires Zarr v3.")
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise SubjectMaskCompactionError("Archive lacks crop_runs.")
    base = _resolve_base_collection(root, str(base_run))
    try:
        target = require_crop_snapshot(
            crop_parent,
            str(target_crop_run),
            label="Target crop run",
        )
    except CropSnapshotIdentityError as exc:
        raise SubjectMaskCompactionError(str(exc)) from exc
    target_rows = int(target["frame_indices"].shape[0]) if "frame_indices" in target else -1
    target_keys, target_key_mode = _target_crop_keys(
        target,
        target_name=str(target_crop_run),
        base=base,
    )
    try:
        target_signature_snapshot = resolve_crop_source_signatures(
            root,
            target,
            label=f"crop_runs/{target_crop_run}",
            instance_keys=target_keys,
        )
    except CropSnapshotIdentityError as exc:
        raise SubjectMaskCompactionError(str(exc)) from exc
    target_signatures = target_signature_snapshot.signatures
    target_spec = target_signature_snapshot.spec
    for required in ("frame_indices", "detection_indices"):
        if required not in target:
            raise SubjectMaskCompactionError(f"Target crop lacks lineage array {required!r}.")

    base_keys = base.instance_keys
    base_signatures = base.source_signatures
    base_spec_digest = base.signature_spec_digest
    replacements = _resolve_replacements(
        root,
        replacement_runs,
        target,
        str(target_crop_run),
        target_keys,
        target_signature_snapshot,
    )
    model_sha256 = _validate_semantics(base, replacements)
    try:
        target_frame_counts = crop_frame_counts(
            root,
            target,
            fallback_frame_count=(
                int(base.manifest_group["frame_counts"].shape[0])
                if "frame_counts" in base.manifest_group
                else None
            ),
        )
        target_detection_source = crop_detection_source(target)
    except CropSnapshotIdentityError as exc:
        raise SubjectMaskCompactionError(str(exc)) from exc
    replacement_keys = [np.asarray(group["instance_key"][:], dtype=np.uint64) for group in replacements]
    plan = build_keyed_replacement_plan(
        target_instance_keys=target_keys,
        target_source_signatures=target_signatures,
        target_signature_spec_digest=target_spec.spec_digest,
        base_instance_keys=base_keys,
        base_source_signatures=base_signatures,
        base_signature_spec_digest=base_spec_digest,
        replacement_instance_keys=replacement_keys,
    )
    delta_targets = plan.replacement_target_rows
    output_name = output_run or f"subject_masks_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not output_name or "/" in output_name:
        raise SubjectMaskCompactionError("output_run must be one non-empty group name.")
    result: dict[str, Any] = {
        "ok": True,
        "status": "would_write" if dry_run else "written",
        "zarr_path": str(archive),
        "output_run": output_name,
        "output_path": f"{SUBJECT_MASK_PARENT}/{output_name}",
        "base_run": str(base_run),
        "base_manifest_path": base.manifest_path,
        "base_run_paths": list(base.run_paths),
        "target_crop_run": str(target_crop_run),
        "replacement_runs": [str(value) for value in replacement_runs],
        "source_signature_modes": {
            "base": list(base.signature_modes),
            "target": target_signature_snapshot.mode,
        },
        "base_instance_key_modes": list(base.instance_key_modes),
        "target_instance_key_mode": target_key_mode,
        "plan": plan.summary(),
        "storage": {
            "mode": COMPOSITE_SUBJECT_MASK_STORAGE_MODE,
            "schema_version": COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION,
            "reference_depth": 1,
            "base_source_count": len(base.run_paths),
            "unchanged_probability_bytes_rewritten": 0,
        },
    }
    if dry_run:
        return result

    parent = require_runs_parent(root, SUBJECT_MASK_PARENT, completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE)
    if output_name in parent:
        raise SubjectMaskCompactionError(f"{SUBJECT_MASK_PARENT}/{output_name} already exists.")
    expected_parent_state = {
        "latest": parent.attrs.get("latest"),
        RUN_LATEST_COMPLETE_ATTR: parent.attrs.get(RUN_LATEST_COMPLETE_ATTR),
    }
    target_derived_arrays = {
        "detection_source": target_detection_source,
        "frame_counts": target_frame_counts,
        "instance_key": target_keys,
    }
    target_lineage_names = tuple(
        name
        for name in (*_TARGET_LINEAGE, "frame_counts")
        if name in target or name in target_derived_arrays
    )
    target_lineage_snapshot = [
        np.asarray(
            target[name][:]
            if name in target
            else target_derived_arrays[name]
        )
        for name in target_lineage_names
    ]
    source_digest = _snapshot_digest(
        target_keys,
        target_signatures,
        np.frombuffer(target_spec.spec_digest.encode("utf-8"), dtype=np.uint8),
        *target_lineage_snapshot,
        base_keys,
        base_signatures,
        np.frombuffer(base_spec_digest.encode("utf-8"), dtype=np.uint8),
        base.flat_run_indices,
        base.flat_local_rows,
        np.frombuffer(stable_json(base.run_paths).encode("utf-8"), dtype=np.uint8),
        *replacement_keys,
    )
    run_group = parent.create_group(output_name)
    mark_run_started(run_group, run_name=output_name, stage="subject_masks")
    note_pending_latest(parent, output_name)
    started = time.perf_counter()
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        primary_base = base.primary
        copied_attrs = {
            name: json_ready(primary_base.attrs.get(name))
            for name in _SEMANTIC_ATTRS
            if name in primary_base.attrs
        }
        if "source_checkpoint" in primary_base.attrs:
            copied_attrs["source_checkpoint"] = json_ready(
                primary_base.attrs.get("source_checkpoint")
            )
        run_group.attrs.update(
            {
                **copied_attrs,
                "schema_id": SUBJECT_MASK_COMPACTION_SCHEMA_ID,
                "schema_version": SUBJECT_MASK_COMPACTION_SCHEMA_VERSION,
                "method": SUBJECT_MASK_COMPACTION_METHOD,
                "run_semantics": "immutable_raw_subject_mask_snapshot",
                "artifact_mutability": "raw_immutable",
                "subject_mask_storage_mode": COMPOSITE_SUBJECT_MASK_STORAGE_MODE,
                "composite_subject_mask_schema_id": COMPOSITE_SUBJECT_MASK_SCHEMA_ID,
                "composite_subject_mask_schema_version": COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION,
                "composite_reference_depth": 1,
                "composite_base_subject_mask_run": (
                    str(base_run) if len(base.run_paths) == 1 else None
                ),
                "composite_base_manifest_run_path": base.manifest_path,
                "composite_base_subject_mask_run_paths": list(base.run_paths),
                "source_crop_run": str(target_crop_run),
                "source_subject_mask_replacement_runs": [str(value) for value in replacement_runs],
                "source_subject_mask_replacement_run_paths": [
                    f"{SUBJECT_MASK_SHARD_PARENT}/{value}" for value in replacement_runs
                ],
                "created_at_utc": created_at,
                "total_rois": int(target_rows),
                "compaction_plan": plan.summary(),
                "source_subject_mask_model_sha256": model_sha256,
                "mask_probs_storage_layout": "composite_multi_base_delta_v2",
                "mask_probs_storage_policy": "immutable_depth_one_multi_base_delta_v2",
                "mask_probs_logical_shape": [
                    int(target_rows),
                    *[int(value) for value in primary_base["mask_probs_roi"].shape[1:]],
                ],
                "mask_probs_delta_rows": int(delta_targets.shape[0]),
                "mask_probs_reused_rows": int(target_rows - delta_targets.shape[0]),
                "masks_roi_materialized": False,
                "binary_masks_materialized": False,
                "binary_masks_source": "threshold(composite mask_probs_roi resolver)",
                "row_identity_mode": "instance_key",
                "row_identity_mode_schema": ROW_IDENTITY_MODE_SCHEMA,
                "instance_key_status": "present",
                "base_instance_key_modes": list(base.instance_key_modes),
                "instance_key_alignment_status": "exact_full_target_crop",
                "stage_selector_eligible": True,
                "selection_policy": "complete_composite_snapshot",
                "tabular_shard_rows": int(tabular_shard_rows),
            }
        )
        for name in _TARGET_LINEAGE:
            if name not in target and name not in target_derived_arrays:
                continue
            values = np.asarray(
                target[name][:]
                if name in target
                else target_derived_arrays[name]
            )
            store_array(run_group, name, values, shard_rows=int(tabular_shard_rows))
        store_array(
            run_group,
            "source_crop_row_ids",
            np.arange(target_rows, dtype=np.int64),
            shard_rows=int(tabular_shard_rows),
        )
        store_array(
            run_group,
            "frame_counts",
            target_frame_counts,
            shard_rows=int(tabular_shard_rows),
        )
        run_group.create_array(
            "available_channels",
            data=np.asarray(primary_base["available_channels"][:], dtype=bool),
            overwrite=True,
        )
        store_array(
            run_group,
            ROW_SOURCE_SIGNATURE_ARRAY,
            target_signatures,
            shard_rows=int(tabular_shard_rows),
        )
        run_group.attrs.update(target_spec.to_attrs())

        payload = run_group.create_group(COMPOSITE_SUBJECT_MASK_PAYLOAD_GROUP)
        source_codes = np.full(target_rows, COMPOSITE_SUBJECT_MASK_SOURCE_DELTA, dtype=np.uint8)
        reusable = plan.source_run_indices == REPLACEMENT_SOURCE_BASE
        source_codes[reusable] = COMPOSITE_SUBJECT_MASK_SOURCE_BASE
        source_rows = np.empty(target_rows, dtype=np.int64)
        source_run_indices = np.full(target_rows, -1, dtype=np.int32)
        reusable_flat_rows = plan.source_row_indices[reusable]
        source_run_indices[reusable] = base.flat_run_indices[reusable_flat_rows]
        source_rows[reusable] = base.flat_local_rows[reusable_flat_rows]
        source_rows[~reusable] = np.arange(delta_targets.shape[0], dtype=np.int64)
        store_array(payload, "source_codes", source_codes, shard_rows=int(tabular_shard_rows))
        store_array(
            payload,
            "source_run_indices",
            source_run_indices,
            shard_rows=int(tabular_shard_rows),
        )
        store_array(payload, "source_row_indices", source_rows, shard_rows=int(tabular_shard_rows))
        store_array(payload, "delta_target_row_indices", delta_targets, shard_rows=int(tabular_shard_rows))
        store_array(payload, "delta_instance_key", target_keys[delta_targets], shard_rows=int(tabular_shard_rows))

        base_surface = primary_base["mask_probs_roi"]
        surface_shape = (int(delta_targets.shape[0]), *tuple(int(v) for v in base_surface.shape[1:]))
        chunks, shards = _surface_layout(base_surface, int(delta_targets.shape[0]))
        kwargs: dict[str, Any] = {
            "shape": surface_shape,
            "dtype": base_surface.dtype,
            "chunks": chunks,
            "overwrite": True,
        }
        if shards is not None:
            kwargs["shards"] = shards
        delta_surface = payload.create_array("mask_probs_roi_delta", **kwargs)
        probability_digest = hashlib.sha256()
        batch_rows = int((delta_surface.shards or delta_surface.chunks)[0])
        for start in range(0, delta_targets.shape[0], batch_rows):
            stop = min(start + batch_rows, delta_targets.shape[0])
            values = _gather_replacement_rows(
                "mask_probs_roi",
                replacements=replacements,
                plan=plan,
                target_rows=delta_targets[start:stop],
                empty_template=base_surface,
            )
            _write_readback(delta_surface, start, stop, values, probability_digest)
        delta_surface.attrs.update(
            {
                "storage_layout": "indexed_sharding_v1" if shards else "regular_chunks_v1",
                "inner_chunk_shape": list(chunks),
                "outer_shard_shape": list(shards) if shards else None,
                "payload_scope": "delta_replacement_rows_only",
                "decoded_sha256": probability_digest.hexdigest(),
            }
        )

        metrics = primary_base.get("metrics")
        if metrics is None:
            raise SubjectMaskCompactionError("Base subject-mask run lacks metrics.")
        output_metrics = run_group.create_group("metrics")
        metric_names = sorted(str(name) for name in metrics.array_keys())
        for name in metric_names:
            if any(
                source.get(f"metrics/{name}") is None for source in base.groups
            ):
                raise SubjectMaskCompactionError(
                    f"Base source metrics are missing {name!r}."
                )
            if any(replacement.get(f"metrics/{name}") is None for replacement in replacements):
                raise SubjectMaskCompactionError(f"Replacement metrics are missing {name!r}.")
            source = metrics[name]
            output = _create_tabular_like(output_metrics, name, source, target_rows, int(tabular_shard_rows))
            digest = hashlib.sha256()
            physical_rows = int((output.shards or output.chunks)[0])
            for start in range(0, target_rows, physical_rows):
                stop = min(start + physical_rows, target_rows)
                values = _gather_all_rows(
                    f"metrics/{name}",
                    base=base,
                    replacements=replacements,
                    plan=plan,
                    start=start,
                    stop=stop,
                )
                _write_readback(output, start, stop, values, digest)
            output.attrs["decoded_sha256"] = digest.hexdigest()

        provenance = build_writer_run_provenance(
            command=SUBJECT_MASK_COMPACTION_METHOD,
            params={
                "zarr_path": str(archive),
                "base_run": str(base_run),
                "target_crop_run": str(target_crop_run),
                "replacement_runs": [str(value) for value in replacement_runs],
                "output_run": output_name,
                "tabular_shard_rows": int(tabular_shard_rows),
            },
            input_run_ids={
                "subject_mask_base_manifest": base.manifest_path,
                "subject_mask_base_runs": list(base.run_paths),
                "subject_mask_replacements": [
                    f"{SUBJECT_MASK_SHARD_PARENT}/{value}" for value in replacement_runs
                ],
                "crop": str(target_crop_run),
            },
            input_artifacts=[
                {
                    "role": "subject_mask_unet_checkpoint",
                    "path": str(primary_base.attrs.get("source_checkpoint") or ""),
                    "sha256": model_sha256,
                    "fingerprint_source": "validated_compaction_inputs",
                }
            ],
            cwd=Path.cwd(),
        )
        run_group.attrs[RUN_PROVENANCE_ATTR] = provenance
        validation = validate_composite_subject_mask_run(
            root, run_group, run_name=output_name, require_complete=False, verify_identity=True
        )
        result["validation"] = validation.to_dict()
        if before_publish is not None:
            before_publish()
        refreshed_target = root[f"crop_runs/{target_crop_run}"]
        refreshed_base = _resolve_base_collection(root, str(base_run))
        try:
            refreshed_target_keys, _ = _target_crop_keys(
                refreshed_target,
                target_name=str(target_crop_run),
                base=refreshed_base,
            )
            refreshed_target_signature_snapshot = resolve_crop_source_signatures(
                root,
                refreshed_target,
                label=f"crop_runs/{target_crop_run}",
                instance_keys=refreshed_target_keys,
            )
            refreshed_target_frame_counts = crop_frame_counts(
                root,
                refreshed_target,
                fallback_frame_count=target_frame_counts.shape[0],
            )
            refreshed_target_detection_source = crop_detection_source(refreshed_target)
        except CropSnapshotIdentityError as exc:
            raise SubjectMaskCompactionError(str(exc)) from exc
        refreshed_replacements = _resolve_replacements(
            root,
            replacement_runs,
            refreshed_target,
            str(target_crop_run),
            np.asarray(refreshed_target_keys, dtype=np.uint64),
            refreshed_target_signature_snapshot,
        )
        if _validate_semantics(refreshed_base, refreshed_replacements) != model_sha256:
            raise SubjectMaskCompactionError(
                "Subject-mask model identity changed before publication."
            )
        refreshed_replacement_keys = [
            np.asarray(group["instance_key"][:], dtype=np.uint64)
            for group in refreshed_replacements
        ]
        if source_digest != _snapshot_digest(
            refreshed_target_keys,
            refreshed_target_signature_snapshot.signatures,
            np.frombuffer(
                refreshed_target_signature_snapshot.spec.spec_digest.encode("utf-8"),
                dtype=np.uint8,
            ),
            *[
                np.asarray(
                    refreshed_target[name][:]
                    if name in refreshed_target
                    else (
                        refreshed_target_frame_counts
                        if name == "frame_counts"
                        else (
                            refreshed_target_keys
                            if name == "instance_key"
                            else refreshed_target_detection_source
                        )
                    )
                )
                for name in target_lineage_names
            ],
            refreshed_base.instance_keys,
            refreshed_base.source_signatures,
            np.frombuffer(
                refreshed_base.signature_spec_digest.encode("utf-8"),
                dtype=np.uint8,
            ),
            refreshed_base.flat_run_indices,
            refreshed_base.flat_local_rows,
            np.frombuffer(
                stable_json(refreshed_base.run_paths).encode("utf-8"),
                dtype=np.uint8,
            ),
            *refreshed_replacement_keys,
        ):
            raise SubjectMaskCompactionError("A compaction input changed before publication.")
        if any(parent.attrs.get(key) != value for key, value in expected_parent_state.items()):
            raise SubjectMaskCompactionError("Subject-mask selection state changed before publication.")
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=output_name,
            run_provenance=provenance,
        )
        duration = time.perf_counter() - started
        run_group.attrs["compaction_duration_seconds"] = float(duration)
        result["compaction_duration_seconds"] = float(duration)
        return result
    except Exception as exc:
        mark_run_failed(run_group, parent_group=parent, run_name=output_name, error=str(exc))
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--target-crop-run", required=True)
    parser.add_argument("--replacement-run", action="append", default=[])
    parser.add_argument("--output-run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tabular-shard-rows", type=int, default=DEFAULT_TABULAR_SHARD_ROWS)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = compact_subject_mask_deltas(
            zarr_path=args.zarr_path,
            base_run=args.base_run,
            target_crop_run=args.target_crop_run,
            replacement_runs=args.replacement_run,
            output_run=args.output_run,
            dry_run=bool(args.dry_run),
            tabular_shard_rows=int(args.tabular_shard_rows),
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    else:
        print(f"{result['status']}: {result['output_path']} ({result['plan']['target_row_count']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
