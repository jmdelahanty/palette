"""Depth-one immutable base-plus-delta storage for raw subject-mask pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fisheye.shared.crop_snapshot_identity import (
    CropSnapshotIdentityError,
    require_crop_snapshot,
    resolve_crop_source_signatures,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    is_run_complete_in_parent,
)


COMPOSITE_SUBJECT_MASK_SCHEMA_ID = "palette.composite_subject_mask"
COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION = 2
COMPOSITE_SUBJECT_MASK_LEGACY_SCHEMA_VERSION = 1
COMPOSITE_SUBJECT_MASK_STORAGE_MODE = "composite"
COMPOSITE_SUBJECT_MASK_PAYLOAD_GROUP = "composite_payload"
COMPOSITE_SUBJECT_MASK_SOURCE_BASE = 0
COMPOSITE_SUBJECT_MASK_SOURCE_DELTA = 1
COMPOSITE_SUBJECT_MASK_READ_MAX_BATCH_BYTES = 64 * 1024 * 1024
_BASE_MANIFEST_LINEAGE = (
    "frame_indices",
    "source_frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
    "detection_source",
)


class CompositeSubjectMaskError(RuntimeError):
    """Raised when a composite subject-mask run is incomplete or ambiguous."""


@dataclass(frozen=True)
class CompositeSubjectMaskValidation:
    run_name: str | None
    base_run_name: str
    base_run_paths: tuple[str, ...]
    target_crop_run: str
    target_rows: int
    base_rows_used: int
    delta_rows: int
    surface_shape: tuple[int, int, int]
    surface_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "base_run_name": self.base_run_name,
            "base_run_paths": list(self.base_run_paths),
            "target_crop_run": self.target_crop_run,
            "target_rows": self.target_rows,
            "base_rows_used": self.base_rows_used,
            "delta_rows": self.delta_rows,
            "surface_shape": list(self.surface_shape),
            "surface_dtype": self.surface_dtype,
        }


def _text(value: object) -> str:
    return str(value or "").strip()


def _int_attr(group: Any, name: str, *, default: int = -1) -> int:
    try:
        return int(group.attrs.get(name, default))
    except (TypeError, ValueError) as exc:
        raise CompositeSubjectMaskError(f"Attr {name!r} must be an integer.") from exc


def _unique_keys(group: Any, *, label: str, rows: int) -> np.ndarray:
    if "instance_key" not in group:
        raise CompositeSubjectMaskError(f"{label} is missing instance_key.")
    array = group["instance_key"]
    if np.dtype(array.dtype) != np.dtype(np.uint64) or tuple(array.shape) != (rows,):
        raise CompositeSubjectMaskError(f"{label}/instance_key must be uint64[{rows}].")
    keys = np.asarray(array[:], dtype=np.uint64).reshape(-1)
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise CompositeSubjectMaskError(f"{label}/instance_key is not unique.")
    return keys


def _bound_run_keys(
    root: Any,
    run_group: Any,
    *,
    label: str,
    rows: int,
) -> np.ndarray:
    crop_name = _text(run_group.attrs.get("source_crop_run"))
    crop_parent = root.get("crop_runs")
    if not crop_name or crop_parent is None:
        raise CompositeSubjectMaskError(f"{label} has no resolvable source crop.")
    try:
        crop = require_crop_snapshot(crop_parent, crop_name, label="Source crop run")
    except CropSnapshotIdentityError as exc:
        raise CompositeSubjectMaskError(str(exc)) from exc
    crop_rows = int(crop["frame_indices"].shape[0]) if "frame_indices" in crop else -1
    if "source_crop_row_ids" not in run_group:
        raise CompositeSubjectMaskError(f"{label} lacks source_crop_row_ids.")
    source_rows = np.asarray(
        run_group["source_crop_row_ids"][:], dtype=np.int64
    ).reshape(-1)
    if source_rows.shape != (rows,) or (
        source_rows.size
        and (source_rows.min() < 0 or source_rows.max() >= crop_rows)
    ):
        raise CompositeSubjectMaskError(f"{label} references invalid source crop rows.")
    actual = (
        _unique_keys(run_group, label=label, rows=rows)
        if "instance_key" in run_group
        else None
    )
    if "instance_key" in crop:
        crop_keys = _unique_keys(crop, label=f"crop_runs/{crop_name}", rows=crop_rows)
    elif actual is not None and np.array_equal(
        np.sort(source_rows),
        np.arange(crop_rows, dtype=np.int64),
    ):
        crop_keys = np.empty(crop_rows, dtype=np.uint64)
        crop_keys[source_rows] = actual
    else:
        raise CompositeSubjectMaskError(
            f"{label} cannot establish keys for keyless crop_runs/{crop_name}."
        )
    expected = crop_keys[source_rows]
    if actual is None:
        return expected
    if not np.array_equal(actual, expected):
        raise CompositeSubjectMaskError(
            f"{label} keys do not match its source crop rows."
        )
    return actual


def _run_source_signatures(root: Any, run_group: Any, *, label: str) -> tuple[np.ndarray, str]:
    crop_name = _text(run_group.attrs.get("source_crop_run"))
    crop_parent = root.get("crop_runs")
    if not crop_name or crop_parent is None or crop_name not in crop_parent:
        raise CompositeSubjectMaskError(f"{label} has no resolvable source crop.")
    try:
        crop = require_crop_snapshot(crop_parent, crop_name, label="Source crop run")
    except CropSnapshotIdentityError as exc:
        raise CompositeSubjectMaskError(str(exc)) from exc
    if "source_crop_row_ids" not in run_group:
        raise CompositeSubjectMaskError(f"{label} lacks source_crop_row_ids.")
    source_rows = np.asarray(run_group["source_crop_row_ids"][:], dtype=np.int64).reshape(-1)
    crop_rows = int(crop["frame_indices"].shape[0]) if "frame_indices" in crop else -1
    if source_rows.size and (source_rows.min() < 0 or source_rows.max() >= crop_rows):
        raise CompositeSubjectMaskError(f"{label} references a crop row out of bounds.")
    run_keys = _bound_run_keys(
        root,
        run_group,
        label=label,
        rows=source_rows.shape[0],
    )
    if "instance_key" in crop:
        crop_keys = _unique_keys(crop, label=f"crop_runs/{crop_name}", rows=crop_rows)
    else:
        if not np.array_equal(
            np.sort(source_rows),
            np.arange(crop_rows, dtype=np.int64),
        ):
            raise CompositeSubjectMaskError(
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
        raise CompositeSubjectMaskError(str(exc)) from exc
    if not np.array_equal(run_keys, crop_keys[source_rows]):
        raise CompositeSubjectMaskError(f"{label} keys do not match its source crop rows.")
    signatures = snapshot.signatures[source_rows]
    return signatures, snapshot.spec.spec_digest


def _raw_base_paths(run_group: Any, *, schema_version: int) -> tuple[str, ...]:
    if schema_version == COMPOSITE_SUBJECT_MASK_LEGACY_SCHEMA_VERSION:
        name = _text(run_group.attrs.get("composite_base_subject_mask_run"))
        return (f"subject_mask_runs/{name}",) if name else ()
    raw = run_group.attrs.get("composite_base_subject_mask_run_paths")
    if not isinstance(raw, (list, tuple)) or not raw:
        return ()
    paths = tuple(_text(value).strip("/") for value in raw)
    return paths if all(paths) else ()


def _resolve_raw_base(root: Any, path: str, *, run_name: str | None) -> Any:
    parts = [part for part in _text(path).strip("/").split("/") if part]
    if len(parts) != 2 or parts[0] not in {
        "subject_mask_runs",
        "subject_mask_shard_runs",
    }:
        raise CompositeSubjectMaskError(
            f"Composite base path {path!r} is not a raw subject-mask run."
        )
    if parts[0] == "subject_mask_runs" and parts[1] == run_name:
        raise CompositeSubjectMaskError("Composite subject-mask run references itself.")
    parent = root.get(parts[0])
    if parent is None or parts[1] not in parent:
        raise CompositeSubjectMaskError(f"Composite base {path!r} is missing.")
    group = parent[parts[1]]
    if not is_run_complete_in_parent(parent, group):
        raise CompositeSubjectMaskError(f"Composite base {path!r} is not complete.")
    if _text(group.attrs.get("subject_mask_storage_mode")) == COMPOSITE_SUBJECT_MASK_STORAGE_MODE:
        raise CompositeSubjectMaskError("Composite bases cannot themselves be composite.")
    if "mask_probs_roi" not in group:
        raise CompositeSubjectMaskError(f"Composite base {path!r} lacks mask_probs_roi.")
    return group


def _manifest_base_identity(
    root: Any,
    run_group: Any,
    *,
    base_paths: tuple[str, ...],
    base_groups: tuple[Any, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], str] | None:
    manifest_path = _text(
        run_group.attrs.get("composite_base_manifest_run_path")
    ).strip("/")
    if not manifest_path or not manifest_path.startswith("refined_subject_masks_runs/"):
        return None
    parts = manifest_path.split("/")
    if len(parts) != 2:
        raise CompositeSubjectMaskError("Composite base manifest path is invalid.")
    parent = root.get(parts[0])
    if parent is None or parts[1] not in parent:
        raise CompositeSubjectMaskError("Composite base manifest is missing.")
    manifest = parent[parts[1]]
    if not is_run_complete_in_parent(parent, manifest):
        raise CompositeSubjectMaskError("Composite base manifest is not complete.")
    raw_paths = manifest.attrs.get("source_subject_mask_shard_run_paths")
    if not isinstance(raw_paths, (list, tuple)):
        raise CompositeSubjectMaskError("Composite base manifest lacks raw shard paths.")
    normalized_paths = tuple(_text(value).strip("/") for value in raw_paths)
    if normalized_paths != base_paths:
        raise CompositeSubjectMaskError(
            "Composite base paths differ from the bound refined manifest."
        )
    rows = int(manifest["instance_key"].shape[0]) if "instance_key" in manifest else -1
    manifest_keys = _unique_keys(manifest, label=manifest_path, rows=rows)
    manifest_signatures, digest = _run_source_signatures(
        root,
        manifest,
        label=manifest_path,
    )
    if sum(int(group["mask_probs_roi"].shape[0]) for group in base_groups) != rows:
        raise CompositeSubjectMaskError(
            "Composite base manifest row count differs from its raw shards."
        )
    key_parts: list[np.ndarray] = []
    signature_parts: list[np.ndarray] = []
    start = 0
    for path, group in zip(base_paths, base_groups, strict=True):
        stop = start + int(group["mask_probs_roi"].shape[0])
        compared = 0
        for name in _BASE_MANIFEST_LINEAGE:
            if name not in manifest or name not in group:
                continue
            if not np.array_equal(
                np.asarray(manifest[name][start:stop]),
                np.asarray(group[name][:]),
            ):
                raise CompositeSubjectMaskError(
                    f"Composite base {path!r} differs from its manifest for {name!r}."
                )
            compared += 1
        if compared < 4:
            raise CompositeSubjectMaskError(
                f"Composite base {path!r} has insufficient manifest lineage."
            )
        key_parts.append(manifest_keys[start:stop])
        signature_parts.append(manifest_signatures[start:stop])
        start = stop
    return tuple(key_parts), tuple(signature_parts), digest


def _keyless_target_keys_from_manifest(
    root: Any,
    run_group: Any,
    target_crop: Any,
    *,
    target_crop_name: str,
    target_rows: int,
) -> np.ndarray:
    manifest_path = _text(
        run_group.attrs.get("composite_base_manifest_run_path")
    ).strip("/")
    parts = manifest_path.split("/")
    if len(parts) != 2 or parts[0] != "refined_subject_masks_runs":
        raise CompositeSubjectMaskError(
            "Keyless composite target lacks an exact refined base manifest."
        )
    parent = root.get(parts[0])
    if parent is None or parts[1] not in parent:
        raise CompositeSubjectMaskError("Composite base manifest is missing.")
    manifest = parent[parts[1]]
    if (
        not is_run_complete_in_parent(parent, manifest)
        or _text(manifest.attrs.get("source_crop_run")) != target_crop_name
        or "instance_key" not in manifest
        or "source_crop_row_ids" not in manifest
    ):
        raise CompositeSubjectMaskError(
            "Composite base manifest does not exactly bind the keyless target crop."
        )
    manifest_rows = int(manifest["instance_key"].shape[0])
    manifest_keys = _unique_keys(manifest, label=manifest_path, rows=manifest_rows)
    target_positions = np.asarray(
        manifest["source_crop_row_ids"][:], dtype=np.int64
    ).reshape(-1)
    if (
        target_positions.shape != (manifest_rows,)
        or not np.array_equal(
            np.sort(target_positions),
            np.arange(target_rows, dtype=np.int64),
        )
    ):
        raise CompositeSubjectMaskError(
            "Composite base manifest does not cover the keyless target exactly once."
        )
    compared = 0
    for name in _BASE_MANIFEST_LINEAGE:
        if name not in manifest or name not in target_crop:
            continue
        if not np.array_equal(
            np.asarray(manifest[name][:]),
            np.asarray(target_crop[name][:])[target_positions],
        ):
            raise CompositeSubjectMaskError(
                f"Composite base manifest differs from target lineage {name!r}."
            )
        compared += 1
    if compared < 4:
        raise CompositeSubjectMaskError(
            "Composite target has insufficient shared manifest lineage."
        )
    keys = np.empty(target_rows, dtype=np.uint64)
    keys[target_positions] = manifest_keys
    return keys


def validate_composite_subject_mask_run(
    root: Any,
    run_group: Any,
    *,
    run_name: str | None = None,
    require_complete: bool = True,
    verify_identity: bool = True,
) -> CompositeSubjectMaskValidation:
    """Validate exact target coverage and a standalone immutable base."""

    if _text(run_group.attrs.get("subject_mask_storage_mode")) != COMPOSITE_SUBJECT_MASK_STORAGE_MODE:
        raise CompositeSubjectMaskError("Subject-mask run is not declared composite.")
    if _text(run_group.attrs.get("composite_subject_mask_schema_id")) != COMPOSITE_SUBJECT_MASK_SCHEMA_ID:
        raise CompositeSubjectMaskError("Composite subject-mask schema id is unsupported.")
    schema_version = _int_attr(run_group, "composite_subject_mask_schema_version")
    if schema_version not in {
        COMPOSITE_SUBJECT_MASK_LEGACY_SCHEMA_VERSION,
        COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION,
    }:
        raise CompositeSubjectMaskError("Composite subject-mask schema version is unsupported.")
    if _int_attr(run_group, "composite_reference_depth") != 1:
        raise CompositeSubjectMaskError("Composite subject-mask reference depth must be one.")
    if require_complete and run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise CompositeSubjectMaskError("Composite subject-mask run is not complete.")
    if "mask_probs_roi" in run_group:
        raise CompositeSubjectMaskError("Composite run must not expose a partial top-level probability array.")

    base_paths = _raw_base_paths(run_group, schema_version=schema_version)
    if not base_paths or len(set(base_paths)) != len(base_paths):
        raise CompositeSubjectMaskError(
            "Composite subject-mask base paths are missing or duplicated."
        )
    base_groups = tuple(
        _resolve_raw_base(root, path, run_name=run_name) for path in base_paths
    )
    base_surfaces = tuple(group["mask_probs_roi"] for group in base_groups)
    base_surface = base_surfaces[0]
    if len(base_surface.shape) != 4:
        raise CompositeSubjectMaskError("Base probability surface must have shape (N,C,H,W).")
    expected_trailing = tuple(int(value) for value in base_surface.shape[1:])
    expected_dtype = np.dtype(base_surface.dtype)
    for path, surface in zip(base_paths, base_surfaces, strict=True):
        if (
            len(surface.shape) != 4
            or tuple(int(value) for value in surface.shape[1:]) != expected_trailing
            or np.dtype(surface.dtype) != expected_dtype
        ):
            raise CompositeSubjectMaskError(
                f"Composite base {path!r} has an incompatible probability surface."
            )

    crop_parent = root.get("crop_runs")
    target_crop_name = _text(run_group.attrs.get("source_crop_run"))
    if crop_parent is None or not target_crop_name or target_crop_name not in crop_parent:
        raise CompositeSubjectMaskError("Composite target crop is missing.")
    try:
        target_crop = require_crop_snapshot(
            crop_parent,
            target_crop_name,
            label="Composite target crop",
        )
    except CropSnapshotIdentityError as exc:
        raise CompositeSubjectMaskError(str(exc)) from exc
    target_rows = (
        int(target_crop["frame_indices"].shape[0])
        if "frame_indices" in target_crop
        else -1
    )
    output_keys = _unique_keys(run_group, label="Composite subject-mask run", rows=target_rows)
    if "instance_key" in target_crop:
        target_keys = _unique_keys(
            target_crop,
            label=f"crop_runs/{target_crop_name}",
            rows=target_rows,
        )
    else:
        target_keys = _keyless_target_keys_from_manifest(
            root,
            run_group,
            target_crop,
            target_crop_name=target_crop_name,
            target_rows=target_rows,
        )
    if not np.array_equal(output_keys, target_keys):
        raise CompositeSubjectMaskError("Composite keys do not exactly equal target crop keys.")
    if "source_crop_row_ids" not in run_group or not np.array_equal(
        np.asarray(run_group["source_crop_row_ids"][:], dtype=np.int64),
        np.arange(target_rows, dtype=np.int64),
    ):
        raise CompositeSubjectMaskError("Composite source_crop_row_ids must cover the target crop in order.")

    payload = run_group.get(COMPOSITE_SUBJECT_MASK_PAYLOAD_GROUP)
    required = (
        "source_codes",
        "source_row_indices",
        "delta_target_row_indices",
        "delta_instance_key",
        "mask_probs_roi_delta",
    )
    if payload is None or any(name not in payload for name in required):
        raise CompositeSubjectMaskError("Composite subject-mask payload is incomplete.")
    source_codes = np.asarray(payload["source_codes"][:], dtype=np.uint8).reshape(-1)
    source_rows = np.asarray(payload["source_row_indices"][:], dtype=np.int64).reshape(-1)
    if schema_version == COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION:
        if "source_run_indices" not in payload:
            raise CompositeSubjectMaskError(
                "Composite v2 payload lacks source_run_indices."
            )
        source_run_indices = np.asarray(
            payload["source_run_indices"][:], dtype=np.int32
        ).reshape(-1)
    else:
        source_run_indices = np.where(
            source_codes == COMPOSITE_SUBJECT_MASK_SOURCE_BASE,
            0,
            -1,
        ).astype(np.int32)
    if (
        source_codes.shape != (target_rows,)
        or source_rows.shape != (target_rows,)
        or source_run_indices.shape != (target_rows,)
    ):
        raise CompositeSubjectMaskError("Composite source mapping does not cover every target row.")
    if np.any(~np.isin(source_codes, [COMPOSITE_SUBJECT_MASK_SOURCE_BASE, COMPOSITE_SUBJECT_MASK_SOURCE_DELTA])):
        raise CompositeSubjectMaskError("Composite source mapping contains an unknown code.")
    if np.any(source_rows < 0):
        raise CompositeSubjectMaskError("Composite source mapping contains a negative row.")
    base_targets = np.flatnonzero(source_codes == COMPOSITE_SUBJECT_MASK_SOURCE_BASE)
    delta_targets = np.flatnonzero(source_codes == COMPOSITE_SUBJECT_MASK_SOURCE_DELTA)
    if np.any(source_run_indices[delta_targets] != -1):
        raise CompositeSubjectMaskError(
            "Composite delta rows must use source_run_indices=-1."
        )
    if base_targets.size and (
        int(source_run_indices[base_targets].min()) < 0
        or int(source_run_indices[base_targets].max()) >= len(base_groups)
    ):
        raise CompositeSubjectMaskError(
            "Composite mapping references a base run out of range."
        )
    used_pairs: list[np.ndarray] = []
    for base_index, surface in enumerate(base_surfaces):
        positions = base_targets[source_run_indices[base_targets] == base_index]
        rows = source_rows[positions]
        if rows.size and int(rows.max()) >= int(surface.shape[0]):
            raise CompositeSubjectMaskError(
                f"Composite mapping references a row outside base {base_index}."
            )
        if rows.size:
            used_pairs.append(
                np.column_stack(
                    (
                        np.full(rows.shape[0], base_index, dtype=np.int64),
                        rows,
                    )
                )
            )
    if used_pairs:
        pairs = np.concatenate(used_pairs, axis=0)
        if np.unique(pairs, axis=0).shape[0] != base_targets.shape[0]:
            raise CompositeSubjectMaskError(
                "Composite mapping reuses one base row more than once."
            )
    if not np.array_equal(np.sort(source_rows[delta_targets]), np.arange(delta_targets.shape[0], dtype=np.int64)):
        raise CompositeSubjectMaskError("Composite mapping does not reference each delta row once.")
    declared_targets = np.asarray(payload["delta_target_row_indices"][:], dtype=np.int64).reshape(-1)
    declared_keys = np.asarray(payload["delta_instance_key"][:], dtype=np.uint64).reshape(-1)
    if not np.array_equal(declared_targets, delta_targets) or not np.array_equal(declared_keys, target_keys[delta_targets]):
        raise CompositeSubjectMaskError("Composite delta identity does not match its target rows.")
    delta_surface = payload["mask_probs_roi_delta"]
    if tuple(int(value) for value in delta_surface.shape) != (int(delta_targets.shape[0]), *expected_trailing):
        raise CompositeSubjectMaskError("Composite delta probability shape differs from its base.")
    if np.dtype(delta_surface.dtype) != expected_dtype:
        raise CompositeSubjectMaskError("Composite delta probability dtype differs from its base.")

    if verify_identity and base_targets.size:
        try:
            target_snapshot = resolve_crop_source_signatures(
                root,
                target_crop,
                label=f"crop_runs/{target_crop_name}",
                instance_keys=target_keys,
            )
        except CropSnapshotIdentityError as exc:
            raise CompositeSubjectMaskError(str(exc)) from exc
        manifest_identity = _manifest_base_identity(
            root,
            run_group,
            base_paths=base_paths,
            base_groups=base_groups,
        )
        for base_index, (path, group, surface) in enumerate(
            zip(base_paths, base_groups, base_surfaces, strict=True)
        ):
            positions = base_targets[
                source_run_indices[base_targets] == base_index
            ]
            if not positions.size:
                continue
            rows = source_rows[positions]
            if manifest_identity is None:
                base_keys = _bound_run_keys(
                    root,
                    group,
                    label=path,
                    rows=int(surface.shape[0]),
                )
                base_signatures, base_spec_digest = _run_source_signatures(
                    root,
                    group,
                    label=path,
                )
            else:
                base_keys = manifest_identity[0][base_index]
                base_signatures = manifest_identity[1][base_index]
                base_spec_digest = manifest_identity[2]
            if base_spec_digest != target_snapshot.spec.spec_digest:
                raise CompositeSubjectMaskError(
                    "Composite reused rows use different signature specifications."
                )
            if not np.array_equal(target_keys[positions], base_keys[rows]):
                raise CompositeSubjectMaskError(
                    "Composite reused keys do not match base rows."
                )
            if not np.array_equal(
                target_snapshot.signatures[positions],
                base_signatures[rows],
            ):
                raise CompositeSubjectMaskError(
                    "Composite reused crop signatures do not match base rows."
                )

    return CompositeSubjectMaskValidation(
        run_name=run_name,
        base_run_name=base_paths[0].split("/")[-1],
        base_run_paths=base_paths,
        target_crop_run=target_crop_name,
        target_rows=target_rows,
        base_rows_used=int(base_targets.shape[0]),
        delta_rows=int(delta_targets.shape[0]),
        surface_shape=expected_trailing,
        surface_dtype=str(np.dtype(base_surface.dtype)),
    )


def _selection_length(selector: object, size: int) -> int | None:
    if isinstance(selector, (int, np.integer)):
        return None
    if isinstance(selector, slice):
        return len(range(*selector.indices(size)))
    values = np.asarray(selector)
    if values.ndim != 1:
        raise IndexError("Composite subject-mask selections must be one-dimensional.")
    return int(values.shape[0])


class CompositeSubjectMaskArray:
    """Read-only probability-array view over immutable bases plus one delta."""

    def __init__(
        self,
        *,
        base_arrays: tuple[Any, ...],
        delta_array: Any,
        source_codes: np.ndarray,
        source_run_indices: np.ndarray,
        source_row_indices: np.ndarray,
    ) -> None:
        if not base_arrays:
            raise CompositeSubjectMaskError("Composite mask resolver has no base arrays.")
        self._base_arrays = base_arrays
        self._delta_array = delta_array
        self._source_codes = np.asarray(source_codes, dtype=np.uint8).reshape(-1)
        self._source_run_indices = np.asarray(
            source_run_indices, dtype=np.int32
        ).reshape(-1)
        self._source_row_indices = np.asarray(source_row_indices, dtype=np.int64).reshape(-1)
        self.shape = (
            int(self._source_codes.shape[0]),
            *tuple(int(v) for v in base_arrays[0].shape[1:]),
        )
        self.ndim = 4
        self.dtype = np.dtype(base_arrays[0].dtype)

    @classmethod
    def open(
        cls,
        root: Any,
        run_group: Any,
        *,
        run_name: str | None = None,
        verify_identity: bool = False,
    ) -> "CompositeSubjectMaskArray":
        validation = validate_composite_subject_mask_run(
            root,
            run_group,
            run_name=run_name,
            require_complete=True,
            verify_identity=verify_identity,
        )
        payload = run_group[COMPOSITE_SUBJECT_MASK_PAYLOAD_GROUP]
        base_groups = tuple(
            _resolve_raw_base(root, path, run_name=run_name)
            for path in validation.base_run_paths
        )
        source_codes = np.asarray(payload["source_codes"][:], dtype=np.uint8)
        if "source_run_indices" in payload:
            source_run_indices = np.asarray(
                payload["source_run_indices"][:], dtype=np.int32
            )
        else:
            source_run_indices = np.where(
                source_codes == COMPOSITE_SUBJECT_MASK_SOURCE_BASE,
                0,
                -1,
            ).astype(np.int32)
        return cls(
            base_arrays=tuple(group["mask_probs_roi"] for group in base_groups),
            delta_array=payload["mask_probs_roi_delta"],
            source_codes=source_codes,
            source_run_indices=source_run_indices,
            source_row_indices=np.asarray(payload["source_row_indices"][:], dtype=np.int64),
        )

    def __len__(self) -> int:
        return int(self.shape[0])

    def _target_rows(self, selector: object) -> tuple[np.ndarray, bool]:
        if isinstance(selector, (int, np.integer)):
            value = int(selector)
            if value < 0:
                value += self.shape[0]
            rows = np.asarray([value], dtype=np.int64)
            scalar = True
        elif isinstance(selector, slice):
            rows = np.arange(*selector.indices(self.shape[0]), dtype=np.int64)
            scalar = False
        else:
            rows = np.asarray(selector, dtype=np.int64).reshape(-1)
            rows = np.where(rows < 0, rows + self.shape[0], rows)
            scalar = False
        if rows.size and (rows.min() < 0 or rows.max() >= self.shape[0]):
            raise IndexError("Composite subject-mask row is out of bounds.")
        return rows, scalar

    def _output_shape(self, row_count: int, trailing: tuple[object, ...]) -> tuple[int, ...]:
        selectors = (*trailing, *([slice(None)] * (3 - len(trailing))))
        if len(selectors) > 3:
            raise IndexError("Too many composite subject-mask indexes.")
        shape: list[int] = [int(row_count)]
        for size, selector in zip(self.shape[1:], selectors):
            length = _selection_length(selector, int(size))
            if length is not None:
                shape.append(length)
        return tuple(shape)

    @staticmethod
    def _read_rows_into(
        array: Any,
        source_rows: np.ndarray,
        output: np.ndarray,
        output_rows: np.ndarray,
        trailing: tuple[object, ...],
    ) -> None:
        rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
        destinations = np.asarray(output_rows, dtype=np.int64).reshape(-1)
        if not rows.size:
            return
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        destinations = destinations[order]
        bytes_per_row = max(1, int(np.prod(output.shape[1:], dtype=np.int64)) * output.dtype.itemsize)
        max_rows = max(1, COMPOSITE_SUBJECT_MASK_READ_MAX_BATCH_BYTES // bytes_per_row)
        start = 0
        while start < rows.shape[0]:
            stop = start + 1
            first = int(rows[start])
            while (
                stop < rows.shape[0]
                and rows[stop] == rows[stop - 1] + 1
                and int(rows[stop]) - first < max_rows
            ):
                stop += 1
            last = int(rows[stop - 1]) + 1
            values = np.asarray(array[(slice(first, last), *trailing)])
            output[destinations[start:stop]] = values[rows[start:stop] - first]
            start = stop

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, tuple):
            row_selector = key[0] if key else slice(None)
            trailing = tuple(key[1:])
        else:
            row_selector = key
            trailing = ()
        target_rows, scalar = self._target_rows(row_selector)
        output = np.empty(self._output_shape(target_rows.shape[0], trailing), dtype=self.dtype)
        codes = self._source_codes[target_rows]
        mapped = self._source_row_indices[target_rows]
        delta_positions = np.flatnonzero(codes == COMPOSITE_SUBJECT_MASK_SOURCE_DELTA)
        selected_run_indices = self._source_run_indices[target_rows]
        for base_index, base_array in enumerate(self._base_arrays):
            base_positions = np.flatnonzero(
                (codes == COMPOSITE_SUBJECT_MASK_SOURCE_BASE)
                & (selected_run_indices == base_index)
            )
            self._read_rows_into(
                base_array,
                mapped[base_positions],
                output,
                base_positions,
                trailing,
            )
        self._read_rows_into(self._delta_array, mapped[delta_positions], output, delta_positions, trailing)
        return output[0] if scalar else output


def find_composite_subject_mask_dependents(
    container: Any,
    base_run_name: str,
    *,
    base_parent_name: str | None = None,
) -> tuple[str, ...]:
    if base_parent_name is None:
        mask_parent = container
        target_path = f"subject_mask_runs/{_text(base_run_name)}"
    else:
        root = container
        mask_parent = root.get("subject_mask_runs")
        if mask_parent is None:
            return ()
        target_path = f"{_text(base_parent_name)}/{_text(base_run_name)}"
    dependents: list[str] = []
    names = mask_parent.group_keys() if hasattr(mask_parent, "group_keys") else mask_parent.keys()
    for name in names:
        group = mask_parent[str(name)]
        if _text(group.attrs.get("subject_mask_storage_mode")) != COMPOSITE_SUBJECT_MASK_STORAGE_MODE:
            continue
        version = _int_attr(
            group,
            "composite_subject_mask_schema_version",
            default=COMPOSITE_SUBJECT_MASK_LEGACY_SCHEMA_VERSION,
        )
        if target_path in _raw_base_paths(group, schema_version=version):
            dependents.append(str(name))
    return tuple(sorted(dependents))


def assert_subject_mask_run_unreferenced(
    container: Any,
    run_name: str,
    *,
    base_parent_name: str | None = None,
) -> None:
    dependents = find_composite_subject_mask_dependents(
        container,
        run_name,
        base_parent_name=base_parent_name,
    )
    if dependents:
        raise CompositeSubjectMaskError(
            f"Refusing to delete subject-mask run {run_name!r}; composite dependents: "
            + ", ".join(dependents)
        )


__all__ = [
    "COMPOSITE_SUBJECT_MASK_SCHEMA_ID",
    "COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION",
    "COMPOSITE_SUBJECT_MASK_STORAGE_MODE",
    "COMPOSITE_SUBJECT_MASK_PAYLOAD_GROUP",
    "COMPOSITE_SUBJECT_MASK_SOURCE_BASE",
    "COMPOSITE_SUBJECT_MASK_SOURCE_DELTA",
    "CompositeSubjectMaskArray",
    "CompositeSubjectMaskError",
    "CompositeSubjectMaskValidation",
    "validate_composite_subject_mask_run",
    "find_composite_subject_mask_dependents",
    "assert_subject_mask_run_unreferenced",
]
