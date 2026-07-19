"""Depth-one immutable base-plus-delta storage for raw subject-mask pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fisheye.shared.row_source_signature import (
    ROW_SOURCE_SIGNATURE_ARRAY,
    load_row_source_signature_spec,
    validate_row_source_signature_array,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    is_run_complete_in_parent,
)


COMPOSITE_SUBJECT_MASK_SCHEMA_ID = "palette.composite_subject_mask"
COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION = 1
COMPOSITE_SUBJECT_MASK_STORAGE_MODE = "composite"
COMPOSITE_SUBJECT_MASK_PAYLOAD_GROUP = "composite_payload"
COMPOSITE_SUBJECT_MASK_SOURCE_BASE = 0
COMPOSITE_SUBJECT_MASK_SOURCE_DELTA = 1
COMPOSITE_SUBJECT_MASK_READ_MAX_BATCH_BYTES = 64 * 1024 * 1024


class CompositeSubjectMaskError(RuntimeError):
    """Raised when a composite subject-mask run is incomplete or ambiguous."""


@dataclass(frozen=True)
class CompositeSubjectMaskValidation:
    run_name: str | None
    base_run_name: str
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


def _run_source_signatures(root: Any, run_group: Any, *, label: str) -> tuple[np.ndarray, str]:
    crop_name = _text(run_group.attrs.get("source_crop_run"))
    crop_parent = root.get("crop_runs")
    if not crop_name or crop_parent is None or crop_name not in crop_parent:
        raise CompositeSubjectMaskError(f"{label} has no resolvable source crop.")
    crop = crop_parent[crop_name]
    if not is_run_complete_in_parent(crop_parent, crop):
        raise CompositeSubjectMaskError(f"crop_runs/{crop_name} is not complete.")
    crop_rows = int(crop["instance_key"].shape[0]) if "instance_key" in crop else -1
    crop_keys = _unique_keys(crop, label=f"crop_runs/{crop_name}", rows=crop_rows)
    if ROW_SOURCE_SIGNATURE_ARRAY not in crop:
        raise CompositeSubjectMaskError(f"crop_runs/{crop_name} lacks signed source rows.")
    validate_row_source_signature_array(crop[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=crop_rows)
    if "source_crop_row_ids" not in run_group:
        raise CompositeSubjectMaskError(f"{label} lacks source_crop_row_ids.")
    source_rows = np.asarray(run_group["source_crop_row_ids"][:], dtype=np.int64).reshape(-1)
    if source_rows.size and (source_rows.min() < 0 or source_rows.max() >= crop_rows):
        raise CompositeSubjectMaskError(f"{label} references a crop row out of bounds.")
    run_keys = _unique_keys(run_group, label=label, rows=source_rows.shape[0])
    if not np.array_equal(run_keys, crop_keys[source_rows]):
        raise CompositeSubjectMaskError(f"{label} keys do not match its source crop rows.")
    signatures = np.asarray(crop[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8)[source_rows]
    return signatures, load_row_source_signature_spec(crop.attrs).spec_digest


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
    if _int_attr(run_group, "composite_subject_mask_schema_version") != COMPOSITE_SUBJECT_MASK_SCHEMA_VERSION:
        raise CompositeSubjectMaskError("Composite subject-mask schema version is unsupported.")
    if _int_attr(run_group, "composite_reference_depth") != 1:
        raise CompositeSubjectMaskError("Composite subject-mask reference depth must be one.")
    if require_complete and run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise CompositeSubjectMaskError("Composite subject-mask run is not complete.")
    if "mask_probs_roi" in run_group:
        raise CompositeSubjectMaskError("Composite run must not expose a partial top-level probability array.")

    mask_parent = root.get("subject_mask_runs")
    base_name = _text(run_group.attrs.get("composite_base_subject_mask_run"))
    if mask_parent is None or not base_name or base_name == run_name or base_name not in mask_parent:
        raise CompositeSubjectMaskError("Composite subject-mask base is missing or invalid.")
    base = mask_parent[base_name]
    if not is_run_complete_in_parent(mask_parent, base):
        raise CompositeSubjectMaskError("Composite subject-mask base is not complete.")
    if _text(base.attrs.get("subject_mask_storage_mode")) == COMPOSITE_SUBJECT_MASK_STORAGE_MODE:
        raise CompositeSubjectMaskError("Composite subject-mask bases cannot themselves be composite.")
    if "mask_probs_roi" not in base:
        raise CompositeSubjectMaskError("Composite subject-mask base lacks mask_probs_roi.")
    base_surface = base["mask_probs_roi"]
    if len(base_surface.shape) != 4:
        raise CompositeSubjectMaskError("Base probability surface must have shape (N,C,H,W).")

    crop_parent = root.get("crop_runs")
    target_crop_name = _text(run_group.attrs.get("source_crop_run"))
    if crop_parent is None or not target_crop_name or target_crop_name not in crop_parent:
        raise CompositeSubjectMaskError("Composite target crop is missing.")
    target_crop = crop_parent[target_crop_name]
    if not is_run_complete_in_parent(crop_parent, target_crop):
        raise CompositeSubjectMaskError("Composite target crop is not complete.")
    target_rows = int(target_crop["instance_key"].shape[0]) if "instance_key" in target_crop else -1
    target_keys = _unique_keys(target_crop, label=f"crop_runs/{target_crop_name}", rows=target_rows)
    output_keys = _unique_keys(run_group, label="Composite subject-mask run", rows=target_rows)
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
    if source_codes.shape != (target_rows,) or source_rows.shape != (target_rows,):
        raise CompositeSubjectMaskError("Composite source mapping does not cover every target row.")
    if np.any(~np.isin(source_codes, [COMPOSITE_SUBJECT_MASK_SOURCE_BASE, COMPOSITE_SUBJECT_MASK_SOURCE_DELTA])):
        raise CompositeSubjectMaskError("Composite source mapping contains an unknown code.")
    if np.any(source_rows < 0):
        raise CompositeSubjectMaskError("Composite source mapping contains a negative row.")
    base_targets = np.flatnonzero(source_codes == COMPOSITE_SUBJECT_MASK_SOURCE_BASE)
    delta_targets = np.flatnonzero(source_codes == COMPOSITE_SUBJECT_MASK_SOURCE_DELTA)
    if base_targets.size and int(source_rows[base_targets].max()) >= int(base_surface.shape[0]):
        raise CompositeSubjectMaskError("Composite mapping references a base row out of range.")
    if np.unique(source_rows[base_targets]).shape[0] != base_targets.shape[0]:
        raise CompositeSubjectMaskError("Composite mapping reuses one base row more than once.")
    if not np.array_equal(np.sort(source_rows[delta_targets]), np.arange(delta_targets.shape[0], dtype=np.int64)):
        raise CompositeSubjectMaskError("Composite mapping does not reference each delta row once.")
    declared_targets = np.asarray(payload["delta_target_row_indices"][:], dtype=np.int64).reshape(-1)
    declared_keys = np.asarray(payload["delta_instance_key"][:], dtype=np.uint64).reshape(-1)
    if not np.array_equal(declared_targets, delta_targets) or not np.array_equal(declared_keys, target_keys[delta_targets]):
        raise CompositeSubjectMaskError("Composite delta identity does not match its target rows.")
    delta_surface = payload["mask_probs_roi_delta"]
    expected_trailing = tuple(int(value) for value in base_surface.shape[1:])
    if tuple(int(value) for value in delta_surface.shape) != (int(delta_targets.shape[0]), *expected_trailing):
        raise CompositeSubjectMaskError("Composite delta probability shape differs from its base.")
    if np.dtype(delta_surface.dtype) != np.dtype(base_surface.dtype):
        raise CompositeSubjectMaskError("Composite delta probability dtype differs from its base.")

    if verify_identity and base_targets.size:
        base_rows = int(base_surface.shape[0])
        base_keys = _unique_keys(base, label=f"subject_mask_runs/{base_name}", rows=base_rows)
        base_signatures, base_spec_digest = _run_source_signatures(
            root, base, label=f"subject_mask_runs/{base_name}"
        )
        if ROW_SOURCE_SIGNATURE_ARRAY not in target_crop:
            raise CompositeSubjectMaskError("Composite target crop lacks signed rows.")
        validate_row_source_signature_array(
            target_crop[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=target_rows
        )
        target_signatures = np.asarray(target_crop[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8)
        target_spec_digest = load_row_source_signature_spec(target_crop.attrs).spec_digest
        mapped_base_rows = source_rows[base_targets]
        if base_spec_digest != target_spec_digest:
            raise CompositeSubjectMaskError("Composite reused rows use different signature specifications.")
        if not np.array_equal(target_keys[base_targets], base_keys[mapped_base_rows]):
            raise CompositeSubjectMaskError("Composite reused keys do not match base rows.")
        if not np.array_equal(target_signatures[base_targets], base_signatures[mapped_base_rows]):
            raise CompositeSubjectMaskError("Composite reused crop signatures do not match base rows.")

    return CompositeSubjectMaskValidation(
        run_name=run_name,
        base_run_name=base_name,
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
    """Read-only probability-array view over one base and one delta payload."""

    def __init__(
        self,
        *,
        base_array: Any,
        delta_array: Any,
        source_codes: np.ndarray,
        source_row_indices: np.ndarray,
    ) -> None:
        self._base_array = base_array
        self._delta_array = delta_array
        self._source_codes = np.asarray(source_codes, dtype=np.uint8).reshape(-1)
        self._source_row_indices = np.asarray(source_row_indices, dtype=np.int64).reshape(-1)
        self.shape = (int(self._source_codes.shape[0]), *tuple(int(v) for v in base_array.shape[1:]))
        self.ndim = 4
        self.dtype = np.dtype(base_array.dtype)

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
        base = root["subject_mask_runs"][validation.base_run_name]
        payload = run_group[COMPOSITE_SUBJECT_MASK_PAYLOAD_GROUP]
        return cls(
            base_array=base["mask_probs_roi"],
            delta_array=payload["mask_probs_roi_delta"],
            source_codes=np.asarray(payload["source_codes"][:], dtype=np.uint8),
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
        base_positions = np.flatnonzero(codes == COMPOSITE_SUBJECT_MASK_SOURCE_BASE)
        delta_positions = np.flatnonzero(codes == COMPOSITE_SUBJECT_MASK_SOURCE_DELTA)
        self._read_rows_into(self._base_array, mapped[base_positions], output, base_positions, trailing)
        self._read_rows_into(self._delta_array, mapped[delta_positions], output, delta_positions, trailing)
        return output[0] if scalar else output


def find_composite_subject_mask_dependents(mask_parent: Any, base_run_name: str) -> tuple[str, ...]:
    dependents: list[str] = []
    names = mask_parent.group_keys() if hasattr(mask_parent, "group_keys") else mask_parent.keys()
    for name in names:
        group = mask_parent[str(name)]
        if (
            _text(group.attrs.get("subject_mask_storage_mode")) == COMPOSITE_SUBJECT_MASK_STORAGE_MODE
            and _text(group.attrs.get("composite_base_subject_mask_run")) == _text(base_run_name)
        ):
            dependents.append(str(name))
    return tuple(sorted(dependents))


def assert_subject_mask_run_unreferenced(mask_parent: Any, run_name: str) -> None:
    dependents = find_composite_subject_mask_dependents(mask_parent, run_name)
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
