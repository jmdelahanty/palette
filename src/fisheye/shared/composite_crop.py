"""Depth-one immutable base-plus-delta storage for dense crop pixels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

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


COMPOSITE_CROP_SCHEMA_ID = "palette.composite_crop"
COMPOSITE_CROP_SCHEMA_VERSION = 1
COMPOSITE_CROP_STORAGE_MODE = "composite"
COMPOSITE_CROP_MAX_REFERENCE_DEPTH = 1
COMPOSITE_CROP_PAYLOAD_GROUP = "composite_payload"
COMPOSITE_CROP_SOURCE_BASE = 0
COMPOSITE_CROP_SOURCE_DELTA = 1
COMPOSITE_CROP_SOURCE_CODE_MAP = {
    "base": COMPOSITE_CROP_SOURCE_BASE,
    "delta": COMPOSITE_CROP_SOURCE_DELTA,
}
COMPOSITE_CROP_READ_MAX_BATCH_BYTES = 64 * 1024 * 1024


class CompositeCropError(RuntimeError):
    """Raised when a composite crop cannot be resolved without ambiguity."""


@dataclass(frozen=True)
class CompositeCropValidation:
    run_name: str | None
    base_run_name: str
    target_rows: int
    base_rows_used: int
    delta_rows: int
    roi_shape: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "base_run_name": self.base_run_name,
            "target_rows": self.target_rows,
            "base_rows_used": self.base_rows_used,
            "delta_rows": self.delta_rows,
            "roi_shape": list(self.roi_shape),
        }


def _text(value: object) -> str:
    return str(value or "").strip()


def _group_names(parent: Any) -> list[str]:
    if hasattr(parent, "group_keys"):
        return sorted(str(value) for value in parent.group_keys())
    return sorted(str(value) for value in parent.keys())


def _roi_shape_from_attrs(group: Any) -> tuple[int, int]:
    value = group.attrs.get("roi_size")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CompositeCropError("Composite crop requires roi_size=[height,width].")
    shape = (int(value[0]), int(value[1]))
    if min(shape) <= 0:
        raise CompositeCropError("Composite crop roi_size must be positive.")
    return shape


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CompositeCropError(
            "Composite ROI pixel contract is not strict JSON."
        ) from exc


def _int_attr(group: Any, name: str, *, default: int = -1) -> int:
    try:
        return int(group.attrs.get(name, default))
    except (TypeError, ValueError) as exc:
        raise CompositeCropError(
            f"Composite crop attr {name!r} must be an integer."
        ) from exc


def _require_complete_standalone_base(
    crop_parent: Any,
    *,
    base_run_name: str,
    target_run_name: str | None,
) -> Any:
    if not base_run_name or base_run_name == target_run_name:
        raise CompositeCropError("Composite crop must reference a different named base run.")
    if base_run_name not in crop_parent:
        raise CompositeCropError(f"Composite crop base run {base_run_name!r} does not exist.")
    base = crop_parent[base_run_name]
    if not is_run_complete_in_parent(crop_parent, base):
        raise CompositeCropError(f"Composite crop base run {base_run_name!r} is not complete.")
    base_mode = _text(base.attrs.get("crop_storage_mode"))
    if base_mode == COMPOSITE_CROP_STORAGE_MODE:
        raise CompositeCropError(
            "Composite crop reference depth would exceed one; compact the base first."
        )
    if base_mode not in {"", "materialized"} or "roi_images" not in base:
        raise CompositeCropError("Composite crop base must be a standalone materialized run.")
    if np.dtype(base["roi_images"].dtype) != np.dtype(np.uint8):
        raise CompositeCropError("Composite crop base roi_images must use uint8 dtype.")
    return base


def validate_composite_crop_run(
    crop_parent: Any,
    run_group: Any,
    *,
    run_name: str | None = None,
    require_complete: bool = True,
    verify_identity: bool = True,
) -> CompositeCropValidation:
    """Validate a complete logical row mapping and its immutable base contract."""

    if _text(run_group.attrs.get("crop_storage_mode")) != COMPOSITE_CROP_STORAGE_MODE:
        raise CompositeCropError("Crop run is not explicitly declared composite.")
    if _text(run_group.attrs.get("composite_crop_schema_id")) != COMPOSITE_CROP_SCHEMA_ID:
        raise CompositeCropError("Composite crop schema id is missing or unsupported.")
    if _int_attr(run_group, "composite_crop_schema_version") != COMPOSITE_CROP_SCHEMA_VERSION:
        raise CompositeCropError("Composite crop schema version is unsupported.")
    if _int_attr(run_group, "composite_reference_depth") != 1:
        raise CompositeCropError("Composite crop must have reference depth exactly one.")
    if "roi_images" in run_group:
        raise CompositeCropError(
            "Composite crop must not expose a partial top-level roi_images array."
        )
    if require_complete and run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise CompositeCropError("Composite crop run is not explicitly complete.")

    base_run_name = _text(run_group.attrs.get("composite_base_crop_run"))
    base = _require_complete_standalone_base(
        crop_parent,
        base_run_name=base_run_name,
        target_run_name=run_name,
    )
    target_roi_shape = _roi_shape_from_attrs(run_group)
    base_shape = tuple(int(value) for value in base["roi_images"].shape)
    if len(base_shape) != 3 or base_shape[1:] != target_roi_shape:
        raise CompositeCropError("Composite base and target ROI shapes differ.")
    if COMPOSITE_CROP_PAYLOAD_GROUP not in run_group:
        raise CompositeCropError("Composite crop is missing composite_payload.")
    payload = run_group[COMPOSITE_CROP_PAYLOAD_GROUP]
    required = (
        "source_codes",
        "source_row_indices",
        "delta_target_row_indices",
        "delta_instance_key",
        "roi_images_delta",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise CompositeCropError("Composite crop payload is incomplete: " + ", ".join(missing))
    if "instance_key" not in run_group or ROW_SOURCE_SIGNATURE_ARRAY not in run_group:
        raise CompositeCropError("Composite crop target identity/signatures are missing.")
    target_rows = int(run_group["instance_key"].shape[0])
    validate_row_source_signature_array(
        run_group[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=target_rows
    )
    source_codes = np.asarray(payload["source_codes"][:], dtype=np.uint8).reshape(-1)
    source_rows = np.asarray(payload["source_row_indices"][:], dtype=np.int64).reshape(-1)
    if np.dtype(payload["source_codes"].dtype) != np.dtype(np.uint8):
        raise CompositeCropError("Composite source_codes must use uint8 dtype.")
    if np.dtype(payload["source_row_indices"].dtype) != np.dtype(np.int64):
        raise CompositeCropError("Composite source_row_indices must use int64 dtype.")
    if source_codes.shape != (target_rows,) or source_rows.shape != (target_rows,):
        raise CompositeCropError("Composite source mapping does not cover every target row.")
    if np.any(
        ~np.isin(
            source_codes,
            [COMPOSITE_CROP_SOURCE_BASE, COMPOSITE_CROP_SOURCE_DELTA],
        )
    ):
        raise CompositeCropError("Composite source mapping contains an unknown source code.")
    if np.any(source_rows < 0):
        raise CompositeCropError("Composite source mapping contains negative row indices.")

    base_target_rows = np.flatnonzero(source_codes == COMPOSITE_CROP_SOURCE_BASE)
    delta_target_rows = np.flatnonzero(source_codes == COMPOSITE_CROP_SOURCE_DELTA)
    base_source_rows = source_rows[base_target_rows]
    delta_source_rows = source_rows[delta_target_rows]
    if base_source_rows.size:
        if int(base_source_rows.max()) >= base_shape[0]:
            raise CompositeCropError("Composite mapping references a base row out of bounds.")
        if np.unique(base_source_rows).shape[0] != base_source_rows.shape[0]:
            raise CompositeCropError("Composite mapping references one base row more than once.")
    delta_count = int(delta_target_rows.shape[0])
    delta_shape = tuple(int(value) for value in payload["roi_images_delta"].shape)
    if delta_shape != (delta_count, *target_roi_shape):
        raise CompositeCropError("Composite delta ROI shape does not match its mapping.")
    if np.dtype(payload["roi_images_delta"].dtype) != np.dtype(np.uint8):
        raise CompositeCropError("Composite delta ROI payload must use uint8 dtype.")
    if not np.array_equal(np.sort(delta_source_rows), np.arange(delta_count, dtype=np.int64)):
        raise CompositeCropError("Composite mapping does not reference each delta row exactly once.")
    declared_delta_targets = np.asarray(
        payload["delta_target_row_indices"][:], dtype=np.int64
    ).reshape(-1)
    if np.dtype(payload["delta_target_row_indices"].dtype) != np.dtype(np.int64):
        raise CompositeCropError("Composite delta target rows must use int64 dtype.")
    if not np.array_equal(declared_delta_targets, delta_target_rows):
        raise CompositeCropError("Composite delta target rows disagree with source_codes.")
    target_keys = np.asarray(run_group["instance_key"][:], dtype=np.uint64).reshape(-1)
    if target_keys.shape != (target_rows,) or np.unique(target_keys).shape[0] != target_rows:
        raise CompositeCropError("Composite target instance_key values must be unique.")
    delta_keys = np.asarray(payload["delta_instance_key"][:], dtype=np.uint64).reshape(-1)
    if np.dtype(payload["delta_instance_key"].dtype) != np.dtype(np.uint64):
        raise CompositeCropError("Composite delta instance_key must use uint64 dtype.")
    if not np.array_equal(delta_keys, target_keys[delta_target_rows]):
        raise CompositeCropError("Composite delta instance_key values disagree with target rows.")

    if verify_identity and base_target_rows.size:
        if "instance_key" not in base or ROW_SOURCE_SIGNATURE_ARRAY not in base:
            raise CompositeCropError("Composite base identity/signatures are missing.")
        base_rows = int(base["instance_key"].shape[0])
        validate_row_source_signature_array(
            base[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=base_rows
        )
        target_spec = load_row_source_signature_spec(run_group.attrs)
        base_spec = load_row_source_signature_spec(base.attrs)
        if target_spec.spec_digest != base_spec.spec_digest:
            raise CompositeCropError("Composite reused rows were signed under different specs.")
        base_keys = np.asarray(base["instance_key"][:], dtype=np.uint64).reshape(-1)
        if not np.array_equal(target_keys[base_target_rows], base_keys[base_source_rows]):
            raise CompositeCropError("Composite reused keys do not match their base rows.")
        target_signatures = np.asarray(
            run_group[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8
        )
        base_signatures = np.asarray(base[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8)
        if not np.array_equal(
            target_signatures[base_target_rows], base_signatures[base_source_rows]
        ):
            raise CompositeCropError("Composite reused signatures do not match their base rows.")

    target_contract = run_group.attrs.get("roi_pixel_contract")
    base_contract = base.attrs.get("roi_pixel_contract")
    if target_contract is None or base_contract is None:
        raise CompositeCropError(
            "Composite base and target must declare ROI pixel contracts."
        )
    if _canonical_json(target_contract) != _canonical_json(base_contract):
        raise CompositeCropError("Composite base and target ROI pixel contracts differ.")
    return CompositeCropValidation(
        run_name=run_name,
        base_run_name=base_run_name,
        target_rows=target_rows,
        base_rows_used=int(base_target_rows.shape[0]),
        delta_rows=delta_count,
        roi_shape=target_roi_shape,
    )


def _read_rows_into(
    array: Any,
    source_rows: np.ndarray,
    output: np.ndarray,
    output_rows: np.ndarray,
) -> None:
    """Read coalesced source ranges directly into caller-owned output memory."""

    sources = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    destinations = np.asarray(output_rows, dtype=np.int64).reshape(-1)
    if sources.shape != destinations.shape:
        raise CompositeCropError("Composite source and destination read maps differ.")
    if sources.size == 0:
        return
    shape = tuple(int(value) for value in array.shape)
    row_bytes = max(1, int(np.prod(shape[1:], dtype=np.int64)))
    max_source_rows = max(1, COMPOSITE_CROP_READ_MAX_BATCH_BYTES // row_bytes)
    order = np.argsort(sources, kind="stable")
    sorted_sources = sources[order]
    sorted_destinations = destinations[order]
    start = 0
    while start < sorted_sources.shape[0]:
        stop = start + 1
        source_start = int(sorted_sources[start])
        while (
            stop < sorted_sources.shape[0]
            and sorted_sources[stop] <= sorted_sources[stop - 1] + 1
            and int(sorted_sources[stop]) - source_start < max_source_rows
        ):
            stop += 1
        source_stop = int(sorted_sources[stop - 1]) + 1
        payload = np.asarray(
            array[source_start:source_stop], dtype=np.uint8
        )
        payload_rows = sorted_sources[start:stop] - source_start
        output[sorted_destinations[start:stop]] = payload[payload_rows]
        start = stop


class CompositeCropArray:
    """Read-only array-like resolver over one standalone base and one delta."""

    def __init__(
        self,
        *,
        base_array: Any,
        delta_array: Any,
        source_codes: np.ndarray,
        source_row_indices: np.ndarray,
        roi_shape: tuple[int, int],
    ) -> None:
        self._base_array = base_array
        self._delta_array = delta_array
        self._source_codes = np.asarray(source_codes, dtype=np.uint8).reshape(-1)
        self._source_row_indices = np.asarray(
            source_row_indices, dtype=np.int64
        ).reshape(-1)
        self.shape = (int(self._source_codes.shape[0]), *roi_shape)
        self.ndim = 3
        self.dtype = np.dtype(np.uint8)

    @classmethod
    def open(
        cls,
        crop_parent: Any,
        run_group: Any,
        *,
        run_name: str | None = None,
        verify_identity: bool = False,
    ) -> "CompositeCropArray":
        validation = validate_composite_crop_run(
            crop_parent,
            run_group,
            run_name=run_name,
            require_complete=True,
            verify_identity=verify_identity,
        )
        payload = run_group[COMPOSITE_CROP_PAYLOAD_GROUP]
        base = crop_parent[validation.base_run_name]
        return cls(
            base_array=base["roi_images"],
            delta_array=payload["roi_images_delta"],
            source_codes=np.asarray(payload["source_codes"][:], dtype=np.uint8),
            source_row_indices=np.asarray(
                payload["source_row_indices"][:], dtype=np.int64
            ),
            roi_shape=validation.roi_shape,
        )

    def __len__(self) -> int:
        return int(self.shape[0])

    def read_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        target_rows = np.asarray(indices, dtype=np.int64).reshape(-1)
        if target_rows.size == 0:
            return np.empty((0, *self.shape[1:]), dtype=np.uint8)
        if target_rows.min() < 0 or target_rows.max() >= self.shape[0]:
            raise IndexError("Composite crop row is out of bounds.")
        output = np.empty((target_rows.shape[0], *self.shape[1:]), dtype=np.uint8)
        codes = self._source_codes[target_rows]
        mapped_rows = self._source_row_indices[target_rows]
        base_positions = np.flatnonzero(codes == COMPOSITE_CROP_SOURCE_BASE)
        delta_positions = np.flatnonzero(codes == COMPOSITE_CROP_SOURCE_DELTA)
        if base_positions.size:
            _read_rows_into(
                self._base_array,
                mapped_rows[base_positions],
                output,
                base_positions,
            )
        if delta_positions.size:
            _read_rows_into(
                self._delta_array,
                mapped_rows[delta_positions],
                output,
                delta_positions,
            )
        return output

    def read_slice(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop < start or stop > self.shape[0]:
            raise IndexError(f"Invalid composite crop slice [{start}:{stop}].")
        return self.read_indices(np.arange(start, stop, dtype=np.int64))

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step == 1:
                return self.read_slice(start, stop)
            return self.read_indices(np.arange(start, stop, step, dtype=np.int64))
        if isinstance(key, (list, tuple, np.ndarray)):
            return self.read_indices(key)
        index = int(key)  # type: ignore[arg-type]
        if index < 0:
            index += self.shape[0]
        return self.read_indices(np.asarray([index], dtype=np.int64))[0]


def find_composite_crop_dependents(
    crop_parent: Any,
    base_run_name: str,
) -> tuple[str, ...]:
    """Return complete, running, or failed composite runs that retain a base."""

    base_name = _text(base_run_name)
    dependents: list[str] = []
    for candidate in _group_names(crop_parent):
        if candidate == base_name:
            continue
        group = crop_parent[candidate]
        if _text(group.attrs.get("crop_storage_mode")) != COMPOSITE_CROP_STORAGE_MODE:
            continue
        if _text(group.attrs.get("composite_base_crop_run")) == base_name:
            dependents.append(candidate)
    return tuple(sorted(dependents))


def assert_crop_run_unreferenced(crop_parent: Any, run_name: str) -> None:
    """Fail closed before deleting a crop run retained by composite children."""

    dependents = find_composite_crop_dependents(crop_parent, run_name)
    if dependents:
        raise CompositeCropError(
            f"Refusing to delete crop run {run_name!r}; composite dependents: "
            + ", ".join(dependents)
        )


__all__ = [
    "COMPOSITE_CROP_SCHEMA_ID",
    "COMPOSITE_CROP_SCHEMA_VERSION",
    "COMPOSITE_CROP_STORAGE_MODE",
    "COMPOSITE_CROP_MAX_REFERENCE_DEPTH",
    "COMPOSITE_CROP_PAYLOAD_GROUP",
    "COMPOSITE_CROP_SOURCE_BASE",
    "COMPOSITE_CROP_SOURCE_DELTA",
    "COMPOSITE_CROP_SOURCE_CODE_MAP",
    "COMPOSITE_CROP_READ_MAX_BATCH_BYTES",
    "CompositeCropError",
    "CompositeCropValidation",
    "CompositeCropArray",
    "validate_composite_crop_run",
    "find_composite_crop_dependents",
    "assert_crop_run_unreferenced",
]
