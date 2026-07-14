"""Bounded, read-only helpers for a generic Marimo Zarr workspace.

This module deliberately knows nothing about Palette visualization contracts.
It gives people and pairing agents a small vocabulary for inspecting an
arbitrary selected Zarr without eagerly loading dense arrays.  The deployment
launcher provides the actual write boundary by mounting the source directory
read-only; these helpers consistently open it with ``mode="r"`` as well.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

from fisheye.shared.zarr_io import open_zarr_root


DEFAULT_MAX_READ_ELEMENTS = 100_000
DEFAULT_MAX_INVENTORY_ITEMS = 250
DEFAULT_MAX_TABLE_ROWS = 10_000
DEFAULT_MAX_TRACE_POINTS = 5_000
DEFAULT_MAX_TRACE_SOURCE_ROWS = 100_000

_DENSE_CHANNEL_INDEX_LAYOUTS = {
    "frame_angles": ("angle_channel_index", "frame_available"),
    "roi_angles": ("angle_channel_index", "roi_available"),
    "frame_qa": ("qa_channel_index", "frame_available"),
    "roi_qa": ("qa_channel_index", "roi_available"),
}


def _normalise_path(path: str | None) -> str:
    value = str(path or "").strip()
    if value in {"", "/"}:
        return ""
    if value.startswith("/"):
        raise ValueError("Zarr node paths must be relative to the selected root.")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid relative Zarr node path: {value!r}")
    return "/".join(parts)


def _is_array(node: Any) -> bool:
    return hasattr(node, "shape") and hasattr(node, "dtype")


def _is_group(node: Any) -> bool:
    return callable(getattr(node, "keys", None)) and not _is_array(node)


def _shape_tuple(node: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in getattr(node, "shape", ()))


def _bounded_attrs(
    node: Any,
    *,
    max_items: int,
    max_value_chars: int,
) -> dict[str, Any]:
    attrs = getattr(node, "attrs", {})
    try:
        items = list(islice(iter(attrs.items()), max_items + 1))
    except (AttributeError, TypeError):
        return {}
    result: dict[str, Any] = {}
    for key, value in items[:max_items]:
        rendered = repr(value)
        if len(rendered) > max_value_chars:
            result[str(key)] = rendered[:max_value_chars] + "…"
        else:
            result[str(key)] = value
    if len(items) > max_items:
        result["__truncated__"] = "additional attributes omitted"
    return result


def _expand_selection(
    shape: tuple[int, ...], selection: Any | None
) -> tuple[tuple[int | slice, ...], int]:
    """Validate basic NumPy indexing and estimate its selected element count."""

    if selection is None:
        raw: tuple[Any, ...] = tuple(slice(None) for _ in shape)
    elif isinstance(selection, tuple):
        raw = selection
    else:
        raw = (selection,)

    ellipsis_count = sum(value is Ellipsis for value in raw)
    if ellipsis_count > 1:
        raise ValueError("A selection may contain at most one ellipsis.")
    if ellipsis_count:
        ellipsis_index = raw.index(Ellipsis)
        explicit = len(raw) - 1
        if explicit > len(shape):
            raise IndexError("Selection has more axes than the array.")
        raw = (
            raw[:ellipsis_index]
            + tuple(slice(None) for _ in range(len(shape) - explicit))
            + raw[ellipsis_index + 1 :]
        )
    if len(raw) > len(shape):
        raise IndexError("Selection has more axes than the array.")
    raw = raw + tuple(slice(None) for _ in range(len(shape) - len(raw)))

    normalised: list[int | slice] = []
    element_count = 1
    for axis, (axis_size, indexer) in enumerate(zip(shape, raw, strict=True)):
        if isinstance(indexer, (int, np.integer)):
            index = int(indexer)
            if index < 0:
                index += axis_size
            if index < 0 or index >= axis_size:
                raise IndexError(f"Index {indexer} is outside axis {axis} of size {axis_size}.")
            normalised.append(index)
            continue
        if not isinstance(indexer, slice):
            raise TypeError(
                "Bounded reads accept only integers, slices, and one ellipsis; "
                "fancy indexing is intentionally excluded."
            )
        start, stop, step = indexer.indices(axis_size)
        length = len(range(start, stop, step))
        element_count *= length
        normalised.append(slice(start, stop, step))
    return tuple(normalised), element_count


@dataclass(frozen=True, repr=False)
class ZarrExplorationWorkspace:
    """Read-only, bounded access to one selected source Zarr."""

    zarr_path: Path
    _root: Any
    max_read_elements: int = DEFAULT_MAX_READ_ELEMENTS

    @classmethod
    def open(
        cls,
        zarr_path: str | Path,
        *,
        max_read_elements: int = DEFAULT_MAX_READ_ELEMENTS,
    ) -> "ZarrExplorationWorkspace":
        path = Path(zarr_path)
        if max_read_elements < 1:
            raise ValueError("max_read_elements must be positive.")
        root = open_zarr_root(path, mode="r")
        return cls(
            zarr_path=path,
            _root=root,
            max_read_elements=int(max_read_elements),
        )

    def _node(self, path: str | None = "") -> Any:
        relative = _normalise_path(path)
        return self._root if not relative else self._root[relative]

    def handle(self, path: str | None = "") -> Any:
        """Return a lazy Zarr group/array handle without reading array values."""

        return self._node(path)

    def info(self, path: str | None = "") -> dict[str, Any]:
        """Describe one group or array using metadata only."""

        relative = _normalise_path(path)
        node = self._node(relative)
        if _is_array(node):
            shape = _shape_tuple(node)
            return {
                "path": relative or "/",
                "kind": "array",
                "shape": shape,
                "dtype": str(getattr(node, "dtype", "unknown")),
                "chunks": tuple(int(value) for value in (getattr(node, "chunks", ()) or ())),
                "ndim": len(shape),
                "elements": int(np.prod(shape, dtype=np.int64)) if shape else 1,
                "nbytes": int(getattr(node, "nbytes", 0) or 0),
            }
        if _is_group(node):
            return {"path": relative or "/", "kind": "group"}
        return {"path": relative or "/", "kind": type(node).__name__}

    def attrs(
        self,
        path: str | None = "",
        *,
        max_items: int = 100,
        max_value_chars: int = 2_000,
    ) -> dict[str, Any]:
        """Return a display-bounded copy of a node's attributes."""

        if max_items < 1 or max_value_chars < 1:
            raise ValueError("Attribute limits must be positive.")
        return _bounded_attrs(
            self._node(path),
            max_items=int(max_items),
            max_value_chars=int(max_value_chars),
        )

    def ls(
        self,
        path: str | None = "",
        *,
        max_items: int = DEFAULT_MAX_INVENTORY_ITEMS,
    ) -> list[dict[str, Any]]:
        """List direct children of a group, stopping at ``max_items``."""

        if max_items < 1:
            raise ValueError("max_items must be positive.")
        parent_path = _normalise_path(path)
        group = self._node(parent_path)
        if not _is_group(group):
            raise TypeError(f"Zarr node {parent_path or '/'} is not a group.")
        rows: list[dict[str, Any]] = []
        for name in group.keys():
            if len(rows) >= max_items:
                break
            child_path = f"{parent_path}/{name}" if parent_path else str(name)
            rows.append(self.info(child_path))
        return sorted(rows, key=lambda row: str(row["path"]))

    def walk(
        self,
        path: str | None = "",
        *,
        max_depth: int = 2,
        max_items: int = DEFAULT_MAX_INVENTORY_ITEMS,
    ) -> list[dict[str, Any]]:
        """Build a bounded breadth-first metadata inventory."""

        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        if max_items < 1:
            raise ValueError("max_items must be positive.")
        start = _normalise_path(path)
        start_node = self._node(start)
        if not _is_group(start_node):
            return [self.info(start)]

        rows: list[dict[str, Any]] = []
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue and len(rows) < max_items:
            group_path, depth = queue.popleft()
            for row in self.ls(group_path, max_items=max_items - len(rows)):
                rows.append(row)
                if row["kind"] == "group" and depth < max_depth:
                    queue.append((str(row["path"]), depth + 1))
                if len(rows) >= max_items:
                    break
        return rows

    def find(
        self,
        text: str,
        *,
        path: str | None = "",
        max_depth: int = 8,
        max_items: int = 2_000,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search a bounded metadata inventory by case-insensitive node path."""

        query = str(text).strip().casefold()
        if not query:
            return []
        return [
            row
            for row in self.walk(path, max_depth=max_depth, max_items=max_items)
            if query in str(row["path"]).casefold()
        ][:max_results]

    def read(
        self,
        path: str,
        selection: Any | None = None,
        *,
        max_elements: int | None = None,
    ) -> np.ndarray:
        """Read an explicitly bounded array selection into a NumPy array."""

        relative = _normalise_path(path)
        array = self._node(relative)
        if not _is_array(array):
            raise TypeError(f"Zarr node {relative or '/'} is not an array.")
        limit = self.max_read_elements if max_elements is None else int(max_elements)
        if limit < 1:
            raise ValueError("max_elements must be positive.")
        indexer, element_count = _expand_selection(_shape_tuple(array), selection)
        if element_count > limit:
            raise ValueError(
                f"Selection requests {element_count:,} elements from {relative!r}; "
                f"the current limit is {limit:,}. Pass a smaller slice."
            )
        return np.asarray(array[indexer])

    def head(
        self,
        path: str,
        rows: int = 100,
        *,
        max_elements: int | None = None,
    ) -> np.ndarray:
        """Read the first ``rows`` entries along an array's leading axis."""

        if rows < 0:
            raise ValueError("rows cannot be negative.")
        array = self._node(path)
        shape = _shape_tuple(array)
        selection: Any = () if not shape else slice(0, min(int(rows), shape[0]))
        return self.read(path, selection, max_elements=max_elements)

    def to_polars(
        self,
        group_path: str | None = "",
        *,
        columns: Sequence[str] | None = None,
        start: int = 0,
        stop: int = 1_000,
        max_rows: int = DEFAULT_MAX_TABLE_ROWS,
        max_columns: int = 100,
    ) -> pl.DataFrame:
        """Load sibling one-dimensional arrays as one bounded Polars table."""

        if start < 0 or stop < start:
            raise ValueError("Expected 0 <= start <= stop.")
        if max_rows < 1 or max_columns < 1:
            raise ValueError("Table limits must be positive.")
        if stop - start > max_rows:
            raise ValueError(
                f"Requested {stop - start:,} rows; the table limit is {max_rows:,}."
            )
        relative = _normalise_path(group_path)
        group = self._node(relative)
        if not _is_group(group):
            raise TypeError(f"Zarr node {relative or '/'} is not a group.")

        requested = [str(name) for name in columns] if columns is not None else None
        candidate_names: Iterable[str]
        if requested is not None:
            candidate_names = requested
        else:
            candidate_names = islice(group.keys(), max_columns * 10)
        data: dict[str, np.ndarray] = {}
        expected_length: int | None = None
        for name in candidate_names:
            if len(data) >= max_columns:
                break
            node = group[name]
            if not _is_array(node):
                if requested is not None:
                    raise TypeError(f"{name!r} is not an array in group {relative or '/'}.")
                continue
            shape = _shape_tuple(node)
            if len(shape) != 1:
                if requested is not None:
                    raise ValueError(f"{name!r} is {len(shape)}D; Polars columns must be 1D.")
                continue
            if expected_length is None:
                expected_length = shape[0]
            elif shape[0] != expected_length:
                if requested is not None:
                    raise ValueError("Selected arrays do not share the same leading length.")
                continue
            data[name] = self.read(
                f"{relative}/{name}" if relative else name,
                slice(start, min(stop, shape[0])),
                max_elements=max_rows,
            )
        if not data:
            raise ValueError(f"No compatible one-dimensional arrays found in {relative or '/'}.")
        return pl.DataFrame(data)

    def _fixed_width_strings(
        self,
        path: str,
        *,
        expected_rows: int | None = None,
    ) -> list[str]:
        """Decode Palette's bounded fixed-width uint8 string matrices."""

        info = self.info(path)
        shape = tuple(info.get("shape", ()))
        if len(shape) != 2 or info.get("dtype") != "uint8":
            raise ValueError(f"{path!r} is not a fixed-width uint8 string matrix.")
        if expected_rows is not None and shape[0] != expected_rows:
            raise ValueError(
                f"{path!r} has {shape[0]} rows; expected {expected_rows}."
            )
        if shape[0] > 2_000 or shape[1] > 4_096:
            raise ValueError(f"Refusing unusually large string index at {path!r}: {shape}.")
        raw = self.read(path, max_elements=shape[0] * shape[1])
        return [
            bytes(np.asarray(row, dtype=np.uint8)).split(b"\0", 1)[0].decode(
                "utf-8", errors="replace"
            )
            for row in raw
        ]

    def channel_index(
        self,
        array_path: str,
        *,
        available_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Resolve named columns for known compact dense Palette arrays.

        Unknown two-dimensional arrays return an empty list; callers may fall
        back to numeric column indices.
        """

        relative = _normalise_path(array_path)
        if "/" in relative:
            parent_path, array_name = relative.rsplit("/", 1)
        else:
            parent_path, array_name = "", relative
        layout = _DENSE_CHANNEL_INDEX_LAYOUTS.get(array_name)
        if layout is None:
            return []
        array_info = self.info(relative)
        shape = tuple(array_info.get("shape", ()))
        if len(shape) != 2:
            return []
        index_name, availability_name = layout
        index_path = f"{parent_path}/{index_name}" if parent_path else index_name
        names_path = f"{index_path}/name"
        availability_path = f"{index_path}/{availability_name}"
        try:
            names = self._fixed_width_strings(names_path, expected_rows=shape[1])
            availability = np.asarray(
                self.read(availability_path, max_elements=shape[1]), dtype=bool
            )
        except (KeyError, TypeError, ValueError):
            return []
        if availability.shape != (shape[1],):
            return []

        units: list[str]
        try:
            units = self._fixed_width_strings(
                f"{index_path}/units", expected_rows=shape[1]
            )
        except (KeyError, TypeError, ValueError):
            units = [""] * shape[1]
        return [
            {
                "index": index,
                "name": names[index],
                "units": units[index],
                "available": bool(availability[index]),
            }
            for index in range(shape[1])
            if not available_only or bool(availability[index])
        ]

    def suggested_coordinate_path(self, array_path: str) -> str | None:
        """Find a conventional one-dimensional coordinate matching axis zero."""

        relative = _normalise_path(array_path)
        info = self.info(relative)
        shape = tuple(info.get("shape", ()))
        if not shape:
            return None
        if "/" in relative:
            parent_path, array_name = relative.rsplit("/", 1)
        else:
            parent_path, array_name = "", relative
        prefix = f"{parent_path}/" if parent_path else ""
        if array_name.startswith("frame_"):
            candidates = (
                f"{prefix}support/frame_time_seconds",
                f"{prefix}frame_time_seconds",
                f"{prefix}time_seconds",
                f"{prefix}time_s",
            )
        elif array_name.startswith("roi_"):
            candidates = (
                f"{prefix}support/time_seconds",
                f"{prefix}time_seconds",
                f"{prefix}time_s",
            )
        else:
            candidates = (
                f"{prefix}time_seconds",
                f"{prefix}time_s",
                f"{prefix}timestamps",
            )
        for candidate in candidates:
            try:
                candidate_info = self.info(candidate)
            except (KeyError, TypeError, ValueError):
                continue
            if (
                candidate_info.get("kind") == "array"
                and tuple(candidate_info.get("shape", ())) == (shape[0],)
            ):
                return candidate
        return None

    def trace_frame(
        self,
        array_path: str,
        *,
        column: int | None = None,
        start: int = 0,
        stop: int | None = None,
        max_points: int = DEFAULT_MAX_TRACE_POINTS,
        max_source_rows: int = DEFAULT_MAX_TRACE_SOURCE_ROWS,
        coordinate_path: str | None = None,
    ) -> pl.DataFrame:
        """Read one bounded numeric trace into a compact Polars DataFrame."""

        relative = _normalise_path(array_path)
        array = self._node(relative)
        if not _is_array(array):
            raise TypeError(f"Zarr node {relative or '/'} is not an array.")
        shape = _shape_tuple(array)
        if len(shape) not in {1, 2}:
            raise ValueError("Trace plotting supports only one- or two-dimensional arrays.")
        try:
            numeric = np.issubdtype(np.dtype(getattr(array, "dtype")), np.number)
        except TypeError:
            numeric = False
        if not numeric:
            raise ValueError(f"Trace plotting requires a numeric dtype, not {array.dtype}.")
        if start < 0 or start >= shape[0]:
            raise ValueError(f"start must be between 0 and {max(0, shape[0] - 1):,}.")
        resolved_stop = min(shape[0], int(stop) if stop is not None else shape[0])
        if resolved_stop <= start:
            raise ValueError("stop must be greater than start.")
        if max_points < 2 or max_source_rows < 1:
            raise ValueError("Trace limits must be positive and max_points must be at least 2.")
        source_rows = resolved_stop - start
        if source_rows > max_source_rows:
            raise ValueError(
                f"Trace window spans {source_rows:,} source rows; the interactive "
                f"limit is {max_source_rows:,}. Use a smaller window."
            )
        if len(shape) == 2:
            if column is None or column < 0 or column >= shape[1]:
                raise ValueError(f"column must be between 0 and {shape[1] - 1}.")
        elif column is not None:
            raise ValueError("column is only valid for two-dimensional arrays.")

        stride = max(1, int(np.ceil(source_rows / max_points)))
        row_selection = slice(start, resolved_stop, stride)
        selection: Any = (
            row_selection if len(shape) == 1 else (row_selection, int(column))
        )
        values = np.asarray(
            self.read(relative, selection, max_elements=max_points), dtype=np.float64
        ).reshape(-1)
        row_indices = np.arange(start, resolved_stop, stride, dtype=np.int64)
        data: dict[str, np.ndarray] = {
            "row_index": row_indices,
            "value": values,
        }

        resolved_coordinate = (
            _normalise_path(coordinate_path)
            if coordinate_path is not None
            else self.suggested_coordinate_path(relative)
        )
        if resolved_coordinate:
            coordinate = np.asarray(
                self.read(
                    resolved_coordinate,
                    row_selection,
                    max_elements=max_points,
                ),
                dtype=np.float64,
            ).reshape(-1)
            if coordinate.shape != values.shape:
                raise ValueError("Coordinate and trace selections have different lengths.")
            data["time_seconds"] = coordinate
        return pl.DataFrame(data)

    def summary(self) -> dict[str, Any]:
        """Return a compact description safe for notebook display."""

        return {
            "zarr_path": str(self.zarr_path),
            "read_only": True,
            "max_read_elements": self.max_read_elements,
            "root_attrs": self.attrs(max_items=25, max_value_chars=500),
        }

    def __repr__(self) -> str:
        return (
            "ZarrExplorationWorkspace("
            f"zarr_path={str(self.zarr_path)!r}, "
            f"max_read_elements={self.max_read_elements:,})"
        )
