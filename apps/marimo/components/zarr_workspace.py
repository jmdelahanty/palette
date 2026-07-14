"""Bounded, read-only helpers for a generic Marimo Zarr workspace.

This module provides a generic bounded inspection vocabulary plus small
schema-aware adapters for recognized Palette analysis families. It never
requires a visualization contract and never eagerly loads dense arrays. The
deployment launcher provides the actual write boundary by mounting the source
directory read-only; these helpers consistently open it with ``mode="r"`` as
well.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

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

_CHANNEL_INDEX_TEXT_FIELDS = (
    "representation",
    "eye",
    "value_kind",
    "source_channel",
    "formula",
    "compatibility_alias_of",
)


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


def _group_names(group: Any, *, max_items: int) -> list[str]:
    if not _is_group(group):
        return []
    return sorted(str(name) for name in islice(group.keys(), int(max_items)))


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
class ZarrAnalysisDataset:
    """Semantic, read-only handle for one analysis-ready persisted dataset.

    Array handles remain lazy. ``to_numpy`` and ``to_polars`` create bounded,
    writable in-memory copies suitable for exploratory analysis; neither can
    mutate the selected source Zarr.
    """

    _workspace: "ZarrExplorationWorkspace"
    descriptor: Mapping[str, Any]

    @property
    def dataset_id(self) -> str:
        return str(self.descriptor["dataset_id"])

    @property
    def value_path(self) -> str:
        return str(self.descriptor["value_path"])

    @property
    def row_count(self) -> int:
        return int(self.descriptor["row_count"])

    def summary(self) -> dict[str, Any]:
        """Return a detached metadata description safe for notebook display."""

        return dict(self.descriptor)

    def handles(self) -> dict[str, Any]:
        """Return lazy physical array handles without reading array values."""

        paths = {
            "values": self.value_path,
            "time_s": str(self.descriptor.get("time_path") or ""),
            "frame_index": str(self.descriptor.get("frame_path") or ""),
        }
        return {
            name: self._workspace.handle(path)
            for name, path in paths.items()
            if path
        }

    def _row_selection(
        self,
        *,
        start: int,
        stop: int | None,
        stride: int,
        max_source_rows: int,
    ) -> tuple[slice, np.ndarray]:
        if start < 0 or start >= self.row_count:
            raise ValueError(
                f"start must be between 0 and {max(0, self.row_count - 1):,}."
            )
        if stride < 1 or max_source_rows < 1:
            raise ValueError("stride and max_source_rows must be positive.")
        resolved_stop = (
            min(self.row_count, int(stop))
            if stop is not None
            else min(self.row_count, int(start) + int(max_source_rows))
        )
        if resolved_stop <= start:
            raise ValueError("stop must be greater than start.")
        source_rows = resolved_stop - int(start)
        if source_rows > max_source_rows:
            raise ValueError(
                f"Selection spans {source_rows:,} source rows; the current copy "
                f"limit is {max_source_rows:,}. Use a smaller window, iterate in "
                "batches, or explicitly raise max_source_rows."
            )
        rows = np.arange(start, resolved_stop, stride, dtype=np.int64)
        return slice(start, resolved_stop, stride), rows

    def to_numpy(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        max_source_rows: int = DEFAULT_MAX_READ_ELEMENTS,
    ) -> np.ndarray:
        """Copy a bounded value slice into a writable NumPy array."""

        selection, rows = self._row_selection(
            start=int(start),
            stop=stop,
            stride=int(stride),
            max_source_rows=int(max_source_rows),
        )
        value_columns = tuple(self.descriptor.get("value_columns") or ("value",))
        element_limit = max(1, int(rows.size) * max(1, len(value_columns)))
        return np.array(
            self._workspace.read(
                self.value_path,
                selection,
                max_elements=element_limit,
            ),
            copy=True,
        )

    def to_polars(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        max_source_rows: int = DEFAULT_MAX_READ_ELEMENTS,
    ) -> pl.DataFrame:
        """Copy aligned values and coordinates into an in-memory Polars frame."""

        selection, rows = self._row_selection(
            start=int(start),
            stop=stop,
            stride=int(stride),
            max_source_rows=int(max_source_rows),
        )
        values = self.to_numpy(
            start=int(start),
            stop=selection.stop,
            stride=int(stride),
            max_source_rows=int(max_source_rows),
        )
        value_columns = tuple(self.descriptor.get("value_columns") or ("value",))
        data: dict[str, np.ndarray] = {"row_index": rows}
        if values.ndim == 1 and len(value_columns) == 1:
            data[value_columns[0]] = values
        elif values.ndim == 2 and values.shape[1] == len(value_columns):
            data.update(
                {
                    name: np.asarray(values[:, index])
                    for index, name in enumerate(value_columns)
                }
            )
        else:
            raise ValueError(
                f"Persisted shape {values.shape} does not match semantic columns "
                f"{value_columns}."
            )

        for output_name, path_key, dtype in (
            ("time_s", "time_path", np.float64),
            ("frame_index", "frame_path", np.int64),
        ):
            path = str(self.descriptor.get(path_key) or "")
            if not path:
                continue
            coordinate = np.asarray(
                self._workspace.read(
                    path,
                    selection,
                    max_elements=max(1, int(rows.size)),
                ),
                dtype=dtype,
            ).reshape(-1)
            if coordinate.shape == rows.shape:
                data[output_name] = coordinate
        ordered = [
            name
            for name in ("row_index", "time_s", "frame_index", *value_columns)
            if name in data
        ]
        return pl.DataFrame({name: data[name] for name in ordered})

    def to_lazy(self, **kwargs: Any) -> pl.LazyFrame:
        """Return a LazyFrame over a bounded in-memory Zarr projection."""

        return self.to_polars(**kwargs).lazy()

    def iter_polars(
        self,
        *,
        batch_rows: int = DEFAULT_MAX_READ_ELEMENTS,
        stride: int = 1,
    ) -> Iterator[pl.DataFrame]:
        """Yield aligned copies in bounded source-row batches."""

        if batch_rows < 1 or stride < 1:
            raise ValueError("batch_rows and stride must be positive.")
        effective_batch_rows = max(
            int(stride),
            (int(batch_rows) // int(stride)) * int(stride),
        )
        for start in range(0, self.row_count, effective_batch_rows):
            stop = min(self.row_count, start + effective_batch_rows)
            yield self.to_polars(
                start=start,
                stop=stop,
                stride=int(stride),
                max_source_rows=effective_batch_rows,
            )

    def __repr__(self) -> str:
        return (
            "ZarrAnalysisDataset("
            f"dataset_id={self.dataset_id!r}, "
            f"measurement={self.descriptor.get('measurement')!r}, "
            f"rows={self.row_count:,}, read_only=True)"
        )


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

        text_columns: dict[str, list[str]] = {}
        for field in ("units", *_CHANNEL_INDEX_TEXT_FIELDS):
            try:
                text_columns[field] = self._fixed_width_strings(
                    f"{index_path}/{field}", expected_rows=shape[1]
                )
            except (KeyError, TypeError, ValueError):
                text_columns[field] = [""] * shape[1]
        return [
            {
                "index": index,
                "name": names[index],
                **{
                    field: values[index]
                    for field, values in text_columns.items()
                },
                "available": bool(availability[index]),
            }
            for index in range(shape[1])
            if not available_only or bool(availability[index])
        ]

    def track_kinematics_datasets(
        self,
        *,
        max_runs: int = 100,
        max_tracks_per_run: int = 100,
    ) -> list[dict[str, Any]]:
        """Discover analysis-ready track datasets using metadata only.

        Preferred v2 movement arrays are selected before flat compatibility
        arrays. Each returned row is one semantic speed, position, heading, or
        angular-motion dataset aligned to its track's time/frame coordinates.
        """

        if max_runs < 1 or max_tracks_per_run < 1:
            raise ValueError("Discovery limits must be positive.")
        parent_path = "analysis/track_kinematics_runs"
        try:
            parent = self._node(parent_path)
        except (KeyError, TypeError, ValueError):
            return []
        if not _is_group(parent):
            return []

        descriptors: list[dict[str, Any]] = []
        run_count = 0
        for scope in _group_names(parent, max_items=max_runs):
            if run_count >= max_runs:
                break
            scope_path = f"{parent_path}/{scope}"
            try:
                scope_group = self._node(scope_path)
            except (KeyError, TypeError, ValueError):
                continue
            if not _is_group(scope_group):
                continue
            run_names = _group_names(
                scope_group,
                max_items=max_runs - run_count,
            )
            scope_attrs = getattr(scope_group, "attrs", {})
            latest_run = str(
                scope_attrs.get("latest_complete")
                or scope_attrs.get("latest")
                or (run_names[-1] if run_names else "")
            )
            for run_name in run_names:
                if run_count >= max_runs:
                    break
                run_path = f"{scope_path}/{run_name}"
                try:
                    run_group = self._node(run_path)
                    tracks_group = self._node(f"{run_path}/tracks")
                except (KeyError, TypeError, ValueError):
                    continue
                if not _is_group(run_group) or not _is_group(tracks_group):
                    continue
                run_count += 1
                run_attrs = getattr(run_group, "attrs", {})
                run_status = str(
                    run_attrs.get("status")
                    or run_attrs.get("completion_status")
                    or ""
                )
                run_method = str(run_attrs.get("method") or "")
                is_latest = str(run_name) == latest_run

                for track_name in _group_names(
                    tracks_group,
                    max_items=max_tracks_per_run,
                ):
                    track_path = f"{run_path}/tracks/{track_name}"
                    try:
                        track_group = self._node(track_path)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not _is_group(track_group):
                        continue
                    track_token = str(track_name).removeprefix("id_")
                    try:
                        track_id: int | str = int(track_token)
                    except ValueError:
                        track_id = track_token
                    time_path = f"{track_path}/time_seconds"
                    frame_path = f"{track_path}/frame_indices"

                    def _aligned_path(candidate: str, row_count: int) -> str:
                        try:
                            info = self.info(candidate)
                        except (KeyError, TypeError, ValueError):
                            return ""
                        return (
                            candidate
                            if info.get("kind") == "array"
                            and tuple(info.get("shape", ())) == (row_count,)
                            else ""
                        )

                    def _add_dataset(
                        *,
                        measurement: str,
                        variant: str,
                        units: str,
                        value_path: str,
                        value_columns: Sequence[str],
                    ) -> None:
                        try:
                            value_info = self.info(value_path)
                        except (KeyError, TypeError, ValueError):
                            return
                        shape = tuple(value_info.get("shape", ()))
                        expected_columns = tuple(str(name) for name in value_columns)
                        shape_matches = (
                            len(shape) == 1
                            and len(expected_columns) == 1
                            or len(shape) == 2
                            and shape[1] == len(expected_columns)
                        )
                        if value_info.get("kind") != "array" or not shape_matches:
                            return
                        row_count = int(shape[0])
                        if row_count < 1:
                            return
                        unit_token = (
                            units.replace("/", "_per_")
                            .replace(" ", "_")
                            .replace("°", "deg")
                        )
                        dataset_id = ":".join(
                            (
                                "track_kinematics",
                                str(scope),
                                str(run_name),
                                str(track_name),
                                measurement,
                                variant,
                                unit_token,
                            )
                        )
                        measurement_label = measurement.replace("_", " ").title()
                        descriptors.append(
                            {
                                "dataset_id": dataset_id,
                                "label": (
                                    f"{measurement_label} · {variant} · {units} · "
                                    f"{track_name} · {run_name}"
                                    + (" · latest" if is_latest else "")
                                ),
                                "family": "track_kinematics",
                                "measurement": measurement,
                                "variant": variant,
                                "units": units,
                                "scope": str(scope),
                                "run_name": str(run_name),
                                "run_path": run_path,
                                "run_status": run_status,
                                "run_method": run_method,
                                "is_latest": is_latest,
                                "track_id": track_id,
                                "track_path": track_path,
                                "value_path": value_path,
                                "time_path": _aligned_path(time_path, row_count),
                                "frame_path": _aligned_path(frame_path, row_count),
                                "value_columns": expected_columns,
                                "row_count": row_count,
                                "dtype": str(value_info.get("dtype") or ""),
                                "chunks": tuple(value_info.get("chunks", ())),
                                "read_only": True,
                            }
                        )

                    for level in ("smoothed", "filtered", "raw", "averaged"):
                        for unit, unit_label in (("mm", "mm/s"), ("px", "px/s")):
                            candidates = (
                                f"{track_path}/movement/speed/{level}/{unit}",
                                f"{track_path}/speed_{level}_{unit}",
                            )
                            value_path = next(
                                (
                                    candidate
                                    for candidate in candidates
                                    if self._path_is_array(candidate)
                                ),
                                "",
                            )
                            if value_path:
                                _add_dataset(
                                    measurement="speed",
                                    variant=level,
                                    units=unit_label,
                                    value_path=value_path,
                                    value_columns=(f"speed_{unit}_s",),
                                )

                    for unit in ("mm", "px"):
                        position_path = f"{track_path}/positions_{unit}"
                        if self._path_is_array(position_path):
                            _add_dataset(
                                measurement="position",
                                variant="calibrated" if unit == "mm" else "pixel",
                                units=unit,
                                value_path=position_path,
                                value_columns=(f"x_{unit}", f"y_{unit}"),
                            )

                    heading_specs = (
                        (
                            "heading",
                            "smoothed",
                            "deg",
                            "smoothed_heading_degrees",
                            "heading_deg",
                        ),
                        ("heading", "raw", "deg", "heading_degrees", "heading_deg"),
                        (
                            "angular_velocity",
                            "smoothed",
                            "deg/s",
                            "angular_velocity_smoothed_deg_s",
                            "angular_velocity_deg_s",
                        ),
                        (
                            "angular_velocity",
                            "raw",
                            "deg/s",
                            "angular_velocity_raw_deg_s",
                            "angular_velocity_deg_s",
                        ),
                        (
                            "angular_speed",
                            "smoothed",
                            "deg/s",
                            "angular_speed_smoothed_deg_s",
                            "angular_speed_deg_s",
                        ),
                        (
                            "angular_speed",
                            "raw",
                            "deg/s",
                            "angular_speed_raw_deg_s",
                            "angular_speed_deg_s",
                        ),
                    )
                    for (
                        measurement,
                        variant,
                        units,
                        array_name,
                        column_name,
                    ) in heading_specs:
                        heading_path = f"{track_path}/{array_name}"
                        if self._path_is_array(heading_path):
                            _add_dataset(
                                measurement=measurement,
                                variant=variant,
                                units=units,
                                value_path=heading_path,
                                value_columns=(column_name,),
                            )

        measurement_rank = {
            "speed": 0,
            "position": 1,
            "heading": 2,
            "angular_velocity": 3,
            "angular_speed": 4,
        }
        variant_rank = {
            "smoothed": 0,
            "filtered": 1,
            "raw": 2,
            "averaged": 3,
            "calibrated": 0,
            "pixel": 1,
        }
        return sorted(
            descriptors,
            key=lambda row: (
                not bool(row["is_latest"]),
                row["track_id"] != 0,
                measurement_rank.get(str(row["measurement"]), 99),
                variant_rank.get(str(row["variant"]), 99),
                0 if row["units"] in {"mm", "mm/s"} else 1,
                str(row["run_name"]),
            ),
        )

    def _path_is_array(self, path: str) -> bool:
        try:
            return self.info(path).get("kind") == "array"
        except (KeyError, TypeError, ValueError):
            return False

    def analysis_datasets(
        self,
        *,
        max_runs: int = 100,
        max_tracks_per_run: int = 100,
    ) -> list[dict[str, Any]]:
        """Return the semantic dataset catalog supported by this workspace."""

        return self.track_kinematics_datasets(
            max_runs=max_runs,
            max_tracks_per_run=max_tracks_per_run,
        )

    def dataset(
        self,
        descriptor_or_id: Mapping[str, Any] | str,
    ) -> ZarrAnalysisDataset:
        """Create a semantic read-only handle from a catalog row or dataset ID."""

        if isinstance(descriptor_or_id, Mapping):
            descriptor = dict(descriptor_or_id)
        else:
            dataset_id = str(descriptor_or_id)
            descriptor = next(
                (
                    row
                    for row in self.analysis_datasets()
                    if row["dataset_id"] == dataset_id
                ),
                {},
            )
            if not descriptor:
                raise KeyError(f"Unknown analysis dataset: {dataset_id!r}")
        value_path = str(descriptor.get("value_path") or "")
        if not value_path or not self._path_is_array(value_path):
            raise ValueError("Analysis dataset descriptor has no readable value array.")
        return ZarrAnalysisDataset(self, descriptor)

    def select_dataset(
        self,
        measurement: str,
        *,
        variant: str | None = None,
        units: str | None = None,
        track_id: int | str | None = 0,
        run_name: str | None = None,
    ) -> ZarrAnalysisDataset:
        """Select the preferred semantic dataset matching scientific terms."""

        candidates = [
            row
            for row in self.analysis_datasets()
            if row["measurement"] == str(measurement)
            and (variant is None or row["variant"] == str(variant))
            and (units is None or row["units"] == str(units))
            and (track_id is None or str(row["track_id"]) == str(track_id))
            and (run_name is None or row["run_name"] == str(run_name))
        ]
        if not candidates:
            requested = {
                "measurement": measurement,
                "variant": variant,
                "units": units,
                "track_id": track_id,
                "run_name": run_name,
            }
            raise KeyError(f"No analysis dataset matches {requested!r}")
        return self.dataset(candidates[0])

    def eye_angle_runs(
        self,
        *,
        max_runs: int = 100,
    ) -> list[dict[str, Any]]:
        """Discover persisted eye-angle runs without scanning the whole Zarr.

        This is a schema-aware metadata adapter for the guided workspace. It
        inspects only the direct children of ``analysis/eye_angle_runs`` and
        the metadata of each run's dense frame array; it never reads frame
        values or recursively inventories unrelated analysis families.
        """

        if max_runs < 1:
            raise ValueError("max_runs must be positive.")
        family_path = "analysis/eye_angle_runs"
        try:
            family = self._node(family_path)
        except (KeyError, TypeError, ValueError):
            return []
        if not _is_group(family):
            return []

        rows: list[dict[str, Any]] = []
        for run_name in islice(family.keys(), int(max_runs)):
            run_path = f"{family_path}/{run_name}"
            try:
                run = self._node(run_path)
            except (KeyError, TypeError, ValueError):
                continue
            if not _is_group(run):
                continue
            attrs = getattr(run, "attrs", {})
            try:
                frame_info = self.info(f"{run_path}/frame_angles")
            except (KeyError, TypeError, ValueError):
                frame_info = {}
            frame_shape = tuple(frame_info.get("shape", ()))
            rows.append(
                {
                    "run_name": str(run_name),
                    "run_path": run_path,
                    "status": str(attrs.get("status", attrs.get("completion_status", ""))),
                    "layout": str(attrs.get("layout", attrs.get("storage_layout", ""))),
                    "method": str(attrs.get("method", "")),
                    "schema_version": str(
                        attrs.get("schema_version", attrs.get("version", ""))
                    ),
                    "frame_count": int(frame_shape[0]) if len(frame_shape) == 2 else 0,
                    "frame_channel_count": (
                        int(frame_shape[1]) if len(frame_shape) == 2 else 0
                    ),
                    "frame_angles_path": (
                        f"{run_path}/frame_angles" if len(frame_shape) == 2 else ""
                    ),
                }
            )
        return sorted(rows, key=lambda row: str(row["run_name"]))

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

    def coordinate_summary(self, array_path: str) -> dict[str, Any] | None:
        """Summarize a conventional time coordinate with at most three scalars."""

        coordinate_path = self.suggested_coordinate_path(array_path)
        if coordinate_path is None:
            return None
        info = self.info(coordinate_path)
        shape = tuple(info.get("shape", ()))
        if len(shape) != 1 or shape[0] < 1:
            return None

        def _scalar(index: int) -> float:
            value = self.read(coordinate_path, index, max_elements=1)
            return float(np.asarray(value).item())

        start = _scalar(0)
        stop = _scalar(shape[0] - 1)
        interval = _scalar(1) - start if shape[0] > 1 else float("nan")
        sample_rate = 1.0 / interval if np.isfinite(interval) and interval > 0 else None
        return {
            "path": coordinate_path,
            "row_count": int(shape[0]),
            "start_seconds": start,
            "stop_seconds": stop,
            "sample_interval_seconds": interval if np.isfinite(interval) else None,
            "sample_rate_hz": sample_rate,
        }

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
