"""Explicit frame-domain resolver for Palette recording stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping

import numpy as np
import numpy.typing as npt

from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import zarr_attrs_dict, zarr_child_group


FRAME_DOMAIN_MAPS_GROUP = "frame_domain_maps"
STORED_ZARR_TO_ACQUISITION_MAP = "stored_zarr_frame_to_acquisition_frame"
RUN_FRAME_TO_ACQUISITION_MAP = "run_frame_to_acquisition_frame"
_SPARSE_DENSE_LOOKUP_MAX_BYTES = 256 * 1024 * 1024


class FrameDomain(str, Enum):
    """Named frame domains for explicit conversion."""

    ACQUISITION = "acquisition_frame"
    STORED_ZARR = "stored_zarr_frame"
    SOURCE_VIDEO = "source_video_frame"
    RUN_FRAME = "run_frame"
    CROP_VIDEO = "crop_video_frame"


@dataclass(frozen=True)
class FrameDomainEdge:
    """Public description of an available conversion edge."""

    source: FrameDomain
    target: FrameDomain
    source_path: str
    mapping_arrays: tuple[str, ...]
    run_name: str | None = None
    parent_path: str | None = None
    confidence: Literal["explicit", "stage_contract"] = "explicit"


@dataclass(frozen=True)
class FrameDomainCapabilities:
    """Summary of available frame-domain counts and conversion edges."""

    domains: frozenset[FrameDomain]
    edges: tuple[FrameDomainEdge, ...]
    counts: Mapping[FrameDomain, int]
    missing: tuple[str, ...]


class FrameDomainError(ValueError):
    """Base exception for frame-domain resolution failures."""


class FrameDomainUnmappedError(FrameDomainError):
    """Raised when a requested frame-domain conversion is not explicitly available."""

    def __init__(
        self,
        message: str,
        *,
        source: FrameDomain,
        target: FrameDomain,
        stage: str | None = None,
        run_name: str | None = None,
        mapping_arrays: tuple[str, ...] = (),
        values: tuple[int, ...] = (),
    ) -> None:
        self.source = source
        self.target = target
        self.stage = stage
        self.run_name = run_name
        self.mapping_arrays = mapping_arrays
        self.values = values
        super().__init__(message)


@dataclass(frozen=True)
class _ConversionEdge:
    public: FrameDomainEdge
    mapping: Mapping[int, int]
    _lookup: "_DenseLookup | _SortedLookup | None" = field(
        default=None,
        init=False,
        repr=False,
    )

    def lookup(self) -> "_DenseLookup | _SortedLookup":
        lookup = self._lookup
        if lookup is None:
            lookup = _build_lookup(self.mapping)
            object.__setattr__(self, "_lookup", lookup)
        return lookup


@dataclass(frozen=True)
class _DenseLookup:
    values: np.ndarray
    valid: np.ndarray | None = None


@dataclass(frozen=True)
class _SortedLookup:
    keys: np.ndarray
    values: np.ndarray


def _build_lookup(mapping: Mapping[int, int]) -> _DenseLookup | _SortedLookup:
    if not mapping:
        return _DenseLookup(values=np.asarray([], dtype=np.int64))

    keys = np.fromiter(
        (int(key) for key in mapping.keys()),
        dtype=np.int64,
        count=len(mapping),
    )
    values = np.fromiter(
        (int(value) for value in mapping.values()),
        dtype=np.int64,
        count=len(mapping),
    )
    if keys.size and int(keys.min()) == 0 and int(keys.max()) == keys.size - 1:
        dense = np.empty(keys.size, dtype=np.int64)
        dense[keys] = values
        return _DenseLookup(values=dense)
    if keys.size and int(keys.min()) >= 0:
        max_key = int(keys.max())
        dense_size = max_key + 1
        estimated_bytes = dense_size * (
            np.dtype(np.int64).itemsize + np.dtype(bool).itemsize
        )
        if max_key <= keys.size * 4 and estimated_bytes <= _SPARSE_DENSE_LOOKUP_MAX_BYTES:
            dense = np.empty(dense_size, dtype=np.int64)
            valid = np.zeros(dense_size, dtype=bool)
            dense[keys] = values
            valid[keys] = True
            return _DenseLookup(values=dense, valid=valid)

    order = np.argsort(keys, kind="stable")
    return _SortedLookup(keys=keys[order], values=values[order])


def _normalize_domain(domain: FrameDomain | str) -> FrameDomain:
    return domain if isinstance(domain, FrameDomain) else FrameDomain(str(domain))


def _array(group: Any | None, name: str) -> Any | None:
    if group is None:
        return None
    try:
        value = group[name]
    except Exception:
        return None
    return value if hasattr(value, "shape") and not hasattr(value, "group_keys") else None


def _array_values(group: Any | None, name: str) -> np.ndarray | None:
    value = _array(group, name)
    if value is None:
        return None
    return np.asarray(value[:])


def _array_path(prefix: str, name: str) -> str:
    prefix_clean = str(prefix).strip("/")
    return f"{prefix_clean}/{name}" if prefix_clean else name


def _first_positive_int(attrs: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = attrs.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _first_frame_array_shape(raw_group: Any | None) -> int | None:
    for name in ("images_full", "images_ds", "images_ds_gray", "images_ds_rgb"):
        value = _array(raw_group, name)
        if value is not None and getattr(value, "shape", None):
            return int(value.shape[0])
    return None


def _mapping_from_pairs(source_values: np.ndarray, target_values: np.ndarray) -> dict[int, int]:
    source_flat = np.asarray(source_values, dtype=np.int64).reshape(-1)
    target_flat = np.asarray(target_values, dtype=np.int64).reshape(-1)
    if source_flat.shape[0] != target_flat.shape[0]:
        raise FrameDomainError(
            "frame-domain mapping arrays must have the same length: "
            f"{source_flat.shape[0]} != {target_flat.shape[0]}"
        )
    mapping: dict[int, int] = {}
    for source, target in zip(source_flat.tolist(), target_flat.tolist(), strict=True):
        source_i = int(source)
        target_i = int(target)
        existing = mapping.get(source_i)
        if existing is not None and existing != target_i:
            raise FrameDomainError(
                f"conflicting frame-domain mapping for {source_i}: {existing} vs {target_i}"
            )
        mapping[source_i] = target_i
    return mapping


def _inverse_mapping(mapping: Mapping[int, int]) -> dict[int, int] | None:
    inverse: dict[int, int] = {}
    for source, target in mapping.items():
        existing = inverse.get(target)
        if existing is not None and existing != source:
            return None
        inverse[target] = source
    return inverse


class FrameDomains:
    """Read-only resolver for explicitly recorded frame-domain mappings."""

    def __init__(
        self,
        *,
        root: Any,
        stage: str | None = None,
        run_group: Any | None = None,
        parent_path: str | None = None,
        run_name: str | None = None,
        resolution: Any | None = None,
    ) -> None:
        self.root = root
        self.stage = stage
        self.run_group = run_group
        self.parent_path = parent_path
        self.run_name = run_name
        self.resolution = resolution
        self._counts: dict[FrameDomain, int] = {}
        self._edges: dict[tuple[FrameDomain, FrameDomain], _ConversionEdge] = {}
        self._missing: list[str] = []
        self._build()

    def capabilities(self) -> FrameDomainCapabilities:
        """Return available domains, edges, counts, and missing explicit mappings."""

        domains = set(self._counts)
        for source, target in self._edges:
            domains.add(source)
            domains.add(target)
        public_edges = tuple(edge.public for edge in self._edges.values())
        return FrameDomainCapabilities(
            domains=frozenset(domains),
            edges=public_edges,
            counts=dict(self._counts),
            missing=tuple(self._missing),
        )

    def count(
        self,
        domain: FrameDomain | str,
        *,
        stage: str | None = None,
        run: str | None = None,
    ) -> int:
        """Return the explicit count for a frame domain."""

        self._assert_scope(stage=stage, run=run)
        normalized = _normalize_domain(domain)
        try:
            return self._counts[normalized]
        except KeyError as exc:
            raise FrameDomainUnmappedError(
                f"no explicit count is available for {normalized.value}",
                source=normalized,
                target=normalized,
                stage=self.stage,
                run_name=self.run_name,
            ) from exc

    def convert(
        self,
        values: npt.ArrayLike,
        source: FrameDomain | str,
        target: FrameDomain | str,
        *,
        stage: str | None = None,
        run: str | None = None,
    ) -> np.ndarray:
        """Convert frame indices between explicit frame domains."""

        self._assert_scope(stage=stage, run=run)
        source_domain = _normalize_domain(source)
        target_domain = _normalize_domain(target)
        input_values = np.asarray(values)
        flat_values = input_values.reshape(-1)
        if source_domain is target_domain:
            return np.asarray(input_values, dtype=np.int64).copy()

        edge = self._edges.get((source_domain, target_domain))
        if edge is None:
            raise FrameDomainUnmappedError(
                f"no explicit frame-domain edge from {source_domain.value} to {target_domain.value}",
                source=source_domain,
                target=target_domain,
                stage=self.stage,
                run_name=self.run_name,
            )

        converted = np.asarray(flat_values, dtype=np.int64)
        output = np.empty(converted.shape[0], dtype=np.int64)
        lookup = edge.lookup()
        if isinstance(lookup, _DenseLookup):
            in_bounds = (converted >= 0) & (converted < lookup.values.shape[0])
            if lookup.valid is None:
                hit_mask = in_bounds
            else:
                hit_mask = np.zeros(converted.shape[0], dtype=bool)
                in_bounds_values = converted[in_bounds]
                hit_mask[in_bounds] = lookup.valid[in_bounds_values]
            output[hit_mask] = lookup.values[converted[hit_mask]]
        else:
            if lookup.keys.shape[0] == 0:
                hit_mask = np.zeros(converted.shape[0], dtype=bool)
            else:
                positions = np.searchsorted(lookup.keys, converted)
                in_bounds = positions < lookup.keys.shape[0]
                safe_positions = np.minimum(positions, lookup.keys.shape[0] - 1)
                hit_mask = in_bounds & (lookup.keys[safe_positions] == converted)
                output[hit_mask] = lookup.values[safe_positions[hit_mask]]
        if not np.all(hit_mask):
            missing = converted[~hit_mask]
            sample = tuple(int(value) for value in missing[:10].tolist())
            raise FrameDomainUnmappedError(
                f"{missing.shape[0]} value(s) are unmappable from {source_domain.value} "
                f"to {target_domain.value}; first values: {sample}",
                source=source_domain,
                target=target_domain,
                stage=self.stage,
                run_name=self.run_name,
                mapping_arrays=edge.public.mapping_arrays,
                values=sample,
            )
        return output.reshape(input_values.shape)

    def _assert_scope(self, *, stage: str | None, run: str | None) -> None:
        if stage is not None and self.stage is not None and str(stage) != str(self.stage):
            raise ValueError(f"FrameDomains was constructed for stage {self.stage!r}, not {stage!r}")
        if run is not None and self.run_name is not None and str(run) != str(self.run_name):
            raise ValueError(f"FrameDomains was constructed for run {self.run_name!r}, not {run!r}")

    def _build(self) -> None:
        self._build_raw_edges()
        if self.run_group is not None:
            self._build_run_edges()

    def _add_count(self, domain: FrameDomain, count: int | None) -> None:
        if count is None:
            return
        count_i = int(count)
        if count_i < 0:
            return
        existing = self._counts.get(domain)
        if existing is None:
            self._counts[domain] = count_i
        elif existing != count_i:
            self._missing.append(
                f"conflicting explicit count for {domain.value}: {existing} vs {count_i}"
            )

    def _add_edge(
        self,
        *,
        source: FrameDomain,
        target: FrameDomain,
        mapping: Mapping[int, int],
        source_path: str,
        mapping_arrays: tuple[str, ...],
        confidence: Literal["explicit", "stage_contract"] = "explicit",
    ) -> None:
        key = (source, target)
        if key in self._edges:
            self._missing.append(f"duplicate edge ignored: {source.value} -> {target.value}")
            return
        public = FrameDomainEdge(
            source=source,
            target=target,
            source_path=source_path,
            mapping_arrays=mapping_arrays,
            run_name=self.run_name,
            parent_path=self.parent_path,
            confidence=confidence,
        )
        self._edges[key] = _ConversionEdge(public=public, mapping=dict(mapping))

    def _add_bidirectional_edge(
        self,
        *,
        source: FrameDomain,
        target: FrameDomain,
        source_values: np.ndarray,
        target_values: np.ndarray,
        source_path: str,
        mapping_arrays: tuple[str, ...],
        confidence: Literal["explicit", "stage_contract"] = "explicit",
    ) -> None:
        try:
            mapping = _mapping_from_pairs(source_values, target_values)
        except FrameDomainError as exc:
            self._missing.append(f"{source.value}->{target.value}: {exc}")
            return
        self._add_edge(
            source=source,
            target=target,
            mapping=mapping,
            source_path=source_path,
            mapping_arrays=mapping_arrays,
            confidence=confidence,
        )
        inverse = _inverse_mapping(mapping)
        if inverse is None:
            self._missing.append(
                f"inverse edge unavailable because mapping is many-to-one: {target.value}->{source.value}"
            )
            return
        self._add_edge(
            source=target,
            target=source,
            mapping=inverse,
            source_path=source_path,
            mapping_arrays=mapping_arrays,
            confidence=confidence,
        )

    def _build_raw_edges(self) -> None:
        raw = zarr_child_group(self.root, "raw_video")
        if raw is None:
            self._missing.append("raw_video group missing; no stored-zarr frame mapping available")
            return
        raw_attrs = zarr_attrs_dict(raw)
        root_attrs = zarr_attrs_dict(self.root)

        stored_count = _first_frame_array_shape(raw)
        maps_group = zarr_child_group(raw, FRAME_DOMAIN_MAPS_GROUP)
        explicit_map = _array_values(maps_group, STORED_ZARR_TO_ACQUISITION_MAP)
        if explicit_map is not None:
            stored_count = int(explicit_map.shape[0])
            stored_values = np.arange(explicit_map.shape[0], dtype=np.int64)
            mapping_path = _array_path(
                f"raw_video/{FRAME_DOMAIN_MAPS_GROUP}",
                STORED_ZARR_TO_ACQUISITION_MAP,
            )
            self._add_bidirectional_edge(
                source=FrameDomain.STORED_ZARR,
                target=FrameDomain.ACQUISITION,
                source_values=stored_values,
                target_values=explicit_map.astype(np.int64, copy=False),
                source_path=mapping_path,
                mapping_arrays=(mapping_path,),
            )

        original = _array_values(raw, "original_frame_indices")
        if original is not None:
            stored_count = int(original.shape[0])
            stored_values = np.arange(original.shape[0], dtype=np.int64)
            mapping_path = "raw_video/original_frame_indices"
            self._add_bidirectional_edge(
                source=FrameDomain.STORED_ZARR,
                target=FrameDomain.SOURCE_VIDEO,
                source_values=stored_values,
                target_values=original.astype(np.int64, copy=False),
                source_path=mapping_path,
                mapping_arrays=(mapping_path,),
            )
            if (FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION) not in self._edges:
                self._add_bidirectional_edge(
                    source=FrameDomain.STORED_ZARR,
                    target=FrameDomain.ACQUISITION,
                    source_values=stored_values,
                    target_values=original.astype(np.int64, copy=False),
                    source_path=mapping_path,
                    mapping_arrays=(mapping_path,),
                    confidence="stage_contract",
                )

        self._add_count(FrameDomain.STORED_ZARR, stored_count)
        acquisition_count = _first_positive_int(
            raw_attrs,
            ("original_video_length", "source_video_total_frames", "total_frames"),
        )
        if acquisition_count is None:
            acquisition_count = _first_positive_int(
                root_attrs,
                ("source_video_total_frames", "original_video_length", "total_frames", "n_frames"),
            )
        self._add_count(FrameDomain.ACQUISITION, acquisition_count)
        self._add_count(FrameDomain.SOURCE_VIDEO, acquisition_count)

        if explicit_map is None and original is None:
            self._missing.append(
                "raw_video has no explicit stored_zarr_frame -> acquisition_frame mapping"
            )

    def _build_run_edges(self) -> None:
        frame_counts = _array(self.run_group, "frame_counts")
        if frame_counts is not None:
            self._add_count(FrameDomain.RUN_FRAME, int(frame_counts.shape[0]))

        run_maps = zarr_child_group(self.run_group, FRAME_DOMAIN_MAPS_GROUP)
        run_map = _array_values(run_maps, RUN_FRAME_TO_ACQUISITION_MAP)
        if run_map is not None:
            run_values = np.arange(run_map.shape[0], dtype=np.int64)
            mapping_path = _array_path(
                f"{self.parent_path or ''}/{self.run_name or ''}/{FRAME_DOMAIN_MAPS_GROUP}",
                RUN_FRAME_TO_ACQUISITION_MAP,
            )
            self._add_bidirectional_edge(
                source=FrameDomain.RUN_FRAME,
                target=FrameDomain.ACQUISITION,
                source_values=run_values,
                target_values=run_map.astype(np.int64, copy=False),
                source_path=mapping_path,
                mapping_arrays=(mapping_path,),
            )

        crop_video = _array_values(self.run_group, "source_crop_video_frame_indices")
        if crop_video is not None:
            self._build_crop_video_edges(crop_video)
        elif self.stage == "crop":
            self._missing.append(
                "crop run has no source_crop_video_frame_indices; crop_video_frame conversions unavailable"
            )

    def _build_crop_video_edges(self, crop_video: np.ndarray) -> None:
        run_attrs = zarr_attrs_dict(self.run_group)
        crop_count = _first_positive_int(
            run_attrs,
            ("source_crop_video_frame_count", "crop_video_frame_count"),
        )
        self._add_count(FrameDomain.CROP_VIDEO, crop_count)

        source_frames = _array_values(self.run_group, "source_frame_indices")
        frame_source_name = "source_frame_indices"
        if source_frames is None:
            # No legacy adapters in this slice. ``frame_indices`` is acceptable only
            # when a current writer explicitly says it is already acquisition-domain.
            frame_domain = normalize_attr(run_attrs.get("frame_indices_domain"))
            if frame_domain == FrameDomain.ACQUISITION.value:
                source_frames = _array_values(self.run_group, "frame_indices")
                frame_source_name = "frame_indices"
        if source_frames is None:
            self._missing.append(
                "source_crop_video_frame_indices present but no explicit acquisition-frame row mapping"
            )
            return
        crop_flat = crop_video.astype(np.int64, copy=False).reshape(-1)
        source_flat = source_frames.astype(np.int64, copy=False).reshape(-1)
        if crop_flat.shape[0] != source_flat.shape[0]:
            self._missing.append(
                "source_crop_video_frame_indices and acquisition-frame row mapping lengths differ"
            )
            return
        valid = crop_flat >= 0
        if not np.any(valid):
            self._missing.append("all crop-video frame indices are unmappable negative values")
            return
        mapping_path = _array_path(
            f"{self.parent_path or ''}/{self.run_name or ''}",
            "source_crop_video_frame_indices",
        )
        source_path = _array_path(f"{self.parent_path or ''}/{self.run_name or ''}", frame_source_name)
        self._add_bidirectional_edge(
            source=FrameDomain.CROP_VIDEO,
            target=FrameDomain.ACQUISITION,
            source_values=crop_flat[valid],
            target_values=source_flat[valid],
            source_path=mapping_path,
            mapping_arrays=(mapping_path, source_path),
        )
