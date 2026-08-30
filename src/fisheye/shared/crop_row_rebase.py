"""Shared fail-closed row rebasing between supported crop publication profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.hybrid_crop_provider import (
    HYBRID_CROP_RUN_SCHEMA_ID,
    validate_hybrid_crop_signed_identity,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)


CROP_REBASE_REQUIRED_AUTHORITY_ARRAYS: tuple[str, ...] = (
    "instance_key",
    "source_refined_row_ids",
    "frame_indices",
    "roi_coordinates_full",
)

CROP_REBASE_CORROBORATION_ARRAYS: tuple[str, ...] = (
    "source_acquisition_frame_index",
    "roi_sizes_full",
    "source_crop_xywh",
    "bbox_img_xyxy",
    "bbox_norm_coords",
    "bbox_roi_xyxy",
)

DIRECT_SAME_CROP_MAPPING_MODE = "direct_same_crop_row_ids"
IDENTITY_REBASE_MAPPING_MODE = "identity_rebase"


@dataclass(frozen=True)
class CropRowSelection:
    """Rows selected from one exact source crop publication."""

    source_label: str
    source_crop_run: str
    source_rows: np.ndarray

    def __post_init__(self) -> None:
        label = str(self.source_label).strip()
        run = str(self.source_crop_run).strip()
        rows = np.asarray(self.source_rows, dtype=np.int64).reshape(-1)
        if not label:
            raise ValueError("Crop-row selection source_label cannot be empty.")
        if not run:
            raise ValueError("Crop-row selection source_crop_run cannot be empty.")
        object.__setattr__(self, "source_label", label)
        object.__setattr__(self, "source_crop_run", run)
        object.__setattr__(self, "source_rows", rows)


@dataclass(frozen=True)
class CropRowRebaseResolution:
    """One exact mapping into the target crop row domain."""

    target_crop_run: str
    target_crop_group: zarr.Group
    target_rows: np.ndarray
    source_crop_runs: tuple[str, ...]
    mapping_mode: str
    authority_arrays: tuple[str, ...]


def _array(group: zarr.Group, name: str) -> Any:
    node = group.get(name)
    if node is None:
        raise ValueError(f"{group.path or '<root>'} is missing array {name!r}.")
    return node


def _selected_rows(array: Any, rows: np.ndarray) -> np.ndarray:
    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    total = int(array.shape[0])
    if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= total):
        raise ValueError("Crop-row selection is outside its source row domain.")
    if not selected.size:
        return np.empty(
            (0, *tuple(int(value) for value in array.shape[1:])), dtype=array.dtype
        )
    start = int(selected[0])
    if np.array_equal(
        selected,
        np.arange(start, start + int(selected.size), dtype=np.int64),
    ):
        return np.asarray(array[start : start + int(selected.size)])
    oindex = getattr(array, "oindex", None)
    if oindex is not None:
        return np.asarray(oindex[selected.tolist()])
    return np.asarray(array[:])[selected]


def _validated_instance_keys(
    group: zarr.Group,
    *,
    label: str,
) -> np.ndarray:
    node = _array(group, "instance_key")
    keys = np.asarray(node[:])
    expected_rows = int(_array(group, "frame_indices").shape[0])
    if keys.dtype != np.dtype(np.uint64) or keys.shape != (expected_rows,):
        raise ValueError(
            f"{label} instance_key must have exact uint64 shape ({expected_rows},)."
        )
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise ValueError(f"{label} instance_key values are not unique.")
    return keys


def _validate_profile_authorities(
    *,
    archive: Path | None,
    target_crop_run: str,
    target_group: zarr.Group,
    source_groups: Mapping[str, zarr.Group],
) -> None:
    target_manifest = target_group.attrs.get("run_manifest")
    target_refined_run: str | None = None
    if isinstance(target_manifest, Mapping):
        if archive is None:
            raise ValueError(
                "A manifest-bound crop rebase requires its archive path for full "
                "publication validation."
            )
        publication = open_persisted_crop_geometry_publication(
            archive,
            run_id=target_crop_run,
        )
        payload = publication.manifest.get("payload")
        refined = (
            payload.get("source_refined_snapshot")
            if isinstance(payload, Mapping)
            else None
        )
        if isinstance(refined, Mapping):
            target_refined_run = str(refined.get("run_id") or "").strip() or None

    for source_run, source_group in source_groups.items():
        if str(source_group.attrs.get("schema_id") or "") != HYBRID_CROP_RUN_SCHEMA_ID:
            continue
        provider_digest = str(
            source_group.attrs.get("provider_record_sha256") or ""
        ).strip()
        validate_hybrid_crop_signed_identity(
            source_group,
            expected_provider_record_sha256=provider_digest,
        )
        source_refined_run = str(
            source_group.attrs.get("source_refined_detect_run") or ""
        ).strip()
        if target_refined_run is not None and source_refined_run != target_refined_run:
            raise ValueError(
                f"Signed hybrid crop crop_runs/{source_run} binds refined run "
                f"{source_refined_run!r}, but crop_runs/{target_crop_run} binds "
                f"{target_refined_run!r}."
            )


def _target_rows_for_instance_keys(
    *,
    sorted_target_keys: np.ndarray,
    target_key_order: np.ndarray,
    source_keys: np.ndarray,
    source_label: str,
    target_crop_run: str,
) -> np.ndarray:
    positions = np.searchsorted(sorted_target_keys, source_keys)
    in_range = positions < sorted_target_keys.shape[0]
    matches = np.zeros(source_keys.shape[0], dtype=bool)
    matches[in_range] = sorted_target_keys[positions[in_range]] == source_keys[in_range]
    if not np.all(matches):
        first = int(np.flatnonzero(~matches)[0])
        raise ValueError(
            f"Could not map {source_label} row {first} by instance_key into "
            f"crop_runs/{target_crop_run}."
        )
    return target_key_order[positions].astype(np.int64, copy=False)


def resolve_crop_row_rebase(
    *,
    crop_parent: zarr.Group,
    target_crop_run: str,
    selections: Sequence[CropRowSelection],
    archive: Path | None = None,
) -> CropRowRebaseResolution:
    """Resolve source crop rows into one exact target crop row domain.

    Cross-profile rebases use ``instance_key`` for indexed lookup, then require
    equality of every shared authority and corroboration array.  A sealed
    geometry target and a signed-hybrid source are each validated through their
    native full-strength profile validator before their rows can be related.
    """

    target_run = str(target_crop_run).strip()
    if not target_run or target_run not in crop_parent:
        raise ValueError(f"target crop run not found: crop_runs/{target_run}")
    resolved = tuple(selections)
    if not resolved:
        raise ValueError("Crop-row rebase requires at least one source selection.")
    target_group = crop_parent[target_run]
    source_runs = tuple(item.source_crop_run for item in resolved)

    if all(source_run == target_run for source_run in source_runs):
        target_total = int(_array(target_group, "frame_indices").shape[0])
        chunks: list[np.ndarray] = []
        for item in resolved:
            rows = np.asarray(item.source_rows, dtype=np.int64).reshape(-1)
            if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= target_total):
                raise ValueError(
                    f"{item.source_label} has source_crop_row_ids outside "
                    f"crop_runs/{target_run}."
                )
            chunks.append(rows)
        target_rows = (
            np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.int64)
        )
        return CropRowRebaseResolution(
            target_crop_run=target_run,
            target_crop_group=target_group,
            target_rows=target_rows,
            source_crop_runs=source_runs,
            mapping_mode=DIRECT_SAME_CROP_MAPPING_MODE,
            authority_arrays=(),
        )

    missing_target = [
        name
        for name in CROP_REBASE_REQUIRED_AUTHORITY_ARRAYS
        if name not in target_group
    ]
    if missing_target:
        raise ValueError(
            f"target crop run crop_runs/{target_run} missing authority arrays: "
            f"{missing_target}"
        )
    source_groups: dict[str, zarr.Group] = {}
    for item in resolved:
        source_group = crop_parent.get(item.source_crop_run)
        if source_group is None:
            raise ValueError(
                f"source crop run not found: crop_runs/{item.source_crop_run}"
            )
        source_groups[item.source_crop_run] = source_group
    _validate_profile_authorities(
        archive=archive,
        target_crop_run=target_run,
        target_group=target_group,
        source_groups=source_groups,
    )

    target_keys = _validated_instance_keys(
        target_group,
        label=f"crop_runs/{target_run}",
    )
    target_key_order = np.argsort(target_keys, kind="stable")
    sorted_target_keys = target_keys[target_key_order]

    source_authority_names: dict[str, tuple[str, ...]] = {}
    source_instance_keys: dict[str, np.ndarray] = {}
    for source_run, source_group in source_groups.items():
        missing_source = [
            name
            for name in CROP_REBASE_REQUIRED_AUTHORITY_ARRAYS
            if name not in source_group
        ]
        if missing_source:
            raise ValueError(
                f"source crop run crop_runs/{source_run} missing authority arrays: "
                f"{missing_source}"
            )
        source_instance_keys[source_run] = _validated_instance_keys(
            source_group,
            label=f"crop_runs/{source_run}",
        )
        source_authority_names[source_run] = tuple(
            name
            for name in (
                *CROP_REBASE_REQUIRED_AUTHORITY_ARRAYS,
                *CROP_REBASE_CORROBORATION_ARRAYS,
            )
            if name in source_group and name in target_group
        )

    mapped_chunks: list[np.ndarray] = []
    compared_names: set[str] = set()
    for item in resolved:
        source_group = source_groups[item.source_crop_run]
        source_rows = np.asarray(item.source_rows, dtype=np.int64).reshape(-1)
        source_keys = _selected_rows(
            source_instance_keys[item.source_crop_run],
            source_rows,
        )
        if (
            source_keys.dtype != np.dtype(np.uint64)
            or source_keys.shape != source_rows.shape
        ):
            raise ValueError(
                f"{item.source_label} source instance_key has the wrong dtype or shape."
            )
        mapped = _target_rows_for_instance_keys(
            sorted_target_keys=sorted_target_keys,
            target_key_order=target_key_order,
            source_keys=source_keys,
            source_label=item.source_label,
            target_crop_run=target_run,
        )
        for name in source_authority_names[item.source_crop_run]:
            if name == "instance_key":
                continue
            source_values = _selected_rows(_array(source_group, name), source_rows)
            target_values = _selected_rows(_array(target_group, name), mapped)
            if source_values.shape != target_values.shape or not np.array_equal(
                source_values,
                target_values,
            ):
                raise ValueError(
                    f"{item.source_label} authority array {name!r} disagrees with "
                    f"crop_runs/{target_run} after instance_key mapping."
                )
            compared_names.add(name)
        mapped_chunks.append(mapped)

    target_rows = (
        np.concatenate(mapped_chunks, axis=0)
        if mapped_chunks
        else np.zeros(0, dtype=np.int64)
    )
    if np.unique(target_rows).shape[0] != target_rows.shape[0]:
        raise ValueError("Crop-row rebase maps multiple source rows to one target row.")
    return CropRowRebaseResolution(
        target_crop_run=target_run,
        target_crop_group=target_group,
        target_rows=target_rows,
        source_crop_runs=source_runs,
        mapping_mode=IDENTITY_REBASE_MAPPING_MODE,
        authority_arrays=tuple(sorted(compared_names | {"instance_key"})),
    )


__all__ = [
    "CROP_REBASE_CORROBORATION_ARRAYS",
    "CROP_REBASE_REQUIRED_AUTHORITY_ARRAYS",
    "DIRECT_SAME_CROP_MAPPING_MODE",
    "IDENTITY_REBASE_MAPPING_MODE",
    "CropRowRebaseResolution",
    "CropRowSelection",
    "resolve_crop_row_rebase",
]
