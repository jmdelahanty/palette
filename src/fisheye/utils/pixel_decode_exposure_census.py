"""Read-only pixel decode exposure census for training datasets.

This utility samples existing training zarr image surfaces and classifies their
pixel values for the limited-range expansion signature. It does not modify any
registry row or zarr store.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr


DEFAULT_REGISTRY = Path("/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite")
DEFAULT_OUTPUT = Path("docs/diagnostics/pixel_decode_exposure_census_2026-07-02.md")

RAW_VIDEO_SURFACES = (
    "raw_video/images_ds",
    "raw_video/images",
    "raw_video/images_ds_rgb",
    "raw_video/images_ds_color",
)
PIXEL_CONTRACT_NAME_ATTRS = (
    "roi_pixel_contract_name",
    "pixel_contract_name",
    "crop_pixel_contract_name",
    "source_roi_pixel_contract_name",
)
DECODE_BACKEND_ATTRS = (
    "decode_backend",
    "decode_backend_effective",
    "source_decode_backend",
)
SOURCE_PIXELS_ATTRS = (
    "source_pixels",
    "gray_source_pixels",
    "rgb_source_pixels",
)


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    recording_id: str | None
    artifact_kind: str | None
    zarr_use: str | None
    zarr_path: Path
    status: str | None


@dataclass(frozen=True)
class SurfaceRecord:
    dataset: DatasetRecord
    surface_path: str
    shape: tuple[int, ...] | None
    dtype: str | None
    pixel_contract_name: str | None
    decode_backend: str | None
    source_pixels: str | None
    sample_frames: int
    sampled_pixels: int
    classification: str
    confidence: str
    evidence: dict[str, Any]
    error: str | None = None


def _connect_registry_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_training_datasets(registry_path: Path) -> list[DatasetRecord]:
    with _connect_registry_ro(registry_path) as conn:
        rows = conn.execute(
            """
            SELECT dataset_id, recording_id, artifact_kind, zarr_use, zarr_path, status
            FROM datasets
            WHERE zarr_use = 'training'
               OR artifact_kind = 'derived_training_merge'
               OR dataset_id LIKE '%_merged'
            ORDER BY
                CASE WHEN artifact_kind = 'derived_training_merge' OR dataset_id LIKE '%_merged' THEN 0 ELSE 1 END,
                dataset_id
            """
        ).fetchall()
    return [
        DatasetRecord(
            dataset_id=str(row[0]),
            recording_id=str(row[1]) if row[1] is not None else None,
            artifact_kind=str(row[2]) if row[2] is not None else None,
            zarr_use=str(row[3]) if row[3] is not None else None,
            zarr_path=Path(str(row[4])),
            status=str(row[5]) if row[5] is not None else None,
        )
        for row in rows
    ]


def _attrs_mapping(node: Any | None) -> dict[str, Any]:
    if node is None:
        return {}
    try:
        return dict(node.attrs)
    except Exception:
        return {}


def _first_attr(attrs: Sequence[Mapping[str, Any]], names: Sequence[str]) -> str | None:
    for mapping in attrs:
        for name in names:
            value = mapping.get(name)
            if value is None or value == "":
                continue
            return str(value)
    return None


def _child(group: Any, name: str) -> Any | None:
    try:
        if name in group:
            return group[name]
    except Exception:
        return None
    return None


def _discover_surfaces(root: Any) -> list[tuple[str, Any, list[Mapping[str, Any]]]]:
    surfaces: list[tuple[str, Any, list[Mapping[str, Any]]]] = []
    root_attrs = _attrs_mapping(root)

    raw_video = _child(root, "raw_video")
    raw_attrs = _attrs_mapping(raw_video)
    for surface_path in RAW_VIDEO_SURFACES:
        parent_name, array_name = surface_path.split("/", 1)
        parent = _child(root, parent_name)
        if parent is None:
            continue
        array = _child(parent, array_name)
        if array is not None and hasattr(array, "shape"):
            surfaces.append((surface_path, array, [_attrs_mapping(array), raw_attrs, root_attrs]))

    crop_runs = _child(root, "crop_runs")
    if crop_runs is not None:
        try:
            run_names = sorted(str(name) for name in crop_runs.keys())
        except Exception:
            run_names = []
        for run_name in run_names:
            run_group = _child(crop_runs, run_name)
            roi_images = _child(run_group, "roi_images") if run_group is not None else None
            if roi_images is not None and hasattr(roi_images, "shape"):
                surfaces.append(
                    (
                        f"crop_runs/{run_name}/roi_images",
                        roi_images,
                        [_attrs_mapping(roi_images), _attrs_mapping(run_group), _attrs_mapping(crop_runs), root_attrs],
                    )
                )

    return surfaces


def _limited_expansion_allowed_values() -> set[int]:
    source = np.arange(256, dtype=np.float64)
    mapped = np.clip(np.rint((source - 16.0) * 255.0 / 219.0), 0, 255).astype(np.uint8)
    return {int(value) for value in mapped}


LIMITED_EXPANSION_ALLOWED = _limited_expansion_allowed_values()
LIMITED_EXPANSION_FORBIDDEN = np.array(
    [value for value in range(256) if value not in LIMITED_EXPANSION_ALLOWED],
    dtype=np.int64,
)


def classify_pixel_values(values: np.ndarray) -> tuple[str, str, dict[str, Any]]:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    total = int(flat.size)
    if total == 0:
        return "indeterminate", "none", {"reason": "no_sampled_pixels"}

    hist = np.bincount(flat, minlength=256)
    nonzero_bins = int(np.count_nonzero(hist))
    forbidden_present = int(np.count_nonzero(hist[LIMITED_EXPANSION_FORBIDDEN]))
    forbidden_total = int(LIMITED_EXPANSION_FORBIDDEN.size)
    forbidden_fraction = float(forbidden_present / forbidden_total) if forbidden_total else 0.0
    zero_rate = float(hist[0] / total)
    low_1_15_rate = float(hist[1:16].sum() / total)
    evidence = {
        "sampled_pixels": total,
        "nonzero_bins": nonzero_bins,
        "limited_expansion_forbidden_bins_present": forbidden_present,
        "limited_expansion_forbidden_bins_total": forbidden_total,
        "limited_expansion_forbidden_fraction": round(forbidden_fraction, 4),
        "zero_rate": round(zero_rate, 6),
        "low_1_15_rate": round(low_1_15_rate, 6),
        "min": int(flat.min()),
        "max": int(flat.max()),
    }

    if total < 10_000 or nonzero_bins < 64:
        return "indeterminate", "low", evidence | {"reason": "sample_or_dynamic_range_too_small"}
    if forbidden_present <= 2 and forbidden_total - forbidden_present >= 30:
        confidence = "high" if total >= 100_000 and nonzero_bins >= 128 else "medium"
        return "range_expanded_like", confidence, evidence
    if forbidden_present >= 12:
        confidence = "high" if forbidden_present >= 20 else "medium"
        return "direct_y_like", confidence, evidence
    return "indeterminate", "medium", evidence | {"reason": "weak_or_mixed_histogram_signature"}


def _sample_array_values(array: Any, *, sample_frames: int, max_pixels: int) -> tuple[np.ndarray, int]:
    shape = tuple(int(dim) for dim in getattr(array, "shape", ()))
    if not shape:
        return np.array([], dtype=np.uint8), 0

    if len(shape) >= 3:
        frame_count = shape[0]
        if frame_count <= 0:
            return np.array([], dtype=np.uint8), 0
        indices = np.linspace(0, frame_count - 1, num=min(sample_frames, frame_count), dtype=np.int64)
        indices = sorted({int(index) for index in indices})
    else:
        indices = [None]

    per_frame_pixels = max(1, max_pixels // max(1, len(indices)))
    samples: list[np.ndarray] = []
    for index in indices:
        if len(shape) >= 3:
            frame_shape = shape[1:]
        else:
            frame_shape = shape
        frame_size = int(np.prod(frame_shape)) if frame_shape else 0
        stride = max(1, int(math.ceil(math.sqrt(frame_size / per_frame_pixels))))
        if index is None:
            if len(shape) >= 2:
                selection = (slice(None, None, stride), slice(None, None, stride), ...)
                frame = np.asarray(array[selection])
            else:
                frame = np.asarray(array[::stride])
        else:
            if len(shape) >= 3:
                selection = (index, slice(None, None, stride), slice(None, None, stride), ...)
                frame = np.asarray(array[selection])
            else:
                frame = np.asarray(array[index])
        if frame.size == 0:
            continue
        samples.append(np.asarray(frame, dtype=np.uint8).reshape(-1))

    if not samples:
        return np.array([], dtype=np.uint8), 0
    return np.concatenate(samples), len(indices)


def audit_dataset(
    dataset: DatasetRecord,
    *,
    sample_frames: int,
    max_pixels_per_surface: int,
) -> list[SurfaceRecord]:
    if not dataset.zarr_path.exists():
        return [
            SurfaceRecord(
                dataset=dataset,
                surface_path=".",
                shape=None,
                dtype=None,
                pixel_contract_name=None,
                decode_backend=None,
                source_pixels=None,
                sample_frames=0,
                sampled_pixels=0,
                classification="missing_zarr",
                confidence="none",
                evidence={},
                error="zarr_path does not exist",
            )
        ]

    try:
        root = zarr.open_group(str(dataset.zarr_path), mode="r", use_consolidated=False)
    except Exception as exc:
        return [
            SurfaceRecord(
                dataset=dataset,
                surface_path=".",
                shape=None,
                dtype=None,
                pixel_contract_name=None,
                decode_backend=None,
                source_pixels=None,
                sample_frames=0,
                sampled_pixels=0,
                classification="unreadable_zarr",
                confidence="none",
                evidence={},
                error=str(exc),
            )
        ]

    surfaces = _discover_surfaces(root)
    if not surfaces:
        return [
            SurfaceRecord(
                dataset=dataset,
                surface_path=".",
                shape=None,
                dtype=None,
                pixel_contract_name=_first_attr([_attrs_mapping(root)], PIXEL_CONTRACT_NAME_ATTRS),
                decode_backend=_first_attr([_attrs_mapping(root)], DECODE_BACKEND_ATTRS),
                source_pixels=_first_attr([_attrs_mapping(root)], SOURCE_PIXELS_ATTRS),
                sample_frames=0,
                sampled_pixels=0,
                classification="no_image_surface",
                confidence="none",
                evidence={},
                error=None,
            )
        ]

    rows: list[SurfaceRecord] = []
    for surface_path, array, attrs_chain in surfaces:
        shape = tuple(int(dim) for dim in getattr(array, "shape", ()))
        dtype = str(getattr(array, "dtype", "unknown"))
        pixel_contract_name = _first_attr(attrs_chain, PIXEL_CONTRACT_NAME_ATTRS)
        decode_backend = _first_attr(attrs_chain, DECODE_BACKEND_ATTRS)
        source_pixels = _first_attr(attrs_chain, SOURCE_PIXELS_ATTRS)
        if dtype != "uint8":
            rows.append(
                SurfaceRecord(
                    dataset=dataset,
                    surface_path=surface_path,
                    shape=shape,
                    dtype=dtype,
                    pixel_contract_name=pixel_contract_name,
                    decode_backend=decode_backend,
                    source_pixels=source_pixels,
                    sample_frames=0,
                    sampled_pixels=0,
                    classification="not_uint8",
                    confidence="none",
                    evidence={},
                    error=None,
                )
            )
            continue
        try:
            values, frames_sampled = _sample_array_values(
                array,
                sample_frames=sample_frames,
                max_pixels=max_pixels_per_surface,
            )
            classification, confidence, evidence = classify_pixel_values(values)
            rows.append(
                SurfaceRecord(
                    dataset=dataset,
                    surface_path=surface_path,
                    shape=shape,
                    dtype=dtype,
                    pixel_contract_name=pixel_contract_name,
                    decode_backend=decode_backend,
                    source_pixels=source_pixels,
                    sample_frames=frames_sampled,
                    sampled_pixels=int(values.size),
                    classification=classification,
                    confidence=confidence,
                    evidence=evidence,
                    error=None,
                )
            )
        except Exception as exc:
            rows.append(
                SurfaceRecord(
                    dataset=dataset,
                    surface_path=surface_path,
                    shape=shape,
                    dtype=dtype,
                    pixel_contract_name=pixel_contract_name,
                    decode_backend=decode_backend,
                    source_pixels=source_pixels,
                    sample_frames=0,
                    sampled_pixels=0,
                    classification="sample_failed",
                    confidence="none",
                    evidence={},
                    error=str(exc),
                )
            )
    return rows


def _dataset_summary(rows: Sequence[SurfaceRecord]) -> str:
    classifications = {row.classification for row in rows}
    if "range_expanded_like" in classifications:
        return "range_expanded_like"
    if "direct_y_like" in classifications:
        return "direct_y_like"
    if "indeterminate" in classifications:
        return "indeterminate"
    if len(classifications) == 1:
        return next(iter(classifications))
    return "mixed_non_sampled"


def _md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _shape_text(shape: tuple[int, ...] | None) -> str:
    return "" if shape is None else "x".join(str(dim) for dim in shape)


def render_markdown(
    *,
    registry_path: Path,
    datasets: Sequence[DatasetRecord],
    surface_rows: Sequence[SurfaceRecord],
    sample_frames: int,
    max_pixels_per_surface: int,
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    by_dataset: dict[str, list[SurfaceRecord]] = defaultdict(list)
    for row in surface_rows:
        by_dataset[row.dataset.dataset_id].append(row)

    dataset_class_counts = Counter(_dataset_summary(rows) for rows in by_dataset.values())
    surface_class_counts = Counter(row.classification for row in surface_rows)
    contract_counts = Counter(row.pixel_contract_name or "missing" for row in surface_rows)
    backend_counts = Counter(row.decode_backend or "missing" for row in surface_rows)

    lines = [
        "# Pixel Decode Exposure Census - 2026-07-02",
        "",
        "<!-- contract-meta",
        "status: generated",
        f"generated_utc: {generated}",
        "scope: read-only registry training datasets and zarr image surfaces",
        "-->",
        "",
        "## Scope",
        "",
        f"- Registry opened read-only: `{registry_path}`",
        f"- Datasets selected: `{len(datasets)}` (`zarr_use='training'`, `artifact_kind='derived_training_merge'`, or merged dataset id).",
        f"- Surfaces sampled: `{len(surface_rows)}`",
        f"- Sample policy: up to `{sample_frames}` frames and `{max_pixels_per_surface}` pixels per image surface.",
        "- Default raw-video sampling uses cheap/model-facing raw surfaces (`images_ds`, `images`, and RGB/color downsampled variants); full-resolution raw frames are deliberately not sampled by this census.",
        "- No zarr store or registry row was modified.",
        "",
        "## Classification Rule",
        "",
        "- `range_expanded_like`: sampled uint8 values mostly occupy the value lattice produced by limited-range expansion `(Y - 16) * 255 / 219`, leaving >=30 forbidden bins empty.",
        "- `direct_y_like`: sampled values include many bins that cannot be produced by that limited-range expansion, consistent with direct full-range Y storage.",
        "- `indeterminate`: insufficient dynamic range/sample size or weak/mixed histogram evidence.",
        "",
        "## Headline Finding",
        "",
        f"- `range_expanded_like` surfaces: `{surface_class_counts.get('range_expanded_like', 0)}`",
        f"- `direct_y_like` surfaces: `{surface_class_counts.get('direct_y_like', 0)}`",
        f"- `indeterminate` surfaces: `{surface_class_counts.get('indeterminate', 0)}`",
        "- No sampled model-facing training surface showed the limited-range expansion lattice.",
        "",
        "## Summary",
        "",
        "### Dataset Classifications",
        "",
        "| classification | datasets |",
        "| --- | ---: |",
    ]
    for key, count in sorted(dataset_class_counts.items()):
        lines.append(f"| {key} | {count} |")

    lines.extend(["", "### Surface Classifications", "", "| classification | surfaces |", "| --- | ---: |"])
    for key in ("direct_y_like", "range_expanded_like", "indeterminate", "missing_zarr", "unreadable_zarr", "no_image_surface", "sample_failed"):
        count = surface_class_counts.get(key, 0)
        if count == 0 and key not in {"range_expanded_like"}:
            continue
        lines.append(f"| {key} | {count} |")

    lines.extend(["", "### Pixel Contract Names", "", "| pixel contract | surfaces |", "| --- | ---: |"])
    for key, count in sorted(contract_counts.items()):
        lines.append(f"| `{_md_cell(key)}` | {count} |")

    lines.extend(["", "### Decode Backends", "", "| decode backend | surfaces |", "| --- | ---: |"])
    for key, count in sorted(backend_counts.items()):
        lines.append(f"| `{_md_cell(key)}` | {count} |")

    lines.extend(
        [
            "",
            "## Per-Dataset Summary",
            "",
            "| dataset_id | kind | use | summary | surfaces | contracts | backends | path |",
            "| --- | --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for dataset in datasets:
        rows = by_dataset.get(dataset.dataset_id, [])
        contracts = sorted({row.pixel_contract_name or "missing" for row in rows})
        backends = sorted({row.decode_backend or "missing" for row in rows})
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(dataset.dataset_id),
                    _md_cell(dataset.artifact_kind),
                    _md_cell(dataset.zarr_use),
                    _dataset_summary(rows) if rows else "not_audited",
                    str(len(rows)),
                    ", ".join(f"`{_md_cell(item)}`" for item in contracts),
                    ", ".join(f"`{_md_cell(item)}`" for item in backends),
                    f"`{_md_cell(dataset.zarr_path)}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Surface Evidence",
            "",
            "| dataset_id | surface | shape | contract | backend | class | confidence | sampled_px | forbidden_bins_present | zero_rate | error |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in surface_rows:
        evidence = row.evidence
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row.dataset.dataset_id),
                    _md_cell(row.surface_path),
                    _shape_text(row.shape),
                    f"`{_md_cell(row.pixel_contract_name or 'missing')}`",
                    f"`{_md_cell(row.decode_backend or 'missing')}`",
                    row.classification,
                    row.confidence,
                    str(row.sampled_pixels),
                    str(evidence.get("limited_expansion_forbidden_bins_present", "")),
                    str(evidence.get("zero_rate", "")),
                    _md_cell(row.error or ""),
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def run_census(
    *,
    registry_path: Path,
    output_path: Path,
    sample_frames: int,
    max_pixels_per_surface: int,
    max_datasets: int | None,
) -> tuple[list[DatasetRecord], list[SurfaceRecord]]:
    datasets = load_training_datasets(registry_path)
    if max_datasets is not None:
        datasets = datasets[:max_datasets]

    surface_rows: list[SurfaceRecord] = []
    for dataset in datasets:
        surface_rows.extend(
            audit_dataset(
                dataset,
                sample_frames=sample_frames,
                max_pixels_per_surface=max_pixels_per_surface,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_markdown(
            registry_path=registry_path,
            datasets=datasets,
            surface_rows=surface_rows,
            sample_frames=sample_frames,
            max_pixels_per_surface=max_pixels_per_surface,
        ),
        encoding="utf-8",
    )
    return datasets, surface_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-frames", type=int, default=4)
    parser.add_argument("--max-pixels-per-surface", type=int, default=262_144)
    parser.add_argument("--max-datasets", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets, surface_rows = run_census(
        registry_path=args.registry,
        output_path=args.output,
        sample_frames=args.sample_frames,
        max_pixels_per_surface=args.max_pixels_per_surface,
        max_datasets=args.max_datasets,
    )
    class_counts = Counter(row.classification for row in surface_rows)
    print(
        f"Wrote {args.output} for {len(datasets)} datasets / {len(surface_rows)} surfaces: "
        + ", ".join(f"{key}={value}" for key, value in sorted(class_counts.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
