#!/usr/bin/env python3
"""Aggregate a training-image data card from registry profile rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.utils import plot_training_image_data_card as plot_data_card


SCHEMA_NAME = "training_image_data_card"
SCHEMA_VERSION = "v1"
METRIC_NAMES = (
    "mean_intensity_p50",
    "contrast_p50",
    "sharpness_p50",
    "clip_dark_rate_mean",
    "clip_bright_rate_mean",
    "illumination_center_edge_p50",
    "illumination_slope_x_p50",
    "illumination_slope_y_p50",
    "fish_bg_contrast_p50",
)
COMPOSITION_FIELDS = ("rig_id", "camera_id", "arena_id", "dish_design", "canvas_name", "protocol_name")


_utc_now = utc_now


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _numeric_stats(values: Sequence[Any]) -> Optional[dict[str, Any]]:
    arr = np.asarray([float(v) for v in values if _as_float(v) is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _parse_json_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        raw = value.decode("utf-8", "ignore")
    elif isinstance(value, str):
        raw = value
    else:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Manifest has no datasets: {path}")
    return dict(payload)


def _resolve_dataset_id_from_registry(registry: Registry, zarr_path: Path) -> Optional[str]:
    candidates = [str(zarr_path)]
    try:
        resolved = str(zarr_path.resolve())
    except OSError:
        resolved = None
    if resolved and resolved not in candidates:
        candidates.append(resolved)
    for candidate in candidates:
        row = registry.conn.execute(
            "SELECT dataset_id FROM datasets WHERE zarr_path = ? LIMIT 1;",
            (candidate,),
        ).fetchone()
        if row is not None:
            return _normalize_text(row["dataset_id"])
    return None


def _manifest_dataset_ids(registry: Registry, manifest: Mapping[str, Any]) -> list[str]:
    rows = manifest.get("datasets")
    if not isinstance(rows, list):
        return []
    dataset_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        dataset_id = _normalize_text(row.get("dataset_id"))
        zarr_path_text = _normalize_text(row.get("zarr_path"))
        if zarr_path_text:
            dataset_id = _resolve_dataset_id_from_registry(registry, Path(zarr_path_text)) or dataset_id
        if dataset_id:
            dataset_ids.append(dataset_id)
    return list(dict.fromkeys(dataset_ids))


def _default_output_path(manifest_path: Path, set_id: Optional[str]) -> Path:
    base = _normalize_text(set_id) or manifest_path.stem
    return manifest_path.parent / f"{base}.training_image_data_card.json"


def _default_plot_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.plots"


def _aggregate_intensity_histograms(profile_summaries: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    canonical_edges: Optional[np.ndarray] = None
    aggregate_counts: Optional[np.ndarray] = None
    included = 0
    skipped = 0
    for summary in profile_summaries:
        hist = summary.get("intensity_histogram")
        if not isinstance(hist, Mapping):
            continue
        edges_raw = hist.get("bin_edges")
        counts_raw = hist.get("counts")
        if not isinstance(edges_raw, Sequence) or not isinstance(counts_raw, Sequence):
            continue
        edges = np.asarray([float(v) for v in edges_raw], dtype=np.float64)
        counts = np.asarray([int(v) for v in counts_raw], dtype=np.int64)
        if edges.size != counts.size + 1:
            continue
        if canonical_edges is None:
            canonical_edges = edges
            aggregate_counts = np.zeros_like(counts, dtype=np.int64)
        if aggregate_counts is None or canonical_edges.shape != edges.shape or not np.allclose(canonical_edges, edges):
            skipped += 1
            continue
        aggregate_counts += counts
        included += 1
    if canonical_edges is None or aggregate_counts is None:
        return None
    return {
        "bin_edges": [float(v) for v in canonical_edges.tolist()],
        "counts": [int(v) for v in aggregate_counts.tolist()],
        "source_dataset_count": int(included),
        "skipped_mismatched_bins": int(skipped),
    }


def _build_composition_counts(profile_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    payload: dict[str, dict[str, int]] = {}
    for field in COMPOSITION_FIELDS:
        counter: Counter[str] = Counter()
        for row in profile_rows:
            value = _normalize_text(row.get(field))
            if value:
                counter[value] += 1
        if counter:
            payload[field] = {key: int(counter[key]) for key in sorted(counter)}
    return payload


def build_training_image_data_card(
    *,
    manifest: Mapping[str, Any],
    profile_rows: Sequence[Mapping[str, Any]],
    profile_summaries: Sequence[Mapping[str, Any]],
    missing_profile_dataset_ids: Sequence[str],
    split: str,
) -> dict[str, Any]:
    metric_values = {
        metric: [
            float(value)
            for row in profile_rows
            for value in [_as_float(row.get(metric))]
            if value is not None
        ]
        for metric in METRIC_NAMES
    }
    metric_stats = {
        metric: stats
        for metric, values in metric_values.items()
        for stats in [_numeric_stats(values)]
        if stats is not None
    }
    profile_refs = [
        {
            "dataset_id": _normalize_text(row.get("dataset_id")),
            "profile_run": _normalize_text(row.get("profile_run")),
            "recording_id": _normalize_text(row.get("recording_id")),
            "zarr_use": _normalize_text(row.get("zarr_use")),
            "source_frame_array": _normalize_text(row.get("source_frame_array")),
        }
        for row in profile_rows
    ]
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "set_id": _normalize_text(manifest.get("set_id")),
        "set_version": _normalize_text(manifest.get("set_version")),
        "selection": {
            "split": split,
            "manifest_dataset_count": int(len(manifest.get("datasets", []))),
            "profiled_dataset_count": int(len(profile_rows)),
            "missing_profile_dataset_count": int(len(missing_profile_dataset_ids)),
            "missing_profile_dataset_ids": list(missing_profile_dataset_ids),
        },
        "profile_run_refs": profile_refs,
        "metric_stats": metric_stats,
        "metric_values": metric_values,
        "intensity_histogram_aggregate": _aggregate_intensity_histograms(profile_summaries),
        "composition_counts": _build_composition_counts(profile_rows),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Training export manifest JSON.")
    parser.add_argument("--registry", type=Path, help="Registry SQLite path.")
    parser.add_argument("--output", type=Path, help="Output data-card JSON path.")
    parser.add_argument("--split", default="all", help="Split label recorded in the data card.")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate plot PNGs after writing card.")
    parser.add_argument("--plot-dir", type=Path, help="Output directory for plots.")
    parser.add_argument("--allow-missing", action="store_true", help="Allow datasets without training-image profiles.")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        dataset_ids = _manifest_dataset_ids(registry, manifest)
        if not dataset_ids:
            raise ValueError("No manifest dataset IDs could be resolved from registry.")
        rows = [dict(row) for row in registry.query_training_image_profile_latest(dataset_ids=dataset_ids)]
    finally:
        registry.close()
    rows_by_dataset = {
        str(row["dataset_id"]): row
        for row in rows
        if _normalize_text(row.get("dataset_id")) is not None
    }
    missing = [dataset_id for dataset_id in dataset_ids if dataset_id not in rows_by_dataset]
    if missing and not args.allow_missing:
        raise ValueError("Missing training image profiles for dataset_id(s): " + ", ".join(missing))
    profile_summaries = [
        summary
        for row in rows
        for summary in [_parse_json_mapping(row.get("profile_json"))]
        if summary is not None
    ]
    card = build_training_image_data_card(
        manifest=manifest,
        profile_rows=rows,
        profile_summaries=profile_summaries,
        missing_profile_dataset_ids=missing,
        split=str(args.split),
    )
    output_path = Path(args.output) if args.output is not None else _default_output_path(manifest_path, _normalize_text(manifest.get("set_id")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(strict_json_dumps(card, sort_keys=True) + "\n", encoding="utf-8")

    generated_plots: list[Path] = []
    if not args.no_plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else _default_plot_dir(output_path)
        generated_plots = plot_data_card.generate_training_image_data_card_plots(
            card_payload=card,
            output_dir=plot_dir,
            prefix=_normalize_text(card.get("set_id")) or output_path.stem,
        )
    print(
        "Training image data card: "
        f"datasets={len(dataset_ids)} profiled={len(rows)} missing={len(missing)} "
        f"output={output_path} plots={len(generated_plots)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
