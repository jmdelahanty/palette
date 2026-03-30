#!/usr/bin/env python3
"""Aggregate a detection training data card from registry profile rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fisheye.shared.batch_logging import utc_now
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils import plot_detection_training_data_card as plot_data_card


SCHEMA_NAME = "detection_training_data_card"
SCHEMA_VERSION = "v1"
HISTOGRAM_NAMES = ("w_norm", "h_norm", "area_norm", "aspect_ratio")
COMPOSITION_FIELDS = ("rig_id", "camera_id", "arena_id", "dish_design", "canvas_name", "protocol_name")


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    zarr_path: Path
    detection_source_type: Optional[str]


@dataclass(frozen=True)
class SubjectLineageCoverage:
    manifest_dataset_count: int
    lineage_covered_dataset_count: Optional[int]
    missing_lineage_dataset_ids: tuple[str, ...]
    coverage_unavailable_reason: Optional[str]


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
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _weighted_mean(values: Sequence[tuple[Optional[float], Optional[float]]]) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in values:
        if value is None:
            continue
        effective_weight = float(weight) if weight is not None and float(weight) > 0 else 1.0
        weighted_sum += float(value) * effective_weight
        total_weight += effective_weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _numeric_stats(values: Sequence[Optional[float]]) -> Optional[dict[str, Any]]:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
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


def _manifest_dataset_refs(registry: Registry, manifest: Mapping[str, Any]) -> list[DatasetRef]:
    rows = manifest.get("datasets")
    assert isinstance(rows, list)
    refs: list[DatasetRef] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        zarr_path_text = _normalize_text(row.get("zarr_path"))
        if zarr_path_text is None:
            raise ValueError("Manifest dataset row missing zarr_path.")
        zarr_path = Path(zarr_path_text)
        # Prefer canonical registry dataset_id keyed by zarr_path. This handles
        # legacy manifests that may store recording_id/session_uuid instead of
        # the collision-safe dataset_id used by registry profile tables.
        dataset_id = _resolve_dataset_id_from_registry(registry, zarr_path) or _normalize_text(row.get("dataset_id"))
        if dataset_id is None:
            raise ValueError(f"Unable to resolve dataset_id for zarr path: {zarr_path}")
        refs.append(
            DatasetRef(
                dataset_id=dataset_id,
                zarr_path=zarr_path,
                detection_source_type=_normalize_text(row.get("detection_source_type")),
            )
        )
    if not refs:
        raise ValueError("Manifest contains no usable dataset rows.")
    return refs


def _normalize_manifest_stem(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def _default_output_path(manifest_path: Path, set_id: Optional[str]) -> Path:
    base = _normalize_manifest_stem(set_id or manifest_path.stem) or "detection_training_data_card"
    return manifest_path.parent / f"{base}.data_card.json"


def _default_plot_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.plots"


def _select_profile_rows(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    rows = registry.query_detection_data_profile_latest(dataset_ids=dataset_ids)
    profile_by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        dataset_id = _normalize_text(payload.get("dataset_id"))
        if dataset_id is None:
            continue
        profile_by_dataset[dataset_id] = payload
    return profile_by_dataset


def _load_legacy_profile_row(
    registry: Registry,
    *,
    dataset_id: str,
    profile_run: str,
) -> Optional[dict[str, Any]]:
    row = registry.conn.execute(
        """
        SELECT
            recording_id,
            zarr_use,
            rig_id,
            camera_id,
            arena_id,
            dish_design,
            canvas_name,
            protocol_name,
            genotype,
            dpf_at_acquisition
        FROM detection_data_profile
        WHERE dataset_id = ? AND profile_run = ?
        LIMIT 1;
        """,
        (dataset_id, profile_run),
    ).fetchone()
    return dict(row) if row is not None else None


def _coalesce_text_value(*values: Any) -> Optional[str]:
    for value in values:
        text = _normalize_text(value)
        if text is not None:
            return text
    return None


def _coalesce_int_value(*values: Any) -> Optional[int]:
    for value in values:
        number = _as_int(value)
        if number is not None:
            return number
    return None


def _apply_profile_context_fallbacks(
    row: dict[str, Any],
    *,
    profile_summary: Mapping[str, Any],
    legacy_row: Optional[Mapping[str, Any]],
) -> None:
    composition = profile_summary.get("composition")
    composition_map = composition if isinstance(composition, Mapping) else {}
    legacy = legacy_row or {}

    row["recording_id"] = _coalesce_text_value(row.get("recording_id"), legacy.get("recording_id"))
    row["zarr_use"] = _coalesce_text_value(row.get("zarr_use"), legacy.get("zarr_use"))

    for field in COMPOSITION_FIELDS:
        row[field] = _coalesce_text_value(
            row.get(field),
            composition_map.get(field),
            legacy.get(field),
        )

    row["genotype"] = _coalesce_text_value(row.get("genotype"), legacy.get("genotype"))
    row["dpf_at_acquisition"] = _coalesce_int_value(
        row.get("dpf_at_acquisition"),
        legacy.get("dpf_at_acquisition"),
    )


def _query_subject_lineage_dataset_ids(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> set[str]:
    if not dataset_ids:
        return set()
    placeholders = ", ".join("?" for _ in dataset_ids)
    sql = (
        "SELECT DISTINCT dataset_id FROM recording_subject_overview "
        f"WHERE dataset_id IN ({placeholders});"
    )
    rows = registry.conn.execute(sql, tuple(dataset_ids)).fetchall()
    covered: set[str] = set()
    for row in rows:
        dataset_id = _normalize_text(row["dataset_id"])
        if dataset_id is not None:
            covered.add(dataset_id)
    return covered


def _evaluate_subject_lineage_coverage(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
    subject_lineage_policy: str,
) -> SubjectLineageCoverage:
    policy = _normalize_text(subject_lineage_policy) or "warn"
    if policy not in {"warn", "require"}:
        raise ValueError(f"Unsupported subject lineage policy: {subject_lineage_policy}")

    unique_dataset_ids = list(dict.fromkeys(dataset_ids))
    total_count = len(unique_dataset_ids)
    try:
        covered_dataset_ids = _query_subject_lineage_dataset_ids(
            registry,
            dataset_ids=unique_dataset_ids,
        )
    except Exception as exc:
        if policy == "require":
            raise ValueError(
                f"Subject lineage coverage unavailable (recording_subject_overview query failed): {exc}"
            ) from exc
        reason = f"recording_subject_overview query failed: {exc}"
        print(
            "Subject lineage coverage: unavailable "
            f"({reason})"
        )
        return SubjectLineageCoverage(
            manifest_dataset_count=int(total_count),
            lineage_covered_dataset_count=None,
            missing_lineage_dataset_ids=tuple(sorted(unique_dataset_ids)),
            coverage_unavailable_reason=reason,
        )

    missing_dataset_ids = sorted(
        dataset_id
        for dataset_id in unique_dataset_ids
        if dataset_id not in covered_dataset_ids
    )
    covered_count = total_count - len(missing_dataset_ids)
    print(f"Subject lineage coverage: {covered_count}/{total_count} datasets")
    if missing_dataset_ids:
        print(
            "Subject lineage missing dataset_id(s): "
            + ", ".join(missing_dataset_ids)
        )
    if policy == "require" and missing_dataset_ids:
        raise ValueError(
            "Missing subject lineage rows in recording_subject_overview for dataset_id(s): "
            + ", ".join(missing_dataset_ids)
        )
    return SubjectLineageCoverage(
        manifest_dataset_count=int(total_count),
        lineage_covered_dataset_count=int(covered_count),
        missing_lineage_dataset_ids=tuple(missing_dataset_ids),
        coverage_unavailable_reason=None,
    )


def _build_subject_coverage_payload(coverage: SubjectLineageCoverage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "manifest_dataset_count": int(coverage.manifest_dataset_count),
        "lineage_covered_dataset_count": (
            int(coverage.lineage_covered_dataset_count)
            if coverage.lineage_covered_dataset_count is not None
            else None
        ),
        "missing_lineage_dataset_ids": list(coverage.missing_lineage_dataset_ids),
    }
    if coverage.coverage_unavailable_reason:
        payload["coverage_unavailable_reason"] = str(coverage.coverage_unavailable_reason)
    return payload


def _build_genotype_counts(profile_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in profile_rows:
        genotype = _normalize_text(row.get("genotype"))
        if genotype is not None:
            counter[genotype] += 1
    return {key: int(counter[key]) for key in sorted(counter)}


def _build_dpf_histogram(
    dpf_values: Sequence[Optional[float]],
    *,
    source_dataset_count: int,
) -> Optional[dict[str, Any]]:
    arr = np.asarray([float(v) for v in dpf_values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    low = int(np.floor(float(np.min(arr))))
    high = int(np.ceil(float(np.max(arr))))
    if high < low:
        return None
    edges = np.arange(low - 0.5, high + 1.5, 1.0, dtype=np.float64)
    counts, _ = np.histogram(arr, bins=edges)
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.astype(np.int64).tolist()],
        "source_dataset_count": int(arr.size),
        "skipped_missing_values": int(max(0, int(source_dataset_count) - int(arr.size))),
    }


def _aggregate_histograms(profile_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for hist_name in HISTOGRAM_NAMES:
        canonical_edges: Optional[np.ndarray] = None
        aggregate_counts: Optional[np.ndarray] = None
        included = 0
        skipped = 0
        for summary in profile_summaries:
            histograms = summary.get("histograms")
            if not isinstance(histograms, Mapping):
                continue
            hist = histograms.get(hist_name)
            if not isinstance(hist, Mapping):
                continue
            edges_raw = hist.get("bin_edges")
            counts_raw = hist.get("counts")
            if not isinstance(edges_raw, Sequence) or not isinstance(counts_raw, Sequence):
                continue
            edges = np.asarray([float(x) for x in edges_raw], dtype=np.float64)
            counts = np.asarray([int(x) for x in counts_raw], dtype=np.int64)
            if edges.size != counts.size + 1:
                continue
            if canonical_edges is None:
                canonical_edges = edges
                aggregate_counts = np.zeros_like(counts, dtype=np.int64)
            if (
                aggregate_counts is None
                or canonical_edges.shape != edges.shape
                or not np.allclose(canonical_edges, edges)
            ):
                skipped += 1
                continue
            aggregate_counts += counts
            included += 1
        if canonical_edges is None or aggregate_counts is None:
            continue
        result[hist_name] = {
            "bin_edges": [float(x) for x in canonical_edges.tolist()],
            "counts": [int(x) for x in aggregate_counts.tolist()],
            "source_dataset_count": int(included),
            "skipped_mismatched_bins": int(skipped),
        }
    return result


def _aggregate_center_heatmap(
    profile_summaries: Sequence[Mapping[str, Any]],
    *,
    weights: Sequence[Optional[float]],
) -> Optional[dict[str, Any]]:
    canonical_shape: Optional[tuple[int, int]] = None
    weighted_sum: Optional[np.ndarray] = None
    total_weight = 0.0
    included = 0
    skipped = 0
    for index, summary in enumerate(profile_summaries):
        spatial = summary.get("spatial")
        if not isinstance(spatial, Mapping):
            continue
        heatmap = spatial.get("center_heatmap")
        if not isinstance(heatmap, Mapping):
            continue
        grid_h = _as_int(heatmap.get("grid_h"))
        grid_w = _as_int(heatmap.get("grid_w"))
        density_raw = heatmap.get("density")
        if grid_h is None or grid_w is None or not isinstance(density_raw, Sequence):
            continue
        density = np.asarray([float(x) for x in density_raw], dtype=np.float64)
        if density.size != int(grid_h) * int(grid_w):
            continue
        shape = (int(grid_h), int(grid_w))
        if canonical_shape is None:
            canonical_shape = shape
            weighted_sum = np.zeros(shape, dtype=np.float64)
        if weighted_sum is None or shape != canonical_shape:
            skipped += 1
            continue
        weight = float(weights[index]) if weights[index] is not None and float(weights[index]) > 0 else 1.0
        weighted_sum += density.reshape(shape) * weight
        total_weight += weight
        included += 1
    if canonical_shape is None or weighted_sum is None or total_weight <= 0:
        return None
    density = weighted_sum / total_weight
    density_sum = float(np.sum(density))
    if density_sum > 0:
        density = density / density_sum
    return {
        "grid_h": int(canonical_shape[0]),
        "grid_w": int(canonical_shape[1]),
        "density": [float(x) for x in density.reshape(-1).tolist()],
        "source_dataset_count": int(included),
        "skipped_mismatched_grids": int(skipped),
    }


def _build_geometry_aggregate(profile_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    detections_weights = [_as_float(row.get("detections_total")) for row in profile_rows]

    def _metric(prefix: str) -> dict[str, Optional[float]]:
        return {
            "p10_weighted": _weighted_mean([(_as_float(row.get(f"{prefix}_p10")), detections_weights[idx]) for idx, row in enumerate(profile_rows)]),
            "p50_weighted": _weighted_mean([(_as_float(row.get(f"{prefix}_p50")), detections_weights[idx]) for idx, row in enumerate(profile_rows)]),
            "p90_weighted": _weighted_mean([(_as_float(row.get(f"{prefix}_p90")), detections_weights[idx]) for idx, row in enumerate(profile_rows)]),
        }

    return {
        "w": _metric("w"),
        "h": _metric("h"),
        "area": _metric("area"),
        "aspect_ratio": _metric("aspect_ratio"),
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


def _build_card(
    *,
    manifest: Mapping[str, Any],
    dataset_refs: Sequence[DatasetRef],
    profile_rows: Sequence[Mapping[str, Any]],
    profile_summaries: Sequence[Mapping[str, Any]],
    subject_lineage_coverage: SubjectLineageCoverage,
    split: str,
) -> dict[str, Any]:
    dataset_count = len(dataset_refs)
    frames_total_values = [_as_float(row.get("frames_total")) for row in profile_rows]
    frames_with_values = [_as_float(row.get("frames_with_detections")) for row in profile_rows]
    detections_total_values = [_as_float(row.get("detections_total")) for row in profile_rows]
    coverage_percent_values = [_as_float(row.get("coverage_percent")) for row in profile_rows]
    dpf_p50_values = [_as_float(row.get("detections_per_frame_p50")) for row in profile_rows]
    dpf_p90_values = [_as_float(row.get("detections_per_frame_p90")) for row in profile_rows]
    lineage_dpf_values = [_as_float(row.get("dpf_at_acquisition")) for row in profile_rows]

    frames_total_sum = int(sum(v for v in frames_total_values if v is not None))
    frames_with_sum = int(sum(v for v in frames_with_values if v is not None))
    detections_total_sum = int(sum(v for v in detections_total_values if v is not None))

    coverage_percent_overall = (
        100.0 * _safe_ratio(float(frames_with_sum), float(frames_total_sum))
        if frames_total_sum > 0
        else None
    )

    detections_weights = [(_as_float(row.get("detections_total")) or 1.0) for row in profile_rows]
    edge_rate_weighted = _weighted_mean(
        [
            (_as_float(row.get("edge_proximity_rate")), detections_weights[idx])
            for idx, row in enumerate(profile_rows)
        ]
    )

    set_id = _normalize_text(manifest.get("set_id"))
    set_version_value = manifest.get("set_version")
    if isinstance(set_version_value, (int, str)):
        set_version = str(set_version_value)
    else:
        set_version = None

    query_filter = manifest.get("query_filter")
    selection_filters = dict(query_filter) if isinstance(query_filter, Mapping) else {}
    source_type = _normalize_text(manifest.get("source_type"))
    if source_type:
        selection_filters.setdefault("source_type", source_type)

    card: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "set_id": set_id,
        "set_version": set_version,
        "selection": {
            "dataset_count": int(dataset_count),
            "split": split,
            "filters": selection_filters,
        },
        "coverage": {
            "frames_total": int(frames_total_sum),
            "frames_with_detections": int(frames_with_sum),
            "coverage_percent_overall": float(coverage_percent_overall) if coverage_percent_overall is not None else None,
            "coverage_percent_dataset_stats": _numeric_stats(coverage_percent_values),
        },
        "counts": {
            "detections_total": int(detections_total_sum),
            "detections_total_dataset_stats": _numeric_stats(detections_total_values),
            "detections_per_frame_p50_weighted": _weighted_mean([(dpf_p50_values[idx], detections_weights[idx]) for idx in range(len(profile_rows))]),
            "detections_per_frame_p90_weighted": _weighted_mean([(dpf_p90_values[idx], detections_weights[idx]) for idx in range(len(profile_rows))]),
        },
        "geometry_norm_aggregate": _build_geometry_aggregate(profile_rows),
        "spatial_aggregate": {
            "edge_proximity_rate_weighted": edge_rate_weighted,
            "center_heatmap": _aggregate_center_heatmap(profile_summaries, weights=detections_weights),
        },
        "histograms_aggregate": _aggregate_histograms(profile_summaries),
        "composition_counts": _build_composition_counts(profile_rows),
        "subject_coverage": _build_subject_coverage_payload(subject_lineage_coverage),
        "genotype_counts": _build_genotype_counts(profile_rows),
        "dpf_stats": _numeric_stats(lineage_dpf_values),
        "dpf_histogram": _build_dpf_histogram(
            lineage_dpf_values,
            source_dataset_count=len(profile_rows),
        ),
        "train_val_parity": None,
        "profile_run_refs": [
            {
                "dataset_id": str(row.get("dataset_id")),
                "profile_run": str(row.get("profile_run")),
            }
            for row in sorted(profile_rows, key=lambda item: str(item.get("dataset_id") or ""))
        ],
    }
    return card


def _build_detection_training_data_card(
    *,
    registry: Registry,
    manifest: Mapping[str, Any],
    split: str,
    allow_mtime_mismatch: bool,
    allow_detection_type_mismatch: bool,
    subject_lineage_policy: str,
) -> dict[str, Any]:
    refs = _manifest_dataset_refs(registry, manifest)
    dataset_ids = [item.dataset_id for item in refs]
    subject_lineage_coverage = _evaluate_subject_lineage_coverage(
        registry,
        dataset_ids=dataset_ids,
        subject_lineage_policy=subject_lineage_policy,
    )
    profiles_by_dataset = _select_profile_rows(registry, dataset_ids=dataset_ids)

    missing_profiles = [item.dataset_id for item in refs if item.dataset_id not in profiles_by_dataset]
    if missing_profiles:
        missing_text = ", ".join(sorted(missing_profiles))
        raise ValueError(f"Missing detection_data_profile_latest rows for dataset_id(s): {missing_text}")

    mtime_mismatches: list[str] = []
    type_mismatches: list[str] = []
    profile_rows: list[dict[str, Any]] = []
    profile_summaries: list[dict[str, Any]] = []
    for item in refs:
        row = dict(profiles_by_dataset[item.dataset_id])
        detection_type = _normalize_text(row.get("detection_type"))
        if (
            not allow_detection_type_mismatch
            and item.detection_source_type is not None
            and detection_type is not None
            and item.detection_source_type != detection_type
        ):
            type_mismatches.append(
                f"{item.dataset_id}:{item.detection_source_type}!={detection_type}"
            )

        expected_mtime = _as_int(row.get("zarr_mtime_ns"))
        if expected_mtime is not None and item.zarr_path.exists():
            try:
                observed_mtime = int(item.zarr_path.stat().st_mtime_ns)
            except OSError:
                observed_mtime = None
            if observed_mtime is not None and observed_mtime != expected_mtime:
                mtime_mismatches.append(
                    f"{item.dataset_id}:{expected_mtime}!={observed_mtime}"
                )

        summary = _parse_json_mapping(row.get("profile_json"))
        if summary is None:
            raise ValueError(
                f"Invalid profile_json for dataset_id '{item.dataset_id}' run '{row.get('profile_run')}'."
            )
        profile_run = _normalize_text(row.get("profile_run"))
        legacy_row = (
            _load_legacy_profile_row(
                registry,
                dataset_id=item.dataset_id,
                profile_run=profile_run,
            )
            if profile_run is not None
            else None
        )
        _apply_profile_context_fallbacks(
            row,
            profile_summary=summary,
            legacy_row=legacy_row,
        )
        profile_rows.append(row)
        profile_summaries.append(summary)

    if type_mismatches:
        mismatch_text = ", ".join(sorted(type_mismatches))
        raise ValueError(f"Detection type mismatches: {mismatch_text}")
    if mtime_mismatches and not allow_mtime_mismatch:
        mismatch_text = ", ".join(sorted(mtime_mismatches))
        raise ValueError(f"Profile zarr_mtime_ns mismatches: {mismatch_text}")

    return _build_card(
        manifest=manifest,
        dataset_refs=refs,
        profile_rows=profile_rows,
        profile_summaries=profile_summaries,
        subject_lineage_coverage=subject_lineage_coverage,
        split=split,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate detection training-data card payload from detection_data_profile_latest rows."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest JSON path.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: <set_id>.data_card.json next to manifest).")
    parser.add_argument("--split", type=str, default="train", help="Split label for selection metadata (default: train).")
    parser.add_argument(
        "--allow-mtime-mismatch",
        action="store_true",
        help="Allow profile zarr_mtime_ns mismatch against current filesystem mtime.",
    )
    parser.add_argument(
        "--allow-detection-type-mismatch",
        action="store_true",
        help="Allow mismatch between manifest detection_source_type and profile detection_type.",
    )
    parser.add_argument(
        "--subject-lineage-policy",
        choices=("warn", "require"),
        default="warn",
        help=(
            "Subject-lineage coverage policy for manifest datasets using "
            "recording_subject_overview (default: warn)."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot PNG generation for the aggregated data card.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Optional output directory for data-card plots (default: <card_stem>.plots).",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        help="Optional filename prefix for data-card plots (default: set_id).",
    )
    parser.add_argument(
        "--plot-heatmap-bin-factor",
        type=int,
        default=2,
        help="Coarsening factor for center heatmap bins in plot PNGs (default: 2).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute card but do not write output JSON.")
    parser.add_argument("--json", action="store_true", help="Print card JSON to stdout.")
    args = parser.parse_args(argv)
    if args.plot_heatmap_bin_factor < 1:
        parser.error("--plot-heatmap-bin-factor must be >= 1.")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Training data card aggregation failed: manifest not found: {manifest_path}")
        return 1

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        manifest = _load_manifest(manifest_path)
        card = _build_detection_training_data_card(
            registry=registry,
            manifest=manifest,
            split=str(args.split),
            allow_mtime_mismatch=bool(args.allow_mtime_mismatch),
            allow_detection_type_mismatch=bool(args.allow_detection_type_mismatch),
            subject_lineage_policy=str(args.subject_lineage_policy),
        )
    except Exception as exc:
        print(f"Training data card aggregation failed: {exc}")
        return 1
    finally:
        registry.close()

    output_path = Path(args.output) if args.output is not None else _default_output_path(
        manifest_path,
        _normalize_text(manifest.get("set_id")),
    )
    if args.json:
        print(json.dumps(card, indent=2))
    if args.dry_run:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else _default_plot_dir(output_path)
        print(
            "Detection training data card: mode=dry-run "
            f"datasets={card['selection']['dataset_count']} output={output_path} "
            f"plots={'off' if args.no_plots else plot_dir} "
            f"heatmap_bin_factor={int(args.plot_heatmap_bin_factor)}"
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "Detection training data card: mode=apply "
        f"datasets={card['selection']['dataset_count']} output={output_path}"
    )
    if not args.no_plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else _default_plot_dir(output_path)
        prefix = _normalize_text(args.plot_prefix) or _normalize_text(card.get("set_id")) or output_path.stem
        try:
            generated = plot_data_card.generate_detection_training_data_card_plots(
                card_payload=card,
                output_dir=plot_dir,
                prefix=str(prefix),
                heatmap_bin_factor=int(args.plot_heatmap_bin_factor),
            )
            print(
                "Detection training data-card plots: mode=apply "
                f"generated={len(generated)} output_dir={plot_dir} "
                f"heatmap_bin_factor={int(args.plot_heatmap_bin_factor)}"
            )
        except Exception as exc:
            print(f"Warning: detection training data-card plot generation skipped: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
