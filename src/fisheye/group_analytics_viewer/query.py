"""Parquet-backed query helpers for group analytics viewer endpoints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fisheye.analytics_exports.contracts import (
    CHASER_BOUT_EVENTS_TABLE,
    CHASER_BOUT_HISTOGRAM_TABLE,
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
    CHASER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_IBI_HISTOGRAM_TABLE,
    CHASER_SPATIAL_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
    DESCRIPTIVE_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    STATISTICS_TABLE,
)
from fisheye.analytics_exports.publication import (
    export_manifest_directory,
    export_manifest_path,
    manifest_selected_part_files,
    manifest_selected_part_files_from_payload,
    validate_publication_envelope,
)
from fisheye.analytics_exports.validation import (
    ExportValidationError,
    validate_export_payload,
)
from fisheye.group_statistics.legacy_arrow import (
    validate_legacy_group_statistics_payload,
)
from fisheye.group_statistics.goodcopbadcop import (
    validate_goodcopbadcop_result_rows,
)
from fisheye.group_statistics.paired import bootstrap_median_ci, wilcoxon_signed_rank_p_value

from .catalog import discover_export_catalog, select_export_run_id
from .models import HealthReport


SPATIAL_TABLE = CHASER_SPATIAL_TABLE
CHASER_SUMMARY_TABLE = CHASER_DISTANCE_SUMMARY_TABLE
EPOCH_BEHAVIOR_TABLE = CHASER_EPOCH_BEHAVIOR_TABLE
EPOCH_BOUT_DISTRIBUTION_TABLE = CHASER_BOUT_EVENTS_TABLE
EPOCH_BOUT_HISTOGRAM_TABLE = CHASER_BOUT_HISTOGRAM_TABLE
EPOCH_INTER_BOUT_INTERVAL_HISTOGRAM_TABLE = CHASER_IBI_HISTOGRAM_TABLE
EPOCH_CENTER_DISTANCE_HISTOGRAM_TABLE = CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE
SPEED_DISTANCE_TABLE = CHASER_SPEED_DISTANCE_TABLE
CHASER_HISTOGRAM_TABLE = CHASER_DISTANCE_HISTOGRAM_TABLE
CRA_SUMMARY_TABLE = CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE
CRA_OBJECT_PHASE_TABLE = CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE
CRA_QUADRANT_OCCUPANCY_TABLE = CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE
CRA_NEAR_FIELD_SUMMARY_TABLE = CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE
CRA_NEAR_FIELD_OBJECT_PHASE_TABLE = CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE
CRA_NEAR_FIELD_RADIAL_TABLE = CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE
CRA_NEAR_FIELD_CDF_TABLE = CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE
EGOCENTRIC_SUMMARY_TABLE = CHASER_EGOCENTRIC_SUMMARY_TABLE
EGOCENTRIC_HISTOGRAM_TABLE = CHASER_EGOCENTRIC_HISTOGRAM_TABLE
POSITION_OCCUPANCY_TABLE = POSITION_OCCUPANCY_HISTOGRAM_TABLE
CORE_CHASER_TABLES = (
    SPATIAL_TABLE,
    CHASER_SUMMARY_TABLE,
    CHASER_HISTOGRAM_TABLE,
    CRA_SUMMARY_TABLE,
    CRA_OBJECT_PHASE_TABLE,
    EGOCENTRIC_SUMMARY_TABLE,
    EGOCENTRIC_HISTOGRAM_TABLE,
)
OPTIONAL_CHASER_TABLES = (
    POSITION_OCCUPANCY_TABLE,
    EPOCH_BEHAVIOR_TABLE,
    EPOCH_BOUT_DISTRIBUTION_TABLE,
    EPOCH_BOUT_HISTOGRAM_TABLE,
    EPOCH_INTER_BOUT_INTERVAL_HISTOGRAM_TABLE,
    EPOCH_CENTER_DISTANCE_HISTOGRAM_TABLE,
    SPEED_DISTANCE_TABLE,
    CRA_QUADRANT_OCCUPANCY_TABLE,
    CRA_NEAR_FIELD_SUMMARY_TABLE,
    CRA_NEAR_FIELD_OBJECT_PHASE_TABLE,
    CRA_NEAR_FIELD_RADIAL_TABLE,
    CRA_NEAR_FIELD_CDF_TABLE,
)
CHASER_TABLES = CORE_CHASER_TABLES + OPTIONAL_CHASER_TABLES

SPATIAL_METRICS = {
    "time_s": "Time (s)",
    "fraction_of_epoch": "Fraction of epoch",
    "fraction_of_detected": "Fraction of detected frames",
    "frame_count": "Frame count",
}
CHASER_METRICS = {
    "mean_distance_mm": "Mean distance (mm)",
    "min_distance_mm": "Minimum distance (mm)",
    "p05_distance_mm": "P05 distance (mm)",
    "p50_distance_mm": "Median distance (mm)",
    "p95_distance_mm": "P95 distance (mm)",
    "fraction_within_threshold": "Fraction within threshold",
}
EPOCH_SPEED_METRICS = {
    "mean_speed_mm_s": "Mean speed (mm/s)",
    "median_speed_mm_s": "Median speed (mm/s)",
    "p95_speed_mm_s": "P95 speed (mm/s)",
    "max_speed_mm_s": "Maximum speed (mm/s)",
    "total_path_mm": "Total path (mm)",
    "tracking_dropout_fraction": "Tracking dropout fraction",
}
EPOCH_BEHAVIOR_METRICS = {
    **EPOCH_SPEED_METRICS,
    "bout_rate_per_min": "Bout rate (/min)",
    "bout_count": "Bout count",
    "mean_bout_duration_s": "Mean bout duration (s)",
    "median_bout_duration_s": "Median bout duration (s)",
    "mean_bout_path_length_mm": "Mean bout distance (mm)",
    "median_bout_path_length_mm": "Median bout distance (mm)",
    "mean_bout_net_heading_change_deg": "Mean net bout heading change (deg)",
    "median_bout_net_heading_change_deg": "Median net bout heading change (deg)",
    "mean_abs_bout_net_heading_change_deg": "Mean abs bout heading change (deg)",
    "median_abs_bout_net_heading_change_deg": "Median abs bout heading change (deg)",
    "mean_bout_heading_path_deg": "Mean bout heading path (deg)",
    "median_bout_heading_path_deg": "Median bout heading path (deg)",
    "inter_bout_interval_count": "Inter-bout interval count",
    "mean_inter_bout_interval_s": "Mean inter-bout interval (s)",
    "median_inter_bout_interval_s": "Median inter-bout interval (s)",
    "inter_bout_interval_rate_per_min": "Inter-bout interval rate (/min)",
    "mean_distance_from_arena_center_mm": "Mean distance from arena center (mm)",
    "median_distance_from_arena_center_mm": "Median distance from arena center (mm)",
    "wall_fraction": "Wall fraction",
    "wall_time_s": "Wall time (s)",
}
EPOCH_BOUT_HISTOGRAM_METRICS = {
    "bout_path_length_mm": "Bout distance (mm)",
    "bout_duration_s": "Bout duration (s)",
    "bout_net_heading_change_deg": "Net bout heading change (deg)",
    "abs_bout_net_heading_change_deg": "Abs net bout heading change (deg)",
    "bout_heading_path_deg": "Bout heading path (deg)",
}
EPOCH_INTER_BOUT_INTERVAL_HISTOGRAM_METRICS = {
    "inter_bout_interval_s": "Inter-bout interval (s)",
}
EGOCENTRIC_METRICS = {
    "mean_alignment_cos": "Mean heading alignment",
    "mean_lateral_sin": "Mean lateral component",
    "circular_mean_bearing_deg": "Circular mean bearing (deg)",
    "circular_resultant_length": "Circular resultant length",
    "fraction_front_45": "Fraction front +/-45 deg",
    "fraction_lateral_45": "Fraction lateral",
    "fraction_behind_45": "Fraction behind +/-45 deg",
}
CRA_OBJECT_PHASE_METRICS = {
    "median_distance_mm": "Median distance (mm)",
    "mean_distance_mm": "Mean distance (mm)",
    "occupancy_fraction": "Object-quadrant occupancy",
    "occupancy_fraction_of_epoch": "Object-quadrant occupancy of epoch",
    "tracking_dropout_fraction": "Tracking dropout fraction",
    "object_max_drift_mm": "Object max drift (mm)",
}
CRA_SUMMARY_METRICS = {
    "delta_agg": "Aggressive distance delta (post - pre)",
    "delta_inert": "Inert distance delta (post - pre)",
    "specificity_distance": "Distance specificity",
    "delta_occ_agg": "Aggressive occupancy delta (post - pre)",
    "delta_occ_inert": "Inert occupancy delta (post - pre)",
    "specificity_occupancy": "Occupancy specificity",
}
CRA_NEAR_FIELD_OBJECT_PHASE_METRICS = {
    "approach_p05_mm": "Close approach p05 (mm)",
    "approach_p10_mm": "Close approach p10 (mm)",
    "near_zone_occupancy_fraction": "Near-zone occupancy",
    "near_zone_occupancy_fraction_of_epoch": "Near-zone occupancy of epoch",
    "near_zone_density_per_mm2": "Near-zone density",
    "near_zone_entry_rate_per_min": "Near-zone entries per min",
    "near_zone_visit_median_dwell_s": "Near-zone median visit dwell (s)",
    "tracking_dropout_fraction": "Tracking dropout fraction",
}
CRA_NEAR_FIELD_SUMMARY_METRICS = {
    "approach_p05_delta_agg": "Aggressive p05 approach delta",
    "approach_p05_delta_inert": "Inert p05 approach delta",
    "approach_p05_specificity": "P05 approach specificity",
    "approach_p10_delta_agg": "Aggressive p10 approach delta",
    "approach_p10_delta_inert": "Inert p10 approach delta",
    "approach_p10_specificity": "P10 approach specificity",
    "nearzone_occ_delta_agg": "Aggressive near-zone occupancy delta",
    "nearzone_occ_delta_inert": "Inert near-zone occupancy delta",
    "nearzone_occ_specificity": "Near-zone occupancy specificity",
    "nearzone_entry_rate_delta_agg": "Aggressive entry-rate delta",
    "nearzone_entry_rate_delta_inert": "Inert entry-rate delta",
    "nearzone_entry_rate_specificity": "Entry-rate specificity",
    "thigmotaxis_frac_pre": "Thigmotaxis pre fraction",
    "thigmotaxis_frac_post": "Thigmotaxis post fraction",
}


@dataclass(frozen=True)
class ViewerContext:
    export_root: Path
    export_run_id: str
    stats_run_id: str | None = None
    allow_legacy_statistics: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _sort_int(value: Any, default: int = 999) -> int:
    parsed = _safe_int(value)
    return default if parsed is None else parsed


def _values(rows: Iterable[Mapping[str, Any]], metric: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _safe_float(row.get(metric))
        if value is not None:
            out.append(value)
    return out


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "sum": None,
            "mean": None,
            "median": None,
            "std_dev": None,
            "sem": None,
            "min": None,
            "max": None,
        }
    total = float(sum(values))
    n = int(len(values))
    mean = total / n
    std_dev = None
    sem = None
    if n > 1:
        variance = sum((float(value) - mean) ** 2 for value in values) / float(n - 1)
        std_dev = math.sqrt(variance)
        sem = std_dev / math.sqrt(float(n))
    return {
        "n": n,
        "sum": _round(total),
        "mean": _round(mean),
        "median": _round(float(median(values))),
        "std_dev": _round(std_dev),
        "sem": _round(sem),
        "min": _round(float(min(values))),
        "max": _round(float(max(values))),
    }


def _json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _manifest_path(export_root: Path, export_run_id: str) -> Path:
    return export_manifest_path(export_root, export_run_id)


def _validate_path_component(value: str, *, label: str) -> str:
    value = str(value).strip()
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_export_run_id(export_root: Path, export_run_id: str) -> str:
    if export_run_id != "latest":
        _validate_path_component(export_run_id, label="export run ID")
    catalog = discover_export_catalog(export_root)
    return select_export_run_id(catalog, export_run_id)


def build_context(
    *,
    export_root: Path,
    export_run_id: str,
    stats_run_id: str | None = None,
    allow_legacy_statistics: bool = False,
) -> ViewerContext:
    resolved_root = export_root.expanduser().resolve()
    resolved_run = resolve_export_run_id(resolved_root, export_run_id)
    manifest = _manifest_path(resolved_root, resolved_run)
    if not manifest.is_file():
        raise FileNotFoundError(f"Export manifest not found: {manifest}")
    resolved_stats = str(stats_run_id).strip() if stats_run_id else None
    if resolved_stats in {"", "auto", "latest"}:
        resolved_stats = None
    if resolved_stats is not None:
        resolved_stats = _validate_path_component(resolved_stats, label="statistics run ID")
    return ViewerContext(
        export_root=resolved_root,
        export_run_id=resolved_run,
        stats_run_id=resolved_stats,
        allow_legacy_statistics=bool(allow_legacy_statistics),
    )


def table_dir(context: ViewerContext, table_name: str, *, export_run_id: str | None = None) -> Path:
    safe_table_name = _validate_path_component(table_name, label="table name")
    safe_run_id = _validate_path_component(
        export_run_id or context.export_run_id,
        label="export run ID",
    )
    return context.export_root / "v1" / safe_table_name / f"export_run_id={safe_run_id}"


def parquet_files(context: ViewerContext, table_name: str, *, export_run_id: str | None = None) -> tuple[Path, ...]:
    safe_table_name = _validate_path_component(table_name, label="table name")
    safe_run_id = _validate_path_component(
        export_run_id or context.export_run_id,
        label="export run ID",
    )
    files = manifest_selected_part_files(
        context.export_root,
        safe_run_id,
        safe_table_name,
    )
    if not files:
        raise FileNotFoundError(
            f"Manifest selects no parquet parts for {safe_table_name!r} in {safe_run_id!r}"
        )
    return files


@lru_cache(maxsize=32)
def _load_table_rows(export_root: str, export_run_id: str, table_name: str) -> tuple[dict[str, Any], ...]:
    import polars as pl

    context = ViewerContext(export_root=Path(export_root), export_run_id=export_run_id)
    files = [str(path) for path in parquet_files(context, table_name)]
    rows = (
        pl.scan_parquet(files, hive_partitioning=False)
        .collect(engine="streaming")
        .to_dicts()
    )
    return tuple(rows)


def load_table_rows(context: ViewerContext, table_name: str) -> list[dict[str, Any]]:
    return [dict(row) for row in _load_table_rows(str(context.export_root), context.export_run_id, table_name)]


def load_optional_table_rows(context: ViewerContext, table_name: str) -> list[dict[str, Any]]:
    try:
        return load_table_rows(context, table_name)
    except FileNotFoundError:
        return []


def _load_table_rows_for_run(context: ViewerContext, table_name: str, export_run_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in _load_table_rows(str(context.export_root), export_run_id, table_name)]


def _table_schema(context: ViewerContext, table_name: str, *, export_run_id: str | None = None) -> list[dict[str, str]]:
    import polars as pl

    first = parquet_files(context, table_name, export_run_id=export_run_id)[0]
    schema = pl.read_parquet_schema(first)
    return [{"name": name, "type": str(dtype)} for name, dtype in schema.items()]


def _validate_statistics_manifest(
    context: ViewerContext,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[str, bool]:
    stats_run_id = manifest_path.name.removeprefix("export_run_id=").removesuffix(
        ".json"
    )
    if manifest.get("export_run_id") != stats_run_id:
        raise ValueError("Statistics manifest identity does not match its filename")
    if (
        manifest.get("schema_id") != EXPORT_SCHEMA_ID
        or manifest.get("schema_version") != EXPORT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Statistics manifest is not {EXPORT_SCHEMA_ID} version "
            f"{EXPORT_SCHEMA_VERSION}: {manifest_path}"
        )
    if manifest.get("source_export_run_id") != context.export_run_id:
        raise ValueError(
            "Statistics manifest source_export_run_id does not match viewed export: "
            f"{manifest.get('source_export_run_id')} != {context.export_run_id}"
        )
    source_manifest = _manifest_path(context.export_root, context.export_run_id)
    source_manifest_sha256 = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    if manifest.get("source_export_manifest_sha256") != source_manifest_sha256:
        raise ValueError(
            "Statistics manifest source_export_manifest_sha256 does not match "
            "the selected base export manifest"
        )
    outputs = manifest.get("output_tables")
    row_counts = manifest.get("row_counts_by_table")
    has_stats_table = (
        isinstance(outputs, list)
        and STATISTICS_TABLE in {str(item) for item in outputs}
    ) or (
        isinstance(row_counts, Mapping)
        and STATISTICS_TABLE in row_counts
    )
    if not has_stats_table:
        raise ValueError(f"Statistics manifest does not declare {STATISTICS_TABLE}")
    is_current_exact = True
    try:
        validate_export_payload(context.export_root, stats_run_id, manifest)
    except ExportValidationError:
        if not context.allow_legacy_statistics:
            raise
        is_current_exact = False
        try:
            validate_legacy_group_statistics_payload(
                context.export_root,
                stats_run_id,
                manifest,
            )
        except ValueError as legacy_error:
            raise ValueError(
                "Statistics publication satisfies neither the current exact "
                "contract nor the explicit legacy inferred contract"
            ) from legacy_error
    validate_publication_envelope(manifest)
    selected = manifest_selected_part_files_from_payload(
        context.export_root,
        manifest,
        STATISTICS_TABLE,
    )
    expected_rows = _safe_int(row_counts.get(STATISTICS_TABLE)) if isinstance(
        row_counts, Mapping
    ) else None
    if expected_rows and not selected:
        raise ValueError("Statistics publication selects no summary part files")
    if any(not path.is_file() for path in selected):
        raise ValueError("Statistics publication selects a missing summary part")
    if is_current_exact:
        summary_rows = _load_table_rows_for_run(
            context,
            STATISTICS_TABLE,
            stats_run_id,
        )
        validate_goodcopbadcop_result_rows(
            STATISTICS_TABLE,
            summary_rows,
            manifest,
        )
        if (
            isinstance(outputs, list)
            and DESCRIPTIVE_TABLE in {str(item) for item in outputs}
        ):
            descriptive_rows = _load_table_rows_for_run(
                context,
                DESCRIPTIVE_TABLE,
                stats_run_id,
            )
            validate_goodcopbadcop_result_rows(
                DESCRIPTIVE_TABLE,
                descriptive_rows,
                manifest,
            )
    return stats_run_id, is_current_exact


def _stats_manifest_candidates(
    context: ViewerContext,
) -> list[tuple[str, dict[str, Any], bool]]:
    manifest_dir = export_manifest_directory(context.export_root)
    candidates: list[tuple[str, dict[str, Any], bool]] = []
    for manifest_path in sorted(manifest_dir.glob("export_run_id=*.json")):
        if not _is_within(manifest_path.resolve(), context.export_root.resolve()):
            continue
        try:
            manifest = _json_file(manifest_path)
        except Exception:
            continue
        try:
            stats_run_id, is_current_exact = _validate_statistics_manifest(
                context,
                manifest_path,
                manifest,
            )
        except (OSError, ValueError):
            continue
        candidates.append((stats_run_id, manifest, is_current_exact))
    candidates.sort(
        key=lambda item: (
            item[2],
            str(item[1].get("created_at_utc") or ""),
            item[0],
        )
    )
    return candidates


def resolve_statistics_run_id(context: ViewerContext) -> str | None:
    if context.stats_run_id:
        manifest = _manifest_path(context.export_root, context.stats_run_id)
        if not manifest.is_file():
            raise FileNotFoundError(f"Statistics manifest not found: {manifest}")
        payload = _json_file(manifest)
        return _validate_statistics_manifest(context, manifest, payload)[0]
    candidates = _stats_manifest_candidates(context)
    return candidates[-1][0] if candidates else None


def build_health_report(context: ViewerContext) -> HealthReport:
    details: dict[str, Any] = {"tables": {}}
    ok = True
    messages: list[str] = []
    manifest = _manifest_path(context.export_root, context.export_run_id)
    if not manifest.is_file():
        ok = False
        messages.append(f"missing manifest: {manifest}")
    for table_name in CHASER_TABLES:
        try:
            files = parquet_files(context, table_name)
            details["tables"][table_name] = {
                "path": str(files[0].parent),
                "part_count": len(files),
            }
        except Exception as exc:
            if table_name not in OPTIONAL_CHASER_TABLES:
                ok = False
            details["tables"][table_name] = {"error": str(exc)}
    try:
        stats_run_id = resolve_statistics_run_id(context)
        details["statistics"] = {
            "available": stats_run_id is not None,
            "stats_run_id": stats_run_id,
        }
    except Exception as exc:
        ok = False
        details["statistics"] = {"available": False, "error": str(exc)}
    return HealthReport(
        ok=ok,
        checked_utc=_utc_now(),
        export_run_id=context.export_run_id,
        export_root=str(context.export_root),
        message="ok" if ok else "; ".join(messages) or "viewer export check failed",
        details=details,
    )


def query_export_summary(context: ViewerContext) -> dict[str, Any]:
    manifest_path = _manifest_path(context.export_root, context.export_run_id)
    manifest = _json_file(manifest_path)
    row_counts = manifest.get("row_counts_by_table")
    if not isinstance(row_counts, Mapping):
        row_counts = {}
    tables: list[dict[str, Any]] = []
    for table_name in CHASER_TABLES:
        try:
            selected_parts = parquet_files(context, table_name)
        except FileNotFoundError:
            selected_parts = ()
        tables.append(
            {
                "table_name": table_name,
                "row_count": _safe_int(row_counts.get(table_name)),
                "part_count": len(selected_parts),
                "table_path": (
                    str(selected_parts[0].parent) if selected_parts else None
                ),
            }
        )

    diagnostics = manifest.get("diagnostics")
    if not isinstance(diagnostics, list):
        diagnostics = []
    collection = manifest.get("collection_manifest")
    stats_run_id = resolve_statistics_run_id(context)
    stats_summary: dict[str, Any] = {"available": stats_run_id is not None, "stats_run_id": stats_run_id}
    if stats_run_id:
        stats_manifest_path = _manifest_path(context.export_root, stats_run_id)
        stats_manifest = _json_file(stats_manifest_path)
        stats_row_counts = stats_manifest.get("row_counts_by_table")
        if not isinstance(stats_row_counts, Mapping):
            stats_row_counts = {}
        stats_summary.update(
            {
                "manifest_path": str(stats_manifest_path),
                "created_at_utc": stats_manifest.get("created_at_utc"),
                "row_count": _safe_int(stats_row_counts.get(STATISTICS_TABLE)),
                "descriptive_row_count": _safe_int(stats_row_counts.get(DESCRIPTIVE_TABLE)),
                "status_counts": stats_manifest.get("status_counts"),
            }
        )
    return {
        "export_run_id": context.export_run_id,
        "manifest_path": str(manifest_path),
        "created_at_utc": manifest.get("created_at_utc"),
        "source_recording_count": _safe_int(manifest.get("source_recording_count")),
        "tables": tables,
        "row_counts_by_table": dict(row_counts),
        "diagnostics_count": len(diagnostics),
        "collection": collection if isinstance(collection, Mapping) else None,
        "palette_git_commit": manifest.get("palette_git_commit"),
        "palette_git_dirty": manifest.get("palette_git_dirty"),
        "statistics": stats_summary,
    }


def _window_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (
        _safe_int(row.get("window_index")) if _safe_int(row.get("window_index")) is not None else 999,
        str(row.get("window_label") or ""),
    )


def canonical_condition(label: Any) -> str | None:
    text = str(label or "").strip().lower().replace("-", "_")
    if not text:
        return None
    if text.startswith("pre"):
        return "pre"
    if text.startswith("train"):
        return "training"
    if text.startswith("post"):
        return "post"
    return text


def query_options(context: ViewerContext) -> dict[str, Any]:
    spatial = load_optional_table_rows(context, SPATIAL_TABLE)
    chaser = load_optional_table_rows(context, CHASER_SUMMARY_TABLE)
    speed = load_optional_table_rows(context, EPOCH_BEHAVIOR_TABLE)
    bout_histogram = load_optional_table_rows(context, EPOCH_BOUT_HISTOGRAM_TABLE)
    ibi_histogram = load_optional_table_rows(context, EPOCH_INTER_BOUT_INTERVAL_HISTOGRAM_TABLE)
    cra_object_phase = load_optional_table_rows(context, CRA_OBJECT_PHASE_TABLE)
    cra_near_field_object_phase = load_optional_table_rows(context, CRA_NEAR_FIELD_OBJECT_PHASE_TABLE)
    egocentric = load_optional_table_rows(context, EGOCENTRIC_SUMMARY_TABLE)
    windows = sorted(
        {
            (
                _safe_int(row.get("window_index")) if _safe_int(row.get("window_index")) is not None else 999,
                str(row.get("window_label") or ""),
            )
            for row in spatial + chaser + speed + bout_histogram + ibi_histogram + egocentric
            if row.get("window_label") is not None
        }
    )
    cra_phases = sorted(
        {
            (
                _safe_int(row.get("phase_axis_index")) if _safe_int(row.get("phase_axis_index")) is not None else 999,
                str(row.get("phase_label") or ""),
            )
            for row in cra_object_phase
            if row.get("phase_label") is not None
        }
    )
    zone_sets = sorted({str(row.get("zone_set_id")) for row in spatial if row.get("zone_set_id")})
    zones = sorted(
        {
            (
                _safe_int(row.get("display_order")) if _safe_int(row.get("display_order")) is not None else 999,
                str(row.get("zone_id") or ""),
                str(row.get("zone_label") or row.get("zone_id") or ""),
            )
            for row in spatial
            if row.get("zone_id") is not None
        }
    )
    chasers = sorted(
        {
            _safe_int(row.get("chaser_index"))
            for row in chaser + egocentric
            if _safe_int(row.get("chaser_index")) is not None
        }
    )
    cra_object_roles = sorted(
        {
            str(row.get("object_role"))
            for row in cra_object_phase + cra_near_field_object_phase
            if row.get("object_role") is not None
        }
    )
    return {
        "windows": [{"window_index": idx, "window_label": label} for idx, label in windows],
        "cra_phases": [{"phase_axis_index": idx, "phase_label": label} for idx, label in cra_phases],
        "zone_sets": zone_sets,
        "zones": [
            {"display_order": order, "zone_id": zone_id, "zone_label": label}
            for order, zone_id, label in zones
        ],
        "chasers": chasers,
        "cra_object_roles": cra_object_roles,
        "spatial_metrics": [{"metric": key, "label": label} for key, label in SPATIAL_METRICS.items()],
        "chaser_metrics": [{"metric": key, "label": label} for key, label in CHASER_METRICS.items()],
        "epoch_speed_metrics": [{"metric": key, "label": label} for key, label in EPOCH_BEHAVIOR_METRICS.items()],
        "epoch_bout_histogram_metrics": [
            {"metric": key, "label": label} for key, label in EPOCH_BOUT_HISTOGRAM_METRICS.items()
        ],
        "epoch_inter_bout_interval_histogram_metrics": [
            {"metric": key, "label": label}
            for key, label in EPOCH_INTER_BOUT_INTERVAL_HISTOGRAM_METRICS.items()
        ],
        "cra_object_phase_metrics": [{"metric": key, "label": label} for key, label in CRA_OBJECT_PHASE_METRICS.items()],
        "cra_summary_metrics": [{"metric": key, "label": label} for key, label in CRA_SUMMARY_METRICS.items()],
        "cra_near_field_object_phase_metrics": [
            {"metric": key, "label": label}
            for key, label in CRA_NEAR_FIELD_OBJECT_PHASE_METRICS.items()
        ],
        "cra_near_field_summary_metrics": [
            {"metric": key, "label": label}
            for key, label in CRA_NEAR_FIELD_SUMMARY_METRICS.items()
        ],
        "egocentric_metrics": [{"metric": key, "label": label} for key, label in EGOCENTRIC_METRICS.items()],
    }


def _group_label(group_key_json: Any) -> str:
    if not group_key_json:
        return "all"
    try:
        group_key = json.loads(str(group_key_json))
    except (TypeError, ValueError):
        return str(group_key_json)
    if not isinstance(group_key, Mapping):
        return str(group_key)
    if group_key.get("behavior_class"):
        return str(group_key.get("behavior_class"))
    if "chaser_index" in group_key:
        return f"chaser {group_key.get('chaser_index')}"
    if group_key.get("zone_label"):
        return str(group_key.get("zone_label"))
    if group_key.get("zone_id"):
        return str(group_key.get("zone_id"))
    parts = [f"{key}={value}" for key, value in sorted(group_key.items())]
    return ", ".join(parts) if parts else "all"


def query_group_statistics(
    context: ViewerContext,
    *,
    metric_family: str | None = None,
    metric_name: str | None = None,
    contrast_name: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    stats_run_id = resolve_statistics_run_id(context)
    if stats_run_id is None:
        return {
            "available": False,
            "stats_run_id": None,
            "manifest_path": None,
            "row_count": 0,
            "rows": [],
            "message": "No statistics export found for this source export run.",
        }
    manifest_path = _manifest_path(context.export_root, stats_run_id)
    manifest = _json_file(manifest_path)
    rows = _load_table_rows_for_run(context, STATISTICS_TABLE, stats_run_id)
    if metric_family:
        rows = [row for row in rows if row.get("metric_family") == metric_family]
    if metric_name:
        rows = [row for row in rows if row.get("metric_name") == metric_name]
    if contrast_name:
        rows = [row for row in rows if row.get("contrast_name") == contrast_name]
    if status:
        rows = [row for row in rows if row.get("status") == status]

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out_rows.append(
            {
                "metric_family": row.get("metric_family"),
                "metric_name": row.get("metric_name"),
                "metric_unit": row.get("metric_unit"),
                "contrast_name": row.get("contrast_name"),
                "condition_a": row.get("condition_a"),
                "condition_b": row.get("condition_b"),
                "group": _group_label(row.get("group_key_json")),
                "group_key_json": row.get("group_key_json"),
                "unit_count": _safe_int(row.get("unit_count")),
                "paired_unit_count": _safe_int(row.get("paired_unit_count")),
                "excluded_unit_count": _safe_int(row.get("excluded_unit_count")),
                "mean_a": _round(_safe_float(row.get("mean_a"))),
                "mean_b": _round(_safe_float(row.get("mean_b"))),
                "mean_difference": _round(_safe_float(row.get("mean_difference"))),
                "median_difference": _round(_safe_float(row.get("median_difference"))),
                "std_difference": _round(_safe_float(row.get("std_difference"))),
                "effect_size": _round(_safe_float(row.get("effect_size"))),
                "effect_size_kind": row.get("effect_size_kind"),
                "ci_low": _round(_safe_float(row.get("ci_low"))),
                "ci_high": _round(_safe_float(row.get("ci_high"))),
                "ci_estimand": row.get("ci_estimand"),
                "p_value": _round(_safe_float(row.get("p_value"))),
                "q_value": _round(_safe_float(row.get("q_value"))),
                "test_method": row.get("test_method"),
                "missing_policy": row.get("missing_policy"),
                "parameters_json": row.get("parameters_json"),
                "status": row.get("status"),
                "skip_reason": row.get("skip_reason"),
                "primary": bool(row.get("primary")),
                "exploratory": bool(row.get("exploratory")),
                "stat_result_id": row.get("stat_result_id"),
            }
        )
    out_rows.sort(
        key=lambda row: (
            str(row.get("metric_family") or ""),
            str(row.get("metric_name") or ""),
            str(row.get("contrast_name") or ""),
            str(row.get("group") or ""),
        )
    )
    row_counts = manifest.get("row_counts_by_table")
    if not isinstance(row_counts, Mapping):
        row_counts = {}
    return {
        "available": True,
        "stats_run_id": stats_run_id,
        "manifest_path": str(manifest_path),
        "created_at_utc": manifest.get("created_at_utc"),
        "source_export_run_id": manifest.get("source_export_run_id"),
        "configured_row_count": _safe_int(row_counts.get(STATISTICS_TABLE)),
        "row_count": len(out_rows),
        "status_counts": manifest.get("status_counts"),
        "rows": out_rows,
    }


def _load_descriptive_rows(context: ViewerContext) -> tuple[str | None, list[dict[str, Any]]]:
    stats_run_id = resolve_statistics_run_id(context)
    if stats_run_id is None:
        return None, []
    try:
        return stats_run_id, _load_table_rows_for_run(context, DESCRIPTIVE_TABLE, stats_run_id)
    except FileNotFoundError:
        return stats_run_id, []


def _parse_group_key(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _group_key_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left:
        left_value = left.get(key)
        right_value = right.get(key)
        if _safe_int(left_value) is not None and _safe_int(right_value) is not None:
            if _safe_int(left_value) != _safe_int(right_value):
                return False
        elif str(left_value) != str(right_value):
            return False
    return True


def _descriptive_row_for(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_family: str,
    metric_name: str,
    condition_name: str,
    group_key: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("metric_family") != metric_family:
            continue
        if row.get("metric_name") != metric_name:
            continue
        if row.get("condition_name") != condition_name:
            continue
        if _group_key_equal(_parse_group_key(row.get("group_key_json")), group_key):
            return row
    return None


def _summary_from_descriptive_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n": _safe_int(row.get("unit_count")) or 0,
        "sum": _round(_safe_float(row.get("sum"))),
        "mean": _round(_safe_float(row.get("mean"))),
        "median": _round(_safe_float(row.get("median"))),
        "std_dev": _round(_safe_float(row.get("std_dev"))),
        "sem": _round(_safe_float(row.get("sem"))),
        "min": _round(_safe_float(row.get("min"))),
        "max": _round(_safe_float(row.get("max"))),
    }


def query_spatial_occupancy(
    context: ViewerContext,
    *,
    metric: str = "time_s",
    value_mode: str = "auto",
    zone_set_id: str | None = None,
    include_recordings: bool = False,
) -> dict[str, Any]:
    if metric not in SPATIAL_METRICS:
        raise ValueError(f"Unsupported spatial metric: {metric}")
    if value_mode not in {"auto", "total", "mean"}:
        raise ValueError("value_mode must be one of: auto, total, mean")
    effective_mode = "total" if value_mode == "auto" and metric in {"time_s", "frame_count"} else value_mode
    if effective_mode == "auto":
        effective_mode = "mean"

    rows = load_table_rows(context, SPATIAL_TABLE)
    if zone_set_id:
        rows = [row for row in rows if row.get("zone_set_id") == zone_set_id]
    _stats_run_id, descriptive_rows = _load_descriptive_rows(context)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            row.get("zone_set_id"),
            _safe_int(row.get("display_order")),
            row.get("zone_id"),
            row.get("zone_label"),
        )
        grouped[key].append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        window_index, window_label, group_zone_set, display_order, zone_id, zone_label = key
        condition_name = canonical_condition(window_label)
        group_key = {
            "zone_set_id": group_zone_set,
            "zone_id": zone_id,
            "zone_label": zone_label,
        }
        descriptive = (
            _descriptive_row_for(
                descriptive_rows,
                metric_family="spatial_occupancy",
                metric_name=metric,
                condition_name=condition_name or "",
                group_key=group_key,
            )
            if condition_name
            else None
        )
        if descriptive is not None:
            stats = _summary_from_descriptive_row(descriptive)
            summary_source = "persisted_descriptive_summary"
        else:
            values = _values(group_rows, metric)
            stats = _summary(values)
            summary_source = "computed_from_export_rows"
        recording_ids = sorted(
            {str(row.get("recording_id")) for row in group_rows if row.get("recording_id")}
        )
        value = stats["sum"] if effective_mode == "total" else stats["mean"]
        out_rows.append(
            {
                "window_index": window_index,
                "window_label": window_label,
                "zone_set_id": group_zone_set,
                "display_order": display_order,
                "zone_id": zone_id,
                "zone_label": zone_label,
                "value": value,
                "value_mode": effective_mode,
                "recording_count": stats["n"] if descriptive is not None else len(recording_ids),
                "summary_source": summary_source,
                **stats,
            }
        )
    out_rows.sort(
        key=lambda row: (
            _sort_int(row.get("window_index")),
            _sort_int(row.get("display_order")),
            str(row.get("zone_id") or ""),
        )
    )

    response: dict[str, Any] = {
        "metric": metric,
        "metric_label": SPATIAL_METRICS[metric],
        "value_mode": effective_mode,
        "summary_source": (
            "persisted_descriptive_summary" if any(row.get("summary_source") == "persisted_descriptive_summary" for row in out_rows)
            else "computed_from_export_rows"
        ),
        "rows": out_rows,
    }
    if include_recordings:
        response["per_recording"] = [
            {
                "recording_id": row.get("recording_id"),
                "window_index": _safe_int(row.get("window_index")),
                "window_label": row.get("window_label"),
                "zone_set_id": row.get("zone_set_id"),
                "display_order": _safe_int(row.get("display_order")),
                "zone_id": row.get("zone_id"),
                "zone_label": row.get("zone_label"),
                "value": _safe_float(row.get(metric)),
            }
            for row in rows
        ]
    return response


def query_position_occupancy_histogram(
    context: ViewerContext,
    *,
    grid_id: str | None = None,
) -> dict[str, Any]:
    """Pool count-first 2D occupancy bins across recordings on one exact grid."""

    rows = load_optional_table_rows(context, POSITION_OCCUPANCY_TABLE)
    if not rows:
        return {
            "available": False,
            "message": "No position occupancy histogram table is present in this export.",
            "rows": [],
            "grid_ids": [],
        }
    grid_ids = sorted(
        {str(row.get("normalized_grid_id")) for row in rows if row.get("normalized_grid_id")}
    )
    if not grid_ids:
        return {
            "available": False,
            "message": "Position occupancy rows do not declare a normalized grid.",
            "rows": [],
            "grid_ids": [],
        }
    selected_grid_id = str(grid_id) if grid_id is not None else grid_ids[0]
    if selected_grid_id not in grid_ids:
        raise ValueError(f"Unknown position occupancy grid: {selected_grid_id}")
    if grid_id is None and len(grid_ids) > 1:
        return {
            "available": False,
            "message": (
                "This export contains incompatible normalized occupancy grids; "
                "select an explicit grid before pooling recordings."
            ),
            "rows": [],
            "grid_ids": grid_ids,
        }
    selected = [
        row for row in rows if str(row.get("normalized_grid_id")) == selected_grid_id
    ]
    grid_shapes = {
        (
            _safe_int(row.get("x_bin_count")),
            _safe_int(row.get("y_bin_count")),
        )
        for row in selected
    }
    if len(grid_shapes) != 1 or None in next(iter(grid_shapes), (None, None)):
        return {
            "available": False,
            "message": "Position occupancy rows disagree on normalized grid shape.",
            "rows": [],
            "grid_ids": grid_ids,
        }
    x_bin_count, y_bin_count = next(iter(grid_shapes))
    if not all(bool(row.get("normalized_grid_uniform")) for row in selected):
        return {
            "available": False,
            "message": "Sparse position occupancy rows require a declared uniform normalized grid.",
            "rows": [],
            "grid_ids": grid_ids,
        }
    windows = sorted(
        {
            (
                _safe_int(row.get("window_index")),
                row.get("window_label"),
            )
            for row in selected
        },
        key=lambda item: (_sort_int(item[0]), str(item[1] or "")),
    )
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for window_index, window_label in windows:
        for y_bin_index in range(int(y_bin_count)):
            for x_bin_index in range(int(x_bin_count)):
                key = (window_index, window_label, y_bin_index, x_bin_index)
                grouped[key] = {
                    "window_index": window_index,
                    "window_label": window_label,
                    "y_bin_index": y_bin_index,
                    "x_bin_index": x_bin_index,
                    "x_bin_left_fraction": x_bin_index / float(x_bin_count),
                    "x_bin_right_fraction": (x_bin_index + 1) / float(x_bin_count),
                    "y_bin_left_fraction": y_bin_index / float(y_bin_count),
                    "y_bin_right_fraction": (y_bin_index + 1) / float(y_bin_count),
                    "pooled_count": 0,
                    "normalized_grid_id": selected_grid_id,
                    "coordinate_frame": "source_image_fraction",
                    "coordinate_origin": "top_left",
                    "x_axis_direction": "right",
                    "y_axis_direction": "down",
                }
    recording_ids_by_window: dict[tuple[Any, Any], set[str]] = defaultdict(set)
    for row in selected:
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            _safe_int(row.get("y_bin_index")),
            _safe_int(row.get("x_bin_index")),
        )
        if key not in grouped:
            raise ValueError("Position occupancy row falls outside its declared grid")
        item = grouped[key]
        item["pooled_count"] += int(_safe_int(row.get("hist_count")) or 0)
        recording_id = str(row.get("recording_id") or "")
        if recording_id:
            recording_ids_by_window[(key[0], key[1])].add(recording_id)

    totals: dict[tuple[Any, Any], int] = defaultdict(int)
    for item in grouped.values():
        totals[(item["window_index"], item["window_label"])] += int(
            item["pooled_count"]
        )
    out_rows: list[dict[str, Any]] = []
    for item in grouped.values():
        x_left = float(item["x_bin_left_fraction"])
        x_right = float(item["x_bin_right_fraction"])
        y_left = float(item["y_bin_left_fraction"])
        y_right = float(item["y_bin_right_fraction"])
        window_key = (item["window_index"], item["window_label"])
        total = totals[window_key]
        count = int(item["pooled_count"])
        out_rows.append(
            {
                **item,
                "x_bin_center_fraction": (x_left + x_right) / 2.0,
                "x_bin_width_fraction": x_right - x_left,
                "y_bin_center_fraction": (y_left + y_right) / 2.0,
                "y_bin_width_fraction": y_right - y_left,
                "pooled_total_count": total,
                "pooled_probability": _round(count / total if total > 0 else None),
                "recording_count": len(recording_ids_by_window[window_key]),
            }
        )
    out_rows.sort(
        key=lambda row: (
            _sort_int(row.get("window_index")),
            _sort_int(row.get("y_bin_index"), 0),
            _sort_int(row.get("x_bin_index"), 0),
        )
    )
    return {
        "available": True,
        "rows": out_rows,
        "grid_id": selected_grid_id,
        "grid_ids": grid_ids,
        "recording_count": len(
            {str(row.get("recording_id")) for row in selected if row.get("recording_id")}
        ),
        "pooling_method": "sum_counts_then_normalize_within_epoch",
    }


def query_position_occupancy_grid_options(
    context: ViewerContext,
) -> list[dict[str, Any]]:
    """List incompatible normalized occupancy grids with explicit cohort coverage."""

    rows = load_optional_table_rows(context, POSITION_OCCUPANCY_TABLE)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        grid_id = str(row.get("normalized_grid_id") or "")
        if not grid_id:
            continue
        item = grouped.setdefault(
            grid_id,
            {
                "grid_id": grid_id,
                "x_bin_count": _safe_int(row.get("x_bin_count")),
                "y_bin_count": _safe_int(row.get("y_bin_count")),
                "recording_ids": set(),
            },
        )
        recording_id = str(row.get("recording_id") or "")
        if recording_id:
            item["recording_ids"].add(recording_id)
    output = []
    for item in grouped.values():
        recording_count = len(item["recording_ids"])
        x_count = item["x_bin_count"]
        y_count = item["y_bin_count"]
        output.append(
            {
                "grid_id": item["grid_id"],
                "x_bin_count": x_count,
                "y_bin_count": y_count,
                "recording_count": recording_count,
                "label": (
                    f"{x_count} × {y_count} native bins · "
                    f"{recording_count} recording{'s' if recording_count != 1 else ''} "
                    f"[{item['grid_id']}]"
                ),
            }
        )
    output.sort(
        key=lambda item: (
            -int(item["recording_count"]),
            _sort_int(item.get("y_bin_count")),
            _sort_int(item.get("x_bin_count")),
            str(item["grid_id"]),
        )
    )
    return output


def position_occupancy_rebin_options(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return exact index-aligned x/y coarsenings for a pooled occupancy grid."""

    if not rows:
        native = [{"factor": 1, "label": "Native"}]
        return {"x": native, "y": native}

    def _axis_options(axis: str) -> list[dict[str, Any]]:
        index_key = f"{axis}_bin_index"
        width_key = f"{axis}_bin_width_fraction"
        indices = sorted(
            {
                int(index)
                for row in rows
                if (index := _safe_int(row.get(index_key))) is not None
            }
        )
        widths = {
            round(float(width), 12)
            for row in rows
            if (width := _safe_float(row.get(width_key))) is not None and width > 0.0
        }
        if not indices or indices != list(range(len(indices))) or len(widths) != 1:
            raise ValueError(f"Cannot re-bin a non-uniform or discontinuous {axis} grid")
        native_width = float(next(iter(widths)))
        return [
            {
                "factor": factor,
                "width_fraction": native_width * factor,
                "label": (
                    f"{native_width:.3g} arena fraction (native)"
                    if factor == 1
                    else f"{native_width * factor:.3g} arena fraction ({factor}× native)"
                ),
            }
            for factor in range(1, min(len(indices), 10) + 1)
        ]

    return {"x": _axis_options("x"), "y": _axis_options("y")}


def rebin_position_occupancy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_bin_factor: int = 1,
    y_bin_factor: int = 1,
) -> list[dict[str, Any]]:
    """Exactly coarsen pooled 2D occupancy bins by summing native counts."""

    x_factor = int(x_bin_factor)
    y_factor = int(y_bin_factor)
    if x_factor < 1 or y_factor < 1:
        raise ValueError("Position occupancy re-bin factors must be positive integers")
    if not rows:
        return []
    options = position_occupancy_rebin_options(rows)
    max_x_factor = len(options["x"])
    max_y_factor = len(options["y"])
    if x_factor > max_x_factor or y_factor > max_y_factor:
        raise ValueError("Position occupancy re-bin factor exceeds the supported grid")

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        x_index = int(_safe_int(row.get("x_bin_index")) or 0)
        y_index = int(_safe_int(row.get("y_bin_index")) or 0)
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            y_index // y_factor,
            x_index // x_factor,
        )
        x_left = _safe_float(row.get("x_bin_left_fraction"))
        x_right = _safe_float(row.get("x_bin_right_fraction"))
        y_left = _safe_float(row.get("y_bin_left_fraction"))
        y_right = _safe_float(row.get("y_bin_right_fraction"))
        if None in (x_left, x_right, y_left, y_right):
            raise ValueError("Position occupancy row has incomplete bin geometry")
        item = grouped.setdefault(
            key,
            {
                "window_index": key[0],
                "window_label": key[1],
                "y_bin_index": key[2],
                "x_bin_index": key[3],
                "x_bin_left_fraction": float(x_left),
                "x_bin_right_fraction": float(x_right),
                "y_bin_left_fraction": float(y_left),
                "y_bin_right_fraction": float(y_right),
                "pooled_count": 0,
                "recording_count": _safe_int(row.get("recording_count")),
                "normalized_grid_id": row.get("normalized_grid_id"),
                "coordinate_frame": row.get("coordinate_frame"),
                "coordinate_origin": row.get("coordinate_origin"),
                "x_axis_direction": row.get("x_axis_direction"),
                "y_axis_direction": row.get("y_axis_direction"),
            },
        )
        item["x_bin_left_fraction"] = min(float(item["x_bin_left_fraction"]), float(x_left))
        item["x_bin_right_fraction"] = max(float(item["x_bin_right_fraction"]), float(x_right))
        item["y_bin_left_fraction"] = min(float(item["y_bin_left_fraction"]), float(y_left))
        item["y_bin_right_fraction"] = max(float(item["y_bin_right_fraction"]), float(y_right))
        item["pooled_count"] += int(_safe_int(row.get("pooled_count")) or 0)

    totals: dict[tuple[Any, Any], int] = defaultdict(int)
    for item in grouped.values():
        totals[(item["window_index"], item["window_label"])] += int(item["pooled_count"])
    output: list[dict[str, Any]] = []
    for item in grouped.values():
        x_left = float(item["x_bin_left_fraction"])
        x_right = float(item["x_bin_right_fraction"])
        y_left = float(item["y_bin_left_fraction"])
        y_right = float(item["y_bin_right_fraction"])
        total = totals[(item["window_index"], item["window_label"])]
        count = int(item["pooled_count"])
        output.append(
            {
                **item,
                "x_bin_center_fraction": (x_left + x_right) / 2.0,
                "x_bin_width_fraction": x_right - x_left,
                "y_bin_center_fraction": (y_left + y_right) / 2.0,
                "y_bin_width_fraction": y_right - y_left,
                "pooled_total_count": total,
                "pooled_probability": _round(count / total if total > 0 else None),
                "x_bin_factor": x_factor,
                "y_bin_factor": y_factor,
            }
        )
    output.sort(
        key=lambda row: (
            _sort_int(row.get("window_index")),
            _sort_int(row.get("y_bin_index"), 0),
            _sort_int(row.get("x_bin_index"), 0),
        )
    )
    return output


def _enrich_chaser_behavior_rows(
    rows: Sequence[Mapping[str, Any]],
    object_phase_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fill unknown chaser roles from the exported quadrant-occupancy mapping."""

    metadata_by_recording_column: dict[tuple[str, int], tuple[str, str | None]] = {}
    for row in object_phase_rows:
        recording_id = str(row.get("recording_id") or "")
        object_column = _safe_int(row.get("object_column_index"))
        role = str(row.get("object_role") or row.get("behavior_class") or "").strip()
        raw_color = str(row.get("raw_color_hex") or "").strip() or None
        if recording_id and object_column is not None and role and role != "unknown":
            metadata_by_recording_column[(recording_id, object_column)] = (
                role,
                raw_color,
            )

    enriched: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        current_role = str(row.get("behavior_class") or "").strip()
        if not current_role or current_role == "unknown":
            recording_id = str(row.get("recording_id") or "")
            chaser_column = _safe_int(row.get("chaser_column_index"))
            if recording_id and chaser_column is not None:
                resolved = metadata_by_recording_column.get(
                    (recording_id, chaser_column)
                )
                if resolved:
                    row["behavior_class"] = resolved[0]
        recording_id = str(row.get("recording_id") or "")
        chaser_column = _safe_int(row.get("chaser_column_index"))
        resolved = (
            metadata_by_recording_column.get((recording_id, chaser_column))
            if recording_id and chaser_column is not None
            else None
        )
        if resolved and resolved[1]:
            row["raw_color_hex"] = resolved[1]
        enriched.append(row)
    return enriched


def _consistent_raw_color(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str]:
    colors = sorted(
        {
            str(row.get("raw_color_hex") or "").strip().lower()
            for row in rows
            if str(row.get("raw_color_hex") or "").strip()
        }
    )
    if not colors:
        return None, "missing"
    if len(colors) == 1:
        return colors[0], "consistent"
    return None, "mixed"


def query_chaser_summary(
    context: ViewerContext,
    *,
    metric: str = "p50_distance_mm",
    stat: str = "mean",
    include_recordings: bool = False,
) -> dict[str, Any]:
    if metric not in CHASER_METRICS:
        raise ValueError(f"Unsupported chaser metric: {metric}")
    if stat not in {"mean", "median"}:
        raise ValueError("stat must be one of: mean, median")
    rows = _enrich_chaser_behavior_rows(
        load_table_rows(context, CHASER_SUMMARY_TABLE),
        load_optional_table_rows(context, CRA_OBJECT_PHASE_TABLE),
    )
    _stats_run_id, descriptive_rows = _load_descriptive_rows(context)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            _safe_int(row.get("chaser_index")),
            row.get("behavior_class"),
        )].append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        window_index, window_label, chaser_index, behavior_class = key
        condition_name = canonical_condition(window_label)
        descriptive = None
        if condition_name and behavior_class and behavior_class != "unknown":
            descriptive = _descriptive_row_for(
                descriptive_rows,
                metric_family="chaser_distance",
                metric_name=metric,
                condition_name=condition_name or "",
                group_key={"behavior_class": behavior_class},
            )
        if descriptive is None and condition_name and chaser_index is not None:
            descriptive = _descriptive_row_for(
                descriptive_rows,
                metric_family="chaser_distance",
                metric_name=metric,
                condition_name=condition_name,
                group_key={"chaser_index": chaser_index},
            )
        if descriptive is not None:
            stats = _summary_from_descriptive_row(descriptive)
            summary_source = "persisted_descriptive_summary"
        else:
            values = _values(group_rows, metric)
            stats = _summary(values)
            summary_source = "computed_from_export_rows"
        value = stats["median"] if stat == "median" else stats["mean"]
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        raw_color_hex, color_status = _consistent_raw_color(group_rows)
        out_rows.append(
            {
                "window_index": window_index,
                "window_label": window_label,
                "chaser_index": chaser_index,
                "behavior_class": behavior_class,
                "raw_color_hex": raw_color_hex,
                "color_status": color_status,
                "value": value,
                "stat": stat,
                "recording_count": stats["n"] if descriptive is not None else len(recording_ids),
                "summary_source": summary_source,
                **stats,
            }
        )
    out_rows.sort(key=lambda row: (_sort_int(row.get("window_index")), _sort_int(row.get("chaser_index"), 0)))
    response: dict[str, Any] = {
        "metric": metric,
        "metric_label": CHASER_METRICS[metric],
        "stat": stat,
        "summary_source": (
            "persisted_descriptive_summary" if any(row.get("summary_source") == "persisted_descriptive_summary" for row in out_rows)
            else "computed_from_export_rows"
        ),
        "rows": out_rows,
    }
    if include_recordings:
        response["per_recording"] = [
            {
                "recording_id": row.get("recording_id"),
                "window_index": _safe_int(row.get("window_index")),
                "window_label": row.get("window_label"),
                "chaser_index": _safe_int(row.get("chaser_index")),
                "behavior_class": row.get("behavior_class"),
                "value": _safe_float(row.get(metric)),
            }
            for row in rows
        ]
    return response


def query_epoch_speed_summary(
    context: ViewerContext,
    *,
    metric: str = "mean_speed_mm_s",
    stat: str = "mean",
    include_recordings: bool = False,
) -> dict[str, Any]:
    if metric not in EPOCH_BEHAVIOR_METRICS:
        raise ValueError(f"Unsupported epoch behavior metric: {metric}")
    if stat not in {"mean", "median"}:
        raise ValueError("stat must be one of: mean, median")
    rows = load_optional_table_rows(context, EPOCH_BEHAVIOR_TABLE)
    source_table = EPOCH_BEHAVIOR_TABLE
    source_label = "persisted_epoch_behavior"
    if not rows:
        return {
            "available": False,
            "metric": metric,
            "metric_label": EPOCH_BEHAVIOR_METRICS[metric],
            "stat": stat,
            "rows": [],
            "source_table": source_table,
            "source_label": source_label,
            "message": "No epoch behavior export found for this cohort.",
        }

    _stats_run_id, descriptive_rows = _load_descriptive_rows(context)
    metric_family = "epoch_behavior"
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            _safe_int(row.get("window_index")),
            row.get("window_label"),
        )].append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        window_index, window_label = key
        condition_name = canonical_condition(window_label)
        descriptive = (
            _descriptive_row_for(
                descriptive_rows,
                metric_family=metric_family,
                metric_name=metric,
                condition_name=condition_name or "",
                group_key={},
            )
            if condition_name
            else None
        )
        if descriptive is not None:
            stats = _summary_from_descriptive_row(descriptive)
            summary_source = "persisted_descriptive_summary"
        else:
            values = _values(group_rows, metric)
            stats = _summary(values)
            summary_source = "computed_from_export_rows"
        value = stats["median"] if stat == "median" else stats["mean"]
        out_rows.append(
            {
                "window_index": window_index,
                "window_label": window_label,
                "value": value,
                "stat": stat,
                "recording_count": stats["n"],
                "source_table": source_table,
                "source_label": source_label,
                "summary_source": summary_source,
                **stats,
            }
        )
    out_rows.sort(key=lambda row: _sort_int(row.get("window_index")))

    response: dict[str, Any] = {
        "available": True,
        "metric": metric,
        "metric_label": EPOCH_BEHAVIOR_METRICS[metric],
        "stat": stat,
        "source_table": source_table,
        "source_label": source_label,
        "summary_source": (
            "persisted_descriptive_summary" if any(row.get("summary_source") == "persisted_descriptive_summary" for row in out_rows)
            else "computed_from_export_rows"
        ),
        "rows": out_rows,
    }
    if include_recordings:
        response["per_recording"] = [
            {
                "recording_id": row.get("recording_id"),
                "window_index": _safe_int(row.get("window_index")),
                "window_label": row.get("window_label"),
                "value": _safe_float(row.get(metric)),
                "mean_speed_mm_s": _safe_float(row.get("mean_speed_mm_s")),
                "median_speed_mm_s": _safe_float(row.get("median_speed_mm_s")),
                "speed_sample_count": _safe_int(row.get("speed_sample_count")),
                "bout_count": _safe_int(row.get("bout_count")),
                "bout_rate_per_min": _safe_float(row.get("bout_rate_per_min")),
                "mean_bout_duration_s": _safe_float(row.get("mean_bout_duration_s")),
                "median_bout_duration_s": _safe_float(row.get("median_bout_duration_s")),
                "mean_bout_path_length_mm": _safe_float(row.get("mean_bout_path_length_mm")),
                "median_bout_path_length_mm": _safe_float(row.get("median_bout_path_length_mm")),
                "bout_heading_sample_count": _safe_int(row.get("bout_heading_sample_count")),
                "mean_abs_bout_net_heading_change_deg": _safe_float(
                    row.get("mean_abs_bout_net_heading_change_deg")
                ),
                "median_abs_bout_net_heading_change_deg": _safe_float(
                    row.get("median_abs_bout_net_heading_change_deg")
                ),
                "mean_bout_heading_path_deg": _safe_float(row.get("mean_bout_heading_path_deg")),
                "median_bout_heading_path_deg": _safe_float(row.get("median_bout_heading_path_deg")),
                "inter_bout_interval_count": _safe_int(row.get("inter_bout_interval_count")),
                "mean_inter_bout_interval_s": _safe_float(row.get("mean_inter_bout_interval_s")),
                "median_inter_bout_interval_s": _safe_float(row.get("median_inter_bout_interval_s")),
                "mean_distance_from_arena_center_mm": _safe_float(
                    row.get("mean_distance_from_arena_center_mm")
                ),
                "median_distance_from_arena_center_mm": _safe_float(
                    row.get("median_distance_from_arena_center_mm")
                ),
                "wall_fraction": _safe_float(row.get("wall_fraction")),
                "wall_time_s": _safe_float(row.get("wall_time_s")),
                "tracking_dropout_fraction": _safe_float(row.get("tracking_dropout_fraction")),
                "source_position_path": row.get("source_position_path"),
                "source_table": source_table,
            }
            for row in rows
        ]
    return response


def query_epoch_center_distance_histogram(
    context: ViewerContext,
    *,
    window_label: str | None = None,
) -> dict[str, Any]:
    rows = load_optional_table_rows(context, EPOCH_CENTER_DISTANCE_HISTOGRAM_TABLE)
    if window_label:
        rows = [row for row in rows if row.get("window_label") == window_label]
    if not rows:
        return {
            "available": False,
            "rows": [],
            "message": "No epoch center-distance histogram export found for this cohort.",
        }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            _safe_int(row.get("bin_index")),
            _safe_float(row.get("bin_left_mm")),
            _safe_float(row.get("bin_right_mm")),
            _safe_float(row.get("bin_center_mm")),
            _safe_float(row.get("bin_width_mm")),
        )
        grouped[key].append(row)

    out_rows: list[dict[str, Any]] = []
    totals_by_window: dict[tuple[Any, ...], int] = {}
    for key, group_rows in grouped.items():
        window_key = key[:2]
        totals_by_window[window_key] = totals_by_window.get(window_key, 0) + sum(
            _safe_int(row.get("hist_count")) or 0 for row in group_rows
        )
    for key, group_rows in grouped.items():
        (
            window_index,
            label,
            bin_index,
            bin_left,
            bin_right,
            bin_center,
            bin_width,
        ) = key
        pooled_count = sum(_safe_int(row.get("hist_count")) or 0 for row in group_rows)
        pooled_total = totals_by_window.get((window_index, label), 0)
        pooled_fraction = float(pooled_count) / float(pooled_total) if pooled_total > 0 else None
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        out_rows.append(
            {
                "window_index": window_index,
                "window_label": label,
                "bin_index": bin_index,
                "bin_left_mm": bin_left,
                "bin_right_mm": bin_right,
                "bin_center_mm": bin_center,
                "bin_width_mm": bin_width,
                "pooled_count": pooled_count,
                "pooled_total_count": pooled_total,
                "pooled_fraction": _round(pooled_fraction),
                "pooled_density_per_mm": _round(
                    pooled_fraction / bin_width
                    if pooled_fraction is not None and bin_width is not None and bin_width > 0
                    else None
                ),
                "recording_count": len(recording_ids),
            }
        )
    out_rows.sort(
        key=lambda row: (_sort_int(row.get("window_index")), _sort_int(row.get("bin_index")))
    )
    return {
        "available": True,
        "rows": out_rows,
        "source_table": EPOCH_CENTER_DISTANCE_HISTOGRAM_TABLE,
        "summary_source": "pooled_export_counts",
    }


def _query_epoch_metric_histogram(
    context: ViewerContext,
    *,
    table_name: str,
    metric: str,
    metrics: Mapping[str, str],
    window_label: str | None = None,
) -> dict[str, Any]:
    if metric not in metrics:
        raise ValueError(f"Unsupported epoch metric histogram metric: {metric}")
    rows = load_optional_table_rows(context, table_name)
    rows = [row for row in rows if row.get("metric_name") == metric]
    if window_label:
        rows = [row for row in rows if row.get("window_label") == window_label]
    if not rows:
        return {
            "available": False,
            "metric": metric,
            "metric_label": metrics[metric],
            "rows": [],
            "source_table": table_name,
            "message": "No epoch metric histogram export found for this cohort.",
        }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bin_left = _safe_float(row.get("bin_left"))
        bin_right = _safe_float(row.get("bin_right"))
        bin_center = _safe_float(row.get("bin_center"))
        bin_width = _safe_float(row.get("bin_width"))
        if bin_width is None and bin_left is not None and bin_right is not None:
            bin_width = bin_right - bin_left
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            row.get("metric_name"),
            row.get("units"),
            _safe_int(row.get("bin_index")),
            bin_left,
            bin_right,
            bin_center,
            bin_width,
        )
        grouped[key].append(row)

    totals_by_window: dict[tuple[Any, ...], int] = {}
    for key, group_rows in grouped.items():
        window_key = key[:4]
        totals_by_window[window_key] = totals_by_window.get(window_key, 0) + sum(
            _safe_int(row.get("hist_count")) or 0 for row in group_rows
        )

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        (
            window_index,
            label,
            metric_name,
            units,
            bin_index,
            bin_left,
            bin_right,
            bin_center,
            bin_width,
        ) = key
        pooled_count = sum(_safe_int(row.get("hist_count")) or 0 for row in group_rows)
        pooled_total = totals_by_window.get((window_index, label, metric_name, units), 0)
        pooled_fraction = float(pooled_count) / float(pooled_total) if pooled_total > 0 else None
        pooled_density = (
            pooled_fraction / bin_width
            if pooled_fraction is not None and bin_width is not None and bin_width > 0
            else None
        )
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        out_rows.append(
            {
                "window_index": window_index,
                "window_label": label,
                "metric_name": metric_name,
                "metric_label": metrics[metric],
                "units": units,
                "bin_index": bin_index,
                "bin_left": bin_left,
                "bin_right": bin_right,
                "bin_center": bin_center,
                "bin_width": bin_width,
                "pooled_count": pooled_count,
                "pooled_total_count": pooled_total,
                "pooled_fraction": _round(pooled_fraction),
                "pooled_density": _round(pooled_density),
                "recording_count": len(recording_ids),
            }
        )
    out_rows.sort(key=lambda row: (_sort_int(row.get("window_index")), _sort_int(row.get("bin_index"))))
    return {
        "available": True,
        "metric": metric,
        "metric_label": metrics[metric],
        "rows": out_rows,
        "source_table": table_name,
        "summary_source": "pooled_export_counts",
    }


def query_epoch_bout_histogram(
    context: ViewerContext,
    *,
    metric: str = "bout_path_length_mm",
    window_label: str | None = None,
) -> dict[str, Any]:
    return _query_epoch_metric_histogram(
        context,
        table_name=EPOCH_BOUT_HISTOGRAM_TABLE,
        metric=metric,
        metrics=EPOCH_BOUT_HISTOGRAM_METRICS,
        window_label=window_label,
    )


def query_epoch_inter_bout_interval_histogram(
    context: ViewerContext,
    *,
    metric: str = "inter_bout_interval_s",
    window_label: str | None = None,
) -> dict[str, Any]:
    return _query_epoch_metric_histogram(
        context,
        table_name=EPOCH_INTER_BOUT_INTERVAL_HISTOGRAM_TABLE,
        metric=metric,
        metrics=EPOCH_INTER_BOUT_INTERVAL_HISTOGRAM_METRICS,
        window_label=window_label,
    )


def query_speed_distance_bins(
    context: ViewerContext,
    *,
    window_label: str | None = None,
    chaser_index: int | None = None,
) -> dict[str, Any]:
    rows = _enrich_chaser_behavior_rows(
        load_optional_table_rows(context, SPEED_DISTANCE_TABLE),
        load_optional_table_rows(context, CRA_OBJECT_PHASE_TABLE),
    )
    if window_label:
        rows = [row for row in rows if row.get("window_label") == window_label]
    if chaser_index is not None:
        rows = [row for row in rows if _safe_int(row.get("chaser_index")) == chaser_index]
    if not rows:
        return {
            "available": False,
            "rows": [],
            "message": "No speed-distance export found for this cohort.",
        }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            _safe_int(row.get("chaser_index")),
            _safe_int(row.get("distance_bin_index")),
            _safe_float(row.get("distance_bin_left_mm")),
            _safe_float(row.get("distance_bin_right_mm")),
            _safe_float(row.get("distance_bin_center_mm")),
            _safe_float(row.get("distance_bin_width_mm")),
        )
        grouped[key].append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        (
            window_index,
            label,
            chaser,
            bin_index,
            bin_left,
            bin_right,
            bin_center,
            bin_width,
        ) = key
        count = sum(int(_safe_int(row.get("speed_sample_count")) or 0) for row in group_rows)
        speed_sum = sum(float(_safe_float(row.get("speed_sum_mm_s")) or 0.0) for row in group_rows)
        values = _values(group_rows, "mean_speed_mm_s")
        stats = _summary(values)
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        pooled_mean = speed_sum / float(count) if count > 0 else None
        raw_color_hex, color_status = _consistent_raw_color(group_rows)
        out_rows.append(
            {
                "window_index": window_index,
                "window_label": label,
                "chaser_index": chaser,
                "raw_color_hex": raw_color_hex,
                "color_status": color_status,
                "distance_bin_index": bin_index,
                "distance_bin_left_mm": bin_left,
                "distance_bin_right_mm": bin_right,
                "distance_bin_center_mm": bin_center,
                "distance_bin_width_mm": bin_width,
                "pooled_speed_sample_count": int(count),
                "pooled_mean_speed_mm_s": _round(pooled_mean),
                "recording_count": len(recording_ids),
                "recording_mean_speed_mm_s": stats["mean"],
                "recording_median_speed_mm_s": stats["median"],
                "recording_std_dev": stats["std_dev"],
                "recording_sem": stats["sem"],
                "recording_min": stats["min"],
                "recording_max": stats["max"],
            }
        )
    out_rows.sort(
        key=lambda row: (
            _sort_int(row.get("window_index")),
            _sort_int(row.get("chaser_index"), 0),
            _sort_int(row.get("distance_bin_index"), 0),
        )
    )
    return {
        "available": True,
        "rows": out_rows,
        "value_label": "Pooled mean speed (mm/s)",
        "definition": "Speed at frame t from fish displacement t->t+1 plotted against distance to chaser at frame t; both frames must be valid and within the same epoch.",
    }


def query_chaser_histogram(
    context: ViewerContext,
    *,
    window_label: str | None = None,
    chaser_index: int | None = None,
) -> dict[str, Any]:
    rows = _enrich_chaser_behavior_rows(
        load_table_rows(context, CHASER_HISTOGRAM_TABLE),
        load_optional_table_rows(context, CRA_OBJECT_PHASE_TABLE),
    )
    if window_label:
        rows = [row for row in rows if row.get("window_label") == window_label]
    if chaser_index is not None:
        rows = [row for row in rows if _safe_int(row.get("chaser_index")) == chaser_index]

    grouped: dict[tuple[Any, ...], int] = defaultdict(int)
    meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    raw_colors: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            _safe_int(row.get("chaser_index")),
            _safe_int(row.get("distance_bin_index")),
            _safe_float(row.get("bin_left_mm")),
            _safe_float(row.get("bin_right_mm")),
            _safe_float(row.get("bin_center_mm")),
            _safe_float(row.get("bin_width_mm")),
        )
        grouped[key] += int(_safe_int(row.get("hist_count")) or 0)
        if row.get("raw_color_hex"):
            raw_colors[key].add(str(row["raw_color_hex"]).strip().lower())
        meta.setdefault(
            key,
            {
                "window_index": key[0],
                "window_label": key[1],
                "chaser_index": key[2],
                "behavior_class": row.get("behavior_class"),
                "distance_bin_index": key[3],
                "bin_left_mm": key[4],
                "bin_right_mm": key[5],
                "bin_center_mm": key[6],
                "bin_width_mm": key[7],
            },
        )

    totals: dict[tuple[Any, Any], int] = defaultdict(int)
    for key, count in grouped.items():
        totals[(key[1], key[2])] += int(count)

    out_rows: list[dict[str, Any]] = []
    for key, count in grouped.items():
        item = dict(meta[key])
        colors = sorted(raw_colors.get(key, set()))
        item["raw_color_hex"] = colors[0] if len(colors) == 1 else None
        item["color_status"] = (
            "consistent" if len(colors) == 1 else "mixed" if colors else "missing"
        )
        total = totals[(key[1], key[2])]
        bin_width = _safe_float(item.get("bin_width_mm")) or 1.0
        density = float(count) / float(total * bin_width) if total > 0 and bin_width > 0 else None
        item.update({
            "pooled_count": int(count),
            "pooled_total_count": int(total),
            "pooled_density": _round(density),
        })
        out_rows.append(item)
    out_rows.sort(
        key=lambda row: (
            _sort_int(row.get("window_index")),
            _sort_int(row.get("chaser_index"), 0),
            _sort_int(row.get("distance_bin_index"), 0),
        )
    )
    return {"rows": out_rows}


def query_cra_object_phase(
    context: ViewerContext,
    *,
    metric: str = "median_distance_mm",
    stat: str = "mean",
    object_role: str | None = None,
    include_recordings: bool = False,
) -> dict[str, Any]:
    if metric not in CRA_OBJECT_PHASE_METRICS:
        raise ValueError(f"Unsupported chaser quadrant-occupancy metric: {metric}")
    if stat not in {"mean", "median"}:
        raise ValueError("stat must be one of: mean, median")
    rows = load_table_rows(context, CRA_OBJECT_PHASE_TABLE)
    if object_role:
        rows = [row for row in rows if row.get("object_role") == object_role]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            _safe_int(row.get("phase_axis_index")),
            row.get("phase_label"),
            row.get("object_role"),
            _safe_int(row.get("object_index")),
            row.get("raw_color_hex"),
            row.get("object_quadrant_label"),
        )].append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        phase_axis_index, phase_label, role, object_index, raw_color_hex, quadrant_label = key
        values = _values(group_rows, metric)
        stats = _summary(values)
        value = stats["median"] if stat == "median" else stats["mean"]
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        out_rows.append(
            {
                "phase_axis_index": phase_axis_index,
                "phase_label": phase_label,
                "object_role": role,
                "object_index": object_index,
                "raw_color_hex": raw_color_hex,
                "object_quadrant_label": quadrant_label,
                "value": value,
                "stat": stat,
                "recording_count": len(recording_ids),
                **stats,
            }
        )
    role_order = {"aggressive": 0, "inert": 1, "random_non_chasing": 2, "unknown": 3}
    out_rows.sort(
        key=lambda row: (
            _sort_int(row.get("phase_axis_index")),
            role_order.get(str(row.get("object_role")), 99),
            _sort_int(row.get("object_index"), 0),
        )
    )
    response: dict[str, Any] = {
        "metric": metric,
        "metric_label": CRA_OBJECT_PHASE_METRICS[metric],
        "stat": stat,
        "rows": out_rows,
    }
    if include_recordings:
        response["per_recording"] = [
            {
                "recording_id": row.get("recording_id"),
                "fish_id": row.get("fish_id"),
                "phase_axis_index": _safe_int(row.get("phase_axis_index")),
                "phase_label": row.get("phase_label"),
                "object_role": row.get("object_role"),
                "object_index": _safe_int(row.get("object_index")),
                "raw_color_hex": row.get("raw_color_hex"),
                "object_quadrant_label": row.get("object_quadrant_label"),
                "value": _safe_float(row.get(metric)),
                "tracking_dropout_fraction": _safe_float(row.get("tracking_dropout_fraction")),
                "object_max_drift_mm": _safe_float(row.get("object_max_drift_mm")),
                "source_cra_primary_endpoint_path": row.get("source_cra_primary_endpoint_path"),
            }
            for row in rows
        ]
    return response


def _profile_pairwise_quadrant_summary_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    chaser_phase_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive profile-level role contrasts from generic per-chaser rows.

    Recording analyses intentionally do not persist an aggressive-vs-inert
    contrast.  A cohort/profile viewer may derive one when those roles are
    actually present.  Multiple chasers with the same role are averaged rather
    than silently choosing an identity.
    """

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in chaser_phase_rows:
        recording_id = str(row.get("recording_id") or "")
        role = str(row.get("object_role") or row.get("behavior_class") or "")
        phase_kind = _phase_kind(row)
        if (
            recording_id
            and role in {"aggressive", "inert"}
            and phase_kind in {"pre", "post"}
        ):
            grouped[(recording_id, role, phase_kind)].append(row)

    def mean_value(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        values = [_safe_float(row.get(key)) for row in rows]
        finite = [float(value) for value in values if value is not None]
        return sum(finite) / float(len(finite)) if finite else None

    def unique_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
        values = {row.get(key) for row in rows if row.get(key) not in {None, ""}}
        return next(iter(values)) if len(values) == 1 else None

    out: list[dict[str, Any]] = []
    for source in summary_rows:
        row = dict(source)
        recording_id = str(row.get("recording_id") or "")
        for role, suffix in (("aggressive", "agg"), ("inert", "inert")):
            pre_rows = grouped.get((recording_id, role, "pre"), [])
            post_rows = grouped.get((recording_id, role, "post"), [])
            d_pre = mean_value(pre_rows, "median_distance_mm")
            d_post = mean_value(post_rows, "median_distance_mm")
            occ_pre = mean_value(pre_rows, "occupancy_fraction")
            occ_post = mean_value(post_rows, "occupancy_fraction")
            row.setdefault(f"d_pre_{suffix}", d_pre)
            row.setdefault(f"d_post_{suffix}", d_post)
            row.setdefault(
                f"delta_{suffix}",
                None if d_pre is None or d_post is None else d_post - d_pre,
            )
            row.setdefault(f"occ_pre_{suffix}", occ_pre)
            row.setdefault(f"occ_post_{suffix}", occ_post)
            row.setdefault(
                f"delta_occ_{suffix}",
                None if occ_pre is None or occ_post is None else occ_post - occ_pre,
            )
            row.setdefault(
                f"pre_{role}_quadrant",
                unique_value(pre_rows, "object_quadrant_label"),
            )
            row.setdefault(
                f"post_{role}_quadrant",
                unique_value(post_rows, "object_quadrant_label"),
            )
            row.setdefault(
                f"{role}_color",
                unique_value((*pre_rows, *post_rows), "raw_color_hex"),
            )
        delta_agg = _safe_float(row.get("delta_agg"))
        delta_inert = _safe_float(row.get("delta_inert"))
        delta_occ_agg = _safe_float(row.get("delta_occ_agg"))
        delta_occ_inert = _safe_float(row.get("delta_occ_inert"))
        row.setdefault(
            "specificity_distance",
            (
                None
                if delta_agg is None or delta_inert is None
                else delta_agg - delta_inert
            ),
        )
        row.setdefault(
            "specificity_occupancy",
            (
                None
                if delta_occ_agg is None or delta_occ_inert is None
                else delta_occ_agg - delta_occ_inert
            ),
        )
        out.append(row)
    return out


def query_cra_summary(
    context: ViewerContext,
    *,
    metric: str | None = None,
    endpoint_status: str | None = None,
    include_rows: bool = True,
) -> dict[str, Any]:
    if metric is not None and metric not in CRA_SUMMARY_METRICS:
        raise ValueError(f"Unsupported chaser quadrant summary metric: {metric}")
    rows = _profile_pairwise_quadrant_summary_rows(
        load_table_rows(context, CRA_SUMMARY_TABLE),
        load_optional_table_rows(context, CRA_OBJECT_PHASE_TABLE),
    )
    if endpoint_status:
        rows = [row for row in rows if row.get("endpoint_status") == endpoint_status]

    metrics = [metric] if metric else list(CRA_SUMMARY_METRICS)
    metric_rows: list[dict[str, Any]] = []
    for metric_name in metrics:
        values = _values(rows, metric_name)
        metric_rows.append(
            {
                "metric": metric_name,
                "metric_label": CRA_SUMMARY_METRICS[metric_name],
                **_summary(values),
            }
        )

    response: dict[str, Any] = {
        "row_count": len(rows),
        "metrics": metric_rows,
        "statuses": sorted({str(row.get("endpoint_status")) for row in rows if row.get("endpoint_status")}),
    }
    if include_rows:
        response["rows"] = [
            {
                "recording_id": row.get("recording_id"),
                "fish_id": row.get("fish_id"),
                "endpoint_status": row.get("endpoint_status"),
                "aggressive_color": row.get("aggressive_color"),
                "inert_color": row.get("inert_color"),
                "d_pre_agg": _safe_float(row.get("d_pre_agg")),
                "d_post_agg": _safe_float(row.get("d_post_agg")),
                "delta_agg": _safe_float(row.get("delta_agg")),
                "d_pre_inert": _safe_float(row.get("d_pre_inert")),
                "d_post_inert": _safe_float(row.get("d_post_inert")),
                "delta_inert": _safe_float(row.get("delta_inert")),
                "specificity_distance": _safe_float(row.get("specificity_distance")),
                "occ_pre_agg": _safe_float(row.get("occ_pre_agg")),
                "occ_post_agg": _safe_float(row.get("occ_post_agg")),
                "delta_occ_agg": _safe_float(row.get("delta_occ_agg")),
                "occ_pre_inert": _safe_float(row.get("occ_pre_inert")),
                "occ_post_inert": _safe_float(row.get("occ_post_inert")),
                "delta_occ_inert": _safe_float(row.get("delta_occ_inert")),
                "specificity_occupancy": _safe_float(row.get("specificity_occupancy")),
                "frac_tracking_dropout_pre": _safe_float(row.get("frac_tracking_dropout_pre")),
                "frac_tracking_dropout_post": _safe_float(row.get("frac_tracking_dropout_post")),
                "pre_aggressive_quadrant": row.get("pre_aggressive_quadrant"),
                "post_aggressive_quadrant": row.get("post_aggressive_quadrant"),
                "pre_inert_quadrant": row.get("pre_inert_quadrant"),
                "post_inert_quadrant": row.get("post_inert_quadrant"),
                "source_cra_primary_endpoint_path": row.get("source_cra_primary_endpoint_path"),
                "source_component_fingerprint": row.get("source_component_fingerprint"),
            }
            for row in rows
        ]
    return response


def _phase_kind(row: Mapping[str, Any]) -> str:
    for key in ("phase_label", "source_window_label", "window_label"):
        text = str(row.get(key) or "").strip().lower()
        if text.startswith("pre"):
            return "pre"
        if text.startswith("post"):
            return "post"
    return str(row.get("phase_label") or row.get("source_window_label") or "")


def _gaussian_kde_rows(
    values: Sequence[float],
    *,
    phase_label: str,
    series_role: str,
    bandwidth: float,
    grid: np.ndarray,
) -> list[dict[str, Any]]:
    finite = np.asarray(
        [float(value) for value in (_safe_float(item) for item in values) if value is not None],
        dtype=np.float64,
    )
    if finite.size == 0:
        return [
            {
                "phase_label": phase_label,
                "series_role": series_role,
                "x": _round(float(x_value)),
                "density": 0.0,
                "n": 0,
            }
            for x_value in grid
        ]
    safe_bandwidth = max(1e-6, float(bandwidth))
    delta = (grid.reshape(-1, 1) - finite.reshape(1, -1)) / safe_bandwidth
    density = np.exp(-0.5 * delta * delta).sum(axis=1)
    density /= float(finite.size) * safe_bandwidth * math.sqrt(2.0 * math.pi)
    return [
        {
            "phase_label": phase_label,
            "series_role": series_role,
            "x": _round(float(x_value)),
            "density": _round(float(y_value)),
            "n": int(finite.size),
        }
        for x_value, y_value in zip(grid, density)
    ]


def _one_sample_signed_rank_summary(
    values: Sequence[float],
    *,
    bootstrap_iterations: int,
    confidence_level: float,
    difference_definition: str,
) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in (_safe_float(item) for item in values) if value is not None],
        dtype=np.float64,
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "status": "skipped",
            "skip_reason": "no finite values",
            "n": 0,
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "ci_low": None,
            "ci_high": None,
            "mean_difference": None,
            "median_difference": None,
            "std_difference": None,
            "test_method": "wilcoxon_signed_rank_unavailable",
            "confidence_level": float(confidence_level),
            "bootstrap_iterations": 0,
            "difference_definition": difference_definition,
        }
    p_value, test_method, rank_biserial, w_plus = wilcoxon_signed_rank_p_value(finite)
    ci_low, ci_high = bootstrap_median_ci(
        finite,
        iterations=int(bootstrap_iterations),
        confidence_level=float(confidence_level),
        rng=np.random.default_rng(0),
    )
    std_difference = float(np.std(finite, ddof=1)) if finite.size > 1 else None
    return {
        "status": "computed" if p_value is not None else "skipped",
        "skip_reason": None if p_value is not None else "wilcoxon_unavailable",
        "n": int(finite.size),
        "statistic": _round(_safe_float(w_plus)),
        "p_value": _round(_safe_float(p_value)),
        "effect_size": _round(_safe_float(rank_biserial)),
        "ci_low": _round(ci_low),
        "ci_high": _round(ci_high),
        "mean_difference": _round(float(np.mean(finite))),
        "median_difference": _round(float(np.median(finite))),
        "std_difference": _round(std_difference),
        "test_method": test_method,
        "confidence_level": float(confidence_level),
        "bootstrap_iterations": int(bootstrap_iterations),
        "difference_definition": difference_definition,
    }


def _pre_post_slope_rows(
    phase_rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    group_key: str,
) -> list[dict[str, Any]]:
    by_unit: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in phase_rows:
        recording_id = str(row.get("recording_id") or "")
        group_value = str(row.get(group_key) or "")
        phase = str(row.get("phase_kind") or _phase_kind(row))
        if recording_id and group_value and phase in {"pre", "post"}:
            by_unit[(recording_id, group_value)][phase] = row

    out: list[dict[str, Any]] = []
    for (recording_id, group_value), phases in sorted(by_unit.items()):
        pre = _safe_float(phases.get("pre", {}).get(value_key))
        post = _safe_float(phases.get("post", {}).get(value_key))
        if pre is None or post is None:
            continue
        pre_row = phases["pre"]
        post_row = phases["post"]
        out.append(
            {
                "recording_id": recording_id,
                "fish_id": pre_row.get("fish_id") or post_row.get("fish_id"),
                group_key: group_value,
                "pre_phase_label": pre_row.get("phase_label"),
                "post_phase_label": post_row.get("phase_label"),
                "pre_value": _round(pre),
                "post_value": _round(post),
                "delta": _round(float(post) - float(pre)),
                "pre_object_quadrant": pre_row.get("object_quadrant_label"),
                "post_object_quadrant": post_row.get("object_quadrant_label"),
            }
        )
    return out


def query_cra_specificity(
    context: ViewerContext,
    *,
    bootstrap_iterations: int = 10000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    object_phase_rows = load_optional_table_rows(context, CRA_OBJECT_PHASE_TABLE)
    summary_rows = _profile_pairwise_quadrant_summary_rows(
        load_optional_table_rows(context, CRA_SUMMARY_TABLE),
        object_phase_rows,
    )
    if not summary_rows:
        return {
            "available": False,
            "table_name": CRA_SUMMARY_TABLE,
            "row_count": 0,
            "recording_count": 0,
            "distance_slope_rows": [],
            "distance_specificity_rows": [],
            "distance_specificity_statistics": None,
            "occupancy_index_phase_rows": [],
            "occupancy_index_slope_rows": [],
            "occupancy_index_specificity_rows": [],
            "occupancy_index_specificity_statistics": None,
            "message": "No chaser quadrant occupancy summary export found for this cohort.",
        }

    distance_phase_rows: list[dict[str, Any]] = []
    distance_specificity_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        fish_id = row.get("fish_id")
        role_specs = (
            ("aggressive", "agg", row.get("pre_aggressive_quadrant"), row.get("post_aggressive_quadrant")),
            ("inert", "inert", row.get("pre_inert_quadrant"), row.get("post_inert_quadrant")),
        )
        for role, suffix, pre_quadrant, post_quadrant in role_specs:
            pre_value = _safe_float(row.get(f"d_pre_{suffix}"))
            post_value = _safe_float(row.get(f"d_post_{suffix}"))
            if pre_value is not None:
                distance_phase_rows.append(
                    {
                        "recording_id": recording_id,
                        "fish_id": fish_id,
                        "object_role": role,
                        "phase_kind": "pre",
                        "phase_label": "pre_static",
                        "median_distance_mm": _round(pre_value),
                        "object_quadrant_label": pre_quadrant,
                    }
                )
            if post_value is not None:
                distance_phase_rows.append(
                    {
                        "recording_id": recording_id,
                        "fish_id": fish_id,
                        "object_role": role,
                        "phase_kind": "post",
                        "phase_label": "post_static",
                        "median_distance_mm": _round(post_value),
                        "object_quadrant_label": post_quadrant,
                    }
                )
        specificity = _safe_float(row.get("specificity_distance"))
        if specificity is not None:
            distance_specificity_rows.append(
                {
                    "recording_id": recording_id,
                    "fish_id": fish_id,
                    "specificity_distance": _round(specificity),
                    "delta_agg": _round(_safe_float(row.get("delta_agg"))),
                    "delta_inert": _round(_safe_float(row.get("delta_inert"))),
                }
            )

    distance_slope_rows = _pre_post_slope_rows(
        distance_phase_rows,
        value_key="median_distance_mm",
        group_key="object_role",
    )

    quadrant_rows = load_optional_table_rows(context, CRA_QUADRANT_OCCUPANCY_TABLE)
    quadrants_by_recording_phase: dict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in quadrant_rows:
        recording_id = str(row.get("recording_id") or "")
        phase_label = str(row.get("phase_label") or "")
        if recording_id and phase_label:
            quadrants_by_recording_phase[(recording_id, phase_label)].append(row)

    occupancy_index_phase_rows: list[dict[str, Any]] = []
    for row in object_phase_rows:
        role = str(row.get("object_role") or "")
        if role not in {"aggressive", "inert"}:
            continue
        recording_id = str(row.get("recording_id") or "")
        phase_label = str(row.get("phase_label") or "")
        object_occ = _safe_float(row.get("occupancy_fraction"))
        if not recording_id or not phase_label or object_occ is None:
            continue
        q_code = _safe_int(row.get("object_quadrant_code"))
        non_values = [
            _safe_float(qrow.get("occupancy_fraction"))
            for qrow in quadrants_by_recording_phase.get((recording_id, phase_label), [])
            if _safe_int(qrow.get("quadrant_code")) != q_code
        ]
        finite_non = [float(value) for value in non_values if value is not None]
        non_mean = (sum(finite_non) / float(len(finite_non))) if finite_non else (1.0 - float(object_occ)) / 3.0
        occupancy_index = float(object_occ) - float(non_mean)
        occupancy_index_phase_rows.append(
            {
                "recording_id": recording_id,
                "fish_id": row.get("fish_id"),
                "object_role": role,
                "phase_axis_index": _safe_int(row.get("phase_axis_index")),
                "phase_label": phase_label,
                "phase_kind": _phase_kind(row),
                "object_quadrant_label": row.get("object_quadrant_label"),
                "object_quadrant_occ": _round(object_occ),
                "non_object_quadrant_mean": _round(non_mean),
                "occupancy_index": _round(occupancy_index),
            }
        )

    occupancy_index_slope_rows = _pre_post_slope_rows(
        occupancy_index_phase_rows,
        value_key="occupancy_index",
        group_key="object_role",
    )
    slopes_by_recording: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in occupancy_index_slope_rows:
        recording_id = str(row.get("recording_id") or "")
        role = str(row.get("object_role") or "")
        if recording_id and role:
            slopes_by_recording[recording_id][role] = row
    occupancy_index_specificity_rows: list[dict[str, Any]] = []
    for recording_id, role_rows in sorted(slopes_by_recording.items()):
        agg = _safe_float(role_rows.get("aggressive", {}).get("delta"))
        inert = _safe_float(role_rows.get("inert", {}).get("delta"))
        if agg is None or inert is None:
            continue
        occupancy_index_specificity_rows.append(
            {
                "recording_id": recording_id,
                "fish_id": role_rows["aggressive"].get("fish_id") or role_rows["inert"].get("fish_id"),
                "delta_index_agg": _round(agg),
                "delta_index_inert": _round(inert),
                "occupancy_index_specificity": _round(float(agg) - float(inert)),
            }
        )

    return {
        "available": True,
        "table_name": CRA_SUMMARY_TABLE,
        "row_count": len(summary_rows),
        "recording_count": len({str(row.get("recording_id")) for row in summary_rows if row.get("recording_id")}),
        "distance_slope_rows": distance_slope_rows,
        "distance_specificity_rows": distance_specificity_rows,
        "distance_specificity_statistics": _one_sample_signed_rank_summary(
            [row["specificity_distance"] for row in distance_specificity_rows],
            bootstrap_iterations=int(bootstrap_iterations),
            confidence_level=float(confidence_level),
            difference_definition="specificity_distance = delta_agg - delta_inert; tested against zero",
        ),
        "occupancy_index_phase_rows": occupancy_index_phase_rows,
        "occupancy_index_slope_rows": occupancy_index_slope_rows,
        "occupancy_index_specificity_rows": occupancy_index_specificity_rows,
        "occupancy_index_specificity_statistics": _one_sample_signed_rank_summary(
            [row["occupancy_index_specificity"] for row in occupancy_index_specificity_rows],
            bootstrap_iterations=int(bootstrap_iterations),
            confidence_level=float(confidence_level),
            difference_definition="occupancy_index_specificity = delta_index_agg - delta_index_inert; tested against zero",
        ),
        "inference_note": (
            "Distance specificity is the profile-level role contrast. Occupancy index is chaser-quadrant occupancy "
            "minus the mean of the other three quadrants within the same phase."
        ),
    }


def _paired_chaser_occupancy_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    confidence_level: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_recording: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not bool(row.get("is_chaser_quadrant")):
            continue
        recording_id = str(row.get("recording_id") or "")
        phase = _phase_kind(row)
        if recording_id and phase in {"pre", "post"}:
            by_recording[recording_id][phase] = row

    paired_rows: list[dict[str, Any]] = []
    differences: list[float] = []
    for recording_id in sorted(by_recording):
        phases = by_recording[recording_id]
        pre = _safe_float(phases.get("pre", {}).get("occupancy_fraction"))
        post = _safe_float(phases.get("post", {}).get("occupancy_fraction"))
        if pre is None or post is None:
            continue
        diff = float(post) - float(pre)
        differences.append(diff)
        pre_row = phases["pre"]
        post_row = phases["post"]
        paired_rows.append(
            {
                "recording_id": recording_id,
                "fish_id": pre_row.get("fish_id") or post_row.get("fish_id"),
                "pre_phase_label": pre_row.get("phase_label"),
                "post_phase_label": post_row.get("phase_label"),
                "pre_chaser_quadrant": pre_row.get("chaser_quadrant_label"),
                "post_chaser_quadrant": post_row.get("chaser_quadrant_label"),
                "pre_chaser_quadrant_occ": _round(pre),
                "post_chaser_quadrant_occ": _round(post),
                "delta_chaser_quadrant_occ": _round(diff),
                "pre_tracking_dropout_fraction": _round(_safe_float(pre_row.get("tracking_dropout_fraction"))),
                "post_tracking_dropout_fraction": _round(_safe_float(post_row.get("tracking_dropout_fraction"))),
                "source_cra_primary_endpoint_path": pre_row.get("source_cra_primary_endpoint_path")
                or post_row.get("source_cra_primary_endpoint_path"),
            }
        )

    diffs = np.asarray(differences, dtype=np.float64)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return paired_rows, {
            "status": "skipped",
            "skip_reason": "no paired pre/post chaser-quadrant occupancy rows",
            "n": 0,
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "ci_low": None,
            "ci_high": None,
            "mean_difference": None,
            "median_difference": None,
            "std_difference": None,
            "test_method": "wilcoxon_signed_rank_unavailable",
            "confidence_level": float(confidence_level),
            "bootstrap_iterations": 0,
        }

    p_value, test_method, rank_biserial, w_plus = wilcoxon_signed_rank_p_value(finite)
    ci_low, ci_high = bootstrap_median_ci(
        finite,
        iterations=int(bootstrap_iterations),
        confidence_level=float(confidence_level),
        rng=np.random.default_rng(0),
    )
    std_difference = float(np.std(finite, ddof=1)) if finite.size > 1 else None
    return paired_rows, {
        "status": "computed" if p_value is not None else "skipped",
        "skip_reason": None if p_value is not None else "wilcoxon_unavailable",
        "n": int(finite.size),
        "statistic": _round(_safe_float(w_plus)),
        "p_value": _round(_safe_float(p_value)),
        "effect_size": _round(_safe_float(rank_biserial)),
        "ci_low": _round(ci_low),
        "ci_high": _round(ci_high),
        "mean_difference": _round(float(np.mean(finite))),
        "median_difference": _round(float(np.median(finite))),
        "std_difference": _round(std_difference),
        "test_method": test_method,
        "confidence_level": float(confidence_level),
        "bootstrap_iterations": int(bootstrap_iterations),
        "difference_definition": "post_chaser_quadrant_occ - pre_chaser_quadrant_occ",
    }


def query_cra_quadrant_occupancy_density(
    context: ViewerContext,
    *,
    bandwidth: float = 0.05,
    bootstrap_iterations: int = 10000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    rows = load_optional_table_rows(context, CRA_QUADRANT_OCCUPANCY_TABLE)
    if not rows:
        return {
            "available": False,
            "table_name": CRA_QUADRANT_OCCUPANCY_TABLE,
            "row_count": 0,
            "message": "No chaser quadrant occupancy export table found for this cohort.",
            "chance": 0.25,
            "rows": [],
            "fish_phase_rows": [],
            "quadrant_rows": [],
            "density_rows": [],
            "paired_rows": [],
            "statistics": None,
        }

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        value = _safe_float(row.get("occupancy_fraction"))
        normalized_rows.append(
            {
                "recording_id": row.get("recording_id"),
                "fish_id": row.get("fish_id"),
                "phase_axis_index": _safe_int(row.get("phase_axis_index")),
                "phase_label": row.get("phase_label"),
                "phase_kind": _phase_kind(row),
                "source_window_label": row.get("source_window_label"),
                "quadrant_code": _safe_int(row.get("quadrant_code")),
                "quadrant_id": row.get("quadrant_id"),
                "quadrant_label": row.get("quadrant_label") or row.get("quadrant_id"),
                "display_order": _safe_int(row.get("display_order")),
                "occupancy_fraction": _round(value),
                "fraction_of_epoch": _round(_safe_float(row.get("fraction_of_epoch"))),
                "frame_count": _safe_int(row.get("frame_count")),
                "valid_frame_count": _safe_int(row.get("valid_frame_count")),
                "quadrant_valid_frame_count": _safe_int(row.get("quadrant_valid_frame_count")),
                "missing_frame_count": _safe_int(row.get("missing_frame_count")),
                "out_of_bounds_frame_count": _safe_int(row.get("out_of_bounds_frame_count")),
                "tracking_dropout_fraction": _round(_safe_float(row.get("tracking_dropout_fraction"))),
                "chaser_quadrant_code": _safe_int(row.get("chaser_quadrant_code")),
                "chaser_quadrant_label": row.get("chaser_quadrant_label"),
                "chaser_quadrant_occ": _round(_safe_float(row.get("chaser_quadrant_occ"))),
                "is_chaser_quadrant": bool(row.get("is_chaser_quadrant")),
                "series_role": "chaser" if bool(row.get("is_chaser_quadrant")) else "non_chaser",
                "source_cra_primary_endpoint_path": row.get("source_cra_primary_endpoint_path"),
                "source_component_fingerprint": row.get("source_component_fingerprint"),
            }
        )

    grouped_quadrants: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        grouped_quadrants[
            (
                row.get("phase_axis_index"),
                row.get("phase_label"),
                row.get("quadrant_code"),
                row.get("quadrant_id"),
                row.get("quadrant_label"),
                row.get("display_order"),
            )
        ].append(row)

    quadrant_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped_quadrants.items():
        phase_axis_index, phase_label, quadrant_code, quadrant_id, quadrant_label, display_order = key
        values = _values(group_rows, "occupancy_fraction")
        stats = _summary(values)
        chaser_count = sum(1 for row in group_rows if bool(row.get("is_chaser_quadrant")))
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        quadrant_rows.append(
            {
                "phase_axis_index": phase_axis_index,
                "phase_label": phase_label,
                "phase_kind": _phase_kind({"phase_label": phase_label}),
                "quadrant_code": quadrant_code,
                "quadrant_id": quadrant_id,
                "quadrant_label": quadrant_label,
                "display_order": display_order,
                "recording_count": len(recording_ids),
                "chaser_recording_count": chaser_count,
                **stats,
            }
        )
    quadrant_rows.sort(
        key=lambda row: (
            _sort_int(row.get("phase_axis_index")),
            _sort_int(row.get("display_order")),
            str(row.get("quadrant_id") or ""),
        )
    )

    by_fish_phase: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        by_fish_phase[(str(row.get("recording_id") or ""), row.get("phase_label"))].append(row)
    fish_phase_rows: list[dict[str, Any]] = []
    for (recording_id, phase_label), group_rows in sorted(by_fish_phase.items()):
        if not recording_id:
            continue
        ordered = sorted(group_rows, key=lambda row: _sort_int(row.get("display_order")))
        chaser_rows = [row for row in ordered if bool(row.get("is_chaser_quadrant"))]
        nonchaser_values = [
            _safe_float(row.get("occupancy_fraction"))
            for row in ordered
            if not bool(row.get("is_chaser_quadrant")) and _safe_float(row.get("occupancy_fraction")) is not None
        ]
        base = ordered[0]
        fish_phase_row = {
            "recording_id": recording_id,
            "fish_id": base.get("fish_id"),
            "phase_axis_index": base.get("phase_axis_index"),
            "phase_label": phase_label,
            "phase_kind": base.get("phase_kind"),
            "chaser_quadrant": chaser_rows[0].get("chaser_quadrant_label") if chaser_rows else None,
            "chaser_quadrant_occ": chaser_rows[0].get("occupancy_fraction") if chaser_rows else None,
            "nonchaser_occ_pooled": [_round(value) for value in nonchaser_values],
            "valid_frame_count": base.get("valid_frame_count"),
            "quadrant_valid_frame_count": base.get("quadrant_valid_frame_count"),
            "tracking_dropout_fraction": base.get("tracking_dropout_fraction"),
            "source_cra_primary_endpoint_path": base.get("source_cra_primary_endpoint_path"),
        }
        for row in ordered:
            key = str(row.get("quadrant_id") or "")
            if key:
                fish_phase_row[f"{key}_occ"] = row.get("occupancy_fraction")
        fish_phase_rows.append(fish_phase_row)

    phases = sorted(
        {
            (
                _safe_int(row.get("phase_axis_index")) if _safe_int(row.get("phase_axis_index")) is not None else 999,
                str(row.get("phase_label") or ""),
                str(row.get("phase_kind") or ""),
            )
            for row in normalized_rows
            if row.get("phase_label") is not None
        }
    )
    grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    density_rows: list[dict[str, Any]] = []
    for _phase_axis_index, phase_label, _phase_kind_value in phases:
        phase_rows = [row for row in normalized_rows if row.get("phase_label") == phase_label]
        chaser_values = [
            float(row["occupancy_fraction"])
            for row in phase_rows
            if bool(row.get("is_chaser_quadrant")) and _safe_float(row.get("occupancy_fraction")) is not None
        ]
        nonchaser_values = [
            float(row["occupancy_fraction"])
            for row in phase_rows
            if not bool(row.get("is_chaser_quadrant")) and _safe_float(row.get("occupancy_fraction")) is not None
        ]
        density_rows.extend(
            _gaussian_kde_rows(
                chaser_values,
                phase_label=phase_label,
                series_role="chaser",
                bandwidth=float(bandwidth),
                grid=grid,
            )
        )
        density_rows.extend(
            _gaussian_kde_rows(
                nonchaser_values,
                phase_label=phase_label,
                series_role="non_chaser",
                bandwidth=float(bandwidth),
                grid=grid,
            )
        )

    paired_rows, statistics = _paired_chaser_occupancy_stats(
        normalized_rows,
        bootstrap_iterations=int(bootstrap_iterations),
        confidence_level=float(confidence_level),
    )

    return {
        "available": True,
        "table_name": CRA_QUADRANT_OCCUPANCY_TABLE,
        "row_count": len(normalized_rows),
        "recording_count": len({str(row.get("recording_id")) for row in normalized_rows if row.get("recording_id")}),
        "chance": 0.25,
        "x_domain": [0.0, 1.0],
        "kde": {
            "bandwidth": float(bandwidth),
            "bandwidth_rule": "fixed occupancy-fraction bandwidth shared across pre/post panels",
            "grid_size": int(grid.size),
        },
        "phases": [
            {"phase_axis_index": phase_axis_index, "phase_label": phase_label, "phase_kind": phase_kind_value}
            for phase_axis_index, phase_label, phase_kind_value in phases
        ],
        "rows": normalized_rows,
        "fish_phase_rows": fish_phase_rows,
        "quadrant_rows": quadrant_rows,
        "density_rows": density_rows,
        "paired_rows": paired_rows,
        "statistics": statistics,
        "inference_note": "Paired Wilcoxon uses one chaser-quadrant pre/post value per recording; pooled non-chaser quadrants are descriptive.",
    }


def query_cra_near_field_object_phase(
    context: ViewerContext,
    *,
    metric: str = "near_zone_occupancy_fraction",
    stat: str = "mean",
    object_role: str | None = None,
    include_recordings: bool = False,
) -> dict[str, Any]:
    if metric not in CRA_NEAR_FIELD_OBJECT_PHASE_METRICS:
        raise ValueError(f"Unsupported chaser near-field metric: {metric}")
    if stat not in {"mean", "median"}:
        raise ValueError("stat must be one of: mean, median")
    rows = load_optional_table_rows(context, CRA_NEAR_FIELD_OBJECT_PHASE_TABLE)
    if object_role:
        rows = [row for row in rows if row.get("object_role") == object_role]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            _safe_int(row.get("phase_axis_index")),
            row.get("phase_label"),
            row.get("object_role"),
            _safe_int(row.get("object_index")),
            row.get("raw_color_hex"),
        )].append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        phase_axis_index, phase_label, role, object_index, raw_color_hex = key
        values = _values(group_rows, metric)
        stats = _summary(values)
        value = stats["median"] if stat == "median" else stats["mean"]
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        out_rows.append(
            {
                "phase_axis_index": phase_axis_index,
                "phase_label": phase_label,
                "object_role": role,
                "object_index": object_index,
                "raw_color_hex": raw_color_hex,
                "value": value,
                "stat": stat,
                "recording_count": len(recording_ids),
                **stats,
            }
        )
    role_order = {"aggressive": 0, "inert": 1, "random_non_chasing": 2, "unknown": 3}
    out_rows.sort(
        key=lambda row: (
            _sort_int(row.get("phase_axis_index")),
            role_order.get(str(row.get("object_role")), 99),
            _sort_int(row.get("object_index"), 0),
        )
    )
    response: dict[str, Any] = {
        "metric": metric,
        "metric_label": CRA_NEAR_FIELD_OBJECT_PHASE_METRICS[metric],
        "stat": stat,
        "rows": out_rows,
    }
    if include_recordings:
        response["per_recording"] = [
            {
                "recording_id": row.get("recording_id"),
                "fish_id": row.get("fish_id"),
                "phase_axis_index": _safe_int(row.get("phase_axis_index")),
                "phase_label": row.get("phase_label"),
                "object_role": row.get("object_role"),
                "object_index": _safe_int(row.get("object_index")),
                "raw_color_hex": row.get("raw_color_hex"),
                "value": _safe_float(row.get(metric)),
                "valid_distance_count": _safe_int(row.get("valid_distance_count")),
                "tracking_dropout_fraction": _safe_float(row.get("tracking_dropout_fraction")),
                "source_cra_near_field_path": row.get("source_cra_near_field_path"),
            }
            for row in rows
        ]
    return response


def _profile_pairwise_near_field_summary_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    chaser_phase_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive profile-specific role contrasts from generic near-field rows."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in chaser_phase_rows:
        recording_id = str(row.get("recording_id") or "")
        role = str(row.get("object_role") or row.get("behavior_class") or "")
        phase_kind = _phase_kind(row)
        if (
            recording_id
            and role in {"aggressive", "inert"}
            and phase_kind in {"pre", "post"}
        ):
            grouped[(recording_id, role, phase_kind)].append(row)

    def mean_value(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        values = [_safe_float(row.get(key)) for row in rows]
        finite = [float(value) for value in values if value is not None]
        return sum(finite) / float(len(finite)) if finite else None

    out: list[dict[str, Any]] = []
    metric_sources = {
        "approach_p05": "approach_p05_mm",
        "approach_p10": "approach_p10_mm",
        "nearzone_occ": "near_zone_occupancy_fraction",
        "nearzone_entry_rate": "near_zone_entry_rate_per_min",
    }
    for source in summary_rows:
        row = dict(source)
        recording_id = str(row.get("recording_id") or "")
        for role, suffix in (("aggressive", "agg"), ("inert", "inert")):
            pre_rows = grouped.get((recording_id, role, "pre"), [])
            post_rows = grouped.get((recording_id, role, "post"), [])
            for output_prefix, source_key in metric_sources.items():
                pre = mean_value(pre_rows, source_key)
                post = mean_value(post_rows, source_key)
                row.setdefault(
                    f"{output_prefix}_delta_{suffix}",
                    None if pre is None or post is None else post - pre,
                )
        for output_prefix in metric_sources:
            aggressive = _safe_float(row.get(f"{output_prefix}_delta_agg"))
            inert = _safe_float(row.get(f"{output_prefix}_delta_inert"))
            row.setdefault(
                f"{output_prefix}_specificity",
                None if aggressive is None or inert is None else aggressive - inert,
            )
        out.append(row)
    return out


def query_cra_near_field_summary(
    context: ViewerContext,
    *,
    metric: str | None = None,
    endpoint_status: str | None = None,
    include_rows: bool = True,
) -> dict[str, Any]:
    if metric is not None and metric not in CRA_NEAR_FIELD_SUMMARY_METRICS:
        raise ValueError(f"Unsupported chaser near-field summary metric: {metric}")
    rows = _profile_pairwise_near_field_summary_rows(
        load_optional_table_rows(context, CRA_NEAR_FIELD_SUMMARY_TABLE),
        load_optional_table_rows(context, CRA_NEAR_FIELD_OBJECT_PHASE_TABLE),
    )
    if endpoint_status:
        rows = [row for row in rows if row.get("endpoint_status") == endpoint_status]

    available_metric_names = [
        metric_name
        for metric_name in CRA_NEAR_FIELD_SUMMARY_METRICS
        if any(row.get(metric_name) is not None for row in rows)
    ]
    metrics = [metric] if metric else available_metric_names
    metric_rows: list[dict[str, Any]] = []
    for metric_name in metrics:
        values = _values(rows, metric_name)
        metric_rows.append(
            {
                "metric": metric_name,
                "metric_label": CRA_NEAR_FIELD_SUMMARY_METRICS[metric_name],
                **_summary(values),
            }
        )

    response: dict[str, Any] = {
        "row_count": len(rows),
        "metrics": metric_rows,
        "statuses": sorted({str(row.get("endpoint_status")) for row in rows if row.get("endpoint_status")}),
    }
    if include_rows:
        response["rows"] = [
            {
                "recording_id": row.get("recording_id"),
                "fish_id": row.get("fish_id"),
                "endpoint_status": row.get("endpoint_status"),
                "aggressive_color": row.get("aggressive_color"),
                "inert_color": row.get("inert_color"),
                "approach_p05_delta_agg": _safe_float(row.get("approach_p05_delta_agg")),
                "approach_p05_delta_inert": _safe_float(row.get("approach_p05_delta_inert")),
                "approach_p05_specificity": _safe_float(row.get("approach_p05_specificity")),
                "approach_p10_delta_agg": _safe_float(row.get("approach_p10_delta_agg")),
                "approach_p10_delta_inert": _safe_float(row.get("approach_p10_delta_inert")),
                "approach_p10_specificity": _safe_float(row.get("approach_p10_specificity")),
                "nearzone_occ_delta_agg": _safe_float(row.get("nearzone_occ_delta_agg")),
                "nearzone_occ_delta_inert": _safe_float(row.get("nearzone_occ_delta_inert")),
                "nearzone_occ_specificity": _safe_float(row.get("nearzone_occ_specificity")),
                "nearzone_entry_rate_delta_agg": _safe_float(row.get("nearzone_entry_rate_delta_agg")),
                "nearzone_entry_rate_delta_inert": _safe_float(row.get("nearzone_entry_rate_delta_inert")),
                "nearzone_entry_rate_specificity": _safe_float(row.get("nearzone_entry_rate_specificity")),
                "thigmotaxis_frac_pre": _safe_float(row.get("thigmotaxis_frac_pre")),
                "thigmotaxis_frac_post": _safe_float(row.get("thigmotaxis_frac_post")),
                "frac_tracking_dropout_pre": _safe_float(row.get("frac_tracking_dropout_pre")),
                "frac_tracking_dropout_post": _safe_float(row.get("frac_tracking_dropout_post")),
                "geometry_status": row.get("geometry_status"),
                "arena_shape": row.get("arena_shape"),
                "source_cra_near_field_path": row.get("source_cra_near_field_path"),
                "source_cra_primary_endpoint_path": row.get("source_cra_primary_endpoint_path"),
                "source_component_fingerprint": row.get("source_component_fingerprint"),
            }
            for row in rows
        ]
    return response


def _aggregate_fish_level_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    index_key: str,
    max_x: float | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        x_value = _safe_float(row.get(x_key))
        y_value = _safe_float(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        if max_x is not None and x_value > float(max_x):
            continue
        key = (
            _safe_int(row.get("phase_axis_index")),
            row.get("phase_label"),
            _phase_kind(row),
            row.get("object_role"),
            _safe_int(row.get(index_key)),
            x_value,
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        phase_axis_index, phase_label, phase_kind, object_role, index_value, x_value = key
        values = _values(group_rows, y_key)
        stats = _summary(values)
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        first = group_rows[0]
        out.append(
            {
                "phase_axis_index": phase_axis_index,
                "phase_label": phase_label,
                "phase_kind": phase_kind,
                "object_role": object_role,
                "raw_color_hex": first.get("raw_color_hex"),
                index_key: index_value,
                x_key: _round(_safe_float(x_value)),
                "recording_count": len(recording_ids),
                "value_metric": y_key,
                **stats,
            }
        )
    out.sort(
        key=lambda row: (
            _sort_int(row.get("phase_axis_index")),
            str(row.get("object_role") or ""),
            _sort_int(row.get(index_key)),
            float(row.get(x_key) or 0.0),
        )
    )
    return out


def query_cra_near_field_curves(
    context: ViewerContext,
    *,
    max_cdf_threshold_mm: float = 15.0,
) -> dict[str, Any]:
    radial_rows = load_optional_table_rows(context, CRA_NEAR_FIELD_RADIAL_TABLE)
    cdf_rows = load_optional_table_rows(context, CRA_NEAR_FIELD_CDF_TABLE)
    if not radial_rows and not cdf_rows:
        return {
            "available": False,
            "radial_table_name": CRA_NEAR_FIELD_RADIAL_TABLE,
            "cdf_table_name": CRA_NEAR_FIELD_CDF_TABLE,
            "radial_row_count": 0,
            "cdf_row_count": 0,
            "radial_rows": [],
            "cdf_rows": [],
            "message": "No chaser near-field radial/CDF exports found for this cohort.",
        }

    radial_curve_rows = _aggregate_fish_level_curve(
        radial_rows,
        x_key="radial_bin_center_mm",
        y_key="radial_density_per_mm2",
        index_key="radial_bin_index",
    )
    radial_wall_excluded_rows = _aggregate_fish_level_curve(
        radial_rows,
        x_key="radial_bin_center_mm",
        y_key="radial_density_wall_excluded_per_mm2",
        index_key="radial_bin_index",
    )
    cdf_curve_rows = _aggregate_fish_level_curve(
        cdf_rows,
        x_key="distance_threshold_mm",
        y_key="cdf_fraction",
        index_key="cdf_threshold_index",
        max_x=float(max_cdf_threshold_mm),
    )
    return {
        "available": True,
        "radial_table_name": CRA_NEAR_FIELD_RADIAL_TABLE,
        "cdf_table_name": CRA_NEAR_FIELD_CDF_TABLE,
        "radial_row_count": len(radial_rows),
        "cdf_row_count": len(cdf_rows),
        "recording_count": len(
            {
                str(row.get("recording_id"))
                for row in [*radial_rows, *cdf_rows]
                if row.get("recording_id")
            }
        ),
        "max_cdf_threshold_mm": float(max_cdf_threshold_mm),
        "radial_rows": radial_curve_rows,
        "radial_wall_excluded_rows": radial_wall_excluded_rows,
        "cdf_rows": cdf_curve_rows,
        "summary_source": "fish_level_mean_by_recording",
        "inference_note": (
            "Curves are aggregated from one value per fish/recording/bin. Frame counts are not pooled across fish."
        ),
    }


def query_egocentric_summary(
    context: ViewerContext,
    *,
    metric: str = "mean_alignment_cos",
    stat: str = "mean",
    include_recordings: bool = False,
) -> dict[str, Any]:
    if metric not in EGOCENTRIC_METRICS:
        raise ValueError(f"Unsupported egocentric metric: {metric}")
    if stat not in {"mean", "median"}:
        raise ValueError("stat must be one of: mean, median")
    rows = _enrich_chaser_behavior_rows(
        load_table_rows(context, EGOCENTRIC_SUMMARY_TABLE),
        load_optional_table_rows(context, CRA_OBJECT_PHASE_TABLE),
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            _safe_int(row.get("chaser_index")),
            row.get("behavior_class"),
        )].append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        window_index, window_label, chaser_index, behavior_class = key
        values = _values(group_rows, metric)
        stats = _summary(values)
        value = stats["median"] if stat == "median" else stats["mean"]
        recording_ids = sorted({str(row.get("recording_id")) for row in group_rows if row.get("recording_id")})
        raw_color_hex, color_status = _consistent_raw_color(group_rows)
        out_rows.append(
            {
                "window_index": window_index,
                "window_label": window_label,
                "chaser_index": chaser_index,
                "behavior_class": behavior_class,
                "raw_color_hex": raw_color_hex,
                "color_status": color_status,
                "value": value,
                "stat": stat,
                "recording_count": len(recording_ids),
                **stats,
            }
        )
    out_rows.sort(key=lambda row: (_sort_int(row.get("window_index")), _sort_int(row.get("chaser_index"), 0)))
    response: dict[str, Any] = {
        "metric": metric,
        "metric_label": EGOCENTRIC_METRICS[metric],
        "stat": stat,
        "rows": out_rows,
    }
    if include_recordings:
        response["per_recording"] = [
            {
                "recording_id": row.get("recording_id"),
                "window_index": _safe_int(row.get("window_index")),
                "window_label": row.get("window_label"),
                "chaser_index": _safe_int(row.get("chaser_index")),
                "behavior_class": row.get("behavior_class"),
                "value": _safe_float(row.get(metric)),
                "egocentric_component_name": row.get("egocentric_component_name"),
            }
            for row in rows
        ]
    return response


def query_egocentric_histogram(
    context: ViewerContext,
    *,
    window_label: str | None = None,
    chaser_index: int | None = None,
) -> dict[str, Any]:
    rows = load_table_rows(context, EGOCENTRIC_HISTOGRAM_TABLE)
    if window_label:
        rows = [row for row in rows if row.get("window_label") == window_label]
    if chaser_index is not None:
        rows = [row for row in rows if _safe_int(row.get("chaser_index")) == chaser_index]

    grouped: dict[tuple[Any, ...], int] = defaultdict(int)
    meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            _safe_int(row.get("chaser_index")),
            _safe_int(row.get("distance_bin_index")),
            _safe_float(row.get("distance_bin_left_mm")),
            _safe_float(row.get("distance_bin_right_mm")),
            _safe_float(row.get("distance_bin_center_mm")),
            _safe_float(row.get("distance_bin_width_mm")),
            _safe_int(row.get("bearing_bin_index")),
            _safe_float(row.get("bearing_bin_left_deg")),
            _safe_float(row.get("bearing_bin_right_deg")),
            _safe_float(row.get("bearing_bin_center_deg")),
            _safe_float(row.get("bearing_bin_width_deg")),
        )
        grouped[key] += int(_safe_int(row.get("hist_count")) or 0)
        meta.setdefault(
            key,
            {
                "window_index": key[0],
                "window_label": key[1],
                "chaser_index": key[2],
                "distance_bin_index": key[3],
                "distance_bin_left_mm": key[4],
                "distance_bin_right_mm": key[5],
                "distance_bin_center_mm": key[6],
                "distance_bin_width_mm": key[7],
                "bearing_bin_index": key[8],
                "bearing_bin_left_deg": key[9],
                "bearing_bin_right_deg": key[10],
                "bearing_bin_center_deg": key[11],
                "bearing_bin_width_deg": key[12],
            },
        )

    totals: dict[tuple[Any, Any], int] = defaultdict(int)
    for key, count in grouped.items():
        totals[(key[1], key[2])] += int(count)

    out_rows: list[dict[str, Any]] = []
    for key, count in grouped.items():
        item = dict(meta[key])
        total = totals[(key[1], key[2])]
        probability = float(count) / float(total) if total > 0 else None
        item.update({
            "pooled_count": int(count),
            "pooled_total_count": int(total),
            "pooled_probability": _round(probability),
        })
        out_rows.append(item)
    out_rows.sort(
        key=lambda row: (
            _sort_int(row.get("window_index")),
            _sort_int(row.get("chaser_index"), 0),
            _sort_int(row.get("distance_bin_index"), 0),
            _sort_int(row.get("bearing_bin_index"), 0),
        )
    )
    return {"rows": out_rows}


def egocentric_rebin_options(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe exact coarsenings available from pooled native histogram bins."""

    if not rows:
        return {
            "native_distance_bin_width_mm": None,
            "native_bearing_bin_width_deg": None,
            "distance": [{"factor": 1, "width": None, "label": "Native"}],
            "bearing": [{"factor": 1, "width": None, "label": "Native"}],
        }

    def _native_axis(axis: str) -> tuple[float, int]:
        width_key = f"{axis}_bin_width_{'mm' if axis == 'distance' else 'deg'}"
        index_key = f"{axis}_bin_index"
        left_key = f"{axis}_bin_left_{'mm' if axis == 'distance' else 'deg'}"
        right_key = f"{axis}_bin_right_{'mm' if axis == 'distance' else 'deg'}"
        widths = {
            round(float(width), 12)
            for row in rows
            if (width := _safe_float(row.get(width_key))) is not None and width > 0.0
        }
        indices = {
            int(index)
            for row in rows
            if (index := _safe_int(row.get(index_key))) is not None
        }
        if len(widths) != 1 or not indices:
            raise ValueError(f"Cannot re-bin a non-uniform {axis} grid")
        ordered = sorted(indices)
        if ordered[0] != 0:
            raise ValueError(f"Cannot re-bin a {axis} grid that does not start at index zero")
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise ValueError(f"Cannot re-bin a discontinuous {axis} grid")
        geometry_by_index: dict[int, set[tuple[float, float]]] = defaultdict(set)
        for row in rows:
            index = _safe_int(row.get(index_key))
            left = _safe_float(row.get(left_key))
            right = _safe_float(row.get(right_key))
            if index is not None and left is not None and right is not None:
                geometry_by_index[int(index)].add(
                    (round(float(left), 12), round(float(right), 12))
                )
        if set(geometry_by_index) != set(indices) or any(
            len(geometry) != 1 for geometry in geometry_by_index.values()
        ):
            raise ValueError(f"Cannot re-bin incompatible {axis} grids")
        return float(next(iter(widths))), len(ordered)

    distance_width, distance_count = _native_axis("distance")
    bearing_width, bearing_count = _native_axis("bearing")
    distance_factors = range(1, min(distance_count, 10) + 1)
    bearing_factors = [
        factor for factor in range(1, bearing_count + 1) if bearing_count % factor == 0
    ]

    def _options(
        factors: Iterable[int],
        *,
        native_width: float,
        unit: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "factor": int(factor),
                "width": float(native_width * factor),
                "label": (
                    f"{native_width:g} {unit} (native)"
                    if factor == 1
                    else f"{native_width * factor:g} {unit} ({factor}× native)"
                ),
            }
            for factor in factors
        ]

    return {
        "native_distance_bin_width_mm": distance_width,
        "native_bearing_bin_width_deg": bearing_width,
        "distance": _options(
            distance_factors,
            native_width=distance_width,
            unit="mm",
        ),
        "bearing": _options(
            bearing_factors,
            native_width=bearing_width,
            unit="deg",
        ),
    }


def rebin_egocentric_histogram_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    distance_bin_factor: int = 1,
    bearing_bin_factor: int = 1,
) -> list[dict[str, Any]]:
    """Exactly coarsen aligned exported bins by summing counts, never interpolation."""

    distance_factor = int(distance_bin_factor)
    bearing_factor = int(bearing_bin_factor)
    if distance_factor < 1 or bearing_factor < 1:
        raise ValueError("Re-bin factors must be positive integers")
    if not rows:
        return []

    options = egocentric_rebin_options(rows)
    distance_bin_count = max(
        int(_safe_int(row.get("distance_bin_index")) or 0) for row in rows
    ) + 1
    bearing_bin_count = max(
        int(_safe_int(row.get("bearing_bin_index")) or 0) for row in rows
    ) + 1
    if distance_factor > distance_bin_count:
        raise ValueError("Distance re-bin factor exceeds the native grid")
    if bearing_factor > bearing_bin_count or bearing_bin_count % bearing_factor != 0:
        raise ValueError("Bearing re-bin factor must divide the complete circular grid")

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        distance_index = _safe_int(row.get("distance_bin_index"))
        bearing_index = _safe_int(row.get("bearing_bin_index"))
        distance_left = _safe_float(row.get("distance_bin_left_mm"))
        distance_right = _safe_float(row.get("distance_bin_right_mm"))
        bearing_left = _safe_float(row.get("bearing_bin_left_deg"))
        bearing_right = _safe_float(row.get("bearing_bin_right_deg"))
        if None in (
            distance_index,
            bearing_index,
            distance_left,
            distance_right,
            bearing_left,
            bearing_right,
        ):
            raise ValueError("Cannot re-bin histogram rows with incomplete bin geometry")
        key = (
            _safe_int(row.get("window_index")),
            row.get("window_label"),
            _safe_int(row.get("chaser_index")),
            int(distance_index) // distance_factor,
            int(bearing_index) // bearing_factor,
        )
        item = grouped.setdefault(
            key,
            {
                "window_index": key[0],
                "window_label": key[1],
                "chaser_index": key[2],
                "distance_bin_index": key[3],
                "distance_bin_left_mm": float(distance_left),
                "distance_bin_right_mm": float(distance_right),
                "bearing_bin_index": key[4],
                "bearing_bin_left_deg": float(bearing_left),
                "bearing_bin_right_deg": float(bearing_right),
                "pooled_count": 0,
            },
        )
        item["distance_bin_left_mm"] = min(
            float(item["distance_bin_left_mm"]), float(distance_left)
        )
        item["distance_bin_right_mm"] = max(
            float(item["distance_bin_right_mm"]), float(distance_right)
        )
        item["bearing_bin_left_deg"] = min(
            float(item["bearing_bin_left_deg"]), float(bearing_left)
        )
        item["bearing_bin_right_deg"] = max(
            float(item["bearing_bin_right_deg"]), float(bearing_right)
        )
        item["pooled_count"] += int(_safe_int(row.get("pooled_count")) or 0)

    totals: dict[tuple[Any, Any], int] = defaultdict(int)
    for item in grouped.values():
        totals[(item["window_label"], item["chaser_index"])] += int(
            item["pooled_count"]
        )

    output: list[dict[str, Any]] = []
    for item in grouped.values():
        distance_left = float(item["distance_bin_left_mm"])
        distance_right = float(item["distance_bin_right_mm"])
        bearing_left = float(item["bearing_bin_left_deg"])
        bearing_right = float(item["bearing_bin_right_deg"])
        total = totals[(item["window_label"], item["chaser_index"])]
        count = int(item["pooled_count"])
        output.append(
            {
                **item,
                "distance_bin_center_mm": (distance_left + distance_right) / 2.0,
                "distance_bin_width_mm": distance_right - distance_left,
                "bearing_bin_center_deg": (bearing_left + bearing_right) / 2.0,
                "bearing_bin_width_deg": bearing_right - bearing_left,
                "pooled_total_count": total,
                "pooled_probability": _round(count / total if total > 0 else None),
                "distance_bin_factor": distance_factor,
                "bearing_bin_factor": bearing_factor,
                "native_distance_bin_width_mm": options[
                    "native_distance_bin_width_mm"
                ],
                "native_bearing_bin_width_deg": options[
                    "native_bearing_bin_width_deg"
                ],
            }
        )
    output.sort(
        key=lambda row: (
            _sort_int(row.get("window_index")),
            _sort_int(row.get("chaser_index"), 0),
            _sort_int(row.get("distance_bin_index"), 0),
            _sort_int(row.get("bearing_bin_index"), 0),
        )
    )
    return output


def query_recordings(context: ViewerContext) -> dict[str, Any]:
    spatial = load_optional_table_rows(context, SPATIAL_TABLE)
    chaser = load_optional_table_rows(context, CHASER_SUMMARY_TABLE)
    speed = load_optional_table_rows(context, EPOCH_BEHAVIOR_TABLE)
    cra_summary = _profile_pairwise_quadrant_summary_rows(
        load_optional_table_rows(context, CRA_SUMMARY_TABLE),
        load_optional_table_rows(context, CRA_OBJECT_PHASE_TABLE),
    )
    cra_near_field_summary = _profile_pairwise_near_field_summary_rows(
        load_optional_table_rows(context, CRA_NEAR_FIELD_SUMMARY_TABLE),
        load_optional_table_rows(context, CRA_NEAR_FIELD_OBJECT_PHASE_TABLE),
    )
    egocentric = load_optional_table_rows(context, EGOCENTRIC_SUMMARY_TABLE)
    records: dict[str, dict[str, Any]] = {}

    for row in spatial:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        record = records.setdefault(
            recording_id,
            {
                "recording_id": recording_id,
                "zarr_path": row.get("zarr_path"),
                "zone_set_id": row.get("zone_set_id"),
            },
        )
        if _safe_int(row.get("zone_index")) == 0:
            label = str(row.get("window_label") or "")
            record[f"{label}_coverage_pct"] = _round(_safe_float(row.get("coverage_pct")), 3)

    for row in chaser:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        record = records.setdefault(
            recording_id,
            {
                "recording_id": recording_id,
                "zarr_path": row.get("zarr_path"),
            },
        )
        label = str(row.get("window_label") or "")
        chaser_idx = _safe_int(row.get("chaser_index"))
        if chaser_idx is not None:
            record[f"{label}_chaser_{chaser_idx}_p50_mm"] = _round(_safe_float(row.get("p50_distance_mm")), 3)
            record[f"{label}_chaser_{chaser_idx}_frac_within"] = _round(
                _safe_float(row.get("fraction_within_threshold")),
                4,
            )

    for row in speed:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        record = records.setdefault(
            recording_id,
            {
                "recording_id": recording_id,
                "zarr_path": row.get("zarr_path"),
            },
        )
        label = str(row.get("window_label") or "")
        record[f"{label}_mean_speed_mm_s"] = _round(_safe_float(row.get("mean_speed_mm_s")), 4)
        record[f"{label}_median_speed_mm_s"] = _round(_safe_float(row.get("median_speed_mm_s")), 4)
        record[f"{label}_speed_samples"] = _safe_int(row.get("speed_sample_count"))

    for row in cra_summary:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        record = records.setdefault(
            recording_id,
            {
                "recording_id": recording_id,
                "zarr_path": row.get("zarr_path"),
            },
        )
        record["cra_endpoint_status"] = row.get("endpoint_status")
        record["cra_delta_agg_mm"] = _round(_safe_float(row.get("delta_agg")), 4)
        record["cra_delta_inert_mm"] = _round(_safe_float(row.get("delta_inert")), 4)
        record["cra_specificity_distance_mm"] = _round(_safe_float(row.get("specificity_distance")), 4)
        record["cra_delta_occ_agg"] = _round(_safe_float(row.get("delta_occ_agg")), 4)
        record["cra_delta_occ_inert"] = _round(_safe_float(row.get("delta_occ_inert")), 4)
        record["cra_specificity_occupancy"] = _round(_safe_float(row.get("specificity_occupancy")), 4)
        record["cra_post_aggressive_quadrant"] = row.get("post_aggressive_quadrant")

    for row in cra_near_field_summary:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        record = records.setdefault(
            recording_id,
            {
                "recording_id": recording_id,
                "zarr_path": row.get("zarr_path"),
            },
        )
        record["cra_near_field_status"] = row.get("endpoint_status")
        record["cra_near_field_approach_p05_specificity_mm"] = _round(
            _safe_float(row.get("approach_p05_specificity")),
            4,
        )
        record["cra_near_field_nearzone_occ_specificity"] = _round(
            _safe_float(row.get("nearzone_occ_specificity")),
            4,
        )
        record["cra_near_field_entry_rate_specificity"] = _round(
            _safe_float(row.get("nearzone_entry_rate_specificity")),
            4,
        )
        record["cra_near_field_geometry_status"] = row.get("geometry_status")
        record["cra_near_field_arena_shape"] = row.get("arena_shape")

    for row in egocentric:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        record = records.setdefault(
            recording_id,
            {
                "recording_id": recording_id,
                "zarr_path": row.get("zarr_path"),
            },
        )
        label = str(row.get("window_label") or "")
        chaser_idx = _safe_int(row.get("chaser_index"))
        if chaser_idx is not None:
            record[f"{label}_chaser_{chaser_idx}_alignment"] = _round(
                _safe_float(row.get("mean_alignment_cos")),
                4,
            )
            record[f"{label}_chaser_{chaser_idx}_front_frac"] = _round(
                _safe_float(row.get("fraction_front_45")),
                4,
            )

    rows = sorted(records.values(), key=lambda row: str(row.get("recording_id") or ""))
    return {"rows": rows, "row_count": len(rows)}


def query_provenance(context: ViewerContext) -> dict[str, Any]:
    summary = query_export_summary(context)
    schemas = {}
    for table_name in CHASER_TABLES:
        try:
            schemas[table_name] = _table_schema(context, table_name)
        except FileNotFoundError:
            if table_name not in OPTIONAL_CHASER_TABLES:
                raise
    stats_run_id = resolve_statistics_run_id(context)
    if stats_run_id is not None:
        schemas[STATISTICS_TABLE] = _table_schema(context, STATISTICS_TABLE, export_run_id=stats_run_id)
    return {"summary": summary, "schemas": schemas}
