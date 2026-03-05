#!/usr/bin/env python3
"""Aggregate an eye-mask training data card from registry profile/performance rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils import plot_eye_mask_training_data_card as plot_data_card


SCHEMA_NAME = "eye_mask_training_data_card"
SCHEMA_VERSION = "v1"
COMPOSITION_FIELDS = (
    "rig_id",
    "camera_id",
    "arena_id",
    "dish_design",
    "canvas_name",
    "protocol_name",
    "method",
    "stage_group",
)


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    zarr_path: Path
    manifest_row: dict[str, Any]
    resolved_by_registry: bool


@dataclass(frozen=True)
class SubjectLineageCoverage:
    manifest_dataset_count: int
    lineage_covered_dataset_count: Optional[int]
    missing_lineage_dataset_ids: tuple[str, ...]
    coverage_unavailable_reason: Optional[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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


def _numeric_histogram(
    values: Sequence[Optional[float]],
    *,
    bins: int = 24,
) -> Optional[dict[str, Any]]:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    bins = max(1, int(bins))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v <= min_v:
        edges = np.asarray([min_v - 0.5, max_v + 0.5], dtype=np.float64)
        counts = np.asarray([int(arr.size)], dtype=np.int64)
    else:
        edges = np.linspace(min_v, max_v, bins + 1, dtype=np.float64)
        counts, _ = np.histogram(arr, bins=edges)

    return {
        "bin_edges": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in counts.astype(np.int64).tolist()],
        "source_value_count": int(arr.size),
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


def _parse_iso_ts(value: Any) -> datetime:
    text = _normalize_text(value)
    if text is None:
        return datetime.min
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return datetime.min


def _value_at_path(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    parsed = _parse_json_mapping(value)
    if isinstance(parsed, Mapping):
        return parsed
    return None


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Manifest is not a JSON object: {path}")

    rows = payload.get("datasets")
    merged_export = payload.get("merged_export")
    has_source_rows = (
        isinstance(merged_export, Mapping)
        and isinstance(merged_export.get("source_datasets"), list)
        and bool(merged_export.get("source_datasets"))
    )
    if (not isinstance(rows, list) or not rows) and not has_source_rows:
        raise ValueError(f"Manifest has no datasets: {path}")
    return dict(payload)


def _manifest_source_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    merged_export = manifest.get("merged_export")
    if isinstance(merged_export, Mapping):
        source_rows = merged_export.get("source_datasets")
        if isinstance(source_rows, list) and source_rows:
            rows: list[dict[str, Any]] = []
            for row in source_rows:
                if isinstance(row, Mapping):
                    rows.append(dict(row))
            if rows:
                return rows

    rows = manifest.get("datasets")
    if not isinstance(rows, list):
        return []

    parsed: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            parsed.append(dict(row))
    return parsed


def _normalize_manifest_stem(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def _default_output_path(
    manifest_path: Optional[Path],
    *,
    set_id: Optional[str],
) -> Path:
    base = _normalize_manifest_stem(set_id or (manifest_path.stem if manifest_path else ""))
    base = base or "eye_mask_training_data_card"
    if manifest_path is not None:
        return manifest_path.parent / f"{base}.data_card.json"
    return Path.cwd() / f"{base}.data_card.json"


def _default_plot_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.plots"


def _add_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    argv.extend([str(flag), str(value)])


def _positive_threshold(value: Any) -> Optional[float]:
    threshold = _as_float(value)
    if threshold is None or threshold <= 0:
        return None
    return float(threshold)


def _low_area_report_rows(
    *,
    card: Mapping[str, Any],
    min_eye_area_p50: Optional[float],
    min_union_area_p50: Optional[float],
) -> list[dict[str, Any]]:
    eye_threshold = _positive_threshold(min_eye_area_p50)
    union_threshold = _positive_threshold(min_union_area_p50)
    if eye_threshold is None and union_threshold is None:
        return []

    audit = _as_mapping(card.get("audit_freshness")) or {}
    source_refs_raw = audit.get("source_run_refs")
    if not isinstance(source_refs_raw, list):
        return []

    flagged: list[dict[str, Any]] = []
    for row in source_refs_raw:
        if not isinstance(row, Mapping):
            continue
        left_area = _as_float(row.get("left_area_p50"))
        right_area = _as_float(row.get("right_area_p50"))
        union_area = _as_float(row.get("union_area_p50"))

        reasons: list[str] = []
        if eye_threshold is not None:
            left_bad = left_area is not None and left_area < eye_threshold
            right_bad = right_area is not None and right_area < eye_threshold
            if left_bad or right_bad:
                reasons.append("eye_area_p50_below_threshold")
        if union_threshold is not None and union_area is not None and union_area < union_threshold:
            reasons.append("union_area_p50_below_threshold")
        if not reasons:
            continue

        flagged.append(
            {
                "dataset_id": _normalize_text(row.get("dataset_id")) or "",
                "profile_run": _normalize_text(row.get("profile_run")),
                "zarr_path": _normalize_text(row.get("zarr_path")),
                "review_state": _normalize_text(row.get("review_state")),
                "left_area_p50": left_area,
                "right_area_p50": right_area,
                "union_area_p50": union_area,
                "successful_roi_pair_rate": _as_float(row.get("successful_roi_pair_rate")),
                "reasons": reasons,
            }
        )

    flagged.sort(key=lambda item: str(item.get("dataset_id") or ""))
    return flagged


def _print_low_area_report(
    *,
    card: Mapping[str, Any],
    min_eye_area_p50: Optional[float],
    min_union_area_p50: Optional[float],
) -> None:
    eye_threshold = _positive_threshold(min_eye_area_p50)
    union_threshold = _positive_threshold(min_union_area_p50)
    if eye_threshold is None and union_threshold is None:
        return

    flagged = _low_area_report_rows(
        card=card,
        min_eye_area_p50=eye_threshold,
        min_union_area_p50=union_threshold,
    )
    dataset_count = _as_int(_value_at_path(card, ("selection", "dataset_count"))) or 0
    print(
        "Eye-mask low-area report: "
        f"datasets_flagged={len(flagged)}/{dataset_count} "
        f"min_eye_area_p50={eye_threshold} "
        f"min_union_area_p50={union_threshold}"
    )
    for row in flagged:
        reasons = ",".join(str(item) for item in row.get("reasons") or [])
        left_text = row.get("left_area_p50")
        right_text = row.get("right_area_p50")
        union_text = row.get("union_area_p50")
        print(
            "low_area\t"
            f"{row.get('dataset_id')}\t"
            f"left_area_p50={left_text if left_text is not None else '-'}\t"
            f"right_area_p50={right_text if right_text is not None else '-'}\t"
            f"union_area_p50={union_text if union_text is not None else '-'}\t"
            f"successful_roi_pair_rate={row.get('successful_roi_pair_rate')}\t"
            f"review_state={row.get('review_state') or '-'}\t"
            f"profile_run={row.get('profile_run') or '-'}\t"
            f"reasons={reasons}\t"
            f"zarr_path={row.get('zarr_path') or '-'}"
        )


def _generate_plots(
    *,
    card: Mapping[str, Any],
    output_dir: Path,
    prefix: str,
    heatmap_bin_factor: int,
) -> list[Path]:
    generator = getattr(plot_data_card, "generate_eye_mask_training_data_card_plots", None)
    if generator is None:
        raise RuntimeError("generate_eye_mask_training_data_card_plots is unavailable")
    generated = generator(
        card_payload=dict(card),
        output_dir=output_dir,
        prefix=prefix,
        heatmap_bin_factor=int(heatmap_bin_factor),
    )
    return [Path(path) for path in generated]


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
    refs: list[DatasetRef] = []
    for row in _manifest_source_rows(manifest):
        zarr_path_text = _normalize_text(row.get("zarr_path"))
        if zarr_path_text is None:
            raise ValueError("Manifest dataset row missing zarr_path.")
        zarr_path = Path(zarr_path_text)
        resolved_dataset_id = _resolve_dataset_id_from_registry(registry, zarr_path)
        dataset_id = resolved_dataset_id or _normalize_text(row.get("dataset_id")) or _normalize_text(row.get("session_uuid"))
        if dataset_id is None:
            raise ValueError(f"Unable to resolve dataset_id for zarr path: {zarr_path}")
        refs.append(
            DatasetRef(
                dataset_id=dataset_id,
                zarr_path=zarr_path,
                manifest_row=dict(row),
                resolved_by_registry=resolved_dataset_id is not None,
            )
        )
    if not refs:
        raise ValueError("Manifest contains no usable dataset rows.")
    return refs


def _load_training_set_manifest(
    registry: Registry,
    *,
    set_id: str,
) -> dict[str, Any]:
    row = registry.conn.execute(
        """
        SELECT set_id, name, task_type, query_filter, dataset_ids_json, invocation_json
        FROM training_sets
        WHERE set_id = ?
        LIMIT 1;
        """,
        (str(set_id),),
    ).fetchone()
    if row is None:
        raise ValueError(f"Training set not found: {set_id}")

    dataset_ids: list[str] = []
    raw_ids = _parse_json_mapping({"dataset_ids": row["dataset_ids_json"]})
    if raw_ids is not None:
        ids_payload = raw_ids.get("dataset_ids")
        if isinstance(ids_payload, list):
            for value in ids_payload:
                text = _normalize_text(value)
                if text is not None:
                    dataset_ids.append(text)
    if not dataset_ids:
        # Fall back to direct JSON decode for normal list payloads.
        try:
            parsed_ids = json.loads(row["dataset_ids_json"])
        except Exception:
            parsed_ids = []
        if isinstance(parsed_ids, list):
            for value in parsed_ids:
                text = _normalize_text(value)
                if text is not None:
                    dataset_ids.append(text)
    dataset_ids = list(dict.fromkeys(dataset_ids))
    if not dataset_ids:
        raise ValueError(f"Training set has no dataset ids: {set_id}")

    placeholders = ", ".join("?" for _ in dataset_ids)
    dataset_rows = registry.conn.execute(
        f"SELECT dataset_id, zarr_path FROM datasets WHERE dataset_id IN ({placeholders});",
        tuple(dataset_ids),
    ).fetchall()
    by_dataset_id: dict[str, dict[str, Any]] = {}
    for dataset_row in dataset_rows:
        dataset_id = _normalize_text(dataset_row["dataset_id"])
        if dataset_id is None:
            continue
        by_dataset_id[dataset_id] = dict(dataset_row)

    missing = [dataset_id for dataset_id in dataset_ids if dataset_id not in by_dataset_id]
    if missing:
        raise ValueError(
            "Training set contains dataset_id(s) not present in datasets table: "
            + ", ".join(sorted(missing))
        )

    datasets_payload: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        dataset_row = by_dataset_id[dataset_id]
        zarr_path = _normalize_text(dataset_row.get("zarr_path"))
        if zarr_path is None:
            raise ValueError(f"Dataset {dataset_id} has no zarr_path in registry.")
        datasets_payload.append(
            {
                "dataset_id": dataset_id,
                "zarr_path": zarr_path,
            }
        )

    query_filter_payload: dict[str, Any] = {}
    parsed_query_filter = _parse_json_mapping(row["query_filter"])
    if isinstance(parsed_query_filter, Mapping):
        query_filter_payload = dict(parsed_query_filter)

    invocation_payload = _parse_json_mapping(row["invocation_json"])
    set_version: Optional[str] = None
    if isinstance(invocation_payload, Mapping):
        set_version_value = invocation_payload.get("set_version")
        if isinstance(set_version_value, (int, str)):
            set_version = str(set_version_value)

    manifest = {
        "set_id": _normalize_text(row["set_id"]) or str(set_id),
        "set_version": set_version,
        "query_filter": query_filter_payload,
        "datasets": datasets_payload,
        "source_type": _normalize_text(row["task_type"]),
    }
    return manifest


def _table_or_view_exists(registry: Registry, name: str) -> bool:
    row = registry.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1;",
        (str(name),),
    ).fetchone()
    return row is not None


def _query_profile_rows_via_sql(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not dataset_ids:
        return []
    placeholders = ", ".join("?" for _ in dataset_ids)
    rows = registry.conn.execute(
        f"SELECT * FROM eye_mask_data_profile_latest WHERE dataset_id IN ({placeholders});",
        tuple(str(dataset_id) for dataset_id in dataset_ids),
    ).fetchall()
    return [dict(row) for row in rows]


def _select_profile_rows_registry_first(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    source = "missing_profile_view"

    query_fn = getattr(registry, "query_eye_mask_data_profile_latest", None)
    if callable(query_fn):
        try:
            queried = query_fn(dataset_ids=list(dataset_ids))
            rows = [dict(row) for row in queried]
            source = "registry_api"
        except Exception as exc:
            print(f"Eye-mask profile registry API query failed: {exc}")
            source = "registry_api_error"

    if not rows and _table_or_view_exists(registry, "eye_mask_data_profile_latest"):
        rows = _query_profile_rows_via_sql(registry, dataset_ids=dataset_ids)
        source = "registry_sql_view"

    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset_id = _normalize_text(row.get("dataset_id"))
        if dataset_id is None:
            continue
        existing = by_dataset.get(dataset_id)
        if existing is None:
            by_dataset[dataset_id] = dict(row)
            continue

        current_ts = _parse_iso_ts(_coalesce(row.get("profile_created_utc"), row.get("updated_utc"), row.get("run_created_utc")))
        existing_ts = _parse_iso_ts(
            _coalesce(existing.get("profile_created_utc"), existing.get("updated_utc"), existing.get("run_created_utc"))
        )
        if current_ts >= existing_ts:
            by_dataset[dataset_id] = dict(row)

    return by_dataset, source


def _query_eye_mask_performance_fallback(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if not dataset_ids:
        return {}
    if not _table_or_view_exists(registry, "eye_mask_performance_latest"):
        raise ValueError("eye_mask_performance_latest is unavailable for fallback reads.")

    placeholders = ", ".join("?" for _ in dataset_ids)
    sql = (
        "SELECT "
        "  emp.*, "
        "  p.rig_id AS rig_id, "
        "  p.camera_id AS camera_id, "
        "  p.arena_id AS arena_id, "
        "  p.dish_design AS dish_design, "
        "  p.canvas_name AS canvas_name, "
        "  p.protocol_name AS protocol_name, "
        "  p.genotype AS genotype, "
        "  p.dpf_at_acquisition AS dpf_at_acquisition "
        "FROM eye_mask_performance_latest emp "
        "LEFT JOIN provenance p ON p.dataset_id = emp.dataset_id "
        f"WHERE emp.dataset_id IN ({placeholders}) "
        "AND emp.stage_group IN ('refined_eye_masks_runs', 'eye_masks_runs') "
        "ORDER BY "
        "  emp.dataset_id ASC, "
        "  CASE WHEN emp.stage_group = 'refined_eye_masks_runs' THEN 0 ELSE 1 END ASC, "
        "  COALESCE(emp.run_created_utc, emp.updated_utc) DESC, "
        "  emp.run_name DESC"
    )
    rows = registry.conn.execute(sql, tuple(str(dataset_id) for dataset_id in dataset_ids)).fetchall()

    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        dataset_id = _normalize_text(payload.get("dataset_id"))
        if dataset_id is None:
            continue
        if dataset_id not in selected:
            payload["profile_run"] = _coalesce(
                _normalize_text(payload.get("run_name")),
                _normalize_text(payload.get("profile_run")),
            )
            selected[dataset_id] = payload
    return selected


def _profile_refresh_remediation_text(registry_path: Path) -> str:
    return (
        "Run scripts/py -m fisheye.utils.sync_eye_mask_profile_registry "
        f"--registry {registry_path} --apply."
    )


def _query_subject_lineage_rows(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not dataset_ids:
        return []
    placeholders = ", ".join("?" for _ in dataset_ids)
    sql = (
        "SELECT dataset_id, genotype, dpf_at_acquisition "
        "FROM recording_subject_overview "
        f"WHERE dataset_id IN ({placeholders});"
    )
    rows = registry.conn.execute(sql, tuple(dataset_ids)).fetchall()
    return [dict(row) for row in rows]


def _evaluate_subject_lineage_coverage(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
    subject_lineage_policy: str,
) -> tuple[SubjectLineageCoverage, list[dict[str, Any]]]:
    policy = _normalize_text(subject_lineage_policy) or "warn"
    if policy not in {"warn", "require"}:
        raise ValueError(f"Unsupported subject lineage policy: {subject_lineage_policy}")

    unique_dataset_ids = list(dict.fromkeys(dataset_ids))
    total_count = len(unique_dataset_ids)
    try:
        lineage_rows = _query_subject_lineage_rows(registry, dataset_ids=unique_dataset_ids)
    except Exception as exc:
        if policy == "require":
            raise ValueError(
                f"Subject lineage coverage unavailable (recording_subject_overview query failed): {exc}"
            ) from exc
        reason = f"recording_subject_overview query failed: {exc}"
        print(f"Subject lineage coverage: unavailable ({reason})")
        return (
            SubjectLineageCoverage(
                manifest_dataset_count=int(total_count),
                lineage_covered_dataset_count=None,
                missing_lineage_dataset_ids=tuple(sorted(unique_dataset_ids)),
                coverage_unavailable_reason=reason,
            ),
            [],
        )

    covered_ids = {
        str(row["dataset_id"]).strip()
        for row in lineage_rows
        if _normalize_text(row.get("dataset_id")) is not None
    }
    missing_ids = sorted(dataset_id for dataset_id in unique_dataset_ids if dataset_id not in covered_ids)
    covered_count = total_count - len(missing_ids)
    print(f"Subject lineage coverage: {covered_count}/{total_count} datasets")
    if missing_ids:
        print("Subject lineage missing dataset_id(s): " + ", ".join(missing_ids))
    if policy == "require" and missing_ids:
        raise ValueError(
            "Missing subject lineage rows in recording_subject_overview for dataset_id(s): "
            + ", ".join(missing_ids)
        )

    return (
        SubjectLineageCoverage(
            manifest_dataset_count=int(total_count),
            lineage_covered_dataset_count=int(covered_count),
            missing_lineage_dataset_ids=tuple(missing_ids),
            coverage_unavailable_reason=None,
        ),
        lineage_rows,
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


def _build_genotype_counts(lineage_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in lineage_rows:
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


def _extract_profile_summary(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("profile_json", "summary_statistics_json", "summary_statistics", "profile_summary"):
        payload = _as_mapping(row.get(key))
        if isinstance(payload, Mapping):
            return payload
    return {}


def _metric_from_row_or_summary(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    row_fields: Sequence[str],
    summary_paths: Sequence[Sequence[str]],
) -> Optional[float]:
    for field in row_fields:
        value = _as_float(row.get(field))
        if value is not None:
            return value
    for path in summary_paths:
        value = _as_float(_value_at_path(summary, path))
        if value is not None:
            return value
    return None


def _text_from_row_or_summary(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    row_fields: Sequence[str],
    summary_paths: Sequence[Sequence[str]],
) -> Optional[str]:
    for field in row_fields:
        value = _normalize_text(row.get(field))
        if value is not None:
            return value
    for path in summary_paths:
        value = _normalize_text(_value_at_path(summary, path))
        if value is not None:
            return value
    return None


def _extract_center_heatmap(summary: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    candidate_paths: tuple[tuple[str, ...], ...] = (
        ("spatial", "center_heatmap"),
        ("center_heatmap",),
        ("spatial", "coverage_heatmap"),
    )
    for path in candidate_paths:
        payload = _value_at_path(summary, path)
        if isinstance(payload, Mapping):
            return payload
    return None


def _aggregate_center_heatmap(
    summaries: Sequence[Mapping[str, Any]],
    *,
    weights: Sequence[Optional[float]],
) -> Optional[dict[str, Any]]:
    canonical_shape: Optional[tuple[int, int]] = None
    weighted_sum: Optional[np.ndarray] = None
    total_weight = 0.0
    included = 0
    skipped = 0

    for index, summary in enumerate(summaries):
        heatmap = _extract_center_heatmap(summary)
        if not isinstance(heatmap, Mapping):
            continue
        grid_h = _as_int(heatmap.get("grid_h"))
        grid_w = _as_int(heatmap.get("grid_w"))
        density_raw = heatmap.get("density")
        if grid_h is None or grid_w is None or not isinstance(density_raw, Sequence):
            continue

        density = np.asarray([float(value) for value in density_raw], dtype=np.float64)
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
        "density": [float(value) for value in density.reshape(-1).tolist()],
        "source_dataset_count": int(included),
        "skipped_mismatched_grids": int(skipped),
    }


def _open_zarr_group(path: Path) -> Any:
    try:
        return zarr.open_group(str(path), mode="r", consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def _split_counts_from_merged_zarr(path: Optional[Path]) -> dict[str, int]:
    if path is None:
        return {}
    try:
        root = _open_zarr_group(path)
    except Exception:
        return {}

    split_group = root.get("splits")
    if split_group is None:
        return {}

    payload: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        array_name = f"{split_name}_indices"
        if array_name not in split_group:
            continue
        try:
            indices = np.asarray(split_group[array_name][:])
        except Exception:
            continue
        payload[split_name] = int(indices.reshape(-1).size)
    return payload


def _decode_string_array(values: np.ndarray) -> list[Optional[str]]:
    flat = np.asarray(values).reshape(-1)
    decoded: list[Optional[str]] = []
    for value in flat.tolist():
        decoded.append(_normalize_text(value))
    return decoded


def _split_dataset_counts_from_merged_zarr(path: Optional[Path]) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    try:
        root = _open_zarr_group(path)
    except Exception:
        return {}

    split_group = root.get("splits")
    source_index_group = root.get("source_index")
    if split_group is None or source_index_group is None:
        return {}
    if "source_dataset_idx" not in source_index_group or "source_dataset_id" not in source_index_group:
        return {}

    try:
        source_dataset_idx = np.asarray(source_index_group["source_dataset_idx"][:], dtype=np.int64).reshape(-1)
        source_dataset_ids = _decode_string_array(np.asarray(source_index_group["source_dataset_id"][:]))
    except Exception:
        return {}
    if source_dataset_idx.ndim != 1 or not source_dataset_ids:
        return {}

    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        array_name = f"{split_name}_indices"
        if array_name not in split_group:
            continue
        try:
            split_indices = np.asarray(split_group[array_name][:], dtype=np.int64).reshape(-1)
        except Exception:
            continue
        if split_indices.size == 0:
            split_counts[split_name] = {}
            continue

        valid = (split_indices >= 0) & (split_indices < source_dataset_idx.shape[0])
        if not np.any(valid):
            split_counts[split_name] = {}
            continue

        counts: Counter[str] = Counter()
        for dataset_idx in source_dataset_idx[split_indices[valid]].tolist():
            if not isinstance(dataset_idx, (int, np.integer)):
                continue
            idx = int(dataset_idx)
            if idx < 0 or idx >= len(source_dataset_ids):
                continue
            dataset_id = source_dataset_ids[idx]
            if dataset_id is None:
                continue
            counts[dataset_id] += 1
        split_counts[split_name] = {key: int(counts[key]) for key in sorted(counts)}

    return split_counts


def _weighted_metric_from_counts(
    *,
    dataset_counts: Mapping[str, int],
    metric_by_dataset: Mapping[str, Optional[float]],
) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0
    for dataset_id, raw_weight in dataset_counts.items():
        weight = float(raw_weight)
        if weight <= 0:
            continue
        value = metric_by_dataset.get(dataset_id)
        if value is None:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _weighted_lineage_dpf(
    *,
    dataset_counts: Mapping[str, int],
    dpf_by_dataset: Mapping[str, Optional[float]],
) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0
    for dataset_id, raw_weight in dataset_counts.items():
        weight = float(raw_weight)
        if weight <= 0:
            continue
        value = dpf_by_dataset.get(dataset_id)
        if value is None:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _weighted_genotype_fractions(
    *,
    dataset_counts: Mapping[str, int],
    genotype_by_dataset: Mapping[str, Optional[str]],
) -> dict[str, float]:
    counter: Counter[str] = Counter()
    total_weight = 0.0
    for dataset_id, raw_weight in dataset_counts.items():
        weight = float(raw_weight)
        if weight <= 0:
            continue
        genotype = _normalize_text(genotype_by_dataset.get(dataset_id))
        if genotype is None:
            continue
        counter[genotype] += int(raw_weight)
        total_weight += weight
    if total_weight <= 0:
        return {}
    return {key: float(counter[key]) / total_weight for key in sorted(counter)}


def _max_fraction_delta(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> Optional[float]:
    keys = set(left.keys()) | set(right.keys())
    if not keys:
        return None
    return max(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def _build_parity(
    *,
    merged_zarr: Optional[Path],
    metrics_by_dataset: Mapping[str, Mapping[str, Optional[float]]],
    lineage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    split_dataset_counts = _split_dataset_counts_from_merged_zarr(merged_zarr)
    train_counts = split_dataset_counts.get("train") or {}
    val_counts = split_dataset_counts.get("val") or {}
    if not train_counts or not val_counts:
        return {
            "available": False,
            "reason": "missing train/val source-index split coverage",
            "split_dataset_row_counts": {
                "train": {key: int(train_counts[key]) for key in sorted(train_counts)},
                "val": {key: int(val_counts[key]) for key in sorted(val_counts)},
            },
            "metrics": {},
            "lineage": {},
        }

    metric_field_specs = (
        ("successful_roi_pair_rate", "successful_roi_pair_rate"),
        ("rois_per_second", "rois_per_second"),
        ("eye_separation_p50", "eye_separation_p50"),
        ("ellipse_major_p50", "ellipse_major_p50"),
        ("ellipse_minor_p50", "ellipse_minor_p50"),
        ("left_area_p50", "left_area_p50"),
        ("right_area_p50", "right_area_p50"),
        ("union_area_p50", "union_area_p50"),
        ("area_lr_ratio_p50", "area_lr_ratio_p50"),
    )

    metric_payload: dict[str, dict[str, Optional[float]]] = {}
    for metric_name, field_name in metric_field_specs:
        train_value = _weighted_metric_from_counts(
            dataset_counts=train_counts,
            metric_by_dataset={
                dataset_id: values.get(field_name)
                for dataset_id, values in metrics_by_dataset.items()
            },
        )
        val_value = _weighted_metric_from_counts(
            dataset_counts=val_counts,
            metric_by_dataset={
                dataset_id: values.get(field_name)
                for dataset_id, values in metrics_by_dataset.items()
            },
        )
        metric_payload[metric_name] = {
            "train": float(train_value) if train_value is not None else None,
            "val": float(val_value) if val_value is not None else None,
            "delta": (
                float(abs(train_value - val_value))
                if train_value is not None and val_value is not None
                else None
            ),
        }

    genotype_by_dataset = {
        str(row.get("dataset_id")): _normalize_text(row.get("genotype"))
        for row in lineage_rows
        if _normalize_text(row.get("dataset_id")) is not None
    }
    dpf_by_dataset = {
        str(row.get("dataset_id")): _as_float(row.get("dpf_at_acquisition"))
        for row in lineage_rows
        if _normalize_text(row.get("dataset_id")) is not None
    }
    train_genotype_fraction = _weighted_genotype_fractions(
        dataset_counts=train_counts,
        genotype_by_dataset=genotype_by_dataset,
    )
    val_genotype_fraction = _weighted_genotype_fractions(
        dataset_counts=val_counts,
        genotype_by_dataset=genotype_by_dataset,
    )
    train_dpf_mean = _weighted_lineage_dpf(
        dataset_counts=train_counts,
        dpf_by_dataset=dpf_by_dataset,
    )
    val_dpf_mean = _weighted_lineage_dpf(
        dataset_counts=val_counts,
        dpf_by_dataset=dpf_by_dataset,
    )

    return {
        "available": True,
        "split_dataset_row_counts": {
            "train": {key: int(train_counts[key]) for key in sorted(train_counts)},
            "val": {key: int(val_counts[key]) for key in sorted(val_counts)},
        },
        "metrics": metric_payload,
        "lineage": {
            "genotype_fraction": {
                "train": train_genotype_fraction,
                "val": val_genotype_fraction,
            },
            "genotype_mix_max_abs_delta": _max_fraction_delta(
                train_genotype_fraction,
                val_genotype_fraction,
            ),
            "dpf_mean": {
                "train": float(train_dpf_mean) if train_dpf_mean is not None else None,
                "val": float(val_dpf_mean) if val_dpf_mean is not None else None,
                "delta": (
                    float(abs(train_dpf_mean - val_dpf_mean))
                    if train_dpf_mean is not None and val_dpf_mean is not None
                    else None
                ),
            },
        },
    }


def _selection_row_counts(source_rows: Sequence[Mapping[str, Any]]) -> tuple[Optional[int], Optional[int]]:
    pre_values: list[int] = []
    post_values: list[int] = []

    for row in source_rows:
        pre = _as_int(_coalesce(row.get("rows_total"), row.get("total_rois"), row.get("samples_total")))
        post = _as_int(_coalesce(row.get("rows_post_gate"), row.get("rows_selected"), pre))
        if pre is not None and pre >= 0:
            pre_values.append(int(pre))
        if post is not None and post >= 0:
            post_values.append(int(post))

    rows_pre_gate = int(sum(pre_values)) if pre_values else None
    rows_post_gate = int(sum(post_values)) if post_values else rows_pre_gate
    return rows_pre_gate, rows_post_gate


def _build_eye_mask_training_data_card(
    *,
    registry: Registry,
    manifest: Mapping[str, Any],
    merged_zarr: Optional[Path],
    split: str,
    subject_lineage_policy: str,
    allow_profile_mtime_mismatch: bool,
    allow_profile_fallback_scan: bool,
) -> dict[str, Any]:
    source_rows = _manifest_source_rows(manifest)
    refs = _manifest_dataset_refs(registry, manifest)
    dataset_ids = [ref.dataset_id for ref in refs]

    subject_lineage_coverage, lineage_rows = _evaluate_subject_lineage_coverage(
        registry,
        dataset_ids=dataset_ids,
        subject_lineage_policy=subject_lineage_policy,
    )

    profile_rows_by_dataset, profile_source = _select_profile_rows_registry_first(
        registry,
        dataset_ids=dataset_ids,
    )

    missing_profile_dataset_ids = [
        ref.dataset_id for ref in refs if ref.dataset_id not in profile_rows_by_dataset
    ]
    fallback_used = False
    fallback_reason: Optional[str] = None

    if missing_profile_dataset_ids:
        if not allow_profile_fallback_scan:
            raise ValueError(
                "Missing eye_mask_data_profile_latest rows for dataset_id(s): "
                + ", ".join(sorted(missing_profile_dataset_ids))
                + ". "
                + _profile_refresh_remediation_text(Path(registry.path))
                + " To continue without profile rows, rerun with --allow-profile-fallback-scan."
            )

        fallback_rows = _query_eye_mask_performance_fallback(
            registry,
            dataset_ids=missing_profile_dataset_ids,
        )
        for dataset_id, row in fallback_rows.items():
            profile_rows_by_dataset[dataset_id] = row

        remaining_missing = [
            ref.dataset_id for ref in refs if ref.dataset_id not in profile_rows_by_dataset
        ]
        if remaining_missing:
            raise ValueError(
                "Missing fallback eye_mask_performance_latest rows for dataset_id(s): "
                + ", ".join(sorted(remaining_missing))
            )

        fallback_used = True
        fallback_reason = (
            "eye_mask_data_profile_latest missing/incomplete; "
            "used eye_mask_performance_latest fallback"
        )

        if profile_source in {"registry_api", "registry_sql_view"}:
            profile_source = "registry_with_performance_fallback"
        else:
            profile_source = "performance_latest_fallback"

    rows_pre_gate, rows_post_gate = _selection_row_counts(source_rows)
    split_counts = _split_counts_from_merged_zarr(merged_zarr)
    if not split_counts and rows_post_gate is not None:
        split_counts = {str(split): int(rows_post_gate)}

    quality_total_rois_values: list[Optional[float]] = []
    quality_successful_pairs_values: list[Optional[float]] = []
    quality_rate_values: list[Optional[float]] = []
    rois_per_second_values: list[tuple[Optional[float], Optional[float]]] = []
    edge_rate_values: list[tuple[Optional[float], Optional[float]]] = []

    eye_separation_values: list[Optional[float]] = []
    ellipse_major_values: list[Optional[float]] = []
    ellipse_minor_values: list[Optional[float]] = []
    left_area_values: list[Optional[float]] = []
    right_area_values: list[Optional[float]] = []
    union_area_values: list[Optional[float]] = []
    area_lr_ratio_values: list[Optional[float]] = []

    composition_rows: list[dict[str, Optional[str]]] = []
    profile_summaries: list[Mapping[str, Any]] = []
    heatmap_weights: list[Optional[float]] = []

    zarr_mtime_mismatch_count = 0
    source_run_refs: list[dict[str, Any]] = []
    metrics_by_dataset: dict[str, dict[str, Optional[float]]] = {}
    profile_run_refs: list[dict[str, str]] = []

    for ref in refs:
        row = profile_rows_by_dataset[ref.dataset_id]
        summary = _extract_profile_summary(row)
        profile_summaries.append(summary)

        total_rois = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("rows_total", "total_rois"),
            summary_paths=(("quality", "rows_total"), ("selection", "total_rois")),
        )
        successful_pairs = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("rows_usable", "successful_roi_pairs", "usable_roi_pairs"),
            summary_paths=(("quality", "rows_usable"), ("quality", "successful_roi_pairs")),
        )
        successful_rate = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("usable_rate", "successful_roi_pair_rate", "usable_pair_rate"),
            summary_paths=(
                ("quality", "usable_rate"),
                ("quality", "successful_roi_pair_rate"),
            ),
        )
        rois_per_second = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("rois_per_second", "inference_average_fps"),
            summary_paths=(("quality", "rois_per_second"),),
        )
        edge_rate = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("edge_proximity_rate",),
            summary_paths=(("spatial", "edge_proximity_rate"),),
        )

        eye_separation_p50 = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("eye_separation_p50", "eye_separation_median"),
            summary_paths=(
                ("geometry", "eye_separation", "stats", "p50"),
                ("geometry", "eye_separation", "p50"),
            ),
        )
        ellipse_major_p50 = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("ellipse_major_p50", "major_axis_p50"),
            summary_paths=(
                ("geometry", "ellipse_major", "stats", "p50"),
                ("geometry", "major_axis", "stats", "p50"),
            ),
        )
        ellipse_minor_p50 = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("ellipse_minor_p50", "minor_axis_p50"),
            summary_paths=(
                ("geometry", "ellipse_minor", "stats", "p50"),
                ("geometry", "minor_axis", "stats", "p50"),
            ),
        )
        left_area_p50_usable = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=(),
            summary_paths=(
                ("geometry", "left_area_usable", "stats", "p50"),
                ("geometry", "area_left_usable", "stats", "p50"),
                ("geometry", "left_eye_area_usable", "stats", "p50"),
                ("geometry", "left_area_usable", "p50"),
                ("geometry", "area_left_usable", "p50"),
                ("geometry", "left_eye_area_usable", "p50"),
            ),
        )
        left_area_p50_all = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("left_area_p50", "area_left_p50", "left_eye_area_p50"),
            summary_paths=(
                ("geometry", "left_area", "stats", "p50"),
                ("geometry", "area_left", "stats", "p50"),
                ("geometry", "left_eye_area", "stats", "p50"),
                ("geometry", "left_area", "p50"),
                ("geometry", "area_left", "p50"),
                ("geometry", "left_eye_area", "p50"),
            ),
        )
        left_area_p50 = left_area_p50_usable if left_area_p50_usable is not None else left_area_p50_all

        right_area_p50_usable = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=(),
            summary_paths=(
                ("geometry", "right_area_usable", "stats", "p50"),
                ("geometry", "area_right_usable", "stats", "p50"),
                ("geometry", "right_eye_area_usable", "stats", "p50"),
                ("geometry", "right_area_usable", "p50"),
                ("geometry", "area_right_usable", "p50"),
                ("geometry", "right_eye_area_usable", "p50"),
            ),
        )
        right_area_p50_all = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("right_area_p50", "area_right_p50", "right_eye_area_p50"),
            summary_paths=(
                ("geometry", "right_area", "stats", "p50"),
                ("geometry", "area_right", "stats", "p50"),
                ("geometry", "right_eye_area", "stats", "p50"),
                ("geometry", "right_area", "p50"),
                ("geometry", "area_right", "p50"),
                ("geometry", "right_eye_area", "p50"),
            ),
        )
        right_area_p50 = right_area_p50_usable if right_area_p50_usable is not None else right_area_p50_all

        union_area_p50_usable = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=(),
            summary_paths=(
                ("geometry", "union_area_usable", "stats", "p50"),
                ("geometry", "area_union_usable", "stats", "p50"),
                ("geometry", "combined_area_usable", "stats", "p50"),
                ("geometry", "area_usable", "stats", "p50"),
                ("geometry", "union_area_usable", "p50"),
                ("geometry", "area_union_usable", "p50"),
                ("geometry", "combined_area_usable", "p50"),
                ("geometry", "area_usable", "p50"),
            ),
        )
        union_area_p50_all = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=("union_area_p50", "area_union_p50", "combined_area_p50", "area_p50"),
            summary_paths=(
                ("geometry", "union_area", "stats", "p50"),
                ("geometry", "area_union", "stats", "p50"),
                ("geometry", "combined_area", "stats", "p50"),
                ("geometry", "area", "stats", "p50"),
                ("geometry", "union_area", "p50"),
                ("geometry", "area_union", "p50"),
                ("geometry", "combined_area", "p50"),
                ("geometry", "area", "p50"),
            ),
        )
        union_area_p50 = union_area_p50_usable if union_area_p50_usable is not None else union_area_p50_all
        area_lr_ratio_p50 = _metric_from_row_or_summary(
            row,
            summary,
            row_fields=(
                "area_lr_ratio_p50",
                "left_right_area_ratio_p50",
                "area_ratio_left_right_p50",
                "lr_area_ratio_p50",
            ),
            summary_paths=(
                ("geometry", "area_lr_ratio", "stats", "p50"),
                ("geometry", "left_right_area_ratio", "stats", "p50"),
                ("geometry", "area_ratio_left_right", "stats", "p50"),
                ("geometry", "lr_area_ratio", "stats", "p50"),
                ("geometry", "area_lr_ratio", "p50"),
                ("geometry", "left_right_area_ratio", "p50"),
                ("geometry", "area_ratio_left_right", "p50"),
                ("geometry", "lr_area_ratio", "p50"),
            ),
        )

        quality_total_rois_values.append(total_rois)
        quality_successful_pairs_values.append(successful_pairs)
        quality_rate_values.append(successful_rate)
        rois_per_second_values.append((rois_per_second, total_rois))
        edge_rate_values.append((edge_rate, total_rois))

        eye_separation_values.append(eye_separation_p50)
        ellipse_major_values.append(ellipse_major_p50)
        ellipse_minor_values.append(ellipse_minor_p50)
        left_area_values.append(left_area_p50)
        right_area_values.append(right_area_p50)
        union_area_values.append(union_area_p50)
        area_lr_ratio_values.append(area_lr_ratio_p50)

        heatmap_weights.append(total_rois)

        metrics_by_dataset[ref.dataset_id] = {
            "successful_roi_pair_rate": successful_rate,
            "rois_per_second": rois_per_second,
            "eye_separation_p50": eye_separation_p50,
            "ellipse_major_p50": ellipse_major_p50,
            "ellipse_minor_p50": ellipse_minor_p50,
            "left_area_p50": left_area_p50,
            "left_area_p50_usable": left_area_p50_usable,
            "left_area_p50_all": left_area_p50_all,
            "right_area_p50": right_area_p50,
            "right_area_p50_usable": right_area_p50_usable,
            "right_area_p50_all": right_area_p50_all,
            "union_area_p50": union_area_p50,
            "union_area_p50_usable": union_area_p50_usable,
            "union_area_p50_all": union_area_p50_all,
            "area_lr_ratio_p50": area_lr_ratio_p50,
        }

        composition_row: dict[str, Optional[str]] = {}
        for field in COMPOSITION_FIELDS:
            summary_paths = (("composition", field),)
            composition_row[field] = _text_from_row_or_summary(
                row,
                summary,
                row_fields=(field,),
                summary_paths=summary_paths,
            )
        composition_rows.append(composition_row)

        expected_mtime = _as_int(row.get("zarr_mtime_ns"))
        observed_mtime = None
        if ref.zarr_path.exists():
            try:
                observed_mtime = int(ref.zarr_path.stat().st_mtime_ns)
            except OSError:
                observed_mtime = None
        mtime_matches = expected_mtime is None or observed_mtime is None or expected_mtime == observed_mtime
        if not mtime_matches:
            zarr_mtime_mismatch_count += 1

        profile_run = _normalize_text(_coalesce(row.get("profile_run"), row.get("run_name")))
        if profile_run is not None:
            profile_run_refs.append({"dataset_id": ref.dataset_id, "profile_run": profile_run})

        source_run_refs.append(
            {
                "dataset_id": ref.dataset_id,
                "zarr_path": str(ref.zarr_path),
                "profile_run": profile_run,
                "review_state": _normalize_text(row.get("review_state")),
                "profile_row_present": True,
                "profile_source": profile_source,
                "zarr_mtime_expected_ns": expected_mtime,
                "zarr_mtime_observed_ns": observed_mtime,
                "zarr_mtime_matches": bool(mtime_matches),
                "stage_group": _normalize_text(row.get("stage_group")),
                "successful_roi_pair_rate": successful_rate,
                "left_area_p50": left_area_p50,
                "left_area_p50_usable": left_area_p50_usable,
                "left_area_p50_all": left_area_p50_all,
                "right_area_p50": right_area_p50,
                "right_area_p50_usable": right_area_p50_usable,
                "right_area_p50_all": right_area_p50_all,
                "union_area_p50": union_area_p50,
                "union_area_p50_usable": union_area_p50_usable,
                "union_area_p50_all": union_area_p50_all,
                "area_metric_source": (
                    "usable"
                    if any(
                        value is not None
                        for value in (
                            left_area_p50_usable,
                            right_area_p50_usable,
                            union_area_p50_usable,
                        )
                    )
                    else "all_rows"
                ),
            }
        )

    if zarr_mtime_mismatch_count > 0 and not allow_profile_mtime_mismatch:
        raise ValueError(
            "Stale eye_mask_data_profile_latest row(s): detected zarr_mtime_ns mismatches. "
            + _profile_refresh_remediation_text(Path(registry.path))
            + " To bypass staleness checks, rerun with --allow-profile-mtime-mismatch."
        )

    total_rois_sum = float(sum(v for v in quality_total_rois_values if v is not None))
    successful_pairs_sum = float(sum(v for v in quality_successful_pairs_values if v is not None))

    successful_rate_overall = _safe_ratio(successful_pairs_sum, total_rois_sum)
    if successful_rate_overall is None:
        successful_rate_overall = _weighted_mean(
            [(quality_rate_values[idx], quality_total_rois_values[idx]) for idx in range(len(quality_rate_values))]
        )

    composition_counts: dict[str, dict[str, int]] = {}
    for field in COMPOSITION_FIELDS:
        counter: Counter[str] = Counter()
        for row in composition_rows:
            value = _normalize_text(row.get(field))
            if value is not None:
                counter[value] += 1
        if counter:
            composition_counts[field] = {key: int(counter[key]) for key in sorted(counter)}

    center_heatmap = _aggregate_center_heatmap(profile_summaries, weights=heatmap_weights)

    parity = _build_parity(
        merged_zarr=merged_zarr,
        metrics_by_dataset=metrics_by_dataset,
        lineage_rows=lineage_rows,
    )

    dpf_values = [_as_float(row.get("dpf_at_acquisition")) for row in lineage_rows]

    set_id = _normalize_text(manifest.get("set_id"))
    set_version = None
    set_version_value = manifest.get("set_version")
    if isinstance(set_version_value, (int, str)):
        set_version = str(set_version_value)

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
            "dataset_count": int(len(refs)),
            "split": str(split),
            "filters": selection_filters,
            "rows_pre_gate": rows_pre_gate,
            "rows_post_gate": rows_post_gate,
            "split_counts": {key: int(value) for key, value in sorted(split_counts.items())},
        },
        "quality": {
            "total_rois": int(total_rois_sum) if total_rois_sum > 0 else None,
            "successful_roi_pairs_total": int(successful_pairs_sum) if successful_pairs_sum > 0 else None,
            "successful_roi_pair_rate_overall": (
                float(successful_rate_overall) if successful_rate_overall is not None else None
            ),
            "successful_roi_pair_rate_dataset_stats": _numeric_stats(quality_rate_values),
            "successful_roi_pair_rate_histogram": _numeric_histogram(quality_rate_values),
            "rois_per_second_weighted": _weighted_mean(rois_per_second_values),
            "review_state_counts": {
                key: int(value)
                for key, value in sorted(
                    Counter(
                        _normalize_text(profile_rows_by_dataset[ref.dataset_id].get("review_state")) or "unknown"
                        for ref in refs
                    ).items()
                )
            },
        },
        "geometry": {
            "eye_separation_p50_dataset_stats": _numeric_stats(eye_separation_values),
            "eye_separation_p50_histogram": _numeric_histogram(eye_separation_values),
            "ellipse_major_p50_dataset_stats": _numeric_stats(ellipse_major_values),
            "ellipse_major_p50_histogram": _numeric_histogram(ellipse_major_values),
            "ellipse_minor_p50_dataset_stats": _numeric_stats(ellipse_minor_values),
            "ellipse_minor_p50_histogram": _numeric_histogram(ellipse_minor_values),
            "left_area_p50_dataset_stats": _numeric_stats(left_area_values),
            "left_area_p50_histogram": _numeric_histogram(left_area_values),
            "right_area_p50_dataset_stats": _numeric_stats(right_area_values),
            "right_area_p50_histogram": _numeric_histogram(right_area_values),
            "union_area_p50_dataset_stats": _numeric_stats(union_area_values),
            "union_area_p50_histogram": _numeric_histogram(union_area_values),
            "area_lr_ratio_p50_dataset_stats": _numeric_stats(area_lr_ratio_values),
            "area_lr_ratio_p50_histogram": _numeric_histogram(area_lr_ratio_values),
        },
        "spatial": {
            "edge_proximity_rate_weighted": _weighted_mean(edge_rate_values),
            "center_heatmap": center_heatmap,
        },
        "composition": {
            "counts": composition_counts,
        },
        "subject_coverage": _build_subject_coverage_payload(subject_lineage_coverage),
        "parity": parity,
        "audit_freshness": {
            "canonical_dataset_id_resolved_count": int(sum(1 for ref in refs if ref.resolved_by_registry)),
            "profile_source": profile_source,
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "missing_profile_dataset_ids": sorted(missing_profile_dataset_ids),
            "zarr_mtime_mismatch_count": int(zarr_mtime_mismatch_count),
            "source_run_refs": sorted(source_run_refs, key=lambda item: str(item.get("dataset_id") or "")),
        },
        "genotype_counts": _build_genotype_counts(lineage_rows),
        "dpf_stats": _numeric_stats(dpf_values),
        "dpf_histogram": _build_dpf_histogram(dpf_values, source_dataset_count=len(refs)),
        "profile_run_refs": sorted(profile_run_refs, key=lambda item: item["dataset_id"]),
    }
    return card


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate eye-mask training-data card payload from manifest/training-set datasets "
            "and registry profile rows."
        )
    )
    parser.add_argument("--manifest", type=Path, help="Training manifest JSON path.")
    parser.add_argument("--training-set", type=str, help="Registry training_sets.set_id identifier.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: <set_id>.data_card.json).")
    parser.add_argument(
        "--merged-zarr",
        type=Path,
        help="Optional merged training Zarr path used for split parity extraction.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split label metadata (default: train).")
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
        "--allow-profile-mtime-mismatch",
        action="store_true",
        help=(
            "Allow stale eye_mask_data_profile_latest rows where zarr_mtime_ns "
            "does not match filesystem mtime."
        ),
    )
    parser.add_argument(
        "--allow-profile-fallback-scan",
        action="store_true",
        help=(
            "Allow fallback to eye_mask_performance_latest rows when "
            "eye_mask_data_profile_latest rows are missing."
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
        help="Coarsening factor for heatmap bins in plot PNGs (default: 2).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open generated/existing plot PNGs via xdg-open after aggregation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force plot regeneration even when existing plot PNGs are present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute card but do not write output JSON.")
    parser.add_argument("--json", action="store_true", help="Print card JSON to stdout.")
    parser.add_argument(
        "--report-min-eye-area-p50",
        type=float,
        help="Print dataset sources whose left/right area p50 falls below this threshold.",
    )
    parser.add_argument(
        "--report-min-union-area-p50",
        type=float,
        help="Print dataset sources whose union-area p50 falls below this threshold.",
    )
    args = parser.parse_args(argv)

    if bool(args.manifest) == bool(args.training_set):
        parser.error("Exactly one of --manifest or --training-set is required.")
    if args.plot_heatmap_bin_factor < 1:
        parser.error("--plot-heatmap-bin-factor must be >= 1.")
    if args.view and args.dry_run:
        parser.error("--view cannot be combined with --dry-run.")
    if args.report_min_eye_area_p50 is not None and args.report_min_eye_area_p50 <= 0:
        parser.error("--report-min-eye-area-p50 must be > 0.")
    if args.report_min_union_area_p50 is not None and args.report_min_union_area_p50 <= 0:
        parser.error("--report-min-union-area-p50 must be > 0.")

    manifest_path = Path(args.manifest) if args.manifest is not None else None
    if manifest_path is not None and not manifest_path.exists():
        print(f"Training data card aggregation failed: manifest not found: {manifest_path}")
        return 1

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        if manifest_path is not None:
            manifest = _load_manifest(manifest_path)
        else:
            assert args.training_set is not None
            manifest = _load_training_set_manifest(registry, set_id=str(args.training_set))

        merged_zarr = Path(args.merged_zarr) if args.merged_zarr is not None else None
        card = _build_eye_mask_training_data_card(
            registry=registry,
            manifest=manifest,
            merged_zarr=merged_zarr,
            split=str(args.split),
            subject_lineage_policy=str(args.subject_lineage_policy),
            allow_profile_mtime_mismatch=bool(args.allow_profile_mtime_mismatch),
            allow_profile_fallback_scan=bool(args.allow_profile_fallback_scan),
        )
    except Exception as exc:
        print(f"Training data card aggregation failed: {exc}")
        return 1
    finally:
        registry.close()

    output_path = Path(args.output) if args.output is not None else _default_output_path(
        manifest_path,
        set_id=_normalize_text(manifest.get("set_id")),
    )

    _print_low_area_report(
        card=card,
        min_eye_area_p50=args.report_min_eye_area_p50,
        min_union_area_p50=args.report_min_union_area_p50,
    )

    if args.json:
        print(json.dumps(card, indent=2))

    if args.dry_run:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else _default_plot_dir(output_path)
        print(
            "Eye-mask training data card: mode=dry-run "
            f"datasets={card['selection']['dataset_count']} output={output_path} "
            f"plots={'off' if args.no_plots else plot_dir} "
            f"heatmap_bin_factor={int(args.plot_heatmap_bin_factor)}"
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "Eye-mask training data card: mode=apply "
        f"datasets={card['selection']['dataset_count']} output={output_path}"
    )

    if not args.no_plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else _default_plot_dir(output_path)
        prefix = _normalize_text(args.plot_prefix) or _normalize_text(card.get("set_id")) or output_path.stem
        if args.view or args.force:
            plot_main = getattr(plot_data_card, "main", None)
            if callable(plot_main):
                plot_cli: list[str] = [
                    "--card",
                    str(output_path),
                    "--heatmap-bin-factor",
                    str(int(args.plot_heatmap_bin_factor)),
                ]
                _add_arg(plot_cli, "--output-dir", plot_dir)
                _add_arg(plot_cli, "--prefix", prefix)
                if args.view:
                    plot_cli.append("--view")
                if args.force:
                    plot_cli.append("--force")
                try:
                    plot_rc = int(plot_main(plot_cli))
                except Exception as exc:
                    print(f"Warning: eye-mask training data-card plot generation skipped: {exc}")
                else:
                    if plot_rc != 0:
                        print(
                            "Warning: eye-mask training data-card plotting returned non-zero "
                            f"exit code: {plot_rc}"
                        )
            else:
                print("Warning: eye-mask plotter main(argv) unavailable; falling back to direct generation.")
                try:
                    generated = _generate_plots(
                        card=card,
                        output_dir=plot_dir,
                        prefix=str(prefix),
                        heatmap_bin_factor=int(args.plot_heatmap_bin_factor),
                    )
                    print(
                        "Eye-mask training data-card plots: mode=apply "
                        f"generated={len(generated)} output_dir={plot_dir} "
                        f"heatmap_bin_factor={int(args.plot_heatmap_bin_factor)}"
                    )
                except Exception as exc:
                    print(f"Warning: eye-mask training data-card plot generation skipped: {exc}")
        else:
            try:
                generated = _generate_plots(
                    card=card,
                    output_dir=plot_dir,
                    prefix=str(prefix),
                    heatmap_bin_factor=int(args.plot_heatmap_bin_factor),
                )
                print(
                    "Eye-mask training data-card plots: mode=apply "
                    f"generated={len(generated)} output_dir={plot_dir} "
                    f"heatmap_bin_factor={int(args.plot_heatmap_bin_factor)}"
                )
            except Exception as exc:
                print(f"Warning: eye-mask training data-card plot generation skipped: {exc}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
