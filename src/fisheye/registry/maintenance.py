"""Maintenance CLI for cleaning stale/invalid registry rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .db import (
    Registry,
    RegistryPaths,
    _extract_detect_quality_rows,
    _extract_keypoint_quality_rows,
    _import_zarr,
)


@dataclass(frozen=True)
class InvalidDatasetCandidate:
    dataset_id: str
    zarr_path: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FailedRunCandidate:
    run_id: str
    set_id: Optional[str]
    status: Optional[str]
    created_utc: Optional[str]


@dataclass(frozen=True)
class EmptyTrainingSetCandidate:
    set_id: str
    name: Optional[str]
    created_utc: Optional[str]


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    run_id: Optional[str]
    detail: str


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registry maintenance (reconcile, prune invalid rows, optional VACUUM).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional root path(s) that scope reconcile/prune operations.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional path to the registry SQLite file.",
    )
    parser.add_argument(
        "--prune-invalid",
        action="store_true",
        help=(
            "Reconcile missing rows, then prune invalid datasets "
            "(status=missing or paths that point inside a Zarr store)."
        ),
    )
    parser.add_argument(
        "--prune-failed-runs",
        action="store_true",
        help="Prune training_runs rows with failed statuses.",
    )
    parser.add_argument(
        "--prune-empty-sets",
        action="store_true",
        help="Prune training_sets rows that have no linked training_runs.",
    )
    parser.add_argument(
        "--backfill-model-tables",
        action="store_true",
        help=(
            "Backfill detection_models/onnx_models/tensorrt_models from "
            "existing training_runs/model_exports rows."
        ),
    )
    parser.add_argument(
        "--backfill-keypoint-quality",
        action="store_true",
        help=(
            "Backfill keypoint_quality rows for datasets that currently have no quality rows."
        ),
    )
    parser.add_argument(
        "--backfill-detect-quality",
        action="store_true",
        help=(
            "Backfill detect_quality rows for datasets that currently have no quality rows."
        ),
    )
    parser.add_argument(
        "--refresh-keypoint-quality",
        action="store_true",
        help=(
            "Refresh keypoint_quality rows for all datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--refresh-detect-quality",
        action="store_true",
        help=(
            "Refresh detect_quality rows for all datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--check-integrity",
        action="store_true",
        help=(
            "Validate training registry integrity for new model tables. "
            "Returns non-zero on failures (CI-friendly)."
        ),
    )
    parser.add_argument(
        "--failed-status",
        action="append",
        help=(
            "Status value treated as failed when using --prune-failed-runs "
            "(repeatable or comma-separated, case-insensitive). Default: failed."
        ),
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run SQLite VACUUM after maintenance actions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without deleting rows or running VACUUM.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=50,
        help="Number of candidate rows to print (0 = no limit).",
    )
    return parser.parse_args(argv)


def _normalize_scope_paths(scope_paths: Optional[Sequence[Path]]) -> List[Path]:
    if not scope_paths:
        return []
    normalized: List[Path] = []
    for path in scope_paths:
        candidate = Path(path).expanduser()
        try:
            normalized.append(candidate.resolve())
        except Exception:
            normalized.append(candidate.absolute())
    return normalized


def _matches_scope(path: str, scope_roots: Sequence[Path]) -> bool:
    if not scope_roots:
        return True
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve()
    except Exception:
        candidate = candidate.absolute()
    for root in scope_roots:
        if candidate == root:
            return True
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_nested_zarr_subpath(path: str) -> bool:
    normalized = path.replace("\\", "/").rstrip("/").lower()
    if ".zarr/" not in normalized:
        return False
    return not normalized.endswith(".zarr")


def _is_zarr_root_path(path: Path) -> bool:
    return (path / "zarr.json").exists() or (path / ".zgroup").exists()


def _collect_invalid_dataset_candidates(
    registry: Registry,
    *,
    scope_paths: Optional[Sequence[Path]] = None,
    include_missing_scan: bool = False,
) -> List[InvalidDatasetCandidate]:
    scope_roots = _normalize_scope_paths(scope_paths)
    rows = registry.conn.execute(
        "SELECT dataset_id, zarr_path, status FROM datasets ORDER BY dataset_id;"
    ).fetchall()

    candidates: List[InvalidDatasetCandidate] = []
    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = str(row["zarr_path"])
        if not _matches_scope(zarr_path, scope_roots):
            continue
        reasons: List[str] = []
        if row["status"] == "missing":
            reasons.append("status_missing")
        elif include_missing_scan:
            candidate = Path(zarr_path).expanduser()
            if not _is_zarr_root_path(candidate):
                reasons.append("status_missing")
        if _is_nested_zarr_subpath(zarr_path):
            reasons.append("nested_zarr_subpath")
        if reasons:
            candidates.append(
                InvalidDatasetCandidate(
                    dataset_id=dataset_id,
                    zarr_path=zarr_path,
                    reasons=tuple(sorted(reasons)),
                )
            )
    return candidates


def _delete_dataset_ids(registry: Registry, dataset_ids: Sequence[str], *, dry_run: bool) -> int:
    if dry_run or not dataset_ids:
        return 0
    with registry.conn:
        for dataset_id in dataset_ids:
            registry.conn.execute("DELETE FROM datasets WHERE dataset_id = ?;", (dataset_id,))
    return len(dataset_ids)


def _normalize_status_values(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values or ():
        for token in str(value).split(","):
            status = token.strip().lower()
            if status:
                normalized.add(status)
    if not normalized:
        normalized.add("failed")
    return tuple(sorted(normalized))


def _collect_failed_run_candidates(
    registry: Registry,
    *,
    status_values: Sequence[str],
) -> List[FailedRunCandidate]:
    target_statuses = {value.strip().lower() for value in status_values if value and value.strip()}
    if not target_statuses:
        target_statuses = {"failed"}
    rows = registry.conn.execute(
        """
        SELECT run_id, set_id, status, created_utc
        FROM training_runs
        ORDER BY created_utc DESC, run_id DESC;
        """
    ).fetchall()
    candidates: List[FailedRunCandidate] = []
    for row in rows:
        status_raw = row["status"]
        if status_raw is None:
            continue
        status = str(status_raw).strip().lower()
        if status not in target_statuses:
            continue
        candidates.append(
            FailedRunCandidate(
                run_id=str(row["run_id"]),
                set_id=row["set_id"],
                status=row["status"],
                created_utc=row["created_utc"],
            )
        )
    return candidates


def _delete_training_run_ids(registry: Registry, run_ids: Sequence[str], *, dry_run: bool) -> int:
    if dry_run or not run_ids:
        return 0
    with registry.conn:
        for run_id in run_ids:
            registry.conn.execute("DELETE FROM training_runs WHERE run_id = ?;", (run_id,))
    return len(run_ids)


def _collect_empty_training_set_candidates(registry: Registry) -> List[EmptyTrainingSetCandidate]:
    rows = registry.conn.execute(
        """
        SELECT ts.set_id, ts.name, ts.created_utc
        FROM training_sets ts
        LEFT JOIN training_runs tr ON tr.set_id = ts.set_id
        GROUP BY ts.set_id, ts.name, ts.created_utc
        HAVING COUNT(tr.run_id) = 0
        ORDER BY ts.created_utc DESC, ts.set_id DESC;
        """
    ).fetchall()
    return [
        EmptyTrainingSetCandidate(
            set_id=str(row["set_id"]),
            name=row["name"],
            created_utc=row["created_utc"],
        )
        for row in rows
    ]


def _delete_training_set_ids(registry: Registry, set_ids: Sequence[str], *, dry_run: bool) -> int:
    if dry_run or not set_ids:
        return 0
    with registry.conn:
        for set_id in set_ids:
            registry.conn.execute("DELETE FROM training_sets WHERE set_id = ?;", (set_id,))
    return len(set_ids)


def _backfill_model_tables(registry: Registry, *, dry_run: bool) -> Dict[str, int]:
    detection_missing = int(
        registry.conn.execute(
            """
            SELECT COUNT(*)
            FROM training_runs tr
            LEFT JOIN detection_models dm ON dm.run_id = tr.run_id
            WHERE dm.run_id IS NULL;
            """
        ).fetchone()[0]
    )
    onnx_missing = int(
        registry.conn.execute(
            """
            SELECT COUNT(*)
            FROM model_exports me
            LEFT JOIN onnx_models om ON om.run_id = me.run_id
            WHERE lower(me.export_type) = 'onnx'
              AND om.run_id IS NULL;
            """
        ).fetchone()[0]
    )
    tensorrt_missing = int(
        registry.conn.execute(
            """
            WITH trt_exports AS (
                SELECT
                    me.run_id AS run_id,
                    COALESCE(
                        NULLIF(lower(json_extract(me.metadata_json, '$.precision')), ''),
                        CASE
                            WHEN lower(COALESCE(me.path, '')) LIKE '%_int8.engine' THEN 'int8'
                            ELSE 'fp16'
                        END
                    ) AS precision
                FROM model_exports me
                WHERE lower(me.export_type) IN ('tensorrt', 'trt')
            )
            SELECT COUNT(*)
            FROM trt_exports te
            LEFT JOIN tensorrt_models tm
              ON tm.run_id = te.run_id AND tm.precision = te.precision
            WHERE tm.run_id IS NULL;
            """
        ).fetchone()[0]
    )

    if dry_run:
        return {
            "detection_missing": detection_missing,
            "onnx_missing": onnx_missing,
            "tensorrt_missing": tensorrt_missing,
            "detection_inserted": 0,
            "onnx_inserted": 0,
            "tensorrt_inserted": 0,
        }

    registry.conn.execute(
        """
        INSERT INTO detection_models (
            run_id, set_id, model_path, model_sha256, metrics_path, metrics_sha256,
            status, final_metrics_json, metadata_json, created_utc
        )
        SELECT
            tr.run_id,
            tr.set_id,
            tr.model_path,
            tr.model_sha256,
            tr.metrics_path,
            tr.metrics_sha256,
            tr.status,
            tr.final_metrics_json,
            json_object('source', 'backfill_training_runs'),
            tr.created_utc
        FROM training_runs tr
        LEFT JOIN detection_models dm ON dm.run_id = tr.run_id
        WHERE dm.run_id IS NULL;
        """
    )
    detection_inserted = int(registry.conn.execute("SELECT changes();").fetchone()[0])

    registry.conn.execute(
        """
        INSERT INTO onnx_models (
            run_id, set_id, detection_model_run_id, path, sha256, manifest_path,
            manifest_sha256, requires_plugins, plugin_ops_json, plugin_versions_json,
            metadata_json, created_utc
        )
        SELECT
            me.run_id,
            tr.set_id,
            me.run_id,
            me.path,
            json_extract(me.metadata_json, '$.sha256'),
            me.manifest_path,
            json_extract(me.metadata_json, '$.manifest_sha256'),
            json_extract(me.metadata_json, '$.requires_plugins'),
            json_extract(me.metadata_json, '$.plugin_ops'),
            json_extract(me.metadata_json, '$.plugin_versions'),
            me.metadata_json,
            me.created_utc
        FROM model_exports me
        JOIN training_runs tr ON tr.run_id = me.run_id
        LEFT JOIN onnx_models om ON om.run_id = me.run_id
        WHERE lower(me.export_type) = 'onnx'
          AND om.run_id IS NULL;
        """
    )
    onnx_inserted = int(registry.conn.execute("SELECT changes();").fetchone()[0])

    registry.conn.execute(
        """
        WITH trt_exports AS (
            SELECT
                me.run_id AS run_id,
                tr.set_id AS set_id,
                me.path AS path,
                me.manifest_path AS manifest_path,
                me.metadata_json AS metadata_json,
                me.created_utc AS created_utc,
                COALESCE(
                    NULLIF(lower(json_extract(me.metadata_json, '$.precision')), ''),
                    CASE
                        WHEN lower(COALESCE(me.path, '')) LIKE '%_int8.engine' THEN 'int8'
                        ELSE 'fp16'
                    END
                ) AS precision,
                json_extract(me.metadata_json, '$.sha256') AS sha256,
                json_extract(me.metadata_json, '$.manifest_sha256') AS manifest_sha256
            FROM model_exports me
            JOIN training_runs tr ON tr.run_id = me.run_id
            WHERE lower(me.export_type) IN ('tensorrt', 'trt')
        )
        INSERT INTO tensorrt_models (
            run_id, set_id, detection_model_run_id, onnx_run_id, precision, path, sha256,
            manifest_path, manifest_sha256, requires_plugins, plugin_ops_json,
            plugin_versions_json, metadata_json, created_utc
        )
        SELECT
            te.run_id,
            te.set_id,
            te.run_id,
            te.run_id,
            te.precision,
            te.path,
            te.sha256,
            te.manifest_path,
            te.manifest_sha256,
            json_extract(te.metadata_json, '$.requires_plugins'),
            json_extract(te.metadata_json, '$.plugin_ops'),
            json_extract(te.metadata_json, '$.plugin_versions'),
            te.metadata_json,
            te.created_utc
        FROM trt_exports te
        LEFT JOIN tensorrt_models tm
          ON tm.run_id = te.run_id AND tm.precision = te.precision
        WHERE tm.run_id IS NULL;
        """
    )
    tensorrt_inserted = int(registry.conn.execute("SELECT changes();").fetchone()[0])
    registry.conn.commit()
    return {
        "detection_missing": detection_missing,
        "onnx_missing": onnx_missing,
        "tensorrt_missing": tensorrt_missing,
        "detection_inserted": detection_inserted,
        "onnx_inserted": onnx_inserted,
        "tensorrt_inserted": tensorrt_inserted,
    }


def _quality_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("refined_created_utc"),
        row.get("source_keypoint_run"),
        row.get("keypoint_method"),
        row.get("review_state"),
        row.get("review_intended_use"),
        row.get("review_reviewer"),
        row.get("review_timestamp_utc"),
        row.get("usable_keypoints"),
        row.get("total_keypoints"),
        row.get("usable_keypoints_rate"),
        row.get("raw_keypoints_success_rate"),
        row.get("raw_keypoints_successful"),
        row.get("zarr_mtime_ns"),
    )


def _backfill_keypoint_quality(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
) -> Dict[str, int]:
    rows = registry.conn.execute(
        """
        SELECT dataset_id, zarr_path
        FROM datasets
        WHERE status IS NULL OR lower(status) != 'missing'
        ORDER BY dataset_id;
        """
    ).fetchall()
    scope_roots = _normalize_scope_paths(scope_paths)
    zarr = _import_zarr()
    summary: Dict[str, int] = {
        "datasets_scanned": 0,
        "datasets_skipped_existing": 0,
        "datasets_missing": 0,
        "datasets_errors": 0,
        "datasets_no_quality": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "rows_deleted": 0,
    }

    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM keypoint_quality WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchall()
        if not refresh and existing_rows:
            summary["datasets_skipped_existing"] += 1
            summary["rows_skipped"] += len(existing_rows)
            continue

        try:
            try:
                root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
            except TypeError:
                root = zarr.open_group(str(zarr_path), mode="r")
            extracted_rows = _extract_keypoint_quality_rows(root, zarr_path=zarr_path)
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_quality"] += 1

        existing_by_refined: Dict[str, Dict[str, object]] = {
            str(existing["refined_run"]): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_refined: Dict[str, Dict[str, object]] = {
            str(extracted["refined_run"]): extracted for extracted in extracted_rows
        }

        for refined_run, extracted in extracted_by_refined.items():
            existing = existing_by_refined.get(refined_run)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _quality_row_signature(existing)
            extracted_sig = _quality_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for refined_run in existing_by_refined:
                if refined_run not in extracted_by_refined:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_keypoint_quality(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_keypoint_quality(
                    dataset_id=dataset_id,
                    refined_run=str(extracted["refined_run"]),
                    refined_created_utc=extracted.get("refined_created_utc"),
                    source_keypoint_run=str(extracted["source_keypoint_run"]),
                    keypoint_method=extracted.get("keypoint_method"),
                    review_state=extracted.get("review_state"),
                    review_intended_use=extracted.get("review_intended_use"),
                    review_reviewer=extracted.get("review_reviewer"),
                    review_timestamp_utc=extracted.get("review_timestamp_utc"),
                    usable_keypoints=extracted.get("usable_keypoints"),
                    total_keypoints=extracted.get("total_keypoints"),
                    usable_keypoints_rate=extracted.get("usable_keypoints_rate"),
                    raw_keypoints_success_rate=extracted.get("raw_keypoints_success_rate"),
                    raw_keypoints_successful=extracted.get("raw_keypoints_successful"),
                    quality_updated_utc=extracted.get("quality_updated_utc"),
                    zarr_mtime_ns=extracted.get("zarr_mtime_ns"),
                )

    return summary


def _detect_quality_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("refined_created_utc"),
        row.get("source_detect_run"),
        row.get("detect_method"),
        row.get("review_state"),
        row.get("review_intended_use"),
        row.get("review_reviewer"),
        row.get("review_timestamp_utc"),
        row.get("review_resolved_group"),
        row.get("total_detections"),
        row.get("real_detections"),
        row.get("interpolated_detections"),
        row.get("interpolated_detections_rate"),
        row.get("zarr_mtime_ns"),
    )


def _backfill_detect_quality(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
) -> Dict[str, int]:
    rows = registry.conn.execute(
        """
        SELECT dataset_id, zarr_path
        FROM datasets
        WHERE status IS NULL OR lower(status) != 'missing'
        ORDER BY dataset_id;
        """
    ).fetchall()
    scope_roots = _normalize_scope_paths(scope_paths)
    zarr = _import_zarr()
    summary: Dict[str, int] = {
        "datasets_scanned": 0,
        "datasets_skipped_existing": 0,
        "datasets_missing": 0,
        "datasets_errors": 0,
        "datasets_no_quality": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "rows_deleted": 0,
    }

    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM detect_quality WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchall()
        if not refresh and existing_rows:
            summary["datasets_skipped_existing"] += 1
            summary["rows_skipped"] += len(existing_rows)
            continue

        try:
            try:
                root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
            except TypeError:
                root = zarr.open_group(str(zarr_path), mode="r")
            extracted_rows = _extract_detect_quality_rows(root, zarr_path=zarr_path)
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_quality"] += 1

        existing_by_refined: Dict[str, Dict[str, object]] = {
            str(existing["refined_run"]): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_refined: Dict[str, Dict[str, object]] = {
            str(extracted["refined_run"]): extracted for extracted in extracted_rows
        }

        for refined_run, extracted in extracted_by_refined.items():
            existing = existing_by_refined.get(refined_run)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _detect_quality_row_signature(existing)
            extracted_sig = _detect_quality_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for refined_run in existing_by_refined:
                if refined_run not in extracted_by_refined:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_detect_quality(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_detect_quality(
                    dataset_id=dataset_id,
                    refined_run=str(extracted["refined_run"]),
                    refined_created_utc=extracted.get("refined_created_utc"),
                    source_detect_run=str(extracted["source_detect_run"]),
                    detect_method=extracted.get("detect_method"),
                    review_state=extracted.get("review_state"),
                    review_intended_use=extracted.get("review_intended_use"),
                    review_reviewer=extracted.get("review_reviewer"),
                    review_timestamp_utc=extracted.get("review_timestamp_utc"),
                    review_resolved_group=extracted.get("review_resolved_group"),
                    total_detections=extracted.get("total_detections"),
                    real_detections=extracted.get("real_detections"),
                    interpolated_detections=extracted.get("interpolated_detections"),
                    interpolated_detections_rate=extracted.get("interpolated_detections_rate"),
                    quality_updated_utc=extracted.get("quality_updated_utc"),
                    zarr_mtime_ns=extracted.get("zarr_mtime_ns"),
                )

    return summary


def _check_registry_integrity(registry: Registry) -> List[IntegrityIssue]:
    issues: List[IntegrityIssue] = []

    def _json_list(raw: object) -> set[str]:
        if raw is None:
            return set()
        try:
            payload = json.loads(str(raw))
        except Exception:
            return set()
        if not isinstance(payload, list):
            return set()
        return {str(item) for item in payload if item}

    def _json_dict(raw: object) -> dict[str, str]:
        if raw is None:
            return {}
        try:
            payload = json.loads(str(raw))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in payload.items()
            if key and value is not None
        }

    # Every training run should have a detection_models row after migration.
    missing_dm_rows = registry.conn.execute(
        """
        SELECT tr.run_id
        FROM training_runs tr
        LEFT JOIN detection_models dm ON dm.run_id = tr.run_id
        WHERE dm.run_id IS NULL
        ORDER BY tr.created_utc DESC, tr.run_id DESC;
        """
    ).fetchall()
    for row in missing_dm_rows:
        issues.append(
            IntegrityIssue(
                code="missing_detection_model_row",
                run_id=str(row["run_id"]),
                detail="training_runs row has no detection_models row",
            )
        )

    # Success runs should have model and metrics path populated and existing (if present).
    success_rows = registry.conn.execute(
        """
        SELECT run_id, model_path, metrics_path
        FROM detection_models
        WHERE lower(COALESCE(status, '')) = 'success'
        ORDER BY created_utc DESC, run_id DESC;
        """
    ).fetchall()
    for row in success_rows:
        run_id = str(row["run_id"])
        model_path = row["model_path"]
        metrics_path = row["metrics_path"]
        if not model_path:
            issues.append(
                IntegrityIssue(
                    code="success_missing_model_path",
                    run_id=run_id,
                    detail="detection_models.status=success but model_path is NULL/empty",
                )
            )
        elif not Path(str(model_path)).exists():
            issues.append(
                IntegrityIssue(
                    code="missing_model_file",
                    run_id=run_id,
                    detail=f"model_path does not exist: {model_path}",
                )
            )
        if metrics_path and not Path(str(metrics_path)).exists():
            issues.append(
                IntegrityIssue(
                    code="missing_metrics_file",
                    run_id=run_id,
                    detail=f"metrics_path does not exist: {metrics_path}",
                )
            )

    # ONNX/TRT rows must have paths and (if set) existing manifests.
    onnx_rows = registry.conn.execute(
        """
        SELECT run_id, path, manifest_path
        FROM onnx_models
        ORDER BY created_utc DESC, run_id DESC;
        """
    ).fetchall()
    for row in onnx_rows:
        run_id = str(row["run_id"])
        path = row["path"]
        manifest_path = row["manifest_path"]
        if not path:
            issues.append(
                IntegrityIssue(
                    code="onnx_missing_path",
                    run_id=run_id,
                    detail="onnx_models.path is NULL/empty",
                )
            )
        elif not Path(str(path)).exists():
            issues.append(
                IntegrityIssue(
                    code="onnx_file_missing",
                    run_id=run_id,
                    detail=f"onnx path does not exist: {path}",
                )
            )
        if manifest_path and not Path(str(manifest_path)).exists():
            issues.append(
                IntegrityIssue(
                    code="onnx_manifest_missing",
                    run_id=run_id,
                    detail=f"onnx manifest path does not exist: {manifest_path}",
                )
            )

    trt_rows = registry.conn.execute(
        """
        SELECT run_id, onnx_run_id, precision, path, manifest_path,
               requires_plugins, plugin_ops_json, plugin_versions_json
        FROM tensorrt_models
        ORDER BY created_utc DESC, run_id DESC;
        """
    ).fetchall()
    for row in trt_rows:
        run_id = str(row["run_id"])
        onnx_run_id = str(row["onnx_run_id"] or row["run_id"])
        precision = str(row["precision"] or "").strip().lower()
        path = row["path"]
        manifest_path = row["manifest_path"]
        requires_plugins = row["requires_plugins"]
        plugin_ops_json = row["plugin_ops_json"]
        plugin_versions_json = row["plugin_versions_json"]
        if not precision:
            issues.append(
                IntegrityIssue(
                    code="trt_missing_precision",
                    run_id=run_id,
                    detail="tensorrt_models.precision is NULL/empty",
                )
            )
        if not path:
            issues.append(
                IntegrityIssue(
                    code="trt_missing_path",
                    run_id=run_id,
                    detail="tensorrt_models.path is NULL/empty",
                )
            )
        elif not Path(str(path)).exists():
            issues.append(
                IntegrityIssue(
                    code="trt_file_missing",
                    run_id=run_id,
                    detail=f"tensorrt path does not exist: {path}",
                )
            )
        if manifest_path and not Path(str(manifest_path)).exists():
            issues.append(
                IntegrityIssue(
                    code="trt_manifest_missing",
                    run_id=run_id,
                    detail=f"tensorrt manifest path does not exist: {manifest_path}",
                )
            )
        if requires_plugins and not plugin_ops_json:
            issues.append(
                IntegrityIssue(
                    code="trt_plugins_missing_ops",
                    run_id=run_id,
                    detail=f"precision={precision} requires_plugins=1 but plugin_ops_json is empty",
                )
            )

        onnx_row = registry.conn.execute(
            """
            SELECT requires_plugins, plugin_ops_json, plugin_versions_json
            FROM onnx_models
            WHERE run_id = ?;
            """,
            (onnx_run_id,),
        ).fetchone()
        if not onnx_row:
            continue
        trt_requires = bool(requires_plugins) if requires_plugins is not None else False
        onnx_requires = (
            bool(onnx_row["requires_plugins"])
            if onnx_row["requires_plugins"] is not None
            else False
        )
        trt_ops = _json_list(plugin_ops_json)
        onnx_ops = _json_list(onnx_row["plugin_ops_json"])
        trt_versions = _json_dict(plugin_versions_json)
        onnx_versions = _json_dict(onnx_row["plugin_versions_json"])
        if (
            trt_requires != onnx_requires
            or trt_ops != onnx_ops
            or trt_versions != onnx_versions
        ):
            issues.append(
                IntegrityIssue(
                    code="trt_plugin_contract_mismatch",
                    run_id=run_id,
                    detail=(
                        f"precision={precision} onnx_run_id={onnx_run_id} "
                        f"trt_requires={int(trt_requires)} onnx_requires={int(onnx_requires)}"
                    ),
                )
            )

    # Keypoint quality rows should be fresh and consistent with current Zarr metadata.
    quality_rows = registry.conn.execute(
        """
        SELECT
            kqc.dataset_id,
            d.zarr_path,
            kqc.refined_run,
            kqc.source_keypoint_run,
            kqc.keypoint_method,
            kqc.review_state,
            kqc.review_intended_use,
            kqc.usable_keypoints,
            kqc.total_keypoints,
            kqc.usable_keypoints_rate,
            kqc.zarr_mtime_ns
        FROM keypoint_quality_current kqc
        JOIN datasets d ON d.dataset_id = kqc.dataset_id
        ORDER BY kqc.dataset_id;
        """
    ).fetchall()
    zarr = _import_zarr()
    extracted_cache: dict[str, dict[str, dict[str, object]]] = {}
    for row in quality_rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"]))
        refined_run = str(row["refined_run"])
        recorded_mtime = row["zarr_mtime_ns"]
        try:
            actual_mtime = int(zarr_path.stat().st_mtime_ns)
        except Exception:
            actual_mtime = None
        if recorded_mtime is None:
            issues.append(
                IntegrityIssue(
                    code="keypoint_quality_missing_mtime",
                    run_id=dataset_id,
                    detail=f"dataset_id={dataset_id} refined_run={refined_run}",
                )
            )
        elif actual_mtime is not None and int(recorded_mtime) != int(actual_mtime):
            issues.append(
                IntegrityIssue(
                    code="keypoint_quality_stale",
                    run_id=dataset_id,
                    detail=f"dataset_id={dataset_id} refined_run={refined_run}",
                )
            )
        cache_key = str(zarr_path)
        if cache_key not in extracted_cache:
            try:
                try:
                    root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
                except TypeError:
                    root = zarr.open_group(str(zarr_path), mode="r")
                extracted = _extract_keypoint_quality_rows(root, zarr_path=zarr_path)
                extracted_cache[cache_key] = {str(item["refined_run"]): item for item in extracted}
            except Exception:
                issues.append(
                    IntegrityIssue(
                        code="keypoint_quality_read_error",
                        run_id=dataset_id,
                        detail=f"dataset_id={dataset_id} zarr_path={zarr_path}",
                    )
                )
                extracted_cache[cache_key] = {}
        extracted_row = extracted_cache[cache_key].get(refined_run)
        if extracted_row is None:
            issues.append(
                IntegrityIssue(
                    code="keypoint_quality_refined_run_missing",
                    run_id=dataset_id,
                    detail=f"dataset_id={dataset_id} refined_run={refined_run}",
                )
            )
            continue
        if (
            str(extracted_row.get("source_keypoint_run")) != str(row["source_keypoint_run"])
            or str(extracted_row.get("keypoint_method") or "") != str(row["keypoint_method"] or "")
            or str(extracted_row.get("review_state") or "") != str(row["review_state"] or "")
            or str(extracted_row.get("review_intended_use") or "") != str(row["review_intended_use"] or "")
            or int(extracted_row.get("usable_keypoints") or -1) != int(row["usable_keypoints"] or -1)
            or int(extracted_row.get("total_keypoints") or -1) != int(row["total_keypoints"] or -1)
            or float(extracted_row.get("usable_keypoints_rate") or -1.0)
            != float(row["usable_keypoints_rate"] or -1.0)
        ):
            issues.append(
                IntegrityIssue(
                    code="keypoint_quality_divergent",
                    run_id=dataset_id,
                    detail=f"dataset_id={dataset_id} refined_run={refined_run}",
                )
            )

    return issues


def _print_candidates(candidates: Sequence[InvalidDatasetCandidate], *, list_limit: int) -> None:
    if not candidates:
        print("No invalid dataset rows found.")
        return

    print(f"Invalid dataset rows: {len(candidates)}")
    limit = len(candidates) if list_limit == 0 else min(len(candidates), list_limit)
    for candidate in candidates[:limit]:
        reasons = ",".join(candidate.reasons)
        print(f" - {candidate.dataset_id} [{reasons}]")
        print(f"   {candidate.zarr_path}")
    if limit < len(candidates):
        print(f" ... {len(candidates) - limit} more rows omitted (use --list-limit 0 to show all).")


def _print_failed_run_candidates(candidates: Sequence[FailedRunCandidate], *, list_limit: int) -> None:
    if not candidates:
        print("No failed training runs found.")
        return

    print(f"Failed training runs: {len(candidates)}")
    limit = len(candidates) if list_limit == 0 else min(len(candidates), list_limit)
    for candidate in candidates[:limit]:
        set_id = candidate.set_id or "—"
        status = candidate.status or "—"
        created_utc = candidate.created_utc or "—"
        print(f" - {candidate.run_id} [set={set_id} status={status} created={created_utc}]")
    if limit < len(candidates):
        print(f" ... {len(candidates) - limit} more rows omitted (use --list-limit 0 to show all).")


def _print_empty_training_set_candidates(
    candidates: Sequence[EmptyTrainingSetCandidate],
    *,
    list_limit: int,
) -> None:
    if not candidates:
        print("No empty training sets found.")
        return

    print(f"Empty training sets (no linked runs): {len(candidates)}")
    limit = len(candidates) if list_limit == 0 else min(len(candidates), list_limit)
    for candidate in candidates[:limit]:
        name = candidate.name or "—"
        created_utc = candidate.created_utc or "—"
        print(f" - {candidate.set_id} [name={name} created={created_utc}]")
    if limit < len(candidates):
        print(f" ... {len(candidates) - limit} more rows omitted (use --list-limit 0 to show all).")


def _summarize_reconcile(stats: Dict[str, int]) -> None:
    checked = int(stats.get("checked", 0))
    marked_missing = int(stats.get("marked_missing", 0))
    print(f"Reconcile missing: checked={checked}, marked_missing={marked_missing}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    if (
        not args.prune_invalid
        and not args.prune_failed_runs
        and not args.prune_empty_sets
        and not args.backfill_model_tables
        and not args.backfill_keypoint_quality
        and not args.backfill_detect_quality
        and not args.refresh_keypoint_quality
        and not args.refresh_detect_quality
        and not args.check_integrity
        and not args.vacuum
    ):
        raise SystemExit(
            "No action selected. Use --prune-invalid, --prune-failed-runs, "
            "--prune-empty-sets, --backfill-model-tables, --backfill-keypoint-quality, "
            "--backfill-detect-quality, --refresh-keypoint-quality, --refresh-detect-quality, "
            "--check-integrity, and/or --vacuum."
        )

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    scope_paths = [Path(path).expanduser() for path in args.paths]

    registry = Registry(registry_path)
    try:
        if args.prune_invalid:
            if args.dry_run:
                print("Dry run: reconcile step is simulated (no status fields are updated).")
                candidates = _collect_invalid_dataset_candidates(
                    registry,
                    scope_paths=scope_paths or None,
                    include_missing_scan=True,
                )
            else:
                stats = registry.reconcile_missing_datasets(scope_paths=scope_paths or None)
                _summarize_reconcile(stats)
                candidates = _collect_invalid_dataset_candidates(registry, scope_paths=scope_paths or None)
            _print_candidates(candidates, list_limit=args.list_limit)
            dataset_ids = [candidate.dataset_id for candidate in candidates]
            if args.dry_run:
                print(f"Dry run: would delete {len(dataset_ids)} dataset row(s).")
            else:
                deleted = _delete_dataset_ids(registry, dataset_ids, dry_run=False)
                print(f"Deleted {deleted} dataset row(s).")

        if args.prune_failed_runs:
            status_values = _normalize_status_values(args.failed_status)
            failed_run_candidates = _collect_failed_run_candidates(
                registry,
                status_values=status_values,
            )
            _print_failed_run_candidates(failed_run_candidates, list_limit=args.list_limit)
            run_ids = [candidate.run_id for candidate in failed_run_candidates]
            status_list = ", ".join(status_values)
            if args.dry_run:
                print(
                    f"Dry run: would delete {len(run_ids)} training run row(s) "
                    f"with status in [{status_list}]."
                )
            else:
                deleted = _delete_training_run_ids(registry, run_ids, dry_run=False)
                print(f"Deleted {deleted} training run row(s) with status in [{status_list}].")

        if args.prune_empty_sets:
            empty_set_candidates = _collect_empty_training_set_candidates(registry)
            _print_empty_training_set_candidates(empty_set_candidates, list_limit=args.list_limit)
            set_ids = [candidate.set_id for candidate in empty_set_candidates]
            if args.dry_run:
                print(f"Dry run: would delete {len(set_ids)} training set row(s) with no linked runs.")
            else:
                deleted = _delete_training_set_ids(registry, set_ids, dry_run=False)
                print(f"Deleted {deleted} training set row(s) with no linked runs.")

        if args.backfill_model_tables:
            summary = _backfill_model_tables(registry, dry_run=bool(args.dry_run))
            print(
                "Backfill candidates: "
                f"detection={summary['detection_missing']} "
                f"onnx={summary['onnx_missing']} "
                f"tensorrt={summary['tensorrt_missing']}"
            )
            if args.dry_run:
                print(
                    "Dry run: would insert "
                    f"{summary['detection_missing']} detection_models row(s), "
                    f"{summary['onnx_missing']} onnx_models row(s), "
                    f"{summary['tensorrt_missing']} tensorrt_models row(s)."
                )
            else:
                print(
                    "Inserted "
                    f"{summary['detection_inserted']} detection_models row(s), "
                    f"{summary['onnx_inserted']} onnx_models row(s), "
                    f"{summary['tensorrt_inserted']} tensorrt_models row(s)."
                )

        if args.backfill_keypoint_quality or args.refresh_keypoint_quality:
            summary = _backfill_keypoint_quality(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_keypoint_quality),
            )
            mode = "refresh" if args.refresh_keypoint_quality else "backfill"
            print(
                f"Keypoint quality {mode}: "
                f"scanned={summary['datasets_scanned']} "
                f"missing={summary['datasets_missing']} "
                f"errors={summary['datasets_errors']} "
                f"no_quality={summary['datasets_no_quality']} "
                f"skipped_existing={summary['datasets_skipped_existing']}"
            )
            if args.dry_run:
                print(
                    "Dry run: would apply "
                    f"inserted={summary['rows_inserted']} "
                    f"updated={summary['rows_updated']} "
                    f"deleted={summary['rows_deleted']} "
                    f"unchanged={summary['rows_skipped']} row(s)."
                )
            else:
                print(
                    "Applied "
                    f"inserted={summary['rows_inserted']} "
                    f"updated={summary['rows_updated']} "
                    f"deleted={summary['rows_deleted']} "
                    f"unchanged={summary['rows_skipped']} row(s)."
                )

        if args.backfill_detect_quality or args.refresh_detect_quality:
            summary = _backfill_detect_quality(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_detect_quality),
            )
            mode = "refresh" if args.refresh_detect_quality else "backfill"
            print(
                f"Detect quality {mode}: "
                f"scanned={summary['datasets_scanned']} "
                f"missing={summary['datasets_missing']} "
                f"errors={summary['datasets_errors']} "
                f"no_quality={summary['datasets_no_quality']} "
                f"skipped_existing={summary['datasets_skipped_existing']}"
            )
            if args.dry_run:
                print(
                    "Dry run: would apply "
                    f"inserted={summary['rows_inserted']} "
                    f"updated={summary['rows_updated']} "
                    f"deleted={summary['rows_deleted']} "
                    f"unchanged={summary['rows_skipped']} row(s)."
                )
            else:
                print(
                    "Applied "
                    f"inserted={summary['rows_inserted']} "
                    f"updated={summary['rows_updated']} "
                    f"deleted={summary['rows_deleted']} "
                    f"unchanged={summary['rows_skipped']} row(s)."
                )

        if args.check_integrity:
            issues = _check_registry_integrity(registry)
            if not issues:
                print("Integrity check passed: no issues found.")
            else:
                print(f"Integrity check failed: {len(issues)} issue(s) found.")
                limit = len(issues) if args.list_limit == 0 else min(len(issues), args.list_limit)
                for issue in issues[:limit]:
                    run_id = issue.run_id or "—"
                    print(f" - [{issue.code}] run={run_id} :: {issue.detail}")
                if limit < len(issues):
                    print(
                        f" ... {len(issues) - limit} more issues omitted "
                        "(use --list-limit 0 to show all)."
                    )
                raise SystemExit(2)

        if args.vacuum:
            if args.dry_run:
                print("Dry run: would run VACUUM.")
            else:
                registry.conn.commit()
                registry.conn.execute("VACUUM;")
                print("VACUUM complete.")
    finally:
        registry.close()


if __name__ == "__main__":
    main()
