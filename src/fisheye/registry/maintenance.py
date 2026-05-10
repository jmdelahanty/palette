"""Maintenance CLI for cleaning stale/invalid registry rows."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import PurePosixPath
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .db import (
    Registry,
    RegistryPaths,
    _extract_crop_quality_rows,
    _extract_detect_performance_rows,
    _extract_detect_quality_rows,
    _extract_eye_mask_quality_rows,
    _extract_eye_mask_performance_rows,
    _extract_keypoint_performance_rows,
    _extract_keypoint_profile_rows,
    _extract_keypoint_quality_rows,
    _extract_subject_mask_component_quality_rows,
    _extract_subject_mask_performance_rows,
    _import_zarr,
    _open_zarr_group_non_consolidated,
)
from fisheye.shared.experiment_setup import subdish_required
from fisheye.tracking.single_subject_per_arena import build_tracking_qc_fields

DEFAULT_ALLOWED_RECORDING_TYPES = {
    "behavior",
    "microscopy",
    "histology",
}
DEFAULT_ALLOWED_RECORDING_SUBTYPES_BY_TYPE = {
    "behavior": {"free", "embedded"},
    "microscopy": {"lightsheet", "confocal", "2p"},
    "histology": {"section", "wholemount"},
}
ALLOWED_BEHAVIOR_MODES = {"free", "embedded", "none"}
RECORDING_TUNING_STEP_NAMES: tuple[str, ...] = (
    "dish_mask",
    "detection_tuning",
    "keypoint_tuning",
    "subject_mask_tuning",
    "eye_mask_tuning",
    "subdish_mask_tuning",
)
RECORDING_STEP_NAMES: tuple[str, ...] = (
    "raw",
    "background",
    "detect",
    "refined_detect",
    "crop",
    "keypoints",
    "refined_keypoints",
    "eye_masks",
    "refined_eye_masks",
    "subject_masks",
    "refined_subject_masks",
    "arena_assignment",
    "tracks",
    "stimulus",
    "calibration",
    *RECORDING_TUNING_STEP_NAMES,
)
RECORDING_STEP_STATUS_VALUES: tuple[str, ...] = ("ok", "missing", "absent", "na", "error")


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
class StaleInProgressRunCandidate:
    run_id: str
    set_id: Optional[str]
    task_type: Optional[str]
    run_status: Optional[str]
    model_status: Optional[str]
    effective_status: Optional[str]
    created_utc: Optional[str]
    age_hours: Optional[float]


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


@dataclass(frozen=True)
class DatasetLineageAuditSummary:
    edge_count: int
    merged_dataset_count: int
    merged_missing_lineage_count: int
    training_set_lineage_mismatch_count: int


@dataclass(frozen=True)
class SetDeleteCandidate:
    set_id: str
    exists: bool
    run_count: int


@dataclass(frozen=True)
class FileDeletePlan:
    eligible_paths: Tuple[Path, ...]
    skipped_paths: Tuple[Tuple[Path, str], ...]
    existing_paths: Tuple[Path, ...]
    existing_bytes: int


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
        "--reconcile-registry",
        action="store_true",
        help=(
            "Run registry reconciliation workflow: reconcile missing dataset statuses, "
            "delete missing dataset rows, then run integrity checks."
        ),
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
        "--prune-missing-datasets",
        action="store_true",
        help=(
            "Reconcile missing rows, then prune dataset rows with missing on-disk Zarr paths "
            "(status=missing)."
        ),
    )
    parser.add_argument(
        "--prune-failed-runs",
        action="store_true",
        help="Prune training_runs rows with failed statuses.",
    )
    parser.add_argument(
        "--reconcile-in-progress-runs",
        action="store_true",
        help=(
            "Mark stale in_progress training runs as failed. "
            "Uses training_runs.created_utc as the latest lifecycle timestamp."
        ),
    )
    parser.add_argument(
        "--in-progress-max-age-hours",
        type=float,
        default=24.0,
        help=(
            "Minimum age (hours) required to treat an in_progress run as stale "
            "when using --reconcile-in-progress-runs (default: 24)."
        ),
    )
    parser.add_argument(
        "--in-progress-task",
        choices=("all", "detect", "pose"),
        default="all",
        help=(
            "Task scope for --reconcile-in-progress-runs. "
            "Uses training_runs.task_type when available with run/set-name fallback (default: all)."
        ),
    )
    parser.add_argument(
        "--delete-run-id",
        action="append",
        help=(
            "Delete a specific training run_id (repeatable or comma-separated). "
            "Cascades to dependent model/export rows."
        ),
    )
    parser.add_argument(
        "--delete-set-id",
        action="append",
        help=(
            "Delete a specific training set_id (repeatable or comma-separated). "
            "By default refuses sets that still have linked runs."
        ),
    )
    parser.add_argument(
        "--delete-set-with-runs",
        action="store_true",
        help=(
            "Allow --delete-set-id to also delete linked training_runs first "
            "(and their cascaded model/export rows)."
        ),
    )
    parser.add_argument(
        "--delete-files",
        action="store_true",
        help=(
            "Also delete on-disk training artifacts for explicit --delete-run-id/--delete-set-id targets. "
            "Never deletes source recordings."
        ),
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
            "Backfill training_models/onnx_models/tensorrt_models from "
            "existing training_runs/model_exports rows."
        ),
    )
    parser.add_argument(
        "--backfill-dataset-lineage",
        action="store_true",
        help=(
            "Backfill dataset_lineage rows for merged training datasets "
            "using training_sets.dataset_ids_json membership."
        ),
    )
    parser.add_argument(
        "--remap-training-set-dataset-ids",
        action="store_true",
        help=(
            "Remap legacy source dataset IDs in training_sets.dataset_ids_json "
            "to current datasets.dataset_id values using datasets.session_uuid."
        ),
    )
    parser.add_argument(
        "--backfill-recording-entities",
        action="store_true",
        help=(
            "Backfill recordings + recording_artifacts and link source datasets "
            "to recording_id/artifact_kind from recording manifests."
        ),
    )
    parser.add_argument(
        "--backfill-subject-dish-cross",
        action="store_true",
        help=(
            "Backfill crosses, dishes, and recording_subjects from source recording "
            "dataset provenance fields."
        ),
    )
    parser.add_argument(
        "--backfill-subjects",
        action="store_true",
        help=(
            "Backfill subjects from recording_subjects/provenance using normalized "
            "dish lineage (subjects.dish_id -> dishes.dish_id)."
        ),
    )
    parser.add_argument(
        "--backfill-keypoint-profiles",
        action="store_true",
        help=(
            "Backfill keypoint_data_profile rows for source recording datasets that currently have no profile rows."
        ),
    )
    parser.add_argument(
        "--backfill-eye-mask-profiles",
        action="store_true",
        help=(
            "Backfill eye_mask_data_profile rows for source recording datasets that currently have no profile rows."
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
        "--backfill-eye-mask-quality",
        action="store_true",
        help=(
            "Backfill eye_mask_quality rows for datasets that currently have no quality rows."
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
        "--backfill-detect-performance",
        action="store_true",
        help=(
            "Backfill detect_performance rows for datasets that currently have no detect performance rows."
        ),
    )
    parser.add_argument(
        "--backfill-keypoint-performance",
        action="store_true",
        help=(
            "Backfill keypoint_performance rows for datasets that currently have no keypoint performance rows."
        ),
    )
    parser.add_argument(
        "--backfill-crop-quality",
        action="store_true",
        help=(
            "Backfill crop_quality rows for datasets that currently have no crop quality rows."
        ),
    )
    parser.add_argument(
        "--backfill-eye-mask-performance",
        action="store_true",
        help=(
            "Backfill eye_mask_performance rows for datasets that currently have no eye-mask performance rows."
        ),
    )
    parser.add_argument(
        "--backfill-subject-mask-performance",
        action="store_true",
        help=(
            "Backfill subject_mask_performance rows for source recording datasets that currently have no subject-mask performance rows."
        ),
    )
    parser.add_argument(
        "--backfill-subject-mask-component-quality",
        action="store_true",
        help=(
            "Backfill subject_mask_component_quality rows for source recording datasets that currently have no subject-mask component rows."
        ),
    )
    parser.add_argument(
        "--backfill-recording-step-status",
        action="store_true",
        help=(
            "Backfill recording_step_status rows from existing Zarrs and append changed rows "
            "to recording_step_status_history."
        ),
    )
    parser.add_argument(
        "--recording-step-recording-id",
        action="append",
        help=(
            "Optional recording_id filter for --backfill-recording-step-status "
            "(repeatable or comma-separated)."
        ),
    )
    parser.add_argument(
        "--recording-step-zarr-use",
        choices=("all", "training", "analysis"),
        default="all",
        help=(
            "Optional zarr_use filter for --backfill-recording-step-status "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--refresh-keypoint-profiles",
        action="store_true",
        help=(
            "Refresh keypoint_data_profile rows for all source recording datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--refresh-eye-mask-profiles",
        action="store_true",
        help=(
            "Refresh eye_mask_data_profile rows for all source recording datasets in scope and remove stale rows."
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
        "--refresh-eye-mask-quality",
        action="store_true",
        help=(
            "Refresh eye_mask_quality rows for all datasets in scope and remove stale rows."
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
        "--refresh-detect-performance",
        action="store_true",
        help=(
            "Refresh detect_performance rows for all datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--refresh-keypoint-performance",
        action="store_true",
        help=(
            "Refresh keypoint_performance rows for all datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--refresh-crop-quality",
        action="store_true",
        help=(
            "Refresh crop_quality rows for all datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--refresh-eye-mask-performance",
        action="store_true",
        help=(
            "Refresh eye_mask_performance rows for all datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--refresh-subject-mask-performance",
        action="store_true",
        help=(
            "Refresh subject_mask_performance rows for all source recording datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--refresh-subject-mask-component-quality",
        action="store_true",
        help=(
            "Refresh subject_mask_component_quality rows for all source recording datasets in scope and remove stale rows."
        ),
    )
    parser.add_argument(
        "--detect-performance-all-datasets",
        action="store_true",
        help=(
            "When backfilling/refreshing detect_performance, include all datasets. "
            "Default scope is source_recording + analysis datasets only."
        ),
    )
    parser.add_argument(
        "--keypoint-performance-all-datasets",
        action="store_true",
        help=(
            "When backfilling/refreshing keypoint_performance, include all datasets. "
            "Default scope is source_recording + analysis datasets only."
        ),
    )
    parser.add_argument(
        "--crop-quality-all-datasets",
        action="store_true",
        help=(
            "When backfilling/refreshing crop_quality, include all datasets. "
            "Default scope is source_recording + analysis datasets only."
        ),
    )
    parser.add_argument(
        "--eye-mask-performance-all-datasets",
        action="store_true",
        help=(
            "When backfilling/refreshing eye_mask_performance, include all datasets. "
            "Default scope is source_recording + analysis datasets only."
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


def _json_text_list(raw: object) -> List[str]:
    if raw is None:
        return []
    try:
        payload = json.loads(str(raw))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if item]


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


def _collect_missing_dataset_candidates(
    registry: Registry,
    *,
    scope_paths: Optional[Sequence[Path]] = None,
    include_missing_scan: bool = False,
) -> List[InvalidDatasetCandidate]:
    """Collect dataset rows that are (or would be) marked missing."""
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


def _normalize_recording_id(*, session_uuid: Optional[str], recording_dir: Path) -> str:
    if session_uuid and str(session_uuid).strip():
        return str(session_uuid).strip()
    digest = sha256(str(recording_dir).encode("utf-8")).hexdigest()[:12]
    return f"path-{digest}"


def _infer_artifact_type(artifact_group: str, relpath: str) -> str:
    rel = relpath.strip()
    rel_lower = rel.lower()
    ext = Path(rel).suffix.lower()
    if artifact_group == "raw":
        if ext == ".h5":
            return "h5_log"
        if ext == ".mp4":
            return "render_video"
        if ext == ".csv":
            if rel_lower.endswith("_update_timing.csv"):
                return "timing_profile_csv"
            return "raw_csv"
    if artifact_group == "cams":
        if ext == ".mp4":
            return "camera_video"
        if ext == ".csv":
            return "camera_metadata_csv"
    if artifact_group == "derived":
        if ext == ".png":
            return "derived_calibration_image"
        if ext:
            return f"derived_{ext.lstrip('.')}"
    if ext:
        return f"file_{ext.lstrip('.')}"
    return "file"


def _iter_manifest_artifacts(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    files = payload.get("files")
    if not isinstance(files, dict):
        return []
    out: List[Tuple[str, str]] = []
    for group in ("raw", "cams", "derived"):
        entries = files.get(group) or []
        if not isinstance(entries, list):
            continue
        for item in entries:
            rel = str(item).strip()
            if not rel:
                continue
            out.append((group, rel))
    return out


def _backfill_recording_entities(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, int]:
    scope_roots = _normalize_scope_paths(scope_paths)
    rows = registry.conn.execute(
        "SELECT dataset_id, session_uuid, zarr_path, recording_id, artifact_kind FROM datasets ORDER BY dataset_id;"
    ).fetchall()

    recordings_scanned = 0
    manifests_missing = 0
    recordings_upserted = 0
    datasets_linked = 0
    artifacts_seen = 0
    artifacts_upserted = 0
    derived_kind_backfilled = 0

    for row in rows:
        dataset_id = str(row["dataset_id"])
        session_uuid = row["session_uuid"]
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        if scope_roots and not _matches_scope(str(zarr_path), scope_roots):
            continue
        normalized = str(zarr_path).replace("\\", "/").lower()
        is_recordings_dataset = "/recordings/" in normalized

        # Backfill artifact kind for non-recording datasets when missing.
        if not is_recordings_dataset and not row["artifact_kind"]:
            inferred_kind = "derived_training_merge" if dataset_id.endswith("_merged") else "derived_analysis"
            derived_kind_backfilled += 1
            if not dry_run:
                registry.conn.execute(
                    "UPDATE datasets SET artifact_kind = ? WHERE dataset_id = ?;",
                    (inferred_kind, dataset_id),
                )
            continue

        if not is_recordings_dataset:
            continue

        recordings_scanned += 1
        try:
            recording_dir = zarr_path.parent.parent
        except Exception:
            manifests_missing += 1
            continue
        manifest_path = recording_dir / "recording_manifest.json"
        if not manifest_path.exists():
            manifests_missing += 1
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifests_missing += 1
            continue

        manifest_session_uuid = manifest.get("session_uuid")
        recording_id = _normalize_recording_id(
            session_uuid=str(manifest_session_uuid or session_uuid or "").strip() or None,
            recording_dir=recording_dir,
        )
        now = registry.conn.execute("SELECT datetime('now') AS now;").fetchone()["now"]
        recording_name = manifest.get("recording_name") or recording_dir.name
        started_utc = manifest.get("session_start_iso8601_utc")
        rig_id = manifest.get("rig_id")
        arena_id = manifest.get("arena_id")
        camera_id = manifest.get("camera_id")
        canvas_name = manifest.get("canvas_name")
        protocol_name = manifest.get("protocol_name_from_definition")
        manifest_dish_design = manifest.get("dish_design")
        provenance_dish_design_row = registry.conn.execute(
            "SELECT dish_design FROM provenance WHERE dataset_id = ? LIMIT 1;",
            (dataset_id,),
        ).fetchone()
        provenance_dish_design = (
            str(provenance_dish_design_row["dish_design"]).strip()
            if provenance_dish_design_row is not None and provenance_dish_design_row["dish_design"] is not None
            else None
        )
        dish_design = (
            str(manifest_dish_design).strip()
            if manifest_dish_design is not None and str(manifest_dish_design).strip()
            else provenance_dish_design
        )
        recording_type = manifest.get("recording_type") or "behavior"
        recording_subtype = manifest.get("recording_subtype")
        if recording_subtype is None and recording_type == "behavior":
            recording_subtype = "free"
        behavior_mode = manifest.get("behavior_mode")
        if behavior_mode is None:
            if recording_type == "behavior" and str(recording_subtype or "").strip() in {"free", "embedded"}:
                behavior_mode = str(recording_subtype)
            else:
                behavior_mode = "none"
        artifact_schema_id = manifest.get("artifact_schema_id") or "behavior_v1"
        experiment_context_status = manifest.get("experiment_context_status")
        experiment_context_source = manifest.get("experiment_context_source")
        experiment_context_status_detail = manifest.get("experiment_context_status_detail")
        stimulus_runs_available_raw = manifest.get("stimulus_runs_available")
        if isinstance(stimulus_runs_available_raw, str):
            stimulus_runs_available = (
                1
                if stimulus_runs_available_raw.strip().lower() in {"1", "true", "yes", "y"}
                else 0
                if stimulus_runs_available_raw.strip().lower() in {"0", "false", "no", "n"}
                else None
            )
        else:
            stimulus_runs_available = (
                None
                if stimulus_runs_available_raw is None
                else int(bool(stimulus_runs_available_raw))
            )

        existing_recording = registry.conn.execute(
            "SELECT 1 FROM recordings WHERE recording_id = ? LIMIT 1;",
            (recording_id,),
        ).fetchone()
        if existing_recording is None:
            recordings_upserted += 1

        if not dry_run:
            registry.conn.execute(
                """
                INSERT INTO recordings (
                    recording_id, session_uuid, recording_name, recording_path, started_utc,
                    recording_type, recording_subtype, behavior_mode, artifact_schema_id,
                    experiment_context_status, experiment_context_source,
                    experiment_context_status_detail, stimulus_runs_available,
                    rig_id, arena_id, camera_id, canvas_name,
                    protocol_name, dish_design, created_utc, updated_utc
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(recording_id) DO UPDATE SET
                    session_uuid=excluded.session_uuid,
                    recording_name=excluded.recording_name,
                    recording_path=excluded.recording_path,
                    started_utc=COALESCE(excluded.started_utc, recordings.started_utc),
                    recording_type=COALESCE(excluded.recording_type, recordings.recording_type),
                    recording_subtype=COALESCE(excluded.recording_subtype, recordings.recording_subtype),
                    behavior_mode=COALESCE(excluded.behavior_mode, recordings.behavior_mode),
                    artifact_schema_id=COALESCE(excluded.artifact_schema_id, recordings.artifact_schema_id),
                    experiment_context_status=COALESCE(
                        excluded.experiment_context_status,
                        recordings.experiment_context_status
                    ),
                    experiment_context_source=COALESCE(
                        excluded.experiment_context_source,
                        recordings.experiment_context_source
                    ),
                    experiment_context_status_detail=COALESCE(
                        excluded.experiment_context_status_detail,
                        recordings.experiment_context_status_detail
                    ),
                    stimulus_runs_available=COALESCE(
                        excluded.stimulus_runs_available,
                        recordings.stimulus_runs_available
                    ),
                    rig_id=COALESCE(excluded.rig_id, recordings.rig_id),
                    arena_id=COALESCE(excluded.arena_id, recordings.arena_id),
                    camera_id=COALESCE(excluded.camera_id, recordings.camera_id),
                    canvas_name=COALESCE(excluded.canvas_name, recordings.canvas_name),
                    protocol_name=COALESCE(excluded.protocol_name, recordings.protocol_name),
                    dish_design=COALESCE(excluded.dish_design, recordings.dish_design),
                    updated_utc=excluded.updated_utc;
                """,
                (
                    recording_id,
                    str(manifest_session_uuid) if manifest_session_uuid else (str(session_uuid) if session_uuid else None),
                    str(recording_name),
                    str(recording_dir),
                    str(started_utc) if started_utc else None,
                    str(recording_type),
                    str(recording_subtype) if recording_subtype else None,
                    str(behavior_mode) if behavior_mode else None,
                    str(artifact_schema_id),
                    str(experiment_context_status) if experiment_context_status else None,
                    str(experiment_context_source) if experiment_context_source else None,
                    str(experiment_context_status_detail) if experiment_context_status_detail else None,
                    stimulus_runs_available,
                    str(rig_id) if rig_id else None,
                    str(arena_id) if arena_id else None,
                    str(camera_id) if camera_id else None,
                    str(canvas_name) if canvas_name else None,
                    str(protocol_name) if protocol_name else None,
                    dish_design,
                    now,
                    now,
                ),
            )

        current_recording_id = row["recording_id"]
        current_artifact_kind = row["artifact_kind"]
        if current_recording_id != recording_id or current_artifact_kind != "source_recording":
            datasets_linked += 1
            if not dry_run:
                registry.conn.execute(
                    "UPDATE datasets SET recording_id = ?, artifact_kind = 'source_recording' WHERE dataset_id = ?;",
                    (recording_id, dataset_id),
                )

        artifact_rows = _iter_manifest_artifacts(manifest)
        artifacts_seen += len(artifact_rows)
        for artifact_group, relpath in artifact_rows:
            artifact_path = recording_dir / relpath
            artifact_type = _infer_artifact_type(artifact_group, relpath)
            file_ext = artifact_path.suffix.lower() if artifact_path.suffix else None
            status = "present" if artifact_path.exists() else "missing"
            size_bytes: Optional[int] = None
            if artifact_path.exists() and artifact_path.is_file():
                try:
                    size_bytes = int(artifact_path.stat().st_size)
                except Exception:
                    size_bytes = None
            existing_artifact = registry.conn.execute(
                "SELECT 1 FROM recording_artifacts WHERE recording_id = ? AND path = ? LIMIT 1;",
                (recording_id, str(artifact_path)),
            ).fetchone()
            if existing_artifact is None:
                artifacts_upserted += 1
            if not dry_run:
                artifact_id = sha256(f"{recording_id}::{artifact_path}".encode("utf-8")).hexdigest()[:16]
                registry.conn.execute(
                    """
                    INSERT INTO recording_artifacts (
                        artifact_id, recording_id, artifact_type, artifact_group, relpath, path,
                        file_ext, status, size_bytes, metadata_json, created_utc, updated_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        artifact_type=excluded.artifact_type,
                        artifact_group=excluded.artifact_group,
                        relpath=excluded.relpath,
                        path=excluded.path,
                        file_ext=excluded.file_ext,
                        status=excluded.status,
                        size_bytes=excluded.size_bytes,
                        metadata_json=excluded.metadata_json,
                        updated_utc=excluded.updated_utc;
                    """,
                    (
                        artifact_id,
                        recording_id,
                        artifact_type,
                        artifact_group,
                        relpath,
                        str(artifact_path),
                        file_ext,
                        status,
                        size_bytes,
                        None,
                        now,
                        now,
                    ),
                )

    if not dry_run:
        registry.conn.commit()

    return {
        "recordings_scanned": recordings_scanned,
        "manifests_missing": manifests_missing,
        "recordings_upserted": recordings_upserted,
        "datasets_linked": datasets_linked,
        "artifacts_seen": artifacts_seen,
        "artifacts_upserted": artifacts_upserted,
        "derived_kind_backfilled": derived_kind_backfilled,
    }


def _backfill_subject_dish_cross_entities(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, int]:
    scope_roots = _normalize_scope_paths(scope_paths)
    rows = registry.conn.execute(
        """
        SELECT
            d.dataset_id,
            d.zarr_path,
            d.recording_id,
            p.fish_id,
            p.dish_id,
            p.cross_id,
            p.species,
            p.sex,
            p.genotype,
            p.line_strain,
            p.parents_json,
            p.dpf_at_acquisition,
            p.subject_count,
            p.snapshot_status,
            p.snapshot_missing_json
        FROM datasets d
        LEFT JOIN provenance p ON p.dataset_id = d.dataset_id
        WHERE d.artifact_kind = 'source_recording'
        ORDER BY d.dataset_id;
        """
    ).fetchall()
    now = registry.conn.execute("SELECT datetime('now') AS now;").fetchone()["now"]
    summary = {
        "source_rows_scanned": 0,
        "crosses_seen": 0,
        "crosses_unique_seen": 0,
        "crosses_would_insert": 0,
        "crosses_upserted": 0,
        "dishes_seen": 0,
        "dishes_unique_seen": 0,
        "dishes_would_insert": 0,
        "dishes_upserted": 0,
        "recording_subject_rows_seen": 0,
        "recording_subjects_unique_seen": 0,
        "recording_subjects_would_insert": 0,
        "recording_subjects_upserted": 0,
        "rows_skipped_missing_recording_id": 0,
        "rows_skipped_missing_subject_id": 0,
    }
    seen_cross_ids: Set[str] = set()
    seen_dish_ids: Set[str] = set()
    seen_recording_subject_keys: Set[Tuple[str, str]] = set()
    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = str(row["zarr_path"])
        if scope_roots and not _matches_scope(zarr_path, scope_roots):
            continue
        summary["source_rows_scanned"] += 1
        recording_id = str(row["recording_id"]).strip() if row["recording_id"] is not None else ""
        legacy_subject_id = str(row["fish_id"]).strip() if row["fish_id"] is not None else ""
        legacy_dish_id = str(row["dish_id"]).strip() if row["dish_id"] is not None else ""
        legacy_cross_id = str(row["cross_id"]).strip() if row["cross_id"] is not None else ""
        legacy_species = str(row["species"]).strip() if row["species"] is not None else ""
        legacy_sex = str(row["sex"]).strip() if row["sex"] is not None else ""
        legacy_genotype = str(row["genotype"]).strip() if row["genotype"] is not None else ""
        legacy_line_strain = str(row["line_strain"]).strip() if row["line_strain"] is not None else ""
        parents_json = str(row["parents_json"]) if row["parents_json"] is not None else None
        snapshot_missing_payload = None
        if row["snapshot_missing_json"] is not None:
            try:
                snapshot_missing_payload = json.loads(str(row["snapshot_missing_json"]))
            except Exception:
                snapshot_missing_payload = row["snapshot_missing_json"]

        normalized_rows = registry.conn.execute(
            """
            SELECT
                subject_id,
                dish_id,
                cross_id,
                dpf_at_acquisition,
                species,
                sex,
                genotype,
                line_strain
            FROM recording_subjects
            WHERE dataset_id = ?
               OR (dataset_id IS NULL AND ? <> '' AND recording_id = ?)
            ORDER BY subject_id;
            """,
            (dataset_id, recording_id, recording_id),
        ).fetchall()

        lineage_rows: list[dict[str, Any]] = []
        if normalized_rows:
            for normalized_row in normalized_rows:
                subject_id = (
                    str(normalized_row["subject_id"]).strip()
                    if normalized_row["subject_id"] is not None
                    else ""
                )
                allow_legacy_fallback = not legacy_subject_id or legacy_subject_id == subject_id
                lineage_rows.append(
                    {
                        "subject_id": subject_id,
                        "dish_id": (
                            str(normalized_row["dish_id"]).strip()
                            if normalized_row["dish_id"] is not None
                            else ""
                        )
                        or (legacy_dish_id if allow_legacy_fallback else ""),
                        "cross_id": (
                            str(normalized_row["cross_id"]).strip()
                            if normalized_row["cross_id"] is not None
                            else ""
                        )
                        or (legacy_cross_id if allow_legacy_fallback else ""),
                        "species": (
                            str(normalized_row["species"]).strip()
                            if normalized_row["species"] is not None
                            else ""
                        )
                        or (legacy_species if allow_legacy_fallback else ""),
                        "sex": (
                            str(normalized_row["sex"]).strip()
                            if normalized_row["sex"] is not None
                            else ""
                        )
                        or (legacy_sex if allow_legacy_fallback else ""),
                        "genotype": (
                            str(normalized_row["genotype"]).strip()
                            if normalized_row["genotype"] is not None
                            else ""
                        )
                        or (legacy_genotype if allow_legacy_fallback else ""),
                        "line_strain": (
                            str(normalized_row["line_strain"]).strip()
                            if normalized_row["line_strain"] is not None
                            else ""
                        )
                        or (legacy_line_strain if allow_legacy_fallback else ""),
                        "dpf_at_acquisition": (
                            normalized_row["dpf_at_acquisition"]
                            if normalized_row["dpf_at_acquisition"] is not None
                            else (row["dpf_at_acquisition"] if allow_legacy_fallback else None)
                        ),
                        "source": "recording_subjects",
                    }
                )
        else:
            lineage_rows.append(
                {
                    "subject_id": legacy_subject_id,
                    "dish_id": legacy_dish_id,
                    "cross_id": legacy_cross_id,
                    "species": legacy_species,
                    "sex": legacy_sex,
                    "genotype": legacy_genotype,
                    "line_strain": legacy_line_strain,
                    "dpf_at_acquisition": row["dpf_at_acquisition"],
                    "source": "provenance_compat",
                }
            )

        for lineage_row in lineage_rows:
            cross_id = str(lineage_row["cross_id"]).strip() if lineage_row["cross_id"] is not None else ""
            dish_id = str(lineage_row["dish_id"]).strip() if lineage_row["dish_id"] is not None else ""
            subject_id = (
                str(lineage_row["subject_id"]).strip() if lineage_row["subject_id"] is not None else ""
            )
            species = (
                str(lineage_row["species"]).strip() if lineage_row["species"] is not None else None
            )
            sex = str(lineage_row["sex"]).strip() if lineage_row["sex"] is not None else None
            genotype = (
                str(lineage_row["genotype"]).strip() if lineage_row["genotype"] is not None else None
            )
            line_strain = (
                str(lineage_row["line_strain"]).strip()
                if lineage_row["line_strain"] is not None
                else None
            )
            dpf_at_acquisition = lineage_row["dpf_at_acquisition"]
            metadata_source = str(lineage_row["source"])

            if cross_id:
                summary["crosses_seen"] += 1
                if cross_id not in seen_cross_ids:
                    seen_cross_ids.add(cross_id)
                    summary["crosses_unique_seen"] += 1
                    existing = registry.conn.execute(
                        "SELECT 1 FROM crosses WHERE cross_id = ? LIMIT 1;",
                        (cross_id,),
                    ).fetchone()
                    if existing is None:
                        summary["crosses_would_insert"] += 1
                        summary["crosses_upserted"] += 1
                if not dry_run:
                    registry.conn.execute(
                        """
                        INSERT INTO crosses (
                            cross_id, line_strain, genotype, parents_json, metadata_json, created_utc, updated_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(cross_id) DO UPDATE SET
                            line_strain=COALESCE(crosses.line_strain, excluded.line_strain),
                            genotype=COALESCE(crosses.genotype, excluded.genotype),
                            parents_json=COALESCE(crosses.parents_json, excluded.parents_json),
                            metadata_json=COALESCE(crosses.metadata_json, excluded.metadata_json),
                            updated_utc=excluded.updated_utc;
                        """,
                        (
                            cross_id,
                            line_strain or None,
                            genotype or None,
                            parents_json,
                            json.dumps(
                                {
                                    "source": metadata_source,
                                    "dataset_id": dataset_id,
                                },
                                sort_keys=True,
                            ),
                            now,
                            now,
                        ),
                    )

            if dish_id:
                summary["dishes_seen"] += 1
                if dish_id not in seen_dish_ids:
                    seen_dish_ids.add(dish_id)
                    summary["dishes_unique_seen"] += 1
                    existing = registry.conn.execute(
                        "SELECT 1 FROM dishes WHERE dish_id = ? LIMIT 1;",
                        (dish_id,),
                    ).fetchone()
                    if existing is None:
                        summary["dishes_would_insert"] += 1
                        summary["dishes_upserted"] += 1
                if not dry_run:
                    registry.conn.execute(
                        """
                        INSERT INTO dishes (
                            dish_id, cross_id, species, metadata_json, created_utc, updated_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(dish_id) DO UPDATE SET
                            cross_id=COALESCE(dishes.cross_id, excluded.cross_id),
                            species=COALESCE(dishes.species, excluded.species),
                            metadata_json=COALESCE(dishes.metadata_json, excluded.metadata_json),
                            updated_utc=excluded.updated_utc;
                        """,
                        (
                            dish_id,
                            cross_id or None,
                            species,
                            json.dumps(
                                {
                                    "source": metadata_source,
                                    "dataset_id": dataset_id,
                                },
                                sort_keys=True,
                            ),
                            now,
                            now,
                        ),
                    )

            if not recording_id:
                summary["rows_skipped_missing_recording_id"] += 1
                continue
            if not subject_id:
                summary["rows_skipped_missing_subject_id"] += 1
                continue
            summary["recording_subject_rows_seen"] += 1
            subject_key = (recording_id, subject_id)
            if subject_key not in seen_recording_subject_keys:
                seen_recording_subject_keys.add(subject_key)
                summary["recording_subjects_unique_seen"] += 1
                existing = registry.conn.execute(
                    """
                    SELECT 1
                    FROM recording_subjects
                    WHERE recording_id = ? AND subject_id = ?
                    LIMIT 1;
                    """,
                    (recording_id, subject_id),
                ).fetchone()
                if existing is None:
                    summary["recording_subjects_would_insert"] += 1
                    summary["recording_subjects_upserted"] += 1
            if dry_run:
                continue

            metadata = {
                "source": metadata_source,
                "dataset_id": dataset_id,
                "subject_count": row["subject_count"],
                "snapshot_status": row["snapshot_status"],
                "snapshot_missing": snapshot_missing_payload,
            }
            registry.conn.execute(
                """
                INSERT INTO recording_subjects (
                    recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition,
                    species, sex, genotype, line_strain, metadata_json, created_utc, updated_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(recording_id, subject_id) DO UPDATE SET
                    dataset_id=COALESCE(recording_subjects.dataset_id, excluded.dataset_id),
                    dish_id=COALESCE(recording_subjects.dish_id, excluded.dish_id),
                    cross_id=COALESCE(recording_subjects.cross_id, excluded.cross_id),
                    dpf_at_acquisition=COALESCE(
                        recording_subjects.dpf_at_acquisition,
                        excluded.dpf_at_acquisition
                    ),
                    species=COALESCE(recording_subjects.species, excluded.species),
                    sex=COALESCE(recording_subjects.sex, excluded.sex),
                    genotype=COALESCE(recording_subjects.genotype, excluded.genotype),
                    line_strain=COALESCE(recording_subjects.line_strain, excluded.line_strain),
                    metadata_json=COALESCE(recording_subjects.metadata_json, excluded.metadata_json),
                    updated_utc=excluded.updated_utc;
                """,
                (
                    recording_id,
                    subject_id,
                    dataset_id,
                    dish_id or None,
                    cross_id or None,
                    dpf_at_acquisition,
                    species,
                    sex,
                    genotype,
                    line_strain,
                    json.dumps(metadata, sort_keys=True),
                    now,
                    now,
                ),
            )
    if not dry_run:
        registry.conn.commit()
    return summary


def _backfill_subjects(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, int]:
    scope_roots = _normalize_scope_paths(scope_paths)
    rows = registry.conn.execute(
        """
        SELECT
            rs.recording_id,
            rs.subject_id,
            rs.dataset_id,
            rs.dish_id AS rs_dish_id,
            rs.species AS rs_species,
            rs.sex AS rs_sex,
            d.zarr_path
        FROM recording_subjects rs
        LEFT JOIN datasets d ON d.dataset_id = rs.dataset_id
        ORDER BY rs.subject_id, rs.recording_id;
        """
    ).fetchall()
    existing_rows = registry.conn.execute(
        """
        SELECT subject_id, dish_id, species, sex
        FROM subjects
        ORDER BY subject_id;
        """
    ).fetchall()
    existing_by_subject: Dict[str, Dict[str, Optional[str]]] = {}
    for row in existing_rows:
        subject_id = str(row["subject_id"]) if row["subject_id"] is not None else ""
        if not subject_id:
            continue
        existing_by_subject[subject_id] = {
            "dish_id": str(row["dish_id"]).strip() if row["dish_id"] is not None else None,
            "species": str(row["species"]).strip() if row["species"] is not None else None,
            "sex": str(row["sex"]).strip() if row["sex"] is not None else None,
        }

    observations: Dict[str, Dict[str, Any]] = {}
    summary = {
        "subject_rows_scanned": 0,
        "rows_skipped_out_of_scope": 0,
        "rows_skipped_missing_subject_id": 0,
        "subject_ids_unique_seen": 0,
        "subjects_existing": 0,
        "subjects_would_insert": 0,
        "subjects_would_enrich": 0,
        "subjects_upserted": 0,
        "subjects_conflict_dish_id": 0,
        "subjects_conflict_species": 0,
        "subjects_conflict_sex": 0,
    }

    for row in rows:
        zarr_path = str(row["zarr_path"]).strip() if row["zarr_path"] is not None else ""
        if scope_roots:
            if not zarr_path or not _matches_scope(zarr_path, scope_roots):
                summary["rows_skipped_out_of_scope"] += 1
                continue
        summary["subject_rows_scanned"] += 1
        subject_id = str(row["subject_id"]).strip() if row["subject_id"] is not None else ""
        if not subject_id:
            summary["rows_skipped_missing_subject_id"] += 1
            continue

        dish_id = str(row["rs_dish_id"]).strip() if row["rs_dish_id"] is not None else ""
        species = str(row["rs_species"]).strip() if row["rs_species"] is not None else ""
        sex = str(row["rs_sex"]).strip() if row["rs_sex"] is not None else ""

        record = observations.setdefault(
            subject_id,
            {
                "dish_ids": set(),
                "species_values": set(),
                "sex_values": set(),
                "row_count": 0,
                "recording_ids": set(),
            },
        )
        record["row_count"] += 1
        recording_id = str(row["recording_id"]).strip() if row["recording_id"] is not None else ""
        if recording_id:
            record["recording_ids"].add(recording_id)
        if dish_id:
            record["dish_ids"].add(dish_id)
        if species:
            record["species_values"].add(species)
        if sex:
            record["sex_values"].add(sex)

    summary["subject_ids_unique_seen"] = len(observations)
    now = registry.conn.execute("SELECT datetime('now') AS now;").fetchone()["now"]
    for subject_id in sorted(observations.keys()):
        observed = observations[subject_id]
        observed_dish_ids = sorted(str(item) for item in observed["dish_ids"] if item)
        observed_species = sorted(str(item) for item in observed["species_values"] if item)
        observed_sex = sorted(str(item) for item in observed["sex_values"] if item)

        existing = existing_by_subject.get(subject_id, {})
        existing_dish_id = str(existing.get("dish_id") or "").strip()
        existing_species = str(existing.get("species") or "").strip()
        existing_sex = str(existing.get("sex") or "").strip()

        if existing:
            summary["subjects_existing"] += 1
        else:
            summary["subjects_would_insert"] += 1
            summary["subjects_upserted"] += 1

        chosen_dish_id = existing_dish_id or (observed_dish_ids[0] if observed_dish_ids else "")
        chosen_species = existing_species or (observed_species[0] if observed_species else "")
        chosen_sex = existing_sex or (observed_sex[0] if observed_sex else "")

        if existing:
            would_enrich = (
                (not existing_dish_id and bool(chosen_dish_id))
                or (not existing_species and bool(chosen_species))
                or (not existing_sex and bool(chosen_sex))
            )
            if would_enrich:
                summary["subjects_would_enrich"] += 1

        if len(observed_dish_ids) > 1 or (
            existing_dish_id and observed_dish_ids and existing_dish_id not in observed_dish_ids
        ):
            summary["subjects_conflict_dish_id"] += 1
        if len(observed_species) > 1 or (
            existing_species and observed_species and existing_species not in observed_species
        ):
            summary["subjects_conflict_species"] += 1
        if len(observed_sex) > 1 or (
            existing_sex and observed_sex and existing_sex not in observed_sex
        ):
            summary["subjects_conflict_sex"] += 1

        if dry_run:
            continue

        metadata_payload = {
            "source": "recording_subjects",
            "row_count": int(observed["row_count"]),
            "recording_ids": sorted(str(item) for item in observed["recording_ids"] if item),
            "observed_dish_ids": observed_dish_ids,
            "observed_species": observed_species,
            "observed_sex": observed_sex,
        }
        registry.conn.execute(
            """
            INSERT INTO subjects (
                subject_id, dish_id, species, sex, metadata_json, created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject_id) DO UPDATE SET
                dish_id=COALESCE(subjects.dish_id, excluded.dish_id),
                species=COALESCE(subjects.species, excluded.species),
                sex=COALESCE(subjects.sex, excluded.sex),
                metadata_json=COALESCE(subjects.metadata_json, excluded.metadata_json),
                updated_utc=excluded.updated_utc;
            """,
            (
                subject_id,
                chosen_dish_id or None,
                chosen_species or None,
                chosen_sex or None,
                json.dumps(metadata_payload, sort_keys=True),
                now,
                now,
            ),
        )

    if not dry_run:
        registry.conn.commit()
    return summary


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


def _normalize_recording_ids(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values or ():
        for token in str(value).split(","):
            recording_id = token.strip()
            if recording_id:
                normalized.add(recording_id)
    return tuple(sorted(normalized))


def _normalize_run_ids(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values or ():
        for token in str(value).split(","):
            run_id = token.strip()
            if run_id:
                normalized.add(run_id)
    return tuple(sorted(normalized))


def _resolve_existing_run_ids(registry: Registry, run_ids: Sequence[str]) -> tuple[List[str], List[str]]:
    existing: List[str] = []
    missing: List[str] = []
    for run_id in run_ids:
        row = registry.conn.execute(
            "SELECT 1 FROM training_runs WHERE run_id = ? LIMIT 1;",
            (run_id,),
        ).fetchone()
        if row is None:
            missing.append(run_id)
        else:
            existing.append(run_id)
    return existing, missing


def _normalize_set_ids(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values or ():
        for token in str(value).split(","):
            set_id = token.strip()
            if set_id:
                normalized.add(set_id)
    return tuple(sorted(normalized))


def _resolve_training_artifact_roots() -> List[Path]:
    roots: List[Path] = []
    env_dataset_root = os.environ.get("PALETTE_TRAINING_DATASETS_ROOT")
    if env_dataset_root:
        roots.append(Path(env_dataset_root).expanduser())
    roots.append(Path("/nvme1/training/datasets"))
    roots.append(Path("/nvme1/models"))
    roots.append((Path.cwd() / "datasets"))
    resolved: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            normalized = root.resolve()
        except Exception:
            normalized = root.absolute()
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(normalized)
    return resolved


def _path_under_any_root(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        if path == root:
            return True
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _looks_like_recordings_path(path: Path) -> bool:
    normalized = PurePosixPath(str(path).replace("\\", "/").lower())
    parts = normalized.parts
    return "recordings" in parts or "/nvme1/recordings" in str(normalized)


def _is_safe_artifact_path(path: Path, roots: Sequence[Path]) -> tuple[bool, str]:
    if _looks_like_recordings_path(path):
        return False, "recordings_path_blocked"
    if not _path_under_any_root(path, roots):
        return False, "outside_training_artifact_roots"
    return True, "ok"


def _coerce_path(path_text: Optional[str]) -> Optional[Path]:
    if not path_text:
        return None
    text = str(path_text).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    try:
        return candidate.resolve()
    except Exception:
        return candidate.absolute()


def _infer_run_dir_for_run_id(path: Path, run_id: str) -> Optional[Path]:
    current = path
    for ancestor in [current] + list(current.parents):
        if ancestor.name == run_id:
            return ancestor
    return None


def _collect_run_artifact_paths(registry: Registry, run_ids: Sequence[str]) -> List[Path]:
    paths: set[Path] = set()
    for run_id in run_ids:
        rows = registry.conn.execute(
            """
            SELECT
                tr.run_id,
                tr.config_path AS tr_config_path,
                tr.manifest_path AS tr_manifest_path,
                tr.model_path AS tr_model_path,
                tr.metrics_path AS tr_metrics_path,
                dm.model_path AS dm_model_path,
                dm.metrics_path AS dm_metrics_path,
                om.path AS onnx_path,
                om.manifest_path AS onnx_manifest_path,
                tm.path AS trt_path,
                tm.manifest_path AS trt_manifest_path,
                me.path AS legacy_export_path,
                me.manifest_path AS legacy_export_manifest_path
            FROM training_runs tr
            LEFT JOIN training_models dm ON dm.run_id = tr.run_id
            LEFT JOIN onnx_models om ON om.run_id = tr.run_id
            LEFT JOIN tensorrt_models tm ON tm.run_id = tr.run_id
            LEFT JOIN model_exports me ON me.run_id = tr.run_id
            WHERE tr.run_id = ?;
            """,
            (run_id,),
        ).fetchall()
        for row in rows:
            for key in (
                "tr_config_path",
                "tr_manifest_path",
                "tr_model_path",
                "tr_metrics_path",
                "dm_model_path",
                "dm_metrics_path",
                "onnx_path",
                "onnx_manifest_path",
                "trt_path",
                "trt_manifest_path",
                "legacy_export_path",
                "legacy_export_manifest_path",
            ):
                candidate = _coerce_path(row[key])
                if candidate is None:
                    continue
                paths.add(candidate)
                inferred_run_dir = _infer_run_dir_for_run_id(candidate, run_id)
                if inferred_run_dir is not None:
                    paths.add(inferred_run_dir)
    return sorted(paths)


def _collect_set_artifact_paths(set_ids: Sequence[str], artifact_roots: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []
    for set_id in set_ids:
        for root in artifact_roots:
            paths.append(root / set_id)
            # Model artifacts are namespaced by task under /.../models/{detect,pose}/<set_id>.
            paths.append(root / "detect" / set_id)
            paths.append(root / "pose" / set_id)
    return paths


def _path_size_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir():
            total = 0
            for sub in path.rglob("*"):
                if sub.is_file():
                    total += int(sub.stat().st_size)
            return total
    except Exception:
        return 0
    return 0


def _build_file_delete_plan(paths: Sequence[Path], *, artifact_roots: Sequence[Path]) -> FileDeletePlan:
    eligible: List[Path] = []
    skipped: List[Tuple[Path, str]] = []
    seen: set[Path] = set()
    for candidate in paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        ok, reason = _is_safe_artifact_path(candidate, artifact_roots)
        if not ok:
            skipped.append((candidate, reason))
            continue
        eligible.append(candidate)
    existing = [path for path in eligible if path.exists()]
    existing_sorted = sorted(existing, key=lambda path: len(path.parts), reverse=True)
    bytes_total = sum(_path_size_bytes(path) for path in existing_sorted)
    return FileDeletePlan(
        eligible_paths=tuple(sorted(eligible)),
        skipped_paths=tuple(sorted(skipped, key=lambda item: str(item[0]))),
        existing_paths=tuple(existing_sorted),
        existing_bytes=int(bytes_total),
    )


def _delete_paths(paths: Sequence[Path], *, dry_run: bool) -> int:
    if dry_run:
        return 0
    deleted = 0
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path)
                deleted += 1
            elif path.is_file():
                path.unlink()
                deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def _collect_set_delete_candidates(registry: Registry, set_ids: Sequence[str]) -> List[SetDeleteCandidate]:
    candidates: List[SetDeleteCandidate] = []
    for set_id in set_ids:
        set_row = registry.conn.execute(
            "SELECT 1 FROM training_sets WHERE set_id = ? LIMIT 1;",
            (set_id,),
        ).fetchone()
        run_count = int(
            registry.conn.execute(
                "SELECT COUNT(*) FROM training_runs WHERE set_id = ?;",
                (set_id,),
            ).fetchone()[0]
        )
        candidates.append(
            SetDeleteCandidate(
                set_id=set_id,
                exists=set_row is not None,
                run_count=run_count,
            )
        )
    return candidates


def _collect_run_ids_for_set_ids(registry: Registry, set_ids: Sequence[str]) -> List[str]:
    run_ids: List[str] = []
    for set_id in set_ids:
        rows = registry.conn.execute(
            """
            SELECT run_id
            FROM training_runs
            WHERE set_id = ?
            ORDER BY created_utc DESC, run_id DESC;
            """,
            (set_id,),
        ).fetchall()
        run_ids.extend(str(row["run_id"]) for row in rows if row["run_id"] is not None)
    return run_ids


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


def _parse_utc_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _infer_training_run_task(
    *,
    task_type: Optional[str],
    set_id: Optional[str],
    run_id: Optional[str],
) -> Optional[str]:
    explicit = str(task_type or "").strip().lower()
    if explicit in {"detect", "pose"}:
        return explicit

    set_text = str(set_id or "").strip().lower()
    run_text = str(run_id or "").strip().lower()
    if "_pose_" in run_text or run_text.startswith("pose_") or run_text.endswith("_pose"):
        return "pose"
    if "_detect_" in run_text or run_text.startswith("detect_") or run_text.endswith("_detect"):
        return "detect"
    if "_pose_" in set_text or set_text.startswith("pose_"):
        return "pose"
    if "_detect_" in set_text or set_text.startswith("detect_"):
        return "detect"
    return None


def _collect_stale_in_progress_run_candidates(
    registry: Registry,
    *,
    max_age_hours: float,
    task_filter: str,
    now_utc: Optional[datetime] = None,
) -> List[StaleInProgressRunCandidate]:
    now = now_utc or datetime.now(timezone.utc)
    threshold_hours = float(max_age_hours)
    if threshold_hours < 0:
        raise ValueError("max_age_hours must be non-negative")
    filter_norm = str(task_filter or "all").strip().lower()
    if filter_norm not in {"all", "detect", "pose"}:
        raise ValueError("task_filter must be one of: all, detect, pose")

    rows = registry.conn.execute(
        """
        SELECT
            tr.run_id,
            tr.set_id,
            tr.task_type,
            tr.status AS run_status,
            dm.status AS model_status,
            tr.created_utc
        FROM training_runs tr
        LEFT JOIN training_models dm ON dm.run_id = tr.run_id
        ORDER BY tr.created_utc DESC, tr.run_id DESC;
        """
    ).fetchall()

    candidates: List[StaleInProgressRunCandidate] = []
    for row in rows:
        run_status = str(row["run_status"]).strip() if row["run_status"] is not None else None
        model_status = str(row["model_status"]).strip() if row["model_status"] is not None else None
        effective_status_raw = model_status or run_status
        effective_status = str(effective_status_raw or "").strip().lower()
        if effective_status != "in_progress":
            continue

        inferred_task = _infer_training_run_task(
            task_type=(str(row["task_type"]) if row["task_type"] is not None else None),
            set_id=(str(row["set_id"]) if row["set_id"] is not None else None),
            run_id=(str(row["run_id"]) if row["run_id"] is not None else None),
        )
        if filter_norm != "all" and inferred_task != filter_norm:
            continue

        created_utc = str(row["created_utc"]).strip() if row["created_utc"] is not None else None
        created_dt = _parse_utc_datetime(created_utc)
        if created_dt is None:
            continue
        age_hours = (now - created_dt).total_seconds() / 3600.0
        if age_hours < threshold_hours:
            continue

        candidates.append(
            StaleInProgressRunCandidate(
                run_id=str(row["run_id"]),
                set_id=(str(row["set_id"]) if row["set_id"] is not None else None),
                task_type=inferred_task,
                run_status=run_status,
                model_status=model_status,
                effective_status=effective_status,
                created_utc=created_utc,
                age_hours=age_hours,
            )
        )

    return candidates


def _json_dict(raw: object) -> Dict[str, Any]:
    if raw is None:
        return {}
    try:
        payload = json.loads(str(raw))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _reconcile_stale_in_progress_runs(
    registry: Registry,
    *,
    candidates: Sequence[StaleInProgressRunCandidate],
    max_age_hours: float,
    task_filter: str,
    dry_run: bool,
    now_utc: Optional[datetime] = None,
) -> int:
    if dry_run or not candidates:
        return 0

    now = now_utc or datetime.now(timezone.utc)
    reconciled_at = now.isoformat()
    threshold_hours = float(max_age_hours)
    filter_norm = str(task_filter or "all").strip().lower()

    with registry.conn:
        for candidate in candidates:
            run_row = registry.conn.execute(
                "SELECT final_metrics_json FROM training_runs WHERE run_id = ?;",
                (candidate.run_id,),
            ).fetchone()
            model_row = registry.conn.execute(
                "SELECT final_metrics_json FROM training_models WHERE run_id = ?;",
                (candidate.run_id,),
            ).fetchone()
            merged_metrics = _json_dict(run_row["final_metrics_json"] if run_row is not None else None)
            if not merged_metrics:
                merged_metrics = _json_dict(model_row["final_metrics_json"] if model_row is not None else None)

            merged_metrics["stage"] = "maintenance_reconcile"
            merged_metrics["status_detail"] = "stale_in_progress_reconciled"
            merged_metrics["error_type"] = "StaleInProgressRun"
            merged_metrics["error_message"] = (
                f"in_progress status older than {threshold_hours:g}h"
            )
            merged_metrics["reconciled_by"] = "fisheye.registry.maintenance"
            merged_metrics["reconciled_at_utc"] = reconciled_at
            merged_metrics["reconcile_policy"] = {
                "max_age_hours": threshold_hours,
                "task_filter": filter_norm,
            }
            if candidate.created_utc:
                merged_metrics["in_progress_since_utc"] = candidate.created_utc
            if candidate.age_hours is not None:
                merged_metrics["in_progress_age_hours"] = round(float(candidate.age_hours), 3)
            if candidate.run_status:
                merged_metrics["previous_run_status"] = candidate.run_status
            if candidate.model_status:
                merged_metrics["previous_model_status"] = candidate.model_status

            payload_json = json.dumps(merged_metrics, sort_keys=True)
            registry.conn.execute(
                """
                UPDATE training_runs
                SET status = ?, final_metrics_json = ?
                WHERE run_id = ?;
                """,
                ("failed", payload_json, candidate.run_id),
            )
            registry.conn.execute(
                """
                UPDATE training_models
                SET status = ?, final_metrics_json = ?
                WHERE run_id = ?;
                """,
                ("failed", payload_json, candidate.run_id),
            )
    return len(candidates)


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
            LEFT JOIN training_models dm ON dm.run_id = tr.run_id
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
        INSERT INTO training_models (
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
        LEFT JOIN training_models dm ON dm.run_id = tr.run_id
        WHERE dm.run_id IS NULL;
        """
    )
    detection_inserted = int(registry.conn.execute("SELECT changes();").fetchone()[0])

    registry.conn.execute(
        """
        INSERT INTO onnx_models (
            run_id, set_id, detection_model_run_id, path, sha256, manifest_path,
            manifest_sha256, nms_conf, nms_iou, nms_topk,
            requires_plugins, plugin_ops_json, plugin_versions_json,
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
            COALESCE(
                json_extract(me.metadata_json, '$.nms.conf'),
                json_extract(me.metadata_json, '$.nms_conf'),
                json_extract(me.metadata_json, '$.conf_threshold')
            ),
            COALESCE(
                json_extract(me.metadata_json, '$.nms.iou'),
                json_extract(me.metadata_json, '$.nms_iou'),
                json_extract(me.metadata_json, '$.iou_threshold')
            ),
            COALESCE(
                json_extract(me.metadata_json, '$.nms.topk'),
                json_extract(me.metadata_json, '$.nms_topk'),
                json_extract(me.metadata_json, '$.topk')
            ),
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
            manifest_path, manifest_sha256, nms_conf, nms_iou, nms_topk,
            requires_plugins, plugin_ops_json,
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
            COALESCE(
                json_extract(te.metadata_json, '$.nms.conf'),
                json_extract(te.metadata_json, '$.nms_conf'),
                json_extract(te.metadata_json, '$.conf_threshold')
            ),
            COALESCE(
                json_extract(te.metadata_json, '$.nms.iou'),
                json_extract(te.metadata_json, '$.nms_iou'),
                json_extract(te.metadata_json, '$.iou_threshold')
            ),
            COALESCE(
                json_extract(te.metadata_json, '$.nms.topk'),
                json_extract(te.metadata_json, '$.nms_topk'),
                json_extract(te.metadata_json, '$.topk')
            ),
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


def _backfill_dataset_lineage(registry: Registry, *, dry_run: bool) -> Dict[str, int]:
    relationship_type = "training_merge_source"
    dataset_rows = registry.conn.execute(
        """
        SELECT dataset_id, artifact_kind
        FROM datasets;
        """
    ).fetchall()
    dataset_kind: Dict[str, str] = {
        str(row["dataset_id"]): str(row["artifact_kind"] or "")
        for row in dataset_rows
        if row["dataset_id"] is not None
    }
    set_rows = registry.conn.execute(
        """
        SELECT set_id, dataset_ids_json
        FROM training_sets
        ORDER BY created_utc DESC, set_id DESC;
        """
    ).fetchall()

    summary = {
        "sets_scanned": 0,
        "merged_scanned": 0,
        "relationships_changed": 0,
        "rows_inserted": 0,
        "rows_deleted": 0,
        "rows_unchanged": 0,
    }
    for set_row in set_rows:
        summary["sets_scanned"] += 1
        set_id = str(set_row["set_id"])
        dataset_ids = sorted(set(_json_text_list(set_row["dataset_ids_json"])))
        merged_ids = [
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id in dataset_kind
            and (
                dataset_kind.get(dataset_id) == "derived_training_merge"
                or dataset_id.endswith("_merged")
            )
        ]
        if not merged_ids:
            continue
        parent_ids = [
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id in dataset_kind and dataset_id not in merged_ids
        ]
        for child_dataset_id in merged_ids:
            summary["merged_scanned"] += 1
            desired_parents: Set[str] = {
                parent_id
                for parent_id in parent_ids
                if parent_id and parent_id != child_dataset_id
            }
            existing_rows = registry.conn.execute(
                """
                SELECT parent_dataset_id
                FROM dataset_lineage
                WHERE child_dataset_id = ? AND relationship_type = ?;
                """,
                (child_dataset_id, relationship_type),
            ).fetchall()
            existing_parents = {
                str(row["parent_dataset_id"])
                for row in existing_rows
                if row["parent_dataset_id"] is not None
            }
            if existing_parents == desired_parents:
                summary["rows_unchanged"] += 1
                continue
            summary["relationships_changed"] += 1
            summary["rows_inserted"] += len(desired_parents - existing_parents)
            summary["rows_deleted"] += len(existing_parents - desired_parents)
            if not dry_run:
                registry.replace_dataset_lineage(
                    child_dataset_id=child_dataset_id,
                    parent_dataset_ids=sorted(desired_parents),
                    relationship_type=relationship_type,
                    source_set_id=set_id,
                    metadata={"producer": "registry.maintenance"},
                )
    return summary


def _remap_training_set_dataset_ids(registry: Registry, *, dry_run: bool) -> Dict[str, int]:
    dataset_rows = registry.conn.execute(
        """
        SELECT dataset_id, session_uuid, artifact_kind
        FROM datasets;
        """
    ).fetchall()
    known_dataset_ids: Set[str] = {
        str(row["dataset_id"])
        for row in dataset_rows
        if row["dataset_id"] is not None
    }
    source_by_session_uuid: Dict[str, str] = {}
    for row in dataset_rows:
        if str(row["artifact_kind"] or "") != "source_recording":
            continue
        session_uuid_raw = row["session_uuid"]
        dataset_id_raw = row["dataset_id"]
        if session_uuid_raw is None or dataset_id_raw is None:
            continue
        session_uuid = str(session_uuid_raw)
        dataset_id = str(dataset_id_raw)
        current = source_by_session_uuid.get(session_uuid)
        if current is None or dataset_id < current:
            source_by_session_uuid[session_uuid] = dataset_id

    set_rows = registry.conn.execute(
        """
        SELECT set_id, dataset_ids_json
        FROM training_sets
        ORDER BY created_utc DESC, set_id DESC;
        """
    ).fetchall()

    summary = {
        "sets_scanned": 0,
        "sets_changed": 0,
        "ids_remapped": 0,
        "ids_unresolved": 0,
    }
    updates: List[Tuple[str, str]] = []
    for row in set_rows:
        summary["sets_scanned"] += 1
        set_id = str(row["set_id"])
        dataset_ids = _json_text_list(row["dataset_ids_json"])
        remapped_ids: List[str] = []
        changed = False
        for dataset_id in dataset_ids:
            if dataset_id in known_dataset_ids:
                remapped_ids.append(dataset_id)
                continue
            mapped_id = source_by_session_uuid.get(dataset_id)
            if mapped_id:
                remapped_ids.append(mapped_id)
                changed = True
                summary["ids_remapped"] += 1
            else:
                remapped_ids.append(dataset_id)
                summary["ids_unresolved"] += 1
        if changed:
            summary["sets_changed"] += 1
            updates.append((set_id, json.dumps(remapped_ids, ensure_ascii=False)))

    if not dry_run and updates:
        with registry.conn:
            for set_id, dataset_ids_json in updates:
                registry.conn.execute(
                    """
                    UPDATE training_sets
                    SET dataset_ids_json = ?
                    WHERE set_id = ?;
                    """,
                    (dataset_ids_json, set_id),
                )
    return summary


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


def _keypoint_profile_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("recording_id"),
        row.get("zarr_use"),
        row.get("keypoint_method"),
        row.get("source_keypoint_path"),
        row.get("source_keypoint_run"),
        row.get("skeleton_id"),
        row.get("kpt_shape"),
        row.get("pose_schema_name"),
        row.get("pose_schema_json"),
        row.get("heading_computation_source"),
        row.get("heading_computation_json"),
        row.get("profile_created_utc"),
        row.get("zarr_mtime_ns"),
        row.get("rows_total"),
        row.get("rows_usable"),
        row.get("usable_keypoints_total"),
        row.get("usable_rate"),
        row.get("confidence_valid_rate"),
        row.get("geometry_valid_rate"),
        row.get("triangle_area_p10"),
        row.get("triangle_area_p50"),
        row.get("triangle_area_p90"),
        row.get("min_angle_p10"),
        row.get("min_angle_p50"),
        row.get("min_angle_p90"),
        row.get("heading_p10"),
        row.get("heading_p50"),
        row.get("heading_p90"),
        row.get("rig_id"),
        row.get("camera_id"),
        row.get("arena_id"),
        row.get("dish_design"),
        row.get("canvas_name"),
        row.get("protocol_name"),
        row.get("genotype"),
        row.get("dpf_at_acquisition"),
        row.get("profile_json"),
    )


def _invoke_with_supported_kwargs(func: object, *args: object, **kwargs: object) -> object:
    if not callable(func):
        raise TypeError("Expected callable.")
    try:
        signature = inspect.signature(func)
    except Exception:
        return func(*args, **kwargs)
    has_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if has_var_kwargs:
        return func(*args, **kwargs)
    filtered_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return func(*args, **filtered_kwargs)


def _row_to_dict(row: object) -> Dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "keys"):
        keys = row.keys()  # type: ignore[attr-defined]
        return {str(key): row[key] for key in keys}  # type: ignore[index]
    raise TypeError("Unsupported row type.")


def _eye_mask_profile_signature_value(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json_text(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                return text
            canonical = _canonical_json_text(parsed)
            return canonical if canonical is not None else text
        return text
    return value


def _eye_mask_data_profile_row_signature(
    row: Dict[str, object],
    *,
    keys: Optional[Sequence[str]] = None,
) -> tuple[object, ...]:
    signature_keys = (
        tuple(keys)
        if keys is not None
        else tuple(
            sorted(
                key
                for key in row.keys()
                if key not in {"dataset_id", "profile_run", "updated_utc", "created_utc"}
            )
        )
    )
    return tuple(_eye_mask_profile_signature_value(row.get(key)) for key in signature_keys)


def _eye_mask_profile_group_names(parent: object) -> List[str]:
    names: List[str] = []
    try:
        if hasattr(parent, "group_keys"):
            names = [str(name) for name in parent.group_keys()]  # type: ignore[attr-defined]
        elif hasattr(parent, "keys"):
            names = [str(name) for name in parent.keys()]  # type: ignore[attr-defined]
    except Exception:
        names = []
    return sorted(name for name in names if name)


def _eye_mask_profile_stat_value(metric_payload: object, key: str) -> Optional[float]:
    metric = _coerce_mapping_value(metric_payload)
    if metric is None:
        return None
    stats = _coerce_mapping_value(metric.get("stats"))
    if stats is not None:
        return _coerce_float_value(stats.get(key))
    return _coerce_float_value(metric.get(key))


def _eye_mask_profile_metric_alias_stat(
    geometry_map: Mapping[str, object],
    *,
    metric_names: Sequence[str],
    stat_key: str,
) -> Optional[float]:
    for metric_name in metric_names:
        value = _eye_mask_profile_stat_value(geometry_map.get(metric_name), stat_key)
        if value is not None:
            return value
    return None


def _extract_eye_mask_profile_rows_fallback(
    root: object,
    *,
    zarr_path: Path,
    dataset_id: str,
    recording_id: Optional[str],
    zarr_use: Optional[str],
    genotype: Optional[str],
    dpf_at_acquisition: Optional[int],
) -> List[Dict[str, Any]]:
    analysis = root.get("analysis") if hasattr(root, "get") else None  # type: ignore[attr-defined]
    if analysis is None or not hasattr(analysis, "get"):
        return []
    runs_parent = analysis.get("eye_mask_profile_runs")  # type: ignore[attr-defined]
    if runs_parent is None:
        return []

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = datetime.now(timezone.utc).isoformat()

    rows: List[Dict[str, Any]] = []
    for profile_run in _eye_mask_profile_group_names(runs_parent):
        try:
            run_group = runs_parent[profile_run]  # type: ignore[index]
        except Exception:
            continue
        attrs = getattr(run_group, "attrs", None)
        summary = _coerce_mapping_value(attrs.get("profile_summary")) if attrs is not None else None  # type: ignore[arg-type]
        if not summary:
            continue

        dataset_map = _coerce_mapping_value(summary.get("dataset")) or {}
        source_map = _coerce_mapping_value(summary.get("source")) or {}
        quality_map = _coerce_mapping_value(summary.get("quality")) or {}
        geometry_map = _coerce_mapping_value(summary.get("geometry")) or {}
        spatial_map = _coerce_mapping_value(summary.get("spatial")) or {}
        composition_map = _coerce_mapping_value(summary.get("composition")) or {}

        row_genotype = _decode_text(composition_map.get("genotype")) or genotype
        row_dpf = _coerce_int_value(composition_map.get("dpf_at_acquisition"))
        if row_dpf is None:
            row_dpf = dpf_at_acquisition

        rows_total = _coerce_int_value(quality_map.get("rows_total"))
        rows_usable = _coerce_int_value(quality_map.get("rows_usable"))
        if rows_usable is None:
            rows_usable = _coerce_int_value(quality_map.get("usable_rows"))
        if rows_usable is None:
            rows_usable = _coerce_int_value(quality_map.get("rows_training_usable"))
        usable_rate = _coerce_float_value(quality_map.get("usable_rate"))
        if usable_rate is None:
            usable_rate = _coerce_float_value(quality_map.get("rows_usable_rate"))

        ellipse_area_map = _coerce_mapping_value(geometry_map.get("ellipse_area")) or {}
        circularity_map = _coerce_mapping_value(geometry_map.get("circularity")) or {}
        interocular_map = _coerce_mapping_value(geometry_map.get("interocular_px")) or {}

        profile_json = _canonical_json_text(summary)
        attrs_recording_id = _decode_text(attrs.get("source_recording_id")) if attrs is not None else None
        attrs_zarr_use = _decode_text(attrs.get("source_zarr_use")) if attrs is not None else None
        attrs_stage_group = _decode_text(attrs.get("source_stage_group")) if attrs is not None else None
        attrs_method = _decode_text(attrs.get("source_eye_mask_method")) if attrs is not None else None
        attrs_source_eye_mask_path = _decode_text(attrs.get("source_eye_mask_path")) if attrs is not None else None
        attrs_source_eye_mask_run = _decode_text(attrs.get("source_eye_mask_run")) if attrs is not None else None
        attrs_source_eye_masks_run = _decode_text(attrs.get("source_eye_masks_run")) if attrs is not None else None
        attrs_source_refined_eye_masks_run = (
            _decode_text(attrs.get("source_refined_eye_masks_run")) if attrs is not None else None
        )
        attrs_source_crop_run = _decode_text(attrs.get("source_crop_run")) if attrs is not None else None
        attrs_source_keypoint_path = _decode_text(attrs.get("source_keypoint_path")) if attrs is not None else None
        attrs_source_keypoint_run = _decode_text(attrs.get("source_keypoint_run")) if attrs is not None else None
        attrs_source_keypoints_run = _decode_text(attrs.get("source_keypoints_run")) if attrs is not None else None
        attrs_profile_created = _decode_text(attrs.get("created_at_utc")) if attrs is not None else None

        rows.append(
            {
                "dataset_id": str(dataset_id),
                "profile_run": str(profile_run),
                "recording_id": attrs_recording_id or _decode_text(dataset_map.get("recording_id")) or recording_id,
                "zarr_use": attrs_zarr_use or _decode_text(dataset_map.get("zarr_use")) or zarr_use,
                "stage_group": (
                    attrs_stage_group
                    or _decode_text(source_map.get("stage_group"))
                    or _decode_text(source_map.get("eye_stage_group"))
                    or _decode_text(source_map.get("source_eye_stage"))
                ),
                "eye_mask_method": (
                    attrs_method
                    or _decode_text(source_map.get("eye_mask_method"))
                    or _decode_text(source_map.get("method"))
                ),
                "source_eye_mask_path": (
                    attrs_source_eye_mask_path
                    or _decode_text(source_map.get("eye_mask_path"))
                    or _decode_text(source_map.get("source_eye_mask_path"))
                ),
                "source_eye_mask_run": (
                    attrs_source_eye_mask_run
                    or attrs_source_eye_masks_run
                    or _decode_text(source_map.get("eye_mask_run"))
                    or _decode_text(source_map.get("source_eye_mask_run"))
                    or _decode_text(source_map.get("eye_masks_run"))
                    or _decode_text(source_map.get("source_eye_masks_run"))
                ),
                "source_eye_masks_run": (
                    attrs_source_eye_masks_run
                    or attrs_source_eye_mask_run
                    or _decode_text(source_map.get("eye_masks_run"))
                    or _decode_text(source_map.get("source_eye_masks_run"))
                    or _decode_text(source_map.get("eye_mask_run"))
                    or _decode_text(source_map.get("source_eye_mask_run"))
                ),
                "source_refined_eye_masks_run": (
                    attrs_source_refined_eye_masks_run
                    or _decode_text(source_map.get("refined_eye_masks_run"))
                ),
                "source_crop_run": (
                    attrs_source_crop_run
                    or _decode_text(source_map.get("source_crop_run"))
                    or _decode_text(source_map.get("crop_run"))
                ),
                "source_keypoint_path": (
                    attrs_source_keypoint_path
                    or _decode_text(source_map.get("keypoint_path"))
                    or _decode_text(source_map.get("source_keypoint_path"))
                ),
                "source_keypoint_run": (
                    attrs_source_keypoints_run
                    or attrs_source_keypoint_run
                    or _decode_text(source_map.get("source_keypoints_run"))
                    or _decode_text(source_map.get("keypoints_run"))
                    or _decode_text(source_map.get("keypoint_run"))
                    or _decode_text(source_map.get("source_keypoint_run"))
                ),
                "source_keypoints_run": (
                    attrs_source_keypoints_run
                    or attrs_source_keypoint_run
                    or _decode_text(source_map.get("source_keypoints_run"))
                    or _decode_text(source_map.get("keypoints_run"))
                    or _decode_text(source_map.get("keypoint_run"))
                    or _decode_text(source_map.get("source_keypoint_run"))
                ),
                "profile_created_utc": attrs_profile_created or _decode_text(summary.get("created_at_utc")),
                "rows_total": rows_total,
                "rows_usable": rows_usable,
                "usable_rate": usable_rate,
                "reviewed_rate": _coerce_float_value(quality_map.get("reviewed_rate")),
                "excluded_rate": _coerce_float_value(quality_map.get("excluded_rate")),
                "ellipse_success_rate": _coerce_float_value(quality_map.get("ellipse_success_rate")),
                "pair_success_rate": _coerce_float_value(quality_map.get("pair_success_rate")),
                "area_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("area", "ellipse_area", "union_area", "area_union"),
                    stat_key="p10",
                ),
                "area_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("area", "ellipse_area", "union_area", "area_union"),
                    stat_key="p50",
                ),
                "area_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("area", "ellipse_area", "union_area", "area_union"),
                    stat_key="p90",
                ),
                "left_area_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("left_area", "area_left", "left_eye_area"),
                    stat_key="p10",
                ),
                "left_area_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("left_area", "area_left", "left_eye_area"),
                    stat_key="p50",
                ),
                "left_area_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("left_area", "area_left", "left_eye_area"),
                    stat_key="p90",
                ),
                "right_area_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("right_area", "area_right", "right_eye_area"),
                    stat_key="p10",
                ),
                "right_area_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("right_area", "area_right", "right_eye_area"),
                    stat_key="p50",
                ),
                "right_area_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("right_area", "area_right", "right_eye_area"),
                    stat_key="p90",
                ),
                "union_area_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("union_area", "area_union", "combined_area", "area"),
                    stat_key="p10",
                ),
                "union_area_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("union_area", "area_union", "combined_area", "area"),
                    stat_key="p50",
                ),
                "union_area_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("union_area", "area_union", "combined_area", "area"),
                    stat_key="p90",
                ),
                "area_lr_ratio_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("area_lr_ratio", "left_right_area_ratio", "area_ratio_left_right"),
                    stat_key="p10",
                ),
                "area_lr_ratio_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("area_lr_ratio", "left_right_area_ratio", "area_ratio_left_right"),
                    stat_key="p50",
                ),
                "area_lr_ratio_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("area_lr_ratio", "left_right_area_ratio", "area_ratio_left_right"),
                    stat_key="p90",
                ),
                "major_axis_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("major_axis", "ellipse_major"),
                    stat_key="p10",
                ),
                "major_axis_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("major_axis", "ellipse_major"),
                    stat_key="p50",
                ),
                "major_axis_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("major_axis", "ellipse_major"),
                    stat_key="p90",
                ),
                "minor_axis_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("minor_axis", "ellipse_minor"),
                    stat_key="p10",
                ),
                "minor_axis_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("minor_axis", "ellipse_minor"),
                    stat_key="p50",
                ),
                "minor_axis_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("minor_axis", "ellipse_minor"),
                    stat_key="p90",
                ),
                "aspect_ratio_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("aspect_ratio", "axis_ratio"),
                    stat_key="p10",
                ),
                "aspect_ratio_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("aspect_ratio", "axis_ratio"),
                    stat_key="p50",
                ),
                "aspect_ratio_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("aspect_ratio", "axis_ratio"),
                    stat_key="p90",
                ),
                "eye_separation_p10": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("eye_separation", "interocular_px"),
                    stat_key="p10",
                ),
                "eye_separation_p50": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("eye_separation", "interocular_px"),
                    stat_key="p50",
                ),
                "eye_separation_p90": _eye_mask_profile_metric_alias_stat(
                    geometry_map,
                    metric_names=("eye_separation", "interocular_px"),
                    stat_key="p90",
                ),
                "edge_proximity_rate": _coerce_float_value(
                    spatial_map.get("edge_proximity_rate")
                    if spatial_map.get("edge_proximity_rate") is not None
                    else spatial_map.get("edge_touch_rate")
                ),
                "review_approved_rate": _coerce_float_value(quality_map.get("review_approved_rate")),
                "review_rejected_rate": _coerce_float_value(quality_map.get("review_rejected_rate")),
                "ellipse_area_p10": _eye_mask_profile_stat_value(ellipse_area_map, "p10"),
                "ellipse_area_p50": _eye_mask_profile_stat_value(ellipse_area_map, "p50"),
                "ellipse_area_p90": _eye_mask_profile_stat_value(ellipse_area_map, "p90"),
                "circularity_p10": _eye_mask_profile_stat_value(circularity_map, "p10"),
                "circularity_p50": _eye_mask_profile_stat_value(circularity_map, "p50"),
                "circularity_p90": _eye_mask_profile_stat_value(circularity_map, "p90"),
                "interocular_px_p10": _eye_mask_profile_stat_value(interocular_map, "p10"),
                "interocular_px_p50": _eye_mask_profile_stat_value(interocular_map, "p50"),
                "interocular_px_p90": _eye_mask_profile_stat_value(interocular_map, "p90"),
                "rig_id": _decode_text(composition_map.get("rig_id")),
                "camera_id": _decode_text(composition_map.get("camera_id")),
                "arena_id": _decode_text(composition_map.get("arena_id")),
                "dish_design": _decode_text(composition_map.get("dish_design")),
                "canvas_name": _decode_text(composition_map.get("canvas_name")),
                "protocol_name": _decode_text(composition_map.get("protocol_name")),
                "genotype": row_genotype,
                "dpf_at_acquisition": row_dpf,
                "profile_json": profile_json,
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc,
            }
        )

    return rows


def _extract_eye_mask_profile_rows_for_maintenance(
    root: object,
    *,
    zarr_path: Path,
    dataset_id: str,
    recording_id: Optional[str],
    zarr_use: Optional[str],
    genotype: Optional[str],
    dpf_at_acquisition: Optional[int],
) -> List[Dict[str, Any]]:
    from . import db as registry_db

    extract_fn = getattr(registry_db, "_extract_eye_mask_profile_rows", None)
    if callable(extract_fn):
        extracted = _invoke_with_supported_kwargs(
            extract_fn,
            root,
            zarr_path=zarr_path,
            dataset_id=dataset_id,
            recording_id=recording_id,
            zarr_use=zarr_use,
            genotype=genotype,
            dpf_at_acquisition=dpf_at_acquisition,
        )
        if extracted is None:
            return []
        rows: List[Dict[str, Any]] = []
        for item in extracted:
            rows.append(_row_to_dict(item))
        return rows

    return _extract_eye_mask_profile_rows_fallback(
        root,
        zarr_path=zarr_path,
        dataset_id=dataset_id,
        recording_id=recording_id,
        zarr_use=zarr_use,
        genotype=genotype,
        dpf_at_acquisition=dpf_at_acquisition,
    )


def _eye_mask_profile_table_columns(registry: Registry) -> tuple[str, ...]:
    rows = registry.conn.execute("PRAGMA table_info(eye_mask_data_profile);").fetchall()
    columns: List[str] = []
    for row in rows:
        try:
            name = row["name"]  # type: ignore[index]
        except Exception:
            name = row[1] if len(row) > 1 else None  # type: ignore[index]
        text = _decode_text(name)
        if text:
            columns.append(text)
    return tuple(columns)


def _normalize_eye_mask_profile_db_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json_text(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    return value


def _upsert_eye_mask_profile_row_sql(
    registry: Registry,
    *,
    dataset_id: str,
    row: Dict[str, Any],
    table_columns: Sequence[str],
) -> None:
    if "dataset_id" not in table_columns or "profile_run" not in table_columns:
        raise RuntimeError(
            "eye_mask_data_profile table is missing required dataset_id/profile_run columns."
        )
    profile_run = _decode_text(row.get("profile_run"))
    if not profile_run:
        return

    normalized = dict(row)
    normalized["dataset_id"] = str(dataset_id)
    normalized["profile_run"] = str(profile_run)

    payload: Dict[str, object] = {}
    for column in table_columns:
        if column not in normalized:
            continue
        payload[column] = _normalize_eye_mask_profile_db_value(normalized[column])
    payload["dataset_id"] = str(dataset_id)
    payload["profile_run"] = str(profile_run)

    ordered_columns = [column for column in table_columns if column in payload]
    placeholders = ", ".join(f":{column}" for column in ordered_columns)
    update_columns = [column for column in ordered_columns if column not in {"dataset_id", "profile_run"}]
    if update_columns:
        conflict_sql = (
            "ON CONFLICT(dataset_id, profile_run) DO UPDATE SET "
            + ", ".join(f"{column}=excluded.{column}" for column in update_columns)
        )
    else:
        conflict_sql = "ON CONFLICT(dataset_id, profile_run) DO NOTHING"

    sql = (
        "INSERT INTO eye_mask_data_profile ("
        + ", ".join(ordered_columns)
        + ") VALUES ("
        + placeholders
        + ") "
        + conflict_sql
        + ";"
    )
    registry.conn.execute(
        sql,
        {column: payload[column] for column in ordered_columns},
    )


def _upsert_eye_mask_profile_row(
    registry: Registry,
    *,
    dataset_id: str,
    row: Dict[str, Any],
) -> None:
    profile_run = _decode_text(row.get("profile_run"))
    if not profile_run:
        return
    payload = dict(row)
    payload["dataset_id"] = str(dataset_id)
    payload["profile_run"] = str(profile_run)

    table_columns = _eye_mask_profile_table_columns(registry)
    _upsert_eye_mask_profile_row_sql(
        registry,
        dataset_id=str(dataset_id),
        row=payload,
        table_columns=table_columns,
    )


def _replace_eye_mask_profile_rows(
    registry: Registry,
    *,
    dataset_id: str,
    records: Sequence[Dict[str, Any]],
) -> None:
    table_columns = _eye_mask_profile_table_columns(registry)
    if "dataset_id" not in table_columns or "profile_run" not in table_columns:
        raise RuntimeError(
            "eye_mask_data_profile table is missing required dataset_id/profile_run columns."
        )
    with registry.conn:
        registry.conn.execute(
            "DELETE FROM eye_mask_data_profile WHERE dataset_id = ?;",
            (str(dataset_id),),
        )
        for row in records:
            _upsert_eye_mask_profile_row_sql(
                registry,
                dataset_id=str(dataset_id),
                row=row,
                table_columns=table_columns,
            )


def _backfill_eye_mask_profiles(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
) -> Dict[str, int]:
    table_columns = _eye_mask_profile_table_columns(registry)
    if "dataset_id" not in table_columns or "profile_run" not in table_columns:
        raise RuntimeError(
            "eye_mask_data_profile table is unavailable. Run eye-mask profile registry migrations first."
        )

    rows = registry.conn.execute(
        """
        SELECT
            d.dataset_id,
            d.zarr_path,
            d.recording_id,
            d.zarr_use,
            p.genotype AS genotype,
            p.dpf_at_acquisition AS dpf_at_acquisition
        FROM datasets d
        LEFT JOIN provenance p ON p.dataset_id = d.dataset_id
        WHERE (status IS NULL OR lower(status) != 'missing')
          AND lower(COALESCE(d.artifact_kind, '')) = 'source_recording'
        ORDER BY d.dataset_id;
        """
    ).fetchall()
    scope_roots = _normalize_scope_paths(scope_paths)
    zarr = _import_zarr()
    summary: Dict[str, int] = {
        "datasets_scanned": 0,
        "datasets_skipped_existing": 0,
        "datasets_missing": 0,
        "datasets_errors": 0,
        "datasets_no_profile": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "rows_deleted": 0,
    }

    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        genotype = _decode_text(row["genotype"])
        dpf_at_acquisition = _coerce_int_value(row["dpf_at_acquisition"])

        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = [
            _row_to_dict(existing)
            for existing in registry.conn.execute(
                "SELECT * FROM eye_mask_data_profile WHERE dataset_id = ?;",
                (dataset_id,),
            ).fetchall()
        ]
        if not refresh and existing_rows:
            summary["datasets_skipped_existing"] += 1
            summary["rows_skipped"] += len(existing_rows)
            continue

        try:
            root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
            extracted_rows = _extract_eye_mask_profile_rows_for_maintenance(
                root,
                zarr_path=zarr_path,
                dataset_id=dataset_id,
                recording_id=recording_id,
                zarr_use=zarr_use,
                genotype=genotype,
                dpf_at_acquisition=dpf_at_acquisition,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_profile"] += 1

        existing_by_run: Dict[str, Dict[str, Any]] = {}
        for existing in existing_rows:
            run_name = _decode_text(existing.get("profile_run"))
            if run_name:
                existing_by_run[run_name] = existing

        extracted_by_run: Dict[str, Dict[str, Any]] = {}
        for extracted in extracted_rows:
            run_name = _decode_text(extracted.get("profile_run"))
            if not run_name:
                continue
            normalized = dict(extracted)
            normalized["dataset_id"] = dataset_id
            normalized["profile_run"] = run_name
            extracted_by_run[run_name] = normalized

        for run_name in sorted(extracted_by_run):
            extracted = extracted_by_run[run_name]
            existing = existing_by_run.get(run_name)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            signature_keys = sorted(
                key
                for key in extracted.keys()
                if key in table_columns
                and key not in {"dataset_id", "profile_run", "updated_utc", "created_utc"}
            )
            existing_sig = _eye_mask_data_profile_row_signature(existing, keys=signature_keys)
            extracted_sig = _eye_mask_data_profile_row_signature(extracted, keys=signature_keys)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for run_name in sorted(existing_by_run):
                if run_name not in extracted_by_run:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        records = [extracted_by_run[run_name] for run_name in sorted(extracted_by_run)]
        if refresh:
            _replace_eye_mask_profile_rows(
                registry,
                dataset_id=dataset_id,
                records=records,
            )
        else:
            with registry.conn:
                for extracted in records:
                    _upsert_eye_mask_profile_row(
                        registry,
                        dataset_id=dataset_id,
                        row=extracted,
                    )

    return summary


def _backfill_keypoint_profiles(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
) -> Dict[str, int]:
    rows = registry.conn.execute(
        """
        SELECT
            d.dataset_id,
            d.zarr_path,
            d.recording_id,
            d.zarr_use,
            p.genotype AS genotype,
            p.dpf_at_acquisition AS dpf_at_acquisition
        FROM datasets d
        LEFT JOIN provenance p ON p.dataset_id = d.dataset_id
        WHERE (status IS NULL OR lower(status) != 'missing')
          AND lower(COALESCE(d.artifact_kind, '')) = 'source_recording'
        ORDER BY d.dataset_id;
        """
    ).fetchall()
    scope_roots = _normalize_scope_paths(scope_paths)
    zarr = _import_zarr()
    summary: Dict[str, int] = {
        "datasets_scanned": 0,
        "datasets_skipped_existing": 0,
        "datasets_missing": 0,
        "datasets_errors": 0,
        "datasets_no_profile": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "rows_deleted": 0,
    }

    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        genotype = _decode_text(row["genotype"])
        dpf_at_acquisition = _coerce_int_value(row["dpf_at_acquisition"])

        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM keypoint_data_profile WHERE dataset_id = ?;",
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
            extracted_rows = _extract_keypoint_profile_rows(
                root,
                zarr_path=zarr_path,
                dataset_id=dataset_id,
                recording_id=recording_id,
                zarr_use=zarr_use,
                genotype=genotype,
                dpf_at_acquisition=dpf_at_acquisition,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_profile"] += 1

        existing_by_run: Dict[str, Dict[str, object]] = {
            str(existing["profile_run"]): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_run: Dict[str, Dict[str, object]] = {
            str(extracted["profile_run"]): extracted for extracted in extracted_rows
        }

        for profile_run, extracted in extracted_by_run.items():
            existing = existing_by_run.get(profile_run)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _keypoint_profile_row_signature(existing)
            extracted_sig = _keypoint_profile_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for profile_run in existing_by_run:
                if profile_run not in extracted_by_run:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_keypoint_data_profile(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_keypoint_data_profile(
                    dataset_id=dataset_id,
                    profile_run=str(extracted["profile_run"]),
                    recording_id=extracted.get("recording_id"),
                    zarr_use=extracted.get("zarr_use"),
                    keypoint_method=extracted.get("keypoint_method"),
                    source_keypoint_path=extracted.get("source_keypoint_path"),
                    source_keypoint_run=extracted.get("source_keypoint_run"),
                    skeleton_id=extracted.get("skeleton_id"),
                    kpt_shape=extracted.get("kpt_shape"),
                    profile_created_utc=extracted.get("profile_created_utc"),
                    rows_total=extracted.get("rows_total"),
                    rows_usable=extracted.get("rows_usable"),
                    usable_keypoints_total=extracted.get("usable_keypoints_total"),
                    usable_rate=extracted.get("usable_rate"),
                    confidence_valid_rate=extracted.get("confidence_valid_rate"),
                    geometry_valid_rate=extracted.get("geometry_valid_rate"),
                    triangle_area_p10=extracted.get("triangle_area_p10"),
                    triangle_area_p50=extracted.get("triangle_area_p50"),
                    triangle_area_p90=extracted.get("triangle_area_p90"),
                    min_angle_p10=extracted.get("min_angle_p10"),
                    min_angle_p50=extracted.get("min_angle_p50"),
                    min_angle_p90=extracted.get("min_angle_p90"),
                    heading_p10=extracted.get("heading_p10"),
                    heading_p50=extracted.get("heading_p50"),
                    heading_p90=extracted.get("heading_p90"),
                    rig_id=extracted.get("rig_id"),
                    camera_id=extracted.get("camera_id"),
                    arena_id=extracted.get("arena_id"),
                    dish_design=extracted.get("dish_design"),
                    canvas_name=extracted.get("canvas_name"),
                    protocol_name=extracted.get("protocol_name"),
                    profile_json=extracted.get("profile_json"),
                    genotype=extracted.get("genotype"),
                    dpf_at_acquisition=extracted.get("dpf_at_acquisition"),
                    zarr_mtime_ns=extracted.get("zarr_mtime_ns"),
                    updated_utc=extracted.get("updated_utc"),
                    pose_schema_name=extracted.get("pose_schema_name"),
                    pose_schema_json=extracted.get("pose_schema_json"),
                    heading_computation_source=extracted.get("heading_computation_source"),
                    heading_computation_json=extracted.get("heading_computation_json"),
                )

    return summary


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


def _eye_mask_quality_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("run_created_utc"),
        row.get("recording_id"),
        row.get("zarr_use"),
        row.get("eye_mask_method"),
        row.get("source_crop_run"),
        row.get("source_keypoint_group"),
        row.get("source_keypoints_run"),
        row.get("source_eye_masks_run"),
        row.get("source_eye_masks_method"),
        row.get("review_state"),
        row.get("review_method"),
        row.get("review_intended_use"),
        row.get("review_reviewer"),
        row.get("review_timestamp_utc"),
        row.get("total_rois"),
        row.get("successful_eyes"),
        row.get("successful_roi_pairs"),
        row.get("successful_roi_pair_rate"),
        row.get("source_keypoint_stale_state"),
        row.get("source_keypoint_stale_reason"),
        row.get("source_keypoint_stale_timestamp_utc"),
        row.get("source_keypoint_stale_json"),
        row.get("lifecycle_state"),
        row.get("lifecycle_reason"),
        row.get("zarr_mtime_ns"),
    )


def _backfill_eye_mask_quality(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
) -> Dict[str, int]:
    rows = registry.conn.execute(
        """
        SELECT dataset_id, zarr_path, recording_id, zarr_use
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
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM eye_mask_quality WHERE dataset_id = ?;",
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
            extracted_rows = _extract_eye_mask_quality_rows(
                root,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_quality"] += 1

        existing_by_key: Dict[tuple[str, str], Dict[str, object]] = {
            (str(existing["stage_group"]), str(existing["run_name"])): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_key: Dict[tuple[str, str], Dict[str, object]] = {
            (str(extracted["stage_group"]), str(extracted["run_name"])): extracted for extracted in extracted_rows
        }

        for key, extracted in extracted_by_key.items():
            existing = existing_by_key.get(key)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _eye_mask_quality_row_signature(existing)
            extracted_sig = _eye_mask_quality_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for key in existing_by_key:
                if key not in extracted_by_key:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_eye_mask_quality(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_eye_mask_quality(
                    dataset_id=dataset_id,
                    stage_group=str(extracted["stage_group"]),
                    run_name=str(extracted["run_name"]),
                    run_created_utc=extracted.get("run_created_utc"),
                    recording_id=extracted.get("recording_id"),
                    zarr_use=extracted.get("zarr_use"),
                    eye_mask_method=extracted.get("eye_mask_method"),
                    source_crop_run=extracted.get("source_crop_run"),
                    source_keypoint_group=extracted.get("source_keypoint_group"),
                    source_keypoints_run=extracted.get("source_keypoints_run"),
                    source_eye_masks_run=extracted.get("source_eye_masks_run"),
                    source_eye_masks_method=extracted.get("source_eye_masks_method"),
                    review_state=extracted.get("review_state"),
                    review_method=extracted.get("review_method"),
                    review_intended_use=extracted.get("review_intended_use"),
                    review_reviewer=extracted.get("review_reviewer"),
                    review_timestamp_utc=extracted.get("review_timestamp_utc"),
                    total_rois=extracted.get("total_rois"),
                    successful_eyes=extracted.get("successful_eyes"),
                    successful_roi_pairs=extracted.get("successful_roi_pairs"),
                    successful_roi_pair_rate=extracted.get("successful_roi_pair_rate"),
                    source_keypoint_stale_state=extracted.get("source_keypoint_stale_state"),
                    source_keypoint_stale_reason=extracted.get("source_keypoint_stale_reason"),
                    source_keypoint_stale_timestamp_utc=extracted.get("source_keypoint_stale_timestamp_utc"),
                    source_keypoint_stale_json=extracted.get("source_keypoint_stale_json"),
                    lifecycle_state=extracted.get("lifecycle_state"),
                    lifecycle_reason=extracted.get("lifecycle_reason"),
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


def _keypoint_performance_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("keypoint_created_utc"),
        row.get("recording_id"),
        row.get("zarr_use"),
        row.get("keypoint_method"),
        row.get("model_run_id"),
        row.get("model_set_id"),
        row.get("model_path"),
        row.get("model_name"),
        row.get("source_crop_run"),
        row.get("source_detect_run"),
        row.get("source_refined_run"),
        row.get("total_rois"),
        row.get("successful_detections"),
        row.get("failed_detections"),
        row.get("success_rate_percent"),
        row.get("frames_with_keypoints"),
        row.get("mean_confidence"),
        row.get("duration_seconds"),
        row.get("inference_duration_seconds"),
        row.get("keypoints_per_second"),
        row.get("inference_average_fps"),
        row.get("batch_size"),
        row.get("imgsz"),
        row.get("conf_threshold"),
        row.get("iou_threshold"),
        row.get("summary_statistics_json"),
        row.get("zarr_mtime_ns"),
    )


def _backfill_keypoint_performance(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
    include_all_datasets: bool = False,
) -> Dict[str, int]:
    if include_all_datasets:
        rows = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path, recording_id, zarr_use
            FROM datasets
            WHERE status IS NULL OR lower(status) != 'missing'
            ORDER BY dataset_id;
            """
        ).fetchall()
    else:
        rows = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path, recording_id, zarr_use
            FROM datasets
            WHERE (status IS NULL OR lower(status) != 'missing')
              AND lower(COALESCE(artifact_kind, '')) = 'source_recording'
              AND lower(COALESCE(zarr_use, '')) = 'analysis'
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
        "datasets_no_performance": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "rows_deleted": 0,
    }

    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM keypoint_performance WHERE dataset_id = ?;",
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
            extracted_rows = _extract_keypoint_performance_rows(
                root,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_performance"] += 1

        existing_by_run: Dict[str, Dict[str, object]] = {
            str(existing["keypoint_run"]): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_run: Dict[str, Dict[str, object]] = {
            str(extracted["keypoint_run"]): extracted for extracted in extracted_rows
        }

        for keypoint_run, extracted in extracted_by_run.items():
            existing = existing_by_run.get(keypoint_run)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _keypoint_performance_row_signature(existing)
            extracted_sig = _keypoint_performance_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for keypoint_run in existing_by_run:
                if keypoint_run not in extracted_by_run:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_keypoint_performance(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_keypoint_performance(
                    dataset_id=dataset_id,
                    keypoint_run=str(extracted["keypoint_run"]),
                    keypoint_created_utc=extracted.get("keypoint_created_utc"),
                    recording_id=extracted.get("recording_id"),
                    zarr_use=extracted.get("zarr_use"),
                    keypoint_method=extracted.get("keypoint_method"),
                    model_run_id=extracted.get("model_run_id"),
                    model_set_id=extracted.get("model_set_id"),
                    model_path=extracted.get("model_path"),
                    model_name=extracted.get("model_name"),
                    source_crop_run=extracted.get("source_crop_run"),
                    source_detect_run=extracted.get("source_detect_run"),
                    source_refined_run=extracted.get("source_refined_run"),
                    total_rois=extracted.get("total_rois"),
                    successful_detections=extracted.get("successful_detections"),
                    failed_detections=extracted.get("failed_detections"),
                    success_rate_percent=extracted.get("success_rate_percent"),
                    frames_with_keypoints=extracted.get("frames_with_keypoints"),
                    mean_confidence=extracted.get("mean_confidence"),
                    duration_seconds=extracted.get("duration_seconds"),
                    inference_duration_seconds=extracted.get("inference_duration_seconds"),
                    keypoints_per_second=extracted.get("keypoints_per_second"),
                    inference_average_fps=extracted.get("inference_average_fps"),
                    batch_size=extracted.get("batch_size"),
                    imgsz=extracted.get("imgsz"),
                    conf_threshold=extracted.get("conf_threshold"),
                    iou_threshold=extracted.get("iou_threshold"),
                    summary_statistics_json=extracted.get("summary_statistics_json"),
                    zarr_mtime_ns=extracted.get("zarr_mtime_ns"),
                    updated_utc=extracted.get("updated_utc"),
                )

    return summary


def _detect_performance_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("detect_created_utc"),
        row.get("recording_id"),
        row.get("zarr_use"),
        row.get("detection_method"),
        row.get("model_run_id"),
        row.get("model_set_id"),
        row.get("model_path"),
        row.get("model_name"),
        row.get("coverage_percent"),
        row.get("frames_with_detections"),
        row.get("frames_zero_detections"),
        row.get("total_frames"),
        row.get("mean_confidence"),
        row.get("min_confidence"),
        row.get("max_confidence"),
        row.get("inference_duration_seconds"),
        row.get("inference_average_fps"),
        row.get("inference_avg_batch_ms"),
        row.get("inference_avg_read_ms"),
        row.get("conf_threshold"),
        row.get("iou_threshold"),
        row.get("batch_size"),
        row.get("inference_width"),
        row.get("inference_height"),
        row.get("zarr_mtime_ns"),
    )


def _crop_quality_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("recording_id"),
        row.get("zarr_use"),
        row.get("crop_created_utc"),
        row.get("source_detect_run"),
        row.get("source_refined_run"),
        row.get("detection_source_type"),
        row.get("detection_source_path"),
        row.get("total_rois"),
        row.get("frames_with_crops"),
        row.get("total_frames"),
        row.get("percent_frames_with_crops"),
        row.get("includes_interpolated"),
        row.get("n_real_detections"),
        row.get("n_interpolated_detections"),
        row.get("review_state"),
        row.get("review_method"),
        row.get("review_intended_use"),
        row.get("review_reviewer"),
        row.get("review_timestamp_utc"),
        row.get("review_notes"),
        row.get("zarr_mtime_ns"),
    )


def _backfill_crop_quality(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
    include_all_datasets: bool = False,
) -> Dict[str, int]:
    if include_all_datasets:
        rows = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path, recording_id, zarr_use
            FROM datasets
            WHERE status IS NULL OR lower(status) != 'missing'
            ORDER BY dataset_id;
            """
        ).fetchall()
    else:
        rows = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path, recording_id, zarr_use
            FROM datasets
            WHERE (status IS NULL OR lower(status) != 'missing')
              AND lower(COALESCE(artifact_kind, '')) = 'source_recording'
              AND lower(COALESCE(zarr_use, '')) = 'analysis'
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
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM crop_quality WHERE dataset_id = ?;",
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
            extracted_rows = _extract_crop_quality_rows(
                root,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_quality"] += 1

        existing_by_run: Dict[str, Dict[str, object]] = {
            str(existing["crop_run"]): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_run: Dict[str, Dict[str, object]] = {
            str(extracted["crop_run"]): extracted for extracted in extracted_rows
        }

        for crop_run, extracted in extracted_by_run.items():
            existing = existing_by_run.get(crop_run)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _crop_quality_row_signature(existing)
            extracted_sig = _crop_quality_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for crop_run in existing_by_run:
                if crop_run not in extracted_by_run:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        registry.replace_crop_quality(dataset_id, extracted_rows)

    return summary


def _backfill_detect_performance(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
    include_all_datasets: bool = False,
) -> Dict[str, int]:
    if include_all_datasets:
        rows = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path, recording_id, zarr_use
            FROM datasets
            WHERE status IS NULL OR lower(status) != 'missing'
            ORDER BY dataset_id;
            """
        ).fetchall()
    else:
        rows = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path, recording_id, zarr_use
            FROM datasets
            WHERE (status IS NULL OR lower(status) != 'missing')
              AND lower(COALESCE(artifact_kind, '')) = 'source_recording'
              AND lower(COALESCE(zarr_use, '')) = 'analysis'
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
        "datasets_no_performance": 0,
        "rows_stale": 0,
        "rows_in_progress": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "rows_deleted": 0,
    }

    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM detect_performance WHERE dataset_id = ?;",
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
            extracted_rows = _extract_detect_performance_rows(
                root,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_performance"] += 1

        existing_by_run: Dict[str, Dict[str, object]] = {
            str(existing["detect_run"]): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_run: Dict[str, Dict[str, object]] = {
            str(extracted["detect_run"]): extracted for extracted in extracted_rows
        }

        for detect_run, extracted in extracted_by_run.items():
            existing = existing_by_run.get(detect_run)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _detect_performance_row_signature(existing)
            extracted_sig = _detect_performance_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for detect_run in existing_by_run:
                if detect_run not in extracted_by_run:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_detect_performance(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_detect_performance(
                    dataset_id=dataset_id,
                    detect_run=str(extracted["detect_run"]),
                    detect_created_utc=extracted.get("detect_created_utc"),
                    recording_id=extracted.get("recording_id"),
                    zarr_use=extracted.get("zarr_use"),
                    detection_method=extracted.get("detection_method"),
                    model_run_id=extracted.get("model_run_id"),
                    model_set_id=extracted.get("model_set_id"),
                    model_path=extracted.get("model_path"),
                    model_name=extracted.get("model_name"),
                    coverage_percent=extracted.get("coverage_percent"),
                    frames_with_detections=extracted.get("frames_with_detections"),
                    frames_zero_detections=extracted.get("frames_zero_detections"),
                    total_frames=extracted.get("total_frames"),
                    mean_confidence=extracted.get("mean_confidence"),
                    min_confidence=extracted.get("min_confidence"),
                    max_confidence=extracted.get("max_confidence"),
                    inference_duration_seconds=extracted.get("inference_duration_seconds"),
                    inference_average_fps=extracted.get("inference_average_fps"),
                    inference_avg_batch_ms=extracted.get("inference_avg_batch_ms"),
                    inference_avg_read_ms=extracted.get("inference_avg_read_ms"),
                    conf_threshold=extracted.get("conf_threshold"),
                    iou_threshold=extracted.get("iou_threshold"),
                    batch_size=extracted.get("batch_size"),
                    inference_width=extracted.get("inference_width"),
                    inference_height=extracted.get("inference_height"),
                    zarr_mtime_ns=extracted.get("zarr_mtime_ns"),
                    updated_utc=extracted.get("updated_utc"),
                )

    return summary


def _eye_mask_performance_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("run_created_utc"),
        row.get("recording_id"),
        row.get("zarr_use"),
        row.get("method"),
        row.get("source_crop_run"),
        row.get("source_keypoint_group"),
        row.get("source_keypoints_run"),
        row.get("source_eye_masks_run"),
        row.get("source_eye_masks_method"),
        row.get("total_rois"),
        row.get("successful_eyes"),
        row.get("successful_roi_pairs"),
        row.get("successful_roi_pair_rate"),
        row.get("duration_seconds"),
        row.get("rois_per_second"),
        row.get("inference_duration_seconds"),
        row.get("inference_average_fps"),
        row.get("reason_counts_json"),
        row.get("summary_statistics_json"),
        row.get("review_state"),
        row.get("review_method"),
        row.get("review_intended_use"),
        row.get("review_reviewer"),
        row.get("review_timestamp_utc"),
        row.get("source_keypoint_stale_state"),
        row.get("source_keypoint_stale_reason"),
        row.get("source_keypoint_stale_timestamp_utc"),
        row.get("source_keypoint_stale_json"),
        row.get("lifecycle_state"),
        row.get("lifecycle_reason"),
        row.get("zarr_mtime_ns"),
    )


def _subject_mask_performance_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("run_created_utc"),
        row.get("recording_id"),
        row.get("zarr_use"),
        row.get("subject_mask_method"),
        row.get("label_schema_id"),
        row.get("source_crop_run"),
        row.get("source_keypoint_group"),
        row.get("source_keypoints_run"),
        row.get("source_subject_mask_run"),
        row.get("source_subject_mask_method"),
        row.get("run_semantics"),
        row.get("probability_semantics"),
        row.get("source_background_run"),
        row.get("source_background_array"),
        row.get("source_dish_mask_array"),
        row.get("tuning_source"),
        row.get("tuning_timestamp"),
        row.get("total_rois"),
        row.get("rows_with_any_mask"),
        row.get("coverage_percent"),
        row.get("duration_seconds"),
        row.get("rois_per_second"),
        row.get("available_component_count"),
        row.get("available_components_json"),
        row.get("unavailable_components_json"),
        row.get("component_review_states_json"),
        row.get("eye_component_mode"),
        row.get("reason_counts_json"),
        row.get("summary_statistics_json"),
        row.get("review_state"),
        row.get("review_method"),
        row.get("review_intended_use"),
        row.get("review_reviewer"),
        row.get("review_timestamp_utc"),
        row.get("source_subject_mask_stale_state"),
        row.get("source_subject_mask_stale_reason"),
        row.get("source_subject_mask_stale_timestamp_utc"),
        row.get("source_subject_mask_stale_json"),
        row.get("lifecycle_state"),
        row.get("lifecycle_reason"),
        row.get("zarr_mtime_ns"),
    )


def _subject_mask_component_quality_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("component_family"),
        row.get("run_created_utc"),
        row.get("recording_id"),
        row.get("zarr_use"),
        row.get("subject_mask_method"),
        row.get("label_schema_id"),
        row.get("eye_component_mode"),
        row.get("source_subject_mask_run"),
        row.get("available"),
        row.get("review_state"),
        row.get("review_method"),
        row.get("review_intended_use"),
        row.get("review_reviewer"),
        row.get("review_timestamp_utc"),
        row.get("total_rois"),
        row.get("rows_with_component_mask"),
        row.get("rows_with_component_mask_rate"),
        row.get("lifecycle_state"),
        row.get("lifecycle_reason"),
        row.get("zarr_mtime_ns"),
    )


def _backfill_eye_mask_performance(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
    include_all_datasets: bool = False,
) -> Dict[str, int]:
    if include_all_datasets:
        rows = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path, recording_id, zarr_use
            FROM datasets
            WHERE status IS NULL OR lower(status) != 'missing'
            ORDER BY dataset_id;
            """
        ).fetchall()
    else:
        rows = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path, recording_id, zarr_use
            FROM datasets
            WHERE (status IS NULL OR lower(status) != 'missing')
              AND lower(COALESCE(artifact_kind, '')) = 'source_recording'
              AND lower(COALESCE(zarr_use, '')) = 'analysis'
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
        "datasets_no_performance": 0,
        "rows_stale": 0,
        "rows_in_progress": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "rows_deleted": 0,
    }

    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM eye_mask_performance WHERE dataset_id = ?;",
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
            extracted_rows = _extract_eye_mask_performance_rows(
                root,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_performance"] += 1
        for extracted in extracted_rows:
            lifecycle_state = str(extracted.get("lifecycle_state") or "").strip().lower()
            if lifecycle_state == "stale":
                summary["rows_stale"] += 1
            elif lifecycle_state == "in_progress":
                summary["rows_in_progress"] += 1

        existing_by_key: Dict[tuple[str, str], Dict[str, object]] = {
            (str(existing["stage_group"]), str(existing["run_name"])): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_key: Dict[tuple[str, str], Dict[str, object]] = {
            (str(extracted["stage_group"]), str(extracted["run_name"])): extracted for extracted in extracted_rows
        }

        for key, extracted in extracted_by_key.items():
            existing = existing_by_key.get(key)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _eye_mask_performance_row_signature(existing)
            extracted_sig = _eye_mask_performance_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for key in existing_by_key:
                if key not in extracted_by_key:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_eye_mask_performance(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_eye_mask_performance(
                    dataset_id=dataset_id,
                    stage_group=str(extracted["stage_group"]),
                    run_name=str(extracted["run_name"]),
                    run_created_utc=extracted.get("run_created_utc"),
                    recording_id=extracted.get("recording_id"),
                    zarr_use=extracted.get("zarr_use"),
                    method=extracted.get("method"),
                    source_crop_run=extracted.get("source_crop_run"),
                    source_keypoint_group=extracted.get("source_keypoint_group"),
                    source_keypoints_run=extracted.get("source_keypoints_run"),
                    source_eye_masks_run=extracted.get("source_eye_masks_run"),
                    source_eye_masks_method=extracted.get("source_eye_masks_method"),
                    total_rois=extracted.get("total_rois"),
                    successful_eyes=extracted.get("successful_eyes"),
                    successful_roi_pairs=extracted.get("successful_roi_pairs"),
                    successful_roi_pair_rate=extracted.get("successful_roi_pair_rate"),
                    duration_seconds=extracted.get("duration_seconds"),
                    rois_per_second=extracted.get("rois_per_second"),
                    inference_duration_seconds=extracted.get("inference_duration_seconds"),
                    inference_average_fps=extracted.get("inference_average_fps"),
                    reason_counts_json=extracted.get("reason_counts_json"),
                    summary_statistics_json=extracted.get("summary_statistics_json"),
                    review_state=extracted.get("review_state"),
                    review_method=extracted.get("review_method"),
                    review_intended_use=extracted.get("review_intended_use"),
                    review_reviewer=extracted.get("review_reviewer"),
                    review_timestamp_utc=extracted.get("review_timestamp_utc"),
                    source_keypoint_stale_state=extracted.get("source_keypoint_stale_state"),
                    source_keypoint_stale_reason=extracted.get("source_keypoint_stale_reason"),
                    source_keypoint_stale_timestamp_utc=extracted.get("source_keypoint_stale_timestamp_utc"),
                    source_keypoint_stale_json=extracted.get("source_keypoint_stale_json"),
                    lifecycle_state=extracted.get("lifecycle_state"),
                    lifecycle_reason=extracted.get("lifecycle_reason"),
                    zarr_mtime_ns=extracted.get("zarr_mtime_ns"),
                    updated_utc=extracted.get("updated_utc"),
                )

    return summary


def _backfill_subject_mask_performance(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
) -> Dict[str, int]:
    rows = registry.conn.execute(
        """
        SELECT dataset_id, zarr_path, recording_id, zarr_use
        FROM datasets
        WHERE (status IS NULL OR lower(status) != 'missing')
          AND lower(COALESCE(artifact_kind, '')) = 'source_recording'
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
        "datasets_no_performance": 0,
        "rows_stale": 0,
        "rows_in_progress": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "rows_deleted": 0,
    }

    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM subject_mask_performance WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchall()
        if not refresh and existing_rows:
            summary["datasets_skipped_existing"] += 1
            summary["rows_skipped"] += len(existing_rows)
            continue

        try:
            root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
            extracted_rows = _extract_subject_mask_performance_rows(
                root,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_performance"] += 1
        for extracted in extracted_rows:
            lifecycle_state = str(extracted.get("lifecycle_state") or "").strip().lower()
            if lifecycle_state == "stale":
                summary["rows_stale"] += 1
            elif lifecycle_state == "in_progress":
                summary["rows_in_progress"] += 1

        existing_by_key: Dict[tuple[str, str], Dict[str, object]] = {
            (str(existing["stage_group"]), str(existing["run_name"])): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_key: Dict[tuple[str, str], Dict[str, object]] = {
            (str(extracted["stage_group"]), str(extracted["run_name"])): extracted for extracted in extracted_rows
        }

        for key, extracted in extracted_by_key.items():
            existing = existing_by_key.get(key)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _subject_mask_performance_row_signature(existing)
            extracted_sig = _subject_mask_performance_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for key in existing_by_key:
                if key not in extracted_by_key:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_subject_mask_performance(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_subject_mask_performance(
                    dataset_id=dataset_id,
                    stage_group=str(extracted["stage_group"]),
                    run_name=str(extracted["run_name"]),
                    run_created_utc=extracted.get("run_created_utc"),
                    recording_id=extracted.get("recording_id"),
                    zarr_use=extracted.get("zarr_use"),
                    subject_mask_method=extracted.get("subject_mask_method"),
                    label_schema_id=extracted.get("label_schema_id"),
                    source_crop_run=extracted.get("source_crop_run"),
                    source_keypoint_group=extracted.get("source_keypoint_group"),
                    source_keypoints_run=extracted.get("source_keypoints_run"),
                    source_subject_mask_run=extracted.get("source_subject_mask_run"),
                    source_subject_mask_method=extracted.get("source_subject_mask_method"),
                    run_semantics=extracted.get("run_semantics"),
                    probability_semantics=extracted.get("probability_semantics"),
                    source_background_run=extracted.get("source_background_run"),
                    source_background_array=extracted.get("source_background_array"),
                    source_dish_mask_array=extracted.get("source_dish_mask_array"),
                    tuning_source=extracted.get("tuning_source"),
                    tuning_timestamp=extracted.get("tuning_timestamp"),
                    total_rois=extracted.get("total_rois"),
                    rows_with_any_mask=extracted.get("rows_with_any_mask"),
                    coverage_percent=extracted.get("coverage_percent"),
                    duration_seconds=extracted.get("duration_seconds"),
                    rois_per_second=extracted.get("rois_per_second"),
                    available_component_count=extracted.get("available_component_count"),
                    available_components_json=extracted.get("available_components_json"),
                    unavailable_components_json=extracted.get("unavailable_components_json"),
                    component_review_states_json=extracted.get("component_review_states_json"),
                    eye_component_mode=extracted.get("eye_component_mode"),
                    reason_counts_json=extracted.get("reason_counts_json"),
                    summary_statistics_json=extracted.get("summary_statistics_json"),
                    review_state=extracted.get("review_state"),
                    review_method=extracted.get("review_method"),
                    review_intended_use=extracted.get("review_intended_use"),
                    review_reviewer=extracted.get("review_reviewer"),
                    review_timestamp_utc=extracted.get("review_timestamp_utc"),
                    source_subject_mask_stale_state=extracted.get("source_subject_mask_stale_state"),
                    source_subject_mask_stale_reason=extracted.get("source_subject_mask_stale_reason"),
                    source_subject_mask_stale_timestamp_utc=extracted.get("source_subject_mask_stale_timestamp_utc"),
                    source_subject_mask_stale_json=extracted.get("source_subject_mask_stale_json"),
                    lifecycle_state=extracted.get("lifecycle_state"),
                    lifecycle_reason=extracted.get("lifecycle_reason"),
                    zarr_mtime_ns=extracted.get("zarr_mtime_ns"),
                    updated_utc=extracted.get("updated_utc"),
                )

    return summary


def _backfill_subject_mask_component_quality(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    refresh: bool,
) -> Dict[str, int]:
    rows = registry.conn.execute(
        """
        SELECT dataset_id, zarr_path, recording_id, zarr_use
        FROM datasets
        WHERE (status IS NULL OR lower(status) != 'missing')
          AND lower(COALESCE(artifact_kind, '')) = 'source_recording'
        ORDER BY dataset_id;
        """
    ).fetchall()
    scope_roots = _normalize_scope_paths(scope_paths)
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
        recording_id = str(row["recording_id"]) if row["recording_id"] else None
        zarr_use = str(row["zarr_use"]) if row["zarr_use"] else None
        if not _matches_scope(str(zarr_path), scope_roots):
            continue
        summary["datasets_scanned"] += 1
        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing"] += 1
            continue

        existing_rows = registry.conn.execute(
            "SELECT * FROM subject_mask_component_quality WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchall()
        if not refresh and existing_rows:
            summary["datasets_skipped_existing"] += 1
            summary["rows_skipped"] += len(existing_rows)
            continue

        try:
            root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
            extracted_rows = _extract_subject_mask_component_quality_rows(
                root,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
            )
        except Exception:
            summary["datasets_errors"] += 1
            continue

        if not extracted_rows:
            summary["datasets_no_quality"] += 1

        existing_by_key: Dict[tuple[str, str, str], Dict[str, object]] = {
            (
                str(existing["stage_group"]),
                str(existing["run_name"]),
                str(existing["component_name"]),
            ): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
        }
        extracted_by_key: Dict[tuple[str, str, str], Dict[str, object]] = {
            (
                str(extracted["stage_group"]),
                str(extracted["run_name"]),
                str(extracted["component_name"]),
            ): extracted
            for extracted in extracted_rows
        }

        for key, extracted in extracted_by_key.items():
            existing = existing_by_key.get(key)
            if existing is None:
                summary["rows_inserted"] += 1
                continue
            existing_sig = _subject_mask_component_quality_row_signature(existing)
            extracted_sig = _subject_mask_component_quality_row_signature(extracted)
            if existing_sig == extracted_sig:
                summary["rows_skipped"] += 1
            else:
                summary["rows_updated"] += 1

        if refresh:
            for key in existing_by_key:
                if key not in extracted_by_key:
                    summary["rows_deleted"] += 1

        if dry_run:
            continue
        if refresh:
            registry.replace_subject_mask_component_quality(dataset_id, extracted_rows)
        else:
            for extracted in extracted_rows:
                registry.upsert_subject_mask_component_quality(
                    dataset_id=dataset_id,
                    stage_group=str(extracted["stage_group"]),
                    run_name=str(extracted["run_name"]),
                    component_name=str(extracted["component_name"]),
                    component_family=extracted.get("component_family"),
                    run_created_utc=extracted.get("run_created_utc"),
                    recording_id=extracted.get("recording_id"),
                    zarr_use=extracted.get("zarr_use"),
                    subject_mask_method=extracted.get("subject_mask_method"),
                    label_schema_id=extracted.get("label_schema_id"),
                    eye_component_mode=extracted.get("eye_component_mode"),
                    source_subject_mask_run=extracted.get("source_subject_mask_run"),
                    available=extracted.get("available"),
                    review_state=extracted.get("review_state"),
                    review_method=extracted.get("review_method"),
                    review_intended_use=extracted.get("review_intended_use"),
                    review_reviewer=extracted.get("review_reviewer"),
                    review_timestamp_utc=extracted.get("review_timestamp_utc"),
                    total_rois=extracted.get("total_rois"),
                    rows_with_component_mask=extracted.get("rows_with_component_mask"),
                    rows_with_component_mask_rate=extracted.get("rows_with_component_mask_rate"),
                    lifecycle_state=extracted.get("lifecycle_state"),
                    lifecycle_reason=extracted.get("lifecycle_reason"),
                    quality_updated_utc=extracted.get("quality_updated_utc"),
                    zarr_mtime_ns=extracted.get("zarr_mtime_ns"),
                )

    return summary


def _decode_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _coerce_float_value(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _coerce_int_value(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed


def _coerce_mapping_value(value: object) -> Optional[Dict[str, object]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _canonical_json_text(value: object) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _resolve_latest_group(parent: object) -> tuple[Optional[str], Optional[object], str]:
    if parent is None:
        return None, None, "none"

    latest = None
    if hasattr(parent, "attrs"):
        try:
            latest = _decode_text(parent.attrs.get("latest"))  # type: ignore[attr-defined]
        except Exception:
            latest = None
    if latest:
        try:
            if latest in parent:  # type: ignore[operator]
                return latest, parent[latest], "latest_attr"  # type: ignore[index]
        except Exception:
            pass

    names = _group_names(parent)
    if not names:
        return None, None, "none"

    fallback = names[-1]
    try:
        return fallback, parent[fallback], "sorted_fallback"  # type: ignore[index]
    except Exception:
        return fallback, None, "sorted_fallback_error"


def _group_names(parent: object) -> List[str]:
    if parent is None:
        return []
    names: List[str] = []
    try:
        if hasattr(parent, "group_keys"):
            names = [str(name) for name in parent.group_keys()]  # type: ignore[attr-defined]
        else:
            names = [str(name) for name in parent.keys()]  # type: ignore[attr-defined]
    except Exception:
        names = []
    return sorted(name for name in names if name)


def _extract_source_keypoints_run(group: object) -> Optional[str]:
    if group is None or not hasattr(group, "attrs"):
        return None
    source_run = _decode_text(group.attrs.get("source_keypoints_run"))  # type: ignore[attr-defined]
    if source_run:
        return source_run
    return _decode_text(group.attrs.get("source_keypoint_run"))  # type: ignore[attr-defined]


def _extract_source_detect_run(group: object) -> Optional[str]:
    if group is None or not hasattr(group, "attrs"):
        return None
    return _decode_text(group.attrs.get("source_detect_run"))  # type: ignore[attr-defined]


def _extract_source_eye_masks_run(group: object) -> Optional[str]:
    if group is None or not hasattr(group, "attrs"):
        return None
    return _decode_text(group.attrs.get("source_eye_masks_run"))  # type: ignore[attr-defined]


def _extract_source_subject_mask_run(group: object) -> Optional[str]:
    if group is None or not hasattr(group, "attrs"):
        return None
    return _decode_text(group.attrs.get("source_subject_mask_run"))  # type: ignore[attr-defined]


def _resolve_group_for_source_run(
    parent: object,
    *,
    source_run: Optional[str],
    source_run_extractor: Callable[[object], Optional[str]],
) -> tuple[Optional[str], Optional[object], str, Optional[str], Optional[str]]:
    latest_run, latest_group, latest_selection = _resolve_latest_group(parent)
    latest_source_run = source_run_extractor(latest_group)
    if source_run is None:
        return latest_run, latest_group, latest_selection, None, None
    if latest_group is not None and latest_source_run == source_run:
        return latest_run, latest_group, f"source_match_{latest_selection}", None, None

    matches: List[str] = []
    for name in _group_names(parent):
        try:
            group = parent[name]  # type: ignore[index]
        except Exception:
            continue
        if source_run_extractor(group) == source_run:
            matches.append(name)
    if matches:
        matched_run = sorted(matches)[-1]
        try:
            return matched_run, parent[matched_run], "source_match_sorted_fallback", None, None  # type: ignore[index]
        except Exception:
            return matched_run, None, "source_match_sorted_fallback_error", None, None

    if latest_run is not None:
        if latest_source_run is None:
            return None, None, "source_mismatch_missing_attr", latest_run, None
        return None, None, "source_mismatch", latest_run, latest_source_run
    return None, None, "none", None, None


def _resolve_refined_keypoints_group(
    parent: object,
    *,
    source_keypoints_run: Optional[str],
) -> tuple[Optional[str], Optional[object], str, Optional[str], Optional[str]]:
    return _resolve_group_for_source_run(
        parent,
        source_run=source_keypoints_run,
        source_run_extractor=_extract_source_keypoints_run,
    )


def _extract_coverage_pct(group: object) -> Optional[float]:
    if group is None or not hasattr(group, "get"):
        return None
    for key in ("frame_counts", "n_detections"):
        try:
            counts_arr = group.get(key)  # type: ignore[attr-defined]
        except Exception:
            counts_arr = None
        if counts_arr is None:
            continue
        try:
            values = counts_arr[:]
        except Exception:
            continue
        try:
            total = int(values.shape[0])  # type: ignore[attr-defined]
        except Exception:
            try:
                total = len(values)  # type: ignore[arg-type]
            except Exception:
                total = 0
        if total <= 0:
            continue
        try:
            present = int((values > 0).sum())  # type: ignore[operator]
        except Exception:
            try:
                present = 0
                for item in values:  # type: ignore[assignment]
                    if _coerce_float_value(item) and float(item) > 0.0:
                        present += 1
            except Exception:
                continue
        return float(present) / float(total) * 100.0
    return None


def _extract_text_list(value: object) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (str, bytes)):
        decoded = _decode_text(value)
        return [decoded] if decoded else []
    if isinstance(value, (list, tuple)):
        items: List[str] = []
        for item in value:
            decoded = _decode_text(item)
            if decoded:
                items.append(decoded)
        return items
    return []


def _extract_mask_present_rate(group: object) -> Optional[float]:
    if group is None or not hasattr(group, "get"):
        return None
    try:
        metrics = group.get("metrics")  # type: ignore[attr-defined]
    except Exception:
        metrics = None
    if metrics is None or not hasattr(metrics, "get"):
        return None
    try:
        mask_present = metrics.get("mask_present")  # type: ignore[attr-defined]
    except Exception:
        mask_present = None
    if mask_present is None:
        return None
    try:
        values = mask_present[:]
    except Exception:
        return None
    try:
        total = int(values.shape[0])  # type: ignore[attr-defined]
    except Exception:
        try:
            total = len(values)  # type: ignore[arg-type]
        except Exception:
            total = 0
    if total <= 0:
        return None
    try:
        present = int(values.any(axis=1).sum())  # type: ignore[attr-defined]
        return float(present) / float(total) * 100.0
    except Exception:
        pass

    present = 0
    try:
        for row in values:  # type: ignore[assignment]
            row_present = False
            for item in row:
                if bool(item):
                    row_present = True
                    break
            if row_present:
                present += 1
    except Exception:
        return None
    return float(present) / float(total) * 100.0


def _extract_subject_mask_coverage_pct(group: object) -> Optional[float]:
    coverage = _extract_coverage_pct(group)
    if coverage is not None:
        return coverage
    return _extract_mask_present_rate(group)


def _extract_subject_mask_component_details(group: object) -> Dict[str, object]:
    if group is None or not hasattr(group, "attrs"):
        return {}

    attrs = group.attrs  # type: ignore[attr-defined]
    mask_labels = _extract_text_list(attrs.get("mask_labels"))
    if not mask_labels:
        return {}

    available_components = list(mask_labels)
    unavailable_components: List[str] = []
    try:
        available_arr = group.get("available_channels")  # type: ignore[attr-defined]
    except Exception:
        available_arr = None
    if available_arr is not None:
        try:
            available_values = available_arr[:]
            if hasattr(available_values, "tolist"):
                available_values = available_values.tolist()
            flags = [bool(value) for value in available_values]
        except Exception:
            flags = []
        if flags:
            padded_flags = list(flags[: len(mask_labels)])
            if len(padded_flags) < len(mask_labels):
                padded_flags.extend([False] * (len(mask_labels) - len(padded_flags)))
            available_components = [label for label, flag in zip(mask_labels, padded_flags) if flag]
            unavailable_components = [label for label, flag in zip(mask_labels, padded_flags) if not flag]

    component_review_states: Dict[str, str] = {}
    component_review_statuses = _coerce_mapping_value(attrs.get("component_review_statuses")) or {}
    for label in mask_labels:
        payload = _coerce_mapping_value(component_review_statuses.get(label))
        if payload is None:
            continue
        state = _decode_text(payload.get("state")) or _decode_text(payload.get("review_state"))
        if state:
            component_review_states[label] = state

    eye_component_mode: Optional[str] = None
    if "eye_left" in mask_labels or "eye_right" in mask_labels:
        eye_component_mode = "lr"
    elif "eyes_union" in mask_labels:
        eye_component_mode = "union"

    details: Dict[str, object] = {
        "label_schema_id": _decode_text(attrs.get("label_schema_id")),
        "mask_labels": mask_labels,
        "available_components": available_components,
        "unavailable_components": unavailable_components,
    }
    if eye_component_mode:
        details["eye_component_mode"] = eye_component_mode
    if component_review_states:
        details["component_review_states"] = component_review_states
    return details


def _extract_detect_method(detect_group: object) -> Optional[str]:
    if detect_group is None or not hasattr(detect_group, "attrs"):
        return None
    method = _decode_text(detect_group.attrs.get("detection_method"))  # type: ignore[attr-defined]
    if method:
        return method
    method = _decode_text(detect_group.attrs.get("method"))  # type: ignore[attr-defined]
    if method:
        return method
    provenance = _coerce_mapping_value(detect_group.attrs.get("provenance"))  # type: ignore[attr-defined]
    if provenance is None:
        return None
    return _decode_text(provenance.get("method"))


def _extract_detect_quality_details(detect_group: object) -> Dict[str, object]:
    if detect_group is None or not hasattr(detect_group, "get"):
        return {}
    try:
        quality_parent = detect_group.get("quality_reports")  # type: ignore[attr-defined]
    except Exception:
        quality_parent = None
    if quality_parent is None:
        return {}

    quality_run, quality_group, _ = _resolve_latest_group(quality_parent)
    if quality_group is None or not hasattr(quality_group, "attrs"):
        return {"detect_quality_run": quality_run} if quality_run else {}

    quality_score = _coerce_mapping_value(quality_group.attrs.get("quality_score")) or {}  # type: ignore[attr-defined]
    quality_summary = _coerce_mapping_value(
        quality_group.attrs.get("detection_quality_summary")  # type: ignore[attr-defined]
    ) or {}
    blip = _coerce_int_value(quality_summary.get("blip_detections"))
    jump = _coerce_int_value(quality_summary.get("jump_detections"))
    multi = _coerce_int_value(quality_summary.get("multi_detections"))
    detect_quality_artifacts: Optional[int] = None
    if blip is not None or jump is not None or multi is not None:
        detect_quality_artifacts = int((blip or 0) + (jump or 0) + (multi or 0))

    return {
        "detect_quality_run": quality_run,
        "detect_quality_grade": _decode_text(quality_score.get("grade")),
        "detect_quality_score": _coerce_float_value(quality_score.get("overall_score")),
        "detect_quality_clean_percent": _coerce_float_value(quality_summary.get("clean_percentage")),
        "detect_quality_artifacts": detect_quality_artifacts,
    }


def _extract_refined_detect_coverage_pct(refined_group: object) -> Optional[float]:
    if refined_group is None:
        return None
    direct = _coerce_float_value(
        refined_group.attrs.get("coverage_percent")  # type: ignore[attr-defined]
    )
    if direct is not None:
        return direct

    try:
        instances_group = refined_group.get("instances")  # type: ignore[attr-defined]
    except Exception:
        instances_group = None
    instances_cov = _extract_coverage_pct(instances_group)
    if instances_cov is not None:
        return instances_cov

    manual_latest = _decode_text(
        refined_group.attrs.get("manual_review_latest")  # type: ignore[attr-defined]
    )
    if manual_latest:
        try:
            manual_group = refined_group[manual_latest]  # type: ignore[index]
            manual_cov = _extract_coverage_pct(manual_group)
            if manual_cov is not None:
                return manual_cov
        except Exception:
            pass

    comparison = _coerce_mapping_value(
        refined_group.attrs.get("coverage_comparison")  # type: ignore[attr-defined]
    )
    if comparison is not None:
        for section in ("interpolated", "original", "filtered"):
            payload = comparison.get(section)
            if isinstance(payload, dict):
                cov = _coerce_float_value(payload.get("coverage_percent"))
                if cov is not None:
                    return cov

    stats = _coerce_mapping_value(
        refined_group.attrs.get("coverage_stats")  # type: ignore[attr-defined]
    )
    if stats is not None:
        for section in ("final", "clean"):
            payload = stats.get(section)
            if isinstance(payload, dict):
                cov = _coerce_float_value(payload.get("coverage_percent"))
                if cov is not None:
                    return cov

    for child_name in ("interpolated", "filtered"):
        try:
            child = refined_group.get(child_name)  # type: ignore[attr-defined]
        except Exception:
            child = None
        if child is None:
            continue
        cov_attr = _coerce_float_value(child.attrs.get("coverage_percent"))  # type: ignore[attr-defined]
        if cov_attr is not None:
            return cov_attr
        cov_counts = _extract_coverage_pct(child)
        if cov_counts is not None:
            return cov_counts

    return _extract_coverage_pct(refined_group)


def _extract_refined_keypoints_coverage_pct(refined_group: object) -> Optional[float]:
    if refined_group is None or not hasattr(refined_group, "attrs"):
        return None
    summary = _coerce_mapping_value(refined_group.attrs.get("summary_statistics"))  # type: ignore[attr-defined]
    blocks: List[Dict[str, object]] = []
    if summary is not None:
        post = summary.get("postprocess")
        if isinstance(post, dict):
            blocks.append(post)
        refine = summary.get("refine")
        if isinstance(refine, dict):
            blocks.append(refine)
        blocks.append(summary)
    for block in blocks:
        for key in ("success_rate_percent", "pass_rate_percent", "usable_keypoints_rate"):
            value = _coerce_float_value(block.get(key))
            if value is not None:
                return value
        usable = _coerce_float_value(block.get("usable_keypoints")) or _coerce_float_value(block.get("usable"))
        total = _coerce_float_value(block.get("total_rois")) or _coerce_float_value(block.get("total"))
        if usable is not None and total is not None and total > 0:
            return float(usable) / float(total) * 100.0

    try:
        usable_arr = refined_group.get("usable_keypoints")  # type: ignore[attr-defined]
    except Exception:
        usable_arr = None
    if usable_arr is None:
        return None
    try:
        values = usable_arr[:]
        total = int(values.shape[0])  # type: ignore[attr-defined]
        if total <= 0:
            return None
        usable = int(values.sum())  # type: ignore[attr-defined]
        return float(usable) / float(total) * 100.0
    except Exception:
        return None


def _extract_tracking_summary_details(tracks_group: object) -> Dict[str, object]:
    if tracks_group is None or not hasattr(tracks_group, "attrs"):
        return {}

    attrs = tracks_group.attrs  # type: ignore[attr-defined]
    summary = _coerce_mapping_value(attrs.get("summary_statistics")) or {}

    n_tracks = _coerce_int_value(summary.get("n_tracks"))
    if n_tracks is None:
        n_tracks = _coerce_int_value(attrs.get("num_tracks"))

    n_assigned_rows = _coerce_int_value(summary.get("n_assigned_rows"))
    n_unassigned_rows = _coerce_int_value(summary.get("n_unassigned_rows"))
    if n_unassigned_rows is None:
        n_unassigned_rows = _coerce_int_value(attrs.get("n_unassigned_rows"))

    n_rows = _coerce_int_value(summary.get("n_rows"))
    if n_assigned_rows is None and n_rows is not None and n_unassigned_rows is not None:
        n_assigned_rows = max(0, int(n_rows) - int(n_unassigned_rows))

    unassigned_row_rate_percent = _coerce_float_value(summary.get("unassigned_row_rate_percent"))
    if unassigned_row_rate_percent is None:
        unassigned_row_rate_percent = _coerce_float_value(attrs.get("unassigned_row_rate_percent"))
    if (
        unassigned_row_rate_percent is None
        and n_unassigned_rows is not None
        and n_rows is not None
        and n_rows > 0
    ):
        unassigned_row_rate_percent = float(n_unassigned_rows) / float(n_rows) * 100.0

    tracking_qc_state = _decode_text(summary.get("tracking_qc_state"))
    if tracking_qc_state is None:
        tracking_qc_state = _decode_text(attrs.get("tracking_qc_state"))
    if tracking_qc_state == "block":
        tracking_qc_state = "warn"
    tracking_warn_threshold_rows = _coerce_int_value(summary.get("tracking_warn_threshold_rows"))
    if tracking_warn_threshold_rows is None:
        tracking_warn_threshold_rows = _coerce_int_value(attrs.get("tracking_warn_threshold_rows"))
    tracking_warn_threshold_percent = _coerce_float_value(summary.get("tracking_warn_threshold_percent"))
    if tracking_warn_threshold_percent is None:
        tracking_warn_threshold_percent = _coerce_float_value(attrs.get("tracking_warn_threshold_percent"))
    tracking_block_threshold_rows = _coerce_int_value(summary.get("tracking_block_threshold_rows"))
    if tracking_block_threshold_rows is None:
        tracking_block_threshold_rows = _coerce_int_value(attrs.get("tracking_block_threshold_rows"))
    tracking_block_threshold_percent = _coerce_float_value(summary.get("tracking_block_threshold_percent"))
    if tracking_block_threshold_percent is None:
        tracking_block_threshold_percent = _coerce_float_value(attrs.get("tracking_block_threshold_percent"))

    if n_unassigned_rows is not None:
        computed_qc = build_tracking_qc_fields(
            n_unassigned_rows=int(n_unassigned_rows),
            unassigned_row_rate_percent=unassigned_row_rate_percent,
            warn_threshold_rows=tracking_warn_threshold_rows if tracking_warn_threshold_rows is not None else 1,
            warn_threshold_percent=tracking_warn_threshold_percent if tracking_warn_threshold_percent is not None else 0.0,
            block_threshold_rows=tracking_block_threshold_rows if tracking_block_threshold_rows is not None else 10,
            block_threshold_percent=tracking_block_threshold_percent if tracking_block_threshold_percent is not None else 1.0,
        )
        if tracking_qc_state is None:
            tracking_qc_state = str(computed_qc["tracking_qc_state"])
        if tracking_warn_threshold_rows is None:
            tracking_warn_threshold_rows = int(computed_qc["tracking_warn_threshold_rows"])
        if tracking_warn_threshold_percent is None:
            tracking_warn_threshold_percent = float(computed_qc["tracking_warn_threshold_percent"])
        if tracking_block_threshold_rows is None:
            tracking_block_threshold_rows = int(computed_qc["tracking_block_threshold_rows"])
        if tracking_block_threshold_percent is None:
            tracking_block_threshold_percent = float(computed_qc["tracking_block_threshold_percent"])

    details: Dict[str, object] = {}
    if n_tracks is not None:
        details["num_tracks"] = int(n_tracks)
    if n_assigned_rows is not None:
        details["n_assigned_rows"] = int(n_assigned_rows)
    if n_unassigned_rows is not None:
        details["n_unassigned_rows"] = int(n_unassigned_rows)
    if unassigned_row_rate_percent is not None:
        details["unassigned_row_rate_percent"] = float(unassigned_row_rate_percent)
    if tracking_qc_state is not None:
        details["tracking_qc_state"] = tracking_qc_state
    if tracking_warn_threshold_rows is not None:
        details["tracking_warn_threshold_rows"] = int(tracking_warn_threshold_rows)
    if tracking_warn_threshold_percent is not None:
        details["tracking_warn_threshold_percent"] = float(tracking_warn_threshold_percent)
    if tracking_block_threshold_rows is not None:
        details["tracking_block_threshold_rows"] = int(tracking_block_threshold_rows)
    if tracking_block_threshold_percent is not None:
        details["tracking_block_threshold_percent"] = float(tracking_block_threshold_percent)
    return details


def _extract_updated_utc(group: object, *, fallback: str) -> str:
    if group is None or not hasattr(group, "attrs"):
        return fallback
    for key in (
        "updated_utc",
        "created_utc",
        "created_at_utc",
        "timestamp_utc",
        "detect_timestamp_utc",
        "timestamp",
        "created_at",
    ):
        try:
            value = _decode_text(group.attrs.get(key))  # type: ignore[attr-defined]
        except Exception:
            value = None
        if value:
            return value
    return fallback


def _mtime_ns_to_utc_text(mtime_ns: Optional[int]) -> str:
    if mtime_ns is None:
        return "1970-01-01T00:00:00+00:00"
    return datetime.fromtimestamp(float(mtime_ns) / 1_000_000_000.0, tz=timezone.utc).isoformat()


def _step_status_from_presence(
    *,
    present: bool,
    is_production: bool,
    prerequisite_statuses: Sequence[str],
) -> tuple[str, str]:
    if present:
        return "ok", "present"
    if is_production:
        return "na", "production_dataset"
    if not prerequisite_statuses:
        return "missing", "run_missing"
    if any(status == "ok" for status in prerequisite_statuses):
        return "missing", "run_missing"
    if any(status == "error" for status in prerequisite_statuses):
        return "error", "upstream_error"
    if all(status == "na" for status in prerequisite_statuses):
        return "na", "upstream_na"
    return "absent", "upstream_missing"


def _make_recording_step_row(
    *,
    dataset_id: str,
    recording_id: str,
    step_name: str,
    status: str,
    run_name: Optional[str],
    method: Optional[str],
    coverage_pct: Optional[float],
    review_status: Optional[Dict[str, object]],
    details: Dict[str, object],
    source: str,
    zarr_mtime_ns: Optional[int],
    updated_utc: str,
) -> Dict[str, object]:
    safe_details = {key: value for key, value in details.items() if value is not None}
    return {
        "dataset_id": dataset_id,
        "recording_id": recording_id,
        "step_name": step_name,
        "status": status,
        "run_name": run_name,
        "method": method,
        "coverage_pct": _coerce_float_value(coverage_pct),
        "review_status_json": _canonical_json_text(review_status) if review_status else None,
        "details_json": _canonical_json_text(safe_details) if safe_details else None,
        "source": source,
        "zarr_mtime_ns": zarr_mtime_ns,
        "updated_utc": updated_utc,
    }


def _build_recording_step_error_rows(
    *,
    dataset_id: str,
    recording_id: str,
    zarr_mtime_ns: Optional[int],
    error_detail: str,
    source: str,
) -> List[Dict[str, object]]:
    updated_utc = _mtime_ns_to_utc_text(zarr_mtime_ns)
    rows: List[Dict[str, object]] = []
    for step_name in RECORDING_STEP_NAMES:
        rows.append(
            _make_recording_step_row(
                dataset_id=dataset_id,
                recording_id=recording_id,
                step_name=step_name,
                status="error",
                run_name=None,
                method=None,
                coverage_pct=None,
                review_status=None,
                details={"reason": "zarr_open_error", "error": error_detail},
                source=source,
                zarr_mtime_ns=zarr_mtime_ns,
                updated_utc=updated_utc,
            )
        )
    return rows


def _build_recording_step_rows_from_root(
    *,
    root: object,
    dataset_id: str,
    recording_id: str,
    zarr_use: Optional[str],
    zarr_mtime_ns: Optional[int],
    source: str,
) -> List[Dict[str, object]]:
    fallback_updated_utc = _mtime_ns_to_utc_text(zarr_mtime_ns)

    pipeline_type = _decode_text(root.attrs.get("pipeline_type"))  # type: ignore[attr-defined]
    zarr_purpose = _decode_text(root.attrs.get("zarr_purpose"))  # type: ignore[attr-defined]
    has_raw_video_attr = root.attrs.get("has_raw_video")  # type: ignore[attr-defined]
    if isinstance(has_raw_video_attr, (bytes, bytearray)):
        has_raw_video_attr = has_raw_video_attr.decode("utf-8", "ignore")
    if isinstance(has_raw_video_attr, str):
        lowered = has_raw_video_attr.strip().lower()
        if lowered in {"true", "1", "yes"}:
            has_raw_video_attr = True
        elif lowered in {"false", "0", "no"}:
            has_raw_video_attr = False
        else:
            has_raw_video_attr = None

    raw_group = root.get("raw_video")  # type: ignore[attr-defined]
    raw_present = bool(raw_group is not None)
    full_present = bool(raw_group is not None and "images_full" in raw_group)
    ds_present = bool(raw_group is not None and "images_ds" in raw_group)
    sampled_present = bool(raw_group is not None and "original_frame_indices" in raw_group)
    is_production = bool(
        zarr_purpose == "production"
        or pipeline_type == "yolo_inference"
        or (has_raw_video_attr is False and not (full_present or ds_present))
    )
    raw_status, raw_reason = _step_status_from_presence(
        present=full_present or ds_present,
        is_production=is_production,
        prerequisite_statuses=(),
    )

    background_full_present = False
    background_ds_present = False
    background_run: Optional[str] = None
    background_group: Optional[object] = None
    background_selection = "none"
    background_parent = root.get("background_runs")  # type: ignore[attr-defined]
    if background_parent is not None:
        background_run, background_group, background_selection = _resolve_latest_group(background_parent)
        if background_group is not None:
            background_full_present = bool("background_full" in background_group)
            background_ds_present = bool("background_ds" in background_group)
    if not (background_full_present and background_ds_present):
        legacy_background = root.get("background")  # type: ignore[attr-defined]
        if legacy_background is not None:
            background_group = legacy_background
            background_run = background_run or "legacy_background"
            if background_selection == "none":
                background_selection = "legacy_group"
            background_full_present = bool("background_full" in legacy_background)
            background_ds_present = bool("background_ds" in legacy_background)
    background_status, background_reason = _step_status_from_presence(
        present=background_full_present and background_ds_present,
        is_production=is_production,
        prerequisite_statuses=(raw_status,),
    )
    background_method = (
        _decode_text(background_group.attrs.get("method"))  # type: ignore[union-attr]
        if background_group is not None and hasattr(background_group, "attrs")
        else None
    )

    detect_parent = root.get("detect_runs")  # type: ignore[attr-defined]
    detect_run, detect_group, detect_selection = _resolve_latest_group(detect_parent)
    detect_status, detect_reason = _step_status_from_presence(
        present=detect_group is not None,
        is_production=is_production,
        prerequisite_statuses=(),
    )
    detect_method = _extract_detect_method(detect_group)
    detect_coverage = _extract_coverage_pct(detect_group)
    detect_quality_details = _extract_detect_quality_details(detect_group)

    refined_detect_parent = root.get("refined_detect_runs") or root.get("refined_runs")  # type: ignore[attr-defined]
    (
        refined_detect_run,
        refined_detect_group,
        refined_detect_selection,
        refined_detect_latest_run,
        refined_detect_latest_source_run,
    ) = _resolve_group_for_source_run(
        refined_detect_parent,
        source_run=detect_run,
        source_run_extractor=_extract_source_detect_run,
    )
    refined_detect_status, refined_detect_reason = _step_status_from_presence(
        present=refined_detect_group is not None,
        is_production=is_production,
        prerequisite_statuses=(detect_status,),
    )
    if refined_detect_group is None and detect_run and refined_detect_parent is not None:
        if refined_detect_selection == "source_mismatch":
            refined_detect_reason = "stale_vs_latest_detect"
        elif refined_detect_selection == "source_mismatch_missing_attr":
            refined_detect_reason = "missing_source_detect_run"
    refined_detect_method = _decode_text(
        refined_detect_group.attrs.get("method")  # type: ignore[union-attr]
    ) if refined_detect_group is not None else None
    if not refined_detect_method and refined_detect_group is not None:
        parameters = _coerce_mapping_value(refined_detect_group.attrs.get("parameters"))  # type: ignore[attr-defined]
        if parameters is not None:
            refined_detect_method = _decode_text(parameters.get("refine_mode"))
    refined_detect_coverage = _extract_refined_detect_coverage_pct(refined_detect_group)
    detect_review_status = (
        _coerce_mapping_value(refined_detect_group.attrs.get("detect_review_status"))  # type: ignore[attr-defined]
        if refined_detect_group is not None
        else None
    )

    crop_parent = root.get("crop_runs")  # type: ignore[attr-defined]
    crop_run, crop_group, crop_selection = _resolve_latest_group(crop_parent)
    crop_status, crop_reason = _step_status_from_presence(
        present=crop_group is not None,
        is_production=is_production,
        prerequisite_statuses=(refined_detect_status, detect_status),
    )
    crop_method = _decode_text(
        crop_group.attrs.get("detection_source_type")  # type: ignore[union-attr]
    ) if crop_group is not None else None
    if not crop_method and crop_group is not None:
        crop_method = _decode_text(crop_group.attrs.get("method"))  # type: ignore[attr-defined]
    crop_review_status = (
        _coerce_mapping_value(crop_group.attrs.get("crop_review_status"))  # type: ignore[attr-defined]
        if crop_group is not None
        else None
    )
    crop_run_state = (
        _decode_text(crop_group.attrs.get("status"))  # type: ignore[attr-defined]
        if crop_group is not None
        else None
    )
    if crop_group is not None and crop_run_state:
        run_state = crop_run_state.lower()
        if run_state in {"failed", "error"}:
            crop_status = "error"
            crop_reason = "run_failed"
        elif run_state in {"running", "in_progress", "started", "pending"}:
            crop_status = "missing"
            crop_reason = "run_in_progress"

    keypoints_parent = root.get("keypoints_runs")  # type: ignore[attr-defined]
    keypoints_run, keypoints_group, keypoints_selection = _resolve_latest_group(keypoints_parent)
    keypoints_status, keypoints_reason = _step_status_from_presence(
        present=keypoints_group is not None,
        is_production=is_production,
        prerequisite_statuses=(crop_status,),
    )
    keypoints_method = _decode_text(
        keypoints_group.attrs.get("method")  # type: ignore[union-attr]
    ) if keypoints_group is not None else None

    refined_keypoints_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")  # type: ignore[attr-defined]
    (
        refined_keypoints_run,
        refined_keypoints_group,
        refined_keypoints_selection,
        refined_keypoints_latest_run,
        refined_keypoints_latest_source_run,
    ) = _resolve_refined_keypoints_group(
        refined_keypoints_parent,
        source_keypoints_run=keypoints_run,
    )
    refined_keypoints_status, refined_keypoints_reason = _step_status_from_presence(
        present=refined_keypoints_group is not None,
        is_production=is_production,
        prerequisite_statuses=(keypoints_status,),
    )
    if refined_keypoints_group is None and keypoints_run and refined_keypoints_parent is not None:
        if refined_keypoints_selection == "source_mismatch":
            refined_keypoints_reason = "stale_vs_latest_keypoints"
        elif refined_keypoints_selection == "source_mismatch_missing_attr":
            refined_keypoints_reason = "missing_source_keypoints_run"
    refined_keypoints_method = _decode_text(
        refined_keypoints_group.attrs.get("method")  # type: ignore[union-attr]
    ) if refined_keypoints_group is not None else None
    if not refined_keypoints_method and refined_keypoints_group is not None:
        refined_keypoints_method = "refine_keypoints"
    refined_keypoints_coverage = _extract_refined_keypoints_coverage_pct(refined_keypoints_group)
    keypoint_review_status = (
        _coerce_mapping_value(refined_keypoints_group.attrs.get("keypoint_review_status"))  # type: ignore[attr-defined]
        if refined_keypoints_group is not None
        else None
    )

    eye_masks_parent = root.get("eye_masks_runs")  # type: ignore[attr-defined]
    eye_masks_expected_source_keypoints_run = refined_keypoints_run or keypoints_run
    (
        eye_masks_run,
        eye_masks_group,
        eye_masks_selection,
        eye_masks_latest_run,
        eye_masks_latest_source_run,
    ) = _resolve_group_for_source_run(
        eye_masks_parent,
        source_run=eye_masks_expected_source_keypoints_run,
        source_run_extractor=_extract_source_keypoints_run,
    )
    eye_masks_status, eye_masks_reason = _step_status_from_presence(
        present=eye_masks_group is not None,
        is_production=is_production,
        prerequisite_statuses=(refined_keypoints_status, keypoints_status),
    )
    if eye_masks_group is None and eye_masks_expected_source_keypoints_run and eye_masks_parent is not None:
        if eye_masks_selection == "source_mismatch":
            eye_masks_reason = "stale_vs_latest_keypoints"
        elif eye_masks_selection == "source_mismatch_missing_attr":
            eye_masks_reason = "missing_source_keypoints_run"
    eye_masks_method = _decode_text(
        eye_masks_group.attrs.get("method")  # type: ignore[union-attr]
    ) if eye_masks_group is not None else None
    eye_masks_coverage = None
    if eye_masks_group is not None:
        rate = _coerce_float_value(eye_masks_group.attrs.get("successful_roi_pair_rate"))  # type: ignore[attr-defined]
        if rate is not None:
            eye_masks_coverage = float(rate) * 100.0 if rate <= 1.0 else float(rate)

    refined_eye_masks_parent = root.get("refined_eye_masks_runs")  # type: ignore[attr-defined]
    (
        refined_eye_masks_run,
        refined_eye_masks_group,
        refined_eye_masks_selection,
        refined_eye_masks_latest_run,
        refined_eye_masks_latest_source_run,
    ) = _resolve_group_for_source_run(
        refined_eye_masks_parent,
        source_run=eye_masks_run,
        source_run_extractor=_extract_source_eye_masks_run,
    )
    refined_eye_masks_status, refined_eye_masks_reason = _step_status_from_presence(
        present=refined_eye_masks_group is not None,
        is_production=is_production,
        prerequisite_statuses=(eye_masks_status,),
    )
    if refined_eye_masks_group is None and eye_masks_run and refined_eye_masks_parent is not None:
        if refined_eye_masks_selection == "source_mismatch":
            refined_eye_masks_reason = "stale_vs_latest_eye_masks"
        elif refined_eye_masks_selection == "source_mismatch_missing_attr":
            refined_eye_masks_reason = "missing_source_eye_masks_run"
    refined_eye_masks_method = _decode_text(
        refined_eye_masks_group.attrs.get("method")  # type: ignore[union-attr]
    ) if refined_eye_masks_group is not None else None
    refined_eye_masks_review_status = (
        _coerce_mapping_value(refined_eye_masks_group.attrs.get("eye_mask_review_status"))  # type: ignore[attr-defined]
        if refined_eye_masks_group is not None
        else None
    )
    refined_eye_masks_coverage = None
    if refined_eye_masks_group is not None:
        rate = _coerce_float_value(
            refined_eye_masks_group.attrs.get("successful_roi_pair_rate")  # type: ignore[attr-defined]
        )
        if rate is not None:
            refined_eye_masks_coverage = float(rate) * 100.0 if rate <= 1.0 else float(rate)

    subject_masks_parent = root.get("subject_mask_runs")  # type: ignore[attr-defined]
    subject_masks_run, subject_masks_group, subject_masks_selection = _resolve_latest_group(subject_masks_parent)
    subject_masks_status, subject_masks_reason = _step_status_from_presence(
        present=subject_masks_group is not None,
        is_production=is_production,
        prerequisite_statuses=(crop_status,),
    )
    subject_masks_method = _decode_text(
        subject_masks_group.attrs.get("method")  # type: ignore[union-attr]
    ) if subject_masks_group is not None else None
    subject_masks_review_status = (
        _coerce_mapping_value(subject_masks_group.attrs.get("subject_mask_review_status"))  # type: ignore[attr-defined]
        if subject_masks_group is not None
        else None
    )
    subject_masks_coverage = _extract_subject_mask_coverage_pct(subject_masks_group)
    subject_masks_component_details = _extract_subject_mask_component_details(subject_masks_group)

    refined_subject_masks_parent = root.get("refined_subject_masks_runs")  # type: ignore[attr-defined]
    (
        refined_subject_masks_run,
        refined_subject_masks_group,
        refined_subject_masks_selection,
        refined_subject_masks_latest_run,
        refined_subject_masks_latest_source_run,
    ) = _resolve_group_for_source_run(
        refined_subject_masks_parent,
        source_run=subject_masks_run,
        source_run_extractor=_extract_source_subject_mask_run,
    )
    refined_subject_masks_status, refined_subject_masks_reason = _step_status_from_presence(
        present=refined_subject_masks_group is not None,
        is_production=is_production,
        prerequisite_statuses=(subject_masks_status,),
    )
    if refined_subject_masks_group is None and subject_masks_run and refined_subject_masks_parent is not None:
        if refined_subject_masks_selection == "source_mismatch":
            refined_subject_masks_reason = "stale_vs_latest_subject_masks"
        elif refined_subject_masks_selection == "source_mismatch_missing_attr":
            refined_subject_masks_reason = "missing_source_subject_mask_run"
    refined_subject_masks_method = _decode_text(
        refined_subject_masks_group.attrs.get("method")  # type: ignore[union-attr]
    ) if refined_subject_masks_group is not None else None
    if not refined_subject_masks_method and refined_subject_masks_group is not None:
        refined_subject_masks_method = "refine_subject_masks"
    refined_subject_masks_review_status = None
    if refined_subject_masks_group is not None:
        refined_subject_masks_review_status = _coerce_mapping_value(
            refined_subject_masks_group.attrs.get("refined_subject_mask_review_status")  # type: ignore[attr-defined]
        )
        if refined_subject_masks_review_status is None:
            refined_subject_masks_review_status = _coerce_mapping_value(
                refined_subject_masks_group.attrs.get("subject_mask_review_status")  # type: ignore[attr-defined]
            )
    refined_subject_masks_coverage = _extract_subject_mask_coverage_pct(refined_subject_masks_group)
    refined_subject_masks_component_details = _extract_subject_mask_component_details(refined_subject_masks_group)

    arena_assignment_parent = root.get("arena_assignment_runs")  # type: ignore[attr-defined]
    arena_assignment_run, arena_assignment_group, arena_assignment_selection = _resolve_latest_group(arena_assignment_parent)
    arena_assignment_status, arena_assignment_reason = _step_status_from_presence(
        present=arena_assignment_group is not None,
        is_production=is_production,
        prerequisite_statuses=(refined_keypoints_status, keypoints_status),
    )
    arena_assignment_method = _decode_text(
        arena_assignment_group.attrs.get("method")  # type: ignore[union-attr]
    ) if arena_assignment_group is not None else None

    tracks_parent = root.get("tracking_runs")  # type: ignore[attr-defined]
    tracks_run, tracks_group, tracks_selection = _resolve_latest_group(tracks_parent)
    tracks_status, tracks_reason = _step_status_from_presence(
        present=tracks_group is not None,
        is_production=is_production,
        prerequisite_statuses=(arena_assignment_status,),
    )
    tracks_method = _decode_text(
        tracks_group.attrs.get("method")  # type: ignore[union-attr]
    ) if tracks_group is not None else None
    tracks_summary_details = _extract_tracking_summary_details(tracks_group)

    stimulus_runs = 0
    stimulus_run: Optional[str] = None
    stimulus_group: Optional[object] = None
    stimulus_selection = "none"
    analysis_group = root.get("analysis")  # type: ignore[attr-defined]
    if analysis_group is not None and "stimulus_runs" in analysis_group:
        stimulus_parent = analysis_group["stimulus_runs"]
        stimulus_run, stimulus_group, stimulus_selection = _resolve_latest_group(stimulus_parent)
        if hasattr(stimulus_parent, "group_keys"):
            stimulus_runs = len(list(stimulus_parent.group_keys()))  # type: ignore[attr-defined]
        else:
            try:
                stimulus_runs = len(list(stimulus_parent.keys()))  # type: ignore[attr-defined]
            except Exception:
                stimulus_runs = 0
    stimulus_status = "ok" if stimulus_runs > 0 else "missing"
    stimulus_reason = "present" if stimulus_runs > 0 else "run_missing"

    calibration_group = root.get("calibration")  # type: ignore[attr-defined]
    calibration_present = calibration_group is not None
    calibration_status = "ok" if calibration_present else "missing"
    calibration_reason = "present" if calibration_present else "missing"

    analysis_meta = root.get("analysis_metadata")  # type: ignore[attr-defined]
    analysis_meta_attrs = analysis_meta.attrs if analysis_meta is not None and hasattr(analysis_meta, "attrs") else {}
    subdish_needed = subdish_required(root.attrs)  # type: ignore[attr-defined]
    tuning_step_statuses: Dict[str, tuple[str, str]] = {}
    for tuning_key in RECORDING_TUNING_STEP_NAMES:
        if tuning_key in analysis_meta_attrs:
            tuning_step_statuses[tuning_key] = ("ok", "present")
            continue
        if tuning_key == "subdish_mask_tuning" and not subdish_needed:
            tuning_step_statuses[tuning_key] = ("na", "subdish_not_required")
            continue
        tuning_step_statuses[tuning_key] = ("missing", "metadata_missing")

    common_details = {
        "is_production": is_production,
        "has_raw_video_attr": has_raw_video_attr,
        "zarr_use": zarr_use,
        "zarr_purpose": zarr_purpose,
        "pipeline_type": pipeline_type,
    }
    refined_detect_details: Dict[str, object] = {
        **common_details,
        "reason": refined_detect_reason,
        "latest_selector": refined_detect_selection,
        "upstream": {"detect": detect_status},
    }
    if detect_run:
        refined_detect_details["expected_source_detect_run"] = detect_run
    refined_detect_source_run = _extract_source_detect_run(refined_detect_group)
    if refined_detect_source_run:
        refined_detect_details["source_detect_run"] = refined_detect_source_run
    if refined_detect_latest_run:
        refined_detect_details["latest_refined_detect_run"] = refined_detect_latest_run
    if refined_detect_latest_source_run:
        refined_detect_details["latest_refined_detect_source_run"] = refined_detect_latest_source_run

    refined_keypoints_details: Dict[str, object] = {
        **common_details,
        "reason": refined_keypoints_reason,
        "latest_selector": refined_keypoints_selection,
        "upstream": {"keypoints": keypoints_status},
    }
    if keypoints_run:
        refined_keypoints_details["expected_source_keypoints_run"] = keypoints_run
    refined_keypoints_source_run = _extract_source_keypoints_run(refined_keypoints_group)
    if refined_keypoints_source_run:
        refined_keypoints_details["source_keypoints_run"] = refined_keypoints_source_run
    if refined_keypoints_latest_run:
        refined_keypoints_details["latest_refined_run"] = refined_keypoints_latest_run
    if refined_keypoints_latest_source_run:
        refined_keypoints_details["latest_refined_source_keypoints_run"] = refined_keypoints_latest_source_run
    eye_masks_details: Dict[str, object] = {
        **common_details,
        "reason": eye_masks_reason,
        "latest_selector": eye_masks_selection,
        "upstream": {
            "keypoints": keypoints_status,
            "refined_keypoints": refined_keypoints_status,
        },
    }
    if eye_masks_expected_source_keypoints_run:
        eye_masks_details["expected_source_keypoints_run"] = eye_masks_expected_source_keypoints_run
    eye_masks_source_run = _extract_source_keypoints_run(eye_masks_group)
    if eye_masks_source_run:
        eye_masks_details["source_keypoints_run"] = eye_masks_source_run
    if eye_masks_latest_run:
        eye_masks_details["latest_eye_masks_run"] = eye_masks_latest_run
    if eye_masks_latest_source_run:
        eye_masks_details["latest_eye_masks_source_keypoints_run"] = eye_masks_latest_source_run

    refined_eye_masks_details: Dict[str, object] = {
        **common_details,
        "reason": refined_eye_masks_reason,
        "latest_selector": refined_eye_masks_selection,
        "upstream": {"eye_masks": eye_masks_status},
    }
    if eye_masks_run:
        refined_eye_masks_details["expected_source_eye_masks_run"] = eye_masks_run
    refined_eye_masks_source_run = _extract_source_eye_masks_run(refined_eye_masks_group)
    if refined_eye_masks_source_run:
        refined_eye_masks_details["source_eye_masks_run"] = refined_eye_masks_source_run
    if refined_eye_masks_latest_run:
        refined_eye_masks_details["latest_refined_eye_masks_run"] = refined_eye_masks_latest_run
    if refined_eye_masks_latest_source_run:
        refined_eye_masks_details["latest_refined_eye_masks_source_run"] = refined_eye_masks_latest_source_run

    subject_masks_details: Dict[str, object] = {
        **common_details,
        "reason": subject_masks_reason,
        "latest_selector": subject_masks_selection,
        "upstream": {"crop": crop_status},
        **subject_masks_component_details,
    }

    refined_subject_masks_details: Dict[str, object] = {
        **common_details,
        "reason": refined_subject_masks_reason,
        "latest_selector": refined_subject_masks_selection,
        "upstream": {"subject_masks": subject_masks_status},
        **refined_subject_masks_component_details,
    }
    if subject_masks_run:
        refined_subject_masks_details["expected_source_subject_mask_run"] = subject_masks_run
    refined_subject_masks_source_run = _extract_source_subject_mask_run(refined_subject_masks_group)
    if refined_subject_masks_source_run:
        refined_subject_masks_details["source_subject_mask_run"] = refined_subject_masks_source_run
    if refined_subject_masks_latest_run:
        refined_subject_masks_details["latest_refined_subject_masks_run"] = refined_subject_masks_latest_run
    if refined_subject_masks_latest_source_run:
        refined_subject_masks_details["latest_refined_subject_masks_source_run"] = refined_subject_masks_latest_source_run

    rows: List[Dict[str, object]] = [
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="raw",
            status=raw_status,
            run_name=None,
            method=None,
            coverage_pct=None,
            review_status=None,
            details={
                **common_details,
                "reason": raw_reason,
                "raw_present": raw_present,
                "full_present": full_present,
                "ds_present": ds_present,
                "sampled_present": sampled_present,
            },
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(raw_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="background",
            status=background_status,
            run_name=background_run,
            method=background_method,
            coverage_pct=None,
            review_status=None,
            details={
                **common_details,
                "reason": background_reason,
                "latest_selector": background_selection,
                "full_present": background_full_present,
                "ds_present": background_ds_present,
                "upstream": {"raw": raw_status},
            },
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(background_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="detect",
            status=detect_status,
            run_name=detect_run,
            method=detect_method,
            coverage_pct=detect_coverage,
            review_status=None,
            details={
                **common_details,
                "reason": detect_reason,
                "latest_selector": detect_selection,
                **detect_quality_details,
            },
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(detect_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="refined_detect",
            status=refined_detect_status,
            run_name=refined_detect_run,
            method=refined_detect_method,
            coverage_pct=refined_detect_coverage,
            review_status=detect_review_status,
            details=refined_detect_details,
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(refined_detect_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="crop",
            status=crop_status,
            run_name=crop_run,
            method=crop_method,
            coverage_pct=_extract_coverage_pct(crop_group),
            review_status=crop_review_status,
            details={
                **common_details,
                "reason": crop_reason,
                "latest_selector": crop_selection,
                "run_state": crop_run_state,
                "upstream": {"detect": detect_status, "refined_detect": refined_detect_status},
            },
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(crop_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="keypoints",
            status=keypoints_status,
            run_name=keypoints_run,
            method=keypoints_method,
            coverage_pct=None,
            review_status=None,
            details={
                **common_details,
                "reason": keypoints_reason,
                "latest_selector": keypoints_selection,
                "upstream": {"crop": crop_status},
            },
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(keypoints_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="refined_keypoints",
            status=refined_keypoints_status,
            run_name=refined_keypoints_run,
            method=refined_keypoints_method,
            coverage_pct=refined_keypoints_coverage,
            review_status=keypoint_review_status,
            details=refined_keypoints_details,
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(refined_keypoints_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="eye_masks",
            status=eye_masks_status,
            run_name=eye_masks_run,
            method=eye_masks_method,
            coverage_pct=eye_masks_coverage,
            review_status=None,
            details=eye_masks_details,
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(eye_masks_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="refined_eye_masks",
            status=refined_eye_masks_status,
            run_name=refined_eye_masks_run,
            method=refined_eye_masks_method,
            coverage_pct=refined_eye_masks_coverage,
            review_status=refined_eye_masks_review_status,
            details=refined_eye_masks_details,
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(refined_eye_masks_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="subject_masks",
            status=subject_masks_status,
            run_name=subject_masks_run,
            method=subject_masks_method,
            coverage_pct=subject_masks_coverage,
            review_status=subject_masks_review_status,
            details=subject_masks_details,
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(subject_masks_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="refined_subject_masks",
            status=refined_subject_masks_status,
            run_name=refined_subject_masks_run,
            method=refined_subject_masks_method,
            coverage_pct=refined_subject_masks_coverage,
            review_status=refined_subject_masks_review_status,
            details=refined_subject_masks_details,
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(refined_subject_masks_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="arena_assignment",
            status=arena_assignment_status,
            run_name=arena_assignment_run,
            method=arena_assignment_method,
            coverage_pct=None,
            review_status=None,
            details={
                **common_details,
                "reason": arena_assignment_reason,
                "latest_selector": arena_assignment_selection,
                "upstream": {
                    "keypoints": keypoints_status,
                    "refined_keypoints": refined_keypoints_status,
                },
            },
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(arena_assignment_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="tracks",
            status=tracks_status,
            run_name=tracks_run,
            method=tracks_method,
            coverage_pct=None,
            review_status=None,
            details={
                **common_details,
                "reason": tracks_reason,
                "latest_selector": tracks_selection,
                "upstream": {"arena_assignment": arena_assignment_status},
                **tracks_summary_details,
            },
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(tracks_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="stimulus",
            status=stimulus_status,
            run_name=stimulus_run,
            method=None,
            coverage_pct=None,
            review_status=None,
            details={
                **common_details,
                "reason": stimulus_reason,
                "stimulus_runs": stimulus_runs,
                "latest_selector": stimulus_selection,
            },
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(stimulus_group, fallback=fallback_updated_utc),
        ),
        _make_recording_step_row(
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="calibration",
            status=calibration_status,
            run_name=None,
            method=None,
            coverage_pct=None,
            review_status=None,
            details={**common_details, "reason": calibration_reason},
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
            updated_utc=_extract_updated_utc(calibration_group, fallback=fallback_updated_utc),
        ),
        *[
            _make_recording_step_row(
                dataset_id=dataset_id,
                recording_id=recording_id,
                step_name=tuning_key,
                status=tuning_step_statuses[tuning_key][0],
                run_name=None,
                method=None,
                coverage_pct=None,
                review_status=None,
                details={
                    **common_details,
                    "reason": tuning_step_statuses[tuning_key][1],
                    "upstream": {"calibration": calibration_status},
                },
                source=source,
                zarr_mtime_ns=zarr_mtime_ns,
                updated_utc=_extract_updated_utc(analysis_meta, fallback=fallback_updated_utc),
            )
            for tuning_key in RECORDING_TUNING_STEP_NAMES
        ],
    ]
    return rows


def _recording_step_row_signature(row: Dict[str, object]) -> tuple[object, ...]:
    zarr_mtime = row.get("zarr_mtime_ns")
    zarr_mtime_norm: Optional[int] = None
    if zarr_mtime is not None:
        try:
            zarr_mtime_norm = int(zarr_mtime)
        except Exception:
            zarr_mtime_norm = None
    return (
        _decode_text(row.get("recording_id")),
        _decode_text(row.get("status")),
        _decode_text(row.get("run_name")),
        _decode_text(row.get("method")),
        _coerce_float_value(row.get("coverage_pct")),
        _decode_text(row.get("review_status_json")),
        _decode_text(row.get("details_json")),
        _decode_text(row.get("source")),
        zarr_mtime_norm,
        _decode_text(row.get("updated_utc")),
    )


def _backfill_recording_step_status(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    recording_ids: Optional[Sequence[str]],
    zarr_use_filter: str,
) -> Dict[str, object]:
    rows = registry.conn.execute(
        """
        SELECT dataset_id, zarr_path, recording_id, zarr_use
        FROM datasets
        WHERE status IS NULL OR lower(status) != 'missing'
        ORDER BY dataset_id;
        """
    ).fetchall()
    scope_roots = _normalize_scope_paths(scope_paths)
    zarr = _import_zarr()
    recording_id_filter = set(_normalize_recording_ids(recording_ids))
    zarr_use_norm = str(zarr_use_filter or "all").strip().lower()
    if zarr_use_norm not in {"all", "analysis", "training"}:
        raise ValueError("zarr_use_filter must be one of: all, analysis, training")

    summary: Dict[str, object] = {
        "datasets_scanned": 0,
        "datasets_in_scope": 0,
        "datasets_skipped_path": 0,
        "datasets_skipped_missing_recording_id": 0,
        "datasets_skipped_recording_filter": 0,
        "datasets_skipped_zarr_use_filter": 0,
        "datasets_missing_zarr": 0,
        "datasets_errors": 0,
        "rows_evaluated": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_skipped": 0,
        "history_rows_inserted": 0,
        "rows_by_status": {status: 0 for status in RECORDING_STEP_STATUS_VALUES},
        "rows_by_step": {step_name: 0 for step_name in RECORDING_STEP_NAMES},
        "filters": {
            "recording_ids": sorted(recording_id_filter),
            "zarr_use": zarr_use_norm,
            "scope_paths": [str(path) for path in scope_roots],
        },
    }

    current_upserts: List[Dict[str, object]] = []
    history_inserts: List[Dict[str, object]] = []
    source = "maintenance_backfill_recording_step_status"

    for row in rows:
        summary["datasets_scanned"] = int(summary["datasets_scanned"]) + 1
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser()
        if not _matches_scope(str(zarr_path), scope_roots):
            summary["datasets_skipped_path"] = int(summary["datasets_skipped_path"]) + 1
            continue
        summary["datasets_in_scope"] = int(summary["datasets_in_scope"]) + 1

        recording_id = _decode_text(row["recording_id"])
        if not recording_id:
            summary["datasets_skipped_missing_recording_id"] = int(summary["datasets_skipped_missing_recording_id"]) + 1
            continue
        if recording_id_filter and recording_id not in recording_id_filter:
            summary["datasets_skipped_recording_filter"] = int(summary["datasets_skipped_recording_filter"]) + 1
            continue

        row_zarr_use = _decode_text(row["zarr_use"])
        row_zarr_use_norm = (row_zarr_use or "").lower()
        if zarr_use_norm != "all" and row_zarr_use_norm != zarr_use_norm:
            summary["datasets_skipped_zarr_use_filter"] = int(summary["datasets_skipped_zarr_use_filter"]) + 1
            continue

        if not _is_zarr_root_path(zarr_path):
            summary["datasets_missing_zarr"] = int(summary["datasets_missing_zarr"]) + 1
            continue

        try:
            zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
        except Exception:
            zarr_mtime_ns = None

        extracted_rows: List[Dict[str, object]]
        try:
            try:
                root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
            except TypeError:
                try:
                    root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
                except TypeError:
                    root = zarr.open_group(str(zarr_path), mode="r")
            extracted_rows = _build_recording_step_rows_from_root(
                root=root,
                dataset_id=dataset_id,
                recording_id=recording_id,
                zarr_use=row_zarr_use,
                zarr_mtime_ns=zarr_mtime_ns,
                source=source,
            )
        except Exception as exc:
            summary["datasets_errors"] = int(summary["datasets_errors"]) + 1
            extracted_rows = _build_recording_step_error_rows(
                dataset_id=dataset_id,
                recording_id=recording_id,
                zarr_mtime_ns=zarr_mtime_ns,
                error_detail=str(exc),
                source=source,
            )

        existing_rows = registry.conn.execute(
            "SELECT * FROM recording_step_status WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchall()
        existing_by_step: Dict[str, Dict[str, object]] = {
            str(existing["step_name"]): {key: existing[key] for key in existing.keys()}
            for existing in existing_rows
            if existing["step_name"] is not None
        }

        for extracted in extracted_rows:
            step_name = str(extracted["step_name"])
            status = str(extracted["status"])
            summary["rows_evaluated"] = int(summary["rows_evaluated"]) + 1
            rows_by_status = summary["rows_by_status"]
            if isinstance(rows_by_status, dict) and status in rows_by_status:
                rows_by_status[status] = int(rows_by_status[status]) + 1
            rows_by_step = summary["rows_by_step"]
            if isinstance(rows_by_step, dict) and step_name in rows_by_step:
                rows_by_step[step_name] = int(rows_by_step[step_name]) + 1

            existing = existing_by_step.get(step_name)
            if existing is None:
                summary["rows_inserted"] = int(summary["rows_inserted"]) + 1
                if not dry_run:
                    current_upserts.append(extracted)
                    history_inserts.append(extracted)
                continue

            if _recording_step_row_signature(existing) == _recording_step_row_signature(extracted):
                summary["rows_skipped"] = int(summary["rows_skipped"]) + 1
                continue

            summary["rows_updated"] = int(summary["rows_updated"]) + 1
            if not dry_run:
                current_upserts.append(extracted)
                history_inserts.append(extracted)

    if not dry_run and current_upserts:
        recorded_utc = datetime.now(timezone.utc).isoformat()
        with registry.conn:
            for payload in current_upserts:
                registry.conn.execute(
                    """
                    INSERT INTO recording_step_status (
                        dataset_id,
                        recording_id,
                        step_name,
                        status,
                        run_name,
                        method,
                        coverage_pct,
                        review_status_json,
                        details_json,
                        source,
                        zarr_mtime_ns,
                        updated_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_id, step_name) DO UPDATE SET
                        recording_id=excluded.recording_id,
                        status=excluded.status,
                        run_name=excluded.run_name,
                        method=excluded.method,
                        coverage_pct=excluded.coverage_pct,
                        review_status_json=excluded.review_status_json,
                        details_json=excluded.details_json,
                        source=excluded.source,
                        zarr_mtime_ns=excluded.zarr_mtime_ns,
                        updated_utc=excluded.updated_utc;
                    """,
                    (
                        payload.get("dataset_id"),
                        payload.get("recording_id"),
                        payload.get("step_name"),
                        payload.get("status"),
                        payload.get("run_name"),
                        payload.get("method"),
                        payload.get("coverage_pct"),
                        payload.get("review_status_json"),
                        payload.get("details_json"),
                        payload.get("source"),
                        payload.get("zarr_mtime_ns"),
                        payload.get("updated_utc"),
                    ),
                )
            for payload in history_inserts:
                registry.conn.execute(
                    """
                    INSERT INTO recording_step_status_history (
                        dataset_id,
                        recording_id,
                        step_name,
                        status,
                        run_name,
                        method,
                        coverage_pct,
                        review_status_json,
                        details_json,
                        source,
                        zarr_mtime_ns,
                        updated_utc,
                        recorded_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        payload.get("dataset_id"),
                        payload.get("recording_id"),
                        payload.get("step_name"),
                        payload.get("status"),
                        payload.get("run_name"),
                        payload.get("method"),
                        payload.get("coverage_pct"),
                        payload.get("review_status_json"),
                        payload.get("details_json"),
                        payload.get("source"),
                        payload.get("zarr_mtime_ns"),
                        payload.get("updated_utc"),
                        recorded_utc,
                    ),
                )
        summary["history_rows_inserted"] = len(history_inserts)

    return summary


def _load_recording_type_subtype_vocab(
    registry: Registry,
) -> tuple[set[str], dict[str, set[str]]]:
    """Load active recording vocab from DB, with hardcoded fallback."""
    type_rows = registry.conn.execute(
        """
        SELECT recording_type
        FROM recording_type_vocab
        WHERE active = 1;
        """
    ).fetchall()
    subtype_rows = registry.conn.execute(
        """
        SELECT recording_type, recording_subtype
        FROM recording_subtype_vocab
        WHERE active = 1;
        """
    ).fetchall()

    allowed_types = {
        str(row["recording_type"]).strip()
        for row in type_rows
        if row["recording_type"] is not None and str(row["recording_type"]).strip()
    }
    allowed_subtypes_by_type: dict[str, set[str]] = {}
    for row in subtype_rows:
        recording_type = str(row["recording_type"]).strip() if row["recording_type"] is not None else ""
        recording_subtype = (
            str(row["recording_subtype"]).strip() if row["recording_subtype"] is not None else ""
        )
        if not recording_type or not recording_subtype:
            continue
        allowed_subtypes_by_type.setdefault(recording_type, set()).add(recording_subtype)

    if not allowed_types:
        allowed_types = set(DEFAULT_ALLOWED_RECORDING_TYPES)
    if not allowed_subtypes_by_type:
        allowed_subtypes_by_type = {
            key: set(values) for key, values in DEFAULT_ALLOWED_RECORDING_SUBTYPES_BY_TYPE.items()
        }

    return allowed_types, allowed_subtypes_by_type


def _check_registry_integrity(registry: Registry) -> List[IntegrityIssue]:
    issues: List[IntegrityIssue] = []
    behavior_v1_required_artifacts = {
        "h5_log",
        "camera_video",
        "camera_metadata_csv",
        "timing_profile_csv",
    }
    allowed_recording_types, allowed_subtypes_by_type = _load_recording_type_subtype_vocab(registry)

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

    dataset_rows = registry.conn.execute(
        """
        SELECT dataset_id, artifact_kind
        FROM datasets;
        """
    ).fetchall()
    dataset_kind: Dict[str, str] = {
        str(row["dataset_id"]): str(row["artifact_kind"] or "")
        for row in dataset_rows
        if row["dataset_id"] is not None
    }

    required_views = (
        "dataset_context_current",
        "dataset_lineage_current",
        "merged_training_datasets",
        "recording_overview",
        "recording_subject_overview",
        "keypoint_quality_current",
        "detect_quality_current",
    )
    view_ok: Dict[str, bool] = {}
    for view_name in required_views:
        exists_row = registry.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'view' AND name = ?
            LIMIT 1;
            """,
            (view_name,),
        ).fetchone()
        if exists_row is None:
            issues.append(
                IntegrityIssue(
                    code="required_view_missing",
                    run_id=view_name,
                    detail=f"required view is missing: {view_name}",
                )
            )
            view_ok[view_name] = False
            continue
        try:
            registry.conn.execute(f"SELECT COUNT(*) FROM {view_name};").fetchone()
            view_ok[view_name] = True
        except Exception as exc:
            issues.append(
                IntegrityIssue(
                    code="required_view_query_error",
                    run_id=view_name,
                    detail=f"required view query failed: {view_name} ({exc})",
                )
            )
            view_ok[view_name] = False

    required_tables = (
        "crosses",
        "dishes",
        "recording_subjects",
        "subjects",
    )
    table_ok: Dict[str, bool] = {}
    for table_name in required_tables:
        exists_row = registry.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            LIMIT 1;
            """,
            (table_name,),
        ).fetchone()
        if exists_row is None:
            issues.append(
                IntegrityIssue(
                    code="required_table_missing",
                    run_id=table_name,
                    detail=f"required table is missing: {table_name}",
                )
            )
            table_ok[table_name] = False
            continue
        try:
            registry.conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()
            table_ok[table_name] = True
        except Exception as exc:
            issues.append(
                IntegrityIssue(
                    code="required_table_query_error",
                    run_id=table_name,
                    detail=f"required table query failed: {table_name} ({exc})",
                )
            )
            table_ok[table_name] = False

    def _relation_queryable(kind: str, name: str) -> bool:
        exists_row = registry.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = ? AND name = ?
            LIMIT 1;
            """,
            (kind, name),
        ).fetchone()
        if exists_row is None:
            return False
        try:
            registry.conn.execute(f"SELECT COUNT(*) FROM {name};").fetchone()
        except Exception:
            return False
        return True

    # Every training run should have a training_models row after migration.
    missing_dm_rows = registry.conn.execute(
        """
        SELECT tr.run_id
        FROM training_runs tr
        LEFT JOIN training_models dm ON dm.run_id = tr.run_id
        WHERE dm.run_id IS NULL
        ORDER BY tr.created_utc DESC, tr.run_id DESC;
        """
    ).fetchall()
    for row in missing_dm_rows:
        issues.append(
            IntegrityIssue(
                code="missing_detection_model_row",
                run_id=str(row["run_id"]),
                detail="training_runs row has no training_models row",
            )
        )

    # Success runs should have model and metrics path populated and existing (if present).
    success_rows = registry.conn.execute(
        """
        SELECT run_id, model_path, metrics_path
        FROM training_models
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
                    detail="training_models.status=success but model_path is NULL/empty",
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

    lineage_rows: List[Any] = []
    if view_ok.get("dataset_lineage_current", False):
        lineage_rows = registry.conn.execute(
            """
            SELECT child_dataset_id, parent_dataset_id, relationship_type
            FROM dataset_lineage_current
            ORDER BY child_dataset_id, parent_dataset_id;
            """
        ).fetchall()
    for row in lineage_rows:
        child_dataset_id = str(row["child_dataset_id"])
        parent_dataset_id = str(row["parent_dataset_id"])
        relationship_type = str(row["relationship_type"] or "")
        if child_dataset_id == parent_dataset_id:
            issues.append(
                IntegrityIssue(
                    code="dataset_lineage_self_edge",
                    run_id=child_dataset_id,
                    detail=f"relationship_type={relationship_type}",
                )
            )

    # Detect lineage cycles in child->parent graph.
    adjacency: Dict[str, Set[str]] = {}
    for row in lineage_rows:
        child_dataset_id = str(row["child_dataset_id"])
        parent_dataset_id = str(row["parent_dataset_id"])
        adjacency.setdefault(child_dataset_id, set()).add(parent_dataset_id)
        adjacency.setdefault(parent_dataset_id, set())

    visited: Set[str] = set()
    active_stack: List[str] = []
    active_set: Set[str] = set()
    cycle_signatures: Set[str] = set()

    def _dfs(node: str) -> None:
        visited.add(node)
        active_stack.append(node)
        active_set.add(node)
        for neighbor in sorted(adjacency.get(node, set())):
            if neighbor in active_set:
                start_idx = active_stack.index(neighbor)
                cycle_nodes = active_stack[start_idx:] + [neighbor]
                signature = "->".join(cycle_nodes)
                if signature not in cycle_signatures:
                    cycle_signatures.add(signature)
                    issues.append(
                        IntegrityIssue(
                            code="dataset_lineage_cycle",
                            run_id=node,
                            detail=f"cycle={signature}",
                        )
                    )
                continue
            if neighbor not in visited:
                _dfs(neighbor)
        active_stack.pop()
        active_set.remove(node)

    for node in sorted(adjacency.keys()):
        if node not in visited:
            _dfs(node)

    merged_dataset_ids = [
        dataset_id
        for dataset_id, artifact_kind in dataset_kind.items()
        if artifact_kind == "derived_training_merge" or dataset_id.endswith("_merged")
    ]
    for dataset_id in merged_dataset_ids:
        parent_count = int(
            registry.conn.execute(
                """
                SELECT COUNT(*)
                FROM dataset_lineage_current
                WHERE child_dataset_id = ? AND relationship_type = 'training_merge_source';
                """,
                (dataset_id,),
            ).fetchone()[0]
        )
        if parent_count == 0:
            issues.append(
                IntegrityIssue(
                    code="merged_dataset_missing_lineage",
                    run_id=dataset_id,
                    detail=f"dataset_id={dataset_id} has no training_merge_source parents",
                )
            )

    training_set_rows = registry.conn.execute(
        """
        SELECT set_id, dataset_ids_json
        FROM training_sets;
        """
    ).fetchall()
    for row in training_set_rows:
        set_id = str(row["set_id"])
        dataset_ids = sorted(set(_json_text_list(row["dataset_ids_json"])))
        merged_ids = [
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id in dataset_kind
            and (
                dataset_kind.get(dataset_id) == "derived_training_merge"
                or dataset_id.endswith("_merged")
            )
        ]
        if not merged_ids:
            continue
        expected_parents = {
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id in dataset_kind and dataset_id not in merged_ids
        }
        for child_dataset_id in merged_ids:
            expected = {item for item in expected_parents if item != child_dataset_id}
            actual_rows = registry.conn.execute(
                """
                SELECT parent_dataset_id
                FROM dataset_lineage_current
                WHERE child_dataset_id = ? AND relationship_type = 'training_merge_source';
                """,
                (child_dataset_id,),
            ).fetchall()
            actual = {
                str(item["parent_dataset_id"])
                for item in actual_rows
                if item["parent_dataset_id"] is not None
            }
            if expected and actual != expected:
                issues.append(
                    IntegrityIssue(
                        code="training_set_lineage_mismatch",
                        run_id=child_dataset_id,
                        detail=(
                            f"set_id={set_id} expected={','.join(sorted(expected))} "
                            f"actual={','.join(sorted(actual))}"
                        ),
                    )
                )

    # Keypoint quality rows should be fresh and consistent with current Zarr metadata.
    if view_ok.get("keypoint_quality_current", False):
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

    if view_ok.get("dataset_context_current", False):
        dcc_cardinality_rows = registry.conn.execute(
            """
            SELECT d.dataset_id, COUNT(dcc.dataset_id) AS row_count
            FROM datasets d
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = d.dataset_id
            GROUP BY d.dataset_id
            HAVING COUNT(dcc.dataset_id) != 1
            ORDER BY d.dataset_id;
            """
        ).fetchall()
        for row in dcc_cardinality_rows:
            dataset_id = str(row["dataset_id"] or "")
            row_count = int(row["row_count"] or 0)
            issues.append(
                IntegrityIssue(
                    code="dataset_context_current_cardinality_mismatch",
                    run_id=dataset_id or None,
                    detail=(
                        f"dataset_id={dataset_id} appears {row_count} time(s) in dataset_context_current; "
                        "expected exactly 1 row per dataset"
                    ),
                )
            )

    # Source recording datasets should link to an existing recording.
    source_rows = registry.conn.execute(
        """
        SELECT d.dataset_id, d.recording_id
        FROM datasets d
        WHERE d.artifact_kind = 'source_recording'
        ORDER BY d.dataset_id;
        """
    ).fetchall()
    for row in source_rows:
        dataset_id = str(row["dataset_id"])
        recording_id = row["recording_id"]
        if not recording_id:
            issues.append(
                IntegrityIssue(
                    code="source_dataset_missing_recording_id",
                    run_id=dataset_id,
                    detail=f"dataset_id={dataset_id} has artifact_kind=source_recording but recording_id is NULL/empty",
                )
            )
            continue
        exists = registry.conn.execute(
            "SELECT 1 FROM recordings WHERE recording_id = ? LIMIT 1;",
            (str(recording_id),),
        ).fetchone()
        if exists is None:
            issues.append(
                IntegrityIssue(
                    code="source_dataset_recording_missing",
                    run_id=dataset_id,
                    detail=f"dataset_id={dataset_id} references missing recording_id={recording_id}",
                )
            )

    # Source-recording subject-count sanity for legacy provenance snapshots.
    source_consistency_rows = registry.conn.execute(
        """
        SELECT
            d.dataset_id,
            p.subject_count
        FROM datasets d
        LEFT JOIN provenance p ON p.dataset_id = d.dataset_id
        WHERE d.artifact_kind = 'source_recording'
        ORDER BY d.dataset_id;
        """
    ).fetchall()
    for row in source_consistency_rows:
        dataset_id = str(row["dataset_id"])
        subject_count = row["subject_count"]
        if subject_count is not None:
            try:
                subject_count_int = int(subject_count)
            except Exception:
                issues.append(
                    IntegrityIssue(
                        code="source_subject_count_invalid",
                        run_id=dataset_id,
                        detail=f"dataset_id={dataset_id} subject_count={subject_count} is non-integer",
                    )
                )
                continue
            if subject_count_int < 1:
                issues.append(
                    IntegrityIssue(
                        code="source_subject_count_invalid",
                        run_id=dataset_id,
                        detail=f"dataset_id={dataset_id} subject_count={subject_count_int} must be >= 1",
                    )
                )

    if (
        table_ok.get("recording_subjects", False)
        and table_ok.get("subjects", False)
        and table_ok.get("dishes", False)
    ):
        missing_subject_rows = registry.conn.execute(
            """
            SELECT rs.recording_id, rs.subject_id
            FROM recording_subjects rs
            LEFT JOIN subjects s ON s.subject_id = rs.subject_id
            WHERE s.subject_id IS NULL
            ORDER BY rs.recording_id, rs.subject_id;
            """
        ).fetchall()
        for row in missing_subject_rows:
            recording_id = str(row["recording_id"] or "")
            subject_id = str(row["subject_id"] or "")
            issues.append(
                IntegrityIssue(
                    code="recording_subject_missing_subject",
                    run_id=recording_id or None,
                    detail=(
                        f"recording_id={recording_id} subject_id={subject_id} "
                        "exists in recording_subjects but not in subjects"
                    ),
                )
            )

        subject_missing_dish_rows = registry.conn.execute(
            """
            SELECT s.subject_id, s.dish_id
            FROM subjects s
            LEFT JOIN dishes d ON d.dish_id = s.dish_id
            WHERE s.dish_id IS NOT NULL AND d.dish_id IS NULL
            ORDER BY s.subject_id;
            """
        ).fetchall()
        for row in subject_missing_dish_rows:
            subject_id = str(row["subject_id"] or "")
            dish_id = str(row["dish_id"] or "")
            issues.append(
                IntegrityIssue(
                    code="subject_missing_dish",
                    run_id=subject_id or None,
                    detail=f"subject_id={subject_id} references missing dish_id={dish_id}",
                )
            )

        cross_mismatch_rows = registry.conn.execute(
            """
            SELECT rs.recording_id, rs.subject_id, rs.cross_id, d.cross_id AS dish_cross_id
            FROM recording_subjects rs
            LEFT JOIN dishes d ON d.dish_id = rs.dish_id
            WHERE rs.cross_id IS NOT NULL
              AND d.cross_id IS NOT NULL
              AND rs.cross_id != d.cross_id
            ORDER BY rs.recording_id, rs.subject_id;
            """
        ).fetchall()
        for row in cross_mismatch_rows:
            recording_id = str(row["recording_id"] or "")
            subject_id = str(row["subject_id"] or "")
            rs_cross_id = str(row["cross_id"] or "")
            dish_cross_id = str(row["dish_cross_id"] or "")
            issues.append(
                IntegrityIssue(
                    code="recording_subject_cross_mismatch_dish",
                    run_id=recording_id or None,
                    detail=(
                        f"recording_id={recording_id} subject_id={subject_id} "
                        f"recording_subjects.cross_id={rs_cross_id} dishes.cross_id={dish_cross_id}"
                    ),
                )
            )

    if _relation_queryable("table", "detect_quality") and _relation_queryable("table", "detect_performance"):
        missing_detect_projection_rows = registry.conn.execute(
            """
            SELECT dq.dataset_id, dq.refined_run, dq.source_detect_run
            FROM detect_quality dq
            LEFT JOIN detect_performance dp
              ON dp.dataset_id = dq.dataset_id
             AND dp.detect_run = dq.source_detect_run
            WHERE TRIM(COALESCE(dq.source_detect_run, '')) != ''
              AND dp.detect_run IS NULL
            ORDER BY dq.dataset_id, dq.refined_run;
            """
        ).fetchall()
        for row in missing_detect_projection_rows:
            dataset_id = str(row["dataset_id"] or "")
            refined_run = str(row["refined_run"] or "")
            source_detect_run = str(row["source_detect_run"] or "")
            issues.append(
                IntegrityIssue(
                    code="detect_quality_missing_source_detect_projection",
                    run_id=f"{dataset_id}:{refined_run}" if dataset_id or refined_run else None,
                    detail=(
                        f"dataset_id={dataset_id} refined_run={refined_run} "
                        f"source_detect_run={source_detect_run} has no matching detect_performance row"
                    ),
                )
            )

    if _relation_queryable("table", "keypoint_performance") and _relation_queryable("table", "crop_quality"):
        missing_crop_projection_rows = registry.conn.execute(
            """
            SELECT kp.dataset_id, kp.keypoint_run, kp.source_crop_run
            FROM keypoint_performance kp
            LEFT JOIN crop_quality cq
              ON cq.dataset_id = kp.dataset_id
             AND cq.crop_run = kp.source_crop_run
            WHERE TRIM(COALESCE(kp.source_crop_run, '')) != ''
              AND cq.crop_run IS NULL
            ORDER BY kp.dataset_id, kp.keypoint_run;
            """
        ).fetchall()
        for row in missing_crop_projection_rows:
            dataset_id = str(row["dataset_id"] or "")
            keypoint_run = str(row["keypoint_run"] or "")
            source_crop_run = str(row["source_crop_run"] or "")
            issues.append(
                IntegrityIssue(
                    code="keypoint_performance_missing_source_crop_projection",
                    run_id=f"{dataset_id}:{keypoint_run}" if dataset_id or keypoint_run else None,
                    detail=(
                        f"dataset_id={dataset_id} keypoint_run={keypoint_run} "
                        f"source_crop_run={source_crop_run} has no matching crop_quality row"
                    ),
                )
            )

    if _relation_queryable("table", "keypoint_performance") and _relation_queryable("table", "detect_performance"):
        missing_keypoint_detect_rows = registry.conn.execute(
            """
            SELECT kp.dataset_id, kp.keypoint_run, kp.source_detect_run
            FROM keypoint_performance kp
            LEFT JOIN detect_performance dp
              ON dp.dataset_id = kp.dataset_id
             AND dp.detect_run = kp.source_detect_run
            WHERE TRIM(COALESCE(kp.source_detect_run, '')) != ''
              AND dp.detect_run IS NULL
            ORDER BY kp.dataset_id, kp.keypoint_run;
            """
        ).fetchall()
        for row in missing_keypoint_detect_rows:
            dataset_id = str(row["dataset_id"] or "")
            keypoint_run = str(row["keypoint_run"] or "")
            source_detect_run = str(row["source_detect_run"] or "")
            issues.append(
                IntegrityIssue(
                    code="keypoint_performance_missing_source_detect_projection",
                    run_id=f"{dataset_id}:{keypoint_run}" if dataset_id or keypoint_run else None,
                    detail=(
                        f"dataset_id={dataset_id} keypoint_run={keypoint_run} "
                        f"source_detect_run={source_detect_run} has no matching detect_performance row"
                    ),
                )
            )

    if _relation_queryable("table", "keypoint_quality") and _relation_queryable("table", "keypoint_performance"):
        missing_keypoint_projection_rows = registry.conn.execute(
            """
            SELECT kq.dataset_id, kq.refined_run, kq.source_keypoint_run
            FROM keypoint_quality kq
            LEFT JOIN keypoint_performance kp
              ON kp.dataset_id = kq.dataset_id
             AND kp.keypoint_run = kq.source_keypoint_run
            WHERE TRIM(COALESCE(kq.source_keypoint_run, '')) != ''
              AND kp.keypoint_run IS NULL
            ORDER BY kq.dataset_id, kq.refined_run;
            """
        ).fetchall()
        for row in missing_keypoint_projection_rows:
            dataset_id = str(row["dataset_id"] or "")
            refined_run = str(row["refined_run"] or "")
            source_keypoint_run = str(row["source_keypoint_run"] or "")
            issues.append(
                IntegrityIssue(
                    code="keypoint_quality_missing_source_keypoint_projection",
                    run_id=f"{dataset_id}:{refined_run}" if dataset_id or refined_run else None,
                    detail=(
                        f"dataset_id={dataset_id} refined_run={refined_run} "
                        f"source_keypoint_run={source_keypoint_run} has no matching keypoint_performance row"
                    ),
                )
            )

    if (
        _relation_queryable("table", "eye_mask_performance")
        and _relation_queryable("table", "keypoint_performance")
        and _relation_queryable("table", "keypoint_quality")
    ):
        missing_eye_keypoint_rows = registry.conn.execute(
            """
            SELECT emp.dataset_id, emp.stage_group, emp.run_name, emp.source_keypoints_run
            FROM eye_mask_performance emp
            WHERE TRIM(COALESCE(emp.source_keypoints_run, '')) != ''
              AND NOT EXISTS (
                    SELECT 1
                    FROM keypoint_performance kp
                    WHERE kp.dataset_id = emp.dataset_id
                      AND kp.keypoint_run = emp.source_keypoints_run
                )
              AND NOT EXISTS (
                    SELECT 1
                    FROM keypoint_quality kq
                    WHERE kq.dataset_id = emp.dataset_id
                      AND kq.refined_run = emp.source_keypoints_run
                )
            ORDER BY emp.dataset_id, emp.stage_group, emp.run_name;
            """
        ).fetchall()
        for row in missing_eye_keypoint_rows:
            dataset_id = str(row["dataset_id"] or "")
            stage_group = str(row["stage_group"] or "")
            run_name = str(row["run_name"] or "")
            source_keypoints_run = str(row["source_keypoints_run"] or "")
            issues.append(
                IntegrityIssue(
                    code="eye_mask_performance_missing_source_keypoint_projection",
                    run_id=f"{dataset_id}:{stage_group}:{run_name}" if dataset_id or run_name else None,
                    detail=(
                        f"dataset_id={dataset_id} stage_group={stage_group} run_name={run_name} "
                        f"source_keypoints_run={source_keypoints_run} has no matching keypoint projection row"
                    ),
                )
            )

    if _relation_queryable("table", "eye_mask_performance"):
        missing_eye_source_rows = registry.conn.execute(
            """
            SELECT emp.dataset_id, emp.run_name, emp.source_eye_masks_run
            FROM eye_mask_performance emp
            WHERE emp.stage_group = 'refined_eye_masks_runs'
              AND TRIM(COALESCE(emp.source_eye_masks_run, '')) != ''
              AND NOT EXISTS (
                    SELECT 1
                    FROM eye_mask_performance src
                    WHERE src.dataset_id = emp.dataset_id
                      AND src.stage_group = 'eye_masks_runs'
                      AND src.run_name = emp.source_eye_masks_run
                )
            ORDER BY emp.dataset_id, emp.run_name;
            """
        ).fetchall()
        for row in missing_eye_source_rows:
            dataset_id = str(row["dataset_id"] or "")
            run_name = str(row["run_name"] or "")
            source_eye_masks_run = str(row["source_eye_masks_run"] or "")
            issues.append(
                IntegrityIssue(
                    code="eye_mask_performance_missing_source_eye_mask_projection",
                    run_id=f"{dataset_id}:refined_eye_masks_runs:{run_name}" if dataset_id or run_name else None,
                    detail=(
                        f"dataset_id={dataset_id} run_name={run_name} "
                        f"source_eye_masks_run={source_eye_masks_run} has no matching eye_mask_performance row"
                    ),
                )
            )

    if (
        _relation_queryable("table", "subject_mask_performance")
        and _relation_queryable("table", "keypoint_performance")
        and _relation_queryable("table", "keypoint_quality")
    ):
        missing_subject_keypoint_rows = registry.conn.execute(
            """
            SELECT smp.dataset_id, smp.stage_group, smp.run_name, smp.source_keypoints_run
            FROM subject_mask_performance smp
            WHERE TRIM(COALESCE(smp.source_keypoints_run, '')) != ''
              AND NOT EXISTS (
                    SELECT 1
                    FROM keypoint_performance kp
                    WHERE kp.dataset_id = smp.dataset_id
                      AND kp.keypoint_run = smp.source_keypoints_run
                )
              AND NOT EXISTS (
                    SELECT 1
                    FROM keypoint_quality kq
                    WHERE kq.dataset_id = smp.dataset_id
                      AND kq.refined_run = smp.source_keypoints_run
                )
            ORDER BY smp.dataset_id, smp.stage_group, smp.run_name;
            """
        ).fetchall()
        for row in missing_subject_keypoint_rows:
            dataset_id = str(row["dataset_id"] or "")
            stage_group = str(row["stage_group"] or "")
            run_name = str(row["run_name"] or "")
            source_keypoints_run = str(row["source_keypoints_run"] or "")
            issues.append(
                IntegrityIssue(
                    code="subject_mask_performance_missing_source_keypoint_projection",
                    run_id=f"{dataset_id}:{stage_group}:{run_name}" if dataset_id or run_name else None,
                    detail=(
                        f"dataset_id={dataset_id} stage_group={stage_group} run_name={run_name} "
                        f"source_keypoints_run={source_keypoints_run} has no matching keypoint projection row"
                    ),
                )
            )

    if _relation_queryable("table", "subject_mask_performance"):
        missing_subject_source_rows = registry.conn.execute(
            """
            SELECT smp.dataset_id, smp.run_name, smp.source_subject_mask_run
            FROM subject_mask_performance smp
            WHERE smp.stage_group = 'refined_subject_masks_runs'
              AND TRIM(COALESCE(smp.source_subject_mask_run, '')) != ''
              AND NOT EXISTS (
                    SELECT 1
                    FROM subject_mask_performance src
                    WHERE src.dataset_id = smp.dataset_id
                      AND src.stage_group = 'subject_mask_runs'
                      AND src.run_name = smp.source_subject_mask_run
                )
            ORDER BY smp.dataset_id, smp.run_name;
            """
        ).fetchall()
        for row in missing_subject_source_rows:
            dataset_id = str(row["dataset_id"] or "")
            run_name = str(row["run_name"] or "")
            source_subject_mask_run = str(row["source_subject_mask_run"] or "")
            issues.append(
                IntegrityIssue(
                    code="subject_mask_performance_missing_source_subject_mask_projection",
                    run_id=f"{dataset_id}:refined_subject_masks_runs:{run_name}" if dataset_id or run_name else None,
                    detail=(
                        f"dataset_id={dataset_id} run_name={run_name} "
                        f"source_subject_mask_run={source_subject_mask_run} has no matching subject_mask_performance row"
                    ),
                )
            )

    # Derived datasets should have recording linkage consistent with lineage parents.
    derived_rows = registry.conn.execute(
        """
        SELECT d.dataset_id, d.recording_id
        FROM datasets d
        WHERE COALESCE(d.artifact_kind, '') <> 'source_recording'
        ORDER BY d.dataset_id;
        """
    ).fetchall()
    for row in derived_rows:
        dataset_id = str(row["dataset_id"])
        child_recording_id = str(row["recording_id"]).strip() if row["recording_id"] is not None else ""
        parent_rows = registry.conn.execute(
            """
            SELECT DISTINCT pd.recording_id
            FROM dataset_lineage_current dl
            JOIN datasets pd ON pd.dataset_id = dl.parent_dataset_id
            WHERE dl.child_dataset_id = ?;
            """,
            (dataset_id,),
        ).fetchall()
        parent_recording_ids = sorted(
            {
                str(item["recording_id"]).strip()
                for item in parent_rows
                if item["recording_id"] is not None and str(item["recording_id"]).strip()
            }
        )
        if not parent_recording_ids:
            continue
        if len(parent_recording_ids) == 1:
            expected = parent_recording_ids[0]
            if not child_recording_id:
                issues.append(
                    IntegrityIssue(
                        code="derived_dataset_missing_recording_id_single_parent",
                        run_id=dataset_id,
                        detail=(
                            f"dataset_id={dataset_id} has one parent recording_id={expected} "
                            "but child recording_id is NULL/empty"
                        ),
                    )
                )
            elif child_recording_id != expected:
                issues.append(
                    IntegrityIssue(
                        code="derived_dataset_recording_id_mismatch_single_parent",
                        run_id=dataset_id,
                        detail=(
                            f"dataset_id={dataset_id} child recording_id={child_recording_id} "
                            f"expected={expected}"
                        ),
                    )
                )
        else:
            if child_recording_id:
                issues.append(
                    IntegrityIssue(
                        code="derived_dataset_recording_id_non_null_multi_parent",
                        run_id=dataset_id,
                        detail=(
                            f"dataset_id={dataset_id} has multiple parent recording_ids="
                            f"{','.join(parent_recording_ids)} but child recording_id={child_recording_id}"
                        ),
                    )
                )

    # Recordings should declare type/schema for downstream validation.
    recording_rows = registry.conn.execute(
        """
        SELECT recording_id, recording_type, recording_subtype, behavior_mode, artifact_schema_id
        FROM recordings
        ORDER BY recording_id;
        """
    ).fetchall()
    for row in recording_rows:
        recording_id = str(row["recording_id"])
        recording_type = (str(row["recording_type"]).strip() if row["recording_type"] is not None else "")
        recording_subtype = (
            str(row["recording_subtype"]).strip() if row["recording_subtype"] is not None else ""
        )
        behavior_mode = (str(row["behavior_mode"]).strip() if row["behavior_mode"] is not None else "")
        artifact_schema_id = (
            str(row["artifact_schema_id"]).strip() if row["artifact_schema_id"] is not None else ""
        )
        if not recording_type:
            issues.append(
                IntegrityIssue(
                    code="recording_missing_type",
                    run_id=recording_id,
                    detail=f"recording_id={recording_id} has NULL/empty recording_type",
                )
            )
        elif recording_type not in allowed_recording_types:
            issues.append(
                IntegrityIssue(
                    code="recording_invalid_type",
                    run_id=recording_id,
                    detail=(
                        f"recording_id={recording_id} recording_type={recording_type} "
                        f"not in allowed={','.join(sorted(allowed_recording_types))}"
                    ),
                )
            )
        allowed_subtypes = allowed_subtypes_by_type.get(recording_type)
        if allowed_subtypes is not None:
            if not recording_subtype:
                issues.append(
                    IntegrityIssue(
                        code="recording_missing_subtype",
                        run_id=recording_id,
                        detail=(
                            f"recording_id={recording_id} recording_type={recording_type} "
                            "has NULL/empty recording_subtype"
                        ),
                    )
                )
            elif recording_subtype not in allowed_subtypes:
                issues.append(
                    IntegrityIssue(
                        code="recording_invalid_subtype",
                        run_id=recording_id,
                        detail=(
                            f"recording_id={recording_id} recording_type={recording_type} "
                            f"recording_subtype={recording_subtype} "
                            f"not in allowed={','.join(sorted(allowed_subtypes))}"
                        ),
                    )
                )
        if not behavior_mode:
            issues.append(
                IntegrityIssue(
                    code="recording_missing_behavior_mode",
                    run_id=recording_id,
                    detail=f"recording_id={recording_id} has NULL/empty behavior_mode",
                )
            )
        elif behavior_mode not in ALLOWED_BEHAVIOR_MODES:
            issues.append(
                IntegrityIssue(
                    code="recording_invalid_behavior_mode",
                    run_id=recording_id,
                    detail=(
                        f"recording_id={recording_id} behavior_mode={behavior_mode} "
                        f"not in allowed={','.join(sorted(ALLOWED_BEHAVIOR_MODES))}"
                    ),
                )
            )
        if recording_type == "behavior" and recording_subtype and behavior_mode:
            if recording_subtype != behavior_mode:
                issues.append(
                    IntegrityIssue(
                        code="recording_behavior_mode_mismatch",
                        run_id=recording_id,
                        detail=(
                            f"recording_id={recording_id} recording_type=behavior "
                            f"requires recording_subtype==behavior_mode, got "
                            f"{recording_subtype}!={behavior_mode}"
                        ),
                    )
                )
        if not artifact_schema_id:
            issues.append(
                IntegrityIssue(
                    code="recording_missing_artifact_schema",
                    run_id=recording_id,
                    detail=f"recording_id={recording_id} has NULL/empty artifact_schema_id",
                )
            )
            continue

        # behavior_v1: validate required artifact types are present.
        if artifact_schema_id == "behavior_v1":
            artifact_rows = registry.conn.execute(
                """
                SELECT artifact_type, status
                FROM recording_artifacts
                WHERE recording_id = ?;
                """,
                (recording_id,),
            ).fetchall()
            present_types = {
                str(item["artifact_type"])
                for item in artifact_rows
                if str(item["status"] or "").lower() != "missing"
            }
            missing_types = sorted(behavior_v1_required_artifacts - present_types)
            if missing_types:
                issues.append(
                    IntegrityIssue(
                        code="recording_artifact_schema_missing_required",
                        run_id=recording_id,
                        detail=(
                            f"recording_id={recording_id} artifact_schema_id=behavior_v1 "
                            f"missing={','.join(missing_types)}"
                        ),
                    )
                )

    return issues


def _summarize_dataset_lineage_audit(registry: Registry) -> DatasetLineageAuditSummary:
    edge_count = int(
        registry.conn.execute(
            "SELECT COUNT(*) FROM dataset_lineage_current;"
        ).fetchone()[0]
    )
    merged_dataset_count = int(
        registry.conn.execute(
            """
            SELECT COUNT(*)
            FROM datasets
            WHERE artifact_kind = 'derived_training_merge' OR dataset_id LIKE '%_merged';
            """
        ).fetchone()[0]
    )
    merged_missing_lineage_count = int(
        registry.conn.execute(
            """
            SELECT COUNT(*)
            FROM datasets d
            WHERE
                (d.artifact_kind = 'derived_training_merge' OR d.dataset_id LIKE '%_merged')
                AND NOT EXISTS (
                    SELECT 1
                    FROM dataset_lineage_current dl
                    WHERE dl.child_dataset_id = d.dataset_id
                      AND dl.relationship_type = 'training_merge_source'
                );
            """
        ).fetchone()[0]
    )
    training_set_lineage_mismatch_count = 0
    set_rows = registry.conn.execute(
        """
        SELECT set_id, dataset_ids_json
        FROM training_sets;
        """
    ).fetchall()
    dataset_rows = registry.conn.execute(
        """
        SELECT dataset_id, artifact_kind
        FROM datasets;
        """
    ).fetchall()
    dataset_kind: Dict[str, str] = {
        str(row["dataset_id"]): str(row["artifact_kind"] or "")
        for row in dataset_rows
        if row["dataset_id"] is not None
    }
    for row in set_rows:
        dataset_ids = sorted(set(_json_text_list(row["dataset_ids_json"])))
        merged_ids = [
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id in dataset_kind
            and (
                dataset_kind.get(dataset_id) == "derived_training_merge"
                or dataset_id.endswith("_merged")
            )
        ]
        if not merged_ids:
            continue
        expected_parents = {
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id in dataset_kind and dataset_id not in merged_ids
        }
        for child_dataset_id in merged_ids:
            expected = {item for item in expected_parents if item != child_dataset_id}
            if not expected:
                continue
            actual_rows = registry.conn.execute(
                """
                SELECT parent_dataset_id
                FROM dataset_lineage_current
                WHERE child_dataset_id = ? AND relationship_type = 'training_merge_source';
                """,
                (child_dataset_id,),
            ).fetchall()
            actual = {
                str(item["parent_dataset_id"])
                for item in actual_rows
                if item["parent_dataset_id"] is not None
            }
            if actual != expected:
                training_set_lineage_mismatch_count += 1
    return DatasetLineageAuditSummary(
        edge_count=edge_count,
        merged_dataset_count=merged_dataset_count,
        merged_missing_lineage_count=merged_missing_lineage_count,
        training_set_lineage_mismatch_count=training_set_lineage_mismatch_count,
    )


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


def _print_stale_in_progress_run_candidates(
    candidates: Sequence[StaleInProgressRunCandidate],
    *,
    list_limit: int,
) -> None:
    if not candidates:
        print("No stale in_progress training runs found.")
        return

    print(f"Stale in_progress training runs: {len(candidates)}")
    limit = len(candidates) if list_limit == 0 else min(len(candidates), list_limit)
    for candidate in candidates[:limit]:
        set_id = candidate.set_id or "—"
        task_type = candidate.task_type or "unknown"
        run_status = candidate.run_status or "—"
        model_status = candidate.model_status or "—"
        created_utc = candidate.created_utc or "—"
        age_text = "—" if candidate.age_hours is None else f"{candidate.age_hours:.2f}h"
        print(
            " - {run_id} [set={set_id} task={task} age={age} created={created} "
            "run_status={run_status} model_status={model_status}]".format(
                run_id=candidate.run_id,
                set_id=set_id,
                task=task_type,
                age=age_text,
                created=created_utc,
                run_status=run_status,
                model_status=model_status,
            )
        )
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


def _run_reconcile_registry(
    registry: Registry,
    *,
    dry_run: bool,
    scope_paths: Optional[Sequence[Path]],
    list_limit: int,
) -> None:
    if dry_run:
        print("Dry run: reconcile step is simulated (no status fields are updated).")
        missing_candidates = _collect_missing_dataset_candidates(
            registry,
            scope_paths=scope_paths,
            include_missing_scan=True,
        )
    else:
        stats = registry.reconcile_missing_datasets(scope_paths=scope_paths or None)
        _summarize_reconcile(stats)
        missing_candidates = _collect_missing_dataset_candidates(
            registry,
            scope_paths=scope_paths,
            include_missing_scan=False,
        )

    _print_candidates(missing_candidates, list_limit=list_limit)
    dataset_ids = [candidate.dataset_id for candidate in missing_candidates]
    if dry_run:
        print(f"Dry run: would delete {len(dataset_ids)} missing dataset row(s).")
    else:
        deleted = _delete_dataset_ids(registry, dataset_ids, dry_run=False)
        print(f"Deleted {deleted} missing dataset row(s).")

    issues = _check_registry_integrity(registry)
    if not issues:
        print("Integrity check passed: no issues found.")
        return

    print(f"Integrity check failed: {len(issues)} issue(s) found.")
    limit = len(issues) if list_limit == 0 else min(len(issues), list_limit)
    for issue in issues[:limit]:
        run_id = issue.run_id or "—"
        print(f" - [{issue.code}] run={run_id} :: {issue.detail}")
    if limit < len(issues):
        print(
            f" ... {len(issues) - limit} more issues omitted "
            "(use --list-limit 0 to show all)."
        )
    raise SystemExit(2)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    if (
        not args.reconcile_registry
        and not args.prune_invalid
        and not args.prune_missing_datasets
        and not args.prune_failed_runs
        and not args.reconcile_in_progress_runs
        and not args.delete_run_id
        and not args.delete_set_id
        and not args.prune_empty_sets
        and not args.backfill_recording_entities
        and not args.backfill_subject_dish_cross
        and not args.backfill_subjects
        and not args.backfill_model_tables
        and not args.remap_training_set_dataset_ids
        and not args.backfill_dataset_lineage
        and not args.backfill_keypoint_profiles
        and not args.backfill_eye_mask_profiles
        and not args.backfill_keypoint_quality
        and not args.backfill_eye_mask_quality
        and not args.backfill_detect_quality
        and not args.backfill_detect_performance
        and not args.backfill_keypoint_performance
        and not args.backfill_crop_quality
        and not args.backfill_eye_mask_performance
        and not args.backfill_subject_mask_performance
        and not args.backfill_subject_mask_component_quality
        and not args.backfill_recording_step_status
        and not args.refresh_keypoint_profiles
        and not args.refresh_eye_mask_profiles
        and not args.refresh_keypoint_quality
        and not args.refresh_eye_mask_quality
        and not args.refresh_detect_quality
        and not args.refresh_detect_performance
        and not args.refresh_keypoint_performance
        and not args.refresh_crop_quality
        and not args.refresh_eye_mask_performance
        and not args.refresh_subject_mask_performance
        and not args.refresh_subject_mask_component_quality
        and not args.check_integrity
        and not args.vacuum
    ):
        raise SystemExit(
            "No action selected. Use --reconcile-registry, --prune-invalid, --prune-missing-datasets, "
            "--prune-failed-runs, --reconcile-in-progress-runs, --delete-run-id, --delete-set-id, "
            "--prune-empty-sets, --backfill-recording-entities, --backfill-subject-dish-cross, "
            "--backfill-subjects, "
            "--backfill-model-tables, --backfill-keypoint-profiles, --backfill-keypoint-quality, "
            "--backfill-eye-mask-quality, "
            "--backfill-eye-mask-profiles, "
            "--remap-training-set-dataset-ids, "
            "--backfill-dataset-lineage, "
            "--backfill-detect-quality, --backfill-detect-performance, --backfill-keypoint-performance, "
            "--backfill-crop-quality, --backfill-eye-mask-performance, "
            "--backfill-subject-mask-performance, --backfill-subject-mask-component-quality, "
            "--backfill-recording-step-status, "
            "--refresh-keypoint-profiles, --refresh-keypoint-quality, "
            "--refresh-eye-mask-quality, "
            "--refresh-eye-mask-profiles, "
            "--refresh-detect-quality, --refresh-detect-performance, "
            "--refresh-keypoint-performance, "
            "--refresh-crop-quality, --refresh-eye-mask-performance, "
            "--refresh-subject-mask-performance, --refresh-subject-mask-component-quality, "
            "--check-integrity, and/or --vacuum."
        )

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    scope_paths = [Path(path).expanduser() for path in args.paths]

    registry = Registry(registry_path)
    try:
        if args.reconcile_registry:
            _run_reconcile_registry(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                list_limit=int(args.list_limit),
            )

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

        if args.prune_missing_datasets:
            if args.dry_run:
                print("Dry run: reconcile step is simulated (no status fields are updated).")
                candidates = _collect_missing_dataset_candidates(
                    registry,
                    scope_paths=scope_paths or None,
                    include_missing_scan=True,
                )
            else:
                stats = registry.reconcile_missing_datasets(scope_paths=scope_paths or None)
                _summarize_reconcile(stats)
                candidates = _collect_missing_dataset_candidates(
                    registry,
                    scope_paths=scope_paths or None,
                    include_missing_scan=False,
                )
            _print_candidates(candidates, list_limit=args.list_limit)
            dataset_ids = [candidate.dataset_id for candidate in candidates]
            if args.dry_run:
                print(f"Dry run: would delete {len(dataset_ids)} missing dataset row(s).")
            else:
                deleted = _delete_dataset_ids(registry, dataset_ids, dry_run=False)
                print(f"Deleted {deleted} missing dataset row(s).")

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

        if args.reconcile_in_progress_runs:
            max_age_hours = float(args.in_progress_max_age_hours)
            if max_age_hours < 0:
                raise SystemExit("--in-progress-max-age-hours must be non-negative.")
            task_filter = str(args.in_progress_task or "all")
            stale_candidates = _collect_stale_in_progress_run_candidates(
                registry,
                max_age_hours=max_age_hours,
                task_filter=task_filter,
            )
            _print_stale_in_progress_run_candidates(
                stale_candidates,
                list_limit=args.list_limit,
            )
            if args.dry_run:
                print(
                    "Dry run: would mark "
                    f"{len(stale_candidates)} stale in_progress run(s) as failed "
                    f"(max_age_hours={max_age_hours:g}, task_filter={task_filter})."
                )
            else:
                reconciled = _reconcile_stale_in_progress_runs(
                    registry,
                    candidates=stale_candidates,
                    max_age_hours=max_age_hours,
                    task_filter=task_filter,
                    dry_run=False,
                )
                print(
                    "Reconciled "
                    f"{reconciled} stale in_progress run(s) to failed "
                    f"(max_age_hours={max_age_hours:g}, task_filter={task_filter})."
                )

        if args.delete_run_id:
            requested_run_ids = _normalize_run_ids(args.delete_run_id)
            existing_run_ids, missing_run_ids = _resolve_existing_run_ids(registry, requested_run_ids)
            file_plan: Optional[FileDeletePlan] = None
            if args.delete_files and existing_run_ids:
                artifact_roots = _resolve_training_artifact_roots()
                candidate_paths = _collect_run_artifact_paths(registry, existing_run_ids)
                file_plan = _build_file_delete_plan(candidate_paths, artifact_roots=artifact_roots)
            print(
                "Delete run-id request: "
                f"requested={len(requested_run_ids)} existing={len(existing_run_ids)} missing={len(missing_run_ids)}"
            )
            if existing_run_ids:
                limit = len(existing_run_ids) if args.list_limit == 0 else min(len(existing_run_ids), args.list_limit)
                print("Target run_ids:")
                for run_id in existing_run_ids[:limit]:
                    print(f" - {run_id}")
                if limit < len(existing_run_ids):
                    print(
                        f" ... {len(existing_run_ids) - limit} more run_id(s) omitted "
                        "(use --list-limit 0 to show all)."
                    )
            if missing_run_ids:
                limit = len(missing_run_ids) if args.list_limit == 0 else min(len(missing_run_ids), args.list_limit)
                print("Missing run_ids (no-op):")
                for run_id in missing_run_ids[:limit]:
                    print(f" - {run_id}")
                if limit < len(missing_run_ids):
                    print(
                        f" ... {len(missing_run_ids) - limit} more run_id(s) omitted "
                        "(use --list-limit 0 to show all)."
                    )
            if args.dry_run:
                print(f"Dry run: would delete {len(existing_run_ids)} training run row(s) by explicit run_id.")
            else:
                deleted = _delete_training_run_ids(registry, existing_run_ids, dry_run=False)
                print(f"Deleted {deleted} training run row(s) by explicit run_id.")
            if args.delete_files and file_plan is not None:
                print(
                    "File delete plan (run targets): "
                    f"eligible={len(file_plan.eligible_paths)} "
                    f"existing={len(file_plan.existing_paths)} "
                    f"bytes={file_plan.existing_bytes}"
                )
                if file_plan.skipped_paths:
                    limit = len(file_plan.skipped_paths) if args.list_limit == 0 else min(len(file_plan.skipped_paths), args.list_limit)
                    print("Skipped unsafe paths:")
                    for path, reason in file_plan.skipped_paths[:limit]:
                        print(f" - {path} [{reason}]")
                    if limit < len(file_plan.skipped_paths):
                        print(
                            f" ... {len(file_plan.skipped_paths) - limit} more skipped path(s) omitted "
                            "(use --list-limit 0 to show all)."
                        )
                if args.dry_run:
                    print(f"Dry run: would delete {len(file_plan.existing_paths)} file/dir path(s) for run targets.")
                else:
                    deleted_paths = _delete_paths(file_plan.existing_paths, dry_run=False)
                    print(f"Deleted {deleted_paths} file/dir path(s) for run targets.")

        if args.delete_set_id:
            requested_set_ids = _normalize_set_ids(args.delete_set_id)
            candidates = _collect_set_delete_candidates(registry, requested_set_ids)
            existing = [candidate for candidate in candidates if candidate.exists]
            missing = [candidate for candidate in candidates if not candidate.exists]
            blocked = [candidate for candidate in existing if candidate.run_count > 0]
            safe_sets = [candidate.set_id for candidate in existing if candidate.run_count == 0]
            file_plan: Optional[FileDeletePlan] = None
            print(
                "Delete set-id request: "
                f"requested={len(requested_set_ids)} "
                f"existing={len(existing)} "
                f"missing={len(missing)} "
                f"blocked_with_runs={len(blocked)}"
            )
            if existing:
                limit = len(existing) if args.list_limit == 0 else min(len(existing), args.list_limit)
                print("Existing set_ids:")
                for candidate in existing[:limit]:
                    print(f" - {candidate.set_id} [linked_runs={candidate.run_count}]")
                if limit < len(existing):
                    print(
                        f" ... {len(existing) - limit} more set_id(s) omitted "
                        "(use --list-limit 0 to show all)."
                    )
            if missing:
                limit = len(missing) if args.list_limit == 0 else min(len(missing), args.list_limit)
                print("Missing set_ids (no-op):")
                for candidate in missing[:limit]:
                    print(f" - {candidate.set_id}")
                if limit < len(missing):
                    print(
                        f" ... {len(missing) - limit} more set_id(s) omitted "
                        "(use --list-limit 0 to show all)."
                    )

            if blocked and not args.delete_set_with_runs:
                blocked_set_text = ", ".join(candidate.set_id for candidate in blocked)
                raise SystemExit(
                    "Refusing --delete-set-id for set(s) with linked runs: "
                    f"{blocked_set_text}. "
                    "Re-run with --delete-set-with-runs to delete linked training runs first."
                )

            delete_run_ids: List[str] = []
            if args.delete_set_with_runs and blocked:
                delete_run_ids = _collect_run_ids_for_set_ids(registry, [candidate.set_id for candidate in blocked])
                print(
                    "Linked runs targeted due to --delete-set-with-runs: "
                    f"{len(delete_run_ids)}"
                )
                if delete_run_ids:
                    limit = len(delete_run_ids) if args.list_limit == 0 else min(len(delete_run_ids), args.list_limit)
                    for run_id in delete_run_ids[:limit]:
                        print(f" - run {run_id}")
                    if limit < len(delete_run_ids):
                        print(
                            f" ... {len(delete_run_ids) - limit} more run_id(s) omitted "
                            "(use --list-limit 0 to show all)."
                        )

            delete_set_ids = safe_sets + [candidate.set_id for candidate in blocked]
            if args.delete_files and delete_set_ids:
                artifact_roots = _resolve_training_artifact_roots()
                candidate_paths = _collect_set_artifact_paths(delete_set_ids, artifact_roots)
                if delete_run_ids:
                    candidate_paths.extend(_collect_run_artifact_paths(registry, delete_run_ids))
                file_plan = _build_file_delete_plan(candidate_paths, artifact_roots=artifact_roots)
            if args.dry_run:
                if delete_run_ids:
                    print(
                        f"Dry run: would delete {len(delete_run_ids)} linked training run row(s) "
                        "for requested set_id(s)."
                    )
                print(f"Dry run: would delete {len(delete_set_ids)} training set row(s) by explicit set_id.")
            else:
                if delete_run_ids:
                    deleted_runs = _delete_training_run_ids(registry, delete_run_ids, dry_run=False)
                    print(f"Deleted {deleted_runs} linked training run row(s) for requested set_id(s).")
                deleted_sets = _delete_training_set_ids(registry, delete_set_ids, dry_run=False)
                print(f"Deleted {deleted_sets} training set row(s) by explicit set_id.")
            if args.delete_files and file_plan is not None:
                print(
                    "File delete plan (set targets): "
                    f"eligible={len(file_plan.eligible_paths)} "
                    f"existing={len(file_plan.existing_paths)} "
                    f"bytes={file_plan.existing_bytes}"
                )
                if file_plan.skipped_paths:
                    limit = len(file_plan.skipped_paths) if args.list_limit == 0 else min(len(file_plan.skipped_paths), args.list_limit)
                    print("Skipped unsafe paths:")
                    for path, reason in file_plan.skipped_paths[:limit]:
                        print(f" - {path} [{reason}]")
                    if limit < len(file_plan.skipped_paths):
                        print(
                            f" ... {len(file_plan.skipped_paths) - limit} more skipped path(s) omitted "
                            "(use --list-limit 0 to show all)."
                        )
                if args.dry_run:
                    print(f"Dry run: would delete {len(file_plan.existing_paths)} file/dir path(s) for set targets.")
                else:
                    deleted_paths = _delete_paths(file_plan.existing_paths, dry_run=False)
                    print(f"Deleted {deleted_paths} file/dir path(s) for set targets.")

        if args.prune_empty_sets:
            empty_set_candidates = _collect_empty_training_set_candidates(registry)
            _print_empty_training_set_candidates(empty_set_candidates, list_limit=args.list_limit)
            set_ids = [candidate.set_id for candidate in empty_set_candidates]
            if args.dry_run:
                print(f"Dry run: would delete {len(set_ids)} training set row(s) with no linked runs.")
            else:
                deleted = _delete_training_set_ids(registry, set_ids, dry_run=False)
                print(f"Deleted {deleted} training set row(s) with no linked runs.")

        if args.backfill_recording_entities:
            summary = _backfill_recording_entities(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
            )
            print(
                "Recording entities backfill: "
                f"scanned={summary['recordings_scanned']} "
                f"manifests_missing={summary['manifests_missing']} "
                f"recordings_upserted={summary['recordings_upserted']} "
                f"datasets_linked={summary['datasets_linked']} "
                f"artifacts_seen={summary['artifacts_seen']} "
                f"artifacts_upserted={summary['artifacts_upserted']} "
                f"derived_kind_backfilled={summary['derived_kind_backfilled']}"
            )

        if args.backfill_subject_dish_cross:
            summary = _backfill_subject_dish_cross_entities(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
            )
            print(
                "Subject/dish/cross backfill: "
                f"source_rows_scanned={summary['source_rows_scanned']} "
                f"crosses_seen={summary['crosses_seen']} "
                f"crosses_unique_seen={summary['crosses_unique_seen']} "
                f"crosses_would_insert={summary['crosses_would_insert']} "
                f"crosses_upserted={summary['crosses_upserted']} "
                f"dishes_seen={summary['dishes_seen']} "
                f"dishes_unique_seen={summary['dishes_unique_seen']} "
                f"dishes_would_insert={summary['dishes_would_insert']} "
                f"dishes_upserted={summary['dishes_upserted']} "
                f"recording_subject_rows_seen={summary['recording_subject_rows_seen']} "
                f"recording_subjects_unique_seen={summary['recording_subjects_unique_seen']} "
                f"recording_subjects_would_insert={summary['recording_subjects_would_insert']} "
                f"recording_subjects_upserted={summary['recording_subjects_upserted']} "
                f"rows_skipped_missing_recording_id={summary['rows_skipped_missing_recording_id']} "
                f"rows_skipped_missing_subject_id={summary['rows_skipped_missing_subject_id']}"
            )
            if args.dry_run:
                print("Dry run: no crosses/dishes/recording_subjects rows were updated.")

        if args.backfill_subjects:
            summary = _backfill_subjects(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
            )
            print(
                "Subjects backfill: "
                f"subject_rows_scanned={summary['subject_rows_scanned']} "
                f"rows_skipped_out_of_scope={summary['rows_skipped_out_of_scope']} "
                f"rows_skipped_missing_subject_id={summary['rows_skipped_missing_subject_id']} "
                f"subject_ids_unique_seen={summary['subject_ids_unique_seen']} "
                f"subjects_existing={summary['subjects_existing']} "
                f"subjects_would_insert={summary['subjects_would_insert']} "
                f"subjects_would_enrich={summary['subjects_would_enrich']} "
                f"subjects_upserted={summary['subjects_upserted']} "
                f"subjects_conflict_dish_id={summary['subjects_conflict_dish_id']} "
                f"subjects_conflict_species={summary['subjects_conflict_species']} "
                f"subjects_conflict_sex={summary['subjects_conflict_sex']}"
            )
            if args.dry_run:
                print("Dry run: no subjects rows were updated.")

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
                    f"{summary['detection_missing']} training_models row(s), "
                    f"{summary['onnx_missing']} onnx_models row(s), "
                    f"{summary['tensorrt_missing']} tensorrt_models row(s)."
                )
            else:
                print(
                    "Inserted "
                    f"{summary['detection_inserted']} training_models row(s), "
                    f"{summary['onnx_inserted']} onnx_models row(s), "
                    f"{summary['tensorrt_inserted']} tensorrt_models row(s)."
                )

        if args.remap_training_set_dataset_ids:
            summary = _remap_training_set_dataset_ids(registry, dry_run=bool(args.dry_run))
            print(
                "Training set dataset_id remap: "
                f"sets_scanned={summary['sets_scanned']} "
                f"sets_changed={summary['sets_changed']} "
                f"ids_remapped={summary['ids_remapped']} "
                f"ids_unresolved={summary['ids_unresolved']}"
            )
            if args.dry_run:
                print("Dry run: no training_sets rows were updated.")
            else:
                print(f"Applied updates to {summary['sets_changed']} training_set row(s).")

        if args.backfill_dataset_lineage:
            summary = _backfill_dataset_lineage(registry, dry_run=bool(args.dry_run))
            print(
                "Dataset lineage backfill: "
                f"sets_scanned={summary['sets_scanned']} "
                f"merged_scanned={summary['merged_scanned']} "
                f"relationships_changed={summary['relationships_changed']}"
            )
            if args.dry_run:
                print(
                    "Dry run: would apply "
                    f"inserted={summary['rows_inserted']} "
                    f"deleted={summary['rows_deleted']} "
                    f"unchanged={summary['rows_unchanged']} relationship(s)."
                )
            else:
                print(
                    "Applied "
                    f"inserted={summary['rows_inserted']} "
                    f"deleted={summary['rows_deleted']} "
                    f"unchanged={summary['rows_unchanged']} relationship(s)."
                )

        if args.backfill_keypoint_profiles or args.refresh_keypoint_profiles:
            summary = _backfill_keypoint_profiles(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_keypoint_profiles),
            )
            mode = "refresh" if args.refresh_keypoint_profiles else "backfill"
            print(
                f"Keypoint profiles {mode}: "
                "scope=source-recording-all-uses "
                f"scanned={summary['datasets_scanned']} "
                f"missing={summary['datasets_missing']} "
                f"errors={summary['datasets_errors']} "
                f"no_profile={summary['datasets_no_profile']} "
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

        if args.backfill_eye_mask_profiles or args.refresh_eye_mask_profiles:
            summary = _backfill_eye_mask_profiles(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_eye_mask_profiles),
            )
            mode = "refresh" if args.refresh_eye_mask_profiles else "backfill"
            print(
                f"Eye-mask profiles {mode}: "
                "scope=source-recording-all-uses "
                f"scanned={summary['datasets_scanned']} "
                f"missing={summary['datasets_missing']} "
                f"errors={summary['datasets_errors']} "
                f"no_profile={summary['datasets_no_profile']} "
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

        if args.backfill_eye_mask_quality or args.refresh_eye_mask_quality:
            summary = _backfill_eye_mask_quality(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_eye_mask_quality),
            )
            mode = "refresh" if args.refresh_eye_mask_quality else "backfill"
            print(
                f"Eye-mask quality {mode}: "
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

        if args.backfill_detect_performance or args.refresh_detect_performance:
            summary = _backfill_detect_performance(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_detect_performance),
                include_all_datasets=bool(args.detect_performance_all_datasets),
            )
            mode = "refresh" if args.refresh_detect_performance else "backfill"
            scope_label = "all-datasets" if args.detect_performance_all_datasets else "source-analysis-only"
            print(
                f"Detect performance {mode}: "
                f"scope={scope_label} "
                f"scanned={summary['datasets_scanned']} "
                f"missing={summary['datasets_missing']} "
                f"errors={summary['datasets_errors']} "
                f"no_performance={summary['datasets_no_performance']} "
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

        if args.backfill_keypoint_performance or args.refresh_keypoint_performance:
            summary = _backfill_keypoint_performance(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_keypoint_performance),
                include_all_datasets=bool(args.keypoint_performance_all_datasets),
            )
            mode = "refresh" if args.refresh_keypoint_performance else "backfill"
            scope_label = "all-datasets" if args.keypoint_performance_all_datasets else "source-analysis-only"
            print(
                f"Keypoint performance {mode}: "
                f"scope={scope_label} "
                f"scanned={summary['datasets_scanned']} "
                f"missing={summary['datasets_missing']} "
                f"errors={summary['datasets_errors']} "
                f"no_performance={summary['datasets_no_performance']} "
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

        if args.backfill_crop_quality or args.refresh_crop_quality:
            summary = _backfill_crop_quality(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_crop_quality),
                include_all_datasets=bool(args.crop_quality_all_datasets),
            )
            mode = "refresh" if args.refresh_crop_quality else "backfill"
            scope_label = "all-datasets" if args.crop_quality_all_datasets else "source-analysis-only"
            print(
                f"Crop quality {mode}: "
                f"scope={scope_label} "
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

        if args.backfill_eye_mask_performance or args.refresh_eye_mask_performance:
            summary = _backfill_eye_mask_performance(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_eye_mask_performance),
                include_all_datasets=bool(args.eye_mask_performance_all_datasets),
            )
            mode = "refresh" if args.refresh_eye_mask_performance else "backfill"
            scope_label = "all-datasets" if args.eye_mask_performance_all_datasets else "source-analysis-only"
            print(
                f"Eye-mask performance {mode}: "
                f"scope={scope_label} "
                f"scanned={summary['datasets_scanned']} "
                f"missing={summary['datasets_missing']} "
                f"errors={summary['datasets_errors']} "
                f"no_performance={summary['datasets_no_performance']} "
                f"stale={summary['rows_stale']} "
                f"in_progress={summary['rows_in_progress']} "
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

        if args.backfill_subject_mask_performance or args.refresh_subject_mask_performance:
            summary = _backfill_subject_mask_performance(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_subject_mask_performance),
            )
            mode = "refresh" if args.refresh_subject_mask_performance else "backfill"
            print(
                f"Subject-mask performance {mode}: "
                "scope=source-recording-all-uses "
                f"scanned={summary['datasets_scanned']} "
                f"missing={summary['datasets_missing']} "
                f"errors={summary['datasets_errors']} "
                f"no_performance={summary['datasets_no_performance']} "
                f"stale={summary['rows_stale']} "
                f"in_progress={summary['rows_in_progress']} "
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

        if args.backfill_subject_mask_component_quality or args.refresh_subject_mask_component_quality:
            summary = _backfill_subject_mask_component_quality(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                refresh=bool(args.refresh_subject_mask_component_quality),
            )
            mode = "refresh" if args.refresh_subject_mask_component_quality else "backfill"
            print(
                f"Subject-mask component quality {mode}: "
                "scope=source-recording-all-uses "
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

        if args.backfill_recording_step_status:
            recording_id_filter = _normalize_recording_ids(args.recording_step_recording_id)
            summary = _backfill_recording_step_status(
                registry,
                dry_run=bool(args.dry_run),
                scope_paths=scope_paths or None,
                recording_ids=recording_id_filter or None,
                zarr_use_filter=str(args.recording_step_zarr_use or "all"),
            )
            rows_by_status = summary.get("rows_by_status", {})
            status_counts = ""
            if isinstance(rows_by_status, dict):
                status_counts = " ".join(
                    f"{status_name}={int(rows_by_status.get(status_name, 0))}"
                    for status_name in RECORDING_STEP_STATUS_VALUES
                )
            print(
                "Recording step status backfill: "
                f"scanned={int(summary.get('datasets_scanned', 0))} "
                f"in_scope={int(summary.get('datasets_in_scope', 0))} "
                f"missing_zarr={int(summary.get('datasets_missing_zarr', 0))} "
                f"errors={int(summary.get('datasets_errors', 0))} "
                f"inserted={int(summary.get('rows_inserted', 0))} "
                f"updated={int(summary.get('rows_updated', 0))} "
                f"unchanged={int(summary.get('rows_skipped', 0))} "
                f"history_rows={int(summary.get('history_rows_inserted', 0))}"
            )
            if status_counts:
                print(f"Recording step status counts: {status_counts}")
            print(f"Recording step status summary JSON: {json.dumps(summary, sort_keys=True)}")

        if args.check_integrity:
            lineage_summary = _summarize_dataset_lineage_audit(registry)
            print(
                "Dataset lineage audit: "
                f"edges={lineage_summary.edge_count} "
                f"merged_datasets={lineage_summary.merged_dataset_count} "
                f"merged_missing_lineage={lineage_summary.merged_missing_lineage_count} "
                f"set_lineage_mismatch={lineage_summary.training_set_lineage_mismatch_count}"
            )
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
