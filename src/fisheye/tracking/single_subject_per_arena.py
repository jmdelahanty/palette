from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console

from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.type_conversions import normalize_attr
from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info


TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA = "single_subject_per_arena"
UNASSIGNED_TRACK_ID = -1
TRACKING_QC_OK = "ok"
TRACKING_QC_WARN = "warn"
TRACKING_WARN_THRESHOLD_ROWS = 1
TRACKING_WARN_THRESHOLD_PERCENT = 0.0
TRACKING_BLOCK_THRESHOLD_ROWS = 10
TRACKING_BLOCK_THRESHOLD_PERCENT = 1.0


class TrackingConflictError(ValueError):
    """Raised when one-subject-per-arena assumptions are violated."""


@dataclass(frozen=True)
class SingleSubjectPerArenaTrackingResult:
    track_ids: np.ndarray
    arena_ids: np.ndarray
    frame_indices: np.ndarray
    source_row_indices: np.ndarray
    track_ids_present: np.ndarray
    track_arena_ids: np.ndarray
    n_unassigned_rows: int


def _sorted_group_keys(group: Optional[zarr.Group]) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    return sorted(key for key in keys if isinstance(key, str))


def _normalize_text(value: object) -> Optional[str]:
    normalized = normalize_attr(value)
    if isinstance(normalized, str) and normalized:
        return normalized
    return None


def derive_tracking_qc_state(
    *,
    n_unassigned_rows: int,
    unassigned_row_rate_percent: Optional[float],
    warn_threshold_rows: int = TRACKING_WARN_THRESHOLD_ROWS,
    warn_threshold_percent: float = TRACKING_WARN_THRESHOLD_PERCENT,
    block_threshold_rows: int = TRACKING_BLOCK_THRESHOLD_ROWS,
    block_threshold_percent: float = TRACKING_BLOCK_THRESHOLD_PERCENT,
) -> str:
    """Derive the structured QA state for tracking completeness.

    Current runtime policy is intentionally non-blocking. Threshold metadata is
    still recorded for future rollout, but any nonzero unassigned count maps to
    `warn` rather than `block`.
    """

    if int(n_unassigned_rows) <= 0:
        return TRACKING_QC_OK

    rate = float(unassigned_row_rate_percent) if unassigned_row_rate_percent is not None else None
    if int(n_unassigned_rows) >= int(warn_threshold_rows):
        return TRACKING_QC_WARN
    if rate is not None and rate > float(warn_threshold_percent):
        return TRACKING_QC_WARN
    return TRACKING_QC_OK


def build_tracking_qc_fields(
    *,
    n_unassigned_rows: int,
    unassigned_row_rate_percent: Optional[float],
    warn_threshold_rows: int = TRACKING_WARN_THRESHOLD_ROWS,
    warn_threshold_percent: float = TRACKING_WARN_THRESHOLD_PERCENT,
    block_threshold_rows: int = TRACKING_BLOCK_THRESHOLD_ROWS,
    block_threshold_percent: float = TRACKING_BLOCK_THRESHOLD_PERCENT,
) -> Dict[str, object]:
    """Build the canonical structured QA metadata for tracking completeness."""

    return {
        "tracking_qc_state": derive_tracking_qc_state(
            n_unassigned_rows=n_unassigned_rows,
            unassigned_row_rate_percent=unassigned_row_rate_percent,
            warn_threshold_rows=warn_threshold_rows,
            warn_threshold_percent=warn_threshold_percent,
            block_threshold_rows=block_threshold_rows,
            block_threshold_percent=block_threshold_percent,
        ),
        "tracking_warn_threshold_rows": int(warn_threshold_rows),
        "tracking_warn_threshold_percent": float(warn_threshold_percent),
        "tracking_block_threshold_rows": int(block_threshold_rows),
        "tracking_block_threshold_percent": float(block_threshold_percent),
    }


def build_single_subject_per_arena_tracking(
    arena_ids: np.ndarray,
    frame_indices: np.ndarray,
    *,
    conflict_policy: str = "fail",
) -> SingleSubjectPerArenaTrackingResult:
    """Build deterministic run-local track IDs from occupied arena IDs."""

    arena_ids = np.asarray(arena_ids, dtype=np.int32)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)

    if arena_ids.ndim != 1 or frame_indices.ndim != 1:
        raise ValueError("arena_ids and frame_indices must be 1-D arrays.")
    if arena_ids.shape[0] != frame_indices.shape[0]:
        raise ValueError("arena_ids and frame_indices must have the same length.")

    if conflict_policy != "fail":
        raise ValueError(
            f"Unsupported conflict_policy '{conflict_policy}'. Only 'fail' is currently implemented."
        )

    assigned_mask = arena_ids >= 0
    if np.any(assigned_mask):
        assigned_rows = np.flatnonzero(assigned_mask)
        assigned_frames = frame_indices[assigned_mask]
        assigned_arenas = arena_ids[assigned_mask]

        order = np.lexsort((assigned_arenas, assigned_frames))
        sorted_frames = assigned_frames[order]
        sorted_arenas = assigned_arenas[order]

        duplicate_pairs = (
            (sorted_frames[1:] == sorted_frames[:-1])
            & (sorted_arenas[1:] == sorted_arenas[:-1])
        )
        if np.any(duplicate_pairs):
            first_conflict_idx = int(np.flatnonzero(duplicate_pairs)[0])
            conflict_frame = int(sorted_frames[first_conflict_idx])
            conflict_arena = int(sorted_arenas[first_conflict_idx])
            conflict_rows = assigned_rows[order[np.array([first_conflict_idx, first_conflict_idx + 1])]]
            raise TrackingConflictError(
                "single_subject_per_arena conflict: multiple detections in "
                f"frame {conflict_frame} for arena_id {conflict_arena} "
                f"(row indices {conflict_rows.tolist()})."
            )

    occupied_arena_ids = np.unique(arena_ids[assigned_mask]).astype(np.int32, copy=False)
    occupied_arena_ids.sort()

    track_ids = np.full(arena_ids.shape[0], UNASSIGNED_TRACK_ID, dtype=np.int32)
    for track_id, arena_id in enumerate(occupied_arena_ids):
        track_ids[arena_ids == arena_id] = np.int32(track_id)

    track_ids_present = np.arange(occupied_arena_ids.shape[0], dtype=np.int32)

    return SingleSubjectPerArenaTrackingResult(
        track_ids=track_ids,
        arena_ids=arena_ids.copy(),
        frame_indices=frame_indices.astype(np.int32, copy=False),
        source_row_indices=np.arange(arena_ids.shape[0], dtype=np.int32),
        track_ids_present=track_ids_present,
        track_arena_ids=occupied_arena_ids,
        n_unassigned_rows=int(np.sum(arena_ids < 0)),
    )


def write_single_subject_per_arena_tracking_run(
    *,
    root: zarr.Group,
    arena_ids: np.ndarray,
    frame_indices: np.ndarray,
    source_detect_run: str,
    source_arena_assignment_run: str,
    source_rowset_path: str,
    source_refined_run: Optional[str] = None,
    conflict_policy: str = "fail",
    console: Optional[Console] = None,
) -> Tuple[str, zarr.Group, Dict[str, object]]:
    """Write a tracking_runs entry for single-subject-per-arena archives."""

    result = build_single_subject_per_arena_tracking(
        arena_ids,
        frame_indices,
        conflict_policy=conflict_policy,
    )

    run_group, run_name = get_run_group(root, "tracking", console)

    n_rows = int(result.track_ids.shape[0])
    n_tracks = int(result.track_ids_present.shape[0])
    row_chunks = (min(4096, n_rows),) if n_rows else (1,)
    track_chunks = (min(1024, n_tracks),) if n_tracks else (1,)

    run_group.create_array("track_ids", data=result.track_ids, chunks=row_chunks, overwrite=True)
    run_group.create_array("arena_ids", data=result.arena_ids, chunks=row_chunks, overwrite=True)
    run_group.create_array("frame_indices", data=result.frame_indices, chunks=row_chunks, overwrite=True)
    run_group.create_array(
        "source_row_indices",
        data=result.source_row_indices,
        chunks=row_chunks,
        overwrite=True,
    )
    run_group.create_array(
        "track_ids_present",
        data=result.track_ids_present,
        chunks=track_chunks,
        overwrite=True,
    )
    run_group.create_array(
        "track_arena_ids",
        data=result.track_arena_ids,
        chunks=track_chunks,
        overwrite=True,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    n_unassigned_rows = int(result.n_unassigned_rows)
    n_assigned_rows = int(n_rows - n_unassigned_rows)
    unassigned_row_rate_percent = (
        float(n_unassigned_rows) / float(n_rows) * 100.0
        if n_rows > 0
        else 0.0
    )
    tracking_qc_fields = build_tracking_qc_fields(
        n_unassigned_rows=n_unassigned_rows,
        unassigned_row_rate_percent=unassigned_row_rate_percent,
    )
    summary_statistics: Dict[str, object] = {
        "n_rows": n_rows,
        "n_tracks": n_tracks,
        "n_assigned_rows": n_assigned_rows,
        "n_unassigned_rows": n_unassigned_rows,
        "unassigned_row_rate_percent": unassigned_row_rate_percent,
        **tracking_qc_fields,
        "track_ids_present": [int(value) for value in result.track_ids_present.tolist()],
        "track_arena_ids": [int(value) for value in result.track_arena_ids.tolist()],
    }

    env_info = get_environment_info()
    provenance_inputs: Dict[str, object] = {
        "source_detect_run": source_detect_run,
        "source_arena_assignment_run": source_arena_assignment_run,
        "source_rowset_path": source_rowset_path,
    }
    if source_refined_run:
        provenance_inputs["source_refined_run"] = source_refined_run

    provenance = build_stage_provenance(
        stage="tracks",
        command="single_subject_per_arena",
        created_at_utc=created_at,
        version=env_info["git"].get("short_hash") or env_info["git"].get("commit_hash"),
        git={
            "commit": env_info["git"].get("commit_hash"),
            "short": env_info["git"].get("short_hash"),
            "branch": env_info["git"].get("branch"),
            "is_dirty": env_info["git"].get("is_dirty"),
            "remote": env_info["git"].get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": env_info["platform"].get("hostname"),
            "system": env_info["platform"].get("system"),
            "release": env_info["platform"].get("release"),
            "python_version": env_info["platform"].get("python_version"),
            "machine": env_info["platform"].get("machine"),
        },
        parameters={
            "tracking_method": TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
            "conflict_policy": conflict_policy,
            "unassigned_track_id": UNASSIGNED_TRACK_ID,
        },
        inputs=provenance_inputs,
    )

    run_group.attrs.update(
        {
            "method": TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
            "tracking_method": TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
            "created_at_utc": created_at,
            "source_detect_run": source_detect_run,
            "source_arena_assignment_run": source_arena_assignment_run,
            "source_rowset_path": source_rowset_path,
            "track_namespace": "local_per_run",
            "unassigned_track_id": UNASSIGNED_TRACK_ID,
            "conflict_policy": conflict_policy,
            "num_tracks": n_tracks,
            "n_assigned_rows": n_assigned_rows,
            "n_unassigned_rows": n_unassigned_rows,
            "unassigned_row_rate_percent": unassigned_row_rate_percent,
            **tracking_qc_fields,
            "summary_statistics": summary_statistics,
        }
    )
    if source_refined_run:
        run_group.attrs["source_refined_run"] = source_refined_run

    write_stage_provenance(run_group, provenance)

    return run_name, run_group, summary_statistics


def resolve_tracking_run(
    root: zarr.Group,
    *,
    expected_detect_run: str,
    expected_refined_run: Optional[str] = None,
    expected_arena_assignment_run: Optional[str] = None,
    expected_source_rowset_path: Optional[str] = None,
) -> Tuple[str, zarr.Group]:
    """Resolve the exact tracking run for a detection/refined-detection lineage."""

    tracking_parent = root.get("tracking_runs")
    if tracking_parent is None:
        raise ValueError(
            "No tracking_runs found in archive. Run fisheye.tracking.arena_assignment to create single-subject tracking."
        )

    latest = _normalize_text(tracking_parent.attrs.get("latest")) if hasattr(tracking_parent, "attrs") else None
    matches: List[str] = []

    for run_name in _sorted_group_keys(tracking_parent):
        run_group = tracking_parent[run_name]
        source_detect = _normalize_text(run_group.attrs.get("source_detect_run")) if hasattr(run_group, "attrs") else None
        source_refined = _normalize_text(run_group.attrs.get("source_refined_run")) if hasattr(run_group, "attrs") else None
        source_arena_assignment = _normalize_text(run_group.attrs.get("source_arena_assignment_run")) if hasattr(run_group, "attrs") else None
        source_rowset_path = _normalize_text(run_group.attrs.get("source_rowset_path")) if hasattr(run_group, "attrs") else None

        if source_detect != expected_detect_run:
            continue
        if source_refined != expected_refined_run:
            continue
        if expected_arena_assignment_run is not None and source_arena_assignment != expected_arena_assignment_run:
            continue
        if expected_source_rowset_path is not None and source_rowset_path != expected_source_rowset_path:
            continue
        matches.append(run_name)

    if not matches:
        expected_refined_text = expected_refined_run or "<none>"
        expected_arena_text = expected_arena_assignment_run or "<any>"
        raise ValueError(
            "No tracking_runs entry matches "
            f"source_detect_run={expected_detect_run}, "
            f"source_refined_run={expected_refined_text}, "
            f"source_arena_assignment_run={expected_arena_text}. "
            f"source_rowset_path={expected_source_rowset_path or '<any>'}. "
            "Rerun fisheye.tracking.arena_assignment for this lineage."
        )

    chosen = latest if latest in matches else matches[-1]
    return chosen, tracking_parent[chosen]


def load_tracking_ids(
    root: zarr.Group,
    track_length: int,
    *,
    expected_detect_run: str,
    expected_refined_run: Optional[str] = None,
    expected_arena_assignment_run: Optional[str] = None,
    expected_source_rowset_path: Optional[str] = None,
    return_metadata: bool = False,
) -> Any:
    """Load source-bound track IDs for a detection rowset."""

    run_name, run_group = resolve_tracking_run(
        root,
        expected_detect_run=expected_detect_run,
        expected_refined_run=expected_refined_run,
        expected_arena_assignment_run=expected_arena_assignment_run,
        expected_source_rowset_path=expected_source_rowset_path,
    )

    if "track_ids" not in run_group:
        raise ValueError(f"Tracking run '{run_name}' is missing required array 'track_ids'.")

    track_ids = np.asarray(run_group["track_ids"][:], dtype=np.int32)
    if track_ids.shape[0] != track_length:
        raise ValueError(
            f"track_ids length ({track_ids.shape[0]}) does not match expected rows ({track_length})."
        )

    metadata: Dict[str, object] = {
        "track_run": run_name,
        "tracking_method": _normalize_text(run_group.attrs.get("tracking_method"))
        or _normalize_text(run_group.attrs.get("method")),
        "source_detect_run": _normalize_text(run_group.attrs.get("source_detect_run")),
        "source_refined_run": _normalize_text(run_group.attrs.get("source_refined_run")),
        "source_arena_assignment_run": _normalize_text(
            run_group.attrs.get("source_arena_assignment_run")
        ),
        "source_rowset_path": _normalize_text(run_group.attrs.get("source_rowset_path")),
        "conflict_policy": _normalize_text(run_group.attrs.get("conflict_policy")),
        "track_id_to_arena_id": {},
    }

    if "track_ids_present" in run_group and "track_arena_ids" in run_group:
        track_ids_present = np.asarray(run_group["track_ids_present"][:], dtype=np.int32)
        track_arena_ids = np.asarray(run_group["track_arena_ids"][:], dtype=np.int32)
        metadata["track_id_to_arena_id"] = {
            int(track_id): int(arena_id)
            for track_id, arena_id in zip(track_ids_present.tolist(), track_arena_ids.tolist())
        }

    if return_metadata:
        return track_ids, metadata
    return track_ids


__all__ = [
    "TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA",
    "UNASSIGNED_TRACK_ID",
    "TRACKING_QC_OK",
    "TRACKING_QC_WARN",
    "TRACKING_WARN_THRESHOLD_ROWS",
    "TRACKING_WARN_THRESHOLD_PERCENT",
    "TRACKING_BLOCK_THRESHOLD_ROWS",
    "TRACKING_BLOCK_THRESHOLD_PERCENT",
    "TrackingConflictError",
    "SingleSubjectPerArenaTrackingResult",
    "derive_tracking_qc_state",
    "build_tracking_qc_fields",
    "build_single_subject_per_arena_tracking",
    "write_single_subject_per_arena_tracking_run",
    "resolve_tracking_run",
    "load_tracking_ids",
]
