#!/usr/bin/env python3
"""Resolve a detection model from registry by recording metadata similarity."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fisheye.registry.db import Registry, RegistryPaths


FEATURE_WEIGHTS: dict[str, float] = {
    "recording_type": 1.0,
    "recording_subtype": 1.0,
    "behavior_mode": 1.0,
    "rig_id": 3.0,
    "arena_id": 3.0,
    "camera_id": 2.0,
    "canvas_name": 1.0,
    "protocol_name": 2.0,
    "dish_design": 2.0,
    "cross_id": 2.0,
    "genotype": 2.0,
    "dpf_at_acquisition": 1.0,
}


def _norm_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _norm_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


@dataclass
class TargetProfile:
    recording_id: str
    recording_type: Optional[str]
    recording_subtype: Optional[str]
    behavior_mode: Optional[str]
    rig_id: Optional[str]
    arena_id: Optional[str]
    camera_id: Optional[str]
    canvas_name: Optional[str]
    protocol_name: Optional[str]
    dish_design: Optional[str]
    cross_id: Optional[str]
    genotype: Optional[str]
    dpf_at_acquisition: Optional[int]


@dataclass
class Candidate:
    run_id: str
    set_id: str
    model_path: str
    created_utc: Optional[str]
    status: Optional[str]
    dataset_count: int
    weighted_score: float
    feature_match_counts: dict[str, int]
    feature_weights_used: float


def _normalize_task_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    alias = {
        "detect": "detect",
        "detection": "detect",
        "pose": "pose",
        "keypoint": "pose",
        "keypoints": "pose",
        "eye_mask": "eye_masks",
        "eye_masks": "eye_masks",
        "eye-mask": "eye_masks",
        "segmentation": "eye_masks",
    }
    return alias.get(text)


def _matches_task(*, run_id: str, set_id: str, model_path: str, task: str) -> bool:
    task_norm = str(task).strip().lower()
    if task_norm == "eye-mask":
        task_norm = "eye_masks"
    if task_norm == "any":
        return True

    if task_norm == "eye_masks":
        tokens = (
            "eye_masks",
            "eye_mask",
            "unet_eye_masks",
            "eye_seg",
            "segmentation",
        )
        run_id_l = run_id.lower()
        set_id_l = set_id.lower()
        model_path_l = model_path.lower()
        return any(token in run_id_l for token in tokens) or any(token in set_id_l for token in tokens) or any(
            token in model_path_l for token in tokens
        )

    run_token = f"_{task_norm}_"
    set_prefix = f"{task_norm}_"
    path_token = f"/{task_norm}/"

    run_id_l = run_id.lower()
    set_id_l = set_id.lower()
    model_path_l = model_path.lower()

    if set_id_l.startswith(set_prefix):
        return True
    if run_token in run_id_l:
        return True
    if path_token in model_path_l:
        return True
    return False


def _resolve_recording_id(
    registry: Registry,
    *,
    recording_id: Optional[str],
    recording_dir: Optional[Path],
) -> str:
    if recording_id:
        row = registry.conn.execute(
            "SELECT recording_id FROM recordings WHERE recording_id = ?;",
            (str(recording_id),),
        ).fetchone()
        if row is None:
            raise SystemExit(f"Recording not found: {recording_id}")
        return str(row["recording_id"])

    if recording_dir is None:
        raise SystemExit("Provide --recording-id or --recording-dir.")

    rec_dir = recording_dir.expanduser().resolve()
    row = registry.conn.execute(
        "SELECT recording_id FROM recordings WHERE recording_path = ?;",
        (str(rec_dir),),
    ).fetchone()
    if row is not None:
        return str(row["recording_id"])

    zarr_prefix = str(rec_dir / "zarr") + "/%"
    rows = registry.conn.execute(
        """
        SELECT DISTINCT recording_id
        FROM datasets
        WHERE recording_id IS NOT NULL
          AND zarr_path LIKE ?;
        """,
        (zarr_prefix,),
    ).fetchall()
    if len(rows) == 1:
        return str(rows[0]["recording_id"])
    if not rows:
        raise SystemExit(f"No recording_id found for recording dir: {rec_dir}")
    raise SystemExit(
        f"Multiple recording_ids matched recording dir {rec_dir}; pass --recording-id explicitly."
    )


def _load_target_profile(registry: Registry, recording_id: str) -> TargetProfile:
    row = registry.conn.execute(
        """
        WITH source_context AS (
            SELECT
                d.recording_id AS recording_id,
                MAX(NULLIF(TRIM(dcc.rig_id), '')) AS rig_id,
                MAX(NULLIF(TRIM(dcc.arena_id), '')) AS arena_id,
                MAX(NULLIF(TRIM(dcc.camera_id), '')) AS camera_id,
                MAX(NULLIF(TRIM(dcc.canvas_name), '')) AS canvas_name,
                MAX(NULLIF(TRIM(dcc.protocol_name), '')) AS protocol_name,
                MAX(NULLIF(TRIM(dcc.dish_design), '')) AS dish_design,
                MAX(NULLIF(TRIM(dcc.cross_id), '')) AS cross_id,
                MAX(NULLIF(TRIM(dcc.genotype), '')) AS genotype,
                MAX(dcc.dpf_at_acquisition) AS dpf_at_acquisition
            FROM datasets d
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = d.dataset_id
            WHERE d.recording_id = ?
              AND d.artifact_kind = 'source_recording'
            GROUP BY d.recording_id
        )
        SELECT
            r.recording_id,
            r.recording_type,
            r.recording_subtype,
            r.behavior_mode,
            COALESCE(NULLIF(TRIM(r.rig_id), ''), sc.rig_id) AS rig_id,
            COALESCE(NULLIF(TRIM(r.arena_id), ''), sc.arena_id) AS arena_id,
            COALESCE(NULLIF(TRIM(r.camera_id), ''), sc.camera_id) AS camera_id,
            COALESCE(NULLIF(TRIM(r.canvas_name), ''), sc.canvas_name) AS canvas_name,
            COALESCE(NULLIF(TRIM(r.protocol_name), ''), sc.protocol_name) AS protocol_name,
            COALESCE(NULLIF(TRIM(r.dish_design), ''), sc.dish_design) AS dish_design,
            sc.cross_id AS cross_id,
            sc.genotype AS genotype,
            sc.dpf_at_acquisition AS dpf_at_acquisition
        FROM recordings r
        LEFT JOIN source_context sc ON sc.recording_id = r.recording_id
        WHERE r.recording_id = ?;
        """,
        (recording_id, recording_id),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Recording not found in recordings table: {recording_id}")

    return TargetProfile(
        recording_id=str(row["recording_id"]),
        recording_type=_norm_text(row["recording_type"]),
        recording_subtype=_norm_text(row["recording_subtype"]),
        behavior_mode=_norm_text(row["behavior_mode"]),
        rig_id=_norm_text(row["rig_id"]),
        arena_id=_norm_text(row["arena_id"]),
        camera_id=_norm_text(row["camera_id"]),
        canvas_name=_norm_text(row["canvas_name"]),
        protocol_name=_norm_text(row["protocol_name"]),
        dish_design=_norm_text(row["dish_design"]),
        cross_id=_norm_text(row["cross_id"]),
        genotype=_norm_text(row["genotype"]),
        dpf_at_acquisition=_norm_int(row["dpf_at_acquisition"]),
    )


def _parse_dataset_ids(raw_json: Any) -> list[str]:
    if raw_json is None:
        return []
    try:
        payload = json.loads(str(raw_json))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if item is not None and str(item).strip()]


def _feature_value_from_row(row: Any, feature: str) -> Any:
    if feature == "dpf_at_acquisition":
        return _norm_int(row[feature])
    return _norm_text(row[feature])


def _feature_value_from_target(target: TargetProfile, feature: str) -> Any:
    value = getattr(target, feature)
    if feature == "dpf_at_acquisition":
        return _norm_int(value)
    return _norm_text(value)


def _score_rows(rows: list[Any], target: TargetProfile) -> tuple[float, dict[str, int], float]:
    if not rows:
        return 0.0, {}, 0.0
    total = float(len(rows))
    score = 0.0
    weight_total = 0.0
    match_counts: dict[str, int] = {}

    for feature, weight in FEATURE_WEIGHTS.items():
        target_value = _feature_value_from_target(target, feature)
        if target_value is None:
            continue
        weight_total += float(weight)
        matches = 0
        for row in rows:
            row_value = _feature_value_from_row(row, feature)
            if row_value == target_value:
                matches += 1
        match_counts[feature] = matches
        score += float(weight) * (float(matches) / total)

    if weight_total <= 0:
        return 0.0, match_counts, 0.0
    return score / weight_total, match_counts, weight_total


def _load_source_rows_for_set(registry: Registry, dataset_ids: list[str]) -> list[Any]:
    if not dataset_ids:
        return []
    placeholders = ", ".join("?" for _ in dataset_ids)
    sql = f"""
    SELECT
        d.dataset_id,
        dcc.recording_type AS recording_type,
        dcc.recording_subtype AS recording_subtype,
        dcc.behavior_mode AS behavior_mode,
        dcc.rig_id AS rig_id,
        dcc.arena_id AS arena_id,
        dcc.camera_id AS camera_id,
        dcc.canvas_name AS canvas_name,
        dcc.protocol_name AS protocol_name,
        dcc.dish_design AS dish_design,
        dcc.cross_id AS cross_id,
        dcc.genotype AS genotype,
        dcc.dpf_at_acquisition AS dpf_at_acquisition
    FROM datasets d
    LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = d.dataset_id
    WHERE d.dataset_id IN ({placeholders})
      AND d.artifact_kind = 'source_recording';
    """
    return registry.conn.execute(sql, dataset_ids).fetchall()


def _load_candidates(
    registry: Registry,
    *,
    target: TargetProfile,
    task: str,
    set_id_filter: Optional[str],
    include_non_success: bool,
) -> list[Candidate]:
    sql = [
        "SELECT",
        "  tr.run_id AS run_id,",
        "  tr.set_id AS set_id,",
        "  tr.task_type AS run_task_type,",
        "  ts.task_type AS set_task_type,",
        "  COALESCE(tm.model_path, tr.model_path) AS model_path,",
        "  COALESCE(tm.status, tr.status) AS status,",
        "  tr.created_utc AS created_utc,",
        "  ts.dataset_ids_json AS dataset_ids_json",
        "FROM training_runs tr",
        "LEFT JOIN training_models tm ON tm.run_id = tr.run_id",
        "LEFT JOIN training_sets ts ON ts.set_id = tr.set_id",
        "WHERE tr.set_id IS NOT NULL",
        "  AND TRIM(tr.set_id) <> ''",
        "  AND COALESCE(tm.model_path, tr.model_path) IS NOT NULL",
    ]
    params: list[Any] = []
    if not include_non_success:
        sql.append("  AND LOWER(COALESCE(tm.status, tr.status, '')) = 'success'")
    if set_id_filter:
        sql.append("  AND tr.set_id = ?")
        params.append(str(set_id_filter))
    sql.append("ORDER BY COALESCE(tr.created_utc, '') DESC, tr.run_id DESC")

    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    candidates: list[Candidate] = []
    for row in rows:
        run_id = str(row["run_id"])
        set_id = str(row["set_id"])
        model_path = str(row["model_path"])
        effective_task = _normalize_task_type(row["run_task_type"]) or _normalize_task_type(row["set_task_type"])
        if task != "any":
            if effective_task is not None and effective_task != task:
                continue
            if effective_task is None and not _matches_task(
                run_id=run_id,
                set_id=set_id,
                model_path=model_path,
                task=task,
            ):
                continue
        dataset_ids = _parse_dataset_ids(row["dataset_ids_json"])
        source_rows = _load_source_rows_for_set(registry, dataset_ids)
        weighted_score, match_counts, weights_used = _score_rows(source_rows, target)
        candidates.append(
            Candidate(
                run_id=run_id,
                set_id=set_id,
                model_path=model_path,
                created_utc=_norm_text(row["created_utc"]),
                status=_norm_text(row["status"]),
                dataset_count=len(source_rows),
                weighted_score=float(weighted_score),
                feature_match_counts=match_counts,
                feature_weights_used=float(weights_used),
            )
        )
    candidates.sort(
        key=lambda item: (
            item.weighted_score,
            item.created_utc or "",
            item.run_id,
        ),
        reverse=True,
    )
    return candidates


def _print_target(target: TargetProfile) -> None:
    print("Target recording profile")
    print(f"  recording_id: {target.recording_id}")
    for key in (
        "recording_type",
        "recording_subtype",
        "behavior_mode",
        "rig_id",
        "arena_id",
        "camera_id",
        "canvas_name",
        "protocol_name",
        "dish_design",
        "cross_id",
        "genotype",
        "dpf_at_acquisition",
    ):
        print(f"  {key}: {getattr(target, key)}")


def _print_candidates(candidates: list[Candidate], limit: int) -> None:
    print("Candidate models (ranked)")
    head = candidates[: max(0, limit)]
    if not head:
        print("  none")
        return
    for idx, item in enumerate(head, start=1):
        print(
            f"  {idx}. run_id={item.run_id} set_id={item.set_id} "
            f"score={item.weighted_score:.4f} datasets={item.dataset_count} "
            f"status={item.status or '-'}"
        )
        print(f"     model_path={item.model_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--recording-id", type=str, help="Recording ID to match.")
    parser.add_argument("--recording-dir", type=Path, help="Recording directory path to resolve recording ID.")
    parser.add_argument(
        "--task",
        choices=("detect", "pose", "eye_masks", "any"),
        default="detect",
        help="Model task family to resolve (default: detect).",
    )
    parser.add_argument("--set-id", type=str, help="Optional set_id filter.")
    parser.add_argument("--include-non-success", action="store_true", help="Include non-success model runs.")
    parser.add_argument("--require-unique", action="store_true", help="Fail if top score is tied.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked candidates to print.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    if not args.recording_id and not args.recording_dir:
        raise SystemExit("Provide --recording-id or --recording-dir.")

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        rec_id = _resolve_recording_id(
            registry,
            recording_id=args.recording_id,
            recording_dir=args.recording_dir,
        )
        target = _load_target_profile(registry, rec_id)
        candidates = _load_candidates(
            registry,
            target=target,
            task=str(args.task),
            set_id_filter=args.set_id,
            include_non_success=bool(args.include_non_success),
        )
    finally:
        registry.close()

    if not candidates:
        raise SystemExit("No candidate detection models found.")

    best = candidates[0]
    if args.require_unique and len(candidates) > 1:
        if abs(candidates[0].weighted_score - candidates[1].weighted_score) < 1e-12:
            raise SystemExit(
                "Top model score is tied. Re-run with --set-id to select deterministically."
            )

    payload = {
        "registry": str(registry_path),
        "task": str(args.task),
        "target": {
            "recording_id": target.recording_id,
            "recording_type": target.recording_type,
            "recording_subtype": target.recording_subtype,
            "behavior_mode": target.behavior_mode,
            "rig_id": target.rig_id,
            "arena_id": target.arena_id,
            "camera_id": target.camera_id,
            "canvas_name": target.canvas_name,
            "protocol_name": target.protocol_name,
            "dish_design": target.dish_design,
            "cross_id": target.cross_id,
            "genotype": target.genotype,
            "dpf_at_acquisition": target.dpf_at_acquisition,
        },
        "best": {
            "run_id": best.run_id,
            "set_id": best.set_id,
            "model_path": best.model_path,
            "score": best.weighted_score,
            "created_utc": best.created_utc,
            "status": best.status,
            "dataset_count": best.dataset_count,
            "feature_match_counts": best.feature_match_counts,
            "feature_weights_used": best.feature_weights_used,
        },
        "candidates": [
            {
                "run_id": item.run_id,
                "set_id": item.set_id,
                "model_path": item.model_path,
                "score": item.weighted_score,
                "created_utc": item.created_utc,
                "status": item.status,
                "dataset_count": item.dataset_count,
                "feature_match_counts": item.feature_match_counts,
                "feature_weights_used": item.feature_weights_used,
            }
            for item in candidates[: max(0, int(args.top_k))]
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _print_target(target)
    print("")
    _print_candidates(candidates, limit=int(args.top_k))
    print("")
    print("Recommended model")
    print(f"  run_id: {best.run_id}")
    print(f"  set_id: {best.set_id}")
    print(f"  score: {best.weighted_score:.4f}")
    print(f"  model_path: {best.model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
