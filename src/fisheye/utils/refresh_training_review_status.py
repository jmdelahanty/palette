"""Audit/repair deterministic training-review surfaces and registry statuses.

This utility captures the operational sequence used after a training-review
bootstrap batch:

1. Select training datasets from the registry.
2. Verify the deterministic review run groups exist.
3. Optionally stamp missing Zarr run-completion markers after contract validation.
4. Optionally backfill ``recording_step_status`` for the same dataset scope.

The command is dry-run by default. Pass ``--apply`` to write Zarr attrs or
registry rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.maintenance import _backfill_recording_step_status
from fisheye.shared.zarr.stage_arrays import STAGES, StageSpec, validate_run
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
)


@dataclass(frozen=True)
class DatasetCandidate:
    dataset_id: str
    recording_id: str
    zarr_path: Path
    zarr_use: str


@dataclass(frozen=True)
class ReviewRunSpec:
    family: str
    parent_group: str
    stage_name: str
    run_name: str


REVIEW_FAMILY_DEFAULTS: tuple[tuple[str, str, str, str], ...] = (
    ("keypoints", "keypoints_runs", "keypoints", "keypoints_training_review_{run_id}"),
    (
        "refined_keypoints",
        "refined_keypoints_runs",
        "refined_keypoints",
        "refined_keypoints_training_review_{run_id}",
    ),
    ("subject_masks", "subject_mask_runs", "subject_masks", "subject_masks_training_review_{run_id}"),
    (
        "refined_subject_masks",
        "refined_subject_masks_runs",
        "refined_subject_masks",
        "refined_subject_masks_training_review_{run_id}",
    ),
)


def _split_csv(values: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or ():
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def _normalize_scope_paths(paths: Sequence[Path] | None) -> list[Path]:
    normalized: list[Path] = []
    for path in paths or ():
        expanded = Path(path).expanduser()
        try:
            normalized.append(expanded.resolve())
        except Exception:
            normalized.append(expanded.absolute())
    return normalized


def _path_in_scope(path: Path, scope_paths: Sequence[Path]) -> bool:
    if not scope_paths:
        return True
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve()
    except Exception:
        candidate = candidate.absolute()
    for scope in scope_paths:
        if candidate == scope:
            return True
        try:
            candidate.relative_to(scope)
            return True
        except ValueError:
            continue
    return False


def _open_zarr_group(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        try:
            return zarr.open_group(str(path), mode=mode, consolidated=False)
        except TypeError:
            return zarr.open_group(str(path), mode=mode)


def _review_run_specs(
    run_id: str,
    *,
    keypoints_run: Optional[str] = None,
    refined_keypoints_run: Optional[str] = None,
    subject_masks_run: Optional[str] = None,
    refined_subject_masks_run: Optional[str] = None,
    families: Sequence[str] | None = None,
) -> list[ReviewRunSpec]:
    overrides = {
        "keypoints": keypoints_run,
        "refined_keypoints": refined_keypoints_run,
        "subject_masks": subject_masks_run,
        "refined_subject_masks": refined_subject_masks_run,
    }
    selected = set(families or ())
    specs: list[ReviewRunSpec] = []
    for family, parent_group, stage_name, template in REVIEW_FAMILY_DEFAULTS:
        if selected and family not in selected:
            continue
        run_name = overrides[family] or template.format(run_id=run_id)
        specs.append(
            ReviewRunSpec(
                family=family,
                parent_group=parent_group,
                stage_name=stage_name,
                run_name=run_name,
            )
        )
    return specs


def _select_dataset_candidates(
    registry: Registry,
    *,
    zarr_use: str,
    path_contains: Sequence[str],
    recording_ids: Sequence[str],
    scope_paths: Sequence[Path],
) -> list[DatasetCandidate]:
    zarr_use_norm = str(zarr_use or "training").strip().lower()
    rows = registry.conn.execute(
        """
        SELECT dataset_id, recording_id, zarr_path, zarr_use
        FROM datasets
        WHERE status IS NULL OR lower(status) != 'missing'
        ORDER BY zarr_path;
        """
    ).fetchall()
    recording_id_filter = set(recording_ids)
    candidates: list[DatasetCandidate] = []
    for row in rows:
        row_zarr_use = str(row["zarr_use"] or "").strip().lower()
        if zarr_use_norm != "all" and row_zarr_use != zarr_use_norm:
            continue
        recording_id = str(row["recording_id"] or "").strip()
        if recording_id_filter and recording_id not in recording_id_filter:
            continue
        zarr_path_text = str(row["zarr_path"] or "")
        if path_contains and not all(token in zarr_path_text for token in path_contains):
            continue
        zarr_path = Path(zarr_path_text).expanduser()
        if not _path_in_scope(zarr_path, scope_paths):
            continue
        candidates.append(
            DatasetCandidate(
                dataset_id=str(row["dataset_id"]),
                recording_id=recording_id,
                zarr_path=zarr_path,
                zarr_use=row_zarr_use,
            )
        )
    return candidates


def _family_summary_template(specs: Sequence[ReviewRunSpec]) -> dict[str, dict[str, int]]:
    return {
        spec.family: {
            "present": 0,
            "missing": 0,
            "complete": 0,
            "unmarked": 0,
            "valid": 0,
            "invalid": 0,
            "stamped": 0,
            "already_complete": 0,
        }
        for spec in specs
    }


def _completion_status(attrs: Mapping[str, Any]) -> str | None:
    value = attrs.get(RUN_COMPLETION_STATUS_ATTR)
    return str(value).strip().lower() if value is not None else None


def audit_and_stamp_review_runs(
    candidates: Sequence[DatasetCandidate],
    specs: Sequence[ReviewRunSpec],
    *,
    apply: bool,
    stamp_completion_markers: bool,
    include_examples: int = 20,
) -> dict[str, object]:
    family_counts = _family_summary_template(specs)
    examples: list[dict[str, object]] = []
    mode = "a" if apply and stamp_completion_markers else "r"

    for candidate in candidates:
        try:
            root = _open_zarr_group(candidate.zarr_path, mode=mode)
        except Exception as exc:
            examples.append(
                {
                    "dataset_id": candidate.dataset_id,
                    "zarr_path": str(candidate.zarr_path),
                    "status": "zarr_open_error",
                    "error": str(exc),
                }
            )
            continue

        for spec in specs:
            counts = family_counts[spec.family]
            parent = root.get(spec.parent_group)
            if parent is None or spec.run_name not in parent:
                counts["missing"] += 1
                if len(examples) < include_examples:
                    examples.append(
                        {
                            "dataset_id": candidate.dataset_id,
                            "family": spec.family,
                            "run_name": spec.run_name,
                            "status": "missing",
                        }
                    )
                continue

            counts["present"] += 1
            run_group = parent[spec.run_name]
            attrs = dict(run_group.attrs)
            if _completion_status(attrs) == RUN_STATUS_COMPLETE:
                counts["complete"] += 1
                counts["already_complete"] += 1
                continue

            counts["unmarked"] += 1
            stage_spec: StageSpec = STAGES[spec.stage_name]
            validation = validate_run(run_group, stage_spec)
            if not validation.valid:
                counts["invalid"] += 1
                if len(examples) < include_examples:
                    examples.append(
                        {
                            "dataset_id": candidate.dataset_id,
                            "family": spec.family,
                            "run_name": spec.run_name,
                            "status": "invalid",
                            "errors": validation.errors[:5],
                        }
                    )
                continue

            counts["valid"] += 1
            if apply and stamp_completion_markers:
                mark_run_complete(run_group, parent_group=parent, run_name=spec.run_name)
                if spec.family == "refined_subject_masks":
                    parent.attrs["refined_subject_mask_review_status_latest"] = spec.run_name
                counts["stamped"] += 1

    return {
        "families": family_counts,
        "examples": examples,
        "apply": bool(apply),
        "stamp_completion_markers": bool(stamp_completion_markers),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Optional exact Zarr path or containing root scope.")
    parser.add_argument("--registry", type=Path, help="Registry SQLite path. Defaults to registry discovery.")
    parser.add_argument("--run-id", required=True, help="Bootstrap run-id used for deterministic training-review run names.")
    parser.add_argument("--zarr-use", default="training", choices=("all", "analysis", "training"))
    parser.add_argument("--path-contains", action="append", default=[], help="Substring filter for dataset zarr_path.")
    parser.add_argument("--recording-id", action="append", default=[], help="Recording id filter; repeat or comma-separate.")
    parser.add_argument("--family", action="append", choices=[row[0] for row in REVIEW_FAMILY_DEFAULTS])
    parser.add_argument("--keypoints-run")
    parser.add_argument("--refined-keypoints-run")
    parser.add_argument("--subject-masks-run")
    parser.add_argument("--refined-subject-masks-run")
    parser.add_argument("--stamp-completion-markers", action="store_true")
    parser.add_argument("--refresh-step-status", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Perform writes. Without this, all actions are dry-runs.")
    parser.add_argument("--allow-all", action="store_true", help="Allow running without path/recording filters.")
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Emit only JSON.")
    return parser


def _require_scope(args: argparse.Namespace, *, path_contains: Sequence[str], recording_ids: Sequence[str]) -> None:
    if args.allow_all:
        return
    if args.paths or path_contains or recording_ids:
        return
    raise SystemExit(
        "Refusing unscoped registry scan. Provide a path, --path-contains, "
        "--recording-id, or pass --allow-all."
    )


def _print_human_summary(summary: Mapping[str, object]) -> None:
    print(f"datasets_selected: {summary['datasets_selected']}")
    print(f"apply: {summary['apply']}")
    audit = summary.get("review_surface_audit")
    if isinstance(audit, Mapping):
        families = audit.get("families")
        if isinstance(families, Mapping):
            for family, counts in families.items():
                if not isinstance(counts, Mapping):
                    continue
                print(
                    f"{family}: present={counts.get('present', 0)} "
                    f"complete={counts.get('complete', 0)} "
                    f"unmarked={counts.get('unmarked', 0)} "
                    f"valid={counts.get('valid', 0)} "
                    f"invalid={counts.get('invalid', 0)} "
                    f"stamped={counts.get('stamped', 0)} "
                    f"missing={counts.get('missing', 0)}"
                )
    backfill = summary.get("recording_step_status_backfill")
    if isinstance(backfill, Mapping):
        print(
            "recording_step_status_backfill: "
            f"inserted={backfill.get('rows_inserted', 0)} "
            f"updated={backfill.get('rows_updated', 0)} "
            f"skipped={backfill.get('rows_skipped', 0)} "
            f"history={backfill.get('history_rows_inserted', 0)}"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path_contains = _split_csv(args.path_contains)
    recording_ids = _split_csv(args.recording_id)
    scope_paths = _normalize_scope_paths(args.paths)
    _require_scope(args, path_contains=path_contains, recording_ids=recording_ids)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(Path(registry_path).expanduser())
    try:
        candidates = _select_dataset_candidates(
            registry,
            zarr_use=args.zarr_use,
            path_contains=path_contains,
            recording_ids=recording_ids,
            scope_paths=scope_paths,
        )
        specs = _review_run_specs(
            args.run_id,
            keypoints_run=args.keypoints_run,
            refined_keypoints_run=args.refined_keypoints_run,
            subject_masks_run=args.subject_masks_run,
            refined_subject_masks_run=args.refined_subject_masks_run,
            families=args.family,
        )
        audit_summary = audit_and_stamp_review_runs(
            candidates,
            specs,
            apply=bool(args.apply),
            stamp_completion_markers=bool(args.stamp_completion_markers),
            include_examples=max(0, int(args.examples)),
        )

        backfill_summary: dict[str, object] | None = None
        if args.refresh_step_status:
            backfill_summary = _backfill_recording_step_status(
                registry,
                dry_run=not bool(args.apply),
                scope_paths=[candidate.zarr_path for candidate in candidates],
                recording_ids=[candidate.recording_id for candidate in candidates],
                zarr_use_filter=args.zarr_use,
            )

        summary: dict[str, object] = {
            "registry": str(Path(registry_path).expanduser()),
            "run_id": args.run_id,
            "zarr_use": args.zarr_use,
            "datasets_selected": len(candidates),
            "dataset_ids": [candidate.dataset_id for candidate in candidates],
            "apply": bool(args.apply),
            "review_surface_audit": audit_summary,
            "recording_step_status_backfill": backfill_summary,
        }
    finally:
        registry.close()

    if not args.json:
        _print_human_summary(summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
