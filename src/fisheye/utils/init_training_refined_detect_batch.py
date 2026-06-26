"""Initialize refined-detect review runs for sampled training Zarrs.

This is the detection-box counterpart to the acquisition-crop pose/mask review
bootstrap. It starts from existing ``detect_runs/<run>`` surfaces in sampled
training Zarrs and creates deterministic
``refined_detect_runs/<run>/instances`` surfaces that web detection assignment
can target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import zarr
from rich.console import Console

from fisheye.refinement.refine_detect import create_refined_run
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.maintenance import _backfill_recording_step_status


@dataclass(frozen=True)
class TrainingRefinedDetectPlan:
    dataset_id: str
    recording_id: str
    zarr_path: Path
    status: str
    reason: str
    detect_run: Optional[str] = None
    refined_run: Optional[str] = None


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


def _resolve_detect_run(root: zarr.Group, detect_run: Optional[str]) -> tuple[Optional[str], str]:
    detect_parent = root.get("detect_runs")
    if detect_parent is None:
        return None, "missing_detect_runs"
    if detect_run:
        return (detect_run, "present") if detect_run in detect_parent else (detect_run, "missing_detect_run")
    selected = detect_parent.attrs.get("latest_complete") or detect_parent.attrs.get("latest")
    if not selected:
        return None, "missing_detect_latest"
    selected_text = str(selected)
    if selected_text not in detect_parent:
        return selected_text, "missing_detect_run"
    return selected_text, "present"


def _refined_run_exists(root: zarr.Group, refined_run: str) -> bool:
    refined_parent = root.get("refined_detect_runs")
    return bool(refined_parent is not None and refined_run in refined_parent)


def _select_registry_rows(
    registry: Registry,
    *,
    path_contains: Sequence[str],
    recording_ids: Sequence[str],
    scope_paths: Sequence[Path],
) -> list[tuple[str, str, Path]]:
    rows = registry.conn.execute(
        """
        SELECT dataset_id, recording_id, zarr_path
        FROM datasets
        WHERE zarr_use = 'training'
          AND (status IS NULL OR lower(status) != 'missing')
        ORDER BY zarr_path;
        """
    ).fetchall()
    recording_id_filter = set(recording_ids)
    selected: list[tuple[str, str, Path]] = []
    for row in rows:
        zarr_path_text = str(row["zarr_path"] or "")
        if path_contains and not all(token in zarr_path_text for token in path_contains):
            continue
        recording_id = str(row["recording_id"] or "")
        if recording_id_filter and recording_id not in recording_id_filter:
            continue
        zarr_path = Path(zarr_path_text).expanduser()
        if not _path_in_scope(zarr_path, scope_paths):
            continue
        selected.append((str(row["dataset_id"]), recording_id, zarr_path))
    return selected


def build_plans(
    registry: Registry,
    *,
    run_id: str,
    refined_run_name: Optional[str],
    detect_run: Optional[str],
    path_contains: Sequence[str],
    recording_ids: Sequence[str],
    scope_paths: Sequence[Path],
    skip_existing: bool = True,
) -> list[TrainingRefinedDetectPlan]:
    refined_run = refined_run_name or f"refined_detect_training_review_{run_id}"
    plans: list[TrainingRefinedDetectPlan] = []
    for dataset_id, recording_id, zarr_path in _select_registry_rows(
        registry,
        path_contains=path_contains,
        recording_ids=recording_ids,
        scope_paths=scope_paths,
    ):
        if not zarr_path.exists():
            plans.append(
                TrainingRefinedDetectPlan(
                    dataset_id=dataset_id,
                    recording_id=recording_id,
                    zarr_path=zarr_path,
                    status="missing",
                    reason="zarr_path_missing",
                    refined_run=refined_run,
                )
            )
            continue
        try:
            root = _open_zarr_group(zarr_path, mode="r")
            selected_detect_run, detect_reason = _resolve_detect_run(root, detect_run)
            if detect_reason != "present":
                plans.append(
                    TrainingRefinedDetectPlan(
                        dataset_id=dataset_id,
                        recording_id=recording_id,
                        zarr_path=zarr_path,
                        status="missing",
                        reason=detect_reason,
                        detect_run=selected_detect_run,
                        refined_run=refined_run,
                    )
                )
                continue
            if skip_existing and _refined_run_exists(root, refined_run):
                plans.append(
                    TrainingRefinedDetectPlan(
                        dataset_id=dataset_id,
                        recording_id=recording_id,
                        zarr_path=zarr_path,
                        status="skipped",
                        reason="refined_run_exists",
                        detect_run=selected_detect_run,
                        refined_run=refined_run,
                    )
                )
                continue
            plans.append(
                TrainingRefinedDetectPlan(
                    dataset_id=dataset_id,
                    recording_id=recording_id,
                    zarr_path=zarr_path,
                    status="ok",
                    reason="ready",
                    detect_run=selected_detect_run,
                    refined_run=refined_run,
                )
            )
        except Exception as exc:
            plans.append(
                TrainingRefinedDetectPlan(
                    dataset_id=dataset_id,
                    recording_id=recording_id,
                    zarr_path=zarr_path,
                    status="error",
                    reason=f"zarr_open_or_plan_error: {exc}",
                    refined_run=refined_run,
                )
            )
    return plans


def _print_plan(plans: Sequence[TrainingRefinedDetectPlan], *, limit: int) -> None:
    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
    print("Plan summary:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    print(f"  total: {len(plans)}")
    shown = plans if limit <= 0 else plans[:limit]
    for plan in shown:
        print(f"{plan.zarr_path}")
        print(f"  dataset_id: {plan.dataset_id}")
        print(f"  status: {plan.status}")
        print(f"  reason: {plan.reason}")
        if plan.detect_run:
            print(f"  detect_run: {plan.detect_run}")
        if plan.refined_run:
            print(f"  refined_run: {plan.refined_run}")
    if limit > 0 and len(plans) > limit:
        print(f"... {len(plans) - limit} more")


def _run_apply(
    plans: Sequence[TrainingRefinedDetectPlan],
    *,
    per_frame_top_k: int | None,
    top_k_score_field: str,
    require_detect_quality: bool,
    console: Console,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for plan in plans:
        payload = asdict(plan)
        payload["zarr_path"] = str(plan.zarr_path)
        if plan.status != "ok":
            payload["result"] = "skipped"
            results.append(payload)
            continue
        assert plan.detect_run is not None
        assert plan.refined_run is not None
        try:
            created = create_refined_run(
                zarr_path=str(plan.zarr_path),
                detect_run=plan.detect_run,
                config={"refine_detect": {}},
                console=console,
                require_detect_quality=bool(require_detect_quality),
                per_frame_top_k=per_frame_top_k,
                top_k_score_field=top_k_score_field,
                refined_run_name=plan.refined_run,
            )
            payload["created_refined_run"] = created
            payload["result"] = "created"
        except Exception as exc:
            payload["result"] = "error"
            payload["error"] = str(exc)
        results.append(payload)
    return results


def _refresh_status(
    registry: Registry,
    plans: Sequence[TrainingRefinedDetectPlan],
    *,
    apply: bool,
) -> dict[str, object]:
    scoped = [plan for plan in plans if plan.status in {"ok", "skipped"}]
    if not scoped:
        return {"rows_evaluated": 0, "rows_inserted": 0, "rows_updated": 0, "rows_skipped": 0}
    return _backfill_recording_step_status(
        registry,
        dry_run=not bool(apply),
        scope_paths=[plan.zarr_path for plan in scoped],
        recording_ids=[plan.recording_id for plan in scoped],
        zarr_use_filter="training",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Optional training Zarr path or containing root scope.")
    parser.add_argument("--registry", type=Path, help="Registry SQLite path. Defaults to registry discovery.")
    parser.add_argument("--run-id", required=True, help="Deterministic run id suffix for refined detect review run names.")
    parser.add_argument("--refined-run-name", help="Explicit refined detect run name. Defaults to refined_detect_training_review_<run-id>.")
    parser.add_argument("--detect-run", help="Explicit source detect run. Defaults to detect_runs.latest_complete/latest per Zarr.")
    parser.add_argument("--path-contains", action="append", default=[], help="Substring filter for training zarr_path.")
    parser.add_argument("--recording-id", action="append", default=[], help="Recording id filter; repeat or comma-separate.")
    parser.add_argument("--per-frame-top-k", type=int, default=1)
    parser.add_argument("--top-k-score-field", default="scores", choices=("scores",))
    parser.add_argument("--allow-missing-quality", action="store_true", help="Allow non-sampled Zarrs without detect_quality reports.")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-refresh-step-status", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Create refined runs. Default is dry-run.")
    parser.add_argument("--allow-all", action="store_true", help="Allow running without path/recording filters.")
    parser.add_argument("--list-limit", type=int, default=20)
    parser.add_argument("--json", action="store_true")
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
        plans = build_plans(
            registry,
            run_id=args.run_id,
            refined_run_name=args.refined_run_name,
            detect_run=args.detect_run,
            path_contains=path_contains,
            recording_ids=recording_ids,
            scope_paths=scope_paths,
            skip_existing=not bool(args.no_skip_existing),
        )
        apply_results: list[dict[str, object]] = []
        if args.apply:
            apply_results = _run_apply(
                plans,
                per_frame_top_k=args.per_frame_top_k,
                top_k_score_field=args.top_k_score_field,
                require_detect_quality=not bool(args.allow_missing_quality),
                console=Console(),
            )
        status_refresh = None
        if not args.no_refresh_step_status:
            status_refresh = _refresh_status(registry, plans, apply=bool(args.apply))
        summary: dict[str, object] = {
            "registry": str(Path(registry_path).expanduser()),
            "apply": bool(args.apply),
            "run_id": args.run_id,
            "plans": [
                {**asdict(plan), "zarr_path": str(plan.zarr_path)}
                for plan in plans
            ],
            "apply_results": apply_results,
            "status_refresh": status_refresh,
        }
    finally:
        registry.close()

    if not args.json:
        _print_plan(plans, limit=int(args.list_limit))
        if status_refresh is not None:
            print(
                "Status refresh: "
                f"inserted={status_refresh.get('rows_inserted', 0)} "
                f"updated={status_refresh.get('rows_updated', 0)} "
                f"skipped={status_refresh.get('rows_skipped', 0)}"
            )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
