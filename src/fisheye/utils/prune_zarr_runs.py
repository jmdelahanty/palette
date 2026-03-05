import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import zarr
from fisheye.diagnostics.check_eye_mask_lineage import _analyze_run_group

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore
    Progress = None  # type: ignore
    SpinnerColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    BarColumn = None  # type: ignore
    TimeElapsedColumn = None  # type: ignore
    TimeRemainingColumn = None  # type: ignore


RUN_PARENTS = [
    "detect_runs",
    "refined_detect_runs",
    "crop_runs",
    "keypoints_runs",
    "refined_keypoints_runs",
    "eye_masks_runs",
    "refined_eye_masks_runs",
    "id_assignment_runs",
    "background_runs",
    "stimulus_runs",
]
RUN_PARENT_SET = set(RUN_PARENTS)

LINEAGE_RUN_PARENTS = [
    "eye_masks_runs",
    "refined_eye_masks_runs",
]


@dataclass
class RunPrunePlan:
    zarr_path: Path
    deletions: Dict[str, List[str]]
    skips: Dict[str, str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, event: str, **fields: object) -> None:
        payload = {"event": event, "ts_utc": _utc_now(), "run_id": self.run_id}
        payload.update(fields)
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "prune_zarr_runs"
    if roots:
        return roots[0] / "logs" / "prune_zarr_runs"
    return Path.cwd() / "logs" / "prune_zarr_runs"


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{os.getpid()}"


def _progress(console: Optional[Console], total: int):
    if console is None or Progress is None:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _iter_zarr(roots: List[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.suffix == ".zarr" and root.exists():
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


def _iter_groups(group: zarr.Group) -> Iterable[zarr.Group]:
    yield group
    for _, sub in group.groups():
        yield from _iter_groups(sub)


def _collect_attr_strings(root: zarr.Group) -> List[str]:
    values: List[str] = []

    def visit(obj: object) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            values.append(obj)
            return
        if isinstance(obj, bytes):
            values.append(obj.decode("utf-8", "ignore"))
            return
        if isinstance(obj, dict):
            for val in obj.values():
                visit(val)
            return
        if isinstance(obj, (list, tuple)):
            for val in obj:
                visit(val)

    for group in _iter_groups(root):
        for _, value in group.attrs.items():
            visit(value)

    return values


def _extract_path_refs(text: str, parents: List[str]) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    for parent in parents:
        marker = f"{parent}/"
        if marker not in text:
            continue
        tail = text.split(marker, 1)[1]
        run = tail.split("/", 1)[0]
        if run:
            refs.append((parent, run))
    return refs


def _list_runs(root: zarr.Group, parents: List[str]) -> Dict[str, List[str]]:
    runs: Dict[str, List[str]] = {}
    for parent in parents:
        group = root.get(parent)
        if group is None:
            continue
        if hasattr(group, "group_keys"):
            runs[parent] = list(group.group_keys())
        else:
            names: List[str] = []
            for name in group.keys():
                try:
                    obj = group[name]
                except Exception:
                    continue
                if isinstance(obj, zarr.Group):
                    names.append(str(name))
            runs[parent] = names
    return runs


def _append_skip(skips: Dict[str, str], parent: str, message: str) -> None:
    existing = skips.get(parent)
    if existing is None:
        skips[parent] = message
        return
    if message in existing:
        return
    skips[parent] = f"{existing}; {message}"


def _build_plan(root: zarr.Group, parents: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    runs = _list_runs(root, parents)
    attr_strings = _collect_attr_strings(root)

    referenced: Dict[str, Set[str]] = {parent: set() for parent in parents}

    # References from full paths inside attrs.
    for text in attr_strings:
        for parent, run in _extract_path_refs(text, parents):
            referenced[parent].add(run)

    # References from raw run-name values.
    run_name_lookup: Dict[str, Set[str]] = {}
    for parent, names in runs.items():
        for name in names:
            run_name_lookup.setdefault(name, set()).add(parent)

    for text in attr_strings:
        parents_for_name = run_name_lookup.get(text)
        if parents_for_name:
            for parent in parents_for_name:
                referenced[parent].add(text)

    deletions: Dict[str, List[str]] = {}
    skips: Dict[str, str] = {}

    for parent, names in runs.items():
        group = root.get(parent)
        if group is None or not names:
            continue

        latest = group.attrs.get("latest")
        if not latest:
            skips[parent] = "no latest attr; skipping prune for safety"
            continue

        keep = set(referenced[parent])
        keep.add(str(latest))

        candidates = [name for name in names if name not in keep]
        if candidates:
            deletions[parent] = sorted(candidates)

    return deletions, skips


def _build_lineage_failure_plan(
    root: zarr.Group,
    parents: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    deletions: Dict[str, List[str]] = {}
    skips: Dict[str, str] = {}

    runs = _list_runs(root, parents)
    for parent, run_names in runs.items():
        group = root.get(parent)
        if group is None or not run_names:
            continue

        latest = group.attrs.get("latest")
        if not latest or str(latest) not in group:
            _append_skip(skips, parent, "no latest attr; skipping prune for safety")
            continue
        latest_name = str(latest)

        for run_name in sorted(str(name) for name in run_names):
            run_group = group.get(run_name)
            if run_group is None:
                continue
            report = _analyze_run_group(
                root=root,
                stage=parent,
                run_name=run_name,
                run_group=run_group,
            )
            if not report.has_issues:
                continue
            if run_name == latest_name:
                _append_skip(
                    skips,
                    parent,
                    f"latest run '{run_name}' has lineage issues; not pruning latest",
                )
                continue
            deletions.setdefault(parent, []).append(run_name)

    for parent, names in deletions.items():
        deletions[parent] = sorted(set(names))
    return deletions, skips


def _resolve_selected_parents(raw: Optional[str]) -> List[str]:
    if raw is None:
        return list(RUN_PARENTS)
    tokens = [token.strip() for token in str(raw).split(",")]
    selected: List[str] = []
    seen: Set[str] = set()
    for token in tokens:
        if not token:
            continue
        if token not in RUN_PARENT_SET:
            allowed = ", ".join(sorted(RUN_PARENT_SET))
            raise SystemExit(f"Unknown run parent '{token}'. Allowed values: {allowed}")
        if token in seen:
            continue
        seen.add(token)
        selected.append(token)
    if not selected:
        raise SystemExit("--parents requires at least one run parent name.")
    return selected


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prune unreferenced run groups from Palette Zarr archives.")
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarrs.")
    parser.add_argument("--apply", action="store_true", help="Delete runs (default is dry-run).")
    parser.add_argument(
        "--lineage-failures-only",
        action="store_true",
        help=(
            "Prune only non-latest eye-mask runs that fail lineage checks "
            "(frame_indices/detection_indices/frame_counts/source_crop_run)."
        ),
    )
    parser.add_argument(
        "--lineage-stage",
        choices=["eye_masks_runs", "refined_eye_masks_runs", "both"],
        default="both",
        help="When --lineage-failures-only is set, choose which stage(s) to inspect (default: both).",
    )
    parser.add_argument(
        "--parents",
        type=str,
        help=(
            "Comma-separated run parents to prune in standard mode "
            f"(default: all). Choices: {', '.join(RUN_PARENTS)}"
        ),
    )
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store JSONL logs.")
    args = parser.parse_args(argv)

    if args.lineage_stage and not args.lineage_failures_only and args.lineage_stage != "both":
        raise SystemExit("--lineage-stage requires --lineage-failures-only.")
    if args.parents and args.lineage_failures_only:
        raise SystemExit("--parents cannot be used with --lineage-failures-only.")

    selected_parents = _resolve_selected_parents(args.parents)

    roots = _resolve_root(args.paths)
    zarr_paths = list(_iter_zarr(roots, args.recursive))
    if not zarr_paths:
        print("No zarr files found.")
        return 1

    plans: List[RunPrunePlan] = []
    if args.lineage_failures_only:
        if args.lineage_stage == "both":
            lineage_parents = list(LINEAGE_RUN_PARENTS)
        else:
            lineage_parents = [str(args.lineage_stage)]
    else:
        lineage_parents = []

    for zarr_path in zarr_paths:
        root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        if args.lineage_failures_only:
            deletions, skips = _build_lineage_failure_plan(root, lineage_parents)
        else:
            deletions, skips = _build_plan(root, selected_parents)
        plans.append(RunPrunePlan(zarr_path=zarr_path, deletions=deletions, skips=skips))

    if not args.apply:
        if args.lineage_failures_only:
            print("Planned lineage-failure run pruning (dry-run):")
        else:
            print("Planned run pruning (dry-run):")
        any_output = False
        for plan in plans:
            if not plan.deletions and not plan.skips:
                continue
            any_output = True
            print(f"{plan.zarr_path}")
            for parent, runs in plan.deletions.items():
                print(f"  {parent}: {len(runs)} candidate(s)")
                for name in runs:
                    print(f"    - {name}")
            for parent, reason in plan.skips.items():
                print(f"  {parent}: skipped ({reason})")
        if not any_output:
            print("No prune candidates found.")
        print("\nUse --apply to delete the candidates listed above.")
        return 0

    log_dir = _resolve_log_dir(args.log_dir, roots)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    log_path = log_dir / f"prune_zarr_runs_{run_id}.jsonl"
    logger = JsonLogger(log_path, run_id)
    print(f"Log file: {log_path}")
    logger.log(
        "run_start",
        lineage_failures_only=bool(args.lineage_failures_only),
        lineage_stage=str(args.lineage_stage),
        selected_parents=selected_parents,
    )

    console = Console() if Console else None
    progress = _progress(console, len(plans))
    task_id = progress.add_task("prune_runs", total=len(plans)) if progress else None

    deleted_total = 0
    for plan in plans:
        logger.log("zarr_start", zarr=str(plan.zarr_path))
        if not plan.deletions:
            logger.log("zarr_noop", zarr=str(plan.zarr_path))
            if progress:
                progress.advance(task_id)
            continue
        root = zarr.open_group(str(plan.zarr_path), mode="a")
        for parent, runs in plan.deletions.items():
            group = root.get(parent)
            if group is None:
                continue
            for name in runs:
                if name in group:
                    del group[name]
                    deleted_total += 1
                    logger.log("deleted_run", zarr=str(plan.zarr_path), parent=parent, run=name)
        logger.log("zarr_done", zarr=str(plan.zarr_path))
        if progress:
            progress.advance(task_id)

    if progress:
        progress.stop()

    logger.log("run_end", deleted_runs=deleted_total)
    logger.close()
    print(f"Deleted runs: {deleted_total}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
