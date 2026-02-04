import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import zarr

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
        if root.is_file() and root.suffix == ".zarr":
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
        runs[parent] = list(group.group_keys())
    return runs


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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prune unreferenced run groups from Palette Zarr archives.")
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarrs.")
    parser.add_argument("--apply", action="store_true", help="Delete runs (default is dry-run).")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store JSONL logs.")
    args = parser.parse_args(argv)

    roots = _resolve_root(args.paths)
    zarr_paths = list(_iter_zarr(roots, args.recursive))
    if not zarr_paths:
        print("No zarr files found.")
        return 1

    plans: List[RunPrunePlan] = []
    for zarr_path in zarr_paths:
        root = zarr.open_group(str(zarr_path), mode="r")
        deletions, skips = _build_plan(root, RUN_PARENTS)
        plans.append(RunPrunePlan(zarr_path=zarr_path, deletions=deletions, skips=skips))

    if not args.apply:
        print("Planned run pruning (dry-run):")
        for plan in plans:
            if not plan.deletions:
                continue
            print(f"{plan.zarr_path}")
            for parent, runs in plan.deletions.items():
                print(f"  {parent}: {len(runs)} candidate(s)")
                for name in runs:
                    print(f"    - {name}")
            for parent, reason in plan.skips.items():
                print(f"  {parent}: skipped ({reason})")
        print("\nUse --apply to delete the candidates listed above.")
        return 0

    log_dir = _resolve_log_dir(args.log_dir, roots)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    log_path = log_dir / f"prune_zarr_runs_{run_id}.jsonl"
    logger = JsonLogger(log_path, run_id)
    print(f"Log file: {log_path}")

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
