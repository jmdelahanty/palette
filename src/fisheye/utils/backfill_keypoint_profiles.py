from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional

import zarr

from fisheye.registry.db import Registry
from fisheye.utils.keypoint_profile import (
    KeypointProfileError,
    KeypointSourceError,
    build_keypoint_profile_summary,
    infer_zarr_use,
    write_keypoint_profile,
)


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


@dataclass(frozen=True)
class BackfillRow:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    run_name: Optional[str] = None
    source_keypoint_path: Optional[str] = None


@dataclass(frozen=True)
class RegistryDatasetContext:
    dataset_id: Optional[str] = None
    recording_id: Optional[str] = None
    zarr_use: Optional[str] = None


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [DEFAULT_RECORDINGS_ROOT]


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        candidates: list[Path] = []
        if root.suffix == ".zarr" and (root.is_dir() or root.is_file()):
            candidates = [root]
        elif root.exists():
            if recursive:
                candidates = sorted(root.rglob("*.zarr"))
            else:
                candidates = sorted(root.glob("*.zarr")) + sorted(root.glob("*/zarr/*.zarr"))
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _open_child_group(parent: zarr.Group, key: str) -> Optional[zarr.Group]:
    store = getattr(parent, "store", None)
    if store is None:
        return None
    parent_path = _normalize_text(getattr(parent, "path", None))
    child_path = f"{parent_path}/{key}" if parent_path else key
    try:
        return zarr.open_group(store=store, path=child_path, mode="r")
    except TypeError:
        try:
            return zarr.open_group(store, mode="r", path=child_path)
        except Exception:
            return None
    except Exception:
        return None


def _get_group(parent: zarr.Group, key: str) -> Optional[zarr.Group]:
    child = parent.get(key)
    if child is not None:
        return child
    try:
        child = parent[key]
    except Exception:
        child = None
    if child is None:
        child = _open_child_group(parent, key)
    return child


def _run_exists(root: zarr.Group, run_name: Optional[str]) -> bool:
    if not run_name:
        return False
    analysis = _get_group(root, "analysis")
    if analysis is None:
        return False
    parent = _get_group(analysis, "keypoint_profile_runs")
    if parent is None:
        return False
    return run_name in parent


def _format_row(row: BackfillRow) -> str:
    source = row.source_keypoint_path or "-"
    reason = row.reason or "-"
    run_name = row.run_name or "-"
    return f"{row.status}\t{row.zarr_path}\t{run_name}\t{source}\t{reason}"


def _default_run_name_from_created(created_at_utc: object) -> Optional[str]:
    if not isinstance(created_at_utc, str):
        return None
    created = created_at_utc.strip()
    if len(created) < 19:
        return None
    date_part = created[0:10]
    time_part = created[11:19]
    if not date_part or not time_part:
        return None
    return f"keypoint_profile_{date_part}_{time_part}".replace(":", "-")


def _lookup_registry_dataset_context(registry: Optional[Registry], zarr_path: Path) -> RegistryDatasetContext:
    if registry is None:
        return RegistryDatasetContext()
    candidates = [str(zarr_path)]
    try:
        resolved = str(zarr_path.resolve())
    except OSError:
        resolved = None
    if resolved and resolved not in candidates:
        candidates.append(resolved)

    for candidate in candidates:
        row = registry.conn.execute(
            "SELECT dataset_id, recording_id, zarr_use FROM datasets WHERE zarr_path = ? LIMIT 1;",
            (candidate,),
        ).fetchone()
        if row is None:
            continue
        return RegistryDatasetContext(
            dataset_id=_normalize_text(row["dataset_id"]),
            recording_id=_normalize_text(row["recording_id"]),
            zarr_use=_normalize_text(row["zarr_use"]),
        )
    return RegistryDatasetContext()


def _process_zarr_path(
    zarr_path: Path,
    *,
    apply: bool,
    zarr_use_filter: str,
    run_name: Optional[str],
    overwrite_existing: bool,
    source_keypoint_path: Optional[str],
    keypoint_run: Optional[str],
    refined_run: Optional[str],
    dataset_id: Optional[str],
    recording_id: Optional[str],
    dataset_zarr_use: Optional[str],
) -> BackfillRow:
    mode = "a" if apply else "r"
    root = zarr.open_group(str(zarr_path), mode=mode)

    observed_use = infer_zarr_use(root, zarr_path)
    if zarr_use_filter != "any" and observed_use != zarr_use_filter:
        return BackfillRow(
            zarr_path=zarr_path,
            status="filtered_zarr_use",
            reason=f"wanted={zarr_use_filter} found={observed_use or 'unknown'}",
        )

    if _run_exists(root, run_name) and not overwrite_existing:
        return BackfillRow(
            zarr_path=zarr_path,
            status="skipped_existing",
            reason=f"analysis/keypoint_profile_runs/{run_name} already exists",
            run_name=run_name,
        )

    try:
        resolved_zarr_use = dataset_zarr_use or observed_use
        if apply:
            result = write_keypoint_profile(
                root,
                zarr_path=zarr_path,
                run_name=run_name,
                overwrite=overwrite_existing,
                source_keypoint_path=source_keypoint_path,
                keypoint_run=keypoint_run,
                refined_run=refined_run,
                dataset_id=dataset_id,
                recording_id=recording_id,
                zarr_use=resolved_zarr_use,
            )
            return BackfillRow(
                zarr_path=zarr_path,
                status="updated",
                run_name=result.run_name,
                source_keypoint_path=result.source_keypoint_path,
            )
        summary = build_keypoint_profile_summary(
            root,
            zarr_path=zarr_path,
            source_keypoint_path=source_keypoint_path,
            keypoint_run=keypoint_run,
            refined_run=refined_run,
            dataset_id=dataset_id,
            recording_id=recording_id,
            zarr_use=resolved_zarr_use,
        )
        computed_run_name = run_name or _default_run_name_from_created(summary.get("created_at_utc"))
        return BackfillRow(
            zarr_path=zarr_path,
            status="would_write",
            run_name=computed_run_name,
            source_keypoint_path=str(summary.get("source", {}).get("keypoint_path") or ""),
        )
    except KeypointSourceError as exc:
        return BackfillRow(
            zarr_path=zarr_path,
            status="missing_source",
            reason=str(exc),
        )
    except (KeypointProfileError, FileExistsError, ValueError) as exc:
        return BackfillRow(
            zarr_path=zarr_path,
            status="error",
            reason=str(exc),
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill canonical keypoint profile summaries into "
            "analysis/keypoint_profile_runs/<run>/attrs['profile_summary']."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--run-name", type=str, help="Explicit profile run name (default: timestamped).")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow replacing an existing run with the same --run-name.",
    )
    parser.add_argument(
        "--source-keypoint-path",
        type=str,
        help="Optional explicit source path (e.g. keypoints_runs/<run> or refined_keypoints_runs/<run>).",
    )
    parser.add_argument("--keypoint-run", type=str, help="Optional keypoint run override.")
    parser.add_argument("--refined-run", type=str, help="Optional refined keypoint run override.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path for dataset identity enrichment.")
    parser.add_argument("--apply", action="store_true", help="Write profile groups (default: dry-run).")
    args = parser.parse_args(argv)

    roots = _resolve_roots(list(args.paths))
    rows: list[BackfillRow] = []
    registry = Registry(args.registry) if args.registry is not None else None
    try:
        for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
            context = _lookup_registry_dataset_context(registry, zarr_path)
            row = _process_zarr_path(
                zarr_path,
                apply=bool(args.apply),
                zarr_use_filter=str(args.zarr_use),
                run_name=args.run_name,
                overwrite_existing=bool(args.overwrite_existing),
                source_keypoint_path=args.source_keypoint_path,
                keypoint_run=args.keypoint_run,
                refined_run=args.refined_run,
                dataset_id=context.dataset_id,
                recording_id=context.recording_id,
                dataset_zarr_use=context.zarr_use,
            )
            rows.append(row)
            print(_format_row(row))
    finally:
        if registry is not None:
            registry.close()

    if not rows:
        print("No zarr files found.")
        return 1

    counts = {
        "zarr_scanned": len(rows),
        "updated": 0,
        "would_write": 0,
        "skipped_existing": 0,
        "filtered_zarr_use": 0,
        "missing_source": 0,
        "error": 0,
    }
    for row in rows:
        if row.status in counts:
            counts[row.status] += 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Keypoint profile backfill: "
        f"mode={mode} scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"filtered_zarr_use={counts['filtered_zarr_use']} missing_source={counts['missing_source']} "
        f"errors={counts['error']}"
    )
    print(
        f"Results: updated={counts['updated']} would_write={counts['would_write']} "
        f"skipped_existing={counts['skipped_existing']}"
    )

    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
