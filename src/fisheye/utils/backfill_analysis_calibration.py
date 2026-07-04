#!/usr/bin/env python3
"""Backfill normalized ``analysis/calibration`` groups from recording H5 files.

Default mode is dry-run. Use ``--apply`` to write updates. The intended batch
target is ``/nvme1/recordings``; each analysis archive is matched to its
stimulus H5 via ``analysis/stimulus_runs/<run>.attrs["source_h5"]`` when
available, then by the affiliated ``raw/*.h5`` file under the recording folder.
"""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import zarr

from fisheye.analysis.import_stimulus_to_zarr import _materialize_analysis_calibration


LEGACY_CALIBRATION_ATTRS = (
    "measured_stimulus_fps",
    "measured_fps",
    "camera_id",
    "camera_model",
    "water_depth_mm",
)


@dataclass(frozen=True)
class BackfillPlan:
    zarr_path: Path
    status: str
    h5_path: Optional[Path] = None
    run_name: Optional[str] = None
    message: str = ""


def _resolve_roots(paths: Sequence[Path]) -> list[Path]:
    if paths:
        return [Path(path).expanduser() for path in paths]
    return [Path("/nvme1/recordings")]


def _open_zarr(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, consolidated=False)


def _group_keys(group: zarr.Group) -> list[str]:
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(key) for key in keys_fn())
        except Exception:
            pass
    return sorted(str(key) for key in group.keys() if isinstance(group.get(key), zarr.Group))


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = root.attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _get_analysis_calibration(root: zarr.Group) -> Optional[zarr.Group]:
    analysis = root.get("analysis")
    if analysis is None:
        return None
    calibration = analysis.get("calibration")
    return calibration if isinstance(calibration, zarr.Group) else None


def _analysis_calibration_complete(root: zarr.Group) -> bool:
    calibration = _get_analysis_calibration(root)
    if calibration is None:
        return False
    attrs = calibration.attrs
    has_source = attrs.get("source_h5") not in (None, "")
    has_run = attrs.get("source_stimulus_run") not in (None, "")
    has_scale = attrs.get("pixel_to_mm") is not None or attrs.get("pixels_per_mm_camera") is not None
    has_homography = "homography_matrix" in calibration or attrs.get("homography_status") not in (None, "")
    return bool(has_source and has_run and has_scale and has_homography)


def _select_stimulus_run(
    root: zarr.Group,
    *,
    requested_run: Optional[str] = None,
) -> tuple[Optional[str], Optional[zarr.Group]]:
    analysis = root.get("analysis")
    if analysis is None:
        return None, None
    runs_parent = analysis.get("stimulus_runs")
    if runs_parent is None:
        return None, None

    if requested_run:
        if requested_run in runs_parent:
            run = runs_parent[requested_run]
            return requested_run, run if isinstance(run, zarr.Group) else None
        return requested_run, None

    names: list[str] = []
    latest = runs_parent.attrs.get("latest")
    if latest is not None:
        names.append(str(latest))
    for name in reversed(_group_keys(runs_parent)):
        if name not in names:
            names.append(name)

    first_available: tuple[Optional[str], Optional[zarr.Group]] = (None, None)
    for name in names:
        if name not in runs_parent:
            continue
        run = runs_parent[name]
        if not isinstance(run, zarr.Group):
            continue
        if first_available == (None, None):
            first_available = (name, run)
        if run.attrs.get("source_h5") not in (None, ""):
            return name, run
    return first_available


def _source_h5_from_existing_calibration(root: zarr.Group) -> Optional[Path]:
    calibration = _get_analysis_calibration(root)
    if calibration is None:
        return None
    source = calibration.attrs.get("source_h5")
    if source in (None, ""):
        return None
    path = Path(str(source)).expanduser()
    return path if path.exists() and path.is_file() else None


def _source_h5_from_stimulus_run(run_group: Optional[zarr.Group]) -> Optional[Path]:
    if run_group is None:
        return None
    source = run_group.attrs.get("source_h5")
    if source in (None, ""):
        return None
    path = Path(str(source)).expanduser()
    return path if path.exists() and path.is_file() else None


def _recording_dir_for_zarr(zarr_path: Path) -> Optional[Path]:
    try:
        parent = zarr_path.resolve().parent
    except OSError:
        parent = zarr_path.parent
    if parent.name == "zarr":
        return parent.parent
    return None


def _source_h5_from_recording_raw(zarr_path: Path) -> tuple[Optional[Path], str]:
    recording_dir = _recording_dir_for_zarr(zarr_path)
    if recording_dir is None:
        return None, "zarr is not under a recording/zarr directory"

    raw_dir = recording_dir / "raw"
    candidates = sorted(path for suffix in ("*.h5", "*.hdf5") for path in raw_dir.glob(suffix))
    if not candidates:
        return None, f"no H5 files found under {raw_dir}"

    zarr_stem = zarr_path.name
    recording_stem = None
    if zarr_stem.endswith("_analysis.zarr"):
        recording_stem = zarr_stem[: -len("_analysis.zarr")]
    elif zarr_stem.endswith("_training.zarr"):
        recording_stem = zarr_stem[: -len("_training.zarr")]
    if recording_stem:
        preferred = [path for path in candidates if path.stem == recording_stem]
        if len(preferred) == 1:
            return preferred[0], ""

    if len(candidates) == 1:
        return candidates[0], ""

    joined = ", ".join(str(path) for path in candidates[:5])
    suffix = "" if len(candidates) <= 5 else f", ... ({len(candidates)} total)"
    return None, f"ambiguous H5 candidates: {joined}{suffix}"


def _resolve_source_h5(
    zarr_path: Path,
    root: zarr.Group,
    run_group: Optional[zarr.Group],
    *,
    explicit_h5_path: Optional[Path],
) -> tuple[Optional[Path], str]:
    if explicit_h5_path is not None:
        path = explicit_h5_path.expanduser()
        if path.exists() and path.is_file():
            return path, ""
        return None, f"explicit H5 path does not exist: {path}"

    for candidate in (
        _source_h5_from_stimulus_run(run_group),
        _source_h5_from_existing_calibration(root),
    ):
        if candidate is not None:
            return candidate, ""

    return _source_h5_from_recording_raw(zarr_path)


def _h5_has_calibration_snapshot(h5_path: Path) -> bool:
    with h5py.File(h5_path, "r") as h5:
        return "calibration_snapshot" in h5


def _copy_legacy_calibration_attrs(root: zarr.Group, *, overwrite_existing: bool) -> None:
    analysis_calibration = _get_analysis_calibration(root)
    legacy_calibration = root.get("calibration")
    if analysis_calibration is None or legacy_calibration is None:
        return
    for key in LEGACY_CALIBRATION_ATTRS:
        if key not in legacy_calibration.attrs:
            continue
        if not overwrite_existing and key in analysis_calibration.attrs:
            continue
        analysis_calibration.attrs[key] = legacy_calibration.attrs[key]


def plan_or_backfill_one(
    zarr_path: Path,
    *,
    apply: bool,
    overwrite_existing: bool,
    requested_run: Optional[str] = None,
    explicit_h5_path: Optional[Path] = None,
    zarr_use: str = "analysis",
    consolidate_metadata: bool = False,
) -> BackfillPlan:
    mode = "a" if apply else "r"
    root = _open_zarr(zarr_path, mode=mode)

    observed_use = _infer_zarr_use(root, zarr_path)
    if zarr_use in {"analysis", "training"} and observed_use != zarr_use:
        return BackfillPlan(zarr_path=zarr_path, status="filtered_zarr_use", message=str(observed_use))

    existing_complete = _analysis_calibration_complete(root)
    if existing_complete and not overwrite_existing:
        return BackfillPlan(zarr_path=zarr_path, status="skipped_existing")

    run_name, run_group = _select_stimulus_run(root, requested_run=requested_run)
    if requested_run and run_group is None:
        return BackfillPlan(zarr_path=zarr_path, status="missing_stimulus_run", run_name=requested_run)

    h5_path, h5_message = _resolve_source_h5(
        zarr_path,
        root,
        run_group,
        explicit_h5_path=explicit_h5_path,
    )
    if h5_path is None:
        status = "ambiguous_h5" if h5_message.startswith("ambiguous") else "missing_h5"
        return BackfillPlan(zarr_path=zarr_path, status=status, run_name=run_name, message=h5_message)

    try:
        has_calibration_snapshot = _h5_has_calibration_snapshot(h5_path)
    except Exception as exc:
        return BackfillPlan(
            zarr_path=zarr_path,
            status="h5_error",
            h5_path=h5_path,
            run_name=run_name,
            message=str(exc),
        )
    if not has_calibration_snapshot:
        return BackfillPlan(
            zarr_path=zarr_path,
            status="missing_calibration_snapshot",
            h5_path=h5_path,
            run_name=run_name,
        )

    planned_status = "would_overwrite" if existing_complete else "would_backfill"
    written_status = "overwritten" if existing_complete else "backfilled"
    source_run_name = run_name or "unknown"
    if not apply:
        return BackfillPlan(
            zarr_path=zarr_path,
            status=planned_status,
            h5_path=h5_path,
            run_name=source_run_name,
        )

    with h5py.File(h5_path, "r") as h5:
        _materialize_analysis_calibration(
            root,
            h5,
            source_h5=h5_path,
            run_name=source_run_name,
            console=None,
        )
    _copy_legacy_calibration_attrs(root, overwrite_existing=overwrite_existing)
    if consolidate_metadata:
        zarr.consolidate_metadata(str(zarr_path))

    return BackfillPlan(
        zarr_path=zarr_path,
        status=written_status,
        h5_path=h5_path,
        run_name=source_run_name,
    )


def _print_result(result: BackfillPlan, *, verbose: bool) -> None:
    noisy_statuses = {
        "would_backfill",
        "would_overwrite",
        "backfilled",
        "overwritten",
        "ambiguous_h5",
        "missing_h5",
        "missing_calibration_snapshot",
        "h5_error",
        "missing_stimulus_run",
    }
    if not verbose and result.status not in noisy_statuses:
        return

    parts = [result.status, str(result.zarr_path)]
    if result.h5_path is not None:
        parts.append(f"h5={result.h5_path}")
    if result.run_name:
        parts.append(f"run={result.run_name}")
    if result.message:
        parts.append(result.message)
    print(": ".join(parts[:2]) + (" " + " ".join(parts[2:]) if len(parts) > 2 else ""))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or analysis Zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Archive scope (default: analysis). Use training or any to process training archives.",
    )
    parser.add_argument("--run-name", help="Use a specific analysis/stimulus_runs/<run> for provenance.")
    parser.add_argument("--h5-path", type=Path, help="Explicit H5 path. Intended for single-archive repairs.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Rewrite complete analysis/calibration groups.")
    parser.add_argument("--apply", action="store_true", help="Write updates (default: dry-run).")
    parser.add_argument(
        "--consolidate-metadata",
        action="store_true",
        help="After each write, refresh Zarr consolidated metadata.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print skipped archives as well as candidates/errors.")
    args = parser.parse_args(argv)

    if args.consolidate_metadata and not args.apply:
        parser.error("--consolidate-metadata requires --apply")
    if args.h5_path is not None and len(args.paths) != 1:
        parser.error("--h5-path requires exactly one zarr path")

    roots = _resolve_roots(args.paths)
    counts: dict[str, int] = {
        "zarr_scanned": 0,
        "errors": 0,
    }

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            result = plan_or_backfill_one(
                zarr_path,
                apply=bool(args.apply),
                overwrite_existing=bool(args.overwrite_existing),
                requested_run=args.run_name,
                explicit_h5_path=args.h5_path,
                zarr_use=str(args.zarr_use),
                consolidate_metadata=bool(args.consolidate_metadata),
            )
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")
            continue

        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status in {"h5_error"}:
            counts["errors"] += 1
        _print_result(result, verbose=bool(args.verbose))

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    summary_keys = [
        "zarr_scanned",
        "would_backfill",
        "would_overwrite",
        "backfilled",
        "overwritten",
        "skipped_existing",
        "filtered_zarr_use",
        "missing_h5",
        "ambiguous_h5",
        "missing_stimulus_run",
        "missing_calibration_snapshot",
        "h5_error",
        "errors",
    ]
    summary = " ".join(f"{key}={counts.get(key, 0)}" for key in summary_keys)
    print(f"Analysis calibration backfill {mode}: {summary}")
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
