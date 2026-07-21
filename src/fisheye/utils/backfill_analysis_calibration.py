#!/usr/bin/env python3
"""Inventory legacy ``analysis/calibration`` migration candidates.

This command is read-only.  The former apply path reconstructed one global
calibration group from incomplete historical evidence and could select camera
scale and homography values from different cameras.  Applying that result is
retired; use this command only to identify H5- or donor-backed candidates for an
explicit, evidence-validated migration.

Each analysis archive is matched to its stimulus H5 via
``analysis/stimulus_runs/<run>.attrs["source_h5"]`` when available, then by the
affiliated ``raw/*.h5`` file under the recording folder.  ``--donor-zarr`` only
inspects the named donor and never authorizes or performs a copy.
"""

from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

import h5py
import zarr

APPLY_RETIRED_MESSAGE = (
    "Legacy global analysis/calibration writes are retired because the available "
    "metadata does not prove one camera, transform direction, reference extent, "
    "and calibration lineage. Run without --apply for read-only candidate inventory; "
    "publish any validated migration as a new lineage-bound run."
)
H5_CANDIDATE_STATUS = "h5_candidate_requires_explicit_migration"
H5_OVERWRITE_CANDIDATE_STATUS = "h5_overwrite_candidate_requires_explicit_migration"
DONOR_CANDIDATE_STATUS = "donor_candidate_requires_explicit_migration"
DONOR_OVERWRITE_CANDIDATE_STATUS = "donor_overwrite_candidate_requires_explicit_migration"


class AnalysisCalibrationApplyRetiredError(RuntimeError):
    """Raised before any archive open when the retired apply path is requested."""


@dataclass(frozen=True)
class BackfillPlan:
    zarr_path: Path
    status: str
    h5_path: Optional[Path] = None
    donor_zarr_path: Optional[Path] = None
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


def _get_analysis_calibration(root: zarr.Group) -> Optional[zarr.Group]:
    analysis = root.get("analysis")
    if analysis is None:
        return None
    calibration = analysis.get("calibration")
    return calibration if isinstance(calibration, zarr.Group) else None


def _get_calibration_group(root: zarr.Group) -> Optional[zarr.Group]:
    calibration = _get_analysis_calibration(root)
    if calibration is not None:
        return calibration
    legacy = root.get("calibration")
    return legacy if isinstance(legacy, zarr.Group) else None


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


def _calibration_has_usable_scale(calibration: zarr.Group) -> bool:
    attrs = calibration.attrs
    return any(
        attrs.get(key) is not None
        for key in ("pixel_to_mm", "pixels_per_mm_camera", "pixels_per_mm")
    )


def _source_label_for_calibration(root: zarr.Group, calibration: zarr.Group) -> str:
    source = calibration.attrs.get("source")
    source_h5 = calibration.attrs.get("source_h5")
    source_run = calibration.attrs.get("source_stimulus_run")
    zarr_id = root.attrs.get("recording_id") or root.attrs.get("recording_name")
    parts = []
    if zarr_id not in (None, ""):
        parts.append(f"recording={zarr_id}")
    if source not in (None, ""):
        parts.append(f"source={source}")
    if source_h5 not in (None, ""):
        parts.append(f"source_h5={source_h5}")
    if source_run not in (None, ""):
        parts.append(f"source_stimulus_run={source_run}")
    return "; ".join(parts)


def _donor_calibration_plan(
    zarr_path: Path,
    *,
    root: zarr.Group,
    donor_zarr_path: Path,
    overwrite_existing: bool,
) -> BackfillPlan:
    existing_complete = _analysis_calibration_complete(root)
    if existing_complete and not overwrite_existing:
        return BackfillPlan(zarr_path=zarr_path, status="skipped_existing", donor_zarr_path=donor_zarr_path)

    donor_root = _open_zarr(donor_zarr_path.expanduser(), mode="r")
    donor_calibration = _get_calibration_group(donor_root)
    if donor_calibration is None:
        return BackfillPlan(
            zarr_path=zarr_path,
            status="missing_donor_calibration",
            donor_zarr_path=donor_zarr_path,
        )
    if not _calibration_has_usable_scale(donor_calibration):
        return BackfillPlan(
            zarr_path=zarr_path,
            status="donor_calibration_missing_scale",
            donor_zarr_path=donor_zarr_path,
        )

    return BackfillPlan(
        zarr_path=zarr_path,
        status=(
            DONOR_OVERWRITE_CANDIDATE_STATUS
            if existing_complete
            else DONOR_CANDIDATE_STATUS
        ),
        donor_zarr_path=donor_zarr_path,
        message=_source_label_for_calibration(donor_root, donor_calibration),
    )


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


def plan_or_backfill_one(
    zarr_path: Path,
    *,
    apply: bool,
    overwrite_existing: bool,
    requested_run: Optional[str] = None,
    explicit_h5_path: Optional[Path] = None,
    donor_zarr_path: Optional[Path] = None,
    donor_note: Optional[str] = None,
    zarr_use: str = "analysis",
    consolidate_metadata: bool = False,
) -> BackfillPlan:
    """Inspect one archive, rejecting retired writes before any archive open."""

    if apply:
        raise AnalysisCalibrationApplyRetiredError(APPLY_RETIRED_MESSAGE)
    # Retain legacy keyword compatibility for read-only callers.  Neither value
    # authorizes a write or changes candidate classification.
    del donor_note, consolidate_metadata

    root = _open_zarr(zarr_path, mode="r")

    observed_use = _infer_zarr_use(root, zarr_path)
    if zarr_use in {"analysis", "training"} and observed_use != zarr_use:
        return BackfillPlan(zarr_path=zarr_path, status="filtered_zarr_use", message=str(observed_use))

    if donor_zarr_path is not None:
        return _donor_calibration_plan(
            zarr_path,
            root=root,
            donor_zarr_path=donor_zarr_path,
            overwrite_existing=overwrite_existing,
        )

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

    planned_status = (
        H5_OVERWRITE_CANDIDATE_STATUS
        if existing_complete
        else H5_CANDIDATE_STATUS
    )
    source_run_name = run_name or "unknown"
    return BackfillPlan(
        zarr_path=zarr_path,
        status=planned_status,
        h5_path=h5_path,
        run_name=source_run_name,
    )


def _print_result(result: BackfillPlan, *, verbose: bool) -> None:
    noisy_statuses = {
        H5_CANDIDATE_STATUS,
        H5_OVERWRITE_CANDIDATE_STATUS,
        DONOR_CANDIDATE_STATUS,
        DONOR_OVERWRITE_CANDIDATE_STATUS,
        "ambiguous_h5",
        "missing_h5",
        "missing_calibration_snapshot",
        "missing_donor_calibration",
        "donor_calibration_missing_scale",
        "h5_error",
        "missing_stimulus_run",
    }
    if not verbose and result.status not in noisy_statuses:
        return

    parts = [result.status, str(result.zarr_path)]
    if result.h5_path is not None:
        parts.append(f"h5={result.h5_path}")
    if result.donor_zarr_path is not None:
        parts.append(f"donor_zarr={result.donor_zarr_path}")
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
    parser.add_argument(
        "--donor-zarr",
        type=Path,
        help=(
            "Inspect calibration from a possible donor Zarr. This read-only inventory "
            "does not authorize or perform a copy."
        ),
    )
    parser.add_argument(
        "--donor-note",
        help="Legacy compatibility option; retained for read-only donor inventory.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Classify an existing global calibration as requiring replacement; never rewrite it.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Retired: requests fail closed without opening an archive for writing.",
    )
    parser.add_argument(
        "--consolidate-metadata",
        action="store_true",
        help="Retired with the legacy write path.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print skipped archives as well as candidates/errors.")
    args = parser.parse_args(argv)

    if args.h5_path is not None and len(args.paths) != 1:
        parser.error("--h5-path requires exactly one zarr path")
    if args.h5_path is not None and args.donor_zarr is not None:
        parser.error("--h5-path and --donor-zarr are mutually exclusive")
    if args.donor_note and args.donor_zarr is None:
        parser.error("--donor-note requires --donor-zarr")
    if args.apply:
        print(f"error: {APPLY_RETIRED_MESSAGE}", file=sys.stderr)
        return 2
    if args.consolidate_metadata:
        parser.error("--consolidate-metadata belongs to the retired --apply path")

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
                donor_zarr_path=args.donor_zarr,
                donor_note=args.donor_note,
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

    summary_keys = [
        "zarr_scanned",
        H5_CANDIDATE_STATUS,
        H5_OVERWRITE_CANDIDATE_STATUS,
        DONOR_CANDIDATE_STATUS,
        DONOR_OVERWRITE_CANDIDATE_STATUS,
        "skipped_existing",
        "filtered_zarr_use",
        "missing_h5",
        "ambiguous_h5",
        "missing_stimulus_run",
        "missing_calibration_snapshot",
        "missing_donor_calibration",
        "donor_calibration_missing_scale",
        "h5_error",
        "errors",
    ]
    summary = " ".join(f"{key}={counts.get(key, 0)}" for key in summary_keys)
    print(f"Analysis calibration read-only inventory: {summary}")
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
