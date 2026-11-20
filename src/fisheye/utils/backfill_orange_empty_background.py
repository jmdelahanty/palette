"""Add Orange's sealed empty-dish background to recordings imported without it.

Import stores the capture as a ``background_runs`` entry
(``import_recording_analysis.import_empty_background``), but only on a first
import. Recordings imported before that step existed keep the sealed capture
in ``raw/acquisition/empty_background`` and have no run. This tool writes the
same run into their published archives.

Default is a dry run: it reads the sealed capture through the contract loader
and reports what would happen. ``--apply`` writes, under the archive
publication lock, then consolidates the archive's metadata again and checks
that the consolidated view selects the new run.

It does not touch the registry, the recording manifest or the import receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import zarr

from fisheye.preprocessing.orange_empty_background_import import (
    RECORD_ATTR,
    STATUS_ALREADY_IMPORTED,
    STATUS_IMPORTED,
    STATUS_REFUSED,
    run_name_for,
)
from fisheye.shared.orange_empty_background import (
    EmptyBackgroundError,
    load_empty_background,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.utils.import_recording_analysis import (
    RecordingAnalysisPlan,
    import_empty_background,
    sealed_empty_background_inputs,
)

TOOL_NAME = "backfill_orange_empty_background"

PLAN_WOULD_IMPORT = "would_import"
PLAN_ALREADY_IMPORTED = STATUS_ALREADY_IMPORTED
PLAN_NO_SEALED_SNAPSHOT = "no_sealed_start_snapshot"
PLAN_CONFLICT = "conflict"


def analysis_zarr_path(recording_dir: Path) -> Path:
    path = recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"
    if not (path / "zarr.json").is_file():
        raise ValueError(f"no analysis archive at {path}")
    return path


def plan_recording(recording_dir: Path) -> dict[str, Any]:
    """Report, without writing, what a backfill of this recording would do."""

    recording_dir = Path(recording_dir)
    zarr_path = analysis_zarr_path(recording_dir)
    plan: dict[str, Any] = {
        "recording": recording_dir.name,
        "recording_dir": str(recording_dir),
        "zarr_path": str(zarr_path),
    }
    inputs = sealed_empty_background_inputs(recording_dir)
    if inputs is None:
        return {**plan, "status": PLAN_NO_SEALED_SNAPSHOT}
    acquisition_dir, camera_serial, snapshot = inputs
    plan["camera_serial"] = camera_serial
    try:
        background = load_empty_background(
            acquisition_dir, camera_serial, start_snapshot=snapshot
        )
    except EmptyBackgroundError as exc:
        return {**plan, "status": STATUS_REFUSED, "code": exc.code, "detail": str(exc)[:500]}
    if not background.usable:
        return {**plan, "status": background.status, "reason": background.reason}
    run_name = run_name_for(str(background.capture_id))
    plan.update(
        run_name=run_name,
        capture_id=background.capture_id,
        image_sha256=background.image_sha256,
    )
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    runs = root["background_runs"] if "background_runs" in root else None
    existing = sorted(runs.group_keys()) if runs is not None else []
    plan["existing_background_runs"] = existing
    if run_name not in existing:
        return {**plan, "status": PLAN_WOULD_IMPORT}
    held = dict(runs[run_name].attrs.get("orange_empty_background") or {})
    if held.get("image_sha256") == background.image_sha256:
        return {**plan, "status": PLAN_ALREADY_IMPORTED}
    return {**plan, "status": PLAN_CONFLICT}


def apply_recording(recording_dir: Path) -> dict[str, Any]:
    """Write the run for one recording and publish it in consolidated metadata."""

    recording_dir = Path(recording_dir)
    plan = plan_recording(recording_dir)
    if plan["status"] == PLAN_CONFLICT:
        raise ValueError(
            f"{plan['recording']}: background_runs/{plan['run_name']} holds a different image"
        )
    if plan["status"] not in (PLAN_WOULD_IMPORT, PLAN_ALREADY_IMPORTED):
        # Nothing to store: leave the published archive exactly as it is.
        return {**plan, "applied": False}
    zarr_path = Path(plan["zarr_path"])
    with archive_metadata_publication_lock(zarr_path):
        record = import_empty_background(
            RecordingAnalysisPlan(
                recording_dir=recording_dir,
                h5_path=None,
                cam_video=None,
                zarr_path=zarr_path,
            )
        )
        if not isinstance(record, Mapping) or record.get("status") not in (
            STATUS_IMPORTED,
            STATUS_ALREADY_IMPORTED,
        ):
            raise ValueError(f"{plan['recording']}: backfill wrote no run: {record!r}")
        consolidate_metadata_capture_expected_warnings(zarr_path)
        published = zarr.open_group(str(zarr_path), mode="r", use_consolidated=True)
        run_name = str(record["run_name"])
        runs = published["background_runs"]
        if run_name not in runs or "background_full" not in runs[run_name]:
            raise ValueError(f"{plan['recording']}: consolidated metadata lacks {run_name}")
        if dict(published.attrs.get(RECORD_ATTR) or {}).get("run_name") != run_name:
            raise ValueError(f"{plan['recording']}: consolidated root lacks the import record")
        latest = runs.attrs.get("latest")
    return {
        **plan,
        "status": record["status"],
        "applied": record["status"] == STATUS_IMPORTED,
        "latest": latest,
        "selected_as_latest": latest == run_name,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the runs; without it nothing is changed",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    results: list[dict[str, Any]] = []
    failed = False
    for recording_dir in args.recording_dirs:
        try:
            results.append(
                apply_recording(recording_dir) if args.apply else plan_recording(recording_dir)
            )
        except Exception as exc:  # one bad recording must not hide the others
            failed = True
            results.append(
                {"recording": Path(recording_dir).name, "status": "error", "error": str(exc)[:500]}
            )
    json.dump(
        {"tool": TOOL_NAME, "mode": "apply" if args.apply else "dry_run", "recordings": results},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
