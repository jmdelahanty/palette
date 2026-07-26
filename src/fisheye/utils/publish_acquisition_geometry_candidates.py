"""Plan or atomically publish recovered acquisition-geometry candidates."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    plan_recovered_acquisition_geometry_candidate,
    publish_arena_geometry_candidate,
    validate_arena_geometry_candidate_run,
)
from fisheye.shared.recording_geometry import RecordingGeometryError
from fisheye.shared.recording_geometry_recovery import RECOVERY_RECEIPT_NAME


def _analysis_zarr(recording: Path) -> Path:
    candidates = sorted((recording / "zarr").glob("*_analysis.zarr"))
    if len(candidates) != 1:
        raise RecordingGeometryError(
            f"Expected exactly one analysis Zarr under {recording / 'zarr'}; "
            f"found {len(candidates)}."
        )
    return candidates[0].resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        type=Path,
        action="append",
        required=True,
        help="Recording root with raw recovery receipt and one analysis Zarr; repeatable.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(
            os.environ.get("TMPDIR", "/tmp")
        )
        / "palette-arena-geometry-candidates",
        help="Node/workstation-local temporary publication root.",
    )
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish candidates. Default is a read-only dry-run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows: list[dict[str, object]] = []
    for raw_recording in args.recording:
        recording = raw_recording.expanduser().resolve()
        source_zarr = _analysis_zarr(recording)
        receipt = recording / "raw" / RECOVERY_RECEIPT_NAME
        plan = plan_recovered_acquisition_geometry_candidate(
            source_zarr=source_zarr,
            receipt_path=receipt,
        )
        existing = None
        if plan.target_run_path.exists():
            existing = validate_arena_geometry_candidate_run(
                plan.target_run_path,
                expected_plan=plan,
                require_complete=True,
                require_eligible=True,
            )
            if not existing["valid"]:
                raise RuntimeError(
                    f"Existing candidate is not the planned immutable run: {existing}"
                )
        if args.apply:
            result = publish_arena_geometry_candidate(
                plan,
                scratch_root=args.scratch_root,
                copy_backend=args.copy_backend,
            )
            status = str(result["status"])
            published = bool(result["published"])
        else:
            status = "already_complete" if existing is not None else "dry_run_validated"
            published = False
        arena = plan.candidate_record["arena_binding"]
        assert isinstance(arena, Mapping)
        rows.append(
            {
                "recording": recording.name,
                "camera_serial": arena["camera_serial"],
                "arena_id": arena["arena_id"],
                "candidate_id": plan.candidate_id,
                "candidate_record_sha256": plan.candidate_record_sha256,
                "target_run_path": str(plan.target_run_path),
                "status": status,
                "published": published,
                "operationally_selected": False,
            }
        )
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "target_count": len(rows),
                "targets": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
