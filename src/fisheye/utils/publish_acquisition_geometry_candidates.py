"""Plan or publish producer-native or recovered acquisition-geometry candidates."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    plan_producer_native_acquisition_geometry_candidate,
    plan_recovered_acquisition_geometry_candidate,
    publish_arena_geometry_candidate,
    validate_arena_geometry_candidate_run,
)
from fisheye.shared.recording_geometry import RecordingGeometryError
from fisheye.shared.recording_geometry_recovery import RECOVERY_RECEIPT_NAME
from fisheye.shared.json_safety import write_json_atomic


def _analysis_zarr(recording: Path) -> Path:
    candidates = sorted((recording / "zarr").glob("*_analysis.zarr"))
    if len(candidates) != 1:
        raise RecordingGeometryError(
            f"Expected exactly one analysis Zarr under {recording / 'zarr'}; "
            f"found {len(candidates)}."
        )
    return candidates[0].resolve()


def _explicit_analysis_zarr(recording: Path, path: Path) -> Path:
    resolved = path.expanduser().resolve()
    expected_parent = (recording / "zarr").resolve()
    if resolved.parent != expected_parent or not resolved.name.endswith(
        "_analysis.zarr"
    ):
        raise RecordingGeometryError(
            "Explicit analysis Zarr must be an *_analysis.zarr child of the "
            "recording's zarr directory."
        )
    if not (resolved / "zarr.json").is_file():
        raise RecordingGeometryError(
            f"Explicit analysis Zarr is not Zarr v3: {resolved}"
        )
    return resolved


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
        "--analysis-zarr",
        type=Path,
        action="append",
        help=(
            "Exact analysis Zarr paired by position with each --recording. "
            "When omitted, each recording must contain exactly one analysis Zarr."
        ),
    )
    parser.add_argument(
        "--geometry-source",
        choices=("producer-folder", "citrus-h5", "recovery-receipt"),
        default="recovery-receipt",
        help=(
            "Authoritative geometry adapter. recovery-receipt is the explicit "
            "historical compatibility route."
        ),
    )
    parser.add_argument(
        "--camera-serial",
        action="append",
        help="Exact camera serial paired by position with each producer-native recording.",
    )
    parser.add_argument(
        "--arena-id",
        action="append",
        help="Exact arena ID paired by position with each producer-native recording.",
    )
    parser.add_argument(
        "--citrus-h5",
        type=Path,
        action="append",
        help="Exact recording-bound Citrus H5 paired with each --recording.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp"))
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
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optional immutable summary receipt for DAG orchestration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    explicit_zarrs = tuple(args.analysis_zarr or ())
    if explicit_zarrs and len(explicit_zarrs) != len(args.recording):
        raise RecordingGeometryError(
            "--analysis-zarr must be omitted or repeated once per --recording."
        )
    camera_serials = tuple(args.camera_serial or ())
    arena_ids = tuple(args.arena_id or ())
    citrus_h5s = tuple(args.citrus_h5 or ())
    producer_native = args.geometry_source in {"producer-folder", "citrus-h5"}
    if producer_native and (
        len(camera_serials) != len(args.recording)
        or len(arena_ids) != len(args.recording)
    ):
        raise RecordingGeometryError(
            "Producer-native geometry requires one --camera-serial and --arena-id "
            "for every --recording."
        )
    if args.geometry_source == "citrus-h5" and len(citrus_h5s) != len(args.recording):
        raise RecordingGeometryError(
            "citrus-h5 geometry requires one --citrus-h5 for every --recording."
        )
    if args.geometry_source != "citrus-h5" and citrus_h5s:
        raise RecordingGeometryError(
            "--citrus-h5 is only valid with --geometry-source citrus-h5."
        )
    rows: list[dict[str, object]] = []
    for index, raw_recording in enumerate(args.recording):
        recording = raw_recording.expanduser().resolve()
        source_zarr = (
            _explicit_analysis_zarr(recording, explicit_zarrs[index])
            if explicit_zarrs
            else _analysis_zarr(recording)
        )
        if args.geometry_source == "recovery-receipt":
            receipt = recording / "raw" / RECOVERY_RECEIPT_NAME
            plan = plan_recovered_acquisition_geometry_candidate(
                source_zarr=source_zarr,
                receipt_path=receipt,
            )
        else:
            plan = plan_producer_native_acquisition_geometry_candidate(
                source_zarr=source_zarr,
                camera_serial=camera_serials[index],
                arena_id=arena_ids[index],
                recording_folder=(
                    recording if args.geometry_source == "producer-folder" else None
                ),
                citrus_h5=(
                    citrus_h5s[index] if args.geometry_source == "citrus-h5" else None
                ),
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
                "geometry_source": args.geometry_source,
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
    payload = {
        "schema_id": "palette.arena_geometry_acquisition_candidate_batch",
        "schema_version": 1,
        "mode": "apply" if args.apply else "dry_run",
        "target_count": len(rows),
        "targets": rows,
        "operational_selection_performed": False,
        "registry_updated": False,
    }
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
