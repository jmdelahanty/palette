"""Plan or publish one coordinate-safe proxy-derived relative-frame candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping
from typing import Sequence

from fisheye.analysis_workflows.core_chaser_relative_frame_adapter import (
    prepare_core_proxy_chaser_relative_frame,
)
from fisheye.analysis_workflows.chaser_proxy_relative_frame_adapter import (
    prepare_proxy_relative_frame,
)
from fisheye.analysis_workflows.materializers.chaser_relative_frame import (
    build_chaser_relative_frame_materialization_plan,
    materialize_chaser_relative_frame,
)


def _default_profile_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "analysis"
        / "profiles"
        / "chaser_behavior_full_v3.yaml"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transform one exact selector-ineligible chaser proxy from arena "
            "coordinates into the exact source-camera frame and materialize the "
            "common relative-frame candidate."
        )
    )
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--proxy-run-name", required=True)
    parser.add_argument("--output-run-name", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument(
        "--analysis-profile",
        type=Path,
        default=_default_profile_path(),
    )
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--expected-proxy-manifest-sha256")
    parser.add_argument("--expected-subject-metadata-sha256")
    parser.add_argument("--expected-timing-authority-sha256")
    parser.add_argument(
        "--core-authority-roster",
        type=Path,
        help=(
            "Exact sealed core-authority roster JSON. Supplying this selects "
            "canonical motion/body mode; failure never falls back to the legacy "
            "proxy fish or body authorities."
        ),
    )
    parser.add_argument("--expected-core-authority-roster-sha256")
    parser.add_argument("--core-track-id", type=int)
    parser.add_argument(
        "--body-frame-run",
        help=(
            "Exact selector-ineligible analysis/body_frame_runs child to "
            "project by acquisition-frame identity. Omit for a position-only run."
        ),
    )
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish the named immutable candidate; otherwise emit a dry-run plan.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    archive = args.analysis_zarr.expanduser().resolve()
    core_arguments = (
        args.core_authority_roster,
        args.expected_core_authority_roster_sha256,
        args.core_track_id,
    )
    if any(value is not None for value in core_arguments):
        if any(value is None for value in core_arguments):
            raise ValueError(
                "Core mode requires roster path, expected roster digest, and "
                "one explicit track ID."
            )
        if args.body_frame_run is not None:
            raise ValueError(
                "Core mode cannot accept a legacy body-frame run or fallback."
            )
        roster_path = args.core_authority_roster.expanduser().resolve()
        raw_roster = json.loads(roster_path.read_text(encoding="utf-8"))
        if not isinstance(raw_roster, Mapping):
            raise ValueError("Core-authority roster JSON must contain one object.")
        roster = dict(raw_roster)
        if roster.get("record_sha256") != args.expected_core_authority_roster_sha256:
            raise ValueError("Core-authority roster digest differs from the task.")
        bound = prepare_core_proxy_chaser_relative_frame(
            archive,
            core_authority_roster=roster,
            core_track_id=args.core_track_id,
            proxy_run_name=args.proxy_run_name,
            analysis_profile_path=args.analysis_profile,
            expected_recording_id=args.expected_recording_id,
            expected_proxy_manifest_sha256=args.expected_proxy_manifest_sha256,
            expected_subject_metadata_sha256=args.expected_subject_metadata_sha256,
            expected_timing_authority_sha256=args.expected_timing_authority_sha256,
        )
    else:
        bound = prepare_proxy_relative_frame(
            archive,
            proxy_run_name=args.proxy_run_name,
            analysis_profile_path=args.analysis_profile,
            expected_recording_id=args.expected_recording_id,
            expected_proxy_manifest_sha256=args.expected_proxy_manifest_sha256,
            expected_subject_metadata_sha256=args.expected_subject_metadata_sha256,
            expected_timing_authority_sha256=args.expected_timing_authority_sha256,
            body_frame_run_name=args.body_frame_run,
        )
    if args.apply:
        publication = materialize_chaser_relative_frame(
            archive,
            prepared=bound.prepared,
            scratch_root=args.scratch_root,
            run_name=args.output_run_name,
            copy_backend=args.copy_backend,
            apply=True,
        )
        return {
            **bound.to_json(),
            "status": "published_selector_ineligible",
            "publication": publication,
        }
    plan = build_chaser_relative_frame_materialization_plan(
        archive,
        scratch_root=args.scratch_root,
        run_name=args.output_run_name,
        prepared=bound.prepared,
    )
    return {
        **bound.to_json(),
        "status": "planned_no_writes",
        "plan": plan.to_json(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            result,
            sort_keys=True if args.json else False,
            indent=None if args.json else 2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
