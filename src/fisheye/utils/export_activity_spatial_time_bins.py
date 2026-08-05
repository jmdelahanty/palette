"""CLI for the exact activity/spatial time-bin query product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.activity_spatial_time_bins import (
    DEFAULT_ACTIVITY_SPATIAL_SOURCE_WINDOW_ROWS,
    export_activity_spatial_time_bins,
)


def _track_run(value: str) -> tuple[int, str]:
    track_text, separator, run_name = value.partition("=")
    if not separator or not run_name:
        raise argparse.ArgumentTypeError("expected TRACK_ID=SWIM_BOUT_RUN")
    try:
        track_id = int(track_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("track ID must be an integer") from exc
    if track_id < 0 or str(track_id) != track_text:
        raise argparse.ArgumentTypeError("track ID must be canonical and nonnegative")
    return track_id, run_name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one explicit track-motion publication and exact per-track "
            "swim-bout authorities into immutable global time-bin summaries."
        )
    )
    parser.add_argument("zarr", type=Path, help="Recording analysis Zarr.")
    parser.add_argument("--track-kinematics-run", required=True)
    parser.add_argument(
        "--track-scope",
        choices=("online", "offline"),
        default="offline",
    )
    bout = parser.add_mutually_exclusive_group(required=True)
    bout.add_argument(
        "--track-swim-bout-run",
        action="append",
        type=_track_run,
        metavar="TRACK_ID=RUN",
        help="Repeat once for every track in a multi-track source.",
    )
    bout.add_argument(
        "--single-track-swim-bout-run",
        help=(
            "One run for a source proven to contain exactly one track; fails "
            "closed for multi-track sources."
        ),
    )
    parser.add_argument("--bin-size-s", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--export-run-id", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument(
        "--source-window-rows",
        type=int,
        default=DEFAULT_ACTIVITY_SPATIAL_SOURCE_WINDOW_ROWS,
    )
    parser.add_argument("--source-frame-start", type=int)
    parser.add_argument("--source-frame-stop-exclusive", type=int)
    parser.add_argument("--row-group-rows", type=int, default=65_536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_map: dict[int, str] | None = None
    if args.track_swim_bout_run is not None:
        run_map = {}
        for track_id, run_name in args.track_swim_bout_run:
            if track_id in run_map:
                raise SystemExit(f"duplicate swim-bout mapping for track {track_id}")
            run_map[track_id] = run_name
    result = export_activity_spatial_time_bins(
        args.zarr,
        track_kinematics_run=args.track_kinematics_run,
        track_scope=args.track_scope,
        requested_bin_size_s=args.bin_size_s,
        output_root=args.output_root,
        export_run_id=args.export_run_id,
        scratch_root=args.scratch_root,
        swim_bout_runs_by_track=run_map,
        single_track_swim_bout_run=args.single_track_swim_bout_run,
        source_window_rows=args.source_window_rows,
        row_group_rows=args.row_group_rows,
        source_frame_start=args.source_frame_start,
        source_frame_stop_exclusive=args.source_frame_stop_exclusive,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"export_run_id\t{result['export_run_id']}")
        print(f"manifest\t{result['manifest_path']}")
        print(
            "rows\tactivity_spatial_time_bins\t"
            f"{result['row_counts_by_table']['activity_spatial_time_bins']}"
        )


if __name__ == "__main__":
    main()
