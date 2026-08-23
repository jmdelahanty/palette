"""Plan or publish one exact provider-aware chaser position suite.

The command recomputes the position-only summary from exact immutable source
authorities.  It returns a revealing dry-run plan by default.  ``--apply`` is
required to atomically publish typed row tables below the analysis Zarr.  It
never updates a selector or registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.provider_chaser_position_suite_publication import (
    build_provider_chaser_position_suite_publication_plan,
    publish_provider_chaser_position_suite_run,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.utils.materialize_provider_chaser_position_suite_canary import (
    _parse_epoch_role,
    _parse_float_list,
    build_canary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--provider-run", required=True)
    parser.add_argument("--geometry-selection-run", required=True)
    parser.add_argument("--expected-selection-record-sha256", required=True)
    parser.add_argument("--expected-physical-authority-sha256", required=True)
    parser.add_argument(
        "--epoch-role",
        type=_parse_epoch_role,
        action="append",
        required=True,
        help="Explicit analysis-role binding in ROLE=WINDOW_ID form; repeat as needed.",
    )
    parser.add_argument("--treatment-role", default="aggressive")
    parser.add_argument("--baseline-role", default="inert")
    parser.add_argument("--radial-bin-width-mm", type=float, default=2.0)
    parser.add_argument("--cdf-thresholds-mm", type=_parse_float_list)
    parser.add_argument("--near-zone-radius-mm", type=float, default=5.0)
    parser.add_argument("--near-entry-radius-mm", type=float, default=5.0)
    parser.add_argument("--near-exit-radius-mm", type=float, default=6.0)
    parser.add_argument("--perimeter-band-mm", type=float, default=5.0)
    parser.add_argument("--min-expected-count", type=float, default=5.0)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish the immutable selector-ineligible run; default is dry-run.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_canary(
        args.analysis_zarr,
        provider_run=args.provider_run,
        geometry_selection_run=args.geometry_selection_run,
        expected_selection_record_sha256=args.expected_selection_record_sha256,
        expected_physical_authority_sha256=args.expected_physical_authority_sha256,
        epoch_role_bindings=args.epoch_role,
        treatment_role=args.treatment_role,
        baseline_role=args.baseline_role,
        radial_bin_width_mm=args.radial_bin_width_mm,
        cdf_thresholds_mm=args.cdf_thresholds_mm,
        near_zone_radius_mm=args.near_zone_radius_mm,
        near_entry_radius_mm=args.near_entry_radius_mm,
        near_exit_radius_mm=args.near_exit_radius_mm,
        perimeter_band_mm=args.perimeter_band_mm,
        min_expected_count=args.min_expected_count,
    )
    plan = build_provider_chaser_position_suite_publication_plan(
        args.analysis_zarr,
        report=report,
        run_name=args.run_name,
        expected_recording_id=args.expected_recording_id,
    )
    result = (
        plan.to_json()
        if not args.apply
        else publish_provider_chaser_position_suite_run(
            plan,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
        )
    )
    if args.output_json is not None:
        write_json_atomic(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
