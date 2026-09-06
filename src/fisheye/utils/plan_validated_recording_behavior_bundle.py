"""Create or revalidate one read-only validated recording-behavior bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.core_chaser_composite_bundle import (
    CORE_CHASER_BUNDLE_ADAPTER_ID,
    ensure_core_chaser_composite_bundle,
)
from fisheye.analysis_workflows.validated_recording_behavior_bundle import (
    CAPABILITY_KEYS,
    CAPABILITY_STATES,
    REASON_CODES_BY_STATE,
    ensure_validated_recording_behavior_bundle,
)
from fisheye.analysis_workflows.validated_behavior_cohort_adapters import (
    RECORDING_BUNDLE_ADAPTER_ID,
)


def _capability_disposition(value: str) -> tuple[str, dict[str, object]]:
    try:
        capability, disposition = value.split("=", 1)
        state, reason = disposition.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected CAPABILITY=STATE:REASON_CODE"
        ) from exc
    if capability not in CAPABILITY_KEYS:
        raise argparse.ArgumentTypeError(
            f"unknown capability {capability!r}; expected one of "
            + ", ".join(CAPABILITY_KEYS)
        )
    if state not in CAPABILITY_STATES or state == "complete":
        raise argparse.ArgumentTypeError(
            "absent capabilities require unavailable, inapplicable, invalid, "
            "stale, or review_required"
        )
    if reason not in REASON_CODES_BY_STATE[state]:
        expected = ", ".join(sorted(str(item) for item in REASON_CODES_BY_STATE[state]))
        raise argparse.ArgumentTypeError(
            f"reason {reason!r} is invalid for {state!r}; expected {expected}"
        )
    return capability, {"state": state, "reason_code": reason, "detail": None}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument(
        "--core-execution-report",
        type=Path,
        help=(
            "When supplied, bind the exact chaser projection to this selected "
            "core-behavior execution report instead of creating a Phase-C bundle."
        ),
    )
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-analysis-zarr", type=Path)
    parser.add_argument("--expected-recording-id")
    parser.add_argument(
        "--absent-capability",
        action="append",
        default=[],
        type=_capability_disposition,
        metavar="CAPABILITY=STATE:REASON_CODE",
        help=(
            "Explicit disposition for each capability not bound by the exact "
            "projection. Repeat once per absent capability."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dispositions: dict[str, dict[str, object]] = {}
    for capability, record in args.absent_capability:
        if capability in dispositions:
            raise SystemExit(
                f"Duplicate --absent-capability declaration for {capability!r}."
            )
        dispositions[capability] = record
    if args.core_execution_report is not None:
        if dispositions:
            raise SystemExit(
                "--absent-capability cannot reinterpret a core-plus-chaser bundle."
            )
        if args.expected_analysis_zarr is None or args.expected_recording_id is None:
            raise SystemExit(
                "Core-plus-chaser composition requires --expected-analysis-zarr "
                "and --expected-recording-id."
            )
        result = ensure_core_chaser_composite_bundle(
            args.core_execution_report,
            args.projection_receipt,
            palette_commit=args.palette_commit,
            output_json=args.output_json,
            expected_analysis_zarr=args.expected_analysis_zarr,
            expected_recording_id=args.expected_recording_id,
        )
        adapter_id = CORE_CHASER_BUNDLE_ADAPTER_ID
    else:
        result = ensure_validated_recording_behavior_bundle(
            args.projection_receipt,
            absent_capability_dispositions=dispositions,
            palette_commit=args.palette_commit,
            output_json=args.output_json,
            expected_analysis_zarr=args.expected_analysis_zarr,
            expected_recording_id=args.expected_recording_id,
        )
        adapter_id = RECORDING_BUNDLE_ADAPTER_ID
    capabilities = result["capabilities"]
    summary = {
        "status": result["status"],
        "mode": result["mode"],
        "bundle_path": result["bundle_path"],
        "recording_id": result["recording_id"],
        "record_sha256": result["record_sha256"],
        "bundle_adapter_id": adapter_id,
        "complete_capabilities": sorted(
            key for key, value in capabilities.items() if value["state"] == "complete"
        ),
        "noncomplete_capabilities": {
            key: {
                "state": value["state"],
                "reason_code": value["reason_code"],
            }
            for key, value in sorted(capabilities.items())
            if value["state"] != "complete"
        },
        "source_binding_count": len(result["source_bindings"]),
        "scientific_child_binding_count": len(result["scientific_child_bindings"]),
        "selector_eligible": result["safety"]["selector_eligible"],
        "production_authority": result["safety"]["production_authority"],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
