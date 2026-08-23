"""Emit bounded readiness evidence for one exact position-suite candidate.

This is deliberately not a registry writer.  A selector-ineligible scientific
candidate can be complete while production/registry readiness remains blocked.
The receipt makes those states separate so a later serialized finalizer can
project only an explicitly promoted run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from fisheye.analysis_workflows.provider_chaser_position_suite_publication import (
    load_provider_chaser_position_suite_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.provider_chaser_position_suite_readiness"
SCHEMA_VERSION = 1
STAGE_ID = "provider_chaser_position_suite"


def build_readiness_receipt(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
) -> dict[str, Any]:
    handle = load_provider_chaser_position_suite_source_handle(
        analysis_zarr,
        run_name=run_name,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
    )
    payload: dict[str, Any] = {
        "stage_id": STAGE_ID,
        "recording_id": handle.recording_id,
        "analysis_zarr": str(handle.analysis_zarr),
        "run_name": handle.run_name,
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "computed_suite_sha256": handle.manifest["computed_suite_sha256"],
        "source_bindings_sha256": handle.manifest["source_bindings_sha256"],
        "scientific_candidate_state": "complete",
        "scientific_scope": "position_only",
        "table_row_counts": dict(handle.manifest["dimensions"]["table_row_counts"]),
        "total_table_row_count": handle.total_table_row_count,
        "array_count": len(handle.arrays),
        "source_verification_mode": handle.verification_mode,
        "direct_consolidated_metadata_equivalent": True,
        "metadata_equivalence": dict(handle.metadata_equivalence),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "production_readiness": "blocked",
        "production_blockers": [
            "required_ci_not_bound",
            "production_selector_not_activated",
        ],
        "registry_projection_eligible": False,
        "registry_update": False,
        "registry_projection_policy": (
            "serialized_finalizer_after_required_ci_and_exact_selector_promotion_v1"
        ),
        "next_action": (
            "pass_required_ci_then_explicitly_promote_one_exact_manifest_before_"
            "serialized_registry_projection"
        ),
    }
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_complete_production_blocked",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_readiness_receipt(
        args.analysis_zarr,
        run_name=args.run_name,
        expected_recording_id=args.expected_recording_id,
    )
    if args.output_json is not None:
        write_json_atomic(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "STAGE_ID",
    "build_readiness_receipt",
    "main",
]
