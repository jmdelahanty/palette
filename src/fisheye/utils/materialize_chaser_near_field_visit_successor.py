"""Plan or publish one exact individual near-field visit successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.chaser_near_field_visit_successor import (
    MIN_VISIT_SAMPLE_COUNT,
    RADIAL_PARITY_ARRAY_NAMES,
    RELATIVE_FRAME_ARRAY_NAMES,
    RELATIVE_FRAME_COLLAPSED_ARRAY_NAMES,
    prepare_chaser_near_field_visit_successor_from_handles,
)
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    load_chaser_relative_frame_source_handle,
)
from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    load_chaser_relative_frame_targeted_source_handle,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    load_composable_chaser_successor_source_handle,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    load_protocol_semantic_chaser_selection_source_handle,
)


def run(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    relative_frame_run: str,
    semantic_selection_run: str,
    radial_near_field_run: str,
    expected_recording_id: str | None = None,
    relative_frame_validation_receipt: str | Path | None = None,
    semantic_selection_validation_receipt: str | Path | None = None,
    radial_validation_receipt: str | Path | None = None,
    minimum_quality_sample_count: int = MIN_VISIT_SAMPLE_COUNT,
    apply: bool = False,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, object]:
    archive = Path(analysis_zarr).expanduser().resolve()
    relative = (
        load_chaser_relative_frame_source_handle(
            archive,
            run_name=relative_frame_run,
            expected_recording_id=expected_recording_id,
            use_consolidated=True,
        )
        if relative_frame_validation_receipt is None
        else load_chaser_relative_frame_targeted_source_handle(
            relative_frame_validation_receipt,
            required_base_arrays=RELATIVE_FRAME_ARRAY_NAMES,
            collapsed_frame_arrays=RELATIVE_FRAME_COLLAPSED_ARRAY_NAMES,
            expected_analysis_zarr=archive,
            expected_recording_id=expected_recording_id,
            expected_run_name=relative_frame_run,
        )
    )
    semantic = load_protocol_semantic_chaser_selection_source_handle(
        archive,
        run_name=semantic_selection_run,
        expected_recording_id=relative.recording_id,
        use_consolidated=True,
        deep_audit=semantic_selection_validation_receipt is None,
        direct_validation_receipt=semantic_selection_validation_receipt,
    )
    radial = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_radial_near_field",
        run_name=radial_near_field_run,
        expected_recording_id=relative.recording_id,
        use_consolidated=True,
        deep_audit=radial_validation_receipt is None,
        direct_validation_receipt=radial_validation_receipt,
        required_array_names=(
            None if radial_validation_receipt is None else RADIAL_PARITY_ARRAY_NAMES
        ),
    )
    prepared = prepare_chaser_near_field_visit_successor_from_handles(
        relative,
        semantic,
        radial,
        minimum_quality_sample_count=minimum_quality_sample_count,
    )
    plan = build_composable_chaser_successor_publication_plan(
        archive,
        run_name=run_name,
        prepared=prepared,
    )
    result: dict[str, object] = {
        "status": "dry_run_plan",
        "successor_kind": plan.successor_kind,
        "run_path": plan.run_path,
        "recording_id": plan.recording_id,
        "scientific_payload_sha256": prepared.payload_digest,
        "position_provider": dict(prepared.manifest["position_provider"]),
        "dimensions": dict(prepared.manifest["dimensions"]),
        "radial_aggregate_parity": dict(prepared.manifest["radial_aggregate_parity"]),
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
    }
    if apply:
        result = publish_composable_chaser_successor_run(
            plan,
            scratch_root=scratch_root,
            copy_backend=copy_backend,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--relative-frame-run", required=True)
    parser.add_argument("--semantic-selection-run", required=True)
    parser.add_argument("--radial-near-field-run", required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--relative-frame-validation-receipt")
    parser.add_argument("--semantic-selection-validation-receipt")
    parser.add_argument("--radial-validation-receipt")
    parser.add_argument(
        "--minimum-quality-sample-count",
        type=int,
        default=MIN_VISIT_SAMPLE_COUNT,
        help=(
            "Visits below this sample count are retained and marked short; "
            "they are never dropped."
        ),
    )
    parser.add_argument("--scratch-root")
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        args.analysis_zarr,
        run_name=args.run_name,
        relative_frame_run=args.relative_frame_run,
        semantic_selection_run=args.semantic_selection_run,
        radial_near_field_run=args.radial_near_field_run,
        expected_recording_id=args.expected_recording_id,
        relative_frame_validation_receipt=(args.relative_frame_validation_receipt),
        semantic_selection_validation_receipt=(
            args.semantic_selection_validation_receipt
        ),
        radial_validation_receipt=args.radial_validation_receipt,
        minimum_quality_sample_count=args.minimum_quality_sample_count,
        apply=args.apply,
        scratch_root=args.scratch_root,
        copy_backend=args.copy_backend,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
