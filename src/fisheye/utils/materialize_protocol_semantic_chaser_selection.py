"""Plan or publish one immutable protocol-semantic chaser hierarchy.

The command accepts one exact stimulus-epoch v2 run plus explicit window IDs.
It validates the materialized producer semantic snapshot and timeline, compiles
the nested hierarchy, and produces a revealing dry-run by default. ``--apply``
publishes one selector-ineligible immutable run; it never activates a selector
or updates the registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.composable_epoch_selection_adapter import (
    EpochRoleBinding,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    STEP_END_EXCLUSIVE,
    STEP_END_PENDING,
    STANDALONE_SOLID_BLACK_ROLE,
    compile_protocol_semantic_chaser_selections,
    load_protocol_semantic_selection_evidence,
    load_protocol_semantic_timeline_evidence,
)
from fisheye.shared.frame_bound_acquisition_identity import (
    load_paired_frame_bound_chaser_source,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    ProtocolSemanticChaserSelectionPublicationPlan,
    build_protocol_semantic_chaser_selection_publication_plan,
    publish_protocol_semantic_chaser_selection_run,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.shared.json_safety import write_json_atomic


def plan_protocol_semantic_chaser_selection_run(
    analysis_zarr: str | Path,
    *,
    source_epoch_selection_run: str,
    expected_source_epoch_manifest_sha256: str,
    run_name: str,
    chaser_pre_window_id: int,
    chaser_training_window_id: int,
    chaser_post_window_id: int,
    standalone_solid_black_window_id: int | None = None,
    frame_bound_companion_h5: str | Path | None = None,
    recording_bundle_root: str | Path | None = None,
    expected_recording_id: str | None = None,
    expected_camera_serial: str | None = None,
    expected_acquisition_camera_id: str | None = None,
    expected_shaman_numeric_camera_id: int | None = None,
    expected_source_total_frames: int | None = None,
) -> ProtocolSemanticChaserSelectionPublicationPlan:
    """Resolve all exact authorities and construct one no-write plan."""

    selection = resolve_exact_stimulus_epoch_selection(
        analysis_zarr,
        run_name=source_epoch_selection_run,
        expected_run_manifest_digest=expected_source_epoch_manifest_sha256,
    )
    frame_bound_source = None
    if frame_bound_companion_h5 is not None:
        required = {
            "recording_bundle_root": recording_bundle_root,
            "expected_recording_id": expected_recording_id,
            "expected_camera_serial": expected_camera_serial,
            "expected_acquisition_camera_id": expected_acquisition_camera_id,
            "expected_shaman_numeric_camera_id": (
                expected_shaman_numeric_camera_id
            ),
            "expected_source_total_frames": expected_source_total_frames,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "Frame-bound semantic selection requires explicit "
                + ", ".join(missing)
                + "."
            )
        frame_bound_source = load_paired_frame_bound_chaser_source(
            Path(frame_bound_companion_h5),
            recording_bundle_root=Path(recording_bundle_root),
            expected_recording_id=expected_recording_id,
            expected_camera_serial=expected_camera_serial,
            expected_acquisition_camera_id=expected_acquisition_camera_id,
            expected_shaman_numeric_camera_id=expected_shaman_numeric_camera_id,
            expected_source_total_frames=expected_source_total_frames,
        )
    protocol_evidence = load_protocol_semantic_selection_evidence(
        analysis_zarr,
        selection,
        step_end_interval_semantics=(
            STEP_END_EXCLUSIVE
            if frame_bound_source is not None
            else STEP_END_PENDING
        ),
        use_consolidated=True,
        frame_bound_source=frame_bound_source,
    )
    timeline_evidence = load_protocol_semantic_timeline_evidence(
        analysis_zarr,
        selection,
    )
    role_bindings = {
        "chaser_pre": EpochRoleBinding.by_window_id(chaser_pre_window_id),
        "chaser_training": EpochRoleBinding.by_window_id(
            chaser_training_window_id
        ),
        "chaser_post": EpochRoleBinding.by_window_id(chaser_post_window_id),
    }
    if standalone_solid_black_window_id is not None:
        role_bindings[STANDALONE_SOLID_BLACK_ROLE] = (
            EpochRoleBinding.by_window_id(standalone_solid_black_window_id)
        )
    selections = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=timeline_evidence,
        protocol_evidence=protocol_evidence,
        role_bindings=role_bindings,
    )
    return build_protocol_semantic_chaser_selection_publication_plan(
        analysis_zarr,
        selections=selections,
        source_selection=selection,
        run_name=run_name,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--source-epoch-selection-run", required=True)
    parser.add_argument("--expected-source-epoch-manifest-sha256", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--chaser-pre-window-id", required=True, type=int)
    parser.add_argument("--chaser-training-window-id", required=True, type=int)
    parser.add_argument("--chaser-post-window-id", required=True, type=int)
    parser.add_argument("--standalone-solid-black-window-id", type=int)
    parser.add_argument("--frame-bound-companion-h5", type=Path)
    parser.add_argument("--recording-bundle-root", type=Path)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--expected-camera-serial")
    parser.add_argument("--expected-acquisition-camera-id")
    parser.add_argument("--expected-shaman-numeric-camera-id", type=int)
    parser.add_argument("--expected-source-total-frames", type=int)
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
    plan = plan_protocol_semantic_chaser_selection_run(
        args.analysis_zarr,
        source_epoch_selection_run=args.source_epoch_selection_run,
        expected_source_epoch_manifest_sha256=(
            args.expected_source_epoch_manifest_sha256
        ),
        run_name=args.run_name,
        chaser_pre_window_id=args.chaser_pre_window_id,
        chaser_training_window_id=args.chaser_training_window_id,
        chaser_post_window_id=args.chaser_post_window_id,
        standalone_solid_black_window_id=(
            args.standalone_solid_black_window_id
        ),
        frame_bound_companion_h5=args.frame_bound_companion_h5,
        recording_bundle_root=args.recording_bundle_root,
        expected_recording_id=args.expected_recording_id,
        expected_camera_serial=args.expected_camera_serial,
        expected_acquisition_camera_id=args.expected_acquisition_camera_id,
        expected_shaman_numeric_camera_id=(
            args.expected_shaman_numeric_camera_id
        ),
        expected_source_total_frames=args.expected_source_total_frames,
    )
    result = (
        plan.to_json()
        if not args.apply
        else publish_protocol_semantic_chaser_selection_run(
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


__all__ = ["main", "plan_protocol_semantic_chaser_selection_run"]
