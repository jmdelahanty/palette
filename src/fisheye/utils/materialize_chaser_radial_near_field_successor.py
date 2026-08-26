"""Plan or publish one exact provider-generic radial/near-field successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis.provider_chaser_position_suite import (
    CircularArena,
    PositionSuiteConfig,
)
from fisheye.analysis_workflows.chaser_radial_near_field_successor import (
    prepare_chaser_radial_near_field_successor_from_handles,
)
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    load_chaser_relative_frame_source_handle,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    load_protocol_semantic_chaser_selection_source_handle,
)
from fisheye.utils.materialize_provider_spatial_canary import (
    load_grid_and_transform_authority,
)


def _parse_float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one number")
    return result


def run(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    relative_frame_run: str,
    semantic_selection_run: str,
    geometry_selection_run: str,
    expected_selection_record_sha256: str,
    expected_physical_authority_sha256: str,
    treatment_role: str = "aggressive",
    baseline_role: str = "inert",
    radial_bin_width_mm: float = 2.0,
    cdf_thresholds_mm: Sequence[float] | None = None,
    near_zone_radius_mm: float = 5.0,
    near_entry_radius_mm: float = 5.0,
    near_exit_radius_mm: float = 6.0,
    perimeter_band_mm: float = 5.0,
    min_expected_count: float = 5.0,
    apply: bool = False,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, object]:
    archive = Path(analysis_zarr).expanduser().resolve()
    relative = load_chaser_relative_frame_source_handle(
        archive, run_name=relative_frame_run, use_consolidated=True
    )
    semantic = load_protocol_semantic_chaser_selection_source_handle(
        archive,
        run_name=semantic_selection_run,
        expected_recording_id=relative.recording_id,
        use_consolidated=True,
        deep_audit=True,
    )
    geometry_task = {
        "analysis_zarr": str(archive),
        "recording_id": relative.recording_id,
        "geometry_source": {
            "selection_run_name": geometry_selection_run,
            "selection_record_sha256": expected_selection_record_sha256,
            "physical_authority_sha256": expected_physical_authority_sha256,
        },
        "grid": {
            "policy_id": "chaser_radial_near_field_successor_arena_grid_v1",
            "bin_width_mm": 1.0,
        },
    }
    grid, transform, evidence = load_grid_and_transform_authority(geometry_task)
    coordinate = evidence["selection"]["record"]["selected_candidate"][
        "coordinate_binding"
    ]
    geometry_authority = {
        "selection_run_name": geometry_selection_run,
        "selection_record_sha256": expected_selection_record_sha256,
        "physical_authority_sha256": expected_physical_authority_sha256,
        "pixel_frame_record_ref": coordinate["pixel_frame_record_ref"],
        "pixel_frame_record_sha256": coordinate["pixel_frame_record_sha256"],
        "grid_policy_digest": grid.policy_digest,
        "transform_sha256": transform.sha256,
        "boundary_role": grid.geometry.boundary_role,
        "observed_feature": grid.geometry.observed_feature,
    }
    prepared = prepare_chaser_radial_near_field_successor_from_handles(
        relative,
        semantic,
        arena=CircularArena(
            center_x_px=grid.geometry.center_x_px,
            center_y_px=grid.geometry.center_y_px,
            radius_px=grid.geometry.radius_px,
            boundary_role=grid.geometry.boundary_role,
            observed_feature=str(grid.geometry.observed_feature),
        ),
        mm_per_pixel=grid.scale.mm_per_pixel,
        arena_geometry_authority=geometry_authority,
        config=PositionSuiteConfig(
            radial_bin_width_mm=radial_bin_width_mm,
            cdf_thresholds_mm=(
                None if cdf_thresholds_mm is None else tuple(cdf_thresholds_mm)
            ),
            near_zone_radius_mm=near_zone_radius_mm,
            near_entry_radius_mm=near_entry_radius_mm,
            near_exit_radius_mm=near_exit_radius_mm,
            perimeter_band_mm=perimeter_band_mm,
            min_expected_count=min_expected_count,
            treatment_role=treatment_role,
            baseline_role=baseline_role,
        ),
    )
    plan = build_composable_chaser_successor_publication_plan(
        archive, run_name=run_name, prepared=prepared
    )
    result: dict[str, object] = {
        "status": "dry_run_plan",
        "successor_kind": plan.successor_kind,
        "run_path": plan.run_path,
        "recording_id": plan.recording_id,
        "scientific_payload_sha256": prepared.payload_digest,
        "position_provider": dict(prepared.manifest["position_provider"]),
        "dimensions": dict(prepared.manifest["dimensions"]),
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
    }
    if apply:
        result = publish_composable_chaser_successor_run(
            plan, scratch_root=scratch_root, copy_backend=copy_backend
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--relative-frame-run", required=True)
    parser.add_argument("--semantic-selection-run", required=True)
    parser.add_argument("--geometry-selection-run", required=True)
    parser.add_argument("--expected-selection-record-sha256", required=True)
    parser.add_argument("--expected-physical-authority-sha256", required=True)
    parser.add_argument("--treatment-role", default="aggressive")
    parser.add_argument("--baseline-role", default="inert")
    parser.add_argument("--radial-bin-width-mm", type=float, default=2.0)
    parser.add_argument("--cdf-thresholds-mm", type=_parse_float_list)
    parser.add_argument("--near-zone-radius-mm", type=float, default=5.0)
    parser.add_argument("--near-entry-radius-mm", type=float, default=5.0)
    parser.add_argument("--near-exit-radius-mm", type=float, default=6.0)
    parser.add_argument("--perimeter-band-mm", type=float, default=5.0)
    parser.add_argument("--min-expected-count", type=float, default=5.0)
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
        geometry_selection_run=args.geometry_selection_run,
        expected_selection_record_sha256=args.expected_selection_record_sha256,
        expected_physical_authority_sha256=args.expected_physical_authority_sha256,
        treatment_role=args.treatment_role,
        baseline_role=args.baseline_role,
        radial_bin_width_mm=args.radial_bin_width_mm,
        cdf_thresholds_mm=args.cdf_thresholds_mm,
        near_zone_radius_mm=args.near_zone_radius_mm,
        near_entry_radius_mm=args.near_entry_radius_mm,
        near_exit_radius_mm=args.near_exit_radius_mm,
        perimeter_band_mm=args.perimeter_band_mm,
        min_expected_count=args.min_expected_count,
        apply=args.apply,
        scratch_root=args.scratch_root,
        copy_backend=args.copy_backend,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
