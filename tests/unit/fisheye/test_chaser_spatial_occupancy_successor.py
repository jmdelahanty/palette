from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.provider_chaser_position_suite import PositionSuiteEpoch
from fisheye.analysis_workflows.chaser_spatial_occupancy_successor import (
    ChaserSpatialOccupancyInput,
    ChaserSpatialOccupancySuccessorError,
    SpatialPositionProviderInput,
    prepare_chaser_spatial_occupancy_successor,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    load_composable_chaser_successor_source_handle,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    ensure_exact_immutable_child_validation_receipt,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.utils.plot_chaser_spatial_occupancy_successor import (
    PLOT_RECIPE_ID,
    main as plot_main,
    render_spatial_occupancy_heatmaps,
)
from fisheye.visualization.chaser_spatial_occupancy_display import (
    DEFAULT_DISPLAY_MODE_ID,
    STATIC_EXPORT_MODE_IDS,
)


def _provider(role: str, provider_id: str, xy: np.ndarray, valid: np.ndarray):
    digest_character = "a" if role == "keypoint" else "b"
    return SpatialPositionProviderInput(
        provider_role=role,
        relative_frame_run_path=f"analysis/chaser_relative_frame_runs/{role}",
        relative_frame_manifest_sha256=digest_character * 64,
        radial_run_path=f"analysis/chaser_radial_near_field_runs/{role}",
        radial_manifest_sha256=("c" if role == "keypoint" else "d") * 64,
        fish_position_authority={
            "provider_id": provider_id,
            "provider_digest": ("e" if role == "keypoint" else "f") * 64,
            "coordinate_authority_id": "/coordinate@pixel_frame",
        },
        fish_xy_px=xy,
        fish_valid=valid,
    )


def _inputs() -> ChaserSpatialOccupancyInput:
    keypoint = np.asarray(
        [
            [-7.0, -7.0],
            [-2.0, -2.0],
            [2.0, -2.0],
            [7.0, -2.0],
            [-2.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    detection = keypoint + np.asarray([1.0, 0.0])
    keypoint_valid = np.ones(6, dtype=bool)
    detection_valid = np.ones(6, dtype=bool)
    detection_valid[4] = False
    epochs = tuple(
        PositionSuiteEpoch(
            analysis_role=role,
            window_id=index,
            source_label=label,
            start_frame=2 * index,
            end_frame=2 * index + 2,
            source_interval_sha256=str(index + 1) * 64,
        )
        for index, (role, label) in enumerate(
            (
                ("chaser_pre", "pre_event"),
                ("chaser_training", "training_event"),
                ("chaser_post", "post_event"),
            )
        )
    )
    return ChaserSpatialOccupancyInput(
        recording_id="recording",
        semantic_selection_run_path=(
            "analysis/protocol_semantic_chaser_selection_runs/exact"
        ),
        semantic_selection_manifest_sha256="9" * 64,
        acquisition_frame_id=np.arange(6, dtype=np.int64),
        selection_member=np.ones(6, dtype=bool),
        epochs=epochs,
        providers=(
            _provider(
                "keypoint",
                "keypoint_anatomical_triad_mean.v1",
                keypoint,
                keypoint_valid,
            ),
            _provider(
                "detection",
                "detection_bbox_centroid.v1",
                detection,
                detection_valid,
            ),
        ),
        arena_center_x_px=0.0,
        arena_center_y_px=0.0,
        arena_radius_px=10.0,
        mm_per_pixel=1.0,
        arena_geometry_authority={
            "selection_record_sha256": "7" * 64,
            "physical_authority_sha256": "8" * 64,
            "pixel_frame_record_ref": "/coordinate@pixel_frame",
            "pixel_frame_record_sha256": "6" * 64,
        },
        arena_boundary_role="reviewed_inner_rim",
        arena_observed_feature="dish_inner_rim_water_side_edge",
        bin_width_mm=5.0,
    )


def test_paired_provider_epoch_heatmaps_conserve_rows_and_missingness() -> None:
    prepared = prepare_chaser_spatial_occupancy_successor(_inputs())

    assert prepared.arrays["occupancy_count"].shape == (2, 3, 4, 4)
    assert prepared.arrays["candidate_frame_count"].tolist() == [
        [2, 2, 2],
        [2, 2, 2],
    ]
    assert prepared.arrays["in_arena_position_frame_count"].tolist() == [
        [2, 2, 2],
        [2, 2, 1],
    ]
    np.testing.assert_allclose(
        prepared.arrays["occupancy_density_valid_in_arena"].sum(axis=(2, 3)),
        np.ones((2, 3)),
    )
    np.testing.assert_allclose(
        prepared.arrays["occupancy_fraction_candidate_epoch"].sum(axis=(2, 3)),
        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.5]],
    )
    assert prepared.manifest["grid"]["coordinate_orientation"] == "+x_right_+y_down"
    assert prepared.manifest["denominators"]["interpolation"] == "prohibited"


def test_semantic_epoch_rows_must_be_present_in_relative_selection() -> None:
    inputs = _inputs()
    selected = np.ones(6, dtype=bool)
    selected[3] = False

    with pytest.raises(
        ChaserSpatialOccupancySuccessorError,
        match="epoch row is absent",
    ):
        prepare_chaser_spatial_occupancy_successor(
            replace(inputs, selection_member=selected)
        )


def test_provider_identities_must_remain_distinct() -> None:
    inputs = _inputs()
    detection = inputs.providers[1]
    duplicated = replace(
        detection,
        fish_position_authority={
            **detection.fish_position_authority,
            "provider_id": "keypoint_anatomical_triad_mean.v1",
        },
    )

    with pytest.raises(
        ChaserSpatialOccupancySuccessorError,
        match="distinct provider identities",
    ):
        prepare_chaser_spatial_occupancy_successor(
            replace(inputs, providers=(inputs.providers[0], duplicated))
        )


def test_publication_deep_audit_rehydrate_and_plot(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording"
    prepared = prepare_chaser_spatial_occupancy_successor(
        replace(_inputs(), bin_width_mm=2.0)
    )
    plan = build_composable_chaser_successor_publication_plan(
        archive,
        run_name="spatial-v1",
        prepared=prepared,
    )
    publication = publish_composable_chaser_successor_run(
        plan,
        scratch_root=tmp_path / "scratch",
    )
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_spatial_occupancy",
        run_name="spatial-v1",
        deep_audit=True,
    )
    rehydrated = handle.prepared_successor()
    png, pdf = render_spatial_occupancy_heatmaps(
        handle,
        output_stem=tmp_path / "occupancy",
    )

    assert publication["selector_eligible"] is False
    assert rehydrated.payload_digest == prepared.payload_digest
    assert png.is_file() and pdf.is_file()

    output_dir = tmp_path / "plots"
    assert (
        plot_main(
            [
                str(archive),
                "--run-name",
                "spatial-v1",
                "--bundle-name",
                "spatial-recipe-v5",
                "--expected-recording-id",
                "recording",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    receipt = output_dir / "spatial-recipe-v5_spatial_occupancy_plot_receipt.json"
    assert receipt.is_file()
    receipt_record = json.loads(receipt.read_text(encoding="utf-8"))
    parameters = receipt_record["plot_parameters"]
    assert receipt_record["schema_version"] == 4
    assert receipt_record["plot_recipe_id"] == PLOT_RECIPE_ID
    assert receipt_record["run_name"] == "spatial-v1"
    assert receipt_record["bundle_name"] == "spatial-recipe-v5"
    assert parameters["scientific_coordinates"]["x_bin_widths_mm"] == [
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ]
    display_recipe = parameters["display_recipe"]
    assert display_recipe["default_display_mode"] == DEFAULT_DISPLAY_MODE_ID
    assert display_recipe["static_export_modes"] == list(STATIC_EXPORT_MODE_IDS)
    assert (
        display_recipe["display_surfaces"]["4mm_valid_in_arena"]["count_aggregation"]
        == "exact_2x2_sum"
    )
    assert parameters["static_rendering"]["png_dpi"] == 180
    assert len(receipt_record["outputs"]) == 4
    assert {output["display_mode_id"] for output in receipt_record["outputs"]} == set(
        STATIC_EXPORT_MODE_IDS
    )
    assert (
        sum(
            output["artifact_role"] == "recommended_default"
            for output in receipt_record["outputs"]
        )
        == 2
    )
    assert all(Path(output["path"]).is_file() for output in receipt_record["outputs"])

    validation_receipt = tmp_path / "spatial.exact_child_receipt.json"
    ensure_exact_immutable_child_validation_receipt(
        archive,
        run_path="analysis/chaser_spatial_occupancy_runs/spatial-v1",
        manifest_attr="composable_chaser_successor_manifest",
        manifest_digest_attr="composable_chaser_successor_manifest_sha256",
        palette_commit="a" * 40,
        output_json=validation_receipt,
        expected_recording_id="recording",
    )
    targeted_output_dir = tmp_path / "targeted-plots"
    assert (
        plot_main(
            [
                str(archive),
                "--run-name",
                "spatial-v1",
                "--bundle-name",
                "spatial-receipt-bound-v5",
                "--expected-recording-id",
                "recording",
                "--output-dir",
                str(targeted_output_dir),
                "--source-validation-receipt",
                str(validation_receipt),
            ]
        )
        == 0
    )
    targeted_receipt = json.loads(
        (
            targeted_output_dir
            / "spatial-receipt-bound-v5_spatial_occupancy_plot_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert targeted_receipt["schema_version"] == 4
    assert targeted_receipt["source_binding"]["deep_content_audit"] is False
    assert targeted_receipt["source_binding"]["verification_mode"] == (
        "receipt_bound_targeted_array_rehash_v1"
    )
    assert targeted_receipt["source_binding"]["verified_array_names"] == sorted(
        (
            "arena_bin_center_mask",
            "candidate_frame_count",
            "in_arena_coverage_fraction_candidate",
            "in_arena_position_frame_count",
            "occupancy_count",
            "occupancy_density_valid_in_arena",
            "occupancy_fraction_candidate_epoch",
            "x_bin_edges_mm",
            "y_bin_edges_mm",
        )
    )
