from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.provider_chaser_position_suite import PositionSuiteEpoch
from fisheye.analysis_workflows.chaser_body_alignment_by_distance_successor import (
    SUMMARY_VIEW_ARRAY_NAMES,
    ChaserBodyAlignmentByDistanceError,
    ChaserBodyAlignmentByDistanceInput,
    prepare_chaser_body_alignment_by_distance_successor,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    build_composable_chaser_successor_publication_plan,
    load_composable_chaser_successor_source_handle,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    ensure_exact_immutable_child_validation_receipt,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.utils.plot_chaser_body_alignment_by_distance_successor import (
    PLOT_RECIPE_ID,
    body_alignment_plot_parameters,
    main as plot_main,
    render_body_alignment_by_distance,
)


def _inputs() -> ChaserBodyAlignmentByDistanceInput:
    n_frames = 6
    n_chasers = 1
    shape = (n_frames, n_chasers)
    bearing = np.asarray([0.0, 90.0, -90.0, 180.0, -180.0, 0.0])[:, None]
    return ChaserBodyAlignmentByDistanceInput(
        recording_id="recording",
        relative_frame_run_path="analysis/chaser_relative_frame_runs/keypoint",
        relative_frame_manifest_sha256="a" * 64,
        semantic_selection_run_path=(
            "analysis/protocol_semantic_chaser_selection_runs/semantic"
        ),
        semantic_selection_manifest_sha256="b" * 64,
        n_frames=n_frames,
        n_chasers=n_chasers,
        acquisition_frame_id=np.arange(n_frames, dtype=np.int64)[:, None],
        selection_member=np.ones(shape, dtype=bool),
        chaser_occurrence_member=np.ones(shape, dtype=bool),
        chaser_identity_code=np.ones(shape, dtype=np.uint16),
        chaser_behavior_role_code=np.full(shape, 2, dtype=np.uint8),
        chaser_behavior_role_valid=np.ones(shape, dtype=bool),
        relative_distance_physical=np.asarray(
            [2.0, 2.0, 7.0, 7.0, 12.0, 12.0], dtype=np.float32
        )[:, None],
        relative_physical_valid=np.ones(shape, dtype=bool),
        relative_physical_reason_code=np.zeros(shape, dtype=np.uint16),
        body_source_row_id=np.arange(n_frames, dtype=np.int64)[:, None],
        body_source_row_valid=np.ones(shape, dtype=bool),
        body_heading_deg=np.zeros(shape, dtype=np.float32),
        body_heading_valid=np.ones(shape, dtype=bool),
        body_heading_reason_code=np.zeros(shape, dtype=np.uint16),
        body_bearing_deg=bearing.astype(np.float32),
        body_bearing_valid=np.ones(shape, dtype=bool),
        body_bearing_reason_code=np.zeros(shape, dtype=np.uint16),
        epochs=tuple(
            PositionSuiteEpoch(
                analysis_role=role,
                window_id=index + 10,
                source_label=f"source-{index}",
                start_frame=index * 2,
                end_frame=index * 2 + 2,
                source_interval_sha256=str(index + 1) * 64,
            )
            for index, role in enumerate(
                ("chaser_pre", "chaser_training", "chaser_post")
            )
        ),
        fish_position_authority={
            "provider_id": "keypoint_anatomical_triad_mean.v1",
            "provider_digest": "c" * 64,
            "coordinate_authority_id": "/coordinate@pixel_frame",
        },
        body_frame_authority={
            "provider_id": "exact_keypoint_body_frame_acquisition_projection_v1",
            "provider_digest": "d" * 64,
            "coordinate_authority_id": "/coordinate@pixel_frame",
        },
        identity_registries={
            "chaser": {"1": "blue-dot"},
            "behavior_role": {"2": "aggressive"},
        },
        scale_policy={"unit": "mm", "pixels_per_unit": 2.0},
        distance_bin_width_mm=5.0,
    )


def _two_chaser_inputs() -> ChaserBodyAlignmentByDistanceInput:
    inputs = _inputs()
    matrix_fields = (
        "acquisition_frame_id",
        "selection_member",
        "chaser_occurrence_member",
        "chaser_identity_code",
        "chaser_behavior_role_code",
        "chaser_behavior_role_valid",
        "relative_distance_physical",
        "relative_physical_valid",
        "relative_physical_reason_code",
        "body_source_row_id",
        "body_source_row_valid",
        "body_heading_deg",
        "body_heading_valid",
        "body_heading_reason_code",
        "body_bearing_deg",
        "body_bearing_valid",
        "body_bearing_reason_code",
    )
    updates = {
        field: np.repeat(getattr(inputs, field), 2, axis=1) for field in matrix_fields
    }
    updates["chaser_identity_code"][:, 1] = 2
    return replace(
        inputs,
        n_chasers=2,
        identity_registries={
            **inputs.identity_registries,
            "chaser": {"1": "blue-dot", "2": "red-dot"},
        },
        **updates,
    )


def test_closed_form_bearings_persist_exact_alignment_components() -> None:
    prepared = prepare_chaser_body_alignment_by_distance_successor(_inputs())

    np.testing.assert_allclose(
        prepared.arrays["frame_alignment_cos"],
        [1.0, 0.0, 0.0, -1.0, -1.0, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        prepared.arrays["frame_lateral_sin"],
        [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
        atol=1e-12,
    )
    assert prepared.arrays["distance_bin_edges_mm"].tolist() == [
        0.0,
        5.0,
        10.0,
        15.0,
    ]
    assert prepared.arrays["summary_joint_valid_row_count"].sum() == 6
    assert prepared.arrays["summary_candidate_row_count"].sum() == 6
    assert prepared.manifest["denominators"]["viewer_rebinning"] == "prohibited"
    assert (
        prepared.manifest["position_provider"]["distance_surface"]
        == "base/relative_distance_physical"
    )


def test_missing_body_row_remains_invalid_without_motion_fallback() -> None:
    inputs = _inputs()
    source_valid = inputs.body_source_row_valid.copy()
    heading_valid = inputs.body_heading_valid.copy()
    bearing_valid = inputs.body_bearing_valid.copy()
    heading = inputs.body_heading_deg.copy()
    bearing = inputs.body_bearing_deg.copy()
    source_valid[1, 0] = False
    heading_valid[1, 0] = False
    bearing_valid[1, 0] = False
    heading[1, 0] = np.nan
    bearing[1, 0] = np.nan

    prepared = prepare_chaser_body_alignment_by_distance_successor(
        replace(
            inputs,
            body_source_row_valid=source_valid,
            body_heading_valid=heading_valid,
            body_bearing_valid=bearing_valid,
            body_heading_deg=heading,
            body_bearing_deg=bearing,
        )
    )

    assert prepared.arrays["frame_alignment_valid"].tolist()[1] is False
    assert prepared.arrays["frame_alignment_reason_code"].tolist()[1] == 3
    assert np.isnan(prepared.arrays["frame_alignment_cos"][1])
    assert prepared.arrays["summary_body_source_missing_row_count"].sum() == 1
    assert prepared.manifest["denominators"]["missing_body_policy"] == (
        "retained_invalid_no_motion_heading_fallback"
    )


def test_body_valid_distance_invalid_is_retained_but_never_binned() -> None:
    inputs = _inputs()
    distance_valid = inputs.relative_physical_valid.copy()
    distance_reason = inputs.relative_physical_reason_code.copy()
    distance_valid[0, 0] = False
    distance_reason[0, 0] = 7

    prepared = prepare_chaser_body_alignment_by_distance_successor(
        replace(
            inputs,
            relative_physical_valid=distance_valid,
            relative_physical_reason_code=distance_reason,
        )
    )

    assert prepared.arrays["frame_body_bearing_valid"][0]
    assert not prepared.arrays["frame_alignment_valid"][0]
    assert prepared.arrays["frame_alignment_reason_code"][0] == 2
    pre_rows = prepared.arrays["summary_epoch_role_code"] == 1
    assert np.unique(
        prepared.arrays["summary_epoch_distance_invalid_body_valid_row_count"][pre_rows]
    ).tolist() == [1]
    assert prepared.arrays["summary_candidate_row_count"].sum() == 5
    assert prepared.arrays["summary_joint_valid_row_count"].sum() == 5


def test_axis_mismatch_fails_instead_of_shortest_axis_truncation() -> None:
    inputs = _inputs()
    with pytest.raises(
        ChaserBodyAlignmentByDistanceError,
        match="exact frame/chaser shape",
    ):
        prepare_chaser_body_alignment_by_distance_successor(
            replace(
                inputs,
                relative_distance_physical=np.zeros((5, 1), dtype=np.float32),
            )
        )


def test_anatomical_fish_heading_must_repeat_across_chasers() -> None:
    inputs = _two_chaser_inputs()
    heading = inputs.body_heading_deg.copy()
    heading[0, 1] = 1.0

    with pytest.raises(
        ChaserBodyAlignmentByDistanceError,
        match="body_heading_deg differs across flattened chaser rows",
    ):
        prepare_chaser_body_alignment_by_distance_successor(
            replace(inputs, body_heading_deg=heading)
        )


def test_publication_round_trip_preserves_persisted_bins(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording"
    prepared = prepare_chaser_body_alignment_by_distance_successor(_inputs())
    plan = build_composable_chaser_successor_publication_plan(
        archive,
        run_name="alignment-v1",
        prepared=prepared,
    )
    publication = publish_composable_chaser_successor_run(
        plan,
        scratch_root=tmp_path / "scratch",
    )
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_body_alignment_by_distance",
        run_name="alignment-v1",
        deep_audit=True,
    )

    assert publication["selector_eligible"] is False
    assert handle.prepared_successor().payload_digest == prepared.payload_digest
    np.testing.assert_array_equal(
        handle.array("distance_bin_edges_mm"),
        prepared.array("distance_bin_edges_mm"),
    )

    png, pdf = render_body_alignment_by_distance(
        handle,
        output_stem=tmp_path / "alignment",
    )
    assert png.is_file() and pdf.is_file()
    parameters = body_alignment_plot_parameters(handle)
    assert parameters["scientific_coordinates"]["distance_bin_edges_mm"] == [
        0.0,
        5.0,
        10.0,
        15.0,
    ]
    assert parameters["viewer_policy"]["rebinning"] == "prohibited"
    assert parameters["rendering"]["chaser_line_style_by_identity_code"] == {"1": "-"}
    assert "no style is inferred" in parameters["rendering"]["role_style_policy"]


def test_receipt_bound_static_plot_rehashes_only_closed_summary_roster(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording"
    prepared = prepare_chaser_body_alignment_by_distance_successor(_inputs())
    plan = build_composable_chaser_successor_publication_plan(
        archive,
        run_name="alignment-v1",
        prepared=prepared,
    )
    publish_composable_chaser_successor_run(
        plan,
        scratch_root=tmp_path / "scratch",
    )
    source_receipt = tmp_path / "alignment-source-receipt.json"
    ensure_exact_immutable_child_validation_receipt(
        archive,
        run_path=plan.run_path,
        manifest_attr=MANIFEST_ATTR,
        manifest_digest_attr=MANIFEST_DIGEST_ATTR,
        palette_commit="0" * 40,
        output_json=source_receipt,
        expected_recording_id="recording",
    )

    output_dir = tmp_path / "plots"
    assert (
        plot_main(
            [
                str(archive),
                "--run-name",
                "alignment-v1",
                "--bundle-name",
                "alignment-recipe-v1",
                "--expected-recording-id",
                "recording",
                "--output-dir",
                str(output_dir),
                "--source-validation-receipt",
                str(source_receipt),
            ]
        )
        == 0
    )
    plot_receipt_path = (
        output_dir / "alignment-recipe-v1_body_alignment_plot_receipt.json"
    )
    receipt = json.loads(plot_receipt_path.read_text(encoding="utf-8"))
    assert receipt["plot_recipe_id"] == PLOT_RECIPE_ID
    assert receipt["source_binding"]["verification_mode"] == (
        "receipt_bound_targeted_array_rehash_v1"
    )
    assert set(receipt["source_binding"]["verified_array_names"]) == set(
        SUMMARY_VIEW_ARRAY_NAMES
    )
    assert receipt["plot_parameters"]["viewer_policy"] == {
        "body_origin_distance_substitution": "prohibited",
        "interpolation": "prohibited",
        "motion_heading_fallback": "prohibited",
        "rebinning": "prohibited",
        "scientific_groupby": "prohibited",
        "scientific_recomputation": False,
    }
    assert all(Path(output["path"]).is_file() for output in receipt["outputs"])
