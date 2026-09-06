"""Adapter contract tests.

The source/canonical-detection fixture uses the existing explicit authority
seams. The not-yet-integrated metric validator is stubbed only for these
adapter tests; it is not evidence of metric-malformation enforcement.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import shutil

import cv2
import numpy as np
import pytest

from fisheye.analysis_workflows.materializers import (
    arena_geometry_candidates as candidates,
)
from fisheye.analysis_workflows.materializers import (
    arena_geometry_comparison as comparison,
)
from fisheye.analysis_workflows.materializers import (
    arena_geometry_fit_review as fit_review,
)
from fisheye.diagnostics.probe_recording_dish_rim_fit import write_review_package
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.zarr_io import open_zarr_root
from tests.unit.fisheye.test_arena_geometry_candidates import _palette_binding
from tests.unit.fisheye.test_arena_geometry_comparison import (
    _archive_with_candidates,
    _write_nested_detection_source,
)
from tests.unit.fisheye.test_arena_geometry_fit_review import _review_package


def _metric_report(report):
    """Explicit contract fixture, not image-derived calibration evidence."""
    report = deepcopy(report)
    metric_windows = {}
    geometry = report["consensus_fit"]["geometry"]
    report["consensus_fit"]["selection_reason"] = (
        "observed_window_medoid_no_radius_averaging_v1"
    )
    for name, window in report["windows"].items():
        window["fit"]["geometry"] = deepcopy(geometry)
        window["fit"]["coordinate_space"] = "camera_native_pixels"
        window["fit"]["frozen_candidates"][0]["geometry"] = deepcopy(geometry)
        metric_windows[name] = {
            "composite_pixel_sha256": window["composite_pixel_sha256"],
            "selected_candidate_id": "candidate_000",
            "candidates": {
                "candidate_000": {
                    "geometry": deepcopy(geometry),
                    "coordinate_space": "camera_native_pixels",
                    "image_shape_px": report["source"]["image_shape_px"],
                    "angular_sample_count": 720,
                    "radial_band_px": 4.0,
                    "gradient_support_cutoff": 1.0,
                    "angular_support_fraction": 0.95,
                    "visible_angular_fraction": 1.0,
                    "longest_unsupported_arc_degrees": 18.0,
                    "radial_residual_p95_px": 0.5,
                    "median_absolute_radial_offset_px": 0.1,
                    "radial_gradient_median": 100.0,
                    "quadrant_support_fractions": [0.95] * 4,
                    "quality_flags": [],
                }
            },
        }
    report["scientific_recipe"] = {
        "recipe_id": "top_rim_preferred_v1",
        "recipe_version": 1,
        "status": "experimental_shadow_only_not_calibrated",
        "effective_parameters": {},
        "permitted_scientific_parameter_overrides": [],
        "intended_target_feature": "visible_dish_top_rim_edge",
        "physical_feature_identified": False,
    }
    report["rim_metrics"] = {
        "schema_id": "palette.diagnostics.dish_rim_metrics",
        "schema_version": 1,
        "status": "measured",
        "semantic_correspondence": "projected_edges_unresolved",
        "physical_feature_identified": False,
        "source_binding_sha256": hashlib.sha256(
            strict_json_dumps(report["source"]).encode()
        ).hexdigest(),
        "windows": metric_windows,
        "temporal": {
            "center_max_pairwise_distance_px": 0.0,
            "radius_range_px": 0.0,
            "rim_family_center_max_distance_px": 0.0,
            "minimum_candidate_count": 1,
            "rim_family_radius_hausdorff_distance_px": 0.0,
        },
        "quality_flags": ["projected_edges_unresolved"],
    }
    return report


def _shadow_archive(tmp_path, monkeypatch, *, stub_metric_validator=True):
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    detection = _write_nested_detection_source(archive)
    # No reviewed Palette candidate survives: the public consumer uses only the
    # immutable blind-fit package and the producer candidate.
    root = open_zarr_root(archive, mode="a")
    del root[f"analysis/{candidates.CANDIDATE_RUNS_PARENT}/{palette}"]
    acquisition_record = dict(
        root[f"analysis/{candidates.CANDIDATE_RUNS_PARENT}/{acquisition}"].attrs[
            "candidate_record"
        ]
    )
    package_dir = _review_package(tmp_path / "package")
    report_path = package_dir / "fit_report.json"
    report = _metric_report(json.loads(report_path.read_text()))
    for name, window in report["windows"].items():
        shutil.copyfile(
            package_dir / f"{name}_palette_fit.png",
            package_dir / f"{name}_acquisition_reveal.png",
        )
        pixels = np.zeros((480, 640), dtype=np.uint8)
        composite_path = package_dir / f"{name}_temporal_median.png"
        assert cv2.imwrite(str(composite_path), pixels)
        window["composite_pixel_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        report["rim_metrics"]["windows"][name]["composite_pixel_sha256"] = window[
            "composite_pixel_sha256"
        ]
        window["files"] = {
            "temporal_median": {
                "path": composite_path.name,
                "sha256": hashlib.sha256(composite_path.read_bytes()).hexdigest(),
            }
        }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    window_support = {
        "status": "measured",
        "geometry_frozen": True,
        "angular_edge_support_fraction": 0.95,
        "median_absolute_radial_offset_px": 0.1,
    }
    reveal = {
        "schema_id": fit_review.ACQUISITION_REVEAL_SCHEMA_ID,
        "schema_version": 1,
        "fit_report": {
            "path": report_path.name,
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
        "acquisition_boundary_edge_support": {
            "status": "measured",
            "fit_frozen_before_measurement": True,
            "coordinate_space": "camera_native_pixels",
            "geometry": acquisition_record["physical_inner_rim"]["geometry"],
            "source_observation_sha256": acquisition_record["acquisition_source"][
                "source_observation_sha256"
            ],
            "windows": {
                name: dict(window_support) for name in ("early", "middle", "late")
            },
        },
    }
    (package_dir / "acquisition_reveal.json").write_text(
        json.dumps(reveal), encoding="utf-8"
    )
    write_review_package(package_dir, acquisition_revealed=True)
    plan = fit_review.build_arena_geometry_fit_review_plan(
        archive, review_package_dir=package_dir
    )
    fit_review.publish_arena_geometry_fit_review(
        plan, scratch_root=tmp_path / "scratch"
    )
    monkeypatch.setattr(
        candidates,
        "_source_camera_candidate_binding",
        lambda *_a, **_k: _palette_binding(),
    )
    if stub_metric_validator:
        monkeypatch.setattr(
            fit_review, "validate_rim_metrics", lambda *_a, **_k: None, raising=False
        )
        monkeypatch.setattr(
            fit_review,
            "validate_rim_metrics_consensus",
            lambda *_a, **_k: None,
            raising=False,
        )
    return archive, {
        "acquisition_candidate_run": acquisition,
        "fit_review_run": plan.run_name,
        "detect_source_group_path": detection,
    }


def test_public_shadow_adapter_needs_no_reviewed_palette_candidate(
    tmp_path, monkeypatch
):
    archive, arguments = _shadow_archive(tmp_path, monkeypatch)
    before = {
        path.relative_to(archive): path.read_bytes()
        for path in archive.rglob("*")
        if path.is_file()
    }
    result = comparison.build_arena_geometry_shadow_evaluation(archive, **arguments)
    assert result["evaluation"]["thresholds_satisfied"] is None
    assert result["evaluation"]["selection_performed"] is False
    assert result["evaluation"]["scientific_acceptance_created"] is False
    assert result["evidence"]["semantic_compatibility"] == "projected_edges_unresolved"
    assert result["evidence"]["same_feature_physical_boundary_metrics"] is None
    assert result["evidence"]["metrics"]["detection_row_count"] == 3
    assert before == {
        path.relative_to(archive): path.read_bytes()
        for path in archive.rglob("*")
        if path.is_file()
    }


def test_shadow_adapter_fails_closed_without_metric_validator(tmp_path, monkeypatch):
    archive, arguments = _shadow_archive(tmp_path, monkeypatch)
    monkeypatch.delattr(fit_review, "validate_rim_metrics")
    with pytest.raises(ValueError, match="validator is not available"):
        comparison.build_arena_geometry_shadow_evaluation(archive, **arguments)


def test_shadow_adapter_reopens_real_immutable_package_before_metrics(
    tmp_path, monkeypatch
):
    archive, arguments = _shadow_archive(tmp_path, monkeypatch)
    root = open_zarr_root(archive, mode="a")
    path = f"analysis/{fit_review.FIT_REVIEW_RUNS_PARENT}/{arguments['fit_review_run']}"
    root[path][fit_review.FIT_REPORT_ARRAY][0] = 0
    with pytest.raises(ValueError, match="Fit-review run is invalid"):
        comparison.build_arena_geometry_shadow_evaluation(archive, **arguments)


def test_shadow_adapter_rejects_wrong_live_coordinate_binding(tmp_path, monkeypatch):
    archive, arguments = _shadow_archive(tmp_path, monkeypatch)
    coordinate, arena, source = _palette_binding()
    coordinate["pixel_frame_record_sha256"] = "f" * 64
    monkeypatch.setattr(
        candidates,
        "_source_camera_candidate_binding",
        lambda *_a, **_k: (coordinate, arena, source),
    )
    with pytest.raises(ValueError, match="coordinate/arena bindings disagree"):
        comparison.build_arena_geometry_shadow_evaluation(archive, **arguments)


def test_shadow_adapter_rejects_nonfinite_detection_rows(tmp_path, monkeypatch):
    archive, arguments = _shadow_archive(tmp_path, monkeypatch)
    root = open_zarr_root(archive, mode="a")
    root[arguments["detect_source_group_path"]]["instances"]["bbox_norm_coords"][
        0, 0
    ] = float("nan")
    with pytest.raises(ValueError):
        comparison.build_arena_geometry_shadow_evaluation(archive, **arguments)


def test_read_only_cli_reaches_real_shadow_adapter(tmp_path, monkeypatch, capsys):
    from fisheye.utils.evaluate_arena_geometry_shadow import main

    archive, arguments = _shadow_archive(tmp_path, monkeypatch)
    assert (
        main(
            [
                str(archive),
                "--acquisition-candidate-run",
                arguments["acquisition_candidate_run"],
                "--fit-review-run",
                arguments["fit_review_run"],
                "--detect-source-group",
                arguments["detect_source_group_path"],
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["evaluation"]["selection_performed"] is False
    assert output["evaluation"]["reason_codes"] == ["thresholds_not_calibrated"]
