"""Real package/publisher/loader checks with bounded synthetic decoded medians."""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers import arena_geometry_fit_review as review
from fisheye.diagnostics import probe_recording_dish_rim_fit as probe


def _probe_args(tmp_path, monkeypatch, *, recipe=None, suffix="probe"):
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"synthetic-decoder-boundary")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "frames_received": 1000,
                "fps": 10.0,
                "video_metadata": {
                    "camera_serial": "2010093",
                    "geometry": {"source_width": 512, "source_height": 512},
                },
            }
        )
    )
    keyframes = tmp_path / "keyframes.json"
    keyframes.write_text(
        json.dumps(
            {
                "total_frames": 1000,
                "fps": 10.0,
                "keyframe_frames": list(range(0, 1000, 10)),
            }
        )
    )
    image = np.full((512, 512), 30, np.uint8)
    cv2.circle(image, (256, 256), 185, 230, 3, cv2.LINE_AA)
    cv2.circle(image, (256, 256), 215, 120, 3, cv2.LINE_AA)

    def decode(path, specs, **kwargs):
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        return (
            {name: image.copy() for name in probe.WINDOW_NAMES},
            {name: digest for name in probe.WINDOW_NAMES},
            {
                "seeks": [
                    {
                        "window": spec.name,
                        "target_frame_index": frame,
                        "decoded_frame_sha256": digest,
                    }
                    for spec in specs
                    for frame in spec.frame_indices
                ]
            },
        )

    monkeypatch.setattr(probe, "decode_keyframe_window_medians_pynvvc", decode)
    monkeypatch.setattr(probe, "_utc_now", lambda: "2026-09-06T00:00:00+00:00")
    argv = [
        "--video",
        str(video),
        "--summary",
        str(summary),
        "--keyframes",
        str(keyframes),
        "--output-dir",
        str(tmp_path / suffix),
        "--coarse-max-dimension-px",
        "512",
        "--max-keyframes-per-window",
        "5",
    ]
    if recipe:
        argv += ["--fit-recipe", recipe]
    return probe.build_parser().parse_args(argv)


def test_default_recipe_report_is_identical_to_explicit_legacy(tmp_path, monkeypatch):
    args = _probe_args(tmp_path, monkeypatch)
    first = probe.run_probe(args).read_bytes()
    args.output_dir = tmp_path / "explicit_legacy"
    args.fit_recipe = "legacy_v1"
    assert probe.run_probe(args).read_bytes() == first
    report = json.loads(first)
    assert "rim_metrics" not in report
    assert "scientific_recipe" not in report
    assert report["fit_method"] == probe.FIT_METHOD


def test_opt_in_real_package_import_retains_metrics_and_original_medians(
    tmp_path, monkeypatch
):
    args = _probe_args(tmp_path, monkeypatch, recipe=probe.TOP_RIM_RECIPE)
    report_path = probe.run_probe(args)
    frozen = report_path.read_bytes()
    report = json.loads(frozen)
    assert (
        report["scientific_recipe"]["effective_parameters"] == probe.TOP_RIM_PARAMETERS
    )
    assert report["target_feature"] == "visible_dish_top_rim_edge"
    assert report["rim_metrics"]["physical_feature_identified"] is False
    review.validate_rim_metrics_consensus(
        report["consensus_fit"], windows=report["windows"]
    )
    archive = tmp_path / "analysis.zarr"
    zarr.open_group(
        str(archive), mode="w", zarr_format=3, use_consolidated=False
    ).require_group("analysis")
    plan = review.build_arena_geometry_fit_review_plan(
        archive, review_package_dir=report_path.parent
    )
    review.publish_arena_geometry_fit_review(plan, scratch_root=tmp_path / "scratch")
    evidence = review.load_arena_geometry_fit_review_evidence(
        archive, run_name=plan.run_name
    )
    assert evidence.fit_report_bytes == frozen
    root = zarr.open_group(str(archive), mode="r", zarr_format=3, use_consolidated=True)
    run = root[f"analysis/{review.FIT_REVIEW_RUNS_PARENT}/{plan.run_name}"]
    assert run.attrs["stage_selector_eligible"] is False
    assert "latest" not in root[f"analysis/{review.FIT_REVIEW_RUNS_PARENT}"].attrs
    for name in probe.WINDOW_NAMES:
        artifact = evidence.review_record["artifacts"][f"metric_composite_{name}"]
        png = np.asarray(run[artifact["zarr_path"]][:], dtype=np.uint8).tobytes()
        median = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        assert (
            hashlib.sha256(median.tobytes(order="C")).hexdigest()
            == report["windows"][name]["composite_pixel_sha256"]
        )


@pytest.mark.parametrize(
    "tamper", ["source", "missing_metrics", "median", "consensus", "pixel_binding"]
)
def test_new_metric_package_import_refuses_invalid_evidence(
    tmp_path, monkeypatch, tamper
):
    args = _probe_args(tmp_path, monkeypatch, recipe=probe.TOP_RIM_RECIPE)
    report_path = probe.run_probe(args)
    report = json.loads(report_path.read_bytes())
    if tamper == "source":
        report["source"]["camera_serial"] = "wrong_camera"
    elif tamper == "missing_metrics":
        del report["rim_metrics"]
    elif tamper == "consensus":
        report["consensus_fit"]["geometry"]["radius_px"] += 0.5
    elif tamper == "pixel_binding":
        report["windows"]["early"]["composite_pixel_sha256"] = "b" * 64
        report["rim_metrics"]["windows"]["early"]["composite_pixel_sha256"] = "b" * 64
    else:
        (report_path.parent / "early_temporal_median.png").write_bytes(b"wrong pixels")
    report_path.write_text(json.dumps(report))
    # Rehash the outer receipt: malformed inner claims must still be rejected.
    probe.write_review_package(report_path.parent, acquisition_revealed=False)
    archive = tmp_path / "analysis.zarr"
    zarr.open_group(str(archive), mode="w", zarr_format=3, use_consolidated=False)
    with pytest.raises(ValueError):
        review.build_arena_geometry_fit_review_plan(
            archive, review_package_dir=report_path.parent
        )
