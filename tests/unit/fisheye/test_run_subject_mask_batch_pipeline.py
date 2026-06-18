from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import zarr

from fisheye.utils import run_subject_mask_batch_pipeline as mod


def test_zarr_paths_from_report_reads_unique_result_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    {"zarr_path": "/data/a_analysis.zarr"},
                    {"zarr_path": "/data/b_analysis.zarr"},
                    {"zarr_path": "/data/a_analysis.zarr"},
                    {"not_zarr_path": "/data/ignored.zarr"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert mod._zarr_paths_from_report(report) == [
        Path("/data/a_analysis.zarr"),
        Path("/data/b_analysis.zarr"),
    ]


def test_zarr_paths_from_report_falls_back_to_plan_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"plans": [{"zarr_path": "/data/planned_analysis.zarr"}]}), encoding="utf-8")

    assert mod._zarr_paths_from_report(report) == [Path("/data/planned_analysis.zarr")]


def test_zarr_paths_from_report_requires_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"results": [{"error": "missing"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="did not contain any zarr_path"):
        mod._zarr_paths_from_report(report)


def test_resolve_crop_run_prefers_latest_any_for_geometry_only(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.create_group("crop_materialized").attrs["crop_storage_mode"] = "materialized"
    crop_parent.create_group("crop_geometry").attrs["crop_storage_mode"] = "geometry_only"

    assert mod._resolve_crop_run(zarr_path) == "crop_geometry"


def test_emit_paths_prints_selected_archives(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.create_group("crop_geometry")
    keypoints_parent = root.require_group("keypoints_runs")
    keypoints_parent.attrs["latest"] = "keypoints_latest"
    keypoints_parent.create_group("keypoints_latest")

    assert mod.main([str(tmp_path), "--emit-paths"]) == 0

    assert capsys.readouterr().out.strip() == str(zarr_path)


def test_inference_command_passes_cache_manifest_and_model_resolution_flags(tmp_path: Path) -> None:
    manifest = tmp_path / "sample.flat_roi_cache.json"
    args = SimpleNamespace(
        registry=tmp_path / "registry.sqlite",
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=7,
        model_require_unique=True,
        model_include_non_success=True,
        device="0",
        batch_size=192,
        mask_probs_dtype="uint8",
        mask_probs_chunk_rois=32,
        output_queue_size=2,
        roi_cache_policy="never",
        roi_live_acceleration="auto",
        roi_live_gpu_chunk_frames=32,
        roi_cache_dir=None,
        roi_cache_manifest=manifest,
        overwrite=False,
    )
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "recording_analysis.zarr"),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoint_run="refined_run",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=True,
    )

    cmd = mod._inference_command(args, plan)

    assert "--roi-cache-manifest" in cmd
    assert cmd[cmd.index("--roi-cache-manifest") + 1] == str(manifest)
    assert "--model-require-unique" in cmd
    assert "--model-include-non-success" in cmd
    assert cmd[cmd.index("--model-top-k") + 1] == "7"
