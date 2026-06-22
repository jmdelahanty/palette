from pathlib import Path

import zarr

from fisheye.shared.zarr_run_completion import mark_run_complete
from fisheye.utils.finalize_refine_keypoints_batch_registry import (
    finalize_refine_keypoints_batch_registry,
)


def test_finalize_refine_keypoints_batch_registry_dry_run_selects_matching_run(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "rec" / "zarr" / "rec_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_keypoints_runs")
    parent.attrs["latest"] = "refined_keypoints_001"
    run = parent.create_group("refined_keypoints_001")
    run.attrs["source_keypoints_run"] = "keypoints_001"
    run.attrs["method"] = "refine_keypoints"
    run.attrs["summary_statistics"] = {
        "total_rois": 10,
        "refined_success": 9,
        "usable_keypoints": 8,
        "pass_rate_percent": 90.0,
    }
    mark_run_complete(run, parent_group=parent, run_name="refined_keypoints_001")

    run_root = tmp_path / "batch"
    run_root.mkdir()
    (run_root / "zarr_paths.txt").write_text(str(zarr_path) + "\n", encoding="utf-8")

    report = finalize_refine_keypoints_batch_registry(
        run_root,
        registry_path=tmp_path / "missing.sqlite",
        keypoint_run="keypoints_001",
        apply=False,
    )

    assert report["status"] == "ok"
    assert report["finalized_count"] == 1
    assert report["upserted_status_rows"] == 0
    assert report["finalized"][0]["run_name"] == "refined_keypoints_001"
    assert report["finalized"][0]["coverage_pct"] == 90.0
