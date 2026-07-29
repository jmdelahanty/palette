from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.benchmark_keypoint_quality_publication import (
    build_deterministic_keypoint_v2_source,
    run_deterministic_keypoint_quality_benchmark,
)
from fisheye.shared.zarr.keypoint_schema import KEYPOINT_SCHEMA_V2


def test_deterministic_source_covers_sparse_and_multi_observation_frames() -> None:
    dimensions, arrays, crop, source_manifest, source = (
        build_deterministic_keypoint_v2_source(
            n_frames=32,
            n_instances=32,
            n_keypoints=3,
            source_width=640,
            source_height=480,
            empty_frames=4,
            seed=7,
        )
    )

    KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop,
        skeleton_digest=source.skeleton_digest,
    )
    counts = np.diff(arrays["frame_row_offsets"])
    assert np.count_nonzero(counts == 0) == 4
    assert np.count_nonzero(counts > 1) > 0
    assert source_manifest["selector_eligible"] is False
    assert "heading" not in arrays


def test_publication_read_benchmark_is_correct_and_selector_ineligible(
    tmp_path: object,
) -> None:
    root = tmp_path / "keypoint_quality"  # type: ignore[operator]
    destination = root / "fixture.zarr"
    report = root / "fixture.benchmark.json"

    result = run_deterministic_keypoint_quality_benchmark(
        destination=destination,
        run_id="quality_fixture_v1",
        shadow_root=root,
        n_frames=64,
        n_instances=64,
        n_keypoints=3,
        source_width=640,
        source_height=480,
        empty_frames=8,
        seed=11,
        random_frame_count=16,
        window_count=4,
        window_frames=7,
        full_scan_batch_rows=16,
        output_json=report,
    )

    assert result["status"] == "passed"
    assert result["promotion_eligible"] is False
    assert result["profile_promoted"] is False
    assert result["correctness"] == {
        "publication_gate": "passed",
        "direct_consolidated_manifest_equal": True,
        "offset_index_equal": True,
        "random_frame_digest_equal": True,
        "window_digest_equal": True,
        "full_scan_digests_equal": True,
    }
    assert result["reads"]["offset_index"]["reads"] == 1
    assert result["reads"]["offset_index"]["later_workload_offset_reads"] == 0
    assert result["publication"]["selector_eligible"] is False
    assert result["publication"]["registry_registered"] is False
    assert result["source_characteristics"]["empty_frame_count"] == 8
    assert result["source_characteristics"]["multi_observation_frame_count"] > 0
    assert destination.is_dir()
    assert report.is_file()


def test_benchmark_rejects_evidence_outside_shadow_root(tmp_path: object) -> None:
    root = tmp_path / "keypoint_quality"  # type: ignore[operator]
    destination = root / "fixture.zarr"

    with pytest.raises(ValueError, match="evidence must be below shadow_root"):
        run_deterministic_keypoint_quality_benchmark(
            destination=destination,
            run_id="quality_fixture_v1",
            shadow_root=root,
            n_frames=8,
            n_instances=8,
            n_keypoints=3,
            source_width=640,
            source_height=480,
            empty_frames=1,
            output_json=tmp_path / "outside.json",  # type: ignore[operator]
        )

    assert not destination.exists()
