from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fisheye.cluster.subject_masks import recording as mod


def _write_cache(tmp_path: Path) -> Path:
    payload_path = tmp_path / "source" / "cache.bin"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(bytes(range(8)))
    manifest_path = payload_path.with_suffix(".flat_roi_cache.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "palette_roi_cache_flat_bin_v1",
                "layout": "flat_bin_v1",
                "cache_complete": True,
                "array": {
                    "bin_path": payload_path.name,
                    "dtype": "uint8",
                    "shape": [2, 2, 2],
                    "order": "C",
                    "total_bytes": 8,
                    "sha256": hashlib.sha256(bytes(range(8))).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _args(tmp_path: Path, *, stage: str) -> SimpleNamespace:
    return SimpleNamespace(
        analysis_zarr=tmp_path / "analysis.zarr",
        registry=tmp_path / "registry.sqlite",
        run_label="combined_canary",
        stage=stage,
        crop_run="crop_exact",
        device="0",
        batch_size=128,
        mask_probs_shard_rois=2048,
        finalize_num_workers=16,
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=5,
        progress_dir=tmp_path / "progress",
        handoff_package_dir=None,
        refined_keypoint_run="refined_keypoints_exact",
    )


def test_stage_flat_roi_cache_copies_payload_and_publishes_manifest_last(
    tmp_path: Path,
) -> None:
    source_manifest = _write_cache(tmp_path)
    staged_manifest, details = mod._stage_flat_roi_cache_manifest(
        source_manifest,
        staging_dir=tmp_path / "scratch",
    )

    payload = json.loads(staged_manifest.read_text(encoding="utf-8"))
    staged_bin = staged_manifest.parent / payload["array"]["bin_path"]
    assert staged_bin.read_bytes() == bytes(range(8))
    assert details["policy"] == "node_scratch_staged_flat_cache"
    assert payload["staging"]["effective_manifest_path"] == str(staged_manifest)

    unrelated = staged_manifest.parent / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    mod._cleanup_staged_cache(staged_manifest)
    assert not staged_manifest.exists()
    assert not staged_bin.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert source_manifest.exists()


def test_stage_flat_roi_cache_rejects_same_size_payload_corruption(
    tmp_path: Path,
) -> None:
    source_manifest = _write_cache(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload_path = source_manifest.parent / manifest["array"]["bin_path"]
    payload_path.write_bytes(bytes(reversed(range(8))))

    try:
        mod._stage_flat_roi_cache_manifest(
            source_manifest,
            staging_dir=tmp_path / "scratch",
        )
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("same-size ROI-cache corruption was accepted")


def test_inference_pipeline_has_no_keypoint_dependency(tmp_path: Path) -> None:
    command = mod._pipeline_args(
        _args(tmp_path, stage="inference"),
        cache_manifest=tmp_path / "staged.flat_roi_cache.json",
    )

    assert "--no-assignment-keypoints" in command
    assert "--assignment-keypoints-run" not in command
    assert "--defer-registry-status" in command
    assert command[command.index("--subject-output-parent") + 1] == (
        "subject_mask_shard_runs"
    )
    assert "--require-production-proof" in command
    assert command[command.index("--mask-probs-shard-rois") + 1] == "2048"


def test_finalization_pipeline_binds_exact_keypoints_and_sampled_contours(
    tmp_path: Path,
) -> None:
    command = mod._pipeline_args(
        _args(tmp_path, stage="finalization"),
        cache_manifest=None,
    )

    assert command[command.index("--assignment-keypoint-group") + 1] == (
        "refined_keypoints_runs"
    )
    assert command[command.index("--assignment-keypoints-run") + 1] == (
        "refined_keypoints_exact"
    )
    assert "--write-sampled-component-contours" in command
    assert "--no-write-component-contours" in command
    assert "--defer-registry-status" in command
    assert command[command.index("--subject-output-parent") + 1] == (
        "subject_mask_shard_runs"
    )
    assert "--require-production-proof" in command
