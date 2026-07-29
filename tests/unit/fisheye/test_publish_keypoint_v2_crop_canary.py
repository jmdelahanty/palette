from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.diagnostics.publish_keypoint_v2_crop_canary import (
    _pose_binding,
    _require_below,
    _stage_cache,
)


def test_canary_paths_fail_closed_outside_namespace(tmp_path: Path) -> None:
    root = tmp_path / "benchmarks"
    accepted = _require_below(root / "run" / "artifact", root, label="test")
    assert accepted == (root / "run" / "artifact").resolve()

    with pytest.raises(ValueError, match="must be below"):
        _require_below(tmp_path / "elsewhere", root, label="test")


def test_cache_stage_keeps_relative_payload_pair_together(tmp_path: Path) -> None:
    source = tmp_path / "nrs"
    source.mkdir()
    manifest = source / "fixture.flat_roi_cache.json"
    payload = source / "fixture.flat_roi_cache.bin"
    payload.write_bytes(b"crop-pixels")
    manifest.write_text(
        json.dumps(
            {
                "schema": "palette_roi_cache_flat_bin_v1",
                "layout": "flat_bin_v1",
                "cache_complete": True,
                "array": {"bin_path": payload.name},
            }
        ),
        encoding="utf-8",
    )

    staged_manifest, receipt = _stage_cache(
        manifest,
        destination_dir=tmp_path / "scratch",
    )

    assert staged_manifest.read_bytes() == manifest.read_bytes()
    assert staged_manifest.with_suffix(".bin").read_bytes() == payload.read_bytes()
    assert receipt["payload"]["bytes"] == len(b"crop-pixels")


def test_canary_pose_binding_freezes_three_point_dtype_semantics() -> None:
    binding = _pose_binding("a" * 64)

    assert binding["model"]["sha256"] == "a" * 64
    assert binding["pose_schema"]["keypoint_labels"] == [
        "swim_bladder",
        "eye_left",
        "eye_right",
    ]
    assert binding["pose_schema"]["kpt_shape"] == [3, 2]
    assert binding["pose_schema"]["metadata"]["model_kpt_shape"] == [3, 3]
