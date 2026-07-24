from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from fisheye.diagnostics.plan_canonical_detection_storage_matrix import (
    _parse_scale,
    main,
)


def test_parse_detection_benchmark_scale_contract() -> None:
    scale = _parse_scale("frames_200k:200000:199734:4512:4512")

    assert scale.scale_id == "frames_200k"
    assert scale.dimension_map == {
        "n_frames": 200_000,
        "n_instances": 199_734,
        "source_width": 4512,
        "source_height": 4512,
    }
    with pytest.raises(argparse.ArgumentTypeError, match="scale must be"):
        _parse_scale("frames_200k:200000")


def test_plan_cli_writes_one_exclusive_strict_json_matrix(tmp_path: Path) -> None:
    output = tmp_path / "matrix.json"
    destination_root = tmp_path / "published"
    argv = [
        "--matrix-id",
        "detection_v1",
        "--destination-root",
        str(destination_root),
        "--scale",
        "frames_200k:200000:199734:4512:4512",
        "--repetitions",
        "2",
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "palette.storage_benchmark_matrix"
    assert manifest["summary"] == {
        "destination_collisions": 0,
        "payload_io_performed": False,
        "planned_trials": 16,
        "removed_duplicate_labels": 12,
        "requested_candidate_labels": 20,
        "unique_physical_candidates": 8,
    }
    with pytest.raises(FileExistsError):
        main(argv)
