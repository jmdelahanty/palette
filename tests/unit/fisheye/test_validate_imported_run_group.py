from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import validate_imported_run_group as mod
from fisheye.utils.run_detection_artifact import REQUIRED_DETECT_ARRAYS, tree_hash


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_group(path: Path, attrs: dict | None = None) -> None:
    _write_json(
        path / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": attrs or {},
        },
    )


def _write_array(path: Path) -> None:
    _write_json(
        path / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [0],
            "data_type": "int32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {},
        },
    )


def _write_imported_detect_run(
    tmp_path: Path,
    *,
    run_family_path: str = "detect_runs",
) -> tuple[Path, Path]:
    zarr_path = tmp_path / "recording_analysis.zarr"
    run_name = "detect_fake"
    run_group = zarr_path / run_family_path / run_name
    source_video = tmp_path / "camera.mp4"
    source_video.write_bytes(b"fake video")
    tarball = tmp_path / "artifact.tar.gz"
    tarball.write_bytes(b"fake tarball")

    _write_group(zarr_path)
    _write_group(zarr_path / run_family_path)
    _write_group(
        run_group,
        {
            "model_path": "/models/best.pt",
            "hostname": "worker01",
            "git_commit": "abc123",
            "decode_backend_effective": "pynvvc_nv12_rgb",
            "timing_summary": {"processing_seconds_total": 12.5},
            "provenance": {
                "command": "scripts/py -m fisheye.utils.run_detection_artifact ...",
                "git": {"commit": "abc123"},
                "platform": {"hostname": "worker01"},
                "artifacts": {"model_path": "/models/best.pt", "device": "cuda"},
                "timing": {"processing_seconds_total": 12.5},
                "parameters": {"decode_backend_effective": "pynvvc_nv12_rgb"},
            },
        },
    )
    for name in REQUIRED_DETECT_ARRAYS:
        _write_array(run_group / name)

    digest = tree_hash(run_group)
    manifest = {
        "layout": "detect_yolo_sparse_v1",
        "target_archive_path": str(zarr_path.resolve()),
        "target_group_path": f"{run_family_path}/{run_name}",
        "run_family": "detect_runs",
        "run_family_path": run_family_path,
        "run_name": run_name,
        "source_inputs": [
            {"path": str(source_video), "role": "source_video"},
            {"path": str(zarr_path), "role": "target_analysis_archive"},
        ],
        "provenance": {
            "cluster": {
                "LSB_JOBID": "123",
                "CUDA_VISIBLE_DEVICES": "0",
            }
        },
        "checksums": {"run_group_tree_hash": digest},
    }
    receipt = {
        "schema_version": 1,
        "source_tarball": str(tarball),
        "source_tarball_sha256": mod._sha256_file(tarball),
        "target_archive_path": str(zarr_path.resolve()),
        "target_group_path": f"{run_family_path}/{run_name}",
        "run_family": "detect_runs",
        "run_family_path": run_family_path,
        "run_name": run_name,
        "layout": "detect_yolo_sparse_v1",
        "final_path": str(run_group.resolve()),
        "manifest": manifest,
    }
    _write_json(zarr_path / run_family_path / ".imports" / f"{run_name}_import_receipt.json", receipt)
    return zarr_path, run_group


def test_validate_imported_run_group_passes_for_receipted_detect_run(tmp_path: Path) -> None:
    zarr_path, _run_group = _write_imported_detect_run(tmp_path)

    result = mod.validate_imported_run_group(
        zarr_path=zarr_path,
        target_group_path="detect_runs/detect_fake",
    )

    assert result["status"] == "ok"
    assert result["validations"]["receipt"]["status"] == "pass"
    assert result["validations"]["required_arrays"]["status"] == "pass"
    assert result["validations"]["run_group_tree_hash"]["status"] == "pass"
    assert result["validations"]["provenance"]["status"] == "pass"
    assert result["validations"]["source_tarball"]["status"] == "pass"


def test_validate_imported_run_group_passes_for_clip_local_target_path(tmp_path: Path) -> None:
    family_path = "clips/clip_000000/cameras/2010093/detect_runs"
    zarr_path, _run_group = _write_imported_detect_run(tmp_path, run_family_path=family_path)

    result = mod.validate_imported_run_group(
        zarr_path=zarr_path,
        target_group_path=f"{family_path}/detect_fake",
    )

    assert result["status"] == "ok"
    assert result["run_family"] == "detect_runs"
    assert result["run_family_path"] == family_path
    assert result["receipt_path"] == str(
        (zarr_path / family_path / ".imports" / "detect_fake_import_receipt.json").resolve()
    )
    assert result["validations"]["receipt_identity"]["status"] == "pass"


def test_validate_imported_run_group_fails_on_hash_mismatch(tmp_path: Path) -> None:
    zarr_path, run_group = _write_imported_detect_run(tmp_path)
    _write_array(run_group / "extra_array")

    result = mod.validate_imported_run_group(
        zarr_path=zarr_path,
        target_group_path="detect_runs/detect_fake",
    )

    assert result["status"] == "failed"
    assert result["validations"]["run_group_tree_hash"]["status"] == "fail"
    assert "final run_group_tree_hash mismatch" in "\n".join(result["errors"])


def test_validate_imported_run_group_allows_detect_quality_mutable_child(tmp_path: Path) -> None:
    zarr_path, run_group = _write_imported_detect_run(tmp_path)
    _write_group(run_group / "quality_reports")
    _write_group(run_group / "quality_reports" / "detect_quality_fake")
    _write_array(run_group / "quality_reports" / "detect_quality_fake" / "quality_flags")

    result = mod.validate_imported_run_group(
        zarr_path=zarr_path,
        target_group_path="detect_runs/detect_fake",
    )

    assert result["status"] == "ok"
    assert result["validations"]["run_group_tree_hash"]["status"] == "pass"
    assert (
        result["validations"]["run_group_tree_hash"]["hash_mode"]
        == "core_excluding_mutable_children"
    )
    assert result["validations"]["run_group_tree_hash"]["excluded_children"] == ["quality_reports"]


def test_validate_imported_run_group_fails_when_provenance_is_incomplete(tmp_path: Path) -> None:
    zarr_path, run_group = _write_imported_detect_run(tmp_path)
    _write_group(run_group, attrs={})

    result = mod.validate_imported_run_group(
        zarr_path=zarr_path,
        target_group_path="detect_runs/detect_fake",
        validate_source_tarball=False,
    )

    assert result["status"] == "failed"
    assert result["validations"]["provenance"]["status"] == "fail"
    assert "provenance missing" in "\n".join(result["errors"])
