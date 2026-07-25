from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.utils import run_detection_artifact as mod


def _write_group(path: Path, *, attributes: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attributes or {},
            }
        ),
        encoding="utf-8",
    )


def _write_array(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )


def test_strict_json_report_rejects_non_finite_json(tmp_path: Path) -> None:
    good = tmp_path / "good" / "zarr.json"
    good.parent.mkdir()
    good.write_text('{"attributes": {"ok": 1}}', encoding="utf-8")
    bad = tmp_path / "bad" / "zarr.json"
    bad.parent.mkdir()
    bad.write_text('{"attributes": {"value": NaN}}', encoding="utf-8")

    report = mod.strict_json_report(tmp_path)

    assert report["status"] == "fail"
    assert report["bad_json_files"] == 1
    assert report["bad_files"][0]["path"] == "bad/zarr.json"


def test_artifact_reports_reject_canonical_instance_identity(tmp_path: Path) -> None:
    run = tmp_path / "run_group"
    _write_group(
        run,
        attributes={
            "coordinate_contract_mode": "artifact_unbound",
            "coordinate_contract": mod.UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
            "stage_selector_eligible": False,
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "palette_run_stage": "detection_artifact",
            "instance_key_recording_identity": "forbidden",
        },
    )
    for name in mod.REQUIRED_DETECT_ARRAYS:
        _write_array(run / name)
    _write_array(run / "instance_key")

    arrays = mod.required_arrays_report(run)
    metadata = mod.artifact_run_metadata_report(run)

    assert arrays["status"] == "fail"
    assert arrays["forbidden_arrays"] == ["instance_key"]
    assert metadata["status"] == "fail"
    assert "instance_key_recording_identity" in metadata["errors"][0]


def test_build_detection_artifact_packages_detect_run_group(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    artifact_dir = tmp_path / "palette_run_group_artifact"
    tarball = tmp_path / "artifact.tar.gz"
    frame_index = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "camera_serial": ["2010093"] * 4,
                "clip_id": ["clip_000003"] * 4,
                "clip_local_frame_index": np.arange(4, dtype=np.int64),
                "parent_frame_index": np.arange(100, 104, dtype=np.int64),
            }
        ),
        frame_index,
    )
    detect_kwargs: dict[str, object] = {}

    def fake_detect_yolo(**kwargs):
        detect_kwargs.update(kwargs)
        print("model summary should not contaminate summary stdout")
        scratch_zarr = Path(kwargs["output_zarr"])
        run_group = scratch_zarr / mod.RUN_FAMILY / "detect_fake"
        _write_group(
            run_group,
            attributes={
                "coordinate_contract_mode": "artifact_unbound",
                "coordinate_contract": mod.UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
                "stage_selector_eligible": False,
                "palette_run_completion_contract": "palette.zarr_run_completion.v1",
                "palette_run_completion_status": "complete",
                "palette_run_stage": "detection_artifact",
            },
        )
        for name in mod.REQUIRED_DETECT_ARRAYS:
            _write_array(run_group / name)
        zarr_payload = json.loads((run_group / "zarr.json").read_text(encoding="utf-8"))
        zarr_payload["attributes"]["timing_summary"] = {
                "decode_backend_effective": kwargs["decode_backend"],
                "frames_processed": 4,
        }
        (run_group / "zarr.json").write_text(json.dumps(zarr_payload), encoding="utf-8")
        return "detect_fake"

    monkeypatch.setattr(mod, "build_detection_candidate", fake_detect_yolo)
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc",
            "short_hash": "abc",
            "branch": "test",
            "is_dirty": False,
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **kwargs: {
            "environment": {
                "environment_name": "palette-py311",
                "python_executable": "/fake/env/bin/python",
            },
            "platform": {
                "hostname": "cluster-node",
                "lsf": {"job_id": "123", "queue": "gpu_l4"},
            },
            "gpu": {
                "available": True,
                "backend": "cuda",
                "devices": [{"index": 0, "name": "NVIDIA L4"}],
            },
            "env_vars": {"CUDA_VISIBLE_DEVICES": "0"},
        },
    )

    summary = mod.build_detection_artifact(
        video_path=video,
        target_zarr=target_zarr,
        artifact_dir=artifact_dir,
        model_path=tmp_path / "model.pt",
        decode_backend="pynvvc_nv12_rgb",
        tarball_output=tarball,
        command=["scripts/py", "-m", "fisheye.utils.run_detection_artifact"],
        workflow_id="sleepyfish_detect_smoke",
        recording_id="sleepyfish_recording",
        clip_id="clip_000003",
        clip_index=3,
        camera_serial="2010093",
        recording_frame_index=frame_index,
    )

    assert summary["status"] == "ok"
    assert summary["run_name"] == "detect_fake"
    assert (artifact_dir / "run_group" / "frame_indices" / "zarr.json").exists()
    manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema"] == mod.ARTIFACT_SCHEMA
    assert manifest["target_group_path"] == "detection_artifact_runs/detect_fake"
    assert (
        manifest["intended_target_group_path"]
        == "clips/clip_000003/cameras/2010093/detection_artifact_runs/detect_fake"
    )
    assert manifest["artifact_scope"] == "clip_camera"
    assert manifest["clip_context"] == {
        "camera_serial": "2010093",
        "clip_camera_key": "clip_000003/camera_2010093",
        "clip_id": "clip_000003",
        "clip_index": 3,
        "recording_id": "sleepyfish_recording",
        "scope": "clip_camera",
        "workflow_id": "sleepyfish_detect_smoke",
    }
    assert manifest["latest_policy"] == "do_not_set_latest"
    assert manifest["selector_policy"] == "never_select_or_promote_v1"
    assert manifest["stage_selector_eligible"] is False
    assert manifest["run_family"] == "detection_artifact_runs"
    assert manifest["validation"] == {
        "canonical_write": "not_performed",
        "artifact_run_metadata": "pass",
        "required_arrays": "pass",
        "strict_json": "pass",
    }
    assert manifest["artifact_timing"]["copy_run_group_seconds_total"] >= 0.0
    assert manifest["provenance"]["decoder_backend"] == "pynvvc_nv12_rgb"
    assert manifest["provenance"]["clip_context"]["clip_id"] == "clip_000003"
    assert manifest["provenance"]["runtime"]["gpu"]["devices"][0]["name"] == "NVIDIA L4"
    assert manifest["provenance"]["runtime"]["platform"]["lsf"]["queue"] == "gpu_l4"
    assert (
        manifest["provenance"]["runtime"]["environment"]["python_executable"]
        == "/fake/env/bin/python"
    )
    assert summary["artifact_scope"] == "clip_camera"
    assert (
        summary["intended_target_group_path"]
        == "clips/clip_000003/cameras/2010093/detection_artifact_runs/detect_fake"
    )
    assert summary["artifact_timing"]["tarball_seconds_total"] >= 0.0
    assert "instance_key_recording_identity" not in detect_kwargs
    assert "instance_key_frame_indices" not in detect_kwargs
    assert "instance_key_frame_mapping_source" not in detect_kwargs
    assert detect_kwargs["coordinate_contract_mode"] == "artifact_unbound"
    assert detect_kwargs["output_run_family"] == "detection_artifact_runs"
    captured = capsys.readouterr()
    assert "model summary should not contaminate summary stdout" not in captured.out
    assert "model summary should not contaminate summary stdout" in captured.err
    assert tarball.exists()
    with tarfile.open(tarball, "r:gz") as tar:
        names = set(tar.getnames())
    assert "palette_run_group_artifact/artifact_manifest.json" in names
    assert "palette_run_group_artifact/run_group/frame_indices/zarr.json" in names


def test_build_detection_artifact_can_request_deterministic_run_name(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    artifact_dir = tmp_path / "artifact"

    def fake_detect_yolo(**kwargs):
        run_name = kwargs["run_name"]
        scratch_zarr = Path(kwargs["output_zarr"])
        run_group = scratch_zarr / mod.RUN_FAMILY / run_name
        _write_group(
            run_group,
            attributes={
                "coordinate_contract_mode": "artifact_unbound",
                "coordinate_contract": mod.UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
                "stage_selector_eligible": False,
                "palette_run_completion_contract": "palette.zarr_run_completion.v1",
                "palette_run_completion_status": "complete",
                "palette_run_stage": "detection_artifact",
            },
        )
        for name in mod.REQUIRED_DETECT_ARRAYS:
            _write_array(run_group / name)
        return run_name

    monkeypatch.setattr(mod, "build_detection_candidate", fake_detect_yolo)

    summary = mod.build_detection_artifact(
        video_path=video,
        target_zarr=target_zarr,
        artifact_dir=artifact_dir,
        tarball_output=tmp_path / "artifact.tar.gz",
        run_name="detect_planned_clip_000000_cam2010093",
    )

    assert summary["run_name"] == "detect_planned_clip_000000_cam2010093"
    assert summary["target_group_path"] == (
        "detection_artifact_runs/detect_planned_clip_000000_cam2010093"
    )


def test_build_detection_artifact_rejects_selector_promotion(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)

    with pytest.raises(ValueError, match="promotion policies are forbidden"):
        mod.build_detection_artifact(
            video_path=video,
            target_zarr=target_zarr,
            artifact_dir=tmp_path / "artifact",
            latest_policy="set_latest_explicit",
        )
