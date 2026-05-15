from __future__ import annotations

import json
import tarfile
from pathlib import Path

from fisheye.utils import run_detection_artifact as mod


def _write_group(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}),
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


def test_build_detection_artifact_packages_detect_run_group(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    artifact_dir = tmp_path / "palette_run_group_artifact"
    tarball = tmp_path / "artifact.tar.gz"

    def fake_detect_yolo(**kwargs):
        scratch_zarr = Path(kwargs["output_zarr"])
        run_group = scratch_zarr / "detect_runs" / "detect_fake"
        _write_group(run_group)
        for name in mod.REQUIRED_DETECT_ARRAYS:
            _write_array(run_group / name)
        zarr_payload = json.loads((run_group / "zarr.json").read_text(encoding="utf-8"))
        zarr_payload["attributes"] = {
            "timing_summary": {
                "decode_backend_effective": kwargs["decode_backend"],
                "frames_processed": 4,
            }
        }
        (run_group / "zarr.json").write_text(json.dumps(zarr_payload), encoding="utf-8")
        return "detect_fake"

    monkeypatch.setattr(mod, "_detect_yolo", fake_detect_yolo)
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

    summary = mod.build_detection_artifact(
        video_path=video,
        target_zarr=target_zarr,
        artifact_dir=artifact_dir,
        model_path=tmp_path / "model.pt",
        decode_backend="pynvvc_nv12_rgb",
        tarball_output=tarball,
        command=["scripts/py", "-m", "fisheye.utils.run_detection_artifact"],
    )

    assert summary["status"] == "ok"
    assert summary["run_name"] == "detect_fake"
    assert (artifact_dir / "run_group" / "frame_indices" / "zarr.json").exists()
    manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema"] == mod.ARTIFACT_SCHEMA
    assert manifest["target_group_path"] == "detect_runs/detect_fake"
    assert manifest["latest_policy"] == "do_not_set_latest"
    assert manifest["validation"] == {
        "canonical_write": "not_performed",
        "required_arrays": "pass",
        "strict_json": "pass",
    }
    assert manifest["provenance"]["decoder_backend"] == "pynvvc_nv12_rgb"
    assert tarball.exists()
    with tarfile.open(tarball, "r:gz") as tar:
        names = set(tar.getnames())
    assert "palette_run_group_artifact/artifact_manifest.json" in names
    assert "palette_run_group_artifact/run_group/frame_indices/zarr.json" in names
