from __future__ import annotations

import json
import tarfile
from pathlib import Path

from fisheye.utils import import_run_group_artifact as mod
from fisheye.utils.run_detection_artifact import ARTIFACT_SCHEMA, REQUIRED_DETECT_ARRAYS, tree_hash


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


def _write_artifact(
    tmp_path: Path,
    *,
    target_zarr: Path,
    corrupt_hash: bool = False,
    latest_policy: str = "do_not_set_latest",
    intended_target_group_path: str | None = None,
) -> Path:
    source_video = tmp_path / "camera.mp4"
    source_video.write_bytes(b"fake")
    artifact_root = tmp_path / "palette_run_group_artifact"
    run_group = artifact_root / "run_group"
    _write_group(run_group)
    for name in REQUIRED_DETECT_ARRAYS:
        _write_array(run_group / name)

    digest = tree_hash(run_group)
    manifest = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "created_at": "2026-05-15T00:00:00+00:00",
        "target_archive_path": str(target_zarr),
        "target_group_path": "detect_runs/detect_fake",
        "run_family": "detect_runs",
        "run_name": "detect_fake",
        "layout": "detect_yolo_sparse_v1",
        "schema_version": 1,
        "latest_policy": latest_policy,
        "source_inputs": [
            {"path": str(source_video), "role": "source_video"},
            {"path": str(target_zarr), "role": "target_analysis_archive"},
        ],
        "provenance": {"command": "scripts/py -m fake"},
        "timing": {},
        "checksums": {"run_group_tree_hash": "bad" if corrupt_hash else digest},
        "validation": {
            "strict_json": "pass",
            "required_arrays": "pass",
            "canonical_write": "not_performed",
        },
    }
    if intended_target_group_path is not None:
        manifest["intended_target_group_path"] = intended_target_group_path
    (artifact_root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifact_root / "validation").mkdir()
    (artifact_root / "validation" / "strict_json_report.json").write_text("{}", encoding="utf-8")
    (artifact_root / "validation" / "array_presence_report.json").write_text("{}", encoding="utf-8")

    tarball = tmp_path / "artifact.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(artifact_root, arcname=artifact_root.name)
    return tarball


def test_build_import_plan_validates_artifact_without_mutating_target(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "ok"
    assert plan["apply"] is False
    assert plan["target_group_path"] == "detect_runs/detect_fake"
    assert plan["final_path"] == str(target_zarr / "detect_runs" / "detect_fake")
    assert plan["incoming_path"] == str(target_zarr / "detect_runs" / ".incoming" / "detect_fake")
    assert plan["validations"]["strict_json"]["status"] == "pass"
    assert plan["validations"]["required_arrays"]["status"] == "pass"
    assert plan["validations"]["run_group_tree_hash"]["status"] == "pass"
    assert not (target_zarr / "detect_runs").exists()


def test_build_import_plan_can_target_clip_local_intended_path(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(
        tmp_path,
        target_zarr=target_zarr,
        intended_target_group_path="clips/clip_000000/cameras/2010093/detect_runs/detect_fake",
    )

    plan = mod.build_import_plan(tarball_path=tarball, use_intended_target=True)

    assert plan["status"] == "ok"
    assert plan["target_group_path"] == "clips/clip_000000/cameras/2010093/detect_runs/detect_fake"
    assert plan["target_group_path_source"] == "intended_target_group_path"
    assert plan["run_family_path"] == "clips/clip_000000/cameras/2010093/detect_runs"
    assert plan["final_path"] == str(
        target_zarr / "clips" / "clip_000000" / "cameras" / "2010093" / "detect_runs" / "detect_fake"
    )
    assert plan["incoming_path"] == str(
        target_zarr
        / "clips"
        / "clip_000000"
        / "cameras"
        / "2010093"
        / "detect_runs"
        / ".incoming"
        / "detect_fake"
    )


def test_build_import_plan_fails_when_final_target_exists(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    _write_group(target_zarr / "detect_runs" / "detect_fake")
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "failed"
    assert plan["validations"]["target_paths"]["status"] == "fail"
    assert "final target already exists" in "\n".join(plan["errors"])


def test_build_import_plan_fails_on_hash_mismatch(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr, corrupt_hash=True)

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "failed"
    assert plan["validations"]["run_group_tree_hash"]["status"] == "fail"
    assert "run_group_tree_hash mismatch" in "\n".join(plan["errors"])


def test_build_import_plan_rejects_unsafe_tar_member(tmp_path: Path) -> None:
    tarball = tmp_path / "unsafe.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("bad", encoding="utf-8")
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(payload, arcname="../escape.txt")

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "failed"
    assert "unsafe tar member path" in "\n".join(plan["errors"])


def test_apply_import_promotes_run_group_and_writes_receipt_sidecar(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    result = mod.apply_import(tarball_path=tarball)

    final_path = target_zarr / "detect_runs" / "detect_fake"
    incoming_path = target_zarr / "detect_runs" / ".incoming" / "detect_fake"
    receipt_path = target_zarr / "detect_runs" / ".imports" / "detect_fake_import_receipt.json"
    assert result["status"] == "ok"
    assert result["apply"] is True
    assert result["applied"] is True
    assert final_path.exists()
    assert not incoming_path.exists()
    assert (final_path / "frame_indices" / "zarr.json").exists()
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["run_name"] == "detect_fake"
    assert receipt["latest_updated"] is False
    parent_attrs = json.loads((target_zarr / "detect_runs" / "zarr.json").read_text(encoding="utf-8"))[
        "attributes"
    ]
    assert "latest" not in parent_attrs


def test_apply_import_can_promote_to_clip_local_intended_path(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    intended = "clips/clip_000000/cameras/2010093/detect_runs/detect_fake"
    tarball = _write_artifact(
        tmp_path,
        target_zarr=target_zarr,
        intended_target_group_path=intended,
    )

    result = mod.apply_import(tarball_path=tarball, use_intended_target=True)

    family = target_zarr / "clips" / "clip_000000" / "cameras" / "2010093" / "detect_runs"
    final_path = family / "detect_fake"
    receipt_path = family / ".imports" / "detect_fake_import_receipt.json"
    assert result["status"] == "ok"
    assert result["applied"] is True
    assert result["target_group_path"] == intended
    assert result["run_family_path"] == "clips/clip_000000/cameras/2010093/detect_runs"
    assert final_path.exists()
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["target_group_path"] == intended
    assert receipt["target_group_path_source"] == "intended_target_group_path"
    assert receipt["run_family"] == "detect_runs"
    assert receipt["run_family_path"] == "clips/clip_000000/cameras/2010093/detect_runs"


def test_apply_import_updates_latest_when_requested(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(
        tmp_path,
        target_zarr=target_zarr,
        latest_policy="set_latest_explicit",
    )

    result = mod.apply_import(tarball_path=tarball)

    parent_attrs = json.loads((target_zarr / "detect_runs" / "zarr.json").read_text(encoding="utf-8"))[
        "attributes"
    ]
    assert result["status"] == "ok"
    assert result["latest_updated"] is True
    assert parent_attrs["latest"] == "detect_fake"


def test_apply_import_moves_incoming_to_failed_on_apply_validation_failure(
    tmp_path: Path, monkeypatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_required_arrays_report = mod.required_arrays_report
    call_count = {"n": 0}

    def fail_second_required_arrays_report(path: Path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return {
                "status": "fail",
                "run_group_zarr_json_present": True,
                "arrays": [],
                "missing_arrays": ["frame_indices"],
            }
        return original_required_arrays_report(path)

    monkeypatch.setattr(mod, "required_arrays_report", fail_second_required_arrays_report)

    result = mod.apply_import(tarball_path=tarball)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["failed_path"] is not None
    assert Path(result["failed_path"]).exists()
    assert not (target_zarr / "detect_runs" / ".incoming" / "detect_fake").exists()
    assert not (target_zarr / "detect_runs" / "detect_fake").exists()
