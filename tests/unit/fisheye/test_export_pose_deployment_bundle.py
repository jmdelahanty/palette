from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from fisheye.shared.pose_deployment_manifest import load_pose_deployment_bundle
from fisheye.utils.export_pose_deployment_bundle import export_bundle, main
from tests.unit.fisheye.pose_deployment_helpers import RUN_ID, _sha, source_bundle


def _inventory(root):
    return {
        str(path.relative_to(root)): _sha(path)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("count", [3, 19])
def test_real_producer_bundle_reader_roundtrip_is_relocatable_and_preserves_sources(
    tmp_path, count
):
    registry, source = source_bundle(tmp_path, count=count)
    source_before, registry_before = _inventory(source["root"]), _sha(registry)
    destination = tmp_path / "deployment_v1"
    result = export_bundle(
        registry=registry, run_id=RUN_ID, destination=destination, apply=True
    )
    manifest = destination / f"{RUN_ID}.canonical.manifest.json"
    assert result["selector_activation"] is False
    assert _sha(registry) == registry_before
    assert _inventory(source["root"]) == source_before
    assert _sha(destination / source["onnx"].name) == _sha(source["onnx"])
    assert json.loads(source["canonical"].read_text())["schema_version"] == 1
    assert json.loads(manifest.read_text())["schema_version"] == 2
    assert not (destination / "weights").exists()

    relocated = tmp_path / "remote" / "model_bundle"
    shutil.copytree(destination, relocated)
    validated = load_pose_deployment_bundle(
        relocated / manifest.name, expected_manifest_sha256=_sha(manifest)
    )
    assert validated["skeleton"]["kpt_shape"] == [count, 3]
    assert validated["skeleton"]["nodes"][0] == {"id": 0, "name": "swim_bladder"}
    assert validated["manifest"]["payload"]["pose_model_skeleton"]["sha256"] == _sha(
        relocated / "pose_model_skeleton.json"
    )
    assert validated["manifest"]["payload"]["onnx_interface"]["outputs"][0][
        "shape"
    ] == [1, 5 + count * 3, 756]


def test_dry_run_has_no_filesystem_or_registry_writes(tmp_path):
    registry, source = source_bundle(tmp_path)
    before = _inventory(tmp_path)
    result = export_bundle(
        registry=registry, run_id=RUN_ID, destination=tmp_path / "new", apply=False
    )
    assert result["status"] == "planned"
    assert _inventory(tmp_path) == before


@pytest.mark.parametrize(
    "role", ["onnx", "canonical", "training_manifest", "input_contract", "weights"]
)
def test_source_tampering_refuses_before_output_creation(tmp_path, role):
    registry, source = source_bundle(tmp_path)
    source[role].write_bytes(source[role].read_bytes() + b" ")
    destination = tmp_path / "new"
    with pytest.raises(ValueError):
        export_bundle(
            registry=registry, run_id=RUN_ID, destination=destination, apply=True
        )
    assert not destination.exists()


def test_never_overwrites_existing_destination_and_retry_uses_fresh_revision(tmp_path):
    registry, source = source_bundle(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_bytes(b"owned by another workflow")
    before = _inventory(destination)
    with pytest.raises(FileExistsError):
        export_bundle(
            registry=registry, run_id=RUN_ID, destination=destination, apply=True
        )
    assert _inventory(destination) == before
    export_bundle(
        registry=registry, run_id=RUN_ID, destination=tmp_path / "revision2", apply=True
    )


@pytest.mark.parametrize(
    "name",
    ["pose_model_skeleton.json", "pose_model_input_contract.json", f"{RUN_ID}.onnx"],
)
def test_reader_rejects_changed_or_missing_runtime_artifact(tmp_path, name):
    registry, _ = source_bundle(tmp_path)
    destination = tmp_path / "bundle"
    export_bundle(registry=registry, run_id=RUN_ID, destination=destination, apply=True)
    manifest = destination / f"{RUN_ID}.canonical.manifest.json"
    expected = _sha(manifest)
    path = destination / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        load_pose_deployment_bundle(manifest, expected_manifest_sha256=expected)
    path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_pose_deployment_bundle(manifest, expected_manifest_sha256=expected)


def test_copy_failure_cannot_publish_a_canonical_manifest(tmp_path, monkeypatch):
    registry, _ = source_bundle(tmp_path)
    destination = tmp_path / "failed"

    def fail_copy(*args, **kwargs):
        raise OSError("injected copy failure")

    monkeypatch.setattr(
        "fisheye.utils.export_pose_deployment_bundle.shutil.copy2", fail_copy
    )
    with pytest.raises(OSError, match="injected"):
        export_bundle(
            registry=registry, run_id=RUN_ID, destination=destination, apply=True
        )
    assert not (destination / f"{RUN_ID}.canonical.manifest.json").exists()


def test_cli_exports_and_validates_real_bundle(tmp_path, capsys):
    registry, _ = source_bundle(tmp_path)
    destination = tmp_path / "cli"
    assert (
        main(
            [
                "export",
                "--registry",
                str(registry),
                "--model-run-id",
                RUN_ID,
                "--destination",
                str(destination),
                "--apply",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = destination / f"{RUN_ID}.canonical.manifest.json"
    assert (
        main(["validate", str(manifest), "--expected-manifest-sha256", _sha(manifest)])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "valid"


def test_source_mutation_during_copy_cannot_publish(tmp_path, monkeypatch):
    registry, source = source_bundle(tmp_path)
    destination = tmp_path / "changed"
    original_copy = shutil.copy2

    def change_source(src, dst):
        result = original_copy(src, dst)
        if src == source["onnx"]:
            src.write_bytes(src.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(
        "fisheye.utils.export_pose_deployment_bundle.shutil.copy2", change_source
    )
    with pytest.raises(ValueError, match="SHA-256"):
        export_bundle(
            registry=registry, run_id=RUN_ID, destination=destination, apply=True
        )
    assert not (destination / f"{RUN_ID}.canonical.manifest.json").exists()


def test_directory_ownership_loss_stops_writes_without_touching_replacement(
    tmp_path, monkeypatch
):
    registry, _ = source_bundle(tmp_path)
    destination = tmp_path / "replaced"
    original_copy = shutil.copy2

    def replace_directory(src, dst):
        result = original_copy(src, dst)
        destination.rename(tmp_path / "displaced")
        destination.mkdir()
        (destination / "keep").write_bytes(b"another owner")
        return result

    monkeypatch.setattr(
        "fisheye.utils.export_pose_deployment_bundle.shutil.copy2", replace_directory
    )
    with pytest.raises(RuntimeError, match="ownership changed"):
        export_bundle(
            registry=registry, run_id=RUN_ID, destination=destination, apply=True
        )
    assert list(destination.iterdir()) == [destination / "keep"]
    assert (destination / "keep").read_bytes() == b"another owner"
    assert not (tmp_path / "displaced" / f"{RUN_ID}.canonical.manifest.json").exists()


def test_publication_failure_leaves_incomplete_revision_and_supports_fresh_retry(
    tmp_path, monkeypatch
):
    import os

    registry, _ = source_bundle(tmp_path)
    destination = tmp_path / "failed_publication"
    real_link = os.link

    def fail_link(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(
        "fisheye.utils.export_pose_deployment_bundle.os.link", fail_link
    )
    with pytest.raises(OSError, match="publication failure"):
        export_bundle(
            registry=registry, run_id=RUN_ID, destination=destination, apply=True
        )
    assert not (destination / f"{RUN_ID}.canonical.manifest.json").exists()
    monkeypatch.setattr(
        "fisheye.utils.export_pose_deployment_bundle.os.link", real_link
    )
    result = export_bundle(
        registry=registry, run_id=RUN_ID, destination=tmp_path / "retry", apply=True
    )
    assert result["status"] == "complete"


@pytest.mark.parametrize(
    "problem", ["failed_run", "wrong_set", "stale_manifest", "wrong_labels"]
)
def test_registry_conflicts_fail_before_creating_output(tmp_path, problem):
    import sqlite3

    registry, _ = source_bundle(tmp_path)
    statements = {
        "failed_run": "UPDATE training_runs SET status='failed'",
        "wrong_set": "UPDATE onnx_models SET set_id='wrong'",
        "stale_manifest": "UPDATE onnx_models SET manifest_sha256='" + "0" * 64 + "'",
        "wrong_labels": 'UPDATE pose_skeleton_specs SET keypoint_labels_json=\'["wrong","eye_left","eye_right"]\'',
    }
    with sqlite3.connect(registry) as connection:
        connection.execute(statements[problem])
    destination = tmp_path / "refused"
    with pytest.raises(ValueError):
        export_bundle(
            registry=registry, run_id=RUN_ID, destination=destination, apply=True
        )
    assert not destination.exists()
