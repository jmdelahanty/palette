"""Canonical ONNX model manifests (v1 pose, v2 pose+detect) and detect promotion."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

import pytest

from fisheye.shared import canonical_onnx_model_manifest as manifest
from fisheye.shared import pose_model_input_contract as contract_module
from fisheye.shared.canonical_onnx_model_manifest import (
    CanonicalOnnxManifestError,
    build_canonical_onnx_manifest,
    validate_canonical_onnx_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.promote_registered_model import promote
from tests.unit.fisheye import test_promote_registered_pose_model as pose_tests

DETECT_RUN = "detect_run_v1"
DETECT_SET = "detect_set_v1"
HEX = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the manifest module ------------------------------------------------------


def _v1_pose_document() -> dict:
    payload = {
        "status": "complete", "run_id": "r", "set_id": "s", "task": "pose",
        "weights": {"relative_path": "weights/best.pt", "sha256": HEX},
        "onnx": {"relative_path": "exports/onnx/r.onnx", "sha256": HEX},
        "source_export_manifest": {"relative_path": "exports/onnx/r.onnx.manifest.json", "sha256": HEX},
        "pose_model_input_contract": {"relative_path": "pose_model_input_contract.json",
                                      "sha256": HEX, "payload_digest": HEX},
    }
    return {"schema_id": "palette.canonical_onnx_model_manifest", "schema_version": 1,
            "payload_digest": canonical_json_sha256(payload), "payload": payload}


def test_a_v1_pose_manifest_as_written_before_v2_still_validates():
    validate_canonical_onnx_manifest(_v1_pose_document())


def _detect_v2() -> dict:
    return build_canonical_onnx_manifest(
        task="detect", run_id="d", set_id="s",
        weights=("weights/best.pt", HEX), onnx=("exports/onnx/d.onnx", HEX),
        source_export_manifest=("exports/onnx/d.onnx.manifest.json", HEX),
    )


def test_v2_follows_digest_form_v1():
    document = _detect_v2()
    assert document["schema_version"] == 2
    assert document["canonicalization"] == "canonical_json_utf8_sort_keys_compact_v1"
    assert document["payload_sha256"].startswith("sha256:")
    assert {document["payload"][k]["sha256"] for k in ("weights", "onnx", "source_export_manifest")} == {
        "sha256:" + HEX
    }
    assert "pose_model_input_contract" not in document["payload"]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["payload"]["weights"].update(sha256=HEX), "does not match"),  # bare hex in v2
        (lambda d: d["payload"].update(run_id="other"), "payload digest mismatch"),
        (lambda d: d["payload"].update(task="segment"), "is not one of"),
        (lambda d: d.update(schema_version=3), "not a supported"),
        (lambda d: d["payload"]["onnx"].update(relative_path="/abs/d.onnx"), "does not match"),
        (lambda d: d["payload"]["onnx"].update(relative_path="../d.onnx"), "does not match"),
        (lambda d: d["payload"].update(pose_model_input_contract={
            "relative_path": "c.json", "sha256": "sha256:" + HEX, "payload_sha256": "sha256:" + HEX}),
         "should not be valid"),
    ],
)
def test_v2_refusals(edit, message):
    document = copy.deepcopy(_detect_v2())
    edit(document)
    with pytest.raises(CanonicalOnnxManifestError, match=message):
        validate_canonical_onnx_manifest(document)


def test_pose_and_detect_carry_the_contract_exactly_when_pose():
    with pytest.raises(CanonicalOnnxManifestError, match="pose model-input contract"):
        build_canonical_onnx_manifest(
            task="pose", run_id="p", set_id="s", weights=("w", HEX), onnx=("o", HEX),
            source_export_manifest=("m", HEX),
        )
    with pytest.raises(CanonicalOnnxManifestError, match="pose model-input contract"):
        build_canonical_onnx_manifest(
            task="detect", run_id="d", set_id="s", weights=("w", HEX), onnx=("o", HEX),
            source_export_manifest=("m", HEX), pose_model_input_contract=("c", HEX, HEX),
        )


@pytest.mark.parametrize("version", [1, 2])
def test_the_published_schemas_are_pinned(monkeypatch, version):
    manifest._validator.cache_clear()
    file_name, _digest = manifest.SCHEMA_FILES[version]
    monkeypatch.setitem(manifest.SCHEMA_FILES, version, (file_name, "0" * 64))
    try:
        with pytest.raises(CanonicalOnnxManifestError, match="packaged_contract_drift"):
            manifest._validator(version)
    finally:
        manifest._validator.cache_clear()


# --- pose promotion now writes v2 --------------------------------------------


def test_pose_promotion_writes_a_v2_manifest(tmp_path):
    package = pose_tests._source_package(tmp_path)
    registry, receipt = pose_tests._registry(tmp_path, package)
    with patch.object(contract_module.importlib.metadata, "version", return_value="8.3.214"):
        result = promote(
            pose_tests._args(registry=registry, backup_receipt=receipt,
                             destination_root=tmp_path / "models", apply=True),
            task="pose",
        )
    document = json.loads(Path(result["canonical_onnx_manifest_path"]).read_text())
    validate_canonical_onnx_manifest(document)
    assert (document["schema_version"], document["payload"]["task"]) == (2, "pose")
    assert document["payload"]["pose_model_input_contract"]["payload_sha256"].startswith("sha256:")


# --- detect promotion ---------------------------------------------------------


def _detect_package(root: Path) -> dict[str, Path]:
    package = root / "detect" / DETECT_SET / DETECT_RUN
    (package / "weights").mkdir(parents=True)
    (package / "exports" / "onnx").mkdir(parents=True)
    weights = package / "weights" / "best.pt"
    weights.write_bytes(b"detect weights")
    metrics = package / "results.csv"
    metrics.write_text("epoch,map\n1,0.5\n")
    onnx = package / "exports" / "onnx" / f"{DETECT_RUN}.onnx"
    onnx.write_bytes(b"detect onnx")
    export_manifest = package / "exports" / "onnx" / f"{DETECT_RUN}.onnx.manifest.json"
    export_manifest.write_text(json.dumps({
        "schema_version": 1, "run_id": DETECT_RUN,
        "weights": {"path": str(weights), "sha256": _sha(weights)},
        "onnx": {"path": str(onnx), "sha256": _sha(onnx)},
    }))
    return {"root": package, "weights": weights, "metrics": metrics, "onnx": onnx,
            "onnx_manifest": export_manifest}


def _detect_registry(tmp_path: Path, package: dict[str, Path]) -> tuple[Path, Path]:
    registry = tmp_path / "registry.sqlite"
    connection = sqlite3.connect(registry)
    connection.executescript(
        """
        CREATE TABLE training_runs (run_id TEXT PRIMARY KEY, set_id TEXT, task_type TEXT, status TEXT,
          model_path TEXT, model_sha256 TEXT, metrics_path TEXT, metrics_sha256 TEXT);
        CREATE TABLE training_models (run_id TEXT PRIMARY KEY, set_id TEXT, task_type TEXT, status TEXT,
          model_path TEXT, model_sha256 TEXT, metrics_path TEXT, metrics_sha256 TEXT, metadata_json TEXT);
        CREATE TABLE onnx_models (run_id TEXT PRIMARY KEY, set_id TEXT, path TEXT, sha256 TEXT,
          manifest_path TEXT, manifest_sha256 TEXT, metadata_json TEXT);
        """
    )
    common = (DETECT_RUN, DETECT_SET, "detect", "success", str(package["weights"]),
              _sha(package["weights"]), str(package["metrics"]), _sha(package["metrics"]))
    connection.execute("INSERT INTO training_runs VALUES (?,?,?,?,?,?,?,?)", common)
    connection.execute("INSERT INTO training_models VALUES (?,?,?,?,?,?,?,?,?)", common + ("{}",))
    connection.execute(
        "INSERT INTO onnx_models VALUES (?,?,?,?,?,?,?)",
        (DETECT_RUN, DETECT_SET, str(package["onnx"]), _sha(package["onnx"]),
         str(package["onnx_manifest"]), _sha(package["onnx_manifest"]), "{}"),
    )
    connection.commit()
    connection.close()
    backup = tmp_path / "registry.backup.sqlite"
    shutil.copy2(registry, backup)
    receipt = tmp_path / "registry.backup.json"
    receipt.write_text(json.dumps({
        "schema_id": "palette.registry_backup_receipt", "schema_version": 1, "status": "complete",
        "source_registry": str(registry.resolve()), "source_sha256": _sha(registry),
        "backup_path": str(backup.resolve()), "backup_sha256": _sha(backup),
    }))
    return registry, receipt


def _detect_args(registry: Path, receipt: Path, destination_root: Path, *, apply: bool):
    return argparse.Namespace(
        registry=registry, model_run_id=DETECT_RUN, destination_model_root=destination_root,
        registry_backup_receipt=receipt, dry_run=not apply, apply=apply,
    )


def test_detect_promotes_in_place_without_touching_existing_files(tmp_path):
    models = tmp_path / "models"
    package = _detect_package(models)
    before = {p.relative_to(package["root"]).as_posix(): _sha(p)
              for p in package["root"].rglob("*") if p.is_file()}
    registry, receipt = _detect_registry(tmp_path, package)

    planned = promote(_detect_args(registry, receipt, models, apply=False), task="detect")
    assert (planned["status"], planned["publication_mode"]) == ("planned", "in_place")

    result = promote(_detect_args(registry, receipt, models, apply=True), task="detect")
    canonical = package["root"] / "exports" / "onnx" / f"{DETECT_RUN}.canonical.manifest.json"
    assert result["canonical_onnx_manifest_path"] == str(canonical)
    document = validate_canonical_onnx_manifest(json.loads(canonical.read_text()))
    assert (document["schema_version"], document["payload"]["task"]) == (2, "detect")
    assert document["payload"]["onnx"] == {
        "relative_path": f"exports/onnx/{DETECT_RUN}.onnx", "sha256": "sha256:" + _sha(package["onnx"]),
    }
    after = {p.relative_to(package["root"]).as_posix(): _sha(p)
             for p in package["root"].rglob("*") if p.is_file()}
    assert {k: after[k] for k in before} == before
    assert set(after) - set(before) == {
        f"exports/onnx/{DETECT_RUN}.canonical.manifest.json", "model_package_publication.json",
    }

    connection = sqlite3.connect(registry)
    connection.row_factory = sqlite3.Row
    onnx = connection.execute("SELECT * FROM onnx_models WHERE run_id=?", (DETECT_RUN,)).fetchone()
    training = connection.execute("SELECT * FROM training_runs WHERE run_id=?", (DETECT_RUN,)).fetchone()
    connection.close()
    assert (onnx["manifest_path"], onnx["manifest_sha256"]) == (str(canonical), _sha(canonical))
    assert onnx["path"] == str(package["onnx"]) and training["model_path"] == str(package["weights"])
    promotion = json.loads(onnx["metadata_json"])["canonical_package_promotion"]
    assert promotion["schema_id"] == "palette.registered_detect_model_package_publication"
    assert "pose_model_input_contract_path" not in promotion

    # A second promotion never overwrites the evidence.
    with pytest.raises(FileExistsError, match="already carries a canonical manifest"):
        promote(_detect_args(registry, receipt, models, apply=False), task="detect")


def test_detect_refuses_a_pose_run_and_a_changed_artifact(tmp_path):
    models = tmp_path / "models"
    package = _detect_package(models)
    registry, receipt = _detect_registry(tmp_path, package)
    with pytest.raises(ValueError, match="not a pose model"):
        promote(_detect_args(registry, receipt, models, apply=False), task="pose")
    package["onnx"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="onnx SHA-256 differs"):
        promote(_detect_args(registry, receipt, models, apply=False), task="detect")


def test_detect_copy_mode_when_the_package_is_elsewhere(tmp_path):
    package = _detect_package(tmp_path / "training")
    registry, receipt = _detect_registry(tmp_path, package)
    models = tmp_path / "models"
    result = promote(_detect_args(registry, receipt, models, apply=True), task="detect")
    destination = models / "detect" / DETECT_SET / DETECT_RUN
    assert result["publication_mode"] == "copy"
    assert (destination / "weights" / "best.pt").read_bytes() == b"detect weights"
    validate_canonical_onnx_manifest(json.loads(
        (destination / "exports" / "onnx" / f"{DETECT_RUN}.canonical.manifest.json").read_text()
    ))
    assert package["weights"].is_file()  # the source package is left untouched
