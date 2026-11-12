"""v1 -> v2 canonical ONNX manifest upgrade for packages promoted before v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

import pytest

from fisheye.shared import pose_model_input_contract as contract_module
from fisheye.shared.canonical_onnx_model_manifest import validate_canonical_onnx_manifest
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.promote_registered_model import promote
from fisheye.utils.upgrade_canonical_onnx_manifest import upgrade, v2_manifest_relative
from tests.unit.fisheye import test_promote_registered_pose_model as pose_tests


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bare(token: str) -> str:
    return token.removeprefix("sha256:")


def _backup(tmp_path: Path, registry: Path, name: str) -> Path:
    backup = tmp_path / f"{name}.sqlite"
    shutil.copy2(registry, backup)
    receipt = tmp_path / f"{name}.json"
    receipt.write_text(json.dumps({
        "schema_id": "palette.registry_backup_receipt", "schema_version": 1, "status": "complete",
        "source_registry": str(registry.resolve()), "source_sha256": _sha(registry),
        "backup_path": str(backup.resolve()), "backup_sha256": _sha(backup),
    }))
    return receipt


def _v1_promoted(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A pose package as promoted before 2026-10-12: its registered canonical
    manifest is v1 (bare hex, ``payload_digest``)."""

    package = pose_tests._source_package(tmp_path)
    registry, receipt = pose_tests._registry(tmp_path, package)
    with patch.object(contract_module.importlib.metadata, "version", return_value="8.3.214"):
        result = promote(
            pose_tests._args(registry=registry, backup_receipt=receipt,
                             destination_root=tmp_path / "models", apply=True),
            task="pose",
        )
    canonical = Path(result["canonical_onnx_manifest_path"])
    v2_payload = json.loads(canonical.read_text())["payload"]
    payload = {
        key: (
            {**value, "sha256": _bare(value["sha256"])} if isinstance(value, dict) else value
        )
        for key, value in v2_payload.items()
    }
    contract = payload["pose_model_input_contract"]
    contract["payload_digest"] = _bare(contract.pop("payload_sha256"))
    v1 = {"schema_id": "palette.canonical_onnx_model_manifest", "schema_version": 1,
          "payload_digest": canonical_json_sha256(payload), "payload": payload}
    validate_canonical_onnx_manifest(v1)
    canonical.write_text(json.dumps(v1, indent=2, sort_keys=True))
    connection = sqlite3.connect(registry)
    connection.execute("UPDATE onnx_models SET manifest_sha256=?", (_sha(canonical),))
    connection.commit()
    connection.close()
    return registry, canonical, Path(result["destination_package_root"])


def _files(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*") if p.is_file()}


def test_upgrade_adds_v2_beside_v1_and_moves_the_pointer(tmp_path):
    registry, v1_path, root = _v1_promoted(tmp_path)
    v1_bytes = v1_path.read_bytes()
    v1 = json.loads(v1_bytes)
    before = _files(root)

    planned = upgrade(registry=registry, run_id=pose_tests.RUN_ID, apply=False)
    assert planned["status"] == "planned" and _files(root) == before

    receipt = _backup(tmp_path, registry, "before_upgrade")
    result = upgrade(registry=registry, run_id=pose_tests.RUN_ID, apply=True,
                     registry_backup_receipt=receipt)
    v2_path = root / v2_manifest_relative(pose_tests.RUN_ID)
    assert result["to_manifest_path"] == str(v2_path)
    assert v1_path.read_bytes() == v1_bytes
    after = _files(root)
    assert {k: after[k] for k in before} == before
    assert set(after) - set(before) == {v2_manifest_relative(pose_tests.RUN_ID).as_posix()}

    v2 = validate_canonical_onnx_manifest(json.loads(v2_path.read_text()))
    assert v2["schema_version"] == 2
    for role in ("weights", "onnx", "source_export_manifest"):
        assert v2["payload"][role] == {
            "relative_path": v1["payload"][role]["relative_path"],
            "sha256": "sha256:" + v1["payload"][role]["sha256"],
        }
    contract = v1["payload"]["pose_model_input_contract"]
    assert v2["payload"]["pose_model_input_contract"] == {
        "relative_path": contract["relative_path"],
        "sha256": "sha256:" + contract["sha256"],
        "payload_sha256": "sha256:" + contract["payload_digest"],
    }

    connection = sqlite3.connect(registry)
    connection.row_factory = sqlite3.Row
    row = connection.execute("SELECT * FROM onnx_models").fetchone()
    connection.close()
    assert (row["manifest_path"], row["manifest_sha256"]) == (str(v2_path), _sha(v2_path))
    metadata = json.loads(row["metadata_json"])
    record = metadata["canonical_manifest_upgrade"]
    assert (record["from_manifest_path"], record["from_manifest_sha256"]) == (
        str(v1_path), _sha(v1_path))
    assert record["to_manifest_sha256"] == _sha(v2_path)
    assert "canonical_package_promotion" in metadata  # the promotion record is kept

    with pytest.raises(ValueError, match="already v2"):
        upgrade(registry=registry, run_id=pose_tests.RUN_ID, apply=False)


def test_upgrade_refuses_a_changed_artifact(tmp_path):
    registry, v1_path, root = _v1_promoted(tmp_path)
    onnx = root / json.loads(v1_path.read_text())["payload"]["onnx"]["relative_path"]
    onnx.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        upgrade(registry=registry, run_id=pose_tests.RUN_ID, apply=False)


def test_upgrade_refuses_a_registry_pointer_that_does_not_match(tmp_path):
    registry, _v1_path, _root = _v1_promoted(tmp_path)
    connection = sqlite3.connect(registry)
    connection.execute("UPDATE onnx_models SET manifest_sha256=?", ("0" * 64,))
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="Registered canonical manifest SHA-256 differs"):
        upgrade(registry=registry, run_id=pose_tests.RUN_ID, apply=False)


def test_upgrade_refuses_an_existing_v2_file_and_a_stale_backup(tmp_path):
    registry, _v1_path, root = _v1_promoted(tmp_path)
    receipt = _backup(tmp_path, registry, "stale")
    connection = sqlite3.connect(registry)
    connection.execute("UPDATE onnx_models SET metadata_json=metadata_json")
    connection.execute("CREATE TABLE touched (x)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="Registry changed after the supplied backup"):
        upgrade(registry=registry, run_id=pose_tests.RUN_ID, apply=True,
                registry_backup_receipt=receipt)
    assert not (root / v2_manifest_relative(pose_tests.RUN_ID)).exists()
    (root / v2_manifest_relative(pose_tests.RUN_ID)).write_text("{}")
    with pytest.raises(FileExistsError, match="already carries a v2"):
        upgrade(registry=registry, run_id=pose_tests.RUN_ID, apply=False)
