#!/usr/bin/env python3
"""Add a v2 canonical ONNX manifest beside a promoted package's v1 manifest.

Packages promoted before 2026-10-12 carry a v1
``exports/onnx/<run_id>.canonical.manifest.json`` (bare-hex digests). Orange
retires v1 acceptance, so such a package needs a v2 manifest
(``fisheye.shared.canonical_onnx_model_manifest``) for the same artifacts.

- The registry's ONNX manifest pointer must name that v1 file with its SHA-256,
  and the v1 manifest must validate.
- Every artifact the v1 manifest binds is re-hashed in the package; the v2
  manifest binds the same relative paths and digests (as ``sha256:`` tokens).
- The v2 manifest is written to ``exports/onnx/<run_id>.canonical.v2.manifest.json``.
  No existing file is rewritten (the v1 bytes stay what engines already bind),
  and the package inventory afterwards must be the original plus that file.
- The registry pointer moves to the v2 manifest in one SQLite transaction
  guarded by a registry backup receipt; ``onnx_models.metadata_json`` records
  the upgrade under ``canonical_manifest_upgrade``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from fisheye.shared.canonical_onnx_model_manifest import (
    build_canonical_onnx_manifest,
    validate_canonical_onnx_manifest,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.utils.promote_registered_model import (
    _backup_identity,
    _file_digests,
    _read_json,
    _registry_rows,
    _require_regular_tree,
    _safe_relative,
    _sha256_file,
)

UPGRADE_SCHEMA_ID = "palette.canonical_onnx_manifest_upgrade"
UPGRADE_SCHEMA_VERSION = 1
ARTIFACT_ROLES = ("weights", "onnx", "source_export_manifest", "pose_model_input_contract")


def v2_manifest_relative(run_id: str) -> Path:
    return Path("exports") / "onnx" / f"{run_id}.canonical.v2.manifest.json"


def _plan(*, registry: Path, run_id: str) -> dict[str, Any]:
    rows = _registry_rows(registry, run_id)
    onnx_row = rows["onnx_models"]
    v1_path = Path(str(onnx_row.get("manifest_path") or "")).expanduser()
    if v1_path.is_symlink() or not v1_path.is_file():
        raise ValueError(f"Registered canonical manifest is missing or symlinked: {v1_path}")
    v1_path = v1_path.resolve()
    if _sha256_file(v1_path) != onnx_row.get("manifest_sha256"):
        raise ValueError("Registered canonical manifest SHA-256 differs from its file")
    v1 = validate_canonical_onnx_manifest(_read_json(v1_path, field="canonical manifest"))
    if v1["schema_version"] != 1:
        raise ValueError(
            f"Registered canonical manifest is already v{v1['schema_version']}; nothing to upgrade"
        )
    payload = v1["payload"]
    training_run = rows["training_runs"]
    if (
        payload["run_id"] != run_id
        or payload["set_id"] != training_run.get("set_id")
        or payload["task"] != training_run.get("task_type")
        or payload["weights"]["sha256"] != training_run.get("model_sha256")
        or payload["onnx"]["sha256"] != onnx_row.get("sha256")
    ):
        raise ValueError("v1 canonical manifest disagrees with the registered run")
    if v1_path.parent.name != "onnx" or v1_path.parent.parent.name != "exports":
        raise ValueError("v1 canonical manifest is not under the package's exports/onnx")
    package_root = v1_path.parent.parent.parent
    if package_root.name != run_id:
        raise ValueError("Model package directory differs from run_id")
    _require_regular_tree(package_root)
    for role in ARTIFACT_ROLES:
        if role not in payload:
            continue
        relative = Path(payload[role]["relative_path"])
        path = package_root / relative
        if _safe_relative(package_root, path, field=role) != relative:
            raise ValueError(f"v1 {role} path is not package-relative")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"v1 {role} artifact is missing or symlinked: {path}")
        if _sha256_file(path) != payload[role]["sha256"]:
            raise ValueError(f"v1 {role} SHA-256 differs from its file")
    contract = payload.get("pose_model_input_contract")
    if contract is not None:
        document = _read_json(package_root / contract["relative_path"], field="pose model-input contract")
        if document.get("payload_digest") != contract["payload_digest"]:
            raise ValueError("Pose model-input contract payload digest differs from v1")
    v2 = build_canonical_onnx_manifest(
        task=payload["task"],
        run_id=run_id,
        set_id=payload["set_id"],
        weights=(payload["weights"]["relative_path"], payload["weights"]["sha256"]),
        onnx=(payload["onnx"]["relative_path"], payload["onnx"]["sha256"]),
        source_export_manifest=(
            payload["source_export_manifest"]["relative_path"],
            payload["source_export_manifest"]["sha256"],
        ),
        pose_model_input_contract=(
            None
            if contract is None
            else (contract["relative_path"], contract["sha256"], contract["payload_digest"])
        ),
    )
    v2_path = package_root / v2_manifest_relative(run_id)
    if v2_path.exists():
        raise FileExistsError(f"Package already carries a v2 canonical manifest: {v2_path}")
    return {
        "run_id": run_id,
        "onnx_sha256": onnx_row["sha256"],
        "package_root": package_root,
        "v1_path": v1_path,
        "v1_sha256": onnx_row["manifest_sha256"],
        "v2_path": v2_path,
        "v2_document": v2,
    }


def _update_registry(
    *, registry: Path, plan: Mapping[str, Any], v2_sha256: str, backup_receipt: Path,
    backed_up_registry_sha256: str,
) -> dict[str, Any]:
    upgrade = {
        "schema_id": UPGRADE_SCHEMA_ID,
        "schema_version": UPGRADE_SCHEMA_VERSION,
        "upgraded_utc": datetime.now(timezone.utc).isoformat(),
        "from_manifest_path": str(plan["v1_path"]),
        "from_manifest_sha256": plan["v1_sha256"],
        "from_schema_version": 1,
        "to_manifest_path": str(plan["v2_path"]),
        "to_manifest_sha256": v2_sha256,
        "to_schema_version": plan["v2_document"]["schema_version"],
        "registry_backup_receipt_path": str(backup_receipt),
        "registry_backup_receipt_sha256": _sha256_file(backup_receipt),
        "selector_activation": False,
    }
    connection = sqlite3.connect(str(registry))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.execute("BEGIN IMMEDIATE")
        if _sha256_file(registry) != backed_up_registry_sha256:
            raise ValueError("Registry changed after backup and before the upgrade")
        row = connection.execute(
            "SELECT metadata_json FROM onnx_models WHERE run_id = ?", (plan["run_id"],)
        ).fetchone()
        if row is None:
            raise ValueError("Registry row disappeared from onnx_models")
        metadata = json.loads(row["metadata_json"] or "{}")
        if not isinstance(metadata, dict):
            raise ValueError("onnx_models.metadata_json is not a mapping")
        metadata["canonical_manifest_upgrade"] = upgrade
        cursor = connection.execute(
            "UPDATE onnx_models SET manifest_path=?, manifest_sha256=?, metadata_json=? "
            "WHERE run_id=? AND sha256=? AND manifest_path=? AND manifest_sha256=?",
            (
                str(plan["v2_path"]), v2_sha256, json.dumps(metadata, sort_keys=True),
                plan["run_id"], plan["onnx_sha256"], str(plan["v1_path"]), plan["v1_sha256"],
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("Registry update preconditions failed for onnx_models")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return upgrade


def upgrade(
    *, registry: Path, run_id: str, apply: bool, registry_backup_receipt: Path | None = None
) -> dict[str, Any]:
    registry = registry.expanduser().resolve()
    plan = _plan(registry=registry, run_id=run_id)
    summary = {
        "status": "planned",
        "run_id": run_id,
        "package_root": str(plan["package_root"]),
        "from_manifest_path": str(plan["v1_path"]),
        "from_manifest_sha256": plan["v1_sha256"],
        "to_manifest_path": str(plan["v2_path"]),
        "to_document": plan["v2_document"],
    }
    if not apply:
        return summary
    if registry_backup_receipt is None:
        raise ValueError("--registry-backup-receipt is required with --apply")
    backup_receipt = registry_backup_receipt.expanduser().resolve()
    backup_identity = _backup_identity(registry, backup_receipt)
    package_root = Path(plan["package_root"])
    before = _file_digests(package_root)
    write_json_atomic(plan["v2_path"], plan["v2_document"])
    after = _file_digests(package_root)
    added = v2_manifest_relative(run_id).as_posix()
    if {name: after.get(name) for name in before} != before or set(after) - set(before) != {added}:
        raise RuntimeError("Upgrade changed files other than the new v2 manifest")
    v2_sha256 = after[added][1]
    validate_canonical_onnx_manifest(_read_json(plan["v2_path"], field="v2 canonical manifest"))
    record = _update_registry(
        registry=registry, plan=plan, v2_sha256=v2_sha256, backup_receipt=backup_receipt,
        backed_up_registry_sha256=str(backup_identity["source_sha256"]),
    )
    return {**summary, "status": "complete", "to_manifest_sha256": v2_sha256, "registry_record": record}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--registry-backup-receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    result = upgrade(
        registry=args.registry, run_id=args.model_run_id, apply=args.apply,
        registry_backup_receipt=args.registry_backup_receipt,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
