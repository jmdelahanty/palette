#!/usr/bin/env python3
"""Publish one successful registered pose model into durable shared storage.

The source training package remains unchanged. Publication copies it through a
hidden sibling, verifies every source file by relative path, size, and SHA-256,
adds relocation-safe deployment evidence, atomically renames the package, and
then updates only the registered artifact paths in one SQLite transaction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Mapping, Sequence
import uuid

from fisheye.shared.atomic_run_publisher import tree_inventory
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.pose_deployment_manifest import CANONICAL_ONNX_MANIFEST_SCHEMA_ID
from fisheye.shared.pose_model_input_contract import (
    POSE_MODEL_INPUT_CONTRACT_FILENAME,
    build_runtime_receipt_pose_model_input_contract,
    load_pose_model_input_contract,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


PUBLICATION_SCHEMA_ID = "palette.registered_pose_model_package_publication"
PUBLICATION_SCHEMA_VERSION = 1
# Historical promotion remains v1; the explicit deployment packaging command
# owns v2. Never restamp a published model package to add deployment metadata.
CANONICAL_ONNX_MANIFEST_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {field} {path}: {exc}") from exc
    if type(value) is not dict:
        raise ValueError(f"{field} must be one JSON mapping")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json_atomic(path, dict(value))


def _safe_relative(root: Path, path: Path, *, field: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} is outside the model package: {resolved}") from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"{field} has an unsafe relative path")
    return relative


def _require_regular_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Model package is missing or symlinked: {root}")
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Model package contains symlinks: {symlinks[0]}")


def _registry_rows(registry: Path, run_id: str) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows: dict[str, dict[str, Any]] = {}
        for table in ("training_runs", "training_models", "onnx_models"):
            found = connection.execute(
                f"SELECT * FROM {table} WHERE run_id = ?", (run_id,)
            ).fetchall()
            if len(found) != 1:
                raise ValueError(f"Expected one {table} row for {run_id!r}")
            rows[table] = dict(found[0])
        return rows
    finally:
        connection.close()


def _validate_registry_source(
    *, registry: Path, run_id: str, destination_model_root: Path
) -> dict[str, Any]:
    rows = _registry_rows(registry, run_id)
    training_run = rows["training_runs"]
    training_model = rows["training_models"]
    onnx_model = rows["onnx_models"]
    if training_run.get("task_type") != "pose" or training_model.get("task_type") != "pose":
        raise ValueError("Registered model is not a pose model")
    if training_run.get("status") != "success" or training_model.get("status") != "success":
        raise ValueError("Registered pose model is not successful")
    set_id = str(training_run.get("set_id") or "")
    if not set_id or training_model.get("set_id") != set_id or onnx_model.get("set_id") != set_id:
        raise ValueError("Registry set identity is incomplete or inconsistent")
    source_model = Path(str(training_run.get("model_path") or "")).expanduser().resolve()
    if Path(str(training_model.get("model_path") or "")).expanduser().resolve() != source_model:
        raise ValueError("training_runs and training_models model paths disagree")
    if source_model.name != "best.pt" or source_model.parent.name != "weights":
        raise ValueError("Registered pose weights do not use weights/best.pt")
    source_root = source_model.parent.parent
    if source_root.name != run_id:
        raise ValueError("Registered model package directory differs from run_id")
    _require_regular_tree(source_root)

    source_metrics = Path(str(training_run.get("metrics_path") or "")).expanduser().resolve()
    source_onnx = Path(str(onnx_model.get("path") or "")).expanduser().resolve()
    source_onnx_manifest = Path(
        str(onnx_model.get("manifest_path") or "")
    ).expanduser().resolve()
    artifacts = {
        "model": (source_model, str(training_run.get("model_sha256") or "")),
        "metrics": (source_metrics, str(training_run.get("metrics_sha256") or "")),
        "onnx": (source_onnx, str(onnx_model.get("sha256") or "")),
        "onnx_manifest": (
            source_onnx_manifest,
            str(onnx_model.get("manifest_sha256") or ""),
        ),
    }
    relative: dict[str, str] = {}
    for role, (path, expected) in artifacts.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Registered {role} artifact is missing or symlinked: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"Registered {role} SHA-256 differs from its file")
        relative[role] = _safe_relative(source_root, path, field=role).as_posix()
    if training_model.get("model_sha256") != training_run.get("model_sha256"):
        raise ValueError("Registry model digests disagree")
    destination = (
        destination_model_root.expanduser().resolve() / "pose" / set_id / run_id
    )
    return {
        "rows": rows,
        "run_id": run_id,
        "set_id": set_id,
        "source_root": source_root,
        "destination": destination,
        "relative": relative,
        "digests": {role: expected for role, (_path, expected) in artifacts.items()},
    }


def _backup_identity(registry: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = _read_json(receipt_path, field="registry backup receipt")
    if (
        receipt.get("schema_id") != "palette.registry_backup_receipt"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "complete"
        or Path(str(receipt.get("source_registry") or "")).resolve() != registry.resolve()
    ):
        raise ValueError("Registry backup receipt does not bind this registry")
    backup = Path(str(receipt.get("backup_path") or "")).resolve()
    if not backup.is_file() or _sha256_file(backup) != receipt.get("backup_sha256"):
        raise ValueError("Registry backup artifact is missing or changed")
    if _sha256_file(registry) != receipt.get("source_sha256"):
        raise ValueError("Registry changed after the supplied backup was created")
    return receipt


def _canonical_onnx_manifest(
    *, plan: Mapping[str, Any], package_root: Path, contract_path: Path
) -> dict[str, Any]:
    relative = plan["relative"]
    source_manifest_path = package_root / relative["onnx_manifest"]
    source_manifest = _read_json(source_manifest_path, field="source ONNX manifest")
    if (
        source_manifest.get("run_id") != plan["run_id"]
        or source_manifest.get("weights", {}).get("sha256") != plan["digests"]["model"]
        or source_manifest.get("onnx", {}).get("sha256") != plan["digests"]["onnx"]
    ):
        raise ValueError("Source ONNX manifest disagrees with registered artifacts")
    contract = _read_json(contract_path, field="pose model-input contract")
    payload = {
        "status": "complete",
        "run_id": plan["run_id"],
        "set_id": plan["set_id"],
        "task": "pose",
        "weights": {
            "relative_path": relative["model"],
            "sha256": plan["digests"]["model"],
        },
        "onnx": {
            "relative_path": relative["onnx"],
            "sha256": plan["digests"]["onnx"],
        },
        "source_export_manifest": {
            "relative_path": relative["onnx_manifest"],
            "sha256": plan["digests"]["onnx_manifest"],
        },
        "pose_model_input_contract": {
            "relative_path": contract_path.relative_to(package_root).as_posix(),
            "sha256": _sha256_file(contract_path),
            "payload_digest": contract["payload_digest"],
        },
    }
    return {
        "schema_id": CANONICAL_ONNX_MANIFEST_SCHEMA_ID,
        "schema_version": CANONICAL_ONNX_MANIFEST_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def _build_contract(
    *, plan: Mapping[str, Any], package_root: Path, arguments: argparse.Namespace
) -> dict[str, Any]:
    return build_runtime_receipt_pose_model_input_contract(
        set_id=plan["set_id"],
        run_id=plan["run_id"],
        model_package_root=package_root,
        weights_relative_path=Path(plan["relative"]["model"]),
        training_manifest_relative_path=arguments.training_manifest_relative_path,
        training_report_relative_path=arguments.training_report_relative_path,
        training_args_relative_path=arguments.training_args_relative_path,
        training_runtime_receipt_relative_path=(
            arguments.training_runtime_receipt_relative_path
        ),
        model_stride=arguments.model_stride,
        runtime_ultralytics_versions=tuple(arguments.runtime_ultralytics_version),
    )


def _update_registry(
    *, registry: Path, plan: Mapping[str, Any], canonical_manifest: Path,
    contract: Path, publication: Path, backup_receipt: Path,
    backed_up_registry_sha256: str,
) -> None:
    destination = Path(plan["destination"])
    relative = plan["relative"]
    model = destination / relative["model"]
    metrics = destination / relative["metrics"]
    onnx = destination / relative["onnx"]
    canonical_manifest_sha256 = _sha256_file(canonical_manifest)
    promotion = {
        "schema_id": PUBLICATION_SCHEMA_ID,
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "package_root": str(destination),
        "package_publication_path": str(publication),
        "package_publication_sha256": _sha256_file(publication),
        "pose_model_input_contract_path": str(contract),
        "pose_model_input_contract_sha256": _sha256_file(contract),
        "registry_backup_receipt_path": str(backup_receipt.resolve()),
        "registry_backup_receipt_sha256": _sha256_file(backup_receipt.resolve()),
        "selector_activation": False,
    }
    connection = sqlite3.connect(str(registry))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.execute("BEGIN IMMEDIATE")
        if _sha256_file(registry) != backed_up_registry_sha256:
            raise ValueError("Registry changed after backup and before promotion")
        for table in ("training_models", "onnx_models"):
            row = connection.execute(
                f"SELECT metadata_json FROM {table} WHERE run_id = ?",
                (plan["run_id"],),
            ).fetchone()
            if row is None:
                raise ValueError(f"Registry row disappeared from {table}")
            metadata = json.loads(row["metadata_json"] or "{}")
            if not isinstance(metadata, dict):
                raise ValueError(f"{table}.metadata_json is not a mapping")
            metadata["canonical_package_promotion"] = promotion
            if table == "training_models":
                cursor = connection.execute(
                    "UPDATE training_models SET model_path=?, metrics_path=?, metadata_json=? "
                    "WHERE run_id=? AND model_sha256=? AND status='success'",
                    (
                        str(model), str(metrics), json.dumps(metadata, sort_keys=True),
                        plan["run_id"], plan["digests"]["model"],
                    ),
                )
            else:
                cursor = connection.execute(
                    "UPDATE onnx_models SET path=?, manifest_path=?, manifest_sha256=?, "
                    "metadata_json=? WHERE run_id=? AND sha256=?",
                    (
                        str(onnx), str(canonical_manifest), canonical_manifest_sha256,
                        json.dumps(metadata, sort_keys=True), plan["run_id"],
                        plan["digests"]["onnx"],
                    ),
                )
            if cursor.rowcount != 1:
                raise ValueError(f"Registry update preconditions failed for {table}")
        cursor = connection.execute(
            "UPDATE training_runs SET model_path=?, metrics_path=? "
            "WHERE run_id=? AND model_sha256=? AND status='success'",
            (str(model), str(metrics), plan["run_id"], plan["digests"]["model"]),
        )
        if cursor.rowcount != 1:
            raise ValueError("Registry update preconditions failed for training_runs")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def promote(args: argparse.Namespace) -> dict[str, Any]:
    registry = args.registry.expanduser().resolve()
    plan = _validate_registry_source(
        registry=registry,
        run_id=args.model_run_id,
        destination_model_root=args.destination_model_root,
    )
    source_root = Path(plan["source_root"])
    destination = Path(plan["destination"])
    source_inventory = tree_inventory(source_root, hash_content=True)
    contract_document = _build_contract(
        plan=plan, package_root=source_root, arguments=args
    )
    summary = {
        "schema_id": PUBLICATION_SCHEMA_ID,
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "planned" if args.dry_run else "complete",
        "run_id": plan["run_id"],
        "set_id": plan["set_id"],
        "source_package_root": str(source_root),
        "destination_package_root": str(destination),
        "source_inventory": source_inventory.to_json(),
        "pose_model_input_contract_payload_digest": contract_document["payload_digest"],
        "registry_path": str(registry),
        "selector_activation": False,
    }
    if args.dry_run:
        if destination.exists():
            raise FileExistsError(f"Promotion destination already exists: {destination}")
        return summary

    if args.registry_backup_receipt is None:
        raise ValueError("--registry-backup-receipt is required with --apply")
    backup_receipt = args.registry_backup_receipt.expanduser().resolve()
    backup_identity = _backup_identity(registry, backup_receipt)
    if destination.exists():
        raise FileExistsError(f"Promotion destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incoming.{uuid.uuid4().hex}"
    try:
        shutil.copytree(source_root, staging, copy_function=shutil.copy2)
        copied_inventory = tree_inventory(staging, hash_content=True)
        if copied_inventory != source_inventory:
            raise RuntimeError("Copied model package differs from the source inventory")

        contract_path = staging / POSE_MODEL_INPUT_CONTRACT_FILENAME
        _write_json(contract_path, contract_document)
        load_pose_model_input_contract(
            contract_path,
            model_path=staging / plan["relative"]["model"],
            expected_set_id=plan["set_id"],
            expected_run_id=plan["run_id"],
            expected_model_sha256=plan["digests"]["model"],
        )
        canonical_manifest_path = (
            staging / "exports" / "onnx" / f"{plan['run_id']}.canonical.manifest.json"
        )
        _write_json(
            canonical_manifest_path,
            _canonical_onnx_manifest(
                plan=plan, package_root=staging, contract_path=contract_path
            ),
        )
        publication_path = staging / "model_package_publication.json"
        publication = {
            **summary,
            "status": "package_complete_registry_update_pending",
            "created_utc": _utc_now(),
            "copy_verification": "relative_path_size_and_sha256_all_source_files",
            "generated_artifacts": {
                POSE_MODEL_INPUT_CONTRACT_FILENAME: _sha256_file(contract_path),
                canonical_manifest_path.relative_to(staging).as_posix(): (
                    _sha256_file(canonical_manifest_path)
                ),
            },
            "registry_backup_receipt_path": str(backup_receipt),
            "registry_backup_receipt_sha256": _sha256_file(backup_receipt),
        }
        _write_json(publication_path, publication)
        os.rename(staging, destination)
        contract_path = destination / POSE_MODEL_INPUT_CONTRACT_FILENAME
        canonical_manifest_path = (
            destination / "exports" / "onnx" / f"{plan['run_id']}.canonical.manifest.json"
        )
        publication_path = destination / "model_package_publication.json"
        _update_registry(
            registry=registry,
            plan=plan,
            canonical_manifest=canonical_manifest_path,
            contract=contract_path,
            publication=publication_path,
            backup_receipt=backup_receipt,
            backed_up_registry_sha256=str(backup_identity["source_sha256"]),
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    final_inventory = tree_inventory(destination, hash_content=True)
    return {
        **summary,
        "model_path": str(destination / plan["relative"]["model"]),
        "onnx_path": str(destination / plan["relative"]["onnx"]),
        "pose_model_input_contract_path": str(
            destination / POSE_MODEL_INPUT_CONTRACT_FILENAME
        ),
        "canonical_onnx_manifest_path": str(
            destination / "exports" / "onnx" / f"{plan['run_id']}.canonical.manifest.json"
        ),
        "final_inventory": final_inventory.to_json(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--destination-model-root", type=Path, required=True)
    parser.add_argument(
        "--training-manifest-relative-path", type=Path, required=True
    )
    parser.add_argument("--training-report-relative-path", type=Path, required=True)
    parser.add_argument(
        "--training-args-relative-path", type=Path, default=Path("args.yaml")
    )
    parser.add_argument(
        "--training-runtime-receipt-relative-path",
        type=Path,
        default=Path("pose_training_runtime_receipt.json"),
    )
    parser.add_argument("--model-stride", type=int, required=True)
    parser.add_argument("--runtime-ultralytics-version", action="append", default=[])
    parser.add_argument("--registry-backup-receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = promote(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
