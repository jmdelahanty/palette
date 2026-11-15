#!/usr/bin/env python3
"""Publish one successful registered pose or detect model as a canonical package.

The package lives at ``<model root>/<task>/<set_id>/<run_id>/``.

- **Copy mode** (the package is elsewhere, e.g. pose packages under the training
  tree): publication copies it through a hidden sibling, verifies every source
  file by relative path, size and SHA-256, adds the generated evidence, renames
  it into place atomically, then rebinds the registered artifact paths.
- **In-place mode** (the package is already at its canonical location, e.g. the
  detect packages under ``models/detect/``): every registered digest is checked,
  only new files are added (never an existing one rewritten), and the inventory
  afterwards must be the original plus exactly those files.

Generated evidence: the canonical ONNX manifest
(``fisheye.shared.canonical_onnx_model_manifest``, v2 with ``sha256:`` tokens),
``model_package_publication.json``, and for pose the pose model-input contract.
The registry's ONNX manifest pointer moves to the canonical manifest in one
SQLite transaction guarded by a registry backup receipt.
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
from fisheye.shared.canonical_onnx_model_manifest import build_canonical_onnx_manifest
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.pose_model_input_contract import (
    POSE_MODEL_INPUT_CONTRACT_FILENAME,
    build_runtime_receipt_pose_model_input_contract,
    load_pose_model_input_contract,
)


PUBLICATION_SCHEMA_IDS = {
    "pose": "palette.registered_pose_model_package_publication",
    "detect": "palette.registered_detect_model_package_publication",
}
PUBLICATION_SCHEMA_VERSION = 1
PUBLICATION_FILENAME = "model_package_publication.json"


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
    *, registry: Path, run_id: str, destination_model_root: Path, task: str
) -> dict[str, Any]:
    rows = _registry_rows(registry, run_id)
    training_run = rows["training_runs"]
    training_model = rows["training_models"]
    onnx_model = rows["onnx_models"]
    if training_run.get("task_type") != task or training_model.get("task_type") != task:
        raise ValueError(f"Registered model is not a {task} model")
    if training_run.get("status") != "success" or training_model.get("status") != "success":
        raise ValueError(f"Registered {task} model is not successful")
    set_id = str(training_run.get("set_id") or "")
    if not set_id or training_model.get("set_id") != set_id or onnx_model.get("set_id") != set_id:
        raise ValueError("Registry set identity is incomplete or inconsistent")
    source_model = Path(str(training_run.get("model_path") or "")).expanduser().resolve()
    if Path(str(training_model.get("model_path") or "")).expanduser().resolve() != source_model:
        raise ValueError("training_runs and training_models model paths disagree")
    if source_model.name != "best.pt" or source_model.parent.name != "weights":
        raise ValueError(f"Registered {task} weights do not use weights/best.pt")
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
        destination_model_root.expanduser().resolve() / task / set_id / run_id
    )
    return {
        "task": task,
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
    *, plan: Mapping[str, Any], package_root: Path, contract_path: Path | None
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
    contract = None
    if contract_path is not None:
        document = _read_json(contract_path, field="pose model-input contract")
        contract = (
            contract_path.relative_to(package_root).as_posix(),
            _sha256_file(contract_path),
            document["payload_digest"],
        )
    return build_canonical_onnx_manifest(
        task=plan["task"],
        run_id=plan["run_id"],
        set_id=plan["set_id"],
        weights=(relative["model"], plan["digests"]["model"]),
        onnx=(relative["onnx"], plan["digests"]["onnx"]),
        source_export_manifest=(relative["onnx_manifest"], plan["digests"]["onnx_manifest"]),
        pose_model_input_contract=contract,
    )


def _canonical_manifest_relative(plan: Mapping[str, Any]) -> Path:
    return Path("exports") / "onnx" / f"{plan['run_id']}.canonical.manifest.json"


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
    contract: Path | None, publication: Path, backup_receipt: Path,
    backed_up_registry_sha256: str,
) -> None:
    destination = Path(plan["destination"])
    relative = plan["relative"]
    model = destination / relative["model"]
    metrics = destination / relative["metrics"]
    onnx = destination / relative["onnx"]
    canonical_manifest_sha256 = _sha256_file(canonical_manifest)
    promotion = {
        "schema_id": PUBLICATION_SCHEMA_IDS[plan["task"]],
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "package_root": str(destination),
        "package_publication_path": str(publication),
        "package_publication_sha256": _sha256_file(publication),
        "canonical_onnx_manifest_path": str(canonical_manifest),
        "canonical_onnx_manifest_sha256": canonical_manifest_sha256,
        "registry_backup_receipt_path": str(backup_receipt.resolve()),
        "registry_backup_receipt_sha256": _sha256_file(backup_receipt.resolve()),
        "selector_activation": False,
    }
    if contract is not None:
        promotion["pose_model_input_contract_path"] = str(contract)
        promotion["pose_model_input_contract_sha256"] = _sha256_file(contract)
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


def _file_digests(root: Path) -> dict[str, tuple[int, str]]:
    return {
        item.relative_path: (item.size_bytes, _sha256_file(root / item.relative_path))
        for item in tree_inventory(root, hash_content=False).files
    }


def _generate(
    *, plan: Mapping[str, Any], package_root: Path, contract_document: Mapping[str, Any] | None,
    summary: Mapping[str, Any], backup_receipt: Path, in_place: bool,
) -> dict[str, Path]:
    """Write the generated evidence into ``package_root``; never overwrite a file."""

    written: dict[str, Path] = {}

    def write(relative: Path, document: Mapping[str, Any]) -> Path:
        path = package_root / relative
        if path.exists():
            raise FileExistsError(f"Promotion would overwrite an existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, document)
        written[relative.as_posix()] = path
        return path

    contract_path = None
    if contract_document is not None:
        contract_path = write(Path(POSE_MODEL_INPUT_CONTRACT_FILENAME), contract_document)
        load_pose_model_input_contract(
            contract_path,
            model_path=package_root / plan["relative"]["model"],
            expected_set_id=plan["set_id"],
            expected_run_id=plan["run_id"],
            expected_model_sha256=plan["digests"]["model"],
        )
    write(
        _canonical_manifest_relative(plan),
        _canonical_onnx_manifest(plan=plan, package_root=package_root, contract_path=contract_path),
    )
    write(
        Path(PUBLICATION_FILENAME),
        {
            **summary,
            "status": "package_complete_registry_update_pending",
            "created_utc": _utc_now(),
            "publication_mode": "in_place" if in_place else "copy",
            "copy_verification": (
                "registered_sha256_of_every_bound_artifact_and_unchanged_inventory"
                if in_place
                else "relative_path_size_and_sha256_all_source_files"
            ),
            "generated_artifacts": {name: _sha256_file(path) for name, path in written.items()},
            "registry_backup_receipt_path": str(backup_receipt),
            "registry_backup_receipt_sha256": _sha256_file(backup_receipt),
        },
    )
    return written


def promote(args: argparse.Namespace, *, task: str = "pose") -> dict[str, Any]:
    registry = args.registry.expanduser().resolve()
    plan = _validate_registry_source(
        registry=registry,
        run_id=args.model_run_id,
        destination_model_root=args.destination_model_root,
        task=task,
    )
    source_root = Path(plan["source_root"])
    destination = Path(plan["destination"])
    in_place = source_root.resolve() == destination.resolve()
    source_inventory = tree_inventory(source_root, hash_content=True)
    contract_document = (
        _build_contract(plan=plan, package_root=source_root, arguments=args)
        if task == "pose"
        else None
    )
    summary = {
        "schema_id": PUBLICATION_SCHEMA_IDS[task],
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "planned" if args.dry_run else "complete",
        "task": task,
        "run_id": plan["run_id"],
        "set_id": plan["set_id"],
        "publication_mode": "in_place" if in_place else "copy",
        "source_package_root": str(source_root),
        "destination_package_root": str(destination),
        "source_inventory": source_inventory.to_json(),
        "registry_path": str(registry),
        "selector_activation": False,
    }
    if contract_document is not None:
        summary["pose_model_input_contract_payload_digest"] = contract_document["payload_digest"]
    canonical_relative = _canonical_manifest_relative(plan)
    if args.dry_run:
        if not in_place and destination.exists():
            raise FileExistsError(f"Promotion destination already exists: {destination}")
        if in_place and (destination / canonical_relative).exists():
            raise FileExistsError(
                f"Package already carries a canonical manifest: {destination / canonical_relative}"
            )
        return summary

    if args.registry_backup_receipt is None:
        raise ValueError("--registry-backup-receipt is required with --apply")
    backup_receipt = args.registry_backup_receipt.expanduser().resolve()
    backup_identity = _backup_identity(registry, backup_receipt)
    if in_place:
        before = _file_digests(destination)
        written = _generate(
            plan=plan, package_root=destination, contract_document=contract_document,
            summary=summary, backup_receipt=backup_receipt, in_place=True,
        )
        after = _file_digests(destination)
        if {name: after.get(name) for name in before} != before or set(after) - set(before) != set(
            written
        ):
            raise RuntimeError("In-place promotion changed files other than its own evidence")
    else:
        if destination.exists():
            raise FileExistsError(f"Promotion destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.incoming.{uuid.uuid4().hex}"
        try:
            shutil.copytree(source_root, staging, copy_function=shutil.copy2)
            copied_inventory = tree_inventory(staging, hash_content=True)
            if copied_inventory != source_inventory:
                raise RuntimeError("Copied model package differs from the source inventory")
            _generate(
                plan=plan, package_root=staging, contract_document=contract_document,
                summary=summary, backup_receipt=backup_receipt, in_place=False,
            )
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    contract_path = (
        destination / POSE_MODEL_INPUT_CONTRACT_FILENAME if contract_document is not None else None
    )
    canonical_manifest_path = destination / canonical_relative
    _update_registry(
        registry=registry,
        plan=plan,
        canonical_manifest=canonical_manifest_path,
        contract=contract_path,
        publication=destination / PUBLICATION_FILENAME,
        backup_receipt=backup_receipt,
        backed_up_registry_sha256=str(backup_identity["source_sha256"]),
    )
    final_inventory = tree_inventory(destination, hash_content=True)
    result = {
        **summary,
        "model_path": str(destination / plan["relative"]["model"]),
        "onnx_path": str(destination / plan["relative"]["onnx"]),
        "canonical_onnx_manifest_path": str(canonical_manifest_path),
        "final_inventory": final_inventory.to_json(),
    }
    if contract_path is not None:
        result["pose_model_input_contract_path"] = str(contract_path)
    return result


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--destination-model-root", type=Path, required=True)
    parser.add_argument("--registry-backup-receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")


def add_pose_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--training-manifest-relative-path", type=Path, required=required
    )
    parser.add_argument("--training-report-relative-path", type=Path, required=required)
    parser.add_argument(
        "--training-args-relative-path", type=Path, default=Path("args.yaml")
    )
    parser.add_argument(
        "--training-runtime-receipt-relative-path",
        type=Path,
        default=Path("pose_training_runtime_receipt.json"),
    )
    parser.add_argument("--model-stride", type=int, required=required)
    parser.add_argument("--runtime-ultralytics-version", action="append", default=[])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pose", "detect"), required=True)
    _add_common_arguments(parser)
    add_pose_arguments(parser, required=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.task == "pose":
        missing = [
            flag for flag, value in (
                ("--training-manifest-relative-path", args.training_manifest_relative_path),
                ("--training-report-relative-path", args.training_report_relative_path),
                ("--model-stride", args.model_stride),
            ) if value is None
        ]
        if missing:
            raise SystemExit(f"--task pose requires {', '.join(missing)}")
    result = promote(args, task=args.task)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
