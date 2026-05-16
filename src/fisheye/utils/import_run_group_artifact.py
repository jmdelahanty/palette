"""Validate and plan import of a Palette run-group transfer artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from fisheye.utils.system import get_git_info
from fisheye.utils.run_detection_artifact import (
    ARTIFACT_SCHEMA,
    LATEST_POLICY_CHOICES,
    REQUIRED_DETECT_ARRAYS,
    required_arrays_report,
    strict_json_report,
    tree_hash,
)


SUPPORTED_LAYOUTS = {
    "detect_yolo_sparse_v1": {
        "run_family": "detect_runs",
        "required_arrays": REQUIRED_DETECT_ARRAYS,
    },
}


def _utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json_strict(path: Path) -> Any:
    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_zarr_group_metadata(path: Path) -> None:
    zarr_json = path / "zarr.json"
    if zarr_json.exists():
        return
    _write_json(
        zarr_json,
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
        },
    )


def _status_from_errors(errors: list[str]) -> str:
    return "pass" if not errors else "fail"


def _add_validation(
    validations: dict[str, dict[str, Any]],
    name: str,
    errors: list[str],
    **details: Any,
) -> None:
    validations[name] = {"status": _status_from_errors(errors), **details}
    if errors:
        validations[name]["errors"] = list(errors)


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        os.path.commonpath([str(path.resolve()), str(directory.resolve())])
    except (OSError, ValueError):
        return False
    return os.path.commonpath([str(path.resolve()), str(directory.resolve())]) == str(
        directory.resolve()
    )


def _safe_extract_artifact(tarball_path: Path, destination: Path) -> Path:
    """Extract a tarball after rejecting links, absolute paths, and traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    roots: set[str] = set()
    with tarfile.open(tarball_path, "r:*") as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError("artifact tarball is empty")
        for member in members:
            member_path = Path(member.name)
            if not member_path.parts:
                continue
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe tar member path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"tar member links are not allowed: {member.name}")
            target_path = destination / member.name
            if not _is_within_directory(target_path, destination):
                raise ValueError(f"tar member escapes destination: {member.name}")
            roots.add(member_path.parts[0])
        tar.extractall(destination, members=members)

    if "palette_run_group_artifact" in roots:
        return destination / "palette_run_group_artifact"
    if len(roots) == 1:
        return destination / next(iter(roots))
    raise ValueError(f"expected one artifact root, found: {sorted(roots)}")


def _manifest_target_path(manifest: dict[str, Any]) -> Optional[Path]:
    value = manifest.get("target_archive_path")
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser()


def _validate_manifest_shape(manifest: Any) -> tuple[Optional[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return None, ["manifest must be a JSON object"]
    if manifest.get("artifact_schema") != ARTIFACT_SCHEMA:
        errors.append(
            f"artifact_schema must be {ARTIFACT_SCHEMA!r}, got {manifest.get('artifact_schema')!r}"
        )
    run_family = manifest.get("run_family")
    run_name = manifest.get("run_name")
    target_group_path = manifest.get("target_group_path")
    layout = manifest.get("layout")
    latest_policy = manifest.get("latest_policy")
    if not isinstance(run_family, str) or not run_family:
        errors.append("run_family must be a non-empty string")
    if not isinstance(run_name, str) or not run_name:
        errors.append("run_name must be a non-empty string")
    if not isinstance(target_group_path, str) or not target_group_path:
        errors.append("target_group_path must be a non-empty string")
    if isinstance(target_group_path, str) and (
        target_group_path.startswith("/") or ".." in Path(target_group_path).parts
    ):
        errors.append("target_group_path must be a relative path without '..'")
    if layout not in SUPPORTED_LAYOUTS:
        errors.append(f"unsupported layout: {layout!r}")
    if latest_policy not in LATEST_POLICY_CHOICES:
        errors.append(f"latest_policy must be one of {LATEST_POLICY_CHOICES}")
    if isinstance(run_family, str) and isinstance(run_name, str) and isinstance(target_group_path, str):
        expected = f"{run_family}/{run_name}"
        if target_group_path != expected:
            errors.append(f"target_group_path must be {expected!r} for this importer slice")
    if layout in SUPPORTED_LAYOUTS and isinstance(run_family, str):
        expected_family = SUPPORTED_LAYOUTS[layout]["run_family"]
        if run_family != expected_family:
            errors.append(f"layout {layout!r} expects run_family {expected_family!r}")
    if _manifest_target_path(manifest) is None:
        errors.append("target_archive_path must be a non-empty string")
    return manifest, errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_latest_attr(family_path: Path, run_name: str) -> None:
    _write_zarr_group_metadata(family_path)
    zarr_json = family_path / "zarr.json"
    payload = _read_json_strict(zarr_json)
    if not isinstance(payload, dict):
        raise ValueError(f"{zarr_json} is not a JSON object")
    attrs = payload.setdefault("attributes", {})
    if not isinstance(attrs, dict):
        raise ValueError(f"{zarr_json} attributes must be an object")
    attrs["latest"] = run_name
    _write_json(zarr_json, payload)


def build_import_plan(
    *,
    tarball_path: Path,
    target_zarr: Optional[Path] = None,
    overwrite: bool = False,
    validate_source_inputs: bool = True,
    keep_extracted: Optional[Path] = None,
) -> dict[str, Any]:
    """Return a dry-run import plan for a run-group artifact tarball."""
    tarball_path = tarball_path.expanduser().resolve()
    errors: list[str] = []
    validations: dict[str, dict[str, Any]] = {}
    planned_actions: list[dict[str, Any]] = []

    if not tarball_path.is_file():
        return {
            "status": "failed",
            "apply": False,
            "source_tarball": str(tarball_path),
            "errors": [f"artifact tarball not found: {tarball_path}"],
            "validations": {},
            "planned_actions": [],
        }

    temp_owner: Optional[tempfile.TemporaryDirectory[str]] = None
    if keep_extracted is None:
        temp_owner = tempfile.TemporaryDirectory(prefix="palette_run_group_import_")
        extract_root = Path(temp_owner.name)
    else:
        extract_root = keep_extracted.expanduser().resolve()
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True)

    try:
        artifact_root = _safe_extract_artifact(tarball_path, extract_root)
        manifest_path = artifact_root / "artifact_manifest.json"
        run_group_dir = artifact_root / "run_group"

        manifest_payload: Any = None
        manifest_errors: list[str] = []
        if not manifest_path.exists():
            manifest_errors.append("artifact_manifest.json is missing")
        else:
            try:
                manifest_payload = _read_json_strict(manifest_path)
            except Exception as exc:
                manifest_errors.append(f"artifact_manifest.json is not strict JSON: {exc}")
        manifest, shape_errors = _validate_manifest_shape(manifest_payload)
        manifest_errors.extend(shape_errors)
        _add_validation(
            validations,
            "manifest",
            manifest_errors,
            manifest_path=str(manifest_path),
        )
        errors.extend(f"manifest: {error}" for error in manifest_errors)

        manifest = manifest or {}
        run_family = str(manifest.get("run_family") or "")
        run_name = str(manifest.get("run_name") or "")
        target_group_path = str(manifest.get("target_group_path") or "")
        latest_policy = str(manifest.get("latest_policy") or "")
        layout = str(manifest.get("layout") or "")
        manifest_target = _manifest_target_path(manifest)
        resolved_target = (
            target_zarr.expanduser().resolve()
            if target_zarr is not None
            else manifest_target.expanduser().resolve()
            if manifest_target is not None
            else None
        )

        strict_report = strict_json_report(run_group_dir) if run_group_dir.exists() else {
            "status": "fail",
            "files_checked": 0,
            "bad_json_files": 0,
            "bad_files": [{"path": "run_group", "error": "run_group directory is missing"}],
        }
        strict_errors = [] if strict_report["status"] == "pass" else ["strict JSON validation failed"]
        _add_validation(validations, "strict_json", strict_errors, report=strict_report)
        errors.extend(f"strict_json: {error}" for error in strict_errors)

        arrays_report = required_arrays_report(run_group_dir) if run_group_dir.exists() else {
            "status": "fail",
            "run_group_zarr_json_present": False,
            "arrays": [],
            "missing_arrays": list(REQUIRED_DETECT_ARRAYS),
        }
        array_errors = [] if arrays_report["status"] == "pass" else ["required array validation failed"]
        _add_validation(validations, "required_arrays", array_errors, report=arrays_report)
        errors.extend(f"required_arrays: {error}" for error in array_errors)

        hash_errors: list[str] = []
        expected_hash = None
        actual_hash = None
        checksums = manifest.get("checksums") if isinstance(manifest, dict) else None
        if isinstance(checksums, dict):
            expected_hash = checksums.get("run_group_tree_hash")
        if not isinstance(expected_hash, str) or not expected_hash:
            hash_errors.append("manifest checksums.run_group_tree_hash is missing")
        if run_group_dir.exists():
            actual_hash = tree_hash(run_group_dir)
            if isinstance(expected_hash, str) and expected_hash and actual_hash != expected_hash:
                hash_errors.append("run_group_tree_hash mismatch")
        else:
            hash_errors.append("run_group directory is missing")
        _add_validation(
            validations,
            "run_group_tree_hash",
            hash_errors,
            expected=expected_hash,
            actual=actual_hash,
        )
        errors.extend(f"run_group_tree_hash: {error}" for error in hash_errors)

        target_errors: list[str] = []
        if resolved_target is None:
            target_errors.append("target archive path could not be resolved")
        elif not resolved_target.exists():
            target_errors.append(f"target archive does not exist: {resolved_target}")
        elif not resolved_target.is_dir():
            target_errors.append(f"target archive is not a directory: {resolved_target}")
        _add_validation(
            validations,
            "target_archive",
            target_errors,
            path=str(resolved_target) if resolved_target is not None else None,
            override_used=target_zarr is not None,
        )
        errors.extend(f"target_archive: {error}" for error in target_errors)

        final_path = resolved_target / target_group_path if resolved_target is not None and target_group_path else None
        family_path = resolved_target / run_family if resolved_target is not None and run_family else None
        incoming_path = family_path / ".incoming" / run_name if family_path is not None and run_name else None
        failed_path_pattern = (
            str(family_path / ".failed" / f"{run_name}_{_utc_now_label()}")
            if family_path is not None and run_name
            else None
        )

        target_path_errors: list[str] = []
        if final_path is None:
            target_path_errors.append("final target path could not be resolved")
        elif final_path.exists() and not overwrite:
            target_path_errors.append(f"final target already exists: {final_path}")
        if incoming_path is not None and incoming_path.exists():
            target_path_errors.append(f"incoming path already exists: {incoming_path}")
        _add_validation(
            validations,
            "target_paths",
            target_path_errors,
            final_path=str(final_path) if final_path is not None else None,
            final_exists=final_path.exists() if final_path is not None else None,
            incoming_path=str(incoming_path) if incoming_path is not None else None,
            incoming_exists=incoming_path.exists() if incoming_path is not None else None,
            overwrite=overwrite,
        )
        errors.extend(f"target_paths: {error}" for error in target_path_errors)

        source_errors: list[str] = []
        checked_sources: list[dict[str, Any]] = []
        source_inputs = manifest.get("source_inputs") if isinstance(manifest, dict) else None
        if validate_source_inputs and isinstance(source_inputs, list):
            for item in source_inputs:
                if not isinstance(item, dict):
                    source_errors.append("source_inputs contains a non-object item")
                    continue
                path_value = item.get("path")
                role = item.get("role")
                if not isinstance(path_value, str) or not path_value:
                    source_errors.append(f"source input {role!r} has no path")
                    continue
                path = Path(path_value).expanduser()
                exists = path.exists()
                checked_sources.append({"role": role, "path": str(path), "exists": exists})
                if not exists:
                    source_errors.append(f"source input does not exist: {path}")
        elif validate_source_inputs:
            source_errors.append("source_inputs must be a list")
        _add_validation(
            validations,
            "source_inputs",
            source_errors,
            checked=checked_sources,
            skipped=not validate_source_inputs,
        )
        errors.extend(f"source_inputs: {error}" for error in source_errors)

        if manifest and validations["manifest"]["status"] == "pass":
            receipt_path = (
                family_path / ".imports" / f"{run_name}_import_receipt.json"
                if family_path is not None and run_name
                else None
            )
            planned_actions.extend(
                [
                    {
                        "action": "unpack_artifact_to_incoming",
                        "path": str(incoming_path) if incoming_path is not None else None,
                    },
                    {
                        "action": "validate_incoming_run_group",
                        "checks": [
                            "strict_json",
                            "required_arrays",
                            "run_group_tree_hash",
                            "target_paths",
                        ],
                    },
                    {
                        "action": "promote_incoming_to_final",
                        "source": str(incoming_path) if incoming_path is not None else None,
                        "destination": str(final_path) if final_path is not None else None,
                        "overwrite": overwrite,
                    },
                    {
                        "action": "update_latest",
                        "policy": latest_policy,
                        "would_update": latest_policy != "do_not_set_latest",
                    },
                    {
                        "action": "write_import_receipt",
                        "path": str(receipt_path) if receipt_path is not None else None,
                    },
                ]
            )

        return {
            "status": "ok" if not errors else "failed",
            "apply": False,
            "source_tarball": str(tarball_path),
            "target_archive_path": str(resolved_target) if resolved_target is not None else None,
            "target_group_path": target_group_path or None,
            "run_family": run_family or None,
            "run_name": run_name or None,
            "layout": layout or None,
            "incoming_path": str(incoming_path) if incoming_path is not None else None,
            "failed_path_pattern": failed_path_pattern,
            "final_path": str(final_path) if final_path is not None else None,
            "receipt_path": (
                str(family_path / ".imports" / f"{run_name}_import_receipt.json")
                if family_path is not None and run_name
                else None
            ),
            "latest_policy": latest_policy or None,
            "validations": validations,
            "planned_actions": planned_actions,
            "errors": errors,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "apply": False,
            "source_tarball": str(tarball_path),
            "errors": [str(exc)],
            "validations": validations,
            "planned_actions": planned_actions,
        }
    finally:
        if temp_owner is not None:
            temp_owner.cleanup()


def apply_import(
    *,
    tarball_path: Path,
    target_zarr: Optional[Path] = None,
    overwrite: bool = False,
    validate_source_inputs: bool = True,
) -> dict[str, Any]:
    """Apply an artifact import using .incoming promotion and receipt sidecar."""
    plan = build_import_plan(
        tarball_path=tarball_path,
        target_zarr=target_zarr,
        overwrite=overwrite,
        validate_source_inputs=validate_source_inputs,
    )
    if plan["status"] != "ok":
        plan["apply"] = True
        plan["applied"] = False
        return plan

    tarball_path = Path(plan["source_tarball"]).expanduser().resolve()
    target_archive = Path(str(plan["target_archive_path"]))
    run_family = str(plan["run_family"])
    run_name = str(plan["run_name"])
    latest_policy = str(plan["latest_policy"])
    incoming_path = Path(str(plan["incoming_path"]))
    final_path = Path(str(plan["final_path"]))
    receipt_path = Path(str(plan["receipt_path"]))
    family_path = target_archive / run_family
    failed_base = family_path / ".failed"
    failed_path = failed_base / f"{run_name}_{_utc_now_label()}"
    apply_validations: dict[str, dict[str, Any]] = {}
    apply_errors: list[str] = []
    latest_updated = False
    imported = False

    temp_owner = tempfile.TemporaryDirectory(prefix="palette_run_group_apply_")
    try:
        extract_root = Path(temp_owner.name)
        artifact_root = _safe_extract_artifact(tarball_path, extract_root)
        manifest_path = artifact_root / "artifact_manifest.json"
        manifest = _read_json_strict(manifest_path)
        run_group_dir = artifact_root / "run_group"
        expected_hash = manifest.get("checksums", {}).get("run_group_tree_hash")

        family_path.mkdir(parents=True, exist_ok=True)
        _write_zarr_group_metadata(family_path)
        (family_path / ".incoming").mkdir(exist_ok=True)
        failed_base.mkdir(exist_ok=True)

        if incoming_path.exists():
            raise FileExistsError(f"incoming path already exists: {incoming_path}")
        if final_path.exists():
            if not overwrite:
                raise FileExistsError(f"final target already exists: {final_path}")
            shutil.rmtree(final_path)

        shutil.copytree(run_group_dir, incoming_path)

        strict_report = strict_json_report(incoming_path)
        strict_errors = [] if strict_report["status"] == "pass" else ["strict JSON validation failed"]
        _add_validation(apply_validations, "incoming_strict_json", strict_errors, report=strict_report)
        apply_errors.extend(f"incoming_strict_json: {error}" for error in strict_errors)

        arrays_report = required_arrays_report(incoming_path)
        array_errors = [] if arrays_report["status"] == "pass" else ["required array validation failed"]
        _add_validation(apply_validations, "incoming_required_arrays", array_errors, report=arrays_report)
        apply_errors.extend(f"incoming_required_arrays: {error}" for error in array_errors)

        actual_hash = tree_hash(incoming_path)
        hash_errors = []
        if actual_hash != expected_hash:
            hash_errors.append("incoming run_group_tree_hash mismatch")
        _add_validation(
            apply_validations,
            "incoming_run_group_tree_hash",
            hash_errors,
            expected=expected_hash,
            actual=actual_hash,
        )
        apply_errors.extend(f"incoming_run_group_tree_hash: {error}" for error in hash_errors)
        if apply_errors:
            raise ValueError("; ".join(apply_errors))

        shutil.move(str(incoming_path), str(final_path))
        imported = True

        if latest_policy != "do_not_set_latest":
            _update_latest_attr(family_path, run_name)
            latest_updated = True

        receipt = {
            "schema_version": 1,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source_tarball": str(tarball_path),
            "source_tarball_sha256": _sha256_file(tarball_path),
            "target_archive_path": str(target_archive),
            "target_group_path": plan["target_group_path"],
            "run_family": run_family,
            "run_name": run_name,
            "layout": plan["layout"],
            "final_path": str(final_path),
            "incoming_path": str(incoming_path),
            "latest_policy": latest_policy,
            "latest_updated": latest_updated,
            "manifest": manifest,
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "importer": {
                "command": " ".join(sys.argv),
                "git": get_git_info(),
            },
            "validations": {
                **plan["validations"],
                **apply_validations,
            },
        }
        _write_json(receipt_path, receipt)

        return {
            **plan,
            "status": "ok",
            "apply": True,
            "applied": True,
            "latest_updated": latest_updated,
            "receipt_path": str(receipt_path),
            "validations": {
                **plan["validations"],
                **apply_validations,
            },
            "errors": [],
        }
    except Exception as exc:
        if incoming_path.exists():
            failed_base.mkdir(parents=True, exist_ok=True)
            if failed_path.exists():
                failed_path = failed_base / f"{run_name}_{_utc_now_label()}_1"
            shutil.move(str(incoming_path), str(failed_path))
        return {
            **plan,
            "status": "failed",
            "apply": True,
            "applied": False,
            "imported": imported,
            "failed_path": str(failed_path) if failed_path.exists() else None,
            "validations": {
                **plan.get("validations", {}),
                **apply_validations,
            },
            "errors": [*plan.get("errors", []), str(exc)],
        }
    finally:
        temp_owner.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run validation and import planning for Palette run-group artifacts."
    )
    parser.add_argument("tarball", type=Path, help="Run-group artifact .tar.gz path")
    parser.add_argument(
        "--target-zarr",
        type=Path,
        default=None,
        help="Override target analysis Zarr path from artifact_manifest.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow planning/apply over an existing final target path",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the import after validation; default is dry-run only",
    )
    parser.add_argument(
        "--skip-source-input-checks",
        action="store_true",
        help="Do not require source_inputs paths to exist during dry-run validation",
    )
    parser.add_argument(
        "--keep-extracted",
        type=Path,
        default=None,
        help="Debug only: extract artifact validation tree to this path instead of a temp dir",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.apply:
        if args.keep_extracted is not None:
            parser.error("--keep-extracted is only supported for dry-run validation")
        result = apply_import(
            tarball_path=args.tarball,
            target_zarr=args.target_zarr,
            overwrite=args.overwrite,
            validate_source_inputs=not args.skip_source_input_checks,
        )
    else:
        result = build_import_plan(
            tarball_path=args.tarball,
            target_zarr=args.target_zarr,
            overwrite=args.overwrite,
            validate_source_inputs=not args.skip_source_input_checks,
            keep_extracted=args.keep_extracted,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
