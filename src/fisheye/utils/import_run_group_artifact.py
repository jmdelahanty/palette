"""Validate and plan import of a Palette run-group transfer artifact."""

from __future__ import annotations

from fisheye.shared.json_safety import write_json_atomic as _write_json
from fisheye.shared.batch_logging import utc_now_compact as _utc_now_label
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

from fisheye.shared.system_metadata import get_git_info
from fisheye.utils.run_detection_artifact import (
    ARTIFACT_SCHEMA,
    DETECTION_ARTIFACT_FAMILY_CONTRACT,
    DETECTION_ARTIFACT_LAYOUT,
    DO_NOT_SET_LATEST,
    LATEST_POLICY_CHOICES,
    REQUIRED_DETECT_ARRAYS,
    RUN_FAMILY,
    artifact_run_metadata_report,
    required_arrays_report,
    strict_json_report,
    tree_hash,
)


SUPPORTED_LAYOUTS = {
    DETECTION_ARTIFACT_LAYOUT: {
        "run_family": RUN_FAMILY,
        "required_arrays": REQUIRED_DETECT_ARRAYS,
    },
}


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json_strict(path: Path) -> Any:
    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


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


def _stamp_artifact_family_metadata(path: Path) -> None:
    """Persist the dedicated family as an explicit non-selector namespace."""

    _write_zarr_group_metadata(path)
    zarr_json = path / "zarr.json"
    payload = _read_json_strict(zarr_json)
    if not isinstance(payload, dict):
        raise ValueError(f"{zarr_json} is not a JSON object")
    attrs = payload.setdefault("attributes", {})
    if not isinstance(attrs, dict):
        raise ValueError(f"{zarr_json} attributes must be an object")
    attrs["artifact_family_contract"] = DETECTION_ARTIFACT_FAMILY_CONTRACT
    attrs["stage_selector_eligible"] = False
    for forbidden in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        if forbidden in attrs:
            raise ValueError(
                "Detection artifact family cannot carry selector attr "
                f"{forbidden!r}."
            )
    _write_json(zarr_json, payload)


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


def _is_safe_relative_group_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


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
        not _is_safe_relative_group_path(target_group_path)
    ):
        errors.append("target_group_path must be a relative path without '..'")
    intended_target_group_path = manifest.get("intended_target_group_path")
    if isinstance(intended_target_group_path, str) and intended_target_group_path:
        if not _is_safe_relative_group_path(intended_target_group_path):
            errors.append("intended_target_group_path must be a relative path without '..'")
    if layout not in SUPPORTED_LAYOUTS:
        errors.append(f"unsupported layout: {layout!r}")
    if latest_policy not in LATEST_POLICY_CHOICES:
        errors.append(f"latest_policy must be one of {LATEST_POLICY_CHOICES}")
    elif latest_policy != DO_NOT_SET_LATEST:
        errors.append(
            "detection_artifact_runs forbids latest or authoritative promotion"
        )
    if manifest.get("selector_policy") != "never_select_or_promote_v1":
        errors.append("selector_policy must be 'never_select_or_promote_v1'")
    if manifest.get("artifact_family_contract") != DETECTION_ARTIFACT_FAMILY_CONTRACT:
        errors.append(
            f"artifact_family_contract must be {DETECTION_ARTIFACT_FAMILY_CONTRACT!r}"
        )
    if manifest.get("stage_selector_eligible") is not False:
        errors.append("stage_selector_eligible must be false")
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


def _effective_target_group_path(
    manifest: dict[str, Any],
    *,
    use_intended_target: bool,
) -> tuple[str, str, list[str]]:
    """Resolve the run-group destination path from artifact manifest metadata."""
    errors: list[str] = []
    target_group_path = manifest.get("target_group_path")
    intended_target_group_path = manifest.get("intended_target_group_path")
    run_family = str(manifest.get("run_family") or "")
    run_name = str(manifest.get("run_name") or "")
    selected_source = "target_group_path"
    selected = target_group_path

    if use_intended_target:
        if not isinstance(intended_target_group_path, str) or not intended_target_group_path:
            errors.append("--use-intended-target was requested, but intended_target_group_path is missing")
        else:
            selected = intended_target_group_path
            selected_source = "intended_target_group_path"

    if not isinstance(selected, str) or not selected:
        return "", selected_source, errors
    if not _is_safe_relative_group_path(selected):
        errors.append(f"{selected_source} must be a relative path without '..'")
        return selected, selected_source, errors
    if run_family and run_name:
        parts = Path(selected).parts
        if len(parts) < 2 or parts[-2:] != (run_family, run_name):
            errors.append(f"{selected_source} must end with {run_family}/{run_name}")
    return selected, selected_source, errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file_snapshot(source: Path, destination: Path) -> str:
    """Copy one immutable transaction snapshot and return its SHA-256 digest."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return _sha256_file(destination)


def _atomic_move(source: Path, destination: Path) -> None:
    """Atomically rename one import path within the target archive filesystem."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _verified_atomic_move(source: Path, destination: Path) -> None:
    """Rename and reconcile a move that may report an error after taking effect."""

    if destination.exists():
        raise FileExistsError(f"atomic-move destination already exists: {destination}")
    try:
        _atomic_move(source, destination)
    except Exception:
        if not source.exists() and destination.exists():
            return
        raise
    if source.exists() or not destination.exists():  # pragma: no cover - hostile FS
        raise RuntimeError(
            f"atomic move did not reach one exact destination: {source} -> {destination}"
        )


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _verified_remove_path(path: Path) -> None:
    """Remove a path and reconcile errors reported after removal completed."""

    try:
        _remove_path(path)
    except Exception:
        if not path.exists():
            return
        raise
    if path.exists():  # pragma: no cover - hostile FS
        raise RuntimeError(f"path removal did not complete: {path}")


def _orphaned_import_transaction_paths(family_path: Path, run_name: str) -> list[Path]:
    incoming_parent = family_path / ".incoming"
    if not incoming_parent.is_dir():
        return []
    prefixes = (
        f".{run_name}_previous_",
        f".{run_name}_previous_receipt_",
        f".{run_name}_import_receipt_",
    )
    return sorted(
        path
        for path in incoming_parent.iterdir()
        if any(path.name.startswith(prefix) for prefix in prefixes)
    )


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
    use_intended_target: bool = False,
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

    staging_owner = tempfile.TemporaryDirectory(prefix="palette_run_group_plan_source_")
    staged_tarball = Path(staging_owner.name) / "artifact.tar"
    source_tarball_sha256: Optional[str] = None
    manifest_sha256: Optional[str] = None
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
        source_tarball_sha256 = _copy_file_snapshot(tarball_path, staged_tarball)
        artifact_root = _safe_extract_artifact(staged_tarball, extract_root)
        manifest_path = artifact_root / "artifact_manifest.json"
        run_group_dir = artifact_root / "run_group"

        manifest_payload: Any = None
        manifest_errors: list[str] = []
        if not manifest_path.exists():
            manifest_errors.append("artifact_manifest.json is missing")
        else:
            try:
                manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
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
        manifest_target_group_path = str(manifest.get("target_group_path") or "")
        effective_target_group_path, target_path_source, target_selection_errors = _effective_target_group_path(
            manifest,
            use_intended_target=use_intended_target,
        )
        target_group_path = effective_target_group_path
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

        metadata_report = (
            artifact_run_metadata_report(run_group_dir)
            if run_group_dir.exists()
            else {
                "status": "fail",
                "errors": ["run_group directory is missing"],
            }
        )
        metadata_errors = (
            []
            if metadata_report["status"] == "pass"
            else list(metadata_report.get("errors") or ["artifact metadata invalid"])
        )
        _add_validation(
            validations,
            "artifact_run_metadata",
            metadata_errors,
            report=metadata_report,
        )
        errors.extend(
            f"artifact_run_metadata: {error}" for error in metadata_errors
        )

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
        run_family_relative_path = str(Path(target_group_path).parent) if target_group_path else run_family
        family_path = resolved_target / run_family_relative_path if resolved_target is not None and run_family_relative_path else None
        incoming_path = family_path / ".incoming" / run_name if family_path is not None and run_name else None
        failed_path_pattern = (
            str(family_path / ".failed" / f"{run_name}_{_utc_now_label()}")
            if family_path is not None and run_name
            else None
        )

        target_path_errors: list[str] = []
        target_path_errors.extend(target_selection_errors)
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
                            "artifact_run_metadata",
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
                        "would_update": False,
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
            "source_tarball_sha256": source_tarball_sha256,
            "manifest_sha256": manifest_sha256,
            "target_archive_path": str(resolved_target) if resolved_target is not None else None,
            "target_group_path": target_group_path or None,
            "manifest_target_group_path": manifest_target_group_path or None,
            "target_group_path_source": target_path_source,
            "use_intended_target": bool(use_intended_target),
            "run_family": run_family or None,
            "run_family_path": run_family_relative_path or None,
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
            "source_tarball_sha256": source_tarball_sha256,
            "manifest_sha256": manifest_sha256,
            "errors": [str(exc)],
            "validations": validations,
            "planned_actions": planned_actions,
        }
    finally:
        if temp_owner is not None:
            temp_owner.cleanup()
        staging_owner.cleanup()


def apply_import(
    *,
    tarball_path: Path,
    target_zarr: Optional[Path] = None,
    overwrite: bool = False,
    validate_source_inputs: bool = True,
    use_intended_target: bool = False,
) -> dict[str, Any]:
    """Apply an artifact import using .incoming promotion and receipt sidecar."""
    plan = build_import_plan(
        tarball_path=tarball_path,
        target_zarr=target_zarr,
        overwrite=overwrite,
        validate_source_inputs=validate_source_inputs,
        use_intended_target=use_intended_target,
    )
    if plan["status"] != "ok":
        plan["apply"] = True
        plan["applied"] = False
        return plan

    tarball_path = Path(plan["source_tarball"]).expanduser().resolve()
    target_archive = Path(str(plan["target_archive_path"]))
    run_family_path = str(plan.get("run_family_path") or plan["run_family"])
    run_name = str(plan["run_name"])
    latest_policy = str(plan["latest_policy"])
    incoming_path = Path(str(plan["incoming_path"]))
    final_path = Path(str(plan["final_path"]))
    receipt_path = Path(str(plan["receipt_path"]))
    family_path = target_archive / run_family_path
    failed_base = family_path / ".failed"
    failed_path: Optional[Path] = None
    failed_container: Optional[Path] = None
    transaction_label = _utc_now_label()
    previous_final_path = (
        family_path / ".incoming" / f".{run_name}_previous_{transaction_label}"
    )
    pending_receipt_path = (
        family_path
        / ".incoming"
        / f".{run_name}_import_receipt_{transaction_label}.json.pending"
    )
    previous_receipt_path = (
        family_path
        / ".incoming"
        / f".{run_name}_previous_receipt_{transaction_label}.json"
    )
    apply_validations: dict[str, dict[str, Any]] = {}
    apply_errors: list[str] = []
    latest_updated = False
    imported = False
    new_final_installed = False
    receipt_installed = False
    transaction_owned = False
    incoming_owned = False
    cleanup_warnings: list[str] = []
    had_final_before = final_path.exists()
    had_receipt_before = receipt_path.exists()
    previous_final_hash: Optional[str] = None
    previous_receipt_hash: Optional[str] = None

    temp_owner = tempfile.TemporaryDirectory(prefix="palette_run_group_apply_")
    try:
        if had_final_before:
            previous_final_hash = tree_hash(final_path)
        if had_receipt_before:
            if not receipt_path.is_file():
                raise ValueError(f"import receipt is not a regular file: {receipt_path}")
            previous_receipt_hash = _sha256_file(receipt_path)

        transaction_root = Path(temp_owner.name)
        staged_tarball = transaction_root / "source_snapshot.tar"
        staged_tarball_sha256 = _copy_file_snapshot(tarball_path, staged_tarball)
        if staged_tarball_sha256 != plan.get("source_tarball_sha256"):
            raise ValueError(
                "source tarball changed after import planning; refusing to apply "
                "a different artifact snapshot"
            )
        extract_root = transaction_root / "extracted"
        artifact_root = _safe_extract_artifact(staged_tarball, extract_root)
        manifest_path = artifact_root / "artifact_manifest.json"
        manifest_payload = _read_json_strict(manifest_path)
        manifest, manifest_shape_errors = _validate_manifest_shape(manifest_payload)
        if manifest_shape_errors or manifest is None:
            raise ValueError(
                "apply artifact manifest is invalid: "
                + "; ".join(manifest_shape_errors or ["manifest is not an object"])
            )
        apply_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if apply_manifest_sha256 != plan.get("manifest_sha256"):
            raise ValueError(
                "artifact manifest differs from the validated import plan"
            )
        run_group_dir = artifact_root / "run_group"
        expected_hash = manifest.get("checksums", {}).get("run_group_tree_hash")

        if plan.get("run_family") != RUN_FAMILY:
            raise ValueError(
                f"Artifact imports are restricted to {RUN_FAMILY!r}."
            )
        if latest_policy != DO_NOT_SET_LATEST:
            raise ValueError(
                "Detection artifact imports cannot update latest or authoritative "
                "selectors."
            )

        orphaned_paths = _orphaned_import_transaction_paths(family_path, run_name)
        if orphaned_paths:
            raise RuntimeError(
                "orphaned import transaction paths require explicit recovery: "
                + ", ".join(str(path) for path in orphaned_paths)
            )
        if previous_final_path.exists() or pending_receipt_path.exists() or previous_receipt_path.exists():
            raise FileExistsError(
                "import transaction path already exists; explicit recovery is "
                "required before retrying"
            )

        family_path.mkdir(parents=True, exist_ok=True)
        _stamp_artifact_family_metadata(family_path)
        (family_path / ".incoming").mkdir(exist_ok=True)
        failed_base.mkdir(exist_ok=True)

        if incoming_path.exists():
            raise FileExistsError(f"incoming path already exists: {incoming_path}")
        if final_path.exists():
            if not overwrite:
                raise FileExistsError(f"final target already exists: {final_path}")

        incoming_path.mkdir()
        incoming_owned = True
        # Acquiring the shared incoming path is the transaction lock for this
        # run name. Recheck generated paths after acquisition so a concurrent
        # loser cannot make us claim or mutate its recovery evidence.
        post_acquire_orphans = _orphaned_import_transaction_paths(family_path, run_name)
        if post_acquire_orphans:
            raise RuntimeError(
                "import transaction paths appeared while acquiring incoming: "
                + ", ".join(str(path) for path in post_acquire_orphans)
            )
        transaction_owned = True
        shutil.copytree(run_group_dir, incoming_path, dirs_exist_ok=True)

        strict_report = strict_json_report(incoming_path)
        strict_errors = [] if strict_report["status"] == "pass" else ["strict JSON validation failed"]
        _add_validation(apply_validations, "incoming_strict_json", strict_errors, report=strict_report)
        apply_errors.extend(f"incoming_strict_json: {error}" for error in strict_errors)

        arrays_report = required_arrays_report(incoming_path)
        array_errors = [] if arrays_report["status"] == "pass" else ["required array validation failed"]
        _add_validation(apply_validations, "incoming_required_arrays", array_errors, report=arrays_report)
        apply_errors.extend(f"incoming_required_arrays: {error}" for error in array_errors)

        metadata_report = artifact_run_metadata_report(incoming_path)
        metadata_errors = (
            []
            if metadata_report["status"] == "pass"
            else list(metadata_report.get("errors") or ["artifact metadata invalid"])
        )
        _add_validation(
            apply_validations,
            "incoming_artifact_run_metadata",
            metadata_errors,
            report=metadata_report,
        )
        apply_errors.extend(
            f"incoming_artifact_run_metadata: {error}"
            for error in metadata_errors
        )

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

        # Build and serialize the receipt before replacing any existing run.
        # Receipt promotion itself is the final commit point.
        receipt = {
            "schema_version": 1,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source_tarball": str(tarball_path),
            "source_tarball_sha256": staged_tarball_sha256,
            "target_archive_path": str(target_archive),
            "target_group_path": plan["target_group_path"],
            "manifest_target_group_path": plan.get("manifest_target_group_path"),
            "target_group_path_source": plan.get("target_group_path_source"),
            "use_intended_target": bool(plan.get("use_intended_target")),
            "run_family": plan["run_family"],
            "run_family_path": run_family_path,
            "run_name": run_name,
            "layout": plan["layout"],
            "final_path": str(final_path),
            "incoming_path": str(incoming_path),
            "latest_policy": latest_policy,
            "latest_updated": latest_updated,
            "manifest": manifest,
            "manifest_sha256": apply_manifest_sha256,
            "importer": {
                "command": " ".join(sys.argv),
                "git": get_git_info(),
            },
            "validations": {
                **plan["validations"],
                **apply_validations,
            },
        }
        _write_json(pending_receipt_path, receipt)

        if final_path.exists():
            _verified_atomic_move(final_path, previous_final_path)
        _verified_atomic_move(incoming_path, final_path)
        new_final_installed = True
        if receipt_path.exists():
            _verified_atomic_move(receipt_path, previous_receipt_path)
        _verified_atomic_move(pending_receipt_path, receipt_path)
        receipt_installed = True
        imported = True

        # Cleanup happens only after both final run and receipt are committed.
        # Failure to remove a recoverable backup is reported but does not roll
        # back an already complete transaction.
        for label, stale_path in (
            ("previous final", previous_final_path),
            ("previous receipt", previous_receipt_path),
        ):
            if not stale_path.exists():
                continue
            try:
                _verified_remove_path(stale_path)
            except Exception as cleanup_exc:  # pragma: no cover - hostile store
                cleanup_warnings.append(f"{label}: {cleanup_exc}")

        return {
            **plan,
            "status": "ok",
            "apply": True,
            "applied": True,
            "imported": True,
            "final_state": "committed",
            "latest_updated": latest_updated,
            "receipt_path": str(receipt_path),
            "cleanup_warnings": cleanup_warnings,
            "validations": {
                **plan["validations"],
                **apply_validations,
            },
            "errors": [],
        }
    except Exception as exc:
        rollback_errors: list[str] = []
        attempted_path: Optional[Path] = None
        if new_final_installed and final_path.exists():
            attempted_path = final_path
        elif incoming_owned and incoming_path.exists():
            attempted_path = incoming_path
        if attempted_path is not None:
            try:
                failed_base.mkdir(parents=True, exist_ok=True)
                failed_container = Path(
                    tempfile.mkdtemp(
                        prefix=f"{run_name}_{_utc_now_label()}_",
                        dir=failed_base,
                    )
                )
                failed_path = failed_container / "run_group"
                _verified_atomic_move(attempted_path, failed_path)
            except Exception as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(f"quarantine attempted import: {rollback_exc}")
        try:
            if transaction_owned and previous_final_path.exists():
                if final_path.exists():
                    raise RuntimeError("replacement final path is still occupied")
                _verified_atomic_move(previous_final_path, final_path)
        except Exception as rollback_exc:  # pragma: no cover - hostile store
            rollback_errors.append(f"restore previous final: {rollback_exc}")
        try:
            if transaction_owned and receipt_installed and receipt_path.exists():
                _verified_remove_path(receipt_path)
            if transaction_owned and previous_receipt_path.exists():
                if receipt_path.exists():
                    raise RuntimeError("replacement receipt path is still occupied")
                _verified_atomic_move(previous_receipt_path, receipt_path)
            if transaction_owned and pending_receipt_path.exists():
                _verified_remove_path(pending_receipt_path)
        except Exception as rollback_exc:  # pragma: no cover - hostile store
            rollback_errors.append(f"restore receipt: {rollback_exc}")

        # Do not claim a clean failure unless the externally visible run and
        # receipt are exactly the pre-transaction versions and no transaction
        # paths remain. Ambiguous state must fail closed for manual recovery.
        state_errors: list[str] = []
        if incoming_owned and incoming_path.exists():
            state_errors.append(f"incoming path remains: {incoming_path}")
        if transaction_owned and pending_receipt_path.exists():
            state_errors.append(f"pending receipt remains: {pending_receipt_path}")
        if transaction_owned and previous_final_path.exists():
            state_errors.append(f"previous final backup remains: {previous_final_path}")
        if transaction_owned and previous_receipt_path.exists():
            state_errors.append(f"previous receipt backup remains: {previous_receipt_path}")
        try:
            if had_final_before:
                if not final_path.exists():
                    state_errors.append("previous final is missing")
                elif previous_final_hash is not None and tree_hash(final_path) != previous_final_hash:
                    state_errors.append("final path does not match the previous run")
            elif final_path.exists():
                state_errors.append("replacement run remains at the final path")
        except Exception as state_exc:  # pragma: no cover - hostile store
            state_errors.append(f"could not verify restored final: {state_exc}")
        try:
            if had_receipt_before:
                if not receipt_path.is_file():
                    state_errors.append("previous import receipt is missing")
                elif (
                    previous_receipt_hash is not None
                    and _sha256_file(receipt_path) != previous_receipt_hash
                ):
                    state_errors.append("import receipt does not match the previous receipt")
            elif receipt_path.exists():
                state_errors.append("replacement import receipt remains")
        except Exception as state_exc:  # pragma: no cover - hostile store
            state_errors.append(f"could not verify restored receipt: {state_exc}")

        rollback_errors.extend(f"state verification: {error}" for error in state_errors)
        rollback_complete = not rollback_errors
        imported = False if rollback_complete else None
        return {
            **plan,
            "status": "failed" if rollback_complete else "rollback_incomplete",
            "apply": True,
            "applied": False if rollback_complete else None,
            "imported": imported,
            "final_state": (
                "restored_pre_transaction_state"
                if rollback_complete
                else "ambiguous_manual_recovery_required"
            ),
            "failed_path": (
                str(failed_path)
                if failed_path is not None and failed_path.exists()
                else None
            ),
            "recovery_paths": {
                "final": str(final_path) if final_path.exists() else None,
                "incoming": str(incoming_path) if incoming_path.exists() else None,
                "previous_final": (
                    str(previous_final_path) if previous_final_path.exists() else None
                ),
                "receipt": str(receipt_path) if receipt_path.exists() else None,
                "pending_receipt": (
                    str(pending_receipt_path) if pending_receipt_path.exists() else None
                ),
                "previous_receipt": (
                    str(previous_receipt_path) if previous_receipt_path.exists() else None
                ),
                "quarantine_container": (
                    str(failed_container)
                    if failed_container is not None and failed_container.exists()
                    else None
                ),
            },
            "rollback_errors": rollback_errors,
            "validations": {
                **plan.get("validations", {}),
                **apply_validations,
            },
            "errors": [
                *plan.get("errors", []),
                str(exc),
                *(f"rollback: {error}" for error in rollback_errors),
            ],
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
        "--use-intended-target",
        action="store_true",
        help=(
            "Import to artifact_manifest.json intended_target_group_path instead of "
            "target_group_path. Use for clip-camera artifacts that should land under "
            "clips/<clip>/cameras/<serial>/<run_family>/<run>."
        ),
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
            use_intended_target=args.use_intended_target,
        )
    else:
        result = build_import_plan(
            tarball_path=args.tarball,
            target_zarr=args.target_zarr,
            overwrite=args.overwrite,
            validate_source_inputs=not args.skip_source_input_checks,
            keep_extracted=args.keep_extracted,
            use_intended_target=args.use_intended_target,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
