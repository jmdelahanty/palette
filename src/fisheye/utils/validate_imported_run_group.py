"""Validate a run group imported from a Palette transfer artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

from fisheye.utils.import_run_group_artifact import SUPPORTED_LAYOUTS
from fisheye.utils.run_detection_artifact import required_arrays_report, strict_json_report, tree_hash


MUTABLE_CHILDREN_BY_LAYOUT = {
    # Existing Palette detect-quality reports are appended under the raw detect
    # run. The imported model-output core remains fingerprinted separately.
    "detect_yolo_sparse_v1": ("quality_reports",),
}


def _read_json_strict(path: Path) -> Any:
    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash_excluding_children(root: Path, excluded_children: Sequence[str]) -> str:
    excluded = {name.strip("/") for name in excluded_children if name.strip("/")}
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel_path = path.relative_to(root)
        if rel_path.parts and rel_path.parts[0] in excluded:
            continue
        rel = rel_path.as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _nested_get(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and not math.isfinite(value))
    if isinstance(value, (dict, list, tuple)):
        return bool(value)
    return True


def _validation(status: str, **details: Any) -> dict[str, Any]:
    return {"status": status, **details}


def _run_group_attrs(run_group_path: Path) -> dict[str, Any]:
    zarr_json = run_group_path / "zarr.json"
    payload = _read_json_strict(zarr_json)
    if not isinstance(payload, dict):
        raise ValueError(f"{zarr_json} must be a JSON object")
    attrs = payload.get("attributes")
    if not isinstance(attrs, dict):
        raise ValueError(f"{zarr_json} attributes must be a JSON object")
    return attrs


def _resolve_target_group(
    *,
    zarr_path: Path,
    target_group_path: Optional[str],
    run_family: Optional[str],
    run_name: Optional[str],
) -> tuple[str, str, str, str, Path]:
    if target_group_path:
        parts = Path(target_group_path).parts
        if (
            len(parts) < 2
            or any(part in {"", ".", ".."} for part in parts)
            or Path(target_group_path).is_absolute()
        ):
            raise ValueError(
                "--target-group-path must be a relative path ending in '<run_family>/<run_name>'"
            )
        family = parts[-2]
        family_path = str(Path(*parts[:-1]))
        name = parts[-1]
        return target_group_path, family, family_path, name, zarr_path / target_group_path
    if not run_family or not run_name:
        raise ValueError("provide --target-group-path or both --run-family and --run-name")
    target = f"{run_family}/{run_name}"
    return target, run_family, run_family, run_name, zarr_path / run_family / run_name


def _default_receipt_path(zarr_path: Path, run_family_path: str, run_name: str) -> Path:
    return zarr_path / run_family_path / ".imports" / f"{run_name}_import_receipt.json"


def _detect_provenance_report(attrs: dict[str, Any], manifest: Optional[dict[str, Any]]) -> dict[str, Any]:
    provenance = attrs.get("provenance") if isinstance(attrs.get("provenance"), dict) else {}
    checks = {
        "git": [
            "provenance.git.commit",
            "git_commit",
        ],
        "command": [
            "provenance.command",
        ],
        "host": [
            "provenance.platform.hostname",
            "hostname",
        ],
        "model": [
            "provenance.artifacts.model_path",
            "model_path",
        ],
        "gpu_or_device": [
            "provenance.artifacts.device",
            "provenance.environment.cuda_support",
        ],
        "timing": [
            "provenance.timing.processing_seconds_total",
            "timing_summary.processing_seconds_total",
        ],
        "decode_backend": [
            "provenance.parameters.decode_backend_effective",
            "decode_backend_effective",
            "timing_summary.decode_backend_effective",
        ],
    }
    source = {"provenance": provenance, **attrs}
    missing: list[str] = []
    resolved: dict[str, Any] = {}
    for name, candidate_paths in checks.items():
        value = None
        for candidate in candidate_paths:
            value = _nested_get(source, candidate) if "." in candidate else source.get(candidate)
            if _is_non_empty(value):
                break
        if _is_non_empty(value):
            resolved[name] = value
        else:
            missing.append(name)

    manifest_cluster = None
    if isinstance(manifest, dict):
        manifest_provenance = manifest.get("provenance")
        if isinstance(manifest_provenance, dict):
            manifest_cluster = manifest_provenance.get("cluster")
            if "cluster_job_id" not in resolved:
                job_id = _nested_get(manifest_provenance, "cluster.LSB_JOBID")
                if _is_non_empty(job_id):
                    resolved["cluster_job_id"] = job_id
            if "cuda_visible_devices" not in resolved:
                cuda_visible = _nested_get(manifest_provenance, "cluster.CUDA_VISIBLE_DEVICES")
                if _is_non_empty(cuda_visible):
                    resolved["cuda_visible_devices"] = cuda_visible

    return {
        "status": "pass" if not missing else "fail",
        "missing": missing,
        "resolved": resolved,
        "manifest_cluster_present": isinstance(manifest_cluster, dict),
    }


def validate_imported_run_group(
    *,
    zarr_path: Path,
    target_group_path: Optional[str] = None,
    run_family: Optional[str] = None,
    run_name: Optional[str] = None,
    receipt_path: Optional[Path] = None,
    layout: Optional[str] = None,
    validate_source_inputs: bool = True,
    validate_source_tarball: bool = True,
    require_provenance: bool = True,
    require_promoted: bool = True,
) -> dict[str, Any]:
    zarr_path = zarr_path.expanduser().resolve()
    errors: list[str] = []
    validations: dict[str, dict[str, Any]] = {}
    try:
        target_group, family, family_path, name, final_path = _resolve_target_group(
            zarr_path=zarr_path,
            target_group_path=target_group_path,
            run_family=run_family,
            run_name=run_name,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "errors": [str(exc)],
            "validations": {},
        }

    receipt_path = (
        receipt_path.expanduser().resolve()
        if receipt_path is not None
        else _default_receipt_path(zarr_path, family_path, name)
    )
    incoming_path = zarr_path / family_path / ".incoming" / name
    manifest: Optional[dict[str, Any]] = None

    if final_path.exists() and final_path.is_dir():
        validations["final_path"] = _validation("pass", path=str(final_path))
    else:
        msg = f"final run group not found: {final_path}"
        validations["final_path"] = _validation("fail", path=str(final_path), errors=[msg])
        errors.append(msg)

    if require_promoted and incoming_path.exists():
        msg = f"incoming path still exists: {incoming_path}"
        validations["promotion"] = _validation("fail", incoming_path=str(incoming_path), errors=[msg])
        errors.append(msg)
    else:
        validations["promotion"] = _validation(
            "pass",
            incoming_path=str(incoming_path),
            incoming_exists=bool(incoming_path.exists()),
        )

    if receipt_path.exists() and receipt_path.is_file():
        try:
            receipt = _read_json_strict(receipt_path)
            if not isinstance(receipt, dict):
                raise ValueError("receipt must be a JSON object")
            validations["receipt"] = _validation("pass", path=str(receipt_path))
            manifest_value = receipt.get("manifest")
            if isinstance(manifest_value, dict):
                manifest = manifest_value
            else:
                msg = "receipt manifest is missing or not an object"
                validations["receipt_manifest"] = _validation("fail", errors=[msg])
                errors.append(msg)

            receipt_mismatches: list[str] = []
            expected_pairs = {
                "target_archive_path": str(zarr_path),
                "target_group_path": target_group,
                "run_family": family,
                "run_name": name,
                "final_path": str(final_path),
            }
            if receipt.get("run_family_path") is not None or family_path != family:
                expected_pairs["run_family_path"] = family_path
            for key, expected in expected_pairs.items():
                observed = receipt.get(key)
                if observed != expected:
                    receipt_mismatches.append(f"{key}: expected {expected!r}, got {observed!r}")
            validations["receipt_identity"] = _validation(
                "pass" if not receipt_mismatches else "fail",
                expected=expected_pairs,
                errors=receipt_mismatches,
            )
            errors.extend(f"receipt_identity: {item}" for item in receipt_mismatches)

            if validate_source_tarball:
                tarball = receipt.get("source_tarball")
                expected_sha = receipt.get("source_tarball_sha256")
                tarball_errors: list[str] = []
                actual_sha = None
                if not isinstance(tarball, str) or not tarball:
                    tarball_errors.append("receipt source_tarball is missing")
                else:
                    tarball_path = Path(tarball).expanduser()
                    if not tarball_path.is_file():
                        tarball_errors.append(f"source tarball not found: {tarball_path}")
                    else:
                        actual_sha = _sha256_file(tarball_path)
                        if isinstance(expected_sha, str) and expected_sha and actual_sha != expected_sha:
                            tarball_errors.append("source tarball sha256 mismatch")
                validations["source_tarball"] = _validation(
                    "pass" if not tarball_errors else "fail",
                    source_tarball=tarball,
                    expected_sha256=expected_sha,
                    actual_sha256=actual_sha,
                    errors=tarball_errors,
                )
                errors.extend(f"source_tarball: {item}" for item in tarball_errors)
        except Exception as exc:
            msg = f"receipt is not strict valid JSON: {exc}"
            validations["receipt"] = _validation("fail", path=str(receipt_path), errors=[msg])
            errors.append(msg)
    else:
        msg = f"receipt not found: {receipt_path}"
        validations["receipt"] = _validation("fail", path=str(receipt_path), errors=[msg])
        errors.append(msg)

    observed_layout = layout or (manifest.get("layout") if isinstance(manifest, dict) else None)
    if observed_layout not in SUPPORTED_LAYOUTS:
        msg = f"unsupported or missing layout: {observed_layout!r}"
        validations["layout"] = _validation("fail", layout=observed_layout, errors=[msg])
        errors.append(msg)
    else:
        validations["layout"] = _validation("pass", layout=observed_layout)

    if final_path.exists():
        strict_report = strict_json_report(final_path)
        if strict_report["status"] == "pass":
            validations["strict_json"] = _validation("pass", report=strict_report)
        else:
            msg = "strict JSON validation failed"
            validations["strict_json"] = _validation("fail", report=strict_report, errors=[msg])
            errors.append(msg)

        arrays_report = required_arrays_report(final_path)
        if arrays_report["status"] == "pass":
            validations["required_arrays"] = _validation("pass", report=arrays_report)
        else:
            msg = "required array validation failed"
            validations["required_arrays"] = _validation("fail", report=arrays_report, errors=[msg])
            errors.append(msg)

        actual_hash = tree_hash(final_path)
        excluded_children = MUTABLE_CHILDREN_BY_LAYOUT.get(str(observed_layout), ())
        core_actual_hash = (
            _tree_hash_excluding_children(final_path, excluded_children)
            if excluded_children
            else actual_hash
        )
        expected_hash = None
        if isinstance(manifest, dict):
            expected_hash = _nested_get(manifest, "checksums.run_group_tree_hash")
        hash_errors: list[str] = []
        hash_mode = "exact"
        if not isinstance(expected_hash, str) or not expected_hash:
            hash_errors.append("manifest checksums.run_group_tree_hash is missing")
        elif actual_hash == expected_hash:
            hash_mode = "exact"
        elif core_actual_hash == expected_hash:
            hash_mode = "core_excluding_mutable_children"
        else:
            hash_errors.append("final run_group_tree_hash mismatch")
        validations["run_group_tree_hash"] = _validation(
            "pass" if not hash_errors else "fail",
            expected=expected_hash,
            actual=actual_hash if hash_mode == "exact" else core_actual_hash,
            full_actual=actual_hash,
            core_actual=core_actual_hash,
            hash_mode=hash_mode,
            excluded_children=list(excluded_children),
            errors=hash_errors,
        )
        errors.extend(f"run_group_tree_hash: {item}" for item in hash_errors)

        if require_provenance:
            try:
                attrs = _run_group_attrs(final_path)
                provenance_report = _detect_provenance_report(attrs, manifest)
                validations["provenance"] = provenance_report
                if provenance_report["status"] != "pass":
                    errors.append(f"provenance missing: {', '.join(provenance_report['missing'])}")
            except Exception as exc:
                msg = f"failed to read run-group provenance: {exc}"
                validations["provenance"] = _validation("fail", errors=[msg])
                errors.append(msg)

    if validate_source_inputs and isinstance(manifest, dict):
        source_errors: list[str] = []
        checked: list[dict[str, Any]] = []
        source_inputs = manifest.get("source_inputs")
        if not isinstance(source_inputs, list):
            source_errors.append("manifest source_inputs is missing or not a list")
        else:
            for item in source_inputs:
                if not isinstance(item, dict):
                    source_errors.append("source_inputs contains a non-object item")
                    continue
                path_value = item.get("path")
                role = item.get("role")
                if not isinstance(path_value, str) or not path_value:
                    source_errors.append(f"source input {role!r} has no path")
                    continue
                source_path = Path(path_value).expanduser()
                exists = source_path.exists()
                checked.append({"role": role, "path": str(source_path), "exists": exists})
                if not exists:
                    source_errors.append(f"source input does not exist: {source_path}")
        validations["source_inputs"] = _validation(
            "pass" if not source_errors else "fail",
            checked=checked,
            errors=source_errors,
        )
        errors.extend(f"source_inputs: {item}" for item in source_errors)

    return {
        "status": "ok" if not errors else "failed",
        "target_archive_path": str(zarr_path),
        "target_group_path": target_group,
        "run_family": family,
        "run_family_path": family_path,
        "run_name": name,
        "final_path": str(final_path),
        "receipt_path": str(receipt_path),
        "layout": observed_layout,
        "validations": validations,
        "errors": errors,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a Palette run group after artifact import."
    )
    parser.add_argument("zarr_path", type=Path, help="Target analysis Zarr path")
    parser.add_argument(
        "--target-group-path",
        help="Imported target group path, e.g. detect_runs/detect_...",
    )
    parser.add_argument("--run-family", help="Run family when --target-group-path is not used")
    parser.add_argument("--run-name", help="Run name when --target-group-path is not used")
    parser.add_argument("--receipt", type=Path, default=None, help="Override import receipt path")
    parser.add_argument("--layout", choices=sorted(SUPPORTED_LAYOUTS), default=None)
    parser.add_argument("--skip-source-input-checks", action="store_true")
    parser.add_argument("--skip-source-tarball-check", action="store_true")
    parser.add_argument("--no-require-provenance", action="store_true")
    parser.add_argument("--allow-incoming", action="store_true", help="Do not fail if .incoming/<run> exists")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = validate_imported_run_group(
        zarr_path=args.zarr_path,
        target_group_path=args.target_group_path,
        run_family=args.run_family,
        run_name=args.run_name,
        receipt_path=args.receipt,
        layout=args.layout,
        validate_source_inputs=not args.skip_source_input_checks,
        validate_source_tarball=not args.skip_source_tarball_check,
        require_provenance=not args.no_require_provenance,
        require_promoted=not args.allow_incoming,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
