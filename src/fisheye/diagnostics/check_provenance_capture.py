#!/usr/bin/env python3
"""Check whether pipeline runs captured provenance metadata."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import zarr

from ..shared.stage_provenance import (
    STAGE_PROVENANCE_CONTRACT_NAME,
    get_stage_git,
)


@dataclass
class ProvenanceCheck:
    stage: str
    run: Optional[str]
    status: str
    has_provenance: bool
    missing_required: List[str]
    missing_optional: List[str]


STAGES = [
    {"label": "detect", "groups": ["detect_runs"]},
    {"label": "refined_detect", "groups": ["refined_detect_runs", "refined_runs"]},
    {"label": "crop", "groups": ["crop_runs"]},
    {"label": "keypoints", "groups": ["keypoints_runs"]},
    {"label": "refined_keypoints", "groups": ["refined_keypoints_runs", "keypoints_refined_runs"]},
    {"label": "eye_masks", "groups": ["eye_masks_runs"]},
    {"label": "refined_eye_masks", "groups": ["refined_eye_masks_runs"]},
    {"label": "subject_masks", "groups": ["subject_mask_runs"]},
    {"label": "refined_subject_masks", "groups": ["refined_subject_masks_runs"]},
    {"label": "arena_assignment", "groups": ["arena_assignment_runs"]},
]

# Strict contract gating is stage-string aware so legacy/non-migrated payloads
# remain compatible during backfill windows.
STRICT_CONTRACT_STAGE_ALIASES: Dict[str, set[str]] = {
    "detect": {"detect"},
    "refined_detect": {"refined_detect", "refine_detect"},
    "crop": {"crop"},
    "keypoints": {"keypoints", "keypoints_detect"},
    "refined_keypoints": {"refined_keypoints", "refine_keypoints"},
    "eye_masks": {"eye_masks"},
    "refined_eye_masks": {"refined_eye_masks", "refine_eye_masks"},
    "subject_masks": {"subject_masks"},
    "refined_subject_masks": {"refined_subject_masks", "refine_subject_masks"},
    "arena_assignment": {"arena_assignment"},
}


REQUIRED_FIELDS = ["timestamp", "parameters", "inputs"]
OPTIONAL_FIELDS = ["git", "environment"]


def _iter_zarr(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_dir() and path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> str:
    for key in ("zarr_use", "zarr_purpose"):
        purpose = root.attrs.get(key)
        if purpose is None:
            continue
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def _read_file_list(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _latest_run(group: zarr.Group) -> Optional[str]:
    latest = group.attrs.get("latest")
    if isinstance(latest, bytes):
        latest = latest.decode("utf-8", "ignore")
    if isinstance(latest, str) and latest in group:
        return latest
    names = list(group.group_keys()) if hasattr(group, "group_keys") else list(group.keys())
    return sorted(names)[-1] if names else None


def _has_timestamp(prov: Optional[dict], attrs: dict) -> bool:
    if isinstance(prov, dict):
        for key in ("created_at_utc", "created_utc", "timestamp_utc"):
            if prov.get(key):
                return True
    for key in attrs.keys():
        if key.endswith("_timestamp_utc"):
            return True
    for key in ("created_at_utc", "created_utc", "timestamp_utc"):
        if attrs.get(key):
            return True
    return False


def _has_parameters(prov: Optional[dict], attrs: dict) -> bool:
    if isinstance(prov, dict) and prov.get("parameters") is not None:
        return True
    return "parameters" in attrs


def _has_inputs(prov: Optional[dict], attrs: dict) -> bool:
    if isinstance(prov, dict):
        inputs = prov.get("inputs")
        if isinstance(inputs, dict) and inputs:
            return True
    return any(str(key).startswith("source_") for key in attrs.keys())


def _has_git(prov: Optional[dict], attrs: dict) -> bool:
    git_payload = get_stage_git(attrs)
    return any(
        value not in (None, "", "unknown", "N/A", "n/a")
        for value in (
            git_payload.get("commit"),
            git_payload.get("short"),
            git_payload.get("branch"),
        )
    )


def _has_environment(prov: Optional[dict], attrs: dict) -> bool:
    if isinstance(prov, dict):
        if prov.get("environment") is not None or prov.get("platform") is not None:
            return True
    return any(key in attrs for key in ("hostname", "system_hostname"))


FIELD_CHECKS = {
    "timestamp": _has_timestamp,
    "parameters": _has_parameters,
    "inputs": _has_inputs,
    "git": _has_git,
    "environment": _has_environment,
}


def _json_loads(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _normalize_stage_name(raw: Any) -> Optional[str]:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if not isinstance(raw, str):
        return None
    normalized = raw.strip()
    return normalized if normalized else None


def _resolved_stage_name(prov: Optional[Dict[str, Any]], attrs: Dict[str, Any]) -> Optional[str]:
    if isinstance(prov, dict):
        stage_name = _normalize_stage_name(prov.get("stage"))
        if stage_name:
            return stage_name
    return _normalize_stage_name(attrs.get("stage"))


def _should_enforce_strict_contract(stage_label: str, prov: Optional[Dict[str, Any]], attrs: Dict[str, Any]) -> bool:
    accepted = STRICT_CONTRACT_STAGE_ALIASES.get(stage_label)
    if not accepted:
        return False
    stage_name = _resolved_stage_name(prov, attrs)
    if stage_name is None:
        return False
    return stage_name in accepted


def _append_unique(target: List[str], field: str) -> None:
    if field not in target:
        target.append(field)


def _collect_missing_contract_fields(prov: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(prov, dict):
        return ["contract"]

    contract_raw = prov.get("contract")
    if not isinstance(contract_raw, dict):
        return ["contract"]

    missing: List[str] = []

    name = contract_raw.get("name")
    if not isinstance(name, str) or not name.strip() or name != STAGE_PROVENANCE_CONTRACT_NAME:
        missing.append("contract.name")

    try:
        version = int(contract_raw.get("version"))
    except (TypeError, ValueError):
        missing.append("contract.version")
    else:
        if version < 1:
            missing.append("contract.version")

    return missing


_FISH_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _check_subject_metadata(root: zarr.Group) -> Dict[str, Any]:
    analysis = root.get("analysis_metadata")
    payload = None
    if analysis is not None:
        payload = _json_loads(analysis.attrs.get("subject_metadata"))

    if not isinstance(payload, dict):
        return {"status": "missing", "missing": ["subject_metadata"]}

    missing: List[str] = []
    warnings: List[str] = []
    if not payload.get("fish_id"):
        missing.append("fish_id")
    else:
        fish_id = str(payload.get("fish_id"))
        if not _FISH_ID_RE.match(fish_id):
            warnings.append("fish_id_format")
    if payload.get("subject_count") is None:
        missing.append("subject_count")
    else:
        subject_count = payload.get("subject_count")
        try:
            subject_count_int = int(subject_count)
        except (TypeError, ValueError):
            warnings.append("subject_count_not_int")
        else:
            if subject_count_int < 1:
                warnings.append("subject_count_lt_1")
    subject_type = payload.get("subject_type")
    if subject_type is not None and payload.get("subject_count") is not None:
        try:
            subject_count_int = int(payload.get("subject_count"))
        except (TypeError, ValueError):
            subject_count_int = None
        if subject_count_int == 1 and str(subject_type).lower() not in {"individual", "single"}:
            warnings.append("subject_type_mismatch_single")
        if subject_count_int and subject_count_int > 1 and str(subject_type).lower() in {"individual", "single"}:
            warnings.append("subject_type_mismatch_group")

    if missing:
        status = "partial"
    elif warnings:
        status = "warn"
    else:
        status = "ok"
    return {"status": status, "missing": missing, "warnings": warnings}


def _check_group(
    stage: str,
    group: Optional[zarr.Group],
    *,
    all_runs: bool,
    require_provenance: bool,
    strict_contract: bool,
) -> List[ProvenanceCheck]:
    if group is None:
        return [ProvenanceCheck(stage, None, "absent", False, REQUIRED_FIELDS.copy(), OPTIONAL_FIELDS.copy())]

    run_names = list(group.group_keys()) if hasattr(group, "group_keys") else list(group.keys())
    if not run_names:
        return [ProvenanceCheck(stage, None, "absent", False, REQUIRED_FIELDS.copy(), OPTIONAL_FIELDS.copy())]

    runs = run_names if all_runs else [name for name in [group.attrs.get("latest")] if name]  # type: ignore[list-item]
    if not all_runs:
        run = group.attrs.get("latest")
        runs = [run] if run in group else ([sorted(run_names)[-1]] if run_names else [])

    checks: List[ProvenanceCheck] = []
    for run_name in runs:
        if run_name not in group:
            continue
        run_group = group[run_name]
        attrs = dict(run_group.attrs)
        prov = _json_loads(attrs.get("provenance"))
        has_prov = isinstance(prov, dict)

        missing_required: List[str] = []
        missing_optional: List[str] = []

        for field in REQUIRED_FIELDS:
            ok = FIELD_CHECKS[field](prov, attrs)
            if not ok:
                missing_required.append(field)
        for field in OPTIONAL_FIELDS:
            ok = FIELD_CHECKS[field](prov, attrs)
            if not ok:
                missing_optional.append(field)

        if require_provenance and not has_prov:
            if "provenance" not in missing_required:
                missing_required.insert(0, "provenance")

        if strict_contract and _should_enforce_strict_contract(stage, prov, attrs):
            for field in _collect_missing_contract_fields(prov):
                _append_unique(missing_required, field)

        if missing_required:
            status = "missing"
        elif missing_optional:
            status = "partial"
        else:
            status = "ok"

        checks.append(ProvenanceCheck(stage, run_name, status, has_prov, missing_required, missing_optional))
    return checks


def _check_zarr(
    zarr_path: Path,
    *,
    root: Optional[zarr.Group] = None,
    all_runs: bool,
    require_provenance: bool,
    check_consistency: bool,
    check_subject_metadata: bool,
    strict_contract: bool,
) -> tuple[Dict[str, Any], Optional[list[str]], Optional[Dict[str, object]]]:
    zarr_root = root or zarr.open(str(zarr_path), mode="r")
    results: Dict[str, Any] = {}
    for stage in STAGES:
        group = None
        for group_name in stage["groups"]:
            if group_name in zarr_root:
                group = zarr_root[group_name]
                break
        results[stage["label"]] = _check_group(
            stage["label"],
            group,
            all_runs=all_runs,
            require_provenance=require_provenance,
            strict_contract=strict_contract,
        )
    issues: Optional[list[str]] = None
    lineage: Optional[Dict[str, object]] = None
    if check_consistency:
        from .check_provenance_consistency import collect_provenance

        record = collect_provenance(zarr_root)
        issues = record.issues
        lineage = {
            "source": record.crop_source_path,
            "detections": record.crop_source_rows,
            "crops": record.crop_rois,
            "keypoints": record.keypoint_rows,
        }
    if check_subject_metadata:
        results["subject_metadata"] = _check_subject_metadata(zarr_root)  # type: ignore[assignment]
    return results, issues, lineage


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or Zarr paths to scan.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for Zarrs under each root.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Check every run in each group (default: latest only).",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="Require a provenance dict in each run (not just fallback attrs).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of a human-readable summary.",
    )
    parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="Also run the provenance consistency checks per recording (default).",
    )
    parser.add_argument(
        "--no-check-consistency",
        action="store_true",
        help="Skip provenance consistency checks.",
    )
    parser.add_argument(
        "--check-subject-metadata",
        action="store_true",
        help="Report whether subject_metadata includes fish_id + subject_count.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Return non-zero when required provenance checks fail. "
            "Also enforces migrated offline stage contract name/version."
        ),
    )
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by purpose (default: any).",
    )

    args = parser.parse_args(argv)

    roots: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            roots.extend(_read_file_list(path))
    roots.extend(args.paths)
    if not roots:
        roots = [Path("/nvme1/recordings")]

    check_consistency = True if args.check_consistency else not args.no_check_consistency

    results_payload: Dict[str, Dict[str, Any]] = {}
    has_failures = False
    for zarr_path in _iter_zarr(roots, args.recursive):
        root = zarr.open(str(zarr_path), mode="r")
        if args.zarr_use != "any":
            observed = _infer_zarr_use(root, zarr_path)
            if observed != args.zarr_use:
                continue
        checks, issues, lineage = _check_zarr(
            zarr_path,
            root=root,
            all_runs=args.all_runs,
            require_provenance=args.require_provenance,
            check_consistency=check_consistency,
            check_subject_metadata=args.check_subject_metadata,
            strict_contract=args.strict,
        )
        for stage, checks_list in checks.items():
            if stage == "subject_metadata":
                continue
            if any(check.status == "missing" for check in checks_list):
                has_failures = True
        if check_consistency and issues:
            has_failures = True
        payload: Dict[str, Any] = {
            stage: [
                {
                    "stage": check.stage,
                    "run": check.run,
                    "status": check.status,
                    "has_provenance": check.has_provenance,
                    "missing_required": check.missing_required,
                    "missing_optional": check.missing_optional,
                }
                for check in checks_list
            ]
            for stage, checks_list in checks.items()
            if stage != "subject_metadata"
        }
        if args.check_subject_metadata and "subject_metadata" in checks:
            payload["subject_metadata"] = checks["subject_metadata"]
        if check_consistency:
            payload["consistency"] = {
                "ok": not issues,
                "issues": issues or [],
                "lineage": lineage or {},
            }
        results_payload[str(zarr_path)] = payload

    if args.json:
        print(json.dumps(results_payload, indent=2, sort_keys=True))
        return 1 if args.strict and has_failures else 0

    for zarr_path, stages in results_payload.items():
        print(f"\n{zarr_path}")
        for stage, checks_list in stages.items():
            if stage == "consistency":
                continue
            if stage == "subject_metadata":
                status = checks_list.get("status") if isinstance(checks_list, dict) else "missing"
                missing = ""
                if isinstance(checks_list, dict) and checks_list.get("missing"):
                    missing = f" missing={','.join(checks_list['missing'])}"
                warnings = ""
                if isinstance(checks_list, dict) and checks_list.get("warnings"):
                    warnings = f" warnings={','.join(checks_list['warnings'])}"
                print(f"  subject_metadata {'—':32s} {status:8s} no-prov{missing}{warnings}")
                continue
            for check in checks_list:
                run = check["run"] or "—"
                status = check["status"]
                prov = "prov" if check["has_provenance"] else "no-prov"
                missing = ""
                if check["missing_required"]:
                    missing = f" missing={','.join(check['missing_required'])}"
                elif check["missing_optional"]:
                    missing = f" optional={','.join(check['missing_optional'])}"
                print(f"  {stage:16s} {run:32s} {status:8s} {prov}{missing}")
        if "consistency" in stages:
            consistency = stages["consistency"]
            issues = consistency["issues"]
            if issues:
                print(f"  consistency      issues={len(issues)}")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print("  consistency      ok")
            lineage = consistency.get("lineage") or {}
            if lineage:
                source = lineage.get("source") or "—"
                det = lineage.get("detections")
                det_label = str(det) if det is not None else "—"
                crop = lineage.get("crops")
                crop_label = str(crop) if crop is not None else "—"
                kpt = lineage.get("keypoints")
                kpt_label = str(kpt) if kpt is not None else "—"
                print(f"  lineage          source={source} det={det_label} crop={crop_label} kpt={kpt_label}")

    return 1 if args.strict and has_failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
