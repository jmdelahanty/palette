#!/usr/bin/env python3
"""Backfill offline run provenance contract/git commit fields.

Default mode is dry-run. Use --apply to write changes.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import zarr

from fisheye.shared.stage_provenance import (
    STAGE_PROVENANCE_CONTRACT_NAME,
    STAGE_PROVENANCE_CONTRACT_VERSION,
    get_stage_git,
)


OFFLINE_RUN_GROUPS = (
    "detect_runs",
    "refined_detect_runs",
    "refined_runs",
    "crop_runs",
    "keypoints_runs",
    "refined_keypoints_runs",
    "keypoints_refined_runs",
    "eye_masks_runs",
    "refined_eye_masks_runs",
    "id_assignment_runs",
)
MISSING_STRING_VALUES = {"", "unknown", "n/a", "na", "none", "null"}
CROP_PARAMETER_ATTR_KEYS = (
    "roi_size",
    "parameter_source",
    "acceleration",
    "scheduler",
    "num_workers",
    "use_distributed",
    "write_backend",
    "write_backend_requested",
    "write_backend_effective",
    "roi_storage",
    "roi_chunk_len",
    "roi_use_sharding",
    "roi_shard_len",
)


@dataclass(frozen=True)
class RunBackfillPlan:
    display_path: str
    missing_contract: bool
    missing_git_commit: bool
    contract_update: Optional[dict[str, Any]]
    git_commit_update: Optional[str]
    parameters_update: Optional[dict[str, Any]]

    @property
    def would_modify(self) -> bool:
        return (
            self.contract_update is not None
            or self.git_commit_update is not None
            or self.parameters_update is not None
        )


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        candidates: list[Path] = []
        if root.suffix == ".zarr" and (root.is_dir() or root.is_file()):
            candidates = [root]
        elif root.exists():
            if recursive:
                candidates = sorted(root.rglob("*.zarr"))
            else:
                candidates = sorted(root.glob("*.zarr")) + sorted(root.glob("*/zarr/*.zarr"))
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    return value


def _is_missing(value: Any) -> bool:
    value = _normalize_scalar(value)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_STRING_VALUES
    return False


def _as_mapping(value: Any) -> dict[str, Any]:
    value = _normalize_scalar(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> str:
    for key in ("zarr_use", "zarr_purpose"):
        purpose = root.attrs.get(key)
        if purpose is not None:
            value = str(purpose).strip().lower()
            if value in {"analysis", "training"}:
                return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def _iter_runs(root: zarr.Group) -> Iterable[tuple[str, str, zarr.Group]]:
    for parent_name in OFFLINE_RUN_GROUPS:
        parent = root.get(parent_name)
        if parent is None:
            continue
        try:
            run_names = sorted(list(parent.group_keys()))
        except Exception:
            run_names = sorted(list(parent.keys()))
        for run_name in run_names:
            if run_name not in parent:
                continue
            child = parent[run_name]
            if not hasattr(child, "attrs"):
                continue
            yield parent_name, str(run_name), child


def _is_valid_contract(raw_contract: Any) -> bool:
    contract = _as_mapping(raw_contract)
    if not contract:
        return False
    name = contract.get("name")
    if not isinstance(name, str) or not name.strip():
        return False
    try:
        version = int(contract.get("version"))
    except (TypeError, ValueError):
        return False
    return version >= 1


def _build_contract_patch(raw_contract: Any) -> dict[str, Any]:
    existing = _as_mapping(raw_contract)
    payload: dict[str, Any] = {
        "name": STAGE_PROVENANCE_CONTRACT_NAME,
        "version": STAGE_PROVENANCE_CONTRACT_VERSION,
    }
    for key, value in existing.items():
        if key in {"name", "version"}:
            continue
        payload[key] = value
    return payload


def _derive_commit(attrs: Mapping[str, Any]) -> Optional[str]:
    commit = _normalize_scalar(get_stage_git(attrs).get("commit"))
    if _is_missing(commit):
        return None
    return str(commit)


def _to_json_compatible(value: Any) -> Any:
    value = _normalize_scalar(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_json_compatible(item) for item in value)
    if hasattr(value, "item"):
        try:
            scalar = value.item()
        except Exception:
            pass
        else:
            return _to_json_compatible(scalar)
    if hasattr(value, "tolist"):
        try:
            as_list = value.tolist()
        except Exception:
            pass
        else:
            return _to_json_compatible(as_list)
    return str(value)


def _build_crop_parameters_patch(attrs: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    if isinstance(attrs.get("parameters"), Mapping):
        existing = _as_mapping(attrs.get("parameters"))
        return existing if existing else None

    patch: dict[str, Any] = {}
    for key in CROP_PARAMETER_ATTR_KEYS:
        value = attrs.get(key)
        if value is None:
            continue
        patch[key] = _to_json_compatible(value)
    if patch:
        return patch

    crop_signature = _as_mapping(attrs.get("crop_signature"))
    if crop_signature:
        signature_patch: dict[str, Any] = {}
        for key in ("roi_size", "parameter_source"):
            if key in crop_signature and crop_signature[key] is not None:
                signature_patch[key] = _to_json_compatible(crop_signature[key])
        if signature_patch:
            return signature_patch
    return None


def _build_parameters_patch(parent_name: str, attrs: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    if parent_name == "crop_runs":
        return _build_crop_parameters_patch(attrs)
    return None


def _build_run_plan(
    zarr_path: Path,
    parent_name: str,
    run_name: str,
    run_group: zarr.Group,
    *,
    backfill_missing_parameters: bool,
) -> RunBackfillPlan:
    attrs = dict(run_group.attrs)
    provenance = _as_mapping(attrs.get("provenance"))

    raw_contract = provenance.get("contract")
    missing_contract = not _is_valid_contract(raw_contract)

    git_payload = _as_mapping(provenance.get("git"))
    missing_git_commit = _is_missing(git_payload.get("commit"))

    contract_update = _build_contract_patch(raw_contract) if missing_contract else None
    git_commit_update: Optional[str] = None
    if missing_git_commit:
        git_commit_update = _derive_commit(attrs)
    parameters_update: Optional[dict[str, Any]] = None
    if backfill_missing_parameters and provenance.get("parameters") is None:
        parameters_update = _build_parameters_patch(parent_name, attrs)

    display_path = f"{zarr_path}:{parent_name}/{run_name}"
    return RunBackfillPlan(
        display_path=display_path,
        missing_contract=missing_contract,
        missing_git_commit=missing_git_commit,
        contract_update=contract_update,
        git_commit_update=git_commit_update,
        parameters_update=parameters_update,
    )


def _apply_run_plan(run_group: zarr.Group, plan: RunBackfillPlan) -> bool:
    if not plan.would_modify:
        return False

    attrs = dict(run_group.attrs)
    provenance = _as_mapping(attrs.get("provenance"))
    changed = False

    if plan.contract_update is not None and provenance.get("contract") != plan.contract_update:
        provenance["contract"] = plan.contract_update
        changed = True

    if plan.git_commit_update is not None:
        git_payload = _as_mapping(provenance.get("git"))
        if git_payload.get("commit") != plan.git_commit_update:
            git_payload["commit"] = plan.git_commit_update
            provenance["git"] = git_payload
            changed = True

    if plan.parameters_update is not None and provenance.get("parameters") != plan.parameters_update:
        provenance["parameters"] = plan.parameters_update
        changed = True

    if changed:
        run_group.attrs["provenance"] = provenance
    return changed


def _print_dry_run_summary(
    *,
    total_runs_scanned: int,
    missing_contract: int,
    missing_git_commit: int,
    first_paths: list[str],
) -> None:
    print(f"total_runs_scanned: {total_runs_scanned}")
    print(f"missing_contract: {missing_contract}")
    print(f"missing_git_commit: {missing_git_commit}")
    print("would_modify_paths_first5:")
    if first_paths:
        for path in first_paths:
            print(f"  {path}")
    else:
        print("  (none)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize offline run provenance by filling missing "
            "provenance.contract and provenance.git.commit."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by use (default: any).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes only (default behavior).")
    parser.add_argument("--apply", action="store_true", help="Write changes to zarr attrs.")
    parser.add_argument(
        "--backfill-missing-parameters",
        action="store_true",
        help=(
            "Also backfill missing provenance.parameters when they can be safely derived "
            "from legacy run attrs (currently crop_runs only)."
        ),
    )
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        raise SystemExit("Choose either --apply or --dry-run, not both.")

    apply = bool(args.apply)
    roots = _resolve_roots(list(args.paths))

    any_zarr = False
    errors = 0
    total_runs_scanned = 0
    missing_contract = 0
    missing_git_commit = 0
    first_would_modify_paths: list[str] = []
    updated = 0

    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        try:
            root = zarr.open_group(str(zarr_path), mode="a" if apply else "r")
        except Exception as exc:
            errors += 1
            print(f"error: {zarr_path}: {exc}")
            continue

        if args.zarr_use != "any":
            observed_use = _infer_zarr_use(root, zarr_path)
            if observed_use != args.zarr_use:
                continue

        for parent_name, run_name, run_group in _iter_runs(root):
            plan = _build_run_plan(
                zarr_path,
                parent_name,
                run_name,
                run_group,
                backfill_missing_parameters=bool(args.backfill_missing_parameters),
            )
            total_runs_scanned += 1
            if plan.missing_contract:
                missing_contract += 1
            if plan.missing_git_commit:
                missing_git_commit += 1
            if plan.would_modify and len(first_would_modify_paths) < 5:
                first_would_modify_paths.append(plan.display_path)
            if apply and plan.would_modify and _apply_run_plan(run_group, plan):
                updated += 1

    if not any_zarr:
        print("No zarr files found.")
        return 1

    if apply:
        print(f"total_runs_scanned: {total_runs_scanned}")
        print(f"missing_contract: {missing_contract}")
        print(f"missing_git_commit: {missing_git_commit}")
        print(f"updated: {updated}")
    else:
        _print_dry_run_summary(
            total_runs_scanned=total_runs_scanned,
            missing_contract=missing_contract,
            missing_git_commit=missing_git_commit,
            first_paths=first_would_modify_paths,
        )

    if errors:
        print(f"errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
