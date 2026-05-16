"""Identity-migrate training label runs to a new crop run.

The crop-representation migration can create a new ``crop_runs/<run>`` whose
row geometry is identical to an existing crop run but whose ROI pixels use a
new representation. Labels in ROI coordinates can be copied unchanged when the
row order and crop geometry are identical. This utility creates new label runs
that point at the new crop run and records that the label transform is identity.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

LABEL_FAMILIES = (
    "keypoints_runs",
    "eye_masks_runs",
    "refined_eye_masks_runs",
    "subject_mask_runs",
    "refined_subject_masks_runs",
)
MIGRATION_VERSION = "training_label_identity_crop_pixel_contract_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _normalize_families(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(LABEL_FAMILIES)
    out: list[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            text = part.strip()
            if not text:
                continue
            if text not in LABEL_FAMILIES:
                choices = ", ".join(LABEL_FAMILIES)
                raise ValueError(f"Unsupported label family '{text}'. Expected one of: {choices}")
            if text not in out:
                out.append(text)
    return out


def _resolve_target_crop(root: Any, target_crop_run: str) -> Any:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise KeyError("Zarr archive is missing crop_runs.")
    if target_crop_run not in crop_parent:
        raise KeyError(f"Target crop run not found: crop_runs/{target_crop_run}.")
    return crop_parent[target_crop_run]


def _resolve_source_crop_run(
    *,
    target_crop_group: Any,
    source_crop_run: str | None,
) -> str:
    if source_crop_run:
        return str(source_crop_run)
    source = target_crop_group.attrs.get("source_crop_run")
    if source:
        return str(source)
    raise ValueError("Unable to infer source crop run from target crop attrs; pass --source-crop-run.")


def _copy_attrs(source: Any, target: Any) -> None:
    target.attrs.update(dict(source.attrs))


def _copy_array(source: Any, target_parent: Any, name: str) -> None:
    chunks = getattr(source, "chunks", None)
    kwargs: dict[str, Any] = {
        "data": np.asarray(source[:]),
        "overwrite": True,
    }
    if chunks is not None:
        kwargs["chunks"] = chunks
    target_parent.create_array(name, **kwargs)


def _copy_group_recursive(source: Any, target: Any) -> None:
    _copy_attrs(source, target)
    for name in sorted(str(k) for k in source.array_keys()):
        _copy_array(source[name], target, name)
    for name in sorted(str(k) for k in source.group_keys()):
        child = target.create_group(name, overwrite=True)
        _copy_group_recursive(source[name], child)


def _array_paths(group: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for name in sorted(str(k) for k in group.array_keys()):
        paths.append(f"{prefix}{name}")
    for name in sorted(str(k) for k in group.group_keys()):
        paths.extend(_array_paths(group[name], prefix=f"{prefix}{name}/"))
    return paths


def _arrays_equal_recursive(source: Any, target: Any) -> dict[str, Any]:
    source_paths = _array_paths(source)
    target_paths = _array_paths(target)
    mismatches: list[str] = []
    if source_paths != target_paths:
        missing = sorted(set(source_paths) - set(target_paths))
        extra = sorted(set(target_paths) - set(source_paths))
        mismatches.extend([f"missing:{path}" for path in missing])
        mismatches.extend([f"extra:{path}" for path in extra])
    for path in source_paths:
        if path not in target_paths:
            continue
        src = np.asarray(source[path][:])
        dst = np.asarray(target[path][:])
        if src.shape != dst.shape or src.dtype != dst.dtype:
            mismatches.append(path)
            continue
        if np.issubdtype(src.dtype, np.floating):
            equal = np.array_equal(src, dst, equal_nan=True)
        else:
            equal = np.array_equal(src, dst)
        if not equal:
            mismatches.append(path)
    return {
        "array_count": int(len(source_paths)),
        "byte_identical": not mismatches,
        "mismatches": mismatches,
    }


def _validate_label_geometry(
    *,
    source_group: Any,
    target_crop_group: Any,
    source_crop_run: str,
    target_crop_run: str,
    allow_source_crop_mismatch: bool,
) -> dict[str, Any]:
    source_attr = source_group.attrs.get("source_crop_run") or source_group.attrs.get("crop_run")
    if source_attr and str(source_attr) != str(source_crop_run) and not allow_source_crop_mismatch:
        raise ValueError(
            f"Label run source_crop_run={source_attr!r} does not match expected source crop "
            f"{source_crop_run!r}."
        )
    checks: dict[str, Any] = {
        "source_crop_attr": str(source_attr) if source_attr is not None else None,
        "target_crop_run": str(target_crop_run),
    }
    for name in ("frame_indices", "detection_indices", "detection_source"):
        if name in source_group and name in target_crop_group:
            src = np.asarray(source_group[name][:])
            crop = np.asarray(target_crop_group[name][:])
            checks[f"{name}_matches_target_crop"] = bool(src.shape == crop.shape and np.array_equal(src, crop))
            if not checks[f"{name}_matches_target_crop"]:
                raise ValueError(
                    f"Label array {name} does not match target crop run {target_crop_run}; "
                    "identity migration is unsafe."
                )
    return checks


def _select_source_runs(parent: Any, *, all_runs: bool) -> list[str]:
    names = [
        str(k)
        for k in parent.group_keys()
        if not parent[str(k)].attrs.get("label_identity_migration_version")
        and not parent[str(k)].attrs.get("source_label_run")
    ]
    names = sorted(names)
    if all_runs:
        return names
    latest = parent.attrs.get("latest")
    if latest and str(latest) in parent:
        return [str(latest)]
    if len(names) == 1:
        return names
    return []


def migrate_training_label_runs_identity(
    *,
    zarr_path: str | Path,
    target_crop_run: str,
    source_crop_run: str | None = None,
    families: Sequence[str] | None = None,
    all_runs: bool = False,
    run_suffix: str = "_pynvvc_luma_v1",
    overwrite: bool = False,
    set_latest: bool = False,
    dry_run: bool = False,
    allow_source_crop_mismatch: bool = False,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    root = zarr.open_group(str(archive_path), mode="a", use_consolidated=False)
    target_crop_group = _resolve_target_crop(root, target_crop_run)
    resolved_source_crop = _resolve_source_crop_run(
        target_crop_group=target_crop_group,
        source_crop_run=source_crop_run,
    )
    selected_families = _normalize_families(families)
    report: dict[str, Any] = {
        "status": "dry_run" if dry_run else "ok",
        "zarr_path": str(archive_path),
        "source_crop_run": str(resolved_source_crop),
        "target_crop_run": str(target_crop_run),
        "families_requested": selected_families,
        "all_runs": bool(all_runs),
        "run_suffix": str(run_suffix),
        "set_latest": bool(set_latest),
        "migrations": [],
        "skipped": [],
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
    }

    target_crop_contract = target_crop_group.attrs.get("roi_pixel_contract")
    target_crop_contract_name = (
        target_crop_contract.get("name") if isinstance(target_crop_contract, Mapping) else None
    )

    for family in selected_families:
        parent = root.get(family)
        if parent is None:
            report["skipped"].append({"family": family, "reason": "missing_parent"})
            continue
        source_runs = _select_source_runs(parent, all_runs=all_runs)
        if not source_runs:
            report["skipped"].append({"family": family, "reason": "no_source_runs"})
            continue
        for source_run in source_runs:
            source_group = parent[source_run]
            source_attr = source_group.attrs.get("source_crop_run") or source_group.attrs.get("crop_run")
            if source_attr and str(source_attr) != str(resolved_source_crop):
                report["skipped"].append(
                    {
                        "family": family,
                        "source_run": source_run,
                        "reason": "source_crop_mismatch",
                        "source_crop_attr": str(source_attr),
                    }
                )
                continue
            target_run = f"{source_run}{run_suffix}"
            if target_run in parent and not overwrite:
                report["skipped"].append(
                    {
                        "family": family,
                        "source_run": source_run,
                        "target_run": target_run,
                        "reason": "target_exists",
                    }
                )
                continue

            geometry_checks = _validate_label_geometry(
                source_group=source_group,
                target_crop_group=target_crop_group,
                source_crop_run=resolved_source_crop,
                target_crop_run=target_crop_run,
                allow_source_crop_mismatch=allow_source_crop_mismatch,
            )
            entry = {
                "family": family,
                "source_run": source_run,
                "target_run": target_run,
                "source_path": f"{family}/{source_run}",
                "target_path": f"{family}/{target_run}",
                "geometry_checks": geometry_checks,
            }
            if dry_run:
                report["migrations"].append(entry)
                continue

            target_group = parent.create_group(target_run, overwrite=bool(overwrite))
            _copy_group_recursive(source_group, target_group)
            original_status = target_group.attrs.get("status")
            target_group.attrs.update(
                {
                    "status": "completed",
                    "completed_at_utc": _utc_now(),
                    "generated_by": "fisheye.utils.migrate_training_label_runs_identity",
                    "label_identity_migration_version": MIGRATION_VERSION,
                    "label_coordinate_transform": "identity",
                    "source_label_run": str(source_run),
                    "source_label_path": f"{family}/{source_run}",
                    "source_crop_run": str(target_crop_run),
                    "source_crop_path": f"crop_runs/{target_crop_run}",
                    "identity_migration_source_crop_run": str(resolved_source_crop),
                    "identity_migration_target_crop_run": str(target_crop_run),
                    "source_roi_pixel_contract": target_crop_contract,
                    "source_roi_pixel_contract_name": target_crop_contract_name,
                }
            )
            if original_status is not None:
                target_group.attrs["identity_migration_source_status"] = original_status
            identity = _arrays_equal_recursive(source_group, target_group)
            if not identity["byte_identical"]:
                raise RuntimeError(
                    f"Identity migration produced array mismatches for {family}/{target_run}: "
                    f"{identity['mismatches'][:8]}"
                )
            if set_latest:
                parent.attrs["latest"] = target_run
            entry["array_identity"] = identity
            report["migrations"].append(entry)

    return _json_safe(report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identity-migrate label runs to a new training crop run.")
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--target-crop-run", required=True)
    parser.add_argument("--source-crop-run")
    parser.add_argument("--families", nargs="*", help="Label families to migrate; default is all supported.")
    parser.add_argument("--all-runs", action="store_true", help="Migrate all source runs in each family.")
    parser.add_argument("--run-suffix", default="_pynvvc_luma_v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set-latest", action="store_true", help="Update each migrated parent latest pointer.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-source-crop-mismatch", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = migrate_training_label_runs_identity(
        zarr_path=args.zarr_path,
        target_crop_run=args.target_crop_run,
        source_crop_run=args.source_crop_run,
        families=args.families,
        all_runs=args.all_runs,
        run_suffix=args.run_suffix,
        overwrite=args.overwrite,
        set_latest=args.set_latest,
        dry_run=args.dry_run,
        allow_source_crop_mismatch=args.allow_source_crop_mismatch,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.json or args.output_json is None:
        print(text)
    else:
        print(
            f"status: {report['status']}\n"
            f"migrations: {len(report['migrations'])}\n"
            f"skipped: {len(report['skipped'])}\n"
            f"target_crop_run: {report['target_crop_run']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
