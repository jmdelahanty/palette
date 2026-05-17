"""Batch migrate reviewed training Zarrs to a new ROI pixel contract.

This is an orchestration layer around:

- :mod:`fisheye.utils.regenerate_training_crops_pynvvc`
- :mod:`fisheye.utils.migrate_training_label_runs_identity`

It defaults to a dry-run inventory. Use ``--apply`` only after reviewing the
JSONL report.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.diagnostics.check_training_crop_pynvvc_pixel_parity import (
    check_training_crop_pynvvc_pixel_parity,
)
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.crop_roi_layout import DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN
from fisheye.shared.roi_pixel_contract import ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
from fisheye.utils.migrate_training_label_runs_identity import (
    LABEL_FAMILIES,
    migrate_training_label_runs_identity,
)
from fisheye.utils.regenerate_training_crops_pynvvc import (
    DECODE_MODES,
    SOURCE_FRAME_INDEX_MODES,
    regenerate_training_crops_pynvvc,
)


MODULE_NAME = "fisheye.utils.batch_migrate_training_crop_pixel_contract"
DEFAULT_RUN_SUFFIX = "_pynvvc_luma_v1"


@dataclass(frozen=True)
class MigrationCandidate:
    zarr_path: Path
    dataset_id: str | None = None
    registry_row: dict[str, Any] | None = None
    quality_row: dict[str, Any] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _row_to_dict(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    return {str(key): _json_safe(row[key]) for key in row.keys()}


def _looks_like_training_artifact_path(zarr_path: Path) -> bool:
    normalized = str(zarr_path).replace("\\", "/").lower()
    stem = zarr_path.stem.lower()
    return "/training/datasets/" in normalized or stem.endswith("_merged")


def _resolve_source_crop_run(root: Any, requested: str | None) -> str:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise KeyError("Zarr archive is missing crop_runs.")
    if requested:
        if requested not in crop_parent:
            raise KeyError(f"Crop run not found: crop_runs/{requested}.")
        return str(requested)
    for attr_name in ("latest_materialized", "latest", "latest_any"):
        candidate = crop_parent.attrs.get(attr_name)
        if candidate and str(candidate) in crop_parent:
            candidate_name = str(candidate)
            candidate_group = crop_parent[candidate_name]
            if _contract_name(candidate_group) == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
                source = candidate_group.attrs.get("source_crop_run")
                if source and str(source) in crop_parent:
                    return str(source)
            return candidate_name
    names = sorted(str(name) for name in crop_parent.group_keys())
    if len(names) == 1:
        return names[0]
    raise ValueError("Unable to resolve source crop run; pass --source-crop-run.")


def _contract_name(crop_group: Any) -> str | None:
    raw = crop_group.attrs.get("roi_pixel_contract") or crop_group.attrs.get("crop_pixel_contract")
    if isinstance(raw, Mapping):
        name = raw.get("name")
        return str(name) if name else None
    if raw:
        return str(raw)
    name = crop_group.attrs.get("roi_pixel_contract_name")
    return str(name) if name else None


def _target_crop_status(root: Any, target_crop_run: str) -> dict[str, Any]:
    crop_parent = root.get("crop_runs")
    if crop_parent is None or target_crop_run not in crop_parent:
        return {"exists": False}
    group = crop_parent[target_crop_run]
    return {
        "exists": True,
        "status": group.attrs.get("status"),
        "contract_name": _contract_name(group),
        "source_crop_run": group.attrs.get("source_crop_run"),
    }


def _default_report_paths() -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path("runs") / "reports" / "training_crop_pixel_contract_migration"
    return (
        root / f"training_crop_pixel_contract_migration_{stamp}.jsonl",
        root / f"training_crop_pixel_contract_migration_{stamp}.summary.json",
    )


def _unique_candidates(candidates: Sequence[MigrationCandidate]) -> list[MigrationCandidate]:
    seen: set[str] = set()
    unique: list[MigrationCandidate] = []
    for candidate in candidates:
        key = str(candidate.zarr_path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _discover_registry_candidates(
    *,
    registry_path: Path,
    zarr_use: str,
    dataset_status: str | None,
    path_contains: str | None,
    limit: int | None,
    include_training_artifacts: bool,
    approval_family: str,
    required_review_state: str | None,
    required_review_intended_use: str | None,
) -> tuple[list[MigrationCandidate], dict[str, Any]]:
    registry = Registry(registry_path)
    try:
        rows = registry.query_datasets(
            zarr_use=zarr_use,
            status=dataset_status,
            path_contains=path_contains,
            limit=limit,
        )
        discovery: dict[str, Any] = {
            "registry_path": str(registry_path),
            "rows_discovered": len(rows),
            "zarr_use": zarr_use,
            "dataset_status": dataset_status,
            "path_contains": path_contains,
            "include_training_artifacts": bool(include_training_artifacts),
            "artifact_rows_skipped": 0,
            "approval_family": approval_family,
            "required_review_state": required_review_state,
            "required_review_intended_use": required_review_intended_use,
            "approval_rows_selected": 0,
            "approval_rows_skipped": 0,
        }
        filtered_rows = []
        for row in rows:
            zarr_path = Path(str(row["zarr_path"]))
            if (
                not include_training_artifacts
                and str(row["zarr_use"] or "").lower() == "training"
                and _looks_like_training_artifact_path(zarr_path)
            ):
                discovery["artifact_rows_skipped"] += 1
                continue
            filtered_rows.append(row)

        quality_by_dataset: dict[str, dict[str, Any]] = {}
        if approval_family != "none":
            dataset_ids = [str(row["dataset_id"]) for row in filtered_rows if row["dataset_id"]]
            if approval_family != "keypoints":
                raise ValueError(f"Unsupported approval family: {approval_family}")
            quality_rows = registry.query_keypoint_quality_current(
                dataset_ids=dataset_ids,
                review_state=required_review_state,
                review_intended_use=required_review_intended_use,
            )
            quality_by_dataset = {
                str(row["dataset_id"]): _row_to_dict(row) for row in quality_rows if row["dataset_id"]
            }
            discovery["approval_rows_selected"] = len(quality_by_dataset)

        candidates: list[MigrationCandidate] = []
        for row in filtered_rows:
            dataset_id = str(row["dataset_id"]) if row["dataset_id"] is not None else None
            quality_row = quality_by_dataset.get(dataset_id or "")
            if approval_family != "none" and quality_row is None:
                discovery["approval_rows_skipped"] += 1
                continue
            candidates.append(
                MigrationCandidate(
                    zarr_path=Path(str(row["zarr_path"])),
                    dataset_id=dataset_id,
                    registry_row=_row_to_dict(row),
                    quality_row=quality_row,
                )
            )
        discovery["rows_selected"] = len(candidates)
        return candidates, discovery
    finally:
        registry.close()


def discover_candidates(
    *,
    zarr_paths: Sequence[Path],
    registry_path: Path | None,
    zarr_use: str,
    dataset_status: str | None,
    path_contains: str | None,
    limit: int | None,
    include_training_artifacts: bool,
    approval_family: str,
    required_review_state: str | None,
    required_review_intended_use: str | None,
) -> tuple[list[MigrationCandidate], dict[str, Any]]:
    discovery: dict[str, Any] = {
        "explicit_paths": len(zarr_paths),
        "registry": None,
    }
    candidates = [
        MigrationCandidate(zarr_path=Path(path).expanduser())
        for path in zarr_paths
    ]
    if registry_path is not None:
        registry_candidates, registry_discovery = _discover_registry_candidates(
            registry_path=registry_path,
            zarr_use=zarr_use,
            dataset_status=dataset_status,
            path_contains=path_contains,
            limit=limit,
            include_training_artifacts=include_training_artifacts,
            approval_family=approval_family,
            required_review_state=required_review_state,
            required_review_intended_use=required_review_intended_use,
        )
        discovery["registry"] = registry_discovery
        candidates.extend(registry_candidates)
    selected = _unique_candidates(candidates)
    discovery["selected_unique"] = len(selected)
    return selected, discovery


def migrate_one_candidate(
    candidate: MigrationCandidate,
    *,
    apply: bool,
    source_crop_run: str | None,
    target_crop_suffix: str,
    label_run_suffix: str,
    families: Sequence[str] | None,
    all_label_runs: bool,
    overwrite: bool,
    set_latest: bool,
    source_frame_index_mode: str,
    crop_decode_mode: str,
    decode_chunk_frames: int,
    roi_chunk_len: int,
    parity_sample_count: int,
    parity_boundary_sample_count: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    archive_path = candidate.zarr_path.expanduser()
    root = zarr.open_group(str(archive_path), mode="r", use_consolidated=False)
    resolved_source_crop = _resolve_source_crop_run(root, source_crop_run)
    target_crop_run = f"{resolved_source_crop}{target_crop_suffix}"
    existing_target = _target_crop_status(root, target_crop_run)

    record: dict[str, Any] = {
        "status": "running" if apply else "planned",
        "mode": "apply" if apply else "dry_run",
        "zarr_path": str(archive_path),
        "dataset_id": candidate.dataset_id,
        "source_crop_run": resolved_source_crop,
        "target_crop_run": target_crop_run,
        "target_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        "target_crop_existing": existing_target,
        "registry_row": candidate.registry_row,
        "quality_row": candidate.quality_row,
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "started_at_utc": _utc_now(),
    }

    if existing_target["exists"] and not overwrite:
        if existing_target.get("contract_name") != ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
            raise ValueError(
                f"Existing target crop_runs/{target_crop_run} has contract "
                f"{existing_target.get('contract_name')!r}, expected "
                f"{ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME!r}."
            )
        if str(existing_target.get("status") or "").lower() not in {"completed", "ok"}:
            raise ValueError(
                f"Existing target crop_runs/{target_crop_run} is not completed "
                f"(status={existing_target.get('status')!r}); pass --overwrite to replace it."
            )
        record["crop_report"] = {
            "status": "existing",
            "source_crop_run": resolved_source_crop,
            "target_crop_run": target_crop_run,
        }
    else:
        record["crop_report"] = regenerate_training_crops_pynvvc(
            zarr_path=archive_path,
            source_crop_run=resolved_source_crop,
            target_crop_run=target_crop_run,
            source_frame_index_mode=source_frame_index_mode,
            decode_mode=crop_decode_mode,
            decode_chunk_frames=decode_chunk_frames,
            roi_chunk_len=roi_chunk_len,
            overwrite=overwrite,
            set_latest=set_latest,
            dry_run=not apply,
        )

    if apply or existing_target["exists"]:
        record["label_report"] = migrate_training_label_runs_identity(
            zarr_path=archive_path,
            target_crop_run=target_crop_run,
            source_crop_run=resolved_source_crop,
            families=families,
            all_runs=all_label_runs,
            run_suffix=label_run_suffix,
            overwrite=overwrite,
            set_latest=set_latest,
            dry_run=not apply,
        )
    else:
        record["label_report"] = {
            "status": "deferred_until_target_crop_exists",
            "reason": "dry_run_target_crop_not_present",
            "families_requested": list(families) if families else list(LABEL_FAMILIES),
            "all_runs": bool(all_label_runs),
            "run_suffix": label_run_suffix,
        }

    if apply and parity_sample_count > 0:
        parity_report = check_training_crop_pynvvc_pixel_parity(
            zarr_path=archive_path,
            crop_run=target_crop_run,
            sample_count=int(parity_sample_count),
            boundary_sample_count=int(parity_boundary_sample_count),
            source_frame_index_mode=source_frame_index_mode,
            decode_chunk_frames=decode_chunk_frames,
        )
        record["parity_report"] = parity_report
        if parity_report.get("status") != "ok":
            record["status"] = "parity_failed"
            record["completed_at_utc"] = _utc_now()
            record["duration_seconds"] = float(time.perf_counter() - started)
            return _json_safe(record)

    record["status"] = "ok" if apply else "planned"
    record["completed_at_utc"] = _utc_now()
    record["duration_seconds"] = float(time.perf_counter() - started)
    return _json_safe(record)


def batch_migrate_training_crop_pixel_contract(
    *,
    candidates: Sequence[MigrationCandidate],
    apply: bool,
    source_crop_run: str | None = None,
    target_crop_suffix: str = DEFAULT_RUN_SUFFIX,
    label_run_suffix: str = DEFAULT_RUN_SUFFIX,
    families: Sequence[str] | None = None,
    all_label_runs: bool = False,
    overwrite: bool = False,
    set_latest: bool = False,
    source_frame_index_mode: str = "auto",
    crop_decode_mode: str = "auto",
    decode_chunk_frames: int = 1,
    roi_chunk_len: int = DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    parity_sample_count: int = 0,
    parity_boundary_sample_count: int = 4,
    fail_fast: bool = False,
    jsonl_report: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    started_at = _utc_now()
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    if jsonl_report is not None:
        jsonl_report.parent.mkdir(parents=True, exist_ok=True)
        jsonl_report.write_text("", encoding="utf-8")

    for index, candidate in enumerate(candidates, start=1):
        try:
            record = migrate_one_candidate(
                candidate,
                apply=apply,
                source_crop_run=source_crop_run,
                target_crop_suffix=target_crop_suffix,
                label_run_suffix=label_run_suffix,
                families=families,
                all_label_runs=all_label_runs,
                overwrite=overwrite,
                set_latest=set_latest,
                source_frame_index_mode=source_frame_index_mode,
                crop_decode_mode=crop_decode_mode,
                decode_chunk_frames=decode_chunk_frames,
                roi_chunk_len=roi_chunk_len,
                parity_sample_count=parity_sample_count,
                parity_boundary_sample_count=parity_boundary_sample_count,
            )
        except Exception as exc:
            record = {
                "status": "error",
                "mode": "apply" if apply else "dry_run",
                "zarr_path": str(candidate.zarr_path),
                "dataset_id": candidate.dataset_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "host": socket.gethostname(),
                "pid": int(os.getpid()),
                "completed_at_utc": _utc_now(),
            }
        record["batch_index"] = index
        record["batch_total"] = len(candidates)
        records.append(_json_safe(record))
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        if jsonl_report is not None:
            with jsonl_report.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
        if record["status"] == "error" and fail_fast:
            break

    summary = {
        "status": "ok" if counts.get("error", 0) == 0 and counts.get("parity_failed", 0) == 0 else "issues",
        "mode": "apply" if apply else "dry_run",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "duration_seconds": float(time.perf_counter() - started),
        "candidates": len(candidates),
        "counts": counts,
        "jsonl_report": str(jsonl_report) if jsonl_report is not None else None,
        "target_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        "target_crop_suffix": target_crop_suffix,
        "label_run_suffix": label_run_suffix,
        "families": list(families) if families else list(LABEL_FAMILIES),
        "all_label_runs": bool(all_label_runs),
        "set_latest": bool(set_latest),
        "overwrite": bool(overwrite),
        "crop_decode_mode": str(crop_decode_mode),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
    }
    return records, _json_safe(summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch migrate approved training Zarr crop pixels to "
            f"{ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME} and identity-copy labels."
        )
    )
    parser.add_argument("zarr_paths", nargs="*", type=Path, help="Training Zarr paths to migrate.")
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path for candidate discovery.")
    parser.add_argument("--zarr-use", default="training", help="Registry zarr_use filter.")
    parser.add_argument("--dataset-status", help="Optional registry dataset status filter.")
    parser.add_argument("--path-contains", help="Optional registry path substring filter.")
    parser.add_argument("--limit", type=int, help="Limit registry-discovered candidates.")
    parser.add_argument(
        "--include-training-artifacts",
        action="store_true",
        help="Include merged/exported training artifacts; default skips them.",
    )
    parser.add_argument(
        "--approval-family",
        choices=("keypoints", "none"),
        default="keypoints",
        help="Registry approval gate used for discovered candidates.",
    )
    parser.add_argument("--required-review-state", default="approved")
    parser.add_argument("--required-review-intended-use", default="training")
    parser.add_argument("--source-crop-run", help="Force a source crop run for every selected Zarr.")
    parser.add_argument("--target-crop-suffix", default=DEFAULT_RUN_SUFFIX)
    parser.add_argument("--label-run-suffix", default=DEFAULT_RUN_SUFFIX)
    parser.add_argument("--families", nargs="*", help="Label families to migrate; default is all supported.")
    parser.add_argument(
        "--all-label-runs",
        dest="all_label_runs",
        action="store_true",
        default=False,
        help="Migrate all non-migrated label runs in each family.",
    )
    parser.add_argument(
        "--latest-label-run-only",
        dest="all_label_runs",
        action="store_false",
        help="Only migrate each family's latest source label run. This is the default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set-latest", action="store_true", help="Update crop and label latest pointers.")
    parser.add_argument("--source-frame-index-mode", choices=SOURCE_FRAME_INDEX_MODES, default="auto")
    parser.add_argument(
        "--crop-decode-mode",
        choices=DECODE_MODES,
        default="auto",
        help="PyNvVideoCodec access pattern for crop regeneration.",
    )
    parser.add_argument(
        "--decode-chunk-frames",
        type=int,
        default=1,
        help=(
            "Frame indices per indexed PyNvVideoCodec request. Default 1 avoids slow wide-span "
            "indexed batches for sparse long training videos."
        ),
    )
    parser.add_argument("--roi-chunk-len", type=int, default=DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN)
    parser.add_argument(
        "--parity-sample-count",
        type=int,
        default=0,
        help=(
            "After --apply, sample this many rows for PyNvVC parity. Default 0 skips parity; "
            "the parity checker currently uses sequential decode and can be very slow for sparse "
            "training zarrs."
        ),
    )
    parser.add_argument("--parity-boundary-sample-count", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Write crop and migrated label runs.")
    mode.add_argument("--dry-run", action="store_true", help="Inventory only. This is the default.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--jsonl-report", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--json", action="store_true", help="Print full summary JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    apply_mode = bool(args.apply)
    registry_path = args.registry
    if registry_path is None and not args.zarr_paths:
        registry_path = RegistryPaths.from_env(Path.cwd()).path
    if registry_path is not None:
        registry_path = registry_path.expanduser()
        if not registry_path.exists():
            raise SystemExit(f"Registry does not exist: {registry_path}")

    default_jsonl, default_summary = _default_report_paths()
    jsonl_report = args.jsonl_report or default_jsonl
    summary_json = args.summary_json or default_summary

    candidates, discovery = discover_candidates(
        zarr_paths=args.zarr_paths,
        registry_path=registry_path,
        zarr_use=args.zarr_use,
        dataset_status=args.dataset_status,
        path_contains=args.path_contains,
        limit=args.limit,
        include_training_artifacts=args.include_training_artifacts,
        approval_family=args.approval_family,
        required_review_state=args.required_review_state,
        required_review_intended_use=args.required_review_intended_use,
    )
    records, summary = batch_migrate_training_crop_pixel_contract(
        candidates=candidates,
        apply=apply_mode,
        source_crop_run=args.source_crop_run,
        target_crop_suffix=args.target_crop_suffix,
        label_run_suffix=args.label_run_suffix,
        families=args.families,
        all_label_runs=args.all_label_runs,
        overwrite=args.overwrite,
        set_latest=args.set_latest,
        source_frame_index_mode=args.source_frame_index_mode,
        crop_decode_mode=args.crop_decode_mode,
        decode_chunk_frames=args.decode_chunk_frames,
        roi_chunk_len=args.roi_chunk_len,
        parity_sample_count=args.parity_sample_count,
        parity_boundary_sample_count=args.parity_boundary_sample_count,
        fail_fast=args.fail_fast,
        jsonl_report=jsonl_report,
    )
    summary["discovery"] = discovery
    summary["records_written"] = len(records)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    else:
        counts = summary.get("counts", {})
        print(
            f"mode: {summary['mode']}\n"
            f"candidates: {summary['candidates']}\n"
            f"counts: {json.dumps(counts, sort_keys=True)}\n"
            f"jsonl_report: {jsonl_report}\n"
            f"summary_json: {summary_json}"
        )
        if not apply_mode:
            print("dry_run_only: pass --apply to write migrated crop and label runs")

    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
