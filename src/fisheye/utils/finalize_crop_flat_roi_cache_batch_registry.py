"""Serial registry finalizer for crop + flat ROI cache LSF batches.

Fan-out crop jobs must not all write to the same SQLite registry on shared
storage.  This utility consumes the per-recording JSON artifacts emitted by
``submit_crop_flat_roi_cache_bsub.sh`` and performs the registry writes from one
process after the batch is complete.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.status_ledger import upsert_recording_step_status
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.observation_coordinate_publication import (
    load_persisted_ordinary_crop_observation_geometry,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
)


@dataclass(frozen=True)
class FinalizedCropStatus:
    zarr_path: Path
    crop_run: str
    dataset_ids: tuple[str, ...]
    crop_quality_dataset_id: str
    crop_quality_rows: int
    flat_roi_cache_manifest: str | None
    flat_roi_cache_payload_bytes: int | None
    coverage_pct: float | None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_time(payload: Mapping[str, Any]) -> str:
    value = payload.get("finished_at_utc")
    return str(value) if value is not None else ""


def _ok_statuses(paths: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        payload = _load_json(path)
        if str(payload.get("status", "")).lower() == "ok":
            rows.append((path, payload))
    rows.sort(key=lambda item: (_status_time(item[1]), str(item[0])))
    return rows


def _latest_crop_status_by_zarr(run_root: Path) -> dict[str, dict[str, Any]]:
    statuses = _ok_statuses(run_root.glob("per_recording/*/*.crop.*.json"))
    by_zarr: dict[str, dict[str, Any]] = {}
    for _path, payload in statuses:
        if payload.get("stage") != "crop_materialized":
            continue
        zarr_path = payload.get("zarr_path")
        crop_run = payload.get("crop_run")
        if not zarr_path or not crop_run:
            continue
        by_zarr[str(Path(str(zarr_path)).expanduser().resolve())] = payload
    return by_zarr


def _latest_cache_status_by_zarr(run_root: Path) -> dict[str, dict[str, Any]]:
    statuses = _ok_statuses(
        path
        for path in run_root.glob("per_recording/*/*.cache.*.json")
        if ".manifest.build." not in path.name
    )
    by_zarr: dict[str, dict[str, Any]] = {}
    for _path, payload in statuses:
        source = payload.get("source")
        if not isinstance(source, Mapping):
            continue
        archive_path = source.get("archive_path")
        if not archive_path:
            continue
        by_zarr[str(Path(str(archive_path)).expanduser().resolve())] = payload
    return by_zarr


def _dataset_rows_for_zarr(registry: Registry, zarr_path: Path) -> list[sqlite3.Row]:
    rows = registry.conn.execute(
        """
        SELECT dataset_id, recording_id
        FROM datasets
        WHERE zarr_path = ?
        ORDER BY CASE WHEN instr(dataset_id, ':') = 0 THEN 0 ELSE 1 END, dataset_id;
        """,
        (str(zarr_path),),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"No registry dataset rows found for {zarr_path}")
    return rows


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_crop_coverage(crop_group: Any) -> float | None:
    try:
        frame_indices = np.asarray(crop_group["frame_indices"][:])
        frame_counts = crop_group.get("frame_counts")
        total_frames = len(frame_counts) if frame_counts is not None else None
        if not total_frames:
            return None
        return float(np.unique(frame_indices).size / int(total_frames) * 100.0)
    except Exception:
        return None


def _cache_details(cache_status: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if cache_status is None:
        return None
    return {
        "status": cache_status.get("status"),
        "stage": cache_status.get("stage"),
        "published_manifest": cache_status.get("published_manifest"),
        "published_bin": cache_status.get("published_bin"),
        "published_bin_size_bytes": cache_status.get("published_bin_size_bytes"),
        "layout": cache_status.get("layout"),
        "schema": cache_status.get("schema"),
        "array": cache_status.get("array"),
        "builder": cache_status.get("builder"),
        "publisher": cache_status.get("publisher"),
    }


def _verify_cache_pair(
    *,
    zarr_path: Path,
    crop_run: str,
    cache_status: Mapping[str, Any] | None,
    require_cache: bool,
) -> None:
    if cache_status is None:
        if require_cache:
            raise RuntimeError(f"No flat ROI cache status JSON found for {zarr_path}")
        return
    source = cache_status.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError(f"Flat ROI cache status has no source payload for {zarr_path}")
    cache_archive = Path(str(source.get("archive_path", ""))).expanduser().resolve()
    if cache_archive != zarr_path:
        raise RuntimeError(
            f"Flat ROI cache archive mismatch for {zarr_path}: {cache_archive}"
        )
    cache_crop_run = source.get("crop_run_name")
    if str(cache_crop_run) != str(crop_run):
        raise RuntimeError(
            f"Flat ROI cache crop run mismatch for {zarr_path}: "
            f"cache={cache_crop_run!r}, crop={crop_run!r}"
        )
    manifest_path = cache_status.get("published_manifest")
    bin_path = cache_status.get("published_bin")
    for label, value in (("manifest", manifest_path), ("payload", bin_path)):
        if not value or not Path(str(value)).exists():
            raise RuntimeError(
                f"Flat ROI cache {label} missing for {zarr_path}: {value!r}"
            )


def finalize_crop_flat_roi_cache_batch_registry(
    run_root: str | Path,
    *,
    registry_path: str | Path,
    apply: bool = False,
    require_cache: bool = True,
) -> dict[str, Any]:
    """Refresh crop registry status from completed crop/cache batch artifacts."""

    root_path = Path(run_root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    crop_by_zarr = _latest_crop_status_by_zarr(root_path)
    cache_by_zarr = _latest_cache_status_by_zarr(root_path)
    if not crop_by_zarr:
        raise RuntimeError(f"No successful crop status JSONs found under {root_path}")

    registry_file = Path(registry_path).expanduser().resolve()
    registry = Registry(registry_file)
    registry.conn.execute("PRAGMA busy_timeout=60000;")
    integrity_before = str(registry.conn.execute("PRAGMA integrity_check;").fetchone()[0])
    if integrity_before != "ok":
        registry.close()
        raise RuntimeError(
            f"Registry failed integrity_check before finalization: {integrity_before}"
        )

    finalized: list[FinalizedCropStatus] = []
    errors: list[dict[str, Any]] = []
    upserted_status_rows = 0

    integrity_after = integrity_before
    try:
        for zarr_text, crop_status in sorted(crop_by_zarr.items()):
            zarr_path = Path(zarr_text)
            crop_run = str(crop_status["crop_run"])
            cache_status = cache_by_zarr.get(zarr_text)
            try:
                _verify_cache_pair(
                    zarr_path=zarr_path,
                    crop_run=crop_run,
                    cache_status=cache_status,
                    require_cache=require_cache,
                )
                root = open_zarr_group_direct(zarr_path, mode="r")
                crop_parent = root["crop_runs"]
                if crop_run not in crop_parent:
                    raise RuntimeError(f"Crop run {crop_run!r} missing in {zarr_path}")
                crop_group = crop_parent[crop_run]
                if not is_run_selector_eligible(
                    crop_group
                ) or not is_run_complete_in_parent(crop_parent, crop_group):
                    raise RuntimeError(
                        f"Crop run {crop_run!r} is not complete and selector-eligible "
                        f"in {zarr_path}"
                    )
                load_persisted_ordinary_crop_observation_geometry(
                    root,
                    f"crop_runs/{crop_run}",
                )
                crop_quality_dataset_id, crop_quality_rows = registry.refresh_crop_quality_from_root(
                    root,
                    zarr_path,
                )
                dataset_rows = _dataset_rows_for_zarr(registry, zarr_path)
                dataset_ids = tuple(str(row["dataset_id"]) for row in dataset_rows)
                coverage_pct = _read_crop_coverage(crop_group)
                attrs = crop_status.get("crop_run_attrs")
                if not isinstance(attrs, Mapping):
                    attrs = {}
                crop_signature = attrs.get("crop_signature")
                if not isinstance(crop_signature, Mapping):
                    crop_signature = {}
                detection_source_type = (
                    attrs.get("detection_source_type")
                    or crop_signature.get("detection_source_type")
                    or "refined"
                )
                detection_source_path = (
                    attrs.get("detection_source_path")
                    or crop_signature.get("detection_source_path")
                )
                review_status = crop_group.attrs.get("crop_review_status")
                if not isinstance(review_status, Mapping):
                    review_status = None
                details = {
                    "reason": "serial_batch_registry_finalizer",
                    "run_state": (
                        str(crop_group.attrs.get("status")).strip().lower()
                        if crop_group.attrs.get("status") is not None
                        else None
                    ),
                    "detection_source_type": detection_source_type,
                    "detection_source_path": detection_source_path,
                    "crop_storage_mode": attrs.get("crop_storage_mode")
                    or crop_group.attrs.get("crop_storage_mode"),
                    "total_detections": attrs.get("total_detections"),
                    "crop_quality_refresh_status": "ok",
                    "crop_quality_refresh_dataset_id": crop_quality_dataset_id,
                    "crop_quality_refresh_rows": int(crop_quality_rows),
                    "crop_quality_refresh_run": crop_run,
                    "source_crop_job_id": crop_status.get("job_id"),
                    "source_crop_host": crop_status.get("host"),
                    "source_run_root": str(root_path),
                    "flat_roi_cache": _cache_details(cache_status),
                }
                if apply:
                    zarr_mtime_ns = zarr_path.stat().st_mtime_ns
                    for row in dataset_rows:
                        upsert_recording_step_status(
                            registry,
                            dataset_id=str(row["dataset_id"]),
                            recording_id=(
                                str(row["recording_id"])
                                if row["recording_id"] is not None
                                else None
                            ),
                            step_name="crop",
                            status="ok",
                            run_name=crop_run,
                            method=(
                                str(detection_source_type)
                                if detection_source_type is not None
                                else None
                            ),
                            coverage_pct=coverage_pct,
                            review_status_json=(
                                dict(review_status)
                                if isinstance(review_status, Mapping)
                                else None
                            ),
                            details_json=details,
                            source="serial_crop_flat_roi_cache_batch_finalizer",
                            zarr_mtime_ns=zarr_mtime_ns,
                        )
                        upserted_status_rows += 1
                finalized.append(
                    FinalizedCropStatus(
                        zarr_path=zarr_path,
                        crop_run=crop_run,
                        dataset_ids=dataset_ids,
                        crop_quality_dataset_id=str(crop_quality_dataset_id),
                        crop_quality_rows=int(crop_quality_rows),
                        flat_roi_cache_manifest=(
                            str(cache_status.get("published_manifest"))
                            if cache_status is not None
                            and cache_status.get("published_manifest") is not None
                            else None
                        ),
                        flat_roi_cache_payload_bytes=(
                            _safe_int(cache_status.get("published_bin_size_bytes"))
                            if cache_status is not None
                            else None
                        ),
                        coverage_pct=coverage_pct,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "zarr_path": str(zarr_path),
                        "crop_run": crop_run,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if apply:
                    raise
    finally:
        registry.conn.commit()
        if apply:
            integrity_after = str(
                registry.conn.execute("PRAGMA integrity_check;").fetchone()[0]
            )
            if integrity_after != "ok":
                registry.close()
                raise RuntimeError(
                    f"Registry failed integrity_check after finalization: {integrity_after}"
                )
        registry.close()

    status = "ok" if not errors else "error"
    return {
        "schema": "palette.crop_flat_roi_cache_batch_registry_finalizer.v1",
        "status": status,
        "apply": bool(apply),
        "run_root": str(root_path),
        "registry_path": str(registry_file),
        "registry_integrity_before": integrity_before,
        "registry_integrity_after": integrity_after,
        "finished_at_utc": utc_now(),
        "crop_status_jsons": len(crop_by_zarr),
        "cache_status_jsons": len(cache_by_zarr),
        "finalized_count": len(finalized),
        "upserted_status_rows": upserted_status_rows,
        "errors": errors,
        "finalized": [
            {
                "zarr_path": str(item.zarr_path),
                "crop_run": item.crop_run,
                "dataset_ids": list(item.dataset_ids),
                "crop_quality_dataset_id": item.crop_quality_dataset_id,
                "crop_quality_rows": item.crop_quality_rows,
                "flat_roi_cache_manifest": item.flat_roi_cache_manifest,
                "flat_roi_cache_payload_bytes": item.flat_roi_cache_payload_bytes,
                "coverage_pct": item.coverage_pct,
            }
            for item in finalized
        ],
    }


def _default_registry_path() -> Path:
    return RegistryPaths.from_env(Path.cwd()).path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serially finalize registry rows for a crop+flat-cache batch.",
    )
    parser.add_argument("run_root", type=Path, help="Batch run root directory.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config/default.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write registry rows. Without this, only validate and report.",
    )
    parser.add_argument(
        "--allow-missing-cache",
        action="store_true",
        help="Do not fail if a crop status has no matching flat-cache status.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the finalizer report JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = finalize_crop_flat_roi_cache_batch_registry(
        args.run_root,
        registry_path=args.registry or _default_registry_path(),
        apply=bool(args.apply),
        require_cache=not bool(args.allow_missing_cache),
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
