"""Backfill acquisition video stream inventory into existing analysis zarrs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

import zarr

from fisheye.shared.acquisition_crop_stream_ledger import (
    validate_current_acquisition_crop_stream_ledger,
)
from fisheye.shared.acquisition_video_streams import (
    build_acquisition_video_stream_inventory,
    write_acquisition_video_stream_inventory,
)
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings


@dataclass(frozen=True)
class InventoryBackfillPlan:
    recording_dir: Path
    manifest_path: Path
    zarr_path: Path
    status: str
    reason: str | None = None
    stream_count: int | None = None
    stream_keys: tuple[str, ...] = ()
    crop_stream_available: bool | None = None
    inventory_status: str | None = None
    crop_ledger_status: str | None = None
    crop_ledger_run: str | None = None
    crop_ledger_row_count: int | None = None


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not a JSON object: {manifest_path}")
    return payload


def _manifest_paths(paths: Iterable[Path], *, recursive: bool) -> list[Path]:
    manifests: set[Path] = set()
    for raw in paths:
        path = raw.expanduser().resolve()
        if path.is_file() and path.name == "recording_manifest.json":
            manifests.add(path)
            continue
        if path.is_dir() and (path / "recording_manifest.json").is_file():
            manifests.add(path / "recording_manifest.json")
            continue
        if path.is_dir():
            pattern = path.rglob("recording_manifest.json") if recursive else path.glob("*/recording_manifest.json")
            manifests.update(candidate for candidate in pattern if candidate.is_file())
    return sorted(manifests)


def _default_zarr_path(recording_dir: Path) -> Path:
    return recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"


def build_plan(manifest_path: Path) -> InventoryBackfillPlan:
    recording_dir = manifest_path.parent
    zarr_path = _default_zarr_path(recording_dir)
    try:
        manifest = _load_manifest(manifest_path)
    except Exception as exc:
        return InventoryBackfillPlan(
            recording_dir=recording_dir,
            manifest_path=manifest_path,
            zarr_path=zarr_path,
            status="failed",
            reason=f"manifest_unreadable: {exc}",
        )

    inventory = build_acquisition_video_stream_inventory(recording_dir, manifest)
    if inventory is None:
        return InventoryBackfillPlan(
            recording_dir=recording_dir,
            manifest_path=manifest_path,
            zarr_path=zarr_path,
            status="skipped",
            reason="no_manifest_video_streams",
        )
    if not zarr_path.exists():
        return InventoryBackfillPlan(
            recording_dir=recording_dir,
            manifest_path=manifest_path,
            zarr_path=zarr_path,
            status="missing",
            reason="analysis_zarr_missing",
            stream_count=int(inventory["stream_count"]),
            stream_keys=tuple(inventory["stream_keys"]),
            crop_stream_available=bool(inventory["crop_stream_available"]),
            inventory_status=str(inventory["inventory_status"]),
        )
    return InventoryBackfillPlan(
        recording_dir=recording_dir,
        manifest_path=manifest_path,
        zarr_path=zarr_path,
        status="ok",
        stream_count=int(inventory["stream_count"]),
        stream_keys=tuple(inventory["stream_keys"]),
        crop_stream_available=bool(inventory["crop_stream_available"]),
        inventory_status=str(inventory["inventory_status"]),
        crop_ledger_status=("planned" if inventory.get("streams", {}).get("crop") else None),
    )


def _json_record(plan: InventoryBackfillPlan, *, event: str, applied: bool) -> dict[str, Any]:
    payload = asdict(plan)
    payload.update(
        {
            "event": event,
            "applied": applied,
            "recording_dir": str(plan.recording_dir),
            "manifest_path": str(plan.manifest_path),
            "zarr_path": str(plan.zarr_path),
            "stream_keys": list(plan.stream_keys),
        }
    )
    return payload


def _emit_jsonl(handle: TextIO | None, payload: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


def _apply_plan(plan: InventoryBackfillPlan, *, imported_at_utc: str) -> InventoryBackfillPlan:
    manifest = _load_manifest(plan.manifest_path)
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    inventory = write_acquisition_video_stream_inventory(
        root,
        plan.recording_dir,
        manifest,
        imported_at_utc=imported_at_utc,
    )
    if inventory is None:
        return InventoryBackfillPlan(
            recording_dir=plan.recording_dir,
            manifest_path=plan.manifest_path,
            zarr_path=plan.zarr_path,
            status="skipped",
            reason="no_manifest_video_streams",
        )
    crop_ledger = inventory.get("streams", {}).get("crop", {}).get("canonical_ledger", {})
    if crop_ledger:
        expected_digest = str(crop_ledger["canonical_ledger_record_sha256"])
        consolidate_metadata_capture_expected_warnings(plan.zarr_path)
        consolidated = zarr.open_group(
            str(plan.zarr_path), mode="r", use_consolidated=True
        )
        validate_current_acquisition_crop_stream_ledger(
            consolidated,
            expected_record_sha256=expected_digest,
        )
    return InventoryBackfillPlan(
        recording_dir=plan.recording_dir,
        manifest_path=plan.manifest_path,
        zarr_path=plan.zarr_path,
        status="applied",
        stream_count=int(inventory["stream_count"]),
        stream_keys=tuple(inventory["stream_keys"]),
        crop_stream_available=bool(inventory["crop_stream_available"]),
        inventory_status=str(inventory["inventory_status"]),
        crop_ledger_status=(str(crop_ledger.get("canonical_ledger_status")) if crop_ledger else None),
        crop_ledger_run=(str(crop_ledger.get("canonical_ledger_run")) if crop_ledger else None),
        crop_ledger_row_count=(
            int(crop_ledger["canonical_ledger_row_count"])
            if crop_ledger.get("canonical_ledger_row_count") is not None
            else None
        ),
    )


def _filter_plans(
    plans: list[InventoryBackfillPlan],
    *,
    path_contains: list[str],
) -> list[InventoryBackfillPlan]:
    if not path_contains:
        return plans
    needles = [value for value in path_contains if value]
    return [
        plan
        for plan in plans
        if all(needle in str(plan.recording_dir) or needle in plan.recording_dir.name for needle in needles)
    ]


def _print_summary(plans: list[InventoryBackfillPlan], *, applied: bool) -> None:
    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
        suffix = f" reason={plan.reason}" if plan.reason else ""
        crop = (
            f" crop_stream_available={plan.crop_stream_available}"
            if plan.crop_stream_available is not None
            else ""
        )
        ledger = (
            f" crop_ledger_status={plan.crop_ledger_status}"
            f" crop_ledger_rows={plan.crop_ledger_row_count}"
            if plan.crop_ledger_status is not None
            else ""
        )
        print(
            f"{plan.status}: {plan.recording_dir.name}"
            f" streams={plan.stream_count} keys={','.join(plan.stream_keys)}"
            f" inventory_status={plan.inventory_status}{crop}{ledger}{suffix}"
        )
    label = "applied" if applied else "dry_run"
    print(f"Summary ({label}):")
    for key in ("applied", "ok", "skipped", "missing", "failed"):
        print(f"  {key}: {counts.get(key, 0)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill analysis/acquisition_video_streams inventory and the complete "
            "canonical acquisition crop ledger from recording_manifest.json. "
            "Dry-run by default."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Recording dirs, roots containing recording dirs, or recording_manifest.json paths.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search roots for manifests.")
    parser.add_argument(
        "--path-contains",
        action="append",
        default=[],
        help="Only include recordings whose path contains this string. May be repeated.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write inventory attrs and a pointer-last immutable crop ledger into each analysis zarr.",
    )
    parser.add_argument("--output-jsonl", type=Path, help="Optional JSONL report path.")

    args = parser.parse_args(argv)
    manifest_paths = _manifest_paths(args.paths, recursive=bool(args.recursive))
    plans = _filter_plans(
        [build_plan(path) for path in manifest_paths],
        path_contains=list(args.path_contains),
    )

    jsonl_handle: TextIO | None = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = args.output_jsonl.open("w", encoding="utf-8")

    imported_at_utc = utc_now()
    output_plans: list[InventoryBackfillPlan] = []
    try:
        for plan in plans:
            if not args.apply or plan.status != "ok":
                output_plans.append(plan)
                _emit_jsonl(jsonl_handle, _json_record(plan, event="plan", applied=False))
                continue
            try:
                applied_plan = _apply_plan(plan, imported_at_utc=imported_at_utc)
            except Exception as exc:
                applied_plan = InventoryBackfillPlan(
                    recording_dir=plan.recording_dir,
                    manifest_path=plan.manifest_path,
                    zarr_path=plan.zarr_path,
                    status="failed",
                    reason=f"apply_failed: {exc}",
                    stream_count=plan.stream_count,
                    stream_keys=plan.stream_keys,
                    crop_stream_available=plan.crop_stream_available,
                    inventory_status=plan.inventory_status,
                )
            output_plans.append(applied_plan)
            _emit_jsonl(jsonl_handle, _json_record(applied_plan, event="apply", applied=True))
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    _print_summary(output_plans, applied=bool(args.apply))
    return 1 if any(plan.status == "failed" for plan in output_plans) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
