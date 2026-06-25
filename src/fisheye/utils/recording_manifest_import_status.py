#!/usr/bin/env python3
"""Backfill/import status fields in recording_manifest.json files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from fisheye.registry.db import Registry
from fisheye.shared.batch_logging import utc_now


@dataclass(frozen=True)
class ManifestImportStatusUpdate:
    recording_dir: Path
    zarr_path: Path
    status: str
    import_log: Optional[Path]
    imported_at_utc: Optional[str]
    import_run_id: Optional[str] = None
    error: Optional[str] = None
    registry_path: Optional[Path] = None
    registry_dataset_id: Optional[str] = None
    registry_synced_at_utc: Optional[str] = None


@dataclass(frozen=True)
class ManifestImportStatusResult:
    manifest_path: Path
    status: str
    changed: bool
    error: Optional[str] = None


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _coerce_path(value: Any) -> Optional[Path]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return Path(text) if text else None


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _manifest_update_payload(update: ManifestImportStatusUpdate) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": update.status,
        "zarr_path": str(update.zarr_path),
        "updated_at_utc": update.imported_at_utc or utc_now(),
    }
    if update.import_log is not None:
        payload["import_log"] = str(update.import_log)
    if update.imported_at_utc is not None:
        payload["imported_at_utc"] = update.imported_at_utc
    if update.import_run_id:
        payload["run_id"] = update.import_run_id
    if update.error:
        payload["error"] = update.error
    if update.registry_path is not None:
        payload["registry_path"] = str(update.registry_path)
    if update.registry_dataset_id:
        payload["registry_dataset_id"] = update.registry_dataset_id
    if update.registry_synced_at_utc:
        payload["registry_synced_at_utc"] = update.registry_synced_at_utc
    return payload


def write_manifest_import_status(update: ManifestImportStatusUpdate) -> ManifestImportStatusResult:
    manifest_path = update.recording_dir / "recording_manifest.json"
    if not manifest_path.is_file():
        return ManifestImportStatusResult(
            manifest_path=manifest_path,
            status="missing_manifest",
            changed=False,
            error=f"manifest not found: {manifest_path}",
        )
    try:
        payload = _read_json_object(manifest_path)
    except Exception as exc:
        return ManifestImportStatusResult(
            manifest_path=manifest_path,
            status="read_failed",
            changed=False,
            error=str(exc),
        )

    old_payload = json.dumps(payload, sort_keys=True, default=str)
    payload["import_status"] = update.status
    if update.import_log is not None:
        payload["import_log"] = str(update.import_log)
    if update.imported_at_utc is not None:
        payload["imported_at_utc"] = update.imported_at_utc
    payload["analysis_zarr_path"] = str(update.zarr_path)
    payload["analysis_import"] = _manifest_update_payload(update)
    if update.registry_path is not None:
        payload["registry_path"] = str(update.registry_path)
    if update.registry_dataset_id:
        payload["registry_dataset_id"] = update.registry_dataset_id
    if update.registry_synced_at_utc:
        payload["registry_synced_at_utc"] = update.registry_synced_at_utc

    changed = json.dumps(payload, sort_keys=True, default=str) != old_payload
    if not changed:
        return ManifestImportStatusResult(manifest_path=manifest_path, status="unchanged", changed=False)
    try:
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        return ManifestImportStatusResult(
            manifest_path=manifest_path,
            status="write_failed",
            changed=False,
            error=str(exc),
        )
    return ManifestImportStatusResult(manifest_path=manifest_path, status="updated", changed=True)


def iter_updates_from_import_log(log_path: Path) -> Iterable[ManifestImportStatusUpdate]:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = payload.get("event")
        if event not in {"recording_ok", "recording_failed", "recording_skipped"}:
            continue
        recording_dir = _coerce_path(payload.get("recording_dir"))
        zarr_path = _coerce_path(payload.get("zarr_path"))
        if recording_dir is None or zarr_path is None:
            continue
        if event == "recording_ok":
            status = "ok"
        elif event == "recording_failed":
            status = "failed"
        else:
            status = _coerce_text(payload.get("status")) or "skipped"
            if status == "missing":
                continue
        yield ManifestImportStatusUpdate(
            recording_dir=recording_dir,
            zarr_path=zarr_path,
            status=status,
            import_log=log_path,
            imported_at_utc=_coerce_text(payload.get("ts_utc")),
            import_run_id=_coerce_text(payload.get("run_id")),
            error=_coerce_text(payload.get("error")) or _coerce_text(payload.get("reason")),
        )


def _read_file_list(path: Path) -> list[Path]:
    values: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        values.append(Path(text))
    return values


def _sync_registry(registry_path: Path, zarr_path: Path) -> Optional[str]:
    registry = Registry(registry_path)
    try:
        return registry.scan_zarr(zarr_path)
    finally:
        registry.close()


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("import_logs", nargs="*", type=Path, help="Import JSONL logs to read.")
    parser.add_argument("--file-list", action="append", type=Path, help="Text file of import JSONL paths.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path to scan each ok zarr.")
    parser.add_argument("--apply", action="store_true", help="Write manifest updates (default is dry-run).")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    log_paths: list[Path] = []
    if args.file_list:
        for file_list in args.file_list:
            log_paths.extend(_read_file_list(file_list))
    log_paths.extend(args.import_logs)
    log_paths = [path.expanduser().resolve() for path in log_paths]
    if not log_paths:
        raise SystemExit("No import logs provided.")

    planned = 0
    updated = 0
    failed = 0
    registry_synced = 0
    registry_failed = 0
    for log_path in log_paths:
        for raw_update in iter_updates_from_import_log(log_path):
            planned += 1
            update = raw_update
            if args.registry is not None and raw_update.status == "ok":
                try:
                    dataset_id = _sync_registry(args.registry, raw_update.zarr_path) if args.apply else None
                except Exception as exc:
                    registry_failed += 1
                    failed += 1
                    print(f"REGISTRY FAILED {raw_update.zarr_path}: {exc}")
                    continue
                if args.apply:
                    registry_synced += 1
                update = ManifestImportStatusUpdate(
                    recording_dir=raw_update.recording_dir,
                    zarr_path=raw_update.zarr_path,
                    status=raw_update.status,
                    import_log=raw_update.import_log,
                    imported_at_utc=raw_update.imported_at_utc,
                    import_run_id=raw_update.import_run_id,
                    error=raw_update.error,
                    registry_path=args.registry,
                    registry_dataset_id=dataset_id,
                    registry_synced_at_utc=utc_now() if args.apply else None,
                )

            if args.apply:
                result = write_manifest_import_status(update)
                if result.error:
                    failed += 1
                    print(f"FAILED {result.manifest_path}: {result.error}")
                elif result.changed:
                    updated += 1
                    print(f"UPDATED {result.manifest_path}")
                else:
                    print(f"UNCHANGED {result.manifest_path}")
            else:
                print(f"WOULD UPDATE {update.recording_dir / 'recording_manifest.json'}")

    summary = {
        "logs": len(log_paths),
        "planned": planned,
        "updated": updated,
        "failed": failed,
        "registry_synced": registry_synced,
        "registry_failed": registry_failed,
        "apply": bool(args.apply),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
