"""Apply a reviewed source-video metadata preflight report with rollback safety."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

from fisheye.shared.source_video_metadata import (
    SOURCE_VIDEO_LAYOUT_SINGLE,
    SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE,
    SOURCE_VIDEO_METADATA_SCHEMA_ID,
    resolve_source_video_from_attrs,
)
from fisheye.utils.preflight_source_video_metadata_backfill import (
    REPORT_SCHEMA_ID,
    read_attrs_strict,
    select_registry_datasets,
)


APPLY_RECEIPT_SCHEMA_ID = "palette.source_video_metadata_backfill_apply.v1"
ALLOWED_ROOT_UPDATE_KEYS = {
    "recording_path",
    "source_path",
    "source_video_metadata",
    "source_video_path",
}
ALLOWED_RAW_UPDATE_KEYS = {"source_path"}


class SourceVideoMetadataApplyError(RuntimeError):
    """Raised when a guarded metadata apply cannot proceed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _strict_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    mode = path.stat().st_mode
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _atomic_replace_bytes(path, _strict_json_bytes(payload))
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_strict_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_reviewed_report(
    report_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes, str]:
    resolved = report_path.expanduser().resolve()
    raw = resolved.read_bytes()
    actual_sha256 = _sha256_bytes(raw)
    if actual_sha256 != expected_sha256.strip().lower():
        raise SourceVideoMetadataApplyError(
            "Preflight report SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    report = json.loads(raw.decode("utf-8"))
    if not isinstance(report, dict):
        raise SourceVideoMetadataApplyError("Preflight report root is not an object")
    if report.get("schema_id") != REPORT_SCHEMA_ID:
        raise SourceVideoMetadataApplyError(
            f"Unsupported preflight report schema: {report.get('schema_id')!r}"
        )
    if report.get("mode") != "read_only_preflight":
        raise SourceVideoMetadataApplyError("Report is not a read-only preflight")
    summary = report.get("summary")
    if not isinstance(summary, Mapping) or summary.get("ready_to_apply") is not True:
        raise SourceVideoMetadataApplyError("Preflight report is not ready_to_apply")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SourceVideoMetadataApplyError("Preflight report contains no rows")
    return report, raw, actual_sha256


def _verify_file_precondition(expected: Mapping[str, Any]) -> Path:
    path_text = expected.get("path")
    if not path_text:
        raise SourceVideoMetadataApplyError("Metadata precondition is missing path")
    path = Path(str(path_text)).expanduser().resolve()
    if not path.is_file():
        raise SourceVideoMetadataApplyError(f"Precondition file is missing: {path}")
    stat = path.stat()
    checks = {
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    for key, actual in checks.items():
        expected_value = expected.get(key)
        if actual != expected_value:
            raise SourceVideoMetadataApplyError(
                f"Precondition drift for {path}: {key} expected={expected_value} actual={actual}"
            )
    return path


def _verify_source_video_precondition(expected: Mapping[str, Any]) -> Path:
    path_text = expected.get("path")
    if not path_text:
        raise SourceVideoMetadataApplyError("Source-video precondition is missing path")
    path = Path(str(path_text)).expanduser().resolve()
    if not path.is_file():
        raise SourceVideoMetadataApplyError(f"Source video is missing: {path}")
    stat = path.stat()
    for key, actual in (
        ("size_bytes", int(stat.st_size)),
        ("mtime_ns", int(stat.st_mtime_ns)),
    ):
        expected_value = expected.get(key)
        if actual != expected_value:
            raise SourceVideoMetadataApplyError(
                f"Source-video drift for {path}: {key} expected={expected_value} actual={actual}"
            )
    return path


def _verify_registry_snapshot(report: Mapping[str, Any]) -> None:
    registry_path = Path(str(report.get("registry_path"))).expanduser().resolve()
    selection = report.get("selection")
    if not isinstance(selection, Mapping):
        raise SourceVideoMetadataApplyError("Report selection is missing")
    path_contains = str(selection.get("path_contains") or "")
    datasets = select_registry_datasets(
        registry_path,
        path_contains=path_contains,
    )
    current = {
        dataset.dataset_id: (
            dataset.recording_id,
            str(dataset.zarr_path.expanduser().resolve()),
        )
        for dataset in datasets
    }
    planned_rows = report.get("rows")
    assert isinstance(planned_rows, list)
    planned = {
        str(row.get("dataset_id")): (
            row.get("recording_id"),
            str(Path(str(row.get("zarr_path"))).expanduser().resolve()),
        )
        for row in planned_rows
        if isinstance(row, Mapping)
    }
    if current != planned:
        raise SourceVideoMetadataApplyError(
            "Registry cohort drifted after preflight; regenerate and review the report"
        )


def verify_apply_preconditions(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    _verify_registry_snapshot(report)
    rows = report.get("rows")
    assert isinstance(rows, list)
    verified: list[dict[str, Any]] = []
    seen_zarrs: set[Path] = set()
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise SourceVideoMetadataApplyError(f"Row {index} is not an object")
        row = dict(raw_row)
        if row.get("disposition") not in {"eligible", "already_v2"}:
            raise SourceVideoMetadataApplyError(
                f"Row {index} is not eligible: {row.get('disposition')!r}"
            )
        if row.get("errors"):
            raise SourceVideoMetadataApplyError(f"Row {index} contains preflight errors")
        zarr_path = Path(str(row.get("zarr_path"))).expanduser().resolve()
        if zarr_path in seen_zarrs:
            raise SourceVideoMetadataApplyError(f"Duplicate Zarr target: {zarr_path}")
        seen_zarrs.add(zarr_path)
        root_updates = row.get("planned_root_updates")
        raw_updates = row.get("planned_raw_video_updates")
        if not isinstance(root_updates, Mapping) or not isinstance(raw_updates, Mapping):
            raise SourceVideoMetadataApplyError(f"Row {index} has invalid planned updates")
        if set(root_updates) - ALLOWED_ROOT_UPDATE_KEYS:
            raise SourceVideoMetadataApplyError(f"Row {index} has unauthorized root update keys")
        if set(raw_updates) - ALLOWED_RAW_UPDATE_KEYS:
            raise SourceVideoMetadataApplyError(f"Row {index} has unauthorized raw update keys")
        metadata = root_updates.get("source_video_metadata")
        if not isinstance(metadata, Mapping):
            raise SourceVideoMetadataApplyError(f"Row {index} has no v2 metadata object")
        locator = metadata.get("locator")
        if (
            metadata.get("schema_id") != SOURCE_VIDEO_METADATA_SCHEMA_ID
            or metadata.get("layout") != SOURCE_VIDEO_LAYOUT_SINGLE
            or not isinstance(locator, Mapping)
            or locator.get("kind") != SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE
        ):
            raise SourceVideoMetadataApplyError(f"Row {index} has an invalid v2 locator")
        root_file = _verify_file_precondition(
            row.get("root_metadata_precondition") or {}
        )
        raw_file = _verify_file_precondition(
            row.get("raw_video_metadata_precondition") or {}
        )
        source_video = _verify_source_video_precondition(
            row.get("source_video_precondition") or {}
        )
        if root_file != zarr_path / "zarr.json":
            raise SourceVideoMetadataApplyError(f"Unexpected root metadata path: {root_file}")
        if raw_file != zarr_path / "raw_video" / "zarr.json":
            raise SourceVideoMetadataApplyError(f"Unexpected raw metadata path: {raw_file}")
        if source_video != Path(str(row.get("source_video_path"))).resolve():
            raise SourceVideoMetadataApplyError(f"Unexpected source video path: {source_video}")
        json.dumps(root_updates, allow_nan=False, sort_keys=True)
        json.dumps(raw_updates, allow_nan=False, sort_keys=True)
        verified.append(
            {
                "index": index,
                "row": row,
                "zarr_path": zarr_path,
                "root_file": root_file,
                "raw_file": raw_file,
                "source_video": source_video,
            }
        )
    return verified


def _safe_name(value: Any) -> str:
    text = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(value)
    )
    return text[:120] or "dataset"


def _create_backup(
    targets: Sequence[Mapping[str, Any]],
    *,
    backup_dir: Path,
    report_bytes: bytes,
    report_sha256: str,
) -> tuple[dict[Path, Path], dict[str, Any]]:
    resolved_backup = backup_dir.expanduser().resolve()
    if resolved_backup.exists():
        raise FileExistsError(f"Backup directory already exists: {resolved_backup}")
    resolved_backup.mkdir(parents=True)
    backup_report = resolved_backup / "reviewed_preflight_report.json"
    backup_report.write_bytes(report_bytes)
    _fsync_file(backup_report)
    backups: dict[Path, Path] = {}
    files: list[dict[str, Any]] = []
    for target in targets:
        row = target["row"]
        index = int(target["index"])
        item_dir = resolved_backup / "metadata" / (
            f"{index:03d}_{_safe_name(row.get('dataset_id'))}"
        )
        item_dir.mkdir(parents=True)
        for label, source in (
            ("root_zarr.json", Path(target["root_file"])),
            ("raw_video_zarr.json", Path(target["raw_file"])),
        ):
            destination = item_dir / label
            shutil.copy2(source, destination)
            _fsync_file(destination)
            source_sha = _sha256_file(source)
            backup_sha = _sha256_file(destination)
            if source_sha != backup_sha:
                raise SourceVideoMetadataApplyError(
                    f"Backup verification failed: {source} -> {destination}"
                )
            backups[source] = destination
            files.append(
                {
                    "source_path": str(source),
                    "backup_path": str(destination),
                    "sha256": source_sha,
                    "size_bytes": source.stat().st_size,
                }
            )
    manifest = {
        "schema_id": "palette.source_video_metadata_backfill_backup.v1",
        "created_at_utc": _utc_now(),
        "report_sha256": report_sha256,
        "file_count": len(files),
        "files": files,
    }
    _atomic_write_json(resolved_backup / "backup_manifest.json", manifest)
    return backups, manifest


def _apply_attr_updates(path: Path, updates: Mapping[str, Any]) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("attributes"), dict):
        raise SourceVideoMetadataApplyError(f"Invalid Zarr v3 metadata document: {path}")
    attrs = payload["attributes"]
    changed = False
    for key, value in updates.items():
        if attrs.get(key) != value:
            attrs[key] = value
            changed = True
    if changed:
        _atomic_replace_bytes(path, _strict_json_bytes(payload))
    return changed


def _validate_applied_target(target: Mapping[str, Any]) -> dict[str, Any]:
    row = target["row"]
    root_attrs, _, root_state = read_attrs_strict(Path(target["zarr_path"]))
    raw_attrs, _, raw_state = read_attrs_strict(Path(target["zarr_path"]) / "raw_video")
    root_updates = row["planned_root_updates"]
    raw_updates = row["planned_raw_video_updates"]
    for key, expected in root_updates.items():
        if root_attrs.get(key) != expected:
            raise SourceVideoMetadataApplyError(
                f"Post-write root validation failed for {target['zarr_path']}: {key}"
            )
    for key, expected in raw_updates.items():
        if raw_attrs.get(key) != expected:
            raise SourceVideoMetadataApplyError(
                f"Post-write raw validation failed for {target['zarr_path']}: {key}"
            )
    resolved = resolve_source_video_from_attrs(
        root_attrs,
        raw_video_attrs=raw_attrs,
        zarr_path=Path(target["zarr_path"]),
        require_exists=True,
    )
    if resolved.path != Path(target["source_video"]):
        raise SourceVideoMetadataApplyError(
            f"Post-write resolver path mismatch for {target['zarr_path']}"
        )
    if (
        resolved.schema_id != SOURCE_VIDEO_METADATA_SCHEMA_ID
        or resolved.layout != SOURCE_VIDEO_LAYOUT_SINGLE
        or resolved.locator_kind != SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE
    ):
        raise SourceVideoMetadataApplyError(
            f"Post-write resolver contract mismatch for {target['zarr_path']}"
        )
    return {
        "dataset_id": row.get("dataset_id"),
        "zarr_path": str(target["zarr_path"]),
        "source_video_path": str(resolved.path),
        "root_metadata_sha256": root_state["sha256"],
        "raw_video_metadata_sha256": raw_state["sha256"],
        "schema_id": resolved.schema_id,
        "locator_kind": resolved.locator_kind,
        "status": "validated",
    }


def _restore_changed_files(
    changed_files: Sequence[Path],
    backups: Mapping[Path, Path],
) -> list[str]:
    restored: list[str] = []
    for path in reversed(changed_files):
        backup = backups[path]
        _atomic_replace_bytes(path, backup.read_bytes())
        if _sha256_file(path) != _sha256_file(backup):
            raise SourceVideoMetadataApplyError(f"Rollback verification failed: {path}")
        restored.append(str(path))
    return restored


def apply_reviewed_report(
    report_path: Path,
    *,
    expected_sha256: str,
    backup_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    report, report_bytes, report_sha256 = load_reviewed_report(
        report_path,
        expected_sha256=expected_sha256,
    )
    targets = verify_apply_preconditions(report)
    backups, backup_manifest = _create_backup(
        targets,
        backup_dir=backup_dir,
        report_bytes=report_bytes,
        report_sha256=report_sha256,
    )
    resolved_backup = backup_dir.expanduser().resolve()
    changed_files: list[Path] = []
    validations: list[dict[str, Any]] = []
    journal_path = resolved_backup / "apply_journal.json"
    started_at = _utc_now()
    try:
        for target in targets:
            row = target["row"]
            raw_file = Path(target["raw_file"])
            root_file = Path(target["root_file"])
            if _apply_attr_updates(raw_file, row["planned_raw_video_updates"]):
                changed_files.append(raw_file)
            if _apply_attr_updates(root_file, row["planned_root_updates"]):
                changed_files.append(root_file)
            validations.append(_validate_applied_target(target))
            _atomic_write_json(
                journal_path,
                {
                    "schema_id": APPLY_RECEIPT_SCHEMA_ID,
                    "status": "in_progress",
                    "started_at_utc": started_at,
                    "report_sha256": report_sha256,
                    "validated_count": len(validations),
                    "target_count": len(targets),
                    "changed_file_count": len(changed_files),
                },
            )
    except Exception as exc:
        restored = _restore_changed_files(changed_files, backups)
        failure_receipt = {
            "schema_id": APPLY_RECEIPT_SCHEMA_ID,
            "status": "rolled_back",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "report_path": str(report_path.expanduser().resolve()),
            "report_sha256": report_sha256,
            "backup_dir": str(resolved_backup),
            "error": f"{type(exc).__name__}: {exc}",
            "restored_files": restored,
        }
        _atomic_write_json(journal_path, failure_receipt)
        _atomic_write_json(receipt_path.expanduser().resolve(), failure_receipt)
        raise SourceVideoMetadataApplyError(
            f"Apply failed and changed files were rolled back: {exc}"
        ) from exc

    receipt = {
        "schema_id": APPLY_RECEIPT_SCHEMA_ID,
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "report_path": str(report_path.expanduser().resolve()),
        "report_sha256": report_sha256,
        "backup_dir": str(resolved_backup),
        "backup_manifest_sha256": _sha256_file(
            resolved_backup / "backup_manifest.json"
        ),
        "target_count": len(targets),
        "metadata_file_count_backed_up": backup_manifest["file_count"],
        "metadata_file_count_changed": len(changed_files),
        "changed_files": [str(path) for path in changed_files],
        "validations": validations,
    }
    _atomic_write_json(journal_path, receipt)
    _atomic_write_json(receipt_path.expanduser().resolve(), receipt)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--receipt-json", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Required acknowledgement for metadata mutation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.apply:
        raise SystemExit("Refusing mutation without --apply")
    receipt = apply_reviewed_report(
        args.report,
        expected_sha256=str(args.expected_report_sha256),
        backup_dir=args.backup_dir,
        receipt_path=args.receipt_json,
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "validations"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
