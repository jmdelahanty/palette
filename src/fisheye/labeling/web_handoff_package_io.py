"""User-handoff package file I/O helpers for web labeling."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from .web_launch_bundle_files import _write_directory_checksums


def _refresh_handoff_directory_checksums(
    *,
    path: Path,
    operator: str,
    reason: str,
) -> dict[str, object]:
    if not path.is_dir():
        raise ValueError(f"Checksum refresh requires a handoff or launch bundle directory: {path}")
    checksums_path = path / "checksums.json"
    if not checksums_path.is_file():
        raise FileNotFoundError(f"Refusing to create new checksums for a directory without checksums.json: {path}")
    previous_bytes, previous_sha256 = _sha256_file(checksums_path)
    previous_payload = json.loads(checksums_path.read_text(encoding="utf-8"))
    previous_count = int(previous_payload.get("count") or 0) if isinstance(previous_payload, Mapping) else 0
    refreshed_at_utc = datetime.now(timezone.utc).isoformat()
    refresh_entry = {
        "schema": "palette.web_labeling_checksum_refresh_event.v1",
        "refreshed_at_utc": refreshed_at_utc,
        "operator": str(operator or ""),
        "reason": str(reason or ""),
        "package_path": str(path),
        "previous_checksums_bytes": previous_bytes,
        "previous_checksums_sha256": previous_sha256,
        "previous_file_count": previous_count,
    }
    refresh_log_path = path / "checksums-refresh-log.jsonl"
    refresh_log_path.parent.mkdir(parents=True, exist_ok=True)
    with refresh_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(refresh_entry, sort_keys=True) + "\n")
    payload = _write_directory_checksums(path, checksums_path)
    payload["refresh"] = refresh_entry
    checksums_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "schema": "palette.web_labeling_checksum_refresh_report.v1",
        "path": str(path),
        "checksums": str(checksums_path),
        "refresh_log": str(refresh_log_path),
        "refresh": refresh_entry,
        "count": payload.get("count", 0),
        "files": payload.get("files", []),
    }



def _safe_checksum_relative_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    rel = Path(text)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    return rel



def _sha256_file(path: Path) -> tuple[int, str]:
    import hashlib

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()



def _sha256_bytes(data: bytes) -> tuple[int, str]:
    import hashlib

    return len(data), hashlib.sha256(data).hexdigest()



def _verify_directory_checksums(path: Path) -> dict[str, object]:
    checksums_path = path / "checksums.json"
    if not checksums_path.exists():
        return {"present": False, "ok": True, "checked": 0, "missing": [], "mismatched": [], "unsafe_paths": []}
    payload = json.loads(checksums_path.read_text(encoding="utf-8"))
    rows = payload.get("files", []) if isinstance(payload, dict) else []
    missing: list[str] = []
    mismatched: list[dict[str, object]] = []
    unsafe_paths: list[str] = []
    checked = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        rel = _safe_checksum_relative_path(row.get("path"))
        if rel is None:
            unsafe_paths.append(str(row.get("path") or ""))
            continue
        file_path = path / rel
        if not file_path.is_file():
            missing.append(rel.as_posix())
            continue
        actual_bytes, actual_sha256 = _sha256_file(file_path)
        expected_bytes = int(row.get("bytes") or -1)
        expected_sha256 = str(row.get("sha256") or "")
        checked += 1
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            mismatched.append(
                {
                    "path": rel.as_posix(),
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual_sha256,
                }
            )
    return {
        "present": True,
        "ok": not missing and not mismatched and not unsafe_paths,
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
        "unsafe_paths": unsafe_paths,
    }



def _verify_zip_checksums(path: Path) -> dict[str, object]:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        checksum_names = sorted(name for name in names if name.endswith("/checksums.json"))
        if not checksum_names:
            return {"present": False, "ok": True, "checked": 0, "missing": [], "mismatched": [], "unsafe_paths": []}
        checksum_name = checksum_names[0]
        checksum_prefix = checksum_name.rsplit("/", 1)[0] + "/"
        payload = json.loads(archive.read(checksum_name).decode("utf-8"))
        rows = payload.get("files", []) if isinstance(payload, dict) else []
        missing: list[str] = []
        mismatched: list[dict[str, object]] = []
        unsafe_paths: list[str] = []
        checked = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            rel = _safe_checksum_relative_path(row.get("path"))
            if rel is None:
                unsafe_paths.append(str(row.get("path") or ""))
                continue
            archive_name = checksum_prefix + rel.as_posix()
            if archive_name not in names:
                missing.append(rel.as_posix())
                continue
            actual_bytes, actual_sha256 = _sha256_bytes(archive.read(archive_name))
            expected_bytes = int(row.get("bytes") or -1)
            expected_sha256 = str(row.get("sha256") or "")
            checked += 1
            if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
                mismatched.append(
                    {
                        "path": rel.as_posix(),
                        "expected_bytes": expected_bytes,
                        "actual_bytes": actual_bytes,
                        "expected_sha256": expected_sha256,
                        "actual_sha256": actual_sha256,
                    }
                )
    return {
        "present": True,
        "ok": not missing and not mismatched and not unsafe_paths,
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
        "unsafe_paths": unsafe_paths,
    }



def _verify_handoff_checksums(path: Path) -> dict[str, object]:
    if path.is_dir():
        return _verify_directory_checksums(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        return _verify_zip_checksums(path)
    return {"present": False, "ok": True, "checked": 0, "missing": [], "mismatched": [], "unsafe_paths": []}



def _load_handoff_documents(path: Path) -> tuple[str, dict[str, object] | None, list[dict[str, object]]]:
    if path.is_dir():
        launch_manifest_path = path / "manifest.json"
        launch_handoffs_index_path = path / "handoffs" / "index.json"
        if launch_manifest_path.exists() and launch_handoffs_index_path.exists():
            launch_manifest = json.loads(launch_manifest_path.read_text(encoding="utf-8"))
            manifests = [
                json.loads(manifest.read_text(encoding="utf-8"))
                for manifest in sorted((path / "handoffs").glob("*/manifest.json"))
            ]
            return "launch", launch_manifest, manifests
        index_path = path / "index.json"
        manifest_path = path / "manifest.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            manifests = [
                json.loads(manifest.read_text(encoding="utf-8"))
                for manifest in sorted(path.glob("*/manifest.json"))
            ]
            return "batch", index, manifests
        if manifest_path.exists():
            return "user", None, [json.loads(manifest_path.read_text(encoding="utf-8"))]
        raise FileNotFoundError(f"No handoff index.json or manifest.json found under {path}")

    if path.is_file() and path.suffix.lower() == ".zip":
        index: dict[str, object] | None = None
        launch_manifest: dict[str, object] | None = None
        manifests: list[dict[str, object]] = []
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/index.json"):
                    payload = json.loads(archive.read(name).decode("utf-8"))
                    if isinstance(payload, dict) and "handoffs" in payload:
                        index = payload
                elif name.endswith("/manifest.json"):
                    payload = json.loads(archive.read(name).decode("utf-8"))
                    if isinstance(payload, dict) and "readiness_ok" in payload and "handoffs_ok" in payload:
                        launch_manifest = payload
                    elif isinstance(payload, dict) and payload.get("user"):
                        manifests.append(payload)
        if launch_manifest is not None:
            return "launch_zip", launch_manifest, manifests
        if index is not None:
            return "batch_zip", index, manifests
        if manifests:
            return "user_zip", None, manifests[:1]
        raise FileNotFoundError(f"No handoff index.json or manifest.json found in {path}")

    raise FileNotFoundError(f"Handoff path does not exist or is not a supported directory/ZIP: {path}")


