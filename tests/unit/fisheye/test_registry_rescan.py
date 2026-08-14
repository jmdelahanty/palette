from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import fisheye.utils.registry_rescan as registry_rescan


class _FakeRegistry:
    def __init__(self, _path: Path, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False
        self.step_status_requests: list[Path] = []

    def scan_zarr(self, path: Path, *, include_step_status: bool = False) -> str:
        if self.fail:
            raise RuntimeError("scan failed")
        if include_step_status:
            self.step_status_requests.append(path)
        return f"dataset-{path.stem}"

    def reconcile_missing_datasets(self, *, scope_paths):
        return {"checked": len(scope_paths), "marked_missing": 0}

    def close(self) -> None:
        self.closed = True


def test_registry_rescan_writes_fail_closed_success_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    zarr_path.mkdir()
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_bytes(b"registry")
    result_path = tmp_path / "result.json"
    fake = _FakeRegistry(registry_path)
    monkeypatch.setattr(registry_rescan, "Registry", lambda _path: fake)
    monkeypatch.setattr(
        registry_rescan,
        "_iter_zarr",
        lambda roots, recursive: iter((zarr_path,)),
    )

    status = registry_rescan.main(
        [
            "--registry",
            str(registry_path),
            "--result-json",
            str(result_path),
            "--fail-on-error",
            str(zarr_path),
        ]
    )

    payload = json.loads(result_path.read_text())
    assert status == 0
    assert payload["status"] == "complete"
    assert payload["updated_count"] == 1
    assert payload["errors"] == []
    assert fake.closed is True


def test_registry_rescan_can_reconcile_step_status_in_same_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    zarr_path.mkdir()
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_bytes(b"registry")
    result_path = tmp_path / "result.json"
    fake = _FakeRegistry(registry_path)
    monkeypatch.setattr(registry_rescan, "Registry", lambda _path: fake)
    monkeypatch.setattr(
        registry_rescan,
        "_iter_zarr",
        lambda roots, recursive: iter((zarr_path,)),
    )

    status = registry_rescan.main(
        [
            "--registry",
            str(registry_path),
            "--result-json",
            str(result_path),
            "--fail-on-error",
            "--reconcile-step-status",
            str(zarr_path),
        ]
    )

    payload = json.loads(result_path.read_text())
    assert status == 0
    assert payload["recording_step_status_reconciled"] is True
    assert fake.step_status_requests == [zarr_path]


def test_registry_rescan_safe_shadow_publishes_with_full_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    zarr_path.mkdir()
    registry_path = tmp_path / "registry.sqlite"
    with sqlite3.connect(registry_path) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL);")
        connection.execute("INSERT INTO marker VALUES ('unchanged');")
        connection.commit()
    backup_path = tmp_path / "backups" / "registry.sqlite"
    result_path = tmp_path / "result.json"
    instances: list[_FakeRegistry] = []

    def registry_factory(path: Path) -> _FakeRegistry:
        fake = _FakeRegistry(path)
        instances.append(fake)
        return fake

    monkeypatch.setattr(registry_rescan, "Registry", registry_factory)
    monkeypatch.setattr(
        registry_rescan,
        "_iter_zarr",
        lambda roots, recursive: iter((zarr_path,)),
    )

    status = registry_rescan.main(
        [
            "--registry",
            str(registry_path),
            "--result-json",
            str(result_path),
            "--fail-on-error",
            "--reconcile-step-status",
            "--safe-shadow-publish",
            "--backup-path",
            str(backup_path),
            str(zarr_path),
        ]
    )

    payload = json.loads(result_path.read_text())
    assert status == 0
    assert payload["registry"] == str(registry_path.resolve())
    assert payload["registry_publication"]["publication_mode"] == (
        "local_shadow_copy_atomic_replace"
    )
    assert payload["registry_publication"]["published_validation"][
        "integrity_check"
    ] == "ok"
    assert backup_path.is_file()
    assert instances[0].step_status_requests == [zarr_path]


def test_registry_rescan_returns_nonzero_and_records_scan_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    zarr_path.mkdir()
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_bytes(b"registry")
    result_path = tmp_path / "result.json"
    fake = _FakeRegistry(registry_path, fail=True)
    monkeypatch.setattr(registry_rescan, "Registry", lambda _path: fake)
    monkeypatch.setattr(
        registry_rescan,
        "_iter_zarr",
        lambda roots, recursive: iter((zarr_path,)),
    )

    status = registry_rescan.main(
        [
            "--registry",
            str(registry_path),
            "--result-json",
            str(result_path),
            "--fail-on-error",
            str(zarr_path),
        ]
    )

    payload = json.loads(result_path.read_text())
    assert status == 1
    assert payload["status"] == "completed_with_errors"
    assert payload["errors"][0]["error_type"] == "RuntimeError"
    assert payload["errors"][0]["error"] == "scan failed"
    assert fake.closed is True
