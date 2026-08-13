from __future__ import annotations

import json
from pathlib import Path

import pytest

import fisheye.utils.registry_rescan as registry_rescan


class _FakeRegistry:
    def __init__(self, _path: Path, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False

    def scan_zarr(self, path: Path) -> str:
        if self.fail:
            raise RuntimeError("scan failed")
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
