from hashlib import sha256
from pathlib import Path
import sqlite3

import pytest

from fisheye.registry.db import Registry
from fisheye.registry import recording_identity_authority as authority
from tests.unit.fisheye.test_clipped_recording_import_receipt import (
    _publish_clipped_import,
)


def test_read_only_registry_admission_reuses_owner_without_schema_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, receipt = _publish_clipped_import(tmp_path, monkeypatch)
    path = tmp_path / "registry with spaces.sqlite"
    registry = Registry(path)
    try:
        expected = registry.finalize_current_source_import(
            zarr_path=output, receipt=receipt, decided_by="test"
        )
    finally:
        registry.close()
    before = (sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    # Constructor denial proves the wrapper cannot initialize/migrate a registry.
    monkeypatch.setattr(
        Registry,
        "__init__",
        lambda *_a, **_k: pytest.fail("read-only admission initialized Registry"),
    )
    actual = authority.load_verified_registry_recording_import(
        registry_path=path, zarr_path=output
    )
    assert actual == expected
    assert (sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize(
    "damage", ["missing", "empty_schema", "unbound", "stale_source"]
)
def test_read_only_registry_admission_refuses_without_database_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    output, receipt = _publish_clipped_import(tmp_path, monkeypatch)
    path = tmp_path / "registry.sqlite"
    if damage == "empty_schema":
        sqlite3.connect(path).close()
    elif damage != "missing":
        registry = Registry(path)
        try:
            if damage == "stale_source":
                registry.finalize_current_source_import(
                    zarr_path=output, receipt=receipt, decided_by="test"
                )
        finally:
            registry.close()
    if damage == "stale_source":
        index = output.parent.parent / "recording_clip_index.json"
        index.write_text(index.read_text() + " ")
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(authority.RecordingIdentityAuthorityError):
        authority.load_verified_registry_recording_import(
            registry_path=path, zarr_path=output
        )
    assert (path.read_bytes() if path.exists() else None) == before
