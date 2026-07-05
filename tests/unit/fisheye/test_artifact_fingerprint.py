from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from fisheye.shared import artifact_fingerprint as mod
from fisheye.shared.artifact_fingerprint import (
    CONTENT_FINGERPRINT_SCHEME,
    MANIFEST_FINGERPRINT_SCHEME,
    fingerprint_artifact,
    fingerprint_directory_manifest,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fingerprint_artifact_computes_content_hash_and_sidecar(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    payload = b"model-weights"
    model.write_bytes(payload)

    result = fingerprint_artifact(model, role="detect_model")

    assert result["role"] == "detect_model"
    assert result["path"] == str(model.resolve())
    assert result["fingerprint_scheme"] == CONTENT_FINGERPRINT_SCHEME
    assert result["sha256"] == _sha256(payload)
    assert result["source"] == "computed"
    sidecar = model.with_name(f"{model.name}.{CONTENT_FINGERPRINT_SCHEME}.json")
    assert json.loads(sidecar.read_text(encoding="utf-8"))["sha256"] == _sha256(payload)


def test_fingerprint_artifact_uses_sidecar_when_stat_matches(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    payload = b"same"
    model.write_bytes(payload)
    first = fingerprint_artifact(model, role="detect_model")

    second = fingerprint_artifact(model, role="detect_model")

    assert first["sha256"] == second["sha256"]
    assert second["source"] == "sidecar"


def test_fingerprint_artifact_recomputes_stale_sidecar(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"old")
    fingerprint_artifact(model, role="detect_model")
    model.write_bytes(b"new-content")
    os.utime(model, None)

    result = fingerprint_artifact(model, role="detect_model")

    assert result["sha256"] == _sha256(b"new-content")
    assert result["source"] == "computed"


def test_fingerprint_artifact_treats_unparseable_sidecar_as_cache_miss(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    sidecar = model.with_name(f"{model.name}.{CONTENT_FINGERPRINT_SCHEME}.json")
    sidecar.write_text("{not json", encoding="utf-8")

    result = fingerprint_artifact(model, role="detect_model")

    assert result["sha256"] == _sha256(b"weights")
    assert result["source"] == "computed"


def test_fingerprint_artifact_trusts_registry_hash_when_stat_matches(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    current = model.stat()

    result = fingerprint_artifact(
        model,
        role="detect_model",
        registry_hash="A" * 64,
        registry_stat={"size_bytes": current.st_size, "mtime_ns": current.st_mtime_ns},
    )

    assert result["sha256"] == "a" * 64
    assert result["source"] == "registry"


def test_fingerprint_artifact_records_registry_hash_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")

    with pytest.warns(RuntimeWarning, match="does not match registry hash"):
        result = fingerprint_artifact(model, role="detect_model", registry_hash="b" * 64)

    assert result["sha256"] == _sha256(b"weights")
    assert result["registry_sha256"] == "b" * 64
    assert result["mismatch"] is True
    assert result["source"] == "computed"


def test_fingerprint_artifact_tolerates_sidecar_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")

    def _raise_replace(_src: Path, _dst: Path) -> None:
        raise PermissionError("read-only sidecar dir")

    monkeypatch.setattr(mod.os, "replace", _raise_replace)

    result = fingerprint_artifact(model, role="detect_model")

    assert result["sha256"] == _sha256(b"weights")
    assert result["source"] == "computed"


def test_fingerprint_artifact_records_missing_file_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"

    with pytest.warns(RuntimeWarning, match="Could not fingerprint"):
        result = fingerprint_artifact(missing, role="detect_model")

    assert result["role"] == "detect_model"
    assert result["fingerprint_scheme"] is None
    assert "error" in result


def test_fingerprint_directory_manifest_hashes_manifest_not_file_contents(tmp_path: Path) -> None:
    runtime = tmp_path / "sam3" / "model"
    runtime.mkdir(parents=True)
    first = runtime / "a.txt"
    second = runtime / "sub" / "b.txt"
    second.parent.mkdir()
    first.write_text("a", encoding="utf-8")
    second.write_text("bb", encoding="utf-8")

    result = fingerprint_directory_manifest(runtime, role="sam3_runtime")

    assert result["role"] == "sam3_runtime"
    assert result["fingerprint_scheme"] == MANIFEST_FINGERPRINT_SCHEME
    assert result["identity_kind"] == "runtime_manifest"
    assert result["manifest_entry_count"] == 2
    assert result["size_bytes"] == 3
    assert len(result["sha256"]) == 64
