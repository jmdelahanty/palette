"""Unit tests for the registry-browser download-bundle plugin.

Covers the security and integrity logic that the live route can't exercise
(paths come from the DB, so a bad path can't be injected through HTTP).

Run:  scripts/py -m pytest docs/registry_browser/tests/test_download_bundle.py -q
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest

# The plugin lives in a sibling dir loaded by Datasette via --plugins-dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))
import download_bundle as db  # noqa: E402


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# --------------------------- resolve_allowlisted ---------------------------

def test_resolve_inside_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    f = root / "best.pt"
    f.write_bytes(b"x")
    assert db.resolve_allowlisted(str(f), [root.resolve()]) == f.resolve()


def test_resolve_outside_root_rejected(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"nope")
    assert db.resolve_allowlisted(str(outside), [root.resolve()]) is None


def test_resolve_symlink_escape_rejected(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"nope")
    link = root / "innocent.pt"
    link.symlink_to(secret)  # lives under root but realpath escapes it
    assert db.resolve_allowlisted(str(link), [root.resolve()]) is None


def test_resolve_missing_file(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    assert db.resolve_allowlisted(str(root / "absent.pt"), [root.resolve()]) is None


# ------------------------------ build_bundle -------------------------------

def _fixture_registry(tmp_path, rows):
    """rows: list of (kind, path, sha) -> minimal model_input_shapes table."""
    reg = tmp_path / "reg.sqlite"
    conn = sqlite3.connect(reg)
    conn.execute(
        "CREATE TABLE model_input_shapes (run_id TEXT, set_id TEXT, task_type TEXT, "
        "artifact_kind TEXT, artifact_path TEXT, artifact_sha256 TEXT)"
    )
    # tensorrt_models supplies the engine *description* in the manifest.
    conn.execute(
        "CREATE TABLE tensorrt_models (run_id TEXT, path TEXT, sha256 TEXT, precision TEXT, "
        "trt_version TEXT, cuda_version TEXT, compute_capability TEXT, gpu_name TEXT, "
        "file_size_bytes INTEGER, requires_plugins INTEGER, plugin_ops_json TEXT)"
    )
    for kind, path, sha in rows:
        conn.execute(
            "INSERT INTO model_input_shapes VALUES (?,?,?,?,?,?)",
            ("run1", "set1", "detect", kind, path, sha),
        )
        if kind == "tensorrt":
            conn.execute(
                "INSERT INTO tensorrt_models (run_id, path, sha256, precision) VALUES (?,?,?,?)",
                ("run1", path, sha, "fp16"),
            )
    conn.commit()
    conn.close()
    return reg


def test_unknown_run_returns_none(tmp_path):
    reg = _fixture_registry(tmp_path, [])
    assert db.build_bundle(str(reg), "missing", include_engine=False) is None


def test_bundle_includes_portable_excludes_engine(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    pt = root / "best.pt"; sha_pt = _write(pt, b"weights-bytes")
    onnx = root / "m.onnx"; sha_onnx = _write(onnx, b"onnx-bytes")
    eng = root / "m.engine"; sha_eng = _write(eng, b"engine-bytes")
    reg = _fixture_registry(tmp_path, [
        ("training", str(pt), sha_pt),
        ("onnx", str(onnx), sha_onnx),
        ("tensorrt", str(eng), sha_eng),
    ])
    monkeypatch.setenv("PALETTE_MODEL_ROOTS", str(root.resolve()))

    body, fname = db.build_bundle(str(reg), "run1", include_engine=False)
    names = zipfile.ZipFile(io.BytesIO(body)).namelist()
    assert "weights/best.pt" in names and "onnx/m.onnx" in names
    assert not any(n.startswith("tensorrt/") for n in names)  # excluded by default
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(body)).read("manifest.json"))
    assert manifest["tensorrt_engine"] is not None  # still described
    assert fname.endswith(".zip")

    # ...and with the flag the engine is shipped.
    body2, _ = db.build_bundle(str(reg), "run1", include_engine=True)
    names2 = zipfile.ZipFile(io.BytesIO(body2)).namelist()
    assert "tensorrt/m.engine" in names2


def test_sha_mismatch_omits_file(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    pt = root / "best.pt"; _write(pt, b"real-bytes")
    reg = _fixture_registry(tmp_path, [("training", str(pt), "deadbeef" * 8)])  # wrong sha
    monkeypatch.setenv("PALETTE_MODEL_ROOTS", str(root.resolve()))

    body, _ = db.build_bundle(str(reg), "run1", include_engine=False)
    z = zipfile.ZipFile(io.BytesIO(body))
    assert "weights/best.pt" not in z.namelist()  # corrupt file not shipped
    manifest = json.loads(z.read("manifest.json"))
    entry = next(f for f in manifest["files"] if f["kind"] == "training")
    assert entry["included"] is False and entry["reason"] == "sha256_mismatch"
    assert any("mismatch" in w for w in manifest["warnings"])


def test_outside_allowlist_omitted(tmp_path, monkeypatch):
    root = tmp_path / "models"; root.mkdir()
    outside = tmp_path / "elsewhere.pt"; sha = _write(outside, b"bytes")
    reg = _fixture_registry(tmp_path, [("training", str(outside), sha)])
    monkeypatch.setenv("PALETTE_MODEL_ROOTS", str(root.resolve()))

    body, _ = db.build_bundle(str(reg), "run1", include_engine=False)
    z = zipfile.ZipFile(io.BytesIO(body))
    assert "weights/best.pt" not in z.namelist()
    manifest = json.loads(z.read("manifest.json"))
    entry = next(f for f in manifest["files"] if f["kind"] == "training")
    assert entry["included"] is False and entry["reason"] == "unreachable_or_disallowed"
