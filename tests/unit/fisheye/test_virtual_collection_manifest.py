from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from fisheye.utils.virtual_collection_manifest import (
    VirtualCollectionManifestError,
    assert_valid_manifest,
    compute_manifest_sha256,
    main,
    validate_manifest,
    verify_manifest_sha256,
    with_manifest_sha256,
)


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "examples"
    / "virtual_collection_manifest_v1.example.json"
)


def _example_manifest() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def test_example_virtual_collection_manifest_is_structurally_valid() -> None:
    manifest = _example_manifest()

    assert validate_manifest(manifest) == []
    assert_valid_manifest(manifest)


def test_manifest_sha256_excludes_only_manifest_sha256() -> None:
    manifest = _example_manifest()

    digest = compute_manifest_sha256(manifest)
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)

    changed_hash_field = copy.deepcopy(manifest)
    changed_hash_field["manifest_sha256"] = "different_placeholder"
    assert compute_manifest_sha256(changed_hash_field) == digest

    changed_content = copy.deepcopy(manifest)
    changed_content["collection_id"] = "different_collection"
    assert compute_manifest_sha256(changed_content) != digest


def test_with_manifest_sha256_populates_verifiable_hash() -> None:
    manifest = _example_manifest()

    assert not verify_manifest_sha256(manifest)

    populated = with_manifest_sha256(manifest)
    assert verify_manifest_sha256(populated)
    assert populated["manifest_sha256"] == compute_manifest_sha256(populated)


def test_manifest_sha256_includes_locator_at_selection() -> None:
    manifest = _example_manifest()
    changed_locator = copy.deepcopy(manifest)
    changed_locator["records"][0]["locator_at_selection"]["uri"] = (
        "s3://cold-storage/palette/example_analysis.zarr"
    )

    assert compute_manifest_sha256(changed_locator) != compute_manifest_sha256(manifest)


def test_manifest_hash_normalizes_unicode_to_nfc() -> None:
    manifest_a = _example_manifest()
    manifest_b = copy.deepcopy(manifest_a)
    manifest_a["collection_name"] = "Cafe\u0301 cohort"
    manifest_b["collection_name"] = "Caf\u00e9 cohort"

    assert compute_manifest_sha256(manifest_a) == compute_manifest_sha256(manifest_b)


def test_manifest_hash_rejects_nonfinite_numbers() -> None:
    manifest = _example_manifest()
    manifest["records"][0]["recording_attrs"]["bad_float"] = math.nan

    with pytest.raises(VirtualCollectionManifestError, match="non-finite"):
        compute_manifest_sha256(manifest)

    errors = validate_manifest(manifest)
    assert any("non-finite" in error for error in errors)


def test_validate_manifest_rejects_ambiguous_empty_source_run() -> None:
    manifest = _example_manifest()
    manifest["records"][0]["source_runs"]["eye_angle_run"] = {}

    errors = validate_manifest(manifest)

    assert any("eye_angle_run.present" in error for error in errors)


def test_validate_manifest_rejects_invalid_registry_snapshot_status() -> None:
    manifest = _example_manifest()
    manifest["query"]["registry_snapshot_status"] = "maybe"

    errors = validate_manifest(manifest)

    assert any("registry_snapshot_status" in error for error in errors)


def test_validate_manifest_rejects_absent_run_with_concrete_path() -> None:
    manifest = _example_manifest()
    tail = manifest["records"][0]["source_runs"]["tail_kinematics_run"]
    tail["path"] = "analysis/tail_kinematics_runs/tail_test"

    errors = validate_manifest(manifest)

    assert any("tail_kinematics_run.path" in error for error in errors)


def test_cli_validate_accepts_example_manifest(capsys) -> None:
    rc = main(["validate", str(EXAMPLE_PATH)])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "valid"
    assert captured.err == ""


def test_cli_hash_prints_manifest_digest(capsys) -> None:
    manifest = _example_manifest()

    rc = main(["hash", str(EXAMPLE_PATH)])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == compute_manifest_sha256(manifest)


def test_cli_stamp_writes_verifiable_manifest(tmp_path: Path, capsys) -> None:
    output = tmp_path / "collection.manifest.json"

    rc = main(["stamp", str(EXAMPLE_PATH), "--output", str(output)])

    captured = capsys.readouterr()
    assert rc == 0
    stamped = json.loads(output.read_text(encoding="utf-8"))
    assert stamped["manifest_sha256"] == captured.out.strip()
    assert verify_manifest_sha256(stamped)


def test_cli_stamp_refuses_to_overwrite_by_default(tmp_path: Path, capsys) -> None:
    output = tmp_path / "collection.manifest.json"
    output.write_text("{}", encoding="utf-8")

    rc = main(["stamp", str(EXAMPLE_PATH), "--output", str(output)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "already exists" in captured.err


def test_cli_validate_check_hash_rejects_placeholder_hash(capsys) -> None:
    rc = main(["validate", str(EXAMPLE_PATH), "--check-hash"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest_sha256" in captured.err
