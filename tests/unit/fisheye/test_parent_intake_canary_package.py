"""The reusable canary must refuse unsafe targets before any subprocess/write."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.canaries.parent_intake import safety


@pytest.fixture
def synthetic_root():
    root = safety.new_fixture_root()
    try:
        (root / "source/context").mkdir(parents=True)
        (root / "delivery").mkdir()
        (root / "delivery/payload.bin").write_bytes(b"synthetic")
        (root / "source/fixture_notice.txt").write_text("SYNTHETIC TEST FIXTURE\n")
        (root / "source/context/synthetic_fixture_provenance.json").write_text(
            json.dumps(
                {
                    "synthetic_session_id": safety.SESSION_ID,
                    "scientific_admission": False,
                    "orange_encoder_evidence": False,
                    "source_clock_evidence": "synthetic_unqualified",
                }
            )
        )
        (root / "producer_smoke_report.json").write_text(
            json.dumps(
                {
                    "status": "actual_pinned_citrus_transfer_and_loadback_passed",
                    "fixture_scope": "synthetic_Orange_envelopes_real_H264_no_scientific_admission",
                    "work_dir": str(root),
                    "source": str(root / "source"),
                    "destination": str(root / "delivery"),
                }
            )
        )
        yield root
    finally:
        shutil.rmtree(root)


def test_valid_fresh_synthetic_tree_is_read_only_validated(synthetic_root):
    before = (synthetic_root / "producer_smoke_report.json").read_bytes()
    path, report = safety.validate_synthetic_fixture(synthetic_root)
    assert path == synthetic_root and report["work_dir"] == str(path)
    assert (synthetic_root / "producer_smoke_report.json").read_bytes() == before


@pytest.mark.parametrize(
    "target",
    [
        Path("/"),
        Path("/tmp"),
        Path("/tmp/not-synthetic"),
        Path("/tmp/a/../palette-citrus-encoded-transfer-test"),
    ],
)
def test_broad_or_noncanonical_output_refused_without_creation(target):
    with pytest.raises(ValueError):
        safety.new_fixture_root(target)


def test_existing_output_refused_without_overwrite(synthetic_root):
    before = (synthetic_root / "delivery/payload.bin").read_bytes()
    with pytest.raises(FileExistsError):
        safety.new_fixture_root(synthetic_root)
    assert (synthetic_root / "delivery/payload.bin").read_bytes() == before


@pytest.mark.parametrize(
    "kind", ["symlink-file", "symlink-directory", "hardlink", "fifo"]
)
def test_external_or_special_payload_refused(synthetic_root, tmp_path, kind):
    outside = tmp_path / "unrelated.bin"
    outside.write_bytes(b"never modify")
    unsafe = synthetic_root / "delivery/unsafe"
    if kind == "symlink-file":
        unsafe.symlink_to(outside)
    elif kind == "symlink-directory":
        unsafe.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "hardlink":
        os.link(outside, unsafe)
    else:
        os.mkfifo(unsafe)
    with pytest.raises(ValueError, match="private regular"):
        safety.validate_synthetic_fixture(synthetic_root)
    assert outside.read_bytes() == b"never modify"


def test_root_alias_refused(synthetic_root, tmp_path):
    alias = tmp_path / "fixture-alias"
    alias.symlink_to(synthetic_root, target_is_directory=True)
    with pytest.raises(ValueError):
        safety.validate_synthetic_fixture(alias)


@pytest.mark.parametrize("mutation", ["path", "notice", "scientific", "clock"])
def test_wrong_scope_or_source_evidence_refused(synthetic_root, mutation):
    report_path = synthetic_root / "producer_smoke_report.json"
    provenance_path = (
        synthetic_root / "source/context/synthetic_fixture_provenance.json"
    )
    if mutation == "path":
        report = json.loads(report_path.read_bytes())
        report["source"] = "/production/recording"
        report_path.write_text(json.dumps(report))
    elif mutation == "notice":
        (synthetic_root / "source/fixture_notice.txt").write_text("real data")
    else:
        provenance = json.loads(provenance_path.read_bytes())
        provenance[
            "scientific_admission"
            if mutation == "scientific"
            else "source_clock_evidence"
        ] = True
        provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        safety.validate_synthetic_fixture(synthetic_root)


def test_source_pin_mismatch_refused(tmp_path):
    source = tmp_path / "producer.py"
    source.write_text("test bytes\n")
    expected = {"producer.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert safety.validate_source_pins(tmp_path, expected) == expected
    source.write_text("changed bytes\n")
    with pytest.raises(ValueError, match="Pinned Citrus"):
        safety.validate_source_pins(tmp_path, expected)


@pytest.mark.parametrize("limit", ["MAX_FIXTURE_FILES", "MAX_FIXTURE_BYTES"])
def test_fixture_budget_refuses_before_payload_read(synthetic_root, monkeypatch, limit):
    monkeypatch.setattr(safety, limit, 1)
    with pytest.raises(ValueError, match="budget"):
        safety.validate_synthetic_fixture(synthetic_root)


@pytest.mark.parametrize("missing", [False, True])
def test_stimulus_builder_preserves_supported_positive_and_negative_grammar(
    synthetic_root, missing
):
    import h5py
    from tests.canaries.parent_intake.stimulus_fixture import write_stimulus_h5

    path = synthetic_root / "source/synthetic_context_02010093.h5"
    evidence = write_stimulus_h5(
        path,
        camera="02010093",
        session_id=safety.SESSION_ID,
        start_ns=1700000000000000000,
        frame_ns=500000000,
        missing_frames=missing,
    )
    assert (
        evidence["protocol_semantic_hash"]
        == "sha256:eb01e71e7d79c5163d53c59d234feff4002238f4a0b52a21a538cd3a020d52b3"
    )
    assert evidence["metadata_only_bypass_requested"] is False
    with h5py.File(path, "r") as h5:
        assert ("video_metadata/frame_metadata" not in h5) == missing
        assert h5["events"]["camera_frame_id"].tolist() == [1, 2, 3, 3]
        assert (
            h5["stimulus_renderer_snapshot"].attrs["schema_id"]
            == "citrus.stimulus_renderer_snapshot"
        )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_stimulus_h5(
            path,
            camera="02010093",
            session_id=safety.SESSION_ID,
            start_ns=1700000000000000000,
            frame_ns=500000000,
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("module", ["generate_fixture", "run"])
def test_optimized_cli_refuses_before_mutation(module):
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            f"tests.canaries.parent_intake.{module}",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "unoptimized Python" in result.stderr


def test_archived_reports_retain_exact_historical_bytes():
    evidence = (
        Path(__file__).resolve().parents[3]
        / "docs/diagnostics/parent_intake_synthetic_canary_2026-09-07/evidence"
    )
    expected = {
        "full_stimulus_positive.json": "9fbb835670b159272f1b901989e3a3a98ba684912f970583555ea3a23d8cc1c0",
        "full_stimulus_negative.json": "c32a3d1ceece8fb8ebabe069f543651adcca9307c58a703d90eb37029d6ee442",
    }
    for name, digest in expected.items():
        payload = (evidence / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        assert (
            json.loads(payload)["palette_commit"]
            == "a5a6ff6dd83df3f49a8f1ed7108ea6846ef3a298"
        )
    manifest = evidence.parent / "SHA256SUMS"
    indexed = set()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        artifact = manifest.parent / relative
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
        indexed.add(artifact.resolve())
    assert indexed == {path.resolve() for path in evidence.rglob("*") if path.is_file()}
