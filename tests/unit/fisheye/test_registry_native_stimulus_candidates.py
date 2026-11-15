"""Native stimulus candidates must never be read as legacy protocols.

Real public importer -> real local Zarr -> unmodified registry scan. A native
``unified_experimental_h5_v1`` run carries no legacy ``protocol_json``; reading
it as one manufactured a nameless, zero-step protocol row shared by every
native run.
"""

from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.analysis.import_stimulus_to_zarr import import_stimulus_to_zarr
from fisheye.registry.db import Registry
from fisheye.registry.extractors.chaser_metadata import (
    extract_recording_chaser_metadata,
)
from fisheye.registry.extractors.stimulus_metadata import extract_stimulus_metadata
from tests.unit.fisheye.test_registry_stimulus_metadata import _write_archive
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture, write_receipt


def _import_native(tmp_path: Path, destination: Path, run_name: str) -> None:
    import_stimulus_to_zarr(
        emit_fixture(tmp_path / "source", "base"),
        destination,
        run_name=run_name,
        overwrite=False,
        verbose=False,
        source_profile="unified_experimental_h5_v1",
        finalization_receipt=write_receipt(tmp_path / "source", "base"),
    )


def _mixed_archive(tmp_path: Path) -> Path:
    zarr_path = _write_archive(tmp_path)
    _import_native(tmp_path, zarr_path, "native_candidate")
    return zarr_path


def test_extractor_excludes_native_candidate_and_reports_it(tmp_path: Path) -> None:
    zarr_path = _mixed_archive(tmp_path)
    root = zarr.open_group(zarr_path, mode="r")

    extraction = extract_stimulus_metadata(
        root, zarr_path=zarr_path, recording_id="mixed"
    )

    assert [row["protocol_name"] for row in extraction.protocols] == ["Mixed canary"]
    assert [row["stimulus_run_id"] for row in extraction.recording_runs] == [
        "stimulus_001"
    ]
    assert {row["stimulus_run_id"] for row in extraction.recording_steps} == {
        "stimulus_001"
    }
    assert {row["stimulus_run_id"] for row in extraction.recording_modes} == {
        "stimulus_001"
    }
    assert extraction.unsupported_native_runs == ("native_candidate",)


def test_registry_scan_persists_no_protocol_for_native_candidate(
    tmp_path: Path,
) -> None:
    zarr_path = _mixed_archive(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.scan_zarr(zarr_path)
        protocols = registry.conn.execute(
            "SELECT protocol_name, step_count FROM stimulus_protocols"
        ).fetchall()
        runs = registry.conn.execute(
            """
            SELECT stimulus_run_id FROM recording_stimulus_runs
            WHERE dataset_id = ? ORDER BY stimulus_run_id
            """,
            (dataset_id,),
        ).fetchall()
    finally:
        registry.close()

    assert [tuple(row) for row in protocols] == [("Mixed canary", 3)]
    assert [row[0] for row in runs] == ["stimulus_001"]


def test_native_only_archive_extracts_nothing(tmp_path: Path) -> None:
    zarr_path = tmp_path / "native_only.zarr"
    _import_native(tmp_path, zarr_path, "native_candidate")
    root = zarr.open_group(zarr_path, mode="r")

    extraction = extract_stimulus_metadata(
        root, zarr_path=zarr_path, recording_id="native"
    )

    assert extraction.protocols == ()
    assert extraction.recording_runs == ()
    assert extraction.unsupported_native_runs == ("native_candidate",)


def test_any_declared_source_profile_is_not_read_as_legacy(tmp_path: Path) -> None:
    """Fail closed: legacy runs never declare a source profile."""

    zarr_path = _write_archive(tmp_path)
    root = zarr.open_group(zarr_path, mode="r+")
    future = root["analysis/stimulus_runs"].create_group("future_profile_run")
    future.attrs["source_profile"] = "some_later_profile_v2"
    future.attrs["protocol_json"] = json.dumps(
        {"protocol_name": "Looks legacy", "steps": []}
    )

    extraction = extract_stimulus_metadata(
        root, zarr_path=zarr_path, recording_id="mixed"
    )

    assert [row["protocol_name"] for row in extraction.protocols] == ["Mixed canary"]
    assert extraction.unsupported_native_runs == ("future_profile_run",)


def test_chaser_extractor_names_native_candidate_accurately(tmp_path: Path) -> None:
    zarr_path = _mixed_archive(tmp_path)
    root = zarr.open_group(zarr_path, mode="r")

    extraction = extract_recording_chaser_metadata(
        root, zarr_path=zarr_path, recording_id="mixed"
    )

    native_issues = [
        issue
        for issue in extraction.issues
        if issue.stimulus_run_id == "native_candidate"
    ]
    assert [issue.reason for issue in native_issues] == [
        "unsupported_native_stimulus_profile"
    ]
    assert all(row["stimulus_run_id"] != "native_candidate" for row in extraction.rows)
