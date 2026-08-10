from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis import detect_bouts_multi_level as bout_writer
from fisheye.diagnostics import build_activity_spatial_source_fixture as fixture
from fisheye.shared.zarr.benchmark_fixture import inventory_tree
from fisheye.shared.zarr_io import open_zarr_root

TRACK_RUN = "track_exact_v1"
HISTORICAL_RUN = "swim_historical_v8"
EXACT_RUN = "swim_exact_v8"


def _fixed_bytes(values: list[str], width: int) -> np.ndarray:
    result = np.zeros((len(values), width), dtype=np.uint8)
    for row, value in enumerate(values):
        encoded = value.encode("utf-8")
        result[row, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return result


def _parameters() -> dict[str, object]:
    return {
        "method": "peak_event",
        "min_bout_duration_s": 0.05,
        "min_gap_duration_s": 0.1,
        "min_gap_frames": None,
        "gap_merge_policy": "sampled_frame_gap",
        "min_peak_height_mm_s": None,
        "min_peak_prominence_mm_s": 4.0,
        "min_peak_distance_s": 0.1,
        "peak_width_rel_height": 0.98,
        "peak_event_boundary_mode": "relative_prominence_width",
        "shape_split_policy": "none",
        "default_level": "speed_exponential",
        "boundary_mode": "threshold",
        "boundary_window_s": 0.25,
        "exponential_tau_s": 0.025,
        "exponential_source_level": "speed_filtered",
        "layout": bout_writer.SWIM_BOUT_LAYOUT_COMPACT_V2,
        "frame_axis_storage": "reference",
    }


def _source_archive(path: Path) -> Path:
    root = open_zarr_root(path, mode="w")
    root.attrs["recording_id"] = "recording_fixture"
    dependencies = root.create_group("dependency_runs")
    dependency = dependencies.create_group("dep_v1")
    dependency.attrs["identity"] = "exact_dependency"
    dependency.create_array(
        "values",
        data=np.asarray([8, 13], dtype=np.int64),
        chunks=(2,),
    )
    crop_runs = root.create_group("crop_runs")
    source_crop = crop_runs.create_group("source_clip_v1")
    source_crop.create_array(
        "instance_key",
        data=np.asarray([21, 34], dtype=np.uint64),
        chunks=(2,),
    )
    analysis = root.create_group("analysis")
    tracks = analysis.create_group("track_kinematics_runs")
    offline = tracks.create_group("offline")
    track = offline.create_group(TRACK_RUN)
    track.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "track_motion_publication_manifest": {
                "schema_id": "palette.track_motion_publication_manifest",
                "schema_version": 2,
                "dependency_group_ref": "/dependency_runs/dep_v1@identity",
                "dependency_array_ref": "/dependency_runs/dep_v1/values",
                "external_path_ignored": "/groups/not_a_zarr_node",
                "legacy_collection_mapping": {
                    "source_proxy_crop_runs": ["source_clip_v1"]
                },
            },
        }
    )
    track.create_array(
        "sentinel",
        data=np.asarray([1, 2, 3], dtype=np.int32),
        chunks=(3,),
    )
    bouts = analysis.create_group("swim_bout_runs")
    historical = bouts.create_group(HISTORICAL_RUN)
    historical.attrs.update(
        {
            "schema_id": bout_writer.SWIM_BOUT_RUN_SCHEMA_ID,
            "schema_version": (
                bout_writer.SWIM_BOUT_RUN_SCHEMA_VERSION_FRAME_AXIS_REFERENCE
            ),
            "layout": bout_writer.SWIM_BOUT_STORED_LAYOUT_COMPACT_V2,
            "method_version": bout_writer.METHOD_VERSION,
            "source_track_kinematics_run": TRACK_RUN,
            "track_id": 0,
            "parameters": _parameters(),
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    historical.create_array(
        "values",
        data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        chunks=(3,),
    )
    indexes = historical.create_group("indexes")
    candidates = indexes.create_group("candidates")
    candidates.attrs["field_dtypes"] = {
        "candidate_name": "|S8",
        "parameters_json": "|S16",
    }
    candidates.create_array(
        "candidate_name",
        data=_fixed_bytes(["fixture"], 8),
        chunks=(1, 8),
    )
    candidates.create_array(
        "parameters_json",
        data=_fixed_bytes(['{"fixture":true}'], 16),
        chunks=(1, 16),
    )
    signals = indexes.create_group("signal_variants")
    signals.attrs["field_dtypes"] = {"parameters_json": "|S16"}
    signals.create_array(
        "parameters_json",
        data=_fixed_bytes(['{"signal":1}', '{"signal":2}'], 16),
        chunks=(2, 16),
    )
    return path


def test_destination_must_be_benchmark_fixture_namespace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".palette_benchmarks"):
        fixture._require_destination(tmp_path / "ordinary" / "fixture")


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("palette_run_completion_status", "running", "explicitly complete"),
        ("stage_selector_eligible", False, "selector eligible"),
    ],
)
def test_projection_requires_complete_eligible_historical_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        fixture,
        "git_identity",
        lambda: {
            "git_sha": "a" * 40,
            "git_short_sha": "a" * 8,
            "git_dirty": False,
            "git_root": str(tmp_path),
        },
    )
    source = _source_archive(tmp_path / "source.zarr")
    root = open_zarr_root(source, mode="a")
    root[f"analysis/swim_bout_runs/{HISTORICAL_RUN}"].attrs[attribute] = value
    destination = (
        tmp_path
        / ".palette_benchmarks"
        / "analytics_query_exports"
        / "fixtures"
        / "activity_rejected_v1"
    )

    with pytest.raises(ValueError, match=message):
        fixture.build_activity_spatial_source_fixture(
            source_zarr=source,
            source_track_run=TRACK_RUN,
            historical_swim_bout_run=HISTORICAL_RUN,
            exact_swim_bout_run=EXACT_RUN,
            destination=destination,
            work_root=tmp_path,
        )

    assert not destination.exists()


def test_builds_lossless_noncanonical_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_archive(tmp_path / "source.zarr")
    source_track = source / "analysis" / "track_kinematics_runs" / "offline" / TRACK_RUN
    source_bout = source / "analysis" / "swim_bout_runs" / HISTORICAL_RUN
    before = {
        "track": inventory_tree(source_track),
        "bout": inventory_tree(source_bout),
    }

    def fake_manifest_writer(run: object, **_kwargs: object) -> dict[str, object]:
        manifest = {"schema_id": "test.exact"}
        run.attrs["array_schema_manifest"] = manifest
        return manifest

    def fake_binding(
        _root: object,
        *,
        zarr_path: str | Path,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            binding={
                "payload_sha256": fixture.canonical_json_sha256(
                    {"zarr_path": str(Path(zarr_path).resolve())}
                )
            }
        )

    monkeypatch.setattr(
        fixture,
        "write_swim_bout_array_manifest",
        fake_manifest_writer,
    )
    monkeypatch.setattr(
        fixture,
        "build_swim_bout_columnar_field_dtypes",
        lambda: {
            "indexes/candidates": {
                "candidate_name": "|S256",
                "parameters_json": "|S8192",
            },
            "indexes/signal_variants": {"parameters_json": "|S8192"},
        },
    )
    monkeypatch.setattr(
        fixture,
        "validate_swim_bout_array_manifest",
        lambda _run, **_kwargs: [],
    )
    monkeypatch.setattr(
        fixture,
        "git_identity",
        lambda: {
            "git_sha": "a" * 40,
            "git_short_sha": "a" * 8,
            "git_dirty": False,
            "git_root": str(tmp_path),
        },
    )
    monkeypatch.setattr(fixture, "bind_activity_spatial_sources", fake_binding)

    destination = (
        tmp_path
        / ".palette_benchmarks"
        / "analytics_query_exports"
        / "fixtures"
        / "activity_exact_v1"
    )
    result = fixture.build_activity_spatial_source_fixture(
        source_zarr=source,
        source_track_run=TRACK_RUN,
        historical_swim_bout_run=HISTORICAL_RUN,
        exact_swim_bout_run=EXACT_RUN,
        destination=destination,
        work_root=tmp_path,
    )

    assert result["schema_id"] == fixture.FIXTURE_SCHEMA_ID
    assert result["payload"]["logical_array_equality"]["equal"] is True
    assert result["payload"]["logical_array_equality"]["array_count"] == 4
    assert result["payload"]["logical_array_equality"]["exact_array_count"] == 1
    assert result["payload"]["logical_array_equality"]["widened_array_count"] == 3
    assert result["payload"]["projection_contract"] == {
        "operation": "lossless_decoded_contract_attestation",
        "scientific_recomputation_performed": False,
        "source_authority_modified": False,
        "isolated_copy_manifest_attested": True,
        "byte_planner_adopted": False,
        "physical_profile_evidence_scope": (
            "separate_exact_tabular_candidate_benchmark"
        ),
    }
    assert (
        result["payload"]["exact_swim_bout"][
            "source_tree_copy_exact_before_attestation"
        ]
        is True
    )
    assert result["payload"]["source_nonmutation"]["unchanged"] is True
    assert result["payload"]["evidence_eligible"] is True
    assert result["payload"]["code_identity"]["git_sha"] == "a" * 40
    assert result["payload"]["production_state_changes"] == []
    assert (destination / fixture.MANIFEST_NAME).is_file()
    published = open_zarr_root(destination / fixture.ARCHIVE_NAME, mode="r")
    assert published.attrs["benchmark_only"] is True
    assert published.attrs["selector_eligible"] is False
    assert (
        published["analysis/swim_bout_runs"][EXACT_RUN].attrs["stage_selector_eligible"]
        is True
    )
    assert published["analysis/swim_bout_runs"].attrs["latest_complete"] == EXACT_RUN
    projected = published[f"analysis/swim_bout_runs/{EXACT_RUN}"]
    assert projected["indexes/candidates/candidate_name"].shape == (1, 256)
    assert projected["indexes/candidates/parameters_json"].shape == (1, 8192)
    assert projected["indexes/signal_variants/parameters_json"].shape == (2, 8192)
    np.testing.assert_array_equal(
        published["dependency_runs/dep_v1/values"][:],
        np.asarray([8, 13], dtype=np.int64),
    )
    dependency_projection = result["payload"]["source_track"]["dependency_projection"]
    assert dependency_projection["node_count"] == 3
    assert (
        dependency_projection["nodes"]["dependency_runs/dep_v1"]["node_type"] == "group"
    )
    assert (
        dependency_projection["nodes"]["dependency_runs/dep_v1/values"]["node_type"]
        == "array"
    )
    assert (
        dependency_projection["nodes"]["crop_runs/source_clip_v1"]["node_type"]
        == "group_tree"
    )
    np.testing.assert_array_equal(
        published["crop_runs/source_clip_v1/instance_key"][:],
        np.asarray([21, 34], dtype=np.uint64),
    )

    after = {
        "track": inventory_tree(source_track),
        "bout": inventory_tree(source_bout),
    }
    assert after["track"].tree_sha256 == before["track"].tree_sha256
    assert after["bout"].tree_sha256 == before["bout"].tree_sha256
