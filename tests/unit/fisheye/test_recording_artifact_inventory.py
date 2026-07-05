from __future__ import annotations

import json
import warnings
from pathlib import Path

import zarr
from zarr.errors import ZarrUserWarning

from fisheye.shared.plot_artifacts import write_png_visualization_artifact
from fisheye.shared.recording_artifact_inventory import _group_names, build_recording_artifact_inventory
from fisheye.shared.zarr_run_completion import COMPLETION_EPOCH_STRICT, mark_run_complete, require_runs_parent
from fisheye.utils import recording_artifact_inventory as cli


def _make_inventory_zarr(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "recording_id": "rec_001",
            "session_uuid": "session_001",
            "zarr_purpose": "analysis",
        }
    )

    detect_parent = require_runs_parent(root, "detect_runs", completion_epoch=COMPLETION_EPOCH_STRICT)
    detect_run = detect_parent.create_group("detect_001")
    write_png_visualization_artifact(
        detect_run,
        "detect_summary_png",
        b"\x89PNG\r\n\x1a\nfake-detect",
        description="Detection summary",
        created_by="test_recording_artifact_inventory",
        created_at_utc="2026-07-05T00:00:00+00:00",
        source_runs={"detect": "detect_001"},
    )
    quality_parent = detect_run.create_group("quality_reports")
    quality_run = quality_parent.create_group("quality_001")
    mark_run_complete(quality_run, parent_group=quality_parent, run_name="quality_001")
    mark_run_complete(detect_run, parent_group=detect_parent, run_name="detect_001")

    refined_parent = require_runs_parent(root, "refined_detect_runs", completion_epoch=COMPLETION_EPOCH_STRICT)
    refined_run = refined_parent.create_group("refined_001")
    refined_run.create_group("instances")
    refined_run.create_group("source_detections")
    mark_run_complete(refined_run, parent_group=refined_parent, run_name="refined_001")

    analysis = root.require_group("analysis")
    track_family = analysis.create_group("track_kinematics_runs")
    offline_parent = track_family.create_group("offline")
    track_run = offline_parent.create_group("tk_001")
    track_run.attrs["schema_id"] = "analysis.track_kinematics_runs"
    track_run.attrs["source_refined_keypoint_run"] = "refined_kp_001"
    mark_run_complete(track_run, parent_group=offline_parent, run_name="tk_001")

    acquisition = analysis.create_group("acquisition_video_streams")
    acquisition.attrs.update(
        {
            "schema_id": "palette.acquisition_video_streams.v1",
            "schema_version": 1,
            "inventory_status": "ok",
            "stream_count": 2,
            "stream_keys": ["crop", "full"],
            "crop_stream_available": True,
        }
    )
    streams = acquisition.create_group("streams")
    crop_stream = streams.create_group("crop")
    crop_stream.attrs.update(
        {
            "availability_status": "ok",
            "output_kind": "crop",
            "width": 348,
            "height": 348,
            "files": {
                "video": {"path": "derived/external_crop_recorder/crop.mp4", "exists": True},
                "metadata": {"path": "derived/external_crop_recorder/crop_meta.csv", "exists": True},
            },
        }
    )
    full_stream = streams.create_group("full")
    full_stream.attrs.update(
        {
            "availability_status": "ok",
            "output_kind": "full",
            "width": 4512,
            "height": 4512,
            "files": {
                "video": {"path": "cams/full.mp4", "exists": True},
            },
        }
    )
    return zarr_path


def _family_by_parent(inventory: dict[str, object], section: str, parent_path: str) -> dict[str, object]:
    families = inventory[section]
    assert isinstance(families, list)
    for family in families:
        assert isinstance(family, dict)
        if family.get("run_parent_path") == parent_path:
            return family
    raise AssertionError(f"missing family {parent_path!r} in {section}")


def test_build_recording_artifact_inventory_lists_runs_visualizations_and_sidecars(tmp_path: Path) -> None:
    zarr_path = _make_inventory_zarr(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="r")

    inventory = build_recording_artifact_inventory(root, zarr_path=zarr_path)

    assert inventory["schema_id"] == "palette.recording_artifact_inventory.v1"
    assert inventory["zarr_use"] == "analysis"
    assert inventory["root_attrs"]["recording_id"] == "rec_001"
    assert inventory["run_count"] == 4
    assert inventory["visualization_artifact_count"] == 1

    detect_family = _family_by_parent(inventory, "root_run_families", "detect_runs")
    assert detect_family["parent"]["resolved_latest_complete"] == "detect_001"
    detect_run = detect_family["runs"][0]
    assert detect_run["complete"] is True
    assert detect_run["completion_status"] == "complete"
    assert detect_run["visualization_count"] == 1
    assert detect_run["visualizations"][0]["artifact_name"] == "detect_summary_png"
    assert detect_run["visualizations"][0]["present"] is True
    assert detect_run["visualizations"][0]["media_type"] == "image/png"

    quality_family = _family_by_parent(
        inventory,
        "nested_report_families",
        "detect_runs/detect_001/quality_reports",
    )
    assert quality_family["family_kind"] == "nested_quality_reports"
    assert quality_family["runs"][0]["name"] == "quality_001"

    refined_family = _family_by_parent(inventory, "root_run_families", "refined_detect_runs")
    assert refined_family["run_count"] == 1
    assert refined_family["runs"][0]["name"] == "refined_001"
    assert refined_family["runs"][0]["group_count"] == 2

    track_family = _family_by_parent(
        inventory,
        "analysis_run_families",
        "analysis/track_kinematics_runs/offline",
    )
    assert track_family["family_path"] == "analysis/track_kinematics_runs"
    assert track_family["scope"] == "offline"
    assert track_family["runs"][0]["attrs"]["source_refined_keypoint_run"] == "refined_kp_001"

    acquisition = inventory["acquisition_video_streams"]
    assert acquisition["available"] is True
    assert acquisition["stream_keys"] == ["crop", "full"]
    assert acquisition["streams"][0]["stream_key"] == "crop"
    assert acquisition["streams"][0]["video_exists"] is True
    assert "recording_artifacts" in inventory["registry_projection_names"]["optional"]


def test_recording_artifact_inventory_cli_emits_json(tmp_path: Path, capsys) -> None:
    zarr_path = _make_inventory_zarr(tmp_path)

    assert cli.main([str(zarr_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["zarr_path"] == str(zarr_path)
    assert payload["root_attrs"]["recording_id"] == "rec_001"
    assert payload["run_family_count"] == 4


def test_inventory_suppresses_known_non_zarr_sidecar_warnings() -> None:
    class NoisyGroup:
        def group_keys(self):
            warnings.warn(
                "Object at logs is not recognized as a component of a Zarr hierarchy.",
                ZarrUserWarning,
                stacklevel=2,
            )
            return ["detect_runs"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _group_names(NoisyGroup()) == ["detect_runs"]

    assert caught == []
