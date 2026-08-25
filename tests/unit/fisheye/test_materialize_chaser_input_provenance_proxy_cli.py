from __future__ import annotations

import argparse
from pathlib import Path

import zarr

from fisheye.analysis_workflows.chaser_input_provenance_proxy_source_handle import (
    load_chaser_input_provenance_proxy_source_handle,
)
from fisheye.shared.zarr.chaser_input_provenance_proxy_schema import (
    CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH,
)
from fisheye.utils.materialize_chaser_input_provenance_proxy import run
from tests.unit.fisheye.test_provider_chaser_stimulus_source_handle import (
    _published_candidate,
)
from tests.unit.fisheye.test_stimulus_coordinate_v6_adapter import (
    _write_raw_chaser_h5,
    _write_v6_fixture,
)


def _args(
    archive: Path,
    source_run_name: str,
    *,
    scratch_root: Path,
    apply: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        analysis_zarr=archive,
        source_run_name=source_run_name,
        output_run_name="input_provenance_proxy_v1",
        scratch_root=scratch_root,
        expected_recording_id="recording-1",
        expected_source_manifest_sha256=None,
        copy_backend="python",
        apply=apply,
        json=True,
    )


def test_cli_dry_run_then_publish_and_strict_readback(tmp_path: Path) -> None:
    archive, source_run_name = _published_candidate(tmp_path)
    target = (
        archive
        / CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH
        / "input_provenance_proxy_v1"
    )

    planned = run(
        _args(
            archive,
            source_run_name,
            scratch_root=tmp_path / "scratch-plan",
            apply=False,
        )
    )

    assert planned["status"] == "planned_no_writes"
    assert planned["plan"]["selector_eligible"] is False
    assert planned["plan"]["selection"] == "none"
    assert not target.exists()

    published = run(
        _args(
            archive,
            source_run_name,
            scratch_root=tmp_path / "scratch-publish",
            apply=True,
        )
    )

    assert published["status"] == "published_selector_ineligible"
    assert published["publication"]["selector_eligible"] is False
    handle = load_chaser_input_provenance_proxy_source_handle(
        archive,
        run_name="input_provenance_proxy_v1",
        expected_recording_id="recording-1",
        expected_manifest_sha256=published["prepared_manifest_sha256"],
    )
    assert handle.selector_eligible is False
    assert handle.dimensions.n_frames == 3
    assert handle.dimensions.n_candidates == 4
    assert handle.dimensions.n_chasers == 2
    assert handle.acquisition_projection_record_sha256 == (
        published["acquisition_projection_record_sha256"]
    )
    handle.assert_current()


def test_cli_plans_from_exact_frame_bound_raw_h5_pair(tmp_path: Path) -> None:
    raw_artifact = _write_raw_chaser_h5(
        tmp_path / "citrus" / "session-1.h5"
    )
    companion = tmp_path / "session.stimulus_coordinate_v6.h5"
    _write_v6_fixture(companion, raw_h5_artifact=raw_artifact)
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(archive, mode="w")
    root.attrs["recording_id"] = "recording-42"

    planned = run(argparse.Namespace(
        analysis_zarr=archive,
        source_run_name=None,
        frame_bound_companion_h5=companion,
        recording_bundle_root=tmp_path,
        output_run_name="frame_bound_proxy_v1",
        scratch_root=tmp_path / "scratch",
        expected_recording_id="recording-42",
        expected_source_manifest_sha256=None,
        expected_camera_serial="CAM-42",
        expected_acquisition_camera_id="CAM-42",
        expected_shaman_numeric_camera_id=0,
        expected_source_total_frames=2,
        copy_backend="python",
        apply=False,
        json=True,
    ))

    assert planned["status"] == "planned_no_writes"
    assert planned["plan"]["selector_eligible"] is False
    assert planned["plan"]["selection"] == "none"
    assert planned["source_run_path"].endswith(
        "session-1.h5#/tracking_data/chaser_states"
    )
