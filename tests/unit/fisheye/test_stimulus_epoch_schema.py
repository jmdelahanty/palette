from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.stimulus_epoch_runs import (
    StimulusEpochResult,
    StimulusEpochWindow,
    write_stimulus_epoch_run,
)
from fisheye.analysis.stimulus_epoch_schema import (
    STIMULUS_EPOCH_LAYOUT,
    STIMULUS_EPOCH_RUN_SCHEMA_ID,
    STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
    validate_legacy_stimulus_epoch_source,
    validate_stimulus_epoch_array_manifest,
    write_stimulus_epoch_array_manifest,
)


def create_legacy_stimulus_epoch_archive(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"recording_id": "recording_1", "fps": 10.0, "total_frames": 30})
    stimulus = root.require_group("analysis").require_group("stimulus_runs").create_group(
        "stimulus_1"
    )
    stimulus.attrs.update(
        {
            "schema_id": "palette.stimulus.import.v1",
            "schema_version": 1,
            "run_name": "stimulus_1",
        }
    )
    events = stimulus.create_group("events")
    events.create_array(
        "camera_frame_id",
        data=np.asarray([0, 10, 20, 30], dtype=np.int64),
    )
    windows = tuple(
        StimulusEpochWindow(
            window_id=index,
            label=label,
            start_frame=start,
            end_frame=end,
            start_time_s=start / 10.0,
            end_time_s=(end + 1) / 10.0,
            duration_s=(end - start + 1) / 10.0,
            source_start_event_name=f"{label.upper()}_START",
            source_end_event_name=f"{label.upper()}_END",
            source_start_event_frame=start,
            source_end_event_frame=end + 1,
            source_policy="inclusive_start_exclusive_end_event_boundary",
        )
        for index, (label, start, end) in enumerate(
            (("pre_event", 0, 9), ("training_event", 10, 19), ("post_event", 20, 29))
        )
    )
    result = StimulusEpochResult(
        zarr_path=str(path),
        recording_id="recording_1",
        run_name="source",
        stimulus_run_name="stimulus_1",
        stimulus_path="analysis/stimulus_runs/stimulus_1",
        fps=10.0,
        total_frames=30,
        windows=windows,
        protocol_profile_id="test_profile",
        protocol_profile_version=1,
        protocol_profile_sha256="a" * 64,
        protocol_profile_source="test_profile.yaml",
        source_adapter_id="test_adapter",
        source_adapter_version=1,
        role_resolver_id="test_roles",
        role_resolver_version=1,
        window_policy_id="test_windows",
        window_policy_version=1,
    )
    write_stimulus_epoch_run(path, result)
    reopened = zarr.open_group(str(path), mode="a", use_consolidated=False)
    reopened["analysis/stimulus_epoch_runs/source"].attrs[
        "stage_selector_eligible"
    ] = True
    return reopened


def promote_fixture_to_exact_v2(run: zarr.Group) -> None:
    run.attrs.update(
        {
            "schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
            "schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
            "layout": STIMULUS_EPOCH_LAYOUT,
        }
    )
    write_stimulus_epoch_array_manifest(run, byte_planner_adopted=False)


def test_legacy_source_contract_accepts_current_writer_exactly(tmp_path: Path) -> None:
    root = create_legacy_stimulus_epoch_archive(tmp_path / "legacy.zarr")
    run = root["analysis/stimulus_epoch_runs/source"]

    assert validate_legacy_stimulus_epoch_source(run) == ()
    assert sorted(path for path, _array in _walk_arrays(run)) == [
        "windows/duration_s",
        "windows/end_frame",
        "windows/end_time_s",
        "windows/label_bytes",
        "windows/source_end_event_frame",
        "windows/source_end_event_name_bytes",
        "windows/source_policy_bytes",
        "windows/source_start_event_frame",
        "windows/source_start_event_name_bytes",
        "windows/start_frame",
        "windows/start_time_s",
        "windows/window_id",
    ]


def _walk_arrays(group: zarr.Group, prefix: str = ""):
    for name, array in group.arrays():
        yield (f"{prefix}/{name}" if prefix else name), array
    for name, child in group.groups():
        child_prefix = f"{prefix}/{name}" if prefix else name
        yield from _walk_arrays(child, child_prefix)


def test_v2_manifest_freezes_inventory_dtypes_shapes_and_semantics(
    tmp_path: Path,
) -> None:
    root = create_legacy_stimulus_epoch_archive(tmp_path / "exact.zarr")
    run = root["analysis/stimulus_epoch_runs/source"]
    promote_fixture_to_exact_v2(run)

    assert validate_stimulus_epoch_array_manifest(
        run, byte_planner_adopted=False
    ) == ()

    manifest = run.attrs["array_schema_manifest"]
    assert manifest["payload"]["run_schema_id"] == STIMULUS_EPOCH_RUN_SCHEMA_ID
    assert manifest["payload"]["dimensions"] == {"n_windows_rows": 3}
    assert len(manifest["payload"]["arrays"]) == 12
    assert manifest["payload"]["enabled_optional_bundles"] == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("overlap", "chronological and non-overlapping"),
        ("wrong_time", "start_time_s differs"),
        ("duplicate_id", "strictly increasing"),
        ("empty_label", r"label_bytes\[0\] is empty"),
    ],
)
def test_v2_semantics_fail_closed_after_manifest_rewrite(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    root = create_legacy_stimulus_epoch_archive(tmp_path / f"{mutation}.zarr")
    run = root["analysis/stimulus_epoch_runs/source"]
    promote_fixture_to_exact_v2(run)
    windows = run["windows"]
    if mutation == "overlap":
        windows["start_frame"][1] = np.int64(9)
        windows["start_time_s"][1] = np.float64(0.9)
        windows["duration_s"][1] = np.float64(1.1)
    elif mutation == "wrong_time":
        windows["start_time_s"][0] = np.float64(1.0)
    elif mutation == "duplicate_id":
        windows["window_id"][1] = np.int32(0)
    else:
        windows["label_bytes"][0, :] = np.uint8(0)

    with pytest.raises(ValueError, match=expected):
        write_stimulus_epoch_array_manifest(run, byte_planner_adopted=False)


def test_v2_rejects_unexpected_array_and_wrong_dtype(tmp_path: Path) -> None:
    root = create_legacy_stimulus_epoch_archive(tmp_path / "bad-inventory.zarr")
    run = root["analysis/stimulus_epoch_runs/source"]
    promote_fixture_to_exact_v2(run)
    run["windows"].create_array("frame_counts", data=np.ones(3, dtype=np.int32))

    errors = validate_stimulus_epoch_array_manifest(
        run, byte_planner_adopted=False
    )
    assert any("unexpected scientific arrays" in error for error in errors)

    del run["windows/frame_counts"]
    values = np.asarray(run["windows/start_frame"][:], dtype=np.int32)
    del run["windows/start_frame"]
    run["windows"].create_array("start_frame", data=values)
    errors = validate_stimulus_epoch_array_manifest(
        run, byte_planner_adopted=False
    )
    assert any("dtype mismatch" in error for error in errors)
