"""Ordinary current-profile clipped import, without patched runtime validators."""

from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.registry.db import Registry
from fisheye.registry.recording_identity_authority import (
    RecordingIdentityAuthorityError,
    load_verified_recording_import_receipt,
)
from fisheye.shared import run_provenance
from fisheye.shared.acquisition_crop_stream_ledger import (
    validate_current_acquisition_crop_stream_ledger,
)
from fisheye.shared.acquisition_frame_clock import resolve_acquisition_frame_clock
from fisheye.shared.acquisition_video_streams import (
    build_acquisition_video_stream_inventory,
)
from fisheye.shared.recording_import_receipt import recording_import_receipt_paths
from fisheye.shared.source_recording_identity import SOURCE_RECORDING_IDENTITY_PROFILE
from fisheye.shared.zarr.crop_pixel_authority import bind_crop_pixel_authority
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.utils import import_recording_analysis as importer
from fisheye.utils import import_organized_recordings_analysis as batch_importer
from fisheye.utils.build_recording_frame_index import build_recording_frame_index
from tests.unit.fisheye.test_clipped_recording_import_receipt import (
    CAMERA,
    GIT_SHA,
    _encoded_clip,
)


def _write_current_clipped_recording(
    tmp_path: Path,
    *,
    crop: bool = False,
    h5: bool = False,
    absolute_index_paths: bool = False,
) -> Path:
    recording = tmp_path / "recordings" / "recording-clipped"
    rows = []
    for clip_index in range(2):
        row = _encoded_clip(recording / "raw" / "acquisition", clip_index)
        for field in (
            "clip_directory",
            "video_path",
            "metadata_path",
            "keyframe_path",
            "clip_manifest_path",
        ):
            row[field] = "raw/acquisition/" + row[field]
        interval = {
            "first_recording_frame_id": row["first_recording_frame_id"],
            "last_recording_frame_id": row["last_recording_frame_id"],
            "frame_count": 2,
        }
        outputs = {
            "full": {
                **interval,
                "output_kind": "full",
                "video": row["video_path"],
                "metadata": row["metadata_path"],
                "keyframes": row["keyframe_path"],
            }
        }
        if crop:
            crop_row = _encoded_clip(
                recording / "raw" / "crop", clip_index, width=16, height=16
            )
            crop_meta = recording / "raw" / "crop" / crop_row["metadata_path"]
            first = row["first_recording_frame_id"]
            crop_meta.write_text(
                "recording_frame_id,has_detection,blank_frame,crop_x,crop_y,crop_w,crop_h,"
                "detection_x,detection_y,detection_w,detection_h,crop_video_frame_index,"
                "session_crop_video_frame_index\n"
                f"{first},1,0,10,12,16,16,12,14,4,4,0,{first - 1}\n"
                f"{first + 1},0,1,,,,,,,,,1,{first}\n"
            )
            outputs["crop"] = {
                **interval,
                "output_kind": "crop",
                "video": "raw/crop/" + crop_row["video_path"],
                "metadata": crop_meta.relative_to(recording).as_posix(),
            }
        (recording / row["clip_manifest_path"]).write_text(
            json.dumps(
                {
                    "schema_id": "palette.transfer_organized_clip_projection.v1",
                    "clip_id": row["clip_id"],
                    "clip_index": clip_index,
                    "recording_outputs": {CAMERA: outputs},
                }
            )
        )
        if absolute_index_paths:
            for field in (
                "clip_directory",
                "video_path",
                "metadata_path",
                "keyframe_path",
                "clip_manifest_path",
            ):
                row[field] = str((recording / row[field]).resolve())
        rows.append(row)
    index = {
        "schema_id": "palette.orange_external_ipc_recording_clip_index.v1",
        "mode": "rolling_clips",
        "recording_backend_mode": "rolling_clips",
        "recording_id": "recording-clipped",
        "session_id": "session-clipped",
        "rows": rows,
        "camera_ranges": {CAMERA: {"clip_count": 2, "total_frame_count": 4}},
    }
    scratch_index = recording / "recording_clip_index.json"
    scratch_index.write_text(json.dumps(index))
    derived = recording / "derived" / "recording_frame_index"
    build_recording_frame_index(
        recording,
        output_parquet=derived / "recording_frame_index.parquet",
        output_manifest=derived / "recording_frame_index_manifest.json",
    )
    scratch_index.rename(derived / "recording_clip_index.json")
    manifest = {
        "source_recording_identity_profile": SOURCE_RECORDING_IDENTITY_PROFILE,
        "recording_id": "recording-clipped",
        "session_uuid": "session-clipped",
        "camera_id": CAMERA,
        "recording_type": "behavior",
        "recording_subtype": "free",
        "behavior_mode": "free",
        "artifact_schema_id": "behavior_v1",
        "source_layout": "rolling_clips",
        "rolling_clip_streams": {
            "schema_id": "palette.orange_rolling_clip_streams.v1",
            "frame_clock": "recording_frame_id",
            "source_profile": "rolling_clip_collection_v1",
            "output_kinds": ["crop", "full"] if crop else ["full"],
            "recording_clip_index": "derived/recording_frame_index/recording_clip_index.json",
            "recording_frame_index": "derived/recording_frame_index/recording_frame_index.parquet",
            "recording_frame_index_manifest": "derived/recording_frame_index/recording_frame_index_manifest.json",
        },
    }
    if h5:
        h5_path = recording / "raw" / "acquisition" / "protocol.h5"
        with h5py.File(h5_path, "w"):
            pass
        manifest["h5_relative_path"] = h5_path.relative_to(recording).as_posix()
    (recording / "recording_manifest.json").write_text(json.dumps(manifest))
    return recording


def _options() -> importer.RecordingImportOptions:
    return importer.RecordingImportOptions(
        import_video_metadata=True,
        video_metadata_overwrite=False,
        import_stimulus=False,
        stimulus_always=False,
        stimulus_run_name=None,
        stimulus_overwrite=False,
        stimulus_quiet=True,
    )


def _stub_only_checkout_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = lambda **_: {"git_sha": GIT_SHA, "git_dirty": False}
    monkeypatch.setattr(importer, "git_identity", identity)
    monkeypatch.setattr(run_provenance, "git_identity", identity)


@pytest.mark.parametrize("crop,h5", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("absolute_index_paths", [False, True])
def test_real_current_clipped_import_to_receipt_registry_and_pixel_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crop: bool,
    h5: bool,
    absolute_index_paths: bool,
) -> None:
    recording = _write_current_clipped_recording(
        tmp_path, crop=crop, h5=h5, absolute_index_paths=absolute_index_paths
    )
    _stub_only_checkout_identity(monkeypatch)
    plan = importer.resolve_single_recording_plan(
        recording_dir=recording, require_h5=False
    )
    assert plan.recording_layout == "clipped_video_collection"
    assert plan.cam_video is None
    assert (plan.h5_path is not None) is h5
    result = importer.process_recording_import(plan, _options())
    assert result.ok, result
    assert result.receipt is not None
    assert load_verified_recording_import_receipt(plan.zarr_path) == result.receipt
    root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=True)
    assert root.attrs["camera_id"] == CAMERA
    assert root.attrs["source_layout"] == "rolling_clips"
    assert root.attrs.get("source_video_path") is None
    assert root.attrs.get("source_video") is None
    assert "images_full" not in root["raw_video"]
    assert "analysis_metadata" not in root
    assert root.attrs["acquisition_video_stream_count"] == (2 if crop else 1)
    assert root.attrs["acquisition_crop_video_available"] is crop
    assert root.attrs["acquisition_crop_ledger_available"] is crop
    clock = resolve_acquisition_frame_clock(root, required=True)
    assert clock.row_count == 4
    assert (
        clock.record["source"]["locator"]
        == "derived/recording_frame_index/recording_frame_index.parquet"
    )
    np.testing.assert_array_equal(
        root[clock.group_path]["parent_frame_index"][:], np.arange(4)
    )
    np.testing.assert_array_equal(
        root[clock.group_path]["recording_frame_id"][:], [1, 2, 3, 4]
    )
    pixel = bind_crop_pixel_authority(
        plan.zarr_path,
        expected_recording_identity="recording-clipped",
        expected_camera_identity=CAMERA,
        expected_n_frames=4,
        expected_source_width=64,
        expected_source_height=48,
    )
    assert pixel.source_video_path is None
    assert len(pixel.source_video_paths) == 2
    if crop:
        ledger = validate_current_acquisition_crop_stream_ledger(root)
        run = root[
            "analysis/acquisition_video_streams/streams/crop/" + ledger.group_path
        ]
        assert ledger.row_count == 4 and ledger.blank_row_count == 2
        np.testing.assert_array_equal(
            run["source_crop_video_frame_indices"][:], [0, 1, 0, 1]
        )
        np.testing.assert_array_equal(
            run["source_session_crop_video_frame_indices"][:], [0, 1, 2, 3]
        )
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        verified = registry.finalize_current_source_import(
            zarr_path=plan.zarr_path, receipt=result.receipt, decided_by="test"
        )
        assert (
            registry.read_verified_recording_import_by_path(plan.zarr_path) == verified
        )
    finally:
        registry.close()
    before = {
        p.relative_to(plan.zarr_path): p.read_bytes()
        for p in plan.zarr_path.rglob("*")
        if p.is_file()
    }
    sealed = importer.process_recording_import(plan, _options())
    assert not sealed.ok and sealed.failed_step == "recording_import_sealed"
    assert {
        p.relative_to(plan.zarr_path): p.read_bytes()
        for p in plan.zarr_path.rglob("*")
        if p.is_file()
    } == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing_index",
        "escaping_index",
        "null_rolling",
        "single_and_rolling",
        "missing_output_kinds",
        "crop_only",
        "unknown_output",
        "duplicate_output",
        "wrong_plan_layout",
        "first_clip_standin",
        "wrong_h5",
        "malformed_clock",
        "wrong_full_role",
        "bad_keyframe_count",
        "absolute_declared_index",
        "absolute_projected_video",
        "absolute_index_row_outside",
        "absolute_index_row_symlink",
    ],
)
def test_current_clipped_import_refuses_wrong_sources_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    recording = _write_current_clipped_recording(tmp_path)
    _stub_only_checkout_identity(monkeypatch)
    plan = importer.resolve_single_recording_plan(
        recording_dir=recording, require_h5=False
    )
    path = recording / "recording_manifest.json"
    manifest = json.loads(path.read_text())
    rolling = manifest["rolling_clip_streams"]
    index_path = recording / rolling["recording_clip_index"]
    index = json.loads(index_path.read_text())
    first_video = recording / index["rows"][0]["video_path"]
    if damage == "missing_index":
        rolling["recording_frame_index"] = "missing.parquet"
    elif damage == "escaping_index":
        rolling["recording_clip_index"] = "../outside.json"
    elif damage == "null_rolling":
        manifest["rolling_clip_streams"] = None
    elif damage == "single_and_rolling":
        manifest["video_streams"] = {
            "streams": {
                "full": {"video": first_video.relative_to(recording).as_posix()}
            }
        }
    elif damage == "missing_output_kinds":
        del rolling["output_kinds"]
    elif damage == "crop_only":
        rolling["output_kinds"] = ["crop"]
    elif damage == "unknown_output":
        rolling["output_kinds"] = ["full", "preview"]
    elif damage == "duplicate_output":
        rolling["output_kinds"] = ["full", "full"]
    elif damage == "wrong_plan_layout":
        plan = replace(plan, recording_layout="single_video", cam_video=first_video)
    elif damage == "first_clip_standin":
        plan = replace(plan, cam_video=first_video)
    elif damage == "wrong_h5":
        plan = replace(plan, h5_path=recording / "unbound.h5")
    elif damage == "malformed_clock":
        frame_index = recording / rolling["recording_frame_index"]
        table = pq.read_table(frame_index)
        table = table.set_column(
            table.schema.get_field_index("timestamp"),
            "timestamp",
            pa.array([1300, 1200, 1100, 1000], type=pa.int64()),
        )
        pq.write_table(table, frame_index)
    elif damage == "wrong_full_role":
        clip_manifest = recording / index["rows"][0]["clip_manifest_path"]
        payload = json.loads(clip_manifest.read_text())
        payload["recording_outputs"][CAMERA]["full"]["video"] = index["rows"][1][
            "video_path"
        ]
        clip_manifest.write_text(json.dumps(payload))
    elif damage == "bad_keyframe_count":
        keyframe = recording / index["rows"][0]["keyframe_path"]
        payload = json.loads(keyframe.read_text())
        payload["total_frames"] = 99
        keyframe.write_text(json.dumps(payload))
    elif damage == "absolute_declared_index":
        rolling["recording_clip_index"] = str(index_path)
    elif damage == "absolute_projected_video":
        clip_manifest = recording / index["rows"][0]["clip_manifest_path"]
        payload = json.loads(clip_manifest.read_text())
        payload["recording_outputs"][CAMERA]["full"]["video"] = str(first_video)
        clip_manifest.write_text(json.dumps(payload))
    elif damage in {"absolute_index_row_outside", "absolute_index_row_symlink"}:
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(first_video.read_bytes())
        selected = outside
        if damage == "absolute_index_row_symlink":
            selected = recording / "outside-alias.mp4"
            selected.symlink_to(outside)
        index["rows"][0]["video_path"] = str(selected)
        index_path.write_text(json.dumps(index))
    path.write_text(json.dumps(manifest))
    result = importer.process_recording_import(plan, _options())
    assert not result.ok
    assert result.receipt is None
    assert recording_import_receipt_paths(plan.zarr_path) == ()


def test_current_full_only_inventory_does_not_invent_crop_stream(
    tmp_path: Path,
) -> None:
    recording = _write_current_clipped_recording(tmp_path)
    manifest = json.loads((recording / "recording_manifest.json").read_text())
    inventory = build_acquisition_video_stream_inventory(recording, manifest)
    assert inventory["stream_keys"] == ["full"]
    assert inventory["crop_stream_available"] is False
    assert inventory["streams"]["full"]["contract"]["frame_count"] == 4


def test_clipped_plan_h5_is_explicit_not_discovered_from_raw_glob(
    tmp_path: Path,
) -> None:
    recording = _write_current_clipped_recording(tmp_path)
    (recording / "raw" / "unrelated.h5").write_bytes(b"not selected")
    plan = importer.resolve_single_recording_plan(
        recording_dir=recording, require_h5=False
    )
    assert plan.h5_path is None
    with pytest.raises(ValueError, match="H5"):
        importer.resolve_single_recording_plan(recording_dir=recording, require_h5=True)


def test_real_batch_import_discovers_and_propagates_clipped_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _write_current_clipped_recording(tmp_path, crop=True)
    _stub_only_checkout_identity(monkeypatch)
    discovered = batch_importer.discover_recording_dirs(
        recording.parent,
        recursive=False,
        import_stimulus=False,
    )
    assert discovered == [recording]
    plans = batch_importer.build_plans(
        discovered, import_stimulus=False, skip_existing=True, check_stimulus=False
    )
    assert len(plans) == 1 and plans[0].status == "ok"
    assert plans[0].recording_layout == "clipped_video_collection"
    assert plans[0].cam_video is None
    assert (
        batch_importer.main(
            [
                str(recording.parent),
                "--apply",
                "--recording-only",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        == 0
    )
    receipt = load_verified_recording_import_receipt(plans[0].zarr_path)
    skipped = batch_importer.build_plans(
        discovered, import_stimulus=False, skip_existing=True, check_stimulus=False
    )
    assert skipped[0].status == "skipped"
    assert skipped[0].recording_layout == "clipped_video_collection"
    assert load_verified_recording_import_receipt(skipped[0].zarr_path) == receipt


def test_declared_crop_ledger_is_required_after_real_current_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _write_current_clipped_recording(tmp_path, crop=True)
    _stub_only_checkout_identity(monkeypatch)
    plan = importer.resolve_single_recording_plan(
        recording_dir=recording, require_h5=False
    )
    result = importer.process_recording_import(plan, _options())
    assert result.ok, result
    root = zarr.open_group(str(plan.zarr_path), mode="r+", use_consolidated=False)
    del root["analysis/acquisition_video_streams/streams/crop"]
    root.attrs["acquisition_crop_ledger_available"] = False
    consolidate_metadata_capture_expected_warnings(str(plan.zarr_path))
    with pytest.raises(
        RecordingIdentityAuthorityError, match="no acquisition crop stream"
    ):
        load_verified_recording_import_receipt(plan.zarr_path)
