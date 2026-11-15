"""frame_identity_v1 (agent-contracts #87): old and new Orange names, never both."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fisheye.shared import acquisition_crop_stream_ledger as ledger
from fisheye.shared.orange_frame_identity import (
    FrameIdentityError,
    crop_meta_stream_frame_field,
    gop_position_field,
    keyframe_field_name,
    keyframe_video_frame_indices,
    require_frame_identity_names,
)
from fisheye.utils import consolidate_external_ipc_rolling_recordings as consolidate
from fisheye.utils import repair_trimmed_video_sidecars as repair
from fisheye.utils.materialize_orange_style_clips import _clip_keyframe_payload
from fisheye.utils.plan_orange_style_clips import load_keyframe_summary
from tests.unit.fisheye.test_acquisition_crop_stream_ledger import _fixture


def test_crop_meta_field_is_old_or_new_never_both():
    assert crop_meta_stream_frame_field(["recording_frame_id", "local_frame_id"], label="x") == "local_frame_id"
    assert crop_meta_stream_frame_field(["recording_frame_id", "stream_frame_id"], label="x") == "stream_frame_id"
    assert crop_meta_stream_frame_field(["recording_frame_id"], label="x") is None
    with pytest.raises(FrameIdentityError, match="no aliases"):
        crop_meta_stream_frame_field(["stream_frame_id", "local_frame_id"], label="x")


def test_an_adopted_file_may_not_keep_an_unsuffixed_frame_reference():
    with pytest.raises(FrameIdentityError, match="without a counter suffix: frame_id"):
        crop_meta_stream_frame_field(["recording_frame_id", "stream_frame_id", "frame_id"], label="x")
    with pytest.raises(FrameIdentityError, match="payload_frame_id"):
        require_frame_identity_names(["recording_frame_id", "payload_frame_id"], label="x")
    # Positions and the four counters are fine.
    require_frame_identity_names(
        ["recording_frame_id", "stream_frame_id", "camera_frame_id", "start_stimulus_frame_num",
         "crop_video_frame_index", "session_crop_video_frame_index", "display_frame_index"],
        label="x",
    )


def test_an_old_file_keeps_its_old_names():
    # Before adoption a bare frame_id alias is legitimate (section 8).
    assert crop_meta_stream_frame_field(["recording_frame_id", "frame_id", "local_frame_id"], label="x") == "local_frame_id"


def test_gop_position_field():
    assert gop_position_field(["gop_index", "frame_index_within_gop"], label="x") == "frame_index_within_gop"
    assert gop_position_field(["recording_frame_id", "stream_frame_id", "position_in_gop"], label="x") == "position_in_gop"
    with pytest.raises(FrameIdentityError):
        gop_position_field(["position_in_gop", "frame_index_within_gop"], label="x")


def test_keyframe_lists():
    assert keyframe_video_frame_indices({"keyframe_frames": [0, 25]}, label="k") == [0, 25]
    assert keyframe_video_frame_indices({"keyframe_video_frame_indices": [0, 25]}, label="k") == [0, 25]
    assert keyframe_field_name({"keyframe_video_frame_indices": [0]}, label="k") == "keyframe_video_frame_indices"
    for bad in ({}, {"keyframe_frames": [0], "keyframe_video_frame_indices": [0]}):
        with pytest.raises(FrameIdentityError):
            keyframe_video_frame_indices(bad, label="k")


def _parse(tmp_path: Path, rename: dict[str, str]):
    recording_dir, _manifest = _fixture(tmp_path)
    metadata = recording_dir / "derived" / "external_crop_recorder" / "crop_meta.csv"
    header, *rows = metadata.read_text().splitlines(keepends=True)
    for old, new in rename.items():
        header = header.replace(old, new)
    metadata.write_text(header + "".join(rows))
    return ledger._parse_crop_metadata(metadata, width=384, height=384)


def test_the_crop_ledger_reads_the_stream_counter_under_either_name(tmp_path):
    before = _parse(tmp_path / "old", {})
    after = _parse(tmp_path / "new", {"local_frame_id": "stream_frame_id"})
    assert set(before.arrays) == set(after.arrays)
    for name in before.arrays:
        np.testing.assert_array_equal(before.arrays[name], after.arrays[name], err_msg=name)
    assert before.code_maps == after.code_maps
    np.testing.assert_array_equal(after.arrays["source_crop_local_frame_ids"], [10, 11])


def test_the_crop_ledger_refuses_both_names(tmp_path):
    with pytest.raises(ValueError, match="no aliases"):
        _parse(tmp_path, {"camera_frame_id": "stream_frame_id"})  # now both local_ and stream_


def test_rolling_consolidation_recognizes_the_renamed_full_metadata(tmp_path):
    for name, header in (
        ("old.csv", "recording_frame_id,timestamp,timestamp_sys,frame_id,local_frame_id,gop_index,frame_index_within_gop,bytes\n"),
        ("new.csv", "recording_frame_id,timestamp,timestamp_sys,stream_frame_id,gop_index,position_in_gop,source_gpu_id,assigned_gpu_id,assigned_shard_id,bytes\n"),
    ):
        path = tmp_path / name
        path.write_text(header + "1,100,90,10,0,0,1000\n")
        assert consolidate._frame_clock_kind(path) == "full", name


def _keyframe_json(tmp_path: Path, field: str) -> Path:
    path = tmp_path / f"{field}.json"
    path.write_text(json.dumps({"total_frames": 100, "fps": 100.0, field: [0, 25, 50, 75]}))
    return path


@pytest.mark.parametrize("field", ["keyframe_frames", "keyframe_video_frame_indices"])
def test_keyframe_readers_and_writers_keep_the_source_name(tmp_path, field):
    summary = load_keyframe_summary(_keyframe_json(tmp_path, field))
    np.testing.assert_array_equal(summary.keyframe_frames, [0, 25, 50, 75])
    clip = _clip_keyframe_payload(
        source_payload=summary.payload,
        source_keyframes=summary.keyframe_frames,
        row={"actual_start_frame": 25, "end_frame_exclusive": 75, "frame_count": 50},
    )
    other = ({"keyframe_frames", "keyframe_video_frame_indices"} - {field}).pop()
    assert clip[field] == [0, 25] and other not in clip


# --- pinned v3 schemas (Orange 2ef8840) -----------------------------------

from fisheye.shared import orange_realtime_products as rp  # noqa: E402
from fisheye.shared import recording_transfer_snapshot as rts  # noqa: E402
from fisheye.shared.recording_transfer_snapshot import TransferSnapshotError  # noqa: E402
from tests.unit.fisheye.test_frame_identity_proof import passed_proof  # noqa: E402

REALTIME = Path(__file__).resolve().parents[2] / "fixtures" / "orange_realtime_products_v2"
LOGS = {"detections": "Cam2010093_yolo_events.jsonl", "pose": "Cam2010093_pose_events.jsonl"}


def _proof_v3(frames: int) -> dict:
    proof = passed_proof(frames)
    proof.pop("legacy_aliases")
    proof["schema_version"] = 3
    return proof


def test_a_v3_proof_is_admitted_and_a_v2_proof_still_is():
    for proof in (passed_proof(4), _proof_v3(4)):
        rts._require_frame_identity_proof("Cam.summary.json", proof, output_kind="full", frame_count=4)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda p: p.update(legacy_aliases={"frame_id": "recording_frame_id"}), "grammar"),
        (lambda p: p.update(schema_version=4), "supported semantic proof validator"),
        (lambda p: p["video_binding"].update(metadata_rows=3), "does not bind"),
    ],
)
def test_a_v3_proof_is_checked_exactly(edit, message):
    proof = _proof_v3(4)
    edit(proof)
    with pytest.raises(TransferSnapshotError, match=message):
        rts._require_frame_identity_proof("Cam.summary.json", proof, output_kind="full", frame_count=4)


def test_a_v2_proof_without_its_aliases_is_refused():
    proof = passed_proof(4)
    proof.pop("legacy_aliases")
    with pytest.raises(TransferSnapshotError, match="grammar"):
        rts._require_frame_identity_proof("Cam.summary.json", proof, output_kind="full", frame_count=4)


def _to_v3(line: dict) -> dict:
    line = json.loads(json.dumps(line))
    line["schema_version"] = 3
    frame = line.get("frame")
    if isinstance(frame, dict):
        frame["stream_frame_id"] = frame.pop("local_frame_id")
        frame.pop("ipc_frame_id", None)
    return line


@pytest.mark.parametrize("product", sorted(LOGS))
def test_a_v3_event_log_scans_like_its_v2_original(tmp_path, product):
    source = REALTIME / LOGS[product]
    v3 = tmp_path / LOGS[product]
    v3.write_text("".join(json.dumps(_to_v3(json.loads(raw))) + "\n" for raw in source.read_text().splitlines()))
    before = rp.scan_event_log(source, product=product, line_schema_version=2)
    after = rp.scan_event_log(v3, product=product, line_schema_version=3)
    assert (after.line_schema_version, after.validation) == (3, "every_line_v3")
    for field in ("frame_rows", "header_rows", "first_recording_frame_id", "last_recording_frame_id",
                  "rows_by_status", "rows_by_kind"):
        assert getattr(after, field) == getattr(before, field), field
    # A v2 line in a v3 log is refused, and vice versa.
    with pytest.raises(rp.RealtimeProductsError, match="violates"):
        rp.scan_event_log(source, product=product, line_schema_version=3)
    with pytest.raises(rp.RealtimeProductsError, match="violates"):
        rp.scan_event_log(v3, product=product, line_schema_version=2)


from fisheye.shared.acquisition_frame_clock import (  # noqa: E402
    AcquisitionFrameClockError,
    parse_acquisition_frame_clock_csv,
)

NEW_META = "recording_frame_id,timestamp,timestamp_sys,stream_frame_id,gop_index,position_in_gop,source_gpu_id,assigned_gpu_id,assigned_shard_id,bytes\n"
OLD_META = "recording_frame_id,timestamp,timestamp_sys,frame_id,local_frame_id,gop_index,frame_index_within_gop,bytes\n"


def test_the_clock_parser_reads_old_and_new_recorder_metadata_alike(tmp_path):
    new = tmp_path / "new.csv"
    new.write_text(NEW_META + "1,1000,900,501,0,0,0,0,0,10\n2,1010,910,502,0,1,0,0,0,10\n")
    old = tmp_path / "old.csv"
    old.write_text(OLD_META + "1,1000,900,1,501,0,0,10\n2,1010,910,2,502,0,1,10\n")
    a, b = parse_acquisition_frame_clock_csv(new), parse_acquisition_frame_clock_csv(old)
    np.testing.assert_array_equal(a.recording_frame_id, b.recording_frame_id)
    np.testing.assert_array_equal(a.camera_timestamp_ns, b.camera_timestamp_ns)


def test_an_adopted_recorder_file_with_a_bare_frame_id_is_refused(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text(NEW_META.replace("bytes\n", "bytes,frame_id\n") + "1,1000,900,501,0,0,0,0,0,10,1\n")
    with pytest.raises(AcquisitionFrameClockError, match="without a counter suffix: frame_id"):
        parse_acquisition_frame_clock_csv(path)
