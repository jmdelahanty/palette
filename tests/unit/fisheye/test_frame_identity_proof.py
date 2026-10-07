"""Orange's frame-identity proof (v2) admits a parent only when it binds the stream exactly."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from fisheye.shared import recording_transfer_snapshot as rts
from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    SNAPSHOT_PATH,
    TransferSnapshotError,
    build_snapshot,
    canonical_bytes,
    plan_parent_recordings,
    verify_transfer_snapshot,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def resign(root: Path) -> None:
    data = canonical_bytes(build_snapshot(root, MARKER_NAME, destination=True))
    (root / SNAPSHOT_PATH).write_bytes(data)
    marker = json.loads((root / MARKER_NAME).read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    marker["snapshot_id"] = "sha256:" + digest
    marker["snapshot"].update(size_bytes=len(data), sha256=digest)
    write_json(root / MARKER_NAME, marker)


def passed_proof(frames: int) -> dict:
    """The shape Orange's recorder writes (Orange 8b359de), for ``frames`` encoded frames."""

    return {
        "assignment_event": "orange_acquisition_recording_frame_sequence",
        "canonical_field": "recording_frame_id",
        "continuity_policy": "encoded_subset",
        "legacy_aliases": {"frame_id": "recording_frame_id"},
        "recording_frame_id_gaps_allowed": True,
        "row_granularity": "one_encoded_video_frame",
        "schema_id": "orange.external_recorder.frame_identity_proof",
        "schema_version": 2,
        "scope": "recording_session_and_camera_stream",
        "source_frames_dropped": 0,
        "source_frames_skipped_by_policy": 0,
        "status": "passed",
        "video_binding": {
            "encoded_video_frames": frames,
            "first_packet_write_error_code": None,
            "identity_mismatches": 0,
            "metadata_rows": frames,
            "metadata_write_event": "completed_gop_after_returned_identity_match",
            "method": "nvenc_input_timestamp_to_output_timestamp_registry",
            "outstanding_submitted_identities": 0,
            "packet_submissions_accepted": frames,
            "packet_submissions_rejected": 0,
            "packet_write_attempts": frames,
            "packet_write_failures": 0,
            "packets_written": frames,
            "returned_identity_matches": frames,
            "submitted_frame_identities": frames,
            "verification_rule_id": "orange.external_recorder.frame_identity.v2",
            "verified": True,
        },
    }


def single(tmp_path: Path, proof: dict | None) -> Path:
    """The single-video fixture with its summary's proof replaced (or removed)."""

    root = Path(shutil.copytree(FIXTURES / "failed_optional_proof", tmp_path / "single"))
    write_json(root / "Cam02010093_full.summary.json", {} if proof is None else {"frame_identity_proof": proof})
    resign(root)
    return root


def single_frames(tmp_path: Path) -> int:
    (parent,) = plan_parent_recordings(verify_transfer_snapshot(single(tmp_path / "count", None)))
    return parent.total_frames


def plan(root: Path):
    return plan_parent_recordings(verify_transfer_snapshot(root))


def test_passed_proof_binding_the_stream_admits(tmp_path: Path) -> None:
    frames = single_frames(tmp_path)
    (parent,) = plan(single(tmp_path, passed_proof(frames)))
    assert parent.total_frames == frames


def mutate(proof: dict, path: str, value) -> dict:
    proof = copy.deepcopy(proof)
    *parents, leaf = path.split(".")
    node = proof
    for key in parents:
        node = node[key]
    if value is KeyError:
        del node[leaf]
    else:
        node[leaf] = value
    return proof


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", 1, "needs a supported semantic proof validator"),
        ("schema_id", "orange.other_proof", "needs a supported semantic proof validator"),
        ("extra_field", True, "grammar"),
        ("video_binding.verification_rule_id", KeyError, "grammar"),
        ("video_binding.identity_mismatches", 1, "grammar"),
        ("video_binding.verified", False, "grammar"),
        ("video_binding.metadata_rows", -1, "grammar"),
        ("source_frames_dropped", 1, "dropped source frames"),
        ("source_frames_skipped_by_policy", 1, "skipped source frames by policy"),
    ],
)
def test_proof_outside_the_contract_refuses(tmp_path: Path, path: str, value, message: str) -> None:
    proof = mutate(passed_proof(single_frames(tmp_path)), path, value)
    with pytest.raises(TransferSnapshotError, match=message):
        plan(single(tmp_path, proof))


@pytest.mark.parametrize("counter", rts._PROOF_EQUAL_COUNTERS)
@pytest.mark.parametrize("delta", [-1, 1])
def test_every_counter_must_equal_the_streams_frames(tmp_path: Path, counter: str, delta: int) -> None:
    frames = single_frames(tmp_path)
    proof = mutate(passed_proof(frames), f"video_binding.{counter}", frames + delta)
    with pytest.raises(TransferSnapshotError, match="does not bind the stream"):
        plan(single(tmp_path, proof))


def test_proof_off_by_one_from_palettes_frame_count_refuses(tmp_path: Path) -> None:
    frames = single_frames(tmp_path)
    with pytest.raises(TransferSnapshotError, match="does not bind the stream's"):
        plan(single(tmp_path, passed_proof(frames + 1)))


def test_failed_proof_still_reports_failed(tmp_path: Path) -> None:
    proof = mutate(passed_proof(single_frames(tmp_path)), "status", "failed")
    with pytest.raises(TransferSnapshotError, match="frame_identity_proof is failed"):
        plan(single(tmp_path, proof))


def rolling(tmp_path: Path, proofs: dict[str, dict], *, shared_kind_path: str | None = None) -> Path:
    """The rolling fixture with one session-level summary per camera stream, shared by its clips.

    ``proofs`` maps "<camera>/<kind>" to the proof for that stream.
    """

    root = Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "rolling"))
    session = json.loads((root / "recording_session.json").read_bytes())
    for clip in session["clips"]:
        for camera, outputs in clip["recording_outputs"].items():
            for kind, output in outputs.items():
                key = f"{camera}/{kind}"
                if key in proofs:
                    output["summary"] = shared_kind_path or f"Cam{camera}_{kind}.summary.json"
    write_json(root / "recording_session.json", session)
    for key, proof in proofs.items():
        camera, kind = key.split("/")
        write_json(root / (shared_kind_path or f"Cam{camera}_{kind}.summary.json"), {"frame_identity_proof": proof})
    resign(root)
    return root


def test_rolling_proof_covers_all_clips_of_its_stream(tmp_path: Path) -> None:
    # Each rolling parent has 3 frames over 2 clips; the session summary binds all 3.
    root = rolling(
        tmp_path,
        {f"{camera}/{kind}": passed_proof(3) for camera in ("02010093", "02010094") for kind in ("full", "crop")},
    )
    parents = plan(root)
    assert [parent.total_frames for parent in parents] == [3, 3]


def test_rolling_proof_counting_one_clip_refuses(tmp_path: Path) -> None:
    root = rolling(tmp_path, {"02010093/full": passed_proof(2)})
    with pytest.raises(TransferSnapshotError, match="does not bind the stream's 3 frames"):
        plan(root)


def test_crop_stream_may_skip_by_policy_but_never_drop(tmp_path: Path) -> None:
    skipped = mutate(passed_proof(3), "source_frames_skipped_by_policy", 5)
    assert len(plan(rolling(tmp_path / "a", {"02010093/crop": skipped}))) == 2
    dropped = mutate(passed_proof(3), "source_frames_dropped", 1)
    with pytest.raises(TransferSnapshotError, match="dropped source frames"):
        plan(rolling(tmp_path / "b", {"02010093/crop": dropped}))


def test_one_proof_shared_by_full_and_crop_refuses(tmp_path: Path) -> None:
    root = rolling(
        tmp_path,
        {"02010093/full": passed_proof(3), "02010093/crop": passed_proof(3)},
        shared_kind_path="Cam02010093.summary.json",
    )
    with pytest.raises(TransferSnapshotError, match="shared by different output kinds"):
        plan(root)


def test_packaged_schema_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    rts._frame_identity_proof_validator.cache_clear()
    monkeypatch.setattr(rts, "FRAME_IDENTITY_PROOF_SCHEMA_SHA256", "0" * 64)
    try:
        with pytest.raises(TransferSnapshotError, match="packaged_contract_drift"):
            rts._frame_identity_proof_validator()
    finally:
        rts._frame_identity_proof_validator.cache_clear()
