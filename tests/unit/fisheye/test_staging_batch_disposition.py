from __future__ import annotations

import json
from pathlib import Path

from fisheye.shared.staging_batch_disposition import (
    DISPOSABLE_DIAGNOSTIC,
    MOVED,
    RETAINED_AUTHORITY,
    UNKNOWN,
    VERIFIED_FANOUT_COPY,
    build_staging_batch_disposition,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _by_relative(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    return {str(row["relative_path"]): row for row in artifacts}


def test_disposition_classifies_moves_verified_copies_authority_and_unknown(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "staging" / "batch"
    batch.mkdir(parents=True)
    shared = batch / "recording_session.json"
    shared.write_bytes(b"shared-session")
    geometry = batch / "recording_geometry_contract.json"
    geometry.write_bytes(b"geometry")
    unknown = batch / "surprise.tmp"
    unknown.write_bytes(b"unknown")

    recording_a = tmp_path / "recordings" / "a"
    recording_b = tmp_path / "recordings" / "b"
    copied_a = recording_a / "raw" / "recording_session.json"
    copied_b = recording_b / "raw" / "recording_session.json"
    copied_a.parent.mkdir(parents=True)
    copied_b.parent.mkdir(parents=True)
    copied_a.write_bytes(shared.read_bytes())
    copied_b.write_bytes(shared.read_bytes())
    moved_source = batch / "citrus" / "a.h5"
    moved_dest = recording_a / "raw" / "a.h5"
    moved_dest.write_bytes(b"h5")
    zarr_a = recording_a / "zarr" / "a_analysis.zarr"
    zarr_b = recording_b / "zarr" / "b_analysis.zarr"
    zarr_a.mkdir(parents=True)
    zarr_b.mkdir(parents=True)
    log = tmp_path / "organize.jsonl"
    _write_jsonl(
        log,
        [
            {"event": "file_moved", "source": str(moved_source), "dest": str(moved_dest)},
            {"event": "file_copied", "source": str(shared), "dest": str(copied_a)},
            {"event": "file_copied", "source": str(shared), "dest": str(copied_b)},
        ],
    )

    payload = build_staging_batch_disposition(
        batch,
        organize_log=log,
        workflow_status="ok",
        apply=True,
        organized_recording_dirs=[recording_a, recording_b],
        zarr_paths=[zarr_a, zarr_b],
    )

    rows = _by_relative(payload)
    assert rows["citrus/a.h5"]["disposition"] == MOVED
    assert rows["recording_session.json"]["disposition"] == VERIFIED_FANOUT_COPY
    assert rows["recording_geometry_contract.json"]["disposition"] == RETAINED_AUTHORITY
    assert rows["surprise.tmp"]["disposition"] == UNKNOWN
    assert payload["cleanup_assessment"]["safe_to_delete_batch"] is False


def test_disposition_does_not_verify_mismatched_copy(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    source = batch / "recording_session.json"
    source.write_bytes(b"source")
    destination = tmp_path / "recording" / "raw" / "recording_session.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"different")
    zarr = tmp_path / "recording" / "zarr" / "recording_analysis.zarr"
    zarr.mkdir(parents=True)
    log = tmp_path / "organize.jsonl"
    _write_jsonl(
        log,
        [{"event": "file_copied", "source": str(source), "dest": str(destination)}],
    )

    payload = build_staging_batch_disposition(
        batch,
        organize_log=log,
        workflow_status="ok",
        apply=True,
        organized_recording_dirs=[tmp_path / "recording"],
        zarr_paths=[zarr],
    )

    row = _by_relative(payload)["recording_session.json"]
    assert row["disposition"] == UNKNOWN
    assert row["reason"] == "copy_destination_missing_or_content_mismatch"


def test_verified_geometry_fanout_remains_cleanup_blocked_until_publication(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    assets = batch / "recording_geometry_assets"
    assets.mkdir(parents=True)
    contract = batch / "recording_geometry_contract.json"
    observation = assets / "observation.json"
    contract.write_bytes(b"contract")
    observation.write_bytes(b"observation")
    recording = tmp_path / "recording"
    copied_contract = recording / "raw/recording_geometry_bundle/recording_geometry_contract.json"
    copied_observation = (
        recording / "raw/recording_geometry_bundle/recording_geometry_assets/observation.json"
    )
    copied_contract.parent.mkdir(parents=True)
    copied_observation.parent.mkdir(parents=True)
    copied_contract.write_bytes(contract.read_bytes())
    copied_observation.write_bytes(observation.read_bytes())
    zarr = recording / "zarr/recording_analysis.zarr"
    zarr.mkdir(parents=True)
    log = tmp_path / "organize.jsonl"
    _write_jsonl(
        log,
        [
            {"event": "file_copied", "source": str(contract), "dest": str(copied_contract)},
            {
                "event": "file_copied",
                "source": str(observation),
                "dest": str(copied_observation),
            },
        ],
    )

    payload = build_staging_batch_disposition(
        batch,
        organize_log=log,
        workflow_status="ok",
        apply=True,
        organized_recording_dirs=[recording],
        zarr_paths=[zarr],
    )

    rows = _by_relative(payload)
    assert rows["recording_geometry_contract.json"]["disposition"] == VERIFIED_FANOUT_COPY
    assert rows["recording_geometry_assets/observation.json"]["disposition"] == VERIFIED_FANOUT_COPY
    assert payload["cleanup_assessment"]["safe_to_delete_batch"] is False
    assert (
        "recording_geometry_candidate_publication_not_implemented"
        in payload["cleanup_assessment"]["blockers"]
    )


def test_disposition_marks_shard_logs_disposable_only_after_merged_validation(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    recorder = batch / "external_recorder"
    recorder.mkdir(parents=True)
    shard_csv = recorder / "Cam2010093_external_encode_shard0_gpu3.csv"
    shard_keyframes = recorder / "Cam2010093_external_keyframes_shard0_gpu3.json"
    shard_csv.write_text("frame,timing\n", encoding="utf-8")
    shard_keyframes.write_text("{}", encoding="utf-8")

    recording = tmp_path / "recording"
    merged_video = recording / "cams" / "Cam2010093_recording.mp4"
    merged_keyframes = recording / "cams" / "Cam2010093_recording_keyframe.json"
    summary_dest = recording / "cams" / "Cam2010093_recording_summary.json"
    merged_video.parent.mkdir(parents=True)
    merged_video.write_bytes(b"video")
    merged_keyframes.write_text("{}", encoding="utf-8")
    summary_dest.write_text(
        json.dumps(
            {
                "worker_failed": False,
                "frames_encoded": 10,
                "frames_received": 10,
                "encode_dropped": 0,
                "external_encode_shards": [
                    {
                        "worker_failed": False,
                        "encode_csv": f"/producer/{shard_csv.name}",
                        "mp4_keyframe": f"/producer/{shard_keyframes.name}",
                        "mp4_retention": {
                            "removed_after_merge": True,
                            "retained": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    zarr = recording / "zarr" / "recording_analysis.zarr"
    zarr.mkdir(parents=True)
    log = tmp_path / "organize.jsonl"
    _write_jsonl(
        log,
        [
            {
                "event": "file_moved",
                "source": str(recorder / "Cam2010093_external.mp4"),
                "dest": str(merged_video),
            },
            {
                "event": "file_moved",
                "source": str(recorder / "Cam2010093_external_keyframes.json"),
                "dest": str(merged_keyframes),
            },
            {
                "event": "file_moved",
                "source": str(recorder / "Cam2010093_external_summary.json"),
                "dest": str(summary_dest),
            },
        ],
    )

    payload = build_staging_batch_disposition(
        batch,
        organize_log=log,
        workflow_status="ok",
        apply=True,
        organized_recording_dirs=[recording],
        zarr_paths=[zarr],
    )

    rows = _by_relative(payload)
    assert rows[shard_csv.relative_to(batch).as_posix()]["disposition"] == DISPOSABLE_DIAGNOSTIC
    assert (
        rows[shard_keyframes.relative_to(batch).as_posix()]["disposition"]
        == DISPOSABLE_DIAGNOSTIC
    )


def test_disposition_retains_shard_logs_when_workflow_is_incomplete(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    recorder = batch / "external_recorder"
    recorder.mkdir(parents=True)
    shard_csv = recorder / "Cam2010093_external_encode_shard0_gpu3.csv"
    shard_csv.write_text("frame,timing\n", encoding="utf-8")

    payload = build_staging_batch_disposition(
        batch,
        organize_log=None,
        workflow_status="failed",
        apply=True,
        organized_recording_dirs=[],
        zarr_paths=[],
    )

    row = _by_relative(payload)[shard_csv.relative_to(batch).as_posix()]
    assert row["disposition"] == RETAINED_AUTHORITY
    assert payload["cleanup_assessment"]["safe_to_delete_batch"] is False
