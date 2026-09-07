"""Clipped receipts use the real metadata, authority, clock and registry owners."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.registry.db import Registry
from fisheye.registry.recording_identity_authority import (
    RecordingIdentityAuthorityError,
    collect_regular_source_recording_identity,
    load_verified_recording_import_receipt,
)
from fisheye.shared import acquisition_frame_clock as clock_contract
from fisheye.shared import run_provenance
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_PUBLISHED_REASON,
    stamp_acquisition_authority_publication_status,
)
from fisheye.shared.clipped_video_collection import (
    build_clipped_video_collection_metadata,
)
from fisheye.shared.import_video_metadata import (
    publish_clipped_video_collection_acquisition_authority,
)
from fisheye.shared.recording_import_receipt import (
    CURRENT_RECORDING_IMPORT_PRODUCER_ID,
    RecordingImportReceipt,
    RecordingImportReceiptError,
    publish_recording_import_receipt,
)
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SourceRecordingIdentity,
    SourceRecordingIdentityClaim,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.utils.build_recording_frame_index import build_recording_frame_index

CAMERA = "02010093"
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _clock_index(path: Path, **changes: object) -> Path:
    columns = {
        "camera_serial": [CAMERA] * 4,
        "recording_frame_id": [1, 2, 3, 4],
        "parent_frame_index": [0, 1, 2, 3],
        "timestamp": [1000, 1100, 1200, 1300],
        "timestamp_sys": [2000, 2100, 2200, 2300],
    }
    columns.update(changes)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def test_clipped_clock_binds_explicit_index_without_csv_fallback(
    tmp_path: Path,
) -> None:
    index = _clock_index(tmp_path / "derived" / "frames.parquet")
    _clock_index(tmp_path / "recording_frame_index.parquet", timestamp=[9] * 4)
    (tmp_path / "first_meta.csv").write_text("not,a,clock\n")
    (tmp_path / "recording_manifest.json").write_text(
        json.dumps(
            {
                "video_streams": {
                    "streams": {
                        "full": {
                            "video": "first.mp4",
                            "frame_clock_metadata": "first_meta.csv",
                        }
                    }
                },
            }
        )
    )
    source = clock_contract.load_clipped_acquisition_frame_clock_source(
        tmp_path,
        camera_id=CAMERA,
        frame_index_path="derived/frames.parquet",
        expected_frame_count=4,
    )
    assert source.source_path == index
    assert source.source_locator == "derived/frames.parquet"
    assert source.source_kind == "recording_frame_index_parquet"
    assert source.camera_id == CAMERA
    assert source.clock_surfaces["camera_timestamp_ns"]["timescale"] == "unspecified"
    assert source.clock_surfaces["camera_timestamp_ns"]["origin"] == "unspecified"
    np.testing.assert_array_equal(source.camera_timestamp_ns, [1000, 1100, 1200, 1300])
    # The source, arrays and persisted digest use the existing Parquet grammar.
    existing = clock_contract._validate_source(
        clock_contract._load_parquet_source(
            index, recording_dir=tmp_path, camera_id=CAMERA
        ),
        expected_frame_count=4,
    )
    assert clock_contract.acquisition_frame_clock_source_sha256(source) == (
        clock_contract.acquisition_frame_clock_source_sha256(existing)
    )


def test_clipped_clock_preserves_partial_timestamp_validity(tmp_path: Path) -> None:
    index = _clock_index(
        tmp_path / "frames.parquet", timestamp=[1000, None, 1200, None]
    )
    source = clock_contract.load_clipped_acquisition_frame_clock_source(
        tmp_path,
        camera_id=CAMERA,
        frame_index_path=index,
        expected_frame_count=4,
    )
    np.testing.assert_array_equal(
        source.camera_timestamp_valid, [True, False, True, False]
    )
    assert source.camera_timestamp_ns[1] == clock_contract.MISSING_TIMESTAMP_SENTINEL
    assert source.system_timestamp_valid.all()


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"parent_frame_index": [0, 1, 1, 3]}, "complete ordered"),
        ({"recording_frame_id": [1, 2, 4, 5]}, "contiguous"),
        ({"timestamp": [1000, 900, 1200, 1300]}, "monotonic"),
        ({"timestamp": [None] * 4, "timestamp_sys": [None] * 4}, "no usable"),
        ({"camera_serial": ["2010093"] * 4}, "no rows for camera"),
    ],
)
def test_clipped_clock_preserves_validation(
    tmp_path: Path, changes: dict, match: str
) -> None:
    index = _clock_index(tmp_path / "frames.parquet", **changes)
    with pytest.raises(clock_contract.AcquisitionFrameClockError, match=match):
        clock_contract.load_clipped_acquisition_frame_clock_source(
            tmp_path,
            camera_id=CAMERA,
            frame_index_path=index,
            expected_frame_count=4,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "outside",
        "symlink_escape",
        "missing",
        "directory",
        "wrong_count",
        "bool_count",
        "camera_whitespace",
    ],
)
def test_clipped_clock_rejects_wrong_locator_and_domain(
    tmp_path: Path, damage: str
) -> None:
    recording = tmp_path / "recording"
    index = _clock_index(recording / "frames.parquet")
    if damage in {"outside", "symlink_escape"}:
        outside = _clock_index(tmp_path / "outside.parquet")
        index = outside
        if damage == "symlink_escape":
            index = recording / "link.parquet"
            index.symlink_to(outside)
    elif damage == "missing":
        index = recording / "missing.parquet"
    elif damage == "directory":
        index = recording
    with pytest.raises(
        (clock_contract.AcquisitionFrameClockError, FileNotFoundError, ValueError)
    ):
        clock_contract.load_clipped_acquisition_frame_clock_source(
            recording,
            camera_id=" " + CAMERA if damage == "camera_whitespace" else CAMERA,
            frame_index_path=index,
            expected_frame_count=(
                True if damage == "bool_count" else 3 if damage == "wrong_count" else 4
            ),
        )


def _encoded_clip(
    recording: Path, clip_index: int, *, width: int = 64, height: int = 48
) -> dict:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip(
            "ffmpeg and ffprobe are required for the encoded-clips integration fixture"
        )
    clip_id = f"clip_{clip_index:06d}"
    directory = recording / "clips" / clip_id
    directory.mkdir(parents=True)
    video = directory / f"Cam{CAMERA}_full.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s={width}x{height}:r=2",
            "-frames:v",
            "2",
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    ids = [1 + 2 * clip_index, 2 + 2 * clip_index]
    metadata = video.with_name(f"{video.stem}_meta.csv")
    metadata.write_text(
        "frame_id,timestamp,timestamp_sys\n"
        + "".join(
            f"{frame_id},{1000 + frame_id * 100},{2000 + frame_id * 100}\n"
            for frame_id in ids
        )
    )
    keyframe = video.with_name(f"{video.stem}_keyframe.json")
    keyframe.write_text(
        json.dumps({"total_frames": 2, "fps": 2, "keyframe_frames": [0]})
    )
    clip_manifest = directory / "clip_manifest.json"
    clip_manifest.write_text(json.dumps({"clip_id": clip_id}))
    return {
        "recording_id": "recording-clipped",
        "session_id": "session-clipped",
        "producer": "test",
        "recording_backend_mode": "materialized_stream_copy",
        "camera_serial": CAMERA,
        "clip_index": clip_index,
        "clip_id": clip_id,
        "clip_directory": directory.relative_to(recording).as_posix(),
        "video_path": video.relative_to(recording).as_posix(),
        "metadata_path": metadata.relative_to(recording).as_posix(),
        "keyframe_path": keyframe.relative_to(recording).as_posix(),
        "clip_manifest_path": clip_manifest.relative_to(recording).as_posix(),
        "frame_count": 2,
        "first_recording_frame_id": ids[0],
        "last_recording_frame_id": ids[-1],
        "first_clip_local_frame_index": 0,
        "last_clip_local_frame_index": 1,
    }


def _publish_clipped_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RecordingImportReceipt]:
    recording = tmp_path / "recordings" / "recording-clipped"
    clips = [_encoded_clip(recording, i) for i in range(2)]
    (recording / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "recording_id": "recording-clipped",
                "session_id": "session-clipped",
                "clips": clips,
                "camera_ranges": {CAMERA: {"clip_count": 2, "total_frame_count": 4}},
            }
        )
    )
    build_recording_frame_index(recording)
    identity = SourceRecordingIdentity.from_mapping(
        {
            "source_recording_identity_profile": SOURCE_RECORDING_IDENTITY_PROFILE,
            "recording_id": "recording-clipped",
            "session_uuid": "session-clipped",
            "camera_id": CAMERA,
        }
    )
    context = {
        "recording_type": "behavior",
        "recording_subtype": "free",
        "behavior_mode": "free",
    }
    (recording / "recording_manifest.json").write_text(
        json.dumps(
            {
                **identity.manifest_fields(),
                **context,
                "artifact_schema_id": "behavior_v1",
                "source_layout": "rolling_clips",
                "recording_clip_index": "recording_clip_index.json",
                "recording_frame_index": "recording_frame_index.parquet",
                "recording_frame_index_manifest": "recording_frame_index_manifest.json",
            }
        )
    )
    output = recording / "zarr" / "recording-clipped_analysis.zarr"
    root = zarr.open_group(str(output), mode="w", zarr_format=3, use_consolidated=False)
    root.attrs.update(
        {
            **identity.analysis_root_fields(),
            **context,
            "source_layout": "rolling_clips",
            "source_video_metadata": build_clipped_video_collection_metadata(recording),
        }
    )
    published = publish_clipped_video_collection_acquisition_authority(root)
    source = clock_contract.load_clipped_acquisition_frame_clock_source(
        recording,
        camera_id=CAMERA,
        frame_index_path="recording_frame_index.parquet",
        expected_frame_count=4,
    )
    clock_contract.publish_acquisition_frame_clock(root, source)
    consolidate_metadata_capture_expected_warnings(str(output))
    receipt = RecordingImportReceipt.create(
        producer_id=CURRENT_RECORDING_IMPORT_PRODUCER_ID,
        producer_git_sha=GIT_SHA,
        config_sha256="a" * 64,
        target_relative_path=output.relative_to(recording).as_posix(),
        identity_claim=collect_regular_source_recording_identity(output),
        acquisition_ownership_ref=published["ownership_record_ref"],
        acquisition_ownership_sha256=published["ownership_record_sha256"],
        acquisition_frame_ref=published["frame_record_ref"],
        acquisition_frame_sha256=published["frame_record_sha256"],
    )
    publish_recording_import_receipt(output, receipt)
    # Only executable checkout provenance is synthetic; scientific/publication validators are real.
    monkeypatch.setattr(
        run_provenance,
        "git_identity",
        lambda **_: {"git_sha": GIT_SHA, "git_dirty": False},
    )
    return output, receipt


def _counts(registry: Registry) -> dict[str, int]:
    return {
        table: registry.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "recordings",
            "datasets",
            "recording_identity_evidence",
            "recording_identity_revisions",
            "recording_identity_current",
            "dataset_recording_identity_current",
            "recording_import_receipt_bindings",
        )
    }


def test_clipped_receipt_real_producer_registry_and_unpatched_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, receipt = _publish_clipped_import(tmp_path, monkeypatch)
    assert load_verified_recording_import_receipt(output) == receipt
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        verified = registry.finalize_current_source_import(
            zarr_path=output, receipt=receipt, decided_by="test"
        )
        replay = registry.read_verified_recording_import_by_path(output)
        assert replay == verified
        assert verified.acquisition_frame.record.source_total_frames == 4
        assert verified.acquisition_frame.record.camera_id == CAMERA
        assert (
            verified.acquisition_frame.record.source_video_metadata["layout"]
            == "clipped_video_collection"
        )
        before = _counts(registry)
        assert (
            registry.finalize_current_source_import(
                zarr_path=output, receipt=receipt, decided_by="test"
            ).receipt
            == receipt
        )
        assert _counts(registry) == before
    finally:
        registry.close()


@pytest.mark.parametrize(
    "damage",
    [
        "member",
        "index",
        "missing_clock",
        "clock_payload",
        "stale_consolidated",
        "missing_consolidated",
        "missing_receipt",
        "wrong_mode",
        "wrong_locator",
        "wrong_identity",
        "wrong_manifest_index",
        "wrong_manifest_frame_index",
        "malformed_rolling",
        "null_rolling",
        "incomplete_rolling",
        "advertised_crop_without_ledger",
    ],
)
@pytest.mark.parametrize("bound", [False, True], ids=["finalization", "bound_read"])
def test_clipped_receipt_refuses_stale_evidence_without_registry_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str, bound: bool
) -> None:
    output, receipt = _publish_clipped_import(tmp_path, monkeypatch)
    recording = output.parent.parent
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        if bound:
            registry.finalize_current_source_import(
                zarr_path=output, receipt=receipt, decided_by="test"
            )
        before = _counts(registry)
        root = zarr.open_group(str(output), mode="r+", use_consolidated=False)
        if damage == "member":
            member = (
                recording
                / root.attrs["source_video_metadata"]["collection"]["members"][0][
                    "relative_path"
                ]
            )
            member.write_bytes(member.read_bytes() + b"changed")
        elif damage == "index":
            index = recording / "recording_clip_index.json"
            index.write_text(index.read_text() + " ")
        elif damage == "missing_clock":
            del root[clock_contract.ACQUISITION_FRAME_CLOCK_RUNS_PATH]
        elif damage == "clock_payload":
            root[root.attrs["acquisition_frame_clock_ref"]]["camera_timestamp_ns"][
                1
            ] = 7
        elif damage == "stale_consolidated":
            root["raw_video"].attrs["unpublished_change"] = True
        elif damage == "missing_consolidated":
            path = output / "zarr.json"
            metadata = json.loads(path.read_text())
            metadata["consolidated_metadata"] = None
            path.write_text(json.dumps(metadata))
        elif damage == "missing_receipt":
            (output / ".imports" / f"{receipt.receipt_sha256}.json").unlink()
        elif damage == "wrong_mode":
            stamp_acquisition_authority_publication_status(
                root,
                root["raw_video"],
                status=ACQUISITION_AUTHORITY_PUBLISHED,
                reason_code=EXTERNAL_ACQUISITION_PUBLISHED_REASON,
                authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
                authority_path=f"analysis/acquisition_camera_frames/{CAMERA}",
            )
        elif damage == "wrong_locator":
            metadata = dict(root.attrs["source_video_metadata"])
            metadata["locator"]["kind"] = "recording_relative"
            root.attrs["source_video_metadata"] = metadata
        elif damage == "wrong_identity":
            root.attrs["camera_id"] = CAMERA.lstrip("0")
        elif damage == "advertised_crop_without_ledger":
            root.attrs["acquisition_crop_ledger_available"] = True
            consolidate_metadata_capture_expected_warnings(str(output))
        elif damage in {
            "wrong_manifest_index",
            "wrong_manifest_frame_index",
            "malformed_rolling",
            "null_rolling",
            "incomplete_rolling",
        }:
            manifest = recording / "recording_manifest.json"
            payload = json.loads(manifest.read_text())
            if damage == "wrong_manifest_index":
                payload["recording_clip_index"] = "wrong.json"
            elif damage == "wrong_manifest_frame_index":
                payload["recording_frame_index"] = "wrong.parquet"
            elif damage == "malformed_rolling":
                payload["rolling_clip_streams"] = []
            elif damage == "null_rolling":
                payload["rolling_clip_streams"] = None
            else:
                payload["rolling_clip_streams"] = {
                    "schema_id": "palette.orange_rolling_clip_streams.v1",
                    "frame_clock": "recording_frame_id",
                    "output_kinds": ["crop", "full"],
                    **{
                        field: payload[field]
                        for field in (
                            "recording_clip_index",
                            "recording_frame_index",
                            "recording_frame_index_manifest",
                        )
                    },
                }
                if damage == "incomplete_rolling":
                    del payload["rolling_clip_streams"]["recording_frame_index"]
            manifest.write_text(json.dumps(payload))
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match=(
                "no acquisition crop stream"
                if damage == "advertised_crop_without_ledger"
                else None
            ),
        ):
            load_verified_recording_import_receipt(output)
        with pytest.raises(
            (RecordingIdentityAuthorityError, RecordingImportReceiptError)
        ):
            if bound:
                registry.read_verified_recording_import_by_path(output)
            else:
                registry.finalize_current_source_import(
                    zarr_path=output, receipt=receipt, decided_by="test"
                )
        assert _counts(registry) == before
    finally:
        registry.close()


def test_receipt_v1_golden_bytes_remain_unchanged() -> None:
    identity = SourceRecordingIdentityClaim.create(
        SourceRecordingIdentity.from_mapping(
            {
                "source_recording_identity_profile": SOURCE_RECORDING_IDENTITY_PROFILE,
                "recording_id": "recording-a",
                "session_uuid": "session-a",
                "camera_id": "2010093",
            }
        )
    )
    receipt = RecordingImportReceipt.create(
        producer_id=CURRENT_RECORDING_IMPORT_PRODUCER_ID,
        producer_git_sha=GIT_SHA,
        config_sha256="a" * 64,
        target_relative_path="recordings/recording-a.zarr",
        identity_claim=identity,
        acquisition_ownership_ref="ownership/recording-a.json",
        acquisition_ownership_sha256="a" * 64,
        acquisition_frame_ref="frames/recording-a.parquet",
        acquisition_frame_sha256="b" * 64,
    )
    assert (
        receipt.receipt_sha256
        == "b232f1ac919c4accf603403f993f0ca3c2f179b845c39acd4b9a610e0cee90cb"
    )
    assert (
        sha256(receipt.to_json_bytes()).hexdigest()
        == "be98ba4ddd2f0db2b2cc9db6ca5be754458b047fa05a752373e343acdeb132fb"
    )
