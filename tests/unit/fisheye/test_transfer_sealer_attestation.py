"""Verify-once: large video/H5 files reuse the sealer's destination digests.

Citrus's sealer (marker v3) hashes every destination file and writes the marker
only when they match the snapshot. Palette then checks those large files by
type, size and modification time instead of hashing them again. Every other
file, and every file of a v2 delivery, is still hashed by Palette.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from fisheye.shared import recording_transfer_snapshot as snapshot_module
from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    TransferSnapshotError,
    canonical_bytes,
    verify_transfer_snapshot,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "recording_transfer_v2" / "rolling"
SEALER = {"package": "citrus-recording-transfer", "version": "2.0.1"}
VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".h265", ".hevc")


def _bundle(tmp_path: Path, *, v3: bool = True, sealer: dict = SEALER) -> Path:
    root = Path(shutil.copytree(FIXTURE, tmp_path / "staging" / "session"))
    marker = json.loads((root / MARKER_NAME).read_bytes())
    if v3:
        marker.update(
            schema_id="citrus.transfer_completion_marker.v3",
            schema_version=3,
            sealer=sealer,
        )
    (root / MARKER_NAME).write_bytes(canonical_bytes(marker))
    # The marker is written after every payload file, as the sealer does.
    sealed_at = (root / MARKER_NAME).stat().st_mtime_ns
    for path in root.rglob("*"):
        if path.is_file() and path.name != MARKER_NAME:
            os.utime(path, ns=(sealed_at - 10**9, sealed_at - 10**9))
    return root


def _videos(root: Path) -> list[Path]:
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    assert videos, "fixture needs at least one video"
    return videos


def _rewrite_same_size(path: Path) -> None:
    """Change content but keep size and the pre-seal modification time."""

    info = path.stat()
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))


@pytest.fixture
def attest_small_files(monkeypatch):
    # Fixture videos are tiny; lower the size floor so they qualify.
    monkeypatch.setattr(snapshot_module, "SEALER_ATTESTED_MIN_BYTES", 0)


@pytest.fixture
def hashed_paths(monkeypatch):
    seen: list[str] = []
    real = snapshot_module.file_ref

    def counting(root, relative):
        seen.append(relative)
        return real(root, relative)

    monkeypatch.setattr(snapshot_module, "file_ref", counting)
    return seen


def test_v3_large_videos_are_not_rehashed(tmp_path, attest_small_files, hashed_paths):
    root = _bundle(tmp_path)
    result = verify_transfer_snapshot(root)

    assert result.content_verification == "sealer_attested_large_files"
    videos = {p.relative_to(root).as_posix() for p in _videos(root)}
    assert not videos & set(hashed_paths)
    # Everything Palette parses is still hashed.
    assert "recording_session.json" in hashed_paths


def test_v3_below_the_size_floor_everything_is_hashed(tmp_path, hashed_paths):
    root = _bundle(tmp_path)
    result = verify_transfer_snapshot(root)

    assert result.content_verification == "sealer_attested_large_files"
    videos = {p.relative_to(root).as_posix() for p in _videos(root)}
    assert videos <= set(hashed_paths)


def test_v2_marker_hashes_every_byte(tmp_path, attest_small_files, hashed_paths):
    root = _bundle(tmp_path, v3=False)
    result = verify_transfer_snapshot(root)

    assert result.content_verification == "palette_sha256_all_bytes"
    videos = {p.relative_to(root).as_posix() for p in _videos(root)}
    assert videos <= set(hashed_paths)


def test_v2_marker_still_catches_a_same_size_rewrite(tmp_path, attest_small_files):
    root = _bundle(tmp_path, v3=False)
    _rewrite_same_size(_videos(root)[0])
    with pytest.raises(TransferSnapshotError, match="mismatch"):
        verify_transfer_snapshot(root)


def test_v3_refuses_a_video_of_another_size(tmp_path, attest_small_files):
    root = _bundle(tmp_path)
    video = _videos(root)[0]
    info = video.stat()
    video.write_bytes(video.read_bytes() + b"\0")
    os.utime(video, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(TransferSnapshotError, match="size differs from the sealed snapshot"):
        verify_transfer_snapshot(root)


def test_v3_refuses_a_video_modified_after_sealing(tmp_path, attest_small_files):
    root = _bundle(tmp_path)
    video = _videos(root)[0]
    sealed_at = (root / MARKER_NAME).stat().st_mtime_ns
    os.utime(video, ns=(sealed_at + 10**9, sealed_at + 10**9))
    with pytest.raises(TransferSnapshotError, match="modified after the delivery was sealed"):
        verify_transfer_snapshot(root)


def test_v3_refuses_a_missing_video(tmp_path, attest_small_files):
    root = _bundle(tmp_path)
    _videos(root)[0].unlink()
    with pytest.raises(TransferSnapshotError):
        verify_transfer_snapshot(root)


def test_v3_still_hashes_parsed_documents(tmp_path, attest_small_files):
    root = _bundle(tmp_path)
    _rewrite_same_size(root / "recording_session.json")
    with pytest.raises(TransferSnapshotError):
        verify_transfer_snapshot(root)


def test_v3_trusts_the_sealer_for_a_same_size_rewrite_with_an_old_mtime(
    tmp_path, attest_small_files
):
    # The accepted limit: a large video rewritten in place with its size and
    # pre-seal mtime restored is not detected; the delivery must stay immutable.
    root = _bundle(tmp_path)
    _rewrite_same_size(_videos(root)[0])
    assert verify_transfer_snapshot(root).content_verification == "sealer_attested_large_files"


def test_sealer_2_0_0_gets_no_shortcut(tmp_path, attest_small_files, hashed_paths):
    # 2.0.0 hashed the destination back through the copying host's page cache.
    root = _bundle(tmp_path, sealer={"package": "citrus-recording-transfer", "version": "2.0.0"})
    result = verify_transfer_snapshot(root)

    assert result.content_verification == "palette_sha256_all_bytes"
    videos = {p.relative_to(root).as_posix() for p in _videos(root)}
    assert videos <= set(hashed_paths)


@pytest.mark.parametrize(
    ("sealer", "reads_storage"),
    [
        ({"package": "citrus-recording-transfer", "version": "2.0.1"}, True),
        ({"package": "citrus-recording-transfer", "version": "2.1.0"}, True),
        ({"package": "citrus-recording-transfer", "version": "10.0.0"}, True),
        ({"package": "citrus-recording-transfer", "version": "2.0.0"}, False),
        ({"package": "citrus-recording-transfer", "version": "1.9.9"}, False),
        ({"package": "citrus-recording-transfer", "version": "2.0"}, False),
        ({"package": "someone-else", "version": "3.0.0"}, False),
        (None, False),
    ],
)
def test_only_sealers_from_2_0_1_read_storage(sealer, reads_storage):
    assert snapshot_module.sealer_reads_storage(sealer) is reads_storage
