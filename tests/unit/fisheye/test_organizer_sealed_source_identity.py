"""Verify-once in the organizer: hard-linked copies of sealed media are not re-hashed.

For a marker v3 delivery, a large video/H5 copy that is the staged file itself
(same device, inode, size and mtime; staged file sealed no later than the
marker) or, after retirement, the inode retirement recorded, is proven without
reading it. Cross-filesystem copies, v2 deliveries, small files and anything
that differs are hashed in full.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import shutil

import pytest

from fisheye.shared import recording_transfer_snapshot as transfer
from fisheye.utils import organize_transfer_recordings as organizer

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"
SEALER = {"package": "citrus-recording-transfer", "version": "2.0.0"}
VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".h265", ".hevc")


@pytest.fixture(autouse=True)
def _placeholder_media_sync_assessment(monkeypatch):
    from fisheye.diagnostics.video import container

    monkeypatch.setattr(
        container,
        "check_hevc_keyframe_flags",
        lambda path, **_: {
            "schema_id": "palette.video.sync_sample_assessment.v1",
            "codec": "h264",
            "container_inspection_status": "ok",
            "sync_sample_proof": "container_declared",
            "message": "placeholder media",
        },
    )


@pytest.fixture(autouse=True)
def _fixture_videos_count_as_large(monkeypatch):
    monkeypatch.setattr(transfer, "SEALER_ATTESTED_MIN_BYTES", 0)


@pytest.fixture
def organizer_hashes(monkeypatch):
    seen: list[str] = []
    real = organizer.file_ref

    def counting(root, relative):
        seen.append(Path(relative).name)
        return real(root, relative)

    monkeypatch.setattr(organizer, "file_ref", counting)
    return seen


def _staging(tmp_path: Path, *, v3: bool = True) -> Path:
    root = Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))
    marker = json.loads((root / transfer.MARKER_NAME).read_bytes())
    if v3:
        marker.update(
            schema_id="citrus.transfer_completion_marker.v3", schema_version=3, sealer=SEALER
        )
    (root / transfer.MARKER_NAME).write_bytes(transfer.canonical_bytes(marker))
    sealed_at = (root / transfer.MARKER_NAME).stat().st_mtime_ns
    for path in root.rglob("*"):
        if path.is_file() and path.name != transfer.MARKER_NAME:
            os.utime(path, ns=(sealed_at - 10**9, sealed_at - 10**9))
    return root


def _video_names(root: Path) -> set[str]:
    names = {p.name for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES}
    assert names
    return names


def _prepared(tmp_path, monkeypatch, *, v3: bool = True):
    source = _staging(tmp_path, v3=v3)
    plan = organizer.build_transfer_organization_plan(
        source, destination_root=tmp_path / "recordings"
    )
    monkeypatch.setattr(
        organizer,
        "_verify_parent_imports",
        lambda *args, **kwargs: {"fixture": "admission_stub_not_authority_evidence"},
    )
    return source, plan


def _video_destinations(plan: dict) -> list[Path]:
    paths = []
    for item in plan["files"]:
        if Path(item["source"]["path"]).suffix.lower() in VIDEO_SUFFIXES:
            for target in item["destinations"]:
                paths.append(
                    organizer._parent_directory(plan, target["recording_id"])
                    / target["relative_path"]
                )
    assert paths
    return paths


def test_v3_hard_linked_videos_are_never_rehashed(tmp_path, monkeypatch, organizer_hashes):
    source, plan = _prepared(tmp_path, monkeypatch)
    videos = _video_names(source)
    before = {
        item["source"]["path"]: (source / item["source"]["path"]).read_bytes()
        for item in plan["files"]
    }
    organizer.prepare_transfer_parent_recordings(plan)
    result = organizer.finalize_transfer_staging(plan)

    assert result["status"] == "complete"
    assert not videos & set(organizer_hashes)
    assert "recording_session.json" in organizer_hashes  # small files still hashed
    for item in plan["files"]:
        for target in item["destinations"]:
            path = (
                organizer._parent_directory(plan, target["recording_id"])
                / target["relative_path"]
            )
            assert path.read_bytes() == before[item["source"]["path"]]


def test_v3_completed_replay_proves_videos_by_retired_inode(
    tmp_path, monkeypatch, organizer_hashes
):
    source, plan = _prepared(tmp_path, monkeypatch)
    organizer.prepare_transfer_parent_recordings(plan)
    result = organizer.finalize_transfer_staging(plan)
    organizer_hashes.clear()

    assert organizer.finalize_transfer_staging(plan) == result
    assert not _video_names(tmp_path / "recordings") & set(organizer_hashes)


def test_v3_replay_hashes_a_rewritten_copy_and_refuses_changed_bytes(
    tmp_path, monkeypatch, organizer_hashes
):
    source, plan = _prepared(tmp_path, monkeypatch)
    organizer.prepare_transfer_parent_recordings(plan)
    organizer.finalize_transfer_staging(plan)
    target = _video_destinations(plan)[0]
    data = target.read_bytes()

    # Rewritten with the same bytes: its mtime moved, so it is hashed, then accepted.
    target.unlink()
    target.write_bytes(data)
    organizer_hashes.clear()
    organizer.finalize_transfer_staging(plan)
    assert target.name in organizer_hashes

    # Rewritten with different bytes of the same size: hashed and refused.
    changed = bytearray(data)
    changed[0] ^= 0xFF
    target.unlink()
    target.write_bytes(bytes(changed))
    with pytest.raises(ValueError, match="differs"):
        organizer.finalize_transfer_staging(plan)


def test_v3_accepted_limit_an_inode_reused_with_restored_size_and_mtime(
    tmp_path, monkeypatch
):
    # The accepted limit, as in the snapshot check: a copy deliberately
    # replaced so that it reuses the recorded inode with the same size and a
    # restored mtime is not detected. Deliveries and recordings are immutable.
    source, plan = _prepared(tmp_path, monkeypatch)
    organizer.prepare_transfer_parent_recordings(plan)
    state = organizer.finalize_transfer_staging(plan)
    target = _video_destinations(plan)[0]
    item = next(
        i for i in plan["files"]
        if any(
            organizer._parent_directory(plan, t["recording_id"]) / t["relative_path"] == target
            for t in i["destinations"]
        )
    )
    identity = organizer._SealedSourceIdentity(plan, state)
    assert identity.proves(target, item["source"])
    recorded = state["retirement_files"][item["source"]["path"]]
    info = target.stat()
    assert [info.st_ino, info.st_size, info.st_mtime_ns] == recorded[1:]


def test_v2_delivery_hashes_every_video_copy(tmp_path, monkeypatch, organizer_hashes):
    source, plan = _prepared(tmp_path, monkeypatch, v3=False)
    videos = _video_names(source)
    organizer.prepare_transfer_parent_recordings(plan)
    assert videos <= set(organizer_hashes)


def test_cross_filesystem_copies_are_hashed(tmp_path, monkeypatch, organizer_hashes):
    source, plan = _prepared(tmp_path, monkeypatch)
    videos = _video_names(source)
    real_link = os.link

    def link(src, dst, **kwargs):
        if Path(src).suffix.lower() in VIDEO_SUFFIXES:
            raise OSError(errno.EXDEV, "cross-device link")
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(organizer.os, "link", link)
    organizer.prepare_transfer_parent_recordings(plan)
    # Each copied temporary is hashed before it is linked into place, and the
    # final copy (a new inode) is hashed again; none is proven by identity.
    assert sum(name.endswith(".intake-partial") for name in organizer_hashes) >= len(
        _video_destinations(plan)
    )
    assert videos & set(organizer_hashes)


def test_a_rewritten_staged_marker_disables_the_shortcut(
    tmp_path, monkeypatch, organizer_hashes
):
    source, plan = _prepared(tmp_path, monkeypatch)
    organizer.prepare_transfer_parent_recordings(plan)
    state = json.loads(
        (organizer._state_directory(plan) / "organization_state.json").read_text()
    )
    marker = source / transfer.MARKER_NAME
    marker.write_bytes(marker.read_bytes() + b" ")
    identity = organizer._SealedSourceIdentity(plan, state)
    assert identity.marker_mtime_ns is None
    assert not identity.proves(_video_destinations(plan)[0], plan["files"][0]["source"])
