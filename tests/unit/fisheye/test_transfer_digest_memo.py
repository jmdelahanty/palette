"""Verify-once within one intake job: hash each unchanged file once, stat identity after.

Every refusal the hashing primitive made before the memo still fires with it:
a rewritten, truncated, appended, replaced, deleted or symlinked artifact is
read again (or refused) because its stat signature moved. The memo is opt-in,
bounded to its block and thread, and never persisted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import threading
import time

import pytest

from fisheye.shared import recording_transfer_snapshot as transfer
from fisheye.utils import organize_transfer_recordings as organizer

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"


class _Reads(list):
    """Names of the files whose bytes were actually read, in order."""

    def __init__(self):
        super().__init__()
        self.signatures: list[tuple] = []


@pytest.fixture
def reads(monkeypatch):
    seen = _Reads()
    real = transfer._sha256_of_regular_file

    def counting(path, relative, before):
        seen.append(Path(relative).name)
        seen.signatures.append(transfer._stat_signature(before))
        return real(path, relative, before)

    monkeypatch.setattr(transfer, "_sha256_of_regular_file", counting)
    return seen


@pytest.fixture
def parses(monkeypatch):
    """Names of the metadata files whose rows were actually parsed, in order."""

    seen: list[str] = []
    real = transfer._frame_map_rows

    def counting(path, kind, previous, crop_offset, on_row):
        seen.append(Path(path).name)
        return real(path, kind, previous, crop_offset, on_row)

    monkeypatch.setattr(transfer, "_frame_map_rows", counting)
    return seen


@pytest.fixture(autouse=True)
def _files_have_settled(monkeypatch):
    """Fixture files were written moments ago; the job's clock says they settled."""

    monkeypatch.setattr(transfer, "_now_ns", lambda: time.time_ns() + 3600 * 10**9)


def _next_timestamp_tick(path: Path) -> None:
    """Return once the filesystem clock has moved past ``path``'s ctime."""

    reference = path.lstat().st_ctime_ns
    probe = path.parent / ".tick"
    while True:
        probe.write_bytes(b"")
        if probe.lstat().st_ctime_ns > reference:
            probe.unlink()
            return
        time.sleep(0.002)


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


def _artifact(tmp_path: Path, data: bytes = b"exact bytes\n") -> tuple[Path, str]:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.json").write_bytes(data)
    return root, "a.json"


def _bump_ctime_only(path: Path) -> None:
    mode = path.stat().st_mode
    os.chmod(path, mode ^ 0o001)
    os.chmod(path, mode)


# ------------------------------------------------------------- the primitive


def test_outside_a_memo_every_call_reads_the_bytes(tmp_path, reads):
    root, rel = _artifact(tmp_path)
    first = transfer.file_ref(root, rel)
    assert transfer.file_ref(root, rel) == transfer.file_ref(root, rel) == first
    assert reads == ["a.json"] * 3
    assert transfer.active_digest_memo() is None


def test_inside_a_memo_an_unchanged_file_is_read_once(tmp_path, reads):
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo() as memo:
        first = transfer.file_ref(root, rel)
        assert transfer.file_ref(root, rel) == transfer.file_ref(root, rel) == first
        assert (memo.hits, memo.misses) == (2, 1)
    assert reads == ["a.json"]
    # Bounded to the block: afterwards the bytes are read again.
    assert transfer.file_ref(root, rel) == first
    assert reads == ["a.json"] * 2


def test_a_hard_linked_copy_shares_one_digest(tmp_path, reads):
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo():
        first = transfer.file_ref(root, rel)
        _next_timestamp_tick(root / rel)
        os.link(root / rel, root / "copy.json")  # link count moved: ctime moved
        copy = transfer.file_ref(root, "copy.json")
        again = transfer.file_ref(root, rel)
        assert copy["sha256"] == again["sha256"] == first["sha256"]
        # One more read for the new ctime, shared by both names.
        assert reads == ["a.json", "copy.json"]


@pytest.mark.parametrize(
    "tamper",
    ["rewrite_same_size", "truncate", "append", "replace_inode", "rewrite_restored_mtime"],
)
def test_a_changed_file_is_read_again_and_its_new_digest_reported(tmp_path, reads, tamper):
    root, rel = _artifact(tmp_path)
    path = root / rel
    before = path.stat()
    with transfer.digest_memo():
        original = transfer.file_ref(root, rel)
        _next_timestamp_tick(path)
        data = path.read_bytes()
        if tamper == "rewrite_same_size":
            changed = bytearray(data)
            changed[0] ^= 0xFF
            path.write_bytes(bytes(changed))
        elif tamper == "truncate":
            path.write_bytes(data[:-1])
        elif tamper == "append":
            path.write_bytes(data + b"!")
        elif tamper == "replace_inode":
            path.unlink()
            path.write_bytes(data[::-1])
        else:
            changed = bytearray(data)
            changed[-2] ^= 0x01
            path.write_bytes(bytes(changed))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert path.stat().st_mtime_ns == before.st_mtime_ns  # mtime forged
        current = transfer.file_ref(root, rel)
    assert current["sha256"] != original["sha256"]
    assert reads == ["a.json", "a.json"]


def test_a_permission_change_rereads_and_agrees(tmp_path, reads):
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo():
        original = transfer.file_ref(root, rel)
        _next_timestamp_tick(root / rel)
        _bump_ctime_only(root / rel)
        assert transfer.file_ref(root, rel) == original
    assert reads == ["a.json", "a.json"]


def test_a_symlink_or_missing_artifact_is_refused_with_a_warm_memo(tmp_path, reads):
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo():
        transfer.file_ref(root, rel)
        (root / rel).unlink()
        with pytest.raises(FileNotFoundError):
            transfer.file_ref(root, rel)
        (root / "target.json").write_bytes(b"other\n")
        os.symlink("target.json", root / rel)
        with pytest.raises(ValueError, match="not a regular artifact"):
            transfer.file_ref(root, rel)
        (root / rel).unlink()
        (root / rel).mkdir()
        with pytest.raises(ValueError, match="not a regular artifact"):
            transfer.file_ref(root, rel)
    assert reads == ["a.json"]


def test_replaced_during_open_is_still_refused(tmp_path, reads, monkeypatch):
    root, rel = _artifact(tmp_path)
    path = root / rel
    real_open = Path.open

    def swap_then_open(self, *args, **kwargs):
        if self == path:
            monkeypatch.setattr(Path, "open", real_open)
            data = self.read_bytes()
            self.unlink()
            (self.parent / "fresh.json").write_bytes(data + b"\n")
            os.rename(self.parent / "fresh.json", self)
        return real_open(self, *args, **kwargs)

    with transfer.digest_memo():
        monkeypatch.setattr(Path, "open", swap_then_open)
        with pytest.raises(ValueError, match="replaced during open|mutated while hashing"):
            transfer.file_ref(root, rel)


def test_the_memo_is_per_thread_unless_shared(tmp_path, reads):
    root, rel = _artifact(tmp_path)
    results: dict[str, object] = {}

    def unshared():
        results["unshared_memo"] = transfer.active_digest_memo()
        results["unshared"] = transfer.file_ref(root, rel)

    def shared(memo):
        with transfer.digest_memo_shared(memo):
            results["shared_memo"] = transfer.active_digest_memo()
            results["shared"] = transfer.file_ref(root, rel)
        results["after_shared"] = transfer.active_digest_memo()

    with transfer.digest_memo() as memo:
        first = transfer.file_ref(root, rel)
        for worker in (threading.Thread(target=unshared), threading.Thread(target=shared, args=(memo,))):
            worker.start()
            worker.join()
    assert results["unshared_memo"] is None and results["unshared"] == first
    assert results["shared_memo"] is memo and results["shared"] == first
    assert results["after_shared"] is None
    assert reads == ["a.json", "a.json"]  # the unshared thread read; the shared one did not


def test_nested_memos_are_independent_and_restore_the_outer_one(tmp_path, reads):
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo() as outer:
        transfer.file_ref(root, rel)
        with transfer.digest_memo() as inner:
            assert inner is not outer and transfer.active_digest_memo() is inner
            transfer.file_ref(root, rel)
        assert transfer.active_digest_memo() is outer
        transfer.file_ref(root, rel)
    assert reads == ["a.json", "a.json"]


# ------------------------------------------------------- the whole delivery


def _staging(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))


def test_snapshot_verification_under_a_memo_is_the_fresh_verification(tmp_path, reads):
    root = _staging(tmp_path)
    fresh = transfer.verify_transfer_snapshot(root)
    fresh_reads = len(reads)
    reads.clear()
    with transfer.digest_memo():
        first = transfer.verify_transfer_snapshot(root)
        first_reads = len(reads)
        second = transfer.verify_transfer_snapshot(root)
    assert (first.snapshot_id, first.snapshot) == (second.snapshot_id, second.snapshot)
    assert (first.snapshot_id, first.snapshot) == (fresh.snapshot_id, fresh.snapshot)
    assert first.content_verification == fresh.content_verification
    # Verification reads every inventory file twice (build, then close the
    # observation window); the memo reads each file once per verification
    # block and the second verification reads nothing.
    assert len(reads) == first_reads == fresh_reads // 2


def test_a_source_change_between_verifications_is_still_refused(tmp_path, reads):
    root = _staging(tmp_path)
    with transfer.digest_memo():
        transfer.verify_transfer_snapshot(root)
        session = root / "recording_session.json"
        data = bytearray(session.read_bytes())
        data[data.index(b"{") + 1 : data.index(b"{") + 1] = b" "
        session.write_bytes(bytes(data))  # same JSON, other bytes
        with pytest.raises(transfer.TransferSnapshotError):
            transfer.verify_transfer_snapshot(root)


def test_organizing_under_a_memo_reports_the_fresh_digests_and_reads_each_file_at_most_twice(
    tmp_path, reads, monkeypatch
):
    source = _staging(tmp_path)
    monkeypatch.setattr(
        organizer,
        "_verify_parent_imports",
        lambda *args, **kwargs: {"fixture": "admission_stub_not_authority_evidence"},
    )
    reported: list[tuple[Path, str, dict]] = []
    real_file_ref = transfer.file_ref

    def recording(root, relative):
        result = real_file_ref(root, relative)
        reported.append((Path(root), relative, result))
        return result

    monkeypatch.setattr(organizer, "file_ref", recording)
    monkeypatch.setattr(transfer, "file_ref", recording)

    with transfer.digest_memo() as memo:
        plan = organizer.build_transfer_organization_plan(
            source, destination_root=tmp_path / "recordings"
        )
        organizer.prepare_transfer_parent_recordings(plan)
        result = organizer.finalize_transfer_staging(plan)
    assert result["status"] == "complete"
    assert memo.hits > memo.misses > 0
    signatures = list(reads.signatures)
    names = list(reads)

    # Every digest the organizer saw is the digest of the bytes on disk now
    # (or, for retired staging names, of the organized copy that outlived them).
    monkeypatch.setattr(transfer, "file_ref", real_file_ref)
    for root, relative, seen in reported:
        if not (root / relative).exists():
            continue
        assert real_file_ref(root, relative) == seen, relative

    # One read per (inode, size, mtime, ctime): a file is read when planned,
    # once more after each hard link moved its ctime, and once more after
    # retirement unlinked the staged name; never per verification pass (28
    # and more reads per file without the memo).
    assert len(signatures) == len(set(signatures)), names
    per_inode: dict[tuple, int] = {}
    for signature in signatures:
        per_inode[signature[:2]] = per_inode.get(signature[:2], 0) + 1
    assert max(per_inode.values()) <= 4, per_inode


def test_organizing_without_a_memo_reads_every_file_on_every_pass(tmp_path, reads, monkeypatch):
    source = _staging(tmp_path)
    monkeypatch.setattr(
        organizer,
        "_verify_parent_imports",
        lambda *args, **kwargs: {"fixture": "admission_stub_not_authority_evidence"},
    )
    plan = organizer.build_transfer_organization_plan(
        source, destination_root=tmp_path / "recordings"
    )
    organizer.prepare_transfer_parent_recordings(plan)
    organizer.finalize_transfer_staging(plan)
    assert reads.count("recording_session.json") >= 6


def test_organizer_refuses_a_copy_rewritten_under_a_warm_memo(tmp_path, reads, monkeypatch):
    source = _staging(tmp_path)
    monkeypatch.setattr(
        organizer,
        "_verify_parent_imports",
        lambda *args, **kwargs: {"fixture": "admission_stub_not_authority_evidence"},
    )
    with transfer.digest_memo():
        plan = organizer.build_transfer_organization_plan(
            source, destination_root=tmp_path / "recordings"
        )
        organizer.prepare_transfer_parent_recordings(plan)
        target = next(
            organizer._parent_directory(plan, t["recording_id"]) / t["relative_path"]
            for item in plan["files"]
            if item["source"]["path"].endswith("_full.csv")
            for t in item["destinations"]
        )
        before = target.stat()
        _next_timestamp_tick(target)
        data = bytearray(target.read_bytes())
        data[-2] ^= 0x01
        os.chmod(target, before.st_mode | 0o200)
        target.write_bytes(bytes(data))  # same size; mtime restored below
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(ValueError, match="differs"):
            organizer.finalize_transfer_staging(plan)
    assert json.loads((tmp_path / "recordings").joinpath(
        organizer._state_directory(plan).relative_to(tmp_path / "recordings"),
        "organization_state.json",
    ).read_bytes())["status"] != "complete"


def test_a_file_written_moments_ago_is_not_memoized(tmp_path, reads, monkeypatch):
    monkeypatch.setattr(transfer, "_now_ns", time.time_ns)  # the real clock
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo() as memo:
        first = transfer.file_ref(root, rel)
        assert transfer.file_ref(root, rel) == first
        assert memo.unsettled == 2 and memo.hits == 0
    assert reads == ["a.json", "a.json"]
    # Once the file has settled it is memoized like any other.
    monkeypatch.setattr(transfer, "_now_ns", lambda: time.time_ns() + transfer.DIGEST_MEMO_SETTLE_NS)
    with transfer.digest_memo() as memo:
        assert transfer.file_ref(root, rel) == transfer.file_ref(root, rel) == first
        assert (memo.hits, memo.misses, memo.unsettled) == (1, 1, 0)
    assert reads == ["a.json"] * 3


# ------------------------------------------------------- unsettled entries


def test_unsettled_entries_are_described_by_path_and_age(tmp_path, reads, monkeypatch):
    monkeypatch.setattr(transfer, "_now_ns", time.time_ns)  # the real clock
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo() as memo:
        transfer.file_ref(root, rel)
        transfer.file_ref(root, rel)
        report = memo.report()
    assert (report["hits"], report["misses"], report["unsettled"]) == (0, 2, 2)
    assert report["unsettled_files"] == 1
    assert [entry["path"] for entry in report["unsettled_sample"]] == [rel, rel]
    ages = [entry["age_s"] for entry in report["unsettled_sample"]]
    assert all(0 <= age < transfer.DIGEST_MEMO_SETTLE_NS / 10**9 for age in ages)
    assert report["unsettled_age_s"] == {"min": min(ages), "max": max(ages)}
    assert report["frame_maps"] == {"hits": 0, "misses": 0, "unsettled": 0}
    assert list(report)[:3] == ["hits", "misses", "unsettled"]


def test_the_sample_is_bounded_and_the_counts_are_not(tmp_path, reads, monkeypatch):
    monkeypatch.setattr(transfer, "_now_ns", time.time_ns)
    monkeypatch.setattr(transfer, "UNSETTLED_SAMPLE_SIZE", 3)
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo() as memo:
        for _ in range(5):
            transfer.file_ref(root, rel)
        report = memo.report()
    assert report["unsettled"] == 5 and len(report["unsettled_sample"]) == 3


def test_a_settled_memo_reports_no_unsettled_detail(tmp_path, reads):
    root, rel = _artifact(tmp_path)
    with transfer.digest_memo() as memo:
        transfer.file_ref(root, rel)
        report = memo.report()
    assert report["unsettled"] == 0 and report["unsettled_files"] == 0
    assert report["unsettled_age_s"] is None and report["unsettled_sample"] == []


# ------------------------------------------------------------- frame maps


def _metadata(tmp_path: Path) -> Path:
    source = FIXTURES / "rolling/clips/clip_0000/Cam02010093_full.csv"
    path = tmp_path / source.name
    shutil.copyfile(source, path)
    return path


def test_outside_a_memo_every_frame_map_parses_the_rows(tmp_path, parses):
    path = _metadata(tmp_path)
    first = transfer.frame_map(path, "full", 0, 0)
    assert transfer.frame_map(path, "full", 0, 0) == first
    assert parses == [path.name, path.name]


def test_inside_a_memo_an_unchanged_frame_map_is_parsed_once(tmp_path, parses):
    path = _metadata(tmp_path)
    fresh = transfer.frame_map(path, "full", 0, 0)
    parses.clear()
    with transfer.digest_memo() as memo:
        first = transfer.frame_map(path, "full", 0, 0)
        second = transfer.frame_map(path, "full", 0, 0)
        assert (memo.frame_map_hits, memo.frame_map_misses) == (1, 1)
    assert first == second == fresh
    assert parses == [path.name]
    # The caller owns its copy: mutating one result never changes a later one.
    first["frame_count"] = -1
    with transfer.digest_memo():
        assert transfer.frame_map(path, "full", 0, 0) == fresh


def test_the_arguments_are_part_of_the_frame_map_key(tmp_path, parses):
    path = _metadata(tmp_path)
    with transfer.digest_memo():
        first = transfer.frame_map(path, "full", 0, 0)
        with pytest.raises(transfer.TransferSnapshotError):
            transfer.frame_map(path, "full", first["last_recording_frame_id"], 0)
        with pytest.raises(transfer.TransferSnapshotError):
            transfer.frame_map(path, "crop", 0, 0)
    assert parses == [path.name] * 3


def test_a_row_callback_always_streams_the_rows(tmp_path, parses):
    path = _metadata(tmp_path)
    streamed: list[int] = []
    with transfer.digest_memo() as memo:
        first = transfer.frame_map(path, "full", 0, 0)
        again = transfer.frame_map(path, "full", 0, 0, on_row=lambda *row: streamed.append(row[1]))
        assert (memo.frame_map_hits, memo.frame_map_misses) == (0, 1)
    assert first == again and len(streamed) == first["frame_count"]
    assert parses == [path.name, path.name]


def test_a_rewritten_metadata_file_is_parsed_again(tmp_path, parses):
    path = _metadata(tmp_path)
    with transfer.digest_memo():
        first = transfer.frame_map(path, "full", 0, 0)
        _next_timestamp_tick(path)
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-1]))  # one row fewer
        second = transfer.frame_map(path, "full", 0, 0)
    assert second["frame_count"] == first["frame_count"] - 1
    assert parses == [path.name, path.name]


def test_a_frame_map_of_a_file_written_moments_ago_is_not_memoized(tmp_path, parses, monkeypatch):
    monkeypatch.setattr(transfer, "_now_ns", time.time_ns)
    path = _metadata(tmp_path)
    with transfer.digest_memo() as memo:
        assert transfer.frame_map(path, "full", 0, 0) == transfer.frame_map(path, "full", 0, 0)
        report = memo.report()
    assert report["frame_maps"] == {"hits": 0, "misses": 2, "unsettled": 2}
    assert report["unsettled_files"] == 1 and report["unsettled_sample"][0]["path"] == str(path)
    assert parses == [path.name, path.name]


def test_snapshot_verification_under_a_memo_parses_each_frame_map_once(tmp_path, parses):
    root = _staging(tmp_path)
    fresh = transfer.verify_transfer_snapshot(root)
    fresh_parses = sorted(parses)
    assert fresh_parses  # every clip output's metadata is validated
    parses.clear()
    with transfer.digest_memo():
        first = transfer.verify_transfer_snapshot(root)
        assert sorted(parses) == fresh_parses
        second = transfer.verify_transfer_snapshot(root)
        assert sorted(parses) == fresh_parses  # nothing parsed again
    assert first.snapshot == second.snapshot == fresh.snapshot


def test_a_changed_metadata_file_under_a_memo_is_refused_by_verification(tmp_path, parses):
    root = _staging(tmp_path)
    with transfer.digest_memo():
        transfer.verify_transfer_snapshot(root)
        path = next(root.rglob("*_full.csv"))
        _next_timestamp_tick(path)
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-1]))
        with pytest.raises(transfer.TransferSnapshotError):
            transfer.verify_transfer_snapshot(root)
