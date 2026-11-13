"""Transfer organization directory identity survives a change of NFS client host.

NFS st_dev is assigned per client: one inode read 98 in a batch job, 51 on
ws1 and 78 on login1 (2026-10-28), so every cross-host resume of a delivery
refused "directory ownership lost". v2 records name the export instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fisheye.utils import organize_transfer_recordings as organizer

EXPORT = "prfs.hhmi.org:/groups/johnson"


def _mountinfo(mount_point: Path, device: int, *, fstype: str = "nfs4", source: str = EXPORT) -> str:
    major, minor = os.major(device), os.minor(device)
    escaped = str(mount_point).replace(" ", "\\040")
    return (
        "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
        f"99 22 {major}:{minor} / {escaped} rw,relatime shared:9 - {fstype} {source} rw\n"
    )


def _as_nfs(monkeypatch, directory: Path, *, device: int | None = None, **kwargs) -> None:
    device = directory.lstat().st_dev if device is None else device
    text = _mountinfo(directory.parent, device, **kwargs)
    monkeypatch.setattr(organizer, "_mountinfo_text", lambda: text)


def _on_another_host(monkeypatch, st_dev: int) -> None:
    """The same inode, as another NFS client numbers its device."""
    real = Path.lstat

    def lstat(self):
        values = list(real(self))
        values[2] = st_dev  # os.stat_result order: st_mode, st_ino, st_dev, ...
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", lstat)


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    path = (tmp_path / "delivery").resolve()
    path.mkdir()
    return path


def test_a_v2_nfs_record_names_the_export_and_inode(directory, monkeypatch) -> None:
    _as_nfs(monkeypatch, directory)
    identity = organizer._directory_identity(directory)
    assert identity == {
        "schema_id": organizer.DIRECTORY_IDENTITY_SCHEMA_ID,
        "filesystem": {"kind": "nfs_export", "fstype": "nfs4", "source": EXPORT},
        "st_ino": directory.lstat().st_ino,
    }
    organizer._require_directory(directory, identity)


def test_a_v2_nfs_record_resumes_on_another_host(directory, monkeypatch) -> None:
    _as_nfs(monkeypatch, directory)
    identity = organizer._directory_identity(directory)
    _on_another_host(monkeypatch, 98)
    _as_nfs(monkeypatch, directory, device=98)
    organizer._require_directory(directory, identity)


def test_a_v1_record_resumes_on_another_nfs_host(directory, monkeypatch) -> None:
    recorded = [98, directory.lstat().st_ino]
    _as_nfs(monkeypatch, directory)
    assert directory.lstat().st_dev != 98
    organizer._require_directory(directory, recorded)


def test_a_v1_record_on_the_recording_device_is_still_accepted_locally(directory) -> None:
    current = directory.lstat()
    organizer._require_directory(directory, [current.st_dev, current.st_ino])


def test_a_v1_record_from_another_device_is_refused_on_a_local_filesystem(directory) -> None:
    current = directory.lstat()
    with pytest.raises(Exception, match="directory ownership lost"):
        organizer._require_directory(directory, [current.st_dev + 1, current.st_ino])


@pytest.mark.parametrize("form", ["v1", "v2"])
def test_a_different_inode_is_refused(directory, monkeypatch, form) -> None:
    _as_nfs(monkeypatch, directory)
    identity = organizer._directory_identity(directory)
    replacement = directory.with_name("replacement")
    replacement.mkdir()
    directory.rmdir()
    replacement.rename(directory)
    recorded = [98, identity["st_ino"]] if form == "v1" else identity
    with pytest.raises(Exception, match="directory ownership lost"):
        organizer._require_directory(directory, recorded)


def test_a_v2_record_from_another_export_is_refused(directory, monkeypatch) -> None:
    _as_nfs(monkeypatch, directory, source="otherfs.hhmi.org:/groups/johnson")
    identity = organizer._directory_identity(directory)
    _as_nfs(monkeypatch, directory)
    with pytest.raises(Exception, match="directory ownership lost"):
        organizer._require_directory(directory, identity)


def test_a_v2_local_record_does_not_match_an_nfs_directory(directory, monkeypatch) -> None:
    identity = organizer._directory_identity(directory)
    assert identity["filesystem"] == {"kind": "local_device", "st_dev": directory.lstat().st_dev}
    _as_nfs(monkeypatch, directory)
    with pytest.raises(Exception, match="directory ownership lost"):
        organizer._require_directory(directory, identity)


@pytest.mark.parametrize("form", ["v1", "v2"])
def test_a_symlinked_path_is_refused(directory, monkeypatch, form) -> None:
    _as_nfs(monkeypatch, directory)
    link = directory.with_name("link")
    link.symlink_to(directory)
    current = directory.lstat()
    recorded = [98, current.st_ino] if form == "v1" else organizer._directory_identity(directory)
    with pytest.raises(Exception, match="symlink"):
        organizer._require_directory(link, recorded)


def test_a_v1_record_is_refused_when_the_path_is_not_canonical(directory, monkeypatch) -> None:
    _as_nfs(monkeypatch, directory)
    moved = directory.parent / "elsewhere" / ".." / directory.name
    (directory.parent / "elsewhere").mkdir()
    with pytest.raises(Exception, match="directory ownership lost"):
        organizer._require_directory(moved, [98, directory.lstat().st_ino])


def test_an_nfs_mount_that_does_not_own_the_device_is_refused(directory, monkeypatch) -> None:
    _as_nfs(monkeypatch, directory, device=directory.lstat().st_dev + 1)
    with pytest.raises(Exception, match="does not own its device"):
        organizer._directory_identity(directory)


def test_mount_points_with_escaped_spaces_resolve(tmp_path: Path, monkeypatch) -> None:
    root = (tmp_path / "with space").resolve()
    directory = root / "delivery"
    directory.mkdir(parents=True)
    text = _mountinfo(root, directory.lstat().st_dev)
    monkeypatch.setattr(organizer, "_mountinfo_text", lambda: text)
    assert organizer._directory_identity(directory)["filesystem"]["kind"] == "nfs_export"
