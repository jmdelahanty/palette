"""Filesystem provenance helpers for storage benchmarks."""

from __future__ import annotations

import os
import platform
import socket
from pathlib import Path
from typing import Any


def _unescape_mountinfo(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _storage_tier(*, filesystem_type: str, mount_source: str) -> str:
    fs_type = str(filesystem_type).lower()
    source = str(mount_source).lower()
    server = source.split(":", maxsplit=1)[0]
    server_labels = {label for label in server.split(".") if label}
    if fs_type in {"nfs", "nfs4"} and "prfs" in server_labels:
        return "prfs"
    if fs_type in {"nfs", "nfs4", "cifs", "smb", "smb2", "ceph", "lustre"}:
        return "network"
    if fs_type:
        return "local"
    return "unknown"


def describe_filesystem(
    path: Path | str,
    *,
    mountinfo_text: str | None = None,
) -> dict[str, Any]:
    """Describe the deepest Linux mount containing ``path``."""

    resolved = Path(path).expanduser().resolve(strict=False)
    if mountinfo_text is None:
        mountinfo_text = Path("/proc/self/mountinfo").read_text(encoding="utf-8")

    matches: list[dict[str, str]] = []
    for line in mountinfo_text.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        trailing = after.split()
        if len(fields) < 6 or len(trailing) < 2:
            continue
        mount_point = Path(_unescape_mountinfo(fields[4]))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        matches.append(
            {
                "mount_point": str(mount_point),
                "mount_options": fields[5],
                "filesystem_type": trailing[0],
                "mount_source": _unescape_mountinfo(trailing[1]),
            }
        )

    match = max(matches, key=lambda item: len(Path(item["mount_point"]).parts), default=None)
    if match is None:
        match = {
            "mount_point": "",
            "mount_options": "",
            "filesystem_type": "",
            "mount_source": "",
        }
    return {
        "path": str(resolved),
        **match,
        "storage_tier": _storage_tier(
            filesystem_type=match["filesystem_type"],
            mount_source=match["mount_source"],
        ),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": int(os.getpid()),
    }


def require_storage_tier(
    description: dict[str, Any],
    expected: str | None,
    *,
    label: str,
) -> None:
    """Fail closed when a benchmark path is not on its declared storage tier."""

    if expected in {None, "", "auto"}:
        return
    actual = str(description.get("storage_tier") or "unknown")
    if actual != str(expected):
        raise ValueError(
            f"{label} must be on storage tier {expected!r}, got {actual!r}: "
            f"path={description.get('path')} mount={description.get('mount_point')} "
            f"type={description.get('filesystem_type')} source={description.get('mount_source')}"
        )
