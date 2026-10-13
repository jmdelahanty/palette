"""Write-once seal for transfer-v2 parent ``recording_manifest.json`` files.

The transfer-v2 organizer (``organize_transfer_recordings._parent_manifest``)
writes each parent manifest exactly once. Import receipt re-verification and
organizer resume compare that file byte-for-byte against the organization
plan, so it must stay immutable afterwards. The organizer marks those
manifests with a ``source_transfer`` binding.

Every in-place manifest mutator calls :func:`require_unsealed_recording_manifest`
before writing. Legacy manifests (no ``source_transfer``) are unaffected.
The organizer's own write-once/verify-on-resume path does not use this guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

RECORDING_MANIFEST_NAME = "recording_manifest.json"
SEALED_MANIFEST_MARKER = "source_transfer"


class SealedRecordingManifestError(RuntimeError):
    """An in-place write targeted a sealed (or unverifiable) recording manifest."""


def is_sealed_recording_manifest(payload: Mapping[str, Any]) -> bool:
    """Return whether ``payload`` is a transfer-v2 parent manifest."""

    return SEALED_MANIFEST_MARKER in payload


def require_unsealed_recording_manifest(manifest_path: Path, *, tool: str) -> None:
    """Refuse an in-place rewrite of a sealed transfer-v2 parent manifest.

    The on-disk file is the authority, not the caller's in-memory copy. A
    missing file is not sealed. A file that exists but is not a readable JSON
    object is refused, because its seal state cannot be established.
    """

    path = Path(manifest_path)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise SealedRecordingManifestError(
            f"{tool}: refusing to rewrite {path}: cannot read it to confirm it is "
            f"not a sealed transfer-v2 manifest ({exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise SealedRecordingManifestError(
            f"{tool}: refusing to rewrite {path}: root is not a JSON object, so it "
            "cannot be confirmed as not a sealed transfer-v2 manifest"
        )
    if is_sealed_recording_manifest(payload):
        raise SealedRecordingManifestError(
            f"{tool}: refusing to modify {path}: it is sealed by transfer-v2 intake "
            f"(carries '{SEALED_MANIFEST_MARKER}') and is immutable after organize"
        )
