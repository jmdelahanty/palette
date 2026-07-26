"""Atomic preservation of fixed-layout Orange recording-geometry bundles."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fisheye.shared.recording_geometry import (
    RECORDING_GEOMETRY_ASSETS_NAME,
    RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH,
    RECORDING_GEOMETRY_CONTRACT_NAME,
    RECORDING_SNAPSHOT_NAME,
    RecordingGeometryBundleVerification,
    RecordingGeometryError,
    verify_recording_geometry_bundle,
)


@dataclass(frozen=True)
class RecordingGeometryBundlePublication:
    verification: RecordingGeometryBundleVerification
    published: bool
    copied_files: tuple[tuple[Path, Path], ...]


def iter_recording_geometry_bundle_files(root: str | Path) -> tuple[Path, ...]:
    bundle_root = Path(root).expanduser().resolve()
    files = [
        bundle_root / RECORDING_SNAPSHOT_NAME,
        bundle_root / RECORDING_GEOMETRY_CONTRACT_NAME,
    ]
    assets_root = bundle_root / RECORDING_GEOMETRY_ASSETS_NAME
    files.extend(sorted(path for path in assets_root.rglob("*") if path.is_file()))
    return tuple(files)


def _same_verification(
    left: RecordingGeometryBundleVerification,
    right: RecordingGeometryBundleVerification,
) -> bool:
    return (
        left.contract_sha256 == right.contract_sha256
        and left.manifest_sha256 == right.manifest_sha256
        and left.manifest_file_count == right.manifest_file_count
        and left.materialized_asset_status == right.materialized_asset_status
        and left.snapshot_pointer_status == right.snapshot_pointer_status
    )


def publish_recording_geometry_bundle(
    *,
    source_root: str | Path,
    recording_root: str | Path,
) -> RecordingGeometryBundlePublication:
    """Verify, copy, and atomically publish one fixed-layout v1 bundle."""

    source_root = Path(source_root).expanduser().resolve()
    recording_root = Path(recording_root).expanduser().resolve()
    source = verify_recording_geometry_bundle(
        source_root,
        require_snapshot_pointer=False,
        verify_all_assets=True,
    )
    destination = recording_root / RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH
    copied_files = tuple(
        (path, destination / path.relative_to(source_root))
        for path in iter_recording_geometry_bundle_files(source_root)
    )
    if destination.exists():
        existing = verify_recording_geometry_bundle(
            destination,
            require_snapshot_pointer=False,
            verify_all_assets=True,
        )
        if not _same_verification(source, existing):
            raise RecordingGeometryError(
                f"Existing geometry bundle conflicts with source: {destination}"
            )
        return RecordingGeometryBundlePublication(
            verification=existing,
            published=False,
            copied_files=(),
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.incoming.",
            dir=str(destination.parent),
        )
    )
    try:
        shutil.copy2(source_root / RECORDING_SNAPSHOT_NAME, temporary / RECORDING_SNAPSHOT_NAME)
        shutil.copy2(
            source_root / RECORDING_GEOMETRY_CONTRACT_NAME,
            temporary / RECORDING_GEOMETRY_CONTRACT_NAME,
        )
        shutil.copytree(
            source_root / RECORDING_GEOMETRY_ASSETS_NAME,
            temporary / RECORDING_GEOMETRY_ASSETS_NAME,
        )
        copied = verify_recording_geometry_bundle(
            temporary,
            require_snapshot_pointer=False,
            verify_all_assets=True,
        )
        if not _same_verification(source, copied):
            raise RecordingGeometryError("Copied geometry bundle does not match its source.")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    published = verify_recording_geometry_bundle(
        destination,
        require_snapshot_pointer=False,
        verify_all_assets=True,
    )
    if not _same_verification(source, published):
        raise RecordingGeometryError("Published geometry bundle failed post-rename verification.")
    return RecordingGeometryBundlePublication(
        verification=published,
        published=True,
        copied_files=copied_files,
    )


__all__ = [
    "RecordingGeometryBundlePublication",
    "iter_recording_geometry_bundle_files",
    "publish_recording_geometry_bundle",
]
