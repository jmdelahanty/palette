"""Persistent canonical source archives for subject-mask finalizer tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import zarr

from tests.persistent_fixture_cache import persistent_directory_fixture


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_ROOT = _REPO_ROOT / ".pytest_cache" / "palette-subject-mask-finalizer-fixture"
_MATCHED = "matched.zarr"
_MISMATCHED = "mismatched.zarr"


def _builder_sha256() -> str:
    """Hash the test-local builders whose output is persisted."""

    from tests.unit.fisheye import test_finalize_subject_masks as finalizer_source
    from tests.unit.fisheye import (
        test_keypoint_coordinate_publication as keypoint_source,
    )

    digest = hashlib.sha256()
    for function in (
        keypoint_source._real_canonical_archive,
        finalizer_source._publish_real_canonical_subject_mask,
    ):
        payload = inspect.getsource(function).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _build_templates(destination: Path) -> None:
    """Build matched and intentionally mismatched immutable source graphs."""

    from tests.unit.fisheye.test_finalize_subject_masks import (
        _publish_real_canonical_subject_mask,
    )
    from tests.unit.fisheye.test_keypoint_coordinate_publication import (
        _real_canonical_archive,
    )

    destination.mkdir()
    builder = destination.parent / "matched-builder"
    builder.mkdir()
    root, _keypoints = _real_canonical_archive(
        builder,
        include_bilateral_eyes=True,
    )
    _publish_real_canonical_subject_mask(root)
    matched = destination / _MATCHED
    (builder / "canonical.zarr").replace(matched)
    builder.rmdir()

    builder = destination.parent / "mismatched-builder"
    builder.mkdir()
    mismatch_root, _keypoints = _real_canonical_archive(
        builder,
        include_bilateral_eyes=True,
        selected_crop_rows=np.asarray([0, 1], dtype="<i8"),
    )
    _publish_real_canonical_subject_mask(mismatch_root)
    mismatched = destination / _MISMATCHED
    (builder / "canonical.zarr").replace(mismatched)
    builder.rmdir()


def _validate_template(path: Path) -> None:
    """Validate immutable cache structure without synchronous Zarr reads."""

    def attributes(archive: Path, relative_path: str) -> dict[str, object]:
        metadata = json.loads(
            (archive / relative_path / "zarr.json").read_text(encoding="utf-8")
        )
        attrs = metadata.get("attributes")
        if not isinstance(attrs, dict):
            raise ValueError(
                f"Cached Zarr node {archive.name}/{relative_path!r} lacks attributes."
            )
        return attrs

    for name in (_MATCHED, _MISMATCHED):
        archive = path / name
        root_attrs = attributes(archive, "")
        keypoint_attrs = attributes(archive, "keypoints_runs/k1")
        subject_parent_attrs = attributes(archive, "subject_mask_runs")
        subject_attrs = attributes(archive, "subject_mask_runs/s1")
        if (
            root_attrs.get("recording_id") != "recording-1"
            or keypoint_attrs.get("palette_run_completion_status") != "complete"
            or keypoint_attrs.get("stage_selector_eligible") is not True
            or subject_parent_attrs.get("latest") != "s1"
            or subject_parent_attrs.get("latest_complete") != "s1"
            or subject_attrs.get("palette_run_completion_status") != "complete"
            or subject_attrs.get("stage_selector_eligible") is not True
            or subject_attrs.get("coordinate_contract") != "canonical_v2"
        ):
            raise ValueError(
                f"Cached subject-mask finalizer archive {name!r} is incomplete."
            )


def resolve_subject_mask_finalizer_archive_template(*, mismatched: bool) -> Path:
    """Return one validated immutable canonical finalizer source archive."""

    fixture = persistent_directory_fixture(
        namespace="canonical-subject-mask-finalizer-archives",
        schema_version="canonical-subject-mask-finalizer-archives-v1",
        source_paths=(
            _REPO_ROOT / "src/fisheye/shared",
            _REPO_ROOT / "src/fisheye/refinement/finalize_subject_masks.py",
            _REPO_ROOT / "tests/unit/fisheye/test_keypoint_coordinate_publication.py",
            _REPO_ROOT / "tests/unit/fisheye/test_finalize_subject_masks.py",
            Path(__file__),
        ),
        dependency_versions={
            "builder_sha256": _builder_sha256(),
            "numpy": np.__version__,
            "zarr": zarr.__version__,
        },
        build=_build_templates,
        validate=_validate_template,
        cache_root=_CACHE_ROOT,
    )
    return fixture.path / (_MISMATCHED if mismatched else _MATCHED)


__all__ = ["resolve_subject_mask_finalizer_archive_template"]
