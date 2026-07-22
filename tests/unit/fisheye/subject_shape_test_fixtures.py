"""Shared persistent source archive for subject-shape publication tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import zarr

from tests.persistent_fixture_cache import persistent_directory_fixture


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _canonical_builder_sha256() -> str:
    """Hash only fixture-construction helpers, not unrelated tests."""

    from tests.unit.fisheye import test_subject_shape_coordinate_publication as source

    digest = hashlib.sha256()
    for name in (
        "_snapshot",
        "_fish_masks",
        "_create_canonical_subject_masks",
        "_create_canonical_refined_masks",
        "_canonical_refined_archive",
    ):
        payload = inspect.getsource(getattr(source, name)).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _build_cached_canonical_refined_archive(destination: Path) -> None:
    # Imported lazily because the canonical builder lives in the publication
    # test module that also consumes this shared cache resolver.
    from tests.unit.fisheye.test_subject_shape_coordinate_publication import (
        _canonical_refined_archive,
    )

    build_root = destination.parent / "builder"
    build_root.mkdir()
    _canonical_refined_archive(build_root)
    (build_root / "canonical.zarr").replace(destination)
    build_root.rmdir()


def _validate_cached_canonical_refined_archive(path: Path) -> None:
    """Validate immutable cache metadata without opening synchronous Zarr."""

    def metadata(relative_path: str) -> dict[str, object]:
        value = json.loads(
            (path / relative_path / "zarr.json").read_text(encoding="utf-8")
        )
        if not isinstance(value, dict):
            raise ValueError(f"Cached Zarr node {relative_path!r} is malformed.")
        return value

    def attributes(relative_path: str) -> dict[str, object]:
        attrs = metadata(relative_path).get("attributes")
        if not isinstance(attrs, dict):
            raise ValueError(f"Cached Zarr node {relative_path!r} lacks attributes.")
        return attrs

    root_attrs = attributes("")
    parent_attrs = attributes("refined_subject_masks_runs")
    run_attrs = attributes("refined_subject_masks_runs/r1")
    masks = metadata("refined_subject_masks_runs/r1/masks_roi")
    keys = metadata("refined_subject_masks_runs/r1/instance_key")
    if (
        root_attrs.get("recording_id") != "recording-1"
        or parent_attrs.get("latest") != "r1"
        or parent_attrs.get("latest_complete") != "r1"
        or run_attrs.get("palette_run_completion_status") != "complete"
        or run_attrs.get("stage_selector_eligible") is not True
        or run_attrs.get("coordinate_contract") != "canonical_v2"
        or masks.get("shape") != [2, 4, 40, 40]
        or masks.get("data_type") != "uint8"
        or keys.get("shape") != [2]
        or keys.get("data_type") != "uint64"
    ):
        raise ValueError(
            "Cached canonical refined-mask archive is incomplete or incompatible."
        )


def resolve_canonical_refined_archive_template() -> Path:
    """Return one validated immutable canonical refined-mask source graph."""

    fixture = persistent_directory_fixture(
        namespace="canonical-refined-subject-mask-archive",
        schema_version="canonical-refined-subject-mask-archive-v1",
        source_paths=(
            _REPO_ROOT / "src/fisheye/shared",
            _REPO_ROOT / "tests/unit/fisheye/test_keypoint_coordinate_publication.py",
            Path(__file__),
        ),
        dependency_versions={
            "fixture_builder_sha256": _canonical_builder_sha256(),
            "numpy": np.__version__,
            "zarr": zarr.__version__,
        },
        build=_build_cached_canonical_refined_archive,
        validate=_validate_cached_canonical_refined_archive,
    )
    return fixture.path


__all__ = ["resolve_canonical_refined_archive_template"]
