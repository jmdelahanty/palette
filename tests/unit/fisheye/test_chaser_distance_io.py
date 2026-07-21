from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
import zarr

from fisheye.analysis import chaser_distance_io as io_module
from fisheye.analysis.chaser_distance_io import (
    CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_ID,
    ChaserDistanceReadError,
    ChaserDistanceReadSnapshot,
    UNAVAILABLE_BEHAVIOR_AUTHORITY_STATUS,
    VERIFIED_AUTHORITY_STATUS,
    load_chaser_distance_run,
    resolve_chaser_distance_run_path,
)
from fisheye.analysis.goodcopbadcop_common import open_distance_run
from fisheye.analysis.chaser_quadrant_occupancy import (
    build_chaser_quadrant_occupancy_result,
)
from fisheye.analysis.chaser_radial_occupancy import (
    build_chaser_radial_occupancy_result,
)
from tests.unit.fisheye.test_chaser_distance_coordinate_publication import (
    _publish_canonical,
)


@pytest.fixture(scope="module")
def published_canonical_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str]:
    template_parent = tmp_path_factory.mktemp("chaser-distance-io-template")
    zarr_path, _root, run = _publish_canonical(template_parent)
    return zarr_path, run.path


@pytest.fixture
def published_canonical_archive(
    tmp_path: Path,
    published_canonical_template: tuple[Path, str],
) -> tuple[Path, zarr.Group, zarr.Group]:
    template, run_path = published_canonical_template
    target = tmp_path / template.name
    shutil.copytree(template, target)
    root = zarr.open_group(str(target), mode="a", use_consolidated=False)
    return target, root, root[run_path]


def test_reader_returns_typed_detached_immutable_snapshot(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, run = published_canonical_archive
    run.attrs.update(
        {
            # Normal readers must ignore stale historical aliases even when a
            # migration tool accidentally leaves them behind.
            "coordinate_frame": "texture",
            "coordinate_origin": "bottom_right",
            "pixels_per_mm_projector": 999.0,
            "fps": 1.0,
            "total_frames": 999,
        }
    )

    snapshot = load_chaser_distance_run(root)

    assert snapshot.authority_status == VERIFIED_AUTHORITY_STATUS
    assert snapshot.run_path == run.path
    assert snapshot.coordinate_space_id == "arena_relative_canvas_px"
    assert snapshot.coordinate_origin == "arena_top_left"
    assert (snapshot.positive_x, snapshot.positive_y) == ("right", "down")
    assert snapshot.pixel_convention == "continuous"
    assert (snapshot.reference_width_px, snapshot.reference_height_px) == (344, 344)
    assert snapshot.pixels_per_mm_projector == 5.0
    assert snapshot.fps == 120.0
    assert snapshot.total_frames == 2
    assert snapshot.fish_centroid_arena_xy.flags.writeable is False
    assert snapshot.distance_mm.flags.writeable is False
    with pytest.raises(ValueError):
        snapshot.distance_mm[0, 0] = 0.0

    authority = snapshot.authority_record()
    assert authority["schema_id"] == CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_ID
    assert authority["run_ref"] == f"/{run.path}"
    assert authority["publication_seal"]["record_sha256"] == (
        snapshot.publication_seal_sha256
    )
    authority["publication_seal"]["record_sha256"] = "mutated"
    assert snapshot.authority_record()["publication_seal"]["record_sha256"] != (
        "mutated"
    )


def test_latest_uses_exact_lifecycle_pointer_not_latest_or_child_order(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, canonical = published_canonical_archive
    parent = root["analysis/chaser_distance_runs"]
    parent.create_group("zzzz_legacy_or_partial")
    parent.attrs["latest"] = "zzzz_legacy_or_partial"

    name, path = resolve_chaser_distance_run_path(root, run_name="latest")
    snapshot = load_chaser_distance_run(root, run_name="latest")

    assert name == canonical.path.rsplit("/", 1)[-1]
    assert path == canonical.path
    assert snapshot.run_path == canonical.path


@pytest.mark.parametrize(
    "pointer",
    ["missing", "nested/run", "", 7],
)
def test_authoritative_pointer_fails_closed_without_fallback(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
    pointer: str | int,
) -> None:
    _zarr_path, root, _run = published_canonical_archive
    parent = root["analysis/chaser_distance_runs"]
    parent.attrs["authoritative_run"] = pointer

    with pytest.raises(ChaserDistanceReadError):
        load_chaser_distance_run(root)


def test_explicit_legacy_child_is_not_a_normal_read(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, _run = published_canonical_archive
    parent = root["analysis/chaser_distance_runs"]
    legacy = parent.create_group("legacy")
    legacy.attrs.update(
        {
            "coordinate_frame": "arena_relative_canvas_px",
            "coordinate_origin": "top_left_of_active_arena",
            "pixels_per_mm_projector": 5.0,
        }
    )

    with pytest.raises(
        ChaserDistanceReadError,
        match="complete coordinate publication",
    ):
        load_chaser_distance_run(root, run_name="legacy")


def test_reader_revalidates_after_copy(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _zarr_path, root, run = published_canonical_archive
    original = io_module.load_bound_chaser_distance_run
    calls = 0

    def mutate_before_second_verification(root_node, run_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            node = root_node[f"{run_path}/distances/distance_mm"]
            node[0, 0] = np.float32(float(node[0, 0]) + 1.0)
        return original(root_node, run_path)

    monkeypatch.setattr(
        io_module,
        "load_bound_chaser_distance_run",
        mutate_before_second_verification,
    )
    with pytest.raises(ChaserDistanceReadError):
        load_chaser_distance_run(root, run_name=run.path.rsplit("/", 1)[-1])
    assert calls == 2


def test_behavior_roles_fail_until_their_semantic_authority_is_sealed(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, _run = published_canonical_archive
    snapshot = load_chaser_distance_run(root)

    assert snapshot.behavior_authority_status == UNAVAILABLE_BEHAVIOR_AUTHORITY_STATUS
    with pytest.raises(ChaserDistanceReadError, match="behavior-role/color"):
        snapshot.require_behavior_authority()


def test_derived_surface_fails_until_its_payload_authority_is_sealed(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, _run = published_canonical_archive
    snapshot = load_chaser_distance_run(root)

    with pytest.raises(
        ChaserDistanceReadError,
        match="no independently verified canonical publication seal",
    ) as error:
        snapshot.require_derived_surface_authority(
            "chaser_bout_response/exact_component/bouts"
        )

    message = str(error.value)
    assert "Remediation: republish" in message
    assert "latest/sorted fallback are forbidden" in message


def test_snapshot_cannot_be_constructed_directly() -> None:
    with pytest.raises(ChaserDistanceReadError, match="cannot be constructed"):
        ChaserDistanceReadSnapshot()


def test_goodcopbadcop_common_uses_the_verified_reader_boundary(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, run = published_canonical_archive

    snapshot = open_distance_run(root)

    assert isinstance(snapshot, ChaserDistanceReadSnapshot)
    assert snapshot.run_path == run.path


def test_radial_consumer_fails_until_arena_geometry_authority_is_sealed(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    zarr_path, _root, run = published_canonical_archive
    run.attrs.update(
        {
            "coordinate_frame": "texture",
            "coordinate_origin": "bottom_right",
            "pixels_per_mm_projector": 999.0,
            "fps": 1.0,
        }
    )

    with pytest.raises(
        ChaserDistanceReadError,
        match="sealed arena centre/radius geometry",
    ) as error:
        build_chaser_radial_occupancy_result(
            zarr_path,
            chaser_distance_run=run.path.rsplit("/", 1)[-1],
            settle_trim_s=0.0,
        )

    assert "root attrs and inferred canvas geometry are forbidden" in str(error.value)


def test_role_dependent_consumer_fails_until_semantics_are_sealed(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    zarr_path, _root, run = published_canonical_archive

    with pytest.raises(ChaserDistanceReadError, match="behavior-role/color"):
        build_chaser_quadrant_occupancy_result(
            zarr_path,
            chaser_distance_run=run.path.rsplit("/", 1)[-1],
        )
