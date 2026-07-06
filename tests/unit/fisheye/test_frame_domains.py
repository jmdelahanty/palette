from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.frame_domains import (
    FRAME_DOMAIN_MAPS_GROUP,
    STORED_ZARR_TO_ACQUISITION_MAP,
    FrameDomain,
    FrameDomainUnmappedError,
    _normalize_domain,
)
from fisheye.shared.recording import open_recording

from .frame_domain_fixtures import (
    build_crop_video_drop_store,
    build_full_identity_store,
    build_full_missing_mapping_store,
    build_subsampled_store,
)


def _edge(capabilities, source: FrameDomain, target: FrameDomain):
    for edge in capabilities.edges:
        if edge.source is source and edge.target is target:
            return edge
    raise AssertionError(f"edge missing: {source.value} -> {target.value}")


def _legacy_convert_loop(
    self,
    values,
    source: FrameDomain | str,
    target: FrameDomain | str,
    *,
    stage: str | None = None,
    run: str | None = None,
) -> np.ndarray:
    """Copy of the pre-vectorized ``FrameDomains.convert`` loop used as an oracle."""

    self._assert_scope(stage=stage, run=run)
    source_domain = _normalize_domain(source)
    target_domain = _normalize_domain(target)
    input_values = np.asarray(values)
    flat_values = input_values.reshape(-1)
    if source_domain is target_domain:
        return np.asarray(input_values, dtype=np.int64).copy()

    edge = self._edges.get((source_domain, target_domain))
    if edge is None:
        raise FrameDomainUnmappedError(
            f"no explicit frame-domain edge from {source_domain.value} to {target_domain.value}",
            source=source_domain,
            target=target_domain,
            stage=self.stage,
            run_name=self.run_name,
        )

    output = np.empty(flat_values.shape[0], dtype=np.int64)
    missing: list[int] = []
    for idx, raw_value in enumerate(flat_values.tolist()):
        value = int(raw_value)
        mapped = edge.mapping.get(value)
        if mapped is None:
            missing.append(value)
            continue
        output[idx] = mapped
    if missing:
        sample = tuple(missing[:10])
        raise FrameDomainUnmappedError(
            f"{len(missing)} value(s) are unmappable from {source_domain.value} "
            f"to {target_domain.value}; first values: {sample}",
            source=source_domain,
            target=target_domain,
            stage=self.stage,
            run_name=self.run_name,
            mapping_arrays=edge.public.mapping_arrays,
            values=sample,
        )
    return output.reshape(input_values.shape)


def _assert_matches_legacy_convert(
    domains,
    values,
    source: FrameDomain,
    target: FrameDomain,
) -> None:
    expected = _legacy_convert_loop(domains, values, source, target)
    actual = domains.convert(values, source, target)

    assert actual.dtype == np.dtype(np.int64)
    assert actual.shape == np.asarray(values).shape
    np.testing.assert_array_equal(actual, expected)


def _assert_convert_error_parity(
    domains,
    values,
    source: FrameDomain,
    target: FrameDomain,
    *,
    stage: str | None = None,
    run: str | None = None,
) -> FrameDomainUnmappedError | ValueError:
    with pytest.raises((FrameDomainUnmappedError, ValueError)) as legacy:
        _legacy_convert_loop(domains, values, source, target, stage=stage, run=run)
    with pytest.raises(type(legacy.value)) as actual:
        domains.convert(values, source, target, stage=stage, run=run)

    assert str(actual.value) == str(legacy.value)
    if isinstance(legacy.value, FrameDomainUnmappedError):
        assert isinstance(actual.value, FrameDomainUnmappedError)
        assert actual.value.source is legacy.value.source
        assert actual.value.target is legacy.value.target
        assert actual.value.stage == legacy.value.stage
        assert actual.value.run_name == legacy.value.run_name
        assert actual.value.mapping_arrays == legacy.value.mapping_arrays
        assert actual.value.values == legacy.value.values
    return actual.value


def test_vectorized_convert_matches_legacy_loop_for_shapes_and_edges(tmp_path: Path) -> None:
    zarr_path = build_subsampled_store(tmp_path)
    domains = open_recording(zarr_path).frame_domains()

    cases = (
        (
            np.asarray([[2, 0]], dtype=np.int32),
            FrameDomain.STORED_ZARR,
            FrameDomain.STORED_ZARR,
        ),
        (
            np.asarray([0, 1, 2], dtype=np.int32),
            FrameDomain.STORED_ZARR,
            FrameDomain.ACQUISITION,
        ),
        (
            np.asarray([0, 2, 4], dtype=np.int16),
            FrameDomain.ACQUISITION,
            FrameDomain.STORED_ZARR,
        ),
        (
            np.asarray([[0, 1], [2, 0]], dtype=np.int64),
            FrameDomain.STORED_ZARR,
            FrameDomain.ACQUISITION,
        ),
        (
            np.asarray([[0, 2], [4, 0]], dtype=np.int64),
            FrameDomain.ACQUISITION,
            FrameDomain.STORED_ZARR,
        ),
        (
            np.asarray([], dtype=np.int64).reshape(0, 2),
            FrameDomain.STORED_ZARR,
            FrameDomain.ACQUISITION,
        ),
        (
            np.asarray([1], dtype=np.int32),
            FrameDomain.STORED_ZARR,
            FrameDomain.ACQUISITION,
        ),
    )
    for values, source, target in cases:
        _assert_matches_legacy_convert(domains, values, source, target)


def test_vectorized_lookup_is_built_lazily_per_edge(tmp_path: Path) -> None:
    zarr_path = build_subsampled_store(tmp_path)
    domains = open_recording(zarr_path).frame_domains()
    forward_edge = domains._edges[(FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION)]
    inverse_edge = domains._edges[(FrameDomain.ACQUISITION, FrameDomain.STORED_ZARR)]

    assert forward_edge._lookup is None
    assert inverse_edge._lookup is None

    domains.count(FrameDomain.STORED_ZARR)
    domains.capabilities()

    assert forward_edge._lookup is None
    assert inverse_edge._lookup is None

    domains.convert([0, 1, 2], FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION)

    assert forward_edge._lookup is not None
    assert inverse_edge._lookup is None


def test_vectorized_convert_preserves_error_parity(tmp_path: Path) -> None:
    zarr_path = build_crop_video_drop_store(tmp_path)
    domains = open_recording(zarr_path).frame_domains(stage="crop", run="crop_001")

    forward_error = _assert_convert_error_parity(
        domains,
        np.asarray(
            [[99, 0, -1, 100], [2, 10, 42, 7], [8, 9, 11, 12], [13, 3, 14, 15]]
        ),
        FrameDomain.CROP_VIDEO,
        FrameDomain.ACQUISITION,
    )
    assert isinstance(forward_error, FrameDomainUnmappedError)
    assert forward_error.values == (99, -1, 100, 10, 42, 7, 8, 9, 11, 12)
    assert forward_error.mapping_arrays == (
        "crop_runs/crop_001/source_crop_video_frame_indices",
        "crop_runs/crop_001/source_frame_indices",
    )

    inverse_error = _assert_convert_error_parity(
        domains,
        np.asarray(
            [[10, 0, 99, 2], [6, 42, 8, 1], [3, 5, 7, 9], [11, 8, 6, 2]]
        ),
        FrameDomain.ACQUISITION,
        FrameDomain.CROP_VIDEO,
    )
    assert isinstance(inverse_error, FrameDomainUnmappedError)
    assert inverse_error.values == (10, 99, 42, 1, 3, 5, 7, 9, 11)
    assert inverse_error.mapping_arrays == (
        "crop_runs/crop_001/source_crop_video_frame_indices",
        "crop_runs/crop_001/source_frame_indices",
    )

    _assert_convert_error_parity(
        domains,
        [0],
        FrameDomain.CROP_VIDEO,
        FrameDomain.ACQUISITION,
        run="crop_other",
    )

    missing_zarr_path = build_full_missing_mapping_store(tmp_path)
    missing_domains = open_recording(missing_zarr_path).frame_domains()
    _assert_convert_error_parity(
        missing_domains,
        [0, 1],
        FrameDomain.STORED_ZARR,
        FrameDomain.ACQUISITION,
    )


def test_full_import_identity_map_round_trips_through_resolver(tmp_path: Path) -> None:
    zarr_path = build_full_identity_store(tmp_path)

    domains = open_recording(zarr_path).frame_domains()

    assert domains.count(FrameDomain.STORED_ZARR) == 5
    assert domains.count(FrameDomain.ACQUISITION) == 5
    assert domains.convert([0, 1, 4], FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION).tolist() == [0, 1, 4]
    assert domains.convert([0, 3, 4], FrameDomain.ACQUISITION, FrameDomain.STORED_ZARR).tolist() == [0, 3, 4]

    capabilities = domains.capabilities()
    edge = _edge(capabilities, FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION)
    assert edge.mapping_arrays == (
        f"raw_video/{FRAME_DOMAIN_MAPS_GROUP}/{STORED_ZARR_TO_ACQUISITION_MAP}",
    )
    assert edge.confidence == "explicit"


def test_subsampled_original_frame_indices_map_stored_to_acquisition(tmp_path: Path) -> None:
    zarr_path = build_subsampled_store(tmp_path)

    domains = open_recording(zarr_path).frame_domains()

    assert domains.count(FrameDomain.STORED_ZARR) == 3
    assert domains.count(FrameDomain.ACQUISITION) == 5
    assert domains.count(FrameDomain.SOURCE_VIDEO) == 5
    assert domains.convert(np.asarray([0, 1, 2]), FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION).tolist() == [0, 2, 4]
    assert domains.convert([0, 2, 4], FrameDomain.ACQUISITION, FrameDomain.STORED_ZARR).tolist() == [0, 1, 2]
    assert domains.convert([0, 1, 2], FrameDomain.STORED_ZARR, FrameDomain.SOURCE_VIDEO).tolist() == [0, 2, 4]


def test_crop_video_drop_fixture_converts_observed_video_frames(tmp_path: Path) -> None:
    zarr_path = build_crop_video_drop_store(tmp_path)

    domains = open_recording(zarr_path).frame_domains(stage="crop", run="crop_001")

    assert domains.count(FrameDomain.RUN_FRAME) == 12
    assert domains.count(FrameDomain.CROP_VIDEO) == 4
    assert domains.convert([0, 1, 2, 3], FrameDomain.CROP_VIDEO, FrameDomain.ACQUISITION).tolist() == [0, 2, 6, 8]
    assert domains.convert([0, 2, 6, 8], FrameDomain.ACQUISITION, FrameDomain.CROP_VIDEO).tolist() == [0, 1, 2, 3]

    capabilities = domains.capabilities()
    edge = _edge(capabilities, FrameDomain.CROP_VIDEO, FrameDomain.ACQUISITION)
    assert edge.run_name == "crop_001"
    assert edge.parent_path == "crop_runs"
    assert edge.mapping_arrays == (
        "crop_runs/crop_001/source_crop_video_frame_indices",
        "crop_runs/crop_001/source_frame_indices",
    )


def test_unmappable_supplemental_and_out_of_range_values_fail_loud(tmp_path: Path) -> None:
    zarr_path = build_crop_video_drop_store(tmp_path)
    domains = open_recording(zarr_path).frame_domains(stage="crop", run="crop_001")

    with pytest.raises(FrameDomainUnmappedError, match="unmappable") as negative:
        domains.convert([-1], FrameDomain.CROP_VIDEO, FrameDomain.ACQUISITION)
    assert negative.value.source is FrameDomain.CROP_VIDEO
    assert negative.value.target is FrameDomain.ACQUISITION
    assert negative.value.values == (-1,)

    with pytest.raises(FrameDomainUnmappedError, match="unmappable") as supplemental:
        domains.convert([10], FrameDomain.ACQUISITION, FrameDomain.CROP_VIDEO)
    assert supplemental.value.values == (10,)

    with pytest.raises(FrameDomainUnmappedError, match="unmappable"):
        domains.convert([99], FrameDomain.CROP_VIDEO, FrameDomain.ACQUISITION)


def test_missing_mapping_is_reported_without_silent_identity_fallback(tmp_path: Path) -> None:
    zarr_path = build_full_missing_mapping_store(tmp_path)

    domains = open_recording(zarr_path).frame_domains()
    capabilities = domains.capabilities()

    assert domains.count(FrameDomain.STORED_ZARR) == 5
    assert any("stored_zarr_frame -> acquisition_frame" in item for item in capabilities.missing)
    with pytest.raises(FrameDomainUnmappedError, match="no explicit frame-domain edge"):
        domains.convert([0, 1], FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION)


def test_crop_video_mapping_requires_explicit_acquisition_row_mapping(tmp_path: Path) -> None:
    zarr_path = build_crop_video_drop_store(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    del root["crop_runs"]["crop_001"]["source_frame_indices"]

    domains = open_recording(zarr_path).frame_domains(stage="crop", run="crop_001")

    assert any("no explicit acquisition-frame row mapping" in item for item in domains.capabilities().missing)
    with pytest.raises(FrameDomainUnmappedError, match="no explicit frame-domain edge"):
        domains.convert([0], FrameDomain.CROP_VIDEO, FrameDomain.ACQUISITION)


def test_crop_video_mapping_reports_missing_crop_video_array(tmp_path: Path) -> None:
    zarr_path = build_crop_video_drop_store(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    del root["crop_runs"]["crop_001"]["source_crop_video_frame_indices"]

    domains = open_recording(zarr_path).frame_domains(stage="crop", run="crop_001")

    assert any("no source_crop_video_frame_indices" in item for item in domains.capabilities().missing)
    with pytest.raises(FrameDomainUnmappedError, match="no explicit frame-domain edge"):
        domains.convert([0], FrameDomain.CROP_VIDEO, FrameDomain.ACQUISITION)


def test_recording_frame_domains_rejects_wrong_scope(tmp_path: Path) -> None:
    zarr_path = build_crop_video_drop_store(tmp_path)
    domains = open_recording(zarr_path).frame_domains(stage="crop", run="crop_001")

    with pytest.raises(ValueError, match="constructed for stage"):
        domains.count(FrameDomain.RUN_FRAME, stage="detect")
    with pytest.raises(ValueError, match="constructed for run"):
        domains.convert([0], FrameDomain.CROP_VIDEO, FrameDomain.ACQUISITION, run="crop_other")


def test_full_identity_fixture_uses_new_map_not_original_frame_indices(tmp_path: Path) -> None:
    zarr_path = build_full_identity_store(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="r")

    assert "original_frame_indices" not in root["raw_video"]
    assert (
        root["raw_video"][FRAME_DOMAIN_MAPS_GROUP][STORED_ZARR_TO_ACQUISITION_MAP][:]
    ).tolist() == [0, 1, 2, 3, 4]


def test_real_nvme_sampled_store_smoke_round_trips_original_frame_indices() -> None:
    zarr_path = Path(
        "/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/"
        "zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_training.zarr"
    )
    if not zarr_path.exists():
        pytest.skip("/nvme1 sampled training store is not available on this machine")

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    original = np.asarray(root["raw_video"]["original_frame_indices"][:], dtype=np.int64)
    sample = np.asarray([0, 1, min(10, original.shape[0] - 1)], dtype=np.int64)

    domains = open_recording(zarr_path).frame_domains()
    resolved = domains.convert(sample, FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION)
    round_trip = domains.convert(resolved, FrameDomain.ACQUISITION, FrameDomain.STORED_ZARR)

    assert resolved.tolist() == original[sample].tolist()
    assert round_trip.tolist() == sample.tolist()
    assert "raw_video/original_frame_indices" in _edge(
        domains.capabilities(),
        FrameDomain.STORED_ZARR,
        FrameDomain.ACQUISITION,
    ).mapping_arrays
