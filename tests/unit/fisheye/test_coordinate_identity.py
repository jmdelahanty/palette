from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.coordinate_identity import (
    INSTANCE_KEY_MODE,
    OBSERVATION_INSTANCE_DOMAIN,
    ROW_IDENTITY_CONTRACT_ATTR,
    ROW_IDENTITY_CONTRACT_DIGEST_ATTR,
    ROW_IDENTITY_KEY_ATTR,
    ROW_IDENTITY_KEY_DIGEST_ATTR,
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_MODE,
    TRACK_SAMPLE_KEY_MODE,
    TRACK_SAMPLE_DOMAIN,
    TRACK_SAMPLE_INTERPOLATION_DTYPE,
    BoundRowIdentityContract,
    RowIdentityContractError,
    build_row_identity_contract,
    build_track_sample_key,
    derive_track_source_instance_values,
    identity_array_content_sha256,
    load_row_identity_contract_attrs,
    load_bound_row_identity_contract,
    parse_row_identity_contract,
    row_identity_contract_attrs,
    row_identity_key_attrs,
    require_bound_row_identity_contract,
    require_bound_source_row_temporal_authority,
    require_bound_track_sample_time_lineage,
    load_bound_source_row_temporal_authority,
    load_bound_track_sample_time_lineage,
    resolve_source_acquisition_frame_indices,
    stamp_row_identity_contract,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
    stamp_track_sample_time_lineage,
    validate_row_identity_contract,
    validate_row_identity_values,
    validate_stamped_row_identity,
)
from fisheye.shared.pixel_frame_authority import (
    load_acquisition_camera_frame,
    load_acquisition_import_ownership,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
)
from fisheye.shared.proof_verification import proof_verification_scope


class _Node:
    _archive = object()

    def __init__(
        self,
        values: np.ndarray | None = None,
        *,
        path: str = "rowset",
    ) -> None:
        self.attrs: dict[str, object] = {}
        self._values = None if values is None else np.asarray(values)
        self.shape = () if self._values is None else self._values.shape
        self.dtype = np.dtype("V0") if self._values is None else self._values.dtype
        self.path = path
        self._coordinate_archive_token = self._archive
        self.read_count = 0

    def __getitem__(self, item):
        if self._values is None:
            raise TypeError("group-like node is not sliceable")
        self.read_count += 1
        return self._values[item]

    def __setitem__(self, item, value) -> None:
        if self._values is None:
            raise TypeError("group-like node is not sliceable")
        self._values[item] = value


class _FailOnceAttrs(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail = True

    def update(self, *args, **kwargs) -> None:
        if self.fail:
            self.fail = False
            super().update({"partial": True})
            raise RuntimeError("injected update failure")
        super().update(*args, **kwargs)


class _NoOpAttrs(dict):
    def update(self, *args, **kwargs) -> None:
        return None


class _MutateKeyOnUpdateAttrs(dict):
    def __init__(self, *args, victim: _Node, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.victim = victim
        self.mutated = False

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        if not self.mutated:
            self.mutated = True
            assert self.victim._values is not None
            self.victim._values[0] += 100


def _issue_codes(exc: RowIdentityContractError) -> set[str]:
    return {issue.code for issue in exc.issues}


def _track_lineage(
    values: np.ndarray,
    *,
    rowset_path: str = "tracks",
    total_frames: int = 200,
    source_instance_keys: np.ndarray | None = None,
) -> tuple[_Node, _Node, object]:
    source_rowset = _Node(path=f"{rowset_path}_source")
    source_key_values = (
        np.arange(41, 41 + values.shape[0], dtype=np.uint64)
        if source_instance_keys is None
        else np.asarray(source_instance_keys)
    )
    source_key = _Node(
        source_key_values,
        path=f"{source_rowset.path}/instance_key",
    )
    source_identity = stamp_and_bind_row_identity_contract(
        source_rowset,
        source_key,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=source_key_values,
        ),
    )
    source_frame = _Node(
        values[:, 1].astype(np.int64, copy=True),
        path=f"{source_rowset.path}/source_acquisition_frame_index",
    )
    rowset = _Node(path=rowset_path)
    key = _Node(values, path=f"{rowset_path}/track_sample_key")
    root = _Node(path="archive_root")
    root.attrs = {
        "recording_id": "recording-1",
        "source_video_metadata": {
            "schema_id": "palette.source_video_metadata.v2",
            "layout": "single_video",
            "camera_id": "camera-a",
            "width": 160,
            "height": 120,
            "total_frames": total_frames,
            "locator": {
                "kind": "recording_relative",
                "relative_path": "camera.mp4",
            },
            "file_fingerprint": {
                "strategy": "size_mtime_sha256_v1",
                "value": "a" * 64,
                "size_bytes": 1234,
                "mtime_ns": 5678,
                "relocation_stable": False,
            },
        },
    }
    acquisition_node = _Node(path="analysis/acquisition_camera_frames/camera-a")
    ownership = stamp_acquisition_import_ownership(root, acquisition_node)
    acquisition = stamp_acquisition_camera_frame(
        root,
        acquisition_node,
        import_ownership=ownership,
    )
    source_temporal = stamp_source_row_temporal_authority(
        source_rowset,
        source_frame,
        source_row_identity=source_identity,
        acquisition_frame=acquisition,
    )
    source_row_index = _Node(
        np.arange(values.shape[0], dtype=np.int64),
        path=f"{rowset_path}/source_row_index",
    )
    output_source_frame = _Node(
        resolve_source_acquisition_frame_indices(
            source_temporal,
            source_row_index[:],
        ),
        path=f"{rowset_path}/source_acquisition_frame_index",
    )
    interpolation_values = np.zeros((values.shape[0],), dtype=TRACK_SAMPLE_INTERPOLATION_DTYPE)
    interpolation_values["left_source_frame_index"] = values[:, 1]
    interpolation_values["right_source_frame_index"] = values[:, 1]
    interpolation = _Node(
        interpolation_values,
        path=f"{rowset_path}/source_frame_interpolation",
    )
    source_instance_values = derive_track_source_instance_values(
        source_temporal,
        source_row_index[:],
    )
    source_instance = _Node(
        source_instance_values,
        path=f"{rowset_path}/source_instance_key",
    )
    lineage = stamp_track_sample_time_lineage(
        rowset,
        key,
        source_row_index,
        output_source_frame,
        interpolation,
        source_instance,
        source_temporal_authority=source_temporal,
    )
    return rowset, key, lineage


def test_observation_instance_contract_is_exact_and_digest_bound() -> None:
    values = np.asarray([9, 17, 42], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )

    assert contract.domain == OBSERVATION_INSTANCE_DOMAIN
    assert contract.mode == INSTANCE_KEY_MODE
    assert contract.leading_dimension == 3
    assert contract.unique is True
    assert contract.key_array.ref == "instance_key"
    assert contract.key_array.components == ("instance_key",)
    assert contract.key_array.dtype == "<u8"
    assert contract.key_array.shape == (3,)
    assert contract.key_array.content_sha256 == identity_array_content_sha256(values)
    assert parse_row_identity_contract(contract.to_dict()) == contract
    assert parse_row_identity_contract(contract.canonical_json()) == contract

    reordered = {key: contract.to_dict()[key] for key in reversed(contract.to_dict())}
    assert parse_row_identity_contract(reordered).digest() == contract.digest()


def test_track_sample_identity_is_track_and_acquisition_frame_not_instance_key() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4, 8], dtype=np.int32),
        np.asarray([100, 101, 100], dtype=np.int64),
    )
    _, _, time_lineage = _track_lineage(values)
    contract = build_row_identity_contract(
        domain=TRACK_SAMPLE_DOMAIN,
        values=values,
        track_time_lineage=time_lineage,
    )

    assert values.dtype == np.int64
    np.testing.assert_array_equal(values, [[4, 100], [4, 101], [8, 100]])
    assert contract.mode == TRACK_SAMPLE_KEY_MODE
    assert contract.key_array.ref == "track_sample_key"
    assert contract.key_array.components == (
        "track_id",
        "acquisition_frame_index",
    )
    assert contract.key_array.shape == (3, 2)

    forged = contract.to_dict()
    forged["mode"] = INSTANCE_KEY_MODE
    forged["key_arrays"][0]["mode"] = INSTANCE_KEY_MODE
    forged["key_arrays"][0]["ref"] = "instance_key"
    forged["key_arrays"][0]["components"] = ["instance_key"]
    forged["key_arrays"][0]["dtype"] = "<u8"
    assert "track_identity_profile_mismatch" in {
        issue.code for issue in validate_row_identity_contract(forged)
    }


def test_track_key_rejects_duplicate_samples_and_negative_values() -> None:
    duplicate = np.asarray([[2, 10], [2, 10]], dtype=np.int64)
    _, _, duplicate_lineage = _track_lineage(duplicate)
    with pytest.raises(RowIdentityContractError) as duplicate_exc:
        build_row_identity_contract(
            domain=TRACK_SAMPLE_DOMAIN,
            values=duplicate,
            track_time_lineage=duplicate_lineage,
        )
    assert "identity_uniqueness_required" in _issue_codes(duplicate_exc.value)

    with pytest.raises(RowIdentityContractError) as negative_exc:
        build_track_sample_key(
            np.asarray([2], dtype=np.int64),
            np.asarray([-1], dtype=np.int64),
        )
    assert "identity_value_negative" in _issue_codes(negative_exc.value)

    with pytest.raises(RowIdentityContractError) as overflow_exc:
        build_track_sample_key(
            np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
            np.asarray([1], dtype=np.uint64),
        )
    assert "identity_value_overflow" in _issue_codes(overflow_exc.value)


def test_track_key_api_does_not_accept_ambiguous_frame_indices_keyword() -> None:
    with pytest.raises(TypeError, match="frame_indices"):
        build_track_sample_key(
            track_ids=np.asarray([1], dtype=np.int64),
            frame_indices=np.asarray([2], dtype=np.int64),  # type: ignore[call-arg]
        )


def test_track_identity_publication_requires_sealed_acquisition_time_lineage() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    rowset, key, lineage = _track_lineage(values, total_frames=3)
    contract = build_row_identity_contract(
        domain=TRACK_SAMPLE_DOMAIN,
        values=values,
        track_time_lineage=lineage,
    )
    with pytest.raises(RowIdentityContractError) as missing_exc:
        stamp_and_bind_row_identity_contract(
            rowset,
            key,
            contract=contract,
        )
    assert "track_time_lineage_unverified" in _issue_codes(missing_exc.value)

    bound = stamp_and_bind_row_identity_contract(
        rowset,
        key,
        contract=contract,
        track_time_lineage=lineage,
    )
    assert bound.track_time_lineage is lineage
    assert bound.contract.time_lineage is not None
    assert require_bound_row_identity_contract(bound) is bound


def test_track_time_lineage_rejects_out_of_range_and_mapping_drift() -> None:
    out_of_range = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([100, 101], dtype=np.int64),
    )
    with pytest.raises(RowIdentityContractError) as range_exc:
        _track_lineage(out_of_range, total_frames=3)
    assert "source_time_frame_out_of_range" in _issue_codes(range_exc.value)

    values = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    _, _, lineage = _track_lineage(values, total_frames=3)
    lineage._source_frame_index_node._values[0] = 0
    with pytest.raises(RowIdentityContractError) as drift_exc:
        lineage.assert_verified()
    assert {
        "track_source_row_frame_mismatch",
        "track_time_lineage_noncanonical",
    } & _issue_codes(drift_exc.value)


def test_track_time_lineage_rejects_wrong_interpolation_formula() -> None:
    values = build_track_sample_key(
        np.asarray([4], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )
    rowset, key, lineage = _track_lineage(values, rowset_path="interpolated", total_frames=3)
    interpolation_values = lineage._interpolation_node._values
    assert interpolation_values is not None
    interpolation_values["left_source_frame_index"] = 0
    interpolation_values["right_source_frame_index"] = 2
    interpolation_values["right_weight"] = 0.25
    with pytest.raises(RowIdentityContractError) as exc_info:
        stamp_track_sample_time_lineage(
            rowset,
            key,
            lineage._source_row_index_node,
            lineage._source_frame_index_node,
            lineage._interpolation_node,
            lineage._source_instance_key_node,
            source_temporal_authority=lineage._source_temporal_authority,
        )
    assert "track_interpolation_unsupported_v1" in _issue_codes(exc_info.value)


def test_track_source_instance_key_is_derived_lineage_not_primary_identity() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4, 4], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    rowset, key, lineage = _track_lineage(values, total_frames=3)
    assert lineage.record.source_instance_key.ref.endswith("/source_instance_key")
    assert np.all(lineage._source_instance_key_node._values["valid"])
    assert np.array_equal(
        lineage._source_instance_key_node._values["instance_key"],
        np.asarray([41, 42, 43], dtype=np.uint64),
    )
    contract = build_row_identity_contract(
        domain=TRACK_SAMPLE_DOMAIN,
        values=values,
        track_time_lineage=lineage,
    )
    assert contract.key_array.ref == "track_sample_key"
    with pytest.raises(RowIdentityContractError):
        build_row_identity_contract(
            domain=TRACK_SAMPLE_DOMAIN,
            values=lineage._source_instance_key_node[:],
            track_time_lineage=lineage,
        )

    lineage._source_instance_key_node._values["instance_key"][0] = np.iinfo(np.uint64).max
    with pytest.raises(RowIdentityContractError) as exc_info:
        lineage.assert_verified()
    assert "track_source_instance_not_derived" in _issue_codes(exc_info.value)


def test_track_subset_reorder_derives_exact_source_instances_41_73() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    rowset, key, lineage = _track_lineage(
        values,
        total_frames=3,
        source_instance_keys=np.asarray([41, 73], dtype=np.uint64),
    )
    source_rows = lineage._source_row_index_node._values
    assert source_rows is not None
    source_rows[:] = [1, 0]
    expected_frames = resolve_source_acquisition_frame_indices(
        lineage._source_temporal_authority,
        source_rows,
    )
    lineage._source_frame_index_node._values[:] = expected_frames
    key._values[:, 1] = expected_frames
    lineage._interpolation_node._values["left_source_frame_index"] = expected_frames
    lineage._interpolation_node._values["right_source_frame_index"] = expected_frames
    lineage._interpolation_node._values["right_weight"] = 0.0
    expected_instances = derive_track_source_instance_values(
        lineage._source_temporal_authority,
        source_rows,
    )
    lineage._source_instance_key_node._values[:] = expected_instances

    rebound = stamp_track_sample_time_lineage(
        rowset,
        key,
        lineage._source_row_index_node,
        lineage._source_frame_index_node,
        lineage._interpolation_node,
        lineage._source_instance_key_node,
        source_temporal_authority=lineage._source_temporal_authority,
    )
    np.testing.assert_array_equal(
        rebound._source_instance_key_node._values["instance_key"],
        np.asarray([73, 41], dtype=np.uint64),
    )


def test_track_lineage_rejects_duplicate_source_rows_and_wrong_source_frame() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    rowset, key, lineage = _track_lineage(values, total_frames=3)
    lineage._source_row_index_node._values[:] = [0, 0]
    with pytest.raises(RowIdentityContractError) as duplicate_exc:
        stamp_track_sample_time_lineage(
            rowset,
            key,
            lineage._source_row_index_node,
            lineage._source_frame_index_node,
            lineage._interpolation_node,
            lineage._source_instance_key_node,
            source_temporal_authority=lineage._source_temporal_authority,
        )
    assert "source_row_index_not_unique" in _issue_codes(duplicate_exc.value)

    lineage._source_row_index_node._values[:] = [1, 0]
    with pytest.raises(RowIdentityContractError) as frame_exc:
        stamp_track_sample_time_lineage(
            rowset,
            key,
            lineage._source_row_index_node,
            lineage._source_frame_index_node,
            lineage._interpolation_node,
            lineage._source_instance_key_node,
            source_temporal_authority=lineage._source_temporal_authority,
        )
    assert "track_source_row_frame_mismatch" in _issue_codes(frame_exc.value)


def test_non_observation_source_requires_canonical_null_instance_lineage() -> None:
    seed_values = build_track_sample_key(
        np.asarray([1, 1], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )
    _, _, seed_lineage = _track_lineage(seed_values, total_frames=3)
    acquisition = seed_lineage._source_temporal_authority.acquisition_frame

    source_rowset = _Node(path="stimulus_source")
    source_keys = np.asarray([11, 12], dtype=np.int64)
    source_key = _Node(source_keys, path="stimulus_source/stimulus_state_key")
    source_identity = stamp_and_bind_row_identity_contract(
        source_rowset,
        source_key,
        contract=build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=source_keys,
            components=("stimulus_state_index",),
        ),
    )
    source_frames = _Node(
        np.asarray([0, 1], dtype=np.int64),
        path="stimulus_source/source_acquisition_frame_index",
    )
    authority = stamp_source_row_temporal_authority(
        source_rowset,
        source_frames,
        source_row_identity=source_identity,
        acquisition_frame=acquisition,
    )
    assert authority.record.observation_instance_key is None

    rowset = _Node(path="stimulus_tracks")
    key_values = build_track_sample_key(
        np.asarray([8, 8], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )
    key = _Node(key_values, path="stimulus_tracks/track_sample_key")
    source_rows = _Node(
        np.asarray([0, 1], dtype=np.int64),
        path="stimulus_tracks/source_row_index",
    )
    output_frames = _Node(
        resolve_source_acquisition_frame_indices(authority, source_rows[:]),
        path="stimulus_tracks/source_acquisition_frame_index",
    )
    interpolation_values = np.zeros(2, dtype=TRACK_SAMPLE_INTERPOLATION_DTYPE)
    interpolation_values["left_source_frame_index"] = output_frames[:]
    interpolation_values["right_source_frame_index"] = output_frames[:]
    interpolation = _Node(
        interpolation_values,
        path="stimulus_tracks/source_frame_interpolation",
    )
    source_instances = _Node(
        derive_track_source_instance_values(authority, source_rows[:]),
        path="stimulus_tracks/source_instance_key",
    )
    lineage = stamp_track_sample_time_lineage(
        rowset,
        key,
        source_rows,
        output_frames,
        interpolation,
        source_instances,
        source_temporal_authority=authority,
    )
    assert np.all(~lineage._source_instance_key_node._values["valid"])
    assert np.all(lineage._source_instance_key_node._values["instance_key"] == 0)

    lineage._source_instance_key_node._values["valid"][0] = True
    lineage._source_instance_key_node._values["instance_key"][0] = 41
    with pytest.raises(RowIdentityContractError) as exc_info:
        lineage.assert_verified()
    assert "track_source_instance_not_derived" in _issue_codes(exc_info.value)


def test_temporal_authorities_reload_fresh_from_persisted_nodes() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    rowset, key, lineage = _track_lineage(values, total_frames=3)
    source = lineage._source_temporal_authority
    fresh_source_identity = load_bound_row_identity_contract(
        source._source_rowset_node,
        source.source_row_identity._key_array_node,
    )
    persisted_acquisition = source.acquisition_frame
    fresh_ownership = load_acquisition_import_ownership(
        persisted_acquisition._root_node,
        persisted_acquisition._authority_node,
    )
    fresh_acquisition = load_acquisition_camera_frame(
        persisted_acquisition._root_node,
        persisted_acquisition._authority_node,
        import_ownership=fresh_ownership,
    )
    fresh_source = load_bound_source_row_temporal_authority(
        source._source_rowset_node,
        source._source_frame_index_node,
        source_row_identity=fresh_source_identity,
        acquisition_frame=fresh_acquisition,
    )
    fresh_track = load_bound_track_sample_time_lineage(
        rowset,
        key,
        lineage._source_row_index_node,
        lineage._source_frame_index_node,
        lineage._interpolation_node,
        lineage._source_instance_key_node,
        source_temporal_authority=fresh_source,
    )
    assert fresh_source.record == source.record
    assert fresh_track.record == lineage.record

    with pytest.raises(RowIdentityContractError) as forged_exc:
        require_bound_source_row_temporal_authority(copy.deepcopy(fresh_source))
    assert "source_temporal_authority_unverified" in _issue_codes(forged_exc.value)


def test_identity_authorities_reuse_one_operation_proof_and_recheck_on_close() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    rowset, key, lineage = _track_lineage(values, total_frames=3)
    identity = stamp_and_bind_row_identity_contract(
        rowset,
        key,
        contract=build_row_identity_contract(
            domain=TRACK_SAMPLE_DOMAIN,
            values=values,
            track_time_lineage=lineage,
        ),
        track_time_lineage=lineage,
    )
    source = lineage._source_temporal_authority
    nodes = (
        source.source_row_identity._key_array_node,
        source._source_frame_index_node,
        key,
        lineage._source_row_index_node,
        lineage._source_frame_index_node,
        lineage._interpolation_node,
        lineage._source_instance_key_node,
    )
    for node in nodes:
        node.read_count = 0

    with proof_verification_scope():
        require_bound_source_row_temporal_authority(source)
        require_bound_track_sample_time_lineage(lineage)
        require_bound_row_identity_contract(identity)
        first_counts = tuple(node.read_count for node in nodes)

        require_bound_source_row_temporal_authority(source)
        require_bound_track_sample_time_lineage(lineage)
        require_bound_row_identity_contract(identity)
        assert tuple(node.read_count for node in nodes) == first_counts

    closing_counts = tuple(node.read_count for node in nodes)
    assert all(after > before for before, after in zip(first_counts, closing_counts))


def test_identity_authority_closing_proof_rejects_temporal_mutation() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    _, _, lineage = _track_lineage(values, total_frames=3)
    source = lineage._source_temporal_authority

    with pytest.raises(RowIdentityContractError) as exc_info:
        with proof_verification_scope():
            require_bound_source_row_temporal_authority(source)
            source._source_frame_index_node._values[0] += 1
    assert "source_temporal_authority_noncanonical" in _issue_codes(exc_info.value)


@pytest.mark.parametrize(
    ("components", "values"),
    (
        (("stimulus_frame_num",), np.asarray([0, 1, 2], dtype=np.int64)),
        (
            ("chaser_index", "stimulus_frame_num"),
            np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        ),
    ),
)
def test_stimulus_state_identity_accepts_explicit_scalar_or_composite_key(
    components: tuple[str, ...],
    values: np.ndarray,
) -> None:
    contract = build_row_identity_contract(
        domain=STIMULUS_STATE_DOMAIN,
        values=values,
        components=components,
    )
    assert contract.mode == STIMULUS_STATE_KEY_MODE
    assert contract.key_array.ref == "stimulus_state_key"
    assert contract.key_array.components == components
    assert contract.key_array.dtype == "<i8"


def test_stimulus_identity_does_not_accept_fake_instance_or_freshness_keys() -> None:
    values = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(RowIdentityContractError) as missing_components:
        build_row_identity_contract(domain=STIMULUS_STATE_DOMAIN, values=values)
    assert "stimulus_identity_components_required" in _issue_codes(
        missing_components.value
    )

    with pytest.raises(RowIdentityContractError) as signature_exc:
        build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=values,
            components=("source_row_signature",),
        )
    assert "stimulus_identity_component_unsupported" in _issue_codes(
        signature_exc.value
    )


def test_contract_json_rejects_duplicate_keys_recursively() -> None:
    values = np.asarray([1, 2], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    payload = contract.canonical_json().replace(
        '"domain":"observation_instance"',
        '"domain":"track_sample","domain":"observation_instance"',
        1,
    )
    with pytest.raises(RowIdentityContractError) as exc_info:
        parse_row_identity_contract(payload)
    assert "row_identity_json_invalid" in _issue_codes(exc_info.value)


def test_group_and_key_array_attrs_cross_bind_the_same_contract() -> None:
    values = np.asarray([11, 12], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    rowset = _Node()
    key = _Node(values, path="rowset/instance_key")
    rowset.attrs["unrelated"] = "keep"
    key.attrs["unrelated"] = "keep"

    stamped = stamp_row_identity_contract(rowset, key, contract=contract)

    assert stamped == contract
    assert rowset.attrs["unrelated"] == "keep"
    assert key.attrs["unrelated"] == "keep"
    assert load_row_identity_contract_attrs(rowset.attrs) == contract
    assert validate_stamped_row_identity(rowset, key) == contract
    assert key.attrs["row_identity_contract_sha256"] == contract.digest()

    wrong_parent = _Node(values, path="other/instance_key")
    with pytest.raises(RowIdentityContractError) as path_exc:
        stamp_row_identity_contract(rowset, wrong_parent, contract=contract)
    assert "key_array_ref_mismatch" in _issue_codes(path_exc.value)


@pytest.mark.parametrize(
    "path",
    (
        "rowset/../rowset/instance_key",
        "rowset/./instance_key",
        "/rowset/instance_key",
        "rowset//instance_key",
        "rowset/instance_key/",
    ),
)
def test_identity_nodes_reject_noncanonical_equivalent_paths(path: str) -> None:
    values = np.asarray([11, 12], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    rowset = _Node()
    key = _Node(values, path=path)

    with pytest.raises(RowIdentityContractError) as exc_info:
        stamp_row_identity_contract(rowset, key, contract=contract)
    assert "row_identity_node_path_noncanonical" in _issue_codes(exc_info.value)


def test_value_verification_cannot_be_disabled_for_publication_or_validation() -> None:
    values = np.asarray([11, 12], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    rowset = _Node()
    key = _Node(values, path="rowset/instance_key")

    with pytest.raises(RowIdentityContractError) as stamp_exc:
        stamp_row_identity_contract(
            rowset,
            key,
            contract=contract,
            verify_values=False,
        )
    assert _issue_codes(stamp_exc.value) == {
        "row_identity_value_verification_required"
    }
    assert rowset.attrs == {}
    assert key.attrs == {}

    stamp_row_identity_contract(rowset, key, contract=contract)
    changed = _Node(
        np.asarray([11, 99], dtype=np.uint64),
        path="rowset/instance_key",
    )
    changed.attrs.update(copy.deepcopy(key.attrs))
    with pytest.raises(RowIdentityContractError) as validate_exc:
        validate_stamped_row_identity(rowset, changed, verify_values=False)
    assert "row_identity_value_verification_required" in _issue_codes(
        validate_exc.value
    )


def test_bound_identity_is_derived_from_exact_stamped_sibling_nodes() -> None:
    values = np.asarray([11, 12], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    rowset = _Node(path="analysis/detect_runs/d1")
    key = _Node(values, path="analysis/detect_runs/d1/instance_key")

    bound = stamp_and_bind_row_identity_contract(
        rowset,
        key,
        contract=contract,
    )
    assert bound.record_ref == (
        "/analysis/detect_runs/d1@row_identity_contract"
    )
    assert bound.record_sha256 == contract.digest()
    assert bound.leading_dimension == 2
    assert load_bound_row_identity_contract(rowset, key) == bound

    key._values = np.asarray([11, 99], dtype=np.uint64)
    with pytest.raises(RowIdentityContractError) as stale_exc:
        bound.assert_verified()
    assert "key_content_digest_mismatch" in _issue_codes(stale_exc.value)


def test_free_constructed_bound_identity_is_not_writer_evidence() -> None:
    values = np.asarray([11, 12], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    forged = BoundRowIdentityContract(
        contract=contract,
        rowset_path="analysis/detect_runs/d1",
        key_array_path="analysis/detect_runs/d1/instance_key",
        record_ref="/analysis/detect_runs/d1@row_identity_contract",
        record_sha256=contract.digest(),
        track_time_lineage=None,
        _archive_identity=archive_identity(_Node(path="analysis/other")),
        _rowset_node=_Node(path="analysis/detect_runs/d1"),
        _key_array_node=_Node(
            values,
            path="analysis/detect_runs/d1/instance_key",
        ),
        _verification_seal=object(),
    )
    with pytest.raises(RowIdentityContractError) as exc_info:
        forged.assert_verified()
    assert _issue_codes(exc_info.value) == {"row_identity_binding_unverified"}


def test_row_identity_subclass_is_rejected_by_central_writer_boundary() -> None:
    class _Subclass(BoundRowIdentityContract):
        pass

    values = np.asarray([11, 12], dtype=np.uint64)
    rowset = _Node(path="analysis/detect_runs/d1")
    key = _Node(values, path="analysis/detect_runs/d1/instance_key")
    bound = stamp_and_bind_row_identity_contract(
        rowset,
        key,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=values,
        ),
    )
    subclass = _Subclass(
        contract=bound.contract,
        rowset_path=bound.rowset_path,
        key_array_path=bound.key_array_path,
        record_ref=bound.record_ref,
        record_sha256=bound.record_sha256,
        track_time_lineage=bound.track_time_lineage,
        _archive_identity=bound.archive_identity,
        _rowset_node=bound._rowset_node,
        _key_array_node=bound._key_array_node,
        _verification_seal=bound._verification_seal,
    )
    with pytest.raises(RowIdentityContractError) as exc_info:
        require_bound_row_identity_contract(subclass)
    assert _issue_codes(exc_info.value) == {"row_identity_binding_unverified"}
    with pytest.raises(RowIdentityContractError):
        subclass.assert_verified()


def test_tampered_group_key_metadata_or_values_fail_closed() -> None:
    values = np.asarray([11, 12], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    rowset = _Node()
    key = _Node(values, path="rowset/instance_key")
    stamp_row_identity_contract(rowset, key, contract=contract)

    broken_group = copy.deepcopy(rowset.attrs)
    broken_group[ROW_IDENTITY_CONTRACT_ATTR]["leading_dimension"] = 99
    with pytest.raises(RowIdentityContractError) as group_exc:
        load_row_identity_contract_attrs(broken_group)
    assert _issue_codes(group_exc.value).intersection(
        {"key_leading_dimension_mismatch", "row_identity_contract_digest_mismatch"}
    )

    broken_key = _Node(values, path="rowset/instance_key")
    broken_key.attrs.update(copy.deepcopy(key.attrs))
    broken_key.attrs[ROW_IDENTITY_KEY_ATTR]["content_sha256"] = "0" * 64
    with pytest.raises(RowIdentityContractError) as key_exc:
        validate_stamped_row_identity(rowset, broken_key)
    assert "row_identity_key_digest_mismatch" in _issue_codes(key_exc.value)

    changed_values = _Node(
        np.asarray([11, 99], dtype=np.uint64),
        path="rowset/instance_key",
    )
    changed_values.attrs.update(copy.deepcopy(key.attrs))
    with pytest.raises(RowIdentityContractError) as values_exc:
        validate_stamped_row_identity(rowset, changed_values)
    assert "key_content_digest_mismatch" in _issue_codes(values_exc.value)


def test_attrs_helpers_are_compact_and_deterministic() -> None:
    values = np.asarray([5], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    group_attrs = row_identity_contract_attrs(contract)
    key_attrs = row_identity_key_attrs(contract)

    assert set(group_attrs) == {
        ROW_IDENTITY_CONTRACT_ATTR,
        ROW_IDENTITY_CONTRACT_DIGEST_ATTR,
    }
    assert group_attrs[ROW_IDENTITY_CONTRACT_DIGEST_ATTR] == contract.digest()
    assert key_attrs[ROW_IDENTITY_KEY_DIGEST_ATTR] == contract.key_array.digest()
    assert json.loads(contract.canonical_json()) == contract.to_dict()


def test_value_validation_detects_wrong_dtype_shape_uniqueness_and_digest() -> None:
    values = np.asarray([7, 8], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )

    wrong_dtype = validate_row_identity_values(
        contract,
        np.asarray([7, 8], dtype=np.int64),
    )
    assert {issue.code for issue in wrong_dtype} == {
        "key_dtype_mismatch",
        "key_content_digest_mismatch",
    }

    duplicate = validate_row_identity_values(
        contract,
        np.asarray([7, 7], dtype=np.uint64),
    )
    assert {issue.code for issue in duplicate} == {
        "identity_values_not_unique",
        "key_content_digest_mismatch",
    }


def test_stamp_validates_before_mutation_and_rolls_back_partial_write() -> None:
    values = np.asarray([3, 4], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    rowset = _Node()
    key = _Node(values, path="rowset/instance_key")
    rowset.attrs = {"before": 1}
    key.attrs = _FailOnceAttrs({"before": 2})

    with pytest.raises(RowIdentityContractError) as exc_info:
        stamp_row_identity_contract(rowset, key, contract=contract)
    assert "row_identity_stamp_preflight_failed" in _issue_codes(exc_info.value)
    assert rowset.attrs == {"before": 1}
    assert key.attrs == {"before": 2}

    wrong = _Node(
        np.asarray([3, 9], dtype=np.uint64),
        path="rowset/instance_key",
    )
    wrong.attrs = {"before": 3}
    with pytest.raises(RowIdentityContractError) as preflight_exc:
        stamp_row_identity_contract(rowset, wrong, contract=contract)
    assert "key_content_digest_mismatch" in _issue_codes(preflight_exc.value)
    assert rowset.attrs == {"before": 1}
    assert wrong.attrs == {"before": 3}


def test_stamp_and_bind_rolls_back_both_attrs_and_key_values_on_reload_failure() -> None:
    values = np.asarray([3, 4], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    rowset = _Node(path="rowset")
    key = _Node(values.copy(), path="rowset/instance_key")
    rowset.attrs = _MutateKeyOnUpdateAttrs(
        {"rowset_before": {"nested": [1]}},
        victim=key,
    )
    key.attrs = {"key_before": {"nested": [2]}}
    rowset_before = copy.deepcopy(dict(rowset.attrs))
    key_before = copy.deepcopy(dict(key.attrs))
    values_before = key[:].copy()

    with pytest.raises(RowIdentityContractError) as exc_info:
        stamp_and_bind_row_identity_contract(rowset, key, contract=contract)
    assert "row_identity_stamp_preflight_failed" in _issue_codes(exc_info.value)
    assert dict(rowset.attrs) == rowset_before
    assert dict(key.attrs) == key_before
    np.testing.assert_array_equal(key[:], values_before)


def test_stamp_rejects_silent_noop_attrs_and_leaves_peer_node_unchanged() -> None:
    values = np.asarray([3, 4], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    rowset = _Node(path="rowset")
    key = _Node(values, path="rowset/instance_key")
    rowset.attrs = _NoOpAttrs({"before": True})
    key.attrs = {"before": 2}

    with pytest.raises(RowIdentityContractError) as exc_info:
        stamp_and_bind_row_identity_contract(rowset, key, contract=contract)
    assert "row_identity_stamp_preflight_failed" in _issue_codes(exc_info.value)
    assert dict(rowset.attrs) == {"before": True}
    assert key.attrs == {"before": 2}


def test_source_and_track_temporal_stampers_reject_hostile_attrs_prewrite() -> None:
    values = build_track_sample_key(
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    track_rowset, track_key, lineage = _track_lineage(values, total_frames=3)
    source = lineage._source_temporal_authority

    source_before = copy.deepcopy(dict(source._source_rowset_node.attrs))
    source._source_rowset_node.attrs = _FailOnceAttrs(source_before)
    with pytest.raises(RowIdentityContractError) as source_exc:
        stamp_source_row_temporal_authority(
            source._source_rowset_node,
            source._source_frame_index_node,
            source_row_identity=source.source_row_identity,
            acquisition_frame=source.acquisition_frame,
        )
    assert "row_identity_stamp_preflight_failed" in _issue_codes(source_exc.value)
    assert dict(source._source_rowset_node.attrs) == source_before
    assert "partial" not in source._source_rowset_node.attrs

    track_before = copy.deepcopy(dict(track_rowset.attrs))
    track_rowset.attrs = _FailOnceAttrs(track_before)
    with pytest.raises(RowIdentityContractError) as track_exc:
        stamp_track_sample_time_lineage(
            track_rowset,
            track_key,
            lineage._source_row_index_node,
            lineage._source_frame_index_node,
            lineage._interpolation_node,
            lineage._source_instance_key_node,
            source_temporal_authority=source,
        )
    assert "row_identity_stamp_preflight_failed" in _issue_codes(track_exc.value)
    assert dict(track_rowset.attrs) == track_before
    assert "partial" not in track_rowset.attrs


def test_identity_schema_versions_require_exact_integers_in_raw_mappings() -> None:
    values = np.asarray([3, 4], dtype=np.uint64)
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    payload = contract.to_dict()
    payload["schema_version"] = 1.0
    with pytest.raises(RowIdentityContractError) as contract_exc:
        parse_row_identity_contract(payload)
    assert "identity_schema_unsupported" in _issue_codes(contract_exc.value)

    rowset = _Node(path="rowset")
    key = _Node(values, path="rowset/instance_key")
    stamp_row_identity_contract(rowset, key, contract=contract)
    key.attrs[ROW_IDENTITY_KEY_ATTR]["schema_version"] = 1.0
    with pytest.raises(RowIdentityContractError) as key_exc:
        validate_stamped_row_identity(rowset, key)
    assert "key_schema_unsupported" in _issue_codes(key_exc.value)
