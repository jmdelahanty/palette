from __future__ import annotations

import copy

import numpy as np

import pytest

import fisheye.shared.coordinate_record as coordinate_record_module
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    CoordinateRecordError,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
    verify_bound_coordinate_record,
)
from fisheye.shared.proof_verification import proof_verification_scope


class _Node:
    _archive = object()

    def __init__(self, *, path: str) -> None:
        self.path = path
        self._coordinate_archive_token = self._archive
        self.attrs: dict[str, object] = {"keep": 1}


class _NoOpAttrs(dict[str, object]):
    def update(self, *args, **kwargs) -> None:
        return None


class _MutateSiblingAttrs(dict[str, object]):
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self["keep"] = 999


def test_stamp_bind_and_reverify_exact_persisted_record() -> None:
    node = _Node(path="analysis/stimulus_runs/run/tracking_data/chaser_states")
    record = {
        "schema_id": "palette.coordinate_import_lineage",
        "schema_version": 1,
        "source_dataset_sha256": "1" * 64,
    }

    bound = stamp_and_bind_persisted_coordinate_record(
        node,
        record,
        attr_name="coordinate_import_lineage",
    )

    assert bound.record_ref == (
        "/analysis/stimulus_runs/run/tracking_data/chaser_states"
        "@coordinate_import_lineage"
    )
    assert len(bound.record_sha256) == 64
    assert bind_persisted_coordinate_record(
        node,
        attr_name="coordinate_import_lineage",
    ).record_sha256 == bound.record_sha256
    assert verify_bound_coordinate_record(bound) is bound


def test_verification_reuses_one_exact_proof_and_rechecks_at_scope_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _Node(path="analysis/run")
    bound = stamp_and_bind_persisted_coordinate_record(
        node,
        {"schema_id": "evidence", "schema_version": 1},
        attr_name="coordinate_lineage",
    )
    original = coordinate_record_module.bind_persisted_coordinate_record
    reloads = 0

    def counting_bind(*args, **kwargs):
        nonlocal reloads
        reloads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        coordinate_record_module,
        "bind_persisted_coordinate_record",
        counting_bind,
    )
    with proof_verification_scope():
        assert verify_bound_coordinate_record(bound) is bound
        assert verify_bound_coordinate_record(bound) is bound
        assert verify_bound_coordinate_record(bound) is bound

    # One initial persisted proof plus one fresh fail-closed closing proof.
    assert reloads == 2


def test_verification_scope_closing_recheck_detects_persisted_drift() -> None:
    node = _Node(path="analysis/run")
    bound = stamp_and_bind_persisted_coordinate_record(
        node,
        {"schema_id": "evidence", "schema_version": 1},
        attr_name="coordinate_lineage",
    )

    with pytest.raises(CoordinateRecordError, match="missing, malformed, or stale"):
        with proof_verification_scope():
            assert verify_bound_coordinate_record(bound) is bound
            node.attrs["coordinate_lineage"] = {
                "schema_id": "evidence",
                "schema_version": 2,
            }


def test_zarr_record_stamp_uses_one_whole_metadata_write(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zarr
    from zarr.core.attributes import Attributes

    root = zarr.open_group(
        str(tmp_path / "authority-stamp.zarr"),
        mode="w",
        zarr_format=3,
    )
    node = root.create_group("analysis/run")
    node.attrs["keep"] = 1
    original_put = Attributes.put
    writes = 0

    def counting_put(self, values):
        nonlocal writes
        writes += 1
        return original_put(self, values)

    monkeypatch.setattr(Attributes, "put", counting_put)
    stamp_and_bind_persisted_coordinate_record(
        node,
        {"schema_id": "evidence", "schema_version": 1},
        attr_name="coordinate_lineage",
    )

    assert writes == 1
    assert node.attrs["keep"] == 1


def test_zarr_record_stamp_rolls_back_with_one_whole_metadata_write(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zarr
    from zarr.core.attributes import Attributes

    root = zarr.open_group(
        str(tmp_path / "authority-rollback.zarr"),
        mode="w",
        zarr_format=3,
    )
    node = root.create_group("analysis/run")
    node.attrs["keep"] = {"nested": [1, 2]}
    before = copy.deepcopy(dict(node.attrs))
    original_put = Attributes.put
    writes = 0

    def counting_put(self, values):
        nonlocal writes
        writes += 1
        return original_put(self, values)

    def fail_reload(*args, **kwargs):
        raise CoordinateRecordError("injected post-write reload failure")

    monkeypatch.setattr(Attributes, "put", counting_put)
    monkeypatch.setattr(
        coordinate_record_module,
        "bind_persisted_coordinate_record",
        fail_reload,
    )
    with pytest.raises(CoordinateRecordError, match="injected post-write"):
        stamp_and_bind_persisted_coordinate_record(
            node,
            {"schema_id": "evidence", "schema_version": 1},
            attr_name="coordinate_lineage",
        )

    assert writes == 2
    assert dict(node.attrs) == before


def test_forged_or_stale_bindings_fail_closed() -> None:
    node = _Node(path="analysis/run")
    bound = stamp_and_bind_persisted_coordinate_record(
        node,
        {"schema_id": "evidence", "schema_version": 1},
        attr_name="coordinate_lineage",
    )
    node.attrs["coordinate_lineage"] = {
        "schema_id": "evidence",
        "schema_version": 2,
    }
    with pytest.raises(CoordinateRecordError, match="missing, malformed, or stale"):
        verify_bound_coordinate_record(bound)

    with pytest.raises(CoordinateRecordError, match="must be loaded"):
        BoundCoordinateRecord(
            record_ref="/analysis/run@coordinate_lineage",
            record_sha256="0" * 64,
            attr_name="coordinate_lineage",
            digest_attr_name="coordinate_lineage_sha256",
            archive=bound.archive_identity,
            node=node,
            record={"schema_id": "forged"},
        )


def test_invalid_record_or_digest_and_noncanonical_path_are_rejected() -> None:
    node = _Node(path="analysis/run")
    node.attrs["coordinate_lineage"] = {"value": float("nan")}
    node.attrs["coordinate_lineage_sha256"] = "0" * 64
    with pytest.raises(CoordinateRecordError, match="finite canonical JSON"):
        bind_persisted_coordinate_record(
            node,
            attr_name="coordinate_lineage",
        )

    bad_path = _Node(path="/analysis/run")
    with pytest.raises(CoordinateRecordError, match="canonical"):
        stamp_and_bind_persisted_coordinate_record(
            bad_path,
            {"schema_id": "evidence"},
            attr_name="coordinate_lineage",
        )
    assert bad_path.attrs == {"keep": 1}


def test_stamp_rolls_back_all_attrs_when_binding_fails() -> None:
    node = _Node(path="/bad/path")
    before = copy.deepcopy(node.attrs)

    with pytest.raises(CoordinateRecordError):
        stamp_and_bind_persisted_coordinate_record(
            node,
            {"schema_id": "evidence", "schema_version": 1},
            attr_name="coordinate_lineage",
        )

    assert node.attrs == before


def test_record_attr_and_digest_names_are_controlled() -> None:
    node = _Node(path="analysis/run")
    with pytest.raises(CoordinateRecordError, match="snake_case"):
        stamp_and_bind_persisted_coordinate_record(
            node,
            {"schema_id": "evidence"},
            attr_name="Coordinate Lineage",
        )
    with pytest.raises(CoordinateRecordError, match="digest attr"):
        stamp_and_bind_persisted_coordinate_record(
            node,
            {"schema_id": "evidence"},
            attr_name="coordinate_lineage",
            digest_attr_name="other_sha256",
        )


@pytest.mark.parametrize(
    "record",
    (
        {"schema_id": "evidence", "values": (1, 2)},
        {"schema_id": "evidence", "value": np.int64(1)},
        {"schema_id": "evidence", "value": np.float64(1.0)},
    ),
)
def test_noncanonical_raw_json_containers_and_scalars_fail_closed(
    record: dict[str, object],
) -> None:
    node = _Node(path="analysis/run")
    with pytest.raises(CoordinateRecordError, match="exact finite JSON"):
        stamp_and_bind_persisted_coordinate_record(
            node,
            record,
            attr_name="coordinate_lineage",
        )
    assert node.attrs == {"keep": 1}

    node.attrs["coordinate_lineage"] = record
    node.attrs["coordinate_lineage_sha256"] = "0" * 64
    with pytest.raises(CoordinateRecordError, match="noncanonical"):
        bind_persisted_coordinate_record(
            node,
            attr_name="coordinate_lineage",
        )


def test_stamp_requires_exact_builtin_dict_before_mutation() -> None:
    class _DictSubclass(dict[str, object]):
        pass

    node = _Node(path="analysis/run")
    before = copy.deepcopy(node.attrs)
    with pytest.raises(CoordinateRecordError, match="exact built-in dict"):
        stamp_and_bind_persisted_coordinate_record(
            node,
            _DictSubclass(schema_id="evidence"),
            attr_name="coordinate_lineage",
        )
    assert node.attrs == before


@pytest.mark.parametrize("attrs_type", (_NoOpAttrs, _MutateSiblingAttrs))
def test_stamp_rejects_hostile_attrs_before_any_write(
    attrs_type: type[dict[str, object]],
) -> None:
    node = _Node(path="analysis/run")
    node.attrs = attrs_type({"keep": 1})
    before = copy.deepcopy(dict(node.attrs))
    with pytest.raises(CoordinateRecordError, match="exact built-in dict"):
        stamp_and_bind_persisted_coordinate_record(
            node,
            {"schema_id": "evidence", "schema_version": 1},
            attr_name="coordinate_lineage",
        )
    assert dict(node.attrs) == before
