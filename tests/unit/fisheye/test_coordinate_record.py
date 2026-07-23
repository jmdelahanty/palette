from __future__ import annotations

import copy

import numpy as np

import pytest

from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    CoordinateRecordError,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
    verify_bound_coordinate_record,
)


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
