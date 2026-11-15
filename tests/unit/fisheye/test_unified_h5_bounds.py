"""Bounded reads and semantic refusal below the outer file-byte seal."""

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.appearance import validate_appearance_witness
from fisheye.shared.unified_h5.common import KeyIndex, parse_json, same_json, uint64
from fisheye.shared.unified_h5.hdf5_types import (
    check_dataset_budget,
    iter_payload,
    read_bounded_string,
)
from fisheye.shared.unified_h5.metadata import native_attributes
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture


@pytest.mark.parametrize(
    "raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b"[]", b"\xff"]
)
def test_closed_json_refusals(raw):
    with pytest.raises(UnifiedH5ContractError):
        parse_json(raw, label="negative")


def test_receipt_equality_preserves_json_types_and_finite_exponents():
    assert not same_json({"schema_version": True}, {"schema_version": 1})
    assert not same_json({"count": 1.0}, {"count": 1})
    with pytest.raises(UnifiedH5ContractError, match="nonfinite_json"):
        parse_json(b'{"value":1e999}', label="overflow")


@pytest.mark.parametrize("value", [True, -1, 2**64, 1.0, "1"])
def test_no_uint64_narrowing(value):
    with pytest.raises(UnifiedH5ContractError):
        uint64(value, "negative")


def test_disk_index_preserves_full_unsigned_keys():
    with KeyIndex() as index:
        for position, value in enumerate((0, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 1)):
            index.add("frames", (value,), position, reason="duplicate")
            assert index.lookup("frames", (value,)) == position
        with pytest.raises(UnifiedH5ContractError, match="duplicate"):
            index.add("frames", (2**64 - 1,), 99, reason="duplicate")


def test_lossless_fixed_and_variable_text_and_attributes(tmp_path):
    with h5py.File(tmp_path / "strings.h5", "w") as source:
        fixed = source.create_dataset("fixed", data=np.asarray(b"abc", dtype="S32"))
        assert b"".join(iter_payload(fixed)) == b"abc" + b"\0" * 29
        variable = source.create_dataset(
            "variable", data="a" * 20000, dtype=h5py.string_dtype("utf-8")
        )
        assert read_bounded_string(variable) == b"a" * 20000
        with pytest.raises(UnifiedH5ContractError, match="budget"):
            read_bounded_string(variable, limit=100)
        source.attrs["negative_zero"] = np.float64(-0.0)
        source.attrs["unsigned"] = np.uint64(2**64 - 1)
        source.attrs["text"] = "wide " + "x" * 20000
        attrs = native_attributes(source)
        assert attrs["negative_zero"]["payload_hex"] == "0000000000000080"
        assert attrs["unsigned"]["payload_hex"] == "ffffffffffffffff"
        assert (
            bytes.fromhex(attrs["text"]["values_hex"][0]).decode()
            == source.attrs["text"]
        )


def test_resource_profile_matches_the_pinned_admission_contract():
    from fisheye.shared.unified_h5 import common
    from fisheye.shared.unified_h5.schema import contract

    profile = contract("unified_h5_admission_v2.json")["resource_profile"]
    assert profile["max_dataset_logical_bytes"] == common.MAX_DATASET_LOGICAL_BYTES
    assert profile["max_dataset_rows"] == common.MAX_DATASET_ROWS
    assert profile["max_single_element_bytes"] == common.MAX_ELEMENT_BYTES
    assert profile["max_json_payload_bytes"] == common.MAX_JSON_BYTES


@pytest.mark.parametrize(
    ("dtype", "rows", "reason"),
    [
        ("u8", 2**40 // 8, None),  # exact byte limit: floor(2^40 / 8) rows
        ("u8", 2**40 // 8 + 1, "byte_budget"),
        ("u1", 2**40, None),  # exact row limit
        ("u1", 2**40 + 1, "row_budget"),
        ("u8", 2_000_001, None),  # the former 2,000,000-row cap no longer applies
        (f"S{64 * 1024 * 1024}", 1, None),  # exact single-element limit
        (f"S{64 * 1024 * 1024 + 1}", 1, "element_budget"),
    ],
)
def test_long_session_limits_are_checked_before_any_read(tmp_path, dtype, rows, reason):
    with h5py.File(tmp_path / "limits.h5", "w") as source:
        # Chunked and never written: HDF5 allocates nothing for these extents.
        dataset = source.create_dataset("table", shape=(rows,), dtype=dtype, chunks=(1,))
        if reason is None:
            check_dataset_budget(dataset)
        else:
            with pytest.raises(UnifiedH5ContractError, match=reason):
                check_dataset_budget(dataset)


def test_non_little_endian_integers_are_refused(tmp_path):
    with h5py.File(tmp_path / "endian.h5", "w") as source:
        big_endian = source.create_dataset("big_endian", shape=(1,), dtype=">u8")
        with pytest.raises(UnifiedH5ContractError, match="integer_type"):
            check_dataset_budget(big_endian)


@pytest.mark.parametrize(
    "field,value",
    [
        ("drive_valid", 2),
        ("contrast_model_id", 999),
        ("effective_drive", np.nan),
        ("requested_contrast", 99.0),
        ("reference_code_luminance", 0.1),
        ("realized_code_luminance", 0.1),
        ("realized_valid", 0),
        ("sample_scope_visible", 0),
        ("clipped", 1),
    ],
)
def test_numerical_witness_refusals_without_outer_digest_shortcut(
    tmp_path, field, value
):
    path = emit_fixture(tmp_path)
    with h5py.File(path, "r+") as source:
        rows = source["/components/visual_appearance/states"][:]
        rows[field][0] = value
        source["/components/visual_appearance/states"][:] = rows
    with h5py.File(path, "r") as source, pytest.raises(UnifiedH5ContractError):
        validate_appearance_witness(source)
