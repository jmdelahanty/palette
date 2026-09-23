"""Bounded reads and semantic refusal below the outer file-byte seal."""

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.appearance import validate_appearance_witness
from fisheye.shared.unified_h5.common import KeyIndex, parse_json, same_json, uint64
from fisheye.shared.unified_h5.hdf5_types import (
    check_dataset_budget,
    dtype_from_descriptor,
    iter_payload,
    read_bounded_string,
    type_descriptor,
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


def test_lazy_oversized_dataset_rejected_before_read(tmp_path):
    with h5py.File(tmp_path / "oversized.h5", "w") as source:
        dataset = source.create_dataset(
            "oversized", shape=(2_000_001,), dtype="u8", chunks=(1024,)
        )
        with pytest.raises(UnifiedH5ContractError, match="budget"):
            check_dataset_budget(dataset)
        byte_budget = source.create_dataset(
            "byte_budget", shape=(1_000_000,), dtype="S128", chunks=(1024,)
        )
        with pytest.raises(UnifiedH5ContractError, match="byte_budget"):
            check_dataset_budget(byte_budget)
        big_endian = source.create_dataset("big_endian", shape=(1,), dtype=">u8")
        with pytest.raises(UnifiedH5ContractError, match="integer_type"):
            check_dataset_budget(big_endian)


def test_all_real_native_type_descriptors_round_trip(tmp_path):
    with h5py.File(emit_fixture(tmp_path), "r") as source:

        def check(name, node):
            if isinstance(node, h5py.Dataset):
                assert (
                    dtype_from_descriptor(type_descriptor(node.id.get_type()))
                    == node.dtype
                )

        source.visititems(check)


def test_hostile_persisted_descriptor_budget():
    spec = {
        "class": "string",
        "variable_length": False,
        "character_set": "utf8",
        "size_bytes": 2**62,
        "padding": "null_terminated",
    }
    with pytest.raises(UnifiedH5ContractError, match="string_type"):
        dtype_from_descriptor(spec)
    array = {
        "class": "array",
        "dimensions": [2**62],
        "size_bytes": 2**62,
        "base": {
            "class": "integer",
            "size_bytes": 1,
            "byte_order": "none",
            "signed": False,
        },
    }
    with pytest.raises(UnifiedH5ContractError, match="budget"):
        dtype_from_descriptor(array)


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
