from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from fisheye.shared.json_safety import (
    decode_fixed_width_bytes,
    decode_null_terminated_text,
    json_attr_safe,
    json_attr_safe_mapping,
    strict_json_dumps,
)


def test_decode_fixed_width_bytes_strips_null_padding() -> None:
    assert decode_fixed_width_bytes(b"reason\x00\x00") == "reason"


def test_decode_null_terminated_text_stops_at_first_null() -> None:
    assert decode_null_terminated_text(b"reason\x00garbage") == "reason"
    assert decode_null_terminated_text("ok\x00tail") == "ok"
    assert decode_null_terminated_text(np.asarray([111, 107, 0, 120], dtype=np.uint8)) == "ok"


def test_json_attr_safe_converts_numpy_bytes_paths_and_nonfinite_values() -> None:
    payload = {
        b"byte-key": np.asarray([1.0, np.nan, np.inf], dtype=np.float32),
        "scalar": np.float64("-inf"),
        "bytes": b"ok\x00\x00",
        "path": Path("/tmp/example"),
        "nested": (np.int64(3), np.bool_(True)),
    }

    safe = json_attr_safe(payload)

    assert safe == {
        "b'byte-key'": [1.0, None, None],
        "scalar": None,
        "bytes": "ok",
        "path": "/tmp/example",
        "nested": [3, True],
    }
    json.dumps(safe, allow_nan=False)


def test_json_attr_safe_mapping_stringifies_keys() -> None:
    assert json_attr_safe_mapping({1: math.nan}) == {"1": None}


def test_strict_json_dumps_rejects_nan_after_sanitizing_to_null() -> None:
    assert strict_json_dumps({"x": math.nan, "y": np.asarray([1, 2])}) == '{"x":null,"y":[1,2]}'
