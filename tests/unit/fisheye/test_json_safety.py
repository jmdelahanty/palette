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
    write_json_atomic,
    write_jsonl_atomic,
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


def test_write_json_atomic_writes_strict_json(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "payload.json"

    write_json_atomic(out, {"b": np.asarray([1, 2]), "a": math.nan})

    assert json.loads(out.read_text(encoding="utf-8")) == {"a": None, "b": [1, 2]}


def test_write_json_atomic_failed_replace_leaves_prior_file(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    out = tmp_path / "payload.json"
    out.write_text('{"old": true}\n', encoding="utf-8")

    def _raise_replace(_src: object, _dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("fisheye.shared.json_safety.os.replace", _raise_replace)

    try:
        write_json_atomic(out, {"old": False})
    except OSError:
        pass
    else:  # pragma: no cover - defensive guard
        raise AssertionError("write_json_atomic did not raise")

    assert out.read_text(encoding="utf-8") == '{"old": true}\n'


def test_write_json_atomic_can_refuse_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "payload.json"
    out.write_text('{"old": true}\n', encoding="utf-8")

    try:
        write_json_atomic(out, {"old": False}, overwrite=False)
    except FileExistsError:
        pass
    else:  # pragma: no cover - defensive guard
        raise AssertionError("write_json_atomic did not raise")

    assert out.read_text(encoding="utf-8") == '{"old": true}\n'


def test_write_jsonl_atomic_writes_one_strict_json_object_per_line(tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"

    write_jsonl_atomic(out, [{"row": 1, "value": np.float32("nan")}, {"row": 2}])

    assert out.read_text(encoding="utf-8").splitlines() == [
        '{"row": 1, "value": null}',
        '{"row": 2}',
    ]
