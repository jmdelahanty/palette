"""Closed v1 grammar and bounded scratch indexing for Citrus unified H5.

These are profile-specific admission mechanics, not scientific acceptance.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from hashlib import sha256
from functools import wraps
import json
import math
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from typing import Any, Mapping

PROFILE = "unified_experimental_h5_v1"
MAX_DATASET_BYTES = 64 * 1024 * 1024
MAX_ROWS = 2_000_000
MAX_JSON_BYTES = 8 * 1024 * 1024
BLOCK_BYTES = 1024 * 1024
MAX_OBJECTS = 100_000
MAX_DEPTH = 32
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class UnifiedH5ContractError(ValueError):
    """The requested native profile is invalid, unsupported, or incomplete."""


def contract_errors(function):
    """Keep malformed native records inside the explicit profile's error API."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except UnifiedH5ContractError:
            raise
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            OSError,
            AttributeError,
        ) as exc:
            raise UnifiedH5ContractError(f"unified_artifact_invalid:{exc}") from exc

    return wrapped


def require(condition: Any, reason: str) -> None:
    if not condition:
        raise UnifiedH5ContractError(reason)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def same_json(left: Any, right: Any) -> bool:
    """Closed receipt equality must not equate JSON true, 1 and 1.0."""
    return canonical_json(left) == canonical_json(right)


def parse_json(data: bytes | str, *, label: str, canonical: bool = False) -> dict:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    require(len(raw) <= MAX_JSON_BYTES, f"json_budget_exceeded:{label}")

    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f"duplicate_json_key:{label}:{key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise UnifiedH5ContractError(f"nonfinite_json:{label}:{value}")

    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), f"nonfinite_json:{label}:{value}")
        return parsed

    try:
        result = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=bad_constant,
            parse_float=finite_float,
        )
    except UnifiedH5ContractError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise UnifiedH5ContractError(f"malformed_json:{label}") from exc
    require(type(result) is dict, f"json_object_required:{label}")
    if canonical:
        require(raw == canonical_json(result), f"noncanonical_json:{label}")
    return result


def exact_keys(value: Mapping, expected, label: str) -> None:
    require(
        type(value) is dict and set(value) == set(expected),
        f"closed_fields_mismatch:{label}",
    )


def uint64(value: Any, label: str) -> int:
    require(type(value) is int and 0 <= value < 2**64, f"invalid_uint64:{label}")
    return value


def text(value: Any, label: str, *, empty: bool = False) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeError as exc:
            raise UnifiedH5ContractError(f"invalid_utf8:{label}") from exc
    require(isinstance(value, str) and (empty or bool(value)), f"invalid_text:{label}")
    return value


def internal_path(value: Any) -> str:
    require(
        isinstance(value, str) and value.startswith("/") and len(value) > 1,
        "invalid_internal_path",
    )
    require(
        all(part not in ("", ".", "..") for part in value[1:].split("/"))
        and not any(c in value for c in "\\\x00\n\r"),
        "invalid_internal_path",
    )
    return value


class KeyIndex(AbstractContextManager):
    """Disk-backed exact-key index; uint64 keys never pass through SQLite REAL.

    SQLite is scratch validation state, not the Palette registry. The connection
    uses the calling Palette Python's SQLite runtime and a bounded page cache.
    """

    def __init__(self):
        self._directory = TemporaryDirectory(prefix="palette-unified-h5-keys-")
        self._connection = sqlite3.connect(
            str(Path(self._directory.name) / "keys.sqlite")
        )
        self._connection.execute("PRAGMA cache_size=-4096")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute(
            "CREATE TABLE keys (namespace TEXT, key BLOB, row_index INTEGER, PRIMARY KEY(namespace,key)) WITHOUT ROWID"
        )

    @staticmethod
    def _key(key) -> bytes:
        return canonical_json(list(key))

    def add(self, namespace: str, key, row_index: int, *, reason: str) -> None:
        try:
            self._connection.execute(
                "INSERT INTO keys VALUES (?,?,?)",
                (namespace, self._key(key), row_index),
            )
        except sqlite3.IntegrityError as exc:
            raise UnifiedH5ContractError(reason) from exc

    def lookup(self, namespace: str, key) -> int | None:
        row = self._connection.execute(
            "SELECT row_index FROM keys WHERE namespace=? AND key=?",
            (namespace, self._key(key)),
        ).fetchone()
        return None if row is None else int(row[0])

    def __exit__(self, *args):
        self._connection.close()
        self._directory.cleanup()
