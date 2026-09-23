"""Decode pinned producer artifacts only into each test's disposable directory."""

from __future__ import annotations

import base64
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

FIXTURES = Path(__file__).parents[2] / "fixtures" / "unified_h5_v1"


@lru_cache(maxsize=24)
def fixture_bytes(name: str) -> bytes:
    inventory = json.loads((FIXTURES / "inventory.json").read_text())["fixtures"]
    entry = inventory[name]
    encoded = (FIXTURES / f"{name}.h5.gz.b64").read_bytes()
    data = gzip.decompress(base64.b64decode(encoded))
    assert len(data) == entry["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    return data


def emit_fixture(directory: Path, name: str = "appearance") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.h5"
    path.write_bytes(fixture_bytes(name))
    return path


def receipt_for(name: str = "appearance") -> dict:
    return json.loads((FIXTURES / f"{name}.receipt.json").read_text())


def write_receipt(directory: Path, name: str = "appearance") -> Path:
    path = directory / f"{name}.receipt.json"
    path.write_bytes((FIXTURES / f"{name}.receipt.json").read_bytes())
    return path


def synthetic_receipt_for_mutated_test_file(
    path: Path, name: str = "appearance"
) -> dict:
    """Untrusted test reseal: expose internal failures behind the byte gate.

    Never changes the vendored fixture or represents a real producer receipt.
    """
    receipt = receipt_for(name)
    receipt["contract"]["h5_artifact"]["size_bytes"] = path.stat().st_size
    receipt["contract"]["h5_artifact"]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    payload = json.dumps(
        receipt["contract"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    receipt["contract_sha256"] = "sha256:" + digest
    receipt["receipt_id"] = "obsbindfin_" + digest
    return receipt
