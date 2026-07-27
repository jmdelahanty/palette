"""Strict canonical-JSON helpers shared by persisted Zarr manifests."""

from __future__ import annotations

import hashlib
import json


CANONICAL_JSON_DIGEST_ALGORITHM = "sha256_canonical_json_v1"


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic strict UTF-8 JSON suitable for contract digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CANONICAL_JSON_DIGEST_ALGORITHM",
    "canonical_json_bytes",
    "canonical_json_sha256",
]
