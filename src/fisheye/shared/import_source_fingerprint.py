"""Cheap stat-based fingerprints for singleton import source files."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


IMPORT_SOURCE_STAT_FINGERPRINT_STRATEGY = "stat_v1"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def source_stat_fingerprint_attrs(
    path: str | Path,
    *,
    attr_prefix: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return import-source fingerprint attrs from path, size, mtime, and metadata.

    This intentionally does not hash full MP4/H5 contents. The goal is a cheap
    import-time identity marker that catches swapped/re-encoded source files
    without making ingest pay an O(file-size) checksum cost.
    """

    source_path = Path(path).expanduser()
    try:
        resolved = source_path.resolve()
    except Exception:
        resolved = source_path
    stat = resolved.stat()
    components: dict[str, Any] = {
        "strategy": IMPORT_SOURCE_STAT_FINGERPRINT_STRATEGY,
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if extra:
        components.update({str(key): _json_safe(value) for key, value in extra.items()})
    digest_payload = json.dumps(_json_safe(components), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return {
        f"{attr_prefix}_fingerprint": digest,
        f"{attr_prefix}_fingerprint_strategy": IMPORT_SOURCE_STAT_FINGERPRINT_STRATEGY,
        f"{attr_prefix}_fingerprint_payload": components,
        f"{attr_prefix}_size_bytes": int(stat.st_size),
        f"{attr_prefix}_mtime_ns": int(stat.st_mtime_ns),
    }


def optional_source_stat_fingerprint_attrs(
    path: str | Path | None,
    *,
    attr_prefix: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Best-effort stat fingerprint attrs with an explicit error on failure."""

    if path in (None, ""):
        return {}
    try:
        return source_stat_fingerprint_attrs(path, attr_prefix=attr_prefix, extra=extra)
    except OSError as exc:
        return {
            f"{attr_prefix}_fingerprint_error": str(exc),
            f"{attr_prefix}_fingerprint_strategy": IMPORT_SOURCE_STAT_FINGERPRINT_STRATEGY,
        }
