"""Content fingerprints for provenance input artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid
import warnings


CONTENT_FINGERPRINT_SCHEME = "content_v1"
MANIFEST_FINGERPRINT_SCHEME = "manifest_v1"


def _normalize_sha256(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


def _stat_payload(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _registry_stat_matches(current: Mapping[str, int], registry_stat: Mapping[str, Any] | None) -> bool:
    if not isinstance(registry_stat, Mapping):
        return False
    try:
        return (
            int(registry_stat["size_bytes"]) == int(current["size_bytes"])
            and int(registry_stat["mtime_ns"]) == int(current["mtime_ns"])
        )
    except Exception:
        return False


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{CONTENT_FINGERPRINT_SCHEME}.json")


def _read_sidecar(path: Path, current: Mapping[str, int]) -> str | None:
    sidecar = _sidecar_path(path)
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("fingerprint_scheme") not in (None, CONTENT_FINGERPRINT_SCHEME):
        return None
    try:
        if int(payload["size_bytes"]) != int(current["size_bytes"]):
            return None
        if int(payload["mtime_ns"]) != int(current["mtime_ns"]):
            return None
    except Exception:
        return None
    return _normalize_sha256(payload.get("sha256"))


def _write_sidecar(path: Path, *, sha256: str, stat_payload: Mapping[str, int]) -> None:
    sidecar = _sidecar_path(path)
    payload = {
        "fingerprint_scheme": CONTENT_FINGERPRINT_SCHEME,
        "sha256": sha256,
        "size_bytes": int(stat_payload["size_bytes"]),
        "mtime_ns": int(stat_payload["mtime_ns"]),
    }
    tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(tmp, sidecar)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_payload(path: Path, *, role: str, error: str) -> dict[str, Any]:
    warnings.warn(f"Could not fingerprint {role} artifact {path}: {error}", RuntimeWarning, stacklevel=2)
    return {
        "role": str(role),
        "path": str(path),
        "fingerprint_scheme": None,
        "error": str(error),
    }


def fingerprint_artifact(
    path: str | Path,
    *,
    role: str,
    registry_hash: str | None = None,
    registry_stat: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a provenance fingerprint for a file artifact.

    The result is correctness-first: registry or sidecar values are used only
    when their stat metadata exactly matches the file currently on disk.
    Missing/unreadable artifacts return a recorded error payload instead of
    raising, because artifact hashing is non-gating provenance in this slice.
    """

    artifact_path = Path(path).expanduser()
    registry_sha256 = _normalize_sha256(registry_hash)
    try:
        resolved = artifact_path.resolve()
    except Exception:
        resolved = artifact_path

    try:
        current_stat = _stat_payload(resolved)
    except Exception as exc:
        return _error_payload(resolved, role=role, error=str(exc))
    if not resolved.is_file():
        return _error_payload(resolved, role=role, error="not a file")

    base = {
        "role": str(role),
        "path": str(resolved),
        "fingerprint_scheme": CONTENT_FINGERPRINT_SCHEME,
        **current_stat,
    }

    if registry_sha256 and _registry_stat_matches(current_stat, registry_stat):
        return {
            **base,
            "sha256": registry_sha256,
            "source": "registry",
        }

    sidecar_sha256 = _read_sidecar(resolved, current_stat)
    if sidecar_sha256 is not None:
        return {
            **base,
            "sha256": sidecar_sha256,
            "source": "sidecar",
        }

    try:
        actual_sha256 = _sha256_file(resolved)
    except Exception as exc:
        return _error_payload(resolved, role=role, error=str(exc))
    _write_sidecar(resolved, sha256=actual_sha256, stat_payload=current_stat)

    payload: dict[str, Any] = {
        **base,
        "sha256": actual_sha256,
        "source": "computed",
    }
    if registry_sha256 and actual_sha256 != registry_sha256:
        payload["registry_sha256"] = registry_sha256
        payload["mismatch"] = True
        warnings.warn(
            f"Computed sha256 for {role} artifact {resolved} does not match registry hash.",
            RuntimeWarning,
            stacklevel=2,
        )
    return payload


def require_artifact_content_identity(
    path: str | Path,
    *,
    role: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Directly hash an artifact and fail closed on missing or changed bytes.

    This is the scientific-commit boundary.  Unlike the best-effort provenance
    helper above, it deliberately ignores registry stat shortcuts and sidecar
    caches so a same-size/same-mtime replacement cannot be accepted.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required {role} artifact is not a file: {resolved}")
    expected = _normalize_sha256(expected_sha256)
    if expected_sha256 is not None and expected is None:
        raise ValueError(f"Expected SHA-256 for {role} is malformed.")
    stat_payload = _stat_payload(resolved)
    actual = _sha256_file(resolved)
    if expected is not None and actual != expected:
        raise ValueError(
            f"Required {role} artifact SHA-256 mismatch: expected {expected}, "
            f"observed {actual}."
        )
    return {
        "role": str(role),
        "path": str(resolved),
        "fingerprint_scheme": CONTENT_FINGERPRINT_SCHEME,
        "sha256": actual,
        "source": "direct_scientific_commit_rehash",
        **stat_payload,
    }


def fingerprint_directory_manifest(
    directory: str | Path,
    *,
    role: str,
    identity_kind: str = "runtime_manifest",
) -> dict[str, Any]:
    """Return a manifest fingerprint over file paths, sizes, and mtimes.

    This intentionally does not hash file contents. It is for large runtime
    directories where the exact weight files are not determinable without
    reaching into an external project's internals.
    """

    directory_path = Path(directory).expanduser()
    try:
        resolved = directory_path.resolve()
    except Exception:
        resolved = directory_path
    if not resolved.exists() or not resolved.is_dir():
        return _error_payload(resolved, role=role, error="not a directory")

    entries: list[dict[str, Any]] = []
    total_size = 0
    try:
        files = sorted(path for path in resolved.rglob("*") if path.is_file())
        for item in files:
            stat = item.stat()
            size = int(stat.st_size)
            total_size += size
            entries.append(
                {
                    "path": item.relative_to(resolved).as_posix(),
                    "size_bytes": size,
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    except Exception as exc:
        return _error_payload(resolved, role=role, error=str(exc))

    manifest_text = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "role": str(role),
        "path": str(resolved),
        "fingerprint_scheme": MANIFEST_FINGERPRINT_SCHEME,
        "sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "source": "computed",
        "identity_kind": str(identity_kind),
        "manifest_entry_count": int(len(entries)),
        "size_bytes": int(total_size),
    }
