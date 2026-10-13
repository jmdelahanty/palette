"""Palette's canonical ONNX model manifest: one owner for building and checking it.

A promoted model package carries ``exports/onnx/<run_id>.canonical.manifest.json``
binding its weights, ONNX, the training export manifest and (pose only) the
pose model-input contract by package-relative path and SHA-256. Orange names
this file by path and digest in its TensorRT engine manifests and may validate
it against the published schemas here.

- v1: pose only; bare-hex digests and ``payload_digest``. Written before
  2026-10-12 and still valid.
- v2: pose and detect; adopts agent-contracts ``orange-palette/digest_form_v1``
  (every ``sha256`` / ``*_sha256`` field is a ``sha256:`` token; the payload's
  digest is ``payload_sha256`` over ``canonical_json_utf8_sort_keys_compact_v1``).
  New promotions write v2.

The schemas are ``contracts/palette_canonical_onnx_model_manifest_v{1,2}.schema.json``;
Palette is their producer, and their SHA-256 is pinned below.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.resources import files as resource_files
import json
from typing import Any, Mapping

from fisheye.shared.zarr.manifest_digest import canonical_json_bytes

SCHEMA_ID = "palette.canonical_onnx_model_manifest"
CURRENT_VERSION = 2
CANONICALIZATION = "canonical_json_utf8_sort_keys_compact_v1"
TASKS = ("pose", "detect")
SCHEMA_FILES = {
    1: (
        "palette_canonical_onnx_model_manifest_v1.schema.json",
        "6b218c139a5f78fd1f0a751dca03e49677d7170ee54c08a6881c0d8fa03b94d8",
    ),
    2: (
        "palette_canonical_onnx_model_manifest_v2.schema.json",
        "50916e1f5e7e626a55d79b33ef79e39a6834f04274e62a8cd060b868286d657d",
    ),
}
TOKEN_PREFIX = "sha256:"


class CanonicalOnnxManifestError(ValueError):
    """A canonical ONNX model manifest does not satisfy its schema or digest."""


def sha256_token(value: str) -> str:
    """A ``sha256:`` token for a bare lowercase hex digest (or an existing token)."""

    text = str(value)
    hex_part = text[len(TOKEN_PREFIX):] if text.startswith(TOKEN_PREFIX) else text
    if len(hex_part) != 64 or any(ch not in "0123456789abcdef" for ch in hex_part):
        raise CanonicalOnnxManifestError(f"not a SHA-256 digest: {value!r}")
    return TOKEN_PREFIX + hex_part


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_token(hashlib.sha256(canonical_json_bytes(payload)).hexdigest())


@lru_cache(maxsize=None)
def _validator(version: int):
    from jsonschema import Draft202012Validator

    file_name, digest = SCHEMA_FILES[version]
    data = resource_files("fisheye.shared").joinpath("contracts").joinpath(file_name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise CanonicalOnnxManifestError("packaged_contract_drift:" + file_name)
    return Draft202012Validator(json.loads(data))


def _artifact(relative_path: str, sha256: str) -> dict[str, str]:
    return {"relative_path": str(relative_path), "sha256": sha256_token(sha256)}


def build_canonical_onnx_manifest(
    *,
    task: str,
    run_id: str,
    set_id: str,
    weights: tuple[str, str],
    onnx: tuple[str, str],
    source_export_manifest: tuple[str, str],
    pose_model_input_contract: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    """A v2 manifest. Each artifact is ``(package-relative path, sha256)``; the
    pose contract is ``(relative path, file sha256, payload digest)``."""

    if task not in TASKS:
        raise CanonicalOnnxManifestError(f"unsupported task {task!r}")
    if (task == "pose") != (pose_model_input_contract is not None):
        raise CanonicalOnnxManifestError(
            "a pose manifest needs the pose model-input contract; a detect manifest has none"
        )
    payload: dict[str, Any] = {
        "status": "complete",
        "run_id": run_id,
        "set_id": set_id,
        "task": task,
        "weights": _artifact(*weights),
        "onnx": _artifact(*onnx),
        "source_export_manifest": _artifact(*source_export_manifest),
    }
    if pose_model_input_contract is not None:
        path, file_sha256, contract_payload = pose_model_input_contract
        payload["pose_model_input_contract"] = {
            "relative_path": str(path),
            "sha256": sha256_token(file_sha256),
            "payload_sha256": sha256_token(contract_payload),
        }
    document = {
        "schema_id": SCHEMA_ID,
        "schema_version": CURRENT_VERSION,
        "canonicalization": CANONICALIZATION,
        "payload_sha256": payload_sha256(payload),
        "payload": payload,
    }
    validate_canonical_onnx_manifest(document)
    return document


def validate_canonical_onnx_manifest(document: Any) -> dict[str, Any]:
    """Refuse a manifest that breaks its version's schema or payload digest."""

    from jsonschema.exceptions import best_match

    version = document.get("schema_version") if isinstance(document, Mapping) else None
    if (
        not isinstance(document, Mapping)
        or document.get("schema_id") != SCHEMA_ID
        or type(version) is not int
        or version not in SCHEMA_FILES
    ):
        raise CanonicalOnnxManifestError("not a supported palette.canonical_onnx_model_manifest")
    error = best_match(_validator(version).iter_errors(document))
    if error is not None:
        raise CanonicalOnnxManifestError(f"canonical ONNX manifest v{version}: {error.message}")
    if version == 1:
        expected = hashlib.sha256(canonical_json_bytes(document["payload"])).hexdigest()
        observed = document["payload_digest"]
    else:
        expected, observed = payload_sha256(document["payload"]), document["payload_sha256"]
    if observed != expected:
        raise CanonicalOnnxManifestError("canonical ONNX manifest payload digest mismatch")
    return dict(document)


__all__ = [
    "CANONICALIZATION",
    "CURRENT_VERSION",
    "CanonicalOnnxManifestError",
    "SCHEMA_FILES",
    "SCHEMA_ID",
    "TASKS",
    "build_canonical_onnx_manifest",
    "payload_sha256",
    "sha256_token",
    "validate_canonical_onnx_manifest",
]
