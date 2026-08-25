"""Small, closed witness for a source-recording import publication."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping
import unicodedata

from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SourceRecordingIdentityClaim,
    SourceRecordingIdentityError,
    load_strict_json_object,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)


RECEIPT_SCHEMA_ID = "palette.recording_import_receipt.v2"
RECEIPT_DIGEST_FIELD = "receipt_sha256"
# The v2 witness is issued only by the current importer. Keep this in the
# receipt contract so every reader applies the same admission rule.
CURRENT_RECORDING_IMPORT_PRODUCER_ID = (
    "fisheye.utils.import_recording_analysis.process_recording_import"
)
MAX_RECEIPT_BYTES = 256 * 1024
RECEIPT_RELATIVE_DIRECTORY = ".imports"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_KEYS = {
    "schema_id", "canonicalization", "identity_profile", "producer",
    "target_relative_path", "identity_claim",
    "acquisition_ownership", "acquisition_frame", RECEIPT_DIGEST_FIELD,
}
_FIELDS = (
    "identity_profile", "producer_id", "producer_git_sha", "producer_git_dirty",
    "config_sha256", "target_relative_path", "identity_claim",
    "acquisition_ownership_ref", "acquisition_ownership_sha256", "acquisition_frame_ref",
    "acquisition_frame_sha256",
)


class RecordingImportReceiptError(ValueError):
    """A receipt is malformed or inconsistent."""


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def _text(value: Any, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RecordingImportReceiptError(f"{field} must be non-empty text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RecordingImportReceiptError(f"{field} must be valid Unicode") from exc
    if any(unicodedata.category(c).startswith("C") for c in value):
        raise RecordingImportReceiptError(f"{field} must not contain control characters")
    return value


def _hex(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    value = _text(value, field)
    if pattern.fullmatch(value) is None:
        raise RecordingImportReceiptError(f"{field} has the wrong hexadecimal shape")
    return value


def _path(value: Any) -> str:
    value = _text(value, "target_relative_path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value == "."
        or path.as_posix() != value
        or ".." in path.parts
    ):
        raise RecordingImportReceiptError(
            "target_relative_path must be a normalized relative POSIX path"
        )
    return value


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise RecordingImportReceiptError(f"{field} has an unsupported shape")
    return value


def _identity_claim(value: Any) -> SourceRecordingIdentityClaim:
    try:
        return SourceRecordingIdentityClaim.from_mapping(value)
    except (SourceRecordingIdentityError, TypeError, ValueError) as exc:
        raise RecordingImportReceiptError(
            "identity_claim failed the source-recording contract"
        ) from exc


def _payload(**v: Any) -> dict[str, Any]:
    return {
        "schema_id": RECEIPT_SCHEMA_ID, "canonicalization": CANONICAL_JSON_DIGEST_ALGORITHM,
        "identity_profile": v["identity_profile"],
        "producer": {
            "id": v["producer_id"], "git_sha": v["producer_git_sha"],
            "git_dirty": v["producer_git_dirty"], "config_sha256": v["config_sha256"],
        },
        "target_relative_path": v["target_relative_path"],
        "identity_claim": v["identity_claim"].as_dict(),
        "acquisition_ownership": {"ref": v["acquisition_ownership_ref"],
                                   "sha256": v["acquisition_ownership_sha256"]},
        "acquisition_frame": {"ref": v["acquisition_frame_ref"],
                               "sha256": v["acquisition_frame_sha256"]},
    }


@dataclass(frozen=True, slots=True)
class RecordingImportReceipt:
    """The bounded v2 receipt; acquisition records remain referenced, not copied."""

    identity_profile: str
    producer_id: str
    producer_git_sha: str
    producer_git_dirty: bool
    config_sha256: str
    target_relative_path: str
    identity_claim: SourceRecordingIdentityClaim
    acquisition_ownership_ref: str
    acquisition_ownership_sha256: str
    acquisition_frame_ref: str
    acquisition_frame_sha256: str
    receipt_sha256: str

    def _payload(self) -> dict[str, Any]:
        return _payload(**{field: getattr(self, field) for field in _FIELDS})

    def __post_init__(self) -> None:
        if self.identity_profile != SOURCE_RECORDING_IDENTITY_PROFILE:
            raise RecordingImportReceiptError("identity_profile is unsupported")
        if self.producer_id != CURRENT_RECORDING_IMPORT_PRODUCER_ID:
            raise RecordingImportReceiptError(
                "producer.id is not the current recording importer"
            )
        _hex(self.producer_git_sha, "producer.git_sha", _GIT_SHA)
        if self.producer_git_dirty is not False or type(self.producer_git_dirty) is not bool:
            raise RecordingImportReceiptError("producer.git_dirty must be literal false")
        _hex(self.config_sha256, "producer.config_sha256", _SHA256)
        _path(self.target_relative_path)
        if type(self.identity_claim) is not SourceRecordingIdentityClaim:
            raise RecordingImportReceiptError("identity_claim has an invalid type")
        _identity_claim(self.identity_claim.as_dict())
        _text(self.acquisition_ownership_ref, "acquisition_ownership.ref")
        _hex(self.acquisition_ownership_sha256, "acquisition_ownership.sha256", _SHA256)
        _text(self.acquisition_frame_ref, "acquisition_frame.ref")
        _hex(self.acquisition_frame_sha256, "acquisition_frame.sha256", _SHA256)
        if self.receipt_sha256 != canonical_json_sha256(self._payload()):
            raise RecordingImportReceiptError("receipt_sha256 does not match its payload")

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), RECEIPT_DIGEST_FIELD: self.receipt_sha256}

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @classmethod
    def create(cls, **kwargs: Any) -> "RecordingImportReceipt":
        kwargs = {
            "identity_profile": SOURCE_RECORDING_IDENTITY_PROFILE,
            "producer_git_dirty": False,
            **kwargs,
        }
        kwargs["identity_claim"] = _identity_claim(kwargs["identity_claim"].as_dict())
        return cls(**kwargs, receipt_sha256=canonical_json_sha256(_payload(**kwargs)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RecordingImportReceipt":
        raw = _exact(value, _KEYS, "receipt")
        if raw["schema_id"] != RECEIPT_SCHEMA_ID:
            raise RecordingImportReceiptError("receipt schema is unsupported")
        if raw["canonicalization"] != CANONICAL_JSON_DIGEST_ALGORITHM:
            raise RecordingImportReceiptError("receipt canonicalization is unsupported")
        producer = _exact(raw["producer"], {"id", "git_sha", "git_dirty", "config_sha256"}, "producer")
        ownership = _exact(raw["acquisition_ownership"], {"ref", "sha256"}, "acquisition_ownership")
        frame = _exact(raw["acquisition_frame"], {"ref", "sha256"}, "acquisition_frame")
        receipt = cls(
            identity_profile=raw["identity_profile"],
            producer_id=producer["id"],
            producer_git_sha=producer["git_sha"],
            producer_git_dirty=producer["git_dirty"],
            config_sha256=producer["config_sha256"],
            target_relative_path=raw["target_relative_path"],
            identity_claim=_identity_claim(raw["identity_claim"]),
            acquisition_ownership_ref=ownership["ref"],
            acquisition_ownership_sha256=ownership["sha256"],
            acquisition_frame_ref=frame["ref"],
            acquisition_frame_sha256=frame["sha256"],
            receipt_sha256=raw[RECEIPT_DIGEST_FIELD],
        )
        if receipt.as_dict() != raw:
            raise RecordingImportReceiptError("receipt is not canonical")
        return receipt

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "RecordingImportReceipt":
        if type(payload) is not bytes or len(payload) > MAX_RECEIPT_BYTES:
            raise RecordingImportReceiptError(f"receipt exceeds {MAX_RECEIPT_BYTES} bytes")
        try:
            value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_finite)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise RecordingImportReceiptError("receipt JSON is not strict JSON") from exc
        return cls.from_mapping(value)

    @classmethod
    def from_path(cls, path: Path) -> "RecordingImportReceipt":
        source = Path(path)
        if source.is_symlink():
            raise RecordingImportReceiptError(
                "receipt path must not be a symbolic link"
            )
        try:
            value = load_strict_json_object(source, max_bytes=MAX_RECEIPT_BYTES)
        except SourceRecordingIdentityError as exc:
            raise RecordingImportReceiptError(str(exc)) from exc
        return cls.from_mapping(value)


def recording_import_receipt_path(
    zarr_path: Path,
    receipt_sha256: str,
) -> Path:
    digest = _hex(receipt_sha256, RECEIPT_DIGEST_FIELD, _SHA256)
    return Path(zarr_path) / RECEIPT_RELATIVE_DIRECTORY / f"{digest}.json"


def recording_import_receipt_paths(zarr_path: Path) -> tuple[Path, ...]:
    """Return validated receipt sidecars without following redirected storage."""

    directory = Path(zarr_path) / RECEIPT_RELATIVE_DIRECTORY
    if directory.is_symlink():
        raise RecordingImportReceiptError(
            "recording import receipt directory must not be a symbolic link"
        )
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise RecordingImportReceiptError(
            "recording import receipt location is not a directory"
        )
    paths: list[Path] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.name.endswith(".json")
            or _SHA256.fullmatch(path.name[:-5]) is None
        ):
            raise RecordingImportReceiptError(
                "recording import receipt directory contains an unsupported entry"
            )
        RecordingImportReceipt.from_path(path)
        paths.append(path)
    return tuple(paths)


def publish_recording_import_receipt(
    zarr_path: Path,
    receipt: RecordingImportReceipt,
) -> Path:
    """Atomically install an immutable digest-named receipt or replay it."""

    if type(receipt) is not RecordingImportReceipt:
        raise RecordingImportReceiptError("receipt has an invalid type")
    target = recording_import_receipt_path(zarr_path, receipt.receipt_sha256)
    if target.parent.is_symlink():
        raise RecordingImportReceiptError(
            "recording import receipt directory must not be a symbolic link"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RecordingImportReceiptError(
            "recording import receipt directory is not a real directory"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=".recording-import-receipt.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(receipt.to_json_bytes())
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if RecordingImportReceipt.from_path(target) != receipt:
                raise RecordingImportReceiptError(
                    "existing receipt digest path has different content"
                )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target

__all__ = [
    "CURRENT_RECORDING_IMPORT_PRODUCER_ID",
    "MAX_RECEIPT_BYTES", "RECEIPT_DIGEST_FIELD", "RECEIPT_RELATIVE_DIRECTORY",
    "RECEIPT_SCHEMA_ID",
    "RecordingImportReceipt", "RecordingImportReceiptError",
    "publish_recording_import_receipt", "recording_import_receipt_path",
    "recording_import_receipt_paths",
]
