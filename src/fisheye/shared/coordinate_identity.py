"""Canonical, digest-bound row identity for coordinate-bearing row sets.

Coordinate semantics and row identity are related but independent contracts.
An ``instance_key`` identifies one acquired observation; it is not a universal
identifier for track samples, stimulus states, calibrations, or subjects.  This
module therefore exposes three strict future-write profiles:

``observation_instance``
    One persisted ``instance_key`` array with unique unsigned 64-bit values.

``track_sample``
    One persisted ``track_sample_key`` array whose two signed 64-bit columns
    are ``track_id`` and ``acquisition_frame_index``.  Stimulus, model-input,
    and run-local row numbers are never substitutes for acquisition time.
    Version 1 is an exact subset/reorder of one sealed immediate source rowset;
    a nullable, mechanically derived source-instance mapping is lineage, not
    primary identity.  Track-stage interpolation is deliberately unsupported.

``stimulus_state``
    One persisted signed 64-bit ``stimulus_state_key`` array.  It may be a
    scalar key or a small composite key with explicitly named components.

The row-set owner stores one compact ``row_identity_contract`` attribute and
its digest.  The key array stores a self-describing key record plus the digest
of the owning contract.  Coordinate descriptors can consequently reference a
single external record rather than repeating key-array details on every
coordinate surface.

This is a canonical future-write API.  Historical positional rows and legacy
``coordinate_row_identity`` arrays belong in audit/migration tooling and are
deliberately not accepted here.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    archive_identity,
    require_same_archive,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    verify_bound_coordinate_record,
)
from fisheye.shared.proof_verification import verify_persisted_proof


ROW_IDENTITY_CONTRACT_ATTR = "row_identity_contract"
ROW_IDENTITY_CONTRACT_DIGEST_ATTR = "row_identity_contract_sha256"
ROW_IDENTITY_CONTRACT_SCHEMA_ID = "palette.row_identity_contract"
ROW_IDENTITY_CONTRACT_SCHEMA_VERSION = 1
ROW_IDENTITY_CONTRACT_CANONICALIZATION = "canonical_json_sort_keys_v1"

ROW_IDENTITY_KEY_ATTR = "row_identity_key"
ROW_IDENTITY_KEY_DIGEST_ATTR = "row_identity_key_sha256"
ROW_IDENTITY_KEY_SCHEMA_ID = "palette.row_identity_key_array"
ROW_IDENTITY_KEY_SCHEMA_VERSION = 1
ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION = "numpy_dtype_shape_c_order_bytes_v1"

_ACTIVE_ROW_AUTHORITY_ARRAY_DIGEST_EVIDENCE: ContextVar[
    tuple[ArchiveIdentity, Mapping[str, Mapping[str, Any]]] | None
] = ContextVar("palette_row_authority_array_digest_evidence", default=None)
ROW_IDENTITY_CONTRACT_REF_ATTR = "row_identity_contract_ref"
ROW_IDENTITY_CONTRACT_SHA256_ATTR = "row_identity_contract_sha256"
ROW_IDENTITY_LOCAL_CONTRACT_REF = "@row_identity_contract"
MANIFEST_ROW_IDENTITY_ATTR = "run_manifest"
TRACK_SAMPLE_TIME_LINEAGE_ATTR = "track_sample_time_lineage"
TRACK_SAMPLE_TIME_LINEAGE_DIGEST_ATTR = "track_sample_time_lineage_sha256"
TRACK_SAMPLE_TIME_LINEAGE_SCHEMA_ID = "palette.track_sample_time_lineage"
TRACK_SAMPLE_TIME_LINEAGE_SCHEMA_VERSION = 1
SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR = "source_row_temporal_authority"
SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR = (
    "source_row_temporal_authority_sha256"
)
SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_ID = (
    "palette.source_row_temporal_authority"
)
SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_VERSION = 1
SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF = "source_acquisition_frame_index"
TRACK_SAMPLE_SOURCE_ROW_INDEX_REF = "source_row_index"
TRACK_SAMPLE_SOURCE_FRAME_INDEX_REF = "source_acquisition_frame_index"
TRACK_SAMPLE_INTERPOLATION_REF = "source_frame_interpolation"
TRACK_SAMPLE_SOURCE_INSTANCE_KEY_REF = "source_instance_key"
TRACK_SAMPLE_INTERPOLATION_DTYPE = np.dtype(
    [
        ("left_source_frame_index", "<i8"),
        ("right_source_frame_index", "<i8"),
        ("right_weight", "<f8"),
    ]
)
TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE = np.dtype(
    [("valid", "?"), ("instance_key", "<u8")]
)

OBSERVATION_INSTANCE_DOMAIN = "observation_instance"
TRACK_SAMPLE_DOMAIN = "track_sample"
STIMULUS_STATE_DOMAIN = "stimulus_state"
ROW_IDENTITY_DOMAINS = frozenset(
    {
        OBSERVATION_INSTANCE_DOMAIN,
        TRACK_SAMPLE_DOMAIN,
        STIMULUS_STATE_DOMAIN,
    }
)

INSTANCE_KEY_MODE = "instance_key"
TRACK_SAMPLE_KEY_MODE = "track_sample_key"
STIMULUS_STATE_KEY_MODE = "stimulus_state_key"
ROW_IDENTITY_MODES = frozenset(
    {INSTANCE_KEY_MODE, TRACK_SAMPLE_KEY_MODE, STIMULUS_STATE_KEY_MODE}
)

INSTANCE_KEY_ARRAY_REF = "instance_key"
TRACK_SAMPLE_KEY_ARRAY_REF = "track_sample_key"
STIMULUS_STATE_KEY_ARRAY_REF = "stimulus_state_key"

STIMULUS_STATE_KEY_COMPONENTS = frozenset(
    {
        "stimulus_state_index",
        "stimulus_frame_index",
        "stimulus_frame_num",
        "chaser_index",
        "trial_index",
        "presentation_index",
        "camera_frame_index",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KEY_REF_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_COMPONENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_NODE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_BOUND_ROW_IDENTITY_SEAL = object()
_BOUND_TRACK_TIME_LINEAGE_SEAL = object()
_BOUND_SOURCE_ROW_TEMPORAL_AUTHORITY_SEAL = object()


@dataclass(frozen=True)
class RowIdentityIssue:
    """One stable, machine-classifiable row-identity validation issue."""

    code: str
    path: str
    message: str


class RowIdentityContractError(ValueError):
    """Raised when a canonical row-identity contract is incomplete or invalid."""

    def __init__(self, issues: Sequence[RowIdentityIssue]):
        normalized = tuple(issues) or (
            RowIdentityIssue(
                code="row_identity_invalid",
                path="$",
                message="Row identity contract is invalid.",
            ),
        )
        self.issues = normalized
        super().__init__(
            "; ".join(
                f"{issue.code} at {issue.path}: {issue.message}"
                for issue in normalized
            )
        )


@dataclass(frozen=True)
class RowIdentityKeyArray:
    """Digest-bound description of the one canonical key array."""

    domain: str
    mode: str
    ref: str
    components: tuple[str, ...]
    dtype: str
    shape: tuple[int, ...]
    leading_dimension: int
    unique: bool
    content_sha256: str
    digest_canonicalization: str = ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": ROW_IDENTITY_KEY_SCHEMA_ID,
            "schema_version": ROW_IDENTITY_KEY_SCHEMA_VERSION,
            "domain": self.domain,
            "mode": self.mode,
            "ref": self.ref,
            "components": list(self.components),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "leading_dimension": self.leading_dimension,
            "unique": self.unique,
            "content_sha256": self.content_sha256,
            "digest_canonicalization": self.digest_canonicalization,
        }

    def digest(self) -> str:
        return _mapping_digest(self.to_dict())


@dataclass(frozen=True)
class TrackSampleTimeLineageRef:
    record_ref: str
    record_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class RowIdentityContract:
    """Canonical identity shared by all row-aligned surfaces in one row set."""

    domain: str
    mode: str
    leading_dimension: int
    unique: bool
    key_arrays: tuple[RowIdentityKeyArray, ...]
    time_lineage: TrackSampleTimeLineageRef | None = None

    @property
    def key_array(self) -> RowIdentityKeyArray:
        return self.key_arrays[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": ROW_IDENTITY_CONTRACT_SCHEMA_ID,
            "schema_version": ROW_IDENTITY_CONTRACT_SCHEMA_VERSION,
            "domain": self.domain,
            "mode": self.mode,
            "leading_dimension": self.leading_dimension,
            "unique": self.unique,
            "key_arrays": [item.to_dict() for item in self.key_arrays],
            "time_lineage": (
                self.time_lineage.to_dict()
                if self.time_lineage is not None
                else None
            ),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrackSampleTimeArrayRecord:
    ref: str
    dtype: str | tuple[tuple[str, str], ...]
    shape: tuple[int, ...]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "dtype": (
                [list(item) for item in self.dtype]
                if isinstance(self.dtype, tuple)
                else self.dtype
            ),
            "shape": list(self.shape),
            "content_sha256": self.content_sha256,
            "canonicalization": ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION,
        }


@dataclass(frozen=True)
class SourceRowTemporalAuthorityRecord:
    """Persisted authority for the acquisition time of one immediate rowset.

    The record deliberately binds an already-persisted row-identity contract
    and an explicit ``source_acquisition_frame_index`` array.  Row count,
    sibling ordering, and a similarly named frame array are never evidence.
    """

    acquisition_camera_frame: TrackSampleTimeLineageRef
    recording_id: str
    camera_id: str
    source_total_frames: int
    source_rowset_ref: str
    source_row_identity: TrackSampleTimeLineageRef
    source_identity_domain: str
    source_identity_mode: str
    source_leading_dimension: int
    source_acquisition_frame_index: TrackSampleTimeArrayRecord
    observation_instance_key: TrackSampleTimeArrayRecord | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_ID,
            "schema_version": SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_VERSION,
            "acquisition_camera_frame": self.acquisition_camera_frame.to_dict(),
            "recording_id": self.recording_id,
            "camera_id": self.camera_id,
            "source_total_frames": self.source_total_frames,
            "source_rowset_ref": self.source_rowset_ref,
            "source_row_identity": self.source_row_identity.to_dict(),
            "source_identity_domain": self.source_identity_domain,
            "source_identity_mode": self.source_identity_mode,
            "source_leading_dimension": self.source_leading_dimension,
            "source_acquisition_frame_index": (
                self.source_acquisition_frame_index.to_dict()
            ),
            "observation_instance_key": (
                self.observation_instance_key.to_dict()
                if self.observation_instance_key is not None
                else None
            ),
        }

    def digest(self) -> str:
        return _mapping_digest(self.to_dict())


@dataclass(frozen=True)
class BoundSourceRowTemporalAuthority:
    """Sealed, live binding of one immediate source rowset to acquisition time."""

    record: SourceRowTemporalAuthorityRecord
    record_ref: str
    record_sha256: str
    source_row_identity: "BoundRowIdentityContract"
    acquisition_frame: Any = field(repr=False, compare=False)
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _source_rowset_node: Any = field(repr=False, compare=False)
    _source_frame_index_node: Any = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)
    _manifest_authority: BoundCoordinateRecord | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _array_digest_evidence: Mapping[str, Mapping[str, Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    def assert_verified(self) -> None:
        require_bound_source_row_temporal_authority(self)


@dataclass(frozen=True)
class TrackSampleTimeLineageRecord:
    source_row_temporal_authority: TrackSampleTimeLineageRef
    recording_id: str
    camera_id: str
    source_total_frames: int
    source_rowset_ref: str
    source_identity_domain: str
    source_identity_mode: str
    source_leading_dimension: int
    leading_dimension: int
    track_sample_key_content_sha256: str
    source_row_index: TrackSampleTimeArrayRecord
    source_frame_index: TrackSampleTimeArrayRecord
    interpolation: TrackSampleTimeArrayRecord
    source_instance_key: TrackSampleTimeArrayRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": TRACK_SAMPLE_TIME_LINEAGE_SCHEMA_ID,
            "schema_version": TRACK_SAMPLE_TIME_LINEAGE_SCHEMA_VERSION,
            "source_row_temporal_authority": (
                self.source_row_temporal_authority.to_dict()
            ),
            "recording_id": self.recording_id,
            "camera_id": self.camera_id,
            "source_total_frames": self.source_total_frames,
            "source_rowset_ref": self.source_rowset_ref,
            "source_identity_domain": self.source_identity_domain,
            "source_identity_mode": self.source_identity_mode,
            "source_leading_dimension": self.source_leading_dimension,
            "leading_dimension": self.leading_dimension,
            "track_sample_key_content_sha256": self.track_sample_key_content_sha256,
            "source_row_index": self.source_row_index.to_dict(),
            "source_frame_index": self.source_frame_index.to_dict(),
            "interpolation": self.interpolation.to_dict(),
            "source_instance_key": self.source_instance_key.to_dict(),
        }

    def digest(self) -> str:
        return _mapping_digest(self.to_dict())


@dataclass(frozen=True)
class BoundTrackSampleTimeLineage:
    record: TrackSampleTimeLineageRecord
    record_ref: str
    record_sha256: str
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _rowset_node: Any = field(repr=False, compare=False)
    _key_array_node: Any = field(repr=False, compare=False)
    _source_row_index_node: Any = field(repr=False, compare=False)
    _source_frame_index_node: Any = field(repr=False, compare=False)
    _interpolation_node: Any = field(repr=False, compare=False)
    _source_instance_key_node: Any = field(repr=False, compare=False)
    _source_temporal_authority: BoundSourceRowTemporalAuthority = field(
        repr=False,
        compare=False,
    )
    _verification_seal: object = field(repr=False, compare=False)

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    def assert_verified(self) -> None:
        require_bound_track_sample_time_lineage(self)


@dataclass(frozen=True)
class BoundRowIdentityContract:
    """Exact persisted rowset/key binding accepted by canonical writers."""

    contract: RowIdentityContract
    rowset_path: str
    key_array_path: str
    record_ref: str
    record_sha256: str
    track_time_lineage: BoundTrackSampleTimeLineage | None = field(
        repr=False,
        compare=False,
    )
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _rowset_node: Any = field(repr=False, compare=False)
    _key_array_node: Any = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)
    _manifest_authority: BoundCoordinateRecord | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _array_digest_evidence: Mapping[str, Mapping[str, Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def leading_dimension(self) -> int:
        return self.contract.leading_dimension

    def assert_verified(self) -> None:
        if (
            type(self) is not BoundRowIdentityContract
            or self._verification_seal is not _BOUND_ROW_IDENTITY_SEAL
        ):
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_binding_unverified",
                        "$",
                        "Row identity was not loaded from exact persisted nodes.",
                    ),
                )
            )
        try:
            current_archive = require_same_archive(
                self._rowset_node,
                self._key_array_node,
            )
        except ArchiveIdentityError as exc:
            raise RowIdentityContractError(
                (_issue("row_identity_archive_mismatch", "$", str(exc)),)
            ) from exc
        if current_archive != self._archive_identity:
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_archive_mismatch",
                        "$",
                        "Row identity nodes moved to a different archive/store.",
                    ),
                )
            )
        expected_key_path = f"{self.rowset_path}/{self.contract.key_array.ref}"
        if self.key_array_path != expected_key_path:
            raise RowIdentityContractError(
                (
                    _issue(
                        "key_array_ref_mismatch",
                        "$.key_array_path",
                        "Bound key path is not the exact rowset sibling.",
                    ),
                )
            )
        if self._manifest_authority is None and self.record_ref != (
            f"/{self.rowset_path}@{ROW_IDENTITY_CONTRACT_ATTR}"
        ):
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_record_ref_mismatch",
                        "$.record_ref",
                        "Bound record ref does not name the exact persisted rowset.",
                    ),
                )
            )
        if (
            self._manifest_authority is None
            and self.record_sha256 != self.contract.digest()
        ):
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_contract_digest_mismatch",
                        "$.record_sha256",
                        "Bound contract digest is stale.",
                    ),
                )
            )
        if self.contract.domain == TRACK_SAMPLE_DOMAIN:
            lineage = require_bound_track_sample_time_lineage(
                self.track_time_lineage
            )
            expected_lineage = TrackSampleTimeLineageRef(
                lineage.record_ref,
                lineage.record_sha256,
            )
            if self.contract.time_lineage != expected_lineage:
                raise RowIdentityContractError(
                    (
                        _issue(
                            "track_time_lineage_mismatch",
                            "$.time_lineage",
                            "Track identity does not bind its exact acquisition-time lineage.",
                        ),
                    )
                )
        elif self.track_time_lineage is not None:
            raise RowIdentityContractError(
                (
                    _issue(
                        "track_time_lineage_unexpected",
                        "$.time_lineage",
                        "Only track-sample identity may retain time lineage.",
                    ),
                )
            )
        if self._manifest_authority is None:
            if self._array_digest_evidence is None:
                current = validate_stamped_row_identity(
                    self._rowset_node,
                    self._key_array_node,
                    verify_values=True,
                    track_time_lineage=self.track_time_lineage,
                )
            else:
                with row_authority_array_digest_evidence_scope(
                    self._rowset_node,
                    self._array_digest_evidence,
                ):
                    current = validate_stamped_row_identity(
                        self._rowset_node,
                        self._key_array_node,
                        verify_values=True,
                        track_time_lineage=self.track_time_lineage,
                    )
        else:
            authority = verify_bound_coordinate_record(self._manifest_authority)
            if (
                authority.archive_identity != self._archive_identity
                or authority.record_ref != self.record_ref
                or authority.record_sha256 != self.record_sha256
                or authority.record_ref != f"/{self.rowset_path}@run_manifest"
                or self.contract.domain != OBSERVATION_INSTANCE_DOMAIN
                or self.track_time_lineage is not None
            ):
                raise RowIdentityContractError(
                    (
                        _issue(
                            "manifest_row_identity_authority_mismatch",
                            "$.record_ref",
                            "Manifest-bound observation identity no longer matches its exact authority.",
                        ),
                    )
                )
            try:
                key_values = np.array(
                    self._key_array_node[:],
                    copy=True,
                    order="C",
                )
                current = build_row_identity_contract(
                    domain=OBSERVATION_INSTANCE_DOMAIN,
                    values=key_values,
                )
            except Exception as exc:
                raise RowIdentityContractError(
                    (
                        _issue(
                            "manifest_row_identity_live_array_invalid",
                            "$.key_array",
                            f"Manifest-bound instance identity is invalid: {exc}.",
                        ),
                    )
                ) from exc
        if current != self.contract:
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_binding_stale",
                        "$",
                        "Persisted row identity changed after it was bound.",
                    ),
                )
            )

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity


def require_bound_row_identity_contract(
    value: Any,
) -> BoundRowIdentityContract:
    """Require the exact sealed row-identity type and re-read its live nodes.

    This is the single canonical writer boundary.  In particular, subclasses
    and duck-typed wrappers are not accepted even when they expose plausible
    contract fields or an ``assert_verified`` method.
    """

    if (
        type(value) is not BoundRowIdentityContract
        or getattr(value, "_verification_seal", None) is not _BOUND_ROW_IDENTITY_SEAL
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_binding_unverified",
                    "$",
                    "A sealed exact BoundRowIdentityContract is required.",
                ),
            )
        )
    lineage = value.track_time_lineage
    verify_persisted_proof(
        (
            "palette.bound_row_identity_contract.v1",
            value.archive_identity.kind,
            value.archive_identity.key,
            value.record_ref,
            value.record_sha256,
            value.rowset_path,
            value.key_array_path,
            None if lineage is None else lineage.record_ref,
            None if lineage is None else lineage.record_sha256,
        ),
        value.assert_verified,
    )
    return value


def _issue(code: str, path: str, message: str) -> RowIdentityIssue:
    return RowIdentityIssue(code=code, path=path, message=message)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _raw_canonical_equal(raw: Any, canonical: Any) -> bool:
    """Compare persisted JSON with exact container and scalar types."""

    if type(raw) is not type(canonical):
        return False
    if isinstance(canonical, Mapping):
        return set(raw) == set(canonical) and all(
            _raw_canonical_equal(raw[name], canonical[name]) for name in canonical
        )
    if isinstance(canonical, (list, tuple)):
        return len(raw) == len(canonical) and all(
            _raw_canonical_equal(left, right)
            for left, right in zip(raw, canonical, strict=True)
        )
    if isinstance(canonical, np.ndarray):
        return bool(np.array_equal(raw, canonical, equal_nan=True))
    result = raw == canonical
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _track_time_lineage_ref(
    value: Any,
    *,
    path: str,
) -> TrackSampleTimeLineageRef:
    if not isinstance(value, Mapping) or set(value) != {
        "record_ref",
        "record_sha256",
    }:
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_ref_invalid",
                    path,
                    "Track time lineage must be one exact record ref and digest.",
                ),
            )
        )
    record_ref = value["record_ref"]
    digest = value["record_sha256"]
    if (
        not isinstance(record_ref, str)
        or not record_ref.startswith("/")
        or not record_ref.endswith(f"@{TRACK_SAMPLE_TIME_LINEAGE_ATTR}")
        or "//" in record_ref
        or "/./" in record_ref
        or "/../" in record_ref
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_ref_invalid",
                    f"{path}.record_ref",
                    "Track time lineage ref must name the canonical rowset attr.",
                ),
            )
        )
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_ref_invalid",
                    f"{path}.record_sha256",
                    "Track time lineage requires a lowercase SHA-256 digest.",
                ),
            )
        )
    return TrackSampleTimeLineageRef(record_ref, digest)


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key {name!r}")
        result[name] = value
    return result


def _mapping_payload(value: Any) -> tuple[Mapping[str, Any] | None, list[RowIdentityIssue]]:
    if isinstance(value, RowIdentityContract):
        return value.to_dict(), []
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None, [
                _issue("row_identity_json_invalid", "$", "JSON bytes are not UTF-8.")
            ]
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=_reject_duplicate_json_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            return None, [
                _issue("row_identity_json_invalid", "$", f"JSON is invalid: {exc}.")
            ]
    if not isinstance(value, Mapping):
        return None, [
            _issue("row_identity_not_mapping", "$", "Contract must be a mapping.")
        ]
    return value, []


def identity_array_content_sha256(values: Any) -> str:
    """Hash exact dtype, shape, and C-order key bytes deterministically."""

    array = np.ascontiguousarray(np.asarray(values))
    header = {
        "canonicalization": ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION,
        "dtype": np.lib.format.dtype_to_descr(array.dtype),
        "shape": [int(item) for item in array.shape],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@contextmanager
def row_authority_array_digest_evidence_scope(
    archive_node: Any,
    values: Mapping[str, Mapping[str, Any]],
) -> Iterator[None]:
    """Reuse one validated immutable-array receipt for row authorities.

    The scope does not create authority.  Its caller must first validate a
    closed payload receipt and its mutation-exclusion lifecycle.  Within that
    proof epoch, row-identity and temporal-authority loaders can compare the
    receipt's exact dtype, shape, and canonical content digest instead of
    decoding the same immutable arrays again.
    """

    if not isinstance(values, Mapping) or any(
        type(path) is not str
        or not path
        or path.startswith("/")
        or not isinstance(record, Mapping)
        for path, record in values.items()
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "row_authority_digest_evidence_invalid",
                    "values",
                    "Receipt evidence must be a canonical array-path mapping.",
                ),
            )
        )
    token = _ACTIVE_ROW_AUTHORITY_ARRAY_DIGEST_EVIDENCE.set(
        (archive_identity(archive_node), dict(values))
    )
    try:
        yield
    finally:
        _ACTIVE_ROW_AUTHORITY_ARRAY_DIGEST_EVIDENCE.reset(token)


def _receipt_bound_array_content_sha256(node: Any) -> str | None:
    """Return exact active receipt evidence, or ``None`` outside a scope."""

    binding = _ACTIVE_ROW_AUTHORITY_ARRAY_DIGEST_EVIDENCE.get()
    if binding is None:
        return None
    expected_archive, evidence = binding
    observed_archive = archive_identity(node)
    path = _persisted_node_path(node, label="receipt_array")
    record = evidence.get(path)
    try:
        dtype = np.dtype(getattr(node, "dtype"))
        shape = tuple(int(value) for value in getattr(node, "shape"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RowIdentityContractError(
            (_issue("row_authority_array_metadata_invalid", path, str(exc)),)
        ) from exc
    digest = record.get("content_sha256") if isinstance(record, Mapping) else None
    if (
        observed_archive != expected_archive
        or not isinstance(record, Mapping)
        or set(record)
        != {"dtype", "shape", "content_sha256", "canonicalization"}
        or record.get("dtype") != dtype.str
        or record.get("shape") != [int(value) for value in shape]
        or record.get("canonicalization")
        != ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION
        or type(digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "row_authority_digest_evidence_mismatch",
                    path,
                    "Receipt evidence is missing or differs from live metadata.",
                ),
            )
        )
    return digest


def _capture_active_row_authority_digest_evidence(
    *nodes: Any,
) -> Mapping[str, Mapping[str, Any]] | None:
    binding = _ACTIVE_ROW_AUTHORITY_ARRAY_DIGEST_EVIDENCE.get()
    if binding is None:
        return None
    _expected_archive, evidence = binding
    captured: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        _receipt_bound_array_content_sha256(node)
        path = _persisted_node_path(node, label="receipt_array")
        captured[path] = copy.deepcopy(evidence[path])
    return captured


def _parse_key_array(
    value: Any,
    *,
    path: str,
    issues: list[RowIdentityIssue],
) -> RowIdentityKeyArray | None:
    if not isinstance(value, Mapping):
        issues.append(_issue("key_array_invalid", path, "Key-array record must be a mapping."))
        return None
    expected = {
        "schema_id",
        "schema_version",
        "domain",
        "mode",
        "ref",
        "components",
        "dtype",
        "shape",
        "leading_dimension",
        "unique",
        "content_sha256",
        "digest_canonicalization",
    }
    for name in sorted(expected - set(value)):
        issues.append(_issue("missing_field", f"{path}.{name}", "Required field is missing."))
    for name in sorted(set(value) - expected):
        issues.append(_issue("unknown_field", f"{path}.{name}", "Field is not in schema version 1."))

    if value.get("schema_id") != ROW_IDENTITY_KEY_SCHEMA_ID:
        issues.append(_issue("key_schema_unsupported", f"{path}.schema_id", "Unsupported key schema."))
    version = value.get("schema_version")
    if type(version) is not int or version != ROW_IDENTITY_KEY_SCHEMA_VERSION:
        issues.append(_issue("key_schema_unsupported", f"{path}.schema_version", "Unsupported key schema version."))

    domain = value.get("domain")
    if domain not in ROW_IDENTITY_DOMAINS:
        issues.append(_issue("identity_domain_unsupported", f"{path}.domain", f"Unsupported domain {domain!r}."))
    mode = value.get("mode")
    if mode not in ROW_IDENTITY_MODES:
        issues.append(_issue("identity_mode_unsupported", f"{path}.mode", f"Unsupported mode {mode!r}."))
    ref = value.get("ref")
    if not isinstance(ref, str) or _KEY_REF_RE.fullmatch(ref) is None:
        issues.append(_issue("key_array_ref_invalid", f"{path}.ref", "Key ref must name one sibling array."))

    raw_components = value.get("components")
    components: tuple[str, ...] = ()
    if not isinstance(raw_components, list) or not raw_components:
        issues.append(_issue("key_components_invalid", f"{path}.components", "Components must be a non-empty list."))
    else:
        parsed_components: list[str] = []
        for index, component in enumerate(raw_components):
            if not isinstance(component, str) or _COMPONENT_RE.fullmatch(component) is None:
                issues.append(_issue("key_component_invalid", f"{path}.components[{index}]", "Component must be a canonical name."))
            else:
                parsed_components.append(component)
        components = tuple(parsed_components)
        if len(set(components)) != len(components):
            issues.append(_issue("key_component_duplicate", f"{path}.components", "Components must be unique."))

    dtype = value.get("dtype")
    if not isinstance(dtype, str):
        issues.append(_issue("key_dtype_invalid", f"{path}.dtype", "dtype must be a canonical NumPy dtype string."))
    else:
        try:
            if np.dtype(dtype).str != dtype:
                issues.append(_issue("key_dtype_noncanonical", f"{path}.dtype", "dtype is not canonical."))
        except TypeError:
            issues.append(_issue("key_dtype_invalid", f"{path}.dtype", "dtype is invalid."))

    raw_shape = value.get("shape")
    shape: tuple[int, ...] = ()
    if not isinstance(raw_shape, list) or not raw_shape:
        issues.append(_issue("key_shape_invalid", f"{path}.shape", "shape must be a non-empty list."))
    else:
        parsed_shape: list[int] = []
        for index, dimension in enumerate(raw_shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                issues.append(_issue("key_shape_invalid", f"{path}.shape[{index}]", "Dimensions must be nonnegative integers."))
            else:
                parsed_shape.append(dimension)
        shape = tuple(parsed_shape)

    leading = value.get("leading_dimension")
    if isinstance(leading, bool) or not isinstance(leading, int) or leading < 0:
        issues.append(_issue("leading_dimension_invalid", f"{path}.leading_dimension", "leading_dimension must be nonnegative."))
    elif shape and leading != shape[0]:
        issues.append(_issue("leading_dimension_mismatch", f"{path}.leading_dimension", "leading_dimension must equal shape[0]."))
    unique = value.get("unique")
    if unique is not True:
        issues.append(_issue("identity_uniqueness_required", f"{path}.unique", "Canonical row identity must assert unique=true."))
    content_digest = value.get("content_sha256")
    if not isinstance(content_digest, str) or _SHA256_RE.fullmatch(content_digest) is None:
        issues.append(_issue("key_content_digest_invalid", f"{path}.content_sha256", "content_sha256 must be lowercase SHA-256."))
    canonicalization = value.get("digest_canonicalization")
    if canonicalization != ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION:
        issues.append(_issue("key_digest_canonicalization_unsupported", f"{path}.digest_canonicalization", "Unsupported key digest canonicalization."))

    if issues:
        return None
    assert isinstance(domain, str)
    assert isinstance(mode, str)
    assert isinstance(ref, str)
    assert isinstance(dtype, str)
    assert isinstance(leading, int)
    assert isinstance(content_digest, str)
    return RowIdentityKeyArray(
        domain=domain,
        mode=mode,
        ref=ref,
        components=components,
        dtype=dtype,
        shape=shape,
        leading_dimension=leading,
        unique=True,
        content_sha256=content_digest,
    )


def _profile_issues(key: RowIdentityKeyArray) -> tuple[RowIdentityIssue, ...]:
    issues: list[RowIdentityIssue] = []
    path = "$.key_arrays[0]"
    if key.domain == OBSERVATION_INSTANCE_DOMAIN:
        expected = (INSTANCE_KEY_MODE, INSTANCE_KEY_ARRAY_REF, ("instance_key",), "<u8")
        if (key.mode, key.ref, key.components, key.dtype) != expected:
            issues.append(_issue("observation_identity_profile_mismatch", path, "Observation identity requires uint64 instance_key."))
        if len(key.shape) != 1:
            issues.append(_issue("observation_identity_rank_mismatch", f"{path}.shape", "instance_key must be rank 1."))
    elif key.domain == TRACK_SAMPLE_DOMAIN:
        expected = (
            TRACK_SAMPLE_KEY_MODE,
            TRACK_SAMPLE_KEY_ARRAY_REF,
            ("track_id", "acquisition_frame_index"),
            "<i8",
        )
        if (key.mode, key.ref, key.components, key.dtype) != expected:
            issues.append(_issue("track_identity_profile_mismatch", path, "Track identity requires int64 track_sample_key[track_id, acquisition_frame_index]."))
        if len(key.shape) != 2 or (len(key.shape) == 2 and key.shape[1] != 2):
            issues.append(_issue("track_identity_rank_mismatch", f"{path}.shape", "track_sample_key must have shape (N, 2)."))
    elif key.domain == STIMULUS_STATE_DOMAIN:
        if key.mode != STIMULUS_STATE_KEY_MODE or key.ref != STIMULUS_STATE_KEY_ARRAY_REF or key.dtype != "<i8":
            issues.append(_issue("stimulus_identity_profile_mismatch", path, "Stimulus identity requires signed-int64 stimulus_state_key."))
        if not set(key.components).issubset(STIMULUS_STATE_KEY_COMPONENTS):
            issues.append(_issue("stimulus_identity_component_unsupported", f"{path}.components", "Stimulus key contains unsupported components."))
        if len(key.shape) == 1:
            if len(key.components) != 1:
                issues.append(_issue("stimulus_identity_component_count_mismatch", f"{path}.components", "A rank-1 key requires one component."))
        elif len(key.shape) == 2:
            if key.shape[1] != len(key.components) or not (1 <= key.shape[1] <= 4):
                issues.append(_issue("stimulus_identity_component_count_mismatch", f"{path}.shape", "Composite stimulus key columns must match one to four components."))
        else:
            issues.append(_issue("stimulus_identity_rank_mismatch", f"{path}.shape", "stimulus_state_key must be rank 1 or 2."))
    return tuple(issues)


def _parse_contract(value: Any) -> RowIdentityContract:
    payload, initial = _mapping_payload(value)
    issues = list(initial)
    if payload is None:
        raise RowIdentityContractError(issues)
    expected = {
        "schema_id",
        "schema_version",
        "domain",
        "mode",
        "leading_dimension",
        "unique",
        "key_arrays",
        "time_lineage",
    }
    for name in sorted(expected - set(payload)):
        issues.append(_issue("missing_field", f"$.{name}", "Required field is missing."))
    for name in sorted(set(payload) - expected):
        issues.append(_issue("unknown_field", f"$.{name}", "Field is not in schema version 1."))
    if payload.get("schema_id") != ROW_IDENTITY_CONTRACT_SCHEMA_ID:
        issues.append(_issue("identity_schema_unsupported", "$.schema_id", "Unsupported row-identity schema."))
    version = payload.get("schema_version")
    if type(version) is not int or version != ROW_IDENTITY_CONTRACT_SCHEMA_VERSION:
        issues.append(_issue("identity_schema_unsupported", "$.schema_version", "Unsupported row-identity schema version."))
    domain = payload.get("domain")
    if domain not in ROW_IDENTITY_DOMAINS:
        issues.append(_issue("identity_domain_unsupported", "$.domain", f"Unsupported domain {domain!r}."))
    mode = payload.get("mode")
    if mode not in ROW_IDENTITY_MODES:
        issues.append(_issue("identity_mode_unsupported", "$.mode", f"Unsupported mode {mode!r}."))
    leading = payload.get("leading_dimension")
    if isinstance(leading, bool) or not isinstance(leading, int) or leading < 0:
        issues.append(_issue("leading_dimension_invalid", "$.leading_dimension", "leading_dimension must be nonnegative."))
    if payload.get("unique") is not True:
        issues.append(_issue("identity_uniqueness_required", "$.unique", "Canonical row identity must assert unique=true."))
    raw_keys = payload.get("key_arrays")
    keys: tuple[RowIdentityKeyArray, ...] = ()
    if not isinstance(raw_keys, list) or len(raw_keys) != 1:
        issues.append(_issue("key_array_count_invalid", "$.key_arrays", "Schema version 1 requires exactly one key array."))
    else:
        key_issues: list[RowIdentityIssue] = []
        parsed = _parse_key_array(raw_keys[0], path="$.key_arrays[0]", issues=key_issues)
        issues.extend(key_issues)
        if parsed is not None:
            keys = (parsed,)
            issues.extend(_profile_issues(parsed))
            if parsed.domain != domain:
                issues.append(_issue("key_domain_mismatch", "$.key_arrays[0].domain", "Key domain disagrees with contract."))
            if parsed.mode != mode:
                issues.append(_issue("key_mode_mismatch", "$.key_arrays[0].mode", "Key mode disagrees with contract."))
            if isinstance(leading, int) and parsed.leading_dimension != leading:
                issues.append(_issue("key_leading_dimension_mismatch", "$.key_arrays[0].leading_dimension", "Key row count disagrees with contract."))
    time_lineage: TrackSampleTimeLineageRef | None = None
    raw_time_lineage = payload.get("time_lineage")
    if domain == TRACK_SAMPLE_DOMAIN:
        if raw_time_lineage is None:
            issues.append(
                _issue(
                    "track_time_lineage_required",
                    "$.time_lineage",
                    "Track-sample identity requires exact acquisition-time lineage.",
                )
            )
        else:
            try:
                time_lineage = _track_time_lineage_ref(
                    raw_time_lineage,
                    path="$.time_lineage",
                )
            except RowIdentityContractError as exc:
                issues.extend(exc.issues)
    elif raw_time_lineage is not None:
        issues.append(
            _issue(
                "track_time_lineage_unexpected",
                "$.time_lineage",
                "Only track-sample identity may carry acquisition-time lineage.",
            )
        )
    if issues:
        raise RowIdentityContractError(issues)
    assert isinstance(domain, str)
    assert isinstance(mode, str)
    assert isinstance(leading, int)
    return RowIdentityContract(
        domain=domain,
        mode=mode,
        leading_dimension=leading,
        unique=True,
        key_arrays=keys,
        time_lineage=time_lineage,
    )


def parse_row_identity_contract(value: Any) -> RowIdentityContract:
    """Parse one strict canonical row-identity contract."""

    return _parse_contract(value)


def validate_row_identity_contract(value: Any) -> tuple[RowIdentityIssue, ...]:
    """Return structured contract issues without raising."""

    try:
        _parse_contract(value)
    except RowIdentityContractError as exc:
        return exc.issues
    return ()


def _rows_unique(values: np.ndarray) -> bool:
    if values.ndim == 1:
        return int(np.unique(values).shape[0]) == int(values.shape[0])
    return int(np.unique(values, axis=0).shape[0]) == int(values.shape[0])


def build_row_identity_contract(
    *,
    domain: str,
    values: Any,
    components: Sequence[str] | None = None,
    track_time_lineage: BoundTrackSampleTimeLineage | None = None,
) -> RowIdentityContract:
    """Build one canonical contract from the exact persisted key values."""

    array = np.asarray(values)
    lineage: BoundTrackSampleTimeLineage | None = None
    if domain == OBSERVATION_INSTANCE_DOMAIN:
        mode = INSTANCE_KEY_MODE
        ref = INSTANCE_KEY_ARRAY_REF
        normalized_components = ("instance_key",)
    elif domain == TRACK_SAMPLE_DOMAIN:
        lineage = require_bound_track_sample_time_lineage(track_time_lineage)
        mode = TRACK_SAMPLE_KEY_MODE
        ref = TRACK_SAMPLE_KEY_ARRAY_REF
        normalized_components = ("track_id", "acquisition_frame_index")
        if identity_array_content_sha256(array) != lineage.record.track_sample_key_content_sha256:
            raise RowIdentityContractError(
                (
                    _issue(
                        "track_time_key_mismatch",
                        "values",
                        "Track key values differ from the exact acquisition-time lineage binding.",
                    ),
                )
            )
    elif domain == STIMULUS_STATE_DOMAIN:
        mode = STIMULUS_STATE_KEY_MODE
        ref = STIMULUS_STATE_KEY_ARRAY_REF
        if components is None:
            raise RowIdentityContractError(
                (_issue("stimulus_identity_components_required", "$.key_arrays[0].components", "Stimulus identity requires explicit components."),)
            )
        normalized_components = tuple(str(item) for item in components)
    else:
        raise RowIdentityContractError(
            (_issue("identity_domain_unsupported", "$.domain", f"Unsupported domain {domain!r}."),)
        )

    if domain != TRACK_SAMPLE_DOMAIN and track_time_lineage is not None:
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_unexpected",
                    "track_time_lineage",
                    "Only track-sample identity accepts acquisition-time lineage.",
                ),
            )
        )

    key = RowIdentityKeyArray(
        domain=domain,
        mode=mode,
        ref=ref,
        components=normalized_components,
        dtype=array.dtype.str,
        shape=tuple(int(item) for item in array.shape),
        leading_dimension=int(array.shape[0]) if array.ndim else 0,
        unique=_rows_unique(array) if array.ndim in {1, 2} else False,
        content_sha256=identity_array_content_sha256(array),
    )
    contract = RowIdentityContract(
        domain=domain,
        mode=mode,
        leading_dimension=key.leading_dimension,
        unique=key.unique,
        key_arrays=(key,),
        time_lineage=(
            TrackSampleTimeLineageRef(
                lineage.record_ref,
                lineage.record_sha256,
            )
            if domain == TRACK_SAMPLE_DOMAIN
            else None
        ),
    )
    parsed = parse_row_identity_contract(contract)
    value_issues = validate_row_identity_values(parsed, array)
    if value_issues:
        raise RowIdentityContractError(value_issues)
    return parsed


def build_track_sample_key(
    track_ids: Any,
    acquisition_frame_indices: Any,
) -> np.ndarray:
    """Build ``(track_id, acquisition_frame_index)`` sample identity.

    The second input must be the source acquisition-camera frame domain.  This
    builder deliberately has no generic ``frame_indices`` parameter because a
    model-input, stimulus, or run-local index would create a different and
    scientifically ambiguous identity.
    """

    tracks = np.asarray(track_ids)
    frames = np.asarray(acquisition_frame_indices)
    if tracks.ndim != 1 or frames.ndim != 1 or tracks.shape != frames.shape:
        raise RowIdentityContractError(
            (_issue("track_identity_shape_mismatch", "$", "track_ids and acquisition_frame_indices must be equal-length rank-1 arrays."),)
        )
    if tracks.dtype.kind not in "iu" or frames.dtype.kind not in "iu":
        raise RowIdentityContractError(
            (_issue("track_identity_dtype_invalid", "$", "track_ids and acquisition_frame_indices must be integer arrays."),)
        )
    if (tracks.size and int(tracks.min()) < 0) or (frames.size and int(frames.min()) < 0):
        raise RowIdentityContractError(
            (_issue("identity_value_negative", "$", "Track identity values must be nonnegative."),)
        )
    int64_max = np.iinfo(np.int64).max
    if (tracks.size and int(tracks.max()) > int64_max) or (
        frames.size and int(frames.max()) > int64_max
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "identity_value_overflow",
                    "$",
                    "Track identity values must fit signed int64 exactly.",
                ),
            )
        )
    return np.column_stack(
        (
            tracks.astype(np.int64, copy=False),
            frames.astype(np.int64, copy=False),
        )
    )


def validate_row_identity_values(
    contract: Any,
    values: Any,
) -> tuple[RowIdentityIssue, ...]:
    """Validate exact key-array metadata, uniqueness, and content digest."""

    parsed = parse_row_identity_contract(contract)
    key = parsed.key_array
    array = np.asarray(values)
    issues: list[RowIdentityIssue] = []
    if array.dtype.str != key.dtype:
        issues.append(_issue("key_dtype_mismatch", "$.key_arrays[0].dtype", f"Persisted dtype {array.dtype.str!r} does not match {key.dtype!r}."))
    if tuple(int(item) for item in array.shape) != key.shape:
        issues.append(_issue("key_shape_mismatch", "$.key_arrays[0].shape", "Persisted shape does not match the contract."))
    if array.ndim not in {1, 2}:
        issues.append(_issue("key_rank_invalid", "$.key_arrays[0].shape", "Identity key must be rank 1 or 2."))
    elif not _rows_unique(array):
        issues.append(_issue("identity_values_not_unique", "$.key_arrays[0].unique", "Persisted identity rows are not unique."))
    if array.dtype.kind in "iu" and array.size and int(array.min()) < 0:
        issues.append(_issue("identity_value_negative", "$.key_arrays[0]", "Canonical identity values must be nonnegative."))
    if identity_array_content_sha256(array) != key.content_sha256:
        issues.append(_issue("key_content_digest_mismatch", "$.key_arrays[0].content_sha256", "Persisted key content does not match its digest."))
    return tuple(issues)


def row_identity_contract_attrs(contract: Any) -> dict[str, Any]:
    parsed = parse_row_identity_contract(contract)
    return {
        ROW_IDENTITY_CONTRACT_ATTR: parsed.to_dict(),
        ROW_IDENTITY_CONTRACT_DIGEST_ATTR: parsed.digest(),
    }


def row_identity_key_attrs(contract: Any) -> dict[str, Any]:
    parsed = parse_row_identity_contract(contract)
    key = parsed.key_array
    return {
        ROW_IDENTITY_KEY_ATTR: key.to_dict(),
        ROW_IDENTITY_KEY_DIGEST_ATTR: key.digest(),
        ROW_IDENTITY_CONTRACT_REF_ATTR: ROW_IDENTITY_LOCAL_CONTRACT_REF,
        ROW_IDENTITY_CONTRACT_SHA256_ATTR: parsed.digest(),
    }


def load_row_identity_contract_attrs(attrs: Mapping[str, Any]) -> RowIdentityContract:
    if ROW_IDENTITY_CONTRACT_ATTR not in attrs:
        raise RowIdentityContractError(
            (_issue("row_identity_contract_missing", f"@{ROW_IDENTITY_CONTRACT_ATTR}", "Row identity contract attr is missing."),)
        )
    if ROW_IDENTITY_CONTRACT_DIGEST_ATTR not in attrs:
        raise RowIdentityContractError(
            (_issue("row_identity_contract_digest_missing", f"@{ROW_IDENTITY_CONTRACT_DIGEST_ATTR}", "Row identity contract digest is missing."),)
        )
    raw_contract = attrs[ROW_IDENTITY_CONTRACT_ATTR]
    contract = parse_row_identity_contract(raw_contract)
    if not _raw_canonical_equal(raw_contract, contract.to_dict()):
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_contract_noncanonical",
                    f"@{ROW_IDENTITY_CONTRACT_ATTR}",
                    "Persisted contract mapping is not exact canonical JSON form.",
                ),
            )
        )
    digest = attrs[ROW_IDENTITY_CONTRACT_DIGEST_ATTR]
    if not isinstance(digest, str) or digest != contract.digest():
        raise RowIdentityContractError(
            (_issue("row_identity_contract_digest_mismatch", f"@{ROW_IDENTITY_CONTRACT_DIGEST_ATTR}", "Stored contract digest does not match canonical content."),)
        )
    return contract


def load_row_identity_key_attrs(
    attrs: Mapping[str, Any],
    *,
    contract: Any,
) -> RowIdentityKeyArray:
    parsed = parse_row_identity_contract(contract)
    required = (
        ROW_IDENTITY_KEY_ATTR,
        ROW_IDENTITY_KEY_DIGEST_ATTR,
        ROW_IDENTITY_CONTRACT_REF_ATTR,
        ROW_IDENTITY_CONTRACT_SHA256_ATTR,
    )
    missing = [name for name in required if name not in attrs]
    if missing:
        raise RowIdentityContractError(
            tuple(_issue("row_identity_key_attr_missing", f"@{name}", "Required key attr is missing.") for name in missing)
        )
    issues: list[RowIdentityIssue] = []
    raw_key = attrs[ROW_IDENTITY_KEY_ATTR]
    key = _parse_key_array(raw_key, path=f"@{ROW_IDENTITY_KEY_ATTR}", issues=issues)
    if issues or key is None:
        raise RowIdentityContractError(issues)
    if not _raw_canonical_equal(raw_key, key.to_dict()):
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_key_noncanonical",
                    f"@{ROW_IDENTITY_KEY_ATTR}",
                    "Persisted key mapping is not exact canonical JSON form.",
                ),
            )
        )
    if attrs[ROW_IDENTITY_KEY_DIGEST_ATTR] != key.digest():
        raise RowIdentityContractError(
            (_issue("row_identity_key_digest_mismatch", f"@{ROW_IDENTITY_KEY_DIGEST_ATTR}", "Stored key-record digest is invalid."),)
        )
    if attrs[ROW_IDENTITY_CONTRACT_REF_ATTR] != ROW_IDENTITY_LOCAL_CONTRACT_REF:
        raise RowIdentityContractError(
            (_issue("row_identity_contract_ref_invalid", f"@{ROW_IDENTITY_CONTRACT_REF_ATTR}", "Key must reference its owning row-set contract."),)
        )
    if attrs[ROW_IDENTITY_CONTRACT_SHA256_ATTR] != parsed.digest():
        raise RowIdentityContractError(
            (_issue("row_identity_contract_digest_mismatch", f"@{ROW_IDENTITY_CONTRACT_SHA256_ATTR}", "Key does not bind the owning contract digest."),)
        )
    if key != parsed.key_array:
        raise RowIdentityContractError(
            (_issue("row_identity_key_contract_mismatch", f"@{ROW_IDENTITY_KEY_ATTR}", "Key metadata disagrees with the owning contract."),)
        )
    return key


def _mutable_attrs(node: Any, *, label: str) -> Any:
    attrs = getattr(node, "attrs", None)
    if attrs is None:
        raise RowIdentityContractError(
            (_issue("row_identity_attrs_unavailable", label, "Node must expose mutable attrs."),)
        )
    # A capability-shaped mapping is not a safe transaction boundary.  A
    # hostile ``dict`` subclass can perform a partial scientific write and
    # then make rollback fail.  Canonical identity publication therefore uses
    # only the exact built-in mapping used by deterministic fakes or the exact
    # Zarr Attributes implementation used by persisted archives.  This check
    # is read-only and happens before the first attrs mutation.
    trusted = type(attrs) is dict
    if not trusted:
        try:
            from zarr.core.attributes import Attributes as ZarrAttributes
        except ImportError:  # pragma: no cover - Palette depends on zarr
            ZarrAttributes = None  # type: ignore[assignment,misc]
        trusted = ZarrAttributes is not None and type(attrs) is ZarrAttributes
    if not trusted:
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_stamp_preflight_failed",
                    label,
                    "Attrs mapping type is not an exact trusted dict or Zarr "
                    "Attributes implementation; no write was attempted.",
                ),
            )
        )
    for operation in ("update", "__setitem__", "__delitem__"):
        if not callable(getattr(attrs, operation, None)):
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_stamp_preflight_failed",
                        label,
                        f"Attrs mapping lacks required {operation} mutation "
                        "support; no write was attempted.",
                    ),
                )
            )
    return attrs


def _persisted_node_path(node: Any, *, label: str) -> str:
    raw_path = getattr(node, "path", None)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raw_path = getattr(node, "name", None)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_node_path_unavailable",
                    label,
                    "Canonical row-identity node must expose its persisted path.",
                ),
            )
        )
    path = raw_path.strip()
    if (
        path != raw_path
        or path.startswith("/")
        or path.endswith("/")
        or "//" in path
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_node_path_noncanonical",
                    label,
                    "Persisted node paths must be canonical archive-relative paths.",
                ),
            )
        )
    segments = path.split("/")
    if any(
        segment in {"", ".", ".."}
        or _NODE_PATH_SEGMENT_RE.fullmatch(segment) is None
        for segment in segments
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_node_path_noncanonical",
                    label,
                    "Persisted node path contains a noncanonical segment.",
                ),
            )
        )
    return path


def _require_key_array_node_ref(
    rowset_node: Any,
    key_array_node: Any,
    *,
    expected_ref: str,
) -> None:
    """Bind a key contract to the exact sibling array named by ``ref``."""

    try:
        require_same_archive(rowset_node, key_array_node)
    except ArchiveIdentityError as exc:
        raise RowIdentityContractError(
            (_issue("row_identity_archive_mismatch", "$", str(exc)),)
        ) from exc
    rowset_path = _persisted_node_path(rowset_node, label="rowset")
    key_path = _persisted_node_path(key_array_node, label="key_array")
    expected_path = f"{rowset_path}/{expected_ref}"
    if key_path != expected_path:
        raise RowIdentityContractError(
            (
                _issue(
                    "key_array_ref_mismatch",
                    "key_array.path",
                    f"Expected sibling path {expected_path!r}, found {key_path!r}.",
                ),
            )
        )


def _time_array_record(
    node: Any,
    *,
    expected_path: str,
    expected_dtype: np.dtype[Any],
    expected_shape: tuple[int, ...],
    label: str,
) -> tuple[np.ndarray | None, TrackSampleTimeArrayRecord]:
    path = _persisted_node_path(node, label=label)
    if path != expected_path:
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_array_path_mismatch",
                    label,
                    f"Expected exact rowset child {expected_path!r}, found {path!r}.",
                ),
            )
        )
    declared_shape = tuple(int(item) for item in getattr(node, "shape", ()))
    try:
        declared_dtype = np.dtype(getattr(node, "dtype"))
    except (AttributeError, TypeError) as exc:
        raise RowIdentityContractError(
            (_issue("track_time_array_dtype_invalid", label, str(exc)),)
        ) from exc
    if (
        declared_shape != expected_shape
        or declared_dtype != expected_dtype
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_array_metadata_mismatch",
                    label,
                    f"Expected dtype {expected_dtype.str!r} and shape {expected_shape!r}.",
                ),
            )
        )
    receipt_digest = _receipt_bound_array_content_sha256(node)
    if receipt_digest is not None:
        return None, TrackSampleTimeArrayRecord(
            ref=f"/{path}",
            dtype=(
                tuple(
                    (name, np.dtype(field[0]).str)
                    for name, field in declared_dtype.fields.items()
                )
                if declared_dtype.fields is not None
                else declared_dtype.str
            ),
            shape=declared_shape,
            content_sha256=receipt_digest,
        )
    try:
        values = np.array(node[:], copy=True, order="C")
    except Exception as exc:
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_array_unreadable",
                    label,
                    f"Unable to read exact time-lineage values: {exc}.",
                ),
            )
        ) from exc
    if values.shape != expected_shape or values.dtype != expected_dtype:
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_array_metadata_mismatch",
                    label,
                    f"Expected dtype {expected_dtype.str!r} and shape {expected_shape!r}.",
                ),
            )
        )
    return values, TrackSampleTimeArrayRecord(
        ref=f"/{path}",
        dtype=(
            tuple((name, np.dtype(field[0]).str) for name, field in values.dtype.fields.items())
            if values.dtype.fields is not None
            else values.dtype.str
        ),
        shape=values.shape,
        content_sha256=identity_array_content_sha256(values),
    )


def _require_live_array_record(
    node: Any,
    record: TrackSampleTimeArrayRecord,
    *,
    label: str,
) -> np.ndarray:
    """Re-read one bound array and reject payload or metadata drift."""

    try:
        values = np.array(node[:], copy=True, order="C")
        declared_shape = tuple(int(item) for item in getattr(node, "shape"))
        declared_dtype = np.dtype(getattr(node, "dtype"))
    except Exception as exc:
        raise RowIdentityContractError(
            (_issue("temporal_array_unreadable", label, str(exc)),)
        ) from exc
    expected_dtype = (
        np.dtype(list(record.dtype))
        if isinstance(record.dtype, tuple)
        else np.dtype(record.dtype)
    )
    if (
        values.shape != record.shape
        or declared_shape != record.shape
        or values.dtype != expected_dtype
        or declared_dtype != expected_dtype
        or identity_array_content_sha256(values) != record.content_sha256
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "temporal_array_changed_during_binding",
                    label,
                    "Temporal-lineage array payload or metadata changed during verification.",
                ),
            )
        )
    return values


def _build_source_row_temporal_authority_record(
    source_rowset_node: Any,
    source_acquisition_frame_index_node: Any,
    *,
    source_row_identity: BoundRowIdentityContract,
    acquisition_frame: Any,
) -> tuple[SourceRowTemporalAuthorityRecord, ArchiveIdentity]:
    """Derive one immediate-source temporal record from exact live authorities."""

    source_identity = require_bound_row_identity_contract(source_row_identity)
    try:
        from fisheye.shared.pixel_frame_authority import (
            PixelFrameAuthorityError,
            require_bound_acquisition_camera_frame,
        )

        acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    except (ImportError, PixelFrameAuthorityError) as exc:
        raise RowIdentityContractError(
            (
                _issue(
                    "acquisition_frame_unverified",
                    "acquisition_frame",
                    f"A sealed acquisition-owned camera frame is required: {exc}.",
                ),
            )
        ) from exc
    source_rowset_path = _persisted_node_path(
        source_rowset_node,
        label="source_rowset",
    )
    if source_identity.rowset_path != source_rowset_path:
        raise RowIdentityContractError(
            (
                _issue(
                    "source_row_identity_mismatch",
                    "source_row_identity",
                    "Source temporal authority must bind the exact selected source rowset.",
                ),
            )
        )
    leading = source_identity.leading_dimension
    source_frames, source_frame_record = _time_array_record(
        source_acquisition_frame_index_node,
        expected_path=(
            f"{source_rowset_path}/{SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF}"
        ),
        expected_dtype=np.dtype("<i8"),
        expected_shape=(leading,),
        label="source_acquisition_frame_index",
    )
    if source_frames is not None and (
        np.any(source_frames < 0)
        or np.any(source_frames >= acquisition.record.source_total_frames)
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "source_time_frame_out_of_range",
                    "source_acquisition_frame_index",
                    "Every source row must map inside the exact acquisition frame domain.",
                ),
            )
        )
    observation_instance_record: TrackSampleTimeArrayRecord | None = None
    if source_identity.contract.domain == OBSERVATION_INSTANCE_DOMAIN:
        source_keys, observation_instance_record = _time_array_record(
            source_identity._key_array_node,
            expected_path=source_identity.key_array_path,
            expected_dtype=np.dtype("<u8"),
            expected_shape=(leading,),
            label="observation_instance_key",
        )
        if source_identity.contract.mode != INSTANCE_KEY_MODE:
            raise RowIdentityContractError(
                (
                    _issue(
                        "source_observation_identity_invalid",
                        "source_row_identity",
                        "Observation temporal authority requires canonical instance_key identity.",
                    ),
                )
            )
        observed_key_digest = (
            observation_instance_record.content_sha256
            if source_keys is None
            else identity_array_content_sha256(source_keys)
        )
        if observed_key_digest != source_identity.contract.key_array.content_sha256:
            raise RowIdentityContractError(
                (
                    _issue(
                        "source_observation_identity_mismatch",
                        "observation_instance_key",
                        "Observation key payload differs from its source row identity.",
                    ),
                )
            )
    try:
        identity = require_same_archive(
            source_rowset_node,
            source_acquisition_frame_index_node,
            source_identity._rowset_node,
            source_identity._key_array_node,
            acquisition._authority_node,
        )
    except ArchiveIdentityError as exc:
        raise RowIdentityContractError(
            (_issue("row_identity_archive_mismatch", "$", str(exc)),)
        ) from exc
    if acquisition.archive_identity != identity:
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_archive_mismatch",
                    "$",
                    "Source row identity, temporal mapping, and acquisition authority span archives.",
                ),
            )
        )
    if source_frames is not None:
        _require_live_array_record(
            source_acquisition_frame_index_node,
            source_frame_record,
            label="source_acquisition_frame_index",
        )
    if observation_instance_record is not None and source_keys is not None:
        _require_live_array_record(
            source_identity._key_array_node,
            observation_instance_record,
            label="observation_instance_key",
        )
    record = SourceRowTemporalAuthorityRecord(
        acquisition_camera_frame=TrackSampleTimeLineageRef(
            acquisition.record_ref,
            acquisition.record_sha256,
        ),
        recording_id=acquisition.record.recording_id,
        camera_id=acquisition.record.camera_id,
        source_total_frames=acquisition.record.source_total_frames,
        source_rowset_ref=f"/{source_rowset_path}",
        source_row_identity=TrackSampleTimeLineageRef(
            source_identity.record_ref,
            source_identity.record_sha256,
        ),
        source_identity_domain=source_identity.contract.domain,
        source_identity_mode=source_identity.contract.mode,
        source_leading_dimension=leading,
        source_acquisition_frame_index=source_frame_record,
        observation_instance_key=observation_instance_record,
    )
    return record, identity


def build_source_row_temporal_authority_record(
    source_rowset_node: Any,
    source_acquisition_frame_index_node: Any,
    *,
    source_row_identity: BoundRowIdentityContract,
    acquisition_frame: Any,
) -> SourceRowTemporalAuthorityRecord:
    """Build, but do not persist, one exact source-row temporal record."""

    record, _ = _build_source_row_temporal_authority_record(
        source_rowset_node,
        source_acquisition_frame_index_node,
        source_row_identity=source_row_identity,
        acquisition_frame=acquisition_frame,
    )
    return record


def load_bound_source_row_temporal_authority(
    source_rowset_node: Any,
    source_acquisition_frame_index_node: Any,
    *,
    source_row_identity: BoundRowIdentityContract,
    acquisition_frame: Any,
) -> BoundSourceRowTemporalAuthority:
    """Load a source temporal authority using only persisted nodes/records."""

    attrs = getattr(source_rowset_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_attrs_unavailable",
                    "source_rowset",
                    "Source rowset attrs are required.",
                ),
            )
        )
    record, identity = _build_source_row_temporal_authority_record(
        source_rowset_node,
        source_acquisition_frame_index_node,
        source_row_identity=source_row_identity,
        acquisition_frame=acquisition_frame,
    )
    raw = attrs.get(SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR)
    if not _raw_canonical_equal(raw, record.to_dict()):
        raise RowIdentityContractError(
            (
                _issue(
                    "source_temporal_authority_noncanonical",
                    f"@{SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR}",
                    "Persisted source temporal authority differs from exact live authorities.",
                ),
            )
        )
    if attrs.get(SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR) != record.digest():
        raise RowIdentityContractError(
            (
                _issue(
                    "source_temporal_authority_digest_mismatch",
                    f"@{SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR}",
                    "Persisted source temporal-authority digest is stale.",
                ),
            )
        )
    rowset_path = _persisted_node_path(source_rowset_node, label="source_rowset")
    return BoundSourceRowTemporalAuthority(
        record=record,
        record_ref=f"/{rowset_path}@{SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR}",
        record_sha256=record.digest(),
        source_row_identity=source_row_identity,
        acquisition_frame=acquisition_frame,
        _archive_identity=identity,
        _source_rowset_node=source_rowset_node,
        _source_frame_index_node=source_acquisition_frame_index_node,
        _manifest_authority=None,
        _array_digest_evidence=_capture_active_row_authority_digest_evidence(
            source_row_identity._key_array_node,
            source_acquisition_frame_index_node,
        ),
        _verification_seal=_BOUND_SOURCE_ROW_TEMPORAL_AUTHORITY_SEAL,
    )


def _bind_manifest_source_row_temporal_authority(
    source_rowset_node: Any,
    source_acquisition_frame_index_node: Any,
    *,
    source_row_identity: BoundRowIdentityContract,
    acquisition_frame: Any,
    manifest_authority: BoundCoordinateRecord,
) -> BoundSourceRowTemporalAuthority:
    """Bind acquisition time through one already validated immutable manifest."""

    authority = verify_bound_coordinate_record(manifest_authority)
    identity = require_bound_row_identity_contract(source_row_identity)
    record, archive = _build_source_row_temporal_authority_record(
        source_rowset_node,
        source_acquisition_frame_index_node,
        source_row_identity=identity,
        acquisition_frame=acquisition_frame,
    )
    rowset_path = _persisted_node_path(source_rowset_node, label="source_rowset")
    if (
        authority.archive_identity != archive
        or authority.record_ref != f"/{rowset_path}@run_manifest"
        or identity.record_ref != authority.record_ref
        or identity.record_sha256 != authority.record_sha256
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "manifest_temporal_authority_mismatch",
                    "$.manifest_authority",
                    "Manifest-bound temporal evidence does not own the exact source rowset and identity.",
                ),
            )
        )
    return BoundSourceRowTemporalAuthority(
        record=record,
        record_ref=authority.record_ref,
        record_sha256=authority.record_sha256,
        source_row_identity=identity,
        acquisition_frame=acquisition_frame,
        _archive_identity=archive,
        _source_rowset_node=source_rowset_node,
        _source_frame_index_node=source_acquisition_frame_index_node,
        _manifest_authority=authority,
        _verification_seal=_BOUND_SOURCE_ROW_TEMPORAL_AUTHORITY_SEAL,
    )


def stamp_source_row_temporal_authority(
    source_rowset_node: Any,
    source_acquisition_frame_index_node: Any,
    *,
    source_row_identity: BoundRowIdentityContract,
    acquisition_frame: Any,
) -> BoundSourceRowTemporalAuthority:
    """Persist and reload the exact immediate-source temporal authority."""

    record, _ = _build_source_row_temporal_authority_record(
        source_rowset_node,
        source_acquisition_frame_index_node,
        source_row_identity=source_row_identity,
        acquisition_frame=acquisition_frame,
    )
    attrs = _mutable_attrs(source_rowset_node, label="source_rowset")
    snapshot = copy.deepcopy(dict(attrs))
    expected = copy.deepcopy(snapshot)
    expected.update(
        {
            SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR: record.to_dict(),
            SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR: record.digest(),
        }
    )
    try:
        attrs.update(copy.deepcopy(expected))
        if not _raw_canonical_equal(dict(attrs), expected):
            raise RowIdentityContractError(
                (
                    _issue(
                        "source_temporal_authority_stamp_mismatch",
                        "source_rowset.attrs",
                        "Reloaded attrs differ from the intended source temporal-authority write.",
                    ),
                )
            )
        result = load_bound_source_row_temporal_authority(
            source_rowset_node,
            source_acquisition_frame_index_node,
            source_row_identity=source_row_identity,
            acquisition_frame=acquisition_frame,
        )
        if not _raw_canonical_equal(dict(attrs), expected):
            raise RowIdentityContractError(
                (
                    _issue(
                        "source_temporal_authority_stamp_mismatch",
                        "source_rowset.attrs",
                        "Attrs changed during source temporal-authority verification.",
                    ),
                )
            )
        return result
    except Exception as exc:
        try:
            _restore_attrs(attrs, snapshot)
        except Exception as rollback_exc:
            raise RowIdentityContractError(
                (
                    _issue(
                        "source_temporal_authority_rollback_failed",
                        "source_rowset.attrs",
                        f"Source temporal-authority stamp failed ({exc}); rollback failed ({rollback_exc}).",
                    ),
                )
            ) from exc
        if isinstance(exc, RowIdentityContractError):
            raise
        raise RowIdentityContractError(
            (_issue("source_temporal_authority_stamp_failed", "$", str(exc)),)
        ) from exc


def require_bound_source_row_temporal_authority(
    value: Any,
) -> BoundSourceRowTemporalAuthority:
    if (
        type(value) is not BoundSourceRowTemporalAuthority
        or value._verification_seal is not _BOUND_SOURCE_ROW_TEMPORAL_AUTHORITY_SEAL
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "source_temporal_authority_unverified",
                    "$",
                    "A sealed exact immediate-source temporal authority is required.",
                ),
            )
        )
    def verify() -> None:
        def reload() -> BoundSourceRowTemporalAuthority:
            if value._manifest_authority is None:
                return load_bound_source_row_temporal_authority(
                    value._source_rowset_node,
                    value._source_frame_index_node,
                    source_row_identity=value.source_row_identity,
                    acquisition_frame=value.acquisition_frame,
                )
            return _bind_manifest_source_row_temporal_authority(
                value._source_rowset_node,
                value._source_frame_index_node,
                source_row_identity=value.source_row_identity,
                acquisition_frame=value.acquisition_frame,
                manifest_authority=value._manifest_authority,
            )

        if value._array_digest_evidence is None:
            current = reload()
        else:
            with row_authority_array_digest_evidence_scope(
                value._source_rowset_node,
                value._array_digest_evidence,
            ):
                current = reload()
        if (
            current.record != value.record
            or current.record_ref != value.record_ref
            or current.record_sha256 != value.record_sha256
            or current.archive_identity != value.archive_identity
        ):
            raise RowIdentityContractError(
                (
                    _issue(
                        "source_temporal_authority_stale",
                        value.record_ref,
                        "Immediate-source temporal authority changed after binding.",
                    ),
                )
            )

    verify_persisted_proof(
        (
            "palette.bound_source_row_temporal_authority.v1",
            value.archive_identity.kind,
            value.archive_identity.key,
            value.record_ref,
            value.record_sha256,
            value.source_row_identity.record_ref,
            value.source_row_identity.record_sha256,
            value.acquisition_frame.record_ref,
            value.acquisition_frame.record_sha256,
        ),
        verify,
    )
    return value


def resolve_source_acquisition_frame_indices(
    source_temporal_authority: BoundSourceRowTemporalAuthority,
    source_row_indices: Any,
) -> np.ndarray:
    """Resolve output acquisition frames from exact immediate-source rows."""

    authority = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    indices = np.asarray(source_row_indices)
    if indices.dtype != np.dtype("<i8") or indices.ndim != 1:
        raise RowIdentityContractError(
            (
                _issue(
                    "source_row_index_dtype_invalid",
                    "source_row_index",
                    "source_row_index must be an exact signed int64 rank-1 array.",
                ),
            )
        )
    if np.any(indices < 0) or np.any(
        indices >= authority.record.source_leading_dimension
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "source_row_index_out_of_range",
                    "source_row_index",
                    "Every source_row_index must resolve inside the exact immediate source rowset.",
                ),
            )
        )
    source_frames = _require_live_array_record(
        authority._source_frame_index_node,
        authority.record.source_acquisition_frame_index,
        label="source_acquisition_frame_index",
    )
    return np.asarray(source_frames[indices], dtype=np.int64)


def derive_track_source_instance_values(
    source_temporal_authority: BoundSourceRowTemporalAuthority,
    source_row_indices: Any,
) -> np.ndarray:
    """Mechanically derive nullable observation lineage for track output rows.

    Callers never choose instance-key values.  Observation sources resolve the
    exact source key at ``source_row_index``; all other identity domains emit
    the canonical null encoding ``valid=false, instance_key=0``.
    """

    authority = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    indices = np.asarray(source_row_indices)
    # Reuse the exact range/dtype validation, even though only its validation
    # side effect is needed here.
    resolve_source_acquisition_frame_indices(authority, indices)
    result = np.zeros(indices.shape, dtype=TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE)
    if authority.record.source_identity_domain != OBSERVATION_INSTANCE_DOMAIN:
        return result
    observation_record = authority.record.observation_instance_key
    if observation_record is None:
        raise RowIdentityContractError(
            (
                _issue(
                    "source_observation_identity_missing",
                    "observation_instance_key",
                    "Observation source temporal authority lacks its exact instance-key binding.",
                ),
            )
        )
    source_keys = _require_live_array_record(
        authority.source_row_identity._key_array_node,
        observation_record,
        label="observation_instance_key",
    )
    result["valid"] = True
    result["instance_key"] = source_keys[indices]
    return result


def _build_track_sample_time_lineage_record(
    rowset_node: Any,
    key_array_node: Any,
    source_row_index_node: Any,
    source_frame_index_node: Any,
    interpolation_node: Any,
    source_instance_key_node: Any,
    *,
    source_temporal_authority: BoundSourceRowTemporalAuthority,
) -> tuple[TrackSampleTimeLineageRecord, ArchiveIdentity]:
    authority = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    rowset_path = _persisted_node_path(rowset_node, label="rowset")
    _require_key_array_node_ref(
        rowset_node,
        key_array_node,
        expected_ref=TRACK_SAMPLE_KEY_ARRAY_REF,
    )
    try:
        key_values = np.array(key_array_node[:], copy=True, order="C")
    except Exception as exc:
        raise RowIdentityContractError(
            (_issue("row_identity_key_unreadable", "key_array", str(exc)),)
        ) from exc
    if (
        key_values.dtype != np.dtype("<i8")
        or key_values.ndim != 2
        or key_values.shape[1] != 2
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "track_identity_profile_mismatch",
                    "key_array",
                    "Time lineage requires exact int64 track_sample_key shape (N,2).",
                ),
            )
        )
    leading = int(key_values.shape[0])
    source_row_indices, source_row_record = _time_array_record(
        source_row_index_node,
        expected_path=f"{rowset_path}/{TRACK_SAMPLE_SOURCE_ROW_INDEX_REF}",
        expected_dtype=np.dtype("<i8"),
        expected_shape=(leading,),
        label="source_row_index",
    )
    if np.any(source_row_indices < 0) or np.any(
        source_row_indices >= authority.record.source_leading_dimension
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "source_row_index_out_of_range",
                    "source_row_index",
                    "Every track row must resolve inside the exact immediate source rowset.",
                ),
            )
        )
    if not _rows_unique(source_row_indices):
        raise RowIdentityContractError(
            (
                _issue(
                    "source_row_index_not_unique",
                    "source_row_index",
                    "Canonical track v1 is an exact subset/reorder; source_row_index must be unique.",
                ),
            )
        )
    source_values, source_record = _time_array_record(
        source_frame_index_node,
        expected_path=f"{rowset_path}/{TRACK_SAMPLE_SOURCE_FRAME_INDEX_REF}",
        expected_dtype=np.dtype("<i8"),
        expected_shape=(leading,),
        label="source_frame_index",
    )
    interpolation_values, interpolation_record = _time_array_record(
        interpolation_node,
        expected_path=f"{rowset_path}/{TRACK_SAMPLE_INTERPOLATION_REF}",
        expected_dtype=TRACK_SAMPLE_INTERPOLATION_DTYPE,
        expected_shape=(leading,),
        label="interpolation",
    )
    source_instance_values, source_instance_record = _time_array_record(
        source_instance_key_node,
        expected_path=f"{rowset_path}/{TRACK_SAMPLE_SOURCE_INSTANCE_KEY_REF}",
        expected_dtype=TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
        expected_shape=(leading,),
        label="source_instance_key",
    )
    expected_frames = resolve_source_acquisition_frame_indices(
        authority,
        source_row_indices,
    )
    if not np.array_equal(source_values, expected_frames) or not np.array_equal(
        key_values[:, 1],
        expected_frames,
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "track_source_row_frame_mismatch",
                    "track_sample_key[:,1]",
                    "Output acquisition frame must equal both source_frame[source_row_index] and track_sample_key[:,1].",
                ),
            )
        )
    left = interpolation_values["left_source_frame_index"]
    right = interpolation_values["right_source_frame_index"]
    weight = interpolation_values["right_weight"]
    if (
        not np.array_equal(left, expected_frames)
        or not np.array_equal(right, expected_frames)
        or not np.array_equal(weight, np.zeros((leading,), dtype=np.float64))
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "track_interpolation_unsupported_v1",
                    "source_frame_interpolation",
                    "Canonical track temporal lineage v1 permits exact source rows only: left=right=target and weight=0.",
                ),
            )
        )
    expected_instances = derive_track_source_instance_values(
        authority,
        source_row_indices,
    )
    if not np.array_equal(source_instance_values, expected_instances):
        raise RowIdentityContractError(
            (
                _issue(
                    "track_source_instance_not_derived",
                    "source_instance_key",
                    "source_instance_key must be mechanically derived from the immediate source observation identity, or canonical null for non-observation sources.",
                ),
            )
        )
    try:
        identity = require_same_archive(
            rowset_node,
            key_array_node,
            source_row_index_node,
            source_frame_index_node,
            interpolation_node,
            source_instance_key_node,
            authority._source_rowset_node,
            authority._source_frame_index_node,
            authority.source_row_identity._key_array_node,
        )
    except ArchiveIdentityError as exc:
        raise RowIdentityContractError(
            (_issue("row_identity_archive_mismatch", "$", str(exc)),)
        ) from exc
    if identity != authority.archive_identity:
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_archive_mismatch",
                    "$",
                    "Track output and immediate-source authority span archives.",
                ),
            )
        )
    records_and_nodes = (
        (source_row_record, source_row_index_node, "source_row_index"),
        (source_record, source_frame_index_node, "source_frame_index"),
        (interpolation_record, interpolation_node, "interpolation"),
        (source_instance_record, source_instance_key_node, "source_instance_key"),
    )
    for record, node, label in records_and_nodes:
        _require_live_array_record(node, record, label=label)
    key_after = np.array(key_array_node[:], copy=True, order="C")
    if (
        key_after.dtype != key_values.dtype
        or key_after.shape != key_values.shape
        or not np.array_equal(key_after, key_values)
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "temporal_array_changed_during_binding",
                    "track_sample_key",
                    "Track key changed during temporal-lineage binding.",
                ),
            )
        )
    record = TrackSampleTimeLineageRecord(
        source_row_temporal_authority=TrackSampleTimeLineageRef(
            authority.record_ref,
            authority.record_sha256,
        ),
        recording_id=authority.record.recording_id,
        camera_id=authority.record.camera_id,
        source_total_frames=authority.record.source_total_frames,
        source_rowset_ref=authority.record.source_rowset_ref,
        source_identity_domain=authority.record.source_identity_domain,
        source_identity_mode=authority.record.source_identity_mode,
        source_leading_dimension=authority.record.source_leading_dimension,
        leading_dimension=leading,
        track_sample_key_content_sha256=identity_array_content_sha256(key_values),
        source_row_index=source_row_record,
        source_frame_index=source_record,
        interpolation=interpolation_record,
        source_instance_key=source_instance_record,
    )
    return record, identity


def load_bound_track_sample_time_lineage(
    rowset_node: Any,
    key_array_node: Any,
    source_row_index_node: Any,
    source_frame_index_node: Any,
    interpolation_node: Any,
    source_instance_key_node: Any,
    *,
    source_temporal_authority: BoundSourceRowTemporalAuthority,
) -> BoundTrackSampleTimeLineage:
    """Fresh-process loader for exact track-to-immediate-source lineage."""

    attrs = getattr(rowset_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise RowIdentityContractError(
            (_issue("row_identity_attrs_unavailable", "rowset", "Rowset attrs are required."),)
        )
    record, identity = _build_track_sample_time_lineage_record(
        rowset_node,
        key_array_node,
        source_row_index_node,
        source_frame_index_node,
        interpolation_node,
        source_instance_key_node,
        source_temporal_authority=source_temporal_authority,
    )
    raw = attrs.get(TRACK_SAMPLE_TIME_LINEAGE_ATTR)
    if not _raw_canonical_equal(raw, record.to_dict()):
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_noncanonical",
                    f"@{TRACK_SAMPLE_TIME_LINEAGE_ATTR}",
                    "Persisted time lineage differs from exact live immediate-source authorities.",
                ),
            )
        )
    if attrs.get(TRACK_SAMPLE_TIME_LINEAGE_DIGEST_ATTR) != record.digest():
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_digest_mismatch",
                    f"@{TRACK_SAMPLE_TIME_LINEAGE_DIGEST_ATTR}",
                    "Persisted time-lineage digest is stale.",
                ),
            )
        )
    rowset_path = _persisted_node_path(rowset_node, label="rowset")
    return BoundTrackSampleTimeLineage(
        record=record,
        record_ref=f"/{rowset_path}@{TRACK_SAMPLE_TIME_LINEAGE_ATTR}",
        record_sha256=record.digest(),
        _archive_identity=identity,
        _rowset_node=rowset_node,
        _key_array_node=key_array_node,
        _source_row_index_node=source_row_index_node,
        _source_frame_index_node=source_frame_index_node,
        _interpolation_node=interpolation_node,
        _source_instance_key_node=source_instance_key_node,
        _source_temporal_authority=source_temporal_authority,
        _verification_seal=_BOUND_TRACK_TIME_LINEAGE_SEAL,
    )


def stamp_track_sample_time_lineage(
    rowset_node: Any,
    key_array_node: Any,
    source_row_index_node: Any,
    source_frame_index_node: Any,
    interpolation_node: Any,
    source_instance_key_node: Any,
    *,
    source_temporal_authority: BoundSourceRowTemporalAuthority,
) -> BoundTrackSampleTimeLineage:
    record, _ = _build_track_sample_time_lineage_record(
        rowset_node,
        key_array_node,
        source_row_index_node,
        source_frame_index_node,
        interpolation_node,
        source_instance_key_node,
        source_temporal_authority=source_temporal_authority,
    )
    attrs = _mutable_attrs(rowset_node, label="rowset")
    snapshot = copy.deepcopy(dict(attrs))
    expected = copy.deepcopy(snapshot)
    expected.update(
        {
            TRACK_SAMPLE_TIME_LINEAGE_ATTR: record.to_dict(),
            TRACK_SAMPLE_TIME_LINEAGE_DIGEST_ATTR: record.digest(),
        }
    )
    try:
        attrs.update(copy.deepcopy(expected))
        if not _raw_canonical_equal(dict(attrs), expected):
            raise RowIdentityContractError(
                (
                    _issue(
                        "track_time_lineage_stamp_mismatch",
                        "rowset.attrs",
                        "Reloaded attrs differ from the exact intended time-lineage write.",
                    ),
                )
            )
        result = load_bound_track_sample_time_lineage(
            rowset_node,
            key_array_node,
            source_row_index_node,
            source_frame_index_node,
            interpolation_node,
            source_instance_key_node,
            source_temporal_authority=source_temporal_authority,
        )
        if not _raw_canonical_equal(dict(attrs), expected):
            raise RowIdentityContractError(
                (
                    _issue(
                        "track_time_lineage_stamp_mismatch",
                        "rowset.attrs",
                        "Attrs changed during time-lineage verification.",
                    ),
                )
            )
        return result
    except Exception as exc:
        try:
            _restore_attrs(attrs, snapshot)
        except Exception as rollback_exc:
            raise RowIdentityContractError(
                (
                    _issue(
                        "track_time_lineage_rollback_failed",
                        "rowset.attrs",
                        f"Time-lineage stamp failed ({exc}); rollback failed ({rollback_exc}).",
                    ),
                )
            ) from exc
        if isinstance(exc, RowIdentityContractError):
            raise
        raise RowIdentityContractError(
            (_issue("track_time_lineage_stamp_failed", "$", str(exc)),)
        ) from exc


def require_bound_track_sample_time_lineage(
    value: Any,
) -> BoundTrackSampleTimeLineage:
    if (
        type(value) is not BoundTrackSampleTimeLineage
        or value._verification_seal is not _BOUND_TRACK_TIME_LINEAGE_SEAL
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_unverified",
                    "$",
                    "A sealed exact track-sample time-lineage binding is required.",
                ),
            )
        )
    def verify() -> None:
        current = load_bound_track_sample_time_lineage(
            value._rowset_node,
            value._key_array_node,
            value._source_row_index_node,
            value._source_frame_index_node,
            value._interpolation_node,
            value._source_instance_key_node,
            source_temporal_authority=value._source_temporal_authority,
        )
        if (
            current.record != value.record
            or current.record_ref != value.record_ref
            or current.record_sha256 != value.record_sha256
            or current.archive_identity != value.archive_identity
        ):
            raise RowIdentityContractError(
                (
                    _issue(
                        "track_time_lineage_stale",
                        value.record_ref,
                        "Track-sample immediate-source lineage changed after binding.",
                    ),
                )
            )

    verify_persisted_proof(
        (
            "palette.bound_track_sample_time_lineage.v1",
            value.archive_identity.kind,
            value.archive_identity.key,
            value.record_ref,
            value.record_sha256,
            value._source_temporal_authority.record_ref,
            value._source_temporal_authority.record_sha256,
        ),
        verify,
    )
    return value


def _restore_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        if name not in snapshot:
            del attrs[name]
    attrs.update(copy.deepcopy(dict(snapshot)))
    if not _raw_canonical_equal(dict(attrs), dict(snapshot)):
        raise RuntimeError("restored attrs differ from their pre-call snapshot")


def _restore_key_values(node: Any, snapshot: np.ndarray) -> None:
    try:
        current = np.array(node[:], copy=True, order="C")
    except Exception as exc:
        raise RuntimeError(f"unable to reload key values during rollback: {exc}") from exc
    if (
        current.dtype == snapshot.dtype
        and current.shape == snapshot.shape
        and np.array_equal(current, snapshot)
    ):
        return
    try:
        node[:] = snapshot
        restored = np.array(node[:], copy=True, order="C")
    except Exception as exc:
        raise RuntimeError(f"unable to restore key values: {exc}") from exc
    if (
        restored.dtype != snapshot.dtype
        or restored.shape != snapshot.shape
        or not np.array_equal(restored, snapshot)
    ):
        raise RuntimeError("restored key values differ from their pre-call snapshot")


def _stamp_row_identity_transaction(
    rowset_node: Any,
    key_array_node: Any,
    *,
    contract: Any,
    verify_values: bool,
    return_bound: bool,
    track_time_lineage: BoundTrackSampleTimeLineage | None,
) -> RowIdentityContract | BoundRowIdentityContract:
    """Write, reload, and optionally bind within one rollback boundary."""

    parsed = parse_row_identity_contract(contract)
    rowset_attrs = _mutable_attrs(rowset_node, label="rowset")
    key_attrs = _mutable_attrs(key_array_node, label="key_array")
    _require_key_array_node_ref(
        rowset_node,
        key_array_node,
        expected_ref=parsed.key_array.ref,
    )
    if parsed.domain == TRACK_SAMPLE_DOMAIN:
        lineage = require_bound_track_sample_time_lineage(track_time_lineage)
        if (
            lineage._rowset_node is not rowset_node
            or lineage._key_array_node is not key_array_node
            or parsed.time_lineage
            != TrackSampleTimeLineageRef(lineage.record_ref, lineage.record_sha256)
            or parsed.key_array.content_sha256
            != lineage.record.track_sample_key_content_sha256
        ):
            raise RowIdentityContractError(
                (
                    _issue(
                        "track_time_lineage_mismatch",
                        "track_time_lineage",
                        "Track contract, key node, and acquisition-time lineage must be exact matches.",
                    ),
                )
            )
    elif track_time_lineage is not None:
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_unexpected",
                    "track_time_lineage",
                    "Only track-sample identity accepts time lineage.",
                ),
            )
        )
    if verify_values is not True:
        raise RowIdentityContractError(
            (
                _issue(
                    "row_identity_value_verification_required",
                    "verify_values",
                    "Canonical identity publication must verify exact key values; "
                    "metadata-only inspection is audit evidence, not publication validation.",
                ),
            )
        )
    try:
        values = np.array(key_array_node[:], copy=True, order="C")
    except Exception as exc:
        raise RowIdentityContractError(
            (_issue("row_identity_key_unreadable", "key_array", f"Unable to read key values: {exc}."),)
        ) from exc
    issues = validate_row_identity_values(parsed, values)
    if issues:
        raise RowIdentityContractError(issues)

    rowset_snapshot = copy.deepcopy(dict(rowset_attrs))
    key_snapshot = copy.deepcopy(dict(key_attrs))
    expected_rowset = copy.deepcopy(rowset_snapshot)
    expected_rowset.update(row_identity_contract_attrs(parsed))
    expected_key = copy.deepcopy(key_snapshot)
    expected_key.update(row_identity_key_attrs(parsed))
    try:
        rowset_attrs.update(row_identity_contract_attrs(parsed))
        key_attrs.update(row_identity_key_attrs(parsed))
        if not _raw_canonical_equal(dict(rowset_attrs), expected_rowset):
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_stamp_reload_mismatch",
                        "rowset.attrs",
                        "Reloaded rowset attrs differ from the exact intended write.",
                    ),
                )
            )
        if not _raw_canonical_equal(dict(key_attrs), expected_key):
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_stamp_reload_mismatch",
                        "key_array.attrs",
                        "Reloaded key attrs differ from the exact intended write.",
                    ),
                )
            )
        reloaded = validate_stamped_row_identity(
            rowset_node,
            key_array_node,
            verify_values=True,
            track_time_lineage=track_time_lineage,
        )
        if reloaded != parsed:
            raise RowIdentityContractError(
                (
                    _issue(
                        "row_identity_stamp_reload_mismatch",
                        "$",
                        "Reloaded row identity differs from the requested contract.",
                    ),
                )
            )
        result: RowIdentityContract | BoundRowIdentityContract
        if return_bound:
            result = load_bound_row_identity_contract(
                rowset_node,
                key_array_node,
                track_time_lineage=track_time_lineage,
            )
            require_bound_row_identity_contract(result)
        else:
            result = reloaded
    except Exception as exc:
        failures: list[str] = []
        for label, attrs, snapshot in (
            ("rowset", rowset_attrs, rowset_snapshot),
            ("key_array", key_attrs, key_snapshot),
        ):
            try:
                _restore_attrs(attrs, snapshot)
            except Exception as rollback_exc:  # pragma: no cover - hostile mapping
                failures.append(f"{label}: {rollback_exc}")
        try:
            _restore_key_values(key_array_node, values)
        except Exception as rollback_exc:  # pragma: no cover - hostile array
            failures.append(f"key_array_values: {rollback_exc}")
        suffix = f"; rollback incomplete: {failures!r}" if failures else ""
        raise RowIdentityContractError(
            (_issue("row_identity_stamp_failed", "$", f"Identity stamping failed: {exc}{suffix}."),)
        ) from exc
    return result


def stamp_row_identity_contract(
    rowset_node: Any,
    key_array_node: Any,
    *,
    contract: Any,
    verify_values: bool = True,
    track_time_lineage: BoundTrackSampleTimeLineage | None = None,
) -> RowIdentityContract:
    """Atomically write and reload the exact row identity and key values."""

    result = _stamp_row_identity_transaction(
        rowset_node,
        key_array_node,
        contract=contract,
        verify_values=verify_values,
        return_bound=False,
        track_time_lineage=track_time_lineage,
    )
    assert type(result) is RowIdentityContract
    return result


def validate_stamped_row_identity(
    rowset_node: Any,
    key_array_node: Any,
    *,
    verify_values: bool = True,
    track_time_lineage: BoundTrackSampleTimeLineage | None = None,
) -> RowIdentityContract:
    """Fail closed unless group, key metadata, and optional content all agree."""

    rowset_attrs = getattr(rowset_node, "attrs", None)
    key_attrs = getattr(key_array_node, "attrs", None)
    if not isinstance(rowset_attrs, Mapping) or not isinstance(key_attrs, Mapping):
        raise RowIdentityContractError(
            (_issue("row_identity_attrs_unavailable", "$", "Both nodes must expose attrs mappings."),)
        )
    contract = load_row_identity_contract_attrs(rowset_attrs)
    _require_key_array_node_ref(
        rowset_node,
        key_array_node,
        expected_ref=contract.key_array.ref,
    )
    load_row_identity_key_attrs(key_attrs, contract=contract)
    if contract.domain == TRACK_SAMPLE_DOMAIN:
        lineage = require_bound_track_sample_time_lineage(track_time_lineage)
        if (
            lineage._rowset_node is not rowset_node
            or lineage._key_array_node is not key_array_node
            or contract.time_lineage
            != TrackSampleTimeLineageRef(lineage.record_ref, lineage.record_sha256)
            or contract.key_array.content_sha256
            != lineage.record.track_sample_key_content_sha256
        ):
            raise RowIdentityContractError(
                (
                    _issue(
                        "track_time_lineage_mismatch",
                        "track_time_lineage",
                        "Persisted track identity does not match live acquisition-time lineage.",
                    ),
                )
            )
    elif track_time_lineage is not None:
        raise RowIdentityContractError(
            (
                _issue(
                    "track_time_lineage_unexpected",
                    "track_time_lineage",
                    "Only track-sample identity accepts time lineage.",
                ),
            )
        )
    shape = tuple(int(item) for item in getattr(key_array_node, "shape", ()))
    dtype = np.dtype(getattr(key_array_node, "dtype", "V0")).str
    issues: list[RowIdentityIssue] = []
    if shape != contract.key_array.shape:
        issues.append(_issue("key_shape_mismatch", "key_array.shape", "Persisted shape disagrees with identity metadata."))
    if dtype != contract.key_array.dtype:
        issues.append(_issue("key_dtype_mismatch", "key_array.dtype", "Persisted dtype disagrees with identity metadata."))
    if verify_values is not True:
        issues.append(
            _issue(
                "row_identity_value_verification_required",
                "verify_values",
                "Canonical identity validation must verify exact key values; "
                "metadata-only inspection cannot establish compatibility.",
            )
        )
    else:
        receipt_digest = _receipt_bound_array_content_sha256(key_array_node)
        if receipt_digest is not None:
            if receipt_digest != contract.key_array.content_sha256:
                issues.append(
                    _issue(
                        "key_content_digest_mismatch",
                        "key_array",
                        "Receipt-bound key content differs from the identity contract.",
                    )
                )
        else:
            try:
                values = np.asarray(key_array_node[:])
            except Exception as exc:
                issues.append(
                    _issue(
                        "row_identity_key_unreadable",
                        "key_array",
                        f"Unable to read key values: {exc}.",
                    )
                )
            else:
                issues.extend(validate_row_identity_values(contract, values))
    if issues:
        raise RowIdentityContractError(issues)
    return contract


def load_bound_row_identity_contract(
    rowset_node: Any,
    key_array_node: Any,
    *,
    track_time_lineage: BoundTrackSampleTimeLineage | None = None,
) -> BoundRowIdentityContract:
    """Load and fully verify the exact identity record used by descriptors."""

    contract = validate_stamped_row_identity(
        rowset_node,
        key_array_node,
        verify_values=True,
        track_time_lineage=track_time_lineage,
    )
    rowset_path = _persisted_node_path(rowset_node, label="rowset")
    key_array_path = _persisted_node_path(key_array_node, label="key_array")
    bound = BoundRowIdentityContract(
        contract=contract,
        rowset_path=rowset_path,
        key_array_path=key_array_path,
        record_ref=f"/{rowset_path}@{ROW_IDENTITY_CONTRACT_ATTR}",
        record_sha256=contract.digest(),
        track_time_lineage=track_time_lineage,
        _archive_identity=archive_identity(rowset_node),
        _rowset_node=rowset_node,
        _key_array_node=key_array_node,
        _manifest_authority=None,
        _array_digest_evidence=_capture_active_row_authority_digest_evidence(
            key_array_node
        ),
        _verification_seal=_BOUND_ROW_IDENTITY_SEAL,
    )
    bound.assert_verified()
    return bound


def _bind_manifest_row_identity_contract(
    rowset_node: Any,
    key_array_node: Any,
    *,
    manifest_authority: BoundCoordinateRecord,
) -> BoundRowIdentityContract:
    """Bind observation identity without adding attrs to an immutable rowset."""

    authority = verify_bound_coordinate_record(manifest_authority)
    identity = require_same_archive(rowset_node, key_array_node)
    rowset_path = _persisted_node_path(rowset_node, label="rowset")
    key_array_path = _persisted_node_path(key_array_node, label="key_array")
    _require_key_array_node_ref(
        rowset_node,
        key_array_node,
        expected_ref=INSTANCE_KEY_ARRAY_REF,
    )
    if (
        authority.archive_identity != identity
        or authority.record_ref != f"/{rowset_path}@run_manifest"
    ):
        raise RowIdentityContractError(
            (
                _issue(
                    "manifest_row_identity_authority_mismatch",
                    "$.manifest_authority",
                    "Manifest authority does not own the exact observation rowset.",
                ),
            )
        )
    values = np.array(key_array_node[:], copy=True, order="C")
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    bound = BoundRowIdentityContract(
        contract=contract,
        rowset_path=rowset_path,
        key_array_path=key_array_path,
        record_ref=authority.record_ref,
        record_sha256=authority.record_sha256,
        track_time_lineage=None,
        _archive_identity=identity,
        _rowset_node=rowset_node,
        _key_array_node=key_array_node,
        _manifest_authority=authority,
        _verification_seal=_BOUND_ROW_IDENTITY_SEAL,
    )
    bound.assert_verified()
    return bound


def stamp_and_bind_row_identity_contract(
    rowset_node: Any,
    key_array_node: Any,
    *,
    contract: Any,
    track_time_lineage: BoundTrackSampleTimeLineage | None = None,
) -> BoundRowIdentityContract:
    """Publish an identity contract and return only after exact revalidation."""

    result = _stamp_row_identity_transaction(
        rowset_node,
        key_array_node,
        contract=contract,
        verify_values=True,
        return_bound=True,
        track_time_lineage=track_time_lineage,
    )
    assert type(result) is BoundRowIdentityContract
    return result


__all__ = [
    "INSTANCE_KEY_ARRAY_REF",
    "INSTANCE_KEY_MODE",
    "MANIFEST_ROW_IDENTITY_ATTR",
    "OBSERVATION_INSTANCE_DOMAIN",
    "ROW_IDENTITY_CONTRACT_ATTR",
    "ROW_IDENTITY_CONTRACT_CANONICALIZATION",
    "ROW_IDENTITY_CONTRACT_DIGEST_ATTR",
    "ROW_IDENTITY_CONTRACT_SCHEMA_ID",
    "ROW_IDENTITY_CONTRACT_SCHEMA_VERSION",
    "ROW_IDENTITY_DOMAINS",
    "ROW_IDENTITY_KEY_ATTR",
    "ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION",
    "ROW_IDENTITY_KEY_DIGEST_ATTR",
    "ROW_IDENTITY_KEY_SCHEMA_ID",
    "ROW_IDENTITY_KEY_SCHEMA_VERSION",
    "ROW_IDENTITY_LOCAL_CONTRACT_REF",
    "ROW_IDENTITY_MODES",
    "STIMULUS_STATE_DOMAIN",
    "STIMULUS_STATE_KEY_ARRAY_REF",
    "STIMULUS_STATE_KEY_COMPONENTS",
    "STIMULUS_STATE_KEY_MODE",
    "SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF",
    "SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR",
    "SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR",
    "SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_ID",
    "SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_VERSION",
    "TRACK_SAMPLE_KEY_MODE",
    "TRACK_SAMPLE_DOMAIN",
    "TRACK_SAMPLE_KEY_ARRAY_REF",
    "TRACK_SAMPLE_INTERPOLATION_DTYPE",
    "TRACK_SAMPLE_INTERPOLATION_REF",
    "TRACK_SAMPLE_SOURCE_FRAME_INDEX_REF",
    "TRACK_SAMPLE_SOURCE_ROW_INDEX_REF",
    "TRACK_SAMPLE_SOURCE_INSTANCE_KEY_REF",
    "TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE",
    "TRACK_SAMPLE_TIME_LINEAGE_ATTR",
    "TRACK_SAMPLE_TIME_LINEAGE_DIGEST_ATTR",
    "BoundTrackSampleTimeLineage",
    "BoundSourceRowTemporalAuthority",
    "SourceRowTemporalAuthorityRecord",
    "TrackSampleTimeArrayRecord",
    "TrackSampleTimeLineageRecord",
    "TrackSampleTimeLineageRef",
    "BoundRowIdentityContract",
    "RowIdentityContract",
    "RowIdentityContractError",
    "RowIdentityIssue",
    "RowIdentityKeyArray",
    "build_row_identity_contract",
    "build_source_row_temporal_authority_record",
    "build_track_sample_key",
    "derive_track_source_instance_values",
    "identity_array_content_sha256",
    "load_row_identity_contract_attrs",
    "load_row_identity_key_attrs",
    "load_bound_row_identity_contract",
    "load_bound_source_row_temporal_authority",
    "load_bound_track_sample_time_lineage",
    "parse_row_identity_contract",
    "row_authority_array_digest_evidence_scope",
    "row_identity_contract_attrs",
    "row_identity_key_attrs",
    "require_bound_row_identity_contract",
    "require_bound_source_row_temporal_authority",
    "require_bound_track_sample_time_lineage",
    "stamp_row_identity_contract",
    "stamp_and_bind_row_identity_contract",
    "stamp_source_row_temporal_authority",
    "stamp_track_sample_time_lineage",
    "resolve_source_acquisition_frame_indices",
    "validate_row_identity_contract",
    "validate_row_identity_values",
    "validate_stamped_row_identity",
]
