"""Receipt-bound core-motion access for downstream paradigm computations.

The handle is minted only from a validated :class:`BoundCoreMotionAndBouts`.
It exposes the canonical track-kinematics paths selected by the core roster and
never resolves ``latest``, a provider fallback, or an implicit track zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CANONICAL_SWIM_BOUTS_CAPABILITY,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .core_authority_roster import (
    BoundCoreMotionAndBouts,
    build_core_authority_consumption_receipt,
    validate_core_authority_consumption_receipt,
    validate_core_authority_roster,
)

CORE_MOTION_SOURCE_HANDLE_SCHEMA_ID = "palette.core_behavior.motion_track_handle"
CORE_MOTION_SOURCE_HANDLE_SCHEMA_VERSION = 1
CORE_MOTION_DEPENDENCY_SCHEMA_ID = "palette.core_behavior.motion_dependency"
CORE_MOTION_DEPENDENCY_SCHEMA_VERSION = 1
_HANDLE_SEAL = object()
_CORE_MOTION_DEPENDENCY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "analysis_zarr",
        "core_authority_roster_sha256",
        "core_authority_consumption_receipt",
        "motion_run_path",
        "motion_manifest_sha256",
        "motion_source_binding_sha256",
        "track_id",
        "swim_bout_run_path",
        "swim_bout_source_binding_sha256",
        "record_sha256",
    }
)
_CORE_AUTHORITY_CONSUMPTION_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "consumer_id",
        "recording_id",
        "analysis_zarr",
        "core_authority_roster_sha256",
        "required_capabilities",
        "capability_binding_digests",
        "selected_track_id",
        "record_sha256",
    }
)
_CAPABILITY_DIGEST_FIELDS = frozenset(
    {
        "profile_id",
        "source_binding_sha256",
        "projection_contract_sha256",
        "join_authority_sha256",
    }
)


class CoreMotionSourceHandleError(ValueError):
    """The selected core-motion track cannot be consumed exactly."""


def _fail(message: str) -> None:
    raise CoreMotionSourceHandleError(message)


def _child(group: Any, path: str) -> Any:
    current = group
    for component in path.split("/"):
        current = current[component]
    return current


def _readonly(values: Any) -> np.ndarray:
    result = np.array(values, copy=True, order="C")
    result.setflags(write=False)
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _binding_dtype(record: Mapping[str, Any]) -> np.dtype[Any]:
    fields = record.get("dtype_fields")
    if fields is None:
        return np.dtype(record["dtype"])
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        _fail("Selected core-motion structured dtype declaration is invalid.")
    names: list[str] = []
    formats: list[np.dtype[Any]] = []
    offsets: list[int] = []
    for field_record in fields:
        if not isinstance(field_record, Mapping):
            _fail("Selected core-motion structured dtype field is invalid.")
        names.append(str(field_record.get("name")))
        formats.append(np.dtype(field_record.get("dtype")))
        offsets.append(int(field_record.get("offset")))
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(record["itemsize"]),
        }
    )


@dataclass(frozen=True, init=False, eq=False)
class CoreMotionTrackSourceHandle:
    """One exact roster-selected track and its receipt-authorized arrays."""

    analysis_zarr_path: Path
    recording_id: str
    run_path: str
    run_name: str
    scope: str
    source_manifest_sha256: str
    source_binding_sha256: str
    core_authority_roster_sha256: str
    track_id: int
    sample_count: int
    source_sample_rate_hz: float
    selected_surfaces: Mapping[str, Mapping[str, Any]]
    consumption_receipt: Mapping[str, Any]
    _bound: BoundCoreMotionAndBouts = field(repr=False, compare=False)
    _track_group: Any = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _HANDLE_SEAL:
            _fail("Core-motion handles can only be minted by the roster resolver.")
        for name, value in values.items():
            if name in {"selected_surfaces", "consumption_receipt"}:
                value = MappingProxyType(dict(value))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_verification_seal", _HANDLE_SEAL)

    @property
    def core_authority_roster(self) -> Mapping[str, Any]:
        """Return the exact validated roster from which this handle was minted."""

        return self._bound.roster

    @property
    def canonical_bout_source(self) -> Any:
        """Expose the exact roster-selected event source to core consumers."""

        self.assert_verified()
        if CANONICAL_SWIM_BOUTS_CAPABILITY not in set(
            self.consumption_receipt["required_capabilities"]
        ):
            _fail("Core-motion receipt does not authorize canonical swim-bout access.")
        try:
            return self._bound.bouts.bout_sources[self.track_id]
        except KeyError as exc:  # pragma: no cover - resolver closes this invariant
            raise CoreMotionSourceHandleError(
                "Core-motion handle has no matching canonical swim-bout source."
            ) from exc

    @property
    def frame_indices(self) -> np.ndarray:
        return self.array("source_acquisition_frame_index")

    @property
    def positions_mm(self) -> np.ndarray:
        return self.array("positions_mm")

    @property
    def transition_valid(self) -> np.ndarray:
        return self.array("transition_valid")

    def array(self, path: str) -> np.ndarray:
        """Read one roster-authorized surface without replaying its content hash."""

        if type(path) is not str or path not in self.selected_surfaces:
            raise KeyError(f"Core-motion surface is not selected: {path!r}.")
        try:
            node = _child(self._track_group, path)
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreMotionSourceHandleError(
                f"Selected core-motion surface is absent: {path!r}."
            ) from exc
        record = self.selected_surfaces[path]
        observed_shape = tuple(int(value) for value in node.shape)
        observed_dtype = np.dtype(node.dtype)
        expected_shape = tuple(int(value) for value in record["shape"])
        expected_dtype = _binding_dtype(record)
        if observed_shape != expected_shape or observed_dtype != expected_dtype:
            _fail(f"Selected core-motion surface metadata changed: {path!r}.")
        return _readonly(node[:])

    def assert_verified(self) -> None:
        """Revalidate the sealed selection without resolving or hashing a payload."""

        if self._verification_seal is not _HANDLE_SEAL:
            _fail("Core-motion handle verification seal is absent.")
        validate_core_authority_consumption_receipt(
            self.consumption_receipt,
            roster=self._bound.roster,
        )
        if self._bound.roster_sha256 != self.core_authority_roster_sha256:
            _fail("Core-motion handle and bound roster digests differ.")

    def assert_current(self) -> None:
        """Compatibility spelling for receipt-backed immutable verification."""

        self.assert_verified()


def bind_core_motion_track_source_handle(
    bound: BoundCoreMotionAndBouts,
    *,
    consumer_id: str,
    required_capabilities: Sequence[str],
    track_id: int,
) -> CoreMotionTrackSourceHandle:
    """Mint one downstream handle from an already revalidated core roster."""

    if type(bound) is not BoundCoreMotionAndBouts:
        raise TypeError("bound must be a resolver-minted BoundCoreMotionAndBouts.")
    if type(track_id) is not int or track_id < 0:
        _fail("Core-motion track ID must be one non-negative exact integer.")
    records = [
        record
        for record in bound.track.binding["tracks"]
        if record["track_id"] == track_id
    ]
    if len(records) != 1 or track_id not in bound.bout_identities:
        _fail("Core-motion track does not resolve exactly once with a bout authority.")
    record = records[0]
    receipt = build_core_authority_consumption_receipt(
        bound.roster,
        consumer_id=consumer_id,
        required_capabilities=required_capabilities,
        selected_track_id=track_id,
    )
    track_group = bound.track.run_group["tracks"][f"id_{track_id}"]
    return CoreMotionTrackSourceHandle(
        _verification_seal=_HANDLE_SEAL,
        analysis_zarr_path=Path(bound.roster["analysis_zarr"]),
        recording_id=str(bound.roster["recording_id"]),
        run_path=str(bound.track.binding["run_path"]),
        run_name=str(bound.track.binding["run_name"]),
        scope=str(bound.track.binding["scope"]),
        source_manifest_sha256=str(bound.track.binding["source_manifest_sha256"]),
        source_binding_sha256=str(bound.track.binding["payload_sha256"]),
        core_authority_roster_sha256=bound.roster_sha256,
        track_id=track_id,
        sample_count=int(record["sample_count"]),
        source_sample_rate_hz=float(bound.track.binding["source_sample_rate_hz"]),
        selected_surfaces=record["selected_surfaces"],
        consumption_receipt=receipt,
        _bound=bound,
        _track_group=track_group,
    )


def require_core_motion_track_source_handle(
    value: object,
) -> CoreMotionTrackSourceHandle:
    """Reject forged or stale core-motion handle substitutes."""

    if type(value) is not CoreMotionTrackSourceHandle:
        raise TypeError("A resolver-minted CoreMotionTrackSourceHandle is required.")
    value.assert_verified()
    return value


def core_motion_dependency_record(
    value: object,
) -> Mapping[str, Any]:
    """Return one sealed source record shared by paradigm publications."""

    handle = require_core_motion_track_source_handle(value)
    bout = handle.canonical_bout_source
    body = {
        "schema_id": CORE_MOTION_DEPENDENCY_SCHEMA_ID,
        "schema_version": CORE_MOTION_DEPENDENCY_SCHEMA_VERSION,
        "recording_id": handle.recording_id,
        "analysis_zarr": str(handle.analysis_zarr_path),
        "core_authority_roster_sha256": handle.core_authority_roster_sha256,
        "core_authority_consumption_receipt": _plain(handle.consumption_receipt),
        "motion_run_path": handle.run_path,
        "motion_manifest_sha256": handle.source_manifest_sha256,
        "motion_source_binding_sha256": handle.source_binding_sha256,
        "track_id": handle.track_id,
        "swim_bout_run_path": bout.binding["run_path"],
        "swim_bout_source_binding_sha256": bout.binding["payload_sha256"],
    }
    return validate_core_motion_dependency_record(
        {**body, "record_sha256": canonical_json_sha256(body)},
        roster=handle.core_authority_roster,
    )


def validate_core_motion_dependency_record(
    value: object,
    *,
    roster: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate one downstream motion/bout dependency without payload reads."""

    if not isinstance(value, Mapping):
        _fail("Core motion dependency record must be one mapping.")
    record = _plain(value)
    if set(record) != _CORE_MOTION_DEPENDENCY_FIELDS:
        _fail("Core motion dependency field set is not exact.")
    digest = record.get("record_sha256")
    body = {key: item for key, item in record.items() if key != "record_sha256"}
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or canonical_json_sha256(body) != digest
    ):
        _fail("Core motion dependency record digest is stale.")
    if (
        record.get("schema_id") != CORE_MOTION_DEPENDENCY_SCHEMA_ID
        or record.get("schema_version") != CORE_MOTION_DEPENDENCY_SCHEMA_VERSION
    ):
        _fail("Core motion dependency schema is unsupported.")
    for field_name in (
        "recording_id",
        "analysis_zarr",
        "motion_run_path",
        "swim_bout_run_path",
    ):
        if type(record.get(field_name)) is not str or not record[field_name]:
            _fail(f"Core motion dependency {field_name!r} is invalid.")
    for field_name in (
        "core_authority_roster_sha256",
        "motion_manifest_sha256",
        "motion_source_binding_sha256",
        "swim_bout_source_binding_sha256",
    ):
        field_digest = record.get(field_name)
        if (
            type(field_digest) is not str
            or len(field_digest) != 64
            or any(character not in "0123456789abcdef" for character in field_digest)
        ):
            _fail(f"Core motion dependency {field_name!r} is not a digest.")
    track_id = record.get("track_id")
    if type(track_id) is not int or track_id < 0:
        _fail("Core motion dependency track identity is invalid.")
    receipt = record.get("core_authority_consumption_receipt")
    if not isinstance(receipt, Mapping):
        _fail("Core motion dependency consumption receipt is absent.")
    receipt_body = {
        key: item for key, item in receipt.items() if key != "record_sha256"
    }
    required_capabilities = receipt.get("required_capabilities")
    if (
        set(receipt) != _CORE_AUTHORITY_CONSUMPTION_FIELDS
        or type(receipt.get("consumer_id")) is not str
        or not receipt.get("consumer_id")
        or receipt.get("schema_id")
        != "palette.core_behavior.authority_consumption_receipt"
        or receipt.get("schema_version") != 1
        or receipt.get("record_sha256") != canonical_json_sha256(receipt_body)
        or receipt.get("recording_id") != record["recording_id"]
        or receipt.get("analysis_zarr") != record["analysis_zarr"]
        or receipt.get("core_authority_roster_sha256")
        != record["core_authority_roster_sha256"]
        or receipt.get("selected_track_id") != track_id
        or not isinstance(required_capabilities, list)
        or required_capabilities != sorted(set(required_capabilities))
        or not {
            "cross_grain_join_authority",
            "kinematics_samples",
            CANONICAL_SWIM_BOUTS_CAPABILITY,
        }.issubset(set(required_capabilities))
    ):
        _fail("Core motion dependency consumption receipt is stale or incomplete.")
    capability_digests = receipt.get("capability_binding_digests")
    if not isinstance(capability_digests, Mapping) or set(capability_digests) != set(
        required_capabilities
    ):
        _fail("Core motion dependency capability-digest roster is inexact.")
    for capability, raw_binding in capability_digests.items():
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != (
            _CAPABILITY_DIGEST_FIELDS
        ):
            _fail(f"Core motion dependency capability {capability!r} is malformed.")
        if type(raw_binding.get("profile_id")) is not str or not raw_binding.get(
            "profile_id"
        ):
            _fail(f"Core motion dependency capability {capability!r} has no profile.")
        for digest_name in ("source_binding_sha256", "join_authority_sha256"):
            digest_value = raw_binding.get(digest_name)
            if (
                type(digest_value) is not str
                or len(digest_value) != 64
                or any(
                    character not in "0123456789abcdef" for character in digest_value
                )
            ):
                _fail(
                    f"Core motion dependency capability {capability!r} has an "
                    f"invalid {digest_name}."
                )
        projection_digest = raw_binding.get("projection_contract_sha256")
        if capability == "cross_grain_join_authority":
            if projection_digest is not None:
                _fail("Cross-grain join authority cannot name a projection digest.")
        elif (
            type(projection_digest) is not str
            or len(projection_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in projection_digest
            )
        ):
            _fail(
                f"Core motion dependency capability {capability!r} has an "
                "invalid projection digest."
            )
    if roster is not None:
        validated_roster = validate_core_authority_roster(roster)
        validated_receipt = validate_core_authority_consumption_receipt(
            receipt,
            roster=validated_roster,
        )
        capabilities = validated_roster["capability_bindings"]
        motion_binding = capabilities["kinematics_samples"]["source_binding"]
        bout_binding = capabilities[CANONICAL_SWIM_BOUTS_CAPABILITY]["source_binding"]
        if (
            validated_roster["record_sha256"] != record["core_authority_roster_sha256"]
            or validated_roster["recording_id"] != record["recording_id"]
            or validated_roster["analysis_zarr"] != record["analysis_zarr"]
            or motion_binding["run_path"] != record["motion_run_path"]
            or motion_binding["source_manifest_sha256"]
            != record["motion_manifest_sha256"]
            or motion_binding["payload_sha256"]
            != record["motion_source_binding_sha256"]
            or bout_binding["run_path"] != record["swim_bout_run_path"]
            or bout_binding["payload_sha256"]
            != record["swim_bout_source_binding_sha256"]
            or validated_receipt["selected_track_id"] != track_id
        ):
            _fail("Core motion dependency differs from its selected roster.")
    return MappingProxyType(record)


__all__ = [
    "CORE_MOTION_SOURCE_HANDLE_SCHEMA_ID",
    "CORE_MOTION_DEPENDENCY_SCHEMA_ID",
    "CORE_MOTION_DEPENDENCY_SCHEMA_VERSION",
    "CoreMotionSourceHandleError",
    "CoreMotionTrackSourceHandle",
    "bind_core_motion_track_source_handle",
    "core_motion_dependency_record",
    "require_core_motion_track_source_handle",
    "validate_core_motion_dependency_record",
]
