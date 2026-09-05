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

from .core_authority_roster import (
    BoundCoreMotionAndBouts,
    build_core_authority_consumption_receipt,
    validate_core_authority_consumption_receipt,
)

CORE_MOTION_SOURCE_HANDLE_SCHEMA_ID = "palette.core_behavior.motion_track_handle"
CORE_MOTION_SOURCE_HANDLE_SCHEMA_VERSION = 1
_HANDLE_SEAL = object()


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


__all__ = [
    "CORE_MOTION_SOURCE_HANDLE_SCHEMA_ID",
    "CoreMotionSourceHandleError",
    "CoreMotionTrackSourceHandle",
    "bind_core_motion_track_source_handle",
    "require_core_motion_track_source_handle",
]
