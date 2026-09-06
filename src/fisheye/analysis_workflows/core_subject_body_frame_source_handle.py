"""Receipt-bound subject body-frame access for paradigm computations.

The handle is minted from the same validated core roster as the selected
motion track.  It reuses the strict subject-shape publication loader and never
discovers ``latest``, accepts a legacy body-frame run, or infers a body source
from a neighboring artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    SUBJECT_BODY_FRAME_CAPABILITY,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    BoundSubjectShapeCoordinatePublication,
)

from .core_authority_roster import (
    BoundCoreMotionAndBouts,
    bind_subject_body_frame_from_core_roster,
    build_core_authority_consumption_receipt,
    build_subject_body_frame_source_binding,
    validate_core_authority_consumption_receipt,
)

CORE_SUBJECT_BODY_FRAME_HANDLE_SCHEMA_ID = (
    "palette.core_behavior.subject_body_frame_handle"
)
CORE_SUBJECT_BODY_FRAME_HANDLE_SCHEMA_VERSION = 1
_HANDLE_SEAL = object()

_SELECTED_SURFACES = (
    "instance_key",
    "source_acquisition_frame_index",
    "body_frame/origin_xy",
    "body_frame/forward_axis_xy",
    "body_frame/left_axis_xy",
    "body_frame/heading_deg",
    "body_frame/axis_valid",
    "body_frame/failure_reason_bytes",
)


class CoreSubjectBodyFrameSourceHandleError(ValueError):
    """The roster-selected subject body-frame source is not consumable."""


def _fail(message: str) -> None:
    raise CoreSubjectBodyFrameSourceHandleError(message)


def _child(group: Any, path: str) -> Any:
    current = group
    for component in path.split("/"):
        current = current[component]
    return current


def _readonly(values: Any) -> np.ndarray:
    result = np.array(values, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, init=False, eq=False)
class CoreSubjectBodyFrameSourceHandle:
    """One exact roster-selected subject-shape body-frame publication."""

    analysis_zarr_path: Path
    recording_id: str
    run_path: str
    run_name: str
    publication_manifest_sha256: str
    source_binding_sha256: str
    row_identity_sha256: str
    body_frame_record_sha256: str
    core_authority_roster_sha256: str
    row_count: int
    source_sample_rate_hz: float
    selected_surfaces: Mapping[str, Mapping[str, Any]]
    consumption_receipt: Mapping[str, Any]
    _bound: BoundCoreMotionAndBouts = field(repr=False, compare=False)
    _publication: BoundSubjectShapeCoordinatePublication = field(
        repr=False, compare=False
    )
    _run: Any = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _HANDLE_SEAL:
            _fail("Core subject-body handles can only be minted by the resolver.")
        for name, value in values.items():
            if name in {"selected_surfaces", "consumption_receipt"}:
                value = MappingProxyType(dict(value))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_verification_seal", _HANDLE_SEAL)

    @property
    def core_authority_roster(self) -> Mapping[str, Any]:
        return self._bound.roster

    @property
    def frame_indices(self) -> np.ndarray:
        return self.array("source_acquisition_frame_index")

    @property
    def instance_keys(self) -> np.ndarray:
        return self.array("instance_key")

    @property
    def origin_xy(self) -> np.ndarray:
        return self.array("body_frame/origin_xy")

    @property
    def forward_axis_xy(self) -> np.ndarray:
        return self.array("body_frame/forward_axis_xy")

    @property
    def left_axis_xy(self) -> np.ndarray:
        return self.array("body_frame/left_axis_xy")

    @property
    def axis_valid(self) -> np.ndarray:
        return self.array("body_frame/axis_valid")

    def array(self, path: str) -> np.ndarray:
        """Read one strict-publication surface without replaying its digest scan."""

        if type(path) is not str or path not in self.selected_surfaces:
            raise KeyError(f"Core subject-body surface is not selected: {path!r}.")
        try:
            node = _child(self._run, path)
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreSubjectBodyFrameSourceHandleError(
                f"Selected core subject-body surface is absent: {path!r}."
            ) from exc
        record = self.selected_surfaces[path]
        observed_shape = tuple(int(value) for value in node.shape)
        observed_dtype = np.dtype(node.dtype)
        expected_shape = tuple(int(value) for value in record["shape"])
        expected_dtype = np.dtype(record["dtype"])
        if observed_shape != expected_shape or observed_dtype != expected_dtype:
            _fail(f"Selected core subject-body metadata changed: {path!r}.")
        return _readonly(node[:])

    def assert_verified(self) -> None:
        """Revalidate the sealed selection without rehashing scientific arrays."""

        if self._verification_seal is not _HANDLE_SEAL:
            _fail("Core subject-body handle verification seal is absent.")
        validate_core_authority_consumption_receipt(
            self.consumption_receipt,
            roster=self._bound.roster,
        )
        if self._bound.roster_sha256 != self.core_authority_roster_sha256:
            _fail("Core subject-body handle and roster digests differ.")
        current = build_subject_body_frame_source_binding(self._publication)
        expected = self._bound.roster["capability_bindings"][
            SUBJECT_BODY_FRAME_CAPABILITY
        ]["source_binding"]
        if current != expected:
            _fail("Core subject-body handle no longer matches its roster binding.")

    def assert_current(self) -> None:
        """Compatibility spelling for receipt-backed immutable verification."""

        self.assert_verified()


def bind_core_subject_body_frame_source_handle(
    bound: BoundCoreMotionAndBouts,
    *,
    consumer_id: str,
    required_capabilities: Sequence[str],
    track_id: int,
) -> CoreSubjectBodyFrameSourceHandle:
    """Mint the exact body-frame handle selected by an admitted core roster."""

    if type(bound) is not BoundCoreMotionAndBouts:
        raise TypeError("bound must be a resolver-minted BoundCoreMotionAndBouts.")
    if SUBJECT_BODY_FRAME_CAPABILITY not in required_capabilities:
        _fail("A subject-body handle requires the subject-body capability.")
    publication = bind_subject_body_frame_from_core_roster(bound)
    binding = build_subject_body_frame_source_binding(publication)
    receipt = build_core_authority_consumption_receipt(
        bound.roster,
        consumer_id=consumer_id,
        required_capabilities=required_capabilities,
        selected_track_id=track_id,
    )
    run = publication._run
    surfaces: dict[str, Mapping[str, Any]] = {}
    for path in _SELECTED_SURFACES:
        try:
            node = _child(run, path)
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreSubjectBodyFrameSourceHandleError(
                f"Strict subject-shape publication lacks {path!r}."
            ) from exc
        shape = tuple(int(value) for value in node.shape)
        if not shape or shape[0] != int(binding["row_count"]):
            _fail(f"Core subject-body surface has another row axis: {path!r}.")
        surfaces[path] = {
            "shape": list(shape),
            "dtype": np.dtype(node.dtype).str,
        }

    return CoreSubjectBodyFrameSourceHandle(
        _verification_seal=_HANDLE_SEAL,
        analysis_zarr_path=Path(bound.roster["analysis_zarr"]),
        recording_id=str(bound.roster["recording_id"]),
        run_path=str(binding["run_path"]),
        run_name=str(binding["run_name"]),
        publication_manifest_sha256=str(binding["publication_manifest_sha256"]),
        source_binding_sha256=str(binding["payload_sha256"]),
        row_identity_sha256=str(binding["row_identity_sha256"]),
        body_frame_record_sha256=str(binding["body_frame_record_sha256"]),
        core_authority_roster_sha256=bound.roster_sha256,
        row_count=int(binding["row_count"]),
        source_sample_rate_hz=float(binding["source_sample_rate_hz"]),
        selected_surfaces=surfaces,
        consumption_receipt=receipt,
        _bound=bound,
        _publication=publication,
        _run=run,
    )


def require_core_subject_body_frame_source_handle(
    value: object,
) -> CoreSubjectBodyFrameSourceHandle:
    """Reject forged or stale subject-body handle substitutes."""

    if type(value) is not CoreSubjectBodyFrameSourceHandle:
        raise TypeError(
            "A resolver-minted CoreSubjectBodyFrameSourceHandle is required."
        )
    value.assert_verified()
    return value


__all__ = [
    "CORE_SUBJECT_BODY_FRAME_HANDLE_SCHEMA_ID",
    "CoreSubjectBodyFrameSourceHandle",
    "CoreSubjectBodyFrameSourceHandleError",
    "bind_core_subject_body_frame_source_handle",
    "require_core_subject_body_frame_source_handle",
]
