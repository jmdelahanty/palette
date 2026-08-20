"""Read-only, typed access to one validated chaser-relative frame table.

This module is deliberately an in-memory boundary.  It accepts arrays that
have already been obtained from an exact chaser-relative candidate, validates
the published base schema again, and exposes the frame and pair axes without
choosing a provider, resolving a selector, interpolating timestamps, or
opening Zarr.  The flat source order is always acquisition-frame major and
chaser-axis minor.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    ChaserRelativeFrameSourceHandle,
    require_chaser_relative_frame_source_handle,
)
from fisheye.shared.zarr.chaser_relative_frame_schema import (
    CHASER_RELATIVE_FRAME_SCHEMA_V1,
    ChaserRelativeFrameDimensions,
)


_RUN_PREFIX = "analysis/chaser_relative_frame_runs/"
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_FRAME_ARRAY_NAMES = frozenset(
    {
        "acquisition_frame_id",
        "track_sample_id",
        "timestamp_ns",
        "timestamp_valid",
        "timestamp_reason_code",
        "fish_source_row_id",
        "fish_source_row_valid",
        "fish_source_row_reason_code",
        "fish_position_xy_px",
        "fish_position_valid",
        "fish_position_reason_code",
        "fish_identity_code",
        "selection_member",
        "acquisition_frame_delta",
        "timestamp_delta_ns",
        "fish_transition_valid",
        "fish_transition_reason_code",
        "nearest_chaser_identity_code",
        "nearest_chaser_source_row_id",
        "nearest_chaser_distance_px",
        "nearest_chaser_distance_physical",
        "nearest_chaser_valid",
        "nearest_chaser_reason_code",
    }
)

_PAIR_ARRAY_NAMES = frozenset(
    {
        "chaser_source_row_id",
        "chaser_source_row_valid",
        "chaser_source_row_reason_code",
        "chaser_position_xy_px",
        "chaser_position_valid",
        "chaser_position_reason_code",
        "chaser_identity_code",
        "chaser_behavior_role_code",
        "chaser_behavior_role_valid",
        "chaser_behavior_role_reason_code",
        "chaser_occurrence_member",
        "nearest_chaser_member",
        "trial_id",
        "trial_valid",
        "trial_reason_code",
        "active_state_code",
        "active_state_valid",
        "active_state_reason_code",
        "row_valid",
        "row_reason_code",
        "relative_transition_valid",
        "relative_transition_reason_code",
        "relative_vector_px_xy",
        "relative_distance_px",
        "relative_px_valid",
        "relative_px_reason_code",
        "relative_vector_physical_xy",
        "relative_distance_physical",
        "relative_physical_valid",
        "relative_physical_reason_code",
    }
)

_OPTIONAL_PAIR_ARRAYS = (
    ("trial_id", "trial_valid", "trial_reason_code"),
    ("active_state_code", "active_state_valid", "active_state_reason_code"),
)

_REQUIRED_FRAME_EQUALITY = tuple(
    sorted(
        {
            "acquisition_frame_id",
            "track_sample_id",
            "timestamp_ns",
            "timestamp_valid",
            "timestamp_reason_code",
            "fish_source_row_id",
            "fish_source_row_valid",
            "fish_source_row_reason_code",
            "fish_position_xy_px",
            "fish_position_valid",
            "fish_position_reason_code",
            "fish_identity_code",
            "selection_member",
            "acquisition_frame_delta",
            "timestamp_delta_ns",
            "fish_transition_valid",
            "fish_transition_reason_code",
            "nearest_chaser_identity_code",
            "nearest_chaser_source_row_id",
            "nearest_chaser_distance_px",
            "nearest_chaser_distance_physical",
            "nearest_chaser_valid",
            "nearest_chaser_reason_code",
        }
    )
)

_REASON_ARRAY_SUFFIX = "_reason_code"


class ChaserRelativeDistanceViewError(ValueError):
    """Raised when a relative-frame base table cannot be viewed safely."""


def _fail(message: str) -> None:
    raise ChaserRelativeDistanceViewError(message)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _exact_run_path(value: object) -> str:
    path = _text(value, field="source_run_path")
    if not path.startswith(_RUN_PREFIX):
        _fail(
            "source_run_path must be one exact relative path under "
            "analysis/chaser_relative_frame_runs/."
        )
    name = path[len(_RUN_PREFIX) :]
    if (
        not name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        or name.startswith(".")
        or name in {
            "latest",
            "latest_complete",
            "latest_provider",
            "latest_any",
            "selected",
            "selected_run",
            "active",
            "active_run",
            "current",
            "current_run",
            "default",
            "default_run",
        }
        or _RUN_NAME_RE.fullmatch(name) is None
    ):
        _fail("source_run_path must name one concrete run, not a selector or traversal path.")
    return path


def _sha256(value: object, *, field: str) -> str:
    digest = _text(value, field=field)
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return digest


def _registry(value: object, *, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{field} must be one non-empty string registry.")
    result: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or not key or key != key.strip():
            _fail(f"{field} keys must be non-empty strings.")
        if type(item) is not str or not item or item != item.strip():
            _fail(f"{field} values must be non-empty strings.")
        if key in result:
            _fail(f"{field} contains duplicate keys.")
        result[key] = item
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class ChaserRelativeDistanceRegistries:
    """Controlled code registries needed to interpret the typed arrays."""

    fish_identity: Mapping[str, str]
    chaser_identity: Mapping[str, str]
    behavior_role: Mapping[str, str]
    reason: Mapping[str, str]
    active_state: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for name in (
            "fish_identity",
            "chaser_identity",
            "behavior_role",
            "reason",
        ):
            object.__setattr__(
                self,
                name,
                _registry(getattr(self, name), field=f"registries.{name}"),
            )
        if self.active_state is not None:
            object.__setattr__(
                self,
                "active_state",
                _registry(self.active_state, field="registries.active_state"),
            )

    @classmethod
    def from_manifest(
        cls,
        identity_registries: Mapping[str, Mapping[str, str]],
        reason_registry: Mapping[str, str],
    ) -> "ChaserRelativeDistanceRegistries":
        if not isinstance(identity_registries, Mapping):
            _fail("identity_registries must be a mapping.")
        expected = {"fish", "chaser", "behavior_role"}
        if set(identity_registries) - expected - {"active_state"} != set():
            _fail("identity_registries contains an unexpected registry.")
        if not expected.issubset(identity_registries):
            _fail("identity_registries is missing fish, chaser, or behavior_role.")
        return cls(
            fish_identity=identity_registries["fish"],
            chaser_identity=identity_registries["chaser"],
            behavior_role=identity_registries["behavior_role"],
            reason=reason_registry,
            active_state=identity_registries.get("active_state"),
        )


@dataclass(frozen=True, slots=True)
class ChaserRelativeDistanceViewInput:
    """Explicit in-memory inputs for one chaser-relative distance view."""

    recording_id: str
    source_run_path: str
    source_run_digest: str
    n_frames: int
    n_chasers: int
    base_arrays: Mapping[str, Any]
    registries: ChaserRelativeDistanceRegistries


def _copy_arrays(base_arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(base_arrays, Mapping):
        _fail("base_arrays must be a string-keyed mapping.")
    copied: dict[str, np.ndarray] = {}
    for name, value in base_arrays.items():
        if type(name) is not str or not name:
            _fail("base_arrays keys must be non-empty strings.")
        try:
            array = np.array(value, copy=True, order="C")
        except (TypeError, ValueError) as exc:
            _fail(f"base array {name!r} cannot be copied as a typed array: {exc}")
        if array.dtype.hasobject:
            _fail(f"base array {name!r} has an object dtype.")
        array.setflags(write=False)
        copied[name] = array
    return copied


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _frame_chaser(array: np.ndarray, *, n_frames: int, n_chasers: int) -> np.ndarray:
    reshaped = array.reshape((n_frames, n_chasers) + array.shape[1:])
    reshaped.setflags(write=False)
    return reshaped


def _validate_frame_evidence(
    arrays: Mapping[str, np.ndarray], *, n_frames: int, n_chasers: int
) -> None:
    if n_frames == 0:
        return
    for name in _REQUIRED_FRAME_EQUALITY:
        values = _frame_chaser(
            arrays[name], n_frames=n_frames, n_chasers=n_chasers
        )
        reference = values[:, :1, ...]
        for column in range(1, n_chasers):
            if not _array_equal(values[:, column : column + 1, ...], reference):
                _fail(
                    f"frame-level evidence {name!r} differs across chaser rows."
                )
    frame_ids = _frame_chaser(
        arrays["acquisition_frame_id"], n_frames=n_frames, n_chasers=n_chasers
    )[:, 0]
    track_ids = _frame_chaser(
        arrays["track_sample_id"], n_frames=n_frames, n_chasers=n_chasers
    )[:, 0]
    if np.unique(frame_ids).size != n_frames:
        _fail("acquisition_frame_id does not identify unique frame-major groups.")
    if np.unique(track_ids).size != n_frames:
        _fail("track_sample_id does not identify unique frame-major groups.")


def _validate_registries(
    arrays: Mapping[str, np.ndarray],
    registries: ChaserRelativeDistanceRegistries,
    *,
    n_frames: int,
    n_chasers: int,
) -> tuple[str, ...]:
    if len(registries.chaser_identity) != n_chasers:
        _fail("chaser identity registry cardinality does not match n_chasers.")
    if len(set(registries.chaser_identity.values())) != n_chasers:
        _fail("chaser identity registry values must be unique.")

    fish_codes = arrays["fish_identity_code"]
    if fish_codes.size:
        if np.unique(fish_codes).size != 1:
            _fail("fish identity code is not stable across the view.")
        fish_code = str(int(fish_codes[0]))
        if fish_code not in registries.fish_identity:
            _fail(f"fish identity code {fish_code!r} is undeclared.")
    elif not registries.fish_identity:
        _fail("fish identity registry cannot be empty.")

    chaser_codes = _frame_chaser(
        arrays["chaser_identity_code"], n_frames=n_frames, n_chasers=n_chasers
    )
    column_codes: list[int] = []
    for column in range(n_chasers):
        values = chaser_codes[:, column]
        if values.size and np.unique(values).size != 1:
            _fail("chaser identity code is unstable within one chaser column.")
        if values.size:
            code = int(values[0])
        else:
            # With no rows the registry still has to describe the declared
            # axis; the sorted numeric keys are the only unambiguous order.
            try:
                code = sorted(int(key) for key in registries.chaser_identity)[column]
            except (TypeError, ValueError, IndexError) as exc:
                _fail(f"empty view cannot resolve chaser identity columns: {exc}")
        if str(code) not in registries.chaser_identity:
            _fail(f"chaser identity code {code!r} is undeclared.")
        column_codes.append(code)
    if len(set(column_codes)) != n_chasers:
        _fail("chaser identity codes do not identify distinct stable columns.")

    role_codes = arrays["chaser_behavior_role_code"]
    if np.any(role_codes == 0):
        _fail("behavior-role code zero is not a declared behavior role.")
    for code in np.unique(role_codes):
        if str(int(code)) not in registries.behavior_role:
            _fail(f"behavior-role code {int(code)!r} is undeclared.")

    reason_arrays = (
        arrays[name]
        for name in arrays
        if name.endswith(_REASON_ARRAY_SUFFIX)
    )
    for values in reason_arrays:
        for code in np.unique(values):
            if str(int(code)) not in registries.reason:
                _fail(f"reason code {int(code)!r} is undeclared.")

    has_active = any(name in arrays for name in _OPTIONAL_PAIR_ARRAYS[1])
    if has_active:
        if registries.active_state is None:
            _fail("active-state arrays require an active-state registry.")
        active_codes = arrays["active_state_code"]
        for code in np.unique(active_codes):
            if str(int(code)) not in registries.active_state:
                _fail(f"active-state code {int(code)!r} is undeclared.")
    return tuple(registries.chaser_identity[str(code)] for code in column_codes)


def _validate_optional_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for group in _OPTIONAL_PAIR_ARRAYS:
        present = [name in arrays for name in group]
        if any(present) and not all(present):
            _fail(f"optional arrays {group!r} must be supplied together.")


@dataclass(frozen=True, slots=True)
class ChaserRelativeDistanceView:
    """Verified, immutable frame and pair accessors for distance analyses."""

    recording_id: str
    source_run_path: str
    source_run_digest: str
    n_frames: int
    n_chasers: int
    n_rows: int
    chaser_identities: tuple[str, ...]
    base_arrays: Mapping[str, np.ndarray]
    frame_arrays: Mapping[str, np.ndarray]
    pair_arrays: Mapping[str, np.ndarray]
    registries: ChaserRelativeDistanceRegistries

    @classmethod
    def from_input(
        cls, inputs: ChaserRelativeDistanceViewInput
    ) -> "ChaserRelativeDistanceView":
        if not isinstance(inputs, ChaserRelativeDistanceViewInput):
            _fail("inputs must be ChaserRelativeDistanceViewInput.")
        recording_id = _text(inputs.recording_id, field="recording_id")
        source_run_path = _exact_run_path(inputs.source_run_path)
        source_run_digest = _sha256(
            inputs.source_run_digest, field="source_run_digest"
        )
        if type(inputs.n_frames) is not int or inputs.n_frames < 0:
            _fail("n_frames must be a nonnegative exact integer.")
        if type(inputs.n_chasers) is not int or inputs.n_chasers <= 0:
            _fail("n_chasers must be a positive exact integer.")
        n_rows = inputs.n_frames * inputs.n_chasers
        arrays = _copy_arrays(inputs.base_arrays)
        dimensions = ChaserRelativeFrameDimensions(n_rows=n_rows)
        try:
            CHASER_RELATIVE_FRAME_SCHEMA_V1.require(
                arrays, dimensions=dimensions, body_arrays=None
            )
        except (TypeError, ValueError) as exc:
            _fail(f"base arrays do not satisfy the exact chaser-relative schema: {exc}")
        _validate_optional_arrays(arrays)
        expected = _FRAME_ARRAY_NAMES | _PAIR_ARRAY_NAMES
        if set(arrays) != expected and set(arrays) - expected:
            _fail(
                "base arrays contain fields outside the typed distance view: "
                + repr(sorted(set(arrays) - expected))
            )
        if not _FRAME_ARRAY_NAMES.issubset(arrays):
            _fail("base arrays omit required frame-level evidence.")
        if not _PAIR_ARRAY_NAMES - {name for group in _OPTIONAL_PAIR_ARRAYS for name in group} <= set(arrays):
            _fail("base arrays omit required pair-level distance evidence.")
        _validate_frame_evidence(
            arrays, n_frames=inputs.n_frames, n_chasers=inputs.n_chasers
        )
        identities = _validate_registries(
            arrays,
            inputs.registries,
            n_frames=inputs.n_frames,
            n_chasers=inputs.n_chasers,
        )

        frame_arrays = {
            name: _frame_chaser(
                arrays[name], n_frames=inputs.n_frames, n_chasers=inputs.n_chasers
            )[:, 0, ...].copy(order="C")
            for name in _FRAME_ARRAY_NAMES
        }
        pair_arrays = {
            name: _frame_chaser(
                arrays[name], n_frames=inputs.n_frames, n_chasers=inputs.n_chasers
            ).copy(order="C")
            for name in _PAIR_ARRAY_NAMES
            if name in arrays
        }
        for value in (*frame_arrays.values(), *pair_arrays.values()):
            value.setflags(write=False)
        return cls(
            recording_id=recording_id,
            source_run_path=source_run_path,
            source_run_digest=source_run_digest,
            n_frames=inputs.n_frames,
            n_chasers=inputs.n_chasers,
            n_rows=n_rows,
            chaser_identities=identities,
            base_arrays=MappingProxyType(dict(arrays)),
            frame_arrays=MappingProxyType(frame_arrays),
            pair_arrays=MappingProxyType(pair_arrays),
            registries=inputs.registries,
        )

    @classmethod
    def from_base_arrays(
        cls,
        *,
        recording_id: str,
        source_run_path: str,
        source_run_digest: str,
        n_frames: int,
        n_chasers: int,
        base_arrays: Mapping[str, Any],
        registries: ChaserRelativeDistanceRegistries,
    ) -> "ChaserRelativeDistanceView":
        return cls.from_input(
            ChaserRelativeDistanceViewInput(
                recording_id=recording_id,
                source_run_path=source_run_path,
                source_run_digest=source_run_digest,
                n_frames=n_frames,
                n_chasers=n_chasers,
                base_arrays=base_arrays,
                registries=registries,
            )
        )

    @classmethod
    def from_source_handle(
        cls,
        handle: ChaserRelativeFrameSourceHandle,
    ) -> "ChaserRelativeDistanceView":
        """Build a durable consumer view from one current verified handle."""

        source = require_chaser_relative_frame_source_handle(handle)
        reason_registry = source.run_manifest.get("reason_codes")
        if not isinstance(reason_registry, Mapping):
            _fail("verified source handle lacks its reason-code registry.")
        registries = ChaserRelativeDistanceRegistries.from_manifest(
            source.identity_registries,
            reason_registry,
        )
        return cls.from_base_arrays(
            recording_id=source.recording_id,
            source_run_path=source.run_path,
            source_run_digest=source.verification_digest,
            n_frames=source.n_frames,
            n_chasers=source.n_chasers,
            base_arrays=source.base_arrays,
            registries=registries,
        )

    def frame_array(self, name: str) -> np.ndarray:
        try:
            return self.frame_arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown frame-level chaser-distance array {name!r}.") from exc

    def pair_array(self, name: str) -> np.ndarray:
        try:
            return self.pair_arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown pair-level chaser-distance array {name!r}.") from exc

    def base_array(self, name: str) -> np.ndarray:
        try:
            return self.base_arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown base chaser-distance array {name!r}.") from exc


def build_chaser_relative_distance_view(
    inputs: ChaserRelativeDistanceViewInput,
) -> ChaserRelativeDistanceView:
    """Build one fail-closed in-memory chaser-relative distance view."""

    return ChaserRelativeDistanceView.from_input(inputs)


def load_chaser_relative_distance_view(
    handle: ChaserRelativeFrameSourceHandle,
) -> ChaserRelativeDistanceView:
    """Create the durable distance view from one exact publication handle."""

    return ChaserRelativeDistanceView.from_source_handle(handle)


__all__ = [
    "ChaserRelativeDistanceRegistries",
    "ChaserRelativeDistanceView",
    "ChaserRelativeDistanceViewError",
    "ChaserRelativeDistanceViewInput",
    "build_chaser_relative_distance_view",
    "load_chaser_relative_distance_view",
]
