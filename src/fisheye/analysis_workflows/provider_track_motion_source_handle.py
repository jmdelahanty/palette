"""Strict consumer boundary for one Phase 3 provider-motion publication.

The Phase 3 writer publishes immutable, selector-ineligible motion successors
under ``analysis/track_kinematics_runs/provider/<run>``.  This module binds one
caller-supplied concrete run and returns copied read-only arrays.  It never
resolves a selector, chooses a fallback, or changes the archive.

The current writer records the FPS supplied to the numerical computation, but
does not yet bind that value to an immutable temporal-authority record.  The
reader therefore exposes that state explicitly as a compatibility status and
does not permit it to satisfy ``require_authoritative_timing=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis_workflows.materializers import provider_track_motion as writer
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_ID = (
    "palette.provider_track_motion_source_handle"
)
PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_VERSION = 1

_HANDLE_SEAL = object()
_SELECTOR_NAMES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "latest_provider",
        "authoritative_run",
        "authoritative",
        "current",
        "default",
        "fallback",
        "selected",
    }
)


class ProviderTrackMotionSourceHandleError(ValueError):
    """Raised when an exact provider-motion consumer binding is invalid."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _canonical_run_path(value: object) -> tuple[str, str]:
    if type(value) is not str:
        raise ProviderTrackMotionSourceHandleError("run_path must be one exact string.")
    prefix = f"{writer.PROVIDER_TRACK_MOTION_PARENT_PATH}/"
    if (
        not value.startswith(prefix)
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or value != value.strip()
    ):
        raise ProviderTrackMotionSourceHandleError(
            "run_path must name one exact provider/<run> path."
        )
    name = value[len(prefix) :]
    if (
        not name
        or "/" in name
        or name in {".", ".."}
        or name in _SELECTOR_NAMES
        or writer._RUN_NAME_RE.fullmatch(name) is None
    ):
        raise ProviderTrackMotionSourceHandleError(
            "run_path must name one concrete provider run, not a selector, "
            "fallback, nested path, or ambiguous name."
        )
    canonical = f"{writer.PROVIDER_TRACK_MOTION_PARENT_PATH}/{name}"
    if value != canonical:
        raise ProviderTrackMotionSourceHandleError("run_path is not canonical.")
    return canonical, name


def _node(group: Any, path: str) -> Any:
    current = group
    for component in path.split("/"):
        current = current[component]
    return current


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderTrackMotionSourceHandleError(f"{name} must be an object.")
    return value


def _require_digest(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProviderTrackMotionSourceHandleError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _binding(
    payload: Mapping[str, Any],
    name: str,
) -> tuple[Mapping[str, Any], str]:
    value = _require_mapping(payload.get(name), name=f"provider {name} binding")
    if set(value) != {"record", "sha256"}:
        raise ProviderTrackMotionSourceHandleError(
            f"Provider {name} binding has an unexpected field set."
        )
    record = _require_mapping(value.get("record"), name=f"provider {name} record")
    digest = _require_digest(value.get("sha256"), name=f"provider {name} digest")
    if canonical_json_sha256(_thaw(record)) != digest:
        raise ProviderTrackMotionSourceHandleError(
            f"Provider {name} record digest is stale."
        )
    return _freeze(record), digest


def _temporal_authority(
    computation: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None, str, bool]:
    """Expose timing evidence without upgrading it to live authority.

    The current provider-motion writer has no source-clock binding.  A future
    record can be retained for inspection once its own digest is valid, but a
    digest-bound JSON object alone is not proof that the referenced frame clock
    is present and current in this archive.  That later live-source verifier is
    deliberately outside this compatibility reader.
    """

    raw = computation.get("temporal_authority")
    if raw is None:
        parameters = computation.get("parameters")
        fps = parameters.get("fps") if isinstance(parameters, Mapping) else None
        if isinstance(fps, (int, float)) and not isinstance(fps, bool):
            return (
                None,
                None,
                "compatibility_caller_fps_only",
                False,
            )
        return None, None, "missing", False
    binding = _require_mapping(raw, name="provider temporal_authority")
    if set(binding) != {"record", "sha256"}:
        raise ProviderTrackMotionSourceHandleError(
            "Provider temporal_authority binding has an unexpected field set."
        )
    record = _require_mapping(
        binding.get("record"), name="provider temporal_authority record"
    )
    digest = _require_digest(
        binding.get("sha256"), name="provider temporal_authority digest"
    )
    if canonical_json_sha256(_thaw(record)) != digest:
        raise ProviderTrackMotionSourceHandleError(
            "Provider temporal_authority record digest is stale."
        )
    return (
        _freeze(record),
        digest,
        "bound_record_unverified_against_source_clock",
        False,
    )


def _read_array(run: Any, path: str) -> np.ndarray:
    try:
        node = _node(run, path)
    except (KeyError, ValueError, TypeError) as exc:
        raise ProviderTrackMotionSourceHandleError(
            f"Provider-motion array is missing: {path!r}."
        ) from exc
    if not isinstance(node, zarr.Array):
        raise ProviderTrackMotionSourceHandleError(
            f"Provider-motion path is not an array: {path!r}."
        )
    return _readonly(node[:])


def _validate_exact_lengths(arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    row_count = int(arrays["track_sample_key"].shape[0])
    track_count = int(arrays["track_ids"].shape[0])
    offsets = arrays["track_row_offsets"]
    if offsets.shape != (track_count + 1,):
        raise ProviderTrackMotionSourceHandleError(
            "Provider track_row_offsets length does not equal track_count + 1."
        )
    if (
        offsets.size == 0
        or int(offsets[0]) != 0
        or int(offsets[-1]) != row_count
        or np.any(offsets < 0)
        or np.any(offsets > row_count)
        or np.any(np.diff(offsets) < 0)
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider track_row_offsets are outside the exact row domain."
        )
    for path in (*writer._PIXEL_SAMPLE_ARRAYS, *writer._PHYSICAL_SAMPLE_ARRAYS):
        if path in arrays and arrays[path].shape[0] != row_count:
            raise ProviderTrackMotionSourceHandleError(
                f"Provider row array {path!r} is not aligned to track samples."
            )
    second_count = int(arrays["per_second/track_second_key"].shape[0])
    for path in (*writer._PIXEL_PER_SECOND_ARRAYS, *writer._PHYSICAL_PER_SECOND_ARRAYS):
        if path in arrays and arrays[path].shape[0] != second_count:
            raise ProviderTrackMotionSourceHandleError(
                f"Provider per-second array {path!r} is not aligned."
            )
    return row_count, track_count, second_count


def _validate_lineage_and_offsets(arrays: Mapping[str, np.ndarray]) -> None:
    row_count, _track_count, _second_count = _validate_exact_lengths(arrays)
    keys = arrays["track_sample_key"]
    if keys.shape != (row_count, 2) or np.unique(keys, axis=0).shape[0] != row_count:
        raise ProviderTrackMotionSourceHandleError(
            "Provider track_sample_key is not a unique [track, frame] row identity."
        )
    if not np.array_equal(keys[:, 1], arrays["source_acquisition_frame_index"]):
        raise ProviderTrackMotionSourceHandleError(
            "Provider track_sample_key disagrees with acquisition-frame lineage."
        )
    offsets = arrays["track_row_offsets"]
    for index, track_id in enumerate(arrays["track_ids"]):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if not np.all(keys[start:stop, 0] == track_id):
            raise ProviderTrackMotionSourceHandleError(
                "Provider track-row offsets do not delimit their declared tracks."
            )

    for path in (
        "source_provider_row_index",
        "source_position_row_index",
        "source_body_frame_row_index",
        "source_tracking_row_index",
    ):
        values = arrays[path]
        if np.any(values < 0) or not np.array_equal(
            np.sort(values), np.arange(row_count, dtype=values.dtype)
        ):
            raise ProviderTrackMotionSourceHandleError(
                f"Provider {path!r} is not an exact source-row permutation."
            )
    observation_keys = arrays["source_observation_instance_key"]
    if np.unique(observation_keys).shape[0] != row_count:
        raise ProviderTrackMotionSourceHandleError(
            "Provider source observation identities are duplicated."
        )
    if np.any(arrays["source_acquisition_frame_index"] < 0):
        raise ProviderTrackMotionSourceHandleError(
            "Provider acquisition-frame lineage contains a negative frame."
        )


def _validate_independent_validity(arrays: Mapping[str, np.ndarray]) -> None:
    for path in (
        "position_source_valid",
        "body_frame_source_valid",
        "linear_sample_valid",
        "angular_sample_valid",
        "transition_valid",
    ):
        if arrays[path].dtype != np.dtype(bool):
            raise ProviderTrackMotionSourceHandleError(
                f"Provider validity array {path!r} is not exact bool."
            )
    if np.any(arrays["linear_sample_valid"] & ~arrays["position_source_valid"]):
        raise ProviderTrackMotionSourceHandleError(
            "Provider linear validity exceeds position-source validity."
        )
    if np.any(arrays["angular_sample_valid"] & ~arrays["body_frame_source_valid"]):
        raise ProviderTrackMotionSourceHandleError(
            "Provider angular validity exceeds body-frame validity."
        )
    if "sample_valid" in arrays:
        raise ProviderTrackMotionSourceHandleError(
            "Provider motion must not synthesize or publish generic sample_valid."
        )


def _verification_digest(
    *,
    run_path: str,
    manifest_sha256: str,
    arrays: Mapping[str, np.ndarray],
    timing_status: str,
) -> str:
    return canonical_json_sha256(
        {
            "schema_id": PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_ID,
            "schema_version": PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_VERSION,
            "run_path": run_path,
            "manifest_sha256": manifest_sha256,
            "timing_status": timing_status,
            "arrays": {
                path: sha256_array(value) for path, value in sorted(arrays.items())
            },
        }
    )


@dataclass(frozen=True, init=False, eq=False)
class ProviderTrackMotionSourceHandle:
    """Immutable, verified snapshot of one exact provider-motion run."""

    analysis_zarr_path: Path
    run_path: str
    run_name: str
    provider_manifest: Mapping[str, Any] = field(repr=False)
    provider_manifest_sha256: str
    selector_eligible: bool
    source_authority_record: Mapping[str, Any] = field(repr=False)
    source_authority_sha256: str
    tracked_input_record: Mapping[str, Any] = field(repr=False)
    tracked_input_sha256: str
    physical_authority_record: Mapping[str, Any] | None = field(repr=False)
    physical_authority_sha256: str | None
    physical_authority_status: str
    computation_record: Mapping[str, Any] = field(repr=False)
    computation_sha256: str
    temporal_authority_record: Mapping[str, Any] | None = field(repr=False)
    temporal_authority_sha256: str | None
    temporal_authority_status: str
    timing_is_authoritative: bool
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    row_count: int
    track_count: int
    per_second_count: int
    verification_digest: str
    _use_consolidated: bool = field(repr=False, compare=False)
    _require_authoritative_timing: bool = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _HANDLE_SEAL:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion source handles can only be minted by the strict loader."
            )
        for name, value in values.items():
            if name == "arrays":
                value = MappingProxyType(
                    {path: _readonly(array) for path, array in value.items()}
                )
            elif name.endswith("_record") or name == "provider_manifest":
                if value is not None:
                    value = _freeze(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_verification_seal", _HANDLE_SEAL)

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Compatibility alias for the exact provider manifest snapshot."""

        return self.provider_manifest

    @property
    def manifest_sha256(self) -> str:
        return self.provider_manifest_sha256

    @property
    def provider_manifest_digest(self) -> str:
        return self.provider_manifest_sha256

    @property
    def source_path(self) -> Path:
        return self.analysis_zarr_path

    @property
    def source_authority(self) -> Mapping[str, Any]:
        return self.source_authority_record

    @property
    def tracked_input(self) -> Mapping[str, Any]:
        return self.tracked_input_record

    @property
    def computation(self) -> Mapping[str, Any]:
        return self.computation_record

    @property
    def physical_authority(self) -> Mapping[str, Any] | None:
        return self.physical_authority_record

    @property
    def temporal_authority(self) -> Mapping[str, Any] | None:
        return self.temporal_authority_record

    @property
    def track_ids(self) -> np.ndarray:
        return self.arrays["track_ids"]

    @property
    def track_row_offsets(self) -> np.ndarray:
        return self.arrays["track_row_offsets"]

    @property
    def track_sample_key(self) -> np.ndarray:
        return self.arrays["track_sample_key"]

    @property
    def source_acquisition_frame_index(self) -> np.ndarray:
        return self.arrays["source_acquisition_frame_index"]

    @property
    def source_observation_instance_key(self) -> np.ndarray:
        return self.arrays["source_observation_instance_key"]

    @property
    def source_provider_row_index(self) -> np.ndarray:
        return self.arrays["source_provider_row_index"]

    @property
    def source_position_row_index(self) -> np.ndarray:
        return self.arrays["source_position_row_index"]

    @property
    def source_body_frame_row_index(self) -> np.ndarray:
        return self.arrays["source_body_frame_row_index"]

    @property
    def source_tracking_row_index(self) -> np.ndarray:
        return self.arrays["source_tracking_row_index"]

    @property
    def time_seconds(self) -> np.ndarray:
        return self.arrays["time_seconds"]

    @property
    def delta_seconds(self) -> np.ndarray:
        return self.arrays["delta_seconds"]

    @property
    def positions_px(self) -> np.ndarray:
        return self.arrays["positions_px"]

    @property
    def positions_mm(self) -> np.ndarray | None:
        return self.arrays.get("positions_mm")

    @property
    def position_source_valid(self) -> np.ndarray:
        return self.arrays["position_source_valid"]

    @property
    def body_frame_source_valid(self) -> np.ndarray:
        return self.arrays["body_frame_source_valid"]

    @property
    def linear_sample_valid(self) -> np.ndarray:
        return self.arrays["linear_sample_valid"]

    @property
    def angular_sample_valid(self) -> np.ndarray:
        return self.arrays["angular_sample_valid"]

    @property
    def transition_valid(self) -> np.ndarray:
        return self.arrays["transition_valid"]

    @property
    def linear_sample_reason_code(self) -> np.ndarray:
        return self.arrays["linear_sample_reason_code"]

    @property
    def angular_sample_reason_code(self) -> np.ndarray:
        return self.arrays["angular_sample_reason_code"]

    @property
    def transition_reason_code(self) -> np.ndarray:
        return self.arrays["transition_reason_code"]

    def array(self, path: str) -> np.ndarray:
        """Return one copied read-only array snapshot by its exact path."""

        try:
            return self.arrays[path]
        except KeyError as exc:
            raise KeyError(f"Unknown provider-motion array {path!r}.") from exc

    def assert_current(self) -> None:
        """Reopen the same run and reject mutation or stale consolidation."""

        if self._verification_seal is not _HANDLE_SEAL:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion source handle verification seal is absent."
            )
        refreshed = load_provider_track_motion_source_handle(
            self.analysis_zarr_path,
            self.run_path,
            use_consolidated=self._use_consolidated,
            expected_manifest_sha256=self.provider_manifest_sha256,
            require_authoritative_timing=self._require_authoritative_timing,
        )
        if refreshed.verification_digest != self.verification_digest:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion source changed after the handle was sealed."
            )

    def assert_verified(self) -> None:
        self.assert_current()


def _load_once(
    archive: Path,
    run_path: str,
    run_name: str,
    *,
    use_consolidated: bool,
    expected_manifest_sha256: str | None,
    require_authoritative_timing: bool,
) -> ProviderTrackMotionSourceHandle:
    try:
        root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
        parent = root[writer.PROVIDER_TRACK_MOTION_PARENT_PATH]
        run = root[run_path]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ProviderTrackMotionSourceHandleError(
            f"Unable to open exact provider-motion run {run_path!r}: {exc}"
        ) from exc
    if not isinstance(run, zarr.Group):
        raise ProviderTrackMotionSourceHandleError(
            f"Provider-motion run {run_path!r} is not a group."
        )
    selector_attrs = set(writer._SELECTOR_ATTRS).intersection(parent.attrs)
    if selector_attrs:
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion namespace contains forbidden selector attributes: "
            f"{sorted(selector_attrs)!r}."
        )
    attrs = run.attrs
    if (
        attrs.get("schema_id") != writer.PROVIDER_TRACK_MOTION_SCHEMA_ID
        or attrs.get("schema_version") != writer.PROVIDER_TRACK_MOTION_SCHEMA_VERSION
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion run schema identity is invalid."
        )
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion run does not satisfy the complete selector-ineligible lifecycle."
        )
    raw_manifest = attrs.get(writer.PROVIDER_TRACK_MOTION_MANIFEST_ATTR)
    if not isinstance(raw_manifest, Mapping):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion manifest is missing."
        )
    try:
        writer.validate_provider_track_motion_run(
            archive,
            run_path,
            use_consolidated=use_consolidated,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        payload, receipt = writer._validate_manifest(
            raw_manifest,
            expected_run_name=run_name,
            expected_status=RUN_STATUS_COMPLETE,
        )
        manifest_sha256 = writer.provider_track_motion_manifest_digest(raw_manifest)
    except (writer.ProviderTrackMotionError, KeyError, TypeError, ValueError) as exc:
        raise ProviderTrackMotionSourceHandleError(str(exc)) from exc
    if attrs.get(writer.PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR) != manifest_sha256:
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion manifest digest attribute is stale."
        )
    if (
        attrs.get(writer.PROVIDER_TRACK_MOTION_STORAGE_PLAN_ATTR)
        != payload["physical_storage_plan"]
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion storage-plan attribute differs from its manifest."
        )
    publication = payload["publication"]
    if (
        attrs.get(writer.PROVIDER_TRACK_MOTION_PUBLICATION_ATTEMPT_ATTR)
        != publication["publication_attempt_uuid"]
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion publication attempt differs from its manifest."
        )
    source_record, source_sha256 = _binding(payload, "source_authority")
    tracked_record, tracked_sha256 = _binding(payload, "tracked_input")
    computation_record, computation_sha256 = _binding(payload, "computation")
    physical_binding = _require_mapping(
        payload["physical_authority"], name="provider physical_authority"
    )
    physical_status = physical_binding["status"]
    if physical_status == "bound":
        # The physical binding is intentionally shaped differently from the
        # ordinary record bindings: status, record, and sha256 are siblings.
        physical_value = _require_mapping(
            physical_binding.get("record"), name="provider physical authority record"
        )
        physical_sha256 = _require_digest(
            physical_binding.get("sha256"), name="provider physical authority digest"
        )
        if canonical_json_sha256(physical_value) != physical_sha256:
            raise ProviderTrackMotionSourceHandleError(
                "Provider physical authority record digest is stale."
            )
        physical_record = _freeze(physical_value)
    elif physical_binding == {
        "status": "omitted_explicit_pixel_only_canary",
        "record": None,
        "sha256": None,
    }:
        physical_record, physical_sha256 = None, None
    else:
        raise ProviderTrackMotionSourceHandleError(
            "Provider physical authority binding is invalid."
        )
    temporal_record, temporal_sha256, timing_status, timing_authoritative = (
        _temporal_authority(computation_record)
    )
    if require_authoritative_timing and not timing_authoritative:
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion run has no authoritative temporal authority; "
            f"status is {timing_status!r}, and caller FPS is compatibility-only."
        )
    arrays = {
        entry.declaration.path: _read_array(run, entry.declaration.path)
        for entry in receipt.entries
    }
    try:
        writer._validate_arrays(arrays)
    except writer.ProviderTrackMotionError as exc:
        raise ProviderTrackMotionSourceHandleError(str(exc)) from exc
    _validate_lineage_and_offsets(arrays)
    _validate_independent_validity(arrays)
    verification = _verification_digest(
        run_path=run_path,
        manifest_sha256=manifest_sha256,
        arrays=arrays,
        timing_status=timing_status,
    )
    return ProviderTrackMotionSourceHandle(
        analysis_zarr_path=archive,
        run_path=run_path,
        run_name=run_name,
        provider_manifest=raw_manifest,
        provider_manifest_sha256=manifest_sha256,
        selector_eligible=False,
        source_authority_record=source_record,
        source_authority_sha256=source_sha256,
        tracked_input_record=tracked_record,
        tracked_input_sha256=tracked_sha256,
        physical_authority_record=physical_record,
        physical_authority_sha256=physical_sha256,
        physical_authority_status=str(physical_status),
        computation_record=computation_record,
        computation_sha256=computation_sha256,
        temporal_authority_record=temporal_record,
        temporal_authority_sha256=temporal_sha256,
        temporal_authority_status=timing_status,
        timing_is_authoritative=timing_authoritative,
        arrays=arrays,
        row_count=_validate_exact_lengths(arrays)[0],
        track_count=_validate_exact_lengths(arrays)[1],
        per_second_count=_validate_exact_lengths(arrays)[2],
        verification_digest=verification,
        _use_consolidated=use_consolidated,
        _require_authoritative_timing=require_authoritative_timing,
        _verification_seal=_HANDLE_SEAL,
    )


def load_provider_track_motion_source_handle(
    analysis_zarr: str | Path,
    run_path: str,
    *,
    use_consolidated: bool = True,
    expected_manifest_sha256: str | None = None,
    require_authoritative_timing: bool = False,
) -> ProviderTrackMotionSourceHandle:
    """Load one exact complete provider-motion run without selector lookup."""

    exact_path, run_name = _canonical_run_path(run_path)
    if type(use_consolidated) is not bool:
        raise ProviderTrackMotionSourceHandleError(
            "use_consolidated must be the exact boolean metadata-read choice."
        )
    if type(require_authoritative_timing) is not bool:
        raise ProviderTrackMotionSourceHandleError(
            "require_authoritative_timing must be the exact boolean."
        )
    if expected_manifest_sha256 is not None:
        _require_digest(
            expected_manifest_sha256, name="expected provider manifest digest"
        )
    archive = Path(analysis_zarr).expanduser().resolve()
    snapshot = _load_once(
        archive,
        exact_path,
        run_name,
        use_consolidated=use_consolidated,
        expected_manifest_sha256=expected_manifest_sha256,
        require_authoritative_timing=require_authoritative_timing,
    )
    if use_consolidated:
        direct = _load_once(
            archive,
            exact_path,
            run_name,
            use_consolidated=False,
            expected_manifest_sha256=snapshot.provider_manifest_sha256,
            require_authoritative_timing=require_authoritative_timing,
        )
        if direct.verification_digest != snapshot.verification_digest:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion direct metadata differs from its published consolidated generation."
            )
    return snapshot


def require_provider_track_motion_source_handle(
    value: object,
) -> ProviderTrackMotionSourceHandle:
    """Require a loader-minted, currently verified provider-motion handle."""

    if type(value) is not ProviderTrackMotionSourceHandle:
        raise ProviderTrackMotionSourceHandleError(
            "A verified ProviderTrackMotionSourceHandle is required."
        )
    value.assert_current()
    return value


__all__ = [
    "PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_ID",
    "PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_VERSION",
    "ProviderTrackMotionSourceHandle",
    "ProviderTrackMotionSourceHandleError",
    "load_provider_track_motion_source_handle",
    "require_provider_track_motion_source_handle",
]
