"""Read-only routing over one validated recording-behavior bundle.

The bundle is the source-choice and compatibility authority.  This module is
the deliberately small consumer boundary: it exposes exact capability
bindings, resolves the already-bound exact-chaser projection receipt, and
loads requested provider-motion arrays without discovering selectors or
scanning sibling capabilities.

Provider-motion arrays are verified against their immutable manifest when
first consumed.  Only the requested arrays and the two track-partition arrays
needed to prove the selected row segment are read and hashed.  Verified arrays
are cached for the lifetime of the source handle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.swim_bout_io import (
    load_exact_selector_ineligible_default_swim_bout_tables,
)
from fisheye.analysis_workflows.exact_chaser_projection_receipt import (
    read_exact_chaser_projection_receipt,
)
from fisheye.analysis_workflows.materializers.provider_track_motion import (
    PROVIDER_TRACK_MOTION_MANIFEST_ATTR,
    PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR,
    PROVIDER_TRACK_MOTION_PARENT_PATH,
    provider_track_motion_manifest_digest,
)
from fisheye.analysis_workflows.validated_recording_behavior_bundle import (
    CAPABILITY_KEYS,
    read_validated_recording_behavior_bundle,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_SELECTOR_PARTS = frozenset(
    {
        "active",
        "active_run",
        "authoritative",
        "authoritative_run",
        "current",
        "current_run",
        "default",
        "default_run",
        "fallback",
        "latest",
        "latest_any",
        "latest_complete",
        "latest_pending",
        "selected",
        "selected_run",
    }
)
_STRUCTURAL_PROVIDER_ARRAYS = frozenset({"track_ids", "track_row_offsets"})


class ValidatedRecordingBehaviorSourceError(ValueError):
    """The requested bundle-backed source projection is not exact or current."""


class ValidatedCapabilityUnavailableError(ValidatedRecordingBehaviorSourceError):
    """A known bundle capability is explicitly not complete."""

    def __init__(
        self,
        capability: str,
        *,
        state: str,
        reason_code: str | None,
        detail: str | None,
    ) -> None:
        self.capability = capability
        self.state = state
        self.reason_code = reason_code
        self.detail = detail
        suffix = f" ({detail})" if detail else ""
        super().__init__(
            f"Validated capability {capability!r} is {state!r}: "
            f"{reason_code or 'no_reason_code'}{suffix}"
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidatedRecordingBehaviorSourceError(f"{field} must be one object.")
    return value


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValidatedRecordingBehaviorSourceError(
            f"{field} must be one lowercase SHA-256 digest."
        )
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValidatedRecordingBehaviorSourceError(
            f"{field} must be one nonempty exact string."
        )
    return value


def _exact_provider_array_path(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValidatedRecordingBehaviorSourceError(
            "Provider-motion array path must be one non-empty exact string."
        )
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(
            part in {"", ".", ".."} or part.casefold() in _SELECTOR_PARTS
            for part in parsed.parts
        )
    ):
        raise ValidatedRecordingBehaviorSourceError(
            f"Provider-motion array path is not exact: {value!r}."
        )
    return value


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ValidatedCapabilityBinding:
    """One complete bundle capability and the exact binding it names."""

    capability: str
    binding_scope: str
    binding_key: str
    binding: Mapping[str, Any]
    bundle_sha256: str


@dataclass(frozen=True)
class ProviderMotionArrayCatalog:
    """Manifest-only catalog for one exact provider-motion track."""

    run_path: str
    manifest_sha256: str
    verification_digest: str
    track_id: int
    track_row_start: int
    track_row_stop: int
    total_row_count: int
    array_records: Mapping[str, Mapping[str, Any]]

    @property
    def sample_array_paths(self) -> tuple[str, ...]:
        return tuple(
            path
            for path, record in self.array_records.items()
            if path not in _STRUCTURAL_PROVIDER_ARRAYS
            and "/" not in path
            and tuple(record["shape"])
            and int(record["shape"][0]) == self.total_row_count
        )


@dataclass(frozen=True)
class ProviderMotionTrackProjection:
    """Verified requested arrays for the one bundle-bound provider track."""

    analysis_zarr: str
    bundle_path: str
    bundle_sha256: str
    run_path: str
    manifest_sha256: str
    verification_digest: str
    track_id: int
    track_row_start: int
    track_row_stop: int
    arrays: Mapping[str, np.ndarray]
    array_sha256: Mapping[str, str]
    source_paths: Mapping[str, str]

    @property
    def row_count(self) -> int:
        return self.track_row_stop - self.track_row_start


@dataclass(frozen=True)
class ValidatedSemanticEpoch:
    """One exact semantic epoch sealed by the recording bundle."""

    window_id: int
    analysis_role: str
    source_label: str
    start_frame: int
    end_frame_exclusive: int
    source_interval_sha256: str
    protocol_semantic_hash: str
    protocol_semantic_step_index: int
    protocol_semantic_step_ref: str
    terminal_frame_excluded_pending_step_end_contract: bool


class ValidatedRecordingBehaviorSource:
    """Open one exact bundle and route bounded reads without source discovery."""

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        expected_analysis_zarr: str | Path | None = None,
        expected_recording_id: str | None = None,
        validate_current_sources: bool = True,
    ) -> None:
        if type(validate_current_sources) is not bool:
            raise TypeError("validate_current_sources must be the exact boolean.")
        path = Path(bundle_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Validated recording-behavior bundle does not exist: {path}"
            )
        bundle = read_validated_recording_behavior_bundle(
            path,
            expected_analysis_zarr=expected_analysis_zarr,
            expected_recording_id=expected_recording_id,
            validate_current_sources=validate_current_sources,
        )
        self.bundle_path = path
        self.bundle = bundle
        self.analysis_zarr = Path(str(bundle["analysis_zarr"])).resolve()
        self.recording_id = str(bundle["recording_id"])
        self.bundle_sha256 = _digest(
            bundle["record_sha256"], field="bundle.record_sha256"
        )
        self._provider_root: Any | None = None
        self._provider_run: Any | None = None
        self._provider_catalog: ProviderMotionArrayCatalog | None = None
        self._verified_provider_arrays: dict[str, np.ndarray] = {}

    def require_analysis_zarr(self, value: str | Path) -> Path:
        archive = Path(value).expanduser().resolve()
        if archive != self.analysis_zarr:
            raise ValidatedRecordingBehaviorSourceError(
                "Validated recording-behavior bundle names another analysis archive."
            )
        return archive

    def capability_record(self, capability: str) -> Mapping[str, Any]:
        if capability not in CAPABILITY_KEYS:
            raise ValidatedRecordingBehaviorSourceError(
                f"Unknown validated behavior capability {capability!r}."
            )
        return self.bundle["capabilities"][capability]

    def capability_states(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType(
            {
                capability: MappingProxyType(
                    {
                        "state": record["state"],
                        "reason_code": record["reason_code"],
                        "detail": record["detail"],
                    }
                )
                for capability, record in self.bundle["capabilities"].items()
            }
        )

    def require_capability(
        self,
        capability: str,
        *,
        expected_binding_scope: str | None = None,
    ) -> ValidatedCapabilityBinding:
        record = self.capability_record(capability)
        if record["state"] != "complete":
            raise ValidatedCapabilityUnavailableError(
                capability,
                state=str(record["state"]),
                reason_code=(
                    str(record["reason_code"])
                    if record["reason_code"] is not None
                    else None
                ),
                detail=str(record["detail"]) if record["detail"] is not None else None,
            )
        scope = str(record["binding_scope"])
        binding_key = str(record["binding_key"])
        if expected_binding_scope is not None and scope != expected_binding_scope:
            raise ValidatedRecordingBehaviorSourceError(
                f"Capability {capability!r} binds {scope!r}, not "
                f"{expected_binding_scope!r}."
            )
        bindings = self.bundle.get(scope)
        if not isinstance(bindings, Mapping) or binding_key not in bindings:
            raise ValidatedRecordingBehaviorSourceError(
                f"Capability {capability!r} no longer resolves its exact binding."
            )
        return ValidatedCapabilityBinding(
            capability=capability,
            binding_scope=scope,
            binding_key=binding_key,
            binding=_freeze(_plain(bindings[binding_key])),
            bundle_sha256=self.bundle_sha256,
        )

    def exact_projection_receipt_path(
        self,
        *,
        explicit_path: str | Path | None = None,
    ) -> Path:
        binding = _mapping(
            self.bundle["projection_receipt"], field="bundle.projection_receipt"
        )
        path = Path(str(binding["receipt_path"])).resolve()
        if explicit_path is not None:
            explicit = Path(explicit_path).expanduser().resolve()
            if explicit != path:
                raise ValidatedRecordingBehaviorSourceError(
                    "Explicit exact-chaser receipt differs from the receipt bound "
                    "by the validated recording-behavior bundle."
                )
        receipt = read_exact_chaser_projection_receipt(
            path,
            expected_analysis_zarr=self.analysis_zarr,
            validate_current_metadata=False,
            validate_child_receipts=False,
        )
        if receipt["record_sha256"] != binding["receipt_sha256"]:
            raise ValidatedRecordingBehaviorSourceError(
                "Exact-chaser projection receipt changed after the validated "
                "recording-behavior bundle was opened."
            )
        return path

    def scientific_child(self, capability: str) -> ValidatedCapabilityBinding:
        return self.require_capability(
            capability, expected_binding_scope="scientific_child_bindings"
        )

    def semantic_epoch_records(self) -> tuple[ValidatedSemanticEpoch, ...]:
        """Return the exact frame intervals selected as semantic epochs.

        The capability points at the immutable scientific child, while the
        transitive source binding carries the source intervals used by that
        child.  Both are required: callers must never rediscover or infer
        pre/training/post boundaries from labels or timing heuristics.
        """

        self.require_capability(
            "semantic_epochs", expected_binding_scope="scientific_child_bindings"
        )
        source_bindings = _mapping(
            self.bundle.get("source_bindings"), field="bundle.source_bindings"
        )
        binding = _mapping(
            source_bindings.get("semantic_epochs"),
            field="bundle.source_bindings.semantic_epochs",
        )
        if binding.get("binding_type") != "exact_child_plus_epoch_transitive_semantic_v1":
            raise ValidatedRecordingBehaviorSourceError(
                "Semantic-epoch source binding has an unsupported type."
            )
        source = _mapping(
            binding.get("source"), field="semantic_epochs.source"
        )
        raw_windows = source.get("position_suite_epochs")
        raw_bindings = source.get("semantic_role_bindings")
        if not isinstance(raw_windows, (tuple, list)) or not raw_windows:
            raise ValidatedRecordingBehaviorSourceError(
                "Semantic-epoch source has no exact position-suite intervals."
            )
        if not isinstance(raw_bindings, (tuple, list)) or not raw_bindings:
            raise ValidatedRecordingBehaviorSourceError(
                "Semantic-epoch source has no exact role bindings."
            )
        role_bindings: dict[int, Mapping[str, Any]] = {}
        for index, raw in enumerate(raw_bindings):
            record = _mapping(raw, field=f"semantic_role_bindings[{index}]")
            window_id = record.get("source_window_id")
            if type(window_id) is not int or window_id in role_bindings:
                raise ValidatedRecordingBehaviorSourceError(
                    "Semantic role bindings do not name unique integer windows."
                )
            role_bindings[window_id] = record

        result: list[ValidatedSemanticEpoch] = []
        seen: set[int] = set()
        for index, raw in enumerate(raw_windows):
            window = _mapping(raw, field=f"position_suite_epochs[{index}]")
            window_id = window.get("window_id")
            start = window.get("start_frame")
            stop = window.get("end_frame_exclusive")
            if (
                type(window_id) is not int
                or window_id in seen
                or type(start) is not int
                or type(stop) is not int
                or start < 0
                or stop <= start
            ):
                raise ValidatedRecordingBehaviorSourceError(
                    "Semantic position-suite intervals are invalid or duplicated."
                )
            seen.add(window_id)
            try:
                role = role_bindings[window_id]
            except KeyError as exc:
                raise ValidatedRecordingBehaviorSourceError(
                    "A semantic position-suite interval lacks its exact role binding."
                ) from exc
            interval_digest = _digest(
                window.get("source_interval_sha256"),
                field=f"position_suite_epochs[{index}].source_interval_sha256",
            )
            if (
                role.get("analysis_role") != window.get("analysis_role")
                or role.get("source_interval_sha256") != interval_digest
                or role.get("selected_start_frame") != start
                or role.get("selected_end_frame_exclusive") != stop
            ):
                raise ValidatedRecordingBehaviorSourceError(
                    "A semantic interval differs from its exact role binding."
                )
            semantic_step_index = role.get("protocol_semantic_step_index")
            terminal_excluded = role.get(
                "terminal_frame_excluded_pending_step_end_contract"
            )
            if type(semantic_step_index) is not int or type(terminal_excluded) is not bool:
                raise ValidatedRecordingBehaviorSourceError(
                    "Semantic role binding has invalid step or terminal-frame evidence."
                )
            result.append(
                ValidatedSemanticEpoch(
                    window_id=window_id,
                    analysis_role=_text(
                        window.get("analysis_role"),
                        field=f"position_suite_epochs[{index}].analysis_role",
                    ),
                    source_label=_text(
                        window.get("source_label"),
                        field=f"position_suite_epochs[{index}].source_label",
                    ),
                    start_frame=start,
                    end_frame_exclusive=stop,
                    source_interval_sha256=interval_digest,
                    protocol_semantic_hash=_text(
                        role.get("protocol_semantic_hash"),
                        field=f"semantic_role_bindings[{window_id}].protocol_semantic_hash",
                    ),
                    protocol_semantic_step_index=semantic_step_index,
                    protocol_semantic_step_ref=_text(
                        role.get("protocol_semantic_step_ref"),
                        field=f"semantic_role_bindings[{window_id}].protocol_semantic_step_ref",
                    ),
                    terminal_frame_excluded_pending_step_end_contract=terminal_excluded,
                )
            )
        if set(role_bindings) != seen:
            raise ValidatedRecordingBehaviorSourceError(
                "Semantic intervals and role bindings do not close the same axis."
            )
        return tuple(sorted(result, key=lambda item: item.window_id))

    def canonical_swim_bout_tables(self) -> Any:
        """Load the exact selector-ineligible bout source sealed by the bundle."""

        capability = self.require_capability(
            "canonical_swim_bouts", expected_binding_scope="source_bindings"
        )
        binding = _mapping(
            capability.binding.get("source"), field="canonical_swim_bouts.source"
        )
        run_path = _text(
            binding.get("run_path"), field="canonical_swim_bouts.source.run_path"
        )
        prefix = "analysis/swim_bout_runs/"
        if not run_path.startswith(prefix) or "/" in run_path[len(prefix) :]:
            raise ValidatedRecordingBehaviorSourceError(
                "Canonical swim-bout binding does not name one exact run."
            )
        root = open_zarr_root(
            self.analysis_zarr, mode="r", use_consolidated=True
        )
        tables = load_exact_selector_ineligible_default_swim_bout_tables(
            root, run_name=run_path.rsplit("/", 1)[-1]
        )
        frame_contract = tables.run_attrs.get("frame_axis_contract")
        motion_authority = tables.run_attrs.get("source_track_motion_authority")
        if not isinstance(frame_contract, Mapping) or not isinstance(
            motion_authority, Mapping
        ):
            raise ValidatedRecordingBehaviorSourceError(
                "Bound swim-bout run lacks exact frame or motion authority."
            )
        expected = {
            "run_path": run_path,
            "lineage_hash": binding.get("lineage_hash"),
            "track_id": int(binding["track_id"]),
            "candidate_id": int(binding["default_candidate_id"]),
            "signal_id": int(binding["default_signal_id"]),
            "signal_level": str(binding["default_signal_level"]),
            "frame_axis_sha256": str(binding["frame_axis_sha256"]),
            "motion_manifest": str(
                binding["source_track_motion_manifest_sha256"]
            ),
            "motion_verification": str(
                binding["source_track_motion_verification_digest"]
            ),
        }
        observed = {
            "run_path": tables.run_path,
            "lineage_hash": tables.run_attrs.get("lineage_hash"),
            "track_id": tables.candidate.track_id,
            "candidate_id": tables.candidate.candidate_id,
            "signal_id": tables.signal.signal_id,
            "signal_level": tables.signal.speed_level,
            "frame_axis_sha256": frame_contract.get("content_sha256"),
            "motion_manifest": frame_contract.get(
                "source_track_motion_manifest_sha256"
            ),
            "motion_verification": motion_authority.get(
                "provider_verification_digest"
            ),
        }
        if observed != expected or tables.signal.role != "detector_response":
            raise ValidatedRecordingBehaviorSourceError(
                "Selected swim-bout source differs from the validated bundle."
            )
        fps = tables.run_attrs.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or float(fps) <= 0
        ):
            raise ValidatedRecordingBehaviorSourceError(
                "Bound swim-bout run lacks exact positive FPS."
            )
        required_interval_fields = {
            "interval_id",
            "valid",
            "prev_bout_id",
            "next_bout_id",
            "prev_end_frame",
            "next_start_frame",
            "interval_frames",
            "prev_end_time_s",
            "next_start_time_s",
            "interval_s",
        }
        observed_fields = set(tables.inter_bout_intervals.dtype.names or ())
        if not required_interval_fields.issubset(observed_fields):
            raise ValidatedRecordingBehaviorSourceError(
                "Bound swim-bout run lacks required interval fields."
            )
        return tables

    def provider_motion_catalog(self) -> ProviderMotionArrayCatalog:
        if self._provider_catalog is not None:
            return self._provider_catalog
        capability = self.require_capability(
            "provider_motion", expected_binding_scope="source_bindings"
        )
        binding = capability.binding
        source = _mapping(binding.get("source"), field="provider_motion.source")
        run_path = str(source.get("run_path"))
        prefix = f"{PROVIDER_TRACK_MOTION_PARENT_PATH}/"
        if not run_path.startswith(prefix) or "/" in run_path[len(prefix) :]:
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion binding does not name one exact provider run."
            )
        manifest_sha256 = _digest(
            source.get("manifest_sha256"),
            field="provider_motion.source.manifest_sha256",
        )
        verification_digest = _digest(
            source.get("verification_digest"),
            field="provider_motion.source.verification_digest",
        )
        track_id = source.get("track_id")
        track_row_start = source.get("track_row_start")
        track_row_stop = source.get("track_row_stop")
        if (
            type(track_id) is not int
            or track_id < 0
            or type(track_row_start) is not int
            or type(track_row_stop) is not int
            or track_row_start < 0
            or track_row_stop < track_row_start
        ):
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion track partition binding is invalid."
            )

        root = open_zarr_root(
            self.analysis_zarr,
            mode="r",
            use_consolidated=True,
        )
        try:
            run = root[run_path]
        except Exception as exc:
            raise ValidatedRecordingBehaviorSourceError(
                f"Exact provider-motion run is absent: {run_path}."
            ) from exc
        manifest = _mapping(
            run.attrs.get(PROVIDER_TRACK_MOTION_MANIFEST_ATTR),
            field="provider-motion manifest",
        )
        try:
            observed_manifest_sha256 = provider_track_motion_manifest_digest(manifest)
        except (TypeError, ValueError) as exc:
            raise ValidatedRecordingBehaviorSourceError(
                f"Provider-motion manifest is invalid: {exc}"
            ) from exc
        payload = _mapping(manifest.get("payload"), field="provider-motion payload")
        if (
            observed_manifest_sha256 != manifest_sha256
            or run.attrs.get(PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR)
            != manifest_sha256
            or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or run.attrs.get("stage_selector_eligible") is not False
            or payload.get("run_path") != run_path
            or payload.get("row_axis") != "track_sample"
            or payload.get("status") != RUN_STATUS_COMPLETE
            or payload.get("stage_selector_eligible") is not False
            or _plain(payload.get("source_authority"))
            != _plain(binding.get("source_authority"))
        ):
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion manifest, lifecycle, or source authority differs "
                "from the validated bundle."
            )
        raw_records = payload.get("arrays")
        if not isinstance(raw_records, (tuple, list)) or not raw_records:
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion manifest has no array declarations."
            )
        records: dict[str, Mapping[str, Any]] = {}
        for index, raw_record in enumerate(raw_records):
            record = _mapping(raw_record, field=f"provider-motion arrays[{index}]")
            if set(record) != {"path", "dtype", "shape", "sha256"}:
                raise ValidatedRecordingBehaviorSourceError(
                    "Provider-motion array declaration field set is inexact."
                )
            path = _exact_provider_array_path(record.get("path"))
            shape = record.get("shape")
            dtype = record.get("dtype")
            if (
                path in records
                or not isinstance(shape, (tuple, list))
                or not shape
                or any(type(size) is not int or size < 0 for size in shape)
                or type(dtype) is not str
            ):
                raise ValidatedRecordingBehaviorSourceError(
                    f"Provider-motion declaration for {path!r} is invalid."
                )
            try:
                normalized_dtype = np.dtype(dtype).str
            except TypeError as exc:
                raise ValidatedRecordingBehaviorSourceError(
                    f"Provider-motion declaration for {path!r} has invalid dtype."
                ) from exc
            if normalized_dtype != dtype:
                raise ValidatedRecordingBehaviorSourceError(
                    f"Provider-motion declaration for {path!r} has noncanonical dtype."
                )
            digest = _digest(
                record.get("sha256"), field=f"provider-motion arrays[{path}].sha256"
            )
            records[path] = MappingProxyType(
                {
                    "path": path,
                    "dtype": dtype,
                    "shape": tuple(shape),
                    "sha256": digest,
                }
            )
        if tuple(records) != tuple(sorted(records)):
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion array declarations are not exactly sorted."
            )
        for required in ("track_ids", "track_row_offsets", "track_sample_key"):
            if required not in records:
                raise ValidatedRecordingBehaviorSourceError(
                    f"Provider-motion manifest lacks required {required!r}."
                )
        total_row_count = int(records["track_sample_key"]["shape"][0])
        self._provider_root = root
        self._provider_run = run
        self._provider_catalog = ProviderMotionArrayCatalog(
            run_path=run_path,
            manifest_sha256=manifest_sha256,
            verification_digest=verification_digest,
            track_id=track_id,
            track_row_start=track_row_start,
            track_row_stop=track_row_stop,
            total_row_count=total_row_count,
            array_records=MappingProxyType(records),
        )
        self._validate_provider_track_partition()
        return self._provider_catalog

    def _read_verified_provider_array(self, path: str) -> np.ndarray:
        if path in self._verified_provider_arrays:
            return self._verified_provider_arrays[path]
        catalog = self.provider_motion_catalog()
        try:
            record = catalog.array_records[path]
        except KeyError as exc:
            raise ValidatedRecordingBehaviorSourceError(
                f"Provider-motion manifest does not declare array {path!r}."
            ) from exc
        try:
            node = self._provider_run[path]
        except Exception as exc:
            raise ValidatedRecordingBehaviorSourceError(
                f"Provider-motion array {path!r} is absent."
            ) from exc
        try:
            node_dtype = np.dtype(getattr(node, "dtype", None)).str
        except TypeError as exc:
            raise ValidatedRecordingBehaviorSourceError(
                f"Provider-motion array {path!r} has invalid dtype metadata."
            ) from exc
        if (
            tuple(getattr(node, "shape", ())) != tuple(record["shape"])
            or node_dtype != record["dtype"]
        ):
            raise ValidatedRecordingBehaviorSourceError(
                f"Provider-motion array {path!r} metadata differs from its manifest."
            )
        values = np.asarray(node[:])
        if sha256_array(values) != record["sha256"]:
            raise ValidatedRecordingBehaviorSourceError(
                f"Provider-motion array {path!r} differs from its manifest digest."
            )
        values = _readonly(values)
        self._verified_provider_arrays[path] = values
        return values

    def _validate_provider_track_partition(self) -> None:
        catalog = self._provider_catalog
        if catalog is None:
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion catalog is unavailable."
            )
        track_ids = self._read_verified_provider_array("track_ids")
        offsets = self._read_verified_provider_array("track_row_offsets")
        if (
            track_ids.dtype != np.dtype("int64")
            or offsets.dtype != np.dtype("int64")
            or track_ids.ndim != 1
            or offsets.ndim != 1
            or offsets.shape != (track_ids.shape[0] + 1,)
            or (track_ids.size and not bool(np.all(np.diff(track_ids) > 0)))
            or offsets.size == 0
            or int(offsets[0]) != 0
            or not bool(np.all(np.diff(offsets) >= 0))
            or int(offsets[-1]) != catalog.total_row_count
        ):
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion track partition arrays are invalid."
            )
        matches = np.flatnonzero(track_ids == catalog.track_id)
        if matches.size != 1:
            raise ValidatedRecordingBehaviorSourceError(
                "Bundle-bound provider track does not occur exactly once."
            )
        index = int(matches[0])
        if (
            int(offsets[index]) != catalog.track_row_start
            or int(offsets[index + 1]) != catalog.track_row_stop
        ):
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion track partition differs from the validated bundle."
            )

    def provider_motion_track_projection(
        self,
        array_paths: Sequence[str],
    ) -> ProviderMotionTrackProjection:
        catalog = self.provider_motion_catalog()
        requested = tuple(_exact_provider_array_path(path) for path in array_paths)
        if not requested or len(set(requested)) != len(requested):
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion projection requires one non-empty unique array set."
            )
        arrays: dict[str, np.ndarray] = {}
        digests: dict[str, str] = {}
        source_paths: dict[str, str] = {}
        for path in requested:
            if path in _STRUCTURAL_PROVIDER_ARRAYS or "/" in path:
                raise ValidatedRecordingBehaviorSourceError(
                    f"Provider-motion array {path!r} is not a track-sample payload."
                )
            try:
                record = catalog.array_records[path]
            except KeyError as exc:
                raise ValidatedRecordingBehaviorSourceError(
                    f"Provider-motion manifest does not declare array {path!r}."
                ) from exc
            shape = tuple(record["shape"])
            if not shape or int(shape[0]) != catalog.total_row_count:
                raise ValidatedRecordingBehaviorSourceError(
                    f"Provider-motion array {path!r} is not aligned to track_sample."
                )
            full = self._read_verified_provider_array(path)
            selected = _readonly(full[catalog.track_row_start : catalog.track_row_stop])
            if (
                int(selected.shape[0])
                != catalog.track_row_stop - catalog.track_row_start
            ):
                raise ValidatedRecordingBehaviorSourceError(
                    f"Provider-motion array {path!r} has an invalid selected row count."
                )
            arrays[path] = selected
            digests[path] = str(record["sha256"])
            source_paths[path] = f"{catalog.run_path}/{path}"
        return ProviderMotionTrackProjection(
            analysis_zarr=str(self.analysis_zarr),
            bundle_path=str(self.bundle_path),
            bundle_sha256=self.bundle_sha256,
            run_path=catalog.run_path,
            manifest_sha256=catalog.manifest_sha256,
            verification_digest=catalog.verification_digest,
            track_id=catalog.track_id,
            track_row_start=catalog.track_row_start,
            track_row_stop=catalog.track_row_stop,
            arrays=MappingProxyType(arrays),
            array_sha256=MappingProxyType(digests),
            source_paths=MappingProxyType(source_paths),
        )


__all__ = [
    "ProviderMotionArrayCatalog",
    "ProviderMotionTrackProjection",
    "ValidatedSemanticEpoch",
    "ValidatedCapabilityBinding",
    "ValidatedCapabilityUnavailableError",
    "ValidatedRecordingBehaviorSource",
    "ValidatedRecordingBehaviorSourceError",
]
