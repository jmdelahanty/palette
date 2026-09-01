"""Receipt-bound streaming adapters for validated-behavior Phase B.

Adapters copy exact persisted sample arrays selected by a validated recording
bundle. They neither discover selectors nor interpolate, re-bin, smooth, or
recompute scientific values. Column batches are bounded and primary-key sorted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .validated_behavior_adapters import (
    ValidatedBehaviorAdapterError,
    _RecordingContext,
)
from .validated_behavior_cohort import ValidatedBehaviorBatchSource


_PROVIDER_MOTION_ARRAYS = (
    "track_sample_key",
    "source_acquisition_frame_index",
    "source_observation_instance_key",
    "source_provider_row_index",
    "source_position_row_index",
    "source_body_frame_row_index",
    "source_tracking_row_index",
    "time_seconds",
    "positions_px",
    "positions_mm",
    "position_source_valid",
    "body_frame_source_valid",
    "linear_sample_valid",
    "linear_sample_reason_code",
    "angular_sample_valid",
    "angular_sample_reason_code",
    "heading_degrees",
    "heading_radians",
    "smoothed_heading_degrees",
    "smoothed_heading_radians",
    "delta_frames",
    "delta_seconds",
    "transition_valid",
    "transition_reason_code",
    "speed_raw_px",
    "speed_filtered_px",
    "speed_smoothed_px",
    "speed_averaged_px",
    "speed_raw_mm",
    "speed_filtered_mm",
    "speed_smoothed_mm",
    "speed_averaged_mm",
    "acceleration_px",
    "smoothed_acceleration_px",
    "acceleration_mm",
    "smoothed_acceleration_mm",
    "frame_path_distance_raw_px",
    "frame_path_distance_filtered_px",
    "frame_path_distance_smoothed_px",
    "cumulative_path_distance_px",
    "frame_path_distance_raw_mm",
    "frame_path_distance_filtered_mm",
    "frame_path_distance_smoothed_mm",
    "cumulative_path_distance_mm",
    "delta_heading_degrees",
    "angular_velocity_raw_deg_s",
    "angular_speed_raw_deg_s",
    "delta_heading_smoothed_degrees",
    "angular_velocity_smoothed_deg_s",
    "angular_speed_smoothed_deg_s",
)

_BODY_ALIGNMENT_FRAME_ARRAYS = (
    "frame_acquisition_frame_id",
    "frame_epoch_role_code",
    "frame_epoch_window_id",
    "frame_selection_member",
    "frame_chaser_occurrence_member",
    "frame_chaser_identity_code",
    "frame_chaser_behavior_role_code",
    "frame_chaser_behavior_role_valid",
    "frame_relative_distance_physical",
    "frame_relative_physical_valid",
    "frame_relative_physical_reason_code",
    "frame_body_source_row_id",
    "frame_body_source_row_valid",
    "frame_body_heading_deg",
    "frame_body_heading_valid",
    "frame_body_heading_reason_code",
    "frame_body_bearing_deg",
    "frame_body_bearing_valid",
    "frame_body_bearing_reason_code",
    "frame_alignment_cos",
    "frame_lateral_sin",
    "frame_alignment_valid",
    "frame_alignment_reason_code",
)

_CONTROLLER_DENSE_ARRAYS = (
    "source_relative_row_id",
    "trial_row_id_by_source_row",
    "trial_envelope_row_id_by_source_row",
    "logged_active_trial_member",
    "trial_envelope_member",
    "trial_gap_member",
    "trial_gap_reason_code_by_source_row",
    "logged_active_trial_id_unavailable",
    "trial_row_id",
    "logged_trial_id",
)

_RELATIVE_BASE_ARRAYS = (
    "acquisition_frame_delta",
    "acquisition_frame_id",
    "active_state_code",
    "active_state_reason_code",
    "active_state_valid",
    "chaser_behavior_role_code",
    "chaser_behavior_role_reason_code",
    "chaser_behavior_role_valid",
    "chaser_identity_code",
    "chaser_occurrence_member",
    "chaser_position_reason_code",
    "chaser_position_valid",
    "chaser_position_xy_px",
    "chaser_source_row_id",
    "chaser_source_row_reason_code",
    "chaser_source_row_valid",
    "fish_identity_code",
    "fish_position_reason_code",
    "fish_position_valid",
    "fish_position_xy_px",
    "fish_source_row_id",
    "fish_source_row_reason_code",
    "fish_source_row_valid",
    "fish_transition_reason_code",
    "fish_transition_valid",
    "nearest_chaser_distance_physical",
    "nearest_chaser_distance_px",
    "nearest_chaser_identity_code",
    "nearest_chaser_member",
    "nearest_chaser_reason_code",
    "nearest_chaser_source_row_id",
    "nearest_chaser_valid",
    "relative_distance_physical",
    "relative_distance_px",
    "relative_physical_reason_code",
    "relative_physical_valid",
    "relative_px_reason_code",
    "relative_px_valid",
    "relative_transition_reason_code",
    "relative_transition_valid",
    "relative_vector_physical_xy",
    "relative_vector_px_xy",
    "row_reason_code",
    "row_valid",
    "selection_member",
    "timestamp_delta_ns",
    "timestamp_ns",
    "timestamp_reason_code",
    "timestamp_valid",
    "track_sample_id",
    "trial_id",
    "trial_reason_code",
    "trial_valid",
)

_RELATIVE_BODY_ARRAYS = (
    "body_axes_reason_code",
    "body_axes_valid",
    "body_forward_axis_xy",
    "body_heading_deg",
    "body_heading_reason_code",
    "body_heading_transition_reason_code",
    "body_heading_transition_valid",
    "body_heading_valid",
    "body_left_axis_xy",
    "body_origin_reason_code",
    "body_origin_valid",
    "body_origin_xy_px",
    "body_reason_code",
    "body_source_row_id",
    "body_source_row_reason_code",
    "body_source_row_valid",
    "body_valid",
)


def _fail(message: str) -> None:
    raise ValidatedBehaviorAdapterError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _repeat(value: Any, count: int) -> list[Any]:
    return [value] * count


def _registry(value: object, *, field: str) -> dict[int, str]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one sealed code registry.")
    try:
        result = {int(key): str(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValidatedBehaviorAdapterError(
            f"{field} contains a non-integer code."
        ) from exc
    if len(result) != len(value) or any(not item for item in result.values()):
        _fail(f"{field} is not one unique non-empty code registry.")
    return result


def _batch_source(
    row_count: int,
    batch_rows: int,
    builder: Callable[[int, int], Mapping[str, Any]],
    *,
    zero_reason: str = "complete-no-rows",
) -> ValidatedBehaviorBatchSource:
    if row_count < 0 or batch_rows <= 0:
        _fail("Dense adapter row or batch count is invalid.")

    def batches() -> Iterator[Mapping[str, Any]]:
        for start in range(0, row_count, batch_rows):
            yield builder(start, min(row_count, start + batch_rows))

    return ValidatedBehaviorBatchSource(
        batches=batches(),
        zero_row_reason=zero_reason if row_count == 0 else None,
    )


def _common_columns(common: Mapping[str, Any], count: int) -> dict[str, Any]:
    return {name: _repeat(value, count) for name, value in common.items()}


def _batch_rows(context: "_PhaseBContext") -> int:
    value = context.base.plan["parameters"]["effective_row_group_rows"]
    if type(value) is not int or value <= 0:
        _fail("Export plan has an invalid effective row-group size.")
    return value


class _PhaseBContext:
    def __init__(
        self,
        plan: Mapping[str, Any],
        membership_member: Mapping[str, Any],
        bundle_member: Mapping[str, Any],
    ) -> None:
        self.base = _RecordingContext(plan, membership_member, bundle_member)
        self._validated_source: Any | None = None
        self._relative: dict[str, Any] = {}
        self._dense_children: dict[str, Any] = {}
        self._proxy: Any | None = None

    @property
    def bundle(self) -> Mapping[str, Any]:
        return self.base.bundle

    @property
    def bundle_common(self) -> dict[str, Any]:
        return self.base.bundle_common

    @property
    def recording_id(self) -> str:
        return self.base.recording_id

    @property
    def analysis_zarr(self) -> Path:
        return self.base.analysis_zarr

    def validated_source(self) -> Any:
        if self._validated_source is None:
            from fisheye.analysis_workflows.validated_recording_behavior_source import (
                ValidatedRecordingBehaviorSource,
            )

            source = ValidatedRecordingBehaviorSource(
                self.base.bundle_path,
                expected_analysis_zarr=self.analysis_zarr,
                expected_recording_id=self.recording_id,
                # The bundle set already seals the bundle bytes. Every consumed
                # dense source below performs its own exact targeted validation.
                validate_current_sources=False,
            )
            if source.bundle_sha256 != self.bundle["record_sha256"]:
                _fail("Validated source differs from the bundle-set member.")
            self._validated_source = source
        return self._validated_source

    def relative(self, role: str) -> Any:
        if role in self._relative:
            return self._relative[role]
        if role not in {"detection", "keypoint"}:
            _fail(f"Unknown relative-frame provider role {role!r}.")
        key = f"chaser_relative_{role}"
        binding = self.base.child_binding(key)
        from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
            load_chaser_relative_frame_targeted_source_handle,
        )

        handle = load_chaser_relative_frame_targeted_source_handle(
            binding["receipt_path"],
            required_base_arrays=_RELATIVE_BASE_ARRAYS,
            required_body_arrays=(
                _RELATIVE_BODY_ARRAYS if role == "keypoint" else ()
            ),
            collapsed_frame_arrays=(),
            expected_analysis_zarr=self.analysis_zarr,
            expected_recording_id=self.recording_id,
            expected_run_name=str(binding["run_path"]).rsplit("/", 1)[-1],
        )
        if (
            handle.run_path != binding["run_path"]
            or handle.manifest_sha256 != binding["manifest_sha256"]
            or handle.payload_digest != binding["payload_digest"]
            or handle.receipt_digest != binding["receipt_sha256"]
        ):
            _fail(f"{key} dense handle differs from the validated bundle.")
        self._relative[role] = handle
        return handle

    def dense_child(
        self,
        key: str,
        *,
        successor_kind: str,
        arrays: Sequence[str],
    ) -> Any:
        if key in self._dense_children:
            return self._dense_children[key]
        binding = self.base.child_binding(key)
        from fisheye.analysis_workflows.composable_chaser_successor_publication import (
            load_composable_chaser_successor_source_handle,
        )

        handle = load_composable_chaser_successor_source_handle(
            self.analysis_zarr,
            successor_kind=successor_kind,
            run_name=str(binding["run_path"]).rsplit("/", 1)[-1],
            expected_recording_id=self.recording_id,
            direct_validation_receipt=binding["receipt_path"],
            required_array_names=arrays,
        )
        if (
            handle.run_path != binding["run_path"]
            or handle.manifest_sha256 != binding["manifest_sha256"]
            or str(handle.manifest["payload_digest"]) != binding["payload_digest"]
            or handle.receipt_digest != binding["receipt_sha256"]
        ):
            _fail(f"Dense child {key!r} differs from the validated bundle.")
        handle.require_verified_arrays(arrays)
        self._dense_children[key] = handle
        return handle

    def proxy(self) -> Any:
        if self._proxy is not None:
            return self._proxy
        relative = self.relative("keypoint")
        context = relative.manifest.get("context")
        if not isinstance(context, Mapping):
            _fail("Relative frame lacks one exact sealed context.")
        envelope = context.get("acquisition_projection_publication")
        if not isinstance(envelope, Mapping) or set(envelope) != {"record", "sha256"}:
            _fail("Relative frame lacks one exact proxy publication binding.")
        record = _plain(envelope["record"])
        if canonical_json_sha256(record) != envelope["sha256"]:
            _fail("Relative-frame proxy publication digest is stale.")
        run_path = str(record.get("run_path"))
        from fisheye.analysis_workflows.chaser_input_provenance_proxy_source_handle import (
            load_chaser_input_provenance_proxy_source_handle,
        )

        handle = load_chaser_input_provenance_proxy_source_handle(
            self.analysis_zarr,
            run_name=run_path.rsplit("/", 1)[-1],
            expected_recording_id=self.recording_id,
            use_consolidated=True,
            expected_manifest_sha256=str(record.get("manifest_sha256")),
        )
        if _plain(handle.publication_binding_record) != record:
            _fail("Proxy source handle differs from the relative-frame binding.")
        self._proxy = handle
        return handle

    def provider_authority(self, role: str) -> Mapping[str, Any]:
        key = f"fish_position_{role}"
        self.base.require_capability(key)
        return self.bundle["source_bindings"][key]["authority"]

    def chaser_maps(self) -> tuple[dict[int, str], dict[int, str]]:
        return self.base.chaser_identity_maps()


class _LastPhaseBContext:
    def __init__(self) -> None:
        self._key: tuple[str, str, str] | None = None
        self._context: _PhaseBContext | None = None

    def get(
        self,
        plan: Mapping[str, Any],
        membership_member: Mapping[str, Any],
        bundle_member: Mapping[str, Any],
    ) -> _PhaseBContext:
        key = (
            str(plan["plan_sha256"]),
            str(membership_member["member_sha256"]),
            str(bundle_member["member_sha256"]),
        )
        if self._key != key:
            self._context = _PhaseBContext(plan, membership_member, bundle_member)
            self._key = key
        assert self._context is not None
        return self._context


def _provider_motion_samples(context: _PhaseBContext) -> ValidatedBehaviorBatchSource:
    source = context.validated_source()
    projection = source.provider_motion_track_projection(_PROVIDER_MOTION_ARRAYS)
    arrays = projection.arrays
    keys = np.asarray(arrays["track_sample_key"])
    row_count = int(keys.shape[0])
    if (
        keys.shape != (row_count, 2)
        or np.any(keys[:, 0] != projection.track_id)
        or not np.array_equal(keys[:, 1], arrays["source_acquisition_frame_index"])
    ):
        _fail("Provider-motion sample keys are inconsistent.")
    authority = context.provider_authority("keypoint")
    common = {
        **context.bundle_common,
        "source_binding_key": "provider_motion",
        "source_run_path": projection.run_path,
        "source_manifest_sha256": projection.manifest_sha256,
        "source_verification_digest": projection.verification_digest,
        "provider_role": "keypoint",
        "position_provider_id": str(authority["provider_id"]),
        "position_provider_digest": str(authority["provider_digest"]),
    }
    track_rows = np.arange(
        projection.track_row_start,
        projection.track_row_stop,
        dtype=np.int64,
    )

    def build(start: int, stop: int) -> Mapping[str, Any]:
        count = stop - start
        sl = slice(start, stop)
        return {
            **_common_columns(common, count),
            "track_id": keys[sl, 0],
            "track_sample_row_id": track_rows[sl],
            "acquisition_frame_id": keys[sl, 1],
            "source_observation_instance_key": arrays[
                "source_observation_instance_key"
            ][sl],
            "source_provider_row_id": arrays["source_provider_row_index"][sl],
            "source_position_row_id": arrays["source_position_row_index"][sl],
            "source_body_frame_row_id": arrays["source_body_frame_row_index"][sl],
            "source_tracking_row_id": arrays["source_tracking_row_index"][sl],
            "time_s": arrays["time_seconds"][sl],
            "position_x_px": arrays["positions_px"][sl, 0],
            "position_y_px": arrays["positions_px"][sl, 1],
            "position_x_mm": arrays["positions_mm"][sl, 0],
            "position_y_mm": arrays["positions_mm"][sl, 1],
            "position_source_valid": arrays["position_source_valid"][sl],
            "body_frame_source_valid": arrays["body_frame_source_valid"][sl],
            "linear_sample_valid": arrays["linear_sample_valid"][sl],
            "linear_sample_reason_code": arrays["linear_sample_reason_code"][sl],
            "angular_sample_valid": arrays["angular_sample_valid"][sl],
            "angular_sample_reason_code": arrays["angular_sample_reason_code"][sl],
            "heading_deg": arrays["heading_degrees"][sl],
            "heading_rad": arrays["heading_radians"][sl],
            "smoothed_heading_deg": arrays["smoothed_heading_degrees"][sl],
            "smoothed_heading_rad": arrays["smoothed_heading_radians"][sl],
            "delta_frames": arrays["delta_frames"][sl],
            "delta_s": arrays["delta_seconds"][sl],
            "transition_valid": arrays["transition_valid"][sl],
            "transition_reason_code": arrays["transition_reason_code"][sl],
            "speed_raw_px_s": arrays["speed_raw_px"][sl],
            "speed_filtered_px_s": arrays["speed_filtered_px"][sl],
            "speed_smoothed_px_s": arrays["speed_smoothed_px"][sl],
            "speed_averaged_px_s": arrays["speed_averaged_px"][sl],
            "speed_raw_mm_s": arrays["speed_raw_mm"][sl],
            "speed_filtered_mm_s": arrays["speed_filtered_mm"][sl],
            "speed_smoothed_mm_s": arrays["speed_smoothed_mm"][sl],
            "speed_averaged_mm_s": arrays["speed_averaged_mm"][sl],
            "acceleration_px_s2": arrays["acceleration_px"][sl],
            "smoothed_acceleration_px_s2": arrays["smoothed_acceleration_px"][sl],
            "acceleration_mm_s2": arrays["acceleration_mm"][sl],
            "smoothed_acceleration_mm_s2": arrays["smoothed_acceleration_mm"][sl],
            "frame_path_distance_raw_px": arrays["frame_path_distance_raw_px"][sl],
            "frame_path_distance_filtered_px": arrays[
                "frame_path_distance_filtered_px"
            ][sl],
            "frame_path_distance_smoothed_px": arrays[
                "frame_path_distance_smoothed_px"
            ][sl],
            "cumulative_path_distance_px": arrays["cumulative_path_distance_px"][sl],
            "frame_path_distance_raw_mm": arrays["frame_path_distance_raw_mm"][sl],
            "frame_path_distance_filtered_mm": arrays[
                "frame_path_distance_filtered_mm"
            ][sl],
            "frame_path_distance_smoothed_mm": arrays[
                "frame_path_distance_smoothed_mm"
            ][sl],
            "cumulative_path_distance_mm": arrays["cumulative_path_distance_mm"][sl],
            "delta_heading_deg": arrays["delta_heading_degrees"][sl],
            "angular_velocity_raw_deg_s": arrays["angular_velocity_raw_deg_s"][sl],
            "angular_speed_raw_deg_s": arrays["angular_speed_raw_deg_s"][sl],
            "delta_heading_smoothed_deg": arrays[
                "delta_heading_smoothed_degrees"
            ][sl],
            "angular_velocity_smoothed_deg_s": arrays[
                "angular_velocity_smoothed_deg_s"
            ][sl],
            "angular_speed_smoothed_deg_s": arrays[
                "angular_speed_smoothed_deg_s"
            ][sl],
        }

    return _batch_source(row_count, _batch_rows(context), build)


def _bout_detector_signal_samples(
    context: _PhaseBContext,
) -> ValidatedBehaviorBatchSource:
    context.base.require_capability("canonical_swim_bouts")
    binding = context.bundle["source_bindings"]["canonical_swim_bouts"]["source"]
    run_path = str(binding["run_path"])
    from fisheye.analysis.swim_bout_io import (
        load_exact_selector_ineligible_default_swim_bout_tables,
    )
    from fisheye.shared.zarr_io import open_zarr_root

    root = open_zarr_root(context.analysis_zarr, mode="r", use_consolidated=True)
    tables = load_exact_selector_ineligible_default_swim_bout_tables(
        root,
        run_name=run_path.rsplit("/", 1)[-1],
    )
    frame_contract = tables.run_attrs.get("frame_axis_contract")
    motion_authority = tables.run_attrs.get("source_track_motion_authority")
    if not isinstance(frame_contract, Mapping) or not isinstance(
        motion_authority, Mapping
    ):
        _fail("Bound swim-bout run lacks exact frame or motion authority.")
    expected = {
        "run_path": run_path,
        "lineage_hash": binding["lineage_hash"],
        "track_id": int(binding["track_id"]),
        "candidate_id": int(binding["default_candidate_id"]),
        "signal_id": int(binding["default_signal_id"]),
        "signal_level": str(binding["default_signal_level"]),
        "frame_axis_sha256": str(binding["frame_axis_sha256"]),
        "motion_manifest": str(binding["source_track_motion_manifest_sha256"]),
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
        _fail("Selected bout detector signal differs from the validated bundle.")
    frames = np.asarray(tables.series.get("frame_indices"))
    signal = np.asarray(tables.series.get("detection_signal_mm_s"))
    if (
        frames.dtype != np.dtype("int64")
        or signal.dtype != np.dtype("float32")
        or frames.ndim != 1
        or signal.shape != frames.shape
    ):
        _fail("Selected bout detector signal or frame axis is invalid.")
    row_count = int(frames.size)
    common = {
        **context.bundle_common,
        "source_binding_key": "canonical_swim_bouts",
        "swim_bout_run_path": run_path,
        "swim_bout_lineage_sha256": str(binding["lineage_hash"]),
        "source_track_motion_manifest_sha256": str(
            binding["source_track_motion_manifest_sha256"]
        ),
        "source_track_motion_verification_digest": str(
            binding["source_track_motion_verification_digest"]
        ),
        "track_id": int(binding["track_id"]),
        "candidate_id": tables.candidate.candidate_id,
        "signal_id": tables.signal.signal_id,
        "signal_level": tables.signal.speed_level,
        "signal_name": tables.signal.signal_name,
        "signal_role": tables.signal.role,
        "source_level": tables.signal.source_level,
    }
    sample_rows = np.arange(row_count, dtype=np.int64)

    def build(start: int, stop: int) -> Mapping[str, Any]:
        count = stop - start
        sl = slice(start, stop)
        return {
            **_common_columns(common, count),
            "signal_sample_row_id": sample_rows[sl],
            "acquisition_frame_id": frames[sl],
            "detection_signal_mm_s": signal[sl],
        }

    return _batch_source(row_count, _batch_rows(context), build)


def _relative_common(
    context: _PhaseBContext, role: str
) -> tuple[Any, dict[str, Any], dict[int, str], dict[int, str], dict[int, str]]:
    handle = context.relative(role)
    key = f"chaser_relative_{role}"
    authority = context.provider_authority(role)
    identities, roles = context.chaser_maps()
    active = _registry(
        handle.manifest["identity_registries"]["active_state"],
        field=f"{key}.active_state",
    )
    common = {
        **context.base.child_common(key),
        "provider_role": role,
        "position_provider_id": str(authority["provider_id"]),
        "position_provider_digest": str(authority["provider_digest"]),
    }
    return handle, common, identities, roles, active


def _chaser_relative_samples(
    context: _PhaseBContext,
) -> ValidatedBehaviorBatchSource:
    batch_rows = _batch_rows(context)
    sources = [
        _relative_common(context, role) for role in ("detection", "keypoint")
    ]
    row_count = sum(int(handle.n_rows) for handle, *_rest in sources)

    def batches() -> Iterator[Mapping[str, Any]]:
        for handle, common, identities, roles, active_registry in sources:
            arrays = handle.base_arrays
            rows = np.arange(handle.n_rows, dtype=np.int64)
            for start in range(0, handle.n_rows, batch_rows):
                stop = min(handle.n_rows, start + batch_rows)
                count = stop - start
                sl = slice(start, stop)
                chaser_codes = np.asarray(arrays["chaser_identity_code"])[sl]
                role_codes = np.asarray(arrays["chaser_behavior_role_code"])[sl]
                active_codes = np.asarray(arrays["active_state_code"])[sl]
                active_valid = np.asarray(arrays["active_state_valid"])[sl]
                trial_ids = np.asarray(arrays["trial_id"])[sl]
                trial_valid = np.asarray(arrays["trial_valid"])[sl]
                yield {
                    **_common_columns(common, count),
                    "relative_frame_row_id": rows[sl],
                    "acquisition_frame_id": arrays["acquisition_frame_id"][sl],
                    "track_sample_id": arrays["track_sample_id"][sl],
                    "timestamp_ns_session": arrays["timestamp_ns"][sl],
                    "timestamp_valid": arrays["timestamp_valid"][sl],
                    "timestamp_reason_code": arrays["timestamp_reason_code"][sl],
                    "fish_source_row_id": arrays["fish_source_row_id"][sl],
                    "fish_source_row_valid": arrays["fish_source_row_valid"][sl],
                    "fish_source_row_reason_code": arrays[
                        "fish_source_row_reason_code"
                    ][sl],
                    "chaser_source_row_id": arrays["chaser_source_row_id"][sl],
                    "chaser_source_row_valid": arrays[
                        "chaser_source_row_valid"
                    ][sl],
                    "chaser_source_row_reason_code": arrays[
                        "chaser_source_row_reason_code"
                    ][sl],
                    "fish_position_x_px": arrays["fish_position_xy_px"][sl, 0],
                    "fish_position_y_px": arrays["fish_position_xy_px"][sl, 1],
                    "fish_position_valid": arrays["fish_position_valid"][sl],
                    "fish_position_reason_code": arrays[
                        "fish_position_reason_code"
                    ][sl],
                    "chaser_position_x_px": arrays["chaser_position_xy_px"][sl, 0],
                    "chaser_position_y_px": arrays["chaser_position_xy_px"][sl, 1],
                    "chaser_position_valid": arrays["chaser_position_valid"][sl],
                    "chaser_position_reason_code": arrays[
                        "chaser_position_reason_code"
                    ][sl],
                    "fish_identity_code": arrays["fish_identity_code"][sl],
                    "chaser_identity_code": chaser_codes,
                    "chaser_identity": [identities[int(code)] for code in chaser_codes],
                    "chaser_behavior_role_code": role_codes,
                    "behavior_role": [roles[int(code)] for code in chaser_codes],
                    "chaser_behavior_role_valid": arrays[
                        "chaser_behavior_role_valid"
                    ][sl],
                    "chaser_behavior_role_reason_code": arrays[
                        "chaser_behavior_role_reason_code"
                    ][sl],
                    "selection_member": arrays["selection_member"][sl],
                    "chaser_occurrence_member": arrays[
                        "chaser_occurrence_member"
                    ][sl],
                    "trial_id": [
                        int(value) if bool(valid) else None
                        for value, valid in zip(trial_ids, trial_valid, strict=True)
                    ],
                    "trial_valid": trial_valid,
                    "trial_reason_code": arrays["trial_reason_code"][sl],
                    "active_state": [
                        active_registry[int(code)] if bool(valid) else None
                        for code, valid in zip(
                            active_codes, active_valid, strict=True
                        )
                    ],
                    "active_state_valid": active_valid,
                    "active_state_reason_code": arrays[
                        "active_state_reason_code"
                    ][sl],
                    "row_valid": arrays["row_valid"][sl],
                    "row_reason_code": arrays["row_reason_code"][sl],
                    "acquisition_frame_delta": arrays[
                        "acquisition_frame_delta"
                    ][sl],
                    "timestamp_delta_ns": arrays["timestamp_delta_ns"][sl],
                    "fish_transition_valid": arrays["fish_transition_valid"][sl],
                    "fish_transition_reason_code": arrays[
                        "fish_transition_reason_code"
                    ][sl],
                    "relative_transition_valid": arrays[
                        "relative_transition_valid"
                    ][sl],
                    "relative_transition_reason_code": arrays[
                        "relative_transition_reason_code"
                    ][sl],
                    "relative_vector_x_px": arrays["relative_vector_px_xy"][sl, 0],
                    "relative_vector_y_px": arrays["relative_vector_px_xy"][sl, 1],
                    "relative_distance_px": arrays["relative_distance_px"][sl],
                    "relative_px_valid": arrays["relative_px_valid"][sl],
                    "relative_px_reason_code": arrays[
                        "relative_px_reason_code"
                    ][sl],
                    "relative_vector_x_mm": arrays[
                        "relative_vector_physical_xy"
                    ][sl, 0],
                    "relative_vector_y_mm": arrays[
                        "relative_vector_physical_xy"
                    ][sl, 1],
                    "relative_distance_mm": arrays[
                        "relative_distance_physical"
                    ][sl],
                    "relative_physical_valid": arrays[
                        "relative_physical_valid"
                    ][sl],
                    "relative_physical_reason_code": arrays[
                        "relative_physical_reason_code"
                    ][sl],
                    "nearest_chaser_member": arrays["nearest_chaser_member"][sl],
                    "nearest_chaser_identity_code": arrays[
                        "nearest_chaser_identity_code"
                    ][sl],
                    "nearest_chaser_source_row_id": arrays[
                        "nearest_chaser_source_row_id"
                    ][sl],
                    "nearest_chaser_distance_px": arrays[
                        "nearest_chaser_distance_px"
                    ][sl],
                    "nearest_chaser_distance_mm": arrays[
                        "nearest_chaser_distance_physical"
                    ][sl],
                    "nearest_chaser_valid": arrays["nearest_chaser_valid"][sl],
                    "nearest_chaser_reason_code": arrays[
                        "nearest_chaser_reason_code"
                    ][sl],
                }

    return ValidatedBehaviorBatchSource(
        batches=batches(),
        zero_row_reason="complete-no-rows" if row_count == 0 else None,
    )


def _body_frame_samples(context: _PhaseBContext) -> ValidatedBehaviorBatchSource:
    handle = context.relative("keypoint")
    if not handle.body_arrays:
        _fail("Anatomical body-frame capability lacks the relative body extension.")
    n_frames, n_chasers = handle.n_frames, handle.n_chasers
    first = np.arange(n_frames, dtype=np.int64) * n_chasers
    repeated_names = (
        "acquisition_frame_id",
        "track_sample_id",
        "timestamp_ns",
        "timestamp_valid",
    )
    body_names = (
        "body_source_row_id",
        "body_source_row_valid",
        "body_source_row_reason_code",
        "body_origin_xy_px",
        "body_forward_axis_xy",
        "body_left_axis_xy",
        "body_origin_valid",
        "body_origin_reason_code",
        "body_axes_valid",
        "body_axes_reason_code",
        "body_heading_deg",
        "body_heading_valid",
        "body_heading_reason_code",
        "body_heading_transition_valid",
        "body_heading_transition_reason_code",
        "body_valid",
        "body_reason_code",
    )
    for name in repeated_names:
        values = handle.base_frame_chaser(name)
        if not np.all(values == values[:, :1]):
            _fail(f"Body-frame base evidence {name!r} differs across chasers.")
    for name in body_names:
        values = handle.body_frame_chaser(name)
        left = values[:, :1]
        equal = (
            np.all(np.equal(values, left))
            if values.dtype.kind not in {"f", "c"}
            else np.all(np.equal(values, left) | (np.isnan(values) & np.isnan(left)))
        )
        if not bool(equal):
            _fail(f"Body-frame evidence {name!r} differs across chasers.")
    authority = context.bundle["source_bindings"]["anatomical_body_frame"][
        "authority"
    ]
    common = {
        **context.base.child_common("chaser_relative_keypoint"),
        "provider_role": "keypoint",
        "body_frame_provider_id": str(authority["provider_id"]),
        "body_frame_provider_digest": str(authority["provider_digest"]),
    }
    base = handle.base_arrays
    body = handle.body_arrays

    def build(start: int, stop: int) -> Mapping[str, Any]:
        count = stop - start
        index = first[start:stop]
        return {
            **_common_columns(common, count),
            "acquisition_frame_id": base["acquisition_frame_id"][index],
            "track_sample_id": base["track_sample_id"][index],
            "timestamp_ns_session": base["timestamp_ns"][index],
            "timestamp_valid": base["timestamp_valid"][index],
            "body_source_row_id": body["body_source_row_id"][index],
            "body_source_row_valid": body["body_source_row_valid"][index],
            "body_source_row_reason_code": body[
                "body_source_row_reason_code"
            ][index],
            "body_origin_x_px": body["body_origin_xy_px"][index, 0],
            "body_origin_y_px": body["body_origin_xy_px"][index, 1],
            "body_forward_x": body["body_forward_axis_xy"][index, 0],
            "body_forward_y": body["body_forward_axis_xy"][index, 1],
            "body_left_x": body["body_left_axis_xy"][index, 0],
            "body_left_y": body["body_left_axis_xy"][index, 1],
            "body_origin_valid": body["body_origin_valid"][index],
            "body_origin_reason_code": body["body_origin_reason_code"][index],
            "body_axes_valid": body["body_axes_valid"][index],
            "body_axes_reason_code": body["body_axes_reason_code"][index],
            "body_heading_deg": body["body_heading_deg"][index],
            "body_heading_valid": body["body_heading_valid"][index],
            "body_heading_reason_code": body["body_heading_reason_code"][index],
            "body_heading_transition_valid": body[
                "body_heading_transition_valid"
            ][index],
            "body_heading_transition_reason_code": body[
                "body_heading_transition_reason_code"
            ][index],
            "body_valid": body["body_valid"][index],
            "body_reason_code": body["body_reason_code"][index],
        }

    return _batch_source(n_frames, _batch_rows(context), build)


def _body_relative_samples(
    context: _PhaseBContext,
) -> ValidatedBehaviorBatchSource:
    handle = context.dense_child(
        "body_alignment_by_distance",
        successor_kind="chaser_body_alignment_by_distance",
        arrays=_BODY_ALIGNMENT_FRAME_ARRAYS,
    )
    arrays = handle.arrays
    row_count = int(np.asarray(arrays["frame_acquisition_frame_id"]).shape[0])
    identities, roles = context.chaser_maps()
    manifest = handle.scientific_manifest
    registries = manifest.get("identity_registries")
    if not isinstance(registries, Mapping):
        _fail("Body-alignment child lacks identity registries.")
    epoch_records = manifest.get("epoch_records")
    if not isinstance(epoch_records, Sequence) or isinstance(
        epoch_records, (str, bytes)
    ):
        _fail("Body-alignment child lacks exact epoch records.")
    try:
        epoch_registry = {
            int(item["epoch_role_code"]): str(item["analysis_role"])
            for item in epoch_records
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidatedBehaviorAdapterError(
            "Body-alignment epoch records are invalid."
        ) from exc
    if (
        len(epoch_registry) != len(epoch_records)
        or 0 in epoch_registry
        or any(not role for role in epoch_registry.values())
    ):
        _fail("Body-alignment epoch records are not unique and non-empty.")
    common = context.base.child_common("body_alignment_by_distance")
    row_ids = np.arange(row_count, dtype=np.int64)

    def build(start: int, stop: int) -> Mapping[str, Any]:
        count = stop - start
        sl = slice(start, stop)
        chaser_codes = np.asarray(arrays["frame_chaser_identity_code"])[sl]
        role_codes = np.asarray(arrays["frame_chaser_behavior_role_code"])[sl]
        role_valid = np.asarray(arrays["frame_chaser_behavior_role_valid"])[sl]
        epoch_codes = np.asarray(arrays["frame_epoch_role_code"])[sl]
        epoch_windows = np.asarray(arrays["frame_epoch_window_id"])[sl]
        return {
            **_common_columns(common, count),
            "body_alignment_row_id": row_ids[sl],
            "acquisition_frame_id": arrays["frame_acquisition_frame_id"][sl],
            "epoch_window_id": [
                int(value) if int(value) >= 0 else None for value in epoch_windows
            ],
            "epoch_role": [
                epoch_registry.get(int(code)) if int(code) != 0 else None
                for code in epoch_codes
            ],
            "selection_member": arrays["frame_selection_member"][sl],
            "chaser_occurrence_member": arrays[
                "frame_chaser_occurrence_member"
            ][sl],
            "chaser_identity_code": chaser_codes,
            "chaser_identity": [identities[int(code)] for code in chaser_codes],
            "chaser_behavior_role_code": role_codes,
            "behavior_role": [
                roles[int(chaser_code)] if bool(valid) else None
                for chaser_code, valid in zip(
                    chaser_codes, role_valid, strict=True
                )
            ],
            "chaser_behavior_role_valid": role_valid,
            "relative_distance_mm": arrays[
                "frame_relative_distance_physical"
            ][sl],
            "relative_physical_valid": arrays[
                "frame_relative_physical_valid"
            ][sl],
            "relative_physical_reason_code": arrays[
                "frame_relative_physical_reason_code"
            ][sl],
            "body_source_row_id": arrays["frame_body_source_row_id"][sl],
            "body_source_row_valid": arrays[
                "frame_body_source_row_valid"
            ][sl],
            "body_heading_deg": arrays["frame_body_heading_deg"][sl],
            "body_heading_valid": arrays["frame_body_heading_valid"][sl],
            "body_heading_reason_code": arrays[
                "frame_body_heading_reason_code"
            ][sl],
            "body_bearing_deg": arrays["frame_body_bearing_deg"][sl],
            "body_bearing_valid": arrays["frame_body_bearing_valid"][sl],
            "body_bearing_reason_code": arrays[
                "frame_body_bearing_reason_code"
            ][sl],
            "alignment_cos": arrays["frame_alignment_cos"][sl],
            "lateral_sin": arrays["frame_lateral_sin"][sl],
            "alignment_valid": arrays["frame_alignment_valid"][sl],
            "alignment_reason_code": arrays["frame_alignment_reason_code"][sl],
        }

    return _batch_source(row_count, _batch_rows(context), build)


def _stimulus_native_state_support(
    context: _PhaseBContext,
) -> ValidatedBehaviorBatchSource:
    proxy = context.proxy()
    relative_binding = context.base.child_common("chaser_relative_keypoint")
    arrays = proxy.arrays
    counts = np.asarray(arrays["candidate_sample_count"], dtype=np.int64)
    frames = np.asarray(arrays["acquisition_frame_index"], dtype=np.int64)
    candidate_frames = np.repeat(frames, counts)
    candidate_ordinals = np.concatenate(
        [np.arange(int(count), dtype=np.int32) for count in counts]
    )
    native_rows = np.asarray(arrays["candidate_native_sample_row_index"])
    n_candidates = int(native_rows.size)
    n_chasers = int(proxy.dimensions.n_chasers)
    if candidate_frames.shape != native_rows.shape:
        _fail("Proxy candidate offsets do not close the native sample axis.")
    selected_native = np.repeat(
        np.asarray(arrays["selected_native_sample_row_index"]), counts
    )
    selected_candidate = native_rows == selected_native
    identities, roles = context.chaser_maps()
    chaser_codes = np.arange(1, n_chasers + 1, dtype=np.int32)
    occurrence = context.bundle["source_bindings"]["row_axis_timing_and_scale"][
        "authority"
    ]["chaser_occurrence"]["chasers"]
    expected_indices = np.asarray(
        [int(item["chaser_index"]) for item in occurrence], dtype=np.int32
    )
    if not np.array_equal(expected_indices, np.arange(n_chasers, dtype=np.int32)):
        _fail("Proxy chaser axis differs from the sealed occurrence registry.")
    projection = proxy.acquisition_projection_record
    reason_registry = {2: "complete", 3: "incomplete_chaser_sample"}
    expanded_count = n_candidates * n_chasers
    common = {
        **relative_binding,
        "temporal_proxy_role": "keypoint",
        "temporal_proxy_run_path": proxy.run_path,
        "temporal_proxy_manifest_sha256": proxy.manifest_sha256,
        "temporal_proxy_verification_digest": proxy.verification_digest,
        "acquisition_projection_record_sha256": (
            proxy.acquisition_projection_record_sha256
        ),
        "source_stimulus_run_path": str(projection["source_run_path"]),
        "source_stimulus_manifest_sha256": str(
            projection["source_manifest_sha256"]
        ),
        "source_stimulus_verification_digest": str(
            projection["source_verification_digest"]
        ),
        "projection_policy_id": str(projection["policy_id"]),
        "scientific_use_class": str(projection["scientific_use_class"]),
        "physical_presentation_verified": bool(
            projection["physical_presentation_verified"]
        ),
    }
    expanded = {
        "acquisition_frame_id": np.repeat(candidate_frames, n_chasers),
        "candidate_ordinal_within_frame": np.repeat(
            candidate_ordinals, n_chasers
        ),
        "native_sample_row_id": np.repeat(native_rows, n_chasers),
        "stimulus_frame_num": np.repeat(
            arrays["candidate_stimulus_frame_num"], n_chasers
        ),
        "timestamp_ns_session": np.repeat(
            arrays["candidate_timestamp_ns_session"], n_chasers
        ),
        "source_acquisition_frame_id": np.repeat(
            arrays["candidate_source_acquisition_frame_index"], n_chasers
        ),
        "candidate_complete": np.repeat(arrays["candidate_complete"], n_chasers),
        "candidate_reason_code": np.repeat(
            arrays["candidate_reason_code"], n_chasers
        ),
        "selected_candidate": np.repeat(selected_candidate, n_chasers),
        "chaser_identity_code": np.tile(chaser_codes, n_candidates),
        "chaser_index": np.tile(expected_indices, n_candidates),
        "source_stimulus_run_row_id": np.asarray(
            arrays["candidate_source_stimulus_run_row_index"]
        ).reshape(-1),
        "source_stimulus_source_row_id": np.asarray(
            arrays["candidate_source_stimulus_source_row_index"]
        ).reshape(-1),
    }

    def build(start: int, stop: int) -> Mapping[str, Any]:
        count = stop - start
        sl = slice(start, stop)
        codes = expanded["chaser_identity_code"][sl]
        reason_codes = expanded["candidate_reason_code"][sl]
        return {
            **_common_columns(common, count),
            **{name: values[sl] for name, values in expanded.items()},
            "candidate_reason": [
                reason_registry[int(code)] for code in reason_codes
            ],
            "chaser_identity": [identities[int(code)] for code in codes],
            "behavior_role": [roles[int(code)] for code in codes],
        }

    return _batch_source(expanded_count, _batch_rows(context), build)


def _controller_dense_source(
    context: _PhaseBContext,
) -> tuple[Any, Any, dict[int, int], dict[int, str], dict[int, str], dict[int, str]]:
    controller = context.dense_child(
        "controller_trials",
        successor_kind="controller_chase_trials",
        arrays=_CONTROLLER_DENSE_ARRAYS,
    )
    relative = context.relative("keypoint")
    source_rows = np.asarray(controller.arrays["source_relative_row_id"])
    if (
        source_rows.shape != (relative.n_rows,)
        or not np.array_equal(source_rows, np.arange(relative.n_rows))
    ):
        _fail("Controller dense evidence differs from the exact relative row axis.")
    trial_rows = np.asarray(controller.arrays["trial_row_id"])
    logged_ids = np.asarray(controller.arrays["logged_trial_id"])
    if trial_rows.shape != logged_ids.shape or np.unique(trial_rows).size != trial_rows.size:
        _fail("Controller compact trial identity table is invalid.")
    trial_ids = {
        int(row): int(logged)
        for row, logged in zip(trial_rows, logged_ids, strict=True)
    }
    identities, roles = context.chaser_maps()
    registries = controller.scientific_manifest.get("identity_registries")
    if not isinstance(registries, Mapping):
        _fail("Controller child lacks identity registries.")
    gaps = _registry(registries["trial_gap_reason"], field="controller.gap_reason")
    return controller, relative, trial_ids, identities, roles, gaps


def _controller_trial_membership(
    context: _PhaseBContext,
) -> ValidatedBehaviorBatchSource:
    controller, relative, trial_ids, identities, roles, _gaps = (
        _controller_dense_source(context)
    )
    arrays = controller.arrays
    member = np.asarray(arrays["logged_active_trial_member"], dtype=bool)
    indices = np.flatnonzero(member).astype(np.int64, copy=False)
    dense_trials = np.asarray(arrays["trial_row_id_by_source_row"])[indices]
    if np.any(dense_trials < 0):
        _fail("Active controller members lack exact trial identities.")
    base = relative.base_arrays
    common = context.base.child_common("controller_trials")

    def build(start: int, stop: int) -> Mapping[str, Any]:
        count = stop - start
        selected = indices[start:stop]
        trial_rows = dense_trials[start:stop]
        chaser_codes = np.asarray(base["chaser_identity_code"])[selected]
        return {
            **_common_columns(common, count),
            "source_relative_row_id": selected,
            "acquisition_frame_id": base["acquisition_frame_id"][selected],
            "chaser_identity_code": chaser_codes,
            "chaser_identity": [identities[int(code)] for code in chaser_codes],
            "behavior_role": [roles[int(code)] for code in chaser_codes],
            "trial_row_id": trial_rows,
            "logged_trial_id": [trial_ids[int(row)] for row in trial_rows],
            "logged_active_trial_id_unavailable": arrays[
                "logged_active_trial_id_unavailable"
            ][selected],
        }

    return _batch_source(int(indices.size), _batch_rows(context), build)


def _controller_trial_gap_evidence(
    context: _PhaseBContext,
) -> ValidatedBehaviorBatchSource:
    controller, relative, trial_ids, identities, roles, gaps = (
        _controller_dense_source(context)
    )
    arrays = controller.arrays
    member = np.asarray(arrays["trial_gap_member"], dtype=bool)
    indices = np.flatnonzero(member).astype(np.int64, copy=False)
    envelope_rows = np.asarray(arrays["trial_envelope_row_id_by_source_row"])[
        indices
    ]
    reason_codes = np.asarray(arrays["trial_gap_reason_code_by_source_row"])[
        indices
    ]
    if np.any(envelope_rows < 0) or np.any(reason_codes == 0):
        _fail("Controller gap evidence lacks envelope or reason identity.")
    base = relative.base_arrays
    common = context.base.child_common("controller_trials")

    def build(start: int, stop: int) -> Mapping[str, Any]:
        count = stop - start
        selected = indices[start:stop]
        trial_rows = envelope_rows[start:stop]
        codes = reason_codes[start:stop]
        chaser_codes = np.asarray(base["chaser_identity_code"])[selected]
        return {
            **_common_columns(common, count),
            "source_relative_row_id": selected,
            "acquisition_frame_id": base["acquisition_frame_id"][selected],
            "chaser_identity_code": chaser_codes,
            "chaser_identity": [identities[int(code)] for code in chaser_codes],
            "behavior_role": [roles[int(code)] for code in chaser_codes],
            "trial_envelope_row_id": trial_rows,
            "logged_trial_id": [trial_ids[int(row)] for row in trial_rows],
            "trial_gap_reason_code": codes,
            "trial_gap_reason": [gaps[int(code)] for code in codes],
            "logged_active_trial_id_unavailable": arrays[
                "logged_active_trial_id_unavailable"
            ][selected],
        }

    return _batch_source(int(indices.size), _batch_rows(context), build)


_PRODUCERS: Mapping[str, Callable[[_PhaseBContext], ValidatedBehaviorBatchSource]] = (
    MappingProxyType(
        {
            "provider_motion_samples": _provider_motion_samples,
            "bout_detector_signal_samples": _bout_detector_signal_samples,
            "stimulus_native_state_support": _stimulus_native_state_support,
            "chaser_relative_samples": _chaser_relative_samples,
            "body_frame_samples": _body_frame_samples,
            "body_relative_samples": _body_relative_samples,
            "controller_trial_membership": _controller_trial_membership,
            "controller_trial_gap_evidence": _controller_trial_gap_evidence,
        }
    )
)


def build_phase_b_dense_row_extractors() -> Mapping[str, Callable[..., Any]]:
    """Return dense extractors sharing one recording-scoped source cache."""

    cache = _LastPhaseBContext()

    def wrap(
        producer: Callable[[_PhaseBContext], ValidatedBehaviorBatchSource],
    ) -> Callable[..., Any]:
        def extract(
            plan: Mapping[str, Any],
            membership_member: Mapping[str, Any],
            bundle_member: Mapping[str, Any],
        ) -> ValidatedBehaviorBatchSource:
            return producer(cache.get(plan, membership_member, bundle_member))

        return extract

    return MappingProxyType(
        {name: wrap(producer) for name, producer in _PRODUCERS.items()}
    )


__all__ = [
    "build_phase_b_dense_row_extractors",
]
