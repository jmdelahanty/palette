"""Bundle-backed row adapters for the compact validated-behavior profile.

The adapter consumes only exact bindings already selected by a validated
recording-behavior bundle.  Compact scientific arrays are loaded through their
receipt-aware source handles with a closed requested-array roster, so export
does not repeat a whole-child audit and does not discover selectors.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .arrow_contract_core import canonical_bytes
from .publication import sha256_file

if TYPE_CHECKING:
    from fisheye.analysis_workflows.composable_chaser_successor_publication import (
        ComposableChaserSuccessorSourceHandle,
    )
    from fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle import (
        ProviderEpochBehaviorSummarySourceHandle,
    )


class ValidatedBehaviorAdapterError(ValueError):
    """One bundle-selected table source is absent, changed, or inconsistent."""


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


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one exact non-empty string.")
    return value


def _scalar(value: Any) -> Any:
    item = value.item() if isinstance(value, np.generic) else value
    if isinstance(item, bytes):
        return item.rstrip(b"\x00").decode("utf-8")
    return item


def _registry(manifest: Mapping[str, Any], name: str) -> dict[int, str]:
    raw = _mapping(
        _mapping(manifest.get("identity_registries"), field="identity_registries").get(
            name
        ),
        field=f"identity_registries.{name}",
    )
    try:
        result = {
            int(key): _text(value, field=f"{name}[{key}]") for key, value in raw.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValidatedBehaviorAdapterError(
            f"Identity registry {name!r} has a non-integer code."
        ) from exc
    if len(result) != len(raw):
        _fail(f"Identity registry {name!r} has duplicate integer codes.")
    return result


def _decode(registry: Mapping[int, str], code: Any, *, field: str) -> str:
    key = int(_scalar(code))
    try:
        return registry[key]
    except KeyError as exc:
        raise ValidatedBehaviorAdapterError(
            f"{field} code {key} is absent from its sealed registry."
        ) from exc


def _decode_optional_zero(
    registry: Mapping[int, str], code: Any, *, field: str
) -> str | None:
    key = int(_scalar(code))
    if key == 0 and key not in registry:
        return None
    return _decode(registry, key, field=field)


def _array_rows(
    arrays: Mapping[str, np.ndarray], names: Sequence[str]
) -> list[dict[str, Any]]:
    requested = tuple(names)
    if not requested:
        return []
    missing = sorted(set(requested) - set(arrays))
    if missing:
        _fail(f"Targeted child handle lacks arrays: {missing!r}.")
    lengths = {int(np.asarray(arrays[name]).shape[0]) for name in requested}
    if len(lengths) != 1:
        _fail("Scientific table arrays do not share one row axis.")
    count = lengths.pop()
    return [
        {name: _scalar(np.asarray(arrays[name])[index]) for name in requested}
        for index in range(count)
    ]


_CONTROLLER_ARRAYS = (
    "active_member_count",
    "chaser_identity_code",
    "end_acquisition_frame_id_inclusive",
    "end_source_frame_row_exclusive",
    "envelope_frame_count",
    "fallback_used",
    "gap_fraction",
    "gap_frame_count",
    "logged_trial_id",
    "start_acquisition_frame_id",
    "start_source_frame_row",
    "trial_ordinal",
    "trial_row_id",
    "trigger_acquisition_frame_id",
    "trigger_source_code",
    "trigger_timestamp_ns",
    "trigger_timestamp_valid",
)

_BOUT_ASSOCIATION_ARRAYS = (
    "attachment_reason_code",
    "base_valid",
    "bearing_at_onset_deg",
    "bout_chaser_row_id",
    "bout_duration_s",
    "bout_id",
    "bout_mean_speed_mm_s",
    "bout_net_displacement_mm",
    "bout_path_length_mm",
    "bout_peak_speed_mm_s",
    "bout_row_id",
    "bout_tortuosity",
    "chaser_identity_code",
    "controller_trial_envelope_row_id",
    "controller_trial_gap_reason_code",
    "controller_trial_row_id",
    "delta_distance_mm",
    "directed_valid",
    "distance_at_end_mm",
    "distance_at_onset_mm",
    "end_acquisition_frame_id",
    "semantic_role_code",
    "source_signal_id",
    "start_acquisition_frame_id",
    "turn_deg",
    "turn_toward_chaser",
)

_BOUT_SUMMARY_ARRAYS = (
    "summary_bout_count",
    "summary_bout_rate_per_min",
    "summary_chaser_identity_code",
    "summary_distance_bin_end_mm",
    "summary_distance_bin_index",
    "summary_distance_bin_start_mm",
    "summary_median_duration_s",
    "summary_median_net_displacement_mm",
    "summary_median_path_length_mm",
    "summary_median_peak_speed_mm_s",
    "summary_role_code",
    "summary_valid_time_s",
)

_ESCAPE_EVENT_ARRAYS = (
    "event_bout_id",
    "event_bout_row_id",
    "event_chaser_identity_code",
    "event_controller_trial_row_id",
    "event_directed_valid",
    "event_distance_at_onset_mm",
    "event_high_turn",
    "event_latency_from_trigger_s",
    "event_onset_acquisition_frame_id",
    "event_peak_speed_mm_s",
    "event_recapture_latency_s",
    "event_recaptured",
    "event_row_id",
    "event_separation_gain_mm",
    "event_source_bout_chaser_row_id",
    "event_trace_exclusion_reason_code",
    "event_trace_valid",
    "event_trigger_distance_mm",
    "event_turn_deg",
)

_ESCAPE_TRIAL_ARRAYS = (
    "trial_bout_count",
    "trial_chaser_identity_code",
    "trial_envelope_frame_count",
    "trial_escape_event_count",
    "trial_escape_event_rate_per_min",
    "trial_escape_speed_class",
    "trial_first_escape_latency_s",
    "trial_freeze_candidate",
    "trial_freeze_low_speed_fraction",
    "trial_freeze_valid_fraction",
    "trial_gap_fraction",
    "trial_gap_frame_count",
    "trial_high_turn_escape_count",
    "trial_logged_active_id_unavailable_count",
    "trial_logged_id",
    "trial_mean_separation_gain_mm",
    "trial_ordinal",
    "trial_recapture_fraction",
    "trial_response_class_code",
    "trial_row_id",
    "trial_trigger_acquisition_frame_id",
    "trial_trigger_distance_mm",
    "trial_valid_time_s",
)

_ESCAPE_SWEEP_ARRAYS = (
    "sweep_escape_event_count",
    "sweep_escape_event_rate_per_min",
    "sweep_row_id",
    "sweep_speed_threshold_mm_s",
    "sweep_trial_row_id",
)

_SPATIAL_ARRAYS = (
    "arena_bin_center_mask",
    "candidate_frame_count",
    "declared_valid_position_frame_count",
    "epoch_end_frame_exclusive",
    "epoch_role_code",
    "epoch_start_frame",
    "epoch_window_id",
    "finite_valid_position_frame_count",
    "in_arena_coverage_fraction_candidate",
    "in_arena_fraction_finite_valid",
    "in_arena_position_frame_count",
    "invalid_position_frame_count",
    "occupancy_count",
    "occupancy_density_valid_in_arena",
    "occupancy_fraction_candidate_epoch",
    "out_of_arena_position_frame_count",
    "provider_role_code",
    "x_bin_edges_mm",
    "y_bin_edges_mm",
)

_RADIAL_METRIC_ARRAYS = (
    "metric_behavior_role_code",
    "metric_candidate_frame_count",
    "metric_chaser_identity_code",
    "metric_distance_mean_mm",
    "metric_distance_p05_mm",
    "metric_distance_p25_mm",
    "metric_distance_p50_mm",
    "metric_distance_p75_mm",
    "metric_distance_p95_mm",
    "metric_epoch_role_code",
    "metric_epoch_window_id",
    "metric_fish_arena_radius_mean_mm",
    "metric_fish_arena_radius_p50_mm",
    "metric_fish_wall_distance_mean_mm",
    "metric_fish_wall_distance_p50_mm",
    "metric_near_zone_boundary_censor_event_count",
    "metric_near_zone_censor_event_count",
    "metric_near_zone_complete_visit_median_dwell_s",
    "metric_near_zone_complete_visit_total_dwell_s",
    "metric_near_zone_dwell_s",
    "metric_near_zone_enrichment_geometric",
    "metric_near_zone_entry_count",
    "metric_near_zone_entry_rate_per_min_valid_time",
    "metric_near_zone_expected_fraction_geometric",
    "metric_near_zone_fraction_candidate",
    "metric_near_zone_fraction_valid",
    "metric_near_zone_frame_count",
    "metric_near_zone_invalid_gap_censor_event_count",
    "metric_near_zone_invalid_gap_count",
    "metric_near_zone_valid_tracked_duration_s",
    "metric_same_quadrant_fraction_candidate",
    "metric_same_quadrant_fraction_valid",
    "metric_same_quadrant_valid_frame_count",
    "metric_valid_distance_fraction",
    "metric_valid_distance_frame_count",
    "metric_wall_excluded_valid_frame_count",
)

_RADIAL_BIN_ARRAYS = (
    "radial_behavior_role_code",
    "radial_bin_end_mm",
    "radial_bin_start_mm",
    "radial_chaser_identity_code",
    "radial_epoch_role_code",
    "radial_epoch_window_id",
    "radial_expected_available_area_mm2_frames",
    "radial_expected_fraction_geometric",
    "radial_observed_count",
    "radial_observed_fraction",
    "radial_selection_index_geometric",
    "radial_wall_excluded_expected_available_area_mm2_frames",
    "radial_wall_excluded_expected_fraction_geometric",
    "radial_wall_excluded_observed_count",
    "radial_wall_excluded_observed_fraction",
    "radial_wall_excluded_selection_index_geometric",
)

_RADIAL_CDF_ARRAYS = (
    "cdf_behavior_role_code",
    "cdf_chaser_identity_code",
    "cdf_epoch_role_code",
    "cdf_epoch_window_id",
    "cdf_fraction_at_or_below",
    "cdf_threshold_mm",
)

_BODY_SUMMARY_ARRAYS = (
    "summary_abs_bearing_p25_deg",
    "summary_abs_bearing_p50_deg",
    "summary_abs_bearing_p75_deg",
    "summary_alignment_cos_p25",
    "summary_alignment_cos_p50",
    "summary_alignment_cos_p75",
    "summary_body_bearing_invalid_row_count",
    "summary_body_heading_invalid_row_count",
    "summary_body_source_missing_row_count",
    "summary_candidate_row_count",
    "summary_chaser_behavior_role_code",
    "summary_chaser_identity_code",
    "summary_circular_mean_bearing_deg",
    "summary_circular_resultant_length",
    "summary_distance_bin_center_mm",
    "summary_distance_bin_end_mm",
    "summary_distance_bin_index",
    "summary_distance_bin_start_mm",
    "summary_epoch_chaser_absent_row_count",
    "summary_epoch_distance_invalid_body_valid_row_count",
    "summary_epoch_distance_invalid_row_count",
    "summary_epoch_distance_valid_row_count",
    "summary_epoch_occurrence_row_count",
    "summary_epoch_role_code",
    "summary_epoch_window_id",
    "summary_joint_valid_row_count",
    "summary_mean_abs_bearing_deg",
    "summary_mean_alignment_cos",
    "summary_other_alignment_invalid_row_count",
)

_EPOCH_FISH_ARRAYS = (
    "track_id",
    "window_id",
    "window_index",
    "window_label",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "total_span_frames",
    "provider_sample_count",
    "valid_tracked_frame_count",
    "missing_frame_count",
    "tracking_dropout_fraction",
    "valid_tracked_duration_s",
    "motion_valid_sample_count",
    "speed_sample_count",
    "mean_speed_mm_s",
    "median_speed_mm_s",
    "p05_speed_mm_s",
    "p95_speed_mm_s",
    "max_speed_mm_s",
    "total_path_mm",
    "bout_count",
    "bout_rate_per_min",
    "median_bout_duration_s",
    "mean_bout_duration_s",
    "median_bout_path_length_mm",
    "mean_bout_path_length_mm",
    "bout_heading_sample_count",
    "mean_bout_net_heading_change_deg",
    "median_bout_net_heading_change_deg",
    "mean_abs_bout_net_heading_change_deg",
    "median_abs_bout_net_heading_change_deg",
    "mean_bout_heading_path_deg",
    "median_bout_heading_path_deg",
    "inter_bout_interval_count",
    "mean_inter_bout_interval_s",
    "median_inter_bout_interval_s",
    "p05_inter_bout_interval_s",
    "p95_inter_bout_interval_s",
    "inter_bout_interval_rate_per_min",
    "rate_denominator",
    "motion_validity_rule",
    "analysis_role",
    "source_interval_sha256",
    "protocol_semantic_hash",
    "protocol_semantic_step_index",
    "protocol_semantic_step_ref",
)

_COMPOSABLE_CHILDREN: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        "controller_trials": ("controller_chase_trials", _CONTROLLER_ARRAYS),
        "generalized_bout_response": (
            "generalized_chaser_bout_response",
            _BOUT_ASSOCIATION_ARRAYS + _BOUT_SUMMARY_ARRAYS,
        ),
        "escape_freeze": (
            "chaser_escape_freeze",
            _ESCAPE_EVENT_ARRAYS + _ESCAPE_TRIAL_ARRAYS + _ESCAPE_SWEEP_ARRAYS,
        ),
        "spatial_occupancy": ("chaser_spatial_occupancy", _SPATIAL_ARRAYS),
        "radial_near_field_keypoint": (
            "chaser_radial_near_field",
            _RADIAL_METRIC_ARRAYS + _RADIAL_BIN_ARRAYS + _RADIAL_CDF_ARRAYS,
        ),
        "radial_near_field_detection": (
            "chaser_radial_near_field",
            _RADIAL_METRIC_ARRAYS + _RADIAL_BIN_ARRAYS + _RADIAL_CDF_ARRAYS,
        ),
        "body_alignment_by_distance": (
            "chaser_body_alignment_by_distance",
            _BODY_SUMMARY_ARRAYS,
        ),
    }
)


class _RecordingContext:
    def __init__(
        self,
        plan: Mapping[str, Any],
        membership_member: Mapping[str, Any],
        bundle_member: Mapping[str, Any],
    ) -> None:
        if bundle_member.get("bundle_state") != "complete":
            _fail("Scientific extraction requires one complete recording bundle.")
        bundle_binding = _mapping(bundle_member.get("bundle"), field="bundle member")
        bundle_path = Path(
            _text(bundle_binding.get("path"), field="bundle path")
        ).resolve()
        if not bundle_path.is_file() or sha256_file(bundle_path) != bundle_binding.get(
            "file_sha256"
        ):
            _fail("Recording bundle file is absent or differs from the bundle set.")
        from fisheye.analysis_workflows.validated_recording_behavior_bundle import (
            read_validated_recording_behavior_bundle,
        )

        bundle = read_validated_recording_behavior_bundle(
            bundle_path,
            expected_analysis_zarr=membership_member["analysis_zarr"],
            expected_recording_id=membership_member["recording_id"],
            validate_current_sources=False,
        )
        if (
            bundle["record_sha256"] != bundle_binding.get("record_sha256")
            or bundle["schema_id"] != bundle_binding.get("schema_id")
            or bundle["schema_version"] != bundle_binding.get("schema_version")
            or bundle["method_id"] != bundle_binding.get("method_id")
            or bundle["status"] != bundle_binding.get("status")
        ):
            _fail("Recording bundle identity differs from its bundle-set member.")
        for capability, normalized in bundle_member["capabilities"].items():
            raw = bundle["capabilities"][capability]
            expected_binding = (
                None
                if raw["state"] != "complete"
                else {"scope": raw["binding_scope"], "key": raw["binding_key"]}
            )
            if (
                normalized["state"] != raw["state"]
                or normalized["reason_code"] != raw["reason_code"]
                or normalized["detail"] != raw["detail"]
                or _plain(normalized["binding"]) != expected_binding
            ):
                _fail(
                    f"Capability {capability!r} differs between bundle and bundle set."
                )
        self.plan = plan
        self.membership_member = membership_member
        self.bundle_member = bundle_member
        self.bundle = bundle
        self.bundle_path = bundle_path
        self.analysis_zarr = Path(bundle["analysis_zarr"])
        self.recording_id = str(bundle["recording_id"])
        self._child_handles: dict[str, ComposableChaserSuccessorSourceHandle] = {}
        self._epoch_handle: ProviderEpochBehaviorSummarySourceHandle | None = None
        self._identity_maps: tuple[dict[int, str], dict[int, str]] | None = None

    @property
    def bundle_common(self) -> dict[str, Any]:
        return {
            "export_run_id": str(self.plan["export_run_id"]),
            "recording_id": self.recording_id,
            "membership_member_sha256": str(self.membership_member["member_sha256"]),
            "bundle_set_member_sha256": str(self.bundle_member["member_sha256"]),
            "bundle_record_sha256": str(self.bundle["record_sha256"]),
        }

    def require_capability(self, capability: str) -> None:
        record = self.bundle["capabilities"].get(capability)
        if not isinstance(record, Mapping) or record.get("state") != "complete":
            _fail(f"Required bundle capability {capability!r} is not complete.")

    def child_binding(self, key: str) -> Mapping[str, Any]:
        self.require_capability(key)
        try:
            return _mapping(
                self.bundle["scientific_child_bindings"][key],
                field=f"scientific_child_bindings.{key}",
            )
        except KeyError as exc:
            raise ValidatedBehaviorAdapterError(
                f"Bundle lacks complete scientific child {key!r}."
            ) from exc

    def child_common(self, key: str) -> dict[str, Any]:
        binding = self.child_binding(key)
        return {
            **self.bundle_common,
            "source_child_key": key,
            "source_run_path": str(binding["run_path"]),
            "source_manifest_sha256": str(binding["manifest_sha256"]),
            "source_payload_sha256": str(binding["payload_digest"]),
            "source_receipt_sha256": str(binding["receipt_sha256"]),
        }

    def composable_child(self, key: str) -> ComposableChaserSuccessorSourceHandle:
        if key in self._child_handles:
            return self._child_handles[key]
        try:
            successor_kind, arrays = _COMPOSABLE_CHILDREN[key]
        except KeyError as exc:
            raise ValidatedBehaviorAdapterError(
                f"No compact composable child adapter is installed for {key!r}."
            ) from exc
        binding = self.child_binding(key)
        run_path = _text(binding.get("run_path"), field=f"{key}.run_path")
        run_name = run_path.rsplit("/", 1)[-1]
        from fisheye.analysis_workflows.composable_chaser_successor_publication import (
            load_composable_chaser_successor_source_handle,
        )

        handle = load_composable_chaser_successor_source_handle(
            self.analysis_zarr,
            successor_kind=successor_kind,
            run_name=run_name,
            expected_recording_id=self.recording_id,
            direct_validation_receipt=binding["receipt_path"],
            required_array_names=arrays,
        )
        observed = {
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "payload_digest": str(handle.manifest["payload_digest"]),
            "receipt_sha256": handle.receipt_digest,
        }
        expected = {
            "run_path": run_path,
            "manifest_sha256": binding["manifest_sha256"],
            "payload_digest": binding["payload_digest"],
            "receipt_sha256": binding["receipt_sha256"],
        }
        if observed != expected:
            changed = sorted(
                name for name in expected if observed[name] != expected[name]
            )
            _fail(
                f"Targeted child {key!r} differs from the recording bundle at "
                f"{changed!r}."
            )
        handle.require_verified_arrays(arrays)
        self._child_handles[key] = handle
        return handle

    def epoch_handle(self) -> ProviderEpochBehaviorSummarySourceHandle:
        if self._epoch_handle is not None:
            return self._epoch_handle
        binding = self.child_binding("epoch_behavior")
        run_path = _text(binding.get("run_path"), field="epoch_behavior.run_path")
        paths = tuple(f"per_epoch_fish/{name}" for name in _EPOCH_FISH_ARRAYS)
        from fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle import (
            load_provider_epoch_behavior_summary_source_handle,
        )

        handle = load_provider_epoch_behavior_summary_source_handle(
            self.analysis_zarr,
            run_name=run_path.rsplit("/", 1)[-1],
            expected_recording_id=self.recording_id,
            direct_validation_receipt=binding["receipt_path"],
            required_array_paths=paths,
        )
        if (
            handle.run_path != run_path
            or handle.manifest_sha256 != binding["manifest_sha256"]
            or handle.payload_digest != binding["payload_digest"]
            or handle.receipt_digest != binding["receipt_sha256"]
        ):
            _fail("Targeted epoch-behavior child differs from the recording bundle.")
        handle.require_verified_arrays(paths)
        self._epoch_handle = handle
        return handle

    def chaser_identity_maps(self) -> tuple[dict[int, str], dict[int, str]]:
        if self._identity_maps is not None:
            return self._identity_maps
        binding = self.child_binding("chaser_relative_keypoint")
        from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
            read_chaser_relative_frame_validation_receipt,
        )

        receipt = read_chaser_relative_frame_validation_receipt(
            binding["receipt_path"],
            expected_analysis_zarr=self.analysis_zarr,
            expected_recording_id=self.recording_id,
            expected_run_name=str(binding["run_path"]).rsplit("/", 1)[-1],
            expected_manifest_sha256=binding["manifest_sha256"],
            validate_current_metadata=False,
        )
        if (
            receipt["record_sha256"] != binding["receipt_sha256"]
            or receipt["payload_digest"] != binding["payload_digest"]
        ):
            _fail("Relative-frame identity receipt differs from the bundle binding.")
        identities = _registry(receipt["run_manifest"], "chaser")
        roles = _registry(receipt["run_manifest"], "behavior_role")
        occurrence = self.bundle["source_bindings"]["row_axis_timing_and_scale"][
            "authority"
        ]["chaser_occurrence"]
        expected_identities = {
            int(item["chaser_index"]) + 1: str(item["identity"])
            for item in occurrence["chasers"]
        }
        expected_roles = {
            int(item["chaser_index"]) + 1: str(item["behavior_role"])
            for item in occurrence["chasers"]
        }
        if identities != expected_identities or roles != expected_roles:
            _fail(
                "Bundle chaser occurrences differ from the sealed identity registries."
            )
        self._identity_maps = identities, roles
        return self._identity_maps


class _LastRecordingContext:
    """Keep only the current shard's compact handles during serial canaries."""

    def __init__(self) -> None:
        self._key: tuple[str, str, str] | None = None
        self._context: _RecordingContext | None = None

    def get(
        self,
        plan: Mapping[str, Any],
        membership_member: Mapping[str, Any],
        bundle_member: Mapping[str, Any],
    ) -> _RecordingContext:
        key = (
            str(plan["plan_sha256"]),
            str(membership_member["member_sha256"]),
            str(bundle_member["member_sha256"]),
        )
        if self._key != key:
            self._context = _RecordingContext(plan, membership_member, bundle_member)
            self._key = key
        assert self._context is not None
        return self._context


def _complete_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    return (rows, None) if rows else ([], "complete-no-rows")


def _recording_source_bindings(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    rows = []
    for key in sorted(context.bundle["source_bindings"]):
        binding = _plain(context.bundle["source_bindings"][key])
        rows.append(
            {
                **context.bundle_common,
                "source_binding_key": key,
                "binding_type": str(binding["binding_type"]),
                "binding_record_sha256": canonical_json_sha256(binding),
                "binding_json": canonical_bytes(binding).decode("ascii"),
            }
        )
    return _complete_rows(rows)


def _position_providers(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    context.require_capability("reviewed_arena_and_scale")
    rows = []
    for role, key in (
        ("keypoint", "fish_position_keypoint"),
        ("detection", "fish_position_detection"),
    ):
        context.require_capability(key)
        authority = context.bundle["source_bindings"][key]["authority"]
        rows.append(
            {
                **context.bundle_common,
                "provider_role": role,
                "position_provider_id": str(authority["provider_id"]),
                "position_provider_digest": str(authority["provider_digest"]),
                "source_authority_id": str(authority["source_authority_id"]),
                "source_authority_digest": str(authority["source_digest"]),
                "coordinate_authority_id": str(authority["coordinate_authority_id"]),
                "row_axis_authority_id": str(authority["row_axis_authority_id"]),
                "row_axis_authority_digest": str(
                    authority["row_axis_authority_digest"]
                ),
                "timing_authority_id": str(authority["timing_authority_id"]),
                "scale_authority_id": str(authority["scale_authority_id"]),
            }
        )
    return _complete_rows(rows)


def _chaser_occurrences(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    identities, roles = context.chaser_identity_maps()
    occurrence = context.bundle["source_bindings"]["row_axis_timing_and_scale"][
        "authority"
    ]["chaser_occurrence"]
    rows = []
    for item in sorted(occurrence["chasers"], key=lambda value: value["chaser_index"]):
        code = int(item["chaser_index"]) + 1
        rows.append(
            {
                **context.bundle_common,
                "chaser_identity_code": code,
                "chaser_index": int(item["chaser_index"]),
                "chaser_identity": identities[code],
                "behavior_role": roles[code],
                "stimulus_run_path": str(occurrence["source_stimulus_run_path"]),
                "source_protocol_sha256": str(occurrence["source_protocol_sha256"]),
                "chaser_identity_policy_id": str(
                    occurrence["chaser_identity_policy_id"]
                ),
                "occurrence_policy_id": str(occurrence["occurrence_policy_id"]),
                "occurrence_semantics": str(occurrence["semantics"]),
            }
        )
    return _complete_rows(rows)


def _semantic_epochs(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    common = context.child_common("semantic_epochs")
    source = context.bundle["source_bindings"]["semantic_epochs"]["source"]
    windows = list(source["position_suite_epochs"])
    bindings = {
        int(item["source_window_id"]): item for item in source["semantic_role_bindings"]
    }
    if set(bindings) != {int(item["window_id"]) for item in windows}:
        _fail("Semantic windows and role bindings do not close the same axis.")
    rows = []
    for window in sorted(windows, key=lambda item: item["window_id"]):
        window_id = int(window["window_id"])
        binding = bindings[window_id]
        if (
            binding["analysis_role"] != window["analysis_role"]
            or binding["source_interval_sha256"] != window["source_interval_sha256"]
            or int(binding["selected_start_frame"]) != int(window["start_frame"])
            or int(binding["selected_end_frame_exclusive"])
            != int(window["end_frame_exclusive"])
        ):
            _fail("Semantic window differs from its exact role binding.")
        rows.append(
            {
                **common,
                "epoch_window_id": window_id,
                "analysis_role": str(window["analysis_role"]),
                "source_label": str(window["source_label"]),
                "start_frame": int(window["start_frame"]),
                "end_frame_exclusive": int(window["end_frame_exclusive"]),
                "source_interval_sha256": str(window["source_interval_sha256"]),
                "protocol_semantic_hash": str(binding["protocol_semantic_hash"]),
                "protocol_semantic_step_index": int(
                    binding["protocol_semantic_step_index"]
                ),
                "protocol_semantic_step_ref": str(
                    binding["protocol_semantic_step_ref"]
                ),
                "terminal_frame_excluded_pending_step_end_contract": bool(
                    binding["terminal_frame_excluded_pending_step_end_contract"]
                ),
                "selection_identity_sha256": str(source["selection_identity_sha256"]),
                "source_epoch_selection_sha256": str(
                    source["source_epoch_selection"]["selection_sha256"]
                ),
                "step_end_interval_semantics": str(
                    source["step_end_interval_semantics"]
                ),
                "trial_index_integrity_status": str(
                    source["trial_index_integrity_status"]
                ),
            }
        )
    return _complete_rows(rows)


def _controller_trials(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle = context.composable_child("controller_trials")
    identities, roles = context.chaser_identity_maps()
    trigger_sources = _registry(handle.scientific_manifest, "trigger_source")
    source_rows = _array_rows(handle.arrays, _CONTROLLER_ARRAYS)
    rows = []
    for source in source_rows:
        code = int(source["chaser_identity_code"])
        trigger_code = int(source["trigger_source_code"])
        rows.append(
            {
                **context.child_common("controller_trials"),
                "trial_row_id": int(source["trial_row_id"]),
                "chaser_identity_code": code,
                "chaser_identity": _decode(identities, code, field="chaser identity"),
                "behavior_role": _decode(roles, code, field="behavior role"),
                "logged_trial_id": int(source["logged_trial_id"]),
                "trial_ordinal": int(source["trial_ordinal"]),
                "start_source_frame_row": int(source["start_source_frame_row"]),
                "end_source_frame_row_exclusive": int(
                    source["end_source_frame_row_exclusive"]
                ),
                "start_acquisition_frame_id": int(source["start_acquisition_frame_id"]),
                "end_acquisition_frame_id_inclusive": int(
                    source["end_acquisition_frame_id_inclusive"]
                ),
                "trigger_acquisition_frame_id": int(
                    source["trigger_acquisition_frame_id"]
                ),
                "trigger_timestamp_ns": int(source["trigger_timestamp_ns"]),
                "trigger_timestamp_valid": bool(source["trigger_timestamp_valid"]),
                "trigger_source_code": trigger_code,
                "trigger_source": _decode(
                    trigger_sources, trigger_code, field="trigger source"
                ),
                "active_member_count": int(source["active_member_count"]),
                "envelope_frame_count": int(source["envelope_frame_count"]),
                "gap_frame_count": int(source["gap_frame_count"]),
                "gap_fraction": float(source["gap_fraction"]),
                "fallback_used": bool(source["fallback_used"]),
            }
        )
    return _complete_rows(rows)


def _bout_source(
    context: _RecordingContext,
) -> tuple[
    ComposableChaserSuccessorSourceHandle,
    list[dict[str, Any]],
    dict[int, str],
    dict[int, str],
    dict[int, str],
]:
    context.require_capability("canonical_swim_bouts")
    handle = context.composable_child("generalized_bout_response")
    identities, roles = context.chaser_identity_maps()
    semantic_roles = _registry(handle.scientific_manifest, "semantic_role")
    source_rows = _array_rows(handle.arrays, _BOUT_ASSOCIATION_ARRAYS)
    return handle, source_rows, identities, roles, semantic_roles


def _canonical_swim_bouts(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle, associations, _identities, _roles, _semantic_roles = _bout_source(context)
    binding = context.bundle["source_bindings"]["canonical_swim_bouts"]["source"]
    source = handle.scientific_manifest["sources"]["swim_bouts"]
    if (
        source["run_path"] != binding["run_path"]
        or source["lineage_sha256"] != binding["lineage_hash"]
        or int(source["signal_id"]) != int(binding["default_signal_id"])
    ):
        _fail("Generalized-bout child binds another canonical swim-bout source.")
    track_id = int(binding["track_id"])
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in associations:
        groups.setdefault(int(item["bout_row_id"]), []).append(item)
    expected_bouts = int(handle.scientific_manifest["dimensions"]["n_bouts"])
    expected_chasers = int(handle.scientific_manifest["dimensions"]["n_chasers"])
    if set(groups) != set(range(expected_bouts)) or any(
        len(items) != expected_chasers for items in groups.values()
    ):
        _fail("Bout-chaser rows do not close the canonical bout axis.")
    copied = (
        "bout_id",
        "source_signal_id",
        "start_acquisition_frame_id",
        "end_acquisition_frame_id",
        "bout_duration_s",
        "bout_path_length_mm",
        "bout_net_displacement_mm",
        "bout_mean_speed_mm_s",
        "bout_peak_speed_mm_s",
        "bout_tortuosity",
    )
    rows = []
    for bout_row_id in sorted(groups):
        items = groups[bout_row_id]
        first = items[0]
        for item in items[1:]:
            for name in copied:
                left = first[name]
                right = item[name]
                if isinstance(left, float) and np.isnan(left) and np.isnan(right):
                    continue
                if left != right:
                    _fail("Repeated bout facts differ across chaser associations.")
        rows.append(
            {
                **context.child_common("generalized_bout_response"),
                "swim_bout_run_path": str(binding["run_path"]),
                "swim_bout_lineage_sha256": str(binding["lineage_hash"]),
                "track_id": track_id,
                "source_signal_id": int(first["source_signal_id"]),
                "bout_id": int(first["bout_id"]),
                "bout_row_id": bout_row_id,
                "start_acquisition_frame_id": int(first["start_acquisition_frame_id"]),
                "end_acquisition_frame_id": int(first["end_acquisition_frame_id"]),
                "duration_s": float(first["bout_duration_s"]),
                "path_length_mm": float(first["bout_path_length_mm"]),
                "net_displacement_mm": float(first["bout_net_displacement_mm"]),
                "mean_speed_mm_s": float(first["bout_mean_speed_mm_s"]),
                "peak_speed_mm_s": float(first["bout_peak_speed_mm_s"]),
                "tortuosity": float(first["bout_tortuosity"]),
            }
        )
    return _complete_rows(rows)


def _bout_chaser_associations(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle, sources, identities, roles, semantic_roles = _bout_source(context)
    attachment = _registry(handle.scientific_manifest, "attachment_reason")
    controller = context.composable_child("controller_trials")
    gap_reasons = _registry(controller.scientific_manifest, "trial_gap_reason")
    track_id = int(
        context.bundle["source_bindings"]["canonical_swim_bouts"]["source"]["track_id"]
    )
    rows = []
    for source in sources:
        chaser_code = int(source["chaser_identity_code"])
        semantic_code = int(source["semantic_role_code"])
        attachment_code = int(source["attachment_reason_code"])
        gap_code = int(source["controller_trial_gap_reason_code"])
        rows.append(
            {
                **context.child_common("generalized_bout_response"),
                "bout_chaser_row_id": int(source["bout_chaser_row_id"]),
                "track_id": track_id,
                "source_signal_id": int(source["source_signal_id"]),
                "bout_id": int(source["bout_id"]),
                "bout_row_id": int(source["bout_row_id"]),
                "chaser_identity_code": chaser_code,
                "chaser_identity": _decode(
                    identities, chaser_code, field="chaser identity"
                ),
                "behavior_role": _decode(roles, chaser_code, field="behavior role"),
                "semantic_role_code": semantic_code,
                "semantic_role": _decode_optional_zero(
                    semantic_roles, semantic_code, field="semantic role"
                ),
                "start_acquisition_frame_id": int(source["start_acquisition_frame_id"]),
                "end_acquisition_frame_id": int(source["end_acquisition_frame_id"]),
                "base_valid": bool(source["base_valid"]),
                "attachment_reason_code": attachment_code,
                "attachment_reason": _decode(
                    attachment, attachment_code, field="attachment reason"
                ),
                "distance_at_onset_mm": float(source["distance_at_onset_mm"]),
                "distance_at_end_mm": float(source["distance_at_end_mm"]),
                "delta_distance_mm": float(source["delta_distance_mm"]),
                "directed_valid": bool(source["directed_valid"]),
                "bearing_at_onset_deg": float(source["bearing_at_onset_deg"]),
                "turn_deg": float(source["turn_deg"]),
                "turn_toward_chaser": bool(source["turn_toward_chaser"]),
                "controller_trial_row_id": int(source["controller_trial_row_id"]),
                "controller_trial_envelope_row_id": int(
                    source["controller_trial_envelope_row_id"]
                ),
                "controller_trial_gap_reason_code": gap_code,
                "controller_trial_gap_reason": _decode(
                    gap_reasons, gap_code, field="controller trial gap reason"
                ),
            }
        )
    return _complete_rows(rows)


def _bout_response_distance_bins(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle = context.composable_child("generalized_bout_response")
    identities, roles = context.chaser_identity_maps()
    semantic_roles = _registry(handle.scientific_manifest, "semantic_role")
    sources = _array_rows(handle.arrays, _BOUT_SUMMARY_ARRAYS)
    rows = []
    for source in sources:
        chaser_code = int(source["summary_chaser_identity_code"])
        semantic_code = int(source["summary_role_code"])
        rows.append(
            {
                **context.child_common("generalized_bout_response"),
                "semantic_role_code": semantic_code,
                "semantic_role": _decode(
                    semantic_roles, semantic_code, field="semantic role"
                ),
                "chaser_identity_code": chaser_code,
                "chaser_identity": _decode(
                    identities, chaser_code, field="chaser identity"
                ),
                "behavior_role": _decode(roles, chaser_code, field="behavior role"),
                "distance_bin_index": int(source["summary_distance_bin_index"]),
                "distance_bin_start_mm": float(source["summary_distance_bin_start_mm"]),
                "distance_bin_end_mm": float(source["summary_distance_bin_end_mm"]),
                "bout_count": int(source["summary_bout_count"]),
                "valid_time_s": float(source["summary_valid_time_s"]),
                "bout_rate_per_min": float(source["summary_bout_rate_per_min"]),
                "median_duration_s": float(source["summary_median_duration_s"]),
                "median_path_length_mm": float(source["summary_median_path_length_mm"]),
                "median_net_displacement_mm": float(
                    source["summary_median_net_displacement_mm"]
                ),
                "median_peak_speed_mm_s": float(
                    source["summary_median_peak_speed_mm_s"]
                ),
            }
        )
    return _complete_rows(rows)


def _escape_source(
    context: _RecordingContext,
) -> tuple[
    ComposableChaserSuccessorSourceHandle,
    dict[int, str],
    dict[int, str],
]:
    handle = context.composable_child("escape_freeze")
    identities, roles = context.chaser_identity_maps()
    return handle, identities, roles


def _trial_escape_freeze_events(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle, identities, roles = _escape_source(context)
    exclusions = _registry(handle.scientific_manifest, "trace_exclusion_reason")
    sources = _array_rows(handle.arrays, _ESCAPE_EVENT_ARRAYS)
    rows = []
    for source in sources:
        code = int(source["event_chaser_identity_code"])
        exclusion_code = int(source["event_trace_exclusion_reason_code"])
        rows.append(
            {
                **context.child_common("escape_freeze"),
                "event_row_id": int(source["event_row_id"]),
                "source_bout_chaser_row_id": int(
                    source["event_source_bout_chaser_row_id"]
                ),
                "bout_row_id": int(source["event_bout_row_id"]),
                "bout_id": int(source["event_bout_id"]),
                "controller_trial_row_id": int(source["event_controller_trial_row_id"]),
                "chaser_identity_code": code,
                "chaser_identity": _decode(identities, code, field="chaser identity"),
                "behavior_role": _decode(roles, code, field="behavior role"),
                "onset_acquisition_frame_id": int(
                    source["event_onset_acquisition_frame_id"]
                ),
                "peak_speed_mm_s": float(source["event_peak_speed_mm_s"]),
                "distance_at_onset_mm": float(source["event_distance_at_onset_mm"]),
                "trigger_distance_mm": float(source["event_trigger_distance_mm"]),
                "latency_from_trigger_s": float(source["event_latency_from_trigger_s"]),
                "separation_gain_mm": float(source["event_separation_gain_mm"]),
                "turn_deg": float(source["event_turn_deg"]),
                "directed_valid": bool(source["event_directed_valid"]),
                "high_turn": bool(source["event_high_turn"]),
                "recaptured": bool(source["event_recaptured"]),
                "recapture_latency_s": float(source["event_recapture_latency_s"]),
                "trace_valid": bool(source["event_trace_valid"]),
                "trace_exclusion_reason_code": exclusion_code,
                "trace_exclusion_reason": _decode(
                    exclusions, exclusion_code, field="trace exclusion reason"
                ),
            }
        )
    return _complete_rows(rows)


ESCAPE_SIGNAL_PROVENANCE_UNAVAILABLE = "unavailable_pre_provenance_run"

_ESCAPE_THRESHOLD_PARAMETERS = (
    "freeze_window_s",
    "freeze_speed_threshold_mm_s",
    "escape_speed_threshold_mm_s",
)


def _escape_signal_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the sealed speed-signal definition behind the freeze metric.

    A legacy run manifest that predates signal provenance yields one explicit
    sentinel row state; a partially recorded manifest fails closed rather than
    guessing which speed level defined ``freeze_low_speed_fraction``.
    """

    sources = manifest.get("sources")
    motion = sources.get("motion") if isinstance(sources, Mapping) else None
    parameters = manifest.get("parameters")
    speed_level = motion.get("speed_level") if isinstance(motion, Mapping) else None
    thresholds = {
        name: (parameters.get(name) if isinstance(parameters, Mapping) else None)
        for name in _ESCAPE_THRESHOLD_PARAMETERS
    }
    values = (speed_level, *thresholds.values())
    if all(value is None for value in values):
        return {
            "speed_level": ESCAPE_SIGNAL_PROVENANCE_UNAVAILABLE,
            **{name: float("nan") for name in _ESCAPE_THRESHOLD_PARAMETERS},
            "signal_provenance_status": ESCAPE_SIGNAL_PROVENANCE_UNAVAILABLE,
        }
    if any(value is None for value in values):
        _fail(
            "Escape/freeze manifest records partial signal provenance; "
            "speed_level and every escape/freeze threshold must be present "
            "together or absent together."
        )
    if (
        type(speed_level) is not str
        or not speed_level
        or speed_level != speed_level.strip()
    ):
        _fail(
            "Escape/freeze manifest speed_level must be one non-empty "
            "exact string."
        )
    result: dict[str, Any] = {"speed_level": speed_level}
    for name, value in thresholds.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or float(value) <= 0.0
        ):
            _fail(
                f"Escape/freeze manifest parameter {name!r} must be one "
                "positive finite number."
            )
        result[name] = float(value)
    result["signal_provenance_status"] = "recorded"
    return result


def _trial_escape_freeze_summaries(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle, identities, roles = _escape_source(context)
    response_classes = _registry(handle.scientific_manifest, "response_class")
    signal_provenance = _escape_signal_provenance(handle.scientific_manifest)
    sources = _array_rows(handle.arrays, _ESCAPE_TRIAL_ARRAYS)
    rows = []
    for source in sources:
        code = int(source["trial_chaser_identity_code"])
        response_code = int(source["trial_response_class_code"])
        rows.append(
            {
                **context.child_common("escape_freeze"),
                "trial_row_id": int(source["trial_row_id"]),
                "chaser_identity_code": code,
                "chaser_identity": _decode(identities, code, field="chaser identity"),
                "behavior_role": _decode(roles, code, field="behavior role"),
                "logged_trial_id": int(source["trial_logged_id"]),
                "trial_ordinal": int(source["trial_ordinal"]),
                "trigger_acquisition_frame_id": int(
                    source["trial_trigger_acquisition_frame_id"]
                ),
                "trigger_distance_mm": float(source["trial_trigger_distance_mm"]),
                "valid_time_s": float(source["trial_valid_time_s"]),
                "envelope_frame_count": int(source["trial_envelope_frame_count"]),
                "gap_frame_count": int(source["trial_gap_frame_count"]),
                "gap_fraction": float(source["trial_gap_fraction"]),
                "logged_active_id_unavailable_count": int(
                    source["trial_logged_active_id_unavailable_count"]
                ),
                "bout_count": int(source["trial_bout_count"]),
                "escape_event_count": int(source["trial_escape_event_count"]),
                "high_turn_escape_count": int(source["trial_high_turn_escape_count"]),
                "escape_event_rate_per_min": float(
                    source["trial_escape_event_rate_per_min"]
                ),
                "first_escape_latency_s": float(source["trial_first_escape_latency_s"]),
                "mean_separation_gain_mm": float(
                    source["trial_mean_separation_gain_mm"]
                ),
                "recapture_fraction": float(source["trial_recapture_fraction"]),
                "freeze_low_speed_fraction": float(
                    source["trial_freeze_low_speed_fraction"]
                ),
                "freeze_valid_fraction": float(source["trial_freeze_valid_fraction"]),
                "escape_speed_class": bool(source["trial_escape_speed_class"]),
                "freeze_candidate": bool(source["trial_freeze_candidate"]),
                "response_class_code": response_code,
                "response_class": _decode(
                    response_classes, response_code, field="response class"
                ),
                **signal_provenance,
            }
        )
    return _complete_rows(rows)


def _trial_escape_freeze_threshold_sweeps(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle, identities, roles = _escape_source(context)
    trials = _array_rows(handle.arrays, _ESCAPE_TRIAL_ARRAYS)
    trial_codes = {
        int(item["trial_row_id"]): int(item["trial_chaser_identity_code"])
        for item in trials
    }
    if len(trial_codes) != len(trials):
        _fail("Escape trial rows are not uniquely keyed.")
    sources = _array_rows(handle.arrays, _ESCAPE_SWEEP_ARRAYS)
    rows = []
    for source in sources:
        trial_row_id = int(source["sweep_trial_row_id"])
        try:
            code = trial_codes[trial_row_id]
        except KeyError as exc:
            raise ValidatedBehaviorAdapterError(
                "Threshold-sweep row references an absent controller trial."
            ) from exc
        rows.append(
            {
                **context.child_common("escape_freeze"),
                "sweep_row_id": int(source["sweep_row_id"]),
                "trial_row_id": trial_row_id,
                "chaser_identity_code": code,
                "chaser_identity": _decode(identities, code, field="chaser identity"),
                "behavior_role": _decode(roles, code, field="behavior role"),
                "speed_threshold_mm_s": float(source["sweep_speed_threshold_mm_s"]),
                "escape_event_count": int(source["sweep_escape_event_count"]),
                "escape_event_rate_per_min": float(
                    source["sweep_escape_event_rate_per_min"]
                ),
            }
        )
    return _complete_rows(rows)


def _epoch_behavior_summary(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle = context.epoch_handle()
    arrays = {
        name: handle.array(f"per_epoch_fish/{name}") for name in _EPOCH_FISH_ARRAYS
    }
    sources = _array_rows(arrays, _EPOCH_FISH_ARRAYS)
    provider_authority = context.bundle["source_bindings"]["provider_motion"][
        "source_authority"
    ]["record"]["position_source"]
    rename = {"window_id": "epoch_window_id"}
    rows = []
    for source in sources:
        row = {
            **context.child_common("epoch_behavior"),
            **{rename.get(name, name): source[name] for name in _EPOCH_FISH_ARRAYS},
            "position_provider_id": str(provider_authority["estimator_id"]),
            "position_provider_digest": str(provider_authority["estimator_sha256"]),
        }
        for name in (
            "track_id",
            "epoch_window_id",
            "window_index",
            "start_frame",
            "end_frame",
            "total_span_frames",
            "provider_sample_count",
            "valid_tracked_frame_count",
            "missing_frame_count",
            "motion_valid_sample_count",
            "speed_sample_count",
            "bout_count",
            "bout_heading_sample_count",
            "inter_bout_interval_count",
            "protocol_semantic_step_index",
        ):
            row[name] = int(row[name])
        for name in (
            "window_label",
            "analysis_role",
            "rate_denominator",
            "motion_validity_rule",
            "source_interval_sha256",
            "protocol_semantic_hash",
            "protocol_semantic_step_ref",
        ):
            row[name] = str(row[name])
        rows.append(row)
    return _complete_rows(rows)


def _spatial_source(
    context: _RecordingContext,
) -> tuple[
    ComposableChaserSuccessorSourceHandle,
    dict[int, str],
    dict[int, str],
    dict[int, Mapping[str, Any]],
    str,
    str,
]:
    handle = context.composable_child("spatial_occupancy")
    manifest = handle.scientific_manifest
    providers = _registry(manifest, "provider_role")
    epochs = _registry(manifest, "epoch_role")
    provider_records = {
        int(item["provider_role_code"]): item
        for item in manifest["sources"]["position_providers"]
    }
    if set(provider_records) != set(providers):
        _fail("Spatial provider records differ from the identity registry.")
    grid_sha = canonical_json_sha256(_plain(manifest["grid"]))
    arena_sha = str(
        manifest["sources"]["arena_geometry_and_scale"]["physical_authority_sha256"]
    )
    return handle, providers, epochs, provider_records, grid_sha, arena_sha


def _spatial_occupancy_support(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle, providers, epochs, provider_records, grid_sha, arena_sha = _spatial_source(
        context
    )
    provider_codes = np.asarray(handle.array("provider_role_code"))
    epoch_codes = np.asarray(handle.array("epoch_role_code"))
    window_ids = np.asarray(handle.array("epoch_window_id"))
    starts = np.asarray(handle.array("epoch_start_frame"))
    stops = np.asarray(handle.array("epoch_end_frame_exclusive"))
    measure_names = (
        "candidate_frame_count",
        "declared_valid_position_frame_count",
        "finite_valid_position_frame_count",
        "invalid_position_frame_count",
        "in_arena_position_frame_count",
        "out_of_arena_position_frame_count",
        "in_arena_coverage_fraction_candidate",
        "in_arena_fraction_finite_valid",
    )
    measures = {name: np.asarray(handle.array(name)) for name in measure_names}
    expected_shape = (len(provider_codes), len(epoch_codes))
    if any(values.shape != expected_shape for values in measures.values()):
        _fail("Spatial support arrays differ from the provider-by-epoch axis.")
    rows = []
    for provider_index, raw_provider_code in enumerate(provider_codes):
        provider_code = int(raw_provider_code)
        provider = provider_records[provider_code]
        for epoch_index, raw_epoch_code in enumerate(epoch_codes):
            epoch_code = int(raw_epoch_code)
            row = {
                **context.child_common("spatial_occupancy"),
                "provider_role_code": provider_code,
                "provider_role": _decode(
                    providers, provider_code, field="provider role"
                ),
                "position_provider_id": str(provider["provider_id"]),
                "position_provider_digest": str(provider["provider_digest"]),
                "epoch_role_code": epoch_code,
                "epoch_role": _decode(epochs, epoch_code, field="epoch role"),
                "epoch_window_id": int(window_ids[epoch_index]),
                "epoch_start_frame": int(starts[epoch_index]),
                "epoch_end_frame_exclusive": int(stops[epoch_index]),
                "grid_recipe_sha256": grid_sha,
                "arena_authority_sha256": arena_sha,
            }
            for name in measure_names:
                value = _scalar(measures[name][provider_index, epoch_index])
                row[name] = float(value) if "fraction" in name else int(value)
            rows.append(row)
    return _complete_rows(rows)


def _spatial_occupancy_bins(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle, providers, epochs, provider_records, grid_sha, arena_sha = _spatial_source(
        context
    )
    provider_codes = np.asarray(handle.array("provider_role_code"))
    epoch_codes = np.asarray(handle.array("epoch_role_code"))
    window_ids = np.asarray(handle.array("epoch_window_id"))
    x_edges = np.asarray(handle.array("x_bin_edges_mm"), dtype=np.float64)
    y_edges = np.asarray(handle.array("y_bin_edges_mm"), dtype=np.float64)
    arena_mask = np.asarray(handle.array("arena_bin_center_mask"), dtype=bool)
    count = np.asarray(handle.array("occupancy_count"))
    density = np.asarray(handle.array("occupancy_density_valid_in_arena"))
    fraction = np.asarray(handle.array("occupancy_fraction_candidate_epoch"))
    expected_shape = (
        len(provider_codes),
        len(epoch_codes),
        len(x_edges) - 1,
        len(y_edges) - 1,
    )
    if (
        count.shape != expected_shape
        or density.shape != expected_shape
        or fraction.shape != expected_shape
        or arena_mask.shape != expected_shape[2:]
    ):
        _fail("Spatial occupancy grids differ from their persisted edge axes.")
    rows = []
    for provider_index, raw_provider_code in enumerate(provider_codes):
        provider_code = int(raw_provider_code)
        provider = provider_records[provider_code]
        for epoch_index, raw_epoch_code in enumerate(epoch_codes):
            epoch_code = int(raw_epoch_code)
            common = {
                **context.child_common("spatial_occupancy"),
                "provider_role_code": provider_code,
                "provider_role": _decode(
                    providers, provider_code, field="provider role"
                ),
                "position_provider_id": str(provider["provider_id"]),
                "position_provider_digest": str(provider["provider_digest"]),
                "epoch_role_code": epoch_code,
                "epoch_role": _decode(epochs, epoch_code, field="epoch role"),
                "epoch_window_id": int(window_ids[epoch_index]),
                "grid_recipe_sha256": grid_sha,
                "arena_authority_sha256": arena_sha,
            }
            for x_index in range(expected_shape[2]):
                for y_index in range(expected_shape[3]):
                    index = (provider_index, epoch_index, x_index, y_index)
                    rows.append(
                        {
                            **common,
                            "x_bin_index": x_index,
                            "y_bin_index": y_index,
                            "x_bin_start_mm": float(x_edges[x_index]),
                            "x_bin_end_mm": float(x_edges[x_index + 1]),
                            "y_bin_start_mm": float(y_edges[y_index]),
                            "y_bin_end_mm": float(y_edges[y_index + 1]),
                            "arena_bin_center_member": bool(
                                arena_mask[x_index, y_index]
                            ),
                            "occupancy_count": int(count[index]),
                            "occupancy_density_valid_in_arena": float(density[index]),
                            "occupancy_fraction_candidate_epoch": float(
                                fraction[index]
                            ),
                        }
                    )
    return _complete_rows(rows)


def _radial_sources(
    context: _RecordingContext,
) -> list[
    tuple[
        str,
        ComposableChaserSuccessorSourceHandle,
        Mapping[str, Any],
        dict[int, str],
        dict[int, str],
        dict[int, str],
    ]
]:
    result = []
    for provider_role, key in (
        ("keypoint", "radial_near_field_keypoint"),
        ("detection", "radial_near_field_detection"),
    ):
        handle = context.composable_child(key)
        manifest = handle.scientific_manifest
        if (
            manifest["position_provider"]["provider_id"]
            != context.bundle["source_bindings"][f"fish_position_{provider_role}"][
                "authority"
            ]["provider_id"]
        ):
            _fail("Radial child position provider differs from the bundle authority.")
        result.append(
            (
                provider_role,
                handle,
                manifest,
                _registry(manifest, "epoch_role"),
                _registry(manifest, "behavior_role"),
                _registry(manifest, "chaser"),
            )
        )
    return result


def _radial_identity(
    *,
    provider_role: str,
    manifest: Mapping[str, Any],
    epoch_roles: Mapping[int, str],
    behavior_roles: Mapping[int, str],
    chasers: Mapping[int, str],
    epoch_code: int,
    epoch_window_id: int,
    behavior_code: int,
    chaser_code: int,
) -> dict[str, Any]:
    provider = manifest["position_provider"]
    return {
        "provider_role": provider_role,
        "position_provider_id": str(provider["provider_id"]),
        "position_provider_digest": str(provider["provider_digest"]),
        "epoch_role_code": epoch_code,
        "epoch_role": _decode(epoch_roles, epoch_code, field="epoch role"),
        "epoch_window_id": epoch_window_id,
        "behavior_role_code": behavior_code,
        "behavior_role": _decode(behavior_roles, behavior_code, field="behavior role"),
        "chaser_identity_code": chaser_code,
        "chaser_identity": _decode(chasers, chaser_code, field="chaser identity"),
    }


def _radial_metric_rows(
    context: _RecordingContext,
) -> list[tuple[dict[str, Any], Mapping[str, Any]]]:
    results: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for provider_role, handle, manifest, epochs, roles, chasers in _radial_sources(
        context
    ):
        key = f"radial_near_field_{provider_role}"
        sources = _array_rows(handle.arrays, _RADIAL_METRIC_ARRAYS)
        config = manifest["config"]
        policies = manifest["policies"]
        arena_sha = str(
            manifest["sources"]["arena_geometry_and_scale"]["physical_authority_sha256"]
        )
        for source in sources:
            epoch_code = int(source["metric_epoch_role_code"])
            behavior_code = int(source["metric_behavior_role_code"])
            chaser_code = int(source["metric_chaser_identity_code"])
            common = {
                **context.child_common(key),
                **_radial_identity(
                    provider_role=provider_role,
                    manifest=manifest,
                    epoch_roles=epochs,
                    behavior_roles=roles,
                    chasers=chasers,
                    epoch_code=epoch_code,
                    epoch_window_id=int(source["metric_epoch_window_id"]),
                    behavior_code=behavior_code,
                    chaser_code=chaser_code,
                ),
                "radial_policy_sha256": canonical_json_sha256(_plain(policies)),
                "arena_authority_sha256": arena_sha,
                "near_zone_radius_mm": float(config["near_zone_radius_mm"]),
                "near_entry_radius_mm": float(config["near_entry_radius_mm"]),
                "near_exit_radius_mm": float(config["near_exit_radius_mm"]),
                "perimeter_band_mm": float(config["perimeter_band_mm"]),
                "min_expected_count": float(config["min_expected_count"]),
            }
            results.append((common, source))
    return results


def _radial_near_field_summary(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    int_fields = (
        "candidate_frame_count",
        "valid_distance_frame_count",
        "wall_excluded_valid_frame_count",
        "near_zone_frame_count",
        "near_zone_entry_count",
        "near_zone_censor_event_count",
        "near_zone_boundary_censor_event_count",
        "near_zone_invalid_gap_count",
        "near_zone_invalid_gap_censor_event_count",
    )
    float_fields = (
        "valid_distance_fraction",
        "distance_mean_mm",
        "distance_p05_mm",
        "distance_p25_mm",
        "distance_p50_mm",
        "distance_p75_mm",
        "distance_p95_mm",
        "fish_arena_radius_mean_mm",
        "fish_arena_radius_p50_mm",
        "fish_wall_distance_mean_mm",
        "fish_wall_distance_p50_mm",
        "near_zone_fraction_candidate",
        "near_zone_fraction_valid",
        "near_zone_dwell_s",
        "near_zone_valid_tracked_duration_s",
        "near_zone_entry_rate_per_min_valid_time",
        "near_zone_complete_visit_total_dwell_s",
        "near_zone_complete_visit_median_dwell_s",
        "near_zone_expected_fraction_geometric",
        "near_zone_enrichment_geometric",
    )
    rows = []
    for common, source in _radial_metric_rows(context):
        row = dict(common)
        for name in int_fields:
            row[name] = int(source[f"metric_{name}"])
        for name in float_fields:
            row[name] = float(source[f"metric_{name}"])
        rows.append(row)
    return _complete_rows(rows)


def _same_quadrant_occupancy(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    quadrant_policies = {
        role: str(manifest["policies"]["quadrant"])
        for role, _handle, manifest, _epochs, _roles, _chasers in _radial_sources(
            context
        )
    }
    rows = []
    for common, source in _radial_metric_rows(context):
        rows.append(
            {
                **{
                    key: value
                    for key, value in common.items()
                    if key
                    not in {
                        "radial_policy_sha256",
                        "arena_authority_sha256",
                        "near_zone_radius_mm",
                        "near_entry_radius_mm",
                        "near_exit_radius_mm",
                        "perimeter_band_mm",
                        "min_expected_count",
                    }
                },
                "candidate_frame_count": int(source["metric_candidate_frame_count"]),
                "valid_distance_frame_count": int(
                    source["metric_valid_distance_frame_count"]
                ),
                "same_quadrant_valid_frame_count": int(
                    source["metric_same_quadrant_valid_frame_count"]
                ),
                "same_quadrant_fraction_candidate": float(
                    source["metric_same_quadrant_fraction_candidate"]
                ),
                "same_quadrant_fraction_valid": float(
                    source["metric_same_quadrant_fraction_valid"]
                ),
                "quadrant_policy": quadrant_policies[common["provider_role"]],
            }
        )
    return _complete_rows(rows)


def _within_group_indexes(
    rows: Sequence[Mapping[str, Any]], key_names: Sequence[str]
) -> list[int]:
    counters: dict[tuple[Any, ...], int] = {}
    result = []
    for row in rows:
        key = tuple(row[name] for name in key_names)
        index = counters.get(key, 0)
        result.append(index)
        counters[key] = index + 1
    return result


def _radial_near_field_density_bins(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    rows = []
    for provider_role, handle, manifest, epochs, roles, chasers in _radial_sources(
        context
    ):
        key = f"radial_near_field_{provider_role}"
        sources = _array_rows(handle.arrays, _RADIAL_BIN_ARRAYS)
        indexes = _within_group_indexes(
            sources,
            (
                "radial_epoch_window_id",
                "radial_chaser_identity_code",
            ),
        )
        policy_sha = canonical_json_sha256(_plain(manifest["policies"]))
        for source, radial_index in zip(sources, indexes, strict=True):
            epoch_code = int(source["radial_epoch_role_code"])
            behavior_code = int(source["radial_behavior_role_code"])
            chaser_code = int(source["radial_chaser_identity_code"])
            rows.append(
                {
                    **context.child_common(key),
                    **_radial_identity(
                        provider_role=provider_role,
                        manifest=manifest,
                        epoch_roles=epochs,
                        behavior_roles=roles,
                        chasers=chasers,
                        epoch_code=epoch_code,
                        epoch_window_id=int(source["radial_epoch_window_id"]),
                        behavior_code=behavior_code,
                        chaser_code=chaser_code,
                    ),
                    "radial_bin_index": radial_index,
                    "radial_bin_start_mm": float(source["radial_bin_start_mm"]),
                    "radial_bin_end_mm": float(source["radial_bin_end_mm"]),
                    "observed_count": int(source["radial_observed_count"]),
                    "observed_fraction": float(source["radial_observed_fraction"]),
                    "expected_available_area_mm2_frames": float(
                        source["radial_expected_available_area_mm2_frames"]
                    ),
                    "expected_fraction_geometric": float(
                        source["radial_expected_fraction_geometric"]
                    ),
                    "selection_index_geometric": float(
                        source["radial_selection_index_geometric"]
                    ),
                    "wall_excluded_observed_count": int(
                        source["radial_wall_excluded_observed_count"]
                    ),
                    "wall_excluded_observed_fraction": float(
                        source["radial_wall_excluded_observed_fraction"]
                    ),
                    "wall_excluded_expected_available_area_mm2_frames": float(
                        source[
                            "radial_wall_excluded_expected_available_area_mm2_frames"
                        ]
                    ),
                    "wall_excluded_expected_fraction_geometric": float(
                        source["radial_wall_excluded_expected_fraction_geometric"]
                    ),
                    "wall_excluded_selection_index_geometric": float(
                        source["radial_wall_excluded_selection_index_geometric"]
                    ),
                    "radial_policy_sha256": policy_sha,
                }
            )
    return _complete_rows(rows)


def _radial_near_field_distance_cdf(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    rows = []
    for provider_role, handle, manifest, epochs, roles, chasers in _radial_sources(
        context
    ):
        key = f"radial_near_field_{provider_role}"
        sources = _array_rows(handle.arrays, _RADIAL_CDF_ARRAYS)
        indexes = _within_group_indexes(
            sources,
            ("cdf_epoch_window_id", "cdf_chaser_identity_code"),
        )
        policy = str(manifest["config"]["cdf_threshold_policy"])
        for source, threshold_index in zip(sources, indexes, strict=True):
            epoch_code = int(source["cdf_epoch_role_code"])
            behavior_code = int(source["cdf_behavior_role_code"])
            chaser_code = int(source["cdf_chaser_identity_code"])
            rows.append(
                {
                    **context.child_common(key),
                    **_radial_identity(
                        provider_role=provider_role,
                        manifest=manifest,
                        epoch_roles=epochs,
                        behavior_roles=roles,
                        chasers=chasers,
                        epoch_code=epoch_code,
                        epoch_window_id=int(source["cdf_epoch_window_id"]),
                        behavior_code=behavior_code,
                        chaser_code=chaser_code,
                    ),
                    "threshold_index": threshold_index,
                    "threshold_mm": float(source["cdf_threshold_mm"]),
                    "fraction_at_or_below": float(source["cdf_fraction_at_or_below"]),
                    "cdf_policy": policy,
                }
            )
    return _complete_rows(rows)


def _body_alignment_distance_bins(
    context: _RecordingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    handle = context.composable_child("body_alignment_by_distance")
    manifest = handle.scientific_manifest
    epochs = {
        int(item["epoch_role_code"]): str(item["analysis_role"])
        for item in manifest["epoch_records"]
    }
    chasers = _registry(manifest, "chaser")
    roles = _registry(manifest, "behavior_role")
    position = manifest["position_provider"]
    body = manifest["sources"]["body_frame_authority"]
    sources = _array_rows(handle.arrays, _BODY_SUMMARY_ARRAYS)
    count_names = (
        "candidate_row_count",
        "joint_valid_row_count",
        "epoch_occurrence_row_count",
        "epoch_distance_valid_row_count",
        "epoch_distance_invalid_row_count",
        "epoch_distance_invalid_body_valid_row_count",
        "epoch_chaser_absent_row_count",
        "body_source_missing_row_count",
        "body_heading_invalid_row_count",
        "body_bearing_invalid_row_count",
        "other_alignment_invalid_row_count",
    )
    float_names = (
        "distance_bin_start_mm",
        "distance_bin_end_mm",
        "distance_bin_center_mm",
        "mean_alignment_cos",
        "alignment_cos_p25",
        "alignment_cos_p50",
        "alignment_cos_p75",
        "mean_abs_bearing_deg",
        "abs_bearing_p25_deg",
        "abs_bearing_p50_deg",
        "abs_bearing_p75_deg",
        "circular_mean_bearing_deg",
        "circular_resultant_length",
    )
    rows = []
    for source in sources:
        epoch_code = int(source["summary_epoch_role_code"])
        chaser_code = int(source["summary_chaser_identity_code"])
        behavior_code = int(source["summary_chaser_behavior_role_code"])
        row = {
            **context.child_common("body_alignment_by_distance"),
            "epoch_role_code": epoch_code,
            "epoch_role": _decode(epochs, epoch_code, field="epoch role"),
            "epoch_window_id": int(source["summary_epoch_window_id"]),
            "chaser_identity_code": chaser_code,
            "chaser_identity": _decode(chasers, chaser_code, field="chaser identity"),
            "behavior_role_code": behavior_code,
            "behavior_role": _decode(roles, behavior_code, field="behavior role"),
            "distance_bin_index": int(source["summary_distance_bin_index"]),
            "position_provider_id": str(position["provider_id"]),
            "position_provider_digest": str(position["provider_digest"]),
            "body_frame_provider_id": str(body["provider_id"]),
            "body_frame_provider_digest": str(body["provider_digest"]),
            "angle_convention_id": str(
                manifest["coordinate_and_angle_convention"]["convention_id"]
            ),
            "distance_bin_recipe_sha256": str(
                manifest["distance_bin_recipe"]["recipe_sha256"]
            ),
        }
        for name in count_names:
            row[name] = int(source[f"summary_{name}"])
        for name in float_names:
            row[name] = float(source[f"summary_{name}"])
        rows.append(row)
    return _complete_rows(rows)


_EXTRACTORS: Mapping[
    str,
    Callable[[_RecordingContext], tuple[list[dict[str, Any]], str | None]],
] = MappingProxyType(
    {
        "recording_source_bindings": _recording_source_bindings,
        "position_providers": _position_providers,
        "chaser_occurrences": _chaser_occurrences,
        "semantic_epochs": _semantic_epochs,
        "controller_trials": _controller_trials,
        "canonical_swim_bouts": _canonical_swim_bouts,
        "bout_chaser_associations": _bout_chaser_associations,
        "bout_response_distance_bins": _bout_response_distance_bins,
        "trial_escape_freeze_events": _trial_escape_freeze_events,
        "trial_escape_freeze_summaries": _trial_escape_freeze_summaries,
        "trial_escape_freeze_threshold_sweeps": _trial_escape_freeze_threshold_sweeps,
        "epoch_behavior_summary": _epoch_behavior_summary,
        "spatial_occupancy_support": _spatial_occupancy_support,
        "spatial_occupancy_bins": _spatial_occupancy_bins,
        "radial_near_field_summary": _radial_near_field_summary,
        "same_quadrant_occupancy": _same_quadrant_occupancy,
        "radial_near_field_density_bins": _radial_near_field_density_bins,
        "radial_near_field_distance_cdf": _radial_near_field_distance_cdf,
        "body_alignment_distance_bins": _body_alignment_distance_bins,
    }
)


def build_phase_a_row_extractors() -> Mapping[str, Callable[..., Any]]:
    """Return recording-scoped extractors sharing one bounded last-record cache."""

    cache = _LastRecordingContext()

    def wrap(
        producer: Callable[
            [_RecordingContext], tuple[list[dict[str, Any]], str | None]
        ],
    ) -> Callable[..., Any]:
        def extract(
            plan: Mapping[str, Any],
            membership_member: Mapping[str, Any],
            bundle_member: Mapping[str, Any],
        ) -> tuple[list[dict[str, Any]], str | None]:
            return producer(cache.get(plan, membership_member, bundle_member))

        return extract

    return MappingProxyType(
        {name: wrap(producer) for name, producer in _EXTRACTORS.items()}
    )


__all__ = [
    "ValidatedBehaviorAdapterError",
    "build_phase_a_row_extractors",
]
