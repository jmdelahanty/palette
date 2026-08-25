"""Normalized SQLite payloads for recording stimulus protocol rows."""

from __future__ import annotations

from typing import Any, Mapping


RECORDING_STIMULUS_RUN_SEMANTIC_COLUMNS = (
    "protocol_semantic_status",
    "protocol_semantic_hash",
    "palette_computed_trial_index_sha256",
    "protocol_trial_index_sha256",
    "producer_protocol_trial_index_hash",
    "protocol_trial_index_integrity_status",
    "protocol_snapshot_schema_version",
    "protocol_snapshot_policy_id",
    "protocol_trial_index_schema_version",
    "protocol_execution_status",
    "protocol_execution_hash",
    "protocol_interval_axis",
    "protocol_acquisition_containment_status",
    "protocol_frame_correspondence_proxy_status",
    "protocol_frame_correspondence_proxy_manifest_sha256",
    "protocol_frame_correspondence_proxy_missing_count",
    "protocol_recipe_schema_id",
    "protocol_recipe_schema_version",
    "protocol_recipe_step_count",
    "protocol_recipe_mode_sequence_json",
    "protocol_recipe_label",
)

RECORDING_STIMULUS_STEP_SEMANTIC_COLUMNS = (
    "protocol_semantic_status",
    "protocol_semantic_hash",
    "protocol_semantic_step_index",
    "protocol_semantic_step_ref",
    "protocol_semantic_stimulus_mode_id",
    "protocol_semantic_duration_s",
    "stimulus_family",
    "display_context",
    "protocol_trial_index_status",
    "resolved_color_rgba8_json",
    "start_stimulus_frame_inclusive",
    "end_stimulus_frame_exclusive",
    "first_camera_frame_id_correspondence",
    "last_camera_frame_id_correspondence",
    "authoritative_interval_axis",
    "execution_completion_status",
    "execution_end_reason",
    "protocol_execution_phases_json",
)

RECORDING_STIMULUS_RUN_INSERT_SQL = """
    INSERT INTO recording_stimulus_runs (
        dataset_id, recording_id, stimulus_run_id, protocol_hash,
        protocol_name, is_latest, step_count, source_path,
        source_metadata_sha256, source_zarr_path, extracted_utc,
        protocol_semantic_status, protocol_semantic_hash,
        palette_computed_trial_index_sha256,
        protocol_trial_index_sha256,
        producer_protocol_trial_index_hash,
        protocol_trial_index_integrity_status,
        protocol_snapshot_schema_version,
        protocol_snapshot_policy_id,
        protocol_trial_index_schema_version,
        protocol_execution_status, protocol_execution_hash,
        protocol_interval_axis,
        protocol_acquisition_containment_status,
        protocol_frame_correspondence_proxy_status,
        protocol_frame_correspondence_proxy_manifest_sha256,
        protocol_frame_correspondence_proxy_missing_count,
        protocol_recipe_schema_id, protocol_recipe_schema_version,
        protocol_recipe_step_count,
        protocol_recipe_mode_sequence_json, protocol_recipe_label
    ) VALUES (
        :dataset_id, :recording_id, :stimulus_run_id, :protocol_hash,
        :protocol_name, :is_latest, :step_count, :source_path,
        :source_metadata_sha256, :source_zarr_path, :extracted_utc,
        :protocol_semantic_status, :protocol_semantic_hash,
        :palette_computed_trial_index_sha256,
        :protocol_trial_index_sha256,
        :producer_protocol_trial_index_hash,
        :protocol_trial_index_integrity_status,
        :protocol_snapshot_schema_version,
        :protocol_snapshot_policy_id,
        :protocol_trial_index_schema_version,
        :protocol_execution_status, :protocol_execution_hash,
        :protocol_interval_axis,
        :protocol_acquisition_containment_status,
        :protocol_frame_correspondence_proxy_status,
        :protocol_frame_correspondence_proxy_manifest_sha256,
        :protocol_frame_correspondence_proxy_missing_count,
        :protocol_recipe_schema_id, :protocol_recipe_schema_version,
        :protocol_recipe_step_count,
        :protocol_recipe_mode_sequence_json, :protocol_recipe_label
    );
"""

RECORDING_STIMULUS_STEP_INSERT_SQL = """
    INSERT INTO recording_stimulus_steps (
        dataset_id, stimulus_run_id, step_index, step_name,
        stimulus_mode, start_camera_frame, end_camera_frame,
        duration_s, step_attrs_json, protocol_semantic_status,
        protocol_semantic_hash, protocol_semantic_step_index,
        protocol_semantic_step_ref,
        protocol_semantic_stimulus_mode_id,
        protocol_semantic_duration_s, stimulus_family,
        display_context, protocol_trial_index_status,
        resolved_color_rgba8_json,
        start_stimulus_frame_inclusive,
        end_stimulus_frame_exclusive,
        first_camera_frame_id_correspondence,
        last_camera_frame_id_correspondence,
        authoritative_interval_axis,
        execution_completion_status, execution_end_reason,
        protocol_execution_phases_json
    ) VALUES (
        :dataset_id, :stimulus_run_id, :step_index, :step_name,
        :stimulus_mode, :start_camera_frame, :end_camera_frame,
        :duration_s, :step_attrs_json, :protocol_semantic_status,
        :protocol_semantic_hash, :protocol_semantic_step_index,
        :protocol_semantic_step_ref,
        :protocol_semantic_stimulus_mode_id,
        :protocol_semantic_duration_s, :stimulus_family,
        :display_context, :protocol_trial_index_status,
        :resolved_color_rgba8_json,
        :start_stimulus_frame_inclusive,
        :end_stimulus_frame_exclusive,
        :first_camera_frame_id_correspondence,
        :last_camera_frame_id_correspondence,
        :authoritative_interval_axis,
        :execution_completion_status, :execution_end_reason,
        :protocol_execution_phases_json
    );
"""


def recording_stimulus_payload(
    record: Mapping[str, Any],
    *,
    dataset_id: str,
    semantic_columns: tuple[str, ...],
) -> dict[str, Any]:
    """Bind one recording row and explicitly default optional semantic fields."""

    payload = {**dict(record), "dataset_id": str(dataset_id)}
    for name in semantic_columns:
        payload.setdefault(name, None)
    return payload


__all__ = [
    "RECORDING_STIMULUS_RUN_INSERT_SQL",
    "RECORDING_STIMULUS_RUN_SEMANTIC_COLUMNS",
    "RECORDING_STIMULUS_STEP_INSERT_SQL",
    "RECORDING_STIMULUS_STEP_SEMANTIC_COLUMNS",
    "recording_stimulus_payload",
]
