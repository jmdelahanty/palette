"""Registry migration catalog.

The migration bodies still live on ``Registry`` for now. Keeping the ordered
catalog here gives future migration extraction a single data-only surface and
keeps the SQLite connection class from owning the migration manifest directly.
"""

from __future__ import annotations

from typing import Callable, Protocol


class BoundMigrationOwner(Protocol):
    def __getattribute__(self, name: str) -> Callable[[], None]: ...


MIGRATION_METHODS: tuple[tuple[int, str, str], ...] = (
    (1, "initial_registry_schema", "_migration_001_initial_schema"),
    (2, "reserved_noop_template", "_migration_002_reserved_noop"),
    (3, "recording_columns_reconcile", "_migration_003_recording_columns_reconcile"),
    (4, "recording_overview_refresh", "_migration_004_recording_overview_refresh"),
    (5, "drop_provenance_zarr_purpose", "_migration_005_drop_provenance_zarr_purpose"),
    (6, "subject_dish_cross_entities", "_migration_006_subject_dish_cross_entities"),
    (7, "subjects_entities_and_query_indexes", "_migration_007_subjects_entities_and_query_indexes"),
    (8, "recording_subject_overview_view", "_migration_008_recording_subject_overview_view"),
    (9, "training_task_type_columns", "_migration_009_training_task_type_columns"),
    (10, "detect_performance_registry", "_migration_010_detect_performance_registry"),
    (11, "detect_model_performance_views", "_migration_011_detect_model_performance_views"),
    (12, "detect_performance_model_identity", "_migration_012_detect_performance_model_identity"),
    (13, "detect_model_performance_summary_views", "_migration_013_detect_model_performance_summary_views"),
    (14, "crop_quality_registry", "_migration_014_crop_quality_registry"),
    (15, "eye_mask_performance_registry", "_migration_015_eye_mask_performance_registry"),
    (16, "model_export_nms_threshold_columns", "_migration_016_model_export_nms_threshold_columns"),
    (17, "eye_mask_performance_review_stale_columns", "_migration_017_eye_mask_performance_review_stale_columns"),
    (18, "keypoint_performance_registry", "_migration_018_keypoint_performance_registry"),
    (19, "recording_step_status_registry", "_migration_019_recording_step_status_registry"),
    (20, "recording_step_status_wide_view", "_migration_020_recording_step_status_wide_view"),
    (21, "detect_keypoint_quality_review_columns", "_migration_021_detect_keypoint_quality_review_columns"),
    (22, "detection_data_profile_registry", "_migration_022_detection_data_profile_registry"),
    (23, "detection_data_profile_lineage_projection", "_migration_023_detection_data_profile_lineage_projection"),
    (24, "keypoint_data_profile_registry", "_migration_024_keypoint_data_profile_registry"),
    (25, "eye_mask_data_profile_registry", "_migration_025_eye_mask_data_profile_registry"),
    (26, "eye_mask_quality_registry", "_migration_026_eye_mask_quality_registry"),
    (27, "detect_quality_wide_view_columns", "_migration_027_detect_quality_wide_view_columns"),
    (28, "keypoint_auto_review_policy_columns", "_migration_028_keypoint_auto_review_policy_columns"),
    (29, "keypoint_quality_current_latest_source_preference", "_migration_029_keypoint_quality_current_latest_source_preference"),
    (30, "tracking_unassigned_warning_wide_view", "_migration_030_tracking_unassigned_warning_wide_view"),
    (31, "tracking_qc_state_wide_view", "_migration_031_tracking_qc_state_wide_view"),
    (32, "subject_mask_registry", "_migration_032_subject_mask_registry"),
    (33, "subject_mask_registry_semantics_columns", "_migration_033_subject_mask_registry_semantics_columns"),
    (34, "dataset_context_current_view", "_migration_034_dataset_context_current_view"),
    (35, "recording_step_status_latest_dataset_context_current", "_migration_035_recording_step_status_latest_dataset_context_current"),
    (36, "subject_mask_component_latest_views", "_migration_036_subject_mask_component_latest_views"),
    (37, "subject_mask_component_eye_compat_latest_views", "_migration_037_subject_mask_component_eye_compat_latest_views"),
    (38, "subject_mask_component_partial_run_preference", "_migration_038_subject_mask_component_partial_run_preference"),
    (39, "subject_mask_component_source_stale_views", "_migration_039_subject_mask_component_source_stale_views"),
    (40, "subject_mask_training_model_discovery", "_migration_040_subject_mask_training_model_discovery"),
    (41, "analytics_manifest_registry", "_migration_041_analytics_manifest_registry"),
    (42, "recording_experiment_context_columns", "_migration_042_recording_experiment_context_columns"),
    (43, "stage_catalog_recording_step_status_wide_view", "_migration_043_stage_catalog_recording_step_status_wide_view"),
    (44, "derived_analysis_recording_step_status_wide_view", "_migration_044_derived_analysis_recording_step_status_wide_view"),
    (45, "tail_behavior_recording_step_status_wide_view", "_migration_045_tail_behavior_recording_step_status_wide_view"),
    (46, "source_freshness_recording_step_status_wide_view", "_migration_046_source_freshness_recording_step_status_wide_view"),
    (47, "bout_stimulus_source_freshness_recording_step_status_wide_view", "_migration_047_bout_stimulus_source_freshness_recording_step_status_wide_view"),
    (48, "eye_shape_source_freshness_recording_step_status_wide_view", "_migration_048_eye_shape_source_freshness_recording_step_status_wide_view"),
    (49, "model_input_shape_registry", "_migration_049_model_input_shape_registry"),
    (50, "detect_quality_current_reviewed_preference", "_migration_050_detect_quality_current_reviewed_preference"),
    (51, "training_image_profile_registry", "_migration_051_training_image_profile_registry"),
    (52, "dataset_source_layout_metadata", "_migration_052_dataset_source_layout_metadata"),
    (53, "model_deployment_artifacts", "_migration_053_model_deployment_artifacts"),
    (54, "crop_quality_pixel_contract_columns", "_migration_054_crop_quality_pixel_contract_columns"),
    (55, "keypoint_performance_pixel_contract_columns", "_migration_055_keypoint_performance_pixel_contract_columns"),
    (56, "acquisition_video_streams_registry", "_migration_056_acquisition_video_streams_registry"),
    (57, "subject_mask_storage_byte_fields", "_migration_057_subject_mask_storage_byte_fields"),
    (58, "tracking_readiness_guard_views", "_migration_058_tracking_readiness_guard_views"),
    (59, "subject_mask_data_profile_registry", "_migration_059_subject_mask_data_profile_registry"),
    (60, "recording_chaser_metadata_registry", "_migration_060_recording_chaser_metadata_registry"),
    (61, "stimulus_protocol_registry", "_migration_061_stimulus_protocol_registry"),
    (62, "analytics_report_registry", "_migration_062_analytics_report_registry"),
    (63, "recording_subject_traits", "_migration_063_recording_subject_traits"),
    (64, "strain_trait_expectations", "_migration_064_strain_trait_expectations"),
    (65, "subject_trait_schema_reconcile", "_migration_065_subject_trait_schema_reconcile"),
)


def bind_migrations(owner: BoundMigrationOwner) -> list[tuple[int, str, Callable[[], None]]]:
    return [(version, name, getattr(owner, method_name)) for version, name, method_name in MIGRATION_METHODS]


__all__ = ["MIGRATION_METHODS", "bind_migrations"]
