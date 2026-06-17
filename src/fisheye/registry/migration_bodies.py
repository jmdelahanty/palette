"""SQLite schema migration method mixin for :mod:`fisheye.registry.db`.

The migration bodies are intentionally split from the connection/CRUD layer but
remain methods so they can call Registry SQL helpers through ``self``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .stage_catalog import recording_tuning_stage_ids


def _require_sql_identifier(value: str) -> str:
    if not value or not all(ch.isalnum() or ch == "_" for ch in value):
        raise ValueError(f"Unsafe SQL identifier fragment: {value!r}")
    return value


def _recording_step_status_pivot_columns() -> str:
    from .stage_catalog import recording_status_stage_ids

    lines = []
    for step_name in recording_status_stage_ids():
        step = _require_sql_identifier(step_name)
        lines.append(
            f"MAX(CASE WHEN step_name = '{step}' THEN status END) AS {step}_status"
        )
    return ",\n                    ".join(lines)


def _recording_tuning_ok_count_sql(alias: str) -> str:
    table_alias = _require_sql_identifier(alias)
    terms = []
    for step_name in recording_tuning_stage_ids():
        step = _require_sql_identifier(step_name)
        terms.append(f"CASE WHEN {table_alias}.{step}_status = 'ok' THEN 1 ELSE 0 END")
    return "\n                        + ".join(terms) if terms else "0"


def _recording_step_status_display_sql(status_expr: str, details_expr: str) -> str:
    return f"""
                    CASE
                        WHEN {status_expr} = 'ok' THEN 'OK'
                        WHEN {status_expr} = 'na' THEN 'N/A'
                        WHEN {status_expr} = 'error' THEN 'ERR'
                        WHEN json_extract({details_expr}, '$.source_freshness_state') = 'stale' THEN 'STALE'
                        WHEN json_extract({details_expr}, '$.source_freshness_state') IN (
                            'missing_source_attrs',
                            'upstream_source_unavailable'
                        ) THEN 'UNVER'
                        ELSE 'MISS'
                    END
    """.strip()


def _json_loads(value: Any) -> Optional[dict[str, Any]]:
    import json

    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _normalize_task_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    norm = text.lower()
    alias = {
        "detect": "detect",
        "detection": "detect",
        "pose": "pose",
        "keypoint": "pose",
        "keypoints": "pose",
        "eye_masks": "eye_masks",
        "eyemasks": "eye_masks",
        "subject_masks": "subject_masks",
        "subjectmasks": "subject_masks",
        "segmentation": "subject_masks",
    }
    return alias.get(norm)


def _infer_task_type(*, explicit: Any = None, set_id: Any = None, run_id: Any = None, config_path: Any = None, manifest_path: Any = None, model_path: Any = None, invocation: Optional[dict[str, Any]] = None, query_filter: Optional[dict[str, Any]] = None) -> Optional[str]:
    def _infer_from_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if text.startswith("detect_") or "_detect_" in text or "/detect/" in text:
            return "detect"
        if text.startswith("pose_") or "_pose_" in text or "/pose/" in text:
            return "pose"
        if text.startswith("keypoint_") or text.startswith("keypoints_") or "keypoint" in text:
            return "pose"
        if text.startswith("eye_mask_") or "eye_mask" in text or "eyemask" in text:
            return "eye_masks"
        if text.startswith("subject_mask_") or "subject_mask" in text or "subjectmask" in text:
            return "subject_masks"
        return None

    direct = _normalize_task_type(explicit)
    if direct:
        return direct
    for candidate in (set_id, run_id, config_path, manifest_path, model_path):
        inferred = _infer_from_text(candidate)
        if inferred:
            return inferred
    for payload in (invocation, query_filter):
        if not isinstance(payload, dict):
            continue
        for key in ("task_type", "task"):
            inferred = _normalize_task_type(payload.get(key))
            if inferred:
                return inferred
        args_payload = payload.get("args")
        if isinstance(args_payload, dict):
            for key in ("task_type", "task"):
                inferred = _normalize_task_type(args_payload.get(key))
                if inferred:
                    return inferred
    return None


class RegistryMigrationMixin:
    def _migration_001_initial_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                zarr_path TEXT NOT NULL,
                recording_id TEXT,
                artifact_kind TEXT,
                zarr_origin TEXT,
                zarr_use TEXT,
                source_layout TEXT,
                source_frame_index_path TEXT,
                source_recording_frame_index_path TEXT,
                source_frame_index_schema TEXT,
                path_hash TEXT,
                created_utc TEXT,
                last_seen_utc TEXT,
                status TEXT
            );
            """
        )
        # Existing registries may predate these columns; add them before creating
        # any index/view that references the new fields.
        self._ensure_columns(
            "datasets",
            {
                "recording_id": "TEXT",
                "artifact_kind": "TEXT",
                "zarr_origin": "TEXT",
                "zarr_use": "TEXT",
                "source_layout": "TEXT",
                "source_frame_index_path": "TEXT",
                "source_recording_frame_index_path": "TEXT",
                "source_frame_index_schema": "TEXT",
            },
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                recording_name TEXT,
                recording_path TEXT,
                started_utc TEXT,
                recording_type TEXT,
                recording_subtype TEXT,
                behavior_mode TEXT,
                artifact_schema_id TEXT,
                experiment_context_status TEXT,
                experiment_context_source TEXT,
                experiment_context_status_detail TEXT,
                stimulus_runs_available INTEGER,
                rig_id TEXT,
                arena_id TEXT,
                camera_id TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                dish_design TEXT,
                created_utc TEXT,
                updated_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_type_vocab (
                recording_type TEXT PRIMARY KEY,
                active INTEGER NOT NULL DEFAULT 1,
                description TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_subtype_vocab (
                recording_type TEXT NOT NULL,
                recording_subtype TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                description TEXT,
                PRIMARY KEY (recording_type, recording_subtype),
                FOREIGN KEY(recording_type) REFERENCES recording_type_vocab(recording_type) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS zarr_origin_vocab (
                zarr_origin TEXT PRIMARY KEY,
                active INTEGER NOT NULL DEFAULT 1,
                description TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS zarr_use_vocab (
                zarr_use TEXT PRIMARY KEY,
                active INTEGER NOT NULL DEFAULT 1,
                description TEXT
            );
            """
        )
        # Existing registries may predate recording_subtype.
        # Add it before creating indexes that reference this column.
        self._ensure_columns(
            "recordings",
            {
                "recording_subtype": "TEXT",
                "behavior_mode": "TEXT",
                "dish_design": "TEXT",
                "experiment_context_status": "TEXT",
                "experiment_context_source": "TEXT",
                "experiment_context_status_detail": "TEXT",
                "stimulus_runs_available": "INTEGER",
            },
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO recording_type_vocab (recording_type, active, description)
            VALUES (?, 1, ?);
            """,
            [
                ("behavior", "Behavior recordings"),
                ("microscopy", "Microscopy recordings"),
                ("histology", "Histology recordings"),
            ],
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO recording_subtype_vocab (
                recording_type, recording_subtype, active, description
            )
            VALUES (?, ?, 1, ?);
            """,
            [
                ("behavior", "free", "Freely swimming behavior"),
                ("behavior", "embedded", "Embedded behavior"),
                ("microscopy", "lightsheet", "Light-sheet microscopy"),
                ("microscopy", "confocal", "Confocal microscopy"),
                ("microscopy", "2p", "Two-photon microscopy"),
                ("histology", "section", "Section histology"),
                ("histology", "wholemount", "Whole-mount histology"),
            ],
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO zarr_origin_vocab (zarr_origin, active, description)
            VALUES (?, 1, ?);
            """,
            [
                ("source", "Source recording artifact"),
                ("derived", "Derived artifact produced from other artifacts"),
                ("imported", "Imported external artifact"),
            ],
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO zarr_use_vocab (zarr_use, active, description)
            VALUES (?, 1, ?);
            """,
            [
                ("training", "Used for model training"),
                ("analysis", "Used for analysis"),
                ("inference", "Inference outputs"),
                ("export", "Exported model/input artifact"),
                ("archive", "Archived/cold artifact"),
            ],
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_artifacts (
                artifact_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_group TEXT,
                relpath TEXT,
                path TEXT NOT NULL,
                file_ext TEXT,
                status TEXT,
                size_bytes INTEGER,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                FOREIGN KEY(recording_id) REFERENCES recordings(recording_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance (
                dataset_id TEXT PRIMARY KEY,
                fish_id TEXT,
                subject_count INTEGER,
                dish_id TEXT,
                dish_design TEXT,
                cross_id TEXT,
                line_strain TEXT,
                genotype TEXT,
                parents_json TEXT,
                species TEXT,
                sex TEXT,
                dpf_at_acquisition INTEGER,
                rig_id TEXT,
                arena_id TEXT,
                camera_id TEXT,
                canvas_name TEXT,
                fps REAL,
                video_codec TEXT,
                video_pix_fmt TEXT,
                format_title TEXT,
                format_comment TEXT,
                format_encoder TEXT,
                encoder_name TEXT,
                encoder_codec TEXT,
                encoder_preset TEXT,
                encoder_tuning TEXT,
                encoder_rc TEXT,
                encoder_bpp REAL,
                encoder_target_bps INTEGER,
                encoder_res TEXT,
                encoder_res_width INTEGER,
                encoder_res_height INTEGER,
                encoder_fps REAL,
                encoder_color INTEGER,
                encoder_params_json TEXT,
                source_video TEXT,
                compression_name TEXT,
                compression_level INTEGER,
                exposure REAL,
                exposure_unit TEXT,
                gain REAL,
                frame_rate REAL,
                pixel_format TEXT,
                binning TEXT,
                adc TEXT,
                camera_model TEXT,
                camera_serial TEXT,
                camera_metadata_json TEXT,
                has_images_ds INTEGER,
                has_images_ds_rgb INTEGER,
                downsample_formats_json TEXT,
                protocol_name TEXT,
                protocol_hash TEXT,
                snapshot_status TEXT,
                snapshot_missing_json TEXT,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_sources (
                dataset_id TEXT NOT NULL,
                refined_run TEXT,
                source_type TEXT,
                counts_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (dataset_id, refined_run, source_type),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_lineage (
                child_dataset_id TEXT NOT NULL,
                parent_dataset_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                source_set_id TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                PRIMARY KEY (child_dataset_id, parent_dataset_id, relationship_type),
                FOREIGN KEY(child_dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
                FOREIGN KEY(parent_dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_dataset_lineage_no_self_insert
            BEFORE INSERT ON dataset_lineage
            FOR EACH ROW
            WHEN NEW.child_dataset_id = NEW.parent_dataset_id
            BEGIN
                SELECT RAISE(ABORT, 'dataset_lineage self-edge is not allowed');
            END;
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_dataset_lineage_no_self_update
            BEFORE UPDATE ON dataset_lineage
            FOR EACH ROW
            WHEN NEW.child_dataset_id = NEW.parent_dataset_id
            BEGIN
                SELECT RAISE(ABORT, 'dataset_lineage self-edge is not allowed');
            END;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS keypoint_quality (
                dataset_id TEXT NOT NULL,
                refined_run TEXT NOT NULL,
                refined_created_utc TEXT,
                source_keypoint_run TEXT NOT NULL,
                keypoint_method TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_notes TEXT,
                review_policy_id TEXT,
                review_policy_version INTEGER,
                review_timestamp_utc TEXT,
                usable_keypoints INTEGER,
                total_keypoints INTEGER,
                usable_keypoints_rate REAL,
                raw_keypoints_success_rate REAL,
                raw_keypoints_successful INTEGER,
                quality_updated_utc TEXT,
                zarr_mtime_ns INTEGER,
                PRIMARY KEY (dataset_id, refined_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detect_quality (
                dataset_id TEXT NOT NULL,
                refined_run TEXT NOT NULL,
                refined_created_utc TEXT,
                source_detect_run TEXT NOT NULL,
                detect_method TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_notes TEXT,
                review_timestamp_utc TEXT,
                review_resolved_group TEXT,
                total_detections INTEGER,
                real_detections INTEGER,
                interpolated_detections INTEGER,
                interpolated_detections_rate REAL,
                quality_updated_utc TEXT,
                zarr_mtime_ns INTEGER,
                PRIMARY KEY (dataset_id, refined_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pose_skeleton_specs (
                skeleton_id TEXT PRIMARY KEY,
                spec_sha256 TEXT NOT NULL UNIQUE,
                name TEXT,
                kpt_shape_json TEXT,
                keypoint_labels_json TEXT,
                edges_json TEXT,
                spec_json TEXT,
                created_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_sets (
                set_id TEXT PRIMARY KEY,
                name TEXT,
                task_type TEXT,
                query_filter TEXT,
                dataset_ids_json TEXT,
                skeleton_id TEXT,
                invocation_json TEXT,
                created_utc TEXT,
                FOREIGN KEY(skeleton_id) REFERENCES pose_skeleton_specs(skeleton_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                set_id TEXT,
                task_type TEXT,
                config_path TEXT,
                manifest_path TEXT,
                skeleton_id TEXT,
                model_path TEXT,
                metrics_path TEXT,
                config_sha256 TEXT,
                manifest_sha256 TEXT,
                model_sha256 TEXT,
                metrics_sha256 TEXT,
                status TEXT,
                final_metrics_json TEXT,
                invocation_json TEXT,
                created_utc TEXT,
                FOREIGN KEY(skeleton_id) REFERENCES pose_skeleton_specs(skeleton_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_exports (
                run_id TEXT NOT NULL,
                export_type TEXT NOT NULL,
                path TEXT,
                manifest_path TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (run_id, export_type),
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_models (
                run_id TEXT PRIMARY KEY,
                set_id TEXT,
                model_path TEXT,
                model_sha256 TEXT,
                metrics_path TEXT,
                metrics_sha256 TEXT,
                status TEXT,
                task_type TEXT,
                label_schema_id TEXT,
                coverage_class TEXT,
                component_coverage_key TEXT,
                mask_labels_json TEXT,
                component_groups_json TEXT,
                best_metric_name TEXT,
                best_metric_value REAL,
                best_epoch INTEGER,
                input_shape TEXT,
                input_layout TEXT,
                input_channels INTEGER,
                img_h INTEGER,
                img_w INTEGER,
                max_batch INTEGER,
                dynamic_shapes INTEGER,
                input_dtype TEXT,
                input_color_space TEXT,
                input_shape_source TEXT,
                input_shape_status TEXT,
                final_metrics_json TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS onnx_models (
                run_id TEXT PRIMARY KEY,
                set_id TEXT,
                skeleton_id TEXT,
                detection_model_run_id TEXT,
                path TEXT,
                sha256 TEXT,
                manifest_path TEXT,
                manifest_sha256 TEXT,
                opset INTEGER,
                nms_conf REAL,
                nms_iou REAL,
                nms_topk INTEGER,
                input_shape TEXT,
                img_h INTEGER,
                img_w INTEGER,
                max_batch INTEGER,
                dynamic_shapes INTEGER,
                file_size_bytes INTEGER,
                exporter_torch_version TEXT,
                exporter_cuda_version TEXT,
                exporter_hostname TEXT,
                requires_plugins INTEGER,
                plugin_ops_json TEXT,
                plugin_versions_json TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                FOREIGN KEY(skeleton_id) REFERENCES pose_skeleton_specs(skeleton_id),
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tensorrt_models (
                run_id TEXT NOT NULL,
                set_id TEXT,
                skeleton_id TEXT,
                detection_model_run_id TEXT,
                onnx_run_id TEXT,
                precision TEXT NOT NULL,
                nms_conf REAL,
                nms_iou REAL,
                nms_topk INTEGER,
                path TEXT,
                sha256 TEXT,
                manifest_path TEXT,
                manifest_sha256 TEXT,
                input_shape TEXT,
                img_h INTEGER,
                img_w INTEGER,
                max_batch INTEGER,
                dynamic_shapes INTEGER,
                file_size_bytes INTEGER,
                trt_version TEXT,
                cuda_version TEXT,
                compute_capability TEXT,
                gpu_name TEXT,
                gpu_uuid TEXT,
                system_hostname TEXT,
                requires_plugins INTEGER,
                plugin_ops_json TEXT,
                plugin_versions_json TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (run_id, precision),
                FOREIGN KEY(skeleton_id) REFERENCES pose_skeleton_specs(skeleton_id),
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_set_id ON training_models(set_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_task_status ON training_models(task_type, status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_label_schema ON training_models(label_schema_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_component_coverage ON training_models(component_coverage_key);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_sets_skeleton_id ON training_sets(skeleton_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_sets_task_type ON training_sets(task_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_runs_skeleton_id ON training_runs(skeleton_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_runs_task_type ON training_runs(task_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_recording_id ON datasets(recording_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_artifact_kind ON datasets(artifact_kind);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_origin_use ON datasets(zarr_origin, zarr_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recordings_session_uuid ON recordings(session_uuid);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recordings_type_subtype ON recordings(recording_type, recording_subtype);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recordings_behavior_mode ON recordings(behavior_mode);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subtype_vocab_type_active ON recording_subtype_vocab(recording_type, active);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_artifacts_recording_id ON recording_artifacts(recording_id);"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_recording_artifacts_recording_path ON recording_artifacts(recording_id, path);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_lineage_child_rel ON dataset_lineage(child_dataset_id, relationship_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_lineage_parent_rel ON dataset_lineage(parent_dataset_id, relationship_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_onnx_models_set_id ON onnx_models(set_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_onnx_models_skeleton_id ON onnx_models(skeleton_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tensorrt_models_set_id ON tensorrt_models(set_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tensorrt_models_skeleton_id ON tensorrt_models(skeleton_id);"
        )
        # Migrate legacy detection_models rows into training_models.
        legacy_detection_models = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='detection_models';"
        ).fetchone()
        if legacy_detection_models is not None:
            cur.execute(
                """
                INSERT OR REPLACE INTO training_models (
                    run_id, set_id, model_path, model_sha256, metrics_path, metrics_sha256,
                    status, final_metrics_json, metadata_json, created_utc
                )
                SELECT
                    run_id, set_id, model_path, model_sha256, metrics_path, metrics_sha256,
                    status, final_metrics_json, metadata_json, created_utc
                FROM detection_models
                """
            )
            cur.execute("DROP TABLE detection_models;")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_quality_dataset_id ON keypoint_quality(dataset_id);"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_keypoint_quality_gate
            ON keypoint_quality(review_state, review_intended_use, keypoint_method, usable_keypoints_rate);
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_quality_dataset_id ON detect_quality(dataset_id);"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_detect_quality_gate
            ON detect_quality(review_state, review_intended_use, detect_method, interpolated_detections_rate);
            """
        )
        # Ensure additive review columns exist before refreshing views on legacy registries.
        self._ensure_columns(
            "keypoint_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        self._ensure_columns(
            "detect_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_current;")
        cur.execute(
            """
            CREATE VIEW keypoint_quality_current AS
            WITH ranked AS (
                SELECT
                    kq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY kq.dataset_id, COALESCE(kq.keypoint_method, '')
                        ORDER BY
                            COALESCE(kq.review_timestamp_utc, kq.refined_created_utc, kq.quality_updated_utc) DESC,
                            COALESCE(kq.refined_created_utc, '') DESC,
                            kq.refined_run DESC
                    ) AS _rn
                FROM keypoint_quality kq
            )
            SELECT
                dataset_id,
                refined_run,
                refined_created_utc,
                source_keypoint_run,
                keypoint_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_notes,
                review_policy_id,
                review_policy_version,
                review_timestamp_utc,
                usable_keypoints,
                total_keypoints,
                usable_keypoints_rate,
                raw_keypoints_success_rate,
                raw_keypoints_successful,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_overview;")
        cur.execute(
            """
            CREATE VIEW keypoint_quality_overview AS
            SELECT
                kqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                kqc.keypoint_method AS keypoint_method,
                kqc.source_keypoint_run AS source_keypoint_run,
                kqc.refined_run AS refined_run,
                kqc.review_state AS review_state,
                kqc.review_method AS review_method,
                kqc.review_intended_use AS review_intended_use,
                kqc.review_policy_id AS review_policy_id,
                kqc.review_policy_version AS review_policy_version,
                kqc.usable_keypoints AS usable_keypoints,
                kqc.total_keypoints AS total_keypoints,
                kqc.usable_keypoints_rate AS usable_keypoints_rate,
                kqc.quality_updated_utc AS quality_updated_utc,
                kqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN kqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM keypoint_quality_current kqc
            LEFT JOIN datasets d ON d.dataset_id = kqc.dataset_id;
            """
        )
        cur.execute("DROP VIEW IF EXISTS detect_quality_current;")
        cur.execute(
            """
            CREATE VIEW detect_quality_current AS
            WITH ranked AS (
                SELECT
                    dq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dq.dataset_id, COALESCE(dq.detect_method, '')
                        ORDER BY
                            COALESCE(dq.review_timestamp_utc, dq.refined_created_utc, dq.quality_updated_utc) DESC,
                            COALESCE(dq.refined_created_utc, '') DESC,
                            dq.refined_run DESC
                    ) AS _rn
                FROM detect_quality dq
            )
            SELECT
                dataset_id,
                refined_run,
                refined_created_utc,
                source_detect_run,
                detect_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_notes,
                review_timestamp_utc,
                review_resolved_group,
                total_detections,
                real_detections,
                interpolated_detections,
                interpolated_detections_rate,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS refined_detect_review_current;")
        cur.execute(
            """
            CREATE VIEW refined_detect_review_current AS
            SELECT * FROM detect_quality_current;
            """
        )
        cur.execute("DROP VIEW IF EXISTS merged_training_datasets;")
        cur.execute(
            """
            CREATE VIEW merged_training_datasets AS
            SELECT
                d.dataset_id,
                d.recording_id,
                d.session_uuid,
                d.zarr_path,
                d.status,
                d.artifact_kind,
                d.zarr_origin,
                d.zarr_use,
                d.zarr_use AS zarr_purpose,
                d.last_seen_utc
            FROM datasets d
            WHERE
                d.artifact_kind = 'derived_training_merge'
                OR d.dataset_id LIKE '%_merged';
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_overview;")
        cur.execute(
            """
            CREATE VIEW recording_overview AS
            SELECT
                r.recording_id AS recording_id,
                r.session_uuid AS session_uuid,
                r.recording_name AS recording_name,
                r.recording_path AS recording_path,
                r.started_utc AS started_utc,
                r.recording_type AS recording_type,
                r.recording_subtype AS recording_subtype,
                r.behavior_mode AS behavior_mode,
                r.artifact_schema_id AS artifact_schema_id,
                r.experiment_context_status AS experiment_context_status,
                r.experiment_context_source AS experiment_context_source,
                r.experiment_context_status_detail AS experiment_context_status_detail,
                r.stimulus_runs_available AS stimulus_runs_available,
                COALESCE(
                    NULLIF(TRIM(r.dish_design), ''),
                    GROUP_CONCAT(DISTINCT NULLIF(TRIM(dcc.dish_design), ''))
                ) AS dish_design,
                r.rig_id AS rig_id,
                r.arena_id AS arena_id,
                r.camera_id AS camera_id,
                r.protocol_name AS protocol_name,
                COUNT(DISTINCT d.dataset_id) AS dataset_count,
                SUM(CASE WHEN d.zarr_use = 'training' THEN 1 ELSE 0 END) AS training_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'analysis' THEN 1 ELSE 0 END) AS analysis_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'inference' THEN 1 ELSE 0 END) AS inference_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'export' THEN 1 ELSE 0 END) AS export_dataset_count,
                SUM(CASE WHEN lower(COALESCE(d.status, 'active')) = 'active' THEN 1 ELSE 0 END) AS active_dataset_count,
                SUM(CASE WHEN lower(COALESCE(d.status, '')) = 'missing' THEN 1 ELSE 0 END) AS missing_dataset_count,
                COALESCE(MAX(d.last_seen_utc), r.updated_utc, r.created_utc) AS last_seen_utc
            FROM recordings r
            LEFT JOIN datasets d ON d.recording_id = r.recording_id
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = d.dataset_id
            GROUP BY
                r.recording_id,
                r.session_uuid,
                r.recording_name,
                r.recording_path,
                r.started_utc,
                r.recording_type,
                r.recording_subtype,
                r.behavior_mode,
                r.artifact_schema_id,
                r.experiment_context_status,
                r.experiment_context_source,
                r.experiment_context_status_detail,
                r.stimulus_runs_available,
                r.dish_design,
                r.rig_id,
                r.arena_id,
                r.camera_id,
                r.protocol_name;
            """
        )
        cur.execute("DROP VIEW IF EXISTS dataset_lineage_current;")
        cur.execute(
            """
            CREATE VIEW dataset_lineage_current AS
            WITH ranked AS (
                SELECT
                    dl.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dl.child_dataset_id, dl.parent_dataset_id, dl.relationship_type
                        ORDER BY COALESCE(dl.updated_utc, dl.created_utc) DESC
                    ) AS _rn
                FROM dataset_lineage dl
            )
            SELECT
                child_dataset_id,
                parent_dataset_id,
                relationship_type,
                source_set_id,
                metadata_json,
                created_utc,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )
        self._ensure_columns(
            "datasets",
            {
                "recording_id": "TEXT",
                "artifact_kind": "TEXT",
                "zarr_origin": "TEXT",
                "zarr_use": "TEXT",
            },
        )
        self._ensure_columns(
            "provenance",
            {
                "fish_id": "TEXT",
                "subject_count": "INTEGER",
                "rig_id": "TEXT",
                "arena_id": "TEXT",
                "camera_id": "TEXT",
                "canvas_name": "TEXT",
                "dish_design": "TEXT",
                "fps": "REAL",
                "video_codec": "TEXT",
                "video_pix_fmt": "TEXT",
                "format_title": "TEXT",
                "format_comment": "TEXT",
                "format_encoder": "TEXT",
                "encoder_name": "TEXT",
                "encoder_codec": "TEXT",
                "encoder_preset": "TEXT",
                "encoder_tuning": "TEXT",
                "encoder_rc": "TEXT",
                "encoder_bpp": "REAL",
                "encoder_target_bps": "INTEGER",
                "encoder_res": "TEXT",
                "encoder_res_width": "INTEGER",
                "encoder_res_height": "INTEGER",
                "encoder_fps": "REAL",
                "encoder_color": "INTEGER",
                "encoder_params_json": "TEXT",
                "source_video": "TEXT",
                "compression_name": "TEXT",
                "compression_level": "INTEGER",
                "exposure": "REAL",
                "exposure_unit": "TEXT",
                "gain": "REAL",
                "frame_rate": "REAL",
                "pixel_format": "TEXT",
                "binning": "TEXT",
                "adc": "TEXT",
                "camera_model": "TEXT",
                "camera_serial": "TEXT",
                "camera_metadata_json": "TEXT",
                "has_images_ds": "INTEGER",
                "has_images_ds_rgb": "INTEGER",
                "downsample_formats_json": "TEXT",
            },
        )
        # Backfill normalized zarr origin/use for legacy registries.
        self.conn.execute(
            """
            UPDATE datasets
            SET zarr_origin = CASE
                WHEN artifact_kind = 'source_recording' THEN 'source'
                WHEN artifact_kind IN ('derived_analysis', 'derived_training_merge', 'model_input_export') THEN 'derived'
                ELSE zarr_origin
            END
            WHERE zarr_origin IS NULL;
            """
        )
        self.conn.execute(
            """
            UPDATE datasets
            SET zarr_use = CASE
                WHEN artifact_kind = 'derived_training_merge' THEN 'training'
                WHEN artifact_kind = 'derived_analysis' THEN 'analysis'
                WHEN artifact_kind = 'model_input_export' THEN 'export'
                ELSE zarr_use
            END
            WHERE zarr_use IS NULL;
            """
        )
        self._ensure_columns(
            "detect_quality",
            {
                "refined_created_utc": "TEXT",
                "source_detect_run": "TEXT",
                "detect_method": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_notes": "TEXT",
                "review_timestamp_utc": "TEXT",
                "review_resolved_group": "TEXT",
                "total_detections": "INTEGER",
                "real_detections": "INTEGER",
                "interpolated_detections": "INTEGER",
                "interpolated_detections_rate": "REAL",
                "quality_updated_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
            },
        )
        self._ensure_columns(
            "keypoint_quality",
            {
                "refined_created_utc": "TEXT",
                "source_keypoint_run": "TEXT",
                "keypoint_method": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_notes": "TEXT",
                "review_timestamp_utc": "TEXT",
                "usable_keypoints": "INTEGER",
                "total_keypoints": "INTEGER",
                "usable_keypoints_rate": "REAL",
                "raw_keypoints_success_rate": "REAL",
                "raw_keypoints_successful": "INTEGER",
                "quality_updated_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
            },
        )
        self._ensure_columns(
            "training_sets",
            {"invocation_json": "TEXT", "skeleton_id": "TEXT", "task_type": "TEXT"},
        )
        self._ensure_columns(
            "training_runs",
            {
                "invocation_json": "TEXT",
                "skeleton_id": "TEXT",
                "task_type": "TEXT",
                "config_sha256": "TEXT",
                "manifest_sha256": "TEXT",
                "model_sha256": "TEXT",
                "metrics_sha256": "TEXT",
                "status": "TEXT",
                "final_metrics_json": "TEXT",
            },
        )
        self._ensure_columns(
            "training_models",
            {
                "set_id": "TEXT",
                "model_path": "TEXT",
                "model_sha256": "TEXT",
                "metrics_path": "TEXT",
                "metrics_sha256": "TEXT",
                "status": "TEXT",
                "input_shape": "TEXT",
                "input_layout": "TEXT",
                "input_channels": "INTEGER",
                "img_h": "INTEGER",
                "img_w": "INTEGER",
                "max_batch": "INTEGER",
                "dynamic_shapes": "INTEGER",
                "input_dtype": "TEXT",
                "input_color_space": "TEXT",
                "input_shape_source": "TEXT",
                "input_shape_status": "TEXT",
                "final_metrics_json": "TEXT",
                "metadata_json": "TEXT",
                "created_utc": "TEXT",
            },
        )
        self._ensure_columns(
            "onnx_models",
            {
                "set_id": "TEXT",
                "skeleton_id": "TEXT",
                "detection_model_run_id": "TEXT",
                "path": "TEXT",
                "sha256": "TEXT",
                "manifest_path": "TEXT",
                "manifest_sha256": "TEXT",
                "opset": "INTEGER",
                "nms_conf": "REAL",
                "nms_iou": "REAL",
                "nms_topk": "INTEGER",
                "input_shape": "TEXT",
                "img_h": "INTEGER",
                "img_w": "INTEGER",
                "max_batch": "INTEGER",
                "dynamic_shapes": "INTEGER",
                "file_size_bytes": "INTEGER",
                "exporter_torch_version": "TEXT",
                "exporter_cuda_version": "TEXT",
                "exporter_hostname": "TEXT",
                "requires_plugins": "INTEGER",
                "plugin_ops_json": "TEXT",
                "plugin_versions_json": "TEXT",
                "metadata_json": "TEXT",
                "created_utc": "TEXT",
            },
        )
        self._ensure_columns(
            "tensorrt_models",
            {
                "set_id": "TEXT",
                "skeleton_id": "TEXT",
                "detection_model_run_id": "TEXT",
                "onnx_run_id": "TEXT",
                "precision": "TEXT",
                "nms_conf": "REAL",
                "nms_iou": "REAL",
                "nms_topk": "INTEGER",
                "path": "TEXT",
                "sha256": "TEXT",
                "manifest_path": "TEXT",
                "manifest_sha256": "TEXT",
                "input_shape": "TEXT",
                "img_h": "INTEGER",
                "img_w": "INTEGER",
                "max_batch": "INTEGER",
                "dynamic_shapes": "INTEGER",
                "file_size_bytes": "INTEGER",
                "trt_version": "TEXT",
                "cuda_version": "TEXT",
                "compute_capability": "TEXT",
                "gpu_name": "TEXT",
                "gpu_uuid": "TEXT",
                "system_hostname": "TEXT",
                "requires_plugins": "INTEGER",
                "plugin_ops_json": "TEXT",
                "plugin_versions_json": "TEXT",
                "metadata_json": "TEXT",
                "created_utc": "TEXT",
            },
        )

    def _migration_002_reserved_noop(self) -> None:
        # Intentionally no-op. Serves as a stable template slot for future append-only migrations.
        return

    def _migration_003_recording_columns_reconcile(self) -> None:
        if not self._table_exists("recordings"):
            return
        # Legacy bootstrapped registries can skip migration_001 execution.
        # Reconcile additive recording columns needed by current maintenance flows.
        self._ensure_columns(
            "recordings",
            {
                "recording_subtype": "TEXT",
                "behavior_mode": "TEXT",
                "dish_design": "TEXT",
                "experiment_context_status": "TEXT",
                "experiment_context_source": "TEXT",
                "experiment_context_status_detail": "TEXT",
                "stimulus_runs_available": "INTEGER",
            },
        )
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS recording_overview;")
        cur.execute(
            """
            CREATE VIEW recording_overview AS
            SELECT
                r.recording_id AS recording_id,
                r.session_uuid AS session_uuid,
                r.recording_name AS recording_name,
                r.recording_path AS recording_path,
                r.started_utc AS started_utc,
                r.recording_type AS recording_type,
                r.recording_subtype AS recording_subtype,
                r.behavior_mode AS behavior_mode,
                r.artifact_schema_id AS artifact_schema_id,
                r.experiment_context_status AS experiment_context_status,
                r.experiment_context_source AS experiment_context_source,
                r.experiment_context_status_detail AS experiment_context_status_detail,
                r.stimulus_runs_available AS stimulus_runs_available,
                COALESCE(
                    NULLIF(TRIM(r.dish_design), ''),
                    GROUP_CONCAT(DISTINCT NULLIF(TRIM(dcc.dish_design), ''))
                ) AS dish_design,
                r.rig_id AS rig_id,
                r.arena_id AS arena_id,
                r.camera_id AS camera_id,
                r.protocol_name AS protocol_name,
                COUNT(DISTINCT d.dataset_id) AS dataset_count,
                SUM(CASE WHEN d.zarr_use = 'training' THEN 1 ELSE 0 END) AS training_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'analysis' THEN 1 ELSE 0 END) AS analysis_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'inference' THEN 1 ELSE 0 END) AS inference_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'export' THEN 1 ELSE 0 END) AS export_dataset_count,
                SUM(CASE WHEN lower(COALESCE(d.status, 'active')) = 'active' THEN 1 ELSE 0 END) AS active_dataset_count,
                SUM(CASE WHEN lower(COALESCE(d.status, '')) = 'missing' THEN 1 ELSE 0 END) AS missing_dataset_count,
                COALESCE(MAX(d.last_seen_utc), r.updated_utc, r.created_utc) AS last_seen_utc
            FROM recordings r
            LEFT JOIN datasets d ON d.recording_id = r.recording_id
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = d.dataset_id
            GROUP BY
                r.recording_id,
                r.session_uuid,
                r.recording_name,
                r.recording_path,
                r.started_utc,
                r.recording_type,
                r.recording_subtype,
                r.behavior_mode,
                r.artifact_schema_id,
                r.experiment_context_status,
                r.experiment_context_source,
                r.experiment_context_status_detail,
                r.stimulus_runs_available,
                r.dish_design,
                r.rig_id,
                r.arena_id,
                r.camera_id,
                r.protocol_name;
            """
        )

    def _migration_004_recording_overview_refresh(self) -> None:
        # Refresh view definition for registries that already applied v3.
        self._migration_003_recording_columns_reconcile()

    def _migration_005_drop_provenance_zarr_purpose(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_overview;")
        cur.execute("DROP VIEW IF EXISTS merged_training_datasets;")

        provenance_cols = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(provenance);").fetchall()
            if row["name"] is not None
        }
        if "zarr_purpose" in provenance_cols:
            cur.execute("ALTER TABLE provenance DROP COLUMN zarr_purpose;")

        cur.execute(
            """
            CREATE VIEW keypoint_quality_overview AS
            SELECT
                kqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                kqc.keypoint_method AS keypoint_method,
                kqc.source_keypoint_run AS source_keypoint_run,
                kqc.refined_run AS refined_run,
                kqc.review_state AS review_state,
                kqc.review_method AS review_method,
                kqc.review_intended_use AS review_intended_use,
                kqc.review_policy_id AS review_policy_id,
                kqc.review_policy_version AS review_policy_version,
                kqc.usable_keypoints AS usable_keypoints,
                kqc.total_keypoints AS total_keypoints,
                kqc.usable_keypoints_rate AS usable_keypoints_rate,
                kqc.quality_updated_utc AS quality_updated_utc,
                kqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN kqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM keypoint_quality_current kqc
            LEFT JOIN datasets d ON d.dataset_id = kqc.dataset_id;
            """
        )
        cur.execute(
            """
            CREATE VIEW merged_training_datasets AS
            SELECT
                d.dataset_id,
                d.recording_id,
                d.session_uuid,
                d.zarr_path,
                d.status,
                d.artifact_kind,
                d.zarr_origin,
                d.zarr_use,
                d.zarr_use AS zarr_purpose,
                d.last_seen_utc
            FROM datasets d
            WHERE
                d.artifact_kind = 'derived_training_merge'
                OR d.dataset_id LIKE '%_merged';
            """
        )

    def _migration_006_subject_dish_cross_entities(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS crosses (
                cross_id TEXT PRIMARY KEY,
                line_strain TEXT,
                genotype TEXT,
                parents_json TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dishes (
                dish_id TEXT PRIMARY KEY,
                cross_id TEXT,
                species TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                FOREIGN KEY(cross_id) REFERENCES crosses(cross_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_subjects (
                recording_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                dataset_id TEXT,
                dish_id TEXT,
                cross_id TEXT,
                dpf_at_acquisition INTEGER,
                species TEXT,
                sex TEXT,
                genotype TEXT,
                line_strain TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                PRIMARY KEY (recording_id, subject_id),
                FOREIGN KEY(recording_id) REFERENCES recordings(recording_id) ON DELETE CASCADE,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE SET NULL,
                FOREIGN KEY(dish_id) REFERENCES dishes(dish_id),
                FOREIGN KEY(cross_id) REFERENCES crosses(cross_id)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dishes_cross_id ON dishes(cross_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_dataset_id ON recording_subjects(dataset_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_dish_id ON recording_subjects(dish_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_cross_id ON recording_subjects(cross_id);"
        )

    def _migration_007_subjects_entities_and_query_indexes(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS subjects (
                subject_id TEXT PRIMARY KEY,
                dish_id TEXT,
                species TEXT,
                sex TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                FOREIGN KEY(dish_id) REFERENCES dishes(dish_id)
            );
            """
        )
        # Common query path: recording_subjects -> subjects -> dishes -> crosses(genotype).
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_crosses_genotype ON crosses(genotype);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subjects_dish_id ON subjects(dish_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_subject_dpf ON recording_subjects(subject_id, dpf_at_acquisition);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_recording_id ON recording_subjects(recording_id);"
        )

    def _migration_008_recording_subject_overview_view(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS recording_subject_overview;")
        cur.execute(
            """
            CREATE VIEW recording_subject_overview AS
            SELECT
                rs.recording_id AS recording_id,
                rs.subject_id AS subject_id,
                rs.dataset_id AS dataset_id,
                COALESCE(rs.dish_id, s.dish_id) AS dish_id,
                COALESCE(rs.cross_id, d.cross_id) AS cross_id,
                c.genotype AS genotype,
                c.line_strain AS line_strain,
                rs.dpf_at_acquisition AS dpf_at_acquisition,
                COALESCE(rs.species, s.species, d.species) AS species,
                COALESCE(rs.sex, s.sex) AS sex,
                r.started_utc AS recording_started_utc,
                r.recording_type AS recording_type,
                r.recording_subtype AS recording_subtype,
                r.behavior_mode AS behavior_mode,
                r.protocol_name AS protocol_name,
                r.rig_id AS rig_id,
                r.arena_id AS arena_id,
                r.camera_id AS camera_id
            FROM recording_subjects rs
            LEFT JOIN subjects s
              ON s.subject_id = rs.subject_id
            LEFT JOIN dishes d
              ON d.dish_id = COALESCE(rs.dish_id, s.dish_id)
            LEFT JOIN crosses c
              ON c.cross_id = COALESCE(rs.cross_id, d.cross_id)
            LEFT JOIN recordings r
              ON r.recording_id = rs.recording_id;
            """
        )

    def _migration_009_training_task_type_columns(self) -> None:
        self._ensure_columns("training_sets", {"task_type": "TEXT"})
        self._ensure_columns("training_runs", {"task_type": "TEXT"})
        cur = self.conn.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_sets_task_type ON training_sets(task_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_runs_task_type ON training_runs(task_type);"
        )

        set_rows = self.conn.execute(
            "SELECT set_id, task_type, query_filter, invocation_json FROM training_sets;"
        ).fetchall()
        for row in set_rows:
            if _normalize_task_type(row["task_type"]):
                continue
            query_filter = _json_loads(row["query_filter"])
            invocation = _json_loads(row["invocation_json"])
            inferred = _infer_task_type(
                set_id=row["set_id"],
                query_filter=query_filter,
                invocation=invocation,
            )
            if inferred:
                self.conn.execute(
                    "UPDATE training_sets SET task_type = ? WHERE set_id = ?;",
                    (inferred, str(row["set_id"])),
                )

        run_rows = self.conn.execute(
            """
            SELECT
                tr.run_id,
                tr.set_id,
                tr.task_type,
                tr.config_path,
                tr.manifest_path,
                tr.model_path,
                tr.invocation_json,
                ts.task_type AS set_task_type
            FROM training_runs tr
            LEFT JOIN training_sets ts ON ts.set_id = tr.set_id;
            """
        ).fetchall()
        for row in run_rows:
            if _normalize_task_type(row["task_type"]):
                continue
            invocation = _json_loads(row["invocation_json"])
            inferred = _infer_task_type(
                set_id=row["set_id"],
                run_id=row["run_id"],
                config_path=row["config_path"],
                manifest_path=row["manifest_path"],
                model_path=row["model_path"],
                invocation=invocation,
                explicit=row["set_task_type"],
            )
            if inferred:
                self.conn.execute(
                    "UPDATE training_runs SET task_type = ? WHERE run_id = ?;",
                    (inferred, str(row["run_id"])),
                )

    def _migration_010_detect_performance_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detect_performance (
                dataset_id TEXT NOT NULL,
                detect_run TEXT NOT NULL,
                detect_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                detection_method TEXT,
                model_run_id TEXT,
                model_set_id TEXT,
                model_path TEXT,
                model_name TEXT,
                coverage_percent REAL,
                frames_with_detections INTEGER,
                frames_zero_detections INTEGER,
                total_frames INTEGER,
                mean_confidence REAL,
                min_confidence REAL,
                max_confidence REAL,
                inference_duration_seconds REAL,
                inference_average_fps REAL,
                inference_avg_batch_ms REAL,
                inference_avg_read_ms REAL,
                conf_threshold REAL,
                iou_threshold REAL,
                batch_size INTEGER,
                inference_width INTEGER,
                inference_height INTEGER,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, detect_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "detect_performance",
            {
                "detect_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "detection_method": "TEXT",
                "model_run_id": "TEXT",
                "model_set_id": "TEXT",
                "model_path": "TEXT",
                "model_name": "TEXT",
                "coverage_percent": "REAL",
                "frames_with_detections": "INTEGER",
                "frames_zero_detections": "INTEGER",
                "total_frames": "INTEGER",
                "mean_confidence": "REAL",
                "min_confidence": "REAL",
                "max_confidence": "REAL",
                "inference_duration_seconds": "REAL",
                "inference_average_fps": "REAL",
                "inference_avg_batch_ms": "REAL",
                "inference_avg_read_ms": "REAL",
                "conf_threshold": "REAL",
                "iou_threshold": "REAL",
                "batch_size": "INTEGER",
                "inference_width": "INTEGER",
                "inference_height": "INTEGER",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_recording ON detect_performance(recording_id, detect_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_coverage ON detect_performance(coverage_percent);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_runtime ON detect_performance(inference_average_fps, inference_avg_read_ms);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_method ON detect_performance(detection_method, model_name);"
        )
        cur.execute("DROP VIEW IF EXISTS detect_performance_latest;")
        cur.execute(
            """
            CREATE VIEW detect_performance_latest AS
            WITH ranked AS (
                SELECT
                    dp.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dp.dataset_id
                        ORDER BY
                            COALESCE(dp.detect_created_utc, dp.updated_utc) DESC,
                            dp.detect_run DESC
                    ) AS _rn
                FROM detect_performance dp
            )
            SELECT
                dataset_id,
                detect_run,
                detect_created_utc,
                recording_id,
                zarr_use,
                detection_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                coverage_percent,
                frames_with_detections,
                frames_zero_detections,
                total_frames,
                mean_confidence,
                min_confidence,
                max_confidence,
                inference_duration_seconds,
                inference_average_fps,
                inference_avg_batch_ms,
                inference_avg_read_ms,
                conf_threshold,
                iou_threshold,
                batch_size,
                inference_width,
                inference_height,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_detect_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_detect_performance_latest AS
            WITH ranked AS (
                SELECT
                    dpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY dpl.recording_id
                        ORDER BY
                            COALESCE(dpl.detect_created_utc, dpl.updated_utc) DESC,
                            dpl.detect_run DESC
                    ) AS _rn
                FROM detect_performance_latest dpl
                LEFT JOIN datasets d ON d.dataset_id = dpl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = dpl.dataset_id
                WHERE dpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                detect_run,
                detect_created_utc,
                zarr_use,
                detection_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                coverage_percent,
                frames_with_detections,
                frames_zero_detections,
                total_frames,
                mean_confidence,
                min_confidence,
                max_confidence,
                inference_duration_seconds,
                inference_average_fps,
                inference_avg_batch_ms,
                inference_avg_read_ms,
                conf_threshold,
                iou_threshold,
                batch_size,
                inference_width,
                inference_height,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_011_detect_model_performance_views(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_model_path ON detect_performance(model_path, model_name, detect_created_utc);"
        )

        cur.execute("DROP VIEW IF EXISTS detect_model_performance_latest;")
        cur.execute(
            """
            CREATE VIEW detect_model_performance_latest AS
            WITH model_rows AS (
                SELECT dp.*
                FROM detect_performance dp
                WHERE
                    trim(COALESCE(dp.model_path, '')) <> ''
                    OR trim(COALESCE(dp.model_name, '')) <> ''
            ),
            ranked AS (
                SELECT
                    mr.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY mr.dataset_id
                        ORDER BY
                            COALESCE(mr.detect_created_utc, mr.updated_utc) DESC,
                            mr.detect_run DESC
                    ) AS _rn
                FROM model_rows mr
            )
            SELECT
                dataset_id,
                detect_run,
                detect_created_utc,
                recording_id,
                zarr_use,
                detection_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                coverage_percent,
                frames_with_detections,
                frames_zero_detections,
                total_frames,
                mean_confidence,
                min_confidence,
                max_confidence,
                inference_duration_seconds,
                inference_average_fps,
                inference_avg_batch_ms,
                inference_avg_read_ms,
                conf_threshold,
                iou_threshold,
                batch_size,
                inference_width,
                inference_height,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_detect_model_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_detect_model_performance_latest AS
            WITH ranked AS (
                SELECT
                    dmpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY dmpl.recording_id
                        ORDER BY
                            COALESCE(dmpl.detect_created_utc, dmpl.updated_utc) DESC,
                            dmpl.detect_run DESC
                    ) AS _rn
                FROM detect_model_performance_latest dmpl
                LEFT JOIN datasets d ON d.dataset_id = dmpl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = dmpl.dataset_id
                WHERE dmpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                detect_run,
                detect_created_utc,
                zarr_use,
                detection_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                coverage_percent,
                frames_with_detections,
                frames_zero_detections,
                total_frames,
                mean_confidence,
                min_confidence,
                max_confidence,
                inference_duration_seconds,
                inference_average_fps,
                inference_avg_batch_ms,
                inference_avg_read_ms,
                conf_threshold,
                iou_threshold,
                batch_size,
                inference_width,
                inference_height,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_012_detect_performance_model_identity(self) -> None:
        # Additive migration: add run/set identity columns and rebuild detect
        # performance views to project them.
        self._ensure_columns(
            "detect_performance",
            {
                "model_run_id": "TEXT",
                "model_set_id": "TEXT",
            },
        )
        self._migration_010_detect_performance_registry()
        self._migration_011_detect_model_performance_views()

    def _migration_013_detect_model_performance_summary_views(self) -> None:
        # Build additive summary views over model-backed latest detect performance.
        self._migration_011_detect_model_performance_views()
        self._create_detect_model_performance_summary_view(
            source_view="detect_model_performance_latest",
            target_view="detect_model_performance_summary",
        )
        self._create_detect_model_performance_summary_view(
            source_view="recording_detect_model_performance_latest",
            target_view="recording_detect_model_performance_summary",
        )

    def _migration_014_crop_quality_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS crop_quality (
                dataset_id TEXT NOT NULL,
                crop_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                crop_created_utc TEXT,
                source_detect_run TEXT,
                source_refined_run TEXT,
                detection_source_type TEXT,
                detection_source_path TEXT,
                crop_storage_mode TEXT,
                roi_image_representation TEXT,
                roi_pixel_contract_name TEXT,
                roi_pixel_contract_json TEXT,
                total_rois INTEGER,
                frames_with_crops INTEGER,
                total_frames INTEGER,
                percent_frames_with_crops REAL,
                includes_interpolated INTEGER,
                n_real_detections INTEGER,
                n_interpolated_detections INTEGER,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                review_notes TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, crop_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "crop_quality",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "crop_created_utc": "TEXT",
                "source_detect_run": "TEXT",
                "source_refined_run": "TEXT",
                "detection_source_type": "TEXT",
                "detection_source_path": "TEXT",
                "crop_storage_mode": "TEXT",
                "roi_image_representation": "TEXT",
                "roi_pixel_contract_name": "TEXT",
                "roi_pixel_contract_json": "TEXT",
                "total_rois": "INTEGER",
                "frames_with_crops": "INTEGER",
                "total_frames": "INTEGER",
                "percent_frames_with_crops": "REAL",
                "includes_interpolated": "INTEGER",
                "n_real_detections": "INTEGER",
                "n_interpolated_detections": "INTEGER",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "review_notes": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_crop_quality_dataset_id ON crop_quality(dataset_id);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_crop_quality_review_gate ON crop_quality(review_state, review_intended_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_crop_quality_source ON crop_quality(detection_source_type, source_refined_run);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_crop_quality_recording ON crop_quality(recording_id, crop_created_utc DESC);"
        )

        cur.execute("DROP VIEW IF EXISTS crop_quality_current;")
        cur.execute(
            """
            CREATE VIEW crop_quality_current AS
            WITH ranked AS (
                SELECT
                    cq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY cq.dataset_id
                        ORDER BY
                            COALESCE(cq.review_timestamp_utc, cq.crop_created_utc, cq.updated_utc) DESC,
                            COALESCE(cq.crop_created_utc, '') DESC,
                            cq.crop_run DESC
                    ) AS _rn
                FROM crop_quality cq
            )
            SELECT
                dataset_id,
                crop_run,
                recording_id,
                zarr_use,
                crop_created_utc,
                source_detect_run,
                source_refined_run,
                detection_source_type,
                detection_source_path,
                crop_storage_mode,
                roi_image_representation,
                roi_pixel_contract_name,
                roi_pixel_contract_json,
                total_rois,
                frames_with_crops,
                total_frames,
                percent_frames_with_crops,
                includes_interpolated,
                n_real_detections,
                n_interpolated_detections,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                review_notes,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_crop_quality_current;")
        cur.execute(
            """
            CREATE VIEW recording_crop_quality_current AS
            WITH ranked AS (
                SELECT
                    cqc.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY cqc.recording_id
                        ORDER BY
                            COALESCE(cqc.review_timestamp_utc, cqc.crop_created_utc, cqc.updated_utc) DESC,
                            COALESCE(cqc.crop_created_utc, '') DESC,
                            cqc.crop_run DESC
                    ) AS _rn
                FROM crop_quality_current cqc
                LEFT JOIN datasets d ON d.dataset_id = cqc.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = cqc.dataset_id
                WHERE cqc.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                crop_run,
                zarr_use,
                crop_created_utc,
                source_detect_run,
                source_refined_run,
                detection_source_type,
                detection_source_path,
                crop_storage_mode,
                roi_image_representation,
                roi_pixel_contract_name,
                roi_pixel_contract_json,
                total_rois,
                frames_with_crops,
                total_frames,
                percent_frames_with_crops,
                includes_interpolated,
                n_real_detections,
                n_interpolated_detections,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                review_notes,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_054_crop_quality_pixel_contract_columns(self) -> None:
        """Expose crop pixel-contract metadata in crop-quality registry views."""

        self._migration_014_crop_quality_registry()

    def _migration_055_keypoint_performance_pixel_contract_columns(self) -> None:
        """Expose source ROI pixel-contract metadata in keypoint-performance views."""

        self._migration_018_keypoint_performance_registry()

    def _migration_056_acquisition_video_streams_registry(self) -> None:
        """Expose recording-level acquisition video stream availability."""

        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS acquisition_video_streams (
                dataset_id TEXT NOT NULL,
                stream_key TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                stream_id TEXT,
                role TEXT,
                output_kind TEXT,
                source TEXT,
                camera_id TEXT,
                frame_clock TEXT,
                video_path TEXT,
                metadata_path TEXT,
                frame_clock_metadata_path TEXT,
                keyframes_path TEXT,
                summary_path TEXT,
                status_path TEXT,
                width INTEGER,
                height INTEGER,
                frame_count INTEGER,
                frame_rate REAL,
                codec TEXT,
                container TEXT,
                encoded_format TEXT,
                pixel_source_format TEXT,
                video_pixel_coordinate_space TEXT,
                source_geometry_coordinate_space TEXT,
                blank_frame_policy TEXT,
                selection_policy TEXT,
                availability_status TEXT,
                inventory_status TEXT,
                video_exists INTEGER,
                metadata_exists INTEGER,
                frame_clock_metadata_exists INTEGER,
                keyframes_exists INTEGER,
                summary_exists INTEGER,
                status_exists INTEGER,
                metadata_row_count INTEGER,
                frame_clock_metadata_row_count INTEGER,
                frames_encoded INTEGER,
                frames_dropped INTEGER,
                contract_json TEXT,
                files_json TEXT,
                summary_json TEXT,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, stream_key),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_acquisition_video_streams_recording "
            "ON acquisition_video_streams(recording_id, stream_key);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_acquisition_video_streams_availability "
            "ON acquisition_video_streams(output_kind, availability_status, video_exists);"
        )

        cur.execute("DROP VIEW IF EXISTS dataset_acquisition_video_streams_current;")
        cur.execute(
            """
            CREATE VIEW dataset_acquisition_video_streams_current AS
            SELECT
                avs.dataset_id,
                dcc.zarr_path AS zarr_path,
                COALESCE(dcc.recording_id, avs.recording_id) AS recording_id,
                COALESCE(dcc.zarr_use, avs.zarr_use) AS zarr_use,
                avs.stream_key,
                avs.stream_id,
                avs.role,
                avs.output_kind,
                avs.source,
                COALESCE(dcc.camera_id, avs.camera_id) AS camera_id,
                avs.frame_clock,
                avs.video_path,
                avs.metadata_path,
                avs.frame_clock_metadata_path,
                avs.keyframes_path,
                avs.summary_path,
                avs.status_path,
                avs.width,
                avs.height,
                avs.frame_count,
                avs.frame_rate,
                avs.codec,
                avs.container,
                avs.encoded_format,
                avs.pixel_source_format,
                avs.video_pixel_coordinate_space,
                avs.source_geometry_coordinate_space,
                avs.blank_frame_policy,
                avs.selection_policy,
                avs.availability_status,
                avs.inventory_status,
                avs.video_exists,
                avs.metadata_exists,
                avs.frame_clock_metadata_exists,
                avs.keyframes_exists,
                avs.summary_exists,
                avs.status_exists,
                avs.metadata_row_count,
                avs.frame_clock_metadata_row_count,
                avs.frames_encoded,
                avs.frames_dropped,
                avs.contract_json,
                avs.files_json,
                avs.summary_json,
                avs.updated_utc
            FROM acquisition_video_streams avs
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = avs.dataset_id;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_acquisition_video_streams_current;")
        cur.execute(
            """
            CREATE VIEW recording_acquisition_video_streams_current AS
            WITH ranked AS (
                SELECT
                    davs.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY davs.recording_id, davs.stream_key, COALESCE(davs.camera_id, '')
                        ORDER BY
                            CASE WHEN davs.zarr_use = 'analysis' THEN 0 ELSE 1 END,
                            COALESCE(davs.updated_utc, '') DESC,
                            davs.dataset_id DESC
                    ) AS _rn
                FROM dataset_acquisition_video_streams_current davs
                WHERE davs.recording_id IS NOT NULL
            )
            SELECT
                dataset_id,
                zarr_path,
                recording_id,
                zarr_use,
                stream_key,
                stream_id,
                role,
                output_kind,
                source,
                camera_id,
                frame_clock,
                video_path,
                metadata_path,
                frame_clock_metadata_path,
                keyframes_path,
                summary_path,
                status_path,
                width,
                height,
                frame_count,
                frame_rate,
                codec,
                container,
                encoded_format,
                pixel_source_format,
                video_pixel_coordinate_space,
                source_geometry_coordinate_space,
                blank_frame_policy,
                selection_policy,
                availability_status,
                inventory_status,
                video_exists,
                metadata_exists,
                frame_clock_metadata_exists,
                keyframes_exists,
                summary_exists,
                status_exists,
                metadata_row_count,
                frame_clock_metadata_row_count,
                frames_encoded,
                frames_dropped,
                contract_json,
                files_json,
                summary_json,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_crop_video_available_current;")
        cur.execute(
            """
            CREATE VIEW recording_crop_video_available_current AS
            SELECT
                recording_id,
                dataset_id,
                zarr_path,
                stream_key,
                stream_id,
                camera_id,
                CASE
                    WHEN output_kind = 'crop'
                     AND availability_status = 'ok'
                     AND COALESCE(video_exists, 0) = 1
                    THEN 1
                    ELSE 0
                END AS crop_stream_available,
                availability_status,
                inventory_status,
                video_path,
                metadata_path,
                width,
                height,
                frame_count,
                frame_rate,
                codec,
                encoded_format,
                pixel_source_format,
                video_pixel_coordinate_space,
                source_geometry_coordinate_space,
                blank_frame_policy,
                selection_policy,
                metadata_row_count,
                frames_encoded,
                frames_dropped,
                updated_utc
            FROM recording_acquisition_video_streams_current
            WHERE output_kind = 'crop';
            """
        )

    def _migration_015_eye_mask_performance_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS eye_mask_performance (
                dataset_id TEXT NOT NULL,
                stage_group TEXT NOT NULL,
                run_name TEXT NOT NULL,
                run_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                method TEXT,
                source_crop_run TEXT,
                source_keypoint_group TEXT,
                source_keypoints_run TEXT,
                source_eye_masks_run TEXT,
                source_eye_masks_method TEXT,
                total_rois INTEGER,
                successful_eyes INTEGER,
                successful_roi_pairs INTEGER,
                successful_roi_pair_rate REAL,
                duration_seconds REAL,
                rois_per_second REAL,
                inference_duration_seconds REAL,
                inference_average_fps REAL,
                reason_counts_json TEXT,
                summary_statistics_json TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                source_keypoint_stale_state TEXT,
                source_keypoint_stale_reason TEXT,
                source_keypoint_stale_timestamp_utc TEXT,
                source_keypoint_stale_json TEXT,
                lifecycle_state TEXT,
                lifecycle_reason TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, stage_group, run_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "eye_mask_performance",
            {
                "run_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "method": "TEXT",
                "source_crop_run": "TEXT",
                "source_keypoint_group": "TEXT",
                "source_keypoints_run": "TEXT",
                "source_eye_masks_run": "TEXT",
                "source_eye_masks_method": "TEXT",
                "total_rois": "INTEGER",
                "successful_eyes": "INTEGER",
                "successful_roi_pairs": "INTEGER",
                "successful_roi_pair_rate": "REAL",
                "duration_seconds": "REAL",
                "rois_per_second": "REAL",
                "inference_duration_seconds": "REAL",
                "inference_average_fps": "REAL",
                "reason_counts_json": "TEXT",
                "summary_statistics_json": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "source_keypoint_stale_state": "TEXT",
                "source_keypoint_stale_reason": "TEXT",
                "source_keypoint_stale_timestamp_utc": "TEXT",
                "source_keypoint_stale_json": "TEXT",
                "lifecycle_state": "TEXT",
                "lifecycle_reason": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_recording ON eye_mask_performance(recording_id, stage_group, run_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_stage_method ON eye_mask_performance(stage_group, method);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_runtime ON eye_mask_performance(rois_per_second, duration_seconds);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_source ON eye_mask_performance(source_keypoints_run, source_eye_masks_run);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_review ON eye_mask_performance(review_state, review_intended_use, lifecycle_state);"
        )

        cur.execute("DROP VIEW IF EXISTS eye_mask_performance_latest;")
        cur.execute(
            """
            CREATE VIEW eye_mask_performance_latest AS
            WITH ranked AS (
                SELECT
                    emp.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY emp.dataset_id, emp.stage_group
                        ORDER BY
                            COALESCE(emp.run_created_utc, emp.updated_utc) DESC,
                            emp.run_name DESC
                    ) AS _rn
                FROM eye_mask_performance emp
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                recording_id,
                zarr_use,
                method,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_eye_masks_run,
                source_eye_masks_method,
                total_rois,
                successful_eyes,
                successful_roi_pairs,
                successful_roi_pair_rate,
                duration_seconds,
                rois_per_second,
                inference_duration_seconds,
                inference_average_fps,
                reason_counts_json,
                summary_statistics_json,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                lifecycle_state,
                lifecycle_reason,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_eye_mask_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_eye_mask_performance_latest AS
            WITH ranked AS (
                SELECT
                    empl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY empl.recording_id, empl.stage_group
                        ORDER BY
                            COALESCE(empl.run_created_utc, empl.updated_utc) DESC,
                            empl.run_name DESC
                    ) AS _rn
                FROM eye_mask_performance_latest empl
                LEFT JOIN datasets d ON d.dataset_id = empl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = empl.dataset_id
                WHERE empl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                zarr_use,
                method,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_eye_masks_run,
                source_eye_masks_method,
                total_rois,
                successful_eyes,
                successful_roi_pairs,
                successful_roi_pair_rate,
                duration_seconds,
                rois_per_second,
                inference_duration_seconds,
                inference_average_fps,
                reason_counts_json,
                summary_statistics_json,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                lifecycle_state,
                lifecycle_reason,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_016_model_export_nms_threshold_columns(self) -> None:
        self._ensure_columns(
            "onnx_models",
            {
                "nms_conf": "REAL",
                "nms_iou": "REAL",
                "nms_topk": "INTEGER",
            },
        )
        self._ensure_columns(
            "tensorrt_models",
            {
                "nms_conf": "REAL",
                "nms_iou": "REAL",
                "nms_topk": "INTEGER",
            },
        )

        # Fast-path backfill from persisted metadata JSON.
        self.conn.execute(
            """
            UPDATE onnx_models
            SET
                nms_conf = COALESCE(
                    nms_conf,
                    json_extract(metadata_json, '$.nms.conf'),
                    json_extract(metadata_json, '$.nms_conf'),
                    json_extract(metadata_json, '$.conf_threshold')
                ),
                nms_iou = COALESCE(
                    nms_iou,
                    json_extract(metadata_json, '$.nms.iou'),
                    json_extract(metadata_json, '$.nms_iou'),
                    json_extract(metadata_json, '$.iou_threshold')
                ),
                nms_topk = COALESCE(
                    nms_topk,
                    json_extract(metadata_json, '$.nms.topk'),
                    json_extract(metadata_json, '$.nms_topk'),
                    json_extract(metadata_json, '$.topk')
                )
            WHERE nms_conf IS NULL OR nms_iou IS NULL OR nms_topk IS NULL;
            """
        )
        self.conn.execute(
            """
            UPDATE tensorrt_models
            SET
                nms_conf = COALESCE(
                    nms_conf,
                    json_extract(metadata_json, '$.nms.conf'),
                    json_extract(metadata_json, '$.nms_conf'),
                    json_extract(metadata_json, '$.conf_threshold')
                ),
                nms_iou = COALESCE(
                    nms_iou,
                    json_extract(metadata_json, '$.nms.iou'),
                    json_extract(metadata_json, '$.nms_iou'),
                    json_extract(metadata_json, '$.iou_threshold')
                ),
                nms_topk = COALESCE(
                    nms_topk,
                    json_extract(metadata_json, '$.nms.topk'),
                    json_extract(metadata_json, '$.nms_topk'),
                    json_extract(metadata_json, '$.topk')
                )
            WHERE nms_conf IS NULL OR nms_iou IS NULL OR nms_topk IS NULL;
            """
        )

        # Slow-path backfill from manifest files when metadata JSON did not include NMS.
        onnx_rows = self.conn.execute(
            """
            SELECT run_id, manifest_path, metadata_json, nms_conf, nms_iou, nms_topk
            FROM onnx_models;
            """
        ).fetchall()
        for row in onnx_rows:
            if (
                row["nms_conf"] is not None
                and row["nms_iou"] is not None
                and row["nms_topk"] is not None
            ):
                continue
            metadata = _json_loads(row["metadata_json"])
            metadata_map = metadata if isinstance(metadata, dict) else None
            manifest_path_text = row["manifest_path"]
            manifest_payload = self._read_json_path(Path(str(manifest_path_text))) if manifest_path_text else {}
            nms_conf, nms_iou, nms_topk = self._extract_nms_thresholds(
                manifest_payload=manifest_payload,
                metadata=metadata_map,
            )
            if nms_conf is None and nms_iou is None and nms_topk is None:
                continue
            self.conn.execute(
                """
                UPDATE onnx_models
                SET
                    nms_conf = COALESCE(nms_conf, ?),
                    nms_iou = COALESCE(nms_iou, ?),
                    nms_topk = COALESCE(nms_topk, ?)
                WHERE run_id = ?;
                """,
                (nms_conf, nms_iou, nms_topk, str(row["run_id"])),
            )

        trt_rows = self.conn.execute(
            """
            SELECT run_id, precision, manifest_path, metadata_json, nms_conf, nms_iou, nms_topk
            FROM tensorrt_models;
            """
        ).fetchall()
        for row in trt_rows:
            if (
                row["nms_conf"] is not None
                and row["nms_iou"] is not None
                and row["nms_topk"] is not None
            ):
                continue
            metadata = _json_loads(row["metadata_json"])
            metadata_map = metadata if isinstance(metadata, dict) else None
            manifest_path_text = row["manifest_path"]
            manifest_payload = self._read_json_path(Path(str(manifest_path_text))) if manifest_path_text else {}
            nms_conf, nms_iou, nms_topk = self._extract_nms_thresholds(
                manifest_payload=manifest_payload,
                metadata=metadata_map,
            )
            if nms_conf is None and nms_iou is None and nms_topk is None:
                continue
            self.conn.execute(
                """
                UPDATE tensorrt_models
                SET
                    nms_conf = COALESCE(nms_conf, ?),
                    nms_iou = COALESCE(nms_iou, ?),
                    nms_topk = COALESCE(nms_topk, ?)
                WHERE run_id = ? AND precision = ?;
                """,
                (
                    nms_conf,
                    nms_iou,
                    nms_topk,
                    str(row["run_id"]),
                    str(row["precision"] or "fp16"),
                ),
            )

    def _migration_017_eye_mask_performance_review_stale_columns(self) -> None:
        # Additive refresh of eye-mask performance schema/views for review + stale reconciliation.
        self._migration_015_eye_mask_performance_registry()

    def _migration_018_keypoint_performance_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS keypoint_performance (
                dataset_id TEXT NOT NULL,
                keypoint_run TEXT NOT NULL,
                keypoint_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                keypoint_method TEXT,
                model_run_id TEXT,
                model_set_id TEXT,
                model_path TEXT,
                model_name TEXT,
                source_crop_run TEXT,
                source_detect_run TEXT,
                source_refined_run TEXT,
                source_crop_storage_mode TEXT,
                source_crop_signature TEXT,
                source_crop_revision INTEGER,
                source_roi_image_representation TEXT,
                source_roi_pixel_contract_name TEXT,
                source_roi_pixel_contract_json TEXT,
                source_roi_read_mode TEXT,
                roi_cache_policy TEXT,
                source_roi_cache_used INTEGER,
                source_roi_cache_backend TEXT,
                source_roi_live_acceleration_effective TEXT,
                source_roi_live_gpu_chunk_frames INTEGER,
                input_mode_requested TEXT,
                input_mode_effective TEXT,
                total_rois INTEGER,
                successful_detections INTEGER,
                failed_detections INTEGER,
                success_rate_percent REAL,
                frames_with_keypoints INTEGER,
                mean_confidence REAL,
                duration_seconds REAL,
                inference_duration_seconds REAL,
                keypoints_per_second REAL,
                inference_average_fps REAL,
                batch_size INTEGER,
                imgsz TEXT,
                conf_threshold REAL,
                iou_threshold REAL,
                summary_statistics_json TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, keypoint_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "keypoint_performance",
            {
                "keypoint_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "keypoint_method": "TEXT",
                "model_run_id": "TEXT",
                "model_set_id": "TEXT",
                "model_path": "TEXT",
                "model_name": "TEXT",
                "source_crop_run": "TEXT",
                "source_detect_run": "TEXT",
                "source_refined_run": "TEXT",
                "source_crop_storage_mode": "TEXT",
                "source_crop_signature": "TEXT",
                "source_crop_revision": "INTEGER",
                "source_roi_image_representation": "TEXT",
                "source_roi_pixel_contract_name": "TEXT",
                "source_roi_pixel_contract_json": "TEXT",
                "source_roi_read_mode": "TEXT",
                "roi_cache_policy": "TEXT",
                "source_roi_cache_used": "INTEGER",
                "source_roi_cache_backend": "TEXT",
                "source_roi_live_acceleration_effective": "TEXT",
                "source_roi_live_gpu_chunk_frames": "INTEGER",
                "input_mode_requested": "TEXT",
                "input_mode_effective": "TEXT",
                "total_rois": "INTEGER",
                "successful_detections": "INTEGER",
                "failed_detections": "INTEGER",
                "success_rate_percent": "REAL",
                "frames_with_keypoints": "INTEGER",
                "mean_confidence": "REAL",
                "duration_seconds": "REAL",
                "inference_duration_seconds": "REAL",
                "keypoints_per_second": "REAL",
                "inference_average_fps": "REAL",
                "batch_size": "INTEGER",
                "imgsz": "TEXT",
                "conf_threshold": "REAL",
                "iou_threshold": "REAL",
                "summary_statistics_json": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_perf_recording ON keypoint_performance(recording_id, keypoint_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_perf_method ON keypoint_performance(keypoint_method, model_name);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_perf_runtime ON keypoint_performance(keypoints_per_second, duration_seconds);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_perf_source ON keypoint_performance(source_crop_run, source_detect_run, source_refined_run);"
        )

        cur.execute("DROP VIEW IF EXISTS keypoint_performance_latest;")
        cur.execute(
            """
            CREATE VIEW keypoint_performance_latest AS
            WITH ranked AS (
                SELECT
                    kp.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY kp.dataset_id
                        ORDER BY
                            COALESCE(kp.keypoint_created_utc, kp.updated_utc) DESC,
                            kp.keypoint_run DESC
                    ) AS _rn
                FROM keypoint_performance kp
            )
            SELECT
                dataset_id,
                keypoint_run,
                keypoint_created_utc,
                recording_id,
                zarr_use,
                keypoint_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                source_crop_run,
                source_detect_run,
                source_refined_run,
                source_crop_storage_mode,
                source_crop_signature,
                source_crop_revision,
                source_roi_image_representation,
                source_roi_pixel_contract_name,
                source_roi_pixel_contract_json,
                source_roi_read_mode,
                roi_cache_policy,
                source_roi_cache_used,
                source_roi_cache_backend,
                source_roi_live_acceleration_effective,
                source_roi_live_gpu_chunk_frames,
                input_mode_requested,
                input_mode_effective,
                total_rois,
                successful_detections,
                failed_detections,
                success_rate_percent,
                frames_with_keypoints,
                mean_confidence,
                duration_seconds,
                inference_duration_seconds,
                keypoints_per_second,
                inference_average_fps,
                batch_size,
                imgsz,
                conf_threshold,
                iou_threshold,
                summary_statistics_json,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_keypoint_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_keypoint_performance_latest AS
            WITH ranked AS (
                SELECT
                    kpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY kpl.recording_id
                        ORDER BY
                            COALESCE(kpl.keypoint_created_utc, kpl.updated_utc) DESC,
                            kpl.keypoint_run DESC
                    ) AS _rn
                FROM keypoint_performance_latest kpl
                LEFT JOIN datasets d ON d.dataset_id = kpl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = kpl.dataset_id
                WHERE kpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                keypoint_run,
                keypoint_created_utc,
                zarr_use,
                keypoint_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                source_crop_run,
                source_detect_run,
                source_refined_run,
                source_crop_storage_mode,
                source_crop_signature,
                source_crop_revision,
                source_roi_image_representation,
                source_roi_pixel_contract_name,
                source_roi_pixel_contract_json,
                source_roi_read_mode,
                roi_cache_policy,
                source_roi_cache_used,
                source_roi_cache_backend,
                source_roi_live_acceleration_effective,
                source_roi_live_gpu_chunk_frames,
                input_mode_requested,
                input_mode_effective,
                total_rois,
                successful_detections,
                failed_detections,
                success_rate_percent,
                frames_with_keypoints,
                mean_confidence,
                duration_seconds,
                inference_duration_seconds,
                keypoints_per_second,
                inference_average_fps,
                batch_size,
                imgsz,
                conf_threshold,
                iou_threshold,
                summary_statistics_json,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_019_recording_step_status_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_step_status (
                dataset_id TEXT NOT NULL,
                recording_id TEXT,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('ok', 'missing', 'absent', 'na', 'error')),
                run_name TEXT,
                method TEXT,
                coverage_pct REAL,
                review_status_json TEXT,
                details_json TEXT,
                source TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT NOT NULL,
                PRIMARY KEY (dataset_id, step_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "recording_step_status",
            {
                "recording_id": "TEXT",
                "run_name": "TEXT",
                "method": "TEXT",
                "coverage_pct": "REAL",
                "review_status_json": "TEXT",
                "details_json": "TEXT",
                "source": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_step_status_history (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                recording_id TEXT,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('ok', 'missing', 'absent', 'na', 'error')),
                run_name TEXT,
                method TEXT,
                coverage_pct REAL,
                review_status_json TEXT,
                details_json TEXT,
                source TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT NOT NULL,
                recorded_utc TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "recording_step_status_history",
            {
                "recording_id": "TEXT",
                "run_name": "TEXT",
                "method": "TEXT",
                "coverage_pct": "REAL",
                "review_status_json": "TEXT",
                "details_json": "TEXT",
                "source": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "recorded_utc": "TEXT",
            },
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_recording_step
            ON recording_step_status(recording_id, step_name);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_dataset_step
            ON recording_step_status(dataset_id, step_name);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_status
            ON recording_step_status(status);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_history_recording_step
            ON recording_step_status_history(recording_id, step_name, recorded_utc DESC);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_history_dataset_step
            ON recording_step_status_history(dataset_id, step_name, recorded_utc DESC);
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_step_status_latest;")
        cur.execute(
            """
            CREATE VIEW recording_step_status_latest AS
            SELECT
                COALESCE(NULLIF(trim(rss.recording_id), ''), dcc.recording_id) AS recording_id,
                rss.dataset_id,
                dcc.session_uuid AS session_uuid,
                dcc.zarr_path AS zarr_path,
                dcc.zarr_use AS zarr_use,
                dcc.artifact_kind AS artifact_kind,
                dcc.dataset_status AS dataset_status,
                dcc.rig_id AS rig_id,
                dcc.arena_id AS arena_id,
                dcc.camera_id AS camera_id,
                dcc.canvas_name AS canvas_name,
                dcc.dish_design AS dish_design,
                dcc.protocol_name AS protocol_name,
                dcc.cross_id AS cross_id,
                dcc.genotype AS genotype,
                dcc.dpf_at_acquisition AS dpf_at_acquisition,
                rss.step_name,
                rss.status,
                rss.run_name,
                rss.method,
                rss.coverage_pct,
                rss.review_status_json,
                rss.details_json,
                rss.source,
                rss.zarr_mtime_ns,
                rss.updated_utc
            FROM recording_step_status rss
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = rss.dataset_id;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_step_overview;")
        cur.execute(
            """
            CREATE VIEW recording_step_overview AS
            WITH base AS (
                SELECT
                    recording_id,
                    dataset_id,
                    lower(step_name) AS step_name,
                    status,
                    updated_utc
                FROM recording_step_status_latest
                WHERE recording_id IS NOT NULL AND trim(recording_id) <> ''
            ),
            dataset_counts AS (
                SELECT
                    recording_id,
                    COUNT(DISTINCT dataset_id) AS dataset_count,
                    COUNT(*) AS step_rows_total,
                    MAX(updated_utc) AS latest_step_update_utc
                FROM base
                GROUP BY recording_id
            ),
            status_counts AS (
                SELECT
                    recording_id,
                    SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_rows,
                    SUM(CASE WHEN status = 'missing' THEN 1 ELSE 0 END) AS missing_rows,
                    SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END) AS absent_rows,
                    SUM(CASE WHEN status = 'na' THEN 1 ELSE 0 END) AS na_rows,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_rows,
                    GROUP_CONCAT(DISTINCT CASE WHEN status IN ('missing', 'absent', 'error') THEN step_name END)
                        AS blocking_steps_csv,
                    GROUP_CONCAT(DISTINCT CASE WHEN status = 'ok' THEN step_name END)
                        AS ok_steps_csv
                FROM base
                GROUP BY recording_id
            ),
            per_step AS (
                SELECT
                    recording_id,
                    SUM(CASE WHEN step_name = 'raw' AND status = 'ok' THEN 1 ELSE 0 END) AS raw_ok_count,
                    SUM(CASE WHEN step_name = 'raw' AND status != 'ok' THEN 1 ELSE 0 END) AS raw_non_ok_count,
                    SUM(CASE WHEN step_name = 'background' AND status = 'ok' THEN 1 ELSE 0 END) AS background_ok_count,
                    SUM(CASE WHEN step_name = 'background' AND status != 'ok' THEN 1 ELSE 0 END) AS background_non_ok_count,
                    SUM(CASE WHEN step_name = 'detect' AND status = 'ok' THEN 1 ELSE 0 END) AS detect_ok_count,
                    SUM(CASE WHEN step_name = 'detect' AND status != 'ok' THEN 1 ELSE 0 END) AS detect_non_ok_count,
                    SUM(CASE WHEN step_name = 'refined_detect' AND status = 'ok' THEN 1 ELSE 0 END) AS refined_detect_ok_count,
                    SUM(CASE WHEN step_name = 'refined_detect' AND status != 'ok' THEN 1 ELSE 0 END) AS refined_detect_non_ok_count,
                    SUM(CASE WHEN step_name = 'crop' AND status = 'ok' THEN 1 ELSE 0 END) AS crop_ok_count,
                    SUM(CASE WHEN step_name = 'crop' AND status != 'ok' THEN 1 ELSE 0 END) AS crop_non_ok_count,
                    SUM(CASE WHEN step_name = 'keypoints' AND status = 'ok' THEN 1 ELSE 0 END) AS keypoints_ok_count,
                    SUM(CASE WHEN step_name = 'keypoints' AND status != 'ok' THEN 1 ELSE 0 END) AS keypoints_non_ok_count,
                    SUM(CASE WHEN step_name = 'refined_keypoints' AND status = 'ok' THEN 1 ELSE 0 END) AS refined_keypoints_ok_count,
                    SUM(CASE WHEN step_name = 'refined_keypoints' AND status != 'ok' THEN 1 ELSE 0 END) AS refined_keypoints_non_ok_count,
                    SUM(CASE WHEN step_name = 'eye_masks' AND status = 'ok' THEN 1 ELSE 0 END) AS eye_masks_ok_count,
                    SUM(CASE WHEN step_name = 'eye_masks' AND status != 'ok' THEN 1 ELSE 0 END) AS eye_masks_non_ok_count,
                    SUM(CASE WHEN step_name = 'refined_eye_masks' AND status = 'ok' THEN 1 ELSE 0 END) AS refined_eye_masks_ok_count,
                    SUM(CASE WHEN step_name = 'refined_eye_masks' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS refined_eye_masks_non_ok_count,
                    SUM(CASE WHEN step_name = 'arena_assignment' AND status = 'ok' THEN 1 ELSE 0 END) AS arena_assignment_ok_count,
                    SUM(CASE WHEN step_name = 'arena_assignment' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS arena_assignment_non_ok_count,
                    SUM(CASE WHEN step_name = 'tracks' AND status = 'ok' THEN 1 ELSE 0 END) AS tracks_ok_count,
                    SUM(CASE WHEN step_name = 'tracks' AND status != 'ok' THEN 1 ELSE 0 END) AS tracks_non_ok_count,
                    SUM(CASE WHEN step_name = 'stimulus' AND status = 'ok' THEN 1 ELSE 0 END) AS stimulus_ok_count,
                    SUM(CASE WHEN step_name = 'stimulus' AND status != 'ok' THEN 1 ELSE 0 END) AS stimulus_non_ok_count,
                    SUM(CASE WHEN step_name = 'calibration' AND status = 'ok' THEN 1 ELSE 0 END) AS calibration_ok_count,
                    SUM(CASE WHEN step_name = 'calibration' AND status != 'ok' THEN 1 ELSE 0 END) AS calibration_non_ok_count,
                    SUM(CASE WHEN step_name = 'dish_mask' AND status = 'ok' THEN 1 ELSE 0 END) AS dish_mask_ok_count,
                    SUM(CASE WHEN step_name = 'dish_mask' AND status != 'ok' THEN 1 ELSE 0 END) AS dish_mask_non_ok_count,
                    SUM(CASE WHEN step_name = 'detection_tuning' AND status = 'ok' THEN 1 ELSE 0 END)
                        AS detection_tuning_ok_count,
                    SUM(CASE WHEN step_name = 'detection_tuning' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS detection_tuning_non_ok_count,
                    SUM(CASE WHEN step_name = 'keypoint_tuning' AND status = 'ok' THEN 1 ELSE 0 END)
                        AS keypoint_tuning_ok_count,
                    SUM(CASE WHEN step_name = 'keypoint_tuning' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS keypoint_tuning_non_ok_count,
                    SUM(CASE WHEN step_name = 'eye_mask_tuning' AND status = 'ok' THEN 1 ELSE 0 END)
                        AS eye_mask_tuning_ok_count,
                    SUM(CASE WHEN step_name = 'eye_mask_tuning' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS eye_mask_tuning_non_ok_count,
                    SUM(CASE WHEN step_name = 'subdish_mask_tuning' AND status = 'ok' THEN 1 ELSE 0 END)
                        AS subdish_mask_tuning_ok_count,
                    SUM(CASE WHEN step_name = 'subdish_mask_tuning' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS subdish_mask_tuning_non_ok_count
                FROM base
                GROUP BY recording_id
            )
            SELECT
                dc.recording_id,
                dc.dataset_count,
                dc.step_rows_total,
                dc.latest_step_update_utc,
                sc.ok_rows,
                sc.missing_rows,
                sc.absent_rows,
                sc.na_rows,
                sc.error_rows,
                sc.blocking_steps_csv,
                sc.ok_steps_csv,
                ps.raw_ok_count,
                ps.raw_non_ok_count,
                ps.background_ok_count,
                ps.background_non_ok_count,
                ps.detect_ok_count,
                ps.detect_non_ok_count,
                ps.refined_detect_ok_count,
                ps.refined_detect_non_ok_count,
                ps.crop_ok_count,
                ps.crop_non_ok_count,
                ps.keypoints_ok_count,
                ps.keypoints_non_ok_count,
                ps.refined_keypoints_ok_count,
                ps.refined_keypoints_non_ok_count,
                ps.eye_masks_ok_count,
                ps.eye_masks_non_ok_count,
                ps.refined_eye_masks_ok_count,
                ps.refined_eye_masks_non_ok_count,
                ps.arena_assignment_ok_count,
                ps.arena_assignment_non_ok_count,
                ps.tracks_ok_count,
                ps.tracks_non_ok_count,
                ps.stimulus_ok_count,
                ps.stimulus_non_ok_count,
                ps.calibration_ok_count,
                ps.calibration_non_ok_count,
                ps.dish_mask_ok_count,
                ps.dish_mask_non_ok_count,
                ps.detection_tuning_ok_count,
                ps.detection_tuning_non_ok_count,
                ps.keypoint_tuning_ok_count,
                ps.keypoint_tuning_non_ok_count,
                ps.eye_mask_tuning_ok_count,
                ps.eye_mask_tuning_non_ok_count,
                ps.subdish_mask_tuning_ok_count,
                ps.subdish_mask_tuning_non_ok_count
            FROM dataset_counts dc
            LEFT JOIN status_counts sc ON sc.recording_id = dc.recording_id
            LEFT JOIN per_step ps ON ps.recording_id = dc.recording_id;
            """
        )

    def _migration_020_recording_step_status_wide_view(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS recording_step_status_wide;")
        cur.execute(
            f"""
            CREATE VIEW recording_step_status_wide AS
            WITH step_rows AS (
                SELECT
                    dataset_id,
                    COALESCE(NULLIF(trim(recording_id), ''), '') AS recording_id,
                    COALESCE(camera_id, 'unknown') AS camera_id,
                    zarr_path,
                    zarr_use,
                    dataset_status,
                    lower(step_name) AS step_name,
                    lower(status) AS status,
                    run_name,
                    method,
                    coverage_pct,
                    review_status_json,
                    details_json
                FROM recording_step_status_latest
            ),
            pivot AS (
                SELECT
                    dataset_id,
                    MAX(NULLIF(recording_id, '')) AS recording_id,
                    MAX(camera_id) AS camera_id,
                    MAX(zarr_path) AS zarr_path,
                    MAX(zarr_use) AS zarr_use,
                    MAX(dataset_status) AS dataset_status,
                    {_recording_step_status_pivot_columns()},
                    MAX(CASE WHEN step_name = 'raw' THEN details_json END) AS raw_details_json,
                    MAX(CASE WHEN step_name = 'background' THEN details_json END) AS background_details_json,
                    MAX(CASE WHEN step_name = 'detect' THEN method END) AS detect_method,
                    MAX(CASE WHEN step_name = 'detect' THEN coverage_pct END) AS detect_coverage_pct,
                    MAX(CASE WHEN step_name = 'detect' THEN details_json END) AS detect_details_json,
                    MAX(CASE WHEN step_name = 'detect_quality' THEN run_name END) AS detect_quality_run_name,
                    MAX(CASE WHEN step_name = 'detect_quality' THEN details_json END) AS detect_quality_details_json,
                    MAX(CASE WHEN step_name = 'refined_detect' THEN method END) AS refined_detect_method,
                    MAX(CASE WHEN step_name = 'refined_detect' THEN coverage_pct END) AS refined_detect_coverage_pct,
                    MAX(CASE WHEN step_name = 'refined_detect' THEN review_status_json END) AS refined_detect_review_json,
                    MAX(CASE WHEN step_name = 'refined_detect' THEN details_json END) AS refined_detect_details_json,
                    MAX(CASE WHEN step_name = 'crop' THEN review_status_json END) AS crop_review_json,
                    MAX(CASE WHEN step_name = 'crop' THEN details_json END) AS crop_details_json,
                    MAX(CASE WHEN step_name = 'keypoints' THEN details_json END) AS keypoints_details_json,
                    MAX(CASE WHEN step_name = 'refined_keypoints' THEN coverage_pct END) AS refined_keypoints_coverage_pct,
                    MAX(CASE WHEN step_name = 'refined_keypoints' THEN review_status_json END) AS refined_keypoints_review_json,
                    MAX(CASE WHEN step_name = 'refined_keypoints' THEN details_json END) AS refined_keypoints_details_json,
                    MAX(CASE WHEN step_name = 'eye_masks' THEN review_status_json END) AS eye_masks_review_json,
                    MAX(CASE WHEN step_name = 'eye_masks' THEN details_json END) AS eye_masks_details_json,
                    MAX(CASE WHEN step_name = 'refined_eye_masks' THEN review_status_json END) AS refined_eye_masks_review_json,
                    MAX(CASE WHEN step_name = 'refined_eye_masks' THEN details_json END) AS refined_eye_masks_details_json,
                    MAX(CASE WHEN step_name = 'subject_masks' THEN review_status_json END) AS subject_masks_review_json,
                    MAX(CASE WHEN step_name = 'subject_masks' THEN details_json END) AS subject_masks_details_json,
                    MAX(CASE WHEN step_name = 'refined_subject_masks' THEN review_status_json END) AS refined_subject_masks_review_json,
                    MAX(CASE WHEN step_name = 'refined_subject_masks' THEN details_json END) AS refined_subject_masks_details_json,
                    MAX(CASE WHEN step_name = 'tracks' THEN details_json END) AS tracks_details_json,
                    MAX(CASE WHEN step_name = 'track_kinematics' THEN details_json END) AS track_kinematics_details_json,
                    MAX(CASE WHEN step_name = 'swim_bouts' THEN details_json END) AS swim_bouts_details_json,
                    MAX(CASE WHEN step_name = 'bout_kinematics' THEN details_json END) AS bout_kinematics_details_json,
                    MAX(CASE WHEN step_name = 'eye_angles' THEN details_json END) AS eye_angles_details_json,
                    MAX(CASE WHEN step_name = 'subject_shape' THEN details_json END) AS subject_shape_details_json,
                    MAX(CASE WHEN step_name = 'tail_kinematics' THEN details_json END) AS tail_kinematics_details_json,
                    MAX(CASE WHEN step_name = 'tail_posture_view' THEN details_json END) AS tail_posture_view_details_json,
                    MAX(CASE WHEN step_name = 'bout_classification' THEN details_json END) AS bout_classification_details_json,
                    MAX(CASE WHEN step_name = 'stimulus_response' THEN details_json END) AS stimulus_response_details_json,
                    MAX(CASE WHEN step_name = 'stimulus' THEN details_json END) AS stimulus_details_json
                FROM step_rows
                GROUP BY dataset_id
            ),
            derived AS (
                SELECT
                    p.*,
                    COALESCE(
                        json_extract(p.raw_details_json, '$.pipeline_type'),
                        json_extract(p.background_details_json, '$.pipeline_type'),
                        json_extract(p.detect_details_json, '$.pipeline_type'),
                        json_extract(p.refined_detect_details_json, '$.pipeline_type'),
                        json_extract(p.crop_details_json, '$.pipeline_type'),
                        json_extract(p.keypoints_details_json, '$.pipeline_type'),
                        json_extract(p.refined_keypoints_details_json, '$.pipeline_type'),
                        json_extract(p.eye_masks_details_json, '$.pipeline_type'),
                        json_extract(p.refined_eye_masks_details_json, '$.pipeline_type'),
                        json_extract(p.subject_masks_details_json, '$.pipeline_type'),
                        json_extract(p.refined_subject_masks_details_json, '$.pipeline_type'),
                        json_extract(p.track_kinematics_details_json, '$.pipeline_type'),
                        json_extract(p.swim_bouts_details_json, '$.pipeline_type'),
                        json_extract(p.bout_kinematics_details_json, '$.pipeline_type'),
                        json_extract(p.eye_angles_details_json, '$.pipeline_type'),
                        json_extract(p.subject_shape_details_json, '$.pipeline_type'),
                        json_extract(p.tail_kinematics_details_json, '$.pipeline_type'),
                        json_extract(p.tail_posture_view_details_json, '$.pipeline_type'),
                        json_extract(p.bout_classification_details_json, '$.pipeline_type'),
                        json_extract(p.stimulus_response_details_json, '$.pipeline_type')
                    ) AS pipeline_type,
                    COALESCE(
                        json_extract(p.raw_details_json, '$.zarr_purpose'),
                        json_extract(p.background_details_json, '$.zarr_purpose'),
                        json_extract(p.detect_details_json, '$.zarr_purpose'),
                        json_extract(p.refined_detect_details_json, '$.zarr_purpose'),
                        json_extract(p.crop_details_json, '$.zarr_purpose'),
                        json_extract(p.keypoints_details_json, '$.zarr_purpose'),
                        json_extract(p.refined_keypoints_details_json, '$.zarr_purpose'),
                        json_extract(p.eye_masks_details_json, '$.zarr_purpose'),
                        json_extract(p.refined_eye_masks_details_json, '$.zarr_purpose'),
                        json_extract(p.subject_masks_details_json, '$.zarr_purpose'),
                        json_extract(p.refined_subject_masks_details_json, '$.zarr_purpose'),
                        json_extract(p.track_kinematics_details_json, '$.zarr_purpose'),
                        json_extract(p.swim_bouts_details_json, '$.zarr_purpose'),
                        json_extract(p.bout_kinematics_details_json, '$.zarr_purpose'),
                        json_extract(p.eye_angles_details_json, '$.zarr_purpose'),
                        json_extract(p.subject_shape_details_json, '$.zarr_purpose'),
                        json_extract(p.tail_kinematics_details_json, '$.zarr_purpose'),
                        json_extract(p.tail_posture_view_details_json, '$.zarr_purpose'),
                        json_extract(p.bout_classification_details_json, '$.zarr_purpose'),
                        json_extract(p.stimulus_response_details_json, '$.zarr_purpose')
                    ) AS zarr_purpose,
                    COALESCE(
                        json_extract(p.raw_details_json, '$.has_raw_video_attr'),
                        json_extract(p.background_details_json, '$.has_raw_video_attr'),
                        json_extract(p.detect_details_json, '$.has_raw_video_attr'),
                        json_extract(p.refined_detect_details_json, '$.has_raw_video_attr'),
                        json_extract(p.crop_details_json, '$.has_raw_video_attr'),
                        json_extract(p.keypoints_details_json, '$.has_raw_video_attr'),
                        json_extract(p.refined_keypoints_details_json, '$.has_raw_video_attr'),
                        json_extract(p.eye_masks_details_json, '$.has_raw_video_attr'),
                        json_extract(p.refined_eye_masks_details_json, '$.has_raw_video_attr'),
                        json_extract(p.subject_masks_details_json, '$.has_raw_video_attr'),
                        json_extract(p.refined_subject_masks_details_json, '$.has_raw_video_attr'),
                        json_extract(p.track_kinematics_details_json, '$.has_raw_video_attr'),
                        json_extract(p.swim_bouts_details_json, '$.has_raw_video_attr'),
                        json_extract(p.bout_kinematics_details_json, '$.has_raw_video_attr'),
                        json_extract(p.eye_angles_details_json, '$.has_raw_video_attr'),
                        json_extract(p.subject_shape_details_json, '$.has_raw_video_attr'),
                        json_extract(p.tail_kinematics_details_json, '$.has_raw_video_attr'),
                        json_extract(p.tail_posture_view_details_json, '$.has_raw_video_attr'),
                        json_extract(p.bout_classification_details_json, '$.has_raw_video_attr'),
                        json_extract(p.stimulus_response_details_json, '$.has_raw_video_attr')
                    ) AS has_raw_video_attr,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.raw_details_json, '$.raw_present'),
                            CASE WHEN p.raw_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS raw_present,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.raw_details_json, '$.full_present'),
                            CASE WHEN p.raw_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS full_present,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.raw_details_json, '$.ds_present'),
                            CASE WHEN p.raw_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS ds_present,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.background_details_json, '$.full_present'),
                            CASE WHEN p.background_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS background_full_present,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.background_details_json, '$.ds_present'),
                            CASE WHEN p.background_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS background_ds_present,
                    CASE WHEN p.detect_status = 'ok' THEN 1 ELSE 0 END AS detect_present,
                    CASE
                        WHEN p.refined_detect_status = 'ok' AND p.refined_detect_coverage_pct IS NULL THEN 100.0
                        ELSE p.refined_detect_coverage_pct
                    END AS refined_detect_coverage_effective,
                    CASE WHEN p.keypoints_status = 'ok' THEN 1 ELSE 0 END AS keypoints_present,
                    CASE WHEN p.refined_keypoints_status = 'ok' THEN 1 ELSE 0 END AS refined_keypoints_present,
                    CASE
                        WHEN p.refined_keypoints_status = 'ok' AND p.refined_keypoints_coverage_pct IS NULL THEN 100.0
                        ELSE p.refined_keypoints_coverage_pct
                    END AS refined_keypoints_success_effective,
                    CASE WHEN p.eye_masks_status = 'ok' THEN 1 ELSE 0 END AS eye_masks_present,
                    CASE WHEN p.refined_eye_masks_status = 'ok' THEN 1 ELSE 0 END AS refined_eye_masks_present,
                    CASE WHEN p.subject_masks_status = 'ok' THEN 1 ELSE 0 END AS subject_masks_present,
                    CASE WHEN p.refined_subject_masks_status = 'ok' THEN 1 ELSE 0 END AS refined_subject_masks_present,
                    CASE WHEN p.arena_assignment_status = 'ok' THEN 1 ELSE 0 END AS arena_assignment_present,
                    CASE WHEN p.tracks_status = 'ok' THEN 1 ELSE 0 END AS track_present,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.n_unassigned_rows'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.n_unassigned_rows')
                        ) AS INTEGER
                    ) AS track_unassigned_rows,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.unassigned_row_rate_percent'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.unassigned_row_rate_percent')
                        ) AS REAL
                    ) AS track_unassigned_rate_percent,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.tracking_warn_threshold_rows'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.tracking_warn_threshold_rows'),
                            1
                        ) AS INTEGER
                    ) AS track_warn_threshold_rows,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.tracking_warn_threshold_percent'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.tracking_warn_threshold_percent'),
                            0.0
                        ) AS REAL
                    ) AS track_warn_threshold_percent,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.tracking_block_threshold_rows'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.tracking_block_threshold_rows'),
                            10
                        ) AS INTEGER
                    ) AS track_block_threshold_rows,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.tracking_block_threshold_percent'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.tracking_block_threshold_percent'),
                            1.0
                        ) AS REAL
                    ) AS track_block_threshold_percent,
                    CASE WHEN p.calibration_status = 'ok' THEN 1 ELSE 0 END AS calibration_present,
                    CAST(
                        COALESCE(
                            json_extract(p.stimulus_details_json, '$.stimulus_runs'),
                            CASE WHEN p.stimulus_status = 'ok' THEN 1 ELSE 0 END,
                            0
                        ) AS INTEGER
                    ) AS stimulus_runs,
                    COALESCE(
                        NULLIF(TRIM(CAST(json_extract(p.crop_details_json, '$.run_state') AS TEXT)), ''),
                        NULL
                    ) AS crop_run_state,
                    CAST(
                        COALESCE(
                            json_extract(p.refined_keypoints_details_json, '$.usable_keypoints_pct'),
                            json_extract(p.refined_keypoints_details_json, '$.usable_percent'),
                            json_extract(p.refined_keypoints_details_json, '$.train_usable_pct')
                        ) AS REAL
                    ) AS refined_keypoints_train_usable_pct,
                    COALESCE(
                        CAST(json_extract(p.detect_quality_details_json, '$.quality_grade') AS TEXT),
                        CAST(json_extract(p.detect_details_json, '$.detect_quality_grade') AS TEXT),
                        CAST(json_extract(p.detect_details_json, '$.grade') AS TEXT)
                    ) AS detect_quality_grade,
                    CAST(
                        COALESCE(
                            json_extract(p.detect_quality_details_json, '$.quality_score'),
                            json_extract(p.detect_details_json, '$.detect_quality_score'),
                            json_extract(p.detect_details_json, '$.score')
                        ) AS REAL
                    ) AS detect_quality_score,
                    CAST(
                        COALESCE(
                            json_extract(p.detect_quality_details_json, '$.clean_percentage'),
                            json_extract(p.detect_details_json, '$.detect_quality_clean_percent'),
                            json_extract(p.detect_details_json, '$.clean_percent'),
                            json_extract(p.detect_details_json, '$.clean_percentage')
                        ) AS REAL
                    ) AS detect_quality_clean_percent,
                    CAST(
                        COALESCE(
                            json_extract(p.detect_details_json, '$.detect_quality_artifacts'),
                            json_extract(p.detect_details_json, '$.artifact_count')
                        ) AS INTEGER
                    ) AS detect_quality_artifacts,
                    COALESCE(p.refined_eye_masks_review_json, p.eye_masks_review_json) AS eye_mask_review_json
                FROM pivot p
            ),
            render AS (
                SELECT
                    d.*,
                    CASE
                        WHEN lower(COALESCE(CAST(d.zarr_purpose AS TEXT), '')) = 'production' THEN 1
                        WHEN lower(COALESCE(CAST(d.pipeline_type AS TEXT), '')) = 'yolo_inference' THEN 1
                        WHEN d.has_raw_video_attr = 0 AND NOT (d.full_present = 1 OR d.ds_present = 1) THEN 1
                        ELSE 0
                    END AS is_production,
                    ({_recording_tuning_ok_count_sql("d")}) AS tuning_ok_count,
                    {len(recording_tuning_stage_ids())} AS tuning_total,
                    NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.state') AS TEXT)), '') AS detect_review_state,
                    NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.method') AS TEXT)), '') AS detect_review_method,
                    NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.intended_use') AS TEXT)), '') AS detect_review_use,
                    NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.resolved_group') AS TEXT)), '') AS detect_review_group,
                    COALESCE(
                        NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.resolved_group') AS TEXT)), ''),
                        NULLIF(TRIM(CAST(json_extract(d.refined_detect_details_json, '$.resolved_group') AS TEXT)), '')
                    ) AS detect_group,
                    NULLIF(TRIM(CAST(json_extract(d.crop_review_json, '$.state') AS TEXT)), '') AS crop_review_state,
                    NULLIF(TRIM(CAST(json_extract(d.crop_review_json, '$.method') AS TEXT)), '') AS crop_review_method,
                    NULLIF(TRIM(CAST(json_extract(d.crop_review_json, '$.intended_use') AS TEXT)), '') AS crop_review_use,
                    NULLIF(TRIM(CAST(json_extract(d.crop_review_json, '$.resolved_group') AS TEXT)), '') AS crop_review_group,
                    NULLIF(TRIM(CAST(json_extract(d.refined_keypoints_review_json, '$.state') AS TEXT)), '') AS keypoint_review_state,
                    NULLIF(TRIM(CAST(json_extract(d.refined_keypoints_review_json, '$.method') AS TEXT)), '') AS keypoint_review_method,
                    NULLIF(TRIM(CAST(json_extract(d.refined_keypoints_review_json, '$.intended_use') AS TEXT)), '') AS keypoint_review_use,
                    NULLIF(TRIM(CAST(json_extract(d.refined_keypoints_review_json, '$.resolved_group') AS TEXT)), '') AS keypoint_review_group,
                    NULLIF(TRIM(CAST(json_extract(d.eye_mask_review_json, '$.state') AS TEXT)), '') AS eye_review_state,
                    NULLIF(TRIM(CAST(json_extract(d.eye_mask_review_json, '$.method') AS TEXT)), '') AS eye_review_method,
                    NULLIF(TRIM(CAST(json_extract(d.eye_mask_review_json, '$.intended_use') AS TEXT)), '') AS eye_review_use,
                    NULLIF(TRIM(CAST(json_extract(d.eye_mask_review_json, '$.resolved_group') AS TEXT)), '') AS eye_review_group,
                    CASE
                        WHEN d.detect_quality_grade IS NOT NULL AND d.detect_quality_score IS NOT NULL
                            THEN d.detect_quality_grade || ' ' || printf('%.1f', d.detect_quality_score)
                        WHEN d.detect_quality_grade IS NOT NULL
                            THEN d.detect_quality_grade
                        WHEN d.detect_quality_score IS NOT NULL
                            THEN printf('%.1f', d.detect_quality_score)
                        ELSE ''
                    END AS detect_quality_head,
                    COALESCE(
                        NULLIF(
                            lower(trim(CAST(json_extract(d.tracks_details_json, '$.tracking_qc_state') AS TEXT))),
                            ''
                        ),
                        CASE
                            WHEN COALESCE(d.track_unassigned_rows, 0) <= 0 THEN 'ok'
                            WHEN d.track_unassigned_rows >= COALESCE(d.track_warn_threshold_rows, 1)
                                OR COALESCE(d.track_unassigned_rate_percent, 0.0) > COALESCE(d.track_warn_threshold_percent, 0.0)
                                THEN 'warn'
                            ELSE 'ok'
                        END
                    ) AS track_qc_state
                FROM derived d
            )
            SELECT
                COALESCE(r.recording_id, r.dataset_id) AS "Recording",
                COALESCE(r.camera_id, 'unknown') AS "Camera",
                CASE
                    WHEN lower(COALESCE(CAST(r.dataset_status AS TEXT), '')) = 'missing' THEN 'MISS'
                    ELSE 'OK'
                END AS "Zarr",
                COALESCE(NULLIF(CAST(r.zarr_use AS TEXT), ''), '—') AS "Use",
                COALESCE(NULLIF(CAST(r.zarr_purpose AS TEXT), ''), '—') AS "Purpose",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.raw_present = 1 AND (r.full_present = 1 OR r.ds_present = 1) THEN 'OK'
                    ELSE 'MISS'
                END AS "Import",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.background_full_present = 1 THEN 'OK'
                    ELSE 'MISS'
                END AS "BG Full",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.background_ds_present = 1 THEN 'OK'
                    ELSE 'MISS'
                END AS "BG DS",
                CASE
                    WHEN r.detect_present != 1 THEN 'MISS'
                    WHEN r.detect_coverage_pct IS NULL AND r.detect_method IS NOT NULL THEN 'OK (' || r.detect_method || ')'
                    WHEN r.detect_coverage_pct IS NULL THEN 'OK'
                    WHEN r.detect_method IS NOT NULL THEN
                        'OK ('
                        || CASE
                            WHEN r.detect_coverage_pct >= 99.999 THEN '100%'
                            ELSE printf('%.1f%%', r.detect_coverage_pct)
                        END
                        || ', registry, '
                        || r.detect_method
                        || ')'
                    ELSE
                        'OK ('
                        || CASE
                            WHEN r.detect_coverage_pct >= 99.999 THEN '100%'
                            ELSE printf('%.1f%%', r.detect_coverage_pct)
                        END
                        || ', registry)'
                END AS "Detect",
                CASE
                    WHEN r.detect_quality_head = '' AND r.detect_quality_clean_percent IS NULL AND r.detect_quality_artifacts IS NULL
                        THEN 'MISS'
                    ELSE
                        CASE
                            WHEN (
                                r.detect_quality_head
                                || CASE
                                    WHEN r.detect_quality_clean_percent IS NOT NULL THEN
                                        CASE WHEN r.detect_quality_head <> '' THEN ', ' ELSE '' END
                                        || 'clean '
                                        || printf('%.1f%%', r.detect_quality_clean_percent)
                                    ELSE ''
                                END
                                || CASE
                                    WHEN r.detect_quality_artifacts IS NOT NULL THEN
                                        CASE
                                            WHEN r.detect_quality_head <> '' OR r.detect_quality_clean_percent IS NOT NULL THEN ', '
                                            ELSE ''
                                        END
                                        || 'art '
                                        || CAST(r.detect_quality_artifacts AS TEXT)
                                    ELSE ''
                                END
                            ) = '' THEN 'OK'
                            ELSE
                                'OK ('
                                || (
                                    r.detect_quality_head
                                    || CASE
                                        WHEN r.detect_quality_clean_percent IS NOT NULL THEN
                                            CASE WHEN r.detect_quality_head <> '' THEN ', ' ELSE '' END
                                            || 'clean '
                                            || printf('%.1f%%', r.detect_quality_clean_percent)
                                        ELSE ''
                                    END
                                    || CASE
                                        WHEN r.detect_quality_artifacts IS NOT NULL THEN
                                            CASE
                                                WHEN r.detect_quality_head <> '' OR r.detect_quality_clean_percent IS NOT NULL THEN ', '
                                                ELSE ''
                                            END
                                            || 'art '
                                            || CAST(r.detect_quality_artifacts AS TEXT)
                                        ELSE ''
                                    END
                                )
                                || ')'
                        END
                END AS "Detect Quality",
                CASE
                    WHEN r.refined_detect_coverage_effective IS NULL THEN 'MISS'
                    WHEN r.refined_detect_method IS NOT NULL THEN
                        CASE
                            WHEN r.refined_detect_coverage_effective >= 99.999 THEN '100%'
                            ELSE printf('%.1f%%', r.refined_detect_coverage_effective)
                        END
                        || ' ('
                        || r.refined_detect_method
                        || ')'
                    ELSE
                        CASE
                            WHEN r.refined_detect_coverage_effective >= 99.999 THEN '100%'
                            ELSE printf('%.1f%%', r.refined_detect_coverage_effective)
                        END
                END AS "Refine Detect",
                COALESCE(r.detect_group, '—') AS "Detect Group",
                CASE
                    WHEN r.detect_review_state IS NULL
                        AND r.detect_review_method IS NULL
                        AND r.detect_review_use IS NULL
                        AND r.detect_review_group IS NULL
                        THEN '—'
                    WHEN COALESCE(r.detect_review_method, '') <> ''
                        OR COALESCE(r.detect_review_use, '') <> ''
                        OR COALESCE(r.detect_review_group, '') <> ''
                        THEN
                            COALESCE(r.detect_review_state, 'review')
                            || ' ('
                            || CASE WHEN COALESCE(r.detect_review_method, '') <> '' THEN r.detect_review_method ELSE '' END
                            || CASE
                                WHEN COALESCE(r.detect_review_use, '') <> '' THEN
                                    CASE WHEN COALESCE(r.detect_review_method, '') <> '' THEN ', ' ELSE '' END
                                    || r.detect_review_use
                                ELSE ''
                            END
                            || CASE
                                WHEN COALESCE(r.detect_review_group, '') <> '' THEN
                                    CASE
                                        WHEN COALESCE(r.detect_review_method, '') <> '' OR COALESCE(r.detect_review_use, '') <> ''
                                            THEN ', '
                                        ELSE ''
                                    END
                                    || 'group='
                                    || r.detect_review_group
                                ELSE ''
                            END
                            || ')'
                    WHEN r.detect_review_state IS NOT NULL THEN r.detect_review_state
                    ELSE '—'
                END AS "Detect Review",
                CASE
                    WHEN r.crop_status = 'ok'
                        THEN COALESCE(NULLIF(lower(COALESCE(r.crop_run_state, '')), ''), 'OK')
                    WHEN r.crop_status = 'error' THEN 'failed'
                    WHEN r.crop_status = 'na' THEN 'na'
                    ELSE 'MISS'
                END AS "Crop",
                CASE
                    WHEN r.crop_review_state IS NULL
                        AND r.crop_review_method IS NULL
                        AND r.crop_review_use IS NULL
                        AND r.crop_review_group IS NULL
                        THEN '—'
                    WHEN COALESCE(r.crop_review_method, '') <> ''
                        OR COALESCE(r.crop_review_use, '') <> ''
                        OR COALESCE(r.crop_review_group, '') <> ''
                        THEN
                            COALESCE(r.crop_review_state, 'review')
                            || ' ('
                            || CASE WHEN COALESCE(r.crop_review_method, '') <> '' THEN r.crop_review_method ELSE '' END
                            || CASE
                                WHEN COALESCE(r.crop_review_use, '') <> '' THEN
                                    CASE WHEN COALESCE(r.crop_review_method, '') <> '' THEN ', ' ELSE '' END
                                    || r.crop_review_use
                                ELSE ''
                            END
                            || CASE
                                WHEN COALESCE(r.crop_review_group, '') <> '' THEN
                                    CASE
                                        WHEN COALESCE(r.crop_review_method, '') <> '' OR COALESCE(r.crop_review_use, '') <> ''
                                            THEN ', '
                                        ELSE ''
                                    END
                                    || 'group='
                                    || r.crop_review_group
                                ELSE ''
                            END
                            || ')'
                    WHEN r.crop_review_state IS NOT NULL THEN r.crop_review_state
                    ELSE '—'
                END AS "Crop Review",
                CASE WHEN r.keypoints_present = 1 THEN 'OK' ELSE 'MISS' END AS "Keypoints",
                CASE
                    WHEN r.refined_keypoints_success_effective IS NULL
                        AND r.refined_keypoints_train_usable_pct IS NULL
                        THEN 'MISS'
                    ELSE
                        COALESCE(
                            CASE
                                WHEN r.refined_keypoints_success_effective >= 99.999 THEN '100%'
                                WHEN r.refined_keypoints_success_effective IS NOT NULL
                                    THEN printf('%.1f%%', r.refined_keypoints_success_effective)
                                ELSE NULL
                            END,
                            '—'
                        )
                        || CASE
                            WHEN r.refined_keypoints_train_usable_pct IS NOT NULL THEN
                                ' (train '
                                || CASE
                                    WHEN r.refined_keypoints_train_usable_pct >= 99.999 THEN '100%'
                                    ELSE printf('%.1f%%', r.refined_keypoints_train_usable_pct)
                                END
                                || ')'
                            ELSE ''
                        END
                END AS "Refined Keypoints (analysis/train)",
                CASE
                    WHEN r.keypoint_review_state IS NULL
                        AND r.keypoint_review_method IS NULL
                        AND r.keypoint_review_use IS NULL
                        AND r.keypoint_review_group IS NULL
                        THEN '—'
                    WHEN COALESCE(r.keypoint_review_method, '') <> ''
                        OR COALESCE(r.keypoint_review_use, '') <> ''
                        OR COALESCE(r.keypoint_review_group, '') <> ''
                        THEN
                            COALESCE(r.keypoint_review_state, 'review')
                            || ' ('
                            || CASE WHEN COALESCE(r.keypoint_review_method, '') <> '' THEN r.keypoint_review_method ELSE '' END
                            || CASE
                                WHEN COALESCE(r.keypoint_review_use, '') <> '' THEN
                                    CASE WHEN COALESCE(r.keypoint_review_method, '') <> '' THEN ', ' ELSE '' END
                                    || r.keypoint_review_use
                                ELSE ''
                            END
                            || CASE
                                WHEN COALESCE(r.keypoint_review_group, '') <> '' THEN
                                    CASE
                                        WHEN COALESCE(r.keypoint_review_method, '') <> '' OR COALESCE(r.keypoint_review_use, '') <> ''
                                            THEN ', '
                                        ELSE ''
                                    END
                                    || 'group='
                                    || r.keypoint_review_group
                                ELSE ''
                            END
                            || ')'
                    WHEN r.keypoint_review_state IS NOT NULL THEN r.keypoint_review_state
                    ELSE '—'
                END AS "Keypoint Review",
                CASE WHEN r.eye_masks_present = 1 THEN 'OK' ELSE 'MISS' END AS "Eye Masks",
                CASE WHEN r.refined_eye_masks_present = 1 THEN 'OK' ELSE 'MISS' END AS "Refined Eye Masks",
                CASE WHEN r.subject_masks_present = 1 THEN 'OK' ELSE 'MISS' END AS "Subject Masks",
                CASE WHEN r.refined_subject_masks_present = 1 THEN 'OK' ELSE 'MISS' END AS "Refined Subject Masks",
                CASE
                    WHEN r.eye_review_state IS NULL
                        AND r.eye_review_method IS NULL
                        AND r.eye_review_use IS NULL
                        AND r.eye_review_group IS NULL
                        THEN '—'
                    WHEN COALESCE(r.eye_review_method, '') <> ''
                        OR COALESCE(r.eye_review_use, '') <> ''
                        OR COALESCE(r.eye_review_group, '') <> ''
                        THEN
                            COALESCE(r.eye_review_state, 'review')
                            || ' ('
                            || CASE WHEN COALESCE(r.eye_review_method, '') <> '' THEN r.eye_review_method ELSE '' END
                            || CASE
                                WHEN COALESCE(r.eye_review_use, '') <> '' THEN
                                    CASE WHEN COALESCE(r.eye_review_method, '') <> '' THEN ', ' ELSE '' END
                                    || r.eye_review_use
                                ELSE ''
                            END
                            || CASE
                                WHEN COALESCE(r.eye_review_group, '') <> '' THEN
                                    CASE
                                        WHEN COALESCE(r.eye_review_method, '') <> '' OR COALESCE(r.eye_review_use, '') <> ''
                                            THEN ', '
                                        ELSE ''
                                    END
                                    || 'group='
                                    || r.eye_review_group
                                ELSE ''
                            END
                            || ')'
                    WHEN r.eye_review_state IS NOT NULL THEN r.eye_review_state
                    ELSE '—'
                END AS "Eye Mask Review",
                CASE WHEN r.arena_assignment_present = 1 THEN 'OK' ELSE 'MISS' END AS "Arena Assignment",
                CASE
                    WHEN r.track_present != 1 THEN 'MISS'
                    WHEN lower(COALESCE(r.track_qc_state, '')) IN ('warn', 'block') AND r.track_unassigned_rate_percent IS NOT NULL THEN
                        'WARN ('
                        || CAST(r.track_unassigned_rows AS TEXT)
                        || ' unassigned, '
                        || printf('%.1f%%', r.track_unassigned_rate_percent)
                        || ')'
                    WHEN lower(COALESCE(r.track_qc_state, '')) IN ('warn', 'block') AND r.track_unassigned_rows IS NOT NULL THEN
                        'WARN ('
                        || CAST(r.track_unassigned_rows AS TEXT)
                        || ' unassigned)'
                    WHEN lower(COALESCE(r.track_qc_state, '')) IN ('warn', 'block') THEN 'WARN'
                    ELSE 'OK'
                END AS "Track",
                CASE
                    WHEN r.track_kinematics_status = 'ok' THEN 'OK'
                    WHEN r.track_kinematics_status = 'na' THEN 'N/A'
                    WHEN r.track_kinematics_status = 'error' THEN 'ERR'
                    ELSE 'MISS'
                END AS "Track Kinematics",
                CASE
                    WHEN r.swim_bouts_status = 'ok' THEN 'OK'
                    WHEN r.swim_bouts_status = 'na' THEN 'N/A'
                    WHEN r.swim_bouts_status = 'error' THEN 'ERR'
                    ELSE 'MISS'
                END AS "Swim Bouts",
                {_recording_step_status_display_sql("r.bout_kinematics_status", "r.bout_kinematics_details_json")} AS "Bout Kinematics",
                {_recording_step_status_display_sql("r.eye_angles_status", "r.eye_angles_details_json")} AS "Eye Angles",
                {_recording_step_status_display_sql("r.subject_shape_status", "r.subject_shape_details_json")} AS "Subject Shape",
                {_recording_step_status_display_sql("r.tail_kinematics_status", "r.tail_kinematics_details_json")} AS "Tail Kinematics",
                {_recording_step_status_display_sql("r.tail_posture_view_status", "r.tail_posture_view_details_json")} AS "Tail Posture View",
                {_recording_step_status_display_sql("r.bout_classification_status", "r.bout_classification_details_json")} AS "Bout Classification",
                {_recording_step_status_display_sql("r.stimulus_response_status", "r.stimulus_response_details_json")} AS "Stimulus Response",
                CAST(r.stimulus_runs AS TEXT) || ' (' || CASE WHEN r.stimulus_runs > 0 THEN 'OK' ELSE 'MISS' END || ')' AS "Stimulus",
                CASE WHEN r.calibration_present = 1 THEN 'OK' ELSE 'MISS' END AS "Calib",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    ELSE CAST(r.tuning_ok_count AS TEXT) || '/' || CAST(r.tuning_total AS TEXT)
                END AS "Tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.dish_mask_status = 'ok' THEN 'OK'
                    WHEN r.dish_mask_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "dish_mask",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.detection_tuning_status = 'ok' THEN 'OK'
                    WHEN r.detection_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "detection_tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.keypoint_tuning_status = 'ok' THEN 'OK'
                    WHEN r.keypoint_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "keypoint_tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.subject_mask_tuning_status = 'ok' THEN 'OK'
                    WHEN r.subject_mask_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "subject_mask_tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.eye_mask_tuning_status = 'ok' THEN 'OK'
                    WHEN r.eye_mask_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "eye_mask_tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.subdish_mask_tuning_status = 'ok' THEN 'OK'
                    WHEN r.subdish_mask_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "subdish_mask_tuning"
            FROM render r;
            """
        )

    def _migration_021_detect_keypoint_quality_review_columns(self) -> None:
        # Additive migration for shared detect/keypoint review fields in quality tables.
        self._ensure_columns(
            "keypoint_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
                "review_policy_id": "TEXT",
                "review_policy_version": "INTEGER",
            },
        )
        self._ensure_columns(
            "detect_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        cur = self.conn.cursor()
        self._refresh_keypoint_quality_current_view()
        cur.execute("DROP VIEW IF EXISTS detect_quality_current;")
        cur.execute(
            """
            CREATE VIEW detect_quality_current AS
            WITH ranked AS (
                SELECT
                    dq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dq.dataset_id, COALESCE(dq.detect_method, '')
                        ORDER BY
                            COALESCE(dq.review_timestamp_utc, dq.refined_created_utc, dq.quality_updated_utc) DESC,
                            COALESCE(dq.refined_created_utc, '') DESC,
                            dq.refined_run DESC
                    ) AS _rn
                FROM detect_quality dq
            )
            SELECT
                dataset_id,
                refined_run,
                refined_created_utc,
                source_detect_run,
                detect_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_notes,
                review_timestamp_utc,
                review_resolved_group,
                total_detections,
                real_detections,
                interpolated_detections,
                interpolated_detections_rate,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS refined_detect_review_current;")
        cur.execute(
            """
            CREATE VIEW refined_detect_review_current AS
            SELECT * FROM detect_quality_current;
            """
        )

    def _migration_022_detection_data_profile_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                detection_type TEXT,
                detection_path TEXT,
                profile_created_utc TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                frames_total INTEGER,
                frames_with_detections INTEGER,
                coverage_percent REAL,
                detections_total INTEGER,
                detections_per_frame_p50 REAL,
                detections_per_frame_p90 REAL,
                w_p10 REAL,
                w_p50 REAL,
                w_p90 REAL,
                h_p10 REAL,
                h_p50 REAL,
                h_p90 REAL,
                area_p10 REAL,
                area_p50 REAL,
                area_p90 REAL,
                aspect_ratio_p10 REAL,
                aspect_ratio_p50 REAL,
                aspect_ratio_p90 REAL,
                edge_proximity_rate REAL,
                rig_id TEXT,
                camera_id TEXT,
                arena_id TEXT,
                dish_design TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                genotype TEXT,
                dpf_at_acquisition INTEGER,
                profile_json TEXT,
                PRIMARY KEY (dataset_id, profile_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "detection_data_profile",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "detection_type": "TEXT",
                "detection_path": "TEXT",
                "profile_created_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "frames_total": "INTEGER",
                "frames_with_detections": "INTEGER",
                "coverage_percent": "REAL",
                "detections_total": "INTEGER",
                "detections_per_frame_p50": "REAL",
                "detections_per_frame_p90": "REAL",
                "w_p10": "REAL",
                "w_p50": "REAL",
                "w_p90": "REAL",
                "h_p10": "REAL",
                "h_p50": "REAL",
                "h_p90": "REAL",
                "area_p10": "REAL",
                "area_p50": "REAL",
                "area_p90": "REAL",
                "aspect_ratio_p10": "REAL",
                "aspect_ratio_p50": "REAL",
                "aspect_ratio_p90": "REAL",
                "edge_proximity_rate": "REAL",
                "rig_id": "TEXT",
                "camera_id": "TEXT",
                "arena_id": "TEXT",
                "dish_design": "TEXT",
                "canvas_name": "TEXT",
                "protocol_name": "TEXT",
                "genotype": "TEXT",
                "dpf_at_acquisition": "INTEGER",
                "profile_json": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_data_profile_recording_created "
            "ON detection_data_profile(recording_id, profile_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_data_profile_detection_scope "
            "ON detection_data_profile(detection_type, zarr_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_data_profile_coverage "
            "ON detection_data_profile(coverage_percent);"
        )
        cur.execute("DROP VIEW IF EXISTS detection_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW detection_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    ddp.dataset_id AS dataset_id,
                    ddp.profile_run AS profile_run,
                    COALESCE(dcc.recording_id, tip.recording_id) AS recording_id,
                    COALESCE(dcc.zarr_use, tip.zarr_use) AS zarr_use,
                    ddp.detection_type AS detection_type,
                    ddp.detection_path AS detection_path,
                    ddp.profile_created_utc AS profile_created_utc,
                    ddp.zarr_mtime_ns AS zarr_mtime_ns,
                    ddp.updated_utc AS updated_utc,
                    ddp.frames_total AS frames_total,
                    ddp.frames_with_detections AS frames_with_detections,
                    ddp.coverage_percent AS coverage_percent,
                    ddp.detections_total AS detections_total,
                    ddp.detections_per_frame_p50 AS detections_per_frame_p50,
                    ddp.detections_per_frame_p90 AS detections_per_frame_p90,
                    ddp.w_p10 AS w_p10,
                    ddp.w_p50 AS w_p50,
                    ddp.w_p90 AS w_p90,
                    ddp.h_p10 AS h_p10,
                    ddp.h_p50 AS h_p50,
                    ddp.h_p90 AS h_p90,
                    ddp.area_p10 AS area_p10,
                    ddp.area_p50 AS area_p50,
                    ddp.area_p90 AS area_p90,
                    ddp.aspect_ratio_p10 AS aspect_ratio_p10,
                    ddp.aspect_ratio_p50 AS aspect_ratio_p50,
                    ddp.aspect_ratio_p90 AS aspect_ratio_p90,
                    ddp.edge_proximity_rate AS edge_proximity_rate,
                    COALESCE(dcc.rig_id, ddp.rig_id) AS rig_id,
                    COALESCE(dcc.camera_id, ddp.camera_id) AS camera_id,
                    COALESCE(dcc.arena_id, ddp.arena_id) AS arena_id,
                    COALESCE(dcc.dish_design, ddp.dish_design) AS dish_design,
                    COALESCE(dcc.canvas_name, ddp.canvas_name) AS canvas_name,
                    COALESCE(dcc.protocol_name, ddp.protocol_name) AS protocol_name,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.genotype
                        ELSE COALESCE(dcc.genotype, ddp.genotype)
                    END AS genotype,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.dpf_at_acquisition
                        ELSE COALESCE(dcc.dpf_at_acquisition, ddp.dpf_at_acquisition)
                    END AS dpf_at_acquisition,
                    ddp.profile_json AS profile_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY ddp.dataset_id
                        ORDER BY
                            COALESCE(ddp.profile_created_utc, ddp.updated_utc) DESC,
                            ddp.profile_run DESC
                    ) AS _rn
                FROM detection_data_profile ddp
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = ddp.dataset_id
            )
            SELECT
                dataset_id,
                profile_run,
                recording_id,
                zarr_use,
                detection_type,
                detection_path,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                frames_total,
                frames_with_detections,
                coverage_percent,
                detections_total,
                detections_per_frame_p50,
                detections_per_frame_p90,
                w_p10,
                w_p50,
                w_p90,
                h_p10,
                h_p50,
                h_p90,
                area_p10,
                area_p50,
                area_p90,
                aspect_ratio_p10,
                aspect_ratio_p50,
                aspect_ratio_p90,
                edge_proximity_rate,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_detection_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW recording_detection_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    ddpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    ROW_NUMBER() OVER (
                        PARTITION BY ddpl.recording_id
                        ORDER BY
                            COALESCE(ddpl.profile_created_utc, ddpl.updated_utc) DESC,
                            ddpl.profile_run DESC,
                            ddpl.dataset_id DESC
                    ) AS _rn
                FROM detection_data_profile_latest ddpl
                LEFT JOIN datasets d ON d.dataset_id = ddpl.dataset_id
                WHERE ddpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                profile_run,
                zarr_use,
                detection_type,
                detection_path,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                frames_total,
                frames_with_detections,
                coverage_percent,
                detections_total,
                detections_per_frame_p50,
                detections_per_frame_p90,
                w_p10,
                w_p50,
                w_p90,
                h_p10,
                h_p50,
                h_p90,
                area_p10,
                area_p50,
                area_p90,
                aspect_ratio_p10,
                aspect_ratio_p50,
                aspect_ratio_p90,
                edge_proximity_rate,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json,
                zarr_path,
                artifact_kind,
                dataset_status
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_023_detection_data_profile_lineage_projection(self) -> None:
        # Append-only follow-up migration: existing registries may already be at
        # v22 from before lineage projection columns were added. Re-run the v22
        # reconciler to ensure columns/views are present with the latest shape.
        self._migration_022_detection_data_profile_registry()

    def _migration_024_keypoint_data_profile_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS keypoint_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                keypoint_method TEXT,
                source_keypoint_path TEXT,
                source_keypoint_run TEXT,
                skeleton_id TEXT,
                kpt_shape TEXT,
                profile_created_utc TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                rows_total INTEGER,
                rows_usable INTEGER,
                usable_keypoints_total INTEGER,
                usable_rate REAL,
                confidence_valid_rate REAL,
                geometry_valid_rate REAL,
                triangle_area_p10 REAL,
                triangle_area_p50 REAL,
                triangle_area_p90 REAL,
                min_angle_p10 REAL,
                min_angle_p50 REAL,
                min_angle_p90 REAL,
                heading_p10 REAL,
                heading_p50 REAL,
                heading_p90 REAL,
                rig_id TEXT,
                camera_id TEXT,
                arena_id TEXT,
                dish_design TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                genotype TEXT,
                dpf_at_acquisition INTEGER,
                profile_json TEXT,
                PRIMARY KEY (dataset_id, profile_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "keypoint_data_profile",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "keypoint_method": "TEXT",
                "source_keypoint_path": "TEXT",
                "source_keypoint_run": "TEXT",
                "skeleton_id": "TEXT",
                "kpt_shape": "TEXT",
                "pose_schema_name": "TEXT",
                "pose_schema_json": "TEXT",
                "heading_computation_source": "TEXT",
                "heading_computation_json": "TEXT",
                "profile_created_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "rows_total": "INTEGER",
                "rows_usable": "INTEGER",
                "usable_keypoints_total": "INTEGER",
                "usable_rate": "REAL",
                "confidence_valid_rate": "REAL",
                "geometry_valid_rate": "REAL",
                "triangle_area_p10": "REAL",
                "triangle_area_p50": "REAL",
                "triangle_area_p90": "REAL",
                "min_angle_p10": "REAL",
                "min_angle_p50": "REAL",
                "min_angle_p90": "REAL",
                "heading_p10": "REAL",
                "heading_p50": "REAL",
                "heading_p90": "REAL",
                "rig_id": "TEXT",
                "camera_id": "TEXT",
                "arena_id": "TEXT",
                "dish_design": "TEXT",
                "canvas_name": "TEXT",
                "protocol_name": "TEXT",
                "genotype": "TEXT",
                "dpf_at_acquisition": "INTEGER",
                "profile_json": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_dataset "
            "ON keypoint_data_profile(dataset_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_recording_created "
            "ON keypoint_data_profile(recording_id, profile_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_method_scope "
            "ON keypoint_data_profile(keypoint_method, zarr_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_method_usable_rate "
            "ON keypoint_data_profile(keypoint_method, usable_rate);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_lineage "
            "ON keypoint_data_profile(genotype, dpf_at_acquisition);"
        )
        cur.execute("DROP VIEW IF EXISTS keypoint_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW keypoint_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    kdp.dataset_id AS dataset_id,
                    kdp.profile_run AS profile_run,
                    dcc.recording_id AS recording_id,
                    dcc.zarr_use AS zarr_use,
                    kdp.keypoint_method AS keypoint_method,
                    kdp.source_keypoint_path AS source_keypoint_path,
                    kdp.source_keypoint_run AS source_keypoint_run,
                    kdp.skeleton_id AS skeleton_id,
                    kdp.kpt_shape AS kpt_shape,
                    kdp.pose_schema_name AS pose_schema_name,
                    kdp.pose_schema_json AS pose_schema_json,
                    kdp.heading_computation_source AS heading_computation_source,
                    kdp.heading_computation_json AS heading_computation_json,
                    kdp.profile_created_utc AS profile_created_utc,
                    kdp.zarr_mtime_ns AS zarr_mtime_ns,
                    kdp.updated_utc AS updated_utc,
                    kdp.rows_total AS rows_total,
                    kdp.rows_usable AS rows_usable,
                    kdp.usable_keypoints_total AS usable_keypoints_total,
                    kdp.usable_rate AS usable_rate,
                    kdp.confidence_valid_rate AS confidence_valid_rate,
                    kdp.geometry_valid_rate AS geometry_valid_rate,
                    kdp.triangle_area_p10 AS triangle_area_p10,
                    kdp.triangle_area_p50 AS triangle_area_p50,
                    kdp.triangle_area_p90 AS triangle_area_p90,
                    kdp.min_angle_p10 AS min_angle_p10,
                    kdp.min_angle_p50 AS min_angle_p50,
                    kdp.min_angle_p90 AS min_angle_p90,
                    kdp.heading_p10 AS heading_p10,
                    kdp.heading_p50 AS heading_p50,
                    kdp.heading_p90 AS heading_p90,
                    COALESCE(dcc.rig_id, kdp.rig_id) AS rig_id,
                    COALESCE(dcc.camera_id, kdp.camera_id) AS camera_id,
                    COALESCE(dcc.arena_id, kdp.arena_id) AS arena_id,
                    COALESCE(dcc.dish_design, kdp.dish_design) AS dish_design,
                    COALESCE(dcc.canvas_name, kdp.canvas_name) AS canvas_name,
                    COALESCE(dcc.protocol_name, kdp.protocol_name) AS protocol_name,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.genotype
                        ELSE COALESCE(dcc.genotype, kdp.genotype)
                    END AS genotype,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.dpf_at_acquisition
                        ELSE COALESCE(dcc.dpf_at_acquisition, kdp.dpf_at_acquisition)
                    END AS dpf_at_acquisition,
                    kdp.profile_json AS profile_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY kdp.dataset_id, COALESCE(kdp.keypoint_method, '')
                        ORDER BY
                            COALESCE(kdp.profile_created_utc, kdp.updated_utc) DESC,
                            kdp.profile_run DESC
                    ) AS _rn
                FROM keypoint_data_profile kdp
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = kdp.dataset_id
            )
            SELECT
                dataset_id,
                profile_run,
                recording_id,
                zarr_use,
                keypoint_method,
                source_keypoint_path,
                source_keypoint_run,
                skeleton_id,
                kpt_shape,
                pose_schema_name,
                pose_schema_json,
                heading_computation_source,
                heading_computation_json,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                rows_total,
                rows_usable,
                usable_keypoints_total,
                usable_rate,
                confidence_valid_rate,
                geometry_valid_rate,
                triangle_area_p10,
                triangle_area_p50,
                triangle_area_p90,
                min_angle_p10,
                min_angle_p50,
                min_angle_p90,
                heading_p10,
                heading_p50,
                heading_p90,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_keypoint_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW recording_keypoint_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    kdpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    ROW_NUMBER() OVER (
                        PARTITION BY kdpl.recording_id, COALESCE(kdpl.keypoint_method, '')
                        ORDER BY
                            COALESCE(kdpl.profile_created_utc, kdpl.updated_utc) DESC,
                            kdpl.profile_run DESC,
                            kdpl.dataset_id DESC
                    ) AS _rn
                FROM keypoint_data_profile_latest kdpl
                LEFT JOIN datasets d ON d.dataset_id = kdpl.dataset_id
                WHERE kdpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                profile_run,
                zarr_use,
                keypoint_method,
                source_keypoint_path,
                source_keypoint_run,
                skeleton_id,
                kpt_shape,
                pose_schema_name,
                pose_schema_json,
                heading_computation_source,
                heading_computation_json,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                rows_total,
                rows_usable,
                usable_keypoints_total,
                usable_rate,
                confidence_valid_rate,
                geometry_valid_rate,
                triangle_area_p10,
                triangle_area_p50,
                triangle_area_p90,
                min_angle_p10,
                min_angle_p50,
                min_angle_p90,
                heading_p10,
                heading_p50,
                heading_p90,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json,
                zarr_path,
                artifact_kind,
                dataset_status
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_025_eye_mask_data_profile_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS eye_mask_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                stage_group TEXT,
                eye_mask_method TEXT,
                source_eye_mask_path TEXT,
                source_eye_mask_run TEXT,
                source_keypoint_path TEXT,
                source_keypoint_run TEXT,
                source_crop_run TEXT,
                profile_created_utc TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                rows_total INTEGER,
                rows_usable INTEGER,
                usable_rate REAL,
                reviewed_rate REAL,
                excluded_rate REAL,
                exclusion_reasons_json TEXT,
                ellipse_success_rate REAL,
                pair_success_rate REAL,
                area_p10 REAL,
                area_p50 REAL,
                area_p90 REAL,
                left_area_p10 REAL,
                left_area_p50 REAL,
                left_area_p90 REAL,
                right_area_p10 REAL,
                right_area_p50 REAL,
                right_area_p90 REAL,
                union_area_p10 REAL,
                union_area_p50 REAL,
                union_area_p90 REAL,
                area_lr_ratio_p10 REAL,
                area_lr_ratio_p50 REAL,
                area_lr_ratio_p90 REAL,
                major_axis_p10 REAL,
                major_axis_p50 REAL,
                major_axis_p90 REAL,
                minor_axis_p10 REAL,
                minor_axis_p50 REAL,
                minor_axis_p90 REAL,
                aspect_ratio_p10 REAL,
                aspect_ratio_p50 REAL,
                aspect_ratio_p90 REAL,
                eye_separation_p10 REAL,
                eye_separation_p50 REAL,
                eye_separation_p90 REAL,
                edge_proximity_rate REAL,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_timestamp_utc TEXT,
                source_keypoint_stale_state TEXT,
                source_keypoint_stale_reason TEXT,
                source_keypoint_stale_timestamp_utc TEXT,
                source_keypoint_stale_json TEXT,
                rig_id TEXT,
                camera_id TEXT,
                arena_id TEXT,
                dish_design TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                genotype TEXT,
                dpf_at_acquisition INTEGER,
                profile_json TEXT,
                PRIMARY KEY (dataset_id, profile_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "eye_mask_data_profile",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "stage_group": "TEXT",
                "eye_mask_method": "TEXT",
                "source_eye_mask_path": "TEXT",
                "source_eye_mask_run": "TEXT",
                "source_keypoint_path": "TEXT",
                "source_keypoint_run": "TEXT",
                "source_crop_run": "TEXT",
                "profile_created_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "rows_total": "INTEGER",
                "rows_usable": "INTEGER",
                "usable_rate": "REAL",
                "reviewed_rate": "REAL",
                "excluded_rate": "REAL",
                "exclusion_reasons_json": "TEXT",
                "ellipse_success_rate": "REAL",
                "pair_success_rate": "REAL",
                "area_p10": "REAL",
                "area_p50": "REAL",
                "area_p90": "REAL",
                "left_area_p10": "REAL",
                "left_area_p50": "REAL",
                "left_area_p90": "REAL",
                "right_area_p10": "REAL",
                "right_area_p50": "REAL",
                "right_area_p90": "REAL",
                "union_area_p10": "REAL",
                "union_area_p50": "REAL",
                "union_area_p90": "REAL",
                "area_lr_ratio_p10": "REAL",
                "area_lr_ratio_p50": "REAL",
                "area_lr_ratio_p90": "REAL",
                "major_axis_p10": "REAL",
                "major_axis_p50": "REAL",
                "major_axis_p90": "REAL",
                "minor_axis_p10": "REAL",
                "minor_axis_p50": "REAL",
                "minor_axis_p90": "REAL",
                "aspect_ratio_p10": "REAL",
                "aspect_ratio_p50": "REAL",
                "aspect_ratio_p90": "REAL",
                "eye_separation_p10": "REAL",
                "eye_separation_p50": "REAL",
                "eye_separation_p90": "REAL",
                "edge_proximity_rate": "REAL",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_timestamp_utc": "TEXT",
                "source_keypoint_stale_state": "TEXT",
                "source_keypoint_stale_reason": "TEXT",
                "source_keypoint_stale_timestamp_utc": "TEXT",
                "source_keypoint_stale_json": "TEXT",
                "rig_id": "TEXT",
                "camera_id": "TEXT",
                "arena_id": "TEXT",
                "dish_design": "TEXT",
                "canvas_name": "TEXT",
                "protocol_name": "TEXT",
                "genotype": "TEXT",
                "dpf_at_acquisition": "INTEGER",
                "profile_json": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_recording_created "
            "ON eye_mask_data_profile(recording_id, profile_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_method_scope "
            "ON eye_mask_data_profile(eye_mask_method, zarr_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_stage_usable_rate "
            "ON eye_mask_data_profile(stage_group, usable_rate);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_stale_state "
            "ON eye_mask_data_profile(source_keypoint_stale_state);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_lineage "
            "ON eye_mask_data_profile(genotype, dpf_at_acquisition);"
        )

        cur.execute("DROP VIEW IF EXISTS eye_mask_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW eye_mask_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    emdp.dataset_id AS dataset_id,
                    emdp.profile_run AS profile_run,
                    dcc.recording_id AS recording_id,
                    dcc.zarr_use AS zarr_use,
                    emdp.stage_group AS stage_group,
                    emdp.eye_mask_method AS eye_mask_method,
                    emdp.source_eye_mask_path AS source_eye_mask_path,
                    emdp.source_eye_mask_run AS source_eye_mask_run,
                    emdp.source_keypoint_path AS source_keypoint_path,
                    emdp.source_keypoint_run AS source_keypoint_run,
                    emdp.source_crop_run AS source_crop_run,
                    emdp.profile_created_utc AS profile_created_utc,
                    emdp.zarr_mtime_ns AS zarr_mtime_ns,
                    emdp.updated_utc AS updated_utc,
                    emdp.rows_total AS rows_total,
                    emdp.rows_usable AS rows_usable,
                    emdp.usable_rate AS usable_rate,
                    emdp.reviewed_rate AS reviewed_rate,
                    emdp.excluded_rate AS excluded_rate,
                    emdp.exclusion_reasons_json AS exclusion_reasons_json,
                    emdp.ellipse_success_rate AS ellipse_success_rate,
                    emdp.pair_success_rate AS pair_success_rate,
                    emdp.area_p10 AS area_p10,
                    emdp.area_p50 AS area_p50,
                    emdp.area_p90 AS area_p90,
                    emdp.left_area_p10 AS left_area_p10,
                    emdp.left_area_p50 AS left_area_p50,
                    emdp.left_area_p90 AS left_area_p90,
                    emdp.right_area_p10 AS right_area_p10,
                    emdp.right_area_p50 AS right_area_p50,
                    emdp.right_area_p90 AS right_area_p90,
                    emdp.union_area_p10 AS union_area_p10,
                    emdp.union_area_p50 AS union_area_p50,
                    emdp.union_area_p90 AS union_area_p90,
                    emdp.area_lr_ratio_p10 AS area_lr_ratio_p10,
                    emdp.area_lr_ratio_p50 AS area_lr_ratio_p50,
                    emdp.area_lr_ratio_p90 AS area_lr_ratio_p90,
                    emdp.major_axis_p10 AS major_axis_p10,
                    emdp.major_axis_p50 AS major_axis_p50,
                    emdp.major_axis_p90 AS major_axis_p90,
                    emdp.minor_axis_p10 AS minor_axis_p10,
                    emdp.minor_axis_p50 AS minor_axis_p50,
                    emdp.minor_axis_p90 AS minor_axis_p90,
                    emdp.aspect_ratio_p10 AS aspect_ratio_p10,
                    emdp.aspect_ratio_p50 AS aspect_ratio_p50,
                    emdp.aspect_ratio_p90 AS aspect_ratio_p90,
                    emdp.eye_separation_p10 AS eye_separation_p10,
                    emdp.eye_separation_p50 AS eye_separation_p50,
                    emdp.eye_separation_p90 AS eye_separation_p90,
                    emdp.edge_proximity_rate AS edge_proximity_rate,
                    emdp.review_state AS review_state,
                    emdp.review_method AS review_method,
                    emdp.review_intended_use AS review_intended_use,
                    emdp.review_timestamp_utc AS review_timestamp_utc,
                    emdp.source_keypoint_stale_state AS source_keypoint_stale_state,
                    emdp.source_keypoint_stale_reason AS source_keypoint_stale_reason,
                    emdp.source_keypoint_stale_timestamp_utc AS source_keypoint_stale_timestamp_utc,
                    emdp.source_keypoint_stale_json AS source_keypoint_stale_json,
                    COALESCE(dcc.rig_id, emdp.rig_id) AS rig_id,
                    COALESCE(dcc.camera_id, emdp.camera_id) AS camera_id,
                    COALESCE(dcc.arena_id, emdp.arena_id) AS arena_id,
                    COALESCE(dcc.dish_design, emdp.dish_design) AS dish_design,
                    COALESCE(dcc.canvas_name, emdp.canvas_name) AS canvas_name,
                    COALESCE(dcc.protocol_name, emdp.protocol_name) AS protocol_name,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.genotype
                        ELSE COALESCE(dcc.genotype, emdp.genotype)
                    END AS genotype,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.dpf_at_acquisition
                        ELSE COALESCE(dcc.dpf_at_acquisition, emdp.dpf_at_acquisition)
                    END AS dpf_at_acquisition,
                    emdp.profile_json AS profile_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY emdp.dataset_id, COALESCE(emdp.stage_group, ''), COALESCE(emdp.eye_mask_method, '')
                        ORDER BY
                            COALESCE(emdp.profile_created_utc, emdp.updated_utc) DESC,
                            emdp.profile_run DESC
                    ) AS _rn
                FROM eye_mask_data_profile emdp
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = emdp.dataset_id
            )
            SELECT
                dataset_id,
                profile_run,
                recording_id,
                zarr_use,
                stage_group,
                eye_mask_method,
                source_eye_mask_path,
                source_eye_mask_run,
                source_keypoint_path,
                source_keypoint_run,
                source_crop_run,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                rows_total,
                rows_usable,
                usable_rate,
                reviewed_rate,
                excluded_rate,
                exclusion_reasons_json,
                ellipse_success_rate,
                pair_success_rate,
                area_p10,
                area_p50,
                area_p90,
                left_area_p10,
                left_area_p50,
                left_area_p90,
                right_area_p10,
                right_area_p50,
                right_area_p90,
                union_area_p10,
                union_area_p50,
                union_area_p90,
                area_lr_ratio_p10,
                area_lr_ratio_p50,
                area_lr_ratio_p90,
                major_axis_p10,
                major_axis_p50,
                major_axis_p90,
                minor_axis_p10,
                minor_axis_p50,
                minor_axis_p90,
                aspect_ratio_p10,
                aspect_ratio_p50,
                aspect_ratio_p90,
                eye_separation_p10,
                eye_separation_p50,
                eye_separation_p90,
                edge_proximity_rate,
                review_state,
                review_method,
                review_intended_use,
                review_timestamp_utc,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_eye_mask_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW recording_eye_mask_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    emdpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    ROW_NUMBER() OVER (
                        PARTITION BY emdpl.recording_id, COALESCE(emdpl.stage_group, ''), COALESCE(emdpl.eye_mask_method, '')
                        ORDER BY
                            COALESCE(emdpl.profile_created_utc, emdpl.updated_utc) DESC,
                            emdpl.profile_run DESC,
                            emdpl.dataset_id DESC
                    ) AS _rn
                FROM eye_mask_data_profile_latest emdpl
                LEFT JOIN datasets d ON d.dataset_id = emdpl.dataset_id
                WHERE emdpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                profile_run,
                zarr_use,
                stage_group,
                eye_mask_method,
                source_eye_mask_path,
                source_eye_mask_run,
                source_keypoint_path,
                source_keypoint_run,
                source_crop_run,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                rows_total,
                rows_usable,
                usable_rate,
                reviewed_rate,
                excluded_rate,
                exclusion_reasons_json,
                ellipse_success_rate,
                pair_success_rate,
                area_p10,
                area_p50,
                area_p90,
                left_area_p10,
                left_area_p50,
                left_area_p90,
                right_area_p10,
                right_area_p50,
                right_area_p90,
                union_area_p10,
                union_area_p50,
                union_area_p90,
                area_lr_ratio_p10,
                area_lr_ratio_p50,
                area_lr_ratio_p90,
                major_axis_p10,
                major_axis_p50,
                major_axis_p90,
                minor_axis_p10,
                minor_axis_p50,
                minor_axis_p90,
                aspect_ratio_p10,
                aspect_ratio_p50,
                aspect_ratio_p90,
                eye_separation_p10,
                eye_separation_p50,
                eye_separation_p90,
                edge_proximity_rate,
                review_state,
                review_method,
                review_intended_use,
                review_timestamp_utc,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json,
                zarr_path,
                artifact_kind,
                dataset_status
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_026_eye_mask_quality_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS eye_mask_quality (
                dataset_id TEXT NOT NULL,
                stage_group TEXT NOT NULL,
                run_name TEXT NOT NULL,
                run_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                eye_mask_method TEXT,
                source_crop_run TEXT,
                source_keypoint_group TEXT,
                source_keypoints_run TEXT,
                source_eye_masks_run TEXT,
                source_eye_masks_method TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                total_rois INTEGER,
                successful_eyes INTEGER,
                successful_roi_pairs INTEGER,
                successful_roi_pair_rate REAL,
                source_keypoint_stale_state TEXT,
                source_keypoint_stale_reason TEXT,
                source_keypoint_stale_timestamp_utc TEXT,
                source_keypoint_stale_json TEXT,
                lifecycle_state TEXT,
                lifecycle_reason TEXT,
                quality_updated_utc TEXT,
                zarr_mtime_ns INTEGER,
                PRIMARY KEY (dataset_id, stage_group, run_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "eye_mask_quality",
            {
                "run_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "eye_mask_method": "TEXT",
                "source_crop_run": "TEXT",
                "source_keypoint_group": "TEXT",
                "source_keypoints_run": "TEXT",
                "source_eye_masks_run": "TEXT",
                "source_eye_masks_method": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "total_rois": "INTEGER",
                "successful_eyes": "INTEGER",
                "successful_roi_pairs": "INTEGER",
                "successful_roi_pair_rate": "REAL",
                "source_keypoint_stale_state": "TEXT",
                "source_keypoint_stale_reason": "TEXT",
                "source_keypoint_stale_timestamp_utc": "TEXT",
                "source_keypoint_stale_json": "TEXT",
                "lifecycle_state": "TEXT",
                "lifecycle_reason": "TEXT",
                "quality_updated_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
            },
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_quality_dataset_id ON eye_mask_quality(dataset_id);"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_eye_mask_quality_gate
            ON eye_mask_quality(review_state, review_intended_use, eye_mask_method, successful_roi_pair_rate);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_eye_mask_quality_stage_method
            ON eye_mask_quality(stage_group, eye_mask_method);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_eye_mask_quality_recording
            ON eye_mask_quality(recording_id, stage_group, run_created_utc DESC);
            """
        )

        cur.execute(
            """
            INSERT INTO eye_mask_quality (
                dataset_id, stage_group, run_name, run_created_utc, recording_id, zarr_use,
                eye_mask_method, source_crop_run, source_keypoint_group, source_keypoints_run,
                source_eye_masks_run, source_eye_masks_method,
                review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                total_rois, successful_eyes, successful_roi_pairs, successful_roi_pair_rate,
                source_keypoint_stale_state, source_keypoint_stale_reason, source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json, lifecycle_state, lifecycle_reason, quality_updated_utc, zarr_mtime_ns
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                recording_id,
                zarr_use,
                method AS eye_mask_method,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_eye_masks_run,
                source_eye_masks_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                successful_eyes,
                successful_roi_pairs,
                successful_roi_pair_rate,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                lifecycle_state,
                lifecycle_reason,
                COALESCE(updated_utc, CURRENT_TIMESTAMP) AS quality_updated_utc,
                zarr_mtime_ns
            FROM eye_mask_performance
            WHERE stage_group = 'refined_eye_masks_runs'
            ON CONFLICT(dataset_id, stage_group, run_name) DO UPDATE SET
                run_created_utc=excluded.run_created_utc,
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                eye_mask_method=excluded.eye_mask_method,
                source_crop_run=excluded.source_crop_run,
                source_keypoint_group=excluded.source_keypoint_group,
                source_keypoints_run=excluded.source_keypoints_run,
                source_eye_masks_run=excluded.source_eye_masks_run,
                source_eye_masks_method=excluded.source_eye_masks_method,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_reviewer=excluded.review_reviewer,
                review_timestamp_utc=excluded.review_timestamp_utc,
                total_rois=excluded.total_rois,
                successful_eyes=excluded.successful_eyes,
                successful_roi_pairs=excluded.successful_roi_pairs,
                successful_roi_pair_rate=excluded.successful_roi_pair_rate,
                source_keypoint_stale_state=excluded.source_keypoint_stale_state,
                source_keypoint_stale_reason=excluded.source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc=excluded.source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json=excluded.source_keypoint_stale_json,
                lifecycle_state=excluded.lifecycle_state,
                lifecycle_reason=excluded.lifecycle_reason,
                quality_updated_utc=excluded.quality_updated_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns;
            """
        )

        cur.execute("DROP VIEW IF EXISTS eye_mask_quality_current;")
        cur.execute(
            """
            CREATE VIEW eye_mask_quality_current AS
            WITH ranked AS (
                SELECT
                    emq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY emq.dataset_id, COALESCE(emq.stage_group, ''), COALESCE(emq.eye_mask_method, '')
                        ORDER BY
                            COALESCE(emq.review_timestamp_utc, emq.run_created_utc, emq.quality_updated_utc) DESC,
                            COALESCE(emq.run_created_utc, '') DESC,
                            emq.run_name DESC
                    ) AS _rn
                FROM eye_mask_quality emq
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                recording_id,
                zarr_use,
                eye_mask_method,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_eye_masks_run,
                source_eye_masks_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                successful_eyes,
                successful_roi_pairs,
                successful_roi_pair_rate,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS eye_mask_quality_overview;")
        cur.execute(
            """
            CREATE VIEW eye_mask_quality_overview AS
            SELECT
                emqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                d.artifact_kind AS artifact_kind,
                d.status AS dataset_status,
                emqc.stage_group AS stage_group,
                emqc.run_name AS run_name,
                emqc.run_created_utc AS run_created_utc,
                emqc.recording_id AS recording_id,
                emqc.eye_mask_method AS eye_mask_method,
                emqc.source_crop_run AS source_crop_run,
                emqc.source_keypoint_group AS source_keypoint_group,
                emqc.source_keypoints_run AS source_keypoints_run,
                emqc.source_eye_masks_run AS source_eye_masks_run,
                emqc.source_eye_masks_method AS source_eye_masks_method,
                emqc.review_state AS review_state,
                emqc.review_method AS review_method,
                emqc.review_intended_use AS review_intended_use,
                emqc.review_reviewer AS review_reviewer,
                emqc.review_timestamp_utc AS review_timestamp_utc,
                emqc.total_rois AS total_rois,
                emqc.successful_eyes AS successful_eyes,
                emqc.successful_roi_pairs AS successful_roi_pairs,
                emqc.successful_roi_pair_rate AS successful_roi_pair_rate,
                emqc.source_keypoint_stale_state AS source_keypoint_stale_state,
                emqc.source_keypoint_stale_reason AS source_keypoint_stale_reason,
                emqc.source_keypoint_stale_timestamp_utc AS source_keypoint_stale_timestamp_utc,
                emqc.source_keypoint_stale_json AS source_keypoint_stale_json,
                emqc.lifecycle_state AS lifecycle_state,
                emqc.lifecycle_reason AS lifecycle_reason,
                emqc.quality_updated_utc AS quality_updated_utc,
                emqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN emqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM eye_mask_quality_current emqc
            LEFT JOIN datasets d ON d.dataset_id = emqc.dataset_id;
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_eye_mask_quality_overview;")
        cur.execute(
            """
            CREATE VIEW recording_eye_mask_quality_overview AS
            WITH ranked AS (
                SELECT
                    emqo.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY emqo.recording_id, COALESCE(emqo.stage_group, ''), COALESCE(emqo.eye_mask_method, '')
                        ORDER BY
                            COALESCE(emqo.review_timestamp_utc, emqo.run_created_utc, emqo.quality_updated_utc) DESC,
                            COALESCE(emqo.run_created_utc, '') DESC,
                            emqo.run_name DESC,
                            emqo.dataset_id DESC
                    ) AS _rn
                FROM eye_mask_quality_overview emqo
                WHERE emqo.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                run_created_utc,
                eye_mask_method,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_eye_masks_run,
                source_eye_masks_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                successful_eyes,
                successful_roi_pairs,
                successful_roi_pair_rate,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_027_detect_quality_wide_view_columns(self) -> None:
        """Re-create wide view to include detect_quality step columns."""
        self._migration_020_recording_step_status_wide_view()

    def _migration_028_keypoint_auto_review_policy_columns(self) -> None:
        self._ensure_columns(
            "keypoint_quality",
            {
                "review_policy_id": "TEXT",
                "review_policy_version": "INTEGER",
            },
        )
        cur = self.conn.cursor()
        self._refresh_keypoint_quality_current_view()
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_overview;")
        cur.execute(
            """
            CREATE VIEW keypoint_quality_overview AS
            SELECT
                kqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                kqc.keypoint_method AS keypoint_method,
                kqc.source_keypoint_run AS source_keypoint_run,
                kqc.refined_run AS refined_run,
                kqc.review_state AS review_state,
                kqc.review_method AS review_method,
                kqc.review_intended_use AS review_intended_use,
                kqc.review_policy_id AS review_policy_id,
                kqc.review_policy_version AS review_policy_version,
                kqc.usable_keypoints AS usable_keypoints,
                kqc.total_keypoints AS total_keypoints,
                kqc.usable_keypoints_rate AS usable_keypoints_rate,
                kqc.quality_updated_utc AS quality_updated_utc,
                kqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN kqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM keypoint_quality_current kqc
            LEFT JOIN datasets d ON d.dataset_id = kqc.dataset_id;
            """
        )

    def _migration_029_keypoint_quality_current_latest_source_preference(self) -> None:
        self._refresh_keypoint_quality_current_view()

    def _migration_030_tracking_unassigned_warning_wide_view(self) -> None:
        """Re-create wide view to expose tracking unassigned-row warnings."""
        self._migration_020_recording_step_status_wide_view()

    def _migration_031_tracking_qc_state_wide_view(self) -> None:
        """Re-create wide view to expose structured tracking QA state."""
        self._migration_020_recording_step_status_wide_view()

    def _migration_032_subject_mask_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS subject_mask_performance (
                dataset_id TEXT NOT NULL,
                stage_group TEXT NOT NULL,
                run_name TEXT NOT NULL,
                run_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                subject_mask_method TEXT,
                label_schema_id TEXT,
                source_crop_run TEXT,
                source_keypoint_group TEXT,
                source_keypoints_run TEXT,
                source_subject_mask_run TEXT,
                source_subject_mask_method TEXT,
                run_semantics TEXT,
                probability_semantics TEXT,
                source_background_run TEXT,
                source_background_array TEXT,
                source_dish_mask_array TEXT,
                tuning_source TEXT,
                tuning_timestamp TEXT,
                total_rois INTEGER,
                rows_with_any_mask INTEGER,
                coverage_percent REAL,
                duration_seconds REAL,
                rois_per_second REAL,
                available_component_count INTEGER,
                available_components_json TEXT,
                unavailable_components_json TEXT,
                component_review_states_json TEXT,
                eye_component_mode TEXT,
                reason_counts_json TEXT,
                summary_statistics_json TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                source_subject_mask_stale_state TEXT,
                source_subject_mask_stale_reason TEXT,
                source_subject_mask_stale_timestamp_utc TEXT,
                source_subject_mask_stale_json TEXT,
                lifecycle_state TEXT,
                lifecycle_reason TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, stage_group, run_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS subject_mask_component_quality (
                dataset_id TEXT NOT NULL,
                stage_group TEXT NOT NULL,
                run_name TEXT NOT NULL,
                component_name TEXT NOT NULL,
                component_family TEXT,
                run_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                subject_mask_method TEXT,
                label_schema_id TEXT,
                eye_component_mode TEXT,
                source_subject_mask_run TEXT,
                available INTEGER,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                total_rois INTEGER,
                rows_with_component_mask INTEGER,
                rows_with_component_mask_rate REAL,
                lifecycle_state TEXT,
                lifecycle_reason TEXT,
                quality_updated_utc TEXT,
                zarr_mtime_ns INTEGER,
                PRIMARY KEY (dataset_id, stage_group, run_name, component_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "subject_mask_performance",
            {
                "run_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "subject_mask_method": "TEXT",
                "label_schema_id": "TEXT",
                "source_crop_run": "TEXT",
                "source_keypoint_group": "TEXT",
                "source_keypoints_run": "TEXT",
                "source_subject_mask_run": "TEXT",
                "source_subject_mask_method": "TEXT",
                "run_semantics": "TEXT",
                "probability_semantics": "TEXT",
                "source_background_run": "TEXT",
                "source_background_array": "TEXT",
                "source_dish_mask_array": "TEXT",
                "tuning_source": "TEXT",
                "tuning_timestamp": "TEXT",
                "total_rois": "INTEGER",
                "rows_with_any_mask": "INTEGER",
                "coverage_percent": "REAL",
                "duration_seconds": "REAL",
                "rois_per_second": "REAL",
                "available_component_count": "INTEGER",
                "available_components_json": "TEXT",
                "unavailable_components_json": "TEXT",
                "component_review_states_json": "TEXT",
                "eye_component_mode": "TEXT",
                "reason_counts_json": "TEXT",
                "summary_statistics_json": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "source_subject_mask_stale_state": "TEXT",
                "source_subject_mask_stale_reason": "TEXT",
                "source_subject_mask_stale_timestamp_utc": "TEXT",
                "source_subject_mask_stale_json": "TEXT",
                "lifecycle_state": "TEXT",
                "lifecycle_reason": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        self._ensure_columns(
            "subject_mask_component_quality",
            {
                "component_family": "TEXT",
                "run_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "subject_mask_method": "TEXT",
                "label_schema_id": "TEXT",
                "eye_component_mode": "TEXT",
                "source_subject_mask_run": "TEXT",
                "available": "INTEGER",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "total_rois": "INTEGER",
                "rows_with_component_mask": "INTEGER",
                "rows_with_component_mask_rate": "REAL",
                "lifecycle_state": "TEXT",
                "lifecycle_reason": "TEXT",
                "quality_updated_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
            },
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_perf_recording ON subject_mask_performance(recording_id, stage_group, run_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_perf_stage_method ON subject_mask_performance(stage_group, subject_mask_method);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_perf_source ON subject_mask_performance(source_keypoints_run, source_subject_mask_run);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_perf_review ON subject_mask_performance(review_state, review_intended_use, lifecycle_state);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_component_dataset_id ON subject_mask_component_quality(dataset_id);"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_subject_mask_component_gate
            ON subject_mask_component_quality(review_state, review_intended_use, component_name, available);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_subject_mask_component_stage
            ON subject_mask_component_quality(stage_group, component_name, subject_mask_method);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_subject_mask_component_recording
            ON subject_mask_component_quality(recording_id, stage_group, component_name, run_created_utc DESC);
            """
        )

    def _migration_033_subject_mask_registry_semantics_columns(self) -> None:
        """Reconcile subject-mask registry schema after legacy bootstrap registries."""
        self._migration_032_subject_mask_registry()
        cur = self.conn.cursor()

        cur.execute("DROP VIEW IF EXISTS subject_mask_performance_latest;")
        cur.execute(
            """
            CREATE VIEW subject_mask_performance_latest AS
            WITH ranked AS (
                SELECT
                    smp.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY smp.dataset_id, smp.stage_group
                        ORDER BY
                            COALESCE(smp.run_created_utc, smp.updated_utc) DESC,
                            smp.run_name DESC
                    ) AS _rn
                FROM subject_mask_performance smp
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                recording_id,
                zarr_use,
                subject_mask_method,
                label_schema_id,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_subject_mask_run,
                source_subject_mask_method,
                run_semantics,
                probability_semantics,
                source_background_run,
                source_background_array,
                source_dish_mask_array,
                tuning_source,
                tuning_timestamp,
                total_rois,
                rows_with_any_mask,
                coverage_percent,
                duration_seconds,
                rois_per_second,
                available_component_count,
                available_components_json,
                unavailable_components_json,
                component_review_states_json,
                eye_component_mode,
                reason_counts_json,
                summary_statistics_json,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                lifecycle_state,
                lifecycle_reason,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_subject_mask_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_subject_mask_performance_latest AS
            WITH ranked AS (
                SELECT
                    smpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    ROW_NUMBER() OVER (
                        PARTITION BY smpl.recording_id, smpl.stage_group
                        ORDER BY
                            COALESCE(smpl.run_created_utc, smpl.updated_utc) DESC,
                            smpl.run_name DESC
                    ) AS _rn
                FROM subject_mask_performance_latest smpl
                LEFT JOIN datasets d ON d.dataset_id = smpl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = smpl.dataset_id
                WHERE smpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                zarr_use,
                subject_mask_method,
                label_schema_id,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_subject_mask_run,
                source_subject_mask_method,
                run_semantics,
                probability_semantics,
                source_background_run,
                source_background_array,
                source_dish_mask_array,
                tuning_source,
                tuning_timestamp,
                total_rois,
                rows_with_any_mask,
                coverage_percent,
                duration_seconds,
                rois_per_second,
                available_component_count,
                available_components_json,
                unavailable_components_json,
                component_review_states_json,
                eye_component_mode,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                lifecycle_state,
                lifecycle_reason,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_current;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_current AS
            WITH ranked AS (
                SELECT
                    smcq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY smcq.dataset_id, smcq.stage_group, smcq.component_name
                        ORDER BY
                            CASE WHEN COALESCE(smcq.available, 0) = 1 THEN 1 ELSE 0 END DESC,
                            COALESCE(smcq.review_timestamp_utc, smcq.run_created_utc, smcq.quality_updated_utc) DESC,
                            COALESCE(smcq.run_created_utc, '') DESC,
                            smcq.run_name DESC
                    ) AS _rn
                FROM subject_mask_component_quality smcq
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                recording_id,
                zarr_use,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_overview;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_overview AS
            SELECT
                smcqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                d.artifact_kind AS artifact_kind,
                d.status AS dataset_status,
                smcqc.stage_group AS stage_group,
                smcqc.run_name AS run_name,
                smcqc.component_name AS component_name,
                smcqc.component_family AS component_family,
                smcqc.run_created_utc AS run_created_utc,
                smcqc.recording_id AS recording_id,
                smcqc.subject_mask_method AS subject_mask_method,
                smcqc.label_schema_id AS label_schema_id,
                smcqc.eye_component_mode AS eye_component_mode,
                smcqc.source_subject_mask_run AS source_subject_mask_run,
                smcqc.available AS available,
                smcqc.review_state AS review_state,
                smcqc.review_method AS review_method,
                smcqc.review_intended_use AS review_intended_use,
                smcqc.review_reviewer AS review_reviewer,
                smcqc.review_timestamp_utc AS review_timestamp_utc,
                smcqc.total_rois AS total_rois,
                smcqc.rows_with_component_mask AS rows_with_component_mask,
                smcqc.rows_with_component_mask_rate AS rows_with_component_mask_rate,
                smp.source_subject_mask_stale_state AS source_subject_mask_stale_state,
                smp.source_subject_mask_stale_reason AS source_subject_mask_stale_reason,
                smp.source_subject_mask_stale_timestamp_utc AS source_subject_mask_stale_timestamp_utc,
                smp.source_subject_mask_stale_json AS source_subject_mask_stale_json,
                CASE
                    WHEN COALESCE(smcqc.available, 0) = 1
                     AND lower(trim(COALESCE(smp.source_subject_mask_stale_state, ''))) = 'stale'
                    THEN 'stale'
                    ELSE smcqc.lifecycle_state
                END AS lifecycle_state,
                CASE
                    WHEN COALESCE(smcqc.available, 0) = 1
                     AND lower(trim(COALESCE(smp.source_subject_mask_stale_state, ''))) = 'stale'
                    THEN COALESCE(NULLIF(trim(smp.source_subject_mask_stale_reason), ''), 'source_subject_mask_stale')
                    ELSE smcqc.lifecycle_reason
                END AS lifecycle_reason,
                smcqc.quality_updated_utc AS quality_updated_utc,
                smcqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN smcqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM subject_mask_component_quality_current smcqc
            LEFT JOIN datasets d ON d.dataset_id = smcqc.dataset_id
            LEFT JOIN subject_mask_performance smp
              ON smp.dataset_id = smcqc.dataset_id
             AND smp.stage_group = smcqc.stage_group
             AND smp.run_name = smcqc.run_name;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_subject_mask_component_quality_overview;")
        cur.execute(
            """
            CREATE VIEW recording_subject_mask_component_quality_overview AS
            WITH ranked AS (
                SELECT
                    smcqo.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY smcqo.recording_id, smcqo.stage_group, smcqo.component_name
                        ORDER BY
                            COALESCE(smcqo.review_timestamp_utc, smcqo.run_created_utc, smcqo.quality_updated_utc) DESC,
                            COALESCE(smcqo.run_created_utc, '') DESC,
                            smcqo.run_name DESC,
                            smcqo.dataset_id DESC
                    ) AS _rn
                FROM subject_mask_component_quality_overview smcqo
                WHERE smcqo.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_034_dataset_context_current_view(self) -> None:
        cur = self.conn.cursor()
        self._ensure_columns(
            "datasets",
            {
                "source_layout": "TEXT",
                "source_frame_index_path": "TEXT",
                "source_recording_frame_index_path": "TEXT",
                "source_frame_index_schema": "TEXT",
            },
        )
        if self._table_exists("recordings"):
            self._ensure_columns(
                "recordings",
                {
                    "experiment_context_status": "TEXT",
                    "experiment_context_source": "TEXT",
                    "experiment_context_status_detail": "TEXT",
                    "stimulus_runs_available": "INTEGER",
                },
            )
        cur.execute("DROP VIEW IF EXISTS dataset_context_current;")
        cur.execute(
            """
            CREATE VIEW dataset_context_current AS
            WITH recording_subject_summary AS (
                SELECT
                    rso.recording_id AS recording_id,
                    COUNT(DISTINCT NULLIF(TRIM(rso.subject_id), '')) AS subject_count_recorded,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.subject_id), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.subject_id), ''))
                        ELSE NULL
                    END AS subject_id,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.dish_id), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.dish_id), ''))
                        ELSE NULL
                    END AS dish_id,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.cross_id), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.cross_id), ''))
                        ELSE NULL
                    END AS cross_id,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.genotype), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.genotype), ''))
                        ELSE NULL
                    END AS genotype,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.line_strain), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.line_strain), ''))
                        ELSE NULL
                    END AS line_strain,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.species), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.species), ''))
                        ELSE NULL
                    END AS species,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.sex), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.sex), ''))
                        ELSE NULL
                    END AS sex,
                    CASE
                        WHEN COUNT(DISTINCT rso.dpf_at_acquisition) = 1
                        THEN MIN(rso.dpf_at_acquisition)
                        ELSE NULL
                    END AS dpf_at_acquisition,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.subject_id), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.subject_id), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS subject_ids_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.dish_id), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.dish_id), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS dish_ids_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.cross_id), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.cross_id), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS cross_ids_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.genotype), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.genotype), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS genotypes_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.line_strain), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.line_strain), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS line_strains_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.species), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.species), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS species_values_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.sex), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.sex), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS sex_values_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT rso2.dpf_at_acquisition AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND rso2.dpf_at_acquisition IS NOT NULL
                            ORDER BY value
                        )
                    ) AS dpf_values_json
                FROM recording_subject_overview rso
                GROUP BY rso.recording_id
            )
            SELECT
                d.dataset_id AS dataset_id,
                d.recording_id AS recording_id,
                d.session_uuid AS session_uuid,
                d.zarr_path AS zarr_path,
                d.artifact_kind AS artifact_kind,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.source_layout AS source_layout,
                d.source_frame_index_path AS source_frame_index_path,
                d.source_recording_frame_index_path AS source_recording_frame_index_path,
                d.source_frame_index_schema AS source_frame_index_schema,
                d.status AS dataset_status,
                d.last_seen_utc AS last_seen_utc,
                r.recording_name AS recording_name,
                r.recording_path AS recording_path,
                r.started_utc AS recording_started_utc,
                r.recording_type AS recording_type,
                r.recording_subtype AS recording_subtype,
                r.behavior_mode AS behavior_mode,
                r.artifact_schema_id AS artifact_schema_id,
                r.experiment_context_status AS experiment_context_status,
                r.experiment_context_source AS experiment_context_source,
                r.experiment_context_status_detail AS experiment_context_status_detail,
                r.stimulus_runs_available AS stimulus_runs_available,
                COALESCE(NULLIF(TRIM(r.rig_id), ''), NULLIF(TRIM(p.rig_id), '')) AS rig_id,
                COALESCE(NULLIF(TRIM(r.arena_id), ''), NULLIF(TRIM(p.arena_id), '')) AS arena_id,
                COALESCE(NULLIF(TRIM(r.camera_id), ''), NULLIF(TRIM(p.camera_id), '')) AS camera_id,
                COALESCE(NULLIF(TRIM(r.canvas_name), ''), NULLIF(TRIM(p.canvas_name), '')) AS canvas_name,
                COALESCE(NULLIF(TRIM(r.protocol_name), ''), NULLIF(TRIM(p.protocol_name), '')) AS protocol_name,
                COALESCE(NULLIF(TRIM(r.dish_design), ''), NULLIF(TRIM(p.dish_design), '')) AS dish_design,
                p.protocol_hash AS protocol_hash,
                p.snapshot_status AS snapshot_status,
                p.snapshot_missing_json AS snapshot_missing_json,
                p.fps AS fps,
                p.video_codec AS video_codec,
                p.video_pix_fmt AS video_pix_fmt,
                p.compression_name AS compression_name,
                p.compression_level AS compression_level,
                p.exposure AS exposure,
                p.exposure_unit AS exposure_unit,
                p.gain AS gain,
                p.frame_rate AS frame_rate,
                p.camera_model AS camera_model,
                p.camera_serial AS camera_serial,
                p.has_images_ds AS has_images_ds,
                p.has_images_ds_rgb AS has_images_ds_rgb,
                p.downsample_formats_json AS downsample_formats_json,
                p.subject_count AS subject_count_snapshot,
                rss.subject_count_recorded AS subject_count_recorded,
                COALESCE(rss.subject_count_recorded, p.subject_count) AS subject_count_effective,
                CASE
                    WHEN rss.recording_id IS NOT NULL THEN 'normalized'
                    WHEN (
                        NULLIF(TRIM(p.fish_id), '') IS NOT NULL
                        OR NULLIF(TRIM(p.dish_id), '') IS NOT NULL
                        OR NULLIF(TRIM(p.cross_id), '') IS NOT NULL
                        OR NULLIF(TRIM(p.genotype), '') IS NOT NULL
                        OR p.dpf_at_acquisition IS NOT NULL
                        OR p.subject_count IS NOT NULL
                    ) THEN 'legacy_provenance'
                    ELSE 'missing'
                END AS subject_context_source,
                NULLIF(TRIM(p.fish_id), '') AS legacy_fish_id,
                NULLIF(TRIM(p.dish_id), '') AS legacy_dish_id,
                NULLIF(TRIM(p.cross_id), '') AS legacy_cross_id,
                NULLIF(TRIM(p.genotype), '') AS legacy_genotype,
                NULLIF(TRIM(p.line_strain), '') AS legacy_line_strain,
                NULLIF(TRIM(p.species), '') AS legacy_species,
                NULLIF(TRIM(p.sex), '') AS legacy_sex,
                p.dpf_at_acquisition AS legacy_dpf_at_acquisition,
                rss.subject_id AS subject_id,
                rss.dish_id AS dish_id,
                rss.cross_id AS cross_id,
                rss.genotype AS genotype,
                rss.line_strain AS line_strain,
                rss.species AS species,
                rss.sex AS sex,
                rss.dpf_at_acquisition AS dpf_at_acquisition,
                rss.subject_ids_json AS subject_ids_json,
                rss.dish_ids_json AS dish_ids_json,
                rss.cross_ids_json AS cross_ids_json,
                rss.genotypes_json AS genotypes_json,
                rss.line_strains_json AS line_strains_json,
                rss.species_values_json AS species_values_json,
                rss.sex_values_json AS sex_values_json,
                rss.dpf_values_json AS dpf_values_json
            FROM datasets d
            LEFT JOIN recordings r ON r.recording_id = d.recording_id
            LEFT JOIN provenance p ON p.dataset_id = d.dataset_id
            LEFT JOIN recording_subject_summary rss ON rss.recording_id = d.recording_id;
            """
        )

    def _migration_035_recording_step_status_latest_dataset_context_current(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS recording_step_status_latest;")
        cur.execute(
            """
            CREATE VIEW recording_step_status_latest AS
            SELECT
                COALESCE(NULLIF(trim(rss.recording_id), ''), dcc.recording_id) AS recording_id,
                rss.dataset_id,
                dcc.session_uuid AS session_uuid,
                dcc.zarr_path AS zarr_path,
                dcc.zarr_use AS zarr_use,
                dcc.artifact_kind AS artifact_kind,
                dcc.dataset_status AS dataset_status,
                dcc.rig_id AS rig_id,
                dcc.arena_id AS arena_id,
                dcc.camera_id AS camera_id,
                dcc.canvas_name AS canvas_name,
                dcc.dish_design AS dish_design,
                dcc.protocol_name AS protocol_name,
                dcc.cross_id AS cross_id,
                dcc.genotype AS genotype,
                dcc.dpf_at_acquisition AS dpf_at_acquisition,
                rss.step_name,
                rss.status,
                rss.run_name,
                rss.method,
                rss.coverage_pct,
                rss.review_status_json,
                rss.details_json,
                rss.source,
                rss.zarr_mtime_ns,
                rss.updated_utc
            FROM recording_step_status rss
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = rss.dataset_id;
            """
        )

    def _migration_036_subject_mask_component_latest_views(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_latest;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_latest AS
            WITH latest_raw AS (
                SELECT
                    dataset_id,
                    run_name AS latest_subject_mask_run
                FROM subject_mask_performance_latest
                WHERE stage_group = 'subject_mask_runs'
            ),
            ranked AS (
                SELECT
                    smcqo.*,
                    CASE
                        WHEN smcqo.stage_group = 'refined_subject_masks_runs'
                         AND COALESCE(smcqo.source_subject_mask_run, '') <> ''
                         AND COALESCE(smcqo.source_subject_mask_run, '') = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 3
                        WHEN smcqo.stage_group = 'subject_mask_runs'
                         AND smcqo.run_name = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 2
                        ELSE 1
                    END AS freshness_rank,
                    CASE
                        WHEN smcqo.stage_group = 'refined_subject_masks_runs' THEN 1
                        ELSE 0
                    END AS stage_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY smcqo.dataset_id, smcqo.component_name
                        ORDER BY
                            CASE
                                WHEN smcqo.stage_group = 'refined_subject_masks_runs'
                                 AND COALESCE(smcqo.source_subject_mask_run, '') <> ''
                                 AND COALESCE(smcqo.source_subject_mask_run, '') = COALESCE(lr.latest_subject_mask_run, '')
                                THEN 3
                                WHEN smcqo.stage_group = 'subject_mask_runs'
                                 AND smcqo.run_name = COALESCE(lr.latest_subject_mask_run, '')
                                THEN 2
                                ELSE 1
                            END DESC,
                            COALESCE(smcqo.review_timestamp_utc, smcqo.run_created_utc, smcqo.quality_updated_utc) DESC,
                            CASE
                                WHEN smcqo.stage_group = 'refined_subject_masks_runs' THEN 1
                                ELSE 0
                            END DESC,
                            COALESCE(smcqo.run_created_utc, '') DESC,
                            smcqo.run_name DESC
                    ) AS _rn
                FROM subject_mask_component_quality_overview smcqo
                LEFT JOIN latest_raw lr ON lr.dataset_id = smcqo.dataset_id
            )
            SELECT
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                recording_id,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_037_subject_mask_component_eye_compat_latest_views(self) -> None:
        self._migration_036_subject_mask_component_latest_views()
        cur = self.conn.cursor()

        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_latest;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_latest AS
            WITH latest_raw AS (
                SELECT
                    dataset_id,
                    run_name AS latest_subject_mask_run
                FROM subject_mask_performance_latest
                WHERE stage_group = 'subject_mask_runs'
            ),
            eye_components AS (
                SELECT 'eye_left' AS component_name
                UNION ALL
                SELECT 'eye_right' AS component_name
            ),
            candidate_rows AS (
                SELECT
                    dataset_id,
                    zarr_path,
                    zarr_origin,
                    zarr_use,
                    zarr_purpose,
                    artifact_kind,
                    dataset_status,
                    stage_group,
                    run_name,
                    component_name,
                    component_family,
                    run_created_utc,
                    recording_id,
                    subject_mask_method,
                    label_schema_id,
                    eye_component_mode,
                    source_subject_mask_run,
                    source_subject_mask_stale_state,
                    source_subject_mask_stale_reason,
                    source_subject_mask_stale_timestamp_utc,
                    source_subject_mask_stale_json,
                    available,
                    review_state,
                    review_method,
                    review_intended_use,
                    review_reviewer,
                    review_timestamp_utc,
                    total_rois,
                    rows_with_component_mask,
                    rows_with_component_mask_rate,
                    lifecycle_state,
                    lifecycle_reason,
                    quality_updated_utc,
                    zarr_mtime_ns,
                    quality_stale
                FROM subject_mask_component_quality_overview
                UNION ALL
                SELECT
                    empl.dataset_id AS dataset_id,
                    d.zarr_path AS zarr_path,
                    d.zarr_origin AS zarr_origin,
                    d.zarr_use AS zarr_use,
                    d.zarr_use AS zarr_purpose,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    empl.stage_group AS stage_group,
                    empl.run_name AS run_name,
                    ec.component_name AS component_name,
                    'eyes' AS component_family,
                    empl.run_created_utc AS run_created_utc,
                    empl.recording_id AS recording_id,
                    empl.method AS subject_mask_method,
                    'subject_v1_lr' AS label_schema_id,
                    'lr' AS eye_component_mode,
                    NULL AS source_subject_mask_run,
                    NULL AS source_subject_mask_stale_state,
                    NULL AS source_subject_mask_stale_reason,
                    NULL AS source_subject_mask_stale_timestamp_utc,
                    NULL AS source_subject_mask_stale_json,
                    1 AS available,
                    empl.review_state AS review_state,
                    empl.review_method AS review_method,
                    empl.review_intended_use AS review_intended_use,
                    empl.review_reviewer AS review_reviewer,
                    empl.review_timestamp_utc AS review_timestamp_utc,
                    empl.total_rois AS total_rois,
                    empl.successful_roi_pairs AS rows_with_component_mask,
                    empl.successful_roi_pair_rate AS rows_with_component_mask_rate,
                    empl.lifecycle_state AS lifecycle_state,
                    empl.lifecycle_reason AS lifecycle_reason,
                    empl.updated_utc AS quality_updated_utc,
                    empl.zarr_mtime_ns AS zarr_mtime_ns,
                    CASE
                        WHEN empl.zarr_mtime_ns IS NULL THEN 1
                        ELSE 0
                    END AS quality_stale
                FROM eye_mask_performance_latest empl
                CROSS JOIN eye_components ec
                LEFT JOIN datasets d ON d.dataset_id = empl.dataset_id
            ),
            scored AS (
                SELECT
                    cr.*,
                    CASE
                        WHEN cr.stage_group = 'refined_subject_masks_runs'
                         AND COALESCE(cr.source_subject_mask_run, '') <> ''
                         AND COALESCE(cr.source_subject_mask_run, '') = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 3
                        WHEN cr.stage_group = 'subject_mask_runs'
                         AND cr.run_name = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 2
                        ELSE 1
                    END AS subject_mask_freshness_rank,
                    CASE
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'refined_subject_masks_runs'
                        THEN 5
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'refined_eye_masks_runs'
                        THEN 4
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'subject_mask_runs'
                        THEN 3
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'eye_masks_runs'
                        THEN 2
                        ELSE 1
                    END AS eye_component_rank,
                    CASE
                        WHEN cr.stage_group = 'refined_subject_masks_runs' THEN 3
                        WHEN cr.stage_group = 'subject_mask_runs' THEN 2
                        ELSE 1
                    END AS subject_component_rank,
                    CASE
                        WHEN cr.stage_group IN ('refined_subject_masks_runs', 'refined_eye_masks_runs') THEN 1
                        ELSE 0
                    END AS refined_stage_rank
                FROM candidate_rows cr
                LEFT JOIN latest_raw lr ON lr.dataset_id = cr.dataset_id
            ),
            ranked AS (
                SELECT
                    s.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dataset_id, s.component_name
                        ORDER BY
                            CASE WHEN COALESCE(s.available, 0) = 1 THEN 1 ELSE 0 END DESC,
                            CASE
                                WHEN s.component_name IN ('eye_left', 'eye_right')
                                THEN s.eye_component_rank
                                ELSE s.subject_component_rank
                            END DESC,
                            s.subject_mask_freshness_rank DESC,
                            COALESCE(s.review_timestamp_utc, s.run_created_utc, s.quality_updated_utc) DESC,
                            s.refined_stage_rank DESC,
                            COALESCE(s.run_created_utc, '') DESC,
                            s.run_name DESC
                    ) AS _rn
                FROM scored s
            )
            SELECT
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                recording_id,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_latest_by_recording;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_latest_by_recording AS
            WITH latest_raw AS (
                SELECT
                    recording_id,
                    run_name AS latest_subject_mask_run
                FROM recording_subject_mask_performance_latest
                WHERE stage_group = 'subject_mask_runs'
            ),
            eye_components AS (
                SELECT 'eye_left' AS component_name
                UNION ALL
                SELECT 'eye_right' AS component_name
            ),
            candidate_rows AS (
                SELECT
                    dataset_id,
                    zarr_path,
                    zarr_origin,
                    zarr_use,
                    zarr_purpose,
                    artifact_kind,
                    dataset_status,
                    stage_group,
                    run_name,
                    component_name,
                    component_family,
                    run_created_utc,
                    recording_id,
                    subject_mask_method,
                    label_schema_id,
                    eye_component_mode,
                    source_subject_mask_run,
                    source_subject_mask_stale_state,
                    source_subject_mask_stale_reason,
                    source_subject_mask_stale_timestamp_utc,
                    source_subject_mask_stale_json,
                    available,
                    review_state,
                    review_method,
                    review_intended_use,
                    review_reviewer,
                    review_timestamp_utc,
                    total_rois,
                    rows_with_component_mask,
                    rows_with_component_mask_rate,
                    lifecycle_state,
                    lifecycle_reason,
                    quality_updated_utc,
                    zarr_mtime_ns,
                    quality_stale
                FROM subject_mask_component_quality_overview
                UNION ALL
                SELECT
                    empl.dataset_id AS dataset_id,
                    d.zarr_path AS zarr_path,
                    d.zarr_origin AS zarr_origin,
                    d.zarr_use AS zarr_use,
                    d.zarr_use AS zarr_purpose,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    empl.stage_group AS stage_group,
                    empl.run_name AS run_name,
                    ec.component_name AS component_name,
                    'eyes' AS component_family,
                    empl.run_created_utc AS run_created_utc,
                    empl.recording_id AS recording_id,
                    empl.method AS subject_mask_method,
                    'subject_v1_lr' AS label_schema_id,
                    'lr' AS eye_component_mode,
                    NULL AS source_subject_mask_run,
                    NULL AS source_subject_mask_stale_state,
                    NULL AS source_subject_mask_stale_reason,
                    NULL AS source_subject_mask_stale_timestamp_utc,
                    NULL AS source_subject_mask_stale_json,
                    1 AS available,
                    empl.review_state AS review_state,
                    empl.review_method AS review_method,
                    empl.review_intended_use AS review_intended_use,
                    empl.review_reviewer AS review_reviewer,
                    empl.review_timestamp_utc AS review_timestamp_utc,
                    empl.total_rois AS total_rois,
                    empl.successful_roi_pairs AS rows_with_component_mask,
                    empl.successful_roi_pair_rate AS rows_with_component_mask_rate,
                    empl.lifecycle_state AS lifecycle_state,
                    empl.lifecycle_reason AS lifecycle_reason,
                    empl.updated_utc AS quality_updated_utc,
                    empl.zarr_mtime_ns AS zarr_mtime_ns,
                    CASE
                        WHEN empl.zarr_mtime_ns IS NULL THEN 1
                        ELSE 0
                    END AS quality_stale
                FROM eye_mask_performance_latest empl
                CROSS JOIN eye_components ec
                LEFT JOIN datasets d ON d.dataset_id = empl.dataset_id
            ),
            scored AS (
                SELECT
                    cr.*,
                    CASE
                        WHEN cr.stage_group = 'refined_subject_masks_runs'
                         AND COALESCE(cr.source_subject_mask_run, '') <> ''
                         AND COALESCE(cr.source_subject_mask_run, '') = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 3
                        WHEN cr.stage_group = 'subject_mask_runs'
                         AND cr.run_name = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 2
                        ELSE 1
                    END AS subject_mask_freshness_rank,
                    CASE
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'refined_subject_masks_runs'
                        THEN 5
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'refined_eye_masks_runs'
                        THEN 4
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'subject_mask_runs'
                        THEN 3
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'eye_masks_runs'
                        THEN 2
                        ELSE 1
                    END AS eye_component_rank,
                    CASE
                        WHEN cr.stage_group = 'refined_subject_masks_runs' THEN 3
                        WHEN cr.stage_group = 'subject_mask_runs' THEN 2
                        ELSE 1
                    END AS subject_component_rank,
                    CASE
                        WHEN cr.stage_group IN ('refined_subject_masks_runs', 'refined_eye_masks_runs') THEN 1
                        ELSE 0
                    END AS refined_stage_rank
                FROM candidate_rows cr
                LEFT JOIN latest_raw lr ON lr.recording_id = cr.recording_id
                WHERE cr.recording_id IS NOT NULL
            ),
            ranked AS (
                SELECT
                    s.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.recording_id, s.component_name
                        ORDER BY
                            CASE WHEN COALESCE(s.available, 0) = 1 THEN 1 ELSE 0 END DESC,
                            CASE
                                WHEN s.component_name IN ('eye_left', 'eye_right')
                                THEN s.eye_component_rank
                                ELSE s.subject_component_rank
                            END DESC,
                            s.subject_mask_freshness_rank DESC,
                            COALESCE(s.review_timestamp_utc, s.run_created_utc, s.quality_updated_utc) DESC,
                            s.refined_stage_rank DESC,
                            COALESCE(s.run_created_utc, '') DESC,
                            s.run_name DESC,
                            s.dataset_id DESC
                    ) AS _rn
                FROM scored s
            )
            SELECT
                recording_id,
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_038_subject_mask_component_partial_run_preference(self) -> None:
        """Refresh component views so partial refined runs remain visible."""
        self._migration_033_subject_mask_registry_semantics_columns()
        self._migration_037_subject_mask_component_eye_compat_latest_views()

    def _migration_039_subject_mask_component_source_stale_views(self) -> None:
        """Expose refined-source stale metadata through component registry views."""
        self._migration_038_subject_mask_component_partial_run_preference()

    def _migration_040_subject_mask_training_model_discovery(self) -> None:
        """Index subject-mask model discovery metadata and expose a focused view."""

        if not self._table_exists("training_models"):
            return
        self._ensure_training_model_discovery_columns()
        self._backfill_training_model_discovery_metadata()
        self._refresh_subject_mask_training_models_view()

    def _migration_041_analytics_manifest_registry(self) -> None:
        """Index immutable analytics collection/export manifests."""

        self._ensure_analytics_manifest_tables()

    def _migration_042_recording_experiment_context_columns(self) -> None:
        """Expose whether a recording has experiment/stimulus context."""

        if self._table_exists("recordings"):
            self._ensure_columns(
                "recordings",
                {
                    "experiment_context_status": "TEXT",
                    "experiment_context_source": "TEXT",
                    "experiment_context_status_detail": "TEXT",
                    "stimulus_runs_available": "INTEGER",
                },
            )
        self._migration_034_dataset_context_current_view()
        if self._table_exists("recordings"):
            self._migration_003_recording_columns_reconcile()

    def _migration_043_stage_catalog_recording_step_status_wide_view(self) -> None:
        """Refresh recording_step_status_wide from the canonical stage catalog."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_044_derived_analysis_recording_step_status_wide_view(self) -> None:
        """Refresh recording_step_status_wide to expose derived-analysis stages."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_045_tail_behavior_recording_step_status_wide_view(self) -> None:
        """Refresh recording_step_status_wide to expose tail/classifier stages."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_046_source_freshness_recording_step_status_wide_view(self) -> None:
        """Refresh recording_step_status_wide to display source freshness states."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_047_bout_stimulus_source_freshness_recording_step_status_wide_view(self) -> None:
        """Refresh wide status display for bout/stimulus source freshness."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_048_eye_shape_source_freshness_recording_step_status_wide_view(self) -> None:
        """Refresh wide status display for eye/shape source freshness."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_049_model_input_shape_registry(self) -> None:
        """Normalize trained-model input shape metadata and expose a shared view."""

        if not self._table_exists("training_models"):
            return
        self._ensure_training_model_input_shape_columns()
        self._backfill_training_model_input_shapes()
        self._refresh_model_input_shapes_view()

    def _migration_050_detect_quality_current_reviewed_preference(self) -> None:
        """Prefer reviewed refined-detect rows over newer unreviewed attempts."""

        if not self._table_exists("detect_quality"):
            return
        self._ensure_columns(
            "detect_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        self._refresh_detect_quality_current_view()

    def _migration_051_training_image_profile_registry(self) -> None:
        """Register image-domain training profiles for dataset-lake queries."""

        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_image_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                source_frame_array TEXT,
                profile_created_utc TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                frames_total INTEGER,
                frames_profiled INTEGER,
                mean_intensity_p50 REAL,
                contrast_p50 REAL,
                sharpness_p50 REAL,
                clip_dark_rate_mean REAL,
                clip_bright_rate_mean REAL,
                illumination_center_edge_p50 REAL,
                illumination_slope_x_p50 REAL,
                illumination_slope_y_p50 REAL,
                fish_bg_contrast_p50 REAL,
                rig_id TEXT,
                camera_id TEXT,
                arena_id TEXT,
                dish_design TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                genotype TEXT,
                dpf_at_acquisition INTEGER,
                profile_json TEXT,
                PRIMARY KEY (dataset_id, profile_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "training_image_profile",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "source_frame_array": "TEXT",
                "profile_created_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "frames_total": "INTEGER",
                "frames_profiled": "INTEGER",
                "mean_intensity_p50": "REAL",
                "contrast_p50": "REAL",
                "sharpness_p50": "REAL",
                "clip_dark_rate_mean": "REAL",
                "clip_bright_rate_mean": "REAL",
                "illumination_center_edge_p50": "REAL",
                "illumination_slope_x_p50": "REAL",
                "illumination_slope_y_p50": "REAL",
                "fish_bg_contrast_p50": "REAL",
                "rig_id": "TEXT",
                "camera_id": "TEXT",
                "arena_id": "TEXT",
                "dish_design": "TEXT",
                "canvas_name": "TEXT",
                "protocol_name": "TEXT",
                "genotype": "TEXT",
                "dpf_at_acquisition": "INTEGER",
                "profile_json": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_image_profile_recording_created "
            "ON training_image_profile(recording_id, profile_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_image_profile_scope "
            "ON training_image_profile(zarr_use, source_frame_array);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_image_profile_domain_metrics "
            "ON training_image_profile(mean_intensity_p50, contrast_p50, sharpness_p50);"
        )
        cur.execute("DROP VIEW IF EXISTS training_image_profile_latest;")
        cur.execute(
            """
            CREATE VIEW training_image_profile_latest AS
            WITH ranked AS (
                SELECT
                    tip.dataset_id AS dataset_id,
                    tip.profile_run AS profile_run,
                    dcc.recording_id AS recording_id,
                    dcc.zarr_use AS zarr_use,
                    tip.source_frame_array AS source_frame_array,
                    tip.profile_created_utc AS profile_created_utc,
                    tip.zarr_mtime_ns AS zarr_mtime_ns,
                    tip.updated_utc AS updated_utc,
                    tip.frames_total AS frames_total,
                    tip.frames_profiled AS frames_profiled,
                    tip.mean_intensity_p50 AS mean_intensity_p50,
                    tip.contrast_p50 AS contrast_p50,
                    tip.sharpness_p50 AS sharpness_p50,
                    tip.clip_dark_rate_mean AS clip_dark_rate_mean,
                    tip.clip_bright_rate_mean AS clip_bright_rate_mean,
                    tip.illumination_center_edge_p50 AS illumination_center_edge_p50,
                    tip.illumination_slope_x_p50 AS illumination_slope_x_p50,
                    tip.illumination_slope_y_p50 AS illumination_slope_y_p50,
                    tip.fish_bg_contrast_p50 AS fish_bg_contrast_p50,
                    COALESCE(dcc.rig_id, tip.rig_id) AS rig_id,
                    COALESCE(dcc.camera_id, tip.camera_id) AS camera_id,
                    COALESCE(dcc.arena_id, tip.arena_id) AS arena_id,
                    COALESCE(dcc.dish_design, tip.dish_design) AS dish_design,
                    COALESCE(dcc.canvas_name, tip.canvas_name) AS canvas_name,
                    COALESCE(dcc.protocol_name, tip.protocol_name) AS protocol_name,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.genotype
                        ELSE COALESCE(dcc.genotype, tip.genotype)
                    END AS genotype,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.dpf_at_acquisition
                        ELSE COALESCE(dcc.dpf_at_acquisition, tip.dpf_at_acquisition)
                    END AS dpf_at_acquisition,
                    tip.profile_json AS profile_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY tip.dataset_id
                        ORDER BY
                            COALESCE(tip.profile_created_utc, tip.updated_utc) DESC,
                            tip.profile_run DESC
                    ) AS _rn
                FROM training_image_profile tip
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = tip.dataset_id
            )
            SELECT
                dataset_id,
                profile_run,
                recording_id,
                zarr_use,
                source_frame_array,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                frames_total,
                frames_profiled,
                mean_intensity_p50,
                contrast_p50,
                sharpness_p50,
                clip_dark_rate_mean,
                clip_bright_rate_mean,
                illumination_center_edge_p50,
                illumination_slope_x_p50,
                illumination_slope_y_p50,
                fish_bg_contrast_p50,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_052_dataset_source_layout_metadata(self) -> None:
        """Expose training source-frame sidecars and rolling-clip layout metadata."""

        self._ensure_columns(
            "datasets",
            {
                "source_layout": "TEXT",
                "source_frame_index_path": "TEXT",
                "source_recording_frame_index_path": "TEXT",
                "source_frame_index_schema": "TEXT",
            },
        )
        self._migration_034_dataset_context_current_view()

    def _migration_053_model_deployment_artifacts(self) -> None:
        """Track hardware/runtime-specific model deployment artifacts."""

        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_deployment_artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                source_onnx_run_id TEXT,
                source_onnx_path TEXT,
                source_onnx_sha256 TEXT,
                artifact_kind TEXT NOT NULL,
                deployment_runtime TEXT NOT NULL,
                target_hardware_class TEXT,
                target_gpu_name TEXT,
                target_compute_capability TEXT,
                precision TEXT,
                engine_path TEXT,
                engine_sha256 TEXT,
                manifest_path TEXT,
                manifest_sha256 TEXT,
                status TEXT NOT NULL DEFAULT 'candidate',
                validation_summary_json TEXT,
                trtexec_path TEXT,
                trt_version TEXT,
                cuda_version TEXT,
                builder_optimization_level INTEGER,
                avg_timing INTEGER,
                profiling_verbosity TEXT,
                cuda_graph INTEGER,
                nms_conf REAL,
                nms_iou REAL,
                nms_topk INTEGER,
                metadata_json TEXT,
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY(source_onnx_run_id) REFERENCES onnx_models(run_id)
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_deployment_artifacts_run_id
            ON model_deployment_artifacts(run_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_deployment_artifacts_source_onnx
            ON model_deployment_artifacts(source_onnx_run_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_deployment_artifacts_runtime_target
            ON model_deployment_artifacts(
                deployment_runtime,
                target_hardware_class,
                status
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_deployment_artifacts_engine_sha
            ON model_deployment_artifacts(engine_sha256);
            """
        )
