-- Schema (DDL only, no rows) of a long-lived labeling store as found on
-- 2026-09-29: it reports schema_version 9 but its receipts table predates
-- released_checkpoint_count. Opening it must upgrade every table to the
-- fresh CREATE shape (test_labeling_store_schema_upgrade.py).
CREATE TABLE labeling_schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
CREATE TABLE recording_assignments (
                recording_id TEXT PRIMARY KEY,
                assignee_user TEXT NOT NULL,
                assigned_by TEXT,
                assigned_at_utc TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                notes TEXT
            );
CREATE TABLE labeling_sessions (
                session_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                user TEXT NOT NULL,
                workflow_kind TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                expires_at_utc TEXT NOT NULL,
                last_seen_at_utc TEXT,
                closed_at_utc TEXT,
                client_label TEXT,
                FOREIGN KEY(task_id) REFERENCES labeling_tasks(task_id) ON DELETE CASCADE
            );
CREATE TABLE labeling_task_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                user TEXT NOT NULL,
                event_type TEXT NOT NULL,
                target_json TEXT,
                before_json TEXT,
                after_json TEXT,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES labeling_tasks(task_id) ON DELETE CASCADE
            );
CREATE TABLE labeling_assignment_events (
                event_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                actor_user TEXT,
                event_type TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT,
                created_at_utc TEXT NOT NULL
            );
CREATE TABLE labeling_users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT,
                email TEXT,
                role TEXT NOT NULL DEFAULT 'labeler',
                status TEXT NOT NULL DEFAULT 'active',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                notes TEXT
            );
CREATE TABLE labeling_user_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                actor_user TEXT,
                event_type TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT,
                created_at_utc TEXT NOT NULL
            );
CREATE TABLE labeling_task_definition_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                actor_user TEXT,
                event_type TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT,
                created_at_utc TEXT NOT NULL
            );
CREATE TABLE labeling_admin_reviews (
                review_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE,
                recording_id TEXT NOT NULL,
                reviewer_user TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                notes TEXT,
                correction_event_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES labeling_tasks(task_id) ON DELETE CASCADE
            );
CREATE TABLE "labeling_tasks" (
                task_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                workflow_kind TEXT NOT NULL,
                dataset_id TEXT,
                zarr_use TEXT,
                stage_group TEXT,
                run_name TEXT,
                component_name TEXT,
                title TEXT,
                scope_json TEXT,
                state TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
    CHECK (state IN ('pending', 'in_progress', 'blocked', 'complete', 'superseded'))
);
CREATE TABLE "labeling_session_checkpoints" (
                checkpoint_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                user TEXT NOT NULL,
                workflow_kind TEXT NOT NULL,
                target_run_path TEXT NOT NULL,
                target_edit_revision INTEGER NOT NULL DEFAULT 0,
                source_rowset_path TEXT,
                roi_idx INTEGER NOT NULL,
                component_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                metadata_json TEXT,
                state TEXT NOT NULL DEFAULT 'active',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                applied_at_utc TEXT,
                apply_id TEXT,
                edit_revision_before INTEGER,
                edit_revision_after INTEGER, snapshot_row_sha256 TEXT,
                UNIQUE(task_id, roi_idx, component_name),
                FOREIGN KEY(task_id) REFERENCES labeling_tasks(task_id) ON DELETE CASCADE,
                FOREIGN KEY(session_id) REFERENCES labeling_sessions(session_id) ON DELETE CASCADE,
    CHECK (state IN ('active', 'applying', 'applied', 'discarded'))
);
CREATE TABLE "labeling_checkpoint_apply_receipts" (
                apply_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                component_name TEXT NOT NULL,
                state TEXT NOT NULL,
                checkpoint_count INTEGER NOT NULL,
                checkpoints_json TEXT NOT NULL,
                claimed_at_utc TEXT NOT NULL,
                applied_at_utc TEXT,
                edit_revision_before INTEGER,
                edit_revision_after INTEGER, secondary_effects_state TEXT NOT NULL DEFAULT 'complete', secondary_effects_completed_at_utc TEXT, checkpoint_snapshot_sha256 TEXT,
                FOREIGN KEY(task_id) REFERENCES labeling_tasks(task_id) ON DELETE CASCADE,
    CHECK (state IN ('applying', 'applied')),
    CHECK (secondary_effects_state IN ('not_ready', 'pending', 'complete'))
);
CREATE INDEX idx_labeling_assignments_assignee
                ON recording_assignments(assignee_user, status);
CREATE INDEX idx_labeling_sessions_task
                ON labeling_sessions(task_id, user, closed_at_utc, expires_at_utc);
CREATE INDEX idx_labeling_events_task
                ON labeling_task_events(task_id, created_at_utc);
CREATE INDEX idx_labeling_assignment_events_recording
                ON labeling_assignment_events(recording_id, created_at_utc);
CREATE INDEX idx_labeling_users_status
                ON labeling_users(status, role, user_id);
CREATE INDEX idx_labeling_user_events_user
                ON labeling_user_events(user_id, created_at_utc);
CREATE INDEX idx_labeling_task_definition_events_task
                ON labeling_task_definition_events(task_id, created_at_utc);
CREATE INDEX idx_labeling_admin_reviews_state
                ON labeling_admin_reviews(state, updated_at_utc);
CREATE INDEX idx_labeling_admin_reviews_recording
                ON labeling_admin_reviews(recording_id, state);
CREATE INDEX idx_labeling_tasks_recording
                ON labeling_tasks(recording_id, state, priority DESC);
CREATE INDEX idx_labeling_session_checkpoints_task_state
                ON labeling_session_checkpoints(task_id, state, updated_at_utc);
CREATE INDEX idx_labeling_session_checkpoints_apply
                ON labeling_session_checkpoints(task_id, apply_id, state);
CREATE INDEX idx_labeling_session_checkpoints_snapshot_order
            ON labeling_session_checkpoints(
                task_id, component_name, state, updated_at_utc, roi_idx,
                checkpoint_id, apply_id, snapshot_row_sha256
            );
CREATE INDEX idx_labeling_checkpoint_apply_receipts_task
                ON labeling_checkpoint_apply_receipts(task_id, state, applied_at_utc);
CREATE INDEX idx_labeling_checkpoint_apply_effects_pending
            ON labeling_checkpoint_apply_receipts(
                task_id, component_name, state, secondary_effects_state,
                applied_at_utc, apply_id
            );
