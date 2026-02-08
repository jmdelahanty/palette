# Recording Registry Normalization TODO

Purpose: move to a recording-first registry model that supports queries like:
"all recordings with genotype X, DPF Y, and protocol step/task Z."

## Key Decision

- [ ] Model `recording` as first-class parent.
- [ ] Keep one row per Zarr as `dataset` (child of recording).
- [ ] Keep workflow state (review/training/export status) separate from immutable provenance.

## Target Model (Phase 1)

- [ ] `recordings` table (one row per recording/session).
  - Example fields:
    - `recording_id` (PK)
    - `session_uuid`
    - `recording_path`
    - `started_utc`
    - `rig_id`, `arena_id`, `canvas_name`
    - `status`, `created_utc`

- [ ] `datasets` stays one row per Zarr, linked to recording.
  - Add `recording_id` FK to `datasets`.
  - Continue storing `zarr_path`, `dataset_id`, and status.

- [ ] Clarify "per-recording Zarr table" role.
  - Option A: reuse `datasets` as the per-Zarr table.
  - Option B: add `recording_datasets` link table if we need many-to-many flexibility.
  - Decision target: start with Option A unless blocked.

- [ ] Store per-Zarr purpose and flags in child rows.
  - `zarr_purpose` (`analysis`, `training`, etc.)
  - artifact flags (e.g. merged export, contains crop/keypoint/detect runs)
  - quality/status flags as needed

## Subject / Dish / Cross Normalization (Phase 2)

- [ ] Add `subjects` table.
  - `subject_id`, `sex`, `species`, `genotype`, `line_strain`, `cross_id`, metadata.

- [ ] Add `recording_subjects` join table.
  - supports one or more subjects per recording.
  - includes `dpf_at_acquisition`, role/count metadata.

- [ ] Add `dishes` + `recording_dishes`.
  - support `recording -> dish_id -> subject_id(s)` lineage.

- [ ] Add `crosses` table if needed for explicit parentage modeling.

## Protocol / Camera Normalization (Phase 3)

- [ ] Add `protocol_runs` table.
  - `protocol_run_id`, `recording_id`, `protocol_name`, `protocol_hash`, timestamps.

- [ ] Add `protocol_steps` table.
  - indexed steps with names/types/timing/params.

- [ ] Add `camera_runs` table.
  - `camera_id`, model, serial, fps, exposure, gain, pixel format, metadata.

## Analysis Run Ledger (Phase 4)

- [ ] Keep artifacts as source of truth; use DB as index/ledger only.
  - Full arrays/results remain in auditable artifacts (zarr/json/report files).
  - Registry stores pointers, hashes, status, and compact query fields.

- [ ] Add `analysis_runs` table.
  - One row per processing step execution (e.g. inference, refine, behavior).
  - Example fields:
    - `analysis_run_id` (PK)
    - `recording_id` FK
    - `dataset_id` FK
    - `step_type` (`detect_infer`, `pose_infer`, `refine`, `behavior_speed`, ...)
    - `input_artifact_path`, `input_artifact_sha256`
    - `output_artifact_path`, `output_artifact_sha256`
    - `model_run_id` (optional FK to training/model registry rows)
    - `params_json`, `invocation_json`
    - `status`, `error_message`
    - `tool_version`, `git_commit`, `created_utc`, `completed_utc`

- [ ] Add `analysis_summaries` table.
  - Stores small queryable metrics by `analysis_run_id`.
  - Example fields:
    - `analysis_run_id` FK
    - `metric_name`
    - `metric_value_real` / `metric_value_text`
    - `unit`
    - `scope` (`recording`, `arena`, `subject`, etc.)
  - Intended for filters/dashboards, not bulk per-frame data.

- [ ] Add optional convenience view: `recording_analysis_overview`.
  - Shows latest successful run per step type per recording.
  - Includes output paths + top summary metrics.

## Trial / Segment Indexing (Phase 5)

- [ ] Add `trials` (or `trial_segments`) table for experiment episode windows.
  - One row per trial/segment so queries do not require full dataset re-parse.
  - Example fields:
    - `trial_id` (PK)
    - `recording_id` FK
    - `dataset_id` FK
    - `analysis_run_id` FK (which pipeline step produced this trial)
    - `session_type` (`chaser`, `feeding`, etc.)
    - `start_frame`, `end_frame`, `start_utc`, `end_utc`
    - `outcome` (`escaped`, `captured`, `timeout`, ...)
    - `outcome_confidence`
    - `trial_artifact_path`, `trial_artifact_sha256`

- [ ] Add `trial_metrics` table for compact per-trial summaries.
  - Example metrics:
    - `latency_to_escape`
    - `max_speed`
    - `distance_to_chaser_min`
  - Keep this table small and query-oriented; store dense time-series in artifacts.

- [ ] Add query views for trial-level exploration.
  - Example target query:
    - "all trials from chaser sessions where the fish escaped."

## Individual Identity Across Recordings (Phase 6)

- [ ] Add `individuals` table (one row per fish across all recordings).
  - `individual_id` (PK, stable identity)
  - optional lineage/sex/species/genotype links
  - identity confidence/notes metadata

- [ ] Add `recording_individuals` join table.
  - Links which individuals are present in each recording.
  - Supports repeated individuals across many recordings.

- [ ] Link trials to individuals.
  - Minimum: `trials.individual_id` for single-subject trials.
  - Preferred flexible model: `trial_individuals(trial_id, individual_id, role)`.
    - role examples: `chaser`, `target`, `bystander`.

- [ ] Add cross-recording individual queries.
  - Examples:
    - all escaped trials for `individual_id = X`
    - all target-role trials with outcome `escaped` in chaser sessions

## Cross / Dish Integration (Phase 7)

- [ ] Integrate canonical `crosses` from external DB.
  - store `cross_id` plus optional `external_cross_id`
  - sync key fields (line, genotype, parentage summary, status)

- [ ] Model `dishes` as children of `crosses`.
  - `dishes.cross_id -> crosses.cross_id`
  - preserve dish metadata used during acquisition.

- [ ] Link `individuals` to dish/cross lineage.
  - minimum: `individuals.dish_id` (implies cross through dish)
  - optional denormalized `individuals.cross_id` for convenience/indexing

- [ ] Keep synchronization strategy explicit.
  - source-of-truth DB for crosses/dishes
  - periodic sync job or on-demand import into palette registry
  - conflict handling/audit fields (`source_updated_utc`, `synced_utc`)

## Compatibility + Write Path

- [ ] Keep current `provenance` table during migration as denormalized cache.
- [ ] Update `Registry.register_from_root(...)` to:
  - resolve/create recording
  - link dataset->recording
  - upsert subjects/dish/protocol/camera entities
  - continue writing compatibility fields

- [ ] Ensure idempotent upserts with stable keys + unique constraints.

- [ ] Update processing tools (inference/refine/behavior) to register runs.
  - Record run start (`in_progress`) and completion (`success`/`failed`).
  - Attach output artifact paths and hashes.
  - Write compact summaries for query use.

## Query UX

- [ ] Add `recording_overview` view for common filters.
- [ ] Add filters for:
  - genotype
  - DPF range
  - protocol step/task
  - subject count
  - camera/rig/arena
  - analysis step completion/status
  - behavior summary thresholds (e.g. mean speed)
  - trial outcomes + trial metric thresholds
  - individual-level history across recordings

## Example SQL Templates

- [ ] Query: all fish from a cross in a given experiment at a specific DPF.

```sql
SELECT
  i.individual_id,
  r.recording_id,
  rs.dpf_at_acquisition,
  pr.protocol_name
FROM individuals i
JOIN dishes d ON d.dish_id = i.dish_id
JOIN crosses c ON c.cross_id = d.cross_id
JOIN recording_individuals rs ON rs.individual_id = i.individual_id
JOIN recordings r ON r.recording_id = rs.recording_id
LEFT JOIN protocol_runs pr ON pr.recording_id = r.recording_id
WHERE c.cross_id = :cross_id
  AND rs.dpf_at_acquisition = :dpf
  AND pr.protocol_name = :experiment_name;
```

- [ ] Query: individuals from a cross with the same experiment at 5, 6, and 7 dpf.

```sql
SELECT
  i.individual_id
FROM individuals i
JOIN dishes d ON d.dish_id = i.dish_id
JOIN crosses c ON c.cross_id = d.cross_id
JOIN recording_individuals rs ON rs.individual_id = i.individual_id
JOIN recordings r ON r.recording_id = rs.recording_id
JOIN protocol_runs pr ON pr.recording_id = r.recording_id
WHERE c.cross_id = :cross_id
  AND pr.protocol_name = :experiment_name
  AND rs.dpf_at_acquisition IN (5, 6, 7)
GROUP BY i.individual_id
HAVING COUNT(DISTINCT rs.dpf_at_acquisition) = 3;
```

- [ ] Query: escaped trials in chaser sessions for a cross at selected DPFs.

```sql
SELECT
  t.trial_id,
  i.individual_id,
  rs.dpf_at_acquisition,
  t.outcome
FROM trials t
JOIN trial_individuals ti ON ti.trial_id = t.trial_id
JOIN individuals i ON i.individual_id = ti.individual_id
JOIN dishes d ON d.dish_id = i.dish_id
JOIN crosses c ON c.cross_id = d.cross_id
JOIN recordings r ON r.recording_id = t.recording_id
JOIN recording_individuals rs
  ON rs.recording_id = r.recording_id
 AND rs.individual_id = i.individual_id
WHERE c.cross_id = :cross_id
  AND t.session_type = 'chaser'
  AND t.outcome = 'escaped'
  AND rs.dpf_at_acquisition IN (5, 6, 7);
```

## Backfill + Validation

- [ ] Add maintenance command:
  - `--backfill-recording-entities`
  - `--dry-run`
  - inserted/updated/skipped/error counters

- [ ] Backfill existing `/nvme1/palette_registry.sqlite`.

- [ ] Integrity checks:
  - every dataset linked to one recording
  - subject/protocol links consistent
  - key query views return expected rows

## Open Questions

- [ ] Is `recording_id` derived from session UUID, path hash, or both?
- [ ] Do we need many-to-many recording<->dataset, or one recording owns each dataset?
- [ ] Which fields are immutable provenance vs mutable workflow annotations?
- [ ] Do we define an explicit controlled vocabulary for `zarr_purpose` + flags?
- [ ] Do we keep one summary table (`analysis_summaries`) or typed summary tables per domain?
- [ ] How do we assign/validate stable `individual_id` when identity is uncertain?
- [ ] Should cross/dish sync be pull-based (scheduled) or push-based (event-driven)?
