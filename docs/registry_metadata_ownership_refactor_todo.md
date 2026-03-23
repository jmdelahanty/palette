# Registry Metadata Ownership Refactor TODO

Purpose: implement the metadata ownership refactor described in
`docs/registry_metadata_ownership_refactor_design.md`.

Date anchored: 2026-03-13.

## Decision Snapshot

- [x] Keep cross-recording `subject_id` identity as a first-class requirement.
- [x] Keep exact stage derivation provenance as a first-class requirement.
- [x] Reduce canonical owners to one owner per metadata family.
- [x] Treat registry profile/quality/performance tables as derived query caches,
  not canonical context owners.
- [x] Prefer additive views + reader migration before destructive schema
  cleanup.

## Phase 0: Contract Lock

- [ ] Finalize the canonical ownership map for:
  - dataset identity,
  - recording context,
  - biological identity,
  - dataset-local technical snapshot,
  - run derivation provenance,
  - operational stage status.
- [ ] Decide which legacy `provenance` fields become:
  - compatibility-only,
  - read-only snapshot fields,
  - eventual drop candidates.
- [ ] Decide whether `protocol_hash` stays canonical in `provenance` or moves
  to `recordings`.
- [ ] Decide whether `subject_count` remains a compatibility snapshot field in
  `provenance` or is fully replaced by derived subject-count views.
- [ ] Document the canonical meaning of `subject_id` vs legacy `fish_id`.

## Phase 1: Additive Context Views

- [x] Add `dataset_context_current` as a one-row-per-dataset view that joins:
  - `datasets`,
  - `recordings`,
  - `provenance`,
  - subject-lineage summary derived from `recording_subject_overview`.
- [x] Make `dataset_context_current` expose both:
  - scalar convenience biology fields when unambiguous,
  - aggregate JSON/list fields for multi-subject recordings.
- [x] Add tests for:
  - single-subject recordings,
  - multi-subject recordings,
  - missing-subject-lineage cases,
  - compatibility fallback behavior.
- [x] Add schema reference documentation for the new context view.

## Phase 2: Reader Migration

- [x] Update registry query paths to read dataset/recording context from
  `dataset_context_current` instead of directly from duplicated `provenance`
  fields.
- [x] Update `recording_step_status_latest` to use `dataset_context_current` for
  recording and lineage context.
- [x] Update `recording_step_status_wide` to inherit context from the same
  canonical read surface.
- [x] Update `query_datasets()` and `registry_query.py` to use the new view for
  scalar context fields.
- [ ] Preserve subject-lineage-specific filters
  (`cross_id`, `genotype`, `dpf`, etc.) through either:
  - `recording_subject_overview`, or
  - aggregate columns + explicit lineage joins when needed.
- [x] Update `check_training_registry` and any status-page query helpers to read
  from the new context view.

## Phase 3: Recording Context Writer Migration

- [ ] Stop treating these `provenance` fields as canonical recording context:
  - `rig_id`
  - `arena_id`
  - `camera_id`
  - `canvas_name`
  - `protocol_name`
  - `dish_design`
- [x] Update `upsert_provenance()` write policy so these values are either:
  - no longer written for new rows, or
  - written only as compatibility snapshots during migration.
- [ ] Update recording backfill/import flows so `recordings` remains the
  canonical write target for recording context.
- [x] Add tests proving source-recording context remains stable when
  `provenance` is missing or legacy-populated.

## Phase 4: Biological Identity Migration

- [ ] Formalize `subject_id` as the canonical biological identity key.
- [ ] Treat legacy `fish_id` as:
  - compatibility import field,
  - query alias where needed,
  - non-canonical long-term.
- [ ] Update maintenance/backfill code so normalized lineage is the primary
  source of truth for:
  - `dish_id`
  - `cross_id`
  - `genotype`
  - `line_strain`
  - `species`
  - `sex`
  - `dpf_at_acquisition`
- [ ] Stop treating denormalized copies of those fields in `provenance` as
  co-equal canonical query fields.
- [ ] Add migration/backfill tests for:
  - same subject across multiple recordings,
  - one recording with multiple subjects,
  - conflicting legacy `fish_id` / `subject_id` inputs.

## Phase 5: Profile / Projection Cleanup

- [ ] Stop adding repeated context columns as authoritative writers in:
  - `detection_data_profile`
  - `keypoint_data_profile`
  - `eye_mask_data_profile`
  - related latest/current/overview views
- [ ] Preserve query-critical lineage columns in projection tables, including:
  - `source_detect_run`
  - `source_refined_run`
  - `source_crop_run`
  - `source_keypoint_run`
  - `source_eye_mask_run`
- [ ] Move repeated context reads in profile latest/current views to joins
  against `dataset_context_current`.
- [ ] Mark repeated context columns in profile tables as deprecated in docs once
  readers stop depending on them.

## Phase 6: Provenance / Lineage Contract Tightening

- [ ] Audit stage writers to ensure canonical run provenance remains the source
  of truth for derivation lineage.
- [ ] Ensure registry projections copy only query-critical lineage from run
  provenance.
- [ ] Add explicit tests for end-to-end lineage questions such as:
  - subject mask run -> keypoint run,
  - keypoint run -> crop run,
  - crop run -> detect run,
  - detect run -> source recording dataset.
- [ ] Document which lineage pointers are allowed in registry projections and
  which must remain on-disk only.

## Phase 7: Integrity Simplification

- [ ] Remove or downgrade integrity checks whose only purpose is defending
  duplicate canonical ownership.
- [ ] Replace them with checks for:
  - missing canonical owner rows,
  - broken FK-like lineage links,
  - ambiguity bugs in `dataset_context_current`,
  - invalid subject/dish/cross lineage joins.
- [ ] Keep integrity checks for:
  - missing `subjects`,
  - missing `dishes`,
  - broken `recording_subjects`,
  - invalid source-run lineage projections.

## Phase 8: Deprecation and Cleanup

- [ ] Add a deprecated-field list to the schema reference and governance docs.
- [ ] Stop writing deprecated duplicate fields once all readers are migrated.
- [ ] Add migration notes for any removed columns or views.
- [ ] Remove dead mismatch checks and compatibility paths only after operator
  tooling confirms zero remaining dependencies.

## Tests / Validation Work

- [x] Add targeted unit tests for `dataset_context_current`.
- [ ] Add regression tests for `registry_query` subject filters under:
  - single-subject,
  - same-subject-across-recordings,
  - multi-subject mixed-genotype recordings.
- [x] Add integrity tests for legacy duplicate-field compatibility during the
  transition window.
- [ ] Add schema-reference freshness checks if new views or deprecated fields
  are introduced.

## Suggested First Implementation Slice

- [x] Implement `dataset_context_current`.
- [x] Migrate `recording_step_status_latest` to read from it.
- [x] Migrate `query_datasets()` / `registry_query.py` to read recording context
  from it.
- [x] Freeze new dual-write expansion in `upsert_provenance()`.
- [ ] Add docs for deprecated duplicated ownership fields.

## Exit Criteria

- [ ] Cross-recording subject identity remains fully supported.
- [ ] Stage derivation provenance remains exact and queryable.
- [ ] Recording context has one canonical registry owner.
- [ ] Biological identity has one canonical normalized owner family.
- [ ] Dataset-local technical snapshot has one canonical owner.
- [ ] Profile/quality/performance tables are clearly derived projections rather
  than shadow canonical stores.
- [ ] Registry integrity logic is simpler and less mismatch-driven than the
  current design.
