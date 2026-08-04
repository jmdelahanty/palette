# Instance, Track, And Subject Identity Contract

<!-- contract-meta
status: active
created: 2026-07-09
owner: jeremy
supersedes-current-state-sections: docs/instance_identity_and_tracking_review.md,
  docs/refined_detect_multisubject_goal.md,
  docs/track_identity_target_architecture.md
-->

## Purpose

This is the authoritative identity contract for Palette observation rows,
tracking, and eventual cross-recording subject assignment. Older design reviews
remain useful history, but their current-state claims do not override this
contract.

Palette borrows SLEAP's useful conceptual separation between instances and
tracks while retaining recording Zarrs as the canonical provenance unit. A
mutable multi-recording project file is not the source of truth.

## Identity Namespaces

| Identifier | Scope | Meaning |
| --- | --- | --- |
| `instance_key` | one observation | Content-derived detection-observation identity, minted at detect/manual-add origin and copied downstream. It is not an animal identity. |
| `refined_row_id` | one curated detect run | Stable logical identity for one mutable curated instance row. It is not physical row position. |
| `track_id` | one `tracking_runs/<run>` | Run-local temporal trajectory identity. A bare `track_id` is never globally unique. |
| `track_sample_key` | one track-kinematics row | Exact pair `(track_id, acquisition_frame_index)` identifying one sample of a run-local trajectory. It is not an observation key or a biological identity. |
| `source_instance_key` | track-sample lineage | Nullable reference to the exact immediate-source observation. It is derived mechanically when the source row identity is `instance_key`; it is never the track row's primary identity. |
| `stimulus_state_key` | one stimulus state row | Producer-defined compound state identity preserved by stimulus import and online refinement. It is not a camera-frame or observation identity. |
| `subject_id` | registry/global biology | Optional known biological identity. Tracking alone must not silently infer it. |

The globally addressable forms are:

- observation: `(recording_id, instance_key)`;
- curated row: `(recording_id, refined_detect_run, refined_row_id)`;
- trajectory: `(recording_id, tracking_run, track_id)`;
- track sample: `(recording_id, track_kinematics_run, track_id,
  acquisition_frame_index)`;
- biological identity: `subject_id`.

Coordinate-frame, transform, calibration, and provenance records have their own
digest-bound record identities. Those records describe authorities; they are not
row keys and must not be substituted for any identity above.

## Canonical Stage Separation

```text
detect/refined detect instances     observation correctness and authoring
arena_assignment_runs               spatial containment
tracking_runs                       temporal association
subject_identity_runs (future)      reviewed track-segment -> subject linkage
track_kinematics_runs               behavior derived from one exact tracking run
```

Detection rows and component channels must not carry implicit temporal
identity. Arena IDs are spatial labels, not subjects. Tracking corrections
modify a tracking authoring revision; they do not rewrite observations.

## Modern Tracking Row Contract

Every new keyed `tracking_runs/<run>` writes these row-aligned arrays:

- `track_ids`, `arena_ids`, `frame_indices`, and `source_row_indices`;
- `instance_key`;
- `source_refined_row_ids` when refined lineage exists;
- `source_detect_row_index` when raw-detect lineage exists.

Optional method-neutral outputs are `tracking_confidence`, `tracking_status`,
and `association_cost`. Every method also writes `track_ids_present` and
`track_arena_ids` on the track axis.

New keyed consumers join by `instance_key`. Physical row equality is a storage
fast path, not the identity contract. Legacy runs without keys remain readable
through an explicitly labeled `legacy_positional` mode, but a keyed source must
not silently consume a legacy tracking run.

That rule is specific to observation-aligned assignment rows in
`tracking_runs`. It does not make `instance_key` the primary identity of a
derived trajectory sample. Every new `track_kinematics_runs/<run>/tracks/id_*`
rowset uses `track_sample_key = (track_id, acquisition_frame_index)` as its
primary row identity. Canonical temporal-lineage version 1 also persists:

- a unique `source_row_index` selecting the exact immediate-source row;
- `source_acquisition_frame_index`, exactly equal to both the selected source
  mapping and `track_sample_key[:, 1]`;
- exact-only `source_frame_interpolation` with left = right = target and weight
  zero; and
- structured nullable `source_instance_key`, mechanically derived for an
  observation source and canonical null for every other source-identity domain.

Track-stage interpolation is not permitted in this version. Refined or
interpolated positions must be materialized upstream with their own row identity
and acquisition-frame mapping before track kinematics selects them. A bare
`frame_indices` array, matching row count, or plausible numerical range cannot
establish track-sample identity.

This rule applies to generic cross-stage row-lineage comparisons as well as
tracking. If exactly one source exposes `instance_key`, validation fails closed;
it must never downgrade to a positional comparison. If both sources are keyless,
the caller must deliberately select `legacy_positional`. New outputs that record
the decision use `row_identity_mode` with schema
`palette.row_identity_mode.v1`; modern outputs use `instance_key` and historical
compatibility outputs use `legacy_positional`.

Current refined-detect, refined-keypoint, and refined-subject-mask writers stamp
that contract on new runs. Their shared validator requires an explicitly modern
run to contain unique keys, rejects an explicit positional downgrade when keys
exist, and gives historical unstamped keyless runs a distinct legacy-compatibility
warning rather than describing `instance_key` as merely optional.

## Source Rowset Fingerprint

Arena assignment and tracking bind to one exact source rowset using a distinct
observation-rowset fingerprint. Version 1 covers:

```text
source_rowset_path
row_count
source_edit_revision, when present
sha256(sorted unique uint64 instance_key values)
```

The final fingerprint is SHA-256 over canonical JSON containing those fields.
Sorting makes physical row reordering identity-neutral; add, delete, replace,
path, and revision changes remain visible. Duplicate keys fail closed.

This fingerprint is distinct from run-lineage `source_fingerprint`, byte-level
artifact hashes, and future `appearance_embedding`/`reid_signature` evidence.
Writers snapshot rowset identity at input resolution and verify it again before
publication. Downstream keyed consumers require the recorded fingerprint and
key set to match their source.

## Per-row source signatures

Rowset fingerprints prove membership, not row-content compatibility. Copy-
forward materialization additionally uses the versioned
`source_row_signature` contract from
`fisheye.shared.row_source_signature`. Each `uint8[32]` digest is attached to
one `instance_key` and covers the explicitly declared content and/or trusted
row revisions that affect a target stage, plus a stage compatibility context.

Changing a bbox while preserving `instance_key` therefore preserves identity
but changes the crop/keypoint/mask source signature. Physical reorder preserves
the keyed signature mapping. Run-wide `edit_revision` and rowset fingerprint
remain publication gates; neither is a substitute for this per-row invalidation
signal. See `docs/stable_identity_incremental_materialization_decision.md` for
the full reuse contract and stage matrix.

## Tracker API

Tracking methods consume one `TrackingObservations` contract and return one
`TrackingResult` contract. The public dispatch surface is
`fisheye.tracking.build_tracking(...)`.

The current registered method is `single_subject_per_arena`, the strict
deterministic fast path. Simultaneous detections in one `(frame, arena)` fail.
Future methods must use the same input/result and persisted contracts. Adding a
multi-subject tracker must not require a `track_kinematics` schema change.

## Mutable Ordering And Revision Decisions

The target high-frequency authoring model is append-order physical rows plus a
CSR-style `frame_index/` lookup. Stable IDs and indexes, not physical sorting,
provide identity and frame lookup. Existing refined-detect writers may continue
sorted whole-surface rewrites until the append/index migration lands; they must
preserve stable IDs across that rewrite.

Tracking edits increment `edit_revision` and append audit events while a run is
mutable. Locking/finalizing creates a reproducible revision boundary. Switching
tracking method or rematerializing against a changed rowset creates a new run.

## Subject And Appearance Boundary

An expected subject count is not an identity namespace. A recording with
`expected_subject_count=N` and no known biological IDs remains valid
`count_only` context: detections receive `instance_key`, tracking may produce up
to N or more run-local `track_id` hypotheses, and `recording_subjects` remains
empty. Neither row count nor a stable track is permission to synthesize
`subject_id`. This is the intended starting point for future multi-subject
recordings whose animals were not individually identified at acquisition.

The future authoritative biological linkage should be a revisioned
`subject_identity_runs/<run>` mapping track segments to optional subjects, with
status, confidence, reviewer, and evidence. The registry projects that surface.
Appearance embeddings are probabilistic evidence only: they may propose a link
but must not mint or overwrite confirmed `subject_id`.

## Implementation Status

As of 2026-07-09:

- sparse refined instances and observation `instance_key` propagation are active;
- keyed tracking rows and rowset fingerprints are active for new runs;
- method-neutral tracker dispatch is active;
- `single_subject_per_arena` remains the only registered tracker;
- multi-subject association, tracking review events, subject identity runs, and
  appearance re-identification remain future work.

## References

- `docs/multi_subject_tracking_phase5_plan.md`
- `docs/sleap_palette_storage_assessment.md`
- `docs/refined_detect_sparse_instances_schema.md`
- `docs/refined_detect_row_identity_contract.md`
- `docs/single_subject_per_arena_tracking_contract.md`
- `docs/realtime_sparse_row_index_contract.md`
- `docs/mutable_review_runs_contract.md`
