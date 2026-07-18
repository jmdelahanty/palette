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
| `subject_id` | registry/global biology | Optional known biological identity. Tracking alone must not silently infer it. |

The globally addressable forms are:

- observation: `(recording_id, instance_key)`;
- curated row: `(recording_id, refined_detect_run, refined_row_id)`;
- trajectory: `(recording_id, tracking_run, track_id)`;
- biological identity: `subject_id`.

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

This rule applies to generic cross-stage row-lineage comparisons as well as
tracking. If exactly one source exposes `instance_key`, validation fails closed;
it must never downgrade to a positional comparison. If both sources are keyless,
the caller must deliberately select `legacy_positional`. New outputs that record
the decision use `row_identity_mode` with schema
`palette.row_identity_mode.v1`; modern outputs use `instance_key` and historical
compatibility outputs use `legacy_positional`.

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
