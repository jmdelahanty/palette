# Repo-Wide Staleness Implementation TODO

<!-- design-meta
status: draft
last_updated: 2026-04-07
-->

## Purpose

Turn the repo-wide staleness policy into a concrete implementation sequence.

This note is intentionally narrower than the policy and checklist docs. It is
the prioritized "what should we build next?" list.

Related notes:

- [repo_wide_staleness_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_policy.md)
- [repo_wide_staleness_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_checklist.md)
- [repo_wide_staleness_gap_matrix.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_gap_matrix.md)
- [repo_wide_staleness_workflow_edge_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_workflow_edge_checklist.md)
- [crop_live_view_vs_materialized_stream_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/crop_live_view_vs_materialized_stream_design.md)
- [refined_detect_collapse_v2.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_detect_collapse_v2.md)
- [refined_detect_multisubject_goal.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_detect_multisubject_goal.md)
- [refined_detect_sparse_instances_schema.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_detect_sparse_instances_schema.md)

## Guardrails

These should be treated as baseline decisions unless a later design note
explicitly changes them.

- Keep raw provenance runs append-only by default.
- Keep refined/manual artifacts as the editable working surface.
- Keep `stale` separate from review state.
- Keep `crop_runs` as the canonical crop geometry/provenance layer.
- Do not collapse downstream lineage to raw detect boxes alone.
- Allow `roi_images` to become optional only in mixed-mode archive classes.
- Keep training/export artifacts materialized by default even if analysis moves
  toward `geometry_only`.

## Priority 1: Canonicalize Subject/Swim Staleness

Why first:

- the local workflow now exists and is practically useful
- the repo still lacks the canonical top-level/stable contract
- this is the clearest parity gap versus eye-mask stale handling

Implement:

- write canonical top-level `source_subject_mask_stale` payloads on refined
  subject-mask runs
- project that payload into registry/query surfaces
- add an explicit subject-mask stale resolution path
- keep stale separate from component/run review state
- preserve the row-level stale queue for targeted review of affected ROIs

Applies to:

- unified subject masks
- swim bladder as a subject-mask component
- eye components once they fully converge on refined subject-mask authoring

## Priority 2: Define Detect/Crop Correction Contract

Why second:

- bbox edits can fan out into crop, keypoints, eye masks, subject masks, and
  swim bladder
- this is the main unresolved repo-wide policy question

Implement:

- define the row-stable case explicitly:
  - move/resize existing bbox
  - same fish, same frame, same row identity
  - targeted downstream stale is allowed
- define the identity-breaking case explicitly:
  - add/delete/split/merge
  - row identity changed or became ambiguous
  - downstream rerun/invalidation is required
- keep `refined_detect_runs` as the canonical curated detect surface; do not
  reintroduce separate preferred detect/crop stage families
- current short-term detect writes may still use the dense refined root as a
  bridge, but consumer-facing reads should converge on the sparse
  `instances/` / `source_detections/` surfaces described in the current detect
  design notes
- encode the crop revision/signature consequences of refined-detect edits
- decide whether crop itself gets an explicit top-level stale payload or remains
  a rerun-only derived stage with strong lineage

## Priority 3: Finish Mixed-Mode Crop Reader Migration

Why third:

- the policy direction is clear, but the reader set is not fully migrated
- changing defaults before reader migration would be premature

Implement:

- migrate remaining viewers, tuners, review tools, and diagnostics to the
  shared crop resolver/cache path
- remove direct assumptions that `crop_group["roi_images"]` always exists,
  except in intentionally materialized-only paths
- validate review/tuning latency on `geometry_only` analysis archives
- keep traditional imported/materialized-only pipelines failing clearly when
  given unsupported geometry-only inputs

## Priority 4: Clarify Training/Export Boundary

Why fourth:

- mixed-mode analysis is compatible with repo direction
- geometry-only training artifacts are not

Implement:

- keep training/export zarrs self-contained and materialized
- update training/export docs and validators to make that boundary explicit
- keep downstream provenance recording `source_crop_storage_mode`
- ensure geometry-only source archives can still feed materialized training
  artifacts through export/build steps

Applies to:

- keypoints
- eye masks
- subject masks
- swim bladder

## Priority 5: Re-evaluate Defaults Later

This is explicitly not the next step.

Only revisit defaults after:

- subject/swim stale is registry-native
- detect/crop correction semantics are defined
- review/tuning tools are mixed-mode capable
- representative analysis benchmarks are acceptable

Then consider:

- whether some analysis workflows should default to `geometry_only`
- whether any additional cache/materialization commands are needed

## Non-Goals For This Pass

- do not switch the repo to "detections only" lineage
- do not make training archives `geometry_only`
- do not silently overwrite curated downstream artifacts after source changes
- do not treat stale as just another spelling of `needs_review`

## Immediate Next Patch Candidates

If work continues immediately, the highest-signal next patches are:

1. top-level `source_subject_mask_stale`
2. registry/query support for subject/swim stale
3. explicit subject-mask stale resolution
4. detect/crop row-stable versus identity-breaking propagation rules
