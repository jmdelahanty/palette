# Keypoint Dataset Statistics (Future Work)

## Goal

Define a lightweight, repeatable way to inspect keypoint training data
representation before/after curation changes, with parity to detect data-card
workflow.

## Why This Matters

- catches pose-distribution drift before training
- improves auditability of row gating and refined quality
- gives objective context when pose model metrics move

## Parity Target

Contract reference:
- `docs/keypoint_training_data_card_contract.md`

This TODO is the implementation plan for that contract.

## Status (2026-02-25)

- Implemented:
  - keypoint data-card aggregator + CLI
  - keypoint data-card plotter + CLI
  - pipeline integration flags/default behavior
  - mixed-skeleton hard-fail enforcement (prepare/export/validate/aggregate)
  - skeleton-graph key derivation + alias metadata
  - subject-lineage coverage policy (`warn|require`)
  - `spatial` payload aggregation
  - `train_val_parity` payload aggregation
  - train/val parity delta test coverage
  - landmark heatmap panel generation in default plot bundle
  - focused unit tests for new utilities and guardrails
- Remaining:
  - run/record operator validation checklist on production datasets

## Required Guardrails

- [x] Enforce single-skeleton training sets everywhere.
  - Fail if selected datasets resolve to mixed `skeleton_id`/`kpt_shape`.
  - Enforce in prepare/export/validate/card aggregation surfaces.
- [x] Do not hard-code fish-specific distance metrics as required fields.
  - Required geometry must be derived from `pose_schema.skeleton`.

## Phase 1: Card Aggregation (`v1`)

- [x] Add keypoint training data-card aggregation command.
  - Suggested entrypoint:
    `scripts/py -m fisheye.utils.aggregate_keypoint_training_data_card --manifest <set>.manifest.json --registry <registry.sqlite>`
- [x] Implement required payload sections:
  - [x] `selection`
  - [x] `quality`
  - [x] `geometry`
  - [x] `skeleton_graph_metrics`
  - [x] `spatial`
  - [x] `composition_counts`
  - [x] `subject_coverage`, `genotype_counts`, `dpf_stats`, `dpf_histogram`
  - [x] `train_val_parity`
  - [x] `audit_freshness`
- [x] Include canonical metric keys and alias metadata.
  - Canonical: `edge_<i>_<j>`, `angle_<i>_<j>_<k>`
  - Alias labels when keypoint labels are available.
- [x] Add subject-lineage precheck policy (`warn|require`) aligned with detect.

## Phase 2: Plotting Parity

- [x] Add keypoint data-card plotting utility.
  - Suggested entrypoint:
    `scripts/py -m fisheye.utils.plot_keypoint_training_data_card --card <set>.data_card.json`
- [x] Generate default plot bundle without auto-view:
  - [x] usable-rate distribution
  - [x] triangle-area distribution
  - [x] min-angle distribution
  - [x] heading distribution
  - [x] landmark heatmap panel
  - [x] genotype counts
  - [x] DPF histogram
- [x] Add `--view` option to open generated/existing plots.

## Phase 3: Pipeline Integration

- [x] Add pipeline flag for keypoint card aggregation.
  - Suggested:
    `scripts/py -m fisheye.utils.run_keypoint_training_pipeline ... --aggregate-training-data-card`
- [x] Decide default behavior for non-dry-run merged builds.
  - Prefer generating card + plots by default after successful build/export.
  - Status: auto-enabled with `--export-merged`; disable via `--no-aggregate-training-data-card`.

## Phase 4: Validation and Tests

- [x] Unit tests for mixed-skeleton hard-fail behavior.
- [x] Unit tests for skeleton graph metric derivation.
  - edge and angle key generation from schema
  - alias metadata emission
- [x] Unit tests for subject-lineage policy behavior (`warn` vs `require`).
- [x] Unit tests for aggregate correctness:
  - [x] quality rates
  - [x] geometry stats
  - [x] train/val parity deltas
- [x] Unit tests for plotting utility output contract.

## Operator Validation Checklist

- [ ] Pre-aggregation:
  - verify selected datasets resolve to one skeleton identity
  - verify keypoint quality gates are fresh (no stale mtime rows)
- [ ] Post-aggregation:
  - verify expected `dataset_count` and `rows_post_gate`
  - verify expected lineage coverage and histogram presence
  - verify default plot bundle exists on disk
