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

## Required Guardrails

- [ ] Enforce single-skeleton training sets everywhere.
  - Fail if selected datasets resolve to mixed `skeleton_id`/`kpt_shape`.
  - Enforce in prepare/export/validate/card aggregation surfaces.
- [ ] Do not hard-code fish-specific distance metrics as required fields.
  - Required geometry must be derived from `pose_schema.skeleton`.

## Phase 1: Card Aggregation (`v1`)

- [ ] Add keypoint training data-card aggregation command.
  - Suggested entrypoint:
    `scripts/py -m fisheye.utils.aggregate_keypoint_training_data_card --manifest <set>.manifest.json --registry <registry.sqlite>`
- [ ] Implement required payload sections:
  - `selection`
  - `quality`
  - `geometry`
  - `skeleton_graph_metrics`
  - `spatial`
  - `composition_counts`
  - `subject_coverage`, `genotype_counts`, `dpf_stats`, `dpf_histogram`
  - `train_val_parity`
  - `audit_freshness`
- [ ] Include canonical metric keys and alias metadata.
  - Canonical: `edge_<i>_<j>`, `angle_<i>_<j>_<k>`
  - Alias labels when keypoint labels are available.
- [ ] Add subject-lineage precheck policy (`warn|require`) aligned with detect.

## Phase 2: Plotting Parity

- [ ] Add keypoint data-card plotting utility.
  - Suggested entrypoint:
    `scripts/py -m fisheye.utils.plot_keypoint_training_data_card --card <set>.data_card.json`
- [ ] Generate default plot bundle without auto-view:
  - usable-rate distribution
  - triangle-area distribution
  - min-angle distribution
  - heading distribution
  - landmark heatmap panel
  - genotype counts
  - DPF histogram
- [ ] Add `--view` option to open generated/existing plots.

## Phase 3: Pipeline Integration

- [ ] Add pipeline flag for keypoint card aggregation.
  - Suggested:
    `scripts/py -m fisheye.utils.run_keypoint_training_pipeline ... --aggregate-training-data-card`
- [ ] Decide default behavior for non-dry-run merged builds.
  - Prefer generating card + plots by default after successful build/export.

## Phase 4: Validation and Tests

- [ ] Unit tests for mixed-skeleton hard-fail behavior.
- [ ] Unit tests for skeleton graph metric derivation.
  - edge and angle key generation from schema
  - alias metadata emission
- [ ] Unit tests for subject-lineage policy behavior (`warn` vs `require`).
- [ ] Unit tests for aggregate correctness:
  - quality rates
  - geometry stats
  - train/val parity deltas
- [ ] Unit tests for plotting utility output contract.

## Operator Validation Checklist

- [ ] Pre-aggregation:
  - verify selected datasets resolve to one skeleton identity
  - verify keypoint quality gates are fresh (no stale mtime rows)
- [ ] Post-aggregation:
  - verify expected `dataset_count` and `rows_post_gate`
  - verify expected lineage coverage and histogram presence
  - verify default plot bundle exists on disk
