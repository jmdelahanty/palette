# Eye-Mask Training Data Card Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
-->

Purpose: define a canonical, reproducible summary payload for eye-mask training
sets with parity to detect/keypoint data-card workflows.

## Scope

In scope:
- eye-mask training data-card schema (`v1`)
- required section keys and metric naming
- registry-first aggregation behavior
- explicit fallback behavior when profile rows are unavailable
- default static plot bundle outputs

Out of scope:
- model pass/fail thresholds
- UI/dashboard implementation details

## Schema Identity

- `schema_name`: `eye_mask_training_data_card`
- `schema_version`: `v1`

## Required Top-Level Sections (`v1`)

1. `selection`
- dataset count
- split metadata
- filter metadata
- optional row-count metadata (`rows_pre_gate`, `rows_post_gate`, `split_counts`)

2. `quality`
- usable/success summary for ROI pairs
- dataset-level usable-rate summary + histogram
- throughput summary (`rois_per_second` weighted aggregate)

3. `geometry`
- eye-mask geometry summaries/histograms (eye-separation and ellipse axes)

4. `spatial`
- edge-proximity aggregate
- optional aggregated center heatmap (`grid_h`, `grid_w`, `density`)

5. `composition`
- count maps over context facets (e.g. rig/camera/arena/dish/canvas/protocol/method)

6. `subject_coverage`
- manifest dataset count
- lineage-covered dataset count
- missing-lineage dataset ids
- optional coverage unavailable reason

7. `parity`
- train/val delta payload when split source-index data is available
- explicit unavailable reason when parity cannot be computed

8. `audit_freshness`
- registry/profile source mode
- fallback usage + reason
- canonical dataset-id resolution count
- zarr mtime mismatch counts
- source run/profile refs used to build the card

## Registry-First Read Contract

Default read order:
1. `query_eye_mask_data_profile_latest(...)` registry API if available.
2. `eye_mask_data_profile_latest` SQL surface if available.
3. Explicit fallback path only when fallback mode is enabled.

Fail-closed defaults:
- missing latest profile rows fail by default with remediation text.
- stale `zarr_mtime_ns` rows fail by default unless an explicit override is enabled.

## Explicit Fallback Policy

Fallback source:
- `eye_mask_performance_latest` (+ provenance join for composition/lineage fields)

Fallback behavior:
- disabled by default
- enabled with `--allow-profile-fallback-scan`
- all fallback usage must be recorded in `audit_freshness`

## Plot Bundle Contract (`v1`)

Default behavior:
- generate plots when card is written unless plots are disabled
- do not auto-open files unless `--view` is requested

Default plot set:
- usable-rate distribution
- eye-separation distribution
- ellipse major-axis distribution
- ellipse minor-axis distribution
- spatial center heatmap
- composition counts summary
- train/val parity delta chart
- genotype counts
- DPF histogram

## Storage Convention

Card JSON:
- `<set_id>.data_card.json`

Default plot directory:
- `<set_id>.data_card.plots/`
