# Design: a subject_mask data-profile path (parity with detection/keypoint)

**Date:** 2026-06-18
**Status:** Design proposal (read-only investigation; nothing implemented).
**Context:** The data-profile pattern exists for detection, keypoint, eye_mask, and
training_image but **not subject_mask** (see
`registry_reconcile_collapse_audit_2026-06-18.md`). Eyes are now a channel within the
unified subject mask and the eye-mask stage is being deprecated
(`eye_mask_severance_plan_2026-05-28.md`), so subject_mask is the stage that should carry
the profile capability going forward.

## What "profile" means here (the pattern to mirror)

For detection/keypoint, a *data profile* is a distribution summary of the **data a model
produced or will train on** — distinct from `*_performance` (throughput/coverage of a run)
and `*_quality`/review (human acceptance). The contract is three layers:

1. **Builder** (`build_<stage>_profile_summary`, e.g. `utils/detection_profile.py`,
   `utils/keypoint_profile.py`) → returns a `profile_summary` dict with stable sections:
   `schema_name`/`schema_version`, `dataset`, `source` (+ review + content fingerprint),
   distribution sections (`coverage`, `counts`, `geometry_norm`, `spatial`, `histograms`
   for detection; `quality`, `geometry`, `derived_metrics` for keypoint), and `composition`
   (rig/camera/arena/dish/protocol/genotype/dpf).
2. **Zarr write** → stored under `analysis/<stage>_profile_runs/<run>/profile_summary`
   (a run group attr), with a `latest` pointer on the parent.
3. **Registry sync** (`sync_<stage>_profile_registry.py` → `<stage>_data_profile` table)
   → `_build_profile_payload()` lifts high-value scalars/percentiles into normalized
   columns for SQL, and archives the whole summary as `profile_json`. Keyed
   `(dataset_id, profile_run)`, idempotent UPSERT.

## What subject_mask already has (so we don't duplicate)

- `subject_mask_performance` — run-level: `total_rois`, `rows_with_any_mask`,
  `coverage_percent`, `rois_per_second`, source/tuning provenance, component
  availability (`available_components_json`, `eye_component_mode`), review, lifecycle.
- `subject_mask_component_quality` — **per component** (`component_name`,
  `component_family`): `available`, `rows_with_component_mask`,
  `rows_with_component_mask_rate`, per-component review + lifecycle.

These are throughput + per-component coverage + review. **None of it is a distribution
profile of mask geometry across the dataset.** That is the gap.

## What a subject_mask profile would ADD

Available in zarr (`refined_subject_masks_runs/<run>`, per `extractors/masks.py`):
`masks_roi (N_rois, N_components, H, W)`, `metrics/mask_present (N_rois, N_components)`,
`mask_labels`, `available_channels`, `total_rois`, `summary_statistics`, source provenance.

A profile would summarize the **distribution** of:
- **Per-component coverage** across ROIs — presence rate per component, and its spread
  (not just the single rate already in component_quality).
- **Component area / fill** — normalized mask area per component (body, eyes_union or
  eye_left/right, swim_bladder): percentiles (p10/p50/p90), so training data imbalance is
  visible (e.g. swim_bladder tiny vs body large).
- **Component co-occurrence / composition** — fraction of ROIs with the full label set vs
  partial; eye-mode (union vs L/R) distribution.
- **Spatial** (optional, mirrors detection) — component centroid / bbox distribution within
  the ROI.
- Standard `composition` block (rig/camera/arena/dish/protocol/genotype/dpf) for
  cross-recording slicing, same as the others.

This is the training-data-assessment view subject_mask currently lacks — the analog of
detection's `geometry_norm`/`histograms`.

## Proposed shape (mirrors the existing three layers)

1. **Builder:** `utils/subject_mask_profile.py::build_subject_mask_profile_summary(...)`
   → `schema_name="subject_mask_dataset_profile"`, `schema_version="v1"`, sections
   `dataset` / `source` / `coverage` / `components` (per-label distribution stats) /
   `composition`. Reuse the percentile/stats helpers already shared across the profile
   builders.
2. **Zarr:** write to `analysis/subject_mask_profile_runs/<run>/profile_summary` with a
   `latest` pointer — identical convention to detection/keypoint.
3. **Table:** `subject_mask_data_profile`, keyed `(dataset_id, profile_run)`, columns:
   ids (`dataset_id`, `profile_run`, `recording_id`, `zarr_use`), source
   (`subject_mask_method`, `label_schema_id`, `source_keypoints_run`, `source_crop_run`,
   `run_semantics`), coverage (`total_rois`, `rows_with_any_mask`, `coverage_percent`,
   `available_component_count`), per-component distribution
   (`component_area_p10/p50/p90` and `component_presence_rate` per label, or a normalized
   long-form companion table if the label set varies), `composition` columns,
   `profile_json`, `profile_created_utc`, `zarr_mtime_ns`.
4. **Sync:** instead of a standalone `sync_subject_mask_profile_registry.py`, register the
   reader as an **extractor** behind the unified reconcile (see below).

## Tie-in to the reconcile collapse

Do **not** add another standalone `sync_` script — that repeats the duplication the
collapse audit flagged. Express the subject_mask profile reader as an extractor matching the
existing contract:

```
_extract_subject_mask_data_profile_rows(root, *, zarr_path, recording_id, zarr_use) -> List[Dict]
```

and a `replace_subject_mask_data_profile(dataset_id, rows)`. Then it runs automatically
under the proposed `reconcile_dataset_from_root()` orchestrator alongside the other
extractors. This makes "add a profile to subject_mask" a ~one-extractor change rather than
another ~600-line sync script — and validates the reconcile interface on a real new case.

## Open questions for the maintainer (domain calls)

1. **Is a distribution profile actually wanted for subject masks**, or does
   `component_quality` already serve the training-data-assessment need in practice? The
   detection/keypoint precedent says yes, but masks may differ.
2. **Per-component columns vs long-form.** `subject_v1_union` (3 labels) vs `subject_v1_lr`
   (4 labels) means the label set varies. Wide per-component columns are simplest to query
   but brittle to schema changes; a long-form `(profile_run, component_name, metric, value)`
   companion is more flexible. Recommend wide for the common labels + `profile_json` for the
   rest.
3. **Build trigger.** At mask-run finalization (like detection/keypoint), or on demand
   during reconcile? Recommend: build at finalization, sync/repair via reconcile.

## Not done

No code written. This is the spec to implement if subject_mask profiling is wanted; the
smallest first step is the extractor + table, slotted under the reconcile orchestrator.
