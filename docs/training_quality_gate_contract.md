# Training Quality Gate Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
-->

## Purpose

Define the vocabulary, value types, activation semantics, and exclusion reason
strings for quality gates across the three training data pipelines (detection,
keypoints, eye masks). The profile schema contracts for each data type
explicitly declare quality gate thresholds out of scope. This document fills
that gap so gate behavior can be audited and compared across types without
reading implementation code.

Related TODO: `docs/keypoint_multi_skeleton_todo.md`

## Scope

In scope:

- gate parameter names, types, and allowed values per data type
- activation semantics (when gates engage)
- exclusion reason string vocabulary and format
- cross-type comparison of gate coverage
- manifest provenance (how gate params and exclusions are recorded)

Out of scope:

- recommended default thresholds for any gate
- CI/CD enforcement policy for gate parameters
- profile computation logic (covered by profile schema contracts)
- export-time array layout or merged zarr structure

## Contract Version

v1. Additive versioning: new gates or reason strings may be added without
bumping the major version. Removing or renaming an existing gate or reason
string requires a v2 bump.

## Gate Architecture

Quality gates operate in a two-tier model:

1. **Dataset-level gates** run inside `prepare_*` scripts. They decide whether
   an entire dataset (zarr) enters the training manifest. All three data types
   implement this tier.

2. **Row-level gates** run inside `export_*` scripts. They decide which
   individual samples (rows) within a passing dataset are included in the
   merged output. Currently only keypoints implements this tier.

Activation follows **any-non-None** semantics: the gate system engages when at
least one gate parameter is supplied (not `None`). If every gate parameter is
`None`, the entire quality gate path is skipped and no registry quality query
is issued.

## Shared Dataset-Level Gates

Two gate parameters are shared across all three data types.

| Parameter | Type | Allowed values | Shared by |
|---|---|---|---|
| `--require-review-state` | `str` (choice) | `approved`, `pending`, `rejected`, `needs_review` | detection, keypoints, eye masks |
| `--require-review-intended-use` | `str` (choice) | `training`, `full_recording` | detection, keypoints, eye masks |

Both default to `None` (inactive). When set, they filter against the
review metadata in each type's quality/profile registry view.

## Detection Dataset-Level Gates

Source: `src/fisheye/utils/prepare_detect_training_from_registry.py`

Naming note: `refined_detect_review_current` is the preferred registry view for
reviewed refined-detect gating rows. `detect_quality_current` remains a
compatibility alias. Both are distinct from raw detect quality reports under
`detect_runs/<run>/quality_reports/<quality_run>`.

### Parameters

| Parameter | Type | Allowed values | Default |
|---|---|---|---|
| `--require-review-state` | `str` (choice) | `approved`, `pending`, `rejected`, `needs_review` | `None` |
| `--require-review-intended-use` | `str` (choice) | `training`, `full_recording` | `None` |
| `--max-interpolated-detections-rate` | `float` | 0.0 -- 1.0 | `None` |

`--max-interpolated-detections-rate` is a legacy-compatibility gate. Current
sparse refined-detect runs should normally rely on review-state/use gating and
carry zero or no interpolation in the curated surface.

### Activation condition

```python
quality_gate_active = (
    args.require_review_state is not None
    or args.require_review_intended_use is not None
    or args.max_interpolated_detections_rate is not None
)
```

### Filtering phases

1. **SQL pre-filter**: `registry.query_refined_detect_review_current()` is called
   twice -- once with all gate params (filtered) and once with only
   `dataset_ids` (unfiltered). Datasets absent from the filtered result set
   are diagnosed against the unfiltered result to determine the specific
   exclusion reason.

2. **Zarr divergence verification**: Every dataset that passed the SQL filter
   is opened on disk. Registry metadata is compared field-by-field against the
   zarr's on-disk attrs (`source_detect_run`, `review_state`,
   `review_intended_use`, `review_resolved_group`, detection counts,
   `interpolated_detections_rate` within tolerance 1e-9 for legacy-compatible
   runs, `zarr_mtime_ns`).
   Any mismatch raises `ValueError` and aborts the run.

## Keypoint Dataset-Level Gates

Source: `src/fisheye/utils/prepare_keypoint_training_from_registry.py`

### Parameters

| Parameter | Type | Allowed values | Default |
|---|---|---|---|
| `--require-review-state` | `str` (choice) | `approved`, `pending`, `rejected`, `needs_review` | `None` |
| `--require-review-intended-use` | `str` (choice) | `training`, `full_recording` | `None` |
| `--min-usable-keypoints-rate` | `float` | 0.0 -- 1.0 | `None` |
| `--allow-cross-method-review-fallback` | `store_true` | flag | `False` |

### Activation condition

```python
quality_gate_active = (
    args.require_review_state is not None
    or args.require_review_intended_use is not None
    or args.min_usable_keypoints_rate is not None
)
```

`--allow-cross-method-review-fallback` does not by itself activate the gate
system. It modifies behavior only when the gate is already active and a
dynamic keypoint-run selector (`latest_traditional` or `latest_yolo`) is in
use: if no reviewed run exists for the requested method, the system falls back
to a reviewed run from a different method.

### Filtering phases

1. **SQL pre-filter**: `registry.query_keypoint_quality_current()` is called
   with `dataset_ids`, `keypoint_method`, `review_state`,
   `review_intended_use`, and `min_usable_keypoints_rate`. The filtered result
   determines passing datasets.

2. **Per-zarr validation**: Each passing zarr is opened and its refined
   keypoint quality metadata is verified against the registry row and CLI
   constraints. A mismatch raises `ValueError`.

## Keypoint Row-Level Gates

Source: `src/fisheye/utils/export_keypoint_training_zarr.py`

### Parameter

| Parameter | Type | Allowed values | Default |
|---|---|---|---|
| `--row-gate-policy` | `str` (choice) | `auto`, `refined_usable`, `raw_success` | `auto` |

### Policy semantics

| Policy | Behavior |
|---|---|
| `auto` | For each dataset, attempt to find a `usable_keypoints` boolean mask from a refined keypoints run linked to the selected keypoint run. If found, use it (`refined_usable`). If not found, fall back to `raw_success`. Different datasets may resolve to different policies, producing a `"mixed"` aggregate policy. |
| `refined_usable` | Require a `usable_keypoints` mask from a compatible refined keypoints run. If not found, raise `ValueError` (hard failure). |
| `raw_success` | Use only the `detection_success` boolean from the keypoint run. Refined runs are not consulted. |

Refined run resolution order:

1. Check manifest `refined_keypoint_run` or `quality_registry_refined_run`.
2. If that run exists and its `source_keypoints_run` matches, use it.
3. Otherwise scan all refined runs whose `source_keypoints_run` matches, sort
   by timestamp descending, take the most recent with a valid
   `usable_keypoints` array.

Rows where the selected mask is `False` are silently excluded from the merged
output. No per-row exclusion reason strings are emitted.

## Eye Mask Dataset-Level Gates

Source: `src/fisheye/utils/prepare_eye_mask_training_from_registry.py`

### Parameters

| Parameter | Type | Allowed values | Default |
|---|---|---|---|
| `--require-review-state` | `str` (choice) | `approved`, `pending`, `rejected`, `needs_review` | `None` |
| `--require-review-intended-use` | `str` (choice) | `training`, `full_recording` | `None` |
| `--min-usable-rate` | `float` | 0.0 -- 1.0 | `None` |
| `--eye-mask-method` | `str` | free-form | `None` |
| `--profile-stage-group` | `str` (choice) | `eye_masks_runs`, `refined_eye_masks_runs` | `None` |

`--eye-stage` (choices: `auto`, `refined_subject_masks_runs`,
`refined_eye_masks_runs`, `eye_masks_runs`; default `auto`) interacts with
the gate logic. The profile registry is still keyed to the historical eye-mask
families, so only `eye_masks_runs` and `refined_eye_masks_runs` act as implicit
`--profile-stage-group` values when no explicit profile stage was provided.
`refined_subject_masks_runs` controls export source selection, not profile-table
filtering.

### Activation condition

```python
quality_filters_active = any([
    args.eye_mask_method is not None,
    args.min_usable_rate is not None,
    args.require_review_state is not None,
    args.require_review_intended_use is not None,
    _default_profile_stage_group(args) is not None,
])
```

Where `_default_profile_stage_group` returns the explicit
`--profile-stage-group` if provided, else `--eye-stage` if it names a concrete
stage, else `None`.

### Filtering phases

1. **SQL pre-filter**: `registry.query_eye_mask_data_profile_latest()` is
   called with `stage_group`, `eye_mask_method`, and `min_usable_rate`. All
   three are conjunctive (AND logic) at the database level.

2. **Candidate selection** (`_choose_profile_candidate`): For each dataset,
   returned profile rows are ranked by:
   - stage-group preference for profile rows (controlled by historical
     eye-stage/profile filters; default order: `refined_eye_masks_runs` first,
     then `eye_masks_runs`)
   - creation time descending (newest first)
   - review state/use match

   The first candidate satisfying all review requirements is selected.

3. Datasets with no surviving candidate are excluded with a single composite
   reason string.

## Exclusion Reason Vocabulary

### Detection exclusion reasons

| Reason string | Meaning |
|---|---|
| `missing_quality_row` | No row exists in `refined_detect_review_current` for this dataset. |
| `review_state_mismatch:{actual}!={expected}` | `--require-review-state` set; actual state differs. `{actual}` is the observed value or `missing` if null. |
| `review_use_mismatch:{actual}!={expected}` | `--require-review-intended-use` set; actual intended use differs. |
| `missing_interpolated_detections_rate` | Legacy compatibility gate only: `--max-interpolated-detections-rate` set but rate is null for this dataset. |
| `interpolated_rate_above_threshold:{rate:.6f}>{threshold:.6f}` | Legacy compatibility gate only: interpolated detection rate exceeds the threshold. Formatted to 6 decimal places. |
| `excluded_by_quality_filters` | Catch-all: SQL filter excluded the dataset but no specific condition matched. |

### Keypoint exclusion reasons

| Reason string | Meaning |
|---|---|
| `missing_quality_row` | No quality row exists for this dataset. |
| `no_quality_for_method:{method}` | Quality row exists but none for the required keypoint method (e.g., `traditional_pose`). |
| `review_state_mismatch:{observed}!={required}` | Review state does not match `--require-review-state`. |
| `review_use_mismatch:{observed}!={required}` | Review intended use does not match `--require-review-intended-use`. |
| `missing_usable_keypoints_rate` | `--min-usable-keypoints-rate` set but `usable_keypoints_rate` is null. |
| `usable_rate_below_threshold:{rate}<{threshold}` | Usable keypoints rate is below the threshold. Formatted to 6 decimal places. |
| `excluded_by_quality_filters` | Catch-all: SQL filter excluded the dataset but no specific condition matched. |

### Eye mask exclusion reasons

| Reason string | Meaning |
|---|---|
| `missing_or_mismatched_eye_mask_profile` | No profile row matched the gate criteria, or all candidates failed review checks. |

Detection and keypoint pipelines emit **granular** per-condition reason strings
with colon-delimited diagnostic detail suffixes (e.g.,
`review_state_mismatch:pending!=approved`). Eye masks emit a single
**composite** reason because filtering is performed by candidate ranking rather
than per-field condition checks.

All three types also silently drop datasets matching the training artifact
heuristic (`zarr_use=training` and path looks like a merged/training artifact)
before quality gates run. These are not recorded in `quality_exclusions`.

## Cross-Type Summary Table

| Gate | Detection | Keypoints | Eye Masks | Level |
|---|---|---|---|---|
| `--require-review-state` | yes | yes | yes | dataset |
| `--require-review-intended-use` | yes | yes | yes | dataset |
| `--max-interpolated-detections-rate` | yes | -- | -- | dataset |
| `--min-usable-keypoints-rate` | -- | yes | -- | dataset |
| `--allow-cross-method-review-fallback` | -- | yes | -- | dataset |
| `--min-usable-rate` | -- | -- | yes | dataset |
| `--eye-mask-method` | -- | -- | yes | dataset |
| `--profile-stage-group` | -- | -- | yes | dataset |
| `--row-gate-policy` | -- | yes | -- | row |

For detection, `--max-interpolated-detections-rate` is retained mainly for
historical refined runs. Review-state and review-intended-use are the primary
current gates for sparse refined-detect data.

Zarr divergence verification (hard-fail assertion, not soft exclusion) is
implemented by detection and keypoints. Eye masks do not perform post-SQL zarr
verification.

## Manifest Provenance

Each `prepare_*` script records gate configuration and exclusions in the output
manifest JSON.

### Gate parameters

Stored under `manifest["query_filter"]`. All three types record the tool name,
task identifier, and every gate parameter value (including `None` for inactive
gates). Eye masks additionally record the resolved `profile_stage_group`.

### Quality exclusions

Stored under `manifest["quality_exclusions"]` as an array of objects:

```json
{
  "dataset_id": "...",
  "zarr_path": "...",
  "reason": "<reason string from vocabulary above>"
}
```

### Per-dataset quality metadata

Detection: not embedded (quality info exists only in console output).

Keypoints: each dataset entry includes `quality_registry_used`,
`quality_registry_refined_run`, `quality_registry_keypoint_method`,
`cross_method_fallback_used`, `usable_keypoints_total`,
`usable_keypoints_rate`, and `keypoint_review_status`.

Eye masks: each dataset entry includes an `eye_mask_profile` block with
`profile_run`, `stage_group`, `eye_mask_method`, `usable_rate`,
`review_state`, `review_intended_use`, and `profile_created_utc`.

### Row-gate provenance (keypoints only)

The keypoint export zarr records row-gate metadata in three locations:

- Keypoint group attrs: `row_gate_policy`, `row_gate_applied`,
  `row_gate_counts`.
- Root `training_export` attr: `row_gate.requested_policy`,
  `row_gate.applied_policy`, `row_gate.per_policy_counts`.
- Summary JSON and merged manifest: per-source `row_gate_policy`,
  `row_gate_refined_run`, `row_gate_selected`, `row_gate_total`,
  `row_gate_raw_success_true`, `row_gate_usable_true`.

## Related Contracts

- `docs/detection_data_profile_schema_contract.md` -- detection profile schema
- `docs/keypoint_data_profile_schema_contract.md` -- keypoint profile schema
- `docs/detection_merged_export_contract.md` -- detection merged export layout
- `docs/keypoint_merged_row_gate_contract.md` -- keypoint row-gate design
- `docs/keypoint_training_data_card_contract.md` -- keypoint training data card schema

## Implementation References

| Data type | Authoritative source |
|---|---|
| Detection (prepare) | `src/fisheye/utils/prepare_detect_training_from_registry.py` |
| Keypoints (prepare) | `src/fisheye/utils/prepare_keypoint_training_from_registry.py` |
| Keypoints (export) | `src/fisheye/utils/export_keypoint_training_zarr.py` |
| Eye masks (prepare) | `src/fisheye/utils/prepare_eye_mask_training_from_registry.py` |
