# Keypoint Auto-Approval TODO

Purpose: define a safe, auditable auto-approval path for refined keypoint runs
without conflating pipeline progression approval with training-label approval.

Date anchored: 2026-03-02.

## Decision Snapshot (2026-03-02)

- Auto-approval is acceptable for `intended_use=full_recording` when strict
  gates pass.
- Auto-approval for `intended_use=training` is deferred.
- Coverage alone is not sufficient; QC and geometry/confidence gates are
  required.

## Draft Status Schema (Auto-Review)

This extends the existing canonical review payload shape in
`docs/review_status_schema_unification_contract.md`.

Top-level payload written to:
`refined_keypoints_runs/<run>.attrs["keypoint_review_status"]`

```json
{
  "state": "approved",
  "method": "algorithmic",
  "intended_use": "full_recording",
  "timestamp_utc": "2026-03-02T00:00:00+00:00",
  "reviewer": "auto:keypoint_policy_v1",
  "notes": "Auto-approved by keypoint policy.",
  "auto_review": {
    "policy_id": "keypoint_auto_review_v1",
    "policy_version": 1,
    "result": "approved",
    "applied_at_utc": "2026-03-02T00:00:00+00:00",
    "thresholds": {
      "remaining_failures_max": 0,
      "refined_success_rate_min": 1.0,
      "usable_keypoints_rate_min": 0.999,
      "confidence_valid_rate_min": 0.999,
      "geometry_valid_rate_min": 0.999
    },
    "evidence": {
      "refined_run": "refined_keypoints_YYYY-MM-DD_HH-MM-SS",
      "source_keypoints_run": "keypoints_YYYY-MM-DD_HH-MM-SS",
      "total_rois": 23287,
      "remaining_failures": 0,
      "refined_success_rate": 1.0,
      "usable_keypoints_rate": 0.9999,
      "confidence_valid_rate": 0.9998,
      "geometry_valid_rate": 0.9999,
      "reason_counts": {
        "manual_correction": 2
      }
    },
    "drift": {
      "enabled": false,
      "status": "not_evaluated",
      "reference_profile_id": null,
      "max_zscore": null,
      "outlier_metric_count": null
    }
  }
}
```

Notes:

- Keep `state/method/intended_use/timestamp_utc` canonical and always present.
- `auto_review` is additive evidence, not a replacement for canonical fields.
- Manual review remains the highest-priority override.

## Phases

### Phase 1: Full-Recording Auto-Approval

Scope:

- Evaluate refined keypoint runs immediately after refinement/finalize.
- If all gates pass, write `keypoint_review_status` with:
  - `state=approved`
  - `method=algorithmic`
  - `intended_use=full_recording`
- If gates fail, write `state=needs_review` with failure evidence in
  `auto_review`.

Initial gates:

- `remaining_failures == 0`
- `refined_success_rate == 1.0`
- `usable_keypoints_rate >= configured threshold`
- `confidence_valid_rate >= configured threshold`
- `geometry_valid_rate >= configured threshold`
- no disqualifying reason tags (for example `detection_issue`)

### Phase 2: Distribution/Drift Checks

Scope:

- Compare generated keypoint profile metrics against reference distributions
  from approved training datasets.
- Add drift result to `auto_review.drift`.
- Convert otherwise approved runs to `needs_review` if drift exceeds policy.

Candidate metric families:

- skeleton geometry (triangle area, angles, inter-keypoint distances)
- temporal smoothness/jitter metrics
- confidence distributions by keypoint
- per-camera/per-rig stratified baselines

### Phase 3: Training Auto-Approval

Status: deferred.

Defer note (2026-03-02):

- Do not auto-approve `intended_use=training` until Phase 1 and Phase 2 have
  stable outcomes and low false-approval rates over multiple batches.

## TODO Checklist

- [x] Implement Phase 1 policy evaluation and status writes.
- [x] Add unit tests for pass/fail gate decisions and payload shape.
- [x] Add operator-facing reporting for auto-review outcomes.
- [ ] Implement Phase 2 drift computation and thresholds.
- [x] Add registry query filters for `method=algorithmic` and policy id/version.
- [x] Defer Phase 3 (training auto-approval) pending validation.

Completion notes (2026-03-02):

- Added `fisheye.utils.auto_keypoint_review` with policy-based evaluation and
  canonical `keypoint_review_status` writes, including additive
  `auto_review` evidence payload.
- Added `--auto-review-full-recording` integration to
  `fisheye.utils.refine_keypoints_batch` with JSON/log reporting and source-run
  matching.
- Added unit coverage for pass/fail, dry-run, source-run resolution, and
  existing manual-review preservation.
- Added registry storage and `registry_query` filters for
  `keypoint_review_method`, `keypoint_review_policy_id`, and
  `keypoint_review_policy_version`.

## Acceptance Criteria

- Auto-approved runs are fully auditable from run attrs alone.
- Manual review can override auto-review without data loss.
- Registry views can distinguish manual vs algorithmic approvals.
- Training pipelines do not treat algorithmic full-recording approval as
  training approval unless explicitly allowed in future policy.
