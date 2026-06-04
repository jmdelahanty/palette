# Implementation Handoff: Eye-Mask Stage Severance — Phase 1 (re-point)

> **Prerequisite:** Phase 0 (`eye_mask_severance_phase0_coverage_audit.md`) should report GO
> before this runs. Phase 1 is code-only and behavior-neutral, but its safety assumes the
> coverage audit confirmed subject-mask eye geometry exists for the recordings you care about.

You are implementing the first code phase of a planned removal of the legacy `eye_masks` /
`refined_eye_masks` pipeline stages from the palette/fisheye repo. Eyes are now a channel
within the unified subject mask. Your job is NOT to design the removal — that analysis is
done — but to verify it against current code and execute Phase 1 correctly.

## Read first (in this order)

1. `docs/diagnostics/eye_mask_severance_plan_2026-05-28.md` — the plan you are executing.
   Focus: §1 blocker verdict, §2 Phase 1, §3 the classification correction, §4 risks.
2. `docs/diagnostics/contract_drift_audit_2026-05-28.md` — why eye-mask contracts are
   "delete-don't-fix."
3. `docs/diagnostics/repo_eval_2026-05-28.md` — repo context (utils/ sprawl, the
   silent-acceptance theme). Skim.

These docs were written by READ-ONLY analysis agents on 2026-05-28. Treat every file:line
reference in them as a HYPOTHESIS, not fact. Line numbers, function names, and
classifications may already have drifted. Your first deliverable is a verification delta,
not an edit.

## Step 0 — Verify before you trust (produce a written delta)

Before changing anything, confirm against the CURRENT code:

- The crux claim: `analysis/eye_angle_analysis.py` consumes only `ellipse_params` /
  `ellipse_success` (elements [2] major, [3] minor, [4] angle_deg) from
  `resolve_eye_geometry_source`, and `masks_roi` is never used in angle computation.
  Re-read the actual functions; do not trust cited line numbers.
- The subject-mask sources (`shared/refined_subject_eye_geometry.py`,
  `analysis/subject_shape_runs.py`) still produce identical ellipse arrays via the same
  `_measure_mask` / `cv2.fitEllipse` path as `refinement/refine_eye_masks.py`.
- The exact current call sites of `mark_downstream_eye_mask_runs_stale`. The plan claims
  both `tune/keypoint_failure_review.py` AND `utils/patch_keypoints_from_crops.py` (the
  latter was a late correction). Confirm both, and run a fresh grep to find any the trace
  missed.
- Every file/symbol in the Phase 1 edit list below still exists where claimed.

Report any discrepancy BEFORE proceeding. **If the crux claim does not hold against current
code, STOP and surface it — the whole plan depends on it.**

## Step 1 — Execute Phase 1 only (re-point; behavior-neutral)

Goal: make eye-angle analysis (and its satellites) read eye geometry exclusively from
subject-mask channels, so the legacy producers become dead code. Do NOT delete any files in
this phase — deletion is Phases 2–4 and is out of scope.

Re-point / strip the legacy `EYE_GEOMETRY_STAGE_REFINED_EYE` fallback in:
- `analysis/eye_angle_analysis.py` — drop the legacy import + `refined_eye_run=` arg.
- `shared/eye_geometry_source.py` — remove `_build_refined_eye_source`, the
  `EYE_GEOMETRY_STAGE_REFINED_EYE` constant, and the fallback branch; resolver becomes
  `subject_shape → refined_subject`.
- `visualization/visualize_eye_angle_overlays.py`.
- `tune/keypoint_failure_review.py` AND `utils/patch_keypoints_from_crops.py` — re-point
  `mark_downstream_eye_mask_runs_stale` to the subject-mask stale marker.
- `utils/resolve_eye_mask_stale.py`, `utils/apply_tuning_by_camera.py`,
  `utils/audit_refined_mask_metrics.py`, `utils/materialize_refined_eye_masks_compat.py`,
  `tune/dispatcher.py` (drop the eye-mask-review deprecation shim).

## Hard constraints

- Phase 1 must be behavior-neutral for any recording that already has refined-subject eye
  geometry. If you cannot make a change behavior-neutral, flag it instead of forcing it.
- Do NOT touch registry migrations, drop tables, or delete files in this phase.
- Do NOT "fix" eye-mask contracts/models — they are being removed, not repaired.
- You are removing the legacy CODE PATH, not legacy data. Call out explicitly if any change
  would break angle computation for a recording that has only legacy
  `refined_eye_masks_runs` and no subject eye geometry (this is the Phase 0 coverage gap).

## Verify your work

- Run the existing eye-angle / eye-geometry tests (under `tests/unit/fisheye/`; grep for
  `eye_angle`, `eye_geometry`, `subject_eye`). Report pass/fail with output.
- Confirm no remaining import of `EYE_GEOMETRY_STAGE_REFINED_EYE` survives (grep).
- Show that `resolve_eye_geometry_source` no longer accepts/uses `refined_eye_run`.
- Report honestly: what you changed, what you verified, what you skipped, and any doc claim
  that turned out to be stale.

## Hand-off out

When done, leave a short note for the Phase 2 agent (delete-now): which of the 36 delete-now
files are now confirmed to have zero remaining importers (you just severed the last live
consumer), and any that still have a surprise reference.
