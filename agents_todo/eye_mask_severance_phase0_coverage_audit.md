# Implementation Handoff: Eye-Mask Stage Severance — Phase 0 (coverage audit)

You are running the GATING audit for a planned removal of the legacy `eye_masks` /
`refined_eye_masks` pipeline stages. **This phase is mostly investigation, not code edits.**
Its output is a GO / NO-GO report that every later phase depends on. Removing the legacy
fallback is only safe if the data it falls back to (subject-mask eye geometry) actually
exists for the recordings that matter. Your job is to prove that — or find where it doesn't.

## Read first

1. `docs/diagnostics/eye_mask_severance_plan_2026-05-28.md` — focus on §1 (the quality
   caveats), §2 Phase 0, and §4 risks. This phase exists to close those risks.
2. Skim `docs/diagnostics/contract_drift_audit_2026-05-28.md` for context on lineage attrs.

These docs are point-in-time analyses; verify claims against current code and current data.

## Why this matters (the one real risk)

Eye-angle analysis will be re-pointed from legacy `refined_eye_masks_runs` to subject-mask
eye channels (`refined_subject_masks_runs/<run>/components/eye_left|eye_right/geometry/`).
The geometry method is identical, so where the subject channels exist, output is equivalent.
**The danger is silent loss:** a recording that has legacy eye masks but NO populated
subject eye geometry will lose its eye-angle source the moment the fallback is removed — with
no error, just missing/empty results.

## Tasks (produce a written report; do not edit pipeline code)

1. **Coverage census across the zarr stores.** For every recording zarr you care about,
   determine whether `refined_subject_masks_runs/<run>/components/eye_left/geometry/ellipse_params`
   (and `eye_right`, and `ellipse_success`) is present and non-empty. Produce a table:
   recording → has-subject-eye-geometry (yes/no) → has-legacy-refined-eye-masks (yes/no).
   The dangerous bucket is **legacy-yes / subject-no**.
   - Confirm the exact array paths against current code (`shared/refined_subject_eye_geometry.py`,
     `shared/eye_geometry_source.py`) before scanning — don't trust the paths above blindly.
   - Prefer read-only zarr opens. Do NOT write to recording stores in this phase.

2. **Backfill assessment.** For any recording in the dangerous bucket, confirm whether
   `utils/backfill_subject_mask_runs.py` can project its legacy eye masks into
   subject_mask_runs (read its source-stage options). Produce the exact backfill command(s)
   that would close the gap — but do NOT run them against real data without the owner's
   explicit go-ahead. List what you would run.

3. **Registry enumeration.** Find how many `.sqlite` registry databases are deployed / in
   use (search configs, runbooks, env, common paths). Phase 4 drops eye-mask tables via
   migrations, and dropping migration bodies before the drop-migration has run on EVERY
   registry breaks history replay. The owner needs the count and locations.

4. **Off-repo / compat consumer check.** `utils/materialize_refined_eye_masks_compat.py`
   and `utils/refined_eye_masks_compat.py` synthesize a `refined_eye_masks_runs`
   compatibility view from subject masks. Search the repo (and ask the owner about external
   notebooks / downstream tooling) for anything that reads that compat group. Deleting it
   later has blast radius this trace cannot see from inside the repo.

5. **Source-of-truth check.** The resolver prefers `subject_shape_runs` first, then
   `refined_subject_masks_runs`. Determine which is actually materialized for your data, so
   the owner knows which source their re-pointed numbers will come from. Not a blocker (both
   are parity), but it should be stated, not assumed.

## Deliverable: GO / NO-GO report

Produce a single report with:
- The coverage table and the count/list of recordings in the dangerous bucket.
- A clear verdict: GO (subject eye geometry covers everything that matters), GO-AFTER-BACKFILL
  (with the exact backfill plan), or NO-GO (with what's missing).
- Registry count + locations.
- Any off-repo consumer found.
- Which source (`subject_shape` vs `refined_subject`) is the effective one for the data.

Be unflinching: if coverage is partial, say so plainly and quantify it. A confident "GO" on
incomplete evidence is the worst outcome here — it converts a known risk into a silent one.

## Hand-off out

If GO (or after backfill completes), Phase 1 (`eye_mask_severance_phase1_repoint.md`) may
proceed. Attach your coverage table so the Phase 1 agent can cite it when asserting
behavior-neutrality.
