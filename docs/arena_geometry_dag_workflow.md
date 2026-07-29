# Arena-Geometry DAG Workflow

## Boundary

Arena geometry is a recording-level artifact. It is estimated from the native
whole-recording video and then may be consumed by either a clipped or a
whole-recording detection/refinement workflow. Clipped and whole production do
not publish different geometry schemas or duplicate the fit per clip.

The workflow is intentionally divided by a human-review barrier:

```text
successful import / registered analysis target
             |                         |
             v                         v
  acquisition candidate       blind keyframe-only probe
  atomic, pointerless          early / middle / late
             |                         |
             +------------+------------+
                          v
              hash-bound review package
              status: awaiting review
                          |
                    HUMAN REVIEW
                          |
                          v
             reviewed Palette candidate
             atomic and pointerless
                          |
             future comparison / selection
                          |
             future keyed detection gating
```

The pre-review campaign schedules only the two jobs above the human-review
line. It does not schedule reviewed-candidate publication, comparison,
selection, detection gating, or a registry mutation.

## Pre-review jobs

For every target, the acquisition and probe tasks are siblings. They may share
the same upstream import dependency, but neither depends on the other. A
production campaign packs all acquisition tasks into one bounded LSF array and
all GPU probes into a second bounded LSF array. Even a one-recording canary uses
the same array envelope. Each acquisition array element receives a distinct
`<job-id>_<array-index>` node-local scratch root.

The acquisition job calls the existing atomic candidate publisher. It builds a
metadata-only Zarr v3 run in node-local scratch, validates it, copies it to a
hidden sibling of the final destination, verifies the copy, atomically renames
it, and completes it without setting `latest` or `latest_complete`.

The GPU probe:

1. reads the exact native video, external summary, and keyframe declaration;
2. checks that summary and keyframe frame counts and frame rates agree;
3. selects an odd, bounded subset of declared keyframes in early, middle, and
   late five-second windows;
4. seeks directly to each declared keyframe and retains exact-seek proof;
5. builds one temporal median and independent circle fit per window;
6. freezes the fit report before reading optional acquisition geometry;
7. produces reveal panels, a three-panel montage, and a receipt binding all
   relevant bytes; and
8. atomically publishes the completed review-package directory.

The probe does not traverse the video from frame zero and does not open or
modify the analysis Zarr or registry.

## Target manifest

The pre-review campaign requires explicit recording-bound paths. It never
searches for a current/latest geometry artifact or substitutes another camera.

```json
{
  "schema": "palette.arena_geometry_review_targets.v1",
  "targets": [
    {
      "target_id": "batman_arena_2_cam2010094",
      "recording_id": "2026-07-22T15-44-40Z_arena_2_Batman",
      "recording_dir": "/groups/.../2026-07-22T15-44-40Z_arena_2_Batman",
      "analysis_zarr": "/groups/.../zarr/2026-07-22T15-44-40Z_arena_2_Batman_analysis.zarr",
      "video": "/groups/.../cams/Cam2010094_2026-07-22T15-44-40Z_arena_2.mp4",
      "summary": "/groups/.../cams/Cam2010094_2026-07-22T15-44-40Z_arena_2_external_summary.json",
      "keyframes": "/groups/.../cams/Cam2010094_2026-07-22T15-44-40Z_arena_2_keyframe.json",
      "recovery_receipt": "/groups/.../raw/recording_geometry_recovery.json",
      "acquisition_observation": "/groups/.../raw/recording_geometry_bundle/recording_geometry_assets/cameras/Cam2010094/daily_registration/rim_observation/observation.json"
    }
  ]
}
```

All source paths and the analysis Zarr must resolve beneath the declared
recording directory. The analysis Zarr must also match the registry row for the
recording. The Palette checkout must be clean, and the exact commit is frozen
in the plan.

Dry run:

```bash
scripts/py -m fisheye.cluster.arena_geometry_campaign \
  --manifest targets.json \
  --run-label batman_geometry_canary \
  --run-root /groups/.../arena_geometry_campaigns/batman_geometry_canary \
  --repo /groups/.../palette-worktrees/<exact-commit> \
  --dry-run --json
```

Submission uses the Citrus login poller by default. Array concurrency is
explicit (`8` acquisition tasks and `4` GPU probes by default) and can be
changed with `--acquisition-array-concurrency` and
`--probe-array-concurrency`:

```bash
scripts/py -m fisheye.cluster.arena_geometry_campaign \
  --manifest targets.json \
  --run-label batman_geometry_canary \
  --run-root /groups/.../arena_geometry_campaigns/batman_geometry_canary \
  --repo /groups/.../palette-worktrees/<exact-commit> \
  --apply --json
```

The immutable bundle contains `plan.json`, `lsf_plan.json`, per-job runtime
status, scheduler logs, and `lsf_submission.json` after submission.

## Human review and later publication

The review package contains:

- `fit_report.json`;
- early, middle, and late temporal medians and fit overlays;
- optional acquisition-reveal overlays;
- `dish_rim_review_montage.png`; and
- `review_package.json`.

A reviewer must inspect the montage before the separate
`build_reviewed_arena_geometry_candidate_fragment` is used. That fragment
recomputes the fit-report and montage digests from the receipt before planning
publication. A changed file fails closed. Publishing the reviewed candidate
still does not select it or apply it to detections.

Comparison, operational selection, keyed detection-gate materialization, and
registry projection remain later explicit workflow modules. They must not be
silently appended below the pre-review campaign.
