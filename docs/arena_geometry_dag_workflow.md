# Arena-Geometry DAG Workflow

## Boundary

Arena geometry is a recording-level artifact. It is estimated from the native
camera stream, represented either by one whole-recording video or by one
indexed rolling-clip collection, and then may be consumed by either a clipped
or a whole-recording detection/refinement workflow. Clipped and whole
production do not publish different geometry schemas or duplicate the fit per
clip.

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
             explicit comparison / selection
                          |
             explicit keyed detection gating
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

For a whole-video source, the GPU probe reads the exact native video, external
summary, and keyframe declaration. For a clipped source, it reads
`recording_clip_index.json`, the recording geometry snapshot, and the indexed
per-clip keyframe declarations. It validates that the collection contains one
camera stream with a dense continuous recording-frame clock, consistent native
shape and frame rate, completed clips, and recording-bound source paths.

The GPU probe then:

1. selects an odd, bounded subset of declared keyframes in early, middle, and
   late five-second windows;
2. maps clipped samples from the continuous recording-frame clock into their
   owning clip-local frames;
3. seeks directly to each declared keyframe and retains exact-seek proof;
4. builds one temporal median and independent circle fit per window;
5. freezes the fit report before reading optional acquisition geometry;
6. produces reveal panels, a three-panel montage, and a receipt binding all
   relevant bytes; and
7. atomically publishes the completed review-package directory.

The fitter does not traverse the video from frame zero and does not open or
modify the analysis Zarr or registry. The standalone
`submit_recording_dish_rim_probe_bsub.sh` wrapper imports a successful review
package into immutable, selector-ineligible
`analysis/arena_geometry_fit_runs/<content-derived-run>` storage by default.
Pass `--diagnostic-only` to retain only the external review package. Default
persistence requires one explicit `--analysis-zarr`, or exactly one
`zarr/*_analysis.zarr` below a clipped `--recording-dir`; missing or ambiguous
targets fail before submission.

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

For an organized rolling-clip recording, omit `video`, `summary`, and
`keyframes`. The recording directory itself is the probe source, and must
contain `recording_clip_index.json` plus the geometry and per-clip keyframe
metadata referenced by that index:

```json
{
  "schema": "palette.arena_geometry_review_targets.v1",
  "targets": [
    {
      "target_id": "2026_08_06_19_13_35_cam2010093",
      "recording_id": "2026_08_06_19_13_35_cam2010093",
      "recording_dir": "/groups/.../2026_08_06_19_13_35_cam2010093",
      "analysis_zarr": "/groups/.../zarr/2026_08_06_19_13_35_cam2010093_analysis.zarr",
      "geometry_source": "producer-folder",
      "geometry_camera_serial": "2010093",
      "geometry_arena_id": "arena_1"
    }
  ]
}
```

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
`--probe-array-concurrency`. The probe queue is explicit as `gpu_l4` or
`gpu_t4`. A serial T4 canary uses `--probe-queue gpu_t4
--probe-array-concurrency 1`:

```bash
scripts/py -m fisheye.cluster.arena_geometry_campaign \
  --manifest targets.json \
  --run-label batman_geometry_canary \
  --run-root /groups/.../arena_geometry_campaigns/batman_geometry_canary \
  --repo /groups/.../palette-worktrees/<exact-commit> \
  --probe-queue gpu_t4 \
  --probe-array-concurrency 1 \
  --apply --json
```

The immutable bundle contains `plan.json`, `lsf_plan.json`, per-job runtime
status, scheduler logs, and `lsf_submission.json` after submission. A final
serialized short-queue job refreshes the exact target rows in the Palette
registry after both arrays succeed.

## Human review and later publication

The node-local review package contains:

- `fit_report.json`;
- early, middle, and late temporal medians and fit overlays;
- optional acquisition-reveal overlays;
- `dish_rim_review_montage.png`; and
- `review_package.json`.

The probe task immediately imports the complete package into immutable,
selector-ineligible
`analysis/arena_geometry_fit_runs/<content-derived-run>` storage, then deletes
node scratch. The embedded run contains the report, receipt, optional reveal,
montage, and three source panels. A reviewer must inspect those embedded
artifacts before the separate
`build_reviewed_arena_geometry_candidate_fragment` is used. That fragment
binds the exact fit-review run and digest. Publishing the reviewed candidate
still does not select it or apply it to detections.

For a clipped fit, reviewed-candidate publication validates the singleton
camera and continuous frame range against the analysis Zarr, rehashes the
current `recording_clip_index.json` and Orange recording snapshot, and retains
clip-local plus one-based recording-frame coordinates for every decoded frame.
Its camera-native coordinate binding is the exact camera frame in the hashed
recording snapshot; it does not synthesize a single-video metadata record for
the collection. `rig_id`, `canvas_name`, and `arena_id` are read from the Zarr
when present, or must be supplied together at approval time. The same path
handles future single-camera rolling-clip recordings without recording-specific
camera or clip-count constants.

Operational selection and keyed detection-gate materialization exist as
separate explicit workflow modules. The campaign registry refresh reports
offline fit completion and review-pending state; selection and gate completion
remain absent. None of those operational steps may be silently appended below
the pre-review campaign.

The implementation keeps the lifecycle boundary visible in code:

- `fisheye.cluster.arena_geometry_review` owns probe, review-package, and
  reviewed-candidate fragments;
- `fisheye.cluster.arena_geometry` owns explicit operational selection and
  keyed detection-gate fragments; and
- `fisheye.cluster.arena_geometry_campaign` submits only the pre-review arrays.
