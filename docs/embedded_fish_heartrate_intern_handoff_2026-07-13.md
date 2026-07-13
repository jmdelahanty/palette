# Embedded-Fish Heart-Rate Trial Handoff - 2026-07-13
<!-- contract-meta
status: handoff
last_updated: 2026-07-13
-->

## Goal

Evaluate whether the fixed-camera heart photometry result from embedded Trial 1
generalizes across all 48 supplied trials. Use the user-drawn side-camera
chamber regions and a frozen compact mask in the top-camera ROI. Do not learn a
new mask, select a new frequency band, or tune smoothing separately for each
trial.

This is an evaluation project, not a request to make every trial succeed.
Failed geometry, weak signals, and no-estimate trials must remain in the final
table.

Read the final interpretation before starting:

- [`heartrate_final_decision_2026-07-12.md`](heartrate_final_decision_2026-07-12.md)
- [`heartrate_top_view_mask_decision_2026-07-11.md`](heartrate_top_view_mask_decision_2026-07-11.md)
- [`playgrounds/heartrate_stabilization/README.md`](../playgrounds/heartrate_stabilization/README.md)

## Data And Frozen Inputs

Repository:

```text
/home/delahantyj@hhmi.org/gitrepos/palette
```

Dataset root:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/
  embedded_fish_positive_control/
```

The dataset contains paired videos for Trials 1--48:

```text
Top_Camera/20250109_F1_10_30_Trial{N}.mp4
Side_Camera/20250109_F1_10_30_heart_{N}.avi
```

Supplied evaluation workbook:

```text
Bradyinfo_20250109_F1_10_30.xlsx
```

### Side-camera primary surface

Use the two polygons in:

```text
side_camera_chambers_trial1.json
```

SHA-256:

```text
a09dce19af3e4a20426ed437faf9ef1ad4761129bdecc1af253c5dc1bf37302b
```

The polygons are fixed source-frame coordinates on `512x512` side-camera
frames. Their anatomical identities are intentionally unassigned.

- Primary side trace: literal Mono8 mean over `chamber_a OR chamber_b`.
- Controls: separate literal means for `chamber_a` and `chamber_b`.
- Do not choose the cleaner chamber separately in each trial.
- Do not redraw the polygons after viewing a supplied rate trace.

The broader `side_camera_roi_trial1.json` and the colleague ROI are optional QC
controls, not the primary surface in this handoff.

### Top-camera ROI

Use the geometry from:

```text
top_camera_roi_trial1.json
```

SHA-256:

```text
b84342ca11bb3cdda0396cc9d2d1cce6bc50c6c4fbe86e4f9493320306d824e9
```

The fixed source-frame ROI is:

```text
frame shape: 500x544 (height x width)
ROI xywh:    261, 77, 22, 28
```

Treat this JSON as a geometry template. Do not rewrite its `source_video` or
fingerprint fields for other trials. The batch runner should accept the target
video separately and record both the template and target paths.

### Meaning of “frozen bottom mask”

For this handoff, **frozen bottom mask** means the `79` ROI-local pixels in the
intersection of the two embedded Trial 1 cross-fit masks. The source artifact
is:

```text
inspection_videos/
  Trial1_top_camera_crossfit_mask_projection_comparison.arrays.npz
```

SHA-256:

```text
8203b5e63cb93fc36d26acb5aa4b56b20c5df91386e0092a3c197938d632f99e
```

Load `fold_masks`, which has shape `(2, 28, 22)`. The masks contain `93` and
`85` pixels; their intersection contains `79` pixels.

- Primary top mask: `fold_masks[0] & fold_masks[1]` (`79` pixels).
- Sensitivity controls: the original `93`- and `85`-pixel masks separately.
- Never select among these masks per trial.
- Never use the freely moving-fish `lower_mask`; it is in an unrelated
  anatomical coordinate system.
- Never resize or warp a `28x22` mask to repair a mismatched ROI.

Materialize the `79`-pixel mask once into the batch output with the source hash,
derivation, pixel count, and ROI geometry. This derived file becomes the frozen
batch mask.

## Frozen Analysis Contract

Use these parameters for every trial:

| Item | Frozen value |
|---|---|
| Frequency band | `1.5..4.0 Hz` |
| Top source rate | Verify; expected approximately `200 fps` |
| Side source rate | Verify; expected approximately `100 fps` |
| Supplied reference rate | `100 fps` |
| Detrending | Linear, separately for each trace or pixel |
| Temporal filter | Third-order zero-phase Butterworth bandpass |
| Filter edge rejection | `0.75 s` at each usable segment edge |
| Primary ridge | `4 s` window, `1 s` step |
| Secondary ridge | `8 s` window, `1 s` step |
| Event polarity | Freeze to `-1` for primary side union and top mean |
| Event prominence | `0.5` robust MAD, matching Trial 1 |
| Gap policy | Never compress time or filter through a decode/validity gap |
| Smoothing | None for metrics; display-only smoothing must be labeled |

Side primary signal:

1. Decode the original Mono8-equivalent grayscale frames.
2. Average pixels in the fixed chamber union for each frame.
3. Detrend the resulting one-dimensional trace.
4. Bandpass it using the frozen contract.

Top primary signal:

1. Extract the fixed `22x28` ROI from each original top-camera frame.
2. Detrend and robust-MAD-scale each ROI pixel using only that trial's image
   data, never the supplied rate reference.
3. Bandpass each scaled pixel trace.
4. Average the frozen `79` mask pixels with equal positive weights.

Per-trial robust normalization is allowed because its formula is frozen and it
does not inspect the supplied outcome. Pixel selection, spatial weights, band,
polarity, windows, and thresholds are not allowed to vary by trial.

## Two-Pass Workflow

### Pass A: inventory, geometry QC, and signal extraction

Pass A must not open the workbook or any supplied heart-rate/bradycardia plot.

1. Inventory all Trials 1--48 and require one top and one side video per trial.
2. Record codec, frame shape, decoded frame count, FPS, duration, duplicate
   fraction, and timestamp assumptions.
3. Generate outcome-blind contact sheets showing the side polygons and top ROI
   plus all three frozen top masks for every trial.
4. Confirm that the fixed geometry still covers the same visible anatomy.
5. Extract and save raw and filtered side/top traces.
6. Save explicit failure reasons and keep failed trials in the manifest.

If geometry drifts, do not manually move an ROI after looking at rate results.
Either mark the trial unusable or propose a single image-only registration rule
that can be frozen before opening the workbook. Dynamic time warping or
reference-guided spatial alignment is prohibited.

### Pass B: outcome evaluation

Only after all Pass A artifacts are written and hashed:

1. Load workbook sheet `heart_rate_trace`; Trial `N` is row `N + 1`.
2. Compare the primary side union, primary top intersection, and top fold-mask
   controls against the supplied reference.
3. Compare the top primary trace against the independently extracted side
   primary trace without shifting either timebase to improve agreement.
4. Report both `4 s` and `8 s` ridge results.
5. Report event-interval concordance as automated signal concordance, not
   manual beat validation.
6. Keep every trial in aggregate results, including no-estimate trials.

The workbook's bradycardia bouts may be summarized descriptively. Do not make a
stimulus-response endpoint primary until authoritative stimulus timing and the
colleague's event/smoothing notebook are available.

## Required Outputs

Use a new output directory such as:

```text
playgrounds/heartrate_stabilization/outputs/
  embedded_positive_control_trials_frozen_v1/
```

Required files:

```text
frozen_contract.json
manifest.csv
preflight.summary.json
preflight_contact_sheet.png
trial_01/
  extraction.summary.json
  signals.npz
  qc.png
  evaluation.summary.json
...
trial_48/
aggregate.csv
aggregate.summary.json
aggregate_rate_metrics.png
failures.csv
```

Every trial row should include:

- paths and file fingerprints;
- source FPS, frame count, duration, and duplicate fraction;
- geometry status and failure reason;
- side/top coverage and event counts;
- `4 s` and `8 s` ridge correlation and MAE against the supplied reference;
- top-versus-side ridge correlation and MAE;
- event-interval MAE and interval CV, clearly labeled as automated;
- top fold-mask agreement;
- no-estimate status rather than fabricated or interpolated results.

Aggregate reporting must show distributions and coverage across all 48 trials,
not only the successful subset. Report Trial 1 separately as development data
and Trials 2--48 as the frozen evaluation set.

## Implementation Work Expected

The current scripts reproduce Trial 1 but do not yet provide this complete
cross-trial batch contract. Implement a new playground runner rather than
editing source annotations or invoking the Trial 1 scripts 48 times with
mutated JSON files. A reasonable target is:

```text
playgrounds/heartrate_stabilization/analyze_embedded_trials_batch.py
```

Recommended CLI shape:

```bash
scripts/py playgrounds/heartrate_stabilization/analyze_embedded_trials_batch.py \
  --dataset-root /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control \
  --side-chambers-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/side_camera_chambers_trial1.json \
  --top-roi-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --top-mask-arrays /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/inspection_videos/Trial1_top_camera_crossfit_mask_projection_comparison.arrays.npz \
  --trials 1-48 \
  --phase extract \
  --output-dir playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trials_frozen_v1
```

Then evaluate the already saved signals:

```bash
scripts/py playgrounds/heartrate_stabilization/analyze_embedded_trials_batch.py \
  --dataset-root /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trials 1-48 \
  --phase evaluate \
  --output-dir playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trials_frozen_v1
```

The exact CLI may change during implementation, but the two-pass separation and
frozen contracts may not.

## Validation And Review Gates

Before the full run:

1. Reproduce the existing Trial 1 side and top results closely enough to
   explain any numerical difference.
2. Run Pass A on Trial 2 only and inspect its outcome-blind overlays.
3. Run Pass B on Trial 2 only and inspect its files and provenance.
4. Run focused tests outside the Codex sandbox, following `AGENTS.md`.
5. Run `git diff --check` and `scripts/py -m py_compile` on new scripts.

Required tests should cover:

- trial-number parsing and exact 1--48 inventory;
- mask derivation (`93`, `85`, and `79` pixels);
- ROI bounds and frame-shape rejection;
- no workbook access during extraction;
- no line/filter continuity across missing samples;
- correct `200 fps` top and `100 fps` side timebases;
- synthetic recovery of a known oscillator;
- inclusion of failed/no-estimate trials in aggregate tables;
- refusal to overwrite a frozen contract with different inputs.

Do not start all 48 trials until the human reviewer has inspected the Trial 2
contact sheet and smoke outputs.

## Interpretation Rules

Allowed conclusions after the batch run depend on the result:

- If Trials 2--48 show good coverage, low error, top/side agreement, and mask
  control agreement, report cross-trial support for fixed-camera window-level
  top-view cardiac-rate estimation.
- If only the side camera generalizes, report the side result and reject top-
  view generalization.
- If geometry or signal quality fails frequently, report coverage and failure
  modes rather than filtering those trials away.
- Never call automatically detected events validated beats without visible
  contraction annotations.
- Do not use this embedded result to rehabilitate the failed freely moving
  rate trajectory without a new synchronized moving-fish experiment.

## Prompt For The Intern's Language Model

Use the following prompt from the Palette repository root:

```text
You are working in /home/delahantyj@hhmi.org/gitrepos/palette.

Read and follow AGENTS.md, especially:
- use scripts/py for every Python command;
- do not install or mutate dependencies;
- run pytest outside the sandbox when required;
- preserve unrelated worktree changes;
- use apply_patch for manual edits.

Primary specification:
docs/embedded_fish_heartrate_intern_handoff_2026-07-13.md

Scientific context and reporting boundary:
docs/heartrate_final_decision_2026-07-12.md
docs/heartrate_top_view_mask_decision_2026-07-11.md
playgrounds/heartrate_stabilization/README.md

Objective:
Implement and validate the two-pass frozen embedded-fish Trial 1--48 analysis
specified in the intern handoff. Use the user-drawn side chamber union as the
primary side signal and the 79-pixel intersection of the two embedded Trial 1
fold masks as the primary top signal. Preserve the two original fold masks as
controls. Do not use the unrelated freely moving-fish lower mask.

Required behavior:
1. Inspect the existing Trial 1 scripts and tests before designing the batch
   runner. Reuse tested helpers where reasonable without weakening provenance.
2. Implement a new batch runner with separate extract and evaluate phases.
   Extraction must not open the supplied workbook or outcome plots.
3. Treat the Trial 1 ROI JSON files as geometry templates and target each video
   explicitly. Do not mutate the source annotations.
4. Freeze the 1.5--4.0 Hz band, 4 s primary/8 s secondary ridges with 1 s step,
   0.75 s filter edges, equal positive top-mask weights, and event polarity -1.
5. Produce a complete manifest, outcome-blind contact sheet, per-trial signals,
   per-trial summaries, aggregate metrics/plots, and explicit failure rows.
6. Keep Trial 1 labeled development and Trials 2--48 labeled evaluation.
7. Do not select a mask, chamber, band, polarity, alignment shift, smoother, or
   threshold separately per trial based on the supplied rate.
8. Add deterministic unit tests for inventory, masks, geometry, timebase, gaps,
   outcome separation, known-signal recovery, failures, and frozen provenance.
9. First reproduce Trial 1, then run an outcome-blind Trial 2 extraction smoke.
   Stop and report for human geometry review before launching Trials 1--48.
10. Run py_compile, git diff --check, and focused pytest. Report exact commands,
    outputs, limitations, and any deviations from the specification.

Important interpretation:
The supplied workbook is an evaluation reference, not mask-training data.
Automated side-camera events are not manual beat ground truth. Failed trials
must remain in aggregate coverage. Do not claim beat-to-beat validation, HRV,
or freely moving heart-rate recovery.

Do not commit or push unless the human explicitly asks. Do not launch the full
48-trial evaluation until the human approves the Trial 2 outcome-blind QC.
```

## Stop And Ask

Stop before proceeding if any of these occur:

- the human intended a different “bottom mask” than the `79`-pixel embedded
  fold-mask intersection;
- side or top geometry does not land on the same anatomy in Trial 2;
- video names, trial counts, shapes, or timebases differ materially;
- the workbook row mapping is inconsistent with Trial 1;
- reproducing Trial 1 requires changing a frozen parameter;
- extraction code needs to read the workbook;
- a full batch was requested before outcome-blind QC approval.
