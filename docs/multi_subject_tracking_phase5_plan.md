# Multi-Subject Tracking Phase 5 And Four-Well Canary Plan

<!-- design-meta
status: ready-for-canary
created: 2026-07-09
owner: jeremy
depends_on: docs/instance_track_subject_identity_contract.md
-->

## Purpose

Define the next tracking implementation slice and the first real-recording
canary. The proposed canary is one camera view containing four individual fish,
each confined to its own physical well. The recording has not yet been run
through detection, refined detection, cropping, arena assignment, keypoints, or
subject masks.

This is an unusually useful first canary because it exercises four simultaneous
instances and the complete row-lineage pipeline while retaining deterministic
identity ground truth: a fish cannot move between wells.

## What This Canary Proves

The four-well recording can validate:

- full-frame detection of multiple simultaneous fish;
- sparse refined-detect support for multiple rows per frame;
- `instance_key` propagation through crop, keypoints, and subject masks;
- four explicit arena definitions and row-level arena assignment;
- creation of four run-local tracks by `single_subject_per_arena`;
- keyed joins and source-rowset fingerprint checks;
- per-well keypoint and mask coverage;
- track-kinematics consumption of the exact crop-backed tracking run;
- registry projection and freshness for the resulting tracking artifacts.

It is also a good operational test for UI/review assumptions that accidentally
retain one-row-per-frame behavior.

## What This Canary Does Not Prove

It does not validate a true interacting-subject tracker. Physical wells remove
the hard temporal-association cases:

- crossings;
- shared pixels and mask overlap;
- prolonged occlusion;
- ambiguous reappearance;
- identity swaps within one arena.

Passing this canary establishes the multi-instance substrate and strict spatial
baseline. A later synthetic crossing fixture and a real shared-arena recording
are still required for `multi_subject_motion_v1`.

Arena identity is also recording-local. A well/track label must not be promoted
to cross-recording `subject_id` unless acquisition metadata supplies that
biological identity.

## Safety And Authority Rules

- Preserve raw video and raw detector outputs as immutable provenance.
- Create new runs for each canary attempt; do not overwrite a reviewed run.
- Resolve exact run names at every handoff; do not let a later smoke run become
  an implicit source through `latest`.
- Do not approve or promote keypoints/masks until per-well visual review passes.
- Use the unified `subject_mask_runs` / `refined_subject_masks_runs` path. Do not
  create a new eye-mask-primary workflow.
- Bind arena assignment and tracking to the exact `crop_runs/<run>` rowset used
  by keypoints and masks.
- Keep `single_subject_per_arena` as the correct canary tracker. Do not use the
  canary to claim that motion association is implemented.

## Canary Inputs To Record

Before execution, fill in:

```text
recording_id:
analysis_zarr:
registry:
camera_id:
frame_count:
fps:
representative_start/middle/end frames:
detection model/run:
keypoint model/run:
subject-mask model/run:
well layout and stable arena IDs:
```

Use arena IDs that describe stable spatial wells. Their numeric values do not
need to equal track IDs.

## Execution Plan

### Gate 0: Recording And Metadata Preflight

Confirm that the organized analysis Zarr points at the intended video and that
frame dimensions/count/FPS are credible. Confirm the experiment setup before
running inference:

```bash
ZARR=/path/to/four_well_analysis.zarr
REGISTRY=/nvme1/palette_registry.sqlite

scripts/py -m fisheye.utils.setup_experiment_metadata "$ZARR" --show
```

The intended setup is:

```text
setup_type = multi_dish
num_dishes = 4
fish_per_dish = 1
total_expected_fish = 4
```

If metadata is absent, write it explicitly:

```bash
scripts/py -m fisheye.utils.setup_experiment_metadata \
  "$ZARR" \
  --num-dishes 4 \
  --fish-per-dish 1
```

Use `--force` only after confirming that existing metadata is wrong.

Acceptance:

- the source video and analysis archive are the intended recording;
- experiment metadata declares four wells and one fish per well;
- no processing run is treated as authoritative yet.

### Gate 1: Define And Review Four Arenas

Draw one non-overlapping rectangle per physical well:

```bash
scripts/py -m fisheye.tune.subdish_mask_tuner "$ZARR"
```

Inspect the first, middle, and last portions of the recording. Each fish centroid
must remain inside exactly one well ROI, and the ROI must not include a neighbor.

Acceptance:

- exactly four arena definitions exist;
- arena rectangles do not overlap;
- well boundaries remain valid across the recording;
- every physical well has one stable arena ID.

### Gate 2: Detect, Inspect, And Refine Multiple Instances

Resolve first, then apply detection:

```bash
palette detect "$ZARR" --registry "$REGISTRY" --json
palette detect "$ZARR" --registry "$REGISTRY" --json --apply
```

Run detection quality and sparse refinement against the exact detect run:

```bash
scripts/py -m fisheye.refinement.detect_quality "$ZARR" --run <detect_run>
scripts/py -m fisheye.refinement.refine_detect \
  "$ZARR" \
  --config configs/fisheye/default.yaml
```

Review representative frames before continuing. The important measurement is
per-well occupancy, not merely total detections per frame.

Acceptance:

- sampled clear frames normally contain four valid refined instances;
- each instance center falls into a different well;
- `refined_detect_runs/<run>/instances/instance_key` is unique;
- multiple rows sharing one frame remain distinct;
- duplicate boxes and cross-well boxes are absent or explicitly rejected;
- missed detections are quantified per well.

Known risk: manual add-row/review paths still contain compatibility limits in
some one-row-per-frame surfaces. A detector miss in a frame that already has
other fish may therefore reveal a review-tool limitation. Record that as a
canary finding rather than bypassing row identity or silently interpolating it.

### Gate 3: Build The Exact Crop Rowset

Resolve and apply crop generation from the selected refined instances:

```bash
palette crop "$ZARR" --json
palette crop "$ZARR" --json --apply
```

Record the exact `<crop_run>`. This rowset becomes the shared observation axis
for arena assignment, tracking, keypoints, masks, and track kinematics.

Acceptance:

- crop row count equals the selected refined-instance row count;
- crop `instance_key` values equal the refined source key set;
- each crop contains only the fish from its assigned well in sampled frames;
- crop/source rowset fingerprint status is complete.

### Gate 4: Arena Assignment And Four-Track Baseline

Run arena assignment against the exact crop rowset:

```bash
scripts/py -m fisheye.tracking.arena_assignment \
  "$ZARR" \
  --source-rowset "crop_runs/<crop_run>" \
  --tracking-method single_subject_per_arena
```

Acceptance:

- `arena_assignment_runs/<run>` and `tracking_runs/<run>` reference the exact
  crop path and the same complete rowset fingerprint;
- `tracking_identity_mode == "instance_key"`;
- four occupied arenas produce four run-local tracks;
- every assigned row preserves its `instance_key` and optional refined/raw row
  lineage;
- no frame contains two accepted observations in the same arena;
- unassigned rows are zero, or any nonzero count is reviewed and explained;
- registry tracking status is `OK` or an understood unassigned-row warning, not
  stale.

Missing observations in a well do not create an identity switch: the strict
tracker retains one run-local track per occupied arena. They do reduce coverage
and must remain visible as a quality metric.

### Gate 5: Keypoints On The Same Crop Rows

Resolve and apply the registry-selected keypoint model:

```bash
palette keypoints "$ZARR" --registry "$REGISTRY" --json
palette keypoints "$ZARR" --registry "$REGISTRY" --json --apply

scripts/py -m fisheye.refinement.refine_keypoints \
  "$ZARR" \
  --config configs/fisheye/default.yaml
```

Acceptance:

- keypoint rows use the exact `<crop_run>`;
- keypoint `instance_key` values match the crop key set;
- success/usable rates are reported per well/track, not only globally;
- sampled eyes, swim bladder, heading, and left/right assignments are visually
  plausible in all four wells;
- no well-specific orientation or illumination bias is hidden by aggregate
  success rates.

### Gate 6: Subject Masks On The Same Crop Rows

Run the current unified subject-mask model against the exact crop and refined
keypoint runs. Use the model-resolution and execution command from
`docs/operator_guide/pipeline_workflow.md`, recording explicit run names.

Acceptance:

- mask rows carry the same `instance_key` set as the crop/keypoint source;
- body, eyes union/left/right, and swim bladder are evaluated per well;
- no mask claims pixels from a neighboring well or fish;
- dense `masks_roi` is present on refined editable outputs;
- failed/pending rows remain explicit and are not treated as valid empty masks;
- no refined mask run is approved before sampled visual review.

### Gate 7: Track Kinematics And Registry Audit

Run track kinematics only after the exact crop-backed tracking run and refined
keypoint run are selected:

```bash
scripts/py -m fisheye.analysis.track_kinematics \
  "$ZARR" \
  --smooth-seconds 1.0

scripts/py -m fisheye.utils.check_recording_steps "$ZARR"
```

Acceptance:

- track kinematics resolves the keyed tracking run without positional fallback;
- output contains four tracks with stable arena metadata;
- no keyed/fingerprint mismatch occurs;
- per-track frame sequences stay within their physical wells;
- speed/heading traces have no cross-well teleport signatures;
- registry rows expose the tracking identity mode and rowset fingerprint and do
  not report stale tracking lineage.

## Canary Evidence Bundle

Preserve a small review bundle containing:

- exact run names and git commit;
- experiment setup and arena definitions;
- row counts and unique `instance_key` counts at every stage;
- per-frame refined detection-count distribution;
- per-well assigned/unassigned/duplicate counts;
- per-well keypoint success and usability;
- per-well mask component coverage and review state;
- tracking source path, identity mode, and rowset fingerprint;
- representative overlays from start/middle/end and any failure frames;
- track-kinematics plots for all four tracks;
- registry/check-recording-steps output.

Do not collapse the result to a single pass/fail number. A four-well aggregate
can look healthy while one camera corner or well systematically fails.

## Phase 5 Evaluation Ladder

Use three increasingly difficult tiers:

1. **Four-well real canary:** validates multi-instance storage, spatial
   assignment, keyed lineage, and downstream row alignment.
2. **Synthetic shared-arena fixtures:** deterministic crossings, missed
   detections, occlusion, birth/death, and reappearance with known truth.
3. **Real shared-arena recording:** validates motion association and exposes
   domain effects absent from synthetic data.

The four-well canary should run before implementing the new tracker. It gives a
real baseline and may expose upstream multi-instance limitations that a motion
algorithm cannot fix.

## `multi_subject_motion_v1` Implementation Slice

After the four-well substrate passes, add the first true temporal method behind
the existing `TrackingObservations` / `TrackingResult` API.

Initial algorithm:

- constant-velocity state prediction;
- Hungarian assignment using `scipy.optimize.linear_sum_assignment`;
- centroid-distance, bounding-box IoU, heading disagreement, and size-change
  costs;
- arena transition as a configurable gate/penalty, not identity;
- explicit birth, missed-frame tolerance, termination, and reactivation;
- ambiguity abstention instead of forced assignment;
- row-level `tracking_confidence`, `tracking_status`, and `association_cost`.

Initial metrics:

- identity switches;
- assignment accuracy;
- track fragmentation;
- false track births;
- missed/ambiguous rate;
- deterministic replay;
- runtime and peak memory.

Defer mask IoU and appearance embeddings until the baseline shows which failure
modes they would address. SLEAP integration remains an evaluation adapter, not
a canonical storage rewrite or required first runtime dependency.

## Promotion Decision

Promote the four-well recording from canary to a permanent regression fixture
only when:

- the evidence bundle is complete;
- all four wells have reviewed detection/keypoint/mask coverage;
- keyed tracking and registry freshness pass;
- any manual-add or UI limitations are either fixed or explicitly retained as
  known blockers;
- the chosen fixture subset can be used legally and practically in automated or
  operator-run regression testing.

Passing this recording authorizes work on `multi_subject_motion_v1`; it does not
by itself validate interacting-subject identity.
