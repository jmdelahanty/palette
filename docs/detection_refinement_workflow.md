# Detection Refinement Workflow

This is the current detect workflow as of 2026-04-07.

## Contract

- `detect_runs/<run>` is raw detector output.
- `detect_runs/<run>/quality_reports/<qrun>` is the raw detect artifact-label
  surface used by refine guardrails.
- `refined_detect_runs/<run>` is the canonical curated detect surface.
- Shared reads and writes should use `refined_detect_runs/<run>/instances`.
- `refined_detect_runs/<run>/source_detections` is the candidate-audit surface
  for the bound raw detect run.
- New work should not treat `filtered/` or `interpolated/` as the primary
  curated surface.

## Recommended Sequence

1. Run detection.
   - Blob:
     ```bash
     scripts/py -m fisheye.detection.detect_traditional /path/to/zarr
     ```
   - YOLO:
     ```bash
     scripts/py -m fisheye.detection.detect_yolo /path/to/zarr --model /path/to/model.pt
     ```
   - Sampled training Zarr:
     ```bash
     scripts/py -m fisheye.utils.predict_training_detections \
       /path/to/training.zarr \
       --registry /nvme1/palette_registry.sqlite \
       --model-run-id <registered_detect_run_id> \
       --run-name detect_seed_<model_or_date> \
       --apply
     ```
2. Run detect quality.
   ```bash
   scripts/py -m fisheye.refinement.detect_quality /path/to/zarr
   ```
   This labels raw-detect artifacts (`quality_flags`,
   `detection_quality_labels`) for the selected detect run. It is distinct from
   later refined-detect review approval. Skip this for sampled training Zarrs:
   `refine_detect` automatically uses sampled-import passthrough mode and does
   not require a detect-quality report there.

   For fixed multi-subject recordings, pass the expected total subject count:
   ```bash
   scripts/py -m fisheye.refinement.detect_quality \
     /path/to/zarr \
     --expected-subject-count 4
   ```
   This keeps frames with four detections clean, labels only
   `frame_counts > 4` as quality label `4`, and skips global temporal jump/blip
   artifact labeling because raw rows interleave multiple subjects before
   arena or identity assignment. Use arena-assignment and
   `single_subject_per_arena` to reject duplicate detections within a specific
   sub-arena.

   The cluster wrappers expose the same policy as
   `--quality-expected-subject-count N`.
3. Initialize the curated refined run.
   ```bash
   scripts/py -m fisheye.refinement.refine_detect /path/to/zarr
   ```
   This writes sparse `instances/` and `source_detections/` on
   `refined_detect_runs/<latest>`.
   Interpolation is no longer part of the normal detect-refinement workflow.
   For seeded sampled training Zarrs, pass the explicit seed run:
   ```bash
   scripts/py -m fisheye.refinement.refine_detect \
     /path/to/training.zarr \
     --detect-run detect_seed_<model_or_date>
   ```
   If a tuned `analysis_metadata.attrs["dish_mask"]` is present, refinement
   applies it as a spatial gate for any source detect run: raw detect candidates
   remain in `source_detections`, but clean candidates whose bbox center falls
   outside the dish are marked `filtered` with reason `outside_dish_mask` and
   cannot enter `instances` or win per-frame top-k selection. Before this test,
   the fitted boundary is expanded by the versioned 0.5 mm physical tolerance.
   The expansion is resolved from camera-space
   `pixels_per_mm_camera` plus full-frame dimensions and is recorded separately
   from the immutable fitted dish geometry.
   In sampled-import
   passthrough mode, jump/blip filters are disabled before this spatial gate and
   the remaining raw detections are materialized into the curated sparse
   `instances/` surface for manual review.
   If a single-subject sampled training Zarr has multiple seed detections per
   frame, use `--per-frame-top-k 1` to accept only the highest-confidence
   candidate per frame while preserving all raw candidates in
   `source_detections`:
   ```bash
   scripts/py -m fisheye.refinement.refine_detect \
     /path/to/training.zarr \
     --detect-run detect_seed_<model_or_date> \
     --per-frame-top-k 1
   ```
   Non-top clean candidates are marked as `duplicate` with reason
   `per_frame_top_k_excluded`; they are not discarded from provenance.

   For web detection assignment on sampled training Zarrs, this refined run is
   the required review surface. Assignment infrastructure should skip a Zarr
   that has only `detect_runs/<run>` and no `refined_detect_runs/<run>`, because
   raw detections are not the editable curated authority. A RedScare-style
   explicit run should look like:
   ```bash
   scripts/py -m fisheye.refinement.refine_detect \
     /path/to/RedScare_training.zarr \
     --detect-run detect_red_scare_training_seed_v004_20260626_01 \
     --per-frame-top-k 1 \
     --run-name refined_detect_training_review_red_scare_training_review_YYYYMMDD_NN
   ```
   This is independent of acquisition crop-video pose/mask review. A training
   Zarr may have `crop_runs/<acquisition_crop_video_run>` plus keypoint and
   subject-mask review surfaces, and therefore be ready for pose/mask tasks,
   while still not being ready for `detect_training` assignment until the
   refined detect run above exists.
4. Edit the curated refined surface if needed.
   ```bash
   scripts/py -m fisheye.tune.detect_review /path/to/zarr
   ```
   Manual review now has two refined modes:
   - one slot per frame for the legacy single-subject workflow
   - one slot per `(frame, arena_id)` when fixed sub-arena ROI definitions are
     available from subdish masks or arena-assignment metadata

   `detect_review` still does not support unconstrained multi-instance editing
   within a single arena/ROI. Legacy sparse manual subgroups may still appear
   when operating on old archives.
5. Approve the refined detect run.
   ```bash
   scripts/py -m fisheye.utils.accept_detect_review /path/to/zarr --state approved --intended-use training --reviewer <name>
   ```
   The interactive reviewer can also approve by pressing `a`; for training use,
   run it with the default `--review-intended-use training` or pass that flag
   explicitly.
   `resolved_group` should normally be `refined` for current runs.
   For `approved` + `training`, approval now also writes a canonical
   `analysis/detection_profile_runs/<run>` profile for the approved detection
   source, records profile/source fingerprints, and syncs the
   `detection_data_profile` registry projection when the Zarr is already
   registered. Use `--skip-detection-profile` only for intentionally incomplete
   or diagnostic approvals.

6. Build crops from the curated refined surface.
   ```bash
   scripts/py -m fisheye.tracking.crop /path/to/zarr --source-type refined
   ```
   `auto` also resolves to the active curated refined surface when it exists.

## Batch Review Queue

For many pending training archives, use the batch wrapper to build a queue and
open `detect_review` one Zarr at a time:

```bash
scripts/py -m fisheye.utils.review_detect_batch \
  --registry /nvme1/palette_registry.sqlite \
  --zarr-use training \
  --path-contains _training.zarr \
  --queue-output /tmp/pending_detect_training_zarrs.txt \
  --details-output /tmp/pending_detect_training_zarrs.tsv \
  --state-file /tmp/detect_review_batch_state.json \
  --all \
  --reviewer "$USER"
```

The wrapper is orchestration only. It does not bulk-approve labels. Each entry
still opens the interactive `detect_review` UI, and approval remains an explicit
review action in that UI.

If the run is interrupted, resume from the same state file:

```bash
scripts/py -m fisheye.utils.review_detect_batch \
  --registry /nvme1/palette_registry.sqlite \
  --zarr-use training \
  --path-contains _training.zarr \
  --state-file /tmp/detect_review_batch_state.json \
  --resume \
  --all \
  --reviewer "$USER"
```

By default, `--resume` skips entries whose last recorded review state is
`approved`, `rejected`, or `needs_review`. Pending entries remain in the queue.

Use `--dry-run` to inspect the planned queue and commands without launching the
review UI.

## Status Fields

The slot-based edit vocabulary is:

- `status_codes`: `present`, `missing`, `filtered_out`, `ambiguous`
- `source_kind_codes`: `none`, `raw_detect`, `interpolated`, `manual`
  - `interpolated` is retained for legacy compatibility/provenance and is not
    the normal outcome for current sparse refined runs.
- `manual_edit_flags`: sticky per-slot marker for manual correction/clear/retune
- `reason`: explanatory tags only

`ambiguous` means the dense/single-slot compatibility view cannot collapse a
frame into one obvious detection, usually because multiple source candidates or
multiple curated instances exist. It is not a failure state by itself.

The sparse refined surfaces carry the consumer-facing state:

- `instances/`: accepted curated detections
- `source_detections/decision_codes`: raw-candidate decisions such as
  `accepted`, `filtered`, `duplicate`, and `manual_clear`

`instances/` may now contain multiple curated rows for the same frame.

## Legacy Compatibility

Older archives may still contain:

- `refined_detect_runs/<run>/filtered`
- `refined_detect_runs/<run>/interpolated`
- `refined_detect_runs/<run>/<manual_group>`

Those sparse groups are now compatibility/provenance inputs. They are not the
primary curated contract for new runs.

If an old training archive has complete reviewed labels in one of those legacy
groups but an incomplete `instances/` surface, migrate it into a new canonical
run instead of editing the historical run in place:

```bash
scripts/py -m fisheye.utils.migrate_legacy_detect_labels \
  --zarr-list /tmp/pending_detect_training_zarrs.txt \
  --reviewer "$USER" \
  --notes "Migrated legacy reviewed/manual detection labels into canonical sparse instances for training export." \
  --apply \
  --registry /nvme1/palette_registry.sqlite \
  --keep-going
```

The migration writes `<source_run>_legacy_labels_canonical`, promotes it to
`refined_detect_runs.attrs["latest"]`, records an approved/training review
payload, writes a detection profile, and preserves the original refined run as
historical provenance. Refresh the registry review projection afterward:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry /nvme1/palette_registry.sqlite \
  --refresh-detect-quality /nvme1/recordings
```

## Diagnostics

- Summarize recording status:
  ```bash
  scripts/py -m fisheye.utils.check_recording_steps /path/to/recordings --recursive
  ```
- Inspect crop source linkage:
  ```bash
  scripts/py -m fisheye.diagnostics.check_crop_sources /path/to/zarr
  ```
- Inspect current refined detect state:
  ```bash
  scripts/py -m fisheye.visualization.detection_visualizer /path/to/zarr --refined-variant refined
  ```
- Inspect current curated refined detect state and source decisions:
  ```bash
  scripts/py -m fisheye.utils.inspect_refined_detect_run /path/to/zarr
  ```
