# Sampled Training Detection Contract — 2026-08-06

Status: implemented for selector-ineligible training canaries.

## Decision

A sampled training Zarr does not share the recording-level canonical detection
frame axis. Its review UI uses local rows `0..F-1`, while scientific identity
must remain tied to acquisition frames. Palette therefore publishes a distinct
sampled-training detection contract rather than weakening or pretending to
satisfy canonical detection v1.

The strict table is `detect_runs/<run>/instances` with the same nine exact
typed arrays used by canonical detection:

- `frame_indices` is the local sampled-row index.
- `source_acquisition_frame_index` equals
  `raw_video/original_frame_indices[frame_indices]` exactly.
- `instance_key` is derived from recording identity, acquisition frame,
  authoritative normalized box, and class.
- `frame_row_offsets` has shape `F+1` and indexes zero, one, or many detections
  per sampled frame.
- normalized boxes are authoritative source-camera geometry; pixel boxes and
  centers are exact derived float32 projections.

One-fish-per-frame is a dataset expectation, not a destructive writer rule.
The publisher records zero/one/multiple counts and never forces `max_det=1` or
drops duplicate candidates during inference. A later refined seed may apply
top-1 while retaining every source row in its audit table.

## Lifecycle

Checklist:

- [x] Copy and content-authenticate the complete training base on node-local
  scratch.
- [x] Run the registered detector into an immutable, selector-free
  `detection_artifact_runs/<run>`.
- [x] Atomically import that artifact without selectors.
- [x] Recompute and validate the frame mapping, geometry, stable keys, logical
  hashes, direct metadata declarations, codecs, chunks, and shards.
- [x] Build an access-aware bound candidate on node-local scratch.
- [x] Atomically publish it under `detect_runs/<run>`.
- [x] Keep both runs selector-ineligible and unregistered.
- [x] Keep the still-editable training root unconsolidated.
- [x] Validate the first Batman canary: 178 one-detection frames, 22 misses,
  zero multi-detection frames; no values were forced or imputed.
- [x] Initialize a selector-ineligible mutable refined review seed after inspecting
  zero/one/multiple detector cardinality.
- [ ] Finish review/correction of all assigned frames.
- [x] Implement the frame-supervision export bridge: sparse positive instances
  remain instance rows, explicit reviewed negatives become image samples with
  empty targets, multiple detections per frame are preserved, and unresolved
  review state fails closed.
- [ ] Export and validate the completed Batman review, then publish crop,
  keypoint, and subject-mask training surfaces.
- [ ] Consolidate only when the complete training artifact becomes immutable
  and selector-visible.

Implementation:

- `fisheye.utils.run_sampled_training_detection_canary`
- `fisheye.shared.zarr.sampled_training_detection_publication`
- `fisheye.shared.detection_tables`

The compatibility reader resolves modern columns from `instances/` and legacy
columns from the run root. New writers do not create root aliases. Manual
curation preserves copied keys and mints new keys using mapped acquisition
frames, preventing sampled datasets from inventing incompatible identities.

## Batman canary evidence

- Palette commit: `ff03ec4be198cc8a26e6143cff051cc602e2c1b9`
- LSF job: `153285127`, `gpu_l4`, host `h08u02`
- Artifact: `detection_artifact_runs/detect_artifact_batman_training_20260806_v1`
- Bound seed: `detect_runs/detect_batman_training_review_20260806_v1`
- Manifest payload digest:
  `dfe46a1c415801b1a4cadcaa0cec0d5ef408eb7358cd8ebb46bcb5de9678e387`
- Source copy/authentication: 11.23 s for 4,269,644,734 bytes
- Inference: 27.25 s
- Artifact publication: 0.69 s
- Binding and bound publication: 2.36 s
- Peak process RSS reported by LSF: 2,271 MB

An independent direct-metadata reopen returned `valid=true` with no errors.
The `detect_runs` parent contains no selector attributes; both run groups are
complete and selector-ineligible. Root consolidated metadata remains `null`, as
required while detection review may still add or correct rows.

The mutable review seed was then initialized at Palette commit
`dbbfc2f56cd68c6c3a77a2f6b76c2e1ee2374631` by LSF job `153285133`:

- `refined_detect_runs/refined_detect_batman_training_review_20260806_v1`
- 178 presented instance rows and 178 source-audit rows
- exact instance-key equality with the bound detector snapshot
- top-1 policy enabled, with zero duplicate rows removed
- 89% coverage and 22 intentionally unresolved sampled frames
- complete but selector-ineligible; no `latest`, `latest_complete`, or pending
  selector was written
- no registry/status projection and no root consolidation

The missing local review frames are `87, 88, 91..109, 113`, corresponding to
acquisition frames `60552, 61248, 63336..75864` in steps of 696, and `78648`.
They should be inspected as a contiguous failure episode rather than silently
filled. The explicit review command is:

```bash
scripts/py -m fisheye.tune.detect_review \
  /path/to/2026-07-21T19-38-32Z_arena_2_Batman_training_base.zarr \
  --refined-run refined_detect_batman_training_review_20260806_v1 \
  --frames 87,88,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,113 \
  --use-full-res
```

This mutable review surface is staging, not the final immutable refined-v1
publication. Accepted edits must later be compacted and validated into the
strict refined snapshot contract before training activation.
