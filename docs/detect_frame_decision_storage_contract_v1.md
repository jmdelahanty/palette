# Detection Frame Decision Storage Contract v1

Status: implemented editable review contract, 2026-08-06.

## Purpose

`refined_detect_runs/<run>/instances` is a sparse table of positive detection
observations. A reviewed frame containing no valid subject must not be encoded
as a fake instance, a zero-area box, or an audit-log-only convention. Palette
stores that decision in a separate frame-axis surface:

```text
detect_frame_decision_runs/<source_refined_detect_run>/
  frame_indices                    int32  [F]
  source_acquisition_frame_index  int64  [F]
  decision_codes                  uint8  [F]
  reason_codes                    uint16 [F]
```

The run is bound by the exact `source_refined_detect_run` attribute. It has no
`latest` selector and is never independently selector-visible. The editable
surface uses Zarr v3, the shared `editable_local_v1` byte planner, regular
unsharded chunks, and direct/unconsolidated reads while review is active.
Opening the explicit review workflow also stamps the bound refined run with
`detect_frame_review_contract = palette.detect_frame_review.v1`. This makes an
incomplete all-positive review fail closed even before its first negative
decision creates the sibling run. Historical runs without that declaration
retain their positive-only compatibility behavior.

## Codes and invariants

- `decision_codes`: `0 = unreviewed`, `1 = negative`.
- `reason_codes`: `0 = none`, `1 = subject_outside_dish`.
- `unreviewed` requires reason `none`; `negative` requires a nonzero reason.
- `frame_indices` is exactly `arange(F)`.
- `source_acquisition_frame_index` exactly matches
  `raw_video/original_frame_indices`, or the identity frame axis when that
  optional lineage array is absent.
- A frame cannot simultaneously contain a refined detection instance and an
  explicit negative decision.

The v1 web action marks only an already-empty sparse frame. If an operator later
adds a positive detection, Palette clears the negative decision after the
refined instance write succeeds. Completion fails closed until every frame in
the assigned review scope either has at least one retained instance or is
explicitly negative.

## Current reason scope

Version 1 intentionally implements only `subject_outside_dish`. Additional
negative-frame reasons or region-level hard negatives require a versioned
contract extension; they must not be represented by overloading this code.

## Publication and training

This group is mutable review state, not a published detection authority. The
merged detection-training exporter implements the promotion bridge. If this
surface exists, export fails until every source frame is either positive or
explicitly negative and fails if decisions change while images are copied.

The immutable artifact uses the frame as its sample and split axis:

```text
detection_training_supervision/
  label_state_codes  uint8  [F]  # 1 positive, 2 negative
  reason_codes       uint16 [F]

refined_detect_runs/<export>/instances/
  frame_indices      int32 [N]
  frame_counts       int32 [F]
  frame_offsets      int64 [F+1]
  ... all instance payload and identity columns [N]
```

`frame_offsets[f]:frame_offsets[f+1]` is the complete target for one training
image. Equal offsets encode a reviewed negative; a range of length one encodes
one subject; a longer range encodes multiple detections. No placeholder row is
created. Train/validation/test indices address the frame axis, preventing boxes
from one image from leaking across splits. `source_index` separately preserves
frame lineage and every instance row's lineage.

Historical sources without a frame-decision surface remain positive-only when
exported. New reviewed sources do not receive that fallback. The SQLite browser
audit remains operational history and is not the scientific label authority.
