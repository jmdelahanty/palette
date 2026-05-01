# Bout Classification Runs Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-05-01
-->

Purpose: define the stable Palette Zarr contract for per-bout classifier
outputs from Megabouts or other external/internal bout-classification tools.

## Design Boundary

Classifier outputs are derived analysis runs. They must not mutate or redefine
the source swim-bout segmentation, tail posture, track kinematics, or subject
shape runs that feed them.

The generic output family is:

```text
analysis/bout_classification_runs/<run>/
```

The first producer is the Megabouts direct-classifier adapter, but this run
family is intentionally tool-neutral enough for future `zebrazoom`, `stytra`,
`beast_style`, or Palette-native classifiers.

## Compatibility Views

Palette should keep external-tool compatibility arrays in explicit derived
view runs rather than embedding every tool-specific representation directly in
canonical writers.

For Megabouts, the classifier expects:

- 10 cumulative tail-angle channels with Megabouts segment-angle semantics
- trajectory windows translated to bout onset and rotated into onset heading
- fixed-duration classifier windows

Those are tool-facing classifier inputs, not Palette's canonical body-shape or
tail-kinematics representation. The current policy is therefore:

- Canonical writers write Palette-native geometry and kinematics.
- Compatibility writers may write explicit view surfaces such as
  `analysis/tail_posture_view_runs/<run>` with
  `view_family="megabouts_compatible"`.
- Classifier adapters consume those views and record the conversion parameters
  in the classifier run.

This avoids making Megabouts a hidden source of truth while still keeping the
operator workflow one-command once the view exists.

## Run Attributes

Required root attrs:

```text
schema_id                         "analysis.bout_classification_runs"
schema_version                    1
classifier_family                 "megabouts" | "zebrazoom" | "stytra" | "beast_style" | "palette_native" | ...
classifier_name                   method/model name
source_mode                       "palette_bouts" | "external_bouts" | ...
row_axis                          "swim_bout_rows"
invalid_window_policy             e.g. "skip_invalid_windows"
source_refs                       JSON-compatible source path payload
parameters                        JSON-compatible parameter payload
```

Recommended root attrs:

```text
classifier_version                package/model/checkpoint version when available
adapter_method                    Palette adapter method name
adapter_method_version            Palette adapter method version
tail_angle_conversion             JSON-compatible conversion payload
trajectory_conversion             JSON-compatible conversion payload
invalid_frame_policy              JSON-compatible invalid-window policy payload
provenance                        Palette stage provenance payload
```

Recommended Megabouts direct-classifier parameter/provenance fields:

```text
classifier_input_mode             "palette_prepared_fixed_windows"
megabouts_preprocessing           false unless Megabouts preprocessing outputs were consumed
megabouts_segmentation            false unless Megabouts segmentation outputs were consumed
source_fps                        resolved source FPS
window_duration_s                 classifier window duration in seconds
window_frames                     classifier window duration in source frames
megabouts_time_sampling           true when Megabouts receives FPS-aware time samples
```

Megabouts-specific provenance attrs may also be present:

```text
megabouts_package_version
megabouts_package_path
megabouts_source_repo
megabouts_git_commit
megabouts_category_labels
```

## Per-Bout Table

The required table is:

```text
analysis/bout_classification_runs/<run>/per_bout/
```

It is stored as a columnar structured table with one row for each source
swim-bout row. Keeping skipped rows in the table preserves source row identity
and makes comparisons across classifier runs straightforward.

Required fields:

```text
source_bout_id
start_frame
end_frame
window_start_frame
window_end_frame
HB1_frame
HB1_offset_frames
category_id
category_label_bytes
subcategory_id
sign
probability
tail_valid_fraction
traj_valid_fraction
max_consecutive_tail_invalid
max_consecutive_traj_invalid
source_window_valid
classified
valid
failure_reason_bytes
```

Field semantics:

- `source_bout_id`: source swim-bout row identity when available, otherwise
  row index.
- `start_frame` / `end_frame`: source swim-bout boundary.
- `window_start_frame` / `window_end_frame`: fixed classifier input window.
- `HB1_frame`: classifier first-half-beat frame in source frame coordinates,
  or `-1` when not classified.
- `HB1_offset_frames`: classifier first-half-beat offset relative to
  `window_start_frame`, or `-1` when not classified.
- `category_id`, `subcategory_id`, `sign`, `probability`: classifier outputs.
- `category_label_bytes`: UTF-8 label bytes for display/readback.
- `source_window_valid`: source window passed Palette eligibility checks.
- `classified`: classifier was actually run for this row.
- `valid`: classification row is usable for downstream analyses.
- `failure_reason_bytes`: UTF-8 reason string; `ok` for usable rows.

Invalid/skipped row convention:

```text
classified = false
valid = false
category_id = -1
subcategory_id = -1
HB1_frame = -1
HB1_offset_frames = -1
probability = NaN
category_label_bytes = "skipped_invalid_window"
failure_reason_bytes = source failure reason
```

## Validation

Palette provides a contract validator and summary utility:

```bash
scripts/py -m fisheye.analysis.bout_classification_runs <analysis.zarr> \
  --run latest
```

Use `--strict` to require recommended provenance/conversion attrs, not only
the fundamental readable schema.

Validator checks include:

- run family and `latest` resolution
- required run attrs
- required `per_bout` field names and arrays
- row-count attrs vs table rows
- `classified` rows have valid source windows, non-negative category ids, and
  finite probabilities
- `valid` rows are classified

## Current Producer

The Megabouts direct adapter writes this family with:

```bash
scripts/py -m fisheye.analysis.megabouts_classifier <analysis.zarr> \
  --tail-posture-view-run latest \
  --track-kinematics-run latest \
  --track-scope offline \
  --track-id 0 \
  --swim-bout-run latest \
  --speed-level default \
  --megabouts-repo ~/gitrepos/megabouts \
  --run-name <run>
```

For the feeding canary, the trusted comparison candidate is:

```text
analysis/bout_classification_runs/megabouts_classifier_onset_aligned_canary_20260501
```

Interpret current Megabouts direct-adapter runs as "Megabouts classifier on
Palette-prepared fixed windows." They are FPS-aware because Palette passes the
resolved source FPS into Megabouts config objects and records the resulting
window frame count, but they do not imply that full Megabouts preprocessing or
Megabouts segmentation was used.
