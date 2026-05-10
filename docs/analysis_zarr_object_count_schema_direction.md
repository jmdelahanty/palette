# Analysis Zarr Object-Count Schema Direction

## Purpose

Palette analysis stores are starting to accumulate many Zarr metadata objects.
This note records the design direction for reducing object count in future
analysis-run schemas without losing provenance, discoverability, or UI
selectability.

This is not a migration plan for existing runs. It is a design constraint for
new schemas and v2 rewrites.

For the current writer-by-writer compact-layout status and migration priority
list, see
[analysis_writer_compact_layout_inventory.md](analysis_writer_compact_layout_inventory.md).

## Current Audit Snapshot

Read-only audit of `/nvme1/recordings` on 2026-05-08:

- `53` analysis stores
- `132,343` total `zarr.json` files
- `14,351` groups
- `117,992` arrays
- `0` invalid `zarr.json` files
- largest outlier:
  `/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/..._analysis.zarr`
  with `17,470` `zarr.json` files

Top global family breakdown:

| Prefix | `zarr.json` count |
| --- | ---: |
| `analysis/swim_bout_runs` | `36,538` |
| `analysis/stimulus_response_runs` | `18,975` |
| `analysis/eye_angle_runs` | `14,384` |
| `analysis/bout_kinematics_runs` | `13,618` |
| `refined_subject_masks_runs` | `9,532` |
| `refined_keypoints_runs` | `7,786` |
| `analysis/track_kinematics_runs` | `6,758` |
| `analysis/subject_shape_runs` | `6,309` |

The new detection-coverage dashboard is not a material contributor. Persisting
one compare dashboard under each latest refined-detect run would add only one
new PNG-array metadata object per store because the `visualizations` groups
already exist.

Re-run the audit with:

```bash
scripts/py -m fisheye.utils.audit_zarr_group_counts \
  --recordings-root /nvme1/recordings \
  --output-dir /tmp/palette_zarr_group_count_audit \
  --format markdown
```

The audit is filesystem-only: it scans `zarr.json` files, reads each
`node_type`, and does not call `zarr.open_group(...)`. Outputs include
`archive_summary.csv`, `family_summary.csv`, `component_summary.csv`,
`audit_summary.json`, and `audit_summary.md`.

## Why This Matters

Zarr is suitable for parallel array access, object stores, and distributed
storage only when object count is controlled. It does not make filesystem
metadata operations free.

On directory stores, each group, array, and chunk can become a filesystem
object. On NFS and similar shared filesystems, the expensive operations are
often metadata-heavy:

- creating many small files
- recursively listing large trees
- stat/open calls for many small objects
- metadata cache invalidation during concurrent writes
- copying or synchronizing directory trees with many small files

Consolidated metadata improves read discovery but does not remove the
underlying object count. Sharding can reduce chunk-file count for large dense
arrays, but it does not fix schema fanout from thousands of groups and small
arrays.

## Main Anti-Patterns

### 1. Parameter-Sweep Fanout

Problem shape:

```text
analysis/swim_bout_runs/<candidate>/<signal_level>/<table>/<column_array>
analysis/swim_bout_runs/<candidate_2>/<signal_level>/<table>/<column_array>
...
```

This is useful during canary exploration, but poor as a production archive
layout. Each parameter variant becomes a new branch, and each speed/signal
representation multiplies nested table groups and column arrays.

Observed examples:

- `detect_bouts_multi_level.py` writes one subgroup per speed level under each
  swim-bout candidate.
- Each level writes separate columnar groups such as `bouts`, `peak_events`,
  `inter_bout_intervals`, histograms, global metrics, and bout points.
- `bout_kinematics.py` repeats the same pattern at a smaller scale by writing
  domain and heading-level groups with their own per-bout metric tables.

Future direction:

- Keep exploratory sweeps temporary or sidecar-only.
- Promote one canonical candidate per stage family into the long-lived
  analysis store.
- Store candidate identity as columns or enum IDs rather than tree branches.
- Store parameter records once in an index table, not repeated in attrs on many
  small groups.

### 2. Representation / Alias Materialization

Problem shape:

```text
angles/roi/<raw_field>
angles/roi/<alias_field>
angles/roi/<derived_variant>
angles/roi/<derived_variant>_smoothed
angles/roi/<derived_variant>_delta
angles/frame/<same families again>
```

This is readable and convenient, but it turns semantic representations into
physical arrays. The storage tree grows when we add a new representation, even
if the representation is a deterministic transform of existing canonical
columns.

Observed examples:

- `eye_angle_analysis.py` stores canonical major-axis values, gaze values,
  nasal gaze summaries, Bianco/Engert eye-frame angles, smoothed variants,
  deltas, speeds, frame-aligned copies, QA arrays, and support arrays.
- The run metadata already describes selectable representations, but readers
  still hardcode many physical arrays.

Future direction:

- Store the minimal canonical columns needed to reconstruct the declared
  representations.
- Put aliases and derived views in a machine-readable variant schema.
- Add resolver helpers that return arrays plus transforms for a requested
  representation.
- Materialize compatibility arrays only for old consumers or explicitly
  persisted cache/profile outputs.

Clarification:

Representation materialization does not mean the representations are
scientifically wrong. It means values that are aliases or deterministic views of
canonical signals are persisted as separate physical arrays. Eye angles are the
clearest example: `left_eye_angle_deg`, `right_eye_angle_deg`,
`vergence_eye_angle_deg`, gaze/minor aliases, nasal-gaze summaries, smoothed
variants, deltas, and frame-aligned copies are useful outputs, but many are
deterministic transforms of canonical major/gaze arrays. A compact migration for
this class of data is mostly a repack/derive operation, not a scientific
recompute from raw masks, as long as the canonical arrays and transform
parameters are present.

### 3. Component-Per-Group Mirrors

Problem shape:

```text
components/<component_name>/mask_present
components/<component_name>/area_px
components/<component_name>/metrics/<metric>
components/<component_name>/finalization_metrics/<metric>
components/<component_name>/reason_bytes
```

This is natural for manual inspection but expensive for component-indexed data.
When the same metric exists for every component, component identity should
usually be an axis or enum column, not a child group.

Observed examples:

- `refined_subject_masks_runs` stores canonical dense `masks_roi`, then also
  component-local mirrors and per-component metrics.
- `subject_shape_runs` stores `components/<component>` groups for common
  metrics, plus many body-specific arrays.

Future direction:

- Keep large canonical dense arrays where they match access patterns.
- Store common component metrics as stacked arrays with a component axis:
  `area_px (N,C)`, `centroid_xy (N,C,2)`, `bbox_xyxy (N,C,4)`,
  `mask_present (N,C)`.
- Store component names and labels as small dictionaries/index arrays.
- Keep specialized component-only geometry in dedicated semantic groups only
  when it is not naturally shared across components.

Clarification:

Component-per-group mirrors are expensive when every component repeats the same
field set:

```text
components/eye_left/area_px
components/eye_right/area_px
components/swim_bladder/area_px
```

For common metrics, component identity should become an axis:

```text
component_names       (C,)
metrics/area_px       (N,C)
metrics/mask_present  (N,C)
metrics/centroid_xy   (N,C,2)
```

This is different from genuinely component-specific geometry. A body centerline
or tail-sample surface can remain in a body-specific semantic group because it
does not naturally apply to every component.

## Recommended V2 Layout Patterns

### Run-Local Index Tables

Use index tables to identify variants, methods, signals, and parameters:

```text
analysis/<family>/<run>/
  variant_index/
    variant_id
    variant_name
    method
    params_json
    source_signal_id
```

Rows in metric tables then reference `variant_id` or `signal_id`.

### Dense Variant Axes

When all variants share the same row axis, prefer dense arrays:

```text
movement/speed/value_mm_s      (R, L)
movement/speed/value_px_s      (R, L)
movement/speed/accel_mm_s2     (R, L)
movement/speed/level_names     (L,)
```

This is better than:

```text
movement/speed/raw/...
movement/speed/filtered/...
movement/speed/smoothed/...
movement/speed/averaged/...
```

### Compact Ragged Track Layout

For track kinematics, prefer run-level ragged arrays over `tracks/id_<track>`
subtrees:

```text
tracks/track_ids        (T,)
tracks/row_offsets      (T+1,)
frame_indices           (R,)
time_seconds            (R,)
positions_px            (R,2)
positions_mm            (R,2)
sample_valid            (R,)
movement/speed/value    (R,L)
```

Keep per-track groups only as compatibility projections during migration.

### Component-Axis Mask and Shape Layout

For subject masks and common subject-shape summaries:

```text
component_names         (C,)
masks_roi               (N,C,H,W)
metrics/mask_present    (N,C)
metrics/area_px         (N,C)
metrics/centroid_xy     (N,C,2)
metrics/bbox_xyxy       (N,C,4)
qa/reason_code          (N,C)
qa/reason_labels        attr/list
```

Keep body-only geometry separate:

```text
body_geometry/centerline_xy
body_geometry/bspline_xy
body_geometry/tail_samples_xy
```

Insertion/editing model:

Flattened layouts should use stable row, component, and keypoint IDs as the
write targets. Refined detections/keypoints are upserted by stable row identity
such as `refined_row_id` or `(frame_index, entity_id)`. Component masks are
updated by `(row_id, component_id)`, not by creating or mutating a component
subtree:

```text
masks_roi[row_id, component_id, :, :] = edited_mask
metrics/area_px[row_id, component_id] = recomputed_area
qa/reason_code[row_id, component_id] = manual_edit_code
edit_applied[row_id, component_id] = true
```

Keypoints follow the same pattern:

```text
keypoints_xy[row_id, keypoint_id, :] = corrected_xy
keypoint_valid[row_id, keypoint_id] = true
```

The component/keypoint axes should be fixed by the run schema. Adding a new
biological component or keypoint class is a schema-version change and should
usually create a new run with a new axis dictionary, rather than appending a new
component casually in-place.

Flattening therefore does not prevent incremental review edits. It requires
helper APIs that resolve names to IDs, upsert rows safely, update dependent
metrics, and respect physical Zarr chunk boundaries during writes.

### Canonical Eye-Angle Columns Plus Resolver

For eye angles:

```text
angles_compact/
  frame_indices              (N,)
  major_signed_deg           (N,2)
  gaze_signed_deg            (N,2)
  gaze_xy                    (N,2,2)
  centroid_angle_deg         (N,2)
  valid_eye                  (N,2)
  qa_code                    (N,2)
support/body_frame/...
```

Representations such as `eye_frame`, `nasal_gaze`, `vergence_eye_angle`, and
legacy aliases should be resolved through schema transforms instead of stored
as independent arrays by default.

## Family-Specific Direction

### `analysis/swim_bout_runs`

Current issue:

- candidate run groups are multiplied by method/parameter choices
- each speed level creates its own nested table set
- columnar tables create one metadata object per field

Direction:

- v2 compact layout with `candidate_index`, `signal_index`, `bouts`, and
  `peak_events` tables
- include `candidate_id` and `signal_id` columns
- store detector traces as dense `detector_signals (S,F)` only when needed
- use sidecar Parquet or scratch-only Zarr for large parameter sweeps
- promote only accepted/canonical bout candidates to the stable analysis store

Focused schema design:
[swim_bout_runs_v2_compact_layout.md](swim_bout_runs_v2_compact_layout.md).

### `analysis/bout_kinematics_runs`

Current issue:

- per-heading-level groups duplicate table structure

Direction:

- keep semantic domains (`movement`, `heading`, `eye_gaze`) but collapse
  heading-level variants into a `heading_level_id` column
- store `analysis_level_index` once per run
- make readers accept both v1 hierarchical and v2 compact layouts

### `analysis/track_kinematics_runs`

Current issue:

- data is duplicated across flat arrays, grouped `movement/speed` arrays, and
  transitional `speed_derivatives` groups
- per-track groups make object count scale with track count

Direction:

- compact run-level ragged/CSR layout for tracks
- speed/acceleration variants as dense level axes
- source-path plot specs and reader helpers should hide physical layout from
  consumers

### `analysis/eye_angle_runs`

Current issue:

- many deterministic representations are materialized as arrays
- frame-aligned copies and smoothed/delta variants multiply object count
- readers hardcode physical paths instead of using the variant schema

Direction:

- canonical compact angle arrays plus variant resolver
- optional compatibility materialization for old consumers only
- interactive specs preferred over many persisted PNG snapshots

### `analysis/subject_shape_runs`

Current issue:

- common component metrics live under per-component groups
- body-specific geometry is wide but mostly justified

Direction:

- common metrics use component axes
- body-only geometry remains in body-specific groups
- reason strings become reason codes plus dictionaries

### `subject_mask_runs` and `refined_subject_masks_runs`

Current issue:

- dense arrays are justified, but component mirrors and per-component metric
  groups inflate metadata
- `source_seed_masks_roi` can duplicate large mask payloads

Direction:

- keep canonical dense `mask_probs_roi` / `masks_roi`
- stack common component metrics by component axis
- keep packed contours, but use one shared contour group rather than one group
  per component
- treat seed masks as optional cache; prefer source refs plus deterministic
  thresholds/finalization policy where possible

## Reader-First Migration Strategy

Before changing writers, add stable resolver helpers:

- `resolve_swim_bout_tables(...)`
- `resolve_bout_kinematics_tables(...)` (implemented for hierarchical-v1 and
  compact-v2 bout-kinematics runs)
- `resolve_track_motion_arrays(...)`
- `resolve_subject_shape_arrays(...)`
- `resolve_eye_angle_representation(...)`
- `resolve_refined_subject_mask_arrays(...)`

Each helper should:

- prefer compact v2 layouts
- fall back to current v1 hierarchical paths
- expose semantic names, not physical Zarr paths, to callers
- return enough source/provenance metadata for UI labels

This lets Marimo, Crimson, and downstream analyses migrate before physical
storage changes.

## Operational Policy

- Treat file/object count as a schema budget, not just a storage backend issue.
- Do not persist exploratory parameter sweeps indefinitely in production
  analysis stores.
- Prefer one canonical/latest run per family in finalized online stores.
- Use sidecar exports or scratch stores for broad sweeps and diagnostics.
- Persist PNG dashboards only for canonical/latest runs unless explicitly
  debugging.
- Re-run the object-count audit after major schema changes.

Suggested warning thresholds remain those in `docs/zarr_storage_lifecycle_policy.md`:

- target finalized online stores below about `10k` files where practical
- investigate any single group contributing more than about `5k` files
- treat stores above about `20k` files as portability/NFS-risk candidates

## Open Design Questions

- Should exploratory sweep outputs live in Parquet sidecars by default, or in
  scratch-only Zarr stores that can be pruned?
- Which compatibility arrays are required for Crimson before v2 compact readers
  exist?
- Should promoted canonical analysis stores have an automated prune step that
  removes non-latest exploratory runs?
- Which large dense derived arrays should be sharded after they become
  immutable?
- Should the registry track per-store object counts and largest contributors as
  first-class health fields?
