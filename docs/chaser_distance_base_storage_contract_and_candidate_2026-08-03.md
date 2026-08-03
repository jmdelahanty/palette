# Chaser-distance sealed-base storage contract and candidate

Date: 2026-08-03

Status: implemented as a selector-ineligible, unpromoted storage candidate.
No production writer, selector, registry, storage profile, or canonical archive
is changed by this checkpoint.

## Authority boundary

The current `palette.chaser_distance.v1` run is not one uniformly sealed tree.
Its canonical coordinate publication provides three nested authorities:

1. `chaser_distance_publication_seal` directly protects 18 base arrays;
2. `epoch_window_identity_authority` protects four epoch-window identity arrays;
3. the digest-bound `chaser_distance_surface_manifest` protects 11 measurement
   arrays and three coordinate arrays (with overlap among the first 18).

Their exact union is the 30-array base below. The source loader recomputes the
seal against the current detection, stimulus, calibration, epoch, and payload
authorities. A candidate may be built only after that complete
`load_bound_chaser_distance_run()` check succeeds through the archive's
published consolidated generation.

The v1 tree also contains behavior labels/colors, role intervals, raw count
tables, threshold fractions, histogram counts, visualizations, and independently
published derived components. Those are not silently blessed by this candidate.
The canonical reader already fails closed for unsealed behavior/protocol
semantics, and derived components retain their separate component-publication
contract.

## Exact 30-array logical inventory

All arrays are required, fixed-width, immutable, and stored without dtype
probing or aliases. Camera-frame arrays are `WINDOWED`; collection and compact
epoch/distribution arrays are `EAGER`. Float precision remains the current
scientific precision (`float32`); no narrowing or recomputation occurs.

| Path | Exact dtype and axes | Units / coordinate space | Fill and null semantics | Role |
|---|---|---|---|---|
| `stimulus_state_key` | `int64[camera_frame]` | acquisition-frame index | exact `arange(F)`; no null | lineage |
| `frames/camera_frame_id` | `int64[camera_frame]` | acquisition-frame index | exact `arange(F)`; no null | lineage |
| `frames/stimulus_frame_num` | `int64[camera_frame]` | stimulus-frame index | `-1` unavailable | lineage |
| `frames/timestamp_ns` | `int64[camera_frame]` | ns | `-1` unavailable | lineage |
| `frames/stimulus_epoch_window_id` | `int32[camera_frame]` | epoch ID | `-1` outside a window | lineage |
| `chasers/chaser_index` | `int16[chaser]` | identity | nonempty, unique, increasing | lineage |
| `chasers/stimulus_instance_id_bytes` | `uint8[chaser,96]` | UTF-8 identity | NUL-padded; empty forbidden | semantic metadata |
| `chasers/source_track_key_bytes` | `uint8[chaser,96]` | UTF-8 lineage | NUL-padded; empty forbidden | lineage |
| `positions/source_detection_row_index` | `int64[camera_frame]` | source row | `-1` means no selected row | lineage |
| `positions/fish_centroid_img_xy` | `float32[camera_frame,xy]` | px; source-camera image | NaN pair when invalid | scientific |
| `positions/fish_centroid_arena_xy` | `float32[camera_frame,xy]` | px; arena-relative canvas | NaN pair when invalid | scientific |
| `positions/chaser_arena_xy` | `float32[camera_frame,chaser,xy]` | px; arena-relative canvas | NaN pair when invalid | scientific |
| `positions/fish_valid` | `bool[camera_frame]` | validity | false unavailable | scientific |
| `positions/chaser_valid` | `bool[camera_frame,chaser]` | validity | false unavailable | scientific |
| `distances/distance_px` | `float32[camera_frame,chaser]` | arena-relative px distance | NaN unless inputs valid | scientific |
| `distances/distance_mm` | `float32[camera_frame,chaser]` | physical mm | NaN unless finite | scientific |
| `distances/nearest_chaser_index` | `int16[camera_frame]` | chaser identity | `-1` when no finite distance | lineage |
| `distances/nearest_distance_mm` | `float32[camera_frame]` | physical mm | NaN unavailable | scientific |
| `epoch_summary/window_id` | `int32[stimulus_epoch_window]` | epoch ID | every row authoritative | lineage |
| `epoch_summary/label_bytes` | `uint8[stimulus_epoch_window,96]` | UTF-8 label | NUL-padded; empty forbidden | semantic metadata |
| `epoch_summary/start_frame` | `int64[stimulus_epoch_window]` | acquisition-frame index | every row authoritative | lineage |
| `epoch_summary/end_frame` | `int64[stimulus_epoch_window]` | acquisition-frame index | every row authoritative | lineage |
| `epoch_summary/mean_distance_mm` | `float32[stimulus_epoch_window,chaser]` | mm | NaN for no finite samples | scientific |
| `epoch_summary/min_distance_mm` | `float32[stimulus_epoch_window,chaser]` | mm | NaN for no finite samples | scientific |
| `epoch_summary/p05_distance_mm` | `float32[stimulus_epoch_window,chaser]` | mm | NaN for no finite samples | scientific |
| `epoch_summary/p50_distance_mm` | `float32[stimulus_epoch_window,chaser]` | mm | NaN for no finite samples | scientific |
| `epoch_summary/p95_distance_mm` | `float32[stimulus_epoch_window,chaser]` | mm | NaN for no finite samples | scientific |
| `epoch_distributions/bin_edges_mm` | `float32[distance_bin_edge]` | mm | exact bin authority | semantic metadata |
| `epoch_distributions/bin_centers_mm` | `float32[distance_bin]` | mm | exact bin authority | semantic metadata |
| `epoch_distributions/hist_density` | `float32[stimulus_epoch_window,chaser,distance_bin]` | per mm | zero for empty distribution | derived cache |

Cross-array validation requires one shared frame extent, one shared chaser
extent, one shared epoch-window extent, `B+1` bin edges for `B` centers, and
`hist_density[W,C,B]`.

Scalar distance arrays intentionally declare units but no coordinate-space ID.
Their governing measurement descriptor binds the source coordinate basis,
calibration, units, and distance operation; coordinate-space IDs remain reserved
for coordinate vectors and positions rather than inventing scalar spaces.

## Physical candidate

Candidates are written under the separate non-authoritative namespace:

```text
analysis/chaser_distance_storage_candidates/<candidate>/
```

The materializer:

- opens the immutable source with consolidated metadata;
- requires an explicit eligible canonical v1 run and fully revalidates its
  source-backed publication seal;
- derives chunks and indexed shards from actual dtype, shape, access unit, and
  the explicit `published_http_v1` byte profile;
- creates every array through the shared Zarr-v3 array factory;
- writes whole, non-overlapping physical chunk/shard units;
- preserves exact decoded values and hashes them in bounded row blocks;
- binds the source run path plus publication-seal, surface-manifest,
  row-identity, input, measurement, collection, and epoch-authority digests;
- persists the complete source-derived declaration inventory, source hashes,
  candidate hashes, storage receipt, and exact group topology;
- validates the local tree, consolidates it, and proves full direct versus
  consolidated group-and-array metadata equivalence;
- copies to a hidden same-filesystem sibling, validates again, renames
  atomically, marks complete, consolidates the archive, and validates again;
- retains a selector-ineligible failed tombstone if failure occurs after a
  public path becomes visible; and
- never changes `latest`, `latest_complete`, `authoritative_run`, the source
  authority provenance, publication policy, publication generation/lease, or
  any registry state.

Both of these attrs must remain exact false:

```text
stage_selector_eligible = false
storage_candidate_profile_promoted = false
```

Example dry run and publication:

```bash
scripts/py -m fisheye.analysis_workflows.materializers.chaser_distance_base \
  recording_analysis.zarr \
  --source-run goodcopbadcop_chaser_distance_v1_20260617 \
  --run-name sealed_base_storage_candidate_20260803 \
  --scratch-root /scratch/$USER/$LSB_JOBID/chaser-distance-candidate

scripts/py -m fisheye.analysis_workflows.materializers.chaser_distance_base \
  recording_analysis.zarr \
  --source-run goodcopbadcop_chaser_distance_v1_20260617 \
  --run-name sealed_base_storage_candidate_20260803 \
  --scratch-root /scratch/$USER/$LSB_JOBID/chaser-distance-candidate \
  --apply
```

## Validation checklist

- [x] Freeze exact dtype, rank, axes, units, coordinate, fill/null, authority,
      access, and immutable-write semantics for all 30 sealed arrays.
- [x] Require a deeply verified canonical v1 source; stale or merely live
      arrays cannot be re-sealed as a candidate.
- [x] Bind the complete source-derived declaration inventory and all governing
      source authority digests.
- [x] Bind source and candidate logical hashes and reject content/dtype changes.
- [x] Reject a regenerated manifest/hash that omits a required array.
- [x] Reject source-path, selector-eligibility, profile-promotion, group-tree,
      and physical-metadata tampering.
- [x] Use the shared byte planner, factory, codec profile, Zarr v3, and atomic
      run publisher.
- [x] Prove direct/consolidated equivalence for the complete persisted subtree.
- [x] Preserve existing source selectors and fail before visibility on copy
      failure.
- [ ] Run one full-duration selector-ineligible canary and mounted consumer
      benchmark before any profile or writer promotion.
- [ ] Independently seal protocol behavior/role semantics before adding them to
      a future base schema.
- [ ] Keep independently published chaser components under their existing
      component manifests; do not fold them into this base candidate.
