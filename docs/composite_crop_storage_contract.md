# Composite Crop Storage Contract

<!-- contract-meta
status: implemented-canary
schema: palette.composite_crop v1
last_updated: 2026-07-18
owner: jeremy
depends_on: docs/stable_identity_incremental_materialization_decision.md,
  docs/instance_track_subject_identity_contract.md,
  docs/dask_zarr_write_safety.md
-->

## Purpose

A composite crop run is a complete immutable logical crop rowset without a
second dense copy of every unchanged ROI. It maps each target row to either:

- one row in an exact complete standalone materialized crop base; or
- one newly computed row in the composite run's dense delta payload.

This is a physical storage optimization. It does not weaken keyed lineage,
source-signature validation, completion, or provenance requirements, and it is
not an active partially materialized run. Every target row resolves at the time
the run is marked complete.

## Persistence Lifecycle And Downstream Fan-Out

Committing an added detection or bbox edit does **not** eagerly persist new crop
pixels. The committed refined-detection revision first records exact identity,
geometry, source signatures, and downstream stale/invalidation state. If no
crop-dependent consumer is requested, no canonical crop-pixel artifact is
created merely because the authoring edit exists.

Pixel persistence begins when reconciliation schedules work that consumes the
crop, such as keypoint or subject-mask inference. The default incremental
fan-out artifact is a keyed crop pixel work package, not a new composite crop
run:

```text
committed refined-detection revision
  -> record keyed invalidation/materialization plan
  -> crop pixel work-package preparation, when a pixel consumer is requested
       -> persist and validate affected ROI pixels once
       -> keypoint inference and subject-mask inference may fan out concurrently
  -> validate and publish downstream replacements independently
```

The prepared work package is a durable shared derived cache, not authoring
authority. Detection geometry, `instance_key`, the source-pixel fingerprint,
crop parameters, and the pixel contract define the reproducible crop. Persisting
the delta at the fan-out boundary nevertheless matters because it:

- guarantees that keypoint and mask jobs consume identical pixels;
- prevents each branch or retry from independently decoding and cropping the
  same source frame;
- gives independently scheduled jobs one stable, provenance-addressable input;
- survives worker, node, or queue failure; and
- allows the two inference branches to run concurrently after one prerequisite.

Preview-only or exploratory reads may crop transiently through an approved live
reader or temporary cache and need not publish a composite. Transient pixels
must not be cited as a completed downstream source. Once a durable keypoint or
mask run binds to a crop run, its provenance records that exact crop run and
pixel contract; the referenced crop/base artifacts remain retained for audit.

The package contract and operator are defined in
`docs/crop_pixel_work_package_contract.md`. Composite crop runs remain an
optional complete base-plus-delta storage strategy when a long-lived logical
crop snapshot should resolve unchanged pixels from a prior base. They are not a
prerequisite for ordinary incremental inference.

The production reconciler/DAG has not yet made work-package preparation a
separately observable registry state. Its intended states are:

```text
planned -> crop_pixels_ready -> downstream_running -> downstream_complete
```

Failure before `crop_pixels_ready` leaves the prior complete crop selected and
does not release keypoint or mask dependents. Failure in one downstream branch
does not invalidate the prepared delta or prevent an independent branch from
being retried according to DAG policy.

## Depth-One Schema

The run root contains the complete compact target authority, including
`instance_key`, `frame_indices`, `bbox_norm_coords`, crop coordinates,
`source_row_signature`, optional row lineage, frame counts, and the persisted
`materialization_plan`. It intentionally has no top-level `roi_images` array.
That absence makes unaware materialized-only readers fail instead of consuming
an incomplete dense array.

Required attrs are:

```text
crop_storage_mode = "composite"
composite_crop_schema_id = "palette.composite_crop"
composite_crop_schema_version = 1
composite_reference_depth = 1
composite_base_crop_run = <exact crop_runs child>
roi_size = [height, width]
roi_pixel_contract = <strict JSON-compatible contract>
```

The physical payload is:

```text
crop_runs/<run>/composite_payload/
  source_codes                 uint8[N]   0=base, 1=delta
  source_row_indices           int64[N]   row in the selected source
  delta_target_row_indices     int64[D]   target rows backed by delta
  delta_instance_key           uint64[D]  exact delta-row identity
  roi_images_delta             uint8[D,H,W]
```

The base must be an explicitly complete standalone materialized run with dense
`uint8 roi_images`, modern unique identity, Phase-1 source signatures, the same
ROI shape, and the same pixel contract. A composite may not reference another
composite. The maximum reference depth is therefore one.

## Validation

Completion fails closed unless:

- source mappings cover all `N` target rows exactly once;
- every source code is known and every row index is nonnegative and in bounds;
- reused base rows are one-to-one and match target `instance_key` and
  `source_row_signature` values;
- every delta row is referenced exactly once and its target row and key agree;
- target and base signature-spec digests agree for reused rows;
- target/base ROI shape, dtype, and pixel contracts agree;
- the base remains complete and standalone;
- source identity, content signatures, optional lineage, and rowset fingerprint
  have not changed while the delta is written; and
- completion and run provenance contracts pass.

The writer reads back every newly written delta chunk. It reads zero dense base
pixels during publication. Compact identity, geometry, signature, and mapping
arrays may be held in memory; source frames and dense ROI payloads are processed
in bounded batches.

## Reader Boundary

`fisheye.shared.crop_image_source.CropImageSource` is the supported resolver.
It presents one read-only logical `uint8[N,H,W]` array and coalesces requested
base/delta rows into bounded source reads. YOLO keypoint inference, modern
subject-mask inference, retry paths, and ROI-cache construction already use
this boundary.

Traditional training, export, tuning, and historical utilities that require a
literal `crop_runs/<run>/roi_images` must continue to use
`resolve_materialized_crop_run`. That resolver explicitly rejects composites.
Such consumers need a future standalone compaction/export step before they can
accept a composite source; they must not grow ad hoc base-plus-delta logic.

## Publication And Compatibility

Composite creation is unselected by default. Marking the run complete does not
change `latest`, `latest_complete`, `latest_materialized`, or `latest_any`.
This permits safe production canaries without changing existing readers.

Explicit composite promotion is a separate opt-in operation. It updates only:

```text
latest_complete
latest_any
latest_composite
publication_generation
```

It deliberately leaves `latest` and `latest_materialized` on the standalone
base. Mixed-mode `CropImageSource` readers may follow `latest_any`; traditional
materialized readers continue to follow the standalone pointer. Publication
still requires serialized ownership of the recording's `crop_runs` parent.

## Retention, Deletion, And Compaction

The base is retained while any complete, running, or failed composite child
references it. Palette crop overwrite/cleanup paths call
`assert_crop_run_unreferenced` before deletion. Operators must delete or compact
dependent composites first; direct filesystem deletion is outside the contract.

Long chains are forbidden. When another edit targets a composite rowset, a
compaction job first resolves the composite one bounded output chunk at a time
and publishes a new standalone complete base. A later composite may reference
that new base. Compaction validates exact key/signature parity and changes
selection pointers only after the standalone output is complete.

Safe aligned-object reuse remains a possible later optimization for a target
whose complete physical chunks or shards exactly match the base metadata and
row placement. It is not used implicitly. The explicit composite mapping is the
fallback whenever byte-level identity and ownership cannot be proven.

## Operator CLI

Dry-run is the default:

```bash
scripts/py -m fisheye.utils.materialize_incremental_crop \
  /path/to/recording_analysis.zarr \
  --source-rowset-path refined_detect_runs/<exact-run> \
  --source-pixel-fingerprint <stable-pixel-fingerprint> \
  --roi-size 512 512 \
  --base-crop-run <exact-standalone-base> \
  --payload-strategy composite \
  --output-run <new-run>
```

`--apply` writes and validates the unselected run. Adding
`--promote-composite` performs the explicit composite-aware pointer update.
Production apply operations belong in LSF compute jobs, never on a login node.

## Current Scope

Version 1 supports Zarr v3, grayscale `uint8 raw_video/images_full`, an exact
modern detection/refined-detection source row group, and one Phase-1 standalone
crop base. External-video decode, clipped proxy resolution, registry projection,
standalone compaction/export, and downstream keypoint/mask base-plus-delta
formats are subsequent adapters.
