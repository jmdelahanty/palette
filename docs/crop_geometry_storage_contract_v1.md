# Crop Geometry Storage Contract v1

Status: implemented for selector-ineligible refined-source production
candidates, frozen cohort planning, and DAG consumption; not a production
selector or writer default

Date: 2026-07-29; workflow-default clarification 2026-08-04

## DAG Consumer Checkpoint

The storage checkpoint was reviewed against current `sun` and integrated on a
separate branch. The shared production entry point is exposed through one
partition-independent workflow fragment and CLI. Both clipped and whole-video
workflows must call this same recording-level publisher after they have
produced and approved one full-acquisition refined-detection snapshot. Clip
boundaries remain compute provenance; they do not define a second crop schema
or namespace.

Downstream flat-cache and keypoint planning bind strict crop runs through a
validated `palette.crop_geometry.run_reference` containing the immutable run
manifest and logical-content digests. Maintained acquisition crop-video runs
use a separate `signed_current_source_v1` reference until their typed manifest
envelope is complete. Historical signature/revision pairs remain available
through an explicitly labelled compatibility profile; unversioned historical
runs are accepted only by the local temporary-cache reader's explicit
compatibility path.

Production candidates stamp `palette.zarr_run_completion.v1` complete/failed
markers in addition to their crop-specific state. They remain selector-
ineligible, registry-unregistered, and unselected. Guarded selector activation
remains a separate blocker. Collection-wide refined-detection activation is
complete for the frozen Batman cohort, and the workflow fragment must consume
that authority without bypassing either contract gate.

## Selector-Ineligible Read Benchmark

The retained `crop_geometry_coordinate_catalog_crimson_20260728_v2` canary was
read directly from PRFS on both the Palette workstation and LSF compute host
`h07u26`. It contains 22,926 rows, 13 arrays, and 3,671,056 logical bytes. Each
of three passes reopened the run, resolved its immutable manifest-bound run
reference, streamed every array, and verified all 13 decoded SHA-256 values.
The first pass also requested page-cache eviction with
`POSIX_FADV_DONTNEED` for all 30 physical files.

| Client | Median direct open | Median consolidated open | Median full scan | Median full-scan rate |
| --- | ---: | ---: | ---: | ---: |
| Workstation | 25.8 ms | 9.2 ms | 86.1 ms | 40.7 MiB/s |
| LSF `h07u26` | 4.5 ms | 4.2 ms | 28.7 ms | 122.0 MiB/s |

The compute run was LSF job `153227338` and completed without stderr. Its
1,024-row, four-array windows took approximately 5.2--5.9 ms after the first
cold window; workstation windows took approximately 23--28 ms after the first
cold window. Evidence is retained under
`recordings/.palette_benchmarks/crop_snapshot_reads/20260729_crop_v2_dag_integration_0d31e0b9/`.

This closes the representative read gate for the small coordinate canary. It
does not substitute for a million-row publication/read benchmark against the
first approved production-shaped refined authority.

## Purpose

This contract freezes the immutable geometry and lineage needed to extract
downstream crop pixels without persisting `roi_images` in an analysis Zarr.
Detection identity remains upstream and unchanged. A single refined-detection
snapshot may produce multiple crop runs for different purposes or crop-size
policies.

## Exact Layout

The run lives at `crop_runs/<run>` and contains exactly these arrays:

| Array | Dtype and shape | Meaning |
| --- | --- | --- |
| `instance_key` | `uint64[N]` | Preserved observation identity; not subject identity. |
| `source_refined_row_ids` | `int64[N]` | Stable row identity in the bound refined lineage. |
| `frame_indices` | `int64[N]` | Sorted acquisition-camera frame per row. |
| `source_acquisition_frame_index` | `int64[N]` | Exact full-acquisition identity mapping. |
| `frame_row_offsets` | `int64[F+1]` | CSR frame-to-row index supporting zero/one/many rows. |
| `bbox_norm_coords` | `float32[N,4]` | Authoritative refined bbox in normalized `cx,cy,w,h`. |
| `bbox_img_xyxy` | `float32[N,4]` | Exact source-camera pixel projection. |
| `centers_img_xy` | `float32[N,2]` | Exact persisted source-camera center. |
| `roi_coordinates_full` | `int32[N,2]` | Exact integer source top-left `[x,y]`. |
| `roi_sizes_full` | `int32[N,2]` | Exact positive `[width,height]`; never assumes 512. |
| `source_crop_xywh` | `float32[N,4]` | Float32 projection of the integer extraction window. |
| `bbox_roi_xyxy` | `float32[N,4]` | Refined bbox translated into crop-local pixels. |
| `source_row_signature` | `uint8[N,32]` | Exact incremental reuse signature. |

`roi_images`, `roi_images_delta`, `frame_counts`, `n_detections`,
`detection_indices`, and `source_frame_indices` are forbidden. Dense pixels
belong to keyed work packages, caches, or immutable training artifacts.

Coordinate meanings are assigned by the shared v1 coordinate-surface catalog:
normalized authority, source-camera continuous pixels/half-open edges, integer
extraction origin and extent, and ROI-local half-open edges remain distinct.
The crop schema exposes this catalog separately without changing the already
frozen v1 run-manifest bytes. Persisting it in the run envelope requires a new
manifest version rather than silently revising v1.

## Crop Policy

Every run binds a canonical-JSON/SHA-256 crop policy containing:

- purpose;
- persisted-center placement;
- NumPy ties-to-even center rounding;
- `rounded_center - floor(size/2)` top-left derivation;
- `fixed_per_run` or `variable_per_row` sizing; and
- `require_fully_contained` or `zero_outside_source_frame` padding.

The policy digest is part of crop identity and row signatures. It is never
part of `instance_key` or the source refined-detection identity.

The maintained zebrafish pose/subject-mask workflow currently defaults to a
fixed 348x348 window with `zero_outside_source_frame`. This is only a workflow
default: every run still persists its exact width, height, padding mode, and
policy digest, and other scientific purposes may choose different dimensions.

## Refined and Pixel Authorities

The first profile requires the complete `instances` rowset of one validated
full-acquisition refined-detection v1 snapshot. The crop manifest binds its:

- run ID and run-manifest digest;
- decoded logical-content digest;
- recording identity;
- lineage UUID and snapshot UUID; and
- exact `instance_key`, refined row ID, frame, offsets, and bbox arrays.

The source-pixel authority separately binds recording/camera identity,
acquisition frame domain, dimensions, decoded `uint8` grayscale semantics, and
the digest of the external authority manifest. The shadow publisher accepts a
typed, already-proven authority.

There are two maintained base pixel-source families plus an explicit hybrid;
none is a fallback for another:

| Source family | Pixel domain | Required lineage |
| --- | --- | --- |
| Full-frame camera video | Decode the full camera frame, then apply each persisted integer crop window and padding rule. | Camera-frame identity, full-frame video authority, crop policy, and `roi_coordinates_full`/`roi_sizes_full`. |
| Acquisition crop video | Decode the already-cropped Orange video frame directly. | Crop-video frame index, recording-frame mapping, crop-meta row, full-frame `source_crop_xywh`, and any explicit supplemental-row routing. |
| Hybrid acquisition/full-frame | Route each row explicitly to acquisition crop video or a full-frame-derived supplemental cache. | Both authority bindings plus `source_pixel_kind_codes` and the corresponding source-row index. |

Both use the accepted Orange monochrome PyNvVC luma semantics today. Their
pixel-contract `source_pixels` values remain distinct (`raw_camera_video`
versus `acquisition_crop_video`) because their frame lookup and geometry are
not interchangeable. Hybrid artifacts declare
`hybrid_acquisition_crop_video_offline_supplement` and bind the per-row routing
array. A cache or work package must bind the exact source family that produced
its bytes.

New acquisition crop-video analysis runs also publish stable observation keys
and per-row source signatures. This makes the maintained geometry/pixel-source
snapshot eligible for keyed keypoint and subject-mask work packages without a
later identity backfill. The signed-current profile is an explicit transition
surface for maintained acquisition sources; strict crop manifests remain the
preferred publication envelope once the acquisition stream has a complete
digest-bound authority record.

`bind_refined_crop_source_pixel_authority()` is the future-facing strict
authority binder for a single external full-frame source video. It reopens the
direct archive metadata, requires the mirrored acquisition publication state to be
`published_canonical_v1` in `external_video_v1` mode, reloads the sealed import
ownership and acquisition-camera frame, resolves the exact source-video
locator, and recomputes the live `stat_v1` fingerprint. It then binds the
`orange_mono_pynvvc_luma_uint8_v1` full-frame decode policy into the crop
authority digest. Recording identity, camera identity, `F`, width, and height
must exactly match the refined handoff. Materialized source arrays, acquisition
crop videos, and clipped collections intentionally require separate typed
authority binders. Acquisition crop video is a maintained current source; its
separate binder reflects different lineage, not legacy status. These are
contract boundaries, not compatibility adapters:
they do not probe dtypes, translate aliases, infer identities, or fall back to
another source.

## Manifest and Publication Gate

The exact envelope is persisted at:

```text
crop_runs/<run>/zarr.json.attributes.run_manifest
```

It contains the logical schema, concrete dimensions, crop policy, storage
plans, refined and pixel authorities, signature spec, per-array decoded
digests, metadata-declaration digest, completion state, and selector
eligibility.

The gate fails closed unless:

1. the envelope and every nested field set match the frozen builders;
2. the outer, logical-content, policy, source, pixel, and signature digests
   agree;
3. decoded crop arrays satisfy all geometry, identity, and CSR invariants;
4. row signatures recompute exactly;
5. decoded refined source arrays and manifest match the crop binding;
6. direct and consolidated metadata are equivalent;
7. physical declarations match `StoragePlan` and the exact codec chain; and
8. every shard write has whole-shard single-writer ownership.

Metadata declaration digests retain attributes. Only the circular root
`run_manifest` attribute and representational `consolidated_metadata` envelope
are excluded.

## Physical Policy

The initial unpromoted crop profile is `published_http_v1`:

- approximately 1 MiB uncompressed inner chunks derived from bytes per row;
- approximately 32 MiB immutable indexed shards where the array is large
  enough;
- `frame_row_offsets` classified `EAGER`;
- all row-aligned identity/geometry/signature columns classified `WINDOWED`;
- Zarr v3 with the shared bytes + Zstd/CRC and indexed-sharding contract; and
- complete trailing row axes in every chunk and shard.

On a Sleepyfish-sized `N=1,187,087`, `F=1,188,000` plan, the 13 arrays are
estimated at 14 payload objects. `source_row_signature` is the only two-shard
column; the remaining arrays each fit in one large shard.

## Successor Reconciliation

Names such as `D2`, `C2`, `Kraw2`, and `Kref2` are explanatory generation
labels only. They are not persisted array names or required run IDs. Actual
artifacts use caller-supplied immutable child names below their versioned run
families, and manifests bind exact run IDs and digests.

A compacted refined-detection successor produces a new complete crop run; it
never appends rows to its parent crop. `crop_successor.py` now implements the
selector-ineligible reconciliation and standalone publication boundary. It:

- requires the target refined manifest to bind the crop parent's immediate
  refined snapshot;
- requires the same recording, refined lineage, crop policy, pixel authority,
  frame domain, and source-camera dimensions;
- compares every row-local identity and geometry field by `instance_key`;
- classifies exact reused, added, changed, and retired key sets; and
- publishes a complete immutable successor without selectors, registries, or
  production-state changes.

The persisted `source_row_signature` remains bound to the exact refined
snapshot and is not weakened. Cross-snapshot planning uses a separate
receipt-only reconciliation signature: it omits changing run/snapshot IDs but
retains the stable lineage and every row-local geometry input. Parent and
target snapshot IDs and manifest digests remain explicitly bound by the
successor receipt.

The real integration test compacts a manual detection addition, publishes the
new crop snapshot, and proves that the three surviving rows are reusable while
only the new observation is computed. Geometry publication still writes all 13
logical arrays as a fresh immutable Zarr; the reuse plan primarily controls
downstream pixel materialization and records invalidation precisely.

## Current Safety Boundary

The standalone writer can create only a fresh child below `/tmp`,
`.palette_scratch`, or `.palette_benchmarks`. The production-candidate boundary
`publish_crop_geometry_production_candidate()` additionally:

1. binds only the approved authoritative refined-detection snapshot;
2. binds and re-verifies the exact published external-video pixel authority;
3. materializes and fully validates a geometry-only crop run on bounded
   node-local scratch;
4. atomically imports the immutable run into `crop_runs/<run>`;
5. rebuilds the exact run manifest after publisher transaction metadata exists;
6. reconsolidates and revalidates the complete imported publication; and
7. proves root and crop-family selector attributes are unchanged.

It does not register an artifact, update `latest`, activate a selector, change
a production default, or replace an existing run. A post-import failure is
retained as an owner-bound selector-ineligible failed child rather than being
made authoritative.

Before production integration:

- [x] implement and validate the external full-frame source-pixel authority
      binder;
- [x] implement node-local materialization and atomic selector-ineligible
      production-candidate import;
- [x] implement a benchmark-only package builder that exercises the production
      candidate path and preserves publication/read handoff evidence;
- [x] obtain parallel Palette producer/DAG review of this exact contract;
- [x] publish a small immutable canary outside production selectors;
- [x] pass the Crimson canonical-v3/refined-v2/crop-v2 coordinate archive gate;
- [x] bind full-frame and acquisition crop-video materializations to distinct
      exact pixel contracts and reject cross-source cache substitution;
- [x] run the real Palette pixel materialization and downstream keypoint/mask
      compute canary;
- [x] benchmark representative row/window/full reads on workstation and LSF;
- [x] benchmark selector-ineligible publication and record object counts;
- [x] benchmark representative row/window reads in Crimson;
- [x] implement exact refined-detection-to-crop successor reconciliation and
      selector-ineligible standalone publication;
- [x] test a real compacted detection addition through complete crop
      publication with unchanged-row reuse;
- [x] allow an explicit selector-ineligible clipped refined candidate to feed
      crop-v2 while revalidating its complete per-clip evidence and the live
      acquisition pixel authority;
- [x] allow the keypoint clip receipts and recording finalizer to consume this
      standalone crop archive without moving pixels into the analysis Zarr;
- [x] freeze an authority-bound 36-recording production-candidate cohort after
      exact crop preflight without publishing any crop arrays;
- [ ] insert the successor publisher into the production DAG's atomic
      selector-ineligible import path;
- [ ] feed its added/changed rows into the raw-keypoint successor materializer;
- [ ] benchmark production-candidate publication and reads at recording scale;
- [ ] add a typed purpose/profile selector with guarded activation; and
- [ ] migrate production writers only after downstream completeness passes.

The first recording-scale Batman candidate plan was frozen at Palette commit
`0f576d2d`. It binds 36 approved refined authorities, 4,987,449 rows, the
shared 348-pixel zero-padding policy, and `published_http_v1` under plan digest
`e39781fc5e46c9add5fdcbab5a0b7fae7da4fd2d5124134a226c4f66b6a1b10f`.
No crop array, selector, or registry record was written. The real-Zarr
publication test suite remains an explicit gate before the single-recording
canary, so this plan freeze does not complete the recording-scale publication
benchmark item above.

The DAG review closed on the integration branch after identifying and fixing
the standard completion-marker gap and replacing the cache planner's legacy-
only identity requirement with the manifest-bound run reference. That review
alone did not imply pixel decode or model-consumer validation; the bounded gate
recorded below supplies that later evidence.

The combined publisher was revalidated at reconciliation commit `2b9b816a`.
The selector-ineligible integrated canary retained the Crimson-tested logical
content digest while adding and validating the standard complete-run envelope.
Its immutable paths and hashes are recorded in
`docs/crop_geometry_read_fixture_contract.md`.

The Crimson coordinate canary passed at implementation commit
`ce478c7d13d2f870e6c711308090e28364872602` and evidence commit `4100719`.
The supplied evidence SHA-256 is
`9918615e142a1f946eb98865f46e264cacff23a2885e008ce0030d87efc6fd7d`.
Crimson validated exact typed opens, CSR offsets, lineage, and ROI-to-source
camera transforms; this closes the coordinate-consumer gate but does not
activate a Palette crop selector.

The downstream pixel/materialization gate passed as LSF job `153227442` at
Palette commit `229ceadd600b27c384684e474fe3940fd077ac13`. It materialized one
`2,048`-row acquisition-video package on node-local scratch and ran the real
YOLO keypoint and unified subject-mask consumers against that one package.
Both outputs retained exact crop-row, `instance_key`, source-signature, and
package identity. The immutable receipt is under
`.palette_benchmarks/crop_pixel_materialization/workflows/`
`20260729_redscare_acquisition_crop_consumers_229ceadd_v4/receipt.json` with
SHA-256
`8fa5ec642b34e1f365ae6b24e2513cc5e06d213e10acb59e8694e590f06fb0fe`.
This closes the bounded pixel-consumer integration gate; recording-scale
publication and guarded selector activation remain separate checkpoints.
