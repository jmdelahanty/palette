# Batman crop-boundary audit and implementation checkpoint

Date: 2026-08-04

Status: implementation validated locally; no LSF jobs, Zarr writes, registry
changes, or selector changes were performed.

## Decision

The ordinary zebrafish crop workflow now defaults to a fixed `348 x 348` pixel
ROI and the explicit `zero_outside_source_frame` padding policy. Crop size is a
workflow/model input, not a universal crop-schema invariant. Callers may still
declare another exact size. A caller that requires containment can select
`require_fully_contained`, which remains fail-closed.

This resolves the false choice between dropping a valid edge observation and
silently changing its geometry. A padded crop preserves the source detection's
`instance_key`, frame identity, center, and fixed ROI placement; only pixels
outside the camera frame are filled with zero.

## Read-only Batman census

The registry contains 36 active Batman analysis archives and 4,987,449
detection rows. A direct read-only placement census produced:

| Fixed crop | Rows crossing the camera frame | Share | Archives affected |
| --- | ---: | ---: | ---: |
| 512 x 512 | 376,360 | 7.546% | 36 / 36 |
| 374 x 374 | 182,555 | 3.660% | 31 / 36 |
| 348 x 348 | 84,133 | 1.687% | 30 / 36 |

Reducing the default from 512 to 348 therefore removes most of the previous
padding, but not all of it. Explicit padding lineage is still required.

The original 512-pixel preflight failure reported source rows 723-730. For row
723, the source center was approximately `(244.3266, 1529.85)` pixels. NumPy
ties-to-even rounding gives `(244, 1530)`, and a 512-pixel crop begins at
`(-12, 1274)`. The same center with a 348-pixel crop begins at `(70, 1356)` and
is fully contained.

A representative crop that still requires padding at 348 pixels is row 3507
of
`2026-07-21T19-38-32Z_arena_3_Batman/detect_2026-07-25_01-57-08`:

- frame index: `3509`
- instance key: `9531451535306491027`
- authoritative normalized box: `[0.9625, 0.42265625, 0.025, 0.0296875]`
- source-pixel center: `(4342.8, 1907.025)`
- rounded center: `(4343, 1907)`
- 348-pixel top-left: `(4169, 1733)`
- source frame: `4512 x 4512`
- right edge: `4517`, so the final five columns are zero-padded

## Governing behavior

`CropGeometryPolicy` already distinguishes
`zero_outside_source_frame` from `require_fully_contained`. The ordinary crop
preflight now accepts a policy rather than an untyped row-size tuple, computes
the exact padded-row set, and rejects crossings only for the containment
policy. Both the established crop materializer and reusable ROI cache
materializer already copy the in-frame intersection into an all-zero fixed-size
destination.

Every newly completed ordinary crop run now persists and validates:

- the canonical crop geometry policy envelope and digest;
- the exact padding mode;
- the padded ROI count;
- source frame shape;
- fixed crop size and source-camera placement arrays; and
- the source detection/frame/instance lineage already required by the
  canonical coordinate publication.

The crop signature includes the policy digest and padding mode. A historical
run without this lineage cannot be silently treated as equivalent to a new
348-pixel, explicitly padded run.

## Validation results

- 123 focused crop/schema/planner tests passed.
- Edge tests cover left, right, top, bottom, and corner padding with exact zero
  values and exact in-frame placement.
- The former failing archive
  `2026-07-21T19-38-32Z_arena_1_Batman_analysis.zarr` now passes a no-write
  preflight as `348 x 348`, `zero_outside_source_frame`.
- A second Batman archive fails earlier on a stale detection cardinality seal.
  This is intentionally not bypassed by the crop change: old detections must be
  republished through the current canonical detection contract before crop
  publication.

## Safe downstream plan

1. Republish each semantically accepted Batman detection rowset through the
   current canonical detection schema and coordinate/cardinality proofs. Do
   not rerun YOLO merely to change physical schema or crop size.
2. Run the cohort crop preflight with the 348-pixel zero-padding policy.
3. Publish immutable crop geometry and materialize clip-local or whole-video
   pixel caches from that same manifest-bound geometry.
4. Run keypoint and subject-mask inference against those caches. Preserve
   `instance_key`, crop row identity, model identity, preprocessing identity,
   and terminal per-clip status through finalization.
5. Publish recording-level keypoint, quality, refined-keypoint, body-frame,
   raw-subject-mask, quality, and refined-subject-mask runs only after their
   existing contract gates pass.
6. Run tracking/kinematics after refined observation surfaces exist. Chaser
   analytics additionally require accepted stimulus/chaser run identity and
   recording-bound arena/coordinate authority.

Padding preserves observation identity for all these stages. Clipping or
excluding the detection would instead change the rowset and must create a new
refined detection snapshot. Rerunning detection creates a new raw run and new
observation identities. A legacy crop path must not be promoted as a current
contract run.

## Dry-run commands

Single recording:

```bash
scripts/py -m fisheye.utils.crop_batch \
  --zarr-use analysis \
  --source-type detect \
  --fail-on-invalid-plan \
  /groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T19-38-32Z_arena_1_Batman/zarr/2026-07-21T19-38-32Z_arena_1_Batman_analysis.zarr
```

Registry cohort (the recording root is an intentional discovery scope):

```bash
scripts/py -m fisheye.utils.crop_batch \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains Batman \
  --zarr-use analysis \
  --source-type detect \
  --fail-on-invalid-plan \
  /groups/johnson/johnsonlab/jeremy/recordings
```

The command has no `--apply`, so it performs no crop or publication writes.

## Open scientific and lineage questions

- Confirm that the 348-pixel field of view contains the complete fish and the
  contextual margin expected by every selected keypoint and segmentation
  model. Model input resizing does not itself prove that the source ROI is
  adequate.
- Decide whether zero-padded rows need a quality/QC flag or per-side padding
  widths in future crop-v2 extensions. Exact placement already makes the widths
  derivable, but an indexed QC surface may be more convenient for filtering.
- Freeze which current canonical/refined detection authority supplies each
  Batman crop snapshot before running the production DAG.
- Publish the missing Batman stimulus/chaser authorities before treating
  chaser-specific analytics as eligible outputs.
