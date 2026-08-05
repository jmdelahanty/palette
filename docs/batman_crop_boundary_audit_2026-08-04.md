# Batman crop-boundary audit and implementation checkpoint

Date: 2026-08-04

Status: crop-policy and canonical-v3 successor implementations validated;
all 36 selector-ineligible successors and the complete crop-v2 preflight have
passed. No LSF jobs, crop writes, registry changes, or selector changes were
performed.

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

## Canonical-v3 successor checkpoint (2026-08-05)

The accepted raw detection rowsets do not need YOLO reinference merely to
adopt the current storage and coordinate contract. Palette now has a dedicated
raw-only successor boundary which:

- requires an exact `detect_runs/<run>` source selected by the existing
  authoritative detection resolver;
- requires and preserves every persisted `uint64 instance_key` exactly;
- writes canonical coordinate-manifest v3 on bounded node-local scratch;
- uses `detection_published_access_aware_v1` storage;
- atomically imports a new immutable child run;
- keeps the child selector-ineligible and leaves every detection selector and
  registry row unchanged;
- reconsolidates metadata only after the final publication metadata write;
- validates direct/consolidated declarations and source-bound logical content
  after copy-back; and
- tombstones the child and reconsolidates fail-closed if final visibility
  validation fails.

A separate batch boundary freezes exact source paths, dimensions, instance-key
hashes, logical source evidence, selector snapshots, and target IDs before any
apply. Apply re-inspects the complete frozen evidence and refuses drift.

The read-only Batman plan contains all 36 active analysis archives, 5,029,464
frames, and 4,987,449 detections. All candidates use the access-aware profile
and target `detect_canonical_v3_crop_20260805`. Its canonical plan digest is:

```text
4be7bbfdd5fa5676197818800eb0256b07cc43c94bb174dc4799fa81549ae296
```

The plan was first written outside the repository at:

```text
/tmp/batman_canonical_v3_successor_plan_20260805.json
```

The plan, batch summaries, cohort preflight, and all per-recording receipts are
retained under:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.processing_logs/
batman_canonical_v3_successor_20260805/
```

Evidence SHA-256 values are:

| Evidence | SHA-256 |
| --- | --- |
| Frozen plan | `b1f369a0def77bf850d1ce5a39f5f34b90eeea5c96cf9c86394d4feee20a39c3` |
| Cohort crop preflight | `e2f448cd012f61e9abdbee33e698275948ca95dcf9485a8a2805c7693e1ffec2` |
| Canary batch result | `af108631feb19fe5a4752d4e0ae48ea1fbd105f22438e112bda328f9dae087f1` |
| Remaining-35 batch result | `a57bb6aabb352480a1a22bd60f6530e77e990fed94c793df9db14ed48842a858` |

Ten focused publication/plan/preflight tests pass. They include exact key
preservation, selector invariance, recomputed-digest tampering, frozen-source
drift, and an injected final-consolidation failure.

The arena-2 canary was published as an immutable selector-ineligible child in
16.99 seconds. Its 126,214 source and successor `instance_key` vectors have the
same SHA-256, all canonical validation errors are empty, and the pre-existing
`latest` and `latest_complete` selectors remain unchanged. The durable receipt
is:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.processing_logs/
batman_canonical_v3_successor_20260805/
2026-07-21T19-38-32Z_arena_2_Batman.json
```

This canary exposed an adapter distinction rather than a data defect. The old
ordinary `crop_batch` materialized writer consumes the legacy canonical-v2
bound-coordinate graph with arrays directly under `detect_runs/<run>`.
Canonical detection v3 uses its strict run manifest/coordinate catalog and an
`instances` table. The old path remains fail-closed rather than treating the
new representation as the legacy graph.

A new read-only canonical-v3 crop-v2 preflight validates the complete manifest,
nine arrays, physical metadata, active acquisition dimensions, and fixed crop
policy before deriving placement. Against the canary it found 1,436 padded
rows out of 126,214 (1.138%), with maximum `[left, top, right, bottom]` padding
of `[0, 0, 118, 75]`. It performed no crop, selector, or registry writes.

The remaining 35 successors then passed the same frozen-plan gate. Across all
36 durable receipts:

- every publication is complete canonical manifest v3;
- every source/successor `instance_key` SHA-256 pair is identical;
- every profile is `detection_published_access_aware_v1`;
- every selector snapshot is unchanged;
- no child is selector-eligible and no registry row was updated; and
- total per-archive publication time was 232.41 seconds, ranging from 5.25 to
  16.99 seconds.

The complete read-only crop-v2 preflight then passed all 36 archives. It
reproduced the independent census exactly: 84,133 padded rows among 4,987,449
detections, affecting 30 archives. Cohort maximum `[left, top, right, bottom]`
padding is `[103, 73, 121, 104]`. The report is bound to the frozen plan digest
and records zero crop writes, selector updates, and registry updates.

Detection authority activation remains a later, separate reviewed operation.
The future-facing crop publisher also requires a refined-detection identity
snapshot; the next implementation checkpoint is the raw-canonical to
all-accepted refined-v1 transition, not a fallback to the legacy materialized
crop writer.

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

Freeze the canonical-v3 successor cohort without Zarr writes:

```bash
scripts/py -m fisheye.utils.publish_canonical_detection_successor_batch \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --scope /groups/johnson/johnsonlab/jeremy/recordings \
  --path-contains Batman \
  --successor-run detect_canonical_v3_crop_20260805 \
  --plan-json /tmp/batman_canonical_v3_successor_plan_20260805.json \
  --result-json /tmp/batman_canonical_v3_successor_plan_result_20260805.json
```

Apply is intentionally a separate invocation that consumes this exact plan,
requires bounded node-local scratch and a durable receipt directory, and can
select one recording with `--only-recording` for the canary.

Read-only canonical-v3 crop-v2 preflight for the canary:

```bash
scripts/py -m fisheye.utils.preflight_canonical_detection_crops \
  --analysis-zarr /groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T19-38-32Z_arena_2_Batman/zarr/2026-07-21T19-38-32Z_arena_2_Batman_analysis.zarr \
  --detection-run detect_canonical_v3_crop_20260805 \
  --roi-width 348 \
  --roi-height 348 \
  --padding-mode zero_outside_source_frame \
  --allow-selector-ineligible-candidate \
  --result-json /tmp/batman_arena2_canonical_v3_crop_preflight_20260805.json
```

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
