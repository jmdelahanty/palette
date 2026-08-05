# Batman crop-boundary audit and implementation checkpoint

Date: 2026-08-04

Status: crop-policy, canonical-v3, and all-accepted refined-v2 successor
implementations validated; all 36 selector-ineligible raw/refined successors
and the complete crop-v2 preflight have passed. No LSF jobs, crop writes,
registry changes, or selector changes were performed.

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
snapshot; that raw-canonical to all-accepted refined transition is recorded in
the next checkpoint. There is no fallback to the legacy materialized crop
writer.

## All-accepted refined-v2 checkpoint (2026-08-05)

Palette now has a dedicated canonical-v3 to refined-v2 root transition. It is
deliberately an identity-preserving initialization, not a claim that a human or
quality model scientifically reviewed every detection. For every canonical
row it:

- preserves `instance_key`, frame identity, source-acquisition frame,
  normalized/pixel geometry, score, and class exactly;
- allocates root `refined_row_ids` as contiguous `int64 [0, N)`;
- records `source_detect_row_index` as the canonical row position;
- records the source decision as accepted and the source kind as raw detect;
- writes independent exact F+1 offsets for presented and source-audit rows;
- writes refined coordinate-manifest v2 and
  `detection_published_access_aware_v1` storage; and
- remains immutable, selector-ineligible, and registry-ineligible.

The production placement boundary accepts only a complete selector-ineligible
canonical-v3 source using the current published profile. It materializes all
28 refined arrays on bounded node-local scratch, validates identity and source
audit joins, atomically imports the child, reconsolidates once after final
metadata, validates direct/consolidated declarations, and finally binds the
published artifact through the strict refined crop-source adapter. Any failure
stops the cohort and tombstones the imported child fail-closed.

The refined cohort is derived from—not rediscovered independently of—the exact
36-recording canonical successor plan. The inherited canonical plan digest is
`4be7bbfdd5fa5676197818800eb0256b07cc43c94bb174dc4799fa81549ae296`.
The refined plan digest is:

```text
64808a5a057686f42232ff166fe61874b95dddf1a76a6d2a3acc0259dd06edc4
```

The arena-2 canary passed first, including the strict 348-pixel crop-v2
preflight. The other 35 archives then passed the unchanged frozen plan. Across
all 36 durable receipts:

- all 4,987,449 canonical rows are represented exactly once;
- every source/refined `instance_key` SHA-256 pair matches;
- every source/refined frame-offset SHA-256 pair matches;
- every manifest uses refined coordinate-manifest version 2;
- every profile is `detection_published_access_aware_v1`;
- every direct/consolidated metadata comparison and crop-source binding passes;
- every selector snapshot is unchanged;
- every child remains selector-ineligible and no registry row was updated; and
- per-archive publication time ranged from 11.08 to 18.75 seconds, with an
  11.89-second median and 439.72 seconds total.

The no-write refined crop-v2 cohort preflight then reopened all 36 live source
video authorities and prepared and hashed all 13 crop-v2 arrays per archive.
It reproduced the independent canonical census exactly: 84,133 padded rows
among 4,987,449 observations, 30 affected archives, and maximum
`[left, top, right, bottom]` padding `[103, 73, 121, 104]`. This proves the
refined identity boundary and the 348-pixel explicit zero-padding geometry are
compatible before any crop run or pixel cache is published.

Durable evidence is under:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.processing_logs/
batman_accept_all_refined_v2_20260805/
```

| Evidence | SHA-256 |
| --- | --- |
| Frozen refined plan | `20d9a3b63836eea7496e5dbfe6e4a40ff1d371a9ef139e9fe04a1d7f257456f0` |
| Plan result | `0c1e1ce276c69a13f39d9077ef4862b40ba3512561eb1ae5b4d86622f0c96a05` |
| Canary batch result | `d9fe32341c83deb0fd4846c9216b27f31485df680b7b53ee78380d9c89e00d94` |
| Remaining-35 batch result | `68906f3133b718b49ae7e09ec62faa748be3d0b04d5bebeff59eca87087a27ed` |
| Canary crop-v2 preflight | `144b748646d2c3f3484fabe31372bdb79d5c77004213a482d88ede6372c43348` |
| Cohort crop-v2 preflight | `54a6b1d99d0ad8fbd330bc66117b18c4e436d3381ac536304908cd66879f477c` |

The publication implementation checkpoint is Palette commit `1a520795`; the
repeatable single-archive crop preflight checkpoint is `6d42c171`. Thirteen
focused tests now cover raw/refined atomic publication, root identity,
selector invariance, frozen-plan drift, exact padding calculation, and cohort
aggregation.

## Refined analysis-authority activation (2026-08-05)

The 36 all-accepted refined-v2 roots are now the approved analysis authorities
for their recording archives. This is an identity-preserving initialization
approval, not a claim of human or model-quality review. The authority envelope
uses `intended_use=analysis`; it does not authorize training.

Activation consumed the exact refined publication plan above rather than
rediscovering archives. A new immutable activation plan bound every pre-change
manifest digest, final-intent manifest digest, logical-content digest,
publication-owner UUID, dimensions, storage profile, and recording identity.
Its plan digest is:

```text
c04f88d4298db4a821f9ea14f7f3c37a3bb854f6c34c05eacd1beb214353c10b
```

The transaction stages the final-intent manifest and analysis-only authority
provenance, reconsolidates, deeply validates the still-invisible candidate,
then leases one owner/generation epoch and writes
`stage_selector_eligible=true` as the literal final archive write. The exact
pre-commit consolidated `false` versus direct post-commit `true` eligibility
lag is the sole permitted metadata-equivalence exception; selection reads the
authority and commit bit from direct metadata. Failures before that final bit
restore the original manifest/provenance and reconsolidate.

The original arena-2 archive was activated first. Normal, non-benchmark
selection reopened it as `approved_authoritative_refined_v1`; the remaining 35
then completed under the unchanged plan. A separate read-only verifier finally
reopened all 36 archives, deep-validated their 28-array publications, compared
direct/consolidated metadata, validated owner/generation leases and authority
provenance, invoked normal production selection, and reconciled every active
state to its frozen plan entry and durable receipt. Results:

- 36 of 36 authorities valid;
- 4,987,449 of 4,987,449 instance rows bound;
- one selection mode: `approved_authoritative_refined_v1`;
- one physical profile: `detection_published_access_aware_v1`;
- every activation generation is 1 from an authority-free generation 0;
- every authority is analysis-only;
- no registry row was changed.

The guarded activation implementation is bound to Palette commit
`bd1bec0575bfc4189822eb6e2cf5a7a91b451e31`. Focused validation passed 36 tests
before live activation; the added active-state verifier and rollback coverage
then passed 6 focused tests.

Durable activation evidence is under:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.processing_logs/
batman_refined_authority_activation_20260805/
```

| Evidence | SHA-256 |
| --- | --- |
| Frozen activation plan | `cb16c28fc068e2844dfa509c3191f6010cb6ff87834530645d060700e4f6da7f` |
| Plan construction result | `26ba86eb19ec4a214b0af7784452297b8be642c28fdbd1f5ce3b1476f8b90a06` |
| Arena-2 canary result | `3ebf334eb747d0d0bced5eb8ba6c2a9472c5e8aeeb470867700344442313cf4c` |
| Remaining-35 result | `970e29f0710b01daf25438e3942dae983963e0287c7148cdd5f3d8e46bac8839` |
| Independent 36-archive verification | `a501d784634aff4ff72a5269008f42fb9e31d4f1b429f9e0a80240cd80a51225` |

The directory also contains 36 per-recording activation receipts.

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

Freeze the all-accepted refined-v2 cohort from the exact canonical plan without
Zarr writes:

```bash
scripts/py -m fisheye.utils.publish_accept_all_refined_detection_batch \
  --canonical-successor-plan /tmp/batman_canonical_v3_successor_plan_20260805.json \
  --refined-run refined_accept_all_v2_crop_20260805 \
  --plan-json /tmp/batman_accept_all_refined_v2_plan_20260805.json \
  --result-json /tmp/batman_accept_all_refined_v2_plan_result_20260805.json
```

Run the full no-write refined crop-v2 cohort preflight:

```bash
scripts/py -m fisheye.utils.preflight_refined_detection_crops \
  --plan-json /tmp/batman_accept_all_refined_v2_plan_20260805.json \
  --result-json /tmp/batman_accept_all_refined_v2_crop_cohort_preflight_20260805.json
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
