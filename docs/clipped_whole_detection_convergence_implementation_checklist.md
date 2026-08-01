# Clipped and Whole-Video Detection Convergence

- Status: implementation planning baseline
- Date: 2026-07-27
- Scope: production inference from physical video inputs through canonical
  detection, quality, arena gating, refinement, crop, keypoint, and subject-mask
  authorities
- Related design: `docs/production_dag_recording_layout_design.md`
- Related storage work: `docs/shared_zarr_storage_policy_design.md`

## Purpose

Palette should support a recording whose pixels live in one video and a
recording whose pixels live in many clips as equally first-class production
inputs. The physical layouts require different work partitioning, but they
must not produce two scientific APIs.

The intended boundary is:

```text
recording-layout adapter
  whole video: one physical work unit + identity frame map
  clipped:     N physical work units + proven local-to-recording frame map
                               |
                               v
                 unbound worker detection evidence
                               |
                 recording-level bind and assembly
                               |
                               v
                  detect_runs/<canonical snapshot>
                               |
                   shared quality / gate / refine
                               |
                               v
             refined_detect_runs/<canonical snapshot>
                               |
                   shared downstream stage APIs
```

The canonical analysis Zarr should expose science-ready detection authorities
through `detect_runs` and `refined_detect_runs`, regardless of source layout.
Clip-local worker outputs may remain temporary or retained audit evidence, but
they must not become an alternative consumer-facing detection authority.

## Summary finding

The overall architecture is sound at the layout-adapter boundary. Clipped
execution should differ in source-media enumeration, frame-map proof,
scheduler packaging, cache ownership, and cross-boundary reconciliation. It
should not differ in model semantics, observation identity, coordinate
authority, canonical run families, downstream readers, validation strength,
or registry meaning.

Current crop, keypoint, and refined-subject-mask workflows mostly follow this
rule. Raw and refined detection publication do not yet fully follow it.

## Appropriate layout differences

| Concern | Whole-video input | Clipped input | Reason the difference is valid |
|---|---|---|---|
| Physical media | One video work unit | Multiple clip work units | Pixels are physically partitioned differently. |
| Frame mapping | Identity mapping into the recording timeline | Checksummed clip-local to canonical-frame mapping | Clip-local frame numbers are not recording frame numbers. |
| Scheduler packaging | Commonly one task per recording | LSF array or bounded bundles over clips | The source files provide independent compute ownership. |
| Boundary handling | Input is already continuous | Stateful temporal results must be reconciled across clips | A jump, gap, or track state can cross a clip boundary. |
| Decode/cache ownership | One source stream and cache | Per-clip decode/cache units with a merged logical view | This preserves efficient source locality without changing consumers. |
| Provenance | Recording and source-video identity | Recording identity plus clip, local-frame, and source-row lineage | Extra lineage is required to reproduce a clipped result. |

These differences belong in adapters, work-unit plans, lineage arrays, and
provenance. They should not require scientific algorithms to branch on a
`clipped` flag.

## Required common behavior

Both layouts must converge on all of the following:

- the same selected detection model and verified model digest;
- the same detection algorithm version, precision declaration, thresholds,
  resize policy, and coordinate conventions;
- one canonical recording-frame domain;
- stable, recording-scoped `instance_key` values;
- one science-ready raw authority under `detect_runs/<run>`;
- one quality schema and label-precedence implementation;
- one selected-arena-geometry and keyed detection-gate contract;
- one science-ready refined authority under `refined_detect_runs/<run>`;
- identical crop, keypoint, and subject-mask source interfaces;
- equivalent indexed-sharding and immutable-publication policies;
- exact cross-stage identity validation; and
- the same registry stage meanings and generic reader behavior.

## Current implementation assessment

### Recording layout and scheduler

`fisheye.cluster.recording_layout` already represents both layouts using a
layout-neutral recording target and one or more video work units. A one-video
recording is a one-member source collection with an identity frame mapping.
This is the correct abstraction.

LSF arrays and bundling remain execution-plan evidence, not scientific
provenance. It is appropriate for a clipped recording to use an array over
clips and for a whole-video cohort to use an array over recordings.

### Raw detection: incomplete convergence and a broken cutover seam

Whole-video detection currently uses the node-local atomic publisher and
publishes a complete top-level `detect_runs/<run>` authority.

The new clipped path is internally inconsistent:

1. `plan_clipped_detect_refine_workflow.py` plans clip-local targets beneath
   `clips/<clip>/cameras/<camera>/detect_runs/<run>`.
2. `run_detection_artifact.py` declares `RUN_FAMILY =
   "detection_artifact_runs"` and places that family in the artifact's
   `intended_target_group_path`.
3. `run_clipped_detection_work_unit.py` imports using the intended artifact
   path, then validates the distinct planner-provided `detect_runs/...` path.
4. The worker unit test mocks build, import, and validation independently, so
   it does not exercise this cross-module path contract.

The artifact contract also deliberately forbids `instance_key` and describes
its coordinate authority as unbound. That is appropriate for worker evidence,
but `materialize_clipped_detect_quality_source.py` requires a modern
`instance_key` on each selected raw detection group. There is currently no
explicit binder between those two contracts.

This seam must be repaired before another new clipped production campaign is
trusted.

### Detection quality

Clipped quality processing correctly recognizes that jump/blip state can cross
clip boundaries. Its recording-level reconciler operates on canonical frame
indices and preserves `instance_key`.

This is not fundamentally a clipped-only algorithm. The common implementation
should accept a recording-ordered canonical detection snapshot. A whole video
then becomes the one-member/no-physical-boundary case, while a clipped source
retains clip-boundary provenance for validation.

### Arena geometry and detection gating

The selected-geometry and gate materializers are already layout-neutral.
`materialize_registered_detection_gate` accepts either a canonical
`detect_runs/<run>` or the current recording-ordered clipped source adapter and
emits the same keyed, sharded `detection_gate_runs/<run>` table.

The remaining work is recipe integration: refinement must consume the exact
gate artifact selected by the immutable plan and verify identical ordered
`instance_key` values.

### Refined detections

The historical clipped authority is a finalized collection manifest selecting
per-clip raw/refined runs through `refined_detect_runs.attrs.latest_collection`.
The registry understands this compatibility representation.

`publish_clipped_refined_detect_snapshot.py` can flatten such a collection into
a recording-level `refined_detect_runs/<run>`, including canonical frame and
clip lineage. It is not yet the generalized production finalizer and does not
yet use the complete new atomic/frozen refined-detection storage contract.

The target representation is one immutable recording-level refined snapshot.
Clip ownership should remain recoverable through lineage, not through a
consumer requirement to open 22 independent selected runs.

Sparse edits and compaction may continue to route an edited observation back
to its owning clip-local evidence using `instance_key` and clip lineage. That
internal edit-routing requirement does not justify a different public read
interface.

### Crop and ROI cache

The clipped workflow appropriately builds pixels from clip-local media and
then publishes a merged recording-level crop geometry proxy. Whole-video and
clipped inference can both feed the same `CropImageSource`/flat-ROI-cache
interface after that adapter boundary.

The canonical crop authority should retain:

- canonical recording frame indices;
- `instance_key` copied from the exact refined detection snapshot;
- source refined row identity;
- source clip and local-frame lineage when applicable; and
- the exact pixel/cache representation used by downstream inference.

### Keypoints

Clipped keypoint inference writes independently owned shard runs and then
finalizes them into ordinary top-level `keypoints_runs/<run>` and
`refined_keypoints_runs/<run>` authorities. Whole-video inference writes the
same final families more directly.

This is the desired pattern: execution differs, final consumer surface does
not. The finalizer's exact `instance_key` checks against the target crop should
remain the common standard.

### Subject masks

Clipped subject-mask inference similarly uses per-clip shard/package evidence
and ultimately publishes a top-level `refined_subject_masks_runs/<run>`.
Whole-video processing uses top-level raw and refined families directly.

Retaining raw probability evidence as a clip collection may be reasonable
because it is very large and read-only. It must nevertheless be represented by
one manifest/capability so generic downstream code does not branch by physical
layout. The final refined mask authority must have identical dense
`masks_roi`, metrics, contour, component, and identity contracts.

### Validation

`clipped_inference_validate.py` currently performs stronger recording-wide
identity validation than `whole_recording_analysis_validate.py`. It compares
ordered or exact keysets across quality sources, refined detections, crops,
keypoints, refined keypoints, raw masks, and refined masks, in addition to
validating parent-frame coverage.

That strength should become the shared recording-level validator. Layout-only
checks should be additive:

- clipped: prove clip coverage, ordering, non-overlap, and local-to-parent
  mappings;
- whole: prove the identity mapping and source-video authority.

After layout validation, both must run the same cross-stage identity and
completion gates.

### Registry and readers

Registry reconciliation currently treats a valid `latest_collection` manifest
as a recording-level detection/refinement authority when no top-level run is
selected. This is valuable compatibility behavior, but it should not be the
steady-state production representation.

Generic detection resolution presently understands top-level `detect_runs` and
`refined_detect_runs` more naturally than collection manifests. Publishing a
canonical recording-level snapshot will simplify the registry, Crimson, review
tools, training exporters, and analytics readers without discarding clip
lineage.

## Target publication rule

The production DAG should publish science-ready detections only through the
canonical run families:

```text
detect_runs/<run>
refined_detect_runs/<run>
```

An unbound worker artifact is not a detection authority. It must remain
selector-ineligible and must never advance `latest` or `latest_complete`.

Preferred steady-state handling is:

1. Write each work unit to node-local scratch.
2. Retain a checksummed artifact and receipt outside the canonical analysis
   Zarr when transfer/retry evidence is needed.
3. Bind and assemble the complete recording from the frozen work-unit set.
4. Atomically publish one recording-level `detect_runs/<run>` candidate.
5. Validate the newly published candidate from its final location.
6. Activate selectors and refresh the registry only after validation.

If unbound artifacts must be retained inside the analysis Zarr temporarily,
`detection_artifact_runs` must remain explicitly internal, selector-ineligible,
and absent from generic reader resolution. It should be retired once external
artifact retention and retry behavior have equivalent operational evidence.

## Implementation checklist

### Phase 0: freeze the integration contract

- [ ] Coordinate with the refined-detection storage work and use its frozen
  schema/manifest builders rather than constructing run metadata manually.
- [ ] Decide whether worker artifacts are retained outside the Zarr or under an
  explicitly temporary internal family during migration.
- [ ] Define the exact canonical raw detection schema assembled from worker
  artifacts, including multi-camera/arena lineage where applicable.
- [ ] Freeze one versioned worker-artifact schema and one versioned
  artifact-to-canonical binding schema.
- [ ] Record the shared default inner chunks and requested outer shards.
- [ ] Decide and document one canonical decode backend, or require explicit
  backend parity/provenance if layouts genuinely need different backends.

### Phase 1: reproduce and repair the clipped artifact seam

- [ ] Add an integration test that builds a real minimal artifact manifest,
  resolves its intended import target, and compares it with the worker's
  validation target.
- [ ] Make the test reproduce the current `detection_artifact_runs` versus
  `detect_runs` mismatch before changing behavior.
- [ ] Remove independent string construction of the target family/path from
  the planner, artifact builder, importer, and validator.
- [ ] Introduce one typed artifact-target record consumed by all four.
- [ ] Ensure reuse validation resolves the same recorded target rather than
  accepting a separately supplied path.
- [ ] Fail closed if an unbound artifact is passed directly to quality,
  refinement, a selector, the registry, or a generic reader.

### Phase 2: implement recording-level artifact binding and assembly

- [ ] Freeze the complete expected work-unit manifest before execution.
- [ ] Require one successful, checksummed artifact for every planned work
  unit; reject missing, duplicate, unexpected, or mixed-plan artifacts.
- [ ] Revalidate model digest, algorithm version, precision, parameters,
  source-media identity, native dimensions, and coordinate declarations across
  all members.
- [ ] For clipped sources, verify the exact checksummed recording-frame map and
  map every local detection row to a canonical parent frame.
- [ ] For whole-video sources, verify the one-member identity frame mapping.
- [ ] Mint deterministic recording-scoped `instance_key` values only after
  canonical recording identity and frame authority are bound.
- [ ] Define deterministic intra-frame ordering and reject duplicate keys.
- [ ] Preserve `artifact_row_id`, source work-unit ID, clip ID/index,
  camera/arena identity, and local frame as lineage rather than authority.
- [ ] Stream assembly by complete output shard; do not load the entire
  recording table solely to publish it.
- [ ] Write one immutable, indexed-sharded `detect_runs/<run>` candidate.
- [ ] Record requested and effective physical chunk/shard ownership.
- [ ] Validate decoded array digests, row counts, frame coverage, key
  uniqueness, model provenance, and coordinate contracts.
- [ ] Publish atomically and activate `latest`/`latest_complete` only after a
  fresh final-location validation.
- [ ] Emit one recording-level completion receipt and registry event.

### Phase 3: converge quality, geometry gating, and refinement

- [ ] Make the canonical `detect_runs/<run>` snapshot the common quality input.
- [ ] Preserve parallel shard-local quality work and the recording-ordered
  reconciler for stateful jump/blip logic.
- [ ] Prove one-member whole-video output is equivalent to the same ordered
  algorithm without artificial boundaries.
- [ ] Make selected arena geometry an explicit reusable DAG capability.
- [ ] Materialize a keyed `detection_gate_runs/<run>` against the exact raw
  snapshot and selected geometry record.
- [ ] Require refinement to consume an explicitly planned gate artifact when
  the policy is enabled or required.
- [ ] Require exact ordered `instance_key` equality among raw detections,
  quality labels, gate decisions, and refinement input.
- [ ] Publish one immutable recording-level `refined_detect_runs/<run>` using
  the frozen refined-detection storage contract.
- [ ] Preserve clip lineage in the refined snapshot without making consumers
  resolve per-clip runs.

### Phase 4: converge downstream fragment inputs

- [ ] Define one typed refined-detection capability consumed by crop/cache.
- [ ] Keep clip-local pixel decoding and cache construction behind the clipped
  source adapter.
- [ ] Publish one recording-level crop geometry authority with exact key
  equality to the refined snapshot.
- [ ] Make whole and clipped keypoint fragments consume the same crop/cache
  capability.
- [ ] Keep per-worker keypoint shards internal and publish the same top-level
  keypoint families.
- [ ] Make whole and clipped subject-mask fragments consume the same crop and
  refined-keypoint capabilities.
- [ ] Represent retained raw mask-probability collections through one typed
  manifest rather than layout branches in downstream code.
- [ ] Require final refined subject masks to expose the same dense authority
  and exact keyset in both layouts.

### Phase 5: unify validation

- [ ] Extract recording-layout validation from scientific cross-stage
  validation.
- [ ] Share one exact-key validation implementation across both layouts.
- [ ] Validate exact key equality—not only row-count equality—across every
  linked stage.
- [ ] Validate canonical frame coverage, acquisition-time domain, camera/arena
  identity, coordinate authority, and native dimensions.
- [ ] Validate source-run digests and immutable manifest bindings.
- [ ] Fail closed on one-sided key loss; permit positional compatibility only
  through an explicitly labeled historical mode.
- [ ] Sample or stream large mask-pixel validation without weakening complete
  metadata and identity validation.
- [ ] Require the common validator before selector promotion and registry
  refresh.

### Phase 6: converge registry, readers, review, and export

- [ ] Project both layouts as the same logical detection, quality, refinement,
  crop, keypoint, and mask stages.
- [ ] Record source layout and clip count as provenance/details, not as a
  different stage status.
- [ ] Update generic detection resolution to prefer canonical top-level
  snapshots.
- [ ] Preserve `latest_collection` resolution as a read-only historical
  compatibility path.
- [ ] Route Crimson, review tooling, training exporters, and analytics through
  the generic recording-level resolver.
- [ ] Verify manual edits resolve by `instance_key` and retain clip lineage for
  source-pixel recovery.
- [ ] Verify delta compaction produces a new immutable recording-level
  snapshot and atomically advances the same selectors for both layouts.

### Phase 7: parity canaries

- [ ] Construct a fixture where one source video is represented both as one
  file and as multiple lossless logical clips with a proven frame map.
- [ ] Run identical model, precision, decode, resize, and threshold settings.
- [ ] Compare canonical frame rows, box coordinates, scores, classes, and
  `instance_key` values after assembly.
- [ ] Require bit-for-bit equality where the decode path permits it; otherwise
  define and justify numerical tolerances before accepting parity.
- [ ] Compare quality labels across artificial clip boundaries.
- [ ] Compare gate decisions and refined detections.
- [ ] Compare crop geometry, keypoints, refined keypoints, and refined mask
  keysets.
- [ ] Run one real whole-video canary and one real clipped canary through the
  same named recipe.
- [ ] Confirm generic readers and the registry do not need layout-specific
  selection logic for the new outputs.

### Phase 8: compatibility migration and subtraction

- [ ] Inventory active archives whose only refined authority is
  `latest_collection`.
- [ ] Publish recording-level refined snapshots from validated collections
  without rerunning scientific computation.
- [ ] Preserve collection manifests and per-clip evidence for audit history.
- [ ] Stop writing new `latest_collection` authorities once the canonical
  snapshot finalizer is proven.
- [ ] Remove duplicated clipped/whole command rendering after fragment parity
  tests pass.
- [ ] Remove obsolete artifact import paths and shell wrappers after retained
  evidence/retry behavior has a supported replacement.
- [ ] Update or archive documents that describe clipped recordings as lacking
  a canonical recording clock or requiring permanent consumer branching.
- [ ] Consolidate registry and validator compatibility branches once the
  historical migration inventory reaches the declared retirement threshold.

## Test matrix

At minimum, implementation must cover:

1. One whole-video work unit with an identity frame map.
2. Multiple clips with complete, ordered, non-overlapping coverage.
3. A gap in clip coverage.
4. Overlapping parent-frame mappings.
5. Duplicate or unexpected worker artifacts.
6. Mixed model digest, algorithm version, precision, or decode backend.
7. Native-dimension or coordinate-contract disagreement.
8. Deterministic instance-key generation across physical layouts.
9. Duplicate payloads within one frame and deterministic ordinal handling.
10. Temporal quality state crossing an artificial clip boundary.
11. Exact gate/refinement key alignment.
12. Atomic publication failure before activation.
13. Source mutation between initial validation and final activation.
14. Reader and registry selection of the canonical snapshot.
15. Historical `latest_collection` compatibility without promotion as a new
    production authority.
16. Delta edit routing back to owning clip evidence followed by compaction into
    a new recording-level snapshot.

## Acceptance criteria

Convergence is complete when:

- the same named workflow recipes accept either recording layout;
- the production DAG publishes science-ready detection arrays only through
  `detect_runs` and `refined_detect_runs`;
- clipped worker artifacts cannot be selected or consumed as canonical data;
- both layouts use the same canonical frame, coordinate, identity, quality,
  gate, refinement, and storage contracts;
- all cross-stage joins are validated by exact `instance_key`;
- downstream stage builders and generic readers do not branch on clipped
  versus whole-video layout;
- registry stage meanings are identical, with layout retained only as
  provenance;
- pointer promotion occurs only after atomic publication and final-location
  validation;
- an artificial split/unsplit parity canary passes; and
- historical collection-based recordings remain readable while new
  production no longer depends on `latest_collection`.

## Explicit non-goals

This work does not require:

- physically concatenating source clip videos;
- eliminating clip-local decode or cache parallelism;
- discarding clip lineage or historical collection manifests;
- forcing large raw mask probabilities into one physical object;
- making scheduler arrays part of scientific identity;
- rerunning valid historical inference merely to migrate its canonical
  representation; or
- replacing independently runnable detection-only, keypoint-only, or
  subject-mask-only stage commands with one mandatory full DAG.
