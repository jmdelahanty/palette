# Subject-Mask Provenance Contract Hardening

Date: 2026-08-05

Status: implemented for selector-ineligible recording publication. No
production selector, registry authority, canonical archive, or running compute
job was changed.

## Decision

Subject-mask publication must prove its scientific and row-domain inputs, not
merely retain opaque digests produced by workers. A digest proves that one JSON
document did not change; it does not prove that the document has the required
fields or that its claims agree with the crop and upstream mask authorities.

The current fail-closed version chain is:

- scientific identity v2;
- recording assembly identity v3;
- coordinate-dependency envelope v3;
- coordinate-aware subject-mask core manifest v4.

Scientific identity v1, recording assembly v2, and coordinate core v3 remain
explicit legacy/selector-ineligible evidence. They are not silently promoted
or interpreted as the stronger contracts.

## Enforced Evidence

### Raw inference workers

Scientific identity v2 requires exact closed-world documents for:

- model artifact role, SHA-256, byte size, registry identities, and label
  schema;
- crop run/path/manifest, ROI shape and placement digest, collection and work
  unit identity, and complete-partition proof;
- decoded pixel shape, dtype, order, digest, pixel contract, and materialized
  cache identity;
- row count and exact hashes for crop-row, instance-key, and acquisition-frame
  arrays;
- model-input transform, probability semantics/dtype/encoding/threshold, and
  overlap policy.

The complete-partition proof is deeply validated: its collection, recording,
camera, clip, video, frame window, global crop-row interval, and validation
claims must all be exact.

### Recording assembly

Recording assembly v3 requires the authoritative ordered work-unit table,
including windows containing zero crop rows. The table must cover `[0,F)` and
`[0,R)` without gaps or overlap. Workers exist only for nonempty units and must
match those units in canonical order.

At coordinate publication, each worker interval is joined back to crop v2:

- the interval equals the crop `frame_row_offsets` slice for its frame window;
- acquisition frames fall within that window;
- `source_crop_row_ids`, `instance_key`, and
  `source_acquisition_frame_index` hashes equal the live crop slice;
- refined workers also match `source_crop_xywh`;
- raw ROI origins, manifest reference, run path, and fixed ROI extent equal the
  crop authority.

### Refined lineage

Every refined worker binds the matching raw worker interval, run path,
scientific-identity digest, semantic-receipt payload digest, and receipt
document digest. The recording-level refined assembly binds the complete raw
recording assembly. The refined coordinate core must bind the same raw producer
named by the raw core manifest.

This prevents a recomputed outer JSON digest from disguising a changed crop row
hash, raw receipt, work window, or refined-to-raw edge.

## Publication Boundary

`publish_recording_subject_mask_bundle` requires an authoritative work-unit
manifest whenever crop v2 is present. The full-duration canary derives this
table from every planned window before dropping zero-row windows from compute.
The publisher then:

1. validates raw worker evidence and creates recording assembly v3;
2. creates coordinate-aware raw core v4;
3. validates refined workers against the exact raw workers and creates refined
   recording assembly v3;
4. creates coordinate-aware refined core v4 bound to the raw core;
5. publishes quality/cache members and the inactive atomic bundle through the
   existing publication path.

Activation remains a separate explicit operation. Existing v3 coordinate cores
must be republished from verifiable worker/crop inputs; changing an attribute is
not a migration.

## Validation Checklist

- [x] Reject empty or structurally incomplete scientific identity v2 documents.
- [x] Reject nested model/pixel/row tampering after recomputing the science
  digest.
- [x] Retain and validate zero-row work windows.
- [x] Reject missing, overlapping, reordered, or incomplete frame/row units.
- [x] Join worker row and ROI claims to crop-v2 arrays and frame offsets.
- [x] Join each refined worker to its exact raw worker receipt.
- [x] Join the refined recording/core publication to the same raw producer.
- [x] Preserve scientific identity v1 and assembly v2 only for explicit legacy
  publication paths.
- [x] Version the stronger coordinate publication as core v4.
- [ ] Run one selector-ineligible multi-clip canary containing an entirely
  empty work window and hand it to downstream consumers.
- [ ] Obtain Crimson compatibility evidence for coordinate core v4 before any
  production activation.
- [ ] Run the keypoint/keypoint-quality launch preflight only after crop geometry
  and pixel-cache receipts are terminal and complete.

## Validation Performed

Focused real-Zarr tests cover raw/refined core publication, atomic bundle
publication, inference worker creation, refinement proofing, clip import,
deeply re-digested tampering, and zero-row work-unit retention. Static Python
compilation, Ruff, Black, and `git diff --check` are required before the
checkpoint is committed.
