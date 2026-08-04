# Eye/Tail Query Export Source Prerequisite

Date: 2026-08-04

Status: fail-closed upstream prerequisite recorded; no production archive,
selector, registry, physical profile, or query export was changed.

## Result

The full-duration eye/tail query-export lane is not currently blocked by its
Parquet publisher. It is blocked earlier, at the scientific authority needed
to produce current subject-shape, eye-angle, and tail-kinematics runs.

The maintained full-duration subject-mask benchmark archive inspected was:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
subject_mask_storage/full_duration/
sleepyfish_cam2010095_20260731_73f7bb5e/analysis.zarr
```

It contains the full 1,169,010-row dense refined subject-mask surface and the
four maintained component labels, but its recording-level shard-finalized run
does not contain the canonical coordinate-publication owner/proof required by
the subject-shape materializer. A real read-only materialization preflight
therefore stopped with:

```text
RefinedSubjectMaskCoordinatePublicationError:
Canonical refined subject-mask run lacks one unguessable publication owner.
```

That error is the intended fail-closed behavior in
`shared/refined_subject_mask_coordinate_publication.py`. Subject-shape loading
routes through `load_persisted_refined_subject_mask_coordinate_surfaces`, and
current subject-shape v4 is a coordinate-authority transition rather than a
metadata-only version bump.

## Root Cause

The recording-level archive was assembled from a shard collection. Current
subject-mask finalization computes:

```text
future_canonical = source.canonical_coordinates and shard_collection is None
```

Consequently the collection finalization path publishes modern storage/core
manifests but does not aggregate and republish canonical coordinate ownership.
The per-clip receipts preserve model, crop, keypoint, preprocessing, and
scientific identities, but those proofs are not promoted into one recording-
level coordinate publication.

The maintained downstream versions are:

- subject shape v4;
- eye angle compact v7; and
- tail kinematics v2.

The historical selected eye/tail runs are v5/v1. Restamping those runs or the
subject-shape v3 surface would falsely claim a coordinate transformation and
is prohibited.

## Required Repair

Before the eye/tail query-export benchmarks can be valid:

1. Recording-level subject-mask shard finalization must aggregate and verify
   every clip's canonical coordinate and identity proof.
2. It must publish one immutable recording-level refined-mask coordinate owner
   with complete coverage and no conflicting clip authority.
3. Subject-shape v4 must be scientifically materialized from that authority.
4. Eye-angle v7 and tail-kinematics v2 can then be materialized from their
   exact current dependencies.
5. Only those explicit selector-ineligible sources may feed the eye/tail
   query-export benchmark.

No validator should be relaxed to make this benchmark run. The missing proof
is an upstream interoperability defect and remains separate from physical
layout optimization.
