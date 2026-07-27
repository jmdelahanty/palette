# Refined Detection To Crop Handoff Contract v1

Status: source binding and keyed planning implemented; crop publication is not
authorized

Date: 2026-07-27

## Purpose

This checkpoint proves that an exact refined-detection v1 snapshot can become a
stable input to incremental crop planning without weakening detection selection,
identity, geometry, or publication rules. It does not route a production crop
writer and does not change a selector or registry.

The executable boundaries are:

- `fisheye.shared.zarr.refined_detection_crop_source` for exact source selection
  and validation; and
- `fisheye.tracking.refined_detection_crop_handoff` for keyed crop copy/compute
  planning.

## Source Requirements

The binder accepts only a complete full-acquisition refined-detection v1 run.
Production selection requires either an explicit selector-eligible run or the
exact approved `authoritative_run` envelope. A selector-ineligible benchmark run
requires both an explicit run ID and the separate benchmark-only option; it is
never selected implicitly.

Before returning a source, the binder verifies:

- the exact refined-v1 run manifest and logical dimensions;
- all required decoded arrays and logical-content digest;
- snapshot identity and any bound parent evidence;
- the promoted storage plan and codec declarations;
- direct and consolidated metadata equivalence; and
- recording identity when the archive publishes one.

Invalid explicit refined input is terminal. There is no raw-detection fallback
inside this handoff.

## Identity And Geometry

`instance_key` remains the durable observation/edit identity. The exact
`instances/refined_row_ids` values are copied into the crop lineage field
`source_refined_row_ids`; a missing, reordered, or conflicting mapping fails
before planning.

The handoff remains entirely in the frozen image-space detection geometry:

- `bbox_norm_coords` is authoritative;
- `bbox_img_xyxy` and `centers_img_xy` are required validated derivatives;
- `source_acquisition_frame_index` preserves camera-frame identity; and
- no homography, physical-coordinate transform, or arena geometry is introduced.

Zero, one, or many instances per frame are supported because crop rows follow
the refined instance table, not a one-row-per-frame assumption.

## Incremental Planning

The crop source signature binds each `instance_key` to its frame, authoritative
normalized box, exact source-pixel fingerprint, source frame dimensions, and ROI
settings. Given a complete prior materialized crop run:

- matching key and signature rows are copied;
- new keys are computed;
- changed geometry or incompatible signature context is recomputed; and
- deleted source keys do not appear in the successor plan.

Reuse requires a complete predecessor with the exact persisted signature
specification. Logical row disjointness alone is not sufficient evidence.

## Intentional Publication Blocker

Every handoff receipt records:

```text
coordinate_status = image_space_values_validated_refined_lineage_publication_pending
crop_publication_authorized = false
```

The existing crop coordinate-publication path still carries raw-detection-
specific source authority assumptions. Production crop publication must remain
blocked until its coordinate lineage can bind the exact refined-v1 manifest,
logical-content digest, source arrays, and approved selection evidence without
relabeling refined rows as raw detections.

## Completion Checklist

- [x] Exact refined-v1 source binding.
- [x] Explicit-only selector-ineligible benchmark boundary.
- [x] Approved refined authority resolution with no `latest` fallback.
- [x] Full publication, content-digest, and metadata-equivalence validation.
- [x] Multi-instance and empty-frame source coverage.
- [x] Exact `refined_row_ids` to `source_refined_row_ids` propagation.
- [x] New-instance and changed-geometry selective recomputation tests.
- [x] Explicit non-authorization of crop publication.
- [ ] Freeze the refined-v1 crop coordinate-lineage envelope.
- [ ] Route the production crop writer through the exact binder.
- [ ] Publish and validate an immutable selector-ineligible crop canary.
- [ ] Add rowset-change completeness gates for keypoints, masks, tracking, and
      training successors before any refined selector activation.
