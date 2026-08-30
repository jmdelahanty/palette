# Deferred keypoint publication-validation optimization — 2026-08-29

Status: **deferred until the active subject-mask/subject-shape writer and
publication optimization is complete and accepted**.

The shared lifecycle decision governing this optimization is
[`publication_receipt_hashing_lifecycle_2026-08-29.md`](publication_receipt_hashing_lifecycle_2026-08-29.md).
Keypoints retain their own receipt profile while using that common
write-time-receipt and receipt-backed-publication protocol.

Audited Palette commit: `98b8223f92a747ac81c269f554c2def7760c71ec`.

## Finding

Canonical keypoint inference already uses the desired fused coordinate-write
shape. For each inference batch it:

1. converts detector-model coordinates to ROI-local coordinates;
2. derives source-camera and normalized point/bounding-box surfaces from the
   exact bound crop placement; and
3. writes the ROI, source-camera, and normalized arrays once with the same row
   identity.

It does **not** persist ROI-local coordinates and later mutate those arrays into
source-camera coordinates. The three coordinate surfaces are intentional
representations, not duplicated observation rows.

Relevant code:

- `src/fisheye/detection/detect_keypoints_yolo.py`, in the inference batch
  decode/derive/write path;
- `src/fisheye/shared/keypoint_coordinate_publication.py::derive_keypoint_coordinate_batch`;
- `src/fisheye/shared/keypoint_coordinate_publication.py::_validate_geometry`;
- `src/fisheye/shared/keypoint_coordinate_publication.py::publish_keypoint_coordinate_surfaces`.

## Deferred performance defect

The remaining amplification occurs during publication validation rather than
coordinate transformation:

- `_validate_geometry` decodes the complete persisted keypoint and pose-bbox
  arrays;
- it re-derives source-camera and normalized surfaces from the ROI arrays and
  checks exact equality; and
- `publish_keypoint_coordinate_surfaces` then starts a fresh root-level load
  that revalidates the records and arrays again.

This is strong fail-closed validation, but it repeats whole-payload reads after
the writer already derived the surfaces from a bound coordinate context.

## Intended later optimization

After the mask/subject-shape work is accepted, investigate a receipt-backed
keypoint publication lifecycle:

1. During batch writing, seal the exact row slice, source placement authority,
   derivation version, array metadata, and physical/content digests into a
   composable writer receipt.
2. At atomic publication, freshly validate the immutable receipt chain and
   Zarr metadata without decoding every coordinate array again.
3. Preserve one explicit deep-audit mode that rehashes and re-derives the full
   payload.
4. Keep the current exact numerical derivation and all authority gates; a
   receipt may replace redundant reads only when it proves the same claims.

Do not share a receipt schema with subject shape merely because both use
coordinate projection. Keypoints have ROI, source-camera, normalized, model
input, pose-schema, and crop-placement claims that require their own declared
receipt profile. Shared low-level digest and atomic-publication mechanisms are
appropriate.

## Acceptance criteria

- Existing and optimized writers produce byte-for-byte equal decoded keypoint
  and pose-bbox surfaces for representative finite, NaN, edge, and failure
  rows.
- A real writer -> publisher -> strict unpatched consumer boundary test passes.
- Tampering with any row identity, crop placement, coordinate array, transform,
  digest, or metadata generation fails closed.
- Counting-store telemetry demonstrates that normal publication no longer
  performs redundant full decoded scans; explicit deep audit still does.
- Canonical and compatibility finalizers cannot accidentally claim one
  another's receipt profile.

This item is intentionally not part of the current mask/subject-shape patch.
