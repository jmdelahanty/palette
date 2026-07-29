# Shared Coordinate Storage Contract v1

Status: schema vocabulary implemented; versioned persisted run envelopes and
selector-ineligible publication paths implemented; consumer adoption pending

Date: 2026-07-28

## Decision

Palette keeps the numeric representation natural to each product and unifies
its meaning instead of forcing every array into one representation:

- detection authority: source-camera normalized `float32 [cx,cy,w,h]`;
- derived detection boxes: source-camera half-open continuous pixel edges;
- keypoint authority: ROI-local continuous pixels, with exact derived
  source-camera pixel and normalized surfaces;
- crop extraction origin: exact integer source-camera indices;
- crop extraction size: exact integer pixel extents;
- crop-local boxes, masks, and contours: explicitly ROI-local; and
- presentation: transform every positional surface to source-camera continuous
  pixels before viewport, renderer, or hit-test transforms.

`source_camera_image_px` is the shared presentation domain, not an integer
array-index type. Integer extraction indices and extents remain distinct typed
storage measurements. Viewport and renderer coordinates remain ephemeral and
must not be persisted as scientific authority.

## Two Contract Layers

The schema-level template in
`fisheye.shared.coordinate_surface_contract` freezes:

- canonical domain/profile;
- geometry type and component order;
- component units;
- pixel convention;
- reference-extent role; and
- the required mapping to source-camera pixels.

The existing canonical coordinate descriptor remains the persisted authority.
It additionally binds the concrete frame extent, row identity, lineage,
directed transform, and digests for one actual array. A schema template is not
permission to infer or fabricate any of that run-specific evidence.

`fisheye.shared.zarr.coordinate_contracts` binds exact versioned
`ArrayContract` IDs to the templates. Existing short `coordinate_space` strings
remain only compatibility annotations; the catalog rejects a coordinate-bearing
array contract that lacks an exact binding.

## Source-Camera Mapping

| Stored surface | Mapping used by consumers |
| --- | --- |
| Source-camera continuous pixels | Direct |
| Source-camera normalized coordinates | Scale by the exact source-camera width and height |
| ROI-local continuous pixels/edges | Apply the exact rowwise ROI-to-source-camera directed transform |
| Integer crop origin/extent | Extraction policy only; do not reinterpret the extent as a point |

The mapping does not choose crop rounding or padding. Those remain part of the
versioned crop policy. It also does not clip out-of-frame geometry silently.

## Current Coverage

The catalog covers every coordinate-bearing contract currently in
`CORE_ARRAY_CONTRACTS`:

- raw and refined detection normalized boxes, pixel boxes, and centers;
- crop origin, extent, source window, and ROI-local detection box;
- keypoints in ROI, source-camera pixels, and source-camera normalized space;
- dense ROI masks; and
- flat ROI contour points.

Detection, refined-detection, and crop schemas expose a deterministic
`coordinate_contract_manifest()` accessor. Runtime detection, crop, and
keypoint coordinate publication uses these shared templates to construct its
canonical descriptors rather than repeating literal profile and component
definitions.

The catalogs are now available in new, opt-in persisted run-manifest versions:

| Stage | Existing manifest | Coordinate-catalog manifest |
| --- | --- | --- |
| canonical detection | v1 legacy conversion; v2 native producer | v3 with explicit `source_evidence_kind` |
| refined detection | v1 | v2 |
| geometry-only crop | v1 | v2 |

The old builders and publisher defaults still emit their original versions.
The new versions add exactly one `payload.coordinate_contract` envelope. The
canonical v3 payload also adds `source_evidence_kind` so source validation is
not inferred from document shape. The envelope binds the complete catalog with
`sha256_canonical_json_v1`, and the containing run manifest binds that envelope
again through its payload digest. Validators compare the complete document to
the frozen stage builder; recomputing both digests after a semantic edit does
not make a changed catalog valid.

Selector-ineligible canonical/refined snapshot publication and geometry-only
crop shadow publication expose `coordinate_catalog=True`. The paired detection
snapshot CLI exposes `--coordinate-catalog`. Refined-detection compaction
preserves the immediate parent's manifest generation automatically, preventing
a v2 snapshot from being silently downgraded to v1.

## Dtype Boundary

Coordinate meaning and dtype are related but separate contracts. This work does
not change any dtype:

- detection and crop geometry remain `float32`/integer as already frozen;
- current keypoint contracts remain `float64`; and
- future keypoint dtype changes require their own versioned numerical and
  consumer review.

Crimson can therefore open exact compile-time dtypes from the logical array
contract, read the coordinate template from a future consolidated run
manifest, and apply the bound transform without dtype probing.

## Implementation Checklist

- [x] Reuse canonical coordinate profiles rather than create competing space
      enums.
- [x] Distinguish continuous pixels, half-open pixel edges, integer extraction
      indices, and pixel extents.
- [x] Bind every current coordinate-bearing `ArrayContract` to one exact
      surface template.
- [x] Make missing bindings and legacy-label drift fail closed.
- [x] Replace repeated detection, crop, and keypoint descriptor literals with
      shared templates.
- [x] Keep existing array IDs, schema versions, dtypes, shapes, and persisted
      v1 manifest bytes unchanged.
- [x] Add the coordinate catalog to new versions of canonical-detection,
      refined-detection, and crop manifests without rewriting old envelopes.
- [x] Add selector-ineligible real-Zarr publication paths and preserve the
      coordinate-aware manifest version through refined compaction.
- [ ] Have Crimson consume and validate the catalog before removing any
      remaining coordinate-name inference.
- [ ] Extend the catalog with exact pose-bbox `ArrayContract` entries when the
      future keypoint logical schema is frozen.
- [ ] Require the same catalog in subject-mask, contour, shape, eye, tail, and
      training-export storage manifests as those schemas migrate to the shared
      storage module.
- [ ] Add cross-language fixture tests proving Palette and Crimson agree on
      domain, components, units, conventions, and transform direction.

## Exit Gate For Persisted Adoption

A stage may persist this catalog only in a new manifest schema version. Before
activation, Palette must validate that every coordinate array's live canonical
descriptor matches its catalog template, consolidated and direct metadata are
equivalent, and Crimson opens the exact declarations without heuristic path,
space, or dtype discovery.

Palette's current selector-ineligible writers meet the manifest, digest,
direct/consolidated metadata, and exact-schema portions of this gate. Production
selection remains a separate reviewed action.
