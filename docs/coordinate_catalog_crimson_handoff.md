# Coordinate Catalog Crimson Handoff

Status: Palette producer checkpoint complete; Crimson validation pending

Date: 2026-07-28

## Purpose

Palette now has one shared semantic vocabulary for coordinate-bearing arrays
without forcing detections, crops, keypoints, masks, and contours into one
numeric representation. This handoff covers the first persisted adoption in
canonical detections, refined detections, and geometry-only crops.

This changes coordinate interpretation only. Array paths, logical schema IDs,
dtypes, shapes, chunk/shard plans, codecs, and selection policy are unchanged.
The new manifests are opt-in and selector-ineligible.

## Persisted Contract

The catalog is stored at:

```text
<run>/zarr.json.attributes.run_manifest.payload.coordinate_contract
```

Its exact envelope is:

```text
schema_id:        palette.persisted_coordinate_catalog
schema_version:   1
digest_algorithm: sha256_canonical_json_v1
digest:           SHA-256 of canonical strict JSON(document)
document:         palette.array_coordinate_catalog v1
```

The enclosing run-manifest payload digest also covers this envelope. Palette
requires the exact envelope fields, strict finite JSON, a correct digest, and
complete equality with the frozen catalog for that stage.

| Run family | Coordinate-aware run-manifest version | Frozen catalog digest | Bindings / surfaces |
| --- | ---: | --- | ---: |
| `detect_runs/<run>` | 3 | `337613bd6e5f283eef9d6a89c14766d50c5b6863dea584f7568b90bb1d936733` | 3 / 3 |
| `refined_detect_runs/<run>` | 2 | `75656615ecd32a215f6b4148a01c9ef75e96b8d7aa6bf9fb8a7d21757fa7a2ed` | 6 / 3 |
| `crop_runs/<run>` | 2 | `e9ce640761ee1de4a6edd72695968bd66ae2fcbdd09d7d2c902450904f6ddfec` | 7 / 7 |

Canonical-detection v3 additionally requires
`payload.source_evidence_kind`, exactly `legacy_conversion` or
`native_detection`. This removes the v1/v2 source-shape inference from the new
consumer boundary.

## Consumer Resolution

For each persisted array, Crimson should:

1. Validate the run-manifest version and outer payload digest.
2. Validate the exact coordinate-catalog envelope and digest.
3. Resolve the array path through the run manifest's frozen logical-schema
   binding to an exact `(array_contract_id, array_contract_version)`.
4. Resolve that key through `document.bindings` to exactly one `surface_id` and
   semantic role.
5. Resolve the surface through `document.surfaces` and enforce domain,
   geometry type, component order, units, pixel convention, reference extent,
   and source-camera mapping.
6. Open the array using the dtype and shape from the logical array contract;
   do not infer dtype or coordinates from the path name or value range.

The catalog is intentionally not a run-specific transform. ROI-local arrays
still require the exact rowwise crop placement/coordinate descriptor bound by
the run. Normalized detection boxes still require the exact source-camera width
and height. Crimson's common presentation space remains continuous
source-camera pixels.

## Fail-Closed Cases

Crimson should reject a coordinate-aware run when any of the following occurs:

- the coordinate envelope is missing, has extra fields, or uses another schema
  or digest algorithm;
- either digest is wrong;
- a recomputed digest accompanies a catalog that differs from the stage's
  accepted catalog;
- a coordinate-bearing array contract has zero or multiple bindings;
- a referenced surface is absent or duplicated;
- the catalog disagrees with the logical array contract or live run-specific
  coordinate descriptor; or
- an explicit coordinate-aware run is invalid. It must not silently fall back
  to a legacy/raw run.

Existing canonical-detection v1/v2, refined-detection v1, and crop v1 remain
explicit compatibility versions. Their absence of `coordinate_contract` is
not permission to treat them as the new contract.

## Palette Evidence And Remaining Gate

Palette tests cover:

- exact envelope construction and strict JSON;
- tampering with both inner and outer digests recomputed;
- unchanged old-version builders and defaults;
- real Zarr write, consolidation, reopen, and publication validation for all
  three run families;
- direct/consolidated declaration equivalence; and
- preservation of refined-detection v2 through delta compaction.

Before activation, Crimson should add cross-language fixtures for all three
catalogs, negative tampering cases, exact typed opens, and transformations into
source-camera pixels. Palette should then publish one selector-ineligible
coordinate-aware canary for Crimson. Production selectors and registries must
remain unchanged until that gate passes.
