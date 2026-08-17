# Subject Position Storage Contract v1

<!-- contract-meta
version: 1
status: draft
implementation: partial
last_updated: 2026-08-16
-->

Purpose: freeze the logical arrays, exact dtypes, coordinate authority,
invalid-value representation, and publication envelope for materialized
subject-position providers. Palette now implements the pure evaluator and
logical in-memory validator; immutable Zarr publication remains a later phase.

The Phase 1 logical-metadata helper is not a complete run manifest and cannot
authorize publication. A pure evaluator result does not itself prove source
coordinates or row lineage; Phase 2 must bind those authorities from the exact
persisted source before constructing the complete immutable manifest below.

This contract does not select a biological position estimator. Estimator and
provider-selection policy are defined separately. It describes how the result
of any accepted estimator is stored and validated.

## Scope and Namespace

Version 1 implements observation-row positions at:

```text
analysis/subject_position_runs/observation/<run>/
```

Its row axis is exactly `observation_instance`. Every row is aligned to one
ordered `instance_key` from one immutable source snapshot.

The family reserves a separate future namespace for identity-resolved temporal
products:

```text
analysis/subject_position_runs/track_sample/<run>/
```

A run belongs to exactly one row axis. Observation rows cannot be relabeled as
track samples, and equal row counts do not prove compatibility.

## Observation-Run Logical Schema

The following arrays are mandatory:

| Path | Exact logical dtype and shape | Meaning |
| --- | --- | --- |
| `position_xy` | `float32[N,2]` | Materialized continuous source-camera `(x,y)` position |
| `valid` | `bool[N]` | Exact estimator validity for each row |
| `failure_reason_codes` | `uint16[N]` | Controlled row-result code; zero means valid |
| `instance_key` | `uint64[N]` | Exact observation/edit-lineage identity |
| `source_acquisition_frame_index` | `int64[N]` | Exact recording acquisition-frame identity |
| `source_row_index` | `int64[N]` | Exact row offset in the bound immutable source snapshot |

All six arrays have identical leading dimension `N`. `position_xy` has exact
trailing axis order `(x, y)`. Writers and readers must reject `float16`,
`float64`, integer coordinates, a transposed `(2,N)` layout, or an untyped
numeric array accepted merely because its values look plausible.

`instance_key` values are unique within an observation run. The ordered key
payload, source-frame payload, and source-row payload are independently
digest-bound by the run manifest. `source_row_index` is an exact source
location, not a replacement identity.

## Optional Support Evidence

An estimator may materialize these diagnostic arrays:

| Path | Exact logical dtype and shape | Meaning |
| --- | --- | --- |
| `support/source_points_xy` | `float32[N,P,2]` | Ordered contributing source points after coordinate projection |
| `support/source_points_valid` | `bool[N,P]` | Per-anchor source-authority validity |
| `support/source_point_reason_codes` | `uint16[N,P]` | Optional controlled per-anchor reason code |
| `support/source_point_confidence` | `float32[N,P]` | Optional exact source confidence when the source contract defines it |

`P` and its ordered anatomy-role IDs are fixed by the estimator profile and
bound in the manifest. Support arrays are evidence, not an alternate position
authority. Consumers must not recompute a different mean from them while
claiming the materialized estimator identity.

If support coordinates are omitted, the manifest must still bind the exact
source coordinate and validity arrays required to reproduce the result. A
confidence array is absent when the source has no compatible quantitative
confidence; it is never synthesized.

## Coordinate Authority

Version 1 observation positions use only:

```text
profile_id: source_camera_image_px.top_left_y_down.v1
geometry_type: point_xy
units: px
pixel_convention: continuous
axis_order: [x, y]
origin: top_left
x_direction: right
y_direction: down
```

`position_xy` owns a canonical, digest-bound coordinate descriptor and exact
source-camera frame/extent authority. Direct overlay is allowed only when that
descriptor declares the exact source frame and
`source_camera_overlay.status == "direct"`.

ROI-local coordinates, normalized coordinates, presentation canvas positions,
reflected Citrus display coordinates, and nominal camera spaces are not
accepted as v1 observation-position authority. They require an exact
direction-labelled transform into the accepted source-camera profile before
evaluation or a future separately versioned position contract.

Physical millimetre positions are not inferred from a root scalar, raster
ratio, or array name. A physical position requires a separately authorized
typed transform and derived surface. Its absence means physical position is
unavailable.

## Computation and Float32 Publication

Estimators evaluate source geometry in `float64`, including component centroid
sums, midpoint arithmetic, and equal-per-point means. The final two-coordinate
result is canonically cast once to `float32` for publication.

This follows modern keypoint-v2, body-frame, and track-position storage while
retaining substantially finer precision than the source imagery can resolve.
Near coordinate 4512, adjacent finite float32 values are approximately
0.00049 pixels apart.

The writer must:

1. recompute the result from the exact bound source arrays in float64;
2. reject finite overflow and any unsupported non-finite source state;
3. cast the final result to float32 using the repository's controlled NumPy
   conversion path;
4. canonicalize invalid output coordinates to the float32 quiet-NaN bit
   pattern `0x7fc00000`;
5. validate exact decoded float32 equality with the staged array; and
6. record the maximum absolute float64-to-float32 quantization error over valid
   rows.

The logical content digest is computed from canonical C-order decoded values
and the exact dtype/shape declaration. The physical plan separately records
Zarr version, endian codec, compression, chunks, shards, and fill values.

## Validity and Fill Semantics

For every row, exactly one of these states is legal:

```text
valid row:
  valid == true
  failure_reason_codes == 0
  position_xy[0] and position_xy[1] are finite

invalid row:
  valid == false
  failure_reason_codes != 0
  position_xy == [float32_qnan_0x7fc00000, float32_qnan_0x7fc00000]
```

One finite coordinate paired with one NaN is invalid storage. Infinity is
never accepted. Readers consult `valid` and `failure_reason_codes`; they do not
infer scientific validity from finiteness alone.

Mandatory identity and frame arrays have no nullable sentinel rows. Structural
source failures—including missing roles, unavailable required run-level
components, duplicate/missing/reordered identities, stale digests, and
coordinate incompatibility—prevent completed publication instead of creating
placeholder identities.

### Failure reason registry

`failure_reason_codes` uses exact `uint16`. The run manifest binds a controlled
code map, its schema ID/version, and its digest. Version 1 reserves:

| Code | Stable tag |
| ---: | --- |
| `0` | `ok` |
| `1` | `source_observation_rejected` |
| `2` | `required_anchor_invalid` |
| `3` | `required_anchor_low_confidence` |
| `4` | `empty_mask_component` |
| `5` | `nonfinite_source_geometry` |
| `6` | `degenerate_source_geometry` |

Unknown codes fail validation. A profile may use only applicable codes, but it
cannot redefine their meanings. When several anchors fail, support validity or
per-anchor reason arrays preserve all available evidence and the estimator
uses its digest-bound deterministic primary-reason precedence.

An unexpected evaluator exception, unsupported operation, or unclassifiable
state aborts the run. It is not converted into a generic row-level failure
code.

Fixed-width reason strings are not authoritative v1 storage. Human-readable
tools decode the controlled map. This avoids multiplying tens of bytes of
repeated text across every observation.

## Estimator and Source Binding

The immutable run manifest binds at least:

- storage schema ID/version and canonical schema digest;
- estimator ID/version, canonical payload, and digest;
- provider-selection and validity-policy IDs/versions/digests;
- anatomy profile and exact source-schema role bindings when applicable;
- exact source run, arrays, row identities, and content digests;
- coordinate descriptor and source-camera frame authority;
- ordered source acquisition frames and source row indices;
- reason-code registry and deterministic precedence;
- logical array descriptors and content digests;
- physical chunk/shard/codec/fill plan;
- software/configuration identity; and
- completion, selector eligibility, and publication-manifest evidence.

Source schema compatibility is established before row evaluation. A provider
must not use matching string labels, equal lengths, or `latest` pointers as a
substitute for exact source identity and role binding.

## Physical Storage and Publication

Physical chunk and shard sizes are selected through a versioned byte-planning
profile and recorded rather than frozen as universal row counts. Every
physical chunk contains whole row records; the two-value XY axis and the
two-column future track key are never split across workers or physical chunks.
Parallel writers own whole, non-overlapping chunks for every array they write.

Runs stage outside the visible final namespace, validate logical content and
the physical plan, publish through Palette's shared atomic run publisher, and
remain selector-ineligible until their scientific promotion policy passes.
Consolidated root metadata is refreshed only as the final visibility step.
Direct and consolidated readers must resolve the same completed manifest and
selector state.

## Future Track-Sample Core

A future track-sample position publication retains the same scientific value
types:

| Path | Exact logical dtype and shape |
| --- | --- |
| `position_xy` | `float32[N,2]` |
| `valid` | `bool[N]` |
| `failure_reason_codes` | `uint16[N]` |
| `track_sample_key` | `int64[N,2]` |
| `source_acquisition_frame_index` | `int64[N]` |

Its full lineage schema additionally binds the exact observation-to-track
projection or temporal transform. Version 1 observation implementation does
not publish this future family. A track-sample publisher requires its own
completed logical-schema review before activation.

## Reader Requirements

A normal reader must validate:

1. the exact family and row-axis namespace;
2. completed immutable publication and selector eligibility required by the
   caller;
3. exact array names, dtypes, ranks, shapes, and leading-dimension equality;
4. unique ordered identity and exact source-frame/source-row records;
5. coordinate descriptor and extent/frame bindings;
6. validity, canonical NaN, and reason-code consistency;
7. estimator, source, anatomy, policy, and manifest digests; and
8. direct/consolidated publication-generation consistency.

A reader must not cast an unsupported dtype, manufacture missing validity,
derive physical units, choose another provider, or silently traverse
unconsolidated metadata for a published immutable run.

## Acceptance Criteria

- Float32 quantization is measured against float64 evaluator output on a
  representative canary and remains within the reviewed precision budget.
- Valid and invalid rows satisfy the exact finite/NaN/code invariants.
- Reordered, missing, duplicated, and stale source identities fail closed.
- Unsupported dtype, shape, coordinate, reason-code, and role-binding variants
  fail closed.
- Equal-component means and pixel-union centroids remain distinct estimator
  products under the same storage schema.
- Empty-detection recordings publish valid zero-length arrays with the exact
  declared shapes and complete empty-row evidence.
- Logical content and direct/consolidated metadata validate after atomic
  publication.

## Related Documents

- [Position, Body-Frame, and Motion Provider Design](position_body_frame_and_motion_provider_design.md)
- [Continuous Points and Half-Open Bounding Boxes](continuous_points_and_half_open_boxes.md)
- [Body Frame Contract](body_frame_contract.md)
- [Keypoint Storage Contract v2](keypoint_storage_contract_v2.md)
- [Derived Analysis Run Contract](derived_analysis_run_contract.md)
- [Dask Zarr Write Safety](dask_zarr_write_safety.md)
