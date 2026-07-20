# Subject Shape Runs Contract
<!-- contract-meta
version: 4
status: active
last_verified: 2026-07-19
-->

Purpose: define the downstream deterministic analysis layer for biological
shape, pose, and cross-component relationships derived from canonical refined
subject masks.

For the boundary between refined-mask-local geometry caches and downstream
analysis products, see
[refined_subject_mask_geometry_cache_and_propagation_design.md](refined_subject_mask_geometry_cache_and_propagation_design.md).
For the current snout-anchored centerline and B-spline workflow runbook, see
[subject_shape_snout_centerline_workflow.md](subject_shape_snout_centerline_workflow.md).
For the user-facing conventions around `caudal_contour_point_xy`,
`tail_base_xy`, `tail_tip_xy`, and `centerline_xy`, see
[subject_shape_landmark_conventions.md](subject_shape_landmark_conventions.md).
For future interoperability with Stytra, ZebraZoom, Megabouts, and
BEAST-style classifiers, see
[tail_kinematics_tool_interop_design.md](tail_kinematics_tool_interop_design.md).

## Scope

`analysis/subject_shape_runs/<run>` is the home for interpreted shape
outputs that should not be stored as mask-review metadata.

It should consume:

- one complete, selector-eligible `refined_subject_masks_runs/<run>` with a
  freshly verified `canonical_v2` coordinate publication as the exact
  mask-pixel, row-identity, ROI-frame, and placement authority
- dense `refined_subject_masks_runs/<run>/masks_roi` as the future-normal
  authoritative pixel surface; compact-only historical runs are migration or
  inspection inputs and are not eligible for new canonical subject-shape runs
- optional refined-subject mask-local geometry primitives
- optional `refined_keypoints_runs/<run>` or heading/track runs when anatomical
  polarity, body heading, or temporal alignment is required

It should produce deterministic derived geometry and analysis-ready shape
features.

The first implementation is `fisheye.analysis.subject_shape_runs`. It writes
row-aligned component summaries, body principal-axis estimates, eye/swim ellipse
summaries, eye-pair relations, and swim/eye-to-body relations with optional
Dask worker-chunk execution. Body centerline and B-spline methods remain
follow-up shape methods under this same run family. The same run family is also
the preferred materialized home for shared body-frame arrays derived from
refined masks, centerlines, B-splines, keypoints, or hybrid estimators.

Storage and execution tuning are tracked separately in
[subject_shape_performance_benchmark.md](subject_shape_performance_benchmark.md).
Logical compute/read chunks need not be physical Zarr file boundaries. Any
parallel writer using indexed physical shards must assign one complete,
non-overlapping shard to a writer, or compute into temporary block artifacts
and assemble final shards deterministically before publication.

## Boundary Rule

Use `refined_subject_masks_runs/<run>` for mask-local primitives:

- component contours
- component centroids
- component area and bbox
- component mask-present and validity metrics
- simple mask-shape descriptors such as component count, hole fraction,
  solidity, and documented ellipse/PCA fits
- eye ellipse parameters and eye-pair separation when used as immediate
  refined-eye geometry/QC

Use `analysis/subject_shape_runs/<run>` for interpreted biology:

- body centerline or spline used as an anatomical coordinate frame
- body B-spline fit, including centerline or outline models with method-specific
  smoothing/knot parameters
- canonical biological body length derived from centerline or B-spline arc
  length
- head/tail-polarized body axis or heading inferred from masks
- body curvature, bend, width profile, or body-shape summaries
- swim-bladder position relative to the body axis or centerline
- swim-bladder distance to body centroid, eye pair, or anatomical landmarks
- analysis-facing eye component geometry when it is part of the same coherent
  body/eyes/swim subject-shape run
- eye-pair metrics that are consumed as biological geometry rather than
  immediate mask-local QC
- eye angles relative to body/head heading
- temporally smoothed or track-aligned shape metrics
- tool-ready tail-posture exports may consume these arrays, but should not make
  third-party tool formats canonical Palette storage

Practical test:

- If the value is recomputable from one component mask without choosing an
  anatomical frame, keep it with `refined_subject_masks_runs`.
- If the value needs a coordinate convention, anatomical polarity, component
  relationship, track identity, temporal context, or smoothing policy, write it
  to `analysis/subject_shape_runs` or a more specific downstream analysis run.

Mask-quality prerequisite:

- `subject_body` mask-local QC belongs with `refined_subject_masks_runs`; see
  [subject_body_mask_qc_design.md](subject_body_mask_qc_design.md).
- Current subject-shape writers snapshot source body-mask QC into
  `components/subject_body/source_mask_qc_*` arrays when
  `components/subject_body/qc` is available on the refined source run.
- Current centerline/tail-base writers fail closed with
  `source_body_mask_qc_failed` when source QC marks
  `severe_qc_failure[row] == true`.
- Review-required but non-severe source mask QC is propagated as a warning
  snapshot; it does not by itself prevent candidate geometry from being
  computed.
- `analysis/subject_shape_runs` should not be the primary authority for whether
  the refined mask pixels are plausible.

## Non-Goals

- Do not store raw model probabilities here.
- Do not edit or approve mask pixels here.
- Do not replace `refined_subject_masks_runs` as the mask authority.
- Do not collapse specialized analyses such as `analysis/eye_angle_runs` or
  `analysis/pose_kinematics_runs` into this stage without a clear migration
  decision.

## Intended Stage Relationship

```text
subject_mask_runs/<run>               # raw probability evidence
  -> refined_subject_masks_runs/<run> # canonical refined masks + mask-local geometry
  -> analysis/subject_shape_runs/<run> # interpreted biological shape geometry
```

Optional inputs:

```text
refined_keypoints_runs/<run>
analysis/track_kinematics_runs/<run>
tracking_runs/<run>
```

## Canonical Coordinate Publication (Schema v4, Method v11)

New subject-shape runs publish one strict coordinate framework; coordinate
meaning is never inferred from a path suffix, array shape, value range, or
historical helper name.

- The selected refined-mask run and subject-shape output must be in the same
  archive. Cross-archive publication and overwrite-in-place are rejected.
- `instance_key`, `source_crop_row_ids`, and
  `source_acquisition_frame_index` are direct children of the output rowset and
  exact dtype-preserving copies of the selected refined rows. The bound
  `observation_instance/instance_key` contract is row identity; frame indices
  and crop row IDs are lineage, not alternate identities.
- Algorithms may compute against ROI-local masks internally. Published point
  geometry is transformed once into `source_camera_image_px` through the exact
  row-specific, direction-labelled ROI-to-camera placement. Schema v4 accepts
  translation-only placement; scale, padding, affine, or projective placement
  fails closed until vector, angle, and distance semantics are explicitly
  rederived.
- No paired ROI-local copies of the published point arrays are retained. A
  consumer that needs ROI-local presentation must apply an explicitly bound
  inverse transform outside the authoritative run.
- Each per-component `(N, 2)` point array uses `geometry_type = "point_xy"`.
  The aggregate `(N, components, 2)` `component_centroid_xy` also uses
  `point_xy`, plus `collection_axis.role = "subject_component"` and the exact
  digest-bound component-label authority. Uncollected `points_xy` is reserved
  for `(N, P, 2)` point sequences; polylines such as centerlines use
  `polyline_xy`.
- Every `bbox_xyxy` array uses pixel-edge, half-open bounds
  `[x_min, y_min, x_max_exclusive, y_max_exclusive]`. Both corners are
  transformed as edges; no consumer may reinterpret maxima as inclusive.
- Source-camera point, bbox, polyline, and ellipse arrays are direct camera
  overlay surfaces. Translation-invariant source-camera offsets use
  `source_camera_image_px.displacement_vector_y_down.v1`: `vector_xy`, pixel
  units, no origin or pixel sampling convention, and overlay status
  `not_suitable`. A displacement can be added to a bound point; it is not
  itself an overlay position. Unit body axes use the narrow
  `source_camera_image_px.unit_vector_y_down.v1` profile: `vector_xy`, unitless,
  `pixel_convention = "not_applicable"`, and overlay status `not_suitable`.
- `tail_tangent_xy` and `tail_normal_xy` are not polylines. They use the
  distinct `vector_sequence_xy` geometry with physical shape `(N, K, 2)` and
  the unit-vector profile above. Their axis 1 is bound to the exact
  `tail_sample_s` array by `palette.subject_shape_tail_sample_axis`; the
  authority declares normalized, unitless, strictly increasing arclength from
  tail base (`0`) to tail tip (`1`).
- Every emitted floating measurement scalar or scientific profile is
  registered in the closed `palette.subject_shape_scalar_surface_inventory`.
  Spline representation arrays such as knots and control-point tuples remain
  bound by the full payload manifest and scientific-configuration record. Each
  registered measurement array carries a
  `palette.subject_shape_scalar_surface` record that digest-binds units,
  quantity, sign convention, validity authority, producer method and
  configuration, row identity, value payload, and all coordinate-bearing basis
  arrays. This is semantic metadata on existing numeric arrays; it does not
  create duplicate numeric surfaces.
- `components/subject_body/tail_curvature_px_inv` is a signed `px^-1`
  row-profile surface. Profile axis 1 is bound to the exact persisted
  `tail_sample_s` payload and cardinality, validity is bound to
  `tail_sample_valid`, and sample direction is tail base to tail tip. In the
  source-camera x-right/y-down frame, the persisted curvature formula declares
  positive curvature clockwise in image view. Readers must consume this typed
  binding rather than infer units or direction from the array name.
- Longitudinal and lateral relation scalars bind the exact persisted basis
  arrays. In particular, offsets derived from the unoriented PCA principal axis
  do not acquire anatomical-forward or anatomical-left meaning merely because
  their paths contain `longitudinal` or `lateral`.
- The aggregate source centroids bind the controlled body estimator. The
  resulting `fish_anatomical_body_frame` v1 record digest-binds
  `origin_xy`, `forward_axis_xy`, `left_axis_xy`, authoritative `axis_valid`,
  exact source payload/schema/validity, estimator formula, and row identity.
- The direct output rowset carries a typed source-acquisition temporal authority
  bound to its exact `source_acquisition_frame_index` and `instance_key`.
- The exact refined component-QC inventory is closed-world and digest-bound.
  Every declared component records either explicit absence or the complete QC
  group attrs and flat array payloads; subject-shape publication revalidates
  this authority before computation and again before final binding.
- A scientific-configuration record seals all run parameters and component,
  relation, and body-frame group attrs that define the computation. A separate
  row-bound heading record binds `heading_deg`, `forward_axis_xy`, `axis_valid`,
  row identity, units, and the formula
  `degrees(atan2(-forward_y, forward_x))`.
- A closed publication manifest digests every output array and binds every
  array-specific coordinate descriptor, the exact refined context, surface and
  component-QC inventories, derivation/transform records, component and sample
  axes, temporal authority, scientific configuration, heading semantics, row
  identity, and body-frame record.

Cluster materialization follows a deferred-binding transaction. Scratch
contains only a completed, explicitly unbound ROI-local numeric stage and a
closed decoded-payload manifest: no canonical descriptor, direct output
identity, selector, or coordinate-completion claim is permitted there. After
deterministic sharding and atomic rename into the authoritative archive,
Palette freshly resolves the exact refined source, consumes the unbound
manifest, creates direct identity/temporal authorities, performs the exact
ROI-to-camera transform, seals descriptors and manifests, marks the child
complete but ineligible, and performs a fresh strict reload. Parent selectors
advance only after that reload; child eligibility is the final scientific and
selection-state mutation. The generic publisher may subsequently update only
the explicitly non-scientific `cluster_output_staging` operational receipt,
which is outside the immutable scientific manifest. Any post-rename failure,
including an operational-receipt failure, removes only the UUID-owned target
and restores the UUID-owned selector epoch.

Subject-shape activation owns an exact structured `latest_pending` receipt and
compares the complete selector/lifecycle epoch: selectors, publication
generation, policy, and lease. It freshly reloads the parent between every
parent write, and freshly reloads the complete child before selector
advancement and again before the final eligibility write. A failed attempt
rolls back only values still equal to that attempt's exact owned values; it
never overwrites a concurrent publisher's state. If the final eligibility
store persists and then raises, a fresh owner-bound eligible read proves the
commit and the publisher returns success rather than deleting a valid run.

The node-local tail-kinematics materializer is the one intentional detached
consumer. It first performs the full canonical subject-shape preflight in the
authoritative archive, then creates a closed, digest-bound staging receipt with
the canonical publication manifest, row identity, body-frame, tail-sample-axis,
tail-curvature semantic digests, source contract attrs, and exact payload
hashes for every permitted staged array. Workers accept that receipt only
through the private staged-subset path and revalidate it before and after
bounded reads. The receipt explicitly has `normal_reader_authority = false`;
it cannot make a partial staged Zarr readable through the normal
subject-shape API.

Published tail-derived runs bind the exact subject-shape input, not only its
run name. Canonical tail-kinematics readers require
`source_subject_shape_publication_manifest_sha256` and reject the run unless it
equals a fresh strict reload of the selected subject-shape publication.
Tail-posture writers persist the same digest both as a run attr and a
provenance input, then revalidate it after source reads and immediately before
activation.

Normal readers accept only complete, selector-eligible schema-v4 publications
that pass a fresh strict reload. Implicit selection requires matching `latest`
and `latest_complete` values naming that exact direct child; disagreement or an
ineligible child is an in-progress/invalid handoff and fails closed. Explicit
selection accepts only a bare direct-child name or the exact path
`analysis/subject_shape_runs/<run>`; extra prefixes or suffixes are rejected.
`historical_inspection=True` is an explicit audit/migration escape hatch and
returns no coordinate authority; scientific or presentation readers must not
use it as a legacy adapter.

Subject-shape eye-geometry and overlay readers follow the same rule. Historical
refined-subject eye geometry is available only through the explicitly named
noncanonical compatibility option, and the resulting source is labelled
`historical_compatibility_noncanonical`. It is not a future-normal adapter.

## Required Provenance

An `analysis/subject_shape_runs/<run>` writer should record:

- `schema_id = "analysis.subject_shape_runs"`
- `schema_version`
- `row_axis = "refined_subject_mask_rows"` for the first row-aligned writer
- `source_refined_subject_masks_run`
- `source_refined_subject_masks_stage = "refined_subject_masks_runs"`
- `source_mask_labels`
- `source_mask_label_schema_id`
- `source_mask_geometry_schema_id` when mask-local geometry was consumed
- `source_mask_store_encoding = "dense_uint8"` for a schema-v4 publication
- `source_mask_storage_surface = "masks_roi"`; historical compact encodings
  may be reported during inspection but are not new-run authorities
- `source_mask_store_path`, the exact physical refined-mask store consumed
- exact refined coordinate-context, surface-inventory, component-QC-inventory,
  refinement-authority, and row-identity record digests in the subject-shape
  derivation record
- method name and method version
- parameter/config hash or serialized config
- creation timestamp

Required when used:

- `source_refined_keypoints_run`
- `source_keypoint_heading_computation`
- `body_frame_schema_id`
- `body_frame_schema_version`
- `body_frame_estimator`
- `body_frame_source_refs`
- `source_tracking_run`
- `source_track_kinematics_run`
- `temporal_window`
- smoothing/filter method and parameters

## Proposed Layout

```text
analysis/subject_shape_runs/
  attrs:
    latest                         "<run_id>"
    latest_complete                "<run_id>"
    latest_pending                 structured owner-bound receipt while publishing only
    subject_shape_publication_generation nonnegative committed epoch
    subject_shape_publication_lease exact owner/run/epoch receipt
  <run_id>/
    attrs:
      schema_id                    "analysis.subject_shape_runs"
      schema_version               4
      source_refined_subject_masks_run
      source_mask_labels
      source_mask_label_schema_id
      source_mask_store_encoding   "dense_uint8"
      source_mask_storage_surface  "masks_roi"
      source_mask_store_path       exact physical source path
      method
      method_version
      created_at_utc
      row_axis                     "refined_subject_mask_rows"
      source_refs                  dict of exact input runs/paths
      coordinate_contract          "canonical_v2"
      bbox_convention              "xyxy_pixel_edge_half_open"
    instance_key                   (N,) uint64 authoritative row key
    source_crop_row_ids            (N,) exact selected refined source row
    source_acquisition_frame_index (N,) exact source-camera frame lineage
    component_centroid_xy          (N, C, 2) collected source-camera points
    component_centroid_valid       (N, C)
    coordinate_records/            exact component/body-frame authorities
      scalar_surface_inventory/    closed typed scalar/profile inventory
    row_index/
      frame_indices                (N,)
      detection_indices            (N,) optional
      source_refined_row_ids        (N,) optional
    source_refined_subject_masks/
      attrs:
        schema_id                   "analysis.subject_shape.source_refined_subject_masks_v1"
        source_stage                "refined_subject_masks_runs"
        source_run                  "<refined run>"
        component_names             list[str]
        row_revision_semantics      historical compatibility description
      row_revision                  (N, C) legacy compatibility snapshot
      row_revision_available        (C,) false for future-normal canonical sources
    body_frame/                     optional shared fish anatomical frame
      origin_xy                     (N, 2)
      forward_axis_xy               (N, 2)
      left_axis_xy                  (N, 2)
      heading_deg                   (N,)
      axis_valid                    (N,) authoritative
      valid                         (N,) explicit compatibility alias only
      failure_reason_bytes          (N, width) optional uint8 utf8-null-terminated tags
      midline_xy                    (N, P, 2) optional
      arclength_px                  (N,) optional
    components/
      subject_body/
        centroid_xy                (N, 2) optional mirror/cache
        contour_ref                optional references into refined mask contours
        source_mask_qc_available   (N,) optional bool snapshot from refined mask QC
        source_mask_qc_severe_failure (N,) optional bool
        source_mask_qc_requires_review (N,) optional bool
        source_mask_qc_reason_bytes (N, width) optional uint8 utf8-null-terminated tags
        snout_tip_xy               (N, 2) optional semantic rostral/nasal landmark
        snout_tip_valid            (N,) optional
        snout_tip_failure_reason_bytes (N, width) optional
        head_endpoint_to_snout_distance_px (N,) optional
        centerline_reaches_snout   (N,) optional
        centerline_snout_check_reason_bytes (N, width) optional
        centerline_xy              (N, P, 2) optional
        centerline_valid           (N,) optional
        bspline_control_points_xy  (N, K, 2) optional
        bspline_sample_xy          (N, P, 2) optional
        bspline_knots              optional
        bspline_degree             scalar attr
        bspline_degree_used        (N,) optional row-level degree used, -1 when invalid
        bspline_valid              (N,) optional
        bspline_failure_reason_bytes (N, width) optional
        centerline_arc_length_px   (N,) optional
        bspline_arc_length_px      (N,) optional
        axis_xy                    (N, 2) optional
        heading_rad                (N,) optional
        tail_tip_xy                (N, 2) optional
        tail_base_xy               (N, 2) optional
        tail_base_valid            (N,) optional
        tail_base_arclength_px     (N,) optional
        tail_base_failure_reason_bytes (N, width) optional
        tail_segment_arclength_px  (N,) optional
        body_arclength_px          (N,) optional
        tail_sample_s              (K,) optional normalized tail arclength samples
        tail_sample_xy             (N, K, 2) optional   [interpolating spline: faithful positions]
        tail_tangent_xy            (N, K, 2) optional   [smoothing spline, see below]
        tail_normal_xy             (N, K, 2) optional   [smoothing spline]
        tail_curvature_px_inv      (N, K) optional      [smoothing spline]
        tail_sample_valid          (N,) optional
        tail_sample_failure_reason_bytes (N, width) optional
        tail_width_px              (N, K) optional
        tail_width_valid           (N, K) optional
        centerline_curvature_px_inv (N, P) whole-body (snout->tail) curvature [smoothing spline]
        validity/
        quality/
          preferred_body_length_px (N,) optional selected centerline/spline length
          preferred_tail_length_px (N,) optional selected tail-segment length
          tail_to_body_length_ratio (N,) optional
          body_length_delta_px     (N,) optional gap-aware temporal delta
          body_length_delta_fraction (N,) optional
          tail_length_delta_px     (N,) optional gap-aware temporal delta
          tail_length_delta_fraction (N,) optional
          body_length_robust_z     (N,) optional recording-local robust z score
          tail_length_robust_z     (N,) optional recording-local robust z score
          tail_to_body_ratio_robust_z (N,) optional
          length_qc_flags          (N,) optional bool
          length_qc_severity       (N,) optional uint8 enum
          length_qc_reason_bytes   (N, width) optional pipe-delimited stable tags
          summaries/               optional run-level quantiles/MAD/histograms
      swim_bladder/
        centroid_xy                (N, 2) optional mirror/cache
        ellipse_params             (N, 5) optional
        caudal_contour_point_xy    (N, 2) optional
        caudal_contour_projection_px (N,) optional
        caudal_contour_valid       (N,) optional
        caudal_contour_failure_reason_bytes (N, width) optional
        validity/
      eye_left/
        ellipse_params             (N, 5) optional mirror/cache
        validity/
      eye_right/
        ellipse_params             (N, 5) optional mirror/cache
        validity/
    relations/
      eye_pair/
        separation_px              (N,) optional mirror/cache
        separation_valid           (N,) optional
      swim_bladder_to_body/
        longitudinal_position      (N,) optional
        lateral_offset_px          (N,) optional
        distance_to_centerline_px  (N,) optional
      eyes_to_body/
        left_eye_angle_rad         (N,) optional
        right_eye_angle_rad        (N,) optional
```

Array presence is intentionally method-specific: a writer should emit only the
surfaces it can scientifically validate. Coordinate authority is not
permissive—every emitted canonical coordinate surface must have its exact
array-specific descriptor and manifest binding.

Schema-v4 tail samples remain subject-shape geometry samples. They are
used to support geometry review, width/curvature profiles, and downstream
resampling. They should not be assumed to be the final low-dimensional
behavioral tail-angle vector.

### Two-spline tail geometry (method v9+, 2026-07-15)

Positions (`tail_sample_xy`) and arc length come from the **interpolating** spline
(`bspline_smoothing = 0`), which faithfully follows the mask skeleton. Every **differentiated**
quantity comes from a single **separate smoothing spline** fit to the same points
(`tail_curvature_method = separate_smoothing_spline_v1`,
`s = n_points * tail_curvature_smoothing_px^2`, default 0.75 px; recorded in attrs
`tail_curvature_method` / `tail_curvature_smoothing_px`):

- **`centerline_curvature_px_inv` (N, K_centerline)** — whole-body (snout→tail) signed curvature,
  available whenever the spline is valid (does not need a valid tail base). This is the array for a
  **whole-body bend / C-coil** metric.
- **`tail_curvature_px_inv` (N, K_tail)**, `tail_tangent_xy`, `tail_normal_xy` — the same, over the
  **tail segment** (tail base → tip, the posterior ~half). This is the array for a **tail-beat**
  metric.

Summarize either by the **integrated angle** — Σ of the between-point curvature × arclength
(= ∫|κ|ds, degrees) — not the max, which is an outlier that tracks the noisy tail tip. Signed
curvature also gives net turn (∫κ ds).

**Why:** curvature is the second derivative of the centerline, and an interpolating spline
through the ±0.5–1 px pixel-quantization jitter of the mask skeleton produces meaningless
sub-pixel bend radii (v8 median max curvature ≈ 1 px radius on a 75 px tail — pure noise). The
smoothing spline removes the jitter while preserving real, coherent bends (a synthetic 20 px-radius
arc reads back at ≈1/20). On the one re-materialized recording this moved the median max-curvature
radius from **1.1 px → 60 px**. Pinned by `test_tail_curvature_uses_a_smoothing_spline...` and
`test_smoothing_preserves_a_real_body_bend`.

**Residual caveats (the fit is fixed; the masks/construction are not).** Whole-body curvature is
U-shaped along the body: a rigid straight trunk (~500 px radius) with curvature elevated at **both
ends**. The endpoints are *not* a spline-endpoint or snout-join artifact — the spline reaches the
real snout tip (centerline point 0 = the anatomical snout tip, distance 0), and the anterior
curvature is a smooth ramp (only ~1.2× the just-inside value at the endpoint), not a spike or a
corner. Instead:

- **trunk** — genuinely rigid and stable over time (temporal std 0.0035);
- **tail** — variable curvature = real tail flexing (and, in the extreme <5 px-radius outliers,
  bad masks / tip curl);
- **head** — elevated curvature that is *variable in sign frame-to-frame* (curvature is
  rotation-invariant, so a rigid head would be sign-stable). This is a mix of real head shape and
  **instability in the head-region centerline construction** (the medial/bridge path near the head
  blob and eyes wanders), not a clean artifact.

The endpoints contribute ~27% of the integrated whole-body bend, so for a robust metric: **work in
the trunk+tail and use curvature *change over time* (a C-start is a transient deviation from the
resting posture), not absolute whole-body curvature**; trimming the head region helps pragmatically
because it is noisier. Summarize by the integrated angle not the max, and QC-reject impossible radii
(< ~5 px). Note `tail_sample_valid` fails ~10% of frames. See
`docs/diagnostics/subject_shape_tail_curvature_2026-07-15.md`.

The low-dimensional behavior-facing representation should be written by
`analysis/tail_kinematics_runs`, defaulting to approximately `K=10` normalized
tail samples unless a method records a different value. That run can evaluate
the valid B-spline/tail geometry at its own `tail_angle_sample_s` positions and
record exact sampling/count conventions.

Schema bump rule:

- Do not bump subject-shape schema solely because a downstream
  `tail_kinematics_runs` writer derives `K=10` behavior samples from existing
  schema-v4 geometry.
- Do bump `analysis.subject_shape_runs` beyond schema v4, and bump the
  subject-shape method version, if this run family changes the semantics,
  default dimensionality, or intended role of `tail_sample_xy` itself.

For realtime viewers such as Crimson, subject-shape runs may add
non-authoritative `frame_index/` and `track_index/` lookup groups so consumers
can resolve rows by frame or track without scanning all row-aligned arrays. The
canonical shape arrays remain sparse and row-aligned. See
[realtime_sparse_row_index_contract.md](realtime_sparse_row_index_contract.md).

For schema-v4 coordinate authority, the direct
`source_acquisition_frame_index` array is digest-bound to the same row identity
as every coordinate surface. Historical `row_index/frame_indices` remains a
lineage/viewer compatibility surface and may be absent. A future CSR-style
`frame_index/` cache may be added for convenience, but it must remain a derived
lookup over the same stable row axis.

## Source Revision And Staleness

Subject-shape runs are downstream analysis products. They should not silently
change when a refined mask row is manually edited.

Future-normal canonical refined-mask runs are immutable snapshots. Any added,
removed, or changed source payload or namespace invalidates the sealed refined
publication and therefore invalidates strict reload of every dependent
subject-shape publication. An accepted mask edit must create and activate a new
refined run/publication; recomputation creates a new subject-shape run from
that exact source. A mutable `row_revision` counter is not a canonical
scientific-consumer authority.

The following arrays remain only as historical archive compatibility surfaces:

```text
analysis/subject_shape_runs/<run>/source_refined_subject_masks/row_revision
analysis/subject_shape_runs/<run>/source_refined_subject_masks/row_revision_available
```

For a future-normal source, values are zero and
`row_revision_available[component]` is false. Explicit migration/audit tooling
may inspect historical mutable archives and compare legacy revisions, but
normal writers and readers do not use that result to grant coordinate or
scientific authority. They require the strict sealed source records instead.

The historical inspection command remains:

```bash
scripts/py -m fisheye.analysis.subject_shape_runs /path/to/analysis.zarr \
  --audit-source-revisions \
  --shape-run <subject_shape_run>
```

This audit is read-only. Stale rows should be recomputed by an explicit
subject-shape recompute command, not by automatic propagation from the mask edit
save path.

## Row Identity, Frame Lookup, And Track Identity

`analysis/subject_shape_runs/<run>` is row-aligned to the selected refined
subject-mask source. Schema v4 directly persists and binds:

```text
instance_key
source_crop_row_ids
source_acquisition_frame_index
```

It may additionally preserve historical or convenience lineage under
`row_index/` when available:

```text
row_index/frame_indices
row_index/detection_indices
row_index/source_refined_row_ids
row_index/source_detect_row_index
```

Those arrays answer lineage questions, not fast viewer lookup questions. For
interactive display, large subject-shape runs should also include the optional
CSR-style `frame_index/` cache described in
[realtime_sparse_row_index_contract.md](realtime_sparse_row_index_contract.md).
That cache maps a displayed frame to the physical subject-shape rows that
should be drawn.

Do not substitute a same-length `row_index` array for the direct, digest-bound
schema-v4 identity and temporal lineage arrays.

Subject-shape rows should not treat `track_id` as primary identity. Track IDs
are optional temporal/biological identity assignments from an exact tracking
source. They are useful for grouping shape rows by animal, but they do not
replace stable row lineage for stale detection or row-local recomputation after
mask edits.

The intended relationship is:

```text
source row ID / subject row ID -> row-local source identity and stale repair
frame_index/                   -> fast frame-to-row lookup for viewers
track_id                       -> optional animal-over-time grouping
```

## Body Frame Placement

`analysis/subject_shape_runs/<run>/body_frame/` is the preferred shared
materialized location for fish anatomical body-frame outputs.

Reasoning:

- a body frame is deterministic derived biology, not reviewed mask-pixel
  authority
- the best estimator may come from body masks, centerlines, B-splines, keypoints,
  mask component centroids, or a hybrid of those sources
- downstream consumers such as eye angles, body/tail shape metrics, and
  bout-level metrics should be able to consume the same frame without
  duplicating sign and polarity conventions

The body-frame contract separates semantic anchors from estimators:

- semantic anchors name anatomy, such as `swim_bladder`, `eye_left`,
  `eye_right`, `subject_body`, `tail_tip`, or `snout_tip`
- estimators declare how those anchors were measured and how frame arrays were
  materialized
- outputs expose shared arrays such as `origin_xy`, `forward_axis_xy`,
  `left_axis_xy`, `heading_deg`, and authoritative `axis_valid`; `valid` is an
  explicitly declared compatibility alias in method v11

Keypoint-only datasets remain valid. A writer may materialize a body frame from
`pose_schema.metadata.heading_computation` when masks or body splines are not
available. Mask/spline estimators should preserve keypoint or mask-component
anchors for head/tail polarity and anatomical left/right resolution.

The body-frame origin is estimator-defined. In the current mask-component
estimator it is the eye-pair midpoint, which is useful for polarity but is not
the most rostral/nasal point of the animal. Consumers should not treat
`body_frame/origin_xy` as `snout_tip`.

See [body_frame_contract.md](body_frame_contract.md).

## Component And Relation Organization

`analysis/subject_shape_runs` should preserve the same semantic component names
used by `refined_subject_masks_runs`, but the meaning is different:

- `refined_subject_masks_runs/components/<component>` owns reviewed mask pixels,
  mask-local QC, and component-local geometry that is directly recomputable from
  one mask channel.
- `analysis/subject_shape_runs/components/<component>` owns interpreted
  biological geometry derived from those component masks.

Use component groups for values whose primary subject is one semantic component:

- `components/subject_body` for centerlines, B-splines, body length, body axis,
  curvature, rostral/snout landmarks, tail-normalized width profiles, and
  body-shape validity.
- `components/swim_bladder` for swim-bladder centroid/blob/ellipse summaries
  and component-specific validity.
- `components/eye_left` and `components/eye_right` for analysis-facing eye
  component geometry, ellipse/axis summaries, and component-specific eye
  validity consumed by coherent subject-shape analysis.

Use `relations/` for values whose meaning depends on more than one component or
an external coordinate frame:

- `relations/eye_pair` for cross-eye metrics such as separation.
- `relations/swim_bladder_to_body` for swim-bladder position along or relative
  to the body axis/centerline.
- `relations/eyes_to_body` for eye angles or offsets relative to body/head
  heading.

Component groups in `analysis/subject_shape_runs` are not approval surfaces. A
shape run may mark a component-derived value invalid or failed without changing
the source component's review state in `refined_subject_masks_runs`.

## Body B-Spline Policy

The canonical body B-spline fit belongs in
`analysis/subject_shape_runs`, not in `refined_subject_masks_runs`.

Reasoning:

- a B-spline is a fitted continuous curve model, not just a direct mask
  primitive
- its output depends on degree, knot count, parameterization, interpolation
  versus smoothing/regularization policy, resampling, and failure policy
- if the spline is used as a body coordinate frame, it also depends on anatomical
  polarity or heading source
- recomputing or improving the fit should create or update a derived analysis
  shape run without mutating the reviewed mask-pixel authority

Allowed refined-mask-side exception:

- a writer may store raw component contours or clearly marked non-canonical debug
  seeds with the refined body component
- those seeds must not be treated as the canonical body spline or body axis

Minimum recommended B-spline provenance:

- `source_refined_subject_masks_run`
- `source_component = "subject_body"`
- contour or mask source used for the fit
- spline method/version
- spline degree
- knot/parameterization policy
- interpolation versus smoothing/regularization mode
- smoothing or regularization parameters, including an explicit no-smoothing
  value when the spline is interpolating
- head/tail polarity source if the spline is oriented
- per-row validity/failure reason

## Rostral/Snout Landmark Policy

`snout_tip` is a semantic anatomical landmark for the most rostral/nasal point
of the fish. Some pose schemas may include a keypoint named `snout_tip`,
`nose_tip`, or an equivalent marker. Subject-shape runs may also estimate this
point from the body mask.

Recommended mask-derived estimator:

```text
subject_body contour
  -> body-frame forward-axis projection
  -> maximum forward-coordinate contour point
  -> snout_tip_xy
```

Recommended fields:

- `components/subject_body/snout_tip_xy`
- `components/subject_body/snout_tip_valid`
- `components/subject_body/snout_tip_failure_reason_bytes`
- `components/subject_body/head_endpoint_to_snout_distance_px`
- `components/subject_body/centerline_reaches_snout`
- `components/subject_body/centerline_snout_check_reason_bytes`

Important distinctions:

- `body_frame/origin_xy` may be an eye-pair midpoint or another
  estimator-defined origin; it is not automatically the snout.
- In schema v3+, `head_endpoint_xy` is the snout-anchored anterior endpoint for
  valid centerlines.
- In schema v2/v5 archives, `head_endpoint_xy` was the anterior endpoint of the
  skeleton-derived centerline estimator and could stop short of the rostral
  contour.
- `snout_tip_xy` should be the preferred semantic rostral endpoint when present.
- pose/keypoint `snout_tip` should stay in the pose/keypoint run; comparisons
  against mask-derived `snout_tip_xy` should be stored as comparison metrics,
  not by overwriting either source.

Adding these arrays is backward-compatible for consumers that feature-detect
optional arrays, but a writer that makes them first-class subject-shape outputs
should bump `method_version` and should bump `schema_version` when the presence
or semantics of snout fields become part of the declared schema contract.

Current implementation:

- `schema_version = 4`
- `method = "subject_shape_from_refined_masks_v11"`
- `method_version = 11`
- `snout_tip_estimator = "subject_body_contour_max_forward_projection_v1"`
- `centerline_method = "snout_anchored_skeleton_longest_endpoint_path_v1"`
- `centerline_skeleton_method = "skeleton_longest_endpoint_path_v1"`
- `centerline_snout_extension_method = "prepend_mask_path_to_body_frame_guided_join_v1"`
- `centerline_snout_join_method = "body_frame_lateral_min_head_region_v1"`
- `head_endpoint_semantics = "validated_snout_tip"`
- `centerline_snout_check_method = "head_endpoint_to_snout_distance_v1"`

The original v8/schema-v3 writer made the snout semantic change explicit; the
v11/schema-v4 writer retains it while adding strict coordinate publication:
`head_endpoint_xy` is written as the validated `snout_tip_xy` for every row with
`centerline_valid = true`. The centerline/spline is generated by prepending a
bounded mask-path snout-to-skeleton segment before resampling. The skeleton
join point is selected from the medial head region using body-frame lateral
coordinates, rather than blindly using the first skeleton endpoint, because
head-side skeleton branches can pull the spline into off-axis mask offshoots.
This avoids rejecting normal curved/rounded head masks just because a straight
chord from the snout to the skeleton endpoint briefly leaves the body, while
also avoiding branch endpoints that are not on the body midline. If the snout is
missing, the bridge is too long, or no bounded mask path can be found, the
centerline fails closed instead of writing a legacy head endpoint under the new
schema.

Older v5/schema-v2 runs did not redefine `head_endpoint_xy`; they wrote
`head_endpoint_to_snout_distance_px` and `centerline_reaches_snout` as an
intermediate QC bridge so the semantic gap could be audited before this schema
bump.

## Body Length Policy

Palette should distinguish approximate mask-QC long-axis measurements from
canonical biological body length.

Mask-local approximations may live in `refined_subject_masks_runs`:

- `major_axis_length_px` from a documented PCA or ellipse fit
- `feret_diameter_px` from the maximum contour-point separation

Those values are useful for QC, triage, and rough size filtering, but they are
not the canonical biological body length because they are sensitive to contour
noise, fins, posture, and the chosen approximation.

Canonical body length should live in `analysis/subject_shape_runs`:

- `centerline_arc_length_px` when derived from a validated centerline
- `bspline_arc_length_px` when derived from a validated body B-spline
- future `snout_to_tail_arclength_px` or equivalent when a validated
  `snout_tip_xy` is used to extend or validate the anterior endpoint

Required semantics:

- length units must be explicit (`px`, or calibrated physical units when
  available)
- the writer must record the source centerline/B-spline method and sampling
  convention
- invalid or ambiguous fits must set the length value to NaN and write a
  validity/failure reason
- if both an approximate long-axis metric and a spline/centerline length exist,
  downstream biological analyses should prefer the spline/centerline length
- if `head_endpoint_xy` and `snout_tip_xy` disagree, consumers should not assume
  body length is snout-to-tail unless the run explicitly declares that estimator
  and validity status

## Subject-Shape Length QC

Length stability is a downstream subject-shape QC problem, not a refined-mask
approval rule.

Reasoning:

- biological body length should be nearly stable within a short recording
- tail-segment length should be stable enough that sudden drops are strong
  evidence for a clipped/truncated tail mask, failed centerline, or bad spline
  endpoint
- the absolute value depends on the selected centerline/spline estimator, so
  this QC belongs with `analysis/subject_shape_runs`
- refined masks remain canonical; length QC should never silently extend or
  repair mask pixels

Recommended row-local metrics:

- `preferred_body_length_px`: selected canonical body length for this method,
  preferring a valid B-spline arc length when present, otherwise a validated
  centerline arc length
- `preferred_tail_length_px`: selected tail-segment length from tail base to
  tail tip using the same geometry model as the body length
- `tail_to_body_length_ratio`: dimensionless tail-length sanity check
- `body_length_delta_px` and `body_length_delta_fraction`: gap-aware temporal
  change from the nearest prior valid row in the same track or single-fish row
  sequence
- `tail_length_delta_px` and `tail_length_delta_fraction`: equivalent
  tail-segment change metrics
- robust recording-local scores such as `body_length_robust_z`,
  `tail_length_robust_z`, and `tail_to_body_ratio_robust_z`

Recommended run-level summaries:

- valid count, invalid count, and reason counts
- median, MAD, min, max, and quantiles such as q01, q05, q25, q75, q95, q99
- optional histogram bin edges/counts for body length, tail length, and
  tail/body ratio

Temporal deltas must be gap-aware. Near-term single-fish-per-dish runs may
compute deltas in row order when `row_axis` is frame-aligned and one subject is
present. Multi-subject or sparse-track runs must compute deltas within a
declared `track_id`/`track_index` grouping and must not compare different
subjects or bridge long gaps as though they were adjacent frames.

Recommended length-QC reason tags:

- `ok`
- `source_mask_qc_failed`
- `centerline_invalid`
- `bspline_invalid`
- `tail_sample_invalid`
- `body_length_missing`
- `tail_length_missing`
- `body_length_low_outlier`
- `body_length_high_outlier`
- `tail_length_low_outlier`
- `tail_length_high_outlier`
- `tail_to_body_ratio_low`
- `tail_to_body_ratio_high`
- `temporal_body_length_drop`
- `temporal_body_length_jump`
- `temporal_tail_length_drop`
- `temporal_tail_length_jump`
- `track_gap`
- `insufficient_baseline`

A single row may have multiple reason tags. The recommended primary encoding is
a null-terminated UTF-8 string in `length_qc_reason_bytes`, using `|` as the
stable delimiter, for example:

```text
tail_length_low_outlier|temporal_tail_length_drop
```

The delimited reason string is the compact audit trail. Writers may also expose
boolean convenience arrays for high-volume consumers, but those arrays must be
derived from the same tags and must not introduce a second source of truth.

Do not start with an opaque scalar QC score as the authority. If a future writer
adds `length_qc_score`, it should be a convenience value derived from explicit
reason tags, source validity, and documented weights. Consumers that need to
explain or review a frame should use the reason tags.

## Relationship To Existing Analysis Runs

`analysis/eye_angle_runs` computes interpreted eye angles from eye geometry plus
heading/keypoint context. It remains a valid specialized analysis run, but it
is not the first authority for mask-derived eye shape geometry in unified
body/eyes/swim workflows.

Current eye-angle v5 runs opt into `analysis/subject_shape_runs` as the
preferred source when left/right eye ellipse geometry is present. They record
`schema_id = "analysis.eye_angle_runs"`, `schema_version = 5`,
`method = "ellipse_and_centroid_eye_angles"`,
`row_axis = "keypoint_detection_rows"`, `source_geometry_kind`, and
`eye_angle_output_schema` so consumers can distinguish subject-shape,
refined-subject, and legacy refined-eye geometry sources. Schema v5 also
records `preferred_angle_family = "gaze"` and
`preferred_eye_axis = "ellipse_major"` because the major axis is the canonical
eye-orientation axis. The gaze/minor direction is derived from the resolved
major axis with eye-specific 90 degree rotations, and keypoint-derived
`support/body_frame/` arrays define signed-angle polarity. It retains the
v3-compatible `vergence_gaze_deg` total/axis separation and adds per-eye nasal
gaze plus
`mean_eye_vergence_gaze_deg` for Johnson/BEAST-style comparisons. Output
schema v6 adds `left_eye_angle_deg`, `right_eye_angle_deg`, and
`vergence_eye_angle_deg` for Bianco/Engert-style nasal-positive eye-frame
angles. Output schema v7 adds `eye_angle_variant_schema` so UI consumers can
select among eye-frame, gaze, nasal-gaze, major-axis, centroid, and legacy
representations from metadata.

`analysis/subject_shape_runs` should not force every specialized metric to move
immediately. It defines the mask-derived shape layer that can later feed or
replace specialized analyses when that migration is justified.

Recommended near-term approach:

- keep refined-subject eye contours, ellipse fits, and eye-pair checks in
  `refined_subject_masks_runs` when they are mask-local QC/source primitives.
- include `eye_left` and `eye_right` component geometry in
  `analysis/subject_shape_runs` when producing a coherent body/eyes/swim shape
  run.
- keep current eye-angle outputs in `analysis/eye_angle_runs`; eye-angle writers
  should consume `analysis/subject_shape_runs` when mask-derived eye geometry is
  available there, with refined-subject and refined-eye geometry retained as
  explicit compatibility fallbacks.
- do not create a separate eye-analysis authority for mask-derived eye geometry
  unless it is a downstream temporal, behavioral, or task-specific analysis.

Near-term subject-shape implementations may target single-fish-per-dish data,
but the contract should remain sparse and row-aligned. Writers and viewers
should allow multiple rows with the same `frame_index`, should not encode
identity in component channels, and should only add `row_index/track_ids` or
track indexes after joining against one exact `tracking_runs/<run>` source.
This preserves a path to future multi-subject tracking without blocking current
body QC, centerline, tail-anchor, and spline work.

## Open Questions

- Which body centerline method is the first supported implementation?
- Should body/eyes/swim shape outputs be track-aligned from the start, or
  remain row-aligned with refined masks until tracking is explicitly requested?
- What approval/quality threshold should be required before a mask/spline body
  frame supersedes a keypoint-only fallback?
- How should the first tail-anchor/spline implementation choose between raw
  centerline samples and B-spline samples for canonical body and tail length?
  See [body_spline_tail_anchor_design.md](body_spline_tail_anchor_design.md).
- What tail sampling density and width-probe policy should become the default
  for tail curvature and mask-width profiles?

## Related Documents

- [body_spline_tail_anchor_design.md](body_spline_tail_anchor_design.md)
- [body_frame_contract.md](body_frame_contract.md)
- [current_pipeline_contract.md](current_pipeline_contract.md)
- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [realtime_sparse_row_index_contract.md](realtime_sparse_row_index_contract.md)
- [refined_subject_masks_runs_contract.md](refined_subject_masks_runs_contract.md)
- [subject_mask_refinement_todo.md](subject_mask_refinement_todo.md)
- [pose_kinematics_run_design.md](pose_kinematics_run_design.md)
- [src/fisheye/docs/eye_angle_conventions.md](../src/fisheye/docs/eye_angle_conventions.md)
