# Subject-mask component area support — 2026-08-30

## Outcome

Tiny component predictions are now rejected during refined subject-mask
finalization using a policy bound to the exact model and its approved training
data. They are rejected before ellipse fitting, so OpenCV never receives the
4--12-pixel masks that produced nondeterministic geometry in the full-duration
canary.

This is a model-support boundary, not an anatomical truth claim. A required
component remains part of the output schema even when its prediction fails.
The refined component is empty for that row, the failure is explicit, and the
publication remains valid. The raw inference probabilities are not modified.

## Training evidence

The active model identity is:

- registry set:
  `subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001`
- registry run: `subject_masks_union_all_components_v001`
- model artifact SHA-256:
  `217da20cd6ed780f5efe2c16add7cb932f40f08aac2f6e44795c0c381283839c`
- model label schema: `subject_v1_union`
- training-manifest SHA-256:
  `1b7b5c7ed69bc52f9120dbad47d704f32f67099b2ec29f10c9c4df83ee074d01`

The historical merged training Zarr has the dense masks but does not contain a
persisted grouped component-metrics profile. The reference statistics were
therefore reconstructed read-only from `metrics/area_px` in the 15 approved
`refined_subject_masks_all_components_training_20260425` source runs named by
the sealed training manifest:

- metadata read mode: `unconsolidated_explicit`
- approved rows: 3,153
- reference mask shape: 512×512
- body positive labels: 3,153
- left-eye positive labels: 3,153
- right-eye positive labels: 3,153
- swim-bladder positive labels: 3,153

No statistics from the four current inference recordings were used to choose
the boundary.

## Derived support profile

The policy uses the minimum positive normalized area represented in the
approved training masks. Left and right eyes share one symmetric eye-family
floor; the smaller of their two observed minima is used for both identities.

| Component family | Training minimum at 512×512 | Applied floor at 384×384 |
|---|---:|---:|
| subject body | 1,945 px | 1,095 px |
| eye (`eyes_union`, `eye_left`, `eye_right`) | 92 px (`left=92`, `right=100`) | 52 px |
| swim bladder | 167 px | 94 px |

Resolution conversion uses ceiling division over normalized ROI area, so the
floor cannot be rounded downward. The exact version-controlled profile is:

- profile ID:
  `cedar_shadow_omnifin0_subject_v1_union_component_area_support_v1`
- profile payload digest:
  `b6d7279f844aa9cafbdfc670453196327815e1536e3f7b4d43f6d506412dd181`
- profile document SHA-256:
  `ef94c1564286f7d75e52ec88da09ffa308742b182b951689ca748c71ada88619`
- profile path:
  `configs/fisheye/subject_mask_component_support/217da20cd6ed780f5efe2c16add7cb932f40f08aac2f6e44795c0c381283839c.json`

Profile resolution requires an exact match on registry set, registry run,
model artifact SHA-256, and label schema. Production-proof finalization fails
closed if this evidence or its matching profile is absent. Older diagnostic
and compatibility sources without a complete model identity retain the
generic deterministic geometry guard but cannot claim this model-bound policy.

## Finalization and publication semantics

The policy is applied at two levels:

1. Raw `subject_body`, `eyes_union`, and `swim_bladder` connected components
   below their model-supported area are removed during deterministic
   finalization.
2. After `eyes_union` is assigned anatomically, each left/right eye is checked
   again against the pooled eye-family floor before ellipse fitting.

Affected rows include these explicit reason tags:

- `cleanup_removed_below_model_supported_area`
- `needs_review_below_model_supported_area`

If one assigned eye is rejected, the row is `assigned_needs_review`. If both
are rejected, it is `failed_below_model_supported_area`. If upstream
`eyes_union` cleanup removes all eye support first, its model-support reason is
combined into both derived eye-component reason records. An all-row component
failure is recorded using
`record_failures_without_blocking_refined_run_publication_v1`; it does not
invalidate an otherwise coherent run.

The complete profile binding is copied into:

- the refined-run attributes;
- stage provenance inputs;
- per-component source/policy records; and
- the refined subject-mask scientific identity.

This makes the decision reproducible from the artifact itself. The raw
subject-mask probability surface remains the immutable inference evidence, so
a future model or policy can be evaluated without recovering discarded
pixels.

## Geometry fallback

`measure_mask_ellipse` also rejects masks with fewer than 13 foreground pixels
before calling `cv2.fitEllipse`. This is a deterministic defensive boundary
for historical or external masks, based directly on the unstable 4--12-pixel
population found in the canary. It is not a substitute for the stronger
model-bound refinement policy.

The stamped geometry method is
`cv2.fitEllipse_component_contour_min_13_foreground_pixels_v2`, so old and new
geometry products cannot silently claim the same estimator.

## Validation contract

Required regression coverage includes:

- exact model-profile resolution and model mismatch rejection;
- exact 512→384 floor scaling;
- incomplete or internally inconsistent profile rejection;
- deterministic component cleanup with explicit failure reasons;
- proof that below-support eyes are rejected before OpenCV is invoked;
- real finalizer publication with the profile bound into attributes and the
  scientific identity;
- process-sharded finalization with every worker required to resolve the exact
  parent-admitted profile ID, payload digest, and document SHA-256; and
- publication of an all-failed eye component with explicit failure evidence.

Existing immutable publications are not mutated. Applying this policy to an
already-published recording requires a new immutable refined successor.

## Follow-up: populate grouped metrics at export time

Reconstruction was necessary only because the historical training export did
not persist the grouped metrics it had enough information to compute. Future
training/export should compute the grouped component distributions from the
admitted dense training masks, seal them to the training manifest, attach the
result to the model artifact/registry identity, and install the runtime support
profile automatically. A separate implementation audit will identify the
single producer and registry boundary for that work; the hand-maintained
profile in this change is the migration case, not the desired steady state.
