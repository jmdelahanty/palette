# Recording-Bundle Subject-Shape Publication

Date: 2026-08-10

Status: the selector-ineligible candidate and the supported selector-visible
profile are implemented and fixture-validated. The supported-profile change is
awaiting CI and recording-scale workload evidence; no production selector,
registry authority, or recording artifact changed in this checkpoint.

## Decision

New subject-shape publications consume one exact recording-level subject-mask
bundle. They do not infer a refined-mask `latest` pointer and do not
reconstruct the historical single-run coordinate authority.

The new logical identity is:

- profile `analysis.subject_shape.full_anatomy_v5`;
- `analysis.subject_shape_runs` schema version 5;
- method `subject_shape_from_recording_mask_bundle_v12`, version 12;
- row axis `recording_subject_mask_bundle_rows`.

Historical profile v4/method v11 remains unchanged for explicit historical
inputs. A v5 run is not represented as v4 and the sealed recording-bundle
source is not represented as `BoundRefinedSubjectMaskCoordinateSurfaces`.

## Publication Boundary

The materializer requires an explicit subject-mask bundle ID. An inactive
bundle is accepted only with the explicit canary authorization flag. A
conflicting refined-run argument fails closed.

Before computation, the bundle source revalidates the bundle, its refined
member, crop-v2 row identity and placement, recording/camera/frame axis, and
the acquisition, continuous-pixel, and half-open-pixel-edge camera-frame
authorities. The persisted source-binding record includes exact record
references and digests for those authorities.

The node-local unbound numeric stage retains the exact bundle ID, source
binding and digest, component order, row identity, acquisition-frame index,
crop-row identity, and activation state observed at derivation. Publication
then:

1. rematerializes the complete closed array inventory through the shared byte
   planner;
2. validates decoded equality with the producer seal;
3. atomically installs the run at its final path;
4. reopens the exact named bundle and binds row identity, temporal authority,
   source-camera frames, coordinate descriptors, body frame, scalar surfaces,
   derivation, and publication manifest;
5. validates the bound access-aware storage receipt and source-manifest link;
6. marks the run complete but selector-ineligible;
7. either preserves that state for the explicit candidate or performs the
   guarded selector activation for the supported profile; and
8. consolidates root metadata as the final published visibility step, then
   reopens the exact run through the canonical resolver and storage validator.

The low-level historical writer rejects direct bundle-v5 publication. A final
v5 run must pass through the access-aware materializer, preventing a valid v5
logical envelope from being published accidentally with the legacy physical
layout. The selector-activation gate admits bundle-v5 only when the run
declares the supported profile and role and carries no candidate envelope. The
candidate is still rejected and retains its no-pointer lifecycle.

The candidate profile `subject_shape_access_aware_candidate_v1` and supported
profile `subject_shape_access_aware_v1` use the same physical policy:
approximately 128 KiB inner chunks for row-aligned arrays, 1 MiB eager chunks,
and 8 MiB indexed shards. They are separate lifecycle profiles rather than one
candidate being mutated in place. Historical materialized v4 and supported
bundle-v5 both resolve through `resolve_canonical_subject_shape_run`; eye-angle
and tail-kinematics consumers require no profile-specific adapter.

## Fixture Evidence

The focused real-Zarr integration fixture proves:

- a four-component recording bundle can produce a bound v5 run;
- exact component, instance, crop-row, and acquisition-frame order survives;
- an unselected bundle is rejected unless explicitly authorized;
- a conflicting refined-run argument is rejected;
- the direct legacy-layout writer rejects bundle-v5 publication;
- `latest` and `latest_complete` remain unchanged;
- the result is complete and selector-ineligible;
- the derivation and publication manifest use their bundle-bound v2
  envelopes;
- activating the same source bundle later does not invalidate the immutable
  derived run; and
- source-binding tampering fails strict reload.

The supported-profile boundary additionally proves real publication through
the unpatched canonical resolver, eye-geometry resolver, tail-kinematics source
resolver, and workflow runtime verifier. The runtime verifier binds the output
to the exact planned refined-mask or recording-bundle authority and rejects a
stale direct/consolidated metadata generation.

Historical v4 coverage remains a required regression gate. The first broad
replay exposed only two test doubles that predated the optional bundle fields;
the publisher now treats absence on those historical test plans as the v4
default without weakening real v5 plans.

Final outside-sandbox validation passed:

- 41 historical subject-shape writer and coordinate-publication tests;
- 8 recording-bundle publication tests, including the v5 materializer gate;
- Ruff, Python compilation, and `git diff --check`.

Zarr emitted its expected v3 consolidated-metadata and non-Zarr sidecar
warnings during these tests; no correctness gate failed.

The 2026-08-25 supported-profile implementation checkpoint additionally
passed 47 materialized/candidate/coordinate-publication coexistence tests, the
historical-v4 materializer-to-runtime-resolver boundary, and the real bundle-v5
writer-to-runtime-resolver-to-eye/tail boundary. The bundle-v5 boundary also
proves wrong planned bundle identity and stale consolidated metadata fail
closed. Required CI and recording-scale execution remain pending.

## Remaining Checklist

- [x] Freeze a distinct v5 logical identity and source kind.
- [x] Bind the exact recording bundle and camera/frame authorities.
- [x] Preserve the historical v4/method-v11 path.
- [x] Require the access-aware materializer for finalized v5 output.
- [x] Keep the candidate output complete, immutable, and selector-ineligible.
- [x] Admit v5 activation only through the distinct supported profile.
- [x] Preserve parent selectors and consolidate only after publication.
- [x] Add positive, inactive-source, conflicting-source, and tamper coverage.
- [ ] Publish one representative selector-ineligible recording-scale canary.
- [ ] Validate at least one empty-frame window and one multi-row frame.
- [ ] Measure object count, bytes, random/windowed latency, full traversal,
      publication time, and peak RSS.
- [x] Add Palette consumer evidence for eye-angle and tail-kinematics inputs.
- [ ] Add Crimson exact-schema and visible-overlay evidence if Crimson exposes
      these subject-shape surfaces.
- [x] Add a distinct supported selector profile while preserving candidate
      rollback and no-pointer semantics.
