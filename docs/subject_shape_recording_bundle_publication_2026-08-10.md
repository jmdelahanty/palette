# Recording-Bundle Subject-Shape Publication

Date: 2026-08-10

Status: implemented and fixture-validated as a selector-ineligible candidate.
No production selector, registry authority, physical-profile promotion, or
recording artifact changed in this checkpoint.

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
6. marks the run complete but selector-ineligible; and
7. consolidates root metadata as the final visibility step.

The low-level historical writer rejects direct bundle-v5 publication. A final
v5 run must pass through the access-aware materializer, preventing a valid v5
logical envelope from being published accidentally with the legacy physical
layout. The historical selector-activation API also rejects v5 while this
profile is unpromoted; promotion requires a later explicit contract change or
promotion receipt rather than an accidental call to the v4 activation path.

The initial physical profile remains
`subject_shape_access_aware_candidate_v1`: approximately 128 KiB inner chunks
for row-aligned arrays, 1 MiB eager chunks, and 8 MiB indexed shards. This
checkpoint adopts that existing candidate for v5 testing; it does not promote
the profile.

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

## Remaining Checklist

- [x] Freeze a distinct v5 logical identity and source kind.
- [x] Bind the exact recording bundle and camera/frame authorities.
- [x] Preserve the historical v4/method-v11 path.
- [x] Require the access-aware materializer for finalized v5 output.
- [x] Keep the first output complete, immutable, and selector-ineligible.
- [x] Reject v5 through the historical selector-activation API.
- [x] Preserve parent selectors and consolidate only after publication.
- [x] Add positive, inactive-source, conflicting-source, and tamper coverage.
- [ ] Publish one representative selector-ineligible recording-scale canary.
- [ ] Validate at least one empty-frame window and one multi-row frame.
- [ ] Measure object count, bytes, random/windowed latency, full traversal,
      publication time, and peak RSS.
- [ ] Add Palette consumer evidence for eye-angle and tail-kinematics inputs.
- [ ] Add Crimson exact-schema and visible-overlay evidence if Crimson exposes
      these subject-shape surfaces.
- [ ] Promote a selector or physical profile only through a separate recorded
      decision with rollback retained.

## Recording-Scale Acceptance Gate

`fisheye.diagnostics.benchmark_subject_shape_v5_canary` is the read-only gate
for the first recording-scale publication. It accepts only an exact named,
complete, selector-ineligible v5 child bound to an exact inactive recording
subject-mask bundle. It does not activate selectors or storage profiles.

The gate replays the canonical coordinate publication, executable storage
plan, and direct/consolidated metadata contracts. It then proves the complete
`frame_row_offsets`/`source_acquisition_frame_index` relationship, requires an
empty frame and a multi-row frame, exercises the eye-geometry and
tail-kinematics consumer boundaries, and records deterministic random-frame,
windowed-row, and full-array traversal measurements. The full traversal hashes
must equal the sealed publication manifest.

Run it only after the materializer report exists:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_shape_v5_canary \
  /path/to/recording_analysis.zarr \
  --run-name <exact-v5-run> \
  --bundle-id <exact-recording-mask-bundle> \
  --publication-report /path/to/subject_shape_v5_result.json \
  --output /path/below/.palette_benchmarks/subject_shape_v5_acceptance.json
```

The two `--allow-missing-...` options are diagnostic relaxations only. Evidence
created with either relaxation is not sufficient for promotion. Filesystem or
TensorStore tracing remains necessary for physical request/transfer telemetry,
and immutable LSF accounting remains necessary for the publication job's peak
RSS because the current materializer report does not capture it. The gate says
so explicitly rather than substituting process-read RSS for publication RSS.

Historical v4 runs remain available through their existing canonical reader
and explicit historical-inspection paths. The v5 gate neither migrates nor
relabels them.

## First Canary Prepublication Finding

The first 22-clip inference array (`153302219`) completed, but the dependent
refinement array (`153302220`) failed before any recording-level publication.
The shard writer had copied crop-row, instance, and acquisition-frame identity
but not the crop-v2 `source_crop_xywh` placement array. Refined scientific
identity correctly rejected the incomplete row inventory, and the recording
publisher and v5 materializer exited through failed dependencies. The 22
completed inference bundles remain immutable evidence; they are not eligible
for refinement or publication.

New non-authoritative shard writes now copy exact float32 `[N,4]`
`source_crop_xywh` values from the crop-v2 rowset, select the same keyed rows as
the pixel work package, and bind the resulting array digest into raw scientific
identity. The array remains a coordinate surface rather than being folded into
the generic row-lineage inventory. Historical raw-v2 identities that omit it
remain readable, but production refined publication continues to require it
and fail closed. The canary must be rerun from inference with a new immutable
run ID; the incomplete bundles must not be patched in place.
