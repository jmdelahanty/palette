# Registered dish geometry production implementation checklist

- Decision/checklist date: 2026-08-12
- Status: implementation started from the clean, CI-green baseline recorded
  below
- Baseline branch: `sun`
- Implementation baseline commit:
  `5e0150fcdd0f1bcd50dd63e6ca9384fb3c7409ae`
- Implementation branch:
  `agent/palette/registered-dish-geometry-production-20260813`
- Implementation worktree:
  `/tmp/palette-registered-dish-geometry-20260813`
- Baseline required CI:
  `https://github.com/jmdelahantyj/palette/actions/runs/31660860452`
  (`15/15` required jobs successful)
- Scope owner: Palette
- Producer repositories: read-only references; do not modify Orange or Citrus

## Objective

Make registered, recording-bound dish geometry a validated production input to
Palette. A configured workflow should preserve independent acquisition and
offline-fit evidence, select geometry only through an explicit versioned
policy, materialize an exact keyed centroid gate over immutable raw detections,
and require downstream detection refinement to consume that gate when the
configured policy requires it.

This work uses geometry captured and frozen at acquisition time for
post-acquisition Palette analysis gating. Live Orange/Citrus masking and
pre-YOLO masking are out of scope for the first release because their real-time
reliability has not yet been validated.

## Implementation hold

Do not begin implementation from the current checkout until all of the
following are true:

- [x] Every open question in this document has a recorded disposition.
- [x] The current `goodbatbadbat` recovery and reader-contract work has been
      reviewed and intentionally committed.
- [x] Required CI for that commit has completed successfully.
- [x] The source checkout is clean.
- [x] A new branch and isolated worktree have been created from the intended
      clean baseline.
- [x] The implementation worktree has its own exact path and commit recorded
      here or in the implementation handoff.

## Non-negotiable scientific and data invariants

### Producer authority

- [ ] Discover producer geometry only through the checksummed
      `recording_geometry_contract.json` pointer in `recording_snapshot.json`,
      or the exact verified Citrus-H5 representation under
      `/recording_geometry_contract`.
- [ ] Preserve and verify the exact producer bytes and checksums.
- [ ] Bind exactly one recording, camera serial, arena ID, native extent, and
      persisted Palette source-camera pixel-frame authority.
- [ ] Fail closed on missing, ambiguous, stale, corrupt, unsupported, or
      mismatched authority.
- [ ] Do not scan calibration directories or choose the newest calibration.
- [ ] Do not borrow geometry from another recording, camera, or arena.
- [ ] Do not substitute nominal Citrus experimental-area geometry.
- [ ] Do not infer authority from mutable current rig configuration.
- [ ] Keep the historical recovery-receipt route as an explicit compatibility
      path, not a requirement for producer-native recordings.

### Two distinct circles

- [ ] Preserve `accepted_inner_rim_boundary.geometry` as the physical,
      water-side inner-rim observation.
- [ ] Preserve `valid_detection_region.geometry` as the separate,
      outward-forgiving bounding-box-centroid gate.
- [ ] Never average, collapse, or relabel the two circles.
- [ ] Use the physical inner rim only for physical-boundary reasoning and any
      separately approved raster masking policy.
- [ ] Use the valid detection region for post-detection centroid gating.
- [ ] Add no Palette 0.5 mm tolerance to the producer gate.
- [ ] Do not use rounded `palette_dish_mask_v2.json` values as precision
      authority.
- [ ] Do not promote legacy `analysis_metadata.attrs["dish_mask"]` to modern
      selected-geometry authority.

### Coordinate contract

- [ ] Treat geometry and detections as native camera coordinates with top-left
      origin, +X right, +Y down, continuous XY pixels, and exact native width
      and height.
- [ ] Do not apply Citrus presentation reflection.
- [ ] Do not apply camera-to-canvas homography for this gate.
- [ ] Do not apply a heuristic Y flip.
- [ ] Include an asymmetric, off-center fixture that would detect any flip,
      transpose, integer rounding, or wrong-extent normalization.

### Immutability and recoverability

- [ ] Do not edit source recordings, producer geometry bundles, Citrus H5
      inputs, raw detection runs, or existing immutable analysis artifacts.
- [ ] Publish acquisition candidates, offline-fit candidates, comparisons,
      selections, and gate tables as separate immutable artifacts.
- [ ] Keep raw detections recoverable so a policy or geometry change can rerun
      comparison, selection, gating, and dependent processing without rerunning
      YOLO.
- [ ] A manual correction must publish a new candidate; it must not rewrite an
      acquisition or Palette candidate.

## Existing Palette surfaces to extend

- `fisheye.shared.recording_geometry`
- `fisheye.analysis_workflows.materializers.arena_geometry_candidates`
- `fisheye.diagnostics.probe_recording_dish_rim_fit`
- `fisheye.diagnostics.audit_arena_geometry_detection_gates`
- `fisheye.analysis_workflows.materializers.arena_geometry_selection`
- `fisheye.analysis_workflows.materializers.registered_detection_gate`
- `fisheye.cluster.arena_geometry_review`
- `fisheye.cluster.arena_geometry`
- `fisheye.refinement.refine_detect`
- `fisheye.cluster.clipped_inference`
- the whole-recording detection/analysis adapters
- the registry stage catalog and readiness extraction surfaces

Do not create a parallel mask authority or a second geometry model.

## Verified baseline and concrete gaps

- [x] Folder and exact Citrus-H5 adapters normalize producer geometry.
- [x] Full-precision physical rim and centroid gate remain separate.
- [x] Normalized geometry binds to persisted source-camera pixel authority.
- [x] Acquisition and reviewed Palette candidates are immutable.
- [x] The blind probe samples early, middle, and late keyframe windows and
      freezes its report before optional acquisition reveal.
- [x] Explicit immutable selection exists.
- [x] A keyed, full-precision, inclusive detection gate exists for whole-video
      and recording-ordered clipped detection sources.
- [x] Raw detections are preserved.
- [x] The recording-layout contract distinguishes an identity-mapped
      `WHOLE_VIDEO` target from an indexed `CLIPPED_COLLECTION`, including a
      clipped collection containing only one work unit.
- [ ] Producer-native candidate planning/publication works without a recovery
      receipt.
- [ ] Every plausible frozen offline circle is persisted, not only the winner
      and candidate count.
- [ ] An immutable canonical comparison artifact exists.
- [ ] A validated automatic-selection policy and state-to-action matrix exist.
- [ ] A newly produced clipped detection artifact is converted to canonical
      recording-bound row identity before any quality or gate consumer requires
      `instance_key`.
- [ ] The production clipped recipe accepts valid cardinalities instead of
      requiring exactly 22 clip-camera work units.
- [ ] Whole-video raw detection continues through a composed quality,
      refinement, finalization, crop, keypoint, and subject-mask recipe.
- [ ] A reviewed offline-fit candidate and the selection snapshot agree on the
      physical-boundary field contract.
- [ ] Required selection proves the configured comparison/policy artifact;
      comparison is not optional in that mode.
- [ ] The keyed gate proves exact camera and pixel-frame authority, not native
      extent alone.
- [ ] Required gate consumption is implemented in refined detection.
- [ ] Default production recipes include the configured geometry chain.
- [ ] Registry/readiness state represents each modern geometry stage.

## Recording-layout terminology and production boundary

There are two production topologies, not three:

1. `WHOLE_VIDEO` is the authoritative unsplit recording video with identity
   frame mapping. Frame zero maps directly to canonical recording frame zero.
2. `CLIPPED_COLLECTION` contains one or more derived processing partitions.
   Every work unit has an explicit indexed mapping back to the canonical parent
   recording timeline.

A clipped collection containing one work unit is not a separate production
topology and must not be used to disguise missing whole-video support. It is
useful only as a degenerate clipped-layout case, including:

- a bounded performance or smoke canary;
- reprocessing one failed partition of a larger collection, without claiming
  that the complete recording has been finalized;
- testing that clipped contracts do not assume a fixed partition count; or
- a derived full-duration video whose lineage still requires a non-identity
  frame map.

For recording-level quality or refinement, a one-work-unit clipped collection
must map the complete canonical timeline. An arbitrary short clip may test one
component but cannot establish recording-level readiness. The recent
`goodbatbadbat` recordings' authoritative full-frame camera videos are
`WHOLE_VIDEO` inputs. They must not be wrapped in artificial one-item clip
plans to bypass missing whole-video orchestration.

## Policy model

Keep these policy axes distinct.

### Geometry availability and gate-consumption mode

- Configuration key: `registered_dish_geometry.gate_requirement`.
- `off`: preserve explicitly ungated behavior.
- `if_available`: apply a fully valid, selected, current gate when available;
  persist an explicit unavailable/not-applied state otherwise.
- `required`: fail closed unless the complete configured geometry, selection,
  gate, and refinement-consumption chain is valid.

### Selection/corroboration policy

Configuration key: `registered_dish_geometry.selection_policy_id`. This is
separate from availability. The first-release policy IDs are
`manual_review_only_v1` and `corroborated_acquisition_v1`. They decide whether
acquisition geometry may become operational automatically, requires review, or
must fail. Numerical thresholds and the complete state-to-action mapping remain
open.

### Raster/pre-inference policy

This is a third, separate policy. If approved later, it uses the physical inner
rim and must not replace post-detection gate auditing. It is not implicitly
enabled by either policy above. It is out of scope for this first release; no
implementation may claim that live or pre-inference masking was applied.

## Implementation slices

### Prerequisite slice: detection-layout convergence

This prerequisite must land before registered geometry is attached to either
production recipe. Geometry must not be used to paper over incompatible raw-row
or downstream authority contracts.

- [ ] Define one canonical recording-bound raw-detection row identity consumed
      by quality, comparison audits, and the keyed gate.
- [ ] Convert artifact-local `artifact_row_id` to immutable canonical
      `instance_key` only at an evidence-bound import/finalization boundary;
      preserve the artifact-local identity as lineage.
- [ ] Ensure the clipped artifact target, canonical recording source, quality
      source, and gate source bind the same ordered rows and frame map.
- [ ] Remove the production planner's exact-22 cardinality assumption while
      preserving complete, non-overlapping canonical-frame coverage checks.
- [ ] Add a composed whole-video quality/refinement/finalization adapter over
      canonical top-level `detect_runs/<run>`.
- [ ] Define one sealed refined-detection authority per layout that downstream
      crop/cache stages can consume without reverting to raw detections.
- [ ] Preserve clipped partition lineage while making whole-video and clipped
      outputs comparable at the canonical recording level.
- [ ] Add focused fixtures for one-work-unit clipped, multi-work-unit clipped,
      and whole-video targets.
- [ ] Prove split/unsplit parity for source-row identity, frame identity,
      quality, refinement, downstream lineage, and registry projection before
      either layout is production-promoted.

### Slice 1: producer-native candidate ingestion

- [ ] Add a read-only planner for ordinary recording-folder geometry using
      `load_registered_dish_masks_from_recording_folder`.
- [ ] Add or reuse the exact Citrus-H5 adapter where the H5 is the chosen
      recording-bound authority.
- [ ] Require explicit recording, camera, and arena selection when the source
      contains multiple entries.
- [ ] Bind the selected normalized geometry to Palette's persisted acquisition
      and continuous source-camera pixel-frame authority.
- [ ] Build the same canonical acquisition-candidate record used by the
      existing publisher.
- [ ] Record exact source artifact paths, identities, byte hashes, schema
      versions, runtime-application status, and native dimensions.
- [ ] Revalidate source bytes and frame authority before and after atomic
      publication.
- [ ] Preserve the recovery-receipt planner under an explicitly named legacy
      route.
- [ ] Update the acquisition-candidate CLI to distinguish native and recovery
      sources without heuristic fallback.

### Slice 2: recording-layout binding

- [ ] Define one recording-level input contract binding recording identity,
      camera serial, arena ID, analysis Zarr, authoritative full-frame video,
      recorder summary, keyframe declaration, native dimensions, source-camera
      authority, and geometry contract digests.
- [ ] Resolve the same recording-level geometry for whole-video and clipped
      downstream layouts.
- [ ] Prove that clips never become independent geometry authorities.
- [ ] Permit acquisition candidate publication, blind fitting, and raw
      detection to run concurrently after their immutable prerequisites exist.
- [ ] Neither candidate path may select geometry as a side effect.

### Slice 3: richer independent fit evidence

- [ ] Keep acquisition geometry unavailable to fitting and candidate ranking.
- [ ] Persist all deduplicated plausible circle candidates for each early,
      middle, and late window.
- [ ] For every candidate, record geometry, angular support, radial residual or
      error, median radial gradient, and rank inputs.
- [ ] Record the selected per-window candidate and deterministic selection
      reason.
- [ ] Record consensus geometry, consensus-selection reason, and between-window
      center/radius variation.
- [ ] Bind exact source frame identities, decoded frame hashes, video identity,
      native dimensions, algorithm/configuration, software commit, and runtime
      environment.
- [ ] Freeze and hash the complete fit evidence before opening acquisition
      geometry for reveal or comparison.
- [ ] Preserve diagnostic concentric rim candidates even when they are not the
      selected consensus candidate.

### Slice 4: immutable comparison artifact

- [ ] Add a canonical `analysis/arena_geometry_comparison_runs/<run>` artifact
      family.
- [ ] Bind exact acquisition and Palette candidate IDs and record digests.
- [ ] Bind recording, camera, arena, native extent, and source-camera authority.
- [ ] Record both observed-feature classifications and semantic compatibility.
- [ ] Permit direct physical-radius comparison only for the same observed
      physical feature.
- [ ] Treat `visible_dish_top_rim_edge` versus the Orange water-side inner rim
      as semantically incompatible unless reviewed evidence explicitly proves
      correspondence.
- [ ] Record center displacement in native pixels and, when authoritative,
      dish-top-rim millimetres.
- [ ] Record signed/absolute radius difference only where semantically valid.
- [ ] Record maximum boundary separation and circle IoU or mask disagreement.
- [ ] Record edge support/residual and early/middle/late stability.
- [ ] Measure acquisition-boundary edge support in recording imagery without
      altering the frozen fit.
- [ ] Optionally bind one exact raw-detection source and record operational gate
      disagreement counts separately from physical-feature comparison.
- [ ] Record a decision status, policy ID/version, complete thresholds, and
      reason codes without selecting or mutating a candidate.

### Slice 5: versioned selection policy

- [ ] Define a new automatic-selection policy contract; do not disguise an
      automatic decision as the current reviewed-selection v1 policy.
- [ ] Derive thresholds from an approved multi-camera and repeated-recording
      canary; do not choose thresholds merely to enable automation.
- [ ] Support at least these evidence outcomes:
  - `corroborated_pass`
  - `review_required`
  - `offline_fit_failed_but_acquisition_geometry_valid`
  - `semantic_feature_incompatible`
  - `producer_geometry_invalid`
  - `coordinate_or_extent_mismatch`
  - `comparison_failed`
- [ ] Record a complete state-to-action matrix for automatic selection, review,
      fail, and any explicitly permitted uncorroborated use.
- [ ] On an approved corroborated pass, select the acquisition candidate and
      retain its operator-confirmed physical-rim and producer gate semantics.
- [ ] Require explicit reviewed selection for a Palette or manual candidate.
- [ ] Keep all candidates unchanged.
- [ ] If thresholds are not yet promotable, publish policy machinery in a
      blocking/unpromoted state and report the missing canary measurements.

### Slice 6: keyed gate and required refinement join

- [ ] Continue materializing `analysis/detection_gate_runs/<run>` against one
      exact raw detection source.
- [ ] Preserve ordered `instance_key`, dense source row identity, frame index,
      native centroid, full-precision signed distance, inclusive decision,
      rejection reason, and exact selection/candidate digests.
- [ ] Keep the normative calculation:

  ```text
  signed_distance_px = radius_px - hypot(x_native_px - cx, y_native_px - cy)
  inside = signed_distance_px >= 0
  ```

- [ ] Accept boundary points.
- [ ] Validate empty-detection sources explicitly.
- [ ] Add a fail-closed gate consumer for refined detection.
- [ ] Require exact source detection identity and identical ordered
      `instance_key` coverage.
- [ ] Reject missing, stale, partial, reordered, duplicated, or extra gate
      rows.
- [ ] Reuse the existing refinement removal-reason records for gate exclusions:
      record `outside_registered_detection_gate`, the exact source
      `instance_key`, and the gate run/digest that supplied the decision.
- [ ] Do not create a separate rejection-lineage stage or competing rejection
      artifact.
- [ ] Do not run the legacy 0.5 mm `analysis_metadata.dish_mask` expansion on a
      modern registered gate.
- [ ] Persist whether gating was required, applied, unavailable, off, or
      rejected as invalid.
- [ ] Preserve the source raw detection run unchanged.

### Slice 7: downstream authority and production recipes

- [ ] Make gated/refined detection the sealed downstream detection authority
      when registered geometry is required.
- [ ] Require crop/cache publication to bind that exact refined authority.
- [ ] Require keypoint and subject-mask stages to consume crop lineage derived
      from that authority rather than a raw-detection bypass.
- [ ] Add capability-level tests proving no required-policy recipe bypass.
- [ ] Integrate the chain into the clipped production recipe while preserving
      executable-plan parity outside the new configured policy.
- [ ] Integrate the same chain into the whole-video recipe through the shared
      canonical detection/refinement authority.
- [ ] Permit separately staged canaries, but do not claim production geometry
      completion until both production topologies pass their required tests.
- [ ] Reuse exact valid geometry/comparison/selection/gate artifacts by digest;
      do not follow mutable pointers during compute.
- [ ] Changing selected geometry should rerun gate and dependent processing,
      not raw YOLO detection.

### Slice 8: registry and readiness

- [ ] Add `recording_geometry_import` readiness evidence for producer geometry
      import.
- [ ] Add `arena_geometry_offline_fit` readiness evidence for offline fit
      completion.
- [ ] Add `arena_geometry_comparison` readiness evidence for semantic
      comparison and policy result.
- [ ] Add `arena_geometry_selection` readiness evidence for the selected
      candidate and its comparison/policy binding.
- [ ] Add `registered_detection_gate` readiness evidence for keyed gate
      completion and currentness.
- [ ] Add `registered_detection_gate_consumption` readiness evidence extracted
      from the finalized refined-detection authority.
- [ ] Represent `review_required` as comparison/selection state, not as an
      independent stage.
- [ ] Add distinct readiness evidence for downstream completion from the gated
      authority.
- [ ] Do not reinterpret the legacy tuning-stage `dish_mask=ok` as this chain.
- [ ] Define migrations and compatibility projections only after the modern
      artifact contracts are stable.

## Required validation coverage

### Producer and coordinate validation

- [ ] Folder and exact Citrus-H5 forms normalize to equivalent geometry.
- [ ] Corrupt contract, pointer, manifest, observation, scope, runtime, or H5
      checksum fails closed.
- [ ] Wrong camera, arena, registration, native extent, or coordinate authority
      fails closed.
- [ ] Physical rim and outward gate remain separate at full precision.
- [ ] No added tolerance is applied; inclusive containment is exact.
- [ ] Asymmetric off-center geometry proves no flip, reflection, transpose, or
      integer rounding.
- [ ] Legacy missing geometry remains explicit; no mask is guessed.

### Fit, comparison, and policy validation

- [ ] Fit evidence is frozen before acquisition reveal.
- [ ] Multiple plausible concentric candidates are retained.
- [ ] Candidate ranking and consensus are deterministic.
- [ ] Same-feature comparison differs from semantic incompatibility.
- [ ] Comparison binds exact candidates and cannot mutate either one.
- [ ] Automatic pass/review/fail outcomes bind an explicit policy version.
- [ ] Every policy outcome maps to an explicit workflow action.
- [ ] Unpromoted thresholds cannot select geometry.

### Gate and downstream validation

- [ ] Gate rows exactly match raw detection keys and order.
- [ ] Empty-detection recordings produce a valid complete empty gate.
- [ ] Missing, stale, partial, reordered, duplicated, or extra rows fail closed.
- [ ] Rejected detections remain recoverable in unchanged raw detection runs.
- [ ] `off`, `if_available`, and `required` behavior is explicit and tested.
- [ ] Modern registered gating does not use the legacy 0.5 mm tolerance.
- [ ] Whole-video and clipped layouts resolve the same recording-level geometry.
- [ ] A one-work-unit clipped canary retains indexed parent-frame lineage and
      cannot claim recording completion unless it covers the full timeline.
- [ ] A whole-video target retains identity mapping and is never silently
      reclassified as a one-work-unit clipped collection.
- [ ] Whole-video and split clipped processing produce equivalent canonical
      row/frame identity and downstream authority on a bounded parity canary.
- [ ] Required-policy crops, keypoints, and masks cannot bypass gated
      refinement.
- [ ] Existing immutable v5, legacy, detection, and analysis artifacts remain
      byte-for-byte unchanged.

### Repository validation and delivery

- [ ] Use `scripts/py` for every Python command.
- [ ] Run focused and integration Zarr tests outside the Codex sandbox.
- [ ] Run static compilation and `git diff --check`.
- [ ] Run all required CI checks and record exact results.
- [ ] Do not describe the branch as complete or merge-ready until required CI
      is green.
- [ ] Record exact implementation commit, worktree path, canary artifacts, and
      any deferred validation.

## Canary measurements required before threshold promotion

- [ ] Select the approved multi-camera and repeated-recording cohort.
- [ ] Freeze cohort membership, recording IDs, camera/arena IDs, and source
      artifact hashes.
- [ ] Measure same-feature center displacement and radius disagreement only
      where semantic correspondence is established.
- [ ] Measure edge support and residual distributions by temporal window.
- [ ] Measure within-recording early/middle/late variation.
- [ ] Measure repeated-recording and cross-camera variation.
- [ ] Measure acquisition-boundary support in recording imagery.
- [ ] Measure operational gate disagreement on exact detection rowsets,
      including boundary-nearest examples.
- [ ] Document failure and semantically incompatible cases.
- [ ] Review the proposed thresholds and false-pass/false-review consequences.
- [ ] Publish a versioned policy evidence/decision document before enabling
      automatic selection in a default recipe.

## Open questions and decision log

Resolve these one at a time. Record the selected answer, rationale, decision
date, and any resulting checklist changes.

### OQ1: first-release runtime scope

Does “use registered dish masks during acquisition” mean:

1. use geometry captured at acquisition time in Palette's post-acquisition
   analysis gate; or
2. also change live Orange/Citrus acquisition or pre-YOLO inference behavior?

Recommended initial boundary: option 1. Keep Orange/Citrus unchanged and defer
pre-inference masking as a separately reviewed policy.

Decision (2026-08-12): option 1. The first release consumes acquisition-frozen
geometry only in Palette's post-acquisition analysis gate. Do not modify Orange
or Citrus, enable live masking, or add pre-YOLO masking. Real-time masking must
undergo a separate reliability validation and explicit rollout decision before
it enters scope.

Rationale: live Orange/Citrus masking has not yet been validated as reliable in
real time and has therefore not been used operationally.

Checklist effect: Slice 6 post-detection gating remains in scope. Optional
pre-inference masking and any acquisition-path changes are deferred and cannot
be used as completion criteria for this implementation.

### OQ2: policy-axis names and configuration surface

What exact configuration names represent geometry availability/gate
consumption, selection/corroboration, and optional raster masking?

Recommended direction: two independent versioned settings in the first
release, with a separately introduced third setting only if pre-inference
masking is later validated. Do not couple any setting through legacy
`dish_mask` configuration.

Decision (2026-08-12): expose two settings in the first release:

```yaml
registered_dish_geometry:
  gate_requirement: off | if_available | required
  selection_policy_id: manual_review_only_v1 | corroborated_acquisition_v1
```

Do not add a pre-inference masking setting in this release. Add that as a third,
independent policy only after the capability and its real-time reliability have
been validated. Neither first-release setting may be inferred from or coupled
to legacy `dish_mask` configuration.

Rationale: availability/consumption and evidence-based selection are different
decisions. A dormant pre-inference option would imply a capability that this
release intentionally does not provide.

Checklist effect: configuration, provenance, tests, recipe plans, and registry
readiness must record both first-release values exactly. Pre-inference masking
is not a supported value or completion path.

### OQ3: uncorroborated but valid acquisition geometry

What action should `offline_fit_failed_but_acquisition_geometry_valid` take
under each configured policy: automatic use, review barrier, or failure?

Recommended default: review barrier. Any permitted uncorroborated use should be
an explicitly named policy, never the default corroborated policy.

Decision (2026-08-12): never select or apply acquisition geometry when the
independent offline fit failed. Do not implement an acquisition-only fallback
policy.

The state-to-action behavior is:

- `gate_requirement: off`: continue explicitly ungated; geometry validation is
  not an operational prerequisite.
- `gate_requirement: if_available`: classify geometry as unavailable for
  operational use, continue explicitly ungated, persist
  `offline_fit_failed_but_acquisition_geometry_valid`, and create a
  review-required disposition.
- `gate_requirement: required`: stop before gate-consuming refinement and
  require investigation/review. The acquisition candidate remains preserved
  evidence but is not operationally selectable from this outcome.

Neither `manual_review_only_v1` nor `corroborated_acquisition_v1` may treat fit
failure as corroboration. A later reviewed workflow may diagnose the failure
and publish new valid offline or manual evidence, but it must not waive the
missing independent evidence by selecting the acquisition candidate alone.

Rationale: inability to start, complete, or review the independent fit likely
indicates a larger recording, imagery, layout, decoding, or geometry problem.
Using acquisition geometry as a fallback would hide that failure at the point
where the workflow is intended to detect it.

Checklist effect: do not add an acquisition-only policy ID. Tests must prove
that fit failure never creates an automatic or manual acquisition-only
selection path, and that `if_available` reports an explicit ungated result
while `required` fails closed.

### OQ4: semantic feature incompatibility

Can any evidence automatically establish correspondence between a visible top
rim and the water-side physical inner rim, or is reviewed classification always
required when those features differ?

Recommended default: require review; operational gate disagreement may still
be reported automatically without claiming physical correspondence.

Decision (2026-08-12): never automatically equate
`visible_dish_top_rim_edge` with Orange's
`dish_inner_rim_water_side_edge`. Geometric proximity, concentricity, or a
small radius difference is not proof that two fits observed the same physical
feature. A direct physical-boundary comparison requires reviewed same-feature
classification. A future validated feature classifier would require a new
policy version before it could replace that review.

Operational gate disagreement may still be measured automatically without
claiming physical-feature equivalence.

Practical qualification: the operator reports that consistently targeting the
top edge versus the inner water-side edge has been difficult despite intending
to fit the latter. Palette must therefore preserve the producer-declared target
without treating that declaration as independent proof of the visually
observed edge. Offline fitting should preserve the stable family of plausible
concentric edges rather than forcing one candidate to inherit the requested
feature label.

Rationale: radius disagreement between different concentric rim edges may
reflect feature choice rather than acquisition or registration error. Treating
it as ordinary fit error would create false failures and could encourage an
incorrect averaged boundary.

Checklist effect: comparison must expose semantic certainty/ambiguity and keep
physical same-feature metrics separate from center, rim-family, image-support,
and operational gate-disagreement evidence. Exact semantic equivalence is not
required merely to compute those non-semantic measurements. Whether an
operationally corroborated but semantically indeterminate acquisition gate may
be automatically selected remains a separate decision below.

The comparison vocabulary must distinguish:

- `same_feature_confirmed`: reviewed evidence supports the same physical edge;
- `different_feature_confirmed`: reviewed evidence supports different physical
  edges; and
- `projected_edges_unresolved`: the physical edge semantics remain distinct,
  but camera pose, projection, visibility, or image resolution prevents a
  reliable assignment of the observed contours.

The third state is not a fit failure and must not be silently converted to
either of the first two states.

### OQ4a: operational corroboration under edge ambiguity

May a future promoted policy automatically select the acquisition centroid gate
when exact rim-edge correspondence is indeterminate, provided independent
evidence shows a stable concentric rim family, compatible center, sufficient
image support, valid producer provenance, and acceptably small operational gate
disagreement on the exact detection rowset?

Recommended direction: allow this only as a distinct, explicitly named policy
outcome that makes no same-feature physical-boundary claim. Derive every
threshold from the frozen canary and require review until that evidence is
promoted. Do not average radii or alter the acquisition gate.

Decision (2026-08-12): yes. A future promoted policy may automatically select
the unchanged acquisition centroid gate when feature correspondence is
`projected_edges_unresolved`, provided independently frozen evidence satisfies
the promoted thresholds for a stable rim family, compatible center, image
support, valid producer provenance, coordinate/extent identity, and acceptable
operational gate disagreement on the exact detection rowset.

This outcome must not claim that the visible top edge and water-side inner rim
are the same physical feature. It must omit same-feature radius-error metrics,
record `projected_edges_unresolved`, and preserve all candidate circles. It
must not average radii or modify the acquisition gate.

Rationale: with a camera tilted relative to the dish, physically distinct rim
edges can project to separable or inconsistently visible contours. With a
camera centered and perpendicular to the dish plane, those contours may become
effectively coincident at the available image resolution. The policy should
represent that observational limitation rather than demand a false semantic
classification.

Rollout constraint: this automatic outcome remains review-only until its
thresholds and failure behavior have been derived from the frozen canary,
reviewed, versioned, and explicitly promoted.

Checklist effect: add `projected_edges_unresolved` to the comparison contract
and policy tests. Permit operational corroboration from non-radius evidence in
that state, but prohibit direct physical-radius error claims and automatic use
before policy promotion.

### OQ5: policy threshold promotion cohort

Which recordings and cameras constitute the bounded canary, and how much
repeated-recording evidence is required before thresholds can become active?

Recommended direction: freeze the cohort before examining proposed thresholds
and include all four cameras plus repeated sessions and known difficult cases.

Decision (2026-08-12): freeze the existing modern producer-native cohort before
examining fitted comparison metrics. Use the 2026-08-10 `goodbatbadbat`
recordings as the threshold-derivation cohort and the 2026-08-11
`goodbatbadbat` recordings as a locked holdout:

- 2026-08-10: 36 recordings, all four cameras, and three independent
  registration snapshots;
- 2026-08-11: 28 recordings, all four cameras, and three different independent
  registration snapshots; and
- report independence primarily as 24 camera-registration cells, not as 64
  independent recordings. Repeated recordings within a registration cell
  measure temporal/recording variability but do not constitute new independent
  registrations.

Use the 16 July 22 Batman recordings only as a historical-recovery
compatibility challenge because their producer association is supplied by an
approved recovery receipt. Use the July 29 and July 30 contracts whose status
is `not_configured` as negative/fail-closed cases rather than geometry evidence.
Use August 6 clips only to validate whole-recording versus clipped-layout
equivalence; clips copied from one parent recording are not independent canary
observations.

Promotion requires the unchanged candidate policy to produce zero false
automatic passes against operator-adjudicated review across the complete August
11 holdout, pass every real and injected fail-closed control, and report results
separately by camera and registration. Thresholds and policy configuration must
remain frozen before holdout evaluation. If any threshold is changed after
holdout inspection, mint a new policy version and reserve a prospectively
frozen incoming dataset as its untouched holdout.

Rationale: the six registration snapshots exercise registration changes while
the repeated recordings within them exercise recording-time variation. The
locked split prevents the same observations from both choosing and validating
thresholds. This bounded canary can support an initially monitored policy; it
does not establish a population-wide failure-rate guarantee.

Checklist effect: publish a checksummed canary manifest before fitting, record
cohort role and registration-cell identity in every result, prohibit clip-level
pseudoreplication, and fail policy promotion if the holdout is incomplete,
unreviewed, or used to tune the candidate thresholds.

### OQ6: implementation order by recording layout

Should required gate consumption ship first for the mature clipped
postprocessing recipe, then add whole-video quality/refinement, or must both
layouts become end-to-end ready in the same change?

Original recommended direction: clipped first, followed by a separately
validated whole-video source adapter; keep shared artifact contracts
layout-neutral.

Decision (2026-08-12): revise the premise. The current clipped production path
is not a sufficient mature foundation: it hard-requires 22 work units, newly
produced artifact-local rows do not yet satisfy the canonical `instance_key`
contract required by recording-level quality/gating, and its finalized
collection remains a topology-specific compatibility authority. Whole-video
raw detection is available, but its composed quality/refinement/downstream
recipe is absent.

Implement the detection-layout convergence prerequisite first. Then attach the
shared geometry chain to clipped and whole-video production adapters in
separately testable slices on the same implementation branch. A clipped canary
may run before the whole-video canary, but the work must not be declared
production-complete and `required` must not be promoted until both production
topologies satisfy their end-to-end contracts.

Do not create a third one-clip production workflow. Retain a one-work-unit
clipped case for bounded canaries, partition recovery, full-duration derived
videos with indexed lineage, and cardinality tests. It cannot stand in for
`WHOLE_VIDEO`, and a partial clip cannot claim recording-level completion.

Checklist effect: add the prerequisite convergence slice above; test one-unit
clipped, multi-unit clipped, and whole-video layouts; remove fixed partition
count assumptions; and require split/unsplit canonical-lineage parity before
promotion.

### OQ7: refinement integration point

Should the keyed gate be joined as a first-class refinement input before
quality filtering, or projected into a new quality-label artifact consumed by
refinement?

Recommended direction: a first-class, exact keyed refinement input with its own
lineage, while preserving quality labels as an independent evidence source.

Decision (2026-08-12): consume the keyed geometry gate as a first-class,
immutable input to the existing refinement stage. Raw detections feed quality
measurement and gate materialization independently over the same complete
ordered rowset. The quality artifact remains geometry-independent and must not
be rewritten or relabeled to encode gate acceptance.

Before applying refinement selection, validate that raw detections, quality
evidence, and the gate bind the exact expected raw source and ordered
`instance_key` coverage. Apply the gate before per-frame ranking or top-k
selection so an outside-gate row cannot occupy a selected slot.

Reuse the existing refinement removal-reason machinery. Add
`outside_registered_detection_gate` and bind that reason to the source
`instance_key` and exact gate run/digest. This is an extension of existing
reason/provenance records, not a new rejection-lineage component, workflow
stage, or competing artifact. Preserve rejected rows in the unchanged raw
detection source.

Checklist effect: keep quality and geometry as separate immutable evidence,
make their exact keyed join mandatory under `required`, and test independent
and simultaneous quality/geometry rejection without imposing a new reason
precedence that loses either cause.

### OQ8: downstream no-bypass enforcement

Which artifact should become the sole authority consumed by crop/cache stages
under `required`: the refined-detection run, a finalized detection collection,
or a new sealed handoff binding both?

Recommended direction: use the existing finalized/refined authority boundary
where possible and add an exact gate-consumption binding rather than teaching
every downstream stage to read the gate independently.

Decision (2026-08-12): the sole downstream detection authority under
`required` is the finalized refined-detection product. Extend the existing
layout-specific finalized authorities rather than creating a third handoff
artifact:

- whole-video uses one immutable `refined_detect_runs/<run>` authority; and
- clipped processing uses one finalized immutable collection that enumerates
  the exact per-clip refined runs in canonical recording order.

Both forms must implement the same gate-consumption binding, including the
exact raw-detection source identity, gate run and digest, selected geometry and
policy identities, ordered source-key coverage, and accepted/rejected counts.
The binding must state whether gating was required, applied, unavailable, or
off according to the configured policy.

Under `required`, crop/cache publication accepts only a complete finalized
refined authority proving valid gate consumption. It must reject raw-detection
sources, ungated refined runs, incomplete collections, stale bindings, and
layout-specific aliases that do not prove the same authority. Keypoint and
subject-mask stages inherit this guarantee through the exact crop/cache
lineage; they do not independently reread or reinterpret the gate.

Rationale: the layout-specific packaging remains useful, but it represents one
conceptual downstream authority: finalized refined detections. Avoiding a new
handoff artifact removes another selector and staleness surface while the
gate-consumption binding prevents downstream bypass.

Checklist effect: version the existing whole-video and clipped finalized
authority contracts with the common binding, make crop/cache validate it, and
test that every raw or ungated bypass fails closed under `required`.

### OQ9: registry stage names and migration timing

What stable stage IDs should represent geometry import, comparison/selection,
gate materialization, and gate consumption, and should registry migration ship
with initial refinement integration or after artifact contracts stabilize?

Recommended direction: stabilize artifacts first, then add explicit modern
stages without changing the meaning of legacy `dish_mask`.

Decision (2026-08-12): use these stable registry stage IDs:

- `recording_geometry_import`
- `arena_geometry_offline_fit`
- `arena_geometry_comparison`
- `arena_geometry_selection`
- `registered_detection_gate`
- `registered_detection_gate_consumption`

`review_required` is a state emitted by comparison/selection, not an
independent stage. Gate consumption is extracted from the finalized
refined-detection authority and must not be marked complete merely because a
gate materialization job ran.

Stabilize and test the immutable artifact contracts first, then add the stage
catalog, registry extraction, views/migration, and readiness tests on the same
implementation branch before production activation. This ordering permits the
registry to reflect final versioned contracts without allowing activation to
precede operator-visible status.

Never infer, alias, or backfill any of these stages from legacy
`dish_mask=ok`. Historical recordings retain the state justified by their own
evidence, such as legacy missing, unassessed, explicitly off, review required,
failed, or complete.

Checklist effect: add the six stages and their dependency/currentness rules,
test status extraction from real artifact states, and require all applicable
modern statuses before declaring a workload geometry-ready.

### OQ10: default activation and rollout

Should newly conforming recordings default to `required` immediately after
policy promotion, or begin with a shadow/`if_available` canary before default
activation?

Recommended direction: shadow or `if_available` with complete discrepancy
reporting, followed by explicit promotion to `required` after canary review.

Decision (2026-08-12): use a staged, explicitly activated rollout:

1. Run the complete geometry chain in shadow mode on the frozen canary. Record
   the gate and every would-be exclusion, but do not change production refined
   detections, selectors, or downstream authorities.
2. Run an `if_available` pilot on an explicitly named incoming cohort. Apply
   only complete, current, corroborated gates. If the chain is unavailable,
   continue ungated with the exact unavailable/review/failed state persisted.
   Do not pool gated and ungated results without retaining and exposing that
   distinction.
3. Permit `required` as an explicit opt-in only after both recording layouts
   pass end-to-end validation, the frozen holdout satisfies its promotion
   criteria, all required CI is green, and operator review accepts the pilot.
4. Make `required` the default for new conforming recordings only through a
   separate timestamped activation decision. Promoting a policy version does
   not itself alter configuration defaults or production selectors.
5. Preserve the configured state of existing and legacy recordings. Do not
   automatically reprocess, select geometry for, or reclassify them.

This rollout applies only to post-acquisition Palette centroid gating. It does
not activate, validate, or make claims about live Orange/Citrus masking or
pre-inference raster masking.

Rationale: policy validity and operational activation are different decisions.
Shadow and bounded pilot evidence expose failure behavior and mixed-availability
cases before a fail-closed requirement can stop production workloads.

Checklist effect: implement explicit shadow, pilot, opt-in-required, and
default-activation states; require timestamped activation provenance; and test
that policy publication alone cannot change selectors or defaults.

## Final implementation handoff requirements

- [ ] Findings by severity with file and line references.
- [ ] Exact schema IDs and versions for new artifacts and policies.
- [ ] Exact commands and test results.
- [ ] Operator-review handoff for non-automatic outcomes.
- [ ] Canary measurements still required, if any.
- [ ] Confirmation that producer geometry, recordings, raw detections, and
      existing immutable artifacts were not rewritten.
- [ ] Confirmation of which recipes and policies are active versus implemented
      but intentionally unpromoted.
