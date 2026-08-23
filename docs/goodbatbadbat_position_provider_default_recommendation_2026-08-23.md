# GoodBatBadBat position-provider default recommendation

Decision ID:
`goodbatbadbat_position_provider_default_recommendation_v1`

Recorded at: `2026-08-23T07:55:59Z`

Status: accepted recommendation; production selector activation remains
blocked on required CI and a separately authorized activation step.

## Decision

For the current GoodBatBadBat cohort and provider-aware chaser analytics:

1. Retain `detection_bbox_centroid.v1` as the default position provider.
2. Keep the keypoint-triad provider available as an explicitly named,
   selector-ineligible comparison provider.
3. Do not silently substitute detection positions for missing keypoint-triad
   positions. Each provider product retains its own validity and lineage.
4. Do not treat this position-provider decision as a body-frame, heading,
   keypoint-model-quality, subject-mask, or cross-protocol promotion.
5. Permit the next position-only chaser analytics canaries to name the
   detection provider explicitly. Production selector activation, registry
   authority, and integration remain separate operations.

## Bound evidence

The decision binds the immutable selector-ineligible cohort canary at:

`/groups/johnson/johnsonlab/jeremy/operations/provider_chaser_distance_comparison_cohort_canary_20260823_v1`

- Cohort artifact-manifest SHA-256:
  `120a002699d5f82d20f4158183cc0e7395d002998e8030192e079466dd22f137`
- Deterministic selection SHA-256:
  `07d350ad096c67e077ecb7aeae9197f7fa27300293a1407490317f113e5f8891`
- Evidence software commit:
  `fe1c17310547c7568cd8330b739da84efb1160e3`
- Source cohort: 84 eligible recordings, eight selected without consulting
  provider agreement or behavioral outcome.
- Selection policy:
  `earliest_latest_eligible_recording_per_arena_v1`.
- Aggregation policy:
  `recording_is_analysis_unit_no_frame_pooling_v1`.
- Camera authorities represented:
  `2010093`, `2010094`, `2010095`, and `2010096`.

The eight selected recordings are the earliest and latest eligible recording
from each arena. All 53 published evidence artifacts and the self-excluding
manifest digest were independently revalidated after publication.

## Evidence summary

Across the eight recording-level observations:

- Detection valid-position coverage ranged from `97.7545%` to `100%`, with a
  recording-level median of `99.9433%`.
- Keypoint-triad valid-position coverage ranged from `70.1481%` to `98.8929%`,
  with a recording-level median of `98.1383%`.
- Keypoint coverage was lower than detection coverage in every selected
  recording. The median deficit was `1.4856` percentage points.
- The largest deficit was `29.8519` percentage points in
  `2026-08-10T17-20-55Z_arena_4_goodbatbadbat` (`2010096`): detection coverage
  was `100%`, while keypoint coverage was `70.1481%`.
- A second early-camera case,
  `2026-08-10T17-20-55Z_arena_3_goodbatbadbat` (`2010095`), had `90.7695%`
  keypoint coverage while detection coverage was `100%`.
- Nearest-chaser identity agreement ranged from `99.7573%` to `99.9777%`,
  with a recording-level median of `99.9207%` on common valid rows.
- Median provider position separation across recordings was `6.4578 px`;
  recording-level p95 separation ranged from `10.6679 px` to `22.3700 px`.
- The maximum epoch/role median absolute chaser-distance difference was at
  most `0.1334 mm` in any selected recording. The corresponding maximum p95
  difference was at most `0.4375 mm`.

Thus the providers generally agree when both are valid, but the keypoint triad
does not yet provide sufficiently uniform coverage to replace detection as the
GoodBatBadBat default. This recommendation is driven by recording-balanced
coverage and agreement evidence, not pooled frame counts.

## Scientific caveats

- The chaser temporal relation is a
  `controller_input_provenance_proxy`. Persisted metadata does not establish
  physical display-presentation time relative to camera exposure.
- This canary did not visually validate keypoint anatomy on every invalid or
  high-disagreement frame.
- This is a bounded provider-default decision for current GoodBatBadBat
  position-only analyses. It is not an inferential cohort comparison and does
  not establish biological effect sizes.
- No provider selector, analysis Zarr, registry, or production authority was
  changed by the evidence campaign or this document.
- Required CI has not yet run for the evidence implementation. The branch and
  this recommendation are therefore not merge-ready or activation-ready.

## Re-evaluation criteria

Reconsider keypoint-triad promotion only after a new immutable evidence set
shows all of the following:

1. reviewed anatomical correctness on representative cameras and epochs;
2. improved and stable coverage, including the early `2010095` and `2010096`
   failure patterns;
3. no systematic camera-, epoch-, state-, or distance-dependent bias;
4. stable downstream occupancy, speed, acceleration, bout, and pre/post
   results under the candidate provider;
5. reproducibility from exact immutable inputs; and
6. a new timestamped decision binding the estimator, model, evidence,
   policies, and required-CI result.

## Completed next scientific evidence step

The first selector-ineligible provider-aware position-only canary is complete
at
`/groups/johnson/johnsonlab/jeremy/operations/provider_chaser_position_suite_canary_20260823_v2`
(artifact-manifest SHA-256
`02de7583e6210c00a269a19505ac1c11d4efe5706ec22bc44308507fb3221910`).
It exercises the explicit detection provider for radial position, quadrant
occupancy, near-chaser occupancy, aggressive-versus-inert comparisons, and
exact pre/training/post summaries without changing a selector or registry.

The bounded multi-camera position-suite cohort canary is now complete at:

`/groups/johnson/johnsonlab/jeremy/operations/provider_chaser_position_suite_cohort_canary_20260823_v3`

It binds the exact frozen task SHA-256
`21dd9f7079de39cac987442bf03a233fc57338b714e3c96cc080a74ca2d8da39`
and artifact-manifest SHA-256
`1bb72ffda8f2dbf932005eed9cfe491f3ee43fc82265f219245ba7aa6123148d`.
The cohort contains the earliest and latest eligible recording in each arena,
uses cameras `2010093`--`2010096`, and aggregates each metric over recordings
rather than frames. All 107 artifacts independently revalidated. The radial
cohort figure displays only bins supported by all eight recordings; sparse
tail-bin evidence remains preserved with explicit support counts in CSV.

This remains selector-ineligible operational evidence and does not change the
provider recommendation, an analysis Zarr, the registry, or a production
selector. The next separate decision is whether to implement sealed
analysis-Zarr publication and profile integration for this position-only
suite. Provider-aware motion, bouts, body-frame, gaze, trial, and escape
products remain later phases, and required CI remains unrun.
