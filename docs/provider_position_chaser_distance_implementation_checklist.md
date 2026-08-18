# Provider-position chaser-distance implementation checklist

Status: implementation canary; not a promoted scientific default  
Decision date: 2026-08-18

## Scientific boundary

- [x] Keep the historical detection-centroid chaser-distance method unchanged.
- [x] Treat the fish-position provider as an explicit input method, never as a
  detection rowset or detection-centroid alias.
- [x] Bind one exact
  `analysis/subject_position_runs/observation/<run>` publication by run path and
  manifest digest.
- [x] Bind one exact canonical stimulus run, stimulus-epoch run, acquisition
  frame domain, source-camera frame, camera-to-canvas transform, arena frame,
  and selected physical scale.
- [x] Require exact source-camera equality between the subject-position and
  stimulus authorities. Do not apply a reflection, heuristic Y flip, or
  resolution-ratio transform.
- [x] Preserve invalid source rows as invalid. Do not interpolate or substitute
  a detection centroid.
- [x] For the current single-subject GoodBatBadBat profile, fail closed if more
  than one valid position row maps to an acquisition frame.
- [x] Preserve the canonical `stimulus_state_key` sample axis. Multiple
  stimulus samples may legitimately map to one acquisition frame; never pick
  the first or last state as an implicit deduplication policy.
- [x] Join each stimulus sample to the fish position through the exact sealed
  `source_acquisition_frame_index`. Preserve the stimulus run row and original
  source-row lineage for every chaser.
- [x] Derive camera FPS, stimulus timestamps, stimulus identity, acquisition
  mapping, and recording extent from persisted authorities. Do not hardcode
  the observed 120 Hz stimulus or 100 fps camera rates.

The first provider is `keypoint_anatomical_triad_mean.v1`, with equal weighting
of the left eye, right eye, and swim bladder. It remains a canary provider. A
successful computation does not promote it to the GoodBatBadBat default.

## Candidate publication

- [x] Materialize an immutable candidate beneath
  `analysis/provider_chaser_distance_candidate_runs/<run>`.
- [x] Keep the candidate and its parent free of `latest`, `latest_complete`,
  `authoritative_run`, and other production selectors.
- [x] Persist stimulus-sample arrays for fish/chaser positions, validity,
  fish-to-chaser distance, nearest-chaser identity, source-position lineage,
  source-stimulus lineage, and exact source acquisition frame.
- [x] Persist pre/training/post epoch summaries and shared-bin distance
  histograms (`hist_counts`, `hist_density`, and `valid_sample_count`).
- [x] Bind every input authority and output array digest in an immutable
  candidate manifest.
- [x] Finish publication by consolidating metadata and proving direct and
  consolidated views agree.
- [x] Leave source subject-position, stimulus, epoch, detection, and motion
  runs unchanged.

## Visualization

- [x] Embed a stimulus-sample fish-to-chaser distance trace.
- [x] Embed per-epoch distance histograms using the persisted common bin edges.
- [x] Label raw sample counts and normalization. Do not silently clip finite
  values.
- [ ] Use semantic chaser behavior labels from sealed role authority when
  available. Never infer behavior from red/blue color or chaser index.
- [ ] For cohort views, provide both pooled descriptive distributions and a
  recording-balanced view with one value per recording x animal unit and
  epoch.

## Canary acceptance

- [x] Use
  `2026-08-10T17-20-55Z_arena_2_goodbatbadbat` as the first canary.
- [x] Verify its exact position manifest, stimulus run, epoch run, and camera
  authority before writing.
- [x] Confirm all valid source positions are finite and no valid acquisition
  frame is duplicated.
- [ ] Visually compare the arena-space trajectory and distance distributions
  with the recording and stimulus playback.
- [ ] Compare provider-position distances with a detection-centroid candidate
  without averaging or overwriting either method.
- [ ] Record coverage, disagreement, invalid-frame patterns, epoch stability,
  and histogram accounting in a timestamped canary decision.

Current immutable evidence:

- Preferred candidate:
  `provider_chaser_distance_keypoint_triad_canary_20260818_v2`.
- Candidate manifest:
  `102da522a6c50faa26ff06e4868244e65986d80d66c324c11e8fb7124aef929b`.
- Source position manifest:
  `3e47c00354477945b191685d8dc8dcd934f382a85b7fdf280c01f20169986d88`.
- 179,984 stimulus samples were preserved; 175,748 had a valid provider
  position and valid distance for each of two chasers.
- All 40 arrays passed manifest validation, and the 48-node direct and
  consolidated subtree declarations agree.
- The parent has no selector attributes. This remains canary evidence, not a
  promoted scientific default.
- The preserved v1 candidate has the same scientific arrays but an imprecise
  histogram y-axis label (`frames`); v2 corrects that label without rewriting
  v1.

## Downstream scope

- [ ] After a sealed provider-aware base contract exists, validate quadrant and
  radial occupancy against an equivalent-position fixture.
- [ ] Validate near-field occupancy only with its required quadrant component
  and explicit motion/immobility authority.
- [ ] Add a distance-only bout response component for onset distance, bout
  rate, duration, and peak speed by distance band.
- [ ] Defer egocentric bearing, gaze, turn-toward, circling, predicted miss,
  and heading-dependent escape analysis until the heading provider has its own
  reviewed promotion evidence.

## Promotion and campaign

- [ ] Do not submit the 84-recording production campaign from candidate success
  alone.
- [ ] Record a timestamped provider-promotion decision binding the canary
  evidence, estimator, policy version, and exact source manifests.
- [ ] Add a provider-aware sealed chaser-distance contract with generic
  `source_position_*` lineage while retaining the detection-specific v1 reader.
- [ ] Require focused tests and all required CI before selector activation or
  production publication.
- [ ] After promotion, launch the 84-recording campaign from the frozen cohort
  input manifest and publish a recording-balanced cohort export.
