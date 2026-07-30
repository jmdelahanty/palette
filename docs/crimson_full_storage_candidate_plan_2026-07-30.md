# Crimson Full Storage Candidate Plan

Date: 2026-07-30

Status: implementation and focused validation complete; immutable deployment,
LSF submission, and Crimson full-duration measurement remain pending.

## Why the full run is required

The existing 23,287-frame / 22,926-row candidate already proves exact schemas,
typed consumer compatibility, empty-frame behavior, multi-row lookup,
cancellation, and presentation. It is an integration fixture. It cannot expose
the object count, shard-index traffic, cache pressure, resident memory, startup,
scrubbing, or sustained traversal behavior of the 1,188,000-frame recording.

The full candidate is therefore a performance and scalability gate, not a
second logical-schema experiment.

## Published candidate surface

One successful handoff contains seven immutable selector-ineligible stores:

1. canonical detections;
2. refined detections;
3. geometry-only crop-v2;
4. raw keypoint-v2;
5. keypoint-quality-v1;
6. refined-keypoint-v2; and
7. body-frame-v1.

The package references the existing recording video. It does not copy video
or persist crop pixels in an analysis archive. The only visibility marker is
`handoff_manifest.json`, written after every receipt, run manifest, decoded
dimension, consolidated declaration, and selector state has been reopened.

The earlier full-duration canonical detection fixture is only a decoded-value
anchor. Its physical layout is useful, but it predates the current embedded
canonical `run_manifest`. The workflow therefore does not pass that older
envelope downstream. Its first job rebuilds the recording table from all 22
completed clip detection groups through the current native binder, verifies
the persisted clip keys and complete recording frame map, proves all nine
decoded arrays equal the old anchor, and writes a fresh manifest-v2
access-aware canonical store on node-local scratch. Only that current store is
used by refined detection, crop, keypoint, and final handoff gates.

## Sleepyfish pins

The first full fixture uses:

- recording: `sleepyfish_2026_05_05_17_45_30_cam2010095`;
- camera frames: `1,188,000`;
- refined/crop/keypoint rows: `1,169,010`;
- source aggregate:
  `keypoints_runs/keypoints_sleepyfish_kp_allclips_20260708_01`;
- source aggregate `zarr.json` SHA-256:
  `57bba596f9c2c76626909d99ff084dec5935e2cd829817a140a836f1b0fdfa03`;
- pose model SHA-256:
  `cce63d534a8f1491db1e2c71cb9236768c445722013dc39faeaf62a9d0a9a377`;
- detection model SHA-256:
  `b365e5da4c712e8ed347baed260915b939251036f06a91e17c96f6028dde1e1d`;
- recording frame-index SHA-256:
  `081c40df4a5a72aa3e77c4eb1c61c8edb2413ae3f8b99d5c17e6aa9c9ed5f7f5`;
- Crimson full-analysis contract commit:
  `dadd9d779f0737c9643f15e3831a7c514bf99665`;
- Crimson contract document SHA-256:
  `aa64a94de7096b6a22e53d76357a619ca92bc5296b38f0549202fd67aee36a86`.

## Historical keypoint boundary

The full recording already has completed decoded keypoint inference. Repeating
1.169 million GPU inferences would test model execution rather than storage.
The full candidate therefore uses an explicitly benchmark-only historical
aggregate adapter.

The adapter fails unless all of these checks pass:

- source group metadata and model bytes match their pinned hashes;
- source completion, selector-ineligible status, recording identity, ordered
  skeleton, and stable-key backfill are exact;
- historical and crop-v2 `instance_key` sets form a bijection;
- historical and crop-v2 frame/acquisition mappings match;
- ROI origins and sizes match for every row;
- float64-to-float32 normalization passes the v2 projection tolerance; and
- source metadata is unchanged after publication.

The four keypoint-family stores are built on bounded node-local scratch and
copied to a hidden shared sibling before a same-filesystem rename. Their
physical shapes come from `published_http_v1` byte planners, never from the
legacy aggregate's chunks. This adapter is not production-writer evidence and
cannot activate selectors.

Future production campaigns instead use the strict per-clip pixel-package and
terminal-receipt path.

## Final handoff gate

The final job requires:

- the exact clean Palette deployment commit named by the plan;
- complete, selector-ineligible, unregistered detection/crop/keypoint results;
- valid refined-detection and keypoint finalization receipts;
- exact expected frames and refined-row counts;
- direct and inline-consolidated run-manifest equality for all seven stores;
- no reported production-state changes; and
- a destination below `.palette_benchmarks`.

The handoff records server and macOS path translations, manifest and logical
digests, metadata fingerprints, file/object counts, apparent bytes, Palette
and Crimson revisions, host, and LSF job identity.

## Execution checklist

- [x] Compose the seven-store DAG and one terminal handoff.
- [x] Add standalone crop/refined archive rebinding.
- [x] Normalize the historical canonical anchor into a current native-v2,
      selector-ineligible, access-aware standalone store.
- [x] Add the benchmark-only recording aggregate adapter.
- [x] Use node-local compute then shared publication for keypoint stores.
- [x] Pin Palette and Crimson commits and source/model hashes.
- [x] Validate focused workflow, real-Zarr reopen, reconciliation, and handoff
      behavior.
- [ ] Commit and deploy this branch as a unique `/groups` worktree.
- [ ] Materialize and review `candidate_plan.json` and `lsf_plan.json`.
- [ ] Submit the full candidate through `login1-citrus-poller`.
- [ ] Require terminal `handoff_manifest.json` before giving paths to Crimson.
- [ ] Run Crimson's fresh-process full-duration startup, seek, traversal,
      physical-I/O, cache, and RSS matrix.
- [ ] Keep every output selector-ineligible regardless of benchmark outcome;
      profile or writer promotion remains a separate explicit decision.

## Validation completed

Thirty-six focused tests pass, including real Zarr v3 publication/reopen tests,
the standalone crop/refined split, vectorized multi-row reconciliation, full
plan pins, canonical-anchor equality, node-local guards, receipt rebinding,
candidate composition, and the terminal handoff. Static compilation, focused
Ruff checks, and `git diff --check` also pass.
