# Refined Keypoint v2 Crimson Fixture Contract

Date: 2026-07-29

Status: Palette fixture complete; Crimson exact-reader gate pending

## Purpose

This immutable follow-on to the raw-keypoint handoff exercises the complete
future-facing chain:

```text
raw keypoints v2
  + keypoint quality v1
  + explicit decisions keyed by instance_key
  → refined keypoints v2
  → body frame v1
```

The package is benchmark-only, selector-ineligible, registry-unregistered, and
contains three explicitly synthetic review decisions. It is consumer and
publication evidence, not scientific review authority, production selection,
or a delta-compaction fixture.

## Immutable Handoff

Server package:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
keypoint_storage/integration/
20260128_cropv2_keypoint_refined_v2_20260729_v2/
```

Mounted macOS package:

```text
/Volumes/johnsonlab/jeremy/recordings/.palette_benchmarks/
keypoint_storage/integration/
20260128_cropv2_keypoint_refined_v2_20260729_v2/
```

| Artifact | Explicit path | Manifest digest | Logical-content digest |
| --- | --- | --- | --- |
| Refined keypoints | `refined.zarr/refined_keypoints_runs/refined_keypoints_crop_v2_synthetic_canary_v2` | `2f594135b69987c62591d9caeb2d12430a9bc5b2f639cbb42388d2a746728dc0` | `21edfe828af62029b47f4e5caceb9b1c526486ff4daf9d5f4aecaac6214168e3` |
| Refined-derived body frame | `body_frame.zarr/analysis/body_frame_runs/body_frame_refined_keypoints_canary_v1` | `857f7c6bd912f19a3b11bdfe8e3fb9bb68ba563aa9f0da6635947b9f324ff16b` | `12604e2e959781e48fdb4cf0034514738465e51be7b8e7f7f4d9dd677dd1232c` |

The package handoff is `handoff_manifest.json`; its SHA-256 is
`d1c0e27303b715c95c645f077406906f691be2f9d86a74307425bb55465606b1`.
Publication ran from clean Palette commit
`274da1839c7f70b2fa39c5a0eec06b5603b3211d` on branch
`agent/palette/crop-storage-publication-integration-20260729`.

The exact sources remain the raw, quality, and crop paths in
`docs/keypoint_v2_crimson_fixture_contract.md`. Their metadata fingerprints
were identical before and after publication. No selector, registry, training
artifact, or source archive changed.

## Persisted Contract

The refined snapshot contains the exact 23 arrays declared by
`palette.stage.refined_keypoints` v2. It retains all shared raw identity,
lineage, coordinate, confidence, validity, bbox, and signature arrays; replaces
`pose_success` with `source_success` and `refined_success`; and adds:

```text
keypoint_edit_flags   bool[N,K]
flip_corrected        bool[N]
confidence_valid      bool[N]
geometry_valid        bool[N]
usable_keypoints      bool[N]
review_state_codes    uint8[N]
reason_codes          uint16[N]
```

There are 23,287 frames, 22,926 rows, three landmarks, 361 empty frames,
unique stable `instance_key` values, and an exact 23,288-element
`frame_row_offsets`. Refined rows preserve raw row order. `refined_success`
exactly marks rows with at least one valid final landmark, while
`usable_keypoints` exactly equals refined success plus the accepted confidence
and geometry gates.

The manifest binds the complete raw, quality, and crop manifests; recording,
skeleton, coordinate-catalog, and source-row identities; canonical
review/reason registries; initial snapshot/lineage identity; retired-key
evidence; reconstructed storage plans; every decoded array digest; and exact
direct/consolidated metadata. Heading remains forbidden in raw and refined
keypoints and is present only in the separately bound body-frame run.

At this bounded size all 23 refined arrays and all ten body-frame arrays are
single unsharded chunks using `bytes → zstd(level=0)`. Refined keypoints use 23
payload and 48 incremental stage objects; body frame uses ten payload and 22
incremental stage objects. This fixture does not select the long-recording
physical profile.

## Synthetic Decisions

The handoff records three exact decisions keyed by `instance_key`:

1. one accepted 0.25-pixel ROI-coordinate correction;
2. one rejected successful source row, cleared in the refined snapshot; and
3. one manually recovered failed source row with three finite benchmark-only
   landmarks.

These cases prove identity-safe correction, rejection, and recovery. They are
not claims about the scientific correctness of those observations. All other
rows are a deterministic no-op refinement of raw values plus the accepted
quality gates.

## Crimson Gate

Crimson should extend its backend-independent exact-schema keypoint reader to:

1. open the explicit refined run without dtype probing or legacy aliases;
2. validate its exact 23 declarations, manifest, source bindings, code maps,
   and direct/consolidated equivalence;
3. retain refined offsets once and preserve the full frame row range;
4. carry `instance_key`, source/refined success, edit flags, accepted QC,
   review state, and reason code through inspection/presentation;
5. prove the correction, rejection, and recovery resolve by stable identity,
   not within-frame ordinal;
6. open the explicit body-frame run only after validating that it binds this
   exact refined manifest and row-signature digest;
7. prove ordinary playback performs zero keypoint-quality payload reads; and
8. fail an invalid explicit refined selection without silently returning to raw
   or a legacy refined run.

The headless workload should repeat the retained-offset, random-frame,
70-frame-window, cancellation, stale-publication, full-traversal digest, file
I/O, cache, deadline, and RSS measurements from the raw fixture. One GUI smoke
should verify that a rejected row disappears, a recovered row renders, and the
refined-derived heading is used.

Negative tests should cover wrong/missing arrays, dtype/rank changes, malformed
offsets, duplicate keys, source-fact changes, live/retired key overlap,
registry tampering with a recomputed outer digest, source-manifest mismatch,
embedded refined heading, and a body frame bound to the raw rather than refined
snapshot.

This accepts the initial immutable snapshot boundary only. Parent/successor
identity, append-only edit deltas, and compaction remain the next Palette
contract checkpoint and must receive their own canary before manual editing is
activated.
