# Parent-recording transfer-v2 intake: implementation handoff

Status authority: `INGEST-001` in the
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md).
This document records evidence and compatibility decisions, not another queue.

## Initial implementation checkpoint

- Owner: Palette root worker.
- Worktree: `/tmp/palette-parent-clipped-intake-20260906`.
- Branch: `agent/palette/parent-clipped-intake-20260906`.
- Exact base: `925f8c6285499d39a99475945ba1732dba7f6834`; all 24 required
  checks passed in [run 34056284642](https://github.com/jmdelahanty/palette/actions/runs/34056284642).
- The user authorized parent-level clipped-intake implementation and isolated
  validation after the six-PR integration. No production rollout, live cron
  change, shared-checkout update, historical repair, source cleanup, dependency
  installation, or mutation of acquisition repositories is included.
- Relevant organizer, import, clipped-collection/frame-map, receipt/identity,
  wrapper, and queue paths were checked across registered worktrees. No dirty
  overlap was found. The original dirty checkout and prior worktrees remain
  preserved. No other worker's uncommitted changes were incorporated.

## Pinned acquisition contract

- Shared schema/handoff/fixtures: [agent-contracts PR 49](https://github.com/jmdelahanty/agent-contracts/pull/49),
  exact `6cb56b4deae6a9a56bf4b3a3186168c15e91f542`.
- Citrus implementation: [PR 3](https://github.com/JohnsonLabJanelia/citrus/pull/3),
  exact `e881f5258be83231b62a00a6f9c4e5fcd69cd548`.
- Both remain draft/unmerged. The prior focused re-review passed 27 producer
  and 6 existing marker tests and closed the two refusal defects plus path
  grammar mismatch. Those results are acquisition-side evidence, not Palette
  end-to-end acceptance or production eligibility.
- Schema SHA-256:
  `65828a1e327a8cd1884c59d51f8422546a6f60d1f6abf2e3f9cd3d104dbb2df7`.
- The user subsequently approved adding/installing `jsonschema` as a test
  dependency. `jsonschema[format-nongpl]>=4.23,<5` is declared in the `dev` extra; the approved
  installation resolved to 4.26.0 in `palette-py311`. Full Draft 2020-12 schema
  tests now exercise the pinned schema and envelopes, reconstructed snapshots,
  nested closure, and refusal cases (final results recorded below).

The live installed poller is
`/home/delahantyj@hhmi.org/bin/citrus_staging_marker_poller.sh`, unchanged at
SHA-256 `6bbdcfc4e81278ddbf1b01efc3ad2af768d778c6c48079cf688a4784cabec172`.
It accepts v1 declarations and uses marker-path claim identity. Its replacement
must be tracked/tested with explicit version routing before any separately
authorized installation. A source checkout or schema addition does not update
live cron.

## Classification and preservation contract

This is a new supported ingestion-layout path plus scoped enforcement and
dispatch corrections. It is not a scientific-parameter change, historical
migration, source-ID remapping, or an excuse to weaken existing admission.

Required invariants:

- One parent identity per acquisition session and exact camera serial; clips
  and full/crop outputs remain children, including a one-clip rolling recording.
- Preserve original payload paths/bytes, source IDs, per-output local row
  order, timestamps, sidecars, original manifests, and existing digest domains.
  Source retention is unchanged; successful import does not authorize deletion.
- Snapshot identity, parent identity, transfer attempt, dispatch claim, and
  import/registry acceptance are separate claims. Changed content at the same
  path cannot reuse old success. Lost/ambiguous submission acknowledgment must
  not trigger blind duplicate submission.
- Validate exact stored snapshot bytes and closed inventory, then independently
  validate source/profile/clock/receipt requirements. Present failed optional
  proof is refusal, even where proof absence is allowed.
- The transport profile permits increasing subsets with gaps. Palette's dense
  acquisition mapping remains unchanged: transport-valid but consumer-ineligible
  inputs must refuse explicitly, not be renumbered or silently downgraded.
- Reuse the existing clipped-collection/frame-map and current source-import
  receipt owners. A metadata shell alone is not a completed source import.
  Existing single-video valid workflows and stable receipt bytes need
  preservation tests before implementation changes.
- Use isolated outputs and a temporary registry for tests. No live recordings,
  scheduler submissions, registry authority, or production selectors may change.

## First boundary implemented locally (20:43 UTC)

- Added `fisheye.shared.recording_transfer_snapshot`: read-only exact-byte
  verification and protocol-specific reconstruction from original payload
  evidence, followed by one parent plan per exact session/camera identity.
  Reconstruction adapts the pinned Citrus reference grammar; strict JSON and
  source-recording ID mapping use existing Palette owners. No live acquisition
  code or external import is needed at runtime.
- Added `fisheye.utils.inspect_recording_transfer`: a read-only CLI that reports
  `parent_plan_only`, explicitly not import completion, registry admission, or
  source-cleanup authorization.
- Preserved the shared schema and three fixture bundles byte-for-byte. Added
  golden, shared adversarial-corpus, sparse-crop/full-gap, changed-generation,
  exact-marker, failed-proof, and no-source-mutation tests before implementation.
  The failed-proof test also removes only that proof from an otherwise identical
  sibling and regenerates its transport envelope to isolate the refusal gate.
- Validation: **174 tests passed** outside the sandbox, comprising 84 new
  boundary/inspection tests and 90 existing source-identity, Citrus runner, and
  single-recording importer tests. Static compilation, import boundaries,
  FPS/keypoint-motion/tail/paradigm ratchets, file-size and Zarr-open-mode
  ratchets, observed metadata literals, contract freshness, generated census
  check after regeneration, and whitespace checks passed. The census changes
  only its scanned-module counts for the two new modules.
- No new commit, push, PR, merge, deployment, production change, installation,
  or source deletion occurred. HEAD remains the exact base above; the new
  source/tests/fixtures/handoff and queue/census edits are uncommitted.

Reproduce the focused validation:

```bash
scripts/py -m pytest -q tests/unit/fisheye/test_recording_transfer_snapshot.py tests/unit/fisheye/test_source_recording_identity_profile.py tests/unit/fisheye/test_run_citrus_session_import.py tests/unit/fisheye/test_import_recording_analysis.py
scripts/py -m fisheye.utils.inspect_recording_transfer tests/fixtures/recording_transfer_v2/rolling
```

The current planner deliberately refuses any present positive/unknown
`frame_identity_proof` until a supported semantic-proof validator is identified;
the supplied transfer contract defines a failed-proof refusal fixture, not a
positive-proof acceptance schema. This is an explicit compatibility limit, not
an invented human-review gate. No existing importer behavior was changed.

## Second local slice: bounded parent index

- Added the opt-in `fisheye.utils.build_transfer_parent_frame_index` adapter,
  reusing the existing `palette.recording_frame_index.v1` table schema and the
  transfer owner's exact CSV parser through a bounded row callback. Existing
  single-video/legacy index behavior and transport golden digest bytes remain
  unchanged. A new explicitly versioned additive source binding ties only this
  path's derived manifest/clip index to the snapshot and exact parent identity.
- Original frame IDs, timestamps, clip IDs, physical local row order, leading
  zero camera serials, and full-versus-crop roles are preserved. Crops are not
  promoted to a dense full-frame source. Uint64 values outside the existing
  signed-int64 index schema refuse explicitly. Batching is an execution/storage
  setting; logical-row equality does not promise equal physical Parquet bytes.
- Output must be an exclusively reserved fresh directory outside the source.
  Dry-run creates nothing. Source generation and output ownership are checked
  before the final success manifest. Interrupted/failed publications retain
  their evidence and refuse in-place reuse; a fresh output attempt can proceed.
  No manifest is an import receipt or authorization to delete retained sources.
- Before the schema-engine addition, **204 focused and preservation tests
  passed** outside the sandbox, including the original 174 plus 27 new index
  cases and three existing frame-index tests. Static compilation, formatting,
  import boundaries, file-size and authority/metadata ratchets passed; generated
  census outputs were refreshed through their existing owner.
- After the approved dependency installation, the full focused suite passed
  **229 tests**, including 25 full Draft 2020-12 engine checks. Installed
  versions: jsonschema 4.26.0, attrs 26.1.0, jsonschema-specifications 2025.9.1,
  referencing 0.37.0, and rpds-py 2026.6.3. No runtime import path now depends
  on jsonschema; it is a test-only dependency.
- Follow-up format enforcement: the base jsonschema install lacked its optional
  `date-time` checker. Four new regression cases first failed, then passed with
  the approved `format-nongpl` extra. Tests now require every format declared by
  the shared schema to have an active checker. Runtime marker syntax likewise
  rejects non-RFC3339 ISO spellings while preserving valid lowercase `t`/`z`
  and fractional UTC timestamps without rewriting source bytes. Final focused
  validation passed **237 tests** (29 schema-engine, 88 transport/planner,
  27 new parent-index, three legacy index, and 90 existing importer/identity).
- Real tiny encoded-media diagnostic succeeded for two cameras, each with two
  full clips of two and one frames (plus retained crop children). The new
  index passed real ffprobe and the **unpatched** clipped-collection metadata
  builder: 32x32 pixels, three parent frames, exact camera and clip identities.
  Evidence: `/tmp/palette-parent-index-encoded-20260906-akcqzjc0/smoke_report.json`.
  The common isolated test root contains the delivery and derived indexes; it
  is not an implemented parent organizer. Envelopes are synthetic Orange-format
  fixtures, not a real acquisition-producer run. This does not validate clocks,
  import-receipt publication, or registry admission.
- The user approved committing/pushing development branches. No new commit or
  push has occurred at this checkpoint; exact-commit CI remains unrun. The
  separate geometry worker branches do not enter this intake implementation.

First published development revision:
`51b5d394ccd01dbe47e91c8e8afb497e2526174e` in draft
[PR 149](https://github.com/jmdelahanty/palette/pull/149), subsequently followed
by the timestamp-format correction above. The PR head records the exact current
revision and CI; a superseded run cannot validate its successor. These remain
incomplete development changes, not a production import path.

## Remaining intake path

These local slices are not the full intake path. Actual dispatcher/submission
routing, snapshot-keyed claims and ambiguous acknowledgments, copy-preserving
parent organization, integration of the new bounded frame index, clipped import
receipt publication/replay, registry integration, and producer-retention acknowledgment
are still unimplemented. The read-only inspector is not a deployed dispatcher.

All candidate required CI is unrun: `generated artifacts`, `import boundaries`,
`file-size ratchet`, `zarr open metadata modes`, `observed metadata literals`,
`active contract freshness`, `package and collection`, `non-gpu tests (shard N)`
for every N from 0 through 15, and `ci-required`. Local analogues do not satisfy
those exact-commit required checks. The branch is incomplete and not merge-ready.

Needed evidence includes producer-to-dispatcher-to-parent-organizer-to-importer-to-unpatched
consumer/registry coverage with real encoded tiny media; partial transfer,
mutation, duplicate arrival, interrupted import, changed generation, failed
publication, ownership loss, and ambiguous acknowledgment/retry tests. Fake
media in the shared structural fixtures cannot stand in for codec or scientific
acceptance. Commit-bound producer evidence, candidate CI, integration, deployment,
and activation must be reported separately.
