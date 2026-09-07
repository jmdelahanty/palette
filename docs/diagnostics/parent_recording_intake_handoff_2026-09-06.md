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
  Original relative paths and bytes remain evidence; relocation into organized
  storage must bind their new destinations explicitly. Acquisition-machine
  retention and Palette staging are different lifecycles. The staging requirement
  was clarified by the user below: no payload is left there after successful intake.
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
routing, snapshot-keyed claims and ambiguous acknowledgments, inventory-complete
parent organization with staging finalization, integration of the new bounded frame index, clipped import
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

## User clarification: staging is temporary, not a retained source archive

At the September 6 parallel-development follow-up, the user explicitly clarified
that no copies should remain in staging after successful intake and that components
not transferred into the recording are unintended behavior. This supersedes the
earlier proposed copy-only preparation stage and ambiguous references to
"copy-preserving" organization. That proposal was stopped before implementation;
its new draft test was withdrawn. No live source or staging files were changed.

The existing organizer is not globally copy-only: `PlannedFile.action` defaults
to `move`, `_apply_plan` dispatches `shutil.move` versus `shutil.copy2`, and shared
context can be explicitly copied. Its current selected-file plans do not establish
that every member of a closed transfer inventory reaches organized storage. The
new path must not declare successful organization from selected-file counts alone.

The implementation contract is now:

- Account for **every** transfer-inventory member plus the original snapshot and
  completion-marker evidence. Bind explicit organized destinations and ownership;
  never omit an unfamiliar component or discard it as incidental staging debris.
- Keep one parent per exact session/camera, with full/crop streams and all clips
  under that identity. Shared session context must have explicit durable placement
  and bindings for all consuming parents; per-camera H5 assignment must use the
  existing exact context/identity contract, not filename guesses.
- Move staging payloads into organized storage as the successful lifecycle result.
  A same-filesystem relocation or a verified cross-filesystem copy-then-retire
  transaction may implement this; a permanently retained duplicate staging bundle
  is not the result. Do not add an extra full-bundle preparation copy by default.
- Verify complete destination coverage and bytes, parent identity, and the
  applicable intake gates before finalizing removal of staging payloads. Preserve
  original producer evidence without rewriting it to impersonate a new transfer.
- Interrupted, conflicting, or failed attempts must remain recoverable and must
  not report success or blindly remove residual files. Resume/finalization needs
  exact generation and destination-ownership checks, not blanket directory cleanup.
- Successful intake leaves no payload behind in staging. This does not itself
  change acquisition-machine retention, nor authorize a live historical cleanup,
  production rollout, or mutation of the installed poller.

Root remains the organizer/interface and queue owner. The next implementation
slice must establish complete destination-accounting and transaction/retry tests
before mutation. Dispatcher, organizer, importer/registry, and staging finalization
remain unimplemented; the two existing validator/index slices are not completion.

At this clarification checkpoint, worktree HEAD is
`0cd250a58b26e5c56fcdf9926b2593eee643b1e9`; only this handoff and the owning queue
are modified locally. The proposed copy-only implementation is absent. PR 149
has 14 successful required checks; non-GPU shards 1, 2, 3, 6, 8, 10, 11, 12, and
15 are still running, and `ci-required` is unrun. No failures are reported at the
21:51 UTC check. The uncommitted documentation clarification has no exact-commit
CI yet. Neither checkpoint authorizes integration or deployment.

## September 7 implementation: inventory-complete opt-in workflow

This checkpoint supersedes the earlier unimplemented organizer/finalization
status above; it does not claim the combined intake path is validated yet.
Root owns worktree `/tmp/palette-parent-clipped-intake-20260906`, branch
`agent/palette/parent-clipped-intake-20260906`, based on
`0cd250a58b26e5c56fcdf9926b2593eee643b1e9`. All 24 required checks passed for
that base; the changes described here are initially uncommitted with new-head
CI unrun. Original-workspace changes and other workers' branches are preserved.

Classification: additive supported intake layout plus enforcement corrections.
Legacy organizer/import invocation defaults, exact session/camera identity,
transport/snapshot golden serialization and existing receipt-v2 grammar remain
unchanged. New organization-plan, source-mapping and projected-clip schemas
explicitly describe Palette derivations; they never impersonate Orange evidence.

Implemented here:

- `organize_transfer_recordings.py` assigns all inventoried payloads and all
  regular transfer-namespace controls to explicit recording destinations. This
  includes Citrus's empty `transfer.lock`, which is outside its scientific
  payload inventory, plus unfamiliar context. Preserved control bytes are
  historical context, not a new scientific authority or active producer lease.
- Same-filesystem hardlinks or bounded cross-device verified copies materialize
  each exact parent. Shared context fans out to all parents. H5 files bind by
  exact camera/session context. Geometry assets use their existing validator
  and canonical bundle layout. An original root PTP summary is byte-preserved
  at the clock owner's established `raw/ptp_sync_summary.json` locator.
- The existing bounded index adapter accepts an explicitly validated
  organization mapping, preserving actual full/crop roles, original frame and
  timestamp values, and confined absolute index-row path grammar. Original
  producer manifests remain unchanged; derived clip projections have their
  own versioned Palette schema and explicit `output_kind`.
- The maintained session runner has an explicit `--transfer-v2` branch with
  mandatory existing scientific-context vocabulary. It calls the ordinary
  batch importer, requires exact successful acknowledgments, and then rechecks
  actual parent receipts, requested stimulus completion and optional registry
  admission. Logs, snapshots and organizer state cannot substitute for admission.
- Staging retirement uses an external digest-bound journal and exact file and
  directory ownership. All parent payloads and import products are fsynced,
  directory entries are fsynced bottom-up, and the initial retiring journal is
  durable before the first source unlink. Every subsequent journal replacement
  is durably ordered. Only recorded files and empty recorded directories are
  retired; the source root is left empty. Partial and completed retries verify
  the exact saved plan and live destinations without rebuilding missing source
  control evidence.
- Status paths are canonicalized, protected source/parent/coordinator/registry
  aliases are refused, existing reports are not overwritten, and fresh output
  is exclusively reserved with inode checks. A separate whole-workflow lease
  excludes competing prepare/import/finalization invocations. The batch child
  inherits that descriptor through `pass_fds`, with close-only release instead
  of explicit unlocking. A real harmless-process SIGKILL regression first
  reproduced the orphan-writer gap and now proves retries remain excluded
  until the surviving batch writer exits. The root also forwards the opt-in
  `PALETTE_RECORDING_IMPORT_LEASE_FD` environment binding; the incoming importer
  owner is correcting propagation into the nested stimulus writer separately.

Independent disposable review reproduced four original blockers: missing
durability ordering, registry status overwrite, dot-dot source status overwrite,
and overlapping importer invocations. Nine failure-first cases reproduced them;
all then passed after corrections. The expanded focused and legacy suite passed
**197 tests in 10.35 s**, including the separate failure-first actual-producer
control-file case. Positive retirement unit cases deliberately stub admission
and prove transaction mechanics only; they do not establish receipt admission.
Final organizer suite including the crash regression: **198 passed, 8.82 s**;
complete collection: **11,931 tests, 34.71 s**, one explicitly reported local
collection skip: `test_train_unet_subject_masks_registry.py:23`, its existing
hyphenated-hostname `torch._dynamo` import guard. A repeat with skip reasons
confirmed that reason and collected 11,931 in 8.00 s; those U-Net registry tests
remain unrun locally, and required remote CI must supply its applicable checks.
Independent safety re-review passed **110 tests, 12.45 s** and closed the
four original blockers plus the supervisor-to-batch lease gap. The nested
stimulus-writer gap remains blocking until the incoming correction is validated.

A fresh actual pinned Citrus CLI transfer with synthetic H5, PTP/geometry
context and real H264 full/crop clips passed transfer, loadback and idempotent
retry. Source snapshot is
`sha256:9499bcff709999ee630c810df57cce48f8a16f96bd80213cb2197e91ac854f0d`;
evidence is `/tmp/palette-citrus-encoded-transfer-20260906-2e72mlrr/producer_smoke_report.json`.
Crop rows include a real encoded blank frame with explicit blank/detection
flags; timestamps and all Orange-shaped envelopes remain declared synthetic.
The actual organizer then prepared both parents and their indexes, and the
unpatched collection builder with real ffprobe accepted both. Preparation-only
evidence is `/tmp/palette-organized-parent-preflight-20260907-njzhinab/preflight_report.json`.
This run left staging intact and did not run import/registry/retirement. The
final combined canary must use fresh parent paths and clean producing code.

Local static gates passed: import boundaries, FPS/keypoint-motion/tail/paradigm
authority ratchets, file-size, explicit Zarr modes, observed metadata literals,
managed contract freshness, registry schema reference and whitespace checks.
The existing census generator updated only the scanned-module counts for the
two new utility modules; generated checks passed. An initial incorrect attempt
to invoke importlinter as a module failed before running it; its installed CLI
entrypoint subsequently passed without an environment change.

Separate prerequisite: clipped importer/admission owner work is committed and
pushed at `2f07800fcee9f5d161226b37d0e24ebdb4b78474` in draft
[PR 155](https://github.com/jmdelahanty/palette/pull/155). It has 452 local focused
passes and is **not integrated here** while its required CI is pending. Its
handoff is on that branch at
`docs/diagnostics/clipped_import_receipt_handoff_2026-09-06.md`.

Every required check remains unrun for this organizer change: generated
artifacts; import boundaries; file-size ratchet; Zarr open metadata modes;
observed metadata literals; active contract freshness; package and collection;
non-GPU shards 0 through 15; and `ci-required`. Successful prerequisite checks
must precede integration, and the combined exact commit requires its own green
CI. The remaining evidence is the unpatched actual pinned Citrus transfer ->
parent organizer/index -> current clipped importer -> clock/crop/receipt ->
isolated registry -> staging retirement and replay test, including negative
controls. No deployment, installed poller mutation, main merge, production
registry mutation, acquisition-machine cleanup or scientific activation is
authorized or performed. Genuine acquisition canary coordination remains
separate; synthetic Orange-shaped envelopes are not encoder/hardware evidence.

## September 7 reviewed development integration

Both incoming exact commits completed all 24 required CI checks before this
integration: organizer `8b5ab93343930fef8f63a0d04c16e497752f5c68` (PR 149) and
clipped importer `45a2a165cbfc597fb3d274694292db286eb51792` (PR 155). The latter
includes the nested stimulus-writer lease correction, independently validated
with 19 passing cases. Both worktrees were clean and their common prerequisite
was `0cd250a58b26e5c56fcdf9926b2593eee643b1e9`. Only their existing generated
writer census overlapped; its disjoint changes merged automatically and remain
subject to the existing generator check. Root owns this development integration;
the separate importer worktree is left unchanged at its reviewed commit.

This merge is not a main merge, deployment, or production activation. All 24
checks must run successfully for the resulting combined commit as well.
Combined local tests and the clean-commit unpatched synthetic canary are next;
neither is claimed complete at this pre-commit checkpoint.

Before committing the combined tree, **368 focused tests passed in 58.32 s**
across organizer, session/batch/ordinary importer, transfer/index, receipt,
read-only registry, acquisition clock and both process-lifetime regressions.
The generated census and whitespace checks also passed without further
generated edits. Full required remote CI and the clean-commit canary remain
unrun at this checkpoint.
Combined local import boundaries, FPS/keypoint/tail/paradigm authority checks,
file-size, Zarr modes, observed metadata, contract freshness and registry schema
reference checks also passed. These are not a substitute for the remote checks.

The fresh positive fixture is
`/tmp/palette-citrus-encoded-transfer-20260906-cp9v8cmx`, and the separately
reserved negative fixture is
`/tmp/palette-citrus-encoded-transfer-20260906-g74o_er5`. Both came through the
actual pinned Citrus CLI, validation and idempotent retry with snapshot
`sha256:8f08c19b53b8a41b1c6e162abfff3a977fcdba2ef7c954f54fa5b5471763ef14`.
The generator now uses the registered-mask reader's required exact asset
locators and validates that reader before transfer. Earlier fixture evidence is
preserved: its generic bundle/import succeeded, but a later registered-mask
read correctly refused its insufficient locator layout. No validator was
weakened and no historical artifact was rewritten to accommodate that fixture.

The canary harness `/tmp/palette_transfer_parent_e2e_20260907.py` requires the
exact clean combined commit and matching imported package path. It will invoke
the maintained CLI with explicit `--transfer-v2 --recording-only --register`
against a fresh isolated registry and destination, then check original-byte
coverage, clock/crop/frame identity, actual receipts/admission, required geometry
loadback, empty staging and byte-stable replay. A separate corruption control
must preserve staging and leave the registry unchanged. This is the explicitly
chosen recording-only workflow with retained synthetic H5/geometry context, not
an authentic stimulus acquisition or Orange hardware/clock canary. Its final
report is an external execution artifact under a fresh
`/tmp/palette-parent-intake-e2e-20260907-*` directory, not a replacement status or
scientific acceptance authority.

## Clean combined-commit synthetic result — September 7

The complete explicitly recording-only synthetic workflow passed against clean
combined code commit `27b48573b01834c09272e72cbe1988d594aeebde`. It invoked the
actual maintained session CLI, ordinary batch/importer, publishers, receipt and
registry readers without monkeypatching git identity or any runtime validator.
The acquisition envelopes, H5 context and timing values were explicitly
synthetic; the H264 media, pinned Citrus transfer execution and Palette import
execution were real. This supersedes only the canary-pending statement above.

- Positive report:
  `/tmp/palette-parent-intake-e2e-20260907-qj248wc6/e2e_report.json`.
  SHA-256: `ece0b01775b22d0082cc77b7716e42040b353d707705bd4f9cca0397418b6b28`.
- Two exact parent cameras, `02010093` and `02010094`, each retained two full
  clips and their crop children, three parent frames, original frame IDs and
  timestamps, and one explicit blank crop row. Both canonical pixel-source
  resolvers retained the two-file collection; neither manufactured a first-clip
  single-video source. Clock and crop readers validated the published arrays.
- Both actual receipt producers bind the clean combined commit above. Receipt
  digests are `ae89b0ffaac78ab39639cddd35d551b39e31c510003b9cde3fa450ea483f668e`
  and `83b0b0f16c4911672bd64a1ca3d779435a39a2980f30445ecb69c9421660560f`.
  The isolated registry's existing shadow-publication path admitted both, and
  the read-only registry resolver verified each exact receipt afterward.
- Complete `PRAGMA integrity_check` and `PRAGMA foreign_key_check` passed
  through `scripts/py -m fisheye.utils.registry_integrity`: `ok`, zero foreign
  key issues, Python SQLite runtime **3.52.0**. Receipt:
  `/tmp/palette-parent-intake-e2e-20260907-qj248wc6/registry_integrity.json`.
- All **39** staged files, including H5, required geometry assets, original
  producer manifests, opaque context, snapshot, completion marker and empty
  transfer lock, matched their explicitly mapped recording destinations by
  content hash and size. Only then were these synthetic staged files retired.
  Staging is empty, while all original synthetic acquisition-source files
  remain unchanged and available. Original control evidence is retained inside
  the recording parents. There was no production or acquisition-machine cleanup.
- Saved-plan replay passed after source retirement, ran no importer command,
  retained the same two receipts, and left all parent bytes and registry bytes
  unchanged. Required registered-mask geometry loadback passed for both cameras.
- Separate negative report:
  `/tmp/palette-parent-intake-e2e-20260907-0en8pw1d/e2e_report.json`.
  SHA-256: `fa11b4a5fd84a84a757b53c4f767296c66f54c9928fa2f0d6ae2d13440c93835`.
  Deliberate corruption of only that disposable delivery's opaque file caused
  the real CLI to refuse with exit 1. It created no recording parents, removed
  no staged files, and left the isolated registry and original acquisition
  source unchanged. The failed delivery remains preserved as diagnostic evidence.

The workflow is implemented and this bounded end-to-end canary is validated;
the development branch is **not yet complete or merge-ready**. At this report's
pre-commit checkpoint, combined-code CI run `34077759883` had six standalone
successes, while package/collection and non-GPU shards 0–15 were running and
`ci-required` was unrun. This documentation-only follow-up requires its own
fresh required CI; successful predecessor checks cannot satisfy that gate.
All 24 follow-up checks are initially unrun: generated artifacts, import
boundaries, file-size ratchet, Zarr open metadata modes, observed metadata
literals, active contract freshness, package and collection, non-GPU test
shards 0–15, and `ci-required`. No source code or producing-artifact identity
is changed by this documentation update, and the canary receipts remain bound
to their actual producer commit rather than being rewritten to the follow-up.

Remaining scope: complete required combined/follow-up CI, then obtain separate
authorization and authentic acquisition-owner inputs for any real deployment
or canary. Live poller rollout, acquisition-machine source-release acknowledgment,
full stimulus-data canary, production registry/selector activation and broader
`INGEST-001` closure are not claimed. No acquisition agent was needed for this
synthetic test; a real acquisition run still needs pinned installed versions,
authentic finalized source evidence, source custody and an authorized destination.
