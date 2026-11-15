# Parent-recording transfer-v2 intake: implementation handoff

Status authority: `INGEST-001` in the
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md).
This document records evidence and compatibility decisions, not another queue.

## Main reconciliation and regenerated inventories — September 7

The user authorized reconciling current CI-green `main` into the existing
root-owned intake branch, regenerating its census artifacts, validating,
committing/pushing and waiting for fresh CI. This does not authorize merging
PR 149, deployment, live acquisition, poller installation or activation.

- Owner/worktree/branch: Palette root worker,
  `/tmp/palette-parent-clipped-intake-20260906`,
  `agent/palette/parent-clipped-intake-20260906`.
- Starting head: `cb60279b2036e5cbb2447571d3150ca63a71596e`, the pushed
  naming-documentation commit. Its 24 required checks were unrun because the
  PR conflicted with newer `main`; the previous green head had the same two
  census conflicts. The naming correction did not introduce them.
- Incoming prerequisite: exact `fc6fea5fd937759f5ed21c79d13279b25c309e13`,
  verified as current `main` with all 24 checks successful in
  [run 34131195011](https://github.com/jmdelahanty/palette/actions/runs/34131195011).
  Relevant worktree commits and dirty scopes were rechecked before integration.
  The original checkout's dirty instructions/documents and the chaser owner's
  untracked outputs remain untouched; no other worker's uncommitted changes
  were adopted.
- Classification: behavior-preserving conflict resolution and documentation,
  incorporating the already accepted mainline chaser/CI changes unchanged.
  Only the two generated census files conflicted. The existing
  `scripts/py -m fisheye.diagnostics.zarr_storage_census --write` generator
  resolved them from the combined source: **1,717 scanned modules**. Its normal
  source-location-derived IDs reflect the combined code; no historical
  scientific identities, transfer digests, source bytes or canary receipts
  were rewritten. No runtime file required manual conflict resolution.
- Local combined-tree validation: **304 tests passed in 74.79 seconds** through
  workstation `scripts/py` outside the sandbox. Coverage includes transfer
  schema/snapshot, parent workflow, portable canary package, session importer,
  organized import, staging finalization, clipped importer/receipt, source
  identity, census, independent CI gates and JUnit summaries. Import boundaries,
  FPS/keypoint-motion/tail/paradigm authority ratchets, file-size, explicit Zarr
  modes, observed metadata literals, managed-contract freshness, registry schema
  reference, regenerated-census freshness and whitespace checks passed.

At this pre-commit checkpoint, the merge and this handoff/queue update remain
uncommitted. All 24 combined-head required checks are unrun: generated artifacts,
import boundaries, file-size ratchet, Zarr metadata modes, observed metadata
literals, active contract freshness, package/collection, non-GPU shards 0–15
and `ci-required`. The new head is **not merge-ready until all succeed**; consult
the exact head and run in [PR 149 checks](https://github.com/jmdelahanty/palette/pull/149/checks).
Earlier clean-code synthetic results remain bound to their original commits,
not this reconciliation. The acquisition request remains draft/unsent; service
alias migration, real-data canary, deployment and broader `INGEST-001` closure
remain separately scoped and incomplete. No shared checkout or live service
was changed, and no dependency was installed.

## Repository and cluster-service naming — September 7

The user clarified these distinct identities:

| Name | Meaning |
| --- | --- |
| Orange | Acquisition repository; discussed through the acquisition agent. |
| Citrus | Separate stimulation-library repository; also discussed through the acquisition agent. |
| Palette | Separate analysis/intake repository. |
| `cluster-login1-poller` | Preferred name for the cluster login-1 polling/submission method and service, formerly called the "Citrus login-1 poller". Not the Citrus repository or an AI-agent identity. |

Repository, AI-agent session, execution host and operational service are
different identities; do not assume a one-to-one mapping. In particular, the
command-center proposal must not create a "Citrus cluster agent" merely from
the old poller name. Its actual operational owner/session remains to be
confirmed separately from the acquisition agent's Orange/Citrus remit.

This checkpoint is a documentation-only terminology correction on the
root-owned `agent/palette/parent-clipped-intake-20260906` worktree at
`/tmp/palette-parent-clipped-intake-20260906`, based on exact
`6f6417f49b29f582cefc21fc0adc51ac6705c9bc`. That base passed all 24 required CI
checks. The user subsequently authorized committing/pushing these three
documentation corrections and waiting for fresh CI. At this pre-commit
checkpoint the naming edits are uncommitted, and all 24 new-head checks are
unrun: generated artifacts, import boundaries, file-size ratchet, Zarr metadata
modes, observed metadata literals, active contract freshness, package/collection,
non-GPU shards 0–15 and `ci-required`. Consult exact-head
[PR 149 checks](https://github.com/jmdelahanty/palette/pull/149/checks) for the
result before any integration. No runtime behavior, data schema, persisted
identity or digest changes; the acquisition request remains draft and unsent.

Local validation before this documentation commit: `git diff --check`,
file-size ratchet, managed-contract freshness, Zarr metadata modes and observed
metadata literal checks passed. The portable canary-package suite passed all
23 tests in 0.75 seconds through workstation `scripts/py` outside the sandbox.
The diff contains no runtime, test, configuration or historical-evidence edits.
These local results do not replace the required new-head CI checks above.

The code inventory still finds `login1-citrus-poller` as an executable SSH
default in deployment/submission scripts and Python consumers, with tests and
historical commands referring to it. Keep that literal until an explicit
compatibility migration verifies the new alias's target, credentials and
restrictions, updates maintained callers/help/tests, and proves unchanged
submission behavior with required CI. Existing `--submit-host` or
`PALETTE_LSF_SUBMIT_HOST` overrides may be used where supported only after the
selected alias is actually configured and validated. Do not replace an alias
merely because the descriptive name changed.

The separately recorded workstation staging-marker poller at
`/home/delahantyj@hhmi.org/bin/citrus_staging_marker_poller.sh` is not renamed
by this correction. Do not infer its execution location from the cluster
service name or rewrite historical script paths, hashes, receipts or Citrus
transfer/source provenance. SSH configuration, cron, installed scripts and
running services remain untouched; any executable/installation rename needs
its own authorized compatibility and rollout work. The broader intake and
production-deployment scope remains unchanged.

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

## Full stimulus synthetic canary — September 7

The user-authorized full-stimulus extension passed on clean producing commit
`a5a6ff6dd83df3f49a8f1ed7108ea6846ef3a298` in
`/tmp/palette-parent-clipped-intake-20260906`, branch
`agent/palette/parent-clipped-intake-20260906`, owned by the root Palette worker.
All 24 required checks for that exact commit passed; this supersedes the
earlier CI-pending checkpoint. This extension is test-only: no production
implementation, validator, scientific default, persisted grammar or identity
contract changed. These handoff/queue additions are subsequent uncommitted
documentation, not part of the clean producing commit or its CI evidence.

The actual session CLI ran with `--transfer-v2 --apply --register` and neither
`--recording-only` nor the stimulus metadata-only bypass. It exercised the
ordinary batch importer, nested stimulus subprocess, publication, existing
readers, isolated shadow-registry admission, staging retirement and saved-plan
replay. The fresh source contains synthetic renderer-only H5 data using the
supported semantic-v1 protocol grammar: three frame rows, four events and two
steps per camera, plus selected calibration/display snapshots. Original H264,
crop/blank rows, geometry, PTP-unqualified context and opaque bytes remain in
scope. The two additional synthetic homography source artifacts bring staged
coverage to **41 files**.

- Positive execution report:
  `/tmp/palette-parent-intake-e2e-20260907-txvsc9t1/e2e_report.json`;
  SHA-256 `9fbb835670b159272f1b901989e3a3a98ba684912f970583555ea3a23d8cc1c0`.
  Source fixture: `/tmp/palette-citrus-encoded-transfer-20260906-uor2oo3v`;
  pinned Citrus snapshot:
  `sha256:124553160c8f269c398ae202196cd8c3fc10fa734b8087d12ff0fa46d052400a`.
- Cameras `02010093` and `02010094` both published complete stimulus runs.
  Their exact H5 event/frame values, semantic JSON/trial-index bytes, selected
  camera matrices, renderer snapshots and protocol step bindings passed
  readback. The maintained stimulus event consumer, selected-calibration and
  physical-coordinate readers, and registry extractor all ran unpatched.
  Synthetic scale bindings validate storage/identity only, not real calibration.
- The registry contains **two stimulus runs and four recording-step rows**.
  Recording-import receipt digests are
  `f13a187b620a65c980b381b6a643f5393dc67d7ab21ebe7a30ce585019a4bd64` and
  `26f399601fff376db1d99bda7bde6aec82cd1a460c53d4a6d345f2a459544718`.
  Both stimulus writer provenance records and import receipts bind the exact
  clean producing commit above.
- Every staged file matched all declared recording destinations by hash and
  size before retirement. Staging is empty; original synthetic acquisition
  copies remain unchanged and all retired material remains recoverable in the
  recording folders/source fixture. Replay ran no importer and changed neither
  parent bytes nor registry bytes.
- The separate malformed-H5 control omitted required frame metadata **before**
  the actual Citrus transfer was sealed. Transport verification passed, then
  real Palette stimulus import refused with exit 1. Both failed candidates
  remained selector-ineligible, no import receipt/admission was acknowledged,
  staging and original sources stayed intact, and registry bytes were unchanged.
  Report: `/tmp/palette-parent-intake-e2e-20260907-5sa_8n0p/e2e_report.json`;
  SHA-256 `c32a3d1ceece8fb8ebabe069f543651adcca9307c58a703d90eb37029d6ee442`.
  Its source fixture is `/tmp/palette-citrus-encoded-transfer-20260906-jm0jfwat`.
- Both isolated registries passed complete integrity and foreign-key checks
  through `scripts/py -m fisheye.utils.registry_integrity`, using Python SQLite
  **3.52.0**. Each execution directory contains `registry_integrity.json`.

The external harness is `/tmp/palette_transfer_parent_e2e_20260907.py`, SHA-256
`4a45b3a90226d683d385a57a9053bdc85843816d6b47fbbbbf74790e2008a490`.
The H5 builder is `/tmp/palette_stimulus_fixture_20260907.py`, SHA-256
`3c3abad49510bcb8b81cf677e8ba8626a01d035f3a9c9f7d3eebcdb3de435ad2`.
The producer report records the generator and base-fixture hashes. Earlier
execution reports remain unchanged: initial verification probes incorrectly
read event-specific columns as plain strings/generic structured columns and
searched the supervisor log instead of the child log for the negative error.
The final fresh runs above use the maintained event reader and actual child
logs. No runtime contract was weakened to satisfy those harness assertions.

This closes the bounded renderer-only full-stimulus synthetic test, not a
chaser-coordinate canary, finalized semantic-v2 execution-index test, sealed
stimulus-to-acquisition mapping claim, authentic hardware/clock validation,
live deployment, acquisition-source release or production activation. No
dependency installation, commit, push, deployment or production write occurred
in this follow-up. Only this handoff and the owning queue are modified in the
repository. Required checks for a future documentation commit remain unrun:
generated artifacts, import boundaries, file-size ratchet, Zarr metadata modes,
observed metadata literals, active contract freshness, package/collection,
non-GPU shards 0–15 and `ci-required`. The green producing commit does not
validate a future documentation commit or authorize integration/activation.

Supplemental focused regression: **108 passed in 142.01 seconds**, outside the
sandbox using the Palette Python wrapper. No test was skipped or failed:

```bash
scripts/py -m pytest \
  tests/unit/fisheye/test_stimulus_import_lease.py \
  tests/unit/fisheye/test_import_stimulus_to_zarr_paths.py \
  tests/unit/fisheye/test_import_stimulus_to_zarr_context.py \
  tests/unit/fisheye/test_registry_stimulus_metadata.py -q
```

All three external fixture/harness scripts passed `py_compile`; the two
documentation changes passed `git diff --check`.

## Reusable package and acquisition-readiness draft — September 7

The user authorized checking in the reusable canary, archiving the evidence,
drafting the acquisition-agent request, committing/pushing the changes and
waiting for fresh required CI. The root worker continues in the same worktree
and branch above, from exact CI-green prerequisite
`a5a6ff6dd83df3f49a8f1ed7108ea6846ef3a298`. The two pre-existing documentation
edits belong to this worker and are preserved. Other worktrees and geometry
branches remain untouched. No acquisition message, real capture or deployment
is authorized or performed by this packaging step.

Classification: behavior-preserving extraction of the external **test tools**,
plus explicitly scoped test-custody enforcement. The maintained package is
[`tests/canaries/parent_intake`](../../tests/canaries/parent_intake/README.md).
It removes machine-specific source-checkout defaults and requires an explicit
Citrus source snapshot matching all four existing byte pins. It rejects
optimized Python and noncanonical/non-synthetic locations, symlinks, shared
hard links, special files and oversized fixture trees before runtime input
mutation. Fresh valid fixture grammar, numerical/row identities, event readers,
production import/receipt/registry validators and byte-preserving retirement
remain unchanged. The canary uses its own bounded temporary destinations and
registry; these safety limits do not change production admission.

The [durable evidence archive](parent_intake_synthetic_canary_2026-09-07/README.md)
contains byte-exact historical reports and non-executable source snapshots with
`SHA256SUMS`; old absolute paths and producing identities were not rewritten.
The old negative generator matches its recorded hash; missing earlier builder
hash fields remain missing. This is a report archive, not a new status or
admission authority and not a copy of disposable Zarr/media payloads.

The [acquisition-agent request](acquisition_parent_intake_canary_request_2026-09-07.md)
is explicitly **draft and unsent**. It requests installed versions, an existing
representative finalized bundle or a capture proposal, source/frame/timing/H5/
geometry facts and custody/destination planning. It does not approve acquiring,
transferring, deploying, changing scientific defaults, releasing source copies
or activating production authority.

Initial package validation: 23 focused tests passed in 0.77 seconds; the earlier
first collection attempt deliberately exposed the missing safety module before
implementation. Full-suite collection then found 12,054 tests in 8.04 seconds,
with the same explicit local U-Net registry module skip at
`test_train_unet_subject_masks_registry.py:23` for this hostname's existing
`torch._dynamo` import issue. This skip is deferred local evidence, not a
successful substitute for that remote required shard. All local import-boundary,
FPS/keypoint/tail/paradigm access, generated census, file-size, Zarr-mode,
observed-metadata, managed-contract, registry-reference and Ruff checks passed.

The package is locally implemented; a clean-commit invocation of the packaged
CLI and final exact-head CI still need to run. All 24 new-head checks are unrun
at this pre-commit checkpoint: generated artifacts, import boundaries,
file-size ratchet, Zarr metadata modes, observed metadata literals, active
contract freshness, package/collection, non-GPU shards 0–15 and `ci-required`.
Do not describe the new package revision as complete or merge-ready until
those exact-head checks succeed. Fresh canary reports will bind their actual
clean producing commit; a later report-only commit does not change that code
identity. No installation, main merge, shared-checkout update or activation.

### Clean packaged-CLI result

All four maintained-module canaries passed on clean local code commit
`3f5aad3eb078ebb97af1648ac0120886fb6b250e`: full-stimulus positive,
missing-frame-metadata refusal, recording-only positive and transport-corruption
refusal. This supersedes only the packaged-CLI-pending statement above. The
[archived case table](parent_intake_synthetic_canary_2026-09-07/README.md#packaged-cli-validation)
links the exact unmodified execution reports and source/SQLite evidence. The
first package commit was intentionally not pushed before these local canaries;
it was experimental and not merge-ready. The current follow-up adds only
documentation, archive bytes and their checksum index, not executable changes.

Both positives admitted the expected two parent recordings, preserved all
41 (full-stimulus) / 39 (recording-only) delivered files before retirement,
emptied staging, retained acquisition originals and replayed without changing
parent or registry bytes. Full stimulus also read back the two runs/four steps,
original event and frame columns, protocol semantic bytes, renderer snapshots,
selected calibration and physical-frame fixture bindings. Both negative cases
refused safely with unchanged registry bytes and retained staging/originals.
All four Palette-runtime SQLite integrity receipts passed. No production data
or acquisition-machine sources were retired.

Full-stimulus fixture/run locations were
`/tmp/palette-citrus-encoded-transfer-42a3bbok` /
`/tmp/palette-parent-intake-e2e-vtinrn2x` and negative
`/tmp/palette-citrus-encoded-transfer-ddce7h1n` /
`/tmp/palette-parent-intake-e2e-cy1yjjcl`. Recording-only locations were
`/tmp/palette-citrus-encoded-transfer-pjmi4w_z` /
`/tmp/palette-parent-intake-e2e-4gm_mszi` and negative
`/tmp/palette-citrus-encoded-transfer-5if1a9ik` /
`/tmp/palette-parent-intake-e2e-aqlfbsz5`. These are historical execution locators,
not permission to reuse successful/previously materialized fixture roots.

The portable package suite passed again: **23 tests in 0.75 seconds**, including
the historical archive checksum census. At this archive-only follow-up's
pre-commit checkpoint, every final-head remote check remains unrun: generated
artifacts, import boundaries, file-size ratchet, Zarr metadata modes, observed
metadata literals, active contract freshness, package/collection, non-GPU
shards 0–15 and `ci-required`. The exact pushed head and its final required CI
must be checked independently in [PR 149](https://github.com/jmdelahanty/palette/pull/149/checks).
There is no remaining local canary failure, but this checkpoint is not a
merge-ready claim. The readiness draft remains unsent; no real-data run,
acquisition deployment, live poller change, production activation or source
release occurred. Broader `INGEST-001` and geometry calibration remain separate.
