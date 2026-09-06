# Ingestion enforcement: prepared candidate handoff, 2026-09-06

Status authority: `INGEST-001` in
[the consolidation queue](authority_consolidation_work_queue_2026-08-25.md).
This is evidence for that queue, not a separate completion authority.

## Authorized main-integration refresh

The user authorized this Palette integration sequence on 2026-09-06. PR 146
landed as main `e60b1195592defc7323ef35fb952955ebd616d60`; all 24
post-merge checks passed in
[run 34048830865](https://github.com/jmdelahanty/palette/actions/runs/34048830865)
before incorporation into this clean owned branch. Incoming ingestion head
`089c1a930688d57e8e07c0ac25c5485b0ee36c05` also had all 24 checks successful.
The conflict-free history-preserving merge changed no source, tests, scripts,
or workflow content. It retains both parents' dated integration handoffs.
Fresh combined validation: 732 tests across the recorded 40-module union passed
in 64.73 seconds, with 12 standard consolidation warnings and no failures or
skips. Generated census, file-size, contract-freshness, and whitespace gates
passed. The owning queue now records the validated main integrations of the
parser/map and clock corrections; this ingestion candidate remains pending CI.

PR 147 now targets main and records the new exact candidate and current CI.
That candidate must pass all 24 required checks before merge; prior green
incoming commits are not combined-candidate acceptance. This section supersedes
the historical preparation-only authorization/status below without changing
the measured evidence, preservation decisions, or compatibility exclusions.
No deployment, shared-checkout update, dependency installation, historical
mutation, production activation, or transfer-v2 adoption is included.

## Validated-prerequisite combination

The implementation snapshot below was committed as
`d74b49914a543505eed099d7aa4a40e8e17caa92`. The clean owned branch was then
combined with clock-publication [PR 146](https://github.com/jmdelahanty/palette/pull/146)
at exact `2fc409beb67dca48f15c30bec73867cd3633b1f7`, only after all 24
required checks completed successfully in
[run 34030389103](https://github.com/jmdelahanty/palette/actions/runs/34030389103).
That prerequisite includes the validated #143 ingestion corrections, #140 CI
gate, and main `3d017867e79b14d11ddca3ee1916d50ac6499c78`.

Source and tests combined without conflicts. The one generated writer-census
count conflict was resolved by regenerating the existing census from the
combined source. Replay now calls the clock owner's public
`acquisition_frame_clock_source_sha256` interface, preserving its existing
record grammar and digest bytes; no private record builder is imported.

Fresh combined validation passed: **732 tests across 40 modules in 59.85 s**, no
failures or skips, with 12 standard Zarr consolidation warnings. The selection
is the union of the 26 ingestion modules listed below, the clock handoff's
13 modules, and `test_ci_required`, `test_ci_quality_gate_independence`,
`test_agents_required_ci_policy`, `test_ci_pytest_shard`, and
`test_ci_pytest_junit_summary`. Full repository collection found **11,718 tests**
in 7.32 s, exit 0; collection is not full-suite execution. Import-linter kept
both contracts (1,583 files, 7,231 dependencies). Generated census, registry
schema reference, all scoped access/metadata ratchets, file-size, freshness,
compilation, and shell-syntax checks passed. All 29 checked local links in the
nine initially changed Markdown files resolve.

The combined candidate still requires its own complete 24-check CI run. The
draft PR records that resulting exact SHA and live CI; prerequisite success
is not combined-candidate success. No documentation PR,
other worker's incomplete branch, production data, or shared checkout was
integrated or changed. Main merge, deployment, and activation remain separate.

## Initial implementation snapshot: ownership, version, and authorization

- Owner: current Palette root worker.
- Worktree: `/tmp/palette-ingestion-enforcement-20260906`.
- Branch: `agent/palette/ingestion-enforcement-20260906`.
- Exact HEAD/prerequisite: `3a85a8c9215894945685eaba6f0055730392db44`.
  The prerequisite had successful required checks; those checks do not cover
  this dirty patch.
- All implementation below is uncommitted. No commit, push, deployment,
  integration, shared-checkout update, or production activation was performed.
- The user subsequently authorized Palette-only PR preparation on 2026-09-06:
  implementation/review, commits, pushes, and required CI. Main merges,
  repository-rule changes, dependency installation, deployment, and activation
  remain separate. Transfer-v2 adoption is explicitly not part of this wave.
- The separate clock-publication and documentation-reconciliation worktrees
  remain untouched and unintegrated. Relevant organizer/inventory paths were
  checked across registered worktrees before editing; there were no overlapping
  dirty changes. This patch must be reconciled with those exact future commits
  before any combined integration.
- User requested ingestion-only bypass removal, explicitly prohibited automatic
  legacy handling and continuing after optional-component failures, requested
  one parallel read-only acquisition PR review, then authorized posting its
  findings as a PR comment. No dependency installation was requested or done.

## Classification and preserved contracts

This is an enforcement correction, not a numerical optimization or cleanup.
New rejection behavior is intentional. Source identity v2, recording-ID mapping
v1, import-receipt v1, acquisition/clock/crop digests, frame identities,
coordinates, validity, camera defaults, and scientific recipes are unchanged.
The removed preflight override remains a fixed `false` entry in import config
v1 solely to preserve valid configuration digest bytes. It is not an option.

Diagnostic absence and failure are different: diagnostics remain opt-in under
the manifest contract; `not_run`, null undeclared stream inventory, and the
recording-only workflow's absent H5 are not invented passes. Any recorded
component failure/error blocks ingestion, including optional H5 checks and
tooling. Warnings without a failure/error remain distinguishable from failures.
No additional human-review or upstream supplier gate was invented.

The documentation contract revisions are manifest v3 and pipeline v5; persisted
artifact grammars are not migrated. Historical artifacts are not rewritten.

## Implemented locally

1. Removed preflight-failure options from the single/batch importers, analysis
   pipeline, Citrus wrapper, staging finalizer, and submission wrapper.
   CLI/API refusals replace override-success tests. Removed the unsupported
   `--no-import-video-metadata` CLI from current source ingestion; direct API
   attempts to omit required acquisition metadata also fail.
2. Parse recorded preflight with the existing bounded strict JSON owner.
   Malformed JSON, duplicate keys, malformed sections, invalid status values,
   any declared section's `fail`/`error`, and non-null diagnostic error payloads block even
   when a summary claims pass/warn/not-run.
3. Require current source identity before source-analysis writes. Precreating an
   output directory cannot select the old unprofiled writer. Existing current
   roots must match their manifest. Immutable receipts prohibit re-import writes.
4. Reject undeclared videos and selecting a declared crop video as the full-frame
   source. Malformed acquisition publication markers are not reclassified as
   unpublished. Clock cardinality comes from verified acquisition authority,
   not the first coercible root/raw attribute.
5. Added read-only `load_verified_recording_import_receipt` to the existing
   registry identity/receipt owner. Both batch skip paths and sealed pipeline
   replay use the live receipt/acquisition verifier, consolidated-generation
   checks, applicable clock source/payload digest checks, and canonical crop
   checks. A directory, receipt filename, or publication marker alone is not
   sufficient. This helper grants no registry authority.
6. Explicit organizer logs never fall back to filesystem discovery. Applied
   batches fail for missing/failed inputs or an empty selected set. Citrus must
   find exactly one fresh invocation log and actual acknowledgments covering
   exactly its organized outputs; plan rows and old logs are not successes.
7. External-IPC organization rejects summaries masquerading as full metadata
   CSVs and never declares crop timestamps as the full clock. The inventory
   ingestion validator also rejects an undeclared conventional clock when
   streams are declared, closing the subsequent loader substitution path.
8. Added `validate_acquisition_video_stream_inventory` in the existing inventory
   owner, used by its writer and source replay. Declared missing/unreadable or
   malformed files, failed summary/status payloads, duplicate-key summaries,
   row-count conflicts, and declared/observed colorimetry conflicts fail before
   inventory mutation. Read-only diagnostic inventory construction still reports
   invalid inputs. Existing crop-specific scientific validators remain intact.
9. Extracted the standalone manifest utility's context rules into the dedicated
   `shared.recording_manifest_context` owner and migrated the utility, new
   import, and sealed replay. Required strings/vocabulary/mode consistency are
   executable gates, not importer `behavior/free` defaults. Existing/root replay
   context must agree with the manifest, including after reconsolidation.
   The utility retains explicit registry-custom vocabularies and its separate
   opt-in historical repair command; source intake uses the standard declared
   manifest vocabulary. The source-identity parser/digest is not redefined.
10. Reconciled sampled-training entry points separately: manual intake requires
    explicit session and recording IDs, never guessed names or session-to-recording
    substitution. Its metadata tool refuses source-analysis archives. The
    manual wrapper, atomic training publisher, and internal sampled constructor
    reject recorded failures; a present manifest is read strictly and validated.
    Genuine manifest absence remains supported by this manual training product.
    `video_only_v1` scientific defaults, sampled frame identity, storage/chunking,
    and selector-ineligible atomic publication are preserved. This does not mint
    regular source-analysis identity or authority for sampled training.

## Tests changed and evidence

Bad success assertions were replaced, not hidden behind skips or patched new
validators: failed-preflight override success; unprofiled legacy intake;
directory-only skip; fabricated receipt-path replay; plan rows treated as
acknowledgments; crop timestamps used for the full video. The crop cardinality
test now asserts earlier refusal with no inventory group created. Current
external-IPC happy fixtures supply distinct full metadata and summary files.

Counterexamples were run before each implementation: preflight/legacy intake,
missing receipt/authority/stale metadata, removed or tampered clocks and changed
clock sources, old/malformed acknowledgment logs, 13 organizer/inventory cases,
two sealed replay cases, and crop-source/conventional-clock substitution.
These failed on their unfixed boundaries; the final regression result is below.

Real-artifact integration uses actual Zarr writers, acquisition/clock authority,
immutable receipts, live reopen, and registry finalization/shadow publication.
It checks missing/tampered/stale evidence, valid sealed replay before detection,
and Citrus acknowledgment validation against real published artifacts. Media
probe and clean producing-code identity doubles keep the fixture bounded;
this is not a real-media decode or dirty-working-tree publication canary.
The two-session/four-camera test exercises the real organizer and eight distinct
recording identities. Its organizer logs now stay under pytest temporary paths;
earlier runs used the test's pre-existing default `/nvme1/recordings/logs` log
directory. No real recording payload or production registry was changed.

Local validation:

- Combined final 26-module regression: 373 passed in 49.09 seconds, no
  failures, skips, or xfails. This supersedes the original 321-test selection,
  the intermediate 370/371-test runs, and the focused follow-up results.
- Final schema-gate review reproduced whitespace-wrapped
  `recording_analysis_v1` passing the manual-training plan's early check.
  The normalized schema is now rejected before publication; both literal and
  whitespace variants pass their refusal tests (21 training regressions pass).
- New context/training counterexamples first produced 19 context, five manual/
  publisher, and three constructor failures against unfixed code. Contradictory
  root context also reproduced false successful replay after reconsolidation;
  its real-artifact refusal and valid sibling now pass.
- Repository-wide collection before the two final schema refusal cases: 11,543 tests collected,
  exit 0, one collection skip. `test_train_unet_subject_masks_registry.py:23`
  skips for a host-specific `torch._dynamo` import bug involving the hyphenated
  `nx-loopback` hostname. Those tests are unrun here, not passing evidence.
  Collection is not full-suite execution.
- Import-linter: both contracts kept (1,583 files, 7,227 dependencies).
- File-size, Zarr metadata-mode, observed metadata, FPS authority, keypoint-motion
  authority, tail-receipt, and paradigm-core access checks passed.
- Generated Zarr census, registry schema-reference check, active contract
  freshness, changed/new Python compilation, shell syntax, and
  `git diff --check` passed.
- The Zarr-mode baseline only tightens four removed implicit opens; no new
  implicit-mode exemption was added.

Focused validation was run on the workstation outside the sandbox using
`scripts/py -m pytest`, with these modules under `tests/unit/fisheye/`:

```text
test_recording_preflight.py
test_import_recording_analysis.py
test_import_recordings_analysis.py
test_import_organized_recordings_analysis.py
test_run_recording_analysis_pipeline.py
test_run_citrus_session_import.py
test_finalize_organized_staging.py
test_organize_recordings_diagnostics.py
test_recording_import_authority_integration.py
test_two_session_four_camera_current_authority.py
test_recording_identity_authority.py
test_registry_recording_import_receipt_bindings.py
test_refresh_recording_preflight.py
test_acquisition_video_stream_intake_enforcement.py
test_organize_recordings_external_ipc.py
test_registry_acquisition_video_streams.py
test_acquisition_crop_stream_ledger.py
test_acquisition_crop_stream_collection_ledger.py
test_backfill_acquisition_video_stream_inventory.py
test_organize_recordings_video_only.py
test_organize_recordings_legacy_h5.py
test_organize_recordings_keyframe_flags.py
test_validate_recording_manifest.py
test_intake_video_only_recording.py
test_training_base_publication.py
test_import_sampled_training_pynvvc.py
```

## Required CI at the initial implementation snapshot

The exact dirty patch has no CI run and is not complete or merge-ready. All 23
required checks remain unrun: `generated artifacts`, `import boundaries`,
`file-size ratchet`, `zarr open metadata modes`, `observed metadata literals`,
`active contract freshness`, `package and collection`, and each
`non-gpu tests (shard N)` for N = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
13, 14, 15. Local diagnostic success does not substitute for these required
exact-commit checks. The planned validated stack additionally requires
`ci-required`, also unrun for this patch. No required check is claimed
intentionally inapplicable.
Wheel build/install validation was not run; dependency mutation remains
unauthorized. After authorized commit/push, require all checks on that exact
commit, and again on any subsequently combined candidate.

## Remaining audit and compatibility work

This patch closes reproduced bypasses; it is not a claim that every entry path
or every ingestion contract has been exhaustively enforced.

- Required context and manual sampled-training entry points are reconciled as
  described above. Deliberately rejected: guessed manual IDs, present failed/
  malformed manifests, invalid required context, and editing source-analysis
  through the manual training tool. Genuine manual-training manifest absence
  does not invoke a legacy source-analysis fallback; this product stays training
  and selector-ineligible. Explicit historical repair, legacy readers, and
  backfill remain separately scoped and do not create current source authority.
- Parent-level clipped intake and producer transfer v2 adoption remain separate
  open work. Do not force the regular single-video source profile onto a clipped
  collection or claim a transfer receipt is Palette ingestion acceptance.
- The validated-prerequisite combination above adopts the clock owner's tested
  public `acquisition_frame_clock_source_sha256` interface; the initial private
  `_build_record` dependency is removed without changing digest grammar.
  Combined validation is required. The broader metadata-equivalence audit
  remains separate.
- Recorded diagnostic summaries are not source-content seals. This change does
  not introduce mandatory diagnostics, full media decoding, or a new manifest/
  H5/clock aggregate receipt. Full producer-transfer-to-Palette-to-registry
  end-to-end media evidence and catalog/AST entrypoint closure remain open.
- No performance benchmark, scientific acceptance, or production activation is
  claimed. Valid numerical products are not invalidated by unrelated diagnostic
  or presentation concerns; no lifecycle status schema was renamed.

## Acquisition-side coordination

The authorized review comment was posted and read back exactly:
[agent-contracts PR 49 comment](https://github.com/jmdelahanty/agent-contracts/pull/49#issuecomment-5558416214).
Reviewed contract head: `ff275f3b82f7d72b268d88581385343fc04dc317`;
referenced Citrus producer: `9602fc4f79c1b12dedefb3c3ac461455aeb36302`.

Two reproduced high-priority producer gaps: single-clip parent/finalization
conditions enforced only for rolling mode, and trailer/output-close errors
accepted despite success booleans. The smaller contract schema mismatch accepts
trailing-slash paths that its runtime rejects. The comment includes concrete
negative fixtures and distinguishes optional proof absence from present failed
proof. The 26 passing producer tests missed those cases. Full Draft 2020-12
schema validation was unrun because the local environment lacks `jsonschema`;
no dependency was installed. No acquisition implementation, PR review state,
merge, or approval was changed.

### Acquisition reply and independent re-review

The user subsequently requested a parallel read-only re-review of the
[acquisition reply](https://github.com/jmdelahanty/agent-contracts/pull/49#issuecomment-5558519563).
Reviewed successor contracts head:
`6cb56b4deae6a9a56bf4b3a3186168c15e91f542`; successor Citrus producer pin:
`e881f5258be83231b62a00a6f9c4e5fcd69cd548`.

On those versions, both original P1 probes now refuse before marker publication
and retain source evidence. The trailing-slash path discrepancy is also fixed.
Independent validation passed 27 revised producer tests plus 6 legacy tests
(no skips), the Node/Python path corpus, and limited transport loadback for all
three committed bundles. Regenerating those bundle trees matched the committed
fixtures byte for byte. This closes the originally reported producer/path
findings, not the outstanding Palette admission work.

The new handoff explicitly requires Palette to reject the transport-valid
`failed_optional_proof` bundle, without legacy fallback or discarding the failed
proof; it does not claim Palette has implemented that gate. Its fake-media proof
fixture contains only status/reason. A future consumer test must assert the
proof-specific failure or use otherwise-valid positive/negative siblings so it
cannot pass merely because schema/media validation fails for a different reason.
Full schema-engine validation and true producer-to-Palette v2 end-to-end evidence
remain open. The re-review made no external comment, review, or PR-state change.

## Initial implementation scope

Source: `recording_identity_authority`, `acquisition_video_streams`,
`recording_preflight`, new `recording_manifest_context`,
`training_base_publication`, `import_sampled_training_pynvvc`,
`intake_video_only_recording`, `validate_recording_manifest`,
`finalize_organized_staging`, both batch analysis
importers, `import_recording_analysis`, `organize_recordings`,
`run_citrus_session_import`, and `run_recording_analysis_pipeline`.
Scripts: Citrus submission wrapper and tightened Zarr open-mode baseline.
Tests: the modified/new ingestion, replay, preflight, inventory, organizer,
crop-ledger, and two-session fixtures listed by `git status --short`.
Docs: manifest/pipeline contract revisions, sampled-import and diagnostic/pipeline
operator guidance (including removal of the retired failure-override instructions),
generated census source-site/count refreshes, this handoff, and the owning queue
entry. All 43 paths are still local at this snapshot; the eventual PR records
the exact candidate commit and dirty/clean state.
