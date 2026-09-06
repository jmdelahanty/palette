# Ingestion validation corrections — 2026-09-06

## Authorized main-integration refresh

The user authorized the Palette PR integration sequence on 2026-09-06. PR 140
landed as main `3c98979be9bf4eb19db5a855950e53f8dcec563f`; all 24 post-merge
checks passed in [run 34044178789](https://github.com/jmdelahanty/palette/actions/runs/34044178789)
before this clean owned branch incorporated it. This branch's incoming head,
`eb224a216ac7774770b33de021c599b741ce7927`, also had all 24 checks successful.
The history-preserving merge leaves source and test content unchanged.
Fresh focused validation on the combined tree: 281 tests passed in 9.29 seconds;
generated census, file-size, contract-freshness, and whitespace gates passed.

With separate explicit user approval, `main-required-ci` ruleset 22372296 now
requires `ci-required` as the 24th check. All original 23 Actions-bound checks,
strict up-to-date enforcement, pull-request rules, and other protections were
preserved; the readback was verified. There are no bypass actors.

The new candidate remains incomplete until its own 24 checks succeed. PR 143
records its exact commit and current CI; successful incoming runs do not replace
that evidence. This section supersedes the historical preparation-only
authorization/status below. Deployment, shared-checkout updates, dependency
installation, historical mutation, and production activation remain outside
this integration task.

## Authorized candidate refresh

The user authorized Palette-only PR preparation, commits, pushes, and CI on
2026-09-06. Merge to main, repository-rule changes, deployments, shared-checkout
updates, and production activation remain separate.

The clean owned candidate at `3a85a8c9215894945685eaba6f0055730392db44`
(all 23 required checks successful) was combined without conflicts with exact
CI-gate head `74926b21cb2e587b1caae151aaa9c4df91e592b8` (all 24 checks
completed/successful in [run 34028051914](https://github.com/jmdelahanty/palette/actions/runs/34028051914)).
That prerequisite includes current main
`3d017867e79b14d11ddca3ee1916d50ac6499c78`; its newer roster work is preserved.
No dirty clock-publication, ingestion-enforcement, or documentation branch was
integrated. This candidate now runs the success-only aggregate gate as well.

The PR records the resulting exact commit and live CI. All 23 original checks
listed below plus `ci-required` must succeed on this fresh combined candidate;
the incoming green runs are not substitute evidence. Until then it remains
incomplete and ineligible for integration into the next candidate. Historical
implementation and test evidence below retain their original scope.

Fresh combined validation: 281 ingestion/clock/map and CI-gate tests passed in
9.06 seconds; generated census, file-size, contract-freshness, and working/index
whitespace checks passed. Full remote CI remains required for the new commit.

## Scope and ownership

Implementation and focused validation are prepared for draft-PR publication;
**not merge-ready** until required CI succeeds.
Implementation status belongs to the
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md),
Stage 1's ingestion-validation companion, not to this evidence document.

- Owner: Palette ingestion-validation worker.
- Worktree: `/tmp/palette-ingestion-validation-20260906`.
- Branch: `agent/palette/ingestion-validation-20260906`.
- Exact implementation base: `1bf9d9195fb026f7bb426c47ee01fdff99cb698d`.
- The user authorized committing and opening a draft PR on 2026-09-06. Its
  publication handoff records the resulting exact commit and current CI state.
  No merge, deployment, shared-checkout update, registry write, or
  production-data mutation is part of this change.
- Prerequisite: the existing owners at that base. The separately published
  [audit PR 142](https://github.com/jmdelahanty/palette/pull/142) is evidence,
  not an implementation dependency; its unmerged commit was not integrated.
- The two source owners were unchanged since the audit's source baseline
  `0a9531f9914f30a487704388a12a088a8d3365fe`. Relevant worktree changes were
  reconciled before implementation; no overlapping dirty source files or queue
  edits were found. Other workers' checkouts were left untouched.

This is an **enforcement correction**, not a scientific-default, persisted
schema, digest-grammar, or identifier migration. The user selected PTP status
classification, exact integer parsing, and clipped frame-map correspondence.
Hardware synchronization and the separate clock-publication ordering defect
remain outside this patch.

## Implemented behavior and preservation

| Existing owner | Correction | Preserved claim/behavior |
| --- | --- | --- |
| `shared/acquisition_frame_clock.py` | Compare the trimmed, lowercased PTP state with complete supported states: `locked`, `slave`, `master`, `synchronized`. Negative compounds and unknown states cannot support epoch inference. | Existing status-field precedence, absent-status policy, other evidence requirements, raw timestamps, unknown-epoch fallback, and explicit inference label. This is not proof of physical synchronization. |
| Same clock owner | Reject non-integer Parquet values, null IDs, booleans, numeric strings, and values outside signed int64 before conversion. CSV range failures receive the existing typed, field/row-labelled error. Validate source vector dtypes and use overflow-safe temporal ordering. | Exact large integers above float's precision limit; null timestamp validity/sentinel semantics; nondecreasing timestamps; existing source-ID origins without rebasing; serialized valid arrays and records. |
| `shared/clipped_video_collection.py` | Require non-null mapping fields and exact, int64-representable integer columns; prove distinct complete parent IDs and `parent - clip_local == unit_start` for every row, in addition to existing interval coverage. | Correct mappings in arbitrary physical Parquet order; complete recording-wide parent coverage; the existing dense per-clip profile; unit keys, locators, source fingerprints, collection metadata, and digest grammar. |

The frame proof is exact, not probabilistic: parent count equals distinct count
and interval length; every local index has the same offset from its unique
parent. Thus duplicates, missing interior frames, and local permutations cannot
pass merely by preserving extrema and counts. The maintained shell producer
and acquisition repair utility already share this owner; no competing helper
or generic authority schema was added.

Deliberately **not** changed:

- Parquet source-ID origin acceptance versus CSV's existing 0/1-origin policy.
  Negative or shifted source IDs are preserved as compatibility behavior, not
  newly endorsed producer profiles. Deciding supported origins requires the
  producer contract and a separate compatibility decision.
- Hardware exposure, acquisition counter, encoded-frame, or stimulus-state
  equivalence; cross-clock transformation; display/scanout evidence; PTP-summary
  finalization/freshness; or the recorded Orange host-clock capture-point claim.
- Reclassification or rewriting of previously persisted clock publications.
  New-source validation does not retroactively repair historical evidence.
- Clock selector publication order, broader ingestion orchestration, registry
  migration 73, crop-ledger integrity, or production authority activation.

## Tests and comparison evidence

Regression tests were added before source changes. Against unchanged source,
the initial 128-test selection produced **76 failures and 52 passes**, exposing
the reproduced defects and error-contract gaps. The valid clock record's golden
digest was captured separately before implementation:
`621faa07511174f0bbc02f0e6703430061b8710cebcb3c2451a7ed7ffc204771`.

Final focused selection: **139 passed in 8.10 s**. This includes pure Arrow/NumPy
tests, real-Parquet public clock-import refusals before writer access, malformed
clipped producer refusals before archive/manifest creation, shuffled valid
producer -> acquisition resolver -> unpatched native-consumer coverage, and the
existing clock publisher/resolver/tampering tests.

```bash
scripts/py -m pytest \
  tests/unit/fisheye/test_acquisition_frame_clock_validation.py \
  tests/unit/fisheye/test_clipped_video_collection_frame_mapping.py \
  tests/unit/fisheye/test_create_clipped_analysis_zarr.py \
  tests/unit/fisheye/test_acquisition_frame_clock.py -q --tb=short
```

Broader preservation selection: **103 passed in 29.04 s**, with 17 Zarr-v3
consolidation-specification warnings and no skips or failures.

```bash
scripts/py -m pytest \
  tests/unit/fisheye/test_import_recording_analysis.py \
  tests/unit/fisheye/test_import_organized_recordings_analysis.py \
  tests/unit/fisheye/test_recording_import_authority_integration.py \
  tests/unit/fisheye/test_two_session_four_camera_current_authority.py \
  tests/unit/fisheye/test_provider_recording_timing_authority.py \
  tests/unit/fisheye/test_chaser_proxy_relative_frame_adapter.py \
  tests/unit/fisheye/test_composable_epoch_selection_adapter.py \
  tests/unit/fisheye/test_repair_missing_frame_clock_declarations.py \
  tests/unit/fisheye/test_create_clipped_training_zarr.py \
  tests/unit/fisheye/test_build_recording_frame_index.py \
  tests/unit/fisheye/test_consolidate_external_ipc_rolling_recordings.py -q --tb=short
```

Total: **242 distinct passing tests** in these two final selections. All pytest
runs used workstation `scripts/py` outside the sandbox: Python 3.11.14,
pytest 8.4.2, Zarr 3.1.3. No dependencies were installed or changed.

An additional real-index fixture comparison loaded the collection builder from
the exact Git baseline and compared its complete result with the revised
builder on the same fresh inputs. All metadata and digest fields were identical.
The test fixture stubs video probing; this is not a hardware/video-decoding
canary. No historical receipt was restamped to make the comparison pass.

Metadata-only cost check, ten clips, three warmed repetitions per implementation:

| Rows | Parquet bytes | Baseline read + validation median | Corrected read + validation median |
| ---: | ---: | ---: | ---: |
| 100,000 | 809,147 | 0.0108 s | 0.0270 s |
| 1,000,000 | 6,814,290 | 0.1090 s | 0.1800 s |

These synthetic local-file measurements include index reading and validation,
not video decoding, publication, or end-to-end ingestion. Exact validation adds
an Arrow offset column and distinct-parent aggregation; the existing full-index
O(rows) memory class remains, with additional O(rows) state. No Python per-frame
mapping objects, decoded frames, or extra video reads were introduced. This is
not a constant-memory streaming or performance-improvement claim.

## Remaining integration gate and change scope

Local source compilation, generated-catalog verification, import-layer checks,
authority-access ratchets, Zarr-open-mode and observed-metadata checks,
file-size ratchet, and contract freshness were checked. The generated writer
census initially failed after source movement; regeneration changes only the
clock writer's source line and its derived census-site ID, not runtime receipts.

Change scope: the two existing source owners, the existing clipped shell test
module, two new validation test modules, the generated
`zarr_production_writer_census.json`, this handoff, and the owning queue update.

At the pre-publication checkpoint, all remote required CI is **unrun**:
`generated artifacts`, `import boundaries`, `file-size ratchet`,
`zarr open metadata modes`, `observed metadata literals`,
`active contract freshness`, `package and collection`, and
`non-gpu tests (shard 0)` through `non-gpu tests (shard 15)`.
Local components are not a replacement for successful required CI. Full-suite,
wheel/package validation, and any later combined-commit CI remain outstanding.
Any authorized source PR must remain explicitly incomplete until those gates
are successful. No canary, historical migration, or production activation was
performed or authorized by this patch.
