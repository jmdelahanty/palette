# Clipped current-import admission handoff — 2026-09-06

Status: the base implementation was committed by root at
`2f07800fcee9f5d161226b37d0e24ebdb4b78474` and pushed as draft PR 155. The narrow
stimulus-lease correction below is uncommitted and paused for root review.
Nothing here is integrated, deployed, activated, complete, or merge-ready;
required incoming and combined CI remain blocking until successful.

## Follow-up: inherited stimulus-writer lease — 2026-09-07

- Reconciled clean follow-up base: `2f07800fcee9f5d161226b37d0e24ebdb4b78474`
  in the same worktree/branch below. Root reported its CI pending at assignment;
  that commit's checks cannot validate this subsequent uncommitted correction.
- Owned dirty scope is only `utils/import_recording_analysis.py`, new
  `test_stimulus_import_lease.py`, and this document. Root owns workflow-lock
  acquisition and environment/descriptor forwarding into the batch/session
  command. No independent commit, push, integration, or production mutation.
- This is an explicit execution-safety enforcement correction, not a receipt,
  scientific, coordinate, identity, or provenance-grammar change. The opt-in
  environment interface is `PALETTE_RECORDING_IMPORT_LEASE_FD`. The actual
  `run_stimulus_import` subprocess validates ASCII decimal/nonnegative input,
  an open descriptor, and regular-file `os.fstat` mode, then forwards
  `pass_fds=(fd,)`. Malformed, oversized, closed, directory, and pipe descriptors
  are refused before launch. An absent variable preserves the exact historical
  command and `check=False` subprocess kwargs. No process here acquires,
  unlocks, or closes the inherited lease; lifetime remains with its owner and
  inherited OS handles.
- Preservation/regression evidence: the unmodified base first reproduced
  concurrent retry admission after batch SIGKILL while a harmless stimulus leaf
  remained alive (the expected flock-contention assertion failed). The same
  real importer/subprocess test passes after the fix: the leaf holds exclusion
  after batch death and outer-handle close, and retry acquires only after the
  leaf finishes. Only the child module is replaced with a disposable harmless
  writer; the importer function and subprocess boundary are not patched.
- New unit coverage also verifies exact no-environment command/kwargs and no
  close/unlock after child success, nonzero exit, or launch exception. Final
  focused validation passed **128 tests in 18.94 s** across new lease tests and
  ordinary/batch/clipped importer plus receipt preservation. Full collection
  passed: **11,962 tests, 7.98 s**. All local static gates passed: two import
  contracts; FPS/keypoint/tail/paradigm authority access; file-size; Zarr modes;
  observed metadata; contract freshness; registry schema reference; generated
  census; scoped formatting; compilation; and diff check. Generated artifacts
  and baselines required no changes. All follow-up edits are now frozen.
- Frozen source SHA-256: `utils/import_recording_analysis.py`
  `8591d2b99593f270937753fa4eb6f513f1c885370b88e2bf9c67bb4034658c4b`.
  New test SHA-256:
  `defadefac3a80006a4a7b8b9f95e0a46c9122aabecb6f836113c8666e68d8252`.
- All 24 remote required checks are unrun for the follow-up: generated
  artifacts, import boundaries, file-size, Zarr metadata modes, observed
  metadata literals, contract freshness, package/collection, non-GPU shards
  0–15, and `ci-required`. Root must review, commit/push, obtain successful
  incoming CI, then separately validate any authorized combined integration.

## Ownership and base

- Worktree: `/tmp/palette-clipped-import-receipt-20260906`.
- Branch: `agent/palette/clipped-import-receipt-20260906`.
- Exact clean prerequisite/base: `0cd250a58b26e5c56fcdf9926b2593eee643b1e9`.
- Owner: delegated clipped receipt/admission worker; root owns organizer,
  mapped indexes, session runner, and the central work queue.
- Dirty implementation scope: `registry/recording_identity_authority.py`,
  additive `shared/acquisition_frame_clock.py` helper,
  `shared/acquisition_video_streams.py`, `utils/import_recording_analysis.py`,
  and minimal layout discovery/propagation in
  `utils/import_organized_recordings_analysis.py`.
- New tests: `test_clipped_recording_import_receipt.py`,
  `test_current_clipped_recording_import.py`, and
  `test_read_only_recording_import_admission.py`; new document: this handoff.
  Generated changes use the existing production-writer census and explicit
  Zarr metadata-mode baseline owners. No independent commit or push occurred.
- `shared/recording_import_receipt.py` is unchanged. No dependency installation,
  production-data mutation, deployment, or selector activation is authorized
  or performed by this work.

## Contracts and preservation

The change is a behavior-preserving extraction of existing read-only
intake-surface checks, explicit source-layout support, and an enforcement
extension admitting canonical clipped collections through current
receipt/registry boundaries. Valid
single-video workflows, receipt v1/config grammar, canonical authority records,
camera text (including leading zeroes), parent frame identity, coordinates,
clock validity/PTP semantics, and source/array digest grammars are preserved.

The new public clock helper is
`load_clipped_acquisition_frame_clock_source(recording_dir, *, camera_id,
frame_index_path, expected_frame_count) -> AcquisitionFrameClockSource`.
It uses the existing Parquet loader and validator; its source must be an exact
existing regular file confined to the recording, with a positive exact frame
count. It has no CSV selection, conventional-index fallback, or first-clip
video stand-in. The single-video loader/API is unchanged.

Every clipped projection/finalization, receipt-bound read, and read-only replay
now verifies the exact layout/publication-mode/locator pairing, existing
collection live-file evidence, declared three-index bindings, strict current
identity/context/preflight, complete recording-wide clock digest and payload,
direct/consolidated metadata equivalence, and a declared crop stream's existing
ledger. A malformed/null rolling declaration cannot fall back to top-level
paths. Supported explicit top-level declarations require
`source_layout=rolling_clips`. Large videos retain the existing cheap stat
identity policy; this is not a new content-hashing or scientific-review claim.

The implemented producer interface maps `RecordingAnalysisPlan.recording_layout`
from default `single_video` to `clipped_video_collection`, with `cam_video=None`
for the latter. A current organized clipped manifest instead declares
`source_layout=rolling_clips` and `rolling_clip_streams`, with no single-video
`video_streams` declaration. Its three paths are
`derived/recording_frame_index/recording_clip_index.json`,
`derived/recording_frame_index/recording_frame_index.parquet`, and
`derived/recording_frame_index/recording_frame_index_manifest.json`.
Truthful `output_kinds` is `['full']` or `['crop', 'full']`; optional H5 is bound
by explicit `h5_relative_path`. No schema/identity backfill or historical
archive upgrade is part of this work.

The ordinary importer creates the strict current identity root, uses the
existing canonical collection metadata/authority publisher, publishes the
recording-wide clock through the indexed helper, and uses existing stream and
crop-ledger owners. It does not invoke the legacy clipped shell, choose a
first-clip stand-in, create materialized full pixels, copy sibling dish/setup
metadata, or manufacture absent H5/calibration/subject information. Existing
sealed import receipts cannot be overwritten. An actively mutated H5 setup
handle now explicitly uses `use_consolidated=False`.

Truthful current rolling declarations require full and optionally crop;
legacy manifests lacking the new explicit current declaration retain their
existing both-stream inventory behavior. Strict current preflight verifies
camera, row counts, projected clip roles/intervals, declared files, and full
index/output agreement. Existing collection, clock, and crop validators retain
their scientific checks. `resolve_acquisition_manifest_file` is the shared
path mechanic: default manifest/H5/projection paths must be relative, while
`allow_absolute=True` is used only for established clip-index row file fields.
Both forms require a regular file confined to the exact recording, including
after symlink resolution. Absolute mapped index rows remain compatible with
existing frame-index/collection/crop-ledger consumers.

The root organizer will place an explicitly selected original PTP summary at
the existing `raw/ptp_sync_summary.json` location, preserving its bytes. This
branch does not invent a clock-summary field or broaden the clock owner to
search arbitrary upstream/opaque context. Numeric and semantic clock policy
remains unchanged; expected source-locator changes are not digest invariants.

The additive read-only API is
`load_verified_registry_recording_import(*, registry_path: Path,
zarr_path: Path) -> VerifiedRecordingImport`. It opens the Palette Python
SQLite runtime using a correctly escaped `mode=ro` URI, enables connection-local
foreign keys and `query_only`, and reuses the existing mixin's bound-path read
and validators. It never constructs `Registry`, initializes schema, inserts an
admission, or creates a missing database; its connection is closed on exit.

Batch discovery now recognizes explicitly current clipped manifests alongside
the unchanged legacy single-video layout, and propagates layout/`cam_video=None`
through its existing ordinary importer. Root owns session-runner adoption and
the real organizer-to-importer combined integration test.

## Local validation

All pytest commands used `scripts/py` outside the workstation sandbox.

- Final path-corrected 17-module focused suite: **452 passed, 102.62 s**, with
  7 existing Zarr-v3 consolidation warnings. It covers all three new modules
  and existing ordinary/batch importer, stream inventory/intake/backfill,
  clock, receipt, identity authority, registry binding, clipped producer, and
  crop collection ledger modules. The source hashes below identify this
  validated uncommitted implementation.
- Corrected ordinary current-clipped module: **28 passed, 13.89 s**. Actual
  ffmpeg/MPEG-4 encoded 2-by-2-frame clips and real ffprobe exercise full-only,
  full+crop, and explicit optional-H5 imports, with both relative and confined
  absolute index rows. Crop fixtures retain blank rows and per-clip/global
  frame identities. The real producer, clock publisher, receipt sidecar,
  isolated registry, replay and unpatched pixel resolver all run.
- New receipt/admission coverage has 48 cases, including mode/locator/identity
  rejection, source/index mutation, stale/missing consolidated metadata,
  malformed/incorrect clock sources and no first-clip/CSV fallback. New
  read-only wrapper coverage has 5 cases, including an existing path containing
  spaces, DB bytes/mtime preservation, missing DB, schema absence, unbound path,
  stale source, and a constructor-prohibition hook.
- Only checkout git provenance is synthetic for the dirty development tree;
  no scientific or runtime admission validator is patched. The read-only
  constructor hook additionally proves the mutating registry constructor is
  never called. Real batch CLI coverage uses its unchanged no-registry option;
  separate positive paths exercise actual registry finalization/read.
- Final complete suite collection: **11,943 collected, 8.03 s**.
- Passed local static gates: import-layer contracts (2 kept), FPS authority,
  keypoint-motion authority, tail payload receipt, paradigm core authority,
  file-size, explicit Zarr open modes, observed metadata literals, managed
  contract freshness, registry schema reference, `py_compile`, and
  `git diff --check`.
- Existing census owner `scripts/py -m
  fisheye.diagnostics.zarr_storage_census --write` regenerated only the
  production-writer census: one clock writer's line-derived identity moved,
  and function-definition count increased by one. Subsequent `--check` passed.
  The explicit-mode owner automatically tightened its baseline from 216 to
  215 after the active H5 mutation handle was fixed; no baseline was manually
  relaxed or edited.
- Complete isolated registry validation through
  `scripts/py -m fisheye.utils.registry_integrity --registry
  /tmp/pytest-of-delahantyj/pytest-6122/test_clipped_receipt_real_prod0/registry.sqlite`:
  `integrity_check=ok`, zero foreign-key issues, Palette Python 3.11 environment,
  loaded SQLite **3.52.0**. No system sqlite3 executable was used.
- Golden receipt v1 semantic digest remains
  `b232f1ac919c4accf603403f993f0ca3c2f179b845c39acd4b9a610e0cee90cb`;
  full canonical bytes SHA-256 remains
  `be98ba4ddd2f0db2b2cc9db6ca5be754458b047fa05a752373e343acdeb132fb`.

## Base implementation source hashes

SHA-256 after the coordinated path correction, before the lease follow-up:

| Source under `src/fisheye/` | SHA-256 |
| --- | --- |
| `registry/recording_identity_authority.py` | `81d44cf346e3e7810670f5ab1f802645e682bc99df70d1eb9e20c110fb5b58a2` |
| `shared/acquisition_frame_clock.py` | `5d5ef8d030001e8cd8547265d762e99f2b13d4e24883108e5c319b4322f1f26c` |
| `shared/acquisition_video_streams.py` | `650754c2ede9477cd443803caecf87bb7ae7d0175226f335e88a670056745f51` |
| `utils/import_recording_analysis.py` | `e10bb211efeb457ad91bff8210269909eb43da4b206c0fdf5ab5e8ff7bda2f3c` |
| `utils/import_organized_recordings_analysis.py` | `407584c192ba9e439dfd6f7994951ad92a08002991ce295c58ba5fe8b35937c4` |

Unchanged `shared/recording_import_receipt.py` SHA-256:
`c2718ab1ab4b7b7bf8fe1649059f62f3c09d612b2d0f0d9a6a842be2b72ed6d5`.

## Unrun required checks and next work

All 24 remote required checks remain unrun for this uncommitted change:
generated artifacts; import boundaries; file-size ratchet; Zarr open metadata
modes; observed metadata literals; active contract freshness; package and
collection; non-GPU test shards 0–15; and `ci-required`. Local evidence does not
satisfy those remote gates. Non-editable wheel build/install validation is
reserved to required CI; no local environment mutation was performed.

Next: finish root independent review, then root-coordinated commit/push and
successful incoming CI; authorized integration
with organizer/session-runner changes and successful combined CI. No incoming
unrun/failing check may be bypassed by the combined branch. No historical
migration, removal, deployment, scientific acceptance, or production activation
is included. Cosmetic review note: the batch plan display still prints absent
`cam_video` as `MISSING` for a valid clipped plan; root has been notified.
