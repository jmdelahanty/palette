# Protocol semantic step identity handoff (2026-08-24)

## Objective

Make Palette distinguish a real protocol-level `SOLID_BLACK` step from the
pre/training/post epochs nested inside a `CHASER` step. This must be based on
the exact checksummed Citrus protocol-semantic snapshot, not a recording name,
run-name suffix, step label, or a Palette-derived hash.

This is prerequisite contract work for gaze, trial, escape, and other
composable chaser analytics. A recording with one `CHASER` step has no
standalone black baseline. That is an applicable protocol variant, not missing
metadata and not an error.

## Isolated checkout

- Worktree: `/tmp/palette-protocol-semantic-step-identity-20260824`
- Branch: `agent/palette/protocol-semantic-step-identity-20260824`
- Base: `origin/main` at
  `3aa91cf47e2563dce29c5d51ada6d6ca95e3add2`
- State: uncommitted, selector-ineligible implementation with focused local
  validation; required repository CI remains unrun
- Production data, registries, selectors, and shared `/groups` checkout were
  not modified.

The ordinary source checkout at
`/home/delahantyj@hhmi.org/gitrepos/palette` is on another historical branch
and has an unrelated untracked document. Do not use or clean that checkout for
this task.

Follow the repository `AGENTS.md`: use `scripts/py`, run pytest outside the
sandbox, and do not merge, promote, or deploy before all required CI is green.

## Continuation status later on 2026-08-24

Work resumed from this handoff in the same isolated checkout. The validator,
normal importer, unpublished-run backfill, exact consolidated reload,
corruption handling, registry migration 72, semantic registry extraction, and
durable contract documentation are now implemented in the working tree.
Completed/selector-visible historical runs report
`requires_immutable_successor`; they are not modified in place.

Focused evidence at this checkpoint:

- all 79 tests in `test_import_stimulus_to_zarr_paths.py` passed outside the
  sandbox;
- 24 protocol-contract/backfill tests passed after the final event-mode check;
- the focused registry extraction/backfill/acquisition compatibility suite had
  25 passing tests, followed by a passing explicit v71-to-v72 migration test.

The branch remains uncommitted, required repository CI has not run, and no
production registry, selector, or shared checkout has been changed. Production
activation for historical snapshot-v1 recordings remains blocked;
selector-ineligible successor work can continue under the conservative v1
boundary policy. Citrus snapshot v2 now resolves the producer checksum and
interval-axis questions for future recordings, as described below.

## Citrus snapshot-v2 and sealed-proxy continuation

The Citrus `isolation` branch now provides the missing versioned producer
contract (implementation commit `e467a97`, reviewed checkout at `b7b99c8`):

- an exact producer-authored trial-index SHA-256;
- `/protocol_execution` exact JSON plus SHA-256;
- canonical half-open protocol-step and chaser-phase intervals on
  `stimulus_frame_num`; and
- camera frame IDs explicitly classified as correspondence evidence only.

Palette now validates and materializes those exact v2 bytes, hashes, step
intervals, and chaser phases. It also seals a per-row correspondence proxy from
the imported frame metadata, with exact array/manifest digests and explicit
step/phase membership. That proxy is intentionally suitable only for
visualization and exploratory alignment; it is not an acquisition-row join and
is never selector-eligible.

The remaining exact-join blocker is identity, not merely additional timing.
Citrus has a Shaman-v2 `recording_frame_id` field, but the normal live H5 path
still logs `triggering_camera_frame_id` and has not connected the frame-bound
runtime handoff. The required future chain is:

```text
Citrus stimulus_frame_num
  -> frame-bound Shaman/Orange recording_frame_id
  -> finalized Orange acquisition row (recording_frame_id - 1)
  -> exact Palette recording and camera
```

Additional timestamps remain valuable for diagnostics and uncertainty bounds,
but cannot prove row identity by themselves. Historical v1 recordings are not
silently relabeled as exact; immutable proxy successors remain scientifically
distinct from future exact-mapping successors.

Palette acceptance was strengthened after that implementation: final
consolidated publication now rebuilds every correspondence-proxy array from the
still-open source H5 rows and exact execution intervals, then compares the
dtype, shape, values, content hashes, manifest, and coverage status. Registry
extraction independently reloads the exact execution JSON, step/phase bindings,
proxy arrays, and seal before projecting any v2 fields.

Validation for the completed increment:

- 41 focused protocol-contract, sealed-proxy, selection, and registry tests
  passed outside the sandbox;
- 170 broader semantic import, backfill, registry, acquisition-compatibility,
  semantic-publication, and provider-position tests passed outside the sandbox
  in 215.78 seconds;
- `py_compile`, generated registry schema regeneration, and `git diff --check`
  passed.

Two producer-status cases remain an explicit consumer-design decision rather
than being silently normalized: snapshot-v2 `contract_status=unsupported`, and
an interrupted execution recipe prefix. Supported, completed chaser recordings
use the validated path above. Unsupported or interrupted v2 inputs currently
fail closed until Palette has versioned non-groupable/incomplete run states for
them; they must not be mislabeled `legacy_missing` or `verified-complete`.

## Frame-bound acquisition identity v6 intake (completed 2026-08-25)

The completed acquisition handoff was reviewed against Citrus
`origin/feature/frame-bound-acquisition-identity-v6` at `06ff00a` and Orange
`origin/feature/shaman-v2-recording-identity` at `fd9f6ed`. Both branches are
pushed but are not merged or deployed. Citrus commits through `84a1a3e` add the
producer recording proof while keeping the original v6 wire names.

The v6 declaration closes the one-based/zero-based conversion and row mapping:

```text
stimulus_frame_num
  -> source_recording_frame_id
  -> source_acquisition_frame_index = source_recording_frame_id - 1
```

It also preserves held-target acquisition provenance separately and provides
directed source-camera/final-display homographies. Shaman-v2 ABI revision 3 now
carries the exact `orange.shaman_v2.recording_identity` token in every positive
recording slot. The companion additionally seals three canonical JSON records:

- `orange_recording_identity_json` binds the recording and derived token;
- `acquisition_camera_binding_json` binds acquisition camera ID, serial, and
  the zero-based numeric Shaman ID (including valid ID `0`); and
- `raw_citrus_h5_binding_json` binds the exact closed raw H5 path, size,
  SHA-256, recording/session/observation identities, and finalized observation
  receipt.

Palette now validates those exact records and their dataset checksums in the
standalone v6 validator. Its loader requires the caller's full expected camera
tuple, verifies the per-slot recording token derivation, freezes all evidence,
and advertises `per_slot_recording_identity_token_verified`. Pre-proof v6
companions and semantic substitutions fail closed.

`load_paired_frame_bound_chaser_source()` then resolves only the sealed
recording-bundle-relative raw path, compares the open file handle's size and
SHA-256, requires exact raw/companion `(chaser_index, stimulus_frame_num)` and
coordinate equality, and obtains raw `timestamp_ns_session` through that exact
join. The resulting native sample handle feeds the existing
input-provenance-proxy selector and materializer. The operator CLI accepts
`--frame-bound-companion-h5` plus `--recording-bundle-root`; all camera and
recording expectations remain mandatory. Publication stays immutable,
selector-ineligible, and explicitly classified as controller-input provenance,
not physical display presentation.

The cross-repository validator passed against a golden generated from Citrus
`06ff00a`. The expanded focused suite passed 18 tests, and the broader chaser
source/proxy/relative-frame/distance-successor/position-suite regression passed
98 tests outside the sandbox. Static compilation and `git diff --check` also
passed.

The producer-design blocker is closed. The remaining acquisition gate is the
controlled four-camera hardware run and review of one real complete companion.
Until that succeeds, Palette persists
`controlled_four_camera_hardware_validation_pending`, creates no selectors,
and must not promote or deploy this path. Even after commissioning, the
contract proves logged controller input, not projector light emission.

## Selector-ineligible semantic successor checkpoint

Work continued without those producer additions by implementing
`palette.protocol_semantic_chaser_selection_adapter.v2` as a pure,
selector-ineligible candidate in
`src/fisheye/analysis_workflows/protocol_semantic_chaser_selection.py`.

The candidate:

- binds the exact semantic snapshot, raw step bounds, source stimulus run/path,
  and the immutable source stimulus fingerprint already sealed into the exact
  epoch-v2 selection;
- requires `chaser_pre`, `chaser_training`, and `chaser_post` to be wholly
  contained in the exact producer `CHASER` step;
- rejects legacy `black_before`, `chaser`, `black_after`, and `all_black` keys;
- models a true standalone `SOLID_BLACK` step independently as selected,
  present-but-unselected, or not applicable for a `CHASER`-only recipe;
- for historical snapshot v1, uses the common safe interval
  `[STEP_START, STEP_END)` while that source's producer end convention remains
  unresolved, excluding exactly one uncertain terminal frame when necessary
  and failing on wider crossings; and
- projects independent semantic-window, standalone-baseline, and production-
  eligibility assessments into the strict chaser profile planner.

The resulting planner behavior for a `CHASER`-only recipe is intentional:
semantic CHASER-window analytics are applicable for exploratory candidates,
standalone-baseline contrasts are inapplicable, and production publication is
blocked for review. No maintained profile, selector, registry, or production
artifact consumes the candidate.

Validation at this checkpoint:

- 11 focused semantic-selection tests passed outside the sandbox, including
  future producer-declared inclusive and exclusive end policies;
- 169 combined semantic/import/backfill/registry/v1-compatibility/v2-selection/
  applicability tests passed outside the sandbox in 149.63 seconds;
- `py_compile` and `git diff --check` passed.

The branch is still uncommitted and required repository CI remains unrun.
Snapshot v2 now supplies the producer checksum and canonical interval axis;
production activation still requires the exact live acquisition-row identity
chain and cannot retroactively make historical snapshot-v1 evidence exact.

## Immutable publication and position-suite binding checkpoint

The next selector-ineligible layer is now implemented:

- `read_materialized_protocol_semantic_snapshot()` strictly reconstructs and
  validates the array-backed semantic snapshot from an exact stimulus run;
- semantic evidence and timeline evidence are loader-minted from the current
  direct/consolidated archive and recheck the full source-stimulus logical
  fingerprint; arbitrary caller-constructed timeline records are rejected;
- `protocol_semantic_chaser_selection_publication.py` plans, atomically
  publishes, consolidates, and strictly reloads an immutable hierarchy below
  `analysis/protocol_semantic_chaser_selection_runs/<run>`;
- its manifest and typed arrays bind every role to the source epoch interval,
  source occurrence, producer semantic step index, conservative selected
  bounds, semantic hash, and exact source-selection identity;
- publication is explicitly selector-ineligible, has no production authority,
  does not update the registry, preserves parent metadata, and rejects source,
  timeline, array, manifest, unknown/duplicate role, and rehashed role
  tampering; its reader recompiles the hierarchy from current authorities;
- `materialize_protocol_semantic_chaser_selection.py` provides a revealing
  dry-run/operator entry point and requires an exact epoch-run manifest digest;
  `--apply` is required for immutable publication, and the existing operator
  path remains restricted to the conservative historical-v1 policy; and
- the existing provider-aware chaser position suite now accepts
  `--protocol-semantic-selection-run`. On that path it obtains
  `chaser_pre`, `chaser_training`, and `chaser_post` from the strict handle,
  rejects simultaneous caller role aliases, and carries the semantic source
  binding into the position-suite manifest and run provenance;
- semantic role compilation requires the exact versioned `pre_event`,
  `training_event`, and `post_event` source labels in chronological order, so a
  caller cannot publish a role-swapped hierarchy; and
- position publication distinguishes `protocol_semantic_selection_v2` from
  `caller_bound_legacy_v1`, reopens the exact semantic run during planning and
  publication, and records the within-epoch behavior-contrast scope, local
  trial-index integrity status, and per-role producer step references.

Focused validation for this increment:

- 35 semantic selection/publication/position-publication tests passed outside
  the sandbox (14 adapter tests, 12 publication/operator tests, and 9
  position-publication tests);
- the broader semantic import/backfill/registry/selection/position regression
  set passed 187 tests, and the existing position-suite compatibility subset
  passed 26 tests;
- `py_compile` and `git diff --check` passed.

At that checkpoint, the first bound scientific consumer was deliberately the
existing position-only suite. Gaze, controller-trial, escape/freeze,
generalized bout-response, and full-profile successors still need explicit
semantic source bindings and their own versioned immutable outputs. No
production data were materialized because the historical completed stimulus
runs correctly require acquisition-produced immutable successors.

## Recording-local motion/bout summary successor checkpoint

The provider epoch behavior summary now has a second, explicitly versioned
semantic path. Supplying one exact
`analysis/protocol_semantic_chaser_selection_runs/<run>` source handle makes
the materializer publish schema/method v2; omitting it preserves the existing
v1 computation and publication identifiers.

The semantic-v2 summary:

- requires the semantic selection to bind the same exact epoch-v2 selection as
  the requested temporal source and reloads it with deep source audit;
- computes motion and swim-bout summaries only for `chaser_pre`,
  `chaser_training`, and `chaser_post`, using the selected half-open bounds;
- repeats `analysis_role`, `protocol_semantic_hash`,
  `protocol_semantic_step_index`, and `protocol_semantic_step_ref` on every
  fish, bout, bout-histogram, and inter-bout-interval-histogram row;
- binds one exact provider-motion manifest/read authority and one exact
  selector-ineligible swim-bout lineage/frame axis;
- reopens and recomputes all sources immediately before publication, comparing
  the source-binding digest and every output table; and
- remains selector-ineligible and explicitly labels the temporal mapping as a
  sealed epoch-selection proxy, not physical stimulus-presentation or exact
  acquisition-row authority.

The frozen provider behavior-chain task preserves schema v1 unchanged. A task
that names `protocol_semantic_selection_run` must use task schema v2, and task
v2 requires that exact run name, preventing a legacy task from silently
changing scientific meaning.

Validation for this increment:

- 17 focused provider-summary/chain tests passed outside the sandbox;
- 82 adjacent summary, semantic-selection/publication, cohort-export, and plot
  tests passed with the two existing expected failures unchanged; and
- `py_compile` and `git diff --check` passed.

This completes the recording-local epoch motion/bout summary successor, not the
generalized bout-response product. No analysis Zarr or production data were
written while validating this implementation.

## Semantic cohort export and plot successor checkpoint

The provider epoch cohort exporter and plot reader now version-dispatch the
legacy and semantic contracts end to end:

- input/export/Arrow schema v1 remains the original exact summary-v1 contract;
- input schema v2 requires top-level
  `epoch_binding_mode=protocol_semantic_selection_v2` and, for every recording,
  freezes `protocol_semantic_selection_run`, its immutable manifest SHA-256,
  and the producer `protocol_semantic_hash` in addition to the exact summary
  run;
- v2 refuses summary v1, mixed versions, stale source bindings, stale semantic
  role-record digests, source-window mismatches, and row-level semantic
  identity tampering;
- output/Arrow schema v2 carries the semantic selection run/manifest plus
  `analysis_role`, `protocol_semantic_hash`,
  `protocol_semantic_step_index`, and `protocol_semantic_step_ref` on every
  fish and bout row;
- `epoch_id` remains the exact source window ID, which may be noncontiguous,
  while `epoch_index` is the stable semantic order `0,1,2`; cohort sorting and
  plotting use the semantic order without rewriting source identity;
- the v2 plot reader validates every Parquet row against the immutable
  publication's exact source lineage and renders
  `chaser_pre`/`chaser_training`/`chaser_post` with neutral presentation colors;
  and
- export manifests and plot receipts repeat
  `sealed_epoch_selection_proxy_not_physical_presentation`, so a derived cohort
  cannot be mistaken for exact physical-presentation or acquisition-row
  authority.

The exact v2 input entry extension is:

```json
{
  "recording_id": "<exact recording>",
  "analysis_zarr": "<absolute immutable analysis zarr>",
  "summary_run": "<exact semantic summary v2 run>",
  "track_id": 0,
  "subject_id": "<operator-authored biological subject or null>",
  "protocol_semantic_selection_run": "<exact semantic selection run>",
  "protocol_semantic_selection_manifest_sha256": "<64 lowercase hex>",
  "protocol_semantic_hash": "sha256:<64 lowercase hex>"
}
```

Validation for this increment:

- 27 focused export/plot tests passed outside the sandbox, including a real
  immutable Parquet publication-to-plot round trip;
- 90 adjacent semantic selection/publication, recording-local summary,
  behavior-chain, legacy summary, cohort export, and plot tests passed with the
  two existing expected failures unchanged; and
- targeted Ruff, `py_compile`, and `git diff --check` passed.

No real cohort manifest or cohort generation was published. The implementation
is ready for an operator-authored schema-v2 manifest only after the required
acquisition-produced immutable stimulus successors and selector-ineligible
recording-local semantic summaries exist.

## Read-only cohort evidence

All 84 inspected GoodBatBadBat Citrus H5 files contain and validate as one
contract:

- `/protocol_snapshot/protocol_semantic_hash`
- `/protocol_snapshot/protocol_semantic_json`
- `/protocol_snapshot/protocol_trial_index_json`

The exact semantic JSON bytes hash to the declared semantic hash, and every
trial index binds the same hash. There are two cohorts:

| Recordings | Exact producer semantic hash | Ordered recipe |
|---:|---|---|
| 48 | `sha256:86cb9c18153c8fe5165124093ab1353912de41919e11587ea9439c2fc66a64ab` | `SOLID_BLACK -> CHASER` |
| 36 | `sha256:538cc2e72cd0e03345b54d9ebff035d0f7a6bd9bad7c3ce430fab82639b4b01d` | `CHASER` |

The two-step recipe has a 300 s solid-black step followed by a 1500 s chaser
step. The one-step recipe has only the 1500 s chaser step.

Representative two-step recording:

`/groups/johnson/johnsonlab/jeremy/recordings/2026-08-11T17-39-52Z_arena_1_goodbatbadbat`

Its already materialized step bounds show `step_0` as `SOLID_BLACK`
(camera frames 862--30863) and `step_1` as `CHASER` (frames 30864--180864).

Representative one-step recording:

`/groups/johnson/johnsonlab/jeremy/recordings/2026-08-10T17-20-55Z_arena_1_goodbatbadbat`

Neither recording filenames nor Palette stimulus-run suffixes contain the
producer semantic hash. Do not rename files to compensate.

## Identity model

Keep these identities separate:

1. Recording filename and stimulus-run name: locators only.
2. Existing registry `protocol_hash`: SHA-256 of the full authored
   `protocol_json`; useful but includes non-semantic content and is not the
   producer semantic authority.
3. Producer `protocol_semantic_hash`: calibration-independent protocol recipe
   authority, checksummed over exact `protocol_semantic_json` bytes.
4. Analytics `protocol_signature_hash`: a derived signature of materialized
   steps; useful for analytics grouping but not a substitute for producer
   authority.

Store the full producer hash in provenance and registry fields. A shortened
hash may be used only for display.

## Scientific naming decision

- A standalone black baseline exists only when an exact semantic recipe step
  is classified from producer fields as `stimulus_family=solid_color`,
  `stimulus_mode=SOLID_BLACK`, with black color evidence.
- The windows inside a `CHASER` step are `chaser_pre`, `chaser_training`, and
  `chaser_post`.
- Do not call those nested windows `black_before` or `black_after` merely
  because the chaser may be absent or black during part of its internal state.
- Future consumers must assert that all nested chaser epochs fall inside the
  exact materialized `CHASER` step.
- The existing `composable_epoch_selection_adapter.py` currently uses the
  misleading legacy labels `black_before`, `chaser`, and `black_after` for
  CHASER-internal windows. Preserve existing immutable outputs, but do not use
  those names as proof of protocol-level step identity. Introduce a versioned
  successor rather than silently changing old artifacts.

## Code already written

### New validator

`src/fisheye/shared/protocol_semantic_contract.py`

Provides:

- indivisible modern/legacy detection for the three H5 fields;
- exact `sha256:<64 lowercase hex>` validation over semantic JSON bytes;
- semantic and trial-index schema/policy checks;
- trial-index-to-semantic-hash binding;
- ordered step count, index, mode ID, and duration agreement;
- typed immutable `ProtocolSemanticSnapshot` and `ProtocolStepIdentity`;
- conservative `display_context` classification from producer mode/family and
  color evidence, never from names.

A partial contract fails closed. Complete absence is explicit legacy state.

### Import helpers

`src/fisheye/analysis/import_stimulus_to_zarr.py`

Added:

- `_materialize_protocol_semantic_snapshot`
- `_bind_protocol_semantic_steps`
- optional semantic snapshot binding in `_materialize_stimulus_steps`

The intended Zarr shape is:

```text
analysis/stimulus_runs/<run>/
  attrs:
    protocol_semantic_status
    protocol_semantic_hash
    protocol_recipe_label
    protocol_recipe_mode_sequence
    protocol_recipe_step_count
  protocol_semantic_snapshot/
    protocol_semantic_json_utf8      # uint8 exact UTF-8 bytes
    protocol_trial_index_json_utf8   # uint8 exact UTF-8 bytes
    attrs: bounded recipe/provenance metadata
  steps/step_<i>/
    attrs:
      protocol_semantic_hash
      protocol_semantic_step_index
      protocol_semantic_step_ref
      stimulus_family
      display_context
      resolved_color_rgba8
```

The full JSON documents are arrays, not attrs, to keep consolidated metadata
bounded.

## Earlier unfinished implementation (now completed locally)

The importer/backfill call sites, fail-closed modern/legacy state machine,
array-backed exact snapshot storage, semantic step binding, migration 72,
registry extraction/backfill, durable contract, focused fixtures, immutable v2
selection publication, and the first position-suite consumer binding are all
implemented in this working tree.

The remaining work is no longer basic semantic ingestion or producer-contract
design. It is:

1. run the controlled four-camera Orange/Citrus acquisition and validate one
   real complete raw-H5/v6-companion pair without changing Shadow's v1 default;
2. after producer and Palette required CI are green, review deployment of the
   producer branches and retain the hardware gate in Palette until the real
   companion passes;
3. run an operator dry-run of the frame-bound proxy and semantic selection
   materializers against that recording, review the exact window IDs and
   hierarchy, and publish only selector-ineligible candidates until scientific
   review;
4. run the provider-aware position, distance, motion/bout, cohort, and plot
   paths with the exact semantic selection and frame-bound proxy sources, then
   review the scientific tables and figures;
5. execute the now-versioned controller-trial, generalized bout-response,
   escape/freeze, gaze, and v4 full-profile successors against one exact real
   recording; create the gaze convention receipt only after reviewing its
   bounded eye/body-frame panel; and
6. run all required CI before describing the branch as merge-ready or changing
   any maintained profile, selector, registry authority, deployment, or shared
   checkout.

## Registry result and consumer follow-up

Migration 72 and the registry extractor now provide nullable, explicit
`protocol_semantic_status`, `protocol_semantic_hash`, bounded recipe metadata,
and semantic per-step fields while preserving the old authored-protocol hash.
Any registry acceptance must still use Palette's Python SQLite runtime; the
system `sqlite3` CLI is not acceptance evidence.

The new versioned composable selection contract now represents two levels:

```text
protocol recipe
  optional standalone SOLID_BLACK step
  CHASER step
    chaser_pre
    chaser_training
    chaser_post
```

Selection uses exact semantic step identity and materialized frame bounds. It
never infers a standalone baseline from a legacy epoch role name. Remaining
consumer successors must carry the immutable selection run identity rather
than re-derive this hierarchy.

## Composable chaser successor checkpoint: 2026-08-25

The previously unversioned successor set is implemented locally:

- `controller_trial_successor.py` groups exact nonnegative logged trial IDs per
  chaser, preserves separate exact-member and first-to-last-envelope row IDs,
  trigger rows, and ordinals, and prohibits inferred fallback segmentation.
  Each envelope gap has a primary reason code (selection, chaser occurrence,
  controller-state validity/inactivity, or trial-ID availability/mismatch).
  Logged-active rows lacking trial identity remain explicit unresolved evidence
  and never become inferred members;
- `generalized_bout_response_successor.py` emits one selected-bout-by-chaser
  row set plus semantic-role/chaser/distance-band valid-time summaries. Its
  position/motion base survives without body orientation, while directed turn
  fields are an optional body-frame extension. Bout onset carries both exact
  trial membership and envelope/gap evidence, but attaches only through the
  exact membership mapping;
- `escape_freeze_successor.py` emits separate speed-escape and optional
  high-turn evidence, per-event and per-trial latency/gain/recapture/freeze
  facts, threshold sweeps, and recording reductions without dropping event
  counts when a trace is unusable. Trial summaries carry envelope/gap counts,
  while event attachment and valid-time accumulation exclude gap rows;
- `gaze_tracking_successor.py` compares only anatomical-left-positive gaze and
  chaser bearing in the fish body frame, producing frame/eye/chaser facts,
  tracking summaries, and contiguous lock intervals; and
- `full_chaser_profile_successor.py` binds the normalized v4 full profile,
  applicability plan, dependency/concurrency graph, and exact immutable module
  products without claiming completion for blocked or merely planned modules.

All five products share one immutable selector-ineligible publication and
strict direct/consolidated loader. Exact archive adapters now require one
relative-frame recording and acquisition axis, semantic-selection identity,
provider-motion track, controller-trial dependency, and selector-ineligible
swim-bout signal. The eye adapter additionally requires compact-v7 validation,
the exact 41-array logical digest, and a self-digesting human review receipt
bound to the rendered convention panel. The selected raw/smoothed gaze fields
and freeze-speed level are recorded explicitly in successor provenance.

Thirty-seven focused successor, source-handle, adapter, and immutable
publication tests pass locally at this checkpoint. The 120-test successor,
profile/applicability, and semantic selection/publication union also passes
after the gap-evidence addition (37 focused successor tests, 12 isolated
semantic-publication tests, and 71 remaining protocol/profile tests). Static
compilation, Ruff, and `git diff --check` pass for the changed successor
surfaces. This is implementation evidence only: required CI is still unrun, no
real four-camera recording has exercised the graph, no successor was promoted,
and no selector, registry, deployment, or shared checkout was changed.

## First successor operator trial and frame-bound v2 consumer: 2026-08-25

`fisheye.utils.materialize_composable_chaser_successors` now provides one
dependency-aware entry point for controller trials, generalized bout response,
escape/freeze, and gaze. It is no-write by default. `--apply` can publish only
immutable selector-ineligible products and never changes a selector, registry,
or production authority. A missing optional gaze source does not block the
controller/bout/escape chain; a missing dependency blocks only its consumers.

The first real no-write eligibility receipt is:

`/tmp/composable_chaser_trial_20260825_v1.eligibility.json`

It inspected the 2026-08-12 arena-1 GoodBatBadBat archive and verified these
existing sources:

- relative frame run
  `chaser_relative_frame_keypoint_triad_cohort_20260821_v2`, manifest
  `7e26c4efff7d4892c9ad084e92db7dda6beb30ecabf35802492696788db2b511`;
- provider motion run
  `provider_motion_goodbatbadbat_keypoint_triad_talk_20260818_v2`, manifest
  `4316545452cecd278dae73797714509cd53295ae0afe195991ab435642e57110`;
- swim-bout run
  `swim_bouts_goodbatbadbat_keypoint_triad_talk_20260818_v2`, retained for its
  exact downstream adapter.

The result is truthfully `blocked_no_products`: the archive has no immutable
protocol-semantic selection run, and gaze also lacks a reviewed compact-v7
source plus convention receipt. Controller trials are blocked by the semantic
source; generalized bout response and escape/freeze are then dependency
blocked; gaze is independently blocked by semantic and eye evidence. The run
did not write the analysis archive or change any selector, registry, or
production authority.

The Palette snapshot-v2 consumer gap found during that trial is now closed
locally. Semantic selection accepts v2 only when all of the following hold:

- the exact materialized execution-index UTF-8 bytes revalidate and equal the
  execution index in the sealed raw Citrus H5;
- the strict frame-bound companion/raw-H5 pair reloads with the same recording,
  camera, token, file, manifest, and verification digests;
- the stimulus import's exact `source_h5` is that sealed raw H5 and the epoch
  selection has the same recording identity and native frame population;
- every producer-authored stimulus frame in every selected step and CHASER
  phase has a companion mapping; and
- the epoch role envelopes equal the mapped `chaser_pre`, `chaser_training`,
  and `chaser_post` phase envelopes exactly.

The acquisition projection is explicitly
`latest_stimulus_frame_per_source_acquisition_index_v1`, matching the existing
chaser input-provenance proxy. This is necessary rather than hypothetical: the
examined historical raw H5 has 215,987 unique stimulus frames but only 179,885
unique triggering camera frames, so 36,102 stimulus rows reuse a current camera
input. Every native stimulus row remains evidence; only the latest exact
stimulus sample owns the projected acquisition row. Acquisition gaps and reuse
counts are recorded separately. A phase with no acquisition row after that
exact projection fails closed.

The frame-bound reload binding, materialized execution reader, semantic v2
projection, publication revalidation, CLI inputs, and all successor modules
pass the 105-test focused successor/protocol integration suite. Static
compilation and Ruff pass. This is still local implementation evidence:
required CI is unrun, no real v6 companion has exercised the end-to-end loader,
the controlled four-camera run is pending, and the branch is not merge-ready or
production-eligible.

## Receipt-bound cohort plotting checkpoint: 2026-08-25

The first spatial-occupancy cohort attempt exposed an archive-lifecycle cost,
not an occupancy-histogram cost. One analysis-root `zarr.json` is
1,460,024,783 bytes while each completed relative-frame child is about 49.5
MiB and its direct child metadata document is about 54 KiB. Repeated strict
consumers reparsed the root consolidated metadata and the failed worker reached
about 11.5 GiB RSS before plotting. Validation had already happened at
publication, but there was no reusable receipt accepted by the session-time
relative-frame consumers.

The local receipt-bound successor now:

- performs one complete direct-subtree audit of every declared relative-frame
  array and seals the exact child manifest, completion authority, provenance,
  direct metadata inventory, archive/run/recording identity, and Palette
  commit in
  `palette.analysis.chaser_relative_frame.reusable_validation_receipt` v1;
- revalidates the small immutable child metadata generation on reuse and
  content-rehashes only the exact arrays requested by the consumer; it never
  reparses archive-root consolidated metadata during receipt reuse;
- feeds those bounded handles to paired keypoint/detection spatial occupancy
  and detailed trace plots while retaining the existing strict deep-audit path
  for older tasks;
- creates a new spatial-occupancy run name rather than changing the existing
  immutable v1 scientific payload; and
- removes archive-root metadata parsing from execution-time frozen-task
  revalidation. Recording identity is instead checked against the already
  frozen exact archive path/basename, raw-H5 stat binding, and all frozen input
  child-metadata digests.

Every current chaser plot receipt is now a versioned self-contained recipe.
The dashboard, detailed bundle, radial/near-field summary, and paired-provider
spatial heatmaps repeat and digest their exact scientific plotting coordinates
(including distance/radial/CDF/occupancy bin boundaries and escape thresholds),
normalizations, missing-value policy, color limits, provider/epoch order,
figure sizes, subplot layouts, colormaps, line/marker settings, and PNG DPI.
Open-ended distance-bin boundaries are represented as JSON `null`, never a
non-finite JSON value. Existing provider-epoch cohort receipts already retain
their exact histogram bin edges, counts, fractions, denominators, and dropped
value reasons.

The successor of frozen cohort task
`0cb0a8b77d7f77e851b7f6543da5a760c42ab2743949476504eba7afd7dced8b`
is:

`/tmp/goodbatbadbat_composable_chaser_cohort_task_receipt_bound_v2_20260825.json`

It contains 84 exact entries, has task digest
`66a4ef4f5bc2f415fa29ab37bb1701cdce567cb3519d22016e5368caf8f06bac`,
and retains selector-ineligible/no-production/no-registry/no-selector safety.
It was derived without reparsing archive roots in 8.10 seconds with 37,864 KiB
maximum RSS; all 84 entries reuse their completed upstream outputs and remain
`resume` only for the new receipts, v2 occupancy successor, and versioned plot
recipes.

This remains experimental implementation evidence. A commit-pinned
high-metadata canary, required CI, and review of the first generated occupancy
and detailed figures are still required. No selector, registry authority,
production publication, or shared checkout has been changed.

## Acceptance boundary

This branch is not complete, not merge-ready, and not production-eligible.
Before integration it requires completed implementation, focused local tests,
review of immutable-backfill/successor behavior for existing recordings, and
all repository-required CI checks passing.
