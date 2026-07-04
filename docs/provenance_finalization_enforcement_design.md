# Provenance Finalization Enforcement Design

<!-- contract-meta
status: proposed
created: 2026-07-04
owner: jeremy
related: docs/provenance_enforcement_roadmap.md,
         docs/palette_cli_narrow_waist_design.md,
         docs/zarr_run_completion_strict_mode_todo.md
-->

## Executive Summary

The mechanism needed for Slice 2 already exists in part. Palette run parents carry a
`palette_completion_epoch` attr, and `palette_completion_epoch >= 1` already means
"strict completion markers": unmarked child runs are no longer trusted complete. The
provenance-enforcement design should extend this existing epoch mechanism instead of
adding a parallel switch.

The key correction from the prompt is scope: the epoch is not store-root scoped in the
current code. It is stamped on each runs-parent group, such as `detect_runs`,
`crop_runs`, or `keypoints_runs`. That parent-level granularity is useful for rollout:
families whose writers are ready can advance to a provenance-required epoch without
breaking unrelated analysis writers.

Recommended direction:

1. Add a new monotonic epoch level, `COMPLETION_EPOCH_REQUIRE_PROVENANCE = 2`.
2. Make `mark_run_complete(...)` the source-of-truth gate: if its `parent_group` has
   epoch >= 2, required run provenance must validate before the completion status or
   `latest_complete` pointer is written.
3. Keep `emit_stage_completion(...)` as a defensive registry gate: an `ok` registry row
   should also refuse a strict-provenance run whose marker/provenance is invalid.
4. Move the CLI provenance builder out of `cli/envelope.py` into shared code so direct
   runner and bsub paths can construct the same payload without going through `palette`.
5. Do not stamp all new parents as epoch 2 until the high-volume writers have been
   updated. Roll out parent-family by parent-family, then make epoch 2 the default for
   new empty parents.

## Current Map

### Completion Markers and Epoch

The completion marker lives in
`src/fisheye/shared/zarr_run_completion.py`.

- The module docstring states that parents with `palette_completion_epoch >= 1` are
  strict and unstamped legacy parents still accept unmarked children for read
  compatibility (`zarr_run_completion.py:1-8`).
- `COMPLETION_EPOCH_ATTR = "palette_completion_epoch"` and
  `COMPLETION_EPOCH_STRICT = 1` are defined at `zarr_run_completion.py:27-28`.
- `effective_legacy_default(parent_group)` reads the parent attr and returns `False`
  once epoch >= 1 (`zarr_run_completion.py:50-54`).
- `require_runs_parent(root, name)` stamps new empty parents with epoch 1
  (`zarr_run_completion.py:66-80`). Existing parents with children are not upgraded
  there; they are intentionally left to the backfill tool.
- `backfill_completion_epoch.py` is the existing upgrade path for historical parents:
  after verification, it stamps `palette_completion_epoch` to epoch 1
  (`backfill_completion_epoch.py:331-378`).

This gives Slice 2 its compatibility story for free: parents below the new
provenance-required epoch keep current behavior; only upgraded parents enforce code
identity at finalization.

### Physical Completion vs Registry Completion

There are two convergence points, and they should have different responsibilities.

Physical Zarr completion:

- `mark_run_complete(...)` writes `palette_run_completion_status = "complete"` and
  `palette_run_completed_at_utc` (`zarr_run_completion.py:107-121`).
- When a `parent_group` and `run_name` are supplied, it also publishes
  `latest_complete` and `latest` on the parent (`zarr_run_completion.py:122-130`).
- It currently accepts no provenance argument and performs no provenance validation.

Registry `ok` completion:

- `emit_stage_completion(...)` is in `src/fisheye/registry/stage_complete.py`.
- For `status == "ok"` and a `run_name`, it resolves the Zarr run group, then calls
  `_validate_completion_run_group(...)` before writing the registry step row
  (`stage_complete.py:316-340`).
- `_validate_completion_run_group(...)` already refuses registry `ok` when
  `is_run_complete_in_parent(...)` says the run is not complete
  (`stage_complete.py:197-208`), then performs stage-array validation
  (`stage_complete.py:210-238`).

The production writers generally do both: write the run, call `mark_run_complete`, then
call `emit_stage_completion` either directly or through a stage-specific helper. For
example, `detect_yolo` writes provenance, calls `mark_run_complete`, and the registry
runner later emits detect step status (`detect_yolo.py:1763-1767`,
`run_detect_with_registry_model.py:553-561`). Subject-mask bsub publishing calls
`mark_run_complete`, then `emit_subject_mask_stage_completion`, which wraps
`emit_stage_completion` (`run_subject_mask_batch_pipeline.py:792-827`,
`subject_mask_registry_status.py:53-68`, `subject_mask_registry_status.py:102-117`).

Design implication: `emit_stage_completion` is not sufficient by itself. If only the
registry gate enforces provenance, a writer can still publish a physical Zarr run as
`latest_complete` before the registry write fails. The source-of-truth check belongs in
`mark_run_complete`; the registry gate should mirror it as defense in depth.

### Existing Provenance Streams

There are two different provenance concepts today.

CLI run provenance:

- `build_run_provenance(...)` is in `src/fisheye/cli/envelope.py:98-115`.
- It records `git_sha`, `git_short_sha`, `git_dirty`, `fisheye_version`,
  `config_hash`, normalized `params`, `input_run_ids`, and `command`.
- `palette detect`, `palette keypoints`, and `palette crop` build this payload and pass
  it into runners only for applied writes (`palette.py:1488-1529`,
  `palette.py:1581-1630`, `palette.py:1719-1813`).
- Current low-level writers stamp that payload as `attrs["cli_provenance"]` when it is
  supplied (`detect_yolo.py:1763-1764`, `detect_keypoints_yolo.py:1203-1204`,
  `crop.py:2001-2002`, `crop.py:3500-3501`).

Stage provenance:

- `build_stage_provenance(...)` and `write_stage_provenance(...)` live in
  `src/fisheye/shared/stage_provenance.py:136-201`.
- Stage provenance is stored as `attrs["provenance"]`.
- It has a useful domain shape: stage name, created time, parameters, inputs, optional
  command/version/git/environment/platform/scheduler/artifacts.
- It is not sufficient as the Slice 2 gate today because many writers omit git fields,
  and it does not consistently include a deterministic `config_hash` over normalized
  parameters. It should remain the stage-domain record; finalization enforcement should
  require the CLI-style run identity payload or a canonical successor of it.

### The bsub/Direct Runner Hole

The roadmap's production hole is confirmed.

- `run_detect_with_registry_model(...)` accepts `cli_provenance: Optional[...] = None`
  (`run_detect_with_registry_model.py:339-363`) and passes it through to `detect_yolo`
  (`run_detect_with_registry_model.py:485-504`).
- Its command-line `main(...)` invokes `run_detect_with_registry_model(...)` without
  constructing or passing `cli_provenance` (`run_detect_with_registry_model.py:630-652`).
- `run_keypoints_with_registry_model(...)` has the same pattern
  (`run_keypoints_with_registry_model.py:455-488`, `run_keypoints_with_registry_model.py:764-795`).
- `scripts/submit_detect_batches_bsub.sh` submits `scripts/py -m
  fisheye.utils.run_detections_batch ...` (`submit_detect_batches_bsub.sh:265-285`).
  That wrapper imports and uses the registry-model detect runner machinery rather than
  the `palette` verb (`run_detections_batch.py:28-36`).
- `scripts/submit_keypoints_batches_bsub.sh` submits `scripts/py -m
  fisheye.utils.run_keypoints_with_registry_model ...`
  (`submit_keypoints_batches_bsub.sh:336-352`).
- `scripts/submit_subject_mask_batches_bsub.sh` submits `scripts/py -m
  fisheye.utils.run_subject_mask_batch_pipeline ...`
  (`submit_subject_mask_batches_bsub.sh:705-725`), and that pipeline finalizes run
  groups directly (`run_subject_mask_batch_pipeline.py:792-827`).

So the waist captures provenance, but production LSF entry points can still produce
complete runs without the CLI provenance block. The minimal closure is to let the
runner/writer path build the same run provenance itself when invoked directly.

## Design Decisions

### 1. Enforcement Hook

Recommendation: enforce in both places, with `mark_run_complete(...)` as authoritative
and `emit_stage_completion(...)` as a registry safety net.

Implementation shape:

- Add a canonical run-provenance attr, for example
  `RUN_PROVENANCE_ATTR = "run_provenance"` with schema
  `palette.run_provenance.v1`.
- Extend `mark_run_complete(...)` with an optional `run_provenance` argument.
- If `parent_group` is supplied and the parent epoch requires provenance, validate the
  explicit `run_provenance` argument or an already-stamped canonical attr before writing
  completion status or parent `latest_complete`.
- If validation fails, leave the run in its prior state and do not publish `latest`.
- Add the same validation to `_validate_completion_run_group(...)` so registry `ok`
  refuses malformed strict-provenance runs.

Tradeoff:

- Enforcing only at `emit_stage_completion` is simpler, but it allows a bad physical
  run to become `latest_complete` before the registry write fails.
- Enforcing only at `mark_run_complete` covers physical completion but gives less
  diagnostic detail in the registry validation telemetry.
- Enforcing both is a small amount of duplication, but it matches the current pattern:
  completion markers are the source of truth and registry finalization independently
  validates the source before recording `ok`.

Open detail: `mark_run_complete(...)` is sometimes called without `parent_group`. The
epoch lives on the parent, so provenance enforcement cannot be decided in that form.
For Slice 2, keep this path compatible but treat no-parent completion as legacy-only:
strict-provenance writers must pass `parent_group`, and `emit_stage_completion` remains
the fail-closed check if an orphan marker is later promoted to registry `ok`.

### 2. Epoch Extension

Recommendation: use a single monotonic completion epoch.

Proposed constants:

```python
COMPLETION_EPOCH_STRICT = 1
COMPLETION_EPOCH_REQUIRE_PROVENANCE = 2
```

Add a helper such as:

```python
def requires_completion_provenance(parent_group: Any) -> bool:
    epoch = _coerce_int(parent_group.attrs.get(COMPLETION_EPOCH_ATTR))
    return epoch is not None and epoch >= COMPLETION_EPOCH_REQUIRE_PROVENANCE
```

Tradeoffs:

- A single monotonic epoch is easy to reason about: epoch 0/unstamped is legacy, epoch 1
  requires completion markers, epoch 2 requires completion markers plus code identity.
- A separate attr such as `palette_require_run_provenance = true` would decouple the two
  guarantees, but it introduces a second policy knob and makes parent state easier to
  misconfigure.

Maintainer call: whether to advance all new empty parents to epoch 2 once the initial
writer set is patched, or to keep `require_runs_parent(...)` stamping epoch 1 globally
and let individual writers opt their parents into epoch 2. I recommend parent-family
opt-in first, then a later default bump when a writer coverage census is green.

### 3. Required Field Set

Recommendation: require a small, code-identity-focused payload at finalization. Do not
try to make the first gate a complete scientific lineage contract.

Hard-required for epoch >= 2:

- `git_sha`: present, non-empty, full commit SHA.
- `config_hash`: present, non-empty SHA256 over normalized final run parameters.
- `params`: present mapping, even if empty.
- `input_run_ids`: present mapping, even if empty for root stages.
- `command`: present, non-empty command or module entry point.
- `fisheye_version`: present key. Its value may be `None` for editable installs only if
  the maintainer explicitly accepts that; otherwise require non-empty. This is the
  current equivalent of the roadmap's `code_version`.

Required to record, but not block initially:

- `git_dirty`: present if git status was available.
- `git_unavailable_reason` / `git_dirty_unavailable_reason`: present when git probing
  fails.
- `git_short_sha`: useful but derivable from `git_sha`.
- Scheduler/job identifiers, host, CUDA/runtime versions (see the captured tier below).

#### Provenance tiers — the gate stays narrow, the record stays rich

The single most important discipline here: **enrich the recorded provenance, do not widen
the finalization gate.** A narrow gate can't break the local/non-cluster path or editable
installs and stays robust; a rich record is what makes runs auditable. Three tiers, and
the table is the contract — the failure mode to guard against is someone later promoting a
field into the blocking tier "for completeness" and silently breaking the local path.

| Tier | Fields | Meaning |
|---|---|---|
| **Value-required (blocks epoch-2 finalization)** | `git_sha`, `config_hash` | The reproducibility identity — *which code, which params*. Missing or empty → refuse to finalize. Keep this set to exactly these two. |
| **Structurally-required (key present, value may be empty)** | `params`, `input_run_ids`, `command`, `fisheye_version` | The key must exist so the record is well-formed, but an empty/`None` value does not block (guards the local path and editable installs). Promoting any of these to *value-required* is a deliberate, separate decision, not a default. |
| **Captured, never blocks (execution context)** | `git_dirty` (+unavailable reasons), `git_short_sha`, **job/submission identifiers**, **system metadata** | Recorded whenever available; absence never blocks. Traceability and forensics, not identity. |

**Job/submission identifiers (captured).** Under LSF the runner reads `LSB_JOBID` and
`LSB_JOBINDEX` from the environment (the index ties a specific recording's run to a
specific array task in the batch submitters). Value is *forensic* — the thread from an
artifact back to the cluster's own records (logs, OOM/preemption, which node). Not in the
gate: a local waist run has no job ID, and requiring it would break every non-cluster
finalization. Capture when present, absent otherwise.

**System metadata (captured, weak-reproducibility).** Host, GPU model, CUDA/driver
version, Python/package versions, environment summary — from the system-metadata
extractor. Mostly forensic context, but the GPU/CUDA parts *can* change numerics, so if two
runs with identical `git_sha`+`config_hash` diverge, this is what reveals the environment
differed. That earns it a place in the record — but never in the gate. See §5 for the
layering consequence of capturing it.

Maintainer calls:

- Should a dirty worktree block epoch-2 finalization? I recommend no for the first
  enforcement pass: require that dirty state is recorded, not that it is clean. Blocking
  dirty trees can be a later policy once cluster and workstation workflows are known to
  keep clean working trees.
- Should package version be hard-required in editable/development installs? I recommend
  requiring the key now and making non-empty version a follow-up after confirming
  `scripts/py` installs reliably expose package metadata.

### 4. Provenance Attr Shape

Recommendation: introduce a canonical `run_provenance` attr for the enforced payload,
and mirror/read legacy `cli_provenance` during migration.

Why not make `cli_provenance` the permanent required attr?

- The provenance must be produced by direct runners and bsub jobs, not just the CLI.
  Keeping a `cli_*` name as the canonical finalization contract makes the data model
  misleading.

Why not use `attrs["provenance"]`?

- `attrs["provenance"]` is already stage-domain provenance with variable shape across
  many writers. It should remain useful, but it is not a reliable code-identity gate
  without a normalization layer.

Migration behavior:

- Writers updated in Slice 2 should pass `run_provenance` into `mark_run_complete(...)`.
- `mark_run_complete(...)` should write `attrs["run_provenance"]`.
- For compatibility with existing palette-run tests and downstream readers, writers may
  continue to mirror the same payload to `attrs["cli_provenance"]` for one transition
  window.
- The validator can accept `attrs["cli_provenance"]` as a legacy alias while emitting a
  warning/detail field, but epoch-2 new writes should produce `run_provenance`.

### 5. Provenance Construction Relocation

Recommendation: move the pure provenance builder from `cli/envelope.py` into shared
code, then have `cli/envelope.py` import/re-export it.

Natural home: `src/fisheye/shared/run_provenance.py`.

Move or wrap:

- `json_ready`
- `stable_json`
- `sha256_payload`
- `git_identity`
- `fisheye_version`
- `build_run_provenance`
- a new `validate_run_provenance` / `normalize_run_provenance` helper

This closes the bsub hole without forcing bsub through the `palette` CLI. Direct
runners can construct provenance in-process with the same semantics as the waist.

#### Layering convergence: capturing system metadata pulls `system.py` down (kills the worst edge)

The captured tier (§3) includes system metadata, whose extractors
(`get_git_info`/`get_gpu_info`/`get_environment_summary`) currently live in
`utils/system.py`. That file is imported by `shared/zarr/schema.py` today — the **single
worst upward layering edge in the codebase** (`shared → utils`, ~75 edges via `system.py`,
per the utils import analysis). Since the provenance builder is moving *into* `shared/`, if
it calls the system extractor while that extractor stays in `utils/`, we would recreate the
exact `shared → utils` edge from a new location.

Therefore: **move the provenance-relevant system extractors into `shared/` as part of this
slice** (alongside `shared/run_provenance.py`, e.g. `shared/system_metadata.py`), and have
`utils/system.py` re-export them during the transition. This is not scope creep — it is the
same relocation utils-Phase-2 already names as its highest-priority move, and provenance is
the natural forcing function for it. Net effect: provenance-done-right removes the codebase's
#1 layering violation as a byproduct.

#### The cluster git-state caveat (make provenance *true*, not just present)

The cluster runs from an rsync'd checkout under `/groups`. Whatever git state that mirror is
in when a bsub job runs is the git state captured — and if the mirror can be detached/dirty/
ambiguous, the `git_sha` recorded on the *production* path may be meaningless, quietly
defeating the point for exactly the runs that most need it. The runner-side provenance
construction must record git state *honestly*: capture the SHA and the dirty/unavailable
reason as they actually are on the cluster node, so provenance never claims a reproducibility
it doesn't have. This is the difference between "provenance enforced" and "provenance
enforced and true." (Confirming what git state cluster jobs actually run in is a
verification item for implementation, not an assumption.)

### 6. Runner and bsub Path

Recommendation: do not make bsub scripts pass a provenance blob. Let the Python runner
construct it at the point where it already knows the selected model, output path, run
name, and resolved input runs.

Examples:

- `run_detect_with_registry_model(...)`: if `cli_provenance`/`run_provenance` is absent
  and the call is not a dry run, build run provenance after registry model resolution and
  before calling `detect_yolo(...)`.
- `run_keypoints_with_registry_model(...)`: same pattern, including selected crop run and
  pose model resolution.
- `crop_detections(...)`: when called outside `palette`, build run provenance from the
  crop plan, source detect/refined-detect run, storage mode, and effective chunking.
- `run_subject_mask_batch_pipeline(...)`: build run provenance before publishing staged
  subject/refined subject-mask runs to the canonical store.

The `palette` verbs can continue to pass provenance explicitly; runners should treat
explicit provenance as authoritative and only synthesize it when absent.

## Proposed Implementation Phases

### Phase 1 - Shared Payload and Validator

- Add `shared/run_provenance.py`.
- Move/re-export the current builder from `cli/envelope.py`.
- Define canonical attr names and a validator for required epoch-2 fields.
- Add unit tests for normalized config hashes, missing required fields, and legacy
  alias handling.

### Phase 2 - Completion Gate

- Add `COMPLETION_EPOCH_REQUIRE_PROVENANCE = 2`.
- Add `requires_completion_provenance(parent_group)`.
- Extend `mark_run_complete(...)` with `run_provenance=None`.
- Validate before writing completion attrs or parent latest pointers when parent epoch
  >= 2.
- Add defensive validation to `_validate_completion_run_group(...)`.
- Add unit tests proving:
  - epoch 1 parent still finalizes without run provenance;
  - epoch 2 parent refuses completion without run provenance;
  - epoch 2 parent writes completion and latest only after valid provenance;
  - registry `ok` refuses an epoch-2 run with missing/invalid provenance.

### Phase 3 - Patch Production Writers

Update high-volume writers first:

- detect: `run_detect_with_registry_model(...)` and `detect_yolo(...)`
- crop: `crop_detections(...)`
- keypoints: `run_keypoints_with_registry_model(...)` and `detect_keypoints_yolo(...)`
- subject masks: `run_subject_mask_batch_pipeline(...)`, `run_sam_subject_masks(...)`,
  and subject-mask registry helpers as needed

Each writer should either receive `run_provenance` from `palette` or synthesize it when
called directly.

### Phase 4 - Parent-Family Opt-In

- Keep `require_runs_parent(...)` stamping epoch 1 by default until writer coverage is
  confirmed.
- For patched production writer families, explicitly upgrade new empty parents to epoch
  2 or pass an opt-in epoch to `require_runs_parent(...)`.
- Run a read-only census over active stores to identify parent families whose current
  writers emit valid run provenance.

### Phase 5 - Default Bump

After the writer census is green for normal production families:

- Change `require_runs_parent(...)` to stamp epoch 2 on new empty parents.
- Keep historical parent backfill separate. Do not upgrade existing parent groups to
  epoch 2 unless their completed children already carry valid run provenance or have
  been intentionally backfilled.

## Backward Compatibility

- Unstamped parents and epoch 1 parents keep current completion behavior with respect to
  provenance. They may still require explicit completion markers if epoch >= 1, but they
  do not require run provenance.
- Existing historical runs are not retroactively invalidated.
- Existing `cli_provenance` readers keep working during the migration window if writers
  mirror the canonical payload there.
- New direct-runner outputs become auditable without requiring operators to use the
  `palette` CLI.
- Epoch-2 rollout is parent-local, so one under-instrumented analysis family does not
  block enforcement for detect/crop/keypoints.

## Maintainer Calls Before Coding

1. **Epoch policy:** approve a monotonic epoch 2 for provenance, or choose a separate
   provenance-required attr. Recommendation: epoch 2.
2. **Initial rollout scope:** decide whether Slice 2 should opt in only detect/crop/
   keypoints first, or include subject masks in the first implementation. Recommendation:
   include the high-volume bsub families: detect, crop, keypoints, subject masks.
3. **Canonical attr name:** approve `run_provenance` as the enforced attr with
   `cli_provenance` as migration alias. Recommendation: yes.
4. **Required field strictness:** decide whether dirty git or missing package version
   blocks finalization. Recommendation: require the fields/state to be recorded, but do
   not block dirty trees or editable installs in the first pass.
5. **Default parent epoch:** decide when `require_runs_parent(...)` should stamp epoch 2
   by default. Recommendation: only after the patched-writer census is green; before
   that, opt in patched parent families explicitly.

## Acceptance Shape for the Implementation Slice

The implementation should be considered complete only when:

- An epoch-2 parent cannot publish a run as complete without valid run provenance.
- A registry `ok` row cannot be emitted for an epoch-2 run with missing/invalid
  provenance.
- Direct runner invocations produce the same required provenance fields as `palette`
  verbs.
- Existing epoch 0/1 stores and tests remain compatible.
- At least one real production path that previously skipped `cli_provenance` is covered
  by a direct-runner test.
