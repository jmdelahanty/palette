# Provenance chain audit (2026-07-24)

<!-- contract-meta
status: diagnostic_snapshot
code_revision: b14dc8e3
authority: evidence and recommendations only; not a runtime contract
-->

A six-lane read-only audit of how provenance actually behaves in this repository,
assessed against practice in distributed scientific / HPC pipelines. Nothing was
modified. Companion to `docs/diagnostics/codebase_review_2026-07-01.md` and
`docs/manual_add_row_propagation_design.md` (the 2026-07-08 add-row audit, whose
conclusions this confirms and extends).

For the tools this audit measures against — Nextflow, Snakemake, DVC,
Pachyderm, DataLad, MLflow, W3C PROV — and a mapping of which palette mechanisms
duplicate them (and which genuinely do not), see
`docs/workflow_provenance_prior_art.md`.

## How to read this

Claims are marked:

- **[V]** — verified directly during this session against `/nvme1/recordings` or
  by executing the code path. Reproduction commands are given inline.
- **[A]** — reported by an audit agent from code reading, with a file:line
  citation, **not** independently re-executed here.

That distinction matters more than usual in this document, because Root Cause 1
means a class of `[A]` claims of the form *"attribute X is absent"* may actually
be *"attribute X is invisible to the reader that looked."* See the
Re-verification section at the end for which ones.

Line numbers are as of `b14dc8e3` on branch `sun`.

---

## Verdict

The provenance **design** is good. The provenance **system** is not, and the gap
between those two statements is the finding.

Very little on this list is a design error. `shared/row_source_signature.py`,
`shared/pose_model_schema_binding.py`, `analysis_workflows/materializers/atomic_run_publisher.py`,
and `shared/tabular_deltas.py` are written to a standard that would pass review
at an organization that does data lineage for a living. The strict completion
epoch was the right idea. Disabling WAL on an NFS-hosted SQLite registry
(`registry/db.py:1233-1236`) is correct and correctly reasoned.

What exists is a **specification-to-runtime gap that nothing detects**, because
provenance here is a deliverable rather than a dependency. No routine operation
breaks when provenance is absent, stale, or wrong. Consequently at least five
separate mechanisms have decayed to zero effect without producing a single red
signal.

Sixty-odd individual findings came out of the sweep. They reduce to four root
causes. Chasing the sixty is how this gets worse.

---

## Root cause 1 — `consolidated_metadata` is a stale read cache over the provenance layer

**[V] This is the highest-severity finding and it was not in any agent report.**

Zarr's root `zarr.json` may carry a `consolidated_metadata` block: a snapshot of
every child group's attributes taken at consolidation time. Reading a child
*through the root* serves that snapshot. No attribute writer in this repo
refreshes it — `shared/zarr_run_completion.py` contains no consolidation call at
all, and neither does `utils/backfill_completion_epoch.py`, which correctly opens
with `use_consolidated=False` for writing (`:813`) and thereby writes exactly
where the readers cannot see.

Census over `/nvme1/recordings` (113 analysis stores):

| Measure | Count |
|---|---|
| Stores carrying `consolidated_metadata` | 104 / 113 |
| Top-level run-parent groups examined | 1404 |
| Groups whose on-disk attrs **disagree** with the snapshot | **988** |
| Groups absent from the snapshot entirely | 0 |

Attributes currently invisible through the consolidated path:

| Attribute | Groups affected |
|---|---|
| `palette_completion_epoch` | 884 |
| `authoritative_run` | 104 |
| `source_video_fingerprint` (+ `_payload`, `_strategy`, `_size_bytes`, `_mtime_ns`) | 104 each |

Reproduction:

```python
import zarr
p = '/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr'
zarr.open_group(p, mode='r')['detect_runs'].attrs
# {'latest': 'detect_2026-02-09_10-10-20'}
zarr.open_group(p + '/detect_runs', mode='r').attrs
# {'latest': 'detect_2026-02-09_10-10-20', 'palette_completion_epoch': 1}
```

**Split-brain read path.** 183 of 430 `open_group` call sites in `src/` pass
`use_consolidated=False`; the remaining ~247 inherit the stale cache. 621 sites
reach run parents by indexing off an already-opened root
(`root["detect_runs"]`-style), versus 1 that opens a `*_runs` path directly.
Whether a provenance attribute exists therefore depends on which idiom the
caller happened to use.

### What this disarms

`shared/zarr_run_completion.py:59-60` and `:66-67` both read
`palette_completion_epoch` off `parent_group.attrs`:

```python
epoch = _coerce_int(getattr(parent_group, "attrs", {}).get(COMPLETION_EPOCH_ATTR))
return not (epoch is not None and epoch >= COMPLETION_EPOCH_STRICT)          # :59-60
return bool(epoch is not None and epoch >= COMPLETION_EPOCH_REQUIRE_PROVENANCE)  # :66-67
```

An invisible epoch means both return the permissive answer. Verified against a
real store: `describe_run_parent(root['detect_runs'])` reports
`completion_epoch: null, legacy_default: true`, and emits the
`zarr_run_completion.py:439` RuntimeWarning about treating unmarked children as
legacy-complete — on a parent that has `palette_completion_epoch = 1` on disk.

Two consequences, both of which an agent found as independent top-severity items
and which are in fact this one bug:

1. **The fail-closed completion gate is running fail-open store-wide.** Unmarked
   child runs are trusted as complete on 884 groups. `require_runs_parent`
   (`:100-105`) stamps the epoch only on parents with no children, so the
   backfill was the only route to arming this, and its writes are invisible.
2. **`run_provenance` records exist in 0 of 4052 provenance-bearing `zarr.json`
   files [A]**, because `_validate_or_record_run_provenance` only enforces when
   `requires_completion_provenance(parent_group)` is true (`:64-68`, `:137-140`),
   which requires reading epoch ≥ 2.

It also universalises what was reported as a conditional degradation: with
`authoritative_run` invisible on 104 groups, `AUTHORITATIVE` resolution silently
falls back to `LATEST_COMPLETE` (`shared/run_resolution.py:279-291` [A]).

**Elapsed time undetected:** the backfill ran 2026-06-13; the strict resolver
branch landed 2026-07-20 (`93177ed5`). Neither event produced a signal, because
nothing depends on the gate being on.

### Fix

**[V] Forbid consolidated reads. Do not attempt to keep the snapshot fresh.**

Measured on
`2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr` (5316 nodes on disk,
5217 in the snapshot, root `zarr.json` = **6.6 MiB**):

| Operation | Consolidated (default) | `use_consolidated=False` |
|---|---|---|
| Open root (NVMe) | 311 ms | **39 ms** |
| Open + read attrs of 12 run parents (NVMe) | 335 ms | **39 ms** |
| Same, on an NFS `/groups` store (3.8 MiB root) | 88 ms | **79 ms** |

**Consolidation is currently 8.6× slower on NVMe and marginally slower on NFS.**
It is a round-trip optimization for high-latency object stores read
tree-exhaustively; this store is on local NVMe and readers touch a handful of
groups, so the 6.6 MiB JSON parse dwarfs the reads it avoids. Bypassing it is
both faster and correct.

Re-consolidating after each stage does **not** work, for three reasons:

1. **The needed atomicity is not on the file write.** Reconsolidation is
   `walk tree → write snapshot`. Measured walk cost on this store: **~660 ms**
   (600 ms `rglob` + 58 ms to read 11.1 MiB of node metadata). Stage A walks,
   stage B creates a group mid-walk, stage A atomically writes a snapshot that
   omits B. The write is perfectly atomic and the content is still wrong.
   Correctness requires a store-wide lock held across the full walk — over NFS,
   where `flock` is lease-based and lock loss is undetected (see C4).
2. **It converts a deterministic bug into a data race.** Today's breakage is a
   one-time backfill mistake: reproducible, findable, fixable once. Per-stage
   reconsolidation makes the loser of a concurrent walk *invisible* rather than
   merely un-epoched, intermittently. This is a serialization point placed
   exactly where the pipeline's design assumes parallel per-stage LSF jobs on one
   store.
3. **A cache with no invalidation protocol is only safe over immutable data.**
   Zarr's `consolidated_metadata` carries no generation counter, ETag, or
   snapshot timestamp, so a reader cannot detect staleness — and validating the
   snapshot would require exactly the per-node reads consolidation exists to
   avoid. It is inherently trust-or-bypass.

**Correct end state, in order:**

- **Now:** `use_consolidated=False` on every attribute read. Faster and correct.
- **Structural:** separate mutable from immutable. Completed runs never change;
  only the selector pointers (`latest`, `latest_complete`, `authoritative_run`)
  mutate — and they currently live in run-parent attrs, i.e. *inside* the
  snapshot, which is exactly why it goes stale. Move pointers to the registry or
  a small always-read-fresh sidecar and the mutable surface approaches zero.
- **Endgame:** consolidate **sealed** stores only — at cohort release or
  archival, when the store is frozen and marked read-only. That is what
  consolidated metadata is for. Natural home:
  `docs/cohort_release_workflow.md`. Combined with content-addressed run names
  (see `docs/workflow_provenance_prior_art.md`), immutability becomes structural
  and the snapshot cannot go stale by construction.

`shared/zarr_helpers.py:317` already exposes
`consolidate_metadata_capture_expected_warnings` (used at
`analysis/import_stimulus_to_zarr.py:1450`); it is the right tool for the sealed-store
step and the wrong tool for a per-stage hook.

---

## Root cause 2 — built, tested, documented, unwired

The dominant structural pattern. Each of these is correct code with no
production caller.

| Mechanism | State | Citation |
|---|---|---|
| `shared/tabular_deltas.py` — append-only edit log, per-row `revision`/`timestamp_ns`/`editor`, immutable writer partitions, generation freeze with aggregate digest | **Zero production callers.** Only importer repo-wide is `tests/unit/fisheye/test_tabular_deltas.py:9` | `shared/tabular_deltas.py:1-391`; admitted at `docs/tabular_delta_compaction_contract.md:206-210` |
| `audit_analysis_staleness` — the general-purpose staleness engine | **Structurally cannot return `stale`.** Compares `expected_fingerprint` (from the `source_fingerprints` attr) against `actual_fingerprint` (from `lineage_hash` on the source). No pipeline stage writes `lineage_hash`; only 2 writers repo-wide emit `source_fingerprints`, neither in `RUN_PARENT_SPECS`. Every edge falls through to `unverifiable_missing_expected_fingerprint` | `utils/audit_analysis_staleness.py:455`, `:270`, `:373`, `:474-476`, `:510`; writers at `training_image_profile.py:696`, `detection_profile.py:954` [A] |
| — and nothing calls it | Sole importer is `utils/inspect_run_lineage_graph.py:17`, which itself has zero importers | [A] |
| `cluster/lsf/inspect.py` — operational status readout | **Does not exist.** Specified at `docs/lsf_submission_framework_design.md:411`. `cluster_job_watch` / `cluster_job_summary` (`docs/cluster_job_dashboard_direction.md:51-61`) also absent | [A] |
| `verify_deployment_artifact_content` — re-hashes a model against the registry hash | Correct, and called by **neither** model resolution path | `registry/model_resolution.py:97-164` defined; `:411-493` and `:618` are the resolution paths that skip it [A] |
| `utils/resolve_latest_registered_model.py` — hard-fails on digest drift, refuses profile reordering | Only callers are its own `main()` and its unit test | `:33-47`, `:62-63`, `:69-72`, `:74-85` [A] |
| `analysis_workflows/materializers/atomic_run_publisher.py` — flock lease, staging copy, inventory verification, atomic directory `os.replace`, owner-UUID rebinding, tombstones, rollback receipts | Wired, but **only to ~8 analysis materializers**. Does not cover detect, crop, keypoints, tracking, subject masks, or any cluster inference stage | `:630-648`, `:765` [A] |
| `keypoint_stale.mark_downstream_eye_mask_runs_stale` | Zero callers outside its own module; targets the deprecated eye-mask stage | `shared/keypoint_stale.py:82`, `:66` [A] |
| LSF status/bundle/submission evidence (`palette.lsf_job_runtime_status.v1` etc.) | Write-only. Grep across `src/ scripts/ tools/ apps/` returns only definition sites | [A] |

**Why this matters more than any individual item:** unwired-but-documented
machinery is worse than absent machinery, because it manufactures false
assurance. `docs/derived_analysis_run_contract.md:300-302` calls
`audit_analysis_staleness` "a gate." It is not.
`docs/manual_add_row_propagation_design.md:110-111` (three months newer) is the
accurate description: *"No queue, no daemon, no dispatcher exists in the pipeline
path… Nothing acts on them."*

---

## Root cause 3 — there is no commit, on the stages that produce the data

**[A]** A stage declares "done" via three independent, non-atomic surfaces:

1. **Child attrs** — `shared/zarr_run_completion.py:194-243`. `mark_run_complete`
   writes `palette_run_completion_contract`, `_status`, `_completed_at_utc`,
   `_name` as **four separate whole-file rewrites** of the child `zarr.json`
   (zarr 3.1.3 has no attrs batching; each `attrs[k] = v` rewrites the file).
2. **Parent selector attrs** — same function, `:233-242`. Then `latest_complete`,
   then `latest`, then delete `latest_pending`: three more separate rewrites.
3. **Registry row** — `registry/stage_complete.py:326-451`, always last, and
   **explicitly non-fatal**: the whole body is wrapped in
   `except Exception: return False` at `:441-448`. 13 of 18 call sites discard
   the return value.

Each of the seven file writes is individually atomic (zarr's `LocalStore._put`
does temp-file + `os.replace`). None are atomic with respect to each other, and
none are atomic with respect to the array data they describe. Arrays are written
by direct `array[slice] = data` with no journal, no staging, no barrier. Nine
`os.fsync` calls exist in the whole tree; none on a zarr chunk, none on a
directory after `os.replace`.

**The dominant steady-state inconsistency is W4: zarr complete, registry silent,
nobody told.** `zarr_mtime_ns` is written into `recording_step_status` and has no
consumer that checks it for drift.

Enumerated crash windows [A]:

| # | Sequence | Resulting state | Detected? |
|---|---|---|---|
| W1 | `create_group` → kill before `mark_run_started` (`analysis/detection_occupancy_runs.py:752-754`; same shape `tracking/crop.py:4193-4194`, `refinement/detect_quality.py:578-579`) | Empty contract-less child. With `legacy_default=True` (now universal — Root Cause 1), `is_run_complete` returns **True** (`zarr_run_completion.py:311-312`), and reverse-lexical scan (`:447`) prefers it because run names are timestamps | No |
| W2 | Kill between `latest_complete = new` and `latest = new` (`:236-237`) | Strict readers fail closed (`:415`); legacy readers resurrect the previous run | No |
| W3 | Completion marked, arrays truncated | Array validation is hard-enforced for 7 of ~30 stages (`registry/stage_complete.py:54-64`); everything else is shadow-mode | Registry writes `ok` |
| W5 | `invalidate_downstream_steps` loops, each `upsert_recording_step_status` opens its own transaction (`registry/step_cascade.py:124-150`, `registry/status_ledger.py:159`) | Some downstream steps invalidated, others still `ok` | Only a full `reconcile_dataset_from_root` |
| W6 | `reconcile_dataset_from_root` commits profiles inside `_transaction_context()`, then calls step-status reconcile **after** the transaction closes (`registry/db.py:7242-7278`, `:7290-7298`) | Profiles current, step statuses stale, no report written | No |
| W7 | Cleanup killed mid-`rmtree` (`cluster/clipped_inference_cleanup.py:70-71`, `cluster/whole_recording_analysis_cache_cleanup.py:328-329` raise on already-absent) | Cannot be re-run; requires a human | `clipped_inference_validate.py:267-275` detects, never repairs |
| W8 | Retry after kill | Run names are timestamps → new group, old orphaned; `_refuse_output_collisions` (`cluster/clipped_inference.py:313-348`) then hard-fails the resubmit | The four `*_recovery.py` modules exist for this; all manual, none scheduled, none referenced by any bsub script |

**Retry is refusal, not idempotence.** This is the design point: content-derived
output names would make a retry a no-op. Timestamps make it a manual incident.

### Concurrency [A]

- **C1 — run-name collision is live.** Every default generator is a 1-second UTC
  timestamp with no PID/host/`LSB_JOBID`/UUID:
  `analysis/subject_shape_runs.py:244-246`, `analysis/tail_kinematics_runs.py:177-179`,
  `refinement/detect_quality.py:573-574`, `cluster/keypoints/clipped_collection.py:48-49`.
  `detection/detect_yolo.py:539-547` is a textbook TOCTOU. The `"already exists"`
  guards (`subject_shape_runs.py:1055`, `tail_kinematics_runs.py:1412`,
  `finalize_subject_masks.py:5106`, `detect_quality_collection.py:895`) are
  check-then-create, and both racers proceed to `create_array(..., overwrite=True)`
  into the same group — silently interleaved arrays, not an error.
- **C2 — `parent.attrs["latest"] = ...` is an unlocked read-modify-write** of a
  shared JSON file at 20+ sites.
- **C3 — chunk-level partial writes.** `analysis/tail_kinematics_runs.py:535-559`
  is the only site that asserts chunk/shard ownership at runtime. Live gap:
  `finalize_subject_masks._parallel_worker_row_chunk_size` (`:2024-2034`) aligns
  workers to `dense_mask_row_chunk = 128` while metric arrays use
  `SUBJECT_MASK_METRIC_ROW_CHUNK = 256`; safe today only because the parallel path
  passes `write_derived_metrics=False` (`:4567`, `:4610`, `:5433`, `:5495`) — a
  convention with no assertion. The correct helper
  (`shared/subject_mask_chunks.py:98-111`) exists and is not called.
  `docs/dask_zarr_write_safety.md:31-38` records that this class already caused
  silent stale metric values once.
- **C4 — `flock` on NFS.** `/groups` is `nfs4 vers=4.1 local_lock=none`, so the
  publisher's lease is server-side. NFSv4 locks are lease-based; neither SQLite
  nor `atomic_run_publisher` detects lock loss. Staging path
  `.{run_name}.publish_tmp.{os.getpid()}` (`:97-100`) is not host-unique.
- **C5 — zarr I/O inside `BEGIN IMMEDIATE`.** `reconcile_dataset_from_root`
  (`registry/db.py:7242`) runs every zarr extractor inside an exclusive write
  transaction; `registry/extractors/masks.py:386` can materialize a whole dense
  `masks_roi` array on the fallback path — multi-GB read holding an exclusive
  lock on a rollback-journal SQLite DB over NFS.

### The NFS + SQLite + no-WAL choice [A]

The reasoning at `registry/db.py:1233-1236` is correct: WAL's `-shm` index is
shared-memory-mapped and coherent only among processes on one host. Enabling it
would corrupt immediately.

What it costs: full serialization (rollback journal, whole-DB granularity);
`busy_timeout = 30_000` (`:50`) with **zero retry logic** anywhere in
`src/fisheye/registry/` despite the comment claiming reliance on "timeout/retry";
past 30s of contention the job dies.

What it does not buy: safety. SQLite's own documentation is explicit that *all*
locking on network filesystems is unreliable, rollback journal included. On disk:

```
palette_registry.sqlite.malformed_before_restore_20260622T050445Z
integrity_check → "Tree 491 page 491 cell 476: 2nd reference to page 6378"
                  "Rowid 5041 out of order"
                  "wrong # of entries in index idx_recording_step_status_status"
```

Duplicate page references and out-of-order rowids are B-tree corruption
consistent with a lost write lock. **The registry has already been corrupted
once in production and restored from backup.** If it is load-bearing, the fix is
a single writer process with the DB on local NVMe replicated out to `/groups`,
not a pragma choice.

---

## Root cause 4 — absence and unreadability are conflated

**[A]** This one has already cost a real analysis (see
`project_canonical_registry` / the "12 fish" bug).

- `registry/db.py:8916` — `_is_zarr_root(path)` is `(path / "zarr.json").is_file()`.
  `is_file()` returns `False` for an unmounted or unreachable NFS path; it does
  not raise. `reconcile_missing_datasets` (`:8689-8707`) walks every non-missing
  row in scope and flips `status='missing'` on that basis, with no
  deleted-vs-stat-failed distinction and no blast-radius abort. Downstream cohort
  resolvers filter on `dataset_status='active'`
  (`analysis/goodcopbadcop_common.py:68`) — silent under-selection.
- `registry/maintenance.py:7900-7903` — a bare `except Exception` around the zarr
  open calls `_build_recording_step_error_rows`, which (`:6348-6375`) emits
  `status="error"` for **every** step in `RECORDING_STEP_NAMES`. One transient EIO
  converts a fully-populated `ok` status set into ~30 `error` rows plus 30 history
  rows. The existence of `_preserve_operator_inferred_calibration` (`:4990-5021`)
  proves the problem is known; exactly one field got an exemption.
- **Reconciliation does not converge.** `upsert_dataset` hardcodes
  `"status": "active"` and its `ON CONFLICT` sets `status=excluded.status`
  unconditionally (`registry/db.py:~2551`, `~2578`). Any `emit_stage_completion`
  resurrects a `missing` row to `active` without consulting the filesystem;
  `reconcile_missing_datasets` re-marks it. Two writers flip the same field
  forever; neither is authoritative.

---

## Lane detail

### Lane A — what provenance actually records

**Two record types, one of which does not exist on disk.** [A]

`palette.run_provenance.v1` (`shared/run_provenance.py:18`) carries `git_sha`,
`git_dirty`, `config_hash`, `params`, `input_artifacts[]` (with `sha256`),
`scheduler.lsf.*`, `runtime.host.*`, optional `system.environment` /
`system.gpu`. **No timestamp, no seed, no container digest, no diff hash.**
`grep -rl --include=zarr.json '"run_provenance"' /nvme1/recordings` → 0 hits
against 4052 files carrying `"provenance"`. Cause: Root Cause 1.

`palette_stage_provenance` v1 (`shared/stage_provenance.py:9-10`) is the record
that exists: `contract.{name,version}`, `stage`, `created_at_utc` (caller-supplied,
unvalidated, `:139`), `parameters`, `inputs` (upstream run **names**), `command`,
`git.{commit,short,branch,is_dirty,remote}` (`:80-92`), `environment`, `platform`,
`scheduler`, and `artifacts` = `{model_path, model_name, ultralytics_version,
device}` — **no hash**.

Reproducibility matrix [A]:

| Dimension | Captured | Note |
|---|---|---|
| Git SHA | Yes | 224/224 runs sampled |
| Dirty-tree flag | Yes | **`True` for 169/224 (75%)** of sampled runs; 16/16 `training_runs` |
| Dirty-tree *content* | **No** | `dirty_files[]` is a ≤50-entry filename list (`shared/system_metadata.py:118-127`); `_code_attrs` reduces it to a bool (`shared/run_lineage_fingerprint.py:225-228`) |
| Container digest | **No** | Zero references repo-wide |
| Lockfile hash | **No** | `environment.yml`, `pixi.lock`, `pip-packages-exact.txt`, `conda-packages-explicit.txt` have **zero** references from `src/`, `scripts/`, `tools/`, `apps/` |
| Model identity | Path only in production | `sha256` lands in `run_provenance.input_artifacts` (Root Cause 1). Flat attrs at `utils/run_detect_with_registry_model.py:254-285` have `_model_path` but **no `_model_sha256`**; same omission `run_keypoints_with_registry_model.py:406-416`, `infer_unet_subject_masks.py:2318-2330` |
| Random seeds | **No, for every inference stage** | `detection/detect_yolo.py:2257` sets `cudnn.benchmark = True` and records it nowhere; `training/train_pose.py:1394` pops `deterministic` from params |
| Inference precision | **No** | See Lane F |
| Timestamps | Mostly tz-aware | 8 naive sites; `refinement/detect_quality.py:705` writes `datetime.now().isoformat()` into a field named `created_at_utc` |

**Content-addressing is nominal where it matters.** Every inter-stage edge is a
name or a path: `provenance.inputs = {"source_crop_run": "crop_2026-02-10_21-05-18"}`;
`source_refs` are zarr paths; `datasets.path_hash` is `sha256(resolved_path)`
(`registry/db.py:574-575`) — nominal identity dressed as a hash.

**≥15 mutually incompatible fingerprint schemes**, 3 algorithms, no composition.
Different canonicalizations (JSON-sorted vs raw C-order bytes vs stat tuples),
different widths (hex sha256 / uint64 / uint8[32]), different homes (zarr attrs /
zarr arrays / sqlite columns / `.content_v1.json` sidecars). No scheme ingests
another's digest. The single attempt at composition —
`_source_fingerprint_attrs` scooping any attr whose *name substring* contains
`"fingerprint"` (`shared/run_lineage_fingerprint.py:195-207`) — is string
matching, and yields `{}` in production.

`shared/row_source_signature.py` is the exception and the model to follow:
spec-digest-pinned, fail-closed on spec mismatch (`:643-646`), explicit
content-vs-revision basis.

**Schema versioning is declared and unenforced.** `normalize_run_provenance`
`setdefault`s the schema id onto any payload (`run_provenance.py:326`);
`_normalize_contract` defaults a missing or garbage version to 1
(`stage_provenance.py:70`). `crop_signature.signature_version` bumped 1→2
(`shared/crop_signature.py:28`) adding 5 keys, with no migration and raw `!=`
comparison at `cluster/keypoints/common.py:502`.

### Lane B — lineage traversal

**[A] Backward trace for `chaser_escape_events` (the strongest published
result): 6 of 8 hops are mechanically walkable.**

| Hop | Carrier | Status |
|---|---|---|
| escape_events → bout_response | `source_bout_response_component`/`_path` (`analysis/chaser_escape_events.py:1119-1122`) | OK |
| bout_response → distance + swim_bout | `source_refs` (`analysis/chaser_bout_response.py:1198-1205`) | OK |
| swim_bout → track_kinematics | `source_track_motion_manifest_sha256` (`analysis/detect_bouts_multi_level.py:3146`) | **the only content hash in the chain** |
| track_kinematics → detect/keypoint/crop/tracking | `source_refs` (`analysis/track_kinematics.py:723-762`) | OK |
| tracking → refined_detect | `source_detect_run`, rowset fingerprint attrs (`tracking/single_subject_per_arena.py:353-374`) | OK |
| refined_detect → detect | `source_detect_run`/`_path` (`refinement/refine_detect.py:1613-1616`) | OK |
| detect → video | `model_path`; `detection_acquisition_frame_mapping` (`detection/detect_yolo.py:3450-3453`, `:3702`) | **degraded** |
| video | `stat_v1` = sha256 of `{path,size,mtime}`, not bytes (`shared/import_source_fingerprint.py:12,42-58`); `relocation_stable: False` | **dark** |

Where it goes dark:

- **The model is not an edge at all.** `model_path` doesn't start with `source_`,
  so `discover_source_refs` (`utils/audit_analysis_staleness.py:314-334`) never
  sees it. It is also excluded from the lineage payload by construction —
  `TRANSIENT_LINEAGE_KEYS` strips `artifacts` (`shared/run_lineage_fingerprint.py:48-69`),
  and model identity lives inside `artifacts`. **Swapping detector weights leaves
  the lineage hash unchanged.**
- **The video is not a node.** `_normalize_internal_path` returns `None` for
  absolute filesystem paths (`utils/audit_analysis_staleness.py:176-182`).
- **`chaser_*` families are outside `RUN_PARENT_SPECS`**
  (`utils/audit_analysis_staleness.py:68-80` vs `analysis/chaser_distance_runs.py:1315`).
  At least 12 families written under `analysis/` are missing from that list. The
  headline scientific result is in the one family with zero staleness coverage
  and zero default graph coverage.

**Invalidation verdict: manual.** Push exists in exactly two narrow places —
`registry/step_cascade.py:65` (scoped by its own docstring at `:12-15` to *new
runs only*, explicitly excluding in-place edits), and
`shared/subject_mask_stale.py:191`, which walks only
`root.get("refined_subject_masks_runs")` (`:217`) and stops — one hop, one edge.
**There is no `source_detect_stale` anywhere in the repo**; a human correcting a
refined detection box propagates nothing. Confirmed as an open gap at
`docs/repo_wide_staleness_gap_matrix.md:40`.

Pull exists in two real pockets: `shared/rowset_fingerprint.py` is a genuine
content identity, enforced by `assert_rowset_fingerprint_matches` at
`tracking/arena_assignment.py:987`, `tracking/single_subject_per_arena.py:221,378`,
`tracking/incremental_crop.py:1531,1966` — covering refined_detect → arena →
tracking → crop, and read at `analysis/track_kinematics.py:12257`. It stops
there. And `source_track_motion_manifest_sha256` is verified on read at
`analysis/megabouts_classifier_inputs.py:445-461`.

**Row lineage dies at the aggregation boundary.** Only 3 of 12+ analysis
families propagate it (`analysis/subject_shape_runs.py:1072-1080`,
`analysis/tail_kinematics_runs.py:70,1447`, `analysis/tail_posture_view_runs.py:630`).
`track_kinematics`, `swim_bout_runs`, and every `chaser_*` component carry none.
Forward row-level tracing is not expressible: every node in the DAG is a *run*
(`utils/inspect_run_lineage_graph.py:35-51`).

**No materialized lineage index.** `build_run_lineage_graph`
(`utils/inspect_run_lineage_graph.py:285`) recursive-descends one archive; there
is no cross-archive traversal at all. A store-wide question means N separate
walks over NFS with no shared index and no incremental refresh.

**The diagnostics do not cover this.** `diagnostics/check_full_provenance.py`
hardcodes `keypoints_runs` (`:71`, not `refined_keypoints_runs`) and the
deprecated `eye_masks_runs` (`:84`); its central assertions (`:135-153`) check
that each stage points at the *current latest*, which flags a
historically-correct pinned run as an error.
`diagnostics/check_provenance_consistency.py` verifies row counts and bbox drift
only, and **`main()` returns `None` (`:717-719`)** — it can never fail a build.
`diagnostics/check_provenance_capture.py` verifies presence, not correctness:
`_has_inputs` (`:112-117`) passes if *any* attr starts with `source_`, so a run
pointing at a nonexistent upstream passes.

**Exported figures have no provenance.** `shared/plot_artifacts.py:137-140`
makes `source_paths`/`source_runs` optional, and zero `savefig` call sites write
a sidecar.

### Lane C — HPC execution provenance

**[A] The kernel is well-built.** `cluster/lsf/` has immutable models, a
validated DAG, topological submit, atomic incremental snapshots, signal
forwarding, and fail-closed scratch cleanup that refuses any path not strictly
under `/scratch/$USER/$LSB_JOBID` (`cluster/lsf/runtime.py:73-115`).
`shared/json_safety.py:101-127` does correct atomic publish
(`NamedTemporaryFile` in target dir → `fsync` → `os.replace`).

Captured: `lsf_plan.json` (`cluster/lsf/models.py:346-355`),
`lsf_submission.json` with real job IDs and exact bsub argv
(`cluster/lsf/submission.py:26-83`), per-task `status/*.json` with
`LSB_JOBID`/`JOBINDEX`/`HOSTS`/`CUDA_VISIBLE_DEVICES`/hostname/start/finish/returncode
(`cluster/lsf/runtime.py:49-59`, `:118-159`). And — better than typical —
`scheduler.lsf.*` plus `runtime.host.hostname` land **unconditionally** in
`run_provenance` (`shared/run_provenance.py:145-180`, `:207-222`, `:262-266`),
which is a real artifact↔job join.

Lost:

- **Resource usage.** Zero `bjobs`/`bacct`/`bhist` calls in `src/`. No max RSS,
  no queue wait, no `TERM_MEMLIMIT`/`TERM_RUNLIMIT`. "Did this OOM?" is
  unanswerable.
- **Git commit in the plan.** `grep -rn 'git_commit|git_sha|git rev-parse' src/fisheye/cluster/`
  → **zero hits**, despite `docs/lsf_submission_framework_design.md:344-346`
  requiring it. `plan.json` records `repo` as a path to a mutable checkout
  (`cluster/clipped_inference.py:62`, `:132`).
- **Plan digest.** Required at `docs/lsf_submission_framework_design.md:370`;
  `_submission_payload` (`cluster/lsf/submission.py:39-57`) has no such field.
- **Back-pointer artifact → submission.** `run_provenance` has `LSB_JOBID` (which
  recycles) but no `workflow_id`, `job_key`, `run_root`, or plan path.
- **Retry/attempt number.** No concept exists.

**F2 — a hard kill leaves a permanently "running" status.**
`cluster/lsf/runtime.py:239-241` traps SIGINT/SIGTERM only. On
`TERM_MEMLIMIT`/`TERM_RUNLIMIT`, LSF SIGKILLs after a short grace; the status
JSON stays `status:"running"`, `finished_at_utc:null` forever. Combined with
"nothing polls LSF," an OOM-killed array element is invisible.

**F3 — partial fan-out failure is byte-identical to success.** 20 of 22 clip
detects succeed → 20 complete `detect_runs` groups, each setting its clip-local
`latest`/`latest_complete`, and two simply absent. **Nothing in the zarr encodes
"22 expected"** — the count lives only in `run_root/plan.json` and at
`cluster/clipped_inference.py:605`. The `experiment_index/finalized_runs/<id>`
collection that would record coverage is written by the job that never runs.
Bundle mode is worse: `cluster/lsf/task_group.py:158-187` lets already-running
siblings finish and **publish** after the first failure, so `detect_refine`
(22 tasks, `max_concurrent=4`, `cluster/clipped_detection.py:389-404`) leaves a
*nondeterministic* number of complete refined-detect groups in the canonical
zarr. Recovery is blocked from both directions
(`cluster/clipped_inference.py:313-348`,
`cluster/clipped_inference_detect_quality_recovery.py:246-277`) and **there is no
cluster-side tool to remove partial refined-detect groups.**

Other unrecorded partial modes: `cluster/clipped_inference_registry_finalize.py:47-61`
(bare loop, no transaction); `cluster/whole_recording_analysis.py:581` (a single
validate job depends on all finalizers — one failure means N-1 complete targets
are never validated); `cluster/keypoints/registry_finalize.py:343-344` (`break`s
after earlier targets are committed).

**F4 — cross-node wall-clock decides staleness.** `cli/palette.py:733`:

```python
if downstream_time is not None and upstream_time > downstream_time:
```

Both operands are `palette_run_completed_at_utc`, stamped by
`datetime.now(timezone.utc)` on whichever compute node ran that stage
(`shared/zarr_run_completion.py:44`, `:230`). This is the `palette plan`
staleness oracle. `:481` additionally picks the authoritative detect-quality
report by lexicographic sort of an ISO timestamp string. Same pattern in ~30
SQLite views via `ROW_NUMBER() OVER (... ORDER BY COALESCE(<utc>) DESC)`
(`registry/db.py:1611-1612`, `:1663-1664`;
`registry/migration_bodies.py:806, 878, 1684, 2051, 2956, 5812, 6032, 7193`).

**F5 — `max(..., key=st_mtime)` selects pipeline inputs across nodes.**
`cluster/keypoints/clipped_collection.py:149` picks which flat-ROI-cache manifest
backs each clip; the error text at `:154-156` advertises this as intended. Same
at `scripts/submit_subject_mask_batches_bsub.sh:499`,
`analysis/import_stimulus_to_zarr.py:288`, `analysis/calibration_manager.py:90`,
`training/export_detection.py:47-49`, `training/export_onnx.py:51-53`.

**F6 — "content-pinned" model verification degrades to (size, mtime).**
`cluster/clipped_inference._verify_binding` (`:204-214`) and
`"model_bindings_are_exact": True` (`:1115`) rest on
`verify_deployment_artifact_content`, which by its own docstring
(`registry/model_resolution.py:104-110`) skips re-hashing when a sidecar is
valid — and validity is exact `(size_bytes, mtime_ns)` equality
(`shared/artifact_fingerprint.py:34-43`, `:49-68`, `:133-152`). `shutil.copy2`
(used at `cluster/subject_masks/recording.py:62`) and `tar -p` preserve mtime.

**F7 — `expected_outputs` is existence-only and near-vacuous.**
`cluster/lsf/runtime.py:162-177` checks `expanded.exists()`; whole-recording
templates point at the run **group directory**
(`cluster/whole_recording_analysis.py:241-247`,
`cluster/keypoints/common.py:731-734`), which exists from `mark_run_started`
onward.

**F8 — wall-clock TTL drives irreversible `rmtree` of live caches.**
`utils/inspect_roi_cache.py:221` (`now()`) minus `:167` (NFS `st_mtime`) →
`:203` → `:229` `should_delete` → `rmtree` under `--apply`. `:227` additionally
marks anything without `cache_complete` for deletion — an in-flight cache being
written by another node has no completion marker yet.

**F9 — "run group" is four unrelated things.** `shared/stage_run_groups.py` is
47 lines of stage→zarr-parent name mapping; `LsfWorkflowFragment`
(`cluster/lsf/models.py:358-401`) is a planning-time grouping with no runtime
identity and no persisted group id; `LsfExecutionGroup` (`:178-216`) is an LSF
array; `docs/cluster_run_group_artifact_workflow.md` means a tarball. **It is a
label, not an atomic unit.**

**F10 — `bsub` over `ssh` has a silent orphan window.**
`cluster/clipped_inference.build_ssh_bsub_runner` (`:1221-1247`) has no retry and
no idempotence key. A connection drop after the remote `bsub` succeeds but
before stdout returns → `submission_failed` recorded
(`cluster/lsf/submission.py:149-172`) while the job runs, and `apply_plan`
refuses to retry (`:1209-1210`).

**F11 — `docs/workload_aware_analysis_scheduling.md` is entirely unimplemented.**
Every allocation is a hardcoded literal (`cluster/clipped_inference.py:567-578`,
`cluster/clipped_detection.py:152-165`,
`cluster/whole_recording_analysis.py:1068-1101`). Grep for
`allocation_tier|workload_aware|effective_stage_rows` → zero.

**F12 — advisory locks on a shared FS.**
`analysis_workflows/materializers/atomic_run_publisher.py:630-648` and
`analysis/bout_kinematics.py:788-791` use **blocking** `flock(LOCK_EX)` with no
timeout and no heartbeat. `tune/keypoint_review_web.py:142-162` puts a lock file
on shared storage but checks liveness with `os.kill(pid, 0)` on a **node-local**
PID (`:128-140`) and writes it non-`O_EXCL` (`:161`).

**Environment capture is a misnomer.** `shared/environment.py` is 33 lines of
path config with zero provenance. `include_system_context` defaults **False** in
both dominant call paths (`shared/run_provenance.py:279`, `:304`).
`run_provenance["command"]` is a hand-written module label, never `sys.argv`
(e.g. `tracking/crop.py:3907`). Two bugs: `tracking/crop.py:2759, 4354` read
`env_info['gpu']['driver_version']`, a key that lives on each *device* dict
(`shared/system_metadata.py:394`) — always `'unknown'`; and `detect_yolo.py`'s
env block is gated on `if created_new_root:` (`:2481`), so re-runs into an
existing zarr capture nothing. TensorRT version is captured build-time only
(`training/export_shared.py:172-186`) and `tensorrt` is absent from the
`get_relevant_packages` allowlist (`shared/system_metadata.py:613-638`). cuDNN
version is never captured.

### Lane D — the registry as source of truth

**[A] Authority per fact class:**

| Fact | Actual authority | Both writable? |
|---|---|---|
| "Stage X complete for Y" | **Split** — zarr gates, registry answers | **Yes, and they diverge silently** |
| "Which model produced run R" | Zarr attrs; registry is a lossy cache | Yes, no reconciliation |
| "Dataset exists / active" | Filesystem, sampled unreliably | Yes, flip-flops |
| Recording identity | **Directory basename** | Yes |
| Dataset identity | `sha256(resolved_path)` | Path *is* the key |

`emit_stage_completion` is fail-closed on the write side — it refuses to record
`ok` without a completion marker, selector eligibility, and provenance
validation (`registry/stage_complete.py:365-389`). The reverse is wide open:
`:441-448` swallows everything, and `:317-322` → `:360-361` returns `False` when
the registry path is unresolvable. Durable state: zarr says done, registry says
nothing, exit code 0.

**The stale-copy recurrence mechanism is not a wrong default.** 44 of 46
decision sites default to `/groups`. The mechanism is that
`RegistryPaths.from_env` (`registry/db.py:106-114`) resolves
`$PALETTE_REGISTRY_PATH` → `<default_root>/configs/fisheye/registry.yaml` →
`<default_root>/runs/registry/palette_registry.sqlite`, and **102 of 104 call
sites pass `Path.cwd()` as `default_root`.** From any cwd other than the repo
root with the env var unset, resolution lands on fallback #3 — and that path is
never checked for existence. `Registry.__init__` (`:1226-1237`) does
`_ensure_parent` (mkdir -p, `:166-167`) → `sqlite3.connect` (creates) →
`_init_schema()` (migrates to v62). **A wrong path yields a pristine,
fully-migrated, empty registry rather than an error.** `registry/query.py:410-411`
and `registry/status.py:52-53` do this too: read-only tools that silently create
a writable DB and report zero rows.

Seven sites make this a *write* path via `require_env_registry_exists=False`:
`inference/predict_detections.py:198`, `tracking/crop.py:1181`,
`tracking/arena_assignment.py:145,161`, `refinement/refine_detect.py:140`,
`refinement/detect_quality.py:96`.

Secondary vectors: `scripts/backfill_protocol_json.py:131-144` is the only
auto-selector that can choose `/nvme1`, and it ignores both override mechanisms;
`scripts/submit_citrus_session_import_bsub.sh:16` reads `PALETTE_REGISTRY`, not
`PALETTE_REGISTRY_PATH`; 191 doc lines across 64 files name the `/nvme1`
registry versus 89 for `/groups`, with `docs/recording_analysis_pipeline_contract.md`
self-contradicting (`/groups` at `:330`, `/nvme1` at `:324, :336, :340, :344`).

**Destructive reconciliation:**

- **`dedupe --apply` ignores its own conflict verdict.**
  `plan_registry_dataset_dedupe` computes `safe_to_auto_apply`
  (`registry/dedupe.py:563`) — a flag referenced **nowhere else in the repo**. The
  apply loop (`:615-656`) never inspects it and unconditionally calls
  `_delete_constraint_conflicts` (`:639` → `:336-378`). Resolution is "canonical
  wins" with no content or recency comparison.
  `tests/unit/fisheye/test_registry_dedupe.py:274-280` asserts this as correct: a
  `detect='ok'` row is deleted and `detect='missing'` survives. No `--dry-run`
  gate, no enforced backup.
- **Relocation → duplicate → dedupe is a closed data-loss loop, and the runbook
  prescribes it.** `dataset_id` is `f"{session_uuid}:z{path_hash[:12]}"`
  (`registry/db.py:7029`).
  `docs/recording_store_relocation_runbook.md:145-155` tells the operator to
  hand-`UPDATE` `zarr_path` and `path_hash` but says nothing about `dataset_id`,
  which still embeds the old hash. Next `register_from_root` mints a second row
  (`registry/db.py:7018`); step 7 (`:167-184`) then instructs `dedupe --apply`.
- **`dedupe` cannot see the duplicate class alias paths produce.**
  `_candidate_duplicate_groups` groups by `GROUP BY zarr_path, COALESCE(path_hash,'')`
  on **raw text** (`registry/dedupe.py:227`) while reconcile normalizes via
  `_normalize_fs_path`.
- **`prune_stale_datasets` follows `$TMPDIR`.** `temp_roots()`
  (`registry/prune_stale_datasets.py:136`) includes `tempfile.gettempdir()`, which
  honors `TMPDIR` — commonly redirected to job scratch on LSF. `path_exists` is
  recorded per candidate (`:206`) and **never consulted** in selection
  (`:337-340`). Mitigating: dry-run default, mandatory `--backup`, integrity + FK
  verification (`:514-533`). Best-guarded destructive path in the module.
- **`temp_store_guard` disables itself based on the registry's own location**
  (`registry/temp_store_guard.py:70-71`).

**Migrations** are forward-only, versioned, one-per-transaction with
`BEGIN IMMEDIATE`, mirrored into `PRAGMA user_version`
(`registry/db.py:1351-1383`). Two gaps: **no downgrade guard** (if
`current > latest`, the loop skips everything and old code operates on a newer
schema silently — grep for any "too new" check returns nothing), and
**`legacy_bootstrap` stamps latest without running anything** (`:1357-1368`).
The live DB was stamped `version=2, legacy_bootstrap` on 2026-02-09, so migration
001's DDL has never run against production shape — confirmed drift:
`training_runs.skeleton_id` has the declared FK at
`registry/migration_bodies.py:527` but `PRAGMA foreign_key_list` on the live DB
returns empty.

**Model selection never verifies content.** `registry/model_resolution.py:411-493`
and `:618` check only `Path(model_path).exists()`.
`registry/extractors/detect_performance.py:334-349` degrades `model_name` to
`Path(model_path).name` when the attr is missing.
`detect_performance.model_run_id` is populated for **60/247 rows (24%)**;
`keypoint_performance.model_run_id` **198/486 (41%)**.

### Lane E — human edits as provenance events

**[A] Verdict: real but partial, single-front-end, non-reconstructible.**

- **Three review front-ends, one audit log.** Only `labeling/web.py` writes audit
  events (`record_event`, `labeling/assignment_store.py:2457-2497`). The tune CLIs
  and standalone servers (`tune/keypoint_review_web.py`, `tune/detect_review_web.py`,
  `tune/keypoint_review.py`, `tune/keypoint_tuner.py`, `tune/detect_review.py`,
  `tune/refined_subject_mask_review.py`) write canonical zarr with **no event at
  all**. `tune/detect_review_web.py:214` takes `reviewer=str(body["reviewer"])`
  straight from the HTTP body.
- **Mask edits destroy prior pixels.** The apply path overwrites
  `runtime.refined.group["masks_roi"][roi_idx]`; the event's `before` is
  `{"canonical_area_px": int(canonical_mask.sum()), "edit_revision": N}`
  (`labeling/web.py:2434`). The raw model output survives in `subject_mask_runs`;
  **human revision N-1 does not.** Contrast keypoints, done correctly:
  `labeling/web.py:1735`, `:1751-1757` capture full pre-edit `points`, `reason`,
  `status`.
- **The checkpoint table is single-slot.** `labeling_session_checkpoints` has
  `UNIQUE(task_id, roi_idx, component_name)`
  (`labeling/assignment_store.py:350-374`); the writer (`:1877-1904`) does
  `ON CONFLICT ... DO UPDATE SET ... applied_at_utc = NULL, apply_id = NULL,
  edit_revision_before = NULL, edit_revision_after = NULL`. Re-editing an applied
  row **erases the record that the earlier apply happened.**
- **Identity is asserted, never authenticated.** `labeling/web_auth.py:25-37`
  resolves an HMAC-signed invite token from a query param or cookie (`:96-111`) —
  a bearer link — or `config.fixed_user`, or a proxy header.
  `tune/keypoint_failure_review.py:1634` falls back to `os.environ.get("USER")`.
  `reviewer` is **optional** in the canonical payload
  (`docs/review_status_schema_unification_contract.md:51-53`);
  `tune/detect_review.py:1513` and `detect_training_promotion_backend.py:710` both
  do `approved_by=str(approval.get("approved_by") or "unknown")`.
- **Zero tool/code version on any human edit.** Grep for
  `app_version|tool_version|code_version|git_commit|__version__` across
  `src/fisheye/labeling/` and `src/fisheye/tune/` → no hits.
  `shared/system_metadata.py:703-747` `build_invocation_record` captures exactly
  this and is used by analysis runs (`analysis/chaser_distance_runs.py:1636`,
  `analysis/tail_kinematics_runs.py:1565`, `tracking/arena_assignment.py:946`) —
  never by a review writer. Not even contemplated in
  `labeling/web_policy.py:102-118` `required_event_fields`. Listed under
  "Recommended future fields" at `docs/mutable_review_runs_contract.md:231`.
- **Approval has no history.** Stored as one overwritten attr dict
  (`utils/set_keypoint_review_status.py:143`). A run that was rejected, re-edited,
  then approved is indistinguishable from one approved first time. CLI approvals
  leave no event.
- **The audit log is cascade-deleted with its task.**
  `labeling/assignment_store.py:304` — `FOREIGN KEY(task_id) REFERENCES
  labeling_tasks(task_id) ON DELETE CASCADE`.

What is right: raw model output runs stay separate from human-edited refined
runs, so the original automatic prediction is always recoverable; detect keeps
manual corrections in a separate subgroup with row-level `manual_edit_flags`,
`source_kind_codes`, `source_detect_row_index`
(`shared/refined_detect_curation.py:89-104`);
`shared/refined_subject_mask_mutation.py:26-48` fails closed on mutating a sealed
canonical publication.

### Lane F — model and training-data provenance

**[A]**

- **Every GPU detection run is silently FP16 and says nothing.**
  `detection/detect_yolo.py:2249-2274`: if CUDA is available,
  `model.half(); model_fp16 = True`, unconditionally. `model_fp16` feeds
  `predict_kwargs["half"]` and dtype selection (`:2885`, `:2998`) and is **never
  written to any attr or provenance payload**. Grep for `precision` returns zero
  hits in `detect_yolo.py`, `detect_keypoints_yolo.py`,
  `infer_unet_subject_masks.py`, `run_detections_batch.py`,
  `run_detection_local_publish.py`. Same video, same weights, GPU vs CPU node →
  numerically different boxes, byte-identical provenance.
- **TRT provenance is excellent in the registry and absent from every output.**
  `registry/migration_bodies.py:639-675` (`tensorrt_models`) and `:7253-7288`
  (`model_deployment_artifacts`) record engine_sha256, source_onnx_sha256,
  trt_version, cuda_version, compute_capability, gpu_name, gpu_uuid, hostname,
  precision, builder settings. No inference code reads those tables —
  `query_model_deployment_artifacts` appears only at `registry/db.py:8623`
  (definition) and one test. No inference path dispatches on `.engine`.
- **The content hash is trust-on-first-use via a co-located writable sidecar.**
  `shared/artifact_fingerprint.py:150-156`: `_read_sidecar` looks for
  `<weights>.content_v1.json` **next to the weights** and returns the cached
  sha256 if `size_bytes` and `mtime_ns` match. That branch returns early, so the
  **registry cross-check is skipped entirely** (`mismatch` is only set in the
  `computed` branch, `:167-176`), and when a mismatch *is* detected it is a
  `warnings.warn` (`:172`), non-gating.
- **Model→training-set binding is a mutable ID list.**
  `registry/migration_bodies.py:516-526` — `training_sets` has
  `dataset_ids_json` and no dataset content fingerprint, no `set_version`, no
  `parent_set_id`. Those IDs point at mutable review surfaces that human edits
  change in place. Every relevant box in `docs/training_dataset_versioning_todo.md`
  is unchecked (`:13-16`, `:18-21`, `:47-50`, `:64`, `:69`). The only staleness
  signal is per-source `zarr_mtime_ns`
  (`utils/prepare_detect_training_from_registry.py:660-673`, checked `:1069-1073`).
- **Train/test leakage is provable for merged exports, unprovable otherwise.**
  Merged exports persist `splits/train_indices|val_indices|test_indices` with
  overlap validation (`training/export_detect_training_zarr.py:1816-1840`,
  `:731-739`; `export_keypoint_training_zarr.py:1435-1455`), honored by the loader
  (`training/zarr_yolo_dataset_loader.py:327-336`, `:1127-1132`). When training
  directly against per-recording zarrs, `_resolve_training_split_paths` returns
  `(None, None)` and the split is computed at runtime
  (`training/zarr_yolo_dataset_loader.py:1119-1161`) from a shared RNG consumed
  sequentially over per-dataset valid-row counts — and **never persisted**:
  `grep -n "split" src/fisheye/training/train_detection.py` → **zero hits**. Any
  subsequent review edit changes valid-row counts and permutes the entire split.
- **Row-level label origin is not in the training data.** Every checklist box in
  `docs/training_label_origin_phase1_audit.md` Detect (`:73-93`), Keypoints
  (`:131-154`), Eye Masks (`:193-208`) is unchecked. Keypoint merged export drops
  `reason` and `retune_id` (`:117-118`).
- **Quality gates default off.** `--require-review-state` and
  `--require-review-intended-use` default to `None`, and "If every gate parameter
  is `None`, the entire quality gate path is skipped"
  (`docs/training_quality_gate_contract.md:53-56`, `:78-83`).

**`shared/pose_model_schema_binding.py` is the exemplary surface** and the model
for the rest: binds keypoint semantics to `model_sha256` + `manifest_sha256` +
`registry_skeleton_spec_sha256`, canonicalizes to `canonical_json_sort_keys_v1`,
re-hashes the `.pt` from disk (`:464-468`) and the manifest (`:479-483`), fails
closed on any disagreement, emits a `binding_sha256` over the whole record
(`:383`).

---

## Claims this pipeline currently cannot defend

1. **"These detections came from model M."** Defensible only via
   `run_provenance.input_artifacts[].sha256` — which is absent store-wide (Root
   Cause 1) and, when present, may reflect a stale sidecar.
2. **"Model M was evaluated on frames it did not train on."** Defensible only for
   models trained from a merged export with persisted `splits/`. Otherwise the
   split existed only in RAM.
3. **"Model M was trained on dataset D, and D is what I'm showing you."** Not
   defensible — `training_sets` has no content fingerprint and no immutability
   enforcement.
4. **"These numbers are FP32 / match the realtime path."** Not defensible in
   either direction.
5. **"Annotator A produced this label on date D with tool version V."** Not
   defensible — bearer-link identity, no tool version anywhere, and two of three
   front-ends emit no event.
6. **"This label was approved by a qualified reviewer."** Weakly defensible —
   `reviewer` optional, `approved_by` falls back to `"unknown"`, no approval
   history.
7. **"Here is what this mask replaced."** Not defensible past one revision.
8. **"Nothing has changed since publication."** Not defensible — `immutable_snapshot`
   is an application contract, not a write lock
   (`docs/tabular_delta_compaction_contract.md:251-253`).
9. **"Show me the dataset as of date D."** Not defensible for any surface.
10. **"Only reviewed data entered training."** Not defensible — gates default off,
    and per-row label origin isn't exported for keypoints or masks.

**Scope note.** These are evidentiary failures, not correctness failures.
Nothing in this audit suggests a published number is wrong. The distinction
between *"my numbers are wrong"* and *"I cannot prove my numbers are right"*
matters, and this repository is firmly in the second category — which is the
category that bites at publication rather than during analysis.

---

## What is right (so it does not get broken)

- `shared/row_source_signature.py` — spec-digest-pinned, fail-closed
  (`:643-646`), explicit content-vs-revision basis. The only scheme here that
  could not be fooled.
- `shared/pose_model_schema_binding.py` — see Lane F.
- `analysis_workflows/materializers/atomic_run_publisher.py` — a genuine commit
  protocol. Pointed at the wrong stages, not wrongly built.
- `shared/rowset_fingerprint.py` + `assert_rowset_fingerprint_matches` — real
  content identity, actually enforced, across refined_detect → arena → tracking →
  crop.
- `shared/json_safety.py:101-127` — correct atomic publish for NFS.
- `cluster/lsf/runtime.py:73-115` — scratch cleanup that refuses any path not
  strictly under `/scratch/$USER/$LSB_JOBID`, and correctly handles
  `LSB_JOBINDEX=0`.
- `registry/repair_recording_identities.py` — mandatory backup, single FK-checked
  transaction, `integrity_check`, post-apply validation, filesystem rollback on
  failure (`:249-274`).
- `shared/run_lineage_fingerprint.py:51-67` — deliberately strips `created_at_utc`,
  `timestamp_utc`, `wall_time_s`, `latest` from the lineage hash. The clock
  exposure is concentrated in the CLI planner and the SQL `_latest` views, **not**
  in the fingerprint path.
- Task success is gated on exit code + output existence, never on "output newer
  than input." Right choice.
- `docs/lsf_submission_framework_design.md:288-297` documents the array-barrier
  hazard correctly, before the code hit it.

---

## Prescription

Not the sixty findings. In order:

1. **Decide the `consolidated_metadata` question repo-wide.** One decision.
   Re-arms two safety systems for free. Everything else is contingent on this.
2. **Re-run the attribute-based portions of this audit afterward** (see below).
3. **Install exactly one forcing function.** Make run resolution work *only*
   through the strict path, no legacy fallback. It will break loudly, everywhere,
   once. After that the gate stays honest permanently, because a regression
   surfaces in minutes rather than five weeks. This is the single change that
   converts provenance from a deliverable into a dependency.
4. **Make the registry distinguish absent from unreadable.** Distinguish `OSError`
   from absence in `_is_zarr_root`; abort `reconcile_missing_datasets` if more
   than a small fraction of scope would flip; skip the dataset on open failure in
   `maintenance.py:7900-7903` instead of writing error rows.
5. **Subtract.** Pick one fingerprint scheme, one review front-end, one registry
   resolver, one zarr read idiom. Delete or explicitly mark-experimental the rest.
   Deleting `shared/tabular_deltas.py` is a better outcome than leaving it, unless
   it gets wired.
6. **Status header on every contract doc** — `implemented` / `partial` /
   `specified-only`. Costs an afternoon; removes the false-assurance problem
   outright without wiring anything.
7. **Two one-liners with outsized payoff:** record inference precision next to
   `model_fp16` in `detect_yolo.py`; promote `model_sha256` to a flat attr
   alongside `model_path` in the three `write_*_model_resolution_provenance`
   functions.
8. **Persist the realized train/val split** from
   `zarr_yolo_dataset_loader.get_split_indices()` into the training run directory
   and registry. Closes claim 2.

Explicitly **do not** commission another audit wave. Sixty findings already
exceeds what can be acted on; more auditing produces more surplus, which is the
underlying problem rather than a remedy for it.

---

## Re-verification needed

Because of Root Cause 1, any `[A]` claim of the form *"attribute X is absent from
the store"* that was established by reading attrs through the zarr API is
suspect and should be re-checked with `use_consolidated=False`. Specifically:

- The `source_fingerprints: {}` observation (Lane B) — re-check on a directly
  opened group.
- `fingerprint_status="best_effort"` on all 17 lineage-bearing groups.
- The 20/20 `crop_runs` `is_dirty=None` observation (Lane A).
- Any "N of M groups have attribute X" count.

Claims established by **filesystem grep** over `zarr.json` are unaffected and
stand — notably the `run_provenance` 0/4052 census, which was
`grep -rl --include=zarr.json`.

---

## Method

Six read-only agents (Opus 5), one per axis: provenance data model; write path
and commit protocol; lineage graph and staleness; HPC execution provenance;
registry as source of truth; human-in-the-loop and model provenance. No agent
had write access; no repository or database state was modified. The
consolidated-metadata finding (Root Cause 1) came from attempting to verify an
agent's headline claim and finding it false in direction — the resolver was not
failing closed as reported, it was failing open, for a reason none of the agents
had looked for.
