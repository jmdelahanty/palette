# Intake: one source of truth and one writer per fact

- **Status:** draft for review. No code changes yet.
- **Owner:** Jeremy Delahanty.
- **Last reviewed:** 2026-10-07.
- **Census basis:** three read-only census passes over `origin/main` @ `98f458e9` (2026-10-07). They covered recording identity and context; subject, setup and stimulus; and video, crops, completion and the registry. File and line references below are to that commit and are relative to `src/fisheye/`.
- **Related:** [canonical subject fields](../2026-10-06-canonical-subject-fields/README.md), workflow runner (PR #288, §4 and §6), [recording identity](../2026-09-23-recording-identity/README.md).

## Goal

Every new recording enters Palette through one path: Orange/Citrus transfer-v2, then organize, then import, then register. Each fact intake records has:

1. **one owner:** a single code path, i.e. one function, that may write it;
2. **one authoritative location**, where readers read it;
3. **derived copies only where something needs them.** Copies are written by the owner in the same step, checked against the authority, and never written by anyone else.

Manual corrections remain possible. They go through the owner's publish function and record why they were made. They are never a second implementation that writes its own shape.

Historical recordings keep being readable. This design is about what new recordings get and who may write it.

## What the census found

Intake is mostly fail-closed: identity, context and the import receipt are cross-checked and refuse on disagreement. The problems are elsewhere:

- many redundant copies;
- writers that bypass the owners;
- a few facts that never reach the place readers look.

### Bugs or likely bugs on the new-recording path

These should be fixed in small PRs regardless of the rest of this design.

| # | Problem | Evidence |
|---|---|---|
| B1 | **Orange can override an H5 subject record.** It publishes whenever the existing record has no `dish_uuid` or `dish_id`. That case covers a legacy or pre-contract H5 that carries `subject_count` and ids but no dish. The result is either that H5 ids are dropped, or that a new subject run is left dangling while the experiment setup still points at the old one, so `resolve_experiment_setup` raises "authorities disagree". | `utils/import_recording_analysis.py:684-704`, `shared/experiment_setup.py:478-483` |
| B2 | **A retried unified intake skips subject projection.** A sealed native run makes `stimulus_runs_present` true. The "stimulus skipped" branch never calls `project_unified_subject_metadata`. | `import_recording_analysis.py:293-304, 1239-1246, 1274-1299` |
| B3 | **The experiment setup counts subject ids with its own rule.** That rule ignores a singular `subject_id`, so a Citrus v3 record with 1 id gets setup `assigned_subject_count=None` and `count_only`. | `shared/experiment_setup.py:154-163` vs `shared/subject_metadata._explicit_subject_ids` |
| B4 | **Profile readers never read the subject record.** The detection, keypoint and subject-mask profiles read root `genotype`/`dpf` and `analysis_metadata`. These don't exist for unified or Orange recordings, so the profile rows are null while the registry has the values. | `utils/detection_profile.py:329-380`, `utils/keypoint_profile.py:266-306`, `shared/subject_mask_profile.py:161-187` |
| B5 | **The registry has no fps, codec, size or color range for transfer-v2 parents.** `provenance` reads `raw_video` attributes, which the clipped path doesn't write. The rolling stream contract omits them, and color range isn't observed at intake at all. | `registry/db.py:833-880`, `shared/acquisition_video_streams.py:~330-370`, `shared/clipped_video_collection.py:485` |
| B6 | **Manifest mutators have no seal guard.** Any of them can break receipt re-verification and organizer resume. They are `refresh_recording_manifest_metadata`, `backfill_hevc_keyframe_flags`, `backfill_video_only_sidecars`, `recording_manifest_import_status`, `refresh_recording_preflight`, `set_recording_subject_metadata` and `intake_video_only_recording --overwrite-manifest`. | census F10 |
| B7 | **Writers outside the identity authority write the identity rows.** These are `register_from_root`/`upsert_recording` on derived/training zarrs that share a `recording_id`, `emit_stage_completion` → `upsert_dataset`, `backfill_clipped_analysis_metadata` (raw SQL) and `repair_recording_identities`. Their writes can contradict authority-bound rows and mostly bypass the single-writer gateway. | `registry/db.py:2457, 2543, 6671`, `registry/stage_complete.py:443`, census F11 |
| B8 | **Job-mode registration can't recover from a registry failure.** After a successful import, a sealed archive left without a registry binding is re-planned as `skipped` and synced with `receipt=None`, which requires an existing binding. This is likely but not reproduced. It only matters while job-mode registration exists. | `import_organized_recordings_analysis.py:513`, `registry/recording_identity_authority.py:1364-1366` |
| B9 | **Some unified H5 biology attributes are never checked against the sealed snapshot.** Genotype, species and sex aren't compared, because `PAIRED_FIELDS` covers ids, dish, statuses and mzb only. | `shared/unified_h5/citrus_subject_snapshot.py:52-67` |
| B10 | **Two keys carry different meanings in different places.**<br>• `source_layout` is `rolling_clips` in the manifest and Zarr root, but `single_video` in the frame-index files for the same single-video parent.<br>• `artifact_schema_id` is `orange_transfer_parent_v1` in the manifest, but `recording_analysis_v1` in the Zarr root and the registry. | `organize_transfer_recordings.py:897`, `build_transfer_parent_frame_index.py:245`, `source_recording_identity.py:32` |

### Redundancy

| Fact | Copies today | Writers today |
|---|---|---|
| Recording identity (`recording_id`, `session_uuid`, `camera_id`) | about 8: plan, state, status JSON, manifest, frame-index files, Zarr root, receipt, registry | 4 implementations, cross-checked. Legacy `recording_id = session_uuid` fallbacks remain in `db.py:601, 2475`, `maintenance.py:729` and `consolidate_external_ipc_rolling_recordings.py` |
| Recording context (type, subtype, mode, intent, origin, source, version) | 6 or more | The Zarr root uses `setdefault` for 4 of its 7 fields, so a stale value survives. The registry never writes NULL. |
| Subject identity and biology | record v1/v2, run attributes, root `genotype`/`dpf`, `analysis_metadata`, manifest (legacy H5), registry | 5 intake call sites plus 3 manual tools. Legacy H5 is published twice (`:640` and `import_stimulus_to_zarr.py:3109`). |
| Experiment setup and subject count | setup run, root `experiment_setup` projection, root `subject_count`, `experiment_setup_status`, stimulus-run `source_metadata` | The canonical publisher, plus 5 writers of the root projection with no run: `setup_experiment_metadata`, `intake_video_only_recording`, the clipped copies and `backfill_subject_context` |
| Video metadata | single-video: 4 copies. Clipped: root `source_video_metadata` plus the frame record | Intake, plus 4 or more unused backfill tools |
| Frame index | files, absolute root attributes, a relative `raw_video` attribute and locator, registry `datasets` | Builder plus a legacy CLI with `--overwrite` and a backfill |
| "Imported" | about 7 surfaces in 3 vocabularies: receipt, binding, `palette_run_completion_*`, ledger `publication_status`, status JSON, JSONL logs, poller markers, plus flags that are stale by design (frame-index manifest `import_complete:false`) | Only the receipt and its binding are authoritative |
| Transfer provenance (`snapshot_id`, `attempt_id`, plan sha, producer, sealer) | plan, state, manifest, frame-index files | Never reaches the Zarr, the receipt or the registry |
| Lookup statuses | root `orange_*` attributes; the record's `subject_lookup`/`fish_reference` | No reader of the root attributes in `src/` |

## Ownership

One row per fact. "Authority" is where readers must read the fact. "Derived" copies are written by the owner, in the same step, and are verified.

| Fact | Owner (only writer) | Authority | Derived copies kept | Removed for new recordings |
|---|---|---|---|---|
| Recording identity | `shared/source_recording_identity` (mint) via the organizer plan | Zarr root identity attributes plus the receipt `identity_claim`. The registry gets them only through `recording_identity_authority`. | manifest, frame-index binding (checked) | legacy `recording_id = session_uuid` fallbacks for current-profile roots; `repair_recording_identities` (unsupported, delete); `consolidate_external_ipc_rolling_recordings` for new recordings |
| Acquisition session | snapshot → identity `session_uuid` | identity `session_uuid` | — | manifest `orange_session_id` duplicate (reads stay for legacy) |
| Recording context | Orange snapshot → `producer_manifest_context` | manifest plus Zarr root, required **equal on all 7 fields** (not `setdefault`) | registry `recordings` as an exact mirror, NULL included, for current-source rows | `validate_recording_manifest --fix` defaults for producer manifests (already skipped) |
| Recording layout | snapshot `recording_layout` | one key, `recording_layout` (`single_video`/`rolling_clips`). The storage form gets its own key, `storage_layout=clip_collection`. | — | the overloaded `source_layout` meaning (renamed; reads of old archives keep working) |
| Transfer provenance | organizer plan | **import receipt** (add `snapshot_id`, `attempt_id`, `plan_sha256`, `sealer`, `orange_producer`); projected to the registry | manifest `source_transfer` (kept, checked) | — |
| Subject identity and biology | `subject_metadata.publish_subject_metadata` with a named translator, **called only from the intake module** for new recordings | the subject record (v2 canonical `subject`) | registry projection | root `genotype`/`dpf`/`species`, `analysis_metadata.subject_metadata`/`session_context`, manifest biology (after B4 moves the readers); the duplicate legacy-H5 publish in `import_stimulus_to_zarr` |
| Subject source precedence | the intake module | **the H5 subject is authoritative whenever the H5 declares any subject record. Orange fills in only when the H5 declares none or a declared absence.** (fixes B1) | Orange's declaration is kept as provenance inside the record's source (not root attributes) | root `orange_subject_reference_*` attributes |
| Experiment setup and subject count | `experiment_setup.publish_experiment_setup`, called only from the intake module; ids come from the subject record (fixes B3) | the setup run | root `experiment_setup` projection: written **only** by the setup publisher | `setup_experiment_metadata`, `intake_video_only_recording`, `backfill_subject_context` writes of root setup; root `experiment_setup_status` |
| Stimulus (new recordings) | unified-H5 sealed reference (`unified_stimulus_import`) | sealed native run | — | the legacy H5 stimulus path for **new** transfer-v2 deliveries (see open question 1) |
| Video metadata | clipped builder plus `_publish_external_acquisition_authority` | the frame record (receipt-bound) plus root `source_video_metadata` | registry `provenance`/`acquisition_video_streams` **read from `source_video_metadata`** (fixes B5); color range observed at intake | 4 or more unused video-metadata backfill tools (`import_video_metadata*`, `apply_source_video_metadata_backfill`, `backfill_import_profile_metadata`) |
| Acquisition streams and crop ledger | `write_acquisition_video_stream_inventory` plus the ledger publishers | stream groups plus the ledger runs | inventory summary on the parent group | the separate rewrite in `backfill_acquisition_video_stream_inventory` when it runs on current-source zarrs (its `clipped_inference` stage calls the owner instead) |
| Frame index | `build_transfer_parent_frame_index` | index files; **one relative locator** in `source_video_metadata` | absolute root attributes derived from the locator; `datasets.source_*` (intake also writes `source_frame_index_schema`) | `build_recording_frame_index --overwrite`, `backfill_clipped_analysis_metadata`; the stale flags `import_complete:false` and `registry_admitted:false` |
| "Imported" | `publish_recording_import_receipt` (the only minter) | receipt plus registry binding | status JSON, logs and runner sentinels as caches only | stale-by-design flags; manifest `import_status` (legacy) |
| `recording_manifest.json` | organizer (`_parent_manifest`, write-once) | **immutable after organize** | — | every in-place mutator either refuses a manifest with `source_transfer`, or is deleted if unused (B6) |
| Registry identity rows | `recording_identity_authority`, via the shadow gateway only | `recordings`, `datasets`, identity tables, bindings | — | non-authority writes to rows owned by a verified source (B7): `register_from_root` refuses a `recording_id` bound to a verified source; `emit_stage_completion` stops upserting `datasets` for current-source rows |
| Registry projection rows | `_project_nonidentity_from_root` | registry tables | — | direct-canonical writers (`fisheye.registry.scan` CLI, `intake_video_only_recording --register`, `import_recordings_training`) route through the gateway |
| Step status `raw` | intake, from the receipt | `recording_step_status` | — | `raw = "na"` for receipt-bound transfer-v2 parents |

## Intake module and the runner contract

All intake facts are written through one module, proposed as `fisheye.intake`. Its public entry points are what the workflow runner calls (PR #288 §4, §5.3 and §6). Nothing else in `src/` may call the owner functions for new recordings. This section was reconciled with cluster-runner (#288 commit `a39c04ac`).

1. **`discover(staging_dir, destination_root)`**
   - Lists targets from both live sealed markers and durable `.transfer_intake/<sha>/organization_state.json` states: `reserved`, `materialized`, `retiring`, and `complete` **where `probe_register(sha)` is false**.
   - Runner sentinels are caches and are never read for discovery.
   - Each target carries its state and its `admission_contract` mode. The mode is kept even after job-mode registration is retired, so an in-flight delivery that started under the old mode is reported rather than resumed in the wrong mode.
2. **`import_delivery(snapshot_sha, run_dir, resume_plan=None)`**
   - Runs in an LSF job and writes nothing to the registry.
   - `run_dir` must not already exist. The runner passes a fresh `<flow_root>/intake/<sha>/attempt-<n>/` for each attempt.
   - When durable state exists, the module loads `state["plan"]` itself. The explicit `resume_plan` is kept only as an operator override.
   - It is idempotent and resumable from any durable state, including `retiring` with the marker already gone.
3. **`register_delivery(snapshot_sha)`**
   - Runs on the writer host only. Host identity (short name vs FQDN) is normalized in one place.
   - Writes **all** of the delivery's zarrs in **one** `publish_registry_shadow` mutation (new batch gateway function), so it is atomic per delivery.
   - Is idempotent.
   - Refuses a synthetic `data_origin`, read from the stored plan.
4. **`probe_import(snapshot_sha)` and `probe_register(snapshot_sha)`**
   - Read durable NFS evidence only: receipts, bindings and intake state.
   - Registry reads are read-only (`mode=ro`).
   - They make no LSF calls.
   - They return a verdict plus an evidence digest, and are the only definition of "done". Each digest is stable across replays:
     - import: sha256 over the sorted (zarr path, receipt sha256) pairs;
     - register: the same pairs plus the registry binding ids.
5. **One claim mechanism.** Intake's own locks are the only claims; the runner takes none.
   - **Import:** the existing transfer workflow lock, `fcntl.flock(LOCK_EX | LOCK_NB)` on `<destination_root>/.transfer_intake/<sha>` plus a lock suffix (`organize_transfer_recordings._coordinator_lock`).
   - **Register:** a new per-delivery lock of the same kind.
   - **How the locks behave across hosts:** on the NFSv4 store this is a server-side byte-range lock. It is released when the holding process exits, or after the NFS lease (about 90 s) if a client host dies. A file left behind does not hold the lock.
   - **Ordering change:** the lock must be taken **before any side effect**. Today the workflow creates its run dir and status file before taking the lock (`citrus_transfer_parent_workflow.py:188-202`), so that order has to change.
   - When the lock is held, the step exits 75 and the runner records "attached".
6. **Exit codes:**

   | Code | Meaning | Runner action |
   |---|---|---|
   | 0 | published, and the matching probe is true | done |
   | 65 | refused: the input is invalid (synthetic origin, mode conflict, contract violation) | not retried; it is an operator incident |
   | 75 | held by another live job | attached, no retry |
   | 1 | any other failure | retried |

Job-mode registration (the LSF job writing the registry) is retired, and the writer host is ws1 (open question 2).

## Enforcement

A document alone does not prevent a second writer. Two kinds of check do, extending existing ratchets rather than adding new ones.

**1. Scoped import and AST checks.** These go into the import-linter contracts and the existing ratchet scripts:
- Only `fisheye.intake` and the owner modules themselves may import `publish_subject_metadata`, `publish_experiment_setup`, `publish_recording_import_receipt`, `project_regular_source_recording_identity` and `write_acquisition_video_stream_inventory`.
- Manual-correction tools are allowed only through a named, allow-listed entry that records a reason.
- A ratchet forbids writes to `recording_manifest.json` outside the organizer.
- A ratchet forbids writable connections to the canonical registry path outside `shadow_publish`.

**2. One end-to-end test as the acceptance gate.** It runs:
1. a sealed transfer fixture through the generated job script (`import_delivery`);
2. `register_delivery` on a temporary registry;
3. both probes, which must be true.

It then replays every step and requires no change and no duplicate. It also covers:
- a resume from `retiring` with the marker gone;
- an H5-plus-Orange subject precedence case (B1);
- a retried unified intake (B2).
- a **second concurrent attempt** on the same delivery, which must exit 75 and leave no new files (the lock is taken before any side effect).

## Rollout

Each step is a small PR. Bugs ship first because they affect new recordings today.

1. **Bug fixes:** B1, B2, B3, B6 (guards), B9.
2. **Readers onto the authorities:** B4 profiles move to `resolve_subject_metadata`; B5 projects video facts from `source_video_metadata` and observes color range.
3. **The `fisheye.intake` module** with the five entry points and the batch registration gateway. The runner's slice 1 builds on it.
4. **Enforcement checks and the end-to-end gate.**
5. **Subtraction:** delete the writers marked "removed" above. Keep the read paths for historical archives.
6. **Key cleanup:** B10, the `recording_layout`/`storage_layout` and `artifact_schema_id` meanings. Each change is versioned; existing archives are not rewritten.

## Open questions for Jeremy

1. **Legacy (non-unified) H5 in new transfer-v2 deliveries:** refuse them, so new recordings always have a sealed unified H5 or none? Recommended: yes. All current Citrus deliveries are unified, and this removes the second stimulus and subject path.
2. **Retire job-mode registration** (registration inside the LSF job), keeping workstation registration only? Recommended: yes. It removes B8 and the `admission_contract` mode split.
3. **Subject precedence:** the H5 is authoritative whenever it declares a subject record; Orange fills in only when the H5 declares none or a declared absence. Agree?
4. **Exact registry mirror:** current-source registry rows mirror the authority exactly, NULL included, instead of COALESCE. Agree?
5. **Deletion list:** approve deleting the unused legacy writers listed under "Removed", after their callers are checked again at the time of each PR.
