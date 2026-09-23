# One recording identity for all ingested recordings

- **Status:** draft. Open questions remain (see [Open questions](#open-questions)).
- **Owner:** Jeremy Delahanty.
- **Last reviewed:** 2026-09-23.
- **Builds on:** [source-of-truth consolidation plan §4](../../diagnostics/source_of_truth_consolidation_plan_2026-08-25.md)
  (current-v2 recording identity), queue item RID-001, and INGEST-001 in the
  [authority consolidation queue](../../diagnostics/authority_consolidation_work_queue_2026-08-25.md).
- **Supersedes:** nothing. This does not replace the v2 design; it decides how
  v2 is adopted, what happens to legacy IDs, and names the entities v2 does not.

## Goal

Every recording Palette ingests, whether single-video or rolling-clip, and
whether its stimulus file is legacy or unified H5, gets its ID from the same
rule. Recordings from one acquisition, the camera recordings an arena
experiment uses, and the subject in each experiment are related by explicit
records rather than by parsing ID text. Existing recordings keep their IDs.

## Current state (measured 2026-09-23)

All measurements used the canonical registry opened read-only.

- **The v2 scheme exists but is unused.**
  `recording_id_from_session_camera` in
  `src/fisheye/shared/source_recording_identity.py` mints
  `source_recording_<sha256(session_uuid, camera_id)>`. It landed on main
  with migration 73 on 2026-08-25. **0 of 296** registry recordings use it, and
  the identity tables (`recording_identity_current` and related) are empty.
- **At least five legacy ID formats are in use:**

  | Format | Rows | Typical source |
  |---|---:|---|
  | `YYYY-MM-DDTHH-MM-SSZ_arena_N_<Protocol>` | 226 | H5 sessions (incl. 17 external-IPC single-clip) |
  | `YYYY-MM-DDTHH-MM-SSZ_arena_N` | 52 | `behavior_v1` |
  | `<prefix>_YYYY_MM_DD_HH_MM_SS_cam<serial>` | 8 | video-only (sleepyfish/sickyfish) |
  | `YYYY_MM_DD_HH_MM_SS_cam<serial>` | 4 | rolling-clip consolidator |
  | `YYYY_MM_DD_HH_MM_SS_Cam<serial>` and `…_recording_only` variants | 6 | one-offs |

- **`session_uuid` cannot group cameras.** 293 distinct values across 296
  recordings. H5 rows store a per-arena value, and 67 rows store the
  recording ID itself. The real acquisition grouping (85 acquisitions, 61 of
  them with four cameras) is recoverable only by parsing the timestamp prefix.
- **Stored identities already disagree in places.** Example: GoodCopBadCop
  arena_1's `recording_manifest.json` has `…_arena_1`, while its analysis Zarr
  root has `…_arena_1_GoodCopBadCop`.
- **Producers mint IDs independently.** Non-v2 minting paths remain in:
  - `utils/consolidate_external_ipc_rolling_recordings.py`:
    `{session}_cam{serial}`, and it also sets `session_uuid = recording_id`.
  - `utils/create_clipped_analysis_zarr.py`: falls back to the directory name.
  - `utils/draft_video_only_organizer_manifest.py`: `{session_uuid}_cam{camera_id}` template.

  The organizer and the transfer-v2 intake on PR 149 already mint v2 IDs.

## Decisions

1. **Legacy recordings are not renamed.** A recording ID is embedded in
   directory and file names, 27 registry tables, sealed receipts and digests,
   analytics exports, and training-set provenance. Renaming would invalidate
   the provenance chain. The existing 296 IDs are frozen.
2. **New recordings use the v2 ID**, one per camera stream per acquisition
   session: `source_recording_<sha256(session_uuid, camera_id)>`, with the
   mapping profile declared so readers can recompute and check it. The
   human-readable label lives in `recording_name`.
3. **Legacy recordings get an append-only crosswalk** from legacy
   `recording_id` to `(acquisition session, camera_id)`, plus the arena
   experiment and subject it belongs to where those are known. It is filled from the parsed timestamp prefix and reviewed once
   before it is trusted. It is read-only history: nothing new writes legacy
   IDs, and nothing resolves new recordings through it.
4. **Every producer uses the v2 minter.** The three non-v2 paths listed above
   are retired, not patched to produce a fourth format.

## Identity model

The recording ID identifies a **camera acquisition stream**. It does not
identify an arena, an experiment, a subject, a crop, a clip, or an H5 file.
Those are separate entities with their own identities, related to recordings
by explicit bindings.

| Entity | Identity | Meaning |
|---|---|---|
| Acquisition session | `session_uuid` (from Citrus; see Q1) | Everything started together |
| Camera recording | v2 `recording_id` = `hash(session_uuid, camera_id)` | One camera's stream within that session |
| Arena experiment | to be defined (see Q2) | One regional protocol execution, referencing one or more camera recordings |
| Subject | registered subject ID | The biological individual assigned to an arena experiment |
| Media products | their own product identities | Full-frame video, crops and clips belonging to one camera recording |

- **Camera recording ↔ arena experiment is many-to-many.** One camera can
  observe several arenas (compartmented dishes: each arena experiment
  references the same recording). One arena can be observed by several
  cameras (future 3D: the experiment references several recordings). Neither
  case changes the recording ID.
- **Subject is separate from arena.** The same fish can take part in
  different arenas or sessions, so subject assignment is its own binding from
  arena experiment to subject, not a property of the arena.
- **`arena_id` is not folded into the recording ID.** Which arenas a camera
  sees is recorded in the arena-experiment binding.
- **`(session_uuid, camera_id)` is declared unique.** This closes package 2 of
  §4.7 in the consolidation plan, and it holds in both cases above.
- **Multi-camera/3D reconstruction remains deferred.** Citrus records the
  associated camera IDs for a session, but that list is not a validated
  multi-camera contract. A 3D experiment will additionally need immutable
  calibration (intrinsics and extrinsics) references on its binding. The
  2026-09-23 review cites the Citrus camera metadata writer
  (`src/logging/session_metadata_writer.cpp:228`) and
  `docs/arena_group_detection_assignment_design_2026-08-30.md:656` (deferred
  work); neither path was found in the local checkouts, so the Citrus agent
  should confirm them.
- Today one camera observes each arena, so every binding has a single member.
  The model covers both cases without a new ID scheme.

## Consequences for consumers

- **Training leakage groups must key on subject.** Grouping by session or by
  arena experiment would still let one fish cross splits when it appears in
  several arenas or sessions. `resolve_training_leakage_group` already prefers
  registered subjects, falling back to acquisition start time and then
  `recording_id`. Subject registration is what keeps the split honest; the
  fallbacks are only as good as the missing subject data allows.
- **Per-subject analyses and exports** that treat one recording as one fish
  must resolve through the arena-experiment and subject bindings. A
  compartmented dish (one recording, several fish) or a 3D arena (several
  recordings, one fish) breaks that assumption, so it has to be settled before
  the first such recording is analysed.

## Open questions

1. **Session field (Citrus).** Which unified H5 field is the canonical
   acquisition-session ID: the session-scoped `recording_id` in
   `/correspondence/acquisition/binding_json`, `citrus_session_uuid`, or a
   planned `session_id`? It must be globally unique, immutable, and present in
   every unified H5 and transfer-v2 manifest. Asked of the Citrus agent.
2. **Recording and experiment bindings (Citrus).** Confirm that the hashed
   recording ID denotes a camera acquisition stream, independent of arena,
   subject, crop and clip identities. Define an explicit many-to-many
   recording-to-arena-experiment binding, with separate subject assignments and
   immutable calibration references. Distinguish fields implemented today
   from planned additions. Asked of the Citrus agent.
3. **Opaque IDs.** Accepted in principle: v2 IDs are hashes, with readable
   text in `recording_name`. This holds only once the entity meanings above
   and a globally unique parent session identity are pinned. Hashing cannot
   resolve ambiguous identity inputs.
4. **v2 IDs for legacy recordings.** Proposed: no. The crosswalk provides
   session grouping; a second ID per legacy recording adds a
   parallel key without new capability.

## Next steps

1. Resolve Q1, Q2 and Q4 and record the answers in the decision log.
2. Declare `(session_uuid, camera_id)` unique (RID-001, plan §4.7 package 2).
3. Retire the three non-v2 minting paths.
4. Design and populate the legacy crosswalk, including handling for
   manifest/Zarr disagreements like the GoodCopBadCop example.
5. Land PR 149 (parent intake, which contains PR 155) so clipped recordings
   enter through the same importer and minter as single-video recordings.

## Decision log

- 2026-09-23: Opened. Decisions 1 and 3 (no rename, legacy crosswalk) agreed; 2 and 4 adopt the existing v2 design. Identity model proposed.
- 2026-09-23: Revised after review. The recording ID denotes a camera stream;
  camera recording ↔ arena experiment is many-to-many; subject is separate
  from arena; media products carry their own identities; training leakage keys
  on subject. Opaque IDs accepted conditionally (Q3). Q1, Q2 and Q4 open.
