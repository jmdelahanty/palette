# Intake: hash each file once per job, scoped parent checks, parallel parents

- **Status:** implemented on branch `agent/palette/intake-verify-once-memo-20261105`; not merged, not deployed. Required CI not yet run on the final commit.
- **Owner:** intake-duration-investigation session (for the user); written 2026-11-05
- **Requested by:** the user, relayed through the data-export session, after the timing report below. Options 1, 2 and 4 of that report were approved; option 3 (trusting the sealer's digest for every large file) was deliberately left out, as the deferred "option B" of 2026-10-07.
- **Builds on:** the 2026-10-07 "verify-once, option A" decision: one independent Palette hash of every delivered byte at arrival, stat identity afterwards.

## What was measured

Live delivery `2026_11_05_17_00_17` (4 arenas, solid black 30 min, LSF 155225503, intake pinned 2931ddc8). Times reconstructed from file mtimes in the job directory and under `/groups/.../recordings/2026_11_05_17_00_17_Cam20100{93..96}`.

| Phase (UTC 2026-11-05) | Duration |
|---|---|
| Organization plan (one whole-delivery verification) | 35 s |
| Hard links and parent directories | 25 s |
| Frame-index build, 4 parents one after another | 7 min 27 s (about 1 min 45 s each) |
| Final materialized-file check and Zarr creation | 17 s |
| 4 stimulus imports, one after another | about 2 min 20 s |
| Finalize and retirement | about 1 min |
| Two 5-minute cron ticks (poller, registrar) | about 3.4 min of waiting |

Inside each parent's 1 min 45 s: about 57 s before the first projection file, 3 s for the parquet, about 44 s before the clip index, 1 s for the sync-sample check and manifest.

The cause is the verification call graph, not the hashing policy. `build_transfer_parent_frame_index` verifies the whole delivery snapshot at its start and again after the parquet is written, and resolves the materialized sources both times, which re-verified every parent's copies. Each snapshot verification walks the inventory twice (build, then close the observation window). With plan, materialize, prepare and finalize passes the job ran about 30 full passes.

What a pass hashes: everything the Citrus sealer does not attest. The sealer (citrus-recording-transfer 3.3.0, marker v3) attests files of 16 MiB and more with video or H5 suffixes, so the 115.7 GB of mp4 and H5 was never rehashed. The 78 CSV and 11 JSONL files (3.3 GB, mostly `*_pose_events.jsonl`, `*_yolo_events.jsonl`, `*_yolo_perf.csv`) were hashed on every pass, 22 to 30 s each. About 70 GB of hashing to verify 3.3 GB once.

Correction to the first report: raw is not 1.7 GB per arena. Each arena carries a 17 GB crop mp4 and an 11.5 GB full mp4; the delivery is 119 GB. Intake time scales with realtime-product size, not video size.

## Decisions

| Question | Decision |
|---|---|
| Where the once-per-job rule lives | In the one hashing primitive, `file_ref` in `fisheye.shared.recording_transfer_snapshot`, as an opt-in `digest_memo()` block. Outside a block every call reads the bytes, as before. |
| What a memo entry is bound to | The file's device, inode, size, mtime and ctime at the moment its bytes were read and verified unchanged. A hard-linked copy shares the staged file's entry. |
| When an entry is reused | Only while all five values match. A rewrite, truncation, replacement, permission change or new hard link changes the ctime, which cannot be set from user space, so the file is read again in full. |
| Files written moments ago | Not memoized. Filesystem timestamps tick coarsely, so a file whose ctime is within 5 s of now could be rewritten in the same tick with an unchanged signature. The next pass hashes it again and records it once settled. The window also covers NFS server/client clock offsets. |
| Lifetime | One process, one `import_delivery` call. Nothing is persisted; a retry or a resumed job starts empty. The registrar and the probes are unchanged. |
| Threads | The memo is thread-local; worker threads join the job's memo explicitly. |
| Scope of a builder's own check | `resolve_materialized_parent_sources` verifies only the selected parent's copies. The whole delivery is still verified when materialized, after every builder (end of preparation), and three times during retirement. |
| Parallel parents | `parent_workers` (default 4, an execution resource) runs the per-parent index build, index verification and sync-sample assessment, and the per-parent imports, in a thread pool. Journal writes and parent manifests stay on the caller's thread in plan order. Results stay in plan order. The first failure propagates only after every started worker finished. |
| Where it is recorded | `execution: {parent_workers, digest_memo: {hits, misses, unsettled}}` in the job status JSON. Additive key; schema id unchanged. `--parent-workers` on `run_citrus_session_import` and `fisheye.intake import-delivery`. |
| Out of scope | Trusting the sealer's digest for non-video large files (option B). Making the registrar or the probes use the memo. Changing the cron tick. |

## What must remain identical, and does

- Every byte written by intake: parquet, clip index, projections, recording manifests, sync assessments, Zarr archives, receipts. The memo returns the digest of the same bytes; the parallel path writes the same files in the same places.
- Every refusal of `file_ref`, `_matches_file`, `verify_transfer_snapshot`, `_verify_materialized_files` and the builder. Tests show each one still fires with a warm memo.
- Sealer attestation rules and the stat-identity rules of `_SealedSourceIdentity`.
- `import_parents` semantics: every planned parent is attempted, the per-parent outcome vocabulary, the workflow lock lent to every stimulus child.

## Accepted limit

A rewrite that keeps the inode, size, mtime and ctime unchanged is not detected within one job. That needs a same-tick rewrite of a settled file (prevented by the settle window) or kernel-level timestamp forgery, which the cross-job verify-once rule already accepts as out of scope.

## Tests

- `tests/unit/fisheye/test_transfer_digest_memo.py` (19): the primitive outside and inside a memo; hard-linked copy; rewrite of the same size, truncation, append, inode replacement, rewrite with a forged mtime; permission change; symlink, directory and missing file; replaced during open; per-thread and shared memos; nested memos; the settle window; whole-delivery snapshot verification equal to the fresh one; a source change between verifications refused; the organizer run reading each stat signature once and reporting the fresh digests; the organizer refusing a rewritten copy under a warm memo.
- `tests/unit/fisheye/test_intake_parent_parallelism.py` (11): a tampered copy of another parent passes the builder's own check and is refused by preparation and finalization; a shared file's tampered copy is pinned to its parent; parallel preparation writes the sequential content with all builders in flight at once; one worker runs inline; a failing parent refuses after its peers finished and publishes no manifest; parents import concurrently in plan order with the lock lent to all; the default import path is sequential; the status records the execution resources; fewer than one worker is refused before any write.
- Mutation check: with `_stat_signature` reduced to (device, inode), 9 of the 19 memo tests fail, including the organizer refusal.
- Regression: 433 tests across the intake, transfer, organizer, snapshot, poller and canary files pass at the working tree.

## Expected effect

Hashing drops from about 30 passes to about 3 (plan, after the hard links moved the ctime, first pass after the 5 s settle). Per-parent work overlaps. Estimated job time for a four-arena delivery falls from about 13.5 min of work to roughly 3 to 5 min; the two 5-minute cron ticks then dominate. To be measured on the next live delivery after a re-pin, through `execution.digest_memo` and per-parent mtimes.

## Handoff

- Worktree `/home/delahantyj@hhmi.org/gitrepos/palette-worktrees/intake-verify-memo`, branch `agent/palette/intake-verify-once-memo-20261105` from `origin/main` 8b3acb08. Uncommitted at the time of writing.
- Changed interfaces: `file_ref` gains the opt-in memo (`digest_memo`, `digest_memo_shared`, `active_digest_memo`, `DIGEST_MEMO_SETTLE_NS`); `_verify_materialized_files(..., recording_ids=)`; `run_parents_in_parallel`; `prepare_transfer_parent_recordings(..., parent_workers=1)`; `import_parents(..., parent_workers=1)`; `import_delivery(..., parent_workers=DEFAULT_PARENT_WORKERS)`; `--parent-workers` on both CLIs; `execution` in the status JSON. Test stubs of `import_parents` accept the new keyword.
- Required CI (import boundaries, file-size ratchet, access ratchets, non-GPU test shards) has not run on this branch; the boundary, ratchet and access checks pass locally. Not merge-ready until CI is green.
- Live intake stays pinned at 2931ddc8. Any re-pin goes through the "Palette H5 import readiness" session on the user's word.
