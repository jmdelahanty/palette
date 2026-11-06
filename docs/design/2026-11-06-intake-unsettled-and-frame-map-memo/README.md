# Intake: explain unsettled digests, parse each frame map once per job

- **Status:** implemented on branch `agent/palette/intake-unsettled-frame-map-memo-20261106`; not merged, not deployed.
- **Owner:** intake-duration-investigation session (for the user); written 2026-11-06.
- **Requested by:** the user, as follow-ups 1 and 2 of the first live measurement of the verify-once memo (PR #430, live on 45b1ac30).
- **Builds on:** `docs/design/2026-11-05-intake-verify-once-memo/`.

## What the first live delivery showed

Delivery `2026_11_05_21_17_22` (4 arenas, reactive phototaxis, 239.5 GB, 252 inventory files, 6.6 GB of CSV/JSONL the sealer does not attest; LSF 155226926, `-n 1`, 16 GB) on 45b1ac30: wall 1000 s, CPU 1183 s. `execution.digest_memo` = hits 22637, misses 1962, unsettled 1227.

- Hashing fell from about 30 passes to about 8, but 1227 entries were refused by the 5 s settle window, about five passes' worth over the 236 non-attested files. The window should cost less than one pass after a 30 s hard-link phase, so the cause is unknown: a compute-node clock ahead of or behind the NFS server, or something linking or touching files later than expected. The status records only the count.
- 5 min 13 s passed between the last hard link and the first projection file, with all four builders in flight. Each builder's start and end run `verify_transfer_snapshot` over the whole delivery, and each verification parses every clip output's metadata CSV (`frame_map`) to rebuild the snapshot. That parse is CPU-bound, not covered by the digest memo, and at `-n 1` four parallel builders serialize on it.

## Decisions

| Question | Decision |
|---|---|
| What an unsettled entry records | Its path, its age (`now - ctime`) and its (device, inode). The status's `digest_memo` gains `unsettled_files` (distinct files), `unsettled_age_s` (`{min, max}` seconds, `null` when none) and `unsettled_sample` (the first 16 entries as `{path, age_s}`). A negative age means the file's ctime is ahead of this host's clock. Each entry is also logged at DEBUG. Additive keys; `hits`, `misses`, `unsettled` unchanged. |
| Where frame-map parsing is memoized | In `frame_map` itself, under the same `digest_memo()` block, keyed by (device, inode, kind, previous, crop_offset) and bound to the same five-field stat signature with the same settle rule. A call with a row callback (the builder streaming its own parent's rows) always parses. The parser body is `_frame_map_rows`, unchanged. |
| What a hit returns | A deep copy of the validated mapping, so a caller cannot change a later result. |
| When an entry is kept | Only when the stat signature before and after the parse is the same and the file has settled; otherwise the next call parses again. No new refusal was added. |
| Where it is recorded | `digest_memo.frame_maps: {hits, misses, unsettled}` in the status; `frame_map` unsettled entries share the sample and age fields above. |
| Out of scope | Scoping the builder's whole-delivery verification to its parent (the memo makes the repeated verification cheap instead); waiting for files to settle after hard-linking; changing `-n`. |

## What must remain identical, and does

- Every byte intake writes; every refusal of `frame_map`, `verify_transfer_snapshot` and the builder. A rewritten metadata file moves its ctime and is parsed again; a changed file under a warm memo is still refused by verification. The streaming builder call compares its own parse against the snapshot as before.
- Outside a `digest_memo()` block, `frame_map` parses on every call, as before.

## Tests

`tests/unit/fisheye/test_transfer_digest_memo.py` gains 11 tests: unsettled entries described by path and age; the sample bounded while the counts are not; a settled memo reports no detail; `frame_map` outside a memo parses every call; inside a memo an unchanged file is parsed once and the caller owns its copy; the arguments are part of the key; a row callback always streams; a rewritten file is parsed again; a file written moments ago is not memoized; snapshot verification under a memo parses each metadata file once and the second verification parses nothing; a changed metadata file under a warm memo is refused. The status test in `test_intake_parent_parallelism.py` checks the new fields. 41 tests in those two files; 818 across the intake, transfer, organizer, snapshot, poller and canary files.

## Replay bench without new recordings

Delivery `2026_11_05_17_00_17` (119 GB, solid black) can be rebuilt as a staging directory from hard links out of its imported recording folders, inverting the saved `organization_plan.json` (`/groups/johnson/johnsonlab/jeremy/scratch/intake_replay/rebuild_replay.sh`). Hard links keep the mtime the sealer attests against, so verification accepts the replay. A plan-only run matched the live plan phase (35 s). `run_citrus_session_import --apply` with `--dest-root` and `--run-dir` under scratch writes only there; the workflow lock lives in the staging copy, registration is separate, and retirement unlinks the scratch links only. Measured 2026-11-06 on the workstation (many cores, so parallel builders do not serialize as they do at `-n 1`; CPU seconds are the comparable number):

| Run | Wall | CPU | Plan to first projection (Cam2010093) | Clip index | Digest memo | Frame maps |
|---|---|---|---|---|---|---|
| 45b1ac30 (live) | 439 s | 492 s | 3 min 13 s | 1 min 36 s | hits 22222, misses 2377, unsettled 1695 | not memoized |
| this branch | 163 s | 209 s | 37 s | 6 s | hits 22218, misses 2381, unsettled 1699, 251 files, ages 0.004 to 4.999 s | hits 272, misses 16, unsettled 0 |

Both runs imported and retired all four parents; the stimulus-import and finalize phases (about 1 min and 40 to 50 s) are unchanged.

**What the unsettled sample says.** Every unsettled entry is a non-attested CSV/JSONL whose age is positive and under 5 s. No clock offset is involved: hard-linking a staged file into its parent directory moves the inode's ctime, and the materialization check plus every builder's opening verification run within 5 s of the last link, so about seven passes over the 251 files are refused and hashed again. The follow-up is to wait, once, until the newest linked inode has settled (at most 5 s) before the first post-link verification; it is not in this branch.

## Handoff

- Worktree `/home/delahantyj@hhmi.org/gitrepos/palette-worktrees/intake-verify-memo`, branch from `origin/main` 8d60d5a4.
- Changed interfaces: `_DigestMemo.record(info, sha256, label="")`, `_DigestMemo.report()`, `lookup_frame_map`/`record_frame_map`, `frame_map` memoized without a callback, `_frame_map_rows`, `UNSETTLED_SAMPLE_SIZE`; `importing._memo_record` returns `memo.report()`; status `execution.digest_memo` gains the keys above.
- Local checks: six access/size ratchets pass; `git diff --check` clean. Required CI not yet run.
- Live intake stays pinned at 45b1ac30. Any re-pin goes through the "Palette H5 import readiness" session on the user's word.
