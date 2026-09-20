# Labeling save latency and keypoint checkpoints

Status: combined implementation assembled, locally validated, and independently
reviewed without a blocking finding; required combined PR CI remains pending.
Not deployed or activated.

## Ownership and prerequisite

The prerequisite is `dc798f1771a729a744fc99ab2e1f7b006157772a`, the
snout-inclusive recovered-review implementation from PR #173. All required
checks passed in CI run `35405067096`. Its base is the fetched `origin/main`
commit `46d8dce8fed3bb82134c6d3fe69ae03b94106b43`.

The running port-18841 preview and its source checkout remain unchanged during
implementation. The user may keep labeling. Deployment requires a coordinated
restart after the user saves the current edits; no restart is included in this
implementation handoff.

| Owner | Branch / worktree | Scope |
| --- | --- | --- |
| Root | `agent/palette/labeling-save-fixes-20260919`, `/tmp/palette-labeling-save-fixes-20260919` | Integration, independent benchmark, final validation and handoff |
| Sol xhigh store agent | `agent/palette/labeling-store-lifecycle-20260919`, `/tmp/palette-labeling-store-lifecycle-20260919` | Shared SQLite lifecycle, initialization, transactional checkpoint contract |
| Sol xhigh keypoint agent | `agent/palette/keypoint-checkpoint-save-20260919`, `/tmp/palette-keypoint-checkpoint-save-20260919` | Keypoint checkpoint payload, overlay/apply, HTTP routes and audit lineage |
| Sol xhigh browser agent | `agent/palette/keypoint-checkpoint-ui-20260919`, `/tmp/palette-keypoint-checkpoint-ui-20260919` | Browser controls and executable JavaScript tests |
| Sol xhigh reviewer | Read-only review of those exact worktrees | Contract, failure, concurrency and preservation review |

The store agent owns the shared checkpoint contract. Incoming commits require
successful required CI before integration; the combined commit requires its
own successful CI. The keypoint backend branch will assemble the accepted store
and UI changes and become the final delivery candidate. Root will adopt that
exact combined commit after its required CI succeeds; the root UI-only worktree
is an intermediate evidence workspace. This is a performance/editing workflow work item, separate
from authority-consolidation and scientific-analysis queues.

## Declared changes and preservation

Connection cleanup and eliminating repeated initialization/image work are
performance and resource-lifecycle changes. Browser checkpoint-first saving
is an intentional HTTP behavior change for mutable keypoint sessions. The
server selects the mode; a client cannot request direct writes as a bypass.
The response and UI must distinguish checkpointed edits from applied labels.

Direct backend/desktop keypoint APIs remain compatible. Immutable keypoint
browser sessions keep explicit direct-delta saving, without an Apply control.
Transactional immutable-delta publication is outside this mutable checkpoint
slice. The existing scientific schema, point order, coordinate conventions,
row identity, manual landmark provenance, QC and visibility rules are
preserved. Missing or out-of-crop points in recovered training rows remain
invalid. Existing labels are not migrated or rewritten during rollout.

Checkpoint saves must not mutate Zarr eligibility, derived data, deltas,
selectors or registry projections. Export readers continue to consume the
last applied labels; pending edits are not exported. Review approval and task
completion remain blocked while pending or uncertain edits exist.

Apply must validate its whole claimed set before writing, serialize physical
writers, preserve a durable recovery state, and never issue a false applied
receipt. Same-ID retries must not apply newer edits after a subsequent save
replaces a row's current checkpoint. This does not add atomic multi-array
visibility to mutable Zarr; no such publication guarantee is claimed.

The shared sidecar schema advances additively from v6 through v7 to v8, retaining full
checkpoint payloads in durable apply records so later row saves cannot erase
earlier retry evidence. This intentionally duplicates payload storage,
including dense mask payloads. A summary-only historical receipt API is
deferred to preserve the existing applied-checkpoint getter's semantics.
Version 8 adds a persisted per-row checkpoint digest and a compact descriptor
query so status construction avoids decoding all dense payloads. Full payload
reads and claims verify the stored digest; Apply still validates the scientific
binding and current row content. Migration preserves historical JSON bytes.
Durable audit/registry follow-up is opt-in: the new keypoint Apply requests it,
while existing mask callers retain their previous completion behavior. Pending
receipts retain the caller's expected snapshot digest for cheap restart state;
that expected binding does not replace full-content validation before writing.
Claims remain bounded to 1,000 rows by default; the keypoint snapshot digest
and claim must select the same deterministic batch. Additional pending rows
remain visible for subsequent apply batches.

Mixed mask applies must atomically finalize applied rows and release the
unapplied remainder. Replay reports the recorded release count, rather than
claiming to perform a new release. Refusing a newer unsupported sidecar schema
is an explicit enforcement correction rather than a performance-only change.

## Investigation evidence

The live 184-row recovered run used mutable in-place keypoint editing with no
registry refresh binding or matching downstream keypoint-dependent mask runs.
Its 19-point coordinates have one row per physical chunk. Confidence,
eligibility, manual-origin flags and reason arrays share 184-row chunks.
Browser source showed no accumulating image history or event listeners.

The original keypoint Save+next reads/encodes the old crop for an audit payload
that discards its image, writes multiple Zarr arrays, calculates progress, then
performs a second request that loads the new crop and calculates progress
again. It lacks the mask editor's in-flight guard. Overlapping requests were
a code risk, not an observed explanation of the user's latency.

Read-only `/proc` inspection found 269 open file descriptors to the live
labeling SQLite database. The original store retains every request thread's
connection until shutdown. Store methods also repeatedly perform full schema
bootstrap statements and commits.

## Scratch baseline

A separate process used a fresh `/tmp` database, synthetic assignment/task,
and ephemeral loopback HTTP port. It never accessed the preview or live Zarr.
The endpoint was `GET /api/me/tasks`, using the real handler and store.
SQLite was the `scripts/py` runtime's SQLite 3.52.0. SQL trace callbacks were
enabled equally for comparison runs. This measures shared HTTP/store overhead,
not image transport, NFS latency, or full keypoint-save latency.

At prerequisite commit `dc798f1771a729a744fc99ab2e1f7b006157772a`:

- Four sequential batches of 50 requests retained 51, 101, 151 and 201 database
  descriptors, starting with one. Median request times were 8.12, 8.18, 8.18
  and 8.21 ms; this short run did not reproduce progressive sequential delay.
- Forty requests from eight concurrent clients brought the total to 241
  retained descriptors. Median latency was 102.10 ms and p95 was 168.79 ms.
- Across 240 requests, the handler performed 71,760 `CREATE` statements and
  3,120 schema-version upserts/commits: thirteen initializations per request
  on this endpoint.
- All database descriptors closed only at server shutdown.

The leak and repeated bootstrap are measured defects. Their exact contribution
to the live user's progressive wait remains unmeasured.

At store-fix commit `03d8030d5e6461cda817108a899479146e99b7e7`, the same
scratch benchmark retained exactly one database descriptor after every batch,
then zero after shutdown. Sequential batch medians were 3.87, 4.22, 4.01 and
4.03 ms. Eight concurrent clients had median 37.89 ms and p95 53.29 ms. During
all 240 requests there were no `CREATE`, schema-version upsert or commit
statements; SQL trace contained only connection pragmas and ordinary selects.
This verifies bounded request connection lifetime and removal of hot-path
bootstrap writes. It is not an end-to-end keypoint/Zarr speed measurement.

The same benchmark was repeated at final store commit
`6e25b7ea8ce61627c3fb2d1b5d41396ec3f8ab3e`: one database descriptor remained
through all 240 requests and zero after shutdown. Sequential batch medians
were 4.02, 4.04, 4.03 and 4.26 ms; eight-client median was 32.93 ms and p95
42.19 ms. Request SQL contained only 960 connection pragmas and 2,400 selects.

A separate real producer/HTTP benchmark created 184 synthetic 128-pixel crops
with the recovered 19-landmark schema in a fresh `/tmp` archive. Each Save+next
filled the four fin points and loaded the next ROI, using the browser's original
two-request path when the server did not return a folded ROI. On prerequisite
`dc798f1771a729a744fc99ab2e1f7b006157772a`, successive 46-row blocks had medians
97.86, 83.14, 81.04, and 84.79 ms. Database handles grew from 3 initially to
95, 187, 279, and 371. Overall median was 83.65 ms and p95 126.92 ms. This
confirms the connection leak through the real keypoint Save+next path but does
not reproduce progressive local latency. It excludes live data and NFS.
Script: `/tmp/palette-keypoint-session-benchmark-20260919.py`; baseline evidence:
`/tmp/palette-keypoint-session-baseline-20260919.json`.

On the stable assembled candidate, the same 184-row benchmark passed with
zero archive-byte changes from checkpoint saves and one database handle at
every boundary. An isolated repeat (without the other root benchmark/smoke
processes) measured successive 46-row medians 64.60, 34.57, 35.91, and 33.53 ms;
overall median 35.91 ms and p95 65.58 ms. This is approximately 2.3 times faster
than the 83.65-ms baseline median for the local fixture. Other workstation
activity is uncontrolled; this is not a live NFS estimate. The earlier run
alongside the other scratch checks measured 29.86-ms overall median and is
retained separately rather than substituted for the isolated comparison.
Evidence: `/tmp/palette-keypoint-session-final-20260919.json` and
`/tmp/palette-keypoint-session-concurrent-20260919.json`.

## Validation and integration status

Store implementation: clean commit `03d8030d5e6461cda817108a899479146e99b7e7`.
The new lifecycle tests passed 16/16, existing assignment-store tests passed
134/134, and web route/security/config tests passed 51/51, outside the sandbox.
The initial regression suite had eight failures and two passes before the
fixes. Independent review found no remaining store blockers, and the exact
commit's comparison benchmark passed above. CI run `35466732429` failed the
generated-artifacts gate because the new module changed the scanned module
count; dependent test shards were therefore unrun. Intermediate correction
`e9d13195ca3793dde938e033943098b0b5cb82ce` updates exactly two generated JSON
counts and passes the generator check plus 14 census/inventory tests. The final
compact descriptor API and successful required CI are recorded below.

Final store commit: `6e25b7ea8ce61627c3fb2d1b5d41396ec3f8ab3e`, clean and
independently reviewed without a blocking finding. Its 22 lifecycle tests and
185 assignment/web/security/config regressions pass. Static, generated-census
and file-size checks pass. Draft PR #175 passed all 24 required PR-based CI
checks in run `35468881963`. Descriptor-query medians were 0.240 ms for
184 rows and 1.072 ms for 1,000 rows, versus 6.869/40.971 ms for full payload
decode and verification. A representative unchanged 87,588-byte dense-mask
checkpoint digest took 0.369 ms median. The migration cursor is bounded to
32 rows; review noted that no migration fixture crosses that boundary, while
the operational 1,005-row test covers successive 1,000/5-row claims.

Browser commit `2050ffe41b49fe2bd5e4e916d4b18f87dccf95e9` passes ten focused
JavaScript/renderer tests and independent review. Manual CI run `35466781762`
passed all 23 individual jobs but failed `ci-required`, because the repository
permits manual full-CI fallback only on main. Draft PR #174 requests the
supported PR-based evidence in run `35468019685`, which passed all 24 jobs,
including `ci-required`. Root integrated that accepted UI commit into the
isolated intermediate branch as `88f78f27`; it is not deployed. Final delivery
uses the backend candidate with the accepted store/UI branches assembled there.
Combined change CI remains pending. The prerequisite CI does not establish
acceptance of these changes.

An independent WIP checkpoint-state benchmark measured median 1.29, 40.56 and
277.91 ms at 1, 184 and 1,000 active rows with a synthetic 19-point payload and
3,033-byte real checkpoint metadata envelope. This excludes Zarr and image I/O.
It exposed repeated full-payload parsing/hashing in status construction; the
shared descriptor work addresses that measured growth. The final state benchmark
compares compact descriptors with full decoded rows through the same state
logic and asserts identical returned states. At 1, 184, and 1,000 pending
rows, compact medians were 0.155, 0.869, and 4.068 ms; full-row medians were
0.359, 25.302, and 174.252 ms. The final real metadata envelope was 4,178 bytes,
so these values should not be treated as a controlled comparison against the
earlier 3,033-byte WIP envelope. This remains SQLite/state-only evidence.
Artifact: `/tmp/palette-keypoint-state-final-20260919.json`.

A fresh real recovered 19-point producer/HTTP smoke on the backend WIP verified
zero archive-byte changes during Save, folded next-row response, pending
overlay after runtime reopen, actual Zarr Apply, training eligibility, and
per-landmark origin/manual flags. Same-ID replay after a subsequent edit failed
against the old v6 store, as expected from its history loss. That partial WIP
result was not final acceptance. The complete smoke now passes on the assembled tree:
Save changes zero archive bytes, the next ROI is folded into its response,
SQLite overlays recover after runtime reopen, Apply persists actual coordinates
and eligibility, per-landmark origins/manual flags remain correct, same-ID
replay after a later same-row checkpoint leaves the newer edit pending and
canonical labels unchanged, and original seed arrays remain byte-identical.
Artifact: `/tmp/palette-keypoint-http-final-20260919.json`.

These independent checks ran against `6be9dd2d9a99a496c6c3e350bd6b6d9c0c5f24f0`
plus the stable final integration changes in the three `web_keypoint_checkpoint*`
Python modules, keypoint JavaScript/template, and three associated test files.
The final commit/CI binding is recorded in the delivery PR. The final combined
focused run passes 127/127: 59 keypoint backend/browser tests, 22 store
lifecycle tests, and 46 complete labeling route/admin projection tests.
Generated-census, file-size, metadata-mode, Python lint/compile, and diff checks
pass. The first combined required PR CI, run `35478643305` for exact head
`1284a4fd644ed6b523f2d774842a5de59c1ab9c7`, failed non-GPU shard 5:
three handoff/export tests reported the browser workflow scope contract as
not ready. The keypoint capability had introduced an unrecognized
`training_zarr_write_mode`, which cascaded into operator-validation
`needs_review`. Thirteen checks had passed and nine were still running when
the correction began; that head remains blocked regardless of their eventual
results. The correction reuses the established `session_checkpoint_then_apply`
vocabulary for the mutable primary path and retains the explicit immutable
direct-delta compatibility field. The original three failures plus an unknown
mode refusal test pass 4/4, and the broader workflow/handoff/route/admin suite
passes 51/51. Required combined PR CI for the corrected commit is unrun.

No changes have been deployed to the labeling preview or shared data.

## Completed test recording

The live task was marked complete at `2026-09-19T21:13:03.098797+00:00`.
Read-only unconsolidated inspection of
`refined_keypoints_runs/head_tail11_fins_edit_tail19_review_v002` in the
`2026-01-28T21-27-20Z_arena_4_Feeding_recovered_training.zarr` archive confirmed:

- All 184 rows contain all 19 finite landmarks inside their 512-pixel crops.
- All 184 rows have `training_eligible`, `usable_keypoints`, `refined_success`,
  `geometry_valid`, `confidence_valid`, and `edit_applied` set true.
- Historical automatic `tail_valid` remains false on rows 21 and 63. Both
  original seed rows had zero finite tail landmarks. Both edited rows now
  contain all 11 finite tail landmarks, each explicitly marked manually
  edited with origin code 3. These flags describe the original derivation,
  not missing current manual labels.
- `stage_selector_eligible` remains false. Task completion and row eligibility
  do not constitute registry activation or broader scientific acceptance.

The current preview uses direct Zarr saves, so this completed task's labels
are already applied. No new checkpoint workflow has been deployed.

## Integration approval block

The backend candidate was initially clean at
`5dc74afc8af459a8c21a7caf7c6899ebb012ef1b` when integration was blocked.
Automatic approval review rejected the attempted store-chain cherry-pick,
stating that combined required CI must pass before integration. No cherry-pick
started. Both incoming final branches independently passed all required CI.
Root requested explicit permission to assemble the candidate and then run full
combined CI before deployment. The user explicitly approved: "Approve
integration, then combined CI." Integration can proceed within that approval;
the assembled candidate still requires full combined CI before deployment.
No equivalent patch or alternate integration was attempted while approval was
pending. The approved store chain was integrated as `d367af57`, `4ad96b06`,
and `48dc987f`; the approved UI commit was integrated as `6be9dd2d`. The final
consumer binding correctly distinguishes compact descriptor digests from full
claimed/history rows: full rows still recompute the shared owner digest and
reject tampered content before writes. The pending audit/registry recovery UI
retains the original Apply ID and digest and offers “Finish Apply.”
