# Mask Apply QC and tail refresh integration

Owner: `/root`. Isolated worktree
`/tmp/palette-mask-apply-qc-tail-integrated-20260923`, branch
`agent/palette/mask-apply-qc-tail-integrated-20260923`.
The unchanged live preview is `f772816a10ad1bc97b42e14fa986665e7aab5d23`.

The user authorized feature-branch pushes, draft PRs, and required CI, then
integration and combined validation after both incoming checks pass. They chose
automatic tail regeneration while retaining manual landmarks, and a separate
refreshed task offered without replacing the active keypoint task/session.

## Incoming contracts and integration work

- Mask candidate `87ed635e8bdd57b1bdc9bd6443bc01847d628fe9`: PR180,
  required CI35824742949 passed in full for this exact commit. Efficient dense writes plus full mask-local QC and
  durable secondary-effects retry. Prior performance-only CI35821242004 passed
  for exact7dd5; that does not validate this expanded commit.
- Tail candidate `dd685e3ab00e4c906faf02f42ac9499ca8bef487`: PR183,
  required CI35824811134 passed in full for this exact commit. New immutable mask/pose seeds and editable successors
  through the maintained publisher, preserving scientific recipes, ROI identity,
  coordinates, original versions/selectors, and recorded manual edits/clears.

The integration calls tail refresh after locked mask QC in both initial Apply
and same-ID recovery. Audit/registry effects finish before the durable receipt
is marked complete. These are application workflow additions; they do not
invent a scientific acceptance schema, copy approval, or activate authority.

Read-only review identified ownership corrections: retry audit/completion must
remain within the mask lock, and a sibling component task cannot advance the
same run while an earlier receipt has pending effects. The run-level guard must
be checked inside the lock before writing; an earlier HTTP preflight alone is
insufficient. Receipt queries use summaries/task source bindings without loading
dense checkpoint payloads. Existing per-task checkpoint and digest grammars are
unchanged.

The browser must retain current pixels/navigation while background Apply
finishes. Only lifecycle/status fields may be adopted; a newer foreground
request prevents an older Apply from replacing its state. Error messages must
name the pending operation and retain its actionable detail, since QC itself
may be complete while tail publication or registry work is pending.

## Version and offer semantics

The new dataset captures labels already applied to Zarr at regeneration time.
Later checkpoints/manual edits on the original active task stay in that task's
original version; independent annotation drafts are not silently merged or
retargeted. Pending pose checkpoints found before generation must first be
applied. New checkpoints arriving during generation remain on the original
task and are not discarded or represented as part of the captured source.

A durable successor event records the exact Apply ID, source mask revision,
source proof, published paths, counts, failures, and offered task identities.
Exact lookup must not depend on scanning only a recent page of events. On reload,
an older offer must not masquerade as the current mask revision. On a later
registry/audit retry, an already completed successor is validated and reused as
the same historical version without rebinding to subsequent source-pose edits.
Partial publication without a completed effect record remains subject to the
publisher's source-change refusal. No successor task/session is reset by retry.

## Validation contract

Prepared preservation tests cover real recovered/native HTTP Save→Apply→open
successor, still-failed rows, source-session/manual-label retention, lost-response
and reload recovery, tail-publication failure and same-ID retry with unchanged
pixels/revisions, sibling component exclusion, and receipt completion while the
physical mask lock is held. SQLite tests cover cross-component/archive identity
and exact event lookup beyond a recent-event page. Node executes the shipped
mask JS for brush-pixel preservation, navigation/checkpoint freshness, durable
offer display, and actionable same-ID retry.

The three Node tests reproduced the unchanged preview's reload/state/error
regressions before implementation and pass after the fixes. The combined local
suite passed 107 tests in 188.67 seconds; four additional recovered/native tests
reject damaged or missing completed publications. The eight real HTTP cases
also verify registry failure recovery after later source labels/checkpoints,
without resetting an already completed successor task. Static checks passed:
import boundaries, file-size and authority ratchets, metadata modes, observation
literals, contract freshness, compilation, and JS syntax. Generated inventories
are regenerated for the combined tree. Full required CI must separately
validate the resulting combined commit.

The earlier core benchmarks and isolated QC/tail timings are separate scopes;
see the incoming branch notes and workstation evidence under
`/tmp/palette-review-apply-validation-20260923`. A combined HTTP smoke must report
its own cost rather than add timings from different datasets or describe core
writer speed as full Apply speed. The fresh local 184-row copy smoke saved 41
unchanged body-mask checkpoints through HTTP in 2.55 seconds; whole HTTP Apply
took 48.59 seconds, including dense writes, full QC, regeneration, publication,
SQLite task/audit writes and response construction. Registry projection was not
configured, and this is local `/tmp` storage, not a live NFS latency claim.
All 184 rows remained tail-valid/training-eligible; all 758 manual points were
retained. Source pose bytes and the source session were preserved, and the
offered successor opened through the real HTTP task route. Evidence:
`/tmp/palette-review-apply-validation-20260923/combined-http-smoke.json`.

## Status

This note records a locally validated integration. Exact incoming and
combined CI results, local tests, failures, benchmark scope, and the final commit
belong in the accompanying handoff. Until all required checks pass, the candidate
is incomplete and must not be described as merge-ready. No deployment, preview
restart, live-label mutation, task retargeting, selector activation, or shared
checkout update is authorized as part of this integration step.
