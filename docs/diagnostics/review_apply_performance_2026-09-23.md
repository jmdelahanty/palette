# Review Apply performance

Owner: `/root`, with implementation by a Sol medium agent. Branch:
`agent/palette/review-apply-performance-20260923`. The isolated worktree is
`/tmp/palette-review-apply-performance-20260923`.

The prerequisite is the deployed preview commit
`f772816a10ad1bc97b42e14fa986665e7aab5d23`, whose required CI passed in
[run 35787456627](https://github.com/jmdelahanty/palette/actions/runs/35787456627).
Fetched `origin/main` at `46d8dce8fed3bb82134c6d3fe69ae03b94106b43` is an
ancestor; the preview contains the newer checkpoint implementation. This patch
preserves that implementation rather than restarting from the older main tree.

## Change classification and preservation

The storage optimizations are:

- Stamp the five existing reason-storage attributes only when a value or its
  JSON type differs. Repair changed attributes together; preserve legacy
  reason migration, width growth, unrelated metadata, and exact schema values.
- Group subject-mask pixel, flag, revision, timestamp, and reason writes by
  row chunk. A serial caller owns each read/modify/write operation. Preserve
  unselected rows/components and repeated-row ordering.
- Reuse one target mask row chunk during browser checkpoint preparation,
  after validating each checkpoint. Do not retain another full stack batch.

There is also an explicit ownership enforcement correction: browser mask Apply
participates in the existing refined-run write lock and resolves a fresh mutable
run under that lock. A cached browser runtime must not hide a newer edit revision
or a sealed target. Pre-write rejection retains the saved checkpoints.

The keypoint recovery journal, checkpoint/receipt grammar, row identities,
coordinates, point provenance, component approval policy, dense mask authority,
and authority activation policy remain unchanged. Approval remains separate
from Apply.

## Expanded mask Apply effects

Browser mask Apply now records the pixel receipt before secondary effects,
then refreshes full canonical mask-local metrics and QC reasons for every
component. It refreshes body and swim-bladder contours and, when both eye
channels exist, the maintained eye ellipse geometry, eye contours, and eye-pair
separation. The named `palette.browser_subject_mask_apply_full_qc_v1` policy
records the effective full-metric recipe and edit revision. Existing compatible
metric, QC-policy, geometry, and contour declarations are preserved; conflicting
declared contracts are rejected before derived writes. Reason-tag refresh
preserves manual/operator tags. Dense pixels, per-row revisions, manual review
decisions, and source bindings are not changed by this derived work. Compact
mask caches remain stale independently.

The run lock covers the fresh mutable run resolution, pixel Apply, QC refresh,
and effect completion. Metrics and contours remain stale until full readback
validation succeeds. A failure after the pixel receipt leaves secondary effects
pending; retrying the same Apply ID reruns QC without rewriting pixels or row
revisions. A retry first marks derived metrics and contours stale, so a second
failure cannot leave an earlier freshness claim visible. Pending effects remain
visible with a resumable Apply ID and block component review-status changes and
task completion. Audit and registry refresh must succeed before the receipt is
marked complete. Existing approval decisions are not inferred from QC values.

Automatic tail-point regeneration is a separately owned, versioned successor
derivation and is not implemented by this branch. The Apply route has one
post-QC insertion point before audit, registry, and effect completion for that
integration. The browser can display successor task IDs and counts when the
separate tail helper returns them; it does not change or navigate the active
task automatically.

## Benchmark method

The original inspection found 2,219 metadata rewrites totaling 616.7 MB during a
184-row keypoint Apply. Most were repeated reason-schema stamps carrying the
growing recovery record. The observed live receipt took 205 seconds; that
historical timing is not a controlled estimate of network cost.

Validation uses independent local copies with unconsolidated metadata:

- Keypoints: reconstruct the 184 pre-Apply coupled row states from the durable
  receipt, stage the original user operations normally against each copy, run
  core Apply, and compare every resulting row with its intended digest.
- Masks: use 41 real saved payloads with the same controlled source-seeded
  pre-edit state for both writers. With the clock fixed for parity, compare all
  refined-run arrays and attributes, including unedited components, revisions,
  provenance, and unchanged derived caches. This is a core-writer comparison,
  not a historical pre-Apply reconstruction or end-to-end browser measurement.

Copy/preparation time, HTTP, UI reloads, and registry secondary effects are
excluded from these benchmarks. Source archives and the live SQLite sidecar
are read-only; mutable outputs are disposable `/tmp` copies. The workstation
evidence and replay harnesses are under
`/tmp/palette-review-apply-validation-20260923`.

Core-only replay results on exact `7dd5bbb0ee183449b188bafa856d1796557e3522`:
keypoint Apply took 34.2476 s before and 17.6483 s
after reason-stamp reuse, with metadata writes falling from 2,219 to 374 and
all 184 row digests matching. Mask core Apply took 7.4178 s before and 0.4170 s
after chunk batching; total store sets fell from 317 to 38 and dense mask-chunk
sets from 41 to 2. Every compared refined-run array and attribute matched.
These timings exclude the new QC/eye refresh, HTTP, audit, registry, and tail
derivation work and must not be presented as full browser Apply timings.

For the expanded QC scope, a private copy of the frozen 200-row RedScare input
with four components completed full mask-local QC and eye refresh in 11.08 s.
The copied dense `masks_roi` files remained byte-identical to the frozen input.
This is a compatibility smoke on a private copy, not a controlled before/after
benchmark or a production activation.

## Validation and deployment boundary

Required coverage includes canonical and malformed reason metadata, legacy
migration/width growth, untouched rows/components, duplicate row revisions and
UTF-8 truncation parity, physical chunk-write counts, interrupted writes and
retry, fresh revision/lifecycle checks under the mask lock, and supported
checkpoint HTTP routes. The expanded scope also has valid eye geometry/pair
readback, incompatible contract refusal, halfway-QC fault and repeat retry,
unchanged pixels/revisions, and pending review/task-completion refusal tests.
The focused related suites passed 118 tests outside the sandbox on the expanded
candidate. Generated Zarr census artifacts are refreshed for the new helper.
Full required CI is a separate gate on the final exact commit. Benchmark speed
alone does not establish acceptance.

This implementation work does not change the running preview, production
labels, approval records, selectors, or the shared `/groups` checkout. The
expanded commit remains incomplete for integration until its own required CI
passes and the separately owned tail regeneration is integrated and validated.
