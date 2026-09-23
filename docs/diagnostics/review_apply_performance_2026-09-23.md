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
coordinates, point provenance, scientific QC, training eligibility, component
approval policy, dense mask authority, stale-cache markers, and authority
activation policy remain unchanged. The mask change does not recompute derived
contours or regenerate tail keypoints. Approval remains separate from Apply.

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

## Validation and deployment boundary

Required coverage includes canonical and malformed reason metadata, legacy
migration/width growth, untouched rows/components, duplicate row revisions and
UTF-8 truncation parity, physical chunk-write counts, interrupted writes and
retry, fresh revision/lifecycle checks under the mask lock, and supported
checkpoint HTTP routes. Full required CI is a separate gate on the final exact
commit. Benchmark speed alone does not establish acceptance.

This implementation work does not change the running preview, production
labels, approval records, selectors, or the shared `/groups` checkout. Final
commit, test results, CI evidence, and deployment status belong in the handoff.
