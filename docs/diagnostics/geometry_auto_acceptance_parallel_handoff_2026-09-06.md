# Geometry automatic-policy development: parallel handoff

Status authority: `GEOM-AUTO-001` in the
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md).
This handoff records ownership/evidence, not a second status queue.

## Authorization and preserved contracts

On September 6 the user authorized parallel implementation of automatic
geometry acceptance with additional metrics, preferring the top dish rim while
acknowledging that Hough fits do not reliably distinguish projected rim edges.
The user subsequently authorized committing and pushing development branches.
No production activation, deployment, shared-checkout update, historical
rewriting, or main-branch merge is authorized by this development task.

This is an explicit scientific recipe/policy extension, not behavior-preserving
cleanup. Existing recipes, human-reviewed paths, persisted semantic labels,
source coordinate bindings, and historical digest bytes remain unchanged.
The new top-rim preference must not claim that an outer observed image edge is
physically identified as the top rim. Ambiguity remains explicit. The blind
probe must not receive acquisition geometry as a fitting target.

The approved-reference catalog is an index of existing producer approval and
validated source/pixel evidence, not a second authority or approval gate. A
missing H5 runtime-application claim must not invalidate a sufficient folder
geometry supplier. The initial policy is shadow-only: no acceptance receipt,
selector, changed radius/center/tolerance, or automatic promotion is produced.
Calibration thresholds require exact applicable evidence; there are no enabled
numerical defaults. Existing OQ4a/OQ5 controls remain prerequisites for a later
operational policy. Historical cohort aggregates are not fresh holdout proof.

## Reconciled ownership

All 77 existing worktrees were inspected at their exact commits before new
implementation. No dirty geometry source/test/document overlap was found.
Each new worktree starts at the known-green main commit
`925f8c6285499d39a99475945ba1732dba7f6834` (24 required checks passed in CI
run `34056284642`). Root coordinates interfaces and owns the queue and this
handoff; workers must not independently edit them.

| Worker | Worktree / branch suffix | Owned interface |
| --- | --- | --- |
| `geometry_auto_policy` | `/tmp/palette-geometry-auto-policy-20260906`; `agent/palette/geometry-auto-policy-20260906` | Pure versioned shadow policy; existing comparison adapter; read-only CLI; dedicated tests/handoff |
| `geometry_rim_metrics` | `/tmp/palette-geometry-rim-metrics-20260906`; `agent/palette/geometry-rim-metrics-20260906` | Existing blind probe recipe/metrics; metric extension validation in fit-review owner; dedicated preservation/refusal tests/handoff |
| `geometry_reference_catalog` | `/tmp/palette-geometry-reference-catalog-20260906`; `agent/palette/geometry-reference-catalog-20260906` | Existing native-candidate planner-backed reference catalog in materializers; read-only CLI; dedicated tests/handoff |

Policy owns the consuming schema. Metrics owns `validate_rim_metrics` and
`validate_rim_metrics_consensus` in the existing fit-review materializer, so
production adapters need not import diagnostics. No legacy fit-review bytes
change when the opt-in extension is absent. Catalog remains in the materializer
layer to reuse the full native planner without reversing shared-layer imports.

No incoming change may be integrated into a candidate until its exact required
CI is green; the resulting combined candidate then needs its own required CI.
Read-only cross-worktree diagnostic tests must disclose exact sources and seams
and cannot stand in for an integrated candidate or production acceptance.

## Read-only cohort evidence

Earlier inspection found 84/84 hash-valid, operator-approved acquisition
geometry suppliers with no quality flags: eight registration digests and 32
camera/registration cells. All 84 stored comparison records were valid and
complete: 62 recorded same-feature and 22 recorded different-feature judgments.
Gate-circle IoU ranged from 0.971883 to 0.998100; these values do not establish
physical-edge equivalence or automatic acceptance.

The catalog worker subsequently validated all 84 exact current source-pixel
bindings through the existing native planner, with zero refusals, and reproduced
32 deduplicated references. No video decoding or new fits were performed.
Metadata mode was explicitly `unconsolidated_diagnostic`; historical validity
used the exact recording timestamp, not today's time. Isolated evidence:

- `/tmp/palette-geometry-reference-live-catalog-20260906-jglc9pe7/report.json`
- `/tmp/palette-geometry-reference-live-catalog-20260906-jglc9pe7/catalog.json`
- `/tmp/palette-geometry-reference-live-catalog-20260906-jglc9pe7/sources.json`
- Catalog SHA-256:
  `8ad0a4824f5f153ceae479959b01c552c67233a4a80206d153a10cf01dec22a1`.

The individual worker handoffs record their actual local tests, dirty scopes,
commit IDs when created, and limits. At this checkpoint all new exact-commit
required CI remains unrun. Locally passing tests and all-84 reference validity
do not establish calibrated automatic acceptance. All new branches remain
incomplete and not merge-ready; nothing is integrated, deployed, or activated.

## Published development revisions

The following separately owned commits were subsequently created and pushed;
all worktrees were clean. Their draft PRs are for required CI, not authorization
to integrate, merge, deploy, or activate:

| Ingredient | Exact commit | Local focused evidence |
| --- | --- | --- |
| Reference catalog | `52d3ea01ed29d4dde9632ed5420356978338aed2` | 76 tests; live 84/84 bindings; draft PR 150 |
| Shadow policy | `9d9440f9e48248534d50922d2830ab22442f070f` | 83 tests; 10 isolated cross-worktree diagnostics; draft PR 151 |
| Rim recipe/metrics | `ac73a61067118fc1a41699267dfa376a15f4bac6` | 73 tests; draft PR publication follows push |

These test counts include overlapping legacy regressions and must not be summed
as distinct tests. The shadow adapter still lacks its metrics prerequisite in
its own branch, deliberately failing closed. Neither independent local tests
nor cross-worktree diagnostics are green integrated-candidate evidence.
