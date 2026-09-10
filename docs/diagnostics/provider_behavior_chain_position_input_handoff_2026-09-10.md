# Provider behavior-chain position, timing, and receipt handoff — September 10, 2026

## Ownership and exact state

- Owner: root implementation agent; no parallel implementation agents.
- Worktree: `/tmp/palette-provider-aware-core-workflow-20260910`.
- Branch: `agent/palette/provider-aware-core-workflow-20260910`.
- Exact prerequisite and current `HEAD`:
  `c68a675e003c689b03ffdb6056e25e25a0bb81f2` (PR 161 candidate).
- Candidate state: uncommitted working tree. There is no exact candidate commit
  yet and required CI has therefore not run on this combined change.
- Classification: behavior-preserving caller migration, a versioned temporal
  provenance extension, and deliberately tighter reuse/receipt enforcement.

This is a selector-ineligible canary utility slice. It does not change the
default `core_behavior_v1` DAG, generic planner/executor admission, production
authority, any selector, the registry, or historical data. Test writes were
limited to temporary fixtures.

## Implemented interfaces

### Profile-neutral position input

Provider behavior-chain task v3 consumes one exact
`analysis/subject_position_runs/observation/<run>` path and its manifest
SHA-256. Unlike v1/v2, it does not create a keypoint-triad position run. The
strict subject-position handle validates the input in both direct and published
consolidated metadata views and compares the manifest and decoded-content
digests. A detection-, keypoint-, or mask-backed estimator can therefore use
the same tracking and dense provider-motion consumer without the chain
discovering a modality or fallback.

Task v1/v2 retain their keypoint-triad compatibility behavior. Task v3 retains
its original undigested result-v1 grammar for compatibility.

### Recording timing authority

Provider-motion computation schema v2 binds the existing strict
`provider_recording_timing_authority` record and SHA-256. Preparation, planning,
local materialization, publication, activation, and final validation repeatedly
reopen that exact authority, require exact nominal-FPS agreement, and validate
every motion source index against the complete acquisition-frame domain. The
strict provider-motion reader reopens the live authority and reports
`bound_live_recording_timing_authority`; stale clock state, an invalid frame
domain, or a mismatched digest fails closed.

Computation schema v1 remains readable as explicit caller-FPS-only
compatibility evidence, but it cannot satisfy
`require_authoritative_timing=True`. The outer provider-motion run and manifest
schemas remain v1 because their existing payload envelope already carries the
versioned computation record.

### Exact reuse validation

Existing outputs are no longer reused by name alone:

- tracking must bind the requested position path, manifest, and decoded
  content;
- motion must bind the recomposed position/body-frame authority, tracking
  input, and live recording timing authority;
- swim bouts must match the exact provider-motion manifest, verification
  digest, track slice, physical frame axis, lineage, and direct/consolidated
  content;
- legacy stimulus epochs must match the exact stimulus source, completion
  lifecycle, logical content, lineage payload, and direct/consolidated view;
- v2 stimulus epochs must bind that exact legacy epoch content and lineage; and
- the epoch summary must bind the exact motion, bout, epoch, and protocol
  semantic-selection evidence and pass its existing deep strict loader.

The provider-motion-to-bout adapter now explicitly declares the existing
`selector_ineligible_canary_v1` publication profile. The real persisted test
found this missing field; no new bout profile or selector was introduced.

### Digested task and durable receipt

Task schema v4 is the receipted successor to v3. It requires a closed canonical
field set, canonical absolute archive path, explicit arena ID, floating-point
FPS, and:

```text
task_sha256 = sha256_canonical_json_v1(task without task_sha256)
```

`build_receipted_task(...)` creates that envelope. `load_task(...)` rejects a
stale digest, tampering, extra fields, and noncanonical persisted values before
any stage runs. `materialize_chain(...)` revalidates the task at its API
boundary.

A successful v4 execution returns
`palette.provider_behavior_chain_receipt` v1. The receipt binds the task digest,
the exact evidence for all seven stages, a canonical SHA-256 for each stage,
the safety claims that selectors and source payloads were not changed, and:

```text
receipt_sha256 = sha256_canonical_json_v1(receipt without receipt_sha256)
```

Every stage must expose the artifact identities appropriate to its contract:
position decoded content and manifest; tracking and motion manifest plus strict
verification digest; bout content, lineage, and source-motion manifest; both
epoch lineage/selection identities; and the summary manifest. The CLI accepts
`--receipt-json` for v4 while preserving `--result-json` as a compatible alias,
writes atomically, and reloads the persisted receipt through
`validate_chain_receipt(...)` before returning success. The validator also
accepts an expected task SHA-256 and refuses task, stage, or envelope tampering.

## Preserved and intentional behavior

Preserved:

- position/body-frame validity, coordinates, and row/frame identities;
- tracking and provider-motion dense numerical arrays and formulas;
- motion smoothing, hysteresis, physical-scale, and validity defaults;
- stimulus-epoch, swim-bout, and summary scientific schemas;
- v1/v2 task behavior, v3 task/result behavior, and computation-v1 reads; and
- selector-ineligible lifecycle with no selector or registry updates.

Intentional enforcement changes:

- the behavior chain now requires a live recording timing authority for new
  and reused provider motion;
- stale or wrong-source existing outputs are refused rather than silently
  reused; and
- v4 tasks and receipts use new, closed, tamper-evident persisted grammars.

## Validation evidence

Tests were run outside the Codex sandbox with `scripts/py`, per repository
policy.

- Combined chain, provider-motion publisher/reader, recording-timing, and bout
  boundary: **80 passed** in 125.44 seconds.
- Adjacent provider-summary, swim-bout, and stimulus-epoch suites:
  **152 passed** in 63.46 seconds.
- Adjacent subject-position, body-frame, tracking, and composition suites:
  **189 passed** in 18.60 seconds.
- Total across those non-overlapping selections: **421 passed**.
- The real unpatched persisted v4 chain publishes detection-backed position ->
  tracking -> timing-bound provider motion -> swim bouts -> both epoch forms ->
  semantic summary, reruns with all seven outputs reused, validates both task
  and execution receipt digests, and refuses a directly tampered `position_xy`
  input. It is included in the 80-test boundary suite.
- Focused negative tests cover wrong tracking position lineage, wrong composed
  motion/timing authority, wrong bout motion manifest, wrong legacy stimulus,
  wrong v2 source epoch, wrong summary inputs, direct/consolidated position
  disagreement, task tampering, stage-receipt tampering, and receipt tampering.

Static acceptance completed locally:

- `scripts/py -m py_compile` on changed Python modules;
- Black on the six newly developed/formatted provider files (the large legacy
  bout module retains its pre-existing formatting style; the scoped four-line
  change is covered by compile/tests and `git diff --check`);
- generated Zarr inventories regenerated and `--check` green;
- file-size, Zarr metadata-mode, FPS-authority, keypoint-motion-authority,
  tail-receipt, paradigm-core-authority, observed-metadata, and contract
  freshness ratchets green;
- both import-linter contracts green;
- ratchet baseline files unchanged;
- the registry schema reference is current; and
- repository-wide pytest collection is green: **11,855 tests collected**, with
  one intentional collection skip, in 32.56 seconds.

Required GitHub CI remains unrun on this uncommitted slice. Under the required
CI contract, the branch is **not merge-ready** and must not be merged,
integrated, deployed, activated, or used to update the shared `/groups`
checkout until an authorized exact commit passes every required job and the
`ci-required` gate.

## Remaining broader `RES-TRACK-001` scope

This slice closes the bounded provider-chain canary preparation; it does not
close the repository-wide `RES-TRACK-001` item. Still separate are:

- a versioned decision and staged migration for the generic core DAG rather
  than silently changing `core_behavior_v1`;
- production selection/admission and scientific promotion policy for provider
  outputs;
- reconciliation and explicit retirement/retention decisions for the existing
  materialized, collection-successor, and sealed crop compatibility branches;
  and
- bounded production canary/performance evidence followed by all required CI
  on the exact integration commit.

No commit, push, PR, merge, deployment, activation, registry mutation,
production Zarr mutation, or dependency installation was performed for this
slice.
