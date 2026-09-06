# Joint plan for the architecture reviews

> Integration update, 2026-09-06: this plan and the completed instruction audit
> are in [PR #145](https://github.com/jmdelahanty/palette/pull/145), refreshed
> onto validated main `04779a6b20512dc01f2a28b6cfb34d6b705a6744` after
> #140, #143, #146, #147, and #142 landed and passed their post-merge CI.
> Main now requires all 23 original checks plus the success-only `ci-required`
> aggregate. The unprotected-main and two-verifier observations below remain
> historical; the September 6 cross-check found six stage-specific verifiers
> and execution-time reuse validation. See the [integration handoff](review_docs_rules_landing_handoff_2026-09-05.md#authorized-main-integration--september-6)
> and live PR for this candidate's exact head and CI. Ownership inventories
> still need fresh reconciliation before new implementation or cleanup.

Date: 2026-09-05, America/New_York. Read-only planning and blocker assessment;
implementation, integration, remote administration, and production changes
are not authorized by this document. This is a review agenda and proposed
sequencing, not another work queue. The existing
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md)
continues to own overlapping implementation status and ordering; separate
scientific, performance, and subtraction queues retain their existing scope.

## Can we start?

**Yes: start joint review, premise reconciliation, and design decisions now.**
Do not start broad overlapping implementation from this dirty old checkout.
The immediate constraint is unfinished work in the same interfaces, not lack
of an architectural direction. Resolve owners and exact integration evidence,
then start small implementation packages from an agreed green main commit.

Deleting every old worktree, completing a universal receipt framework, or
finishing every unrelated PR is not a prerequisite. Unrelated unknowns must
not block a valid workload's admitted dependency closure.

## Fresh blocker check

Observed during this pass, around 2026-09-06 02:54 UTC / September 5 22:54 EDT.
These are snapshots; worktrees and remote state can change afterward.

| Item | Evidence now | What it blocks |
|---|---|---|
| Main baseline | GitHub main is `6af66c5a6ba3b35ea0bf00cfc74add7bb22da2b2`; [CI run 34004329735](https://github.com/jmdelahanty/palette/actions/runs/34004329735) completed with all 23 jobs successful | No baseline-CI blocker for read-only planning or a fresh source checkout at this commit |
| Current checkout | HEAD `73bee0d5194c662e3b7e535be3de92db7ef53f63`, dirty AGENTS.md and untracked reviews; local main is 0 ahead / 110 behind the verified remote commit | Do not treat this checkout or local main as the fresh implementation base; no reset or checkout switch is needed to plan |
| Overlapping canary work | `/tmp/palette-core-chaser-canary-main-20260906`: 27 changed paths, including runtime verification, execution, admission, eye/tail/track readers, exports, and publication | New independent edits to those interfaces need owner coordination and an exact handoff first; current dirty contents have no commit-bound CI evidence |
| Other unfinished overlap | Subject-shape optimization, clipped downstream analytics, deterministic mask/finalizer work, and targeted keypoint reconstruction contain dirty changes | Coordinate the relevant owner before touching those families; do not infer abandonment from age or lack of an open PR |
| Open PR 57 | `subject-shape-v5-supported`, head `d8cb0c21cc1352ad075135ac6e74da00399dfa8e`; merge state DIRTY, old successful checks | Not ready to integrate as-is; decide whether to finish, supersede, or preserve separately before overlapping shape work |
| Open PR 27 | Draft coordinate-successor PR, head `93bd252cf57d19d08801f7a7e724bf68be90420a`; DIRTY, generated-artifacts failure and skipped test matrix | Not ready to integrate; not an automatic blocker for unrelated current-main work |
| Docs/rules landing | `/tmp/palette-review-docs-rules-20260905` remains an uncommitted 28-path draft on the green base; the completed instruction audit and this plan are not yet included | No candidate CI or integrated common handoff yet; reconcile these documents before the planned documentation landing |
| GitHub enforcement | Main reports `protected=false`; repository ruleset list is empty | Server-side CI enforcement is not installed. Recommend enabling it before broad wave merges; it is not a technical prerequisite for local review/tests, and existing success-only integration policy remains mandatory |
| Preservation and cleanup | 64 registered workstation worktrees, eight dirty at this scan; earlier verified checkpoints exist | No blanket cleanup clearance. Refresh preservation for later edits and obtain owner/job-reference clearance before any removal |

The six review agents in this conversation are completed. That does not prove
other sessions or cluster jobs are idle. The owner/activity of the canary
worktree has been requested from the user; file changes establish overlap,
not process liveness. No cluster-job or `/groups` deployment inventory was
refreshed in this pass.

The earlier checkpoint is at
`/home/delahantyj@hhmi.org/gitrepos/palette-preservation-20260905-DhRVvF/checkpoint-v1`;
the earlier draft snapshot is `draft-v1` beside it. They preserve their capture
times, not every subsequent edit. Do not reuse them as current deletion
clearance. The only write in this pass is this plan.

## Reconcile the evidence before assigning fixes

Read the [second opinion](review_wave_second_opinion_2026-09-04.md),
[independent design assessment](independent_codebase_design_assessment_2026-09-04.md),
[shared-helper review](shared_helper_consolidation_review_2026-09-04.md), and
[instruction audit](agent_instruction_audit_2026-09-05.md) as the synthesis.
Use the original five-lens, propagation, provenance, training, receipt, storage,
pipeline/correction-loop, strategy, and export reviews as supporting evidence.
The second opinion's section 8 already maps many duplicated checklist items
into the existing queue; extend that disposition instead of restarting it.

For every proposed package, record one disposition in the owning queue/handoff:
still supported; partly addressed; already addressed with evidence; superseded;
or needs fresh measurement. Include the reviewed base and relevant unfinished
diff. A previously passing characterization probe of a defect is not a test
that establishes the desired corrected behavior.

One correction is already freshly verified: main's
`analysis_workflows/runtime_verification.py:308` registers **eye angles and
subject shape**, not only subject shape as in the September 4 review. Its
generic no-verifier branch still returns discovery availability. This is a
remaining shared-boundary concern, but neither the old count nor the extent of
the gap can be copied unchanged. The canary further changes this file by
265 added / 26 removed lines at inspection time.

The canary also introduces `execution_profiles.py`. It declares production vs
selector-ineligible lifecycle profiles, **not scientific rig recipes**. Do not
mistake that useful work for completion of the scientific-parameter proposal,
or overwrite it by inventing an unrelated second profile system.

The completion gate, strict metadata comparator, registry DB/query/model
resolver, viewer query, and LSF submission files were byte-unchanged between
the old review commit and verified main. That preserves relevant code
evidence, not proof that every historical probe remains valid under all new
callers and dependencies. Reproduce the affected case on the chosen base.

Do not execute the old wave-1 brief unchanged: its default-off gate patch is
not full admission/publication closure; its `sun` starting ref is stale; and
its packages share a queue-editing surface. Retain the second opinion's
corrections against blanket selector backfill, universal authority/envelopes,
unversioned digest replacement, and unsupported contamination claims.

## Joint component-review agenda

Use one representative motion/bout workflow to connect the components, then
test a second supported profile to expose accidental specialization. Prefer
the existing core-motion/chaser canary after its owner supplies accepted
evidence; do not assume its inputs or outputs are already admitted. Synthetic
fixtures can establish boundary behavior without touching production data.

| Review session | Components and questions to settle together | Concrete output / existing home |
|---|---|---|
| 1. Working agreement and integration | Review disposition, instruction scope, worktree ownership, CI rules, push/deployment targeting, exact-code execution identity | Agreed source base and ownership map; documentation/CI rollout under GOV-001 and Stage 11, not another queue |
| 2. Scientific identity and provenance | Recording/frame/coordinate identity; sufficient suppliers; content vs layout vs execution vs review digests; immutable training membership/splits; recipe defaults and applicability | Invariant list and version/compatibility decisions using RID-001, ADM-001, ACC-001/002 and existing receipt/proof helpers; no universal schema rewrite |
| 3. Registry and model/query services | Read-only vs write/upgrade capabilities; same-subject query correlation; transaction boundaries; task aliases, admission-before-ranking, per-plan provenance; relocation-safe locators | Query-grain and transaction tests; one model-selection policy owner; registry/reporting items adopted into Stage 10, with training-specific work kept in its existing scope |
| 4. Admission, reuse, and DAG execution | Discovery vs concrete evidence; pending outputs; producer/profile declarations; exact recipe/input/code binding; numerical vs presentation dependencies; maintained entry points | One traceable admitted closure through existing catalogs, provider offers, and resolvers: ADM-001–003, RES-* and REP-001 |
| 5. Publication and recovery | Candidate validation before selection; prior accepted result vs failed attempt; owner-fenced rollback; consolidated generation; serial registry projection; scheduler submit/ack crash windows and retry | Failure-state table and adversarial tests through existing publisher/LSF boundaries: TEST-001, REP-001, Stage 11 |
| 6. Storage, inference, and readers | Physical chunk/shard ownership; asynchronous writer shutdown; staging inventory; bounded copy/window/index reads; receipt reuse; viewer-generation cache; full staged export validation | Shared mechanics with byte/identity fixtures and access budgets; existing Zarr, training, Arrow, and specialist reader owners |
| 7. Public API, training feedback, and observability | Thin CLI/application boundary; explicit read/result errors; correction-to-export-to-model lineage; evaluation eligibility including unknown membership; figure identity; failure/error-budget reporting | End-to-end user-visible behavior, lineage evidence, and measurable acceptance; REP-001, ACC-003, producer work, and existing reporting/model queues |
| 8. Subtraction and expansion | Which copies and compatibility routes remain; caller adoption; import/AST enforcement; next scientific profile | Delete only after caller and conformance evidence; SUB-001 and existing subtraction gates |

Each session should finish with four things: the current behavior, the invariant
that must survive, the smallest proposed change, and the evidence that would
accept it. Settle scientific or persisted-contract decisions before the
corresponding implementation, not after a cleanup has silently made them.

## Proposed implementation staging after those decisions

### Stage A — Establish the shared starting point

Resolve disposition of the overlapping work: land only after exact candidate
and combined-main CI is green, or explicitly preserve/defer it with an owner
agreement. Do not blindly merge unfinished branches to obtain a single tree.
Bring the docs/rules draft up to the final base, add the completed instruction
audit and this plan, correct historical statuses/links, and reconcile queue
rows with actual landed evidence. Commit/push/PR work needs authorization.

Apply the already written two-phase CI plan after administrative approval:
first require the existing checks/current base without bypass; separately add
and validate the success-only aggregate gate, then require it too. The plan is
currently in the draft at
`docs/diagnostics/server_side_ci_enforcement_plan_2026-09-05.md`.
Do not weaken required CI while waiting for that administrative work.

Retiring old worktrees can follow individually after preservation and owner/job
clearance. It need not hold up work in a fresh, nonoverlapping source worktree.

### Stage B — Close narrow, reproduced boundary defects

Capture desired rejection/preservation tests first. Candidate small packages
include strict metadata equality; same-subject range queries; model-task
normalization and single/batch admission parity; correctly rebound cached
provenance; read-only registry access and local transaction composition; and
forwarded-argument ownership protection. Recheck these against unfinished work.

These are corrections, not automatically behavior-preserving extractions.
Identify intentionally changed refusal/selection behavior and preserve valid
artifact bytes and scientific defaults. Avoid broad serializers, selector
backfills, or historical data mutation. Each package should have one owner and
its own complete required CI evidence.

### Stage C — Prove one complete numerical-to-reader path

Connect existing sufficient supplier handles, concrete admission, explicit
scientific recipe, code-bound execution/reuse, candidate validation, immutable
publication, registry projection, and bounded reading. Extend current owners
rather than introducing another general publisher or workflow framework.

Preserve computation complete, presentation complete, and deliverable complete
as distinct claims. Numerical results remain reusable after an independent
plot failure when their existing contract permits it; a plot-required
deliverable remains incomplete. Persisted status changes need an explicit
compatible/versioned plan, not a renamed boolean.

Exercise wrong recipe/source, stale evidence, registry-disabled execution,
failure before/after selection, projection failure after publication, ownership
takeover, and a generation replacement while reading. Preserve the previous
accepted result and report failed attempts separately. Do not add a second
upstream or human gate when a validated supplier is sufficient.

### Stage D — Consolidate recovery and bounded execution mechanics

With the boundaries fixed, migrate the stronger asynchronous writer lifecycle,
whole-physical-unit writes, selected-file staging, atomic receipt writing,
candidate failure/tombstone handling, and owner-guarded rollback. Keep rollback
and post-execution rejection as distinct protocols. Share full family-specific
decoded validation before and after export selection; preserve Parquet policy.

Specify stable submission/attempt identity and reconciliation of scheduler
acceptance before automatic retry. A changed status file or existing output
path does not establish a successful attempt's artifact identity. Verify the
actual stage process's repository/commit, not just the wrapper's import path.
Keep scientific partitioning, scheduler tasks, and storage ownership separate.

### Stage E — Measure, migrate remaining callers, then subtract

Benchmark fixed scientific inputs and outputs with cold/warm metadata access,
bytes read per useful row/window, peak memory, bounded live buffers, physical
write/copy amplification, GPU idle intervals, queue wait, validation cost, and
first-window latency. Use existing telemetry and report local vs shared-store
conditions; choose targets from the measured baseline, not arbitrary promises.

Migrate one caller at a time, then all maintained callers of that helper family.
Extend existing catalogs, conformance tests, shim import bans, and metadata-mode
ratchets after adoption. Keep clone counts and file sizes as maintenance signals,
not correctness gates. Remove copies only with compatibility disposition and
successful full CI; do not merge scientifically different validators just
because their loops look alike.

## Work that can proceed independently

After file/interface ownership is agreed, registry/query/model work, storage
worker mechanics, CI/instruction work, and workflow/admission work may proceed
in separate worktrees. The queue and shared schema/interface each need one
owner; multiple agents must not independently edit them. Do not create parallel
implementations to avoid coordinating an interface.

Training membership/split/parent-model and correction-lineage work can progress
alongside the numerical path. It needs immutable dataset/model evidence, not
completion of every pipeline stage. Treat training-recording inference,
held-out evaluation, and exact-example overlap as different uses. Unknown
membership is not a clean result, and training-recording inference is not
universally forbidden.

The historical live findings are a separate verification/remediation lane:
ineligible registry projections, missing active frame-index locators,
path-derived merged IDs, consolidated-selector disagreement, retained/deleted
experiment history, and the four-camera recording-level membership result.
Re-measure only the relevant scope with snapshot/generation identity before
acting. None authorizes blanket pointer backfill, deleting history, rewriting
old receipt paths, or trusting a diagnostic script's zero exit as admission.
Missing evidence blocks the affected consumer, not all library development.

## Decisions and permissions needed later, not blockers to this review

- Identify owners and intended disposition of the canary and other overlapping
  worktrees. Whether to finish or supersede PRs 57/27 is a user/owner decision.
- Approve a docs/rules landing and GitHub administration separately; this plan
  performs neither. Choose any independent human-review policy deliberately.
- For the representative real workload, confirm scope, scientific recipe and
  rig applicability, permitted overrides, expected products, and acceptance use.
  Preserve current calibrated values unless an explicit scientific change is
  requested; do not invent rig identifiers or held-out grouping requirements.
- Before retry/durability changes, state supported filesystem failure/recovery
  assumptions and registry writer ownership. Do not assume a database-backend
  migration, network WAL, or a new service is required.
- Before real-data canaries/promotion, identify approved inputs, fresh output
  destinations, resource limits, cleanup disposition, and exact pinned code.
  Required CI and applicable scientific/authority acceptance remain separate.

Crimson integration, a new scheduler/framework, microservices, a universal
authority envelope, and repository-wide digest migration remain deferred until
a demonstrated requirement makes them necessary.

## Validation and handoff

This pass used read-only GitHub status APIs, Git/worktree status and diffs,
document/source inspection, and byte-equivalence checks. No runtime defect was
fixed or declared freshly reproduced; no live registry/store measurement or
cluster submission was performed. No source, instructions, queue status,
configuration, Git refs, or server settings were changed. The plan is
uncommitted and has no candidate CI; planning completion is not merge readiness.
