# Palette agent-instruction audit

> Landing update added 2026-09-06: copied from the original checkout into the
> uncommitted main-based documentation draft; the original report is unchanged.
> This completed audit supersedes the [preliminary audit](agent_instruction_audit_preliminary_2026-09-05.md).
> Its final “not yet copied” statement records the original audit-time state.
> No instruction/helper fix is implied by this copy; see the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation).

Date: 2026-09-05. Scope confirmed by the user: `~/gitrepos/palette` only.
Completed read-only instruction audit; recommendations are not implemented.
This report supersedes the scope-pending preliminary audit in the isolated
review-docs draft. It is evidence, not a new implementation or status queue.

## Verdict

Keep the domain contracts. Correct the worktree-targeting guidance and broken
reference, preserve the newer main instructions, and narrow the applicability
of the long consolidation playbook. Palette does not have a repository-local
skill collection to trim. Its main opportunity is making AGENTS.md a more
precise guide, not replacing enforceable constraints with trust in a model.

## Inventory and verified premises

| Surface | Verified state |
|---|---|
| Original checkout | `/home/delahantyj@hhmi.org/gitrepos/palette`, branch `agent/palette/refined-assignment-rebinding-gaze-20260831`, HEAD `73bee0d5194c662e3b7e535be3de92db7ef53f63` |
| Original AGENTS.md | Read completely: 318 lines, 21,678 bytes; includes a pre-existing uncommitted 107-line collaboration section |
| Main-based Palette draft | `/tmp/palette-review-docs-rules-20260905`, HEAD `6af66c5a6ba3b35ea0bf00cfc74add7bb22da2b2`; AGENTS.md has 324 lines / 22,050 bytes |
| Difference between those instruction files | Exactly six additional main lines prohibiting campus-login-node pytest and LSF test submissions; byte comparison passed after removing that exact block |
| Source-owned instruction inventory | One AGENTS.md; no nested AGENTS.md, AGENTS.override.md, CLAUDE.md, or SKILL.md found in the scoped tree |
| Local skill/config directories | Root `.agents` and `.codex` directories are empty; no repository-local skills to assess for competing triggers |
| AGENTS.md literal repository references | Eight exist; one missing: `docs/sandbox_zarr_fallback.md` |
| Push/deployment helper versions | Both helpers are byte-identical between the original checkout and the selected main-based draft |

Original AGENTS.md SHA-256:
`c3d7e2340a44c456d718e3cc4ef20dee3701c9fc9b72c089a631065922414106`.
Draft AGENTS.md SHA-256:
`9ce71897fafc21a7969f89aea1ca8dd85606528c244ec7f11fe187405e80d067`.
These hashes identify the actual dirty files, not just their base commits.

Inventory included hidden and ignored source files. It excluded `.git`, nested
`.claude/worktrees`, `.pixi`, node_modules, virtualenvs, and bytecode caches.
The earlier unrestricted inventory found two installed marimo AGENTS.md copies
under `.pixi/envs`; those vendor copies are not Palette instructions. Personal
skills, plugin caches, other repositories, and deployment copies are outside
the approved audit. `.claude/settings.local.json` is configuration, not an
AGENTS.md or skill body, and was not read or changed.

The draft comparison concerns another checkout of this same Palette repository,
not a wider worktree inventory. Remote main was not refreshed in this pass.

## Ranked findings

### 1. Push guidance is not reliably worktree-specific

[AGENTS.md](../../AGENTS.md), line 44, gives a push-only command with the original
checkout hardcoded in `git -C`. From a linked worktree, that command still acts
on the original checkout; it does not select the caller's worktree or branch.
The selected remote destination also depends on that checkout's Git settings.

The recommended [shared-checkout helper](../../scripts/push_and_update_groups_checkout.sh)
has the same original-checkout default at line 59. Its line 122 requires `.git`
to be a directory, so supplying `--repo` with a normal linked worktree is not a
sufficient correction: linked worktrees use a `.git` file. The inspected
main-based draft has such a file. This is established by code and filesystem
inspection; no push/deploy command was executed.

Recommendation: make the original-checkout-only scope explicit immediately in
a future documentation change. For a general push-only workflow, resolve and
verify the intended source, branch, remote ref, and commit before pushing.
The [pinned-deployment helper](../../scripts/deploy_palette_cluster_worktree.sh)
already demonstrates worktree-aware source resolution (lines 5–8 and 174–176)
and branch/cleanliness checks (lines 194–205). Reuse those mechanics if improving
the push helper; do not deploy merely to obtain a worktree-aware push.

The shared helper performs an ordinary push/fetch/fast-forward; it does not
query required CI. Its cleanliness option also defaults off. Invoking it is
therefore not evidence that Palette's independent CI/integration requirements
were met. Do not delete those requirements because a helper exists.

### 2. An active fallback instruction points to an archived path

[AGENTS.md](../../AGENTS.md), line 236, refers to
`docs/sandbox_zarr_fallback.md`, which does not exist in either inspected
checkout. The document is present at
[docs/archive/sandbox_zarr_fallback.md](../archive/sandbox_zarr_fallback.md).
Commit `6c653b7245672ccd44adeccaea040f9d24efc77d` moved it on 2026-07-08.

Recommendation: reconcile active fallback guidance with its actual owner.
Simply fixing the path is not full review: the archived examples use `/nvme1`
and test review-state attributes, not content digests or complete authority
acceptance. Preserve their role as metadata diagnostics, never as substitutes
for current runtime admission or published-generation validation. Prefer a
maintained, explicitly scoped operations reference over reviving the archived
file unchanged as a current contract.

### 3. The old checkout is missing newer main safety instructions

The main-based draft's AGENTS.md adds the workstation-only test policy under
Sandbox Zarr Test Policy. Its `scripts/py` rejects recognized pytest invocations
on login1/login2 before Python starts, and `tests/conftest.py` repeats that host
check before collection. The old checkout lacks those additions.

This is version drift, not a newly discovered defect in main. Any instruction
cleanup must start from the final integration base and preserve these six
lines' meaning. Do not replace main's AGENTS.md wholesale with the old file.
The no-LSF rule remains an instruction; the inspected host-name checks alone
do not establish enforcement against every cluster execution route.

### 4. The new collaboration section needs clearer task boundaries

Lines 95–152 contain valuable preservation, migration, benchmarking, and
conformance requirements. The heading suggests simplification work, but
several imperatives can be read as applying to every task: preservation tests
first, negative cases, comparable-input benchmarks, and public-path exercises.
A documentation correction should not require new numerical fixtures or a
performance benchmark.

Recommendation: explicitly tie this playbook to behavior, schema, identity,
storage, and shared-interface implementation. Keep it mandatory where those
claims are affected. A small text-only change should receive proportionate
local checks, without any waiver of required CI before integration. This is a
scope clarification, not evidence that unnecessary tests were actually run.

Likewise, detailed ownership/prerequisite handoffs belong to implementation
and integration work; an explanatory answer need not manufacture a branch
ownership exercise. The existing phrase "before overlapping implementation
work" is already well scoped and should be retained.

### 5. Environment instructions repeat without one clear decision path

Sandbox fallback (lines 234–237), Sandbox Zarr Test Policy (259–272), and
Outside-Sandbox Validation Notes (302–312) overlap. The real exceptions are
important; repeating the general escalation/fallback story is not.

Recommendation: keep one short root rule directing tests to an allowed
workstation outside the sandbox, plus a conditional reference for marimo,
CUDA, real-Zarr diagnostics, and metadata-file fallbacks. Keep the immutable
publication policy separate: outside-sandbox execution does not make stale
metadata valid. Preserve bounded retries and an explicit deferred-validation
handoff when the required execution path is unavailable.

## Section-by-section disposition

| Existing section | Disposition for a future edit |
|---|---|
| Python Environment Rule | Keep interpreter binding and install approval; shorten duplicated invocation examples only |
| Registry SQLite Runtime Rule | Keep exact runtime and acceptance commands; the backup wrapper really delegates through its adjacent scripts/py |
| Git Push Rule | Correct source-selection scope; preserve key, exact-commit deployment, and shared-checkout restrictions |
| Required CI and Integration Rule | Keep success-only admission, explicit incomplete handoffs, and non-authoritative experimental deployment limits |
| Worktree/interface ownership | Keep; restrict coordination to relevant overlapping implementation as already stated |
| Contract preservation and staged consolidation | Keep invariants in root; conditionally route the detailed migration/testing playbook |
| Product completion | Keep computation/presentation/deliverable distinctions and the prohibition on bypassing existing persisted contracts |
| Handoff requirements | Keep exact evidence requirements for implementation/integration; make their task scope explicit |
| Authority Roles and Supplier Sufficiency | Keep domain distinctions and the conditional acceptance-checklist reference |
| Consolidated Metadata Read Policy | Keep the lifecycle distinction and publication-generation requirements |
| Sandbox and outside-sandbox guidance | Consolidate repetition; correct the fallback link; retain newer main host/LSF restrictions |
| Dask / Parallel Zarr Write Rule | Keep whole-physical-chunk ownership and requested/effective chunking provenance |
| Subject Mask Direction | Keep dense edit/training authority and compatibility boundaries; shorten only redundant descriptions |
| Examples | All referenced paths exist, including src/test_fisheye.py; reduce repetition rather than alleging stale paths without evidence |

## What the article does and does not establish here

The previously read [Provencher article](https://x.com/pvncher/status/2095991462416490862)
supports reviewing trigger breadth, compulsory reading, and ambiguous stopping
boundaries. It does not establish that Palette's scientific rules are obsolete,
that its tests have no production access, or that a newer model can safely
infer authority and digest contracts. No blanket test permission or safety
relaxation follows from the article.

Official [AGENTS.md documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
describes a root-to-working-directory instruction chain with a default combined
32 KiB limit. Both inspected files fit below that limit; this audit found no
evidence of truncation. Moving critical rules to nested files is not by itself
a guarantee that a root-started session will load them: retain an explicit
task-triggered reference in the root.

Official [skill documentation](https://learn.chatgpt.com/docs/build-skills)
describes short discovery metadata followed by full instructions on use.
There are no local skill descriptions here to shorten. Do not create skills
solely to relocate mandatory repository contracts into an optional mechanism.
Keep instructions usable by all repository contributors' agents, not only one
model generation.

## Safe next change, if authorized

First reconcile the final main instructions and correct active reference/source
scope defects. Separately shorten the root playbook with a before/after mapping
that accounts for every preserved invariant and its trigger. Keep helper
behavior changes in their own tested implementation change. Reuse the owning
authority/consolidation queue for overlapping work; this report creates none.

Acceptance should include representative task walkthroughs: text-only edit,
digest-preserving extraction, storage/chunking optimization, authority change,
linked-worktree push-only handoff, and unavailable real-Zarr validation. Verify
each task loads its required constraints without unrelated mandatory work.
These walkthroughs complement, not replace, the exact candidate's required CI.

## Validation and limitations

Performed static inventory, full instruction reads, literal-reference checks,
helper inspection, Git-history lookup, and cross-checkout byte comparisons.
No source, scripts, AGENTS.md, skill, configuration, registry, or store was
changed in this pass. No tests, deploy helpers, pushes, or remote administration
were run. Runtime contract implementation was not comprehensively re-audited.

The sole intended write is this new report in the original checkout. It is
uncommitted and not yet copied into the earlier isolated landing draft. That
draft and its checkpoint remain unchanged. Required CI is unrun for this new
document; completion of the read-only audit is not a merge-readiness claim.
