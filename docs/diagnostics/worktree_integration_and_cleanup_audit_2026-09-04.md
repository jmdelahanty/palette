# Worktree, agent activity, and clean-main integration audit

> Status correction added 2026-09-06: the unprotected-main finding below is
> historical. The September 6 cross-check observed 23 strict required checks on
> main; the separate success-only gate PR remains unmerged and not activated.
> Worktree counts, process observations, and PR states below are not current
> cleanup clearance. See the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation) and owning queue before acting.

Date: 2026-09-04. Final main/PR observation: 22:39 EDT (2026-09-05 02:39 UTC). Worktree status snapshot: approximately 22:38 EDT. These are dated observations, not a live dashboard.

Scope: read-only inspection of Git state, relevant diffs and patch identity, GitHub PR/CI/rules, workstation process names/working directories, and LSF job status. Only this report was written. No source, registry, store, Git ref/index, PR, deployment, job, or `/groups` content was intentionally changed by this audit. No fetch, merge, commit, push, pruning, removal, installations, or tests were run. Other sessions changed worktrees concurrently.

This is evidence and a proposed integration/cleanup procedure, not another implementation fix queue or authorization to perform cleanup. Architectural changes remain governed by the existing authority-consolidation queue and applicable specialist queues.

## Recommendation

Establish a bounded integration cut, not a requirement to merge every branch that still exists. Finish the agreed active work, decide the disposition of the genuinely unlanded leftovers, preserve all dirty work, and then establish one clean, CI-green main baseline. Old experimental branches can be safely archived without becoming requirements for the next architecture wave.

There is no need to rewrite history or reset the current review checkout. A separate clean integration/main checkout is safer while many sessions still start from the current repository directory. Workstation worktree cleanup and shared deployment retirement must be separate operations.

## Verified premises

| Premise | Verified result | Consequence |
|---|---|---|
| Current worktree | `/home/delahantyj@hhmi.org/gitrepos/palette`, branch `agent/palette/refined-assignment-rebinding-gaze-20260831`, HEAD `73bee0d5194c662e3b7e535be3de92db7ef53f63` | This is not main and is not clean. |
| Live GitHub main and local origin/main | Both `68f7b7a7cff64f2483df2b39c135dcb369ca82ea` | Comparisons used a remotely verified SHA, not an assumed-fresh tracking ref. |
| Local main | `6a9fa41793fc5a946ed13a4b910ff87ae4016f82`; 0 commits ahead and 82 behind the verified main | A later ordinary fast-forward is possible; no reset/rebase is needed. No registered workstation worktree currently checks out main. |
| Main CI | All 23 jobs passed for the verified main in [run 33938662510](https://github.com/jmdelahanty/palette/actions/runs/33938662510) | Main was still running early in the audit; it was green by the final check. This does not validate future combinations. |
| Workstation worktrees | 62 registered, all present, none marked locked or prunable | The earlier inventory was 61 before a new chaser checkout appeared. No stale-registration cleanup is presently justified. |
| Local state partition | 7 dirty; 40 clean with HEAD ancestral to main; 4 additional clean with verified squash/patch landing; 11 other clean checkouts needing reconciliation | 44 landed-and-clean checkouts are a review pool, not an approved deletion list. Deployment use and ownership remain separate tests. |
| Local branch refs | 197; 61 heads ancestral to main and 136 not ancestral | Non-ancestry is not proof of missing work: squash merges are common. The older 136 refs were not all individually audited for semantic supersession. |
| GitHub PRs | Six open: #136, #134, #117, #110, #57, #27 | The real integration backlog is much smaller than the worktree count. It also does not capture all local unpublished work. |
| Server-side enforcement | Branch API reports `protected=false`; effective main rules endpoint returns `[]` | Required CI is repository policy but currently not enforced by GitHub branch rules. |
| Shared repository | 281 registered worktrees in the separate shared Git repository; 265 locked; no prunable registrations reported | These are not included in the 62 workstation registrations. Do not unlock/prune/remove them as part of workstation cleanup. Presence/dirty state of every shared deployment was not audited. |
| Shared primary checkout | `/groups/johnson/johnsonlab/jeremy/gitrepos/palette`, main at `c1882f9b3565016a777acc5ab33e1a57b858eb06` | Its older head is not automatically a defect: pinned deployments intentionally avoid moving this shared checkout. |
| LSF observation | `ssh -o BatchMode=yes -o ConnectTimeout=10 login1-citrus-poller 'bjobs -w'` returned “No unfinished job found” | Only a point-in-time observation for the queried account/cluster. It does not clear scheduled retries, other accounts, notebooks, services, or retained absolute-path references. |

The current CI workflow defines seven independent gate/package jobs plus 16 non-GPU test shards. Generated-artifact failure suppresses the shard matrix, so a skipped matrix is missing evidence, not a pass. The server-side rule observation came from read-only `gh api repos/jmdelahanty/palette/branches/main` and `gh api repos/jmdelahanty/palette/rules/branches/main`.

## Agents and work actually in progress

The six parallel reviewers from this conversation have finished: publication, registry, workflow, reader/export, inference/storage, and enforcement. Their results are in [the consolidation review](shared_helper_consolidation_review_2026-09-04.md). They are not six pending implementation branches.

Other sessions are not globally visible through this conversation's agent roster. A workstation process snapshot found 13 processes named `codex` and eight named `claude` with the root Palette directory as their working directory, plus a transient Codex command process in the explorer-cache checkout. Those counts include long-lived/idle sessions and are **not counts of agents actively writing code**. Process names, PIDs, parent relationships, and working directories were inspected; prompts, environments, and conversation histories were not.

There are two concrete current-work signals:

- `/tmp/palette-distribution-ibi-display-label-20260904` is now on **explorer-receipt-cache**, not the branch suggested by its directory name. PR #136 is running CI, and a Codex/sleep command chain was observed there. Treat it as in use.
- `/tmp/palette-chaser-core-authority-20260904` appeared during the audit, starts at the latest main, and subsequently acquired edits to core export contracts/adapters plus new `core_authority_roster.py` and its tests. This is new, uncommitted work, distinct from the already merged **core-motion-authority** PR #135.

The five older dirty non-root checkouts did not show a process using their directory at the sampled instant. That does not establish abandonment: agents routinely use `git -C` or short-lived subprocesses from the root. Owners must confirm whether each patch is continuing, superseded, or to be archived. Do not switch the root checkout while those sessions are still using it.

## What should land, and what should not be merged again

| PR / intent | Exact head | Observed checks | Disposition before baseline |
|---|---|---|---|
| [#136](https://github.com/jmdelahanty/palette/pull/136) visualization: cache receipt-bound explorer projections | `de75bfbf4e46eb769dea71e34de56eff9b41ebec` | 17 success, 6 in_progress; mergeable | Active performance work: wait for remaining CI; validate the current-base combination. |
| [#134](https://github.com/jmdelahanty/palette/pull/134) Record receipt-composed canary evidence | `a0296d6347f0481d63e3505c5b1b3a09dc8425e6` | 23 success; mergeable | Docs-only evidence candidate; confirm final claims and land through CI. |
| [#117](https://github.com/jmdelahanty/palette/pull/117) Document distribution review and temporal alignment boundary | `824527b75fff6eeed1e2c323a6b6627706330498` | 23 success; mergeable | Docs-only candidate; use this PR, not its duplicate integration-evidence branch. |
| [#110](https://github.com/jmdelahanty/palette/pull/110) perf: reuse keypoint proofs during eye planning | `aca708a0e0485abbcc52df2e90144c9ecf32d49d` | 23 success; mergeable | Performance candidate; reconcile with newer proof helpers and confirm or collect the canary evidence mentioned in its handoff. |
| [#57](https://github.com/jmdelahanty/palette/pull/57) Support bundle-backed subject-shape v5 publication | `d8cb0c21cc1352ad075135ac6e74da00399dfa8e` | 23 success; conflicting | Conflicting old PR; likely substantially superseded by #62. Reconcile, do not merge blindly; preserve dirty work separately. |
| [#27](https://github.com/jmdelahanty/palette/pull/27) Publish coordinate-complete inference successors | `93bd252cf57d19d08801f7a7e724bf68be90420a` | 1 failure, 6 success, 1 skipped; conflicting | Conflicting draft, failed generated-artifact gate and skipped tests. Explicit salvage/retire decision, not a baseline prerequisite. |

At the final PR observation, #136's uncompleted checks were non-GPU shards **1, 3, 9, 10, 12, and 15**. The precise list will age immediately; re-query its exact head before any integration. No failed checks were observed on that head. #27's failed check was **generated artifacts**, and its non-GPU matrix was skipped.

“23 successful checks” describes the observed PR head/check rollup, not a fresh test of every subsequent main combination. Mergeability also is not scientific or runtime acceptance. Any updated candidate must satisfy all applicable checks again. Do not combine failing/unrun required-check work into another merge candidate under the repository's Required CI and Integration Rule.

### Receipt/performance work already on main

The user's important contract work has substantially landed. Examples include:

- #123: required receipts for maintained subject-shape loading.
- #125: subject-shape receipt reuse during loading.
- #126/#127: tail schema sealing and canonical acquisition-rate binding.
- #128: full-rate core motion.
- #130: co-located validated-behavior product catalog.
- #132: composed validated-behavior publication receipts.
- #135: physical motion as a core export authority; its PR head `a83eaa24dd44b94cf0c8496db974086a6a4dce02` had 23 passing checks and merged as the verified main.

Consequently, the architecture reports at `73bee0d5` must be reconciled against the final baseline before opening implementation tasks. An old review finding must not replace or weaken these newer receipt, digest, authority, or bounded-access contracts.

### Squash merges: verified, not inferred from names

Stable aggregate patch IDs matched each original PR change to its landed single-parent squash commit:

| Worktree history | PR | Landed commit | Result |
|---|---|---|---|
| `62873ccb` direct hybrid | #58 | `4f3c6d26` | Same aggregate patch |
| `ec374edf` protocol-semantic | #59 | `dfa061b5` | Same aggregate patch |
| `b97536e6` canonical downstream | #62 | `9e71fea3` | Same aggregate patch |
| `e240fe16` chaser body frame | #66 | `65b06a2f` | Same aggregate patch |

The canonical-downstream checkout has one additional commit, `a17706b2`; `git cherry` marks it patch-equivalent to a change already on main. Thus those four clean checkouts do not need another code merge. Patch identity establishes historical landing, not that later main never evolved or reverted a behavior.

PR #62 explicitly states that it includes the supported bundle subject-shape fixes required by its path. This makes old #57 a strong supersession candidate, but I did not prove every #57 behavior redundant. Reconcile its requirements/tests before closing it. Its checkout and the separate subject-shape optimization checkout both have valuable uncommitted changes that are **not** disposed of by closing #57.

The two changed documents in `9882d2ed` (distribution integration evidence) are byte-identical to the corresponding files in #117 at `824527b7`. Land one history, not both; preserve the original until that disposition is recorded.

## Local work that an open-PR list would miss

### The current review branch is not fully landed

Its original PR #90 merged an earlier tip, `7d116bcb`. The current tip has **24 non-ancestral commits relative to current main**, all changing documentation and two diagnostic scripts; there are no source changes in that branch-only diff. The current branch is 16 commits ahead of its local tracking ref. That is local tracking evidence, not an independent live check of the old remote branch.

There are also the uncommitted 107-line addition to `AGENTS.md`, three prior review reports, and this new report. The 107-line collaboration-rule addition is absent from current main; comparing current `AGENTS.md` to main showed only those additions, with no deletions. Carry that additive change, not an old replacement of the whole file, into the docs/rules landing.

Preserve all 24 committed changes and all uncommitted reports. Then bring the intended docs/scripts/rules into a clean current-main-based candidate, resolving competing edits to existing roadmap documents and keeping historical findings explicitly dated. This can use reviewed cherry-picks/patches after preservation and authorization; it must not silently overwrite newer documentation. Until required CI passes, it is incomplete work, not a merge-ready bundle.

### Four Claude worktrees need explicit scientific-work disposition

- `abba9d3f`: strategy-state analysis. Its implementation, CLI, and test paths are absent from main.
- `bbc10923`: rotated virtual-twin nulls. Its implementation, CLI, and test paths are absent from main.
- `b6e97ed3`: role contrasts, with two local commits. Its implementation, CLI, and test paths are absent from main.
- `305fb4b1`: escape/freeze speed-signal provenance. Its changed files exist on main but differ; this is **not** proof that the intended behavior is missing or already fully implemented. Review the specific delta against newer motion/receipt contracts.

All four are clean but have non-ancestral commits and no matching open PR in the observed list. Their code's readiness/CI was not established by this audit. Keep them or explicitly archive them; do not remove `.claude/` merely because the root status calls it untracked.

Two older documentation branches, `d065360c` (handoff/review queues) and `10bdd1cc` (dry-run audit), also retain non-ancestral work without a matching open PR. Preserve and compare with current documentation rather than treating old age as authorization to delete.

PR #27 has no registered workstation checkout. Its local branch points at `2c2029025d89ca27fbe344311752f732cf08e2f8`, while its local remote-tracking ref and live PR point at `93bd252cf57d19d08801f7a7e724bf68be90420a`. A same-name local branch is not necessarily the PR's current content.

## Proposed staging toward the new baseline

1. **Agree the integration cut and an owner for integration.** Record each included workstream's owner/session, exact head, dirty scope, intended destination, dependencies, and missing evidence. Ask owners to finish that bounded scope or hand off a preserved checkpoint; do not start overlapping architectural refactors during the cut. Do not terminate their processes as “cleanup.”
2. **Preserve before rearranging.** After owner coordination, capture refs/commits in a verified local Git bundle or equivalent recoverable backup, including any important detached heads. Separately preserve staged/unstaged binary patches and untracked files. Git bundles do not contain dirty files, ignored files, LFS payloads, or external scientific stores. Do not create a giant blind archive of the root's data/caches. Keep existing archive/backup/checkpoint branches until their contents are explicitly accounted for.
3. **Land the intended active changes with contracts intact.** Finish #136; finish or explicitly defer the new chaser-core-authority work. Review #110 against current coordinate/receipt helpers and its stated canary requirement. There is no demonstrated hard Git-ancestry dependency between #110 and #136; their shared concern is preserving proof/cache contracts. Sequence through one integrator, testing the combined state as main evolves. #117/#134 can be reviewed as docs-only candidates, not used to claim missing code acceptance.
4. **Resolve leftovers explicitly.** Salvage only genuinely missing work from #57/#27 and the five older dirty checkouts; retire superseded PRs only with evidence and owner approval. Separately decide whether the three unlanded scientific-analysis modules belong before or after this architecture wave. They need not all be merged merely to make a clean baseline possible.
5. **Land review evidence and collaboration rules.** Preserve the current branch's dated audit history and add the rules/report changes in a dedicated reviewed candidate. Ensure all applicable CI passes. This step must reconcile existing document edits, not establish a new parallel fix queue.
6. **Declare the baseline only on the exact final green main SHA.** Verify the remote SHA, required PR checks, current combined/main CI, and any required runtime evidence. Record the exact commit and workflows. Fast-forward local main in an agreed clean checkout; do not reset the root review worktree. A new sibling integration checkout is preferable while root-based sessions remain open. Validate `scripts/py` and the imported `fisheye` path from that checkout. No environment installation is implied.
7. **Clean approved local worktrees in small batches.** For each exact path, recheck head, dirty/untracked/ignored files, owning sessions, retained jobs/deployments, and whether committed work is landed or durably archived. Use ordinary `git worktree remove` without force only after clearance; then consider its exact branch separately. Git's clean check does not establish that ignored files are disposable. Stop on changed state or refusal. Do not bulk-run `git clean`, force-remove checkouts, or use age/branch-name patterns as deletion authority.
8. **Start the architecture wave from that baseline.** Re-verify each finding against the new main, adopt only surviving work into existing queues, assign one owner per shared invariant, and use narrow branches with adversarial contract tests. Keep extraction, performance, enforcement, and schema/authority changes separately reviewable. Preserve numerical values, rig defaults, receipt/digest grammars, and accepted supplier sufficiency.

The desired end state is one clean workstation main checkout, a small number of explicitly owned active worktrees, and preserved/retired historical work with a traceable disposition. It is **not** zero branches or zero deployed commits.

Separately, with explicit repository-administration approval, enable server-side protection for main: required named CI checks, no unintended skipping, and a defined current-base integration policy. This closes the concrete enforcement gap observed here. It should not be silently installed by an audit.

## Deployment retention is a separate task

The shared repository's 281 registrations include historical canaries, retained code snapshots, and locked commit-pinned deployments. Only registry metadata was enumerated; I did not prove every path's existence, clean state, external references, or reproducibility requirements.

The queried account had no unfinished LSF jobs, but that alone is insufficient clearance. Preserve paths/full SHAs referenced by planned jobs, canary receipts, saved commands, services, and provenance. Removing a workstation source checkout does not automatically remove a separate shared deployment; conversely, a merged commit does not make a referenced deployed path disposable.

Do not move the shared primary checkout as a side effect of getting workstation main clean. Do not unlock or delete shared deployments in this cleanup pass. A later separately authorized retirement audit can identify paths whose retained evidence has an approved replacement.

## Workstation inventory

`P` means `/home/delahantyj@hhmi.org/gitrepos/palette`. “Clean” excludes tracked/untracked changes but does not mean absent ignored caches. Initial ignored-file enumeration found cache/build directories in many worktrees. The root additionally has local settings/environment state, outputs, runs, and a model weight file; these were not opened and are not cleanup targets.

The four clean squash/patch-landed worktrees and 40 clean ancestral worktrees are only potential cleanup candidates after the per-path checks above. Directory names are labels, not reliable branch identities.

| Worktree | Actual branch | HEAD | Working tree / disposition |
|---|---|---|---|
| `P` | `agent/palette/refined-assignment-rebinding-gaze-20260831` | `73bee0d5` | DIRTY; Preserve: review branch + uncommitted rules/reports |
| `P/.claude/worktrees/agent-a4952768e724061c4` | `agent/palette/validated-behavior-strategy-states-20260901` | `abba9d3f` | Clean tracked/untracked; Preserve: reconciliation/owner decision |
| `P/.claude/worktrees/agent-a7c58d1a0842b2c80` | `agent/palette/chaser-relative-twin-nulls-20260901` | `bbc10923` | Clean tracked/untracked; Preserve: reconciliation/owner decision |
| `P/.claude/worktrees/agent-a9b9aedb33aefcd10` | `agent/palette/validated-behavior-role-contrasts-20260901` | `b6e97ed3` | Clean tracked/untracked; Preserve: reconciliation/owner decision |
| `P/.claude/worktrees/agent-adfb9d0f3a11a1824` | `agent/palette/escape-freeze-signal-provenance-20260901` | `305fb4b1` | Clean tracked/untracked; Preserve: reconciliation/owner decision |
| `/tmp/claude-64406/-home-delahantyj-hhmi-org-gitrepos-palette/b1875c60-4f12-428e-966e-2a7586a8e569/scratchpad/docs-worktree` | `agent/palette/data-handoff-review-docs-20260821` | `d065360c` | Clean tracked/untracked; Preserve: reconciliation/owner decision |
| `/tmp/palette-canonical-clipped-track-fps-20260902` | `agent/palette/canonical-clipped-track-fps-20260902` | `162a887a` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-canonical-downstream-20260825` | `agent/palette/canonical-downstream-20260825` | `a17706b2` | Clean tracked/untracked; Landed: Squash #62 + patch-equivalent follow-up |
| `/tmp/palette-chaser-core-authority-20260904` | `agent/palette/chaser-core-authority-20260904` | `68f7b7a7` | DIRTY; ACTIVE edits: new core-authority work |
| `/tmp/palette-chaser-docs-pr-20260902` | `agent/palette/chaser-docs-pr-20260902` | `824527b7` | Clean tracked/untracked; Open PR #117 |
| `/tmp/palette-chaser-exact-successor-marimo-20260826` | `agent/palette/chaser-body-frame-projection-contract-20260827` | `e240fe16` | Clean tracked/untracked; Landed: Squash #66 verified |
| `/tmp/palette-chaser-near-field-visit-cohort-20260903` | `agent/palette/chaser-near-field-visit-cohort-20260903` | `eb9eee46` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-chaser-near-field-visit-plots-20260903` | `agent/palette/chaser-near-field-visit-plots-20260903` | `49626110` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-chaser-near-field-visits-20260902` | `agent/palette/chaser-near-field-visits-20260902` | `9aac6ff2` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-clipped-downstream-analytics-20260824` | `agent/palette/subject-shape-v5-supported-20260824` | `d8cb0c21` | DIRTY; Preserve dirty work; owner disposition required |
| `/tmp/palette-core-motion-authority-20260904` | `agent/palette/core-motion-authority-20260904` | `a83eaa24` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-core-motion-framewise-20260903` | `agent/palette/core-motion-framewise-20260903` | `30ab32f6` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-deploy-eye-angle-receipt-68d4edbd` | `agent/deploy/eye-angle-receipt-68d4edbd` | `68d4edbd` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-deterministic-ellipse-20260830` | `agent/palette/geometry-only-subject-mask-rebase-20260830` | `6e3a2361` | DIRTY; Preserve dirty work; owner disposition required |
| `/tmp/palette-direct-hybrid-main-20260825` | `agent/palette/direct-hybrid-keypoint-evidence-20260825` | `62873ccb` | Clean tracked/untracked; Landed: Squash #58 verified |
| `/tmp/palette-distribution-ibi-display-label-20260904` | `agent/palette/explorer-receipt-cache-20260904` | `de75bfbf` | Clean tracked/untracked; Open PR #136 |
| `/tmp/palette-dry-run-audit-mainline-20260821` | `agent/palette/dry-run-audit-mainline-20260821` | `10bdd1cc` | Clean tracked/untracked; Preserve: reconciliation/owner decision |
| `/tmp/palette-eye-angle-access-aware-receipt-20260901` | `agent/palette/eye-angle-access-aware-receipt-20260901` | `f94b00e0` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-eye-angle-admission-receipt-reuse-20260901` | `agent/palette/eye-angle-admission-receipt-reuse-20260901` | `5c4df0b7` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-eye-angle-consolidated-activation-20260901` | `agent/palette/eye-angle-consolidated-activation-20260901` | `48b0acfc` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-eye-angle-evidence-planning-20260902` | `agent/palette/eye-angle-evidence-planning-20260902` | `aca708a0` | Clean tracked/untracked; Open PR #110 |
| `/tmp/palette-eye-angle-receipt-deploy-caceb58e` | `agent/palette/eye-angle-receipt-deploy-caceb58e` | `caceb58e` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-eye-angle-shared-receipt-deploy-853848dc` | `agent/deploy/eye-angle-shared-receipt-853848dc` | `853848dc` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-eye-angle-shared-receipt-validation-20260901` | `agent/palette/eye-angle-shared-receipt-validation-20260901` | `11236b5e` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-eye-angle-source-profile-fix-20260901` | `agent/palette/eye-angle-source-profile-fix-20260901` | `e6be9587` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-eye-dag-receipt-20260902` | `agent/palette/eye-dag-receipt-20260902` | `5baa1a1c` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-fullrate-successor-20260904` | `agent/palette/fullrate-successor-20260904` | `34cddfdf` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-gaze-cohort-receipt-aware-plots-20260831` | `agent/palette/gaze-cohort-receipt-aware-plots-20260831` | `a7eab26a` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-gaze-convention-cohort-review-20260831` | `agent/palette/gaze-convention-cohort-review-20260831` | `cfa0c792` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-keypoint-lineage-resolver-20260902` | `agent/palette/keypoint-lineage-resolver-20260902` | `b9d52b30` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-mask-activation-20260831` | `agent/palette/direct-bundle-eye-keypoint-resolver-20260831` | `f10893de` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-merged-32150c90` | `(detached)` | `32150c90` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-protocol-semantic-step-identity-20260824` | `agent/palette/protocol-semantic-step-identity-20260824` | `ec374edf` | Clean tracked/untracked; Landed: Squash #59 verified |
| `/tmp/palette-recording-behavior-distributions-20260903` | `agent/palette/recording-behavior-distributions-20260903` | `b66949e6` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-recording-behavior-distributions-main-canary-20260903` | `agent/palette/recording-behavior-distributions-main-canary-20260903` | `a27a8489` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-recording-distribution-histogram-bars-20260903` | `agent/palette/recording-distribution-histogram-bars-20260903` | `1c6119ab` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-sleepyfish-cohort-export-20260902` | `agent/palette/sleepyfish-cohort-export-20260902` | `289e9ddc` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-sleepyfish-eye-angle-20260831` | `agent/palette/clipped-source-fps-authority-20260831` | `1b6d14c0` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-spatial-occupancy-display-v4-20260903` | `agent/palette/spatial-occupancy-display-v4-20260903` | `a33ebc67` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-subject-mask-cpu-optimization-20260831` | `agent/palette/subject-mask-cpu-optimization-20260831` | `ba20b813` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-subject-shape-optimization-20260825` | `agent/palette/subject-shape-read-amplification-20260825` | `d8cb0c21` | DIRTY; Preserve dirty work; owner disposition required |
| `/tmp/palette-subject-shape-receipt-failclosed-20260903` | `agent/palette/subject-shape-receipt-failclosed-20260903` | `86111041` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-subject-shape-receipt-load-20260903` | `agent/palette/subject-shape-receipt-load-20260903` | `edebebc7` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-tail-receipt-publication-20260902` | `agent/palette/tail-receipt-publication-20260902` | `07db267c` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-tail-standard-schema-20260903` | `agent/palette/tail-canonical-fps-20260903` | `f8e9e212` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-targeted-reconstruct-20260821` | `(detached)` | `5a3a8786` | DIRTY; Preserve dirty work; owner disposition required |
| `/tmp/palette-v011-deploy-source-eafbeaf5` | `agent/palette/schema-mask-completeness-v018-deploy` | `4277cd5d` | DIRTY; Preserve dirty work; owner disposition required |
| `/tmp/palette-validated-behavior-chaser-appearance-export-20260901` | `agent/palette/validated-behavior-chaser-appearance-export-20260901` | `32a6ee90` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-validated-behavior-distributions-20260902` | `agent/palette/validated-behavior-distributions-20260902` | `bec43b9a` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-validated-behavior-distributions-integration-evidence-20260902` | `agent/palette/validated-behavior-distributions-integration-evidence-20260902` | `9882d2ed` | Clean tracked/untracked; Duplicate documents of #117; preserve until disposition |
| `/tmp/palette-validated-behavior-phase-c-production-20260902` | `agent/deploy/validated-behavior-phase-c-production-20260902` | `19a006cc` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-validated-behavior-phase-c-production-evidence-20260902` | `agent/palette/validated-behavior-phase-c-production-evidence-20260902` | `dd8c6153` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-validated-behavior-product-catalog-20260903` | `agent/palette/validated-behavior-product-catalog-20260903` | `8aae12f9` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-validated-behavior-product-catalog-main-20260904-0d82d778` | `agent/deploy/validated-behavior-product-catalog-main-20260904` | `0d82d778` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-validated-behavior-receipt-v2-20260904` | `agent/palette/validated-behavior-receipt-v2-20260904` | `c13fcbed` | Clean tracked/untracked; Landed; owner/use/retention check |
| `/tmp/palette-validated-behavior-receipt-v2-canary-20260904` | `agent/deploy/validated-behavior-receipt-v2-canary-20260904` | `916f9d61` | Clean tracked/untracked; Landed; deployment/evidence retention check |
| `/tmp/palette-validated-behavior-receipt-v2-evidence-20260904` | `agent/palette/validated-behavior-receipt-v2-evidence-20260904` | `a0296d63` | Clean tracked/untracked; Open PR #134 |

## Dirty-file preservation inventory

The following lists are observations, not instructions to stage them together. In particular, do not stage nested worktrees as part of the review-doc commit. The root list predates creation of this report; this report is an additional untracked file.

### Current review checkout

Path: `/home/delahantyj@hhmi.org/gitrepos/palette`; HEAD: `73bee0d5194c662e3b7e535be3de92db7ef53f63`.

```text
M AGENTS.md
?? .claude/worktrees/agent-a4952768e724061c4/
?? .claude/worktrees/agent-a7c58d1a0842b2c80/
?? .claude/worktrees/agent-a9b9aedb33aefcd10/
?? .claude/worktrees/agent-adfb9d0f3a11a1824/
?? docs/diagnostics/independent_codebase_design_assessment_2026-09-04.md
?? docs/diagnostics/review_wave_second_opinion_2026-09-04.md
?? docs/diagnostics/shared_helper_consolidation_review_2026-09-04.md
```

### palette-chaser-core-authority-20260904

Path: `/tmp/palette-chaser-core-authority-20260904`; HEAD: `68f7b7a7cff64f2483df2b39c135dcb369ca82ea`.

```text
 M src/fisheye/analysis_workflows/core_behavior_cohort_adapter.py
 M src/fisheye/analytics_exports/validated_behavior_contracts.py
?? src/fisheye/analysis_workflows/core_authority_roster.py
?? tests/unit/fisheye/test_core_authority_roster.py
```

### palette-clipped-downstream-analytics-20260824

Path: `/tmp/palette-clipped-downstream-analytics-20260824`; HEAD: `d8cb0c21cc1352ad075135ac6e74da00399dfa8e`.

```text
M src/fisheye/analysis/detect_bouts_multi_level.py
 M src/fisheye/analysis_workflows/execution.py
 M tests/unit/fisheye/test_analysis_workflow_execution.py
?? tests/unit/fisheye/test_detect_bout_point_mapping.py
?? tests/unit/fisheye/test_detect_bouts_level_parallel_equivalence.py
```

### palette-deterministic-ellipse-20260830

Path: `/tmp/palette-deterministic-ellipse-20260830`; HEAD: `6e3a2361913ce4fe827e3057017e40633fd452d1`.

```text
M src/fisheye/diagnostics/build_subject_mask_finalizer_ab_fixture.py
 M src/fisheye/refinement/finalize_subject_masks.py
 M src/fisheye/utils/finalize_keypoint_shards.py
 M tests/unit/fisheye/test_finalize_subject_masks.py
```

### palette-subject-shape-optimization-20260825

Path: `/tmp/palette-subject-shape-optimization-20260825`; HEAD: `d8cb0c21cc1352ad075135ac6e74da00399dfa8e`.

```text
M src/fisheye/analysis_workflows/materializers/subject_shape.py
 M src/fisheye/shared/coordinate_frame_record.py
 M src/fisheye/shared/subject_shape_coordinate_publication.py
 M tests/unit/fisheye/test_coordinate_frame_record.py
 M tests/unit/fisheye/test_subject_shape_coordinate_publication.py
?? src/fisheye/analysis_workflows/handoff_validation.py
?? src/fisheye/shared/array_payload_read_telemetry.py
?? tests/unit/fisheye/test_analysis_workflow_handoff_validation.py
```

### palette-targeted-reconstruct-20260821

Path: `/tmp/palette-targeted-reconstruct-20260821`; HEAD: `5a3a8786be1a83bd2b9c8515b7ca70b545f4aa3f`.

```text
M src/fisheye/detection/detect_keypoints_yolo.py
 M src/fisheye/utils/run_keypoints_with_registry_model.py
```

### palette-v011-deploy-source-eafbeaf5

Path: `/tmp/palette-v011-deploy-source-eafbeaf5`; HEAD: `4277cd5d9c8be9deaf18cb35b2c12e2afda9db57`.

```text
M docs/swim_bladder_polar_boundary_design.md
?? docs/diagnostics/sleepyfish_cam2010094_swim_bladder_model_failure_2026-08-23.md
```

## Verification limits and reproduction

Read-only evidence used:

- `git --no-optional-locks worktree list --porcelain`, per-worktree status (including untracked files), ignored-file listings, branch refs, `rev-list`, merge-base ancestry, targeted diffs/logs, and stable aggregate patch IDs.
- Live GitHub open PR metadata/checks, the 100 most recent merged PR records, main's current commit/rules, and the exact main workflow run.
- Workstation process metadata outside the sandbox, without commands' full arguments or session contents.
- Read-only shared Git registration enumeration and a single current-account LSF job query.

No test suite was run: this is a state/integration audit, not revalidation of all pending code. Successful GitHub results are reported as observed external evidence. Owner confirmation, global agent activity, all historical branch supersession, every ignored artifact's value, and deployment-retirement eligibility remain unverified. Refresh this inventory before acting: one worktree appeared and became dirty during the audit itself.
