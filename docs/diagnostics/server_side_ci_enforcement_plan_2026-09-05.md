# Server-side CI enforcement rollout for Palette

> Integration update, 2026-09-06: Phase B is active. Main requires all original
> 23 checks **plus `ci-required`**, with strict current-base checks and GitHub
> Actions source app 15368. [PR #140](https://github.com/jmdelahanty/palette/pull/140)
> merged as `3c98979be9bf4eb19db5a855950e53f8dcec563f`; all 24 post-merge
> checks passed in [run 34044178789](https://github.com/jmdelahanty/palette/actions/runs/34044178789)
> before the approved aggregate addition to existing ruleset `22372296` at
> 16:23 UTC. No other protections or bypass settings changed. The ruleset and
> effective main rules were read back and verified. See the
> [integration evidence](review_docs_rules_landing_handoff_2026-09-05.md#authorized-main-integration--september-6).

## Historical proposal — September 5

The proposal, observed settings, and “not performed” statements below describe
the September 5 audit, not the completed September 6 rollout. Do not recreate
either phase. Subsequent successful ordinary protected merges provide positive
server-side evidence; no intentionally failing/skipped disposable PR was created
to claim the broader adversarial server-side acceptance described below.

Date: 2026-09-05. Proposed configuration only: no GitHub settings or CI workflow were changed.

## Observed repository state

- Repository: `jmdelahanty/palette`, public, owned by a personal account (`User`).
- Current credentials report repository admin access.
- Verified main: `6af66c5a6ba3b35ea0bf00cfc74add7bb22da2b2`.
- Its CI [run 34004329735](https://github.com/jmdelahanty/palette/actions/runs/34004329735) completed with all 23 jobs successful.
- Main is unprotected, and the repository ruleset list is empty.
- Successful checks came from GitHub Actions: app slug `github-actions`, app ID `15368`. Re-query before configuring; do not accept arbitrary status publishers.

Branch rulesets are available for public repositories without changing repository ownership or purchasing a different plan. Use a **branch ruleset**, not a push ruleset. [GitHub availability](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)

## Phase A: enforce the existing checks

After approval to change repository administration, open [Palette repository rules settings](https://github.com/jmdelahanty/palette/settings/rules). Create a new branch ruleset:

| Setting | Proposed value |
|---|---|
| Name | `main-required-ci` |
| Enforcement status | Active |
| Target | Exactly `refs/heads/main` |
| Bypass list | Empty: no blanket administrator, collaborator, bot, or deployment bypass |
| Require pull request | Enabled |
| Required approving reviews | 0 initially; independent human-review policy needs a separate named-reviewer decision |
| Required status checks | All 23 exact names below |
| Expected check source | GitHub Actions, verified app ID 15368 |
| Require branch up to date | Enabled (strict current-base checks) |
| Block force pushes | Enabled |
| Restrict deletions | Enabled |
| Restrict updates | Leave disabled: that rule requires bypass to update and is not the ordinary CI gate |
| Require linear history | Leave disabled: existing merge-commit policy is unchanged |
| Deployment/signing/merge-queue requirements | Do not introduce in this rollout |

Keep ordinary merge commits allowed. The up-to-date requirement automates the current-base discipline used for #110; other PRs may need a branch update and another CI run after a competing merge. Empty bypass means normal merges must satisfy the rules, including merges by administrators; administrators can still edit the rules themselves, so retain configuration/audit evidence. [Rule behavior and strict checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)

Exact current status-check contexts (not workflow IDs or display guesses):

```text
generated artifacts
import boundaries
file-size ratchet
zarr open metadata modes
observed metadata literals
active contract freshness
package and collection
non-gpu tests (shard 0)
non-gpu tests (shard 1)
non-gpu tests (shard 2)
non-gpu tests (shard 3)
non-gpu tests (shard 4)
non-gpu tests (shard 5)
non-gpu tests (shard 6)
non-gpu tests (shard 7)
non-gpu tests (shard 8)
non-gpu tests (shard 9)
non-gpu tests (shard 10)
non-gpu tests (shard 11)
non-gpu tests (shard 12)
non-gpu tests (shard 13)
non-gpu tests (shard 14)
non-gpu tests (shard 15)
```

Select the individual reported names, including every shard 0 through 15. Recheck that these names have recently succeeded before activation. A renamed check requires a coordinated workflow/ruleset update; do not remove a requirement merely to clear a blocked PR.

The API equivalent is a reviewed repository ruleset using `target: branch`, `enforcement: active`, `conditions.ref_name.include: [refs/heads/main]`, no bypass actors, and the `pull_request`, `required_status_checks`, `non_fast_forward`, and `deletion` rules. Required status checks must set strict current-base behavior and the verified integration source. Prefer the UI for the initial administrative action unless an exact, current API payload has been separately reviewed; no mutating API command is provided or run here.

## Phase B: match Palette's stronger success-only contract

**Phase A is useful but not sufficient to claim exact enforcement of AGENTS.md.** GitHub treats `success`, `skipped`, and `neutral` as acceptable required-check conclusions. Palette instead requires actual success unless its contract explicitly declares a check inapplicable. A dependency-skipped test matrix must not become a pass. [GitHub's documented skipped-check behavior](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)

Prepare a separate, narrow CI-only PR adding a stable aggregate check, for example `ci-required`. Do not mix that workflow change into this docs/rules draft.

Required behavior of that follow-up:

- Depend on every required gate and the test matrix, and run its verification even after dependency failure/skip using an unconditional terminal job such as `if: always()`. Do not use a success-only terminal job that can itself silently skip. [Dependency handling](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idneeds)
- Verify the seven required non-matrix jobs and every expected shard 0 through 15 are accounted for by successful evidence for the exact candidate/workflow execution. Merely checking an aggregate matrix result does not prove that all 16 expected shards were instantiated.
- Fail closed on missing, skipped, cancelled, failed, timed-out, neutral, conflicting, or ambiguous evidence. Treat explicitly allowed inapplicability as a declared/tested exception, not a guessed default. Currently these 23 jobs are all required.
- Specify and test rerun semantics. A failed-jobs-only rerun may retain valid earlier successful jobs for the same candidate, but must not accidentally accept stale results from another SHA or workflow. Requiring a full workflow rerun is a conservative initial alternative.
- Keep expected job/shard ownership tied to the workflow and test drift. Do not create a second independently maintained shard list without a conformance check.
- Preserve failure exit codes; no `continue-on-error` workaround. Retain exact names and the existing required jobs.
- Use read-only permissions needed to inspect evidence. Do not introduce `pull_request_target` executing untrusted PR code or elevated write tokens just to implement this gate.
- Add synthetic tests for all-success, upstream failure with skipped dependents, single skipped shard, missing shard, cancellation/timeout, duplicate/conflicting evidence, and rerun behavior. These tests should not deliberately alter production main.

After all required CI for that PR passes, merge it under the existing gate, verify the resulting main run, and **add `ci-required` to the required checks without removing the existing 23**. Do not require a check name before its implementing workflow exists and has reported it successfully.

Until Phase B is implemented and verified, handoffs must say that GitHub enforces the named checks but that its skipped/neutral semantics remain weaker than Palette's policy.

## Trust boundary and approvals

These rules enforce server-reported results under the reviewed workflow. They do not prove that a job still executes the intended tests if someone edits the workflow to report success unconditionally. Selecting GitHub Actions as the expected source blocks other status publishers but does not distinguish trustworthy and untrustworthy workflows from that same app.

Protect workflow, gate, and relevant test/ratchet configuration changes through explicit code review. If independent human approval is desired, designate another eligible reviewer and configure approval/CODEOWNERS requirements deliberately. Multiple agents using the same GitHub identity are not independent approvers, and a PR author cannot supply their own required approval. Do not impose a one-reviewer rule on a sole-maintainer workflow without resolving that ownership first. [GitHub workflow security guidance](https://docs.github.com/en/actions/reference/security/secure-use)

No rule makes the repository administrator unable to change administration. Empty bypass, reviewed workflow changes, and recorded ruleset changes provide separate layers. This is not a claim that all conceivable bypasses are impossible.

The rule protects main updates on GitHub. It cannot prevent local branch combinations, local source edits, direct store writes, or movement of an unrelated shared filesystem checkout. Existing integration, commit-pinned deployment, and authority-activation contracts continue to apply.

## Why not start with a merge queue?

Palette is personally owned. GitHub documents merge queues for organization-owned public repositories or qualifying private organization repositories; do not plan on that feature in the current ownership setup. Strict up-to-date checks are the supported starting point. A later separately approved ownership/queue change would also require `merge_group` CI triggers. No transfer or trigger change is proposed now. [Merge-queue availability](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)

## Acceptance and operational handoff

1. Export/read the prior settings before applying the approved rule. Do not replace unrelated future rulesets.
2. Read back both the created ruleset and the effective rules for main; verify active enforcement, exact main target, empty bypass, required contexts/source, strictness, deletion restriction, and force-push blocking.
3. On separately authorized disposable PRs, verify that pending/failed checks and an out-of-date base block integration; after Phase B, cover skipped/missing evidence too. Inspect the blocked state without attempting an unsafe merge into main as a “test.”
4. Verify an ordinary current-base successful PR can use the normal merge path. Record the exact head/base/check evidence and subsequent main CI.
5. Record the ruleset ID/export, check-source identity, gate commit, acceptance evidence, and approval. These are administrative evidence, not another code fix queue.
6. Do not use `gh pr merge --admin`, delete/recreate a required check, disable rules, or add a bypass to work around a failed run. Any emergency administrative change needs explicit approval and its own audit trail.

No acceptance tests of server-side enforcement have been performed: the configuration is still only proposed.
