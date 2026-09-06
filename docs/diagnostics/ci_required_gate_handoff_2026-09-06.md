# Success-only CI gate: implementation handoff

Date: 2026-09-06. Owner: the CI-gate implementation session.
Local-validation snapshot, before publication: implemented and locally validated;
uncommitted and **not merge-ready**. The subsequent PR records the published
candidate SHA and live CI status; this snapshot does not claim remote CI success.

## Location and scope

- Worktree: `/tmp/palette-ci-required-gate-20260906`.
- Branch: `agent/palette/ci-required-gate-20260906`.
- Exact implementation base: `d3d9b741b9d03cdd9ea6b933511e59ef5ae29504`.
- Prerequisite: that main commit; no other incoming branch is incorporated.
- At this snapshot, there was no candidate commit and all six paths were uncommitted.
- The user subsequently authorized committing/pushing this branch and opening
  its PR. Merge, ruleset changes, deployment, shared-checkout updates, production
  mutation, and dependency installation remain outside that authorization.

Classification: enforcement correction. GitHub accepts skipped and neutral
required-check conclusions; this gate deliberately requires completed/success
for every prerequisite job. See [GitHub's required-check semantics](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).

Preserved: all seven existing single-job names, all sixteen test shards, every
existing job/step/command, workflow triggers, concurrency, and test selection.
No scientific, schema, receipt, digest, runtime-product, or publication contract
changes. The only generated-data differences are scanned Python module counts
`1707 -> 1708`; no Zarr writer/schema findings change.

Changed/new files:

- [Workflow](../../.github/workflows/ci.yml): append `ci-required`; no existing job changes.
- [Gate](../../scripts/ci_required.py): standard-library verifier.
- [Tests](../../tests/unit/fisheye/test_ci_required.py): evidence and workflow-wiring coverage.
- [Schema census](zarr_array_schema_census.json) and
  [writer census](zarr_production_writer_census.json): generated module counts only.
- This handoff.

## Gate contract

`ci-required` has `if: always()` and directly needs all eight existing job IDs,
including the matrix job. It requires both the exact dependency inventory with
successful `needs` results and all 23 individual successful API job records.
The matrix aggregate alone is not sufficient.

Evidence is bound to repository, workflow path/name, event, run ID, attempt, and
head SHA. Local checkout HEAD must equal `GITHUB_SHA`. For pull requests, the
API head SHA is the PR head while checkout tests GitHub's merge SHA; these are
validated separately, not incorrectly equated. All upstream checkout steps
retain their default candidate checkout.

The verifier uses the [attempt-specific jobs API](https://docs.github.com/en/rest/actions/workflow-jobs#list-jobs-for-a-workflow-run-attempt),
checks pagination totals, rejects duplicate IDs/names and unexpected jobs, and
rereads run identity afterward to detect attempt rollover. Missing, skipped,
neutral, cancelled, timed-out, failed, incomplete, malformed, or foreign evidence
fails closed. There are no inapplicable prerequisite jobs in this workflow.
The running terminal job is not its own prerequisite. Optional fixture-cache
and timing steps keep their existing conditions; the contract concerns job
conclusions, not requiring every optional step to run.

Retry policy: all 23 prerequisites must succeed in one attempt. Earlier-attempt
successes are never reused, even for the same run/SHA. Partial or gate-only
reruns fail with instructions to select **Re-run all jobs**. A full successful
rerun can pass. This deliberately costs more than retrying only failed jobs.
See [GitHub's rerun controls](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

The gate has only `actions: read` and `contents: read`, disables checkout's
persisted credentials, and exposes `GH_TOKEN` only to the verification step.
It uses runner-provided GitHub CLI and Python's standard library, with no new
dependency install. GitHub documents [CLI availability and token use](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-github-cli).
No privileged `pull_request_target` execution is added.

The inventory is owned by `scripts/ci_required.py` and guarded against workflow
drift by its tests, alongside the existing quality-gate independence tests.
Workflow/script review remains a trust boundary: a contributor able to change
and approve CI could change or bypass the gate. This is not a claim that an
Actions-app required context prevents malicious workflow changes.

## Local evidence

All Python validation used `scripts/py` and the workstation's `palette-py311`
runtime. Pytest ran outside the sandbox.

```bash
scripts/py -B -m pytest -q --color=no \
  tests/unit/fisheye/test_ci_required.py \
  tests/unit/fisheye/test_ci_quality_gate_independence.py \
  tests/unit/fisheye/test_agents_required_ci_policy.py \
  tests/unit/fisheye/test_ci_pytest_shard.py \
  tests/unit/fisheye/test_ci_pytest_junit_summary.py
```

Result: **142 passed**. Tests were added before the implementation and initially
failed on its missing import. Coverage includes every missing prerequisite,
non-success conclusions, collapsed/skipped matrices, pagination, duplicates,
wrong identities, partial/full reruns, attempt rollover, API failures, checkout
mismatch, and workflow inventory/permissions/wiring.

Full collection: `scripts/py -B -m pytest --collect-only -q --color=no -rs tests/`
returned success with **11,383 collected** and one module skipped:
`test_train_unet_subject_masks_registry.py:23`, whose existing guard reports a
host-specific `torch._dynamo` import bug from a hyphenated `nx-loopback` hostname.
One existing invalid-escape deprecation warning was reported from
`web_personal_renderers.py:1344`. Collection is not full test execution; that
skipped module remains unvalidated locally.

Also passed: Python compilation; generated census freshness; import-layer
contracts (2 kept, 0 broken); file-size, Zarr metadata-mode, FPS authority,
keypoint-motion authority, tail receipt, and paradigm authority ratchets;
observed-metadata literals; contract freshness; registry schema reference;
unchanged ratchet baselines; and `git diff --check`. A structural comparison
against HEAD confirmed that removing the new terminal job yields the unchanged
original workflow. `actionlint` is not installed and was not run or installed.

A read-only, standard-library-only API smoke successfully validated all 23
actual prerequisite records from existing main
[run 34012958031, attempt 1](https://github.com/jmdelahanty/palette/actions/runs/34012958031).
This used locally supplied `needs` inputs and the existing base SHA; it verifies
API compatibility, **not candidate CI execution or runner-token permissions**.

## Required evidence still missing / rollout

At this local-validation snapshot, every remote check is unrun for the candidate:

- `generated artifacts`
- `import boundaries`
- `file-size ratchet`
- `zarr open metadata modes`
- `observed metadata literals`
- `active contract freshness`
- `package and collection` (including installed-wheel validation)
- `non-gpu tests (shard 0)` through `non-gpu tests (shard 15)`, all sixteen
- New `ci-required` (including real runner scheduling and read-only token access)

No failing local gate tests remain, but these local results do not replace the
unrun required GitHub checks. Live candidate execution and full/partial rerun
scheduling have not been exercised on GitHub; rerun semantics have unit coverage.

Publication authorization now permits committing/pushing this branch and opening a PR.
Require all existing 23 checks **and the new gate** to finish successfully before
merge. If the base changes, reconcile and validate the resulting candidate;
the previously green base is not evidence for a new combination. After an
authorized merge, verify the resulting main run as well. Then add `ci-required`
as the 24th required context in `main-required-ci`, selecting GitHub Actions as
its source and preserving the existing 23 contexts and other protections.
That ruleset change is not part of this local implementation authorization.

Implemented: yes. Locally validated: as above. Integrated, deployed, and
activated: no. Compatibility/removal work: none beyond the explicit full-rerun
policy; no existing checks or helpers are removed.
