# DAG motion dependency pinning — September 9, 2026

## Ownership and status

The status below records the pre-CI implementation checkpoint. On September 9,
the user subsequently authorized committing this scoped change, pushing its
branch, opening a draft PR, and monitoring all required CI. The branch's PR
records the exact candidate commit and current CI results. Merge, deployment,
and shared-checkout updates remain outside that authorization.

This is evidence for the scoped correction in the
[owning queue](authority_consolidation_work_queue_2026-08-25.md#scoped-dag-dependency-pinning-correction--2026-09-09),
not another status authority. It implements only slice 1 of the
[generic DAG review](generic_dag_gap_refresh_2026-09-09.md).

- Owner: root implementation agent; no parallel implementation agents.
- Worktree: `/tmp/palette-generic-dag-gap-review-20260909`, retained Git-locked.
- Branch: `agent/palette/generic-dag-gap-review-20260909`.
- Exact base/prerequisite: `fc6fea5fd937759f5ed21c79d13279b25c309e13`.
  All changes are uncommitted; no new implementation commit exists.
- Implemented and focused-locally validated. Required new-change CI is unrun:
  **not complete or merge-ready**. Not integrated, deployed, or activated.
- No commit, push, dependency installation, cluster submission, historical
  mutation, selector change, registry write, or shared-checkout update.
- Reconciled the older clipped-analytics worktree's overlapping `execution.py`
  change (swim-bout worker cap at base `d8cb0c21`); that separate performance
  edit and all other workers' checkouts remain untouched.

## Contract and implementation

Classification: **enforcement correction**. Scientific formulas/defaults,
coordinates, row/frame identity, source payloads, manifests/digest bytes, and
existing lifecycle/authority validators remain unchanged. No persisted grammar,
review gate, or public helper interface was introduced.

`analysis_workflows/execution.py` now passes exactly one `--tracking-run` for
production as well as canary motion commands. It uses the dependency already
resolved by the planner. Users can still let planning select available inputs;
they do not need to manually name every production dependency.

`analysis_workflows/runtime_verification.py` retains the full strict motion
loader, then compares its validated manifest's derivation `source_refs` against
the plan's selected `tracks` and `refined_keypoints`, when those entries exist.
Raw keypoints use the existing bare-name encoding; refined selections use
`refined/<name>`. Exact root-relative source paths also match. Raw and refined
families with the same leaf name are not equivalent.

Intentional tightening: an internally valid motion artifact cannot satisfy an
explicitly conflicting tracking/keypoint selection, an unresolved `latest`, or
an empty selected tracking name. This applies to post-output verification and
reuse, including the supported canary profile. No selector is resolved again.

Preserved supplier sufficiency: dependencies absent from the plan remain absent;
a validated reused supplier can close its branch without newly requiring its
sealed ancestors as independently selected authorities. Existing payload,
lineage, completion, eligibility, and publication checks still run first.

Dirty implementation scope: the two runtime files above;
`test_analysis_workflow_execution.py`,
`test_single_subject_per_arena_tracking.py`,
`test_track_motion_publication.py`, and new
`test_analysis_workflow_runtime_verification.py` under `tests/unit/fisheye/`.
Documentation scope: this handoff, its owning-queue entry, and the pre-existing
untracked review document (with an additive handoff link and corrected first
bout-refusal location). No compatibility adapter or executable copy was removed.

## Validation evidence

All pytest used workstation `scripts/py` outside the sandbox, synthetic temporary
inputs, and `-p no:cacheprovider`. Python resolved to `palette-py311`; `fisheye`
imported from this exact worktree's `src` tree. No production archive was tested.

Tests preceded the implementation. The real DAG argv -> materializer parser ->
materialization plan -> real tracking loader regression selected
`tracking_latest` instead of `tracking_selected` before the fix; twelve valid
production/canary and sealed-branch preservation cases passed at that point.
Earlier mismatch tests also exposed acceptance of conflicting source selections.
Six initial positive canary fixture failures were fixture errors (profile set
after sealing), corrected by setting the profile before the real seal. They are
not claimed as six additional runtime defects.

First post-fix run: **81 passed, 54.21 seconds**:

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-dag-pin-pycache scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_analysis_workflow_execution.py \
  tests/unit/fisheye/test_analysis_workflow_runtime_verification.py \
  tests/unit/fisheye/test_single_subject_per_arena_tracking.py -q --tb=short
```

After adding four canary conflict cases, broader run: **253 passed, 22 Zarr
structured-dtype warnings, 151.00 seconds**. Across these runs, 290 distinct tests
passed; their shared runtime-verification cases are not counted twice.

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-dag-pin-pycache scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_analysis_workflow_runtime_verification.py \
  tests/unit/fisheye/test_analysis_workflow_dag.py \
  tests/unit/fisheye/test_track_kinematics_materializer.py \
  tests/unit/fisheye/test_track_motion_publication.py \
  tests/unit/fisheye/test_track_kinematics_io.py \
  tests/unit/fisheye/test_track_kinematics_lifecycle.py \
  tests/unit/fisheye/test_submit_analysis_workflow_bsub.py -q --tb=short
```

Final rerun of the first command after formatting and the added canary cases:
**85 passed, 60.00 seconds**. This does not add new distinct tests to the 290 total.

The new 48-case runtime suite uses the existing deterministic in-memory motion
producer/seal fixture and its supplied position authority, temporary discovery
metadata, and an adapted archive-opening boundary. Neither the strict motion
loader nor the public DAG verifier is patched. Coverage includes raw/refined
sources, exact and conflicting selections, post-output/reuse, production/canary,
root-relative paths, missing ancestors, tampered numerical/manifest/source
payloads, and incomplete output. This is not a claim of a complete generic CLI
to atomic-filesystem-publication end-to-end test.

Local checks passed: Python compilation, `git diff --check`, generated Zarr
inventory, registry schema reference, both import-layer contracts, file-size
ratchet, Zarr metadata-mode ratchet, observed metadata literals, active contract
freshness, and FPS/keypoint-motion/tail-payload/paradigm-core access ratchets.
Ratchet baselines were not rewritten (`--no-update-on-shrink` where supported).
The import checker used the installed `importlinter.cli` through `scripts/py`
with `--config pyproject.toml --no-cache`; no dependencies were installed.

Black passed for the new test module, command builder and its tests, and the
changed line ranges in the other three modified Python files. The initial
sandboxed Black check stalled and was interrupted; the outside-sandbox rerun
reported existing whole-file formatting differences outside the changed blocks.
Those unrelated formatting differences were preserved, so whole-file Black for
those three files is not claimed green. Black is not a required CI job in this
workflow; this does not waive any required CI gate below.

## Remaining validation and integration

Every new-change required CI job in `.github/workflows/ci.yml` remains **unrun**:
`generated artifacts`, `import boundaries`, `file-size ratchet`,
`zarr open metadata modes`, `observed metadata literals`,
`active contract freshness`, `package and collection`,
`non-gpu tests (shard 0)` through `non-gpu tests (shard 15)` (all sixteen), and
`ci-required`. Local component checks do not satisfy these 24 CI jobs. There is
no new-change CI attempt with a successful, failed, or cancelled result to cite.

Full test collection, package wheel/install validation, all non-GPU shards, a
new generic DAG canary, and benchmarks were not run. The earlier review's
interrupted subject-shape candidate test remains deferred evidence for its
separate lifecycle slice; this correction does not mark it passed.

The authorized CI workflow will commit this exact scope and obtain all required
CI on that candidate. Any later combined integration commit needs its own
successful required CI. Merge and deployment are separate actions and remain
unauthorized. Broader generic admission,
shape/eye/bout canary wiring, clipped intake, and proof-catalog work remain open.
