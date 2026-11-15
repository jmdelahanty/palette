# DAG catalog tracking alignment — September 10, 2026

## Ownership and status

This is the implementation checkpoint for the bounded catalog/status follow-up
to the merged exact-dependency pinning change. It is evidence for the
[owning queue](authority_consolidation_work_queue_2026-08-25.md#scoped-dag-catalog-tracking-alignment--2026-09-10),
not a second authority or a claim that provider-neutral tracking is complete.

- Owner: root implementation agent; no parallel implementation agents.
- Worktree: `/tmp/palette-dag-catalog-alignment-20260910`.
- Branch: `agent/palette/dag-catalog-alignment-20260910`.
- Exact clean base/prerequisite:
  `6fd555d05a50dec82c85e5edc567159a0c491f4d`, the merge of PR 160.
- The seven implementation/documentation files listed below are uncommitted.
  There is no candidate commit or CI run.
- Implemented and focused-locally validated; not committed, pushed, integrated,
  deployed, activated, complete, or merge-ready.
- No dependency installation, cluster submission, registry write, Zarr write,
  selector change, historical mutation, or shared-checkout update occurred.

## Contract and implementation

Classification: **enforcement correction**. Scientific formulas and defaults,
position/body-frame semantics, coordinates, row/frame identity, tracking and
motion manifests, persisted digest bytes, runtime selection, producer commands,
and authority activation are unchanged.

The stage catalog now declares the already-enforced runtime dependency in both
directions used by registry surfaces:

- `track_kinematics.depends_on = (refined_keypoints, tracks)`; and
- `tracks.invalidates = (track_kinematics,)`.

Intentional behavior change: after a new `tracks` run is registered, runtime
cascade invalidation marks `track_kinematics` and all catalog-declared motion
descendants missing. The generic Palette plan/status view also reports `tracks`
as a direct blocker for missing track kinematics. This does not change the
analysis-workflow graph or PR 160's exact `--tracking-run` execution behavior.

`docs/analysis_workflow_dag.md` now distinguishes catalog granularity from the
packaged profile's composite `tracks` adapter. It also records the intended
provider boundary: tracking should eventually consume an exact typed
`subject_position` publication, while detection, keypoint, and subject-mask
measurements remain peer producers under named estimator policies. Keypoints
may independently provide body-frame/heading authority.

The documentation deliberately does not call refined keypoints the production
position default. The current offline compatibility lane uses crop/detection-
centroid position plus keypoint heading; replacing it with a keypoint-triad
position is a scientific change requiring an explicit recipe and preservation
evidence.

Dirty scope:

- `src/fisheye/registry/stage_catalog.py`;
- `tests/unit/fisheye/test_stage_catalog.py`;
- `tests/unit/fisheye/test_step_cascade.py`;
- `tests/unit/fisheye/test_palette_cli.py`; and
- `docs/analysis_workflow_dag.md`;
- `docs/diagnostics/authority_consolidation_work_queue_2026-08-25.md`; and
- this handoff document.

No new stage ID, position selector, admission schema, manifest grammar,
compatibility adapter, or executable provider path was introduced. The broader
`RES-TRACK-001` position-provider migration remains open.

## Validation evidence

Tests were added before the source edit. Against the unchanged base, the focused
suite produced the four expected failures: the catalog lacked the dependency,
`tracks` was a cascade leaf, a tracking-triggered cascade did not write motion
status, and Palette planning reported motion as merely missing rather than
blocked by tracks. The other 44 tests passed.

After the source correction, the first workstation suite passed **48 tests**:

```bash
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_stage_catalog.py \
  tests/unit/fisheye/test_step_cascade.py \
  tests/unit/fisheye/test_palette_cli.py -q
```

The expanded workflow/finalizer suite passed **86 tests**:

```bash
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_analysis_workflow_dag.py \
  tests/unit/fisheye/test_analysis_workflow_execution.py \
  tests/unit/fisheye/test_stage_catalog_drift.py \
  tests/unit/fisheye/test_derived_analysis_registry_finalize.py -q
```

All pytest ran outside the Codex sandbox through repository `scripts/py`. The
expanded run emitted eight expected Zarr-v3 consolidated-metadata warnings; no
test failed. Across the two post-fix commands, **134 distinct tests passed**.

`scripts/py -m py_compile src/fisheye/registry/stage_catalog.py` and
`git diff --check` also passed. No production or canary data was required for
this metadata-only enforcement correction.

## Remaining validation and integration

Every required new-change CI check is unrun: `generated artifacts`, `import
boundaries`, `file-size ratchet`, `zarr open metadata modes`, `observed metadata
literals`, `active contract freshness`, `package and collection`, `non-gpu
tests (shard 0)` through `non-gpu tests (shard 15)`, and `ci-required`.

No commit, push, draft PR, CI invocation, merge, deployment, or activation is
authorized by this checkpoint. A future exact candidate commit must pass all
required CI before it may be described as complete or merge-ready.

Separately remaining under `RES-TRACK-001`: define production selection and
admission for the profile-neutral position surface, preserve the current
crop/detection-centroid plus keypoint-heading recipe as an explicit
compatibility branch, integrate at least one provider-aware workflow path, and
prove real producer-to-tracking-to-motion reload behavior without weakening the
existing selector-ineligible canary contracts.
