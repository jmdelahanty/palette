# Docs Pass: Storage Relocation, Review Proxies, Registry Dedupe (2026-05-28)

## Scope

This pass reviewed the active docs most likely to drift after the sleepyfish
PRFS migration and review-proxy work:

- `docs/recording_store_relocation_components.md`
- `docs/detection_review_web_todo.md`
- `docs/cluster_run_group_artifact_workflow.md`
- `docs/cluster_batching_guide.md`
- `docs/clipped_training_zarr_implementation_checklist.md`
- `docs/cluster_pipeline_migration_checklist.md`
- `docs/registry_repair_playbook.md`
- the 2026-05-20 diagnostics shard reports

It also ran targeted grep checks over `docs/`, `scripts/`, and
`src/fisheye/utils/` for review-proxy command names, old proxy-sharding claims,
and sleepyfish `/nvme1` or `palette_smoke` references.

## Current State Confirmed

- Sleepyfish cams `2010093` through `2010096` have recording roots under
  `/groups/johnson/johnsonlab/jeremy/recordings`.
- Each relocated root has 22 rolling clip directories.
- Registry sleepyfish recording paths now point at the PRFS recording roots.
- Registry duplicate clipped-training dataset rows for cams `2010094`,
  `2010095`, and `2010096` were removed.
- Cam `2010093` keeps the old palette-smoke analysis dataset id and has a
  separate suffixed clipped-training dataset id:
  `sleepyfish_2026_05_05_17_45_30_cam2010093:z3fdd176a8abc`.
- Registry validation after dedupe was clean: no old suffixed id refs,
  `PRAGMA foreign_key_check` clean, and `PRAGMA quick_check` returned `ok`.
- All four relocated sleepyfish roots now have contained review-proxy manifests
  with 22 clips and no missing proxy-video paths.

## Corrections Applied

### `recording_store_relocation_components.md`

- Bumped `last_verified` to 2026-05-28.
- Updated the review-proxy relocation policy to prefer
  `scripts/submit_review_proxy_videos_sharded_bsub.sh` for long clipped
  recordings.
- Replaced the one-camera sleepyfish smoke example with the current four-camera
  migration result.
- Documented the cam `2010093` analysis/training dataset-id split.
- Documented duplicate-row cleanup policy used for cams `2010094`-`2010096`.
- Documented contained review-proxy status for all four sleepyfish roots.

### `detection_review_web_todo.md`

- Added a document-status section that routes clipped collection, proxy-video,
  promotion, and relocation policy to narrower docs.
- Left implementation status for the browser reviewer in place.
- Replaced the long proxy-video contract section with a short pointer to
  `docs/review_proxy_video_contract.md`.

### `review_proxy_video_contract.md`

- Added a dedicated proxy-video contract with manifest shape, source/proxy
  coordinate policy, builder commands, sharded LSF workflow, timing notes, and
  validation gates.
- Updated the manifest example to use the durable
  `/groups/johnson/johnsonlab/jeremy/recordings` root instead of the older
  `palette_smoke` example path.
- Removed stale wording that implied the supported cluster proxy wrapper was
  still intentionally one sequential job only.

## Findings

### No Severe Contradictions Found

The high-use docs now agree with the code and current operator workflow on:

- review-proxy builder module:
  `fisheye.utils.build_review_proxy_videos`;
- sharded proxy submitter:
  `fisheye.utils.submit_review_proxy_videos_sharded_bsub` and
  `scripts/submit_review_proxy_videos_sharded_bsub.sh`;
- sharded defaults:
  `--encoder h264_nvenc`, `--hwaccel cuda`, `--scale-flags bilinear`;
- dry-run by default for direct proxy builds;
- final manifest ownership by the finalizer, not the shard jobs.

### Still Intentionally Historical Or Example Paths

Many `/nvme1/recordings` and `palette_smoke` references remain in dated
benchmark reports, historical smoke logs, or local-workstation examples. They
are not all stale. The active relocation policy now distinguishes:

- active location pointers, which should be rewritten during migration;
- historical provenance and dated benchmark evidence, which should not be
  silently rewritten.

### `cluster_run_group_artifact_workflow.md`

Classification: current.

The review-proxy section already documents the sharded workflow and finalizer
contract correctly. It still contains old cam `2010093` `palette_smoke` smoke
examples in earlier sections. Those are dated evidence for the original smoke,
not the current canonical storage-root contract.

### `cluster_batching_guide.md`

Classification: current.

The clipped detect/refine smoke notes remain accurate as historical throughput
evidence. The guide does not yet mention the later cam `2010095` fixed PynvVC
collection or the four-camera PRFS migration, but that is not required for its
batching-policy purpose.

### `clipped_training_zarr_implementation_checklist.md`

Classification: current enough.

The checklist already points relocation semantics at
`recording_store_relocation_components.md`, and it correctly records the full
sleepyfish clipped-training Zarr creation and detection-label promotion state.
It does not need to duplicate the later PRFS registry/proxy migration details.

### `cluster_pipeline_migration_checklist.md`

Classification: partially stale but non-severe.

It still says broad core reader/editor support remains in migration. That is
acceptable if "core readers" means every generic reader. It is less precise now
because selected readers already support clipped finalized collections:

- Palette `video_detect_review_web` can inspect clipped finalized collections;
- Crimson has a clipped collection resolver path;
- the browser review path works best with review proxies.

Recommended future edit: add a dated note under the clipped-reader section that
specific viewers are implemented while broad stage-reader generalization remains
open.

### `registry_repair_playbook.md`

Classification: current.

The dedupe section correctly says `fisheye.registry.dedupe` is read-only and
that conflicts need an explicit merge policy before apply-mode cleanup. The
sleepyfish cleanup followed that policy manually:

- delete duplicate current rows where canonical rows already existed;
- preserve `recording_step_status_history` by moving events to the canonical
  dataset id;
- delete duplicate dataset rows only after confirming no old-id references
  remained.

## Residual Documentation Work

1. Rename or split `detection_review_web_todo.md`.
   It has become a status/contract document for two reviewers plus proxy videos
   and promotion. The 2026-05-20 detection audit already flagged this. A first
   split was completed by moving clipped collection, proxy-video, and relocation
   semantics into narrower companion docs and cross-links, but the file name
   still reflects its historical TODO origin.

2. Add a dedicated clipped finalized-detect resolver contract.
   Multiple docs reference finalized collections and
   `recording_frame_index.parquet`, but the resolver contract is spread across
   cluster, Crimson, and review-web docs.
   Addressed by `docs/clipped_finalized_detect_collection_contract.md`.

3. Add an operator runbook for registry-backed storage-root relocation.
   `recording_store_relocation_components.md` defines the policy; this pass
   found that the actual backup/copy/rewrite/verify sequence needed a separate
   operator sequence rather than more design prose.
   Addressed by `docs/recording_store_relocation_runbook.md`.

4. Add an apply-capable registry dedupe/merge tool only if this repeats.
   Current `fisheye.registry.dedupe` is intentionally dry-run. The manual
   sleepyfish dedupe was safe, but repeated storage migrations would benefit
   from a narrow apply tool with explicit conflict policies.

## Bottom Line

Docs remain broadly healthy. The two concrete stale claims from the latest
work were updated in place. Remaining issues are organizational rather than
severe contradictions: proxy/review docs have grown beyond TODO shape. The
highest-value split points now have companion docs, so a future cleanup can
rename/archive the historical TODO without losing contract content.
