# Detection candidate and publication contract
<!-- contract-meta
version: 1
status: active
implementation: implemented
last_verified: 2026-08-13
stage_arrays_spec: DETECT_SPEC
-->

Canonical detection has one construction boundary and one production
artifact-to-canonical publication pipeline. The numerical writer never chooses
or mutates a live canonical run.

This is the authoritative detection-publication contract. For stage ordering,
see [recording_analysis_pipeline_contract.md](recording_analysis_pipeline_contract.md).

## Construction boundary

`fisheye.shared.detection_candidate.build_detection_candidate` is the shared
entry point for creating a complete disposable detection candidate. It invokes
the low-level YOLO implementation but does not promote selectors or update the
registry.

The low-level writer refuses an existing canonical analysis Zarr before any
model loading or output mutation. A full-recording candidate may write only to
a node-local overlay carrying
`palette_detection_candidate_build_authority`. Detached clipped artifacts use
the explicitly unbound artifact coordinate contract. Brand-new standalone
outputs remain useful for development and compute smokes, but they are not
canonical publications and must not be presented as a recording's selected
detect run.

## Historical direct full-recording publication

`fisheye.utils.run_detection_local_publish`:

1. verifies the registered model SHA-256;
2. resolves and streams the canonical source video from its acquisition
   locator;
3. copies only the verified acquisition authority into node-local scratch;
4. builds and validates the completed candidate there;
5. atomically copies the run group into `detect_runs` while it remains selector
   ineligible;
6. reopens and proves the published coordinate/lineage contract;
7. updates the direct `latest` and `latest_complete` selectors and marks the
   completed run selector eligible;
8. consolidates the archive root as the final published visibility step;
9. proves the selected run's complete direct/consolidated subtree equivalence
   and exact selector agreement; and
10. emits successful registry bookkeeping only after reopening the selected
    run through consolidated metadata.

Failure before activation leaves the previous canonical selectors and registry
success state unchanged. Consolidation or visibility-validation failure after
the direct selector write restores the prior selectors, retains the attempted
run as an immutable failed and selector-ineligible tombstone, and reconsolidates
that fail-closed state. Failed local scratch is retained only when explicitly
requested.

This direct adapter is retained as an explicit compatibility surface. New
production recipes must not route through it because its detector-local layout
is published directly below `detect_runs` before canonical assembly.

## Production artifact and canonical publication

`fisheye.utils.run_detection_artifact` uses the same candidate builder but
packages an unbound run-group artifact below `detection_artifact_runs`. The
native binder validates all work-unit packages, binds canonical recording/frame
identity, and publishes the sole production authority below `detect_runs`.
Whole videos use one identity-mapped work unit; clipped recordings use one or
more indexed work units.

The complete accepted boundary and implementation checklist are in
[detection_artifact_and_canonical_publication_boundary.md](detection_artifact_and_canonical_publication_boundary.md).

## Removed paths

The direct canonical batch call, the direct single-recording pipeline call, and
the `submit_detect_quality_refine_bsub.sh` latest-based convenience chain were
retired. The supported combined workflow is
`submit_detect_artifact_quality_refine_bsub.sh`; ordinary full-recording arrays
use `submit_detect_batches_bsub.sh` and node-local atomic publication.

The completed implementation brief and former batch-agent work plan are kept
under `docs/archive/`; neither is an operator runbook.
