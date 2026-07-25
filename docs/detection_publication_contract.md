# Detection candidate and publication contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-07-24
stage_arrays_spec: DETECT_SPEC
-->

Canonical detection has one construction boundary and two publication
adapters. The numerical writer never chooses or mutates a live canonical run.

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

## Full-recording publication

`fisheye.utils.run_detection_local_publish`:

1. verifies the registered model SHA-256;
2. resolves and streams the canonical source video from its acquisition
   locator;
3. copies only the verified acquisition authority into node-local scratch;
4. builds and validates the completed candidate there;
5. atomically copies the run group into `detect_runs` while it remains selector
   ineligible;
6. reopens and proves the published coordinate/lineage contract;
7. activates `latest` and `latest_complete`; and
8. emits successful registry bookkeeping only after activation.

Failure before activation leaves the previous canonical selectors and registry
success state unchanged. Failed local scratch is retained only when explicitly
requested.

`run_detect_with_registry_model`, `run_detections_batch`, `palette detect`, and
the recording pipeline all route through this adapter. Canonical execution
requires a registered model path, digest, run ID, and set ID. Detection cannot
create or overwrite acquisition metadata; importing the recording is a
prerequisite.

## Clipped and transport publication

`fisheye.utils.run_detection_artifact` uses the same candidate builder but
packages an unbound run-group artifact. The clip-aware importer binds canonical
recording/frame identity, validates the package, and publishes it. This remains
the appropriate adapter for clipped fan-out and explicit transport boundaries.

## Removed paths

The direct canonical batch call, the direct single-recording pipeline call, and
the `submit_detect_quality_refine_bsub.sh` latest-based convenience chain were
retired. The supported combined workflow is
`submit_detect_artifact_quality_refine_bsub.sh`; ordinary full-recording arrays
use `submit_detect_batches_bsub.sh` and node-local atomic publication.

The completed implementation brief and former batch-agent work plan are kept
under `docs/archive/`; neither is an operator runbook.
