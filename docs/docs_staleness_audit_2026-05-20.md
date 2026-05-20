# Documentation Staleness Audit: 2026-05-20

## Scope

This pass reviewed active Palette docs around:

- analysis-to-training detection promotion;
- browser detection review and online save behavior;
- clipped training Zarr creation and promotion;
- clipped recording frame-map consumers;
- cluster clipped detect/refine workflow status.

## Corrections Applied

- `docs/analysis_to_training_promotion_contract.md`
  - Updated the current position from "video-backed analysis review is future
    work" to the current split between `detect_review_web` and
    `video_detect_review_web`.
  - Recorded that online detection promotion is implemented for traditional and
    clipped analysis saves when the web server is launched with
    `--edit --promote-training-zarr`.
  - Recorded that batch saves group analysis writes and call promotion once per
    save batch.

- `docs/detection_review_web_todo.md`
  - Replaced the stale statement that the web reviewer is not a general
    analysis-video viewer.
  - Clarified that `detect_review_web` is the materialized-image reviewer while
    `video_detect_review_web` is the analysis-video detection reviewer.
  - Clarified that review proxy MP4s are the preferred browser path for long
    clipped recordings.

- `docs/clipped_training_zarr_implementation_checklist.md`
  - Marked detection bbox promotion from clipped analysis finalized collections
    as implemented.
  - Marked finalized detect/refined-detect clip collections as implemented.
  - Narrowed the deferred label-import item to keypoints, masks, and
    multi-instance labels.

## Remaining Non-Severe Gaps

- Crimson save-hook promotion is still documented as not wired. That matches the
  current Palette state.
- Unified/exported training artifact integration for promoted per-recording
  rows is still documented as deferred. That matches the current state.
- Clipped keypoint/mask promotion is still deferred; current promotion is
  detection-bbox only.
- Some cluster docs still describe broad "core reader/editor support" as in
  migration. This remains acceptable because only selected readers/viewers have
  clipped finalized-collection support today.

## Severity

No severe current contradiction remains in the reviewed Palette docs after this
pass. The main stale issue was wording that predated `video_detect_review_web`
and online batch promotion.
