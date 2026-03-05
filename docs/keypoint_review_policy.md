# Keypoint Review Policy and Missing-Keypoint Frames

This document summarizes the current keypoint review workflow and how we handle
frames where a fish is present but no keypoints can be resolved.

## Current workflow (what exists today)
- **Refined runs are the editable copy.**
  - Manual edits and retunes write only to `refined_keypoints_runs/<run>`.
- **Reason tags are stored per ROI.**
  - Primary encoding: `reason_bytes` (`uint8[N,width]`, null-terminated UTF-8).
  - Secondary/fallback: `reason` string array.
  - Final fallback: derive labels from `detection_source` (`0=clean`, `1=interpolated`).
  - Refined runs record `reason_fallback_order=["reason_bytes","reason","detection_source"]`.
  - Common tags:
    - `manual_correction`
    - `fish_present_no_keypoints`
    - `detection_issue`
    - `geometry_issue`
- **Quality arrays used downstream:**
  - `usable_keypoints` (bool)
  - `refined_success` (bool)
  - `confidence_valid`, `geometry_valid`, etc.
- **Review status metadata:**
  - `refined_keypoints_runs/<run>.attrs["keypoint_review_status"]`
    includes `state`, `method`, `intended_use`, `timestamp`, `reviewer`, `notes`.
  - Parent attr: `refined_keypoints_runs.attrs["keypoint_review_status_latest"]`
- **Manual review UI:**
  - `--manual` reviews keypoints and writes corrections.
  - `--all` iterates all ROIs (otherwise only failures).
  - Reason columns are synchronized on write (`reason_bytes` + `reason`).
  - Hotkeys:
    - `x`: mark `fish_present_no_keypoints`
    - `d`: mark `detection_issue` (and flag for detection review)

## How to interpret “fish present but no keypoints”
This condition is **not a clean negative example**. It means the instance is
present but unlabeled/uncertain. We keep these frames in the refined run for
provenance, but they should be **excluded by default** when exporting training
data unless the training format explicitly supports visibility flags.

Recommended default:
- **Training export**: exclude ROIs tagged `fish_present_no_keypoints`.
- **Full-recording analysis**: keep them, but treat keypoints as missing.

## Optional visibility-flag training (when supported)
Some keypoint training formats (e.g., COCO-style) allow per-keypoint visibility
flags (`v=0/1/2`). If your exporter and trainer support this:
- You may **include** these frames **only if** you emit an instance (bbox) and
  mark all keypoints as not visible.
- This keeps the instance in the dataset without teaching false negatives.

If visibility flags are **not** supported by your training pipeline, keep the
default behavior (exclude `fish_present_no_keypoints`).

## Notes / follow-ups
- Current merged export supports explicit inclusion via
  `--row-gate-policy raw_success_plus_box_only` (box-only rows with `visibility=0`).
- Approval decisions should consider the proportion of
  `fish_present_no_keypoints` frames when targeting training use.

## Related Contract
- Late correction + ROI flagging + downstream stale marking:
  `docs/keypoint_late_correction_contract.md`

## Related Incident
- Refined-keypoint coordinate-space mismatch investigation and recovery:
  `docs/keypoint_refined_coordinate_space_incident_2026-03-04.md`
