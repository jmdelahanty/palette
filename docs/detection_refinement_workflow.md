# Detection Refinement Workflow (Blob + YOLO)

This note documents the end-to-end detection refinement workflow we use in Palette.
It is meant as a practical runbook for new agents and future batch runs.

Key principles:
- **Append-only provenance**: never overwrite raw detection runs.
- **Refined runs are immutable**: new params => new refined run.
- **Manual corrections are separate** (detection only): stored under a manual subgroup.

Pipeline context (recording analysis archives):
- canonical stage order is `import -> detect -> refine -> register`
- use `fisheye.utils.run_recording_analysis_pipeline` or
  `fisheye.utils.import_recordings_analysis` when you want orchestration
- contract reference: `docs/recording_analysis_pipeline_contract.md`

## Terminology (Zarr layout)

- `detect_runs/<run>`: raw detections (blob or YOLO).
- `refined_detect_runs/<run>`: refined detections (filtered/interpolated).
- `refined_detect_runs/<run>/<manual_group>`: manual/retune corrections.
- `manual_review_latest`: pointer on refined run to the current manual group.
- `retune_id` + `retune_params`: per-detection retune label and mapping (when retune is used).
- `retune_base_group`: source subgroup used as the retune baseline (for incremental retunes).
- `detect_review_status`: review metadata on the refined run (approval + resolved group).

## Recommended sequence (per recording)

1) **Run detection**
   - Blob:
     ```bash
     python -m fisheye.detection.detect_traditional /path/to/zarr
     ```
   - YOLO:
     ```bash
     python -m fisheye.detection.detect_yolo /path/to/zarr --model /path/to/model.pt
     ```

2) **Run detection quality (required before refinement in production)**
   ```bash
   python -m fisheye.refinement.detect_quality /path/to/zarr
   ```
   Defaults:
   - `--threshold-mode scaled` with `--threshold 100` and `--threshold-reference-width 640`
   - This means "100 px at 640px width", scaled automatically for other resolutions.

   Why:
   - writes frame-level and detection-level quality labels used by refinement
   - avoids fallback behavior where refinement assumes all detections are clean

   Batch option:
   ```bash
   scripts/py -m fisheye.utils.detect_quality_batch /nvme1/recordings --recursive --apply
   ```

3) **Refine detections**
   ```bash
   python -m fisheye.refinement.refine_detect /path/to/zarr
   ```
   Notes:
   - If the video is a **sampled training import**, refinement auto-disables
     filters and interpolation (passthrough mode).
   - This keeps refinement safe for large frame gaps.

4) **Retune missing detections (blob only)**
   ```bash
   python -m fisheye.tune.detect_review /path/to/zarr --retune
   ```
   - Writes a manual/retune subgroup under the latest refined run and sets
     `manual_review_latest`.
   - Records `retune_id` and `retune_params`.
   - If a manual subgroup already exists, retune is **incremental** and
     uses that subgroup as the base (recorded in `retune_base_group`).
   - Use `--overwrite` to update an existing manual subgroup.

5) **Approve refined detections (review status)**
   - In the review UI, press `a` to approve and record `detect_review_status`
     on the refined run (and `detect_review_status_latest` on the parent group).
   - Crop runs created with `--crop-source preferred`/`auto` resolve the group
     using this review status and store a snapshot on the crop run.
   - For Crimson-driven acceptance, follow:
     - `docs/crimson_detect_review_acceptance_contract.md`

6) **Manual corrections**
   ```bash
   python -m fisheye.tune.detect_review /path/to/zarr
   ```
   - Writes `refined_detect_runs/<latest>/<manual_group>` (default: `manual`).
   - Leaves raw detections untouched.

7) **Downstream stages**
   - Crop and later stages will prefer the **manual subgroup** when present.
   - Otherwise they use `interpolated` (or `filtered`) from the refined run.
   - `--crop-source preferred`/`auto` uses `detect_review_status` plus a policy
     chain; the chosen policy is stored in `detection_preferred_policy`.

## YOLO-specific guidance

Retuning is **not supported** for YOLO runs. If thresholds or models change:

1) Run a **new YOLO detect run** (new `detect_runs/<run>`).
2) Run `detect_quality` for that detect run.
3) Refine that run.
4) Manual review if needed.

Manual review still applies to YOLO; it simply writes a corrected subgroup under
the refined run.

## Detect-Quality Guardrails

- Run `detect_quality` after every new detect run and before `refine_detect`.
- If `quality_reports/latest` is missing for the selected detect run, do not
  treat refinement as production-grade.
- Treat quality as stale when the selected detect run differs from the run used
  to produce the current refined run (`source_detect_run` mismatch).
- Record and preserve threshold provenance (`jump_threshold`,
  `blip_gap_threshold`) from the quality run attrs.
- In automation, classify this as its own stage failure (`detect_quality`) so
  operators can retry quality without rerunning detect.

## Versioning rules (important)

- **Never overwrite** a `detect_runs/<run>`.
- If parameters change, **create a new refined run**.
- Manual corrections should go into a new subgroup (or overwrite a manual subgroup
  intentionally with `--overwrite`).

## Diagnostics (recommended)

- Verify crops reference refined detections:
  ```bash
  python -m fisheye.diagnostics.check_refined_roi_links /path/to/zarr
  ```
- Summarize pipeline status:
  ```bash
  python src/fisheye/utils/check_recording_steps.py /path/to/recordings --recursive
  ```

## Notes on sampled training imports

If `import_mode=sampled` (large frame gaps):
- Refinement will be **passthrough** (no interpolation).
- Quality metrics that depend on temporal continuity are not meaningful.
- Manual review still works; use it for missing detections only.
- Coverage percent is computed over the sampled frame universe (not the
  original full timeline), so 100% means “all sampled frames have detections.”

## Status reporting conventions (refined detect)

Refined runs always use the same on-disk layout (`filtered/` + `interpolated`)
to keep downstream tooling simple, even when refinement is a no-op. To reduce
confusion, the status reporter uses explicit labels:

- **passthrough**: Training/sample imports where refinement is intentionally
  disabled (no filtering or interpolation). The refined groups mirror the
  original detections, but the metadata records `refine_mode=passthrough`.
- **unchanged**: Standard refinement ran, but it removed 0 detections and
  added 0 interpolations (i.e., the refined data is identical to the source).
- **filtered/interpolated**: Refinement meaningfully changed the data.

Tradeoff: we keep a stable schema (always present `filtered/` and `interpolated`)
to avoid breaking downstream consumers, and rely on explicit labels to indicate
when refinement was a no-op. This keeps both auditability and operational
simplicity without overloading the term "interpolated."
