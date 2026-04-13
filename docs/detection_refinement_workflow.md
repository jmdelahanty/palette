# Detection Refinement Workflow

This is the current detect workflow as of 2026-04-07.

## Contract

- `detect_runs/<run>` is raw detector output.
- `detect_runs/<run>/quality_reports/<qrun>` is the raw detect artifact-label
  surface used by refine guardrails.
- `refined_detect_runs/<run>` is the canonical curated detect surface.
- Shared reads and writes should use `refined_detect_runs/<run>/instances`.
- `refined_detect_runs/<run>/source_detections` is the candidate-audit surface
  for the bound raw detect run.
- New work should not treat `filtered/` or `interpolated/` as the primary
  curated surface.

## Recommended Sequence

1. Run detection.
   - Blob:
     ```bash
     scripts/py -m fisheye.detection.detect_traditional /path/to/zarr
     ```
   - YOLO:
     ```bash
     scripts/py -m fisheye.detection.detect_yolo /path/to/zarr --model /path/to/model.pt
     ```
2. Run detect quality.
   ```bash
   scripts/py -m fisheye.refinement.detect_quality /path/to/zarr
   ```
   This labels raw-detect artifacts (`quality_flags`,
   `detection_quality_labels`) for the selected detect run. It is distinct from
   later refined-detect review approval.
3. Initialize the curated refined run.
   ```bash
   scripts/py -m fisheye.refinement.refine_detect /path/to/zarr
   ```
   This writes sparse `instances/` and `source_detections/` on
   `refined_detect_runs/<latest>`.
   Interpolation is no longer part of the normal detect-refinement workflow.
4. Edit the curated refined surface if needed.
   ```bash
   scripts/py -m fisheye.tune.detect_review /path/to/zarr
   ```
   Manual review now has two refined modes:
   - one slot per frame for the legacy single-subject workflow
   - one slot per `(frame, arena_id)` when fixed sub-arena ROI definitions are
     available from subdish masks or arena-assignment metadata

   `detect_review` still does not support unconstrained multi-instance editing
   within a single arena/ROI. Legacy sparse manual subgroups may still appear
   when operating on old archives.
5. Approve the refined detect run.
   ```bash
   scripts/py -m fisheye.utils.accept_detect_review /path/to/zarr --state approved --intended-use training --reviewer <name>
   ```
   `resolved_group` should normally be `refined` for current runs.
6. Build crops from the curated refined surface.
   ```bash
   scripts/py -m fisheye.tracking.crop /path/to/zarr --source-type refined
   ```
   `auto` also resolves to the active curated refined surface when it exists.

## Status Fields

The slot-based edit vocabulary is:

- `status_codes`: `present`, `missing`, `filtered_out`, `ambiguous`
- `source_kind_codes`: `none`, `raw_detect`, `interpolated`, `manual`
  - `interpolated` is retained for legacy compatibility/provenance and is not
    the normal outcome for current sparse refined runs.
- `manual_edit_flags`: sticky per-slot marker for manual correction/clear/retune
- `reason`: explanatory tags only

The sparse refined surfaces carry the consumer-facing state:

- `instances/`: accepted curated detections
- `source_detections/decision_codes`: raw-candidate decisions such as
  `accepted`, `filtered`, `duplicate`, and `manual_clear`

`instances/` may now contain multiple curated rows for the same frame.

## Legacy Compatibility

Older archives may still contain:

- `refined_detect_runs/<run>/filtered`
- `refined_detect_runs/<run>/interpolated`
- `refined_detect_runs/<run>/<manual_group>`

Those sparse groups are now compatibility/provenance inputs. They are not the
primary curated contract for new runs.

## Diagnostics

- Summarize recording status:
  ```bash
  scripts/py -m fisheye.utils.check_recording_steps /path/to/recordings --recursive
  ```
- Inspect crop source linkage:
  ```bash
  scripts/py -m fisheye.diagnostics.check_crop_sources /path/to/zarr
  ```
- Inspect current refined detect state:
  ```bash
  scripts/py -m fisheye.visualization.detection_visualizer /path/to/zarr --refined-variant refined
  ```
- Inspect current curated refined detect state and source decisions:
  ```bash
  scripts/py -m fisheye.utils.inspect_refined_detect_run /path/to/zarr
  ```
