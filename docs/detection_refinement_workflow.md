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

   Export/view saved detect quality visualization artifacts:
   ```bash
   scripts/py -m fisheye.utils.export_detect_quality_overview /nvme1/recordings --recursive --zarr-use training --artifact detect_quality_overview_png --view
   scripts/py -m fisheye.utils.export_detect_quality_overview /nvme1/recordings --recursive --zarr-use training --artifact refinement_pipeline_overview_png --view
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

8) **Finalize detect/refinement visualization artifacts (approved runs)**
   ```bash
   scripts/py -m fisheye.utils.finalize_refinement_artifacts /nvme1/recordings --recursive --zarr-use training --required-intended-use training --apply
   ```
   - Writes both artifacts under each eligible refined run:
     - `visualizations/detect_quality_overview_png`
     - `visualizations/refinement_pipeline_overview_png`
   - `refinement_pipeline_overview_png` includes the manual subgroup when present.

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
- Finalize and inspect approved refinement artifacts:
  ```bash
  scripts/py -m fisheye.utils.finalize_refinement_artifacts /path/to/recordings --recursive --zarr-use training --required-intended-use training --apply
  scripts/py -m fisheye.utils.export_detect_quality_overview /path/to/recordings --recursive --zarr-use training --artifact detect_quality_overview_png --view
  scripts/py -m fisheye.utils.export_detect_quality_overview /path/to/recordings --recursive --zarr-use training --artifact refinement_pipeline_overview_png --view
  ```

## Detection Profile Registry Runbook (2026-02-24)

This is the operational sequence used to expose detection-profile summaries via
registry query views (without ad-hoc `python -c` snippets).

### 1) Populate/refresh dataset rows in registry

```bash
scripts/py -m fisheye.utils.registry_rescan /nvme1/recordings --recursive --registry /nvme1/registry.sqlite
```

### 2) Backfill profile summaries into each training Zarr

```bash
scripts/py -m fisheye.utils.backfill_detection_profiles /nvme1/recordings --recursive --zarr-use training --registry /nvme1/registry.sqlite --apply
```

Observed on 2026-02-24:
- `zarr_scanned=105`
- `filtered_zarr_use=53`
- `updated=52`
- `errors=0`

### 3) Sync latest profile runs into registry table

```bash
scripts/py -m fisheye.utils.sync_detection_profile_registry --registry /nvme1/registry.sqlite --zarr-use training --apply
```

Observed on 2026-02-24 after full rescan:
- `datasets=52`
- `updated=51`
- `missing_profile=1`

### 4) Recover missing-profile edge case

When one dataset reported `missing_profile` but already had
`analysis/detection_profile_runs` on disk:

- dataset id:
  `2026-01-28T19-22-28Z_arena_1:zc66de17bea1b`
- targeted backfill returned:
  `A group exists ... at path 'analysis/detection_profile_runs'`

Root cause:
- an intermittent Zarr group lookup failure (`group.get(...)`/`group[...]`) could
  return "missing" for an existing group in this store layout.

Fix implemented:
- detection-profile writer now falls back to reopening child groups by
  `store + path` before creating groups.
- registry sync reader uses the same fallback when resolving:
  - `analysis`
  - `analysis/detection_profile_runs`
  - selected profile run group.

Targeted recovery command:

```bash
scripts/py -m fisheye.utils.sync_detection_profile_registry --registry /nvme1/registry.sqlite --dataset-id 2026-01-28T19-22-28Z_arena_1:zc66de17bea1b --apply
```

### 5) Query/verify the operator-facing view

Count of training recording-latest rows:

```bash
scripts/py -m fisheye.utils.registry_query --registry /nvme1/registry.sqlite --recording-detection-data-profile-latest --profile-zarr-use training --json | jq 'length'
```

Observed on 2026-02-24: `52`.

Spot-check a compact row payload:

```bash
scripts/py -m fisheye.utils.registry_query \
  --registry /nvme1/registry.sqlite \
  --detection-data-profile-latest \
  --profile-dataset-id 2026-01-28T21-47-47Z_arena_1:z36ae7c3bf7e1 \
  --json \
| jq '.[0] | {dataset_id,profile_run,recording_id,zarr_use,detection_type,coverage_percent,detections_total,detection_path}'
```

### Dataset vs recording semantics (for query mode choice)

- `dataset_id` identifies a specific Zarr dataset instance (recording + stable suffix).
- `recording_id` identifies the recording event across dataset variants.
- `--detection-data-profile-latest` returns latest profile per dataset.
- `--recording-detection-data-profile-latest` returns latest profile per recording.

### Subject-lineage query note (dish vs genotype vs DPF)

- `dish_design` and `genotype` are distinct fields.
  - `dish_design`: capture context (for example `cedar`, `alpine`)
  - `genotype`: subject lineage (for example `Tg(elavl3:gcamp7f)`)
- `dpf_at_acquisition` is queried via `--dpf`, `--dpf-min`, `--dpf-max`.

Example:

```bash
scripts/py -m fisheye.utils.registry_query \
  --registry /nvme1/registry.sqlite \
  --dish-design cedar \
  --genotype 'Tg(elavl3:gcamp7f)' \
  --dpf-min 6 \
  --dpf-max 8 \
  --json
```

### Subject-lineage projection refresh (one-time after schema/code update)

If `genotype` / `dpf_at_acquisition` projection fields were added after profile
rows already existed, run a one-time sync refresh to populate existing registry
rows:

```bash
scripts/py -m fisheye.utils.sync_detection_profile_registry --registry /nvme1/registry.sqlite --zarr-use any --apply
```

Notes:
- This refresh is required for existing `detection_data_profile` rows.
- Rewriting Zarr profile runs is optional for registry projection because sync
  can fall back to registry provenance when lineage fields are absent in older
  profile summaries.
- If you also want lineage fields embedded in each profile run payload, rerun:
  `backfill_detection_profiles --apply`, then rerun sync.

### Subject-lineage aggregate validation (pre/post aggregation)

Use this when preparing a training data card aggregation run.

Set shared paths:

```bash
REGISTRY=/nvme1/registry.sqlite
MANIFEST=/nvme1/training/datasets/<set_id>/<set_id>.manifest.json
```

Pre-aggregation checks:

1) Keep dish/capture and lineage checks separate:

```bash
scripts/py -m fisheye.utils.registry_query --registry "$REGISTRY" --zarr-use training --dish-design cedar --json | jq 'length'
scripts/py -m fisheye.utils.registry_query --registry "$REGISTRY" --zarr-use training --genotype 'Tg(elavl3:gcamp7f)' --json | jq 'length'
scripts/py -m fisheye.utils.registry_query --registry "$REGISTRY" --zarr-use training --dpf-min 6 --dpf-max 8 --json | jq 'length'
```

2) Gate manifest lineage coverage before aggregation:

```bash
scripts/py -m fisheye.utils.aggregate_detection_training_data_card \
  --manifest "$MANIFEST" \
  --registry "$REGISTRY" \
  --subject-lineage-policy require \
  --dry-run
```

Expected pre-aggregation output:
- `Subject lineage coverage: <n>/<n> datasets`
- no `Subject lineage missing dataset_id(s): ...` line

Expected checks before/after aggregation:
- before apply: dry-run with `--subject-lineage-policy require` succeeds.
- after apply: `selection.dataset_count` in the card matches manifest dataset count.
- after subject aggregates are enabled in card payload:
  - `subject_coverage` manifest dataset count equals `selection.dataset_count`
  - `subject_coverage` lineage-covered dataset count equals manifest dataset count
  - `sum(genotype_counts.*)` equals lineage-covered dataset count
  - `dpf_stats.count` equals lineage-covered dataset count
  - `sum(dpf_histogram.counts) == dpf_stats.count`

Current state (2026-02-24):
- aggregation enforces lineage precheck (`warn|require`).
- training card payload includes `subject_coverage`, `genotype_counts`,
  `dpf_stats`, and `dpf_histogram`.
- data-card plotting includes `genotype_counts` and `dpf_histogram` outputs by
  default (still only opens with `--view`).

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
