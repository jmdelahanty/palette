# Provenance Checks

This document describes the provenance diagnostics provided by:

- `python -m fisheye.diagnostics.check_provenance_capture`
- `python -m fisheye.diagnostics.check_provenance_consistency`
- `python -m fisheye.diagnostics.check_full_provenance`

Contract reference: `docs/provenance_contract_draft.md`.
Boundary guidance: `docs/pipeline_metadata_boundaries.md`.

It focuses on what each check reports, how to interpret results, and the
optional subject metadata validations.

## check_provenance_capture

Purpose: verify that each pipeline run captured a provenance payload and the
expected minimal fields (timestamp, parameters, inputs).

Example:

```bash
python -m fisheye.diagnostics.check_provenance_capture /nvme1/recordings --recursive
```

Optional subject metadata validation:

```bash
python -m fisheye.diagnostics.check_provenance_capture /nvme1/recordings --recursive --check-subject-metadata
```

What it checks per stage (latest run by default):

- `timestamp`: created timestamp exists (in provenance or attrs)
- `parameters`: parameter payload exists
- `inputs`: source/input pointers exist
- `git` (optional): git metadata exists
- `environment` (optional): environment/platform metadata exists

Stage names:

- `detect`
- `refined_detect`
- `crop`
- `keypoints`
- `refined_keypoints`
- `eye_masks`
- `refined_eye_masks`
- `id_assignment`

Strict mode:

- `--strict` returns non-zero when required checks fail.
- In strict mode, refinement stages additionally require:
  - `provenance.contract.name == "palette_stage_provenance"`
  - `provenance.contract.version >= 1`

Subject metadata (optional):

When `--check-subject-metadata` is enabled, the tool inspects
`analysis_metadata.subject_metadata` and reports:

- `missing`: `fish_id`, `subject_count`, or `subject_metadata`
- `warnings`:
  - `fish_id_format`: fish_id is not a UUID string
  - `subject_count_not_int`: subject_count is not an integer
  - `subject_count_lt_1`: subject_count < 1
  - `subject_type_mismatch_single`: subject_count == 1 but subject_type not in {individual, single}
  - `subject_type_mismatch_group`: subject_count > 1 but subject_type in {individual, single}

The subject metadata check is informational only. It never fails the script.

## check_provenance_consistency

Purpose: validate that downstream outputs are consistent with their source data.
This is primarily about counts and lineage, not metadata completeness.

Example:

```bash
python -m fisheye.diagnostics.check_provenance_consistency /path/to/recording.zarr
```

What it checks:

- Crop ROI count matches the detection source used for that crop run.
- Keypoint row count matches the crop ROI count.

Output includes a summary of the detected lineage:

- `source` (path of detection source for crops)
- `detections` (row count in source detections)
- `crops` (ROI count)
- `keypoints` (row count)

## check_full_provenance

Purpose: deep audit of provenance payloads and their structure. Use when you
need a full report for a single recording or for debugging inconsistent metadata.

Example:

```bash
python -m fisheye.diagnostics.check_full_provenance /path/to/recording.zarr
```

## Subject metadata schema (summary)

Subject metadata is mirrored from H5 into the Zarr at:

- `analysis_metadata.subject_metadata`

Recommended fields (capture-time provenance):

- `fish_id` (UUID string when applicable)
- `subject_count` (int)
- `subject_type` (individual, group, etc)
- `fish_count` (if provided)
- `dish_id`, `cross_id`, `genotype`, `line_strain`
- `species`, `sex`

See `docs/zebrobot_snapshot.md` for the canonical snapshot handling rules.
