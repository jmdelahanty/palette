# Recovered reviewed pose-head merged export (2026-09-15)

## Outcome

The recovered per-recording pose crops now have a reusable source adapter in
the existing immutable keypoint-training exporter. The adapter accepts exact
`palette.training.recovered_pose_head_crop_visibility.v1` runs, validates all
array and contract digests, requires the bound manual training review and
`Danio rerio` species claim, and selects only rows whose three keypoints all
have visibility `2`.

The pinned source manifest is
`docs/diagnostics/pose_head_192_recovered_reviewed_sources_v001.json`. It binds
51 recovered training Zarrs, 26 registry-derived biological leakage groups,
the exact crop run, source content and contract digests, and the registry
grouping snapshot. The cohort contains only the January 2026 DefaultScreen and
Feeding recordings represented by the recovered sources. Neither Sleepyfish
nor Danionella appears in the manifest.

The per-recording crop runs remain unchanged with 10,721 rows. Four rows with
partial visibility are retained there for source fidelity and excluded during
merged-source discovery:

| Recording | Source rows | Exported rows | Excluded rows |
| --- | ---: | ---: | ---: |
| `2026-01-28T19-36-18Z_arena_2` | 185 | 183 | 2 |
| `2026-01-28T21-18-51Z_arena_2` | 229 | 228 | 1 |
| `2026-01-28T22-22-57Z_arena_4` | 188 | 187 | 1 |

The selector-ineligible merged artifact is:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/pose_head_merged_v1/pose_head_192_recovered_reviewed_v001/zarr/pose_head_192_recovered_reviewed_v001_merged.zarr`

Its merged run is `merged_export_20260915T184826Z`. It contains 10,717 fully
supervised 192×192 crops with the `traditional_v1` three-point pose schema.
The biological-group split uses seed 42 and contains 8,538 training rows and
2,179 validation rows, with no leakage-group overlap. The logical dataset
SHA-256 is
`91336badb5dac92892f3afeb456caf610f00c03b57b5c1a7aba24d145cd6506d`.
The pinned source-manifest SHA-256 is
`31e50be2e70e1a64f4229d5fceb54fd8c7e2c509cbb64724336e7f7d7b23edd9`.
The output occupies approximately 196 MiB.

The generated training config is beside the artifact at
`pose_head_192_recovered_reviewed_v001.yaml`. `PoseConfig`, the authoritative
pose loader-config builder, and the unmodified `ZarrYOLODataset` all accept it.
An independent read built the 2,179-row validation dataset, loaded samples as
3×192×192 tensors, and observed visibility `2` for all three keypoints in each
sample checked.

## Geometry and lineage contract

This is a scientific/schema addition. Existing legacy and full-sensor
materialized source behavior is preserved. Recovered sources have a distinct
merged lineage contract version because their crop origins are known only in
the recovered 512×512 pose ROI. The exporter persists:

- original pose-local and detection-local row IDs;
- original merged pose and detection row IDs;
- recovered source frame IDs;
- crop origins in `recovered_pose_roi_512_xy` coordinates;
- the historical normalized detector box as lineage with an explicitly
  unrecoverable source canvas;
- source pixels, keypoints, visibility, review, and recovery bindings and
  digests.

The historical normalized detector box is not relabeled as a crop-local or
full-sensor box. The trainer-facing bounding box and
`crop_bbox_norm_coords` are computed from the three visible landmarks for
these recovered sources. No full-sensor crop equivalence is claimed.

## Validation and status

The actual 51-source preflight resolved 10,717 eligible rows and 26 groups.
Focused exporter, validator, storage, recovery, configuration, and loader tests
passed outside the sandbox. Coverage includes partial-row exclusion, exact
source-row mapping, old materialized-source behavior, source pixel/lineage
tampering, review-state refusal, immutable merged validation, logical-hash
tampering, config parsing, and an unpatched loader read. Ruff, Python
compilation, generated Zarr census freshness, file-size, metadata-literal,
contract-freshness, and whitespace checks passed.

The first publication command completed and validated the numerical Zarr, then
exposed a missing `recovered_pose_crop` member in `PoseConfig` while generating
the YAML. Commit `d6319835` adds that configuration compatibility and its
regression test; the YAML was then generated and independently validated. The
numerical artifact records clean producing commit
`fe1f8f4c37792cadad3f2c9b4523677072aec686`. The preceding recovery and crop
work is commit `5a827a01497ce5368a1f7487aa49a0e9072e547b`.

The numerical and presentation products are complete for experimental use.
The artifact is not registered, activated, promoted, or used to train a model.
Required GitHub CI has not run for these commits, so the branch is not
merge-ready and the dataset remains under the benchmark training surface.
