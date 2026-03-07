# Legacy Analysis and Training Archive Migration Policy

Date: 2026-03-06
Type: Policy note / migration guidance

## Purpose

Document how existing `Palette` analysis zarrs and existing training zarrs
should be handled during the crop-storage-mode redesign and future dataset
curation work.

This is specifically about:

- corrected analysis archives that already contain reviewed detections,
  keypoints, approvals, and related provenance,
- existing dedicated training artifacts,
- future curated training datasets derived from videos, sampled frames, and
  deliberate human cleanup.

## Decision Summary

- Existing corrected analysis zarrs should be preserved as stable source
  archives.
- Existing training zarrs should remain materialized, self-contained training
  artifacts.
- Immediate migration should prefer metadata/provenance backfill over
  destructive in-place conversion.
- Analysis archives and training artifacts should remain different products with
  different guarantees.
- Temporary ROI inference caches should be treated as runtime accelerators, not
  as canonical archive content or training artifacts.
- Future merged datasets may pull curated rows from both dedicated training
  artifacts and selected manually corrected frames from analysis archives, but
  the merged output should itself remain a stable materialized training
  artifact.

## Existing Corrected Analysis Zarrs

Corrected analysis zarrs already represent meaningful curated work. They should
be treated as stable source-of-truth archives rather than as the first targets
for destructive geometry-only conversion.

Recommended handling:

- back them up,
- preserve existing `roi_images`,
- backfill new metadata and pointer fields,
- validate/backfill provenance fields needed by the new crop contract,
- avoid removing image arrays or changing `crop_runs.latest` semantics in place.

This gives immediate forward compatibility without destabilizing trusted
archives.

## Existing Training Zarrs

Existing training zarrs should also be touched up, but they should not be
treated as future `geometry_only` candidates. Their role is different.

Recommended handling:

- keep them materialized,
- backfill new crop-storage and provenance metadata,
- validate that required lineage/provenance fields exist,
- regenerate only when regeneration is cleaner than patching and lineage can be
  preserved clearly.

Training artifacts are allowed to duplicate image data on purpose. That
duplication is part of the artifact contract.

## Temporary ROI Inference Caches

Some analysis workflows will still benefit from materialized ROI tensors even
when the canonical archive remains lean.

This is especially true when source videos are large enough that full-frame
decode becomes the bottleneck for downstream ROI-model inference. In that case,
the right tool is a temporary ROI cache, not a change to the archival contract.

Recommended handling:

- keep the canonical analysis archive lineage-first,
- allow an explicit temporary cache of ROI tensors for fast inference,
- keep that cache outside the canonical archive by default,
- treat the cache as disposable/regenerable runtime state,
- do not confuse temporary caches with curated training artifacts.

This creates three distinct products:

- analysis archives: canonical, lineage-rich, may eventually be mixed-mode,
- temporary ROI caches: runtime accelerators for throughput-sensitive
  inference,
- training artifacts: durable, versioned, curated datasets.

## Training Artifact Philosophy

The intended training-data lifecycle is sound:

1. Start from a set of videos.
2. Import a selected set of frames or sampled images (for example, every nth
   frame or another deliberate sampling policy).
3. Generate automated labels/candidates.
4. Perform deliberate human cleanup and correction.
5. Freeze the result as a dedicated training artifact.
6. Reuse those curated artifacts as high-trust sources for future merged
   datasets.

This is good practice because it makes the training artifact:

- self-contained,
- stable and repeatable,
- versionable,
- auditable,
- portable across future training/export workflows.

It also creates a clear distinction between:

- operational/analysis archives, which may continue evolving as production or
  review records,
- temporary runtime caches, which may be created and discarded to accelerate
  inference,
- curated training artifacts, which should represent a best-effort labeled
  dataset snapshot at a point in time.

## Best-Possible-Effort Training Datasets

Dedicated training zarrs should be treated as "best possible effort" labeled
artifacts for the subset of data they cover.

That means they should aim to capture:

- the exact ROI/image tensors used for training,
- the exact labels after manual cleanup,
- the lineage back to source videos / source archives / source runs,
- the sampling policy used to choose frames,
- any approval/review signals relevant to label trust,
- a versioned artifact identity.

In other words: these artifacts should not just be convenient training inputs.
They should function as curated, documented dataset releases.

## Relationship To Future Merged Datasets

Future merged datasets can and should eventually be able to gather:

- curated rows from dedicated training artifacts,
- selected manually corrected or approved frames from analysis archives.

But the merged output should still be treated as a fresh, materialized training
artifact with its own:

- version,
- manifest,
- lineage,
- source row references,
- materialized tensors.

This keeps the merged dataset reproducible and self-documenting even when its
sources are heterogeneous.

## Backfill vs Regenerate

### Prefer backfill when:

- the existing archive already contains trusted manual review/approval work,
- regeneration would risk drift or accidental change,
- the artifact is already being relied upon operationally.

### Prefer regenerate when:

- the artifact is easy to rebuild,
- the current artifact is incomplete or inconsistent,
- regeneration produces a cleaner contract than patching in place,
- lineage/versioning can be preserved clearly.

The default recommendation for legacy corrected analysis zarrs is backfill.
The recommendation for legacy training zarrs is "backfill unless regeneration
is clearly cleaner and low-risk."

## Immediate Migration Policy

### For existing corrected analysis zarrs

- backup first,
- backfill crop-storage metadata (`crop_storage_mode=materialized`),
- backfill pointer attrs (`latest`, `latest_materialized`, `latest_any` as
  applicable),
- validate/backfill provenance (`frame_counts`, `detection_indices`,
  `roi_size`, `crop_signature`, source lineage),
- do not remove `roi_images` yet.

### For existing training zarrs

- backup first,
- keep them materialized,
- backfill the same core crop/provenance metadata where relevant,
- mark them clearly as training artifacts in docs/manifests/metadata,
- decide case-by-case whether backfill or regeneration is cleaner.

## Long-Term Policy

- Analysis archives should stay lean and lineage-first.
- Analysis archives should not require imported/downsampled image payloads as a
  baseline contract.
- Temporary ROI caches may exist for performance, but they should not become
  the baseline archive contract.
- Training artifacts should stay materialized and self-contained.
- Mixed-mode support should primarily affect analysis archives first.
- Training pipelines should consume stable materialized artifacts even when
  source analysis archives eventually support geometry-only crop runs.

## Bottom Line

Use existing corrected analysis zarrs and existing training zarrs as the
foundation for the redesign, but do so conservatively:

- preserve trusted archives,
- backfill metadata first,
- use temporary ROI caches when large-video inference needs them,
- keep training artifacts materialized,
- treat curated training zarrs as best-effort labeled dataset releases,
- let future merged datasets inherit from both training artifacts and selected
  corrected analysis data while remaining materialized, versioned artifacts
  themselves.
