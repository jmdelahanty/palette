# Fixed-192 pose-head crops in existing training Zarrs (2026-09-14)

The new per-recording dataset version is `crop_runs/pose_head_192_traditional_v1_v001`.
It is a materialized `uint8` mono crop with its own three-point
`traditional_v1` labels, visibility, sensor-pixel offsets, source boxes, and
source row/frame IDs. Model input preparation is declared as mono replication
to three identical channels followed by division by 255, with no mean/std
normalization. No merged pose dataset was created.

The crop recipe is `pose_head_fixed_192_truncated_centroid_v1`: truncate the
full-sensor `xyxy` box midpoint toward zero, subtract 96, clamp to the sensor
frame, and copy an exact 192×192 pixel window. There is no resize, padding, or
rotation. The fixed size is the requested override of the formula in
`~/pose_head_crop_training_request_2026_09_14.md`; containing the three head
landmarks is the acceptance condition, not containing the entire source box.

| Source cohort | Archives | Usable rows | Box longer side p5 / p50 / p95 (sensor px) | Formula size | Required points outside 192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RedScare | 28 | 5,478 | 126.019 / 139.238 / 156.863 | 224 | 0 |
| DefaultScreen | 16 | 1,971 | 121.524 / 129.132 / 137.430 | 192 | 0 |
| GoodCopBadCop | 4 | 475 | 122.053 / 127.781 / 133.950 | 192 | 0 |
| Sleepyfish clipped | 4 | 946 | 127.201 / 146.065 / 195.726 | 256 | 0 |
| **Total** | **52** | **8,870** | **122.920 / 136.594 / 158.625** | **224 pooled** | **0** |

The 52 new run directories occupy approximately 239 MiB on disk.

Each RedScare and GoodCopBadCop run binds its completed refined-keypoint run to
that keypoint run's own acquisition crop, then uses the crop's full-sensor
`bbox_img_xyxy`. DefaultScreen binds its selected five-point keypoint run to
its own crop. This avoids the known RedScare mismatch from using a later,
unrelated refined-detection box. Sleepyfish uses a distinct legacy adapter:
keypoint and crop rows join by exact local frame/detection identity; crop rows
join the selected authoritative refined-detection rows; normalized `cxcywh`
boxes are converted with the full-sensor dimensions. The source runs are left
untouched. Source skeletons with five labels are projected by name to the
existing `traditional_v1` ordering of swim bladder, left eye, right eye.

All 52 new runs are complete and selector-ineligible. The aggregate post-write
audit checked direct and consolidated metadata, array shapes, the three-point
schema, completion markers, and unchanged `latest`, `latest_complete`, and
`authoritative_run` crop selectors. The writer validated every local and
published pixel/label digest. An independent readback recomputed all 8,870
origins, projected labels, and visibility values and checked a source-pixel
crop in each cohort; there were no mismatches. One RedScare run uses schema
version 1 (the first published source-binding form), 47 direct-box runs use
version 2, and four Sleepyfish runs use version 3 for the explicit legacy
detection binding. The reader validator accepts all three forms. No selector
or production authority was activated.

Batman keypoint-review archives were excluded: their roots still say
`training_artifact_status=review_active`, `stage_selector_eligible=false`,
and have no selected complete refined-keypoint run. Their geometric fit was
checked earlier, but these mutable review snapshots are not a frozen source
for this materialization pass.

Implementation worktree: `/tmp/palette-pose-head-crops-20260914`, branch
`agent/palette/pose-head-crops-20260914`, based on fetched `origin/main`
`999358c3523fcd0cbf344eca669ded4e7b41e586`. The implementation and this
handoff are uncommitted. Dirty scope is the new geometry and materializer
modules, their two focused test files, this handoff, and three regenerated
Zarr census files. The current Codex worktree owns these files. This is a
scientific/schema/identity addition: the
fixed crop and projected three-point identity are new; original frame pixels,
keypoints, crop runs, selection metadata, and scientific defaults remain in
place. It uses the existing atomic run publisher and final root metadata
consolidation. The tracked Zarr storage census was regenerated for the new
writer. `scripts/py -m pytest` on the two focused geometry/publication test
files passed (8 tests) outside the sandbox; Ruff check/format, `py_compile`,
the Zarr census check, the Zarr open-mode ratchet, file-size ratchet, observed
metadata literals, and contract freshness checks passed locally. The GitHub
CI jobs (`generated artifacts`, `import boundaries`, `file-size ratchet`,
`zarr open metadata modes`, `observed metadata literals`, `active contract
freshness`, `package and collection`, non-GPU test shards, and `ci-required`)
have not run on this uncommitted branch, so this work is not merge-ready.
Nothing was pushed, integrated, deployed, or activated.

The current pose training loader still follows selected crop/keypoint runs and
does not consume this self-contained selector-ineligible crop version. A later
explicit consumer/export path must select this run by ID and honor its
per-keypoint visibility before model training. Batman should be reconsidered
only after its keypoint review is frozen and its source binding can be
validated. These are separate from the completed crop materializations above.
