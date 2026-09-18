# Recover masks and seed tail/fin pose review

This workflow joins a surviving merged subject-mask export to an existing
source-only recovered training archive. The join uses recording identity and
the historical sampled training-row key, then verifies identical crop pixels
and matching crop boxes (absolute tolerance `1e-7`, no relative tolerance).
Repeated images are allowed; ambiguous or missing row identities are rejected.

The default `head_tail11_fins_v2` skeleton contains 19 points:

- Existing `swim_bladder`, `eye_left`, `eye_right`, preserving indices 0–2.
- `tail_base`, `tail_point_01` through `tail_point_09`, `tail_tip`.
- Right and left pectoral fin insertion and tip, using the names from
  `traditional_v3`.
- `snout_tip`, also using the existing `traditional_v3` name.

Version 2 appends `snout_tip` at index 18, preserving all 18 indices from
`head_tail11_fins_v1`. The earlier schema remains available for historical
artifacts; it is never reinterpreted as a 19-point skeleton. New versions use
v2 by default. `--pose-schema head_tail11_fins_v1` is an explicit compatibility
option for historical reproduction/resume.

This is a scientific/schema addition with a crop-only review compatibility
path. The existing three- and ten-point schemas, analytics sampling defaults,
original recovery digest index, 192-pixel head crops, and stage selectors remain
unchanged. The browser's conversion of missing JSON points to zero is corrected
for all skeletons; recovered training saves additionally reject points outside
their crop.

## Geometry and provenance

The named recipe uses the existing head points for anatomical orientation and
the maintained subject-shape centerline, caudal contour, and spline functions.
The caudal-most swim-bladder contour point is projected onto the oriented body
centerline. Its normalized arc-length position determines the spline's start
parameter, matching the existing analytics convention. The tail ends at the
posterior skeleton endpoint. The eleven stations include both endpoints.

Unlike the existing parameter-spaced analytics output, this recipe integrates
the fitted spline's speed at 4,097 parameter samples and inverts cumulative
arc length. The intended stations are 0%, 10%, …, 100% of tail length;
integration is numerical, not an assertion of exact analytic distances.
The recipe records this resolution, spline degree/smoothing, intermediate
polyline base, contour anchor, output spline parameters, and arc length.

The snout uses the maintained subject-shape estimator: select the body-contour
point nearest the anatomical midline among candidates within one pixel of the
most forward contour projection. This is a mask-derived label for review;
its validity and failure reason are retained separately from tail validity.
Missing, fragmented, invalid, or out-of-crop snout estimates remain unannotated.

No mask cleanup is performed. Fragmentation, missing/ambiguous geometry, body
border contact, and tail stations outside the body produce retained rows with
missing tail points and explicit reasons. All four fins start missing. The
initial seed preserves head coordinates exactly and keeps per-landmark origins
(`missing`, `recovered_head`, `mask_derived`). Editable copies add `manual`
origin only to changed landmarks and retain a per-landmark edit flag. The
immutable seed retains the prior values and derivation lineage.

Surviving source paths, run attrs, source ROI/detection identifiers, supervision
codes, array digests, producing-code/environment provenance, and row joins are
retained. Historical manual edit events are not fabricated. `frame_indices`
on these compatibility runs means **source training row**, not a recovered
camera frame. Sensor-pixel origins are unavailable and are never synthesized.
The web editors display that distinction.

## Create one version

Run from the checkout that will also serve the recovered review workflow:

```bash
scripts/py -m fisheye.training.recover_merged_subject_masks \
  /path/to/recording_recovered_training.zarr \
  --merged /path/to/subject_mask_merged.zarr \
  --version v001 \
  --report /path/to/recovery-v001.json \
  --apply
```

Omit `--apply` to verify the join and report planned paths. Published sources
normally require consolidated metadata. For a historical merged export that
never published it, explicitly pass `--legacy-unconsolidated-source`; the
receipt records this archaeology mode. The source is not silently rewritten.

Each recording gets new, explicitly named, selector-ineligible runs:

| Family | Contents |
| --- | --- |
| `crop_runs/recovered_full_roi_<version>` | Identical full-size recovered ROI pixels and row lineage |
| `subject_mask_runs/recovered_masks_<version>` | Dense source masks and supervision provenance |
| `refined_subject_masks_runs/recovered_masks_edit_<version>` | Editable dense masks through the existing component editor |
| `keypoints_runs/head_tail11_fins_seed_<version>` | Immutable initial points, derivation diagnostics, and origin codes |
| `refined_keypoints_runs/head_tail11_fins_edit_<version>` | Editable pose labels with empty fin slots |

One recording is processed at a time, with a 512 MiB input payload limit. Image
and mask writes use one row per physical chunk and one writer per run. The
existing atomic run publisher imports each child. No selectors are advanced;
only after all children validate is root metadata consolidated and the report
with tasks released. On interruption, `--resume` accepts only identical,
unedited children. Existing edits, conflicting sources, and tampered payloads
are refused rather than overwritten. Use a new report filename for each call.

## Review and corrections

The report's `tasks` list is accepted by the existing task importer. It contains
an all-row keypoint task and, when needed, a mask task scoped to failed ROI
indices. Each failure also lists its merged row, recovered pose row, historical
source row, and reason. Task import does not assign a recording automatically.

```bash
scripts/py -m fisheye.labeling.web --store /path/to/labeling.sqlite \
  import-tasks --input /path/to/recovery-v001.json
# Add --apply after examining the import report and existing assignment.
```

Use the viewer from this implementation: older versions require sensor origins
and convert missing landmarks to zero. Click a landmark name to select it, then
place it in the image; `[` and `]` also cycle through all 19 points. Initial load
and reset select the first missing point. Saving requires every point to be
finite and inside the crop. `training_eligible` is false until a complete save
passes the existing head-geometry QC. Rejecting/clearing a row clears eligibility.
The required points include the snout; a row with a missing snout is incomplete.
No new merged export or model training is performed by recovery. A future
export adapter must consume this explicit schema and its eligibility/visibility
checks; these runs do not masquerade as the existing head-only crop product.

In the mask editor, choose **View → Binary mask** to inspect the selected label
as white foreground on black background. Painting, erasing, and lasso editing
work in both views. Switching views preserves unsaved mask edits and the current
zoom; the selected view stays active while navigating between ROIs.

Mask corrections stay in dense editable masks. To regenerate from corrected
masks, use a **new** version and add:

```text
--refined-mask-run recovered_masks_edit_v001
```

The new derivation snapshots those exact dense pixels and records their digest
and edit-source attrs. Previous pose edits and seeds remain intact in their
existing version. This initial adapter does not transfer manually edited pose
points between versions: finish mask corrections before investing in fin
annotation. Label-based transfer from ten-point or previously edited skeletons
is a separate extension; positional copying from `traditional_v3` is invalid.

A targeted mask task is diagnostic coverage of the listed rows. Its completion
does not establish a new whole-recording mask review or activate authority.
Historical eye-union supervision remains a union; it is not split into invented
left/right eye labels.
