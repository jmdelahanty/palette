# Palette to webKnossos bridge spike

<!-- contract-meta
status: proposed
created: 2026-07-05
owner: jeremy
audience: Palette maintainers + webKnossos administrators
related: docs/labeling_platform_build_vs_adopt.md,
         docs/web_labeling_modularization_plan.md
-->

## Purpose

Evaluate whether webKnossos can become the multi-user annotation surface for
Palette while Palette remains the source-of-truth data-management backbone.

Prep worksheet: `docs/webknossos_bridge_spike_prep_checklist.md`.

The spike must answer one operational question before larger integration work:

> Can webKnossos read a Palette-derived dataset view without creating a
> manually divergent copy of the training Zarr pixels?

If yes, webKnossos is a strong candidate for masks, bounding boxes, and possibly
keypoints. If no, the current Palette editor remains safer for canonical review
until a controlled adapter/cache contract exists.

## Source-of-truth rule

Palette remains canonical.

webKnossos may receive:

- a read-only filesystem view,
- a symlinked dataset view,
- a remote read-only Zarr/OME-NGFF view,
- or a disposable compatibility cache.

webKnossos must not become the canonical owner of Palette imagery, task scope,
row lineage, run identity, review state, or approved training labels.

Any annotation imported from webKnossos must be attached back to Palette with:

- `dataset_id`,
- `recording_id`,
- source training Zarr path,
- source crop/refined run IDs,
- row/slice/time mapping,
- annotation export ID/path,
- annotator identity,
- created/imported timestamps,
- import code version,
- and the Palette run/revision receiving the import.

## Facts established so far

### webKnossos is open-source and self-hostable

webKnossos is an open-source web application and can be installed on a server
with Docker/Docker Compose. Its documentation describes self-hosting with a
domain, HTTPS, and a `binaryData` directory for datasets.

Relevant docs:

- https://docs.webknossos.org/webknossos/open_source/installation.html
- https://github.com/scalableminds/webknossos

### There is also an open-source Python library

The `webknossos` Python package supports dataset and annotation operations,
including upload/download, remote dataset access, volume annotations, and
skeleton annotations. It is useful for bridge automation and import/export
experiments, not only for local file conversion.

Relevant docs:

- https://docs.webknossos.org/webknossos-py/
- https://github.com/scalableminds/webknossos-libs
- https://pypi.org/project/webknossos/

### Annotation primitives are plausible for Palette

Norman Rzepka confirmed that webKnossos should support:

- boxes,
- points through skeleton tools,
- segmentations.

Documentation confirms:

- volume annotations support trace, brush, eraser, fill, quick-select, and
  interpolation tools,
- volume annotations can be exported as metadata plus voxel label data,
- skeleton annotations are node/edge/tree structures and can be exported as
  NML or CSV,
- bounding-box geometry exists in the Python API.

Relevant docs:

- https://docs.webknossos.org/webknossos/volume_annotation/tools.html
- https://docs.webknossos.org/webknossos/volume_annotation/import_export.html
- https://docs.webknossos.org/webknossos/skeleton_annotation/tools.html
- https://docs.webknossos.org/webknossos/skeleton_annotation/import_export.html
- https://docs.webknossos.org/api/webknossos/geometry/bounding_box.html

Remaining uncertainty:

- Whether webKnossos's box tooling exports exactly the object-detection
  annotation semantics Palette needs.
- Whether named Palette pose landmarks map cleanly enough onto skeleton nodes.

### webKnossos can use shared filesystem datasets

For self-hosted instances, webKnossos can import large datasets placed directly
under:

```text
<WEBKNOSSOS directory>/binaryData/<Organization name>/<Dataset name>
```

It also documents symlink support if the Docker container can see the symlink
target through mounted volumes.

Relevant docs:

- https://docs.webknossos.org/webknossos/open_source/dataset_handling.html

This is the strongest path for avoiding copied pixel exports, but only if the
dataset view is in a webKnossos-readable layout.

### webKnossos does not read arbitrary Palette Zarr internals directly

Palette training Zarrs are domain stores, for example:

```text
crop_runs/<crop_run>/roi_images
refined_subject_masks_runs/<run>/masks_roi
raw_video/images_ds
raw_video/original_frame_indices
```

webKnossos expects a dataset/layer layout such as OME-Zarr/NGFF, webKnossos
Zarr, WKW, N5, Neuroglancer Precomputed, or image-stack inputs converted into
one of those formats. A direct symlink to a full Palette training Zarr is
unlikely to be sufficient.

### OME-Zarr is a practical interface, not a microscopy claim

OME-NGFF is awkwardly named for this use case, but the useful part is not the
microscopy branding. The useful part is a chunked, multiscale, n-dimensional
image interface that webKnossos already understands.

webKnossos documents OME-Zarr/NGFF support, including:

- Zarr as a default dataset format,
- OME-Zarr v0.4/v0.5 remote streaming,
- image arrays up to five dimensions,
- axis order with time before channel before spatial axes,
- time-series and n-dimensional datasets for Zarr,
- performance recommendations for chunking/downsampling.

Relevant docs:

- https://docs.webknossos.org/webknossos/data/zarr.html
- https://docs.webknossos.org/webknossos/data/streaming.html

For Palette video/ROI review, the candidate interpretations are:

| Palette concept | webKnossos view option | Notes |
|---|---|---|
| ROI row/frame | `z` slice in a 3D stack | Likely best performance; semantically odd but simple |
| ROI row/frame | `t` in a time-series view | Semantically closer to video; may be less native |
| grayscale ROI | single color layer | Direct match for `uint8` luma/crop pixels |
| mask component | independent segmentation layer/task | Preserves overlapping Palette components |

## Hard guardrails

1. Do not create a manual export that becomes the source of truth.
2. Do not let webKnossos overwrite Palette training Zarrs directly.
3. Do not flatten overlapping Palette mask components into one mutually
   exclusive integer label volume unless the component being edited is truly
   independent.
4. Do not lose row lineage. Every webKnossos slice/time position must map back
   to Palette row identity.
5. Treat any copied pixel export as a disposable cache with a manifest and
   content provenance.

## Candidate integration paths

### Path A: metadata-only OME-Zarr wrapper with symlinked array

Create a small OME-Zarr-compatible wrapper whose image array path points to an
existing Palette array, for example:

```text
webknossos_views/
  redscare_roi_stack.zarr/
    .zgroup
    .zattrs
    0 -> /groups/.../RedScare_training.zarr/crop_runs/<crop_run>/roi_images
```

First interpretation:

- expose `roi_images` shape `(200, 384, 384)` as a `z, y, x` stack,
- treat `z` as Palette row index,
- keep row semantics in a sidecar manifest.

Why this is attractive:

- no pixel copy,
- webKnossos sees a standard-ish image group,
- Palette remains canonical,
- the view is disposable metadata.

Main risk:

- webKnossos may require exact OME-Zarr metadata/chunk layout or reject the
  symlinked array.

Acceptance:

- webKnossos imports the dataset by filesystem scan,
- images render correctly,
- row order matches `crop_runs/<crop_run>/frame_indices`,
- no pixel data was copied.

### Path B: webKnossos filesystem dataset with symlinked layers

Create a webKnossos-readable dataset directory under `binaryData` and symlink
the layer data to Palette-accessible storage.

Why this is attractive:

- aligns with self-hosted webKnossos dataset handling,
- keeps access local to the shared filesystem,
- avoids HTTP/proxy complexity.

Main risk:

- Palette arrays may still need to be wrapped or converted into webKnossos Zarr
  or OME-Zarr layout.

Acceptance:

- the webKnossos container can read `/groups` or the relevant mounted path,
- symlink targets resolve inside the container,
- dataset scan/import succeeds,
- no copied pixel data is required.

### Path C: remote read-only OME-Zarr adapter

Expose a small HTTP endpoint or static read-only directory that serves a
Palette-derived OME-Zarr view. webKnossos imports it as a remote dataset.

Why this is attractive:

- remote datasets are explicitly streamed read-only,
- webKnossos docs say remote datasets are not copied in full,
- the adapter can enforce Palette read policies.

Main risk:

- requires a small service or static view generator,
- still needs OME-Zarr-compatible metadata,
- may be slower than local filesystem access.

Acceptance:

- webKnossos imports the URL as a remote dataset,
- images stream without full copy,
- annotations are stored in webKnossos while imagery remains read-only.

### Path D: tiny copied compatibility control

Create a tiny copied OME-Zarr or image-stack export only as a control test.

Why this is useful:

- separates "webKnossos can display this kind of data" from "our symlink/view
  approach works."

Why this is not the preferred production path:

- copied pixels create divergence risk,
- cache lifecycle must be managed,
- operators could mistake the copy for canonical data.

Acceptance:

- only a tiny canary copy is created,
- it is marked disposable,
- it has a manifest linking it to Palette source identity,
- it is never treated as the canonical label store.

## Step-by-step spike plan

### Step 0: ask the webKnossos admin the deployment questions

Ask:

1. Can the webKnossos Docker container mount `/groups` read-only?
2. What is the webKnossos `binaryData` path for our organization?
3. Are symlinked datasets allowed under `binaryData`?
4. If symlinks are allowed, do symlink targets need to stay within mounted
   Docker volumes?
5. Which webKnossos version is running?
6. Can we create a scratch organization/team/dataset for RedScare canaries?
7. Can we get an API token for Python-library testing?
8. What is the preferred annotation export path on this campus instance?
9. Do they already export volume annotations as Zarr/WKW and skeletons as
   NML/CSV?

Success criterion:

- we know whether local filesystem/symlink testing is possible.

### Step 1: choose the smallest Palette canary

Use the RedScare training Zarr canary:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr
```

Use crop run:

```text
crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare
```

Primary image surface:

```text
crop_runs/<crop_run>/roi_images
```

Known shape:

```text
(200, 384, 384)
```

Lineage fields to preserve in the bridge manifest:

- Palette row index,
- parent frame index,
- source crop video frame index,
- source crop local frame ID,
- source crop geometry/bbox if available.

Success criterion:

- one 200-row canary is enough to test display, annotation, export, and import.

### Step 2: create a bridge manifest before any webKnossos import

Create a manifest next to the webKnossos view/cache. Minimum fields:

```json
{
  "schema": "palette.webknossos_bridge_manifest.v1",
  "dataset_id": "2026-06-23T16-01-09Z_arena_1:z92f469b75d66",
  "recording_id": "2026-06-23T16-01-09Z_arena_1",
  "zarr_use": "training",
  "source_zarr_path": "/groups/.../RedScare_training.zarr",
  "source_crop_run": "crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare",
  "source_array": "crop_runs/<crop_run>/roi_images",
  "view_axis_interpretation": "z_y_x",
  "webknossos_dataset_name": "redscare_roi_stack_canary",
  "row_count": 200,
  "row_mapping": "row_mapping.csv"
}
```

Create `row_mapping.csv` with at least:

```text
webknossos_z,palette_row,frame_index,source_crop_video_frame_index,source_crop_local_frame_id
```

Success criterion:

- before webKnossos sees the data, Palette can already map every rendered slice
  back to canonical row identity.

### Step 3: test the no-copy symlink wrapper first

Try a metadata-only OME-Zarr wrapper:

```text
redscare_roi_stack.zarr/
  .zgroup
  .zattrs
  0 -> <Palette training zarr>/crop_runs/<crop_run>/roi_images
```

Use `z, y, x` semantics first.

Reason:

- Norman suggested a 3D stack should perform better if it is not too confusing.
- The source array already has exactly three dimensions.
- No reshape or rechunk is required if webKnossos accepts the wrapper.

Success criterion:

- webKnossos imports and displays all 200 slices with correct intensity/range.

Failure handling:

- if webKnossos rejects the wrapper, record the exact error,
- do not immediately switch to full copied exports,
- proceed to Step 4 as a controlled compatibility check.

### Step 4: run a tiny copied control only if the symlink wrapper fails

Create a tiny OME-Zarr or image-stack copy for a handful of rows, for example
10 rows.

Purpose:

- prove the webKnossos side can display the data at all,
- isolate whether the failure is format/layout-specific or data-specific.

Success criterion:

- copied control imports and displays.

Interpretation:

- if copied control works and symlink wrapper fails, we need an adapter/view
  layer, not a different annotation strategy.
- if copied control also fails, investigate pixel dtype/range/layout before any
  annotation bridge work.

### Step 5: test segmentation first

Create one scratch volume annotation for one component, preferably:

```text
subject_body
```

Use only a few slices at first.

Export the annotation.

Expected webKnossos export forms:

- annotation metadata, likely NML,
- voxel label data as Zarr or WKW inside/exported with the annotation.

Success criterion:

- exported annotation can be read by the `webknossos` Python library or by a
  simple parser,
- edited voxels map back to Palette rows,
- component identity is unambiguous.

Palette import target:

```text
refined_subject_masks_runs/<target_run>/<component>
```

or the equivalent `MaskStore` component abstraction.

### Step 6: test overlapping component policy

Palette components such as body, eyes, and swim bladder can overlap
semantically. A single mutually exclusive integer label image cannot represent
all Palette components safely.

For v1, use one independent editable surface per component:

| Palette component | webKnossos bridge representation |
|---|---|
| `subject_body` | one segmentation layer/task |
| `swim_bladder` | one segmentation layer/task |
| `eye_left` | one segmentation layer/task |
| `eye_right` | one segmentation layer/task |

Success criterion:

- importing one component does not destroy or reprioritize another component.

### Step 7: test bounding boxes second

Create a small box annotation workflow after mask import/export is understood.

Questions to answer:

1. Are boxes first-class annotation objects in the UI for this task type?
2. Are boxes exported through the Python API, NML/CSV, or project/task export?
3. Are coordinates exported in dataset voxel coordinates?
4. Can boxes be attached to a row/slice/time position cleanly?

Success criterion:

- one webKnossos box maps exactly to one Palette detection row:

```text
webknossos_box -> palette_row -> detect_runs/<run>/bbox_roi_xyxy
```

### Step 8: test keypoints last

Use skeleton nodes only after masks and boxes are working.

Questions to answer:

1. Can one Palette pose instance be represented as one skeleton tree?
2. Can node names/comments preserve Palette keypoint labels?
3. Does NML/CSV export preserve named landmarks reliably?
4. Are missing/NaN landmarks easy to represent?
5. Is the UI less efficient than the current Palette keypoint editor?

Success criterion:

- one exported skeleton maps back to:

```text
keypoint_runs/<run>/points[row, keypoint_id, xy]
```

with named schema identity preserved.

### Step 9: define the import boundary

Do not apply webKnossos annotations directly to approved Palette runs.

Import should create or update a review run through a controlled Palette command:

```text
palette webknossos import ...
```

The import should:

- read the bridge manifest,
- read webKnossos annotation export,
- validate source dataset identity,
- validate row mapping,
- validate component/keypoint/box schema,
- write only to an explicit Palette target run,
- append import/audit events,
- mark review state pending unless explicitly approved later.

Success criterion:

- webKnossos annotation import is repeatable, auditable, and does not depend on
  manual filename interpretation.

## Decision gates

### Gate 1: no-copy image display

Pass if:

- webKnossos can display Palette ROI pixels through a symlinked/local/remote
  view without copying full pixel data.

Fail if:

- the only working path is manual full export/import with no enforceable
  manifest.

### Gate 2: mask round-trip

Pass if:

- one component mask can be edited in webKnossos,
- exported,
- imported into a Palette refined subject-mask run,
- and mapped back to exact Palette rows.

Fail if:

- annotation exports cannot be tied reliably to Palette row/component identity.

### Gate 3: operator ergonomics

Pass if:

- setup can be scripted,
- labelers see understandable tasks,
- admins do not need manual file bookkeeping.

Fail if:

- every assignment requires hand-built exports and ad hoc reconciliation.

## Preferred outcome

The preferred long-term architecture is:

```text
Palette training Zarrs
  -> read-only webKnossos-compatible view/cache
  -> webKnossos annotation task
  -> exported annotation artifact
  -> Palette import command
  -> Palette review/training run with lineage + audit
```

Not:

```text
Palette training Zarr
  -> copied image stack
  -> manually labeled copy
  -> hand-reconciled files
```

The first path preserves Palette's current strength: one canonical data plane
with explicit assignment, lineage, review state, and training-run provenance.
