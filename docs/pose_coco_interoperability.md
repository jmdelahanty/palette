# Pose / COCO Interoperability

Last verified: 2026-07-02

## Purpose

Palette should not lock pose annotations into Palette-only tooling. The
canonical in-repository representation is still the Palette Zarr pose contract,
but reviewed pose labels should be exportable to common external formats,
especially COCO-style keypoint datasets, so other campus tools can train,
inspect, or reuse the data.

This document defines the boundary:

- Palette Zarr is the source of truth for pipeline provenance and review state.
- COCO-style JSON is an interchange/export format.
- Export adapters must make coordinate space, skeleton identity, keypoint order,
  and visibility semantics explicit.

## Native Palette Pose Contract

Palette stores pose as dense numeric arrays under:

- `keypoints_runs/<run>`
- `refined_keypoints_runs/<run>`

The main arrays are:

| Array | Shape | Meaning |
| --- | --- | --- |
| `keypoints_roi` | `(N, K, 2)` | Keypoint coordinates in ROI/crop pixels |
| `keypoints_img` | `(N, K, 2)` | Keypoint coordinates in full-image pixels |
| `keypoints_norm` | `(N, K, 2)` | Full-image normalized coordinates |
| `keypoint_confidences` | `(N, K)` | Per-keypoint confidence scores |

The semantic contract comes from run attributes, not hard-coded indices:

- `keypoint_labels`
- `skeleton_id`
- `kpt_shape`
- `pose_schema`
- `pose_schema.edges`
- `pose_schema.metadata.heading_computation`

Runtime Palette coordinate arrays are `(K, 2)`: x/y only. External training or
interchange formats often use `(K, 3)` by appending visibility or confidence;
that third channel is an export-layer construction, not a native Zarr
coordinate dimension.

Current packaged fish schemas include:

- `traditional_v1`: 3 keypoints
- `traditional_v2`: 5 keypoints
- `traditional_v3`: 10 keypoints

For `traditional_v3`, the labels are:

1. `swim_bladder`
2. `eye_left`
3. `eye_right`
4. `snout_tip`
5. `tail_tip`
6. `mid_tail`
7. `right_pectoral_fin_insertion`
8. `right_pectoral_fin_tip`
9. `left_pectoral_fin_insertion`
10. `left_pectoral_fin_tip`

## COCO Compatibility

Palette pose can be represented as COCO-style keypoint annotations, but it is
not COCO-human pose.

The correct interpretation is:

- COCO is a container schema.
- Palette provides a custom `fish` category with custom fish keypoint labels.
- Tools hard-coded for COCO's 17 human joints will not understand Palette
  semantics without a custom category/skeleton config.
- Tools that accept arbitrary keypoint categories can use the export directly.

## Palette To COCO Mapping

A COCO-style export should produce:

```text
dataset/
  images/
    <image_id>.png
  annotations_train.json
  annotations_val.json
  metadata.json
```

Each COCO `image` record should include:

- `id`
- `file_name`
- `width`
- `height`

Each COCO `annotation` record should include:

- `id`
- `image_id`
- `category_id`
- `bbox`
- `area`
- `keypoints`
- `num_keypoints`
- `iscrowd = 0`

The COCO `category` record should include:

- `id = 1`
- `name = "fish"`
- `keypoints = keypoint_labels`
- `skeleton = pose_schema.edges`, converted from Palette's zero-based node IDs
  to COCO's one-based skeleton indices

## Coordinate Policy

Exporters must require an explicit coordinate mode.

For crop-image exports:

- image files are ROI/crop images
- keypoints come from `keypoints_roi`
- coordinates are crop-local pixels
- bbox is crop-local `xywh` in pixels

For full-frame exports:

- image files are full-frame images
- keypoints come from `keypoints_img`
- coordinates are full-image pixels
- bbox comes from the matching detection/refined-detection/crop lineage,
  transformed into full-image `xywh` pixels

Do not silently mix crop-local and full-frame coordinates in one COCO file. If a
dataset contains both, write separate exports or include separate annotation
sets with explicit provenance.

## Visibility Policy

COCO keypoints use flattened triples:

```text
[x1, y1, v1, x2, y2, v2, ...]
```

Recommended Palette mapping:

- `v = 2`: keypoint is labeled and finite
- `v = 0`: keypoint is missing, NaN, or row is box-only / fish-present-no-pose
- `v = 1`: reserved for future occluded-but-labeled review state

Palette currently does not have a general occlusion-state field, so exporters
should not invent `v = 1` unless a specific reviewed source surface provides
that state.

`num_keypoints` should count keypoints with `v > 0`.

## Bounding Boxes

COCO `bbox` is absolute pixel `xywh`.

For pose-only crop exports, there are two valid policies:

1. Use a pose-derived tight bbox around visible keypoints, with a documented
   margin.
2. Use the source crop/detection bbox if the crop itself is the training object.

The exporter must record which policy was used in `metadata.json`.

For full-frame exports, prefer the reviewed/refined detection bbox associated
with the same row identity as the keypoints. Do not use a crop-video provenance
box as a canonical full-frame detection box unless it has been explicitly
converted and labeled as such.

## Provenance Requirements

`metadata.json` should preserve enough information to round-trip or audit the
export:

- source zarr path
- source dataset id, if registry-backed
- source keypoint group and run
- source crop run
- source detect/refined-detect run, when used for bbox
- `skeleton_id`
- `kpt_shape`
- `keypoint_labels`
- full `pose_schema`
- coordinate mode: `roi` or `full_image`
- bbox policy
- visibility policy
- image pixel contract / input format
- export tool name, version, invocation, and timestamp

This metadata is not optional. Without it, a COCO file can be syntactically
valid but scientifically ambiguous.

## Import Policy

Palette can import COCO-style keypoint data only when the target pose schema is
known.

An importer should fail closed unless one of these is true:

- the COCO `categories[].keypoints` list exactly matches a packaged Palette
  pose schema, or
- the user provides an explicit keypoint-label mapping to a target
  `skeleton_id`.

An importer must also declare whether coordinates are crop-local or full-frame.
COCO itself does not know Palette row lineage, crop runs, review status, or
stage provenance, so those must be reconstructed or explicitly supplied.

## Relation To YOLO Pose

Ultralytics/YOLO pose training is also an export view, not the native Palette
pose contract.

YOLO pose labels usually store normalized bbox plus normalized keypoints with a
visibility channel. That is closer to Palette's training surface than COCO JSON
in some workflows, but it still depends on the same facts:

- one `skeleton_id` per training job
- stable `keypoint_labels` order
- explicit `kpt_shape`
- explicit coordinate normalization and image size
- explicit visibility policy

Palette should be able to export both COCO-style and YOLO-style pose datasets
from the same reviewed Zarr source.

## Recommended Implementation Direction

Add a dedicated exporter rather than changing the native store:

```bash
scripts/py -m fisheye.utils.export_pose_coco \
  /path/to/training_or_analysis.zarr \
  --keypoint-parent refined_keypoints_runs \
  --keypoint-run <run> \
  --coordinate-mode roi \
  --image-source crop_run \
  --out-dir /path/to/export \
  --apply
```

Initial acceptance criteria:

- hard-fail on mixed skeleton identities
- hard-fail on missing `keypoint_labels`, `skeleton_id`, or invalid `kpt_shape`
- hard-fail if coordinate mode cannot be proven
- write `metadata.json`
- validate that COCO skeleton indices match the exported keypoint list
- sample-read a few images and annotations after export

This keeps Palette interoperable without making COCO the internal datastore or
losing provenance that Palette needs for review, registry queries, and
downstream analysis.
