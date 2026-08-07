# Five-keypoint merged training v3 full-corpus gate — 2026-08-07

Verdict: **PASS FOR SELECTOR-INELIGIBLE TRAINING CANDIDATE; TRAINING AND REGISTRY ACTIVATION NOT YET AUTHORIZED**

## Artifact

- Zarr: `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/keypoint_merged_v3/five_point_reviewed_full_v3_v002/pose_five_point_reviewed_full_v3_v002_merged.zarr`
- Training config: sibling `pose_five_point_reviewed_full_v3_v002.yaml`
- Artifact writer commit: `942e8346e29e3536e795eaba10116d504d1b7357`
- Complete-config generator commit: `e1093a8de57060f94219c00f7533fdbff2ad0ba1`
- Zarr files: 74
- Apparent size: approximately 1.6 GiB

The Zarr is immutable, complete, selector-ineligible, absent from production
selection, and registry activation is deferred.

## Logical and provenance validation

- Schema: merged keypoint training `3.0.0`.
- Sources: 61.
- Samples: 12,704, all full keypoint supervision.
- Split: 10,096 train, 2,608 validation, 0 test.
- Leakage groups: 29 total; 24 train and 5 validation; zero overlap.
- Frame lineage: exact local sample-row and acquisition-frame arrays; the v2
  ambiguous frame alias is absent.
- Detection lineage: 3,047 rows are explicitly pose-only and cannot support
  joint detection/pose claims.
- Skeleton: `pose_skel_traditional_v2`, `[5,3]`.
- Labels: `swim_bladder`, `eye_left`, `eye_right`, `snout_tip`, `tail_tip`.
- Keypoint storage: float32 with checked per-source float64 conversion.
- Metadata: direct and consolidated declarations agree for crop, keypoint,
  split, and source-index surfaces.

The stable source-manifest digest is
`a1dc33d8356dc3b1825500c05084114a8bb9d0d19893fe00303e5530fb7d8299`.
The summary digest is
`7fa9cab2026430cfd790894a72a4171710654e7d9bfbe35c22e04bba55e346e7`.
The merged-manifest digest is
`4b84af223163547af7e4f360e2da144a0cd2ef1794b9f34cedb29068d426c894`.

## Batman transform validation

The reviewed Batman contribution contains 181 rows. Each 348×348 image was
centered in a 512×512 image with 82 zero-valued pixels on every side. The
interior pixels are byte-identical to the source and the translated keypoints
have zero numerical error. No resizing was applied.

## Trainer-reader gate

The first generated YAML exposed a real handoff defect: it contained only a
dataset pointer and seed, so `PoseConfig` rejected it. The exporter now starts
from complete repository pose defaults, overlays any explicit training
configuration, forces exact task/dataset/skeleton fields, derives the image
size from the materialized ROI, and validates `PoseConfig` before writing.

The corrected sidecar SHA-256 is
`e75ec3d00f50b592cebc6e07b2638f791f3c5826403d5d62a3badd47b21565b8`.
The immutable Zarr was not rewritten.

The production diagnostic command was:

```bash
scripts/py -m fisheye.training.diagnose_pose_batch \
  /groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/keypoint_merged_v3/five_point_reviewed_full_v3_v002/pose_five_point_reviewed_full_v3_v002.yaml \
  --batch-size 4
```

It passed with:

- 2,608 validation samples and 2,608 instances;
- zero empty-label samples;
- image batch `[4,3,512,512]`;
- box batch `[4,4]`;
- keypoint batch `[4,5,3]`;
- canonical five-label order;
- artifact splits used rather than recomputed row splits.

The environment emitted an existing Torch/Torchvision compatibility warning.
No dependency mutation was performed, and the loader smoke itself completed
successfully.

## Code validation

- 58 focused exporter, preparation, census, and refined-publication tests
  passed.
- Black passed outside the sandbox.
- Ruff, Python compilation, and `git diff --check` passed.

## Remaining gate

This artifact is ready to serve as the input to a bounded model-training and
evaluation run. It must remain selector-ineligible and registry-deferred until
that model result is reviewed. The current generic `yolov8n-pose.pt` training
initialization in the generated config is a default, not a scientific choice;
the intended starting checkpoint and final hyperparameters should be selected
explicitly before submission.
