# Reusable materialized-pose merged export (2026-09-14)

The exact materialized source manifest is
`docs/diagnostics/pose_head_192_merged_sources_v001.json`. It selects
`crop_runs/pose_head_192_traditional_v1_v001` in 52 existing recording
training Zarrs: 28 RedScare, 16 DefaultScreen, 4 GoodCopBadCop, and 4 clipped
Sleepyfish. Its source rows total 8,870. The crop recipe is the user-approved
fixed 192-pixel override of the size formula, with all three required head
keypoints inside every crop. The source runs remain selector-ineligible and
their original selectors remain unchanged.

The exporter now accepts the explicit `source_kind=materialized_pose_crop`
input contract. The reusable reader in
`src/fisheye/shared/zarr/materialized_pose_training_source.py` validates the
materialized schema, completion, pixel/keypoint/visibility digests, ROI
geometry, ordered labels, and source-row lineage. The source manifest pins
these content digests and the exact run ID for every source. Its registry
grouping snapshot records subject IDs, acquisition starts, and the resulting
42 leakage groups. The exporter rejects source drift and partial visibility
instead of silently converting it into fully visible supervision. A later
merged-schema version can carry partial per-point visibility; this export
requires all three points visible.

The existing immutable merged exporter owns output storage and publication.
The merged `source_index` records source dataset/run bindings, original
keypoint and crop row IDs, local training-frame and acquisition-frame IDs,
sensor-pixel crop origins and source detector boxes. Unsupported historical
detection-row identities remain `-1` under the existing pose-only lineage
policy. The training `bbox_norm_coords` is the three-point pose envelope;
`crop_bbox_norm_coords` is the visible intersection of the source detector
box with the 192-pixel crop. The output records a logical SHA-256 over row
order, pixels, labels, boxes, lineage, source bindings, and fixed splits,
separately from the physical Zarr publication inventory.

The selector-ineligible, registry-deferred immutable benchmark output is:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/pose_head_merged_v1/pose_head_192_traditional_v1_reuse_v001/zarr/pose_head_192_traditional_v1_reuse_v001_merged.zarr`

Its merged run is `merged_export_20260914T135250Z`. It contains 8,870
`uint8` mono images of shape 192×192 and 8,870 `float32` three-point poses in
`traditional_v1` order. The fixed leakage-group split is 7,037 train and
1,833 validation rows, with no group overlap. The logical dataset SHA-256 is
`fed55545dce7bf3783039f8abac431dc0c4a54c8c30ed9461d97a78102c96d78`.
The durable source manifest SHA-256 is
`0e5eb8dd81130d4105b0b8b1b4819eabe1fb013ea11f5178d1951fa8c008e7e1`.
The output directory occupies approximately 205 MiB. The generated
`pose_head_192_traditional_v1_reuse_v001.yaml` is beside its `zarr` directory.

| Split | RedScare | DefaultScreen | GoodCopBadCop | Sleepyfish |
| --- | ---: | ---: | ---: | ---: |
| Train rows | 4,318 | 1,773 | 0 | 946 |
| Validation rows | 1,160 | 198 | 475 | 0 |

All 52 Zarrs contribute rows to the merged artifact. GoodCopBadCop is one
acquisition group and occurs only in validation; Sleepyfish is one acquisition
group and occurs only in training. This split cannot assess within-cohort
generalization for either cohort. A different split policy or separate
cohort-held-out evaluation is required for that claim.

Validation: the 52-source read-only preflight resolved 8,870 rows and 42
groups; 33 existing exporter/materializer/adapter tests passed outside the
sandbox, followed by focused tests of the generated config and unpatched pose
loader. The immutable publisher validated the local candidate, published
copy, direct/consolidated metadata, storage plan, source identity, split
isolation, and logical hash. An independent published-artifact read confirmed
the config, 1,833-row validation loader, 3×192×192 image, and three visible
keypoints. Ruff, `py_compile`, Zarr storage census, file-size and Zarr-open
ratchets, observed metadata literals, contract freshness, and `git diff
--check` passed locally.

Implementation worktree: `/tmp/palette-pose-head-crops-20260914`, branch
`agent/palette/pose-head-crops-20260914`, based on fetched `origin/main`
`999358c3523fcd0cbf344eca669ded4e7b41e586`. The current Codex worktree
owns the new shared source validator, exporter and pose config changes, the
new integration test, source manifest, this handoff, prior crop-materializer
work, and regenerated Zarr census files. All code and documents are
uncommitted. The scientific/schema addition is the exact three-point
materialized source type and its merged row provenance; existing legacy
export behavior remains covered by regression tests. Required GitHub CI
checks have not run on this uncommitted branch, including generated
artifacts, import boundaries, package/collection, test shards, and
`ci-required`. `lint-imports` was unavailable as a shell command locally.
The published run provenance records base Git SHA
`999358c3523fcd0cbf344eca669ded4e7b41e586` with `git_dirty=true`;
the base SHA alone therefore does not identify the exact producing code.
The benchmark is implemented, locally validated, and published; it is not
merged, deployed for production, registered, activated, or merge-ready.

## YOLO11n-pose experimental training smoke

On 2026-09-14, the unpatched pose trainer consumed the published merged
artifact with `--no-log-registry` using `scripts/py` and the existing
`palette-py311-gpu` interpreter (`PALETTE_PYTHON` override). The run used the
official `yolo11n-pose.pt` pretrained checkpoint, SHA-256
`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`,
retained in the benchmark `pretrained` directory. The run snapshots its
experimental config at `inputs/yolo11n_pose_smoke_v001.yaml`:
one epoch, batch 64, image size 192, four loader workers, AdamW, no
augmentation, seed 42. The trainer adapted the COCO model head from 17
keypoints/80 classes to the declared three keypoints/one fish class. It used
the artifact split of 7,037 train and 1,833 validation rows.

The successful run is
`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/pose_head_merged_v1/pose_head_192_traditional_v1_reuse_v001/runs/yolo11n_pose_192_smoke_20260914`.
Its `pose_training_runtime_receipt.json` verifies the starting checkpoint
hash, 192×192 input and first normalized 64-image batch. `weights/best.pt`
has SHA-256
`566d80dea1691052000db9ed4f5d02fc400185af3d2ab6008b28052143d4b684`.
Final held-out crop metrics were pose mAP50 0.993 and pose mAP50–95 0.905;
box mAP50 0.971 and box mAP50–95 0.644. The saved checkpoint passed the
trainer's final validation, and the validation prediction panels were spot
checked against the labels. This is a one-epoch integration and feasibility
result, not a full training study or whole-video evaluation. No registry run,
selector activation, or production publication was performed.

The run-aware pose exporter was then tested on this checkpoint at opset 17,
static batch 1 and 192×192. The canonical ONNX is under the smoke run's
`exports/onnx` directory, with its Palette export manifest. Independent ONNX
checking confirmed FP32 `1×3×192×192 → 1×14×756`, one standard opset-17
domain, and no NMS node. ONNX Runtime CPU output matched PyTorch within
`rtol=1e-3, atol=1e-3`. A staged `model_sources/pose/<run_id>` bundle adds
the ordered labels, dataset logical hash, source-manifest hash, crop
percentiles, preprocessing contract, and validation results. It is an
experimental bundle, not an orange-side deployment.

## 100-epoch run and structured ONNX export

The 100-epoch config is
`experimental_training/yolo11n_pose_100e_v001.yaml` in this worktree. It
keeps the successful smoke settings (YOLO11n-pose, batch 64, image size 192,
four workers, AdamW, seed 42, no augmentation) and sets `epochs=100` and
`patience=100` so early stopping cannot shorten the requested run. The
pretrained source checkpoint is the hash-verified copy under the benchmark
`pretrained` directory. The training command uses `--no-log-registry`,
`--export-onnx --onnx-opset 17 --onnx-batch 1`, with no dynamic-shape or
TensorRT flags. The runner is
`experimental_training/run_yolo11n_pose_100e.sh`, supervised by detached
tmux server socket `palette-pose-100e`, session `yolo11n_pose_100e` on the
workstation. The first managed-shell launch was cleanly interrupted before
the first batch to move it under tmux; its partial outputs were preserved as
`yolo11n_pose_192_100e_20260914.interrupted_launch1*`.

The active run path is the benchmark `runs/yolo11n_pose_192_100e_20260914`
directory. Its sibling `.log`, `.pid`, and `.status` files record progress,
process identity, and eventual exit code. After `train_pose` completes,
`experimental_training/finalize_pose_onnx.py` validates exactly 100 recorded
epochs, the exported graph contract, ONNX Runtime/PyTorch output parity,
content hashes, source dataset identity, and a verified training receipt.
Only then does it stage the ONNX and complete model-source manifest at
`model_sources/pose/yolo11n_pose_192_100e_20260914` under the run directory.
The finalizer was exercised successfully against the smoke run. The orange
device path `/home/jeremy/orange_data/model_sources/pose` is not mounted on
this workstation; the benchmark bundle is the current staging location.
The run's `.status` records exit 0, and `results.csv` contains exactly 100
epochs. Its final pose mAP50 is 0.99423 and pose mAP50–95 is 0.97949; box
mAP50 is 0.98594 and box mAP50–95 is 0.72884. The finalizer produced the
experimental selector-ineligible bundle and ONNX manifest. The best checkpoint
SHA-256 is `d601afc85bbccbb3203e291b3cb2017a23f4fd1f988845dab694288b244c3dc6`;
the ONNX SHA-256 is
`01a15598f6704468616ab37cc1463dfc011d4903747ff61558062e8c58d952b6`.
This model was trained on the earlier 8,870-row merged cohort, which includes
Sleepyfish. It has not been evaluated on whole videos or promoted to the
registry, and it is distinct from the newly recovered 10,721-row crop cohort.
No production selector or orange engine has been changed.
