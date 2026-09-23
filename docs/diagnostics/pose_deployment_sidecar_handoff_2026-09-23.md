# Pose deployment sidecar implementation checkpoint — 2026-09-23

Owner: root Palette implementation agent. Worktree:
`/tmp/palette-pose-deployment-sidecar-20260923`; branch:
`agent/palette/pose-deployment-sidecar-20260923`; prerequisite/base:
`a43f21a6abebf41dd26707d1922f13103e92dc24` (includes registered pose promotion).
This document is the pre-CI checkpoint. The draft PR records its exact committed
head and superseding check results; do not infer green CI from this checkpoint.
The original dirty workstation checkout and other workers' branches are untouched.

## Scope and contract ownership

Classification: explicit metadata/schema extension. Scientific outputs, weights,
ONNX bytes, exact training keypoint order, preprocessing, registry rows, historical
serialization/digests, and selectors are preserved. The original v1 promotion
manifest remains unchanged; the new standalone manifest is explicitly v2.

Existing `pose_model_schema_binding` owns registered training/schema resolution.
New `pose_model_skeleton` projects it without another resolver/authority; new
`pose_deployment_manifest` owns portable v2 validation; `pose_onnx_interface`
checks the declared static raw-pose interface. The CLI is
`fisheye.utils.export_pose_deployment_bundle`. These interfaces and the
[contract](../pose_model_deployment_contract.md) are owned by this branch.
Existing promotion only imports the shared schema-ID constant; its v1 producer
and public behavior remain intact.

Implementation adds no database migration, dependencies, source-data rewrite,
engine rebuild, model retraining, Zarr writer, activation, or parallel acceptance
schema. Generic training exporters remain compatibility export artifacts;
the documented final packaging command is required for the new acquisition
handoff. Orange/Citrus consumption and recording-field adoption remain there.

## Validation at this checkpoint

The following outside-sandbox command passed **103 tests** using `scripts/py`:

```bash
scripts/py -m pytest -q \
  tests/unit/fisheye/test_pose_model_skeleton.py \
  tests/unit/fisheye/test_export_pose_deployment_bundle.py \
  tests/unit/fisheye/test_pose_deployment_manifest.py \
  tests/unit/fisheye/test_pose_onnx_interface.py \
  tests/unit/fisheye/test_promote_registered_pose_model.py \
  tests/unit/fisheye/test_pose_model_schema_binding.py \
  tests/unit/fisheye/test_pose_model_input_contract.py \
  tests/unit/fisheye/test_export_onnx_paths.py \
  tests/unit/fisheye/test_onnx_to_tensorrt_paths.py
```

Coverage includes real registered producer -> immutable publication -> relocated
unpatched reader for 3- and 19-keypoint fixtures; source package/registry byte
preservation; wrong source, stale registry, tampered/incomplete files; closed
versions/paths; unsupported ONNX layouts/external data; dry-run/no writes;
directory ownership loss; copy/publication failure and fresh-revision retry.
No core producer/resolver/reader is patched in the successful round trip.
This is metadata/contract conformance, not numerical ONNX/TensorRT parity.

An additional **13 tests** passed in `test_train_pose_runtime_contract.py` and
`test_train_pose_schema_authority.py` (116 focused/regression tests in total).

Local import boundaries, authority-access ratchets, file-size/Zarr-open-mode
ratchets, observed metadata literals, contract freshness, registry-schema
reference and diff checks pass. Generated census output changes only the scanned
Python module count (four new modules); no storage schema or writer changed.

Read-only dry-run also admits the actual registered
`pose_head_192_recovered_reviewed_v001_yolo11n_100e_20260915` model from
`/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`. It preserves
ONNX SHA `db5c8c3305b71c52f147152507dd8cd890d42396162adf74b1b8afbafe84f12f`,
`pose_schema:traditional_v1`, and model keypoint shape `[3,3]`. `/nvme1`'s registry
does not contain this run and is not the source for this handoff. No registry
health/acceptance claim is made by this selected-row read.

## Required CI and remaining work

At the pre-CI checkpoint every remote required check is **unrun**: generated
artifacts; import boundaries; file-size ratchet; Zarr open metadata modes;
observed metadata literals; active contract freshness; package and collection;
non-GPU test shards 0–15; and aggregate `ci-required`. Local focused passes do
not substitute for them. The user authorized commit, push and draft PR after
local tests, but not merge, shared-checkout update, remote transfer or activation.

Implemented and locally validated; not integrated, deployed or activated.
Not complete/merge-ready until required CI on the exact head is green. After
commit, create a new local candidate bundle from clean pinned source, validate
it, and record its manifest/file SHA identities in the PR handoff. Transfer to
pancake0 and acquisition-agent instructions follow separately; preserve the old
flat handoff and original engine receipts. Hardware parity/soak and downstream
consumer capability tests are not performed by this Palette implementation.
