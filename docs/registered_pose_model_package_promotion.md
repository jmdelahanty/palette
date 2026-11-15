# Registered pose-model package promotion

`fisheye.utils.promote_registered_pose_model` relocates one already successful
pose training run from node-local storage into the durable shared model root. It
does not create a second training identity, alter model or ONNX bytes, change a
status, or activate a selector.

The command requires the exact existing `training_runs`, `training_models`, and
`onnx_models` rows. It verifies the registered hashes, rejects symlinks, copies
the complete run through a hidden sibling, compares every source file by
relative path, size, and SHA-256, adds a relocation-safe ONNX manifest and a
digest-bound pose model-input contract, then atomically publishes the package.
Only after publication does one SQLite transaction rebind the existing model,
metrics, ONNX, and ONNX-manifest paths.

Apply requires a fresh receipt from `scripts/backup_palette_registry.sh`. The
publisher compares the live registry hash with that receipt again while holding
its write transaction. Run the Palette-runtime integrity validator after apply:

```bash
scripts/py -m fisheye.utils.registry_integrity --registry REGISTRY.sqlite
```

Modern pose runs can carry
`palette.pose_training_runtime_receipt.v2`. Their version-3 model-input contract
binds that immutable receipt instead of adding inferred fields to the original
training manifest. It accepts only exact identity geometry observed for both
train and validation loading, `uint8` luma repeated across three channels,
float32 `/255` normalization, disabled augmentation, and matching source and
network dimensions. Historical version-1 and empirical version-2 contracts
remain supported without changing their serialized bytes.

Registry `status=success` is the existing model-candidate admission mechanism;
there is no separate promoted-model column. Exact deployment workflows should
pin both the model set and run. Package publication records
`selector_activation=false`; activation remains a later decision based on a
commit-pinned canary and its visual or quantitative review.
