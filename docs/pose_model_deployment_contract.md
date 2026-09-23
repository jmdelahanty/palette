# Portable pose model deployment contract

<!-- contract-meta
version: 1
status: draft
implementation: partial
last_verified: 2026-09-23
-->

Palette owns this versioned export contract. Orange's engine builder/runtime
and Citrus's acquisition configuration/recording code must explicitly implement
it before claiming support. Their implementation is outside this Palette change.
This is a metadata/schema extension, not a scientific parameter, model, training
order, preprocessing, authority, or selector change.

## Produce and verify

After registered pose-model promotion, use this final packaging step for a new
acquisition handoff. Raw training ONNX exports and the historical three-file
handoff alone do not satisfy this portable contract.

```bash
scripts/py -m fisheye.utils.export_pose_deployment_bundle export \
  --registry /path/to/palette_registry.sqlite \
  --model-run-id REGISTERED_POSE_RUN \
  --destination /path/to/new-deployment-revision \
  --dry-run
# Review the plan, then repeat with --apply instead of --dry-run.

scripts/py -m fisheye.utils.export_pose_deployment_bundle validate \
  /path/to/new-deployment-revision/REGISTERED_POSE_RUN.canonical.manifest.json \
  --expected-manifest-sha256 SHA256_FROM_TRUSTED_HANDOFF
```

The producer opens SQLite read-only using Palette's Python runtime, verifies the
registered successful pose run/model, original promoted canonical manifest,
exact ONNX/checkpoint bytes, training-manifest/schema binding, and full v3 input
contract with its training-runtime receipt. It creates a fresh directory only
after admission. All source files are copied byte-for-byte; the admitted source
is checked again before publication. The canonical manifest is installed last
without replacing another publication. Failed output directories are retained
as incomplete evidence; retry at a new path. Existing destinations are refused.

The command does not update a registry, rewrite a promoted package, move a
selector, build TensorRT engines, or establish numerical/scientific acceptance.
`status=complete` means this metadata package is complete, not acquisition-ready.

## Files and version boundaries

The standalone directory contains these seven files:

| Filename | Meaning |
| --- | --- |
| `<run>.onnx` | Exact registered ONNX bytes |
| `<run>.onnx.manifest.json` | Unchanged source export evidence |
| `source.canonical.manifest.json` | Unchanged promoted v1 canonical evidence |
| `training.manifest.json` | Exact training manifest, not the training dataset |
| `pose_model_input_contract.json` | Unchanged v3 preprocessing/runtime contract |
| `pose_model_skeleton.json` | New v1 projection of the existing schema binding |
| `<run>.canonical.manifest.json` | New v2 portable bundle/completion manifest |

The existing `palette.canonical_onnx_model_manifest` v1 uses model-package-root
paths. It remains immutable and its producer remains compatible. New v2 uses
`payload.path_base="manifest_directory"`; every deployable artifact reference
is `{relative_path, sha256}` relative to that directory. The input-contract
reference additionally retains `payload_digest`. Paths must be unambiguous
relative POSIX paths without traversal or symlink components.

Historical paths inside copied evidence and the embedded schema binding are
sealed lineage, not instructions to open workstation files. A receiver needs
neither the source registry, checkpoint, training Zarr, nor original directory.
The included training manifest provides dataset/schema references and is
cross-checked against the model binding. Weights remain a digest-only reference.

Consumers must dispatch on both `schema_id` and integer `schema_version` and
reject unsupported versions, absent required files, contradictions, and digest
failures. They must not reinterpret v1 as v2 or silently fall back to a config
skeleton. The Python reader is the executable v2 conformance reference; this
document does not define a second schema resolver.

## Skeleton sidecar v1

The closed top-level fields are `schema_id`, `schema_version`, `skeleton_id`,
`nodes`, `edges`, `kpt_shape`, `source`, and `model_schema_binding`:

- `schema_id`: `palette.pose_model_skeleton`; `schema_version`: integer `1`.
- `nodes`: ordered `{id, name}` objects. IDs are consecutive zero-based output
  indices; names are exact training-manifest labels. Never sort or relabel them.
- `edges`: zero-based pairs in the validated training/schema order.
- `kpt_shape`: model/export `[K,D]`, not a coordinate-only decoded `[K,2]`.
  For the current head model it is `[3,3]`, with indices `swim_bladder`,
  `eye_left`, `eye_right` and edges `[[0,1],[0,2],[1,2]]`.
- `source`: `run_id`, `set_id`, `weights_sha256`, `onnx_sha256`, and
  `training_manifest_sha256`. These must join to the canonical manifest and
  copied evidence exactly.
- `model_schema_binding`: the existing complete
  `palette.pose_model_schema_binding` v1 record. Palette's registered resolver
  owns it; this is not a manual assertion or another skeleton authority.

Direct sidecar fields must exactly project the embedded binding. In that
historical binding `pose_schema.kpt_shape` is coordinate-only `[K,2]`, while
`pose_schema.metadata.model_kpt_shape` supplies the sidecar's model `[K,D]`.
Heading and other schema metadata remain available in the binding; this change
does not invent new anatomical semantics for another skeleton.

## Digests and trust

All file `sha256` fields hash raw file bytes with lowercase SHA-256. In
particular, `payload.pose_model_skeleton.sha256` identifies the actual received
sidecar, not a config string or parsed/reserialized equivalent. Transfer its
bytes unchanged. Receive the canonical manifest file SHA-256 through the trusted
handoff, then verify it before following artifact references. Hash consistency
alone is not authentication of an arbitrary sender.

The v2 envelope is exactly `{schema_id, schema_version, payload_digest, payload}`.
Its closed payload comprises `status`, `task`, `run_id`, `set_id`, `path_base`,
`selector_activation`, `weights`, `onnx_interface`, `producer`, and the six
artifact roles shown above (excluding the completion manifest itself).
The first two are `complete` and `pose`; `selector_activation` must be false.
Producer provenance records code commit/dirty state, source hashes, and observed
Python/ONNX/SQLite versions. It is execution provenance, not model authority.

`payload_digest` uses existing `sha256_canonical_json_v1`: sorted object keys,
compact separators, UTF-8, no ASCII escaping, no NaN/Infinity. The embedded
binding's `binding_sha256` retains its DIFFERENT existing
`canonical_json_sort_keys_v1` grammar: ASCII escaping enabled, digest over the
binding object excluding `binding_sha256`. Preserve both grammars; do not use
one for the other. Reject duplicate JSON keys. Do not recompute historical
evidence with a new serialization. Source-file changes legitimately change
producer/manifest identities even when model and skeleton fields are identical.

## Decoder and preprocessing compatibility

The first supported profile is `ultralytics_raw_pose_static_v1`: one FLOAT
`images` input `[1,3,H,W]`, one FLOAT `output0` output `[1,4+C+K*D,N]`, static
positive dimensions, contiguous zero-based class names, pose metadata agreeing
with `[K,D]`, and explicit `nms=False`. `H,W` must match the validated input
contract. The current model is `[1,3,192,192] -> [1,14,756]`. Embedded NMS,
dynamic batches/shapes, external ONNX tensor data, or other tensor/decoder
profiles are refused rather than guessed. Limits are 512 MiB ONNX and 8 MiB per
JSON file. No model inference is performed by this validator.

Preprocessing remains owned by `pose_model_input_contract.json`: runtime profile,
crop geometry, resize/letterbox, color, layout, normalization and equivalence
probe must all be honored. The sidecar does not authorize changing them. A
different skeleton also requires actual runtime/overlay capability support;
passing a generic K-point metadata test does not prove that support.

## Orange/Citrus adoption and returned engines

This is the requested downstream protocol, not a claim that those repositories
already implement it:

1. Verify the trusted manifest SHA and every referenced artifact; validate the
   supported versions and joins before selecting/building/loading the model.
2. Use the sidecar's exact ordered labels/edges for decoding and overlays; check
   the real engine I/O and supported runtime preprocessing profile. Reject
   unsupported K, semantics, versions, or stale/mismatched evidence explicitly.
3. Retain the exact sidecar next to the ONNX and each engine (copy unchanged when
   engine outputs use another directory). A new immutable engine-build/adoption
   manifest must bind source ONNX SHA, portable canonical manifest SHA, sidecar
   SHA/version, input-contract SHA, engine SHA, and observed target/build runtime.
   Do not edit an old engine receipt to make it claim it consumed new evidence.
4. Citrus should carry this exact artifact identity into configuration and the
   recording manifest/H5 using its existing versioned metadata owner. Orange
   should expose the loaded identities for that join. Agree the recording field
   mapping there; do not add a parallel authority or guessed H5 fields here.
5. Return new build/adoption evidence to Palette's existing target-specific
   deployment-artifact workflow as a candidate. Numerical ONNX/TRT parity,
   acquisition smoke/soak, and activation remain separate claims and gates.

The existing 192-pixel ONNX and engines need no numerical change merely to add
this metadata. Existing engines can be associated only after their source
identity and I/O are verified, in new evidence. This does not retroactively
change the original build or establish missing parity.
