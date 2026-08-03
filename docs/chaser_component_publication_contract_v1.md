# Chaser Component Publication Contract v1

Date: 2026-08-03

Status: shared logical and atomic-publication primitives implemented;
scientific-writer/workflow adoption pending. This contract does not activate a
production selector.

## Purpose

A verified `analysis/chaser_distance_runs/<run>` base does not make arbitrary
child groups authoritative. Every maintained derived component must be an
independent immutable publication whose exact payload and semantics remain
bound to that base run.

The implementation lives in
`src/fisheye/analysis/chaser_component_publication.py`.

## Lifecycle

1. A workflow builds one component under node-local scratch.
2. The component is complete before it is sealed.
3. `persist_chaser_component_manifest()` inventories every group and array,
   hashes decoded array values, and writes an ineligible manifest.
4. The workflow copies the component to a hidden sibling on the destination
   filesystem, reopens it, and calls `validate_chaser_component_manifest()`.
5. The validated hidden sibling is renamed atomically to a new immutable final
   component name. Maintained publication must never delete and rewrite a
   visible component.
6. Only then may `persist_chaser_component_selector()` publish the exact
   digest-bound authority envelope on the component-family parent.
7. A reader validates the selector, base binding, manifest, attributes, dtypes,
   shapes, and content digests before treating the component as scientific
   authority. `latest`, sorted-child, and raw-child fallback remain forbidden.

The logical module owns steps 3, 4 validation, 6, and 7.
`analysis_workflows/materializers/chaser_component.py` owns the destination
lock, hidden copy, revalidation, immutable rename, completion receipt,
conditional selector rollback, and literal final eligibility commit. Scientific
writers still need to adopt that materializer instead of writing directly into
the authoritative archive.

## Manifest Envelope

The component root stores:

- `chaser_component_publication_manifest`;
- `chaser_component_publication_manifest_sha256`.

The exact manifest contains:

- fixed manifest schema ID/version and complete/ineligible lifecycle state;
- exact component family, name, relative path, semantic schema ID/version, and
  method ID/version;
- base run path plus publication-seal, surface-manifest, row-identity, and
  detached read-authority digests;
- exact source authorities and algorithm parameters;
- an ordered declaration of every group and every array;
- all authoritative group and array attributes except the two circular
  component-manifest attributes and the explicit non-scientific mechanical
  publisher attributes (owner, staging telemetry, completion/error state, and
  selector eligibility);
- exact array dtype, shape, and decoded-value SHA-256.

Nested field sets are closed. Rehashing an envelope after adding fields or
changing a known semantic contract is rejected. Object arrays and non-finite
JSON values are rejected.

## Selector Envelope

The component-family parent stores:

- `chaser_component_publication_authority`;
- `chaser_component_publication_authority_sha256`.

This envelope contains one approved component name/path, the component
manifest reference and digest, and the exact base run/publication-seal binding.
It intentionally does not mutate historical `latest` or `latest_complete`
attributes. Maintained readers must consume only this authority envelope.

## Write Ownership and Safety

- A component name is immutable after sealing.
- A manifest cannot be rewritten.
- A component is selector-ineligible while being built and copied.
- Publishing another revision creates another component name and then changes
  the authority envelope; it does not overwrite the selected directory.
- The workflow must own the entire component directory during copy and rename.
- Parallel workers may prepare independent components, but only one finalizer
  may mutate a given component-family selector at a time.
- Array writers must still obey whole, non-overlapping physical chunk/shard
  ownership.

## Adoption Checklist

- [x] Exact manifest envelope and canonical digest.
- [x] Exact payload inventory, authoritative attributes, dtypes, shapes, and
      decoded-value hashes.
- [x] Base publication-seal, surface-manifest, row-identity, and read-authority
      binding.
- [x] Digest-bound component-family selector envelope with no fallback.
- [x] Adversarial tests for payload, parameter, nested-field, selector, and base
      identity tampering.
- [x] Add the destination hidden-copy/validate/rename publisher and a failed,
      selector-ineligible recovery tombstone.
- [ ] Give every maintained component an exact semantic schema declaration.
- [ ] Migrate all maintained component writers off direct visible mutation.
- [ ] Migrate readers/exports to validated component handles.
- [ ] Expand the runner receipt to bind every requested component manifest and
      selector result.
- [x] Run focused real-Zarr success and post-selector failure/rollback tests
      outside the sandbox.
- [ ] Run end-to-end component-family workflow tests outside the sandbox.

Until the remaining items pass, the existing
`reject_unsealed_chaser_derived_publication()` quarantine stays in place and
production component selectors remain unchanged.
