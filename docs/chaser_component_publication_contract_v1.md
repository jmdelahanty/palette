# Chaser Component Publication Contract v1

Date: 2026-08-03

Status: shared logical/atomic-publication primitives and maintained scientific
writer adoption implemented. Publications made by the scientific writers are
selector-ineligible candidates. This contract does not activate a production
selector.

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
writers use `analysis/chaser_component_writer.py` to build only in a private
node-local archive and then submit the sealed component directory to that
materializer.

The public scientific writer path never requests selector activation. The
materializer's `activate_selector=True` mode exists only for a separate,
explicit activation operation after review. Merely passing a historical
`overwrite=True` argument does not authorize replacement or activation; a
same-name immutable child fails closed.

## Maintained Writer Adoption

The following ten maintained component families use the same sealed staging
boundary:

1. `chaser_bout_response`
2. `egocentric_bearing`
3. `epoch_behavior_summary`
4. `chaser_escape_events`
5. `chaser_escape_freeze`
6. `gaze_tracking`
7. `chaser_near_field_occupancy`
8. `chaser_quadrant_occupancy`
9. `chaser_radial_occupancy`
10. `chaser_response_regimes`

Each declaration binds its existing `SCHEMA_ID`, `SCHEMA_VERSION`, `METHOD`,
and `METHOD_VERSION`. The staging boundary requires an exact canonical
run-lineage payload and binds its source refs, source fingerprints, parameters,
and lineage digest into the component contract. Egocentric bearing and epoch
behavior summary now emit the same exact lineage envelope as the other eight
writers.

The returned path remains a `str` for compatibility and carries a detached
`publication_receipt` containing component identity, manifest digest, payload
counts, final validation, and the complete atomic-publication receipt. The
published component also retains the atomic receipt in
`cluster_output_staging`. No legacy `latest` or `latest_complete` attribute is
written.

Writer receipt v2 also carries one
`palette.chaser_component_dependency_handle.v1` record. The handle binds the
exact base publication seal, component path, component manifest digest,
semantic schema, and method identity, and self-digests that closed field set.
`load_explicit_chaser_component()` accepts that handle without consulting a
selector, scanning children, or following `latest`. This is intentionally a
workflow-dependency authority only: it makes an immutable candidate usable by
the next node without making it generally selectable.

`open_explicit_chaser_component_group()` applies the same exact handle,
base-seal, manifest-digest, semantic-schema, and method validation before
returning a group opened through the caller's read-only archive root. It is the
streaming alternative for scientific consumers that should not detach every
declared array. CLI/workflow boundaries load handles through one strict JSON
loader that rejects non-object and non-finite documents before archive access.

Component-owned arrays and visualization artifacts are included in the sealed
payload. Historical run-level dashboard refreshes executed by some private
payload builders occur only in node-local staging and are not copied into the
authoritative archive. Rebuilding those compatibility dashboards is a separate
post-activation maintenance concern, not part of scientific component
authority.

The first maintained chained consumers now use this boundary:

- `chaser_bout_response` accepts an exact egocentric-bearing handle; and
- `chaser_escape_events` accepts an exact bout-response handle.

Both consumers remain fail closed when asked to rediscover a newly published,
selector-ineligible candidate through `latest`. Their result and persisted
run-lineage records bind the exact upstream component manifest digest. The
historical swim-bout fixture path is available only through the explicit
`legacy_swim_bout_compatibility` policy; the default remains the exact current
swim-bout contract.

Other component-to-component consumers and exports still require migration.
Candidate publication does not weaken their selection rules merely to preserve
historical one-process chaining.

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
- [x] Give every maintained component an exact semantic schema declaration.
- [x] Migrate all maintained component writers off direct visible mutation.
- [ ] Migrate all readers/exports to validated component handles. The exact
      builder, validator, detached/group loaders, strict JSON boundary, and
      writer-receipt binding are implemented. Bout response now consumes
      egocentric bearing explicitly, and escape events consumes bout response
      explicitly; remaining component chains and exports are still open.
- [x] Return a digest-bound writer publication receipt for every component.
- [x] Expand the cluster orchestration target receipt to bind every requested
      component manifest, explicit dependency handle, successful validation,
      and the selector-ineligible explicit-authority result. Target receipt v2
      embeds a self-digested runner receipt built by reopening the archive only
      after all requested steps finish.
- [x] Run focused real-Zarr success and post-selector failure/rollback tests
      outside the sandbox.
- [x] Run the selector-ineligible egocentric -> bout-response -> escape-event
      chain through explicit handles outside the sandbox.
- [ ] Run end-to-end activated component-family workflow tests outside the
      sandbox after separately reviewed selector activation is authorized.

Until the remaining reader/activation items pass, production component
selectors remain unchanged. The old direct-writer quarantine remains available
for legacy callers, while every maintained public scientific writer now routes
through the sealed staging capability instead of calling it.
