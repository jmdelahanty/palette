# Provider-aware chaser position-suite storage contract

Status: implemented selector-ineligible publication contract  
Schema: `palette.analysis.provider_chaser_position_suite`, version 1  
Layout: `flat_typed_tables_v1`

## Scope

This contract persists the provider-aware, position-only chaser summaries from
`sealed_provider_position_chaser_spatial_suite_v1`. It does not infer motion,
heading, body frame, bouts, gaze, trials, or escapes. Its temporal relationship
to stimulus presentation remains the explicitly recorded controller-input
provenance proxy; it must not claim exact presentation/exposure alignment.

Runs are immutable children of:

```text
analysis/provider_chaser_position_suite_runs/<exact_run_name>
```

The initial publication boundary is always selector-ineligible. Publishing a
run does not create or change `latest`, `selected`, `current`, production
authority, or registry state.

## Source authority

A publisher must recompute the suite from exact, validated inputs and bind:

- one immutable `provider_chaser_distance_runs` source and manifest digest;
- its exact position-provider, row-axis, coordinate, scale, and timing
  authorities;
- the exact resolved stimulus-epoch candidate and selection;
- explicit caller-supplied analysis-role-to-window bindings;
- one reviewed circular arena selection and its selected physical-frame scale;
- semantic equivalence between provider and recording physical frames; and
- the exact source-camera-to-arena-millimetre transform.

The publisher must retain `temporal_alignment_class =
controller_input_provenance_proxy` and `physical_presentation_verified = false`.
Missing or contradictory authority fails closed.

## Typed tables

The seven scientific row products are stored in Zarr arrays, not attributes:

1. `epoch_roles`
2. `per_epoch_chaser_metrics`
3. `distance_cdf`
4. `radial_occupancy`
5. `quadrant_joint_occupancy`
6. `role_contrasts`
7. `role_radial_contrasts`

Each logical column is a flat array named `<table>__<column>`. Every table has
an exact dense `int64` `<table>__row_index` from zero through `row_count - 1`.
All arrays belonging to a table have the same first-axis length.

Column encoding is fixed as follows:

- integers: dense `int64`, non-nullable;
- strings: dense `int32` codes into a sorted, table-column-local registry in
  the immutable manifest;
- floating point: `float64` plus a boolean `<value_array>_valid` array;
  unavailable values are represented by `valid = false` and a NaN payload;
  measured zero is therefore distinct from unavailable.

The manifest declares every logical column, value/validity array, dtype,
shape, table ownership, and one-time content digest. It records table row
counts and their total. Unknown, missing, duplicated, or misaligned arrays fail
validation.

## Metadata boundary

Run attributes contain only bounded metadata:

- storage and scientific schema identities;
- exact run and recording identities;
- selector-ineligible lifecycle state;
- table contracts and compact string registries;
- array declarations and content digests;
- scientific policies, parameters, arena, units, and caveats;
- exact source bindings and their digest; and
- writer provenance.

Row dictionaries are never copied into attributes, manifests, atomic
publication receipts, or consolidated root metadata. The immutable manifest is
limited to 256 KiB. Atomic receipts contain counts, table row counts, manifest
identity, and a digest of the array-path list; the manifest remains the single
readable declaration of those paths instead of repeating the list for every
publication phase.

## Publication and reads

The command
`fisheye.utils.materialize_provider_chaser_position_suite` returns a revealing
no-write plan unless `--apply` is supplied. Apply mode writes a complete local
run, validates it, atomically publishes it, proves parent metadata unchanged,
then consolidates the root as the final visibility step.

Ordinary source handles:

- require an exact run name, never an alias;
- validate direct/consolidated metadata equivalence;
- validate completion, schema, declarations, dtypes, shapes, registries,
  validity encoding, and selector-ineligible state;
- return read-only arrays; and
- do not re-hash upstream dense sources or ordinary output arrays.

`deep_audit_provider_chaser_position_suite_run` is the explicit maintenance
path that recomputes every declared output-array digest.

## Promotion boundary

This version intentionally defines no mutable selector and no production
default. Profile, DAG, registry-readiness, and selector activation are separate
integration decisions. A later promotion must bind an exact manifest digest,
retain provider identity, and pass required CI before becoming production
authority.
