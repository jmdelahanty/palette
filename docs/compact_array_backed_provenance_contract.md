# Compact Array-Backed Provenance Contract

<!-- contract-meta
version: 1
status: active
implementation: implemented
last_verified: 2026-08-20
decision_date: 2026-08-20
scope: provider-aware selection-trajectory-occupancy-contrast chain
-->

Palette provenance must be both human-interpretable and mechanically
verifiable. A digest alone is not an explanation, and copying an upstream
manifest into every downstream attribute is not durable provenance.

The compact contract separates four concerns:

1. Metadata explains the scientific meaning: method and policy IDs, formulas,
   parameters, units, coordinate and timing authorities, source roles, counts,
   and immutable source paths.
2. Typed Zarr arrays hold values that scale with frames, rows, observations,
   occurrences, grid edges, or resolved membership.
3. Array references state the exact path, dtype, shape, and content digest.
4. Manifest and source digests verify that the named records and arrays have
   not changed. They supplement the readable record; they never replace it.

## Normative metadata boundary

The following belong in arrays, not attributes or recursively embedded source
manifests:

- frame and row identities;
- `instance_key`, track ID, and acquisition-frame vectors;
- per-row validity and reason-code vectors;
- resolved frame membership and occurrence identities;
- per-occurrence result or conservation vectors;
- grid-edge and image-like numeric arrays; and
- any other value whose serialized size grows with observations, frames,
  rows, occurrences, or pixels.

Metadata retains fixed-schema codebooks and summaries. For example, a failure
reason codebook and per-code counts are readable fixed provenance, while one
reason string per row is array data. A contrast retains the explicit formula
`treatment.occupancy_fraction - baseline.occupancy_fraction`, arm roles,
normalization, grid bounds and bin shape, while exact grid edges remain the
referenced `x_edges` and `y_edges` arrays.

An array reference must identify:

```json
{
  "array_path": "selection/acquisition_frame",
  "dtype": "<i8",
  "shape": [60000],
  "content_sha256": "..."
}
```

The path and declaration make the evidence discoverable and interpretable.
The digest makes it tamper-evident. Neither is sufficient without the other.

## Source-chain rule

A downstream run references an immutable upstream run by its canonical path,
schema identity, manifest digest, relevant array references, and a compact
readable scientific subrecord. It must not copy the upstream manifest payload
or its row-level evidence into its own attributes.

To explain a figure, a consumer traverses this chain:

```text
figure recipe
  -> contrast run and explicit formula
  -> occupancy runs and denominator/grid/timing policies
  -> trajectory run and provider/selection/transform policies
  -> selection, position, tracking, geometry, and timing source runs
  -> exact typed arrays
```

Each hop is independently readable and independently digest-bound. This keeps
the provenance complete without making every downstream node a recursive copy
of the entire graph.

## Provider-spatial v2 implementation

New writers use compact v2 records:

- composable stimulus selections store exact requested-selection and timeline-
  authority JSON as `uint8` arrays, with normalized resolved selection arrays;
- provider-track evidence stores source array declarations, a fixed reason
  codebook, and reason counts instead of row-sized reason lists;
- trajectories retain compact selection authority and exact selection-array
  references instead of embedding every selected frame and membership;
- occupancy conservation names the authoritative arrays and invariant formulas
  instead of embedding one value per occurrence; and
- contrasts retain readable source-arm, formula, policy, grid-summary, and
  source-run records while referencing occurrence and edge arrays instead of
  embedding upstream occupancy manifests.

`require_cardinality_independent_metadata` is a publication-time structural
guard for these schemas. It rejects NumPy arrays and schema-declared
cardinality or recursive-payload fields in metadata. Exact schema validation
remains the primary boundary; this guard is an additional fail-closed check.
A universal metadata byte ceiling is intentionally not the scientific rule,
although publication tooling may add a separate operational size budget later.

## Compatibility and migration

Existing immutable v1 runs are not rewritten. Readers that already support a
v1 representation retain that compatibility explicitly. New publications use
v2 compact records. If a clean consolidated root is required, publish v2
successors into a clean archive or metadata generation and validate that final
consolidated generation before selecting it.

Deleting fields from the existing 2026-08-19 canary root is not part of this
contract. That canary remains evidence of the old publication shape until an
explicit successor/extraction decision is made.
