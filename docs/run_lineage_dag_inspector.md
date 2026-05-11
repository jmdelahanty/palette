# Run Lineage DAG Inspector

<!-- design-meta
status: implemented
last_updated: 2026-05-11
-->

## Purpose

Palette stores one recording's authoritative arrays and derived analysis runs in
that recording's Zarr archive. Many derived runs depend on multiple upstream
runs, and the same upstream run can feed multiple downstream products. That
means run lineage is a directed acyclic graph (DAG), not a strict tree.

The run-lineage DAG inspector is a read-only debugging and provenance tool for
showing those links.

## Canonical Model

The canonical representation is a pair of tables:

```text
nodes
  node_id
  path
  family
  run_id
  exists
  schema_id
  schema_version
  method
  method_version
  lineage_hash
  fingerprint_status
  latest_parent_path
  latest_run_id
  is_latest

edges
  source_node_id
  target_node_id
  edge_key
  source_path
  target_path
  status
  message
  expected_fingerprint
  actual_fingerprint
  actual_fingerprint_status
```

Edge direction is always:

```text
source_node -> target_node
```

For example:

```text
analysis/track_kinematics_runs/offline/tk_... -> analysis/swim_bout_runs/bouts_...
analysis/swim_bout_runs/bouts_...             -> analysis/bout_kinematics_runs/bk_...
analysis/eye_angle_runs/eye_...               -> analysis/bout_kinematics_runs/bk_...
```

This direction keeps topological order intuitive: upstream inputs come before
downstream derived products.

## Rendered Views

Text trees, Mermaid, and DOT are generated views of the same node/edge tables.
They are not separate sources of truth.

Supported CLI formats:

```bash
scripts/py -m fisheye.utils.inspect_run_lineage_graph <archive>.zarr --format tree
scripts/py -m fisheye.utils.inspect_run_lineage_graph <archive>.zarr --format json
scripts/py -m fisheye.utils.inspect_run_lineage_graph <archive>.zarr --format mermaid
scripts/py -m fisheye.utils.inspect_run_lineage_graph <archive>.zarr --format dot
```

To inspect a single downstream run and its upstream dependencies:

```bash
scripts/py -m fisheye.utils.inspect_run_lineage_graph <archive>.zarr \
  --root analysis/bout_kinematics_runs/<run> \
  --format tree
```

When `--root` is omitted, the inspector discovers all known derived analysis
runs listed in `audit_analysis_staleness.RUN_PARENT_SPECS`. `--run-family` can
limit that root set.

## Source Resolution

The inspector reuses the same source resolver as
`fisheye.utils.audit_analysis_staleness`:

- explicit `source_refs` are preferred;
- legacy/common `source_*_run` attrs are mapped to known parent groups;
- compact table references such as
  `tables/bouts?candidate_id=0&signal_id=4` keep the table path as edge
  metadata but collapse the graph node to the owning run;
- direct metadata-file fallback is used when normal Zarr group listings are
  stale but `zarr.json` or `.zattrs` exists on disk.

This avoids having a separate lineage interpretation for graph rendering.

## DAG Versus Tree

The tree output is a human projection of a DAG. If the same upstream run appears
under multiple downstream roots, the first occurrence is expanded and later
occurrences are marked as already shown.

This is intentional. Repeating shared nodes as if they were independent parents
would hide important reuse and make stale-source diagnosis harder.

## Status Semantics

Edge status is inherited from `audit_source_ref`:

- `fresh`: source resolves and fingerprint checks match.
- `source_not_latest`: source exists but does not match the parent group's
  `latest` pointer.
- `unverifiable_missing_expected_fingerprint`: target run did not record an
  expected source fingerprint.
- `unverifiable_missing_actual_fingerprint`: source run does not expose a
  current run-level fingerprint.
- `missing_source`: source path does not resolve in the archive.
- `source_explicit_stale`: source carries explicit stale attrs.
- `stale`: source fingerprint mismatch, or non-latest source when
  `--require-latest-sources` is set.

The inspector is diagnostic only. It does not mutate Zarr archives, registry
state, manifests, or export tables.

## Future Integrations

The node/edge JSON can later be reused by:

- Marimo lineage panels;
- Crimson run-dependency views;
- virtual collection manifest validation;
- export-lake provenance sidecars;
- Graphviz-rendered SVG/PNG artifacts for reports.

Those consumers should read the canonical node/edge model rather than parsing
Mermaid or DOT text.
