# Group analytics viewer clustered-inference readiness — 2026-08-10

Status: implementation checkpoint; selector and production state unchanged.

## Consumer boundary

Base-export discovery now calls the shared full analytics payload validator
before an export can enter the selectable catalog. Selection therefore depends
on the complete publication inventory, payload digests, exact Arrow schemas,
row counts, table contracts, capabilities, and registry-identity envelope. A
missing part is rejected rather than offered as an inspectable but incomplete
dataset. Historical non-exclusive publications remain available only through
their existing explicit legacy paths; this change does not make legacy layouts
current.

## Inference presentation policy

The API and both viewer implementations use
`prefer_computed_clustered_v1`:

1. When the declared session-clustered analysis has status `computed` or
   `boundary_zero_variance`, its estimate, confidence interval, p-value, and
   q-value are the displayed inferential result.
2. When session clustering was requested but is unavailable, the displayed
   inferential fields are empty. The naïve result remains available only in
   explicitly labelled diagnostic columns, accompanied by the cluster reason.
3. When the publication explicitly declares `cluster_mode=none` and
   `cluster_status=disabled`, the naïve result may be displayed, labelled
   `Naïve inference (clustering disabled)`.
4. Statistics admitted through the opt-in legacy adapter are labelled
   `Legacy naïve inference`; they are never described as clustered.

The projection carries the cluster mode, unit, method, status, reason, session
and unit counts, clustered estimate/standard error/interval/p/q, session and
residual variances, and intraclass correlation. The original naïve fields are
also retained under explicit `naive_*` names.

## Validation coverage

- Full-payload catalog rejection for missing parts and payload digest changes.
- Full-payload catalog rejection for missing or semantically tampered
  `registry_identity` envelopes.
- Query-level computed, unavailable, and explicitly disabled cluster cases.
- Marimo warning/success presentation behavior.
- JavaScript syntax and displayed/naïve column separation.

## Integration dependency

This lane is based on Palette `ed905617` and intentionally does not alter the
analytics export or Arrow schema versions. It must be reconciled with the
parallel export-schema versioning lane before integration. In particular, the
shared validator called by catalog discovery must dispatch the installed
current contract and any explicitly supported historical contract without
weakening the viewer's fail-closed selection rule.
