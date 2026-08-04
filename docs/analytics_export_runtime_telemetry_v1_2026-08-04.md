# Analytics export runtime telemetry v1

Date: 2026-08-04

Status: implemented operational measurement contract; no selector, registry,
physical profile, or scientific authority change.

## Decision

The four exact opt-in query exporters return one process-local
`palette.analytics_export.runtime_telemetry` v1 record. The record is never
written into the immutable export manifest and therefore never enters source
bindings, content digests, scientific identity, registry projection, or
selector state.

The phase order is closed and non-overlapping:

1. `source_binding_before`;
2. `scratch_parquet_write`;
3. `source_binding_after`;
4. `scratch_to_staging_copy`;
5. `staged_decoded_validation`;
6. `manifest_validation`;
7. `publication_staged_validation`;
8. `publication_generation_rename`;
9. `publication_manifest_commit`; and
10. `published_payload_validation`.

The recorder rejects unknown, nested, duplicate, omitted, or reordered phases.
It reports monotonic phase seconds, their exact sum, total wall elapsed time,
and unmeasured orchestration overhead. Its returned record is intended to be
combined with the existing process-tree sampler for CPU, RSS, process, and
thread telemetry.

## Interpretation

`scratch_parquet_write` includes bounded scientific projection and Parquet
encoding on node-local scratch. `scratch_to_staging_copy` includes copy and
source/destination digest comparison. `staged_decoded_validation` reopens the
staged Parquet bytes. The three publication phases are measured inside the
shared manifest-exclusive publisher, so its staged physical validation,
generation rename, and manifest compare-and-swap are not conflated.

The final phase reloads only manifest-selected published parts and performs the
table-specific full decoded validation. External benchmark processes must
measure any additional independent validator or reader workloads separately.

## Validation

- All four publisher results validate against the exact telemetry schema.
- Their persisted manifests are asserted not to contain `runtime_telemetry`.
- Atomic-publication callers that do not request telemetry retain their
  existing API and behavior.
- The focused four-export, telemetry, and atomic-publication matrix passes
  110/110 tests.

This instrumentation is a prerequisite for the representative short- and
full-duration benchmark matrix. It is not performance evidence by itself.
