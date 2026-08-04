# Analytics Query-Export Benchmark Execution — 2026-08-04

Status: active execution record. The first full-duration kinematics attempt
exposed and fixed a controller validation defect; no matrix result was
accepted. A new immutable request and namespace are required for the rerun.

## Failed V1 Attempt

Request digest:
`d81664b497c2cbd3100f8a25e59587f9d6522167e3a3742a01384f1ec9bce910`.

Immutable evidence root:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
analytics_query_exports/20260804_full_duration_v1/evidence/kinematics
```

The maintained kinematics publisher completed successfully and committed its
benchmark-only manifest-selected Parquet generation. Its ten measured phases
totalled 19.859 seconds, including 8.341 seconds for scratch Parquet writing,
4.999 seconds for staged decoded validation, and 5.360 seconds for published
payload validation.

The matrix controller then failed before read trials and before writing a
matrix result. `publication_result.json` was serialized with canonical sorted
JSON keys, which alphabetized the nested `phases_seconds` object. The runtime
validator incorrectly treated JSON object member order as the phase sequence
even though the contract already carries a required ordered `phase_order`
array. The publication itself reported the exact required phase set and order.

This attempt is retained as failure evidence. Its export generation and
evidence directory must not be deleted, rewritten, reused as a completed
matrix, or cited as promotion evidence. The unused v1 activity request was
never executed.

## Correction

Runtime telemetry validation now:

- requires the exact phase-name set;
- requires the exact explicit `phase_order` array;
- reads durations in that declared order when reconciling totals; and
- does not assign semantics to JSON object member order.

A regression test performs a sorted-JSON encode/decode round trip before
validation. The complete four-export telemetry and benchmark gate passes
69/69 tests.

## Rerun Boundary

The rerun must use a new versioned benchmark namespace, evidence directory,
export-run ID, and request digest. It must execute from a clean commit that
contains the correction. The ordinary source-metadata before/after guard,
fresh-process read trials, and `promotion_authorized=false` requirements
remain unchanged.

