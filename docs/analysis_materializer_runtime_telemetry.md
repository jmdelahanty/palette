# Analysis materializer runtime telemetry

Palette analysis materializers emit versioned, machine-readable runtime
telemetry in their command report. The telemetry is operational evidence, not
scientific data.

## Identity boundary

The schema is `palette.materializer_phase_telemetry`, version 1. Its identity
policy is:

```text
report_only_excluded_from_scientific_identity_and_payload_digests
```

Runtime timestamps, host names, scheduler IDs, CPU use, and timings vary from
run to run. They must not affect scientific payload digests, storage identity,
selector eligibility, or reproducibility comparisons. In particular, the
shared publisher returns telemetry to the materializer report but does not add
it to the persisted `cluster_output_staging` attribute.

## Measurements

Each phase records:

- wall time;
- process and completed-child user/system CPU time;
- average effective CPU cores;
- process and completed-child peak RSS observed by `getrusage`;
- Linux `/proc/self/io` counter deltas when available;
- success or failure and the exception type on failure.

The execution context records the host, PID, relevant LSF variables, requested
worker count, shard rows, and copy backend when those values are known.
`/proc/self/io` describes the materializer process itself; subprocess I/O is
not attributed by that counter. Whole-process sampling with
`fisheye.diagnostics.run_with_resource_telemetry` remains useful when a
benchmark needs process-tree RSS, thread count, and time-series CPU samples.
Guarded jobs rendered by `scripts/submit_analysis_workflow_bsub.sh` run through
that sampler and publish its summary, JSONL samples, and captured workflow log
beside the execution report.

## Track-kinematics phases

The track materializer reports local phases for planning, scratch creation,
numeric staging, local validation, shard materialization and decoded
validation, publishing-state transition, sharded validation, authoritative
publication, and scratch cleanup.

The nested shared-publisher report further separates:

1. local run validation;
2. publication lock wait;
3. source inventory and hashing;
4. physical tree copy;
5. rsync checksum verification;
6. target inventory and inventory comparison;
7. hidden-target ownership stamping and validation;
8. atomic rename and renamed-owner verification;
9. final-path coordinate binding;
10. pre-pointer validation;
11. completion and pointer publication;
12. final validation and pointer verification;
13. final metadata write and selector activation.

These boundaries are intended to explain why a job is slow before changing
its correctness contract.

## Evidence-led optimization order

The first Sleepyfish baseline showed approximately 1,474–1,538 seconds of LSF
wall time per track job, while numeric staging took about 113–119 seconds and
local shard creation about 3.5–3.7 seconds. Publication-related work was over
230 seconds, and much of the remaining time could not be assigned precisely by
the older coarse timers. Therefore worker-count tuning is not the first
optimization.

After collecting the new telemetry on an immutable canary, evaluate these
changes in order:

1. Perform exhaustive decoded-value validation once on the node-local output.
2. Bind that validation to an immutable payload digest and carry a versioned
   validation receipt into publication.
3. Fully validate the hidden authoritative copy before its atomic rename.
4. After rename, verify identity, metadata, ownership, pointers, and receipt
   binding instead of decoding every floating array again.
5. Avoid combining `rsync --checksum` with repeated full decoded comparisons
   when one content-bound inventory proves the same immutable transfer.
6. Retain bitwise validation for small canaries and exact discrete fields;
   apply the versioned numerical-tolerance policy to floating derived arrays.
7. Prefer references to authoritative axes over storing and revalidating
   duplicate frame axes where the storage contract permits it.

Any reduction in validation must be implemented as a new, reviewed publication
contract. Telemetry alone does not authorize skipping an existing proof step.

## Receipt-mode track publication

The guarded track-kinematics v3 publisher implements the first versioned
receipt optimization described above. See
[`zarr_payload_validation_receipt_contract.md`](zarr_payload_validation_receipt_contract.md).
Its post-rename metadata records separate durations for physical receipt
construction, canonical coordinate binding, and post-binding physical
verification. Completion still performs one exhaustive full-motion scientific
validation. Final validation uses the bound receipt, and activation freshly
rehashes the immutable physical payload before selector eligibility is exposed.

Canonical binding additionally records nested durations for receipt preflight,
initial exhaustive payload validation, attribute snapshotting, per-track
coordinate publication, post-binding physical/metadata reverification, binding
status commit, and the closing array-proof recheck. Receipt-backed binding
omits the immediate exhaustive publication reload and duplicate final staging
validator. The materializer then checks the persisted receipt identities rather
than repeating the binder's just-completed physical rehash.

## Receipt-mode Sleepyfish canary

The first production-scale receipt canary completed successfully on 2026-07-24:

- Palette commit: `6d976a57`
- LSF job: `153171881`
- host: `h07u31`
- source: cam2010095, `1,169,010` track rows
- output: `track_kinematics_sleepyfish_cam2010095_receipt_canary_20260724_v001`
- requested slots / shard workers: `8` / `8`
- application wall time: `576.0` seconds
- LSF execution time: `592` seconds
- maximum RSS: `1.7` GiB

The like-for-like authoritative publication phase fell from `1,521.7` to
`448.4` seconds: a `70.5%` reduction and approximately `3.4x` speedup.

| Publication phase | Before (s) | Receipt canary (s) |
|---|---:|---:|
| Post-rename binding | 247.3 | 240.5 |
| Completion / full-motion sealing | 646.6 | 192.1 |
| Two authoritative validations | 125.2 | 1.3 combined |
| Final selector activation / resealing | 490.6 | 1.2 |
| Receipt construction | n/a | 0.4 |
| Post-binding physical verification | n/a | 0.6 |

The baseline and canary decoded-payload roots were exactly equal:
`0d246d1df9424314bd7c9c2cd9246fb64206fb3f38961522e64447bd87bab6e3`.
This binds equal persisted logical values across all `106` arrays despite the
different immutable run names, code commits, hosts, and provenance records.
The normal exhaustive public full-motion loader also accepted the completed
canary and rebound one track with `104` derived-motion surfaces.

The canary confirms that receipt hashing did not move the bottleneck. The two
remaining dominant phases are canonical binding (`240.5` seconds) and the one
required scientific completion seal (`192.1` seconds). Eight workers were
useful for only the `3.7`-second sharded-copy phase; average LSF CPU efficiency
was `13.78%`. Future tuning should profile those two serial scientific phases
and separately reconsider the default slot count.

## Canonical-binding proof-reuse canary

A second like-for-like Sleepyfish canary validated operation-scoped payload
proof reuse and the receipt-backed binding path on 2026-07-24:

- Palette commit: `d8495219`
- materialization LSF job: `153172030` on `h07u29`
- exhaustive public-reader LSF job: `153172069` on `h07u18`
- output: `track_kinematics_sleepyfish_cam2010095_binding_canary_20260724_v001`
- source rows: `1,169,010`; requested slots / shard workers: `8` / `8`
- application wall time: `445.9` seconds
- authoritative publication: `325.1` seconds
- maximum LSF RSS: `1.9` GiB

The decoded-payload root again exactly matched the preceding canary:
`0d246d1df9424314bd7c9c2cd9246fb64206fb3f38961522e64447bd87bab6e3`.
The independent normal public loader completed successfully in `157` seconds
and accepted one track with `104` full-motion surfaces.

| Phase | Receipt canary (s) | Binding canary (s) | Reduction |
|---|---:|---:|---:|
| Total application | 576.0 | 445.9 | 22.6% |
| Authoritative publication | 448.4 | 325.1 | 27.5% |
| Post-rename canonical binding | 240.5 | 133.7 | 44.4% |
| Completion / full-motion sealing | 192.1 | 176.1 | 8.4% |

The new nested binding telemetry attributes the `133.2`-second binder body to:

| Binding subphase | Seconds |
|---|---:|
| Initial exhaustive staging-payload validation | 13.0 |
| Per-track coordinate publication | 117.0 |
| Closing array-proof reverification | 2.3 |
| Post-binding physical/metadata reverification | 0.8 |
| Receipt preflight, snapshots, and binding-status commit | 0.1 combined |

This confirms that duplicate full-array hashing and repeated final validation
were real costs, while preserving the scientific and publication proofs. It
also localizes the next optimization target: the sealed per-track coordinate
publisher itself, not receipt hashing, array-proof closure, transfer, or worker
count. Any change there should retain the freshly minted typed authorities and
the same rollback boundary rather than weakening coordinate lineage checks.
