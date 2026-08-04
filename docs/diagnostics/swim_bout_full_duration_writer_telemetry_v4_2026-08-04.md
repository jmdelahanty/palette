# Swim-bout full-duration writer telemetry v4 — 2026-08-04

Status: benchmark-only execution passed; not profile-promotion evidence.

## Scope

This checkpoint reran the maintained exact-tabular swim-bout candidate writer
against the disposable full-duration Sleepyfish Cam2010095 archive after
candidate execution receipt v4 began retaining the shared atomic publisher's
nested telemetry.

- Palette commit: `7808967aa508d85047467f8dc093c86b6949e0cb`
- Git state: clean detached worktree
- source run: `analysis/swim_bout_runs/swim_bouts_sleepyfish_exact_v8_eligible_source_20260804_e8f3e020`
- candidate run: `analysis/swim_bout_runs/swim_bouts_sleepyfish_published_http_v1_telemetry_v4_20260804_7808967a`
- receipt schema: `palette.analysis_candidate_execution_receipt` v4
- receipt payload digest: `e614c56a51f51612dca8ca1fdfbcf713fe72bde0cb79d65786b6f9ae9fa665e8`
- disposable receipt: `/tmp/.palette_benchmarks/derived_analytics_storage/swim_bouts_sleepyfish_full_20260804_e8f3e020/execution_telemetry_v4_7808967a/receipt.json`

The candidate remained selector-ineligible and used the explicit unpromoted
`published_http_v1` candidate role. The receipt's selector, registry, and
production-profile before/after hashes were unchanged. Exact decoded equality,
local and published direct/consolidated equivalence, coordinate lineage, and
the current receipt validator all passed.

`publication_gate_passed=false` is expected: this execution requested Linux
`/proc/self/io`, which is explicitly not a filesystem/network-transfer scope.
It is useful process sensitivity evidence but cannot promote a profile.

## Top-level writer phases

Top-level measured wall time summed to 222.967 seconds. Published validation,
decoded equality, physical inventory, and acceptance are children of atomic
publication and are therefore not double-counted.

| Phase | Wall seconds |
|---|---:|
| Plan | 0.237 |
| Source staging | 5.094 |
| Logical rematerialization | 4.054 |
| Local validation | 2.142 |
| Local consolidation | 0.128 |
| Local direct/consolidated comparison | 0.150 |
| Atomic publication | 211.163 |

Within the atomic-publication parent, caller-owned publication acceptance took
196.945 seconds, or 93.27% of the parent. Published run validation took 2.050
seconds, published direct/consolidated comparison 0.078 seconds, decoded
equality 1.415 seconds, and physical inventory 0.015 seconds.

Peak process-tree RSS was 2,209,624,064 bytes (2.21 GB) during atomic
publication/acceptance. Atomic publication consumed 233.69 user-CPU seconds
and 37.89 system-CPU seconds.

## Mechanical atomic publisher phases

Receipt v4 proves that the nested publisher trace lies within the materializer
parent and preserves its exact phase order. The important costs were:

| Atomic publisher phase | Wall seconds |
|---|---:|
| Local source validation | 1.994 |
| Physical tree copy | 0.061 |
| Hidden-target validation | 2.124 |
| Atomic rename | 0.000155 |
| Pre-pointer validation | 2.203 |
| Final run validation | 2.147 |
| Final callback (`selector_activation` telemetry name) | 201.771 |
| Atomic total excluding final callback | 9.391 |

The four publisher-owned validation passes totaled 8.469 seconds. Physical
copy wrote 34,983,936 `/proc/self/io` bytes and completed in 61 ms; rename took
0.15 ms. The final callback is named `selector_activation` by the generic
atomic-publisher telemetry, but this nonpromoting candidate used it for archive
consolidation plus acceptance and did not activate a production selector.
Caller acceptance accounts for 196.945 of its 201.771 seconds.

This localizes the next optimization target to acceptance validation and its
repeated logical proof work. Copy, rename, lock acquisition, physical
inventory, and ordinary metadata writes are not the bottleneck. Any proof
reuse must remain bound to the exact immutable payload, declarations,
coordinate evidence, and direct/consolidated metadata and must not weaken the
current fail-closed gate.

## Physical output and I/O boundary

- output files: 257 total; 144 metadata and 113 payload objects
- apparent bytes: 34,407,272
- allocated bytes: 35,053,568
- process read bytes: 4,096
- process write bytes: 149,307,392
- process read operations: 570,661
- process write operations: 2,415

The process counters do not measure cached file reads, subprocess I/O,
filesystem server transfer, SMB/HTTP ranges, or network bytes. The separate
suite-v2 Linux process-tree trace remains the current read-side file-request
sensitivity evidence, while mounted-consumer transfer remains an open gate.

