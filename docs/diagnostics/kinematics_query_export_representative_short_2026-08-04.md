# Kinematics Query Export Representative-Short Gate

Date: 2026-08-04

Status: passed for the exact query-export writer/read/publication and decoded-
equality boundary; no source authority, selector, registry, physical profile,
or production default was changed or promoted.

## Scope

The benchmark uses the same full-duration Sleepyfish track authority, 10 Hz
sampling request, 131,072-row source window, 65,536-row Parquet row group, and
reader workload as the prior full-duration matrix. The only semantic change is
projection-v2's explicit acquisition-frame interval:

```text
[0, 200000)
```

The exporter still streams and rehashes every selected source surface. Its
writer timing is therefore a conservative bounded-output measurement, not an
optimistic simulation of a physically shorter source archive.

## Immutable Evidence

The first successful matrix at `18458f9b` recorded unrelated untracked
`agents_todo` files and correctly marked its Git identity dirty. It is retained
as nonpromotion development evidence and is not used below.

The accepted five-process matrix was rerun from a detached clean worktree at:

```text
Palette commit: 18458f9b67a660ba634e70f2738f3f0b722767d6
Matrix digest: 31dddaedf6630ea8b29e3ccd7efbe0c5ed2e7ee21ff2aa7f1aef9567735e7a3b
```

Evidence root:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
analytics_query_exports/
20260804_representative_short_v2_clean_18458f9b/
```

The independent v1-slice/v2 equality validator ran from clean commit
`1f67fc5979e675117cb67575c8c6665a19d5284f` and produced:

```text
equivalence/kinematics_v1_slice_vs_v2.json
payload digest: d095ebdf3a6caec1251879a26aa48e92fab7be4342030421eb31c7dce8aee079
```

The receipt binds both immutable manifests, validates both complete exports,
filters the earlier full-duration projection-v1 table to `[0, 200000)`, and
requires identical ordered decoded hashes for all 23 scientific columns. It
passed with 66,464 rows and decoded payload digest
`77ae3ea5a4dc4b37d807efa48e9244d66a703ff428945814e7365ea83f9b3e8c`.
The visible output axis is `0..199998` because the unchanged 10 Hz projection
selects every third 30 Hz acquisition frame.

## Measurements

The prior clean full-duration comparison is the matrix at commit `00296a5c`
with digest
`835559ff7634845471e300ffd76281d9bbab93bf252611218ec7d4e817e6dc14`.

| Metric | 200,000-frame window | Full duration | Interpretation |
|---|---:|---:|---|
| Export rows | 66,464 | 389,689 | Same 10 Hz selection over different temporal extents |
| Publication wall | 10.668 s | 19.367 s | Short result is 44.9% lower |
| Scratch Parquet write | 7.647 s | 7.844 s | Nearly fixed because source rehash remains full-duration |
| Median validation | 0.948 s | 5.360 s | 82.3% lower |
| Median full scan | 0.136 s | 0.651 s | 79.2% lower |
| Median random-frame p95 | 9.63 ms | 7.70 ms | 1.93 ms absolute increase; uncontrolled filesystem cache |
| Median 4,096-frame-window p95 | 8.83 ms | 6.96 ms | 1.87 ms absolute increase; uncontrolled filesystem cache |
| Publication peak process RSS | 1.069 GB | 1.710 GB | 37.5% lower |
| Apparent publication bytes | 2,226,076 | 11,385,394 | 80.4% lower |
| Publication objects | 2 | 2 | One manifest and one Parquet part at both scales |

The short publication process used 10.84 CPU-seconds, averaged 0.82 effective
CPU cores, requested 452.7 million read characters in 11,673 read-like
syscalls, and reported 2.19 MB of storage-layer reads. Requested characters
include cache-served data and are not network transfer.

All five fresh readers agreed on full-scan digest
`fde2685c38399522d1452d2d51e05a754d02ba80d470f9864202d33820afaf00`.
The 123-file selected-source metadata guard remained unchanged.

## Verdict And Remaining Gates

This passes the representative-short kinematics query-export writer,
publication, deterministic random/window/full-scan reader, source nonmutation,
and exact semantic-subset gates. It also proves the scale label is executable:
an unbounded or non-200,000-frame `representative_short` request is rejected.

It does not measure network transfer or compressed remote requests, does not
exercise Crimson, does not compare a new physical Zarr profile, and does not
authorize promotion. Those fields remain explicitly unavailable or false in
the matrix.
