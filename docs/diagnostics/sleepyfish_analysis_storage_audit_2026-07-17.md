# Sleepyfish completed-analysis storage audit (2026-07-17)

## Question

After sharding the current subject-shape, tail-kinematics,
track-kinematics, swim-bout, and eye-angle products, which completed analysis
writer is the next useful storage/publication target?

## Method

The audit resolved only authoritative `latest_complete` pointers under
`analysis/*_runs`. It read `zarr.json` metadata and statted payload files; it
did not decode array values or write to the source archive. Root, family, and
selected-run metadata hashes matched before and after the pass.

```bash
scripts/py -m fisheye.diagnostics.audit_analysis_storage_candidates \
  /groups/johnson/johnsonlab/jeremy/recordings/\
sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/\
sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr \
  --include-family '^analysis/' \
  --output-json /tmp/sleepyfish_analysis_storage_audit_20260717.json \
  --output-markdown /tmp/sleepyfish_analysis_storage_audit_20260717.md
```

The six selected runs contain 520 arrays, 3,236 measured payload files,
6.81 GiB logical data, and 3.84 GiB physical payload data.

| Payload-file rank | Latest completed run | Arrays sharded | Payload files | Physical | Logical |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `bout_kinematics_sleepyfish_core_canary_20260713_01` | 0 / 115 | 1,359 | 12.3 MiB | 44.3 MiB |
| 2 | `subject_shape_materializer_canary_20260715_01` | 102 / 104 | 883 | 2.47 GiB | 4.34 GiB |
| 3 | `tk_sleepyfish_sharded_canary_20260716_02` | 102 / 102 | 478 | 133.3 MiB | 416.4 MiB |
| 4 | `eye_angles_sleepyfish_semantic16_20260717_01` | 39 / 39 | 253 | 937.0 MiB | 1.42 GiB |
| 5 | `swim_bouts_sleepyfish_sharded_track_smoke_20260716_02` | 100 / 133 | 142 | 39.8 MiB | 135.0 MiB |
| 6 | `tail_kinematics_hardened_w2_canary_20260715_01` | 25 / 27 | 121 | 279.5 MiB | 482.7 MiB |

## Conclusion

Bout kinematics is the next target. Its current latest pointer still selects a
pre-sharding 2026-07-13 run. It occupies the least physical space but creates
the most payload objects: 1,359 files for only 12.3 MiB, or about 9.2 KiB per
file on average. At least 260 of those files are in repeated small column
arrays. This is schema/object fanout, not a large-array compression problem.

The next controlled step should rematerialize the same bout-kinematics product
with the shared columnar sharding default, verify logical parity through the
resolver, and measure file count, bytes, read latency, and writer time before
promoting its `latest_complete` pointer. Because this writer does not yet
record the shared staged-publication evidence, the same pass should determine
whether it can use the common atomic publisher rather than adding another
publication implementation.

The current sharded track, eye, and tail products are not urgent object-count
targets. Subject shape remains the largest byte surface, but 102 of 104 arrays
are already sharded; its next question is selective-read behavior for three
wide scientific matrices, not additional blanket sharding. The current
swim-bout product is also low-object-count, although its publication strategy
is not yet recorded.

An exact all-family audit was intentionally deferred. Primary tracking,
keypoint, and refined-mask surfaces contain enough objects that statting every
payload takes minutes on PRFS; that should be a scheduled infrastructure audit,
while a metadata-only expected-object mode would be more appropriate for quick
interactive triage.

## Bout small-run benchmark and retained Zarr authority

The follow-up decision is to keep the per-recording bout product authoritative
inside the analysis Zarr. Although Parquet would be efficient for this small
table, the run-local completion state, source lineage, parameters, validation,
and FileGlancer discovery are valuable enough to retain one coherent Zarr run.
Parquet remains the regenerated cross-recording/query representation.

A read-only storage-only benchmark compared the completed source with the
production 262,144-row capped-shard profile. Both candidates preserved logical
chunks and passed path, shape, dtype, and representative decoded-value parity;
the source metadata digest was unchanged.

| Measure | Regular source layout | Capped small-run shards |
| --- | ---: | ---: |
| Payload files | 1,359 | 110 |
| Total apparent bytes | 12.52 MiB | 12.59 MiB |
| Write time | 4.15 s | 3.17 s |
| Full-table scan | 0.68 s | 0.55 s |
| Three 1,024-row windows per eligible array | 0.39 s | 0.60 s |

The object-count improvement is decisive, but the bounded-window result is a
real tradeoff rather than a universal read-speed win. The small-run profile
therefore shards only arrays spanning multiple logical row chunks, caps the
outer shard to the complete useful row grid, and leaves single-chunk metadata
and visualization arrays regular.

The production materializer publishes a new named candidate through the shared
atomic publisher and deliberately leaves `latest` and `latest_complete`
unchanged. Promotion is a separate decision after inspecting the persisted
report. Reproduce the disposable benchmark with:

```bash
scripts/py -m fisheye.diagnostics.benchmark_columnar_zarr_sharding \
  /path/to/analysis.zarr/analysis/bout_kinematics_runs/SOURCE_RUN \
  --output-root /tmp/bout-small-run-benchmark \
  --shard-rows 262144
```

## Persisted cluster trial

LSF job `153131273` ran the production materializer from commit `e852b6a7` on
`h07u20` and completed without stderr. It published the named candidate
`bout_kinematics_sleepyfish_smallrun_candidate_20260717_01`; it did not
promote that candidate.

The source and published candidate have the same resolver-level scientific
fingerprint:
`af4bce16120f1bbe2f8a426e77f26445a8766b25dcb639308bb2be4bc14654db`.
All 115 arrays passed final validation, with 104 stored as capped row-aligned
shards and 11 single-chunk arrays retained in their regular layout. The
candidate has 110 payload files and 12.6 MiB of physical data, compared with
1,359 payload files and 12.5 MiB for the source. The node-local decoded copy
processed 44.3 MiB in 2.55 seconds and performed exact decoded readback for
each outer shard before publication.

The shared publisher then copied 232 total files to a hidden PRFS sibling,
validated the temporary and final run, and atomically installed the candidate.
The parent pointers were identical before and after publication:

```text
latest          = bout_kinematics_sleepyfish_core_canary_20260713_01
latest_complete = bout_kinematics_sleepyfish_core_canary_20260713_01
```

This verifies the intended architecture: the analysis Zarr remains the
authoritative per-recording run archive, physical-layout improvements can be
published as complete named runs with full lineage, and cross-recording
Parquet exports can remain derived query products rather than competing
authorities.

## Native compute and publication canary

The validated storage candidate was promoted under the bout-family publication
lock after its resolver fingerprint was recomputed. Both pointers moved from
`bout_kinematics_sleepyfish_core_canary_20260713_01` to
`bout_kinematics_sleepyfish_smallrun_candidate_20260717_01`.

Commit `c7bbc798` then moved the workflow's native bout stage onto the same
execution model used by the other hardened materializers: authoritative inputs
are opened read-only, the writer creates its final sharded layout in a
node-local Zarr, and the shared publisher copies to a hidden PRFS sibling,
validates, atomically renames, completes, and updates the pointers under one
lock. The native writer now records its requested row-shard size and a run-level
physical-layout summary.

LSF job `153131310` exercised that path on `h07u11` with one core. It completed
successfully in 104 seconds with empty stderr and a maximum memory use of 1.5
GiB. Native computation took 37.97 seconds and atomic PRFS publication took
4.51 seconds. The published run,
`bout_kinematics_sleepyfish_native_atomic_canary_20260717_01`, contains 115
arrays (104 sharded and 11 regular), 110 payload files, and passed final
resolver validation. Its 26,565-row movement, raw-heading, and smoothed-heading
record hashes exactly match the previously validated run; the overall logical
fingerprint differs only because provenance-bearing scientific attributes now
include the native storage parameters and newly pinned source identities.

The same workflow also persisted the bounded track-kinematics visualization
contract that was missing before the canary. Fast Recording Explorer discovery
now resolves `offline/tk_sleepyfish_sharded_canary_20260716_02` through the
supported `palette-track-kinematics-summary-v1` renderer.
