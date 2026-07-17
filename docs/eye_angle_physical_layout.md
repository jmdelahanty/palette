# Eye-angle physical column layout

Palette's compact eye-angle contract remains one comprehensive two-dimensional
`roi_angles` and `frame_angles` matrix plus the authoritative
`angle_channel_index`. Consumers must resolve columns by
`angle_channel_index/name`; a numeric column index has no scientific meaning and
must not be persisted as an analysis contract.

The matrix is intentionally comprehensive. It preserves raw, smoothed,
delta, derivative, compatibility, and alternative-representation channels in
one immutable scientific product. The physical layout is optimized for the
more common interactive operation: reading a few named series over a bounded
time window.

## Production profile

New compact eye-angle runs use
`palette.eye_angle_semantic_column_order.v1` with profile
`semantic_bundles_v1`:

- related left-eye, right-eye, and binocular/vergence channels are adjacent;
- raw, smoothed, delta, and smoothed-delta triplets are kept together where
  available;
- remaining support, compatibility, version, and derivative channels fill the
  unused positions deterministically;
- `roi_angles` and `frame_angles` use inner chunks of approximately
  `(2048 rows, 8 columns)`;
- materialized production runs use outer shards of approximately
  `(131072 rows, 32 columns)`.

The exact edge chunks are bounded by array shape, and outer shard dimensions
are rounded upward to the inner-chunk grid. Both requested and effective grids
are recorded in `physical_storage_layout`, `node_local_materialization`, and
the materializer report. Other arrays in an eye-angle run retain their normal
layout.

This layout means a three-series plot typically touches one 8-column chunk per
row block instead of decompressing all 141 columns. It does not make the
canonical matrix a user-facing table: the name index and the guided
eye-angle catalog remain the interface.

## Safe writing

The shared sharded publisher copies one complete, non-overlapping outer-row
band per array task. A task owns all column shards in its row band, so no two
workers write the same physical chunk or shard. Every copied row band is read
back and compared by decoded SHA-256 before publication.

## Benchmark

The read-only benchmark creates disposable candidates outside the source
archive and compares the previous all-column layout with 8-column and
16-column semantic layouts:

```bash
scripts/py -m fisheye.diagnostics.benchmark_eye_angle_column_layout \
  /path/to/recording_analysis.zarr \
  --output-root /tmp/eye-angle-column-layout-benchmark
```

It reports physical file/byte counts, warm local read medians, estimated inner
chunk bytes decoded for three-channel, six-channel, and bounded full-table
workloads. It also compares three named channels across the complete bounded
duration against the complete table and checks exact decoded-value equality
after resolving every column by name. The source Zarr is always opened
read-only.
