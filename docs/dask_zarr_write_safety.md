# Dask Zarr Write Safety

## Rule

Parallel Zarr writes are safe only when each worker writes whole,
non-overlapping physical Zarr chunks for every array it mutates.

Disjoint logical slices are not enough. If two workers write different row
ranges inside the same physical Zarr chunk, the writes can race because partial
chunk writes are effectively read-modify-write operations.

## Failure Mode

This pattern is unsafe:

```text
metric array physical chunks: rows 0-255, 256-511, ...
Dask worker A writes rows 0-63
Dask worker B writes rows 64-127
```

Even though the row ranges are logically disjoint, both workers touch physical
chunk `0-255`:

```text
worker A reads chunk 0-255
worker B reads chunk 0-255
worker A modifies rows 0-63 and writes chunk 0-255
worker B modifies rows 64-127 and writes its stale copy of chunk 0-255
```

The second write can overwrite the first worker's changes with stale values.
This was observed in subject-mask finalization metric refresh: mask edits were
visible after reopen, but Dask-refreshed metric arrays could retain stale area
values because worker row chunks were smaller than the metric-array Zarr row
chunk.

## Required Design Check

Before adding or modifying a Dask-backed Zarr writer, check:

- Which arrays are written by each worker?
- What are the physical Zarr chunks for each written array?
- Can two workers write into the same physical chunk of any output array?
- If worker chunks are adjusted for safety, are both requested and effective
  chunks recorded in provenance?

## Safe Patterns

Use one of these patterns:

- Align worker regions to the physical Zarr chunk grid of all arrays being
  written.
- Serialize writes for arrays that cannot be chunk-aligned safely.
- Write per-worker temporary outputs, then merge into the canonical arrays in a
  deterministic single-writer pass.
- If arrays have incompatible physical chunk grids, align to a common safe grid
  or separate compute parallelism from write serialization.

## Writer Ownership Contract

Every Dask-backed Zarr writer should make writer ownership explicit. The safe
default is:

- The driver creates the target run group, arrays, attrs, provenance, `latest`
  pointers, and consolidated metadata.
- Workers compute independent slices and either return arrays to the driver or
  write only full physical chunks that they exclusively own.
- Workers do not create groups, delete groups, update attrs, update `latest`, or
  consolidate metadata.
- The driver performs final validation after workers finish and only then marks
  the run complete.

This is stricter than "Dask arrays can write to Zarr." It is the Palette
contract for mutable analysis stores. Dask may own compute parallelism; Palette
must still own schema mutation and provenance finalization.

For new stages, prefer this sequence:

1. Implement a serial writer first.
2. Add archive-level parallelism for independent recordings.
3. Add internal Dask only after the output row/chunk contract is explicit.
4. Prove the Dask path with tests that exercise misaligned requested chunk
   sizes and verify the effective worker chunks are safe.

## Network Filesystem And Scratch Strategy

Zarr is friendly to distributed analysis when reads and writes are planned
around chunks. It is not automatically friendly to high-concurrency metadata
mutation or many small partial-chunk writes on NFS.

For cluster execution, the safest high-throughput pattern is:

1. Copy or stage the source recording/Zarr to node-local NVMe/SSD scratch when
   practical.
2. Run compute-heavy stages against scratch.
3. Write complete output runs on scratch.
4. Validate the scratch result, including strict JSON metadata checks when
   attrs were written.
5. Transfer the completed run back to the canonical analysis store with a
   single controlled merge/copy step.
6. Update `latest` and consolidated metadata only after the copied run is
   complete.

If a full archive copy is too expensive, use per-run staging: write the new
analysis run to a temporary scratch Zarr or directory, then copy that completed
run group into the canonical store. Avoid many workers writing partial chunks
directly over NFS unless chunk ownership is guaranteed and the filesystem has
been benchmarked for that pattern.

Tarballs can be useful for network transfer because they collapse many small
chunk files into one sequential transfer. Treat tarball transfer as a transport
optimization, not as the canonical storage format: unpack or merge into the
canonical Zarr only after validation.

## Chunked Writes And Read-Modify-Write

Partial chunk writes can incur read-modify-write even for ordinary chunked
arrays. Compression makes this unavoidable because the codec works on whole
chunks, but the same practical issue can still apply without compression: the
store object is the chunk, so updating only part of that object often requires
reading the existing chunk, patching the requested slice, and writing the chunk
object back.

The safe rule is therefore independent of compression:

- Full-chunk writes by one owner are safe.
- Partial-chunk writes by one serialized writer are safe but may be slower.
- Partial-chunk writes by multiple workers into the same chunk are unsafe.
- Shards add another ownership layer; if sharding is enabled, workers must not
  race on the same physical shard either.

## Stage-Level Guidance

Do not add Dask to a stage simply because the stage loops over frames, bouts, or
windows. The stage must have both a compute partition and a write partition that
are safe.

Recommended order for downstream movement analysis:

- `eye_angle_analysis`: already has a Dask worker-chunk backend. Keep worker
  chunks aligned to output chunks and record scheduler/worker provenance.
- `bout_kinematics`: best future candidate for internal Dask because per-bout
  rows are mostly independent. Prefer partitioned per-bout computation followed
  by chunk-aligned writes or a deterministic single-writer merge.
- `tail_posture_view_runs`: also a good future candidate because each ROI row
  is independent. Use a single setup/finalize writer and worker chunks aligned
  exactly to output row chunks.
- `track_kinematics`: only parallelize internally after defining boundary state
  for hysteresis, smoothing, speed derivatives, and heading derivatives.
- `detect_bouts_multi_level`: only parallelize internally after defining how
  peak events, threshold components, and gap merges are reconciled across chunk
  boundaries.

On the cluster, prefer recording-level parallelism before internal stage-level
Dask. Independent recordings write independent Zarr stores, which is usually
safer and easier to schedule than multiple workers mutating one store. If both
layers are enabled, cap the product of `recording_jobs * internal_workers` to
what the filesystem and node can sustain.

## Deferred Design: Tail Posture View Dask Backend

`fisheye.analysis.tail_posture_view_runs` is currently a serial NumPy writer:
it reads the source subject-shape tail arrays, computes all output rows, then
writes final arrays. The computation is row-local, so a Dask backend is
straightforward if the writer follows the chunk-safety rules above.

Safe implementation shape:

1. Resolve the source `analysis/subject_shape_runs/<run>` and target
   `analysis/tail_posture_view_runs/<run>` in the driver.
2. Create the target run group and all output arrays once in the driver.
3. Choose `effective_worker_chunk_size` equal to the physical row chunk size
   used by every row-aligned output array.
4. Build worker slices on that exact row-chunk grid.
5. Each worker reads only its source row slice from `tail_sample_s`,
   `tail_sample_xy`, `head_xy`, `tail_sample_valid`, `bspline_valid`, and
   optional failure-reason arrays.
6. Each worker computes that slice's `valid`, `failure_reason_bytes`,
   `head_xy`, `head_yaw_rad`, `tail_keypoints_xy`, `tail_angle_rad`, and
   `tail_angle_deg`.
7. Each worker writes only the matching full row-chunk slice of each output
   array.
8. The driver waits for all workers, aggregates summary counts, writes
   attrs/provenance, updates `latest`, and optionally consolidates metadata.

Do not let workers create groups, mutate attrs, update `latest`, delete target
runs, or consolidate metadata.

Required provenance for this backend:

- `execution_backend = "dask_worker_chunks"`
- `scheduler`
- `num_workers`
- `requested_chunk_size`
- `effective_worker_chunk_size`
- `chunk_alignment_policy = "align_to_output_row_chunks"`
- `chunk_count`
- optional per-chunk timing summaries

The serial backend should remain the default until the Dask backend has tests
that prove worker slices map one-to-one to physical output chunks.

## Provenance

When runtime chunking differs from requested chunking, provenance should store
both values. Use explicit names such as:

- `requested_chunk_size`
- `effective_worker_chunk_size`
- `chunk_alignment_policy`
- backend-specific aliases already used by the writer, such as
  `dask_requested_chunk_size`, `dask_chunk_size`, `worker_chunk_size`, and
  `dask_chunk_alignment`
- `scheduler`
- `num_workers`
- `worker_count_requested`
- `worker_count_effective`
- `host`
- `palette_git_commit`
- `started_at_utc`
- `finished_at_utc`
- `wall_time_s`
- optional per-chunk timing summaries, such as `chunk_compute_time_s_sum` and
  `chunk_write_time_s_sum`

This makes downstream debugging possible without guessing whether observed
outputs came from requested chunking or from a safety-aligned write plan.
