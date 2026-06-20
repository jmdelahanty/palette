# Subject-mask finalization: current parallelization architecture

**Date:** 2026-06-20
**Status:** Read-only documentation of what exists today + a brief for a future
implementation agent. No behavior change.
**Scope note:** `finalize_subject_masks.py` is ~125 KB and actively edited; line numbers
below were confirmed on 2026-06-20 but anchor on symbol names — re-confirm with `grep -n`
before relying on a line number.

## Why this exists

We set out to find "avenues to parallelize subject-mask finalization with Dask." Read-only
exploration showed the parallelization is **largely already implemented**: Dask is an
orchestrator, there are three execution backends, the batch path runs parallel by default,
and cross-recording throughput is handled by LSF array jobs. This note records that design
so it isn't re-derived, and stages a precise brief for the one un-built dimension
(component-axis) should it ever be worth pursuing.

## 1. Current architecture (what exists)

### Execution backends
`finalize_subject_masks.py:119` — `_EXECUTION_BACKENDS = ("serial_driver", "dask_worker_chunks", "process_shards")`.

- **`dask_worker_chunks`** — one `delayed(_process_and_write_finalizer_chunk)` per row-chunk,
  computed by `_compute_finalizer_dask_tasks` (`:2933`). Scheduler ∈
  `single-threaded | threads | processes | distributed` (`_SUPPORTED_SCHEDULERS`, `:118`);
  the `distributed` path builds a `LocalCluster` + `Client`.
- **`process_shards`** — `ProcessPoolExecutor`; `_row_chunk_shards` (`:991`) partitions the
  row-chunks into contiguous per-worker shards, each run by `_process_and_write_finalizer_shard`
  via `_compute_finalizer_process_shards` (`:3017`).
- **`serial_driver`** — serial fallback loop.

Entry point: `finalize_subject_mask_run` (`:3092`). The per-chunk worker
`_process_and_write_finalizer_chunk` (`:2890` → `_process_and_write_finalizer_chunk_open`, `:2754`)
processes **all** `required_raw_components` **serially within one task**.

### Defaults — how parallelism is actually turned on
- `finalize_subject_mask_run(...)` defaults: `execution_backend=serial_driver`,
  `scheduler=single-threaded`, `num_workers=None`. **Direct CLI use is serial.**
- The batch path makes it **parallel by default** (`run_subject_mask_batch_pipeline.py`):
  `--finalize-execution-backend process_shards` (`:929`), `--finalize-scheduler processes`
  (`:933`), `--finalize-num-workers 8` (`:934`), `--finalize-chunk-size 64` (`:927`).

### Cross-recording parallelism — already solved, at the cluster level
The batch driver loops recordings **serially**, but `scripts/submit_subject_mask_batches_bsub.sh`
fans finalization out as an **LSF array job, one recording per task**:
`sm_finalize[1-N]%${FINALIZE_MAX_ACTIVE}` (`:732`), submitted dependent on the inference array
(`:798`). So cross-recording throughput is the scheduler's job, not in-process.

### Write-safety and the row-axis cap
`shared/subject_mask_chunks.py`: `SUBJECT_MASK_STORAGE_ROW_CHUNK = 16` (`:10`),
`SUBJECT_MASK_METRIC_ROW_CHUNK = 256` (`:11`). `refined_subject_mask_dask_worker_row_chunk`
(`:53`) clamps the worker row-chunk to **integer multiples of 256** so concurrent workers can
never split a metric chunk (race-free writes).

Consequences:
- **Row-axis parallel width = `ceil(total_rows / 256)`.** A run with ≤256 ROIs yields exactly
  one task — the row axis cannot parallelize it.
- masks_roi and the metric arrays are chunked **size-1 on the component axis**, so
  component-parallel writes touch distinct chunks and are **safe by construction**.

### Per-component compute (the cost floor)
`subject_mask_finalization.py:128` `finalize_component_mask` — per (ROI, component):
`cv2.connectedComponents` (×4–6), `cv2.morphologyEx` (close), `scipy.ndimage.binary_fill_holes`,
largest-component selection. This per-row Python+cv2 work is the real cost; `subject_body`
(large blobby mask) dominates over `eyes_union` / `swim_bladder`.

### Eye coupling
`eyes_union` is finalized, then `_assign_finalized_eyes_union_rows` (`:819`) derives
`eye_left`/`eye_right` from the finalized union + keypoints for the same row range, **inside
the same task**. So the independent unit is the *raw* component (3 of them), with eyes carried
as the union(+L/R) bundle — not 5 independent outputs.

### Provenance and tests
- `dask_metadata` (`:2510`, `:3161`) records execution_backend, scheduler, num_workers,
  requested-vs-aligned chunk size, chunk alignment, and dask version; merged into the summary,
  run attrs, and provenance_inputs.
- `tests/unit/fisheye/test_finalize_subject_masks.py` has cross-backend parity tests:
  `..._dask_worker_chunks_writes_disjoint_rows` (`:867`) and
  `..._process_shards_writes_disjoint_rows` (`:910`), using a 2-row fixture with `chunk_size=1`.

## 2. The one un-built dimension (component-axis / "avenue #2")

Row-axis (within-run) and cross-run (cluster) are done. The only unexploited parallel axis is
`(row_chunk × raw_component)`. It is **bounded** (~1.5–2×, skewed by `subject_body` dominating
wall-clock) and **only helps ≤256-ROI runs** — exactly the corner where the 256-row clamp pins
the row axis to a single task. Component writes are chunk-safe, so it is the *only* safe
parallel axis available in that corner. Whether it's worth building depends on how common
≤256-ROI runs are, which has not been measured.

## 3. Brief for a future implementation agent (gated on measurement — not a build order)

1. **Measure first.** Histogram `total_rows` (= `masks_roi.shape[0]`) across `subject_mask_runs`
   in the registry / zarr stores. If few runs are ≤256 ROIs, **stop** — avenue #2 buys little.
   (Read-only: registry query or zarr scan.)
2. **Confirm the eye-dependency seam.** Verify `_assign_finalized_eyes_union_rows` (`:819`)
   needs only the finalized `eyes_union` batch + same-row keypoints. Prefer keeping
   `eyes_union`(+L/R) as one task (parallel width = 3 raw groups) over a delayed-dependency DAG
   — simpler, preserves correctness.
3. **Map the three change sites; confirm no regression of existing backends:**
   - dask task list (currently one `delayed` per row-chunk, in `finalize_subject_mask_run`) →
     one per `(row-chunk, component-group)`.
   - `process_shards` shard formation (`_row_chunk_shards`, `:991`) → 2-D
     `(chunk, component-group)` shards.
   - result-merge loop (review_counts, `reason_labels_by_component[...][row_slice]`,
     `eyes_union_assignment_summary`) → group results by `chunk_index` first, then per component.
4. **Make it adaptive.** Only expand the component axis when `row_chunks < num_workers`;
   otherwise the row axis already saturates cores and extra tasks add IO/scheduling overhead
   (each re-opens zarr and re-reads a component slice).
5. **Parity is the safety net.** New partitioning must produce byte-identical refined outputs
   vs `serial_driver`; extend the parity tests (`:867`, `:910`) to cover a single-row-chunk run
   exercised on the component axis.
6. **Extend provenance.** Add the component-partition mode to `dask_metadata` so runs record it.
7. **Name the non-Dask floor.** Per-row cv2/scipy ops are the real cost; cross-axis Dask only
   fills cores. Vectorizing/batching the morphology (or GPU) is a separate, larger lever and
   should not be conflated with this parallelization work.

## Critical files

- `src/fisheye/refinement/finalize_subject_masks.py` — backends, task build, worker, merge, provenance, argparse.
- `src/fisheye/refinement/subject_mask_finalization.py` — `finalize_component_mask` per-component compute (`:128`).
- `src/fisheye/shared/subject_mask_chunks.py` — chunk sizes + the 256-row worker clamp.
- `src/fisheye/utils/run_subject_mask_batch_pipeline.py` — batch defaults (parallel-by-default).
- `scripts/submit_subject_mask_batches_bsub.sh` — cross-run LSF array-job fan-out.
- `tests/unit/fisheye/test_finalize_subject_masks.py` — backend parity tests.
