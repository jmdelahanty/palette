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
   should not be conflated with this parallelization work. See the status table below.

### Vectorization status (current) — substantiates item 7

A repo-wide check (2026-06-20): mask compute uses a **batched-output convention** (functions
take `(N,...)`, pre-allocate arrays, write back by row-slice), but much of the expensive
topology/shape compute is still per-row Python loops over cv2/scipy. No GPU/batched-morphology
library is used for masks (`cupy` is present but only for video decode/crop/tracking, not masks).

| Operation | Vectorized over batch? | Location |
|---|---|---|
| area / mask_present | **Yes** — shared batch reduction | `shared/mask_geometry.py:74` `batch_mask_spatial_metrics` |
| centroid / bbox in finalizer | **Yes** — shared batch reduction | `refinement/finalize_subject_masks.py:1532` `_compute_component_spatial_metrics` |
| centroid / bbox in review helpers | **Yes** — shared batch reduction | `tune/refined_subject_mask_review.py:758` `_compute_geometry_metrics` |
| centroid / bbox in subject-shape | **Yes** — shared batch reduction | `analysis/subject_shape_runs.py:1223` `_compute_component_batch` |
| principal axis (PCA) | No — per-row loop + `eigh` | `analysis/subject_shape_runs.py:1173` |
| ellipse fit | No — per-row loop over `cv2` | `analysis/subject_shape_runs.py:1211` |
| connected-components / fill-holes / morphologyEx / select | No — per-(ROI,component) loop | `refinement/subject_mask_finalization.py:128` |
| QC (solidity/holes/skeleton) | No — per-row loop | `refinement/subject_body_mask_qc.py:155` |

The same per-row-loop pattern still repeats in **three** places for topology/QC/shape fitting
(finalization topology, component QC, subject-shape). The low-risk spatial-metric cleanup for
area, centroid, and bbox is now shared across finalization, review helpers, and subject-shape.

**Why the hard ones are hard, and the ladder of options:**
- **Already vectorized in shared code:** area, mask-present, centroid, and bbox are pure
  reductions (`sum`, weighted mean, min/max over a fixed grid). They are safe numpy work, not
  Dask work.
- **Connected-components** is a graph flood-fill (iterative union-find, data-dependent control
  flow, ragged output), not a reduction — it can't be expressed as one numpy array op. Faster
  via `cc3d` (CPU C++ labeler) or `cucim` (GPU labeler, RAPIDS/`cupy`).
- **Morphology** (close/open/dilate/erode) batches well on GPU via `kornia` (max-pool based) or
  `cucim`.
- **`cv2.fitEllipse`** is the stubborn one: it needs a contour first (`findContours`, irregular)
  and then a variable-size conic least-squares fit (point count differs per mask, so inputs
  can't be stacked). Best case is GPU-per-contour or accepting the loop.
- Caveat for `cc3d`/`cucim`: a stack of independent 2D masks `(N,H,W)` is not a 3D volume —
  use planar/2D connectivity or per-slice calls, else 3D labeling merges unrelated ROIs across
  the N axis.

**Padding stabilizes the output, not the computation.** Ragged results (variable component
count, variable contour length) can be written into fixed-shape arrays by padding to a chosen
`K_max` with a sentinel (NaN/-1) plus a present/overflow flag — and the storage already does this
via the fixed component axis (`mask_present`/`area_px` are `(N, n_components)` with absent
components flagged) and the ellipse `np.full((N,5), nan)` fill. That lets downstream reductions
vectorize (`np.nanmax`, `(areas > t).sum(axis=1)`), but it does **not** speed up the per-mask
labeling/contour/fit — the irregular op still runs per row. Never pad data that feeds a solver
(contour points → ellipse): fake points bias the fit.

**Is the irregularity fundamental? It depends on the op's class:**
- *Fixed-depth local ops* — area, centroid, bbox, and dilate/erode/close (fixed kernel) — are
  not irregular at all; fixed-size stencils/reductions that vectorize and GPU-batch cleanly
  (`kornia`/`cucim`). Morphology is in this class — just unbatched *here*.
- *Global-connectivity ops* — connected-components, `binary_fill_holes`, largest-component
  selection, watershed — are **inherently** not a single SIMD op: deciding "same blob?" needs
  propagation whose *depth = the component's diameter* (data-dependent), and the component count
  is data-dependent (ragged). Intrinsic to the problem. Still *parallelizable* on GPU (iterative
  label propagation / block union-find, as in `cucim`) — just not as numpy broadcasting.
- *Ragged-input geometric fits* — `fitEllipse` — are irregular mainly because of their **input**:
  the conic least-squares is a masked reduction (scatter matrix = Σ of fixed 6-vectors) plus a
  fixed 6×6 eigensolve, which *is* batchable given padded points + a validity mask. The
  genuinely irregular upstream step is contour extraction (boundary tracing, ragged point count).
  So the fit math could batch; the contour can't easily.

Net: connectivity is essentially irregular (data-dependent propagation depth); ellipse is
irregular by composition (its core fit could batch, its contour input can't); morphology isn't
irregular at all. None collapse to a numpy reduction, but all are GPU-parallelizable.

### Orthogonal future lever: canonical body-frame masks

Canonicalizing ROIs into a shared body frame is a separate idea from vectorizing the current
operations. It does **not** make connected-components or `cv2.fitEllipse` naturally vectorized:
a flood-fill is still data-dependent, and ellipse fitting still needs irregular contours. The
potential win is different: for rigid structures such as eyes and swim bladder, a reliable
heading-aligned ROI could make fixed spatial priors, component templates, or per-pixel atlases
meaningful enough to avoid some global search work.

The distinction is the reduction axis:
- Scalar reductions such as `area_px.mean()` discard position; alignment is irrelevant because
  an area is an area.
- Atlas reductions such as `masks[:, swim_bladder].mean(axis=0)` keep `(H,W)` position; without
  registration, pixel `(x,y)` is not the same anatomical location across ROIs, so the result is a
  smear rather than a useful template.

This is most plausible for eyes and swim bladder, less so for body/tail masks because articulation
keeps those components high-variance even after heading alignment. Treat it as a read-only
diagnostic first: warp a sample of existing masks using refined keypoint heading, build component
atlases, and test whether atlas residuals predict current QC failures. Do not change the storage
contract or finalizer until that diagnostic proves the priors are robust.

**Registration needs both location and size.** A crisp atlas requires each ROI normalized for
translation, rotation, and scale. Translation is already free (crops are centered on the
detection); rotation needs heading alignment; scale needs an explicit step (the fixed `roi_size`
window does not normalize apparent size). So "similar locations" needs rotation on top of the
free centering, and "similar sizes" needs scale — full canonicalization is a *similarity
transform* per ROI, not just a rotation. Record any heading/scale you remove rather than discard
it: apparent size and heading can be signal, not nuisance.

**Eyes are only semi-rigid — their orientation is signal.** Eye *socket location* is body-rigid
(a location atlas would be clean), but eye *orientation* is a free degree of freedom: the eyes
rotate nasally (converged) vs temporally/laterally (diverged) frame to frame. The pipeline already
measures exactly this — `analysis/eye_angle_runs` derives per-eye angle, vergence (eyes opposite),
and version (eyes together) from the eye-mask ellipse major axis (`analysis/eye_angle_analysis.py`,
`analysis/eye_angle_io.py`). So an atlas of eye *shape* would smear by the vergence angle, and that
angle is **signal, not nuisance** — it is measured, never normalized away. A useful eye atlas would
register *location* only, leaving orientation as the quantity of interest.

**Keep stored ROI geometry axis-aligned.** Crops are centered, axis-aligned windows
(`tracking/crop.py`), so a mask in ROI space projects back to full-video coordinates with only
the crop offset — no rotation. Canonicalizing the *stored* masks into a heading-aligned frame
would force every downstream consumer that maps masks onto raw video to carry and invert a
per-ROI rotation. So any body-frame work must stay an analysis-time overlay, never a change to
the stored crop/mask geometry.

## 4. Non-canonical implementation checklist

This checklist keeps the current ROI-local subject-mask contract intact. Canonical body-frame
analysis is a later diagnostic path, not a prerequisite for the following work.

1. **Make crop-row lineage mandatory for modern subject/refined masks.**
   - Require `source_crop_row_ids` on `subject_mask_runs` and `refined_subject_masks_runs`.
   - Validate `crop_runs/<source_crop_run>/frame_indices[source_crop_row_ids] == frame_indices`.
   - Allow direct-row fallback only while upgrading legacy runs that have exact row/frame parity.
   - Status: implemented in the current worktree; tests cover finalization, assembly, review, and
     contract validation.
2. **Share vectorized spatial metrics across mask consumers.**
   - Keep one policy-free helper for area, mask-present, centroid, and bbox.
   - Use it from finalization, review helpers, and subject-shape runs.
   - Status: implemented in the current worktree through `shared.mask_geometry.batch_mask_spatial_metrics`.
3. **Reduce duplicate topology passes without changing semantics.**
   - Reuse connected-component stats across thresholding, small-component removal, selection,
     metrics, and reason labels.
   - Preserve row-wise cv2/scipy behavior for connected-components, hole stats, and ellipse/QC.
   - Status: partially implemented; hysteresis thresholding now returns reusable component stats,
     and existing min-area/selection paths already pass stats forward.
4. **Improve operational telemetry and defaults.**
   - Keep per-chunk phase timing for finalization, topology/hole metrics, eye assignment,
     dense/RLE writes, contours, and component metric writes.
   - Keep workflow-level JSONL timing for staging, inference, finalization, validation, publish,
     and cleanup.
   - Record which run groups and handoff packages were published so PRFS/NRS publish cost can be
     interpreted after cluster runs.
   - Status: implemented in the current worktree.
5. **Benchmark optional faster morphology/labeling libraries before adopting them.**
   - Test `cc3d` for CPU connected components and `cucim`/`kornia` for GPU morphology in a
     diagnostic harness.
   - Require parity/equivalence evidence before production use.
   - Status: diagnostic harness added at
     `fisheye.diagnostics.benchmark_subject_mask_primitives`; no production dependency or
     finalizer behavior change yet. Real GoodCopBadCop slices show `cc3d` is
     parity-correct and sometimes faster than OpenCV for connected components, but
     not consistently faster across components/ranges. Treat it as a guarded
     candidate backend, not a default replacement.
6. **Continue storage/publish optimization separately.**
   - Keep dense masks for current consumers; use RLE/chunk-size experiments to reduce publication
     cost without changing biological semantics.
   - Treat sharded storage as a later stable-surface option after edit/write patterns settle.
   - Status: future production-hardening track.

## Critical files

- `src/fisheye/refinement/finalize_subject_masks.py` — backends, task build, worker, merge, provenance, argparse.
- `src/fisheye/refinement/subject_mask_finalization.py` — `finalize_component_mask` per-component compute (`:128`).
- `src/fisheye/shared/mask_geometry.py` — shared spatial-mask reductions and binary-mask primitives.
- `src/fisheye/diagnostics/benchmark_subject_mask_primitives.py` — optional primitive backend
  benchmark for `cc3d`, `cucim`, and `kornia`.
- `src/fisheye/shared/subject_mask_chunks.py` — chunk sizes + the 256-row worker clamp.
- `src/fisheye/utils/run_subject_mask_batch_pipeline.py` — batch defaults (parallel-by-default).
- `scripts/submit_subject_mask_batches_bsub.sh` — cross-run LSF array-job fan-out.
- `tests/unit/fisheye/test_finalize_subject_masks.py` — backend parity tests.

## Primitive Backend Benchmark Commands

Synthetic CPU smoke:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_primitives \
  --row-count 256 \
  --height 512 \
  --width 512 \
  --repeat 3
```

CUDA/GPU candidate smoke, when a GPU is visible:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_primitives \
  --row-count 256 \
  --height 512 \
  --width 512 \
  --repeat 3 \
  --include-gpu
```

Real subject-mask slice:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_primitives \
  --zarr /path/to/analysis.zarr \
  --subject-run subject_masks_<run> \
  --component subject_body \
  --start-row 0 \
  --row-count 1024 \
  --repeat 3
```

Interpretation rule: parity comes first. A backend that is faster but returns
`parity="fail"` is not a candidate for production until the semantic mismatch is
understood and documented.

Implementation note: cuCIM GPU backends run in a subprocess worker. On the 2026-06-20
workstation check, cuCIM completed successfully but could segfault during Python
interpreter teardown if used in-process; subprocess isolation keeps the diagnostic
command itself reliable.

Real-slice check on
`/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop/...`
using `subject_masks_unet_registry_subject_mask_dense_and_rle_smoke_20260620_apply_01`
showed clean parity for `cc3d`, `skimage`, and `cucim` against the OpenCV reference.
The speed result was component/range dependent:

- rows `0:1024`, `subject_body`: OpenCV 2344 masks/s, `cc3d` 2302 masks/s.
- rows `0:1024`, `swim_bladder`: OpenCV 2609 masks/s, `cc3d` 2324 masks/s.
- rows `0:1024`, `eyes_union`: OpenCV 3427 masks/s, `cc3d` 2270 masks/s.
- rows `60000:61024`, `subject_body`: OpenCV 1473 masks/s, `cc3d` 2292 masks/s.
- rows `60000:61024`, `swim_bladder`: OpenCV 3084 masks/s, `cc3d` 2222 masks/s.
- rows `60000:61024`, `eyes_union`: OpenCV 2193 masks/s, `cc3d` 2310 masks/s.

Interpretation: `cc3d` is worth keeping in the benchmark harness and may be worth
an opt-in connected-components backend after stage-level profiling, but the current
data does not justify replacing OpenCV globally. cuCIM was parity-correct but slower
than both OpenCV and `cc3d` on these CPU-sized real slices.
