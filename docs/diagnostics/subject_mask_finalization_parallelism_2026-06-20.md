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

## 5. Eye-assignment optimization checklist

The current subject-mask eye assignment is intentionally more expensive than the refined-keypoint
left/right check. Keypoint refinement rotates three points into a body frame and swaps labels if
needed (`refinement/refine_keypoints.py:_detect_eye_flip`). Subject-mask assignment must split a
full `eyes_union` raster into `eye_left`/`eye_right`, select connected components, and fit ellipses
so review routing remains comparable to existing refined eye-mask behavior
(`refinement/subject_eye_assignment.py:assign_eyes_union_to_lr`).

Current cluster attribution from the bitpacked-only GoodCopBadCop canary:

| Phase | Attributed seconds | Notes |
|---|---:|---|
| `eye_assignment_split_by_keypoint` | ~206s | currently broadcasts two dense 512x512 distance fields per row subbatch |
| `eye_assignment_select_components` | ~105s | connected-components on split left/right masks |
| `eye_assignment_measure_ellipse` | ~160s | contour extraction + `cv2.fitEllipse` for both eyes |

Treat these phase values as summed worker attribution, not direct wall-time percentages. They are
still useful for ranking which repeated work to attack first.

### Phase A: exact split rewrite

- Replace the current two-distance split with the algebraically equivalent keypoint bisector
  half-plane:
  `dist_left <= dist_right` can be rewritten as one signed linear expression in `(x, y)`.
- Preserve the tie rule exactly: ties go left, matching `dist_left <= dist_right`.
- Keep output masks byte-identical to the current implementation for all candidate rows.
- Add direct unit tests for:
  - random masks/keypoints with finite left/right points;
  - coincident and nearly coincident keypoints;
  - empty masks;
  - points outside the ROI;
  - tie-plane pixels.
- Add a diagnostic benchmark that runs current split vs half-plane split on real
  `eyes_union` chunks and reports parity plus time.
- Acceptance gate: `np.array_equal(left_old, left_new)` and
  `np.array_equal(right_old, right_new)` on synthetic and real slices.
- Current status: investigated, not promoted. A half-plane implementation and benchmark diagnostic
  now exist, but real GoodCopBadCop validation exposed float32 boundary pixels where the algebraic
  expression can disagree with the current squared-distance implementation. A narrow exact-distance
  boundary correction restores parity, but it makes the dense half-plane path slower than the
  current distance path on the tested 1024-row 512x512 slice:
  - current distance batch: best 0.494s, ~2072 candidate rows/s;
  - exact-corrected half-plane batch: best 0.815s, ~1255 candidate rows/s.
  Keep production on the current distance batch unless a future sparse/boundary strategy beats it
  with byte-identical outputs.

### Phase B: foreground-sparse split benchmark

- Benchmark a sparse splitter that only evaluates foreground pixels from `np.nonzero(eyes_union)`.
- Compare against the dense half-plane splitter on real GoodCopBadCop slices, because the winner
  depends on mask density and NumPy allocation overhead.
- Preserve the exact tie rule and output parity.
- Acceptance gate: sparse split is adopted only if it is faster on real slices and exact-parity
  against the half-plane split.
- Current status: investigated, not promoted. A sparse explicit-distance implementation now exists
  as a benchmark candidate and is byte-identical to the current distance batch on synthetic and
  real slices. It was not faster on the tested GoodCopBadCop 1024-row 512x512 slice despite low
  foreground density (~0.15%):
  - current distance batch: best 0.494s, ~2071 candidate rows/s;
  - sparse batch: best 0.509s, ~2010 candidate rows/s;
  - row-wise sparse probe: best 0.564s.
  The current dense NumPy distance-field implementation remains hard to beat because it is simple,
  contiguous, and vectorized even though it touches many background pixels.

### Phase C: exact component fast path

- Revisit the rejected whole-component fast path, but make it fail-closed:
  - label `eyes_union` components;
  - compute each component's side(s) relative to the keypoint split plane;
  - assign whole components only when every foreground pixel in that component lies on one side;
  - fall back to pixel split when any component crosses the plane or when component count/geometry
    is ambiguous.
- Preserve existing `assignment_status` and `reason_labels` for every tested row.
- Acceptance gate: status counts, reason counts, left/right masks, and downstream ellipse failure
  labels are unchanged on real benchmark slices. If the fast path changes review labels, reject it
  unless the review contract is intentionally revised.
- Current status: investigated, not promoted. The fail-closed component path handled 768/1023
  candidate rows on the tested GoodCopBadCop slice and preserved left/right masks,
  `assignment_status`, `reason_labels`, and ellipse success flags. However, a few tiny components
  produced different `cv2.fitEllipse` parameter values despite byte-identical masks, and the real
  slice speedup was only modest:
  - standard assignment: best 2.149s for 1024 rows, ~477 rows/s;
  - component fast path: best 2.063s for 1024 rows, ~496 rows/s.
  Keep this path disabled by default. The risk/reward is weak unless a later implementation can
  make ellipse geometry bit-stable or explicitly narrow the contract to mask/status/reason parity.

### Phase D: reduce duplicate shape work

- Audit whether assignment-time `_measure_mask()` results can be carried forward into
  postcompute eye geometry instead of fitting ellipses twice.
- If reuse is possible, store the assignment measurements in worker-local outputs and merge them
  deterministically with the same row-sharded process plan.
- Keep postcompute as the authoritative writer for persisted geometry until parity is proven.
- Acceptance gate: persisted eye geometry, contours, review statuses, and component-quality rows
  match the current implementation on real slices.
- Current status: implemented for assignment-derived eyes. `assign_eyes_union_to_lr` now returns the
  ellipse parameters, success flags, centroids, and contours it already computes for
  `eye_left`/`eye_right`. The smart finalizer carries those payloads through serial, Dask, and
  process-shard chunk results and writes the normal refined subject eye-geometry surface through
  the existing sharded geometry writer with `eye_geometry_postcompute_backend="assignment_reuse"`.
  If assignment geometry shards are incomplete, the code falls back to the previous postcompute
  path. Component contours for `subject_body` and `swim_bladder` still use the normal postcompute
  path.
- Local validation on a 512-row GoodCopBadCop slice with `process_shards`, `num_workers=4`,
  `write_eye_geometry=true`, and `write_component_contours=true` completed successfully:
  - finalizer wall time: 9.75s; reported finalizer duration: 9.66s; ~53 rows/s;
  - `write_eye_geometry_from_assignment`: 0.151s;
  - `eye_assignment`: 0.885s, with split 0.303s, select-components 0.230s, ellipse-measure 0.237s;
  - `write_component_contours`: 2.675s.
  This confirms assignment reuse removed the separate eye-geometry recomputation from the
  production path; component contours are now the larger remaining postcompute cost.

### Phase F: chunked component-contour mask reads

- The direct/serial contour writer previously used `MaskStore.read_dense(rows=row_idx, ...)` once
  per row. This was safe, but expensive for dense Zarr masks because each component contour cache
  made hundreds or thousands of tiny reads before doing the same OpenCV contour extraction.
- `build_component_contours_from_masks` now reads contiguous row chunks from the selected
  `MaskStore` surface, then runs the unchanged per-row `extract_largest_external_contour` loop over
  the in-memory chunk. This preserves the output contract and works for dense, bitpacked, and RLE
  mask stores through the same `MaskStore.read_dense` abstraction.
- Unit coverage pins that a 5-row component with `read_chunk_size=2` performs three mask-store
  reads (`0:2`, `2:4`, `4:5`) rather than five per-row reads.
- Local validation on the same 512-row GoodCopBadCop benchmark showed:
  - finalizer wall time: 7.70s; reported finalizer duration: 7.58s; ~67.5 rows/s;
  - `write_component_contours`: 0.447s, down from 2.675s in the previous local run;
  - `write_eye_geometry_from_assignment`: 0.162s.
  This makes direct component-contour writing no longer the dominant postcompute cost on the tested
  local slice; the remaining floor is the row-sharded finalization compute itself.

### Phase G: component centroid selector attempt

- Investigated replacing `select_component_near_point`'s `connectedComponents` plus NumPy
  `bincount` centroid calculation with `cv2.connectedComponentsWithStats`.
- Result: rejected. The helper produced compatible selections in focused tests, but the real
  1024-row GoodCopBadCop assignment benchmark slowed the standard assignment path from the prior
  ~2.15s local slice to ~2.99s. The production helper remains on the existing
  `connectedComponents` + cached-grid/`bincount` centroid path.
- Revert confirmation: after restoring the original helper, the same 1024-row benchmark returned to
  ~1.94s for standard assignment in a shorter two-repeat check.
- Lesson: OpenCV's stats/centroid convenience path is not automatically cheaper for these small,
  sparse split-eye masks. Keep the explicit measured implementation unless a future candidate
  beats it on the real assignment benchmark.

### Full-recording cluster canary after Phases D/F/G

- Cluster canary:
  `subject_mask_finalizer_opt_canary_20260621_140111`,
  variant `v01_process_shards_processes_w8_c256`, run on one `short` queue job with 8 worker
  processes against the GoodCopBadCop bitpacked-only subject-mask canary.
- Result: 120,221 rows finalized in 739.24s, or 162.63 rows/s end-to-end.
- Wall-time phases:
  - `process_shard_compute`: 617.83s;
  - `write_component_contours`: 101.73s;
  - total `duration_seconds`: 739.24s.
- Largest summed worker/chunk attributions remain:
  - `finalize_subject_body`: 1081.14s;
  - `finalize_eyes_union`: 968.43s;
  - `finalize_swim_bladder`: 962.34s;
  - `eye_assignment`: 535.12s, including split 198.21s, ellipse measurement 162.91s, and
    component selection 103.18s;
  - dense mask writes are still visible at ~142-144s per component in summed attribution.
- Interpretation: chunked contour reads fixed the local postcompute read bottleneck, but the
  full-recording cluster floor is now the row-sharded finalization compute plus dense mask writes.
  Further wins should target component finalization/eye assignment semantics or dense-write volume,
  not another serial contour-read cleanup.
- Operational note: `process_shards` now emits one `process_shard_submitted` progress JSONL record
  per planned shard, including `shard_index`, `start_row`, `stop_row`, `chunk_count`,
  `total_shards`, and `worker_count`. This does not change scheduling, but it makes future long
  cluster jobs interpretable before the first large shard completes.

### Phase E: contract decision, not optimization

- Decide whether ellipse failure should remain part of assignment status or become a later QC
  status. Moving it out of assignment could make assignment much cheaper, but it changes the
  meaning of `assigned` vs `assigned_needs_review`.
- Do not change this in the optimization slice. Write a separate contract proposal if we want
  assignment to mean only "left/right anatomical ownership resolved" while QC means "shape passed".

### Benchmark ladder

Run each candidate in this order:

1. Synthetic parity and microbenchmarks for the split helper.
2. Real 4096-row diagnostic against the same GoodCopBadCop run used in
   `subject_mask_finalizer_performance_2026-06-18.md`.
3. Full-recording finalizer canary with `--mask-storage bitpacked_v1`,
   `--write-eye-geometry`, and `--write-component-contours`.
4. Registry refresh and compare:
   - `subject_mask_performance.rois_per_second`;
   - `smart_finalizer_timing_summary.phase_seconds.eye_assignment_*`;
   - `eyes_union_assignment_summary.status_counts`;
   - component-quality rows for `eye_left` and `eye_right`.

## Critical files

- `src/fisheye/refinement/finalize_subject_masks.py` — backends, task build, worker, merge, provenance, argparse.
- `src/fisheye/refinement/subject_eye_assignment.py` — current `eyes_union` to
  `eye_left`/`eye_right` split, component selection, ellipse check, reason labels.
- `src/fisheye/refinement/subject_mask_finalization.py` — `finalize_component_mask` per-component compute (`:128`).
- `src/fisheye/refinement/refine_keypoints.py` — scalar left/right keypoint
  flip check; useful contrast but not the same workload as raster eye assignment.
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
