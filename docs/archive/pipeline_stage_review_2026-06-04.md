<!-- ARCHIVED 2026-07-04: dated point-in-time snapshot / spent work ticket, retained for history only. -->

# Pipeline Stage-by-Stage Review — 2026-06-04

A walk through the pipeline stages, reviewing design and implementation at the
same depth as `import_step_design_review_2026-06-04.md`. Point-in-time; line
numbers reference the live repo (`gitrepos/palette`, branch `sun`) on this date.

Stage order (from `core/pipeline.py`): import → background → detect →
detect_quality → crop → keypoints / eye_masks / refined masks → track → refine →
assign_ids.

- Import is covered separately in `import_step_design_review_2026-06-04.md`.

---

## Orchestration: two competing paths (read this first)

There is no single canonical orchestrator. Two paradigms coexist, and they
differ not just in plumbing but in **which algorithms run by default** — so the
distinction is semantic, not cosmetic.

**Path A — the in-process monolith (`core/pipeline.py`, `Pipeline` class).**
- This is the `python -m fisheye` entry point: `__main__.py` calls
  `core.pipeline.main()`, which builds a large argparse and runs
  `Pipeline(config).run()`.
- It defines the full stage set as in-process methods (`_run_import`,
  `_run_background`, `_run_detect`, `_run_crop`, `_run_keypoints`,
  `_run_eye_masks`, `_run_refined_subject_masks`, `_run_refine`,
  `_run_detect_quality`, `_run_assign_ids`, …).
- **It wires the *traditional* (classical-CV) detector** — `_run_detect` →
  `detect_fish` (`detection/detect_traditional.py`), not YOLO. Keypoints default
  to `method='traditional'` too (YOLO is opt-in via config).
- Still live and recently touched (the 05-20 run-completion sweep), so not dead.

**Path B — the registry/cluster constellation (the production path).**
- ~17 per-stage `run_*_batch.py` / `run_*_pipeline.py` scripts
  (`run_detections_batch`, `run_keypoints_batch`, `run_eye_masks_batch`,
  `run_subject_mask_batch_pipeline`, `run_recording_analysis_pipeline`, …) that
  **shell out via `subprocess` to per-module CLIs** (`python -m
  fisheye.detection.detect_yolo`, etc.), coordinated by the registry
  (`recording_step_status`).
- **Uses the YOLO detector** (`detect_yolo`). The interactive TUI
  (`cli/interactive_launcher.py`) also belongs here — it builds commands and
  `subprocess.Popen`s them rather than instantiating `Pipeline`.
- There is **no single modern orchestrator**: `run_recording_analysis_pipeline`
  covers only detect → detect_quality → refine_detect → keypoints →
  refine_keypoints; every other stage (crop, masks, tracking, bouts, training)
  has its own batch runner. `docs/inference_pipeline_divergence_analysis.md`
  inventories the modern engines and **does not even list the monolith's detect
  path**; `docs/cluster_pipeline_migration_checklist.md` is an active
  `working_checklist` with many unchecked items.

### Verdict

`core/pipeline.py` is a **superseded-but-still-wired legacy monolith**, not the
canonical production path — yet also not safe to delete, because it is the only
single-command full-stage runner and the `python -m fisheye` entry point. The
canonical *production* path is the registry-coordinated, YOLO-based batch-runner
constellation, and that migration is incomplete.

The sharp risk: **`python -m fisheye … detect` silently runs classical-CV
detection, a different algorithm from the YOLO path the rest of the system
standardized on.** The two orchestrators can produce divergent scientific
outputs from the same inputs. This is the strongest confirmation of the owner's
"duplicated / unenforced functionality" concern.

**Consequence for this review:** the monolith's stage *methods* are the legacy
in-process versions; production behavior for detect/keypoints/masks lives in the
per-module CLIs and their batch runners. Where they differ, the per-module CLI is
the one that matters. The detection review below covers `detect_yolo` (Path B);
the monolith's `_run_detect`/`detect_fish` is the legacy Path-A variant.

---

## Background (`preprocessing/background.py`, 291 LOC)

Computes a static background model (mode or median over sampled frames) into a
timestamped `background_runs/<run>` group with `background_full` / `background_ds`
arrays.

**Who consumes it:** the *traditional* (classical-CV) detection path
(`detection/detect_traditional.py`, `detect_keypoints_traditional.py`), subject/
eye segmentation, and segmentation training (background negatives, e.g.
`training/zarr_eye_mask_dataset.py`). The **YOLO detection path
(`detect_yolo.py`) does not use it** — zero references. So background is a real,
consumed artifact for the traditional + segmentation paths, not vestigial, but
it is not on the modern YOLO detection path. The eye-segmentation config sets
`background_from_downsampled: false`, so **full-resolution background is used in
practice.**

**Positive:** unlike import, this stage correctly routes through the
run-completion contract — `mark_run_started` / `mark_run_complete` /
`note_pending_latest` (`shared/zarr_run_completion.py`). It also warns when the
background runs on a different git commit than the import (lines 99-107).

### Findings

1. **`fast_mode_bincount` is a pure-Python per-pixel double loop (lines 32-36),
   and `mode` is the default method on full-resolution data.**
   ```python
   for i in range(height):
       for j in range(width):
           counts = np.bincount(data_chunk[:, i, j], minlength=256)
           result[i, j] = np.argmax(counts)
   ```
   On a full-res 4512×4512 frame that is ~20.4M Python-level iterations, each a
   `bincount`+`argmax` — minutes to hours where a vectorized mode (histogram over
   the stack, sort-based, or `scipy.stats.mode`) is seconds. The name actively
   misleads. Single biggest issue in the stage; exercised because full-res
   background is requested by config.

2. **Whole sample loaded into RAM at once (lines 188-190, 206-208).**
   `get_orthogonal_selection((frame_indices, :, :))` materializes
   `(sample_size, H, W)`. Default `sample_size=100` ≈ 2 GB full-res (fine), but
   the documented `sample_size=-1` ("all frames") on full-res is
   `n_frames × 4512²` → hundreds of GB → hard OOM. Mode and median can both be
   computed incrementally/streamed; they are not. The `-1` option is a
   documented foot-gun.

3. **Yet another git-provenance schema (lines 236-250).** Stamps a `code_version`
   dict with `git_commit` / `git_short` / `git_branch` / `git_dirty` — a third
   spelling of fields import wrote as `git_commit_hash`/… and `stage_provenance`
   writes differently again. Concrete instance of the metadata-fragmentation
   finding in `repo_enforcement_review_2026-06-04.md`.

**Minor:** redundant double `zarr.open_group(mode='r+')` (lines 97, 119); uses
Python's global `random.seed`/`random.sample` (perturbs global RNG state) rather
than a local `Generator`; `source_frame_indices` is stored as the string
`"sampled_N_frames"` once ≥100 frames (line 235), discarding exact indices so
reproducibility leans entirely on seed+count.

**Net:** architecturally sound and correctly wired into run-completion (a
positive contrast to import), but the default mode computation is a severe
performance trap on full-resolution data and the all-frames option is an OOM
trap — the kind of thing that passes on 640×640 downsampled in testing and melts
on full-res in production.

---

## Detection (`detection/detect_yolo.py`, 1858 LOC)

The modern YOLO inference path: decode video → YOLO predict → write a sparse
detection layout to `detect_runs/<run>` (`frame_indices`, `bbox_norm_coords`,
`scores`, `class_ids`, plus counts). Writes to a standalone
`<stem>_detections.zarr` (`has_raw_video: False`, `zarr_purpose: production`),
distinct from the analysis zarr.

### Positives

- **Resolution-independent bbox output.** `accumulate_results` normalizes boxes
  by `inference_width/height` into `[0,1]` cx/cy/w/h (lines 1162-1167), so the
  stored geometry is decode/resize-independent and downstream un-normalizes to
  any target resolution. This sidesteps the coordinate-space class of bugs.
- **Good provenance citizen — better than import or background.** It routes
  through both `build_stage_provenance`/`write_stage_provenance` (lines
  1649-1694) and the run-completion contract (`mark_run_started`/
  `note_pending_latest`/`mark_run_complete`, lines 1137-1696).
- Sensible chunk sizing (lines 1525-1528), detailed per-stage timings, and
  defensive config-conflict detection (resize_dims vs imgsz vs legacy
  video.resize, lines 603-657).

### Findings

1. **`detect_yolo` is a single ~1,200-line function (lines 531-1744) with two to
   three near-duplicate decode/inference loops** — pynvvc (1215-1278), decord
   (1279-~1485), and opencv — each re-implementing batching, timing, and the
   `accumulate_results` call. This is the central maintainability problem of the
   stage: a change to the inference/accumulate path must be made in each loop.
   The function badly wants decomposition into a reader abstraction + one loop.

2. **Two detection implementations and two orchestrators.** `detect_yolo` (this
   file, modern) is invoked by the registry/cluster runners
   (`utils/run_recording_analysis_pipeline.py`, `run_detections_batch.py`,
   `run_detect_with_registry_model.py`, `inference/predict_detections.py`).
   Meanwhile `detect_fish` (`detection/detect_traditional.py`, classical CV) is
   what the monolithic `core/pipeline.py` `Pipeline._run_detect` actually calls.
   The `Pipeline` class is instantiated only by `__main__.py`,
   `cli/interactive_launcher.py`, and a diagnostic — i.e. the in-process
   monolith wires the *traditional* detector, while production YOLO detection
   runs through a *separate* orchestration path. This is an
   in-progress-migration fork (see `inference_pipeline_divergence_analysis.md`,
   `cluster_pipeline_migration_checklist.md`) and a concrete instance of the
   owner's "duplicated/unenforced functionality" worry: two orchestrators, two
   detectors, unclear which is canonical.

3. **Decoder is config-dependent, so the pixels fed to the model vary by flags.**
   `auto` uses pynvvc only when GPU is on *and* `resize_dims` is set (lines
   780-784); otherwise decord; otherwise opencv. The same video can therefore be
   detected through pynvvc, decord, or opencv depending on configuration.
   Detection geometry is safe (normalized), but the model *input pixels* differ
   by backend — the same decord-vs-pynvvc-vs-cv2 parity surface flagged in
   `import_step_design_review_2026-06-04.md`, now also inside detection.

4. **Provenance is written twice, in two systems.** Despite correctly using
   `build_stage_provenance`, the function also dumps ~70 inline root attrs
   (git/system/LSF/SLURM/disk/memory, lines 901-1009) — and writes the core
   video-metadata block twice (always at 901-915, then again at 917-970 for new
   roots, with overlapping keys). Cleaner than import's inline-only approach, but
   still redundant: the same provenance lives in `detect_group` stage-provenance
   and in root attrs.

5. **Duplicate schema arrays.** `n_detections` and `frame_counts` are created
   from the identical `frame_counts` array (lines 1536-1537) — two arrays, same
   bytes, one a legacy alias. Minor, but it is exactly the kind of
   never-retired alias that downstream code then disagrees about.

**Net:** the detection *outputs* are well-designed (normalized sparse layout,
real stage-provenance + run-completion). The problems are structural: a
1,200-line multi-loop function, a config-dependent decoder that reopens the pixel
-parity question, redundant double-written provenance, and — most importantly —
a genuine two-orchestrator / two-detector split where the legacy monolith and the
modern registry runner disagree on which detector is canonical.

---

## Traditional vs YOLO detection: divergence & danger assessment

Goal: the traditional/blob detector (`detect_fish`, `detection/detect_traditional.py`)
should remain supported, or at least not be dangerous. Verdict: it is **structurally
safe but has one genuine silent-semantic hazard** (constant scores).

### Compatible (safe by construction)

- **Identical array schema.** Both write `detect_runs/<run>/` with
  `frame_indices`, `bbox_norm_coords` (`[N,4]` f8), `scores` (f32), `class_ids`
  (i32), `frame_counts`, `n_detections` — same names, dtypes, sparse layout
  (`detect_traditional.py:364-404` vs `detect_yolo.py:1532-1537`).
- **Compatible coordinate space.** Both store `[0,1]` frame-fraction normalized
  `cx/cy/w/h` — traditional by `ds_img_shape` (`detect_traditional.py:248-249`),
  YOLO by inference dims (`detect_yolo.py:1162-1165`). Downstream
  un-normalization (× full-res W/H) is identical, so no coordinate corruption.
- **Both stamp `detection_method`** (`'blob'` at `detect_traditional.py:457-458`
  vs `'yolo'`) and both route through run-completion + `build_stage_provenance`.
  A discriminator exists for any downstream that wants to branch.

So nothing crashes and no geometry is mis-transformed across the two. This is the
good news, and it means traditional is legitimately supportable.

### Divergent (the actual risks)

1. **Constant `scores = 1.0` (blob) vs real confidences (YOLO)** —
   `detect_traditional.py:379` writes `np.ones(...)`. This is the one real
   hazard. `refine_detect`'s top-k-by-score selection is a core feature
   (`refine_detect.py:478,540-566,970-990`), and on constant scores it silently
   degrades to "keep the first k rows" — which, because `detect_fish` sorts blobs
   area-descending (`:240`), means **top-k-by-confidence becomes top-k-by-blob-area**.
   Deterministic, but a different selection criterion than the operator expects,
   with **no warning**: `refine_detect` reads `detection_method` only for a
   diagnostic print in an edge branch (`:1143-1144`) and substitutes placeholder
   `1.0` scores without adjusting ranking (`:1154-1160`). The code *knows* blob
   has no scores and does not compensate.
2. **Constant `class_ids = 0` (blob)** — fine for single-class fish, diverges for
   any multi-class use.
3. **Different input domain.** Blob detects on *downsampled* frames after import
   and **requires** `raw_video/images_ds` + a background run; YOLO reads the video
   directly at inference resolution. Different effective resolution and a
   background dependency the YOLO path does not have.
4. **Different default output archive.** Blob writes `detect_runs` in-place into
   the analysis zarr; YOLO defaults to a standalone `<stem>_detections.zarr`
   (configurable).

### Danger verdict

Not structurally dangerous — compatible schema and coordinates, method recorded.
The single genuine hazard is the constant `scores=1.0`: any
confidence-threshold or top-k-by-confidence workflow silently does something
different on blob output (degrades to area ranking) with no operator-visible
signal. So traditional is *supportable and mostly safe*, but **not safe by
construction** for score-dependent downstream.

### To keep traditional supported and make it non-dangerous

1. **Give blobs a meaningful pseudo-score** (e.g. normalized area, or the
   threshold margin at which the blob survived) instead of constant `1.0`, so
   top-k ranking is sensible and monotonic. Lowest-effort, highest-payoff fix.
2. **Make score-dependent code method-aware.** Have `refine_detect` top-k check
   `detection_method == 'blob'` and either warn loudly or switch to an explicit
   `area` ranking field rather than silently tie-breaking on constant scores.
3. **Surface the detector at the entrypoint.** `python -m fisheye … detect`
   should announce `detection_method=blob (traditional)` so operators are not
   surprised that the monolith ran classical CV rather than YOLO.
4. **Carry `detection_method` into label-origin provenance** so approved training
   labels record whether they originated from blob or YOLO detection (ties into
   the label-origin work already in flight).

Traditional blob detection is a legitimate, model-free, dependency-light detector
— valuable for bootstrapping labels before a model exists. The fixes above keep
it as a first-class supported path while removing the one silent-score trap.

---

## Detect quality (`refinement/detect_quality.py`, 1054 LOC)

Reads a `detect_runs/<run>`, analyzes coverage/gaps, temporal artifacts
(jumps/blips), and bbox validity, then writes an advisory quality report +
A–F grade into `detect_runs/<run>/quality_reports/<quality_run>`. It is the
modern path's required step between detect and refine_detect, and — per the
enforcement review — the *only* stage with hard array validation in the registry.

### Positives (this is one of the better-built stages)

- **Clean read/compute/write split**: `analyze_detect_quality` opens the archive
  read-only and returns a report dict; `save_quality_report` writes. No mutation
  during analysis.
- **Coordinate-aware jump thresholds.** Because bboxes are normalized,
  `_effective_jump_threshold_pixels` resolves a `pixels` / `normalized` / `scaled`
  threshold against actual width/height (lines 210-232, 771-777) — exactly the
  coordinate-space care missing elsewhere.
- **Sampled-import aware.** When the import was frame-sampled, temporal artifact
  detection is skipped (consecutive rows aren't temporally adjacent) and a
  dedicated `calculate_sampled_quality_score` is used (lines 800-841). This
  avoids fabricated "jumps" on sampled data.
- **Required before refine.** `refine_detect` refuses to run without a usable
  detect_quality context (`refine_detect.py:368-369,417,1080-1092`), waiving it
  only for sampled imports.
- Defensive `frame_counts` padding (lines 781-783); routes through
  run-completion (`mark_run_started`/`note_pending_latest`/`mark_run_complete`).

### Findings

1. **The grade is advisory only — no automated threshold gates the pipeline.**
   `require_detect_quality` enforces *presence* (a quality run exists), not a
   minimum score. An `F`-grade detection refines and can flow to training with
   **no automated block**; quality control depends entirely on a human reading
   the grade. Reasonable for a human-in-the-loop design, but combined with the
   no-CI posture it means detection QC is fully manual — the rich scoring
   machinery is, like other mechanisms in this repo, built but not wired as a
   gate. Worth making explicit (and offering an optional `--min-grade`/`--min-score`
   gate for batch/cluster runs).
2. **Hardcoded heuristic weights and grade cutoffs.** The score is
   `coverage*0.5 + artifact*0.3 + bbox*0.2` (lines 409-413) with A–F at
   90/80/70/60 (lines 416-425) — none configurable. Fine as a heuristic, but it
   is presented as a single authoritative "quality_score" that downstream/humans
   may over-trust.
3. **Note on the registry array-validation special-case.** detect_quality being
   the lone hard-validated stage validates *its own* output arrays exist — it is
   not a detection-quality gate. The naming invites confusion between "the stage
   that is enforced" and "the stage that enforces quality"; it does the former,
   not the latter.

**Minor:** the report is stored as attrs on a run-within-a-run group
(`detect_runs/<run>/quality_reports/<run>`), adding path depth; grades/scores
live in attrs rather than arrays.

**Net:** genuinely well-engineered — coordinate-aware, sampled-aware, cleanly
separated, correctly required-before-refine. Its one structural gap is the same
repo-wide theme: it *measures* quality thoroughly but *enforces* nothing about
it, so a poor detection silently proceeds unless a human intervenes.

---

## Crop (`tracking/crop.py`, 3917 LOC)

Extracts fixed-size ROI crops around each detection into `crop_runs/<run>` —
either materialized (`roi_images` stored) or geometry-only (coordinates only).
This is the largest and most complex stage in the pipeline by a wide margin.

### Geometry & correctness (sound)

- **Fixed-size ROI centered on the normalized bbox center.** `_compute_roi_coordinates`
  (lines 292-322) takes `bbox_coords[:, :2]` (the normalized cx/cy from detect),
  scales by `[video_w, video_h]`, rounds, and places a fixed `roi_sz` box —
  intentionally ignoring bbox width/height so all crops are uniform for the
  downstream model. Correctly consumes detect's coordinate convention.
- **Out-of-bounds by design, extraction zero-pads.** Top-left coords are not
  clamped (can be negative near frame edges); the extraction kernels initialize
  `crops` to zeros and copy only the valid overlap into the correctly-offset
  sub-window (lines 950-962), matching the `zero outside source-frame bounds`
  pixel contract.
- **Honors the canonical-pixels intent.** Prefers the zarr `raw_video` source
  over re-decoding (`get_video_source`/`_resolve_video_source`), and enforces the
  training-materialized contract (`_enforce_training_materialized_crop_contract`)
  so training crops can't be geometry-only. Good provenance citizen:
  `build_stage_provenance`, pixel-contract attrs (`_set_crop_pixel_contract_attrs`),
  step-status emission, run-completion.

### Findings

1. **Complexity is concentrated here, and it is extreme.** 3917 LOC (≈2× the next
   largest stage). `crop_detections` takes **21 parameters** and runs ~830 lines.
   There is a combinatorial explosion of crop kernels — `crop_batch_gpu`,
   `crop_batch_cpu`, `crop_batch_cpu_from_top_left`, `_process_chunk_gpu`,
   `_process_chunk_gpu_from_top_left` — across {GPU, CPU} × {from-bbox,
   from-stored-top-left}, plus a parallel external-ROI-cache materialization
   subsystem (`materialize_external_roi_cache*`, three functions). Combined with
   6 detection source types (detect/filtered/interpolated/manual/refined/auto),
   2 storage modes, and 2 video sources, this single stage carries more
   configuration surface than the rest of the pipeline combined. It is the
   prime candidate for decomposition (a reader abstraction + a single kernel +
   a config dataclass would collapse most of it).
2. **The external-video path reintroduces the cv2-seek + cv2-grayscale hazards.**
   `crop_batch_cpu_from_top_left` (lines 914-924) decodes with
   `cv2.VideoCapture` + `cap.set(CAP_PROP_POS_FRAMES, …)` + `BGR2GRAY` — the same
   unreliable long-GOP random seek and the same third grayscale conversion
   flagged in `import_step_design_review_2026-06-04.md` for
   `create_clipped_training_zarr`. When crop materializes from external video
   instead of the zarr, ROI pixels can come from a mis-seeked frame and a
   different gray formula than the canonical import pixels. Mitigated whenever
   `raw_video` is present (the preferred path), but external is a supported and
   sometimes-used route, so the parity exposure is real for crop too.
3. **21-parameter entry point.** `crop_detections`'s signature is a config
   dataclass struggling to be born — every external-write knob (backend,
   storage, sharding, chunk/shard sizes, kvikio requirement) is a separate
   positional/keyword arg. High cognitive load and easy to mis-wire from callers.
4. **`use_consolidated=False` on open (line 2872).** A deliberate workaround for
   stale consolidated metadata (per AGENTS.md) — correct here, but a recurring
   signal that the zarr consolidated-metadata story is fragile across the repo.

**Net:** the crop *outputs* and geometry are correct and well-contracted (uniform
ROIs, zero-pad semantics, canonical-pixel preference, real provenance). The
problem is sheer mass: this is where the pipeline's accreted complexity lives —
a 3917-LOC module, a 21-arg/830-line main function, and a combinatorial set of
crop kernels — plus the external-video path quietly re-opening the
decode/seek/grayscale parity surface. It is the strongest candidate in the
pipeline for refactoring, and the place where the owner's "bloat / unenforced
shared helpers" worry is most concretely justified within a single stage.

---

## Keypoints (`detection/detect_keypoints_yolo.py`, 1146 LOC)

Runs YOLO pose on the materialized crop ROIs and writes
`keypoints_runs/<run>` with keypoints in three coordinate spaces plus heading and
validity flags. **This is one of the cleanest, most correct stages in the
pipeline** — notable because keypoint coordinate-space handling previously caused
an incident (`keypoint_refined_coordinate_space_incident_2026-03-04.md`), and the
current code handles it carefully.

### Correctness (this is the good example)

- **Three coordinate spaces, consistently derived** (lines 739-746):
  - `keypoints_roi` — raw YOLO pose output in ROI pixels, clipped to
    `[0, roi_w-1] × [0, roi_h-1]`.
  - `keypoints_img = keypoints_roi + roi_top_left` — full-resolution image space,
    using the crop's stored `roi_coordinates_full`.
  - `keypoints_norm = keypoints_img / [full_W, full_H]` — `norm_factor` confirmed
    to be full-frame dims (line 619), so normalization is correct.
- **Schema-driven, not hardcoded.** `n_keypoints = pose_schema_obj.num_keypoints`
  (line 507); heading is computed via `compute_heading_from_spec` from the pose
  schema's `heading_computation` metadata (lines 287-292) — declarative, not
  baked in.
- **Honest validity flags.** `detection_success` is set only when a detection
  with valid keypoints is found; `heading_finite` and `heading_usable`
  (= success ∧ `detection_source == 0` ∧ finite, lines 790-795) let downstream
  trust only primary-source, finite headings. NaN fill everywhere else rather
  than silent zeros.
- **Careful input preprocessing.** Tensor input mode is gated by
  `_tensor_input_blocker` (square ROIs, `imgsz == roi`, divisible-by-32) and
  normalizes /255 with dtype-aware scaling (lines 309-355); reads the canonical
  crop ROIs; one fish per crop via `_select_detection` (highest-confidence box).
- Good provenance citizen: `build_stage_provenance` + run-completion.

### Findings (minor — this stage is in good shape)

1. **Keypoint clipping can mask out-of-crop landmarks.** Keypoints are clipped to
   ROI bounds (lines 739-740). For a fish larger than the fixed ROI (or poorly
   centered), a true landmark outside the crop becomes an edge-pinned value
   rather than being flagged invalid. Worth a per-keypoint "on-border" flag so
   downstream can distinguish a real edge landmark from a clipped one.
2. **Placeholder NaN arrays.** `effective_threshold` and `effective_se2_radius`
   are created and written all-NaN here (lines 782-783); they are populated by
   the later keypoint-refinement stage. Deliberate two-stage design, but — like
   import's `timestamps` — they are empty-on-write in this stage and a reader
   that doesn't know that could misinterpret them.
3. **Three stored coordinate spaces** triple the keypoint storage and create a
   consistency-maintenance burden. Understandable as defense-in-depth after the
   2026-03-04 incident, but a single canonical space + derivation helpers would
   be leaner; if all three are kept, an invariant test (`img == roi + offset`,
   `norm == img / dims`) should guard them.

**Net:** a well-engineered stage — correct multi-space coordinates, schema-driven
heading/keypoint counts, honest NaN/validity semantics, canonical-crop input, and
real provenance. It is the counter-example to crop: a model-based stage kept tight
and correct.

---

## Segmentation / masks (`segmentation/`, ~376 KB across 8 modules)

The other branch off crops: per-ROI masks for eye(s) and subject body. This is the
**most fragmented stage family in the repo**, and the strongest segmentation-side
instance of the bloat worry.

### The fragmentation (headline)

- **Eye masks have three independent inference backends**, each a separate large
  module, selected by `eye_masks.method`:
  - `eye_segmentation.py` (traditional CV, 37 KB)
  - `eye_segmentation_yolo.py` (YOLO-seg, 56 KB)
  - `infer_unet_eye_masks.py` (U-Net, 42 KB)
  - **The default method is `traditional`** (`run_eye_masks_batch.py:575`) — so,
    as with detection, the out-of-the-box path is classical CV and U-Net is
    opt-in.
- **Subject masks** run U-Net (`infer_unet_subject_masks.py` 52 KB +
  `train_unet_subject_masks.py` 49 KB) plus a SAM path
  (`run_sam_subject_masks_batch`) and a traditional `subject_segmentation.py`
  (36 KB). `swim_bladder_segmentation.py` (58 KB) is a further part-specific
  segmenter.
- Total ≈ 376 KB of segmentation code. The U-Net *model* is correctly shared
  (`segmentation/unet.py`, `UNetSmall`, imported by all four train/infer U-Net
  modules) — so the duplication is in the per-backend **inference/training
  harnesses**, not the model. Three eye-mask harnesses and three subject-mask
  approaches are a large maintained surface for one conceptual stage.

### Schema compatibility & danger

All three eye backends write the same `eye_masks_runs/<run>` family with
`masks_roi` / `mask_probs` in ROI space (`eye_segmentation.py:707`,
`eye_segmentation_yolo.py:881`, `infer_unet_eye_masks.py`), so downstream
consumes them uniformly and nothing crashes — the same parallel-schema property
that makes detection's traditional/YOLO split non-fatal. The soft risk is
**probability semantics**: U-Net writes sigmoid probabilities, YOLO-seg writes
mask confidences, and the traditional path writes thresholded values — a
downstream consumer thresholding `mask_probs` at a fixed cutoff is implicitly
assuming one backend's semantics. Less acute than detection's constant-scores
trap, but the same family of "schema-compatible, semantically divergent."

Note also that the **crop requirement varies by method**
(`run_eye_masks_batch.py:595-596`): traditional eye segmentation requires
*materialized* crops, while U-Net/YOLO accept geometry-only — a real behavioral
divergence between the backends that callers must track.

### The U-Net path is well-built (the modern canonical)

`infer_unet_subject_masks.py` is performance- and provenance-conscious:
multi-channel `mask_probs_roi` with configurable precision (uint8-quantized or
float16, lines 527-545), optional binary `masks_roi`, rich on-device spatial
metrics (`prob_max`, `mask_present`, `area_px`, `centroid_xy`, `bbox_xyxy`, …,
lines 721-727), async writers with explicit error propagation, masks in ROI
space (consistent with the crop-centric convention), `detection_source` carried
forward, and `build_stage_provenance` + run-completion. The same quality as the
keypoints stage.

### Net

The modern U-Net mask path is well-engineered and consistent with the other good
model-based stages. The problem is the family's breadth: three eye-mask backends
and three subject-mask approaches (~376 KB), with classical CV as the *default*
eye method. The model is shared but the harnesses are not. Alongside crop, this
is where the owner's bloat/duplication concern is most concretely justified — and
a candidate for consolidating to one canonical backend per mask type (with the
others explicitly marked legacy) once U-Net parity is confirmed.

---

## Pattern across the model-based half

The newer, model-based stages (`detect_yolo`, `detect_quality`,
`detect_keypoints_yolo`, `infer_unet_subject_masks`) are consistently well-built
— normalized/validated outputs, real stage-provenance, run-completion,
coordinate-aware, performance-conscious. The pipeline's risk concentrates instead
in (a) the orchestration split (legacy monolith vs registry runners, traditional
vs YOLO), (b) the foundational import stage's inline-provenance bypass and decode
choices, (c) crop's accreted mass, and (d) the segmentation family's breadth
(three eye backends, classical CV as default). The science kernels are largely
sound; the seams between stages, the multiplicity of per-stage backends, and the
legacy plumbing are where the exposure lives.

---

## Analysis & geometry side (`shared/coordinate_transform.py` + `analysis/`)

The downstream science layer — calibration, projector↔camera↔mm transforms,
visual angle, stimulus/camera frame alignment, behavioral metrics — flagged in
`repo_enforcement_review_2026-06-04.md` as the highest "silently-wrong-science"
risk. Investigated directly; the picture is narrower and more reassuring than
that flag implied, with one genuine residual gap.

### What is sound

- **The transform math is correct by inspection and pure/testable.**
  `coordinate_transform.py` (326 LOC) is the linchpin (consumed by ~10 analysis
  modules: bouts, kinematics, chaser metrics, stimulus import). `projector_to_camera_px`
  applies a standard homography (homogeneous → matmul → normalize by w, with a
  `|w|<1e-12` divide guard, lines 190-207); `projector_to_camera_mm` is
  `cam_px × pixel_to_mm`; `visual_angle_deg` is `2·arctan(r/z_eff)` (the standard
  formula). The modern path is mm-calibrated and homography-based — more rigorous
  than `steps.md`'s legacy "texture × 12.604" scaling.
- **Calibration loading IS tested.** `test_coordinate_transform.py` (203 LOC)
  covers `load_calibration_transform`: field loading, the
  `pixels_per_mm → 1/pixel_to_mm` inversion, missing-field handling, and the
  source-precedence fallbacks (root and stimulus-run). So the merge *precedence*
  is pinned — contrary to the enforcement review's "parsing only, nothing
  guarded" framing, the loading logic is genuinely covered.
- **Frame alignment is timestamp-based, not a naive ratio.**
  `chaser_state_interpolator.py` aligns stimulus (≈120 Hz, `DEFAULT_TIMESTAMP_DELTA_NS
  = 8_333_333`) to camera frames by interpolating on timestamps, not by assuming
  a fixed 2:1 frame ratio as `steps.md` described. This is the robust approach and
  avoids the drift a fixed ratio would accumulate.

### The genuine residual risks

1. **Homography direction/validity is unchecked — the real silent-science
   exposure.** `_read_homography` (lines 60-69) validates only the 3×3 shape. The
   function names assume projector→camera, but nothing verifies the stored matrix
   is in that direction (vs the inverse), is non-degenerate, or maps known
   projector points into camera bounds. A stored inverse or ill-conditioned
   homography would produce plausible-looking but wrong mm/visual-angle values
   with no error. This is the one place a calibration mistake propagates silently
   into every spatial metric.
2. **The transform math itself is value-untested.** The test imports
   `projector_to_camera_px` / `visual_angle_deg` / etc. but exercises only
   calibration loading — there is no known-point round-trip (projector point →
   expected camera pixel) and no visual-angle value assertion. The math is simple
   enough to be right by inspection, but it is unguarded against regression.
3. **First-source-wins merge can mix fields across calibration sources.**
   `_merge_calibration_group` fills each field only if still `None` (lines 72-106),
   with precedence `analysis/calibration` → `root/calibration` → stimulus-run. So
   a partial `analysis/calibration` can supply the homography while `pixel_to_mm`
   comes from a *different* source, with no check that the two originate from the
   same calibration session. The precedence is tested; the cross-field
   *consistency* is not.

### Recommendations (small, high-value)

- Add a **homography sanity check** at load: project the arena/dish corners and
  assert they land within camera bounds (and reject near-singular matrices). This
  closes the single highest silent-science gap cheaply.
- Add **value round-trip tests**: a known projector point → expected camera
  pixel, and a known `(radius_mm, z_eff_mm)` → expected visual angle. Turns
  "right by inspection" into "guarded."
- Stamp the **calibration source** onto each loaded field (or refuse mixed-source
  homography + pixel_to_mm) so a mismatched pair is detectable.

### Heatmaps / behavioral metrics

The heatmap and metric generators (`roi_heatmap_generator`, the
`analysis/*_metrics`/`*_kinematics` modules) are consumers of the transforms
above — their spatial correctness *inherits* from the calibration. They were not
read in depth this pass; given the transform math is sound and frame alignment is
timestamp-based, their main exposure is the same unchecked-homography risk
upstream, not their own arithmetic.

### Net

The downstream science is in better shape than the enforcement flag suggested:
correct transform math, tested calibration loading, robust timestamp-based frame
alignment. The one real gap is **no validation that the stored homography is the
right direction and well-conditioned** — a cheap fix (corner round-trip check)
that would close the highest silent-science exposure in the codebase. The
"untested coordinate transforms" concern is real but specific: it is the
homography *direction*, not the arithmetic, that lacks a guard.
