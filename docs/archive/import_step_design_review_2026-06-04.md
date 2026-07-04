<!-- ARCHIVED 2026-07-04: dated point-in-time snapshot / spent work ticket, retained for history only. -->

# Import Step Design Review — 2026-06-04

Scope: review of the video import stage as implemented in
`src/fisheye/capture/import_video.py`, plus how it is invoked
(`src/fisheye/core/pipeline.py`) and the Zarr archive it produces
(`src/fisheye/shared/zarr/schema.py`).

This is a point-in-time review, not a change. Line numbers reference the
files as of this date.

## What the import step does

Turns a source video into a Zarr v3 archive under `raw_video/`
(`images_full`, `images_ds`, `images_ds_rgb`, `timestamps`, and
`original_frame_indices` for sampled imports).

The fast path is the well-built part:

- **decord (GPU decode) → torch → cupy (zero-copy `from_dlpack`) → kvikIO
  `GDSStore`** writes directly from GPU memory to disk via GPUDirect Storage.
  Grayscale conversion and downsampling run on-GPU in fp16.
- **Zarr v3 sharding** with shard sizing targeting ~128 MB
  (`schema._auto_shard_frames`); writes are aligned to shard boundaries and
  are single-process/sequential, so they are chunk-safe.
- **Frame sampling** for training imports, preserving the source-frame map in
  `raw_video/original_frame_indices`.
- **Heavy provenance capture** — git, platform, GPU telemetry, LSF/SLURM job
  info, package versions — for HPC reproducibility.
- **Process isolation** via `fork()` + `_exit(0)` to avoid CUDA/decord
  segfaults during interpreter shutdown.

The GPU decode→GDS→Zarr core is the product of real performance work. The
issues below are mostly at the edges: a second code path that does not work,
contract drift between the writer and its validator, and some fragile
process/import-time handling.

## Findings, roughly by severity

### 1. CPU / non-kvikIO path is a stub that leaves a corrupt archive
`import_video.py:1084-1087`. `--cpu-only` / `force_cpu` is fully plumbed
through `main()` → `import_video()` and from `_run_import`. The standard path
runs `create_palette_zarr` (creating arrays + root attrs) and *then* hits
`raise NotImplementedError("CPU processing path not shown")`. Result: a
half-initialized Zarr on disk followed by a crash, rather than a fast failure.
The "not implemented in this example" comment suggests this was adapted from
example code and never completed.

### 2. Module cannot be imported on a CPU-only host — optional-deps guards are dead code
Lines 19-24 hard-import `torch`, `decord`, `cupy` at module top level. The
`try/except` guards immediately below (`_HAVE_CUPY`, `_HAVE_KVIKIO`,
lines 37-48) can never see an absent cupy/torch because the hard imports
already failed 14 lines earlier. The graceful-degradation story is illusory.

### 3. Top-level import side effect mutates global GDS config
Lines 12-14: `os.environ["KVIKIO_COMPAT_MODE"] = "OFF"` runs on *any* import of
this module, forcing GDS-only mode process-wide — even on hosts without GDS and
for unrelated code that later imports kvikIO. This belongs inside the function,
gated on capability detection.

### 4. `fork()` with CUDA is a latent landmine
CUDA contexts do not survive `fork()`. This works today only because import is
normally the first stage, so the parent has not initialized CUDA before the
fork. But `core/pipeline.py` runs many stages in one process; anything that
touches CUDA before `_run_import` would make the forked child's CUDA calls
undefined behavior. The module already imports `subprocess`; a `spawn`-based
subprocess (or `subprocess.run([sys.executable, ...])`) would be safe and would
also propagate child errors. Today the child's traceback prints only to its own
console and the parent re-raises a bare `RuntimeError(exit_code)`
(`import_video.py:1364`).

### 5. Writer/validator contract drift: `import_stage` and `total_frames`
- `import_stage`: kvikIO path writes `"complete"` (line 906); standard path
  writes `"full_resolution"` (line 986). `validate_import` checks
  `!= 'complete'` (line 1395), so it would always report the standard path as
  failed even if it worked.
- `total_frames`: kvikIO path stores `n_import_frames` (line 909); standard
  path stores full `n_frames` (line 990); `create_palette_zarr` sets the *root*
  `total_frames` from full `n_frames`. For sampled / tail-skipped imports these
  disagree within one archive.

### 6. Validation is shallow and assumes `images_full` exists
`_validate_stage` (`pipeline.py:1044`) calls `get_import_stats`, which
hard-requires `raw_video/images_full` (`import_video.py:1408`). A
`resolutions: downsampled` import passes import but crashes validation. The
stricter frame-count check `validate_import` is never wired into the pipeline.

### 7. `timestamps` is created but never populated
Allocated `fill_value=NaN` in `_finalize_kvikio_zarr_metadata` (line 520) and
never written. Given how much downstream depends on 60↔120 fps frame alignment
(see `steps.md`), an all-NaN `timestamps` array is a trap. decord exposes
`get_frame_timestamp()`; capturing real PTS here would be cheap.

### 8. fps silently defaults to 30
Lines 908 and 989 use `vid_meta.get("fps", 30)`. This rig records at 60 fps. A
missing or zero `get_avg_fps()` quietly poisons `duration_seconds` and any
downstream timing. This should fail loudly rather than guess.

### 9. Two sources of truth for array specs
Array shape/chunk/shard logic is duplicated between the creation block
(lines 821-889) and `_finalize_kvikio_zarr_metadata` (lines 486-518), and the
two do not fully agree (`compressors=None` vs `compressors=[]`; finalize does
not re-set the `format` / `resolution` attrs the creation path set). They must
be kept in lockstep by hand. A single shared helper would remove the drift.

## Decode backend: no pynvvc path in import (and a parity gap)

The import step decodes **exclusively through decord** — `decord.gpu(0)` for GPU
and `decord.cpu()` for CPU (`import_video._setup_video_reader`, lines 529-546).
There is no `PyNvVideoCodec` (pynvvc) reference anywhere in `capture/`.

pynvvc exists elsewhere: `shared/pynvvc_luma_rgb.py` (`PynvvcLumaRgbReader`,
NV12/luma→RGB preprocessing) is wired into **detection** (`detect_yolo.py`),
training-crop generation, and the flat-ROI caches — not import.

Two asymmetries follow:

1. **Backend choice is asymmetric.** Detection exposes a configurable decode
   backend (`auto / pynvvc_nv12_rgb / pynvvc_luma_rgb / decord_gpu / decord_cpu
   / opencv`; `detect_yolo.py:56-65`). Import is hardwired to decord with no
   config knob.

2. **Detection re-decodes the source video; it does not consume the import's
   stored frames.** `detect_yolo` takes `video_path` and builds its own reader
   (`_init_decord_reader(video_path, …)` / `PynvvcLumaRgbReader(video_path)`).
   It consults `raw_video/images_full` and `images_ds` only for
   metadata/presence (`detect_yolo.py:122-166`), not as the inference input.

   Consequence: the frames import decodes-and-stores (decord) are not
   guaranteed to be the frames detection runs on (potentially pynvvc). decord
   vs PyNvVideoCodec can differ in NV12 luma/chroma handling and resize, so the
   archived `images_*` arrays and the model's actual input can diverge at the
   pixel level. This is the same gap the repo's own diagnostics already track:
   `diagnostics/compare_detect_decode_backend_predictions.py`,
   `diagnostics/check_training_crop_pynvvc_pixel_parity.py`,
   `docs/inference_pipeline_divergence_analysis.md`,
   `docs/detect_decode_backend_benchmark_todo.md`.

   Open question for the pipeline owners: are the import's `images_full` /
   `images_ds` meant to be the canonical inference input (in which case
   detection should read them, or at least share import's decode backend), or
   are they only for review/visualization while detection owns its own decode?
   The current split leaves that ambiguous.

## `images_full` / `images_ds` as canonical training pixels: intent vs reality

Stated design intent (from the pipeline owner): `raw_video/images_full` and
`images_ds` are meant to be a stable "training file" artifact — the canonical,
inspectable, decode-once source of pixels for building merged training datasets
across all training zarrs in a recordings directory, so that downstream training
never has to follow video files around or re-decode frames itself.

Where the intent **is** honored:

- **Per-recording crop reads the zarr first.** `tracking/crop.py:682-685`
  (`_resolve_video_source`) returns `('zarr', None)` when `raw_video` is present
  and only falls back to the external video otherwise ("Try zarr first because
  it is typically faster").
- **Training archives forbid geometry-only crops.** `tracking/crop.py:176-180`
  raises if `crop_storage_mode != "materialized"` for a training archive, so a
  training zarr cannot degrade into store-coordinates / re-decode-later.
- The pixel contract names the zarr-read backend `read_slice` →
  `canonical_reader_path` and the video-re-decode backend `pynvvc_luma` →
  `provisional_until_crop_pixel_parity_passes` (`shared/roi_pixel_contract.py`).

Where the intent **breaks** — at the cross-recording merge step:

- **`utils/create_clipped_training_zarr.py` does not read `images_full` /
  `images_ds`; it re-decodes the source videos with OpenCV.**
  `_decode_clip_frames` (line 229) opens each `video_path` with
  `cv2.VideoCapture`, seeks per frame via `cap.set(CAP_PROP_POS_FRAMES, …)` +
  `cap.read()`, BGR→GRAY converts, and writes the merged zarr's `images_full` /
  `images_ds` from those frames (lines 556-588). It is keyed off a table of
  `(video_path, clip_local_frame_index)`.
- **`utils/regenerate_training_crops_pynvvc.py`** rewrites materialized
  `roi_images` by decoding the video via PyNvVideoCodec (docstring lines 3-4).

Net effect — three decoders produce pixels all treated as "the same frames":

| Stage | Decoder | Pixel source |
|---|---|---|
| Import → `images_full` / `images_ds` | decord | video → zarr |
| Per-recording crop | reads zarr (decord pixels) | `images_full` / `images_ds` |
| Merged / clipped training zarr | OpenCV (cv2) | re-decodes video |
| Detect / regenerate training crops / geometry-only cache | PyNvVideoCodec | re-decodes video |

Two consequences:

1. The merged training dataset reintroduces the exact video dependency the
   import artifact was meant to remove — it requires every source `video_path`
   reachable and re-decodes them, instead of copying the canonical `images_full`
   already decoded at import.
2. cv2 `CAP_PROP_POS_FRAMES` random seeking is unreliable on long-GOP H.264 /
   H.265 and can silently land on the wrong frame near non-keyframes — a
   mislabeled-frame risk that is worse for a training set than for analysis.
   This is on top of the decord-vs-pynvvc-vs-cv2 pixel-parity question the
   repo's own diagnostics already track
   (`diagnostics/check_training_crop_pynvvc_pixel_parity.py`,
   `diagnostics/compare_detect_decode_backend_predictions.py`).

Suggested direction: `create_clipped_training_zarr._decode_clip_frames` is
deliberately isolated (docstring: "intentionally small so… a future
PyNvVideoCodec backend can replace only this layer"). Make that layer **copy
slices from the source recording's `images_full` / `images_ds` when a source
zarr is available** — the clip table already carries the mapping and the file
already validates `raw_video/original_frame_indices` against the clipped frames
(lines 442-459) — and decode from video only as a fallback. That makes the
merged dataset actually inherit the canonical import pixels and collapses three
decoders back toward one.

## Human-curation → training promotion: uneven across label types

Stated intent (pipeline owner): when an operator corrects a label, the
correction is promoted into a training zarr, so the entire training pool is
human-curated. This should hold for detections, keypoints, and segmentations.
There are two conceptual tiers: per-recording `<filename>_training.zarr` and
merged (cross-recording) training zarrs.

What is actually built is two tiers using two different mechanisms, and the
coverage is uneven.

**Merged tier — uniform for all three label types.** A parallel family of
registry-driven exporters exists: `prepare_detect_training_from_registry.py`,
`prepare_keypoint_training_from_registry.py`,
`prepare_pose_training_from_registry.py`,
`prepare_eye_mask_training_from_registry.py`,
`prepare_subject_mask_training_from_registry.py`. All query the registry and
gate on review state (`--require-review-state approved
--require-review-intended-use training`). So "only human-approved labels enter
the merged training pool" is enforced for detect, keypoints, and segmentations.

**Per-recording `_training.zarr` promotion tier — detections only.** Only detect
has an explicit promotion path that copies operator-corrected labels from the
analysis zarr into that recording's `_training.zarr`:

- backend `tune/detect_training_promotion_backend.py`
- CLI `utils/promote_analysis_detect_to_training.py`
- save-hook in `tune/video_detect_review_web.py` (`--edit --promote-training-zarr`)
- promoted rows mirrored into `refined_detect_runs/<run>/instances` for the
  merged exporter
- `docs/analysis_to_training_promotion_contract.md` scope: "detect bbox
  promotion from analysis zarr to per-recording training zarr."

There is no `promote_analysis_keypoints_to_training` or `…mask…` equivalent —
the only promotion backend/CLI in the tree is the detect pair. Keypoint and mask
corrections instead live as review-approved refined runs inside the analysis
zarr and are selected directly from the registry at export time.

| Label type | Review-gated merge tier | Promote → per-recording `_training.zarr` |
|---|---|---|
| Detections | yes | yes (Crimson auto-save hook still pending) |
| Keypoints | yes (`approved`/`training`) | none |
| Segmentations (eye + subject masks) | yes | none |

**Why the gap matters — but the keypoint training read is safer than it first
looks.** The per-recording `_training.zarr` is what gives detect the property the
import artifact was designed for: corrected labels travel with materialized
canonical pixels (`images_full`/`images_ds`), decode-once, inspectable, no video
chasing. Keypoints/masks skip that promotion tier, but a trace of the keypoint
training read (2026-06-04) shows the trainer does NOT re-decode video:

- `utils/prepare_keypoint_training_from_registry.py` only emits a `PoseConfig`
  manifest of `{zarr_path, source_crop_run, keypoint_run}` after review-state
  gating (lines 800-801); it reads no pixels.
- `training/train_pose.py` delegates to `create_zarr_dataset` /
  `ZarrDatasetConfig` (line 66).
- `training/zarr_yolo_dataset_loader.py` resolves the frame source to
  `crop_runs/<run>/roi_images` (lines 477-478, 519-520, 572) and hard-requires
  it (line 1167 accesses `roi_images.shape` unconditionally, so geometry-only
  crop runs raise `KeyError`).
- `_get_detect_frame` (lines 1283-1314) reads `roi_images[frame_idx]` as a plain
  stored Zarr slice behind an LRU chunk cache — no decord/pynvvc/cv2 and no
  `CropImageSource` live reader.

So keypoint/pose training reads materialized, decode-once stored pixels and
cannot silently fall back to re-decoding.

A follow-up trace (2026-06-04) confirms the same holds for both mask trainers:

- Eye masks: `utils/export_eye_mask_training_zarr.py` reads
  `crop_runs/<run>/roi_images`, hard-requires it (lines 670-671), and copies the
  stored pixels into a physically merged training zarr (`roi_images_dest`,
  line 1543). `segmentation/train_unet_eye_masks.py` reads
  `store.roi_array[start:stop]` (line 140) as a plain stored Zarr slice.
- Subject masks: `training/zarr_subject_mask_dataset.py:198` sets
  `roi_array = crop_group["roi_images"]` (unconditional, so geometry-only runs
  raise `KeyError`); `SubjectMaskChunkedDataset.__getitem__` reads stored slices
  via `_StoreChunkCache`.

Uniform result across all four training label types:

| Label type | Prep output | Terminal pixel read | Re-decode at train time? |
|---|---|---|---|
| Detect (crop-based) | manifest | `crop_runs/<run>/roi_images` slice | no |
| Keypoints / pose | manifest | `crop_runs/<run>/roi_images` slice | no |
| Eye masks | export → merged zarr (copies `roi_images`) | exported `roi_images` slice | no |
| Subject masks | manifest | `crop_runs/<run>/roi_images` slice | no |

Every training path reads materialized `roi_images` as plain stored Zarr slices
and hard-requires `roi_images`, so geometry-only crop runs cannot be trained
from at all. The decode-backend choice (and the decord-vs-pynvvc-vs-cv2 parity
question) therefore lives entirely *upstream*, at crop materialization in
`tracking/crop.py`, which derives `roi_images` from `raw_video/images_*`
(decord) when present and re-decodes the source video only when `raw_video` is
absent. The residual exposure is thus narrow: a recording whose analysis zarr
lacks `raw_video` will have had its training crops materialized from a fresh
video decode rather than from the canonical import pixels.

The `create_clipped_training_zarr.py` cv2 re-decode flagged earlier is
correspondingly narrower than it first appeared — it is specific to the detect
clipped-finalized-collection training-zarr builder, not the crop→`roi_images`
pipeline that keypoints and masks ride on. One structural difference remains:
eye masks physically merge into a new training zarr (copying `roi_images`),
while keypoints and subject masks keep a manifest of source zarrs and read each
in place; the stored-pixel guarantee is the same either way.

Two design questions for the owners:

1. Mechanism consistency — give keypoints and masks the same
   promote-into-`_training.zarr` flow detect has, or accept "review-gated
   refined runs in the analysis zarr" as the canonical curation surface for
   those two and treat detect's promotion tier as the exception.
2. Pixel canonicality — if keypoints/masks stay export-from-analysis, force
   their reviewed source recordings to be materialized (or copy from
   `images_full`) so approved training crops are also decode-once-canonical.

## Proposal: unify the decode backend around PyNvVideoCodec (pynvvc)

The findings above leave the pipeline with three decoders feeding pixels that
are all treated as interchangeable: decord (import → `images_full`/`images_ds`),
OpenCV/cv2 (`create_clipped_training_zarr`), and PyNvVideoCodec (detection,
`regenerate_training_crops_pynvvc`, geometry-only flat cache). The cleanest way
to make stored training pixels bit-identical to what the model sees is to
collapse these to one backend. Detection already runs pynvvc, so standardizing
on **PyNvVideoCodec NV12 luma (Y′) plane, uint8** is the lowest-divergence
target: `raw_video/images_full` would then be exactly the model's input surface.

### Why pynvvc is the right unification target

- It already exists and is wired: `shared/pynvvc_luma_rgb.py` (`PynvvcLumaRgbReader`,
  `preprocess_luma_rgb`), with a named pixel contract
  (`ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME`) and parity diagnostics
  (`diagnostics/check_training_crop_pynvvc_pixel_parity.py`,
  `diagnostics/compare_detect_decode_backend_predictions.py`).
- The reader yields CUDA tensors via dlpack (`decode_next` / `iter_frames`,
  `pynvvc_luma_rgb.py:48-86`), so it drops straight into the import's existing
  zero-copy `cupy → kvikIO GDS` write path — no architecture change to the fast
  write.
- The NV12 luma plane (`frame[:source_height, :]`) is already grayscale, so
  import could store the Y′ plane directly and drop the BT.601 RGB→gray
  weighting it does today. Same pixels detection consumes.

### Concrete change points

1. **Import (`capture/import_video.py`)** — replace `_setup_video_reader`'s
   decord reader with `PynvvcLumaRgbReader`; fill the GPU buffer from the Y′
   plane crop `[:H, :W]` (drops NV12 pitch padding) instead of RGB→gray. Keep
   the cupy→kvikIO GDS write and on-GPU downsample unchanged.
2. **Crop fallback (`tracking/crop.py`)** — when `raw_video` is absent and a
   re-decode is required, use pynvvc (via the existing `_crop_pynvvc_luma_frame`
   in `shared/flat_roi_cache.py`) instead of decord, so the fallback matches
   stored pixels.
3. **Clipped builder (`create_clipped_training_zarr._decode_clip_frames`)** —
   replace cv2 with pynvvc; the docstring already anticipates this ("a future
   PyNvVideoCodec backend can replace only this layer"). Better still, copy from
   the source recording's `images_full` when present and only pynvvc-decode as
   fallback.
4. **Single decode module** — factor one canonical decode entry point (extend
   `shared/pynvvc_luma_rgb.py`) that import, crop-fallback, and the clipped
   builder all call, eliminating the three-sources-of-truth problem.
5. **Promote the contract** — once parity passes, flip the `pynvvc_luma` pixel
   contract from `production_status="provisional_until_crop_pixel_parity_passes"`
   to canonical (`shared/roi_pixel_contract.py`) and retire the assumption that
   `read_slice` pixels originate from decord.

### Open issues to resolve before committing

- **GPU-only.** pynvvc requires NVIDIA NVDEC; there is no CPU decode path. This
  hard-locks decode to NVIDIA hardware (acceptable for this lab, but it should
  be a stated decision, and it pairs naturally with fixing the stubbed CPU
  import path — finding #1 — by making decode fail fast without a GPU).
- **Colorimetry / range.** NV12 Y′ is typically limited-range (16–235),
  BT.601/709 per encode; decord's RGB→gray was full-range. Storing raw Y′ shifts
  values versus legacy archives. Parity with the model input argues for storing
  raw Y′ and recording `color_range` and `color_matrix` in the pixel contract /
  `raw_video` attrs rather than normalizing.
- **Sparse / random access.** `PynvvcLumaRgbReader` is sequential and
  start-at-zero (`pynvvc_luma_rgb.py:21-25`). Sampled training imports
  (`--frame-step`) and clipped collections that need specific frames must either
  decode sequentially and select, or gain a seek capability. Quantify the cost
  of decoding everything when sampling a small fraction.
- **Pitched surfaces.** NV12 decode surfaces can be pitch-padded; confirm the
  `[:H, :W]` luma crop plus `.contiguous()` yields a clean buffer for the
  kvikIO write at every resolution.
- **Migration / mixed provenance.** Existing decord-imported zarrs become
  off-contract. Stamp `decode_backend` on `raw_video` and every crop run, use
  `regenerate_training_crops_pynvvc.py` (or re-import) to rewrite legacy
  `roi_images`, and make the merged-dataset prep refuse to silently mix decode
  backends.

### Validation

Gate the switch on the existing parity diagnostics
(`check_training_crop_pynvvc_pixel_parity.py`,
`compare_detect_decode_backend_predictions.py`) — they should pass trivially
once both sides decode with pynvvc — and add a decode-backend-consistency
assertion to the `prepare_*_training_from_registry.py` exporters.

### Suggested phasing

- Phase 0 — stamp `decode_backend` provenance everywhere (cheap; makes
  mixed-backend data detectable before changing any pixels).
- Phase 1 — swap the isolated, low-risk paths: clipped builder cv2→pynvvc and
  crop-fallback decord→pynvvc.
- Phase 2 — switch import decord→pynvvc luma; record colorimetry; resolve
  sparse-access for sampled imports.
- Phase 3 — promote the pynvvc_luma contract to canonical, add the
  backend-consistency gate, and regenerate/migrate legacy crop runs.

## Design-level observation

The import step dumps ~50 provenance/system attributes directly onto the
`raw_video` group inline (lines 1110-1289), while the repo has an entire
registry + stage-provenance subsystem (`registry/`,
`shared/stage_provenance.py`, `docs/provenance_contract_draft.md`,
`docs/pipeline_metadata_boundaries.md`). Import predates or bypasses that
contract. Open question: is import supposed to route provenance through the
registry like later stages, making this the oldest stage carrying a legacy
inline pattern the rest of the pipeline has moved past?

## Net

The GPU decode→GDS→Zarr core is well-designed. The weaknesses are: an
unfinished CPU path that should be implemented or made to fail fast; fragile
process/import-time handling; and several writer↔validator contract mismatches
that bite specifically on the non-default modes (CPU, downsampled-only,
sampled).

## Suggested follow-ups (not yet done)

- Decide CPU path: implement it or make `--cpu-only` fail fast before any Zarr
  is created.
- Move `KVIKIO_COMPAT_MODE` and the hard cupy/torch/decord imports behind
  capability checks so the module imports on CPU-only hosts.
- Replace `fork()` with a spawned subprocess and propagate child errors.
- Reconcile `import_stage` / `total_frames` semantics across both writers and
  the validators; wire `validate_import` into `_validate_stage`.
- Populate `timestamps` from decord PTS, or remove the array.
- Fail loudly on missing fps.
- Unify array-spec creation between the create and finalize paths.
