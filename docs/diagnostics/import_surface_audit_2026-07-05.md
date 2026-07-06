# Import surface audit — 2026-07-05

status: active — diagnostic. Read-only investigation (3-agent parallel sweep) of the
experiment-import surface: the import structures, registry linkage, and metadata-capture
reliability. No code changed. Seeds the "bring import inside the perimeter" brief family.

## Motivating question

How does an experiment get into Palette? Specifically: what import structures exist, is
the registry updated on import, and are we reliably capturing metadata at import time?

## TLDR

Import is the root of the pipeline DAG and the **least-governed stage in it**. Everything
from `detect` onward now has fail-closed completion markers, enforced run provenance
(epoch 2), and — as of `62a8e52` — content-hashed model artifacts. Import has none of
that: structural completion detection, flag-gated registration, and best-effort-everything
metadata. This is the 2026-07-01 review's throughline ("disciplined core, silent-wrong-data
at the boundaries") resurfacing at the one boundary the remediation never reached.

Three headline answers:
1. **Two coexisting import paths** capture different things; the one production runs is the
   metadata-only path, and the richer provenance block lives on the *legacy* path.
2. **The registry IS updated at import — but only if `--registry` is passed** (default-on in
   the cluster chain, absent for hand-runs), with no enforcement that stores are registered
   before stages run, and a manual-only reconcile safety net.
3. **Metadata capture is NOT reliable** — loud on structural failure, silent on metadata
   absence. A recording can import "successfully" with `unknown` codec, all-NaN timestamps,
   no source fingerprint, no colorimetry, and zero experiment identity.

---

## 1. The import structures — two coexisting paths

### Path A — pixel importer (`capture/import_video.py`, ~1687 lines)
GPU-decodes frames into a zarr. **Driven by the LEGACY orchestrator** (`core/pipeline.py`,
which prints `LEGACY_ORCHESTRATOR_NOTICE`). Also used for sampled/training imports via
pynvvc.
- **Produces:** `raw_video/images_full` (n,H,W), `images_ds` (downsampled gray), optional
  `images_ds_rgb`, `timestamps` (see gap #1), BT.601 grayscale via the newly-centralized
  `shared/grayscale.py`.
- **Frame-domain identity:** `stamp_stored_zarr_frame_identity_mapping()`
  (`import_video.py:99`) writes `raw_video/frame_domain_maps/stored_zarr_frame_to_acquisition_frame`
  = `arange(n)`, `semantics="identity_map_zero_based_full_import"`; sampled imports write
  `raw_video/original_frame_indices` instead. These feed `FrameDomains._build_raw_edges()`.
- **Provenance:** writes a rich git/branch/dirty + host/GPU/GDS + LSF/SLURM block to
  `raw_video.attrs` (`:1255-1405`) — but this is inline attrs, NOT `mark_run_complete` /
  completion-epoch (see §4).
- **Process model:** forks a child, `_exit(0)` to skip atexit (avoids CUDA-teardown
  segfault); parent `waitpid`s.
- **Dead code:** CPU path raises `NotImplementedError` (`:533`, `:1200`).
- **Self-labeled legacy pixels:** `_decode_contract_metadata` (`:626`) marks its output
  `legacy_decord_pending_pynvvc_unification`; canonical target is `pynvvc_luma`
  (`utils/import_sampled_training_pynvvc.py`).

### Path B — production metadata-only path (what the `raw` verb + cluster run)
No frames decoded. Verb wiring: `cli/palette.py:689` (`stage_id=="raw"`, verb `"import"`) →
`fisheye.utils.import_organized_recordings_analysis --registry <db> <dir>`.
- **Batch entry:** `import_organized_recordings_analysis.py::main` (`:338`) — discovers
  recording dirs from `$PALETTE_RECORDINGS_ROOT` (default `/nvme1/recordings`) or an
  organize-log JSONL; preflight gate (`shared/recording_preflight.py::preflight_gate_reason`
  `:194`); per recording → `process_recording_import` then `Registry.scan_zarr`.
- **Per-recording engine:** `utils/import_recording_analysis.py::process_recording_import`
  (`:201`): (1) `ensure_analysis_archive` stamps `zarr_purpose="analysis"`, session_uuid,
  recording_id/type, camera/rig/arena/protocol/genotype from `recording_manifest.json`,
  plus `write_acquisition_video_stream_inventory`; (2) `apply_video_metadata` →
  `shared/import_video_metadata.py::write_video_metadata` probes via **OpenCV+ffprobe**,
  writes `import_method="metadata_only"`, `import_stage="metadata_only"`, frame counts,
  codec (NO pixel arrays); (3) `run_stimulus_import` subprocess → copies stimulus/chaser/
  events/protocol/calibration from the H5 into `analysis/stimulus_runs/`.
- `utils/import_video_metadata.py` is a thin CLI wrapper over the shared module
  (attrs-only onto an existing store; NOT a pure shim — it owns a live CLI, per the
  2026-07-05 utils-shim census).

### Conflict flag
Both paths write the same `raw_video` group with **conflicting** `import_method` /
`import_stage` / `import_mode` values (`complete`/`full` vs `metadata_only`). The richer
provenance block is written only by Path A (legacy); Path B (production) omits it.

### Operational arrival of a new experiment (today)
Citrus transfer session → `submit_citrus_session_import_bsub.sh` (one LSF job/session,
`--registry` default-on) → `run_citrus_session_import.py`: organize → metadata import →
registry scan. **The transfer watcher does not exist yet** —
`docs/interface_and_execution_strategy.md:85-88,124` marks it 🆕 to-build; a human/cron
submits today.

---

## 2. Registry linkage — is the registry updated on import?

**Yes, synchronously — but conditionally.**
- `capture/import_video.py` has **zero** registry references. The pixel path never registers.
- The batch wrapper's `_sync_registry` (`import_organized_recordings_analysis.py:246`) calls
  `Registry.scan_zarr(zarr_path)` (`registry/db.py:8198`) → `register_from_root`
  (`db.py:6609`) → full extractor sweep in one transaction, including
  `replace_acquisition_video_streams` (`db.py:6670`), datasets, recordings, provenance,
  quality/performance tables. This is a **direct write at import**, not deferred to reconcile.
- **BUT `--registry` is optional** (argparse `:417`, no `required`, no default). A bare
  import writes nothing. The cluster submitter defaults it on; hand-runs and `--no-register`
  do not.
- **`acquisition_video_streams` rows** appear at import (if `--registry`) or at first
  reconcile/rescan otherwise — re-derived from zarr truth on every scan (DELETE-then-INSERT
  `replace_*`), never waiting for stage completion.
- **No guarantee stages run only on registered stores.** `_with_optional_registry`
  (`cli/palette.py:674`) appends the registry only if present; pipeline stages run on
  unregistered zarrs.
- **Reconcile is the safety net but is manual-only:** no reconcile/sync verb in
  `cli/palette.py`, no cron. Triggers are `registry.maintenance --reconcile-dataset/
  --reconcile-registry`, `utils/registry_rescan.py`, or the import wrapper's inline
  `scan_zarr`. `reconcile_dataset_from_root` (`db.py:6747`) is idempotent (tests in
  `2a74f2e`); moved stores are marked `status='missing'` (path not rewritten).
- **Designed intent vs reality:** `docs/diagnostics/registry_design_assessment_2026-06-18.md`
  §Weaknesses#1 names it — "capture is scattered and not idempotent-by-design." The completed
  `brief_registry_reconcile.md` built `reconcile_dataset_from_root` as the unifying engine,
  but nothing auto-invokes it and import-time `scan_zarr` is still a separate path.

---

## 3. Metadata capture reliability

**Loud on structural failure, silent on metadata absence.** Hard-fails are narrow: missing
video file (`:762`), zero frames (`:798`), `skip_tail_frames >= n` (`:810`). Everything else
degrades silently.

Capture summary (item → where it lands → status):
- **Video stream props** (fps, w/h, total_frames, codec, pix_fmt): `raw_video.attrs` /
  root. total_frames mandatory; codec/pix_fmt best-effort → `"unknown"` on any ffprobe/iio
  hiccup (`:698 except: pass`), stored as if authoritative.
- **Encoder tags** (`shared/encoder_tags.py::parse_encoder_comment`): `raw_video.attrs` +
  registry provenance columns; best-effort (empty dict, silent, if no MP4 comment).
- **Decode contract** (`decode_backend`, `stored_luma_transform`): mandatory on pixel path
  but a **hardcoded assumption** (`stored_luma_color_range="legacy_decord_rgb_full_range_assumed"`,
  `:641`), not probed.
- **Frame-domain identity maps:** mandatory on their respective paths (the reliable part).
- **System/git/GPU/HPC provenance:** Path A only; whole block in a try/except that only
  prints a warning (`:1409-1412`) — a "successful" import can carry none of it.
- **Experiment metadata** (genotype, dpf, protocol, dish, rig/arena/camera, session_uuid,
  subject/fish, cross/line/species/sex): reaches `recording_manifest.json` +
  registry columns + stimulus snapshots — **only via the H5, only when stimulus import runs.**
  Never stamped into `raw_video` or zarr root as first-class attrs by either importer.

### Reliability mechanics
- Preflight (`recording_preflight.py`) checks only media/tooling + H5 presence/readability;
  gates only on `status=="fail"`, bypassable with `--allow-preflight-failures`. Does NOT
  validate experiment-metadata *content*.
- Manifest required fields (`validate_recording_manifest.py:28-33`) are only recording_type/
  subtype/behavior_mode/artifact_schema_id — and auto-defaulted, not hard-required.
- Post-import checks: `validate_import` checks frame-count + `import_stage` only.
  `schema.validate_zarr_structure` checks attrs but only **warns**, and is **dead code**
  (imported `core/pipeline.py:40`, never called — flagged in
  `contract_drift_audit_2026-05-28.md` #9, still true). The only real metadata auditor is
  `audit_zarr_pixel_contracts.py` — separate, manual, opt-in.

---

## Gap list (honest, prioritized)

Ordered by data-integrity risk. Each is a candidate slice.

1. **`raw_video/timestamps` is created and never written** — all-NaN on every path (only
   ref `import_video.py:597`). No per-frame acquisition-clock mapping exists anywhere.
   → Populate from the decode/H5, or delete the array and stop implying it exists.
2. **No experiment identity in the zarr.** Genotype/protocol/subject/arena/rig/session_uuid
   survive only in manifest + registry + stimulus snapshots, and only when the H5 exists.
   Pixel-only stores know nothing about the fish. → Stamp manifest context into zarr root
   attrs at import (both paths), fail-closed on absence for production imports.
3. **No source-video fingerprint at import.** Nothing detects a swapped/re-encoded source
   MP4; `stat_v1` (`audit_zarr_pixel_contracts.py:1198`) is a manual post-hoc backfill.
   Note the irony: model inputs are now content-hashed (`62a8e52`) but the *video* — the
   primary experimental input — is not. → Fingerprint the source at import (cheap; `stat_v1`
   exists, just move it into the import path).
4. **No colorimetry probed at import** (color range/space/transfer/primaries) — only a
   hardcoded "full-range assumed" string; ffprobe backfill only via the audit tool.
5. **Experiment metadata is never validated.** Nothing fails or warns on missing genotype/
   protocol/subject. → Extend preflight or add a post-import completeness gate.
6. **Silent degradation stored as authoritative:** codec/pix_fmt → `"unknown"`; decode
   contract assumed; system provenance best-effort. → Make these loud or explicitly tag as
   unknown-provenance.
7. **The one completeness validator is dead code** (`validate_zarr_structure`, warns-only,
   never called). → Wire it in as an error gate, or delete it (no `status:active` orphan).
8. **Path A/B convergence + import outside the completion-epoch perimeter** (§4). → The
   structural decision that governs the rest.

---

## 4. Import sits outside the provenance perimeter

- Import IS cataloged: `registry/stage_catalog.py:38` — `StageSpec(id="raw",
  aliases=("import",), category=CORE_PIPELINE)`, dependency-free root of the DAG.
- **Completion is STRUCTURAL, not marker-based:** `core/pipeline.py::_is_stage_complete`
  for `'import'` = `'raw_video' in root and 'images_ds' in root['raw_video']`
  (`:1127-1128`); launcher mirrors it (`interactive_launcher.py:437`).
- **No completion epoch / run provenance on `raw_video`.** The epoch regime
  (`shared/zarr_run_completion.py`, `mark_run_complete`, epoch 2) governs timestamped
  run-groups under stage parents (detect_runs, crop_runs, keypoints_runs…). `raw_video` is a
  singleton group, not a run-group container, so no importer stamps an epoch or run-provenance
  marker. Import's only "provenance" is Path A's inline attr block — which Path B omits.
- Consequence: the fail-closed guarantees built this session stop one stage short of the
  root. Bringing import inside the perimeter (gap #8) is the structural counterpart to the
  content-hash work.

---

## Method / provenance of this audit
Three parallel read-only Opus agents (import-flow map, registry-linkage trace, metadata
reliability audit), 2026-07-05, against `sun` at ~`89b29e7`. All file:line references above
were reported with evidence; spot-checks by the commander session confirmed the load-bearing
claims (two-path split, `--registry` optionality, all-NaN timestamps, dead validator). Related
prior diagnostics: `pixel_contract_audit_2026-06-05.md`, `acquisition_crop_video_integration_2026-06-17.md`,
`contract_drift_audit_2026-05-28.md`, `registry_design_assessment_2026-06-18.md`,
`model_input_shapes_metadata_gaps_2026-06-17.md`.
