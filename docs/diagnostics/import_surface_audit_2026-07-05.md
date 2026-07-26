# Import surface audit — 2026-07-05

status: superseded — historical diagnostic. Read-only investigation (3-agent parallel sweep, followed by
a focused correction pass) of the experiment-import surface: the import structures,
registry linkage, and metadata-capture reliability. No code changed. Seeds the "bring
import inside the perimeter" brief family.

Superseded 2026-07-25: the Decord import writer and broad
`create_palette_zarr` bootstrap described below were retired. Current writers are
`import_recording_analysis` for metadata-only analysis archives and
`import_sampled_training_pynvvc` for sampled training pixels. Historical Decord
attrs remain classifier/read compatibility only.

## Motivating question

How does an experiment get into Palette? Specifically: what import structures exist, is
the registry updated on import, and are we reliably capturing metadata at import time?

## TLDR

Import is the root of the pipeline DAG and the **least-governed active boundary**.
Everything from `detect` onward now has fail-closed completion markers, enforced run
provenance (epoch 2), and — as of `62a8e52` — content-hashed model artifacts. Import
does not yet have an equivalent singleton contract: completion is inferred from
structure/profile-specific artifacts, registration is still flag-gated, and several
metadata fields degrade to best-effort or `unknown`. This is the 2026-07-01 review's
throughline ("disciplined core, silent-wrong-data at the boundaries") resurfacing at the
one boundary the remediation never reached.

Three headline answers:
1. **Several import-like paths** capture different things. The production recording path
   has two active singleton profiles: metadata-only analysis import, and sampled training
   pixel import via PyNvVC. The old `capture/import_video.py` Decord pixel importer is
   legacy compatibility and is not a future support target.
2. **The registry IS updated at import — but only if `--registry` is passed** (default-on in
   the cluster chain, absent for hand-runs), with no enforcement that stores are registered
   before stages run, and a manual-only reconcile safety net.
3. **Metadata capture is NOT reliable** — loud on structural failure, silent on metadata
   absence. A recording can import "successfully" with `unknown` codec, no source
   fingerprint, no colorimetry, and experiment identity that is present only when the
   manifest/H5 path supplied it and not validated for completeness. Main pixel/schema paths
   can also expose `raw_video/timestamps` without writing usable values.

Two scope decisions from the correction pass:

- Do **not** invest in the legacy `core/pipeline.py` orchestration path; keep it as
  compatibility/reference only while active workflows move through the `palette` CLI and
  batch/cluster wrappers.
- Keep import as a **singleton** surface, not a timestamped run parent. The target is a
  mandatory singleton import/profile contract on `raw_video`/root attrs, not a new
  `raw_import_runs` family.

---

## 1. The import structures — active singleton profiles plus historical pixel path

### Path A — production metadata-only analysis path (what the `raw` verb + cluster run)
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

### Path B — production sampled training pixel importer
This is the active pixel-materialization path for per-recording `_training.zarr` archives.
It is not the old `capture/import_video.py` path.

- **Batch entry:** `utils/import_recordings_training.py` plans
  `zarr/<recording>_training.zarr`, resolves frame step from `--target-sampled-frames` or
  `--frame-step`, and defaults to `--decode-backend pynvvc-luma`.
- **Pixel engine:** default backend invokes
  `fisheye.utils.import_sampled_training_pynvvc` (`import_recordings_training.py:409-428`),
  which sequentially decodes the source MP4 with `PynvvcLumaRgbReader`, writes
  `raw_video/images_full`, optional `raw_video/images_ds`, and
  `raw_video/original_frame_indices`, and stamps the
  `orange_mono_pynvvc_luma_uint8_v1`/PyNvVC luma pixel contract.
- **Training functionality remains required.** This path is how new sampled full-frame
  detector-training zarrs are materialized. It should be included in the singleton import
  profile contract rather than deprecated.
- **Legacy escape hatch:** the wrapper still exposes
  `--decode-backend legacy-decord --allow-legacy-decode-contract`, which calls
  `fisheye.capture.import_video --training-data`; that branch is the legacy path, not the
  default training workflow.

### Path C — historical Decord pixel importer (`capture/import_video.py`, ~1687 lines)
GPU-decodes frames into a zarr. It is reachable from the old orchestrator
(`core/pipeline.py`, which prints `LEGACY_ORCHESTRATOR_NOTICE`) and some compatibility/
intake paths, but it is not the forward production target. Current sampled training imports use
`utils/import_sampled_training_pynvvc.py`; `capture/import_video.py` is used only through
legacy or explicit legacy-decord paths.
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

### Conflict flag
These paths write the same singleton `raw_video` group with **different** `import_method` /
`import_stage` / `import_mode` values (`metadata_only`, `pynvvc_luma_sampled_training`,
legacy `complete`/`full`). Metadata-only import does not overwrite existing attrs unless
explicitly asked, so re-imports can preserve stale values. Since the legacy Decord path is
not a forward support target, the action is not "make all paths equivalent"; it is to
define and enforce production singleton import profiles, including a metadata-only
analysis profile and a sampled/materialized training profile.

### Additional import-like surfaces to keep in scope
The first audit under-counted import-like writers. A production cleanup should include or
explicitly classify these surfaces:

- `utils/import_recording_analysis.py` — canonical metadata-only analysis bootstrap and
  stimulus-import surface.
- `utils/run_recording_analysis_pipeline.py` and `utils/import_recordings_analysis.py` —
  wrapper/pipeline import paths with optional registry scan.
- `utils/import_recordings_training.py` / `utils/import_sampled_training_pynvvc.py` —
  active sampled training pixel import path, listed above as Path B.
- `utils/intake_video_only_recording.py` — video-only intake that can call the legacy pixel
  importer, patch metadata, and optionally register.
- `utils/create_clipped_analysis_zarr.py` and `utils/create_clipped_training_zarr.py` —
  clipped metadata/pixel import-like creators.
- `utils/recording_manifest_import_status.py` — import-log manifest backfill that can scan
  successful imports into the registry.

### Operational arrival of a new experiment (today)
Citrus transfer session → `submit_citrus_session_import_bsub.sh` (one LSF job/session,
`--registry` default-on) → `run_citrus_session_import.py`: organize → metadata import →
registry scan. **The transfer watcher does not exist yet** —
`docs/interface_and_execution_strategy.md:85-88,124` marks it 🆕 to-build; a human/cron
submits today.

### NFS-safe audit discovery note

When auditing a production recordings root on `/groups`/PRFS/NFS, do **not** run an
unbounded recursive scan such as:

```bash
find /groups/johnson/johnsonlab/jeremy/recordings -path '*/zarr/*.zarr' -type d
```

That form still walks the entire recording tree, including videos, staging artifacts,
caches, logs, and other large payload directories before it can decide whether each path
matches. In the 2026-07-06 follow-up audit it remained inside `find` for multiple minutes
before any Zarr metadata was opened.

Use Palette recording-layout discovery instead: loose `*.zarr` archives directly under
the root and one-level `*/zarr/*.zarr` archives under recording directories. The
import-profile checker now exposes this directly:

```bash
scripts/py -m fisheye.utils.check_import_profile \
  --recordings-root /groups/johnson/johnsonlab/jeremy/recordings \
  --jsonl \
  --compact \
  --summary /tmp/import_profiles_summary.json \
  > /tmp/import_profiles.jsonl
```

This is intentionally non-recursive. If a future storage layout needs deeper discovery,
add an explicit bounded policy for that layout rather than defaulting to recursive
filesystem traversal over the whole recordings tree.

---

## 2. Registry linkage — is the registry updated on import?

**Yes, synchronously — but conditionally.**
- `capture/import_video.py` has **zero** registry references. The historical Decord pixel
  engine never registers itself. The active sampled-training wrapper
  (`utils/import_recordings_training.py`) can register created `_training.zarr` archives,
  but only when `--register` is passed; the cluster submitter defaults that on.
- The batch wrapper's `_sync_registry` (`import_organized_recordings_analysis.py:246`) calls
  `Registry.scan_zarr(zarr_path)` (`registry/db.py:8198`) → `register_from_root`
  (`db.py:6609`) → core dataset/recording/provenance/projection refresh in one transaction,
  including `replace_acquisition_video_streams` (`db.py:6670`). This is a **direct write at
  import**, not deferred to reconcile. `reconcile_dataset_from_root` (`db.py:6747`) is the
  broader superset for profile/data-card extractors.
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
- **Designed intent vs reality:** `docs/archive/registry_design_assessment_2026-06-18.md`
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
  subject/fish, cross/line/species/sex): production organized import stamps many
  manifest-derived fields onto root attrs via `ensure_analysis_archive`, and the registry
  also projects normalized context. Gaps remain: the fields are not mirrored into
  `raw_video`, direct pixel/direct metadata paths are weaker, and content completeness is
  not validated. H5/protocol source files also lack a fingerprint/hash.

### Reliability mechanics
- Preflight (`recording_preflight.py`) can include video probe/timing/GOP/decode and H5
  diagnostics when organize ran them, but import gating only blocks on `status=="fail"` and
  remains bypassable with `--allow-preflight-failures`. It does NOT validate
  experiment-metadata *content*, and it does not check freshness against source mtimes or
  manifest changes.
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

1. **Unreliable `raw_video/timestamps` contract.** Main pixel/schema import paths create
   `raw_video/timestamps` but do not populate it with usable acquisition timestamps.
   Metadata-only imports write no timestamp array. Clipped training creators can write
   timestamp data, so the problem should be scoped to the main import surfaces rather than
   described as literally every path.
   → Populate from the decode/H5/frame metadata where reliable, or delete/rename the array
   and stop implying a universal timestamp contract.
2. **Experiment identity is partial and unvalidated.** Production organized import does
   stamp manifest context onto root attrs when present, but not onto `raw_video`; pixel-only
   and direct metadata paths remain weaker; and missing genotype/protocol/subject context
   is not a hard failure.
   → Define required-vs-optional manifest context for production imports, stamp a canonical
   singleton context/provenance block, and fail or explicitly mark unknown-provenance when
   required context is absent.
3. **Existing stores lack source-video fingerprints.** Future production metadata-only
   analysis imports and sampled PyNvVC training imports now stamp cheap `stat_v1`
   source-video fingerprints, but historical `/groups` stores still need an explicit
   backfill before this can be enforced. The fingerprint is intentionally stat/metadata
   based rather than a full-MP4 content hash.
4. **Existing stores lack H5/protocol/source metadata fingerprints.** Future organized
   analysis imports and sampled-training imports stamp H5 `stat_v1` fingerprints when the
   path is known, but historical stores still need backfill. Protocol/sidecar file
   fingerprint policy remains less complete than the H5/source-video path.
5. **Existing stores lack source-video stream colorimetry.** Future shared metadata-only
   imports and sampled PyNvVC training imports probe ffprobe stream
   color_range/color_space/color_transfer/color_primaries and stamp them as
   `video_color_*` attrs. Existing stores still require ffprobe backfill; legacy Decord
   archives may remain historical/off-contract rather than repaired.
6. **Experiment metadata is not validated.** Nothing fails or warns on missing genotype/
   protocol/subject. → Extend preflight or add a post-import completeness gate.
7. **Silent degradation stored as authoritative:** codec/pix_fmt → `"unknown"`; decode
   contract assumed; system provenance best-effort. → Make these loud or explicitly tag as
   unknown-provenance.
8. **The one completeness validator is dead code** (`validate_zarr_structure`, warns-only,
   never called). → Wire it in as an error gate, or delete it (no `status:active` orphan).
9. **Stale attrs on re-import.** Production root identity uses `setdefault`, and video
   metadata only overwrites with explicit overwrite behavior. A re-import can leave old
   context or video attrs in place.
10. **Import outside the completion/provenance perimeter** (§4). → Define the singleton
    import profile contract and make it the one active completion/provenance rule.

---

## 4. Import sits outside the provenance perimeter

- Import IS cataloged: `registry/stage_catalog.py:38` — `StageSpec(id="raw",
  aliases=("import",), category=CORE_PIPELINE)`, dependency-free root of the DAG.
- **Completion is STRUCTURAL and inconsistent, not marker-based:** old surfaces disagree
  about what "raw/import complete" means (`images_ds`, `images_full`, any `raw_video`,
  sampled `original_frame_indices`, or metadata-only analysis markers). The legacy
  pipeline disagreement is not a forward support target, but active status/registry
  surfaces still need one production rule.
- **No completion epoch / run provenance on `raw_video`.** The epoch regime
  (`shared/zarr_run_completion.py`, `mark_run_complete`, epoch 2) governs timestamped
  run-groups under stage parents (detect_runs, crop_runs, keypoints_runs…). By decision,
  `raw_video` remains a singleton group rather than becoming `raw_import_runs`.
- **Target rule:** bring import inside the perimeter as a singleton, not as a run family.
  Production import should stamp mandatory singleton attrs/provenance/profile status on
  `raw_video`/root, and `palette status`, registry step status, validators, and batch
  submitters should all read the same profile contract.
- Consequence: the fail-closed guarantees built this session stop one stage short of the
  root. Bringing import inside the perimeter is the structural counterpart to the
  content-hash work, but the implementation target is a singleton import profile contract.

---

## Method / provenance of this audit
Three parallel read-only Opus agents (import-flow map, registry-linkage trace, metadata
reliability audit), 2026-07-05, against `sun` at ~`89b29e7`, followed by a focused
correction pass after review. Corrections applied here: production import root attrs do
include manifest-derived experiment context; current sampled training import uses
`import_sampled_training_pynvvc`; `scan_zarr/register_from_root` is not the full reconcile
superset; the timestamp problem is scoped to main import surfaces; legacy pipeline support
is out of scope; and import remains a singleton by design. Related prior diagnostics:
`pixel_contract_audit_2026-06-05.md`, `acquisition_crop_video_integration_2026-06-17.md`,
`contract_drift_audit_2026-05-28.md`, `registry_design_assessment_2026-06-18.md`,
`model_input_shapes_metadata_gaps_2026-06-17.md`.
