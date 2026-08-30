# Sleepyfish Data Handoff Draft — historical snapshot

<!-- contract-meta
status: stale_diagnostic_snapshot
observed: 2026-08-20 through 2026-08-21
code_revision: eac89226
authority: evidence only; not an active data handoff or operational contract
-->

> [!CAUTION]
> This unapproved draft preserves a dated inventory and must not be used as a
> current handoff. Processing state may have changed, and parts of its metadata
> guidance predate Palette's consolidated-metadata lifecycle policy. Reverify
> the stores and rewrite the operational sections before issuing a collaborator
> handoff.

**Date:** 2026-08-20
**Audience:** collaborator consuming sleepyfish recording outputs in place on `/groups` (read-only; no data transfer or migration involved).
**Scope:** the four sleepyfish recordings under `/groups/johnson/johnsonlab/jeremy/recordings` and everything derived from them.

---

## 1. What sleepyfish is (and is not)

Sleepyfish is a **single acquisition session recorded simultaneously on four cameras**, not an experiment paradigm with dedicated analysis code:

```
/groups/johnson/johnsonlab/jeremy/recordings/
  sleepyfish_2026_05_05_17_45_30_cam2010093
  sleepyfish_2026_05_05_17_45_30_cam2010094
  sleepyfish_2026_05_05_17_45_30_cam2010095
  sleepyfish_2026_05_05_17_45_30_cam2010096
```

Naming convention: `<protocol>_<YYYY_MM_DD_HH_MM_SS>_cam<serial>`. All four started 2026-05-05 21:45:30 UTC. Each is a **rolling-clip acquisition** (22 clips per camera, `clip_000000` …) of freely behaving fish at 4512×4512 HEVC, **30 fps** (`source_video_metadata.fps = 30.0`; NVENC `rc=vbr` ~61 Mbps — this predates the later 100 fps / 150 Mbps acquisition profile described in the storage docs). 1,188,000 frames ÷ 30 fps = **11 hours** per camera, 30 minutes per clip — an overnight recording, as the name suggests.

Three facts that shape everything downstream:

1. **Video-only intake — no stimulus layer.** The registry records `experiment_context_status = absent`: "Video-only intake has no H5/protocol source; stimulus-dependent analyses are unavailable." Any stimulus-response, chaser, or protocol-aligned analysis simply does not apply here. Only stimulus-independent analyses (detection, pose, masks, tracking, kinematics, swim bouts) exist.
2. **No registered subject identity.** Sleepyfish sources lack subject rows in the registry. For any train/test splitting, group by acquisition-time cohort, not per-fish identity (see `docs/keypoint_training_source_census_2026-08-06.md`).
3. **It doubles as the repo's canary cohort.** Sleepyfish (especially `cam2010095`) is the canonical full-scale benchmark/canary fixture for clipped-recording work — many run names in its stores are canary/benchmark publications. That does not make the data less real, but it explains why `cam2010095` carries extra experimental runs (and the only `tail_kinematics` publications) alongside the production ones.

## 2. Where the data lives

Everything you need is on `/groups`. Do not use anything under `/nvme1` — that is workstation scratch and its registry copy is **stale**.

Inside each recording directory:

| Path | Contents |
|---|---|
| `cams/` | Master acquisition MP4 (4512×4512 HEVC, lossy) + keyframe/meta sidecars |
| `clips/clip_NNNNNN/` | Rolling-clip acquisition units (22 per camera) |
| `derived/` | Cadence probes, pipeline perf CSVs, cluster artifacts, review proxy videos |
| `zarr/` | The analysis and training Zarr stores — **your main target** |
| `recording_manifest.json`, `recording_clip_index.{csv,json}`, `recording_frame_index.parquet` | Acquisition manifests / frame index |

The `zarr/` directory holds (per camera):

- `<rec>_analysis.zarr` — the per-recording analysis archive: `detect_runs`, `refined_detect_runs`, `keypoints_runs`, `refined_keypoints_runs`, `subject_mask_runs`, `refined_subject_masks_runs`, `tracking_runs`, `arena_assignment_runs`, `analysis/` (kinematics, swim bouts, …), plus `clips/` and `experiment_index/` (clipped-archive machinery).
- `<rec>_training.zarr` and `<rec>_clipped_training.zarr` — sampled training-frame exports; only relevant if you are building training sets.

**Scientific source of truth = the Zarr stores.** The registry is a fast searchable index over them, never the authority (`docs/registry_data_governance_policy.md`).

## 3. Environment setup

```bash
conda env create -n palette-py311 -f environment.yml
$HOME/miniconda3/envs/palette-py311/bin/python -m pip install -e .
```

Then run everything through the repo wrapper `scripts/py` (it resolves the conda env and sets `LD_LIBRARY_PATH`; no `conda activate` needed). **Do not create a `.venv` and do not use uv inside the palette repo** — it is conda-only. Full instructions: `docs/environment_setup.md`.

If you only ever read the data directly (Option B in §7), your own environment just needs `zarr>=3` and `sqlite3`; the full palette env is required only when you use its reader modules or CLI at the extraction boundary (Option A — recommended).

## 4. Discovering what exists: the registry

Canonical registry (read-only for you):

```
/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

Older docs show `--registry /nvme1/palette_registry.sqlite` in examples — those are stale paths; always point at the `/groups` file. WAL is deliberately off (NFS), so open it read-only:

```python
import sqlite3
con = sqlite3.connect(
    "file:/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite?mode=ro",
    uri=True)
```

Useful entry points:

- **`palette` CLI** — `palette status <recording>`, `palette plan <recording>`, `palette artifacts <recording> --json` (read-only artifact inventory; contract in `docs/recording_artifact_inventory_contract.md`). Targets accept a recording id, dataset id, zarr path, or directory containing one zarr.
- **Datasette browser** — read-only web UI over all registry tables/views with CSV/JSON export on every URL: `docs/registry_browser/README.md` (launch with the `/groups` registry path, SSH-forward port 8011).
- **Key tables/views**: `recordings`, `recording_step_status` (+ `recording_step_status_wide` for a dashboard-style view), `dataset_lineage_current`, `provenance`, `detect_quality`, `keypoint_quality`.
- **Schema reference** for all 57 tables / 55 views: `docs/registry_schema_reference.md`.

## 5. Processing status (verified in the zarrs, 2026-08-20)

All four recordings are registered (`protocol_name = 'sleepyfish'`, `recording_type = behavior`, `behavior_mode = free`, `dish_design = palm`) and **all four are fully processed through the analysis layer**. Verified directly against each `_analysis.zarr` (completion attrs read with `use_consolidated=False`):

| Analysis family | cam093 | cam094 | cam095 | cam096 | Selector-eligible? |
|---|---|---|---|---|---|
| track_kinematics (offline) | ✓ 2026-07-24 | ✓ 2026-07-24 | ✓ 2026-07-24 | ✓ 2026-07-24 | **yes** |
| swim_bouts | ✓ exponential v013 | ✓ exponential v002 | ✓ exponential v003 | ✓ exponential v001 | **yes** |
| eye_angles | ✓ complete | ✓ complete | ⚠ stale chain | ✓ complete | attr absent |
| subject_shape | ✓ complete | ✓ complete | ⚠ stale chain | ✓ complete | attr absent |
| bout_kinematics | ⚠ stale source | ⚠ stale source | ⚠ stale source | ⚠ stale source | attr absent |
| tail_kinematics | — | — | ⚠ stale chain (canary) | — | attr absent |

Upstream stages (detect, refined detect/keypoints/subject-masks, tracking, arena assignment) are `ok` on all four cameras at 98–100% coverage, per the registry.

Notes:

- **"Selector-eligible: attr absent"** means the run is `complete` and `latest_complete` points at it, but the child lacks a literal `stage_selector_eligible=true`. The strict resolver will therefore refuse it; read these families through their logical reader modules with an explicit run name (the `latest_complete` target) rather than the strict selector, and treat them as complete-but-candidate publications.
- **⚠ stale source (bout_kinematics, all cameras):** the bout_kinematics runs are complete but were built from the older 2026-07-20 swim_bout runs; the current swim_bouts are the 2026-07-24 `exponential` runs. Bout-level kinematics therefore do not correspond to the current bout segmentation. If your analysis pairs bouts with bout kinematics, either use the matching older swim_bout run explicitly or ask Jeremy to re-run bout_kinematics against the exponential runs.
- **⚠ stale chain (cam2010095 only):** cam095's eye_angles/subject_shape/tail_kinematics publications reference upstream refined-keypoint/subject-shape sources that are no longer resolvable as current (`upstream_refined_keypoints_missing` / `upstream_subject_shape_missing`). Prefer cams 093/094/096 for eye angles and shape; treat cam095's as canary-era output.
- The registry `recording_step_status` table was reconciled against these zarrs on 2026-08-20 and now reflects the above (a stale row can still appear if publications happen again without a reconcile — when registry and store disagree, the store wins).
- Several families carry **multiple complete runs** (canary/benchmark publications alongside production ones, e.g. `*_canary_*`, `*_intern_export_*`, `*_temporal_repair_*`). Never pick a run by name-sorting — follow `latest_complete`.
- `missing` / `na` registry rows and absent families (e.g. tail_kinematics off cam095) are valid states, not corruption (`src/fisheye/docs/zarr_structure.md`).
- Eye-**mask** stages are deprecated repo-wide (subject masks subsume them) — distinct from eye **angles**, which are published and current.

## 6. Trusting a run before you read it

Palette runs are immutable publications with an explicit completion contract (`docs/zarr_run_completion_contract.md`). Before consuming any run:

1. Resolve via the API, not by eyeballing group names:
   ```python
   from fisheye.api import open_recording, resolve_run
   rec = open_recording("/groups/.../sleepyfish_2026_05_05_17_45_30_cam2010095")
   ```
   `resolve_run` / the `Recording` accessor (`detections()`, `keypoints()`, `subject_masks()`, `frame_domains()`, `artifact_inventory()`) enforce the contract for you: `latest` and `latest_complete` must agree, the child must be `complete`, and `stage_selector_eligible` must be literally `true`. A disagreement means *stop*, not "fall back to an older run."
2. **Never trust consolidated Zarr metadata.** There is a known store-wide split-brain: the root `zarr.json` consolidated snapshot is stale on most stores and hides completion/provenance attributes. Open groups with `fisheye.shared.zarr_io.open_zarr_root` / `open_zarr_group_direct` (both pass `use_consolidated=False`). If you open stores with raw `zarr.open_group`, pass `use_consolidated=False` yourself.
3. Archive-wide safety check when in doubt:
   ```bash
   scripts/py -m fisheye.utils.check_zarr_run_completion <archive.zarr> --fail-on-unsafe
   ```
4. **Clipped-archive caveat (applies to sleepyfish):** in these archives the flat top-level `refined_detect_runs/<run>/instances` is a *derived projection*; the finalized per-clip collection under `clips/` + `experiment_index/` is the authority. `refined_row_ids` are clip-local — always pair them with their run path.

## 7. Reading the outputs into your own analysis stack

You are expected to do your analysis with **your own tools and library** — palette is the data source, not your analysis framework. You do not need to adopt its export pipeline, its Marimo apps, or its downstream analysis modules. The recommended pattern is **extract-and-leave**: use a small amount of palette code at the boundary to pull trustworthy arrays out of the zarr, convert them to whatever your library wants (numpy / DataFrame / Parquet), and do everything else in your own world.

**Option A (recommended): use palette's readers only at the extraction boundary.** The per-family reader modules encode the trust checks, validity gating, and layout quirks so you don't have to reimplement them:

| Data | Reader |
|---|---|
| Track kinematics | `fisheye.analysis.track_kinematics_io` (guide: `docs/kinematics_zarr_access_guide.md`) |
| Swim bouts | `fisheye.analysis.swim_bout_io` (the `tracks/*/swim_bouts` mirror is deprecated) |
| Bout kinematics | `fisheye.analysis.bout_kinematics.resolve_bout_kinematics_tables` |
| Subject shape | `fisheye.analysis.subject_shape_io` |
| Detections / keypoints / masks | `Recording` accessor methods (`fisheye.shared.recording`) |
| Arena geometry | `fisheye.shared.arena_geometry.resolve_arena_geometry` |

A one-time extraction script that reads via these and writes plain Parquet/CSV/NPZ into your own workspace is entirely legitimate — the zarr stays authoritative and your extracts are disposable/regenerable by design.

**Option B: read the zarr directly with plain `zarr`/your own code.** Fine for arrays the readers don't cover, but then the trust rules in §6 and the footguns in §8 are **your responsibility**. Minimum checklist for a hand-rolled read of any `*_runs` family:
1. Open with `use_consolidated=False` (always).
2. Take the parent's `latest_complete` attr; require the child's `palette_run_completion_status == "complete"`. Where present, require `stage_selector_eligible == True` (see §5 for the families where the attr is absent).
3. Apply `sample_valid` (and `transition_valid` for derivatives) before any statistic.
4. Check `coordinate_space` / units attrs before interpreting positions; never assume mm.

For reference, palette also has its own Parquet **analytics export** layer (`/groups/johnson/johnsonlab/palette_analytics`, `docs/cross_recording_analytics_export_design.md`) and interactive Marimo viewers (`apps/marimo/`) — you can ignore both, though the existing exports may save you an extraction script if one already covers what you need (check the `analytics_exports` registry table).

## 7a. Measured array inventory (cam2010093, latest_complete runs, 2026-08-21)

Enumerated live from the store so an agent can inventory without guessing. Shapes are cam2010093's; the other cameras match modulo row count. Full per-array semantics: `src/fisheye/docs/zarr_structure.md`.

**Two row universes.** N_frames = **1,188,000** (22 clips × 54,000 frames @ 30 fps = 11 h; the 1 Hz resultant arrays' 39,563 rows ≈ 39,600 s confirms the rate) and N_rows = **1,182,938** (one row per accepted detection; single fish ⇒ ≤1 row/frame; ~5k frames have none). Any `(1188000,)` array is frame-indexed; any `(1182938,)` array is detection-row-indexed; join rows→frames via `frame_indices`.

**Lineage columns (present in every row family — this is *why* half the arrays exist):** each row carries its ancestry so any value is traceable to upstream rows and clips: `instance_key` (uint64, stable identity), `detection_indices` (ordinal — NOT stable), `source_refined_row_ids` / `source_detect_row_index` / `source_crop_row_ids` (upstream joins), `source_clip_indices` + `source_clip_local_frame_indices` (clip provenance), `frame_indices` / `source_frame_indices`. Validity/QA columns (`*_valid`, `*_usable`, `*_success`, `reason_bytes`) exist so consumers can gate rather than trust; fixed-width `*_bytes (N, K) uint8` strings are null-terminated UTF-8 (`row.tobytes().split(b"\0")[0].decode()`).

**Clipped-archive placement:** raw `detect_runs` / `refined_detect_runs` / `subject_mask_runs` live per-clip under `clips/clip_NNNNNN/cameras/<serial>/…`; the top-level families below are full-recording projections/publications. Top-level `crop_runs` holds per-clip geometry proxies only (crop pixels come from the crop video).

| Family → run | Key arrays (shape, dtype) | Why present |
|---|---|---|
| `keypoints_runs` (26 arrays) | `keypoints_roi/_img/_norm (N,5,2) f64`; `keypoint_confidences (N,5)`; `heading (N,) f64` + `heading_usable/_finite (N,) bool`; `n_keypoints`, `frame_counts (1188000,)` | 5-point pose in three coordinate systems (ROI-local px, full-image px, normalized); heading from pose geometry; per-frame counts on the frame axis |
| `refined_keypoints_runs` (~50 arrays) | everything above plus `edit_applied`, `flip_corrected`, `heading_temporal_outlier (N,) bool`; `geometry_valid`, `edge_distances (N,6)` + `edge_pairs (6,2)`; `derived_metric_values (N,4)`; `quality_labels (N,) i8`; `reason`/`reason_bytes`; `failure_indices (3472,)` | the curated layer: refinement provenance (what was edited/flip-corrected), triangle-geometry QA, per-row failure reasons; `failure_indices` lists the rows that failed refinement |
| `refined_subject_masks_runs` (~120 arrays) | `masks_roi (N,4,512,512) u8` — dense masks, channel order `[subject_body, eye_left, eye_right, swim_bladder]` per `available_channels (4,) bool`; per-component `components/<c>/{mask_present, area_px, centroid_xy, reason_bytes}` + `metrics/{solidity, hole_count, ipr, …}` + `finalization_metrics/*` | the pixel authority for shape; per-component metrics exist so you can QC-filter without touching pixels; `finalization_metrics` are before/after deltas from automated mask repair; `relations/eye_pair/*` links the eye channels |
| `tracking_runs` / `arena_assignment_runs` (9 + 2) | `track_ids (N,) i32` (single track `id_0`), `arena_ids (N,) i32`, `n_detections_per_arena (1188000,1)` | assigns rows to track/arena; trivial here (1 fish, 1 arena) but the join keys are still required by downstream readers |
| `analysis/track_kinematics_runs` → `offline/…current_coordinates…20260724` (106 arrays, all under `tracks/id_0/`) | positions px+mm; per-level speed/acceleration/path-distance (`raw/filtered/smoothed` × `px/mm`); `heading_degrees/_radians`, `angular_velocity_*`, `delta_*`; `sample_valid`-family gates; 1 Hz resultants `(39563,)` | one array per (quantity × smoothing level × unit) — the multiplicity exists so the speed level is an explicit choice (§8); read via `track_kinematics_io`, which resolves level paths for you |
| `analysis/swim_bout_runs` → `…exponential_20260724` (132 arrays: `tables/`, `signals/`, `indexes/`) | `tables/bouts/*` (176,738 rows: `bout_id`, `candidate_id`, start/end frames+times, durations, `gap_censored`); `tables/histograms/*`; `signals/` | **tabular** layout: column arrays keyed `(candidate_id, bout_id)`. Multiple detector parameterizations coexist — that's why bout count ≫ real bouts; `swim_bout_io` selects the accepted candidate. Never aggregate across all candidates |
| `analysis/eye_angle_runs` (frame + row axes) | `frame_angles (1188000,141) f32`, `roi_angles (N,141) f32`, `roi_vectors (N,2,2)`, `frame_qa/roi_qa (·,7) u16`; `angle_channel_index/{name,units,formula,eye,…} (141,·) u8` | 141 channels = every angle representation + compatibility alias published side-by-side; decode `angle_channel_index/name` to select a channel (mind §8's vergence naming footgun); QA channels gate validity |
| `analysis/subject_shape_runs` (~45 arrays) | `components/subject_body/{centerline_xy (N,64,2), snout_tip_xy, head_endpoint_xy, principal_axis_xy, centroid_xy}` f32 + `*_valid` + `*_reason_bytes` | 64-point midline and body landmarks derived from masks, **in ROI pixels**; per-quantity validity because midline extraction fails on bad masks |
| `analysis/bout_kinematics_runs` (162 arrays: `level_index/`, `movement_metrics/`, `heading_metrics/`, `eye_gaze_metrics/`) | tables keyed `(bout_id, analysis_level_id)`; `level_index` defines the 4 analysis levels; `eye_gaze_metrics` has 0 rows on cam093 | per-bout summaries of the kinematics at each smoothing level — but see §5: **stale vs the exponential swim_bouts**; join `bout_id` only against the matching older swim_bout run |

Regenerate this inventory anytime with ~15 lines of `zarr` (open `use_consolidated=False`, follow each family's `latest_complete`, walk `.arrays()` recursively) — worth doing per camera before a big extraction, since run contents differ slightly across cameras (e.g. cam095's extra canary families).

## 8. Footguns — read this section twice

- **Speed:** never use raw centroid speed. It has a ~1.6 mm/s noise floor; immobility/threshold metrics on it are artifacts. Speed level is an explicit choice (`raw`/`filtered`/`smoothed`/`averaged`); `filtered` is the documented default for behavioral summaries. Use the grouped v2 layout `tracks/id_*/movement/speed/<level>/…`; flat `speed_smoothed_mm` is a sealed compatibility alias.
- **Validity masks:** gate per-frame quantities on `sample_valid` and derivatives (speed, acceleration, path distance) additionally on `transition_valid`.
- **Arena geometry:** always go through `fisheye.shared.arena_geometry`. The nominal `experimental_area` circle is ~3 mm off from the fitted dish mask and has silently inverted thigmotaxis results before. `out_of_bounds_fraction` should be ~0 — if not, the geometry is wrong, not the fish.
- **Two pixel spaces:** offline detections/keypoints live in camera space (4512×4512, ~5.8 px/mm); anything online/texture lives at 358×358 (~0.44 px/mm). Check the `coordinate_space` attribute; never scale texture coordinates into camera space and then apply camera calibration. `positions_mm` is only valid under the typed `physical_coordinate_authority` — a root `pixel_to_mm` scalar does not authorize it.
- **ROI vs image coordinates:** keypoints come in three parallel arrays (`keypoints_roi` / `keypoints_img` / `keypoints_norm`); subject-shape geometry and dense masks are ROI-local. Treating ROI coordinates as image coordinates is a classic error.
- **Frame domains:** row index ≠ acquisition frame. Use `Recording.frame_domains().convert(...)` (`fisheye.shared.frame_domains`); conversions are explicit-only and raise when no recorded mapping exists. In sampled training zarrs, stored row `i` maps through `original_frame_indices[i]`.
- **Row identity:** `detection_indices` is an ordinal into a resolved rowset, not stable identity; `refined_row_ids` are stable logical ids but never biological identity (sleepyfish has no subject identity at all).
- **Timestamps:** `system_timestamp_ns` is POSIX wall clock, not time-since-recording-start; check each surface's declared `clock_domain`/`origin` attrs rather than trusting a `_ns` suffix.

## 9. Ground rules

- **Read-only.** Do not write into the recording directories, the Zarr stores, or the registry. Your extracts, tables, figures, and notebooks live in **your own workspace** — never inside `/groups/.../recordings/<rec>/` and never in the palette repo (deliberate policy; the repo is shared infrastructure).
- **Everything on `/groups`, nothing from `/nvme1`.**
- If a result of yours ever needs to flow *back* into the palette stores (a new derived dataset, corrected annotations), that goes through the repo's publication machinery — talk to Jeremy first rather than writing directly.
- When citing numbers, record the run name you read (`latest_complete` target) alongside the recording id — runs are immutable, so that pair makes your extraction reproducible.

## 10. Reading list (in order)

1. `README.md` + `docs/environment_setup.md` — setup and repo rules.
2. `src/fisheye/docs/zarr_structure.md` — the authoritative Zarr layout spec (v3).
3. `docs/kinematics_zarr_access_guide.md` — the consumer-facing how-to-read guide.
4. `docs/zarr_run_completion_contract.md` — what makes a run safe to read.
5. `docs/registry_schema_reference.md` + `docs/registry_browser/README.md` — discovery.
6. `docs/analytics_math_primer.md` — semantics and listed pitfalls for derived metrics.
7. `docs/registry_data_governance_policy.md` — what is and isn't authoritative.

---

*Compiled 2026-08-20 from a three-agent survey of the repo plus live queries against the canonical registry and the `/groups` recording store; revised 2026-08-21 (verified against main @ eac89226, reframed §7/§9 for consumption with an external analysis library). Status snapshot in §5 reflects the registry at compile time and will drift; re-query rather than trusting the table.*
