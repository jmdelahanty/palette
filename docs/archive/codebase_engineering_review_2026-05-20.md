<!-- ARCHIVED 2026-07-04: superseded by docs/diagnostics/codebase_review_2026-07-01.md (current review). -->

# Palette Codebase Engineering Review (2026-05-20)

Scope: code quality, workflow, and distributed-pipeline best-practice posture.
Excluded: docs inventory (covered by a sibling agent). Evidence references file
paths and line numbers anchored to the working tree at this date.

## Verdict

Palette is a research codebase that has clearly grown by accretion. The
artifact contracts (`shared/zarr/stage_arrays.py`, `registry/stage_catalog.py`,
`shared/refined_detect_curation.py`) and the registry/status model are
genuinely thoughtful — the *vocabulary* of stages, the staleness/freshness
states surfaced through `emit_stage_completion`, and the per-stage provenance
fingerprints (`shared/run_lineage_fingerprint.py`, `shared/crop_signature.py`)
are above average for a solo-author scientific pipeline. There is a real
mental model of what the data should look like.

The *runtime* that sits on top of that vocabulary, however, is not at the same
level. The "orchestrator" (`src/fisheye/core/pipeline.py`, 1843 lines) is a
hard-coded `if/elif` ladder of stage names with three parallel sources of
truth about what a stage is: `STAGE_ORDER`/`STAGE_DEPENDENCIES` here, the
`StageSpec` catalog under `registry/stage_catalog.py`, and the ad-hoc
"latest"-attribute checks in `_is_stage_complete` (line 1369). The
"distributed" story is mostly Dask local-scheduler use plus one LSF `bsub`
submitter for a single workflow (`utils/submit_clipped_detect_refine_plan_bsub.py`).
The single largest file in the repo is `registry/db.py` at **15,865 lines** —
a SQLite layer that has absorbed extractors, schema, and migration policy for
every stage. `registry/maintenance.py` is **9,666 lines**. These are not
"large modules"; they are end states of a codebase that never refactored.

Top three risks, in order:

1. **The registry layer is a single-file monolith with no enforced module
   boundary against the rest of the stack.** `shared/registry_stage_complete.py`
   imports `..registry.db`, while `registry/db.py` imports
   `fisheye.shared.batch_logging`. That is a literal cycle held together by
   import ordering. The blast radius of any registry-schema edit is the entire
   codebase, and 15.8k lines in one file means edits are necessarily done
   under-tested.
2. **Idempotence is by ad-hoc convention ("latest" attribute), not by
   contract.** There are no completion markers, no content-addressed run
   directories, no atomic-rename pattern around stage outputs. Re-runs decide
   freshness by looking up `<group>_runs.attrs['latest']` and trusting it.
   Partial writes from killed workers will leave a half-populated run group
   that still reads as "latest." The staleness policy is encoded only
   downstream (in `step_cascade.invalidate_downstream_steps`), not as a
   physical invariant on disk.
3. **There is no DAG executor.** `core/pipeline.py` runs stages serially in a
   single process, in a fixed list, with hard-coded `_run_<stage>` methods.
   `cli/batch_runner.py` is a `subprocess.run` loop. The only "distributed"
   submitter (`submit_clipped_detect_refine_plan_bsub.py`) is bespoke for one
   pipeline. Multi-recording parallelism today is "fan out shell commands by
   hand or with bsub." Snakemake/Nextflow/Dask-delayed wiring of stage
   dependencies does not exist; the dependency graph in `stage_catalog.py` is
   declarative metadata that *nothing executes*.

## Strengths

- **Stage vocabulary is canonicalized.** `src/fisheye/registry/stage_catalog.py:37`
  defines `STAGE_SPECS` with `depends_on`, `invalidates`, `artifact_families`,
  and category. `canonical_stage_id()` resolves aliases. This is a real
  contract, and it is reused by the launcher and registry.
- **Stage-completion event has a single chokepoint.**
  `src/fisheye/shared/registry_stage_complete.py:60` `emit_stage_completion`
  is called from ~49 sites; it bundles dataset upsert, step status, and
  downstream invalidation. That centralization is the right shape even if the
  module placement is wrong (see Weakness #2).
- **Per-stage signature / fingerprint discipline exists.** Content-addressed
  hashing of inputs is implemented in
  `src/fisheye/shared/crop_signature.py:18`,
  `src/fisheye/shared/crop_geometry.py:135`,
  `src/fisheye/shared/run_lineage_fingerprint.py:297` (SHA-256 over canonical
  JSON lineage). When a stage chooses to use them, this is the right
  mechanism. The problem is that only some stages do.
- **Per-stage provenance attrs are formalized.**
  `shared/stage_provenance.py`, `shared/provenance_attrs.py`, and the
  `*_runs/<name>/attrs` payload set described in
  `shared/zarr/schema.py:71-99` are consistent across detect / refine / crop.
- **Atomic JSON sidecar writes are used where they exist.** e.g.
  `src/fisheye/utils/submit_clipped_detect_refine_plan_bsub.py:56` uses
  `tempfile`-then-`os.replace`. Good pattern; it just hasn't propagated to the
  Zarr run-group writers, which are the actually-load-bearing case.
- **Test suite is broad, ~405 unit tests** under `tests/unit/fisheye/`.
  Coverage skews toward leaf utilities (backfill scripts, IO helpers, review
  backends), which is honest for a research pipeline — many of these are
  exactly the brittle bits.
- **Dependencies are partly pinned.** `environment.yml` constrains
  `numpy>=2,<2.3` and `zarr>=3,<4`. `pip-packages-exact.txt` (190 lines) and
  `conda-packages-explicit.txt` (371 lines) give reconstructable snapshots,
  even though they're not the env definition.
- **The `StageSpec.invalidates` graph encodes downstream cascade semantics**,
  and `registry/step_cascade.py` consumes it. The intent — "marking refined_keypoints
  ok invalidates eye_masks / arena_assignment / track_kinematics / eye_angles"
  — is correct and useful.
- **Decoupled review web UIs** (`tune/*_review_web*`) are kept as standalone
  Flask-style services rather than entangled with the pipeline runtime,
  which is the right separation for a human-in-the-loop step.

## Weaknesses & Risks

### 1. The "pipeline orchestrator" is a hand-coded if/elif chain — claim, evidence, impact, fix

**Claim:** `core/pipeline.py` is not an orchestrator in any modern sense. It
is a 1843-line class with one fixed `STAGE_ORDER` list, hard-coded
`_run_<stage>` methods, and an `_is_stage_complete` function that
reimplements freshness logic against zarr attributes inline.

**Evidence:**
- `src/fisheye/core/pipeline.py:156-188`: `STAGE_ORDER`, `STAGE_DEPENDENCIES`,
  and `ANALYSIS_STAGES`/`DATA_STAGES` are three sets defined here, parallel
  to but not derived from `registry/stage_catalog.py`. They drift.
- `src/fisheye/core/pipeline.py:434-486`: `_run_stage` is an `if stage ==
  'import': self._run_import() elif ...` chain. New stages require editing
  this method, `STAGE_ORDER`, `STAGE_DEPENDENCIES`, and `_is_stage_complete`.
- `src/fisheye/core/pipeline.py:391-432`: `_resolve_dependencies` reaches into
  the target zarr at plan time (`zarr.open_group(...).attrs.get('latest')`)
  to decide which deps to skip, including a hard-coded special case
  ("if `stage in ('refine','detect_quality')` and `dep in
  ['import','background','downsample']` and `'detect' in existing_stages:
  continue`"). This is the kind of thing a DAG would express declaratively.
- `src/fisheye/core/pipeline.py:1369-1400+`: `_is_stage_complete` is a
  per-stage bespoke check ("if `latest is None`, not complete"). It silently
  accepts a half-written run group as complete.

**Impact:** Adding a stage requires editing four hand-maintained tables. The
"orchestrator" can't be reused for distributed execution, can only run a
single recording's stages serially in one process, and can't resume cleanly
when interrupted.

**Recommendation:** Demote `core/pipeline.py` to a single-recording sequential
driver and stop pretending it's the brain. Derive `STAGE_ORDER` from
`stage_catalog.dependency_map()` (topological sort). Replace
`_is_stage_complete` with a single function that asks the **registry**
(`recording_step_status`) — which is already the canonical source of truth in
`shared/registry_stage_complete.py`. If you want a real orchestrator, generate
a Snakemake or Dask-delayed graph from `STAGE_SPECS` and let an external
runner schedule it. The work to migrate is bounded because the per-stage
entry points already exist as functions.

### 2. registry/shared/* import cycle and registry/db.py monolith

**Claim:** The module boundary between `shared/` and `registry/` is broken,
and the SQLite layer has absorbed the entire data-extraction surface for
every stage.

**Evidence:**
- `src/fisheye/registry/db.py:16` imports `fisheye.shared.batch_logging`.
- `src/fisheye/shared/registry_stage_complete.py:9-17` imports
  `..registry.db` and `..registry.status_ledger` and `..registry.step_cascade`.
- Python doesn't crash today because the cycle is at function-call time, not
  at module-load time — but the dependency direction is now ambiguous.
  "Shared" should not depend on "registry," or "registry" should not depend
  on "shared." Currently both hold.
- `wc -l src/fisheye/registry/db.py` → **15,865** lines. `grep -n "^def "` →
  ~hundreds of `_extract_*` helpers (e.g., `_extract_detect_quality_rows`,
  `_extract_keypoint_performance_rows`, `_extract_subject_mask_*`, etc.).
  Stage-specific extraction logic lives inside the registry layer. That means
  every stage's on-disk schema is *also* defined here, in SQL CASE
  statements; see `_recording_step_status_pivot_columns` at line 27 and the
  inline SQL generators throughout the first 60 lines.
- `src/fisheye/registry/maintenance.py:9666` lines. Maintenance has absorbed
  invariant-checking, repair, and "should we mark this stale" logic for
  every stage.

**Impact:** Touching the SQLite schema (or any stage's run-group attribute
shape) means editing a 15.8k-line file. Tests of the registry can't isolate
a single stage without pulling in everyone's extraction code. The cycle
means refactoring the registry's location is now actually hard.

**Recommendation:** Split `registry/db.py` along the natural seam: keep
`Registry`, `RegistryPaths`, schema bootstrap, and the SQL-only API in
`registry/db.py`; move every `_extract_<stage>_rows` to
`registry/extractors/<stage>.py`. Break the cycle by moving
`registry_stage_complete` *into* `registry/` (it already only talks to
registry primitives), leaving `shared/` as the leaf.

### 3. Idempotence and resume semantics are by convention only

**Claim:** "Did stage X complete?" is answered by inspecting
`<stage>_runs.attrs['latest']`. There is no completion marker, no
content-addressed output directory, no atomic rename of the run group, and
killed workers can leave a half-written run group that still reads as
`latest`.

**Evidence:**
- `src/fisheye/core/pipeline.py:1381-1400`: stage completion check is `latest
  is not None`. No invariant on the contents.
- `grep -rn "completion_marker\|tempfile.*rename\|os.replace" --include='*.py'
  src/fisheye/` — `os.replace` appears in `submit_clipped_detect_refine_plan_bsub.py`
  for JSON sidecars only. **No zarr run-group writer uses atomic rename or a
  completion sentinel** that the orchestrator checks.
- The Zarr v3 schema doc (`src/fisheye/shared/zarr/schema.py:31`) and the
  per-stage `*_runs` attribute lists do not include a `completion_state`
  field. Provenance is written, but a partial write leaves provenance attrs
  set and a half-populated array tree.
- Re-runs depend on a "Refinement stages are designed to be repeatable;
  always allow rerun" carve-out at `core/pipeline.py:1375`, which is a
  sentence in a comment, not a contract.

**Impact:** Kill -9 (or OOM, or LSF eviction) during stage write produces
silent corruption that downstream stages will happily consume. This is the
exact failure mode distributed pipelines optimize against.

**Recommendation:**
- Adopt a `*_runs/<name>/_complete` sentinel file or attribute set as a
  postcondition of every writer. Make `emit_stage_completion` refuse to mark
  the registry "ok" unless the sentinel is present.
- Write each run under `<group>/<name>.partial/` and `os.rename`/zarr-group-
  move to `<group>/<name>/` only after sentinel-write. Refuse to consider
  `.partial` as `latest`.
- Promote "latest" to be derived from the registry, not from a zarr attr.

### 4. There is no real distributed-execution layer

**Claim:** Despite Slurm being in the user's environment and Dask being a
dependency, the actual cross-recording parallelism is `subprocess.run` in
`batch_runner.py`. The "Dask" stages are local-scheduler thread/process
pools inside one recording.

**Evidence:**
- `src/fisheye/cli/batch_runner.py:149-180`: `run_single` is
  `subprocess.run(cmd, check=True, capture_output=True)`. Sequential, no
  concurrency between recordings.
- `grep -rn 'SLURMCluster\|submitit'` → no hits. `LSFCluster` → no hits. The
  only cluster integration is `utils/submit_clipped_detect_refine_plan_bsub.py`
  which shells out to `bsub` for one specific workflow.
- The `--scheduler` flag (`'processes' | 'threads' | 'single-threaded' |
  'distributed'`) is exposed at `src/fisheye/cli/batch_runner.py:343` and in
  many stage modules, but "distributed" maps to a Dask LocalCluster, not a
  cluster scheduler. See `src/fisheye/analysis/subject_shape_runs.py:24` —
  `from dask.distributed import Client, LocalCluster`. There is no code path
  in the repo that constructs a Slurm or LSF cluster object.

**Impact:** The "distributed pipeline" claim is aspirational. Cluster
deployment requires the user to write per-recording bsub wrappers (which
they do, in `runs/cluster/`). There is no DAG-aware scheduling, no per-stage
resource specification, no automatic retry, no work-stealing.

**Recommendation:** Pick one — `dask-jobqueue` (LSFCluster/SLURMCluster) or
Snakemake — and commit. The `STAGE_SPECS` graph plus per-stage entry-point
functions already give you ~80% of what Snakemake needs. The current
half-Dask, half-bsub, half-subprocess situation is the worst of all options
because nobody can tell what's actually load-bearing.

### 5. `src/` top-level loose-script sprawl

**Claim:** There are ~46 ad-hoc Python files at `src/` top level (not under
`src/fisheye/`). Mtimes show recent edits — they are not all dead.

**Evidence:**
- `ls src/*.py | wc -l` → 46.
- Mtimes: `src/video_diagnostic_tool.py` and `src/video_integrity_checker.py`
  were last modified **2026-04-15**. `src/zarr_inspector.py` 2026-01-21. So
  these are still in use.
- The names are a code-smell catalog: `debug_*.py` (5), `test_*.py` (4 — but
  not under `tests/`), `plot_*.py` (3), `*_audit.py` (2), `*_analyzer.py`
  (~8), `roi_*` (6).
- Repo root also has `check_frame_gaps.py`, `check_heading.py`,
  `inspect_enum_structure.py`, `verify_enum_format.py`,
  `patch_pose_schema.py`, `speed_test*.py`, `test_frame_sampling*.py`,
  `test_pose_schema_loading.py`, `trtexec_diagnostics.py`,
  `visual_angle_visualizer.py` — yet more loose scripts above `src/`.
- Some are simple matplotlib widget demos with no package imports
  (`visual_angle_visualizer.py`), which suggests they were never even
  considered part of "the package."
- These compete with `src/fisheye/diagnostics/` (which has 60+ similarly
  named files) and `src/fisheye/utils/` (248 files).

**Impact:** New contributors / future-you cannot tell what's load-bearing.
Refactors that should be safe (e.g. moving `detect_dataset_audit.py`) risk
breaking a workflow that's invoked once a month from somebody's shell
history.

**Recommendation:**
1. Move every `src/*.py` and root-level loose script into
   `src/fisheye/scripts/` (a new package) or `scraps/` (already exists). Make
   them entry-points in `setup.py` so their existence is at least declared.
2. The `src/fisheye/diagnostics/` directory has the same problem at smaller
   scale — 80+ `check_*`, `inspect_*`, `benchmark_*`, `preview_*`. Split
   into `diagnostics/checks/`, `diagnostics/benchmarks/`,
   `diagnostics/inspectors/`, or accept that it's a junk drawer and put a
   one-line `README` declaring it so.

### 6. Several modules are far past a reasonable size

**Claim:** Multiple files in `src/fisheye/` are large enough that they are no
longer reviewable as units.

**Evidence (lines of code, this tree):**
- `src/fisheye/registry/db.py` — **15,865**
- `src/fisheye/registry/maintenance.py` — **9,666**
- `src/fisheye/shared/refined_detect_curation.py` — **2,943**
- `src/fisheye/analysis/stimulus_response.py` — **2,799**
- `src/fisheye/analysis/track_kinematics.py` — **2,787**
- `src/fisheye/analysis/subject_shape_runs.py` — **2,644**
- `src/fisheye/training/train_detection.py` — **2,133**
- `src/fisheye/cli/interactive_launcher.py` — **1,921**
- `src/fisheye/analysis/import_stimulus_to_zarr.py` — **1,915**
- `src/fisheye/core/pipeline.py` — **1,843**
- `src/fisheye/training/zarr_yolo_dataset_loader.py` — **1,636**
- `src/fisheye/training/train_pose.py` — **1,608**
- `src/fisheye/analysis/plot_track_kinematics.py` — **1,514**

**Impact:** Edits to any of these are risky because the diff context is
small relative to the file. Tests of these modules can only cover a fraction
of branches.

**Recommendation:** Split first three on the natural seams that already exist
(extractors → per-stage modules; curation → readers/writers/validators;
maintenance → checks/repairs/migrations).

### 7. `analysis/` is the bin where multi-purpose stuff has accumulated

**Claim:** `src/fisheye/analysis/` mixes (a) library code that other modules
import, (b) CLI scripts (`plot_*.py`, `diagnose_*.py`), and (c) `inspect_*`
debugging tools. There is no separation between authoring library and
consumer scripts.

**Evidence:** `ls src/fisheye/analysis/` shows
`bout_classification_runs.py`, `bout_kinematics.py`,
`calibration_manager.py`, plus `plot_*`, `inspect_*`,
`diagnose_*`, `fix_*` scripts side by side. Some are 2000+ LOC stage
writers; some are one-off plotting tools.

**Impact:** Importing from `fisheye.analysis` is ambiguous — you don't know
whether you're getting library code or a runnable.

**Recommendation:** Within `analysis/`, split into `analysis/stages/`
(library, writers), `analysis/plotting/`, `analysis/diagnostics/`. This is
purely organizational and cheap; the win is downstream clarity.

### 8. compact-v2 zarr layout migration is partially done

**Claim:** The "compact v2" layout that several design docs describe is
implemented for some analysis stages and not others. The codebase has both
`_is_compact_v2_group` / `_write_*_compact_v2` paths and legacy paths,
selected at write time by config.

**Evidence:**
- `src/fisheye/analysis/swim_bout_io.py:228,274,394,473`: explicit
  `_is_compact_v2_group`, `_candidates_from_compact_v2_group`,
  `_load_compact_v2_tables`.
- `src/fisheye/analysis/stimulus_response.py:1694,2010`:
  `_write_stimulus_response_compact_v2` with `if/else` against a legacy path.
- `src/fisheye/analysis/detect_bouts_multi_level.py:113,1315,2190`: layout is
  a CLI choice (`SWIM_BOUT_LAYOUT_COMPACT_V2 = "compact_v2"`), default
  documented as compact_v2.
- Design docs exist for each: `docs/swim_bout_runs_v2_compact_layout.md`,
  `docs/bout_kinematics_compact_v2_layout.md`,
  `docs/stimulus_response_compact_v2_design.md`,
  `docs/eye_angle_compact_v2_design.md`,
  `docs/compact_v2_readiness_audit_2026-05-11.md`.
- But there is no archive-level schema version that says "this run group is
  compact_v2 vs legacy." Detection is heuristic
  (`_is_compact_v2_group(run_group)` inspects the group shape).

**Impact:** Readers must branch at every load. The audit doc exists, but the
finishing of the migration is not enforced; nothing prevents a new run group
being written in the legacy layout.

**Recommendation:** Add a `layout_version` attribute to each run group's
contract and require writers to set it. Add a deprecation warning on legacy
load. Set a calendar date to delete legacy load paths.

### 9. Integration test directory exists but is empty

**Claim:** `tests/integration/` is present but contains no `*.py` files.

**Evidence:** `find tests/integration/ -name "*.py"` → no output. The
directory exists; the suite does not.

**Impact:** The 405 unit tests are leaf-level. There is no test that takes a
tiny zarr fixture and walks it through `import → background → detect →
detect_quality → refine → crop`. The most common breakage mode in a
pipeline like this — stage-to-stage contract drift — is exactly what's
untested.

**Recommendation:** Pick the smallest realistic recording, commit it as a
fixture, and add one end-to-end integration test that runs the full
core-pipeline DAG. Even one of these would catch ~80% of contract regressions
between stages.

### 10. Subprocess-style "unit" tests

**Claim:** 18 of 405 "unit" tests under `tests/unit/fisheye/` invoke
`subprocess.run` / `sys.executable`.

**Evidence:** `grep -l "subprocess\|sys.executable" tests/unit/fisheye/ | wc -l`
→ 18.

**Impact:** These tests are slow, environment-dependent, and fragile — they
should be in `tests/integration/`.

**Recommendation:** Move them. It's also a forcing function to actually fill
out `tests/integration/`.

### 11. Three competing stage taxonomies

**Claim:** The codebase has three lists of "what is a pipeline stage" and
they don't fully agree.

**Evidence:**
- `src/fisheye/registry/stage_catalog.py:37` — `STAGE_SPECS` with canonical
  ids (`raw`, `refined_detect`, `refined_keypoints`, `arena_assignment`,
  `tracks`, plus DERIVED_ANALYSIS, TUNING categories).
- `src/fisheye/core/pipeline.py:156` — `STAGE_ORDER` with runtime names
  (`import`, `refine`, `keypoints_refine`, `assign_ids`, `track`).
- `src/fisheye/cli/interactive_launcher.py:65` — `STAGE_INFO` which calls
  `canonical_stage_id` to bridge the two — but redefines `requires` and
  display metadata.

`canonical_stage_id` exists to translate between (1) and (2). That's the
right *workaround*, but the underlying duplication remains: the launcher's
`STAGE_INFO` and the pipeline's `STAGE_DEPENDENCIES` are hand-maintained
parallel structures.

**Impact:** Drift. Renaming a stage requires touching all three.

**Recommendation:** Make `core/pipeline.py` derive its stage order and
dependencies from `stage_catalog.dependency_map()`. Aliases already exist for
the runtime-vs-canonical name gap.

### 12. The "interactive launcher" is 1,921 LOC of Textual UI inside `cli/`

**Claim:** `cli/interactive_launcher.py` mixes UI, business logic, and
process orchestration in one file.

**Evidence:** `wc -l src/fisheye/cli/interactive_launcher.py` → 1921. Imports
Textual widgets, opens zarr files directly (line 18), constructs subprocess
commands, and renders progress.

**Impact:** Hard to maintain, impossible to unit test, ties the launcher to
the in-process pipeline shape. When you eventually move to a real DAG
runner, this will need to be rebuilt anyway.

**Recommendation:** Treat the TUI as a thin client over a pipeline API. The
business logic (what stages can I run? what's complete?) should live in a
single library called from both the TUI and the CLI.

### 13. Heavy module-load side effects in core pipeline

**Claim:** `core/pipeline.py:23-35` imports the implementation of every stage
at module load time, including GPU-only modules (yolo detection,
TensorRT-backed pose, etc.).

**Evidence:** Lines 23-35 of `core/pipeline.py` import `import_video`,
`compute_background`, `detect_fish`, traditional and (transitively) YOLO
keypoints, crop, arena assignment, refinement, eye-mask batch utils, and
zarr schema validators at the top of the file.

**Impact:** Any `python -m fisheye` invocation pays the full import cost and
will fail at startup if any single backend is broken in the environment.

**Recommendation:** Lazy-import per stage inside `_run_<stage>`.

### 14. Magic strings and stringly-typed status values

**Claim:** Stage status values, source-kind labels, and review states are
strings scattered across the codebase, often with embedded `code_map` dicts
written into zarr attrs.

**Evidence:**
- `src/fisheye/shared/refined_detect_curation.py:973-1093`: `row_sort_order`,
  `source_kind_code_map`, `decision_code_map`, `status_code_map` are written
  as attrs but their canonical definitions live in module-level constants.
- `src/fisheye/registry/maintenance.py:50`:
  `RECORDING_STEP_STATUS_VALUES: tuple = ('ok','missing','absent','na','error')`
  — a tuple, not an enum.
- `shared/citrus_enums.py:205` lines suggests enums *are* being introduced
  for some axes, but inconsistently.

**Impact:** Typos in status strings will not be caught by type checkers, and
the same vocabulary lives in 5+ places.

**Recommendation:** Promote `RECORDING_STEP_STATUS_VALUES` and the various
`*_code_map` constants to `enum.StrEnum`. The on-wire / on-disk
representation stays the same.

## Distributed Pipeline Best-Practice Scorecard

| Practice | Status | Evidence | Gap |
|---|---|---|---|
| Explicit DAG, declaratively defined | Partial | `registry/stage_catalog.py:37` defines `depends_on` / `invalidates` per stage | Nothing executes it; runtime uses parallel `STAGE_DEPENDENCIES` table in `core/pipeline.py:173` |
| Idempotent stages | Weak | "rerun is OK if `latest` attr is set" convention; some stages explicitly idempotent (`core/pipeline.py:1375`) | No completion sentinel; partial writes look complete |
| Content-addressed outputs | Partial | `crop_signature.py`, `run_lineage_fingerprint.py` (SHA-256 over canonical JSON) | Hashes are recorded, not used as the run directory name; "latest" is timestamp-tagged |
| Atomic writes / safe partial-failure | Weak | `submit_clipped_detect_refine_plan_bsub.py:56-63` uses tempfile+`os.replace` for JSON sidecars | No zarr writer uses atomic rename; no `.partial` → `final` pattern |
| Registry as single source of truth | Mixed | `emit_stage_completion` upserts to SQLite (`shared/registry_stage_complete.py:60`) | `core/pipeline.py:_is_stage_complete` ignores the registry and reads zarr attrs directly |
| Schema versioning | Partial | `ZARR_SCHEMA_VERSION = "3.0.0"` (`shared/zarr/schema.py:30`); per-stage layout flags for compact_v2 | No per-run-group `layout_version` attr; legacy/compact_v2 detection is heuristic |
| Failure recovery / retry | Absent | — | No retry logic, no checkpoint, no resume from partial. `batch_runner.py` will run stages sequentially and a single failure aborts unless `--continue-on-error` |
| Test coverage of pipeline glue | Weak | 405 unit tests, leaf-heavy; `tests/integration/` empty | No end-to-end DAG test; 18 unit tests cheat by `subprocess.run` |
| Reproducible env | Good | `environment.yml`, `pip-packages-exact.txt`, `conda-packages-explicit.txt`; some version pins | No `uv`/`poetry` lockfile; setup.py is minimal (just two entry points); decord wheel pinned to FFmpeg 4.x via comment, not constraint |
| Observability / structured logging | Mixed | `shared/batch_logging.py` provides JsonLogger; rich console used pervasively for human output | No central log aggregation; per-recording logs are scattered under `runs/cluster/...`; no metrics export |
| Distributed scheduler integration | Absent (1 exception) | One LSF bsub submitter for clipped detect/refine workflow only (`utils/submit_clipped_detect_refine_plan_bsub.py`) | No `SLURMCluster`/`LSFCluster`/submitit anywhere; `--scheduler distributed` means local Dask cluster |
| Stage-output contract validation | Partial | `shared/zarr/stage_arrays.py` declares per-stage array specs and validator | Validators exist (`validate_zarr_structure`) but are not enforced on stage completion |

## Recommended Next Moves (Ranked)

1. **Make stage completion a contract.** Add a `complete: bool` (or
   `_complete` sentinel array) to every `*_runs/<name>` group. Make
   `emit_stage_completion` require its presence before marking registry "ok."
   Make `core/pipeline.py:_is_stage_complete` consult the registry instead of
   re-reading zarr attrs. This is one week of work and closes the silent-
   corruption door (Risk #3, Weakness #3).

2. **Break the `registry/shared` import cycle and split `registry/db.py`.**
   Move `shared/registry_stage_complete.py` into `registry/`. Move every
   `_extract_<stage>_rows` out of `db.py` into `registry/extractors/<stage>.py`.
   Goal: `db.py` under 2000 lines, `maintenance.py` under 2000 lines, no
   imports from `shared/` *into* `registry/db.py`. This is mostly mechanical
   but enables everything that follows (Weakness #2, #6).

3. **Delete the parallel stage taxonomy in `core/pipeline.py`.** Generate
   `STAGE_ORDER` and `STAGE_DEPENDENCIES` from
   `stage_catalog.dependency_map()` via topological sort. Use canonical ids
   internally; translate to runtime names only at user-facing layers
   (Weakness #11, partly #1).

4. **Add one end-to-end integration test.** Pick a tiny recording. Run
   `import → background → detect → detect_quality → refine → crop →
   keypoints` against a fixture zarr. Assert the registry has all stages
   "ok." This will catch ~80% of contract-drift breakage and forces the
   completion-marker work above (Weakness #9, #10).

5. **Decide the distributed-execution story.** Either (a) commit to
   Snakemake driving the `stage_catalog` DAG, with per-stage entry points
   already in hand; or (b) commit to `dask-jobqueue` with one `LSFCluster`
   constructor that the pipeline knows about. Don't keep the current state
   of "five different ways to run things, all half-implemented" (Weakness
   #4).

6. **Move loose scripts out of `src/` top-level.** Either into
   `src/fisheye/scripts/` as declared entry points, or to `scraps/` with a
   commit explicitly retiring them. Same for the repo-root `check_*.py`,
   `test_*.py`, `inspect_*.py` files. This is a half-day's work and the
   readability win is large (Weakness #5).

7. **Finish the compact-v2 layout migration or stop adding new ones.** Add
   `layout_version` to every run group attr and require writers to set it.
   Pick a date to delete legacy load paths in the IO modules
   (`swim_bout_io.py`, `stimulus_response.py`, `bout_kinematics.py`)
   (Weakness #8).

8. **Make the interactive launcher a thin client.** Extract the
   "what-can-I-run-now" logic into a library function `pipeline_plan(zarr,
   stages)` that returns the resolved DAG and the per-stage status from the
   registry. Have both the TUI and the CLI consume it. Cuts
   `interactive_launcher.py` roughly in half (Weakness #12).

9. **Promote string status / code-map vocabularies to `enum.StrEnum`.** Wire
   value to disk untouched; static checkers gain coverage; typos at write
   sites get caught (Weakness #14).

10. **Lazy-import per stage in `core/pipeline.py`.** One-line move per
    function; eliminates the "import the whole world" startup cost
    (Weakness #13).

The above is roughly in dependency order: 1 enables 4; 2 enables 3; 3 enables
5. If only one thing happens, it should be #1 — silent partial-write
corruption is the single most likely way a research pipeline of this kind
will quietly produce wrong figures, and the fix is small relative to the
risk.
