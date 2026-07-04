<!-- ARCHIVED 2026-07-04: superseded by docs/diagnostics/codebase_review_2026-07-01.md (current review). -->

# Palette codebase review — 2026-06-10

Six parallel read-only review agents, each covering one dimension (architecture, Python
quality, performance/IO, provenance/contracts, testing, repo hygiene). ~405k lines /
703 Python files sampled representatively; every claim below was grounded in code the
reviewer actually read. Grades: strong / solid / mixed / weak.

| Dimension | Grade |
|---|---|
| Architecture & module organization | mixed — strong domain modeling inside an effectively fully-connected dependency graph |
| Python code quality & idiom | solid — excellent typing/hygiene, undermined by god-modules, copy-paste, swallowed exceptions |
| Performance & IO | solid — sophisticated, correct GPU fast paths; cost is two parallel storage systems |
| Provenance & data contracts | solid — real integrity scaffolding; contracts documented/audited more than enforced |
| Testing & verification | mixed — genuinely behavioral unit coverage; suite currently red, no e2e, no CI |
| Repo hygiene & shareability | mixed — strong env reproducibility; no license, stub README, heavy sprawl |

## Immediate defects found during review

- **Real SQL bug in shipping code**: `sqlite3.OperationalError: no such column: tip.recording_id`
  raised from `src/fisheye/registry/db.py:5803`, surfaced by `tests/unit/fisheye/test_registry_query.py`
  (2 failures). The test is doing its job; it isn't being run.
- **Test collection errors** poisoning the suite: `test_video_import.py:20` (ImportError on
  `_process_video_cpu`) and `test_list_incomplete_refined_detect_groups.py:36` (KeyError `'interpolated'`).
- **`except: pass` in the orchestration core**: `src/fisheye/core/pipeline.py:419` wraps the
  entire completed-stage probe — corruption/permission/API errors are treated as "stage absent."

## Cross-cutting themes

1. **Two systems coexisting, convergence started but never finished.** The intended
   architecture (stage_catalog as single source of truth, zarr as canonical store, contracts
   enforced) and the actual one (three stage tables + overrides list, a parallel flat-binary
   store, contracts backfilled by audit tools) run side by side. Drift tests and audit scripts
   are transition scaffolding being lived in as steady state.
2. **Documented ≠ enforced.** Pixel contracts, stage-array validation, provenance minimums,
   and the stage DAG are all *described* (docs, audits, warnings in details_json) but mostly
   not *preconditions* (writer refusals, CI gates, import-linter contracts).
3. **Boundaries have stopped carrying information.** utils/ (142k lines, 35% of the package)
   is the application layer; 23 subpackage import cycles held together by 44 deferred imports;
   41 cross-module imports of `_private` names in src and 355/412 test files importing
   underscore internals; loose scripts inside the import path.
4. **Verification exists but isn't wired to anything.** 2,842 tests, many genuinely
   behavioral — and the suite is red, uncollected-clean, with no CI, and the pipeline-level
   tests mock every real stage.
5. **Shareability is blocked at the cheapest possible layer.** No LICENSE, no CITATION,
   4-sentence README, setup.py with zero install_requires, no pyproject.toml — despite
   sharing being the stated goal.

---

## 1. Architecture & module organization — mixed

### Strengths
- **registry/stage_catalog.py is a textbook declarative stage vocabulary** — canonical IDs,
  aliases, depends_on/invalidates, derived maps; tests enforce reference
  (`stage_catalog.py:19-47,301-333`).
- **shared/ is a real contracts library** — 49 cohesive domain modules, imported 411×,
  mostly correct dependency direction.
- **Uniform operational-script idiom** — shared argparse fragments (`cli/shared_args.py`),
  JSON event logs, dry-run/apply gating, machine-readable reason codes.
- **Deliberate optional-dependency isolation** — registry-only commands work without zarr;
  decode backends degrade gracefully (`registry/db.py:116-136`).

### Weaknesses
- **HIGH — utils/ is the application layer wearing a utility name.** 253 files / 142k lines,
  234 with `__main__` blocks, 149 of 230 documented entry points; imports from cli (17×),
  analysis, tune, diagnostics — everything — while 211 imports depend on it.
  *Fix:* move operational scripts to `fisheye/apps/`; keep a tiny dependency-free utils;
  enforce layering with import-linter.
- **HIGH — 23 two-node subpackage import cycles**, including shared↔tracking,
  shared↔registry, shared↔refinement; survived via 44 function-level deferred imports and
  shims (`shared/registry_stage_complete.py:9`, `shared/crop_image_source.py:375,402,933`).
  *Fix:* declare a layer order (shared < registry < stages < apps), relocate the offending
  functions, add import-linter to CI.
- **HIGH — stage graph defined three times** (`core/pipeline.py:163-195`,
  `registry/stage_catalog.py`, `cli/interactive_launcher.py:62-110`) plus
  KNOWN_PIPELINE_DEPENDENCY_OVERRIDES documenting six disagreeing stages.
  *Fix:* derive pipeline + launcher tables from the catalog; retire the overrides table.
- **MED — two competing orchestrators**: in-process `core/pipeline.py` (1,842 lines, the
  packaged default) vs subprocess-based `utils/run_recording_analysis_pipeline.py` (what the
  operator guide actually uses). *Fix:* one orchestrator, in-process stage functions with a
  uniform signature; subprocess only where GPU isolation demands it.
- **MED — no installable CLI surface**: 230 documented `python -m` incantations, two
  console_scripts, no pyproject.toml, setup.py declares no dependencies.
  *Fix:* one `palette` console script with subcommands.
- **MED — god modules**: registry/db.py 7,799 lines (107-method Registry class),
  maintenance.py 9,703, tracking/crop.py 3,917, detect_yolo() spanning ~531-1745.
- **MED — dead weight in the package**: 44 loose scripts in src/ (36 don't import fisheye),
  test scripts inside the package (`fisheye/test_standalone.py` with sys.path hack), empty
  io/ package, stale `__all__` in shared/__init__.py.
- **LOW — no public/private boundary**: 41 cross-module `_private` imports; production code
  living in diagnostics/ (`prepare_detect_training`); inference/ as a duplicate import path.

## 2. Python code quality & idiom — solid

### Strengths
- **95.8% of ~9,200 functions fully annotated incl. returns; zero mutable default args.**
- **print() banished from library tiers** (0–2 in core/shared/detection/tracking/pose);
  the 4,879 raw prints are in CLI/diagnostic tiers where stdout is the product.
- **flat_roi_cache.py is exemplary defensive IO** — keyword-only args, manifest validation,
  byte-size cross-checks, actionable errors.
- **SQL identifier guarding** (`_require_sql_identifier`) on all dynamic registry SQL.
- **Training tier actively extracts shared helpers** (export_shared, training_run_shared).

### Weaknesses
- **HIGH — god-functions**: 609 functions >100 lines, 108 >300; `_check_registry_integrity`
  1,198 lines; `eye_angle_analysis.run` 1,415; two utils main()s at 1,896/1,584.
  *Fix:* split db.py into a subpackage; extract named phases; enforce a size ceiling in CI.
- **HIGH — `except: pass` at core/pipeline.py:419** (see Immediate defects).
- **MED — verbatim copy-paste across export_*_training_zarr modules**: `_json_dict`,
  `_json_list`, `_write_string_array` byte-identical in three ~2,500-line files (the
  eye-mask copy is a deletion candidate anyway). *Fix:* hoist to a shared module.
- **MED — ~143 swallowed exceptions** outside core (utils 52, registry 14, shared 12) plus
  14 truly bare `except:`. *Fix:* `except Exception` minimum; log with exc_info on
  best-effort paths.
- **MED — 57 orphaned scripts** at root and src/ (debug_/test_/speed_test one-offs).
- **LOW — shallow typing**: 3,272 `Any`, Mapping[str, Any] manifest-passing, pydantic in
  2 files. *Fix:* model load-bearing manifests/configs as dataclasses/pydantic.
- **LOW — public docstring coverage 31%** (715/2,284 public functions).

## 3. Performance & IO — solid

### Strengths
- **Correct decoder-owned surface lifetime handling** after a documented postmortem —
  clone-before-advance, owned staging buffers (`pynvvc_luma_rgb.py:68-86`,
  `flat_roi_cache.py:791-935`).
- **Flat ROI cache is well-engineered**: memmap reads, contiguous-run-coalesced writes,
  atomic os.replace, double-buffered pinned-memory async writer overlapping copy with disk IO.
- **Transfer-minimizing 20MP preprocessing**: resize planes *before* YUV→RGB so full-res RGB
  is never materialized; fused fp16 channels_last model.
- **Import store layout matches GDS use case**: uncompressed + sharding for GPU-direct
  writes, tuned kvikio settings; fast lz4 codec on transient scratch.
- **Geometry-only crop storage** avoids pixel duplication; CropImageSource unifies three
  source kinds behind one API.

### Weaknesses
- **MED — per-frame GPU preprocessing + per-frame CUDA sync** in the detect path
  (`detect_yolo.py:476-516`): batch-of-1 interpolate and a synchronize per frame. Surface
  safety only requires copy-before-advance, not full per-frame sync.
  *Fix:* copy luma into an owned ring buffer, batch-interpolate once per batch, one sync.
- **MED — two parallel storage systems**: images_full chunked whole-frame/uncompressed is
  pathological for ROI access — which is why the flat .bin + manifest + parquet system
  exists. Pixel-contract parity must now be maintained in three places (already a documented
  bug source). *Fix:* pick the canonical ROI store; either chunk zarr spatially or declare
  zarr cold/archival.
- **LOW — permanent fine-grained timing instrumentation** (~30 perf_counter fields per write
  batch) and cross-module `_private` coupling between the two cache builders.
  *Fix:* gate behind a debug flag; promote shared writer code to a public module.
- **LOW — CPU live-read fallback random-seeks per frame** (`crop_image_source.py:503-520`) —
  forces keyframe re-decode; a silent cliff without GPU. *Fix:* sorted sequential streaming.
- **LOW — onnx_to_tensorrt.py hardcodes a versioned trtexec path**, no engine cache keyed on
  (onnx hash, precision, shapes, GPU arch).

## 4. Provenance & data contracts — solid

### Strengths
- **Real versioned migration framework**: 53 ordered named migrations under BEGIN IMMEDIATE,
  rollback-on-exception, dual version tracking (`migrations.py:17-75`, `db.py:1262-1294`).
- **Fail-closed stage-completion gate** tied to on-disk zarr completion markers
  (`stage_complete.py:295-317`).
- **Two-tier staleness** (runtime cascade + authoritative reconciliation) with an
  append-only step-status history table.
- **Machine-readable layout spec** (zarr_structure.md ↔ shared.zarr.stage_arrays); the
  completion contract is adopted by 21 production writers.

### Weaknesses
- **HIGH — pixel/decode contracts documented and audited, not enforced at write time.**
  audit_zarr_pixel_contracts.py exists because contracts weren't stamped; its own findings
  doc admits "substantial under-labeling" of raw video, crop runs, merged exports; the
  merged exporter doesn't refuse incompatible source contracts.
  *Fix:* make the contract a writer precondition; fail finalization closed.
- **HIGH — `legacy_default=True` lets uninstrumented runs masquerade as complete**
  (`zarr_run_completion.py:102-114`); the promised strict mode is never engaged.
  *Fix:* flip to strict for new stores keyed on a store-level schema epoch.
- **MED — stage-array validation enforced for exactly one stage**
  (`_ENFORCE_STAGE_ARRAY_VALIDATION_FOR = {'detect_quality'}`); elsewhere validation is a
  warning in details_json. *Fix:* track the allowlist to 100% as a finite milestone.
- **MED — doc corpus diverges**: 240+ docs, four staleness audits in five days, ≥4
  overlapping provenance docs, inconsistent contract-meta headers.
  *Fix:* contract-meta header required on contract docs; TTL + archive for dated docs;
  merge overlapping todos.
- **MED — provenance fully nullable + mtime fingerprints**: ~50-column table where nothing
  is required; stat_v1 (path+size+mtime) instead of content hash; recording-context columns
  duplicated and COALESCE-reconciled. *Fix:* minimum required set per artifact_kind;
  content hashes or explicit weak-fingerprint marking; single owner table.
- **MED — the audit tool is also an in-place zarr metadata writer** using inferred values,
  bypassing the zarr API and run-completion markers; inferred values land in the same attrs
  as recorded ones. *Fix:* separate audit from mutation; namespace inferred provenance.

## 5. Testing & verification — mixed

### Strengths
- **167 of ~412 files build real zarr stores and assert on read-back behavior** —
  run-completion/latest-pointer logic tested through actual attr mutations; pixel-parity
  tests assert pixel-exact agreement.
- **Known-answer numerical coverage**: visual-angle identities, homography
  identity/scale cases, vergence sign conventions (`test_coordinate_transform.py:143-203`).
- **The failing tests are catching real regressions** (the db.py:5803 SQL bug).

### Weaknesses
- **HIGH — no true end-to-end path**: every pipeline test monkeypatches all real stages and
  subprocess.run, asserting call ordering and command strings. *Fix:* one e2e test on a tiny
  synthetic video through real stages, mocking only GPU inference.
- **HIGH — suite is red and ungated**: two collection errors, two real failures, no CI
  config anywhere. *Fix:* get green, add `pytest -m 'not gpu'` on push, treat red as a
  build break.
- **MED — 355/412 files import underscore internals** (test_registry_maintenance.py imports
  ~50 `_helpers`) — the structural cause of collection breakage on rename.
  *Fix:* test through public entry points; assert on zarr/registry state.
- **MED — no fixture strategy**: conftest.py has three trivial fixtures; every file
  hand-rolls zarr builders; 77 files carry sys.path.insert hacks (package not installed
  editable). *Fix:* canonical store-factory and tiny-video fixtures; `pip install -e .`.
- **MED — GPU/decode paths essentially untested** (1 gpu-marked file); seeding in ~9 files.
- **LOW — loose test_*.py outside tests/** never collected (testpaths=tests), rot silently.

## 6. Repo hygiene & shareability — mixed

### Strengths
- **Thoughtful reproducible environment**: curated environment.yml with rationale per pin
  (ffmpeg 4.4 Decord ABI, numpy<2.3, zarr 3.x), environment_setup.md, two lockfiles.
- **Clean src-layout + disciplined .gitignore** — no tracked __pycache__/.pyc/binaries
  (except trtexec_output JPGs, below).
- **Tribal knowledge captured**: AGENTS.md (scripts/py wrapper, zarr sandbox caveats,
  Dask/zarr chunk-ownership rule), curated operator_guide/.

### Weaknesses
- **HIGH — no LICENSE, no CITATION.cff.** Legally all-rights-reserved; a hard blocker for
  the stated sharing goal. *Fix:* BSD-3/MIT for code, CC-BY-4.0 for data, CITATION.cff.
- **HIGH — README is a 4-sentence stub** — no install, no quickstart, no pointer into the
  239-file docs corpus.
- **MED — root/src sprawl + committed binary scratch**: 13 root scripts, 11 ALL-CAPS status
  docs, 16 committed JPGs in trtexec_output/, 44 loose src/ scripts, 9 stray test files.
- **MED — no CI of any kind** (no .github/, pre-commit, tox).
- **MED — setup.py declares zero dependencies**; package not self-describing or
  pip-installable standalone. *Fix:* pyproject.toml (PEP 621) with extras [gpu], [dev].
- **MED — docs/ has 239 top-level files, no index**; operator_guide/ invisible.
- **LOW — lockfile provenance unmanaged** (no generation date/platform/commit).

---

## Prioritized action plan

Ordered by leverage-per-effort for a solo maintainer running parallel LM agents.
Forcing functions first — they keep every future agent-written change honest.

1. **Get the suite green and gate it** (days): fix db.py:5803, the two collection errors,
   core/pipeline.py:419; add a minimal GitHub Actions job (`pip install -e .`, import check,
   `pytest -m 'not gpu'`). Everything else compounds from here.
2. **Shareability basics** (one afternoon): LICENSE + CITATION.cff, real README with
   quickstart, pyproject.toml with declared dependencies.
3. **Finish the convergences already planned** (weeks, incremental): stage catalog drives
   pipeline.py and launcher (retire the overrides table entry by entry); flip
   legacy_default→strict for new stores; grow _ENFORCE_STAGE_ARRAY_VALIDATION_FOR to all
   stages; make pixel contracts writer preconditions.
4. **Mechanical boundary enforcement** (days): import-linter layer contract in CI
   (shared < registry < stages < apps); ruff size ceilings; CI check that src/ contains
   only the package.
5. **The sweep** (one day, mostly deletion): archive/delete 57 loose scripts, ALL-CAPS docs,
   trtexec JPGs; move stray tests into tests/; delete empty io/; docs/README.md index;
   eye-mask modules per the deprecation.
6. **Structural refactors, only after 1–4** (ongoing): utils→apps split, db.py
   decomposition, single orchestrator, one canonical ROI store, e2e test on synthetic video.
