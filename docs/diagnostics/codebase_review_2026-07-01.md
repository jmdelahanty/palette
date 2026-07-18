# Palette codebase review — 2026-07-01

**Supersedes** `codebase_review_2026-06-20.md`, `codebase_review_2026-06-10.md`, and
`codebase_engineering_review_2026-05-20.md` as the current review; those remain as
trajectory baselines.

Six parallel review agents (architecture, data pipeline, registry/provenance, apps/UI,
testing, repo hygiene), each grounded in code actually read, with load-bearing claims
independently re-verified by the synthesizer against HEAD (branch `sun`). This is the
fourth in the series — it builds on and measures the trajectory since
`codebase_engineering_review_2026-05-20.md`, `codebase_review_2026-06-10.md`, and
`codebase_review_2026-06-20.md`. ~515k lines / 790 Python files in `src/`, 483 test files.

| Dimension | Grade |
|---|---|
| Domain modeling & contracts (`shared/`, stage_catalog) | strong |
| Architecture & module boundaries | mixed — strong domain library, regressing coupling |
| Data pipeline correctness | mixed — disciplined core, silent-wrong-data at the boundaries |
| Registry & provenance | mixed — strong static design, exposed operational guarantees |
| Testing & verification | mixed — real behavioral core, non-functional test *system* |
| Repo hygiene & shareability | weak — accreting faster than it's cleaned |
| Measurement validity (analytics axis) † | mixed — rigorous downstream statistics, but CV-layer QC measures self-consistency, not correctness (no ground-truth anchor; see #8) |

† *Row added 2026-07-08, after the original 07-01 snapshot, on the analytics axis (does the
pipeline measure the right quantity, and are the numbers trustworthy as science?). Graded from
the five-domain analytics survey in the [methodology addendum](#addendum-2026-07-08-analytics-methodology-assessment--self-consistency-vs-correctness);
"mixed" splits a strong downstream/group-statistics layer against an unanchored CV-layer QC.*

## Verdict

The engineering craft is well above what a solo LM-assisted project usually shows: the
domain modeling, the mask/coordinate invariants, the migration framework, and a real core
of behavioral tests are genuinely good. But the repo is **not converging — it is
accreting.** The three prior reviews each named the same structural debt and a short
forcing-function checklist; 130 commits later almost none of it is done, and the coupling
metrics that matter have *regressed*. Two findings rise to fix-this-week: a live
correctness inversion in the default pipeline, and a concurrency/atomicity gap in the
registry that contradicts the stated cluster workflow.

## The throughline

Every layer shows the same pattern: **investigations produce documents, one-offs produce
scripts, deprecations produce plans — but the cleanup rarely executes.** The repo is
fluent at *describing* its own entropy (four staleness audits in five days, a
history-shrink plan, an eye-mask severance plan, three prior codebase reviews) while the
described work does not land. The measured regression since 2026-06-20:

| Metric | 2026-06-10 | 2026-06-20 | 2026-07-01 |
|---|---|---|---|
| `utils/` size | 142k | 151k | **165k** lines (32% of package) |
| Function-level deferred imports (cycle tell) | 44 | — | **65** |
| Cross-module `_private` imports in src | 41 | 56 | **61** |
| `registry/db.py` | 7,799 | — | **8,186** lines |
| Suite state | red (2 collect errs + 2 fails) | — | **red (8 collect errors)** |

These are the numbers that should be going down.

---

## Highest-severity findings (verified)

### 1. CRITICAL correctness — default pipeline runs the deprecated path, skips its replacement
`core/pipeline.py:310` sets `refine_eye_masks.enabled: True`; `core/pipeline.py:336` sets
`refine_subject_masks.enabled: False`. `STAGE_ORDER` includes `eye_masks` +
`refined_eye_masks`; there is **no raw `subject_masks` producer stage in the DAG at all**,
and `refined_subject_masks` carries empty dependencies (`STAGE_DEPENDENCIES`,
`pipeline.py:192`). So a default run executes the delete-slated eye machinery and skips
subject-mask refinement — the opposite of the stated migration direction. The legacy eye
code also cannot be deleted yet: live subject-mask code imports
`refine_eye_masks._measure_mask` from ~6 sites (`subject_eye_assignment.py:12`,
`finalize_subject_masks.py:1943`, `shared/refined_subject_eye_geometry.py:54`,
`analysis/subject_shape_runs.py:32`, …). "Eye masks are legacy, don't fix" is currently
aspirational — they run on every default recording.
*Fix:* flip the defaults; relocate `_measure_mask` + geometry primitives to
`shared/mask_geometry.py`; then execute the severance plan.

### 2. CRITICAL registry — default SQLite concurrency under a many-writer cluster
`registry/db.py:1200` is a bare `sqlite3.connect(str(self.path))` with **no WAL, no
`busy_timeout`, no `synchronous` pragma anywhere in `registry/`** (grep confirmed zero
hits). Defaults are therefore `journal_mode=delete`, `busy_timeout=0`. The registry is
driven from dozens of concurrent LSF jobs (`submit_*_bsub.sh`), so any concurrent writer
hits immediate `SQLITE_BUSY`, and a migration's `BEGIN IMMEDIATE` (`db.py:1307`) aborts if
anything holds a lock. Compounding it, `register_from_root` (`db.py:6378`) is **not
atomic**: ~15 `replace_*` calls each commit independently (`with self.conn:` per call,
e.g. `db.py:2968`) with no outer transaction — a crash or lock mid-sequence leaves a
dataset with some derived tables refreshed and others stale (silent partial provenance).
*Fix:* set `PRAGMA journal_mode=WAL; busy_timeout=30000; synchronous=NORMAL` in
`Registry.__init__`; wrap the `replace_*` sequence in one outer transaction.

### 3. HIGH provenance — code version and params are not in the chain
Reproducibility is the stated core value, but there is no git SHA / `fisheye` version /
config-hash column on any analysis artifact or run row (grep for
`git_sha|code_version|commit|config_hash` across `registry/` returns nothing relevant).
Source video → output is traceable; **code + params → output is not.** Fingerprints are
`stat_v1` (path+size+mtime), not content hashes. Separately, there is **no
`subject_mask_data_profile` table** (grep empty) while the deprecated eye-mask path has a
full data-profile pipeline — investment is inverted toward the stage being deleted
(`eye_mask` 264 refs vs `subject_mask` 157 in `maintenance.py`). The designed
`reconcile_dataset_from_root` orchestrator is still unimplemented.
*Fix:* stamp git SHA + version + resolved-config hash on each run/derived-artifact row;
move toward content hashes or explicitly mark fingerprints weak; land the reconcile
orchestrator with `subject_mask_data_profile` as its first extractor.

**Live case study — the SAM3 canary (2026-07-01).** The newest strategic artifact shows
this gap concretely, with mixed news:

- *Better than its plan doc claims:* `utils/run_sam_subject_masks.py` **does** record
  `sam_checkpoint_path`, `method`, and `sam_prompt_policy` in run attrs
  (`run_sam_subject_masks.py:1711-1727,1869-1903`) — the unchecked "record attrs" Phase 3
  box in `docs/sam3_subject_mask_canary_plan.md` is stale. It also **enforces** the
  `bbox_norm_coords` semantics contract: `box_prompt_source='detect'` raises on
  noncanonical (crop-frame-normalized) boxes (`run_sam_subject_masks.py:1062-1068`)
  rather than silently projecting them — a rare documented-*and*-enforced contract.
- *The gap that remains:* checkpoint identity is a **filesystem path, not a content
  hash** — the same stat_v1-class weak fingerprinting as the registry. The path
  `/groups/.../models/sam3/sam3.pt` is mutable, and the plan doc records the checkpoint
  as downloaded **2026-07-01** while the approved composed refined run
  (`refined_subject_masks_sam3_body_existing_eye_swim_red_scare_v3_canary_20260628_01`)
  was produced **2026-06-28** — so which checkpoint generated the approved teacher masks
  must be read from that run's attrs and, even then, rests on a path whose contents have
  since been (re)written. A sha256 of the checkpoint in run attrs closes this.
- *Registry blind spot:* these SAM3 `subject_mask_runs` land in a subsystem with no
  `subject_mask_data_profile` path (see above), so the strategic new mask producer is
  invisible to the profile layer the deprecated eye-mask path enjoys.
- Before Phase 5 (exporting SAM-derived masks as training supervision), checkpoint
  content-hash + prompt-policy provenance should be verified present on the approved
  run — teacher labels with unrecoverable provenance are exactly what finding #3 warns
  about.

### 4. HIGH testing — suite is red and ungated, again
Live `pytest --collect-only`: **3,278 tests collected, 8 collection errors.** A full run
is also red: **3,253 passed / 24 failed** (6.5 min), in three clusters — ~19 subject-mask
lineage failures (`ValueError: subject_mask_runs/... cannot create refined_subject...`,
the mandatory `source_crop_row_ids` contract vs fixtures that don't seed it: the same P0-1
the 06-20 review named, now with a *higher* failure count); detect-refactor drift
(`AttributeError` on `run_detect_with_registry_model` / `run_detections_batch` symbols a
refactor renamed); and a stale hand-rolled zarr fake (`'_FakeGroup' object has no
attribute 'array_keys'` in `test_track_kinematics_turning`).

Of the 8 collection errors: four are from the uncommitted labeling refactor —
`labeling/web_policy.py` uses `BROWSER_CLIENT_AUTHORITY` at four sites
(`web_policy.py:100,125,150,183`) without importing it (defined only in `web.py`), a
`NameError` stopping `test_labeling_web_routes/_config/_security/_signed_links` from
collecting. Two (`test_train_unet_*_registry`) are **environmental, not a repo bug** — a
torch `_dynamo` import fails because this host's name contains a hyphen (`nx-loopback`)
that torch splices into a generated dataclass identifier, raising `SyntaxError`; worth
knowing so it isn't chased as a code defect.

There is still **no CI of any kind** — no `.github/`, no pre-commit, no Makefile/tox — which
is exactly why a branch can sit non-importable *and* 24-red unnoticed. Separately, the
orchestration tests (`test_run_recording_analysis_pipeline.py`) mock every stage and
assert on argv-string tokens, so a stage producing *wrong output* passes; 384 test files
import underscore internals and 77 carry `sys.path.insert` hacks (package not
`pip install -e .`-able). Fixed since June: the `db.py:5803` SQL bug and the two prior
collection errors.
*Fix:* minimal GitHub Actions job — `pip install -e .`, collection check,
`pytest -m 'not gpu'`; treat red-at-collect as a build break. The subject-mask lineage
cluster needs the deliberate decision the 06-20 review asked for (seed crop-row lineage in
fixtures vs. relax the contract), not further deferral.

### 5. HIGH architecture — the stage graph is defined three times and diverges
`core/pipeline.py:180` (`STAGE_DEPENDENCIES`), `registry/stage_catalog.py` (`depends_on`),
and `cli/interactive_launcher.py:65` (`STAGE_INFO.requires`) each hand-maintain the DAG.
`pipeline.py` imports only `canonical_stage_id` from the catalog (`pipeline.py:32`), not
the edges — so the catalog is a documented "single source of truth" that nothing reads at
runtime, and they already disagree (`refined_subject_masks: []` in pipeline vs real
prerequisites in the catalog). Two orchestrators also coexist: in-process
`core/pipeline.py` (packaged default) and subprocess-based
`utils/run_recording_analysis_pipeline.py` (what the operator guide uses).
*Fix:* derive `pipeline.py` and the launcher from the catalog; delete the copies.

### 6. HIGH pipeline correctness — silent-wrong-data at the boundaries
- **Crop path assumes frame-sorted rows.** `crop.py:2757-2878` places crops by cumulative
  detection-count offsets over a sequential cursor, correct only if source rows ascend by
  `frame_index`; nothing asserts it. A refined rowset ordered by `refined_row_id` would
  silently store crops against the wrong frames. Also `n_detections_per_frame[unique_frames]`
  (`crop.py:3515`) `IndexError`s if any `frame_index >= num_images` (arena assignment pads
  for this; crop does not).
- **Training loader pairs blank images with real labels.** `zarr_yolo_dataset_loader.py:1568`
  substitutes `np.zeros_like` on an out-of-bounds frame index but *keeps* the real
  bbox/keypoint label, no warning — trains the model on garbage pairs.
- **Three grayscale conventions for the same pixels.** Import luma `0.2989/0.5870/0.1140`
  (`import_video.py:319`); GPU crop unweighted `(R+G+B)/3` (`crop.py:847`); CPU crop cv2
  luma `0.299/0.587/0.114` (`crop.py:920`). Same detection cropped CPU vs GPU feeds the
  models different intensities.
- **OpenCV fallback drops the final partial batch** (`detect_yolo.py:1450-1491`, no
  post-loop flush) — trailing detections lost or `IndexError`, bounded to the OpenCV path.
- **`mask_probs_roi` has two byte-incompatible encodings** (float16 unit-float vs uint8
  0-255) disambiguated only by a `probabilities_encoding` attr (`finalize_subject_masks.py:445`);
  a missing/wrong attr silently rescales 255×. Make it a hard read-time assertion.

### 7. HIGH apps — `web.py` is a 47,112-line single file
*(Measured at HEAD; the working tree mid-extraction measures ~38.6k — the refactor is
actively shrinking it, so treat the exact count as volatile.)*
One handler class (`LabelingWorkHandler`, `web.py:10001`) with a ~1,277-line `do_GET`
(`web.py:11736`) and ~1,292-line `do_POST` (`web.py:13013`). The in-progress
modularization is *honest* — extracted names are genuinely removed from `web.py`, assets
now read from `templates/`/`static/` files, path handling is traversal-safe, no Python
async, decent same-origin/session security for an internal tool. But it is ~1 of 6 phases
done and moves `_private` names out only to import them back (40+ underscore imports from
`web_runtimes`/`web_policy`), so it is currently a distributed monolith, not a modular
server. The dispatch methods that matter are untouched, and `_datasets_html` (~1,560 lines
inline) and other admin blobs remain.
*Fix:* go depth-first on one route family with a public `(state, request) -> response`
interface and delete its branch from `do_GET`; copy the `group_analytics_viewer/` shape.

### 8. HIGH measurement-validity — the CV-layer QC has no ground-truth anchor
*(Added 2026-07-08 on the analytics axis, not part of the original 07-01 verified set. Full
statement, evidence, and fix in the [analytics methodology addendum](#addendum-2026-07-08-analytics-methodology-assessment--self-consistency-vs-correctness)
below.)* The detection/keypoint/default-mask quality gates measure self-consistency, never
correctness — no OKS/PCK/RMSE, no IoU/Dice against labels — so a systematically-wrong model
passes every gate. Orthogonal to #1–7: a run can close all of them and still be blind to this.

---

## Lower-severity but worth knowing

- **Root directory is a tracked scratch drawer** — ~26 stale scripts/`.md`s, all 8–12
  months untouched, including `CRITICAL_REIMPORT_NEEDED.md` (frozen Nov 2025, reads as an
  active alert). Plus `scraps/`, `trtexec_output/` (16 committed JPGs), a root
  `diagnostics/` colliding by name with `docs/diagnostics/`, and `tmp_*.py` now leaking
  into `scripts/`.
- **`docs/` is a plan graveyard** — 290 top-level `.md` files, 67 with `todo` in the
  filename, no "current" pointer among three stacked codebase reviews.
- **Shareability still blocked at the cheapest layer** — no LICENSE, no `pyproject.toml`,
  `setup.py` with zero `install_requires`, 5-line README stub. `pip install -e .` pulls in
  nothing, which is *why* 77 test files carry `sys.path.insert` hacks.
- **`visualization/t.py`** — 783 lines of commented-out scratch with a hardcoded
  `/nvme1/...` path, committed to the import path. ~7k lines of eye-mask visualizers
  alongside it.
- **`except: pass` at `core/pipeline.py:419`** still swallows corruption/permission errors
  as "stage absent" — flagged in every prior review, still present.
- **Broken TensorRT version regexes** (`export_shared.py:70-78`, literal backslashes in
  raw strings) mean `tensorrt_version` in export provenance is always `None`.
- **`io/` package is still empty**; `shared/__init__.py` declares `__all__ = ["ZARR_SCHEMA"]`
  for a name not defined there (stale export).
- **~2,400 lines / 112 files of `eye_mask` code** remain; the severance was designed in
  May and never executed.
- **SAM3, the newest and most strategic raw-mask producer, lives in `utils/`**
  (`run_sam_subject_masks.py`) rather than `segmentation/` — consistent with existing
  convention, but it feeds the utils-as-application-layer problem and belongs on the
  eventual `utils`→`apps` migration list.

---

## Prioritized action plan

Ordered by leverage-per-effort for a solo maintainer running parallel LM agents.

1. **Fix the two live hazards this week.** Flip pipeline defaults (disable
   `refine_eye_masks`, wire `refine_subject_masks`) and relocate `_measure_mask` into
   `shared/mask_geometry.py`. Add the three registry PRAGMAs and wrap `register_from_root`
   in one outer transaction. Both are small, targeted, and match reality.
2. **Install the missing forcing function.** Minimal GitHub Actions job: `pip install -e .`
   (forces a `pyproject.toml` with real deps, deletes the 77 `sys.path` hacks), a
   collection check, `pytest -m 'not gpu'`; treat red-at-collect as a build break. Add an
   import-linter layer contract (`shared < registry < stages < apps`) and a ruff size
   ceiling in the same job. Without this, every fix below is re-eroded by the next feature.
3. **Make the stage catalog load-bearing.** Derive `pipeline.py` and the launcher from it;
   delete the hand-maintained copies. Kills a live divergence class.
4. **Put code version + config hash into provenance,** or stop calling the project
   reproducible.
5. **The subtraction pass (one low-risk day):** `git rm` the stale root scripts/docs,
   `t.py`, the trtexec JPGs; execute the eye-mask severance (~2,400 lines) once #1 unblocks
   it; archive spent plan docs. Establish the rule that a plan doc is deleted when its diff
   lands.
6. **Structural refactors, only after 1–4:** `utils`→`apps` split behind the import-linter
   gate; `db.py`/`maintenance.py` decomposition; single orchestrator; one true e2e test on
   a tiny synthetic video through real stages.

---

## Addendum (2026-07-01): the ROI pixel store — design rationale and the actual contract gap

Discussion with the maintainer corrected the framing of the "two parallel storage systems"
critique carried over from the 06-10 review. Recording it here so the next reviewer doesn't
re-litigate a decision that is actually sound.

**Design rationale (maintainer).** Decoding the 20MP videos tops out around ~100 FPS, so the
analysis pipeline decodes + crops **once** into a large transient flat `.bin` ROI cache, then
runs N models against that cache instead of re-decoding per model. This is correct
materialized-view engineering, and the persisted timing profiles prove it won: on a warm
cache, `roi_read` is ~1% of runtime while `model_predict` is ~85% — the dataloader problem
is solved. The cache implementation (`shared/flat_roi_cache.py`) is also the
best-engineered I/O code in the repo (manifest validation, byte-size cross-checks, atomic
replaces).

**Historical context for `images_full`.** The original pipeline imported *every* frame to
zarr because inference on the very large raw frames was problematic at the time; the
full-frame store is a residue of that era. Currently `images_full` is primarily intended
for `_training.zarr` files (the sampled training surface), not as an analysis-time ROI
read path.

**Resolution of the "canonical ROI store" question.** The flat cache is the blessed
transient ROI read surface for analysis; `images_full` is the training-sample surface and
archival decode source. This should be stated explicitly in the zarr layout docs
(`zarr_structure.md`) so the ROI-access pretense on full-frame chunked arrays is dropped
rather than maintained.

**The real gap: transient caches are only safe if re-materialization is bit-identical —
and it currently isn't.** The cache is deleted after use, so the pixels the models
actually saw cease to exist. Rebuilding from video must reproduce them byte-for-byte, and
finding #6 documents two reasons it wouldn't:

- three grayscale conventions (import ITU luma vs GPU-crop `(R+G+B)/3` vs CPU-crop cv2
  luma) — rebuilding through a different path shifts every pixel intensity;
- round-vs-truncate crop-center quantization — up to a 1px spatial shift for identical
  detections.

Neither matters while a cache lives; both matter the moment it is rebuilt, because
"re-run model B on the same inputs" silently becomes "re-run model B on slightly
different inputs," and nothing records which decode path built the original cache.

**Recommended enforcement (ordered by leverage):**

1. **One blessed decode+crop function** behind the existing `CropImageSource` seam — a
   single code path (GPU-primary, bit-matching CPU fallback) used by the cache builder,
   the live-read fallback, and training export.
2. **A golden pixel-parity test**: tiny video, decode via every path, assert byte
   equality — including cache-rebuild-equals-original. Converts the invariant from
   convention to enforcement.
3. **Stamp the decode contract** (decoder backend, grayscale coefficients, rounding mode,
   crop-video vs raw-video source) into both the cache manifest and downstream run attrs —
   the "all parameters captured" goal applied to the one input currently invisible to
   provenance: the pixels themselves.

---

*Bottom line: a strong domain core inside a weak repo, and the repo is winning. The fix is
not more engineering — it is a CI gate plus a deletion habit, so the good work stops being
diluted.*

---

## Remediation delta (2026-07-02, ~24h after review)

The review's fix-this-week tier and action-plan items 1–2 were executed by coordinated
agents within a day of the review landing. First trajectory row where the numbers go
down:

| Metric | 06-10 | 07-01 (review) | 07-02 |
|---|---|---|---|
| Suite state | red | red (8 collect errs, 24 fails) | **green + CI-gated** (3,191 passed) |
| CI | none | none | **GitHub Actions: collect gate + not-gpu suite, required-check pending** |
| `utils/` lines | 142k | 165,379 | **148,550** |
| `visualization/` lines | — | 24,109 | **19,839** |
| `eye_mask` file refs in src | — | 112 | **54** (live subject-eye components + deprecated markers) |
| pip-installable | no | no | **yes** (pyproject + extras; console scripts) |

Closed since the review (commits on `sun`):
- **Finding #1 (pipeline inversion)**: eye default off in code+YAML with test locks
  (f72d8aa); full eye-mask severance executed — census (0 NEEDS-CONVERSION, 6ab7843),
  helper relocations, compat-mirror stopped, producers/visualizers/write-paths deleted,
  −48,912 lines across 150 files (cfc0d02…56d4a8f). Catalog models deprecation as a
  first-class field. `core/pipeline.py` marked legacy (live path documented in the
  orchestration notes).
- **Finding #2 (registry concurrency)**: busy_timeout everywhere, WAL deliberately not
  enabled (multi-host NFS), `register_from_root` atomic (2919603). Single-writer funnel
  still queued.
- **Finding #4 (testing)**: suite green (2f05aa3), CI gating pushes (972e8fa, a51b551).
- **Finding #3 (provenance), partially**: the `palette` narrow waist
  (docs/palette_cli_narrow_waist_design.md; a5f1621, 6bf5912, b5d733a) stamps
  git SHA + config hash + params into runs created through it (`cli_provenance`).
  Writer-native stamping, content hashes, and `subject_mask_data_profile` remain open.

Still open: finding #5 (stage graph ×3 — but the gap list now exists:
docs/stage_catalog_reality_gaps.md), #6 (crop/loader/grayscale silent-wrong-data — fix
slice in flight on `agent/silent-wrong-data`; the systemic design behind these bugs is
analyzed in docs/identity_lineage_staleness_review.md), #7
(web.py monolith; modularization regressions were fixed in 2f05aa3), LICENSE/README,
docs/root sprawl, `db.py` (8,234 — still growing), single-writer funnel, orchestrator
convergence decision.

## Remediation delta (2026-07-04, landed on `sun`)

Note on sourcing: `origin/sun` is unreachable from this environment (no push-key
access), and the cached `remotes/origin/sun` ref is stale (pinned at `e4b421f`, ~15
commits behind). This delta is derived from local `sun` (HEAD `d31743e`), verified by
`git log --oneline` + targeted `grep`, not transcribed from prior status docs.

**Finding #3 (provenance), API-semantics arc — landed:**
- **A — CLI import/packaging.** `palette` import is independent of the textual
  launcher (c2e49ca) and an installed-entrypoint smoke test runs in CI (5cc344d).
- **B — honest failure envelopes.** `crop` reports failed/empty outcomes as failed
  envelopes rather than false-success (f3eecb4); `plan` hints are now replayable and
  shell-safe (1c1d86c).
- **C — fail-closed review approval.** Keypoint (177ad54) and subject-mask (f9d1726)
  review approval are fail-closed; detect review is wired to the authoritative
  `approve` path (7a5846b, following the two-pointer reconciliation write-up in
  95b7ae2).
- **D — verbs decoupled from argparse.** Verified directly in
  `src/fisheye/cli/palette.py`: `status`, `plan`, `approve`, `detect`, `keypoints`, and
  `crop` are all `def verb(request: <Verb>Request) -> dict[str, Any]`, built from typed
  request dataclasses (`ApproveRequest`, `StatusRequest`, `PlanRequest`,
  `DetectRequest`, `CropRequest`, `KeypointsRequest`) and returning the
  `build_envelope(...)` payload — no `argparse.Namespace` in the verb signatures
  (f4cbb44, ebfa966, 61a53ba). The CLI parser now constructs a request and calls the
  same function the library would call.

**Stage-array validation enforcement — landed, verified count.** Current
`_ENFORCE_STAGE_ARRAY_VALIDATION_FOR` in `src/fisheye/registry/stage_complete.py`
contains **7 stages**: `arena_assignment`, `crop`, `detect`, `detect_quality`,
`keypoints`, `refined_keypoints`, `tracking` (up from the 1-stage `detect_quality`
allowlist at the 07-01/07-02 checkpoints; catalog reconciliation in e846e17). Still
short of "every specced, non-deprecated stage" — see
docs/archive/provenance_enforcement_roadmap.md.

**Catalog accuracy — landed.** `detect`'s `invalidates` no longer includes
`background` (verified in `stage_catalog.py`; reconciled together with the `--force`
override in e846e17). `eye_mask_tuning` now carries `deprecated=True` (ae236bf,
building on the eye-mask severance already noted above). `palette` verbs support an
explicit `--force` catalog-dependency override, surfaced in the envelope as
`forced_dependency_overrides`/`forced: true` (verified in `palette.py`, wired via
e846e17).

**Identity Rec 1 (authoritative-run pointers + `palette approve`) — landed.**
`AUTHORITATIVE_RUN_ATTR` exists in `src/fisheye/shared/zarr_run_completion.py`
(1d4aa07); default-run resolution reads it authoritative-first (708bd03); `approve`
is a callable verb in `palette.py` (6894031, e6a4edb) with review tooling wired to it
(f61956f, 7a5846b, f9d1726, 177ad54). Documented in
docs/identity_lineage_staleness_review.md Rec 1.

**IN-FLIGHT, not on `sun` — do not report as landed:**
- The `Recording` accessor (`f78e092`) and the `RunResolution` resolver (`4293f7f`)
  live on branch `agent/run-resolution-accessor` only; `git merge-base --is-ancestor`
  against `sun` confirms neither commit is an ancestor of `sun` HEAD. This is the
  narrow-waist doc's step 3 (`Recording` accessor) and picks up the detect
  two-pointer reconciliation once merged.

**Correction to the drafting brief for this delta:** the utils Phase 1 drift-fix
slice (docs/utils_reorganization_strategy.md) is **not** in-flight — it is
substantially landed on `sun` as of the last 6 commits at HEAD (1c8f5ce, fc756fa,
cfff5f9, 7ff3448, 42311c8, d31743e), all timestamped within the hour before this
delta was written. Verified by grep: local `_iter_zarr` definitions are down to 3
(all in deprecated eye-mask diagnostics files slated for removal, not live call
sites); local `_write_json` definitions are down to 2; the canonical
`write_json_atomic` home exists at `src/fisheye/shared/json_safety.py:129`.
(The `_utc_now` leg, noted partial while this delta was being drafted, completed in
the final commit `42311c8` — re-verified: 0 local `_iter_zarr` / `_utc_now` / live
`_write_json` definitions remain. Net: **Phase 1 is fully landed**, not merely
planned; the remaining utils reorg — helper relocations, dead-code deletion, runner
moves, the import-linter gate — is the separate later work per the strategy doc.)

## Proportion finding (2026-07-05) — the meta-observation this review omitted

Findings #1–7 catalog defects; none states the forest. Measured 2026-07-05:
**~488k LOC source / 802 files**, 158k LOC tests, **131k lines across 406 docs**.
The shape, not any single number, is the finding:

- **58% of source files (463/802) carry `__main__`.** This is an application layer
  wearing a library's name — nearly every schema change deposited a permanent script.
  `utils/` is 144k LOC / 272 files (30% of source; down from 165k as Phase 1 landed),
  `diagnostics/` another 108 files / 33k of un-deleted one-off investigations.
- **Docs outweigh the science-core code.** 43 design + 43 todo + 42 diagnostics + 18
  review docs; 86 formal `contract-meta` files whose `status:` vocabulary has drifted
  to ~20 ad-hoc one-offs (governance grew slower than the corpus).
- **The science/CV core** (detection + tracking + segmentation + refinement + pose +
  training ≈ 41k LOC, or ~96k counting analysis) **is a minority of the codebase**;
  the majority is infrastructure. This is the advisor's science-vs-engineering
  tension made measurable — a proportion problem, not a competence one.

**Direction of travel is correct, and worth recording as such.** Every slice landed
the week of 2026-07-04/05 was *subtraction*: eye-mask severance (−48,912 lines), utils
Phase 1–2 (edges severed, scripts→shims), the registry-reconcile collapse (sync scripts
deleted with equality proofs), the legacy detect-pointer retirement (writers down to two
annotated migrations), authority unified through one `approve` verb. The overengineering
is accreted history; this week bent the curve the other way. The highest-leverage next
move is not more code but the two forcing functions from `utils_reorganization_strategy.md`
(import-linter `forbidden_modules = fisheye.utils` + the `__main__`-only-in-`apps/`
predicate) — they make the sprawl deletable and stop re-accretion. See
[[user-context-solo-lm]] for the advisory framing.

## Addendum (2026-07-08): analytics methodology assessment — self-consistency vs. correctness

Prior reviews graded the *engineering* of the analytics (contracts, provenance, coupling,
suite state). This addendum grades the *measurement science* — do the analytics measure the
right things, and are the numbers they emit trustworthy as science, not just as reproducible
byte-streams? Five parallel survey agents, one per analytic domain (keypoints/pose,
crop/detection/ROI, subject-mask/segmentation, downstream/behavioral, provenance/QC), each
grounded in code read in full for the core modules. This is an orthogonal axis to findings
#1–7: a run can be perfectly provenanced, CI-green, and contract-clean while measuring the
wrong quantity.

### Finding #8 (analytics axis, added 2026-07-08) — HIGH measurement-validity: the CV-layer QC has no ground-truth anchor

*Promoted from the addendum's throughline prose to a discrete finding at the maintainer's
request, so it tracks alongside #1–7. Distinct axis from those: #1–7 grade whether the
pipeline is engineered correctly; #8 grades whether it measures the right quantity. A run can
close every one of #1–7 and still be blind to this.*

**Nearly every quality signal in the CV layer asks "does this output agree with my own
geometric/statistical assumptions?" and almost none asks "does this match reality?"** There
is no ground-truth anchor anywhere in detection, keypoints, or the default mask path — no
OKS/PCK/RMSE for pose, no IoU/Dice for the traditional mask path, no labeled hold-out any
produced surface is scored against. **Failure mode: a model that is confidently, systematically
wrong passes every gate** — coverage is high, geometry is "valid," confidence clears
threshold, review is approved — because every check is internal. The evidence, by producer:

- **Keypoints.** `registry/extractors/keypoint_performance.py` records success rate, mean
  confidence, and timing — no `map50`/OKS field. Ultralytics *already computes* pose
  mAP/OKS in `.val()` during training and it is discarded rather than persisted. The
  traditional path's "confidence" is `min(1.0, blob_area/100)`
  (`detection/detect_keypoints_traditional.py::calculate_confidence`) — blob size, not a
  probability — and it feeds the `confidence_valid` gate.
- **Masks.** The default background-subtraction path
  (`tune/subject_mask_tuner.py`) has zero accuracy metric; its "metrics" (`area_px`,
  `prob_max`, component counts) are self-referential descriptors. The U-Net path is the one
  producer with a real held-out metric (validation Dice, `_compute_masked_dice`).
- **Detection.** `refinement/detect_quality.py` emits a 0–100 letter grade from coverage +
  temporal-artifact + bbox-validity sub-scores — all internal consistency, none compared to
  a labeled box.

This is a coherent, defensible engineering choice — self-consistency checks need no labels,
so they scale to the whole fleet for free — and it catches *breakage* (a stage crashed,
geometry malformed, coverage dropped) well. But it is structurally blind to *systematic
error* (a biased detector, anatomically-off masks, a silently inverted eye-flip convention).
The pipeline is instrumented for the first failure class and not the second. This is the
measurement-science analogue of finding #6's "silent-wrong-data at the boundaries": #6 is
about pixels changing under you; this is about *never having measured whether the pixels were
right in the first place.*

Two structural circularities make it worse, because they mean the QC cannot catch error it
helped create:

- **Keypoints.** The traditional detector's geometric heuristics
  (`detection/detect_keypoints_traditional.py::identify_keypoints_by_geometry` — bladder =
  smallest interior angle, L/R eyes heading-relative) generate the labels that train YOLO
  (`red_to_yolo`/pose export). The `geometry_valid` gate in
  `refinement/refine_keypoints.py` then validates YOLO output against *those same
  assumptions*. Nothing in the loop is independent of the heuristic.
- **Masks.** The U-Net (`segmentation/train_unet_subject_masks.py`) is the one place with a
  real held-out metric (per-channel validation Dice, `_compute_masked_dice`) — but its
  training labels are the traditional background-subtraction seeder + operator edits. High
  Dice means "the net reproduces the classical pipeline," not "the masks are anatomically
  correct." The metric is real; its referent is not ground truth.

*Fix (detailed as leverage moves 1–3 below; ordered by correctness-per-effort, none needing
large-scale annotation): (a) persist the accuracy metrics already computed and thrown away —
Ultralytics pose mAP/OKS and U-Net validation Dice — into the registry next to the
self-consistency rates (zero new labeling); (b) wire the existing `utils/detection_profile.py`
distributions into a per-rig baseline as a label-free anomaly floor (zero new labeling); (c)
break one circularity with a small independent hand-labeled frame set scored by a metric not
derived from the traditional heuristic (a few dozen frames, one-time). Until at least one
lands, every accept/reject decision rests on self-consistency plus human review — no
independent signal that the pipeline is right, only that it did not visibly break.*

### Second throughline: a precision chassis around uncalibrated constants

The engineering discipline is exceptional and the measurement rigor sitting inside it is
thin — the gap is wide and consistent across every domain. Provenance, lineage fingerprints,
coordinate-space tracking, pixel contracts, completion gates: exceptional. The numbers those
frames carry: uncalibrated magic constants with no derivation study on record —

- Detect quality score weights `0.5*coverage + 0.3*artifact + 0.2*bbox` (and sampled
  `0.6/0.3/0.1`), grade cutoffs 90/80/70/60, jump=100px@640, blip-gap=10, size-outlier=3σ
  (`refinement/detect_quality.py`).
- Keypoint temporal-outlier = 120°/3-frame, min_angle=10°, min_area=100px², confidence=0.3,
  and the traditional-path "confidence" = `min(1.0, blob_area/100)` — blob *size*, not a
  probability, feeding the `confidence_valid` gate (`detection/detect_keypoints_traditional.py::calculate_confidence`,
  `shared/keypoint_temporal_heading.py`).
- Mask QC min_solidity=0.15/0.20, max_thin_spur_score=8.0, min_largest_component_fraction=0.90–0.95
  (`refinement/subject_body_mask_qc.py`, `finalize_subject_masks.py`).

None has a calibration study or a measured false-positive rate against posture edge cases.

### Per-domain, in one line each

- **Keypoints** — strongest engineering, weakest science: no accuracy metric at all;
  Ultralytics already computes pose mAP/OKS in `.val()` and it is discarded rather than
  persisted (`registry/extractors/keypoint_performance.py` records success rate + mean
  confidence + timing, no `map50`/`fitness`). The stub `drift: {enabled: false}` in
  `utils/auto_keypoint_review.py` is scaffolding for the regression detector that was never
  computed.
- **Masks** — U-Net path has a real objective; the *default* (background-subtraction) path
  has zero accuracy metric, stores `diff/255` as `mask_probs_roi` with the same `prob_max`
  machinery as the U-Net's true sigmoid outputs (a downstream reader cannot distinguish a
  calibrated confidence from a brightness delta), and computes Otsu only to discard it in
  favor of a hand-tuned fixed threshold (`tune/subject_mask_tuner.py`).
- **Crop/detection** — the best latent asset in the repo: `utils/detection_profile.py`
  builds genuinely distributional profiles (percentiles, histograms, spatial heatmaps,
  content-hash fingerprint) — the correct shape for drift detection — and **nothing gates on
  them.** Size outliers are computed at 3σ (`detect_quality.py::validate_bboxes`) and then
  not penalized in the score. The drift alarm's sensor already exists; only the alarm is
  missing. (This is the same `subject_mask_data_profile` asymmetry noted in finding #3, seen
  from the analytics side: profiles built, never used to decide anything.)
- **Downstream/behavioral** — **overturns the "data factory that does no science" worry.**
  `analysis/` is ~50 modules of real behavioral quantification (kinematics, bout detection,
  Megabouts integration *with* a `megabouts_convention_audit.py`, stimulus-response assays, a
  pre/post avoidance-learning paradigm with declared primary endpoints) backed by
  hand-rolled non-parametric group statistics (Wilcoxon signed-rank, sign-flip permutation,
  bootstrap CIs, rank-biserial, BH-FDR in `group_statistics/paired.py`, scipy-free for
  determinism). Statistically the instincts are correct (non-parametric for small-n
  behavior; effect sizes + CIs + multiplicity control). This is the most scientifically
  ambitious part of the repo and it is a genuine strength.
- **Provenance/QC** — mature fingerprinting + fail-closed completion gate, but the gates
  record metrics without thresholding any of them: quality "gating" is entirely
  run-completion existence + human review state, with no automated "coverage < X ⇒ block."
  Watch the `tracks` (catalog) vs `tracking` (`_ENFORCE_STAGE_ARRAY_VALIDATION_FOR`
  allowlist) name mismatch — a gate keyed on a non-canonical stage id can silently no-op.

### On results in the repo — a retracted inference (corrected 2026-07-08)

An earlier draft of this addendum flagged "instrument finished, findings absent," reasoning
from the fact that the survey found extensive analysis code and tests on synthetic fixtures
but **zero committed real results, figures, or notebooks** (zero `.ipynb` in the tree). The
maintainer corrected this the same day: **results and analyses are deliberately kept out of
this repository.** `palette` is shared infrastructure that other people install and run, and
one researcher's analysis outputs, figures, and per-experiment notebooks do not belong in a
reusable library — they live in a separate analysis/paper context. That is correct practice
(it keeps the shared tool clean and its dependency surface honest), and it fully explains the
in-repo absence. The inference was wrong: I read "not in this repo" as "does not exist," and
the former is a deliberate, sound boundary, not deferral.

The residual, stated honestly and without overreach: this means the code-review instrument
**cannot speak to whether an end-to-end scientific result has shipped** — that question is
out of its scope by the maintainer's own (correct) design, not answered either way by
anything in the tree. So this is not a finding; it is a scope boundary. The one durable point
that survives is downstream and independent of where results live: the CV-layer QC proves the
pipeline did not *break*, but nothing in it proves the pipeline is *right* (no ground-truth
anchor — the throughline above). That gap is about measurement validity feeding whatever
analysis happens, wherever it happens, and it stands on its own without the results-in-repo
claim. See also the proportion finding (2026-07-05): the science-core is a minority of the
codebase — a real observation about *this repo's* shape, which is exactly the shape you'd
expect if results are correctly kept elsewhere.

### Highest-leverage moves (analytics axis, ordered)

1. **Wire the profiles you already built into an automated per-rig baseline.** One
   comparison ("this run's w/h/area distribution is Nσ off the last 20 runs of this rig ⇒
   flag") converts `detection_profile.py` from built-and-unused into the automated quality
   floor the pipeline currently lacks — and it needs no ground-truth labels. Biggest
   correctness-per-effort win available.
2. **Persist the accuracy metrics that already get computed and thrown away.** Surface
   Ultralytics pose mAP/OKS and U-Net validation Dice into the registry next to the
   self-consistency rates. Cheapest possible correctness signal; the numbers exist, they are
   just discarded at the boundary.
3. **Break one circularity with a small independent ground-truth set.** A few dozen
   hand-labeled frames scored with a metric *not* derived from the traditional heuristic is
   the only thing that can detect systematic error the geometry gate helped create. Small,
   one-time, and it is the difference between "self-consistent" and "correct."
4. **Stop calling `diff/255` a probability, or mark it uncalibrated at the type level.** A
   read-time assertion or distinct attr so a downstream consumer cannot silently treat a
   brightness delta as a confidence — the mask-domain analogue of finding #6's
   `probabilities_encoding` hazard.
5. **Either calibrate the letter-grade/threshold constants against a downstream outcome
   (pose success, review rejection) or delete them.** An uncalibrated grade that predicts
   nothing measured is worse than no grade — it reads as rigor it does not have.

*Bottom line for this axis: an unusually good engineer of a scientific pipeline whose weakest
link is measurement validity. The CV-layer QC proves the pipeline did not break; it does not
yet prove the pipeline is right — no ground-truth anchor closes that gap. The
downstream/behavioral + group-statistics layer is a genuine strength and is where results are
produced (correctly, outside this repo); the highest-leverage work is on the validity of the
CV measurements feeding it, not on the analysis machinery, which is already strong.*
