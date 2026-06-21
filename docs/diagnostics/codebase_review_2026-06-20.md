# Palette — Whole-Repository Code Review (2026-06-20)

## Method

A thorough, multi-agent re-review of the entire repo — strengths, weaknesses, problems. A prior
six-agent review exists (`docs/diagnostics/codebase_review_2026-06-10.md`, 10 days old). Rather than
repeat it, six read-only auditing agents (architecture, Python quality, testing/CI,
data/storage/provenance, repo hygiene, pipeline/CLI) **re-validated each prior claim against current
HEAD and extended it**. Repo scale today: 733 src files / ~428K LOC, 440 test files / ~130K LOC,
branch `sun`. All review work was read-only; nothing was modified to produce this report.

---

## Overall verdict

**Mixed — strong domain engineering inside an unconverged, ungated, unshareable shell.**
The science-facing machinery (provenance fingerprints, RLE mask invariants, row-lineage
refusal-to-guess, known-answer numerical tests) is genuinely better than typical research
code. But the structural debt the last review named is **mostly unfixed and several metrics
regressed**, there is still **no CI and no end-to-end test**, and the repo still **cannot be
legally or practically shared** (no license, stub README, zero-dependency packaging).

| Dimension | 06-10 | 06-20 | Movement |
|---|---|---|---|
| Architecture & module org | mixed | mixed | utils grew 142k→151k; god-files grew; override table retired (good) |
| Python quality & idiom | solid | solid (regressing) | god-funcs 609→641, private imports 41→56, swallows ↑ |
| Testing & verification | mixed | mixed | prior RED genuinely fixed; **but `sun` is red (20 fails), still no CI/e2e** |
| Provenance & contracts | solid | solid | drift now test-guarded (good); "documented≠enforced" unchanged |
| Performance & IO | solid | solid | dual store now clearly subordinate (good) |
| Repo hygiene & shareability | mixed | mixed/worse | nothing remediated; sprawl grew |

---

## What is genuinely good (keep / build on)

- **`registry/stage_catalog.py`** — clean declarative DAG (`depends_on`/`invalidates`/aliases),
  now spanning import→analysis. The right shape; just not yet the runtime source of truth.
- **RLE mask integrity** — `validate_component_rle_mask_store_invariants` (`shared/mask_store.py:827-990`)
  verifies indptr monotonicity, per-row count sums == H*W, present==(area>0), bbox sanity; full
  round-trip validator streams dense-vs-decoded equality. Real, not token.
- **Row-lineage refuses position guessing** — `shared/row_lineage.py` validates lengths/frame
  alignment and raises on primary/fallback disagreement; the new `source_crop_row_ids` contract
  closes the "row i == crop row i" foot-gun.
- **Deterministic, self-labeling provenance fingerprints** — `shared/run_lineage_fingerprint.py`
  (strict JSON, NFC dedup, `fingerprint_status` marks weak fingerprints weak).
- **Behavioral test discipline** — ~2,543 state-touching asserts vs only 18 mock-call asserts;
  317/435 test files build real zarr stores. Stage drift is now test-guarded
  (`tests/unit/fisheye/test_stage_catalog_drift.py`).
- **Reproducible environment** — `environment.yml` with per-pin rationale + two lockfiles.
- **marimo explorer apps** — the cleanest new code: documented viewer-only boundary, spec-routing
  renderer registry, src-side data loading. The layering the rest of the repo lacks.

---

## Problems, by severity

### P0 — immediate (this branch / cheap + high-impact)

1. **`sun` branch is red: 20 failing tests, never run as a suite.**
   Concentrated in in-progress subject-mask lineage work:
   `test_resolve_subject_mask_stale.py`, `test_row_lineage.py`, `test_subject_mask_inspector.py`,
   `test_sync_refined_subject_mask_metadata.py`. Shared root cause: the lineage-hardening commit
   made `source_crop_row_ids` mandatory (`tune/refined_subject_mask_review.py:1524` raises) but the
   new test helpers build runs without it. Committed core suite (3,069 tests) is green; this is
   unrun feature work. **Decide deliberately: seed crop-row lineage in the fixtures, or relax the
   contract for them — don't paper over.**
2. **No CI of any kind** (no `.github/workflows`, pre-commit, tox). This is *why* #1 can sit on a
   branch unnoticed, and was the prior review's #1 unaddressed recommendation. A minimal GHA job
   (`pip install -e .` → import smoke → `pytest -m "not gpu and not slow"`) would have blocked it.
3. **Shareability metadata (one afternoon, unblocks the stated goal of sharing).**
   Add `LICENSE` + `CITATION.cff`; rewrite the 4-line `README.md` with what-it-is + conda install +
   one runnable command + pointer to `docs/operator_guide/`. Until a license exists it is legally
   all-rights-reserved.

### P1 — correctness / integrity

4. **`core/pipeline.py:419` bare `except:` + `pass`** around the completed-stage probe (packaged
   default orchestrator). A corrupt/permission-denied/half-written zarr is silently classed
   "stage absent" → re-runs or treats partial as fresh; also swallows `KeyboardInterrupt`.
   Carried over from prior review, still untouched. Narrow to `except (OSError, KeyError, zarr...)`
   + logged warning.
5. **RLE stale-marker is safe-by-convention, not by construction.** `open_mask_store` correctly
   rejects stale RLE on read, but `mark_mask_rle_stale_attrs` has **one caller** while **~13 modules
   write `masks_roi[...]` dense**. One un-marked dense edit on a `dense_and_rle` run → silently stale
   RLE for downstream readers. Route all dense mutation through one stale-marking helper, or add a
   finalize-time invariant. (Related to a finding in the earlier `sun`-diff review:
   whole-store stale marker can never be cleared by per-component refresh —
   see `docs/diagnostics/code_review_goodcopbadcop_dashboard_2026-06-20.md`.)
6. **Contracts documented/audited, not enforced** (unchanged thesis). Stage-array validation
   *raises* for exactly one stage (`shared/stage_complete.py:64` allowlist = `{detect_quality}`);
   pixel-contract parity is opt-in (`--required-pixel-contract-name`, default off); the
   `provenance` table is 55 columns / zero NOT NULL; source-video fingerprint is `stat_v1`
   (path+size+mtime, never bytes). Each is one allowlist/flag/constraint away from being a precondition.

### P2 — architecture / maintainability

7. **`utils/` is a 151k-line application layer wearing a utility name** (268 files, 249 `__main__`,
   imported by 192, still imports `cli`). It anchors 5 of 6 subpackage import cycles. **Grew since
   06-10.** Fix: create `fisheye/apps/`, leave a tiny dependency-free `utils`, add import-linter to CI.
8. **Stage graph defined three times, edges disagree** — `stage_catalog.py` vs `core/pipeline.py:163-195`
   vs `cli/interactive_launcher.py`. Concrete hazards: `refined_subject_masks` has `depends_on=[]`
   in the live orchestrator but `subject_masks` in the catalog; there is **no first-class
   `subject_masks` stage** in the in-process orchestrator (raw subject masks are produced as an
   eye_masks side-effect there). Derive all three from the catalog. (Override table already retired —
   finish the job.)
9. **No single end-to-end orchestrator.** `core/pipeline.py` stops before analysis (no
   chaser/bouts/kinematics/stimulus_response); analysis is orchestrated only by shell scripts
   chaining `python -m` calls. Two import/detect/refine orchestrators overlap
   (`core/pipeline.py` vs `utils/run_recording_analysis_pipeline.py`).
10. **Registry god-modules** — `maintenance.py` 9.7k lines, `db.py` 8.1k, `migration_bodies.py` 7.3k
    (~82% of the registry subpackage). Plus a real load-time `shared↔registry` cycle
    (`shared/subject_mask_registry_status.py:10`). Split into subpackages; move registry-status
    helpers out of `shared/`.
11. **Eye→subject-mask migration: a wiring problem, not a capability gap.**
    (CORRECTED — the initial pipeline-agent finding overstated this; see note below.)
    A native unified subject-mask producer **already exists**: `segmentation/infer_unet_subject_masks.py`
    ("Run a trained U-Net segmenter to produce unified subject-mask probabilities") emits all components
    in one model — schemas `subject_v1_union = (subject_body, eyes_union, swim_bladder)` and
    `subject_v1_lr = (subject_body, eye_left, eye_right, swim_bladder)`, `run_semantics =
    "unet_subject_mask_inference"`, with keypoint-based `eyes_union`→L/R splitting. So eyes do **not**
    fundamentally require the eye-mask model.
    There are three coexisting production strategies: (a) the unified U-Net above; (b) modular
    single-component producers — SAM body-only (`run_sam_subject_masks.py`, channels `(True,False,False)`),
    traditional body-only (`segmentation/subject_segmentation.py`), swim-bladder segmenter
    (`(False,False,True)`) — stitched by `utils/merge_subject_mask_runs.py` (target `(True,True,True,False)`);
    (c) the legacy eye→subject projection (`run_eye_masks_batch.py::_project_subject_masks_after_eye_run`).
    Eye-mask code is still *referenced* in only two places: the **in-process `core/pipeline.py`
    orchestrator** (which wires strategy (c)), and an *optional* `refined_eye_masks_runs` eye-source
    branch in `refinement/assemble_refined_subject_masks.py:659,710` — the same assembler equally
    supports taking eyes from a subject run's own `eyes_union` channel (`:907-922` +
    `assign_eyes_union_to_lr`).
    **Milestone (revised):** rewire `core/pipeline.py` to invoke `infer_unet_subject_masks` instead of
    eye-projection, and retire the legacy `refined_eye_masks_runs` eye-source branch in the assembler.
    This is a wiring/config change, not a modeling gap. (Still open: confirm which path the
    batch/cluster runs actually use, vs the in-process orchestrator.) The reverse-compat bridge
    `utils/materialize_refined_eye_masks_compat.py` remains a separate legacy consumer to retire on
    its own schedule.

    *Review-process note:* the original "eye_masks is the live producer, can't delete it" conclusion
    came from an agent tracing only `core/pipeline.py` (which does use projection) and never finding
    `infer_unet_subject_masks.py`. A reminder that single-path tracing by a sub-agent can yield a
    confidently-wrong architectural conclusion; corroborate producer claims across all orchestration
    paths.

### P3 — hygiene / quality (low-risk, high-tidiness)

12. **Packaging:** migrate `setup.py` (zero `install_requires`) → `pyproject.toml` with real deps
    mirrored from `environment.yml`, extras `[gpu]`/`[dev]`. Package is currently not self-describing.
13. **Deletion sweep:** 5 `test_*`/scratch `.py` *inside* `src/fisheye/`, empty `src/fisheye/io/`,
    44 loose `src/*.py` scratch scripts, 13 root one-offs, `scraps/`, top-level `diagnostics/`,
    `trtexec_output/*.jpg` (1.2MB), `apps/manim/data/cache/*.npz`, 10 ALL-CAPS root status docs
    (e.g. `CRITICAL_REIMPORT_NEEDED.md`). Add `docs/README.md` index (255 flat docs, no index).
14. **Code-quality trends to arrest:** 641 functions >100 lines (113 >300; worst `main()` = 1896
    lines); ~452 unlogged `except Exception` swallows (esp. utils 75 / registry 22 / analysis 20);
    byte-identical `_json_dict`/`_json_list` copied across three `export_*_training_zarr.py`;
    22 genuine private-symbol cross-module imports (e.g. `resolve_detect_model._load_*` imported by
    6 batch runners — a de-facto public API hiding as private); 317 `print()` in library tiers.

---

## Recommended remediation roadmap (subtraction + forcing functions first)

Ordered for a solo maintainer; each early item is a forcing function that prevents regression
of the later ones.

1. **Green the branch + wire a CI gate (P0-1, P0-2).** Fix the 20 `sun` failures, then add the
   minimal GHA job. After this, red can't land silently — the single highest-leverage move.
2. **Shareability metadata (P0-3).** LICENSE + CITATION + real README. One afternoon; unblocks sharing.
3. **Fix the two live correctness gaps (P1-4, P1-5).** Narrow the `pipeline.py:419` except; make the
   RLE stale-marker structural. Small, high blast radius.
4. **pyproject + deletion sweep (P3-12, P3-13).** Mostly `git rm` + metadata; makes the repo *look*
   and *install* shareable and de-pollutes the import path.
5. **Stage-graph single-source + import-linter (P2-7, P2-8).** Derive pipeline/launcher edges from
   the catalog; add import-linter to the CI job to freeze the utils-layer/cycle debt from growing.
6. **Then the larger structural work (P2-9/10/11, P1-6):** orchestrator consolidation, registry
   god-file split, eye-mask *rewire* (point `core/pipeline.py` at `infer_unet_subject_masks`, retire
   the assembler's legacy eye-source branch — capability already exists), contract-enforcement
   allowlist growth — these are real projects, not afternoons; do them once the gates above stop the
   bleeding.

## Verification (for any fix above)

- Suite health: `scripts/py -m pytest -m "not gpu and not slow" -q` (expect ~3,089 collected;
  target 0 failed). Targeted: the 4 red `sun` files above.
- Import hygiene (after import-linter added): `lint-imports` / `import-linter` in CI.
- Packaging: `pip install -e .` in a clean env then `python -c "import fisheye"` and one
  documented `python -m fisheye...` command.
- Contract enforcement: extend `_ENFORCE_STAGE_ARRAY_VALIDATION_FOR` and assert a writer *refuses*
  an invalid stage array in a unit test.

---

*Companion report: `docs/diagnostics/code_review_goodcopbadcop_dashboard_2026-06-20.md`
(focused review of the `sun`-branch GoodCopBadCop dashboard + component-scoped RLE refresh diff).
Prior whole-repo review: `docs/diagnostics/codebase_review_2026-06-10.md`.*
