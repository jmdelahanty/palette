# Repository Enforcement & Sprawl Review — 2026-06-04

Scope: a repo-wide critical review prompted by the owner's concern that shared
helpers, metadata, and general functionality are inadequately enforced and
documented, and that the repository is bloated. Conducted as a five-way parallel
read across (1) shared-helper duplication, (2) metadata/provenance enforcement,
(3) repo bloat/hygiene, (4) documentation health, (5) tests/CI enforcement.

This is a point-in-time assessment, not a change. Counts are approximate and
reflect the repo on this date.

## Headline: the problem is an enforcement vacuum, not bloat

The current checkout is lean — roughly 25 MB of text across ~1,167 `.py` files,
and the history rewrite has already been applied to this repo: `.git` is ~22 MB
(the ~473 MB of long-deleted model weights, TensorRT engines, and zips that
bloated the pre-rewrite backup are purged). Residual history bloat is minor — a
couple of `test_frames/*.jpg` (~3.6 MB) and two notebooks still in history. So
"bloat" is essentially solved at the history level; what remains is in-tree
cleanup, not size. (This finding was first measured against the pre-rewrite
backup `palette-pre-history-rewrite-20260528.bak`, whose 965 MB pack still
carried the dead blobs; the numbers here are re-measured against the live
post-rewrite repo.)

The actual problem — and it is real — is the one the owner named: mechanisms are
built but never wired as gates. The same pattern recurs in every dimension
audited: the right tool exists, nothing enforces its use, and entropy wins by
default.

| Mechanism that exists | Why it is bypassed | Evidence |
|---|---|---|
| Canonical zarr helper `open_zarr_group_direct` (`shared/zarr_helpers.py`) | nothing forbids inline use | ~506 inline `zarr.open_group(` calls repo-wide |
| Stage provenance + registry as system-of-record | import never calls it; only one stage hard-validates | import writes ~50 inline attrs; `registry/stage_complete.py:64` `_ENFORCE_STAGE_ARRAY_VALIDATION_FOR = {"detect_quality"}`; all others soft-pass via `legacy_default=True` (`:183`) |
| 2,767 tests with executable contract validators | no CI runs them | no `.github/`, pre-commit, tox, or nox anywhere |
| Contract-freshness checker + `contract-meta` headers (46/53 contracts) | never invoked | `scripts/check_contract_freshness.py` wired into nothing |
| Single anchor doc `current_pipeline_contract.md` | no index; ~33% of docs are decaying audits | `doc_code_divergence_inventory_2026-05-01.md` invalidated its own findings within ~9 days |

Every row is the same story: good bones, no immune system. The 33 `backfill_*.py`
scripts and the four `docs_staleness_audit_2026-05-{17,18,20,21}.md` files written
in five days are not the disease — they are the owner manually performing the
work a gate would automate.

## Findings by dimension

### 1. Shared helpers — real duplication, effectively unenforced

`shared/` ≈ 46 modules / 16K LOC; `utils/` ≈ 210 modules / 139K LOC.

- **Zarr open/discover/run-resolution is the worst cluster.** Four+ overlapping
  modules: `shared/zarr_helpers.py`, `shared/zarr_discovery.py` (4 near-identical
  `discover_*`), `shared/zarr_run_completion.py`, `shared/zarr/schema.py`
  (a *second* `get_latest_run`), plus a *third* latest-run resolver in
  `utils/zarr_metadata.py:125`. Three independent "latest run" resolvers.
- **~506 inline `zarr.open_group`/`open` calls** despite the canonical helper —
  the helper is decorative. Examples: `utils/run_keypoints_batch.py:281,297`,
  `inference/predict_detections.py:131`.
- **5+ grayscale/luma converters** (`crop_image_source.py:280`,
  `utils/training_image_profile.py:206`, `create_clipped_training_zarr.py:258`,
  `segmentation/subject_segmentation.py:165`, hardcoded BT.601 weights at
  `refinement/regenerate_interpolated_crops.py:519-521`), separate from the
  pynvvc luma path. (Ties directly to the decode-unification proposal in
  `import_step_design_review_2026-06-04.md`.)
- **3 provenance/lineage attr writers** (`stage_provenance.py`,
  `run_lineage_fingerprint.py`, `subject_mask_component_provenance.py`).
- Confirmed truly-dead library module: `shared/registry_stage_complete.py`
  (0 importers, a back-compat shim). Most 0-import `utils/` modules are
  legitimate CLI entrypoints, not dead.

### 2. Metadata / provenance — genuinely fragmented

At least 6–7 distinct, overlapping mechanisms:

1. Registry SQLite ledger (`registry/db.py`, `status_ledger.py`) — system-of-record for "is a stage done."
2. Stage-completion bridge (`registry/stage_complete.py:256 emit_stage_completion`) — the only fail-closed gate, and it hard-validates arrays for exactly one stage.
3. Zarr run-completion contract (`shared/zarr_run_completion.py`) — per-run `palette_run_completion_status` + parent `latest`/`latest_complete`.
4. Stage provenance payload (`shared/stage_provenance.py`) — `palette_stage_provenance` v1, emitted ad hoc by ~35 modules, validated by none except one narrow path.
5. Source/row lineage attrs (`shared/provenance_attrs.py`, `row_lineage.py`).
6. Run lineage fingerprint (`shared/run_lineage_fingerprint.py`) — analysis stages only.
7. Inline `raw_video` attrs dump (`capture/import_video.py:899-1290`) — ~50 attrs with its *own* `git_*` fields and its *own* completion marker `import_stage`.

Enforcement is written-but-not-validated. The import stage — the foundational
producer of `raw_video` — calls none of `emit_stage_completion`,
`build_stage_provenance`, or the registry (zero grep hits), and is not a
registered registry stage. Two competing completion vocabularies (`import_stage`
attrs vs `palette_run_completion_status`/`latest`) never reconcile.
`diagnostics/check_provenance_consistency.py` exists *because* the systems
disagree — and it reads only zarr attrs, never the registry DB, so
registry-vs-zarr drift is unchecked.

### 3. Bloat — history already trimmed; residual is in-tree cleanup

- `.git` ≈ 22 MB: the history rewrite has been applied to this repo and the
  ~473 MB of dead `*.pt`/`*.engine`/`*.zip` blobs are gone. (The pre-rewrite
  backup still carries them in a 965 MB pack.)
- Residual history bloat is minor: `test_frames/*.jpg` (~3.6 MB) and two
  `*.ipynb` notebooks (~2 MB) remain in history — candidates for a later trim,
  not urgent.
- Still illegitimately-tracked generated files in the tree: `trtexec_output/*.jpg`
  and `apps/manim/*.npz`.
- **44 loose `src/`-root scripts, ~18K LOC, 43 of 44 imported by nothing** — a
  self-contained dead island with zero references from `src/fisheye/`. Plus root
  scratch docs (`ENUM_*`, `*_SUMMARY`, `*_GUIDE`) and one-off `check_*`/`speed_test*`.

### 4. Documentation — intent present, control absent

- 304 markdown under `docs/` + 13 root `*.md`.
- ~33% ephemeral churn (41 `*_todo.md`, 13 dated audits, 11 `*_status.md`, 11
  `*staleness*`); ~44% durable (53 `*_contract.md`, 36 `*_design.md`, 15
  workflow), only 2 `*_reference.md`.
- The staleness audits themselves report ongoing doc/code drift; the
  2026-05-01 divergence inventory had findings marked stale by a 05-09/05-10
  recheck appended in place.
- **No index, TOC, or architecture overview**; README does not link into
  `docs/`. A single anchor (`current_pipeline_contract.md`,
  `palette_pipeline_overview.mmd`) exists but the rest is an unindexed flat pile.
- Real but unwired infra: `check_contract_freshness.py`,
  `generate_registry_schema_reference.py`; 46/53 contracts carry `contract-meta`
  headers. None invoked by any gate.

### 5. Tests / CI — strong suite, zero automatic enforcement

- 413 test files under `tests/unit/fisheye/`, ~119K LOC, ~2,767 test functions —
  a large, real suite with executable validators (`roi_pixel_contract` ×21,
  `validate_run` ×15, `stage_complete`, run-completion `latest` logic, etc.).
- **No CI of any kind.** Tests run only on manual `pytest` invocation.
- Orphaned tests: 5 stray root/`src/` `test_*.py` excluded by `pytest.ini`
  `testpaths = tests` — false confidence.
- Thinnest coverage on the highest-consequence math: coordinate/homography
  transforms have one test that checks calibration *parsing*, not pixel→mm→
  projector round-trip correctness.
- Real-zarr and CUDA/pynvvc-parity tests are routinely deferred/skipped in
  sandbox (per AGENTS.md), so the integration edges rarely run.

## What is genuinely risky vs. merely untidy

Risky (root-cause and high-consequence):

1. **No CI at all** — the root cause that renders every other mechanism opt-in.
2. **Coordinate/homography transforms essentially unguarded** — the one place
   where "no test" can mean "silently wrong science," not just "messy."
3. **Import bypasses the registry/provenance machinery** — foundational stage
   off-contract; two unreconciled completion vocabularies. (Corroborated
   independently of, and consistent with, `import_step_design_review_2026-06-04.md`.)
4. **Metadata genuinely fragmented** — 6–7 overlapping mechanisms; the
   consistency diagnostic exists because they disagree and doesn't even cover
   registry-vs-zarr.

Untidy (cleanup, low urgency):

- 44 orphaned `src/`-root scripts (~18K dead LOC).
- Residual history binaries (`test_frames/*.jpg`, notebooks) and tracked
  generated files (`trtexec_output/*.jpg`, `apps/manim/*.npz`).
- ~33% ephemeral docs with no `archive/`.
- Duplicate grayscale converters and latest-run resolvers.

## Highest-leverage recommendations

1. **Add minimal CI with three gates**, in order:
   - `pytest` on push — turns 2,767 latent tests into a real safety net (largest ROI).
   - A lint rule banning raw `zarr.open_group` outside the canonical helper —
     makes the helper load-bearing; the ~506 bypasses become a burn-down list.
   - Wire `check_contract_freshness.py` (already written).
2. **Add a coordinate-transform round-trip test** — the one targeted test-debt
   item where absence risks incorrect results, not just mess.
3. **Bring import onto the registry/provenance contract** — route it through
   `emit_stage_completion` + `build_stage_provenance`; collapse `import_stage`
   into `palette_run_completion_status`. (Detailed in the import review.)
4. **Cleanup (low urgency):** delete the 44 orphaned `src/`-root scripts;
   untrack the generated `trtexec_output/*.jpg` and `apps/manim/*.npz`; move
   ephemeral audits/todos to `docs/archive/`; add a `docs/` index linked from
   README. (The major history rewrite is already done; a later filter-repo pass
   could drop the residual `test_frames/*.jpg` and notebooks.)

## Counterweight

This is not a rotting mess; it is a strong system with no immune system. The
bones are genuinely good — a registry system-of-record, executable validators,
a single anchor contract, ~2,767 tests, and a contract-freshness checker most
projects never write. The failure mode is that the discipline was kept manual,
so it does not scale past the owner's own attention. CI converts that already-
practiced discipline into something the repository enforces automatically, and
most of the observed fragmentation stops growing the moment a gate exists.

## Cross-references

- `docs/import_step_design_review_2026-06-04.md` — import stage design,
  decode-backend unification, provenance bypass.
- `docs/current_pipeline_contract.md` — existing anchor contract.
- `scripts/check_contract_freshness.py` — unwired freshness checker.
