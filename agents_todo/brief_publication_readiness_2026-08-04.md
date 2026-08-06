# Brief: publication readiness — make the repo clonable, citable, and installable

**From:** commander session, 2026-08-04
**Status: READY. Independent of the reorg — can run in parallel with any other brief.**
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Read first:** `README.md`, `AGENTS.md`, `pyproject.toml`, `docs/environment_setup.md`.

Ground rules: local `sun` is ground truth; fresh worktree from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python`.

---

## Why this brief exists

This repository is intended to back a publication. Verified 2026-08-04, it currently
cannot support one:

- **No `LICENSE`, no `CITATION.cff`, no `CHANGELOG`, and zero git tags.** `version` in
  `pyproject.toml` has been `0.1.0` since inception. There is no way for a reviewer or
  replicator to cite the code that produced a figure. `README.md` itself states no license
  is declared.
- **`origin/HEAD` points at `main`, whose tip is 2025-08-13.** The real trunk is `sun`,
  **1,857 commits ahead**. Anyone who clones lands on a year-stale tree and sees
  essentially none of the work. The CI badge already hardcodes `?branch=sun`, so the
  maintainer knows — a cloner does not.
- **The install instructions break at step one.** `README.md:29` says
  `scripts/py -m pip install -e ".[dev]"`, but `scripts/py` resolves only
  `$PALETTE_PYTHON`, `$HOME/miniconda3/envs/palette-py311/bin/python`, or
  `$HOME/miniforge3/envs/palette-py311/bin/python`, and otherwise exits **127** with
  *"Could not find palette-py311 Python."* The README never says
  `conda env create -f environment.yml`, so on a fresh machine the very first command
  fails. (The `PALETTE_PYTHON` escape hatch is undocumented in the README — worth
  surfacing there too.)

None of this is hard. All of it is blocking.

---

## Scope — three checkpoints, stop and report after each

### Checkpoint 1: unbreak the first five minutes

1. **Trunk.** Repoint `origin/HEAD` to `sun`, **or** fast-forward `main` to `sun` and adopt
   `main` as trunk. Recommend one in your report with reasoning; this is a commander
   decision — **prepare it, state the exact commands, do not execute the remote change.**
2. **Install path.** Add the missing `conda env create -f environment.yml` step to the
   README before the `pip install -e` line. Then **follow your own instructions from a
   clean shell** and report exactly where they break. Fix each break. An install doc that
   has not been executed is a hypothesis.
3. **`AGENTS.md` broken pointer.** `:38` points at `docs/sandbox_zarr_fallback.md`, which
   was archived to `docs/archive/`. The 2026-07-23 "fix references" pass missed it. This is
   a live instruction file for coding agents pointing at a moved path — highest blast
   radius broken link in the repo. Also drop the `scripts/py src/test_fisheye.py` example
   at `:120` (that file is a deletion target in checkpoint 3), and reconcile the test
   command with `README.md:115` — they currently contradict each other.
4. **STOP and report.**

### Checkpoint 2: make it citable

5. Add `LICENSE`. **This is an HHMI institutional decision, not yours** — surface the
   question with 2–3 concrete options and their implications (permissive vs copyleft vs
   institution-specific) and let the commander choose. Do not pick one.
6. Add `CITATION.cff` with the author, affiliation, and repository URL.
7. Cut an annotated tag at the commit backing the GoodCopBadCop analyses, and bump
   `version` in `pyproject.toml` to match. Identify that commit from the analysis history
   and **state your evidence for the choice** — a wrong tag is worse than no tag.
8. Add a short `CHANGELOG.md` seeded from the tag forward. Do not attempt to reconstruct
   1,857 commits of history.
9. **README "Scientific Methods" section.** A paper reviewer currently has no pointer to
   where any behavioral metric is defined. Add a section naming the modules that compute
   the published quantities — thigmotaxis / arena geometry (`fisheye.shared.arena_geometry`
   — and note the dish-mask-vs-experimental-area distinction, it silently inverted a result
   once), speed and kinematics (`analysis/track_kinematics.py`, and note that
   `speed_smoothed_mm` is the valid one — raw centroid speed has a ~1.6 mm/s noise floor
   that produced an artifactual result), bout detection, and escape detection
   (`chaser_escape_events` — read `gain_mm`/`recapture_mm`, never `net_mm`).
   **Verify each module path before writing it down.**
10. **STOP and report.**

### Checkpoint 3: delete the junk drawer

All counts verified 2026-08-04 — **recount before deleting; branches move.**

11. **The 44 loose top-level `src/*.py` scripts — 18,067 LOC, zero inbound imports** from
    `src/fisheye`, `tests`, `scripts`, `apps`, or `tools`. These are not merely dead:
    because `pip install -e .` puts bare `src/` on `sys.path`, `import visualizer`,
    `import models`, and `import test_fisheye` all currently resolve as top-level names.
    Deleting them removes a live shadowing hazard.
    - `src/config_models.py` is one of the 44 but is owned by
      `brief_failclosed_science_defaults_2026-08-04.md` checkpoint 3. **Coordinate — do not
      both delete it.**
12. **`src/chaser_analysis/` — 48 files, 22,799 LOC, last touched 2025-10-30.** It has
    exactly one edge into the live package: `analysis/swim_bout_statistics.py:38` imports
    `BoutWithUnits`, `EnhancedBoutAnalyzer`, `CalibrationData` from
    `chaser_analysis/swimming_bout_analysis.py`. That file is **1,324 lines and
    self-contained** — it imports nothing else from `chaser_analysis`. Port it into
    `fisheye/analysis/`, then the other **47 files (~21,500 LOC) drop free**.
    `swim_bout_statistics` itself is live (registered in
    `analysis_workflows/surface_classification_catalog.py:689` with a test) — this is a
    port, not a delete.
13. **`src/models/`** — 1 file, 178 LOC, imported only by `training/export_onnx.py:35`
    inside a `try`. Vendor it into `fisheye/training/`.
14. **After 11–13, set `[tool.setuptools.packages.find] include = ["fisheye*"]`.** This is
    the structural win — larger than the LOC. `src/palette.egg-info/top_level.txt`
    currently installs `chaser_analysis`, `models`, and `trt_cpp_predictor` as real
    top-level packages, and `root_package = "fisheye"` makes import-linter structurally
    blind to all of them.
15. **The 4 `test_*.py` shipped inside the package** —
    `src/fisheye/test_{standalone,decord_minimal,video_end_frames,system_utils}.py`.
    `test_standalone.py:8` does a `sys.path.insert` that exists only because of the loose-
    script layout.
16. **Stray top-level files**, all cold 5+ months: `check_frame_gaps.py`, `cufile.json`,
    `test.yaml`, `test_video_integrity.sh`, `pipeline_flowchart.html`. **Keep
    `runnables.yaml`** — it is a live Fileglancer service descriptor.
17. **`pip-packages-exact.txt` and `conda-packages-explicit.txt`** — last touched
    2025-09-24, referenced by nothing but `docs/environment_setup.md`.
    `conda-packages-explicit.txt` is **py310 throughout** while everything else pins 3.11;
    `pip-packages-exact.txt` pins `torch==2.5.1+cu121` against `environment.yml`'s
    `pytorch-cuda=12.4` and a `zarr==3.0.0a5` alpha. Files named "exact" that are ten
    months stale and describe the wrong interpreter are worse than absent. Delete both and
    fix the doc reference. **State in the report that this leaves the project with no
    pipeline lockfile** — generating one is a separate brief, and the commander needs that
    on the record.
18. Empty `.agents/` and `.codex/` directories.

---

## Explicitly OUT of scope

- Generating a real conda lockfile. Named as a gap, not fixed here.
- `playgrounds/` (63 files, last touched 2026-07-23 — **active**), `tools/` and
  `examples/` (2026-07-18). Not dead. Do not touch.
- **Anything eye-*angle*.** `analysis/eye_angle_{analysis,io,schema,storage}.py` were all
  committed 2026-08-03 and are under active development. Eye *masks* are the deprecated
  surface and are already nearly severed. Confusing the two deletes live work.
- `src/fisheye/` internals — `brief_subtraction_wave_2026-08-03.md` owns those.
- `docs/` reorganization and the missing `docs/README.md` index (596 files, no index) —
  worth doing, separate brief.

## Constraints

- **Do not execute the `origin/HEAD` change or push any tag.** Prepare and report the exact
  commands; the commander runs them.
- Every deletion is preceded by a fresh reference census over `src/`, `tests/`, `scripts/`,
  `tools/`, `apps/`, `configs/`, `docs/` **plus a string grep over `*.sh`, `*.yaml`,
  `*.toml`, `*.json`** — this repo dispatches some jobs by module-name string, so the
  import graph alone is insufficient.
- Docs referencing a deleted file are part of the deletion, not a follow-up.
- One commit per concern. The `chaser_analysis` port and the 47-file delete are **separate
  commits**, port first, suite green in between.

## Validation bar

- Full non-GPU suite green, rebased on current `sun`. Establish your own baseline — the
  suite is currently red for unrelated reasons.
- After step 14: `pip install -e .` in a clean env, then confirm `import visualizer`,
  `import models`, and `import chaser_analysis` all **fail**, and `import fisheye`
  succeeds. Paste the proof.
- `palette --help` exits 0.
- `git diff --check` + `py_compile` clean.
- Grep proof for every deleted module.

## Reporting

Branch `agent/palette/publication-readiness` from current `sun`. Per checkpoint: the trunk
recommendation with commands, exactly where the README install broke when you ran it, the
license options surfaced, your evidence for the chosen tag commit, the verified module
paths in the Methods section, the deletion census, LOC delta, and the on-the-record note
that no pipeline lockfile exists.
