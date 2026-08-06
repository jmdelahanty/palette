# Brief: fail-closed science defaults — kill the silent substitutions

**From:** commander session, 2026-08-04
**Status: READY. Independent of the reorg — can run in parallel with any other brief.**
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Read first:** `docs/arena_geometry*` / `fisheye/shared/arena_geometry.py` (the incident
this brief generalizes), `src/fisheye/analysis/chaser_distance_coordinate_publication.py:690`
(the correct pattern you are propagating).

Ground rules: local `sun` is ground truth; fresh worktree from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python`.

---

## Why this brief exists

This repo has already had three results corrupted by silent substitution, all recorded in
project memory: the arena-geometry circle that inverted thigmotaxis (0.35→0.35 instead of
0.37→0.87), the raw-centroid speed noise floor that manufactured a false avoidance result,
and the stale registry copy that under-selected the cohort. Each was found by luck or by
someone re-deriving a number by hand.

The same failure class is still live in the code, in one case directly beneath a docstring
warning about the original incident. **A wrong number that raises is recoverable; a wrong
number that defaults to 1.0 is a retracted figure.**

This brief is scientific-integrity work. It is not a refactor, and it must not become one.

---

## Scope — three checkpoints, stop and report after each

### Checkpoint 1: the calibration defaults (highest severity)

`src/fisheye/analysis/chaser_epoch_behavior_summary.py`, verified 2026-08-04:

- `:360` — `pixels_per_mm = _optional_float(run_group.attrs.get("pixels_per_mm_projector")) or 1.0`
  A missing calibration silently becomes **1 pixel per mm** and is fed straight into the
  geometry resolver. Lines `:355-358` of the same file document the arena-geometry
  incident. The warning and the bug are two lines apart.
- `:642` — `float(run_group.attrs.get("fps") or 1.0)`
- `:1303` — `fps = float(attrs.get("fps") or 1.0)`
  A missing frame rate silently becomes **1 fps**, making every rate and duration wrong by
  roughly 100×.

1. Replace all three with the fail-closed pattern already used in this codebase at
   `analysis/chaser_distance_coordinate_publication.py:690-694` (`_fail("... fallbacks are
   forbidden")`). Match that module's error shape and message style — do not invent a new
   exception type.
2. **Then sweep for the class, not just the instances.** Grep `analysis/`, `shared/`, and
   `analysis_workflows/` for `or 1.0`, `or 1`, `or 0.0`, `.get(..., 1.0)`, and
   `getattr(..., 1.0)` applied to **any calibration, frame-rate, scale, or unit-conversion
   quantity**. Report every hit with a keep/fix decision and a one-line reason. A default
   on a cosmetic quantity (a plot alpha, a max-points cap) is fine; a default on a
   quantity that multiplies into a published number is not.
3. **Blast-radius check, required:** for each site you change, determine whether any run
   in the canonical store is currently **missing** that attr. If yes, the fix will start
   raising on real data — that is correct behavior, but the commander must know before it
   lands. Report counts; do not backfill anything (that is
   `brief_migration_ledger_2026-08-04.md`'s territory).
4. **STOP and report.**

### Checkpoint 2: config strictness

Verified 2026-08-04: `extra="forbid"` appears **zero times** across `src/` and `tests/`.
`extra="allow"` appears 6 times (`diagnostics/prepare_detect_training.py`). The remaining
~18 pydantic models use pydantic's default `ignore`. A typo'd key in
`configs/fisheye/*.yaml` is silently dropped and the default silently used. 46 modules
call `yaml.safe_load`; roughly four feed a schema.

5. Add `model_config = ConfigDict(extra="forbid")` to the default-mode models in
   `training/config.py` and elsewhere. For the 6 explicit `extra="allow"` models, either
   justify the allow in a comment or tighten it — an unjustified `allow` is the same bug
   with extra steps.
6. `detection/detect_yolo.py` `load_config` searches six hardcoded fallback paths
   (including `Path.home()/'gitrepos/palette/...'`) and **returns `{}`** when nothing is
   found (`:1501`) — a missing config runs on defaults, silently. Make it raise
   `FileNotFoundError` naming every path it tried. Remove the `Path.home()` fallback:
   a config resolution that depends on the operator's home directory is not reproducible.
7. Add one test that round-trips every file in `configs/fisheye/*.yaml` through its owning
   model. This test is the regression guard — without it, `forbid` decays.
8. **Expect breakage.** Some existing config file almost certainly carries a key no model
   declares. Each one you find is a finding: report whether the key was a typo (fix the
   config), a removed feature (delete the key), or an undeclared real option (add it to
   the model). Do not silently delete keys to make the test pass.
9. **STOP and report.**

### Checkpoint 3: delete the decoy

10. `src/config_models.py` is dead and actively misleading: **zero importers** anywhere in
    `src/`, `tests/`, `scripts/`, `apps/`, `tools/`; uses the pydantic-v1 `@validator` API
    against pydantic 2.11; and its `check_zarr_path` validator requires a `.zgroup` file —
    i.e. zarr **v2**, while `pyproject.toml` pins `zarr>=3,<4`. It would reject every store
    in the project. Delete it. Recount importers first.
11. Grep for and fix any doc that points at it as the config-model home.

---

## Explicitly OUT of scope

- Backfilling any missing attr the checkpoint-1 sweep uncovers. Report counts only.
- Re-running or re-deriving any published analysis. If a fix changes a number, that is a
  commander decision with its own brief.
- The `or 1.0` idiom on non-scientific quantities. Report, do not fix.
- Anything in `labeling/web.py`'s HTTP boundary — the untyped `body.get(...)` writes are a
  real problem and a separate brief; do not open that surface here.
- Touching `arena_geometry.py` itself. It is correct; it is the standard you are applying
  elsewhere.

## Constraints

- **Every change must make behavior stricter, never looser.** If a fix would require
  loosening something else to keep tests green, stop and report — that is a signal the
  test was encoding the bug.
- Watch for tests that codify the old permissive behavior. This repo has precedent
  (`tests/unit/fisheye/test_registry_dedupe.py:250-280` asserts a destructive dedupe is
  correct). An agent working to keep tests green will preserve a bug and conclude it was
  intentional. **If a test asserts a silent default is fine, that test is part of the
  defect** — report it, and fix the test in the same commit with a comment explaining what
  changed and why.
- One commit per concern, independently revertible.

## Validation bar

- Focused tests: each fail-closed site raises on a missing attr and passes on a present
  one; every `configs/fisheye/*.yaml` validates against its model; an unknown key is
  rejected with a useful message.
- Full non-GPU suite green, rebased on current `sun`. Establish your own baseline — the
  suite is currently red for unrelated reasons.
- `git diff --check` + `py_compile` clean.
- Grep proof that `src/config_models.py` has no remaining references.

## Reporting

Branch `agent/palette/failclosed-science-defaults` from current `sun`. Per checkpoint:
the sites changed, the full sweep table with keep/fix decisions and reasons, the
blast-radius counts (how many canonical-store runs lack each attr you now require), the
config keys that turned out to be typos vs removed features vs undeclared options, and any
test you found that was encoding a silent-default bug.
