# Brief: artifact-inventory accessor + verb port (narrow-waist integration)

**From:** commander session, 2026-07-05
**Status: READY.**
**Branch:** `agent/inventory-accessor-verb` from current `sun`. One commit per concern.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
Do NOT push or merge — the commander verifies and merges.

## Why this exists

The recording-artifact-inventory module landed on `sun` (`shared/recording_artifact_inventory.py`,
merged at `ca4444a`) but only its **`fisheye.utils` CLI wrapper** came with it. A parallel
agent had also built the narrow-waist integration layer — `Recording.artifact_inventory()`,
a `fisheye.api` export, and a `palette artifacts` verb — but that branch was discarded during
cleanup, so those three surfaces are NOT on `sun`. Today the inventory is reachable only via
`python -m fisheye.utils.recording_artifact_inventory`, which is on the utils Phase 2
deprecation path. This slice ports the integration layer fresh against the merged module and
retires the utils entry point.

**Do NOT try to salvage the old orphaned commit.** Its module version differs from the one that
merged and would conflict. Write against `shared/recording_artifact_inventory.py` as it is on `sun`.

## What already exists (build ON this — do not modify the module)

- **The builder:** `build_recording_artifact_inventory(root: zarr.Group, *, zarr_path=None) -> dict`
  at `src/fisheye/shared/recording_artifact_inventory.py:553`. Read-only, JSON-serializable,
  schema `palette.recording_artifact_inventory.v1`. Leave this file untouched.
- **The Recording handle:** `src/fisheye/shared/recording.py`. `Recording.__init__` already opens
  `self.root = open_zarr_group_direct(ref.zarr_path, mode="r")` and holds `self.path`. So the
  accessor is a thin call — no new zarr-opening logic.
- **The verb/envelope pattern to mirror:** `status`/`plan` in `src/fisheye/cli/palette.py`
  (`StatusRequest`/`PlanRequest` dataclasses at ~line 65; `status()`/`plan()` verbs at ~line 911;
  dispatch in `_run_readonly` at line 937; subparser registration at line 2065; the standard
  read-only envelope — `schema/command/status/reason_code/recording/run/artifacts/metrics/next_hints/provenance`).
- **The api surface:** `src/fisheye/api.py` re-exports the verbs + `Recording`.

## Scope — four concerns, one commit each

### 1. Accessor: `Recording.artifact_inventory()`
In `src/fisheye/shared/recording.py`, add a method that returns
`build_recording_artifact_inventory(self.root, zarr_path=self.path)`. It is read-only, takes no
run-resolution args (the inventory spans all run families by design), and returns the raw dict.
Mirror the docstring style of the sibling accessors (`detections`/`keypoints`/`subject_masks`).

### 2. Verb: `palette artifacts <recording>`
In `src/fisheye/cli/palette.py`:
- `ArtifactsRequest(recording, registry=None)` dataclass beside `StatusRequest`/`PlanRequest`.
- `artifacts(request: ArtifactsRequest) -> dict` verb. Resolve the recording the SAME way
  `status`/`plan` do (`_resolve_dataset_and_stages` / the shared recording resolver), open the
  resolved recording via `open_recording`, call `.artifact_inventory()`, and wrap it in the
  standard read-only envelope (`read_only: True` provenance, `_utc_now()` timestamp). Return the
  `RECORDING_NOT_FOUND`/blocked envelope on resolution failure exactly as `status` does.
- Register an `artifacts` subparser (`recording` positional, `--registry`, `--json`) and dispatch
  it through `_run_readonly` (extend the handler's command branch).
- Text (non-`--json`) renderer: port the summary from the utils wrapper's `_print_text_summary`
  (run-family / run / visualization counts + acquisition streams + per-family resolved run). Keep
  it terse; `--json` emits the full inventory dict inside the envelope.

**Resolution ambiguity to decide + REPORT:** a recording can have both `analysis.zarr` and
`training.zarr`. The utils wrapper took an explicit zarr path; `status`/`plan` resolve a dataset.
Pick the resolution `status`/`plan` already use and document which zarr the verb inventories.
If the shared resolver returns a single zarr per recording, that answers it — state it. If it can
return more than one, default to the analysis zarr and note the limitation in `next_hints`; do NOT
invent a multi-zarr envelope in this slice.

### 3. api export
Add `artifacts` + `ArtifactsRequest` to `src/fisheye/api.py`'s imports and `__all__`, keeping the
list sorted as it is now.

### 4. Retire the utils wrapper + fix the doc
- DELETE `src/fisheye/utils/recording_artifact_inventory.py` (solo-use repo: no deprecation
  wrapper). Grep-proof no remaining references in src/, tests/, docs/, or registry-browser/TUI
  tooling before deleting.
- Update `docs/recording_artifact_inventory_contract.md` — the CLI example at line 8
  (`scripts/py -m fisheye.utils.recording_artifact_inventory ... --json`) becomes
  `palette artifacts <recording> --json`. Scan the rest of that doc and
  `docs/artifact_storage_map.md` for other invocation references and retarget them. Standing
  rule: no doc landmines pointing at a deleted module.

## Explicitly OUT of scope

- Any change to `shared/recording_artifact_inventory.py` (the builder). Behavior is frozen; this
  is pure integration.
- New inventory fields, cross-recording rollups, or registry writes. Read-only, single recording.
- Multi-zarr (analysis + training in one call) envelopes — flag, don't build.
- Moving the builder out of `shared/` or touching the utils Phase 2 layering work.

## Constraints

- Read-only end to end: no zarr writes, no registry writes, `read_only: True` in provenance.
- The verb must fail the way `status`/`plan` fail (blocked/`RECORDING_NOT_FOUND` envelope, matching
  exit code), not raise.
- `~/miniconda3/envs/palette-py311/bin/python` for everything; no venv; sync code (asyncio is
  fragile in this sandbox).

## Validation bar

- Focused tests: accessor returns the schema-v1 dict against a synthetic recording zarr fixture;
  `artifacts(ArtifactsRequest(...))` returns a well-formed envelope for a resolvable recording AND
  the blocked envelope for a missing one; `palette artifacts <path> --json` round-trips the
  inventory through the CLI. Reuse the existing inventory test fixtures where possible.
- Full non-GPU suite: `PYTHONPATH=src ~/miniconda3/envs/palette-py311/bin/python -m pytest tests
  -m "not gpu" -q -n 16`. Baseline was 3372 passed / 2 skipped at `sun` tip `8af2443` — recount.
  If the sandbox blocks `-n 16`, fall back to `-n 4`/serial and say so.
- `git diff --check` + `py_compile` clean on every touched file.
- Grep proof: zero references to `fisheye.utils.recording_artifact_inventory` anywhere after the
  delete.

## Reporting

Verb resolution decision (which zarr, and the evidence from the shared resolver), the envelope
shape as landed, deleted-module grep proof, focused-test results, full-suite counts and how run,
files touched with line counts, anything left undone.
