# Brief: provenance content hashes — model/input artifact identity in the run chain

**From:** commander session, 2026-07-05
**Status: READY.**
**Branch:** `agent/provenance-content-hashes` from current local `sun`. One commit per
concern. Do NOT push or merge — the commander verifies and merges.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `docs/provenance_finalization_enforcement_design.md` (the landed Slice 2
+ Phase 5 substrate you are extending), `docs/provenance_enforcement_roadmap.md`,
`docs/diagnostics/tensorrt_export_realtime_context_2026-07-05.md` (engines context),
`src/fisheye/shared/run_provenance.py` (the payload you are extending),
`HANDOFF_2026-07-05.md` operating-notes section only.

## Why this exists

This is the last substantive gap of review Finding 3 (`docs/diagnostics/codebase_review_2026-07-01.md`)
and the remaining gate for the autonomous executor. Run provenance now enforces
`git_sha` + `config_hash` at finalization (epoch 2, default for new parents as of
`0087be4`), but the **model artifacts a run consumed are identified only by mutable
filesystem path**. A run that re-executes tomorrow against a silently-swapped checkpoint
produces different data with identical provenance. Named canaries:
- the SAM3 runtime, resolved as a mutable checkout via `PALETTE_SAM3_ROOT`/`SAM3_ROOT`
  (`utils/run_sam_subject_masks.py:61,237-257`) — no identity recorded at all;
- YOLO `.pt` checkpoints loaded by path in detect/keypoints writers;
- TensorRT engines for the realtime path (registered with hashes, but nothing verifies
  the file on disk still matches its registration).

**What already exists — build ON it, the gap is narrower than the review implied:**
- `utils/register_model_deployment_artifact.py` already computes `_sha256_file` for
  ONNX/engine/manifest at registration (`:25,129-149`); registry rows via
  `db.record_model_deployment_artifact` (`db.py:7968`) carry those hashes.
- Subject-mask model candidates already carry `model_sha256`/`metrics_sha256` through
  `registry/model_resolution.py` (`SubjectMaskModelCandidate`, ~line 457).
- `shared/run_provenance.py` has `sha256_payload`, `build_run_provenance`,
  `normalize_run_provenance`, `validate_run_provenance` — the payload plumbing.
The missing piece is the **join**: nothing puts artifact content hashes INTO
`run_provenance`, and nothing verifies a registry-known hash against the file actually
loaded.

## Locked design decisions (do not re-litigate; report friction instead)

1. **Scheme:** `content_v1` = lowercase hex sha256 of the file bytes. Recorded per
   artifact as `{role, path, fingerprint_scheme: "content_v1", sha256, size_bytes,
   mtime_ns, source}` where `source` ∈ `computed | registry | sidecar`.
2. **Where it lives:** a new `input_artifacts: [ ... ]` list inside the
   `run_provenance` payload built by `build_run_provenance` /
   `build_writer_run_provenance`. Normalization (`normalize_run_provenance`) passes it
   through; it participates in nothing else.
3. **NON-GATING in this slice.** The value-required validation gate stays exactly
   `git_sha` + `config_hash` (a locked Slice-2 decision). `input_artifacts` is recorded,
   never required. Making model-hash presence mandatory is a later epoch bump, not
   yours.
4. **Correctness first, cache best-effort.** Hashing a checkpoint costs seconds and the
   jobs run minutes–hours: always producing a correct hash beats caching. Optimization
   layers, in order of trust:
   a. **Registry-known hash** (deployment artifacts, subject-mask candidates): use it
      WITHOUT re-hashing only if current file `(size, mtime_ns)` matches what was true
      at registration where recorded; if stat mismatches or stat metadata is absent,
      re-hash. If the re-hash DISAGREES with the registry hash, record both
      (`sha256` = actual, `registry_sha256` = expected, `source: "computed"`,
      `mismatch: true`) and emit a loud warning — do NOT fail the run in this slice.
   b. **Sidecar cache**: best-effort `<artifact>.content_v1.json` (`{sha256, size_bytes,
      mtime_ns}`) written next to the artifact after a computed hash; trusted on exact
      `(size, mtime_ns)` match; ALL sidecar write failures silently ignored (read-only
      NFS artifact dirs are normal).
   c. Otherwise compute directly.
5. **SAM3 identity (the hard case):** record the sam3 checkout's `git_sha` (it is a git
   checkout; use the `run_provenance.git_identity(cwd=sam3_root)` helper) plus
   `content_v1` hashes of the checkpoint file(s) actually loaded IF the load path is
   determinable from `_inspect_sam3_runtime`/the predictor construction. If per-file
   determination is not feasible without touching SAM3 internals, fall back to a
   `manifest_v1` fingerprint of the sam3 `model/` dir (sorted relative-path + size +
   mtime_ns list, sha256 of that manifest) and REPORT the fallback — do not hash
   multi-GB weight trees file-by-file speculatively.
6. **Writers in scope** = the epoch-2 instrumented writers that load a model file:
   `detection/detect_yolo.py`, `detection/detect_keypoints_yolo.py`,
   `utils/run_sam_subject_masks.py`, `utils/run_subject_mask_batch_pipeline.py`.
   `tracking/crop.py` consumes detections, not a model — verify and, if it truly loads
   no model artifact, record nothing for it and say so. Census any OTHER epoch-2 writer
   you find loading a model file; instrument it too if trivial, otherwise list it.

## Scope — four concerns, one commit each

### 1. `shared/artifact_fingerprint.py` — the hashing core
`fingerprint_artifact(path, *, role, registry_hash=None, registry_stat=None) -> dict`
implementing decisions 1/4 (scheme, registry-trust, sidecar cache, mismatch handling),
plus the `manifest_v1` directory fallback for decision 5. Pure `shared/` (imports
nothing above `shared/`; the import-linter contract will hold you to this). Unit tests:
hash correctness, sidecar round-trip, stale-sidecar (mtime change) recompute,
registry-hash trust + mismatch path, unwritable-sidecar tolerance.

### 2. `run_provenance` payload extension
`input_artifacts` accepted/normalized/JSON-serialized through `build_run_provenance`,
`build_writer_run_provenance`, `normalize_run_provenance`; `validate_run_provenance`
UNCHANGED in what it requires (add a test asserting a payload with no `input_artifacts`
still validates — the non-gating guarantee).

### 3. Writer instrumentation
Each in-scope writer records the model artifact(s) it loads: detect/keypoints record
their `.pt` (`role: "detect_model"` / `"keypoint_model"`), the SAM writers record the
SAM3 identity per decision 5 (`role: "sam3_runtime"` / checkpoint roles). Where
resolution came through `registry/model_resolution.py` candidates carrying
`model_sha256`, pass it as `registry_hash` so decision 4a applies. Keep writer diffs
minimal — resolve-then-fingerprint-then-include; no restructuring.

### 4. Deployment-artifact verification + docs
- A small verification helper (natural home: `registry/model_resolution.py` or the new
  `shared/artifact_fingerprint.py` — your call, report it) that, given a deployment-
  artifact row and its on-disk path, reports `match | mismatch | missing`; wire it into
  whatever surface already inspects deployment artifacts (registry TUI/status if a
  natural hook exists — if none does, expose it as a function + test and note that no
  CLI surface consumes it yet; do NOT invent a new verb in this slice).
- Docs: mark the content-hash slice landed in `docs/provenance_enforcement_roadmap.md`;
  update the "Risks to manage" item 1 in
  `docs/diagnostics/tensorrt_export_realtime_context_2026-07-05.md` (engine identity —
  registration hashes now verifiable; realtime-side check still external); note the new
  `input_artifacts` field in `docs/provenance_finalization_enforcement_design.md`'s
  payload description or its addendum section.

## Explicitly OUT of scope
- Expanding the fail-closed validation gate (stays `git_sha` + `config_hash`).
- Source-video fingerprints (`stat_v1` in `audit_zarr_pixel_contracts.py` is a separate
  concern — leave untouched).
- The acquisition library / realtime engine loading (external repo).
- New registry tables, new CLI verbs, epoch bumps, backfilling hashes onto historical
  runs.
- Any change to `registry/db.py` beyond what concern 4's helper strictly needs to READ
  (the file is under a size ratchet; if you must grow it, the ratchet allows +200 from
  baseline — stay well under).

## Constraints
- Env: `~/miniconda3/envs/palette-py311/bin/python` (conda; never uv, never a `.venv`).
  Sync code only.
- No behavior change to any run's data path — provenance recording only. A hashing
  failure (unreadable file, etc.) must degrade to a recorded
  `{path, error, fingerprint_scheme: null}` entry + warning, never a run failure.
- CI now enforces import-linter layering and the file-size ratchet — run both locally
  (`lint-imports --config pyproject.toml`; `python scripts/check_file_size_ratchet.py`)
  before declaring done.
- Line numbers cited above may have drifted (heavy merge traffic) — re-locate by
  content. If a premise here is wrong, report it and stop that item; don't improvise.

## Validation bar
- Focused tests per concern (see concern 1 list; plus: writer-level test that a
  completed run's `run_provenance` attr contains a correct `input_artifacts` entry for
  a small fixture "model" file; non-gating test in concern 2).
- Full non-GPU suite: `PYTHONPATH=src ~/miniconda3/envs/palette-py311/bin/python -m
  pytest tests -m "not gpu" -q -n 16`. Baseline **3,382 passed / 2 skipped** at `sun`
  tip `8bfe960` — recount on your branch first.
- `git diff --check` + `py_compile` clean; import-linter + ratchet green.

## Reporting
Branch + commit SHAs; the SAM3 identity outcome (per-file hashes vs `manifest_v1`
fallback, with evidence of what the predictor actually loads); the crop-writer census
answer; where the verification helper landed and what consumes it; any registry-hash
mismatches encountered on real artifacts during testing; full-suite counts; anything
left undone.
