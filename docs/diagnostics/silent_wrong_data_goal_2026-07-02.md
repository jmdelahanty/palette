# Silent-Wrong-Data Goal Checklist

<!-- contract-meta
status: active
created: 2026-07-02
owner: jeremy
branch: agent/silent-wrong-data
worktree: /home/delahantyj@hhmi.org/gitrepos/palette-silent-wrong-data
related: docs/diagnostics/codebase_review_2026-07-01.md,
         docs/acquisition_video_stream_source_policy.md,
         docs/identity_lineage_staleness_review.md
-->

## Purpose

Track the remaining silent-wrong-data work in one place so implementation can resume
without reconstructing state from chat context. The work converts silent data corruption
paths into checked failures, unifies forward pixel semantics, and records enough
provenance to distinguish future data from historical data.

Use `scripts/py` for Python commands. For zarr-backed tests, run pytest outside the
Codex sandbox with:

```bash
PYTHONPATH=/home/delahantyj@hhmi.org/gitrepos/palette-silent-wrong-data/src \
  /home/delahantyj@hhmi.org/gitrepos/palette-silent-wrong-data/scripts/py \
  -m pytest -q -m 'not gpu' tests/
```

## Current State

Worktree: `/home/delahantyj@hhmi.org/gitrepos/palette-silent-wrong-data`

Branch: `agent/silent-wrong-data`

Completed commits:

- `2e663f6 training: drop mismatched loader rows at index construction`
- `eb510d0 crop: convert ordering, bounds, and decode assumptions to loud failures`
- `29e9e6a pixels: enforce gpu-primary decode with full-range contract and stamping`
- `e2520c5 masks: require probabilities_encoding at decode time`

Next implementation item:

- Item 5: golden pixel parity.

Item 3 validation:

- `89 passed` for the widened focused Item 3 run, including flat ROI cache, clipped
  flat ROI cache, training crop PyNvVC parity, crop image source, detect backend
  contract, training/export stamping, detect compute smoke, pixel contract audit,
  PyNvVC crop regeneration, and clipped-training provenance tests.
- Full non-GPU gate: `3203 passed, 2 skipped, 1 deselected, 204 warnings`.

Item 3 guard-placement ruling:

- `test_build_flat_roi_cache_pynvvc_luma_streams_rows_in_source_order` remains in CI.
  It injects a fake PyNvVC reader to verify row-ordering and pixel-contract behavior
  without CUDA.
- No-fallback guards belong at backend-selection boundaries. Once a reader is explicitly
  selected or injected, row streaming remains backend-agnostic and accepts non-CUDA fake
  tensors.

Item 4 validation:

- `110 passed` for the focused Item 4 run, covering mask probability encoding,
  mask-source loading, subject-mask finalization, review and tuner paths, merge/backfill
  utilities, and subject-mask diagnostic benchmark readers.
- Full non-GPU gate: `3209 passed, 2 skipped, 1 deselected, 201 warnings`.

## Active Item 5 Next Step

- [ ] Review existing pixel parity/cache tests before adding the golden contract test.
- [ ] Generate a synthetic full-range mono video with a deliberately misleading `tv`
  container/range flag.
- [ ] Keep the primary PyNvVC direct-Y assertion GPU-marked while keeping CI coverage for
  retained opt-in CPU/cache paths.
- [ ] Assert cache build/read/rebuild byte equality for the supported CPU-free test path.

## Completed Item 4 Notes

- [x] Located readers of `mask_probs_roi` / mask probability stores that decode based
  on `probabilities_encoding` or guess when the attr is missing.
- [x] Implemented one shared required-encoding decoder in a neutral shared home.
- [x] Routed these readers through it: `finalize_subject_masks.py`,
  `refined_subject_mask_review.py`, `subject_mask_tuner.py`, `shared/mask_source.py`,
  `backfill_subject_mask_runs.py`, `merge_subject_mask_runs.py`, and subject-mask
  diagnostic benchmarks.
- [x] Missing or unrecognized `probabilities_encoding` raises with run path and observed
  dtype.
- [x] Added focused tests and reran the full non-GPU gate before committing Item 4.

Useful Item 3 focused command that is known green:

```bash
PYTHONPATH=/home/delahantyj@hhmi.org/gitrepos/palette-silent-wrong-data/src \
  /home/delahantyj@hhmi.org/gitrepos/palette-silent-wrong-data/scripts/py \
  -m pytest -q tests/unit/fisheye/test_flat_roi_cache.py \
    tests/unit/fisheye/test_crop_image_source.py \
    tests/unit/fisheye/test_detect_yolo_resize_contract.py \
    tests/unit/fisheye/test_import_sampled_training_pynvvc.py
```

## Silent-Wrong-Data Checklist

- [x] Item 1 - training loader: drop out-of-bounds frame-index rows during index
  construction; count and fail if drop rate is systematic.
- [x] Item 2 - crop validation: assert sorted source rows where required, validate frame
  bounds, and fail on decode failures instead of writing plausible black crops.
- [x] Item 3 - pixel contract and decode policy:
  - [x] Convert implicit GPU to CPU decode fallback paths into loud failures.
  - [x] Keep CPU decode only as explicit inspection/non-production paths.
  - [x] Stamp new crop and training outputs with `source_pixels`, decode backend, applied
    range semantics, and center rounding.
  - [x] Use Orange source contract semantics: full-range mono8 in the Y plane, even when
    existing containers advertise `color_range=tv`.
  - [x] Use `np.round` for crop-center quantization in forward paths.
  - [x] Fix the remaining flat ROI cache test/gate failure.
  - [x] Full non-GPU gate green.
- [x] Item 4 - probabilities encoding:
  - [x] Implement one shared required-encoding decoder for mask probability stores.
  - [x] Route all current readers through it.
  - [x] Missing or unknown `probabilities_encoding` raises with run path and dtype.
  - [x] Add focused tests.
  - [x] Full non-GPU gate green.
- [ ] Item 5 - golden pixel parity:
  - [ ] Generate a synthetic full-range mono video with a `tv` container/range flag.
  - [ ] GPU-marked primary assertion: PyNvVC direct-Y decode equals encoded Y.
  - [ ] CI-runnable assertions cover any retained CPU inspection path and cache
    build/read/rebuild byte equality.
- [ ] Item 6 - OpenCV detect fallback:
  - [ ] Flush final partial batch after EOF regardless of reported frame count.
  - [ ] Gate OpenCV as explicit/opt-in, not silent fallback.
  - [ ] Add fake-capture test for over-reported `CAP_PROP_FRAME_COUNT`.
- [ ] Item 7 - exposure census:
  - [ ] Read-only census of existing training datasets and major run families.
  - [ ] Include forensic classifier for historical datasets with no pixel-contract attrs:
    direct-Y-like, range-expanded-like, or indeterminate.
  - [ ] Use limited-range expansion signature: unreachable intensity comb from
    `(Y - 16) * 255 / 219` plus clamped lows.
  - [ ] Commit report as `docs/diagnostics/pixel_decode_exposure_census_2026-07-02.md`.

## Reporting Requirements

Final report for this branch should lead with the exposure census. Include:

- Per-item test evidence and final suite counts.
- Pixel/range measurements and rounding impact. Current expected note: grayscale formulas
  are arithmetic no-ops for Orange mono content; crop-center rounding can move future
  crops by up to 1 px relative to truncation.
- Census table and classification confidence basis.
- Existing tests changed because they relied on old silent behavior.
- Any backward-compat tripwire that fired.
- CPU decode paths deleted versus retained as explicit inspection paths, with caller
  analysis.

## Queued Follow-Up: Authoritative-Run Pointer

Do not start until this silent-wrong-data slice has merged and the exposure census has
been read.

Spec basis: `docs/identity_lineage_staleness_review.md` Rec 1.

Tightened implementation constraints:

- Store the pointer where completion state already lives, on the stage parent. Prefer:
  - `authoritative_run = <run_name>`
  - `authoritative_run_provenance = {approved_by, approved_at, git_sha, note}`
- The resolver is read-only and never writes fallback state.
- The setter refuses missing, incomplete, failed, or otherwise non-complete runs.
- If no authoritative pointer exists anywhere for a recording, behavior must remain
  exactly current/latest-complete behavior.
- Do not blanket-replace every `resolve_latest_complete_run_name` call. Classify call
  sites as:
  - scientific/default run resolution
  - staleness/oracle
  - UI display
  - maintenance/latest inventory
- Reroute default scientific resolution and staleness/oracle paths first. Leave true
  inventory/maintenance latest queries alone unless there is a specific reason.
- Smoke-run stale rule: if an authoritative pointer exists, later smoke runs do not make
  downstream stages stale. Without an authoritative pointer, fallback behavior remains
  latest-complete.

Planned commits:

- `shared: add authoritative-run pointer primitive`
- `shared: resolve default run and staleness authoritative-first`
- `cli: add palette approve verb`

Acceptance for that follow-up:

- Unit tests for set/read/clear, fallback-to-latest, read-only resolver behavior, and a
  newer complete run not changing authority.
- A false-stale regression test: approved composed run plus later bounded-apply smoke run
  reports the composed run as authoritative and not stale.
- `palette approve` dry-run and apply tests with the run-verb envelope.
- Full non-GPU suite green.
