# Brief: vectorized FrameDomains conversion (`agent/frame-domains-vectorized`)

**From:** commander session, 2026-07-05 (late evening)
**Status: READY.** One slice, one agent.
**Do NOT push or merge — the commander verifies and merges.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `HANDOFF_2026-07-05.md` operating-notes section, then
`docs/diagnostics/frame_domains_consumer_census_2026-07-05.md` (the census this slice
unblocks). Ground rules: local `sun` is ground truth; fresh worktree on
`agent/frame-domains-vectorized` from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python` (conda; never uv, never a `.venv`); sync
code only; re-locate cited line numbers by content. Local gates before "done":
import-linter via `scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py`, `git diff --check`, `py_compile`, focused
tests, then full suite `PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
Baseline: recount on your branch first; last commander-verified count is recorded at
the bottom of this brief.

## Why

Slice C migrated every consumer the current resolver API can safely handle (census
items 1–4 done, 5 partial). The remaining consumers — census items 5-remainder, 6,
and 8 (`utils/regenerate_training_crops_pynvvc.py`, `detection/detect_keypoints_yolo.py`,
refine/curation/tracking paths) — map **large frame arrays**, and
`FrameDomains.convert()` (`src/fisheye/shared/frame_domains.py`) is a per-value Python
loop over `edge.mapping.get(value)`. Migrating them today would replace vectorized
NumPy indexing with a Python loop — the hot-path guard in the census forbids exactly
that. This slice removes the blocker at the resolver, so consumers can migrate against
it in a later slice.

## Scope

1. **Vectorize `convert()` internally. Do NOT add a new public method.** Same
   signature, same semantics, same exceptions — every existing caller silently gets
   the fast path, and there is no second API for future consumers to choose wrong.
   Implementation is your call, but the structure of the edges points at:
   - Forward raw/run edges have `arange(n)` sources (see `_build_raw_edges` /
     `_build_run_edges`) → a dense `np.ndarray` lookup indexed directly.
   - Inverse edges have arbitrary integer keys → sorted-key arrays +
     `np.searchsorted` (verify hit with an equality check after the gather).
   Build the per-edge array representation lazily on first `convert()` of that edge
   and cache it on the internal edge object, so `FrameDomains` construction cost does
   not change for count-only consumers (`detect_quality.py`,
   `detection_coverage_dashboard.py` construct the resolver and never convert).

2. **Exact behavior parity, pinned by tests that keep the old loop as oracle:**
   - Copy the current per-value loop into the test module verbatim as
     `_legacy_convert_loop(...)` and assert equality of outputs (values, dtype
     int64, shape round-trip including multi-dimensional inputs and empty arrays)
     across: identity conversion, forward edge, inverse edge, and scalar-ish
     1-element inputs.
   - **Error parity:** unmapped values must still raise `FrameDomainUnmappedError`
     with the same fields — including the `first values:` sample being the first
     ≤10 unmappable values **in flat input order** (vectorized code must not
     reorder or sort them; `flat[~hit_mask][:10]` preserves order) and
     `mapping_arrays` populated. Missing-edge and scope-assertion errors unchanged.
   - `count()`, `capabilities()`, the `FrameDomainEdge` public dataclass fields,
     and `_missing` bookkeeping are all untouched surfaces — no signature or field
     changes anywhere public.

3. **Evidence of the speedup, not just correctness:** a quick benchmark (in the
   report, not committed as a test) converting ~1e6 values through a forward and an
   inverse edge, old loop vs new path. If the vectorized path is not at least ~10×
   faster on 1e6 values, say so and stop rather than landing complexity for nothing.

4. **Optionally ONE demonstration test** showing a hot-path-shaped call (large array
   through `convert()`) — a test, not a consumer migration.

Out of scope — **hard boundaries, not suggestions:**
- **NO consumer migrations in this slice.** Census items 5-remainder/6/8 migrate in a
  follow-up slice against the landed API. Do not touch
  `regenerate_training_crops_pynvvc.py`, `detect_keypoints_yolo.py`, or any
  refine/curation/tracking file.
- No new public API (`convert_batch`, `convert_many`, etc.) — vectorization is
  internal to `convert()`.
- No changes to domain semantics, edge construction rules, duplicate/many-to-one
  handling (`_inverse_mapping` returning `None` is load-bearing —
  `training_image_profile.py`'s legacy fallback depends on the edge being absent),
  or the `FrameDomainError`/`FrameDomainUnmappedError` hierarchy.
- No changes to `Recording.frame_domains()` or the import-time identity stamp.

## Reporting

Branch + commit SHAs, the benchmark numbers (sizes, old vs new timings), grep/test
proof that no consumer files changed, full-suite counts (recounted baseline → final),
and any premise discrepancies.

Last commander-verified suite baseline: **3,426 passed / 2 skipped** on `sun` at
`30ede1e` (post-checkpoint-4 merge, full non-GPU suite, `-n 16`). Recount on your
branch regardless — merge traffic is heavy.
