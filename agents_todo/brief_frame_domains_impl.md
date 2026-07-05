# Brief: FrameDomains resolver — implementation slice 1 (resolver + fixture, zero consumers)

**From:** commander session, 2026-07-05
**Status: READY.** Design APPROVED — `docs/frame_domains_resolver_design.md` (status:
approved, see its Approval Record for the maintainer's answers to the open questions).
The design doc is the specification; this brief only scopes the slice and adds
operational rules. **Where this brief and the design doc disagree, the design doc wins —
report the disagreement.**
**Read first:** the design doc IN FULL, then `src/fisheye/shared/recording.py` and
`src/fisheye/shared/run_resolution.py` (the API model you are following).

## Scope — what this slice builds

1. **`src/fisheye/shared/frame_domains.py`** — the `FrameDomains` resolver per the
   design doc's "Resolver API" section:
   - Constructed ONLY from recorded mapping arrays and explicit stage/schema semantics.
     **Length-derived totals (`len(arr)`, `max()+1`) are forbidden as construction
     inputs** — this is the RedScare anti-pattern the design exists to kill. If you
     find yourself needing one, that's a design gap: stop and report.
   - Named domains per the doc's Domain Contract (acquisition_frame is the canonical
     hub; supplemental `-1` crop-video values are unmappable, NOT a fifth domain).
   - Vectorized `convert(values, from_domain, to_domain)` — fail-loud by default on
     unmappable values or absent edges (raise; no masked returns, no sentinels).
   - `capabilities()` — which conversion edges exist for this recording and which
     mapping arrays back them.
   - Per-domain `count()` — the authoritative per-frame axis length in a domain.
   - Structured conversion provenance modeled on `RunResolution` (which arrays/attrs
     were used; which edge failed and why).
   - ZERO legacy adapters in this slice (Approval Record answer 3). Old stores lacking
     mappings get honest `capabilities()` gaps and loud `convert()` refusals.
2. **`Recording.frame_domains()`** on the accessor (`shared/recording.py`), following
   the existing method style. Read-only, like everything on `Recording`.
3. **Synthetic fixture** exercising BOTH dropped crop-video frames AND subsampling
   (the case absent from `/nvme1`), plus the plain cases (full import, subsampled-only).
   Build it as a reusable test fixture (conftest or fixture module beside the other
   zarr-building test helpers), not a one-off inside a single test.
4. **Import-time identity stamp (Approval Record answer 2):** `capture/import_video.py`
   writes the explicit `stored_zarr_frame -> acquisition_frame` identity mapping for
   full (non-subsampled) imports going forward, using whatever array/attr form the
   design doc specifies. Going-forward only; no backfill. Separate commit.

## Explicitly OUT of scope

- Migrating ANY consumer (detect_keypoints_yolo, detect_quality, crop_image_source,
  the crop-run builders, visualizers). Those are later slices, each deleting local
  arithmetic at one site — and crop-video consumers are HARD-GATED on a `/groups`
  census (Approval Record answer 1).
- Legacy adapters (answer 3), historical backfills, RedScare two-run backfill.
- The design doc's migration forcing function (grep-based CI check) — later slice,
  alongside the first consumer migration.
- Any write path other than the item-4 import stamp.

## Constraints

- The resolver is pure read-side library code in `shared/`: no `__main__`, no argparse,
  no upward imports (shared imports nothing above it — utils Phase 2 just fixed this
  layer; do not regress it).
- Do not modify `RunResolution`/`Recording` behavior beyond adding the new method.
- Commits: resolver + tests; accessor method + tests; fixture (may fold into resolver
  commit if entangled); import stamp + tests. Each independently revertible.

## Validation bar

- Unit tests: every conversion edge in both directions on the synthetic fixtures;
  fail-loud cases (absent edge, unmappable `-1`, out-of-range); `capabilities()`
  honesty on stores missing each mapping array; `count()` per domain; provenance
  contents; the import identity stamp round-trip (import → resolver reads the edge
  without degradation).
- A REAL-STORE smoke test (read-only) against one `/nvme1` recording with
  `original_frame_indices` + `frame_step=100`: resolver constructs, `capabilities()`
  reflects reality, a stored↔acquisition round-trip on real arrays is consistent. Mark
  it skip-if-path-absent so the suite stays green off-box.
- Full non-GPU suite: `PYTHONPATH=src ~/miniconda3/envs/palette-py311/bin/python -m
  pytest tests -m "not gpu" -q -n 16` (~60-80s). Baseline 3353 passed / 2 skipped as of
  3598499 — recount, it grows.
- `git diff --check` + `py_compile` clean.

## Reporting

Branch `agent/frame-domains-resolver` from current `sun`. Report: commits; the domain
graph as implemented (nodes + edges + backing arrays); any point where the design doc
was ambiguous and what you chose (flag for the doc to be amended); fixture design;
real-store smoke result; validation counts; PYTHONPATH=src confirmation.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
