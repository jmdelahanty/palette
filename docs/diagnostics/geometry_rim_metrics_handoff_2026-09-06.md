# Geometry rim metrics and opt-in top-rim preference handoff

This is a development evidence snapshot, not a merge, deployment, selector,
scientific-acceptance, or automatic-policy promotion approval.

## Ownership and exact baseline

- Owner: `geometry_rim_metrics`, coordinated by the integration owner.
- Worktree: `/tmp/palette-geometry-rim-metrics-20260906`.
- Branch: `agent/palette/geometry-rim-metrics-20260906`.
- Exact baseline/HEAD when this pre-commit snapshot was written:
  `925f8c6285499d39a99475945ba1732dba7f6834` (reported all 24 required CI checks
  green by the integration owner). That evidence does not validate this diff.
- Changed source owners: `diagnostics/probe_recording_dish_rim_fit.py` and
  `analysis_workflows/materializers/arena_geometry_fit_review.py`.
- Additional scope: two new `test_dish_rim_metrics_*.py` files, this handoff,
  and mechanically refreshed `zarr_production_writer_census.json`.
- The policy owner owns the separate shadow evaluator/comparison adapter;
  the reference-catalog owner owns acquisition-reference identity. Neither is
  incorporated into this branch. Their shared validator dependency is the
  fit-review owner listed below, not a diagnostics import or copied helper.
- No unrelated dirty files, historical artifact rewrites, registry writes,
  selector writes, installation, cluster jobs, or production data mutations.
  User authorization now permits development commits/pushes, coordinated by
  the integration owner; this snapshot precedes that coordinated handoff.

## Classification and preserved contracts

This is a named/versioned scientific extraction recipe addition, with
enforcement for its new metric extension. It is not a behavior-preserving
cleanup of the legacy fitter and does not change the active default.

`--fit-recipe legacy_v1` remains the default and preserves the existing fit
method, default `CircleFit.to_json()` bytes, report envelope/fields, median
consensus, acquisition reveal ordering, and old fit-review artifact identities.
A golden serialized circle SHA-256 and full default-versus-explicit-legacy
report-byte comparison cover preservation; existing probe, candidate,
comparison, and fit-review regressions also pass.

The opt-in `--fit-recipe top_rim_preferred_v1` changes only that named recipe:

- The existing independent Hough/radial fitter supplies a blind search seed.
  A bounded median angular/radial profile searches a local family of gradient
  edges, refining them in narrow bands so distinct contours are not collapsed
  by the legacy wide-band refinement.
- The preferred diagnostic circle is the outermost sufficiently supported,
  fully visible, approximately concentric member in the bounded search. If no
  member qualifies, the fit explicitly records unresolved preference; absent
  support fails fitting. A contour-count is **not** a count of physical rims:
  two sides of one bright image contour can produce two gradient edges.
- Intended feature is `visible_dish_top_rim_edge`, but every observed contour
  remains `unclassified_concentric_rim_edge`. Metrics always retain
  `projected_edges_unresolved` and `physical_feature_identified=false`.
  Hough, outermost ordering, concentricity, and image support do not prove the
  top edge or identify an inner water-side edge, reflection, or projected edge.
- The representative consensus is an actual observed window medoid, with
  stable early/middle/late tie-breaking. No new radius is manufactured by
  averaging different projected edges.
- No acquisition reference, gate, center, radius, or tolerance is passed to
  extraction. Existing acquisition reveal still occurs after report freezing.
- The report adds an explicitly versioned `scientific_recipe` with all new
  extraction settings and a `rim_metrics` extension. Status remains
  `experimental_shadow_only_not_calibrated`. Historical reports are untouched.

Early/middle/late five-second keyframe sampling and its bounded decode path
are unchanged. The new radial profile uses 720 angles, at most a 256px radius
offset in either direction, and at most 16 retained candidate edges. Candidate
support scans reuse each window's gradient cutoff rather than repeatedly
scanning the full image. Only three temporal medians are persisted and decoded
one at a time for native-raster/pixel-identity verification; no full video or
cohort was decoded for this work.

## Agreed interfaces and evidence

The existing fit-review owner exports:

- `RIM_METRICS_SCHEMA_ID = "palette.diagnostics.dish_rim_metrics"`, version 1.
- `rim_metrics_source_sha256(source)`: strict canonical JSON SHA-256 of the
  **complete** existing probe source record. Non-finite identity is rejected,
  never normalized into the same identity as JSON null.
- `validate_rim_metrics(metrics, *, source, windows) -> None`.
- `summarize_rim_metric_temporal(windows)`.
- `select_rim_metrics_consensus_window(windows) -> str` and
  `validate_rim_metrics_consensus(consensus, *, windows) -> None`.

`rim_metrics.windows` contains exactly early/middle/late. Each window binds its
composite pixel digest, selected candidate ID, and every frozen candidate keyed
by ID. Candidate geometry is the existing native-pixel circle grammar and is
checked against the exact enclosing report. Measurements include visible and
supported angular fractions, four quadrant support fractions, longest missing
arc, median absolute and p95 radial offsets, gradient evidence, explicit
clipping/no-support/low-support flags, and native image shape. Absent support
has null residuals, not invented zero-error evidence.

The temporal summary records selected-center maximum pairwise distance,
selected-radius range, maximum within-window family-center spread, minimum
candidate count, and cross-window family-radius Hausdorff distance. The last
metric detects family dropout/movement using nearest image-edge radii; it does
not establish physical-feature correspondence. No acquisition physical-radius
error, gate alteration, or second acquisition tolerance is introduced.

New metric-bearing imports retain the original three median PNGs under
`visualizations/metric_composite_{early,middle,late}_png`. Import and immutable
loading validate both encoded PNG SHA-256 and the decoded native mono8 raster
SHA-256 against the frozen report. PNG dimensions/depth/color are checked
before decoding. Metrics remain reproducible after disposable probe scratch
is removed. Existing no-extension runs retain their old artifact set.

## Validation and outstanding gates

Preservation/adversarial tests were written first and initially failed on the
missing new APIs. Final focused workstation command (outside the sandbox):

```bash
scripts/py -m pytest -q \
  tests/unit/fisheye/test_dish_rim_metrics_recipe.py \
  tests/unit/fisheye/test_dish_rim_metrics_publication.py \
  tests/unit/fisheye/test_probe_recording_dish_rim_fit.py \
  tests/unit/fisheye/test_arena_geometry_fit_review.py \
  tests/unit/fisheye/test_arena_geometry_candidates.py \
  tests/unit/fisheye/test_arena_geometry_comparison.py
```

Result: **73 passed in 8.42s** on the final focused rerun. The decoder boundary is synthetic; fitting,
PNG generation, hash-bound package creation, real Zarr publication, immutable
loading, and metric validation are unpatched. Controls cover no support,
occlusion, clipping, invalid/non-finite coordinates, source mismatch, image
pixel mismatch despite a rehashed outer receipt, missing windows/metrics,
candidate mismatch, fabricated semantics, altered medoid, temporal drift,
and radial-family dropout. No GPU/real-video performance claim is made.

Local static checks passed: import-layer contracts, FPS/keypoint/tail/paradigm
authority ratchets, file-size ratchet, explicit Zarr metadata modes, observed
metadata literals, contract freshness, Python compilation, and diff whitespace.
The generated writer census initially reported stale line identities; its
existing generator refreshed that evidence, not a manual rewrite. The final
`scripts/py -m fisheye.diagnostics.zarr_storage_census --check` passed after
formatting and all source edits; authority/metadata ratchet baselines were
checked unchanged.

Required remote checks for this changed branch are all **unrun** at this
pre-commit snapshot: generated artifacts; import boundaries; file-size ratchet;
Zarr open metadata modes; observed metadata literals; active contract
freshness; package and collection; non-GPU test shards 0–15; and `ci-required`.
Local checks are not a substitute for successful required CI on the exact
incoming commit, nor for CI on the future combined branch.

Still unrun: real-video/CUDA canary, representative camera/registration
calibration, frozen August 10 derivation and locked August 11 holdout,
operator-adjudicated zero-false-pass and injected failure controls, and
whole-recording/clipped-layout equivalence for the new recipe. Gradient support
is not a validated semantic classifier or a calibrated false-positive bound.
No automatic selection is active; no threshold promotion or generic human
review claim is manufactured. Integration, deployment, and activation remain
separate and unauthorized until their respective gates are satisfied.

No compatibility removal is proposed. The legacy fitter/report remains the
maintained default; policy activation and any historical remeasurement remain
separately scoped work in the owning geometry queue.
