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

## Real-recording canary and numerical correction (later September 6 update)

This section is a new evidence snapshot; the pre-commit snapshot above and all
saved canary outputs remain unchanged. The exercised clean implementation was
`ac73a61067118fc1a41699267dfa376a15f4bac6` (draft PR 152). Its remote CI status
is tracked separately by the integration owner; no integration or promotion
was performed by this canary.

Source: `/groups/johnson/johnsonlab/jeremy/recordings/2026-08-10T17-20-55Z_arena_1_goodbatbadbat`,
camera `2010093`, video `cams/Cam2010093_2026-08-10T17-20-55Z_arena_1.mp4`.
The existing source-camera/pixel-authority loader and read-only media probe
validated 152,035 frames, 100 Hz, native 4512×4512, HEVC `yuvj420p`, full-range
`pc`, and pixel-frame identity
`070aa3ebe2959a50a279bdfbe95779e99ff991cd71629edf83b3b93cea994050`.
The exact 28,507,864,226-byte source hash is
`ac7f298cdaf6cc63d01983cd3e258fcd126314c67e320db157b176335378967c`.

All evidence is under `/tmp/palette-rim-real-canary-20260906.pxVuAn`:
`preflight.json`, `unpatched_probe_result.json`, `unpatched_probe.log`,
`host_frame_check.json`, `host_luma_parity.json`,
`host_adapter_canary_result.json`, `legacy_common_median_comparison.json`,
the frozen `host_adapter_top_rim_probe` package, and `isolated_analysis.zarr`.
No recording, live analysis archive, registry, selector, or production data
was written. Source size and modification time were checked unchanged.

The **unpatched** device-output CLI failed in 1.82 s at `torch.from_dlpack`
with `AssertionError: Torch not compiled with CUDA enabled`. The workstation
has an RTX A6000 and PyNvVideoCodec 2.1.0, but `scripts/py` loads CPU-only
PyTorch 2.7.1. No dependency installation was attempted.

An explicitly approved external diagnostic adapter changed only NVDEC output
placement to documented `usedevicememory=False`. Native NV12 output, direct
Y-plane slicing, exact-seek proof, source hashing, sampling, fitting, package
creation, planner, publisher, and loader were unchanged. Adapter SHA-256
`fe30b8520fe1c0e9e65ba7538b935c45834351ff0258b7453a999875513562fb`
was embedded **before** fit-report freezing; no synthetic pixels or fabricated
seek proof were used. This does not validate the unpatched device-output CLI.

The first sampled frame, 14975, is 4512×4512 uint8 with values 9–255. Its
native luma bytes were identical to independent one-frame FFmpeg
`extractplanes=y` output, SHA-256
`735eb20fa9611e5bf69dd27511c8cf6fcef2abf503a0a4099ed728ee23adda87`.
No RGB conversion, intensity remapping, resizing, or full-video decode occurred.

Default scientific settings were preserved: early/middle/late five-second
windows, maximum 21 declared samples, actual 19 samples/window, and 2048px
coarse fitting. All 57 seeks landed on the exact declared target keyframe;
each submitted three packets (171 total), including decoder latency. Decode
took 39.21 s, probe/report/presentation took 101.04 s total, and isolated
publication/loading finished at 106.08 s. Peak process RSS was 1,797,956 KiB;
the largest uint8 window stack was 386,804,736 bytes and three medians total
61,074,432 bytes. Whole-source hashing used the existing bounded 1MiB block
reader; it was a 28.5GB provenance read, not an eager video decode.

The real package and required three-panel montage were produced and visually
inspected, then passed the unpatched fit-review planner/publisher/loader in
isolated scratch. Report hash:
`c6c57dd78df3f4c1d8224f0d238a11e3f8f4063102c671348d9e744f826532c1`.
Immutable fit-review run: `arena-geometry-fit-review-9b620fee922ab3027c8940a9`;
record hash `9b620fee922ab3027c8940a932a561108f1af54734bd77257178d643608e4920`.
It remains selector-ineligible with no latest selectors. Pixel/PNG/source
metric bindings and observed-medoid validation passed.

However, the canary exposed a real eligibility defect: all candidates had
p95 radial offsets approximately `4.00003..4.00005` pixels because distances
were reconstructed from float32-angle points, although every radial sample
lies on the exact `[-4,4]` pixel grid. The inclusive 4px extraction cutoff
therefore incorrectly rejected every candidate. All three windows honestly
recorded `top_rim_preference_unresolved_no_eligible_edge_v1`; this is not a
successful physical-top-rim finding or an automatic acceptance.

That saved result has selected-center spread 1.21954px, selected-radius range
0.63624px, family-center spread 3.78145px, and family-radius Hausdorff distance
7.50627px. A bounded legacy diagnostic on the **same three frozen medians**
took 8.13 s: legacy radii were 2160.80494, 2160.24787, 2159.99877px versus
the defective preferred/fallback radii 2154.08000, 2153.70895, 2153.44375px.
These are algorithm-selected image contours, not physical-radius errors.

The follow-up is a numerical measurement/enforcement correction, not threshold
relaxation. New metrics read the exact selected signed radial-grid offsets
from the existing sampler. Legacy callers still receive the identical two
arrays; their combined golden SHA-256 remains
`f8741e0af57235058987c30dfa5f0ea44988e25f2b9a567c2a50b18d44d782db`.
The named effective parameter
`radial_offset_measurement_method=signed_sampling_grid_offset_v1` explicitly
changes the scientific-recipe digest, preventing old frozen threshold
configuration from silently reusing corrected measurements. The 4px cutoff
and all other scientific settings are unchanged. Historical outputs are not
rewritten, and whole-receipt equality is not claimed for corrected outputs.

The regression first reproduced the wrong value `4.000005910296061` on a small
deterministic gradient. After correction, the focused suite including
`test_dish_rim_offset_precision.py` passed **76 tests in 8.34s**. Corrected
real-median replay and exact-commit CI for that follow-up remain pending at
this snapshot. The CPU-only unpatched CLI limitation, calibration/locked
holdout, clipped-layout equivalence, and automatic policy promotion remain
outstanding; no arbitrary acceptance thresholds have been added.
