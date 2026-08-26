# Exact chaser successor Marimo status — 2026-08-26

## Purpose

This note records the boundary between persisted chaser publications and the
read-only Marimo recording explorer after the protocol-semantic successor work.
Persisted PNG/PDF bundles and Marimo are independent consumers of the same
immutable analysis products. External images do not replace, disable, or alter
interactive exploration.

## Implemented in this change

The explorer discovers a new `stimulus_chaser_exact_successors` capability only
when one complete, selector-ineligible spatial-occupancy successor seals an
ordered keypoint/detection bundle and each referenced relative-frame and radial
child still has the exact bound manifest digest and provider identity.

The first interactive views are:

1. paired-provider distance, moving-reference radial selection, and exact-time
   near-field summaries;
2. full-session and exact protocol-epoch fish–chaser distance traces; and
3. exact protocol-epoch fish positions with logged chaser-position overlays in
   the reviewed circular arena.

The adapter resolves no `latest`, default, promoted, or fallback selector. It
deep-audits the small spatial/radial successor arrays. For frame-scale panels it
reads and content-hashes only the relative-frame arrays required by the selected
view. This is a bounded replacement for rescanning unrelated archive payloads.

Plotly point reduction is display-only and is recorded in figure metadata. The
distance projection preserves source order, local first/last/minimum/maximum
values, and all observed missing-data breaks. The position projection preserves
source order and coordinate extrema. Neither projection is a scientific input,
and neither writes a derived product to Zarr.

## Persisted but not yet mounted in this capability

These products are safe candidates for later read-only panels, but were kept
out of this first change so the three requested figures can land independently:

- paired-provider spatial-occupancy heatmaps from
  `analysis/chaser_spatial_occupancy_runs`;
- exact controller-trial summaries and trigger-aligned distance views;
- generalized bout-response summaries;
- escape/freeze trial and event summaries;
- gaze/controller-trial views where a complete gaze successor is present; and
- the full-profile readiness and module-binding envelope.

Each addition should anchor to the same exact bundle or to an exact full-profile
binding, validate child manifest and payload identities, and render persisted
arrays without recomputing trial membership, event classification, timing, or
geometry.

## Still evidence-blocked or deferred

- Ring-entry video clips remain deferred. A safe implementation needs explicit
  frame-to-video identity, clip-boundary policy, and an immutable media receipt.
- Camera-exposure or physical-stimulus synchronization must not be inferred from
  session timestamps. Current exact-time panels retain their declared session-
  time semantics.
- Promotion or production selector activation remains outside the explorer. The
  current successors are intentionally selector-ineligible.
- Cohort comparison remains a separate product. This recording explorer does
  not silently aggregate recordings or choose among multiple bundles.

## Audited recording smoke target

The implementation is exercised against:

`2026-08-10T17-20-55Z_arena_1_goodbatbadbat`

Its spatial bundle binds the first-class providers
`keypoint_anatomical_triad_mean.v1` and `detection_bbox_centroid.v1`, three exact
chaser epochs, their paired radial successors, and their paired relative-frame
runs.
