# Exact chaser successor Marimo status — 2026-08-26

<!-- contract-meta
status: active
implementation: partial
last_verified: 2026-08-27
-->

## Purpose

This note records the boundary between persisted chaser publications and the
read-only Marimo recording explorer after the protocol-semantic successor work.
Persisted PNG/PDF bundles and Marimo are independent consumers of the same
immutable analysis products. External images do not replace, disable, or alter
interactive exploration.

The implementation plan and acceptance gates for the findings below are in
[`chaser_exact_successor_interactive_visualization_implementation_checklist_2026-08-27.md`](chaser_exact_successor_interactive_visualization_implementation_checklist_2026-08-27.md).

## Implemented in code

The explorer contains a `stimulus_chaser_exact_successors` capability. It is
intended to discover a recording only when one complete, selector-ineligible
spatial-occupancy successor seals an ordered keypoint/detection bundle and each
referenced relative-frame and radial child still has the exact bound manifest
digest and provider identity. The 2026-08-27 live-artifact audit below found a
reader compatibility defect that currently prevents receipt-bound v4 bundles
from reaching this capability.

The declared interactive views are:

1. paired-provider distance, moving-reference radial selection, and exact-time
   near-field summaries;
2. full-session and exact protocol-epoch fish–chaser distance traces;
3. exact protocol-epoch fish positions with logged chaser-position overlays in
   the reviewed circular arena; and
4. exact manifest, child identity, provider authority, and display-projection
   provenance.

The adapter resolves no `latest`, default, promoted, or fallback selector. It
deep-audits the small spatial/radial successor arrays. For frame-scale panels it
reads and content-hashes only the relative-frame arrays required by the selected
view. This is a bounded replacement for rescanning unrelated archive payloads.

Plotly point reduction is display-only and is recorded in figure metadata. The
distance projection preserves source order, local first/last/minimum/maximum
values, and all observed missing-data breaks. The position projection preserves
source order and coordinate extrema. Neither projection is a scientific input,
and neither writes a derived product to Zarr.

The discovery specification is currently synthesized in memory. The
`artifact_path` ending in `interactive` is a capability address, not a
persisted Zarr child: the audited spatial successor contains scientific arrays
and its sealed manifest, but no `interactive/zarr.json` or `visualizations`
subgroup. This can remain a valid read-only architecture if the runtime adapter
and its display semantics are versioned and tested. A separately persisted
interactive descriptor would be an additive visualization publication, not a
reason to mutate an immutable scientific successor.

## Live-artifact audit correction — 2026-08-27

### Static publication is complete and independently reproducible

The selector-ineligible execution completed for 80 eligible GoodBatBadBat
recordings: task 1 was the controlled canary and task indices `2-76,81-84`
completed in LSF array job `153756073`. Indices `77-80` remain excluded by the
known protocol-semantic ordering/overlap failure and were not silently treated
as members of the successful cohort.

The final read-only audit found:

- 80 unique cohort receipts, each bound to Palette commit
  `65b06a2f6ab4c4c30a92a8248a7ffb1742d70b0c` and frozen task digest
  `bbbcc9710d38c6bd5c0a611bc68b40c24ef908f1faaf93913b26b914a0256509`;
- 240 canonical plot receipts and 720 source-validation receipts;
- 720 PNG and 720 PDF files, exactly 18 outputs for every eligible recording;
- successful independent rehashing of all 1,440 plot outputs; and
- `selector_eligible=false`, `production_authority=false`, and
  `registry_update=false` throughout.

The Zarr publications contain the scientific arrays needed by an interactive
viewer. For example, the spatial successor persists exact epoch/provider
occupancy grids, bin edges, candidate and validity denominators, and arena
membership evidence. External plot-receipt JSON persists exact source
bindings, source-validation receipt digests, rendering and binning parameters,
and output path/size/SHA-256 records. Those JSON files are safe static
publication receipts; they are not standalone interactive datasets and the
current Marimo exact-successor adapter does not consume them.

### Consolidated visibility is correct

The root consolidated metadata for the audited recording contains the exact v4
spatial successor and all of its arrays. Direct metadata and consolidated
discovery both expose the run. The parent has no `latest`, `selected`,
`authoritative`, `default`, or other forbidden selector attribute. Therefore
the zero-option result is not caused by missing or stale consolidated metadata.

### Pre-fix exact Marimo discovery returned zero live v4 bundles

A read-only smoke against recording
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` and its exact spatial run
`goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260827_body_frame_projection_receipt_bound_v4`
returned zero `palette-chaser-exact-successor-explorer-v1` options.

All scientific identities pass individually:

- the spatial manifest is complete, canonical, selector-ineligible, and
  digest-valid;
- both relative-frame children validate against their exact run path, manifest
  digest, recording identity, provider identity, and completion state; and
- both radial children validate against their exact run path, manifest digest,
  recording identity, provider identity, and completion state.

Discovery fails only because it compares two versioned representations of the
same relative child by literal dictionary equality. A radial successor records
the minimal immutable child identity:

```json
{
  "run_path": "analysis/chaser_relative_frame_runs/<exact-run>",
  "manifest_sha256": "<exact-manifest-digest>"
}
```

The receipt-bound spatial bundle correctly enriches that identity:

```json
{
  "run_path": "analysis/chaser_relative_frame_runs/<exact-run>",
  "manifest_sha256": "<exact-manifest-digest>",
  "validation_receipt_sha256": "<exact-receipt-digest>",
  "verification_mode": "receipt_bound_targeted_array_rehash_v1"
}
```

The `run_path` and `manifest_sha256` agree exactly. The additional fields are
stronger publication evidence, not a different child. Both discovery in
`apps/marimo/components/registry.py` and the projection loader in
`apps/marimo/components/chaser_exact_successors.py` currently require whole-
object equality and therefore fail closed for the enriched binding.

The unit fixture uses the minimal two-field binding in both parent and child,
so it does not exercise the real receipt-bound shape. This is a test-boundary
gap. The fix must not loosen exact identity: it must validate a closed
receipt-bound binding schema, compare normalized child identity
`(run_path, manifest_sha256)`, retain and validate the receipt digest and
verification mode, and make discovery and loading use the same helper.

This defect affects reader admission only. It does not invalidate or require
rewriting the scientific Zarr publications, static images, plot receipts, or
source-validation receipts. No cohort recomputation or Zarr migration is
needed for the compatibility fix.

### Local reader correction and acceptance evidence — 2026-08-27

The isolated implementation worktree
`/tmp/palette-chaser-interactive-reader-20260827` now contains the reader-only
correction. It is based on `origin/main` commit
`b90bb2a5904f382289fb5e6e6f714f2d56505dac`; the implementation itself is not
yet committed or CI-qualified.

The correction adds one Marimo-independent exact-relative-child binding
validator. It accepts only the closed minimal and receipt-bound profiles,
normalizes scientific identity to exact `run_path` plus `manifest_sha256`,
and retains the validation-receipt digest and mode in an immutable proof. Both
registry discovery and projection loading use that validator. Neither path
reopens the external receipt, and the recorded validation behavior says so.
The synthesized explorer spec is now schema version 2.

Local fail-closed validation passed:

- 25 focused unit tests, including the production asymmetric binding shape,
  wrong child identity, malformed receipt evidence, unsupported mode,
  unexpected fields, provider mismatch/order, recording mismatch, incomplete
  child, forbidden selector state, and no unconsolidated retry;
- Ruff check, Python compilation, and `git diff --check`; and
- `scripts/py -m marimo check apps/marimo/palette_explorer.py` outside the
  sandbox.

Read-only acceptance against the frozen smoke recording now discovers exactly
one schema-v2 option in 0.359 seconds. Every currently declared analysis loads
successfully:

- `radial_near_field`: 3.265 seconds, deep-auditing the exact spatial and two
  radial children without loading relative arrays;
- `provenance`: 3.209 seconds;
- `distance_traces`: 9.491 seconds; and
- `trajectory_overlays`: 8.350 seconds.

Each relative-array load content-hashed the 13 required array names across
both 149,936-frame, two-chaser providers, representing 26,988,480 decoded
array bytes. Both receipt digests remain visible in projection provenance, but
are described as sealed manifest evidence that was not independently reopened.

Metadata-only consolidated discovery over the frozen cohort task manifest
passed 80/80 eligible recordings in 31.079 seconds. Task indices `77-80`
remained explicitly excluded, and the task digest was
`bbbcc9710d38c6bd5c0a611bc68b40c24ef908f1faaf93913b26b914a0256509`.
No Zarr, selector, registry, authority, receipt, or static plot was written or
changed. Required CI and a commit-pinned representative deep-load report still
gate merge readiness.

## Persisted but not yet mounted in this capability

These products are safe candidates for later read-only panels, but were kept
out of the initial component:

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

The currently declared exact-successor analyses are limited to
`radial_near_field`, `distance_traces`, `trajectory_overlays`, and
`provenance`. They do not provide interactive equivalents for all nine static
figure families. In particular, paired-provider spatial heatmaps,
controller-trial/bout-response detail, escape/freeze detail, and the composed
full dashboard remain absent from this exact reader. Older GoodCopBadCop
components or candidate views must not be used as an implicit fallback for
these receipt-bound successors.

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

## Live recording smoke target and current result

The required implementation smoke target is:

`2026-08-10T17-20-55Z_arena_1_goodbatbadbat`

Its spatial bundle binds the first-class providers
`keypoint_anatomical_triad_mean.v1` and `detection_bbox_centroid.v1`, three exact
chaser epochs, their paired radial successors, and their paired relative-frame
runs. The pre-fix reader returned zero options despite correct direct and
consolidated metadata. The locally validated correction now discovers exactly
one option and loads all four declared analyses without a selector, legacy,
candidate, or unconsolidated fallback. It remains an uncommitted local result
until the implementation is commit-pinned and required CI is green.
