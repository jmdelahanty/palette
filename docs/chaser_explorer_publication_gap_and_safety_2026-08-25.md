# Chaser explorer publication gap and safe-rendering plan (2026-08-25)

## Scope

The receipt-bound GoodBatBadBat cohort publication currently emits six figure
families per completed recording:

1. composable controller/bout/escape dashboard;
2. matched-provider distance CDFs;
3. generalized bout-response details;
4. trial escape/freeze details;
5. exact-trigger-aligned trial distance traces; and
6. pre/training/post paired-provider spatial-occupancy heatmaps.

That bundle is not equivalent to every interactive panel exposed by the
Marimo Palette explorer, and it is not by itself a full-profile-complete
claim.  The full-profile successor is a digest-bound applicability and product
composition envelope; it does not execute missing numerical modules or render
their figures.

All products discussed here remain selector-ineligible, non-authoritative
candidate evidence.  Adding a plot does not activate a registry selector or
promote a scientific source.

## Gap inventory

| Explorer product | Current publication state | Safe next requirement |
| --- | --- | --- |
| Full-session and per-epoch chaser distance traces | Trial-only traces and persisted distance CDFs are present | Render both first-class providers directly from receipt-validated relative-frame distance and session-timestamp arrays, using exact persisted epoch bounds and no interpolation |
| Epoch behavior summary and distributions | Absent | A new protocol-semantic successor must persist fish activity, bout, and inter-bout summary tables with exact exposure denominators before plotting |
| Egocentric bearing and polar distance | Absent | Validated body-frame supplier projected only by exact acquisition-frame identity, with missing source rows and present-but-invalid anatomical axes retained as distinct invalid evidence; no separate keypoint-authority review is required |
| Fish heading and alignment | Absent | The same validated body-frame supplier and exact projection; position-only relative-frame runs cannot support these claims, and motion heading or interpolation is not an anatomical fallback |
| Fish trajectory with chaser overlay | Spatial heatmaps exist, but trajectory/overlay context does not | Render source-camera fish and chaser positions from the two receipt-validated relative-frame providers, restrict epoch panels by exact persisted half-open acquisition-frame bounds, retain missing rows, avoid lines that invent continuity between position samples, and rasterize dense exact-row point artists |
| Configured-zone and chaser-quadrant occupancy | Absent; the existing similarly named product is a Cartesian occupancy heatmap | A modern semantic successor must persist exact zone definitions, geometry identity, denominators, and per-zone counts/fractions; do not silently reuse the legacy module |
| Chaser radial/near-field summary | The two successors and paired CDF plot are present, but their distance, ring-selection, near-fraction, dwell, and entry-rate panels are not in the cohort bundle | Render only the persisted paired-provider summary and radial tables after verifying identical semantic, geometry, chaser, occurrence, and timing authorities |
| Individual near-field visits | Absent | A successor schema must persist visit rows, entry/exit/censor identities, and exact timestamps; radial successor v1 contains aggregate visit evidence only |
| Escape-onset-aligned distance traces | Absent | Escape successor must persist the exact event-to-relative-row alignment rather than reconstructing it in a plotter |
| Eye--chaser gaze tracking | Absent | Validated body-frame supplier and separately reviewed eye-orientation authority bound to the same acquisition rows |
| Response regimes | Absent | A protocol-semantic successor and reviewed state/threshold contract are required |
| Ring-entry video | Deferred | A sealed video successor must bind source frames, timing, overlays, codec/rendering parameters, and output digest |

Legacy Marimo components may remain useful for discovery and qualitative
review, but their presence is not evidence that the corresponding calculation
is eligible for the successor publication.  Products that require new
scientific aggregation remain fail-closed until the required successor exists.

## Authorized render-only expansion

Three additions can be produced without rerunning or changing scientific
analytics because all numerical evidence already exists in immutable,
receipt-validated products:

1. **Paired-provider radial/near-field summary**: persisted median/IQR
   distance, area-corrected radial selection, near-zone fraction, exact dwell,
   and entry rate for keypoint and detection-centroid providers.
2. **Full-session and exact-epoch distance traces**: direct fish--chaser
   distance arrays for both providers and each chaser, with exact persisted
   pre/training/post acquisition-frame bounds.
3. **Exact-epoch fish trajectories with chaser overlays**: direct source-camera
   fish positions for each provider and the byte-identical logged chaser
   positions, with the reviewed circular-arena boundary shown for context.

These figures are views of existing evidence, not new scientific tables.
Their plotter may mask invalid rows, sort persisted table coordinates, convert
timestamps to a displayed relative origin, and rasterize dense artists while
retaining all exact source rows.  Position samples must remain unconnected so
the display cannot invent movement across gaps or logged position changes.  It
must not interpolate, impute, pool providers, infer trials, or recompute a
scientific summary.

## Required fail-closed checks

Before rendering the expanded bundle, the plotter must verify:

- one recording identity across controller, bout, escape, both radial, and
  both relative-frame inputs;
- deep or exact receipt-bound validation of every consumed array;
- distinct first-class fish-position provider identities;
- each radial product's exact binding to its relative-frame product;
- identical protocol-semantic selection and arena-geometry/scale authorities;
- identical epoch records and half-open acquisition-frame bounds;
- byte-identical acquisition-frame, timestamp, selection, chaser identity,
  occurrence, behavior-role, chaser-position, and chaser-position-validity
  arrays across providers;
- repeated per-chaser fish-position rows agree within each provider before
  collapsing them to one fish-position sample series;
- missing distance or position rows remain missing and break displayed traces;
  and
- every numerical coordinate, row-retention rule, rasterization choice,
  output path, size, and SHA-256 is sealed in the plot receipt.

## Publication/versioning decision

The expanded detailed bundle is a new immutable plot recipe and output bundle.
Existing recipe-v3 receipts and files must remain unchanged.  A cohort task
must name the new bundle explicitly; receipt validation must reject or rerun an
older recipe rather than treating its smaller output set as complete.

The expanded bundle remains `selector_eligible=false`,
`production_authority=false`, and `registry_update=false`.  Required CI must
pass before the branch is merge-ready.  Cohort rendering should first run on
one commit-pinned canary recording; only after receipt and visual review should
the same commit-pinned task fan out across the remaining eligible recordings.
