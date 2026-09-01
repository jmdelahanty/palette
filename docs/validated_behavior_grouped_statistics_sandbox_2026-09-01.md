# Validated-behavior grouped statistics sandbox — 2026-09-01

## Status

This is a sandbox implementation over the immutable Phase-B
GoodBatBadBat cohort export. It includes the grouped-statistics computation,
strict readers, normalized renderer payloads, an atomic static-report writer,
and a standalone modular Marimo explorer. It is read-only with respect to the
source export, selector-ineligible, non-authoritative, and exploratory. It does
not update a registry or selector.

Worktree:

```text
/tmp/palette-validated-behavior-grouped-statistics-main-20260901
```

Real sandbox result:

```text
/tmp/goodbatbadbat-grouped-statistics-sandbox-v003
```

Preferred reviewed static report:

```text
/tmp/goodbatbadbat-grouped-statistics-report-v009
```

## Boundary

The implementation consumes only the receipt-validated, manifest-selected
Parquet parts exposed by `ValidatedBehaviorExportDataset`. It never discovers
`latest`, globs generation contents, opens recording Zarrs, or recomputes a
scientific successor.

Every metric specification declares:

- the exact source table and value column;
- the measurement unit and interpretation;
- its condition axis and exact allowed conditions;
- its cohort grouping dimensions;
- whether recording-level values are retained;
- its paired contrast set and multiplicity family, if applicable; and
- `recording_id` as the experimental unit with equal recording weight.

The ordinary reducer is `unique_exact_row`. If dropping a dimension would
silently produce multiple rows for one recording/condition/group, the
computation fails instead of averaging them. The additional
`terminal_at_max_order_v1` reducer is restricted to already-persisted
cumulative measurements. It first requires constant declared source identity
and a unique gapless order axis, then selects exactly one maximum-order row.
Null and nonfinite metric values remain explicit exclusions.

## Implemented families

The current registry contains 45 scalar metrics plus two recording-histogram
products:

| Family | Metrics | Cohort use |
|---|---:|---|
| Core behavior | 8 | Paired pre/training/post speed, bout, heading, IBI, and dropout summaries |
| Distance traveled | 4 | Whole-session terminal cumulative path plus exact-epoch path, speed, and dropout summaries |
| Near field | 8 | Distance, near occupancy, entry rate, dwell, geometric enrichment, and coverage by provider/role |
| Same quadrant | 1 | Same-quadrant fraction by provider/role/epoch |
| Occupancy support | 2 | Provider/epoch tracking and in-arena coverage |
| Bout response by distance | 4 | Cohort curves for bout rate and kinematics by distance |
| Body alignment by distance | 3 | Alignment, absolute bearing, and circular concentration curves |
| Trial response | 9 | Trigger distance, escape/freeze, separation, recapture, and response fractions by trial ordinal |
| Spatial occupancy | 2 | Equal-recording-weight cohort heatmap surfaces for both persisted denominators |
| Radial distribution | 3 | Radial fractions and geometric selection index |
| Distance CDF | 1 | Cohort distance CDF by provider, role, and epoch |
| Signed body bearing | 1 histogram | Equal-recording 10-degree signed anatomical-bearing distributions |
| Body bearing by distance | 1 histogram | Equal-recording 5-mm by 30-degree joint bearing--distance distributions |

The predeclared epoch contrasts are training minus pre, post minus pre, and
post minus training. They use recording-level bootstrap intervals and paired
sign-flip tests. Benjamini-Hochberg adjustment is applied within each declared
metric family across its registered contrasts/groups.

All results remain exploratory. Authoritative acquisition-batch identity is
missing, so no batch adjustment is performed and no row may be labeled
confirmatory.

## Shared visualization boundary

`ValidatedBehaviorStatisticsViewSource` first reopens the exact statistics
generation through its strict reader. `build_statistics_view_payload` then
decodes group-key JSON into ordinary semantic dimensions and produces one
self-digested, backend-neutral payload. Static Matplotlib and interactive
Plotly renderers consume that same payload; neither renderer queries Parquet or
recomputes statistics independently.

The payload carries the exact statistics-manifest and source-export digests,
metric identities, units, condition order, role/provider display semantics,
recording/descriptive/contrast rows, experimental-unit policy, and exploratory
status. Plotly figures embed these identities in `layout.meta`; the atomic
static-report manifest binds every payload digest, image digest, renderer-code
digest, DPI, and occupancy color-scale parameter.

Display rules discovered against the real cohort and now covered explicitly:

- the terminal generalized-bout distance interval is open-ended and is shown
  from its persisted bin index as `50–∞`, never assigned a fabricated midpoint;
- spatial boundary coordinates may have separate arena-member and
  nonmember recording strata because reviewed arena geometry varies by
  recording; occupancy heatmaps display the arena-member stratum and expose
  finite-recording support rather than averaging the strata together;
- condition colors encode exact pre/training/post roles;
- aggressive/inert colors encode semantic behavior roles, never raw stimulus
  dot colors; and
- static signed-bearing polar axes explicitly use the persisted `[-180, 180]`
  domain; leaving Matplotlib's default `[0, 360]` domain in place would
  autoscale signed samples to a misleading 540-degree, half-circle display;
  and
- detection/keypoint provider identity remains explicit and is additionally
  distinguished by line style where both occur in one panel.

The standalone explorer is:

```text
apps/marimo/validated_behavior_group_statistics_explorer.py
```

It builds one selected view payload lazily, provides metric/provider/role/epoch
controls as appropriate, renders the Plotly view, and exposes exact descriptive
rows, paired contrasts, and provenance in expandable tables.

## Signed-bearing histogram boundary

The signed anatomical-bearing extension deliberately separates source
authority from a reusable statistical reduction:

1. The recording Zarr remains authoritative for exact, unbinned body-relative
   samples: signed anatomical bearing, physical fish--chaser distance,
   validity, epoch membership, chaser occurrence, identity, and semantic
   behavior role.
2. Phase-B Parquet is the lossless tabular projection of those sealed samples.
3. The grouped-statistics generation persists one recording-level histogram
   row per exact epoch/role/bin, including raw bin count, exact denominator,
   normalized fraction, bin edges, recipe identity, and source-query identity.
4. Cohort summaries average the recording-level fractions with equal recording
   weight. They never pool source frames across recordings.
5. Static and interactive viewers consume only the persisted histogram and
   cohort-summary rows. They may change layout or color scaling, but they may
   not re-bin source samples or recompute denominators.

This preserves the most general composable interface before export while also
making the chosen 10-degree signed-bearing marginal and 5-mm by 30-degree
bearing--distance surface reproducible and inexpensive to reuse. A future
recording-Zarr histogram cache or successor is explicitly deferred. If added,
it must be a digest-bound derived product and must not replace the exact
unbinned samples as scientific authority. The current cohort extension does
not depend on that future cache and does not require recording-level successor
republication.

## Sandbox output contract

One atomic output directory contains:

- `recording_metric_values.parquet` for retained recording-level values and
  paired-line diagnostics;
- `descriptive_statistics.parquet` for equal-recording-weight cohort summaries;
- `paired_contrasts.parquet` for exploratory paired effects and uncertainty;
- `recording_histogram_bins.parquet` for exact recording/epoch/semantic-role
  bin counts, denominators, fractions, source identity, and resolved edges;
- `histogram_descriptive_statistics.parquet` for equal-recording cohort
  summaries of those persisted fractions; and
- `manifest.json` binding the source export/plan/analysis-unit policy, metric
  and histogram specifications, resolved recipes, queries, output schemas,
  file digests, and safety flags.

The strict reader validates the manifest self-digest, exact file inventory,
file digests and sizes, Arrow schemas, primary keys, group-key digests,
denominator accounting, probability bounds, and source/statistics identities.

## Real cohort canary

Source export manifest:

```text
230c30e032352c95bb9919f5704da6eba9d94a369b464089f575048639791d05
```

Initial scalar-only v1 result manifest record:

```text
7e75cf51c3393954c720c35517353db489060a8fbe0946fcfb2f5f743065cbd4
```

Measured results:

| Surface | Rows |
|---|---:|
| Recording metric values | 14,391 |
| Descriptive statistics | 23,784 |
| Paired contrasts | 144 |

All 41 metric grains were unambiguous on the real cohort. Of 144 contrasts,
141 computed. Three complete-visit median-dwell contrasts were correctly
skipped because only two recordings had finite values in both conditions for
the affected inert/provider comparison. This is missing scientific support,
not a fabricated zero or a pipeline failure.

The superseding v2 result adds the persisted signed-bearing histograms. Its
manifest record SHA-256 is:

```text
437b4ce1559ae5762f2ed0d3fb5b378d7f74f3e61a52da9c1aec6e0b822627ea
```

| Histogram surface | Recording-bin rows | Cohort-bin rows | Contributors |
|---|---:|---:|---:|
| Signed body bearing, 10 degrees | 17,280 | 216 | 80 |
| Signed bearing by distance, 30 degrees by 5 mm | 92,160 | 1,152 | 80 |

The exact source audit found 23,945,312 body-relative sample rows. Of these,
23,945,152 carry an exact pre/training/post condition and 22,609,044 meet the
declared bearing/joint validity intersection. The 160 null-condition rows
remain nonmember evidence; 152 of them otherwise meet membership and validity
requirements. They are not assigned to an epoch. The cohort-wide jointly valid
distance maximum resolves the zero-anchored 5-mm recipe through 80 mm, covering
every selected row. Each recording/epoch/semantic-role stratum maps to one
exact chaser identity.

The four other parent cohort members remain explicit noncontributors rather
than disappearing from the denominator. Every one of the 80 contributing
recordings has a finite denominator in every persisted histogram panel.

The prior reviewed `v008` report contains twelve PNG figures plus an HTML
index. Its manifest record SHA-256 is:

```text
43344d568a48488c8deecac20437aa82c93722e6f8a6da869b089c9b9dffa14d
```

Earlier `v001` through `v007` report directories are retained as immutable
development evidence. `v008` supersedes them for visual review because it uses
the v2 statistics receipt, adds the signed-bearing polar and joint-distance
surfaces, binds the final formatted renderer sources, and constrains the static
signed-bearing display to exactly one full circle. The `v006` static bearing
figures exposed Matplotlib's erroneous `[-180, 360]` autoscaled domain; `v007`
proved the domain correction, and `v008` additionally removes colliding polar
axis labels while preserving their units in the figure subtitles.

The superseding `v003` statistics generation adds the four first-class
distance-traveled metrics without recomputing the Phase-B export. It contains
45 scalar specifications, 15,191 retained recording values, 23,794
descriptive rows, and 153 paired contrasts. Its manifest record SHA-256 is:

```text
af3ea7310138390663c4bf65b52f02ba6f1c7ccaa7575e81081edb264b477478
```

The superseding `v009` report contains thirteen PNG figures plus its HTML
index. Its manifest record SHA-256 is:

```text
58633f5a8f001507756363396825d0d747877a1b4f1721a137eaf8039ffd14bd
```

The distance family has 800 retained values, ten descriptive strata, and nine
paired contrasts across the 80 complete recordings. Its whole-session
cumulative-path reducer selects the exact terminal provider-motion row only
after checking constant bundle/provider/run/track lineage and a unique gapless
`track_sample_row_id` axis. See
`docs/cumulative_path_distance_composition_implementation_2026-09-01.md`.

## Command

```bash
scripts/py -m fisheye.utils.compute_validated_behavior_group_statistics \
  --export-root /groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_phase_b_20260901_b45aa6a5/publication \
  --source-export-run-id goodbatbadbat-validated-behavior-phase-b-20260901-b45aa6a5 \
  --statistics-run-id goodbatbadbat-grouped-statistics-sandbox-v003 \
  --bootstrap-iterations 5000 \
  --permutation-iterations 5000 \
  --random-seed 20260901 \
  --output-dir /tmp/goodbatbadbat-grouped-statistics-sandbox-v003 \
  --apply
```

Render the exact statistics generation without recomputation:

```bash
scripts/py -m fisheye.utils.render_validated_behavior_group_statistics \
  --statistics-dir /tmp/goodbatbadbat-grouped-statistics-sandbox-v003 \
  --report-run-id goodbatbadbat-grouped-statistics-report-v009 \
  --output-dir /tmp/goodbatbadbat-grouped-statistics-report-v009 \
  --apply
```

Run the read-only Marimo explorer:

```bash
scripts/py -m marimo run \
  apps/marimo/validated_behavior_group_statistics_explorer.py -- \
  --statistics-dir /tmp/goodbatbadbat-grouped-statistics-sandbox-v003
```

## Validation completed in the sandbox worktree

- 62 focused source-routing, grouped-statistics, reducer, payload, report,
  static-renderer, and interactive-renderer tests pass outside the Codex
  sandbox; 15 predeclared explorer compatibility cases remain expected
  failures.
- Real-data payload and Plotly generation succeeds for all thirteen view
  families, including six-panel signed-bearing and bearing--distance figures.
- All twelve static signed-bearing axes are regression-checked at exactly
  `[-180, 180]`; visual review confirms that both anatomical sides occupy one
  full circle in the polar and bearing--distance surfaces.
- The preferred real static report strictly reopens with all thirteen artifact
  sizes and SHA-256 digests intact.
- The Marimo reactive-graph checker passes.
- Python compilation and `git diff --check` pass.

This evidence is focused local validation, not repository-required CI.

## Before integration

- Review the 45-metric exploratory registry and multiplicity-family boundaries.
- Decide whether the statistics generation should use the generic derived
  publication lifecycle or a dedicated versioned statistics profile.
- Decide whether to surface this standalone component inside the existing
  general group explorer in addition to keeping the modular app.
- Run the required broader tests and CI before merge or deployment.
- Do not promote this `/tmp` result or describe it as confirmatory.
