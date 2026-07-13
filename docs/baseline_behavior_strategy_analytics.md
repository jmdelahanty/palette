# Baseline Behavior Strategy Analytics

Status: implemented downstream analytics v1

## Purpose

This module combines two compatible traditions for describing pre-stimulus
behavior:

- zebrafish measures of swim bouts, activity, wall affinity, spatial coverage,
  and within-session habituation;
- rodent open-field measures of progression versus dwelling, wall/center
  movement, dominant dwell locations, and excursions organized around a
  home-base-like location.

The result is a descriptive behavioral phenotype. It does not infer anxiety.
Wall following, inactivity, distributed exploration, and temporal adaptation
are stored as separate dimensions because one animal can express more than one
of them.

Implementation:

```text
src/fisheye/baseline_strategy/
  contracts.py
  features.py
  cohort.py
  workflow.py
```

## Source contract

The workflow consumes the immutable cross-recording export tables:

1. `baseline_behavior_summary`, required;
2. `baseline_behavior_time_bins`, optional but needed for temporal strategy;
3. `baseline_kinematic_samples`, optional but needed for corrected occupancy,
   progression episodes, wall-following geometry, and dominant-dwell analysis.

The source export is validated before analysis. The output root must be
separate from, outside, and not a parent of the source export root. No source
Parquet or recording Zarr is opened for writing.

Kinematic sample parts are processed one source recording at a time, matching
the exporter's one-part-per-recording publication policy. The full sample table
is not accumulated in memory across the cohort.

## Chaser cohort boundary

For the cross-protocol analysis, cohort eligibility is defined by the
registry's normalized latest stimulus mode `CHASER`, not by a protocol name.
This combines RedScare, GoodCopBadCop, and future chaser experiment types in
one immutable virtual collection. The behavioral features themselves use only
the pre-stimulus window; `CHASER` defines which recordings enter the cohort and
does not leak post-stimulus behavior into the baseline phenotype.

Protocol name, protocol hash, acquisition date, rig, and arena should remain
available as diagnostic strata. The primary robust scaling and strategy
classification are fit over the declared combined chaser cohort. Validation
must then check protocol cross-tabs, protocol-balanced sensitivity analyses,
and leave-one-protocol-out stability so a discovered cluster is not merely a
RedScare-versus-GoodCopBadCop or acquisition-batch split.

Build the source collection through the read-only registry query described in
`docs/virtual_collection_manifest_schema.md`, then generate a new baseline-
enabled cross-recording export from that frozen collection. Never concatenate
independently standardized protocol exports: that destroys the common feature
scale needed for a combined classification.

Operational note (2026-07-12): the shared registry schema contains the
normalized stimulus tables, but the deployed registry had zero indexed
`recording_stimulus_mode_counts` rows when audited. Existing analysis datasets
must be reconciled/backfilled on cluster compute nodes before the live
`CHASER` collection can be materialized. This is an index-population gap, not a
reason to infer cohort membership from filenames or protocol strings.

## Feature families

### Locomotor vigor

- path per minute;
- bout rate;
- active sample fraction;
- median and upper-quantile speed;
- progression-episode count, path length, displacement, and tortuosity.

An active sample is defined by a declared speed threshold. The default is an
operational starting point and is serialized in every manifest; it is not
presented as a universal biological cutoff.

### Boundary affinity

For a circular arena with radius `R` and wall-band width `w`, the expected wall
occupancy under uniform use is:

```text
p_wall_uniform = 1 - ((R - w) / R)^2
```

The feature table reports observed wall occupancy, the uniform-area
expectation, their enrichment ratio, normalized center distance, active wall
occupancy, and the fraction of progression episodes classified as
wall-following.

In current circular exports, distance to the experimental-area boundary is
`arena_radius_mm - center_distance_mm`. The denominator for `wall_fraction` is
all valid position frames; the denominator for `active_wall_fraction` is active
valid portable samples. These definitions and the boundary method are explicit
columns rather than implicit assumptions.

This boundary is the experimental arena/area boundary, never the fish subject
mask. New exports include `distance_to_arena_boundary_mm` on portable samples.
For a future non-circular experimental-area mask, the exporter should compute
an inward Euclidean distance transform of that mask and sample it at each fish
centroid. That future method must use a distinct `boundary_distance_method` and
must provide the accessible wall-band area before uniform wall enrichment is
computed; it must not silently use the circular formula.

Wall-following episodes require both boundary occupancy and tangential motion.
This prevents a fish that remains inactive beside the wall from being treated
as equivalent to a fish actively following the boundary.

### Spatial distribution and dominant dwell organization

Portable positions are placed on arena-centered grids. Boundary cells receive
a deterministic subcell estimate of their accessible circular area; a cell is
included when any of it intersects the arena. Uniform expected occupancy is
proportional to accessible cell area rather than treating a partially clipped
boundary cell like a full interior cell. These weights contribute to:

- accessible-area-normalized occupancy entropy;
- visited accessible-cell fraction;
- Jensen-Shannon divergence from uniform accessible occupancy;
- maximum cell occupancy;
- latency to half of final spatial coverage.

A second declared grid describes low-motion dwell samples. It produces the
dominant dwell cell, its concentration relative to the second cell, visit
count, radial location, excursions originating there, and return fraction.
The term `home_base_like` is deliberately descriptive. A grid concentration is
not by itself proof that fish and rodent home bases are homologous.

### Temporal adaptation

Fixed baseline time bins provide early/late values, late-minus-early changes,
and robustly interpretable whole-baseline slopes for:

- wall fraction;
- normalized center distance;
- speed;
- distance travelled;
- bout count.

Decreasing wall affinity and decreasing normalized center distance contribute
to the `temporal_expansion` axis. This supports descriptions such as
`initial_wall_bias_then_expansion` without averaging the transition away.

## Progression episodes

`baseline_exploration_episodes` is one row per active locomotor episode. It
contains:

```text
episode timing and sample count
sampled path length and net displacement
tortuosity
minimum center distance and maximum inward excursion
wall-sample fraction and tangential alignment
wall-following state
origin/destination dominant-dwell state
return-to-dominant-dwell state
```

Portable 5–10 Hz samples are adequate for coarse routes and dwell organization.
They do not replace full-resolution swim-bout or tail-kinematics data. Path
length is explicitly named `portable_sample_xy_chord_sum` so it cannot be
mistaken for the authoritative full-rate track distance.

## Cohort-relative classification

Metrics are transformed as declared, robustly standardized by median and MAD
(with IQR or standard-deviation fallback), and combined into five scores:

```text
activity
boundary
spatial_distribution
home_base
temporal_expansion
```

The factorized labels are:

```text
activity_state:
  inactive | typical_activity | active

boundary_strategy:
  boundary_neutral | mixed_boundary | wall_following

spatial_organization:
  localized | intermediate | broad_even | home_base_like

temporal_pattern:
  contracting | stable_or_mixed | expanding
```

An optional `primary_strategy` provides convenient display vocabulary:

```text
inactive_or_low_activity
active_wall_following
home_base_like_explorer
broad_even_explorer
localized_explorer
initial_wall_bias_then_expansion
mixed_or_uncertain
```

These v1 thresholds are cohort-relative. They must be frozen against a declared
reference cohort before labels are compared across export runs. The
`classification_confidence_score` is a descriptive distance from the relative
threshold, not a calibrated probability.

## Optional cluster discovery

The cluster table fits Gaussian mixtures to complete factor scores. Candidate
component counts are compared with BIC and may include one component. Selecting
one component is reported as `no_multimodal_structure`; the workflow does not
force behavioral types to exist.

For multi-component results, every row includes assignment probability and a
thresholded uncertainty state. Repeated 80% subsamples provide median adjusted
Rand index as cluster-stability evidence. Cluster IDs have no biological name
until their feature centroids, trajectories, and representative animals are
reviewed.

## Output contract

The separate derived output contains:

```text
baseline_strategy_features
baseline_exploration_episodes
baseline_strategy_classification
baseline_strategy_clusters
```

along with a `palette.baseline_strategy_analytics` version-1 manifest. The
manifest records source validation, source export identity, configuration,
row counts, output parts, and the explicit anxiety-interpretation guardrail.

Published tables remain lazily queryable with Polars:

```python
from fisheye.baseline_strategy import scan_strategy_table

features = scan_strategy_table(
    output_root,
    analysis_run_id,
    "baseline_strategy_features",
)
```

`scan_strategy_table` resolves only manifest-declared files under the authorized
root and returns a `polars.LazyFrame`, preserving projection and predicate
pushdown until collection.

## Running

```bash
scripts/py -m fisheye.baseline_strategy.workflow \
  --source-export-root /groups/johnson/johnsonlab/palette_analytics \
  --source-export-run-id <immutable-export-run> \
  --output-root /groups/johnson/johnsonlab/palette_strategy_analytics \
  --analysis-run-id <new-analysis-run>
```

Useful declared controls include:

```text
--active-speed-mm-s
--spatial-grid-size
--dwell-grid-size
--relative-score-threshold
--cluster-max-components
--cluster-stability-resamples
```

Every analysis run is immutable. A changed threshold, grid, or source export
requires a new analysis run ID.

For shared data, render or submit the CPU job instead of running analysis on an
LSF login node:

```bash
scripts/submit_baseline_strategy_analytics_bsub.sh \
  --source-export-run-id <immutable-export-run> \
  --analysis-run-id <new-analysis-run> \
  --submit
```

The submitter captures the exact shared Palette commit, runs the analysis only
inside the LSF allocation, validates the published derived manifest and all
Parquet parts, and retains the job script, logs, parsed job ID, and completion
status beneath the derived output root.

## Interpretation and validation gates

Before using semantic labels in a scientific comparison:

1. inspect feature distributions and missingness;
2. review representative trajectories and occupancy maps for every proposed
   group;
3. verify stability across acquisition date, arena, protocol, sex, strain, and
   tracking-quality strata when those metadata are available;
4. use grouped validation for repeated fish or acquisition batches;
5. freeze a reference cohort and feature policy;
6. retain continuous scores and uncertainty alongside every label;
7. do not translate wall affinity into anxiety without an independent assay.

The design draws on progression/lingering segmentation in rodent exploration
([Drai et al., 2000](https://doi.org/10.1016/S0165-0270(99)00194-6)),
home-base organization
([Eilam and Golani, 1989](https://doi.org/10.1016/S0166-4328(89)80102-0)),
replicable wall-versus-center movement measures
([Lipkind et al., 2004](https://doi.org/10.1152/japplphysiol.00148.2004)),
and zebrafish exploratory clusters that include wall-huggers and active
explorers
([Rajput et al., 2022](https://doi.org/10.1242/bio.059443)).
