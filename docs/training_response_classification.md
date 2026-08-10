# Whole-Training Chaser Response Classification

Status: identity-aware downstream analytics v3 implemented; production rollout pending integration

## Purpose

This workflow describes how each fish behaves during the complete chaser
training period, using its own pre period and the normalized `aggressive` and
`inert` chaser roles as references. It complements the pre-only baseline
strategy analysis; it does not replace it or mix baseline traits with stimulus
responses in one label.

The primary result is factorized and descriptive. A fish may independently be
more active, more boundary-biased, relatively far from the aggressive chaser,
and similarly distant from both chaser roles. The workflow does not infer fear,
anxiety, causal avoidance, or escape success.

Implementation:

```text
src/fisheye/training_response/
  contracts.py
  features.py
  cohort.py
  workflow.py
  query.py
  validation.py
```

## Source and cohort contract

The workflow consumes four immutable export tables:

```text
chaser_epoch_behavior_summary
chaser_epoch_distance_summary
chaser_egocentric_epoch_summary
chaser_speed_distance_bins
```

Every source export is validated before analysis. Output is published beneath
a separate root and every output row carries `source_export_run_id`, the
registry-derived `session_id`, and the effective registry `subject_id`. These
identities are required non-null exact-schema columns in the feature,
classification, and cluster tables. The source manifest's validated registry
identity receipt is copied opaquely into the derived manifest. Training-response
does not duplicate the export layer's versioned receipt semantics. Publication
validation instead proves that the copied receipt exactly matches the
digest-bound source manifest and that every output identity tuple occurs in
the source export's digest-declared Parquet parts.
The source manifest SHA-256 and frozen collection-manifest SHA-256 are also
recorded, and the source Parquet files are never modified.

The receipt is intentionally opaque here so analytics-export receipt v1 and
the forthcoming digest-bound experimental-session receipt v2 remain owned by
the export contract. The workflow first runs the installed source-export
validator. During reconciliation, the shared receipt validator can be called
at that boundary without changing the training-response v3 tables.

The recording/session/subject tuple is preserved while grouping source tables,
building features, assigning cohort-relative classifications, discovering
clusters, and joining the three output tables for QC. One `recording_id` may
not acquire conflicting session or subject bindings across source tables.

The combined cohort is defined by the registry's normalized `CHASER` stimulus
mode rather than a protocol-name prefix. Protocol remains a diagnostic stratum.
Role comparisons use normalized role metadata, so aggressive-versus-inert
meaning does not depend on whether a recording assigned aggression to chaser
index 0 or 1. Future experiments may contain more chasers; v1 requires one
normalized aggressive and one normalized inert summary while preserving
physical chaser indices in the source export.

## Feature families

### Pre-to-training locomotor response

Training/pre log2 ratios are calculated for mean and p95 speed, path per
minute, bout rate, mean bout path length, and mean absolute bout heading
change. Ratios compare each fish with itself and duration-normalize path before
comparison.

### Pre-to-training boundary response

The feature table retains pre and training wall fractions, normalized median
distance from arena center, and their training-minus-pre changes. This remains
a spatial response measure, not an anxiety label.

### Aggressive-chaser proximity

Raw training p05 and p50 distance and fraction within the declared near-field
threshold are retained in millimetres/fractions. The cohort-relative axis uses
clear vocabulary:

```text
closer_than_cohort
cohort_typical_proximity
farther_than_cohort
```

These states describe position within the valid source cohort during training.
They are not exposure doses and do not by themselves establish active
avoidance.

### Role-distance selectivity

Aggressive-minus-inert p05 distance, p50 distance, and near-field fraction are
computed within each recording. This axis is separate from absolute aggressive
proximity:

```text
farther_from_inert
similar_role_proximity
farther_from_aggressive
```

### Close-contact vigor and orientation

Distance-binned speed sums and counts provide sample-weighted speed within and
outside the aggressive chaser's declared near threshold. Their difference is
retained as `aggressive_near_minus_far_speed_mm_s`. Egocentric alignment,
front/behind fractions, and circular concentration are retained as continuous
pre, training, and change features; they are not converted into a causal
success label.

## Quality gates and temporal limitation

The default minimum valid-position fraction is 0.75 in both the pre and
training periods, and the minimum training duration is 30 seconds. Missing
pre/training role summaries also invalidate classification. Invalid recordings
remain in every output table with explicit reasons; they are not silently
dropped from the cohort inventory.

The current all-chaser export contains whole-epoch summaries and distributions,
but no training-period time bins or portable samples. Therefore v1 explicitly
sets temporal training features unavailable and does not classify onset,
adaptation, habituation, or within-training transitions. Those capabilities
require a future export contract rather than reconstruction from summary rows.

## Cohort-relative classification

Metrics are robustly standardized over complete rows using median and MAD,
with IQR and standard-deviation fallbacks. Weighted standardized metrics form
five independent axes:

```text
locomotor_response
boundary_response
aggressive_proximity
role_distance_selectivity
close_contact_vigor
```

The default descriptive threshold is ±0.75 robust units. It is serialized in
the immutable analysis manifest and can be changed only by publishing a new
analysis run. The optional `primary_training_profile` is a display summary;
continuous physical metrics and factor scores remain authoritative.

The `profile_separation_score` describes distance beyond a rule threshold. It
is not a calibrated probability. Optional Gaussian-mixture clusters are
exploratory, use BIC model selection including a one-component model, require a
declared minimum cohort size per candidate component, report assignment
probability and resampling stability, and retain biologically unnamed numeric
cluster IDs. A multi-component solution below the declared median-ARI
stability threshold is published as `unstable_model`, not as a validated set
of behavioral types.

## Output schema, compatibility, and lazy querying

Each validated recording/session/subject binding contributes one row to each
output table:

```text
training_response_features
training_response_classification
training_response_clusters
```

New output uses `palette.training_response_analytics` schema version 3 and
Arrow-contract envelope version 2. Version 3 is the first training-response
schema to preserve registry session and subject identities. Its primary key is
`analysis_run_id`, `recording_id`, `session_id`, and `subject_id`.

The shared derived-publication directory named `v2` describes the immutable
publication protocol, not this family's logical schema version. Readers inspect
the manifest's `schema_version` and fail closed unless it is version 3.

Frozen exact schema v2 remains readable only when callers explicitly pass
`allow_legacy_v2=True`; that compatibility path validates the original v2
Arrow schemas, manifest, receipts, and primary keys but cannot invent missing
session or subject identities. Historical schema-v1 layout remains a separate
explicit compatibility mode. Neither compatibility mode participates in
default catalog selection.

Manifest-declared parts can be scanned lazily with Polars:

```python
from fisheye.training_response import scan_training_response_table

responses = scan_training_response_table(
    output_root,
    analysis_run_id,
    "training_response_classification",
)
```

## Running and cluster submission

Direct execution is appropriate for local fixtures:

```bash
scripts/py -m fisheye.training_response.workflow \
  --source-export-root /groups/johnson/johnsonlab/palette_analytics \
  --source-export-run-id <immutable-export-run> \
  --output-root /groups/johnson/johnsonlab/palette_training_response_analytics \
  --analysis-run-id <new-analysis-run>
```

Shared production work must run inside an LSF allocation:

```bash
scripts/submit_training_response_analytics_bsub.sh \
  --source-export-run-id <immutable-export-run> \
  --analysis-run-id <new-analysis-run> \
  --queue short \
  --submit
```

The submitter pins the shared Palette commit, fails closed if the output run
already exists, and records job status and logs beneath the derived root.

## Read-only Marimo QC

The baseline strategy explorer also discovers matching training-response runs:

```bash
scripts/py -m marimo run apps/marimo/baseline_strategy_explorer.py -- \
  --strategy-root /groups/johnson/johnsonlab/palette_strategy_analytics \
  --training-response-root /groups/johnson/johnsonlab/palette_training_response_analytics \
  --export-root /groups/johnson/johnsonlab/palette_analytics
```

Only a training run whose `source_export_run_id`, source export manifest hash,
and frozen collection hash match the selected baseline run is selectable. The
app lazily joins the three small derived
tables, provides protocol and validity filters, shows factorized category
counts and continuous distributions, and states the temporal limitation
directly. It never writes to an export or recording Zarr.

The QC section also shows a Sankey diagram from complete pre-period
`primary_strategy` labels to complete whole-training
`primary_training_profile` labels. Links respect the active protocol and
training-validity filters. Counts are recording-level focal-fish sessions, not
deduplicated biological individuals, and the flow is a descriptive
correspondence rather than evidence of a causal state transition.
