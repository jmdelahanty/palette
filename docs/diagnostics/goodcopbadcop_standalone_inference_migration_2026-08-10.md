# GoodCopBadCop standalone-inference guard and migration inventory

Date: 2026-08-10

Status: implementation checkpoint; scientific migrations remain open

Scope: historical `analyze_goodcopbadcop_*` scripts plus the two inferential
`plot_goodcopbadcop_*` scripts

## Decision

Standalone GoodCopBadCop scripts are exploratory inspection surfaces. They are not an
alternative publication path for confirmatory statistics. Their p-values and p-based
annotations are outside the versioned analytics-export schema, registered metric catalog,
session-clustered inference, and family-level FDR contract.

Every inventoried entry point now fails before reading the registry unless the operator
passes `--exploratory-only`. An acknowledged run:

- prints a strict-JSON `PALETTE_STANDALONE_ANALYSIS_STATUS` receipt to stderr;
- identifies itself as exploratory, publication-ineligible, and without multiplicity
  control;
- writes figures with an `EXPLORATORY ONLY` watermark and `_exploratory` filename;
- writes a strict-JSON `.exploratory.json` receipt beside every figure and the per-fish
  CSV; and
- leaves all historical scientific calculations unchanged.

The acknowledgement is intentionally inconvenient. It prevents an old command in a
notebook, shell history, or methods document from silently producing a
confirmatory-looking artifact.

Example:

```bash
scripts/py -m fisheye.analysis.analyze_goodcopbadcop_escape --exploratory-only
```

## Census and migration map

“Candidate destination” describes the closest maintained immutable surface. It does not
assert scientific equivalence. “Open” means a metric definition, experimental unit,
contrast, clustering rule, or multiplicity family still requires an explicit scientific
decision before migration.

| Standalone entry point | Current standalone inference or annotation | Candidate maintained destination | Migration state |
|---|---|---|---|
| `analyze_goodcopbadcop_approach_avoidance` | Mid-band direction, approach, and minimum-distance Wilcoxon/sign-flip tests | `cra_near_field` approach and occupancy families | Open: bands and estimands are not exact matches |
| `analyze_goodcopbadcop_bout_kinematics_distance` | Distance-binned bout kinematics and near/far difference-in-difference | `epoch_behavior` bout summaries | Open: registered metrics are not distance-stratified |
| `analyze_goodcopbadcop_bout_vigor_prepost` | Pre/post near-minus-far bout-vigor gradients | `epoch_behavior` bout summaries | Open: gradient and object-specific contrast are absent |
| `analyze_goodcopbadcop_escape` | Escape-rate pre/chase paired tests | New registered escape-event family | Open: no exact registered metric |
| `analyze_goodcopbadcop_habituation` | Early/late escape, freeze, and latency tests | New registered trial-ordinal/habituation family | Open: trial-level estimands are absent |
| `analyze_goodcopbadcop_immobility_artifact` | Raw/smoothed immobility threshold sweep | Diagnostic-only quality-control export | Keep exploratory: this is an artifact investigation, not a primary endpoint |
| `analyze_goodcopbadcop_lateral_gaze` | Object/virtual lateral-orientation tests by distance | `egocentric_alignment` | Partial: registered orientation metrics exist; distance-specific contrasts need review |
| `analyze_goodcopbadcop_learning_mixed_model` | Standalone MixedLM for occupancy and freeze indices | `cra_primary_endpoint` plus registered session-cluster inference | Partial: occupancy candidate exists; freeze-learning metric is absent; remove the script-local fallback before any migration |
| `analyze_goodcopbadcop_per_fish` | Per-fish scorecard and unadjusted correlations | Immutable per-recording analytics exports; group inference only in registered statistics | Partial: descriptive rows may migrate; correlations require declared families |
| `analyze_goodcopbadcop_radial_kinematics` | Per-bin object/virtual radial and steering tests | New distance-resolved chaser-response export family | Open: no exact registered metric |
| `analyze_goodcopbadcop_radial_turn_direction` | Cluster-bootstrap distance-bin annotations and shell Wilcoxon tests | New visit-clustered turn-direction family | Open: cluster unit and family boundaries need review |
| `analyze_goodcopbadcop_wall_mediator` | Wall concentration, trial correlations, and partial-correlation tests | `epoch_behavior` wall metrics plus a separately declared mediation analysis | Partial: descriptive wall metrics exist; mediation estimand is absent |
| `plot_goodcopbadcop_bout_rate` | Cohort pre/chase and pre/post p-value annotations | `exploratory|epoch_behavior`, `bout_rate_per_min` | Direct candidate after immutable source/contrast parity is demonstrated |
| `plot_goodcopbadcop_freeze` | Paired immobility p-value annotations | Diagnostic/null-result family, if retained | Keep exploratory until an explicit smoothed-immobility metric is registered |

## Publication boundary

The sidecar is an exclusion receipt, not a scientific result manifest. Its exact contract
is `palette.goodcopbadcop.standalone_exploratory_status`, version 1. It declares:

- `analysis_tier = exploratory`;
- `publication_eligibility = ineligible`;
- `confirmatory_use = false`;
- `multiplicity_control = none`; and
- `registered_group_statistics = false`.

No registry selector, analytics-export manifest, group-statistics manifest, or production
publication API accepts this receipt. Removing the warning, renaming an artifact, or
copying its p-value into another document does not promote it.

## Remaining scientific work

1. Decide which distance-resolved, trial-ordinal, escape, immobility, and mediation
   estimands remain scientifically useful.
2. Define their experimental units, session grouping, minimum-session rules, contrasts,
   and primary/exploratory families before adding them to the registry.
3. Add immutable producer tables and exact Arrow contracts for accepted metrics.
4. Reproduce accepted descriptive values from immutable exports and then run only the
   registered group-statistics implementation.
5. Retire each standalone inferential calculation after parity evidence exists; retain
   descriptive plotting only when it consumes accepted statistics rather than
   recomputing p-values.

Until those steps are complete, these scripts are suitable for exploratory diagnosis and
method development only.
