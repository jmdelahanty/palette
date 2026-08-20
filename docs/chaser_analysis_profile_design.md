# Generic chaser analysis profiles

## Decision

Palette models GoodCopBadCop as a protocol adapter over generic chaser
analyses. Protocol names may identify acquisition metadata or a registry
cohort, but they must not define reusable analysis schema families.

The reusable stack has five layers:

1. Generic source resolution identifies canonical chaser instances and their
   source tracks.
2. A versioned protocol adapter translates acquisition metadata into
   canonical chaser roles, stimulus events, and window-policy inputs.
3. A configurable stimulus-window policy derives reusable epoch windows.
4. A versioned chaser-analysis profile catalogs independent generic analysis
   modules, dependencies, cardinality, and defaults. The maintained runner
   resolves that graph and records the exact selection beside the normalized
   profile snapshot.
5. Cohort-specific exports and figures consume the generic persisted schemas.

GoodCopBadCop remains an appropriate name for its source adapter and cohort
reports. It is not an analysis family.

The accepted provider-aware migration, heading/body-frame dependency, event-
unit, visualization, and default full-profile execution checklist is
[`composable_chaser_analytics_implementation_checklist_2026-08-20.md`](composable_chaser_analytics_implementation_checklist_2026-08-20.md).

## Canonical chaser identity and roles

Identity and experimental role are separate contracts.

- `stimulus_instance_id` is the stable identity within a recording.
- `chaser_index` connects that identity to persisted chaser geometry.
- Role assignments are long-form intervals with inclusive start and end
  frames.
- Multiple chasers may share one role.
- A role may change over time without changing stimulus identity.
- Analyses must support one or more chasers and must not encode dedicated
  `chaser_0` or `chaser_1` columns.

New chaser-distance runs persist this split directly. `chasers/` stores stable
identity, track linkage, display color, and a whole-recording role summary;
`chaser_role_intervals/` is the authoritative variable-length role table. A
consumer may use the summary only after verifying that the authoritative table
contains one stable whole-recording interval. The v1 quadrant and near-field
modules perform that check and fail closed on time-varying roles; a later
schema can summarize intervals without changing stimulus identity.

The acquisition vocabulary is currently `unknown`, `aggressive`,
`random_non_chasing`, and `inert`. Protocol adapters map source fields into
that vocabulary and record their adapter version.

## Versioned profiles

`chaser_event_windows_v1.yaml` is the generic default protocol adapter profile.
It records:

- the metadata adapter and role-resolver versions;
- event aliases and fallbacks;
- the inclusive-start/exclusive-end epoch policy; and
- protocol parameters needed by generic modules.

`chaser_behavior_v1.yaml` is protocol-neutral. It identifies generic modules,
their schema contracts, dependencies, execution cardinality, and defaults.
The runner snapshots both normalized profiles into provenance.

`chaser_behavior_full_v2.yaml` selects the complete maintained generic catalog,
including bout response, escape events, radial occupancy, and response regimes.
The `goodcopbadcop_v2` runner preset selects this full profile and supplies
versioned output names; it does not invoke protocol-branded analysis modules.

`chaser_behavior_full_v3.yaml` is the provider-aware full-profile declaration.
It adds an explicit `full` scope, provider-selection policy IDs, plot-recipe
policy, module requirement classes, and controlled input capabilities while
keeping every mature module planned by default. Its policies require exact
provider bindings; the profile does not name a detection, keypoint, or mask run
and does not promote a provider. A digest-bound applicability plan must
distinguish `not_applicable` protocol features from missing, invalid, stale, or
review-pending inputs. Until that per-recording plan and the provider-aware
relative-frame consumers are wired into the maintained runner, v3 is a
validated declaration rather than the production script default.

The first v3 runtime foundation is intentionally selector-ineligible. Its pure
computation, typed base/body schemas, bounded preparation record, and atomic
publisher are implemented in:

- `fisheye.analysis_workflows.chaser_relative_frame`;
- `fisheye.shared.zarr.chaser_relative_frame_schema`;
- `fisheye.analysis_workflows.chaser_relative_frame_storage`; and
- `fisheye.analysis_workflows.materializers.chaser_relative_frame`.

Its read boundary is also explicit:

- `fisheye.analysis_workflows.provider_chaser_stimulus_source_handle` validates
  the existing native stimulus-sample candidate without replacing its writer
  or collapsing duplicate acquisition-frame mappings;
- `fisheye.analysis_workflows.chaser_relative_frame_source_handle` validates
  an exact immutable relative-frame publication and its consolidated metadata;
  and
- `fisheye.analysis_workflows.chaser_relative_distance_view` exposes a
  provider-neutral distance relation while retaining the full chaser axis.

These surfaces consume an already projected acquisition-frame chaser track.
They do not decide how multiple native stimulus samples map onto one camera
frame when the persisted rates differ. Both rates must come from their exact
metadata authorities. The scientific projection policy remains an explicit
gate, not an implementation default.

`--enable-chaser-module` adds a module and its dependency closure.
`--disable-chaser-module` is fail-closed: selection is rejected if any selected
module needs the disabled module. Unknown modules and dependency cycles are
also rejected. Every selected module must match an exact maintained runner
adapter; a profile cannot silently select an implementation the runner does
not execute. The run directory records the normalized profiles, their digests,
explicit overrides, and resolved execution order in `profiles.json`;
`declared_chaser_modules.txt` and `enabled_chaser_modules.txt` are the
corresponding line-oriented catalog and selection receipts.

Legacy `--skip-*` flags remain execution/reuse overrides. They do not rewrite
the profile or silently discard dependency declarations. This preserves old
submission behavior while making new workflow composition declarative.

`goodcopbadcop_source_v1.yaml` remains loadable as an immutable compatibility
profile for runs that already named that adapter. New generic chaser releases
do not use that protocol name as their default.

## Generic schemas

The remaining protocol-branded schemas are replaced by new immutable schema
families:

- `palette.chaser.quadrant_occupancy.v1`
- `palette.chaser.near_field_occupancy.v1`
- `palette.chaser.epoch_behavior_summary.v1`
- `palette.chaser.escape_freeze_summary.v1`

Quadrant occupancy stores chaser-relative occupancy for every chaser. Near
field occupancy stores distance, dwell, entry, radial-density, and CDF metrics
for every chaser. Epoch behavior stores fish, bout, and per-chaser summaries
for every configured window. Escape/freeze is a per-chaser module whose
profile execution cardinality defaults to all applicable chasers.

Pairwise aggressive-versus-inert contrasts are cohort statistics or figure
choices. They are not required fields in the recording-level schema.

## Migration and compatibility

Existing GoodCopBadCop-branded runs remain immutable historical artifacts.
The production migration is a new analysis generation followed by new
exports; old Parquet tables do not need to be accepted by the new viewer.
Thin Python import shims may remain temporarily so repository callers can be
migrated without changing historical Zarr data.

No registry or archive mutation is part of this contract change. A single
recording canary must validate the new schemas before cohort execution.
