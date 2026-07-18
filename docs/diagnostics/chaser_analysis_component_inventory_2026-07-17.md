# Chaser analysis component inventory — 2026-07-17

## Decision

GoodCopBadCop should be represented as a protocol profile consumed by generic
chaser-analysis modules. It should not become its own analysis family.

The current on-disk layout largely supports that direction already. The
physical parent is the protocol-neutral
`analysis/chaser_distance_runs/<run>/`, and most derived components declare
protocol-neutral schemas. The remaining GoodCopBadCop-branded schemas and
methods are compatibility debt to migrate deliberately, not evidence that a
separate GoodCopBadCop pipeline is needed.

This document records a read-only physical/schema census. No registry row,
Zarr metadata, pointer, or analysis payload was changed.

## Scope and method

The census selected the 40 active GoodCopBadCop analysis datasets from the
canonical registry at:

`/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`

It then read the Zarr v3 `zarr.json` files below each archive's `analysis`
group. It did not open the archives for mutation and stopped traversal at array
nodes, so chunk payloads were not scanned.

The reusable inventory command is
`fisheye.utils.inventory_analysis_components`. It reports physical run-family
paths separately from declared schema and method identities. A run name that
contains a protocol name is retained as provenance but does not create a new
semantic stage.

Retained generated reports:

- `/groups/johnson/johnsonlab/jeremy/reports/analysis_inventory/goodcopbadcop_analysis_component_inventory_20260717T230638Z.json`
- `/groups/johnson/johnsonlab/jeremy/reports/analysis_inventory/goodcopbadcop_analysis_component_inventory_20260717T230638Z.md`

## Census result

The scan completed without an archive error.

| Measure | Count |
| --- | ---: |
| Analysis archives | 40 |
| Metadata groups classified | 8,325 |
| Physical run families | 10 |
| Nested component-family paths | 50 |
| Component families with declared contracts | 18 |
| Declared schemas | 32 |
| Protocol-branded schemas | 4 |
| Schemas or methods with protocol branding | 5 |

The principal physical run-family coverage was:

| Physical run family | Recordings | Complete run instances |
| --- | ---: | ---: |
| `stimulus_runs` | 40 | 40 |
| `chaser_distance_runs` | 33 | 33 |
| `eye_angle_runs` | 33 | 34 |
| `stimulus_epoch_runs` | 33 | 33 |
| `swim_bout_runs` | 33 | 65 |
| `track_kinematics_runs` | 33 | 65 |
| `detection_occupancy_runs` | 32 | 32 |
| `detection_comparison_runs` | 4 | 4 |
| `bout_kinematics_runs` | 1 | 2 |
| `subject_shape_runs` | 1 | 3 |

Twelve additional stimulus run instances report `running`. The inventory
records that physical completion state but does not decide whether those runs
are scientifically authoritative.

The one recording with `subject_shape_runs` has three complete historical run
instances. This direct observation also demonstrates why a registry stage
ledger alone is not currently sufficient for an analysis census.

## Existing protocol-neutral chaser modules

These components already use generic schemas even though the observed cohort
is GoodCopBadCop:

| Component | Recordings | Schema |
| --- | ---: | --- |
| Chaser distance | 33 | `palette.chaser_distance.v1` |
| Egocentric bearing | 33 | `palette.chaser_egocentric_bearing.v1` |
| Gaze tracking | 33 | `palette.chaser_gaze_tracking.v1` |
| Bout response | 32 | `palette.chaser_bout_response.v1` |
| Escape events | 32 | `palette.chaser_escape_events.v3` |
| Radial occupancy | 32 | `palette.chaser_radial_occupancy.v1` |
| Response regimes | 32 | `palette.chaser_response_regimes.v1` |
| Detection spatial occupancy | 32 | `palette.spatial_occupancy_zones.v1` |

This is the reusable analytical core. Each module may depend on generic source
contracts such as tracks, arena geometry, stimulus epochs, or chaser positions,
but should not depend on a recording name containing `GoodCopBadCop`.

## Protocol-branded compatibility surfaces

Only the following materialized contracts encode the protocol in a schema or
method:

| Component | Recordings | Current schema | Current method |
| --- | ---: | --- | --- |
| Stimulus epoch windows | 33 | `palette.stimulus_epoch_windows.v1` | `goodcopbadcop_chaser_epochs` |
| CRA primary endpoint | 32 | `palette.goodcopbadcop.cra_primary_endpoint.v1` | `goodcopbadcop_object_relative_pre_post_endpoint` |
| CRA near field | 32 | `palette.goodcopbadcop.cra_near_field.v1` | `goodcopbadcop_object_relative_near_field` |
| Epoch behavior summary | 32 | `palette.goodcopbadcop.epoch_behavior_summary.v1` | `goodcopbadcop_epoch_behavior_summary` |
| Escape/freeze canary | 31 | `palette.goodcopbadcop.chaser_escape_freeze_canary.v1` | `goodcopbadcop_chaser_centric_escape_freeze_canary` |

The generic stimulus-window schema is already appropriate; only its resolver
method names the protocol. That method can become a versioned profile or policy
identifier rather than a new analysis type.

The other four should gain protocol-neutral successor schemas when their
scientific contracts are next revised. Historical readers should continue to
accept the existing version-1 schemas. Existing artifacts do not need to be
renamed or rewritten merely to clean up terminology.

## Proposed modular boundary

The reusable workflow should be composed from five layers:

1. **Source resolvers.** Resolve approved tracks, arena geometry, stimulus
   events, and the canonical recording timeline through declared lineage.
2. **Stimulus-window resolver.** Convert events into named windows according to
   a versioned protocol profile. GoodCopBadCop supplies its pre/training/post
   event policy here.
3. **Chaser geometry base.** Materialize chaser identity/roles, positions,
   distances, relative bearing, and coordinate-frame provenance without
   assuming a named protocol.
4. **Independent analysis modules.** Run bout response, gaze, occupancy,
   escape events, response regimes, near-field, endpoints, and summaries based
   on declared input capabilities. Modules can be selected independently.
5. **Cohort export and reporting.** Select recordings through registry context
   and protocol metadata, then combine compatible component schemas. This is
   where a GoodCopBadCop-specific paper figure or cohort comparison belongs;
   it is not a per-recording storage stage.

A GoodCopBadCop profile should therefore contain configuration such as:

- stimulus event-to-phase mapping;
- chaser role labels and active/reference object policy;
- pre, training, post, and trial-window definitions;
- virtual or rotated control-reference policy;
- metric thresholds and their versioned parameter profile;
- expected arena/stimulus capabilities.

The generic runner should validate those capabilities and fail closed when a
required input is absent. Cohort selection should use registry protocol context
or a manifest, never infer the scientific contract solely from a path or run
name.

## Follow-up, without mutating historical data

1. Define a versioned `ChaserProtocolProfile` contract and express the current
   GoodCopBadCop policy as one profile instance.
2. Separate the GoodCopBadCop cohort resolver from the generic module runners.
3. Introduce protocol-neutral successor schemas for the four branded component
   contracts when their next semantic versions are written.
4. Keep compatibility readers for all current schemas and retain protocol names
   in run-instance provenance where useful.
5. Make the inventory command usable for RedScare and future chaser protocols,
   then compare capability coverage by schema rather than by hand-maintained
   analytics-stage names.
