# Derived-Analytics Storage Reconciliation

Date: 2026-08-01

> Historical snapshot: the eye-angle row below records the v6 state observed
> on 2026-08-01. The maintained contract was superseded by exact compact run
> schema v7 on 2026-08-03; this dated evidence is intentionally not rewritten.

Status: executable contract catalog and serialized track/eye registry
publication reconciled; remaining families are intentionally not activated

## Scope

This checkpoint covers maintained deterministic, array-bearing analysis runs:

- track kinematics;
- swim bouts;
- bout kinematics;
- eye angles;
- subject shape;
- tail kinematics; and
- stimulus response.

It does not activate selectors or change scientific values. It establishes one
executable inventory of live writer constants, verifies publication ownership,
corrects stale documentation, and records the remaining physical-policy work.

## Findings

Every family in scope has a scientific schema, exact source-lineage contract,
stage-specific logical validation, node-local materialization path, and a
production materializer using `palette.atomic_run_group_publisher` version 1.
The shared publisher owns hidden-copy publication, verification, atomic rename,
pointer updates, and rollback. Family-specific callbacks retain ownership of
scientific validation and completion semantics.

The scientific and transaction contracts are therefore substantially more
mature than the physical-policy organization. No family in this census yet
derives every array grid from the shared byte-budgeted storage planner used by
the newer detection, crop, keypoint, and subject-mask contracts.

`fisheye.shared.zarr.analysis_stage_arrays` is a partial earlier-generation
logical inventory. It currently describes stimulus-response inputs and legacy
hierarchical outputs with the diagnostic `stage_arrays.ArraySpec`; it neither
owns all seven families nor supplies `ArrayIntent` values to the byte planner.
It should be migrated or retired as exact family contracts are added, rather
than being treated as the canonical physical-policy module.

## Executable Catalog

`fisheye.analysis_workflows.storage_contract_catalog` names, for each maintained
family:

- the canonical stage and run parent;
- the module and constant names that own schema, method, and layout versions;
- the production materializer module;
- the current physical-policy owner; and
- the registry-publication mode; and
- whether the shared byte planner has been adopted.

The catalog resolves live constants from each writer rather than copying schema
numbers. Focused tests require unique stages and parents, agreement with the
registry stage catalog, and identity with the canonical atomic publisher.

## Current Physical-Policy Ownership

| Stage | Current schema | Layout/policy owner | Direct array-creation call sites | Shared byte planner |
| --- | --- | --- | ---: | --- |
| track kinematics | `analysis.track_kinematics_runs` v1 | track-specific preload grids plus materializer rechunking | 67 | no |
| swim bouts | `palette.swim_bout_runs` v8 | one regular detector trace plus shared columnar tables | 1, plus shared-columnar calls | no |
| bout kinematics | `analysis.bout_kinematics_runs` v7 | shared columnar tables | 0; shared-columnar only | no |
| eye angles | `analysis.eye_angle_runs` v6 / output v9 | semantic dense-v2 stage policy | 5 | no |
| subject shape | `analysis.subject_shape_runs` v4 | subject-shape helper plus materializer sharding | 2 helper definitions and 68 helper call sites | no |
| tail kinematics | `analysis.tail_kinematics_runs` v2 | process-owned stage shards | 2 helper definitions and 9 helper call sites | no |
| stimulus response | `palette.stimulus_response` v2 | compact shared-columnar tables; hierarchical compatibility writer remains direct | 28 compatibility/direct sites plus shared-columnar tables | no |

Call-site counts are a drift indicator, not array counts: helper calls may emit
multiple field arrays and conditional paths may be mutually exclusive.

## Registry And Selection Boundary

The executable catalog now owns derived workflow-availability parents, including
the scoped offline selection parent for track kinematics. The shared
registry/status mapping includes the canonical root parent for every in-scope
family. Track-kinematics and eye-angle runtime publication additionally use a
serialized registry finalizer so parallel analysis workers never contend for
the SQLite writer lock. The other families have paths registered for discovery
but do not gain new completion projection or activation behavior here.

Publication activation remains separate from storage reconciliation. Merging
this catalog and documentation must not change production pointers, eligibility,
or selector policy.

## Implementation Checklist

- [x] Inventory maintained derived-analysis schemas and production materializers.
- [x] Confirm every in-scope production materializer uses the shared atomic
      publisher.
- [x] Add an executable writer-backed schema/publication catalog.
- [x] Correct eye-angle, subject-shape, and tail-kinematics versions in the
      canonical matrix.
- [x] Reconcile the registry-publication agent's clean commit.
- [x] Replace the duplicated workflow-availability declarations with
      catalog-owned derived run-parent mappings, while preserving the scoped
      offline track-selection parent.
- [ ] Add completion projection for the remaining maintained derived families,
      with family-specific activation gates.
- [ ] Inventory exact per-array shapes, dtypes, access classes, chunks, shards,
      and codecs before migrating physical policy.
- [ ] Migrate fixed-row physical policies to the shared byte planner one family
      at a time, preserving logical schemas and validating decoded parity.
- [ ] Benchmark each migrated family through its real consumer workload before
      changing its production physical profile.

## Recommended Migration Order

1. Registry/status path consolidation, because it changes discovery rather than
   numerical or physical data.
2. Eye angles, whose compact dense tables already have explicit semantic access
   classes and a bounded number of creation sites.
3. Tail kinematics, whose row-aligned dense arrays and process-owned shards map
   cleanly to byte-budget planning.
4. Subject shape, after separating small semantic arrays from large row-aligned
   geometry.
5. Track kinematics, whose many per-track arrays and compatibility mirrors need
   the largest ArraySpec census.
6. Shared columnar families after the central columnar writer itself becomes
   byte-budgeted; changing each caller independently would recreate drift.
