# Stimulus coordinate v6 real-data gate — 2026-08-05

Status: **BLOCKED on a producer-native v6 H5 artifact**. The v6 validator and
commit-pinned cluster deployment pass, and the existing migrated canonical
stimulus publication passes its own strict contract. No archive, selector,
registry, or shared Palette checkout was changed during this gate.

## Intended gate

The intended operation was to validate one real Citrus coordinate-v6 source,
publish a selector-ineligible Palette stimulus canary, validate direct and
consolidated metadata, and hand the exact run to Crimson. Palette must not
manufacture a v6 receipt for a legacy artifact or relabel a v5 migration as a
producer-native v6 canary.

## Source census

The standard recording tree contained 162 H5 artifacts at the inspected
`<recording>/<kind>/*.h5` depth. Every artifact opened read-only; none contained
the required `/stimulus_coordinate_v6` group.

The representative Batman source pair was:

- legacy source:
  `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T19-38-32Z_arena_1_Batman/raw/2026-07-21T19-38-32Z_arena_1_Batman.h5`
- immutable migration derivative:
  `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T19-38-32Z_arena_1_Batman/derived/stimulus_coordinate_migration/2026-07-21T19-38-32Z_arena_1_Batman.canonical_stimulus_v1.h5`
- derivative receipt:
  `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T19-38-32Z_arena_1_Batman/derived/stimulus_coordinate_migration/2026-07-21T19-38-32Z_arena_1_Batman.canonical_stimulus_v1.h5.receipt.json`

Neither H5 contains `/stimulus_coordinate_v6`. The receipt identifies the
derivative as `palette.stimulus_h5_derivative_artifact` v1 and records:

- source SHA-256:
  `543f4a57ba59e4c9df524f8cfbe122eef47f0ec7b68cf11e146ac1c9a51420f8`
- derivative SHA-256:
  `292d680c332a25a6eff62609b34f70a23dab3150422414086d2d22363060f3e7`

## Existing canonical publication

The analysis archive already contained a migrated canonical stimulus run before
this gate:

- archive:
  `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T19-38-32Z_arena_1_Batman/zarr/2026-07-21T19-38-32Z_arena_1_Batman_analysis.zarr`
- run: `analysis/stimulus_runs/stimulus_20260805_082709`
- source: the immutable migration derivative above
- direct parent selectors: `latest_complete` and `latest` both name this run
- completion: `complete`
- selector eligibility: `true`

This is therefore not a suitable selector-ineligible canary. It was not
rewritten, duplicated, or demoted.

Read-only validation with unconsolidated H5 preflight and the published
consolidated Zarr reader passed:

- source policy: `canonical_required_v1`
- rows: 165,557
- coordinate surfaces: 3
- source-contract SHA-256:
  `89db35cf0eafdf47317cad2d77624e68d354692514b03c8099bd596d534e0b3d`
- bound row-identity SHA-256:
  `fa955c4fdcc526c322b0767bd7b86d02f56ca6b9a4e88a5fe7c36bad0b28cc9f`
- output-manifest SHA-256:
  `4d16b154f78c4e87333d8f461a0679af60e29b9047e5812432c5b08d16005dfe`
- surface-manifest SHA-256:
  `a669a85d9e2fbac6d7debd13678d04be30d6a0bf1a8673b8de28e0a2fdf3fa94`

## Exact deployment

Palette commit:
`32a73d1855b912928ea42af04addc7f5f1376f05`

Commit-pinned cluster worktree:

```text
PALETTE_GROUPS_REPO=/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/sun-32a73d18
```

The deployment helper verified the commit, clean state, and imported `fisheye`
path through `login1-citrus-poller`. It also verified that the shared checkout
remained unchanged at `83ac49be7c0ee8cf8326b1f38b63ca0c9558582d`.

## Validation

Focused producer-v6 adapter tests:

```text
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_stimulus_coordinate_v6_adapter.py -q
```

Result: 4 passed.

The synthetic golden oracle proves the v6 parser, closed receipt, exact dtype
and shape validation, packed-row digest validation, and tamper rejection. It is
not a substitute for a real producer artifact.

## Crimson handoff and unblock condition

Crimson may use `stimulus_20260805_082709` only to test the already-published
canonical migrated surface. It must not report that run as producer-native v6
evidence.

The real v6 canary becomes unblocked when Citrus supplies one immutable H5 with
the exact `/stimulus_coordinate_v6` receipt and v6 source arrays. At that point:

1. run `validate_citrus_stimulus_coordinate_v6_artifact(...)` read-only;
2. publish a new immutable, selector-ineligible stimulus run;
3. compare direct and consolidated declarations and exact logical digests;
4. confirm no production selector or registry mutation;
5. hand Crimson the archive path, explicit run name, Palette commit, and
   manifest/content digests.

