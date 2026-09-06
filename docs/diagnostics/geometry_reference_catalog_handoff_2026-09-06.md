# Read-only acquisition geometry reference catalog — development handoff

## Ownership and state

- Owner: `geometry_reference_catalog` implementation worker; integration and
  shared authority work queue remain owned by the root coordinator.
- Worktree: `/tmp/palette-geometry-reference-catalog-20260906`.
- Branch: `agent/palette/geometry-reference-catalog-20260906`.
- Exact prerequisite/base: `925f8c6285499d39a99475945ba1732dba7f6834` (the
  coordinator verified all 24 required checks for that base).
- Scope at handoff preparation: three new implementation/test files, this new
  handoff, and two mechanically regenerated census count lines. No existing
  runtime module, authority schema, selector, registry, or scientific recipe is
  edited. The coordinator's handoff records the exact resulting development
  commit after commit creation; this document identifies its prerequisite and
  independently hashed implementation below.
- Implemented locally and validated as described below; not integrated,
  deployed, activated, or merge-ready. Required CI for the new development
  changes is unrun. Local tests are not a substitute for those checks.
- The user subsequently authorized development commits/pushes. The coordinator
  owns pushes and CI; this worker does not independently push or merge.

This is evidence linked from the existing authority consolidation work queue,
not a second status authority or a scientific acceptance receipt.

## Classification and preserved contracts

This is an additive, versioned diagnostic-index schema, with fail-closed index
conflict/staleness checks. Existing producer, candidate, pixel-frame, digest,
approval, and publication grammars are unchanged. It is not a new per-modality
acceptance schema or an additional human-review gate.

The implementation reuses the existing producer-native candidate planner,
which requires the recording-folder geometry loader and binds the exact live
Palette source-camera pixel authority. This intentionally lives in
`analysis_workflows.materializers`, not `shared`, so it does not introduce a
forbidden shared-to-application import or another coordinate binder.

Preserved invariants:

- Exact producer `rig_id`, `canvas_name`, `arena_id`, and textual
  `camera_serial` (including leading zeroes).
- Exact registration ID/digest, observation artifact ID/digest, source-pixel
  coordinate profile/dimensions and space, convention, units, origin and axes.
- Separate `physical_inner_rim` and already-expanded `valid_detection_region`;
  no additional Palette tolerance and no newly manufactured top-rim label.
- Existing producer operator-acceptance and quality flags. Quality flags remain
  visible; the index does not invent a new approval from their absence.
- The existing candidate record and its existing digest are returned unchanged.
  The candidate remains candidate-only, not operationally selected.
- Folder/bundle geometry remains sufficient without consulting the independent
  Citrus runtime-applied flag. H5-only and recovered references are outside
  this initial index profile; their existing adapters remain unchanged.

The index never proves more than its sources: neither its own digest nor a
prior validated in-memory dictionary grants durable admission. Validation
reopens all indexed producer assets and pixel bindings. Exact resolution
revalidates the selected source again. No latest, nearest, other-recording,
current-calibration, or human-review fallback is introduced.

## API and digest ownership

New runtime files:

- `src/fisheye/analysis_workflows/materializers/arena_geometry_reference_catalog.py`
- `src/fisheye/utils/inspect_geometry_reference_catalog.py`

The catalog exports `GeometryReferenceSource`, `GeometryReferenceKey`,
`build_geometry_reference_catalog`, `validate_geometry_reference_catalog`,
`resolve_geometry_reference`, and the closed source-descriptor parser.
The policy worker agreed that resolution supplies the existing acquisition
candidate record, rather than a competing policy/acceptance payload.

`palette.recording_geometry_reference_catalog` version 1 declares
`catalog_role=non_authoritative_index`. Canonical serialization/digest ownership
is the existing `fisheye.shared.zarr.manifest_digest` module. The index digest
is not substituted for any producer file hash, candidate digest, or pixel
authority digest. Reference keys/digests omit recording-specific locators and
bindings for deterministic deduplication; every occurrence retains those exact
recording-specific claims separately.

Expiration is evaluated only against an explicit `applicable_at_utc`.
`None` reports `not_requested`, not indefinite validity. Naive or malformed
timestamps fail; the declared expiration boundary is exclusive. Historical
references are never silently evaluated against the current wall clock.

The inspection CLI writes only stdout:

```bash
scripts/py -m fisheye.utils.inspect_geometry_reference_catalog build --source-list /absolute/sources.json
scripts/py -m fisheye.utils.inspect_geometry_reference_catalog validate --catalog /absolute/catalog.json
scripts/py -m fisheye.utils.inspect_geometry_reference_catalog resolve --catalog /absolute/catalog.json --key-json /absolute/key.json --source-zarr /absolute/recording/zarr/recording_analysis.zarr
```

Source-list JSON is a list of exact absolute `recording_root`, `source_zarr`,
`rig_id`, `canvas_name`, `arena_id`, textual `camera_serial`, and nullable
`applicable_at_utc` objects. It does not infer camera or registration choice.

## Validation evidence

All Python commands used `scripts/py`, resolving to
`/home/delahantyj@hhmi.org/miniconda3/envs/palette-py311/bin/python`.
Pytest ran outside the sandbox on the workstation, not on login nodes/LSF.

The new 21 tests exercise real synthetic folder/bundle files and real tiny
persisted pixel authorities through the unpatched existing planner. They
cover deterministic cross-recording deduplication, candidate-record identity,
leading-zero serials, wrong rig/canvas/arena/camera, missing/tampered assets,
missing producer approval, partial producer state, live pixel-proof drift,
expiry, geometry/registration conflicts, recomputed-index-digest tampering,
wrong exact lookup, and the read-only CLI.

```bash
scripts/py -m pytest tests/unit/fisheye/test_recording_geometry_reference_catalog.py tests/unit/fisheye/test_recording_geometry.py tests/unit/fisheye/test_arena_geometry_candidates.py -q
# 53 passed in 3.15s
scripts/py -m pytest tests/unit/fisheye/test_arena_geometry_comparison.py tests/unit/fisheye/test_arena_geometry_workflow.py -q
# 23 passed in 3.30s
```

Total focused regression evidence: 76 passed. Also passed: `py_compile`,
Black formatting, `git diff --check`, both import-linter contracts, file-size
ratchet, Zarr metadata-mode ratchet, observed-metadata literals, active contract
freshness, FPS authority access, keypoint-motion authority access, tail receipt
access, paradigm authority access, and generated-artifact check. The initial
generated-artifact check identified changed module counts; owner regeneration
changed only `scanned_python_module_count` from 1711 to 1713 in the existing
schema/writer census JSON files, after which the check passed.

## Authorized live 84-recording exercise

Read-only exercise finished at `2026-09-06T21:15:35.184851+00:00`:

- Input: `/tmp/palette-goodbatbadbat-geometry-audit-20260906/audit.json`;
  byte SHA-256 `a109fee395d436c3464074a4f3966865d45b40b5d861a31505298bdfc890a021`.
- Report directory:
  `/tmp/palette-geometry-reference-live-catalog-20260906-jglc9pe7/`;
  `report.json` records every exact input, result, source/pixel digest, and time.
  `sources.json` and `catalog.json` are the explicit reusable diagnostic inputs.
- All 84 existing approved folder/bundle sources and live source-pixel bindings
  validated; 0 refused. They deduplicate to 32 exact references across eight
  registration digests. The combined index was independently rebuilt after the
  individual checks. This is validation of existing evidence, not new fitting.
- Catalog canonical digest:
  `8ad0a4824f5f153ceae479959b01c552c67233a4a80206d153a10cf01dec22a1`.
  Pretty-printed catalog byte hash (a distinct identity):
  `3545b049768e9e19ce5cdb66c74c82dcee173d25a2fbdb78ec18a8d2a75b61bc`.
- Implementing module byte SHA-256:
  `e1c94c7acc86ea9724b8916c079e33ab5ae6535efe7911fc463163e60ca6d6dc`.
- Metadata mode: explicit `unconsolidated_diagnostic`, inherited from the
  existing native candidate planner. No published consolidated generation is
  asserted or repaired by this diagnostic.
- Applicability times: exact UTC timestamps encoded in the audited recording
  directory names, declared explicitly in each source descriptor. This checks
  historical recording-time expiration, not present-day reuse eligibility.
- Individual validation plus combined rebuild elapsed 98.999 seconds; no video
  decode, refit, detection-rowset recomputation, dataset write, registry write,
  selector mutation, deployment, or new scientific acceptance occurred.
- A subsequent real CLI `validate` invocation reopened all 84 references and
  returned `validated_index_only`, `reference_count=32`, `source_count=84`,
  the same catalog digest, and `authority_activated=false`.

The `/tmp` outputs are diagnostic evidence, not production operational storage.
They do not replace later exact source validation or authorize a deployment.

## Remaining gates and compatibility

Every required remote CI check for these new changes is **UNRUN**:
`generated artifacts`, `import boundaries`, `file-size ratchet`,
`zarr open metadata modes`, `observed metadata literals`,
`active contract freshness`, `package and collection`,
`non-gpu tests (shard 0)`, `non-gpu tests (shard 1)`,
`non-gpu tests (shard 2)`, `non-gpu tests (shard 3)`,
`non-gpu tests (shard 4)`, `non-gpu tests (shard 5)`,
`non-gpu tests (shard 6)`, `non-gpu tests (shard 7)`,
`non-gpu tests (shard 8)`, `non-gpu tests (shard 9)`,
`non-gpu tests (shard 10)`, `non-gpu tests (shard 11)`,
`non-gpu tests (shard 12)`, `non-gpu tests (shard 13)`,
`non-gpu tests (shard 14)`, `non-gpu tests (shard 15)`, and `ci-required`.
The coordinator must push the exact development commit, obtain all required
green checks, and separately validate any later combined integration commit.

This diagnostic adds no production catalog caller or policy activation. The
parallel policy/comparison work remains separately owned. H5-only/recovered
index support, production policy promotion, collection acceptance, historical
mutation, and deployment are not implemented or implied here. No executable
compatibility path was removed, and no dependency installation occurred.
