# Unified experimental H5 import integration

Status authority: `INGEST-001` in the
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md).
This is implementation/handoff evidence, not another work queue.

## Ownership, scope, and prerequisites

- Owner: Palette root worker, user-authorized on September 12 to implement on
  an isolated branch, commit/push a PR, and obtain required CI. Stop before
  merge or deployment. No installation, production-data mutation, default
  change, historical backfill, or companion retirement is authorized.
- Worktree: `/tmp/palette-unified-h5-import-20260912`.
- Branch: `agent/palette/unified-h5-import-20260912`.
- Base: `592e9c499cdee034f121b54e565be59474305402`, with successful exact
  `main-integration` same-tree evidence in run `34705786606`, as permitted by
  the current Required CI and Integration contract v2.
- The original checkout's dirty instructions and review documents and every
  other worker's checkout are preserved. No dirty overlapping runtime work
  was adopted. The relevant public stimulus importer and coordinate contract
  have no mainline changes since the parent-intake checkpoint below.
- Parent-transfer PR 149 remains separately owned/scoped at
  `0653e75e5faad6b05a4ec715621cc6d9ee727f5d`, with all 24 required checks
  successful in run `34134814331`. Its 109-file transfer/intake change is not
  needed to add this profile to the existing public stimulus importer and is
  not incorporated here. Parent-transfer integration remains a separate gate.
- Reviewed contract: agent-contracts PR 51 at
  `408307c9a6b11258546b4465fa43323cb3668013`, based on PR 50 at
  `f775aa652e524a13fbdff9e832937eb74ba191bc`.
- Producer: Citrus `74083cee35196b4b1fc58e8bea180455a0b924ad`; base unified
  fixture/schema evidence: `0a9e2d490ab0772dbaa7fdda3f837693806caad0`.
- [Scoped artifact approval](https://github.com/jmdelahanty/agent-contracts/pull/51#issuecomment-5648036140)
  is not producer CI, Palette integration CI, or production admission.

## Classification and preservation contract (before implementation)

This is an additive, versioned input/storage profile and an enforcement
correction for that explicit profile. It is not a behavior-preserving alias for
v5/v6 and must not silently pass native geometry through the old coordinate
normalization path. Existing v5/v6 reader behavior and persisted digest bytes
remain unchanged. No renderer, camera/rig parameter, timing convention, source
row identity, or scientific acceptance policy changes.

The public `import_stimulus_to_zarr` entry point will accept an explicit
`source_profile=unified_experimental_h5_v1` and an explicit external finalization
receipt. A recognized unified H5 without that selection must fail before any
destination mutation. Unknown profiles, incomplete finalization, altered bytes,
wrong receipts, malformed schemas, unresolved joins, and declared-but-missing
appearance must refuse, not fall back to metadata-only or legacy import.

The first supported result is an immutable, selector-ineligible native-profile
candidate under the existing stimulus run family. It exposes the common frame
ledger, all native component tables, correspondence, protocol, geometry, timing,
and captured lineage without manufacturing a coordinate companion or physical
registration authority. Native paths and explicit validity remain intact;
renderer code values are not measured projector luminance. The existing
production selectors, registry acceptance, source-camera authority, and
physical-coordinate publication must remain untouched.

Preserve both independent integrity domains: internal schema/logical payload
digests and the external exact-H5-byte receipt. A changed physical Zarr layout
gets its own versioned storage manifest, not a claim that H5 container bytes or
old Palette output digests survive rechunking. Candidate readers must verify
copied payloads and metadata before returning native facts.

Validation and copying must be bounded and single-writer. Retain the producer's
64 MiB dataset, 2,000,000-row, and 8 MiB JSON/string budgets; stream payload
hashing/copying and use bounded/disk-backed key validation. Reuse the existing
stimulus ownership/failure guard and run-completion provenance. All source
validation precedes destination creation; recheck exact source generation and
ownership before completion. Failures retain only an owned ineligible tombstone
and cannot overwrite another owner's output. Retries require a new run name.

## Planned preservation and refusal evidence

Before runtime implementation, pin the emitted synthetic fixtures and test:

- Appearance-bearing globally finalized H5: 89 dependencies, 37 schema-tagged
  tables, exact frame/source/target joins, profile/enum identity, all 34 appearance
  members and quantization bridge; compare declared invariants with old format.
- Earlier successful unified H5 without appearance: valid optional absence,
  unchanged shared core/schema/digest identities.
- Frame-only, grid, and grating component-scope fixtures: scoped joins without
  requiring Chaser; never mislabel them globally finalized recordings.
- Reused profile tuple in a later step with a different Chaser, plus duplicate
  owner/key, wrong step/profile/Chaser/reference, invalid boolean/enum/fill,
  malformed manifest, source/receipt mutation, incomplete lifecycle and bounded
  read refusals. Preserve exact uint64 keys above `2**63`.
- Public importer -> real local Zarr -> unpatched native candidate reader,
  payload/metadata tampering, stale source, failed completion/consolidation,
  ownership loss, immutable retry, and unchanged production selectors.
- Existing stimulus coordinate, protocol, import-path, lease and publication
  tests; package resources, generated inventories and repository CI gates.

## Implemented interface and storage

The public Python importer and CLI now implement the explicit profile and
receipt arguments. Its legacy branch is unchanged apart from rejecting a
recognized native file before legacy fallback. The native branch uses the
existing stimulus creation/failure/ownership guard, completion provenance, and
metadata consolidation helper. Every lifecycle attribute write is owner-checked;
copying, source revalidation, direct/consolidated verification, failed cleanup,
and fresh-name retry are exercised through real local Zarr.

`fisheye.shared.unified_h5` owns the native schema and admission contract.
Protocol validation reuses the existing semantic/execution owners, Orange
mapping validation reuses the existing closed source-map validator, and
appearance replay uses the unchanged pinned portable producer evaluator.
Component-only appearance join evidence remains separate from mandatory full
artifact protocol and numerical replay. A generic component descriptor permits
its scoped geometry reference; whole-artifact geometry admission additionally
requires every geometry-bearing table to reference the validated native authority.

The new `palette.unified_h5_native_storage@1` layout preserves packed native
records instead of extending the existing 512-byte-limited columnar string path.
It uses the existing ArrayContract/planner/array-factory owner, regular
`scratch_compute_v1` byte arrays and one physical writer. All native paths,
types, shapes, payload bytes and typed attribute bits are bound in a bounded
manifest. The reader is an explicit native candidate reader, not a selector or
physical-coordinate supplier. See [usage and exact limits](../unified_h5_native_import.md)
and [pinned resource provenance](../../src/fisheye/shared/unified_h5/contracts/README.md).

No compatibility adapter was removed. No scientific/default/coordinate identity,
existing digest grammar, parent-transfer dispatch, authority selector, registry,
or production-data write was changed. No dependencies were installed. This
technical admission does not add a modality-specific scientific acceptance
schema; the four claims in the acceptance checklist remain independent.

## Local validation checkpoint before PR CI

The original 21 frozen-producer preservation tests were captured and passed
before maintained runtime implementation; the planned APIs initially failed
collection as expected. The final new suite adds full-reader and refusal coverage:

- Original/appearance H5 values and all 34 appearance members; full byte/type
  round-trip of every copied dataset, wide strings, signed timestamps, typed
  attributes, negative zero, and uint64 keys above `2**63`.
- All dependency/schema digests, scoped joins without Chaser, optional appearance
  absence, step-local profile ownership, golden replay vectors, explicit booleans,
  zero fill, enum validity, malformed JSON, and bounded allocation/read refusal.
- Wrong/missing external receipt, failed/pending finalization, source handle/path
  mismatch, source replacement during copying, external links and cycles.
- Public importer -> real Zarr -> unpatched native reader; unchanged existing
  production selectors; payload/manifest/metadata/provenance tampering; stale
  consolidated metadata; copy/completion/consolidation/no-op/persist-then-raise
  failure; owned tombstones; fresh-name retry; takeover during copying and during
  completion with no subsequent foreign-owner writes or cleanup.

Recorded local results: 96 new tests passed together before the last additional
closed-JSON type case; 116 existing importer/context/v6/protocol regression tests
passed. Full test collection succeeded (12,073 at that checkpoint). Subsequent
final-head coverage and the exact CI revision are recorded on the PR; do not use
these historical counts as an exact-head CI attestation.

Import boundaries and all four scoped authority-access checks, file-size ratchet,
explicit Zarr metadata modes, observed metadata literals, contract freshness and
registry schema reference passed locally. The Zarr inventories are regenerated
from the changed source. A non-editable wheel was built using the existing
setuptools backend without installation; importing directly from that wheel
loaded all four hash-pinned contracts and the portable evaluator.

All 24 new-head required checks must succeed: generated artifacts, import
boundaries, file-size ratchet, Zarr open metadata modes, observed metadata literals,
active contract freshness, package and collection, non-GPU shards 0–15, and
`ci-required`. Their live results and exact commit belong to the isolated branch's
PR, not this pre-PR local-evidence checkpoint. The branch must not be merged,
integrated, deployed, or described as merge-ready while any required check is
failing, unrun, cancelled, unexpectedly skipped, or pending.

Status at this checkpoint: implemented and locally validated; not integrated,
deployed, or activated. The only dirty scope is this isolated adapter, its tests,
documentation and regenerated inventories; other checkouts remain untouched.

## Compatibility and later work

Canonical physical-coordinate normalization for native unified inputs,
downstream production analysis adoption, parent-transfer dispatch, real-data
canary, Citrus production admission/default change and companion retirement
remain separate, explicit work. This candidate import must not imply those
claims or make an experiment/profile-wide acceptance decision.

## September 12 integration follow-up

The user subsequently authorized merging PRs 168 and 169. PR 168's exact head
`b6aef85ceeec0b062bc980230542e3a04a6e13b3` passed all 24 required checks in
run `34719387296`, then merged as `7498ea029a41eb4f5592f571d801d63d754394f9`.
Its exact `main-integration` gate passed in run `34733875255` before this branch
incorporated that main commit. PR 169's incoming head
`dc00de7c762d1d819a0cd920159f485ac7ff13c7` independently passed all 24 checks
in run `34722320905`, plus 98 new and 116 legacy local tests.

The root worker owns the combination in the same isolated PR 169 worktree and
branch named above. The integration preserves both implementations and their
contracts unchanged; its only merge conflicts were generated Zarr inventories,
which are regenerated from the combined source. The original checkout, PR 168's
worktree, and commit-pinned deployment are not modified. Fresh combined-head
tests and all 24 required CI checks must pass before PR 169 merges; the exact
combined commit and results are recorded on [PR 169](https://github.com/jmdelahanty/palette/pull/169).
The resulting main commit must then pass its own integration gate. Merge
authorization does not authorize deployment, shared-checkout advancement,
registry mutation, or production activation.
