# Detection Snapshot Storage Profile Promotion

Date: 2026-07-27

Status: profile contract promoted in code; selector-ineligible promoted-ID
canary passed; production selectors unchanged

## Decision

`detection_published_access_aware_v1` is the default physical profile for new
immutable canonical-detection v1 and refined-detection v1 snapshots.

The exact profile is:

- `WINDOWED` and `INDEXED` inner chunks: 128 KiB uncompressed;
- `EAGER` inner chunks: 1 MiB uncompressed;
- indexed outer shards: 8 MiB target and maximum;
- Zarr format: v3;
- codec profile: `zstd_fast_v1`;
- shard-index codecs: little-endian bytes plus CRC32C, index at end;
- sharding: immutable and whole-shard-owned writes only; and
- estimated payload-object budget: 4,096 per array.

This is a named production profile, not a renamed benchmark object. Its byte,
shard, codec, and access-class values are exactly the previously tested
access-aware candidate. Tests compare every resolved canonical/refined chunk,
shard, and codec declaration between the evidence candidate and the promoted
profile.

`detection_regular_rollback_v1` is the explicit rollback profile. It uses
exact 1 MiB uncompressed chunks and no outer sharding. It is never selected by
default.

## Evidence

The canonical-detection evidence is recorded in
`docs/diagnostics/canonical_detection_storage_access_aware_result_2026-07-24.md`.
At full Sleepyfish scale, the access-aware layout reduced payload objects from
88 to 16, improved publication and mounted-reader time, improved random reads
and sequential throughput, and did not increase peak RSS. Later Crimson
residency and full-application work attributed the old frozen-gate failures to
scheduler and process-memory behavior rather than this physical layout.

The refined-detection Palette gate is recorded in
`docs/diagnostics/refined_detection_physical_profile_canary_plan_2026-07-27.md`.
It compared exact 1,188,000-frame, 1,187,087-row snapshots and proved identical
decoded hashes, exact source-audit equality, one 1,188,001-entry offset index,
direct/consolidated equivalence, and no production-state changes. Observed
payload objects fell from 220 to 42, a 5.24x reduction.

Crimson then reported its mounted-macOS full-duration detection-isolated gate
passed at benchmark implementation
`9cf04acee9682a6f4f5fae005c0af6077ec5cc4b` and evidence verdict
`258f25811c76dd48e206183d73f9807e140e7264`:

- exact decoded detections and traversal digests matched;
- traversal transfer fell from 4.65 MiB to 0.61 MiB, an 86.8% reduction;
- whole-process transfer fell from 16.95 MiB to 5.33 MiB, a 68.5% reduction;
- median current-frame p95 improved from 144.2 ms to 48.0 ms;
- both layouts had zero post-warmup deadline misses;
- access-aware used 21 smaller reads instead of 14 larger reads; and
- payload objects remained 42 versus 220.

The Crimson files are
`docs/diagnostics/refined_detection_physical_profile_canary_2026-07-27/README.md`,
`aggregate.json`, and
`refined_detection_physical_profile_comparison.png`. The immutable handoff
binds `aggregate.json` to SHA-256
`0be6f191b0d684914cdd48bc938267cd8bb2fb6e066e158ba65cf2339d466d32`.

Palette then published the named-profile verification canary documented in
`docs/diagnostics/detection_promoted_profile_canary_2026-07-27.md`. LSF job
`153192513`, pinned to Palette
`d7436b083af0f6bc4b8004952a4ff8542b98f156`, passed. A separate fresh process
validated all 9 canonical and 28 refined arrays, proved the logical hashes
equal to the previously measured full-duration pair, and proved every refined
physical declaration equal to the frozen access-aware candidate while the
persisted profile ID was `detection_published_access_aware_v1`.

## Activation Boundary

Promotion means:

- `plan_canonical_detection_storage()` defaults to the named promoted profile;
- `plan_refined_detection_storage()` defaults to the same profile;
- selector-ineligible canonical/refined v1 snapshot writers inherit that
  default unless an exact profile is passed explicitly;
- persisted manifests reconstruct and validate their own embedded profile, so
  older `published_http_v1` and frozen benchmark artifacts remain readable;
  and
- a manifest that uses a registered profile ID with changed budgets fails
  closed.

Promotion does not:

- change `latest`, `latest_complete`, `authoritative_run`, or a registry;
- retroactively rechunk an existing archive;
- make benchmark artifacts canonical;
- convert the streaming legacy `detect_yolo` output into canonical v1; or
- convert the current clipped refined aggregate, whose older logical schema
  and lineage still require the dedicated clipped-v1 transition.

Those legacy writers remain compatibility surfaces. New v1 publication is a
compute/edit/append-to-delta followed by immutable compaction and publication
boundary; workers must own complete non-overlapping physical units.

## Remaining Checklist

- [x] Publish one fresh selector-ineligible canonical/refined pair using only
      the promoted defaults and validate it in a separate process.
- [x] Obtain and bind the full immutable Crimson commit IDs and evidence
      digest.
- [ ] Have Crimson perform an exact-schema open-only smoke against the named
      profile canary. No new performance matrix is required.
- [ ] Enable a selector only through a separately reviewed publication change.
- [ ] Design detection delta v2 and the immutable compactor before routing
      manual additions through this snapshot writer.
