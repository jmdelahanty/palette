# Validated-behavior chaser appearance export successor — 2026-09-01

## Decision

Protocol-authored chaser color is a first-class, receipt-bound occurrence
attribute. It is not a role lookup, a filename convention, or a display
fallback. The stable join grain is:

```text
(recording_id, chaser_identity_code)
```

Behavior role and experimental color are independent channels:

- fill/stroke color may encode the exact protocol RGBA;
- marker shape, line pattern, and text may encode behavior role;
- role must never be inferred from color;
- color must never be inferred from role or chaser index.

## Version boundary

The successor installs the profile
`validated_recording_behavior_phase_c_v1`. Phase A and Phase B remain
unchanged and continue to resolve their immutable version-1
`chaser_occurrences` contract. Phase C preserves the complete Phase-B table
roster but replaces only `chaser_occurrences` with Arrow schema version 2.

This avoids silently changing the meaning or physical schema of an existing
publication. A Phase-C export is a new immutable publication; it does not
mutate or relabel the completed Phase-B cohort.

## Authority chain

The Phase-C adapter follows one closed chain:

1. exact cohort membership member;
2. exact complete recording-bundle member;
3. bundle-bound `chaser_relative_keypoint` validation receipt;
4. exact relative-frame run manifest and chaser occurrence envelope;
5. exact `analysis/stimulus_runs/<run>` named by that occurrence;
6. `protocol_json` whose canonical digest equals
   `source_protocol_sha256`;
7. explicit per-chaser `color_r`, `color_g`, `color_b`, and `color_a`.

Before exporting a row, the adapter proves that the occurrence record in the
relative-frame receipt is exactly equal to the occurrence sealed by the
recording bundle. The existing fail-closed appearance projection then proves
column-axis identity, behavior-role agreement, protocol digest, explicit RGBA
presence, finite unit-range channels, and glyph-policy coverage. There is no
selector discovery and no color fallback.

## Exported occurrence surface

The original occurrence identity and provenance columns remain intact. Schema
version 2 adds:

- `behavior_role_code`;
- `experimental_color_r`, `experimental_color_g`,
  `experimental_color_b`, and `experimental_color_a`;
- `experimental_color_hex` and `experimental_color_css`;
- `contrast_outline_hex`;
- `plotly_role_symbol` and `matplotlib_role_marker`;
- appearance schema, version, policy, projection digest, and occurrence digest;
- explicit color semantics, role semantics, and color/role independence.

The renderer-specific marker columns are conveniences sealed by the shared
appearance policy. Consumers remain free to choose another role glyph, but
they must retain the independent `behavior_role` data and must not reinterpret
experimental color as role.

## Composable reader pattern

The existing lazy reader exposes the surface without a protocol-specific
export API:

```python
appearance = dataset.table("chaser_occurrences")
rows = appearance.scan(
    columns=(
        "recording_id",
        "chaser_identity_code",
        "chaser_identity",
        "behavior_role",
        "experimental_color_hex",
        "experimental_color_a",
        "plotly_role_symbol",
        "appearance_projection_sha256",
    )
).collect()
```

Frame-, trial-, bout-, and histogram-grain tables join to the dimension by
recording and chaser identity code. Consumers should also retain the export
manifest and table-contract digests returned by `query_identity()` in their
own receipts.

## Plotting composition rules

Per-recording points, paths, and object overlays can use the exact experimental
RGBA directly after the occurrence join. Role remains visible through marker
shape, line pattern, label, or an equivalent independent layer.

A cohort summary grouped only by `behavior_role` may contain recordings in
which that role used different experimental colors. Such a summary does not
have one truthful protocol color. A viewer must use one of these explicit
choices:

1. show recording-level marks in their exact colors and use a role glyph;
2. stratify the statistic by an exact color dimension and state that grouping;
3. use a neutral or condition color for the aggregate and retain role text or
   glyph.

It must not select an arbitrary member color, hardcode a color-to-role map, or
average RGBA values and present the result as a protocol-authored color.

## Implementation checklist

- [x] Preserve the installed Phase-A and Phase-B contracts.
- [x] Add a Phase-C profile with `chaser_occurrences` schema version 2.
- [x] Reuse the exact bundle-bound relative-frame receipt.
- [x] Prove bundle/receipt occurrence equality before reading color.
- [x] Resolve only the exact digest-bound stimulus protocol.
- [x] Export RGBA, role code, glyph policy, and projection provenance.
- [x] Add tests for schema isolation, color/role independence, exact axis
  resolution, and occurrence drift.
- [x] Copy the small exact occurrence roster into grouped-statistics source
  provenance, with its own digest and source-query identity.
- [x] Make static and Marimo grouped renderers consume truthful aggregate
  colors and independent role glyphs. A role with multiple protocol colors is
  rendered neutrally rather than assigned an invented protocol color.
- [x] Run a real one-recording Phase-C canary from a clean commit-pinned
  worktree.
- [x] Confirm the produced schema-v2 Parquet rows against the real cohort
  authority and shard receipt.
- [x] Confirm finalized-manifest lazy-reader reopening after a complete
  Phase-C cohort publication.
- [x] Pass every required CI check before merge or production use.
- [x] Publish and validate a new selector-ineligible Phase-C cohort generation.
- [ ] Optionally add recording-level color overlays to grouped-statistics
  figures. The required exact occurrence roster is now present in their
  statistics/view payload; renderers must not reach around it to rediscover
  source data.

## Current status

The implementation commit is
`909d67bcadd4e8f2dc8a6ebca966534b2a14c5f9`, based on Palette commit
`6a9fa41793fc5a946ed13a4b910ff87ae4016f82`. PR 109 passed all 23 required
checks and merged as `19a006cc0e774a7a98a65fc917627552407a94f9`.

The complete Phase-C cohort publication described below is available through
its exact run ID and manifest. It remains deliberately selector-ineligible and
does not mutate or implicitly replace the Phase-B publication.

## Commit-pinned canary evidence — 2026-09-01

The selector-ineligible plan
`phase-c-chaser-appearance-canary-909d67bc` reused the exact 84-member
membership and bundle-set authorities from the completed Phase-B cohort. Its
plan digest is
`a7d073d97f6c0d7358e5542d0d4c99deec7726335d86c1f122b436b9ee0baed5`.
The plan safety record sets source/Zarr mutation, registry update, selector
activation, production authority, and selector eligibility to false.

Member 1,
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat`, completed all 30 table parts.
The exact shard receipt digest is
`f5a73b46ccb5035586495f8651343c2d6104f3bf4ca81dc2b580411ae101dcba`.
The `chaser_occurrences` part contains two rows, uses Arrow schema version 2,
has contract digest
`b0fa11f3f70f2c90772596d83f16f20a9f76426afdb0b312a0a29e51ef9f7605`,
and file digest
`eee2329a0a44d7a0adb18888338d1f2fbb280908631ef39ebe4d3c4b7d8391cf`.

Both exact protocol occurrences have RGBA `(0, 0, 0, 1)` and
`experimental_color_hex = #000000`. Their roles remain different and
independently encoded:

- identity code 1: `aggressive`, Plotly `star`, Matplotlib `*`;
- identity code 2: `inert`, Plotly `circle`, Matplotlib `o`.

Both rows bind appearance projection
`966103d63f0b1081b43c97ea1068cb9adc2ea7e1f05cbc6c1d3b4156453b45e6`
and occurrence record
`713540f2c9b634dd772a4f38b2040dbb5ef285f6103f9ecdac4c96eab7333067`.
This real specimen directly demonstrates why color and behavior role cannot be
collapsed into one channel.

## Full cohort production evidence — 2026-09-02

The merged commit was deployed as one clean, locked cluster worktree at:

```text
/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/validated-behavior-cohort-phase-c-20260902-19a006cc
```

The deployment helper verified the imported `fisheye` path from
`login1-citrus-poller` and left the shared `/groups` checkout unchanged. The
publication reused the exact immutable Phase-B cohort inputs:

- membership file SHA-256
  `d13a511e020c4a21708923c44ce40807c8ba3d0ed43e85770d3742667437a615`
  and record digest
  `1f438ace63dfe66dcc53cc7560f63d76924bfefaec2bc8ca0069bd767c1208b1`;
- bundle-set file SHA-256
  `29c00c6ce76044d14fa85f9d65c12e00ba840b6b5d234b933dfb3759d92f7503`
  and record digest
  `16f7035712bf68f8686027f279a3c736f3d81b596033396383dac06ed30e5ed1`.

The operation root is:

```text
/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_phase_c_20260902_19a006cc
```

The immutable export run ID is
`goodbatbadbat-validated-behavior-phase-c-20260902-19a006cc`; its plan digest is
`cbf26791b2d69606120fc69e618727f44ff2d64ac75486a9d15b074e680fd676`.
LSF array `153829521` completed all 84 members with a `%12` concurrency bound,
84 successful job footers, 84 exact shard receipts, and no nonempty stderr
logs. Shard runtime was 36 seconds median, 45 seconds at the 95th percentile,
and 101 seconds maximum. Peak memory was 984.5 MiB median, 1,030 MiB at the
95th percentile, and 7,085 MiB maximum under a 16 GiB request.

The all-success-dependent finalizer `153829522` completed in 3,217 seconds
with 3,214.94 CPU seconds, 737 MiB peak memory, and empty stderr. It performed
the full global part, primary-key, and foreign-key validation twice: once to
create the validation receipt and once against the fixed staged bytes before
the atomic manifest commit.

The committed generation ID is `87d5a958b505457eb3566eb98c267ab0`.
It contains 30 tables, 2,520 Parquet parts, 140,542,546 rows, and
6,992,426,183 Parquet bytes. The generation has exactly 2,605 files: the 2,520
parts, 84 copied shard receipts, and one validation receipt. The staging
namespace was empty after commit. Retaining both independently validated
shards and the immutable publication uses approximately 14 GiB.

The exact publication identities are:

- manifest record digest
  `8fb2c7ecabeff2b13b6178416842f477d99c55ae8d7df540f5dd71eea7ad1646`;
- manifest file SHA-256
  `7b23e5a5a44ba57b0cc5a9cef38deeebab5a05fb79d05b6c65a11c2f19885db5`;
- part-inventory digest
  `cdfe5131e8ea114a64d90430260cf031c2f533dfbbc57518edb0e65e7c8c9856`;
- validation-receipt record digest
  `382b8cfb2cc0ef5898e942094eeadd1bffd99bc2985c5dbe422214ec297ece8b`;
- validation-receipt file SHA-256
  `1e29a0063405241c850032c834a23a15dc4471ba8d31f2ee67a04edac29cdc4d`.

All 84 member receipts bind the same plan digest and merged Palette commit.
Eighty recordings retain complete bundles; the four previously adjudicated
invalid members, ordinals 77--80, retain typed zero-row scientific parts and
are not silently dropped. The aggregate row counts match Phase B exactly.
Only the `chaser_occurrences` schema and bytes changed to add the Phase-C
appearance surface.

An independent receipt-mode `ValidatedBehaviorExportDataset.open(...)`
completed in 25.377 seconds. A bounded manifest-selected query returned 160
unique schema-v2 chaser occurrences under contract digest
`b0fa11f3f70f2c90772596d83f16f20a9f76426afdb0b312a0a29e51ef9f7605`:
80 aggressive/star rows and 80 inert/circle rows. Both roles have the exact
protocol color `#000000`, proving that the persisted role channel remains
independent of color across the full cohort.

The plan, manifest, and validation receipt all set source/Zarr mutation,
registry update, selector activation, production authority, and selector
eligibility to false. This is an immutable validated publication available by
exact identity, not a production selector promotion.
