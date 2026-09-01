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
- [ ] Run a real one-recording Phase-C canary from a clean commit-pinned
  deployment.
- [ ] Confirm the produced Parquet row and lazy reader query against the real
  cohort authority.
- [ ] Pass every required CI check before merge or production use.
- [ ] Publish a new Phase-C cohort generation if the feature is accepted.
- [ ] Optionally add recording-level color overlays to grouped-statistics
  figures. The required exact occurrence roster is now present in their
  statistics/view payload; renderers must not reach around it to rediscover
  source data.

## Current status

The implementation is an engineering candidate based on Palette commit
`6a9fa41793fc5a946ed13a4b910ff87ae4016f82`. It is not yet merge-ready,
selector-eligible, deployed, or a replacement for the completed Phase-B
cohort publication.
