# Baseline-strategy and training-response Arrow v2 contracts — 2026-08-04

## Outcome

Palette’s two downstream Parquet families no longer infer physical schemas.
They now publish distinct v2 contracts through the same exact Arrow and
immutable-generation machinery used by canonical cross-recording exports.

The maintained surfaces are:

| Family | Exact tables | Ordered fields | Primary-key grain |
|---|---|---|---|
| Baseline strategy v2 | `baseline_strategy_features`, `baseline_exploration_episodes`, `baseline_strategy_classification`, `baseline_strategy_clusters` | 86, 30, 35, 24 | analysis run × recording × track × baseline window; episodes additionally include episode ID |
| Whole-training response v2 | `training_response_features`, `training_response_classification`, `training_response_clusters` | 102, 35, 25 | analysis run × recording |

`training_window_id` remains nullable lineage rather than part of the
training-response primary key because an invalid recording can legitimately
have no selected training window. Every nonempty primary key is embedded in
the exact table-contract payload and therefore participates in its SHA-256.

This change does not modify recording-local Zarr authorities, production Zarr
selectors, registry authority, or physical Zarr profiles.

## Exact representation decisions

- Every table has one ordered field inventory with exact Arrow types and
  nullability. Row normalization rejects unexpected fields, missing or null
  required values, wrong primitive types, integer overflow, non-finite
  floating-point values, and wrong installed schema/method constants.
- Status branches no longer change physical schemas. Absent optional results
  are explicit nulls.
- Training-response v2 freezes the current role-derived vocabulary to
  `aggressive` and `inert`. A configuration that would generate different
  column names fails closed rather than creating another inferred schema.
- Variable `bic_by_component_count` structs are retired. V2 persists
  `bic_by_component_count_json` as canonical strict JSON with positive decimal
  component keys and finite float64 values. Legacy mappings are converted only
  at the v1-to-v2 producer boundary; already named v2 JSON must be canonical
  byte-for-byte.
- Every table, including a zero-row table, has exactly one Parquet part with
  its full exact schema and contract footer. Empty data is not represented by
  an empty directory or a zero-column frame.

## Publication and selection contract

Both families use this lifecycle:

1. Normalize every row and validate the complete table’s primary keys.
2. Write all exact parts beneath a hidden v2 staging generation.
3. Record each part’s manifest-relative path, SHA-256, size, and row count.
4. Validate the closed table inventory, exact Arrow envelope, footer digests,
   part receipts, row counts, in-row run identity, and absence of extra files.
5. Rename the complete directory once into its immutable generation path.
6. Compare-and-swap one strict-JSON manifest as the sole visibility boundary.

The manifest declares `state=complete`, explicit boolean
`selector_eligible`, intended use `analysis`, the exact primary keys, all part
receipts, and a digest over the complete outer payload. A validation failure or
lost manifest race removes only the unpublished generation. Catalog `latest`
selection validates the complete v2 publication and will not fall back to v1.

Historical v1 layouts remain readable only when a caller explicitly passes the
family’s legacy-layout policy. V1 is not reinterpreted as v2.

## Implementation checklist

- [x] Extract one reusable exact Arrow declaration/envelope/footer core without
      changing canonical contract bytes.
- [x] Freeze all four baseline-strategy schemas and primary keys.
- [x] Freeze all three training-response schemas and primary keys.
- [x] Normalize status-dependent columns into closed row vocabularies.
- [x] Replace dynamic BIC structs with canonical JSON.
- [x] Emit exact physical zero-row parts.
- [x] Publish immutable generations with exact part receipts and manifest-last
      compare-and-swap.
- [x] Make strict queries resolve only manifest-selected v2 parts.
- [x] Make discovery fail closed on incomplete, ineligible, schema-tampered,
      digest-tampered, or missing publications.
- [x] Keep historical v1 reads behind an explicit compatibility argument.
- [x] Cover exact declarations, values, primary keys, empty parts, staged
      validation failure, content tampering, and lost manifest races.
- [ ] Add a cross-language consumer fixture only if either family becomes a
      Crimson-visible surface; they are currently Palette query products.

## Promotion boundary

This checkpoint activates v2 only for newly written family-local outputs. It
does not migrate historical v1 directories in place. A future migration tool
must read v1 explicitly, normalize into v2, publish a new immutable run ID, and
retain the source-manifest digest. It must never overwrite or relabel v1 bytes.

## Validation evidence

The checkpoint matrix passed 335/335 tests. It covered the canonical Arrow
contracts and registry, cross-recording exporter, both family workflows and
catalogs, exact schema/value/primary-key suites, immutable publication failure
recovery, explicit legacy boundaries, and the baseline-strategy QC component.
Static `py_compile`, Ruff, `git diff --check`, and strict-JSON publication also
passed. The ten emitted warnings are existing legacy completion-marker
warnings; none arose from these Parquet v2 contracts.
