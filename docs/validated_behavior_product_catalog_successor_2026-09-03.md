# Validated-behavior product catalog successor — 2026-09-03

## Decision

Validated-behavior cohort exports and the products derived from them should be
physically easy to discover as one dataset package. They must not, however, be
collapsed into one mutable publication directory or treated as authoritative
because they happen to be nearby.

The exact cohort export remains under its existing immutable `publication/`
root. Derived products are published as immutable siblings beneath
`products/validated_behavior/v1/`. A versioned product catalog provides the
reverse edge from one exact export manifest to the exact product manifests.

The catalog is a non-authoritative discovery index. Scientific authority
remains with the source export manifest and each product's own manifest,
schemas, receipts, and payload digests.

## Package layout

```text
<cohort-package>/
├── publication/                                  # unchanged cohort export
│   └── validated_behavior/v1/...
└── products/
    └── validated_behavior/
        └── v1/
            ├── behavior_distribution/
            │   └── run_id=<distribution-run-id>/
            │       ├── manifest.json
            │       └── *.parquet
            ├── behavior_distribution_report/
            │   └── run_id=<report-run-id>/
            │       ├── manifest.json
            │       ├── index.html
            │       └── *.png
            ├── group_statistics/
            │   └── run_id=<statistics-run-id>/...
            ├── group_statistics_report/
            │   └── run_id=<report-run-id>/...
            └── catalog/
                ├── manifests/
                │   └── export_run_id=<export-run-id>.json
                └── .generations/
                    └── export_run_id=<export-run-id>/
                        └── generation=<catalog-generation-id>/catalog.json
```

Nothing is added below the closed export generation. Existing source exports,
source products, Zarrs, selectors, and registries are not mutated.

## Catalog contract

Each product entry records:

- product kind and exact run ID;
- canonical package-relative product and manifest paths;
- product schema, version, and status;
- manifest byte size, file SHA-256, and record SHA-256;
- exact source export run ID and export-manifest record SHA-256;
- the exact parent-product identity for report products;
- a self-digest over the complete entry.

Each catalog generation records the exact source-export manifest and validation
receipt identity, the sorted product roster and its digest, and the previous
catalog generation. Updates are append-only. An existing `(product_kind,
product_run_id)` key can be reused only when the entire immutable entry is
identical.

Catalog generations are committed with an advisory lock and manifest
compare-and-swap. The selected catalog manifest and its retained immutable
generation must be identical. A lost concurrent update fails rather than
dropping another product.

The v1 catalog accepts these independent product lineages:

- `behavior_distribution`;
- `behavior_distribution_report`;
- `group_statistics`;
- `group_statistics_report`.

The type registry is the extension point for later validated products. Adding a
new kind requires a product reader and an explicit source-export or
parent-product binding; the catalog does not infer scientific relationships
from filenames.

## Discovery behavior

The distribution explorer retains direct exact access:

```bash
scripts/run_validated_behavior_distribution_explorer.sh \
  --distribution-dir /exact/distribution/generation
```

It also accepts the source export identity:

```bash
scripts/run_validated_behavior_distribution_explorer.sh \
  --export-root /exact/cohort-package/publication \
  --source-export-run-id exact-export-run-id \
  --distribution-run-id exact-distribution-run-id
```

The distribution run ID may be omitted only when the selected catalog contains
exactly one compatible distribution. Zero matches and multiple matches fail
closed. Discovery never scans or globs product directories and never selects a
lexical, timestamp, or filesystem “latest” product.

The catalog resolver verifies the small catalog, export-manifest, and product-
manifest bindings. The selected distribution reader then performs its existing
payload validation exactly once. Catalog lookup does not rescan the 6.6 GB
source cohort export.

An already-open `ValidatedBehaviorExportDataset` exposes the same relationship
through `dataset.products(...)` and `dataset.product(...)`. These methods return
catalog-selected handles; they do not join, copy, or reinterpret scientific
tables.

## Publication and adoption

New distribution computation can omit `--output-dir` with `--apply`. The
producer then writes the canonical co-located directory and appends it to the
catalog after strict reopening. An explicit noncanonical `--output-dir` remains
available for sandbox work and is reported as uncataloged.

The same rule applies to new static distribution reports. A report can enter
the catalog only after its exact source distribution has been cataloged, and
the report's recorded source path and manifest digest must identify that
co-located parent.

The one-time adoption utility handles products that were safely published
before this layout existed:

```bash
scripts/py -m fisheye.utils.publish_validated_behavior_product \
  --export-root /exact/cohort-package/publication \
  --source-export-run-id exact-export-run-id \
  --product-kind behavior_distribution \
  --source-product-dir /exact/existing/distribution
```

Without `--apply`, this fully validates the export and product and prints the
canonical target. With `--apply`, it copies exact bytes into a temporary
co-located directory, fully validates the copy, atomically exposes it, and then
publishes a new catalog generation. It never moves or rewrites the original.
The operation is idempotent when the canonical product and catalog entry are
already exact.

## GoodBatBadBat migration target

The source cohort export is:

```text
/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_phase_c_20260902_19a006cc/publication
```

The existing 16 MB distribution is:

```text
/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_distributions_20260902_0f223d30/distribution
```

No histogram recomputation is required. After required CI and merge, exact-byte
adoption should place it at:

```text
/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_phase_c_20260902_19a006cc/products/validated_behavior/v1/behavior_distribution/run_id=goodbatbadbat-validated-behavior-distributions-20260902-0f223d30-v1
```

Existing reports should not merely be copied because their manifests bind the
old absolute distribution path. They should be rerendered from the adopted
distribution into canonical report-product directories; this changes display
artifacts and their manifests, not histogram evidence.

## Implementation checklist

- [x] Define canonical co-located product paths outside `publication/`.
- [x] Define a closed, extensible product-kind registry.
- [x] Bind every product to the exact export manifest.
- [x] Bind report products to their exact cataloged parent products.
- [x] Publish append-only catalog generations with manifest compare-and-swap.
- [x] Retain immutable prior catalog generations.
- [x] Reject a missing selector when retained generations prove prior catalog
      history exists.
- [x] Reject duplicate keys carrying different evidence.
- [x] Reject missing, foreign, tampered, or ambiguous product evidence.
- [x] Add exact-byte, validate-before-and-after adoption for existing products.
- [x] Make new distribution publication co-located and cataloged by default.
- [x] Make new distribution-report publication co-located and cataloged by
      default.
- [x] Add catalog discovery to the Marimo distribution explorer and launcher.
- [x] Expose catalog-selected product handles from the lazy export reader.
- [x] Preserve direct `--distribution-dir` access.
- [x] Add focused tests for adoption, immutability, append-only history,
      ambiguity, foreign source evidence, and tampering.
- [x] Pass the neighboring unit and Marimo validation suite (111 passed, 15
      expected failures; Marimo structural check passed).
- [ ] Pass every required repository CI check.
- [ ] Merge and pass post-merge CI before operational use.
- [ ] Deploy the exact accepted commit to a commit-pinned cluster worktree.
- [ ] Dry-run and then adopt the existing GoodBatBadBat distribution.
- [ ] Rerender canonical full-evidence and central-99 reports from the adopted
      distribution.
- [ ] Confirm export-root discovery in the deployed Marimo explorer.
