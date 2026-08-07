# Recording subject trait contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-08-07
-->

## Purpose

Palette distinguishes a strain's expected reference phenotype from an animal's
observed phenotype. Strain expectations are reusable defaults; recording-scoped
observations override them without rewriting biological strain identity.

Three normalized tables participate:

- `strain_label_mappings` maps one exact source husbandry label to a canonical
  strain. It does not parse stock, colony, or cohort semantics.
- `strain_trait_expectations` stores expected reference traits for a canonical
  `(species, strain)` pair.
- `recording_subject_traits` stores observations keyed by
  `(recording_id, subject_id, trait_name)`.

`recording_subject_trait_resolved` applies the order
`subject observation -> strain expectation -> unknown` and reports
`value_origin` as `subject_observed` or `strain_expected`. All assignments carry
method, actor, time, and evidence provenance.

## Pigmentation vocabulary v1

The overall trait is `pigmentation_phenotype`. Its vocabulary ID is
`palette.pigmentation_phenotype.v1`, with values:

- `wild_type_pigmented`
- `hypopigmented`
- `amelanotic`
- `hyperpigmented`
- `transparent`
- `altered_pattern`
- `mosaic`
- `other`
- `unknown`

Pigment-cell status is represented independently for `melanophore_status`,
`xanthophore_status`, and `iridophore_status` using
`palette.pigment_cell_status.v1`:

- `normal`
- `reduced`
- `absent`
- `increased`
- `altered_distribution`
- `unknown`

`pigment_pattern_status` uses `palette.pigment_pattern_status.v1`:
`wild_type`, `altered`, `mosaic`, or `unknown`.

`optical_transparency` uses `palette.optical_transparency.v1`:
`normal`, `partially_transparent`, `transparent`, or `unknown`.

These axes intentionally avoid treating transparency, absence of melanin, and
absence of all pigment-cell classes as synonyms.

Absence from the resolved view means unknown coverage. Dataset/model cards
must report missing coverage and distinguish observed values from inherited
strain expectations.

## Identity boundary

- `species`, canonical strain, source husbandry label, and genotype remain
  separate biological identity/lineage fields.
- Source labels are preserved verbatim. A mapping to a canonical strain is an
  explicit assertion, never a parser heuristic.
- Stock, colony, and cohort fields remain withheld until their source-system
  semantics are defined.
- A strain expectation must not be described as an observation.
- Traits are recording-scoped so later observations do not silently rewrite
  historical acquisition context.
- Updating the registry trait does not mutate an analysis or training Zarr.
  Future published training/model artifacts must bind the resulting data-card
  snapshot and manifest digests; a live registry lookup is not model identity.

## AB reference and Batman observation

For *Danio rerio* AB, Palette records the expected reference traits
`wild_type_pigmented`, normal melanophores/xanthophores/iridophores, wild-type
pattern, and normal optical transparency. ZFIN identifies AB as a wild-type
line (`ZDB-GENO-960809-7`).

The exact Batman source label `AB [AB IC] SEPT25` maps to canonical strain
`AB`; the bracketed and dated suffix remains uninterpreted. The project
operator visually confirmed typical wild-type pigmentation for all 36 Batman
recording subjects, so those six resolved traits are also recorded as
`subject_observed`, not merely inherited expectations.

Validate the versioned allocation against a registry without writing:

```bash
scripts/py -m fisheye.utils.apply_recording_subject_trait_allocation \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --allocation docs/diagnostics/batman_keypoint_recording_allocation_20260807.json
```

After the migration is deployed and the dry run passes, add `--apply` to write
the exact strain mapping, six AB expectations, and 216 recording-subject
observations in one transaction. Repeating the command is idempotent.
