# Group-statistics Arrow contract checkpoint — 2026-08-03

Status: implementation candidate in an isolated worktree. This checkpoint does
not change production selectors, publish an export, or promote an authority.

## Scope

This checkpoint closes the physical and semantic contracts for the two tables
written together by the GoodCopBadCop group-statistics publisher:

- `group_statistical_summary`
- `group_descriptive_summary`

The ordered Arrow declarations in
`fisheye.analytics_exports.arrow_contracts` are executable authority. The
statistical table has 45 fields and the descriptive table has 30. Both use an
exact schema, exact footer metadata, and a digest-bound Arrow-contract
envelope. `export_schema_version` is `int32`; row counts and iteration counts
are `int64`; scientific values are nullable `float64`; flags are `bool`; and
identities, enums, units, JSON, paths, digests, and timestamps are `string`.

Historical inferred group-statistics exports remain compatibility artifacts.
They are not silently accepted as the new exact contract. The viewer rejects
them by default and exposes them only through the explicit read-only
`--allow-legacy-statistics` option. Automatic selection still prefers every
valid exact publication over a newer legacy publication. Compatibility reads
must not be used as evidence that an exact publication passed.

## Scientific semantics

Every row now carries `metric_unit`. The closed metric registry rejects an
unregistered `(source_table, metric_name)` pair rather than guessing a unit.
The persisted `unit` value remains the statistical unit (`recording`), not the
measurement unit.

The wide statistical table keeps one `effect_size` and one CI pair but removes
their previous ambiguity:

| Test family | `effect_size_kind` | `ci_estimand` |
| --- | --- | --- |
| paired condition contrast | `paired_mean_difference_over_sample_sd` | `paired_mean_difference` |
| one-sample CRA contrast | `rank_biserial_correlation` | `one_sample_median` |

The validator binds these values to the contrast shape and accepted test
method. It also freezes the two statistical missing-data policies and the
descriptive policy. Descriptive rows use
`available_recording_values_by_condition`: source nulls are excluded before
per-recording condition aggregation, and non-finite aggregates are excluded by
the finite-value reducer before descriptive statistics are computed. The former
`complete_recording_values_by_condition` label incorrectly implied a
cross-condition complete-case requirement.

Null means unavailable, not zero. Non-finite values are rejected at the
publication boundary. A real zero remains valid, including an all-zero
one-sample contrast (`effect_size=0`, CI `[0, 0]`, `p_value=1`). An empty
descriptive group has `unit_count=0` and null aggregates; a singleton has null
sample standard deviation and SEM. A skipped inferential row retains counts
and any available condition means, has null unavailable inferential values,
and carries a non-null `skip_reason`.

## Identity and publication

`stat_result_id` and `descriptive_result_id` remain deterministic SHA-256
identities. Their payloads now include the source table and exact source export
manifest digest in addition to source run, metric, grouping, and contrast or
condition identity. Statistical identities also bind conditions, measurement
unit, effect-size kind, CI estimand, missing policy, and parameter JSON.

Before bytes are written, validation requires:

- canonical strict `group_key_json`;
- exact source-run, source-manifest-digest, and statistics-run binding;
- recomputed result IDs and uniqueness within each table;
- registered metric units;
- consistent nonnegative unit counts and iteration counts;
- accepted status, method, missing-policy, effect-size, and CI enums;
- finite scientific values and bounded p/q values;
- exact fields with no undeclared columns and no null required fields.

The publisher builds the PyArrow table with the installed schema rather than
inferring from observed values. Therefore an all-null column remains its
declared type instead of becoming Arrow `null`. The existing atomic generation
publisher still validates part hashes, row counts, exact footer declarations,
and the manifest-exclusive inventory before the manifest visibility commit.

The logical table-contract version for these two tables is 2. The physical
Arrow table version is 1 because this is their first exact physical schema.

## Verification checklist

- [x] Exact ordered fields, dtypes, and nullability are executable.
- [x] Exact footer metadata and schema digest are written.
- [x] Re-signed schema-envelope tampering fails.
- [x] Extra and missing required row fields fail.
- [x] Source-digest tampering and duplicate result IDs fail.
- [x] All-null inferential columns retain `float64`/`string` types.
- [x] The current group-statistics producer and generic export validator use
      the same installed contract.
- [x] Historical inferred-v2 reads require explicit opt-in and cannot outrank
      a valid exact publication during automatic selection.
- [x] Independent diff review accepted the frozen lane after an authoritative
      suite of 158 passing tests and 5 expected xfails.
- [x] Representative exact publication, strict viewer semantics, explicit
      legacy opt-in, exact-first selection, and adversarial read evidence are
      covered by the focused suite.
- [x] The reviewed lane is committed as an isolated checkpoint.
- [x] The independently reviewed checkpoint was serialized into the integration
      branch as `f7d02e9e` without production publication or selection changes.
