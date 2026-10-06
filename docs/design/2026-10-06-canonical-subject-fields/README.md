# One set of subject fields for every recording source

- **Status:** draft for review. Nothing is implemented yet.
- **Owner:** Jeremy Delahanty.
- **Last reviewed:** 2026-10-06.
- **Builds on:**
  - [recording identity](../2026-09-23-recording-identity/README.md);
  - agent-contracts PR 52 (Citrus subject identity v3);
  - agent-contracts PR 53 (Orange subject references);
  - agent-contracts PR 54 (MetaZebrobot consumers).
- **Why now:** no real Orange subject-reference recording or Citrus v3 recording has been imported yet. Today the new key variants exist only in code and tests, not in the store, so changing them now costs no migration.

## Goal

A subject fact has one name, one type and one meaning, whichever source it came from:
- a legacy H5;
- a Citrus unified H5;
- an Orange MetaZebrobot reference;
- a manual assertion or backfill.

Consumers such as the registry, cohorts, training data cards and dashboards read only those names. Each source owns exactly one translation into them.

## Current state (Palette main, 2026-10-06)

### One container, source-shaped contents

Every writer publishes through `fisheye.shared.subject_metadata.publish_subject_metadata` into one immutable record type (`palette.subject_metadata.v1`) under `analysis/subject_metadata_runs/`, and one resolver reads it. The identity rule is also shared: `subject_ids` comes from `subject_ids`/`fish_ids`, then `subject_id`, then legacy `fish_id` (#257).

The record's `subject_metadata` mapping, though, is whatever the source sent. The writers:

| Writer | Where | Shape of `subject_metadata` |
|---|---|---|
| Legacy H5 import | `analysis/import_stimulus_to_zarr.py:3109` | `/subject_metadata` attrs as strings |
| Unified H5 projection | `utils/import_recording_analysis.py` (`project_unified_subject_metadata`) | `/metadata/subject` attrs: strings plus int64 revisions |
| Orange reference | `utils/import_recording_analysis.py` (`import_zebrobot_subject_reference`) | `zebrobot_subject_reference.resolve_subject_reference` output |
| Manual assertion | `utils/set_recording_subject_metadata.py:81-88` | hand-built mapping |
| Count-only migration | `utils/migrate_count_only_subject_context.py` | hand-built mapping |
| Backfill | `utils/backfill_subject_context.py`, `utils/backfill_subject_experiment_setup.py` | hand-built mapping |

### The same fact under different names and types

| Fact | Legacy / Citrus H5 | Orange reference | Manual / backfill |
|---|---|---|---|
| Age (dpf) | `days_post_fertilization`: string, can be `"invalid_format"` / `"invalid_future_date"` | `dpf_at_recording` (int) | `dpf_at_acquisition` and `days_post_fertilization` (int) |
| Fertilization date | `date_of_fertilization` | `dof` (`YYYYMMDD`) | `date_of_fertilization` |
| Dish revision | `dish_revision` (int64, Citrus v3) | `dish_revision_at_recording`, `dish_revision_at_intake` | — |
| Dish population | `fish_count` (string), `source_dish_population_count` | `source_dish_population_count` (int) | — |
| Parents | `"8750_M08_B2 [unknown]"` (string) | `[{identifier, sex}]` | — |
| Subject count | `"1"` (string) | int | int |
| Group subject type | `dish_group` (Citrus v3) | — | `group` (`set_recording_subject_metadata.py:85`) |
| Lookup status | `citrus_subject_lookup_*` / `citrus_fish_reference_*` root attrs (#265) | `orange_subject_reference_status` / `_reason` root attrs | — |

### Translation already happens, downstream and piecemeal

The registry has its own tolerant reader, `registry/db.py:_extract_provenance` (`:1012`) and `_normalize_parents` (`:549`):
- It accepts dish and cross fields both nested and flat.
- It accepts parents as a string, a list or JSON.
- It falls back to a legacy `analysis_metadata` `zebrobot_snapshot` attr (`:688-700`).

Its gaps:
- **Age.** It takes `dpf_at_acquisition` from `days_post_fertilization` only (`:1031`), falling back to `dpf_at_acquisition` / `dpf` (`:3101`). **Orange's `dpf_at_recording` is not read, so a recording-only Orange session reaches the registry without an age.**
- **Identity.** It reads `fish_id` for the provenance identity (`:1022`), not `subject_id`.

Every new consumer either repeats these tolerances or gets them wrong.

## Proposal

### 1. Canonical fields

Names are chosen from what consumers already read, to keep churn low. All are optional unless the source declares them.

| Field | Type | Meaning |
|---|---|---|
| `subject_ids` | list[str] | Citrus-local subject ids (never MetaZebrobot fish ids) |
| `subject_type` | `individual` \| `dish_group` | `group` is read as `dish_group` |
| `subject_count` | int ≥ 1 | operator-declared |
| `species`, `sex`, `genotype`, `line_strain`, `cross_id` | str | as served or declared |
| `parents` | list[{`identifier`: str, `sex`: str\|null}] | the `_normalize_parents` rule, moved upstream |
| `date_of_fertilization` | str `YYYYMMDD` | |
| `dpf_at_acquisition` | int | days from fertilization to the recording date; never the server's same-day `dpf` |
| `dish_id`, `dish_uuid` | str | |
| `dish_revision` | int | the revision **at recording** |
| `dish_updated_at` | str | verbatim server string |
| `source_dish_population_count` | int | fish in the source dish; **not** the subject count |
| `mzb_fish_id`, `mzb_fish_revision`, `mzb_fish_updated_at` | str, int, str | verified only when `fish_reference.status == collected` |
| `subject_lookup` | {`status`, `reason`} | `collected` \| `not_collected` \| `lookup_failed` |
| `fish_reference` | {`status`, `reason`} | same vocabulary |
| `zebrobot_service` | {`service_commit`, `service_commit_dirty`, `consumer_schema_sha256`, `matches_pin`} | the build that served the dish, when sealed |
| `intake_observations` | {`dish_revision_at_intake`, `dish_changed_since_recording`, `dish_fish_changed_since_recording`, `cross_fields_resolved_at`} | what Palette saw at intake; never mixed into the facts above |

`dpf_at_acquisition` is computed once, by Palette, from `date_of_fertilization` and the recording date. A source value that can't be parsed is not coerced: the field is absent and the reason is recorded.

### 2. One translator per source, kept next to its raw input

`palette.subject_metadata.v2` records hold three parts:
- `subject` — the canonical fields above, and the only part consumers read;
- `source_attributes` — the source mapping, byte-for-byte as today's `subject_metadata`, for audit;
- `source` — which translator ran, plus its input pins (H5 reference digest, Orange schema digest, MetaZebrobot pin).

Each source has exactly one translator in `fisheye.shared.subject_metadata`:
- `from_legacy_h5_attrs`;
- `from_citrus_v3`;
- `from_orange_reference`;
- `from_manual_assertion`.

The translators are pure functions. Writers call one, and none builds a canonical mapping by hand.

### 3. Statuses move into the record

`subject_lookup`, `fish_reference` and `zebrobot_service` live in the record instead of the `orange_*` / `citrus_*` root attrs. A session with a declared absence (no dish) publishes a record whose `subject` holds only the statuses. The bound-recording rule is unchanged: the H5 record is authoritative, and an Orange reference cross-checks it (`dish_uuid`, else `dish_id`). When the H5 declares no dish, the Orange reference supplies the record.

### 4. Readers

- `resolve_subject_metadata` returns `subject`. For a stored v1 record it runs the matching translator on the fly. Stored v1 records stay valid and are never rewritten (the immutability rule), so no migration is needed.
- `registry/db.py:_extract_provenance` reads canonical fields only. Its tolerances move into the translators.
- The legacy `zebrobot_snapshot` fallback stays read-only, behind the legacy-H5 translator.

### 5. Parity test (the acceptance gate)

One fish, from one dish, recorded on one date, arrives as:
- a legacy H5;
- a Citrus v3 H5;
- an Orange v2 reference.

All three must produce identical `subject` fields, and identical registry rows, `dpf_at_acquisition` included. A new source adds its case to this test before it may publish.

## Rollout

1. Write the canonical fields and translators, and switch `resolve_subject_metadata` to return them, translating v1 on read. Add the parity test.
2. Switch writers to publish v2: Orange and Citrus v3 first, since nothing in the store uses them yet; then the manual and backfill CLIs.
3. Switch the registry and other readers to canonical fields, and delete their local tolerances.
4. Retire the `orange_*` / `citrus_*` root status attrs once nothing reads them. They are only on main since #255/#265, and no production recording carries them yet.

## Open questions

1. `subject_type` for a group: adopt `dish_group` (the Citrus contract) and read the manual CLI's `group` as `dish_group`, or the other way round? Recommended: `dish_group`, because Citrus's contract is pinned.
2. Should `sex`, `genotype` and similar come from MetaZebrobot at intake (Orange) while a Citrus H5 carries its own copy? Recommended: the H5 copy is authoritative for bound sessions, as now. Intake only adds `intake_observations` when the two differ.
3. Should the canonical field list itself become an agent-contracts document, so that Orange and Citrus can check against it? Recommended: yes, after step 1 lands, as a Palette-owned consumer schema.
