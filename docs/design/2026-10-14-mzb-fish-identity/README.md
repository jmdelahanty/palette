# MetaZebrobot fish identity in the Palette registry (Phase A)

Status: design draft, 2026-10-14. No code yet. Base: `origin/main` d81b2ee4.

## Problem

Palette admits Citrus zebrobot snapshot v3 from each camera H5 and carries
`mzb_fish_id`, `mzb_fish_revision`, `dish_uuid` and `dish_revision` into the
zarr subject record (`fisheye.shared.subject_fields.from_h5_attributes`). The
registry then drops them: `upsert_subject_snapshot_entities`
(`src/fisheye/registry/db.py:3045`) writes `subjects`, `dishes` and
`recording_subjects` with no MetaZebrobot (MZB) fish id, no dish UUID and no
revision. "All recordings of fish Z5" cannot be answered from the registry.

## Decisions this design rests on

- **Single id (Jeremy, 2026-10-14).** For an individual fish the only identity
  is the MZB `fish_id`, fixed at record start. In Citrus subject identity v4,
  `subject_id == mzb_fish_id` on every path, including outbox registration.
  Confirmed by Citrus (design d1b8610, amended 9ea82cd) and MZB (main 322bad7).
- **Palette is read-only for identity.** It never creates MZB fish.
- **No separate `fish` table.** Under v4 a `subjects` row for an individual
  *is* the fish. v3 recordings (Citrus-local `subject_id` beside
  `mzb_fish_id`) are few ("tens, not hundreds" before the v4 cutover) and stay
  as recorded; they join on `mzb_fish_id`.
- **The H5 is the as-of truth.** MZB has no as-of reads, so the revision a
  recording saw is only known from the sealed snapshot. The registry stores it
  per recording and never refreshes it from MZB.

## Change classification

Additive registry schema change plus a writer change. Existing columns, rows
and ids keep their values; no subject id is rewritten. The new columns are
populated only from sealed recording evidence.

## Schema (one additive migration, next free number after 076)

| Table | New column | Meaning |
|---|---|---|
| `subjects` | `mzb_fish_id TEXT` | MZB fish this subject is. Equals `subject_id` for v4 individuals; differs for v3. Non-unique index (v3 rows share a fish). |
| `recording_subjects` | `mzb_fish_id TEXT` | Fish recorded in this recording. |
| | `mzb_fish_revision INTEGER` | Fish revision the recording sealed. |
| | `mzb_fish_updated_at TEXT` | As sealed. |
| | `fish_reference_status TEXT` | `collected` (verified) or the declared status; an operator-declared id under `lookup_failed` is stored but marked unverified. |
| | `dish_uuid TEXT` | Dish UUID the recording sealed. |
| | `dish_revision INTEGER` | Dish revision the recording sealed. |
| `dishes` | `dish_uuid TEXT` | Immutable MZB dish UUID. |

Plus a read-only view `fish_recordings` over `recording_subjects` keyed by
`mzb_fish_id`, joined to `recordings` for rig/arena/camera, for per-fish
timelines inside Palette.

## Writer rules (`upsert_subject_snapshot_entities`)

1. Read the new values from canonical subject fields only (one name per fact).
2. Write `mzb_fish_id` only when `subject_type == "individual"`, exactly one
   subject id, and `subject_count == 1`. Any other shape carrying an MZB fish
   id is refused.
3. `recording_subjects` values are as-recorded and immutable. Re-registering
   the same recording with a different fish id, revision or dish UUID raises;
   it is never `COALESCE`-overwritten.
4. `subjects.mzb_fish_id` is set once. A later conflicting value raises.
5. `dishes.dish_uuid` is set once per `dish_id`. A different UUID for the same
   human `dish_id` (a re-created dish) raises rather than merging two dishes.
   Rekeying `dishes` by UUID is out of scope.
6. No MZB call at registry time.

## Existing data

The live registry is written only through `register_from_root` /
reconcile on the affected recordings, after Jeremy approves that write.
Before then: list which v3 recordings with `mzb_fish_id` are registered
(the 2026-10-09 snow session has 3 of 12 files with fish ids; Citrus expects
tens more before v4), and back up with `scripts/backup_palette_registry.sh`.

## Tests

- v3 individual: Citrus-local `subject_id`, distinct `mzb_fish_id`; rows and
  view correct; two v3 recordings of the same fish join on `mzb_fish_id`.
- v4-shaped individual: `subject_id == mzb_fish_id`.
- `not_collected` / `dish_group`: no fish id written.
- Refusals: fish id with count != 1; conflicting re-registration; conflicting
  `subjects.mzb_fish_id`; conflicting `dish_uuid` for one `dish_id`.
- Migration on a copy of a pre-migration registry; integrity via
  `scripts/py -m fisheye.utils.registry_integrity`.

## Out of scope

- Phase B: Citrus v4 admission (needs the agent-contracts v4 contract, owed by
  Citrus, and MZB's history table and idempotent POST). v4 admission should
  refuse an individual whose `subject_id != mzb_fish_id`.
- Phase C: experiment usage records in MZB (open; Jeremy leans to Palette
  confirming at intake).
- Group fish ids: see `group_fish_ids_open_question.md`.
