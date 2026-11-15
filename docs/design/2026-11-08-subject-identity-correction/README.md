# Correcting a recording's fish after a mis-pick

- **Status:** draft
- **Owner:** Palette intake session; Jeremy decides
- **Last reviewed:** 2026-11-08
- **Builds on:** [`2026-10-14-mzb-fish-identity`](../2026-10-14-mzb-fish-identity/README.md)
  (single MZB fish id; as-recorded columns immutable; manual linking)

## Problem

An operator who picks the wrong fish in the Citrus UI seals that fish into the
arena's H5. Citrus never edits a sealed H5, and Palette admits the H5 as the
as-of truth. Palette has no route today to say "this arena was really fish X".

| Recording | Arena | Sealed | Really | Palette state |
|---|---|---|---|---|
| 2026_10_17_15_57_07 | arena_3 | Z5 `784c1937` (also arena_1) | Z6 `387a61ca-e7be-4039-ac3d-5e47b6a32a53` (well-order rule, Jeremy 2026-10-19) | registered 2026-11-04 with the duplicate |
| 2026_10_16_17_28_25 | arena_3 / arena_4 | `5870527e` in both | not stated | not in the store |
| 2026_11_06_15_48_57 | arena_3 (Cam2010095) | Z5 `5a50397a-55fc-4c5b-931f-221d679e96ad` (also arena_2) | Z6 `a21ac2c1-6f7c-45e6-9ce8-bfaadf7398db` (19498_4:C2) | held in `staging_hold/`, not imported |
| 2026_11_06_17_16_17 | arena_3 | Z9 `f3fce668-c25c-43fb-b88a-43f4a9e85bdb` (also arena_4) | Z10 `efb06650-fbf7-4441-af16-c5734cb3d4b8` (19498_4:B3) | held in `staging_hold/`, not imported |

The 11-06 values were confirmed by Jeremy in citrus-agent's session and
relayed. Each correction record below must be attested by Jeremy himself.

The 2026-10-14 manual-linking design cannot express these cases. It links a
recording whose fish was never confirmed, and its resolution order puts
"sealed and registered" above any link. These fish were confirmed, and wrongly.

## Change classification

This is a scientific identity change, expressed additively. The sealed H5, the
as-recorded `subject_metadata_runs` record, the import receipts and the
registry's as-recorded `recording_subjects` rows keep their bytes. What changes
is the **effective** fish that consumers read, through one resolver that
reports where its answer came from.

## Already built: the refusal

`fisheye.intake.importing._require_distinct_individual_fish` (PR "intake: refuse a delivery that records one fish in two arenas", branch `agent/palette/duplicate-fish-refusal`, 445c0443)
refuses a delivery when one individual fish appears in more than one arena
(`delivery_duplicate_individual_fish`). It runs after the parents are imported
and before staging is retired or anything is registered. The as-recorded
archives stay for a correction to bind to. The staged session stays in place.
Without corrections it refuses every such delivery, so it is useful on its own:
a mis-pick can no longer register unnoticed, as 10-17 did.

## The correction record

One immutable JSON document per corrected arena, schema
`palette.subject_identity_correction.v1`:

| Field | Meaning |
|---|---|
| `binds.citrus_session_id` | `citsess_…` of the H5 |
| `binds.h5_sha256` | sealed H5 file digest |
| `binds.camera_serial`, `binds.arena_id` | the arena being corrected |
| `recorded` | `mzb_fish_id` and `mzb_fish_revision` exactly as sealed; must match the H5 on apply |
| `corrected` | `mzb_fish_id`, plus `dish_id` and well as served by MZB, and the fish revision served when checked (`GET /fish/{id}`, read only) |
| `reason` | `operator_mis_pick` (only value in v1) |
| `attested_by`, `attested_at_utc`, `evidence` | Jeremy's statement, verbatim, with where he said it |
| `supersedes` | the `correction_id` it replaces, or null |
| `correction_id` | `sha256:` of the RFC 8785 canonical bytes of the record without this field |

The rules:

- A correction never edits the H5, the as-recorded subject run or a registry
  as-recorded row. A wrong correction is superseded by a new record, not edited.
- `recorded` must equal what the H5 sealed, so a correction cannot bind to the
  wrong file.
- `corrected` must name an existing MZB fish in the same dish. A different dish
  is refused in v1; that would be a different kind of error.
- Correcting one arena of a duplicate is enough. The other arena keeps its
  sealed fish.

## Where it lives

**Recommended: inside the recording's analysis.zarr, plus a registry
projection.**

1. **Archive.** A new additive group,
   `analysis/subject_identity_corrections/<correction_id>`, holds the record
   with its digest. `resolve_subject_metadata` returns the as-recorded record
   unchanged. A new `resolve_recording_fish(root)` returns the effective fish
   and `source` (`sealed` or `correction:<id>`). Every per-fish consumer
   (cohort selection, exports, the MZB usage-link replay) moves to it.
   Check before building: adding a group after import must not invalidate
   the import receipt or the manifest seal. The seal covers
   `recording_manifest.json`, not the zarr tree, but this needs a test.
2. **Registry.** One additive migration (next free number) adds the table
   `recording_subject_corrections`: correction id, recording id, dataset id,
   recorded fish, corrected fish, attested by and at, supersedes. A view
   `recording_subjects_effective` overlays the newest unsuperseded correction
   on `recording_subjects`. Registration projects corrections from the
   archive. Per-fish queries read the view.
3. **The duplicate guard** reads `resolve_recording_fish`. A delivery whose
   *effective* fish are distinct is admitted.

**Rejected alternatives:**

- A file ledger alone under `/groups/.../jeremy/corrections/`. Every consumer
  would need to find it, and an archive copied elsewhere loses its correction.
- A new `subject_metadata_runs` run selected as latest, as
  `set_recording_subject_metadata` does for count-only repairs. That tool
  refuses sealed transfer-v2 recordings on purpose. It would also make the
  as-recorded record unreachable through the normal resolver.

## Order of operations for a held delivery

1. Jeremy attests each correction in the Palette session. A tool writes the
   record into a pending directory and checks `recorded` against the held H5
   and `corrected` against MZB.
2. With Jeremy's go, the session moves from `staging_hold/` to `staging/`.
3. Intake imports it. After the parents import and before the duplicate guard
   runs, intake publishes every pending correction whose `binds` match a
   parent into that parent's archive.
4. The guard sees distinct effective fish and admits the delivery.
   Registration projects the corrections.

For the registered 10-17 15_57_07, step 3 is a separate tool run on the four
archives. A registry reconcile follows, after a registry backup and Jeremy's
approval of that write.

## Out of scope

- Fixing the MZB usage links that list Z5 and Z9 twice. That belongs to Jeremy
  and fish-id-minting; the replay reads `resolve_recording_fish` once built.
- A Citrus start-time guard against one fish in two arenas. Citrus has
  proposed one; that is Jeremy's decision in Citrus.
- 10-16 17_28_25 until Jeremy says which fish was in the other arena.

## Tests (when built)

- A correction whose `recorded` differs from the H5 is refused, as is one whose
  `binds.h5_sha256` matches no parent.
- `corrected` that is unknown to MZB, or in another dish, is refused.
- A superseded correction is ignored, and the supersession chain is reported.
- The guard refuses with no correction and admits with one. As-recorded subject
  run bytes and import receipts are unchanged after a correction.
- The registry view returns the corrected fish and its source. As-recorded
  rows are unchanged. The migration is tested on a copy of the live registry,
  and integrity is checked with `fisheye.utils.registry_integrity`.

## Decision log

- 2026-11-08: drafted after Jeremy chose `staging_hold/` for the two 11-06
  deliveries and asked for this design and the duplicate refusal.
