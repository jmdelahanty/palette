# Open question: who mints ids for fish recorded in groups

Status: recorded 2026-10-14, deliberately deferred by Jeremy. Not blocking
Phase A or Phase B.

## Agreed

- Under Citrus subject identity v4, a `dish_group` recording names no fish at
  runtime: `subject_id` is null; the snapshot carries a count, a dish
  reference and optional traceable members (`placed_from_registered_units`
  provenance).
- Individual identities for fish in a group come later, after the tracking
  pipeline runs.

## The disagreement

**Citrus (citrus-agent, 2026-10-14):** group fish ids "are minted later after
tracking, by MZB, not Citrus".

**MetaZebrobot (metazebrobot agent, 2026-10-14):** the v4 principles say MZB
holds only animals that can be found again. A tracked individual inside a
group recording can change when tracking is rerun, so it must never become an
MZB fish. A post-tracking id becomes an MZB fish only if the animal is then
physically singled out (for example isolated into a well), and is registered
as a real fish with provenance "identified from recording S, track T".

## Palette's reading

These are two different things that the phrase "group fish id" covers:

1. **Track identity.** "Fish #3 in recording S, tracking run R." Recording-
   and run-scoped, can change on rerun. Lives only in Palette, bound to the
   tracking run that produced it.
2. **Re-findable fish.** An animal singled out after the recording. An MZB fish
   created by the normal creation path (rig or web UI), carrying a reference
   to recording S / track T.

Under either reading Palette does not create MZB fish; Palette may supply the
recording/track reference that a person or the creating client records.

## To decide when group work starts

- Confirm the two-kinds split with Citrus and MZB, and correct the Citrus
  wording "minted ... by MZB".
- Who creates the MZB fish in case 2 (web UI review step, rig, other).
- Schema for the "identified from recording S, track T" reference and how
  Palette's tracking-run identity is bound into it.
- Whether a track-to-fish link in Palette is a registry row, a review receipt,
  or both.
