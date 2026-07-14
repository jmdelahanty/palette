# Responsive-ring traversal figures

**Module:** `src/fisheye/visualization/chaser_ring_traversal.py`
**Tests:** `tests/unit/fisheye/test_chaser_ring_traversal.py`

Animates the fish moving through the **responsive rings** around an object, one entry at a
time, with every bout of the entry drawn as its own segment. It ties three findings into one
picture:

- **Rings** — the object is ringed by the `chaser_bout_response` distance bands. The **8–16 mm
  responsive shell** is shaded amber: that is where the steering dose-response peaks (+0.574 at
  10 mm, +0.551 at 14 mm) and where escapes fire (median onset 13.5 mm, drawn as a red dashed
  circle). The rings are the axis the whole analysis is measured on, not decoration.
- **Entries** — a "visit" is entering 15 mm and not leaving until the fish passes 20 mm. Each
  entry is drawn separately, because a median over 4–10 entries is noise; the honest object is
  the trajectory.
- **Bouts per entry** — each bout is its own segment. **Escape bouts (peak > 100 mm/s) are red
  and thick with a ★; ordinary bouts are thin blue.** In the animation each bout lights up as
  it is executed and stays lit, so the *set* of bouts builds up as you watch.

## Entry points

```bash
python -m fisheye.visualization.chaser_ring_traversal <archive.zarr> \
    --epochs post_event --out-dir <dir> \
    --gif <dir>/traversal.mp4 --gif-max-entries 10
```

- `render_ring_entries_png(scenes, meta)` — a static panel per entry (the reference sheet).
- `write_ring_traversal_gif(scenes, meta, out)` — the animation (`.mp4` preferred; `.gif` works).
- `collect_ring_entries(zarr)` — reuses `chaser_visit_trajectories.collect_visits` for the
  trajectories and canonical frame, then attaches the bout segments.

## Two frames, one per kind of epoch

**Static epochs (pre / post) — `collect_ring_entries`.** Object at the origin, arena centre
rotated onto +x, so pre and post entries — whose objects sit at different arena positions — land
on top of each other and are comparable. The wall is the grey arc it really is; since the object
sits ~7 mm from it, a wall-following fish sweeps through the rings for free, and that confound is
drawn rather than hidden.

**Training epoch (chase) — `collect_chase_ring_entries`.** The object *moves*, so a ring centred
on its median position would be a fiction. Here the **moving chaser is fixed at the origin and
the fish is drawn relative to it, rotated so the pursuit direction points +y** — the same
chaser-centric frame `chaser_escape_freeze` scores the escape/freeze response in (built from its
tested `chaser_frame_transform` / `_heading_angles_from_chaser`). The rings are literal
distance-to-chaser bands; there is **no fixed wall arc**, because nothing is fixed but the
chaser. The aggressive object is resolved from the CRA endpoint role codes (never index order),
falling back to chaser 0 with a note if no endpoint is present. In this frame the escapes read
directly: the fish darts *up and outward* (away from the pursuer) and is pulled back in.

The CLI dispatches on the requested epoch: `--epochs training_event` (or `chase`) uses the
chaser-centric collector, anything else uses the static one. **They cannot be mixed in one
figure** — the frames are different — so request them separately. Output filenames carry the
epoch (`..._ring_entries_<epoch>.png`), and the figure caption states which frame it is drawn in.

For the pre / training / post trio, run the tool three times:

```bash
for ep in pre_event training_event post_event; do
  python -m fisheye.visualization.chaser_ring_traversal <archive.zarr> \
      --epochs $ep --out-dir <dir> --gif <dir>/ring_traversal_$ep.mp4 --gif-max-entries 10
done
```

## Three things that are not cosmetic

1. **Bout segments come from the `chaser_bout_response` bout table** — the same table the escape
   counts come from — so the red ★ here and the escape rate in `chaser_escape_events` cannot
   disagree. If that component is absent, entries render with no bout segments (onsets only)
   rather than crashing.
2. **Each bout is attached to the entry whose frame window contains its onset**, and to no
   other. Pinned by `test_bouts_are_attached_to_the_entry_that_contains_them` and
   `test_a_bout_is_not_double_counted_across_entries`.
3. **The animation bounds each entry to `frames_per_entry` display frames** regardless of how
   long the fish lingered, so a 4 s approach and a 130 s wall-dwell take the same wall-clock.
   Long entries are subsampled (real xy indices are kept, so bout onsets still register); short
   ones play at full detail. Entries are ordered escape-first and capped at `max_entries`.

## The thing the data will show you

A **wall-pinned fish does not have discrete entries** — it lives next to the wall-adjacent
object, producing a few 150–265-bout lingering "entries". That is honest (the visit definition
is not wrong; the fish really is there), but it makes a hairball. `_entry_sort_key` floats the
legible, escape-containing, closer, shorter passes to the top, and the animation caps entries,
so both figures lead with the readable ones. For a clean demonstration, pick a mid- or
centre-dwelling fish (e.g. `2026-06-14T21-50-10Z_arena_3`: 10 discrete post entries, median
5 s, 26 escapes on the aggressive object and **0 on the inert one** — the red-avoidance,
visible by eye).
