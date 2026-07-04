# Instance identity, association, and tracking — architecture review & roadmap

<!-- contract-meta
status: proposed
created: 2026-07-03
owner: jeremy
audience: maintainer (forward-looking design; no code change yet)
related: docs/identity_lineage_staleness_review.md,
         docs/palette_cli_narrow_waist_design.md,
         docs/diagnostics/codebase_review_2026-07-01.md
-->

## Purpose

Assess how Palette maps **detection instances → keypoints → segmentation masks**, how it
handles **multiple subjects**, and what is required to support the future goal of
**multiple *interacting* subjects** that need identity tracking (including identity-swap
handling). Based on a four-agent read-only pass (instance→keypoint, instance→mask,
multi-subject/arena, tracking/identity) with file:line evidence, plus best-practice
judgment for multi-object tracking (MOT). No code was changed.

The audience is the maintainer, who is building this kind of system for the first time,
so the document explains the *why* and the *steps*, not just the findings.

---

## 1. The core finding: identity is encoded as *position*, never as a *thing*

Every stage carries instance identity as **array row position**, with parallel-array
"pointers" that are synthesized as `arange(N)` on the happy path. There is no
content-derived, persistent per-animal instance key.

| Boundary | How identity is carried | Evidence |
|---|---|---|
| detect → keypoints | `detection_indices = arange(N)`; pose model runs per-crop with `max_det=1 + argmax(conf)` — the pose↔detection binding is asserted, never checked | `tracking/crop.py:2843`; `detection/detect_keypoints_yolo.py:552-559,681` |
| detect → masks | `source_crop_row_ids = arange(N)`; component channels `(N,C,H,W)` co-located by row; pixel content chosen by `keep_largest_component` (area) + nearest-eye-keypoint (instance-blind) | `segmentation/infer_unet_subject_masks.py:1021`; `segmentation/subject_segmentation.py:60` |
| multi-subject → "tracking" | `track_id` is a dense relabel of `arena_id` (a fixed point-in-box test); no temporal state at all | `tracking/single_subject_per_arena.py:161-166`; `tracking/arena_assignment.py:682-694` |

The strongest identity anchor in the entire system is `(frame, within-frame-ordinal)` —
"which slot, in which frame" (`shared/refined_detect_curation.py:413-421`). That is the
complete identity model.

---

## 2. Why this is a *good* design for the current regime

Position-as-identity is the **correct and cheap** encoding when there is no ambiguity to
resolve. For the current regimes — one subject per view, or multiple subjects in disjoint
fixed sub-regions that cannot interact — position genuinely *is* identity, so the hard
problem is correctly avoided. Properties this buys:

- **Zero ID-switch risk** — with no cross-frame association, there is nothing to switch.
  Strictly more reliable than any probabilistic tracker for non-interacting subjects.
- **Deterministic, stateless, reproducible** — vectorized point-in-box; no association
  thresholds, no drift.
- **Real fail-loud lineage discipline** — row-length validation (`shared/row_lineage.py:72-91`),
  frame-alignment of `source_crop_row_ids` against crop frames (`:118-149`), and a 1:1
  fail-loud component merge (`refinement/assemble_refined_subject_masks.py:202-230`).
- **The arena/spatial-partition model is well-engineered for its stated boundary.**

The maintainer's "explicit sub-regions" instinct is **sound**, not naive: spatial
partitioning to make identity free is a standard, legitimate simplification in the
animal-tracking field. **Keep it as a permanent fast path for genuinely partitioned
experiments.**

---

## 3. The honest boundary: a wall, not a slope

Moving to interacting subjects does **not** degrade the current system into "less
accurate." There is a hard discontinuity, and every failure mode is either fatal or
silent:

- **Two subjects, one region, one frame → the stage aborts.** `conflict_policy` supports
  only `"fail"` → `TrackingConflictError` → arena-assignment errors out for the whole
  recording (`single_subject_per_arena.py:131-159` → `arena_assignment.py:833-868`).
- **A subject crosses regions → silent identity swap.** Crossing into a neighbor's ROI
  relabels it as that neighbor with no continuity check; leaving all ROIs sets
  `arena_id=-1 → track_id=-1` and the row is silently dropped downstream
  (`arena_assignment.py:679,694`; `analysis/track_kinematics.py:680`).
- **Overlapping ROIs → wrong pixels, confidently labeled.** When fish A and B touch, A's
  crop contains B's body; `keep_largest_component` may keep B's blob or a merged A+B blob
  and store it as instance A. The eye split can bind B's eye to A. Reported status:
  "assigned."
- **The one automated defense is disabled exactly when needed.** Temporal jump/blip QC —
  whose teleport signature *is* a swap detector — is skipped whenever `expected_count > 1`
  (`refinement/detect_quality.py:854-855`); the multi-detection flag (`4`) is advisory only.

There is no degraded-but-usable multi-subject mode in the current design. This is the wall.

---

## 4. Latent bug *today* (independent of the tracking roadmap)

**S1 — keypoint↔mask identity is assumed, not reconciled.** The eyes_union→left/right
split matches keypoints to masks by bare row index with only a row-count guard; it does
**not** compare `source_crop_row_ids` between the keypoint run and the mask run
(`refinement/assemble_refined_subject_masks.py:559-612`; `refinement/subject_eye_assignment.py:407-417`).
If the assignment keypoint run came from a different crop ordering, a reconciled crop set,
or a different detection subset with a coincidentally equal N, every eye is split using the
**wrong instance's** eye keypoints — silently, with a confident "assigned" status. The same
helpers are reused in `finalize_subject_masks.py`, so its eye path shares the gap.

*Fix (small, worth doing on its own merits):* before the eye split, reconcile
`source_crop_row_ids` between the keypoint run and the mask run (equality, not just row
count); fail-closed on mismatch. This closes the one place the otherwise-explicit lineage
discipline is dropped.

**S2 — identity arrays are optional in the fail-loud check.** `assert_row_lineage_sources_equal`
treats `source_crop_row_ids` / `detection_indices` as presence-optional
(`shared/row_lineage.py:437-452`; `detection_indices` is `required=False` everywhere). A run
pair lacking both collapses "instance identity" to "same frame + same row count." Two
detections in one frame are then distinguished only by row order, which nothing enforces to
be stable across stages. *Fix path:* fold into the instance-key work (§5.i) — once a real
key exists, make it required in the equality check.

---

## 5. What interacting subjects require — three primitives

These are independent build efforts that **stack**, with a dependency order: (i) enables
both (ii) and (iii).

### (i) A persistent per-animal instance key — the prerequisite

A content-derived token minted at **detect time** and *copied* (never regenerated as
`arange`) through crop → keypoint → mask. Shape suggestion: `(recording_id,
acquisition_frame_id, detection_ordinal)` or a short hash additionally covering the bbox.
This is **Rec 2 of `docs/identity_lineage_staleness_review.md`**, and this analysis
**upgrades it from "cheap insurance" to "the prerequisite"**: without it there is nowhere
to record "this row is track T" or "same fish as last frame." Everything else depends on it.

Steps:
1. Mint the key on the detect run as one new array (threads through existing lineage-array
   discipline).
2. Have crop/keypoint/mask writers **copy** it instead of synthesizing `arange`.
3. Make it required in `assert_row_lineage_sources_equal` (closes S2).
4. Chain verification becomes O(1) key-equality instead of positional trust; any reordering
   bug fails loud at the first mismatched key.

### (ii) Intra-frame disambiguation — "which fish in this ROI is mine"

Today's crop is a bbox that, under interaction, contains foreign fish; the segmenter is
semantic (no instance channel) and the disambiguators are area + per-row eye proximity.
Required additions:
1. **Pose:** `max_det=1` → `max_det=N`, then a pose→detection assignment step (IoU /
   Hungarian of candidate poses against the source bbox) instead of `argmax(conf)`.
2. **Body mask:** select the connected component nearest *this detection's* center/keypoints,
   not `keep_largest_component`.
3. **Eyes:** reconcile keypoint↔mask `source_crop_row_ids` before splitting (this is the S1
   fix, now doubly motivated).
4. **Cross-instance overlap:** detect when instance A's and instance B's `subject_body`
   claim the same pixels (nothing does today) and flag for review.

### (iii) Temporal association — real MOT (a new component, not a parameter)

Link instances across frames. This is the tracking-by-detection association problem, absent
today. Required: persistent cross-frame identity, a frame-to-frame cost + assignment solver,
a motion model, and swap detection/repair + track birth/death.

---

## 6. The reusable asset: the seam a tracker slots into

All four reviews independently identified the same seam. A real tracker fits as **a new
`tracking_runs` method** that writes the same row-aligned arrays (`track_ids`,
`frame_indices`, `source_row_indices`) where `single_subject_per_arena` writes today.
Consumers (`analysis/track_kinematics.py` via `load_tracking_ids`, `:2544-2609`) need **zero
changes** — detection upstream and kinematics downstream are untouched. Changing the *source*
of `track_id` is the cleanest possible seam in the codebase.

And Palette already computes **exceptional association features that are not yet wired into
any matching step**:
- **Keypoints + headings** (`analysis/track_kinematics.py`): position + orientation give a
  motion-consistent cost far better than bbox-IoU for elongated, fast-turning fish; a
  predicted heading is a strong gating signal through a crossing.
- **Subject masks:** mask-IoU and shape descriptors are the strongest discriminator when two
  fish overlap — the exact case bbox-IoU fails.
- **Crops (`crop_runs`):** the natural substrate for a lightweight appearance / re-ID
  embedding if swap-free long-session identity is later required.

The architecture is **tracker-ready at its interface** despite having no tracker. That is the
strength to build on.

---

## 7. Build vs adopt

**Adopt the association algorithm; build only the thin bridge to the row store.** Options:

- **ByteTrack-style association** (recommended first tracker): constant-velocity Kalman
  predict + IoU/distance cost + Hungarian, with the low-confidence second-pass trick.
  Dependency-light, faithful to reimplement against the row store. Strong default.
- **SLEAP** (worth a serious look): purpose-built for animals, tracking-by-detection with
  flow/identity models; its instance/keypoint data model maps naturally onto what Palette
  already produces, and it dovetails with the webKnossos annotation direction.
- **idtracker.ai-style global identity** (adopt selectively): appearance fingerprints to
  survive long crossings swap-free — the gold standard when swaps are the failure mode.
  Heavier; reserve for when swap-free long-session identity is the actual requirement.

**Spatial priors verdict:** sound where they apply (physically partitioned dishes), a dead
end as the identity mechanism for interacting subjects (a fixed ROI gives no signal exactly
when subjects share a region). **Demote spatial priors from *decider* to a *gating prior* in
the association cost** (cross-arena transitions high-cost); do not extend arena-assignment
into a tracker.

---

## 8. Sequenced roadmap — smallest first

No time pressure: the maintainer needs more labeling and RedScare cleanup before training
additional models, so this is forward-looking. Ordered by leverage and dependency:

1. **Mint the per-animal instance key (§5.i).** Foundational; unblocks everything;
   also hardens today's lineage (closes S2). Smallest change with the largest downstream
   payoff.
2. **Fix S1 now (§4).** Reconcile keypoint↔mask `source_crop_row_ids` before the eye split.
   A real latent bug today, independent of tracking; do it on its own merits.
3. **Introduce a second `tracking_runs` method** that writes the same arrays but derives IDs
   from association, and drop the `conflict_policy="fail"` invariant (two-per-region becomes
   legal input). Changes the *source* of `track_id`, nothing downstream.
4. **Re-enable + repurpose the temporal jump/blip QC as a swap flag** (§3, `detect_quality.py:854-855`).
   Emit per-transition confidence so swaps are *flagged in lineage* before auto-repair — fits
   the existing review-workflow culture and gives a human the hook. Cheap early win.
5. **Implement ByteTrack-style association** as the new method (centroid+heading cost,
   Kalman predict, Hungarian; arena as gating prior).
6. **Add mask-IoU to the cost** once masks are stable — highest-value overlap discriminator,
   already computed.
7. **Intra-frame disambiguation (§5.ii):** `max_det=N` + pose→detection assignment,
   component-by-proximity body selection, cross-instance overlap detection.
8. **Only if long swap-free identity is required:** appearance / re-ID head over `crop_runs`
   (idtracker-style), plus automatic swap repair.

Steps 1–2 are worth doing regardless of the tracking timeline (they harden current
correctness). Steps 3–4 make identity a real, swappable, reviewable output. Steps 5–8 are the
actual tracker, layered onto the seam without disturbing detection or kinematics.

---

## 9. Mental model to keep

**Today: identity = position** — correct and free when there is no ambiguity; a hard wall
the moment there is. The move to interacting subjects is **not** strengthening a weak tracker;
it is building the identity primitive that was never needed before, behind a seam
(`tracking_runs` row-aligned `track_ids`) that is already the right shape, using association
features (keypoints, headings, masks) the pipeline already computes but does not yet use.
