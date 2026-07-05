# Brief: FrameDomains resolver — design document (NOT implementation)

**From:** commander session, 2026-07-05
**Deliverable:** `docs/frame_domains_resolver_design.md` (status: proposed) on branch
`agent/frame-domains-design`, plus your report. **Zero changes under `src/`.** This is
the design phase of identity review Rec 3 — the last open recommendation. The
implementation happens in later slices only after the maintainer approves the design.
**Read first:** `docs/identity_lineage_staleness_review.md` §3 (the four-domain analysis
and the resolver mandate — the doc this design executes),
`docs/palette_cli_narrow_waist_design.md` (the `Recording` accessor the resolver should
compose with), `docs/run_resolution_semantics.md`.

## The problem (from the review + the 2026-07-04 audit)

At least four frame domains coexist per recording, and every consumer re-derives the
conversions locally:
1. **Acquisition-local frame ID** — hardware clock; the review declares this canonical.
2. **Raw-video frame number.**
3. **Stored zarr index** — `original_frame_indices` + `frame_step` (subsampling).
4. **Crop-video frame index** — `source_crop_video_frame_indices`; may have DROPPED
   frames per the Orange contract, so it is not a contiguous subset.
5. (Quasi-domain found by audit) `-1` supplemental rows in
   `build_hybrid_acquisition_offline_crop_run.py` (~line 902) — rows with no source
   frame at all.

This bug class already produced RedScare (per-frame array sized to the wrong domain —
writer fixed in 662eea3, the class remains). Known re-derivation sites (verify and
extend this list; lines drift):
- `detection/detect_keypoints_yolo.py` `_resolve_full_image_shape` + local bincount
  fallback (~822-836) — the RedScare seam.
- `refinement/detect_quality.py` (~124-144) — own `frame_step`/`original_frame_indices`
  mapping logic.
- `capture/import_video.py` — the ORIGIN of `original_frame_indices`/`frame_step`
  (subsampling math ~line 94).
- `shared/crop_image_source.py` (~780-1063) — per-row
  `source_crop_video_frame_indices` resolution, own bounds checks.
- The four crop-run builders (`build_analysis_acquisition_crop_run`,
  `export_acquisition_crop_pose_training_zarr`, `append_acquisition_crop_video_training`,
  `build_hybrid_acquisition_offline_crop_run`) — each independently writes and
  annotates `source_crop_video_frame_indices` semantics.
- `visualization/detection_visualizer.py` (~1327) — local aliasing.

## What the design doc must decide (the load-bearing sections)

1. **Domain inventory as a contract.** Name each domain precisely, define its index
   space, its recorded mapping arrays, and which stores/arrays are expressed in it.
   Resolve whether the `-1` supplemental rows are a fifth domain or an "unmappable"
   marker within domain 4.
2. **Canonical-domain semantics.** The review says acquisition-local frame ID is
   canonical. Make that operational: every cross-domain conversion goes through it;
   define what happens for rows/frames with no canonical ID.
3. **The resolver API.** One per-recording object (working name `FrameDomains`),
   constructed ONLY from recorded mapping arrays — never inferred from array lengths or
   `max()+1` (the RedScare anti-pattern; be explicit that length-derived totals are
   forbidden). Specify: construction path (it should hang off the `Recording` accessor —
   e.g. `recording.frame_domains()` — not be a free function every caller wires up),
   vectorized to/from conversions between all domain pairs, per-domain `count()`
   (the authoritative answer to "how long is a per-frame array in domain X"), and
   fail-loud semantics for unmappable indices (error vs masked return vs sentinel —
   pick ONE policy, justify it, apply it uniformly).
4. **Missing-mapping degradation.** Old stores may lack some mapping arrays. Define
   explicitly what the resolver does per absent array: refuses those conversions
   loudly, never guesses identity mappings silently. A `capabilities()`-style
   introspection is acceptable; silent identity fallback is not.
5. **Ground-truth census (verify-against-data before baking semantics — standing
   rule).** Read-only survey of real stores under `/nvme1/recordings/<rec>/zarr/`
   (per-recording `*training.zarr` + `*analysis.zarr`): which mapping arrays actually
   exist, observed `frame_step` values, whether recordings with BOTH dropped crop
   frames and subsampling exist (the review's required validation case), and any
   contradiction between recorded arrays and the doc's domain definitions. If no
   drops+subsampling recording exists, say so — the implementation slice then needs a
   synthetic fixture, and the design must note it.
6. **Migration plan.** Resolver-first (introduce + validate, no consumers), then
   per-consumer slices in risk order with the specific local arithmetic each one
   deletes. Include the forcing function: how do we prevent NEW local frame-domain
   arithmetic after the resolver exists (import-linter can't see this — propose
   something honest, even if it's a grep-based CI check or a review rule).
7. **Non-goals.** No historical-data rewriting; no RedScare backfill (separate,
   optional); no changes to how `import_video.py` RECORDS the mappings (it's the
   producer; the resolver consumes what it writes — unless the census finds its
   recordings insufficient, which is a finding for the doc, not a fix).

## Doc conventions

contract-meta header (`status: proposed`, created 2026-07-05, owner jeremy, related:
`identity_lineage_staleness_review.md`, `palette_cli_narrow_waist_design.md`,
`run_resolution_semantics.md`). Match the register and structure of
`docs/provenance_finalization_enforcement_design.md` — decisions with justifications,
alternatives considered, a tiers/sequencing table. Cite file:line for every claim about
current code (they will be re-verified).

## Constraints

- Design doc + report only. No `src/` edits, no test edits.
- Census is READ-ONLY against `/nvme1/recordings/` — open zarr attrs/arrays, never
  write, never lock. If the path is unreachable from your sandbox, report that and
  design from the code's write paths instead, flagging the census as a pre-approval
  gap the maintainer must close.
- Where the identity review's §3 and observed reality disagree, reality wins — report
  the discrepancy, don't paper over it.

## Reporting

Branch `agent/frame-domains-design`, one commit (`docs: design frame domains resolver`).
Do not push. Report: the census results table, the 3-5 decisions you made that most
need maintainer eyes (with your recommendation and the alternative), any site-list
corrections vs this brief, and open questions blocking approval.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
