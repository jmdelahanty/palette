# Identity, lineage, and staleness — design review

<!-- contract-meta
status: proposed
created: 2026-07-02
owner: jeremy
related: docs/palette_cli_narrow_waist_design.md, docs/stage_catalog_reality_gaps.md,
         docs/diagnostics/codebase_review_2026-07-01.md
-->

## Verdict

The lineage *mechanics* are strong — mandatory `source_crop_row_ids`, 1:1 fail-loud
subject-mask assembly, append-only step-status history, fail-closed completion markers.
The weakness is *semantic*: identity in this system rests on two fragile foundations,
**array position** and **wall-clock time**. Nearly every lineage incident of 2026-06/07
traces to one of the two:

| Incident | Root foundation |
|---|---|
| Crop cumulative-offset bug (rows assumed frame-sorted) | position |
| Training loader OOB frame indices pairing blank images with real labels | position |
| `CAP_PROP_FRAME_COUNT` final-batch drop | position (frame-count trust) |
| 16-row smoke run flagging the *approved* composed run as stale | time |
| Approval recorded in a markdown doc, invisible to tooling | time (no authority concept) |
| `bbox_norm_coords` carrying two meanings | naming (field-level identity rot) |

Three recommendations follow, ranked by leverage. All are **stamp-going-forward**
designs — none migrates existing data.

## 1. Authoritative-run pointers (do this first)

A run currently has two machine-readable properties: **recency** (timestamps,
latest-pointers) and **completeness** (markers). The property the science actually
keys on — **authority**, "this is the run to use" — exists only in prose (e.g. the
RedScare composed run "approved as of 2026-06-30" *in the canary plan doc*, followed by
a paragraph in the segmentation todo explaining that the review-ready surface is not
the latest run).

Because everything downstream resolves "latest," any smoke test or bounded apply
hijacks default resolution just by existing later — the exact mechanism behind the
false-stale finding on `refined_subject_masks_sam3_body_existing_eye_swim_*`.

**Design:** a per-stage authoritative-run pointer, set explicitly by review tooling,
read preferentially by everything else (staleness checks, downstream default run
resolution, the `palette plan` oracle, training exports).

- Storage: a small attr/group at the stage level (`<stage>_runs/.authoritative` or a
  registry column), carrying `{run, approved_by, approved_at, git_sha}`.
- Write path: a waist verb — `palette approve <recording> <stage> <run>` — making
  approval a stamped, auditable act (and fixing approval-lives-in-docs for free).
- Staleness redefinition: "my input's **authoritative** run changed," not "a newer run
  exists." Smoke runs become harmless: nothing makes them authoritative.
- Fallback: stages with no pointer resolve to latest-complete (current behavior), so
  adoption is incremental.

## 2. Content-derived instance keys (cheap insurance)

A detection instance is currently "row *i* of run X," with ancestry as parallel-array
pointer chains (mask row → crop row → detection index → frame). Every hop is
positional trust; reordering produces plausible wrong data (the crop bug), and no
stable identity survives a re-run.

**Design:** mint an instance key **once, at detect time** —
`(recording_id, acquisition_frame_id, detection_ordinal)` or a short hash additionally
covering the bbox — stored as one array on the detect run. Every downstream stage
**copies it, never recomputes it**. Consequences:

- Chain verification collapses from positional trust to O(1) equality checks; any
  reordering bug fails loudly at the first mismatched key.
- Cross-run/cross-version questions ("what happened to this detection?") become
  queryable.
- Cost: one array per run, threaded through the existing lineage-array discipline.

## 3. A single frame-domain resolver

At least four frame domains coexist: acquisition-local frame ID (hardware clock),
raw-video frame number, stored zarr index (subsampled; `original_frame_indices` +
`frame_step`), and crop-video frame index (`source_crop_video_frame_indices` — which
may have *dropped frames* per the Orange contract, so not even a contiguous subset).
The mapping data is faithfully recorded — but each consumer re-derives the conversion
arithmetic, which is where it keeps failing (loader OOB, frame-count trust,
`check_frame_gaps.py` existing at the repo root as archaeology).

**Design:**

- Declare **acquisition-local frame ID the canonical instant identifier** (closest to
  hardware truth; already flows through crop-video lineage).
- One shared `FrameDomains` resolver per recording — the *only* place domain
  conversion happens — constructed from the recorded mapping arrays, validated once
  against a recording with both drops and subsampling.
- Consumers call the resolver; per-consumer frame arithmetic is deleted as it's
  migrated. (Same narrow-waist move as the CLI, applied to frame math.)

## Smaller item: field-name identity rot

`bbox_norm_coords` carries full-frame-normalized values by contract but
crop-frame-normalized values in early RedScare acquisition-crop-video runs,
distinguished only by an attrs note. Current handling is correct (readers *refuse*
noncanonical semantics rather than guessing — `run_sam_subject_masks.py`), but the end
state must be rename-or-backfill (`bbox_crop_norm_coords` exists as the planned home).
One name must never keep two meanings permanently.

## Sequencing and relationship to other work

1. **Authoritative pointers** — smallest change, fixes staleness + review-state
   fragmentation + smoke-run pollution simultaneously; makes the `plan` oracle
   trustworthy. Wants the `palette approve` verb (narrow-waist design) and connects to
   provenance stamping (an approval carries who/when/git-sha).
2. **Instance keys** — one writer change (detect) + copy-through; converts a whole bug
   class from silent to loud. Complements (does not replace) the assertions landing in
   the silent-wrong-data slice.
3. **FrameDomains resolver** — most work; retires the bug class the silent-wrong-data
   slice patches case-by-case. Reasonable to defer until that slice's assertions have
   soaked.

Non-goals: no rewrite of existing runs, no migration of historical lineage arrays, no
change to the row-oriented storage model itself — positions remain the storage layout;
they just stop being the *identity*.
