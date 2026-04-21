# Repo-Wide Staleness Gap Matrix

<!-- design-meta
status: draft
last_updated: 2026-04-06
-->

## Purpose

Summarize the main gaps between:

- current shipped behavior
- current local docs/contracts
- the desired repo-wide staleness policy

This is not only a docs drift audit. It also captures missing implementation
and places where one stage family has a stronger contract than another.

For the prioritized implementation sequence, see
[repo_wide_staleness_implementation_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_implementation_todo.md).

## Reading Guide

Gap types used below:

- `aligned`: current behavior is basically consistent with the desired policy
- `doc drift`: docs and code disagree
- `missing implementation`: policy direction is clear but the canonical
  implementation does not exist yet
- `registry gap`: state exists locally but is not projected into registry/query
  surfaces
- `contract ambiguity`: code works, but the repo has not yet defined the
  intended long-term contract clearly enough

## Matrix

| Area | Current behavior | Desired contract | Gap type | Priority |
| --- | --- | --- | --- | --- |
| Detect raw vs refined | Raw `detect_runs/<run>` is append-only; the canonical curated surface lives at `refined_detect_runs/<run>/instances`, with `source_detections/` as the audit surface and sparse legacy subgroups retained only for historical/provenance paths. Downstream crop should resolve `instances/` first under `refined` or `auto`. | Keep raw detect immutable and keep refined detect as the canonical curated surface. | aligned | low |
| Detect manual correction -> downstream stale | Manual corrections affect downstream crop/keypoints/masks operationally, but there is no canonical `source_detect_stale` payload or row-level downstream stale contract for detect edits inside an existing refined run. | Define whether detect-in-place corrections produce explicit downstream stale payloads or are treated as rerun-only events. | missing implementation | high |
| Crop as an upstream change source | Crop has strong lineage attrs (`source_detect_run`, `source_crop_run`, signatures/revisions), but no canonical `source_crop_stale` payload or explicit downstream stale contract. | Either formalize crop as rerun-only derived geometry, or define a bounded `source_crop_stale` contract for stable row-local crop updates. | contract ambiguity | high |
| Runtime cascade invalidation | New-run invalidation is clearly scoped and implemented: new runs mark downstream steps `missing`. | Keep cascade for new-run identity changes only. | aligned | low |
| Keypoint correction -> eye-mask stale | Eye-mask downstream staleness is first-class: `source_keypoint_stale` exists, includes reason/indices, and has explicit resolution. | Keep this as the canonical precedent for source-drift handling. | aligned | low |
| Keypoint correction -> subject-mask / swim stale | Refined subject/swim rows can be checked, auto-synced, or marked stale locally, but the canonical top-level `source_subject_mask_stale` payload is not written. | Mirror the eye-mask pattern with top-level stale payload plus row-local queue details. | missing implementation | high |
| Subject/swim stale -> registry | Registry extraction already knows how to read `source_subject_mask_stale`, but current stale state lives only in component attrs like `source_row_stale` and `source_update_pending_rows`. | Project subject/swim stale into registry-backed lifecycle and query surfaces. | registry gap | high |
| Subject/swim stale reviewer selection | Swim stale review currently bypasses the registry and scans zarr attrs directly for pending rows. | Registry-backed stale selection should work for run-level discovery, with local attrs remaining the detailed row queue. | registry gap | medium |
| Review state vs stale state | Eye masks keep explicit stale separate from review. Subject/swim currently mark stale rows and then flip component review to `needs_review`. | Keep stale separate from review across all stage families. | missing implementation | high |
| Explicit stale resolution for subject/swim | Eye masks have `resolve_eye_mask_stale`; subject/swim have no equivalent top-level resolution path yet. | Add a first-class subject-mask stale resolution action that preserves stale evidence while acknowledging accepted preserved edits. | missing implementation | high |
| `manual_override` / preserve semantics | Subject/swim preserve behavior works, but `manual_override` is still bootstrapped from `edit_applied` or current source difference for legacy runs. | Add an explicit curated/preserve-on-source-update bit that is semantically separate from `edit_applied`. | contract ambiguity | medium |
| Swim-bladder coarse source cache | Coarse swim-bladder masks can now refresh in place and feed refined stale review, but that cache role is not yet encoded as a canonical repo-wide stage contract. | Keep coarse swim bladder as a refreshable non-curated cache, clearly distinguished from refined curation authority. | contract ambiguity | medium |
| Eye masks vs subject masks vs swim bladder | The new docs now distinguish them conceptually, but the codebase still has stronger stale contracts for eye masks than for unified subject-mask components. | Unify stale vocabulary while preserving stage-family-specific operational differences. | missing implementation | medium |

## Main Conclusion

There are only a few pure docs-vs-code drift issues here.

Most of the important gaps are structural:

- detect and crop do not yet have a canonical within-run stale contract
- subject/swim local stale handling exists, but is not yet canonical or
  registry-native
- stale and review are still partially conflated outside the eye-mask path

## Near-Term Priority Order

1. Add top-level `source_subject_mask_stale` writing for refined subject-mask
   stale events.
2. Add explicit subject-mask stale resolution, analogous to eye masks.
3. Keep stale separate from review for subject/swim.
4. Decide the repo-wide policy for detect/crop corrections inside an existing
   run:
   - targeted downstream stale
   - or rerun-only
5. Project stale state into registry-backed selection/query paths wherever a
   canonical stale payload exists.

## References

- [repo_wide_staleness_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_policy.md)
- [repo_wide_staleness_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_checklist.md)
- [keypoint_late_correction_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_late_correction_contract.md)
- [refined_subject_mask_staleness_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_mask_staleness_todo.md)
