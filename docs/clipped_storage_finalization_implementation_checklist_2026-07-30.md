# Clipped Storage Finalization Implementation Checklist

Date: 2026-07-30

Status: strict selector-ineligible recording finalization implemented; main
campaign adoption and archive import remain gated.

## Goal

Keep clip parallelism as a compute optimization without making clip-local
Zarr layout or identity the published contract. The maintained boundary is:

```text
clip detect/refine evidence
  -> recording canonical detection
  -> recording refined detection
  -> geometry-only crop-v2
  -> clip pixel packages and keypoint inference
  -> recording raw/quality/refined/body-frame keypoints
```

Every recording-level writer plans chunks and shards from logical bytes. No
finalizer copies physical chunk or shard declarations from clip outputs.

## Implemented

- [x] Accept clipped-recording lineage in the shared immutable refined
      publisher.
- [x] Require the exact clipped collection, camera, media, and frame-map
      binding.
- [x] Require every bound clip manifest and logical array set again at the
      final publication gate.
- [x] Preserve recording-stable `instance_key` values.
- [x] Require globally allocated, non-overlapping `refined_row_ids`; do not
      silently rebase clip identities.
- [x] Rebase clip-local frame and source-row positions into the recording
      tables while retaining explicit clip-local lineage columns.
- [x] Rebuild instance and source-audit `frame_row_offsets` as exact `F+1`
      indexes, including empty and multi-instance frames.
- [x] Prove the joined source-audit table exactly equals all nine arrays in the
      strict recording canonical detection run.
- [x] Publish the refined snapshot with the promoted access-aware byte planner,
      consolidated metadata, exact codecs, and no selector or registry write.
- [x] Persist one canonical/refined pair receipt binding the canonical digest,
      every clip source, the refined manifest, and storage profile.
- [x] Allow the crop binder and crop manifest validator to consume clipped
      recording snapshots with their external evidence.
- [x] Publish geometry-only crop-v2 from an explicit selector-ineligible
      refined candidate and a reverified live pixel authority.
- [x] Allow keypoint terminal receipts and recording finalization to read the
      standalone crop archive while clip inference arrays remain in the source
      analysis archive.
- [x] Add an LSF fragment for refined then crop publication and a composition
      helper that freezes its dependency edge into keypoint-v2 finalization.
- [x] Cover empty frames, two clips, canonical mismatch, overlapping identity,
      publication/reopen, crop handoff, DAG ordering, and selector safety.

## Required before a real campaign

- [ ] Make every clip detector/refiner publish strict full-acquisition
      canonical/refined evidence rather than only legacy groups.
- [ ] Allocate manual `refined_row_ids` from one recording-global range before
      clip workers run. A clip-local `0..N` allocator is intentionally rejected.
- [ ] Mint raw and manual `instance_key` values in the recording frame and
      recording identity domain, not the clip-local frame domain.
- [ ] Generate the exact clipped-binding JSON from the finalized collection
      and recording-frame index; never hand-author media or frame-map digests.
- [ ] Insert the storage fragments into the maintained clipped campaign after
      canonical publication and before pixel-package/keypoint terminal gates.
- [ ] Ensure pixel packages bind the new crop manifest and row signatures.
- [ ] Run one small selector-ineligible campaign canary with an empty frame,
      multiple subjects in one frame, a rejected raw detection, and a manual
      addition.
- [ ] Run the recording-scale publication/read/peak-RSS benchmark. Record
      detection, refined, crop, and keypoint phase timings separately.
- [ ] Add one atomic archive-import transaction for the complete candidate set.
      A partial import must never become selector-visible.
- [ ] Review the imported candidate in Palette and Crimson before any guarded
      selector activation.

## Promotion rule

Completion of the standalone chain does not authorize production selection.
Promotion still requires exact logical equality, direct/consolidated metadata
equivalence, codec/CRC validation, no stale publications, complete downstream
row coverage, and the already frozen Crimson performance gates. Failed or
incomplete candidates remain selector-ineligible and registry-unregistered.
