# Clipped Storage Finalization Implementation Checklist

Date: 2026-07-30

Status: strict selector-ineligible clip evidence, binding, and recording
finalization implemented as composable DAG fragments; main campaign insertion,
canary execution, and archive import remain gated.

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
- [x] Make the standalone crop reopen contract name its separate refined-source
      archive explicitly. Reopen validation still recomputes the crop logical
      digest and compares the 13 crop arrays with the bound refined arrays.
- [x] Add an LSF fragment for refined then crop publication and a composition
      helper that freezes its dependency edge into keypoint-v2 finalization.
- [x] Cover empty frames, two clips, canonical mismatch, overlapping identity,
      publication/reopen, crop handoff, DAG ordering, and selector safety.
- [x] Convert each complete compatibility clip detect/refine pair into fresh
      strict canonical/refined evidence below `/tmp` or `.palette_benchmarks`.
- [x] Prove each strict clip canonical row/frame interval against the native
      recording canonical manifest and all nine recording arrays.
- [x] Allocate automated raw-backed `refined_row_ids` from recording canonical
      source-row positions, independent of clip worker scheduling.
- [x] Reject manual clip rows at this adoption boundary. Manual additions use
      recording-level deltas and compaction, where one global allocator owns
      new row IDs and keys.
- [x] Build the clipped binding from the finalized collection, recording clip
      manifest, streaming recording-frame-index digest, strict clip receipts,
      and freshly reopened refined manifests. The binding is not hand-authored.
- [x] Merge clip-local reason registries into one deterministic recording
      registry and remap `uint16` codes during finalization.
- [x] Add a composable dependency chain for strict clip evidence -> clipped
      binding -> recording refined snapshot -> crop-v2.
- [x] Add one selector-ineligible Crimson candidate composer spanning strict
      clip evidence, recording refined detections, crop-v2, raw/quality/refined
      keypoints, body frame, and a final fail-closed handoff manifest.
- [x] Add a benchmark-only adapter for the existing full recording keypoint
      aggregate. It requires pinned source metadata and model digests, exact
      `instance_key` set equality, crop origin/size equality, frame-map
      equality, node-local materialization, and shared-byte-planner output.
- [x] Add a benchmark-only canonical adapter for the current v003 clipped
      collection. It rebuilds from every clip detection group, verifies stable
      keys and the recording frame map, pins the expected recording row count,
      and publishes a current native-manifest-v2 access-aware store on
      node-local scratch before shared placement. The earlier v002 Crimson
      fixture is an independent performance baseline, not an equality
      authority for the v003 refined chain.
- [x] Vectorize recording keypoint row reconciliation by `instance_key`; the
      full-duration finalizer no longer performs one Python dictionary/loop
      operation per observation.

## Adoption and physical-layout boundary

This checkpoint adopts the logical and publication contracts without claiming
that one chunk budget is optimal forever. Every new strict canonical/refined,
crop, and keypoint output uses a named, versioned storage profile and the shared
byte planner. A later benchmark may promote a new profile ID; it must not alter
the logical dtype, identity, coordinate, or lineage contract in place.

Clip evidence is compute/publication evidence, not the selected recording
authority. It is safe to write clips independently because every worker owns a
fresh standalone Zarr. The recording finalizers then rematerialize complete
physical shards from logical rows; they never copy clip chunk or shard metadata.

The first maintained adoption is intentionally raw-backed-only. It supports
empty frames, multiple detections in a frame, filtered source detections, and
multiple subjects because all lookup uses `F+1` offsets and stable keys. It
does not assign manual observations inside clip workers. A manual addition is a
recording-level delta event; compaction publishes a new immutable refined run,
then crop/keypoint completion derives the corresponding new rows.

Binding digests are evidence digests, not a second copy of video pixels. The
global video digest hashes the strict `recording_clip_index.json`; each media
digest hashes its exact clip/camera source descriptor plus that global digest.
The recording and per-clip frame-map digests stream canonical ordered rows from
`recording_frame_index.parquet` and require
`recording_frame_id == parent_frame_index + 1`. Crop publication separately
revalidates the live pixel authority before any geometry becomes usable.

## Required before a real campaign

- [x] Add the strict clip canonical/refined evidence publisher and bounded LSF
      array. The main campaign still needs to invoke the fragment.
- [ ] Allocate manual `refined_row_ids` through the recording delta/compaction
      allocator. Clip-local `0..N` manual allocation remains rejected.
- [x] Require raw `instance_key` values in the recording frame and recording
      identity domain by comparing every clip key with the native recording
      canonical slice. Manual keys remain owned by delta compaction.
- [x] Generate exact clipped-binding JSON from persisted collection and frame
      evidence; no caller supplies media or frame-map digests by hand.
- [ ] Insert the storage fragments into the maintained clipped campaign after
      native canonical publication and legacy refinement, before
      pixel-package/keypoint terminal gates. The composable dependency chain is
      implemented; the parallel DAG review owns final monolith insertion.
- [ ] Ensure pixel packages bind the new crop manifest and row signatures.
- [ ] Run one small selector-ineligible campaign canary with an empty frame,
      multiple subjects in one frame, a rejected raw detection, and a manual
      addition.
- [x] Implement the immutable full-duration plan, LSF composition, commit pin,
      node-local keypoint republish, and final seven-artifact handoff gate.
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
