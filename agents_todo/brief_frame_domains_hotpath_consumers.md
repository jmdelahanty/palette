# Brief: FrameDomains hot-path consumer migrations (`agent/frame-domains-hotpath`)

**From:** commander session, 2026-07-06
**Status: READY.** One agent, three stages in order, **mandatory CHECKPOINT after
Stage 2** (the first true writer migration — it sets the writer pattern).
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `docs/archive/HANDOFF_2026-07-05.md` operating-notes section, then
`docs/archive/frame_domains_consumer_census_2026-07-05.md` — this brief executes
census items 5-remainder, 6, and 8, which are UNBLOCKED as of the vectorized resolver
(`9ee35e9`): `FrameDomains.convert()` is now internally vectorized (lazy per-edge
lookup caches), so large-array conversion through the resolver is no longer a
Python-loop downgrade. Ground rules: local `sun` is ground truth; fresh worktree on
`agent/frame-domains-hotpath` from CURRENT `sun`; rebase onto current `sun` at the
checkpoint and before declaring done (merge traffic is heavy — re-locate cited line
numbers by content); env `~/miniconda3/envs/palette-py311/bin/python` (conda; never
uv, never a `.venv`); sync code only. Local gates before every checkpoint:
import-linter via `scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py`, `git diff --check`, `py_compile`,
focused tests, then full suite `PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
Baseline: **3,433 passed / 2 skipped** at merge `62be09a`; recount on your branch.

## Standing rules (from the four merged Slice C checkpoints — follow, don't rederive)

- **Behavior-preserving.** Every migration carries a test with the OLD computation
  kept verbatim in the test module as the oracle. If a local translation looks wrong,
  migrate it faithfully and report the latent bug loudly — never fix silently.
- **Resolver-error fallback:** wrap resolver use so any `FrameDomainError` falls back
  to the verbatim legacy computation (exemplar:
  `utils/training_image_profile.py::_map_detection_frames_to_rows` and its narrow
  `FrameDomainUnmappedError` fallback). No new failure modes: an archive that worked
  before must work identically after.
- **Pattern:** direct `FrameDomains(root=...)` for root-holding code;
  `Recording.frame_domains()` only where a `Recording` is already in hand.
- **Histograms are not conversions.** `np.bincount(...)` and boolean-mask coverage
  math STAY as-is everywhere. What migrates is (a) index-domain *conversion*
  (`original_frame_indices[local]` style) and (b) frame-universe *length resolution*
  (`shape[0]` / attrs / `max()+1` precedence chains) — the latter via
  `FrameDomains.count(...)` inserted at the SAME precedence position the legacy
  source occupied, exactly as checkpoint 3 did in `detect_quality.py`.
- Preserve existing exception types and message formats at migrated sites (e.g. the
  `IndexError` bounds message in Stage 1's mapper). Checkpoint 1 kept the legacy
  bounds check before `convert()` for exactly this reason — do the same.

## Stage 1 — census item 5-remainder: `utils/regenerate_training_crops_pynvvc.py`

`_map_source_frame_indices` (~line 349) is structurally the same function checkpoint 1
migrated in `diagnostics/check_training_crop_pynvvc_pixel_parity.py` — same
`_should_use_original_frame_indices` heuristic, same bounds check, then vectorized
`original_frame_indices[local]`. Replay that pattern: route the mapping through
`FrameDomains.convert(STORED_ZARR → SOURCE_VIDEO)`, keep the heuristic, bounds check,
and metadata dict byte-identical. This tool WRITES regenerated crop artifacts, so the
equivalence test must pin the mapped index array old-vs-new on a fixture store
(reuse/extend the checkpoint-1 fixture), not just spot values.
**Out of scope in this file:** the `source_frame_index_parquet` mode and the
clip-local index computation (~lines 204–233) — parquet/clip mappings are external
sidecar lineage, not zarr frame-domain edges. `direct` mode also stays untouched.

## Stage 2 — census item 6: `detection/detect_keypoints_yolo.py` — WRITER, then CHECKPOINT

The frame-domain surface here is the **frame-universe length** feeding written
arrays, not index conversion: when `_resolve_full_image_shape` yields no
`total_frames`, the code falls back to `frame_indices.max() + 1` (~line 839), and
that length sizes `np.bincount(frame_indices, minlength=total_frames)` which is
WRITTEN as both `n_rois` and (when not lineage-copied) `frame_counts` (~lines
841–860). Migration:
1. Insert a `FrameDomains`-resolved count for the appropriate domain of
   `frame_indices` at the same precedence slot (after `_resolve_full_image_shape`,
   before the `max()+1` fallback), with the `max()+1` fallback retained verbatim
   below it. Determining the correct domain for `crop_source.frame_indices` (crop-run
   frames vs acquisition frames — check what the crop run stamps and what
   `frame_indices_override` implies) is part of the job: **write the determination
   and its evidence in the checkpoint report.** If the domain of `frame_indices`
   cannot be established with confidence from stamps/contracts, STOP and report —
   do not guess a count source for a writer.
2. The written arrays are the acceptance bar: on a fixture store, run the writer
   path twice (new code vs legacy via monkeypatched count resolution, the
   checkpoint-3 technique) and assert `n_rois` and `frame_counts` are
   **byte-identical** (`np.array_equal` on values AND dtype `i4` preserved).
3. `np.bincount`, the lineage-copy branch, and `frame_indices` override semantics
   stay untouched.
**CHECKPOINT: stop and report after Stage 2.** Include: branch+SHA, the domain
determination for `frame_indices` with evidence, equivalence-test output, full-suite
count. The commander merges and approves the writer pattern before Stage 3.

## Stage 3 — census item 8: refine/curation/tracking count resolution (4 files)

Apply the approved pattern per file; one commit per file; each with its own
legacy-oracle equivalence test:
- `refinement/refine_detect.py` — the frame-axis resolution helper (~lines 133–167:
  `original_frame_indices.shape[0]` → `frame_counts.shape[0]` precedence) migrates to
  `count(STORED_ZARR)` / `count(RUN_FRAME)` at the same precedence slots, legacy
  fallback retained (checkpoint-3 replay).
- `shared/refined_detect_curation.py` — `total_frames` is a *parameter* here and the
  `bincount` fallback stays; migrate only if a call site inside these files derives
  that parameter with local arithmetic. If `total_frames` always arrives from
  callers outside this brief's files, REPORT "no migration site" and move on — do
  not chase callers into new files.
- `tracking/arena_assignment.py` — the attrs-precedence total-frames inference
  (~lines 94–121) and `len(frame_counts)` (~line 656) migrate to resolver counts at
  the same precedence slots; the attrs chains stay as fallback in their exact order.
- `tracking/crop.py` — HOTTEST path, most conservative treatment: only the
  frame-universe length used by the validation helper (~lines 604–617) may migrate,
  and only if it currently derives from zarr-local arithmetic rather than
  `get_total_frames` metadata (verify first; if it's all metadata-sourced, REPORT
  "no migration site" — that is a fine outcome). No conversion of bulk index arrays
  in this file in this slice.

## Out of scope — hard boundaries
- **Census item 7 (`shared/crop_image_source.py`): FORBIDDEN** — gated on real
  acquisition crop-video dropped-frame evidence per the design approval record.
- **Census item 9 (producer/exporter stampers): FORBIDDEN** — writer-stamping phase
  comes after read-side patterns stabilize.
- No changes to `shared/frame_domains.py` itself. If the resolver is missing
  something a migration needs (e.g. a domain count not derivable on a real store),
  that is a REPORTED finding, not an in-slice resolver patch.
- No eye-mask paths (deprecated), no domain-semantics changes, no silent fixes of
  latent off-by-ones, no new public APIs.

## Reporting (checkpoint and final)
Branch + commit SHAs, per-file landed/skipped/no-migration-site with grep/test proof,
the Stage 2 domain determination with evidence, premise discrepancies (cited line
numbers WILL have drifted — re-locate by content and say so), full-suite counts
(recounted baseline → final).
