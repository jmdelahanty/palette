# Brief: mechanical wave — shim deletion, grayscale definition sites, FrameDomains consumers

**From:** commander session, 2026-07-05 (evening)
**Status: READY.** Three independent slices, one agent each, dispatch in parallel.
**Do NOT push or merge — the commander verifies and merges.**
**Merge order (commander-side): A → B → C** (smallest blast radius first; C rebases onto
`sun` after A/B land — C and B share files, different regions).
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first (all agents):** `HANDOFF_2026-07-05.md` operating-notes section only.
Ground rules: local `sun` is ground truth (origin stale/unreachable); fresh worktree on
`agent/<slice>` from CURRENT `sun`; env `~/miniconda3/envs/palette-py311/bin/python`
(conda; never uv, never a `.venv`); sync code only; re-locate cited line numbers by
content (merge traffic is heavy); if a premise is wrong, report and stop that item.
Local gates before "done": `lint-imports --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py`, `git diff --check`, `py_compile`, focused
tests, then full suite `PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
Baseline **3,401 passed / 2 skipped** at merge `62a8e52` — recount on your branch first.

**Dropped from this wave (already done — do not "finish" it):** detect-review pointer
writer retirement. Verified 2026-07-05: the only remaining
`detect_review_status_latest` writers are exactly the two annotated migrations
(`utils/migrate_legacy_detect_labels.py`, `utils/migrate_refined_detect_sparse.py`) —
the sanctioned end state from `brief_promotion_authority.md`. Remaining refs
(`shared/run_resolution.py`, `labeling/task_generation.py`, batch pipelines, schema)
are READERS serving legacy-store resolution; retiring them requires a census of stores
lacking `authoritative_run` and is a separate maintainer decision, not mechanical work.

---

## Slice A — delete the utils re-export shims (`agent/utils-shim-deletion`)

**State (verified):** Phase 2's retargeting was complete — grep finds ZERO importers of
the moved modules via `fisheye.utils.*` / `..utils` forms anywhere in `src/` or
`tests/` (outside `utils/` itself). Five shims are 3-line pure re-exports:
`zarr_io.py`, `zarr_metadata.py`, `calibration.py`, `encoder_tags.py`,
`recording_preflight.py`. `system.py` is 14 lines (may still hold the old
monkeypatch-seam wrapper). `import_video_metadata.py` is **125 lines — NOT a pure
shim**. `metadata.py` also remains (Phase 2 item 3 was a per-symbol decision — check
the utils-phase2 agent's report/commits for what was decided).

Scope, in order:
1. **Delete the five 3-line shims.** Before each delete, grep-proof zero references in
   the FULL tree including string forms: dotted imports, relative imports, AND
   monkeypatch/string paths (`"fisheye.utils.zarr_io"` etc. — patch targets don't show
   up as imports). One commit for the batch.
2. **`system.py`:** if it is now a pure re-export with zero importers/patch-targets,
   delete it; if the compatibility wrapper seam is still referenced by tests, retarget
   those tests to `shared/system_metadata` first, then delete. This closes the
   `brief_utils_phase2.md` loose end.
3. **`import_video_metadata.py` (125 lines) and `metadata.py`:** census, don't assume.
   If the strategy doc's move was completed and these hold only re-exports plus dead
   weight, finish per the same rule. If they still hold LIVE logic that was never
   moved, that is a finding — leave the file, report exactly what remains and why.
   Do not do a fresh module move inside this slice.
4. **Tighten the import-linter contract** (`pyproject.toml [tool.importlinter]`): the
   forbidden-modules list for severed utils modules can now reference deleted modules —
   update the contract to match reality (import-linter errors on forbidding
   nonexistent modules; verify behavior and adjust). Ratchet note: `utils/` deletions
   only shrink files — no ratchet risk.

Out of scope: any new module move; runner relocations (Phase 5 of the strategy doc);
anything in `utils/` that is a live runner/tool.

Validation adds: `lint-imports` green with the updated contract; grep-proof table in
the report (per deleted file: the three grep forms, all zero).

## Slice B — grayscale definition-site unification (`agent/grayscale-definition-sites`)

**The finding (review 6c, last open piece of Finding 6):** three grayscale conventions
coexist as inline literals — `capture/import_video.py` (~line 381: Matlab-style luma
`0.2989/0.5870/0.1140`), `shared/crop_image_source.py` (~342-344: cv2-style luma
`0.299/0.587/0.114`), `tracking/crop.py` (~961: GPU **unweighted** `.mean(dim=-1)`).
The divergence is already contracted/stamped (decode-contract work, commit `29e9e6a`);
what's missing is a single definition site.

**LOCKED DECISION — zero pixel change.** This slice unifies WHERE the conventions are
defined, not WHAT they compute. The three callsites produce bit-identical output before
and after. Converging the VALUES (e.g., making GPU crop luma-weighted, or reconciling
the two luma variants — note `0.2989 ≠ 0.299`) changes produced pixels and therefore
has model-retraining implications; that is a separate maintainer decision. Flag it,
don't do it.

Scope:
1. New `shared/grayscale.py` (or a better existing shared home — your call, report it)
   defining the three NAMED conventions (e.g. `LUMA_BT601_MATLAB`, `LUMA_BT601_CV2`,
   `UNWEIGHTED_MEAN`) with conversion functions for the numpy and torch paths as
   needed. Docstring must state plainly that the three differ, that each callsite's
   choice is contractual, and point at the pixel-decode census + decode-contract docs.
2. Retarget the three callsites to import their EXACT current convention from the
   shared module. Minimal diffs; no reordering of surrounding decode logic.
3. **Bit-identity tests:** for each callsite path, a test converting a fixed synthetic
   RGB array through the new shared function and asserting exact equality
   (`np.array_equal` / `torch.equal`) with the OLD inline computation (reproduce the
   old expression verbatim inside the test as the oracle).
4. If the existing decode-contract stamping has a natural place to record the
   convention NAME (an attr already being written at these sites), add it; if it would
   require threading new state, skip and note it.
5. Report ends with a short "convergence decision memo": what changing GPU-unweighted →
   luma would affect (which artifacts/models consume those pixels), so the maintainer
   can decide the follow-up with real information. Analysis only — no code.

Out of scope: changing any weight value, touching GPU-primary decode logic, the
`stat_v1` video fingerprints, retraining anything.

## Slice C — FrameDomains consumer migrations (`agent/frame-domains-consumers`)

**State:** resolver landed (`shared/frame_domains.py`, `Recording.frame_domains()`,
import-time identity stamp) — impl slice 1 of `docs/frame_domains_resolver_design.md`.
Today only `capture/import_video.py` and `shared/recording.py` touch it; every other
frame-domain translation in the codebase is still ad-hoc — the bug class the design
exists to kill.

Scope:
1. **Read the design doc's consumer/migration section** — it is the spec for which
   callsites move and in what order. If the doc does not enumerate consumers, build the
   census yourself: every site translating between crop-row / detect-frame /
   video-frame / analysis-frame index domains with local arithmetic (grep starting
   points: `frame_index`, `source_frame`, offsets applied to frame arrays near zarr
   reads), classify by domain pair, and migrate in risk order (read-only consumers
   first, writers last).
2. **CHECKPOINT after the first consumer migration:** stop and report before
   proceeding. The first one establishes the pattern (how the resolver is threaded —
   via `Recording` accessor vs direct `frame_domains(root)`), and surfaces API friction
   while the blast radius is one file. Include in the checkpoint report: the consumer
   census with your classification, and the proposed order for the rest.
3. Migrate the remainder per the approved pattern. Behavior-preserving: each migration
   carries a test pinning old-vs-new equivalence on a fixture store (reuse the
   frame-domains fixture from impl slice 1).
4. Files you touch WILL overlap Slice B's (`tracking/crop.py`,
   `capture/import_video.py`) in different regions: rebase onto current `sun` at every
   checkpoint and before declaring done; do not begin your `crop.py` edits until you've
   rebased past Slice B's merge if it has landed (commander merges B before C).

Out of scope: changing any domain semantics, fixing latent off-by-one bugs you FIND
(report them loudly as findings — a wrong translation converted faithfully is a
documented bug, a "fixed" one is a silent data change), touching eye-mask paths
(deprecated), retiring the resolver's legacy fallbacks.

## Reporting (all slices)
Branch + commit SHAs, what landed vs skipped and why, grep/test proof for every claim,
full-suite counts (recounted baseline → final), premise discrepancies, and each
slice's named deliverable: A = grep-proof table + contract diff; B = bit-identity test
evidence + convergence memo; C = consumer census + checkpoint pattern decision.
