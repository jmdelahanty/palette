# Audit: collapsing the registry sync/backfill scripts into one reconcile path

**Date:** 2026-06-18
**Method:** Three parallel read-only code traces (repair-script inventory, core ingest
engine, model-export path). No changes made. Companion to
`registry_design_assessment_2026-06-18.md`.

## Headline

A unified "re-derive registry state from disk truth" engine **already exists** —
`Registry.register_from_root()` (`db.py:6154`) is idempotent and rebuilds ~10 derived
tables from one zarr root via a DELETE-then-INSERT (`replace_*`) pattern. The drawer of
repair scripts mostly re-implements boilerplate *around the same primitives*, or covers
sources `register_from_root` doesn't reach. So this is **assembly, not invention** — but
only ~7 of the scripts genuinely belong in one reconcile; a few others are a different
concern and should stay separate. Don't oversell the collapse.

## The capture landscape today

Capture is organized by **table family**, not by **reconciliation scope**. Four layers:

| Layer | Entry point | Scope | Idempotent | Re-derives all tables? |
|---|---|---|---|---|
| Scan/register | `scan_zarr()` → `register_from_root()` (`db.py:6154`) | 1 zarr root | ✅ REPLACE | ✅ ~10 zarr-derived tables |
| Discovery | `scan_paths()` + `reconcile_missing_datasets()` (`db.py:7862`) | N roots | ✅ | ⚠️ dataset rows + missing-flag only |
| Inline refresh | `refresh_*_from_root()` (5+ variants, `db.py`) | 1 table, 1 root | ✅ REPLACE | ❌ one table |
| Maintenance backfill | `_backfill_*()` (9+ variants, `maintenance.py`) | 1 table, all datasets | ✅ REPLACE/dataset | ❌ one table |

Plus the standalone CLI scripts in `utils/` and `scripts/`.

The extractors already share a uniform contract (no base class, uniform by convention):

```
_extract_<TABLE>_rows(root, *, zarr_path, recording_id, zarr_use) -> List[Dict]
```

(`extractors/{crop,quality,masks,detect_performance,keypoint_performance}.py`). They are
pure functions; insertion happens via `Registry.replace_*()`. **This uniform shape is the
collapse seam.**

## The collapse map — three categories that fit one interface

All three reduce to **`source → field-mapping → idempotent upsert`**:

### A. Zarr-attr → registry extractors  *(already uniform; collapse first)*
The inline `refresh_*_from_root` family and the `maintenance.py` `_backfill_*` family are
per-table slices of the same operation `register_from_root` already does whole. Missing
piece is one orchestrator:

```python
def reconcile_dataset_from_root(self, root, zarr_path, *, include_step_status=False) -> dict:
    """Re-derive ALL registry state for one dataset from zarr truth, idempotently."""
```

This subsumes `register_from_root` + the inline-refresh family + the per-table backfills.

### B. Profile syncs  *(same shape, different zarr location; fold in)*
`sync_detection_profile_registry.py`, `sync_keypoint_profile_registry.py`,
`sync_eye_mask_profile_registry.py` read from `analysis/*_profile_runs/*/profile_summary`
(a different zarr location than the extractors) and upsert `*_data_profile` tables. They
are **~90% identical scaffold, ~40–50% identical mapping** (~1900 lines that could be
~600). They are extractors in all but name — register them as `_extract_*_profile_rows`
and they join category A.

### C. Model-artifact capture  *(parallel reconcile from manifest sidecars)*
`onnx_models` / `tensorrt_models` rows are **~95% re-derivable** by re-reading the
`*.onnx.manifest.json` / `*.tensorrt.manifest.json` sidecars (same
`read JSON → _resolve_shape_fields → upsert(ON CONFLICT)` shape as a zarr extractor).
`backfill_pose_onnx_registry_metadata.py` already does exactly this for one case and would
collapse into a general `reconcile_model_from_manifests()`. `training_models` is only
**~60%** re-derivable (shape via stored metrics + code logic; `set_id`/`skeleton_id` need
DB lookups) — recoverable but lossier.

## What should NOT collapse (be honest about the boundaries)

- **Zarr-attr backfills** — `backfill_keypoint_auto_review_policy.py`,
  `backfill_detect_review_status.py`, `scripts/backfill_protocol_json.py`,
  `backfill_detection_profiles.py` write to **zarr `.attrs`, not the registry**. They
  repair *disk truth*, which is the opposite direction of a registry reconcile. Folding
  them in would conflate "fix the source" with "reindex from the source." Keep separate.
- **Audit / read-only scripts** — `audit_registry_dataset_paths.py`,
  `check_training_registry.py`, `check_contract_freshness.py` are reporting, not capture.
  They belong with the shared *read* API (a separate refactor), not the reconcile engine.
- **Export-time live capture** — `build_env` (`trt_version`, `gpu_uuid`, `gpu_name`,
  `compute_capability`, `torch_version`) is queried live at export and is **not
  re-derivable** except from the persisted manifest. Reconcile can *recover it from the
  manifest file* but cannot regenerate it if the manifest is gone. So export-time capture
  stays; reconcile is the recovery/convergence path on top.

## Idempotency note

The convergence machinery is sound: `register_from_root` and `replace_*` use
DELETE-then-INSERT; `upsert_recording_step_status` is keyed and append-logs history;
profile syncs use payload signatures; most CLI scripts default to dry-run/`--apply`. A
unified reconcile inherits these guarantees if it composes the existing primitives rather
than adding new write paths.

## Rough magnitude

- **Collapsible into one reconcile:** the 3 profile syncs + `backfill_pose_onnx_*` +
  `dish_mask_registry_sync` + the inline-refresh family + the `maintenance.py` per-table
  backfills — all expressible as `reconcile_dataset_from_root()` (zarr) and
  `reconcile_model_from_manifests()` (models) over the uniform extractor interface.
- **Stays separate (different concern):** ~4 zarr-attr backfills, ~3 audit/read scripts.
- The profile-sync trio alone is ~1900 → ~600 lines.

## Per-stage profile coverage (correction to category B)

The "data profile" family (`build_*_profile_summary` → `analysis/*_profile_runs` zarr →
`sync_*_profile_registry.py` → `*_data_profile` table, plus an `aggregate_*_training_data_card`)
is applied **unevenly**, and backwards relative to the roadmap:

| Stage | profile builder | sync → data_profile | training-data-card | performance | quality | Note |
|---|---|---|---|---|---|---|
| detection | ✓ | ✓ | ✓ | ✓ | `detect_quality` | active |
| keypoint | ✓ | ✓ | ✓ | ✓ | `keypoint_quality` | active |
| eye_mask | ✓ | ✓ | ✓ (+plot) | ✓ | `eye_mask_quality` | **LEGACY — to deprecate** |
| training_image | ✓ | ✓ | — | — | — | active |
| subject_mask | ✗ | ✗ | ✗ | ✓ | `component_quality` (per-component) | **future — profile gap** |
| crop | — | — | — | — | `crop_quality` | active |

Two findings:

1. **subject_mask has no profile path at all** — no `build_subject_mask_profile_summary`,
   no `subject_mask_profile_runs` zarr group, no `sync_*`, no `subject_mask_data_profile`
   table, no training-data-card. It is modeled with `subject_mask_performance` +
   per-component `subject_mask_component_quality` only. Since subject masks subsume eye
   masks and are the strategic stage, the absent distribution-profile / training-data-card
   is a genuine **capability gap**, not merely a different modeling choice.

2. **eye_mask (legacy) is the most heavily-served stage** — ~21 `utils/*eye_mask*` files
   including the full profile/data-card/quality stack and a `prune_legacy_eye_mask_profile_runs.py`.
   Registry surface is misallocated: the component being removed has more coverage than the
   one replacing it.

**Correction to "Category B" above:** the eye_mask profile sync is **not** a fold-into-reconcile
candidate — it is a **subtraction target** on the deprecation path. The collapse and the
deprecation are the same lever in two directions:

- **Eye_mask:** delete the profile stack rather than fold it in.
- **Subject_mask:** make the reconcile/extractor interface the place to *add* the missing
  profile extractor, instead of writing another standalone `sync_` script.

Note: many more per-stage utilities exist beyond reconciliation (`run_*`,
`prepare_*_training_from_registry`, `export_*_training_zarr`, `validate_*`,
`resolve_*_stale`, `merge_*_runs`, `aggregate_*_training_data_card`). Those are pipeline
actions, not registry reconcile, and are out of scope for the collapse — but they are the
bulk of the per-stage sprawl noted in the design assessment.

## Bottom line

You are closer than the script count suggests. The reconcile *engine* exists
(`register_from_root` + `replace_*` + the uniform extractor contract); what is missing is
**one per-dataset orchestrator** that runs every extractor (zarr + profile + model) and
the step-status refresh in a single idempotent pass, plus registering the profile/model
readers as extractors. That single function would let most of the repair-script drawer be
deleted — while the zarr-attr repairers and the read-only auditors correctly remain their
own thing.

## Smallest first step (proposed, not done)

Add `reconcile_dataset_from_root(root, zarr_path)` that loops the existing extractors +
`replace_*` + step-status refresh for one dataset (pure assembly of current primitives),
then re-express the 3 profile syncs as extractors behind it. That alone validates the
interface and retires the most duplicated scripts, before tackling model-manifest
reconcile.
