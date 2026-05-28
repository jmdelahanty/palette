# Contract Drift Audit — pydantic & zarr-attr field-level cross-check

**Date:** 2026-05-28
**Method:** Read-only parallel audit — 8 contract units, each pipelined through declare → trace → diff (extract declared schema → trace every consumer/writer/config → diff into field-level drift), then a synthesis pass. 25 agents, no files modified.
**Question driving it:** Should we adopt pydantic validation for the data contracts, and are any existing pydantic models stale / out of sync with code?
**Companion doc:** [`repo_eval_2026-05-28.md`](repo_eval_2026-05-28.md) — this audit drills into that eval's Tier 1 "documented-not-enforced" finding.

> **Eye-mask findings are delete-don't-fix.** The standalone `eye_masks` stage is legacy and slated for deprecation/removal — eyes are now a channel within the unified subject mask (`SUBJECT_MASK_LABEL_SCHEMAS`: `eyes_union` / `eye_left` / `eye_right`). So **EyeMask\* contract drift below (`min_positive_area`, the legacy `batch` field, the EyeMask\* models themselves) should be removed *with the stage*, not fixed or migrated to pydantic.** Removal is non-trivial (~102 files; `eye_masks` is a registered stage that `refined_eye_masks` depends on, with live downstream consumer eye-angle analysis). The severance blocker is re-sourcing eye geometry from subject-mask channels.

---

## Headline

The **live pydantic config models are clean**. All the drift is in the **hand-rolled zarr-attr contracts** — the exact layer the pydantic question was about. The sharpest result: **pydantic v2 would fix only ~3 of the drifts found, and only if you also commit to routing every zarr read/write through the models.** The library is not the fix; the boundary discipline is. Adopting pydantic without that discipline would add a *third* contract layer that drifts from the other two.

Separately, two root-level pydantic **v1 orphans are stale and dead**: `src/config_models.py` (zero imports, duplicates the live v2 models) and `src/red_to_yolo.py` (orphan v1 CLI script). Both should be deleted/relocated.

---

## In sync vs out of sync

| Contract unit | Verdict |
|---|---|
| `training/config.py :: DetectConfig/PoseConfig/DatasetConfig/TrainingParams` | ✅ In sync |
| `prepare_detect_training.py :: manifest models` | ✅ In sync |
| `training/config.py :: EyeMask*` | ✅ except one dead field (`min_positive_area`) |
| `shared/zarr_run_completion.py` | ✅ except 2 hardcoded-string-vs-constant slips |
| `training/config.py :: SubjectMask*` | ⚠️ `Literal` bypass + validator invariant violable |
| `shared/stage_provenance.py` | ❌ 3 lineage fields written+read, never declared |
| `shared/run_lineage_fingerprint.py` | ❌ wrong declared type + forked status set |
| `shared/zarr/schema.py` run-metadata | ❌ worst — required fields read everywhere, never written |

---

## Ranked drifts

| # | Field | Unit | Kind | Why it matters |
|---|---|---|---|---|
| 1 | `recording_id`, `session_uuid`, `experiment_setup`, `dish_design` | `zarr/schema.py` | writer_reader_mismatch | Read 1080/189/74× incl. registry `db.py:597`; **never written** in `create_palette_zarr()` (schema.py:162–199). Registry assumes the pipeline sets these; nothing does. Highest blast radius. |
| 2 | `source_crop_run`, `source_detect_run`, `source_refined_run` | `stage_provenance.py` | used_undeclared | Written (`refine_keypoints.py:1146–1150`), read by the provenance-consistency checker (`check_provenance_consistency.py:402/470/591`), entirely absent from the contract. Lineage validation depends on undeclared fields. |
| 3 | `label_schema_id` | `SubjectMask*` | type_mismatch | `Literal["auto","subject_v1_union","subject_v1_lr"]` bypassed: in the `"auto"` path, arbitrary zarr str flows back into config (`train_unet_subject_masks.py:665`) with validation skipped (`zarr_subject_mask_dataset.py:188–192`). Silent config-type corruption. |
| 4 | `lineage_payload_json_structure` | `run_lineage_fingerprint.py` | type_mismatch | Declared `Mapping[str,Any]`, actual attr is a JSON **string** from `canonical_lineage_json()` (line 279/386). **Contract doc is wrong, not the code.** |
| 5 | `FINGERPRINT_STATUSES` | `run_lineage_fingerprint.py` | config_key_drift | Defined twice with incompatible values: shared `{complete,best_effort,missing}` (line 30) vs `virtual_collection_manifest.py:36` adds `not_applicable`. |
| 6 | `zarr_purpose` | `zarr/schema.py` | used_undeclared | Written (schema.py:197–199), read 190+× for train/analysis filtering, undeclared. High blast radius if it drifts. |
| 7 | `names` / `nc` (overwrite) | `SubjectMask*` | writer_reader_mismatch | `model_validator` invariant (config.py:423–446) silently violated by unconditional overwrite at `train_unet_subject_masks.py:666–667`. Validator runs at deserialize, not on assignment. |
| 8 | `zarr_python` | `zarr/schema.py` | used_undeclared | Written (schema.py:188), undeclared. Library-version stamp. |
| 9 | `validate_zarr_structure` | `zarr/schema.py` | declared_unused | Defined (673–774), exported, **imported (`pipeline.py:35`) but never called.** The schema's own enforcement hook is dead. |
| 10 | `palette_run_failed_at_utc`, `palette_run_error` | `zarr_run_completion.py` | naming_inconsistency | Hardcoded strings (lines 93/95) vs the constant pattern used elsewhere. Rename-refactor hazard. |
| 11 | `min_positive_area` | `EyeMask*` | declared_unused | Declared with `ge=0` (config.py:294), never read. Dead constraint. |
| 12 | `ZARR_SCHEMA`, `batch`, `LINEAGE_ATTR_NAMES`, `source_video_path` default | various | declared_unused / default | Documentary or dead. |

---

## Per-unit pydantic verdict

Pydantic v2 validates at the **(de)serialization boundary** (`model_validate` / `model_dump`). It does **not** police raw `zarr.attrs` dict access, array dtype/shape, or post-construction attribute assignment unless `validate_assignment=True` is set.

| Unit / drift | Would pydantic v2 prevent it? |
|---|---|
| `zarr/schema.py` — `recording_id` etc. read-never-written | **NO — orthogonal.** Code reads raw `root.attrs[...]`. Helps *only* if every read goes through `Model.model_validate(root.attrs)`. The model alone is inert. |
| `zarr/schema.py` — `zarr_purpose`/`zarr_python` undeclared writes | **PARTIAL.** Writing via `model_dump()` (with `extra="forbid"`) keeps declaration and write in sync — only if writes go through the model. |
| `zarr/schema.py` — `ZARR_SCHEMA` / `validate_zarr_structure` dead | **NO.** Dead-code / unused-enforcement, not a tooling gap. |
| `zarr/schema.py` — array dtype/shape, `source_video_path` default | **NO.** Pydantic doesn't validate zarr array dtype/shape. Keep the custom validator. |
| `stage_provenance.py` — `source_*_run` used_undeclared | **PARTIAL → YES if boundary-enforced.** A provenance model with these fields + `extra="forbid"`, all writes via `model_dump`, reads via `model_validate`, makes "write an undeclared field" a validation error. Strongest pydantic win in the set. |
| `SubjectMask*` — `label_schema_id` Literal bypass | **PARTIAL.** Pydantic rejects out-of-Literal values *at deserialization*, but the bypass is the `"auto"` path that **deliberately skips validation**. Pydantic helps only if you remove the skip and re-validate. The lenient skip is the real defect. |
| `SubjectMask*` — `names`/`nc` overwrite | **YES, narrowly.** `ConfigDict(validate_assignment=True)` re-triggers the after-validator on the overwrite (lines 666–667). Genuine v2 capability the custom path lacks. |
| `run_lineage_fingerprint.py` — `lineage_payload_json` type | **NO — arguably mislabeled.** The string is intentional. Fix the contract declaration, not the tooling. |
| `run_lineage_fingerprint.py` — forked `FINGERPRINT_STATUSES` | **NO.** Code-organization. A shared `Literal`/enum imported by both fixes it — that's deduplication, not pydantic. |
| `zarr_run_completion.py` — hardcoded strings | **NO.** Refactor hygiene. A module constant does the same. |
| `EyeMask*` — `min_positive_area` dead | **NO.** Pydantic doesn't detect unused fields. |

**Decision rule:** adopt pydantic v2 per unit only if you can answer **yes** to *"will every zarr-attr read and write for this contract go through the model?"* That's `stage_provenance`, run-metadata, and SubjectMask (`validate_assignment`). Skip it everywhere else.

---

## Remediation — ordered, subtraction first

1. **Delete dead surface (pure subtraction, zero risk).** `validate_zarr_structure` + its unused import (`pipeline.py:35`); `ZARR_SCHEMA` if it stays documentary; `min_positive_area` (config.py:294); `SubjectMaskTrainingParams.batch` (config.py:191–193); `LINEAGE_ATTR_NAMES` from the doc if internal. Also delete the stale v1 orphans `src/config_models.py` and `src/red_to_yolo.py` (relocate the latter to `tools/` if still wanted).
2. **Collapse the forked constant.** Delete `virtual_collection_manifest.py`'s `FINGERPRINT_STATUSES` (lines 36–41); import the shared one (`run_lineage_fingerprint.py:30`). If `not_applicable` is real, add it to the single shared definition.
3. **Fix the 2 logic defects pydantic will NOT fix (forcing function = make the lenient path fail loud).**
   - Remove the `expected_label_schema_id is None` validation skip (`zarr_subject_mask_dataset.py:188–192`); after `"auto"` resolves, re-validate against the `Literal`.
   - Give `recording_id`/`session_uuid`/`experiment_setup`/`dish_design` a **single enforced write path**: either `create_palette_zarr()` writes them (and fails if absent), or exactly one documented populator. 1080 reads currently depend on an invariant nobody enforces (`db.py:597`).
4. **Decide pydantic v2 with the single forcing-function test** above. Most clearly justified for **stage_provenance** (declare `source_*_run`, `extra="forbid"`) and **run-metadata** (declare `zarr_purpose`/`zarr_python`; required fields enforced at the read boundary). Use `ConfigDict(validate_assignment=True)` on SubjectMask configs to kill the names/nc overwrite violation.
5. **Keep array dtype/shape on the custom validator. Do not migrate it** — pydantic adds nothing and would create a fourth drift vector.
6. **Cosmetic / lowest priority:** add `RUN_FAILED_AT_ATTR` / `RUN_ERROR_ATTR` constants, replace hardcoded strings (`zarr_run_completion.py:93/95`) — free if that unit moves to a model, otherwise a 2-line edit.
7. **Fix the doc, not the code, for `lineage_payload_json`** — declare it `str` (canonical JSON), since the string is intentional (`run_lineage_fingerprint.py:279/386`).

**Net:** ~5 deletions + 2 logic fixes recover most of the integrity *before* any pydantic migration. Pydantic v2 is justified for 2–3 metadata units **only under boundary discipline**; it is the wrong tool for arrays, the `"auto"` skip, dead code, naming hygiene, and forked constants.
