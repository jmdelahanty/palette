# Root Scratch Inventory - 2026-07-05

Status: cleanup applied

This inventory classifies tracked repository-root scratch files and adjacent generated
artifacts reviewed during the 2026-07-05 hygiene sweep. Deletions were limited to files
that were unimported by `src/` and `tests/`, not referenced by live documentation except
the cleanup brief itself, and clearly obsolete one-off notes, diagnostics, or generated
outputs.

## Deleted

| Path | Reason |
| --- | --- |
| `CRITICAL_REIMPORT_NEEDED.md` | One-time enum migration warning; referenced only by obsolete enum notes, archive reviews, and this cleanup brief. |
| `ENUM_COLUMNAR_FORMAT_CHANGES.md` | Historical enum migration implementation note; superseded by live import code/tests and archive reviews. |
| `ENUM_FINAL_SUMMARY.md` | Historical enum migration summary; referenced only by the old enum note cluster. |
| `ENUM_IMPLEMENTATION_SUMMARY.md` | Historical enum migration summary; referenced only by archive reviews and the old enum note cluster. |
| `ENUM_PATHS_QUICK_REFERENCE.md` | Historical C++ enum path note; referenced only by the old enum note cluster. |
| `FRAME_SAMPLING_IMPLEMENTATION.md` | Historical implementation summary; frame sampling is covered by live code/tests. |
| `POSE_SCHEMA_GUIDE.md` | Stale root guide flagged by archive review as divergent from current schema representation. |
| `steps.md` | Legacy pipeline walkthrough referenced only by archived reviews. |
| `timing_notes.md` | Loose timing note with no live references. |
| `check_heading.py` | Hardcoded local-path diagnostic for one historical recording. |
| `inspect_enum_structure.py` | One-off enum migration inspection script tied to the deleted enum note cluster. |
| `patch_pose_schema.py` | One-off legacy migration script; pose schema propagation/backfill behavior now lives in package code/tests. |
| `speed_test.py` | One-off local disk speed probe with hardcoded Desktop output. |
| `speed_test_zarr.py` | Commented one-off Zarr speed probe with hardcoded Desktop paths. |
| `test_frame_sampling.py` | Root-level ad hoc test superseded by unit tests under `tests/unit/fisheye/`. |
| `test_frame_sampling_simple.py` | Root-level ad hoc test superseded by unit tests under `tests/unit/fisheye/`. |
| `test_pose_schema_loading.py` | Root-level ad hoc schema test superseded by package tests. |
| `trtexec_diagnostics.py` | One-off TensorRT diagnostic; current TensorRT parsing is covered by unit tests. |
| `verify_enum_format.py` | One-off enum migration verifier tied to deleted enum note cluster. |
| `visual_angle_visualizer.py` | Standalone visual calculator, referenced only as loose scratch in archive reviews. |
| `scraps/` | Hardcoded coordinate repair experiments with no inbound references. |
| `trtexec_output/` | Generated JPG output artifacts explicitly flagged by review as tracked generated files. |

## Kept

| Path | Reason |
| --- | --- |
| `AGENTS.md` | Active agent/worktree instructions. |
| `README.md` | Active repository landing page; expanded in this sweep. |
| `setup.py` | Packaging compatibility shim; pyproject remains authoritative but this file is intentionally left alone. |
| `HANDOFF_2026-07-04.md` | Referenced by active agent briefs for operating notes. |
| `CRIMSON_DATA_FORMATS.md` | Unreferenced from code, but may still be useful as external Crimson/Palette format context; left for maintainer decision. |
| `check_frame_gaps.py` | Referenced by `docs/identity_lineage_staleness_review.md` as frame-domain archaeology; left until a FrameDomains follow-up replaces or archives it. |
| `diagnostics/` | Contains documented sandbox/zarr/asyncio reproduction scripts, not generic scratch. |

## Verification

- `rg` found no live `src/` or `tests/` imports of the deleted root files.
- `rg` found no `fisheye.io` imports before removing the empty package.
- Generated `trtexec_output/*.jpg` files were referenced only by review/archive cleanup notes and `src/test_tensort.py`'s default output directory string.
