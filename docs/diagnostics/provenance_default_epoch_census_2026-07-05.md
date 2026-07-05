# Provenance Default Epoch Census

<!-- contract-meta
status: implemented
created: 2026-07-05
owner: jeremy
related: docs/provenance_finalization_enforcement_design.md,
         docs/provenance_enforcement_roadmap.md,
         agents_todo/brief_review_remediation_wave_2026-07-05.md
-->

## Summary

Slice C's default epoch bump is ready and implemented.

The initial census in this document found many live writers that created run parents
through the default epoch and then called `mark_run_complete(...)` without
`run_provenance`. Those writers have since been instrumented with writer-side
provenance, and the static census is now green: there are zero completion calls without
either explicit provenance or a recorded legacy-maintenance bypass.

`require_runs_parent(...)` now stamps new empty parents with
`palette_completion_epoch = 2` by default. Existing parents with children remain
grandfathered until an explicit backfill/migration stamps their epoch.

## Census Method

Static AST census over `src/`:

```bash
scripts/py - <<'PY'
from __future__ import annotations
import ast
from pathlib import Path
root = Path("src")
mark = []
req = []
for path in root.rglob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name == "mark_run_complete":
            kws = {kw.arg for kw in node.keywords if kw.arg}
            mark.append((str(path), node.lineno, "run_provenance" in kws, "allow_missing_run_provenance" in kws))
        if name == "require_runs_parent":
            kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            epoch = "none"
            if "completion_epoch" in kws:
                epoch = ast.unparse(kws["completion_epoch"])
            req.append((str(path), node.lineno, epoch))
print(len(mark), len(req))
PY
```

## Counts

| Surface | Count |
|---|---:|
| `mark_run_complete(...)` calls in `src/` | 58 |
| Completion calls with explicit `run_provenance` | 56 |
| Completion calls with explicit bypass | 2 |
| Completion calls without provenance/bypass | 0 |
| `require_runs_parent(...)` calls in `src/` | 58 |
| Parent creators with explicit `completion_epoch` | 8 |
| Parent creators relying on the default epoch-2 policy | 50 |

## Already Epoch-2 Instrumented

These are the high-volume paths already using `COMPLETION_EPOCH_REQUIRE_PROVENANCE`
and passing `run_provenance` to completion:

- `src/fisheye/detection/detect_yolo.py`
- `src/fisheye/detection/detect_keypoints_yolo.py`
- `src/fisheye/tracking/crop.py`
- `src/fisheye/utils/run_sam_subject_masks.py`
- `src/fisheye/utils/run_subject_mask_batch_pipeline.py`

The direct registry-model runners synthesize run provenance before invoking those
writers:

- `src/fisheye/utils/run_detect_with_registry_model.py`
- `src/fisheye/utils/run_keypoints_with_registry_model.py`

## Intentional Bypasses

The census finds two completion calls that deliberately bypass provenance validation.
Both are maintenance paths that re-mark existing runs rather than creating new scientific
outputs:

| File | Line | Reason |
|---|---:|---|
| `src/fisheye/utils/backfill_completion_epoch.py` | 354 | Legacy completion backfill re-mark of a pre-provenance run |
| `src/fisheye/utils/refresh_training_review_status.py` | 281 | Training review status refresh re-marks existing runs |

## Parent Creators Relying On Default Epoch

The corresponding parent-creation surface relies on the default epoch at 50 call sites.
Those new parents now enforce provenance immediately because the completion writers have
been instrumented.

Representative groups:

- analysis run parents: kinematics, stimulus, occupancy, shape, bout, chaser-distance
- refinement/review parents: refined detect, refined keypoints, refined subject masks,
  detect quality
- legacy/traditional production parents: traditional detect/keypoints, background,
  subject segmentation
- training/export parents: detect/keypoint/subject-mask training exports, acquisition
  crop pose exports
- profile/diagnostic parents: detection/keypoint/subject-mask profiles, realtime/offline
  comparison

## Recommendation

Keep the global default at epoch 2 for new run parents. Future completion writers should
either pass valid `run_provenance` or, only for legacy maintenance operations that
re-mark existing runs, use `allow_missing_run_provenance=True` with an explicit recorded
reason.

The static census should remain at zero missing provenance/bypass calls. Any new
`mark_run_complete(...)` call without one of those two mechanisms is a regression.
