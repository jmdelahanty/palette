from __future__ import annotations

from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.registry.model_resolution import (
    load_subject_mask_model_candidates,
    resolve_best_subject_mask_model,
)


def _final_metrics(
    *,
    dice: float,
    coverage_class: str = "dense_all_components",
    component_coverage_key: str = "body+eyes+swim_bladder",
) -> dict[str, object]:
    groups = component_coverage_key.split("+") if component_coverage_key else []
    return {
        "stage": "completed",
        "best_val_dice": dice,
        "best_epoch": 3,
        "label_schema_id": "subject_v1_union",
        "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
        "subject_mask_model_summary": {
            "label_schema_id": "subject_v1_union",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "coverage_class": coverage_class,
            "component_groups": groups,
            "component_coverage_key": component_coverage_key,
            "available_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "missing_labels": [],
        },
    }


def _record_model(
    registry: Registry,
    *,
    run_id: str,
    model_path: Path,
    dice: float,
    coverage_class: str = "dense_all_components",
) -> None:
    registry.record_training_run(
        run_id=run_id,
        set_id="subject_mask_set",
        task_type="subject_masks",
        config_path=None,
        manifest_path=None,
        model_path=model_path,
        metrics_path=None,
        model_sha256=f"sha-{run_id}",
        status="success",
        final_metrics=_final_metrics(dice=dice, coverage_class=coverage_class),
    )


def test_resolve_subject_mask_model_prefers_highest_dense_success_metric(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    weak_model = tmp_path / "weak.pt"
    strong_model = tmp_path / "strong.pt"
    partial_model = tmp_path / "partial.pt"
    for path in (weak_model, strong_model, partial_model):
        path.write_text("weights", encoding="utf-8")

    _record_model(registry, run_id="subject_masks_weak", model_path=weak_model, dice=0.5)
    _record_model(registry, run_id="subject_masks_strong", model_path=strong_model, dice=0.9)
    _record_model(
        registry,
        run_id="subject_masks_partial",
        model_path=partial_model,
        dice=0.99,
        coverage_class="partial_subject_masks",
    )

    best, candidates = resolve_best_subject_mask_model(registry)

    assert best.run_id == "subject_masks_strong"
    assert best.model_path == str(strong_model)
    assert best.best_metric_value == 0.9
    assert [candidate.run_id for candidate in candidates] == [
        "subject_masks_strong",
        "subject_masks_weak",
    ]
    registry.close()


def test_resolve_subject_mask_model_filters_missing_paths_by_default(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    existing_model = tmp_path / "existing.pt"
    missing_model = tmp_path / "missing.pt"
    existing_model.write_text("weights", encoding="utf-8")

    _record_model(registry, run_id="subject_masks_existing", model_path=existing_model, dice=0.8)
    _record_model(registry, run_id="subject_masks_missing", model_path=missing_model, dice=0.9)

    candidates = load_subject_mask_model_candidates(registry)

    assert [candidate.run_id for candidate in candidates] == ["subject_masks_existing"]
    registry.close()


def test_resolve_subject_mask_model_accepts_exact_set_and_run_filters(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    _record_model(registry, run_id="subject_masks_first", model_path=first, dice=0.8)
    _record_model(registry, run_id="subject_masks_second", model_path=second, dice=0.9)

    best, candidates = resolve_best_subject_mask_model(
        registry,
        set_id="subject_mask_set",
        run_id="subject_masks_first",
    )

    assert best.run_id == "subject_masks_first"
    assert [candidate.run_id for candidate in candidates] == ["subject_masks_first"]
    registry.close()
