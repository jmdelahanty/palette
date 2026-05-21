from __future__ import annotations

from fisheye.core.pipeline import Pipeline
from fisheye.registry.stage_catalog import canonical_stage_id, dependency_map


KNOWN_PIPELINE_DEPENDENCY_OVERRIDES = {
    # The pipeline still permits crop directly from raw detections; the catalog
    # models the newer refined-detect-first path.
    "crop": (("detect",), ("refined_detect",)),
    # Traditional keypoints historically required background as an extra input.
    "keypoints": (("crop", "background"), ("crop",)),
    # The runtime stage invokes raw eye-mask inference before keypoint review.
    "eye_masks": (("keypoints",), ("refined_keypoints",)),
    # Subject-mask refinement is exposed without its raw subject-mask stage in
    # the legacy pipeline launcher.
    "refined_subject_masks": ((), ("subject_masks",)),
    # Arena assignment and tracking predate the refined-keypoint/arena catalog
    # dependency chain.
    "arena_assignment": (("detect",), ("refined_keypoints",)),
    "tracks": (("keypoints",), ("arena_assignment",)),
}


def _canonical_deps(values: list[str]) -> tuple[str, ...]:
    return tuple(canonical_stage_id(value) for value in values)


def test_pipeline_stage_names_resolve_to_catalog_ids() -> None:
    for stage in Pipeline.STAGE_ORDER:
        canonical_stage_id(stage)


def test_pipeline_dependency_drift_is_explicit() -> None:
    catalog_deps = dependency_map()
    pipeline_deps = {
        canonical_stage_id(stage): _canonical_deps(deps)
        for stage, deps in Pipeline.STAGE_DEPENDENCIES.items()
    }

    unexpected_mismatches = {
        stage: (actual, catalog_deps.get(stage, ()))
        for stage, actual in sorted(pipeline_deps.items())
        if actual != catalog_deps.get(stage, ())
        and stage not in KNOWN_PIPELINE_DEPENDENCY_OVERRIDES
    }
    assert unexpected_mismatches == {}

    observed_overrides = {
        stage: (pipeline_deps[stage], catalog_deps[stage])
        for stage in sorted(KNOWN_PIPELINE_DEPENDENCY_OVERRIDES)
    }
    assert observed_overrides == KNOWN_PIPELINE_DEPENDENCY_OVERRIDES
