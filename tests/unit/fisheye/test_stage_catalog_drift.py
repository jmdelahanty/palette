from __future__ import annotations

import ast
from pathlib import Path

from fisheye.core.pipeline import Pipeline
from fisheye.registry.stage_catalog import canonical_stage_id, dependency_map


REPO_ROOT = Path(__file__).resolve().parents[3]


KNOWN_PIPELINE_DEPENDENCY_OVERRIDES = {
    # INTENT: keep until the frozen legacy orchestrator is retired or split.
    # Current YOLO detection does not universally require background; traditional
    # background-subtraction detection owns that runtime precondition.
    "detect": (("background",), ()),
    # INTENT: converge. The pipeline still permits crop directly from raw
    # detections; the catalog models the newer refined-detect-first path.
    "crop": (("detect",), ("refined_detect",)),
    # INTENT: keep until the traditional keypoint path is retired or split.
    # Traditional keypoints historically required background as an extra input.
    "keypoints": (("crop", "background"), ("crop",)),
    # INTENT: converge. Subject-mask refinement is exposed without its raw
    # subject-mask stage in the legacy pipeline launcher.
    "refined_subject_masks": ((), ("subject_masks",)),
    # INTENT: converge. Arena assignment in the legacy pipeline still starts
    # from raw detect, while the catalog models the canonical refined-detect
    # rowset.
    "arena_assignment": (("detect",), ("refined_detect",)),
    "tracks": (("keypoints",), ("arena_assignment",)),
}


KNOWN_LAUNCHER_DEPENDENCY_OVERRIDES = {
    # INTENT: keep until the launcher grows separate traditional/model detect
    # paths. The catalog models raw detect without a universal background input.
    "detect": (("background",), ()),
    # INTENT: converge. The launcher still exposes crop from raw detect because
    # it wraps the legacy orchestrator; the catalog models refined-detect-first.
    "crop": (("detect",), ("refined_detect",)),
    # INTENT: keep until the traditional keypoint launcher path is retired or
    # split. Traditional keypoints historically required background as an extra
    # input.
    "keypoints": (("crop", "background"), ("crop",)),
    # INTENT: converge. The launcher refine checkbox starts directly from raw
    # detect, while the catalog inserts detect_quality as the curation gate.
    "refined_detect": (("detect",), ("detect_quality",)),
    # INTENT: converge. The launcher still starts arena assignment from raw
    # detect, while the catalog models the canonical refined-detect rowset.
    "arena_assignment": (("detect",), ("refined_detect",)),
    # INTENT: converge. The launcher track checkbox follows the older keypoint
    # dependency, while the catalog models track identity after arena assignment.
    "tracks": (("keypoints",), ("arena_assignment",)),
}


def _launcher_stage_info(module_path: Path) -> dict[str, dict[str, object]]:
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "STAGE_INFO"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError("STAGE_INFO is not a dict literal")
        out: dict[str, dict[str, object]] = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            stage_name = ast.literal_eval(key_node)
            if not isinstance(value_node, ast.Call):
                raise AssertionError(
                    f"STAGE_INFO[{stage_name!r}] does not use _launcher_stage_info"
                )
            if (
                not isinstance(value_node.func, ast.Name)
                or value_node.func.id != "_launcher_stage_info"
            ):
                raise AssertionError(
                    f"STAGE_INFO[{stage_name!r}] does not use _launcher_stage_info"
                )
            keywords = {keyword.arg: keyword.value for keyword in value_node.keywords}
            out[stage_name] = {
                "requires": ast.literal_eval(keywords["requires"]),
            }
        return out
    raise AssertionError(f"STAGE_INFO not found in {module_path}")


def _canonical_deps(values: list[str]) -> tuple[str, ...]:
    return tuple(canonical_stage_id(value) for value in values)


def _dep_set(values: tuple[str, ...]) -> frozenset[str]:
    return frozenset(values)


def test_pipeline_stage_names_resolve_to_catalog_ids() -> None:
    for stage in Pipeline.STAGE_ORDER:
        canonical_stage_id(stage)


def test_pipeline_dependency_drift_is_explicit() -> None:
    catalog_deps = dependency_map()
    pipeline_deps = {
        canonical_stage_id(stage): _canonical_deps(deps)
        for stage, deps in Pipeline.STAGE_DEPENDENCIES.items()
    }

    unexpected_mismatches: dict[str, tuple[frozenset[str], frozenset[str]]] = {
        stage: (_dep_set(actual), _dep_set(catalog_deps.get(stage, ())))
        for stage, actual in sorted(pipeline_deps.items())
        if _dep_set(actual) != _dep_set(catalog_deps.get(stage, ()))
        and stage not in KNOWN_PIPELINE_DEPENDENCY_OVERRIDES
    }
    assert unexpected_mismatches == {}

    observed_overrides = {
        stage: (_dep_set(pipeline_deps[stage]), _dep_set(catalog_deps[stage]))
        for stage in sorted(KNOWN_PIPELINE_DEPENDENCY_OVERRIDES)
    }
    expected_overrides = {
        stage: (_dep_set(actual), _dep_set(catalog))
        for stage, (actual, catalog) in KNOWN_PIPELINE_DEPENDENCY_OVERRIDES.items()
    }
    assert observed_overrides == expected_overrides


def test_launcher_dependency_drift_is_explicit() -> None:
    catalog_deps = dependency_map()
    launcher_path = REPO_ROOT / "src/fisheye/cli/interactive_launcher.py"
    launcher_deps = {
        canonical_stage_id(stage): _canonical_deps(info["requires"])  # type: ignore[arg-type]
        for stage, info in _launcher_stage_info(launcher_path).items()
    }

    unexpected_mismatches: dict[str, tuple[frozenset[str], frozenset[str]]] = {
        stage: (_dep_set(actual), _dep_set(catalog_deps.get(stage, ())))
        for stage, actual in sorted(launcher_deps.items())
        if _dep_set(actual) != _dep_set(catalog_deps.get(stage, ()))
        and stage not in KNOWN_LAUNCHER_DEPENDENCY_OVERRIDES
    }
    assert unexpected_mismatches == {}

    observed_overrides = {
        stage: (_dep_set(launcher_deps[stage]), _dep_set(catalog_deps[stage]))
        for stage in sorted(KNOWN_LAUNCHER_DEPENDENCY_OVERRIDES)
    }
    expected_overrides = {
        stage: (_dep_set(actual), _dep_set(catalog))
        for stage, (actual, catalog) in KNOWN_LAUNCHER_DEPENDENCY_OVERRIDES.items()
    }
    assert observed_overrides == expected_overrides
