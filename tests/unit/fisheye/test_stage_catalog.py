from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fisheye.registry.maintenance import RECORDING_STEP_NAMES, RECORDING_TUNING_STEP_NAMES
from fisheye.registry.stage_catalog import (
    ALIAS_TO_STAGE_ID,
    STAGE_BY_ID,
    STAGE_SPECS,
    artifact_family_map,
    canonical_stage_id,
    dependency_map,
    invalidation_map,
    recording_status_stage_ids,
    recording_tuning_stage_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _literal_assignment(module_path: Path, assignment_name: str) -> object:
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == assignment_name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{assignment_name} not found in {module_path}")


def _literal_class_assignment(
    module_path: Path,
    class_name: str,
    assignment_name: str,
) -> object:
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for class_node in node.body:
            if not isinstance(class_node, ast.Assign):
                continue
            for target in class_node.targets:
                if isinstance(target, ast.Name) and target.id == assignment_name:
                    return ast.literal_eval(class_node.value)
    raise AssertionError(f"{class_name}.{assignment_name} not found in {module_path}")


def _launcher_stage_info(module_path: Path) -> dict[str, dict[str, object]]:
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "STAGE_INFO" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError("STAGE_INFO is not a dict literal")
        out: dict[str, dict[str, object]] = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            stage_name = ast.literal_eval(key_node)
            if not isinstance(value_node, ast.Call):
                raise AssertionError(f"STAGE_INFO[{stage_name!r}] does not use _launcher_stage_info")
            if not isinstance(value_node.func, ast.Name) or value_node.func.id != "_launcher_stage_info":
                raise AssertionError(f"STAGE_INFO[{stage_name!r}] does not use _launcher_stage_info")
            helper_stage_name = ast.literal_eval(value_node.args[0])
            keywords = {keyword.arg: keyword.value for keyword in value_node.keywords}
            out[stage_name] = {
                "helper_stage_name": helper_stage_name,
                "requires": ast.literal_eval(keywords["requires"]),
            }
        return out
    raise AssertionError(f"STAGE_INFO not found in {module_path}")


def test_stage_catalog_ids_are_unique_and_aliases_resolve() -> None:
    ids = [spec.id for spec in STAGE_SPECS]
    assert len(ids) == len(set(ids))

    for alias, canonical_id in ALIAS_TO_STAGE_ID.items():
        assert canonical_id in STAGE_BY_ID
        assert canonical_stage_id(alias) == canonical_id

    for spec in STAGE_SPECS:
        assert canonical_stage_id(spec.id) == spec.id

    alias_id_collisions = set(ALIAS_TO_STAGE_ID).intersection(STAGE_BY_ID)
    assert alias_id_collisions == set()


def test_stage_catalog_covers_registry_recording_step_names() -> None:
    assert RECORDING_STEP_NAMES == recording_status_stage_ids()
    assert set(RECORDING_STEP_NAMES).issubset(STAGE_BY_ID)


def test_stage_catalog_tuning_stage_projection() -> None:
    assert recording_tuning_stage_ids() == (
        "dish_mask",
        "detection_tuning",
        "keypoint_tuning",
        "subject_mask_tuning",
        "eye_mask_tuning",
        "subdish_mask_tuning",
    )
    assert RECORDING_TUNING_STEP_NAMES == recording_tuning_stage_ids()


def test_stage_catalog_dependency_and_invalidation_maps_are_canonical() -> None:
    deps = dependency_map()
    invalidates = invalidation_map()

    assert deps["refined_detect"] == ("detect_quality",)
    assert invalidates["detect_quality"] == ("refined_detect",)
    assert deps["arena_assignment"] == ("refined_detect",)
    assert deps["tracks"] == ("arena_assignment",)
    assert invalidates["arena_assignment"] == ("tracks",)

    for spec in STAGE_SPECS:
        for dep in spec.depends_on:
            assert dep in STAGE_BY_ID
            assert dep not in ALIAS_TO_STAGE_ID
        for downstream in spec.invalidates:
            assert downstream in STAGE_BY_ID
            assert downstream not in ALIAS_TO_STAGE_ID


def test_stage_catalog_declared_artifact_families_are_unique() -> None:
    families: dict[str, str] = {}
    for stage_id, stage_families in artifact_family_map().items():
        for family in stage_families:
            assert family not in families, (
                f"{family!r} declared for both {families[family]!r} and {stage_id!r}"
            )
            families[family] = stage_id

    assert artifact_family_map()["arena_assignment"] == ("arena_assignment_runs",)


def test_current_pipeline_stage_names_resolve_or_are_intentionally_legacy() -> None:
    pipeline_path = REPO_ROOT / "src/fisheye/core/pipeline.py"
    stage_order = set(_literal_class_assignment(pipeline_path, "Pipeline", "STAGE_ORDER"))
    stage_deps = _literal_class_assignment(pipeline_path, "Pipeline", "STAGE_DEPENDENCIES")
    assert isinstance(stage_deps, dict)

    stage_names = set(stage_order)
    for stage_name, deps in stage_deps.items():
        stage_names.add(stage_name)
        stage_names.update(deps)

    unresolved = set()
    for stage_name in stage_names:
        try:
            canonical_stage_id(stage_name)
        except KeyError:
            unresolved.add(stage_name)

    assert unresolved == set()

    pipeline_module = __import__(
        "fisheye.core.pipeline",
        fromlist=["Pipeline"],
    )
    stage_canonical_ids = pipeline_module.Pipeline.STAGE_CANONICAL_IDS
    assert stage_canonical_ids == {
        stage_name: canonical_stage_id(stage_name)
        for stage_name in stage_order
    }


def test_current_launcher_stage_names_resolve_or_are_intentionally_legacy() -> None:
    launcher_path = REPO_ROOT / "src/fisheye/cli/interactive_launcher.py"
    stage_order = set(_literal_assignment(launcher_path, "STAGE_ORDER"))
    stage_info = _launcher_stage_info(launcher_path)

    stage_names = set(stage_order).union(stage_info)
    for info in stage_info.values():
        stage_names.update(info.get("requires", ()))

    unresolved = set()
    for stage_name in stage_names:
        try:
            canonical_stage_id(stage_name)
        except KeyError:
            unresolved.add(stage_name)

    assert unresolved == set()
    for stage_name in stage_order:
        assert stage_info[stage_name]["helper_stage_name"] == stage_name
        assert canonical_stage_id(stage_name)


def test_unknown_stage_alias_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown stage id or alias"):
        canonical_stage_id("definitely_not_a_stage")
