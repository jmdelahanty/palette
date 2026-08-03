from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.eye_angle_schema import (
    EyeAngleDimensions,
    build_eye_angle_array_declarations,
    eye_angle_dimensions_from_run_attrs,
    eye_angle_array_schema_manifest,
    eye_angle_channel_metadata,
    expected_eye_angle_channel_index_attrs,
    expected_eye_angle_channel_index_content,
    is_current_eye_angle_run_contract,
    is_supported_legacy_eye_angle_run,
    validate_eye_angle_compact_arrays,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import FLOAT32, ArrayContract
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode


def _text_rows(values: tuple[str, ...], width: int) -> np.ndarray:
    result = np.zeros((len(values), width), dtype=np.uint8)
    for row, value in enumerate(values):
        payload = value.encode("utf-8")
        result[row, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return result


def _valid_arrays(dimensions: EyeAngleDimensions) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    resolved = dimensions.contract_dimensions
    for declaration in build_eye_angle_array_declarations():
        shape = tuple(
            int(resolved[value]) if isinstance(value, str) else int(value)
            for value in declaration.contract.shape_template
        )
        arrays[declaration.path] = np.zeros(
            shape,
            dtype=np.dtype(declaration.contract.dtype.numpy_dtype),
        )
    declarations = {
        declaration.path: declaration
        for declaration in build_eye_angle_array_declarations()
    }
    for path, values in expected_eye_angle_channel_index_content(
        angle_block_width=dimensions.angle_block_width
    ).items():
        if values and type(values[0]) is bool:
            arrays[path] = np.asarray(values, dtype=bool)
        else:
            width = declarations[path].contract.shape_template[1]
            assert type(width) is int
            arrays[path] = _text_rows(values, width)
    return arrays


def _valid_index_attrs(
    dimensions: EyeAngleDimensions,
) -> dict[str, dict[str, object]]:
    return expected_eye_angle_channel_index_attrs(
        angle_block_width=dimensions.angle_block_width
    )


def test_shared_analysis_declaration_is_strict_and_json_safe() -> None:
    contract = ArrayContract(
        schema_id="test.array",
        schema_version=1,
        dtype=FLOAT32,
        shape_template=("n_rows",),
        axis_names=("row",),
        description="test",
    )
    declaration = AnalysisArrayDeclaration(
        path="values",
        contract=contract,
        required=True,
        access_pattern=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        fill_semantics="nan_means_invalid",
        null_semantics="none",
        physical_policy_owner="test_policy",
        byte_planner_adopted=False,
    )
    assert declaration.as_manifest()["authority_role"] == "scientific_authority"
    assert declaration.as_manifest()["byte_planner_adopted"] is False

    kwargs = {**declaration.__dict__}
    for path in (
        " ../values",
        "../values",
        "a\\b",
        "a\nb",
        "/values",
        "a /b",
        "a b",
    ):
        with pytest.raises(ValueError):
            AnalysisArrayDeclaration(**{**kwargs, "path": path})
    with pytest.raises(TypeError):
        AnalysisArrayDeclaration(**{**kwargs, "contract": object()})
    with pytest.raises(ValueError):
        AnalysisArrayDeclaration(**{**kwargs, "authority_role": "near_alias"})
    with pytest.raises(TypeError):
        AnalysisArrayDeclaration(**{**kwargs, "byte_planner_adopted": 0})
    for invalid_path in (Path("values"), b"values"):
        with pytest.raises(TypeError, match="exact str"):
            AnalysisArrayDeclaration(**{**kwargs, "path": invalid_path})

    class PathText(str):
        pass

    with pytest.raises(TypeError, match="exact str"):
        AnalysisArrayDeclaration(**{**kwargs, "path": PathText("values")})


def test_exact_compact_eye_angle_inventory_and_manifest_validate() -> None:
    dimensions = EyeAngleDimensions(
        n_roi_rows=2,
        n_frames=3,
    )
    declarations = build_eye_angle_array_declarations()
    assert len(declarations) == 41
    assert all(item.byte_planner_adopted is False for item in declarations)
    assert {
        item.physical_policy_owner
        for item in declarations
        if item.path in {"roi_angles", "frame_angles"}
    } == {"eye_angle_materializer_explicit_semantic_shards"}

    arrays = _valid_arrays(dimensions)
    manifest = eye_angle_array_schema_manifest(dimensions)
    assert not validate_eye_angle_compact_arrays(
        arrays,
        dimensions=dimensions,
        persisted_manifest=manifest,
        channel_index_attrs=_valid_index_attrs(dimensions),
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (lambda arrays: arrays.pop("roi_angles"), "missing_required_array"),
        (
            lambda arrays: arrays.__setitem__(
                "unexpected", np.zeros((2,), dtype=np.float32)
            ),
            "unexpected_array",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "frame_angles",
                arrays["frame_angles"].astype(np.float64),
            ),
            "array_contract_violation",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "roi_vectors", np.zeros((2, 2), dtype=np.float32)
            ),
            "array_contract_violation",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "frame_vectors", np.zeros((3, 2, 2), dtype=np.float32)
            ),
            "unexpected_array",
        ),
        (
            lambda arrays: arrays["angle_channel_index/name"].__setitem__(
                0, np.zeros(256, dtype=np.uint8)
            ),
            "channel_index_content_mismatch",
        ),
        (
            lambda arrays: arrays["angle_channel_index/frame_available"].__setitem__(
                0, ~arrays["angle_channel_index/frame_available"][0]
            ),
            "channel_availability_mismatch",
        ),
        (
            lambda arrays: arrays["qa_channel_index/dtype"].__setitem__(
                0, np.zeros(64, dtype=np.uint8)
            ),
            "channel_index_content_mismatch",
        ),
        (
            lambda arrays: arrays["vector_channel_index/representation"].__setitem__(
                0, np.zeros(256, dtype=np.uint8)
            ),
            "channel_index_content_mismatch",
        ),
    ),
)
def test_exact_compact_eye_angle_validator_rejects_tampering(
    mutation, expected_code: str
) -> None:
    dimensions = EyeAngleDimensions(2, 3)
    arrays = _valid_arrays(dimensions)
    mutation(arrays)
    issues = validate_eye_angle_compact_arrays(
        arrays,
        dimensions=dimensions,
        persisted_manifest=eye_angle_array_schema_manifest(dimensions),
        channel_index_attrs=_valid_index_attrs(dimensions),
    )
    assert expected_code in {issue.code for issue in issues}


def test_exact_manifest_tampering_and_frame_time_rules_fail_closed() -> None:
    dimensions = EyeAngleDimensions(2, 3)
    arrays = _valid_arrays(dimensions)
    manifest = deepcopy(eye_angle_array_schema_manifest(dimensions))
    manifest["byte_planner_adopted"] = True
    issues = validate_eye_angle_compact_arrays(
        arrays,
        dimensions=dimensions,
        persisted_manifest=manifest,
        channel_index_attrs=_valid_index_attrs(dimensions),
    )
    assert "array_schema_manifest_mismatch" in {issue.code for issue in issues}

    optional = EyeAngleDimensions(0, 3)
    optional_arrays = _valid_arrays(optional)
    assert not validate_eye_angle_compact_arrays(
        optional_arrays,
        dimensions=optional,
        persisted_manifest=eye_angle_array_schema_manifest(optional),
        channel_index_attrs=_valid_index_attrs(optional),
    )
    required_arrays = _valid_arrays(dimensions)
    required_arrays.pop("support/frame_time_seconds")
    issues = validate_eye_angle_compact_arrays(
        required_arrays,
        dimensions=dimensions,
        persisted_manifest=eye_angle_array_schema_manifest(dimensions),
        channel_index_attrs=_valid_index_attrs(dimensions),
    )
    assert "missing_required_array" in {issue.code for issue in issues}
    with pytest.raises(ValueError, match="positive frame count"):
        EyeAngleDimensions(0, 0)


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    (
        ("num_detections", "2", "num_detections"),
        ("num_detections", 2.0, "num_detections"),
        ("num_detections", True, "num_detections"),
        ("num_frames", "3", "num_frames"),
        ("num_frames", 3.0, "num_frames"),
        ("num_frames", True, "num_frames"),
    ),
)
def test_run_dimensions_reject_numeric_coercion(
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    attrs: dict[str, object] = {
        "num_detections": 2,
        "num_frames": 3,
        "angle_column_order_contract": {"semantic_bundle_width": 16},
    }
    attrs[field_name] = replacement
    with pytest.raises(ValueError, match=message):
        eye_angle_dimensions_from_run_attrs(attrs)


def test_heading_dense_channel_is_an_exact_body_frame_compatibility_alias() -> None:
    metadata = eye_angle_channel_metadata(("heading_deg",))
    assert metadata == {
        "name": ("heading_deg",),
        "representation": ("body_frame_compatibility_alias",),
        "eye": ("none",),
        "value_kind": ("heading",),
        "units": ("deg",),
        "source_channel": ("support/body_frame/heading_deg",),
        "formula": ("exact_value_alias(support/body_frame/heading_deg)",),
        "compatibility_alias_of": ("support/body_frame/heading_deg",),
    }
def test_current_run_contract_requires_exact_scalar_types() -> None:
    attrs: dict[str, object] = {
        "schema_id": "analysis.eye_angle_runs",
        "schema_version": 7,
        "layout": "compact_dense_v2",
    }
    assert is_current_eye_angle_run_contract(attrs)
    for replacement in (7.0, "7", True):
        mutated = {**attrs, "schema_version": replacement}
        assert not is_current_eye_angle_run_contract(mutated)


@pytest.mark.parametrize(
    "attrs",
    (
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 2,
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 2,
            "layout": "hierarchical_v1",
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 3,
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 3,
            "layout": "hierarchical_v1",
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 4,
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 4,
            "layout": "hierarchical_v1",
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 5,
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 5,
            "layout": "compact_dense_v2",
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 6,
            "layout": "hierarchical_v1",
        },
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 6,
            "layout": "compact_dense_v2",
        },
    ),
)
def test_legacy_eye_angle_contracts_are_an_exact_closed_allowlist(
    attrs: dict[str, object],
) -> None:
    assert is_supported_legacy_eye_angle_run(attrs)


@pytest.mark.parametrize("version", (1, 7, 8, "6", 6.0, True))
def test_unknown_or_wrong_type_legacy_eye_angle_contracts_are_rejected(
    version: object,
) -> None:
    assert not is_supported_legacy_eye_angle_run(
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": version,
            "layout": "compact_dense_v2",
        }
    )


@pytest.mark.parametrize(
    "group_name",
    ("angle_channel_index", "vector_channel_index", "qa_channel_index"),
)
def test_exact_channel_index_group_attrs_reject_tampering(group_name: str) -> None:
    dimensions = EyeAngleDimensions(2, 3)
    attrs = _valid_index_attrs(dimensions)
    attrs[group_name]["unexpected"] = True
    issues = validate_eye_angle_compact_arrays(
        _valid_arrays(dimensions),
        dimensions=dimensions,
        persisted_manifest=eye_angle_array_schema_manifest(dimensions),
        channel_index_attrs=attrs,
    )
    assert any(
        issue.code == "channel_index_attrs_mismatch" and issue.path == group_name
        for issue in issues
    )


@pytest.mark.parametrize("replacement", (True, 1.0))
def test_exact_manifest_comparison_preserves_json_scalar_types(
    replacement: object,
) -> None:
    dimensions = EyeAngleDimensions(2, 3)
    manifest = deepcopy(eye_angle_array_schema_manifest(dimensions))
    manifest["schema_version"] = replacement
    issues = validate_eye_angle_compact_arrays(
        _valid_arrays(dimensions),
        dimensions=dimensions,
        persisted_manifest=manifest,
        channel_index_attrs=_valid_index_attrs(dimensions),
    )
    assert "array_schema_manifest_mismatch" in {issue.code for issue in issues}
