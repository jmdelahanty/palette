from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
    SubjectMaskQualityDimensions,
    SubjectMaskQualityMetricDefinition,
    SubjectMaskQualityProfile,
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_quality_storage import (
    plan_subject_mask_quality_storage,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    derive_subject_mask_frame_row_offsets,
)


def _components() -> SubjectMaskComponentRegistry:
    return SubjectMaskComponentRegistry(("subject_body", "eye_left"))


def _dimensions() -> SubjectMaskQualityDimensions:
    return SubjectMaskQualityDimensions(
        n_frames=4,
        n_rois=4,
        n_channels=2,
        roi_height=16,
        roi_width=20,
        n_component_metrics=2,
        n_observation_metrics=1,
    )


def _profile() -> SubjectMaskQualityProfile:
    return SubjectMaskQualityProfile(
        profile_id="observation_local_subject_mask_qc",
        profile_version=1,
        policy_digest="ab" * 32,
        component_metrics=(
            SubjectMaskQualityMetricDefinition(
                metric_id="area_fraction_roi",
                metric_version=1,
                units="fraction",
                higher_is_worse=False,
                description="Foreground area divided by ROI area.",
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="border_touch_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description="Fraction of foreground boundary on the ROI border.",
            ),
        ),
        observation_metrics=(
            SubjectMaskQualityMetricDefinition(
                metric_id="component_overlap_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description="Cross-component overlap divided by union area.",
            ),
        ),
        component_flag_map={1: "missing_component", 2: "touches_roi_border"},
        observation_flag_map={1: "invalid_component_overlap"},
    )


def _fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    keys = np.asarray([101, 102, 201, 301], dtype=np.uint64)
    component_values = np.asarray(
        [
            [[0.5, 0.0], [0.1, 0.0]],
            [[0.4, 0.2], [np.nan, np.nan]],
            [[0.6, 0.0], [0.2, 0.1]],
            [[0.7, 0.0], [0.2, 0.0]],
        ],
        dtype=np.float32,
    )
    observation_values = np.asarray([[0.0], [np.nan], [0.01], [0.02]], dtype=np.float32)
    arrays = {
        "instance_key": keys.copy(),
        "source_mask_row_ids": np.arange(4, dtype=np.int64),
        "source_acquisition_frame_index": frames.copy(),
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(frames, n_frames=4),
        "component_metric_values": component_values,
        "component_metric_valid": np.isfinite(component_values),
        "observation_metric_values": observation_values,
        "observation_metric_valid": np.isfinite(observation_values),
        "component_quality_flags": np.asarray(
            [[0, 0], [0, 1], [0, 0], [0, 0]], dtype=np.uint16
        ),
        "observation_quality_flags": np.asarray([0, 0, 0, 0], dtype=np.uint16),
        "proposed_component_usable": np.asarray(
            [[True, True], [True, False], [True, True], [True, True]], dtype=bool
        ),
        "proposed_observation_usable": np.asarray([True, True, True, True], dtype=bool),
    }
    source = {
        "instance_key": keys.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "available_channels": np.asarray([True, True], dtype=bool),
    }
    return arrays, source


def test_subject_mask_quality_accepts_multirow_and_empty_frames() -> None:
    arrays, source = _fixture()
    dimensions = _dimensions()
    profile = _profile()
    components = _components()

    SUBJECT_MASK_QUALITY_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        components=components,
        profile=profile,
        source_mask_arrays=source,
    )

    source_reference = SubjectMaskQualitySourceReference(
        run_name="refined_masks_001",
        manifest_digest="11" * 32,
        dense_array_values_sha256="22" * 32,
        component_registry_digest=canonical_json_sha256(components.as_manifest()),
    )
    manifest = SUBJECT_MASK_QUALITY_SCHEMA_V1.as_manifest(
        dimensions=dimensions,
        components=components,
        profile=profile,
        source=source_reference,
    )
    assert arrays["frame_row_offsets"].tolist() == [0, 2, 2, 3, 4]
    assert len(manifest["bindings"]) == 12
    assert manifest["base_path"] == "subject_mask_quality_runs/<run>"
    assert manifest["source"]["run_path"].startswith("refined_subject_masks_runs/")
    assert manifest["invariants"]["instances_per_frame"] == "zero_one_or_many"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (
            lambda arrays, source: arrays["source_mask_row_ids"].__setitem__(2, 1),
            "incomplete_source_row_coverage",
        ),
        (
            lambda arrays, source: source["instance_key"].__setitem__(0, 999),
            "source_mask_binding_mismatch",
        ),
        (
            lambda arrays, source: arrays["component_metric_valid"].__setitem__(
                (0, 0, 0), False
            ),
            "metric_validity_mismatch",
        ),
        (
            lambda arrays, source: arrays["component_quality_flags"].__setitem__(
                (0, 0), 8
            ),
            "undeclared_quality_flag",
        ),
        (
            lambda arrays, source: source["available_channels"].__setitem__(1, False),
            "unavailable_component_proposed_usable",
        ),
    ),
)
def test_subject_mask_quality_rejects_tampering(mutation, expected_code: str) -> None:
    arrays, source = _fixture()
    mutation(arrays, source)

    issues = SUBJECT_MASK_QUALITY_SCHEMA_V1.validate(
        arrays,
        dimensions=_dimensions(),
        components=_components(),
        profile=_profile(),
        source_mask_arrays=source,
    )
    assert expected_code in {issue.code for issue in issues}


def test_subject_mask_quality_rejects_temporal_metrics_without_track_lineage() -> None:
    with pytest.raises(ValueError, match="observation-local"):
        SubjectMaskQualityMetricDefinition(
            metric_id="temporal_area_jump",
            metric_version=1,
            units="fraction",
            higher_is_worse=True,
            description="Not valid without explicit subject lineage.",
        )


def test_subject_mask_quality_storage_is_byte_planned_and_immutable() -> None:
    plans = plan_subject_mask_quality_storage(_dimensions())
    assert tuple(entry.rule.path for entry in plans.entries) == (
        SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths
    )
    assert all(entry.plan.write_mode == "immutable" for entry in plans.entries)
    offsets = next(
        entry for entry in plans.entries if entry.rule.path == "frame_row_offsets"
    )
    assert offsets.plan.access_pattern == "eager"
    assert plans.estimated_stage_objects > len(plans.entries)
