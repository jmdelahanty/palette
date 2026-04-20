from __future__ import annotations

from typing import Any, Sequence

from ..pose.schema import resolve_head_triangle_indices

DERIVED_METRICS_SCHEMA_VERSION = 1


def _head_triangle_selector(keypoint_labels: Sequence[str]) -> tuple[list[int], list[str]]:
    labels = [
        str(raw_label).strip() or f"k{index}"
        for index, raw_label in enumerate(keypoint_labels)
    ]
    resolved = resolve_head_triangle_indices(
        labels,
        keypoint_count=len(labels),
        allow_legacy_3point_fallback=True,
    )
    indices = [int(idx) for idx in resolved.as_tuple]
    return indices, [labels[idx] for idx in indices]


def build_refined_keypoint_derived_metrics_schema(
    *,
    keypoint_labels: Sequence[str],
) -> dict[str, Any]:
    triangle_indices, triangle_labels = _head_triangle_selector(keypoint_labels)
    triangle_metric_name = "eye_triangle_geometry"

    return {
        "schema_version": DERIVED_METRICS_SCHEMA_VERSION,
        "entity_kind": "keypoint_roi",
        "metrics": [
            {
                "name": triangle_metric_name,
                "kind": "triangle_3pt",
                "source": {
                    "array": "keypoints_roi",
                    "value_kind": "point_xy",
                    "coordinate_space": "roi_pixels",
                },
                "selectors": {
                    "indices": triangle_indices,
                    "labels": triangle_labels,
                },
                "outputs": [
                    {
                        "name": "triangle_area",
                        "array": "triangle_area",
                        "value_kind": "scalar",
                        "units": "px2",
                    },
                    {
                        "name": "triangle_angles",
                        "array": "triangle_angles",
                        "value_kind": "vector_3",
                        "units": "deg",
                        "order_labels": triangle_labels,
                    },
                    {
                        "name": "min_angle",
                        "array": "min_angle",
                        "value_kind": "scalar",
                        "units": "deg",
                        "reduction": "min",
                    },
                ],
            }
        ],
        "quality_gates": [
            {
                "name": "geometry_valid",
                "kind": "boolean_gate",
                "output": {
                    "array": "geometry_valid",
                    "value_kind": "bool",
                },
                "evaluation": "all_conditions",
                "conditions": [
                    {
                        "metric": triangle_metric_name,
                        "output": "triangle_area",
                        "op": "is_finite",
                    },
                    {
                        "metric": triangle_metric_name,
                        "output": "min_angle",
                        "op": "is_finite",
                    },
                    {
                        "metric": triangle_metric_name,
                        "output": "min_angle",
                        "op": ">=",
                        "threshold_attr": "summary_statistics.min_triangle_angle",
                    },
                    {
                        "metric": triangle_metric_name,
                        "output": "triangle_area",
                        "op": ">=",
                        "threshold_attr": "summary_statistics.min_triangle_area",
                    },
                    {
                        "metric": triangle_metric_name,
                        "output": "triangle_area",
                        "op": "<=",
                        "threshold_attr": "summary_statistics.max_triangle_area",
                        "when_attr_present": True,
                    },
                ],
            }
        ],
    }
