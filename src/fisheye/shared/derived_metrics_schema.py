from __future__ import annotations

from typing import Any, Sequence


DERIVED_METRICS_SCHEMA_VERSION = 1


def _normalized_metric_labels(keypoint_labels: Sequence[str]) -> list[str]:
    labels: list[str] = []
    for index, raw_label in enumerate(list(keypoint_labels)[:3]):
        label = str(raw_label).strip()
        labels.append(label or f"k{index}")
    while len(labels) < 3:
        labels.append(f"k{len(labels)}")
    return labels


def build_refined_keypoint_derived_metrics_schema(
    *,
    keypoint_labels: Sequence[str],
) -> dict[str, Any]:
    triangle_labels = _normalized_metric_labels(keypoint_labels)
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
                    "indices": [0, 1, 2],
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
