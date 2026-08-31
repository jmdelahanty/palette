"""Profile-aware keypoint row-success authority resolution.

Strict keypoint publications own one success meaning per declared grammar.
Compatibility aliases remain readable only for artifacts without a manifest;
once a manifest selects a profile, validation failures never fall through to a
different grammar.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import zarr

from .zarr.keypoint_manifest import (
    KEYPOINT_RUN_MANIFEST_SCHEMA_ID,
    validate_keypoint_run_manifest,
)
from .zarr.refined_keypoint_manifest import (
    REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_ID,
    validate_refined_keypoint_run_manifest,
)

RAW_KEYPOINT_V2_SUCCESS_DATASET = "pose_success"
REFINED_KEYPOINT_V2_SUCCESS_DATASET = "usable_keypoints"
LEGACY_RAW_KEYPOINT_SUCCESS_DATASET = "detection_success"
LEGACY_KEYPOINT_SUCCESS_DATASET_CANDIDATES = (
    "usable_keypoints",
    "detection_success",
    "refined_success",
    "source_success",
)


def _read_exact_success_array(
    group: zarr.Group,
    *,
    run_name: str,
    dataset_name: str,
    expected_rows: int | None = None,
) -> np.ndarray:
    node = group.get(dataset_name)
    if node is None:
        raise ValueError(
            f"Keypoint run {run_name!r} is missing its exact "
            f"{dataset_name} success authority."
        )
    if np.dtype(node.dtype) != np.dtype(bool):
        raise ValueError(
            f"Keypoint run {run_name!r} {dataset_name} must use exact bool dtype."
        )
    success = np.asarray(node[:])
    if success.ndim != 1:
        raise ValueError(
            f"Keypoint run {run_name!r} {dataset_name} must be one-dimensional."
        )
    if expected_rows is not None and success.shape != (int(expected_rows),):
        raise ValueError(
            f"Keypoint run {run_name!r} {dataset_name} has shape "
            f"{success.shape!r}; expected ({int(expected_rows)},)."
        )
    keypoints_roi = group.get("keypoints_roi")
    if keypoints_roi is not None and int(keypoints_roi.shape[0]) != int(
        success.shape[0]
    ):
        raise ValueError(
            f"Keypoint run {run_name!r} {dataset_name} row count differs "
            "from keypoints_roi."
        )
    return np.asarray(success, dtype=bool)


def _manifest_instance_count(manifest: Mapping[str, Any]) -> int:
    try:
        value = manifest["payload"]["logical_schema"]["dimensions"]["n_instances"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Keypoint run manifest lacks its exact n_instances dimension."
        ) from exc
    if type(value) is not int or value < 0:
        raise ValueError(
            "Keypoint run manifest n_instances must be a nonnegative integer."
        )
    return int(value)


def resolve_keypoint_success_array(
    group: zarr.Group,
    run_name: str,
) -> tuple[np.ndarray, str]:
    """Return the selected profile's exact row-success array and dataset name."""

    manifest = group.attrs.get("run_manifest")
    if manifest is not None:
        if not isinstance(manifest, Mapping):
            raise ValueError(
                f"Keypoint run {run_name!r} has a malformed run_manifest."
            )
        schema_id = manifest.get("schema_id")
        if schema_id == KEYPOINT_RUN_MANIFEST_SCHEMA_ID:
            errors = validate_keypoint_run_manifest(manifest)
            if errors:
                raise ValueError(
                    f"Keypoint run {run_name!r} raw-v2 manifest is invalid: "
                    + "; ".join(errors)
                )
            legacy_aliases = [
                name
                for name in LEGACY_KEYPOINT_SUCCESS_DATASET_CANDIDATES
                if name in group
            ]
            if legacy_aliases:
                raise ValueError(
                    f"Keypoint run {run_name!r} raw-v2 profile contains "
                    "forbidden legacy success aliases: "
                    + ", ".join(legacy_aliases)
                )
            return (
                _read_exact_success_array(
                    group,
                    run_name=run_name,
                    dataset_name=RAW_KEYPOINT_V2_SUCCESS_DATASET,
                    expected_rows=_manifest_instance_count(manifest),
                ),
                RAW_KEYPOINT_V2_SUCCESS_DATASET,
            )
        if schema_id == REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_ID:
            errors = validate_refined_keypoint_run_manifest(manifest)
            if errors:
                raise ValueError(
                    f"Keypoint run {run_name!r} refined-v2 manifest is invalid: "
                    + "; ".join(errors)
                )
            return (
                _read_exact_success_array(
                    group,
                    run_name=run_name,
                    dataset_name=REFINED_KEYPOINT_V2_SUCCESS_DATASET,
                    expected_rows=_manifest_instance_count(manifest),
                ),
                REFINED_KEYPOINT_V2_SUCCESS_DATASET,
            )
        raise ValueError(
            f"Keypoint run {run_name!r} declares unsupported success-authority "
            f"profile {schema_id!r}; compatibility fallback is disabled."
        )

    for dataset_name in LEGACY_KEYPOINT_SUCCESS_DATASET_CANDIDATES:
        if dataset_name in group:
            return (
                _read_exact_success_array(
                    group,
                    run_name=run_name,
                    dataset_name=dataset_name,
                ),
                dataset_name,
            )
    raise ValueError(
        f"Keypoint run {run_name!r} missing success flags "
        f"({RAW_KEYPOINT_V2_SUCCESS_DATASET}, "
        f"{', '.join(LEGACY_KEYPOINT_SUCCESS_DATASET_CANDIDATES)}); "
        "cannot assign eyes_union."
    )


def resolve_raw_keypoint_success_array(
    group: zarr.Group,
    run_name: str,
) -> tuple[np.ndarray, str]:
    """Resolve only a raw-keypoint success profile.

    Manifest-free canonical-v1 compatibility artifacts own the exact
    ``detection_success`` leaf.  A manifested source is delegated to the
    profile resolver and must resolve to raw-v2 ``pose_success``; refined and
    unknown profiles cannot satisfy this boundary.
    """

    if group.attrs.get("run_manifest") is None:
        if group.get(LEGACY_RAW_KEYPOINT_SUCCESS_DATASET) is None:
            raise ValueError(
                f"Keypoint run {run_name!r} legacy raw-keypoint profile requires "
                "the exact detection_success leaf; fallback success aliases are "
                "unsupported."
            )
        return (
            _read_exact_success_array(
                group,
                run_name=run_name,
                dataset_name=LEGACY_RAW_KEYPOINT_SUCCESS_DATASET,
            ),
            LEGACY_RAW_KEYPOINT_SUCCESS_DATASET,
        )
    success, dataset_name = resolve_keypoint_success_array(group, run_name)
    if dataset_name != RAW_KEYPOINT_V2_SUCCESS_DATASET:
        raise ValueError(
            f"Keypoint run {run_name!r} resolves {dataset_name!r}, but this "
            "boundary requires a raw-keypoint success profile."
        )
    return success, dataset_name


__all__ = [
    "LEGACY_RAW_KEYPOINT_SUCCESS_DATASET",
    "LEGACY_KEYPOINT_SUCCESS_DATASET_CANDIDATES",
    "RAW_KEYPOINT_V2_SUCCESS_DATASET",
    "REFINED_KEYPOINT_V2_SUCCESS_DATASET",
    "resolve_raw_keypoint_success_array",
    "resolve_keypoint_success_array",
]
