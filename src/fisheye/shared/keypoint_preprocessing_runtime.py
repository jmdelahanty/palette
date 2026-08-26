"""Resolve immutable keypoint preprocessing into one observed runtime contract.

Keypoint publication profiles may encode their pixel-provider declaration and
their observed model runtime at different nesting levels.  Consumers must not
interpret those profiles independently: this module is the single resolver
used by both coordinate-successor publication and persisted coordinate reload.
"""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.shared.keypoint_terminal_pixel_evidence import (
    DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE,
)
from fisheye.shared.model_input_transform import (
    ModelInputTransform,
    model_input_transform_from_attrs,
)


def resolve_keypoint_preprocessing_runtime(
    preprocessing: Any,
) -> tuple[ModelInputTransform, str]:
    """Return the exact model transform and submitted input mode.

    The ordinary keypoint profile records these fields directly in its
    preprocessing document.  Direct-hybrid finalization records the same
    runtime evidence under ``observed_runtime`` because its outer input mode
    describes the row-signature-bound hybrid pixel provider.  Every supported
    profile is validated at its own full strength before a runtime is returned.
    """

    document = preprocessing.document
    if not isinstance(document, Mapping):
        raise ValueError("Keypoint preprocessing document must be an object.")

    runtime: Mapping[str, Any] | None = None
    if preprocessing.profile_id == DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE:
        if preprocessing.profile_version != 1:
            raise ValueError(
                "Direct-hybrid keypoint preprocessing profile version is unsupported."
            )
        runtime_value = document.get("observed_runtime")
        if (
            not isinstance(runtime_value, Mapping)
            or document.get("evidence_semantics")
            != "observed_completed_inference_runtime_v1"
            or document.get("coordinate_contract_mode") != "legacy_noncanonical"
            or preprocessing.input_mode != "numpy_list"
            or document.get("observed_input_mode_effective") != "numpy-list"
            or runtime_value.get("input_mode_effective") != "numpy-list"
        ):
            raise ValueError(
                "Direct-hybrid keypoint preprocessing runtime evidence is inconsistent."
            )
        runtime = runtime_value
        transform_value = runtime.get("model_input_transform")
        submitted_input_mode = "numpy-list"
    else:
        transform_value = document.get("model_input_transform")
        submitted_input_mode = document.get("model_input_mode")
        if submitted_input_mode is None and preprocessing.input_mode in {
            "numpy-list",
            "tensor",
        }:
            submitted_input_mode = preprocessing.input_mode

    if not isinstance(transform_value, Mapping):
        raise ValueError("Keypoint preprocessing lacks model_input_transform.")
    if submitted_input_mode not in {"numpy-list", "tensor"}:
        raise ValueError(
            "Keypoint preprocessing lacks an exact submitted model input mode."
        )
    transform = model_input_transform_from_attrs(dict(transform_value))

    if runtime is not None:
        expected_shapes = {
            "model_input_shape_hw": list(transform.model_shape),
            "model_network_input_shape_hw": list(transform.model_shape),
            "native_roi_shape_hw": list(transform.native_shape),
        }
        if any(
            runtime.get(name) != expected for name, expected in expected_shapes.items()
        ):
            raise ValueError(
                "Direct-hybrid runtime extents differ from model_input_transform."
            )

    return transform, str(submitted_input_mode)


__all__ = ["resolve_keypoint_preprocessing_runtime"]
