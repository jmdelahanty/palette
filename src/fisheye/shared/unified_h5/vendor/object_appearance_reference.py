#!/usr/bin/env python3
"""Portable reference evaluator for Citrus object-appearance contract v1.

This is an independent, standard-library-only oracle.  It evaluates one local
underlay/object sample; reconstructing an entire rendered frame additionally
requires the dependencies named by the recording's appearance replay manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROFILE_SCHEMA_ID = "citrus.object_appearance_profile"
REQUEST_SCHEMA_ID = "citrus.object_appearance_replay_request"
RESULT_SCHEMA_ID = "citrus.object_appearance_replay_result"
MANIFEST_SCHEMA_ID = "citrus.visual_appearance_replay_dependency_manifest"
SCHEMA_VERSION = 1

LUMA_R = 0.2126
LUMA_G = 0.7152
LUMA_B = 0.0722
MINIMUM_REFERENCE_LUMINANCE = 1.0e-6

PROFILE_FIELDS = {
    "schema_id",
    "schema_version",
    "scope_id",
    "contrast_model",
    "calculation_domain",
    "reference",
    "alpha_policy",
    "invalid_reference_policy",
    "baseline_contrast",
    "peak_contrast",
    "baseline_chromatic_strength",
    "peak_chromatic_strength",
    "modulation",
}
MODULATION_FIELDS = {"mode", "curve", "invalid_drive_policy"}
REQUEST_FIELDS = {
    "schema_id",
    "schema_version",
    "appearance_profile",
    "drive",
    "local_underlay_rgba",
    "authored_target_rgba",
}
DRIVE_FIELDS = {"value", "valid"}
MANIFEST_FIELDS = {
    "schema_id",
    "schema_version",
    "component_id",
    "runtime_state_authority",
    "physical_claim",
    "adapters",
}
ADAPTER_FIELDS = {
    "renderer_adapter_id",
    "appearance_formula_id",
    "quantization_id",
    "reference_evaluation",
    "audit_witness_policy_id",
    "composition_order_id",
    "dependencies",
}
DEPENDENCY_FIELDS = {"role", "object_ref", "required_attributes"}
DEPENDENCY_ROLES = [
    "appearance_runtime_state",
    "authored_protocol_definition",
    "appearance_profile_index",
    "protocol_semantic_identity",
    "chaser_runtime_state",
    "stimulus_frame_timeline",
    "renderer_snapshot",
    "runtime_geometry",
    "presentation_timing",
    "appearance_enum_definitions",
    "build_identity",
]


class ContractError(ValueError):
    """The replay request does not satisfy the closed v1 contract."""


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _f32_add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _f32_mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _clamp01(value: float) -> float:
    value = _f32(value)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _interpolate(start: float, end: float, amount: float) -> float:
    return _f32_add(start, _f32_mul(amount, _f32_add(end, -_f32(start))))


def _luminance(rgba: Iterable[float]) -> float:
    r, g, b, _ = rgba
    return _f32_add(
        _f32_add(_f32_mul(LUMA_R, r), _f32_mul(LUMA_G, g)),
        _f32_mul(LUMA_B, b),
    )


def _require_exact_fields(value: Dict[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        missing = sorted(fields - set(value))
        unknown = sorted(set(value) - fields)
        raise ContractError(
            f"{label}_fields_mismatch:missing={missing}:unknown={unknown}"
        )


def _rgba(value: Any, label: str) -> List[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ContractError(f"{label}_must_be_rgba_array")
    result = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ContractError(f"{label}_component_must_be_number")
        component = _f32(component)
        if not math.isfinite(component) or component < 0.0 or component > 1.0:
            raise ContractError(f"{label}_component_out_of_range")
        result.append(component)
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label}_must_be_number")
    result = _f32(value)
    if not math.isfinite(result):
        raise ContractError(f"{label}_must_be_finite")
    return result


def validate_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        raise ContractError("appearance_profile_must_be_object_or_null")
    _require_exact_fields(profile, PROFILE_FIELDS, "appearance_profile")
    if profile["schema_id"] != PROFILE_SCHEMA_ID or profile["schema_version"] != 1:
        raise ContractError("unsupported_appearance_profile_schema")
    scope_id = profile["scope_id"]
    if (
        not isinstance(scope_id, str)
        or not 1 <= len(scope_id.encode("ascii", errors="ignore")) <= 64
        or not scope_id.isascii()
        or any(not (ch.isalnum() or ch in "._:-") for ch in scope_id)
    ):
        raise ContractError("invalid_appearance_scope_id")
    constants = {
        "contrast_model": "signed_weber_luminance_v1",
        "calculation_domain": "renderer_nominal_code_luminance_v1",
        "reference": "local_rendered_background_v1",
        "alpha_policy": "preserve_authored_alpha_v1",
        "invalid_reference_policy": "preserve_underlay_v1",
    }
    for field, expected in constants.items():
        if profile[field] != expected:
            raise ContractError(f"unsupported_{field}")
    modulation = profile["modulation"]
    if not isinstance(modulation, dict):
        raise ContractError("modulation_must_be_object")
    _require_exact_fields(modulation, MODULATION_FIELDS, "modulation")
    if modulation["mode"] not in {"constant", "normalized_drive"}:
        raise ContractError("unsupported_modulation_mode")
    if modulation["curve"] not in {"linear_v1", "smoothstep_v1"}:
        raise ContractError("unsupported_modulation_curve")
    if modulation["invalid_drive_policy"] != "return_to_baseline_v1":
        raise ContractError("unsupported_invalid_drive_policy")

    result = dict(profile)
    result["modulation"] = dict(modulation)
    for field in ("baseline_contrast", "peak_contrast"):
        result[field] = _finite_number(profile[field], field)
        if result[field] < -1.0:
            raise ContractError(f"{field}_below_signed_weber_minimum")
    for field in ("baseline_chromatic_strength", "peak_chromatic_strength"):
        result[field] = _finite_number(profile[field], field)
        if not 0.0 <= result[field] <= 1.0:
            raise ContractError(f"{field}_outside_unit_interval")
    return result


def _is_absolute_internal_h5_path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and (
            value == "/"
            or (
                len(value) > 1
                and value.startswith("/")
                and not value.endswith("/")
                and "//" not in value
                and "/../" not in value
                and "/./" not in value
                and not any(char in value for char in "{}\\\n\r")
            )
        )
    )


def validate_dependency_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the exact v1 adapter/dependency vocabulary.

    This validates the portable declaration itself. H5 consumers must
    additionally resolve each object_ref and every required attribute inside
    the recording before accepting it as replay-complete.
    """
    if not isinstance(manifest, dict):
        raise ContractError("dependency_manifest_must_be_object")
    _require_exact_fields(manifest, MANIFEST_FIELDS, "dependency_manifest")
    if (
        manifest["schema_id"] != MANIFEST_SCHEMA_ID
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["component_id"] != "visual_appearance"
        or manifest["runtime_state_authority"]
        != "recorded_frame_state_v1_not_random_seed_resimulation"
        or manifest["physical_claim"]
        != "renderer_code_values_only_not_emitted_projector_luminance"
    ):
        raise ContractError("unsupported_dependency_manifest_identity")
    adapters = manifest["adapters"]
    if not isinstance(adapters, list) or len(adapters) != 1:
        raise ContractError("invalid_adapter_set")
    adapter = adapters[0]
    if not isinstance(adapter, dict):
        raise ContractError("adapter_must_be_object")
    _require_exact_fields(adapter, ADAPTER_FIELDS, "adapter")
    expected_adapter = {
        "renderer_adapter_id": "chaser_compact_object_cuda_v1",
        "appearance_formula_id": "compact_object_signed_weber_code_luminance_v1",
        "quantization_id": "rgba8_round_half_up_v1",
        "reference_evaluation": "per_pixel_local_underlay_before_foreground_v1",
        "audit_witness_policy_id": "object_center_v1",
        "composition_order_id": "chaser_first_matching_index_v1",
    }
    for field, expected in expected_adapter.items():
        if adapter[field] != expected:
            raise ContractError(f"unsupported_{field}")
    dependencies = adapter["dependencies"]
    if not isinstance(dependencies, list) or len(dependencies) != len(
        DEPENDENCY_ROLES
    ):
        raise ContractError("invalid_dependency_set")
    for index, (dependency, expected_role) in enumerate(
        zip(dependencies, DEPENDENCY_ROLES)
    ):
        if not isinstance(dependency, dict):
            raise ContractError("dependency_must_be_object")
        _require_exact_fields(dependency, DEPENDENCY_FIELDS, "dependency")
        if dependency["role"] != expected_role:
            raise ContractError("invalid_dependency_role_order")
        if not _is_absolute_internal_h5_path(dependency["object_ref"]):
            raise ContractError("invalid_dependency_object_ref")
        attributes = dependency["required_attributes"]
        expected_attributes = (
            ["software_version", "git_commit_hash"]
            if index == len(DEPENDENCY_ROLES) - 1
            else []
        )
        if attributes != expected_attributes:
            raise ContractError("invalid_dependency_required_attributes")
    return manifest


def quantize_rgba8(rgba: Iterable[float]) -> List[int]:
    return [int(_f32(_f32(_clamp01(value) * 255.0) + 0.5)) for value in rgba]


def stored_h5_witness_v1(oracle_result: Dict[str, Any],
                         sample_scope_visible: bool) -> Dict[str, Any]:
    """Map the pre-quantization oracle result to the producer's H5 witness.

    The renderer first emits RGBA8. The logger then converts each byte back to
    float32 code values using a float32 1/255 multiplier and computes the
    stored code luminance from those converted RGB values. Occlusion leaves
    the H5 realized payload canonically zero with realized_valid=0.
    """
    if not isinstance(sample_scope_visible, bool):
        raise ContractError("sample_scope_visible_must_be_boolean")
    rgba8 = oracle_result.get("realized_rgba8")
    if (not isinstance(rgba8, list) or len(rgba8) != 4 or
            any(type(channel) is not int or channel < 0 or channel > 255
                for channel in rgba8)):
        raise ContractError("invalid_oracle_rgba8")
    if not sample_scope_visible:
        return {"realized_valid": False,
                "realized_rgba": [0.0, 0.0, 0.0, 0.0],
                "realized_code_luminance": 0.0}
    byte_to_code = _f32(1.0 / 255.0)
    realized = [_f32_mul(channel, byte_to_code) for channel in rgba8]
    return {"realized_valid": True,
            "realized_rgba": realized,
            "realized_code_luminance": _luminance(realized)}


def evaluate_request(request: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(request, dict):
        raise ContractError("replay_request_must_be_object")
    _require_exact_fields(request, REQUEST_FIELDS, "replay_request")
    if request["schema_id"] != REQUEST_SCHEMA_ID or request["schema_version"] != 1:
        raise ContractError("unsupported_replay_request_schema")
    underlay = _rgba(request["local_underlay_rgba"], "local_underlay_rgba")
    target = _rgba(request["authored_target_rgba"], "authored_target_rgba")
    drive = request["drive"]
    if not isinstance(drive, dict):
        raise ContractError("drive_must_be_object")
    _require_exact_fields(drive, DRIVE_FIELDS, "drive")
    if not isinstance(drive["valid"], bool):
        raise ContractError("drive_valid_must_be_boolean")
    drive_value = _finite_number(drive["value"], "drive_value")

    profile_value = request["appearance_profile"]
    if profile_value is None:
        realized = list(target)
        result = {
            "effective_drive": 0.0,
            "requested_contrast": 0.0,
            "requested_chromatic_strength": 0.0,
            "reference_code_luminance": 0.0,
            "requested_code_luminance": 0.0,
            "realized_rgba": realized,
            "realized_code_luminance": _luminance(realized),
            "drive_used": False,
            "drive_valid": False,
            "invalid_drive_fallback_used": False,
            "reference_valid": True,
            "authored_chromaticity_valid": True,
            "clipped": False,
        }
    else:
        profile = validate_profile(profile_value)
        modulation = profile["modulation"]
        drive_used = modulation["mode"] == "normalized_drive"
        accepted_drive = drive_used and drive["valid"]
        effective_drive = 0.0
        if accepted_drive:
            effective_drive = _clamp01(drive_value)
            if modulation["curve"] == "smoothstep_v1":
                effective_drive = _f32_mul(
                    _f32_mul(effective_drive, effective_drive),
                    _f32_add(3.0, -_f32_mul(2.0, effective_drive)),
                )
        requested_contrast = _interpolate(
            profile["baseline_contrast"],
            profile["peak_contrast"],
            effective_drive,
        )
        requested_chromatic_strength = _interpolate(
            profile["baseline_chromatic_strength"],
            profile["peak_chromatic_strength"],
            effective_drive,
        )
        reference_luminance = _luminance(underlay)
        realized = [underlay[0], underlay[1], underlay[2], target[3]]
        result = {
            "effective_drive": effective_drive,
            "requested_contrast": requested_contrast,
            "requested_chromatic_strength": requested_chromatic_strength,
            "reference_code_luminance": reference_luminance,
            "requested_code_luminance": 0.0,
            "realized_rgba": realized,
            "realized_code_luminance": reference_luminance,
            "drive_used": drive_used,
            "drive_valid": accepted_drive,
            "invalid_drive_fallback_used": drive_used and not accepted_drive,
            "reference_valid": reference_luminance > MINIMUM_REFERENCE_LUMINANCE,
            "authored_chromaticity_valid": False,
            "clipped": False,
        }
        if result["reference_valid"]:
            requested_luminance = _f32_mul(
                reference_luminance, _f32_add(1.0, requested_contrast)
            )
            result["requested_code_luminance"] = requested_luminance
            background_scale = _f32(requested_luminance / reference_luminance)
            intensity = [
                _f32_mul(underlay[index], background_scale) for index in range(3)
            ]
            target_luminance = _luminance(target)
            chromaticity_valid = target_luminance > MINIMUM_REFERENCE_LUMINANCE
            result["authored_chromaticity_valid"] = chromaticity_valid
            chromatic = list(intensity)
            if chromaticity_valid:
                target_scale = _f32(requested_luminance / target_luminance)
                chromatic = [
                    _f32_mul(target[index], target_scale) for index in range(3)
                ]
            strength = _clamp01(requested_chromatic_strength)
            requested_rgb = [
                _interpolate(intensity[index], chromatic[index], strength)
                for index in range(3)
            ]
            result["clipped"] = any(value < 0.0 or value > 1.0 for value in requested_rgb)
            realized = [_clamp01(value) for value in requested_rgb] + [target[3]]
            result["realized_rgba"] = realized
            result["realized_code_luminance"] = _luminance(realized)

    result["realized_rgba8"] = quantize_rgba8(result["realized_rgba"])
    return {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "result": result,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="validate and canonicalize a replay-dependency manifest instead",
    )
    parser.add_argument("input", type=Path, help="closed JSON input")
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.input.read_text(encoding="utf-8"))
        output = (
            validate_dependency_manifest(value)
            if args.manifest
            else evaluate_request(value)
        )
        print(json.dumps(output, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, json.JSONDecodeError, ContractError) as error:
        print(f"object_appearance_replay_failed:{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
