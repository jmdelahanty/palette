"""Digest-bound pose-model input and deployment preprocessing contracts.

The model graph alone cannot say which source-pixel window produced its
training examples.  This module binds that missing evidence to one exact model
artifact and derives a fail-closed runtime plan for a native ROI cache.

For historical models, the contract is reconstructed from immutable package
artifacts.  It deliberately distinguishes matching the trained object scale
from reproducing the real source context: padding a smaller ROI to the training
source extent is a diagnostic profile, not an assertion that the missing
camera pixels were present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from fisheye.shared.model_input_transform import (
    ModelInputTransform,
    resolve_model_input_transform,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)


POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID = "palette.pose_model_input_contract"
POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION_V1 = 1
POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION = 2
POSE_MODEL_INPUT_CONTRACT_FILENAME = "pose_model_input_contract.json"
SCALE_MATCHED_RUNTIME_PROFILE_ID = "scale_matched_center_pad_ultralytics_v1"
SCALE_MATCHED_RUNTIME_CLASSIFICATION = "scale_matched_diagnostic_not_training_context"
ULTRALYTICS_NUMPY_LIST_ADAPTER = "ultralytics_numpy_list_predict_v1"
PALETTE_PREPARED_TENSOR_ADAPTER = "palette_prepared_tensor_predict_v1"
POSE_PREPROCESSING_PROBE_SCHEMA_ID = "palette.pose_preprocessing_probe"
POSE_PREPROCESSING_PROBE_SCHEMA_VERSION = 1
POSE_PREPROCESSING_PROBE_PATTERN = "uint8_luma_mod_251_x3_y5_repeated_three_channels"
POSE_TENSOR_PREPROCESSING_PROBE_SCHEMA_ID = "palette.pose_tensor_preprocessing_probe"
POSE_TENSOR_PREPROCESSING_PROBE_SCHEMA_VERSION = 1
EMPIRICAL_RUNTIME_PROFILE_STATUSES = ("accepted", "rejected")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class PoseModelInputContractError(ValueError):
    """Raised when model-input evidence is missing, stale, or ambiguous."""


def _fail(message: str) -> None:
    raise PoseModelInputContractError(message)


def _required_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{field} must be one nonempty canonical string.")
    return value


def _required_identifier(value: Any, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _fail(f"{field} must be lowercase snake_case.")
    return text


def _required_sha256(value: Any, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_RE.fullmatch(text) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return text


def _shape(value: Any, *, field: str) -> tuple[int, int]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        _fail(f"{field} must be an exact positive [height, width] list.")
    return int(value[0]), int(value[1])


def _exact_fields(value: Any, *, fields: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        _fail(f"{field} does not use the exact controlled field set.")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail(f"Unable to hash model-input evidence {path}: {exc}.")
    return digest.hexdigest()


def _relative_artifact(value: Any, *, field: str) -> tuple[Path, str]:
    record = _exact_fields(
        value,
        fields={"relative_path", "sha256"},
        field=field,
    )
    raw_path = _required_text(record["relative_path"], field=f"{field}.relative_path")
    path = Path(raw_path)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        _fail(f"{field}.relative_path must remain inside the model package.")
    digest = _required_sha256(record["sha256"], field=f"{field}.sha256")
    return path, digest


def _model_package_root(model_path: Path, relative_weights: Path) -> Path:
    candidate = model_path
    for _ in relative_weights.parts:
        candidate = candidate.parent
    expected = (candidate / relative_weights).resolve()
    if expected != model_path.resolve():
        _fail(
            "Model path does not occupy the weights location declared by its "
            "model-input contract."
        )
    return candidate.resolve()


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Unable to read {field} {path}: {exc}.")
    if type(value) is not dict:
        _fail(f"{field} must be one JSON mapping.")
    return value


def _read_yaml_evidence(path: Path, *, field: str) -> dict[str, Any]:
    """Load historical YAML without constructing Python-tagged objects."""

    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, yaml.YAMLError) as exc:
        _fail(f"Unable to read {field} {path}: {exc}.")
    if type(value) is not dict:
        _fail(f"{field} must be one YAML mapping.")
    return value


def _evidence_int(value: Any, *, field: str) -> int:
    if type(value) not in {str, int}:
        _fail(f"{field} must contain one exact integer.")
    try:
        result = int(value)
    except (TypeError, ValueError):
        _fail(f"{field} must contain one exact integer.")
    if result <= 0 or str(result) != str(value):
        _fail(f"{field} must contain one positive canonical integer.")
    return result


def _evidence_bool(value: Any, *, field: str) -> bool:
    if type(value) is bool:
        return value
    if type(value) is str and value in {"true", "false"}:
        return value == "true"
    _fail(f"{field} must contain one canonical boolean.")


def _report_source_shapes(report: Mapping[str, Any]) -> set[tuple[int, int]]:
    history = report.get("training_history")
    sources = (
        history.get("source_zarr_metadata") if isinstance(history, Mapping) else None
    )
    if not isinstance(sources, Mapping) or not sources:
        _fail("Training report lacks source_zarr_metadata crop evidence.")
    result: set[tuple[int, int]] = set()
    for source in sources.values():
        crop = source.get("crop_info") if isinstance(source, Mapping) else None
        raw_shape = crop.get("roi_size") if isinstance(crop, Mapping) else None
        if not isinstance(raw_shape, list) or len(raw_shape) != 2:
            _fail("Training report contains a source without exact crop roi_size.")
        result.add(
            (
                _evidence_int(raw_shape[0], field="training report roi height"),
                _evidence_int(raw_shape[1], field="training report roi width"),
            )
        )
    return result


@dataclass(frozen=True)
class PoseModelInputRuntimePlan:
    """One contract-derived native-ROI inference plan."""

    transform: ModelInputTransform
    network_shape_hw: tuple[int, int]
    model_stride: int
    input_mode: str
    profile_id: str
    classification: str
    contract_path: Path
    contract_sha256: str
    contract_payload_digest: str
    runtime_ultralytics_versions: tuple[str, ...] = ()
    preprocessing_probe: Mapping[str, Any] = field(default_factory=dict)

    @property
    def network_imgsz(self) -> int:
        if self.network_shape_hw[0] != self.network_shape_hw[1]:
            _fail("Current pose runtime requires one square network input extent.")
        return int(self.network_shape_hw[0])

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "classification": self.classification,
            "input_mode": self.input_mode,
            "network_shape_hw": list(self.network_shape_hw),
            "model_stride": self.model_stride,
            "submitted_transform": self.transform.to_attrs(),
            "contract_path": str(self.contract_path),
            "contract_sha256": self.contract_sha256,
            "contract_payload_digest": self.contract_payload_digest,
            "runtime_ultralytics_versions": list(self.runtime_ultralytics_versions),
            "preprocessing_probe": dict(self.preprocessing_probe),
        }


@dataclass(frozen=True)
class PoseModelInputContractBinding:
    """Verified contract and model-package evidence."""

    path: Path
    schema_version: int
    sha256: str
    payload_digest: str
    set_id: str
    run_id: str
    weights_sha256: str
    training_source_shape_hw: tuple[int, int]
    network_shape_hw: tuple[int, int]
    model_stride: int
    input_mode: str
    ultralytics_version: str
    runtime_ultralytics_versions: tuple[str, ...]
    preprocessing_probe: Mapping[str, Any]
    runtime_profile: Mapping[str, Any]
    runtime_profiles: tuple[Mapping[str, Any], ...]
    document: Mapping[str, Any]

    def plan_for_native_shape(
        self,
        native_shape_hw: tuple[int, int],
    ) -> PoseModelInputRuntimePlan:
        native_height, native_width = (int(native_shape_hw[0]), int(native_shape_hw[1]))
        if native_height <= 0 or native_width <= 0:
            _fail("Native ROI shape must be positive.")
        if self.schema_version == POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION:
            matching = tuple(
                profile
                for profile in self.runtime_profiles
                if tuple(profile["native_shape_hw"]) == (native_height, native_width)
            )
            accepted = tuple(
                profile for profile in matching if profile["status"] == "accepted"
            )
            if len(accepted) > 1:
                _fail(
                    "Pose model-input contract has multiple accepted profiles for "
                    f"native shape {(native_height, native_width)}."
                )
            if not accepted:
                if matching:
                    _fail(
                        "Every exact pose preprocessing profile for native shape "
                        f"{(native_height, native_width)} is explicitly rejected."
                    )
                _fail(
                    "Pose model-input contract has no exact accepted profile for "
                    f"native shape {(native_height, native_width)}."
                )
            profile = accepted[0]
            submitted_shape = tuple(int(item) for item in profile["submitted_shape_hw"])
            transform = resolve_model_input_transform(
                (native_height, native_width),
                mode=(
                    "identity"
                    if submitted_shape == (native_height, native_width)
                    else "pad_to_size"
                ),
                model_hw=submitted_shape,
            )
            submitted = profile["submitted_to_network"]
            return PoseModelInputRuntimePlan(
                transform=transform,
                network_shape_hw=tuple(
                    int(item) for item in profile["network_shape_hw"]
                ),
                model_stride=int(profile["model_stride"]),
                input_mode=str(profile["input_mode"]),
                profile_id=str(profile["profile_id"]),
                classification=str(profile["classification"]),
                contract_path=self.path,
                contract_sha256=self.sha256,
                contract_payload_digest=self.payload_digest,
                runtime_ultralytics_versions=tuple(
                    str(item) for item in submitted["runtime_ultralytics_versions"]
                ),
                preprocessing_probe=dict(submitted["preprocessing_probe"]),
            )

        training_height, training_width = self.training_source_shape_hw
        if native_height > training_height or native_width > training_width:
            _fail(
                "Scale-matched diagnostic padding cannot shrink a native ROI; "
                f"native={native_shape_hw}, training_source={self.training_source_shape_hw}."
            )
        transform = resolve_model_input_transform(
            (native_height, native_width),
            mode=(
                "identity"
                if (native_height, native_width) == self.training_source_shape_hw
                else "pad_to_size"
            ),
            model_hw=self.training_source_shape_hw,
        )
        classification = (
            "training_source_geometry_exact"
            if transform.is_identity
            else SCALE_MATCHED_RUNTIME_CLASSIFICATION
        )
        return PoseModelInputRuntimePlan(
            transform=transform,
            network_shape_hw=self.network_shape_hw,
            model_stride=self.model_stride,
            input_mode=self.input_mode,
            profile_id=str(self.runtime_profile["profile_id"]),
            classification=classification,
            contract_path=self.path,
            contract_sha256=self.sha256,
            contract_payload_digest=self.payload_digest,
            runtime_ultralytics_versions=self.runtime_ultralytics_versions,
            preprocessing_probe=self.preprocessing_probe,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "payload_digest": self.payload_digest,
            "set_id": self.set_id,
            "run_id": self.run_id,
            "weights_sha256": self.weights_sha256,
            "training_source_shape_hw": list(self.training_source_shape_hw),
            "network_shape_hw": list(self.network_shape_hw),
            "model_stride": self.model_stride,
            "input_mode": self.input_mode,
            "training_ultralytics_version": self.ultralytics_version,
            "runtime_ultralytics_versions": list(self.runtime_ultralytics_versions),
            "preprocessing_probe": dict(self.preprocessing_probe),
            "runtime_profile": dict(self.runtime_profile),
            "runtime_profiles": [dict(profile) for profile in self.runtime_profiles],
        }


def _runtime_versions(value: Any) -> tuple[str, ...]:
    if type(value) is not list or not value:
        _fail("Runtime Ultralytics versions must be one nonempty ordered list.")
    versions = tuple(
        _required_text(item, field="runtime Ultralytics version") for item in value
    )
    if versions != tuple(sorted(set(versions))):
        _fail("Runtime Ultralytics versions must be unique and sorted.")
    return versions


def _validate_preprocessing_probe(
    value: Any,
    *,
    source_shape_hw: tuple[int, int],
    network_shape_hw: tuple[int, int],
    model_stride: int,
) -> dict[str, Any]:
    probe = _exact_fields(
        value,
        fields={
            "schema_id",
            "schema_version",
            "input_pattern",
            "source_shape_hw",
            "network_shape_hw",
            "model_stride",
            "output_shape_nchw",
            "output_dtype",
            "output_sha256",
        },
        field="runtime preprocessing probe",
    )
    expected = {
        "schema_id": POSE_PREPROCESSING_PROBE_SCHEMA_ID,
        "schema_version": POSE_PREPROCESSING_PROBE_SCHEMA_VERSION,
        "input_pattern": POSE_PREPROCESSING_PROBE_PATTERN,
        "source_shape_hw": list(source_shape_hw),
        "network_shape_hw": list(network_shape_hw),
        "model_stride": model_stride,
        "output_shape_nchw": [1, 3, *network_shape_hw],
        "output_dtype": "float32_little_endian",
    }
    if {key: probe[key] for key in expected} != expected:
        _fail("Runtime preprocessing probe geometry or dtype is unsupported.")
    _required_sha256(probe["output_sha256"], field="preprocessing probe digest")
    return probe


def build_pose_preprocessing_equivalence_probe(
    *,
    source_shape_hw: tuple[int, int],
    network_shape_hw: tuple[int, int],
    model_stride: int,
) -> dict[str, Any]:
    """Execute the maintained Ultralytics numpy-list preprocessing probe."""

    import numpy as np
    import torch
    from types import SimpleNamespace
    from ultralytics.engine.predictor import BasePredictor

    source_height, source_width = source_shape_hw
    network_height, network_width = network_shape_hw
    y, x = np.indices((source_height, source_width))
    luma = ((x * 3 + y * 5) % 251).astype(np.uint8)
    image = np.repeat(luma[..., None], 3, axis=2)
    predictor = BasePredictor(overrides={"imgsz": max(network_shape_hw), "rect": False})
    predictor.imgsz = (network_height, network_width)
    predictor.model = SimpleNamespace(
        stride=model_stride,
        pt=True,
        dynamic=False,
        imx=False,
        device=torch.device("cpu"),
        fp16=False,
    )
    output = predictor.preprocess([image])
    canonical = np.asarray(output.detach().cpu().numpy(), dtype="<f4")
    return {
        "schema_id": POSE_PREPROCESSING_PROBE_SCHEMA_ID,
        "schema_version": POSE_PREPROCESSING_PROBE_SCHEMA_VERSION,
        "input_pattern": POSE_PREPROCESSING_PROBE_PATTERN,
        "source_shape_hw": [source_height, source_width],
        "network_shape_hw": [network_height, network_width],
        "model_stride": model_stride,
        "output_shape_nchw": list(canonical.shape),
        "output_dtype": "float32_little_endian",
        "output_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def build_pose_tensor_preprocessing_equivalence_probe(
    *,
    native_shape_hw: tuple[int, int],
    submitted_shape_hw: tuple[int, int],
    model_stride: int,
) -> dict[str, Any]:
    """Build a deterministic digest for Palette's prepared-tensor adapter.

    The probe covers the native uint8 pixels, exact centered zero padding,
    uint8-to-float32 normalization, and luma-to-three-channel expansion.  A
    prepared tensor is already at the model's submitted extent, so Ultralytics
    must not perform a second spatial resize.
    """

    import numpy as np

    transform = resolve_model_input_transform(
        native_shape_hw,
        mode=(
            "identity"
            if tuple(native_shape_hw) == tuple(submitted_shape_hw)
            else "pad_to_size"
        ),
        model_hw=submitted_shape_hw,
    )
    native_height, native_width = native_shape_hw
    y, x = np.indices((native_height, native_width))
    luma = ((x * 3 + y * 5) % 251).astype(np.uint8)
    submitted = transform.apply_numpy_luma_batch(luma[None, ...])
    normalized = np.asarray(submitted, dtype="<f4") / np.float32(255.0)
    canonical = np.repeat(normalized[:, None, :, :], 3, axis=1)
    return {
        "schema_id": POSE_TENSOR_PREPROCESSING_PROBE_SCHEMA_ID,
        "schema_version": POSE_TENSOR_PREPROCESSING_PROBE_SCHEMA_VERSION,
        "input_pattern": POSE_PREPROCESSING_PROBE_PATTERN,
        "native_shape_hw": list(native_shape_hw),
        "submitted_shape_hw": list(submitted_shape_hw),
        "model_stride": int(model_stride),
        "output_shape_nchw": list(canonical.shape),
        "output_dtype": "float32_little_endian",
        "output_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _validate_tensor_preprocessing_probe(
    value: Any,
    *,
    native_shape_hw: tuple[int, int],
    submitted_shape_hw: tuple[int, int],
    model_stride: int,
) -> dict[str, Any]:
    probe = _exact_fields(
        value,
        fields={
            "schema_id",
            "schema_version",
            "input_pattern",
            "native_shape_hw",
            "submitted_shape_hw",
            "model_stride",
            "output_shape_nchw",
            "output_dtype",
            "output_sha256",
        },
        field="tensor preprocessing probe",
    )
    expected = build_pose_tensor_preprocessing_equivalence_probe(
        native_shape_hw=native_shape_hw,
        submitted_shape_hw=submitted_shape_hw,
        model_stride=model_stride,
    )
    if probe != expected:
        _fail("Tensor preprocessing probe geometry, dtype, or digest is unsupported.")
    return probe


def validate_pose_runtime_compatibility(
    binding: PoseModelInputContractBinding,
    runtime_plan: PoseModelInputRuntimePlan | None = None,
) -> dict[str, Any]:
    """Fail unless this runtime reproduces an explicitly reviewed adapter."""

    if binding.schema_version == POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION:
        if runtime_plan is None:
            _fail("A v2 pose contract requires its exact selected runtime plan.")
        approved_versions = runtime_plan.runtime_ultralytics_versions
        expected_probe = dict(runtime_plan.preprocessing_probe)
    else:
        approved_versions = binding.runtime_ultralytics_versions
        expected_probe = dict(binding.preprocessing_probe)
    runtime_version = importlib.metadata.version("ultralytics")
    if runtime_version not in approved_versions:
        _fail(
            "Installed Ultralytics version is not an approved runtime: "
            f"runtime={runtime_version!r}, "
            f"approved={approved_versions!r}."
        )
    if runtime_plan is not None and runtime_plan.input_mode == "tensor":
        observed = build_pose_tensor_preprocessing_equivalence_probe(
            native_shape_hw=runtime_plan.transform.native_shape,
            submitted_shape_hw=runtime_plan.transform.model_shape,
            model_stride=runtime_plan.model_stride,
        )
    else:
        observed = build_pose_preprocessing_equivalence_probe(
            source_shape_hw=(
                runtime_plan.transform.model_shape
                if runtime_plan is not None
                else binding.training_source_shape_hw
            ),
            network_shape_hw=(
                runtime_plan.network_shape_hw
                if runtime_plan is not None
                else binding.network_shape_hw
            ),
            model_stride=(
                runtime_plan.model_stride
                if runtime_plan is not None
                else binding.model_stride
            ),
        )
    if observed != expected_probe:
        _fail(
            "Installed Ultralytics runtime failed the preprocessing equivalence probe."
        )
    return {
        "runtime_ultralytics_version": runtime_version,
        "approved_runtime_ultralytics_versions": list(approved_versions),
        "preprocessing_probe": observed,
    }


def _validate_runtime_profile(
    value: Any,
    *,
    source_shape_hw: tuple[int, int],
    network_shape_hw: tuple[int, int],
    model_stride: int,
) -> dict[str, Any]:
    profile = _exact_fields(
        value,
        fields={
            "profile_id",
            "classification",
            "input_mode",
            "native_to_submitted",
            "submitted_to_network",
            "result_coordinates",
        },
        field="payload.runtime_profile",
    )
    if (
        profile["profile_id"] != SCALE_MATCHED_RUNTIME_PROFILE_ID
        or profile["classification"] != SCALE_MATCHED_RUNTIME_CLASSIFICATION
        or profile["input_mode"] != "numpy-list"
    ):
        _fail("Runtime profile identity or input mode is unsupported.")
    first = _exact_fields(
        profile["native_to_submitted"],
        fields={"policy", "padding_mode", "padding_value_uint8"},
        field="runtime_profile.native_to_submitted",
    )
    if first != {
        "policy": "center_pad_to_training_source_shape",
        "padding_mode": "constant",
        "padding_value_uint8": 0,
    }:
        _fail("Runtime native-to-submitted padding policy is unsupported.")
    second = _exact_fields(
        profile["submitted_to_network"],
        fields={
            "adapter",
            "runtime_ultralytics_versions",
            "preprocessing_probe",
            "imgsz_hw",
            "rect",
            "letterbox_interpolation",
            "letterbox_padding_value_uint8",
            "luma_channel_policy",
            "normalization",
        },
        field="runtime_profile.submitted_to_network",
    )
    runtime_versions = _runtime_versions(second["runtime_ultralytics_versions"])
    preprocessing_probe = _validate_preprocessing_probe(
        second["preprocessing_probe"],
        source_shape_hw=source_shape_hw,
        network_shape_hw=network_shape_hw,
        model_stride=model_stride,
    )
    expected_second = {
        "adapter": ULTRALYTICS_NUMPY_LIST_ADAPTER,
        "runtime_ultralytics_versions": list(runtime_versions),
        "preprocessing_probe": preprocessing_probe,
        "imgsz_hw": list(network_shape_hw),
        "rect": False,
        "letterbox_interpolation": "opencv_inter_linear",
        "letterbox_padding_value_uint8": 114,
        "luma_channel_policy": "uint8_luma_repeated_bgr_then_equivalent_rgb",
        "normalization": "uint8_divide_255_to_float",
    }
    if second != expected_second:
        _fail("Runtime submitted-to-network preprocessing policy is unsupported.")
    if profile["result_coordinates"] != (
        "ultralytics_results_original_submitted_pixel_coordinates"
    ):
        _fail("Runtime result-coordinate policy is unsupported.")
    return profile


def _validate_empirical_profile_evidence(
    value: Any,
    *,
    status: str,
) -> dict[str, Any]:
    evidence = _exact_fields(
        value,
        fields={
            "evidence_id",
            "artifact_path",
            "receipt_sha256",
            "total_rows",
            "successful_rows",
            "decision",
        },
        field="runtime profile evidence",
    )
    _required_identifier(evidence["evidence_id"], field="evidence.evidence_id")
    _required_text(evidence["artifact_path"], field="evidence.artifact_path")
    _required_sha256(evidence["receipt_sha256"], field="evidence.receipt_sha256")
    total = evidence["total_rows"]
    successful = evidence["successful_rows"]
    if (
        type(total) is not int
        or total <= 0
        or type(successful) is not int
        or successful < 0
        or successful > total
    ):
        _fail("Runtime profile evidence row counts are invalid.")
    decision = _required_text(evidence["decision"], field="evidence.decision")
    expected_decision = {
        "accepted": "accepted_for_selector_ineligible_candidate_inference",
        "rejected": "rejected_for_candidate_inference",
    }[status]
    if decision != expected_decision:
        _fail("Runtime profile evidence decision disagrees with profile status.")
    return evidence


def _validate_empirical_runtime_profile(value: Any) -> dict[str, Any]:
    profile = _exact_fields(
        value,
        fields={
            "profile_id",
            "status",
            "classification",
            "native_shape_hw",
            "submitted_shape_hw",
            "network_shape_hw",
            "model_stride",
            "input_mode",
            "native_to_submitted",
            "submitted_to_network",
            "result_coordinates",
            "evidence",
        },
        field="payload.runtime_profiles[]",
    )
    _required_identifier(profile["profile_id"], field="runtime profile id")
    status = profile["status"]
    if status not in EMPIRICAL_RUNTIME_PROFILE_STATUSES:
        _fail("Runtime profile status must be accepted or rejected.")
    _required_identifier(
        profile["classification"], field="runtime profile classification"
    )
    native_shape = _shape(profile["native_shape_hw"], field="native_shape_hw")
    submitted_shape = _shape(profile["submitted_shape_hw"], field="submitted_shape_hw")
    network_shape = _shape(profile["network_shape_hw"], field="network_shape_hw")
    stride = profile["model_stride"]
    if type(stride) is not int or stride <= 0:
        _fail("Runtime profile model_stride must be a positive exact integer.")
    if submitted_shape[0] != submitted_shape[1] or network_shape[0] != network_shape[1]:
        _fail("Current pose inference requires square submitted and network extents.")
    if any(extent % stride for extent in network_shape):
        _fail("Runtime profile network extent must align to the model stride.")
    first = _exact_fields(
        profile["native_to_submitted"],
        fields={"policy", "padding_mode", "padding_value_uint8"},
        field="runtime profile native_to_submitted",
    )
    if submitted_shape == native_shape:
        expected_policy = "identity"
    elif all(
        submitted >= native
        for submitted, native in zip(submitted_shape, native_shape, strict=True)
    ):
        expected_policy = "center_pad_to_submitted_shape"
    else:
        _fail("Empirical runtime profiles may not resize or crop native pixels.")
    if first != {
        "policy": expected_policy,
        "padding_mode": "constant",
        "padding_value_uint8": 0,
    }:
        _fail("Runtime profile native-to-submitted transform is unsupported.")

    mode = profile["input_mode"]
    submitted = profile["submitted_to_network"]
    if mode == "tensor":
        second = _exact_fields(
            submitted,
            fields={
                "adapter",
                "runtime_ultralytics_versions",
                "preprocessing_probe",
                "imgsz_hw",
                "ultralytics_spatial_preprocessing",
                "luma_channel_policy",
                "normalization",
            },
            field="tensor submitted_to_network",
        )
        versions = _runtime_versions(second["runtime_ultralytics_versions"])
        probe = _validate_tensor_preprocessing_probe(
            second["preprocessing_probe"],
            native_shape_hw=native_shape,
            submitted_shape_hw=submitted_shape,
            model_stride=int(stride),
        )
        expected_second = {
            "adapter": PALETTE_PREPARED_TENSOR_ADAPTER,
            "runtime_ultralytics_versions": list(versions),
            "preprocessing_probe": probe,
            "imgsz_hw": list(network_shape),
            "ultralytics_spatial_preprocessing": "bypassed_prepared_tensor",
            "luma_channel_policy": "uint8_luma_repeated_three_channels",
            "normalization": "uint8_divide_255_to_float32",
        }
        if submitted_shape != network_shape or second != expected_second:
            _fail(
                "Prepared-tensor profiles require submitted and network shapes "
                "to match with no second spatial resize."
            )
    elif mode == "numpy-list":
        second = _exact_fields(
            submitted,
            fields={
                "adapter",
                "runtime_ultralytics_versions",
                "preprocessing_probe",
                "imgsz_hw",
                "rect",
                "letterbox_interpolation",
                "letterbox_padding_value_uint8",
                "luma_channel_policy",
                "normalization",
            },
            field="numpy-list submitted_to_network",
        )
        versions = _runtime_versions(second["runtime_ultralytics_versions"])
        probe = _validate_preprocessing_probe(
            second["preprocessing_probe"],
            source_shape_hw=submitted_shape,
            network_shape_hw=network_shape,
            model_stride=int(stride),
        )
        expected_second = {
            "adapter": ULTRALYTICS_NUMPY_LIST_ADAPTER,
            "runtime_ultralytics_versions": list(versions),
            "preprocessing_probe": probe,
            "imgsz_hw": list(network_shape),
            "rect": False,
            "letterbox_interpolation": "opencv_inter_linear",
            "letterbox_padding_value_uint8": 114,
            "luma_channel_policy": ("uint8_luma_repeated_bgr_then_equivalent_rgb"),
            "normalization": "uint8_divide_255_to_float",
        }
        if second != expected_second:
            _fail("Numpy-list empirical preprocessing policy is unsupported.")
    else:
        _fail("Runtime profile input_mode must be tensor or numpy-list.")
    if profile["result_coordinates"] != (
        "ultralytics_results_original_submitted_pixel_coordinates"
    ):
        _fail("Runtime result-coordinate policy is unsupported.")
    _validate_empirical_profile_evidence(profile["evidence"], status=status)
    return profile


def _validate_empirical_runtime_profiles(value: Any) -> tuple[dict[str, Any], ...]:
    if type(value) is not list or not value:
        _fail("payload.runtime_profiles must be one nonempty ordered list.")
    profiles = tuple(_validate_empirical_runtime_profile(item) for item in value)
    profile_ids = tuple(str(profile["profile_id"]) for profile in profiles)
    if profile_ids != tuple(sorted(set(profile_ids))):
        _fail("Runtime profiles must have unique IDs in canonical sorted order.")
    accepted_shapes = [
        tuple(profile["native_shape_hw"])
        for profile in profiles
        if profile["status"] == "accepted"
    ]
    if len(accepted_shapes) != len(set(accepted_shapes)):
        _fail("Only one accepted runtime profile is permitted per native shape.")
    if not accepted_shapes:
        _fail("A v2 pose contract requires at least one accepted runtime profile.")
    return profiles


def load_pose_model_input_contract(
    contract_path: Path,
    *,
    model_path: Path,
    expected_set_id: str,
    expected_run_id: str,
    expected_model_sha256: str,
) -> PoseModelInputContractBinding:
    """Load and revalidate a complete pose-model input contract."""

    path = contract_path.expanduser().resolve()
    document = _read_json(path, field="pose model-input contract")
    top = _exact_fields(
        document,
        fields={
            "schema_id",
            "schema_version",
            "digest_algorithm",
            "payload_digest",
            "payload",
        },
        field="pose model-input contract",
    )
    schema_version = top["schema_version"]
    if (
        top["schema_id"] != POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID
        or schema_version
        not in {
            POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION_V1,
            POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION,
        }
        or top["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        _fail("Pose model-input contract schema identity is unsupported.")
    payload_fields = (
        {"status", "model", "evidence", "training_input", "runtime_profile"}
        if schema_version == POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION_V1
        else {"status", "model", "evidence", "training_input", "runtime_profiles"}
    )
    payload = _exact_fields(
        top["payload"],
        fields=payload_fields,
        field="pose model-input contract payload",
    )
    payload_digest = _required_sha256(top["payload_digest"], field="payload_digest")
    if payload_digest != canonical_json_sha256(payload):
        _fail("Pose model-input contract payload digest is stale.")
    if payload["status"] != "complete":
        _fail("Pose model-input contract is not complete.")

    model = _exact_fields(
        payload["model"],
        fields={"set_id", "run_id", "weights"},
        field="payload.model",
    )
    set_id = _required_text(model["set_id"], field="model.set_id")
    run_id = _required_text(model["run_id"], field="model.run_id")
    if set_id != expected_set_id or run_id != expected_run_id:
        _fail("Pose model-input contract selects a different model set or run.")
    weights_relative, weights_sha256 = _relative_artifact(
        model["weights"], field="model.weights"
    )
    expected_digest = _required_sha256(
        expected_model_sha256, field="expected_model_sha256"
    )
    if weights_sha256 != expected_digest:
        _fail("Pose model-input contract weights digest differs from the registry.")
    resolved_model = model_path.expanduser().resolve()
    package_root = _model_package_root(resolved_model, weights_relative)

    evidence = _exact_fields(
        payload["evidence"],
        fields={"training_manifest", "training_report", "training_args"},
        field="payload.evidence",
    )
    artifacts: dict[str, Path] = {"weights": resolved_model}
    expected_hashes: dict[str, str] = {"weights": weights_sha256}
    for role in ("training_manifest", "training_report", "training_args"):
        relative, digest = _relative_artifact(evidence[role], field=f"evidence.{role}")
        artifacts[role] = (package_root / relative).resolve()
        expected_hashes[role] = digest
        if package_root not in artifacts[role].parents:
            _fail(f"Evidence path for {role} escapes the model package.")
    for role, artifact_path in artifacts.items():
        if not artifact_path.is_file():
            _fail(f"Required model-input evidence is missing: {artifact_path}.")
        if _sha256_file(artifact_path) != expected_hashes[role]:
            _fail(f"Model-input evidence digest changed for {role}.")

    training = _exact_fields(
        payload["training_input"],
        fields={
            "source_roi_shape_hw",
            "network_shape_hw",
            "input_format",
            "roi_pixel_contract_name",
            "ultralytics_version",
            "model_stride",
            "training_rect",
            "training_multi_scale",
            "training_semantics",
        },
        field="payload.training_input",
    )
    source_shape = _shape(
        training["source_roi_shape_hw"], field="training_input.source_roi_shape_hw"
    )
    network_shape = _shape(
        training["network_shape_hw"], field="training_input.network_shape_hw"
    )
    input_format = _required_identifier(
        training["input_format"], field="training_input.input_format"
    )
    pixel_contract = _required_text(
        training["roi_pixel_contract_name"],
        field="training_input.roi_pixel_contract_name",
    )
    ultralytics_version = _required_text(
        training["ultralytics_version"],
        field="training_input.ultralytics_version",
    )
    stride = training["model_stride"]
    if type(stride) is not int or stride <= 0:
        _fail("training_input.model_stride must be a positive exact integer.")
    training_rect = training["training_rect"]
    training_multi_scale = training["training_multi_scale"]
    if type(training_rect) is not bool or type(training_multi_scale) is not bool:
        _fail("Training rect and multi-scale declarations must be exact booleans.")
    if training_rect or training_multi_scale:
        _fail(
            "Historical pose deployment currently requires rect=false and "
            "multi_scale=false."
        )
    if training["training_semantics"] != (
        "augmented_training_pipeline_with_deterministic_validation_imgsz_reference"
    ):
        _fail("Training preprocessing semantics are unsupported.")

    manifest = _read_json(artifacts["training_manifest"], field="training manifest")
    if manifest.get("task") != "pose":
        _fail("Training manifest task differs from the pose model contract.")
    if manifest.get("set_id") != set_id:
        _fail("Training manifest set_id differs from the model contract.")
    if tuple(manifest.get("imgsz") or ()) != network_shape:
        _fail("Training manifest imgsz differs from the model contract.")
    if manifest.get("input_format") != input_format:
        _fail("Training manifest input_format differs from the model contract.")
    if manifest.get("roi_pixel_contract_name") != pixel_contract:
        _fail("Training manifest pixel contract differs from the model contract.")

    args = _read_yaml_evidence(artifacts["training_args"], field="training args")
    if (
        args.get("task") != "pose"
        or _evidence_int(args.get("imgsz"), field="training args imgsz")
        != network_shape[0]
    ):
        _fail("Training args task/imgsz differs from the model contract.")
    if (
        _evidence_bool(args.get("rect"), field="training args rect") != training_rect
        or _evidence_bool(args.get("multi_scale"), field="training args multi_scale")
        != training_multi_scale
    ):
        _fail("Training args geometry augmentation differs from the model contract.")
    if network_shape[0] != network_shape[1]:
        _fail("Historical scalar training imgsz requires a square network extent.")

    report = _read_yaml_evidence(artifacts["training_report"], field="training report")
    if _report_source_shapes(report) != {source_shape}:
        _fail("Training report source ROI shape differs from the model contract.")
    report_params = report.get("training_params")
    if (
        not isinstance(report_params, Mapping)
        or _evidence_int(report_params.get("imgsz"), field="training report imgsz")
        != network_shape[0]
    ):
        _fail("Training report imgsz differs from the model contract.")
    if (
        _evidence_bool(report_params.get("rect"), field="training report rect")
        != training_rect
    ):
        _fail("Training report rect mode differs from the model contract.")
    history = report.get("training_history")
    if not isinstance(history, Mapping) or history.get("ultralytics_version") != (
        ultralytics_version
    ):
        _fail("Training report Ultralytics version differs from the model contract.")

    if schema_version == POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION_V1:
        runtime_profile = _validate_runtime_profile(
            payload["runtime_profile"],
            source_shape_hw=source_shape,
            network_shape_hw=network_shape,
            model_stride=int(stride),
        )
        runtime_profiles: tuple[Mapping[str, Any], ...] = (runtime_profile,)
    else:
        runtime_profiles = _validate_empirical_runtime_profiles(
            payload["runtime_profiles"]
        )
        if any(
            int(profile["model_stride"]) != int(stride) for profile in runtime_profiles
        ):
            _fail("Runtime profile stride differs from the model training contract.")
        runtime_profile = next(
            profile for profile in runtime_profiles if profile["status"] == "accepted"
        )
    submitted = runtime_profile["submitted_to_network"]
    runtime_versions = _runtime_versions(submitted["runtime_ultralytics_versions"])
    if schema_version == POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION_V1:
        preprocessing_probe = _validate_preprocessing_probe(
            submitted["preprocessing_probe"],
            source_shape_hw=source_shape,
            network_shape_hw=network_shape,
            model_stride=int(stride),
        )
        default_network_shape = network_shape
        default_stride = int(stride)
    else:
        preprocessing_probe = dict(submitted["preprocessing_probe"])
        default_network_shape = tuple(runtime_profile["network_shape_hw"])
        default_stride = int(runtime_profile["model_stride"])
    return PoseModelInputContractBinding(
        path=path,
        schema_version=int(schema_version),
        sha256=_sha256_file(path),
        payload_digest=payload_digest,
        set_id=set_id,
        run_id=run_id,
        weights_sha256=weights_sha256,
        training_source_shape_hw=source_shape,
        network_shape_hw=default_network_shape,
        model_stride=default_stride,
        input_mode=str(runtime_profile["input_mode"]),
        ultralytics_version=ultralytics_version,
        runtime_ultralytics_versions=runtime_versions,
        preprocessing_probe=preprocessing_probe,
        runtime_profile=runtime_profile,
        runtime_profiles=runtime_profiles,
        document=document,
    )


def build_historical_pose_model_input_contract(
    *,
    set_id: str,
    run_id: str,
    model_package_root: Path,
    weights_relative_path: Path,
    training_manifest_relative_path: Path,
    training_report_relative_path: Path,
    training_args_relative_path: Path,
    model_stride: int,
    runtime_ultralytics_versions: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a contract from audited immutable historical model artifacts."""

    root = model_package_root.expanduser().resolve()
    relative_paths = {
        "weights": weights_relative_path,
        "training_manifest": training_manifest_relative_path,
        "training_report": training_report_relative_path,
        "training_args": training_args_relative_path,
    }
    resolved: dict[str, Path] = {}
    for role, raw_path in relative_paths.items():
        relative, _ = _relative_artifact(
            {"relative_path": str(raw_path), "sha256": "0" * 64},
            field=role,
        )
        resolved[role] = (root / relative).resolve()
        if root not in resolved[role].parents or not resolved[role].is_file():
            _fail(
                f"Historical model artifact is missing or outside package: {resolved[role]}."
            )

    manifest = _read_json(resolved["training_manifest"], field="training manifest")
    report = _read_yaml_evidence(resolved["training_report"], field="training report")
    args = _read_yaml_evidence(resolved["training_args"], field="training args")
    source_shapes = _report_source_shapes(report)
    if len(source_shapes) != 1:
        _fail("Historical training report does not have one source ROI shape.")
    source_shape = next(iter(source_shapes))
    manifest_imgsz = _shape(manifest.get("imgsz"), field="training manifest imgsz")
    args_imgsz = _evidence_int(args.get("imgsz"), field="training args imgsz")
    if manifest_imgsz != (args_imgsz, args_imgsz):
        _fail("Training manifest and args disagree on imgsz.")
    training_rect = _evidence_bool(args.get("rect"), field="training args rect")
    training_multi_scale = _evidence_bool(
        args.get("multi_scale"), field="training args multi_scale"
    )
    if training_rect or training_multi_scale:
        _fail("Historical pose backfill requires rect=false and multi_scale=false.")
    report_params = report.get("training_params")
    if (
        not isinstance(report_params, Mapping)
        or _evidence_bool(report_params.get("rect"), field="training report rect")
        != training_rect
    ):
        _fail("Training report and args disagree on rect mode.")
    history = report.get("training_history")
    version = (
        history.get("ultralytics_version") if isinstance(history, Mapping) else None
    )
    version = _required_text(version, field="training report ultralytics_version")
    builder_version = importlib.metadata.version("ultralytics")
    if builder_version != version:
        _fail(
            "Historical preprocessing reference must be built under the training "
            f"Ultralytics version: builder={builder_version!r}, training={version!r}."
        )
    approved_runtime_versions = tuple(
        sorted(
            {
                version,
                *(
                    _required_text(item, field="runtime Ultralytics version")
                    for item in runtime_ultralytics_versions
                ),
            }
        )
    )
    preprocessing_probe = build_pose_preprocessing_equivalence_probe(
        source_shape_hw=source_shape,
        network_shape_hw=manifest_imgsz,
        model_stride=model_stride,
    )
    if type(model_stride) is not int or model_stride <= 0:
        _fail("model_stride must be a positive exact integer assertion.")

    def artifact(role: str) -> dict[str, str]:
        return {
            "relative_path": str(relative_paths[role]),
            "sha256": _sha256_file(resolved[role]),
        }

    payload = {
        "status": "complete",
        "model": {
            "set_id": _required_text(set_id, field="set_id"),
            "run_id": _required_text(run_id, field="run_id"),
            "weights": artifact("weights"),
        },
        "evidence": {
            "training_manifest": artifact("training_manifest"),
            "training_report": artifact("training_report"),
            "training_args": artifact("training_args"),
        },
        "training_input": {
            "source_roi_shape_hw": list(source_shape),
            "network_shape_hw": list(manifest_imgsz),
            "input_format": manifest.get("input_format"),
            "roi_pixel_contract_name": manifest.get("roi_pixel_contract_name"),
            "ultralytics_version": version,
            "model_stride": int(model_stride),
            "training_rect": training_rect,
            "training_multi_scale": training_multi_scale,
            "training_semantics": (
                "augmented_training_pipeline_with_deterministic_validation_imgsz_reference"
            ),
        },
        "runtime_profile": {
            "profile_id": SCALE_MATCHED_RUNTIME_PROFILE_ID,
            "classification": SCALE_MATCHED_RUNTIME_CLASSIFICATION,
            "input_mode": "numpy-list",
            "native_to_submitted": {
                "policy": "center_pad_to_training_source_shape",
                "padding_mode": "constant",
                "padding_value_uint8": 0,
            },
            "submitted_to_network": {
                "adapter": ULTRALYTICS_NUMPY_LIST_ADAPTER,
                "runtime_ultralytics_versions": list(approved_runtime_versions),
                "preprocessing_probe": preprocessing_probe,
                "imgsz_hw": list(manifest_imgsz),
                "rect": False,
                "letterbox_interpolation": "opencv_inter_linear",
                "letterbox_padding_value_uint8": 114,
                "luma_channel_policy": ("uint8_luma_repeated_bgr_then_equivalent_rgb"),
                "normalization": "uint8_divide_255_to_float",
            },
            "result_coordinates": (
                "ultralytics_results_original_submitted_pixel_coordinates"
            ),
        },
    }
    return {
        "schema_id": POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID,
        "schema_version": POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION_V1,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def build_empirical_pose_runtime_profile(
    *,
    profile_id: str,
    status: str,
    classification: str,
    native_shape_hw: tuple[int, int],
    submitted_shape_hw: tuple[int, int],
    network_shape_hw: tuple[int, int],
    model_stride: int,
    input_mode: str,
    runtime_ultralytics_versions: tuple[str, ...],
    evidence_id: str,
    evidence_artifact_path: str,
    evidence_receipt_sha256: str,
    evidence_total_rows: int,
    evidence_successful_rows: int,
) -> dict[str, Any]:
    """Build one exact evidence-backed v2 runtime profile.

    This helper intentionally does not infer a transform from training ROI
    dimensions.  Callers must declare the native, submitted, and network
    extents established by an empirical candidate comparison.
    """

    native_shape = tuple(int(item) for item in native_shape_hw)
    submitted_shape = tuple(int(item) for item in submitted_shape_hw)
    network_shape = tuple(int(item) for item in network_shape_hw)
    versions = tuple(sorted(set(runtime_ultralytics_versions)))
    native_to_submitted = {
        "policy": (
            "identity"
            if native_shape == submitted_shape
            else "center_pad_to_submitted_shape"
        ),
        "padding_mode": "constant",
        "padding_value_uint8": 0,
    }
    if input_mode == "tensor":
        submitted_to_network = {
            "adapter": PALETTE_PREPARED_TENSOR_ADAPTER,
            "runtime_ultralytics_versions": list(versions),
            "preprocessing_probe": build_pose_tensor_preprocessing_equivalence_probe(
                native_shape_hw=native_shape,
                submitted_shape_hw=submitted_shape,
                model_stride=int(model_stride),
            ),
            "imgsz_hw": list(network_shape),
            "ultralytics_spatial_preprocessing": "bypassed_prepared_tensor",
            "luma_channel_policy": "uint8_luma_repeated_three_channels",
            "normalization": "uint8_divide_255_to_float32",
        }
    elif input_mode == "numpy-list":
        submitted_to_network = {
            "adapter": ULTRALYTICS_NUMPY_LIST_ADAPTER,
            "runtime_ultralytics_versions": list(versions),
            "preprocessing_probe": build_pose_preprocessing_equivalence_probe(
                source_shape_hw=submitted_shape,
                network_shape_hw=network_shape,
                model_stride=int(model_stride),
            ),
            "imgsz_hw": list(network_shape),
            "rect": False,
            "letterbox_interpolation": "opencv_inter_linear",
            "letterbox_padding_value_uint8": 114,
            "luma_channel_policy": ("uint8_luma_repeated_bgr_then_equivalent_rgb"),
            "normalization": "uint8_divide_255_to_float",
        }
    else:
        _fail("Empirical pose runtime profile input_mode is unsupported.")
    profile = {
        "profile_id": profile_id,
        "status": status,
        "classification": classification,
        "native_shape_hw": list(native_shape),
        "submitted_shape_hw": list(submitted_shape),
        "network_shape_hw": list(network_shape),
        "model_stride": int(model_stride),
        "input_mode": input_mode,
        "native_to_submitted": native_to_submitted,
        "submitted_to_network": submitted_to_network,
        "result_coordinates": (
            "ultralytics_results_original_submitted_pixel_coordinates"
        ),
        "evidence": {
            "evidence_id": evidence_id,
            "artifact_path": evidence_artifact_path,
            "receipt_sha256": evidence_receipt_sha256,
            "total_rows": evidence_total_rows,
            "successful_rows": evidence_successful_rows,
            "decision": {
                "accepted": "accepted_for_selector_ineligible_candidate_inference",
                "rejected": "rejected_for_candidate_inference",
            }.get(status),
        },
    }
    return _validate_empirical_runtime_profile(profile)


def build_pose_model_input_contract_v2(
    *,
    historical_contract: Mapping[str, Any],
    runtime_profiles: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Upgrade immutable v1 training evidence with reviewed runtime profiles."""

    # Round-trip through canonical JSON to detach the new document from caller
    # mutations and to reject non-JSON values.
    try:
        source = json.loads(json.dumps(historical_contract))
    except (TypeError, ValueError) as exc:
        _fail(f"Historical pose contract is not strict JSON: {exc}.")
    top = _exact_fields(
        source,
        fields={
            "schema_id",
            "schema_version",
            "digest_algorithm",
            "payload_digest",
            "payload",
        },
        field="historical pose model-input contract",
    )
    if (
        top["schema_id"] != POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID
        or top["schema_version"] != POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION_V1
        or top["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        _fail("Only a complete v1 pose model-input contract may be upgraded.")
    old_payload = _exact_fields(
        top["payload"],
        fields={"status", "model", "evidence", "training_input", "runtime_profile"},
        field="historical pose model-input payload",
    )
    if old_payload["status"] != "complete" or top[
        "payload_digest"
    ] != canonical_json_sha256(old_payload):
        _fail("Historical pose contract is incomplete or has a stale digest.")
    profiles = _validate_empirical_runtime_profiles(list(runtime_profiles))
    payload = {
        "status": "complete",
        "model": old_payload["model"],
        "evidence": old_payload["evidence"],
        "training_input": old_payload["training_input"],
        "runtime_profiles": [dict(profile) for profile in profiles],
    }
    return {
        "schema_id": POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID,
        "schema_version": POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


__all__ = [
    "POSE_MODEL_INPUT_CONTRACT_FILENAME",
    "POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID",
    "POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION",
    "POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION_V1",
    "POSE_PREPROCESSING_PROBE_SCHEMA_ID",
    "POSE_PREPROCESSING_PROBE_SCHEMA_VERSION",
    "PoseModelInputContractBinding",
    "PoseModelInputContractError",
    "PoseModelInputRuntimePlan",
    "SCALE_MATCHED_RUNTIME_CLASSIFICATION",
    "SCALE_MATCHED_RUNTIME_PROFILE_ID",
    "build_historical_pose_model_input_contract",
    "build_empirical_pose_runtime_profile",
    "build_pose_model_input_contract_v2",
    "build_pose_preprocessing_equivalence_probe",
    "build_pose_tensor_preprocessing_equivalence_probe",
    "load_pose_model_input_contract",
    "validate_pose_runtime_compatibility",
]
