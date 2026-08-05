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

from dataclasses import dataclass
import hashlib
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
POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION = 1
POSE_MODEL_INPUT_CONTRACT_FILENAME = "pose_model_input_contract.json"
SCALE_MATCHED_RUNTIME_PROFILE_ID = "scale_matched_center_pad_ultralytics_v1"
SCALE_MATCHED_RUNTIME_CLASSIFICATION = (
    "scale_matched_diagnostic_not_training_context"
)
ULTRALYTICS_NUMPY_LIST_ADAPTER = "ultralytics_numpy_list_predict_v1"

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
    sources = history.get("source_zarr_metadata") if isinstance(history, Mapping) else None
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
        }


@dataclass(frozen=True)
class PoseModelInputContractBinding:
    """Verified contract and model-package evidence."""

    path: Path
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
    runtime_profile: Mapping[str, Any]
    document: Mapping[str, Any]

    def plan_for_native_shape(
        self,
        native_shape_hw: tuple[int, int],
    ) -> PoseModelInputRuntimePlan:
        native_height, native_width = (int(native_shape_hw[0]), int(native_shape_hw[1]))
        training_height, training_width = self.training_source_shape_hw
        if native_height <= 0 or native_width <= 0:
            _fail("Native ROI shape must be positive.")
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
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "payload_digest": self.payload_digest,
            "set_id": self.set_id,
            "run_id": self.run_id,
            "weights_sha256": self.weights_sha256,
            "training_source_shape_hw": list(self.training_source_shape_hw),
            "network_shape_hw": list(self.network_shape_hw),
            "model_stride": self.model_stride,
            "input_mode": self.input_mode,
            "ultralytics_version": self.ultralytics_version,
            "runtime_profile": dict(self.runtime_profile),
        }


def _validate_runtime_profile(
    value: Any,
    *,
    network_shape_hw: tuple[int, int],
    ultralytics_version: str,
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
            "ultralytics_version",
            "imgsz_hw",
            "rect",
            "letterbox_interpolation",
            "letterbox_padding_value_uint8",
            "luma_channel_policy",
            "normalization",
        },
        field="runtime_profile.submitted_to_network",
    )
    if second != {
        "adapter": ULTRALYTICS_NUMPY_LIST_ADAPTER,
        "ultralytics_version": ultralytics_version,
        "imgsz_hw": list(network_shape_hw),
        "rect": False,
        "letterbox_interpolation": "opencv_inter_linear",
        "letterbox_padding_value_uint8": 114,
        "luma_channel_policy": "uint8_luma_repeated_bgr_then_equivalent_rgb",
        "normalization": "uint8_divide_255_to_float",
    }:
        _fail("Runtime submitted-to-network preprocessing policy is unsupported.")
    if profile["result_coordinates"] != (
        "ultralytics_results_original_submitted_pixel_coordinates"
    ):
        _fail("Runtime result-coordinate policy is unsupported.")
    return profile


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
    if (
        top["schema_id"] != POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID
        or top["schema_version"] != POSE_MODEL_INPUT_CONTRACT_SCHEMA_VERSION
        or top["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        _fail("Pose model-input contract schema identity is unsupported.")
    payload = _exact_fields(
        top["payload"],
        fields={"status", "model", "evidence", "training_input", "runtime_profile"},
        field="pose model-input contract payload",
    )
    payload_digest = _required_sha256(
        top["payload_digest"], field="payload_digest"
    )
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
    if args.get("task") != "pose" or _evidence_int(
        args.get("imgsz"), field="training args imgsz"
    ) != network_shape[0]:
        _fail("Training args task/imgsz differs from the model contract.")
    if (
        _evidence_bool(args.get("rect"), field="training args rect")
        != training_rect
        or _evidence_bool(
            args.get("multi_scale"), field="training args multi_scale"
        )
        != training_multi_scale
    ):
        _fail("Training args geometry augmentation differs from the model contract.")
    if network_shape[0] != network_shape[1]:
        _fail("Historical scalar training imgsz requires a square network extent.")

    report = _read_yaml_evidence(
        artifacts["training_report"], field="training report"
    )
    if _report_source_shapes(report) != {source_shape}:
        _fail("Training report source ROI shape differs from the model contract.")
    report_params = report.get("training_params")
    if not isinstance(report_params, Mapping) or _evidence_int(
        report_params.get("imgsz"), field="training report imgsz"
    ) != network_shape[0]:
        _fail("Training report imgsz differs from the model contract.")
    if _evidence_bool(
        report_params.get("rect"), field="training report rect"
    ) != training_rect:
        _fail("Training report rect mode differs from the model contract.")
    history = report.get("training_history")
    if not isinstance(history, Mapping) or history.get("ultralytics_version") != (
        ultralytics_version
    ):
        _fail("Training report Ultralytics version differs from the model contract.")

    runtime_profile = _validate_runtime_profile(
        payload["runtime_profile"],
        network_shape_hw=network_shape,
        ultralytics_version=ultralytics_version,
    )
    return PoseModelInputContractBinding(
        path=path,
        sha256=_sha256_file(path),
        payload_digest=payload_digest,
        set_id=set_id,
        run_id=run_id,
        weights_sha256=weights_sha256,
        training_source_shape_hw=source_shape,
        network_shape_hw=network_shape,
        model_stride=int(stride),
        input_mode=str(runtime_profile["input_mode"]),
        ultralytics_version=ultralytics_version,
        runtime_profile=runtime_profile,
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
            _fail(f"Historical model artifact is missing or outside package: {resolved[role]}.")

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
    if not isinstance(report_params, Mapping) or _evidence_bool(
        report_params.get("rect"), field="training report rect"
    ) != training_rect:
        _fail("Training report and args disagree on rect mode.")
    history = report.get("training_history")
    version = history.get("ultralytics_version") if isinstance(history, Mapping) else None
    version = _required_text(version, field="training report ultralytics_version")
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
                "ultralytics_version": version,
                "imgsz_hw": list(manifest_imgsz),
                "rect": False,
                "letterbox_interpolation": "opencv_inter_linear",
                "letterbox_padding_value_uint8": 114,
                "luma_channel_policy": (
                    "uint8_luma_repeated_bgr_then_equivalent_rgb"
                ),
                "normalization": "uint8_divide_255_to_float",
            },
            "result_coordinates": (
                "ultralytics_results_original_submitted_pixel_coordinates"
            ),
        },
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
    "PoseModelInputContractBinding",
    "PoseModelInputContractError",
    "PoseModelInputRuntimePlan",
    "SCALE_MATCHED_RUNTIME_CLASSIFICATION",
    "SCALE_MATCHED_RUNTIME_PROFILE_ID",
    "build_historical_pose_model_input_contract",
    "load_pose_model_input_contract",
]
