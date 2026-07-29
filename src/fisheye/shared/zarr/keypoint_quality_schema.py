"""Exact logical contract for immutable keypoint-quality snapshots.

The quality snapshot is a complete row-for-row diagnostic view of one raw
keypoint snapshot.  It owns derived measurements and policy proposals, but it
does not own landmark coordinates, accepted review state, longitudinal
identity, or body-frame/heading geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    FRAME_ROW_OFFSETS_V1,
    KEYPOINT_FRAME_INDICES_V1,
    KEYPOINT_INSTANCE_KEY_V1,
    KEYPOINT_QUALITY_ARRAY_CONTRACTS,
    KEYPOINT_QUALITY_KEYPOINT_FLAGS_V1,
    KEYPOINT_QUALITY_KEYPOINT_METRIC_VALID_V1,
    KEYPOINT_QUALITY_KEYPOINT_METRIC_VALUES_V1,
    KEYPOINT_QUALITY_POSE_FLAGS_V1,
    KEYPOINT_QUALITY_POSE_METRIC_VALID_V1,
    KEYPOINT_QUALITY_POSE_METRIC_VALUES_V1,
    KEYPOINT_QUALITY_PROPOSED_KEYPOINT_VALID_V1,
    KEYPOINT_QUALITY_PROPOSED_POSE_USABLE_V1,
    KEYPOINT_QUALITY_SOURCE_KEYPOINT_ROW_IDS_V1,
    KEYPOINT_QUALITY_SOURCE_KEYPOINT_ROW_SIGNATURE_V1,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_ID,
    KEYPOINT_SCHEMA_VERSION,
    derive_frame_row_offsets,
)


KEYPOINT_QUALITY_SCHEMA_ID = "palette.stage.keypoint_quality"
KEYPOINT_QUALITY_SCHEMA_VERSION = 1
KEYPOINT_QUALITY_LAYOUT = "immutable_source_bound_diagnostics_v1"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_METRIC_TOKENS = frozenset(
    {"heading", "temporal", "track", "trajectory"}
)

FORBIDDEN_KEYPOINT_QUALITY_ARRAYS = (
    "forward_axis_xy",
    "heading",
    "heading_deg",
    "heading_delta_next_deg",
    "heading_delta_prev_deg",
    "heading_finite",
    "heading_temporal_outlier",
    "heading_usable",
    "keypoints_img",
    "keypoints_roi",
    "left_axis_xy",
    "origin_xy",
)


@dataclass(frozen=True)
class QualityMetricDefinition:
    """One ordered, versioned diagnostic metric axis entry."""

    metric_id: str
    metric_version: int
    units: str
    higher_is_worse: bool
    description: str

    def __post_init__(self) -> None:
        metric_id = str(self.metric_id).strip()
        if not _IDENTIFIER.fullmatch(metric_id):
            raise ValueError("metric_id must be lowercase snake_case.")
        tokens = frozenset(metric_id.split("_"))
        if tokens.intersection(_FORBIDDEN_METRIC_TOKENS):
            raise ValueError(
                "keypoint-quality v1 metrics must be observation-local and "
                "cannot contain heading, temporal, track, or trajectory terms."
            )
        if type(self.metric_version) is not int or self.metric_version <= 0:
            raise ValueError("metric_version must be a positive exact integer.")
        units = str(self.units).strip()
        description = str(self.description).strip()
        if not units or "\n" in units:
            raise ValueError("units must be one nonempty line.")
        if not description:
            raise ValueError("description cannot be empty.")
        if type(self.higher_is_worse) is not bool:
            raise TypeError("higher_is_worse must be an exact bool.")
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "description", description)

    def as_manifest(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "units": self.units,
            "higher_is_worse": self.higher_is_worse,
            "description": self.description,
        }


def _normalize_flag_map(
    value: Mapping[int, str], *, field_name: str
) -> Mapping[int, str]:
    normalized: dict[int, str] = {}
    for raw_bit, raw_label in value.items():
        if type(raw_bit) is not int or not (1 <= raw_bit <= 0x8000):
            raise ValueError(
                f"{field_name} keys must be positive uint16 bit values."
            )
        if raw_bit & (raw_bit - 1):
            raise ValueError(f"{field_name} keys must each contain exactly one bit.")
        label = str(raw_label).strip()
        if not _IDENTIFIER.fullmatch(label):
            raise ValueError(
                f"{field_name} labels must be lowercase snake_case."
            )
        if raw_bit in normalized:
            raise ValueError(f"{field_name} contains duplicate bit {raw_bit}.")
        normalized[raw_bit] = label
    if len(set(normalized.values())) != len(normalized):
        raise ValueError(f"{field_name} labels must be unique.")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class KeypointQualityProfile:
    """Digest-bound metric axes, flag registries, and policy identity."""

    profile_id: str
    profile_version: int
    policy_digest: str
    keypoint_metrics: tuple[QualityMetricDefinition, ...]
    pose_metrics: tuple[QualityMetricDefinition, ...]
    keypoint_flag_map: Mapping[int, str]
    pose_flag_map: Mapping[int, str]

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        if not _IDENTIFIER.fullmatch(profile_id):
            raise ValueError("profile_id must be lowercase snake_case.")
        if type(self.profile_version) is not int or self.profile_version <= 0:
            raise ValueError("profile_version must be a positive exact integer.")
        policy_digest = str(self.policy_digest).strip()
        if not _SHA256.fullmatch(policy_digest):
            raise ValueError("policy_digest must be lowercase hexadecimal SHA-256.")
        keypoint_metrics = tuple(self.keypoint_metrics)
        pose_metrics = tuple(self.pose_metrics)
        if not keypoint_metrics and not pose_metrics:
            raise ValueError("A quality profile must declare at least one metric.")
        metric_ids = [
            metric.metric_id for metric in (*keypoint_metrics, *pose_metrics)
        ]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("Metric IDs must be unique across both metric axes.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "policy_digest", policy_digest)
        object.__setattr__(self, "keypoint_metrics", keypoint_metrics)
        object.__setattr__(self, "pose_metrics", pose_metrics)
        object.__setattr__(
            self,
            "keypoint_flag_map",
            _normalize_flag_map(
                self.keypoint_flag_map,
                field_name="keypoint_flag_map",
            ),
        )
        object.__setattr__(
            self,
            "pose_flag_map",
            _normalize_flag_map(self.pose_flag_map, field_name="pose_flag_map"),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "policy_digest": self.policy_digest,
            "keypoint_metrics": [
                metric.as_manifest() for metric in self.keypoint_metrics
            ],
            "pose_metrics": [metric.as_manifest() for metric in self.pose_metrics],
            "keypoint_flag_map": {
                str(bit): label for bit, label in self.keypoint_flag_map.items()
            },
            "pose_flag_map": {
                str(bit): label for bit, label in self.pose_flag_map.items()
            },
            "zero_flag_semantics": "no_quality_finding",
        }

    @property
    def profile_digest(self) -> str:
        encoded = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @property
    def declared_keypoint_flag_mask(self) -> int:
        return sum(self.keypoint_flag_map)

    @property
    def declared_pose_flag_mask(self) -> int:
        return sum(self.pose_flag_map)

    def as_manifest(self) -> dict[str, object]:
        return {**self._payload(), "profile_digest": self.profile_digest}


@dataclass(frozen=True)
class KeypointQualitySourceReference:
    """Immutable raw-keypoint authority bound by a quality run manifest."""

    run_name: str
    manifest_digest: str
    skeleton_id: str
    skeleton_digest: str
    keypoint_row_signatures_digest: str

    def __post_init__(self) -> None:
        run_name = str(self.run_name).strip()
        skeleton_id = str(self.skeleton_id).strip()
        if not run_name or "/" in run_name:
            raise ValueError("run_name must be one nonempty archive group name.")
        if not skeleton_id:
            raise ValueError("skeleton_id cannot be empty.")
        for field_name in (
            "manifest_digest",
            "skeleton_digest",
            "keypoint_row_signatures_digest",
        ):
            value = str(getattr(self, field_name)).strip()
            if not _SHA256.fullmatch(value):
                raise ValueError(
                    f"{field_name} must be lowercase hexadecimal SHA-256."
                )
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "skeleton_id", skeleton_id)

    def as_manifest(self) -> dict[str, object]:
        return {
            "stage": "keypoints",
            "run_name": self.run_name,
            "run_path": f"keypoints_runs/{self.run_name}",
            "schema_id": KEYPOINT_SCHEMA_ID,
            "schema_version": KEYPOINT_SCHEMA_VERSION,
            "manifest_digest": self.manifest_digest,
            "skeleton_id": self.skeleton_id,
            "skeleton_digest": self.skeleton_digest,
            "keypoint_row_signatures_digest": (
                self.keypoint_row_signatures_digest
            ),
            "coverage": "every_source_row_exactly_once_in_source_order",
        }


@dataclass(frozen=True)
class KeypointQualityDimensions:
    n_frames: int
    n_instances: int
    n_keypoints: int
    n_keypoint_metrics: int
    n_pose_metrics: int

    def __post_init__(self) -> None:
        for name in (
            "n_frames",
            "n_instances",
            "n_keypoints",
            "n_keypoint_metrics",
            "n_pose_metrics",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer.")
        if self.n_frames <= 0:
            raise ValueError("n_frames must be positive.")
        if self.n_instances < 0:
            raise ValueError("n_instances cannot be negative.")
        if self.n_keypoints <= 0:
            raise ValueError("n_keypoints must be positive.")
        if self.n_keypoint_metrics < 0 or self.n_pose_metrics < 0:
            raise ValueError("Metric dimensions cannot be negative.")
        if self.n_keypoint_metrics + self.n_pose_metrics == 0:
            raise ValueError("At least one metric dimension must be positive.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_frames": self.n_frames,
            "n_frame_boundaries": self.n_frames + 1,
            "n_instances": self.n_instances,
            "n_keypoints": self.n_keypoints,
            "n_keypoint_metrics": self.n_keypoint_metrics,
            "n_pose_metrics": self.n_pose_metrics,
        }

    def as_manifest(self) -> dict[str, int]:
        return self.contract_dimensions


@dataclass(frozen=True)
class KeypointQualitySchemaIssue:
    code: str
    path: str
    message: str

    def as_manifest(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class KeypointQualitySchemaError(ValueError):
    def __init__(self, issues: tuple[KeypointQualitySchemaIssue, ...]) -> None:
        self.issues = issues
        summary = "; ".join(
            f"{issue.code}:{issue.path}:{issue.message}" for issue in issues
        )
        super().__init__(summary)


def _issue(code: str, path: str, message: str) -> KeypointQualitySchemaIssue:
    return KeypointQualitySchemaIssue(code=code, path=path, message=message)


def _materialize(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[:])
    except (IndexError, TypeError):
        return np.asarray(value)


_CONTRACTS: tuple[tuple[str, ArrayContract], ...] = (
    ("instance_key", KEYPOINT_INSTANCE_KEY_V1),
    ("source_keypoint_row_ids", KEYPOINT_QUALITY_SOURCE_KEYPOINT_ROW_IDS_V1),
    (
        "source_keypoint_row_signature",
        KEYPOINT_QUALITY_SOURCE_KEYPOINT_ROW_SIGNATURE_V1,
    ),
    ("frame_indices", KEYPOINT_FRAME_INDICES_V1),
    ("frame_row_offsets", FRAME_ROW_OFFSETS_V1),
    ("keypoint_metric_values", KEYPOINT_QUALITY_KEYPOINT_METRIC_VALUES_V1),
    ("keypoint_metric_valid", KEYPOINT_QUALITY_KEYPOINT_METRIC_VALID_V1),
    ("pose_metric_values", KEYPOINT_QUALITY_POSE_METRIC_VALUES_V1),
    ("pose_metric_valid", KEYPOINT_QUALITY_POSE_METRIC_VALID_V1),
    ("keypoint_quality_flags", KEYPOINT_QUALITY_KEYPOINT_FLAGS_V1),
    ("pose_quality_flags", KEYPOINT_QUALITY_POSE_FLAGS_V1),
    (
        "proposed_keypoint_valid",
        KEYPOINT_QUALITY_PROPOSED_KEYPOINT_VALID_V1,
    ),
    ("proposed_pose_usable", KEYPOINT_QUALITY_PROPOSED_POSE_USABLE_V1),
)

KEYPOINT_QUALITY_BINDINGS = tuple(
    ArrayContractBinding(
        path=path,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=True,
    )
    for path, contract in _CONTRACTS
)


@dataclass(frozen=True)
class KeypointQualitySchema:
    schema_id: str
    schema_version: int
    stage: str
    layout: str
    base_path: str
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def __post_init__(self) -> None:
        paths = self.binding_paths
        if len(paths) != len(set(paths)):
            raise ValueError("Keypoint-quality binding paths must be unique.")
        for binding in self.bindings:
            if not binding.required:
                raise ValueError("Every keypoint-quality v1 array is required.")
            self.contracts.resolve(binding.contract_id, binding.contract_version)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: KeypointQualityDimensions,
        profile: KeypointQualityProfile,
        source_keypoint_arrays: Mapping[str, Any] | None,
    ) -> tuple[KeypointQualitySchemaIssue, ...]:
        issues: list[KeypointQualitySchemaIssue] = []
        expected = set(self.binding_paths)
        invalid_paths: set[str] = set()

        if len(profile.keypoint_metrics) != dimensions.n_keypoint_metrics:
            issues.append(
                _issue(
                    "keypoint_metric_catalog_size_mismatch",
                    "keypoint_metric_values",
                    "The keypoint metric axis must equal the ordered profile catalog.",
                )
            )
        if len(profile.pose_metrics) != dimensions.n_pose_metrics:
            issues.append(
                _issue(
                    "pose_metric_catalog_size_mismatch",
                    "pose_metric_values",
                    "The pose metric axis must equal the ordered profile catalog.",
                )
            )

        for path in sorted(set(arrays) - expected):
            code = (
                "heading_or_coordinate_payload_forbidden"
                if path in FORBIDDEN_KEYPOINT_QUALITY_ARRAYS
                else "unexpected_array"
            )
            issues.append(
                _issue(
                    code,
                    path,
                    "The exact keypoint-quality v1 schema does not declare this array.",
                )
            )

        for binding in self.bindings:
            path = binding.path
            if path not in arrays:
                invalid_paths.add(path)
                issues.append(
                    _issue(
                        "missing_required_array",
                        path,
                        "Required keypoint-quality array is absent.",
                    )
                )
                continue
            contract = self.contracts.resolve(
                binding.contract_id, binding.contract_version
            )
            try:
                errors = contract.validate_observation(
                    arrays[path], dimensions=dimensions.contract_dimensions
                )
            except Exception as exc:
                errors = (f"array metadata is unreadable: {exc}",)
            if errors:
                invalid_paths.add(path)
                issues.extend(
                    _issue("array_contract_violation", path, error)
                    for error in errors
                )

        values: dict[str, np.ndarray] = {}
        for path in self.binding_paths:
            if path in invalid_paths:
                continue
            try:
                values[path] = _materialize(arrays[path])
            except Exception as exc:
                issues.append(
                    _issue("array_read_failure", path, f"Array could not be read: {exc}")
                )

        frames = values.get("frame_indices")
        frames_valid = False
        if frames is not None:
            frames_valid = bool(
                np.all(frames >= 0) and np.all(frames < dimensions.n_frames)
            )
            if not frames_valid:
                issues.append(
                    _issue(
                        "frame_index_out_of_bounds",
                        "frame_indices",
                        "Every row frame must be in [0, n_frames).",
                    )
                )
            if frames.size > 1 and np.any(np.diff(frames) < 0):
                frames_valid = False
                issues.append(
                    _issue(
                        "frame_indices_not_sorted",
                        "frame_indices",
                        "Rows must be contiguous in nondecreasing frame order.",
                    )
                )

        offsets = values.get("frame_row_offsets")
        if offsets is not None:
            if int(offsets[0]) != 0:
                issues.append(
                    _issue(
                        "offset_start_mismatch",
                        "frame_row_offsets",
                        "Offsets must start at zero.",
                    )
                )
            if np.any(np.diff(offsets) < 0):
                issues.append(
                    _issue(
                        "offsets_not_monotonic",
                        "frame_row_offsets",
                        "Offsets must be nondecreasing.",
                    )
                )
            if int(offsets[-1]) != dimensions.n_instances:
                issues.append(
                    _issue(
                        "offset_end_mismatch",
                        "frame_row_offsets",
                        "The final offset must equal n_instances.",
                    )
                )
        if frames is not None and frames_valid and offsets is not None:
            expected_offsets = derive_frame_row_offsets(
                frames, n_frames=dimensions.n_frames
            )
            if not np.array_equal(offsets, expected_offsets):
                issues.append(
                    _issue(
                        "frame_row_offsets_mismatch",
                        "frame_row_offsets",
                        "Offsets must exactly index the contiguous quality rows.",
                    )
                )

        keys = values.get("instance_key")
        if keys is not None and np.unique(keys).size != keys.size:
            issues.append(
                _issue(
                    "duplicate_instance_key",
                    "instance_key",
                    "instance_key values must be unique.",
                )
            )
        source_rows = values.get("source_keypoint_row_ids")
        if source_rows is not None and not np.array_equal(
            source_rows, np.arange(dimensions.n_instances, dtype=np.int64)
        ):
            issues.append(
                _issue(
                    "incomplete_source_row_coverage",
                    "source_keypoint_row_ids",
                    "Quality v1 must cover every raw keypoint row exactly once and in order.",
                )
            )

        self._validate_metric_values(values, issues)
        self._validate_flag_values(values, profile=profile, issues=issues)
        self._validate_source_binding(
            values,
            source_keypoint_arrays=source_keypoint_arrays,
            dimensions=dimensions,
            issues=issues,
        )
        return tuple(issues)

    @staticmethod
    def _validate_metric_values(
        values: Mapping[str, np.ndarray],
        issues: list[KeypointQualitySchemaIssue],
    ) -> None:
        for value_path, valid_path in (
            ("keypoint_metric_values", "keypoint_metric_valid"),
            ("pose_metric_values", "pose_metric_valid"),
        ):
            metric_values = values.get(value_path)
            metric_valid = values.get(valid_path)
            if metric_values is None or metric_valid is None:
                continue
            finite = np.isfinite(metric_values)
            if not np.array_equal(metric_valid, finite):
                issues.append(
                    _issue(
                        "metric_validity_mismatch",
                        valid_path,
                        "Metric validity must exactly mark finite metric values.",
                    )
                )
            if np.any(~metric_valid & ~np.isnan(metric_values)):
                issues.append(
                    _issue(
                        "invalid_metric_not_nan",
                        value_path,
                        "Every invalid metric value must use canonical NaN.",
                    )
                )

    @staticmethod
    def _validate_flag_values(
        values: Mapping[str, np.ndarray],
        *,
        profile: KeypointQualityProfile,
        issues: list[KeypointQualitySchemaIssue],
    ) -> None:
        for path, declared_mask in (
            ("keypoint_quality_flags", profile.declared_keypoint_flag_mask),
            ("pose_quality_flags", profile.declared_pose_flag_mask),
        ):
            flags = values.get(path)
            if flags is None:
                continue
            undeclared_mask = np.uint32(0xFFFF ^ declared_mask)
            unknown = np.bitwise_and(flags.astype(np.uint32), undeclared_mask)
            if np.any(unknown):
                issues.append(
                    _issue(
                        "undeclared_quality_flag",
                        path,
                        "Persisted flags contain bits absent from the profile registry.",
                    )
                )

    @staticmethod
    def _validate_source_binding(
        values: Mapping[str, np.ndarray],
        *,
        source_keypoint_arrays: Mapping[str, Any] | None,
        dimensions: KeypointQualityDimensions,
        issues: list[KeypointQualitySchemaIssue],
    ) -> None:
        if source_keypoint_arrays is None:
            issues.append(
                _issue(
                    "missing_source_keypoint_evidence",
                    "source_keypoint_row_ids",
                    "Exact raw keypoint arrays are required for quality validation.",
                )
            )
            return
        required = {
            "instance_key": ((dimensions.n_instances,), np.dtype(np.uint64)),
            "frame_indices": ((dimensions.n_instances,), np.dtype(np.int64)),
            "keypoint_row_signature": (
                (dimensions.n_instances, 32),
                np.dtype(np.uint8),
            ),
            "keypoint_valid": (
                (dimensions.n_instances, dimensions.n_keypoints),
                np.dtype(bool),
            ),
            "pose_success": ((dimensions.n_instances,), np.dtype(bool)),
        }
        missing = sorted(set(required) - set(source_keypoint_arrays))
        if missing:
            issues.append(
                _issue(
                    "incomplete_source_keypoint_evidence",
                    "source_keypoint_row_ids",
                    f"Bound raw keypoint evidence is missing {missing!r}.",
                )
            )
            return
        try:
            source = {
                path: _materialize(source_keypoint_arrays[path]) for path in required
            }
        except Exception as exc:
            issues.append(
                _issue(
                    "source_keypoint_read_failure",
                    "source_keypoint_row_ids",
                    f"Bound raw keypoint evidence could not be read: {exc}",
                )
            )
            return
        malformed = False
        for path, (shape, dtype) in required.items():
            if source[path].shape != shape or source[path].dtype != dtype:
                malformed = True
                issues.append(
                    _issue(
                        "source_keypoint_contract_violation",
                        path,
                        f"Expected source shape {shape!r} and dtype {dtype}, got "
                        f"{source[path].shape!r} and {source[path].dtype}.",
                    )
                )
        if malformed:
            return

        comparisons = (
            ("instance_key", "instance_key"),
            ("frame_indices", "frame_indices"),
            ("source_keypoint_row_signature", "keypoint_row_signature"),
        )
        for target_path, source_path in comparisons:
            target = values.get(target_path)
            if target is not None and not np.array_equal(target, source[source_path]):
                issues.append(
                    _issue(
                        "source_keypoint_binding_mismatch",
                        target_path,
                        f"Values do not exactly equal source {source_path}.",
                    )
                )

        proposed_keypoints = values.get("proposed_keypoint_valid")
        if proposed_keypoints is not None and np.any(
            proposed_keypoints & ~source["keypoint_valid"]
        ):
            issues.append(
                _issue(
                    "proposed_keypoint_resurrects_invalid_source",
                    "proposed_keypoint_valid",
                    "A quality proposal cannot make an invalid raw landmark valid.",
                )
            )
        proposed_pose = values.get("proposed_pose_usable")
        if proposed_pose is not None:
            if np.any(proposed_pose & ~source["pose_success"]):
                issues.append(
                    _issue(
                        "proposed_pose_resurrects_failed_source",
                        "proposed_pose_usable",
                        "A quality proposal cannot make a failed raw pose usable.",
                    )
                )
            if proposed_keypoints is not None and np.any(
                proposed_pose & ~np.any(proposed_keypoints, axis=1)
            ):
                issues.append(
                    _issue(
                        "usable_pose_without_usable_keypoint",
                        "proposed_pose_usable",
                        "A proposed usable pose must retain at least one usable landmark.",
                    )
                )

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: KeypointQualityDimensions,
        profile: KeypointQualityProfile,
        source_keypoint_arrays: Mapping[str, Any] | None,
    ) -> None:
        issues = self.validate(
            arrays,
            dimensions=dimensions,
            profile=profile,
            source_keypoint_arrays=source_keypoint_arrays,
        )
        if issues:
            raise KeypointQualitySchemaError(issues)

    def as_manifest(
        self,
        *,
        dimensions: KeypointQualityDimensions,
        profile: KeypointQualityProfile,
        source: KeypointQualitySourceReference,
    ) -> dict[str, object]:
        if len(profile.keypoint_metrics) != dimensions.n_keypoint_metrics:
            raise ValueError("Profile keypoint metric count does not match dimensions.")
        if len(profile.pose_metrics) != dimensions.n_pose_metrics:
            raise ValueError("Profile pose metric count does not match dimensions.")
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "layout": self.layout,
            "base_path": self.base_path,
            "dimensions": dimensions.as_manifest(),
            "source": source.as_manifest(),
            "profile": profile.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "forbidden_arrays": list(FORBIDDEN_KEYPOINT_QUALITY_ARRAYS),
            "array_contracts": self.contracts.as_manifest(),
            "invariants": {
                "row_identity": "source_instance_key",
                "row_order": "exact_raw_keypoint_row_order",
                "frame_lookup": "frame_row_offsets_csr",
                "instances_per_frame": "zero_one_or_many",
                "metric_invalid_value": "canonical_nan",
                "quality_output": "diagnostic_and_proposed_not_accepted_review",
                "coordinates": "forbidden_reference_bound_keypoint_snapshot",
                "heading": "forbidden_use_bound_body_frame_run",
                "longitudinal_metrics": "forbidden_without_track_lineage_new_schema",
            },
        }


KEYPOINT_QUALITY_SCHEMA_V1 = KeypointQualitySchema(
    schema_id=KEYPOINT_QUALITY_SCHEMA_ID,
    schema_version=KEYPOINT_QUALITY_SCHEMA_VERSION,
    stage="keypoint_quality",
    layout=KEYPOINT_QUALITY_LAYOUT,
    base_path="keypoint_quality_runs/<run>",
    bindings=KEYPOINT_QUALITY_BINDINGS,
    contracts=KEYPOINT_QUALITY_ARRAY_CONTRACTS,
)


__all__ = [
    "FORBIDDEN_KEYPOINT_QUALITY_ARRAYS",
    "KEYPOINT_QUALITY_BINDINGS",
    "KEYPOINT_QUALITY_LAYOUT",
    "KEYPOINT_QUALITY_SCHEMA_ID",
    "KEYPOINT_QUALITY_SCHEMA_V1",
    "KEYPOINT_QUALITY_SCHEMA_VERSION",
    "KeypointQualityDimensions",
    "KeypointQualityProfile",
    "KeypointQualitySourceReference",
    "KeypointQualitySchema",
    "KeypointQualitySchemaError",
    "KeypointQualitySchemaIssue",
    "QualityMetricDefinition",
]
