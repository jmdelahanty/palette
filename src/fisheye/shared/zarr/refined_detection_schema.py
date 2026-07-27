"""Frozen logical schema for immutable refined-detection snapshots.

This module performs no Zarr I/O.  It defines the exact canonical arrays,
dtypes, row identities, validity encodings, source-candidate audit joins, and
frame-row offset invariants that a future writer must satisfy before immutable
publication.  Mutable compatibility writers and sparse edit deltas are outside
this schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    DETECTION_BBOX_IMG_XYXY_V1,
    DETECTION_BBOX_NORM_COORDS_V1,
    DETECTION_CENTERS_IMG_XY_V1,
    DETECTION_CLASS_IDS_V1,
    DETECTION_FRAME_INDICES_V1,
    DETECTION_INSTANCE_KEY_V1,
    DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    FRAME_ROW_OFFSETS_V1,
    REFINED_DETECTION_ARRAY_CONTRACTS,
    REFINED_DETECTION_MANUAL_EDIT_FLAGS_V1,
    REFINED_DETECTION_REASON_CODES_V1,
    REFINED_DETECTION_REFINED_ROW_IDS_V1,
    REFINED_DETECTION_SCORES_V1,
    REFINED_DETECTION_SCORE_VALID_V1,
    REFINED_DETECTION_SOURCE_DETECT_ROW_INDEX_V1,
    REFINED_DETECTION_SOURCE_KIND_CODES_V1,
    REFINED_INSTANCE_SOURCE_CLIP_DETECT_ROW_INDEX_V1,
    REFINED_INSTANCE_SOURCE_CLIP_INDICES_V1,
    REFINED_INSTANCE_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1,
    REFINED_INSTANCE_SOURCE_RECORDING_FRAME_IDS_V1,
    REFINED_INSTANCE_SOURCE_REFINED_ROW_IDS_V1,
    REFINED_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    REFINED_SOURCE_BBOX_IMG_XYXY_V1,
    REFINED_SOURCE_BBOX_NORM_COORDS_V1,
    REFINED_SOURCE_CENTERS_IMG_XY_V1,
    REFINED_SOURCE_CLASS_IDS_V1,
    REFINED_SOURCE_CLIP_DETECT_ROW_INDEX_V1,
    REFINED_SOURCE_CLIP_INDICES_V1,
    REFINED_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1,
    REFINED_SOURCE_DECISION_CODES_V1,
    REFINED_SOURCE_DETECT_ROW_INDEX_V1,
    REFINED_SOURCE_FRAME_INDICES_V1,
    REFINED_SOURCE_FRAME_ROW_OFFSETS_V1,
    REFINED_SOURCE_INSTANCE_KEY_V1,
    REFINED_SOURCE_REASON_CODES_V1,
    REFINED_SOURCE_RECORDING_FRAME_IDS_V1,
    REFINED_SOURCE_RESOLVED_REFINED_ROW_ID_V1,
    REFINED_SOURCE_RESOLVED_SOURCE_REFINED_ROW_ID_V1,
    REFINED_SOURCE_SCORES_V1,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)
from fisheye.shared.zarr.detection_schema import (
    MAX_CANONICAL_CLASS_ID,
    derive_canonical_detection_geometry,
)


REFINED_DETECTION_SCHEMA_ID = "palette.stage.refined_detection"
REFINED_DETECTION_SCHEMA_VERSION = 1
REFINED_DETECTION_LAYOUT = (
    "immutable_sparse_instances_with_source_audit_and_frame_row_offsets_v1"
)
REFINED_DETECTION_INSTANCE_GROUP = "instances"
REFINED_DETECTION_SOURCE_GROUP = "source_detections"

SOURCE_KIND_CODE_MAP = {"raw_detect": 1, "manual": 3}
SOURCE_DECISION_CODE_MAP = {
    "accepted": 0,
    "filtered": 1,
    "duplicate": 2,
    "manual_clear": 3,
}
NO_SOURCE_ROW = -1
NO_RESOLVED_REFINED_ROW = -1
NO_REASON_CODE = 0


class RefinedDetectionLineageProfile(str, Enum):
    """Conditional exact lineage bindings for one publication context."""

    FULL_ACQUISITION = "full_acquisition"
    CLIPPED_RECORDING_SNAPSHOT = "clipped_recording_snapshot"


def _binding(path: str, contract: ArrayContract) -> ArrayContractBinding:
    return ArrayContractBinding(
        path=path,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=True,
    )


def _instance(name: str) -> str:
    return f"{REFINED_DETECTION_INSTANCE_GROUP}/{name}"


def _source(name: str) -> str:
    return f"{REFINED_DETECTION_SOURCE_GROUP}/{name}"


REFINED_DETECTION_CORE_BINDINGS = (
    _binding(_instance("frame_indices"), DETECTION_FRAME_INDICES_V1),
    _binding(
        _instance("source_acquisition_frame_index"),
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    ),
    _binding(_instance("instance_key"), DETECTION_INSTANCE_KEY_V1),
    _binding(_instance("refined_row_ids"), REFINED_DETECTION_REFINED_ROW_IDS_V1),
    _binding(_instance("bbox_norm_coords"), DETECTION_BBOX_NORM_COORDS_V1),
    _binding(_instance("bbox_img_xyxy"), DETECTION_BBOX_IMG_XYXY_V1),
    _binding(_instance("centers_img_xy"), DETECTION_CENTERS_IMG_XY_V1),
    _binding(_instance("scores"), REFINED_DETECTION_SCORES_V1),
    _binding(_instance("score_valid"), REFINED_DETECTION_SCORE_VALID_V1),
    _binding(_instance("class_ids"), DETECTION_CLASS_IDS_V1),
    _binding(
        _instance("source_kind_codes"),
        REFINED_DETECTION_SOURCE_KIND_CODES_V1,
    ),
    _binding(
        _instance("manual_edit_flags"),
        REFINED_DETECTION_MANUAL_EDIT_FLAGS_V1,
    ),
    _binding(
        _instance("source_detect_row_index"),
        REFINED_DETECTION_SOURCE_DETECT_ROW_INDEX_V1,
    ),
    _binding(_instance("reason_codes"), REFINED_DETECTION_REASON_CODES_V1),
    _binding(_instance("frame_row_offsets"), FRAME_ROW_OFFSETS_V1),
    _binding(
        _source("source_detect_row_index"),
        REFINED_SOURCE_DETECT_ROW_INDEX_V1,
    ),
    _binding(_source("frame_indices"), REFINED_SOURCE_FRAME_INDICES_V1),
    _binding(
        _source("source_acquisition_frame_index"),
        REFINED_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    ),
    _binding(_source("instance_key"), REFINED_SOURCE_INSTANCE_KEY_V1),
    _binding(_source("bbox_norm_coords"), REFINED_SOURCE_BBOX_NORM_COORDS_V1),
    _binding(_source("bbox_img_xyxy"), REFINED_SOURCE_BBOX_IMG_XYXY_V1),
    _binding(_source("centers_img_xy"), REFINED_SOURCE_CENTERS_IMG_XY_V1),
    _binding(_source("scores"), REFINED_SOURCE_SCORES_V1),
    _binding(_source("class_ids"), REFINED_SOURCE_CLASS_IDS_V1),
    _binding(_source("decision_codes"), REFINED_SOURCE_DECISION_CODES_V1),
    _binding(
        _source("resolved_refined_row_id"),
        REFINED_SOURCE_RESOLVED_REFINED_ROW_ID_V1,
    ),
    _binding(_source("reason_codes"), REFINED_SOURCE_REASON_CODES_V1),
    _binding(
        _source("frame_row_offsets"),
        REFINED_SOURCE_FRAME_ROW_OFFSETS_V1,
    ),
)

REFINED_DETECTION_CLIPPED_LINEAGE_BINDINGS = (
    _binding(
        _instance("source_recording_frame_ids"),
        REFINED_INSTANCE_SOURCE_RECORDING_FRAME_IDS_V1,
    ),
    _binding(
        _instance("source_clip_indices"),
        REFINED_INSTANCE_SOURCE_CLIP_INDICES_V1,
    ),
    _binding(
        _instance("source_clip_local_frame_indices"),
        REFINED_INSTANCE_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1,
    ),
    _binding(
        _instance("source_clip_detect_row_index"),
        REFINED_INSTANCE_SOURCE_CLIP_DETECT_ROW_INDEX_V1,
    ),
    _binding(
        _instance("source_refined_row_ids"),
        REFINED_INSTANCE_SOURCE_REFINED_ROW_IDS_V1,
    ),
    _binding(
        _source("source_recording_frame_ids"),
        REFINED_SOURCE_RECORDING_FRAME_IDS_V1,
    ),
    _binding(
        _source("source_clip_indices"),
        REFINED_SOURCE_CLIP_INDICES_V1,
    ),
    _binding(
        _source("source_clip_local_frame_indices"),
        REFINED_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1,
    ),
    _binding(
        _source("source_clip_detect_row_index"),
        REFINED_SOURCE_CLIP_DETECT_ROW_INDEX_V1,
    ),
    _binding(
        _source("source_resolved_refined_row_id"),
        REFINED_SOURCE_RESOLVED_SOURCE_REFINED_ROW_ID_V1,
    ),
)

FORBIDDEN_REFINED_DETECTION_PATHS = (
    "frame_counts",
    "frame_offsets",
    "n_detections",
    _instance("frame_counts"),
    _instance("frame_offsets"),
    _instance("n_detections"),
    _instance("confidence_scores"),
    _instance("reason"),
    _instance("reason_bytes"),
    _instance("review_notes"),
    _source("frame_counts"),
    _source("frame_offsets"),
    _source("n_detections"),
    _source("confidence_scores"),
    _source("reason"),
    _source("reason_bytes"),
    _source("review_notes"),
)


@dataclass(frozen=True)
class RefinedDetectionDimensions:
    """Concrete dimensions and lineage profile for one immutable snapshot."""

    n_frames: int
    n_instances: int
    n_source_detections: int
    source_width: int
    source_height: int
    lineage_profile: RefinedDetectionLineageProfile = (
        RefinedDetectionLineageProfile.FULL_ACQUISITION
    )

    def __post_init__(self) -> None:
        for name in (
            "n_frames",
            "n_instances",
            "n_source_detections",
            "source_width",
            "source_height",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer.")
        if self.n_frames < 0:
            raise ValueError("n_frames cannot be negative.")
        if self.n_instances < 0 or self.n_source_detections < 0:
            raise ValueError("Refined-detection row counts cannot be negative.")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("Source-camera width and height must be positive.")
        if self.n_frames > int(np.iinfo(np.int32).max):
            raise ValueError("n_frames exceeds the canonical int32 frame domain.")
        object.__setattr__(
            self,
            "lineage_profile",
            RefinedDetectionLineageProfile(self.lineage_profile),
        )

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_frames": self.n_frames,
            "n_instances": self.n_instances,
            "n_source_detections": self.n_source_detections,
            "n_frame_boundaries": self.n_frames + 1,
        }

    def as_manifest(self) -> dict[str, object]:
        return {
            **self.contract_dimensions,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "lineage_profile": self.lineage_profile.value,
        }


@dataclass(frozen=True)
class RefinedDetectionSchemaIssue:
    code: str
    path: str
    message: str

    def as_manifest(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class RefinedDetectionSchemaError(ValueError):
    def __init__(self, issues: tuple[RefinedDetectionSchemaIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(
            f"Refined detection schema validation failed with "
            f"{len(issues)} issue(s): {detail}"
        )


def _issue(code: str, path: str, message: str) -> RefinedDetectionSchemaIssue:
    return RefinedDetectionSchemaIssue(code=code, path=path, message=message)


def _materialize(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    try:
        return np.asarray(array[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(array)


def _bbox_valid(values: np.ndarray) -> bool:
    half = np.float32(0.5)
    one = np.float32(1.0)
    return bool(
        np.isfinite(values).all()
        and np.all(values[:, 2] > 0)
        and np.all(values[:, 3] > 0)
        and np.all(values[:, 0] - values[:, 2] * half >= 0)
        and np.all(values[:, 1] - values[:, 3] * half >= 0)
        and np.all(values[:, 0] + values[:, 2] * half <= one)
        and np.all(values[:, 1] + values[:, 3] * half <= one)
    )


def _expected_offsets(frames: np.ndarray, n_frames: int) -> np.ndarray:
    counts = np.bincount(frames.astype(np.int64, copy=False), minlength=n_frames)
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    if n_frames:
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


@dataclass(frozen=True)
class RefinedDetectionSchema:
    schema_id: str
    schema_version: int
    core_bindings: tuple[ArrayContractBinding, ...]
    clipped_lineage_bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog

    def __post_init__(self) -> None:
        if not self.schema_id.strip():
            raise ValueError("schema_id cannot be empty.")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive exact integer.")
        all_bindings = (*self.core_bindings, *self.clipped_lineage_bindings)
        paths = [binding.path for binding in all_bindings]
        if len(paths) != len(set(paths)):
            raise ValueError("Refined detection binding paths must be unique.")
        for binding in all_bindings:
            if not binding.required:
                raise ValueError("Conditional profiles use required, not optional, bindings.")
            self.contracts.resolve(binding.contract_id, binding.contract_version)

    def bindings_for(
        self,
        dimensions: RefinedDetectionDimensions,
    ) -> tuple[ArrayContractBinding, ...]:
        if (
            dimensions.lineage_profile
            is RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT
        ):
            return (*self.core_bindings, *self.clipped_lineage_bindings)
        return self.core_bindings

    def binding_paths_for(
        self,
        dimensions: RefinedDetectionDimensions,
    ) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings_for(dimensions))

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: RefinedDetectionDimensions,
    ) -> tuple[RefinedDetectionSchemaIssue, ...]:
        issues: list[RefinedDetectionSchemaIssue] = []
        bindings = self.bindings_for(dimensions)
        expected_paths = {binding.path for binding in bindings}
        invalid_paths: set[str] = set()

        for path in FORBIDDEN_REFINED_DETECTION_PATHS:
            if path in arrays:
                issues.append(
                    _issue(
                        "forbidden_legacy_binding",
                        path,
                        "Legacy count, offset alias, float64-era naming, or "
                        "variable-text arrays are outside refined-detection v1.",
                    )
                )
        for path in arrays:
            if (
                path.startswith(f"{REFINED_DETECTION_INSTANCE_GROUP}/")
                or path.startswith(f"{REFINED_DETECTION_SOURCE_GROUP}/")
            ) and path not in expected_paths:
                issues.append(
                    _issue(
                        "unexpected_array",
                        path,
                        "Canonical refined-detection groups are exact; add a "
                        "new schema version for additional arrays.",
                    )
                )

        for binding in bindings:
            path = binding.path
            if path not in arrays:
                invalid_paths.add(path)
                issues.append(
                    _issue(
                        "missing_required_array",
                        path,
                        "Required refined-detection array is absent.",
                    )
                )
                continue
            contract = self.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            try:
                errors = contract.validate_observation(
                    arrays[path],
                    dimensions=dimensions.contract_dimensions,
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
        for path in expected_paths - invalid_paths:
            try:
                values[path] = _materialize(arrays[path])
            except Exception as exc:
                issues.append(
                    _issue("array_read_failure", path, f"Cannot read values: {exc}")
                )

        self._validate_table(
            values,
            issues,
            prefix=REFINED_DETECTION_INSTANCE_GROUP,
            row_count=dimensions.n_instances,
            n_frames=dimensions.n_frames,
            source_width=dimensions.source_width,
            source_height=dimensions.source_height,
            refined=True,
        )
        self._validate_table(
            values,
            issues,
            prefix=REFINED_DETECTION_SOURCE_GROUP,
            row_count=dimensions.n_source_detections,
            n_frames=dimensions.n_frames,
            source_width=dimensions.source_width,
            source_height=dimensions.source_height,
            refined=False,
        )
        self._validate_refined_semantics(values, issues, dimensions)
        if (
            dimensions.lineage_profile
            is RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT
        ):
            self._validate_clipped_lineage(values, issues)
        return tuple(issues)

    @staticmethod
    def _validate_table(
        values: Mapping[str, np.ndarray],
        issues: list[RefinedDetectionSchemaIssue],
        *,
        prefix: str,
        row_count: int,
        n_frames: int,
        source_width: int,
        source_height: int,
        refined: bool,
    ) -> None:
        frame_path = f"{prefix}/frame_indices"
        acquisition_path = f"{prefix}/source_acquisition_frame_index"
        key_path = f"{prefix}/instance_key"
        bbox_norm_path = f"{prefix}/bbox_norm_coords"
        bbox_img_path = f"{prefix}/bbox_img_xyxy"
        centers_path = f"{prefix}/centers_img_xy"
        offsets_path = f"{prefix}/frame_row_offsets"
        frames = values.get(frame_path)
        if frames is not None:
            if not bool(np.all(frames >= 0) and np.all(frames < n_frames)):
                issues.append(
                    _issue(
                        "frame_index_out_of_bounds",
                        frame_path,
                        "Every frame index must be in [0, n_frames).",
                    )
                )
            if frames.size > 1 and np.any(np.diff(frames) < 0):
                issues.append(
                    _issue(
                        "frame_indices_not_sorted",
                        frame_path,
                        "Rows must be contiguous in nondecreasing frame order.",
                    )
                )
        acquisition = values.get(acquisition_path)
        if acquisition is not None and np.any(acquisition < 0):
            issues.append(
                _issue(
                    "invalid_acquisition_frame_index",
                    acquisition_path,
                    "Acquisition frame indices must be nonnegative.",
                )
            )
        if frames is not None and acquisition is not None and not np.array_equal(
            acquisition,
            frames.astype(np.int64, copy=False),
        ):
            issues.append(
                _issue(
                    "acquisition_frame_join_mismatch",
                    acquisition_path,
                    "Recording-level snapshots use acquisition frame identity; "
                    "source_acquisition_frame_index must exactly equal "
                    "frame_indices represented as int64.",
                )
            )
        keys = values.get(key_path)
        if keys is not None and np.unique(keys).shape[0] != row_count:
            issues.append(
                _issue(
                    "duplicate_instance_key",
                    key_path,
                    "Instance keys must be unique within each table.",
                )
            )
        bbox_norm = values.get(bbox_norm_path)
        bbox_img = values.get(bbox_img_path)
        centers = values.get(centers_path)
        if bbox_norm is not None:
            if not _bbox_valid(bbox_norm):
                issues.append(
                    _issue(
                        "invalid_bbox_norm_coords",
                        bbox_norm_path,
                        "Boxes must be finite, positive-area, and contained in "
                        "the normalized source-camera extent.",
                    )
                )
            elif bbox_img is not None and centers is not None:
                expected_bbox, expected_centers = derive_canonical_detection_geometry(
                    bbox_norm,
                    source_width=source_width,
                    source_height=source_height,
                )
                if not np.array_equal(bbox_img, expected_bbox):
                    issues.append(
                        _issue(
                            "bbox_img_projection_mismatch",
                            bbox_img_path,
                            "Pixel boxes must exactly project bbox_norm_coords.",
                        )
                    )
                if not np.array_equal(centers, expected_centers):
                    issues.append(
                        _issue(
                            "center_projection_mismatch",
                            centers_path,
                            "Centers must exactly equal bbox_img_xyxy midpoints.",
                        )
                    )
        offsets = values.get(offsets_path)
        if offsets is not None:
            if not offsets.size or int(offsets[0]) != 0:
                issues.append(
                    _issue("offset_start_mismatch", offsets_path, "Offsets must start at zero.")
                )
            if offsets.size and np.any(np.diff(offsets) < 0):
                issues.append(
                    _issue("offsets_not_monotonic", offsets_path, "Offsets must be nondecreasing.")
                )
            if not offsets.size or int(offsets[-1]) != row_count:
                issues.append(
                    _issue(
                        "offset_end_mismatch",
                        offsets_path,
                        "Final offset must equal the table row count.",
                    )
                )
            if frames is not None and bool(
                np.all(frames >= 0) and np.all(frames < n_frames)
            ):
                if not np.array_equal(offsets, _expected_offsets(frames, n_frames)):
                    issues.append(
                        _issue(
                            "frame_row_offsets_mismatch",
                            offsets_path,
                            "Offsets must exactly describe frame_indices ranges.",
                        )
                    )
        if refined:
            row_ids = values.get(f"{prefix}/refined_row_ids")
            if row_ids is not None:
                if np.any(row_ids < 0) or np.unique(row_ids).shape[0] != row_count:
                    issues.append(
                        _issue(
                            "invalid_refined_row_ids",
                            f"{prefix}/refined_row_ids",
                            "Refined row IDs must be unique and nonnegative.",
                        )
                    )
                if frames is not None and row_count:
                    expected_order = np.lexsort((row_ids, frames))
                    if not np.array_equal(expected_order, np.arange(row_count)):
                        issues.append(
                            _issue(
                                "refined_rows_not_sorted",
                                f"{prefix}/refined_row_ids",
                                "Rows must be sorted by frame_indices then refined_row_ids.",
                            )
                        )

    @staticmethod
    def _validate_refined_semantics(
        values: Mapping[str, np.ndarray],
        issues: list[RefinedDetectionSchemaIssue],
        dimensions: RefinedDetectionDimensions,
    ) -> None:
        inst = REFINED_DETECTION_INSTANCE_GROUP
        source = REFINED_DETECTION_SOURCE_GROUP
        inst_scores = values.get(f"{inst}/scores")
        score_valid = values.get(f"{inst}/score_valid")
        if inst_scores is not None and score_valid is not None:
            valid_scores = bool(
                np.isfinite(inst_scores).all()
                and np.all(inst_scores >= np.float32(0.0))
                and np.all(inst_scores <= np.float32(1.0))
                and np.all(inst_scores[~score_valid] == np.float32(0.0))
            )
            if not valid_scores:
                issues.append(
                    _issue(
                        "invalid_score_encoding",
                        f"{inst}/scores",
                        "Scores must be finite in [0,1], and invalid scores must "
                        "use exact value zero with score_valid=false.",
                    )
                )
        source_scores = values.get(f"{source}/scores")
        if source_scores is not None and not bool(
            np.isfinite(source_scores).all()
            and np.all(source_scores >= np.float32(0.0))
            and np.all(source_scores <= np.float32(1.0))
        ):
            issues.append(
                _issue(
                    "invalid_source_score",
                    f"{source}/scores",
                    "Source scores must be finite float32 values in [0,1].",
                )
            )
        for path in (f"{inst}/class_ids", f"{source}/class_ids"):
            class_ids = values.get(path)
            if class_ids is not None and not bool(
                np.all(class_ids >= 0)
                and np.all(class_ids <= MAX_CANONICAL_CLASS_ID)
            ):
                issues.append(
                    _issue(
                        "invalid_class_id",
                        path,
                        f"Class IDs must be in [0, {MAX_CANONICAL_CLASS_ID}].",
                    )
                )

        source_rows = values.get(f"{source}/source_detect_row_index")
        if source_rows is not None and not np.array_equal(
            source_rows,
            np.arange(dimensions.n_source_detections, dtype=np.int64),
        ):
            issues.append(
                _issue(
                    "source_rows_not_contiguous",
                    f"{source}/source_detect_row_index",
                    "Source audit rows must be exact contiguous row identities.",
                )
            )
        decisions = values.get(f"{source}/decision_codes")
        resolved = values.get(f"{source}/resolved_refined_row_id")
        refined_ids = values.get(f"{inst}/refined_row_ids")
        if decisions is not None:
            allowed_decisions = np.asarray(
                list(SOURCE_DECISION_CODE_MAP.values()), dtype=np.uint8
            )
            if not np.isin(decisions, allowed_decisions).all():
                issues.append(
                    _issue(
                        "unknown_source_decision_code",
                        f"{source}/decision_codes",
                        "Decision codes must use the frozen v1 map.",
                    )
                )
        if decisions is not None and resolved is not None and refined_ids is not None:
            accepted = decisions == SOURCE_DECISION_CODE_MAP["accepted"]
            if np.any(resolved[~accepted] != NO_RESOLVED_REFINED_ROW):
                issues.append(
                    _issue(
                        "unaccepted_source_has_resolution",
                        f"{source}/resolved_refined_row_id",
                        "Only accepted source rows may resolve to a refined row.",
                    )
                )
            if np.any(resolved[accepted] < 0) or not np.isin(
                resolved[accepted], refined_ids
            ).all():
                issues.append(
                    _issue(
                        "accepted_source_resolution_invalid",
                        f"{source}/resolved_refined_row_id",
                        "Every accepted source row must resolve to a present refined row.",
                    )
                )

        kind = values.get(f"{inst}/source_kind_codes")
        inst_source_rows = values.get(f"{inst}/source_detect_row_index")
        manual_flags = values.get(f"{inst}/manual_edit_flags")
        if kind is not None:
            allowed_kinds = np.asarray(list(SOURCE_KIND_CODE_MAP.values()), dtype=np.uint8)
            if not np.isin(kind, allowed_kinds).all():
                issues.append(
                    _issue(
                        "unknown_source_kind_code",
                        f"{inst}/source_kind_codes",
                        "Source kind codes must be raw_detect or manual in v1.",
                    )
                )
        if (
            kind is None
            or inst_source_rows is None
            or manual_flags is None
            or score_valid is None
            or refined_ids is None
        ):
            return
        raw_mask = kind == SOURCE_KIND_CODE_MAP["raw_detect"]
        manual_mask = kind == SOURCE_KIND_CODE_MAP["manual"]
        instance_keys = values.get(f"{inst}/instance_key")
        source_keys = values.get(f"{source}/instance_key")
        if (
            instance_keys is not None
            and source_keys is not None
            and np.isin(instance_keys[manual_mask], source_keys).any()
        ):
            issues.append(
                _issue(
                    "manual_instance_key_collision",
                    f"{inst}/instance_key",
                    "A manual instance key must not collide with any bound "
                    "source-candidate key.",
                )
            )
        if np.any(inst_source_rows[manual_mask] != NO_SOURCE_ROW):
            issues.append(
                _issue(
                    "manual_source_row_mismatch",
                    f"{inst}/source_detect_row_index",
                    "Manual rows must use source_detect_row_index=-1.",
                )
            )
        if np.any(~manual_flags[manual_mask]) or np.any(score_valid[manual_mask]):
            issues.append(
                _issue(
                    "manual_row_semantics_invalid",
                    f"{inst}/source_kind_codes",
                    "Manual rows require manual_edit_flags=true and score_valid=false.",
                )
            )
        if np.any(inst_source_rows[raw_mask] < 0) or np.any(
            inst_source_rows[raw_mask] >= dimensions.n_source_detections
        ):
            issues.append(
                _issue(
                    "raw_source_row_out_of_bounds",
                    f"{inst}/source_detect_row_index",
                    "Raw-backed rows must address the source audit table.",
                )
            )
            return
        if np.any(~score_valid[raw_mask]):
            issues.append(
                _issue(
                    "raw_score_missing",
                    f"{inst}/score_valid",
                    "Raw-backed rows must preserve a valid source score.",
                )
            )
        raw_positions = np.flatnonzero(raw_mask)
        source_positions = inst_source_rows[raw_mask].astype(np.int64, copy=False)
        if np.unique(source_positions).size != source_positions.size:
            issues.append(
                _issue(
                    "duplicate_raw_source_resolution",
                    f"{inst}/source_detect_row_index",
                    "One source candidate may resolve to at most one refined row.",
                )
            )
        joins = (
            ("instance_key", "instance_key"),
            ("frame_indices", "frame_indices"),
            ("source_acquisition_frame_index", "source_acquisition_frame_index"),
            ("scores", "scores"),
        )
        for inst_name, source_name in joins:
            inst_values = values.get(f"{inst}/{inst_name}")
            source_values = values.get(f"{source}/{source_name}")
            if inst_values is not None and source_values is not None and not np.array_equal(
                inst_values[raw_positions], source_values[source_positions]
            ):
                issues.append(
                    _issue(
                        "raw_source_join_mismatch",
                        f"{inst}/{inst_name}",
                        f"Raw-backed {inst_name} must equal its source audit row.",
                    )
                )
        unedited_raw_positions = raw_positions[~manual_flags[raw_mask]]
        unedited_source_positions = source_positions[~manual_flags[raw_mask]]
        for name in ("bbox_norm_coords", "class_ids"):
            inst_values = values.get(f"{inst}/{name}")
            source_values = values.get(f"{source}/{name}")
            if (
                inst_values is not None
                and source_values is not None
                and not np.array_equal(
                    inst_values[unedited_raw_positions],
                    source_values[unedited_source_positions],
                )
            ):
                issues.append(
                    _issue(
                        "unedited_raw_value_mismatch",
                        f"{inst}/{name}",
                        f"Unedited raw-backed {name} must equal its source audit row; "
                        "set manual_edit_flags=true for a correction.",
                    )
                )
        if decisions is not None and resolved is not None:
            if np.any(
                decisions[source_positions] != SOURCE_DECISION_CODE_MAP["accepted"]
            ) or not np.array_equal(resolved[source_positions], refined_ids[raw_positions]):
                issues.append(
                    _issue(
                        "raw_resolution_join_mismatch",
                        f"{source}/resolved_refined_row_id",
                        "Raw-backed instances require accepted source rows resolving "
                        "to the same refined_row_id.",
                    )
                )
            accepted_positions = np.flatnonzero(
                decisions == SOURCE_DECISION_CODE_MAP["accepted"]
            )
            if not np.array_equal(
                np.sort(source_positions),
                accepted_positions,
            ):
                issues.append(
                    _issue(
                        "accepted_source_rowset_mismatch",
                        f"{source}/decision_codes",
                        "Accepted source rows must be exactly the source rows "
                        "represented by raw-backed refined instances.",
                    )
                )

    @staticmethod
    def _validate_clipped_lineage(
        values: Mapping[str, np.ndarray],
        issues: list[RefinedDetectionSchemaIssue],
    ) -> None:
        nonnegative_paths = (
            _instance("source_clip_indices"),
            _instance("source_clip_local_frame_indices"),
            _instance("source_refined_row_ids"),
            _source("source_clip_indices"),
            _source("source_clip_local_frame_indices"),
            _source("source_clip_detect_row_index"),
        )
        positive_paths = (
            _instance("source_recording_frame_ids"),
            _source("source_recording_frame_ids"),
        )
        for path in nonnegative_paths:
            data = values.get(path)
            if data is not None and np.any(data < 0):
                issues.append(
                    _issue(
                        "invalid_clipped_lineage",
                        path,
                        "Clipped lineage values must be nonnegative.",
                    )
                )
        for path in positive_paths:
            data = values.get(path)
            if data is not None and np.any(data <= 0):
                issues.append(
                    _issue(
                        "invalid_recording_frame_id",
                        path,
                        "Recording frame IDs are one-based and must be positive.",
                    )
                )
        for prefix in (
            REFINED_DETECTION_INSTANCE_GROUP,
            REFINED_DETECTION_SOURCE_GROUP,
        ):
            recording_ids = values.get(f"{prefix}/source_recording_frame_ids")
            acquisition = values.get(
                f"{prefix}/source_acquisition_frame_index"
            )
            if (
                recording_ids is not None
                and acquisition is not None
                and not np.array_equal(recording_ids, acquisition + np.int64(1))
            ):
                issues.append(
                    _issue(
                        "recording_frame_join_mismatch",
                        f"{prefix}/source_recording_frame_ids",
                        "One-based recording frame IDs must equal acquisition "
                        "frame indices plus one.",
                    )
                )
        kind = values.get(_instance("source_kind_codes"))
        clip_rows = values.get(_instance("source_clip_detect_row_index"))
        if kind is not None and clip_rows is not None:
            manual = kind == SOURCE_KIND_CODE_MAP["manual"]
            raw = kind == SOURCE_KIND_CODE_MAP["raw_detect"]
            if np.any(clip_rows[manual] != NO_SOURCE_ROW) or np.any(clip_rows[raw] < 0):
                issues.append(
                    _issue(
                        "clip_source_row_semantics_invalid",
                        _instance("source_clip_detect_row_index"),
                        "Manual rows use -1; raw-backed rows require a clip-local row.",
                    )
                )

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: RefinedDetectionDimensions,
    ) -> None:
        issues = self.validate(arrays, dimensions=dimensions)
        if issues:
            raise RefinedDetectionSchemaError(issues)

    def as_manifest(
        self,
        *,
        dimensions: RefinedDetectionDimensions,
    ) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": "refined_detect",
            "layout": REFINED_DETECTION_LAYOUT,
            "base_path": "refined_detect_runs/<run>",
            "dimensions": dimensions.as_manifest(),
            "bindings": [
                binding.as_manifest() for binding in self.bindings_for(dimensions)
            ],
            "forbidden_legacy_bindings": list(FORBIDDEN_REFINED_DETECTION_PATHS),
            "array_contracts": self.contracts.as_manifest(),
            "code_maps": {
                "source_kind_codes": dict(SOURCE_KIND_CODE_MAP),
                "source_detections/decision_codes": dict(SOURCE_DECISION_CODE_MAP),
                "reason_codes": {
                    "zero": NO_REASON_CODE,
                    "registry": "run_manifest.reason_code_map",
                    "registry_requirement": "exact_map_and_digest_required",
                },
            },
            "invariants": {
                "artifact_mutability": "immutable_snapshot",
                "edit_model": "external_sparse_delta_then_new_compacted_snapshot",
                "row_order": "frame_indices_then_refined_row_ids",
                "row_identity": "instance_key",
                "artifact_row_identity": "refined_row_ids",
                "frame_lookup": "frame_row_offsets_csr",
                "instances_per_frame": "zero_one_or_many",
                "missing_instance_representation": "absent_row",
                "geometry_authority": "bbox_norm_coords",
                "geometry_projections": ["bbox_img_xyxy", "centers_img_xy"],
                "continuous_geometry_dtype": "float32",
                "score_missing_encoding": {
                    "score_valid": False,
                    "scores": 0.0,
                },
                "manual_source_row_encoding": NO_SOURCE_ROW,
                "unaccepted_resolution_encoding": NO_RESOLVED_REFINED_ROW,
                "source_audit_table": "source_detections",
                "source_audit_order": "source_detect_row_index_contiguous",
                "accepted_source_cardinality": (
                    "one_to_one_with_raw_backed_refined_instances"
                ),
                "raw_edit_semantics": (
                    "manual_edit_flags_false_preserves_source_bbox_and_class"
                ),
                "frame_counts": "derived_not_persisted",
                "variable_length_text": "excluded_from_canonical_arrays",
                "publication_completeness": (
                    "all_values_direct_metadata_and_consolidated_metadata_"
                    "validated_before_selector_visibility"
                ),
            },
        }


REFINED_DETECTION_SCHEMA_V1 = RefinedDetectionSchema(
    schema_id=REFINED_DETECTION_SCHEMA_ID,
    schema_version=REFINED_DETECTION_SCHEMA_VERSION,
    core_bindings=REFINED_DETECTION_CORE_BINDINGS,
    clipped_lineage_bindings=REFINED_DETECTION_CLIPPED_LINEAGE_BINDINGS,
    contracts=REFINED_DETECTION_ARRAY_CONTRACTS,
)


__all__ = [
    "FORBIDDEN_REFINED_DETECTION_PATHS",
    "NO_REASON_CODE",
    "NO_RESOLVED_REFINED_ROW",
    "NO_SOURCE_ROW",
    "REFINED_DETECTION_CLIPPED_LINEAGE_BINDINGS",
    "REFINED_DETECTION_CORE_BINDINGS",
    "REFINED_DETECTION_INSTANCE_GROUP",
    "REFINED_DETECTION_LAYOUT",
    "REFINED_DETECTION_SCHEMA_ID",
    "REFINED_DETECTION_SCHEMA_V1",
    "REFINED_DETECTION_SCHEMA_VERSION",
    "REFINED_DETECTION_SOURCE_GROUP",
    "SOURCE_DECISION_CODE_MAP",
    "SOURCE_KIND_CODE_MAP",
    "RefinedDetectionDimensions",
    "RefinedDetectionLineageProfile",
    "RefinedDetectionSchema",
    "RefinedDetectionSchemaError",
    "RefinedDetectionSchemaIssue",
]
