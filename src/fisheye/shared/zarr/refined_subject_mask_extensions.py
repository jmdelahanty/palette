"""Editable-draft and published-cache extensions for refined subject masks.

The dense scientific core is defined in :mod:`subject_mask_schema`.  This
module deliberately keeps mutable review state separate from immutable
published cache evidence:

* draft audit arrays support row/component compare-and-swap editing;
* cache receipts prove one non-authoritative cache was regenerated and fully
  validated against one exact published dense authority.

Nothing in this module creates Zarr arrays or changes selector state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.refined_subject_component_contours import (
    DEFAULT_BOUNDARY_POLICY,
    DEFAULT_CONTOUR_COORDINATE_SPACE,
    DEFAULT_CONTOUR_METHOD,
    DEFAULT_CONTOUR_METHOD_VERSION,
    DEFAULT_SAMPLED_CONTOUR_COUNTS,
    SAMPLED_COMPONENT_CONTOUR_CANONICALIZATION,
    SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_ID,
    SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_VERSION,
)

from fisheye.shared.zarr.array_contracts import (
    REFINED_SUBJECT_MASK_COMPONENT_EDIT_APPLIED_V1,
    REFINED_SUBJECT_MASK_EDIT_APPLIED_V1,
    REFINED_SUBJECT_MASK_MANUAL_OVERRIDE_V1,
    REFINED_SUBJECT_MASK_ROW_REVISION_V1,
    REFINED_SUBJECT_MASK_ROW_UPDATED_AT_UTC_BYTES_V1,
    REFINED_SUBJECT_MASK_ROW_UPDATE_REASON_BYTES_V1,
    SUBJECT_MASK_BITPACKED_V1,
    SUBJECT_MASK_CONTOUR_LEN_V1,
    SUBJECT_MASK_CONTOUR_POINTS_XY_V1,
    SUBJECT_MASK_CONTOUR_PTR_V1,
    SUBJECT_MASK_RLE_AREA_PX_V1,
    SUBJECT_MASK_RLE_BBOX_XYXY_V1,
    SUBJECT_MASK_RLE_COUNTS_V1,
    SUBJECT_MASK_RLE_INDPTR_V1,
    SUBJECT_MASK_RLE_PRESENT_V1,
    SUBJECT_MASK_SAMPLED_CONTOUR_POINTS_XY_V1,
    SUBJECT_MASK_SAMPLED_CONTOUR_SOURCE_POINT_COUNT_V1,
    SUBJECT_MASK_SAMPLED_CONTOUR_VALID_V1,
    ArrayContract,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_schema import (
    REFINED_SUBJECT_MASK_CORE_SCHEMA_ID,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_VERSION,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    SubjectMaskSchemaError,
    SubjectMaskSchemaIssue,
)

REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_ID = (
    "palette.stage.refined_subject_mask.editable_draft_audit"
)
REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_VERSION = 1
REFINED_SUBJECT_MASK_DRAFT_AUDIT_MANIFEST_ATTRIBUTE = "editable_draft_audit_manifest"
REFINED_SUBJECT_MASK_CACHE_EXTENSION_SCHEMA_ID = (
    "palette.stage.refined_subject_mask.published_cache_extension"
)
REFINED_SUBJECT_MASK_CACHE_EXTENSION_SCHEMA_VERSION = 1
SUBJECT_MASK_CACHE_RECEIPT_SCHEMA_ID = (
    "palette.refined_subject_mask.derived_cache_receipt"
)
SUBJECT_MASK_CACHE_RECEIPT_SCHEMA_VERSION = 1
SUBJECT_MASK_CACHE_RECEIPT_DIGEST_ALGORITHM = "sha256_canonical_json_v1"
SUBJECT_MASK_CACHE_VALIDATION_MODE = "full_dense_equivalence"
SUBJECT_MASK_CONTOUR_CACHE_PROFILE_SCHEMA_ID = (
    "palette.refined_subject_mask.contour_cache_profile"
)
SUBJECT_MASK_CONTOUR_CACHE_PROFILE_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SubjectMaskDerivedCacheKind(str, Enum):
    BITPACKED = "mask_bitpacked"
    RLE = "mask_rle"
    SAMPLED_CONTOURS = "sampled_contours"
    FULL_CONTOURS = "contours"


@dataclass(frozen=True)
class SubjectMaskSampledContourProfile:
    """Exact default sampled-contour cache semantics for one component set."""

    sample_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        normalized: dict[str, int] = {}
        for raw_label, raw_count in self.sample_counts.items():
            label = str(raw_label).strip()
            if not label:
                raise ValueError("Sampled-contour component labels cannot be empty.")
            if type(raw_count) is not int or raw_count <= 0:
                raise ValueError(
                    "Sampled-contour counts must be positive exact integers."
                )
            normalized[label] = raw_count
        if not normalized or len(normalized) != len(self.sample_counts):
            raise ValueError("Sampled-contour component labels must be unique.")
        object.__setattr__(
            self, "sample_counts", MappingProxyType(dict(sorted(normalized.items())))
        )

    def require_components(self, components: SubjectMaskComponentRegistry) -> None:
        if set(self.sample_counts) != set(components.labels):
            raise ValueError(
                "Sampled-contour counts must exactly cover the component registry."
            )

    def as_manifest(
        self, *, components: SubjectMaskComponentRegistry
    ) -> dict[str, object]:
        self.require_components(components)
        return {
            "schema_id": SUBJECT_MASK_CONTOUR_CACHE_PROFILE_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_CONTOUR_CACHE_PROFILE_SCHEMA_VERSION,
            "profile_id": SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_ID,
            "profile_version": (SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_VERSION),
            "authority": "derived_from_dense_masks_roi",
            "default_cache": {
                "kind": SubjectMaskDerivedCacheKind.SAMPLED_CONTOURS.value,
                "required_for_profile": True,
                "component_sample_counts": {
                    label: self.sample_counts[label] for label in components.labels
                },
                "source_contour_method": DEFAULT_CONTOUR_METHOD,
                "source_contour_method_version": DEFAULT_CONTOUR_METHOD_VERSION,
                "boundary_policy": DEFAULT_BOUNDARY_POLICY,
                "coordinate_space": DEFAULT_CONTOUR_COORDINATE_SPACE,
                "point_order": "xy",
                "sampling_method": "closed_arc_length_uniform",
                "sampling_method_version": 2,
                "point_canonicalization": (SAMPLED_COMPONENT_CONTOUR_CANONICALIZATION),
                "winding": "clockwise_in_roi_y_down",
                "start_point": "topmost_then_leftmost_vertex",
                "duplicate_closing_point": False,
                "invalid_row_encoding": "all_nan_points_valid_false",
            },
            "full_contours": {
                "kind": SubjectMaskDerivedCacheKind.FULL_CONTOURS.value,
                "required_for_profile": False,
                "role": "optional_cold_inspection_or_export_cache",
            },
            "freshness": "receipt_bound_full_dense_equivalence",
        }


def default_subject_mask_sampled_contour_profile(
    components: SubjectMaskComponentRegistry,
) -> SubjectMaskSampledContourProfile:
    """Return the exact default profile or fail for an undeclared component."""

    profile = SubjectMaskSampledContourProfile(
        {
            label: DEFAULT_SAMPLED_CONTOUR_COUNTS[label]
            for label in components.labels
            if label in DEFAULT_SAMPLED_CONTOUR_COUNTS
        }
    )
    profile.require_components(components)
    return profile


def _cache_path_matches_kind(
    kind: SubjectMaskDerivedCacheKind,
    path: str,
) -> bool:
    parts = tuple(part for part in str(path).strip("/").split("/") if part)
    if kind is SubjectMaskDerivedCacheKind.BITPACKED:
        return parts == ("mask_bitpacked",)
    if kind is SubjectMaskDerivedCacheKind.RLE:
        return parts == ("mask_rle",) or (
            len(parts) == 3 and parts[:2] == ("mask_rle", "components")
        )
    if kind is SubjectMaskDerivedCacheKind.SAMPLED_CONTOURS:
        return (
            len(parts) == 3
            and parts[0] == "components"
            and parts[2] == "sampled_contours"
        )
    return len(parts) == 3 and parts[0] == "components" and parts[2] == "contours"


def _issue(code: str, path: str, message: str) -> SubjectMaskSchemaIssue:
    return SubjectMaskSchemaIssue(code=code, path=path, message=message)


def _materialize(array: Any) -> np.ndarray:
    try:
        return np.asarray(array[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(array)


def _require_contract(
    *,
    path: str,
    array: Any,
    contract: ArrayContract,
    dimensions: Mapping[str, int],
) -> list[SubjectMaskSchemaIssue]:
    try:
        errors = contract.validate_observation(array, dimensions=dimensions)
    except Exception as exc:
        errors = (f"array metadata is unreadable: {exc}",)
    return [_issue("array_contract_violation", path, error) for error in errors]


def _decode_fixed_utf8_rows(
    values: np.ndarray,
    *,
    path: str,
) -> tuple[list[str], list[SubjectMaskSchemaIssue]]:
    decoded: list[str] = []
    issues: list[SubjectMaskSchemaIssue] = []
    for row_index, row in enumerate(np.asarray(values, dtype=np.uint8)):
        raw = bytes(row.tolist())
        terminator = raw.find(b"\x00")
        payload = raw if terminator < 0 else raw[:terminator]
        if terminator >= 0 and any(raw[terminator:]):
            issues.append(
                _issue(
                    "invalid_nul_padding",
                    path,
                    f"Row {row_index} has nonzero bytes after its first NUL.",
                )
            )
        try:
            decoded.append(payload.decode("utf-8", errors="strict"))
        except UnicodeDecodeError:
            decoded.append("")
            issues.append(
                _issue(
                    "invalid_utf8",
                    path,
                    f"Row {row_index} is not valid UTF-8.",
                )
            )
    return decoded, issues


def _is_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None
        and parsed.utcoffset().total_seconds() == 0
    )


@dataclass(frozen=True)
class RefinedSubjectMaskDraftAuditSchema:
    """Exact mutable audit arrays for an editable dense-mask draft."""

    schema_id: str = REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_ID
    schema_version: int = REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_VERSION

    def binding_paths(
        self,
        components: SubjectMaskComponentRegistry,
    ) -> tuple[str, ...]:
        paths = ["edit_applied"]
        for label in components.labels:
            prefix = f"components/{label}"
            paths.extend(
                (
                    f"{prefix}/edit_applied",
                    f"{prefix}/manual_override",
                    f"{prefix}/row_revision",
                    f"{prefix}/row_updated_at_utc_bytes",
                    f"{prefix}/row_update_reason_bytes",
                )
            )
        return tuple(paths)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: SubjectMaskDimensions,
        components: SubjectMaskComponentRegistry,
    ) -> tuple[SubjectMaskSchemaIssue, ...]:
        components.require_dimensions(dimensions)
        expected = set(self.binding_paths(components))
        observed = set(arrays)
        issues: list[SubjectMaskSchemaIssue] = []
        for path in sorted(expected - observed):
            issues.append(
                _issue("missing_required_array", path, "Draft audit array is absent.")
            )
        for path in sorted(observed - expected):
            issues.append(
                _issue(
                    "unexpected_array",
                    path,
                    "Array is outside the exact draft-audit extension.",
                )
            )
        if issues:
            return tuple(issues)

        contract_dimensions = dimensions.contract_dimensions
        root_contract_issues = _require_contract(
            path="edit_applied",
            array=arrays["edit_applied"],
            contract=REFINED_SUBJECT_MASK_EDIT_APPLIED_V1,
            dimensions=contract_dimensions,
        )
        issues.extend(root_contract_issues)
        root_edit = _materialize(arrays["edit_applied"])
        for component_index, label in enumerate(components.labels):
            prefix = f"components/{label}"
            contracts = {
                f"{prefix}/edit_applied": REFINED_SUBJECT_MASK_COMPONENT_EDIT_APPLIED_V1,
                f"{prefix}/manual_override": REFINED_SUBJECT_MASK_MANUAL_OVERRIDE_V1,
                f"{prefix}/row_revision": REFINED_SUBJECT_MASK_ROW_REVISION_V1,
                f"{prefix}/row_updated_at_utc_bytes": REFINED_SUBJECT_MASK_ROW_UPDATED_AT_UTC_BYTES_V1,
                f"{prefix}/row_update_reason_bytes": REFINED_SUBJECT_MASK_ROW_UPDATE_REASON_BYTES_V1,
            }
            component_contract_failed = False
            for path, contract in contracts.items():
                contract_issues = _require_contract(
                    path=path,
                    array=arrays[path],
                    contract=contract,
                    dimensions=contract_dimensions,
                )
                component_contract_failed |= bool(contract_issues)
                issues.extend(contract_issues)
            if component_contract_failed or root_contract_issues:
                continue

            component_edit = _materialize(arrays[f"{prefix}/edit_applied"])
            manual = _materialize(arrays[f"{prefix}/manual_override"])
            revisions = _materialize(arrays[f"{prefix}/row_revision"])
            timestamps, timestamp_issues = _decode_fixed_utf8_rows(
                _materialize(arrays[f"{prefix}/row_updated_at_utc_bytes"]),
                path=f"{prefix}/row_updated_at_utc_bytes",
            )
            reasons, reason_issues = _decode_fixed_utf8_rows(
                _materialize(arrays[f"{prefix}/row_update_reason_bytes"]),
                path=f"{prefix}/row_update_reason_bytes",
            )
            issues.extend(timestamp_issues)
            issues.extend(reason_issues)
            if not np.array_equal(component_edit, root_edit[:, component_index]):
                issues.append(
                    _issue(
                        "component_edit_mirror_mismatch",
                        f"{prefix}/edit_applied",
                        "Component edit state must exactly mirror root edit_applied.",
                    )
                )
            if not np.array_equal(manual, component_edit):
                issues.append(
                    _issue(
                        "manual_override_mismatch",
                        f"{prefix}/manual_override",
                        "Draft v1 manual_override must exactly match edit_applied.",
                    )
                )
            if np.any(revisions < 0):
                issues.append(
                    _issue(
                        "negative_row_revision",
                        f"{prefix}/row_revision",
                        "Row revisions must be nonnegative.",
                    )
                )
            for row_index, revision in enumerate(revisions.tolist()):
                timestamp = timestamps[row_index]
                reason = reasons[row_index]
                if int(revision) == 0:
                    if timestamp or reason:
                        issues.append(
                            _issue(
                                "unrevised_row_has_audit_text",
                                prefix,
                                f"Row {row_index} has revision zero but nonempty audit text.",
                            )
                        )
                else:
                    if not timestamp or not _is_utc_timestamp(timestamp):
                        issues.append(
                            _issue(
                                "invalid_row_revision_timestamp",
                                f"{prefix}/row_updated_at_utc_bytes",
                                f"Revised row {row_index} lacks an exact UTC timestamp.",
                            )
                        )
                    if not reason:
                        issues.append(
                            _issue(
                                "missing_row_revision_reason",
                                f"{prefix}/row_update_reason_bytes",
                                f"Revised row {row_index} lacks an audit reason.",
                            )
                        )
        return tuple(issues)

    def require(self, arrays: Mapping[str, Any], **kwargs: Any) -> None:
        issues = self.validate(arrays, **kwargs)
        if issues:
            raise SubjectMaskSchemaError(issues)

    def as_manifest(
        self,
        *,
        dimensions: SubjectMaskDimensions,
        components: SubjectMaskComponentRegistry,
    ) -> dict[str, object]:
        components.require_dimensions(dimensions)
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "surface_role": "mutable_edit_audit_not_scientific_identity",
            "selector_eligible": False,
            "dimensions": dimensions.as_manifest(),
            "components": components.as_manifest(),
            "paths": list(self.binding_paths(components)),
            "write_contract": {
                "granularity": "one_roi_one_component",
                "concurrency": "compare_and_swap_row_revision",
                "revision": "monotonic_int64_increment_on_every_authority_write",
                "publication": "compacts_to_new_immutable_refined_snapshot",
            },
        }


REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_V1 = RefinedSubjectMaskDraftAuditSchema()


def _require_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest.")
    return normalized


def _require_utc_timestamp(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized or not _is_utc_timestamp(normalized):
        raise ValueError(f"{name} must be a nonempty ISO-8601 UTC timestamp.")
    return normalized


@dataclass(frozen=True)
class SubjectMaskDerivedCacheReceipt:
    """Digest-bound proof that one published cache matches dense authority."""

    cache_kind: SubjectMaskDerivedCacheKind
    cache_path: str
    source_dense_core_manifest_digest: str
    source_dense_array_values_sha256: str
    component_registry_digest: str
    logical_content_digest: str
    generator_id: str
    generator_version: int
    generated_at_utc: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cache_kind", SubjectMaskDerivedCacheKind(self.cache_kind)
        )
        cache_path = str(self.cache_path).strip().strip("/")
        if not cache_path or ".." in cache_path.split("/"):
            raise ValueError("cache_path must be a nonempty relative archive path.")
        if not _cache_path_matches_kind(self.cache_kind, cache_path):
            raise ValueError(
                f"cache_path {cache_path!r} does not match cache kind "
                f"{self.cache_kind.value!r}."
            )
        object.__setattr__(self, "cache_path", cache_path)
        for name in (
            "source_dense_core_manifest_digest",
            "source_dense_array_values_sha256",
            "component_registry_digest",
            "logical_content_digest",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        generator_id = str(self.generator_id).strip()
        if not generator_id:
            raise ValueError("generator_id cannot be empty.")
        object.__setattr__(self, "generator_id", generator_id)
        if type(self.generator_version) is not int or self.generator_version <= 0:
            raise ValueError("generator_version must be a positive exact integer.")
        object.__setattr__(
            self,
            "generated_at_utc",
            _require_utc_timestamp(self.generated_at_utc, name="generated_at_utc"),
        )

    def payload(self) -> dict[str, object]:
        return {
            "cache_kind": self.cache_kind.value,
            "cache_path": self.cache_path,
            "source": {
                "schema_id": REFINED_SUBJECT_MASK_CORE_SCHEMA_ID,
                "schema_version": REFINED_SUBJECT_MASK_CORE_SCHEMA_VERSION,
                "dense_core_manifest_digest": self.source_dense_core_manifest_digest,
                "dense_array_values_sha256": self.source_dense_array_values_sha256,
                "component_registry_digest": self.component_registry_digest,
            },
            "logical_content_digest": self.logical_content_digest,
            "generator": {
                "id": self.generator_id,
                "version": self.generator_version,
            },
            "generated_at_utc": self.generated_at_utc,
            "validation": {
                "mode": SUBJECT_MASK_CACHE_VALIDATION_MODE,
                "status": "passed",
            },
            "stale": False,
            "authoritative_pixels": False,
        }

    def as_manifest(self) -> dict[str, object]:
        payload = self.payload()
        return {
            "schema_id": SUBJECT_MASK_CACHE_RECEIPT_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_CACHE_RECEIPT_SCHEMA_VERSION,
            "digest_algorithm": SUBJECT_MASK_CACHE_RECEIPT_DIGEST_ALGORITHM,
            "payload": payload,
            "payload_digest": canonical_json_sha256(payload),
        }


def validate_subject_mask_cache_receipt(
    document: Mapping[str, Any],
) -> tuple[str, ...]:
    """Deeply validate an untrusted persisted cache receipt."""

    errors: list[str] = []
    expected_envelope = {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload",
        "payload_digest",
    }
    if set(document) != expected_envelope:
        errors.append("cache receipt envelope has an unexpected field set")
    if (
        document.get("schema_id") != SUBJECT_MASK_CACHE_RECEIPT_SCHEMA_ID
        or document.get("schema_version") != SUBJECT_MASK_CACHE_RECEIPT_SCHEMA_VERSION
        or document.get("digest_algorithm")
        != SUBJECT_MASK_CACHE_RECEIPT_DIGEST_ALGORITHM
    ):
        errors.append("cache receipt schema header mismatch")
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "cache receipt payload must be an object")
    expected_payload = {
        "cache_kind",
        "cache_path",
        "source",
        "logical_content_digest",
        "generator",
        "generated_at_utc",
        "validation",
        "stale",
        "authoritative_pixels",
    }
    if set(payload) != expected_payload:
        errors.append("cache receipt payload has an unexpected field set")
    cache_kind: SubjectMaskDerivedCacheKind | None = None
    try:
        cache_kind = SubjectMaskDerivedCacheKind(payload.get("cache_kind"))
    except (TypeError, ValueError):
        errors.append("cache receipt cache_kind is unknown")
    cache_path = payload.get("cache_path")
    if (
        not isinstance(cache_path, str)
        or not cache_path.strip().strip("/")
        or ".." in cache_path.strip().strip("/").split("/")
    ):
        errors.append("cache receipt cache_path is invalid")
    elif cache_kind is not None and not _cache_path_matches_kind(
        cache_kind,
        cache_path,
    ):
        errors.append("cache receipt path does not match cache_kind")
    source = payload.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "schema_id",
        "schema_version",
        "dense_core_manifest_digest",
        "dense_array_values_sha256",
        "component_registry_digest",
    }:
        errors.append("cache receipt source has an unexpected field set")
    else:
        if (
            source.get("schema_id") != REFINED_SUBJECT_MASK_CORE_SCHEMA_ID
            or source.get("schema_version") != REFINED_SUBJECT_MASK_CORE_SCHEMA_VERSION
        ):
            errors.append("cache receipt source schema mismatch")
        for field in (
            "dense_core_manifest_digest",
            "dense_array_values_sha256",
            "component_registry_digest",
        ):
            try:
                _require_sha256(source.get(field), name=f"source.{field}")
            except ValueError as exc:
                errors.append(str(exc))
    try:
        _require_sha256(
            payload.get("logical_content_digest"), name="logical_content_digest"
        )
    except ValueError as exc:
        errors.append(str(exc))
    generator = payload.get("generator")
    if (
        not isinstance(generator, Mapping)
        or set(generator) != {"id", "version"}
        or not isinstance(generator.get("id"), str)
        or not str(generator.get("id")).strip()
        or type(generator.get("version")) is not int
        or int(generator.get("version", 0)) <= 0
    ):
        errors.append("cache receipt generator is invalid")
    try:
        _require_utc_timestamp(payload.get("generated_at_utc"), name="generated_at_utc")
    except ValueError as exc:
        errors.append(str(exc))
    if payload.get("validation") != {
        "mode": SUBJECT_MASK_CACHE_VALIDATION_MODE,
        "status": "passed",
    }:
        errors.append("cache receipt does not prove full dense equivalence")
    if payload.get("stale") is not False:
        errors.append("published cache receipt must declare stale=false")
    if payload.get("authoritative_pixels") is not False:
        errors.append("published cache receipt must declare authoritative_pixels=false")
    try:
        expected_digest = canonical_json_sha256(payload)
    except Exception as exc:
        errors.append(f"cache receipt payload is not canonical JSON: {exc}")
    else:
        if document.get("payload_digest") != expected_digest:
            errors.append("cache receipt payload digest mismatch")
    return tuple(errors)


def _validate_exact_cache_fields(
    arrays: Mapping[str, Any],
    *,
    expected: set[str],
) -> list[SubjectMaskSchemaIssue]:
    issues: list[SubjectMaskSchemaIssue] = []
    for path in sorted(expected - set(arrays)):
        issues.append(_issue("missing_required_array", path, "Cache array is absent."))
    for path in sorted(set(arrays) - expected):
        issues.append(
            _issue("unexpected_array", path, "Array is outside this cache schema.")
        )
    return issues


def validate_subject_mask_cache_arrays(
    kind: SubjectMaskDerivedCacheKind | str,
    arrays: Mapping[str, Any],
    *,
    dimensions: SubjectMaskDimensions,
    sample_count: int | None = None,
) -> tuple[SubjectMaskSchemaIssue, ...]:
    """Validate exact logical arrays and canonical packing for one cache."""

    cache_kind = SubjectMaskDerivedCacheKind(kind)
    issues: list[SubjectMaskSchemaIssue] = []
    dims = dict(dimensions.contract_dimensions)

    if cache_kind is SubjectMaskDerivedCacheKind.BITPACKED:
        expected = {"masks_packed"}
        issues.extend(_validate_exact_cache_fields(arrays, expected=expected))
        if issues:
            return tuple(issues)
        dims["packed_W"] = (dimensions.roi_width + 7) // 8
        issues.extend(
            _require_contract(
                path="masks_packed",
                array=arrays["masks_packed"],
                contract=SUBJECT_MASK_BITPACKED_V1,
                dimensions=dims,
            )
        )
        return tuple(issues)

    if cache_kind is SubjectMaskDerivedCacheKind.RLE:
        expected = {"counts", "indptr", "present", "area_px", "bbox_xyxy"}
        issues.extend(_validate_exact_cache_fields(arrays, expected=expected))
        if issues:
            return tuple(issues)
        dims["n_rle_counts"] = int(arrays["counts"].shape[0])
        dims["n_rle_boundaries"] = dimensions.n_rois + 1
        contracts = {
            "counts": SUBJECT_MASK_RLE_COUNTS_V1,
            "indptr": SUBJECT_MASK_RLE_INDPTR_V1,
            "present": SUBJECT_MASK_RLE_PRESENT_V1,
            "area_px": SUBJECT_MASK_RLE_AREA_PX_V1,
            "bbox_xyxy": SUBJECT_MASK_RLE_BBOX_XYXY_V1,
        }
        for path, contract in contracts.items():
            issues.extend(
                _require_contract(
                    path=path, array=arrays[path], contract=contract, dimensions=dims
                )
            )
        if issues:
            return tuple(issues)
        counts = _materialize(arrays["counts"])
        indptr = _materialize(arrays["indptr"])
        present = _materialize(arrays["present"])
        area = _materialize(arrays["area_px"])
        bbox = _materialize(arrays["bbox_xyxy"])
        offsets_valid = bool(
            indptr[0] == 0
            and not np.any(np.diff(indptr) <= 0)
            and indptr[-1] == counts.size
        )
        if not offsets_valid:
            issues.append(
                _issue(
                    "noncanonical_rle_offsets",
                    "indptr",
                    "RLE offsets must start at zero, strictly increase, and end at counts length.",
                )
            )
        else:
            row_sums = np.add.reduceat(
                np.asarray(counts, dtype=np.int64),
                np.asarray(indptr[:-1], dtype=np.int64),
            )
            expected_pixels = dimensions.roi_height * dimensions.roi_width
            if np.any(row_sums != expected_pixels):
                issues.append(
                    _issue(
                        "invalid_rle_pixel_count",
                        "counts",
                        "Every RLE row must decode to the complete ROI pixel count.",
                    )
                )
        if np.any(area < 0) or np.any(
            area > dimensions.roi_height * dimensions.roi_width
        ):
            issues.append(
                _issue(
                    "invalid_rle_area", "area_px", "RLE area is outside the ROI domain."
                )
            )
        if not np.array_equal(present, area > 0):
            issues.append(
                _issue(
                    "rle_presence_mismatch",
                    "present",
                    "present must equal area_px > 0.",
                )
            )
        if np.any(bbox < 0):
            issues.append(
                _issue("invalid_rle_bbox", "bbox_xyxy", "RLE boxes cannot be negative.")
            )
        if np.any(bbox[:, (0, 2)] > dimensions.roi_width) or np.any(
            bbox[:, (1, 3)] > dimensions.roi_height
        ):
            issues.append(
                _issue(
                    "invalid_rle_bbox", "bbox_xyxy", "RLE boxes exceed the ROI domain."
                )
            )
        if np.any(bbox[~present] != 0):
            issues.append(
                _issue(
                    "invalid_empty_rle_bbox",
                    "bbox_xyxy",
                    "Absent rows require zero boxes.",
                )
            )
        if np.any(present):
            present_boxes = bbox[present]
            if np.any(present_boxes[:, 2] <= present_boxes[:, 0]) or np.any(
                present_boxes[:, 3] <= present_boxes[:, 1]
            ):
                issues.append(
                    _issue(
                        "invalid_present_rle_bbox",
                        "bbox_xyxy",
                        "Present rows require positive-area half-open boxes.",
                    )
                )
        return tuple(issues)

    if cache_kind is SubjectMaskDerivedCacheKind.SAMPLED_CONTOURS:
        expected = {"points_xy", "valid", "source_point_count"}
        issues.extend(_validate_exact_cache_fields(arrays, expected=expected))
        if issues:
            return tuple(issues)
        if type(sample_count) is not int or sample_count <= 0:
            return (
                _issue(
                    "invalid_sample_count",
                    "points_xy",
                    "sample_count must be positive.",
                ),
            )
        dims["n_samples"] = sample_count
        contracts = {
            "points_xy": SUBJECT_MASK_SAMPLED_CONTOUR_POINTS_XY_V1,
            "valid": SUBJECT_MASK_SAMPLED_CONTOUR_VALID_V1,
            "source_point_count": SUBJECT_MASK_SAMPLED_CONTOUR_SOURCE_POINT_COUNT_V1,
        }
        for path, contract in contracts.items():
            issues.extend(
                _require_contract(
                    path=path, array=arrays[path], contract=contract, dimensions=dims
                )
            )
        if issues:
            return tuple(issues)
        points = _materialize(arrays["points_xy"])
        valid = _materialize(arrays["valid"])
        source_count = _materialize(arrays["source_point_count"])
        if np.any(source_count < 0):
            issues.append(
                _issue(
                    "negative_source_point_count",
                    "source_point_count",
                    "Counts cannot be negative.",
                )
            )
        if np.any(valid & ~np.isfinite(points).all(axis=(1, 2))):
            issues.append(
                _issue(
                    "invalid_sampled_points",
                    "points_xy",
                    "Valid rows require finite points.",
                )
            )
        if np.any(~valid & ~np.isnan(points).all(axis=(1, 2))):
            issues.append(
                _issue(
                    "noncanonical_invalid_samples",
                    "points_xy",
                    "Invalid rows require all-NaN points.",
                )
            )
        return tuple(issues)

    expected = {"ptr", "len", "points_xy"}
    issues.extend(_validate_exact_cache_fields(arrays, expected=expected))
    if issues:
        return tuple(issues)
    dims["n_contour_points"] = int(arrays["points_xy"].shape[0])
    contracts = {
        "ptr": SUBJECT_MASK_CONTOUR_PTR_V1,
        "len": SUBJECT_MASK_CONTOUR_LEN_V1,
        "points_xy": SUBJECT_MASK_CONTOUR_POINTS_XY_V1,
    }
    for path, contract in contracts.items():
        issues.extend(
            _require_contract(
                path=path, array=arrays[path], contract=contract, dimensions=dims
            )
        )
    if issues:
        return tuple(issues)
    ptr = _materialize(arrays["ptr"])
    length = _materialize(arrays["len"])
    points = _materialize(arrays["points_xy"])
    if np.any(length < 0):
        issues.append(
            _issue(
                "negative_contour_length", "len", "Contour lengths cannot be negative."
            )
        )
        return tuple(issues)
    expected_offset = 0
    for row_index, (offset, count) in enumerate(zip(ptr.tolist(), length.tolist())):
        if int(count) == 0:
            if int(offset) != -1:
                issues.append(
                    _issue(
                        "noncanonical_empty_contour",
                        "ptr",
                        f"Empty row {row_index} requires ptr=-1.",
                    )
                )
        else:
            if int(offset) != expected_offset:
                issues.append(
                    _issue(
                        "noncanonical_contour_packing",
                        "ptr",
                        f"Row {row_index} begins at {int(offset)}, expected {expected_offset}.",
                    )
                )
            expected_offset += int(count)
    if expected_offset != int(points.shape[0]):
        issues.append(
            _issue(
                "orphan_contour_points",
                "points_xy",
                "Published full contours must contain no placeholders or orphan points.",
            )
        )
    if not np.all(np.isfinite(points)):
        issues.append(
            _issue(
                "nonfinite_contour_points",
                "points_xy",
                "Full contour points must be finite.",
            )
        )
    return tuple(issues)


def published_subject_mask_cache_extension_manifest(
    receipts: tuple[SubjectMaskDerivedCacheReceipt, ...],
) -> dict[str, object]:
    """Build one closed-world cache extension from validated receipt objects."""

    if not receipts:
        raise ValueError("At least one published cache receipt is required.")
    keys = [(receipt.cache_kind.value, receipt.cache_path) for receipt in receipts]
    if len(keys) != len(set(keys)):
        raise ValueError("Published cache receipts must have unique kind/path pairs.")
    source_identities = {
        (
            receipt.source_dense_core_manifest_digest,
            receipt.source_dense_array_values_sha256,
            receipt.component_registry_digest,
        )
        for receipt in receipts
    }
    if len(source_identities) != 1:
        raise ValueError(
            "Every cache extension receipt must bind the same dense authority."
        )
    manifests = [receipt.as_manifest() for receipt in receipts]
    return {
        "schema_id": REFINED_SUBJECT_MASK_CACHE_EXTENSION_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_CACHE_EXTENSION_SCHEMA_VERSION,
        "authority": "non_authoritative_derived_caches",
        "freshness_rule": "only_full_regeneration_and_dense_equivalence_may_clear_stale",
        "receipts": manifests,
        "receipts_digest": canonical_json_sha256(manifests),
    }


def validate_published_subject_mask_cache_extension(
    document: Mapping[str, Any],
) -> tuple[str, ...]:
    """Deeply validate a persisted closed-world cache extension."""

    errors: list[str] = []
    expected_fields = {
        "schema_id",
        "schema_version",
        "authority",
        "freshness_rule",
        "receipts",
        "receipts_digest",
    }
    if set(document) != expected_fields:
        errors.append("cache extension has an unexpected field set")
    if (
        document.get("schema_id") != REFINED_SUBJECT_MASK_CACHE_EXTENSION_SCHEMA_ID
        or document.get("schema_version")
        != REFINED_SUBJECT_MASK_CACHE_EXTENSION_SCHEMA_VERSION
        or document.get("authority") != "non_authoritative_derived_caches"
        or document.get("freshness_rule")
        != "only_full_regeneration_and_dense_equivalence_may_clear_stale"
    ):
        errors.append("cache extension schema header mismatch")
    receipts = document.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        return (*errors, "cache extension receipts must be a nonempty array")
    identities: set[tuple[object, object, object]] = set()
    keys: list[tuple[object, object]] = []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            errors.append(f"cache extension receipt {index} must be an object")
            continue
        errors.extend(
            f"receipt {index}: {error}"
            for error in validate_subject_mask_cache_receipt(receipt)
        )
        payload = receipt.get("payload")
        if not isinstance(payload, Mapping):
            continue
        keys.append((payload.get("cache_kind"), payload.get("cache_path")))
        source = payload.get("source")
        if isinstance(source, Mapping):
            identities.add(
                (
                    source.get("dense_core_manifest_digest"),
                    source.get("dense_array_values_sha256"),
                    source.get("component_registry_digest"),
                )
            )
    if len(keys) != len(set(keys)):
        errors.append("cache extension contains duplicate kind/path pairs")
    if len(identities) != 1:
        errors.append("cache extension receipts bind different dense authorities")
    try:
        expected_digest = canonical_json_sha256(receipts)
    except Exception as exc:
        errors.append(f"cache extension receipts are not canonical JSON: {exc}")
    else:
        if document.get("receipts_digest") != expected_digest:
            errors.append("cache extension receipts digest mismatch")
    return tuple(errors)


__all__ = [
    "REFINED_SUBJECT_MASK_CACHE_EXTENSION_SCHEMA_ID",
    "REFINED_SUBJECT_MASK_CACHE_EXTENSION_SCHEMA_VERSION",
    "REFINED_SUBJECT_MASK_DRAFT_AUDIT_MANIFEST_ATTRIBUTE",
    "REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_ID",
    "REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_V1",
    "REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_VERSION",
    "SUBJECT_MASK_CACHE_RECEIPT_DIGEST_ALGORITHM",
    "SUBJECT_MASK_CACHE_RECEIPT_SCHEMA_ID",
    "SUBJECT_MASK_CACHE_RECEIPT_SCHEMA_VERSION",
    "SUBJECT_MASK_CACHE_VALIDATION_MODE",
    "RefinedSubjectMaskDraftAuditSchema",
    "SubjectMaskDerivedCacheKind",
    "SubjectMaskDerivedCacheReceipt",
    "published_subject_mask_cache_extension_manifest",
    "validate_published_subject_mask_cache_extension",
    "validate_subject_mask_cache_arrays",
    "validate_subject_mask_cache_receipt",
]
