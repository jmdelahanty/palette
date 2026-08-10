"""Deterministic scientific identity and immutable attempt lineage for masks."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID, uuid4

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_ID = "palette.subject_mask.scientific_identity"
SUBJECT_MASK_SCIENTIFIC_IDENTITY_LEGACY_SCHEMA_VERSION = 1
SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION = 2
SUBJECT_MASK_ATTEMPT_SCHEMA_ID = "palette.subject_mask.attempt"
SUBJECT_MASK_ATTEMPT_SCHEMA_VERSION = 1


def _optional_run_name(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or "/" in normalized:
        raise ValueError(f"{name} must be one nonempty run name when provided.")
    return normalized


def build_subject_mask_scientific_identity(
    *,
    stage_kind: str,
    model: Mapping[str, Any],
    crop: Mapping[str, Any],
    pixels: Mapping[str, Any],
    row_identity: Mapping[str, Any],
    inference_contract: Mapping[str, Any],
    schema_version: int | None = None,
) -> dict[str, object]:
    """Bind the scientific inputs independently of runtime and storage layout."""

    stage = str(stage_kind).strip()
    if stage not in {
        "raw_subject_mask",
        "refined_subject_mask",
        "subject_mask_quality",
    }:
        raise ValueError(f"Unsupported subject-mask scientific stage {stage!r}.")
    resolved_version = (
        SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION
        if schema_version is None
        and stage in {"raw_subject_mask", "refined_subject_mask"}
        else (
            SUBJECT_MASK_SCIENTIFIC_IDENTITY_LEGACY_SCHEMA_VERSION
            if schema_version is None
            else int(schema_version)
        )
    )
    if resolved_version not in {
        SUBJECT_MASK_SCIENTIFIC_IDENTITY_LEGACY_SCHEMA_VERSION,
        SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"Unsupported subject-mask scientific identity version {resolved_version}."
        )
    payload = {
        "stage_kind": stage,
        "model": dict(model),
        "crop": dict(crop),
        "pixels": dict(pixels),
        "row_identity": dict(row_identity),
        "inference_contract": dict(inference_contract),
    }
    canonical_json_bytes(payload)
    result: dict[str, object] = {
        "schema_id": SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_ID,
        "schema_version": resolved_version,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    errors = validate_subject_mask_scientific_identity(result)
    if errors:
        raise ValueError(
            "Invalid subject-mask scientific identity: " + "; ".join(errors)
        )
    return result


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nonempty_string(value: Any) -> bool:
    return type(value) is str and bool(value.strip()) and value == value.strip()


def _validate_manifest_reference(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload_digest",
    }:
        return [f"{name} fields are not exact"]
    errors: list[str] = []
    if not _is_nonempty_string(value.get("schema_id")):
        errors.append(f"{name} schema_id is invalid")
    if type(value.get("schema_version")) is not int or value["schema_version"] <= 0:
        errors.append(f"{name} schema_version is invalid")
    if not _is_sha256(value.get("payload_digest")):
        errors.append(f"{name} payload digest is invalid")
    return errors


def _validate_array_reference(
    value: Any,
    *,
    name: str,
    row_count: int,
    trailing_shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
) -> list[str]:
    if not isinstance(value, Mapping) or set(value) != {"shape", "dtype", "sha256"}:
        return [f"{name} array reference fields are not exact"]
    shape = value.get("shape")
    errors: list[str] = []
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        or shape[0] != row_count
        or (trailing_shape is not None and tuple(shape[1:]) != trailing_shape)
    ):
        errors.append(f"{name} array shape is invalid")
    if not _is_nonempty_string(value.get("dtype")) or (
        dtype is not None and value.get("dtype") != dtype
    ):
        errors.append(f"{name} array dtype is invalid")
    if not _is_sha256(value.get("sha256")):
        errors.append(f"{name} array digest is invalid")
    return errors


def _validate_row_identity(
    value: Any,
    *,
    required_arrays: Mapping[str, tuple[str, tuple[int, ...]]],
    allowed_arrays: set[str],
    optional_array_contracts: Mapping[str, tuple[str, tuple[int, ...]]] | None = None,
    non_row_aligned_arrays: set[str] | None = None,
) -> tuple[list[str], int | None]:
    if not isinstance(value, Mapping) or set(value) != {"row_count", "arrays"}:
        return ["scientific row_identity fields are not exact"], None
    row_count = value.get("row_count")
    arrays = value.get("arrays")
    if type(row_count) is not int or row_count <= 0:
        return ["scientific row_identity row_count is invalid"], None
    if not isinstance(arrays, Mapping):
        return ["scientific row_identity arrays are invalid"], row_count
    errors: list[str] = []
    if not set(required_arrays) <= set(arrays) or not set(arrays) <= allowed_arrays:
        errors.append("scientific row_identity array inventory is invalid")
    array_contracts = {
        **dict(optional_array_contracts or {}),
        **dict(required_arrays),
    }
    for name, record in arrays.items():
        expected = array_contracts.get(str(name))
        shape = record.get("shape") if isinstance(record, Mapping) else None
        expected_rows = (
            shape[0]
            if str(name) in (non_row_aligned_arrays or set())
            and isinstance(shape, list)
            and shape
            else row_count
        )
        errors.extend(
            _validate_array_reference(
                record,
                name=f"row_identity.{name}",
                row_count=expected_rows,
                dtype=expected[0] if expected is not None else None,
                trailing_shape=expected[1] if expected is not None else None,
            )
        )
    return errors, row_count


def _validate_model_input_transform(value: Any, *, roi_shape: list[int]) -> list[str]:
    fields = {
        "name",
        "native_shape_hw",
        "model_shape_hw",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
        "coordinate_mapping",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        return ["raw model_input_transform fields are not exact"]
    errors: list[str] = []
    native = value.get("native_shape_hw")
    model = value.get("model_shape_hw")
    pads = [
        value.get(name) for name in ("pad_top", "pad_bottom", "pad_left", "pad_right")
    ]
    if native != roi_shape or not isinstance(model, list) or len(model) != 2:
        errors.append("raw model_input_transform shapes are invalid")
    elif any(type(item) is not int or item <= 0 for item in model):
        errors.append("raw model_input_transform model shape is invalid")
    if any(type(item) is not int or item < 0 for item in pads):
        errors.append("raw model_input_transform padding is invalid")
    elif (
        isinstance(native, list)
        and isinstance(model, list)
        and len(native) == len(model) == 2
    ):
        if (
            native[0] + pads[0] + pads[1] != model[0]
            or native[1] + pads[2] + pads[3] != model[1]
        ):
            errors.append(
                "raw model_input_transform padding does not produce model shape"
            )
    if value.get("name") not in {"identity", "pad_to_size"}:
        errors.append("raw model_input_transform name is invalid")
    if value.get("coordinate_mapping") != "native_xy = model_xy - [pad_left, pad_top]":
        errors.append("raw model_input_transform coordinate mapping changed")
    return errors


def _validate_source_input_binding(value: Any, *, name: str) -> list[str]:
    fields = {
        "run_path",
        "run_manifest",
        "scientific_identity_digest",
        "worker_semantic_receipt_binding",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        return [f"{name} fields are not exact"]
    errors = _validate_manifest_reference(
        value.get("run_manifest"), name=f"{name}.run_manifest"
    )
    if not _is_nonempty_string(value.get("run_path")) or "/" not in value["run_path"]:
        errors.append(f"{name} run_path is invalid")
    digest = value.get("scientific_identity_digest")
    if digest is not None and not _is_sha256(digest):
        errors.append(f"{name} scientific identity digest is invalid")
    receipt = value.get("worker_semantic_receipt_binding")
    if receipt is not None:
        expected = {
            "schema_id",
            "schema_version",
            "payload_digest",
            "relative_path",
            "document_sha256",
            "storage",
        }
        if not isinstance(receipt, Mapping) or set(receipt) != expected:
            errors.append(f"{name} worker receipt binding fields are not exact")
        elif (
            not _is_nonempty_string(receipt.get("schema_id"))
            or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] <= 0
            or not _is_sha256(receipt.get("payload_digest"))
            or not _is_sha256(receipt.get("document_sha256"))
            or not _is_nonempty_string(receipt.get("relative_path"))
            or receipt.get("storage") != "strict_json_sidecar_v1"
        ):
            errors.append(f"{name} worker receipt binding is invalid")
    return errors


def _validate_collection_partition_contract(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        return ["raw collection partition contract fields are not exact"]
    payload = value.get("payload")
    if (
        value.get("schema_id") != "palette.subject_mask.complete_collection_partition"
        or value.get("schema_version") != 1
        or not isinstance(payload, Mapping)
        or value.get("payload_digest") != canonical_json_sha256(payload)
    ):
        return ["raw collection partition contract is unsupported or stale"]
    expected = {
        "role",
        "coverage_semantics",
        "work_package_id",
        "collection",
        "frame_window",
        "crop_rows",
        "validation",
    }
    if set(payload) != expected:
        return ["raw collection partition payload fields are not exact"]
    errors: list[str] = []
    collection = payload.get("collection")
    window = payload.get("frame_window")
    rows = payload.get("crop_rows")
    validation = payload.get("validation")
    if payload.get("role") != "complete_collection_partition":
        errors.append("raw collection partition role is invalid")
    if (
        payload.get("coverage_semantics")
        != "exact_complete_crop_rows_for_acquisition_frame_window_v1"
    ):
        errors.append("raw collection partition coverage semantics changed")
    if not _is_sha256(payload.get("work_package_id")):
        errors.append("raw collection partition work-package digest is invalid")
    collection_fields = {
        "source_collection_id",
        "source_collection_path",
        "source_clip_id",
        "source_clip_index",
        "source_work_unit_id",
        "source_shard_id",
    }
    if not isinstance(collection, Mapping) or set(collection) != collection_fields:
        errors.append("raw collection partition collection identity is invalid")
    else:
        for name in collection_fields - {"source_clip_index"}:
            if not _is_nonempty_string(collection.get(name)):
                errors.append(f"raw collection partition {name} is invalid")
        if (
            type(collection.get("source_clip_index")) is not int
            or collection["source_clip_index"] < 0
        ):
            errors.append("raw collection partition source_clip_index is invalid")
    if not isinstance(window, Mapping) or set(window) != {
        "schema_id",
        "schema_version",
        "recording_identity",
        "camera_identity",
        "clip_id",
        "actual_start_frame",
        "end_frame_exclusive",
        "frame_count",
        "clip_index_document_sha256",
        "clip_video_sha256",
    }:
        errors.append("raw collection partition frame window is invalid")
    elif (
        window.get("schema_id") != "palette.acquisition_video_frame_window"
        or window.get("schema_version") != 1
        or type(window.get("actual_start_frame")) is not int
        or type(window.get("end_frame_exclusive")) is not int
        or type(window.get("frame_count")) is not int
        or window["actual_start_frame"] < 0
        or window["frame_count"] <= 0
        or window["end_frame_exclusive"]
        != window["actual_start_frame"] + window["frame_count"]
        or not _is_nonempty_string(window.get("recording_identity"))
        or not _is_nonempty_string(window.get("camera_identity"))
        or not _is_nonempty_string(window.get("clip_id"))
        or not _is_sha256(window.get("clip_index_document_sha256"))
        or not _is_sha256(window.get("clip_video_sha256"))
    ):
        errors.append("raw collection partition frame window values are invalid")
    elif isinstance(collection, Mapping) and (
        window.get("clip_id") != collection.get("source_clip_id")
    ):
        errors.append("raw collection partition clip identities differ")
    if not isinstance(rows, Mapping) or set(rows) != {
        "start",
        "stop",
        "count",
        "source_crop_total_rows",
    }:
        errors.append("raw collection partition crop rows are invalid")
    elif (
        any(type(rows.get(name)) is not int for name in rows)
        or rows["start"] < 0
        or rows["stop"] <= rows["start"]
        or rows["count"] != rows["stop"] - rows["start"]
        or rows["source_crop_total_rows"] < rows["stop"]
    ):
        errors.append("raw collection partition crop-row values are invalid")
    expected_validation = {
        "work_package_opened_and_content_verified": True,
        "row_interval_contiguous": True,
        "frame_offset_coverage_exact": True,
        "acquisition_frames_within_window": True,
    }
    if validation != expected_validation:
        errors.append("raw collection partition validation claims are invalid")
    return errors


def _validate_raw_scientific_payload(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    model = payload.get("model")
    model_fields = {
        "artifact_role",
        "artifact_sha256",
        "artifact_size_bytes",
        "registry_set_id",
        "registry_run_id",
        "label_schema_id",
    }
    if not isinstance(model, Mapping) or set(model) != model_fields:
        errors.append("raw scientific model fields are not exact")
    else:
        if not _is_sha256(model.get("artifact_sha256")):
            errors.append("raw model artifact digest is invalid")
        if (
            type(model.get("artifact_size_bytes")) is not int
            or model["artifact_size_bytes"] <= 0
        ):
            errors.append("raw model artifact size is invalid")
        for name in ("artifact_role", "label_schema_id"):
            if not _is_nonempty_string(model.get(name)):
                errors.append(f"raw model {name} is invalid")
        for name in ("registry_set_id", "registry_run_id"):
            if model.get(name) is not None and not _is_nonempty_string(model.get(name)):
                errors.append(f"raw model {name} is invalid")

    crop = payload.get("crop")
    crop_fields = {
        "run_id",
        "run_group_path",
        "run_manifest",
        "storage_mode",
        "roi_shape_hw",
        "roi_coordinates_full",
        "source_collection_id",
        "source_clip_id",
        "source_clip_index",
        "source_work_unit_id",
        "source_shard_id",
        "collection_partition_contract",
    }
    row_count: int | None = None
    roi_shape: list[int] = []
    if not isinstance(crop, Mapping) or set(crop) != crop_fields:
        errors.append("raw scientific crop fields are not exact")
    else:
        for name in ("run_id", "run_group_path", "storage_mode"):
            if not _is_nonempty_string(crop.get(name)):
                errors.append(f"raw crop {name} is invalid")
        roi_shape_value = crop.get("roi_shape_hw")
        if (
            not isinstance(roi_shape_value, list)
            or len(roi_shape_value) != 2
            or any(type(item) is not int or item <= 0 for item in roi_shape_value)
        ):
            errors.append("raw crop ROI shape is invalid")
        else:
            roi_shape = list(roi_shape_value)
        coordinate_record = crop.get("roi_coordinates_full")
        coordinate_shape = (
            coordinate_record.get("shape")
            if isinstance(coordinate_record, Mapping)
            else None
        )
        if isinstance(coordinate_shape, list) and coordinate_shape:
            row_count = coordinate_shape[0]
        errors.extend(
            _validate_array_reference(
                coordinate_record,
                name="raw crop roi_coordinates_full",
                row_count=row_count if type(row_count) is int else -1,
                trailing_shape=(2,),
                dtype="int32",
            )
        )
        errors.extend(
            _validate_manifest_reference(
                crop.get("run_manifest"), name="raw crop run_manifest"
            )
        )
        errors.extend(
            _validate_collection_partition_contract(
                crop.get("collection_partition_contract")
            )
        )
        for name in (
            "source_collection_id",
            "source_clip_id",
            "source_work_unit_id",
            "source_shard_id",
        ):
            if crop.get(name) is not None and not _is_nonempty_string(crop.get(name)):
                errors.append(f"raw crop {name} is invalid")
        if crop.get("source_clip_index") is not None and (
            type(crop.get("source_clip_index")) is not int
            or crop["source_clip_index"] < 0
        ):
            errors.append("raw crop source_clip_index is invalid")

    row_errors, identity_rows = _validate_row_identity(
        payload.get("row_identity"),
        required_arrays={
            "source_crop_row_ids": ("int64", ()),
            "instance_key": ("uint64", ()),
            "source_acquisition_frame_index": ("int64", ()),
        },
        allowed_arrays={
            "source_crop_row_ids",
            "instance_key",
            "source_acquisition_frame_index",
            "source_crop_xywh",
            "frame_indices",
            "source_frame_indices",
            "source_clip_indices",
            "source_clip_local_frame_indices",
            "source_refined_row_ids",
            "source_detect_row_index",
        },
        optional_array_contracts={
            # Historical v2 worker identities omitted placement because the
            # crop block separately bound integer origins and fixed ROI shape.
            # New shard writers include the exact canonical float32 placement
            # so recording-level refined publication can bind the physical
            # source array without invalidating those historical identities.
            "source_crop_xywh": ("float32", (4,)),
        },
    )
    errors.extend(row_errors)
    if (
        row_count is not None
        and identity_rows is not None
        and row_count != identity_rows
    ):
        errors.append("raw crop and row-identity row counts differ")

    pixels = payload.get("pixels")
    pixel_fields = {
        "profile",
        "decoded_shape",
        "decoded_dtype",
        "decoded_order",
        "decoded_pixels_sha256",
        "declared_pixels_sha256",
        "cache_key",
        "pixel_materialization_id",
        "pixel_contract",
        "work_package_role",
    }
    if not isinstance(pixels, Mapping) or set(pixels) != pixel_fields:
        errors.append("raw scientific pixels fields are not exact")
    else:
        decoded_shape = pixels.get("decoded_shape")
        if (
            not isinstance(decoded_shape, list)
            or len(decoded_shape) != 3
            or any(type(item) is not int or item <= 0 for item in decoded_shape)
            or (identity_rows is not None and decoded_shape[0] != identity_rows)
            or (roi_shape and decoded_shape[1:] != roi_shape)
        ):
            errors.append("raw decoded pixel shape is invalid")
        if pixels.get("decoded_dtype") != "uint8" or pixels.get("decoded_order") != "C":
            errors.append("raw decoded pixel representation changed")
        if not _is_sha256(pixels.get("decoded_pixels_sha256")):
            errors.append("raw decoded pixel digest is invalid")
        declared = pixels.get("declared_pixels_sha256")
        if declared is not None and (
            not _is_sha256(declared) or declared != pixels.get("decoded_pixels_sha256")
        ):
            errors.append("raw declared pixel digest differs from decoded pixels")
        if not isinstance(pixels.get("pixel_contract"), Mapping):
            errors.append("raw pixel contract is absent")
        for name in ("profile",):
            if not _is_nonempty_string(pixels.get(name)):
                errors.append(f"raw pixels {name} is invalid")
        for name in ("cache_key", "pixel_materialization_id", "work_package_role"):
            if pixels.get(name) is not None and not _is_nonempty_string(
                pixels.get(name)
            ):
                errors.append(f"raw pixels {name} is invalid")

    inference = payload.get("inference_contract")
    inference_fields = {
        "segmenter",
        "label_schema_id",
        "mask_labels",
        "model_input_transform",
        "probability_semantics",
        "probability_dtype",
        "probability_encoding",
        "mask_probability_threshold",
        "overlap_policy",
    }
    if not isinstance(inference, Mapping) or set(inference) != inference_fields:
        errors.append("raw inference contract fields are not exact")
    else:
        if inference.get("segmenter") != "unet":
            errors.append("raw inference segmenter changed")
        labels = inference.get("mask_labels")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not _is_nonempty_string(item) for item in labels)
        ):
            errors.append("raw inference mask labels are invalid")
        if isinstance(model, Mapping) and inference.get("label_schema_id") != model.get(
            "label_schema_id"
        ):
            errors.append("raw inference label schema differs from the model")
        errors.extend(
            _validate_model_input_transform(
                inference.get("model_input_transform"), roi_shape=roi_shape
            )
        )
        if inference.get("probability_semantics") != "sigmoid_multilabel_logits":
            errors.append("raw probability semantics changed")
        if inference.get("probability_dtype") not in {"uint8", "float16"}:
            errors.append("raw probability dtype is invalid")
        expected_encoding = (
            "linear_uint8_0_255"
            if inference.get("probability_dtype") == "uint8"
            else "unit_float"
        )
        if inference.get("probability_encoding") != expected_encoding:
            errors.append("raw probability encoding is invalid")
        threshold = inference.get("mask_probability_threshold")
        if type(threshold) not in {int, float} or not 0.0 <= float(threshold) <= 1.0:
            errors.append("raw probability threshold is invalid")
        if inference.get("overlap_policy") != "independent_sigmoid":
            errors.append("raw overlap policy changed")
    return errors


def _validate_refined_scientific_payload(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    model = payload.get("model")
    if not isinstance(model, Mapping) or set(model) != {
        "role",
        "method",
        "source_input_binding",
    }:
        errors.append("refined scientific model fields are not exact")
        binding = None
    else:
        binding = model.get("source_input_binding")
        if model.get(
            "role"
        ) != "deterministic_refinement_policy" or not _is_nonempty_string(
            model.get("method")
        ):
            errors.append("refined model policy identity is invalid")
        errors.extend(
            _validate_source_input_binding(binding, name="refined source input binding")
        )

    crop = payload.get("crop")
    roi_shape: list[int] = []
    if not isinstance(crop, Mapping) or set(crop) != {
        "run_id",
        "source_crop_snapshot",
        "roi_shape_hw",
    }:
        errors.append("refined scientific crop fields are not exact")
    else:
        if not _is_nonempty_string(crop.get("run_id")) or not isinstance(
            crop.get("source_crop_snapshot"), Mapping
        ):
            errors.append("refined crop source identity is invalid")
        shape = crop.get("roi_shape_hw")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(item) is not int or item <= 0 for item in shape)
        ):
            errors.append("refined crop ROI shape is invalid")
        else:
            roi_shape = list(shape)

    row_errors, row_count = _validate_row_identity(
        payload.get("row_identity"),
        required_arrays={
            "source_crop_row_ids": ("int64", ()),
            "instance_key": ("uint64", ()),
            "source_acquisition_frame_index": ("int64", ()),
            "source_crop_xywh": ("float32", (4,)),
            "available_channels": ("bool", ()),
        },
        allowed_arrays={
            "source_crop_row_ids",
            "instance_key",
            "source_acquisition_frame_index",
            "source_crop_xywh",
            "available_channels",
        },
        non_row_aligned_arrays={"available_channels"},
    )
    errors.extend(row_errors)

    pixels = payload.get("pixels")
    if not isinstance(pixels, Mapping) or set(pixels) != {
        "semantic_input",
        "surface_kind",
        "surface_path",
        "probability_encoding",
        "source_input_binding",
    }:
        errors.append("refined scientific pixels fields are not exact")
    else:
        if pixels.get("semantic_input") != "raw_subject_mask_surface":
            errors.append("refined semantic pixel input changed")
        for name in ("surface_kind", "surface_path"):
            if not _is_nonempty_string(pixels.get(name)):
                errors.append(f"refined pixels {name} is invalid")
        if pixels.get("source_input_binding") != binding:
            errors.append("refined model and pixel source bindings differ")

    inference = payload.get("inference_contract")
    fields = {
        "method",
        "finalization_semantics",
        "output_component_order",
        "component_sources_and_policies",
        "eye_assignment_contract",
        "authoritative_output",
        "derived_cache_policy",
    }
    if not isinstance(inference, Mapping) or set(inference) != fields:
        errors.append("refined inference contract fields are not exact")
    else:
        components = inference.get("output_component_order")
        policies = inference.get("component_sources_and_policies")
        if (
            not isinstance(components, list)
            or not components
            or len(components) != len(set(components))
            or any(not _is_nonempty_string(item) for item in components)
            or not isinstance(policies, Mapping)
            or set(policies) != set(components)
        ):
            errors.append("refined component policy inventory is invalid")
        if inference.get("eye_assignment_contract") is not None and not isinstance(
            inference.get("eye_assignment_contract"), Mapping
        ):
            errors.append("refined eye-assignment contract is invalid")
        if (
            inference.get("finalization_semantics")
            != "smart_probability_to_refined_candidate"
        ):
            errors.append("refined finalization semantics changed")
        if inference.get("authoritative_output") != "dense_uint8_masks_roi":
            errors.append("refined authoritative output changed")
        if (
            inference.get("derived_cache_policy")
            != "bitpacked_rle_metrics_contours_non_authoritative"
        ):
            errors.append("refined derived-cache policy changed")
        if isinstance(model, Mapping) and inference.get("method") != model.get(
            "method"
        ):
            errors.append("refined inference method differs from its model policy")
    if row_count is not None and roi_shape and row_count <= 0:
        errors.append("refined row domain is invalid")
    return errors


def build_subject_mask_attempt(
    *,
    scientific_identity: Mapping[str, Any],
    run_path: str,
    attempt_id: str | None = None,
    retry_of_attempt_id: str | None = None,
    supersedes_run: str | None = None,
) -> dict[str, object]:
    """Create one execution identity without changing scientific identity."""

    errors = validate_subject_mask_scientific_identity(scientific_identity)
    if errors:
        raise ValueError(
            "Invalid subject-mask scientific identity: " + "; ".join(errors)
        )
    resolved_attempt = str(UUID(attempt_id)) if attempt_id else str(uuid4())
    resolved_retry = (
        str(UUID(str(retry_of_attempt_id))) if retry_of_attempt_id is not None else None
    )
    if resolved_retry == resolved_attempt:
        raise ValueError("retry_of_attempt_id cannot equal attempt_id.")
    normalized_path = str(run_path).strip().strip("/")
    if not normalized_path or "/" not in normalized_path:
        raise ValueError("run_path must contain a family and run name.")
    payload = {
        "attempt_id": resolved_attempt,
        "scientific_identity_digest": scientific_identity["digest"],
        "run_path": normalized_path,
        "retry_of_attempt_id": resolved_retry,
        "supersedes_run": _optional_run_name(
            supersedes_run,
            name="supersedes_run",
        ),
        "retry_policy": "new_immutable_run_name_same_scientific_identity",
        "supersedes_policy": "explicit_predecessor_only_no_implicit_latest",
    }
    return {
        "schema_id": SUBJECT_MASK_ATTEMPT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_ATTEMPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def validate_subject_mask_scientific_identity(
    value: Mapping[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    if set(value) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "digest",
        "payload",
    }:
        errors.append("scientific identity envelope fields are not exact")
    payload = value.get("payload")
    if (
        value.get("schema_id") != SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_ID
        or value.get("schema_version")
        not in {
            SUBJECT_MASK_SCIENTIFIC_IDENTITY_LEGACY_SCHEMA_VERSION,
            SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
        }
        or value.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("scientific identity envelope mismatch")
    if not isinstance(payload, Mapping) or set(payload) != {
        "stage_kind",
        "model",
        "crop",
        "pixels",
        "row_identity",
        "inference_contract",
    }:
        errors.append("scientific identity payload fields are not exact")
    else:
        try:
            if value.get("digest") != canonical_json_sha256(payload):
                errors.append("scientific identity digest mismatch")
        except (TypeError, ValueError) as exc:
            errors.append(f"scientific identity is not strict JSON: {exc}")
        if (
            value.get("schema_version")
            == SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION
        ):
            stage = payload.get("stage_kind")
            if stage == "raw_subject_mask":
                errors.extend(_validate_raw_scientific_payload(payload))
            elif stage == "refined_subject_mask":
                errors.extend(_validate_refined_scientific_payload(payload))
            else:
                errors.append(
                    "scientific identity v2 supports raw and refined masks only"
                )
    return tuple(errors)


def validate_subject_mask_attempt(value: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if set(value) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("attempt envelope fields are not exact")
    payload = value.get("payload")
    if (
        value.get("schema_id") != SUBJECT_MASK_ATTEMPT_SCHEMA_ID
        or value.get("schema_version") != SUBJECT_MASK_ATTEMPT_SCHEMA_VERSION
        or value.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("attempt envelope mismatch")
    if not isinstance(payload, Mapping) or set(payload) != {
        "attempt_id",
        "scientific_identity_digest",
        "run_path",
        "retry_of_attempt_id",
        "supersedes_run",
        "retry_policy",
        "supersedes_policy",
    }:
        errors.append("attempt payload fields are not exact")
        return tuple(errors)
    try:
        UUID(str(payload.get("attempt_id")))
        if payload.get("retry_of_attempt_id") is not None:
            UUID(str(payload.get("retry_of_attempt_id")))
    except (TypeError, ValueError):
        errors.append("attempt UUID lineage is invalid")
    if value.get("payload_digest") != canonical_json_sha256(payload):
        errors.append("attempt payload digest mismatch")
    return tuple(errors)


def resolve_subject_mask_attempt_lineage(
    *,
    parent: Any,
    current_run_name: str,
    scientific_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
    retry_of_attempt_id: str | None,
    supersedes_run: str | None,
    scientific_identity_attr: str = "subject_mask_scientific_identity",
    attempt_attr: str = "subject_mask_attempt",
) -> dict[str, object]:
    """Bind retry/supersession claims to immutable terminal sibling runs."""

    scientific_errors = validate_subject_mask_scientific_identity(scientific_identity)
    attempt_errors = validate_subject_mask_attempt(attempt)
    if scientific_errors or attempt_errors:
        raise ValueError(
            "Invalid subject-mask attempt records: "
            f"science={scientific_errors!r}, attempt={attempt_errors!r}."
        )
    attempt_id = str(attempt["payload"]["attempt_id"])
    retry_matches: list[tuple[str, Any, Mapping[str, Any]]] = []
    for sibling_name in parent.keys():
        if sibling_name == current_run_name:
            continue
        sibling = parent[sibling_name]
        sibling_attempt = sibling.attrs.get(attempt_attr)
        if not isinstance(sibling_attempt, Mapping):
            continue
        errors = validate_subject_mask_attempt(sibling_attempt)
        if errors:
            raise ValueError(
                f"Sibling {sibling_name!r} has malformed subject-mask attempt "
                f"metadata: {errors!r}."
            )
        sibling_attempt_id = str(sibling_attempt["payload"]["attempt_id"])
        if sibling_attempt_id == attempt_id:
            raise ValueError(
                f"Subject-mask attempt_id {attempt_id!r} is already in use by "
                f"{sibling_name!r}."
            )
        if retry_of_attempt_id is not None and sibling_attempt_id == str(
            retry_of_attempt_id
        ):
            retry_matches.append((str(sibling_name), sibling, sibling_attempt))

    retry_evidence: dict[str, object] | None = None
    if retry_of_attempt_id is not None:
        if len(retry_matches) != 1:
            raise ValueError(
                "retry_of_attempt_id must identify exactly one sibling attempt; "
                f"found {len(retry_matches)}."
            )
        retry_name, retry_group, retry_attempt = retry_matches[0]
        retry_science = retry_group.attrs.get(scientific_identity_attr)
        if (
            retry_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != "failed"
            or not isinstance(retry_science, Mapping)
            or retry_science.get("digest") != scientific_identity.get("digest")
        ):
            raise ValueError(
                "A retry must reference one failed attempt with the exact same "
                "scientific identity."
            )
        retry_evidence = {
            "run_name": retry_name,
            "run_path": (
                f"{str(getattr(parent, 'path', '')).strip('/')}/{retry_name}"
            ).strip("/"),
            "attempt_id": str(retry_of_attempt_id),
            "attempt_payload_digest": retry_attempt.get("payload_digest"),
            "scientific_identity_digest": retry_science.get("digest"),
            "completion_status": "failed",
        }

    supersedes_evidence: dict[str, object] | None = None
    if supersedes_run is not None:
        predecessor_name = str(supersedes_run).strip()
        if predecessor_name == current_run_name or predecessor_name not in parent:
            raise ValueError(
                "supersedes_run must identify a different existing sibling run."
            )
        predecessor = parent[predecessor_name]
        if predecessor.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            raise ValueError("A superseded subject-mask run must be complete.")
        predecessor_attempt = predecessor.attrs.get(attempt_attr)
        predecessor_science = predecessor.attrs.get(scientific_identity_attr)
        supersedes_evidence = {
            "run_name": predecessor_name,
            "run_path": (
                f"{str(getattr(parent, 'path', '')).strip('/')}/{predecessor_name}"
            ).strip("/"),
            "completion_status": RUN_STATUS_COMPLETE,
            "attempt_payload_digest": (
                predecessor_attempt.get("payload_digest")
                if isinstance(predecessor_attempt, Mapping)
                else None
            ),
            "scientific_identity_digest": (
                predecessor_science.get("digest")
                if isinstance(predecessor_science, Mapping)
                else None
            ),
        }
    return {
        "retry_of": retry_evidence,
        "supersedes": supersedes_evidence,
        "lineage_policy": "explicit_terminal_sibling_binding_v1",
    }


__all__ = [
    "SUBJECT_MASK_ATTEMPT_SCHEMA_ID",
    "SUBJECT_MASK_ATTEMPT_SCHEMA_VERSION",
    "SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_ID",
    "SUBJECT_MASK_SCIENTIFIC_IDENTITY_LEGACY_SCHEMA_VERSION",
    "SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION",
    "build_subject_mask_attempt",
    "build_subject_mask_scientific_identity",
    "resolve_subject_mask_attempt_lineage",
    "validate_subject_mask_attempt",
    "validate_subject_mask_scientific_identity",
]
