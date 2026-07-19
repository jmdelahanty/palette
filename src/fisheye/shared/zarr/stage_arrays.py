from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numpy as np
import zarr

from fisheye.shared.row_lineage import (
    ROW_IDENTITY_MODE_INSTANCE_KEY,
    ROW_IDENTITY_MODE_LEGACY_POSITIONAL,
    ROW_IDENTITY_MODE_SCHEMA,
    ROW_IDENTITY_MODES,
)

ShapeDim = Union[str, int]


@dataclass(frozen=True)
class ArraySpec:
    name: str
    dtype: str
    shape_template: Tuple[ShapeDim, ...]
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    zarr_group: str
    specs: Tuple[ArraySpec, ...]
    subgroups: Dict[str, Tuple[ArraySpec, ...]] = field(default_factory=dict)
    required_attrs: Tuple[str, ...] = ()
    required_attr_values: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]


def _shape_to_text(shape_template: Tuple[ShapeDim, ...]) -> str:
    if not shape_template:
        return "()"
    parts = ", ".join(str(dim) for dim in shape_template)
    if len(shape_template) == 1:
        parts += ","
    return f"({parts})"


def describe_array(spec: ArraySpec) -> str:
    return f"{_shape_to_text(spec.shape_template)} {spec.dtype}"


def array_specs_by_name(
    stage_spec: StageSpec,
    *,
    subgroup: str | None = None,
) -> Dict[str, ArraySpec]:
    specs = stage_spec.subgroups[subgroup] if subgroup else stage_spec.specs
    return {item.name: item for item in specs}


def required_array_specs(
    stage_spec: StageSpec,
    *,
    subgroup: str | None = None,
) -> Tuple[ArraySpec, ...]:
    specs = stage_spec.subgroups[subgroup] if subgroup else stage_spec.specs
    return tuple(item for item in specs if item.required)


def required_array_names(
    stage_spec: StageSpec,
    *,
    subgroup: str | None = None,
) -> Tuple[str, ...]:
    return tuple(item.name for item in required_array_specs(stage_spec, subgroup=subgroup))


def _infer_expected_kinds(dtype: str) -> set[str]:
    expected: set[str] = set()
    for part in dtype.lower().split("/"):
        token = part.strip()
        if token.startswith("uint"):
            expected.add("uint")
        elif token.startswith("int"):
            expected.add("int")
        elif token.startswith("float"):
            expected.add("float")
        elif token == "bool":
            expected.add("bool")
        elif token == "string":
            expected.add("string")
    return expected


def _kind_from_numpy(dtype: np.dtype) -> str:
    kind = dtype.kind
    if kind == "u":
        return "uint"
    if kind == "i":
        return "int"
    if kind == "f":
        return "float"
    if kind == "b":
        return "bool"
    if kind in {"U", "S", "O", "T"}:
        return "string"
    return kind


def _kind_from_any_dtype(dtype_obj: object) -> str:
    try:
        return _kind_from_numpy(np.dtype(dtype_obj))
    except TypeError:
        dtype_name = type(dtype_obj).__name__.lower()
        dtype_text = str(dtype_obj).lower()
        if "variablelengthutf8" in dtype_name or "variablelengthutf8" in dtype_text:
            return "string"
        kind_attr = getattr(dtype_obj, "kind", None)
        if isinstance(kind_attr, str):
            if kind_attr == "u":
                return "uint"
            if kind_attr == "i":
                return "int"
            if kind_attr == "f":
                return "float"
            if kind_attr == "b":
                return "bool"
            if kind_attr in {"U", "S", "O", "T"}:
                return "string"
        if "string" in dtype_name or "utf" in dtype_name:
            return "string"
        return dtype_name


def _is_group_like(obj: object) -> bool:
    return isinstance(obj, zarr.Group) or (
        hasattr(obj, "group_keys")
        and hasattr(obj, "__contains__")
        and hasattr(obj, "__getitem__")
    )


def _validate_specs(
    group: zarr.Group,
    specs: Tuple[ArraySpec, ...],
    *,
    group_label: str,
    errors: List[str],
    warnings: List[str],
) -> None:
    leading_dims: Dict[str, int] = {}

    for spec in specs:
        if spec.name not in group:
            message = f"{group_label}: missing {'required' if spec.required else 'optional'} array '{spec.name}'"
            if spec.required:
                errors.append(message)
            else:
                warnings.append(message)
            continue

        array_obj = group[spec.name]
        if not hasattr(array_obj, "shape") or not hasattr(array_obj, "dtype"):
            errors.append(f"{group_label}/{spec.name}: expected array, found non-array object")
            continue

        shape = tuple(int(dim) for dim in array_obj.shape)
        raw_dtype = array_obj.dtype

        expected_kinds = _infer_expected_kinds(spec.dtype)
        if expected_kinds:
            actual_kind = _kind_from_any_dtype(raw_dtype)
            if actual_kind not in expected_kinds:
                expected_text = "/".join(sorted(expected_kinds))
                errors.append(
                    f"{group_label}/{spec.name}: dtype kind mismatch (expected {expected_text} from '{spec.dtype}', got {actual_kind} [{raw_dtype}])"
                )

        template = spec.shape_template
        if len(shape) != len(template):
            errors.append(
                f"{group_label}/{spec.name}: rank mismatch (expected ndim={len(template)} from {_shape_to_text(template)}, got ndim={len(shape)} with shape={shape})"
            )
            continue

        for idx, expected_dim in enumerate(template):
            if isinstance(expected_dim, int) and shape[idx] != expected_dim:
                errors.append(
                    f"{group_label}/{spec.name}: dimension {idx} mismatch (expected {expected_dim}, got {shape[idx]})"
                )

        if shape and isinstance(template[0], str):
            leading_key = template[0]
            current = shape[0]
            previous = leading_dims.get(leading_key)
            if previous is None:
                leading_dims[leading_key] = current
            elif previous != current:
                errors.append(
                    f"{group_label}: leading dimension mismatch for '{leading_key}' ({spec.name} has {current}, expected {previous})"
                )


_REFINED_IDENTITY_TARGETS = {
    "refined_detect": "instances",
    "refined_keypoints": None,
    "refined_subject_masks": None,
}


def _validate_refined_instance_key_identity(
    group: zarr.Group,
    spec: StageSpec,
    *,
    errors: List[str],
    warnings: List[str],
) -> None:
    if spec.stage_name not in _REFINED_IDENTITY_TARGETS:
        return

    subgroup_name = _REFINED_IDENTITY_TARGETS[spec.stage_name]
    identity_group = group
    identity_label = spec.stage_name
    if subgroup_name is not None:
        if subgroup_name not in group or not _is_group_like(group[subgroup_name]):
            return
        identity_group = group[subgroup_name]
        identity_label = f"{spec.stage_name}/{subgroup_name}"

    missing_optional = f"{identity_label}: missing optional array 'instance_key'"
    warnings[:] = [message for message in warnings if message != missing_optional]

    attrs = getattr(group, "attrs", {})
    raw_mode = attrs.get("row_identity_mode")
    mode = str(raw_mode).strip() if raw_mode is not None else None
    schema = attrs.get("row_identity_mode_schema")
    key_array = identity_group.get("instance_key")
    has_keys = key_array is not None

    if mode is not None and mode not in ROW_IDENTITY_MODES:
        errors.append(
            f"{spec.stage_name}: unknown row_identity_mode {mode!r}; "
            f"expected one of {sorted(ROW_IDENTITY_MODES)}"
        )
        return
    if mode is not None and schema != ROW_IDENTITY_MODE_SCHEMA:
        errors.append(
            f"{spec.stage_name}: row_identity_mode_schema must be "
            f"{ROW_IDENTITY_MODE_SCHEMA!r} when row_identity_mode is set"
        )

    if mode == ROW_IDENTITY_MODE_INSTANCE_KEY and not has_keys:
        errors.append(
            f"{identity_label}: modern row_identity_mode='instance_key' requires instance_key"
        )
        return
    if mode == ROW_IDENTITY_MODE_LEGACY_POSITIONAL and has_keys:
        errors.append(
            f"{identity_label}: legacy_positional mode cannot contain instance_key; "
            "refusing an identity downgrade"
        )
        return

    instance_key_status = getattr(identity_group, "attrs", {}).get("instance_key_status")
    if not has_keys:
        if instance_key_status == "present":
            errors.append(
                f"{identity_label}: instance_key_status='present' but instance_key is missing"
            )
        elif mode == ROW_IDENTITY_MODE_LEGACY_POSITIONAL:
            warnings.append(
                f"{identity_label}: modern identity array absent; explicit legacy_positional compatibility mode"
            )
        elif mode is None:
            warnings.append(
                f"{identity_label}: modern identity array absent; inferred legacy compatibility mode"
            )
        return

    if instance_key_status == "missing":
        errors.append(
            f"{identity_label}: instance_key_status='missing' but instance_key is present"
        )

    if np.dtype(key_array.dtype) != np.dtype(np.uint64):
        errors.append(
            f"{identity_label}/instance_key: dtype must be uint64, got {key_array.dtype}"
        )
    values = np.asarray(key_array[:], dtype=np.uint64).reshape(-1)
    unique_count = int(np.unique(values).size)
    if unique_count != int(values.size):
        errors.append(
            f"{identity_label}/instance_key: duplicate values "
            f"({unique_count}/{int(values.size)} unique)"
        )
    if mode is None:
        warnings.append(
            f"{identity_label}: instance_key present but row_identity_mode is unstamped; "
            "inferred instance_key compatibility mode"
        )


_REFINED_SUBJECT_MASK_RLE_COMPONENT_SPECS: Tuple[ArraySpec, ...] = (
    ArraySpec("counts", "uint32", ("n_counts",)),
    ArraySpec("indptr", "int64", ("n_rois_plus_one",)),
    ArraySpec("present", "bool", ("n_rois",)),
    ArraySpec("area_px", "int32", ("n_rois",)),
    ArraySpec("bbox_xyxy", "int32", ("n_rois", 4)),
)


def _validate_shape_hw_attr(
    attrs: object,
    *,
    attr_name: str,
    group_label: str,
    errors: List[str],
) -> tuple[int, int] | None:
    if not hasattr(attrs, "get"):
        errors.append(f"{group_label}: attrs object does not support get()")
        return None
    raw_shape = attrs.get(attr_name)
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
        errors.append(f"{group_label}: attr '{attr_name}' must be a 2-item [height, width] sequence")
        return None
    try:
        height = int(raw_shape[0])
        width = int(raw_shape[1])
    except (TypeError, ValueError):
        errors.append(f"{group_label}: attr '{attr_name}' must contain integer dimensions, got {raw_shape!r}")
        return None
    if height <= 0 or width <= 0:
        errors.append(f"{group_label}: attr '{attr_name}' dimensions must be positive, got {(height, width)!r}")
        return None
    return (height, width)


def _group_key_names(group: object) -> Tuple[str, ...]:
    if hasattr(group, "group_keys"):
        return tuple(str(key) for key in group.group_keys())
    if hasattr(group, "keys"):
        return tuple(str(key) for key in group.keys())
    return ()


def _validate_refined_subject_mask_rle_component_integrity(
    component_group: object,
    *,
    component_name: str,
    parent_shape_hw: tuple[int, int] | None,
    errors: List[str],
) -> None:
    label = f"refined_subject_masks/mask_rle/components/{component_name}"
    attrs = getattr(component_group, "attrs", {})
    component_shape_hw = None
    if hasattr(attrs, "get") and "encoded_shape_hw" in attrs:
        component_shape_hw = _validate_shape_hw_attr(
            attrs,
            attr_name="encoded_shape_hw",
            group_label=label,
            errors=errors,
        )
        if parent_shape_hw is not None and component_shape_hw is not None and component_shape_hw != parent_shape_hw:
            errors.append(
                f"{label}: attr 'encoded_shape_hw' {component_shape_hw!r} does not match parent {parent_shape_hw!r}"
            )

    arrays: dict[str, object] = {}
    for array_name in ("counts", "indptr", "present", "area_px", "bbox_xyxy"):
        arrays[array_name] = component_group.get(array_name) if hasattr(component_group, "get") else None
    if not all(hasattr(array_obj, "shape") for array_obj in arrays.values()):
        return

    counts = arrays["counts"]
    indptr = arrays["indptr"]
    present = arrays["present"]
    area_px = arrays["area_px"]
    bbox_xyxy = arrays["bbox_xyxy"]

    row_count = int(present.shape[0])
    if int(area_px.shape[0]) != row_count:
        errors.append(f"{label}: area_px length {int(area_px.shape[0])} must equal present length {row_count}")
    if int(bbox_xyxy.shape[0]) != row_count:
        errors.append(f"{label}: bbox_xyxy length {int(bbox_xyxy.shape[0])} must equal present length {row_count}")
    if int(indptr.shape[0]) != row_count + 1:
        errors.append(
            f"{label}: indptr length {int(indptr.shape[0])} must equal present length + 1 ({row_count + 1})"
        )
        return

    try:
        pointers = np.asarray(indptr[:], dtype=np.int64).reshape(-1)
    except Exception as exc:  # pragma: no cover - defensive for unusual stores
        errors.append(f"{label}: failed to read indptr for structural validation: {exc}")
        return
    if pointers.size == 0:
        errors.append(f"{label}: indptr must contain at least one pointer")
        return
    if int(pointers[0]) != 0 or np.any(np.diff(pointers) < 0):
        errors.append(f"{label}: indptr must start at zero and be monotonically non-decreasing")
    counts_len = int(counts.shape[0])
    if int(pointers[-1]) != counts_len:
        errors.append(f"{label}: indptr terminates at {int(pointers[-1])}, but counts has {counts_len} values")


def _validate_refined_subject_mask_bitpacked_storage(
    group: zarr.Group,
    *,
    errors: List[str],
) -> bool:
    bitpacked_group = group.get("mask_bitpacked") if hasattr(group, "get") else None
    if bitpacked_group is None:
        return False

    start_error_count = len(errors)
    label = "refined_subject_masks/mask_bitpacked"
    if not _is_group_like(bitpacked_group):
        errors.append(f"{label}: exists but is not a group")
        return False

    attrs = getattr(bitpacked_group, "attrs", {})
    required_attrs = (
        "schema_id",
        "mask_encoding",
        "mask_value_semantics",
        "layout",
        "logical_shape",
        "encoded_shape",
        "component_names",
        "packed_axis",
        "packed_bitorder",
        "packed_width_bytes",
    )
    for attr_name in required_attrs:
        if attr_name not in attrs:
            errors.append(f"{label}: missing required attr '{attr_name}'")
    if attrs.get("schema_id") != "palette_mask_bitpacked_binary_v1":
        errors.append(
            f"{label}: attr 'schema_id' expected 'palette_mask_bitpacked_binary_v1', "
            f"got {attrs.get('schema_id')!r}"
        )
    if attrs.get("mask_encoding") != "bitpacked_binary_v1":
        errors.append(
            f"{label}: attr 'mask_encoding' expected 'bitpacked_binary_v1', "
            f"got {attrs.get('mask_encoding')!r}"
        )
    if attrs.get("layout") != "packed_width_array":
        errors.append(
            f"{label}: attr 'layout' expected 'packed_width_array', got {attrs.get('layout')!r}"
        )
    if attrs.get("packed_axis") != "width":
        errors.append(f"{label}: attr 'packed_axis' expected 'width', got {attrs.get('packed_axis')!r}")
    if attrs.get("packed_bitorder") != "little":
        errors.append(f"{label}: attr 'packed_bitorder' expected 'little', got {attrs.get('packed_bitorder')!r}")

    raw_logical_shape = attrs.get("logical_shape")
    raw_encoded_shape = attrs.get("encoded_shape")
    logical_shape: tuple[int, int, int, int] | None = None
    encoded_shape: tuple[int, int, int, int] | None = None
    if not isinstance(raw_logical_shape, (list, tuple)) or len(raw_logical_shape) != 4:
        errors.append(f"{label}: attr 'logical_shape' must contain [n_rois, n_channels, height, width]")
    else:
        logical_shape = tuple(int(value) for value in raw_logical_shape)
        if min(logical_shape) <= 0:
            errors.append(f"{label}: attr 'logical_shape' dimensions must be positive, got {logical_shape!r}")
    if not isinstance(raw_encoded_shape, (list, tuple)) or len(raw_encoded_shape) != 4:
        errors.append(f"{label}: attr 'encoded_shape' must contain [n_rois, n_channels, height, packed_width]")
    else:
        encoded_shape = tuple(int(value) for value in raw_encoded_shape)
    if logical_shape is not None and encoded_shape is not None:
        expected_encoded_shape = (
            int(logical_shape[0]),
            int(logical_shape[1]),
            int(logical_shape[2]),
            int((logical_shape[3] + 7) // 8),
        )
        if encoded_shape != expected_encoded_shape:
            errors.append(
                f"{label}: attr 'encoded_shape' {encoded_shape!r} does not match expected "
                f"{expected_encoded_shape!r}"
            )
        if int(attrs.get("packed_width_bytes", -1)) != int(expected_encoded_shape[3]):
            errors.append(
                f"{label}: attr 'packed_width_bytes' {attrs.get('packed_width_bytes')!r} "
                f"does not match expected {expected_encoded_shape[3]}"
            )

    raw_names = attrs.get("component_names")
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        errors.append(f"{label}: attr 'component_names' must be a non-empty list")
    elif logical_shape is not None and len(raw_names) != int(logical_shape[1]):
        errors.append(
            f"{label}: component_names length {len(raw_names)} does not match channel count {logical_shape[1]}"
        )

    packed = bitpacked_group.get("masks_packed") if hasattr(bitpacked_group, "get") else None
    if packed is None or not hasattr(packed, "shape"):
        errors.append(f"{label}: missing required array 'masks_packed'")
    else:
        if encoded_shape is not None and tuple(int(value) for value in packed.shape) != encoded_shape:
            errors.append(
                f"{label}/masks_packed: shape {tuple(int(value) for value in packed.shape)!r} "
                f"does not match encoded_shape {encoded_shape!r}"
            )
        try:
            dtype = np.dtype(getattr(packed, "dtype", None))
        except Exception:
            dtype = None
        if dtype != np.dtype("uint8"):
            errors.append(f"{label}/masks_packed: dtype must be uint8, got {getattr(packed, 'dtype', None)!r}")

    return len(errors) == start_error_count


def _validate_refined_subject_mask_storage(
    group: zarr.Group,
    *,
    errors: List[str],
    warnings: List[str],
) -> None:
    """Allow compact mask stores as refined-subject mask physical surfaces."""

    dense_missing_error = "refined_subject_masks: missing required array 'masks_roi'"
    has_valid_compact = _validate_refined_subject_mask_bitpacked_storage(group, errors=errors)
    rle_group = group.get("mask_rle") if hasattr(group, "get") else None
    if rle_group is None:
        if has_valid_compact and dense_missing_error in errors:
            errors[:] = [message for message in errors if message != dense_missing_error]
        return

    start_error_count = len(errors)
    if not _is_group_like(rle_group):
        errors.append("refined_subject_masks/mask_rle: exists but is not a group")
        return

    attrs = getattr(rle_group, "attrs", {})
    for attr_name in ("schema_id", "mask_encoding", "mask_value_semantics", "layout", "encoded_shape_hw", "component_names"):
        if attr_name not in attrs:
            errors.append(f"refined_subject_masks/mask_rle: missing required attr '{attr_name}'")
    if attrs.get("schema_id") != "palette_mask_rle_binary_v1":
        errors.append(
            "refined_subject_masks/mask_rle: attr 'schema_id' expected "
            "'palette_mask_rle_binary_v1', got {!r}".format(attrs.get("schema_id"))
        )
    if attrs.get("layout") != "component_groups":
        errors.append(
            "refined_subject_masks/mask_rle: attr 'layout' expected "
            "'component_groups', got {!r}".format(attrs.get("layout"))
        )
    parent_shape_hw = _validate_shape_hw_attr(
        attrs,
        attr_name="encoded_shape_hw",
        group_label="refined_subject_masks/mask_rle",
        errors=errors,
    )

    components = rle_group.get("components") if hasattr(rle_group, "get") else None
    if components is None or not _is_group_like(components):
        errors.append("refined_subject_masks/mask_rle: missing required subgroup 'components'")
        return
    component_names = _group_key_names(components)
    if not component_names:
        errors.append("refined_subject_masks/mask_rle/components: no component groups found")
        return

    for component_name in component_names:
        component_group = components[component_name]
        if not _is_group_like(component_group):
            errors.append(
                f"refined_subject_masks/mask_rle/components/{component_name}: exists but is not a group"
            )
            continue
        _validate_specs(
            component_group,
            _REFINED_SUBJECT_MASK_RLE_COMPONENT_SPECS,
            group_label=f"refined_subject_masks/mask_rle/components/{component_name}",
            errors=errors,
            warnings=warnings,
        )
        _validate_refined_subject_mask_rle_component_integrity(
            component_group,
            component_name=component_name,
            parent_shape_hw=parent_shape_hw,
            errors=errors,
        )

    has_valid_compact = has_valid_compact or len(errors) == start_error_count
    if has_valid_compact and dense_missing_error in errors:
        errors[:] = [message for message in errors if message != dense_missing_error]


def validate_run(group: zarr.Group, spec: StageSpec) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    attrs = getattr(group, "attrs", {})
    for attr_name in spec.required_attrs:
        if attr_name not in attrs:
            errors.append(f"{spec.stage_name}: missing required attr '{attr_name}'")
    for attr_name, expected_value in spec.required_attr_values.items():
        if attr_name not in attrs:
            errors.append(f"{spec.stage_name}: missing required attr '{attr_name}'")
        elif attrs[attr_name] != expected_value:
            errors.append(
                f"{spec.stage_name}: attr '{attr_name}' expected {expected_value!r}, got {attrs[attr_name]!r}"
            )

    if spec.specs:
        _validate_specs(
            group,
            spec.specs,
            group_label=spec.stage_name,
            errors=errors,
            warnings=warnings,
        )

    if spec.subgroups:
        for subgroup_name, subgroup_specs in spec.subgroups.items():
            if subgroup_name not in group:
                errors.append(f"{spec.stage_name}: missing required subgroup '{subgroup_name}'")
                continue
            subgroup = group[subgroup_name]
            if not _is_group_like(subgroup):
                errors.append(f"{spec.stage_name}: '{subgroup_name}' exists but is not a group")
                continue

            _validate_specs(
                subgroup,
                subgroup_specs,
                group_label=f"{spec.stage_name}/{subgroup_name}",
                errors=errors,
                warnings=warnings,
            )

    _validate_refined_instance_key_identity(
        group,
        spec,
        errors=errors,
        warnings=warnings,
    )

    if spec.stage_name == "refined_subject_masks":
        _validate_refined_subject_mask_storage(group, errors=errors, warnings=warnings)

    if spec.stage_name == "subject_masks":
        storage_mode = str(getattr(group, "attrs", {}).get("subject_mask_storage_mode") or "").strip()
        probabilities = group.get("mask_probs_roi") if hasattr(group, "get") else None
        if storage_mode == "composite":
            if probabilities is not None:
                errors.append("subject_masks: composite run must not expose top-level 'mask_probs_roi'")
            payload = group.get("composite_payload") if hasattr(group, "get") else None
            for name in (
                "source_codes",
                "source_row_indices",
                "delta_target_row_indices",
                "delta_instance_key",
                "mask_probs_roi_delta",
            ):
                if payload is None or name not in payload:
                    errors.append(f"subject_masks/composite_payload: missing required array '{name}'")
            try:
                composite_version = int(
                    getattr(group, "attrs", {}).get(
                        "composite_subject_mask_schema_version",
                        1,
                    )
                )
            except (TypeError, ValueError):
                composite_version = -1
            if composite_version >= 2 and (
                payload is None or "source_run_indices" not in payload
            ):
                errors.append(
                    "subject_masks/composite_payload: missing required array "
                    "'source_run_indices' for schema v2"
                )
        elif probabilities is None:
            errors.append("subject_masks: missing required array 'mask_probs_roi'")

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


RAW_VIDEO_SPEC = StageSpec(
    stage_name="raw_video",
    zarr_group="raw_video/",
    specs=(
        ArraySpec("images_full", "uint8", ("n_frames", "H", "W"), required=False),
        ArraySpec(
            "images_ds",
            "uint8",
            ("n_frames", "H_ds", "W_ds"),
            required=False,
            description="Optional downsampled import; at least one of images_full/images_ds should exist.",
        ),
        ArraySpec("images_ds_rgb", "uint8", ("n_frames", "H_ds", "W_ds", 3), required=False),
        ArraySpec(
            "original_frame_indices",
            "int32",
            ("n_import_frames",),
            required=False,
            description="Present when frame_step > 1.",
        ),
    ),
)

STIMULUS_SPEC = StageSpec(
    stage_name="stimulus",
    zarr_group="analysis/stimulus_runs/<run>/",
    specs=(
        ArraySpec(
            "interpolation_mask",
            "bool",
            ("n_metadata_frames",),
            description="Per-row flag for interpolated stimulus metadata records.",
        ),
    ),
    subgroups={
        "video_metadata/frame_metadata": (
            ArraySpec(
                "stimulus_frame_num",
                "uint64",
                ("n_metadata_frames",),
                description="Stimulus frame counter in the imported H5 metadata stream.",
            ),
        ),
        "frame_alignment": (
            ArraySpec(
                "camera_to_metadata_index",
                "int64",
                ("n_camera_frames",),
                description="Absolute camera frame to stimulus metadata row map; missing frames are -1.",
            ),
        ),
    },
)

KEYPOINT_PROFILE_SPEC = StageSpec(
    stage_name="keypoint_profile",
    zarr_group="analysis/keypoint_profile_runs/<run>/",
    specs=(),
    required_attrs=(
        "schema_name",
        "schema_version",
        "created_at_utc",
        "profile_summary",
    ),
)

DETECTION_PROFILE_SPEC = StageSpec(
    stage_name="detection_profile",
    zarr_group="analysis/detection_profile_runs/<run>/",
    specs=(),
    required_attrs=(
        "schema_name",
        "schema_version",
        "created_at_utc",
        "profile_summary",
    ),
)

EYE_MASK_PROFILE_SPEC = StageSpec(
    stage_name="eye_mask_profile",
    zarr_group="analysis/eye_mask_profile_runs/<run>/",
    specs=(),
    required_attrs=(
        "schema_name",
        "schema_version",
        "created_at_utc",
        "profile_summary",
    ),
)

SUBJECT_SHAPE_SPEC = StageSpec(
    stage_name="subject_shape",
    zarr_group="analysis/subject_shape_runs/<run>/",
    specs=(),
    required_attrs=(
        "schema_id",
        "schema_version",
        "method",
        "method_version",
        "created_at_utc",
        "row_axis",
        "source_refined_subject_masks_run",
        "source_refs",
    ),
    subgroups={
        "row_index": (
            ArraySpec("frame_indices", "int32", ("n_rows",)),
            ArraySpec("source_refined_row_ids", "int64", ("n_rows",)),
        ),
        "components/subject_body": (
            ArraySpec("mask_present", "bool", ("n_rows",)),
            ArraySpec("area_px", "float32", ("n_rows",)),
            ArraySpec("centroid_xy", "float32", ("n_rows", 2)),
            ArraySpec("centroid_valid", "bool", ("n_rows",)),
            ArraySpec("bbox_xyxy", "float32", ("n_rows", 4)),
            ArraySpec("bbox_valid", "bool", ("n_rows",)),
        ),
    },
)

TAIL_POSTURE_VIEW_SPEC = StageSpec(
    stage_name="tail_posture_view",
    zarr_group="analysis/tail_posture_view_runs/<run>/",
    specs=(
        ArraySpec("valid", "bool", ("n_rows",)),
        ArraySpec("failure_reason_bytes", "uint8", ("n_rows", "width")),
        ArraySpec("frame_index", "int32/int64", ("n_rows",)),
        ArraySpec("head_xy", "float32", ("n_rows", 2)),
        ArraySpec("head_yaw_rad", "float32", ("n_rows",)),
        ArraySpec("tail_keypoints_xy", "float32", ("n_rows", "n_tail_keypoints", 2)),
        ArraySpec("tail_angle_rad", "float32", ("n_rows", "n_tail_angles")),
        ArraySpec("tail_angle_deg", "float32", ("n_rows", "n_tail_angles")),
    ),
    required_attrs=(
        "schema_id",
        "schema_version",
        "method",
        "method_version",
        "created_at_utc",
        "row_axis",
        "view_family",
        "source_subject_shape_run",
        "source_subject_shape_path",
        "source_tail_geometry_kind",
        "head_source",
        "keypoint_count",
        "angle_count",
        "angle_convention",
        "angle_units_primary",
        "frame_index_source",
        "source_refs",
        "algorithm_provenance",
    ),
    subgroups={
        "row_index": (
            ArraySpec("frame_indices", "int32", ("n_rows",)),
            ArraySpec("source_refined_row_ids", "int64", ("n_rows",)),
        ),
    },
)

BOUT_CLASSIFICATION_SPEC = StageSpec(
    stage_name="bout_classification",
    zarr_group="analysis/bout_classification_runs/<run>/",
    specs=(),
    required_attrs=(
        "schema_id",
        "schema_version",
        "classifier_family",
        "classifier_name",
        "source_mode",
        "row_axis",
        "invalid_window_policy",
        "source_refs",
        "parameters",
    ),
    subgroups={
        "per_bout": (
            ArraySpec("source_bout_id", "int32/int64", ("n_bouts",)),
            ArraySpec("start_frame", "int32/int64", ("n_bouts",)),
            ArraySpec("end_frame", "int32/int64", ("n_bouts",)),
            ArraySpec("window_start_frame", "int32/int64", ("n_bouts",)),
            ArraySpec("window_end_frame", "int32/int64", ("n_bouts",)),
            ArraySpec("HB1_frame", "int32/int64", ("n_bouts",)),
            ArraySpec("HB1_offset_frames", "int32/int64", ("n_bouts",)),
            ArraySpec("category_id", "int32/int64", ("n_bouts",)),
            ArraySpec("category_label_bytes", "uint8", ("n_bouts", "width")),
            ArraySpec("subcategory_id", "int32/int64", ("n_bouts",)),
            ArraySpec("sign", "int32/int64", ("n_bouts",)),
            ArraySpec("probability", "float32", ("n_bouts",)),
            ArraySpec("tail_valid_fraction", "float32", ("n_bouts",)),
            ArraySpec("traj_valid_fraction", "float32", ("n_bouts",)),
            ArraySpec("max_consecutive_tail_invalid", "int32/int64", ("n_bouts",)),
            ArraySpec("max_consecutive_traj_invalid", "int32/int64", ("n_bouts",)),
            ArraySpec("source_window_valid", "bool", ("n_bouts",)),
            ArraySpec("classified", "bool", ("n_bouts",)),
            ArraySpec("valid", "bool", ("n_bouts",)),
            ArraySpec("failure_reason_bytes", "uint8", ("n_bouts", "width")),
        ),
    },
)

TAIL_KINEMATICS_SPEC = StageSpec(
    stage_name="tail_kinematics",
    zarr_group="analysis/tail_kinematics_runs/<run>/",
    specs=(
        ArraySpec("valid", "bool", ("n_rows",)),
        ArraySpec("failure_reason_bytes", "uint8", ("n_rows", "width")),
        ArraySpec("frame_index", "int32/int64", ("n_rows",)),
        ArraySpec("tail_angle_sample_s", "float32/float64", ("n_tail_samples",)),
        ArraySpec("tail_angle_rad", "float32", ("n_rows", "n_tail_samples")),
        ArraySpec("tail_angle_deg", "float32", ("n_rows", "n_tail_samples")),
        ArraySpec("tail_tip_angle_rad", "float32", ("n_rows",)),
        ArraySpec("tail_tip_angle_deg", "float32", ("n_rows",)),
        ArraySpec("max_abs_tail_angle_rad", "float32", ("n_rows",)),
        ArraySpec("max_abs_tail_angle_deg", "float32", ("n_rows",)),
        ArraySpec("tail_angle_rms_rad", "float32", ("n_rows",)),
        ArraySpec("tail_angle_rms_deg", "float32", ("n_rows",)),
    ),
    required_attrs=(
        "schema_id",
        "schema_version",
        "method",
        "method_version",
        "created_at_utc",
        "row_axis",
        "source_subject_shape_run",
        "source_subject_shape_path",
        "source_tail_geometry_kind",
        "tail_angle_sample_count",
        "frame_index_source",
        "source_refs",
    ),
    subgroups={
        "row_index": (
            ArraySpec("frame_indices", "int32", ("n_rows",)),
            ArraySpec("source_refined_row_ids", "int64", ("n_rows",)),
        ),
    },
)

TRACK_KINEMATICS_SPEC = StageSpec(
    stage_name="track_kinematics",
    zarr_group="analysis/track_kinematics_runs/<run_type>/<run>/",
    specs=(
        ArraySpec("track_ids", "int32/int64", ("n_tracks",)),
        ArraySpec("track_arena_ids", "int32/int64", ("n_tracks",)),
    ),
    required_attrs=(
        "track_manifest",
        "provenance",
        "created_at_utc",
        "git_commit",
        "git_branch",
        "method",
        "fps",
        "source_tracking_run",
        "summary",
        "num_tracks",
    ),
)

EYE_ANGLE_SPEC = StageSpec(
    stage_name="eye_angle",
    zarr_group="analysis/eye_angle_runs/<run>/",
    specs=(),
    required_attrs=(
        "report_version",
        "provenance",
        "source_eye_geometry_stage",
        "source_eye_geometry_run",
        "source_keypoint_run",
        "source_keypoints_run",
        "fps",
        "num_detections",
        "num_frames",
    ),
    required_attr_values={"status": "complete"},
    subgroups={
        "support": (
            ArraySpec("frame_indices", "int32/int64", ("n_rows",)),
            ArraySpec("time_seconds", "float32/float64", ("n_rows",)),
            ArraySpec("ellipse_major", "float32/float64", ("n_rows",)),
            ArraySpec("ellipse_minor", "float32/float64", ("n_rows",)),
            ArraySpec("ellipse_ratio", "float32/float64", ("n_rows",)),
        ),
    },
)

BOUT_KINEMATICS_SPEC = StageSpec(
    stage_name="bout_kinematics",
    zarr_group="analysis/bout_kinematics_runs/<run>/",
    specs=(),
    required_attrs=(
        "schema_version",
        "method_version",
        "created_at_utc",
        "parameters",
        "provenance",
        "source_refs",
        "source_track_id",
        "source_swim_bout_run",
        "source_swim_bout_speed_level",
        "source_track_kinematics_run",
        "default_heading_level",
        "heading_levels",
    ),
    required_attr_values={
        "schema_id": "analysis.bout_kinematics_runs",
        "method": "heading_window_and_within_bout_metrics",
        "row_axis": "swim_bout_rows",
    },
)

BACKGROUND_SPEC = StageSpec(
    stage_name="background",
    zarr_group="background_runs/<run>/",
    specs=(
        ArraySpec("background_full", "uint8", ("H", "W"), required=False),
        ArraySpec("background_ds", "uint8", ("H_ds", "W_ds"), required=False),
        ArraySpec(
            "frame_indices",
            "int32",
            ("n_samples",),
            required=False,
            description="Optional materialized sample list; current writer records source_frame_indices in attrs.",
        ),
    ),
)

DETECT_SPEC = StageSpec(
    stage_name="detect",
    zarr_group="detect_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_detections",)),
        ArraySpec("bbox_norm_coords", "float32", ("n_detections", 4)),
        ArraySpec("scores", "float32", ("n_detections",)),
        ArraySpec("class_ids", "int32", ("n_detections",)),
        ArraySpec(
            "instance_key",
            "uint64",
            ("n_detections",),
            required=False,
            description="Content-derived per-detection key minted once at detect time and copied downstream.",
        ),
        ArraySpec("frame_counts", "int32", ("n_frames",)),
        ArraySpec(
            "n_detections",
            "int32",
            ("n_frames",),
            required=False,
            description="Legacy alias of frame_counts.",
        ),
        ArraySpec(
            "centers_px",
            "float32",
            ("n_detections", 2),
            required=False,
            description="Written by blob method.",
        ),
    ),
)

DETECT_QUALITY_SPEC = StageSpec(
    stage_name="detect_quality",
    zarr_group="detect_runs/<run>/quality_reports/<qrun>/",
    specs=(
        ArraySpec("quality_flags", "int8", ("n_frames",)),
        ArraySpec("detection_quality_labels", "int8", ("n_detections",)),
    ),
)

_REFINED_DETECT_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("bbox_norm_coords", "float32", ("n_refined", 4)),
    ArraySpec("scores", "float32", ("n_refined",)),
    ArraySpec("frame_indices", "int32", ("n_refined",)),
    ArraySpec("class_ids", "int32", ("n_refined",)),
    ArraySpec("frame_counts", "int32", ("n_frames",)),
    ArraySpec(
        "n_detections",
        "int32",
        ("n_frames",),
        required=False,
        description="Legacy alias of frame_counts.",
    ),
    ArraySpec(
        "detection_source",
        "int8",
        ("n_refined",),
        description="0=real, 1=interpolated.",
    ),
    ArraySpec("reason_bytes", "uint8", ("n_refined", "width")),
    ArraySpec(
        "frame_mapping",
        "int32",
        ("n_refined",),
        description="Legacy alias of frame_indices.",
    ),
)

_REFINED_DETECT_SOURCE_DETECTIONS_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("source_detect_row_index", "int32", ("n_source_detections",)),
    ArraySpec("instance_key", "uint64", ("n_source_detections",), required=False),
    ArraySpec("frame_indices", "int32", ("n_source_detections",)),
    ArraySpec("bbox_img_xyxy", "float64", ("n_source_detections", 4)),
    ArraySpec("bbox_norm_coords", "float64", ("n_source_detections", 4)),
    ArraySpec("decision_codes", "int8", ("n_source_detections",)),
    ArraySpec("resolved_refined_row_id", "int64", ("n_source_detections",)),
    ArraySpec("reason_bytes", "uint8", ("n_source_detections", "width")),
    ArraySpec("confidence_scores", "float32", ("n_source_detections",), required=False),
    ArraySpec("class_ids", "int32", ("n_source_detections",), required=False),
    ArraySpec("review_notes", "string", ("n_source_detections",), required=False),
)

_REFINED_DETECT_INSTANCES_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("refined_row_ids", "int64", ("n_instances",)),
    ArraySpec("frame_indices", "int32", ("n_instances",)),
    ArraySpec("frame_offsets", "int64", ("n_frame_offsets",)),
    ArraySpec("bbox_img_xyxy", "float64", ("n_instances", 4)),
    ArraySpec("bbox_norm_coords", "float64", ("n_instances", 4)),
    ArraySpec("source_kind_codes", "int8", ("n_instances",)),
    ArraySpec("manual_edit_flags", "bool", ("n_instances",)),
    ArraySpec("source_detect_row_index", "int32", ("n_instances",)),
    ArraySpec("instance_key", "uint64", ("n_instances",), required=False),
    ArraySpec("frame_counts", "int32", ("n_frames",)),
    ArraySpec("reason_bytes", "uint8", ("n_instances", "width")),
    ArraySpec("confidence_scores", "float32", ("n_instances",), required=False),
    ArraySpec("class_ids", "int32", ("n_instances",), required=False),
    ArraySpec("review_notes", "string", ("n_instances",), required=False),
)

REFINED_DETECT_SPEC = StageSpec(
    stage_name="refined_detect",
    zarr_group="refined_detect_runs/<run>/",
    specs=(),
    subgroups={
        "source_detections": _REFINED_DETECT_SOURCE_DETECTIONS_ARRAYS,
        "instances": _REFINED_DETECT_INSTANCES_ARRAYS,
    },
)

CROP_SPEC = StageSpec(
    stage_name="crop",
    zarr_group="crop_runs/<run>/",
    specs=(
        ArraySpec("roi_images", "uint8", ("n_rois", "h", "w"), required=False),
        ArraySpec("roi_coordinates_full", "int32", ("n_rois", 2)),
        ArraySpec("roi_coordinates_ds", "int32", ("n_rois", 2), required=False),
        ArraySpec("bbox_img_xyxy", "float32", ("n_rois", 4), required=False),
        ArraySpec("bbox_norm_coords", "float32", ("n_rois", 4)),
        ArraySpec("bbox_crop_norm_coords", "float32", ("n_rois", 4), required=False),
        ArraySpec("frame_indices", "int32", ("n_rois",)),
        ArraySpec("frame_counts", "int32", ("n_frames",)),
        ArraySpec("detection_indices", "int32", ("n_rois",)),
        ArraySpec(
            "source_refined_row_ids",
            "int64",
            ("n_rois",),
            required=False,
            description="Stable refined detection row IDs copied from the source refined instances rowset.",
        ),
        ArraySpec(
            "source_detect_row_index",
            "int32",
            ("n_rois",),
            required=False,
            description="Raw detect row lineage copied from refined instances when available; -1 for manual rows.",
        ),
        ArraySpec(
            "instance_key",
            "uint64",
            ("n_rois",),
            required=False,
            description="Detection instance key copied verbatim from the source detect/refined-detect row when available.",
        ),
        ArraySpec("detection_source", "int8", ("n_rois",), required=False),
    ),
)

KEYPOINTS_SPEC = StageSpec(
    stage_name="keypoints",
    zarr_group="keypoints_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",)),
        ArraySpec("frame_counts", "int32", ("n_frames",)),
        ArraySpec("n_rois", "int32", ("n_frames",), required=False, description="Legacy alias of frame_counts."),
        ArraySpec("detection_indices", "int32", ("n_rois",)),
        ArraySpec("source_refined_row_ids", "int64", ("n_rois",), required=False),
        ArraySpec("source_detect_row_index", "int32", ("n_rois",), required=False),
        ArraySpec("instance_key", "uint64", ("n_rois",), required=False),
        ArraySpec("keypoints_roi", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("keypoints_img", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("keypoints_norm", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("heading", "float64", ("n_rois",)),
        ArraySpec("confidence", "float64", ("n_rois",)),
        ArraySpec("keypoint_confidences", "float64", ("n_rois", "n_keypoints")),
        ArraySpec("effective_threshold", "float64", ("n_rois",)),
        ArraySpec("effective_se2_radius", "float64", ("n_rois",)),
        ArraySpec("detection_success", "bool", ("n_rois",)),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("heading_finite", "bool", ("n_rois",)),
        ArraySpec("heading_usable", "bool", ("n_rois",)),
        ArraySpec(
            "heading_delta_prev_deg",
            "float32",
            ("n_rois",),
            required=False,
            description="Circular absolute delta to the previous usable temporal heading neighbor; omitted when temporal-heading review is disabled (for example sampled imports).",
        ),
        ArraySpec(
            "heading_delta_next_deg",
            "float32",
            ("n_rois",),
            required=False,
            description="Circular absolute delta to the next usable temporal heading neighbor; omitted when temporal-heading review is disabled (for example sampled imports).",
        ),
        ArraySpec(
            "heading_temporal_outlier",
            "bool",
            ("n_rois",),
            required=False,
            description="True when both temporal heading deltas exceed the configured outlier threshold; omitted when temporal-heading review is disabled (for example sampled imports).",
        ),
        ArraySpec("n_keypoints", "int32", ("n_frames",)),
        ArraySpec(
            "triangle_angles",
            "float64",
            ("n_rois", 3),
            required=False,
            description="Traditional triangle compatibility/QC output; not emitted by general YOLO pose skeletons.",
        ),
        ArraySpec(
            "triangle_angles_raw",
            "float64",
            ("n_rois", 3),
            required=False,
            description="Traditional triangle compatibility/QC output; not emitted by general YOLO pose skeletons.",
        ),
        ArraySpec(
            "triangle_area",
            "float64",
            ("n_rois",),
            required=False,
            description="Traditional triangle compatibility/QC output; not emitted by general YOLO pose skeletons.",
        ),
    ),
)

REFINED_KEYPOINTS_SPEC = StageSpec(
    stage_name="refined_keypoints",
    zarr_group="refined_keypoints_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",)),
        ArraySpec("frame_counts", "int32", ("n_frames",)),
        ArraySpec("n_rois", "int32", ("n_frames",), required=False, description="Legacy alias of frame_counts."),
        ArraySpec("detection_indices", "int32", ("n_rois",)),
        ArraySpec("source_refined_row_ids", "int64", ("n_rois",), required=False),
        ArraySpec("source_detect_row_index", "int32", ("n_rois",), required=False),
        ArraySpec("instance_key", "uint64", ("n_rois",), required=False),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("retune_id", "int32", ("n_rois",), description="-1 indicates none."),
        ArraySpec("keypoints_roi", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("keypoints_img", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("keypoints_norm", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("heading", "float64", ("n_rois",)),
        ArraySpec("confidence", "float64", ("n_rois",)),
        ArraySpec(
            "keypoint_confidences",
            "float64",
            ("n_rois", "n_keypoints"),
            required=False,
            description="Copied from source run when present.",
        ),
        ArraySpec(
            "effective_threshold",
            "float64",
            ("n_rois",),
            required=False,
            description="Present when available in the source run.",
        ),
        ArraySpec(
            "effective_se2_radius",
            "float64",
            ("n_rois",),
            required=False,
            description="Present when available in the source run.",
        ),
        ArraySpec("triangle_area", "float64", ("n_rois",)),
        ArraySpec("min_angle", "float64", ("n_rois",)),
        ArraySpec("triangle_angles", "float64", ("n_rois", 3)),
        ArraySpec("quality_labels", "int8", ("n_rois",)),
        ArraySpec("refined_success", "bool", ("n_rois",)),
        ArraySpec("source_success", "bool", ("n_rois",)),
        ArraySpec("flip_corrected", "bool", ("n_rois",)),
        ArraySpec("heading_finite", "bool", ("n_rois",)),
        ArraySpec("heading_usable", "bool", ("n_rois",)),
        ArraySpec("confidence_valid", "bool", ("n_rois",)),
        ArraySpec("geometry_valid", "bool", ("n_rois",)),
        ArraySpec("usable_keypoints", "bool", ("n_rois",)),
        ArraySpec(
            "edge_pairs",
            "int16/int32",
            ("n_edges", 2),
            required=False,
            description="Optional skeleton edge index pairs for per-ROI distance auditing.",
        ),
        ArraySpec(
            "edge_distances",
            "float32",
            ("n_rois", "n_edges"),
            required=False,
            description="Optional per-ROI Euclidean edge distances in ROI pixels.",
        ),
        ArraySpec(
            "edge_distances_norm",
            "float32",
            ("n_rois", "n_edges"),
            required=False,
            description="Optional per-ROI edge distances normalized by ROI diagonal.",
        ),
        ArraySpec(
            "edge_distance_valid",
            "bool",
            ("n_rois", "n_edges"),
            required=False,
            description="Optional validity mask for per-ROI edge distances.",
        ),
        ArraySpec(
            "derived_metric_values",
            "float32",
            ("n_rois", "n_metrics"),
            required=False,
            description="Optional schema-driven named derived keypoint metrics in ROI pixels.",
        ),
        ArraySpec(
            "derived_metric_values_norm",
            "float32",
            ("n_rois", "n_metrics"),
            required=False,
            description="Optional derived metrics normalized by the configured metric schema normalization.",
        ),
        ArraySpec(
            "derived_metric_valid",
            "bool",
            ("n_rois", "n_metrics"),
            required=False,
            description="Optional validity mask for schema-driven derived keypoint metrics.",
        ),
        ArraySpec(
            "edit_applied",
            "bool",
            ("n_rois",),
            required=False,
            description="Current writer emits this manual-edit marker; older refined keypoint runs may omit it.",
        ),
        ArraySpec(
            "reason_bytes",
            "uint8",
            ("n_rois", "width"),
            required=False,
            description="TensorStore-safe reason encoding; older runs may only carry reason strings.",
        ),
        ArraySpec(
            "reason",
            "string",
            ("n_rois",),
            required=False,
            description="Legacy read-only compatibility column; new runs write reason_bytes only.",
        ),
        ArraySpec(
            "failure_indices",
            "int32",
            ("n_failures",),
            required=False,
            description="Derived source-failure index list; omitted by older refined keypoint runs.",
        ),
    ),
)

EYE_MASKS_SPEC = StageSpec(
    stage_name="eye_masks",
    zarr_group="eye_masks_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",), required=False),
        ArraySpec("frame_counts", "int32", ("n_frames",), required=False),
        ArraySpec("detection_indices", "int32", ("n_rois",), required=False),
        ArraySpec("source_crop_row_ids", "int64", ("n_rois",)),
        ArraySpec("source_refined_row_ids", "int64", ("n_rois",), required=False),
        ArraySpec("source_detect_row_index", "int32", ("n_rois",), required=False),
        ArraySpec("instance_key", "uint64", ("n_rois",), required=False),
        ArraySpec("masks_roi", "uint8", ("n_rois", 2, "H", "W")),
        ArraySpec("mask_probs_roi", "float16/float32/uint8", ("n_rois", 2, "H", "W"), required=False),
        ArraySpec("mask_scores", "float32", ("n_rois",), required=False),
        ArraySpec("ellipse_params", "float32", ("n_rois", 2, 5)),
        ArraySpec("ellipse_success", "bool", ("n_rois", 2)),
        ArraySpec("eye_separation", "float32", ("n_rois",)),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("contour_left_ptr", "int32", ("n_rois",)),
        ArraySpec("contour_left_len", "int32", ("n_rois",)),
        ArraySpec("contour_right_ptr", "int32", ("n_rois",)),
        ArraySpec("contour_right_len", "int32", ("n_rois",)),
        ArraySpec("contours_left", "float32", ("n_points", 2)),
        ArraySpec("contours_right", "float32", ("n_points", 2)),
        ArraySpec("reason", "string", ("n_rois",)),
    ),
)

_REFINED_EYE_MASK_METRICS: Tuple[ArraySpec, ...] = (
    ArraySpec("area_refined", "float32", ("n_rois", 2)),
    ArraySpec("area_source", "float32", ("n_rois", 2)),
    ArraySpec("area_zscore", "float32", ("n_rois", 2)),
    ArraySpec("area_delta_vs_source", "float32", ("n_rois", 2)),
    ArraySpec("area_ratio_vs_source", "float32", ("n_rois", 2)),
    ArraySpec("centroid_error", "float32", ("n_rois", 2)),
    ArraySpec("symmetry_offsets", "float32", ("n_rois", 2)),
    ArraySpec("axis_ratio", "float32", ("n_rois", 2)),
    ArraySpec("circularity", "float32", ("n_rois", 2)),
    ArraySpec("probability_mean", "float32", ("n_rois", 2)),
    ArraySpec("probability_max", "float32", ("n_rois", 2)),
    ArraySpec("probability_var", "float32", ("n_rois", 2)),
    ArraySpec("probability_high_fraction", "float32", ("n_rois", 2)),
    ArraySpec("area_union_refined", "float32", ("n_rois",)),
    ArraySpec("area_union_source", "float32", ("n_rois",)),
    ArraySpec("area_ratio_left_right", "float32", ("n_rois",)),
    ArraySpec("area_diff_left_right", "float32", ("n_rois",)),
    ArraySpec("area_union_delta", "float32", ("n_rois",)),
    ArraySpec("area_union_ratio", "float32", ("n_rois",)),
    ArraySpec("symmetry_sum", "float32", ("n_rois",)),
    ArraySpec("symmetry_abs_diff", "float32", ("n_rois",)),
    ArraySpec("separation_refined", "float32", ("n_rois",)),
    ArraySpec("separation_keypoint", "float32", ("n_rois",)),
    ArraySpec("separation_delta", "float32", ("n_rois",)),
    ArraySpec("connectivity_flags", "uint8", ("n_rois",)),
    ArraySpec("smoothing_flags", "uint8", ("n_rois", 2)),
    ArraySpec("pixels_reassigned", "int32", ("n_rois",)),
    ArraySpec("probabilities_used", "bool", ("n_rois",)),
    ArraySpec("filter_flags", "bool", ("n_rois", 2)),
    ArraySpec("reason_bytes", "uint8", ("n_rois", "width")),
)

REFINED_EYE_MASKS_SPEC = StageSpec(
    stage_name="refined_eye_masks",
    zarr_group="refined_eye_masks_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",), required=False),
        ArraySpec("frame_counts", "int32", ("n_frames",), required=False),
        ArraySpec("detection_indices", "int32", ("n_rois",), required=False),
        ArraySpec("source_crop_row_ids", "int64", ("n_rois",)),
        ArraySpec("source_refined_row_ids", "int64", ("n_rois",), required=False),
        ArraySpec("source_detect_row_index", "int32", ("n_rois",), required=False),
        ArraySpec("instance_key", "uint64", ("n_rois",), required=False),
        ArraySpec("masks_roi", "uint8", ("n_rois", 2, "H", "W")),
        ArraySpec("edit_applied", "bool", ("n_rois", 2)),
        ArraySpec("ellipse_params", "float32", ("n_rois", 2, 5)),
        ArraySpec("ellipse_success", "bool", ("n_rois", 2)),
        ArraySpec("eye_separation", "float32", ("n_rois",)),
        ArraySpec("retune_id", "int32", ("n_rois",), required=False),
        ArraySpec("mask_probs_roi_refined", "float16", ("n_rois", 2, "H", "W"), required=False),
        ArraySpec("contour_left_ptr", "int32", ("n_rois",)),
        ArraySpec("contour_left_len", "int32", ("n_rois",)),
        ArraySpec("contour_right_ptr", "int32", ("n_rois",)),
        ArraySpec("contour_right_len", "int32", ("n_rois",)),
        ArraySpec("contours_left", "float32", ("n_points", 2)),
        ArraySpec("contours_right", "float32", ("n_points", 2)),
    ),
    subgroups={"metrics": _REFINED_EYE_MASK_METRICS},
)

_SUBJECT_MASK_METRICS: Tuple[ArraySpec, ...] = (
    ArraySpec("prob_max", "float32", ("n_rois", "n_channels")),
    ArraySpec("mask_present", "bool", ("n_rois", "n_channels")),
    ArraySpec("area_px", "float32", ("n_rois", "n_channels"), required=False),
    ArraySpec("centroid_xy", "float32", ("n_rois", "n_channels", 2), required=False),
    ArraySpec("centroid_valid", "bool", ("n_rois", "n_channels"), required=False),
    ArraySpec("bbox_xyxy", "float32", ("n_rois", "n_channels", 4), required=False),
    ArraySpec("bbox_valid", "bool", ("n_rois", "n_channels"), required=False),
)

SUBJECT_MASKS_SPEC = StageSpec(
    stage_name="subject_masks",
    zarr_group="subject_mask_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",), required=False),
        ArraySpec("frame_counts", "int32", ("n_frames",), required=False),
        ArraySpec("detection_indices", "int32", ("n_rois",), required=False),
        ArraySpec("source_crop_row_ids", "int64", ("n_rois",)),
        ArraySpec("source_refined_row_ids", "int64", ("n_rois",), required=False),
        ArraySpec("source_detect_row_index", "int32", ("n_rois",), required=False),
        ArraySpec("instance_key", "uint64", ("n_rois",), required=False),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("masks_roi", "uint8", ("n_rois", "n_channels", "H", "W"), required=False),
        ArraySpec(
            "mask_probs_roi",
            "float16/float32/uint8",
            ("n_rois", "n_channels", "H", "W"),
            required=False,
            description="Required for standalone runs; composite runs resolve base plus delta payloads.",
        ),
        ArraySpec("available_channels", "bool", ("n_channels",)),
    ),
    subgroups={"metrics": _SUBJECT_MASK_METRICS},
)

_REFINED_SUBJECT_MASK_METRICS: Tuple[ArraySpec, ...] = (
    ArraySpec("mask_present", "bool", ("n_rois", "n_channels")),
    ArraySpec("area_px", "float32", ("n_rois", "n_channels")),
    ArraySpec("centroid_xy", "float32", ("n_rois", "n_channels", 2)),
    ArraySpec("centroid_valid", "bool", ("n_rois", "n_channels")),
    ArraySpec("bbox_xyxy", "float32", ("n_rois", "n_channels", 4)),
    ArraySpec("bbox_valid", "bool", ("n_rois", "n_channels")),
)

REFINED_SUBJECT_COMPONENT_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("mask_present", "bool", ("n_rois",)),
    ArraySpec("area_px", "float32", ("n_rois",)),
    ArraySpec("edit_applied", "bool", ("n_rois",)),
    ArraySpec("reason_bytes", "uint8", ("n_rois", "width")),
)

REFINED_SUBJECT_COMPONENT_METRICS: Tuple[ArraySpec, ...] = (
    ArraySpec("component_count", "int32", ("n_rois",)),
    ArraySpec("largest_component_fraction", "float32", ("n_rois",)),
    ArraySpec("hole_count", "int32", ("n_rois",)),
    ArraySpec("hole_area_fraction", "float32", ("n_rois",)),
    ArraySpec("sigma_noise", "float32", ("n_rois",)),
    ArraySpec("curvature_var", "float32", ("n_rois",)),
    ArraySpec("ipr", "float32", ("n_rois",)),
    ArraySpec("solidity", "float32", ("n_rois",)),
)

REFINED_SUBJECT_EYE_GEOMETRY: Tuple[ArraySpec, ...] = (
    ArraySpec("ellipse_params", "float32", ("n_rois", 5)),
    ArraySpec("ellipse_success", "bool", ("n_rois",)),
)

REFINED_SUBJECT_EYE_CONTOURS: Tuple[ArraySpec, ...] = (
    ArraySpec("ptr", "int64", ("n_rois",)),
    ArraySpec("len", "int32", ("n_rois",)),
    ArraySpec("points_xy", "float32", ("n_points", 2)),
)

REFINED_SUBJECT_EYE_PAIR_METRICS: Tuple[ArraySpec, ...] = (
    ArraySpec("separation_px", "float32", ("n_rois",)),
    ArraySpec("separation_valid", "bool", ("n_rois",)),
)

REFINED_SUBJECT_MASKS_SPEC = StageSpec(
    stage_name="refined_subject_masks",
    zarr_group="refined_subject_masks_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",), required=False),
        ArraySpec("frame_counts", "int32", ("n_frames",), required=False),
        ArraySpec("detection_indices", "int32", ("n_rois",), required=False),
        ArraySpec("source_crop_row_ids", "int64", ("n_rois",)),
        ArraySpec("source_refined_row_ids", "int64", ("n_rois",), required=False),
        ArraySpec("source_detect_row_index", "int32", ("n_rois",), required=False),
        ArraySpec("instance_key", "uint64", ("n_rois",), required=False),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("masks_roi", "uint8", ("n_rois", "n_channels", "H", "W")),
        ArraySpec("available_channels", "bool", ("n_channels",)),
        ArraySpec("edit_applied", "bool", ("n_rois", "n_channels")),
        ArraySpec("reason_bytes", "uint8", ("n_rois", "width"), required=False),
        ArraySpec(
            "reason",
            "string",
            ("n_rois",),
            required=False,
            description="Legacy read-only compatibility column; new runs write reason_bytes only.",
        ),
    ),
    subgroups={"metrics": _REFINED_SUBJECT_MASK_METRICS},
)

ARENA_ASSIGNMENT_SPEC = StageSpec(
    stage_name="arena_assignment",
    zarr_group="arena_assignment_runs/<run>/",
    specs=(
        ArraySpec("arena_ids", "int32", ("n_detections",)),
        ArraySpec("n_detections_per_arena", "int32", ("n_frames", "n_arenas")),
        ArraySpec("confidence", "float32", ("n_detections",), required=False),
    ),
)

TRACKING_SPEC = StageSpec(
    stage_name="tracking",
    zarr_group="tracking_runs/<run>/",
    specs=(
        ArraySpec("track_ids", "int32", ("n_detections",)),
        ArraySpec("arena_ids", "int32", ("n_detections",)),
        ArraySpec("frame_indices", "int32", ("n_detections",)),
        ArraySpec("source_row_indices", "int32", ("n_detections",)),
        ArraySpec("instance_key", "uint64", ("n_detections",), required=False),
        ArraySpec("source_refined_row_ids", "int64", ("n_detections",), required=False),
        ArraySpec("source_detect_row_index", "int32", ("n_detections",), required=False),
        ArraySpec("tracking_confidence", "float32", ("n_detections",), required=False),
        ArraySpec("tracking_status", "int8", ("n_detections",), required=False),
        ArraySpec("association_cost", "float32", ("n_detections",), required=False),
        ArraySpec("track_ids_present", "int32", ("n_tracks",)),
        ArraySpec("track_arena_ids", "int32", ("n_tracks",)),
    ),
)

STAGES: Dict[str, StageSpec] = {
    RAW_VIDEO_SPEC.stage_name: RAW_VIDEO_SPEC,
    STIMULUS_SPEC.stage_name: STIMULUS_SPEC,
    DETECTION_PROFILE_SPEC.stage_name: DETECTION_PROFILE_SPEC,
    EYE_MASK_PROFILE_SPEC.stage_name: EYE_MASK_PROFILE_SPEC,
    KEYPOINT_PROFILE_SPEC.stage_name: KEYPOINT_PROFILE_SPEC,
    SUBJECT_SHAPE_SPEC.stage_name: SUBJECT_SHAPE_SPEC,
    TAIL_POSTURE_VIEW_SPEC.stage_name: TAIL_POSTURE_VIEW_SPEC,
    BOUT_CLASSIFICATION_SPEC.stage_name: BOUT_CLASSIFICATION_SPEC,
    TAIL_KINEMATICS_SPEC.stage_name: TAIL_KINEMATICS_SPEC,
    TRACK_KINEMATICS_SPEC.stage_name: TRACK_KINEMATICS_SPEC,
    EYE_ANGLE_SPEC.stage_name: EYE_ANGLE_SPEC,
    BOUT_KINEMATICS_SPEC.stage_name: BOUT_KINEMATICS_SPEC,
    BACKGROUND_SPEC.stage_name: BACKGROUND_SPEC,
    DETECT_SPEC.stage_name: DETECT_SPEC,
    DETECT_QUALITY_SPEC.stage_name: DETECT_QUALITY_SPEC,
    REFINED_DETECT_SPEC.stage_name: REFINED_DETECT_SPEC,
    CROP_SPEC.stage_name: CROP_SPEC,
    KEYPOINTS_SPEC.stage_name: KEYPOINTS_SPEC,
    REFINED_KEYPOINTS_SPEC.stage_name: REFINED_KEYPOINTS_SPEC,
    EYE_MASKS_SPEC.stage_name: EYE_MASKS_SPEC,
    REFINED_EYE_MASKS_SPEC.stage_name: REFINED_EYE_MASKS_SPEC,
    SUBJECT_MASKS_SPEC.stage_name: SUBJECT_MASKS_SPEC,
    REFINED_SUBJECT_MASKS_SPEC.stage_name: REFINED_SUBJECT_MASKS_SPEC,
    ARENA_ASSIGNMENT_SPEC.stage_name: ARENA_ASSIGNMENT_SPEC,
    TRACKING_SPEC.stage_name: TRACKING_SPEC,
}

__all__ = [
    "ArraySpec",
    "StageSpec",
    "ValidationResult",
    "RAW_VIDEO_SPEC",
    "STIMULUS_SPEC",
    "DETECTION_PROFILE_SPEC",
    "EYE_MASK_PROFILE_SPEC",
    "KEYPOINT_PROFILE_SPEC",
    "SUBJECT_SHAPE_SPEC",
    "TAIL_POSTURE_VIEW_SPEC",
    "BOUT_CLASSIFICATION_SPEC",
    "TAIL_KINEMATICS_SPEC",
    "TRACK_KINEMATICS_SPEC",
    "EYE_ANGLE_SPEC",
    "BOUT_KINEMATICS_SPEC",
    "BACKGROUND_SPEC",
    "DETECT_SPEC",
    "DETECT_QUALITY_SPEC",
    "REFINED_DETECT_SPEC",
    "CROP_SPEC",
    "KEYPOINTS_SPEC",
    "REFINED_KEYPOINTS_SPEC",
    "EYE_MASKS_SPEC",
    "REFINED_EYE_MASKS_SPEC",
    "SUBJECT_MASKS_SPEC",
    "REFINED_SUBJECT_MASKS_SPEC",
    "REFINED_SUBJECT_COMPONENT_ARRAYS",
    "REFINED_SUBJECT_COMPONENT_METRICS",
    "REFINED_SUBJECT_EYE_GEOMETRY",
    "REFINED_SUBJECT_EYE_CONTOURS",
    "REFINED_SUBJECT_EYE_PAIR_METRICS",
    "ARENA_ASSIGNMENT_SPEC",
    "TRACKING_SPEC",
    "STAGES",
    "array_specs_by_name",
    "describe_array",
    "required_array_names",
    "required_array_specs",
    "validate_run",
]
