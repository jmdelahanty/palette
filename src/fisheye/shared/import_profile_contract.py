"""Classify and validate Palette singleton import profiles.

Import is intentionally a singleton surface: active writers stamp root attrs
and ``raw_video`` attrs/arrays instead of creating timestamped import run
groups. This module defines the read-only contract used to distinguish the
active metadata-only analysis profile, the active PyNvVC sampled-training pixel
profile, and the historical Decord pixel profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import zarr_attrs_dict, zarr_child_group


IMPORT_PROFILE_SCHEMA_ID = "palette.import_profile_contract.v1"

PROFILE_METADATA_ONLY_ANALYSIS = "metadata_only_analysis"
PROFILE_SAMPLED_TRAINING_PYNVVC_LUMA = "sampled_training_pynvvc_luma"
PROFILE_LEGACY_DECORD_TRAINING_OR_FULL = "legacy_decord_training_or_full"
PROFILE_UNKNOWN = "unknown_raw_video_profile"
PROFILE_MISSING_RAW_VIDEO = "missing_raw_video"

IMPORT_PROFILE_NAMES = (
    PROFILE_METADATA_ONLY_ANALYSIS,
    PROFILE_SAMPLED_TRAINING_PYNVVC_LUMA,
    PROFILE_LEGACY_DECORD_TRAINING_OR_FULL,
    PROFILE_UNKNOWN,
    PROFILE_MISSING_RAW_VIDEO,
)

SOURCE_VIDEO_FINGERPRINT_ATTRS = (
    "source_video_fingerprint",
    "source_video_stat_fingerprint",
    "video_fingerprint",
)
SOURCE_VIDEO_PATH_ATTRS = (
    "source_video_path",
    "source_path",
    "source_video",
)
SOURCE_H5_FINGERPRINT_ATTRS = (
    "source_h5_fingerprint",
    "source_h5_stat_fingerprint",
    "h5_fingerprint",
    "protocol_fingerprint",
)
SOURCE_H5_PATH_ATTRS = (
    "source_h5_path",
    "source_h5",
    "h5_path",
)
COLORIMETRY_ATTRS = (
    "color_range",
    "color_space",
    "color_matrix",
    "color_transfer",
    "color_primaries",
    "container_color_range_observed",
    "video_color_range",
)


@dataclass(frozen=True)
class ImportProfileReport:
    """Machine-readable read-only import profile validation result."""

    schema_id: str
    profile: str
    status: str
    reason_codes: tuple[str, ...] = ()
    required_missing: tuple[str, ...] = ()
    recommended_missing: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    arrays_present: tuple[str, ...] = ()
    attrs_observed: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _norm(value: Any) -> str:
    normalized = normalize_attr(value)
    return "" if normalized is None else str(normalized).strip()


def _lower(value: Any) -> str:
    return _norm(value).lower()


def _first_attr(attrs_list: Sequence[Mapping[str, Any]], names: Iterable[str]) -> Any | None:
    for attrs in attrs_list:
        for name in names:
            value = attrs.get(name)
            if value not in (None, ""):
                return value
    return None


def _has_any_attr(attrs_list: Sequence[Mapping[str, Any]], names: Iterable[str]) -> bool:
    return _first_attr(attrs_list, names) is not None


def _missing_attrs(
    attrs_list: Sequence[Mapping[str, Any]],
    names: Iterable[str],
    *,
    prefix: str,
) -> list[str]:
    return [f"{prefix}.{name}" for name in names if not _has_any_attr(attrs_list, (name,))]


def _child_exists(group: Any | None, name: str) -> bool:
    if group is None:
        return False
    try:
        return name in group
    except Exception:
        return False


def _array_present(group: Any | None, name: str) -> bool:
    if not _child_exists(group, name):
        return False
    try:
        return hasattr(group[name], "shape")
    except Exception:
        return False


def _raw_array_names(raw_video: Any | None) -> tuple[str, ...]:
    names = []
    for name in ("images_full", "images_ds", "images_ds_rgb", "original_frame_indices", "timestamps"):
        if _array_present(raw_video, name):
            names.append(f"raw_video/{name}")
    return tuple(names)


def _has_source_video_fingerprint(attrs_list: Sequence[Mapping[str, Any]]) -> bool:
    return _has_any_attr(attrs_list, SOURCE_VIDEO_FINGERPRINT_ATTRS)


def _has_h5_fingerprint(attrs_list: Sequence[Mapping[str, Any]]) -> bool:
    return _has_any_attr(attrs_list, SOURCE_H5_FINGERPRINT_ATTRS)


def _has_colorimetry(attrs_list: Sequence[Mapping[str, Any]]) -> bool:
    return _has_any_attr(attrs_list, COLORIMETRY_ATTRS)


def _known_bad_or_unknown_value(value: Any) -> bool:
    text = _lower(value)
    return text in {"", "unknown", "none", "null", "unspecified"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def _json_safe_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in attrs.items()}


def _classify_profile(
    *,
    root_attrs: Mapping[str, Any],
    raw_attrs: Mapping[str, Any],
    raw_arrays: Sequence[str],
) -> str:
    fields = " ".join(
        _lower(value)
        for value in (
            raw_attrs.get("import_method"),
            raw_attrs.get("import_mode"),
            raw_attrs.get("import_stage"),
            raw_attrs.get("decode_backend"),
            raw_attrs.get("decode_backend_family"),
            raw_attrs.get("decode_contract_status"),
            raw_attrs.get("pixel_contract_name"),
            raw_attrs.get("roi_pixel_contract_name"),
            root_attrs.get("zarr_purpose"),
            root_attrs.get("zarr_use"),
        )
    )
    if (
        "pynvvc_luma_sampled_training" in fields
        or "pynvvc_luma" in fields
        or "pynvvc-luma" in fields
        or "canonical_orange_mono_pynvvc_luma" in fields
        or "orange_mono_pynvvc_luma" in fields
    ):
        return PROFILE_SAMPLED_TRAINING_PYNVVC_LUMA

    if "metadata_only" in fields:
        return PROFILE_METADATA_ONLY_ANALYSIS

    if (
        "fisheye.capture.import_video" in fields
        or "legacy_decord" in fields
        or "decord" in fields
        or (("complete" in fields or "full" in fields) and bool(raw_arrays))
    ):
        return PROFILE_LEGACY_DECORD_TRAINING_OR_FULL

    return PROFILE_UNKNOWN


def _profile_required(
    profile: str,
    *,
    raw_video: Any | None,
    root_attrs: Mapping[str, Any],
    raw_attrs: Mapping[str, Any],
) -> list[str]:
    attrs = (raw_attrs, root_attrs)
    missing: list[str] = []
    if raw_video is None:
        return ["raw_video"]

    if profile == PROFILE_METADATA_ONLY_ANALYSIS:
        missing.extend(
            _missing_attrs(
                attrs,
                ("import_method", "import_stage", "total_frames", "fps"),
                prefix="raw_video.attrs",
            )
        )
        if not _has_any_attr(attrs, SOURCE_VIDEO_PATH_ATTRS):
            missing.append("root_or_raw_video.attrs.source_video_path")
        return missing

    if profile == PROFILE_SAMPLED_TRAINING_PYNVVC_LUMA:
        for array_name in ("images_full", "original_frame_indices"):
            if not _array_present(raw_video, array_name):
                missing.append(f"raw_video/{array_name}")
        missing.extend(
            _missing_attrs(
                attrs,
                (
                    "import_method",
                    "decode_backend",
                    "frame_step",
                    "source_frame_count",
                    "pixel_contract_name",
                ),
                prefix="raw_video.attrs",
            )
        )
        if not _has_any_attr(attrs, SOURCE_VIDEO_PATH_ATTRS):
            missing.append("root_or_raw_video.attrs.source_video_path")
        return missing

    if profile == PROFILE_LEGACY_DECORD_TRAINING_OR_FULL:
        if not (_array_present(raw_video, "images_full") or _array_present(raw_video, "images_ds")):
            missing.append("raw_video/images_full_or_images_ds")
        if not _has_any_attr(attrs, ("import_method", "import_stage", "import_mode")):
            missing.append("raw_video.attrs.import_method_or_stage")
        return missing

    return missing


def classify_import_profile(root: Any) -> ImportProfileReport:
    """Classify a Palette Zarr root's singleton import profile.

    The function is intentionally read-only and accepts Zarr-like fake groups
    for unit tests. It validates profile-specific required metadata and reports
    missing recommended provenance without mutating the store.
    """

    root_attrs = zarr_attrs_dict(root)
    raw_video = zarr_child_group(root, "raw_video")
    if raw_video is None:
        return ImportProfileReport(
            schema_id=IMPORT_PROFILE_SCHEMA_ID,
            profile=PROFILE_MISSING_RAW_VIDEO,
            status="incomplete",
            reason_codes=("MISSING_RAW_VIDEO",),
            required_missing=("raw_video",),
            attrs_observed={"root": _json_safe_attrs(root_attrs), "raw_video": {}},
        )

    raw_attrs = zarr_attrs_dict(raw_video)
    raw_arrays = _raw_array_names(raw_video)
    profile = _classify_profile(root_attrs=root_attrs, raw_attrs=raw_attrs, raw_arrays=raw_arrays)
    required_missing = _profile_required(
        profile,
        raw_video=raw_video,
        root_attrs=root_attrs,
        raw_attrs=raw_attrs,
    )
    recommended_missing: list[str] = []
    warnings: list[str] = []
    reason_codes: list[str] = []
    attrs = (raw_attrs, root_attrs)

    if profile == PROFILE_UNKNOWN:
        reason_codes.append("UNKNOWN_IMPORT_PROFILE")
        warnings.append("raw_video exists but import attrs do not match a supported singleton profile")

    if not _has_source_video_fingerprint(attrs):
        recommended_missing.append("root_or_raw_video.attrs.source_video_fingerprint")
        reason_codes.append("MISSING_SOURCE_VIDEO_FINGERPRINT")

    if _has_any_attr(attrs, SOURCE_H5_PATH_ATTRS) and not _has_h5_fingerprint(attrs):
        recommended_missing.append("root_or_raw_video.attrs.source_h5_fingerprint")
        reason_codes.append("MISSING_SOURCE_H5_FINGERPRINT")

    if not _has_colorimetry(attrs):
        recommended_missing.append("root_or_raw_video.attrs.colorimetry")
        reason_codes.append("MISSING_COLORIMETRY")

    for attr_name in ("video_codec", "codec", "video_pix_fmt", "pix_fmt"):
        value = _first_attr(attrs, (attr_name,))
        if value is not None and _known_bad_or_unknown_value(value):
            warnings.append(f"{attr_name} is {value!r}")
            reason_codes.append(f"UNKNOWN_{attr_name.upper()}")

    if required_missing:
        status = "incomplete"
        reason_codes.insert(0, "MISSING_REQUIRED_IMPORT_PROFILE_FIELDS")
    elif profile == PROFILE_UNKNOWN:
        status = "unknown"
    elif recommended_missing or warnings:
        status = "warning"
    else:
        status = "ok"

    return ImportProfileReport(
        schema_id=IMPORT_PROFILE_SCHEMA_ID,
        profile=profile,
        status=status,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        required_missing=tuple(required_missing),
        recommended_missing=tuple(dict.fromkeys(recommended_missing)),
        warnings=tuple(dict.fromkeys(warnings)),
        arrays_present=raw_arrays,
        attrs_observed={
            "root": _json_safe_attrs(root_attrs),
            "raw_video": _json_safe_attrs(raw_attrs),
        },
    )
