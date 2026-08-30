"""Model-bound component-area support profiles for subject-mask refinement.

The profile is a scientific policy, not an anatomical truth claim.  It records
the smallest normalized positive component area represented by the approved
training masks for one exact model artifact.  Production refinement can then
reject predictions below that observed support boundary while retaining the
raw inference surface unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from fisheye.shared.runtime_config import runtime_config_dirs
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

COMPONENT_AREA_SUPPORT_SCHEMA_ID = "palette.subject_mask.component_area_support_profile"
COMPONENT_AREA_SUPPORT_SCHEMA_VERSION = 1
COMPONENT_AREA_SUPPORT_DERIVATION_METHOD = (
    "minimum_positive_normalized_area_from_approved_training_masks_v1"
)
COMPONENT_AREA_SUPPORT_CONFIG_DIR = "subject_mask_component_support"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_BINDING_FIELDS = {
    "registry_set_id",
    "registry_run_id",
    "artifact_sha256",
    "label_schema_id",
}
_TRAINING_EVIDENCE_FIELDS = {
    "training_manifest_sha256",
    "approved_source_run",
    "approved_source_label_schema_id",
    "approved_source_count",
    "approved_row_count",
    "reference_mask_shape_hw",
    "metadata_read_mode",
    "area_source",
}
_COMPONENT_FAMILY_FIELDS = {
    "family",
    "applies_to",
    "minimum_area_px_reference",
    "positive_label_count",
    "observed_minimum_by_source_label",
}
_SUPPORTED_COMPONENTS = {
    "subject_body",
    "eyes_union",
    "eye_left",
    "eye_right",
    "swim_bladder",
}


class SubjectMaskComponentSupportError(RuntimeError):
    """Raised when component-area support evidence is missing or invalid."""


def _require_text(value: object, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SubjectMaskComponentSupportError(f"{name} must be nonempty text.")
    return text


def _require_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or int(value) <= 0:
        raise SubjectMaskComponentSupportError(f"{name} must be a positive integer.")
    return int(value)


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_text(value, name=name).lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise SubjectMaskComponentSupportError(
            f"{name} must be a lowercase hexadecimal SHA-256 digest."
        )
    return digest


def _require_exact_fields(
    value: object,
    fields: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SubjectMaskComponentSupportError(
            f"{name} must contain exactly {sorted(fields)!r}."
        )
    return dict(value)


@dataclass(frozen=True)
class ComponentAreaSupportFloor:
    family: str
    applies_to: tuple[str, ...]
    minimum_area_px_reference: int
    positive_label_count: int
    observed_minimum_by_source_label: Mapping[str, int]


@dataclass(frozen=True)
class SubjectMaskComponentAreaSupportProfile:
    profile_id: str
    model_binding: Mapping[str, str]
    training_evidence: Mapping[str, Any]
    component_families: tuple[ComponentAreaSupportFloor, ...]
    payload: Mapping[str, Any]
    payload_digest: str
    document_sha256: str
    source_path: str

    @property
    def reference_mask_shape_hw(self) -> tuple[int, int]:
        shape = self.training_evidence["reference_mask_shape_hw"]
        return int(shape[0]), int(shape[1])

    def floor_for_component(self, component_name: str) -> ComponentAreaSupportFloor:
        requested = str(component_name).strip()
        matches = [
            item for item in self.component_families if requested in item.applies_to
        ]
        if len(matches) != 1:
            raise SubjectMaskComponentSupportError(
                f"Profile {self.profile_id!r} has {len(matches)} support floors for "
                f"component {requested!r}; expected exactly one."
            )
        return matches[0]

    def minimum_area_px(
        self,
        component_name: str,
        *,
        mask_shape_hw: Sequence[int],
    ) -> int:
        if len(mask_shape_hw) != 2:
            raise SubjectMaskComponentSupportError(
                "mask_shape_hw must contain exactly height and width."
            )
        height = _require_positive_int(int(mask_shape_hw[0]), name="mask height")
        width = _require_positive_int(int(mask_shape_hw[1]), name="mask width")
        reference_height, reference_width = self.reference_mask_shape_hw
        floor = self.floor_for_component(component_name)
        numerator = int(floor.minimum_area_px_reference) * height * width
        denominator = int(reference_height) * int(reference_width)
        return int(math.ceil(numerator / denominator))

    def component_binding(
        self,
        component_name: str,
        *,
        mask_shape_hw: Sequence[int],
    ) -> dict[str, Any]:
        floor = self.floor_for_component(component_name)
        height, width = int(mask_shape_hw[0]), int(mask_shape_hw[1])
        return {
            "schema_id": COMPONENT_AREA_SUPPORT_SCHEMA_ID,
            "schema_version": COMPONENT_AREA_SUPPORT_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "profile_payload_digest": self.payload_digest,
            "profile_document_sha256": self.document_sha256,
            "derivation_method": COMPONENT_AREA_SUPPORT_DERIVATION_METHOD,
            "component_name": str(component_name),
            "component_family": floor.family,
            "mask_shape_hw": [height, width],
            "minimum_area_px": self.minimum_area_px(
                component_name, mask_shape_hw=(height, width)
            ),
            "reference_mask_shape_hw": list(self.reference_mask_shape_hw),
            "minimum_area_px_reference": int(floor.minimum_area_px_reference),
            "model_binding": dict(self.model_binding),
            "training_manifest_sha256": str(
                self.training_evidence["training_manifest_sha256"]
            ),
        }


def _parse_profile(
    raw: object,
    *,
    source_path: Path,
    document_sha256: str,
) -> SubjectMaskComponentAreaSupportProfile:
    if not isinstance(raw, Mapping):
        raise SubjectMaskComponentSupportError(
            f"Component-area support profile must be an object: {source_path}"
        )
    payload = dict(raw)
    expected_top_level = {
        "schema_id",
        "schema_version",
        "profile_id",
        "derivation_method",
        "model_binding",
        "training_evidence",
        "component_families",
    }
    if set(payload) != expected_top_level:
        raise SubjectMaskComponentSupportError(
            "Component-area support profile fields are not exact."
        )
    if payload["schema_id"] != COMPONENT_AREA_SUPPORT_SCHEMA_ID:
        raise SubjectMaskComponentSupportError(
            "Unsupported component-area support profile schema_id."
        )
    if payload["schema_version"] != COMPONENT_AREA_SUPPORT_SCHEMA_VERSION:
        raise SubjectMaskComponentSupportError(
            "Unsupported component-area support profile schema_version."
        )
    if payload["derivation_method"] != COMPONENT_AREA_SUPPORT_DERIVATION_METHOD:
        raise SubjectMaskComponentSupportError(
            "Unsupported component-area support derivation method."
        )

    profile_id = _require_text(payload["profile_id"], name="profile_id")
    model = _require_exact_fields(
        payload["model_binding"], _MODEL_BINDING_FIELDS, name="model_binding"
    )
    normalized_model = {
        "registry_set_id": _require_text(
            model["registry_set_id"], name="model_binding.registry_set_id"
        ),
        "registry_run_id": _require_text(
            model["registry_run_id"], name="model_binding.registry_run_id"
        ),
        "artifact_sha256": _require_sha256(
            model["artifact_sha256"], name="model_binding.artifact_sha256"
        ),
        "label_schema_id": _require_text(
            model["label_schema_id"], name="model_binding.label_schema_id"
        ),
    }
    evidence = _require_exact_fields(
        payload["training_evidence"],
        _TRAINING_EVIDENCE_FIELDS,
        name="training_evidence",
    )
    evidence["training_manifest_sha256"] = _require_sha256(
        evidence["training_manifest_sha256"],
        name="training_evidence.training_manifest_sha256",
    )
    for key in (
        "approved_source_run",
        "approved_source_label_schema_id",
        "metadata_read_mode",
        "area_source",
    ):
        evidence[key] = _require_text(evidence[key], name=f"training_evidence.{key}")
    evidence["approved_source_count"] = _require_positive_int(
        evidence["approved_source_count"],
        name="training_evidence.approved_source_count",
    )
    evidence["approved_row_count"] = _require_positive_int(
        evidence["approved_row_count"], name="training_evidence.approved_row_count"
    )
    shape = evidence["reference_mask_shape_hw"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise SubjectMaskComponentSupportError(
            "training_evidence.reference_mask_shape_hw must be [height, width]."
        )
    evidence["reference_mask_shape_hw"] = [
        _require_positive_int(shape[0], name="reference mask height"),
        _require_positive_int(shape[1], name="reference mask width"),
    ]

    raw_families = payload["component_families"]
    if not isinstance(raw_families, list) or not raw_families:
        raise SubjectMaskComponentSupportError(
            "component_families must be a nonempty list."
        )
    families: list[ComponentAreaSupportFloor] = []
    claimed_components: set[str] = set()
    for index, raw_family in enumerate(raw_families):
        item = _require_exact_fields(
            raw_family,
            _COMPONENT_FAMILY_FIELDS,
            name=f"component_families[{index}]",
        )
        family = _require_text(
            item["family"], name=f"component_families[{index}].family"
        )
        applies_raw = item["applies_to"]
        if not isinstance(applies_raw, list) or not applies_raw:
            raise SubjectMaskComponentSupportError(
                f"component_families[{index}].applies_to must be nonempty."
            )
        applies_to = tuple(
            _require_text(value, name=f"component_families[{index}].applies_to")
            for value in applies_raw
        )
        if len(set(applies_to)) != len(applies_to) or claimed_components.intersection(
            applies_to
        ):
            raise SubjectMaskComponentSupportError(
                "Component-area support profiles may bind each component only once."
            )
        claimed_components.update(applies_to)
        minimum = _require_positive_int(
            item["minimum_area_px_reference"],
            name=f"component_families[{index}].minimum_area_px_reference",
        )
        positive_count = _require_positive_int(
            item["positive_label_count"],
            name=f"component_families[{index}].positive_label_count",
        )
        observed = item["observed_minimum_by_source_label"]
        if not isinstance(observed, Mapping) or not observed:
            raise SubjectMaskComponentSupportError(
                f"component_families[{index}].observed_minimum_by_source_label "
                "must be nonempty."
            )
        normalized_observed = {
            _require_text(key, name="source label"): _require_positive_int(
                value, name=f"observed minimum for {key!r}"
            )
            for key, value in observed.items()
        }
        if minimum != min(normalized_observed.values()):
            raise SubjectMaskComponentSupportError(
                f"Component family {family!r} minimum does not equal its observed "
                "source-label minimum."
            )
        families.append(
            ComponentAreaSupportFloor(
                family=family,
                applies_to=applies_to,
                minimum_area_px_reference=minimum,
                positive_label_count=positive_count,
                observed_minimum_by_source_label=normalized_observed,
            )
        )
    if claimed_components != _SUPPORTED_COMPONENTS:
        raise SubjectMaskComponentSupportError(
            "Component-area support profile coverage must be exact; "
            f"expected {sorted(_SUPPORTED_COMPONENTS)!r}, got "
            f"{sorted(claimed_components)!r}."
        )

    return SubjectMaskComponentAreaSupportProfile(
        profile_id=profile_id,
        model_binding=normalized_model,
        training_evidence=evidence,
        component_families=tuple(families),
        payload=payload,
        payload_digest=canonical_json_sha256(payload),
        document_sha256=document_sha256,
        source_path=str(source_path),
    )


def load_subject_mask_component_area_support_profile(
    path: str | Path,
) -> SubjectMaskComponentAreaSupportProfile:
    source_path = Path(path).expanduser().resolve()
    document = source_path.read_bytes()
    try:
        raw = json.loads(document)
    except json.JSONDecodeError as exc:
        raise SubjectMaskComponentSupportError(
            f"Component-area support profile is not valid JSON: {source_path}"
        ) from exc
    return _parse_profile(
        raw,
        source_path=source_path,
        document_sha256=hashlib.sha256(document).hexdigest(),
    )


def require_subject_mask_component_area_support_profile(
    model_identity: Mapping[str, object],
    *,
    search_dirs: Sequence[str | Path] | None = None,
) -> SubjectMaskComponentAreaSupportProfile:
    model = {
        "registry_set_id": _require_text(
            model_identity.get("registry_set_id"), name="model.registry_set_id"
        ),
        "registry_run_id": _require_text(
            model_identity.get("registry_run_id"), name="model.registry_run_id"
        ),
        "artifact_sha256": _require_sha256(
            model_identity.get("artifact_sha256"), name="model.artifact_sha256"
        ),
        "label_schema_id": _require_text(
            model_identity.get("label_schema_id"), name="model.label_schema_id"
        ),
    }
    directories = (
        tuple(Path(value).expanduser().resolve() for value in search_dirs)
        if search_dirs is not None
        else runtime_config_dirs(COMPONENT_AREA_SUPPORT_CONFIG_DIR)
    )
    candidates = [
        directory / f"{model['artifact_sha256']}.json" for directory in directories
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise SubjectMaskComponentSupportError(
            "No component-area support profile exists for model artifact "
            f"{model['artifact_sha256']}. Tried "
            + ", ".join(str(path) for path in candidates)
        )
    # Source checkouts and installed wheels are alternate locations for the same
    # version-controlled resource.  Match the precedence used by the other
    # runtime configuration loaders: source checkout first, installed fallback.
    profile = load_subject_mask_component_area_support_profile(existing[0])
    if dict(profile.model_binding) != model:
        raise SubjectMaskComponentSupportError(
            "Component-area support profile model binding differs from the sealed "
            "subject-mask model identity."
        )
    return profile


__all__ = [
    "COMPONENT_AREA_SUPPORT_DERIVATION_METHOD",
    "COMPONENT_AREA_SUPPORT_SCHEMA_ID",
    "COMPONENT_AREA_SUPPORT_SCHEMA_VERSION",
    "ComponentAreaSupportFloor",
    "SubjectMaskComponentAreaSupportProfile",
    "SubjectMaskComponentSupportError",
    "load_subject_mask_component_area_support_profile",
    "require_subject_mask_component_area_support_profile",
]
