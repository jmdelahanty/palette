"""Immutable publication input shared by subject-position adapters and writers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.subject_position_expression import (
    ESTIMATOR_PROFILE_RECORDS,
    canonicalize_estimator_profile,
    estimator_profile_digest,
)
from fisheye.shared.subject_position_storage import (
    OBSERVATION_POSITION_ARRAYS,
    OBSERVATION_POSITION_MANDATORY_ARRAYS,
    canonical_observation_position_logical_metadata,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_record(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{name} must be one nonempty mapping.")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be strings.")
    encoded = json.dumps(
        json_attr_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    result = json.loads(encoded)
    if not isinstance(result, dict):  # pragma: no cover - defensive
        raise TypeError(f"{name} did not canonicalize to an object.")
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _bind_record(
    record: Mapping[str, Any],
    digest: str,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    canonical = _canonical_record(record, name=name)
    expected = canonical_json_sha256(canonical)
    actual = _require_digest(digest, name=f"{name}_sha256")
    if actual != expected:
        raise ValueError(f"{name} digest does not match its canonical record.")
    return canonical, actual


def _copy_readonly_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping of exact NumPy arrays.")
    copied: dict[str, np.ndarray] = {}
    for path, value in arrays.items():
        if type(path) is not str:
            raise TypeError("array paths must be strings.")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{path} must be a NumPy array.")
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        copied[path] = array
    return copied


@dataclass(frozen=True)
class SubjectPositionPreparedInput:
    """Generic authorized output supplied by a source adapter."""

    arrays: Mapping[str, np.ndarray]
    estimator_record: Mapping[str, Any]
    estimator_sha256: str
    anatomy_record: Mapping[str, Any]
    anatomy_sha256: str
    source_record: Mapping[str, Any]
    source_sha256: str
    policy_record: Mapping[str, Any]
    policy_sha256: str
    software_record: Mapping[str, Any]
    software_sha256: str
    coordinate_record: Mapping[str, Any]
    coordinate_sha256: str

    def __post_init__(self) -> None:
        copied = _copy_readonly_arrays(self.arrays)
        object.__setattr__(self, "arrays", MappingProxyType(copied))
        unknown = set(copied) - set(OBSERVATION_POSITION_ARRAYS)
        if unknown:
            raise ValueError(f"Unknown subject-position arrays: {sorted(unknown)!r}.")
        if not set(OBSERVATION_POSITION_MANDATORY_ARRAYS).issubset(copied):
            missing = sorted(set(OBSERVATION_POSITION_MANDATORY_ARRAYS) - set(copied))
            raise ValueError(f"Missing subject-position arrays: {missing!r}.")

        estimator = canonicalize_estimator_profile(self.estimator_record)
        estimator_id = str(estimator["estimator_id"])
        if estimator_id not in ESTIMATOR_PROFILE_RECORDS:
            raise ValueError("Only the four registered estimator records are allowed.")
        if self.estimator_sha256 != estimator_profile_digest(estimator):
            raise ValueError("Estimator digest does not match the built-in record.")
        object.__setattr__(self, "estimator_record", _freeze_json(estimator))

        for field_name, label in (
            ("anatomy_record", "anatomy"),
            ("source_record", "source"),
            ("policy_record", "policy"),
            ("software_record", "software"),
        ):
            record, digest = _bind_record(
                getattr(self, field_name),
                getattr(self, f"{field_name[:-7]}_sha256"),
                name=label,
            )
            object.__setattr__(self, field_name, _freeze_json(record))
            object.__setattr__(self, f"{field_name[:-7]}_sha256", digest)

        coordinate = _canonical_record(self.coordinate_record, name="coordinate")
        coordinate_metadata = canonical_observation_position_logical_metadata(
            coordinate
        )
        descriptor_digest = coordinate_metadata["coordinate_descriptor_sha256"]
        declared_descriptor_digest = coordinate.get("coordinate_descriptor_sha256")
        if (
            declared_descriptor_digest is not None
            and declared_descriptor_digest != descriptor_digest
        ):
            raise ValueError("Coordinate descriptor digest is stale.")
        coordinate_digest = _require_digest(
            self.coordinate_sha256,
            name="coordinate_sha256",
        )
        if coordinate_digest != canonical_json_sha256(coordinate):
            raise ValueError("Coordinate record digest does not match its record.")
        object.__setattr__(self, "coordinate_record", _freeze_json(coordinate))


__all__ = ["SubjectPositionPreparedInput"]
