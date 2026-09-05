"""Closed installed profiles for the generic validated-behavior engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .validated_behavior_adapters import (
    build_phase_a_row_extractors,
    build_phase_c_compact_row_extractors,
)
from .validated_behavior_contracts import (
    CORE_METADATA_PROFILE_ID,
    CORE_TABLE_SPECS,
    ValidatedBehaviorTableSpec,
)
from .validated_behavior_core_behavior_adapters import (
    build_core_behavior_row_extractors,
)
from .validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1,
    CORE_BEHAVIOR_TABLE_SPECS,
    CORE_BEHAVIOR_TABLE_SPECS_V1,
)
from .validated_behavior_phase_a_contracts import (
    PHASE_A_PROFILE_ID,
    PHASE_A_TABLE_SPECS,
)
from .validated_behavior_phase_b_adapters import (
    build_phase_b_dense_row_extractors,
)
from .validated_behavior_phase_b_contracts import (
    PHASE_B_PROFILE_ID,
    PHASE_B_TABLE_SPECS,
)
from .validated_behavior_phase_c_contracts import (
    PHASE_C_PROFILE_ID,
    PHASE_C_TABLE_SPECS,
)


class ValidatedBehaviorProfileError(ValueError):
    """An export names an uninstalled or malformed profile."""


@dataclass(frozen=True)
class ValidatedBehaviorExportProfile:
    profile_id: str
    table_specs: Mapping[str, ValidatedBehaviorTableSpec]
    row_extractor_factory: Callable[[], Mapping[str, Callable[..., Any]]]

    def row_extractors(self) -> Mapping[str, Callable[..., Any]]:
        return self.row_extractor_factory()


def _no_extractors() -> Mapping[str, Callable[..., Any]]:
    return MappingProxyType({})


def _phase_b_extractors() -> Mapping[str, Callable[..., Any]]:
    extractors = dict(build_phase_a_row_extractors())
    extractors.update(build_phase_b_dense_row_extractors())
    return MappingProxyType(extractors)


def _phase_c_extractors() -> Mapping[str, Callable[..., Any]]:
    extractors = dict(build_phase_c_compact_row_extractors())
    extractors.update(build_phase_b_dense_row_extractors())
    return MappingProxyType(extractors)


_COMPETING_CORE_MOTION_TABLES = frozenset(
    {"kinematics_samples", "provider_motion_samples"}
)


def _validated_profile_map(
    profiles: Mapping[str, ValidatedBehaviorExportProfile],
) -> Mapping[str, ValidatedBehaviorExportProfile]:
    for profile_id, profile in profiles.items():
        if profile.profile_id != profile_id:
            raise ValidatedBehaviorProfileError(
                "Installed validated-behavior profile key and ID differ."
            )
        competing = _COMPETING_CORE_MOTION_TABLES.intersection(profile.table_specs)
        if len(competing) > 1:
            raise ValidatedBehaviorProfileError(
                f"Profile {profile_id!r} contains competing core-motion projections: "
                f"{sorted(competing)!r}. Paradigm extensions must join the selected "
                "kinematics_samples authority instead."
            )
    return MappingProxyType(dict(profiles))


INSTALLED_VALIDATED_BEHAVIOR_PROFILES: Mapping[str, ValidatedBehaviorExportProfile] = (
    _validated_profile_map(
        {
            CORE_METADATA_PROFILE_ID: ValidatedBehaviorExportProfile(
                profile_id=CORE_METADATA_PROFILE_ID,
                table_specs=CORE_TABLE_SPECS,
                row_extractor_factory=_no_extractors,
            ),
            CORE_BEHAVIOR_EXPORT_PROFILE_ID: ValidatedBehaviorExportProfile(
                profile_id=CORE_BEHAVIOR_EXPORT_PROFILE_ID,
                table_specs=CORE_BEHAVIOR_TABLE_SPECS,
                row_extractor_factory=build_core_behavior_row_extractors,
            ),
            CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1: ValidatedBehaviorExportProfile(
                profile_id=CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1,
                table_specs=CORE_BEHAVIOR_TABLE_SPECS_V1,
                row_extractor_factory=build_core_behavior_row_extractors,
            ),
            PHASE_A_PROFILE_ID: ValidatedBehaviorExportProfile(
                profile_id=PHASE_A_PROFILE_ID,
                table_specs=PHASE_A_TABLE_SPECS,
                row_extractor_factory=build_phase_a_row_extractors,
            ),
            PHASE_B_PROFILE_ID: ValidatedBehaviorExportProfile(
                profile_id=PHASE_B_PROFILE_ID,
                table_specs=PHASE_B_TABLE_SPECS,
                row_extractor_factory=_phase_b_extractors,
            ),
            PHASE_C_PROFILE_ID: ValidatedBehaviorExportProfile(
                profile_id=PHASE_C_PROFILE_ID,
                table_specs=PHASE_C_TABLE_SPECS,
                row_extractor_factory=_phase_c_extractors,
            ),
        }
    )
)


def resolve_validated_behavior_profile(
    profile_id: object,
) -> ValidatedBehaviorExportProfile:
    if type(profile_id) is not str:
        raise ValidatedBehaviorProfileError("Export profile ID must be one string.")
    try:
        return INSTALLED_VALIDATED_BEHAVIOR_PROFILES[profile_id]
    except KeyError as exc:
        raise ValidatedBehaviorProfileError(
            f"Validated-behavior export profile is not installed: {profile_id!r}."
        ) from exc


def profile_id_from_record(path: str | Path, *, record_kind: str) -> str:
    """Read only the routing ID; the selected profile then validates everything."""

    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorProfileError(
            f"Cannot read {record_kind} profile routing record: {source}."
        ) from exc
    if not isinstance(value, dict):
        raise ValidatedBehaviorProfileError(f"{record_kind} must be one JSON object.")
    profile = value.get("export_profile")
    if not isinstance(profile, dict) or type(profile.get("profile_id")) is not str:
        raise ValidatedBehaviorProfileError(
            f"{record_kind} lacks one exact export-profile ID."
        )
    return str(profile["profile_id"])


__all__ = [
    "INSTALLED_VALIDATED_BEHAVIOR_PROFILES",
    "ValidatedBehaviorExportProfile",
    "ValidatedBehaviorProfileError",
    "profile_id_from_record",
    "resolve_validated_behavior_profile",
]
