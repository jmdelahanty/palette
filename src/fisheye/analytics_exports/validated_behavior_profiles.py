"""Closed installed profiles for the generic validated-behavior engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .validated_behavior_adapters import build_phase_a_row_extractors
from .validated_behavior_contracts import (
    CORE_METADATA_PROFILE_ID,
    CORE_TABLE_SPECS,
    ValidatedBehaviorTableSpec,
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


INSTALLED_VALIDATED_BEHAVIOR_PROFILES: Mapping[str, ValidatedBehaviorExportProfile] = (
    MappingProxyType(
        {
            CORE_METADATA_PROFILE_ID: ValidatedBehaviorExportProfile(
                profile_id=CORE_METADATA_PROFILE_ID,
                table_specs=CORE_TABLE_SPECS,
                row_extractor_factory=_no_extractors,
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
