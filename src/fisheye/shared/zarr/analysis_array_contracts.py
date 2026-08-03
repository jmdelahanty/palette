"""Reusable exact array declarations for maintained analysis products.

The declaration composes the logical :class:`ArrayContract` with lifecycle
facts needed by analytics writers and validators.  It deliberately records,
but does not resolve, physical storage policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fisheye.shared.zarr.array_contracts import ArrayContract
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode


class AnalysisAuthorityRole(str, Enum):
    """Closed authority categories shared by maintained analytics families."""

    SCIENTIFIC_AUTHORITY = "scientific_authority"
    LINEAGE_INDEX = "lineage_index"
    SEMANTIC_METADATA = "semantic_metadata"
    QUALITY_DIAGNOSTIC = "quality_diagnostic"
    DERIVED_CACHE = "derived_cache"
    COMPATIBILITY_ALIAS = "compatibility_alias"


@dataclass(frozen=True)
class AnalysisArrayDeclaration:
    """Bind one exact logical contract to an analysis-run path."""

    path: str
    contract: ArrayContract
    required: bool
    access_pattern: AccessPattern
    write_mode: WriteMode
    authority_role: AnalysisAuthorityRole
    fill_semantics: str
    null_semantics: str
    physical_policy_owner: str
    byte_planner_adopted: bool

    def __post_init__(self) -> None:
        if type(self.path) is not str:
            raise TypeError("path must be an exact str.")
        path = self.path
        if (
            path != path.strip()
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
        ):
            raise ValueError(
                "Analysis array paths must be nonempty canonical relative paths."
            )
        components = path.split("/")
        if any(
            not component
            or component in {".", ".."}
            or component != component.strip()
            or any(character.isspace() for character in component)
            for component in components
        ):
            raise ValueError(
                "Analysis array paths cannot contain empty, dot, parent, or whitespace components."
            )
        if not isinstance(self.contract, ArrayContract):
            raise TypeError("contract must be an ArrayContract instance.")
        object.__setattr__(self, "access_pattern", AccessPattern(self.access_pattern))
        object.__setattr__(self, "write_mode", WriteMode(self.write_mode))
        object.__setattr__(
            self, "authority_role", AnalysisAuthorityRole(self.authority_role)
        )
        for field_name in (
            "fill_semantics",
            "null_semantics",
            "physical_policy_owner",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or value != value.strip() or not value:
                raise ValueError(f"{field_name} must be one exact nonempty string.")
        if type(self.required) is not bool:
            raise TypeError("required must be an exact bool.")
        if type(self.byte_planner_adopted) is not bool:
            raise TypeError("byte_planner_adopted must be an exact bool.")

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.path,
            "required": self.required,
            "logical_contract": self.contract.as_manifest(),
            "access_pattern": self.access_pattern.value,
            "write_mode": self.write_mode.value,
            "authority_role": self.authority_role.value,
            "fill_semantics": self.fill_semantics,
            "null_semantics": self.null_semantics,
            "physical_policy_owner": self.physical_policy_owner,
            "byte_planner_adopted": self.byte_planner_adopted,
        }
