"""Versioned, typed cohort query specification.

The cohort query intentionally exposes no raw SQL.  It is a reusable scientific
definition; a frozen cohort manifest records the exact registry rows selected by
one evaluation of that definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_ID = "palette.cohort_query"
SCHEMA_VERSION = 1
COHORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SUBJECT_MATCH_POLICIES = frozenset(
    {"unambiguous_recording", "any_subject", "all_subjects"}
)
MISSING_METADATA_POLICIES = frozenset({"error", "exclude"})


class CohortSpecError(ValueError):
    """Raised when a cohort query specification is invalid."""


def _reject_unknown(raw: Mapping[str, Any], *, allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise CohortSpecError(
            f"{label} contains unknown field(s): {', '.join(unknown)}"
        )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CohortSpecError(f"{label} must be a mapping")
    return value


def _strings(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise CohortSpecError(f"{label} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise CohortSpecError(f"{label} cannot contain empty values")
    if len(set(result)) != len(result):
        raise CohortSpecError(f"{label} cannot contain duplicate values")
    return result


def _optional_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CohortSpecError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CohortSpecError(f"{label} must be an integer") from exc
    if result < 0:
        raise CohortSpecError(f"{label} must be non-negative")
    return result


@dataclass(frozen=True)
class DatasetSelector:
    statuses: tuple[str, ...] = ("active",)
    zarr_uses: tuple[str, ...] = ("analysis",)
    zarr_origins: tuple[str, ...] = ("source",)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetSelector":
        _reject_unknown(
            raw,
            allowed={"statuses", "zarr_uses", "zarr_origins"},
            label="dataset",
        )
        statuses = _strings(raw.get("statuses", ["active"]), label="dataset.statuses")
        zarr_uses = _strings(
            raw.get("zarr_uses", ["analysis"]), label="dataset.zarr_uses"
        )
        origins = _strings(
            raw.get("zarr_origins", ["source"]), label="dataset.zarr_origins"
        )
        if not statuses or not zarr_uses or not origins:
            raise CohortSpecError(
                "dataset.statuses, dataset.zarr_uses, and dataset.zarr_origins cannot be empty"
            )
        return cls(statuses=statuses, zarr_uses=zarr_uses, zarr_origins=origins)


@dataclass(frozen=True)
class ProtocolSelector:
    stimulus_modes_any: tuple[str, ...] = ()
    protocol_names_any: tuple[str, ...] = ()
    protocol_hashes_any: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProtocolSelector":
        _reject_unknown(
            raw,
            allowed={
                "stimulus_modes_any",
                "protocol_names_any",
                "protocol_hashes_any",
            },
            label="protocol",
        )
        hashes = _strings(
            raw.get("protocol_hashes_any"), label="protocol.protocol_hashes_any"
        )
        for value in hashes:
            if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise CohortSpecError(
                    "protocol.protocol_hashes_any values must be 64-character SHA-256 hex strings"
                )
        return cls(
            stimulus_modes_any=tuple(
                value.upper()
                for value in _strings(
                    raw.get("stimulus_modes_any"),
                    label="protocol.stimulus_modes_any",
                )
            ),
            protocol_names_any=_strings(
                raw.get("protocol_names_any"), label="protocol.protocol_names_any"
            ),
            protocol_hashes_any=tuple(value.lower() for value in hashes),
        )


@dataclass(frozen=True)
class DpfSelector:
    values: tuple[int, ...] = ()
    minimum: int | None = None
    maximum: int | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DpfSelector":
        _reject_unknown(raw, allowed={"values", "min", "max"}, label="subjects.dpf")
        values_raw = raw.get("values")
        values: tuple[int, ...] = ()
        if values_raw is not None:
            if isinstance(values_raw, (str, bytes)) or not isinstance(
                values_raw, Sequence
            ):
                raise CohortSpecError(
                    "subjects.dpf.values must be a sequence of integers"
                )
            values = tuple(
                _optional_int(value, label="subjects.dpf.values[]")  # type: ignore[arg-type]
                for value in values_raw
            )
            if any(value is None for value in values):
                raise CohortSpecError("subjects.dpf.values cannot contain null")
            values = tuple(int(value) for value in values)
            if len(set(values)) != len(values):
                raise CohortSpecError("subjects.dpf.values cannot contain duplicates")
        minimum = _optional_int(raw.get("min"), label="subjects.dpf.min")
        maximum = _optional_int(raw.get("max"), label="subjects.dpf.max")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise CohortSpecError("subjects.dpf.min cannot exceed subjects.dpf.max")
        return cls(values=values, minimum=minimum, maximum=maximum)

    @property
    def active(self) -> bool:
        return bool(self.values) or self.minimum is not None or self.maximum is not None


@dataclass(frozen=True)
class SubjectSelector:
    dpf: DpfSelector = field(default_factory=DpfSelector)
    line_strains_any: tuple[str, ...] = ()
    genotypes_any: tuple[str, ...] = ()
    cross_ids_any: tuple[str, ...] = ()
    match_policy: str = "unambiguous_recording"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SubjectSelector":
        _reject_unknown(
            raw,
            allowed={
                "dpf",
                "line_strains_any",
                "genotypes_any",
                "cross_ids_any",
                "match_policy",
            },
            label="subjects",
        )
        policy = str(raw.get("match_policy", "unambiguous_recording")).strip()
        if policy not in SUBJECT_MATCH_POLICIES:
            raise CohortSpecError(
                f"subjects.match_policy must be one of {sorted(SUBJECT_MATCH_POLICIES)}"
            )
        return cls(
            dpf=DpfSelector.from_mapping(
                _mapping(raw.get("dpf"), label="subjects.dpf")
            ),
            line_strains_any=_strings(
                raw.get("line_strains_any"), label="subjects.line_strains_any"
            ),
            genotypes_any=_strings(
                raw.get("genotypes_any"), label="subjects.genotypes_any"
            ),
            cross_ids_any=_strings(
                raw.get("cross_ids_any"), label="subjects.cross_ids_any"
            ),
            match_policy=policy,
        )


@dataclass(frozen=True)
class PrerequisiteSelector:
    required_steps_ok: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PrerequisiteSelector":
        _reject_unknown(
            raw,
            allowed={"required_steps_ok"},
            label="prerequisites",
        )
        return cls(
            required_steps_ok=tuple(
                value.lower()
                for value in _strings(
                    raw.get("required_steps_ok"),
                    label="prerequisites.required_steps_ok",
                )
            )
        )


@dataclass(frozen=True)
class CohortSpec:
    cohort_id: str
    cohort_name: str
    dataset: DatasetSelector = field(default_factory=DatasetSelector)
    protocol: ProtocolSelector = field(default_factory=ProtocolSelector)
    subjects: SubjectSelector = field(default_factory=SubjectSelector)
    prerequisites: PrerequisiteSelector = field(default_factory=PrerequisiteSelector)
    missing_selected_metadata: str = "error"
    purpose: str = "registry-defined analysis cohort"
    schema_id: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CohortSpec":
        _reject_unknown(
            raw,
            allowed={
                "schema_id",
                "schema_version",
                "cohort_id",
                "cohort_name",
                "purpose",
                "dataset",
                "protocol",
                "subjects",
                "prerequisites",
                "missing_selected_metadata",
            },
            label="cohort query",
        )
        schema_id = str(raw.get("schema_id") or "").strip()
        schema_version = int(raw.get("schema_version") or 0)
        if schema_id != SCHEMA_ID or schema_version != SCHEMA_VERSION:
            raise CohortSpecError(
                f"cohort spec must use schema_id={SCHEMA_ID!r}, schema_version={SCHEMA_VERSION}"
            )
        cohort_id = str(raw.get("cohort_id") or "").strip()
        if not COHORT_ID_PATTERN.fullmatch(cohort_id):
            raise CohortSpecError(
                "cohort_id must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_', or '-'"
            )
        cohort_name = str(raw.get("cohort_name") or "").strip()
        if not cohort_name:
            raise CohortSpecError("cohort_name is required")
        missing_policy = str(raw.get("missing_selected_metadata", "error")).strip()
        if missing_policy not in MISSING_METADATA_POLICIES:
            raise CohortSpecError(
                "missing_selected_metadata must be 'error' or 'exclude'"
            )
        purpose = str(raw.get("purpose") or "registry-defined analysis cohort").strip()
        if not purpose:
            raise CohortSpecError("purpose cannot be empty")
        return cls(
            cohort_id=cohort_id,
            cohort_name=cohort_name,
            dataset=DatasetSelector.from_mapping(
                _mapping(raw.get("dataset"), label="dataset")
            ),
            protocol=ProtocolSelector.from_mapping(
                _mapping(raw.get("protocol"), label="protocol")
            ),
            subjects=SubjectSelector.from_mapping(
                _mapping(raw.get("subjects"), label="subjects")
            ),
            prerequisites=PrerequisiteSelector.from_mapping(
                _mapping(raw.get("prerequisites"), label="prerequisites")
            ),
            missing_selected_metadata=missing_policy,
            purpose=purpose,
        )

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["subjects"]["dpf"]["min"] = raw["subjects"]["dpf"].pop("minimum")
        raw["subjects"]["dpf"]["max"] = raw["subjects"]["dpf"].pop("maximum")
        return raw

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_cohort_spec(path: str | Path) -> CohortSpec:
    spec_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise CohortSpecError(f"cohort spec must contain a mapping: {spec_path}")
    return CohortSpec.from_mapping(raw)
