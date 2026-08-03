"""Strict byte-derived storage planning for exact analysis array schemas.

This module is the bridge between logical ``AnalysisArrayDeclaration`` values
and the shared physical storage planner.  It does not create arrays, choose a
production profile, or mutate an archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import ArrayContract, DTypeContract
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import StoragePlan, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import StorageProfile


ANALYSIS_STORAGE_PLAN_SCHEMA_ID = "palette.analysis_storage_plan_receipt"
ANALYSIS_STORAGE_PLAN_SCHEMA_VERSION = 1


def _require_canonical_path(value: Any) -> str:
    if type(value) is not str:
        raise TypeError("Analysis array fact path must be an exact string.")
    path = value
    if (
        not path
        or path != path.strip()
        or path.startswith("/")
        or "\\" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        raise ValueError(f"Invalid canonical analysis array path {path!r}.")
    components = path.split("/")
    if any(
        not component
        or component in {".", ".."}
        or component != component.strip()
        or any(character.isspace() for character in component)
        for component in components
    ):
        raise ValueError(f"Invalid canonical analysis array path {path!r}.")
    return path


def _normalize_exact_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("Analysis array shape must be a tuple or JSON array.")
    if any(type(dimension) is not int for dimension in value):
        raise TypeError("Analysis array shape dimensions must be exact integers.")
    shape = tuple(value)
    if any(dimension < 0 for dimension in shape):
        raise ValueError("Analysis array shape dimensions cannot be negative.")
    return shape


def _normalize_dimensions(value: Mapping[str, int] | None) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("dimensions must be a mapping.")
    normalized: dict[str, int] = {}
    for raw_name, raw_extent in value.items():
        if type(raw_name) is not str or not raw_name or raw_name != raw_name.strip():
            raise ValueError("Dimension names must be exact nonempty strings.")
        if type(raw_extent) is not int or raw_extent < 0:
            raise ValueError(
                f"Dimension {raw_name!r} must have an exact nonnegative integer extent."
            )
        normalized[raw_name] = raw_extent
    return normalized


@dataclass(frozen=True)
class AnalysisArrayStorageFacts:
    """Observed fixed-width array facts needed for physical planning.

    A complete logical record is always the access unit: the growth axis has
    extent one and every other axis retains its full observed extent.  This
    prevents a generic byte budget from splitting keypoint, vector, channel,
    or fixed-width text records across inner chunks.
    """

    path: str
    shape: tuple[int, ...]
    dtype: Any
    access_unit_semantics: str
    growth_axis: int | None = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _require_canonical_path(self.path))
        shape = _normalize_exact_shape(self.shape)
        object.__setattr__(self, "shape", shape)
        dtype = np.dtype(self.dtype)
        if dtype.hasobject or dtype.itemsize <= 0:
            raise ValueError(
                "Analysis byte planning requires an actual fixed-width dtype."
            )
        object.__setattr__(self, "dtype", dtype)
        semantics = self.access_unit_semantics
        if (
            type(semantics) is not str
            or not semantics
            or semantics != semantics.strip()
        ):
            raise ValueError("access_unit_semantics must be one exact string.")
        if not shape:
            if self.growth_axis not in (None, 0):
                raise ValueError("Scalar facts cannot name a growth axis.")
            object.__setattr__(self, "growth_axis", None)
            return
        if type(self.growth_axis) is not int:
            raise TypeError("Non-scalar growth_axis must be an exact integer.")
        if not 0 <= self.growth_axis < len(shape):
            raise ValueError(
                f"growth_axis must address rank {len(shape)}; got {self.growth_axis}."
            )

    @property
    def access_unit_shape(self) -> tuple[int, ...]:
        if not self.shape:
            return ()
        assert self.growth_axis is not None
        return tuple(
            1 if axis == self.growth_axis else max(1, extent)
            for axis, extent in enumerate(self.shape)
        )

    @property
    def shard_axes(self) -> tuple[int, ...]:
        if self.growth_axis is None:
            return ()
        return (self.growth_axis,)

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "itemsize_bytes": int(self.dtype.itemsize),
            "growth_axis": self.growth_axis,
            "access_unit_shape": list(self.access_unit_shape),
            "access_unit_semantics": self.access_unit_semantics,
            "shard_axes": list(self.shard_axes),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> AnalysisArrayStorageFacts:
        expected = {
            "path",
            "shape",
            "dtype",
            "itemsize_bytes",
            "growth_axis",
            "access_unit_shape",
            "access_unit_semantics",
            "shard_axes",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(
                "Analysis array storage facts have an unexpected field set."
            )
        result = cls(
            path=value["path"],
            shape=_normalize_exact_shape(value["shape"]),
            dtype=value["dtype"],
            access_unit_semantics=value["access_unit_semantics"],
            growth_axis=value["growth_axis"],
        )
        if result.as_manifest() != dict(value):
            raise ValueError("Analysis array storage facts are not canonical.")
        return result


@dataclass(frozen=True)
class AnalysisArrayStoragePlanReceipt:
    """One declaration-bound physical plan and its array object estimate."""

    declaration: AnalysisArrayDeclaration
    facts: AnalysisArrayStorageFacts
    resolved_dimensions: tuple[tuple[str, int], ...]
    plan: StoragePlan

    def __post_init__(self) -> None:
        if self.resolved_dimensions != tuple(sorted(self.resolved_dimensions)) or len(
            self.resolved_dimensions
        ) != len(dict(self.resolved_dimensions)):
            raise ValueError(
                "Analysis plan receipt dimensions must be unique and sorted."
            )
        if self.declaration.path != self.facts.path:
            raise ValueError("Analysis plan receipt path identity mismatch.")
        if self.plan.array_name != self.declaration.path:
            raise ValueError("Analysis plan receipt plan identity mismatch.")
        if self.plan.logical_shape != self.facts.shape:
            raise ValueError("Analysis plan receipt shape identity mismatch.")
        if self.plan.logical_dtype != str(self.facts.dtype):
            raise ValueError("Analysis plan receipt dtype identity mismatch.")
        if self.plan.access_unit_shape != self.facts.access_unit_shape:
            raise ValueError("Analysis plan receipt access-unit mismatch.")
        if self.plan.shard_axes != self.facts.shard_axes:
            raise ValueError("Analysis plan receipt shard-axis mismatch.")
        if self.plan.access_pattern != self.declaration.access_pattern.value:
            raise ValueError("Analysis plan receipt access classification mismatch.")
        if self.plan.write_mode != self.declaration.write_mode.value:
            raise ValueError("Analysis plan receipt lifecycle mismatch.")

    @property
    def lifecycle_classification(self) -> str:
        if self.declaration.write_mode is WriteMode.IMMUTABLE:
            return "immutable_snapshot_array"
        if self.declaration.write_mode is WriteMode.APPEND_ONLY:
            return "append_only_array"
        return "random_update_array"

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.declaration.path,
            "declaration": self.declaration.as_manifest(),
            "observed_facts": self.facts.as_manifest(),
            "resolved_dimensions": dict(self.resolved_dimensions),
            "lifecycle_classification": self.lifecycle_classification,
            "access_unit_semantics": self.facts.access_unit_semantics,
            "object_estimate": {
                "logical_nbytes": self.plan.logical_nbytes,
                "inner_chunk_count": self.plan.estimated_chunk_count,
                "payload_objects": self.plan.estimated_payload_objects,
                "array_metadata_objects": 1,
                "array_objects_excluding_group_metadata": (
                    self.plan.estimated_payload_objects + 1
                ),
                "estimate_basis": (
                    "shape-derived populated-payload upper bound; compressor ratio "
                    "and fill-value elision are excluded"
                ),
            },
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class AnalysisStoragePlanReceipt:
    """Deterministic JSON-ready receipt for an exact set of analysis arrays."""

    profile: StorageProfile
    dimensions: tuple[tuple[str, int], ...]
    entries: tuple[AnalysisArrayStoragePlanReceipt, ...]

    def __post_init__(self) -> None:
        if self.dimensions != tuple(sorted(self.dimensions)) or len(
            self.dimensions
        ) != len(dict(self.dimensions)):
            raise ValueError(
                "Analysis storage receipt dimensions must be unique and sorted."
            )
        paths = tuple(entry.declaration.path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError(
                "Analysis storage receipt paths must be unique and sorted."
            )
        if any(
            entry.plan.profile_id != self.profile.profile_id for entry in self.entries
        ):
            raise ValueError("Analysis storage receipt profile identity mismatch.")

    def _payload(self) -> dict[str, object]:
        return {
            "storage_profile": self.profile.as_manifest(),
            "dimensions": dict(self.dimensions),
            "object_estimate": {
                "logical_nbytes": sum(
                    entry.plan.logical_nbytes for entry in self.entries
                ),
                "inner_chunk_count": sum(
                    entry.plan.estimated_chunk_count for entry in self.entries
                ),
                "payload_objects": sum(
                    entry.plan.estimated_payload_objects for entry in self.entries
                ),
                "array_metadata_objects": len(self.entries),
                "array_objects_excluding_group_metadata": sum(
                    entry.plan.estimated_payload_objects + 1 for entry in self.entries
                ),
                "sharded_arrays": sum(entry.plan.is_sharded for entry in self.entries),
                "empty_arrays": sum(
                    entry.plan.logical_nbytes == 0 for entry in self.entries
                ),
                "scope": (
                    "array payload objects and one zarr.json per array; group/root "
                    "metadata objects are intentionally excluded"
                ),
                "estimate_basis": (
                    "shape-derived populated-payload upper bound; compressor ratio "
                    "and fill-value elision are excluded"
                ),
            },
            "arrays": [entry.as_manifest() for entry in self.entries],
        }

    def as_manifest(self) -> dict[str, object]:
        payload = self._payload()
        return {
            "schema_id": ANALYSIS_STORAGE_PLAN_SCHEMA_ID,
            "schema_version": ANALYSIS_STORAGE_PLAN_SCHEMA_VERSION,
            "payload": payload,
            "payload_digest": canonical_json_sha256(payload),
        }


def analysis_array_declaration_from_manifest(
    value: Mapping[str, Any],
) -> AnalysisArrayDeclaration:
    """Parse one exact canonical ``AnalysisArrayDeclaration`` manifest."""

    expected = {
        "path",
        "required",
        "logical_contract",
        "access_pattern",
        "write_mode",
        "authority_role",
        "fill_semantics",
        "null_semantics",
        "physical_policy_owner",
        "byte_planner_adopted",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Analysis array declaration has an unexpected field set.")
    contract_value = value["logical_contract"]
    contract_fields = {
        "schema_id",
        "schema_version",
        "dtype",
        "shape_template",
        "axis_names",
        "description",
        "units",
        "coordinate_space",
    }
    if (
        not isinstance(contract_value, Mapping)
        or set(contract_value) != contract_fields
    ):
        raise ValueError("Analysis logical contract has an unexpected field set.")
    dtype_value = contract_value["dtype"]
    dtype_fields = {
        "dtype_id",
        "numpy_dtype",
        "variable_length",
        "itemsize_bytes",
    }
    if not isinstance(dtype_value, Mapping) or set(dtype_value) != dtype_fields:
        raise ValueError("Analysis dtype contract has an unexpected field set.")
    if type(dtype_value["variable_length"]) is not bool:
        raise TypeError("variable_length must be an exact bool.")
    if (
        dtype_value["numpy_dtype"] is not None
        and type(dtype_value["numpy_dtype"]) is not str
    ):
        raise TypeError("numpy_dtype must be an exact string or null.")
    dtype_contract = DTypeContract(
        dtype_id=dtype_value["dtype_id"],
        numpy_dtype=dtype_value["numpy_dtype"],
        variable_length=dtype_value["variable_length"],
    )
    if dtype_contract.as_manifest() != dict(dtype_value):
        raise ValueError("Analysis dtype contract is not canonical.")
    shape_template = contract_value["shape_template"]
    axis_names = contract_value["axis_names"]
    if not isinstance(shape_template, list) or not isinstance(axis_names, list):
        raise TypeError("Logical shape_template and axis_names must be JSON arrays.")
    description = contract_value["description"]
    if type(description) is not str or not description:
        raise TypeError("Logical contract description must be a nonempty string.")
    for field_name in ("units", "coordinate_space"):
        field_value = contract_value[field_name]
        if field_value is not None and type(field_value) is not str:
            raise TypeError(f"Logical contract {field_name} must be a string or null.")
    contract = ArrayContract(
        schema_id=contract_value["schema_id"],
        schema_version=contract_value["schema_version"],
        dtype=dtype_contract,
        shape_template=tuple(shape_template),
        axis_names=tuple(axis_names),
        description=description,
        units=contract_value["units"],
        coordinate_space=contract_value["coordinate_space"],
    )
    if contract.as_manifest() != dict(contract_value):
        raise ValueError("Analysis logical contract is not canonical.")
    declaration = AnalysisArrayDeclaration(
        path=value["path"],
        contract=contract,
        required=value["required"],
        access_pattern=value["access_pattern"],
        write_mode=value["write_mode"],
        authority_role=AnalysisAuthorityRole(value["authority_role"]),
        fill_semantics=value["fill_semantics"],
        null_semantics=value["null_semantics"],
        physical_policy_owner=value["physical_policy_owner"],
        byte_planner_adopted=value["byte_planner_adopted"],
    )
    if declaration.as_manifest() != dict(value):
        raise ValueError("Analysis array declaration is not canonical.")
    return declaration


def _coerce_declaration(
    value: AnalysisArrayDeclaration | Mapping[str, Any],
) -> AnalysisArrayDeclaration:
    if isinstance(value, AnalysisArrayDeclaration):
        return value
    return analysis_array_declaration_from_manifest(value)


def _coerce_facts(
    value: AnalysisArrayStorageFacts | Mapping[str, Any],
) -> AnalysisArrayStorageFacts:
    if isinstance(value, AnalysisArrayStorageFacts):
        return value
    return AnalysisArrayStorageFacts.from_manifest(value)


def _bind_declaration_dimensions(
    declaration: AnalysisArrayDeclaration,
    shape: tuple[int, ...],
    dimensions: dict[str, int],
) -> dict[str, int]:
    contract = declaration.contract
    errors = contract.validate_shape(shape, dimensions=dimensions)
    if errors:
        raise ValueError(
            f"{declaration.path}: shape contract failed: {'; '.join(errors)}."
        )
    if len(shape) != len(contract.shape_template):
        raise ValueError(f"{declaration.path}: shape rank mismatch.")
    result = dict(dimensions)
    for expected, actual in zip(contract.shape_template, shape, strict=True):
        if not isinstance(expected, str):
            continue
        previous = result.get(expected)
        if previous is not None and previous != actual:
            raise ValueError(
                f"{declaration.path}: symbolic dimension {expected!r} disagrees; "
                f"expected {previous}, got {actual}."
            )
        result[expected] = actual
    return result


def plan_analysis_array_storage(
    declaration: AnalysisArrayDeclaration | Mapping[str, Any],
    facts: AnalysisArrayStorageFacts | Mapping[str, Any],
    *,
    profile: StorageProfile,
    dimensions: Mapping[str, int] | None = None,
) -> AnalysisArrayStoragePlanReceipt:
    """Validate exact logical/observed facts and derive one byte-based plan."""

    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be a caller-supplied StorageProfile.")
    declaration = _coerce_declaration(declaration)
    facts = _coerce_facts(facts)
    if declaration.path != facts.path:
        raise ValueError(
            f"Declaration path {declaration.path!r} does not match facts path "
            f"{facts.path!r}."
        )
    resolved_dimensions = _bind_declaration_dimensions(
        declaration, facts.shape, _normalize_dimensions(dimensions)
    )
    if not declaration.contract.dtype.matches(facts.dtype):
        raise ValueError(
            f"{declaration.path}: dtype mismatch; expected "
            f"{declaration.contract.dtype.dtype_id}, got {facts.dtype}."
        )
    if declaration.contract.dtype.variable_length:
        raise ValueError(
            f"{declaration.path}: variable-width dtypes cannot be byte-planned "
            "without an exact fixed-width representation."
        )
    intent = declaration.contract.storage_intent(
        shape=facts.shape,
        access=declaration.access_pattern,
        write_mode=declaration.write_mode,
        access_unit_shape=facts.access_unit_shape,
        growth_axis=facts.growth_axis or 0,
        shard_axes=facts.shard_axes,
        logical_itemsize_bytes=int(facts.dtype.itemsize),
        whole_shard_writes=declaration.write_mode is WriteMode.IMMUTABLE,
        name=declaration.path,
        dimensions=resolved_dimensions,
    )
    plan = plan_storage(intent, profile)
    relevant_dimension_names = {
        name for name in declaration.contract.shape_template if isinstance(name, str)
    }
    relevant_dimensions = tuple(
        (name, resolved_dimensions[name]) for name in sorted(relevant_dimension_names)
    )
    return AnalysisArrayStoragePlanReceipt(
        declaration=declaration,
        facts=facts,
        resolved_dimensions=relevant_dimensions,
        plan=plan,
    )


def plan_analysis_storage(
    declarations: Iterable[AnalysisArrayDeclaration | Mapping[str, Any]],
    facts_by_path: Mapping[str, AnalysisArrayStorageFacts | Mapping[str, Any]],
    *,
    profile: StorageProfile,
    dimensions: Mapping[str, int] | None = None,
) -> AnalysisStoragePlanReceipt:
    """Plan one exact declaration/fact set and return a digest-bound receipt."""

    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be a caller-supplied StorageProfile.")
    declaration_by_path: dict[str, AnalysisArrayDeclaration] = {}
    for raw_declaration in declarations:
        declaration = _coerce_declaration(raw_declaration)
        if declaration.path in declaration_by_path:
            raise ValueError(f"Duplicate analysis declaration {declaration.path!r}.")
        declaration_by_path[declaration.path] = declaration
    if not isinstance(facts_by_path, Mapping):
        raise TypeError("facts_by_path must be a mapping.")
    canonical_facts_by_path: dict[
        str, AnalysisArrayStorageFacts | Mapping[str, Any]
    ] = {}
    for raw_path, raw_facts in facts_by_path.items():
        path = _require_canonical_path(raw_path)
        if path in canonical_facts_by_path:
            raise ValueError(f"Duplicate analysis fact path {path!r}.")
        canonical_facts_by_path[path] = raw_facts
    fact_paths = set(canonical_facts_by_path)
    unexpected = sorted(fact_paths - set(declaration_by_path))
    missing = sorted(
        declaration.path
        for declaration in declaration_by_path.values()
        if declaration.required and declaration.path not in fact_paths
    )
    if unexpected:
        raise ValueError(f"Unexpected analysis array facts: {unexpected!r}.")
    if missing:
        raise ValueError(f"Missing required analysis array facts: {missing!r}.")

    resolved_dimensions = _normalize_dimensions(dimensions)
    normalized_facts: dict[str, AnalysisArrayStorageFacts] = {}
    for path, raw_facts in canonical_facts_by_path.items():
        facts = _coerce_facts(raw_facts)
        if path != facts.path:
            raise ValueError(
                f"Fact mapping key {path!r} does not match payload path {facts.path!r}."
            )
        declaration = declaration_by_path[path]
        resolved_dimensions = _bind_declaration_dimensions(
            declaration, facts.shape, resolved_dimensions
        )
        normalized_facts[path] = facts

    entries = tuple(
        plan_analysis_array_storage(
            declaration_by_path[path],
            normalized_facts[path],
            profile=profile,
            dimensions=resolved_dimensions,
        )
        for path in sorted(normalized_facts)
    )
    return AnalysisStoragePlanReceipt(
        profile=profile,
        dimensions=tuple(sorted(resolved_dimensions.items())),
        entries=entries,
    )


__all__ = [
    "ANALYSIS_STORAGE_PLAN_SCHEMA_ID",
    "ANALYSIS_STORAGE_PLAN_SCHEMA_VERSION",
    "AnalysisArrayStorageFacts",
    "AnalysisArrayStoragePlanReceipt",
    "AnalysisStoragePlanReceipt",
    "analysis_array_declaration_from_manifest",
    "plan_analysis_array_storage",
    "plan_analysis_storage",
]
