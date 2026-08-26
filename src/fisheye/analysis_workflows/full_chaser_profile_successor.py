"""Digest-bound full chaser-profile successor envelope.

This module composes immutable recording-local products; it does not execute
their numerical work.  The envelope freezes the normalized profile, explicit
overrides, capability/applicability decisions, dependency order, concurrency
waves, and exact product digests.  A planned or blocked envelope is useful as
an honest handoff, but only an applicability plan whose readiness is
``complete`` may produce a full-profile-complete claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.chaser_profiles import (
    ChaserAnalysisProfile,
    resolve_chaser_analysis_modules,
)
from fisheye.analysis_workflows.chaser_profile_applicability import (
    ChaserProfileApplicabilityPlan,
    ModuleApplicabilityState,
    ProfileReadiness,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.analysis.chaser_full_profile_successor"
SCHEMA_VERSION = 1
METHOD_ID = "digest_bound_applicability_and_immutable_module_composition_v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_RUN_PATH_RE = re.compile(r"^analysis/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+$")

STATE_CODES = MappingProxyType(
    {
        ModuleApplicabilityState.APPLICABLE.value: 1,
        ModuleApplicabilityState.INAPPLICABLE.value: 2,
        ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY.value: 3,
        ModuleApplicabilityState.BLOCKED_INVALID_SOURCE.value: 4,
        ModuleApplicabilityState.BLOCKED_REVIEW_REQUIRED.value: 5,
        ModuleApplicabilityState.STALE.value: 6,
        ModuleApplicabilityState.COMPLETE.value: 7,
    }
)


class FullChaserProfileSuccessorError(ValueError):
    """Raised when a full-profile composition is ambiguous or stale."""


def _fail(message: str) -> None:
    raise FullChaserProfileSuccessorError(message)


def _id(value: object, *, name: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        _fail(f"{name} must be one controlled identifier.")
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ImmutableModuleProductBinding:
    module_id: str
    schema_id: str
    schema_version: int
    run_path: str
    manifest_sha256: str
    payload_sha256: str
    selector_eligible: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        _id(self.module_id, name="module_id")
        if type(self.schema_id) is not str or not self.schema_id.startswith("palette."):
            _fail("schema_id must be one Palette schema identifier.")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            _fail("schema_version must be one positive exact integer.")
        if (
            type(self.run_path) is not str
            or _RUN_PATH_RE.fullmatch(self.run_path) is None
            or any(part in {"latest", "current", "selected", "authoritative"} for part in self.run_path.split("/"))
        ):
            _fail("run_path must name one exact immutable analysis child, not a selector.")
        _digest(self.manifest_sha256, name="manifest_sha256")
        _digest(self.payload_sha256, name="payload_sha256")
        if self.selector_eligible is not False or self.production_authority is not False:
            _fail("Full-profile inputs must remain selector-ineligible and non-authoritative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "run_path": self.run_path,
            "manifest_sha256": self.manifest_sha256,
            "payload_sha256": self.payload_sha256,
            "selector_eligible": False,
            "production_authority": False,
        }


@dataclass(frozen=True, slots=True)
class PreparedFullChaserProfile:
    recording_id: str
    profile_id: str
    profile_version: int
    readiness: str
    full_profile_complete: bool
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown full-profile array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _execution_waves(modules: Sequence[Any]) -> tuple[int, ...]:
    wave_by_id: dict[str, int] = {}
    waves: list[int] = []
    for module in modules:
        dependencies = tuple(module.depends_on)
        missing = [value for value in dependencies if value not in wave_by_id]
        if missing:
            _fail(
                f"Profile execution order is not dependency ordered for "
                f"{module.module_id!r}: {missing!r}."
            )
        wave = 0 if not dependencies else 1 + max(wave_by_id[value] for value in dependencies)
        wave_by_id[module.module_id] = wave
        waves.append(wave)
    return tuple(waves)


def prepare_full_chaser_profile_successor(
    *,
    profile: ChaserAnalysisProfile,
    applicability: ChaserProfileApplicabilityPlan,
    products: Sequence[ImmutableModuleProductBinding],
) -> PreparedFullChaserProfile:
    """Compose an exact profile/applicability/product envelope."""

    if type(profile) is not ChaserAnalysisProfile:
        raise TypeError("profile must be one validated ChaserAnalysisProfile.")
    if type(applicability) is not ChaserProfileApplicabilityPlan:
        raise TypeError("applicability must be one validated applicability plan.")
    if profile.profile_scope != "full":
        _fail("A full-profile successor requires profile_scope='full'.")
    if (
        applicability.profile_id != profile.profile_id
        or applicability.profile_version != profile.profile_version
        or applicability.profile_sha256 != profile.sha256
        or applicability.profile_scope != profile.profile_scope
    ):
        _fail("Applicability plan differs from the normalized profile identity.")
    selected = resolve_chaser_analysis_modules(
        profile,
        enable=applicability.explicit_enable,
        disable=applicability.explicit_disable,
    )
    selected_ids = tuple(module.module_id for module in selected)
    if selected_ids != applicability.execution_order:
        _fail("Applicability execution order differs from profile resolution.")
    module_by_id = {module.module_id: module for module in selected}
    product_rows = tuple(products)
    if any(type(row) is not ImmutableModuleProductBinding for row in product_rows):
        raise TypeError("products must contain immutable module product bindings.")
    product_by_id = {row.module_id: row for row in product_rows}
    if len(product_by_id) != len(product_rows):
        _fail("Module product bindings are duplicated.")
    unknown = sorted(set(product_by_id) - set(selected_ids))
    if unknown:
        _fail(f"Products bind unselected modules: {unknown!r}.")

    decisions = {row.module_id: row for row in applicability.module_decisions}
    normalized_products: list[dict[str, Any]] = []
    for module_id in selected_ids:
        decision = decisions[module_id]
        product = product_by_id.get(module_id)
        if decision.state is ModuleApplicabilityState.COMPLETE:
            if product is None:
                _fail(f"Completed module {module_id!r} lacks an immutable product binding.")
            module = module_by_id[module_id]
            if (
                product.schema_id != module.schema_id
                or product.schema_version != module.schema_version
            ):
                _fail(f"Product schema differs from profile module {module_id!r}.")
            normalized_products.append(product.to_dict())
        elif product is not None:
            _fail(
                f"Module {module_id!r} is {decision.state.value!r}, not complete, "
                "but has a product binding."
            )

    waves = _execution_waves(selected)
    product_bound = np.asarray(
        [module_id in product_by_id for module_id in selected_ids], dtype=bool
    )
    arrays = {
        "module_code": np.arange(1, len(selected_ids) + 1, dtype=np.uint16),
        "applicability_state_code": np.asarray(
            [STATE_CODES[decisions[module_id].state.value] for module_id in selected_ids],
            dtype=np.uint8,
        ),
        "dependency_count": np.asarray(
            [len(module.depends_on) for module in selected], dtype=np.uint16
        ),
        "execution_wave": np.asarray(waves, dtype=np.uint16),
        "product_bound": product_bound,
    }
    readonly = {name: _readonly(values) for name, values in arrays.items()}
    normalized_profile = profile.to_dict()
    applicability_record = applicability.record()
    product_records = sorted(normalized_products, key=lambda row: selected_ids.index(row["module_id"]))
    reuse_identity = {
        "profile_sha256": profile.sha256,
        "applicability_sha256": applicability.sha256,
        "product_manifest_sha256": {
            row["module_id"]: row["manifest_sha256"] for row in product_records
        },
        "product_payload_sha256": {
            row["module_id"]: row["payload_sha256"] for row in product_records
        },
    }
    complete = applicability.readiness is ProfileReadiness.COMPLETE
    if complete and not all(
        decisions[module_id].state
        in {ModuleApplicabilityState.COMPLETE, ModuleApplicabilityState.INAPPLICABLE}
        for module_id in selected_ids
    ):
        _fail("Complete full-profile readiness contains a nonterminal module.")
    manifest_body = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "recording_id": applicability.recording_id,
        "normalized_profile": normalized_profile,
        "normalized_profile_sha256": profile.sha256,
        "applicability_plan": applicability_record,
        "applicability_plan_sha256": applicability.sha256,
        "execution_order": list(selected_ids),
        "execution_waves": [
            {
                "wave": wave,
                "module_ids": [
                    module_id
                    for module_id, module_wave in zip(selected_ids, waves)
                    if module_wave == wave
                ],
            }
            for wave in sorted(set(waves))
        ],
        "module_products": product_records,
        "reuse_identity": reuse_identity,
        "reuse_identity_sha256": canonical_json_sha256(reuse_identity),
        "readiness": applicability.readiness.value,
        "full_profile_complete": complete,
        "array_declarations": [
            {
                "path": name,
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "content_sha256": array_values_sha256(values),
            }
            for name, values in sorted(readonly.items())
        ],
        "policy": {
            "module_execution": "dependency_order_with_independent_concurrency_waves",
            "reuse": "exact_profile_plan_and_all_product_digests_only",
            "inapplicable_modules": "terminal_without_product_binding",
            "blocked_or_pending_modules": "retained_without_completion_claim",
            "selector_activation": "separate_explicit_operation",
        },
        "identity_registries": {
            "module": {
                str(index + 1): module_id for index, module_id in enumerate(selected_ids)
            },
            "applicability_state": {
                str(code): state for state, code in STATE_CODES.items()
            },
        },
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = _freeze(
        {
            **manifest_body,
            "payload_digest": canonical_json_sha256(manifest_body),
        }
    )
    return PreparedFullChaserProfile(
        recording_id=applicability.recording_id,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        readiness=applicability.readiness.value,
        full_profile_complete=complete,
        arrays=MappingProxyType(readonly),
        manifest=manifest,
    )


__all__ = [
    "METHOD_ID",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "FullChaserProfileSuccessorError",
    "ImmutableModuleProductBinding",
    "PreparedFullChaserProfile",
    "prepare_full_chaser_profile_successor",
]
