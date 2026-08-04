"""Exact suite and source evidence for typed chaser-distance base execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.chaser_distance_base_schema import (
    build_chaser_distance_base_declarations,
)
from fisheye.analysis.chaser_distance_base_storage import (
    base_logical_hashes,
    build_base_storage_receipt,
)
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.storage_profiles import get_storage_profile

CHASER_DISTANCE_EXECUTION_FAMILY_ID = "chaser_distance"
CHASER_DISTANCE_EXECUTION_ARRAY_COUNT = 30
CHASER_DISTANCE_EXECUTION_PROFILE_ID = "published_http_v1"


@dataclass(frozen=True)
class _ArrayView:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    @property
    def ndim(self) -> int:
        return len(self.shape)


class _GroupView:
    def __init__(self) -> None:
        self._arrays: dict[str, _ArrayView] = {}
        self._groups: dict[str, _GroupView] = {}

    def add(self, path: str, array: _ArrayView) -> None:
        parts = path.split("/")
        current = self
        for part in parts[:-1]:
            current = current._groups.setdefault(part, _GroupView())
        current._arrays[parts[-1]] = array

    def __getitem__(self, name: str) -> Any:
        if name in self._arrays:
            return self._arrays[name]
        return self._groups[name]


def build_chaser_distance_execution_suite(
    source_group: Any,
    *,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    """Plan the exact sealed 30-array source projection."""

    receipt = build_base_storage_receipt(
        source_group,
        profile=get_storage_profile(CHASER_DISTANCE_EXECUTION_PROFILE_ID),
    )
    suite = build_analysis_benchmark_suite(
        family_id=CHASER_DISTANCE_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id="explicit_chaser_distance_run",
            dimensions=receipt.dimensions,
            description=(
                "Exact sealed canonical chaser-distance base rematerialized into "
                "the primitive byte-planned v2 candidate."
            ),
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_chaser_distance_execution_suite(
        CHASER_DISTANCE_EXECUTION_FAMILY_ID,
        suite,
    )
    return suite


def require_chaser_distance_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Reconstruct every declaration in the sealed-base suite."""

    if stage_id != CHASER_DISTANCE_EXECUTION_FAMILY_ID:
        raise ValueError("chaser-distance suite validator owns only chaser_distance")
    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if payload["family_id"] != stage_id:
        raise ValueError("chaser-distance suite family differs")
    receipt = payload["storage_plan_receipt"]["payload"]
    if receipt["storage_profile"]["profile_id"] != (
        CHASER_DISTANCE_EXECUTION_PROFILE_ID
    ):
        raise ValueError("chaser-distance suite storage profile differs")
    records = receipt["arrays"]
    if not isinstance(records, list) or len(records) != (
        CHASER_DISTANCE_EXECUTION_ARRAY_COUNT
    ):
        raise ValueError("chaser-distance suite must contain exactly 30 arrays")

    group = _GroupView()
    observed: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("chaser-distance suite array record is invalid")
        facts = record.get("observed_facts")
        declaration = record.get("declaration")
        if not isinstance(facts, Mapping) or not isinstance(declaration, Mapping):
            raise ValueError("chaser-distance suite array lacks facts/declaration")
        path = facts.get("path")
        shape = facts.get("shape")
        dtype = facts.get("dtype")
        if (
            type(path) is not str
            or not isinstance(shape, list)
            or type(dtype) is not str
        ):
            raise ValueError("chaser-distance suite facts are not canonical")
        group.add(
            path,
            _ArrayView(
                shape=tuple(int(value) for value in shape),
                dtype=np.dtype(dtype),
            ),
        )
        observed.append(declaration)
    expected = [
        declaration.as_manifest()
        for declaration in build_chaser_distance_base_declarations(group)
    ]
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError(
            "chaser-distance suite declarations differ from the sealed projection"
        )


def chaser_distance_decoded_identity_sha256(source_group: Any) -> str:
    declarations = build_chaser_distance_base_declarations(source_group)
    return canonical_json_sha256(base_logical_hashes(source_group, declarations))


def build_chaser_distance_source_preservation_evidence(
    source_binding: Mapping[str, Any],
) -> dict[str, object]:
    """Bind every sealed source record without minting a new authority."""

    roles = (
        "chaser_collection",
        "epoch_window_identity",
        "input_authority",
        "measurement_authority",
        "publication_seal",
        "row_identity",
        "surface_manifest",
    )
    authorities: list[dict[str, str]] = []
    for role in roles:
        pointer = source_binding.get(role)
        if not isinstance(pointer, Mapping) or set(pointer) != {
            "record_ref",
            "record_sha256",
        }:
            raise ValueError(f"chaser-distance source {role} pointer is invalid")
        digest = pointer["record_sha256"]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"chaser-distance source {role} digest is invalid")
        authorities.append({"role": role, "sha256": digest})
    validation = {
        "schema_id": "palette.chaser_distance_source_preservation_validation",
        "schema_version": 1,
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "source_run_path": source_binding.get("source_run_path"),
        "projection_id": "sealed_base_30_arrays_v1",
        "authority_minting": "forbidden_diagnostic_projection",
    }
    return {
        "role": "canonical_producer",
        "status": "verified_source_preservation_nonminting",
        "source_authority_digests": authorities,
        "published_authority_sha256": None,
        "published_authority_ref": None,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": (
            f"{__name__}:build_chaser_distance_source_preservation_evidence"
        ),
        "validation_receipt_sha256": canonical_json_sha256(validation),
        "coordinate_gate_passed": False,
    }


__all__ = [
    "CHASER_DISTANCE_EXECUTION_ARRAY_COUNT",
    "CHASER_DISTANCE_EXECUTION_FAMILY_ID",
    "CHASER_DISTANCE_EXECUTION_PROFILE_ID",
    "build_chaser_distance_execution_suite",
    "build_chaser_distance_source_preservation_evidence",
    "chaser_distance_decoded_identity_sha256",
    "require_chaser_distance_execution_suite",
]
