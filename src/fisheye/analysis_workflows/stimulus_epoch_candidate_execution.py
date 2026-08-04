"""Exact suite binding for typed stimulus-epoch v1-to-v2 execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.exact_tabular_storage import (
    build_exact_tabular_storage_receipt,
)
from fisheye.analysis.stimulus_epoch_schema import (
    STIMULUS_SOURCE_FINGERPRINT_ALGORITHM,
    build_stimulus_epoch_array_declarations,
    stimulus_group_logical_fingerprint as compute_stimulus_group_logical_fingerprint,
    stimulus_epoch_logical_content_sha256,
    validate_legacy_stimulus_epoch_source,
)
from fisheye.shared.run_lineage_fingerprint import (
    canonical_lineage_json,
    compute_run_lineage_hash,
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

STIMULUS_EPOCH_EXECUTION_FAMILY_ID = "stimulus_epochs"
STIMULUS_EPOCH_EXECUTION_ARRAY_COUNT = 12
STIMULUS_EPOCH_EXECUTION_PROFILE_ID = "published_http_v1"
STIMULUS_EPOCH_SOURCE_IDENTITY_SCHEMA_ID = (
    "palette.stimulus_epoch_execution_source_identity"
)
STIMULUS_EPOCH_SOURCE_IDENTITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _ArrayView:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]


class _GroupView:
    def __init__(self) -> None:
        self._arrays: dict[str, _ArrayView] = {}
        self._groups: dict[str, _GroupView] = {}

    def add(self, path: str, array: _ArrayView) -> None:
        parts = path.split("/")
        current = self
        for part in parts[:-1]:
            current = current._groups.setdefault(part, _GroupView())
        name = parts[-1]
        if name in current._arrays or name in current._groups:
            raise ValueError(f"stimulus-epoch suite repeats path {path!r}")
        current._arrays[name] = array

    def arrays(self):
        return tuple(self._arrays.items())

    def groups(self):
        return tuple(self._groups.items())


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def compute_stimulus_epoch_logical_hashes(source_group: Any) -> dict[str, str]:
    """Hash the exact twelve decoded arrays, including dtype and shape."""

    declarations = build_stimulus_epoch_array_declarations(
        source_group,
        byte_planner_adopted=False,
    )
    hashes: dict[str, str] = {}
    for declaration in declarations:
        array = _array_at_path(source_group, declaration.path)
        values = np.ascontiguousarray(array[...])
        digest = hashlib.sha256()
        digest.update(str(np.dtype(array.dtype)).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        hashes[declaration.path] = digest.hexdigest()
    return hashes


def _validated_lineage_identity(source_group: Any) -> tuple[str, str]:
    lineage_json = source_group.attrs.get("lineage_payload_json")
    if type(lineage_json) is not str:
        raise ValueError("stimulus-epoch source lacks canonical lineage JSON")
    try:
        payload = json.loads(lineage_json)
    except json.JSONDecodeError as exc:
        raise ValueError("stimulus-epoch source lineage JSON is invalid") from exc
    if type(payload) is not dict or lineage_json != canonical_lineage_json(payload):
        raise ValueError("stimulus-epoch source lineage is not canonical")
    lineage_hash = compute_run_lineage_hash(payload)
    for field in ("source_fingerprint", "source_lineage_hash", "lineage_hash"):
        if source_group.attrs.get(field) != lineage_hash:
            raise ValueError(f"stimulus-epoch source {field} differs from lineage")
    return lineage_hash, hashlib.sha256(lineage_json.encode("utf-8")).hexdigest()


def build_stimulus_epoch_source_identity(
    source_group: Any,
    *,
    source_stimulus_fingerprint: str,
) -> dict[str, object]:
    """Bind one legacy source table to its stimulus tree and temporal axis."""

    errors = validate_legacy_stimulus_epoch_source(source_group)
    if errors:
        raise ValueError(
            "invalid legacy stimulus-epoch execution source: " + "; ".join(errors)
        )
    if (
        type(source_stimulus_fingerprint) is not str
        or len(source_stimulus_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_stimulus_fingerprint
        )
    ):
        raise ValueError("source stimulus fingerprint is not lowercase SHA-256")
    source_stimulus_run = source_group.attrs.get("source_stimulus_run")
    source_stimulus_path = source_group.attrs.get("source_stimulus_path")
    if (
        type(source_stimulus_run) is not str
        or not source_stimulus_run
        or source_stimulus_path != f"analysis/stimulus_runs/{source_stimulus_run}"
    ):
        raise ValueError("stimulus-epoch source stimulus binding is invalid")
    lineage_hash, lineage_payload_sha256 = _validated_lineage_identity(source_group)
    document: dict[str, object] = {
        "schema_id": STIMULUS_EPOCH_SOURCE_IDENTITY_SCHEMA_ID,
        "schema_version": STIMULUS_EPOCH_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_schema_id": source_group.attrs.get("schema_id"),
        "source_schema_version": source_group.attrs.get("schema_version"),
        "source_epoch_lineage_hash": lineage_hash,
        "source_epoch_lineage_payload_sha256": lineage_payload_sha256,
        "source_epoch_logical_content_sha256": (
            stimulus_epoch_logical_content_sha256(source_group)
        ),
        "source_array_logical_hashes": compute_stimulus_epoch_logical_hashes(
            source_group
        ),
        "source_stimulus_run": source_stimulus_run,
        "source_stimulus_path": source_stimulus_path,
        "source_stimulus_fingerprint_algorithm": (
            STIMULUS_SOURCE_FINGERPRINT_ALGORITHM
        ),
        "source_stimulus_fingerprint": source_stimulus_fingerprint,
        "fps": source_group.attrs.get("fps"),
        "total_frames": source_group.attrs.get("total_frames"),
    }
    canonical_json_bytes(document)
    return document


def stimulus_epoch_source_identity_sha256(
    source_group: Any,
    *,
    source_stimulus_fingerprint: str,
) -> str:
    return canonical_json_sha256(
        build_stimulus_epoch_source_identity(
            source_group,
            source_stimulus_fingerprint=source_stimulus_fingerprint,
        )
    )


def build_stimulus_epoch_temporal_axis_evidence(
    *,
    source_run_path: str,
    source_group: Any,
    source_stimulus_fingerprint: str,
) -> dict[str, object]:
    """Build promotable evidence for the immutable acquisition-frame axis."""

    if not source_run_path.startswith("analysis/stimulus_epoch_runs/"):
        raise ValueError("stimulus-epoch source path is not one immutable run")
    source_identity = build_stimulus_epoch_source_identity(
        source_group,
        source_stimulus_fingerprint=source_stimulus_fingerprint,
    )
    temporal_axis = {
        "schema_id": "palette.stimulus_epoch_temporal_axis_validation",
        "schema_version": 1,
        "source_run_ref": f"/{source_run_path}",
        "source_identity_sha256": canonical_json_sha256(source_identity),
        "source_epoch_lineage_hash": source_identity["source_epoch_lineage_hash"],
        "source_epoch_logical_content_sha256": source_identity[
            "source_epoch_logical_content_sha256"
        ],
        "source_stimulus_path": source_identity["source_stimulus_path"],
        "source_stimulus_fingerprint": source_stimulus_fingerprint,
        "fps": source_identity["fps"],
        "total_frames": source_identity["total_frames"],
    }
    temporal_axis_sha256 = canonical_json_sha256(temporal_axis)
    authorities = [
        {
            "role": "source_stimulus_epoch_lineage",
            "sha256": source_identity["source_epoch_lineage_hash"],
        },
        {
            "role": "source_stimulus_logical_tree",
            "sha256": source_stimulus_fingerprint,
        },
    ]
    return {
        "role": "temporal_axis_only",
        "status": "verified_temporal_axis",
        "source_authority_digests": authorities,
        "published_authority_sha256": None,
        "published_authority_ref": None,
        "temporal_axis_sha256": temporal_axis_sha256,
        "temporal_axis_ref": f"/{source_run_path}#stimulus_epoch_temporal_axis_v1",
        "validator_ref": (f"{__name__}:build_stimulus_epoch_temporal_axis_evidence"),
        "validation_receipt_sha256": temporal_axis_sha256,
        "coordinate_gate_passed": True,
    }


def build_stimulus_epoch_execution_suite(
    source_group: Any,
    *,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    """Build the exact candidate suite from one validated legacy-v1 source."""

    declarations = build_stimulus_epoch_array_declarations(
        source_group,
        byte_planner_adopted=True,
    )
    receipt = build_exact_tabular_storage_receipt(
        source_group,
        declarations=declarations,
        profile=get_storage_profile(STIMULUS_EPOCH_EXECUTION_PROFILE_ID),
    )
    suite = build_analysis_benchmark_suite(
        family_id=STIMULUS_EPOCH_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id="explicit_stimulus_epoch_run",
            dimensions=receipt.dimensions,
            description=(
                "Exact explicit legacy-v1 stimulus epoch table rematerialized as "
                "primitive byte-planned v2 arrays."
            ),
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_stimulus_epoch_execution_suite(
        STIMULUS_EPOCH_EXECUTION_FAMILY_ID,
        suite,
    )
    return suite


def require_stimulus_epoch_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Require the exact twelve-array live stimulus-epoch v2 projection."""

    if stage_id != STIMULUS_EPOCH_EXECUTION_FAMILY_ID:
        raise ValueError("stimulus-epoch suite validator owns only stimulus_epochs")
    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if payload["family_id"] != stage_id:
        raise ValueError("stimulus-epoch suite family differs")
    receipt_payload = payload["storage_plan_receipt"]["payload"]
    if receipt_payload["storage_profile"]["profile_id"] != (
        STIMULUS_EPOCH_EXECUTION_PROFILE_ID
    ):
        raise ValueError("stimulus-epoch suite storage profile differs")
    records = receipt_payload["arrays"]
    if not isinstance(records, list) or len(records) != (
        STIMULUS_EPOCH_EXECUTION_ARRAY_COUNT
    ):
        raise ValueError("stimulus-epoch suite must contain exactly twelve arrays")

    group = _GroupView()
    observed: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("stimulus-epoch suite contains a non-object array record")
        facts = record.get("observed_facts")
        declaration = record.get("declaration")
        if not isinstance(facts, Mapping) or not isinstance(declaration, Mapping):
            raise ValueError("stimulus-epoch suite array lacks facts or declaration")
        path = facts.get("path")
        shape = facts.get("shape")
        dtype = facts.get("dtype")
        if (
            type(path) is not str
            or not isinstance(shape, list)
            or type(dtype) is not str
        ):
            raise ValueError("stimulus-epoch suite array facts are not canonical")
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
        for declaration in build_stimulus_epoch_array_declarations(
            group,
            byte_planner_adopted=True,
        )
    ]
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError(
            "stimulus-epoch suite declarations differ from the live v2 projection"
        )


__all__ = [
    "STIMULUS_EPOCH_EXECUTION_ARRAY_COUNT",
    "STIMULUS_EPOCH_EXECUTION_FAMILY_ID",
    "STIMULUS_EPOCH_EXECUTION_PROFILE_ID",
    "STIMULUS_EPOCH_SOURCE_IDENTITY_SCHEMA_ID",
    "STIMULUS_EPOCH_SOURCE_IDENTITY_SCHEMA_VERSION",
    "build_stimulus_epoch_source_identity",
    "build_stimulus_epoch_temporal_axis_evidence",
    "build_stimulus_epoch_execution_suite",
    "compute_stimulus_group_logical_fingerprint",
    "compute_stimulus_epoch_logical_hashes",
    "require_stimulus_epoch_execution_suite",
    "stimulus_epoch_source_identity_sha256",
]
