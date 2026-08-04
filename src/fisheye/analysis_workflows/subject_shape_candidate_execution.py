"""Exact suite, identity, and coordinate evidence for subject-shape v4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.subject_shape_storage import (
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    build_subject_shape_storage_receipt,
)
from fisheye.shared.coordinate_frame_record import ARRAY_PAYLOAD_CANONICALIZATION
from fisheye.shared.coordinate_record import coordinate_record_sha256
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_MANIFEST_ATTR,
    SUBJECT_SHAPE_MANIFEST_SCHEMA_ID,
    SUBJECT_SHAPE_SCHEMA_VERSION,
    load_completed_ineligible_subject_shape_coordinate_publication,
    load_persisted_subject_shape_coordinate_publication,
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

SUBJECT_SHAPE_EXECUTION_FAMILY_ID = "subject_shape"
SUBJECT_SHAPE_EXECUTION_PROFILE_ID = SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
SUBJECT_SHAPE_EXECUTION_SOURCE_IDENTITY_SCHEMA_ID = (
    "palette.subject_shape_execution_source_identity"
)
SUBJECT_SHAPE_EXECUTION_SOURCE_IDENTITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _ArrayView:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]


class _GroupView:
    """Small read-only group surface used to replay a suite declaration."""

    path = ""

    def __init__(self) -> None:
        self._arrays: dict[str, _ArrayView] = {}
        self._groups: dict[str, _GroupView] = {}
        self.attrs: dict[str, Any] = {}

    def add(self, path: str, array: _ArrayView) -> None:
        current = self
        parts = path.split("/")
        for part in parts[:-1]:
            current = current._groups.setdefault(part, _GroupView())
        leaf = parts[-1]
        if leaf in current._arrays or leaf in current._groups:
            raise ValueError(f"subject-shape suite repeats path {path!r}")
        current._arrays[leaf] = array

    def arrays(self):
        return tuple(self._arrays.items())

    def groups(self):
        return tuple(self._groups.items())

    def array_keys(self):
        return tuple(self._arrays)

    def group_keys(self):
        return tuple(self._groups)

    def __getitem__(self, key: str):
        current: Any = self
        for component in key.split("/"):
            if component in current._arrays:
                current = current._arrays[component]
            else:
                current = current._groups[component]
        return current

    def get(self, key: str, default: Any = None):
        try:
            return self[key]
        except KeyError:
            return default


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _iter_arrays(group: Any, prefix: str = ""):
    for name in sorted(str(value) for value in group.array_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield path, group[name]
    for name in sorted(str(value) for value in group.group_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield from _iter_arrays(group[name], path)


def _array_digest(array: Any, *, block_rows: int = 65_536) -> str:
    dtype = np.dtype(array.dtype)
    shape = tuple(int(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
                "dtype": np.lib.format.dtype_to_descr(dtype),
                "shape": list(shape),
            }
        )
    )
    digest.update(b"\x00")
    if not shape:
        digest.update(np.asarray(array[...]).tobytes(order="C"))
        return digest.hexdigest()
    for start in range(0, shape[0], max(1, int(block_rows))):
        stop = min(shape[0], start + max(1, int(block_rows)))
        digest.update(np.ascontiguousarray(array[start:stop]).tobytes(order="C"))
    return digest.hexdigest()


def compute_subject_shape_logical_hashes(group: Any) -> dict[str, str]:
    """Hash every closed-v4 decoded array including exact dtype and shape."""

    return {path: _array_digest(array) for path, array in _iter_arrays(group)}


def _manifest_record(group: Any) -> tuple[Mapping[str, Any], str]:
    manifest = group.attrs.get(SUBJECT_SHAPE_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        raise ValueError("subject-shape source lacks its bound v4 manifest")
    digest = coordinate_record_sha256(manifest)
    if (
        manifest.get("schema_id") != SUBJECT_SHAPE_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != SUBJECT_SHAPE_SCHEMA_VERSION
        or group.attrs.get(f"{SUBJECT_SHAPE_MANIFEST_ATTR}_sha256") != digest
        or group.attrs.get("publication_manifest_sha256") != digest
    ):
        raise ValueError("subject-shape source manifest identity differs")
    return manifest, digest


def build_subject_shape_source_identity(
    source_group: Any,
    *,
    source_run_path: str,
    refined_authority_sha256: str,
) -> dict[str, object]:
    """Bind one exact v4 output table to its refined-mask authority."""

    if not source_run_path.startswith("analysis/subject_shape_runs/"):
        raise ValueError("subject-shape source path is not one explicit run")
    if (
        type(refined_authority_sha256) is not str
        or len(refined_authority_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in refined_authority_sha256)
    ):
        raise ValueError("refined subject-mask authority is not lowercase SHA-256")
    manifest, manifest_sha256 = _manifest_record(source_group)
    source_refined = source_group.attrs.get("source_refined_subject_masks_run")
    if type(source_refined) is not str or not source_refined or "/" in source_refined:
        raise ValueError("subject-shape source refined run is invalid")
    logical = compute_subject_shape_logical_hashes(source_group)
    if set(logical) != set(manifest.get("arrays", {})):
        raise ValueError("subject-shape source arrays differ from its v4 manifest")
    document: dict[str, object] = {
        "schema_id": SUBJECT_SHAPE_EXECUTION_SOURCE_IDENTITY_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_EXECUTION_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_run_path": source_run_path,
        "source_schema_id": source_group.attrs.get("schema_id"),
        "source_schema_version": source_group.attrs.get("schema_version"),
        "source_profile_id": source_group.attrs.get("profile_id"),
        "source_manifest_attr": SUBJECT_SHAPE_MANIFEST_ATTR,
        "source_manifest_sha256": manifest_sha256,
        "source_refined_subject_masks_run": source_refined,
        "source_refined_authority_sha256": refined_authority_sha256,
        "source_array_logical_hashes": logical,
    }
    canonical_json_bytes(document)
    return document


def subject_shape_source_identity_sha256(
    source_group: Any,
    *,
    source_run_path: str,
    refined_authority_sha256: str,
) -> str:
    return canonical_json_sha256(
        build_subject_shape_source_identity(
            source_group,
            source_run_path=source_run_path,
            refined_authority_sha256=refined_authority_sha256,
        )
    )


def build_subject_shape_coordinate_evidence(
    *,
    source_publication: Any,
    candidate_publication: Any,
) -> dict[str, object]:
    """Return canonical-producer evidence from two freshly loaded authorities."""

    source_manifest_sha256 = source_publication.manifest.record_sha256
    refined_sha256 = source_publication.source.scientific_manifest.record_sha256
    candidate_manifest_sha256 = candidate_publication.manifest.record_sha256
    if candidate_publication.source.scientific_manifest.record_sha256 != refined_sha256:
        raise ValueError("subject-shape candidate refined authority differs")
    validation = {
        "schema_id": "palette.subject_shape_coordinate_execution_validation",
        "schema_version": 1,
        "source_run_path": source_publication.run_path,
        "source_manifest_sha256": source_manifest_sha256,
        "candidate_run_path": candidate_publication.run_path,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "refined_subject_mask_scientific_manifest_sha256": refined_sha256,
        "source_row_count": int(source_publication.row_identity.leading_dimension),
        "candidate_row_count": int(
            candidate_publication.row_identity.leading_dimension
        ),
    }
    if validation["source_row_count"] != validation["candidate_row_count"]:
        raise ValueError("subject-shape candidate row count differs")
    validation_sha256 = canonical_json_sha256(validation)
    return {
        "role": "canonical_producer",
        "status": "verified_canonical_publication",
        "source_authority_digests": [
            {"role": "source_refined_subject_masks", "sha256": refined_sha256},
            {"role": "source_subject_shape_v4", "sha256": source_manifest_sha256},
        ],
        "published_authority_sha256": candidate_manifest_sha256,
        "published_authority_ref": (
            f"/{candidate_publication.run_path}@{SUBJECT_SHAPE_MANIFEST_ATTR}"
        ),
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": f"{__name__}:build_subject_shape_coordinate_evidence",
        "validation_receipt_sha256": validation_sha256,
        "coordinate_gate_passed": True,
    }


def load_subject_shape_coordinate_evidence(
    root: Any,
    *,
    source_run_path: str,
    candidate_run_path: str,
    candidate_owner: str,
) -> dict[str, object]:
    """Freshly validate the live source and ineligible candidate authorities."""

    return build_subject_shape_coordinate_evidence(
        source_publication=load_persisted_subject_shape_coordinate_publication(
            root,
            source_run_path,
        ),
        candidate_publication=(
            load_completed_ineligible_subject_shape_coordinate_publication(
                root,
                candidate_run_path,
                expected_publication_owner=candidate_owner,
            )
        ),
    )


def build_subject_shape_execution_suite(
    source_group: Any,
    *,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    """Plan the exact current-v4 access-aware physical candidate."""

    receipt = build_subject_shape_storage_receipt(source_group, phase="bound")
    suite = build_analysis_benchmark_suite(
        family_id=SUBJECT_SHAPE_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id="explicit_subject_shape_v4_run",
            dimensions=receipt.dimensions,
            description=(
                "Exact canonical subject-shape full-anatomy v4 arrays recomputed "
                "from their bound refined subject-mask authority."
            ),
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_subject_shape_execution_suite(SUBJECT_SHAPE_EXECUTION_FAMILY_ID, suite)
    return suite


def _role_for_authority(value: object) -> str:
    mapping = {
        "lineage_index": "row_identity",
        "compatibility_alias": "compatibility_row_lineage",
        "quality_diagnostic": "validity_or_flag",
        "semantic_metadata": "sample_axis",
        "scientific_authority": "scientific_authority",
    }
    try:
        return mapping[str(value)]
    except KeyError as exc:
        raise ValueError("subject-shape suite authority role is unsupported") from exc


def require_subject_shape_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Replay every declaration and storage decision from observed suite facts."""

    if stage_id != SUBJECT_SHAPE_EXECUTION_FAMILY_ID:
        raise ValueError("subject-shape suite validator owns only subject_shape")
    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if payload["family_id"] != stage_id:
        raise ValueError("subject-shape suite family differs")
    receipt = payload["storage_plan_receipt"]
    receipt_payload = receipt["payload"]
    if receipt_payload["storage_profile"]["profile_id"] != (
        SUBJECT_SHAPE_EXECUTION_PROFILE_ID
    ):
        raise ValueError("subject-shape suite storage profile differs")
    records = receipt_payload["arrays"]
    if not isinstance(records, list) or not records:
        raise ValueError("subject-shape suite has no closed v4 arrays")

    group = _GroupView()
    roles: dict[str, dict[str, str]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("subject-shape suite array record is not an object")
        facts = record.get("observed_facts")
        declaration = record.get("declaration")
        if not isinstance(facts, Mapping) or not isinstance(declaration, Mapping):
            raise ValueError("subject-shape suite array lacks facts or declaration")
        path = facts.get("path")
        shape = facts.get("shape")
        dtype = facts.get("dtype")
        if (
            type(path) is not str
            or not isinstance(shape, list)
            or type(dtype) is not str
        ):
            raise ValueError("subject-shape suite observed facts are malformed")
        group.add(
            path,
            _ArrayView(
                shape=tuple(int(value) for value in shape),
                dtype=np.dtype(dtype),
            ),
        )
        roles[path] = {"role": _role_for_authority(declaration.get("authority_role"))}
    group.attrs[SUBJECT_SHAPE_MANIFEST_ATTR] = {
        "schema_id": SUBJECT_SHAPE_MANIFEST_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
        "schema_inventory": {
            "phase": "bound",
            "closed_array_inventory": True,
            "arrays": roles,
        },
    }
    replayed = build_subject_shape_storage_receipt(group, phase="bound").as_manifest()
    if canonical_json_bytes(replayed) != canonical_json_bytes(receipt):
        raise ValueError(
            "subject-shape suite differs from the exact live v4 storage projection"
        )


__all__ = [
    "SUBJECT_SHAPE_EXECUTION_FAMILY_ID",
    "SUBJECT_SHAPE_EXECUTION_PROFILE_ID",
    "SUBJECT_SHAPE_EXECUTION_SOURCE_IDENTITY_SCHEMA_ID",
    "SUBJECT_SHAPE_EXECUTION_SOURCE_IDENTITY_SCHEMA_VERSION",
    "build_subject_shape_coordinate_evidence",
    "build_subject_shape_execution_suite",
    "build_subject_shape_source_identity",
    "compute_subject_shape_logical_hashes",
    "load_subject_shape_coordinate_evidence",
    "require_subject_shape_execution_suite",
    "subject_shape_source_identity_sha256",
]
