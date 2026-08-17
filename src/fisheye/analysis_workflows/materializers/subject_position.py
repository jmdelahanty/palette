"""Immutable, selector-ineligible observation subject-position publications.

The source adapters deliberately live outside this module.  They prepare an
already-authorized row-aligned result and bind its scientific records here.
This boundary only plans, validates, and publishes that exact prepared result;
it never resolves ``latest`` or chooses a detection, keypoint, or mask source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.coordinate_descriptor import (
    canonical_coordinate_descriptor_v2_attrs,
    load_canonical_coordinate_descriptor_attrs,
    parse_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.subject_position_expression import (
    canonicalize_estimator_profile,
    estimator_profile_digest,
)
from fisheye.shared.subject_position_prepared_input import (
    SubjectPositionPreparedInput,
)
from fisheye.shared.subject_position_storage import (
    OBSERVATION_POSITION_ARRAYS,
    OBSERVATION_POSITION_MANDATORY_ARRAYS,
    OBSERVATION_POSITION_NAMESPACE,
    canonical_observation_position_arrays_sha256,
    canonical_observation_position_logical_metadata,
    validate_observation_position_arrays,
)
from fisheye.shared.subject_position_types import SUBJECT_POSITION_STORAGE_SCHEMA_ID
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisStoragePlanReceipt,
    AnalysisArrayStorageFacts,
    analysis_storage_plan_receipt_from_manifest,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_contracts import (
    ArrayContract,
    BOOL,
    FLOAT32,
    INT64,
    UINT16,
    UINT64,
)
from fisheye.shared.zarr.array_factory import (
    create_array_from_plan,
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import (
    PUBLISHED_HTTP_V1,
    StorageProfile,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


SUBJECT_POSITION_PUBLISH_SCHEMA_ID = "palette.subject_position_run_publish"
SUBJECT_POSITION_PUBLISH_SCHEMA_VERSION = 1
SUBJECT_POSITION_MANIFEST_SCHEMA_ID = "palette.subject_position_run_manifest"
SUBJECT_POSITION_MANIFEST_SCHEMA_VERSION = 1
SUBJECT_POSITION_PARENT_PATH = "analysis/subject_position_runs/observation"
SUBJECT_POSITION_STORAGE_PLAN_ATTR = "subject_position_storage_plan"
SUBJECT_POSITION_MANIFEST_ATTR = "subject_position_manifest"
SUBJECT_POSITION_MANIFEST_DIGEST_ATTR = "subject_position_manifest_sha256"
SUBJECT_POSITION_PUBLICATION_ATTEMPT_ATTR = "subject_position_publication_attempt_uuid"
SUBJECT_POSITION_COORDINATE_SURFACE_ATTR = "coordinate_surface_contract"
SUBJECT_POSITION_PUBLISH_POLICY = "subject_position_atomic_nonpromoting_v1"
SUBJECT_POSITION_RETRY_POLICY = "new_immutable_run_name_required"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "authoritative_run_provenance",
)


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


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
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


def _safe_run_name(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None
    ):
        raise ValueError(f"Invalid subject-position run name: {value!r}.")
    return value


SubjectPositionPreparation = SubjectPositionPreparedInput


@dataclass(frozen=True)
class SubjectPositionRunPlan:
    """Read-only exact publication plan for one immutable attempt."""

    source_zarr: Path
    run_name: str
    scratch_root: Path
    local_zarr: Path
    prepared: SubjectPositionPreparedInput
    storage_profile: StorageProfile
    storage_receipt: AnalysisStoragePlanReceipt
    parent_selector_attrs: Mapping[str, Any]
    publication_attempt_uuid: str
    run_provenance: Mapping[str, Any]

    @property
    def parent_path(self) -> str:
        return SUBJECT_POSITION_PARENT_PATH

    @property
    def run_path(self) -> str:
        return f"{self.parent_path}/{self.run_name}"

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr.joinpath(*self.run_path.split("/"))

    @property
    def final_manifest_sha256(self) -> str:
        return subject_position_manifest_digest(
            build_subject_position_manifest(self, status=RUN_STATUS_COMPLETE)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "palette.subject_position_run_plan",
            "schema_version": 1,
            "source_zarr": str(self.source_zarr),
            "parent_path": self.parent_path,
            "run_name": self.run_name,
            "run_path": self.run_path,
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
            "publication_attempt_uuid": self.publication_attempt_uuid,
            "parent_selector_attrs": dict(self.parent_selector_attrs),
            "storage_profile": self.storage_profile.as_manifest(),
            "storage_plan_digest": self.storage_receipt["payload_digest"]
            if isinstance(self.storage_receipt, Mapping)
            else self.storage_receipt.as_manifest()["payload_digest"],
            "publication_policy": SUBJECT_POSITION_PUBLISH_POLICY,
            "retry_policy": SUBJECT_POSITION_RETRY_POLICY,
        }


def _array_contracts() -> dict[str, ArrayContract]:
    common = SUBJECT_POSITION_STORAGE_SCHEMA_ID
    return {
        "position_xy": ArrayContract(
            common, 1, FLOAT32, ("N", 2), ("observation", "xy"),
            "continuous source-camera subject position", coordinate_space="source_camera_image_px.top_left_y_down.v1",
        ),
        "valid": ArrayContract(
            common, 1, BOOL, ("N",), ("observation",), "estimator validity",
        ),
        "failure_reason_codes": ArrayContract(
            common, 1, UINT16, ("N",), ("observation",), "controlled failure reason code",
        ),
        "instance_key": ArrayContract(
            common, 1, UINT64, ("N",), ("observation",), "observation identity",
        ),
        "source_acquisition_frame_index": ArrayContract(
            common, 1, INT64, ("N",), ("observation",), "source acquisition frame identity",
        ),
        "source_row_index": ArrayContract(
            common, 1, INT64, ("N",), ("observation",), "immutable source row identity",
        ),
        "support/source_points_xy": ArrayContract(
            common, 1, FLOAT32, ("N", "P", 2), ("observation", "point", "xy"),
            "ordered contributing source points", coordinate_space="source_camera_image_px.top_left_y_down.v1",
        ),
        "support/source_points_valid": ArrayContract(
            common, 1, BOOL, ("N", "P"), ("observation", "point"), "source point validity",
        ),
        "support/source_point_reason_codes": ArrayContract(
            common, 1, UINT16, ("N", "P"), ("observation", "point"), "source point reason code",
        ),
        "support/source_point_confidence": ArrayContract(
            common, 1, FLOAT32, ("N", "P"), ("observation", "point"), "source point confidence",
        ),
    }


def _declarations(arrays: Mapping[str, np.ndarray]) -> tuple[AnalysisArrayDeclaration, ...]:
    contracts = _array_contracts()
    roles = {
        "position_xy": AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        "valid": AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        "failure_reason_codes": AnalysisAuthorityRole.QUALITY_DIAGNOSTIC,
        "instance_key": AnalysisAuthorityRole.LINEAGE_INDEX,
        "source_acquisition_frame_index": AnalysisAuthorityRole.LINEAGE_INDEX,
        "source_row_index": AnalysisAuthorityRole.LINEAGE_INDEX,
    }
    declarations = []
    for path in sorted(arrays):
        declarations.append(
            AnalysisArrayDeclaration(
                path=path,
                contract=contracts[path],
                required=path in OBSERVATION_POSITION_MANDATORY_ARRAYS,
                access_pattern="per_row",
                write_mode="immutable",
                authority_role=roles.get(path, AnalysisAuthorityRole.QUALITY_DIAGNOSTIC),
                fill_semantics="explicit contract fill value; logical values are fully materialized",
                null_semantics="validity and reason codes define row missingness",
                physical_policy_owner="fisheye.analysis_workflows.materializers.subject_position",
                byte_planner_adopted=True,
            )
        )
    return tuple(declarations)


def _storage_receipt(
    arrays: Mapping[str, np.ndarray],
    profile: StorageProfile,
) -> AnalysisStoragePlanReceipt:
    facts = {
        path: AnalysisArrayStorageFacts(
            path=path,
            shape=tuple(int(value) for value in array.shape),
            dtype=array.dtype,
            access_unit_semantics="one complete observation row with all fixed trailing axes indivisible",
        )
        for path, array in arrays.items()
    }
    return plan_analysis_storage(
        _declarations(arrays),
        facts,
        profile=profile,
        dimensions={"N": int(arrays["position_xy"].shape[0])},
    )


def _selector_attrs_from_parent(parent: Any) -> dict[str, Any]:
    attrs = getattr(parent, "attrs", {})
    return {
        key: json_attr_safe(attrs[key])
        for key in _SELECTOR_ATTRS
        if key in attrs
    }


def _get_node(group: Any, path: str) -> Any:
    node = group
    for component in path.strip("/").split("/"):
        node = node[component]
    return node


def plan_subject_position_run(
    source_zarr: str | Path,
    prepared: SubjectPositionPreparedInput,
    *,
    run_name: str | None = None,
    scratch_root: str | Path,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    publication_attempt_uuid: str | None = None,
) -> SubjectPositionRunPlan:
    """Build a read-only plan without resolving or mutating an authoritative run."""

    if not isinstance(prepared, SubjectPositionPreparedInput):
        raise TypeError("prepared must be SubjectPositionPreparedInput.")
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Authoritative analysis Zarr does not exist: {source}")
    if scratch == source or scratch.is_relative_to(source):
        raise ValueError("Scratch root must be outside the authoritative archive.")
    if not isinstance(storage_profile, StorageProfile):
        raise TypeError("storage_profile must be a StorageProfile.")

    attempt = str(uuid.UUID(publication_attempt_uuid)) if publication_attempt_uuid else str(uuid.uuid4())
    chosen_name = _safe_run_name(run_name or f"position_{uuid.UUID(attempt).hex}")
    local_zarr = scratch / f"{chosen_name}.zarr"
    target = source.joinpath(*f"{SUBJECT_POSITION_PARENT_PATH}/{chosen_name}".split("/"))
    if target.exists():
        raise FileExistsError(f"Refusing existing subject-position attempt: {target}")
    if local_zarr.exists():
        raise FileExistsError(f"Refusing existing local subject-position attempt: {local_zarr}")

    root = open_zarr_root(source, mode="r", use_consolidated=False)
    parent = None
    try:
        parent = _get_node(root, SUBJECT_POSITION_PARENT_PATH)
    except (KeyError, ValueError):
        pass
    parent_selector_attrs = _selector_attrs_from_parent(parent) if parent is not None else {}
    receipt = _storage_receipt(prepared.arrays, storage_profile)
    run_provenance = build_writer_run_provenance(
        command="subject_position_run_materializer",
        params={
            "estimator_sha256": prepared.estimator_sha256,
            "source_sha256": prepared.source_sha256,
            "policy_sha256": prepared.policy_sha256,
            "storage_profile_id": storage_profile.profile_id,
        },
        input_run_ids={"source_record_sha256": prepared.source_sha256},
        cwd=source,
        include_system_context=False,
    )
    return SubjectPositionRunPlan(
        source_zarr=source,
        run_name=chosen_name,
        scratch_root=scratch,
        local_zarr=local_zarr,
        prepared=prepared,
        storage_profile=storage_profile,
        storage_receipt=receipt,
        parent_selector_attrs=MappingProxyType(dict(parent_selector_attrs)),
        publication_attempt_uuid=attempt,
        run_provenance=_freeze_json(run_provenance),
    )


def build_subject_position_manifest(
    plan: SubjectPositionRunPlan,
    *,
    status: str,
) -> dict[str, Any]:
    """Build the exact digest-bound run manifest for one lifecycle state."""

    if status not in {RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE}:
        raise ValueError("Subject-position manifests support running or complete state.")
    prepared = plan.prepared
    arrays = [
        {
            "path": path,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(array).tobytes(order="C")
            ).hexdigest(),
        }
        for path, array in sorted(prepared.arrays.items())
    ]
    logical = canonical_observation_position_logical_metadata(
        _thaw_json(prepared.coordinate_record)
    )
    payload = {
        "namespace": OBSERVATION_POSITION_NAMESPACE,
        "row_axis": "observation_instance",
        "run_name": plan.run_name,
        "run_path": plan.run_path,
        "status": status,
        "stage_selector_eligible": False,
        "logical_metadata": logical,
        "estimator": {
            "record": _thaw_json(prepared.estimator_record),
            "sha256": prepared.estimator_sha256,
        },
        "anatomy": {
            "record": _thaw_json(prepared.anatomy_record),
            "sha256": prepared.anatomy_sha256,
        },
        "source": {
            "record": _thaw_json(prepared.source_record),
            "sha256": prepared.source_sha256,
        },
        "policy": {
            "record": _thaw_json(prepared.policy_record),
            "sha256": prepared.policy_sha256,
        },
        "software": {
            "record": _thaw_json(prepared.software_record),
            "sha256": prepared.software_sha256,
        },
        "coordinate": {
            "record": _thaw_json(prepared.coordinate_record),
            "sha256": prepared.coordinate_sha256,
            "descriptor_sha256": logical["coordinate_descriptor_sha256"],
        },
        "arrays": arrays,
        "decoded_content_sha256": canonical_observation_position_arrays_sha256(
            prepared.arrays
        ),
        "physical_storage_plan": plan.storage_receipt.as_manifest(),
        "publication": {
            "policy_id": SUBJECT_POSITION_PUBLISH_POLICY,
            "retry_policy": SUBJECT_POSITION_RETRY_POLICY,
            "publication_attempt_uuid": plan.publication_attempt_uuid,
            "selector_activation": "forbidden",
            "parent_selector_mutation": "forbidden",
        },
    }
    return {
        "schema_id": SUBJECT_POSITION_MANIFEST_SCHEMA_ID,
        "schema_version": SUBJECT_POSITION_MANIFEST_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def subject_position_manifest_digest(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_id", "schema_version", "payload", "payload_digest"
    }:
        raise ValueError("Subject-position manifest envelope is not exact.")
    if manifest["schema_id"] != SUBJECT_POSITION_MANIFEST_SCHEMA_ID:
        raise ValueError("Subject-position manifest schema ID mismatch.")
    if manifest["schema_version"] != SUBJECT_POSITION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Subject-position manifest schema version mismatch.")
    payload = manifest["payload"]
    if not isinstance(payload, Mapping) or manifest["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Subject-position manifest payload digest mismatch.")
    return str(manifest["payload_digest"])


def validate_subject_position_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_run_name: str | None = None,
    expected_status: str | None = None,
) -> dict[str, Any]:
    digest = subject_position_manifest_digest(manifest)
    payload = manifest["payload"]
    expected_fields = {
        "namespace", "row_axis", "run_name", "run_path", "status",
        "stage_selector_eligible", "logical_metadata", "estimator", "anatomy",
        "source", "policy", "software", "coordinate", "arrays",
        "decoded_content_sha256", "physical_storage_plan", "publication",
    }
    if set(payload) != expected_fields:
        raise ValueError("Subject-position manifest payload field set is not exact.")
    if payload["namespace"] != OBSERVATION_POSITION_NAMESPACE or payload["row_axis"] != "observation_instance":
        raise ValueError("Subject-position manifest namespace or row axis is invalid.")
    run_name = _safe_run_name(payload["run_name"])
    if expected_run_name is not None and run_name != expected_run_name:
        raise ValueError("Subject-position manifest run name differs from the plan.")
    status = payload["status"]
    if status not in {RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE}:
        raise ValueError("Subject-position manifest status is invalid.")
    if expected_status is not None and status != expected_status:
        raise ValueError("Subject-position manifest status differs from the required phase.")
    if payload["stage_selector_eligible"] is not False:
        raise ValueError("Subject-position publications must remain selector-ineligible.")
    for field in ("estimator", "anatomy", "source", "policy", "software"):
        value = payload[field]
        if not isinstance(value, Mapping) or set(value) != {"record", "sha256"}:
            raise ValueError(f"Subject-position {field} binding is not exact.")
        _bind_record(value["record"], value["sha256"], name=field)
    estimator = canonicalize_estimator_profile(payload["estimator"]["record"])
    if estimator_profile_digest(estimator) != payload["estimator"]["sha256"]:
        raise ValueError("Subject-position estimator digest is stale.")
    coordinate = payload["coordinate"]
    if not isinstance(coordinate, Mapping) or set(coordinate) != {"record", "sha256", "descriptor_sha256"}:
        raise ValueError("Subject-position coordinate binding is not exact.")
    coordinate_record, coordinate_digest = _bind_record(
        coordinate["record"], coordinate["sha256"], name="coordinate"
    )
    coordinate_metadata = canonical_observation_position_logical_metadata(
        coordinate_record
    )
    if coordinate["descriptor_sha256"] != coordinate_metadata["coordinate_descriptor_sha256"]:
        raise ValueError("Subject-position coordinate descriptor digest is stale.")
    logical = payload["logical_metadata"]
    if logical != canonical_observation_position_logical_metadata(coordinate_record):
        raise ValueError("Subject-position logical metadata is not canonical.")
    arrays = payload["arrays"]
    if not isinstance(arrays, list) or not arrays:
        raise ValueError("Subject-position manifest must declare arrays.")
    paths: list[str] = []
    for entry in arrays:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "dtype", "shape", "sha256"}:
            raise ValueError("Subject-position array manifest entry is not exact.")
        _safe_path = str(entry["path"])
        if _safe_path not in OBSERVATION_POSITION_ARRAYS:
            raise ValueError(f"Unknown subject-position manifest array {_safe_path!r}.")
        _require_digest(entry["sha256"], name=f"array {_safe_path}")
        if not isinstance(entry["dtype"], str) or not isinstance(entry["shape"], list):
            raise ValueError(f"Subject-position array declaration {_safe_path!r} is invalid.")
        paths.append(_safe_path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Subject-position array declarations must be sorted and unique.")
    if not set(OBSERVATION_POSITION_MANDATORY_ARRAYS).issubset(paths):
        raise ValueError("Subject-position manifest omits a mandatory array.")
    _require_digest(payload["decoded_content_sha256"], name="decoded_content_sha256")
    receipt = analysis_storage_plan_receipt_from_manifest(payload["physical_storage_plan"])
    receipt_paths = [entry.declaration.path for entry in receipt.entries]
    if receipt_paths != paths:
        raise ValueError("Subject-position manifest arrays differ from its storage plan.")
    for entry, planned in zip(arrays, receipt.entries, strict=True):
        if (
            entry["dtype"] != np.dtype(planned.facts.dtype).str
            or tuple(entry["shape"]) != planned.facts.shape
        ):
            raise ValueError(
                f"Subject-position array declaration {entry['path']!r} differs from its plan."
            )
    publication = payload["publication"]
    if not isinstance(publication, Mapping) or set(publication) != {
        "policy_id", "retry_policy", "publication_attempt_uuid",
        "selector_activation", "parent_selector_mutation",
    }:
        raise ValueError("Subject-position publication policy is not exact.")
    if publication["policy_id"] != SUBJECT_POSITION_PUBLISH_POLICY or publication["retry_policy"] != SUBJECT_POSITION_RETRY_POLICY:
        raise ValueError("Subject-position publication policy identity differs.")
    if publication["selector_activation"] != "forbidden" or publication["parent_selector_mutation"] != "forbidden":
        raise ValueError("Subject-position publication cannot activate or mutate selectors.")
    str(uuid.UUID(str(publication["publication_attempt_uuid"])))
    return {
        "valid": True,
        "manifest_sha256": digest,
        "payload": payload,
        "coordinate_record": coordinate_record,
        "coordinate_sha256": coordinate_digest,
        "storage_receipt": receipt,
    }


def _fill_value(path: str) -> Any:
    if path == "position_xy" or path.endswith("source_points_xy") or path.endswith("confidence"):
        return np.float32(0.0)
    if path == "valid" or path.endswith("source_points_valid"):
        return False
    if path.endswith("reason_codes") or path in {"failure_reason_codes"}:
        return np.uint16(0)
    if path in {"instance_key", "source_acquisition_frame_index", "source_row_index"}:
        return np.int64(0) if path != "instance_key" else np.uint64(0)
    raise KeyError(path)


def _write_arrays(run_group: Any, plan: SubjectPositionRunPlan) -> None:
    entries = {entry.declaration.path: entry for entry in plan.storage_receipt.entries}
    for path, values in sorted(plan.prepared.arrays.items()):
        parent_path, _, leaf = path.rpartition("/")
        parent = run_group.require_group(parent_path) if parent_path else run_group
        entry = entries[path]
        destination = create_array_from_plan(
            parent,
            name=leaf or path,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=_fill_value(path),
        )
        if values.size:
            destination[...] = values
        if path == "position_xy":
            coordinate = _thaw_json(plan.prepared.coordinate_record)
            destination.attrs.update(
                canonical_coordinate_descriptor_v2_attrs(
                    coordinate["coordinate_descriptor"]
                )
            )
            destination.attrs[SUBJECT_POSITION_COORDINATE_SURFACE_ATTR] = (
                json_attr_safe(coordinate["coordinate_surface_contract"])
            )


def _manifest_attrs(run_group: Any, manifest: Mapping[str, Any], plan: SubjectPositionRunPlan) -> None:
    run_group.attrs[SUBJECT_POSITION_MANIFEST_ATTR] = json_attr_safe(manifest)
    run_group.attrs[SUBJECT_POSITION_MANIFEST_DIGEST_ATTR] = subject_position_manifest_digest(manifest)
    run_group.attrs[SUBJECT_POSITION_STORAGE_PLAN_ATTR] = json_attr_safe(plan.storage_receipt.as_manifest())
    run_group.attrs[SUBJECT_POSITION_PUBLICATION_ATTEMPT_ATTR] = plan.publication_attempt_uuid
    run_group.attrs["stage_selector_eligible"] = False
    run_group.attrs["run_provenance"] = json_attr_safe(plan.run_provenance)


def _read_direct_declaration(path: Path) -> dict[str, Any]:
    metadata_path = path / "zarr.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing direct Zarr metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Direct Zarr metadata is not an object: {metadata_path}")
    return payload


def _validate_direct_array_metadata(run_path: Path, receipt: AnalysisStoragePlanReceipt) -> None:
    entries = {entry.declaration.path: entry for entry in receipt.entries}
    for path, entry in entries.items():
        declaration = _read_direct_declaration(run_path.joinpath(*path.split("/")))
        errors = validate_array_metadata_declaration_from_plan(
            declaration,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=_fill_value(path),
        )
        if errors:
            raise ValueError(f"Direct metadata failed for {path!r}: {errors!r}")


def _arrays_from_group(run_group: Any, paths: Sequence[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path in paths:
        node = _get_node(run_group, path)
        if not isinstance(node, zarr.Array):
            raise TypeError(f"Subject-position path is not an array: {path}")
        arrays[path] = np.asarray(node[:])
    return arrays


def _validate_run_group(
    run_group: Any,
    run_path: Path,
    *,
    expected_run_name: str,
    expected_status: str | None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = run_group.attrs.get(SUBJECT_POSITION_MANIFEST_ATTR)
    manifest_result = validate_subject_position_manifest(
        manifest,
        expected_run_name=expected_run_name,
        expected_status=expected_status,
    )
    if expected_manifest_sha256 is not None and manifest_result["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Subject-position manifest digest differs from the plan.")
    if run_group.attrs.get("stage_selector_eligible") is not False:
        raise ValueError("Subject-position run is selector eligible.")
    if run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != manifest_result["payload"]["status"]:
        raise ValueError("Subject-position completion status differs from its manifest.")
    receipt = manifest_result["storage_receipt"]
    arrays = _arrays_from_group(run_group, [entry.declaration.path for entry in receipt.entries])
    _validate_direct_array_metadata(run_path, receipt)
    position_node = _get_node(run_group, "position_xy")
    coordinate_record = manifest_result["coordinate_record"]
    descriptor = parse_canonical_coordinate_descriptor(
        coordinate_record["coordinate_descriptor"]
    )
    load_canonical_coordinate_descriptor_attrs(
        position_node.attrs,
        row_identity_contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=arrays["instance_key"],
        ),
        expected_row_identity_record_ref=descriptor.row_identity.record_ref,
        owner_shape=arrays["position_xy"].shape,
    )
    if position_node.attrs.get(SUBJECT_POSITION_COORDINATE_SURFACE_ATTR) != (
        coordinate_record["coordinate_surface_contract"]
    ):
        raise ValueError("position_xy coordinate-surface contract is stale.")
    validate_observation_position_arrays(
        arrays,
        coordinate_metadata=manifest_result["coordinate_record"],
        manifest_metadata=manifest_result["payload"]["logical_metadata"],
    )
    if canonical_observation_position_arrays_sha256(arrays) != manifest_result["payload"]["decoded_content_sha256"]:
        raise ValueError("Subject-position decoded content digest differs from its manifest.")
    if run_group.attrs.get(SUBJECT_POSITION_MANIFEST_DIGEST_ATTR) != manifest_result["manifest_sha256"]:
        raise ValueError("Subject-position manifest attribute digest is stale.")
    return {
        "valid": True,
        "run_path": str(run_path),
        "status": manifest_result["payload"]["status"],
        "row_count": int(arrays["position_xy"].shape[0]),
        "manifest_sha256": manifest_result["manifest_sha256"],
        "decoded_content_sha256": manifest_result["payload"]["decoded_content_sha256"],
        "storage_plan_digest": receipt.as_manifest()["payload_digest"],
    }


def validate_subject_position_run(
    analysis_zarr: str | Path,
    run_path: str,
    *,
    use_consolidated: bool = False,
    expected_status: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one run through direct or explicitly consolidated metadata."""

    archive = Path(analysis_zarr).expanduser().resolve()
    root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
    run_group = _get_node(root, run_path)
    return _validate_run_group(
        run_group,
        archive.joinpath(*run_path.split("/")),
        expected_run_name=run_path.rstrip("/").split("/")[-1],
        expected_status=expected_status,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def _materialize_local(plan: SubjectPositionRunPlan) -> dict[str, Any]:
    if plan.local_zarr.exists():
        raise FileExistsError(f"Refusing existing local attempt: {plan.local_zarr}")
    plan.local_zarr.parent.mkdir(parents=True, exist_ok=True)
    local_root = zarr.open_group(
        str(plan.local_zarr),
        mode="w-",
        zarr_format=3,
        use_consolidated=False,
    )
    parent = require_runs_parent(
        local_root,
        plan.parent_path,
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    run_group = parent.create_group(plan.run_name)
    mark_run_started(run_group, run_name=plan.run_name, stage="subject_position")
    _write_arrays(run_group, plan)
    running_manifest = build_subject_position_manifest(plan, status=RUN_STATUS_RUNNING)
    _manifest_attrs(run_group, running_manifest, plan)
    _validate_run_group(
        run_group,
        plan.local_run_path,
        expected_run_name=plan.run_name,
        expected_status=RUN_STATUS_RUNNING,
    )
    mark_run_complete(
        run_group,
        parent_group=None,
        run_name=plan.run_name,
        run_provenance=plan.run_provenance,
    )
    complete_manifest = build_subject_position_manifest(plan, status=RUN_STATUS_COMPLETE)
    _manifest_attrs(run_group, complete_manifest, plan)
    _validate_run_group(
        run_group,
        plan.local_run_path,
        expected_run_name=plan.run_name,
        expected_status=RUN_STATUS_COMPLETE,
        expected_manifest_sha256=subject_position_manifest_digest(complete_manifest),
    )
    return {
        "local_zarr": str(plan.local_zarr),
        "local_run_path": str(plan.local_run_path),
        "manifest_sha256": subject_position_manifest_digest(complete_manifest),
    }


def publish_subject_position_run(
    plan: SubjectPositionRunPlan,
    *,
    copy_backend: str = "python",
    keep_scratch: bool = True,
) -> dict[str, Any]:
    """Materialize and atomically publish one non-promoting position attempt."""

    local = _materialize_local(plan)
    expected_manifest = plan.final_manifest_sha256
    acceptance: dict[str, Any] = {}

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_run_group(
            open_zarr_root(path, mode="r", use_consolidated=False),
            path,
            expected_run_name=plan.run_name,
            expected_status=RUN_STATUS_COMPLETE,
            expected_manifest_sha256=expected_manifest,
        )

    def prepare(root: Any) -> tuple[Any]:
        analysis = root.require_group("analysis")
        return (
            require_runs_parent(
                analysis,
                "subject_position_runs/observation",
                completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
            ),
        )

    def complete(_root: Any, _parent: Any, run_group: Any) -> None:
        mark_run_complete(
            run_group,
            parent_group=None,
            run_name=plan.run_name,
            run_provenance=run_group.attrs.get("run_provenance"),
        )
        run_group.attrs["stage_selector_eligible"] = False

    def verify(root: Any) -> None:
        parent = _get_node(root, plan.parent_path)
        if _selector_attrs_from_parent(parent) != dict(plan.parent_selector_attrs):
            raise RuntimeError("Subject-position publication changed parent selectors.")
        run_group = parent[plan.run_name]
        _validate_run_group(
            run_group,
            plan.target_run_path,
            expected_run_name=plan.run_name,
            expected_status=RUN_STATUS_COMPLETE,
            expected_manifest_sha256=expected_manifest,
        )

    def finalize(_root: Any, _parent: Any, _run_group: Any) -> None:
        validate_subject_position_run(
            plan.source_zarr,
            plan.run_path,
            use_consolidated=False,
            expected_status=RUN_STATUS_COMPLETE,
            expected_manifest_sha256=expected_manifest,
        )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        metadata_receipt = validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=plan.run_path,
        )
        consolidated = validate_subject_position_run(
            plan.source_zarr,
            plan.run_path,
            use_consolidated=True,
            expected_status=RUN_STATUS_COMPLETE,
            expected_manifest_sha256=expected_manifest,
        )
        acceptance.update(
            direct_consolidated=metadata_receipt.to_json(),
            consolidated_validation=consolidated,
        )

    def repair_failed(_target: Path) -> None:
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="subject-position",
            publish_schema_id=SUBJECT_POSITION_PUBLISH_SCHEMA_ID,
            policy=SUBJECT_POSITION_PUBLISH_POLICY,
            rollback_policy="retain_failed_tombstone_leave_parent_selectors_untouched",
            content_checksum=True,
            publication_attempt_uuid=plan.publication_attempt_uuid,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=finalize,
        repair_failed_publication_visibility=repair_failed,
        accept_persisted_activation_on_callback_error=False,
        payload_metadata={
            "run_manifest_sha256": expected_manifest,
            "source_record_sha256": plan.prepared.source_sha256,
            "estimator_record_sha256": plan.prepared.estimator_sha256,
            "storage_plan_digest": plan.storage_receipt.as_manifest()["payload_digest"],
            "selector_ineligible": True,
        },
    )
    result = {
        "plan": plan.as_dict(),
        "local": local,
        "publication": publication,
        "acceptance": acceptance,
    }
    if not keep_scratch and plan.local_zarr.exists():
        shutil.rmtree(plan.local_zarr)
    return result


__all__ = [
    "SUBJECT_POSITION_MANIFEST_ATTR",
    "SUBJECT_POSITION_MANIFEST_SCHEMA_ID",
    "SUBJECT_POSITION_MANIFEST_SCHEMA_VERSION",
    "SUBJECT_POSITION_PARENT_PATH",
    "SUBJECT_POSITION_PUBLISH_SCHEMA_ID",
    "SUBJECT_POSITION_PUBLISH_SCHEMA_VERSION",
    "SubjectPositionPreparedInput",
    "SubjectPositionPreparation",
    "SubjectPositionRunPlan",
    "build_subject_position_manifest",
    "plan_subject_position_run",
    "publish_subject_position_run",
    "subject_position_manifest_digest",
    "validate_subject_position_manifest",
    "validate_subject_position_run",
]
