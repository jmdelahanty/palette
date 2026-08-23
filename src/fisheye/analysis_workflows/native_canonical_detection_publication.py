"""Atomically publish one native canonical-detection candidate.

The candidate is built and completely validated on node-local storage.  This
module is the only shared-archive mutation boundary: it revalidates the native
manifest and its recording-bound authority records, copies the fresh run group
through the common atomic publisher, reconsolidates the archive, and validates
the public copy again.  It deliberately leaves selectors and the registry
unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

import zarr

from fisheye.shared.atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
    ATOMIC_PUBLICATION_TOMBSTONE_ATTR,
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.detection.native_canonical_candidate import (
    NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_ID,
    NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_VERSION,
    NativeCanonicalDetectionCandidate,
    validate_native_canonical_detection_candidate,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR,
    CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3,
    CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR,
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION,
    NATIVE_CANONICAL_DETECTION_PUBLICATION_SCHEMA_ID,
    NATIVE_CANONICAL_DETECTION_PUBLICATION_SCHEMA_VERSION,
    validate_canonical_detection_publication,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    canonical_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
)
from fisheye.shared.zarr.detection_storage import (
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
    reconsolidate_zarr_metadata,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    require_runs_parent,
)


NATIVE_CANONICAL_DETECTION_PUBLICATION_POLICY = (
    "node_local_native_manifest_v2_then_atomic_selector_ineligible_import_v1"
)
NATIVE_CANONICAL_DETECTION_ACTIVE_PUBLICATION_POLICY = (
    "node_local_coordinate_manifest_v3_then_atomic_activation_v1"
)
NATIVE_CANONICAL_DETECTION_ROLLBACK_POLICY = (
    "retain_failed_owner_bound_selector_ineligible_child_v1"
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)
_ACTIVATION_PARENT_ATTRS = (
    *_SELECTOR_ATTRS,
    CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR,
    CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR,
)
NATIVE_DETECTION_ACTIVATION_CONSOLIDATION_POLICY = (
    "canonical_detection_v3_selector_activation_direct_consolidated_verified_v1"
)
NATIVE_DETECTION_FAILED_ACTIVATION_REPAIR_POLICY = (
    "canonical_detection_v3_failed_activation_rollback_verified_v1"
)


@dataclass
class _NativeDetectionActivation:
    archive: Path
    run_id: str
    manifest: Mapping[str, Any]
    plans: Any
    snapshot: dict[str, tuple[bool, Any]] | None = None
    attempted: dict[str, Any] | None = None
    visibility_report: dict[str, Any] | None = None

    def _validate_visibility(self) -> dict[str, Any]:
        direct_root = zarr.open_group(
            str(self.archive), mode="r", zarr_format=3, use_consolidated=False
        )
        consolidated_root = zarr.open_group(
            str(self.archive), mode="r", zarr_format=3, use_consolidated=True
        )
        for label, root in (
            ("direct", direct_root),
            ("consolidated", consolidated_root),
        ):
            family = root["detect_runs"]
            run = family[self.run_id]
            if (
                family.attrs.get("latest") != self.run_id
                or family.attrs.get("latest_complete") != self.run_id
                or family.attrs.get(CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR)
                != CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
                or family.attrs.get(CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR)
                != self.manifest.get("payload_digest")
                or run.attrs.get("palette_run_completion_status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not True
                or dict(run.attrs.get("run_manifest") or {})
                != dict(self.manifest)
            ):
                raise RuntimeError(
                    f"{label} canonical detection is not selected, complete, "
                    "and selector eligible."
                )
        direct, consolidated = canonical_detection_metadata_declaration_maps(
            self.archive,
            run_id=self.run_id,
            plans=self.plans,
        )
        arrays = {
            path: direct_root[f"detect_runs/{self.run_id}/{path}"]
            for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        }
        errors = validate_canonical_detection_publication(
            self.manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
        )
        if errors:
            raise RuntimeError(
                "Activated canonical detection validation failed: "
                + "; ".join(errors)
            )
        return {
            "policy": NATIVE_DETECTION_ACTIVATION_CONSOLIDATION_POLICY,
            "run_id": self.run_id,
            "selectors": {
                "latest": self.run_id,
                "latest_complete": self.run_id,
            },
            "manifest_digest": self.manifest.get("payload_digest"),
        }

    def activate(self, _root: Any, family: Any, run: Any) -> None:
        self.snapshot = {
            name: (name in family.attrs, copy.deepcopy(family.attrs.get(name)))
            for name in _ACTIVATION_PARENT_ATTRS
        }
        self.attempted = {
            "latest": self.run_id,
            "latest_complete": self.run_id,
            CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR: (
                CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
            ),
            CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR: self.manifest.get(
                "payload_digest"
            ),
        }
        family.attrs["latest"] = self.run_id
        family.attrs["latest_complete"] = self.run_id
        family.attrs[CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR] = (
            CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
        )
        family.attrs[CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR] = (
            self.manifest.get("payload_digest")
        )
        if family.attrs.get("latest_pending") == self.run_id:
            del family.attrs["latest_pending"]
            self.attempted["latest_pending"] = None
        run.attrs["stage_selector_eligible"] = True
        consolidation = reconsolidate_zarr_metadata(
            self.archive,
            policy=NATIVE_DETECTION_ACTIVATION_CONSOLIDATION_POLICY,
            fail_on_error=True,
        )
        self.visibility_report = {
            **self._validate_visibility(),
            "consolidation": consolidation,
        }

    def rollback(self) -> None:
        if self.snapshot is None or self.attempted is None:
            return
        root = open_zarr_root(self.archive, mode="a")
        family = root["detect_runs"]
        for name, (present, value) in self.snapshot.items():
            attempted = self.attempted.get(name, object())
            current_present = name in family.attrs
            current_value = family.attrs.get(name)
            owned = (
                (attempted is None and not current_present)
                or (
                    attempted is not None
                    and current_present
                    and current_value == attempted
                )
            )
            if not owned:
                continue
            if present:
                family.attrs[name] = copy.deepcopy(value)
            elif name in family.attrs:
                del family.attrs[name]
        if self.run_id in family:
            family[self.run_id].attrs["stage_selector_eligible"] = False

    def repair_failed_visibility(self, _target_path: Path) -> None:
        reconsolidate_zarr_metadata(
            self.archive,
            policy=NATIVE_DETECTION_FAILED_ACTIVATION_REPAIR_POLICY,
            fail_on_error=True,
        )


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _require_run_id(value: str) -> str:
    normalized = str(value).strip()
    if not _RUN_ID_RE.fullmatch(normalized):
        raise ValueError("run_id must be one safe nonempty group name.")
    return normalized


def _dimensions_and_plans(manifest: Mapping[str, Any]):
    payload = manifest.get("payload")
    logical = payload.get("logical_schema") if isinstance(payload, Mapping) else None
    raw_dimensions = logical.get("dimensions") if isinstance(logical, Mapping) else None
    if not isinstance(raw_dimensions, Mapping):
        raise ValueError("Native candidate manifest has no logical dimensions.")
    dimensions = CanonicalDetectionDimensions(
        n_frames=raw_dimensions.get("n_frames"),
        n_instances=raw_dimensions.get("n_instances"),
        source_width=raw_dimensions.get("source_width"),
        source_height=raw_dimensions.get("source_height"),
    )
    if raw_dimensions.get("n_frame_boundaries") != dimensions.n_frames + 1:
        raise ValueError("Native candidate n_frame_boundaries is inconsistent.")
    raw_storage = payload.get("storage_plan") if isinstance(payload, Mapping) else None
    raw_profile = (
        raw_storage.get("storage_profile")
        if isinstance(raw_storage, Mapping)
        else None
    )
    if not isinstance(raw_profile, Mapping):
        raise ValueError("Native candidate manifest has no storage profile.")
    profile = storage_profile_from_manifest(raw_profile)
    plans = plan_canonical_detection_storage(dimensions, profile=profile)
    if plans.as_manifest() != dict(raw_storage):
        raise ValueError("Native candidate storage plan differs from the frozen planner.")
    return dimensions, plans


def load_native_canonical_detection_candidate(
    candidate_zarr: Path,
    *,
    run_id: str,
    expected_manifest_schema_version: int = (
        CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION
    ),
) -> NativeCanonicalDetectionCandidate:
    """Reopen a local candidate without trusting an in-memory writer result."""

    output_path = candidate_zarr.expanduser().resolve()
    normalized_run_id = _require_run_id(run_id)
    if expected_manifest_schema_version not in {
        CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION,
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError(
            "expected_manifest_schema_version must select canonical v2 or v3."
        )
    run_path = output_path / "detect_runs" / normalized_run_id
    if not run_path.is_dir():
        raise FileNotFoundError(f"Native candidate run not found: {run_path}")
    run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
    manifest = dict(run.attrs.get("run_manifest") or {})
    if manifest.get("schema_version") != expected_manifest_schema_version:
        version_label = (
            "v2"
            if expected_manifest_schema_version
            == CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION
            else "v3"
        )
        raise ValueError(
            f"Native candidate must carry run-manifest {version_label}; its "
            "persisted version differs from the explicit caller contract."
        )
    _, plans = _dimensions_and_plans(manifest)
    arrays = {
        path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
    }
    receipt_path = output_path / "native_detection_candidate_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Native candidate receipt not found: {receipt_path}")
    receipt = _read_strict_json(receipt_path)
    expected_receipt = {
        "schema_id": NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_ID,
        "schema_version": NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_VERSION,
        "status": "complete",
        "run_id": normalized_run_id,
        "native_run_manifest_schema_version": (
            expected_manifest_schema_version
        ),
        "logical_schema_version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
        "run_manifest_digest": manifest.get("payload_digest"),
        "stage_selector_eligible": False,
        "registry_registered": False,
    }
    receipt_errors = [
        f"{name}={receipt.get(name)!r}, expected {expected!r}"
        for name, expected in expected_receipt.items()
        if receipt.get(name) != expected
    ]
    if receipt_errors:
        raise ValueError(
            "Native candidate receipt differs from the complete candidate: "
            + "; ".join(receipt_errors)
        )
    candidate = NativeCanonicalDetectionCandidate(
        output_path=output_path,
        run_id=normalized_run_id,
        plans=plans,
        manifest=manifest,
        arrays=arrays,
        receipt=receipt,
    )
    errors = validate_native_canonical_detection_candidate(candidate)
    if errors:
        raise ValueError("Invalid native candidate: " + "; ".join(errors))
    return candidate


def _resolve_archive_node(root: Any, node_path: str) -> Any:
    normalized = str(node_path).strip()
    if not normalized.startswith("/") or normalized in {"", "/"}:
        raise ValueError("Authority record_ref requires an absolute non-root path.")
    parts = normalized[1:].split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Authority record_ref path is noncanonical.")
    node = root
    for part in parts:
        node = node[part]
    return node


def _verify_authority_record(root: Any, pointer: Mapping[str, Any], *, name: str) -> None:
    if set(pointer) != {"record_ref", "record_sha256"}:
        raise ValueError(f"{name} is not an exact authority pointer.")
    record_ref = str(pointer["record_ref"])
    if record_ref.count("@") != 1:
        raise ValueError(f"{name}.record_ref must select exactly one attribute.")
    node_path, attr_name = record_ref.split("@", 1)
    if not attr_name or "/" in attr_name:
        raise ValueError(f"{name}.record_ref has an invalid attribute selector.")
    node = _resolve_archive_node(root, node_path)
    if attr_name not in node.attrs:
        raise ValueError(f"{name} record is missing from the target archive.")
    record = node.attrs[attr_name]
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} authority record must be an object.")
    observed = canonical_json_sha256(record)
    expected = str(pointer["record_sha256"])
    if observed != expected:
        raise ValueError(f"{name} record content digest differs from the manifest.")
    persisted_digest = node.attrs.get(f"{attr_name}_sha256")
    if persisted_digest is not None and str(persisted_digest) != expected:
        raise ValueError(f"{name} persisted digest attribute disagrees with the record.")


def _verify_native_source_authorities(
    root: Any,
    *,
    manifest: Mapping[str, Any],
    recording_identity: str,
) -> None:
    payload = manifest.get("payload")
    evidence = payload.get("source_evidence") if isinstance(payload, Mapping) else None
    if not isinstance(evidence, Mapping):
        raise ValueError("Native manifest source evidence is missing.")
    if evidence.get("recording_identity") != recording_identity:
        raise ValueError("Native source recording identity differs from the archive.")
    _verify_authority_record(
        root,
        evidence.get("source_frame_authority"),
        name="source_frame_authority",
    )
    _verify_authority_record(
        root,
        evidence.get("source_pixel_authority"),
        name="source_pixel_authority",
    )


def _direct_declarations(run_path: Path, *, plans: Any) -> dict[str, dict[str, Any]]:
    declarations: dict[str, dict[str, Any]] = {}
    for relative in ("", "instances", *(entry.rule.path for entry in plans.entries)):
        path = run_path if not relative else run_path / relative
        declarations[relative] = _read_strict_json(path / "zarr.json")
    return declarations


def _validate_standalone_run(
    run_path: Path,
    *,
    manifest: Mapping[str, Any],
    plans: Any,
) -> dict[str, object]:
    try:
        run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
        arrays = {
            path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        }
        direct = _direct_declarations(run_path, plans=plans)
        errors = list(
            validate_canonical_detection_publication(
                manifest,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=direct,
                arrays=arrays,
            )
        )
        if dict(run.attrs.get("run_manifest") or {}) != dict(manifest):
            errors.append("persisted run manifest differs from the local candidate")
        if run.attrs.get("status") != "complete":
            errors.append("published run status is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors.append("published run became selector eligible")
        return {"valid": not errors, "errors": list(dict.fromkeys(errors))}
    except Exception as exc:
        return {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}


def _prepare_parent(root: Any) -> tuple[Any, ...]:
    return (
        require_runs_parent(
            root, "detect_runs", completion_epoch=COMPLETION_EPOCH_STRICT
        ),
    )


def _require_unselected(root: Any, *, run_id: str) -> None:
    family = root["detect_runs"]
    collisions = [name for name in _SELECTOR_ATTRS if family.attrs.get(name) == run_id]
    if collisions:
        raise RuntimeError(
            f"Selector-ineligible native run {run_id!r} is selected by {collisions!r}."
        )
    run = family[run_id]
    if run.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError(f"Native run {run_id!r} became selector eligible.")


def tombstone_native_canonical_detection_postcopy_failure(
    *,
    analysis_zarr: Path,
    run_id: str,
    expected_publication_owner: str,
    failure: BaseException | str,
) -> dict[str, object]:
    """Fail-close one exact owned run after a terminal post-copy failure."""

    archive = analysis_zarr.expanduser().resolve()
    normalized_run_id = _require_run_id(run_id)
    owner = str(expected_publication_owner).strip()
    if not owner:
        raise ValueError("expected_publication_owner cannot be empty.")
    root = open_zarr_root(archive, mode="a")
    _require_unselected(root, run_id=normalized_run_id)
    run = root[f"detect_runs/{normalized_run_id}"]
    observed_owner = run.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
    if observed_owner != owner:
        raise RuntimeError(
            "Refusing to tombstone a native run with a different publication owner."
        )
    tombstone: dict[str, object] = {
        "schema_id": "palette.native_canonical_detection.postcopy_failure",
        "schema_version": 1,
        "failed_at_utc": utc_now(),
        "run_id": normalized_run_id,
        "publication_owner_attr": ATOMIC_PUBLICATION_OWNER_ATTR,
        "publication_owner_uuid": owner,
        "failure": str(failure),
        "selector_eligible": False,
        "registry_updated": False,
    }
    attrs = dict(run.attrs)
    attrs.pop("palette_run_completed_at_utc", None)
    attrs.update(
        {
            "status": "failed",
            "palette_run_completion_status": "failed",
            "stage_selector_eligible": False,
            "native_canonical_detection_postcopy_failure": str(failure),
            ATOMIC_PUBLICATION_TOMBSTONE_ATTR: json_attr_safe(tombstone),
        }
    )
    run.attrs.put(attrs)

    check_root = open_zarr_root(archive, mode="r")
    _require_unselected(check_root, run_id=normalized_run_id)
    check = check_root[f"detect_runs/{normalized_run_id}"]
    if (
        check.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR) != owner
        or check.attrs.get("status") != "failed"
        or check.attrs.get("palette_run_completion_status") != "failed"
        or check.attrs.get("stage_selector_eligible") is not False
        or check.attrs.get(ATOMIC_PUBLICATION_TOMBSTONE_ATTR)
        != json_attr_safe(tombstone)
    ):
        raise RuntimeError("Native post-copy failure tombstone did not persist.")
    return json_attr_safe(tombstone)


def publish_native_canonical_detection_candidate(
    *,
    analysis_zarr: Path,
    candidate_zarr: Path,
    run_id: str,
    recording_identity: str,
    expected_manifest_schema_version: int = (
        CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION
    ),
    activate: bool = False,
    copy_backend: str = "python",
    result_json: Path | None = None,
) -> dict[str, object]:
    """Atomically publish one canonical raw run and optionally activate it."""

    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be 'python' or 'rsync'.")
    if type(activate) is not bool:
        raise TypeError("activate must be an exact bool.")
    identity = str(recording_identity).strip()
    if not identity:
        raise ValueError("recording_identity cannot be empty.")
    candidate = load_native_canonical_detection_candidate(
        candidate_zarr,
        run_id=run_id,
        expected_manifest_schema_version=expected_manifest_schema_version,
    )
    payload = candidate.manifest.get("payload")
    publication_contract = (
        payload.get("publication") if isinstance(payload, Mapping) else None
    )
    if not isinstance(publication_contract, Mapping):
        raise ValueError("Native candidate has no publication contract.")
    if publication_contract.get("stage_selector_eligible") is not activate:
        raise ValueError(
            "Candidate manifest selector eligibility differs from the requested "
            "publication mode."
        )
    target = archive / "detect_runs" / candidate.run_id
    if target.exists():
        raise FileExistsError(f"Immutable native detection target exists: {target}")

    root = open_zarr_root(archive, mode="r")
    if str(root.attrs.get("recording_id") or "").strip() != identity:
        raise ValueError("Requested recording identity differs from the archive.")
    _verify_native_source_authorities(
        root,
        manifest=candidate.manifest,
        recording_identity=identity,
    )

    local_run = candidate.output_path / "detect_runs" / candidate.run_id
    activation = (
        _NativeDetectionActivation(
            archive=archive,
            run_id=candidate.run_id,
            manifest=candidate.manifest,
            plans=candidate.plans,
        )
        if activate
        else None
    )

    def validator(path: Path) -> Mapping[str, Any]:
        return _validate_standalone_run(
            path,
            manifest=candidate.manifest,
            plans=candidate.plans,
        )

    def complete_run(_root: Any, _parent: Any, run: Any) -> None:
        if (
            run.attrs.get("status") != "complete"
            or run.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError("Imported native run is not complete and ineligible.")

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=archive,
            local_run_path=local_run,
            target_run_path=target,
            run_name=candidate.run_id,
            lock_suffix="native_canonical_detection_publication",
            publish_schema_id=NATIVE_CANONICAL_DETECTION_PUBLICATION_SCHEMA_ID,
            policy=(
                NATIVE_CANONICAL_DETECTION_ACTIVE_PUBLICATION_POLICY
                if activate
                else NATIVE_CANONICAL_DETECTION_PUBLICATION_POLICY
            ),
            rollback_policy=NATIVE_CANONICAL_DETECTION_ROLLBACK_POLICY,
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validator,
        prepare_parents=_prepare_parent,
        complete_run=complete_run,
        verify_pointers=lambda current: _require_unselected(
            current,
            run_id=candidate.run_id,
        ),
        activate_run=(activation.activate if activation is not None else None),
        rollback_activation=(activation.rollback if activation is not None else None),
        repair_failed_publication_visibility=(
            activation.repair_failed_visibility
            if activation is not None
            else None
        ),
        accept_persisted_activation_on_callback_error=False,
        payload_metadata={
            "snapshot_role": (
                "canonical_raw_detection_coordinate_v3"
                if expected_manifest_schema_version
                == CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
                else "canonical_raw_detection_native_v2"
            ),
            "logical_schema_version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
            "selector_activation": "atomic" if activate else "deferred",
        },
    )
    try:
        if activate and (
            activation is None or activation.visibility_report is None
        ):
            raise RuntimeError(
                "Canonical detection activation completed without visibility proof."
            )
        consolidation = consolidate_metadata_capture_expected_warnings(archive)
        root = open_zarr_root(archive, mode="r")
        _verify_native_source_authorities(
            root,
            manifest=candidate.manifest,
            recording_identity=identity,
        )
        run = root[f"detect_runs/{candidate.run_id}"]
        direct, consolidated = canonical_detection_metadata_declaration_maps(
            archive,
            run_id=candidate.run_id,
            plans=candidate.plans,
        )
        arrays = {
            path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        }
        errors = validate_canonical_detection_publication(
            candidate.manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
        )
        if activate:
            family = root["detect_runs"]
            if (
                family.attrs.get("latest") != candidate.run_id
                or family.attrs.get("latest_complete") != candidate.run_id
                or family.attrs.get(CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR)
                != CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
                or family.attrs.get(CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR)
                != candidate.manifest.get("payload_digest")
                or run.attrs.get("stage_selector_eligible") is not True
            ):
                raise RuntimeError(
                    "Canonical detection activation is not visible after copy."
                )
        else:
            _require_unselected(root, run_id=candidate.run_id)
        if errors:
            raise RuntimeError(
                "Published native canonical detection validation failed: "
                + "; ".join(errors)
            )
    except BaseException as exc:
        try:
            if activation is not None:
                activation.rollback()
                activation.repair_failed_visibility(target)
            tombstone_native_canonical_detection_postcopy_failure(
                analysis_zarr=archive,
                run_id=candidate.run_id,
                expected_publication_owner=str(
                    publication.get("publication_owner_uuid") or ""
                ),
                failure=exc,
            )
        except BaseException as tombstone_exc:
            raise RuntimeError(
                "Native post-copy validation failed and its owned child could "
                f"not be tombstoned: {tombstone_exc}"
            ) from exc
        raise

    result: dict[str, object] = {
        "schema_id": NATIVE_CANONICAL_DETECTION_PUBLICATION_SCHEMA_ID,
        "schema_version": NATIVE_CANONICAL_DETECTION_PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "published_at_utc": utc_now(),
        "analysis_zarr": str(archive),
        "recording_identity": identity,
        "run_id": candidate.run_id,
        "group_path": f"detect_runs/{candidate.run_id}",
        "native_run_manifest_schema_version": expected_manifest_schema_version,
        "logical_schema_version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
        "run_manifest_digest": candidate.manifest["payload_digest"],
        "storage_profile_id": candidate.plans.profile.profile_id,
        "candidate_receipt": dict(candidate.receipt),
        "publication": publication,
        "consolidation": consolidation,
        "source_authorities_revalidated_after_copy": True,
        "selector_eligible": activate,
        "selector_activation": "complete" if activate else "deferred",
        "activation_visibility": (
            activation.visibility_report if activation is not None else None
        ),
        "registry_updated": False,
    }
    safe_result = json_attr_safe(result)
    if result_json is not None:
        write_json_atomic(result_json.expanduser().resolve(), safe_result)
    return safe_result


__all__ = [
    "NATIVE_CANONICAL_DETECTION_PUBLICATION_SCHEMA_ID",
    "NATIVE_CANONICAL_DETECTION_PUBLICATION_SCHEMA_VERSION",
    "load_native_canonical_detection_candidate",
    "publish_native_canonical_detection_candidate",
    "tombstone_native_canonical_detection_postcopy_failure",
]
