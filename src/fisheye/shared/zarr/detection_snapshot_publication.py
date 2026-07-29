"""Publish selector-ineligible canonical/refined detection snapshots.

This is the production placement boundary for the frozen detection snapshot
contracts.  It converts complete full-acquisition compatibility runs on
node-local scratch, validates both immutable snapshots, atomically imports the
two run groups into a recording archive, reconsolidates archive metadata, and
validates the published result again.  It deliberately does not update a
selector or registry.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping
import uuid

import numpy as np
import zarr

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.canonical_detection_manifest import (
    validate_canonical_detection_publication,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    CanonicalDetectionShadowPublication,
    publish_legacy_canonical_detection_shadow,
    validate_canonical_detection_shadow_publication,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
    validate_refined_detection_publication,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
)
from fisheye.shared.zarr.refined_detection_shadow import (
    publish_refined_detection_shadow,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    refined_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.refined_detection_storage import (
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_refined_detection_transition,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root


DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_ID = (
    "palette.detection_snapshot.production_publication"
)
DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_VERSION = 1
DETECTION_SNAPSHOT_PUBLICATION_POLICY = (
    "node_local_v1_materialization_then_atomic_selector_ineligible_import_v1"
)
DETECTION_SNAPSHOT_ROLLBACK_POLICY = (
    "retain_failed_owner_bound_selector_ineligible_child_v1"
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


def _require_run_id(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not _RUN_ID_RE.fullmatch(normalized):
        raise ValueError(f"{label} must be one safe nonempty run id.")
    return normalized


def _require_relative_group_path(value: str, *, label: str) -> str:
    path = Path(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a nonescaping archive-relative group path.")
    return path.as_posix()


def _require_node_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Snapshot scratch root does not exist: {resolved}")
    if resolved in {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/scratch").resolve(),
    } or str(resolved).startswith(("/groups/", "/nrs/")):
        raise ValueError(
            "Detection snapshot scratch must be a bounded node-local path."
        )
    return resolved


def _direct_declarations(
    run_path: Path,
    *,
    relative_paths: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {token}")

    declarations: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = run_path if not relative else run_path / relative
        with (metadata_path / "zarr.json").open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                parse_constant=reject_nonfinite,
            )
        if not isinstance(value, dict):
            raise ValueError(f"Expected object metadata at {metadata_path}.")
        declarations[relative] = value
    return declarations


def _canonical_arrays(run: Any) -> dict[str, Any]:
    return {path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths}


def _refined_arrays(run: Any, *, dimensions: Any) -> dict[str, Any]:
    return {
        path: run[path]
        for path in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)
    }


def _validate_canonical_run_path(
    run_path: Path,
    *,
    publication: CanonicalDetectionShadowPublication,
) -> dict[str, object]:
    try:
        run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
        arrays = _canonical_arrays(run)
        relative_paths = (
            "",
            "instances",
            *(entry.rule.path for entry in publication.plans.entries),
        )
        direct = _direct_declarations(run_path, relative_paths=relative_paths)
        errors = validate_canonical_detection_publication(
            dict(run.attrs["run_manifest"]),
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=direct,
            arrays=arrays,
        )
        if run.attrs.get("status") != "complete":
            errors = (*errors, "snapshot status is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors = (*errors, "snapshot is not selector-ineligible")
        return {"valid": not errors, "errors": list(errors)}
    except Exception as exc:
        return {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}


def _validate_refined_run_path(
    run_path: Path,
    *,
    transition: Any,
    plans: Any,
) -> dict[str, object]:
    try:
        run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
        arrays = _refined_arrays(run, dimensions=transition.dimensions)
        relative_paths = (
            "",
            "instances",
            "source_detections",
            *(entry.rule.path for entry in plans.entries),
        )
        direct = _direct_declarations(run_path, relative_paths=relative_paths)
        errors = validate_refined_detection_publication(
            dict(run.attrs["run_manifest"]),
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=direct,
            arrays=arrays,
        )
        if run.attrs.get("status") != "complete":
            errors = (*errors, "snapshot status is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors = (*errors, "snapshot is not selector-ineligible")
        return {"valid": not errors, "errors": list(errors)}
    except Exception as exc:
        return {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}


def _mark_local_candidate(run: Any, *, source_group_path: str) -> None:
    attrs = dict(run.attrs)
    attrs.pop("shadow_only", None)
    attrs.update(
        {
            "immutable_snapshot": True,
            "production_candidate": True,
            "stage_selector_eligible": False,
            "production_selector_activation": "deferred",
            "compatibility_source_group_path": source_group_path,
        }
    )
    run.attrs.put(attrs)
    for table_name in ("instances", "source_detections"):
        table = run.get(table_name)
        if not isinstance(table, zarr.Group):
            continue
        for array in table.arrays():
            node = array[1]
            node_attrs = dict(node.attrs)
            node_attrs.pop("shadow_only", None)
            node_attrs["selector_eligible"] = False
            node.attrs.put(node_attrs)


def _prepare_parent(root: Any, family_name: str) -> tuple[Any, ...]:
    return (root.require_group(family_name),)


def _require_unselected(root: Any, *, family_name: str, run_id: str) -> None:
    family = root[family_name]
    collisions = [name for name in _SELECTOR_ATTRS if family.attrs.get(name) == run_id]
    if collisions:
        raise RuntimeError(
            f"Selector-ineligible snapshot {run_id!r} is selected by {collisions!r}."
        )
    run = family[run_id]
    if run.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError(f"Snapshot {run_id!r} became selector-eligible.")


def publish_detection_snapshot_pair(
    *,
    analysis_zarr: Path,
    source_detect_group_path: str,
    source_refined_group_path: str,
    recording_identity: str,
    canonical_run_id: str,
    refined_run_id: str,
    scratch_root: Path,
    allow_initialize_missing_source_keys: bool = False,
    allow_manual_score_reset: bool = False,
    copy_backend: str = "python",
    keep_scratch: bool = False,
    coordinate_catalog: bool = False,
) -> dict[str, object]:
    """Publish one full-acquisition immutable pair without selecting it."""

    if type(coordinate_catalog) is not bool:
        raise TypeError("coordinate_catalog must be an exact bool.")
    started = time.perf_counter()
    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    source_detect_relative = _require_relative_group_path(
        source_detect_group_path,
        label="source_detect_group_path",
    )
    source_refined_relative = _require_relative_group_path(
        source_refined_group_path,
        label="source_refined_group_path",
    )
    canonical_id = _require_run_id(canonical_run_id, label="canonical_run_id")
    refined_id = _require_run_id(refined_run_id, label="refined_run_id")
    identity = str(recording_identity).strip()
    if not identity:
        raise ValueError("recording_identity cannot be empty.")
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be 'python' or 'rsync'.")
    scratch = _require_node_local_scratch(scratch_root)
    target_canonical = archive / "detect_runs" / canonical_id
    target_refined = archive / "refined_detect_runs" / refined_id
    collisions = [path for path in (target_canonical, target_refined) if path.exists()]
    if collisions:
        raise FileExistsError(
            "Immutable snapshot target already exists: "
            + ", ".join(str(path) for path in collisions)
        )

    archive_root = open_zarr_root(archive, mode="r")
    persisted_recording_identity = str(
        archive_root.attrs.get("recording_id") or ""
    ).strip()
    if not persisted_recording_identity:
        raise ValueError(
            "Detection snapshot publication requires root recording_id authority."
        )
    if persisted_recording_identity != identity:
        raise ValueError(
            "Requested recording_identity differs from the archive recording_id."
        )

    source_detect = archive / source_detect_relative
    source_refined = archive / source_refined_relative
    if not source_detect.is_dir() or not source_refined.is_dir():
        raise FileNotFoundError(
            "Detection snapshot sources are missing: "
            f"detect={source_detect}, refined={source_refined}"
        )

    session = scratch / f"palette_detection_snapshots_{uuid.uuid4().hex}"
    local_root = session / ".palette_benchmarks" / "production_candidate"
    local_root.mkdir(parents=True, exist_ok=False)
    success = False
    try:
        canonical = publish_legacy_canonical_detection_shadow(
            source_group_path=source_detect,
            recording_identity=identity,
            source_run_id=source_detect.name,
            destination=local_root / "canonical.zarr",
            run_id=canonical_id,
            shadow_root=local_root,
            coordinate_catalog=coordinate_catalog,
        )
        source_refined_group = zarr.open_group(
            str(source_refined), mode="r", use_consolidated=False
        )
        source_detect_group = zarr.open_group(
            str(source_detect), mode="r", use_consolidated=False
        )
        transition = build_refined_detection_transition(
            source_refined_group,
            n_frames=canonical.dimensions.n_frames,
            source_width=canonical.dimensions.source_width,
            source_height=canonical.dimensions.source_height,
            recording_identity=identity,
            source_detect_group=source_detect_group,
            allow_manual_score_reset=bool(allow_manual_score_reset),
            allow_initialize_missing_source_keys=bool(
                allow_initialize_missing_source_keys
            ),
        )
        refined_ids = np.asarray(
            transition.arrays["instances/refined_row_ids"], dtype=np.int64
        )
        next_refined_row_id = int(refined_ids.max()) + 1 if refined_ids.size else 0
        lineage = RefinedDetectionSnapshotLineage(
            lineage_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"palette:refined-detection-lineage:{identity}:{source_refined_relative}",
                )
            ),
            snapshot_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"palette:refined-detection-snapshot:{identity}:{refined_id}",
                )
            ),
            recording_identity=identity,
            next_refined_row_id=next_refined_row_id,
        )
        refined = publish_refined_detection_shadow(
            transition,
            destination=local_root / "refined.zarr",
            run_id=refined_id,
            lineage=lineage,
            canonical_source=canonical,
            shadow_root=local_root,
            coordinate_catalog=coordinate_catalog,
        )
        refined_plans = plan_refined_detection_storage(transition.dimensions)
        if refined_plans.profile.profile_id != canonical.plans.profile.profile_id:
            raise RuntimeError(
                "Canonical and refined snapshot storage profiles unexpectedly differ."
            )

        local_canonical_run = canonical.output_path / "detect_runs" / canonical_id
        local_refined_run = refined.output_path / "refined_detect_runs" / refined_id
        _mark_local_candidate(
            zarr.open_group(str(local_canonical_run), mode="a", use_consolidated=False),
            source_group_path=source_detect_relative,
        )
        _mark_local_candidate(
            zarr.open_group(str(local_refined_run), mode="a", use_consolidated=False),
            source_group_path=source_refined_relative,
        )

        canonical_local_validation = _validate_canonical_run_path(
            local_canonical_run,
            publication=canonical,
        )
        refined_local_validation = _validate_refined_run_path(
            local_refined_run,
            transition=transition,
            plans=refined_plans,
        )
        if not canonical_local_validation["valid"]:
            raise RuntimeError(
                f"Canonical local candidate is invalid: {canonical_local_validation}"
            )
        if not refined_local_validation["valid"]:
            raise RuntimeError(
                f"Refined local candidate is invalid: {refined_local_validation}"
            )

        def canonical_validator(path: Path) -> Mapping[str, Any]:
            return _validate_canonical_run_path(path, publication=canonical)

        def refined_validator(path: Path) -> Mapping[str, Any]:
            return _validate_refined_run_path(
                path,
                transition=transition,
                plans=refined_plans,
            )

        def complete_run(_root: Any, _parent: Any, run: Any) -> None:
            if (
                run.attrs.get("status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError(
                    "Imported immutable snapshot is not complete and staged."
                )

        canonical_publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=archive,
                local_run_path=local_canonical_run,
                target_run_path=target_canonical,
                run_name=canonical_id,
                lock_suffix="detection_snapshot_publication",
                publish_schema_id=DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_ID,
                policy=DETECTION_SNAPSHOT_PUBLICATION_POLICY,
                rollback_policy=DETECTION_SNAPSHOT_ROLLBACK_POLICY,
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=canonical_validator,
            prepare_parents=lambda root: _prepare_parent(root, "detect_runs"),
            complete_run=complete_run,
            verify_pointers=lambda root: _require_unselected(
                root, family_name="detect_runs", run_id=canonical_id
            ),
            payload_metadata={
                "snapshot_role": "canonical_raw_detection_v1",
                "source_group_path": source_detect_relative,
                "selector_activation": "deferred",
            },
        )
        refined_publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=archive,
                local_run_path=local_refined_run,
                target_run_path=target_refined,
                run_name=refined_id,
                lock_suffix="detection_snapshot_publication",
                publish_schema_id=DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_ID,
                policy=DETECTION_SNAPSHOT_PUBLICATION_POLICY,
                rollback_policy=DETECTION_SNAPSHOT_ROLLBACK_POLICY,
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=refined_validator,
            prepare_parents=lambda root: _prepare_parent(root, "refined_detect_runs"),
            complete_run=complete_run,
            verify_pointers=lambda root: _require_unselected(
                root, family_name="refined_detect_runs", run_id=refined_id
            ),
            payload_metadata={
                "snapshot_role": "refined_detection_v1",
                "source_group_path": source_refined_relative,
                "canonical_source_run_id": canonical_id,
                "selector_activation": "deferred",
            },
        )

        consolidation = consolidate_metadata_capture_expected_warnings(archive)
        root = open_zarr_root(archive, mode="r")
        canonical_run = root[f"detect_runs/{canonical_id}"]
        refined_run = root[f"refined_detect_runs/{refined_id}"]
        # The shadow validator is also the complete canonical v1 validator and
        # additionally reopens the immutable compatibility source evidence.
        # Rebind it to the published archive so source drift during copy-back
        # cannot pass as a valid production candidate.
        canonical_errors = validate_canonical_detection_shadow_publication(
            CanonicalDetectionShadowPublication(
                output_path=archive,
                run_id=canonical_id,
                dimensions=canonical.dimensions,
                plans=canonical.plans,
                manifest=dict(canonical_run.attrs["run_manifest"]),
                arrays=_canonical_arrays(canonical_run),
                receipt=canonical.receipt,
            )
        )
        refined_direct, refined_consolidated = (
            refined_detection_metadata_declaration_maps(
                archive,
                run_id=refined_id,
                plans=refined_plans,
            )
        )
        refined_errors = validate_refined_detection_publication(
            dict(refined_run.attrs["run_manifest"]),
            direct_metadata_declarations=refined_direct,
            consolidated_metadata_declarations=refined_consolidated,
            arrays=_refined_arrays(refined_run, dimensions=transition.dimensions),
        )
        _require_unselected(root, family_name="detect_runs", run_id=canonical_id)
        _require_unselected(root, family_name="refined_detect_runs", run_id=refined_id)
        if canonical_errors or refined_errors:
            raise RuntimeError(
                "Published snapshot validation failed: "
                f"canonical={list(canonical_errors)}, refined={list(refined_errors)}"
            )

        result: dict[str, object] = {
            "schema_id": DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_ID,
            "schema_version": DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "published_at_utc": utc_now(),
            "analysis_zarr": str(archive),
            "recording_identity": identity,
            "node_local_materialization": {
                "session_path": str(session),
                "retained_after_success": bool(keep_scratch),
            },
            "source": {
                "detect_group_path": source_detect_relative,
                "refined_group_path": source_refined_relative,
            },
            "snapshots": {
                "canonical": {
                    "run_id": canonical_id,
                    "group_path": f"detect_runs/{canonical_id}",
                    "manifest_digest": canonical.manifest["payload_digest"],
                    "manifest_schema_version": canonical.manifest["schema_version"],
                    "publication": canonical_publication,
                },
                "refined": {
                    "run_id": refined_id,
                    "group_path": f"refined_detect_runs/{refined_id}",
                    "manifest_digest": refined.manifest["payload_digest"],
                    "manifest_schema_version": refined.manifest["schema_version"],
                    "publication": refined_publication,
                },
            },
            "selector_eligible": False,
            "selector_activation": "deferred_separate_reviewed_change",
            "registry_updated": False,
            "storage_profiles": {
                "canonical": canonical.plans.profile.profile_id,
                "refined": refined_plans.profile.profile_id,
            },
            "transition_report": dict(transition.report),
            "consolidation": consolidation,
            "validation": {
                "canonical_errors": [],
                "refined_errors": [],
                "direct_consolidated_metadata_equal": True,
            },
            "total_seconds": float(time.perf_counter() - started),
        }
        success = True
        return json_attr_safe(result)
    finally:
        if session.exists() and success and not keep_scratch:
            shutil.rmtree(session)


__all__ = [
    "DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_ID",
    "DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_VERSION",
    "publish_detection_snapshot_pair",
]
