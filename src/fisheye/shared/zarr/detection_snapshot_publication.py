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

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.canonical_detection_benchmark_input import (
    CanonicalDetectionBenchmarkInput,
    load_detection_benchmark_input,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    build_legacy_detection_source_evidence,
    validate_canonical_detection_publication,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    CanonicalDetectionShadowPublication,
    publish_legacy_canonical_detection_shadow,
    validate_canonical_detection_shadow_publication,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import (
    plan_canonical_detection_storage,
)
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
from fisheye.shared.zarr_run_completion import mark_run_complete


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
CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_ID = (
    "palette.canonical_detection_successor.production_publication"
)
CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_VERSION = 1
CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_POLICY = (
    "node_local_canonical_v3_then_atomic_selector_ineligible_import_v1"
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


def _selector_snapshot(family: Any) -> dict[str, dict[str, object]]:
    return {
        name: {
            "present": name in family.attrs,
            "value": family.attrs.get(name),
        }
        for name in _SELECTOR_ATTRS
    }


def _require_selector_snapshot(
    root: Any,
    *,
    family_name: str,
    expected: Mapping[str, object],
) -> None:
    observed = _selector_snapshot(root[family_name])
    if observed != dict(expected):
        raise RuntimeError(
            f"{family_name} selectors changed during successor publication."
        )


def _canonical_input_preserving_source_keys(
    source_group_path: Path,
    *,
    recording_identity: str,
) -> CanonicalDetectionBenchmarkInput:
    benchmark = load_detection_benchmark_input(
        source_group_path,
        recording_identity=recording_identity,
        frame_limit=None,
    )
    source = zarr.open_group(
        str(source_group_path),
        mode="r",
        use_consolidated=False,
    )
    if "instance_key" not in source:
        raise ValueError(
            "Canonical successor source lacks persisted instance_key values."
        )
    keys = np.ascontiguousarray(np.asarray(source["instance_key"][:]))
    arrays = dict(benchmark.arrays)
    arrays["instances/instance_key"] = keys
    source_identity = dict(benchmark.source_identity)
    conversion = dict(source_identity.get("conversion") or {})
    conversion["instance_key"] = "preserved_from_source_explicit_policy"
    source_identity["conversion"] = conversion
    return CanonicalDetectionBenchmarkInput(
        dimensions=benchmark.dimensions,
        arrays=arrays,
        source_identity=source_identity,
    )


def inspect_canonical_detection_successor_source(
    *,
    analysis_zarr: Path,
    source_detect_group_path: str,
    recording_identity: str,
    successor_run_id: str,
) -> dict[str, object]:
    """Read and validate one raw source without creating a successor."""

    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    source_relative = _require_relative_group_path(
        source_detect_group_path,
        label="source_detect_group_path",
    )
    source_parts = Path(source_relative).parts
    if len(source_parts) != 2 or source_parts[0] != "detect_runs":
        raise ValueError(
            "Canonical successor source must be exactly detect_runs/<run>."
        )
    successor_id = _require_run_id(
        successor_run_id,
        label="successor_run_id",
    )
    target_relative = f"detect_runs/{successor_id}"
    if source_relative == target_relative:
        raise ValueError("Canonical successor cannot replace its source run.")
    identity = str(recording_identity).strip()
    if not identity:
        raise ValueError("recording_identity cannot be empty.")

    root = open_zarr_root(archive, mode="r")
    observed_identity = str(root.attrs.get("recording_id") or "").strip()
    if observed_identity != identity:
        raise ValueError(
            "Requested recording_identity differs from the archive recording_id."
        )
    source_path = archive / source_relative
    if not source_path.is_dir():
        raise FileNotFoundError(f"Detection source not found: {source_path}")
    target_path = archive / target_relative
    if target_path.exists():
        raise FileExistsError(f"Immutable successor target exists: {target_path}")

    benchmark = _canonical_input_preserving_source_keys(
        source_path,
        recording_identity=identity,
    )
    source_group = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    source_evidence = build_legacy_detection_source_evidence(
        source_group,
        source_group_path=source_path,
        source_run_id=source_parts[1],
        recording_identity=identity,
    )
    plans = plan_canonical_detection_storage(benchmark.dimensions)
    return json_attr_safe(
        {
            "schema_id": ("palette.canonical_detection_successor.source_inspection"),
            "schema_version": 1,
            "status": "ready",
            "analysis_zarr": str(archive),
            "recording_identity": identity,
            "source_group_path": source_relative,
            "source_run_id": source_parts[1],
            "successor_run_id": successor_id,
            "successor_group_path": target_relative,
            "dimensions": benchmark.dimensions.as_manifest(),
            "source_instance_key_sha256": sha256_array(
                benchmark.arrays["instances/instance_key"]
            ),
            "instance_key_policy": "preserved_from_source",
            "source_evidence": source_evidence,
            "storage_profile_id": plans.profile.profile_id,
            "coordinate_catalog": True,
            "selectors_before": _selector_snapshot(root["detect_runs"]),
            "selector_eligible": False,
            "registry_updated": False,
        }
    )


def publish_canonical_detection_successor(
    *,
    analysis_zarr: Path,
    source_detect_group_path: str,
    recording_identity: str,
    successor_run_id: str,
    scratch_root: Path,
    copy_backend: str = "python",
    keep_scratch: bool = False,
    result_json: Path | None = None,
) -> dict[str, object]:
    """Publish one canonical-v3 raw successor without selecting it."""

    started = time.perf_counter()
    inspection = inspect_canonical_detection_successor_source(
        analysis_zarr=analysis_zarr,
        source_detect_group_path=source_detect_group_path,
        recording_identity=recording_identity,
        successor_run_id=successor_run_id,
    )
    archive = Path(str(inspection["analysis_zarr"]))
    source_relative = str(inspection["source_group_path"])
    source_path = archive / source_relative
    successor_id = str(inspection["successor_run_id"])
    identity = str(inspection["recording_identity"])
    target = archive / "detect_runs" / successor_id
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be 'python' or 'rsync'.")
    scratch = _require_node_local_scratch(scratch_root)
    selectors_before = dict(inspection["selectors_before"])

    session = scratch / f"palette_canonical_detection_successor_{uuid.uuid4().hex}"
    local_root = session / ".palette_benchmarks" / "production_candidate"
    local_root.mkdir(parents=True, exist_ok=False)
    success = False
    try:
        canonical = publish_legacy_canonical_detection_shadow(
            source_group_path=source_path,
            recording_identity=identity,
            source_run_id=Path(source_relative).name,
            destination=local_root / "canonical.zarr",
            run_id=successor_id,
            shadow_root=local_root,
            coordinate_catalog=True,
            preserve_source_instance_keys=True,
        )
        if canonical.receipt.get("instance_key_policy") != "preserved_from_source":
            raise RuntimeError(
                "Canonical successor did not preserve source instance keys."
            )
        local_run = canonical.output_path / "detect_runs" / successor_id
        _mark_local_candidate(
            zarr.open_group(str(local_run), mode="a", use_consolidated=False),
            source_group_path=source_relative,
        )
        local_validation = _validate_canonical_run_path(
            local_run,
            publication=canonical,
        )
        if not local_validation["valid"]:
            raise RuntimeError(
                f"Canonical local successor is invalid: {local_validation}"
            )

        def validator(path: Path) -> Mapping[str, Any]:
            return _validate_canonical_run_path(path, publication=canonical)

        def complete_run(_root: Any, parent: Any, run: Any) -> None:
            if (
                run.attrs.get("status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError(
                    "Imported canonical successor is not complete and ineligible."
                )
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=successor_id,
                allow_missing_run_provenance=True,
                missing_run_provenance_reason=(
                    "canonical_v3_successor_manifest_binds_legacy_source_evidence"
                ),
            )

        postcopy: dict[str, object] = {}

        def finalize_visibility(_root: Any, _parent: Any, _run: Any) -> None:
            consolidation = consolidate_metadata_capture_expected_warnings(archive)
            published_root = open_zarr_root(archive, mode="r")
            _require_unselected(
                published_root,
                family_name="detect_runs",
                run_id=successor_id,
            )
            _require_selector_snapshot(
                published_root,
                family_name="detect_runs",
                expected=selectors_before,
            )
            published_run = published_root[f"detect_runs/{successor_id}"]
            errors = validate_canonical_detection_shadow_publication(
                CanonicalDetectionShadowPublication(
                    output_path=archive,
                    run_id=successor_id,
                    dimensions=canonical.dimensions,
                    plans=canonical.plans,
                    manifest=dict(published_run.attrs["run_manifest"]),
                    arrays=_canonical_arrays(published_run),
                    receipt=canonical.receipt,
                )
            )
            if errors:
                raise RuntimeError(
                    "Published canonical successor validation failed: "
                    + "; ".join(errors)
                )
            postcopy.update(
                {
                    "consolidation": consolidation,
                    "canonical_errors": [],
                    "direct_consolidated_metadata_equal": True,
                    "source_binding_revalidated_after_copy": True,
                }
            )

        def repair_failed_visibility(_target: Path) -> None:
            consolidate_metadata_capture_expected_warnings(archive)

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=archive,
                local_run_path=local_run,
                target_run_path=target,
                run_name=successor_id,
                lock_suffix="canonical_detection_successor_publication",
                publish_schema_id=(CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_ID),
                policy=CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_POLICY,
                rollback_policy=DETECTION_SNAPSHOT_ROLLBACK_POLICY,
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validator,
            prepare_parents=lambda root: _prepare_parent(root, "detect_runs"),
            complete_run=complete_run,
            verify_pointers=lambda root: (
                _require_unselected(
                    root,
                    family_name="detect_runs",
                    run_id=successor_id,
                ),
                _require_selector_snapshot(
                    root,
                    family_name="detect_runs",
                    expected=selectors_before,
                ),
            ),
            payload_metadata={
                "snapshot_role": "canonical_raw_detection_v3_successor",
                "source_group_path": source_relative,
                "instance_key_policy": "preserved_from_source",
                "selector_activation": "deferred",
            },
            activate_run=finalize_visibility,
            repair_failed_publication_visibility=repair_failed_visibility,
        )
        if not postcopy:
            raise RuntimeError(
                "Canonical successor publication omitted final visibility validation."
            )
        result: dict[str, object] = {
            "schema_id": CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_ID,
            "schema_version": (
                CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_VERSION
            ),
            "status": "complete",
            "published_at_utc": utc_now(),
            "analysis_zarr": str(archive),
            "recording_identity": identity,
            "source": {
                "group_path": source_relative,
                "run_id": Path(source_relative).name,
                "instance_key_sha256": inspection["source_instance_key_sha256"],
            },
            "successor": {
                "run_id": successor_id,
                "group_path": f"detect_runs/{successor_id}",
                "manifest_schema_version": canonical.manifest["schema_version"],
                "manifest_digest": canonical.manifest["payload_digest"],
                "logical_content_digest": canonical.manifest["payload"][
                    "logical_content"
                ]["digest"],
                "instance_key_sha256": sha256_array(
                    canonical.arrays["instances/instance_key"]
                ),
            },
            "dimensions": canonical.dimensions.as_manifest(),
            "storage_profile_id": canonical.plans.profile.profile_id,
            "coordinate_catalog": True,
            "instance_key_policy": "preserved_from_source",
            "publication": publication,
            "validation": postcopy,
            "selectors_before": selectors_before,
            "selectors_after": selectors_before,
            "selector_eligible": False,
            "selector_activation": "deferred_separate_reviewed_change",
            "registry_updated": False,
            "node_local_materialization": {
                "session_path": str(session),
                "retained_after_success": bool(keep_scratch),
            },
            "total_seconds": float(time.perf_counter() - started),
        }
        safe_result = json_attr_safe(result)
        if result_json is not None:
            write_json_atomic(result_json.expanduser().resolve(), safe_result)
        success = True
        return safe_result
    finally:
        if session.exists() and success and not keep_scratch:
            shutil.rmtree(session)


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
    "CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_ID",
    "CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_VERSION",
    "DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_ID",
    "DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_VERSION",
    "inspect_canonical_detection_successor_source",
    "publish_canonical_detection_successor",
    "publish_detection_snapshot_pair",
]
