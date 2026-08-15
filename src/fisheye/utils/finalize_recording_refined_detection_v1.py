"""Finalize one working refined run as an immutable recording-level v1 authority."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
import uuid

import numpy as np
import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.canonical_detection_manifest import (
    canonical_detection_dimensions_from_manifest,
    require_active_coordinate_canonical_detection,
    refined_source_identity_from_canonical_manifest,
    validate_canonical_detection_publication,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    canonical_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import plan_canonical_detection_storage
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
    build_coordinate_refined_detection_run_manifest,
    validate_refined_detection_publication,
)
from fisheye.shared.zarr.refined_detection_schema import REFINED_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.refined_detection_snapshot import (
    publish_selector_ineligible_refined_detection_snapshot,
    refined_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.refined_detection_storage import (
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_refined_detection_transition,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    storage_profile_from_manifest,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)


FINALIZATION_SCHEMA_ID = "palette.recording_refined_detection.finalization"
FINALIZATION_SCHEMA_VERSION = 1
FINALIZATION_POLICY = "working_refined_to_immutable_recording_snapshot_v1"
ROLLBACK_POLICY = "retain_failed_selector_ineligible_finalized_child_v1"
_SELECTOR_ATTRIBUTES = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "authoritative_refined_run",
)


def _require_scratch_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Finalization scratch root not found: {resolved}")
    if resolved in {Path("/").resolve(), Path("/tmp").resolve(), Path("/scratch").resolve()}:
        raise ValueError("Finalization scratch must be one bounded node-local directory.")
    if str(resolved).startswith(("/groups/", "/nrs/")):
        raise ValueError("Finalization scratch must be node-local, not shared storage.")
    return resolved


def _require_child_name(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized or "/" in normalized or normalized in {".", ".."}:
        raise ValueError(f"{label} must be one nonempty child-group name.")
    return normalized


def _load_canonical_source(root: zarr.Group, archive: Path, run_id: str):
    parent = root.get("detect_runs")
    if parent is None or run_id not in parent:
        raise ValueError(f"Canonical detection run not found: detect_runs/{run_id}")
    run = parent[run_id]
    manifest = run.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Canonical detection source lacks its exact run_manifest.")
    dimensions = canonical_detection_dimensions_from_manifest(manifest)
    arrays = {
        path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
    }
    profile = storage_profile_from_manifest(
        manifest["payload"]["storage_plan"]["storage_profile"]
    )
    plans = plan_canonical_detection_storage(dimensions, profile=profile)
    direct, consolidated = canonical_detection_metadata_declaration_maps(
        archive,
        run_id=run_id,
        plans=plans,
    )
    errors = validate_canonical_detection_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
    )
    if errors:
        raise ValueError("Canonical detection source is invalid: " + "; ".join(errors))
    return run, manifest, dimensions, arrays


def _validate_gate_binding(
    working: zarr.Group,
    *,
    requirement: str,
    expected_gate_run: str | None,
) -> dict[str, object]:
    mode = str(requirement).strip()
    if mode not in {"off", "if_available", "required"}:
        raise ValueError("registered_gate_requirement must be off, if_available, or required.")
    observed_mode = str(
        working.attrs.get("registered_detection_gate_requirement") or ""
    ).strip()
    if observed_mode != mode:
        raise ValueError(
            "Working refined run gate requirement differs from finalization input: "
            f"{observed_mode!r} != {mode!r}."
        )
    evidence = working.attrs.get("registered_detection_gate")
    if not isinstance(evidence, Mapping):
        raise ValueError("Working refined run lacks registered detection gate evidence.")
    normalized = json_attr_safe(dict(evidence))
    if normalized.get("requirement") != mode:
        raise ValueError("Gate evidence requirement differs from the working run policy.")
    applied = normalized.get("applied") is True and normalized.get("status") == "applied"
    observed_gate = str(normalized.get("gate_run") or "").strip() or None
    expected = str(expected_gate_run or "").strip() or None
    if expected is not None and observed_gate != expected:
        raise ValueError(
            f"Working refined run consumed gate {observed_gate!r}, expected {expected!r}."
        )
    if mode == "required" and not applied:
        raise ValueError("Required geometry finalization needs an applied exact gate.")
    if mode == "off" and (applied or normalized.get("status") != "off"):
        raise ValueError("Configured-off finalization has inconsistent gate evidence.")
    if mode == "if_available" and normalized.get("status") not in {
        "applied",
        "unavailable",
        "rejected_invalid",
    }:
        raise ValueError("if_available finalization has an unknown gate disposition.")
    return dict(normalized)


def _prepare_refined_parent(root: zarr.Group) -> tuple[zarr.Group, ...]:
    return (
        require_runs_parent(
            root,
            "refined_detect_runs",
            completion_epoch=COMPLETION_EPOCH_STRICT,
        ),
    )


def _require_unselected(root: zarr.Group, *, run_id: str) -> None:
    parent = root["refined_detect_runs"]
    selected = [name for name in _SELECTOR_ATTRIBUTES if parent.attrs.get(name) == run_id]
    if selected:
        raise RuntimeError(
            f"Selector-ineligible finalized run {run_id!r} is selected by {selected!r}."
        )
    run = parent[run_id]
    if run.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError(f"Finalized run {run_id!r} became selector-eligible.")


def _validate_decoded_run(path: Path, *, dimensions, expected_hashes) -> dict[str, object]:
    try:
        run = zarr.open_group(str(path), mode="r", use_consolidated=False)
        arrays = {
            name: run[name]
            for name in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)
        }
        issues = REFINED_DETECTION_SCHEMA_V1.validate(arrays, dimensions=dimensions)
        observed_hashes = {
            name: sha256_array(np.asarray(array[...]))
            for name, array in arrays.items()
        }
        errors = [
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        ]
        if observed_hashes != expected_hashes:
            errors.append("decoded finalized arrays differ from the frozen transition")
        if run.attrs.get("status") != "complete":
            errors.append("finalized run status is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors.append("finalized run is not selector-ineligible")
        return {"valid": not errors, "errors": errors}
    except Exception as exc:
        return {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}


def _mark_failed_import(path: Path, *, error: BaseException) -> None:
    if not path.is_dir():
        return
    try:
        run = zarr.open_group(str(path), mode="a", use_consolidated=False)
        run.attrs.update(
            {
                "status": "failed",
                "stage_selector_eligible": False,
                "publication_failure": f"{type(error).__name__}: {error}",
            }
        )
        mark_run_failed(
            run,
            run_name=path.name,
            error=f"{type(error).__name__}: {error}",
        )
    except Exception:
        pass


def finalize_recording_refined_detection_v1(
    *,
    analysis_zarr: Path,
    canonical_detect_run: str,
    working_refined_run: str,
    output_run: str,
    recording_identity: str,
    registered_gate_requirement: str,
    registered_gate_run: str | None,
    selection_policy_id: str,
    scratch_root: Path,
    copy_backend: str = "python",
    keep_scratch: bool = False,
    require_active_canonical_source: bool = False,
) -> dict[str, object]:
    """Freeze, validate, and atomically import one immutable refined authority."""

    started = time.perf_counter()
    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir() or archive.suffix != ".zarr":
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    canonical_name = _require_child_name(canonical_detect_run, label="canonical_detect_run")
    working_name = _require_child_name(working_refined_run, label="working_refined_run")
    final_name = _require_child_name(output_run, label="output_run")
    if final_name == working_name:
        raise ValueError("output_run must differ from the working refined run.")
    identity = str(recording_identity).strip()
    if not identity:
        raise ValueError("recording_identity cannot be empty.")
    policy_id = str(selection_policy_id).strip()
    if policy_id not in {"manual_review_only_v1", "corroborated_acquisition_v1"}:
        raise ValueError("Unsupported registered geometry selection policy id.")
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be python or rsync.")
    if type(require_active_canonical_source) is not bool:
        raise TypeError("require_active_canonical_source must be an exact bool.")
    if registered_gate_requirement == "required" and not str(
        registered_gate_run or ""
    ).strip():
        raise ValueError("Required geometry finalization needs one exact gate run.")
    scratch = _require_scratch_root(scratch_root)
    target = archive / "refined_detect_runs" / final_name
    if target.exists():
        raise FileExistsError(f"Immutable finalized refined run already exists: {target}")

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    if require_active_canonical_source:
        require_active_coordinate_canonical_detection(
            root,
            group_path=f"detect_runs/{canonical_name}",
        )
    canonical, canonical_manifest, dimensions, _canonical_arrays = _load_canonical_source(
        root,
        archive,
        canonical_name,
    )
    source_evidence = canonical_manifest["payload"].get("source_evidence")
    if (
        not isinstance(source_evidence, Mapping)
        or source_evidence.get("recording_identity") != identity
    ):
        raise ValueError(
            "Canonical detection recording identity differs from finalization input."
        )
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None or working_name not in refined_parent:
        raise ValueError(f"Working refined run not found: refined_detect_runs/{working_name}")
    working = refined_parent[working_name]
    if working.attrs.get("palette_run_completion_status") != "complete":
        raise ValueError("Working refined run is not explicitly complete.")
    if str(working.attrs.get("source_detect_run") or "").strip() != canonical_name:
        raise ValueError("Working refined run does not bind the canonical detection run.")
    gate_evidence = _validate_gate_binding(
        working,
        requirement=registered_gate_requirement,
        expected_gate_run=registered_gate_run,
    )
    if gate_evidence.get("applied") is True and gate_evidence.get(
        "comparison_policy_id"
    ) != policy_id:
        raise ValueError(
            "Configured selection policy differs from the comparison policy bound "
            "to the consumed registered gate."
        )
    transition = build_refined_detection_transition(
        working,
        n_frames=dimensions.n_frames,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
        recording_identity=identity,
        source_detect_group=canonical,
    )
    refined_ids = np.asarray(
        transition.arrays["instances/refined_row_ids"], dtype=np.int64
    ).reshape(-1)
    next_id = 0 if refined_ids.size == 0 else int(np.max(refined_ids)) + 1
    lineage = RefinedDetectionSnapshotLineage(
        lineage_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"palette:refined-lineage:{identity}")),
        snapshot_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"palette:refined-snapshot:{identity}:{final_name}",
            )
        ),
        recording_identity=identity,
        next_refined_row_id=next_id,
    )
    source = refined_source_identity_from_canonical_manifest(canonical_manifest)
    expected_hashes = {
        name: sha256_array(np.asarray(values))
        for name, values in transition.arrays.items()
    }

    session = scratch / f"palette_refined_finalize_{uuid.uuid4().hex}"
    local_archive = session / ".palette_benchmarks" / "finalized_refined.zarr"
    local_archive.parent.mkdir(parents=True, exist_ok=False)
    imported = False
    success = False
    try:
        _publication = publish_selector_ineligible_refined_detection_snapshot(
            dimensions=transition.dimensions,
            arrays=transition.arrays,
            instance_reason_codes=transition.instance_reason_codes,
            source_reason_codes=transition.source_reason_codes,
            destination=local_archive,
            run_id=final_name,
            lineage=lineage,
            source=source,
            created_by="recording_refined_detection_finalizer",
            publication_kind="recording_level_finalized_refined_authority",
            safe_root=local_archive.parent,
            profile=DETECTION_PUBLISHED_ACCESS_AWARE_V1,
            run_attributes={
                "immutable_snapshot": True,
                "finalized_recording_authority": True,
                "production_candidate": True,
                "source_working_refined_run": working_name,
                "source_working_refined_completion_uuid": working.attrs.get(
                    "palette_run_completion_uuid"
                ),
                "registered_detection_gate_requirement": registered_gate_requirement,
                "registered_detection_gate": gate_evidence,
                "registered_detection_gate_consumed": bool(gate_evidence.get("applied")),
                "registered_geometry_selection_policy_id": policy_id,
            },
            coordinate_catalog=True,
        )
        local_run_path = local_archive / "refined_detect_runs" / final_name
        local_run = zarr.open_group(str(local_run_path), mode="a", use_consolidated=False)
        local_run.attrs["benchmark_only"] = False
        for name in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(transition.dimensions):
            array_attrs = dict(local_run[name].attrs)
            array_attrs["benchmark_only"] = False
            local_run[name].attrs.put(array_attrs)
        consolidate_metadata_capture_expected_warnings(local_archive)

        def validator(path: Path) -> Mapping[str, object]:
            return _validate_decoded_run(
                path,
                dimensions=transition.dimensions,
                expected_hashes=expected_hashes,
            )

        def complete_run(_root: zarr.Group, _parent: zarr.Group, run: zarr.Group) -> None:
            mark_run_complete(run, run_name=final_name)

        atomic_receipt = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=archive,
                local_run_path=local_run_path,
                target_run_path=target,
                run_name=final_name,
                lock_suffix="recording_refined_detection_finalization",
                publish_schema_id=FINALIZATION_SCHEMA_ID,
                policy=FINALIZATION_POLICY,
                rollback_policy=ROLLBACK_POLICY,
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validator,
            prepare_parents=_prepare_refined_parent,
            complete_run=complete_run,
            verify_pointers=lambda current: _require_unselected(current, run_id=final_name),
            payload_metadata={
                "source_working_refined_run": working_name,
                "source_canonical_detect_run": canonical_name,
                "registered_gate_requirement": registered_gate_requirement,
                "registered_gate_run": gate_evidence.get("gate_run"),
                "selection_policy_id": policy_id,
            },
        )
        imported = True

        imported_run = zarr.open_group(str(target), mode="a", use_consolidated=False)
        imported_arrays = {
            name: imported_run[name]
            for name in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(
                transition.dimensions
            )
        }
        plans = plan_refined_detection_storage(
            transition.dimensions,
            profile=DETECTION_PUBLISHED_ACCESS_AWARE_V1,
        )
        consolidate_metadata_capture_expected_warnings(archive)
        direct, consolidated = refined_detection_metadata_declaration_maps(
            archive,
            run_id=final_name,
            plans=plans,
        )
        final_manifest = build_coordinate_refined_detection_run_manifest(
            run_id=final_name,
            dimensions=transition.dimensions,
            storage_plan=plans,
            lineage=lineage,
            source=source,
            instance_reason_codes=transition.instance_reason_codes,
            source_reason_codes=transition.source_reason_codes,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            selector_eligible=False,
        )
        imported_run.attrs["run_manifest"] = final_manifest
        consolidate_metadata_capture_expected_warnings(archive)
        direct, consolidated = refined_detection_metadata_declaration_maps(
            archive,
            run_id=final_name,
            plans=plans,
        )
        errors = validate_refined_detection_publication(
            final_manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=imported_arrays,
        )
        if errors:
            raise RuntimeError(
                "Imported finalized refined authority is invalid: " + "; ".join(errors)
            )
        bound = bind_refined_detection_crop_source(
            archive,
            run_id=final_name,
            allow_selector_ineligible_benchmark=True,
        )
        if bound.manifest.get("payload_digest") != final_manifest["payload_digest"]:
            raise RuntimeError("Final crop-source rebind returned a different manifest.")
        _require_unselected(
            zarr.open_group(str(archive), mode="r", use_consolidated=False),
            run_id=final_name,
        )
        success = True
        return json_attr_safe(
            {
                "schema_id": FINALIZATION_SCHEMA_ID,
                "schema_version": FINALIZATION_SCHEMA_VERSION,
                "status": "complete",
                "finalized_at_utc": utc_now(),
                "analysis_zarr": str(archive),
                "canonical_detect_run": canonical_name,
                "required_active_canonical_source": (
                    require_active_canonical_source
                ),
                "working_refined_run": working_name,
                "output_run": final_name,
                "output_group_path": f"refined_detect_runs/{final_name}",
                "recording_identity": identity,
                "run_manifest_digest": final_manifest["payload_digest"],
                "logical_content_digest": bound.logical_content_digest,
                "registered_detection_gate": gate_evidence,
                "selection_policy_id": policy_id,
                "selector_eligible": False,
                "registry_updated": False,
                "raw_detection_unchanged": True,
                "working_refined_unchanged": True,
                "atomic_publication": atomic_receipt,
                "total_seconds": float(time.perf_counter() - started),
            }
        )
    except BaseException as exc:
        if imported:
            _mark_failed_import(target, error=exc)
            try:
                consolidate_metadata_capture_expected_warnings(archive)
            except Exception:
                pass
        raise
    finally:
        if session.exists() and success and not keep_scratch:
            shutil.rmtree(session)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--canonical-detect-run", required=True)
    parser.add_argument("--working-refined-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument(
        "--registered-gate-requirement",
        choices=("off", "if_available", "required"),
        required=True,
    )
    parser.add_argument("--registered-gate-run")
    parser.add_argument("--selection-policy-id", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--require-active-canonical-source", action="store_true")
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = finalize_recording_refined_detection_v1(
            analysis_zarr=args.analysis_zarr,
            canonical_detect_run=args.canonical_detect_run,
            working_refined_run=args.working_refined_run,
            output_run=args.output_run,
            recording_identity=args.recording_identity,
            registered_gate_requirement=args.registered_gate_requirement,
            registered_gate_run=args.registered_gate_run,
            selection_policy_id=args.selection_policy_id,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
            keep_scratch=bool(args.keep_scratch),
            require_active_canonical_source=args.require_active_canonical_source,
        )
    except Exception as exc:
        result = {
            "schema_id": FINALIZATION_SCHEMA_ID,
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "status": "failed",
            "analysis_zarr": str(args.analysis_zarr),
            "output_run": args.output_run,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FINALIZATION_SCHEMA_ID",
    "FINALIZATION_SCHEMA_VERSION",
    "finalize_recording_refined_detection_v1",
]
