"""Atomically import one validated keypoint-v2 chain into an analysis archive.

Computation and strict manifest construction happen outside the recording Zarr.
This module copies only complete immutable run groups, leaves every selector
unchanged, consolidates once after all four imports, and then re-runs the exact
logical and physical publication gates against the public paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import zarr

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.body_frame_manifest import validate_body_frame_publication
from fisheye.shared.zarr.body_frame_publication import (
    body_frame_metadata_declaration_maps,
)
from fisheye.shared.zarr.clipped_keypoint_finalization import (
    ClippedKeypointPublicationChain,
)
from fisheye.shared.zarr.crop_manifest import CROP_RUN_MANIFEST_ATTRIBUTE
from fisheye.shared.zarr.keypoint_manifest import validate_keypoint_publication
from fisheye.shared.zarr.keypoint_publication import (
    keypoint_metadata_declaration_maps,
)
from fisheye.shared.zarr.keypoint_quality_manifest import (
    validate_keypoint_quality_publication,
)
from fisheye.shared.zarr.keypoint_quality_publication import (
    keypoint_quality_metadata_declaration_maps,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.refined_keypoint_manifest import (
    validate_refined_keypoint_publication,
)
from fisheye.shared.zarr.refined_keypoint_publication import (
    refined_keypoint_metadata_declaration_maps,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    is_run_complete_in_parent,
    require_runs_parent,
)


KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_SCHEMA_ID = (
    "palette.keypoint.bundle.production_publication"
)
KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_SCHEMA_VERSION = 1
KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_POLICY = (
    "node_local_v2_chain_then_atomic_selector_ineligible_import_v1"
)
KEYPOINT_BUNDLE_PRODUCTION_ROLLBACK_POLICY = (
    "retain_failed_owner_bound_selector_ineligible_child_v1"
)
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "approved_run",
)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object at {path}.")
    return value


def _metadata_tree(path: Path) -> dict[str, Mapping[str, Any]]:
    root = path.expanduser().resolve()
    declarations: dict[str, Mapping[str, Any]] = {}
    for metadata_path in sorted(root.rglob("zarr.json")):
        declarations[metadata_path.relative_to(root).as_posix()] = _read_json(
            metadata_path
        )
    if not declarations:
        raise ValueError(f"Run group contains no Zarr metadata: {root}")
    return declarations


def _candidate_validator(local_run: Path) -> Callable[[Path], Mapping[str, Any]]:
    expected = _metadata_tree(local_run)

    def validate(path: Path) -> Mapping[str, Any]:
        errors: list[str] = []
        try:
            observed = _metadata_tree(path)
        except Exception as exc:
            return {"valid": False, "errors": [f"metadata reopen failed: {exc}"]}
        if observed != expected:
            errors.append("run metadata declarations differ from the sealed candidate")
        try:
            run = zarr.open_group(str(path), mode="r", use_consolidated=False)
            if run.attrs.get("status") != "complete":
                errors.append("run status is not complete")
            if run.attrs.get("stage_selector_eligible") is not False:
                errors.append("run became selector eligible")
            if run.attrs.get("production_candidate") is not True:
                errors.append("run is not a production candidate")
            if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
                errors.append("strict run completion is absent")
        except Exception as exc:
            errors.append(f"run reopen failed: {exc}")
        return {"valid": not errors, "errors": errors}

    return validate


def _require_unselected(root: Any, *, parent_path: str, run_id: str) -> None:
    parent = root
    for part in parent_path.split("/"):
        parent = parent[part]
    selected = [name for name in _SELECTOR_ATTRS if parent.attrs.get(name) == run_id]
    if selected:
        raise RuntimeError(
            f"Selector-ineligible keypoint candidate {parent_path}/{run_id} "
            f"is selected by {selected!r}."
        )


def _prepare_parent(parent_path: str) -> Callable[[Any], tuple[Any, ...]]:
    parts = parent_path.split("/")

    def prepare(root: Any) -> tuple[Any, ...]:
        parents: list[Any] = []
        node = root
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                node = require_runs_parent(
                    node, part, completion_epoch=COMPLETION_EPOCH_STRICT
                )
            else:
                node = node.require_group(part)
            parents.append(node)
        return tuple(parents)

    return prepare


def _complete_candidate(_root: Any, _parent: Any, run: Any) -> None:
    if (
        run.attrs.get("status") != "complete"
        or run.attrs.get("stage_selector_eligible") is not False
        or run.attrs.get("production_candidate") is not True
        or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
    ):
        raise RuntimeError("Imported keypoint-family run lost its sealed lifecycle.")


def _require_exact_crop_source(
    analysis_zarr: Path,
    chain: ClippedKeypointPublicationChain,
) -> Mapping[str, Any]:
    root = zarr.open_group(
        str(analysis_zarr), mode="r", zarr_format=3, use_consolidated=True
    )
    parent = root.get("crop_runs")
    if parent is None or chain.crop.run_id not in parent:
        raise RuntimeError(f"Destination archive lacks crop_runs/{chain.crop.run_id}.")
    run = parent[chain.crop.run_id]
    if not is_run_complete_in_parent(parent, run):
        raise RuntimeError("Destination crop-v2 source is not strictly complete.")
    persisted = run.attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(persisted, Mapping) or dict(persisted) != dict(
        chain.crop.manifest
    ):
        raise RuntimeError(
            "Destination crop-v2 manifest differs from finalization input."
        )
    return validate_direct_consolidated_subtree(
        analysis_zarr, subtree_path=f"crop_runs/{chain.crop.run_id}"
    ).to_json()


def _import_one(
    *,
    analysis_zarr: Path,
    local_archive: Path,
    parent_path: str,
    run_id: str,
    copy_backend: str,
) -> Mapping[str, Any]:
    local_run = local_archive / parent_path / run_id
    target = analysis_zarr / parent_path / run_id
    validator = _candidate_validator(local_run)
    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=analysis_zarr,
            local_run_path=local_run,
            target_run_path=target,
            run_name=run_id,
            lock_suffix="keypoint_bundle_production_publication",
            publish_schema_id=KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_SCHEMA_ID,
            policy=KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_POLICY,
            rollback_policy=KEYPOINT_BUNDLE_PRODUCTION_ROLLBACK_POLICY,
            content_checksum=True,
            publication_owner_attr="atomic_publication_owner_uuid",
            persist_run_receipt=False,
        ),
        copy_backend=copy_backend,
        validate_run=validator,
        prepare_parents=_prepare_parent(parent_path),
        complete_run=_complete_candidate,
        verify_pointers=lambda root: _require_unselected(
            root, parent_path=parent_path, run_id=run_id
        ),
        payload_metadata={
            "selector_activation": "deferred",
            "run_parent": parent_path,
        },
    )


def _validate_public_chain(
    analysis_zarr: Path,
    chain: ClippedKeypointPublicationChain,
) -> tuple[str, ...]:
    root = zarr.open_group(
        str(analysis_zarr),
        mode="r",
        zarr_format=3,
        use_consolidated=True,
    )
    errors: list[str] = []

    raw_run = root[f"keypoints_runs/{chain.raw.run_id}"]
    raw_arrays = {path: raw_run[path] for path in chain.raw.prepared.arrays}
    raw_direct, raw_consolidated = keypoint_metadata_declaration_maps(
        analysis_zarr, run_id=chain.raw.run_id, plans=chain.raw.plans
    )
    errors.extend(
        f"raw: {error}"
        for error in validate_keypoint_publication(
            chain.raw.manifest,
            direct_metadata_declarations=raw_direct,
            consolidated_metadata_declarations=raw_consolidated,
            arrays=raw_arrays,
            source_crop_arrays=chain.crop.arrays,
            source_crop_manifest=chain.crop.manifest,
        )
    )

    quality_run = root[f"keypoint_quality_runs/{chain.quality.run_id}"]
    quality_arrays = {path: quality_run[path] for path in chain.quality.prepared.arrays}
    quality_direct, quality_consolidated = keypoint_quality_metadata_declaration_maps(
        analysis_zarr, run_id=chain.quality.run_id, plans=chain.quality.plans
    )
    errors.extend(
        f"quality: {error}"
        for error in validate_keypoint_quality_publication(
            chain.quality.manifest,
            direct_metadata_declarations=quality_direct,
            consolidated_metadata_declarations=quality_consolidated,
            arrays=quality_arrays,
            source_arrays=raw_arrays,
            source_manifest=chain.raw.manifest,
        )
    )

    refined_run = root[f"refined_keypoints_runs/{chain.refined.run_id}"]
    refined_arrays = {path: refined_run[path] for path in chain.refined.prepared.arrays}
    refined_direct, refined_consolidated = refined_keypoint_metadata_declaration_maps(
        analysis_zarr, run_id=chain.refined.run_id, plans=chain.refined.plans
    )
    errors.extend(
        f"refined: {error}"
        for error in validate_refined_keypoint_publication(
            chain.refined.manifest,
            arrays=refined_arrays,
            source_crop_arrays=chain.crop.arrays,
            raw_manifest=chain.raw.manifest,
            quality_manifest=chain.quality.manifest,
            crop_manifest=chain.crop.manifest,
            raw_arrays=raw_arrays,
            quality_arrays=quality_arrays,
            retired_instance_keys=chain.refined.retired_instance_keys,
            direct_metadata_declarations=refined_direct,
            consolidated_metadata_declarations=refined_consolidated,
        )
    )

    body_run = root[f"analysis/body_frame_runs/{chain.body_frame.run_id}"]
    body_arrays = {path: body_run[path] for path in chain.body_frame.prepared.arrays}
    body_source_arrays = {
        path: refined_arrays[path]
        for path in (
            "instance_key",
            "frame_indices",
            "keypoint_row_signature",
            "keypoints_img",
            "keypoint_valid",
        )
    }
    body_source_arrays["pose_usable"] = refined_arrays["refined_success"]
    body_direct, body_consolidated = body_frame_metadata_declaration_maps(
        analysis_zarr, run_id=chain.body_frame.run_id, plans=chain.body_frame.plans
    )
    errors.extend(
        f"body_frame: {error}"
        for error in validate_body_frame_publication(
            chain.body_frame.manifest,
            direct_metadata_declarations=body_direct,
            consolidated_metadata_declarations=body_consolidated,
            arrays=body_arrays,
            source_arrays=body_source_arrays,
            source_manifest=chain.refined.manifest,
        )
    )
    for parent_path, run_id in (
        ("keypoints_runs", chain.raw.run_id),
        ("keypoint_quality_runs", chain.quality.run_id),
        ("refined_keypoints_runs", chain.refined.run_id),
        ("analysis/body_frame_runs", chain.body_frame.run_id),
    ):
        try:
            _require_unselected(root, parent_path=parent_path, run_id=run_id)
        except RuntimeError as exc:
            errors.append(str(exc))
    return tuple(errors)


def publish_keypoint_v2_production_candidate_chain(
    *,
    analysis_zarr: Path,
    chain: ClippedKeypointPublicationChain,
    copy_backend: str = "python",
    result_json: Path | None = None,
) -> Mapping[str, Any]:
    """Import a complete four-surface chain without activating any selector."""

    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir() or archive.suffix != ".zarr":
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be 'python' or 'rsync'.")
    crop_source_validation = _require_exact_crop_source(archive, chain)
    imports: dict[str, Mapping[str, Any]] = {}
    for name, local_archive, parent_path, run_id in (
        ("raw_keypoints", chain.raw.output_path, "keypoints_runs", chain.raw.run_id),
        (
            "keypoint_quality",
            chain.quality.output_path,
            "keypoint_quality_runs",
            chain.quality.run_id,
        ),
        (
            "refined_keypoints",
            chain.refined.output_path,
            "refined_keypoints_runs",
            chain.refined.run_id,
        ),
        (
            "body_frame",
            chain.body_frame.output_path,
            "analysis/body_frame_runs",
            chain.body_frame.run_id,
        ),
    ):
        imports[name] = _import_one(
            analysis_zarr=archive,
            local_archive=local_archive,
            parent_path=parent_path,
            run_id=run_id,
            copy_backend=copy_backend,
        )

    consolidation = consolidate_metadata_capture_expected_warnings(archive)
    errors = _validate_public_chain(archive, chain)
    if errors:
        raise RuntimeError(
            "Published keypoint-v2 production candidate gate failed: "
            + "; ".join(errors)
        )
    result = json_attr_safe(
        {
            "schema_id": KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_SCHEMA_ID,
            "schema_version": KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "analysis_zarr": str(archive),
            "runs": {
                "raw_keypoints": chain.raw.run_id,
                "keypoint_quality": chain.quality.run_id,
                "refined_keypoints": chain.refined.run_id,
                "body_frame": chain.body_frame.run_id,
            },
            "atomic_publications": imports,
            "crop_source_validation": crop_source_validation,
            "consolidation": consolidation,
            "selector_eligible": False,
            "selector_activation": "deferred_separate_reviewed_change",
            "registry_updated": False,
            "public_validation_errors": [],
        }
    )
    if result_json is not None:
        write_json_atomic(result_json.expanduser().resolve(), result)
    return result


__all__ = [
    "KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_SCHEMA_ID",
    "KEYPOINT_BUNDLE_PRODUCTION_PUBLICATION_SCHEMA_VERSION",
    "publish_keypoint_v2_production_candidate_chain",
]
