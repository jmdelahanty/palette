"""Exact refined-detection v1 source binding for downstream crop planning.

This module validates selection, the complete logical/physical publication,
and decoded image-space geometry.  It deliberately does not publish crop
coordinate records: the existing canonical crop publisher is bound to raw
``detect_runs`` lineage, and refined lineage must not be relabeled as raw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE,
    REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE,
    REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE,
    refined_detection_dimensions_from_manifest,
    refined_detection_logical_content_digest,
    validate_refined_detection_authority_provenance,
    validate_refined_detection_publication,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    refined_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.refined_detection_storage import (
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_io import open_zarr_root


REFINED_DETECTION_CROP_SOURCE_SCHEMA_ID = (
    "palette.refined_detection.crop_source_handoff"
)
REFINED_DETECTION_CROP_SOURCE_SCHEMA_VERSION = 1
REFINED_DETECTION_CROP_COORDINATE_STATUS = (
    "image_space_values_validated_refined_lineage_publication_pending"
)


class RefinedDetectionCropSourceError(RuntimeError):
    """Raised when a refined snapshot is unsafe for downstream crop planning."""


@dataclass(frozen=True)
class BoundRefinedDetectionCropSource:
    """One exact full-acquisition refined-v1 rowset and its evidence."""

    archive_path: Path
    run_id: str
    run_path: str
    instances_path: str
    selection_mode: str
    manifest: Mapping[str, Any]
    dimensions: RefinedDetectionDimensions
    arrays: Mapping[str, Any]
    run_group: Any
    instances_group: Any
    logical_content_digest: str
    handoff_manifest: Mapping[str, Any]


def _run_arrays(
    run_group: Any,
    *,
    dimensions: RefinedDetectionDimensions,
) -> dict[str, Any]:
    return {
        path: run_group[path]
        for path in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)
    }


def _resolve_selected_run(
    parent: Any,
    *,
    run_id: str | None,
    allow_selector_ineligible_benchmark: bool,
) -> tuple[str, str, Mapping[str, Any] | None]:
    explicit = None if run_id is None else str(run_id).strip()
    if explicit:
        if "/" in explicit:
            raise RefinedDetectionCropSourceError(
                "Refined run_id must be one child-group name."
            )
        mode = (
            "explicit_selector_ineligible_benchmark"
            if allow_selector_ineligible_benchmark
            else "explicit_refined_v1"
        )
        return explicit, mode, None
    if allow_selector_ineligible_benchmark:
        raise RefinedDetectionCropSourceError(
            "Benchmark refined sources require an explicit run_id."
        )
    selected = parent.attrs.get(REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE)
    authority = parent.attrs.get(REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE)
    if not isinstance(selected, str) or not selected.strip():
        raise RefinedDetectionCropSourceError(
            "No approved authoritative refined-detection run is selected."
        )
    if not isinstance(authority, Mapping):
        raise RefinedDetectionCropSourceError(
            "Authoritative refined selection lacks its provenance envelope."
        )
    authority_errors = validate_refined_detection_authority_provenance(authority)
    if authority_errors:
        raise RefinedDetectionCropSourceError(
            "Invalid refined authority provenance: " + "; ".join(authority_errors)
        )
    intended_use = authority["payload"]["intended_use"]
    if intended_use not in {"analysis", "analysis_and_training"}:
        raise RefinedDetectionCropSourceError(
            "Selected refined authority is not approved for analysis use."
        )
    return selected.strip(), "approved_authoritative_refined_v1", authority


def _automatic_parent_evidence(
    parent: Any,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    lineage = manifest["payload"]["snapshot_lineage"]
    parent_ref = lineage["parent_snapshot"]
    if parent_ref is None:
        return None, None
    parent_run_id = parent_ref["run_id"]
    if parent_run_id not in parent:
        return None, None
    parent_run = parent[parent_run_id]
    parent_manifest = parent_run.attrs.get(REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(parent_manifest, Mapping):
        return None, None
    parent_dimensions = refined_detection_dimensions_from_manifest(parent_manifest)
    return parent_manifest, _run_arrays(
        parent_run,
        dimensions=parent_dimensions,
    )


def bind_refined_detection_crop_source(
    archive_path: Path,
    *,
    run_id: str | None = None,
    allow_selector_ineligible_benchmark: bool = False,
    parent_manifest: Mapping[str, Any] | None = None,
    parent_arrays: Mapping[str, Any] | None = None,
) -> BoundRefinedDetectionCropSource:
    """Open and prove one refined-v1 source before any crop write is planned."""

    path = archive_path.expanduser().resolve()
    if not path.is_dir() or path.suffix != ".zarr":
        raise RefinedDetectionCropSourceError(
            f"Refined crop source is not a Zarr directory: {path}"
        )
    root = open_zarr_root(path, mode="r")
    parent = root.get("refined_detect_runs")
    if parent is None:
        raise RefinedDetectionCropSourceError("Archive has no refined_detect_runs.")
    selected, selection_mode, authority = _resolve_selected_run(
        parent,
        run_id=run_id,
        allow_selector_ineligible_benchmark=allow_selector_ineligible_benchmark,
    )
    if selected not in parent:
        raise RefinedDetectionCropSourceError(
            f"Refined detection run {selected!r} does not exist."
        )
    run = parent[selected]
    manifest = run.attrs.get(REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise RefinedDetectionCropSourceError(
            "Refined detection run lacks its exact run_manifest."
        )
    dimensions = refined_detection_dimensions_from_manifest(manifest)
    if (
        dimensions.lineage_profile
        is not RefinedDetectionLineageProfile.FULL_ACQUISITION
    ):
        raise RefinedDetectionCropSourceError(
            "Ordinary crop handoff requires a full-acquisition refined snapshot."
        )
    payload = manifest["payload"]
    if payload["run_id"] != selected:
        raise RefinedDetectionCropSourceError(
            "Selected run name differs from the manifest run_id."
        )
    eligible = payload["publication"]["stage_selector_eligible"]
    if run.attrs.get("status") != "complete":
        raise RefinedDetectionCropSourceError(
            "Refined crop source is not explicitly complete."
        )
    if run.attrs.get("stage_selector_eligible") is not eligible:
        raise RefinedDetectionCropSourceError(
            "Run and manifest selector-eligibility declarations differ."
        )
    if not allow_selector_ineligible_benchmark and eligible is not True:
        raise RefinedDetectionCropSourceError(
            "Production crop planning requires a selector-eligible refined run."
        )
    if authority is not None:
        authority_payload = authority["payload"]
        if (
            authority_payload["run_id"] != selected
            or authority_payload["run_manifest_digest"] != manifest["payload_digest"]
        ):
            raise RefinedDetectionCropSourceError(
                "Authoritative pointer provenance does not bind the selected manifest."
            )

    arrays = _run_arrays(run, dimensions=dimensions)
    profile = storage_profile_from_manifest(payload["storage_plan"]["storage_profile"])
    plans = plan_refined_detection_storage(dimensions, profile=profile)
    try:
        direct, consolidated = refined_detection_metadata_declaration_maps(
            path,
            run_id=selected,
            plans=plans,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RefinedDetectionCropSourceError(
            f"Cannot reconstruct direct/consolidated refined metadata: {exc}"
        ) from exc

    if (parent_manifest is None) != (parent_arrays is None):
        raise RefinedDetectionCropSourceError(
            "Parent manifest and arrays must be supplied together."
        )
    if parent_manifest is None:
        parent_manifest, parent_arrays = _automatic_parent_evidence(parent, manifest)
    publication_errors = validate_refined_detection_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
        parent_manifest=parent_manifest,
        parent_arrays=parent_arrays,
    )
    if publication_errors:
        raise RefinedDetectionCropSourceError(
            "Refined crop source publication is invalid: "
            + "; ".join(publication_errors)
        )

    recording_identity = payload["snapshot_lineage"]["manual_instance_key_allocator"][
        "recording_identity"
    ]
    root_recording_identity = root.attrs.get("recording_id")
    if (
        root_recording_identity is not None
        and str(root_recording_identity) != recording_identity
    ):
        raise RefinedDetectionCropSourceError(
            "Archive recording_id differs from refined snapshot identity."
        )
    logical_digest = refined_detection_logical_content_digest(
        arrays,
        dimensions=dimensions,
    )
    instances = run["instances"]
    evidence_paths = (
        "instance_key",
        "refined_row_ids",
        "frame_indices",
        "source_acquisition_frame_index",
        "bbox_norm_coords",
        "bbox_img_xyxy",
        "centers_img_xy",
    )
    handoff = {
        "schema_id": REFINED_DETECTION_CROP_SOURCE_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_CROP_SOURCE_SCHEMA_VERSION,
        "selection_mode": selection_mode,
        "run_id": selected,
        "run_path": f"refined_detect_runs/{selected}",
        "instances_path": f"refined_detect_runs/{selected}/instances",
        "run_manifest_digest": manifest["payload_digest"],
        "logical_content_digest": logical_digest,
        "recording_identity": recording_identity,
        "dimensions": dimensions.as_manifest(),
        "row_identity": "instance_key",
        "array_content_sha256": {
            name: sha256_array(instances[name]) for name in evidence_paths
        },
        "coordinate_status": REFINED_DETECTION_CROP_COORDINATE_STATUS,
        "crop_publication_authorized": False,
    }
    return BoundRefinedDetectionCropSource(
        archive_path=path,
        run_id=selected,
        run_path=f"refined_detect_runs/{selected}",
        instances_path=f"refined_detect_runs/{selected}/instances",
        selection_mode=selection_mode,
        manifest=manifest,
        dimensions=dimensions,
        arrays=arrays,
        run_group=run,
        instances_group=instances,
        logical_content_digest=logical_digest,
        handoff_manifest=handoff,
    )


__all__ = [
    "REFINED_DETECTION_CROP_COORDINATE_STATUS",
    "REFINED_DETECTION_CROP_SOURCE_SCHEMA_ID",
    "REFINED_DETECTION_CROP_SOURCE_SCHEMA_VERSION",
    "BoundRefinedDetectionCropSource",
    "RefinedDetectionCropSourceError",
    "bind_refined_detection_crop_source",
]
