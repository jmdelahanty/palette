"""Fresh selector-ineligible publisher for refined-detection v1 shadows.

This module is deliberately unable to write into a recording archive. It only
creates a new standalone Zarr store below an explicit safe shadow root, never
updates a selector or registry, and validates the complete publication before
returning success. Shadows are integration artifacts, not profile-promotion
evidence or production authorities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from fisheye.shared.zarr.canonical_detection_shadow import (
    CanonicalDetectionShadowPublication,
    validate_canonical_detection_shadow_publication,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    publish_selector_ineligible_refined_detection_snapshot,
)
from fisheye.shared.zarr.refined_detection_transition import (
    RefinedDetectionTransitionResult,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)


DEFAULT_REFINED_DETECTION_SHADOW_ROOT = Path("/tmp/palette-refined-detection-shadows")
SHADOW_RECEIPT_SCHEMA_ID = "palette.refined_detection.shadow_publication"
SHADOW_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RefinedDetectionShadowPublication:
    """Completed standalone shadow and its validated evidence."""

    output_path: Path
    run_id: str
    manifest: Mapping[str, object]
    receipt: Mapping[str, object]


def require_safe_refined_detection_shadow_destination(
    destination: Path,
    *,
    shadow_root: Path = DEFAULT_REFINED_DETECTION_SHADOW_ROOT,
) -> Path:
    """Require a fresh child path in a shadow/benchmark namespace."""

    path = destination.expanduser().resolve()
    root = shadow_root.expanduser().resolve()
    temporary_root = Path("/tmp").resolve()
    root_is_safe = root.is_relative_to(temporary_root) or (
        ".palette_benchmarks" in root.parts
    )
    if not root_is_safe:
        raise ValueError(
            "Shadow roots must be below /tmp or a .palette_benchmarks namespace."
        )
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Shadow destination must be a child of {root}.")
    if path.suffix != ".zarr":
        raise ValueError("Shadow destination must use a .zarr suffix.")
    if path.exists():
        raise FileExistsError(f"Shadow destination already exists: {path}")
    if any(part.endswith("_analysis.zarr") for part in path.parts[:-1]):
        raise ValueError("Shadow publication cannot be nested in a recording archive.")
    return path


def _validate_transition_source_matches_canonical(
    transition: RefinedDetectionTransitionResult,
    canonical_source: CanonicalDetectionShadowPublication,
) -> tuple[str, ...]:
    """Prove the refined source-audit projection binds one canonical rowset."""

    errors: list[str] = []
    if transition.dimensions.n_frames != canonical_source.dimensions.n_frames:
        errors.append("refined and canonical source frame counts differ")
    if (
        transition.dimensions.n_source_detections
        != canonical_source.dimensions.n_instances
    ):
        errors.append("refined and canonical source row counts differ")
    comparisons = (
        ("frame_indices", "frame_indices"),
        ("source_acquisition_frame_index", "source_acquisition_frame_index"),
        ("instance_key", "instance_key"),
        ("bbox_norm_coords", "bbox_norm_coords"),
        ("bbox_img_xyxy", "bbox_img_xyxy"),
        ("centers_img_xy", "centers_img_xy"),
        ("scores", "scores"),
        ("class_ids", "class_ids"),
        ("frame_row_offsets", "frame_row_offsets"),
    )
    for refined_name, canonical_name in comparisons:
        refined_path = f"source_detections/{refined_name}"
        canonical_path = f"instances/{canonical_name}"
        if refined_path not in transition.arrays:
            errors.append(f"refined source evidence lacks {refined_path!r}")
            continue
        if canonical_path not in canonical_source.arrays:
            errors.append(f"canonical source evidence lacks {canonical_path!r}")
            continue
        refined_values = np.asarray(transition.arrays[refined_path])
        canonical_values = np.asarray(canonical_source.arrays[canonical_path][...])
        if not np.array_equal(refined_values, canonical_values):
            errors.append(
                f"refined source evidence differs from canonical {canonical_path!r}"
            )
    source_rows = np.asarray(
        transition.arrays.get("source_detections/source_detect_row_index", []),
        dtype=np.int64,
    )
    if not np.array_equal(
        source_rows,
        np.arange(canonical_source.dimensions.n_instances, dtype=np.int64),
    ):
        errors.append("refined source row identities are not canonical row positions")
    return tuple(dict.fromkeys(errors))


def publish_refined_detection_shadow(
    transition: RefinedDetectionTransitionResult,
    *,
    destination: Path,
    run_id: str,
    lineage: RefinedDetectionSnapshotLineage,
    canonical_source: CanonicalDetectionShadowPublication,
    shadow_root: Path = DEFAULT_REFINED_DETECTION_SHADOW_ROOT,
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
) -> RefinedDetectionShadowPublication:
    """Write and fully validate one standalone full-acquisition shadow."""

    output_path = require_safe_refined_detection_shadow_destination(
        destination,
        shadow_root=shadow_root,
    )
    if (
        transition.dimensions.lineage_profile
        is not RefinedDetectionLineageProfile.FULL_ACQUISITION
    ):
        raise ValueError(
            "The first shadow publisher supports only full-acquisition transitions."
        )
    if transition.report.get("status") != "contract_ready":
        raise ValueError("Shadow publication requires a contract-ready transition.")
    if transition.report.get("selector_eligible") is not False:
        raise ValueError("Transition report must remain selector-ineligible.")
    REFINED_DETECTION_SCHEMA_V1.require(
        transition.arrays,
        dimensions=transition.dimensions,
    )
    canonical_errors = validate_canonical_detection_shadow_publication(canonical_source)
    if canonical_errors:
        raise ValueError(
            "Canonical source shadow is invalid: " + "; ".join(canonical_errors)
        )
    source_errors = _validate_transition_source_matches_canonical(
        transition,
        canonical_source,
    )
    if source_errors:
        raise ValueError(
            "Refined source audit does not match canonical evidence: "
            + "; ".join(source_errors)
        )
    source = canonical_source.refined_source_identity()
    publication = publish_selector_ineligible_refined_detection_snapshot(
        dimensions=transition.dimensions,
        arrays=transition.arrays,
        instance_reason_codes=transition.instance_reason_codes,
        source_reason_codes=transition.source_reason_codes,
        destination=output_path,
        run_id=run_id,
        lineage=lineage,
        source=source,
        created_by="refined_detection_shadow",
        publication_kind="canonical_transition_shadow",
        safe_root=shadow_root,
        profile=profile,
        run_attributes={
            "shadow_only": True,
            "transition_report": dict(transition.report),
        },
        selection_contract="none_shadow_direct_path_only",
    )
    receipt = {
        **publication.receipt,
        "schema_id": SHADOW_RECEIPT_SCHEMA_ID,
        "schema_version": SHADOW_RECEIPT_SCHEMA_VERSION,
        "source_manifest_digest": source.run_manifest_digest,
        "source_shadow_path": str(canonical_source.output_path),
    }
    with (output_path / "shadow_publication_receipt.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            receipt,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    return RefinedDetectionShadowPublication(
        output_path=output_path,
        run_id=str(run_id),
        manifest=publication.manifest,
        receipt=receipt,
    )


__all__ = [
    "DEFAULT_REFINED_DETECTION_SHADOW_ROOT",
    "SHADOW_RECEIPT_SCHEMA_ID",
    "SHADOW_RECEIPT_SCHEMA_VERSION",
    "RefinedDetectionShadowPublication",
    "publish_refined_detection_shadow",
    "require_safe_refined_detection_shadow_destination",
]
