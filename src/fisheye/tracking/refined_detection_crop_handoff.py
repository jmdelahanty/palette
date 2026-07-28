"""Plan immutable crop materialization from an exact refined-v1 snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.shared.keyed_delta import KeyedDeltaPlan
from fisheye.shared.zarr.refined_detection_crop_source import (
    REFINED_DETECTION_CROP_COORDINATE_STATUS,
    BoundRefinedDetectionCropSource,
)
from fisheye.tracking.incremental_crop import (
    DEFAULT_SIGNATURE_BATCH_ROWS,
    CropSourceSnapshot,
    IncrementalCropError,
    build_incremental_crop_plan,
    capture_crop_source_snapshot,
)


REFINED_DETECTION_CROP_HANDOFF_PLAN_SCHEMA_ID = (
    "palette.refined_detection.crop_handoff_plan"
)
REFINED_DETECTION_CROP_HANDOFF_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RefinedDetectionCropHandoffPlan:
    """Verified refined source plus its keyed crop copy/compute decision."""

    source_snapshot: CropSourceSnapshot
    keyed_plan: KeyedDeltaPlan
    receipt: Mapping[str, Any]


def plan_refined_detection_crop_handoff(
    source: BoundRefinedDetectionCropSource,
    *,
    source_pixel_fingerprint: str,
    roi_size: tuple[int, int],
    base_crop_group: Any | None = None,
    signature_batch_rows: int = DEFAULT_SIGNATURE_BATCH_ROWS,
) -> RefinedDetectionCropHandoffPlan:
    """Plan row reuse without authorizing a crop publication."""

    if type(source) is not BoundRefinedDetectionCropSource:
        raise TypeError("source must be an exact bound refined crop source.")
    frame_shape = (
        int(source.dimensions.source_height),
        int(source.dimensions.source_width),
    )
    snapshot = capture_crop_source_snapshot(
        source.instances_group,
        source_path=source.instances_path,
        source_pixel_fingerprint=source_pixel_fingerprint,
        frame_shape=frame_shape,
        roi_size=roi_size,
        signature_batch_rows=signature_batch_rows,
    )
    expected_row_ids = np.asarray(
        source.instances_group["refined_row_ids"][:],
        dtype=np.int64,
    )
    actual_row_ids = snapshot.optional_row_arrays.get("source_refined_row_ids")
    if actual_row_ids is None or not np.array_equal(actual_row_ids, expected_row_ids):
        raise IncrementalCropError(
            "Refined crop handoff did not preserve exact refined_row_ids lineage."
        )
    plan = build_incremental_crop_plan(
        snapshot,
        base_group=base_crop_group,
        roi_size=roi_size,
    )
    receipt = {
        "schema_id": REFINED_DETECTION_CROP_HANDOFF_PLAN_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_CROP_HANDOFF_PLAN_SCHEMA_VERSION,
        "source": dict(source.handoff_manifest),
        "source_pixel_fingerprint": str(source_pixel_fingerprint),
        "roi_size": [int(roi_size[0]), int(roi_size[1])],
        "row_count": snapshot.row_count,
        "row_signature_spec_digest": snapshot.signature_spec.spec_digest,
        "keyed_plan": plan.summary(),
        "coordinate_status": REFINED_DETECTION_CROP_COORDINATE_STATUS,
        "crop_publication_authorized": False,
    }
    return RefinedDetectionCropHandoffPlan(
        source_snapshot=snapshot,
        keyed_plan=plan,
        receipt=receipt,
    )


__all__ = [
    "REFINED_DETECTION_CROP_HANDOFF_PLAN_SCHEMA_ID",
    "REFINED_DETECTION_CROP_HANDOFF_PLAN_SCHEMA_VERSION",
    "RefinedDetectionCropHandoffPlan",
    "plan_refined_detection_crop_handoff",
]
