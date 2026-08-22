"""Run a trained U-Net segmenter to produce unified subject-mask probabilities."""

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import os
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from dataclasses import dataclass
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import torch
import zarr
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from ..pose.schema import resolve_required_keypoint_indices_from_attrs
from ..shared.crop_image_source import CropImageSource
from ..shared.composite_subject_mask import assert_subject_mask_run_unreferenced
from ..shared.artifact_fingerprint import require_artifact_content_identity
from ..shared.inference_timing import InferenceTimingProfiler
from ..shared.model_input_transform import (
    MODEL_INPUT_TRANSFORM_CHOICES,
    ModelInputTransform,
    resolve_model_input_transform,
)
from ..shared.provenance_attrs import (
    build_assignment_keypoint_attrs,
    build_source_crop_snapshot_attrs,
    build_source_roi_pixel_attrs,
)
from ..shared.row_alignment import assert_row_alignment
from ..shared.row_lineage import (
    copy_row_lineage_arrays,
    copy_selected_crop_row_lineage_arrays,
    write_direct_source_crop_row_ids,
)
from ..shared.row_source_signature import copy_selected_row_source_signatures
from ..shared.run_provenance import (
    append_input_artifacts,
    build_run_provenance_from_stage_record,
)
from ..shared.subject_mask_registry_status import emit_subject_mask_stage_completion
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import (
    DEFAULT_MASK_PROBS_SHARD_ROIS,
    subject_mask_metric_row_chunk,
    subject_mask_storage_chunks,
)
from ..shared.subject_mask_attempt import (
    build_subject_mask_attempt,
    build_subject_mask_scientific_identity,
    validate_subject_mask_attempt,
    validate_subject_mask_scientific_identity,
)
from ..shared.subject_mask_worker_receipt import (
    RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    build_subject_mask_worker_semantic_receipt,
)
from ..shared.subject_mask_component_provenance import (
    write_subject_mask_component_provenance,
)
from ..shared.subject_mask_coordinate_publication import (
    SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
    SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
    SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
    _activate_validated_subject_mask_coordinate_surfaces,
    capture_subject_mask_coordinate_publication_checkpoint,
    load_persisted_subject_mask_crop_source,
    prepare_subject_mask_coordinate_context,
    publish_subject_mask_coordinate_surfaces,
    require_direct_subject_mask_crop_pixel_source,
    rollback_subject_mask_coordinate_publication,
    selected_subject_mask_crop_values,
)
from ..shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from ..shared.zarr.manifest_digest import canonical_json_bytes
from ..shared.zarr.training_crop_materialization import (
    bind_training_crop_materialization,
)
from ..shared.zarr.subject_mask_validation_receipt import (
    SubjectMaskArrayUnitAccumulator,
    streaming_array_sha256,
    subject_mask_array_unit_document,
)
from ..registry.db import Registry, RegistryPaths
from ..registry.model_resolution import (
    build_resolution_payload,
    resolve_best_subject_mask_model,
)
from ..shared.system_metadata import get_environment_info, get_git_info
from .unet import UNetSmall

SUBJECT_MASK_SCHEMAS: dict[str, tuple[str, ...]] = {
    "subject_v1_union": ("subject_body", "eyes_union", "swim_bladder"),
    "subject_v1_lr": ("subject_body", "eye_left", "eye_right", "swim_bladder"),
}
SUBJECT_MASK_LABEL_SCHEMA = "subject_v1_union"
SUBJECT_MASK_LABELS: tuple[str, ...] = SUBJECT_MASK_SCHEMAS[SUBJECT_MASK_LABEL_SCHEMA]
_SUBJECT_MASKS_STATUS_SOURCE = "runtime_infer_unet_subject_masks"
KEYPOINT_GROUP_CHOICES = ("refined_keypoints_runs", "keypoints_runs")
KEYPOINT_SUCCESS_DATASET_CANDIDATES = (
    "usable_keypoints",
    "detection_success",
    "refined_success",
    "source_success",
)
EYE_KEYPOINT_LABELS = ("eye_left", "eye_right")
SUBJECT_MASK_OUTPUT_PARENTS = ("subject_mask_runs", "subject_mask_shard_runs")
SUBJECT_MASK_CANONICAL_OUTPUT_PARENT = "subject_mask_runs"
SUBJECT_MASK_SHARD_OUTPUT_PARENT = "subject_mask_shard_runs"
ROI_WORK_PACKAGE_ROLE_DELTA = "delta_replacement_rows"
ROI_WORK_PACKAGE_ROLE_COMPLETE_PARTITION = "complete_collection_partition"
ROI_WORK_PACKAGE_ROLES = (
    ROI_WORK_PACKAGE_ROLE_DELTA,
    ROI_WORK_PACKAGE_ROLE_COMPLETE_PARTITION,
)
COLLECTION_PARTITION_CONTRACT_SCHEMA_ID = (
    "palette.subject_mask.complete_collection_partition"
)
COLLECTION_PARTITION_CONTRACT_SCHEMA_VERSION = 1
RECORDING_WORK_UNIT_CONTRACT_SCHEMA_ID = (
    "palette.subject_mask.complete_recording_work_unit"
)
RECORDING_WORK_UNIT_CONTRACT_SCHEMA_VERSION = 1
MASK_PROBS_WORKING_ARRAY = "_mask_probs_roi_working"
MASK_PROBS_CANONICAL_ARRAY = "mask_probs_roi"
MASK_PROBS_SHARDING_SCHEMA = "palette.subject_mask_probability_postpack.v1"
MASK_PROBS_DIRECT_SHARDING_SCHEMA = (
    "palette.subject_mask_probability_double_buffered_shards.v1"
)
MASK_PROBS_DESTINATION_VALIDATION_FULL = "full_decoded_reread_v1"
MASK_PROBS_DESTINATION_VALIDATION_FINAL_LAYOUT = "receipt_bound_final_layout_unit_v1"
MASK_PROBS_DESTINATION_VALIDATION_MODES = (
    MASK_PROBS_DESTINATION_VALIDATION_FULL,
    MASK_PROBS_DESTINATION_VALIDATION_FINAL_LAYOUT,
)
SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR = "subject_mask_scientific_identity"
SUBJECT_MASK_ATTEMPT_ATTR = "subject_mask_attempt"
SUBJECT_MASK_ATTEMPT_LINEAGE_EVIDENCE_ATTR = "subject_mask_attempt_lineage_evidence"
SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_ATTR = (
    "subject_mask_worker_semantic_receipt_binding"
)
SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SIDECAR = "worker_semantic_receipt.json"
_SUBJECT_MASK_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "authoritative_run_provenance",
    SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
    SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
)


def _snapshot_selected_attrs(
    node: Any,
    names: Sequence[str],
) -> dict[str, tuple[bool, Any]]:
    attrs = getattr(node, "attrs", None)
    if attrs is None or not hasattr(attrs, "keys"):
        raise RuntimeError("Cannot snapshot subject-mask selector attrs.")
    return {name: (name in attrs, copy.deepcopy(attrs.get(name))) for name in names}


def _restore_owned_subject_mask_selectors(
    parent: Any,
    snapshot: dict[str, tuple[bool, Any]],
    *,
    run_name: str | None,
    owner_token: str | None,
) -> None:
    """Restore only selector values still visibly owned by this attempt."""

    if run_name is None or owner_token is None:
        return
    attrs = parent.attrs
    failures: list[str] = []
    lease = attrs.get(SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR)
    lease_owned = (
        isinstance(lease, dict)
        and lease.get("run_path") == f"subject_mask_runs/{run_name}"
        and lease.get("publication_owner") == owner_token
    )
    lease_present, lease_snapshot = snapshot[SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR]
    if (
        lease is not None
        and not lease_owned
        and (not lease_present or lease != lease_snapshot)
    ):
        raise RuntimeError(
            "Subject-mask parent publication lease was replaced by a foreign "
            "attempt; refusing silent selector rollback."
        )
    for name in ("latest", "latest_complete", "latest_pending"):
        try:
            if attrs.get(name) != run_name:
                continue
            present, value = snapshot[name]
            if present:
                attrs[name] = copy.deepcopy(value)
            else:
                del attrs[name]
            if present and attrs.get(name) != value:
                raise RuntimeError("restored value differs from snapshot")
            if not present and name in attrs:
                raise RuntimeError("owned selector survived rollback")
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{name}: {exc}")
    if lease_owned:
        for name in (
            SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
            SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
            SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        ):
            try:
                present, value = snapshot[name]
                if present:
                    attrs[name] = copy.deepcopy(value)
                elif name in attrs:
                    del attrs[name]
                if present and attrs.get(name) != value:
                    raise RuntimeError("restored value differs from snapshot")
                if not present and name in attrs:
                    raise RuntimeError("owned publication state survived rollback")
            except BaseException as exc:  # pragma: no cover - hostile store
                failures.append(f"{name}: {exc}")
    if failures:
        raise RuntimeError(
            f"Subject-mask owned selector rollback was incomplete: {failures!r}."
        )


class _SubjectMaskAttemptFailureBoundary:
    """Fail one newly created subject-mask attempt closed across the writer."""

    def __init__(self) -> None:
        self.root: Any | None = None
        self.parent: Any | None = None
        self.run: Any | None = None
        self.run_name: str | None = None
        self.run_path: str | None = None
        self.parent_selector_snapshot: dict[str, tuple[bool, Any]] | None = None
        self.crop_source: Any | None = None
        self.coordinate_checkpoint: Any | None = None
        self.owner_token: str | None = None
        self.finalized = False

    def prepare(self, *, root: Any, parent: Any) -> None:
        if self.root is not None or self.parent is not None:
            raise RuntimeError(
                "A subject-mask attempt cannot bind more than one run parent."
            )
        self.root = root
        self.parent = parent
        self.parent_selector_snapshot = _snapshot_selected_attrs(
            parent,
            _SUBJECT_MASK_SELECTOR_ATTRS,
        )
        self.owner_token = uuid4().hex

    def bind_run(self, run: Any, run_name: str) -> None:
        if self.run is not None:
            raise RuntimeError("A subject-mask attempt cannot bind more than one run.")
        if (
            self.owner_token is None
            or run.attrs.get(SUBJECT_MASK_PUBLICATION_OWNER_ATTR) != self.owner_token
        ):
            raise RuntimeError(
                "Subject-mask child did not persist its atomic publication owner."
            )
        self.run = run
        self.run_name = str(run_name)
        self.run_path = str(getattr(run, "path", "")).strip("/") or None

    def bind_crop_source(self, crop_source: Any) -> None:
        self.crop_source = crop_source

    def close_crop_source(self) -> None:
        if self.crop_source is None:
            return
        self.crop_source.close()
        self.crop_source = None

    def bind_coordinate_checkpoint(self, checkpoint: Any) -> None:
        self.coordinate_checkpoint = checkpoint

    def fresh_parent(self) -> Any:
        if self.parent is None:
            raise RuntimeError("Subject-mask publication parent is not bound.")
        path = str(getattr(self.parent, "path", "")).strip("/")
        if self.root is not None and path:
            try:
                parent = self.root[path]
            except BaseException as exc:
                raise RuntimeError(
                    "Subject-mask publication parent disappeared."
                ) from exc
            self.parent = parent
        return self.parent

    def require_owned_run(self) -> Any:
        if self.parent is None or self.run_name is None or self.owner_token is None:
            raise RuntimeError("Subject-mask publication ownership is not bound.")
        try:
            run = self.fresh_parent()[self.run_name]
        except BaseException as exc:
            raise RuntimeError("Subject-mask publication child disappeared.") from exc
        if run.attrs.get(SUBJECT_MASK_PUBLICATION_OWNER_ATTR) != self.owner_token:
            raise RuntimeError(
                "Subject-mask publication child was replaced by another owner."
            )
        self.run = run
        return run

    def mark_finalized(self) -> None:
        self.finalized = True
        self.coordinate_checkpoint = None

    def fail(self, original: BaseException) -> None:
        failures: list[str] = []
        if self.run is not None and not self.finalized:
            try:
                run_for_failure = self.require_owned_run()
            except BaseException:
                # The child disappeared or now belongs to a different attempt.
                # Touch neither that replacement nor selectors that now resolve
                # to it; this is concurrent state, not a rollback target.
                run_for_failure = None
            if run_for_failure is not None and self.coordinate_checkpoint is not None:
                if run_for_failure.attrs.get("stage_selector_eligible") is not True:
                    try:
                        rollback_subject_mask_coordinate_publication(
                            self.coordinate_checkpoint
                        )
                    except BaseException as exc:  # pragma: no cover - hostile store
                        failures.append(f"coordinate publication: {exc}")
            publication_committed = (
                run_for_failure is not None
                and run_for_failure.attrs.get("stage_selector_eligible") is True
            )
            if run_for_failure is not None and not publication_committed:
                try:
                    run_for_failure.attrs["stage_selector_eligible"] = False
                    mark_run_failed(
                        run_for_failure,
                        parent_group=None,
                        run_name=self.run_name,
                        error=f"subject-mask writer failed: {original}",
                    )
                    run_for_failure.attrs["stage_selector_eligible"] = False
                except BaseException as exc:  # pragma: no cover - hostile store
                    failures.append(f"run completion: {exc}")
            if (
                run_for_failure is not None
                and not publication_committed
                and self.parent_selector_snapshot is not None
            ):
                try:
                    _restore_owned_subject_mask_selectors(
                        self.fresh_parent(),
                        self.parent_selector_snapshot,
                        run_name=self.run_name,
                        owner_token=self.owner_token,
                    )
                except BaseException as exc:  # pragma: no cover - hostile store
                    failures.append(f"owned parent selectors: {exc}")
        if self.crop_source is not None:
            try:
                self.close_crop_source()
            except BaseException as exc:  # pragma: no cover - hostile source
                failures.append(f"crop source close: {exc}")
        if failures:
            raise RuntimeError(
                "Subject-mask attempt failed and fail-closed rollback was incomplete: "
                f"{failures!r}."
            ) from original


_ACTIVE_SUBJECT_MASK_ATTEMPT: ContextVar[_SubjectMaskAttemptFailureBoundary | None] = (
    ContextVar("active_subject_mask_attempt", default=None)
)


def _fail_closed_subject_mask_attempt(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        boundary = _SubjectMaskAttemptFailureBoundary()
        token = _ACTIVE_SUBJECT_MASK_ATTEMPT.set(boundary)
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            boundary.fail(exc)
            raise
        finally:
            _ACTIVE_SUBJECT_MASK_ATTEMPT.reset(token)

    return wrapped


@dataclass(frozen=True)
class _SubjectMaskOutputBatch:
    start: int
    stop: int
    probs_out: np.ndarray
    binary: Optional[np.ndarray]
    metrics: dict[str, np.ndarray]


def _raw_worker_validation_accumulators(
    *,
    total_rois: int,
    n_channels: int,
    height: int,
    width: int,
    probability_dtype: np.dtype[Any],
    write_masks_roi: bool,
    unit_rows: int,
) -> dict[str, SubjectMaskArrayUnitAccumulator]:
    shapes_dtypes: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "mask_probs_roi": (
            (total_rois, n_channels, height, width),
            np.dtype(probability_dtype),
        ),
        "metrics/prob_max": ((total_rois, n_channels), np.dtype(np.float32)),
        "metrics/mask_present": ((total_rois, n_channels), np.dtype(bool)),
        "metrics/area_px": ((total_rois, n_channels), np.dtype(np.float32)),
        "metrics/centroid_xy": (
            (total_rois, n_channels, 2),
            np.dtype(np.float32),
        ),
        "metrics/centroid_valid": ((total_rois, n_channels), np.dtype(bool)),
        "metrics/bbox_xyxy": (
            (total_rois, n_channels, 4),
            np.dtype(np.float32),
        ),
        "metrics/bbox_valid": ((total_rois, n_channels), np.dtype(bool)),
    }
    if write_masks_roi:
        shapes_dtypes["masks_roi"] = (
            (total_rois, n_channels, height, width),
            np.dtype(np.uint8),
        )
    return {
        path: SubjectMaskArrayUnitAccumulator(
            shape=shape,
            dtype=dtype,
            unit_rows=int(unit_rows),
        )
        for path, (shape, dtype) in shapes_dtypes.items()
    }


def _append_raw_worker_validation_batch(
    accumulators: Mapping[str, SubjectMaskArrayUnitAccumulator],
    batch: _SubjectMaskOutputBatch,
) -> None:
    values: dict[str, np.ndarray] = {
        "mask_probs_roi": batch.probs_out,
        **{f"metrics/{name}": value for name, value in batch.metrics.items()},
    }
    if "masks_roi" in accumulators:
        if batch.binary is None:
            raise ValueError("Raw worker receipt expected materialized binary masks.")
        values["masks_roi"] = batch.binary
    if set(values) != set(accumulators):
        raise ValueError("Raw worker receipt output inventory changed.")
    for path, accumulator in accumulators.items():
        accumulator.append(int(batch.start), values[path])


def _freeze_subject_mask_output_batch(batch: _SubjectMaskOutputBatch) -> None:
    """Make a queued CPU batch immutable until the output worker releases it."""

    arrays = [batch.probs_out, *batch.metrics.values()]
    if batch.binary is not None:
        arrays.append(batch.binary)
    for values in arrays:
        values.setflags(write=False)


def _seal_raw_worker_semantic_receipt(
    *,
    run_group: zarr.Group,
    run_path: str,
    scientific_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
    accumulators: Mapping[str, SubjectMaskArrayUnitAccumulator],
    unit_rows: int,
) -> dict[str, object]:
    output_document = {
        path: accumulator.as_document() for path, accumulator in accumulators.items()
    }
    available_document = subject_mask_array_unit_document(
        {"available_channels": run_group["available_channels"]},
        ("available_channels",),
        unit_rows=max(1, int(unit_rows)),
    )
    array_document = {**output_document, **available_document}
    required_paths = list(RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS)
    if "masks_roi" in output_document:
        required_paths.insert(1, "masks_roi")
    roi_aligned_paths = tuple(
        path for path in required_paths if path != "available_channels"
    )
    payload = scientific_identity.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Raw worker scientific identity payload is absent.")
    return build_subject_mask_worker_semantic_receipt(
        stage_kind="raw_subject_mask",
        run_path=run_path,
        scientific_identity=scientific_identity,
        attempt=attempt,
        scope={
            "crop": payload.get("crop"),
            "pixels": payload.get("pixels"),
            "row_identity": payload.get("row_identity"),
        },
        row_count=int(run_group["mask_probs_roi"].shape[0]),
        array_document=array_document,
        required_paths=required_paths,
        roi_aligned_paths=roi_aligned_paths,
    )


def _compression_kwargs(array: zarr.Array) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    sentinel = object()
    compressors = getattr(array, "compressors", sentinel)
    if compressors is not sentinel:
        if compressors:
            kwargs["compressors"] = compressors
    else:
        try:
            compressor = array.compressor
        except (TypeError, AttributeError):
            compressor = None
        if compressor is not None:
            kwargs["compressor"] = compressor

    chunk_codecs = getattr(array, "chunk_codecs", None)
    if chunk_codecs:
        kwargs.setdefault("chunk_codecs", chunk_codecs)
    filters = getattr(array, "filters", None)
    if filters:
        kwargs.setdefault("filters", filters)
    return kwargs


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device_str)
    if device_str.lower() == "cpu":
        return torch.device("cpu")
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    return torch.device(device_str)


def _normalise_roi_tensor(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim == 3:
        batch = batch.unsqueeze(1)
    if batch.ndim != 4:
        raise ValueError(f"Unexpected ROI batch shape {tuple(batch.shape)}")
    if batch.shape[1] not in (1, 3):
        raise ValueError(f"Unsupported ROI channel count: {int(batch.shape[1])}")

    if not batch.is_floating_point():
        max_value = float(torch.iinfo(batch.dtype).max)
        batch = batch.to(dtype=torch.float32)
        if max_value > 0:
            batch = batch / max_value
    else:
        batch = batch.to(dtype=torch.float32)
        if batch.numel():
            max_val = torch.nan_to_num(
                batch, nan=0.0, posinf=float("inf"), neginf=float("-inf")
            ).amax()
            batch = batch / torch.maximum(
                max_val, torch.tensor(1.0, device=batch.device, dtype=torch.float32)
            )

    batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=0.0)
    batch = torch.clamp(batch, 0.0, 1.0)
    if batch.device.type == "cuda":
        batch = batch.contiguous(memory_format=torch.channels_last)
    return batch


def _probabilities_from_logits(
    logits: torch.Tensor, *, mask_probs_dtype: str
) -> np.ndarray:
    probs = torch.sigmoid(logits)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = torch.clamp(probs, 0.0, 1.0)
    if mask_probs_dtype == "uint8":
        probs = torch.round(probs * 255.0).to(dtype=torch.uint8)
    else:
        probs = probs.to(dtype=torch.float16)
    return probs.cpu().numpy()


def _postprocess_logits_on_device(
    logits: torch.Tensor,
    *,
    mask_probs_dtype: str,
    return_binary: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, np.ndarray]]:
    probs = torch.sigmoid(logits)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = torch.clamp(probs, 0.0, 1.0)
    if probs.ndim == 3:
        probs = probs.unsqueeze(1)

    if mask_probs_dtype == "uint8":
        probs_out = torch.round(probs * 255.0).to(dtype=torch.uint8)
        binary = (probs_out >= 128).to(dtype=torch.uint8)
        probs_for_metrics = probs_out.to(dtype=torch.float32) / 255.0
    else:
        probs_out = probs.to(dtype=torch.float16)
        binary = (probs_out >= 0.5).to(dtype=torch.uint8)
        probs_for_metrics = probs_out.to(dtype=torch.float32)

    area_px = binary.sum(dim=(2, 3), dtype=torch.float32)
    prob_max = probs_for_metrics.amax(dim=(2, 3))
    spatial_metrics = _compute_spatial_metrics_from_binary_tensor(
        binary, area_px=area_px
    )
    probs_out_cpu = probs_out.cpu().numpy()
    if mask_probs_dtype == "uint8":
        # Canonical uint8 metrics use max-then-decode on the CPU. Computing
        # divide-then-max on the GPU can differ by one float32 ULP for the same
        # persisted bytes, which defeats cross-backend logical digests.
        prob_max_cpu = np.max(probs_out_cpu, axis=(2, 3)).astype(
            np.float32, copy=False
        ) / np.float32(255.0)
    else:
        prob_max_cpu = prob_max.cpu().numpy().astype(np.float32, copy=False)
    metrics = {
        "prob_max": prob_max_cpu,
        "mask_present": (area_px > 0.0).cpu().numpy().astype(bool, copy=False),
        "area_px": area_px.cpu().numpy().astype(np.float32, copy=False),
        "centroid_xy": spatial_metrics["centroid_xy"]
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "centroid_valid": spatial_metrics["centroid_valid"]
        .cpu()
        .numpy()
        .astype(bool, copy=False),
        "bbox_xyxy": spatial_metrics["bbox_xyxy"]
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "bbox_valid": spatial_metrics["bbox_valid"]
        .cpu()
        .numpy()
        .astype(bool, copy=False),
    }
    binary_out = binary.cpu().numpy() if return_binary else None
    return probs_out_cpu, binary_out, metrics


def _compute_spatial_metrics_from_binary_tensor(
    binary: torch.Tensor,
    *,
    area_px: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    if binary.ndim != 4:
        raise ValueError("binary must have shape (N,C,H,W).")

    mask = binary > 0
    _row_count, _channel_count, height, width = mask.shape
    if area_px is None:
        area_px = mask.sum(dim=(2, 3), dtype=torch.float32)
    else:
        area_px = area_px.to(dtype=torch.float32)
    valid = area_px > 0.0
    denominator = torch.clamp(area_px, min=1.0)

    y_counts = mask.sum(dim=3, dtype=torch.float32)
    x_counts = mask.sum(dim=2, dtype=torch.float32)
    y_float = torch.arange(height, device=mask.device, dtype=torch.float32).view(
        1, 1, height
    )
    x_float = torch.arange(width, device=mask.device, dtype=torch.float32).view(
        1, 1, width
    )
    y_sum = (y_counts * y_float).sum(dim=2)
    x_sum = (x_counts * x_float).sum(dim=2)

    centroid_xy = torch.stack((x_sum / denominator, y_sum / denominator), dim=2)
    centroid_xy = torch.where(
        valid.unsqueeze(2), centroid_xy, torch.zeros_like(centroid_xy)
    )

    row_has_mask = mask.any(dim=3)
    col_has_mask = mask.any(dim=2)
    y_int = torch.arange(height, device=mask.device, dtype=torch.int32).view(
        1, 1, height
    )
    x_int = torch.arange(width, device=mask.device, dtype=torch.int32).view(1, 1, width)
    y_min = torch.where(
        row_has_mask,
        y_int,
        torch.full((1, 1, height), height, device=mask.device, dtype=torch.int32),
    ).amin(dim=2)
    y_max_exclusive = torch.where(
        row_has_mask,
        y_int + 1,
        torch.zeros((1, 1, height), device=mask.device, dtype=torch.int32),
    ).amax(dim=2)
    x_min = torch.where(
        col_has_mask,
        x_int,
        torch.full((1, 1, width), width, device=mask.device, dtype=torch.int32),
    ).amin(dim=2)
    x_max_exclusive = torch.where(
        col_has_mask,
        x_int + 1,
        torch.zeros((1, 1, width), device=mask.device, dtype=torch.int32),
    ).amax(dim=2)

    bbox_xyxy = torch.stack(
        (x_min, y_min, x_max_exclusive, y_max_exclusive),
        dim=2,
    ).to(dtype=torch.float32)
    bbox_xyxy = torch.where(valid.unsqueeze(2), bbox_xyxy, torch.zeros_like(bbox_xyxy))
    return {
        "centroid_xy": centroid_xy,
        "centroid_valid": valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": valid,
    }


def _compute_channel_metrics(
    binary_masks: np.ndarray, probs: np.ndarray
) -> dict[str, np.ndarray]:
    binary = np.asarray(binary_masks, dtype=np.uint8)
    prob_arr = np.asarray(probs, dtype=np.float32)
    if binary.ndim != 3 or prob_arr.shape != binary.shape:
        raise ValueError("binary_masks and probs must both have shape (N,H,W).")

    row_count = int(binary.shape[0])
    area_px = binary.sum(axis=(1, 2), dtype=np.int64).astype(np.float32)
    mask_present = area_px > 0.0
    prob_max = (
        prob_arr.max(axis=(1, 2)).astype(np.float32, copy=False)
        if row_count
        else np.zeros((0,), dtype=np.float32)
    )
    spatial_metrics = _compute_channel_spatial_metrics(binary)

    return {
        "prob_max": prob_max,
        "mask_present": mask_present.astype(bool, copy=False),
        "area_px": area_px,
        **spatial_metrics,
    }


def _compute_channel_spatial_metrics(binary_masks: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(binary_masks, dtype=np.uint8)
    if binary.ndim != 3:
        raise ValueError("binary_masks must have shape (N,H,W).")

    row_count = int(binary.shape[0])

    centroid_xy = np.zeros((row_count, 2), dtype=np.float32)
    centroid_valid = np.zeros((row_count,), dtype=bool)
    bbox_xyxy = np.zeros((row_count, 4), dtype=np.float32)
    bbox_valid = np.zeros((row_count,), dtype=bool)

    for row_idx in range(row_count):
        ys, xs = np.nonzero(binary[row_idx] > 0)
        if ys.size == 0:
            continue
        centroid_xy[row_idx, 0] = float(xs.mean())
        centroid_xy[row_idx, 1] = float(ys.mean())
        centroid_valid[row_idx] = True
        bbox_xyxy[row_idx] = np.asarray(
            [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ],
            dtype=np.float32,
        )
        bbox_valid[row_idx] = True

    return {
        "centroid_xy": centroid_xy,
        "centroid_valid": centroid_valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": bbox_valid,
    }


def _load_checkpoint(
    path: Path, device: torch.device
) -> Tuple[UNetSmall, Dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    model_cfg = checkpoint.get("model_config")
    if not model_cfg:
        raise ValueError(
            "Checkpoint missing 'model_config'; retrain with updated trainer."
        )
    model = UNetSmall(**model_cfg)
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'model_state'.")
    state_dict = _normalize_checkpoint_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def _normalize_checkpoint_state_dict(state_dict: object) -> object:
    """Normalize checkpoint keys saved from torch.compile-wrapped modules."""

    if not isinstance(state_dict, dict):
        return state_dict
    prefix = "_orig_mod."
    if not any(str(key).startswith(prefix) for key in state_dict):
        return state_dict
    normalized: dict[object, object] = {}
    for key, value in state_dict.items():
        key_text = str(key)
        normalized_key = key_text[len(prefix) :] if key_text.startswith(prefix) else key
        if normalized_key in normalized:
            raise ValueError(
                f"Checkpoint state_dict key collision after removing {prefix!r}."
            )
        normalized[normalized_key] = value
    return normalized


def _resolve_checkpoint_schema(
    checkpoint: Dict[str, object],
) -> Tuple[str, Tuple[str, ...]]:
    label_schema_id = (
        str(checkpoint.get("label_schema_id") or "").strip()
        or SUBJECT_MASK_LABEL_SCHEMA
    )
    expected_labels = SUBJECT_MASK_SCHEMAS.get(label_schema_id)
    if expected_labels is None:
        raise ValueError(
            f"Unsupported subject-mask checkpoint label_schema_id {label_schema_id!r}. "
            f"Expected one of {sorted(SUBJECT_MASK_SCHEMAS)}."
        )

    mask_labels_raw = checkpoint.get("mask_labels")
    if isinstance(mask_labels_raw, (list, tuple)) and mask_labels_raw:
        mask_labels = tuple(str(item) for item in mask_labels_raw)
    else:
        mask_labels = expected_labels
    if mask_labels != expected_labels:
        raise ValueError(
            f"Checkpoint mask_labels {mask_labels!r} do not match schema "
            f"{label_schema_id!r}: {expected_labels!r}."
        )
    return label_schema_id, mask_labels


def _resolve_assignment_keypoint_attrs(
    root: zarr.Group,
    *,
    assignment_keypoint_group: Optional[str],
    assignment_keypoints_run: Optional[str],
    total_rois: int,
    mask_labels: Sequence[str],
) -> dict[str, object]:
    has_group = bool(assignment_keypoint_group)
    has_run = bool(assignment_keypoints_run)
    if has_group != has_run:
        raise ValueError(
            "Pass both --assignment-keypoint-group and --assignment-keypoint-run, or neither."
        )
    if not has_group or not has_run:
        return {}
    if "eyes_union" not in {str(label) for label in mask_labels}:
        raise ValueError(
            "Assignment keypoints are only valid for subject-mask schemas that expose eyes_union."
        )

    group_name = str(assignment_keypoint_group)
    parent = root.get(group_name)
    run_name = str(assignment_keypoints_run)
    if parent is None or run_name not in parent:
        raise ValueError(
            f"Assignment keypoint source not found: {group_name}/{run_name}."
        )
    kp_group = parent[run_name]
    keypoints_roi = kp_group.get("keypoints_roi")
    if keypoints_roi is None:
        raise ValueError(
            f"Assignment keypoint source {group_name}/{run_name} missing keypoints_roi."
        )
    success_name = next(
        (name for name in KEYPOINT_SUCCESS_DATASET_CANDIDATES if name in kp_group), None
    )
    if success_name is None:
        raise ValueError(
            f"Assignment keypoint source {group_name}/{run_name} missing success dataset "
            f"({', '.join(KEYPOINT_SUCCESS_DATASET_CANDIDATES)})."
        )
    success_arr = kp_group.get(success_name)
    assert_row_alignment(
        int(total_rois),
        (
            (f"{group_name}/{run_name}/keypoints_roi", keypoints_roi),
            (f"{group_name}/{run_name}/{success_name}", success_arr),
        ),
        stage="subject-mask assignment keypoint source",
    )
    keypoint_shape = getattr(keypoints_roi, "shape", ())
    keypoint_count = int(keypoint_shape[1]) if len(keypoint_shape) >= 2 else None
    eye_indices = resolve_required_keypoint_indices_from_attrs(
        kp_group.attrs,
        EYE_KEYPOINT_LABELS,
        keypoint_count=keypoint_count,
    )
    payload = build_assignment_keypoint_attrs(
        run_name,
        assignment_keypoint_group=group_name,
        selection="cli_explicit",
    )
    payload["assignment_keypoint_success_dataset"] = str(success_name)
    payload["assignment_keypoint_eye_indices"] = {
        key: int(value) for key, value in eye_indices.items()
    }
    return payload


def _prepare_run_group(
    root: zarr.Group,
    *,
    run_name: Optional[str],
    overwrite: bool,
    output_parent: str = SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
) -> Tuple[zarr.Group, str]:
    if output_parent not in SUBJECT_MASK_OUTPUT_PARENTS:
        raise ValueError(
            f"output_parent must be one of {SUBJECT_MASK_OUTPUT_PARENTS}; got {output_parent!r}."
        )
    parent = require_runs_parent(
        root,
        output_parent,
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    boundary = _ACTIVE_SUBJECT_MASK_ATTEMPT.get()
    if boundary is not None:
        boundary.prepare(root=root, parent=parent)
        assert boundary.parent_selector_snapshot is not None
        selector_snapshot = boundary.parent_selector_snapshot
        assert boundary.owner_token is not None
        owner_token = boundary.owner_token
    else:
        selector_snapshot = _snapshot_selected_attrs(
            parent,
            _SUBJECT_MASK_SELECTOR_ATTRS,
        )
        owner_token = uuid4().hex
    resolved_name = run_name
    if resolved_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"subject_masks_{timestamp}"
        resolved_name = base_name
        suffix = 1
        while resolved_name in parent:
            resolved_name = f"{base_name}_{suffix:03d}"
            suffix += 1
    target_selectors = tuple(
        selector
        for selector in ("latest", "latest_complete", "authoritative_run")
        if parent.attrs.get(selector) == str(resolved_name)
    )
    if resolved_name not in parent and target_selectors:
        raise ValueError(
            f"Refusing to create {output_parent}/{resolved_name}: stale selector(s) "
            f"{target_selectors!r} already name the missing child."
        )
    pending = parent.attrs.get("latest_pending")
    if (
        output_parent == SUBJECT_MASK_CANONICAL_OUTPUT_PARENT
        and pending is not None
        and pending != ""
        and (resolved_name not in parent or pending != str(resolved_name))
    ):
        raise ValueError(
            f"Refusing to start {output_parent}/{resolved_name}: latest_pending "
            f"is already owned by {pending!r}."
        )
    if resolved_name in parent:
        if not overwrite:
            raise ValueError(
                f"{output_parent}/{resolved_name} already exists. Pass --overwrite to replace it."
            )
        existing = parent[resolved_name]
        selected_by = tuple(
            selector
            for selector in (
                "latest",
                "latest_complete",
                "latest_pending",
                "authoritative_run",
            )
            if parent.attrs.get(selector) == str(resolved_name)
        )
        if (
            existing.attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_COMPLETE
            or selected_by
        ):
            reason = (
                "it is complete"
                if not selected_by
                else f"it is selected by {', '.join(selected_by)}"
            )
            raise ValueError(
                f"Refusing to overwrite {output_parent}/{resolved_name}: {reason}."
            )
        assert_subject_mask_run_unreferenced(
            root,
            str(resolved_name),
            base_parent_name=output_parent,
        )
        del parent[resolved_name]
    sentinel_attrs = {
        "stage_selector_eligible": False,
        SUBJECT_MASK_PUBLICATION_OWNER_ATTR: owner_token,
    }
    run_group = parent.create_group(
        resolved_name,
        attributes=sentinel_attrs,
    )
    if (
        run_group.attrs.get("stage_selector_eligible") is not False
        or run_group.attrs.get(SUBJECT_MASK_PUBLICATION_OWNER_ATTR) != owner_token
    ):
        raise RuntimeError(
            "Subject-mask child did not persist its atomic fail-closed sentinel."
        )
    if boundary is not None:
        boundary.bind_run(run_group, str(resolved_name))
    mark_run_started(run_group, run_name=str(resolved_name), stage="subject_masks")
    if output_parent == SUBJECT_MASK_CANONICAL_OUTPUT_PARENT:
        note_pending_latest(parent, str(resolved_name))
    verify_parent = boundary.fresh_parent() if boundary is not None else parent
    for selector in (
        "latest",
        "latest_complete",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        present, value = selector_snapshot[selector]
        if (selector in verify_parent.attrs) is not present or (
            present and verify_parent.attrs.get(selector) != value
        ):
            raise RuntimeError(
                "Subject-mask setup observed concurrent parent-selector mutation."
            )
    if (
        output_parent == SUBJECT_MASK_CANONICAL_OUTPUT_PARENT
        and verify_parent.attrs.get("latest_pending") != str(resolved_name)
    ):
        raise RuntimeError("Subject-mask attempt did not retain its pending selector.")
    return run_group, str(resolved_name)


def _write_canonical_subject_mask_selection(
    run_group: zarr.Group,
    source: Any,
) -> dict[str, np.ndarray]:
    """Persist the exact full canonical crop selection used by inference."""

    row_count = int(source.crop_geometry.row_identity.leading_dimension)
    rows = np.arange(row_count, dtype="<i8")
    selected = selected_subject_mask_crop_values(source, rows)
    row_chunks = (max(1, min(row_count, 4096)),)
    placement_chunks = (row_chunks[0], 4)
    run_group.create_array(
        "source_crop_row_ids",
        data=selected["source_crop_row_ids"],
        chunks=row_chunks,
        overwrite=True,
    )
    run_group.create_array(
        "instance_key",
        data=np.asarray(selected["instance_key"], dtype="<u8"),
        chunks=row_chunks,
        overwrite=True,
    )
    run_group.create_array(
        "source_acquisition_frame_index",
        data=np.asarray(selected["source_acquisition_frame_index"], dtype="<i8"),
        chunks=row_chunks,
        overwrite=True,
    )
    placement = np.ascontiguousarray(selected["source_crop_xywh"])
    run_group.create_array(
        "source_crop_xywh",
        data=placement,
        chunks=placement_chunks,
        overwrite=True,
    )
    return selected


_PACKAGE_SUBJECT_MASK_SELECTION_DTYPES: dict[str, np.dtype] = {
    "source_crop_row_ids": np.dtype("<i8"),
    "instance_key": np.dtype("<u8"),
    "source_acquisition_frame_index": np.dtype("<i8"),
    "source_crop_xywh": np.dtype("<f4"),
}


def _selected_package_subject_mask_crop_values(
    crop_group: zarr.Group,
    source_crop_row_ids: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Resolve the exact crop-v2 identity and placement used by one package."""

    raw_rows = np.asarray(source_crop_row_ids)
    if raw_rows.ndim != 1 or raw_rows.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "Package-backed subject-mask crop rows must be one integer vector."
        )
    rows = np.asarray(raw_rows, dtype="<i8")
    if rows.size == 0:
        raise ValueError("Package-backed subject-mask selection cannot be empty.")
    if "frame_indices" not in crop_group:
        raise ValueError("Package-backed subject masks require crop frame_indices.")
    crop_row_count = int(crop_group["frame_indices"].shape[0])
    if int(rows.min()) < 0 or int(rows.max()) >= crop_row_count:
        raise ValueError(
            "Package-backed subject-mask selection contains an out-of-bounds crop row."
        )
    if int(np.unique(rows).shape[0]) != int(rows.shape[0]):
        raise ValueError(
            "Package-backed subject-mask selection requires unique crop rows."
        )

    selected: dict[str, np.ndarray] = {"source_crop_row_ids": rows}
    source_shapes = {
        "instance_key": (crop_row_count,),
        "source_acquisition_frame_index": (crop_row_count,),
        "source_crop_xywh": (crop_row_count, 4),
    }
    for name, expected_shape in source_shapes.items():
        if name not in crop_group:
            raise ValueError(
                f"Package-backed subject masks require crop-v2 array {name!r}."
            )
        source_array = crop_group[name]
        expected_dtype = _PACKAGE_SUBJECT_MASK_SELECTION_DTYPES[name]
        if tuple(map(int, source_array.shape)) != expected_shape:
            raise ValueError(
                f"Crop-v2 array {name!r} has shape {source_array.shape}, expected "
                f"{expected_shape}."
            )
        if np.dtype(source_array.dtype) != expected_dtype:
            raise ValueError(
                f"Crop-v2 array {name!r} has dtype {source_array.dtype}, expected "
                f"{expected_dtype}."
            )
        values = np.asarray(source_array[rows])
        if values.dtype != expected_dtype:
            raise RuntimeError(
                f"Selected crop-v2 array {name!r} changed dtype during indexed read."
            )
        selected[name] = np.ascontiguousarray(values)
    return selected


def _write_package_subject_mask_crop_placement(
    run_group: zarr.Group,
    crop_group: zarr.Group,
    source_crop_row_ids: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Persist subject-mask-specific crop placement for a pixel work package."""

    selected = _selected_package_subject_mask_crop_values(
        crop_group,
        source_crop_row_ids,
    )
    placement = selected["source_crop_xywh"]
    row_count = int(placement.shape[0])
    source_chunks = getattr(crop_group["source_crop_xywh"], "chunks", None)
    source_row_chunk = (
        int(source_chunks[0])
        if source_chunks is not None and len(source_chunks) > 0
        else 4096
    )
    # This is a terminal worker array, not the recording-level publication.
    # Finalization re-plans and rematerializes its complete immutable shard grid.
    chunks = (max(1, min(row_count, source_row_chunk)), 4)
    run_group.create_array(
        "source_crop_xywh",
        data=placement,
        chunks=chunks,
        overwrite=True,
    )
    _validate_package_subject_mask_selection(
        run_group,
        crop_group,
        source_crop_row_ids,
        expected=selected,
    )
    return selected


def _validate_package_subject_mask_selection(
    run_group: zarr.Group,
    crop_group: zarr.Group,
    source_crop_row_ids: Sequence[int] | np.ndarray,
    *,
    expected: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Fail closed unless a package worker retains its exact crop-v2 selection."""

    resolved = (
        dict(expected)
        if expected is not None
        else _selected_package_subject_mask_crop_values(
            crop_group,
            source_crop_row_ids,
        )
    )
    for name, expected_dtype in _PACKAGE_SUBJECT_MASK_SELECTION_DTYPES.items():
        if name not in run_group:
            raise RuntimeError(
                f"Package-backed subject-mask output is missing required array {name!r}."
            )
        observed_array = run_group[name]
        observed = np.asarray(observed_array[:])
        selected = np.asarray(resolved[name])
        if np.dtype(observed_array.dtype) != expected_dtype:
            raise RuntimeError(
                f"Package-backed subject-mask array {name!r} has dtype "
                f"{observed_array.dtype}, expected {expected_dtype}."
            )
        if observed.shape != selected.shape or not np.array_equal(observed, selected):
            raise RuntimeError(
                f"Package-backed subject-mask array {name!r} differs from its "
                "exact selected crop-v2 rows."
            )


def _output_parent_from_args(args: argparse.Namespace) -> str:
    output_parent = str(
        getattr(args, "output_parent", SUBJECT_MASK_CANONICAL_OUTPUT_PARENT)
    )
    if output_parent not in SUBJECT_MASK_OUTPUT_PARENTS:
        raise ValueError(
            f"--output-parent must be one of {SUBJECT_MASK_OUTPUT_PARENTS}; got {output_parent!r}."
        )
    return output_parent


def _is_shard_output_parent(output_parent: str) -> bool:
    return str(output_parent) == SUBJECT_MASK_SHARD_OUTPUT_PARENT


def _sha256_document(value: Any) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": [int(dimension) for dimension in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _source_pixel_manifest(crop_source: CropImageSource) -> Mapping[str, Any] | None:
    source_array = getattr(crop_source, "_roi_images", None)
    manifest = getattr(source_array, "manifest", None)
    if isinstance(manifest, Mapping):
        return manifest
    manifest_path = getattr(crop_source, "pixel_materialization_manifest", None)
    if manifest_path is None:
        return None
    try:
        loaded = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            "Unable to load the consumed crop-pixel materialization manifest."
        ) from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("Crop-pixel materialization manifest must be an object.")
    return loaded


def _canonical_document_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_complete_collection_partition_attrs(
    *,
    crop_group: zarr.Group,
    crop_source: CropImageSource,
    selected_crop_rows: np.ndarray | None,
    total_rois: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Prove one work package is the exact crop-row partition for its window."""

    required_arguments = {
        "source_collection_id": args.source_collection_id,
        "source_collection_path": args.source_collection_path,
        "source_clip_id": args.source_clip_id,
        "source_clip_index": args.source_clip_index,
        "source_work_unit_id": args.source_work_unit_id,
        "source_shard_id": args.source_shard_id,
    }
    missing = [
        name
        for name, value in required_arguments.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    if missing:
        raise ValueError(
            "Complete collection-partition inference requires exact collection, "
            f"clip, work-unit, and shard identities; missing {missing!r}."
        )
    if int(args.source_clip_index) < 0:
        raise ValueError(
            "Complete collection-partition source_clip_index must be nonnegative."
        )
    if selected_crop_rows is None:
        raise ValueError(
            "Complete collection-partition inference requires selected work-package crop rows."
        )
    rows = np.asarray(selected_crop_rows, dtype=np.int64).reshape(-1)
    if rows.shape != (int(total_rois),) or rows.size == 0:
        raise ValueError(
            "Complete collection-partition work-package rows do not match the inference row count."
        )
    if not np.array_equal(rows, np.arange(int(rows[0]), int(rows[0]) + rows.size)):
        raise ValueError(
            "Complete collection-partition crop rows must be one contiguous ascending interval."
        )

    manifest = _source_pixel_manifest(crop_source)
    if not isinstance(manifest, Mapping):
        raise ValueError(
            "Complete collection-partition inference requires its authenticated work-package manifest."
        )
    if (
        manifest.get("schema_id") != "palette.crop_pixel_work_package"
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
    ):
        raise ValueError(
            "Complete collection-partition inference requires a complete crop pixel work-package v1 manifest."
        )
    builder = manifest.get("builder")
    if (
        not isinstance(builder, Mapping)
        or builder.get("semantics")
        != "global_crop_rows_from_authenticated_acquisition_video_window_v1"
    ):
        raise ValueError(
            "Complete collection partitions require authenticated acquisition-video-window materialization semantics."
        )
    binding = manifest.get("materialization_binding")
    expected_binding_fields = {
        "schema_id",
        "schema_version",
        "recording_identity",
        "camera_identity",
        "clip_id",
        "actual_start_frame",
        "end_frame_exclusive",
        "frame_count",
        "clip_index_document_sha256",
        "clip_video_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_binding_fields:
        raise ValueError(
            "Complete collection-partition materialization binding fields are not exact."
        )
    start_frame = binding.get("actual_start_frame")
    end_frame = binding.get("end_frame_exclusive")
    frame_count = binding.get("frame_count")
    if (
        binding.get("schema_id") != "palette.acquisition_video_frame_window"
        or binding.get("schema_version") != 1
        or type(start_frame) is not int
        or type(end_frame) is not int
        or type(frame_count) is not int
        or start_frame < 0
        or frame_count <= 0
        or end_frame != start_frame + frame_count
    ):
        raise ValueError(
            "Complete collection-partition materialization interval is invalid."
        )
    if str(binding.get("clip_id")) != str(args.source_clip_id):
        raise ValueError(
            "Complete collection-partition clip identity differs from --source-clip-id."
        )

    selection = manifest.get("selection")
    crop_row_count = int(crop_group["frame_indices"].shape[0])
    if (
        not isinstance(selection, Mapping)
        or selection.get("identity_mode") != "instance_key"
        or selection.get("ordering") != "ascending_source_crop_row"
        or selection.get("row_count") != int(total_rois)
        or selection.get("source_crop_total_rows") != crop_row_count
    ):
        raise ValueError(
            "Complete collection-partition work-package selection is not the exact crop-row contract."
        )
    if "frame_row_offsets" not in crop_group:
        raise ValueError(
            "Complete collection-partition inference requires crop frame_row_offsets."
        )
    offsets = np.asarray(crop_group["frame_row_offsets"][:], dtype=np.int64).reshape(-1)
    if (
        offsets.ndim != 1
        or offsets.size <= int(end_frame)
        or offsets[0] != 0
        or offsets[-1] != crop_row_count
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(
            "Complete collection-partition crop frame_row_offsets are invalid for the bound window."
        )
    expected_start = int(offsets[int(start_frame)])
    expected_stop = int(offsets[int(end_frame)])
    expected_rows = np.arange(expected_start, expected_stop, dtype=np.int64)
    if not np.array_equal(rows, expected_rows):
        raise ValueError(
            "Complete collection-partition rows do not exactly cover the crop rows in the bound frame window."
        )
    active_frames = np.asarray(crop_source.frame_indices, dtype=np.int64).reshape(-1)
    if (
        active_frames.shape != rows.shape
        or np.any(active_frames < int(start_frame))
        or np.any(active_frames >= int(end_frame))
    ):
        raise ValueError(
            "Complete collection-partition acquisition frames fall outside the bound frame window."
        )

    package_id = str(manifest.get("package_id") or "")
    if len(package_id) != 64 or any(
        character not in "0123456789abcdef" for character in package_id
    ):
        raise ValueError(
            "Complete collection-partition work package lacks a lowercase SHA-256 package_id."
        )
    payload = {
        "role": ROI_WORK_PACKAGE_ROLE_COMPLETE_PARTITION,
        "coverage_semantics": "exact_complete_crop_rows_for_acquisition_frame_window_v1",
        "work_package_id": package_id,
        "collection": {
            "source_collection_id": str(args.source_collection_id),
            "source_collection_path": str(args.source_collection_path),
            "source_clip_id": str(args.source_clip_id),
            "source_clip_index": int(args.source_clip_index),
            "source_work_unit_id": str(args.source_work_unit_id),
            "source_shard_id": str(args.source_shard_id),
        },
        "frame_window": dict(binding),
        "crop_rows": {
            "start": expected_start,
            "stop": expected_stop,
            "count": int(rows.size),
            "source_crop_total_rows": crop_row_count,
        },
        "validation": {
            "work_package_opened_and_content_verified": True,
            "row_interval_contiguous": True,
            "frame_offset_coverage_exact": True,
            "acquisition_frames_within_window": True,
        },
    }
    return {
        "schema_id": COLLECTION_PARTITION_CONTRACT_SCHEMA_ID,
        "schema_version": COLLECTION_PARTITION_CONTRACT_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": _canonical_document_sha256(payload),
    }


def _require_complete_recording_work_unit_attrs(
    *,
    crop_group: zarr.Group,
    crop_source: CropImageSource,
    selected_crop_rows: np.ndarray | None,
    total_rois: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Prove one flat-cache worker owns one complete recording work unit."""

    manifest_path_value = getattr(args, "expected_work_units_manifest", None)
    if manifest_path_value is None:
        raise ValueError("Recording work-unit inference requires its exact manifest.")
    manifest_path = Path(manifest_path_value).expanduser().resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read recording work-unit manifest: {manifest_path}"
        ) from exc
    units = document.get("units") if isinstance(document, Mapping) else None
    if (
        not isinstance(document, Mapping)
        or document.get("schema_id") != "palette.subject_mask.expected_work_units"
        or document.get("schema_version") != 1
        or not isinstance(units, list)
        or len(units) != 1
        or document.get("units_digest") != _canonical_document_sha256(units)
    ):
        raise ValueError(
            "Recording work-unit manifest must be one exact, digest-bound unit."
        )
    unit = units[0]
    required_unit_fields = {
        "work_unit_id",
        "work_unit_index",
        "source_clip_id",
        "source_clip_index",
        "frame_start",
        "frame_stop",
        "row_start",
        "row_stop",
    }
    if not isinstance(unit, Mapping) or set(unit) != required_unit_fields:
        raise ValueError("Recording work-unit fields are not exact.")

    required_arguments = {
        "source_collection_id": getattr(args, "source_collection_id", None),
        "source_collection_path": getattr(args, "source_collection_path", None),
        "source_clip_id": getattr(args, "source_clip_id", None),
        "source_clip_index": getattr(args, "source_clip_index", None),
        "source_work_unit_id": getattr(args, "source_work_unit_id", None),
        "source_shard_id": getattr(args, "source_shard_id", None),
    }
    missing = [
        name
        for name, value in required_arguments.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    if missing:
        raise ValueError(
            "Recording work-unit inference requires exact source identities; "
            f"missing {missing!r}."
        )
    if (
        required_arguments["source_collection_path"] != str(manifest_path)
        or required_arguments["source_collection_id"] != unit.get("source_clip_id")
        or required_arguments["source_clip_id"] != unit.get("source_clip_id")
        or required_arguments["source_clip_index"] != unit.get("source_clip_index")
        or required_arguments["source_work_unit_id"] != unit.get("work_unit_id")
        or required_arguments["source_shard_id"] != unit.get("work_unit_id")
        or unit.get("work_unit_index") != 0
    ):
        raise ValueError(
            "Recording worker source identity differs from its work-unit manifest."
        )

    crop_row_count = int(crop_group["frame_indices"].shape[0])
    rows = (
        np.arange(crop_row_count, dtype=np.int64)
        if selected_crop_rows is None
        else np.asarray(selected_crop_rows, dtype=np.int64).reshape(-1)
    )
    if (
        int(total_rois) != crop_row_count
        or rows.shape != (crop_row_count,)
        or not np.array_equal(rows, np.arange(crop_row_count, dtype=np.int64))
    ):
        raise ValueError(
            "Recording work-unit inference must cover every crop row exactly once."
        )
    if "frame_row_offsets" not in crop_group:
        raise ValueError(
            "Recording work-unit inference requires crop frame_row_offsets."
        )
    offsets = np.asarray(crop_group["frame_row_offsets"][:], dtype=np.int64).reshape(-1)
    n_frames = int(offsets.size - 1)
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != crop_row_count
        or np.any(offsets[1:] < offsets[:-1])
        or unit.get("frame_start") != 0
        or unit.get("frame_stop") != n_frames
        or unit.get("row_start") != 0
        or unit.get("row_stop") != crop_row_count
    ):
        raise ValueError(
            "Recording work-unit manifest differs from crop frame/row coverage."
        )
    active_frames = np.asarray(crop_source.frame_indices, dtype=np.int64).reshape(-1)
    if (
        active_frames.shape != rows.shape
        or np.any(active_frames < 0)
        or np.any(active_frames >= n_frames)
    ):
        raise ValueError(
            "Recording work-unit acquisition frames fall outside the recording."
        )

    pixel_manifest = _source_pixel_manifest(crop_source)
    array = pixel_manifest.get("array") if isinstance(pixel_manifest, Mapping) else None
    cache_key = (
        pixel_manifest.get("cache_key") if isinstance(pixel_manifest, Mapping) else None
    )
    array_sha256 = array.get("sha256") if isinstance(array, Mapping) else None
    array_shape = array.get("shape") if isinstance(array, Mapping) else None
    if (
        not isinstance(pixel_manifest, Mapping)
        or pixel_manifest.get("schema") != "palette_roi_cache_flat_bin_v1"
        or pixel_manifest.get("layout") != "flat_bin_v1"
        or pixel_manifest.get("cache_complete") is not True
        or not isinstance(cache_key, str)
        or len(cache_key) != 64
        or any(character not in "0123456789abcdef" for character in cache_key.lower())
        or not isinstance(array_sha256, str)
        or len(array_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in array_sha256.lower()
        )
        or array_shape
        != [
            crop_row_count,
            int(crop_source.roi_shape[0]),
            int(crop_source.roi_shape[1]),
        ]
    ):
        raise ValueError(
            "Recording work-unit inference requires one complete authenticated flat ROI cache."
        )

    payload = {
        "role": "complete_recording_work_unit",
        "coverage_semantics": (
            "exact_complete_crop_rows_for_recording_frame_window_v1"
        ),
        "work_unit_manifest": {
            "schema_id": document["schema_id"],
            "schema_version": document["schema_version"],
            "units_digest": document["units_digest"],
            "work_unit_index": 0,
        },
        "pixel_source": {
            "schema": pixel_manifest["schema"],
            "layout": pixel_manifest["layout"],
            "cache_key": cache_key,
            "array_sha256": array_sha256,
            "array_shape": list(array_shape),
        },
        "collection": {
            "source_collection_id": str(required_arguments["source_collection_id"]),
            "source_collection_path": str(required_arguments["source_collection_path"]),
            "source_clip_id": str(required_arguments["source_clip_id"]),
            "source_clip_index": int(required_arguments["source_clip_index"]),
            "source_work_unit_id": str(required_arguments["source_work_unit_id"]),
            "source_shard_id": str(required_arguments["source_shard_id"]),
        },
        "frame_window": {
            "schema_id": "palette.subject_mask.recording_frame_window",
            "schema_version": 1,
            "recording_identity": str(required_arguments["source_collection_id"]),
            "clip_id": str(required_arguments["source_clip_id"]),
            "actual_start_frame": 0,
            "end_frame_exclusive": n_frames,
            "frame_count": n_frames,
        },
        "crop_rows": {
            "start": 0,
            "stop": crop_row_count,
            "count": crop_row_count,
            "source_crop_total_rows": crop_row_count,
        },
        "validation": {
            "expected_work_unit_manifest_validated": True,
            "flat_cache_manifest_validated": True,
            "row_interval_contiguous": True,
            "frame_offset_coverage_exact": True,
            "acquisition_frames_within_window": True,
        },
    }
    return {
        "schema_id": RECORDING_WORK_UNIT_CONTRACT_SCHEMA_ID,
        "schema_version": RECORDING_WORK_UNIT_CONTRACT_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": _canonical_document_sha256(payload),
    }


def _roi_work_package_publication_attrs(
    *,
    crop_group: zarr.Group,
    crop_source: CropImageSource,
    selected_crop_rows: np.ndarray | None,
    total_rois: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    expected_work_units_manifest = getattr(args, "expected_work_units_manifest", None)
    package_id = getattr(crop_source, "pixel_materialization_id", None)
    if expected_work_units_manifest is not None:
        if (
            package_id is not None
            or getattr(args, "roi_work_package_role", None) is not None
        ):
            raise ValueError(
                "Recording work-unit and crop work-package bindings are mutually exclusive."
            )
        contract = _require_complete_recording_work_unit_attrs(
            crop_group=crop_group,
            crop_source=crop_source,
            selected_crop_rows=selected_crop_rows,
            total_rois=total_rois,
            args=args,
        )
        return {
            "roi_work_package_role": "complete_recording_work_unit",
            "incremental_materialization_role": "complete_recording_work_unit",
            "canonical_finalization_policy": "collection_shard_finalization_allowed",
            "collection_partition_contract": contract,
        }
    if package_id is None:
        if args.roi_work_package_role is not None:
            raise ValueError(
                "--roi-work-package-role requires --roi-work-package-manifest."
            )
        return {}
    role = str(args.roi_work_package_role or ROI_WORK_PACKAGE_ROLE_DELTA)
    attrs: dict[str, object] = {
        "source_crop_pixel_work_package_id": str(package_id),
        "source_crop_pixel_work_package_manifest": str(
            getattr(crop_source, "pixel_materialization_manifest", "")
        ),
        "source_crop_pixel_work_package_rows": int(total_rois),
        "roi_work_package_role": role,
        "incremental_materialization_role": role,
    }
    if role == ROI_WORK_PACKAGE_ROLE_DELTA:
        attrs["canonical_finalization_policy"] = "incremental_compaction_required"
        return attrs
    if role != ROI_WORK_PACKAGE_ROLE_COMPLETE_PARTITION:
        raise ValueError(f"Unsupported ROI work-package role {role!r}.")
    contract = _require_complete_collection_partition_attrs(
        crop_group=crop_group,
        crop_source=crop_source,
        selected_crop_rows=selected_crop_rows,
        total_rois=total_rois,
        args=args,
    )
    attrs.update(
        {
            "canonical_finalization_policy": "collection_shard_finalization_allowed",
            "collection_partition_contract": contract,
        }
    )
    return attrs


def _subject_mask_scientific_documents(
    *,
    run_group: zarr.Group,
    crop_group: zarr.Group,
    crop_source: CropImageSource,
    crop_run_name: str,
    checkpoint_artifact: Mapping[str, Any],
    selected_model: Mapping[str, Any],
    label_schema_id: str,
    mask_labels: Sequence[str],
    model_input_transform: ModelInputTransform,
    mask_probs_dtype: str,
    observed_pixels_sha256: str,
    work_package_attrs: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Build exact science-facing inputs while excluding runtime/storage knobs."""

    pixel_manifest = _source_pixel_manifest(crop_source)
    declared_pixels_sha256: str | None = None
    manifest_schema: object = None
    if pixel_manifest is not None:
        array = pixel_manifest.get("array")
        if not isinstance(array, Mapping):
            raise ValueError("Consumed ROI-pixel manifest lacks its array contract.")
        declared = array.get("sha256")
        if (
            not isinstance(declared, str)
            or len(declared) != 64
            or any(character not in "0123456789abcdef" for character in declared)
        ):
            raise ValueError(
                "Consumed ROI-pixel manifests require array.sha256 as lowercase SHA-256."
            )
        declared_pixels_sha256 = declared
        if declared_pixels_sha256 != observed_pixels_sha256:
            raise RuntimeError(
                "ROI pixels changed between authenticated staging/open and inference: "
                f"expected {declared_pixels_sha256}, observed "
                f"{observed_pixels_sha256}."
            )
        manifest_schema = pixel_manifest.get("schema_id") or pixel_manifest.get(
            "schema"
        )

    crop_manifest = crop_group.attrs.get("run_manifest")
    crop_manifest_reference: dict[str, object] | None = None
    if isinstance(crop_manifest, Mapping):
        payload_digest = crop_manifest.get("payload_digest")
        if not isinstance(payload_digest, str) or len(payload_digest) != 64:
            raise ValueError("Crop run_manifest lacks its exact payload digest.")
        crop_manifest_reference = {
            "schema_id": crop_manifest.get("schema_id"),
            "schema_version": crop_manifest.get("schema_version"),
            "payload_digest": payload_digest,
        }

    row_paths = (
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "frame_indices",
        "source_frame_indices",
        "source_clip_indices",
        "source_clip_local_frame_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
    )
    row_arrays = {
        path: {
            "shape": [int(value) for value in run_group[path].shape],
            "dtype": str(np.dtype(run_group[path].dtype)),
            "sha256": streaming_array_sha256(run_group[path]),
        }
        for path in row_paths
        if path in run_group
    }
    required_row_paths = {
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
    }
    if not required_row_paths <= set(row_arrays):
        raise ValueError(
            "Subject-mask scientific identity requires exact crop-row, instance, "
            "and acquisition-frame arrays."
        )

    model = {
        "artifact_role": checkpoint_artifact.get("role"),
        "artifact_sha256": checkpoint_artifact.get("sha256"),
        "artifact_size_bytes": checkpoint_artifact.get("size_bytes"),
        "registry_set_id": selected_model.get("set_id"),
        "registry_run_id": selected_model.get("run_id"),
        "label_schema_id": label_schema_id,
    }
    roi_coordinates = getattr(crop_source, "roi_coordinates_full", None)
    if roi_coordinates is None and "roi_coordinates_full" in crop_group:
        roi_coordinates = np.asarray(crop_group["roi_coordinates_full"][:])
    if roi_coordinates is None:
        raise ValueError(
            "Subject-mask scientific identity requires exact ROI placement geometry."
        )
    crop = {
        "run_id": str(crop_run_name),
        "run_group_path": str(getattr(crop_group, "path", "")).strip("/"),
        "run_manifest": crop_manifest_reference,
        "storage_mode": str(crop_source.storage_mode),
        "roi_shape_hw": [int(value) for value in crop_source.roi_shape],
        "roi_coordinates_full": _sha256_document(
            np.asarray(roi_coordinates, dtype="<i4")
        ),
        "source_collection_id": args.source_collection_id,
        "source_clip_id": args.source_clip_id,
        "source_clip_index": args.source_clip_index,
        "source_work_unit_id": args.source_work_unit_id,
        "source_shard_id": args.source_shard_id,
        "collection_partition_contract": work_package_attrs.get(
            "collection_partition_contract"
        ),
    }
    pixels = {
        "profile": (
            str(manifest_schema)
            if manifest_schema is not None
            else str(crop_source.roi_read_mode)
        ),
        "decoded_shape": [
            int(crop_source.total_rois),
            int(crop_source.roi_shape[0]),
            int(crop_source.roi_shape[1]),
        ],
        "decoded_dtype": "uint8",
        "decoded_order": "C",
        "decoded_pixels_sha256": observed_pixels_sha256,
        "declared_pixels_sha256": declared_pixels_sha256,
        "cache_key": getattr(crop_source, "roi_cache_key", None),
        "pixel_materialization_id": getattr(
            crop_source, "pixel_materialization_id", None
        ),
        "pixel_contract": getattr(crop_source, "roi_pixel_contract", None),
        "work_package_role": work_package_attrs.get("roi_work_package_role"),
    }
    row_identity = {
        "row_count": int(crop_source.total_rois),
        "arrays": row_arrays,
    }
    inference_contract = {
        "segmenter": "unet",
        "label_schema_id": label_schema_id,
        "mask_labels": [str(label) for label in mask_labels],
        "model_input_transform": model_input_transform.to_attrs(),
        "probability_semantics": "sigmoid_multilabel_logits",
        "probability_dtype": str(mask_probs_dtype),
        "probability_encoding": (
            "linear_uint8_0_255" if mask_probs_dtype == "uint8" else "unit_float"
        ),
        "mask_probability_threshold": 0.5,
        "overlap_policy": "independent_sigmoid",
    }
    return build_subject_mask_scientific_identity(
        stage_kind="raw_subject_mask",
        model=model,
        crop=crop,
        pixels=pixels,
        row_identity=row_identity,
        inference_contract=inference_contract,
    )


def _resolve_subject_mask_attempt_lineage_once(
    *,
    parent: zarr.Group,
    current_run_name: str,
    scientific_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
    retry_of_attempt_id: str | None,
    supersedes_run: str | None,
) -> dict[str, object]:
    """Resolve requested lineage against immutable terminal siblings."""

    scientific_errors = validate_subject_mask_scientific_identity(scientific_identity)
    attempt_errors = validate_subject_mask_attempt(attempt)
    if scientific_errors or attempt_errors:
        raise ValueError(
            "Invalid subject-mask attempt records: "
            f"science={scientific_errors!r}, attempt={attempt_errors!r}."
        )
    attempt_id = str(attempt["payload"]["attempt_id"])
    retry_matches: list[tuple[str, Any, Mapping[str, Any]]] = []
    for sibling_name in parent.keys():
        if sibling_name == current_run_name:
            continue
        sibling = parent[sibling_name]
        sibling_attempt = sibling.attrs.get(SUBJECT_MASK_ATTEMPT_ATTR)
        if not isinstance(sibling_attempt, Mapping):
            continue
        errors = validate_subject_mask_attempt(sibling_attempt)
        if errors:
            raise ValueError(
                f"Sibling {sibling_name!r} has malformed subject-mask attempt "
                f"metadata: {errors!r}."
            )
        sibling_attempt_id = str(sibling_attempt["payload"]["attempt_id"])
        if sibling_attempt_id == attempt_id:
            raise ValueError(
                f"Subject-mask attempt_id {attempt_id!r} is already in use by "
                f"{sibling_name!r}."
            )
        if retry_of_attempt_id is not None and sibling_attempt_id == str(
            retry_of_attempt_id
        ):
            retry_matches.append((str(sibling_name), sibling, sibling_attempt))

    retry_evidence: dict[str, object] | None = None
    if retry_of_attempt_id is not None:
        if len(retry_matches) != 1:
            raise ValueError(
                "--retry-of-attempt-id must identify exactly one sibling attempt; "
                f"found {len(retry_matches)}."
            )
        retry_name, retry_group, retry_attempt = retry_matches[0]
        retry_science = retry_group.attrs.get(SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR)
        if (
            retry_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != "failed"
            or not isinstance(retry_science, Mapping)
            or retry_science.get("digest") != scientific_identity.get("digest")
        ):
            raise ValueError(
                "A retry must reference one failed attempt with the exact same "
                "scientific identity."
            )
        retry_evidence = {
            "run_name": retry_name,
            "run_path": f"{str(getattr(parent, 'path', '')).strip('/')}/{retry_name}",
            "attempt_id": str(retry_of_attempt_id),
            "attempt_payload_digest": retry_attempt.get("payload_digest"),
            "scientific_identity_digest": retry_science.get("digest"),
            "completion_status": "failed",
        }

    supersedes_evidence: dict[str, object] | None = None
    if supersedes_run is not None:
        predecessor_name = str(supersedes_run).strip()
        if predecessor_name == current_run_name or predecessor_name not in parent:
            raise ValueError(
                "--supersedes-run must identify a different existing sibling run."
            )
        predecessor = parent[predecessor_name]
        if predecessor.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            raise ValueError("A superseded subject-mask run must be complete.")
        predecessor_attempt = predecessor.attrs.get(SUBJECT_MASK_ATTEMPT_ATTR)
        predecessor_science = predecessor.attrs.get(
            SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR
        )
        supersedes_evidence = {
            "run_name": predecessor_name,
            "run_path": f"{str(getattr(parent, 'path', '')).strip('/')}/{predecessor_name}",
            "completion_status": RUN_STATUS_COMPLETE,
            "attempt_payload_digest": (
                predecessor_attempt.get("payload_digest")
                if isinstance(predecessor_attempt, Mapping)
                else None
            ),
            "scientific_identity_digest": (
                predecessor_science.get("digest")
                if isinstance(predecessor_science, Mapping)
                else None
            ),
        }
    return {
        "retry_of": retry_evidence,
        "supersedes": supersedes_evidence,
        "lineage_policy": "explicit_terminal_sibling_binding_v1",
    }


def _resolve_subject_mask_attempt_lineage(
    *,
    parent: zarr.Group,
    current_run_name: str,
    scientific_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
    retry_of_attempt_id: str | None,
    supersedes_run: str | None,
) -> dict[str, object]:
    """Retry read-only sibling discovery across transient NFS ESTALE faults."""

    delays = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    for retry_index in range(len(delays) + 1):
        try:
            return _resolve_subject_mask_attempt_lineage_once(
                parent=parent,
                current_run_name=current_run_name,
                scientific_identity=scientific_identity,
                attempt=attempt,
                retry_of_attempt_id=retry_of_attempt_id,
                supersedes_run=supersedes_run,
            )
        except OSError as exc:
            if exc.errno != errno.ESTALE or retry_index == len(delays):
                raise
            time.sleep(delays[retry_index])
    raise AssertionError("unreachable subject-mask lineage retry state")


def _subject_mask_stage_path(
    output_parent: str, run_name: str, artifact: str = "mask_probs_roi"
) -> str:
    return f"{output_parent}/{run_name}/{artifact}"


def _shard_attrs_from_args(
    args: argparse.Namespace, *, output_parent: str
) -> dict[str, object]:
    if not _is_shard_output_parent(output_parent):
        return {}
    alias_manifest = args.source_roi_cache_alias_manifest or args.roi_cache_manifest
    attrs: dict[str, object] = {
        "is_collection_shard": True,
        "stage_selector_eligible": False,
        "subject_mask_output_parent": output_parent,
        "canonical_stage_parent": SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
        "canonical_selector_publication": "suppressed_for_collection_shard",
        "registry_status_publication": "suppressed_for_collection_shard",
    }
    optional_values = {
        "source_collection_id": args.source_collection_id,
        "source_collection_path": args.source_collection_path,
        "source_clip_id": args.source_clip_id,
        "source_clip_index": args.source_clip_index,
        "source_work_unit_id": args.source_work_unit_id,
        "source_roi_cache_alias_manifest": alias_manifest,
        "source_roi_cache_row_index_path": args.source_roi_cache_row_index_path,
        "source_shard_id": args.source_shard_id,
    }
    for key, value in optional_values.items():
        if value is None:
            continue
        if isinstance(value, Path):
            attrs[key] = str(value)
        else:
            attrs[key] = int(value) if key == "source_clip_index" else str(value)
    return attrs


def _copy_detection_source_array(
    run_group: zarr.Group,
    crop_group: zarr.Group,
    *,
    source_crop_row_ids: np.ndarray | None = None,
) -> None:
    array_name = "detection_source"
    if array_name not in crop_group:
        return
    src = crop_group[array_name]
    data = (
        src[:]
        if source_crop_row_ids is None
        else src[np.asarray(source_crop_row_ids, dtype=np.int64)]
    )
    chunks = getattr(src, "chunks", None)
    if not chunks:
        chunks = tuple(max(1, min(dim, 1024)) for dim in data.shape)
    run_group.create_array(array_name, data=data, chunks=chunks, overwrite=True)


def _write_subject_mask_output_batch(
    batch: _SubjectMaskOutputBatch,
    *,
    probs_arr: zarr.Array,
    masks_arr: Optional[zarr.Array],
    prob_max: np.ndarray,
    mask_present: np.ndarray,
    area_px: np.ndarray,
    centroid_xy: np.ndarray,
    centroid_valid: np.ndarray,
    bbox_xyxy: np.ndarray,
    bbox_valid: np.ndarray,
    progress: Progress,
    task: int,
    profiler: InferenceTimingProfiler,
    validation_accumulators: (
        Mapping[str, SubjectMaskArrayUnitAccumulator] | None
    ) = None,
) -> None:
    start = int(batch.start)
    stop = int(batch.stop)
    batch_count = max(0, stop - start)

    probability_stage = str(getattr(probs_arr, "timing_stage", "output_write_probs"))
    with profiler.time(probability_stage, items=batch_count):
        probs_arr[start:stop] = batch.probs_out
    if masks_arr is not None:
        if batch.binary is None:
            raise ValueError(
                "masks_roi materialization requested but binary masks were not returned."
            )
        with profiler.time("output_write_binary", items=batch_count):
            masks_arr[start:stop] = batch.binary

    with profiler.time("metric_compute", items=batch_count):
        prob_max[start:stop, :] = batch.metrics["prob_max"]
        mask_present[start:stop, :] = batch.metrics["mask_present"]
        area_px[start:stop, :] = batch.metrics["area_px"]
        centroid_xy[start:stop, :, :] = batch.metrics["centroid_xy"]
        centroid_valid[start:stop, :] = batch.metrics["centroid_valid"]
        bbox_xyxy[start:stop, :, :] = batch.metrics["bbox_xyxy"]
        bbox_valid[start:stop, :] = batch.metrics["bbox_valid"]

    if validation_accumulators is not None:
        with profiler.time("semantic_receipt_hash", items=batch_count):
            _append_raw_worker_validation_batch(validation_accumulators, batch)

    with profiler.time("progress_update", items=batch_count):
        progress.advance(task, batch_count)


def _probability_array_digest(array: zarr.Array, *, row_step: int) -> str:
    digest = hashlib.sha256()
    total_rows = int(array.shape[0])
    channels = int(array.shape[1])
    for channel in range(channels):
        for start in range(0, total_rows, int(row_step)):
            stop = min(total_rows, start + int(row_step))
            values = np.asarray(array[start:stop, channel, :, :])
            digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _probability_array_digests_by_channel(
    array: zarr.Array, *, row_step: int
) -> list[str]:
    channel_digests: list[str] = []
    total_rows = int(array.shape[0])
    for channel in range(int(array.shape[1])):
        digest = hashlib.sha256()
        for start in range(0, total_rows, int(row_step)):
            stop = min(total_rows, start + int(row_step))
            values = np.asarray(array[start:stop, channel, :, :])
            digest.update(np.ascontiguousarray(values).view(np.uint8))
        channel_digests.append(digest.hexdigest())
    return channel_digests


def _aggregate_channel_digests(channel_digests: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in channel_digests:
        digest.update(bytes.fromhex(str(value)))
    return digest.hexdigest()


class _DoubleBufferedProbabilityShardWriter:
    """Accumulate inference batches and write each physical probability shard once."""

    timing_stage = "output_shard_buffer_submit"

    def __init__(
        self,
        destination: zarr.Array,
        *,
        shard_rows: int,
        profiler: InferenceTimingProfiler,
        buffer_count: int = 2,
        destination_validation_mode: str = MASK_PROBS_DESTINATION_VALIDATION_FULL,
    ) -> None:
        shape = tuple(int(value) for value in destination.shape)
        if len(shape) != 4:
            raise ValueError(
                f"Probability destination must have shape (N,C,H,W); got {shape}."
            )
        resolved_buffer_count = int(buffer_count)
        if resolved_buffer_count != 2:
            raise ValueError(
                "Double-buffered probability writing requires exactly 2 buffers; "
                f"got {buffer_count}."
            )
        self.destination = destination
        self.total_rows, self.channel_count, self.height, self.width = shape
        self.shard_rows = int(shard_rows)
        self.buffer_rows = min(self.shard_rows, max(1, self.total_rows))
        self.profiler = profiler
        self.buffer_count = resolved_buffer_count
        self.destination_validation_mode = str(destination_validation_mode)
        if self.destination_validation_mode not in (
            MASK_PROBS_DESTINATION_VALIDATION_MODES
        ):
            raise ValueError(
                "Unsupported probability destination validation mode "
                f"{self.destination_validation_mode!r}."
            )
        self.buffers = [
            np.empty(
                (self.channel_count, self.buffer_rows, self.height, self.width),
                dtype=destination.dtype,
            )
            for _ in range(self.buffer_count)
        ]
        self._free_buffers: Queue[int] = Queue(maxsize=self.buffer_count)
        for index in range(self.buffer_count):
            self._free_buffers.put(index)
        self._flush_queue: Queue[object] = Queue(maxsize=self.buffer_count)
        self._sentinel = object()
        self._errors: list[BaseException] = []
        self._active_index: int | None = None
        self._active_start = 0
        self._active_rows = 0
        self._next_row = 0
        self._source_digests = (
            [hashlib.sha256() for _ in range(self.channel_count)]
            if self.destination_validation_mode
            == MASK_PROBS_DESTINATION_VALIDATION_FULL
            else None
        )
        self._write_seconds = 0.0
        self._full_shards_written = 0
        self._partial_shards_written = 0
        self._worker = Thread(
            target=self._flush_worker,
            name="subject-mask-probability-shard-writer",
            daemon=True,
        )
        self._worker.start()

    def _raise_error(self) -> None:
        if self._errors:
            raise RuntimeError(
                "Double-buffered probability shard writer failed."
            ) from self._errors[0]

    def _acquire_buffer(self, *, start: int) -> None:
        self._raise_error()
        index = self._free_buffers.get()
        self._raise_error()
        self._active_index = int(index)
        self._active_start = int(start)
        self._active_rows = 0

    def _submit_active(self) -> None:
        if self._active_index is None or self._active_rows <= 0:
            return
        self._flush_queue.put(
            (self._active_index, self._active_start, self._active_rows)
        )
        self._active_index = None
        self._active_rows = 0

    def __setitem__(self, key: slice, values: np.ndarray) -> None:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError(
                "Probability shard writer accepts contiguous row slices only."
            )
        start = int(0 if key.start is None else key.start)
        stop = int(self.total_rows if key.stop is None else key.stop)
        if start != self._next_row:
            raise ValueError(
                f"Probability batches must be sequential; expected row {self._next_row}, got {start}."
            )
        batch = np.asarray(values, dtype=self.destination.dtype)
        expected_shape = (stop - start, self.channel_count, self.height, self.width)
        if tuple(int(value) for value in batch.shape) != expected_shape:
            raise ValueError(
                f"Probability batch shape {batch.shape} does not match {expected_shape}."
            )

        source_offset = 0
        while source_offset < int(batch.shape[0]):
            if self._active_index is None:
                self._acquire_buffer(start=self._next_row)
            assert self._active_index is not None
            capacity = self.buffer_rows - self._active_rows
            take = min(capacity, int(batch.shape[0]) - source_offset)
            target_start = self._active_rows
            target_stop = target_start + take
            source_stop = source_offset + take
            with self.profiler.time("output_shard_buffer_fill", items=take):
                np.copyto(
                    self.buffers[self._active_index][:, target_start:target_stop, :, :],
                    np.moveaxis(batch[source_offset:source_stop, :, :, :], 0, 1),
                )
            self._active_rows = target_stop
            self._next_row += take
            source_offset = source_stop
            if self._active_rows == self.buffer_rows:
                self._submit_active()
        self._raise_error()

    def _flush_worker(self) -> None:
        failed = False
        while True:
            item = self._flush_queue.get()
            try:
                if item is self._sentinel:
                    return
                index, start, row_count = item
                index = int(index)
                start = int(start)
                row_count = int(row_count)
                if failed:
                    continue
                stop = start + row_count
                write_started = time.perf_counter()
                with self.profiler.time("output_shard_write", items=row_count):
                    for channel in range(self.channel_count):
                        channel_values = self.buffers[index][channel, :row_count, :, :]
                        if self._source_digests is not None:
                            self._source_digests[channel].update(
                                channel_values.view(np.uint8)
                            )
                        self.destination[
                            start:stop,
                            channel : channel + 1,
                            :,
                            :,
                        ] = channel_values[:, None, :, :]
                self._write_seconds += float(time.perf_counter() - write_started)
                if row_count == self.shard_rows:
                    self._full_shards_written += 1
                else:
                    self._partial_shards_written += 1
            except (
                BaseException
            ) as exc:  # pragma: no cover - exercised through caller failure
                self._errors.append(exc)
                failed = True
            finally:
                if item is not self._sentinel:
                    self._free_buffers.put(int(item[0]))
                self._flush_queue.task_done()

    def finish(
        self,
        *,
        validation_row_step: int,
    ) -> dict[str, object]:
        self._raise_error()
        self._submit_active()
        self._flush_queue.put(self._sentinel)
        self._flush_queue.join()
        self._worker.join()
        self._raise_error()
        if self._next_row != self.total_rows:
            raise RuntimeError(
                f"Probability shard writer received {self._next_row} of {self.total_rows} rows."
            )

        validation_mode = self.destination_validation_mode
        source_digests = (
            [digest.hexdigest() for digest in self._source_digests]
            if self._source_digests is not None
            else None
        )
        destination_digests: list[str] | None = None
        validation_seconds = 0.0
        if validation_mode == MASK_PROBS_DESTINATION_VALIDATION_FULL:
            assert source_digests is not None
            validation_started = time.perf_counter()
            with self.profiler.time("output_shard_validate", items=self.total_rows):
                destination_digests = _probability_array_digests_by_channel(
                    self.destination,
                    row_step=int(validation_row_step),
                )
            validation_seconds = float(time.perf_counter() - validation_started)
            if source_digests != destination_digests:
                raise RuntimeError(
                    "Direct-sharded probability digest mismatch: "
                    f"source={source_digests} destination={destination_digests}."
                )
        source_digest = (
            _aggregate_channel_digests(source_digests)
            if source_digests is not None
            else None
        )
        destination_digest = (
            _aggregate_channel_digests(destination_digests)
            if destination_digests is not None
            else None
        )
        buffer_bytes_each = int(self.buffers[0].nbytes)
        return {
            "schema_id": MASK_PROBS_DIRECT_SHARDING_SCHEMA,
            "status": "complete",
            "write_mode": "double_buffered_direct",
            "source_working_array_created": False,
            "canonical_array": MASK_PROBS_CANONICAL_ARRAY,
            "row_count": self.total_rows,
            "channel_count": self.channel_count,
            "inner_chunk_shape": [int(value) for value in self.destination.chunks],
            "outer_shard_shape": [int(value) for value in self.destination.shards],
            "inner_chunks_per_shard": int(
                self.shard_rows // int(self.destination.chunks[0])
            ),
            "buffer_count": self.buffer_count,
            "buffer_shape_channel_first": list(self.buffers[0].shape),
            "buffer_bytes_each": buffer_bytes_each,
            "total_buffer_bytes": int(buffer_bytes_each * self.buffer_count),
            "full_row_shards_written": self._full_shards_written,
            "partial_row_shards_written": self._partial_shards_written,
            "write_seconds": self._write_seconds,
            "destination_validation_mode": validation_mode,
            "destination_validation_status": (
                "complete"
                if validation_mode == MASK_PROBS_DESTINATION_VALIDATION_FULL
                else "deferred_to_mandatory_final_layout_unit"
            ),
            "validation_seconds": validation_seconds,
            "digest_scheme": (
                "sha256_per_channel_then_sha256_v1"
                if validation_mode == MASK_PROBS_DESTINATION_VALIDATION_FULL
                else "semantic_receipt_units_then_mandatory_final_layout_v1"
            ),
            "source_sha256_by_channel": source_digests,
            "destination_sha256_by_channel": destination_digests,
            "source_sha256": source_digest,
            "destination_sha256": destination_digest,
            "exact_match": (
                True
                if validation_mode == MASK_PROBS_DESTINATION_VALIDATION_FULL
                else None
            ),
        }


def _postpack_probability_shards(
    run_group: zarr.Group,
    *,
    source_name: str,
    shard_rows: int,
    profiler: InferenceTimingProfiler,
) -> dict[str, object]:
    source = run_group[source_name]
    if int(source.ndim) != 4:
        raise ValueError(
            f"{source_name} must have shape (N,C,H,W); got {source.shape}."
        )
    chunks = tuple(int(value) for value in source.chunks)
    inner_rows = int(chunks[0])
    resolved_shard_rows = int(shard_rows)
    if resolved_shard_rows <= inner_rows:
        raise ValueError(
            f"--mask-probs-shard-rois must exceed inner chunk rows {inner_rows}; "
            f"got {resolved_shard_rows}."
        )
    if resolved_shard_rows % inner_rows != 0:
        raise ValueError(
            f"--mask-probs-shard-rois must be an integer multiple of "
            f"--mask-probs-chunk-rois ({inner_rows}); got {resolved_shard_rows}."
        )

    shape = tuple(int(value) for value in source.shape)
    shards = (resolved_shard_rows, 1, shape[2], shape[3])
    if MASK_PROBS_CANONICAL_ARRAY in run_group:
        del run_group[MASK_PROBS_CANONICAL_ARRAY]
    destination = run_group.create_array(
        MASK_PROBS_CANONICAL_ARRAY,
        shape=shape,
        dtype=source.dtype,
        chunks=chunks,
        shards=shards,
        fill_value=source.fill_value,
        overwrite=True,
        **_compression_kwargs(source),
    )
    destination.attrs.update(
        {
            "storage_layout": "indexed_sharding_v1",
            "inner_chunk_shape": list(chunks),
            "outer_shard_shape": list(shards),
            "postpack_source_array": source_name,
        }
    )

    write_started = time.perf_counter()
    with profiler.time("output_postpack_shards", items=shape[0]):
        for channel in range(shape[1]):
            for start in range(0, shape[0], resolved_shard_rows):
                stop = min(shape[0], start + resolved_shard_rows)
                values = np.asarray(source[start:stop, channel : channel + 1, :, :])
                destination[start:stop, channel : channel + 1, :, :] = values
    write_seconds = float(time.perf_counter() - write_started)

    validation_started = time.perf_counter()
    with profiler.time("output_postpack_validate", items=shape[0]):
        source_digest = _probability_array_digest(source, row_step=inner_rows)
        destination_digest = _probability_array_digest(destination, row_step=inner_rows)
    validation_seconds = float(time.perf_counter() - validation_started)
    if source_digest != destination_digest:
        raise RuntimeError(
            "Post-packed probability digest mismatch: "
            f"source={source_digest} destination={destination_digest}."
        )

    del run_group[source_name]
    return {
        "schema_id": MASK_PROBS_SHARDING_SCHEMA,
        "status": "complete",
        "source_working_array_removed": True,
        "canonical_array": MASK_PROBS_CANONICAL_ARRAY,
        "row_count": shape[0],
        "channel_count": shape[1],
        "inner_chunk_shape": list(chunks),
        "outer_shard_shape": list(shards),
        "inner_chunks_per_shard": int(resolved_shard_rows // inner_rows),
        "write_seconds": write_seconds,
        "validation_seconds": validation_seconds,
        "source_sha256": source_digest,
        "destination_sha256": destination_digest,
        "exact_match": True,
    }


def _raise_async_writer_error(errors: Sequence[BaseException]) -> None:
    if errors:
        raise RuntimeError("Async subject-mask output writer failed.") from errors[0]


def _write_subject_mask_outputs(
    run_group: zarr.Group,
    model: UNetSmall,
    roi_source: CropImageSource,
    *,
    batch_size: int,
    device: torch.device,
    mask_labels: Sequence[str],
    mask_probs_chunk_rois: Optional[int],
    mask_probs_shard_rois: Optional[int],
    mask_probs_dtype: str,
    write_masks_roi: bool,
    async_output: bool,
    output_queue_size: int,
    model_input_transform: ModelInputTransform,
    show_progress: bool,
    console: Console,
    timing_profiler: Optional[InferenceTimingProfiler],
    input_pixels_sha256: Any | None = None,
    validation_accumulators: (
        Mapping[str, SubjectMaskArrayUnitAccumulator] | None
    ) = None,
    mask_probs_destination_validation: str = (MASK_PROBS_DESTINATION_VALIDATION_FULL),
) -> float:
    total_rois = int(roi_source.total_rois)
    height, width = map(int, roi_source.roi_shape)
    n_channels = int(len(mask_labels))
    storage_chunks = subject_mask_storage_chunks(total_rois, height, width)
    if mask_probs_chunk_rois is not None:
        storage_chunks = (
            max(1, min(int(mask_probs_chunk_rois), max(1, total_rois))),
            storage_chunks[1],
            storage_chunks[2],
            storage_chunks[3],
        )
    if mask_probs_shard_rois is not None:
        shard_rows = int(mask_probs_shard_rois)
        if (
            shard_rows <= int(storage_chunks[0])
            or shard_rows % int(storage_chunks[0]) != 0
        ):
            raise ValueError(
                "--mask-probs-shard-rois must exceed and be an integer multiple of "
                f"the effective inner chunk rows ({int(storage_chunks[0])}); got {shard_rows}."
            )
    elif mask_probs_destination_validation != MASK_PROBS_DESTINATION_VALIDATION_FULL:
        raise ValueError(
            "Deferred probability destination validation requires indexed sharding."
        )
    metric_row_chunk = subject_mask_metric_row_chunk(total_rois)

    roi_array = roi_source.roi_array
    compression_kwargs = _compression_kwargs(roi_array) if roi_array is not None else {}
    stored_prob_dtype = np.dtype(
        np.uint8 if mask_probs_dtype == "uint8" else np.float16
    )
    profiler = timing_profiler or InferenceTimingProfiler(enabled=False)

    masks_arr: Optional[zarr.Array] = None
    if write_masks_roi:
        masks_arr = run_group.create_array(
            "masks_roi",
            shape=(total_rois, n_channels, height, width),
            dtype=np.uint8,
            chunks=storage_chunks,
            fill_value=0,
            overwrite=True,
            **compression_kwargs,
        )
    probability_shard_writer: _DoubleBufferedProbabilityShardWriter | None = None
    if mask_probs_shard_rois is None:
        probs_arr = run_group.create_array(
            MASK_PROBS_CANONICAL_ARRAY,
            shape=(total_rois, n_channels, height, width),
            dtype=stored_prob_dtype,
            chunks=storage_chunks,
            fill_value=np.uint8(0) if mask_probs_dtype == "uint8" else np.float16(0.0),
            overwrite=True,
            **compression_kwargs,
        )
    else:
        outer_shards = (int(mask_probs_shard_rois), 1, height, width)
        probability_destination = run_group.create_array(
            MASK_PROBS_CANONICAL_ARRAY,
            shape=(total_rois, n_channels, height, width),
            dtype=stored_prob_dtype,
            chunks=storage_chunks,
            shards=outer_shards,
            fill_value=np.uint8(0) if mask_probs_dtype == "uint8" else np.float16(0.0),
            overwrite=True,
            **compression_kwargs,
        )
        probability_destination.attrs.update(
            {
                "storage_layout": "indexed_sharding_v1",
                "inner_chunk_shape": list(storage_chunks),
                "outer_shard_shape": list(outer_shards),
                "write_mode": "double_buffered_direct",
                "buffer_count": 2,
            }
        )
        probability_shard_writer = _DoubleBufferedProbabilityShardWriter(
            probability_destination,
            shard_rows=int(mask_probs_shard_rois),
            profiler=profiler,
            buffer_count=2,
            destination_validation_mode=mask_probs_destination_validation,
        )
        probs_arr = probability_shard_writer
    run_group.create_array(
        "available_channels",
        data=np.ones((n_channels,), dtype=bool),
        overwrite=True,
    )

    metrics_group = run_group.require_group("metrics")
    prob_max = np.zeros((total_rois, n_channels), dtype=np.float32)
    mask_present = np.zeros((total_rois, n_channels), dtype=bool)
    area_px = np.zeros((total_rois, n_channels), dtype=np.float32)
    centroid_xy = np.zeros((total_rois, n_channels, 2), dtype=np.float32)
    centroid_valid = np.zeros((total_rois, n_channels), dtype=bool)
    bbox_xyxy = np.zeros((total_rois, n_channels, 4), dtype=np.float32)
    bbox_valid = np.zeros((total_rois, n_channels), dtype=bool)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not show_progress,
    )
    task = progress.add_task("[cyan]Running inference[/cyan]", total=total_rois)

    def _sync_cuda(stage: str, *, items: int) -> None:
        profiler = timing_profiler or InferenceTimingProfiler(enabled=False)
        if profiler.enabled and device.type == "cuda":
            with profiler.time(stage, items=items):
                torch.cuda.synchronize(device)

    output_queue: Optional[Queue[object]] = None
    output_worker: Optional[Thread] = None
    output_sentinel = object()
    output_errors: list[BaseException] = []
    if async_output:
        output_queue = Queue(maxsize=max(1, int(output_queue_size)))

        def _output_worker() -> None:
            failed = False
            while True:
                item = output_queue.get()
                try:
                    if item is output_sentinel:
                        return
                    if failed:
                        continue
                    assert isinstance(item, _SubjectMaskOutputBatch)
                    _write_subject_mask_output_batch(
                        item,
                        probs_arr=probs_arr,
                        masks_arr=masks_arr,
                        prob_max=prob_max,
                        mask_present=mask_present,
                        area_px=area_px,
                        centroid_xy=centroid_xy,
                        centroid_valid=centroid_valid,
                        bbox_xyxy=bbox_xyxy,
                        bbox_valid=bbox_valid,
                        progress=progress,
                        task=task,
                        profiler=profiler,
                        validation_accumulators=validation_accumulators,
                    )
                except (
                    BaseException
                ) as exc:  # pragma: no cover - exercised through caller failure
                    output_errors.append(exc)
                    failed = True
                finally:
                    output_queue.task_done()

        output_worker = Thread(
            target=_output_worker,
            name="subject-mask-output-writer",
            daemon=True,
        )

    stage_start = time.perf_counter()
    with progress, torch.no_grad():
        if output_worker is not None:
            output_worker.start()
        try:
            for start in range(0, total_rois, batch_size):
                stop = min(start + batch_size, total_rois)
                batch_count = stop - start

                _raise_async_writer_error(output_errors)
                with profiler.time("roi_read", items=batch_count):
                    roi_np = roi_source.read_slice(start, stop)
                if input_pixels_sha256 is not None:
                    roi_bytes = np.ascontiguousarray(roi_np)
                    if roi_bytes.dtype != np.dtype(np.uint8):
                        raise ValueError(
                            "Subject-mask ROI pixel identity requires decoded uint8 inputs."
                        )
                    input_pixels_sha256.update(roi_bytes.view(np.uint8))

                _sync_cuda("sync_before_h2d", items=batch_count)
                with profiler.time("h2d_copy", items=batch_count):
                    imgs = torch.from_numpy(roi_np).to(device, non_blocking=True)
                _sync_cuda("sync_after_h2d", items=batch_count)
                with profiler.time("input_normalize", items=batch_count):
                    imgs = _normalise_roi_tensor(imgs)
                with profiler.time("input_transform", items=batch_count):
                    imgs = model_input_transform.apply_torch_image_batch(imgs)
                _sync_cuda("sync_after_normalize", items=batch_count)

                amp_module = getattr(torch, "amp", None)
                if (
                    device.type == "cuda"
                    and amp_module is not None
                    and hasattr(amp_module, "autocast")
                ):
                    autocast_cm = amp_module.autocast("cuda")
                elif device.type == "cuda" and hasattr(torch.cuda, "amp"):
                    autocast_cm = torch.cuda.amp.autocast()
                else:
                    autocast_cm = nullcontext()

                _sync_cuda("sync_before_forward", items=batch_count)
                with profiler.time("model_forward", items=batch_count):
                    with autocast_cm:
                        logits = model(imgs)
                    logits = model_input_transform.crop_torch_output(logits)
                _sync_cuda("sync_after_forward", items=batch_count)

                _sync_cuda("sync_before_d2h", items=batch_count)
                with profiler.time("d2h_copy", items=batch_count):
                    probs_out, binary, output_metrics = _postprocess_logits_on_device(
                        logits,
                        mask_probs_dtype=mask_probs_dtype,
                        return_binary=write_masks_roi,
                    )
                _sync_cuda("sync_after_d2h", items=batch_count)

                if probs_out.ndim == 3:
                    probs_out = probs_out[:, None, :, :]
                if binary is not None and binary.ndim == 3:
                    binary = binary[:, None, :, :]
                if probs_out.shape[1] != n_channels:
                    raise ValueError(
                        f"Checkpoint/model produced {probs_out.shape[1]} channels but expected {n_channels}."
                    )

                with profiler.time("output_postprocess", items=batch_count):
                    if mask_probs_dtype == "uint8":
                        probs_out = probs_out.astype(np.uint8, copy=False)
                    else:
                        probs_out = probs_out.astype(np.float16, copy=False)
                    if binary is not None:
                        binary = binary.astype(np.uint8, copy=False)

                output_batch = _SubjectMaskOutputBatch(
                    start=start,
                    stop=stop,
                    probs_out=probs_out,
                    binary=binary,
                    metrics=output_metrics,
                )
                if output_queue is None:
                    _write_subject_mask_output_batch(
                        output_batch,
                        probs_arr=probs_arr,
                        masks_arr=masks_arr,
                        prob_max=prob_max,
                        mask_present=mask_present,
                        area_px=area_px,
                        centroid_xy=centroid_xy,
                        centroid_valid=centroid_valid,
                        bbox_xyxy=bbox_xyxy,
                        bbox_valid=bbox_valid,
                        progress=progress,
                        task=task,
                        profiler=profiler,
                        validation_accumulators=validation_accumulators,
                    )
                else:
                    _raise_async_writer_error(output_errors)
                    _freeze_subject_mask_output_batch(output_batch)
                    with profiler.time("output_queue_put", items=batch_count):
                        output_queue.put(output_batch)
        finally:
            if output_queue is not None:
                with profiler.time("output_queue_drain", items=total_rois):
                    output_queue.put(output_sentinel)
                    output_queue.join()
            if output_worker is not None:
                output_worker.join()
        _raise_async_writer_error(output_errors)

    metrics_group.create_array(
        "prob_max", data=prob_max, chunks=(metric_row_chunk, n_channels), overwrite=True
    )
    metrics_group.create_array(
        "mask_present",
        data=mask_present,
        chunks=(metric_row_chunk, n_channels),
        overwrite=True,
    )
    metrics_group.create_array(
        "area_px", data=area_px, chunks=(metric_row_chunk, n_channels), overwrite=True
    )
    metrics_group.create_array(
        "centroid_xy",
        data=centroid_xy,
        chunks=(metric_row_chunk, n_channels, 2),
        overwrite=True,
    )
    metrics_group.create_array(
        "centroid_valid",
        data=centroid_valid,
        chunks=(metric_row_chunk, n_channels),
        overwrite=True,
    )
    metrics_group.create_array(
        "bbox_xyxy",
        data=bbox_xyxy,
        chunks=(metric_row_chunk, n_channels, 4),
        overwrite=True,
    )
    metrics_group.create_array(
        "bbox_valid",
        data=bbox_valid,
        chunks=(metric_row_chunk, n_channels),
        overwrite=True,
    )
    if probability_shard_writer is not None:
        run_group.attrs["mask_probs_shard_write"] = probability_shard_writer.finish(
            validation_row_step=int(storage_chunks[0]),
        )
    return float(time.perf_counter() - stage_start)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer unified subject-mask probabilities using a trained U-Net segmenter."
    )
    parser.add_argument("zarr_path", help="Path to Palette Zarr archive.")
    parser.add_argument(
        "checkpoint", nargs="?", help="Path to trained U-Net checkpoint (.pt)."
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_option",
        help="Path to trained U-Net checkpoint (.pt).",
    )
    parser.add_argument(
        "--resolve-model-from-registry",
        action="store_true",
        help="Resolve the checkpoint from subject_mask_training_models instead of passing a path.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Registry SQLite path for --resolve-model-from-registry.",
    )
    parser.add_argument(
        "--model-set-id",
        help="Require one exact subject-mask registry set id.",
    )
    parser.add_argument(
        "--model-run-id",
        help="Require one exact subject-mask registry run id.",
    )
    parser.add_argument(
        "--model-coverage-class",
        default="dense_all_components",
        help="Required subject-mask model coverage_class when resolving from registry.",
    )
    parser.add_argument(
        "--model-component-coverage-key",
        help="Optional component_coverage_key filter when resolving from registry.",
    )
    parser.add_argument(
        "--model-label-schema-id",
        help="Optional label_schema_id filter when resolving from registry.",
    )
    parser.add_argument(
        "--model-include-non-success",
        action="store_true",
        help="Allow non-success registry model rows during model resolution.",
    )
    parser.add_argument(
        "--model-allow-missing-path",
        action="store_true",
        help="Do not filter resolved registry candidates whose model_path is missing on disk.",
    )
    parser.add_argument(
        "--model-require-unique",
        action="store_true",
        help="Fail registry model resolution when the top metric is tied.",
    )
    parser.add_argument(
        "--model-top-k",
        type=int,
        default=5,
        help="Number of registry candidates to retain in model-resolution provenance.",
    )
    parser.add_argument(
        "--crop-run",
        help="Explicit crop run providing ROI images (default: latest/auto).",
    )
    parser.add_argument(
        "--geometry-crop-run",
        help=(
            "Optional strict crop-v2 geometry authority for authenticated pixels "
            "opened from --crop-run. Rows are rebound only after exact instance-key, "
            "frame, and placement validation."
        ),
    )
    parser.add_argument(
        "--require-training-materialization-binding",
        action="store_true",
        help=(
            "Require --crop-run to be an exact self-contained training crop "
            "materialization. This may write only terminal subject-mask shards; "
            "canonical publication remains bound to source crop-v2."
        ),
    )
    parser.add_argument("--run-name", help="Optional name for the output run.")
    parser.add_argument(
        "--attempt-id",
        help="Optional UUID for this execution attempt; generated when omitted.",
    )
    parser.add_argument(
        "--retry-of-attempt-id",
        help="Optional prior failed attempt UUID with the same scientific identity.",
    )
    parser.add_argument(
        "--supersedes-run",
        help="Optional explicit predecessor run name; latest is never inferred.",
    )
    parser.add_argument(
        "--output-parent",
        choices=SUBJECT_MASK_OUTPUT_PARENTS,
        default=SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
        help=(
            "Output parent group. Use subject_mask_shard_runs for clipped-collection "
            "shards that must not publish ordinary subject_mask_runs selectors."
        ),
    )
    parser.add_argument(
        "--source-collection-id",
        help="Optional clipped collection id for shard outputs.",
    )
    parser.add_argument(
        "--source-collection-path",
        help="Optional clipped collection path for shard outputs.",
    )
    parser.add_argument("--source-clip-id", help="Optional clip id for shard outputs.")
    parser.add_argument(
        "--source-clip-index", type=int, help="Optional clip index for shard outputs."
    )
    parser.add_argument(
        "--source-work-unit-id", help="Optional work-unit id for shard outputs."
    )
    parser.add_argument(
        "--source-shard-id", help="Optional shard id for non-clip collection shards."
    )
    parser.add_argument(
        "--expected-work-units-manifest",
        type=Path,
        help=(
            "Exact single-unit recording plan for a complete flat-cache worker. "
            "Mutually exclusive with --roi-work-package-manifest."
        ),
    )
    parser.add_argument(
        "--source-roi-cache-alias-manifest",
        type=Path,
        help=(
            "Optional cache alias manifest used for this shard. Defaults to "
            "--roi-cache-manifest when omitted."
        ),
    )
    parser.add_argument(
        "--source-roi-cache-row-index-path",
        type=Path,
        help="Optional row-index parquet path for the clipped collection flat ROI cache shard.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used during inference (default: 256).",
    )
    parser.add_argument("--device", help="Torch device to use (e.g. 'cuda:0', 'cpu').")
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=None,
        help="Optional square model input size. When larger than native ROI size, pair with --model-input-transform auto/pad_to_size.",
    )
    parser.add_argument(
        "--model-input-transform",
        choices=MODEL_INPUT_TRANSFORM_CHOICES,
        default="auto",
        help=(
            "Reversible transform from native ROI crops to model input size. "
            "'auto' is identity when sizes match and centered zero-padding when --model-input-size is larger."
        ),
    )
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-cache-dir",
        type=Path,
        default=None,
        help="Optional scratch directory for temporary ROI caches.",
    )
    roi_manifest_group = parser.add_mutually_exclusive_group()
    roi_manifest_group.add_argument(
        "--roi-cache-manifest",
        type=Path,
        default=None,
        help="Optional flat_bin_v1 ROI cache manifest to read instead of materializing/re-decoding ROIs.",
    )
    roi_manifest_group.add_argument(
        "--roi-work-package-manifest",
        type=Path,
        default=None,
        help=(
            "Keyed subset ROI package for delta or proven complete-partition "
            "inference. Requires --output-parent subject_mask_shard_runs."
        ),
    )
    parser.add_argument(
        "--roi-work-package-role",
        choices=ROI_WORK_PACKAGE_ROLES,
        default=None,
        help=(
            "Publication role for --roi-work-package-manifest. Defaults to "
            "delta_replacement_rows. complete_collection_partition additionally "
            "proves exact frame-offset coverage before inference."
        ),
    )
    parser.add_argument(
        "--roi-cache-expected-archive-path",
        type=Path,
        default=None,
        help=(
            "Canonical analysis zarr path expected by --roi-cache-manifest. "
            "Use when writing into a staged zarr overlay whose logical source "
            "archive is the original recording zarr."
        ),
    )
    parser.add_argument("--source-crop-row-start", type=int, default=None)
    parser.add_argument("--source-crop-row-stop", type=int, default=None)
    parser.add_argument(
        "--roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Live ROI read acceleration for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads (default: 32).",
    )
    parser.add_argument(
        "--mask-probs-chunk-rois",
        type=int,
        default=32,
        help="ROI chunk length override for mask_probs_roi and masks_roi outputs (default: 32).",
    )
    probability_storage = parser.add_mutually_exclusive_group()
    probability_storage.add_argument(
        "--mask-probs-shard-rois",
        type=int,
        default=DEFAULT_MASK_PROBS_SHARD_ROIS,
        help=(
            "Outer storage-shard row count for mask_probs_roi (default: 2048). Two host-memory buffers "
            "accumulate inference batches and write each complete indexed shard once; the final "
            "destination follows --mask-probs-destination-validation."
        ),
    )
    probability_storage.add_argument(
        "--no-mask-probs-sharding",
        dest="mask_probs_shard_rois",
        action="store_const",
        const=None,
        help="Use ordinary mask_probs_roi chunks instead of the default indexed-sharded layout.",
    )
    parser.add_argument(
        "--mask-probs-dtype",
        choices=("float16", "uint8"),
        default="uint8",
        help="Storage dtype for mask_probs_roi (default: uint8 for analysis runs).",
    )
    parser.add_argument(
        "--mask-probs-destination-validation",
        choices=MASK_PROBS_DESTINATION_VALIDATION_MODES,
        default=MASK_PROBS_DESTINATION_VALIDATION_FULL,
        help=(
            "Validation of the completed probability destination. The default "
            "rereads and hashes the decoded array. receipt_bound_final_layout_unit_v1 "
            "is restricted to complete, non-authoritative collection partitions whose "
            "caller must build and verify a final-layout unit before publication."
        ),
    )
    parser.add_argument(
        "--write-masks-roi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Materialize thresholded binary masks_roi alongside mask_probs_roi "
            "(default: false for probability-first raw runs; use --write-masks-roi for compatibility output)."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the Rich progress bar during inference (default: false for log-friendly runs).",
    )
    parser.add_argument(
        "--async-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlap model inference with a bounded background writer for Zarr output and spatial metrics (default: true).",
    )
    parser.add_argument(
        "--output-queue-size",
        type=int,
        default=2,
        help="Maximum number of dense output batches buffered by --async-output (default: 2).",
    )
    parser.add_argument(
        "--assignment-keypoint-group",
        choices=KEYPOINT_GROUP_CHOICES,
        help="Keypoint group to use later when splitting eyes_union into anatomical LR eyes.",
    )
    parser.add_argument(
        "--assignment-keypoint-run",
        help="Keypoint run to use later when splitting eyes_union into anatomical LR eyes.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Collect per-stage timing diagnostics.",
    )
    parser.add_argument(
        "--defer-registry-status",
        action="store_true",
        help=(
            "Write and complete the zarr run group without emitting registry "
            "step status. Use for local staged output that will be published to "
            "the canonical archive before status is emitted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the requested output run if it exists.",
    )
    return parser


def _resolve_registry_checkpoint(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        selected, candidates = resolve_best_subject_mask_model(
            registry,
            set_id=args.model_set_id,
            run_id=args.model_run_id,
            coverage_class=args.model_coverage_class,
            component_coverage_key=args.model_component_coverage_key,
            label_schema_id=args.model_label_schema_id,
            include_non_success=bool(args.model_include_non_success),
            require_existing_path=not bool(args.model_allow_missing_path),
            require_unique=bool(args.model_require_unique),
        )
    finally:
        registry.close()

    payload = build_resolution_payload(
        registry_path=registry_path,
        selected=selected,
        candidates=candidates,
        top_k=int(args.model_top_k),
        parameters={
            "set_id": args.model_set_id,
            "run_id": args.model_run_id,
            "coverage_class": args.model_coverage_class,
            "component_coverage_key": args.model_component_coverage_key,
            "label_schema_id": args.model_label_schema_id,
            "include_non_success": bool(args.model_include_non_success),
            "require_existing_path": not bool(args.model_allow_missing_path),
            "require_unique": bool(args.model_require_unique),
            "top_k": int(args.model_top_k),
        },
    )
    return Path(selected.model_path).expanduser().resolve(), payload


@_fail_closed_subject_mask_attempt
def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    console.print("[bold cyan]Running U-Net subject-mask inference[/bold cyan]\n")

    output_parent = _output_parent_from_args(args)
    canonical_output = output_parent == SUBJECT_MASK_CANONICAL_OUTPUT_PARENT
    if args.expected_work_units_manifest is not None and (
        args.roi_work_package_manifest is not None
        or args.roi_work_package_role is not None
    ):
        raise ValueError(
            "--expected-work-units-manifest is mutually exclusive with crop "
            "work-package inference."
        )
    if args.expected_work_units_manifest is not None and not _is_shard_output_parent(
        output_parent
    ):
        raise ValueError(
            "Recording work-unit inference may write only to "
            "subject_mask_shard_runs before recording-level publication."
        )
    if (
        args.expected_work_units_manifest is not None
        and args.roi_cache_manifest is None
    ):
        raise ValueError("Recording work-unit inference requires --roi-cache-manifest.")
    if args.roi_work_package_manifest is not None and not _is_shard_output_parent(
        output_parent
    ):
        raise ValueError(
            "Crop pixel work packages may write only to subject_mask_shard_runs; "
            "finalize shards before publishing a canonical subject-mask run."
        )
    if (args.source_crop_row_start is None) != (args.source_crop_row_stop is None):
        raise ValueError(
            "--source-crop-row-start and --source-crop-row-stop must be provided together."
        )
    if args.source_crop_row_start is not None:
        if (
            args.roi_cache_manifest is not None
            or args.roi_work_package_manifest is not None
        ):
            raise ValueError(
                "Direct crop-row partitions cannot be combined with ROI cache or work-package manifests."
            )
        if not _is_shard_output_parent(output_parent):
            raise ValueError(
                "Direct crop-row partitions may write only subject_mask_shard_runs outputs."
            )
    if (
        args.roi_work_package_role is not None
        and args.roi_work_package_manifest is None
    ):
        raise ValueError(
            "--roi-work-package-role requires --roi-work-package-manifest."
        )
    if args.roi_work_package_manifest is not None and (
        args.assignment_keypoint_group or args.assignment_keypoint_run
    ):
        raise ValueError(
            "Subset subject-mask inference does not consume assignment keypoints. "
            "Bind the complete refined-keypoint run during mask finalization."
        )
    if canonical_output and args.roi_cache_manifest is not None:
        raise ValueError(
            "Canonical subject-mask inference requires direct persisted crop "
            "roi_images and rejects ROI cache manifests."
        )
    if args.require_training_materialization_binding and canonical_output:
        raise ValueError(
            "A training-materialized crop may feed only non-authoritative terminal "
            "subject-mask arrays. Canonical publication must finalize against the "
            "bound source crop-v2 authority."
        )
    if args.require_training_materialization_binding and not args.crop_run:
        raise ValueError(
            "Strict training materialization input requires an explicit --crop-run."
        )
    if (
        args.mask_probs_destination_validation
        == MASK_PROBS_DESTINATION_VALIDATION_FINAL_LAYOUT
    ) and (
        canonical_output
        or not _is_shard_output_parent(output_parent)
        or args.roi_work_package_manifest is None
        or args.roi_work_package_role != ROI_WORK_PACKAGE_ROLE_COMPLETE_PARTITION
        or args.mask_probs_shard_rois is None
    ):
        raise ValueError(
            "receipt_bound_final_layout_unit_v1 is allowed only for an indexed-sharded "
            "subject_mask_shard_runs output backed by an exact "
            "complete_collection_partition work package."
        )
    zarr_path = Path(args.zarr_path).expanduser().resolve()
    training_materialization = (
        bind_training_crop_materialization(zarr_path, run_id=str(args.crop_run))
        if args.require_training_materialization_binding
        else None
    )

    checkpoint_value = args.checkpoint_option or args.checkpoint
    if checkpoint_value and args.resolve_model_from_registry:
        raise ValueError(
            "Pass either a checkpoint path or --resolve-model-from-registry, not both."
        )
    model_resolution_payload: Optional[dict[str, Any]] = None
    if checkpoint_value:
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    elif args.resolve_model_from_registry:
        checkpoint_path, model_resolution_payload = _resolve_registry_checkpoint(args)
        selected = model_resolution_payload.get("selected", {})
        console.print(
            "[dim]Resolved subject-mask model from registry: "
            f"{selected.get('run_id')} -> {checkpoint_path}[/dim]"
        )
    else:
        raise ValueError(
            "U-Net subject-mask inference requires a checkpoint path or --resolve-model-from-registry."
        )
    selected_model = (
        model_resolution_payload.get("selected", {})
        if model_resolution_payload is not None
        else {}
    )
    if not isinstance(selected_model, dict):
        selected_model = {}
    registry_checkpoint_hash = (
        selected_model.get("model_sha256")
        if isinstance(selected_model.get("model_sha256"), str)
        else None
    )
    if model_resolution_payload is not None and registry_checkpoint_hash is None:
        raise ValueError(
            "Registry-selected subject-mask models require an exact model_sha256."
        )
    checkpoint_artifact_before_load = require_artifact_content_identity(
        checkpoint_path,
        role="subject_mask_unet_checkpoint",
        expected_sha256=registry_checkpoint_hash,
    )
    device = _resolve_device(args.device)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    checkpoint_artifact_after_load = require_artifact_content_identity(
        checkpoint_path,
        role="subject_mask_unet_checkpoint",
        expected_sha256=registry_checkpoint_hash,
    )
    artifact_identity_fields = (
        "path",
        "fingerprint_scheme",
        "sha256",
        "size_bytes",
        "mtime_ns",
    )
    if any(
        checkpoint_artifact_before_load.get(name)
        != checkpoint_artifact_after_load.get(name)
        for name in artifact_identity_fields
    ):
        raise RuntimeError(
            "Subject-mask checkpoint changed while it was being loaded; refusing "
            "to publish ambiguous model lineage."
        )
    checkpoint_artifact = checkpoint_artifact_before_load

    label_schema_id, mask_labels = _resolve_checkpoint_schema(checkpoint)

    root = zarr.open(str(zarr_path), mode="a", use_consolidated=False)

    if args.roi_work_package_manifest is not None:
        crop_source = CropImageSource.open_work_package(
            root,
            manifest_path=args.roi_work_package_manifest,
            zarr_path=zarr_path,
            crop_run=args.crop_run,
        )
    else:
        crop_source = CropImageSource.open(
            root,
            crop_run=args.crop_run,
            # Canonical publication must retain the archive-root identity of the
            # exact crop_runs/<run>/roi_images owner.
            zarr_path=None if canonical_output else zarr_path,
            roi_cache_policy=args.roi_cache_policy,
            roi_live_acceleration=args.roi_live_acceleration,
            roi_live_gpu_chunk_frames=args.roi_live_gpu_chunk_frames,
            roi_cache_dir=args.roi_cache_dir,
            roi_cache_manifest=args.roi_cache_manifest,
            roi_cache_expected_archive_path=args.roi_cache_expected_archive_path,
            source_crop_row_start=args.source_crop_row_start,
            source_crop_row_stop=args.source_crop_row_stop,
            console=console,
        )
    if args.geometry_crop_run is not None:
        crop_source.bind_geometry_crop(
            args.geometry_crop_run,
            zarr_path=zarr_path,
        )
    boundary = _ACTIVE_SUBJECT_MASK_ATTEMPT.get()
    if boundary is not None:
        boundary.bind_crop_source(crop_source)
    crop_group = crop_source.crop_group
    crop_run_name = crop_source.crop_run_name
    selected_crop_rows = getattr(crop_source, "source_crop_row_ids", None)
    total_rois = int(crop_source.total_rois)
    if training_materialization is not None:
        if (
            crop_run_name != training_materialization.run_id
            or total_rois != training_materialization.row_count
            or tuple(int(value) for value in crop_source.roi_shape)
            != training_materialization.roi_shape
            or getattr(crop_source, "frame_source_kind", None) != "roi_images"
            or bool(getattr(crop_source, "roi_cache_used", False))
        ):
            raise ValueError(
                "Active subject-mask pixel source differs from the strict training "
                "crop materialization binding."
            )
    if total_rois == 0:
        if boundary is not None:
            boundary.close_crop_source()
        else:  # pragma: no cover - public writer is boundary-decorated
            crop_source.close()
        raise ValueError("ROI image array is empty; nothing to segment.")
    work_package_attrs = _roi_work_package_publication_attrs(
        crop_group=crop_group,
        crop_source=crop_source,
        selected_crop_rows=selected_crop_rows,
        total_rois=total_rois,
        args=args,
    )
    canonical_crop_source = None
    canonical_selected: dict[str, np.ndarray] | None = None
    if canonical_output:
        canonical_crop_path = f"crop_runs/{crop_run_name}"
        if str(getattr(crop_group, "path", "")).strip("/") != canonical_crop_path:
            raise ValueError(
                "Canonical subject-mask inference requires the exact selected "
                f"persisted crop rowset at {canonical_crop_path!r}."
            )
        if selected_crop_rows is not None:
            raise ValueError(
                "Canonical subject-mask inference requires the direct complete "
                "materialized crop rowset and rejects selected-row proxy sources."
            )
        if (
            getattr(crop_source, "storage_mode", None) != "materialized"
            or getattr(crop_source, "frame_source_kind", None) != "roi_images"
            or getattr(crop_source, "roi_read_mode", None) != "materialized_crop_run"
            or bool(getattr(crop_source, "roi_cache_used", False))
        ):
            raise ValueError(
                "Canonical subject-mask inference requires direct root-owned "
                "materialized roi_images; caches, live/composite pixels, and work "
                "packages are unsupported."
            )
        canonical_crop_source = load_persisted_subject_mask_crop_source(
            root,
            canonical_crop_path,
        )
        require_direct_subject_mask_crop_pixel_source(
            canonical_crop_source,
            getattr(crop_source, "_roi_images", None),
        )
        canonical_rows = np.arange(
            canonical_crop_source.crop_geometry.row_identity.leading_dimension,
            dtype="<i8",
        )
        canonical_selected = selected_subject_mask_crop_values(
            canonical_crop_source,
            canonical_rows,
        )
        active_placement = np.asarray(crop_source.roi_coordinates_full)
        source_placement = canonical_selected["source_crop_xywh"]
        active_frames = np.asarray(crop_source.frame_indices, dtype="<i8")
        expected_roi_shape = (
            int(canonical_crop_source.roi_frame.endpoint.height),
            int(canonical_crop_source.roi_frame.endpoint.width),
        )
        if (
            total_rois != int(canonical_rows.shape[0])
            or tuple(map(int, crop_source.roi_shape)) != expected_roi_shape
            or active_placement.shape != source_placement[:, :2].shape
            or not np.array_equal(active_placement, source_placement[:, :2])
            or not np.array_equal(
                active_frames,
                canonical_selected["source_acquisition_frame_index"],
            )
        ):
            raise ValueError(
                "Active crop pixels/rows do not equal the exact persisted canonical "
                "crop selection, placement, ROI extent, and acquisition mapping."
            )
    roi_height, roi_width = map(int, crop_source.roi_shape)
    model_input_size = (
        int(args.model_input_size)
        if args.model_input_size is not None
        else max(roi_height, roi_width)
    )
    model_input_transform = resolve_model_input_transform(
        (roi_height, roi_width),
        mode=str(args.model_input_transform),
        model_hw=(model_input_size, model_input_size),
    )
    try:
        assignment_keypoint_attrs = _resolve_assignment_keypoint_attrs(
            root,
            assignment_keypoint_group=args.assignment_keypoint_group,
            assignment_keypoints_run=args.assignment_keypoint_run,
            total_rois=total_rois,
            mask_labels=mask_labels,
        )
    except Exception:
        if boundary is not None:
            boundary.close_crop_source()
        else:  # pragma: no cover - public writer is boundary-decorated
            crop_source.close()
        raise

    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    git_info = get_git_info()
    shard_output = _is_shard_output_parent(output_parent)
    run_group, resolved_run_name = _prepare_run_group(
        root,
        run_name=args.run_name,
        overwrite=bool(args.overwrite),
        output_parent=output_parent,
    )
    shard_attrs = _shard_attrs_from_args(args, output_parent=output_parent)
    timing_profiler = InferenceTimingProfiler(enabled=bool(args.profile_timings))
    canonical_coordinate_context = None
    package_selection: dict[str, np.ndarray] | None = None

    try:
        if canonical_crop_source is not None:
            assert canonical_selected is not None
            written_selection = _write_canonical_subject_mask_selection(
                run_group,
                canonical_crop_source,
            )
            for name in (
                "source_crop_row_ids",
                "instance_key",
                "source_acquisition_frame_index",
                "source_crop_xywh",
            ):
                if not np.array_equal(
                    written_selection[name], canonical_selected[name]
                ):
                    raise RuntimeError(
                        f"Canonical subject-mask selection changed while writing {name}."
                    )
            run_group.attrs["mask_labels"] = list(mask_labels)
            run_group.attrs["label_schema_id"] = label_schema_id
            run_group.attrs["model_input_transform"] = model_input_transform.to_attrs()
            run_group.attrs["mask_probability_threshold"] = 0.5
            run_group.attrs["source_checkpoint"] = str(checkpoint_path)
            run_group.attrs["subject_mask_model_artifact"] = checkpoint_artifact
            if boundary is None or boundary.owner_token is None:
                raise RuntimeError("Canonical subject-mask preflight lacks ownership.")
            canonical_coordinate_context = prepare_subject_mask_coordinate_context(
                root,
                f"subject_mask_runs/{resolved_run_name}",
                expected_publication_owner=boundary.owner_token,
                crop_path=f"crop_runs/{crop_run_name}",
                mask_labels=mask_labels,
                model_input_transform=model_input_transform,
                model_artifact=checkpoint_artifact,
                mask_probability_threshold=0.5,
            )
            # Preflight reloads through the root; keep using that exact handle.
            run_group = boundary.require_owned_run()
        else:
            if selected_crop_rows is not None:
                copy_selected_crop_row_lineage_arrays(
                    run_group,
                    crop_group,
                    selected_crop_rows,
                )
                if (
                    getattr(crop_source, "pixel_materialization_id", None) is not None
                    or getattr(crop_source, "geometry_crop_rebase", None) is not None
                ):
                    package_selection = _write_package_subject_mask_crop_placement(
                        run_group,
                        crop_group,
                        selected_crop_rows,
                    )
            else:
                copy_row_lineage_arrays(run_group, crop_group, total_rois=total_rois)
                write_direct_source_crop_row_ids(run_group, total_rois=total_rois)
        if (
            getattr(crop_source, "pixel_materialization_id", None) is not None
            or getattr(crop_source, "geometry_crop_rebase", None) is not None
        ):
            if selected_crop_rows is None:
                raise ValueError(
                    "Package-backed subject-mask inference lacks selected crop rows."
                )
            copy_selected_row_source_signatures(
                run_group,
                crop_group,
                selected_crop_rows,
                shard_rows=int(
                    args.mask_probs_shard_rois or DEFAULT_MASK_PROBS_SHARD_ROIS
                ),
                root=root,
            )
        if not canonical_output:
            _copy_detection_source_array(
                run_group,
                crop_group,
                source_crop_row_ids=selected_crop_rows,
            )
            if "detection_source" not in run_group:
                run_group.create_array(
                    "detection_source",
                    data=np.zeros((total_rois,), dtype=np.int8),
                    overwrite=True,
                )

        validation_unit_rows = int(
            args.mask_probs_shard_rois
            if args.mask_probs_shard_rois is not None
            else args.mask_probs_chunk_rois
        )
        validation_accumulators = _raw_worker_validation_accumulators(
            total_rois=total_rois,
            n_channels=len(mask_labels),
            height=roi_height,
            width=roi_width,
            probability_dtype=np.dtype(
                np.uint8 if args.mask_probs_dtype == "uint8" else np.float16
            ),
            write_masks_roi=bool(args.write_masks_roi),
            unit_rows=validation_unit_rows,
        )
        input_pixels_sha256 = hashlib.sha256()
        duration = _write_subject_mask_outputs(
            run_group,
            model,
            crop_source,
            batch_size=int(args.batch_size),
            device=device,
            mask_labels=mask_labels,
            mask_probs_chunk_rois=args.mask_probs_chunk_rois,
            mask_probs_shard_rois=args.mask_probs_shard_rois,
            mask_probs_dtype=str(args.mask_probs_dtype),
            write_masks_roi=bool(args.write_masks_roi),
            async_output=bool(args.async_output),
            output_queue_size=int(args.output_queue_size),
            model_input_transform=model_input_transform,
            show_progress=bool(args.progress),
            console=console,
            timing_profiler=timing_profiler,
            input_pixels_sha256=input_pixels_sha256,
            validation_accumulators=validation_accumulators,
            mask_probs_destination_validation=str(
                args.mask_probs_destination_validation
            ),
        )
        if package_selection is not None:
            assert selected_crop_rows is not None
            _validate_package_subject_mask_selection(
                run_group,
                crop_group,
                selected_crop_rows,
                expected=package_selection,
            )
    finally:
        if boundary is not None:
            boundary.close_crop_source()
        else:  # pragma: no cover - public writer is boundary-decorated
            crop_source.close()

    observed_pixels_sha256 = input_pixels_sha256.hexdigest()
    scientific_identity = _subject_mask_scientific_documents(
        run_group=run_group,
        crop_group=crop_group,
        crop_source=crop_source,
        crop_run_name=str(crop_run_name),
        checkpoint_artifact=checkpoint_artifact,
        selected_model=selected_model,
        label_schema_id=label_schema_id,
        mask_labels=mask_labels,
        model_input_transform=model_input_transform,
        mask_probs_dtype=str(args.mask_probs_dtype),
        observed_pixels_sha256=observed_pixels_sha256,
        work_package_attrs=work_package_attrs,
        args=args,
    )
    attempt = build_subject_mask_attempt(
        scientific_identity=scientific_identity,
        run_path=f"{output_parent}/{resolved_run_name}",
        attempt_id=args.attempt_id,
        retry_of_attempt_id=args.retry_of_attempt_id,
        supersedes_run=args.supersedes_run,
    )
    lineage_evidence = _resolve_subject_mask_attempt_lineage(
        parent=root[output_parent],
        current_run_name=resolved_run_name,
        scientific_identity=scientific_identity,
        attempt=attempt,
        retry_of_attempt_id=args.retry_of_attempt_id,
        supersedes_run=args.supersedes_run,
    )
    run_group.attrs[SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR] = scientific_identity
    run_group.attrs[SUBJECT_MASK_ATTEMPT_ATTR] = attempt
    run_group.attrs[SUBJECT_MASK_ATTEMPT_LINEAGE_EVIDENCE_ATTR] = lineage_evidence
    run_path = f"{output_parent}/{resolved_run_name}"
    worker_receipt = _seal_raw_worker_semantic_receipt(
        run_group=run_group,
        run_path=run_path,
        scientific_identity=scientific_identity,
        attempt=attempt,
        accumulators=validation_accumulators,
        unit_rows=validation_unit_rows,
    )
    receipt_bytes = canonical_json_bytes(worker_receipt)
    receipt_relative_path = f"{run_path}/{SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SIDECAR}"
    receipt_path = Path(zarr_path) / receipt_relative_path
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt_path = receipt_path.with_name(
        f".{receipt_path.name}.{uuid4().hex}.tmp"
    )
    temporary_receipt_path.write_bytes(receipt_bytes)
    with temporary_receipt_path.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary_receipt_path, receipt_path)
    run_group.attrs[SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_ATTR] = {
        "schema_id": worker_receipt["schema_id"],
        "schema_version": worker_receipt["schema_version"],
        "payload_digest": worker_receipt["payload_digest"],
        "relative_path": receipt_relative_path,
        "document_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "storage": "strict_json_sidecar_v1",
    }

    created_at = datetime.now(timezone.utc).isoformat()
    crop_snapshot_attrs = build_source_crop_snapshot_attrs(
        crop_group.attrs,
        source_crop_storage_mode=crop_source.storage_mode,
    )
    crop_pixel_attrs = build_source_roi_pixel_attrs(crop_source)
    run_group.attrs.update(
        {
            "method": "unet_subject_mask_segmenter",
            "run_semantics": "unet_subject_mask_inference",
            "subject_mask_output_parent": output_parent,
            "source_crop_run": crop_run_name,
            **crop_snapshot_attrs,
            **crop_pixel_attrs,
            **work_package_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "source_roi_cache_used": bool(crop_source.roi_cache_used),
            "source_roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
            "source_roi_cache_canonical_path": getattr(
                crop_source, "roi_cache_canonical_path", None
            ),
            "source_roi_cache_expected_archive_path": (
                str(args.roi_cache_expected_archive_path)
                if args.roi_cache_expected_archive_path is not None
                else None
            ),
            "source_roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
            "source_roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
            "source_roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
            "source_roi_live_gpu_chunk_frames": int(
                crop_source.roi_live_gpu_chunk_frames
            ),
            "source_pixel_crop_run": getattr(
                crop_source, "pixel_source_crop_run_name", None
            ),
            "source_geometry_crop_rebase": getattr(
                crop_source, "geometry_crop_rebase", None
            ),
            "label_schema_id": label_schema_id,
            "mask_labels": list(mask_labels),
            "output_semantics": "multilabel",
            "overlap_policy": "independent_sigmoid",
            "probability_semantics": "sigmoid_multilabel_logits",
            "probabilities_dtype": str(args.mask_probs_dtype),
            "probabilities_encoding": (
                "linear_uint8_0_255"
                if args.mask_probs_dtype == "uint8"
                else "unit_float"
            ),
            "mask_probability_threshold": 0.5,
            "masks_roi_materialized": bool(args.write_masks_roi),
            "binary_masks_materialized": bool(args.write_masks_roi),
            "binary_masks_source": (
                "threshold(mask_probs_roi, threshold=0.5)"
                if args.write_masks_roi
                else "not_materialized"
            ),
            "bbox_xyxy_convention": "pixel_edge_half_open",
            "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
            "input_format": "gray",
            "model_input_transform": model_input_transform.to_attrs(),
            "model_input_transform_name": model_input_transform.name,
            "model_input_shape_hw": list(model_input_transform.model_shape),
            "native_roi_shape_hw": list(model_input_transform.native_shape),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_best_val_dice": float(
                checkpoint.get("best_val_dice", float("nan"))
            ),
            "source_roi_pixels_sha256": observed_pixels_sha256,
            SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR: scientific_identity,
            SUBJECT_MASK_ATTEMPT_ATTR: attempt,
            SUBJECT_MASK_ATTEMPT_LINEAGE_EVIDENCE_ATTR: lineage_evidence,
            "inference_device": str(device),
            "inference_batch_size": int(args.batch_size),
            "async_output": bool(args.async_output),
            "output_queue_size": int(args.output_queue_size),
            "semantic_receipt_hash_execution": (
                "ordered_async_output_worker_v1"
                if args.async_output
                else "ordered_inline_output_v1"
            ),
            "mask_probs_destination_validation": str(
                args.mask_probs_destination_validation
            ),
            "duration_seconds": float(duration),
            "inference_duration_seconds": float(duration),
            "profile_timings_enabled": bool(args.profile_timings),
            "created_at_utc": created_at,
            **shard_attrs,
        }
    )
    if assignment_keypoint_attrs:
        run_group.attrs.update(assignment_keypoint_attrs)
    if crop_source.roi_cache_key is not None:
        run_group.attrs["source_roi_cache_key"] = crop_source.roi_cache_key
    if crop_source.roi_cache_path is not None:
        run_group.attrs["source_roi_cache_path"] = crop_source.roi_cache_path
    if training_materialization is not None:
        run_group.attrs["source_training_crop_materialization_binding"] = dict(
            training_materialization.binding
        )
        run_group.attrs["source_training_crop_materialization_binding_digest"] = (
            training_materialization.binding["payload_digest"]
        )
    if args.mask_probs_chunk_rois is not None:
        run_group.attrs["mask_probs_chunk_rois"] = int(args.mask_probs_chunk_rois)
    if args.mask_probs_shard_rois is not None:
        run_group.attrs["mask_probs_shard_rois"] = int(args.mask_probs_shard_rois)
        run_group.attrs["mask_probs_storage_layout"] = "indexed_sharding_v1"
        run_group.attrs["mask_probs_storage_policy"] = "default_indexed_sharding_v1"
    else:
        run_group.attrs["mask_probs_storage_layout"] = "regular_chunks_v1"
        run_group.attrs["mask_probs_storage_policy"] = (
            "explicit_regular_chunks_override"
        )
    run_group.attrs["mask_probs_default_shard_rois"] = int(
        DEFAULT_MASK_PROBS_SHARD_ROIS
    )
    if model_resolution_payload is not None:
        selected = model_resolution_payload.get("selected", {})
        if not isinstance(selected, dict):
            selected = {}
        run_group.attrs["model_resolution_mode"] = "registry"
        run_group.attrs["model_resolution_task"] = "subject_masks"
        run_group.attrs["model_resolution_registry_path"] = (
            model_resolution_payload.get("registry_path")
        )
        run_group.attrs["model_resolution_resolved_at_utc"] = (
            model_resolution_payload.get("resolved_at_utc")
        )
        run_group.attrs["model_resolution_selected_run_id"] = selected.get("run_id")
        run_group.attrs["model_resolution_selected_set_id"] = selected.get("set_id")
        run_group.attrs["model_resolution_selected_model_path"] = selected.get(
            "model_path"
        )
        run_group.attrs["model_resolution_selected_coverage_class"] = selected.get(
            "coverage_class"
        )
        run_group.attrs["model_resolution_selected_component_coverage_key"] = (
            selected.get("component_coverage_key")
        )
        run_group.attrs["model_resolution_selected_metric_name"] = selected.get(
            "best_metric_name"
        )
        run_group.attrs["model_resolution_selected_metric_value"] = selected.get(
            "best_metric_value"
        )
        run_group.attrs["model_resolution_candidates_json"] = json.dumps(
            model_resolution_payload.get("candidates", []),
            sort_keys=True,
        )
    else:
        selected = {}

    metrics_group = run_group.get("metrics")
    mask_present_array = (
        metrics_group.get("mask_present") if metrics_group is not None else None
    )
    if mask_present_array is not None:
        mask_present_values = np.asarray(mask_present_array[:], dtype=bool)
        nonempty_rows = (
            np.any(mask_present_values, axis=1)
            if mask_present_values.ndim == 2
            else np.zeros((total_rois,), dtype=bool)
        )
    elif "masks_roi" in run_group:
        masks_array = np.asarray(run_group["masks_roi"][:], dtype=np.uint8)
        nonempty_rows = np.any(masks_array > 0, axis=(1, 2, 3))
    else:
        nonempty_rows = np.zeros((total_rois,), dtype=bool)
    run_group.attrs["summary_statistics"] = {
        "rows_total": int(total_rois),
        "rows_with_nonempty_masks": int(np.sum(nonempty_rows)),
        "rows_empty_masks": int(total_rois - np.sum(nonempty_rows)),
        "output_run": resolved_run_name,
        "crop_run": str(crop_run_name),
        "duration_seconds": float(duration),
        "created_at_utc": created_at,
        "masks_roi_materialized": bool(args.write_masks_roi),
    }

    for label in mask_labels:
        write_subject_mask_component_provenance(
            run_group,
            component_name=label,
            source_stage=output_parent,
            source_run=resolved_run_name,
            source_method=str(run_group.attrs["method"]),
            source_channels=[label],
            source_label_schema_id=label_schema_id,
            source_created_at_utc=created_at,
        )

    if timing_profiler.enabled:
        run_group.attrs["timing_profile"] = timing_profiler.summary(
            total_items=int(total_rois),
            wall_seconds=float(duration),
            notes=[
                "roi_read measures ROI slice fetch from the active crop image source.",
                "sync_before_* and sync_after_* measure explicit CUDA synchronize calls used to attribute queued GPU work deterministically.",
                "input_normalize runs after the device transfer so dtype conversion, scaling, and clipping can execute on GPU.",
                "h2d_copy and model_forward are measured separately for the U-Net loop.",
                "d2h_copy includes sigmoid + clamp + dtype conversion, on-device spatial metrics, probability transfer, and optional binary transfer when masks_roi is materialized.",
                "output_write_probs measures ordinary probability Zarr writes when indexed sharding is disabled; output_write_binary is present only when masks_roi is materialized.",
                "output_shard_buffer_submit includes batch submission and any wait for one of two shard buffers; output_shard_buffer_fill measures copies into channel-major host buffers.",
                "output_shard_write writes complete immutable probability storage shards from the background buffer while inference continues; output_shard_validate appears only when the writer itself performs the full decoded destination reread.",
                "receipt_bound_final_layout_unit_v1 defers that redundant writer reread only for a complete non-authoritative partition; mandatory final-layout packaging rereads the persisted probabilities, verifies their semantic receipt, and seals encoded objects before publication.",
                "metric_compute covers copying precomputed per-batch metrics into full-run metric arrays.",
                "semantic_receipt_hash hashes each immutable CPU output batch in row order after its output-worker write; with --async-output this overlaps the next GPU inference batch.",
                "output_queue_put and output_queue_drain appear when --async-output overlaps inference with background output writes.",
            ],
        )

    platform_info = env_info.get("platform", {})
    provenance_inputs = {
        "output_parent": output_parent,
        "source_crop_run": str(crop_run_name),
        **crop_snapshot_attrs,
        **crop_pixel_attrs,
        **work_package_attrs,
        "source_roi_read_mode": crop_source.roi_read_mode,
        "roi_cache_policy": crop_source.roi_cache_policy,
        "roi_cache_used": bool(crop_source.roi_cache_used),
        "roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
        "roi_cache_key": crop_source.roi_cache_key,
        "roi_cache_path": crop_source.roi_cache_path,
        "roi_cache_canonical_path": getattr(
            crop_source, "roi_cache_canonical_path", None
        ),
        "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
        "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
        "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
        "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
        "frame_source": crop_source.frame_source_kind,
        "source_video_path": crop_source.frame_source_path
        or crop_group.attrs.get("video_source_path"),
        "source_roi_pixels_sha256": observed_pixels_sha256,
        SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR: scientific_identity,
        SUBJECT_MASK_ATTEMPT_ATTR: attempt,
        SUBJECT_MASK_ATTEMPT_LINEAGE_EVIDENCE_ATTR: lineage_evidence,
    }
    provenance_inputs.update(shard_attrs)
    provenance_inputs.update(assignment_keypoint_attrs)
    if model_resolution_payload is not None:
        provenance_inputs["model_resolution"] = model_resolution_payload
    if crop_source.roi_cache_path is not None:
        provenance_inputs["roi_cache_path"] = crop_source.roi_cache_path
    if getattr(crop_source, "roi_cache_canonical_path", None) is not None:
        provenance_inputs["roi_cache_canonical_path"] = (
            crop_source.roi_cache_canonical_path
        )
    provenance = build_stage_provenance(
        stage="subject_masks",
        command=" ".join(sys.argv),
        created_at_utc=created_at,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        parameters={
            "batch_size": int(args.batch_size),
            "device": str(device),
            "output_parent": output_parent,
            "model_input_size": int(model_input_size),
            "model_input_transform": model_input_transform.to_attrs(),
            "label_schema_id": label_schema_id,
            "mask_labels": list(mask_labels),
            "mask_probs_chunk_rois": int(args.mask_probs_chunk_rois),
            "mask_probs_shard_rois": (
                int(args.mask_probs_shard_rois)
                if args.mask_probs_shard_rois is not None
                else None
            ),
            "mask_probs_storage_layout": run_group.attrs["mask_probs_storage_layout"],
            "mask_probs_storage_policy": run_group.attrs["mask_probs_storage_policy"],
            "mask_probs_default_shard_rois": int(DEFAULT_MASK_PROBS_SHARD_ROIS),
            "mask_probs_dtype": str(args.mask_probs_dtype),
            "mask_probs_destination_validation": str(
                args.mask_probs_destination_validation
            ),
            "write_masks_roi": bool(args.write_masks_roi),
            "async_output": bool(args.async_output),
            "output_queue_size": int(args.output_queue_size),
            "semantic_receipt_hash_execution": run_group.attrs[
                "semantic_receipt_hash_execution"
            ],
            "progress": bool(args.progress),
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_cache_manifest": (
                str(args.roi_cache_manifest) if args.roi_cache_manifest else None
            ),
            "roi_work_package_manifest": (
                str(args.roi_work_package_manifest)
                if args.roi_work_package_manifest
                else None
            ),
            "roi_work_package_role": work_package_attrs.get("roi_work_package_role"),
            "collection_partition_contract_digest": (
                work_package_attrs.get("collection_partition_contract", {}).get(
                    "payload_digest"
                )
                if isinstance(
                    work_package_attrs.get("collection_partition_contract"), Mapping
                )
                else None
            ),
            "source_roi_cache_alias_manifest": (
                str(args.source_roi_cache_alias_manifest)
                if args.source_roi_cache_alias_manifest
                else None
            ),
            "source_roi_cache_row_index_path": (
                str(args.source_roi_cache_row_index_path)
                if args.source_roi_cache_row_index_path
                else None
            ),
            "roi_live_acceleration": crop_source.roi_live_acceleration_requested,
            "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
            "mask_probs_shard_write": run_group.attrs.get("mask_probs_shard_write"),
            "mask_probs_postpack": run_group.attrs.get("mask_probs_postpack"),
            "scientific_identity_digest": scientific_identity["digest"],
            "attempt_payload_digest": attempt["payload_digest"],
        },
        inputs=provenance_inputs,
        artifacts={
            "checkpoint_path": str(checkpoint_path),
            "segmenter": "unet",
            "label_schema_id": label_schema_id,
            "masks_roi_materialized": bool(args.write_masks_roi),
            "mask_probs_storage_layout": run_group.attrs["mask_probs_storage_layout"],
            "mask_probs_storage_policy": run_group.attrs["mask_probs_storage_policy"],
            "mask_probs_shard_write": run_group.attrs.get("mask_probs_shard_write"),
            "mask_probs_postpack": run_group.attrs.get("mask_probs_postpack"),
            "model_resolution": model_resolution_payload,
        },
    )
    write_stage_provenance(run_group, provenance)
    run_provenance = append_input_artifacts(
        build_run_provenance_from_stage_record(provenance),
        [checkpoint_artifact],
    )
    run_path = f"{output_parent}/{resolved_run_name}"
    published_coordinate_surfaces = None
    if canonical_coordinate_context is not None:
        if boundary is None:
            raise RuntimeError(
                "Canonical subject-mask publication lacks its failure boundary."
            )
        run_group = boundary.require_owned_run()
        coordinate_checkpoint = capture_subject_mask_coordinate_publication_checkpoint(
            root,
            run_path,
            expected_publication_owner=boundary.owner_token,
        )
        if boundary is not None:
            boundary.bind_coordinate_checkpoint(coordinate_checkpoint)
        published_coordinate_surfaces = publish_subject_mask_coordinate_surfaces(
            root,
            run_path,
            expected_publication_owner=boundary.owner_token,
        )
        # Publication writes through fresh root-resolved handles.  Complete only
        # through another fresh handle so descriptors cannot be lost.
        run_group = boundary.require_owned_run()
    mark_run_complete(
        run_group,
        # Canonical activation owns selector publication after a fresh exact
        # coordinate reload.  Shards remain deliberately unselected.
        parent_group=None,
        run_name=resolved_run_name,
        run_provenance=run_provenance,
    )
    if shard_output:
        run_group.attrs["registry_status_deferred_reason"] = (
            "collection_shard_not_canonical_stage_output"
        )
    if canonical_coordinate_context is not None:
        assert boundary is not None
        assert published_coordinate_surfaces is not None
        run_group = boundary.require_owned_run()
        assert boundary.parent_selector_snapshot is not None
        assert boundary.owner_token is not None
        _activate_validated_subject_mask_coordinate_surfaces(
            root,
            root[output_parent],
            published_coordinate_surfaces,
            run_name=resolved_run_name,
            publication_owner_token=boundary.owner_token,
            selector_snapshot=boundary.parent_selector_snapshot,
        )
        run_group = boundary.require_owned_run()
    if boundary is not None:
        boundary.mark_finalized()

    if not args.defer_registry_status and not shard_output:
        try:
            emit_subject_mask_stage_completion(
                root,
                zarr_path,
                run_group=run_group,
                run_name=resolved_run_name,
                source=_SUBJECT_MASKS_STATUS_SOURCE,
                console=console,
                invalidate_on_ok=True,
            )
        except Exception as exc:  # telemetry is explicitly post-commit
            console.print(
                "[yellow]Warning:[/yellow] canonical subject-mask output is "
                f"committed, but registry/status telemetry failed: {exc}"
            )

    console.print(
        f"\n[green]✓[/green] U-Net subject masks written to "
        f"[cyan]{_subject_mask_stage_path(output_parent, resolved_run_name)}[/cyan] "
        f"({total_rois:,} ROIs processed in {duration:.1f}s)."
    )
    if timing_profiler.enabled:
        console.print("[bold]Timing Profile:[/bold]")
        for line in timing_profiler.render_lines(
            total_items=total_rois, wall_seconds=duration, limit=8
        ):
            console.print(f"[dim]{line}[/dim]")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
