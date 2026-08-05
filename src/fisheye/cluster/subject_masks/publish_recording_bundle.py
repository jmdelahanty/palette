"""Publish one proof-bound whole-recording subject-mask bundle.

The compute archive is treated as a selector-ineligible draft.  This command
turns its raw and refined producer receipts into production-streaming source
receipts, rematerializes both cores through the shared storage planner,
computes the independent quality run, and atomically imports the three members
as one inactive bundle.  When ``--cache-run`` is supplied, it also derives the
recording-level sampled-contour presentation cache and emits bundle v3.
Activation is a separate explicit operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.subject_mask_attempt import (
    validate_subject_mask_attempt,
    validate_subject_mask_scientific_identity,
)
from fisheye.shared.subject_mask_worker_receipt import (
    build_recording_subject_mask_source_receipt,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    activate_subject_mask_bundle,
    publish_subject_mask_bundle_candidate,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SubjectMaskCoreValidationMode,
    build_subject_mask_core_coordinate_dependencies,
    publish_selector_ineligible_subject_mask_core_snapshot,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_ATTRIBUTE,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.subject_mask_cache_publication import (
    DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES,
    publish_selector_ineligible_subject_mask_sampled_contours,
)
from fisheye.shared.zarr.subject_mask_quality_manifest import (
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_quality_publication import (
    publish_selector_ineligible_subject_mask_quality_snapshot,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    validate_subject_mask_source_run_manifest,
    validate_subject_mask_source_validation_receipt,
)
from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    derive_subject_mask_frame_row_offsets,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

_SCIENCE_ATTR = "subject_mask_scientific_identity"
_ATTEMPT_ATTR = "subject_mask_attempt"
_WORKER_RECEIPT_ATTR = "subject_mask_worker_semantic_receipt_binding"
_RECORDING_SOURCE_RECEIPT_ATTR = "subject_mask_recording_source_receipt_binding"


class _ConcatenatedRows:
    """Read-only first-axis concatenation used during bounded publication."""

    def __init__(self, arrays: Sequence[Any]) -> None:
        if not arrays:
            raise ValueError("Concatenated array requires at least one source.")
        trailing = tuple(int(value) for value in arrays[0].shape[1:])
        dtype = np.dtype(arrays[0].dtype)
        for value in arrays:
            if tuple(int(item) for item in value.shape[1:]) != trailing:
                raise ValueError("Concatenated array trailing shapes differ.")
            if np.dtype(value.dtype) != dtype:
                raise ValueError("Concatenated array dtypes differ.")
        self._arrays = tuple(arrays)
        self._offsets = np.cumsum(
            [0, *(int(value.shape[0]) for value in self._arrays)],
            dtype=np.int64,
        )
        self.shape = (int(self._offsets[-1]), *trailing)
        self.dtype = dtype
        self.ndim = len(self.shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        if key is Ellipsis:
            key = (slice(None),) * self.ndim
        elif not isinstance(key, tuple):
            key = (key,)
        key = (*key, *(slice(None),) * (self.ndim - len(key)))
        first, trailing = key[0], key[1:]
        if isinstance(first, (int, np.integer)):
            index = int(first)
            if index < 0:
                index += self.shape[0]
            if not 0 <= index < self.shape[0]:
                raise IndexError(index)
            source_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
            local = index - int(self._offsets[source_index])
            return np.asarray(self._arrays[source_index][(local, *trailing)])
        if isinstance(first, slice):
            start, stop, step = first.indices(self.shape[0])
            if step != 1:
                indices = np.arange(start, stop, step, dtype=np.int64)
                return self[(indices, *trailing)]
            pieces: list[np.ndarray] = []
            for source_index, source in enumerate(self._arrays):
                source_start = int(self._offsets[source_index])
                source_stop = int(self._offsets[source_index + 1])
                left = max(start, source_start)
                right = min(stop, source_stop)
                if left < right:
                    pieces.append(
                        np.asarray(
                            source[
                                (
                                    slice(left - source_start, right - source_start),
                                    *trailing,
                                )
                            ]
                        )
                    )
            if not pieces:
                return np.empty((0, *self.shape[1:]), dtype=self.dtype)[
                    (slice(None), *trailing)
                ]
            return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        indices = np.asarray(first, dtype=np.int64).reshape(-1)
        return np.stack([self[(int(index), *trailing)] for index in indices], axis=0)


def _paths(group: Any, names: Sequence[str]) -> dict[str, Any]:
    missing = [name for name in names if name not in group]
    if missing:
        raise ValueError(f"Group {group.path!r} lacks exact arrays: {missing!r}.")
    return {name: group[name] for name in names}


def _crop_arrays(crop: Any) -> dict[str, Any]:
    return _paths(
        crop,
        (
            "instance_key",
            "source_acquisition_frame_index",
            "frame_row_offsets",
            "source_crop_xywh",
        ),
    )


def _require_complete_order(run: Any, crop: Any, *, role: str) -> np.ndarray:
    rows = np.asarray(run["source_crop_row_ids"][...], dtype=np.int64)
    expected = np.arange(int(crop["instance_key"].shape[0]), dtype=np.int64)
    if not np.array_equal(rows, expected):
        raise ValueError(
            f"{role} draft must cover the complete crop rowset in canonical order."
        )
    return rows


def _raw_arrays(run: Any, crop: Any, *, n_frames: int) -> dict[str, Any]:
    _require_complete_order(run, crop, role="Raw subject-mask")
    arrays = _paths(
        run,
        (
            "source_crop_row_ids",
            "instance_key",
            "source_acquisition_frame_index",
            "mask_probs_roi",
            "available_channels",
            "metrics/prob_max",
            "metrics/mask_present",
            "metrics/area_px",
            "metrics/centroid_xy",
            "metrics/centroid_valid",
            "metrics/bbox_xyxy",
            "metrics/bbox_valid",
        ),
    )
    frames = np.asarray(arrays["source_acquisition_frame_index"][...], dtype=np.int64)
    arrays["frame_row_offsets"] = derive_subject_mask_frame_row_offsets(
        frames, n_frames=n_frames
    )
    arrays["source_crop_xywh"] = crop["source_crop_xywh"]
    return arrays


def _refined_arrays(run: Any, crop: Any, *, n_frames: int) -> dict[str, Any]:
    rows = _require_complete_order(run, crop, role="Refined subject-mask")
    arrays = _paths(
        run,
        (
            "source_crop_row_ids",
            "masks_roi",
            "available_channels",
            "metrics/mask_present",
            "metrics/area_px",
            "metrics/centroid_xy",
            "metrics/centroid_valid",
            "metrics/bbox_xyxy",
            "metrics/bbox_valid",
        ),
    )
    frames = np.asarray(crop["source_acquisition_frame_index"][rows], dtype=np.int64)
    arrays.update(
        {
            "instance_key": crop["instance_key"],
            "source_acquisition_frame_index": frames,
            "frame_row_offsets": derive_subject_mask_frame_row_offsets(
                frames, n_frames=n_frames
            ),
            "source_crop_xywh": crop["source_crop_xywh"],
        }
    )
    return arrays


def _strict_attrs(attrs: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        if name in attrs:
            value = attrs[name]
            json.dumps(value, allow_nan=False)
            result[name] = value
    return result


def _worker_evidence(
    archive: Path, run: Any, *, global_start_row: int = 0
) -> dict[str, Any]:
    science = run.attrs.get(_SCIENCE_ATTR)
    attempt = run.attrs.get(_ATTEMPT_ATTR)
    binding = run.attrs.get(_WORKER_RECEIPT_ATTR)
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ValueError(f"{run.path} is not a complete producer run.")
    if not isinstance(science, Mapping) or validate_subject_mask_scientific_identity(
        science
    ):
        raise ValueError(f"{run.path} lacks a valid scientific identity.")
    if not isinstance(attempt, Mapping) or validate_subject_mask_attempt(attempt):
        raise ValueError(f"{run.path} lacks a valid attempt identity.")
    if not isinstance(binding, Mapping):
        raise ValueError(f"{run.path} lacks a worker semantic receipt binding.")
    relative_path = str(binding.get("relative_path") or "").strip()
    if not relative_path:
        raise ValueError(f"{run.path} worker receipt is not durably persisted.")
    receipt_bytes = (archive / relative_path).read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != binding.get("document_sha256"):
        raise ValueError(f"{run.path} worker receipt document digest changed.")
    receipt = json.loads(receipt_bytes)
    if receipt.get("payload_digest") != binding.get("payload_digest"):
        raise ValueError(f"{run.path} worker receipt payload binding changed.")
    return {
        "global_start_row": int(global_start_row),
        "scientific_identity": dict(science),
        "attempt": dict(attempt),
        "receipt": receipt,
    }


def _raw_shard_collection(
    runs: Sequence[Any],
    crop: Any,
    *,
    n_frames: int,
    archive: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Expose ordered raw shards as one exact recording array inventory."""

    if not runs:
        raise ValueError("Raw shard collection cannot be empty.")
    ordered: list[tuple[int, Any, np.ndarray]] = []
    for run in runs:
        rows = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64)
        if rows.size == 0 or np.any(np.diff(rows) != 1):
            raise ValueError(f"{run.path} crop rows are not one contiguous interval.")
        ordered.append((int(rows[0]), run, rows))
    ordered.sort(key=lambda item: item[0])
    expected_rows = np.arange(int(crop["instance_key"].shape[0]), dtype=np.int64)
    observed_rows = np.concatenate([item[2] for item in ordered])
    if not np.array_equal(observed_rows, expected_rows):
        raise ValueError(
            "Raw shard union must cover every canonical crop row exactly once in order."
        )
    labels = tuple(str(value) for value in ordered[0][1].attrs["mask_labels"])
    threshold = float(ordered[0][1].attrs.get("mask_probability_threshold", 0.5))
    available = np.asarray(ordered[0][1]["available_channels"][:], dtype=bool)
    for _start, run, _rows in ordered[1:]:
        if tuple(str(value) for value in run.attrs["mask_labels"]) != labels:
            raise ValueError("Raw shard component registries differ.")
        if float(run.attrs.get("mask_probability_threshold", 0.5)) != threshold:
            raise ValueError("Raw shard probability thresholds differ.")
        if not np.array_equal(
            np.asarray(run["available_channels"][:], dtype=bool), available
        ):
            raise ValueError("Raw shard available-channel declarations differ.")
    row_paths = (
        "mask_probs_roi",
        "metrics/prob_max",
        "metrics/mask_present",
        "metrics/area_px",
        "metrics/centroid_xy",
        "metrics/centroid_valid",
        "metrics/bbox_xyxy",
        "metrics/bbox_valid",
    )
    arrays: dict[str, Any] = {
        path: _ConcatenatedRows([item[1][path] for item in ordered])
        for path in row_paths
    }
    arrays.update(
        {
            "source_crop_row_ids": expected_rows,
            "instance_key": crop["instance_key"],
            "source_acquisition_frame_index": crop["source_acquisition_frame_index"],
            "frame_row_offsets": derive_subject_mask_frame_row_offsets(
                np.asarray(crop["source_acquisition_frame_index"][:], dtype=np.int64),
                n_frames=n_frames,
            ),
            "source_crop_xywh": crop["source_crop_xywh"],
            "available_channels": available,
        }
    )
    workers = [
        _worker_evidence(archive, run, global_start_row=start)
        for start, run, _rows in ordered
    ]
    return arrays, workers


def _refined_shard_collection(
    runs: Sequence[Any],
    crop: Any,
    *,
    n_frames: int,
    archive: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Expose ordered refined clip outputs as one exact recording inventory."""

    if not runs:
        raise ValueError("Refined shard collection cannot be empty.")
    ordered: list[tuple[int, Any, np.ndarray]] = []
    for run in runs:
        rows = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64)
        if rows.size == 0 or np.any(np.diff(rows) != 1):
            raise ValueError(f"{run.path} crop rows are not one contiguous interval.")
        ordered.append((int(rows[0]), run, rows))
    ordered.sort(key=lambda item: item[0])
    expected_rows = np.arange(int(crop["instance_key"].shape[0]), dtype=np.int64)
    observed_rows = np.concatenate([item[2] for item in ordered])
    if not np.array_equal(observed_rows, expected_rows):
        raise ValueError(
            "Refined shard union must cover every canonical crop row exactly once "
            "in order."
        )
    labels = tuple(str(value) for value in ordered[0][1].attrs["mask_labels"])
    available = np.asarray(ordered[0][1]["available_channels"][:], dtype=bool)
    for _start, run, _rows in ordered[1:]:
        if tuple(str(value) for value in run.attrs["mask_labels"]) != labels:
            raise ValueError("Refined shard component registries differ.")
        if not np.array_equal(
            np.asarray(run["available_channels"][:], dtype=bool), available
        ):
            raise ValueError("Refined shard available-channel declarations differ.")
    row_paths = (
        "masks_roi",
        "metrics/mask_present",
        "metrics/area_px",
        "metrics/centroid_xy",
        "metrics/centroid_valid",
        "metrics/bbox_xyxy",
        "metrics/bbox_valid",
    )
    arrays: dict[str, Any] = {
        path: _ConcatenatedRows([item[1][path] for item in ordered])
        for path in row_paths
    }
    frames = np.asarray(crop["source_acquisition_frame_index"][:], dtype=np.int64)
    arrays.update(
        {
            "source_crop_row_ids": expected_rows,
            "instance_key": crop["instance_key"],
            "source_acquisition_frame_index": frames,
            "frame_row_offsets": derive_subject_mask_frame_row_offsets(
                frames,
                n_frames=n_frames,
            ),
            "source_crop_xywh": crop["source_crop_xywh"],
            "available_channels": available,
        }
    )
    workers = [
        _worker_evidence(archive, run, global_start_row=start)
        for start, run, _rows in ordered
    ]
    return arrays, workers


def _consistent_refined_source_attrs(
    runs: Sequence[Any],
    *,
    labels: Sequence[str],
) -> dict[str, Any]:
    names = (
        "assignment_keypoint_group",
        "assignment_keypoints_run",
        "assignment_keypoint_run",
    )
    result: dict[str, Any] = {"mask_labels": [str(value) for value in labels]}
    for name in names:
        values = [run.attrs.get(name) for run in runs]
        encoded = [
            json.dumps(value, sort_keys=True, allow_nan=False) for value in values
        ]
        if len(set(encoded)) != 1:
            raise ValueError(f"Refined shard {name} declarations differ.")
        if values[0] is not None:
            result[name] = values[0]
    return result


def _prebuilt_source_documents(
    archive: Path,
    run: Any,
    *,
    kind: str,
    source_run_path: str,
    schema: Any,
    arrays: Mapping[str, Any],
    dimensions: SubjectMaskDimensions,
    components: SubjectMaskComponentRegistry,
    threshold: float | None,
) -> tuple[dict[str, object], dict[str, object]] | None:
    binding = run.attrs.get(_RECORDING_SOURCE_RECEIPT_ATTR)
    if binding is None:
        return None
    if not isinstance(binding, Mapping) or set(binding) != {
        "schema_id",
        "schema_version",
        "payload_digest",
        "relative_path",
        "document_sha256",
        "storage",
    }:
        raise ValueError(f"{run.path} recording source receipt binding is invalid.")
    relative = str(binding.get("relative_path") or "")
    if (
        binding.get("storage") != "strict_json_sidecar_v1"
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError(f"{run.path} recording source receipt path is unsafe.")
    receipt_bytes = (archive / relative).read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != binding.get("document_sha256"):
        raise ValueError(f"{run.path} recording source receipt document changed.")
    receipt = json.loads(receipt_bytes)
    if receipt.get("payload_digest") != binding.get("payload_digest"):
        raise ValueError(f"{run.path} recording source receipt payload changed.")
    manifest = run.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{run.path} recording source manifest is absent.")
    validated_manifest = validate_subject_mask_source_run_manifest(manifest)
    validated_receipt = validate_subject_mask_source_validation_receipt(
        receipt,
        kind=kind,
        source_run_path=source_run_path,
        source_manifest=validated_manifest,
        schema=schema,
        arrays=arrays,
        dimensions=dimensions,
        components=components,
        threshold=threshold,
    )
    return validated_manifest, validated_receipt


def publish_recording_subject_mask_bundle(
    *,
    analysis_zarr: Path,
    draft_zarr: Path,
    crop_run: str,
    raw_draft_parent: str = "subject_mask_runs",
    raw_draft_run: str,
    raw_draft_runs: Sequence[str] | None = None,
    refined_draft_run: str,
    raw_run: str,
    refined_run: str,
    quality_run: str,
    bundle_id: str,
    local_output_root: Path,
    quality_scratch_root: Path,
    cache_run: str | None = None,
    activate: bool = False,
    refined_draft_runs: Sequence[str] | None = None,
    cache_source_compute_block_bytes: int = DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES,
    cache_compute_workers: int = 1,
    coordinate_contract_policy: str = "require_crop_v2",
) -> dict[str, object]:
    target = analysis_zarr.expanduser().resolve()
    draft_path = draft_zarr.expanduser().resolve()
    output = local_output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    draft = open_zarr_root(draft_path, mode="r")
    crop = draft[f"crop_runs/{crop_run}"]
    policy = str(coordinate_contract_policy).strip()
    if policy not in {"require_crop_v2", "legacy_allow_missing"}:
        raise ValueError("Unsupported coordinate_contract_policy.")
    crop_manifest_value = crop.attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE)
    crop_manifest: dict[str, Any] | None = None
    if isinstance(crop_manifest_value, Mapping):
        crop_errors = validate_crop_run_manifest(crop_manifest_value)
        if crop_errors:
            raise ValueError("Invalid crop run manifest: " + "; ".join(crop_errors))
        if (
            crop_manifest_value.get("schema_version")
            != CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("Recording subject-mask publication requires crop v2.")
        crop_manifest = dict(crop_manifest_value)
    elif policy == "require_crop_v2":
        raise ValueError(
            "Recording subject-mask publication requires a persisted crop-v2 "
            "coordinate manifest."
        )
    if raw_draft_parent not in {"subject_mask_runs", "subject_mask_shard_runs"}:
        raise ValueError("raw_draft_parent must name a canonical or shard raw parent.")
    raw_names = tuple(
        dict.fromkeys(
            str(value)
            for value in (raw_draft_runs or (raw_draft_run,))
            if str(value).strip()
        )
    )
    if not raw_names:
        raise ValueError("At least one raw draft run is required.")
    raw_drafts = [draft[f"{raw_draft_parent}/{name}"] for name in raw_names]
    raw_draft = raw_drafts[0]
    refined_names = tuple(
        dict.fromkeys(
            str(value)
            for value in (refined_draft_runs or (refined_draft_run,))
            if str(value).strip()
        )
    )
    if not refined_names:
        raise ValueError("At least one refined draft run is required.")
    refined_drafts = [
        draft[f"refined_subject_masks_runs/{name}"] for name in refined_names
    ]
    refined_draft = refined_drafts[0]
    n_frames = int(crop["frame_row_offsets"].shape[0]) - 1
    raw_labels = tuple(str(value) for value in raw_draft.attrs["mask_labels"])
    refined_labels = tuple(str(value) for value in refined_draft.attrs["mask_labels"])
    raw_components = SubjectMaskComponentRegistry(raw_labels)
    refined_components = SubjectMaskComponentRegistry(refined_labels)
    if len(raw_drafts) == 1:
        raw_arrays = _raw_arrays(raw_draft, crop, n_frames=n_frames)
        raw_workers = [_worker_evidence(draft_path, raw_draft)]
        raw_source_path = f"{raw_draft_parent}/{raw_names[0]}"
    else:
        if raw_draft_parent != "subject_mask_shard_runs":
            raise ValueError("Multiple raw drafts must be immutable shard runs.")
        raw_arrays, raw_workers = _raw_shard_collection(
            raw_drafts,
            crop,
            n_frames=n_frames,
            archive=draft_path,
        )
        raw_source_path = f"subject_mask_shard_collections/{raw_run}"
    if len(refined_drafts) == 1:
        refined_arrays = _refined_arrays(refined_draft, crop, n_frames=n_frames)
        refined_workers: list[dict[str, Any]] | None = None
        refined_source_path = f"refined_subject_masks_runs/{refined_names[0]}"
        refined_source_attrs = _strict_attrs(
            refined_draft.attrs,
            (
                "mask_labels",
                "assignment_keypoint_group",
                "assignment_keypoints_run",
                "assignment_keypoint_run",
                _SCIENCE_ATTR,
                _ATTEMPT_ATTR,
                _WORKER_RECEIPT_ATTR,
            ),
        )
    else:
        refined_arrays, refined_workers = _refined_shard_collection(
            refined_drafts,
            crop,
            n_frames=n_frames,
            archive=draft_path,
        )
        refined_source_path = f"refined_subject_mask_shard_collections/{refined_run}"
        refined_source_attrs = _consistent_refined_source_attrs(
            refined_drafts,
            labels=refined_labels,
        )
    raw_dimensions = SubjectMaskDimensions(
        n_frames=n_frames,
        n_rois=int(raw_arrays["mask_probs_roi"].shape[0]),
        n_channels=int(raw_arrays["mask_probs_roi"].shape[1]),
        roi_height=int(raw_arrays["mask_probs_roi"].shape[2]),
        roi_width=int(raw_arrays["mask_probs_roi"].shape[3]),
    )
    refined_dimensions = SubjectMaskDimensions(
        n_frames=n_frames,
        n_rois=int(refined_arrays["masks_roi"].shape[0]),
        n_channels=int(refined_arrays["masks_roi"].shape[1]),
        roi_height=int(refined_arrays["masks_roi"].shape[2]),
        roi_width=int(refined_arrays["masks_roi"].shape[3]),
    )
    for name in ("n_frames", "n_rois", "roi_height", "roi_width"):
        if getattr(raw_dimensions, name) != getattr(refined_dimensions, name):
            raise ValueError(
                "Raw and refined subject-mask row/pixel domains differ for "
                f"{name!r}."
            )
    threshold = float(raw_draft.attrs.get("mask_probability_threshold", 0.5))
    raw_documents = _prebuilt_source_documents(
        draft_path,
        raw_draft,
        kind="raw_probability_uint8",
        source_run_path=raw_source_path,
        schema=RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
        arrays=raw_arrays,
        dimensions=raw_dimensions,
        components=raw_components,
        threshold=threshold,
    )
    raw_source_manifest, raw_receipt = raw_documents or (
        build_recording_subject_mask_source_receipt(
            kind="raw_probability_uint8",
            stage_kind="raw_subject_mask",
            source_run_path=raw_source_path,
            schema=RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
            arrays=raw_arrays,
            dimensions=raw_dimensions,
            components=raw_components,
            threshold=threshold,
            workers=raw_workers,
        )
    )
    raw_coordinate_dependencies = (
        build_subject_mask_core_coordinate_dependencies(
            kind="raw_probability_uint8",
            crop_run_path=f"crop_runs/{crop_run}",
            crop_manifest=crop_manifest,
            source_crop_arrays=_crop_arrays(crop),
            source_run_path=raw_source_path,
            source_validation_receipt=raw_receipt,
            n_rois=raw_dimensions.n_rois,
        )
        if crop_manifest is not None
        else None
    )
    raw_store = output / "raw.zarr"
    raw_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        raw_arrays,
        source_crop_arrays=_crop_arrays(crop),
        source_manifest=raw_source_manifest,
        n_frames=n_frames,
        components=raw_components,
        destination=raw_store,
        run_id=raw_run,
        kind="raw_probability_uint8",
        source_run_path=raw_source_path,
        source_attributes=(
            _strict_attrs(
                raw_draft.attrs,
                (
                    "mask_labels",
                    _SCIENCE_ATTR,
                    _ATTEMPT_ATTR,
                    _WORKER_RECEIPT_ATTR,
                ),
            )
            if len(raw_drafts) == 1
            else {"mask_labels": list(raw_labels)}
        ),
        threshold=threshold,
        validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        source_validation_receipt=raw_receipt,
        coordinate_dependencies=raw_coordinate_dependencies,
        created_by="publish_recording_subject_mask_bundle",
    )
    refined_documents = (
        _prebuilt_source_documents(
            draft_path,
            refined_draft,
            kind="refined_dense_core",
            source_run_path=refined_source_path,
            schema=REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
            arrays=refined_arrays,
            dimensions=refined_dimensions,
            components=refined_components,
            threshold=None,
        )
        if len(refined_drafts) == 1
        else None
    )
    refined_source_manifest, refined_receipt = refined_documents or (
        build_recording_subject_mask_source_receipt(
            kind="refined_dense_core",
            stage_kind="refined_subject_mask",
            source_run_path=refined_source_path,
            schema=REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
            arrays=refined_arrays,
            dimensions=refined_dimensions,
            components=refined_components,
            threshold=None,
            workers=(
                refined_workers
                if refined_workers is not None
                else [_worker_evidence(draft_path, refined_draft)]
            ),
            assembly_context={
                "source_raw_publication_manifest_digest": (
                    raw_publication.manifest["payload_digest"]
                )
            },
        )
    )
    refined_coordinate_dependencies = (
        build_subject_mask_core_coordinate_dependencies(
            kind="refined_dense_core",
            crop_run_path=f"crop_runs/{crop_run}",
            crop_manifest=crop_manifest,
            source_crop_arrays=_crop_arrays(crop),
            source_run_path=refined_source_path,
            source_validation_receipt=refined_receipt,
            n_rois=refined_dimensions.n_rois,
            raw_core_manifest=raw_publication.manifest,
        )
        if crop_manifest is not None
        else None
    )
    refined_store = output / "refined.zarr"
    refined_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        refined_arrays,
        source_crop_arrays=_crop_arrays(crop),
        source_manifest=refined_source_manifest,
        n_frames=n_frames,
        components=refined_components,
        destination=refined_store,
        run_id=refined_run,
        kind="refined_dense_core",
        source_run_path=refined_source_path,
        source_attributes=refined_source_attrs,
        validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        source_validation_receipt=refined_receipt,
        coordinate_dependencies=refined_coordinate_dependencies,
        created_by="publish_recording_subject_mask_bundle",
    )
    cache_publication = None
    cache_store = output / "cache.zarr"
    if cache_run is not None:
        cache_publication = publish_selector_ineligible_subject_mask_sampled_contours(
            refined_snapshot_root=refined_store,
            refined_run_id=refined_run,
            destination=cache_store,
            cache_run_id=cache_run,
            source_compute_block_bytes=cache_source_compute_block_bytes,
            compute_workers=cache_compute_workers,
            created_by="publish_recording_subject_mask_bundle",
        )
    refined_root = open_zarr_root(refined_store, mode="r")
    refined_published_run = refined_root[f"refined_subject_masks_runs/{refined_run}"]
    refined_docs = refined_publication.manifest["payload"]["logical_content"][
        "document"
    ]["arrays"]
    quality_source = SubjectMaskQualitySourceReference(
        run_name=refined_run,
        manifest_digest=canonical_json_sha256(refined_publication.manifest),
        dense_array_values_sha256=str(refined_docs["masks_roi"]["sha256"]),
        component_registry_digest=canonical_json_sha256(
            refined_components.as_manifest()
        ),
        source_array_values_sha256={
            path: str(refined_docs[path]["sha256"])
            for path in (
                "masks_roi",
                "instance_key",
                "source_crop_row_ids",
                "source_acquisition_frame_index",
                "frame_row_offsets",
                "available_channels",
            )
        },
    )
    quality_store = output / "quality.zarr"
    quality_publication = publish_selector_ineligible_subject_mask_quality_snapshot(
        {
            path: refined_published_run[path]
            for path in (
                "masks_roi",
                "instance_key",
                "source_crop_row_ids",
                "source_acquisition_frame_index",
                "frame_row_offsets",
                "available_channels",
            )
        },
        n_frames=n_frames,
        components=refined_components,
        source=quality_source,
        source_manifest=refined_publication.manifest,
        destination=quality_store,
        run_id=quality_run,
        shadow_root=output,
        scratch_root=quality_scratch_root,
        created_by="publish_recording_subject_mask_bundle",
    )
    recording_identity = str(
        open_zarr_root(target, mode="r").attrs.get("recording_id") or ""
    )
    bundle = publish_subject_mask_bundle_candidate(
        analysis_zarr=target,
        recording_identity=recording_identity,
        raw_snapshot_root=raw_store,
        raw_run_id=raw_run,
        refined_snapshot_root=refined_store,
        refined_run_id=refined_run,
        quality_snapshot_root=quality_store,
        quality_run_id=quality_run,
        bundle_id=bundle_id,
        cache_snapshot_root=(cache_store if cache_publication is not None else None),
        cache_run_id=(
            cache_publication.run_id if cache_publication is not None else None
        ),
    )
    authority = (
        activate_subject_mask_bundle(analysis_zarr=target, bundle_id=bundle_id)
        if activate
        else None
    )
    return {
        "status": "complete",
        "recording_identity": recording_identity,
        "n_frames": n_frames,
        "n_rois": raw_dimensions.n_rois,
        "raw_manifest_digest": raw_publication.manifest["payload_digest"],
        "refined_manifest_digest": refined_publication.manifest["payload_digest"],
        "quality_manifest_digest": quality_publication.manifest["payload_digest"],
        "cache_manifest_digest": (
            cache_publication.manifest["payload_digest"]
            if cache_publication is not None
            else None
        ),
        "bundle": bundle,
        "authority": authority,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--draft-zarr", required=True, type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument(
        "--raw-draft-parent",
        choices=("subject_mask_runs", "subject_mask_shard_runs"),
        default="subject_mask_runs",
    )
    parser.add_argument("--raw-draft-run", action="append", required=True)
    parser.add_argument("--refined-draft-run", action="append", required=True)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument(
        "--cache-run",
        help=(
            "Optional immutable sampled-contour cache run; supplying it emits "
            "the four-member bundle-v3 candidate."
        ),
    )
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--local-output-root", required=True, type=Path)
    parser.add_argument("--quality-scratch-root", required=True, type=Path)
    parser.add_argument(
        "--cache-source-compute-block-bytes",
        type=int,
        default=DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES,
    )
    parser.add_argument(
        "--cache-compute-workers",
        type=int,
        default=1,
        help=(
            "Parallel dense-row contour extraction workers. Zarr publication "
            "remains single-owner (default: 1)."
        ),
    )
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = publish_recording_subject_mask_bundle(
        analysis_zarr=args.analysis_zarr,
        draft_zarr=args.draft_zarr,
        crop_run=args.crop_run,
        raw_draft_parent=args.raw_draft_parent,
        raw_draft_run=args.raw_draft_run[0],
        raw_draft_runs=args.raw_draft_run,
        refined_draft_run=args.refined_draft_run[0],
        refined_draft_runs=args.refined_draft_run,
        raw_run=args.raw_run,
        refined_run=args.refined_run,
        quality_run=args.quality_run,
        cache_run=args.cache_run,
        bundle_id=args.bundle_id,
        local_output_root=args.local_output_root,
        quality_scratch_root=args.quality_scratch_root,
        cache_source_compute_block_bytes=args.cache_source_compute_block_bytes,
        cache_compute_workers=args.cache_compute_workers,
        activate=bool(args.activate),
    )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "publish_recording_subject_mask_bundle"]
