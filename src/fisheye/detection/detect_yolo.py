#!/usr/bin/env python3
"""
Direct video inference with YOLO - no import required.

The preferred CLI entry point is :mod:`fisheye.inference.predict_detections`,
which wraps this module and keeps inference scripts in one namespace. This
module still provides the core implementation and legacy CLI.
"""

import copy
import hashlib
import os
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Mapping, Sequence, Callable

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")

# Try to import decord for faster video decoding (GPU or CPU) before other FFmpeg users.
try:
    import decord  # type: ignore
    from decord import VideoReader, cpu, gpu  # type: ignore
    _DECORD_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    decord = None  # type: ignore
    VideoReader = None  # type: ignore
    cpu = None  # type: ignore
    gpu = None  # type: ignore
    _DECORD_IMPORT_ERROR = exc

import sys
import time
import yaml
import zarr
import numpy as np
import torch
import cv2
from datetime import datetime, timezone
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.markup import escape
from ultralytics import YOLO

from fisheye.shared.pynvvc_luma_rgb import BACKEND_PYNVVC_LUMA_RGB
from fisheye.shared.pynvvc_luma_rgb import BACKEND_PYNVVC_NV12_RGB
from fisheye.shared.pynvvc_luma_rgb import PYNVVC_BACKENDS
from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader
from fisheye.shared.pynvvc_luma_rgb import preprocess_luma_rgb
from fisheye.shared.pynvvc_luma_rgb import preprocess_nv12_rgb
from fisheye.shared.instance_keys import (
    instance_key_attrs,
    mint_detection_instance_keys,
)
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.observation_coordinate_publication import (
    BoundDetectionFrameEvidence,
    DETECTION_ACQUISITION_MAPPING_ATTR,
    DETECTION_ACQUISITION_MAPPING_SCHEMA_ID,
    DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION,
    DETECTION_INSTANCE_KEY_DERIVATION_ATTR,
    DETECTION_BACKEND_RESULT_PROJECTION_ATTR,
    DETECTION_OBSERVATION_CARDINALITY_ATTR,
    EMPTY_OBSERVATION_DECLARATION_ATTR,
    EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID as _EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID,
    EMPTY_OBSERVATION_DECLARATION_SCHEMA_VERSION as _EMPTY_OBSERVATION_DECLARATION_SCHEMA_VERSION,
    OBSERVATION_ROW_COUNT_ATTR,
    SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
    build_bound_detection_frame_evidence,
    capture_observation_coordinate_publication_checkpoint,
    derive_detection_source_camera_geometry,
    _load_persisted_detection_observation_geometry,
    publish_detection_observation_geometry,
    publish_detection_backend_result_projection,
    publish_detection_instance_key_derivation,
    publish_detection_observation_cardinality,
    publish_empty_detection_observation_declaration,
    restore_observation_coordinate_publication_checkpoint,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.directed_transform_chain import (
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import stamp_directed_transform_v2
from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    normalized_to_pixel_matrix,
    stamp_normalized_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.source_video_metadata import resolve_source_video
from fisheye.shared.transform_authority import (
    stamp_normalized_to_pixel_transform_authority,
)
from fisheye.shared.immutable_yolo_storage import validate_immutable_yolo_storage
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.artifact_fingerprint import fingerprint_artifact
from fisheye.shared.run_provenance import (
    CLI_RUN_PROVENANCE_ATTR,
    RUN_PROVENANCE_ATTR,
    append_input_artifacts,
    build_run_provenance,
)
from fisheye.shared.zarr.chunk_profiles import (
    create_geometry_preload_array,
    geometry_preload_chunks_for_data,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETED_AT_ATTR,
    RUN_LATEST_COMPLETE_ATTR,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from fisheye.shared.import_video_metadata import (
    probe_ffprobe_video_metadata,
    write_video_metadata,
)
from fisheye.shared.system_metadata import get_environment_info, get_git_info


DECODE_BACKEND_AUTO = "auto"
DECODE_BACKEND_DECORD_GPU = "decord_gpu"
DECODE_BACKEND_DECORD_CPU = "decord_cpu"
DECODE_BACKEND_OPENCV = "opencv"
DECODE_BACKEND_CHOICES = (
    DECODE_BACKEND_AUTO,
    *PYNVVC_BACKENDS,
    DECODE_BACKEND_DECORD_GPU,
    DECODE_BACKEND_DECORD_CPU,
    DECODE_BACKEND_OPENCV,
)
PYNVVC_SURFACE_MATERIALIZATION = "stream_event_owned_batch_v2"
DETECT_SHARD_WRITE_SCHEMA = "palette.detect_materialized_shards.v1"
DEFAULT_DETECT_ROW_SHARD_ROWS = 131_072
DEFAULT_DETECT_FRAME_SHARD_ROWS = 131_072
DETECTION_RUN_FAMILY = "detect_runs"
DETECTION_ARTIFACT_RUN_FAMILY = "detection_artifact_runs"
DETECTION_COORDINATE_CONTRACT_MODES = ("canonical", "artifact_unbound")
UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT = (
    "unbound_detection_artifact_v1"
)
DETECTION_ARTIFACT_ROW_ID_ARRAY = "artifact_row_id"
DETECTION_ARTIFACT_ROW_ID_CONTRACT = "palette.detection_artifact_row_id.v1"
DETECTION_ARTIFACT_STORAGE_ATTR = "detection_artifact_storage_validation"
DETECTION_ARTIFACT_STORAGE_SCHEMA = "palette.detection_artifact_storage.v1"
EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID = (
    _EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID
)
EMPTY_OBSERVATION_DECLARATION_SCHEMA_VERSION = (
    _EMPTY_OBSERVATION_DECLARATION_SCHEMA_VERSION
)
_RUN_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "authoritative_run_provenance",
)

_CANONICAL_DETECTION_PUBLICATION_ATTRS = frozenset(
    {
        "coordinate_contract",
        "coordinate_descriptor",
        "coordinate_descriptor_sha256",
        "row_identity_contract",
        "row_identity_contract_sha256",
        "row_identity_key",
        "row_identity_key_sha256",
        "row_identity_contract_ref",
        "source_row_temporal_authority",
        "source_row_temporal_authority_sha256",
        DETECTION_ACQUISITION_MAPPING_ATTR,
        f"{DETECTION_ACQUISITION_MAPPING_ATTR}_sha256",
        DETECTION_BACKEND_RESULT_PROJECTION_ATTR,
        f"{DETECTION_BACKEND_RESULT_PROJECTION_ATTR}_sha256",
        DETECTION_INSTANCE_KEY_DERIVATION_ATTR,
        f"{DETECTION_INSTANCE_KEY_DERIVATION_ATTR}_sha256",
        DETECTION_OBSERVATION_CARDINALITY_ATTR,
        f"{DETECTION_OBSERVATION_CARDINALITY_ATTR}_sha256",
        "detection_bbox_projection",
        "detection_bbox_projection_sha256",
        "bbox_center_derivation",
        "bbox_center_derivation_sha256",
        "pixel_frame_authority",
        "pixel_frame_authority_sha256",
        "transform_authority",
        "transform_authority_sha256",
        "directed_transform_v2",
        "directed_transform_v2_sha256",
        EMPTY_OBSERVATION_DECLARATION_ATTR,
        f"{EMPTY_OBSERVATION_DECLARATION_ATTR}_sha256",
        OBSERVATION_ROW_COUNT_ATTR,
    }
)


class _DetectionPublicationAttempt:
    """Own one detect writer's resources and atomic publication rollback state."""

    def __init__(self) -> None:
        self.root: Any | None = None
        self.parent_group: Any | None = None
        self.run_group: Any | None = None
        self.run_name: str | None = None
        self.selector_snapshot: Mapping[str, tuple[bool, Any]] | None = None
        self.coordinate_checkpoints: list[Any] = []
        self.cap: Any | None = None
        self.pynvvc_reader: Any | None = None
        self.decord_reader: Any | None = None
        self.named_child_owned = False
        self.resources_closed = False

    def prepare(
        self,
        *,
        root: Any,
        parent_group: Any,
        run_name: str,
        selector_snapshot: Mapping[str, tuple[bool, Any]] | None,
    ) -> None:
        self.root = root
        self.parent_group = parent_group
        self.run_name = str(run_name)
        self.selector_snapshot = selector_snapshot

    def bind_run(self, run_group: Any) -> None:
        self.run_group = run_group

    def claim_named_child(self) -> None:
        """Claim the preflight-absent child before its create call begins."""

        if self.parent_group is None or self.run_name is None:
            raise RuntimeError("Detection publication attempt is not prepared.")
        if self.run_name in self.parent_group:
            raise ValueError(
                f"Detection run {self.run_name!r} appeared before creation."
            )
        self.named_child_owned = True

    def track_cap(self, cap: Any) -> None:
        self.cap = cap

    def track_pynvvc_reader(self, reader: Any) -> None:
        self.pynvvc_reader = reader

    def track_decord_reader(self, reader: Any) -> None:
        self.decord_reader = reader

    def close_video_resources(self) -> list[str]:
        if self.resources_closed:
            return []
        failures: list[str] = []
        reader = self.pynvvc_reader
        self.pynvvc_reader = None
        if reader is not None:
            try:
                reader.close()
            except BaseException as exc:  # pragma: no cover - hostile external resource
                failures.append(f"PyNvVideoCodec reader: {exc}")
        cap = self.cap
        self.cap = None
        if cap is not None:
            try:
                cap.release()
            except BaseException as exc:  # pragma: no cover - hostile external resource
                failures.append(f"OpenCV capture: {exc}")
        reader = self.decord_reader
        self.decord_reader = None
        if reader is not None:
            try:
                _close_partially_initialized_decord_reader(reader)
            except BaseException as exc:  # pragma: no cover - backend dependent
                failures.append(f"Decord reader: {exc}")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException as exc:  # pragma: no cover - backend dependent
            failures.append(f"CUDA cache cleanup: {exc}")
        self.resources_closed = True
        return failures

    def rollback(self, cause: BaseException, *, cleanup_failures: Sequence[str]) -> None:
        if self.parent_group is None or self.run_name is None:
            if cleanup_failures:
                raise RuntimeError(
                    "Detection failed before publication and video cleanup was incomplete: "
                    f"{list(cleanup_failures)!r}."
                ) from cause
            return
        failures = list(cleanup_failures)
        run_group = self.run_group
        if run_group is None and self.named_child_owned:
            try:
                if self.run_name in self.parent_group:
                    run_group = self.parent_group[self.run_name]
                    self.run_group = run_group
            except BaseException as exc:  # pragma: no cover - hostile persistent mapping
                failures.append(f"reopen partially created run: {exc}")
        if run_group is None:
            if self.selector_snapshot is not None:
                try:
                    _restore_parent_selectors(
                        self.parent_group,
                        self.selector_snapshot,
                    )
                except BaseException as exc:  # pragma: no cover - hostile persistent mapping
                    failures.append(f"selectors: {exc}")
            if failures:
                raise RuntimeError(
                    "Detection publication failed before its named child could be "
                    f"recovered and rollback was incomplete: {failures!r}."
                ) from cause
            return
        _rollback_failed_detection_run(
            run_group=run_group,
            parent_group=self.parent_group,
            run_name=self.run_name,
            selector_snapshot=self.selector_snapshot,
            coordinate_checkpoints=self.coordinate_checkpoints,
            cause=cause,
            error=f"detection publication attempt failed: {cause}",
            cleanup_failures=failures,
            delete_run_when_safe=True,
        )


_CURRENT_DETECTION_PUBLICATION_ATTEMPT: ContextVar[
    _DetectionPublicationAttempt | None
] = ContextVar("palette_detection_publication_attempt", default=None)


def _active_detection_publication_attempt() -> _DetectionPublicationAttempt:
    attempt = _CURRENT_DETECTION_PUBLICATION_ATTEMPT.get()
    if attempt is None:  # pragma: no cover - detects accidental undecorated use
        raise RuntimeError("Detection publication attempt context is not active.")
    return attempt


def _guard_detection_publication_attempt(func: Callable[..., str]) -> Callable[..., str]:
    """Wrap the complete producer in one resource and publication transaction."""

    @wraps(func)
    def guarded(*args: Any, **kwargs: Any) -> str:
        attempt = _DetectionPublicationAttempt()
        token = _CURRENT_DETECTION_PUBLICATION_ATTEMPT.set(attempt)
        failure: BaseException | None = None
        try:
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                failure = exc
                raise
            finally:
                cleanup_failures = attempt.close_video_resources()
                if failure is not None:
                    attempt.rollback(
                        failure,
                        cleanup_failures=cleanup_failures,
                    )
                elif cleanup_failures:
                    cleanup_error = RuntimeError(
                        "Detection completed but video cleanup was incomplete: "
                        f"{cleanup_failures!r}."
                    )
                    attempt.rollback(
                        cleanup_error,
                        cleanup_failures=cleanup_failures,
                    )
                    raise cleanup_error
            return result
        finally:
            _CURRENT_DETECTION_PUBLICATION_ATTEMPT.reset(token)

    return guarded


def _require_group_path(root: Any, path: str) -> Any:
    current = root
    for name in path.strip("/").split("/"):
        current = current.require_group(name)
    return current


def _snapshot_parent_selectors(parent_group: Any) -> dict[str, tuple[bool, Any]]:
    attrs = parent_group.attrs
    return {
        name: (name in attrs, copy.deepcopy(attrs.get(name)))
        for name in _RUN_SELECTOR_ATTRS
    }


def _restore_parent_selectors(
    parent_group: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
) -> None:
    attrs = parent_group.attrs
    failures: list[str] = []
    for name, (present, value) in snapshot.items():
        try:
            if present:
                attrs[name] = copy.deepcopy(value)
            elif name in attrs:
                del attrs[name]
        except BaseException as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(f"{name}: {exc}")
    for name, (present, value) in snapshot.items():
        try:
            if (name in attrs) != present or (
                present and attrs.get(name) != value
            ):
                failures.append(f"{name}: persisted value differs")
        except BaseException as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(f"{name}: verification failed: {exc}")
    if failures:
        raise RuntimeError(
            f"Detection selector rollback was incomplete: {failures!r}."
        )


def _selector_collisions(
    snapshot: Mapping[str, tuple[bool, Any]],
    run_name: str,
) -> tuple[str, ...]:
    """Return selector pointers that already reference an absent candidate name."""

    collisions: list[str] = []
    for name in ("latest", "latest_complete", "latest_pending", "authoritative_run"):
        present, value = snapshot[name]
        if present and str(value).strip() == str(run_name):
            collisions.append(name)
    return tuple(collisions)


def _publish_validated_detection_selection(
    parent_group: Any,
    run_group: Any,
    *,
    run_name: str,
) -> None:
    """Prepare parent pointers while ineligible, then expose the run last."""

    if run_group.attrs.get("palette_run_completion_status") != "complete":
        raise RuntimeError("Detection selection requires an explicitly complete run.")
    if run_group.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError(
            "Detection selection requires a freshly validated ineligible candidate."
        )
    name = str(run_name)
    parent_group.attrs[RUN_LATEST_COMPLETE_ATTR] = name
    parent_group.attrs["latest"] = name
    # This is the publication commit point. Do not perform fallible work after it.
    run_group.attrs["stage_selector_eligible"] = True


def _disarm_failed_run_selectors(parent_group: Any, run_name: str) -> list[str]:
    """Ensure a retained or deleted failed run cannot remain selected."""

    failures: list[str] = []
    attrs = parent_group.attrs
    authoritative_cleared = False
    for name in ("latest", "latest_complete", "latest_pending", "authoritative_run"):
        try:
            if name not in attrs or str(attrs.get(name)).strip() != str(run_name):
                continue
            del attrs[name]
            authoritative_cleared = authoritative_cleared or name == "authoritative_run"
        except BaseException as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(f"disarm {name}: {exc}")
    if authoritative_cleared:
        try:
            if "authoritative_run_provenance" in attrs:
                del attrs["authoritative_run_provenance"]
        except BaseException as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(f"disarm authoritative_run_provenance: {exc}")
    return failures


def _next_run_name(parent_group: Any, *, prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{prefix}_{timestamp}"
    name = base
    suffix = 1
    while name in parent_group:
        name = f"{base}_{suffix:03d}"
        suffix += 1
    return name


def _iter_zarr_nodes(node: Any) -> Sequence[Any]:
    nodes: list[Any] = [node]
    group_keys = getattr(node, "group_keys", None)
    if callable(group_keys):
        for name in tuple(group_keys()):
            try:
                nodes.extend(_iter_zarr_nodes(node[name]))
            except BaseException as exc:
                raise RuntimeError(
                    f"Unable to traverse detection child group {name!r}: {exc}"
                ) from exc
    array_keys = getattr(node, "array_keys", None)
    if callable(array_keys):
        for name in tuple(array_keys()):
            try:
                nodes.append(node[name])
            except BaseException as exc:
                raise RuntimeError(
                    f"Unable to traverse detection array {name!r}: {exc}"
                ) from exc
    return nodes


def _strip_canonical_detection_publication_attrs(run_group: Any) -> list[str]:
    failures: list[str] = []
    for node in _iter_zarr_nodes(run_group):
        attrs = getattr(node, "attrs", None)
        if attrs is None:
            continue
        for name in _CANONICAL_DETECTION_PUBLICATION_ATTRS:
            if name not in attrs:
                continue
            try:
                del attrs[name]
            except BaseException as exc:  # pragma: no cover - hostile persistent mapping
                failures.append(f"{getattr(node, 'path', '<node>')}:{name}: {exc}")
    return failures


def _delete_new_failed_run(
    parent_group: Any,
    *,
    run_group: Any,
    run_name: str,
) -> bool:
    try:
        if run_name not in parent_group:
            return True
        persisted = parent_group[run_name]
        if canonical_node_path(persisted) != canonical_node_path(run_group):
            return False
        del parent_group[run_name]
        return run_name not in parent_group
    except BaseException:  # pragma: no cover - hostile persistent mapping
        return False


def _rollback_failed_detection_run(
    *,
    run_group: Any,
    parent_group: Any,
    run_name: str,
    selector_snapshot: Mapping[str, tuple[bool, Any]] | None,
    coordinate_checkpoints: Sequence[Any],
    cause: BaseException,
    error: str,
    cleanup_failures: Sequence[str] = (),
    delete_run_when_safe: bool = False,
) -> None:
    failures: list[str] = list(cleanup_failures)
    for checkpoint in reversed(tuple(coordinate_checkpoints)):
        try:
            restore_observation_coordinate_publication_checkpoint(
                checkpoint,
                cause=cause,
            )
        except BaseException as exc:
            failures.append(f"coordinate attrs: {exc}")
    try:
        failures.extend(_strip_canonical_detection_publication_attrs(run_group))
    except BaseException as exc:  # pragma: no cover - hostile persistent mapping
        failures.append(f"canonical attr traversal: {exc}")
    try:
        if RUN_COMPLETED_AT_ATTR in run_group.attrs:
            del run_group.attrs[RUN_COMPLETED_AT_ATTR]
        run_group.attrs["stage_selector_eligible"] = False
        # Pointer restoration is handled separately and exactly below.
        mark_run_failed(
            run_group,
            parent_group=None,
            run_name=run_name,
            error=error,
        )
    except BaseException as exc:  # pragma: no cover - hostile persistent mapping
        failures.append(f"failed-state marker: {exc}")
    if selector_snapshot is not None:
        try:
            _restore_parent_selectors(parent_group, selector_snapshot)
        except BaseException as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(f"selectors: {exc}")
    if delete_run_when_safe:
        deleted = _delete_new_failed_run(
            parent_group,
            run_group=run_group,
            run_name=run_name,
        )
        if not deleted:
            failures.append("failed run deletion")
        failures.extend(_disarm_failed_run_selectors(parent_group, run_name))
    if failures:
        raise RuntimeError(
            "Detection publication failed and rollback was incomplete: "
            f"{failures!r}."
        ) from cause


def _load_published_detection_acquisition_frame(root: Any) -> Any:
    """Select canonical detection solely from typed published acquisition state."""

    status = load_acquisition_authority_publication_status(root)
    if status.status != ACQUISITION_AUTHORITY_PUBLISHED:
        raise ValueError(
            "Canonical detection requires a published acquisition authority; "
            f"found status={status.status!r}, reason={status.reason_code!r}."
        )
    ownership, acquisition = load_persisted_acquisition_camera_authority(root)
    expected_path = (
        f"analysis/acquisition_camera_frames/{acquisition.record.camera_id}"
    )
    if (
        status.authority_mode != ownership.record.mode
        or status.authority_path != expected_path
    ):
        raise ValueError(
            "Published acquisition status mode/path disagrees with the exact "
            "persisted acquisition authority."
        )
    ownership.assert_verified()
    acquisition.assert_verified()
    return acquisition


def _load_full_acquisition_video_source(
    root: Any,
    *,
    video_path: Path,
    output_zarr: Path,
    decoded_frame_count: int,
    decoded_width: int,
    decoded_height: int,
) -> Any:
    """Seal a full-video identity mapping from live locator/fingerprint evidence."""

    acquisition = _load_published_detection_acquisition_frame(root)
    resolved = resolve_source_video(
        root,
        zarr_path=output_zarr,
        require_exists=True,
    )
    requested = video_path.expanduser().resolve()
    if resolved.path.expanduser().resolve() != requested:
        raise ValueError(
            "Canonical detection requires the exact acquisition source-video "
            f"locator: requested={requested}, authority={resolved.path}."
        )
    metadata = acquisition.record.source_video_metadata
    fingerprint = metadata.get("file_fingerprint")
    if not isinstance(fingerprint, Mapping) or fingerprint.get("strategy") != "stat_v1":
        raise ValueError(
            "Canonical external-video detection currently requires the exact "
            "stat_v1 acquisition fingerprint; unsupported evidence must fail closed."
        )
    live = source_stat_fingerprint_attrs(
        requested,
        attr_prefix="source_video",
        extra={
            "codec": metadata.get("codec"),
            "pix_fmt": metadata.get("pix_fmt"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "fps": metadata.get("fps"),
            "frame_count": metadata.get("total_frames"),
        },
    )
    expected_fingerprint = {
        "strategy": live["source_video_fingerprint_strategy"],
        "value": live["source_video_fingerprint"],
        "size_bytes": live["source_video_size_bytes"],
        "mtime_ns": live["source_video_mtime_ns"],
        "relocation_stable": False,
    }
    if dict(fingerprint) != expected_fingerprint:
        raise ValueError(
            "Live detection video differs from the exact acquisition source "
            "locator/fingerprint; decode-to-acquisition identity is unproven."
        )
    observed = (int(decoded_width), int(decoded_height), int(decoded_frame_count))
    expected = (
        int(acquisition.record.width_px),
        int(acquisition.record.height_px),
        int(acquisition.record.source_total_frames),
    )
    if observed != expected:
        raise ValueError(
            "Canonical detection requires a full untrimmed acquisition decode: "
            f"observed width/height/frames={observed}, authority={expected}."
        )
    return acquisition


def _restore_detection_coordinate_checkpoints(
    checkpoints: Sequence[Any],
    *,
    cause: BaseException,
) -> list[str]:
    """Restore every checkpoint even when rollback itself sees BaseException."""

    failures: list[str] = []
    for checkpoint in reversed(tuple(checkpoints)):
        try:
            restore_observation_coordinate_publication_checkpoint(
                checkpoint,
                cause=cause,
            )
        except BaseException as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(str(exc))
    return failures


def _publish_detection_frame_evidence(
    root: Any,
    run_group: Any,
    *,
    acquisition_frame: Any,
) -> tuple[BoundDetectionFrameEvidence, tuple[Any, ...]]:
    camera_id = acquisition_frame.record.camera_id
    camera_node = _require_group_path(
        root,
        (
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_POINT_PIXEL_CONVENTION}"
        ),
    )
    bbox_camera_node = _require_group_path(
        root,
        (
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_BBOX_PIXEL_CONVENTION}"
        ),
    )
    checkpoints: list[Any] = [
        capture_observation_coordinate_publication_checkpoint(
            camera_node,
            bbox_camera_node,
        )
    ]
    try:
        # Source-camera stamping is included in the same transaction as the
        # run-local normalized frame and transform.  It is commonly idempotent,
        # but its exact pre-publication attrs are still rollback authority.
        source_camera = stamp_source_camera_pixel_frame_authority(
            camera_node,
            # This is the archive-wide continuous source-camera authority.
            # Stimulus/calibration publishers share the same write-once node,
            # so its frame identity must be producer-independent.
            frame_id=f"{camera_id}_source_camera",
            pixel_convention=SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
            acquisition_frame=acquisition_frame,
        )
        bbox_source_camera = stamp_source_camera_pixel_frame_authority(
            bbox_camera_node,
            frame_id=f"{camera_id}_source_camera_pixel_edge_half_open",
            pixel_convention=SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
            acquisition_frame=acquisition_frame,
        )
        frame_group = run_group.require_group("coordinate_frames")
        normalized_node = frame_group.require_group("source_camera_normalized")
        checkpoints.append(
            capture_observation_coordinate_publication_checkpoint(normalized_node)
        )
        token = hashlib.sha256(str(run_group.path).encode("utf-8")).hexdigest()[:16]
        normalized = stamp_normalized_pixel_frame_authority(
            normalized_node,
            frame_id=f"detect_source_camera_normalized_{token}",
            pixel_frame=bbox_source_camera,
        )
        transform_group = run_group.require_group("coordinate_transforms")
        expected_matrix = normalized_to_pixel_matrix(bbox_source_camera)
        if "source_camera_normalized_to_image" in transform_group:
            matrix_node = transform_group["source_camera_normalized_to_image"]
            if not np.array_equal(np.asarray(matrix_node[:]), expected_matrix):
                raise ValueError(
                    "Existing normalized-to-image matrix payload changed before "
                    "final publication."
                )
        else:
            matrix_node = transform_group.create_array(
                "source_camera_normalized_to_image",
                data=expected_matrix,
                chunks=(3, 3),
            )
        authority_node = transform_group.require_group(
            "source_camera_normalized_to_image_authority"
        )
        checkpoints.append(
            capture_observation_coordinate_publication_checkpoint(
                matrix_node,
                authority_node,
            )
        )
        authority = stamp_normalized_to_pixel_transform_authority(
            authority_node,
            authority_id=f"detect_source_camera_normalized_to_image_{token}",
            matrix_node=matrix_node,
            source_frame=normalized,
            target_frame=bbox_source_camera,
        )
        transform = stamp_directed_transform_v2(
            matrix_node,
            transform_id=f"detect_source_camera_normalized_to_image_{token}",
            authority=authority,
            source_frame=normalized,
            target_frame=bbox_source_camera,
        )
        evidence = build_bound_detection_frame_evidence(
            source_camera_frame=source_camera,
            bbox_source_camera_frame=bbox_source_camera,
            normalized_frame=normalized,
            normalized_to_source_camera=resolve_bound_directed_transform_chain(
                (transform,)
            ),
        )
    except BaseException as exc:
        failures = _restore_detection_coordinate_checkpoints(
            checkpoints,
            cause=exc,
        )
        if failures:
            raise RuntimeError(
                "Detection frame-evidence publication failed and checkpoint "
                f"rollback was incomplete: {failures!r}."
            ) from exc
        raise
    return evidence, tuple(checkpoints)


def _array_mapping_payload(node: Any) -> dict[str, Any]:
    values = np.asarray(node[:])
    return {
        "array_ref": f"/{canonical_node_path(node)}",
        "dtype": values.dtype.str,
        "shape": [int(value) for value in values.shape],
        "content_sha256": array_payload_sha256(node),
    }


def _publish_detection_acquisition_mapping(
    run_group: Any,
    *,
    acquisition_frame: Any,
) -> BoundCoordinateRecord:
    decode_node = run_group["frame_indices"]
    acquisition_node = run_group["source_acquisition_frame_index"]
    decode_frames = np.asarray(decode_node[:])
    acquisition_frames = np.asarray(acquisition_node[:])
    if (
        decode_frames.dtype.kind not in "iu"
        or acquisition_frames.dtype != np.dtype("<i8")
        or decode_frames.shape != acquisition_frames.shape
        or not np.array_equal(decode_frames.astype(np.int64), acquisition_frames)
    ):
        raise ValueError(
            "Canonical detection rows must preserve the mechanically proven "
            "full-video identity mapping into acquisition frame indices."
        )
    metadata = acquisition_frame.record.source_video_metadata
    record = {
        "schema_id": DETECTION_ACQUISITION_MAPPING_SCHEMA_ID,
        "schema_version": DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION,
        "operation": "full_untrimmed_video_decode_identity_to_acquisition_v1",
        "direction": "decode_frame_index_to_source_acquisition_frame_index",
        "decode_frame_index": _array_mapping_payload(decode_node),
        "source_acquisition_frame_index": _array_mapping_payload(acquisition_node),
        "acquisition_camera_frame": {
            "record_ref": acquisition_frame.record_ref,
            "record_sha256": acquisition_frame.record_sha256,
        },
        "source_video_locator": metadata["locator"],
        "source_video_fingerprint": metadata["file_fingerprint"],
        "source_total_frames": int(acquisition_frame.record.source_total_frames),
        "proof": (
            "exact_locator_and_stat_fingerprint_revalidated_after_full_decode"
        ),
    }
    return stamp_and_bind_persisted_coordinate_record(
        run_group,
        record,
        attr_name=DETECTION_ACQUISITION_MAPPING_ATTR,
    )


def _publish_empty_detection_observation_declaration(
    run_group: Any,
    *,
    acquisition_frame: Any,
    decoded_frame_count: int,
    decode_domain_proof: str,
) -> BoundCoordinateRecord:
    """Bind a genuine empty rowset through the shared consumer schema."""

    return publish_empty_detection_observation_declaration(
        run_group,
        acquisition_frame=acquisition_frame,
        decoded_frame_count=decoded_frame_count,
        decode_domain_proof=decode_domain_proof,
    )


def _decord_available() -> bool:
    return decord is not None and VideoReader is not None and cpu is not None


def _normalize_decode_backend(value: Optional[str]) -> str:
    if value is None:
        return DECODE_BACKEND_AUTO
    normalized = str(value).strip().lower()
    if not normalized:
        return DECODE_BACKEND_AUTO
    if normalized not in DECODE_BACKEND_CHOICES:
        choices = ", ".join(DECODE_BACKEND_CHOICES)
        raise ValueError(f"Unsupported decode backend {value!r}; expected one of: {choices}")
    return normalized


def _collect_zarr_metadata(zarr_path: Path, console: Optional[Console] = None) -> Dict[str, Any]:
    """
    Return palette metadata (width, height, fps, etc.) from an existing Zarr archive.

    The lookup is best-effort: we consult root attributes, then raw_video attrs,
    and finally dataset shapes. Missing entries are omitted from the result.
    """
    metadata: Dict[str, Any] = {}
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception as exc:  # pragma: no cover - best-effort metadata lookup
        if console:
            console.print(f"[yellow]Warning:[/yellow] Unable to open source Zarr '{zarr_path}': {escape(str(exc))}")
        return metadata

    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _get_attr(source, *names) -> Optional[Any]:
        for name in names:
            if name in source:
                return source.get(name)
        return None

    # Root-level metadata
    width_attr = _as_int(_get_attr(root.attrs, "width", "video_width"))
    height_attr = _as_int(_get_attr(root.attrs, "height", "video_height"))
    fps = _get_attr(root.attrs, "fps", "video_fps")
    total_frames = _get_attr(root.attrs, "total_frames", "n_frames")
    duration_seconds = _get_attr(root.attrs, "duration_seconds", "video_duration_seconds")
    video_codec = _get_attr(root.attrs, "video_codec")
    video_pix_fmt = _get_attr(root.attrs, "video_pix_fmt")

    # raw_video group metadata provides additional detail
    raw_group = root.get("raw_video")
    if raw_group is not None:
        raw_attrs = raw_group.attrs
        width_attr = width_attr or _as_int(_get_attr(raw_attrs, "video_width"))
        height_attr = height_attr or _as_int(_get_attr(raw_attrs, "video_height"))
        fps = fps or raw_attrs.get("fps")
        total_frames = total_frames or raw_attrs.get("total_frames")
        duration_seconds = duration_seconds or raw_attrs.get("video_duration_seconds")
        video_codec = video_codec or raw_attrs.get("video_codec")
        video_pix_fmt = video_pix_fmt or raw_attrs.get("video_pix_fmt")

        original_resolution = raw_attrs.get("original_resolution")
        if original_resolution and len(original_resolution) == 2:
            metadata["original_resolution"] = [int(original_resolution[0]), int(original_resolution[1])]

        if "has_full_resolution" in raw_attrs:
            metadata["has_full_resolution"] = bool(raw_attrs.get("has_full_resolution"))
        else:
            metadata["has_full_resolution"] = "images_full" in raw_group

        if "has_downsampled" in raw_attrs:
            metadata["has_downsampled"] = bool(raw_attrs.get("has_downsampled"))
        else:
            metadata["has_downsampled"] = "images_ds" in raw_group

        downsampled_resolution = raw_attrs.get("downsampled_resolution")
        if downsampled_resolution and len(downsampled_resolution) == 2:
            metadata["downsampled_resolution"] = [int(downsampled_resolution[0]), int(downsampled_resolution[1])]

        downsample_method = raw_attrs.get("downsample_method")
        if downsample_method:
            metadata["downsample_method"] = downsample_method

        for key in ("images_full", "images", "images_ds"):
            if key in raw_group:
                shape = raw_group[key].shape
                if shape and len(shape) >= 3:
                    height = int(shape[-2])
                    width = int(shape[-1])
                    if key == "images_full":
                        width_attr = width_attr or width
                        height_attr = height_attr or height
                        metadata.setdefault("original_resolution", [height, width])
                    if key == "images_ds":
                        metadata.setdefault("downsampled_resolution", [height, width])

    if width_attr is not None:
        metadata["full_width"] = int(width_attr)
    if height_attr is not None:
        metadata["full_height"] = int(height_attr)
    if fps is not None:
        metadata["fps"] = float(fps)
    if total_frames is not None:
        metadata["total_frames"] = int(total_frames)
    if duration_seconds is not None:
        metadata["duration_seconds"] = float(duration_seconds)
    if video_codec:
        metadata["video_codec"] = video_codec
    if video_pix_fmt:
        metadata["video_pix_fmt"] = video_pix_fmt

    return metadata


def _gpu_decode_unavailable(reason: str) -> RuntimeError:
    return RuntimeError(
        "GPU decode unavailable; refusing CPU fallback - pixels would differ from the "
        f"production path ({reason})"
    )


def _close_partially_initialized_decord_reader(reader: Any) -> None:
    """Close a Decord-like reader that failed before attempt ownership transfer."""

    close = getattr(reader, "close", None)
    if callable(close):
        close()
        return
    release = getattr(reader, "release", None)
    if callable(release):
        release()


def _raise_after_decord_init_cleanup(
    exc: BaseException,
    *,
    reader: Any | None,
    message: str,
    gpu_failure: bool,
) -> None:
    cleanup_failure: BaseException | None = None
    if reader is not None:
        try:
            _close_partially_initialized_decord_reader(reader)
        except BaseException as cleanup_exc:  # pragma: no cover - hostile backend
            cleanup_failure = cleanup_exc
    if cleanup_failure is not None:
        raise RuntimeError(
            "Decord initialization failed and its partial reader could not be "
            f"closed: {cleanup_failure}."
        ) from exc
    if not isinstance(exc, Exception):
        raise exc
    if gpu_failure:
        raise _gpu_decode_unavailable(message) from exc
    raise RuntimeError(message) from exc


def _init_decord_reader(video_path: Path, prefer_gpu: bool, console: Console) -> Optional[Dict[str, Any]]:
    """Initialise an explicitly requested Decord VideoReader without backend fallback."""
    if not _decord_available():
        if _DECORD_IMPORT_ERROR:
            console.print(f"[yellow]Decord unavailable: {escape(str(_DECORD_IMPORT_ERROR))}[/yellow]")
        return None

    if prefer_gpu:
        if not torch.cuda.is_available():
            raise _gpu_decode_unavailable("Decord GPU requested but CUDA is unavailable")
        vr = None
        try:
            decord.bridge.set_bridge('torch')
            vr = VideoReader(str(video_path), ctx=gpu(0))
            first = vr[0]
            height, width = int(first.shape[0]), int(first.shape[1])
            fps = vr.get_avg_fps()
            console.print("[green]✓[/green] Using Decord GPU decoder")
            return {
                'reader': vr,
                'type': 'decord_gpu',
                'on_gpu': True,
                'width': width,
                'height': height,
                'fps': fps,
            }
        except BaseException as exc:
            _raise_after_decord_init_cleanup(
                exc,
                reader=vr,
                message=f"Decord GPU decoder failed: {exc}",
                gpu_failure=True,
            )

    vr = None
    try:
        decord.bridge.set_bridge('native')
        vr = VideoReader(str(video_path), ctx=cpu())
        first = vr[0]
        height, width = int(first.shape[0]), int(first.shape[1])
        fps = vr.get_avg_fps()
        console.print("[green]✓[/green] Using Decord CPU decoder")
        return {
            'reader': vr,
            'type': 'decord_cpu',
            'on_gpu': False,
            'width': width,
            'height': height,
            'fps': fps,
        }
    except BaseException as exc:
        _raise_after_decord_init_cleanup(
            exc,
            reader=vr,
            message=f"Requested Decord CPU decoder failed: {exc}",
            gpu_failure=False,
        )
    raise AssertionError("Decord initialization cleanup unexpectedly returned.")


def get_video_metadata(video_path: Path, cap: Optional[cv2.VideoCapture], width: int, height: int, n_frames: int, fps: float) -> Dict[str, Any]:
    """
    Get finite source metadata from decoded observations plus ffprobe.
    """
    cap_owner = False
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        cap_owner = True
        if not cap.isOpened():
            cap.release()
            cap = None
    try:
        meta = {
            "source_video": str(video_path.name),
            "source_path": str(video_path.absolute()),
            "width": width,
            "height": height,
            "total_frames": n_frames,
            "fps": fps,
            "duration_seconds": n_frames / fps if fps > 0 else 0,
        }
        stream = probe_ffprobe_video_metadata(video_path)
        meta["codec"] = stream.get("codec", "unknown")
        meta["pix_fmt"] = stream.get("pix_fmt", "unknown")
        if meta["codec"] == "unknown":
            if cap is not None:
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                if fourcc > 0:
                    codec_str = "".join(
                        chr((fourcc >> 8 * i) & 0xFF) for i in range(4)
                    )
                    meta["codec"] = codec_str
                else:
                    meta["codec"] = "unknown"
        tags = stream.get("format_tags")
        if isinstance(tags, dict) and tags:
            meta["format_tags"] = tags
        return meta
    finally:
        if cap_owner and cap is not None:
            cap.release()


def _read_cv2_video_properties(video_path: Path) -> Tuple[int, float, int, int]:
    """Return (n_frames, fps, width, height) using OpenCV metadata only."""

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video metadata: {video_path}")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return n_frames, fps, width, height


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    if config_path is None:
        # Try default locations
        default_paths = [
            Path('yolo_detect_config.yaml'),  # Current directory
            Path('configs/fisheye/yolo_detect_config.yaml'),  # Standard location
            Path(__file__).parent.parent.parent / 'configs/fisheye/yolo_detect_config.yaml',  # Relative to module
            Path.home() / 'gitrepos/palette/configs/fisheye/yolo_detect_config.yaml',  # Absolute
            Path('src/fisheye/yolo_detect_config.yaml'),  # Old location
            Path(__file__).parent / 'yolo_detect_config.yaml',  # Same dir as this script
        ]
        
        console = Console()
        console.print("[dim]Searching for config file...[/dim]")
        
        for path in default_paths:
            console.print(f"[dim]  Checking: {path}[/dim]")
            if path.exists():
                console.print(f"[green]  ✓ Found config: {path}[/green]")
                config_path = path
                break
        
        if config_path is None:
            console.print("[yellow]  No config file found, using defaults[/yellow]")
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    return {}


def _normalize_imgsz(value: Optional[Any]) -> Optional[int | list[int]]:
    """Normalize imgsz to Ultralytics-compatible int or [h, w] list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        ints: list[int] = []
        for item in value:
            if item is None:
                continue
            try:
                ints.append(int(item))
            except (TypeError, ValueError):
                continue
        if not ints:
            return None
        if len(ints) == 1:
            return ints[0]
        return [ints[0], ints[1]]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _imgsz_to_resize_dims(value: Optional[int | list[int]]) -> Optional[list[int]]:
    """Normalize imgsz-like values to canonical [h, w] resize dims."""
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return [value, value]
    if not value:
        return None
    if len(value) == 1:
        edge = int(value[0])
        if edge <= 0:
            return None
        return [edge, edge]
    height = int(value[0])
    width = int(value[1])
    if height <= 0 or width <= 0:
        return None
    return [height, width]


def _normalize_resize_dims(value: Optional[Any]) -> Optional[list[int]]:
    """Normalize canonical resize dims to [h, w]."""
    return _imgsz_to_resize_dims(_normalize_imgsz(value))


def _normalize_legacy_video_resize(value: Optional[Any]) -> Optional[list[int]]:
    """
    Normalize legacy video.resize [w, h] to canonical [h, w].

    Historical detect configs use video.resize as [width, height].
    """
    parsed = _normalize_imgsz(value)
    if parsed is None:
        return None
    if isinstance(parsed, int):
        return _imgsz_to_resize_dims(parsed)
    if len(parsed) == 1:
        return _imgsz_to_resize_dims(parsed)
    width = int(parsed[0])
    height = int(parsed[1])
    if width <= 0 or height <= 0:
        return None
    return [height, width]


def _resize_dims_to_imgsz(value: Optional[list[int]]) -> Optional[int | list[int]]:
    """Convert canonical [h, w] dims to an Ultralytics imgsz value."""
    if value is None:
        return None
    if value[0] == value[1]:
        return int(value[0])
    return [int(value[0]), int(value[1])]


def _actual_input_resize_dims(
    requested_resize_dims: Optional[list[int]],
    pre_resize_dims: Optional[list[int]],
    *,
    decord_on_gpu: bool,
    tensor_input: bool = False,
) -> Optional[list[int]]:
    """
    Return the frame dimensions that will be explicitly fed to YOLO.

    Ultralytics applies imgsz preprocessing for numpy/list inputs, but torch
    tensor inputs are treated as already prepared. Tensor-backed decoder paths
    must apply canonical detection.resize_dims explicitly or inference silently
    runs at source-video resolution.
    """
    if decord_on_gpu or tensor_input:
        if requested_resize_dims is None:
            raise RuntimeError(
                "GPU tensor decode paths require detection.resize_dims or --resize-dims. "
                "Without an explicit resize, Ultralytics treats tensor inputs as already "
                "prepared and can run inference at source-video resolution."
            )
        return [int(requested_resize_dims[0]), int(requested_resize_dims[1])]
    if pre_resize_dims is not None:
        return [int(pre_resize_dims[0]), int(pre_resize_dims[1])]
    return None


def _record_timing(timings: Dict[str, float], key: str, start: float) -> float:
    """Accumulate a stage duration from a perf-counter start timestamp."""
    elapsed = time.perf_counter() - start
    timings[key] = float(timings.get(key, 0.0) + elapsed)
    return elapsed


def _aligned_shards(chunks: Sequence[int], shard_rows: int | None) -> tuple[int, ...] | None:
    if shard_rows is None:
        return None
    requested = int(shard_rows)
    if requested <= 0:
        raise ValueError("Detection shard rows must be positive.")
    inner_rows = int(chunks[0])
    outer_rows = int(((requested + inner_rows - 1) // inner_rows) * inner_rows)
    return (outer_rows, *tuple(int(value) for value in chunks[1:]))


def _digest_values(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _digest_zarr_array(array: zarr.Array, *, row_step: int) -> str:
    digest = hashlib.sha256()
    step = max(1, int(row_step))
    for start in range(0, int(array.shape[0]), step):
        stop = min(start + step, int(array.shape[0]))
        values = np.ascontiguousarray(array[start:stop, ...])
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _write_detection_output_arrays(
    detect_group: zarr.Group,
    *,
    frame_indices: np.ndarray,
    bbox_coords: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    instance_keys: np.ndarray,
    frame_counts: np.ndarray,
    det_chunk: int,
    detect_row_shard_rows: int | None,
    detect_frame_shard_rows: int,
    source_acquisition_frame_indices: np.ndarray | None = None,
    bbox_img_xyxy: np.ndarray | None = None,
    centers_img_xy: np.ndarray | None = None,
    identity_array_name: str = "instance_key",
) -> dict[str, Any] | None:
    """Write materialized YOLO detections, optionally as complete indexed shards."""

    identity_name = str(identity_array_name).strip()
    if identity_name not in {"instance_key", DETECTION_ARTIFACT_ROW_ID_ARRAY}:
        raise ValueError(f"Unsupported detection identity array name: {identity_name!r}.")
    detection_values = {
        "frame_indices": np.asarray(frame_indices),
        "bbox_norm_coords": np.asarray(bbox_coords),
        "scores": np.asarray(scores),
        "class_ids": np.asarray(class_ids),
        identity_name: np.asarray(instance_keys),
    }
    canonical_values = (
        source_acquisition_frame_indices,
        bbox_img_xyxy,
        centers_img_xy,
    )
    if any(value is not None for value in canonical_values):
        if not all(value is not None for value in canonical_values):
            raise ValueError(
                "Canonical detection coordinates require source acquisition "
                "frames, source-camera bboxes, and source-camera centers together."
            )
        detection_values.update(
            {
                "source_acquisition_frame_index": np.asarray(
                    source_acquisition_frame_indices
                ),
                "bbox_img_xyxy": np.asarray(bbox_img_xyxy),
                "centers_img_xy": np.asarray(centers_img_xy),
            }
        )
    frame_values = {
        "n_detections": np.asarray(frame_counts),
        "frame_counts": np.asarray(frame_counts),
    }
    detection_chunks = {
        "frame_indices": (int(det_chunk),),
        "bbox_norm_coords": (int(det_chunk), 4),
        "scores": (int(det_chunk),),
        "class_ids": (int(det_chunk),),
        identity_name: (int(det_chunk),),
    }
    if all(value is not None for value in canonical_values):
        detection_chunks.update(
            {
                "source_acquisition_frame_index": (int(det_chunk),),
                "bbox_img_xyxy": (int(det_chunk), 4),
                "centers_img_xy": (int(det_chunk), 2),
            }
        )

    if detect_row_shard_rows is None:
        for name, values in detection_values.items():
            detect_group.create_array(
                name,
                data=values,
                chunks=detection_chunks[name],
                overwrite=True,
            )
        for name, values in frame_values.items():
            create_geometry_preload_array(
                detect_group,
                name,
                data=values,
                overwrite=True,
            )
        return None

    count_chunks = geometry_preload_chunks_for_data(frame_counts) or (1,)
    detection_shards = {
        name: _aligned_shards(chunks, int(detect_row_shard_rows))
        for name, chunks in detection_chunks.items()
    }
    frame_shards = _aligned_shards(count_chunks, int(detect_frame_shard_rows))

    destinations: dict[str, zarr.Array] = {}
    for name, values in detection_values.items():
        shards = detection_shards[name]
        assert shards is not None
        destinations[name] = detect_group.create_array(
            name,
            shape=values.shape,
            dtype=values.dtype,
            chunks=detection_chunks[name],
            shards=shards,
            overwrite=True,
        )
    for name, values in frame_values.items():
        assert frame_shards is not None
        destinations[name] = create_geometry_preload_array(
            detect_group,
            name,
            shape=values.shape,
            dtype=values.dtype,
            chunks=count_chunks,
            shards=frame_shards,
            overwrite=True,
        )

    all_values = {**detection_values, **frame_values}
    source_hashes = {name: _digest_values(values) for name, values in all_values.items()}
    write_started = time.perf_counter()
    first_detection_shards = next(iter(detection_shards.values()))
    assert first_detection_shards is not None
    effective_detection_rows = int(first_detection_shards[0])
    for start in range(0, int(frame_indices.shape[0]), effective_detection_rows):
        stop = min(start + effective_detection_rows, int(frame_indices.shape[0]))
        for name, values in detection_values.items():
            destinations[name][start:stop, ...] = values[start:stop, ...]
    assert frame_shards is not None
    effective_frame_rows = int(frame_shards[0])
    for start in range(0, int(frame_counts.shape[0]), effective_frame_rows):
        stop = min(start + effective_frame_rows, int(frame_counts.shape[0]))
        for name, values in frame_values.items():
            destinations[name][start:stop, ...] = values[start:stop, ...]
    write_seconds = float(time.perf_counter() - write_started)

    validation_started = time.perf_counter()
    destination_hashes = {
        name: _digest_zarr_array(
            array,
            row_step=(effective_frame_rows if name in frame_values else effective_detection_rows),
        )
        for name, array in destinations.items()
    }
    validation_seconds = float(time.perf_counter() - validation_started)
    if source_hashes != destination_hashes:
        raise RuntimeError(
            "YOLO detection shard digest mismatch: "
            f"source={source_hashes} destination={destination_hashes}"
        )
    return {
        "schema_id": DETECT_SHARD_WRITE_SCHEMA,
        "status": "complete",
        "write_mode": "materialized_complete_shards",
        "detection_row_count": int(frame_indices.shape[0]),
        "frame_row_count": int(frame_counts.shape[0]),
        "detect_row_shard_rows_requested": int(detect_row_shard_rows),
        "detect_row_shard_rows_effective": effective_detection_rows,
        "detect_frame_shard_rows_requested": int(detect_frame_shard_rows),
        "detect_frame_shard_rows_effective": effective_frame_rows,
        "write_seconds": write_seconds,
        "validation_seconds": validation_seconds,
        "source_sha256_by_array": source_hashes,
        "destination_sha256_by_array": destination_hashes,
        "exact_match": True,
    }


def _validate_unbound_detection_artifact_storage(
    run_group: Any,
    *,
    source_total_frames: int,
) -> dict[str, Any]:
    """Validate an artifact rowset without granting canonical row identity."""

    arrays = {str(name): array for name, array in run_group.arrays()}
    required = {
        "frame_indices",
        "bbox_norm_coords",
        "scores",
        "class_ids",
        DETECTION_ARTIFACT_ROW_ID_ARRAY,
        "frame_counts",
        "n_detections",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise RuntimeError(
            f"Unbound detection artifact is missing required arrays: {missing!r}."
        )
    if "instance_key" in arrays or any(
        str(name).startswith("instance_key_") for name in run_group.attrs
    ):
        raise RuntimeError(
            "Unbound detection artifacts must not expose canonical instance_key "
            "arrays or identity metadata."
        )

    frame_indices = np.asarray(arrays["frame_indices"][:])
    row_count = int(frame_indices.shape[0]) if frame_indices.ndim == 1 else -1
    row_specs = {
        "frame_indices": (np.dtype("<i4"), (row_count,)),
        "bbox_norm_coords": (np.dtype("<f8"), (row_count, 4)),
        "scores": (np.dtype("<f4"), (row_count,)),
        "class_ids": (np.dtype("<i4"), (row_count,)),
        DETECTION_ARTIFACT_ROW_ID_ARRAY: (np.dtype("<u8"), (row_count,)),
    }
    errors: list[str] = []
    for name, (expected_dtype, expected_shape) in row_specs.items():
        node = arrays[name]
        actual_shape = tuple(int(value) for value in node.shape)
        actual_dtype = np.dtype(node.dtype)
        if actual_shape != expected_shape or actual_dtype != expected_dtype:
            errors.append(
                f"{name} shape/dtype={actual_shape}/{actual_dtype}, expected "
                f"{expected_shape}/{expected_dtype}"
            )

    artifact_row_ids = np.asarray(
        arrays[DETECTION_ARTIFACT_ROW_ID_ARRAY][:],
        dtype=np.uint64,
    )
    if not np.array_equal(
        artifact_row_ids,
        np.arange(max(0, row_count), dtype=np.uint64),
    ):
        errors.append("artifact_row_id is not the exact run-local dense row range")

    frame_count = int(source_total_frames)
    if frame_count < 0:
        errors.append("source_total_frames must be nonnegative")
    frame_counts = np.asarray(arrays["frame_counts"][:])
    n_detections = np.asarray(arrays["n_detections"][:])
    expected_frame_shape = (frame_count,)
    for name, values in (
        ("frame_counts", frame_counts),
        ("n_detections", n_detections),
    ):
        if values.shape != expected_frame_shape or values.dtype != np.dtype("<i4"):
            errors.append(
                f"{name} shape/dtype={values.shape}/{values.dtype}, expected "
                f"{expected_frame_shape}/int32"
            )
    if (
        frame_indices.dtype == np.dtype("<i4")
        and frame_indices.shape == (max(0, row_count),)
        and frame_count >= 0
    ):
        if np.any(frame_indices < 0) or np.any(frame_indices >= frame_count):
            errors.append("frame_indices contains a row outside the artifact frame domain")
        else:
            expected_counts = np.bincount(
                frame_indices.astype(np.int64, copy=False),
                minlength=frame_count,
            ).astype(np.int32, copy=False)
            if not np.array_equal(frame_counts, expected_counts):
                errors.append("frame_counts differs from exact frame_indices cardinality")
            if not np.array_equal(n_detections, expected_counts):
                errors.append("n_detections differs from exact frame_indices cardinality")

    shard_write = run_group.attrs.get("detect_shard_write")
    if run_group.attrs.get("detect_row_shard_rows") is None:
        if shard_write is not None:
            errors.append("regular artifact storage must not carry a shard-write claim")
    elif (
        not isinstance(shard_write, Mapping)
        or shard_write.get("status") != "complete"
        or shard_write.get("exact_match") is not True
        or shard_write.get("detection_row_count") != row_count
        or shard_write.get("frame_row_count") != frame_count
        or shard_write.get("source_sha256_by_array")
        != shard_write.get("destination_sha256_by_array")
        or set(shard_write.get("destination_sha256_by_array") or {}) != required
    ):
        errors.append("artifact shard-write proof is missing or inconsistent")

    if errors:
        raise RuntimeError(
            "Refusing to complete unbound detection artifact: " + "; ".join(errors)
        )
    report = {
        "schema_id": DETECTION_ARTIFACT_STORAGE_SCHEMA,
        "status": "ok",
        "row_identity": "artifact_run_local_dense_row_id",
        "row_identity_array": DETECTION_ARTIFACT_ROW_ID_ARRAY,
        "row_count": row_count,
        "frame_count": frame_count,
        "canonical_instance_key_present": False,
    }
    run_group.attrs[DETECTION_ARTIFACT_STORAGE_ATTR] = report
    return report


def _iter_opencv_rgb_batches(
    *,
    cap: Any,
    batch_size: int,
    pre_resize_dims: Optional[list[int] | tuple[int, int]],
    cv2_module: Any = cv2,
):
    """Yield RGB frame batches from an explicitly requested OpenCV decoder.

    OpenCV's reported CAP_PROP_FRAME_COUNT is not authoritative. Batches are
    flushed on the decoded stream's EOF, so an over-reported frame count cannot
    silently drop the final partial batch.
    """

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    frame_idx = 0
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    read_seconds = 0.0
    preprocess_seconds = 0.0

    while True:
        read_start = time.perf_counter()
        ret, frame = cap.read()
        read_seconds += time.perf_counter() - read_start

        if not ret:
            break

        preprocess_start = time.perf_counter()
        if pre_resize_dims:
            frame = cv2_module.resize(frame, (int(pre_resize_dims[1]), int(pre_resize_dims[0])))
        frame_rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        preprocess_seconds += time.perf_counter() - preprocess_start

        batch_frames.append(frame_rgb)
        batch_indices.append(frame_idx)
        frame_idx += 1

        if len(batch_frames) == batch_size:
            yield batch_indices, batch_frames, read_seconds, preprocess_seconds, frame_idx
            batch_frames = []
            batch_indices = []
            read_seconds = 0.0
            preprocess_seconds = 0.0

    if batch_frames:
        yield batch_indices, batch_frames, read_seconds, preprocess_seconds, frame_idx


def _read_and_preprocess_pynvvc_batch(
    *,
    frame_iter: Any,
    max_batch_frames: int,
    decode_backend_effective: str,
    source_height: int,
    device: torch.device,
    dtype: torch.dtype,
    resize_hw: list[int] | tuple[int, int],
) -> tuple[Optional[torch.Tensor], int, float, float]:
    """Read PyNvVC frames into an owned preallocated model-input batch.

    PyNvVideoCodec frames can be backed by decoder-owned reusable surfaces. Do
    not retain raw decoded tensors across further decode calls; preprocess each
    yielded frame into owned resized RGB tensor memory before advancing. CUDA
    events scope the wait to the materialization work that reads the decoder
    surface, rather than synchronizing the whole current stream unconditionally.
    """

    height, width = int(resize_hw[0]), int(resize_hw[1])
    max_batch_frames = max(0, int(max_batch_frames))
    if max_batch_frames <= 0:
        return None, 0, 0.0, 0.0

    processed_batch = torch.empty(
        (max_batch_frames, 3, height, width),
        device=device,
        dtype=dtype,
    ).contiguous(memory_format=torch.channels_last)
    read_seconds = 0.0
    preprocess_seconds = 0.0
    actual_count = 0
    for batch_index in range(max_batch_frames):
        read_start = time.perf_counter()
        try:
            frame_tensor = next(frame_iter)
        except StopIteration:
            break
        read_seconds += time.perf_counter() - read_start

        preprocess_start = time.perf_counter()
        if decode_backend_effective == BACKEND_PYNVVC_NV12_RGB:
            processed_frame = preprocess_nv12_rgb(
                [frame_tensor],
                source_height=source_height,
                device=device,
                dtype=dtype,
                resize_hw=resize_hw,
            )
        else:
            processed_frame = preprocess_luma_rgb(
                [frame_tensor],
                source_height=source_height,
                device=device,
                dtype=dtype,
                resize_hw=resize_hw,
            )
        processed_batch[batch_index : batch_index + 1].copy_(
            processed_frame,
            non_blocking=bool(getattr(processed_frame, "is_cuda", False)),
        )
        if getattr(processed_batch, "is_cuda", False):
            # PyNvVideoCodec may recycle the decoded surface on the next decode.
            # Wait only for the event marking completion of this frame's
            # materialization into owned tensor memory before advancing.
            stream = torch.cuda.current_stream(device=device)
            materialized = torch.cuda.Event(blocking=False)
            materialized.record(stream)
            materialized.synchronize()
        preprocess_seconds += time.perf_counter() - preprocess_start
        actual_count += 1
        del frame_tensor
        del processed_frame

    if actual_count <= 0:
        return None, 0, read_seconds, preprocess_seconds
    processed = processed_batch[:actual_count]
    if actual_count != max_batch_frames:
        processed = processed.contiguous(memory_format=torch.channels_last)
    return (
        processed,
        actual_count,
        read_seconds,
        preprocess_seconds,
    )


def _input_orig_shape(frame: Any, *, label: str) -> tuple[int, int]:
    shape = tuple(int(value) for value in getattr(frame, "shape", ()))
    if len(shape) < 2 or shape[-2] <= 0 or shape[-1] <= 0:
        raise ValueError(f"{label} has no valid image height/width shape: {shape!r}.")
    if isinstance(frame, torch.Tensor):
        return shape[-2], shape[-1]
    if len(shape) >= 3 and shape[-1] in {1, 3, 4}:
        return shape[-3], shape[-2]
    return shape[-2], shape[-1]


def _validate_model_batch_results(
    results: Any,
    *,
    expected_orig_shapes: Sequence[tuple[int, int]],
    backend: str,
) -> list[Any]:
    """Prove one model result exists for every exact model input image."""

    materialized = list(results)
    expected = [tuple(int(value) for value in shape) for shape in expected_orig_shapes]
    if len(materialized) != len(expected):
        raise ValueError(
            "YOLO result cardinality mismatch for "
            f"{backend}: results={len(materialized)}, inputs={len(expected)}."
        )
    for index, (result, expected_shape) in enumerate(zip(materialized, expected)):
        raw_shape = getattr(result, "orig_shape", None)
        try:
            observed_shape = tuple(int(value) for value in raw_shape)
        except (TypeError, ValueError):
            observed_shape = ()
        if len(observed_shape) != 2 or min(observed_shape, default=0) <= 0:
            raise ValueError(
                "YOLO result orig_shape is missing or invalid for "
                f"{backend} result {index}: {raw_shape!r}."
            )
        if observed_shape != expected_shape:
            raise ValueError(
                "YOLO result orig_shape mismatch for "
                f"{backend} result {index}: result={observed_shape}, "
                f"input={expected_shape}."
            )
    return materialized


def _validate_decoded_batch_cardinality(
    batch: Any,
    *,
    requested_indices: Sequence[int],
    backend: str,
) -> int:
    """Require a decoder batch row for every exact requested frame index."""

    shape = getattr(batch, "shape", None)
    try:
        observed = int(shape[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"{backend} returned a decoded batch without a valid leading dimension."
        ) from exc
    expected = len(requested_indices)
    if observed != expected:
        raise ValueError(
            f"{backend} decoded batch cardinality mismatch: "
            f"decoded={observed}, requested={expected}, "
            f"indices={list(requested_indices)!r}."
        )
    return observed


def _finalize_full_decode_domain_proof(
    *,
    backend: str,
    processed_frame_count: int,
    expected_frame_count: int,
    pynvvc_frame_iter: Any | None = None,
    decord_reader: Any | None = None,
    opencv_stream_exhausted: bool = False,
) -> str:
    """Prove the decoder covered exactly the declared acquisition frame domain."""

    processed = int(processed_frame_count)
    expected = int(expected_frame_count)
    if processed != expected:
        raise ValueError(
            "Detection did not decode the complete declared frame domain: "
            f"processed={processed}, expected={expected}, backend={backend}."
        )
    if backend in PYNVVC_BACKENDS:
        if pynvvc_frame_iter is None:
            raise ValueError("PyNvVideoCodec full-decode proof lacks its frame iterator.")
        try:
            trailing_frame = next(pynvvc_frame_iter)
        except StopIteration:
            return "pynvvc_exact_count_and_eof_probe_v1"
        del trailing_frame
        raise ValueError(
            "PyNvVideoCodec produced frames beyond the declared acquisition domain."
        )
    if backend in {DECODE_BACKEND_DECORD_GPU, DECODE_BACKEND_DECORD_CPU}:
        if decord_reader is None:
            raise ValueError("Decord full-decode proof lacks its reader.")
        observed_domain = int(len(decord_reader))
        if observed_domain != expected:
            raise ValueError(
                "Decord frame domain changed during inference: "
                f"observed={observed_domain}, expected={expected}."
            )
        return "decord_index_domain_and_exact_batches_v1"
    if backend == DECODE_BACKEND_OPENCV:
        if not opencv_stream_exhausted:
            raise ValueError("OpenCV full-decode proof did not observe stream EOF.")
        return "opencv_stream_eof_and_exact_count_v1"
    raise ValueError(f"Unsupported full-decode proof backend {backend!r}.")


@_guard_detection_publication_attempt
def detect_yolo(
    video_path: str,
    model_path: Optional[str] = None,
    output_zarr: str = None,
    config_path: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_det: Optional[int] = None,
    batch_size: Optional[int] = None,
    resize_dims: Optional[list[int] | tuple[int, int]] = None,
    imgsz: Optional[int | list[int] | tuple[int, int]] = None,
    decode_backend: Optional[str] = None,
    console: Optional[Console] = None,
    use_gpu: Optional[bool] = None,
    write_raw_video_metadata: bool = False,
    overwrite_raw_video_metadata: bool = False,
    run_name: Optional[str] = None,
    model_sha256: Optional[str] = None,
    detect_row_shard_rows: Optional[int] = DEFAULT_DETECT_ROW_SHARD_ROWS,
    detect_frame_shard_rows: int = DEFAULT_DETECT_FRAME_SHARD_ROWS,
    instance_key_recording_identity: Optional[str] = None,
    instance_key_frame_indices: Optional[np.ndarray] = None,
    instance_key_frame_mapping_source: Optional[str] = None,
    cli_provenance: Optional[Mapping[str, Any]] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
    coordinate_frame_evidence: Optional[BoundDetectionFrameEvidence] = None,
    coordinate_contract_mode: str = "canonical",
    output_run_family: str = DETECTION_RUN_FAMILY,
) -> str:
    """
    Run YOLO inference directly on video file, creating minimal zarr output.
    
    This is the INFERENCE pathway - for getting detections from a trained model.
    Does NOT import full video, only saves detection results.
    
    Args:
        video_path: Path to input video file
        model_path: Path to trained YOLO model (.pt) - optional if in config
        model_sha256: Optional expected SHA-256 for the model path from registry metadata
        output_zarr: Path for output zarr - optional, will auto-generate if None
        config_path: Path to YAML config file (optional)
        conf_threshold: Confidence threshold (overrides config)
        iou_threshold: IoU threshold for NMS (overrides config)
        max_det: Max detections per frame (overrides config)
        batch_size: Frames to process at once (overrides config)
        resize_dims: Canonical inference size [h, w]; mapped to YOLO imgsz
        imgsz: Legacy YOLO inference size alias; normalized into resize_dims
        decode_backend: Decoder backend (`auto`, `pynvvc_nv12_rgb`,
            `pynvvc_luma_rgb`, `decord_gpu`, `decord_cpu`, or `opencv`)
        console: Rich console
        use_gpu: Use GPU for inference (overrides config)
        write_raw_video_metadata: Create/update raw_video attrs (no frames) for registry/provenance
        overwrite_raw_video_metadata: Overwrite existing raw_video attrs when writing metadata
        run_name: Optional explicit detect run group name. Used by cluster planners
            when downstream jobs need deterministic paths.
        detect_row_shard_rows: Optional outer shard rows for detection-domain arrays.
        detect_frame_shard_rows: Outer shard rows for frame-domain count arrays when
            detection sharding is enabled.
        instance_key_recording_identity: Optional exact canonical acquisition
            recording identity assertion. Canonical runs derive identity from the
            sealed acquisition record and reject any supplied value that differs.
        instance_key_frame_indices: Optional canonical full-video identity mapping.
            Unbound artifacts reject every canonical instance-key input.
        instance_key_frame_mapping_source: Canonical mapping provenance label.
        cli_provenance: Optional Palette CLI provenance block stamped before completion.
        coordinate_frame_evidence: Deprecated injection guard. Canonical acquisition
            archives load and construct their own exact frame evidence; detached
            caller evidence cannot enable canonical publication.
        coordinate_contract_mode: ``canonical`` for ordinary persisted runs, or
            ``artifact_unbound`` for scratch transfer artifacts only.
        output_run_family: Exact output parent. Ordinary output is restricted to
            ``detect_runs``; unbound artifacts are restricted to the dedicated
            ``detection_artifact_runs`` non-selector family.
        
    Returns:
        Name of detect_runs group
    """
    if console is None:
        console = Console()
    if coordinate_contract_mode not in DETECTION_COORDINATE_CONTRACT_MODES:
        raise ValueError(
            "Unsupported coordinate_contract_mode "
            f"{coordinate_contract_mode!r}; expected one of "
            f"{DETECTION_COORDINATE_CONTRACT_MODES}."
        )

    if coordinate_contract_mode == "canonical":
        if output_run_family != DETECTION_RUN_FAMILY:
            raise ValueError(
                "Canonical detection must persist under detect_runs."
            )
        if coordinate_frame_evidence is not None:
            raise ValueError(
                "Canonical detection rejects detached coordinate_frame_evidence; "
                "the exact archive acquisition authority is always selected."
            )
    elif output_run_family != DETECTION_ARTIFACT_RUN_FAMILY:
        raise ValueError(
            "Unbound detection artifacts must persist under the dedicated "
            "detection_artifact_runs family."
        )
    if coordinate_contract_mode == "artifact_unbound" and (
        coordinate_frame_evidence is not None
    ):
        raise ValueError(
            "Unbound detection artifacts cannot accept coordinate frame evidence."
        )
    if coordinate_contract_mode == "artifact_unbound" and any(
        value is not None
        for value in (
            instance_key_recording_identity,
            instance_key_frame_indices,
            instance_key_frame_mapping_source,
        )
    ):
        raise ValueError(
            "Unbound detection artifacts cannot accept canonical instance-key "
            "identity or frame-mapping inputs; binding happens only during "
            "authoritative installation."
        )
    canonical_frame_evidence: BoundDetectionFrameEvidence | None = None
    canonical_acquisition_frame = None
    
    console.rule("[bold]YOLO Video Inference[/bold]")
    
    # Load config
    config = load_config(config_path)
    
    # Get parameters from config with CLI overrides
    model_path = model_path or config.get('model', {}).get('path')
    if model_path is None:
        raise ValueError("model_path required (via argument or config file)")
    
    # Auto-generate output path if not provided
    video_path = Path(video_path)
    if output_zarr is None:
        output_zarr = video_path.parent / f"{video_path.stem}_detections.zarr"
    
    # Get detection parameters with CLI overrides taking precedence
    detect_config = config.get('detection', {})
    conf_threshold = conf_threshold if conf_threshold is not None else detect_config.get('conf_threshold', 0.40)
    iou_threshold = iou_threshold if iou_threshold is not None else detect_config.get('iou_threshold', 0.45)
    max_det = max_det if max_det is not None else detect_config.get('max_det', 20)
    batch_size = batch_size if batch_size is not None else detect_config.get('batch_size', 32)

    cli_resize_dims = _normalize_resize_dims(resize_dims)
    cli_imgsz_legacy = _normalize_imgsz(imgsz)
    cli_imgsz_as_resize = _imgsz_to_resize_dims(cli_imgsz_legacy)
    if cli_resize_dims is not None and cli_imgsz_as_resize is not None and cli_resize_dims != cli_imgsz_as_resize:
        raise ValueError(
            f"Conflicting CLI overrides: resize_dims={cli_resize_dims} and imgsz={cli_imgsz_as_resize}. "
            "Set one, or make them equal."
        )

    config_resize_dims = _normalize_resize_dims(detect_config.get("resize_dims"))
    config_imgsz_legacy = _normalize_imgsz(detect_config.get("imgsz"))
    config_imgsz_as_resize = _imgsz_to_resize_dims(config_imgsz_legacy)
    if config_resize_dims is not None and config_imgsz_as_resize is not None and config_resize_dims != config_imgsz_as_resize:
        raise ValueError(
            "Conflicting config values: detection.resize_dims and detection.imgsz differ. "
            "Use detection.resize_dims as canonical, or keep them equal."
        )

    requested_resize_dims = (
        cli_resize_dims
        or cli_imgsz_as_resize
        or config_resize_dims
        or config_imgsz_as_resize
    )
    if cli_resize_dims is not None:
        resize_dims_source = "cli:resize_dims"
    elif cli_imgsz_as_resize is not None:
        resize_dims_source = "cli:imgsz"
    elif config_resize_dims is not None:
        resize_dims_source = "config:detection.resize_dims"
    elif config_imgsz_as_resize is not None:
        resize_dims_source = "config:detection.imgsz"
    else:
        resize_dims_source = "none"

    # Get video processing parameters
    video_config = config.get('video', {})
    decode_backend_requested = _normalize_decode_backend(
        decode_backend
        if decode_backend is not None
        else detect_config.get("decode_backend", video_config.get("decode_backend"))
    )
    legacy_video_resize_dims = _normalize_legacy_video_resize(video_config.get("resize"))
    legacy_video_resize_ignored = False
    if requested_resize_dims is not None:
        pre_resize_dims = None
        if legacy_video_resize_dims is not None:
            legacy_video_resize_ignored = True
    else:
        requested_resize_dims = legacy_video_resize_dims
        pre_resize_dims = legacy_video_resize_dims
        if legacy_video_resize_dims is not None:
            resize_dims_source = "config:video.resize"

    imgsz_applied = _resize_dims_to_imgsz(requested_resize_dims)

    source_zarr_config = (
        video_config.get('source_zarr')
        or video_config.get('source_zarr_path')
        or video_config.get('zarr_path')
        or video_config.get('palette_zarr')
    )
    source_zarr_path: Optional[Path] = None
    source_zarr_meta: Dict[str, Any] = {}
    if source_zarr_config:
        source_zarr_path = Path(source_zarr_config).expanduser()
        if source_zarr_path.exists():
            source_zarr_meta = _collect_zarr_metadata(source_zarr_path, console)
        else:
            console.print(
                f"[yellow]Warning:[/yellow] source Zarr path '{source_zarr_path}' not found; continuing without full-resolution metadata."
            )
    
    # GPU/device configuration
    device_config = config.get('model', {}).get('device', 'auto')
    if use_gpu is None:
        use_gpu = device_config != 'cpu'
    
    # Validate inputs
    model_path = Path(model_path).expanduser()
    output_zarr = Path(output_zarr)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model_artifact = fingerprint_artifact(
        model_path,
        role="detect_model",
        registry_hash=model_sha256,
    )
    console.print(f"Video: [cyan]{video_path}[/cyan]")
    console.print(f"Model: [cyan]{model_path}[/cyan]")
    console.print(f"Output: [cyan]{output_zarr}[/cyan]")
    
    # Print parameters
    console.print("\n[bold]Detection Parameters:[/bold]")
    console.print(f"  Confidence threshold: {conf_threshold}")
    console.print(f"  IoU threshold: {iou_threshold}")
    console.print(f"  Max detections: {max_det}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Decode backend requested: {decode_backend_requested}")
    if requested_resize_dims is not None:
        console.print(
            f"  Resize dims (canonical): {requested_resize_dims[0]}×{requested_resize_dims[1]} "
            f"[{resize_dims_source}]"
        )
    else:
        console.print("  Resize dims (canonical): None")
    if imgsz_applied is not None:
        console.print(f"  YOLO imgsz applied: {imgsz_applied}")
    if pre_resize_dims:
        console.print(
            f"  Legacy frame pre-resize (video.resize): {pre_resize_dims[1]}×{pre_resize_dims[0]}"
        )
    else:
        console.print("  Legacy frame pre-resize: None")
    if legacy_video_resize_ignored:
        console.print(
            "[yellow]  Note:[/yellow] Ignored legacy video.resize because canonical detection resize_dims/imgsz was set."
        )

    # Load model
    console.print("\n[bold]Loading model...[/bold]")
    model = YOLO(str(model_path))
    try:
        model.fuse()
    except AttributeError:
        pass  # Older versions may not expose fuse; Ultralytics will handle it.
    
    # Check device and move model
    model_fp16 = False
    
    if not use_gpu:
        model.to('cpu')
        console.print("[green]✓[/green] Model loaded on CPU")
    else:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            model.to('cuda')
            model.model = model.model.to(memory_format=torch.channels_last)
            model.half()
            model_fp16 = True
            console.print(f"[green]✓[/green] Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            console.print(f"[cyan]  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB[/cyan]")
        else:
            console.print("[yellow]⚠[/yellow]  CUDA not available, using CPU")
            use_gpu = False

    predict_kwargs: Dict[str, Any] = {
        "conf": conf_threshold,
        "iou": iou_threshold,
        "max_det": max_det,
        "verbose": False,
        "device": 'cuda' if use_gpu else 'cpu',
        "half": model_fp16,
    }
    if imgsz_applied is not None:
        predict_kwargs["imgsz"] = imgsz_applied
    
    # Open video to get metadata
    console.print("\n[bold]Opening video...[/bold]")
    
    decord_info: Optional[Dict[str, Any]] = None
    vr = None
    cap = None
    pynvvc_reader: Optional[PynvvcLumaRgbReader] = None
    use_pynvvc = False
    use_decord = False
    decode_backend_effective = DECODE_BACKEND_OPENCV

    def _open_pynvvc_reader(backend: str) -> PynvvcLumaRgbReader:
        if not use_gpu or not torch.cuda.is_available():
            raise RuntimeError(f"{backend} requires CUDA inference.")
        if requested_resize_dims is None:
            raise RuntimeError(
                f"{backend} requires detection.resize_dims or --resize-dims; "
                "it emits tensor inputs that must be explicitly resized before YOLO."
            )
        return PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)

    auto_pynvvc_candidate = decode_backend_requested == DECODE_BACKEND_AUTO
    if decode_backend_requested in PYNVVC_BACKENDS or auto_pynvvc_candidate:
        pynvvc_backend = (
            BACKEND_PYNVVC_LUMA_RGB
            if decode_backend_requested == DECODE_BACKEND_AUTO
            else decode_backend_requested
        )
        try:
            pynvvc_reader = _open_pynvvc_reader(pynvvc_backend)
            _active_detection_publication_attempt().track_pynvvc_reader(
                pynvvc_reader
            )
            n_frames, fps_cv2, width_cv2, height_cv2 = _read_cv2_video_properties(video_path)
            width = int(pynvvc_reader.source_width or width_cv2)
            height = int(pynvvc_reader.source_height or height_cv2)
            fps = float(pynvvc_reader.frame_rate or fps_cv2)
            if n_frames <= 0:
                raise RuntimeError("OpenCV metadata did not report a positive frame count.")
            video_reader_type = pynvvc_backend
            decode_backend_effective = pynvvc_backend
            use_pynvvc = True
            console.print(f"[green]✓[/green] Using PyNvVideoCodec {pynvvc_backend} CUDA decoder")
        except Exception as exc:
            if decode_backend_requested != DECODE_BACKEND_AUTO:
                raise
            raise _gpu_decode_unavailable(
                f"auto requires PyNvVideoCodec {pynvvc_backend}; initialization failed: {exc}"
            ) from exc

    if not use_pynvvc and decode_backend_requested != DECODE_BACKEND_OPENCV:
        if decode_backend_requested == DECODE_BACKEND_AUTO:
            raise _gpu_decode_unavailable("auto did not initialize PyNvVideoCodec luma decode")
        prefer_decord_gpu = decode_backend_requested == DECODE_BACKEND_DECORD_GPU
        decord_info = _init_decord_reader(video_path, prefer_gpu=prefer_decord_gpu, console=console)
        if (
            decode_backend_requested == DECODE_BACKEND_DECORD_GPU
            and decord_info is not None
            and not bool(decord_info.get("on_gpu"))
        ):
            raise RuntimeError("Requested decord_gpu, but Decord fell back to CPU.")
        if decord_info is None and decode_backend_requested in {
            DECODE_BACKEND_DECORD_GPU,
            DECODE_BACKEND_DECORD_CPU,
        }:
            raise RuntimeError(f"Requested {decode_backend_requested}, but Decord is unavailable.")

    if not use_pynvvc and decord_info:
        vr = decord_info['reader']
        _active_detection_publication_attempt().track_decord_reader(vr)
        n_frames = len(vr)
        width = decord_info['width']
        height = decord_info['height']
        fps = decord_info['fps']
        video_reader_type = decord_info['type']
        decode_backend_effective = str(decord_info['type'])
        use_decord = True
    elif not use_pynvvc:
        cap = cv2.VideoCapture(str(video_path))
        _active_detection_publication_attempt().track_cap(cap)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        console.print("[green]✓[/green] Using OpenCV decoder")
        video_reader_type = 'opencv'
        decode_backend_effective = DECODE_BACKEND_OPENCV

    decord_on_gpu = bool(decord_info and decord_info.get('on_gpu'))
    tensor_input_path = bool(decord_on_gpu or use_pynvvc)
    effective_input_resize_dims = _actual_input_resize_dims(
        requested_resize_dims,
        pre_resize_dims,
        decord_on_gpu=decord_on_gpu,
        tensor_input=use_pynvvc,
    )
    
    # Determine dimensions for normalization (actual or resized)
    if effective_input_resize_dims:
        inference_height, inference_width = effective_input_resize_dims
        console.print(f"[green]✓[/green] Video: {n_frames} frames, {fps:.1f} fps, {width}×{height}")
        console.print(f"[cyan]  Will feed {inference_width}×{inference_height} frames to YOLO[/cyan]")
        if tensor_input_path and requested_resize_dims is not None:
            console.print("[cyan]  Tensor decoder path: applying canonical resize before predict[/cyan]")
    else:
        inference_width, inference_height = width, height
        console.print(f"[green]✓[/green] Video: {n_frames} frames, {fps:.1f} fps, {width}×{height}")
    
    console.print("\n[bold]Preparing output Zarr...[/bold]")
    if output_zarr.exists():
        root = zarr.open_group(str(output_zarr), mode='r+', use_consolidated=False)
        created_new_root = False
        console.print(f"[cyan]Appending detect run to existing archive:[/cyan] {output_zarr}")
    else:
        root = zarr.open_group(str(output_zarr), mode='w')
        created_new_root = True
        console.print(f"[cyan]Created new detection archive:[/cyan] {output_zarr}")

    if coordinate_contract_mode == "canonical":
        canonical_acquisition_frame = _load_full_acquisition_video_source(
            root,
            video_path=video_path,
            output_zarr=output_zarr,
            decoded_frame_count=int(n_frames),
            decoded_width=int(width),
            decoded_height=int(height),
        )
        sealed_recording_identity = str(
            canonical_acquisition_frame.record.recording_id
        ).strip()
        if (
            instance_key_recording_identity is not None
            and (
                type(instance_key_recording_identity) is not str
                or instance_key_recording_identity != sealed_recording_identity
            )
        ):
            raise ValueError(
                "Canonical detection instance_key_recording_identity must equal "
                "the sealed acquisition recording_id: "
                f"provided={instance_key_recording_identity!r}, "
                f"sealed={sealed_recording_identity!r}."
            )
        identity_mapping = np.arange(int(n_frames), dtype=np.int64)
        if instance_key_frame_indices is not None and not np.array_equal(
            np.asarray(instance_key_frame_indices),
            identity_mapping,
        ):
            raise ValueError(
                "Canonical full-video detection rejects caller-provided frame "
                "permutations; only the mechanically proven acquisition identity "
                "mapping is supported."
            )
        instance_key_frame_indices = identity_mapping
        instance_key_frame_mapping_source = (
            f"{canonical_acquisition_frame.record_ref}#"
            "full_untrimmed_video_decode_identity_v1"
        )
    else:
        # Artifact rows intentionally carry no camera-frame declaration.  They
        # are transfer inputs, not selectable observations.
        canonical_acquisition_frame = None

    source_video_width = int(width)
    source_video_height = int(height)
    source_full_width = int(source_zarr_meta.get("full_width", source_video_width))
    source_full_height = int(source_zarr_meta.get("full_height", source_video_height))

    # Get comprehensive video metadata
    vid_meta = get_video_metadata(video_path, cap, width, height, n_frames, fps)
    
    # Get git info for reproducibility (matching import_video.py)
    git_info = get_git_info()
    
    # Get full environment info for provenance
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(output_zarr),
        collect_ip=False
    )
    
    # Always keep core video metadata on the root so downstream consumers (and diagnostics) can read it.
    root.attrs.update({
        'source_video': vid_meta['source_video'],
        'source_video_path': vid_meta['source_path'],
        'source_path': vid_meta['source_path'],
        'video_width': int(width),
        'video_height': int(height),
        'width': int(width),
        'height': int(height),
        'fps': float(fps) if fps and fps > 0 else fps,
        'n_frames': int(n_frames),
        'total_frames': int(n_frames),
        'duration_seconds': float(vid_meta['duration_seconds']),
        'video_codec': vid_meta.get('codec', root.attrs.get('video_codec', 'unknown')),
        'video_pix_fmt': vid_meta.get('pix_fmt', root.attrs.get('video_pix_fmt', 'unknown')),
    })

    if created_new_root:
        root.attrs.update({
            # Video source info
            'source_video': vid_meta['source_video'],
            'source_video_path': vid_meta['source_path'],
            'source_path': vid_meta['source_path'],  # Alias for compatibility
            
            # Video properties
            'video_width': width,
            'video_height': height,
            'width': width,  # Alias
            'height': height,  # Alias
            'fps': fps,
            'n_frames': n_frames,
            'total_frames': n_frames,  # Alias
            'duration_seconds': vid_meta['duration_seconds'],
            
            # Codec info
            'video_codec': vid_meta.get('codec', 'unknown'),
            'video_pix_fmt': vid_meta.get('pix_fmt', 'unknown'),
            
            # Pipeline info
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'pipeline_type': 'yolo_inference',
            'zarr_purpose': 'production',
            'has_raw_video': False,
            'detection_method': 'yolo',
            
            # Model info
            'model_path': str(model_path.absolute()),
            'model_name': model_path.name,
            
            # Processing info
            'inference_width': inference_width,
            'inference_height': inference_height,
            'resized_for_inference': effective_input_resize_dims is not None,
            
            # Git provenance
            'git_commit_hash': git_info.get('commit_hash', 'unknown'),
            'git_short_hash': git_info.get('short_hash', 'unknown'),
            'git_branch': git_info.get('branch', 'unknown'),
            'git_is_dirty': git_info.get('is_dirty', False),
            'git_remote_url': git_info.get('remote_url', 'unknown'),
            
            # System provenance
            'system_hostname': env_info['platform']['hostname'],
            'system_fqdn': env_info['platform']['fqdn'],
            'system_os': env_info['platform']['system'],
            'system_os_release': env_info['platform']['release'],
            'system_machine': env_info['platform']['machine'],
            'system_python_version': env_info['platform']['python_version'],
            'system_username': env_info['platform']['username'],
            'system_cpu_cores': env_info['platform']['cpu_cores'],
        })

        if 'cpu_details' in env_info['platform']:
            cpu = env_info['platform']['cpu_details']
            root.attrs.update({
                'cpu_model': cpu.get('model', 'unknown'),
                'cpu_arch': cpu.get('arch', 'unknown'),
            })

        if 'memory' in env_info['platform']:
            mem = env_info['platform']['memory']
            root.attrs.update({
                'memory_total_gb': mem.get('total_gb', 0),
                'memory_available_gb': mem.get('available_gb', 0),
                'memory_percent_used': mem.get('percent_used', 0),
            })

        if 'disk' in env_info['platform']:
            disk = env_info['platform']['disk']
            root.attrs.update({
                'disk_path': disk.get('path', str(output_zarr)),
                'disk_total_gb': disk.get('total_gb', 0),
                'disk_available_gb': disk.get('available_gb', 0),
                'disk_percent_used': disk.get('percent_used', 0),
            })

        if 'lsf' in env_info['platform']:
            lsf = env_info['platform']['lsf']
            root.attrs.update({
                'hpc_scheduler': 'LSF',
                'lsf_job_id': lsf.get('job_id', 'unknown'),
                'lsf_job_name': lsf.get('job_name', 'unknown'),
                'lsf_queue': lsf.get('queue', 'unknown'),
                'lsf_hosts': lsf.get('hosts', 'unknown'),
            })
        elif 'slurm' in env_info['platform']:
            slurm = env_info['platform']['slurm']
            root.attrs.update({
                'hpc_scheduler': 'SLURM',
                'slurm_job_id': slurm.get('job_id', 'unknown'),
                'slurm_job_name': slurm.get('job_name', 'unknown'),
                'slurm_node_list': slurm.get('node_list', 'unknown'),
            })

        if env_info.get('gpu', {}).get('available'):
            gpu_info = env_info['gpu']
            root.attrs.update({
                'gpu_available': True,
                'gpu_backend': gpu_info.get('backend', 'unknown'),
                'gpu_count': len(gpu_info.get('devices', [])),
            })
            if 'cuda_version' in gpu_info:
                root.attrs['cuda_version'] = gpu_info['cuda_version']
            if gpu_info.get('devices'):
                primary_gpu = gpu_info['devices'][0]
                root.attrs.update({
                    'gpu_name': primary_gpu.get('name', 'unknown'),
                    'gpu_compute_capability': primary_gpu.get('compute_capability', 'unknown'),
                    'gpu_memory_total_gb': primary_gpu.get('total_memory_gb', 0),
                })

        env_summary = env_info.get('environment', {})
        if env_summary:
            root.attrs.update({
                'environment_type': env_summary.get('environment_type', 'unknown'),
                'environment_name': env_summary.get('environment_name', 'unknown'),
                'total_packages': env_summary.get('total_packages', 0),
            })
            if 'deep_learning_framework' in env_summary:
                root.attrs['deep_learning_framework'] = env_summary['deep_learning_framework']
            if 'key_packages' in env_summary:
                import json
                root.attrs['key_packages_json'] = json.dumps(env_summary['key_packages'])

        import json
        root.attrs['_full_environment_info'] = json.dumps(env_info, default=str)

    if source_zarr_meta:
        palette_attr_map = {
            "fps": "palette_fps",
            "total_frames": "palette_total_frames",
            "duration_seconds": "palette_duration_seconds",
            "video_codec": "palette_video_codec",
            "video_pix_fmt": "palette_video_pix_fmt",
            "downsampled_resolution": "palette_downsampled_resolution",
            "downsample_method": "palette_downsample_method",
            "has_full_resolution": "palette_has_full_resolution",
            "has_downsampled": "palette_has_downsampled",
            "original_resolution": "palette_original_resolution",
        }
        for key, attr_name in palette_attr_map.items():
            value = source_zarr_meta.get(key)
            if value is None:
                continue
            if isinstance(value, tuple):
                value = list(value)
            root.attrs[attr_name] = value
        if source_zarr_meta.get("full_width") is not None:
            root.attrs["palette_video_width"] = int(source_zarr_meta["full_width"])
        if source_zarr_meta.get("full_height") is not None:
            root.attrs["palette_video_height"] = int(source_zarr_meta["full_height"])

    root.attrs['source_video_width'] = source_video_width
    root.attrs['source_video_height'] = source_video_height
    root.attrs['source_video_resolution'] = [source_video_width, source_video_height]
    root.attrs['source_full_width'] = source_full_width
    root.attrs['source_full_height'] = source_full_height
    root.attrs['source_full_resolution'] = [source_full_width, source_full_height]
    root.attrs['inference_resolution'] = [int(inference_width), int(inference_height)]
    root.attrs['inference_width'] = int(inference_width)
    root.attrs['inference_height'] = int(inference_height)
    root.attrs['resized_for_inference'] = effective_input_resize_dims is not None
    root.attrs['resize_dims_requested'] = (
        [int(requested_resize_dims[0]), int(requested_resize_dims[1])]
        if requested_resize_dims is not None
        else None
    )
    root.attrs['resize_dims_source'] = resize_dims_source
    root.attrs['imgsz_applied'] = imgsz_applied
    root.attrs['imgsz_legacy_input'] = cli_imgsz_legacy
    root.attrs['pre_resize_dims'] = (
        [int(pre_resize_dims[0]), int(pre_resize_dims[1])]
        if pre_resize_dims is not None
        else None
    )
    root.attrs['effective_input_resize_dims'] = (
        [int(effective_input_resize_dims[0]), int(effective_input_resize_dims[1])]
        if effective_input_resize_dims is not None
        else None
    )
    root.attrs['tensor_resize_dims'] = (
        [int(effective_input_resize_dims[0]), int(effective_input_resize_dims[1])]
        if tensor_input_path and effective_input_resize_dims is not None
        else None
    )
    root.attrs['decode_backend_requested'] = decode_backend_requested
    root.attrs['decode_backend_effective'] = decode_backend_effective
    root.attrs['video_reader_type'] = video_reader_type
    if source_zarr_path is not None:
        root.attrs['source_zarr_path'] = str(source_zarr_path)

    if write_raw_video_metadata:
        updates = write_video_metadata(
            root,
            vid_meta,
            overwrite=overwrite_raw_video_metadata,
            import_purpose="production",
        )
        raw_updates = updates.get("raw_video", {})
        root_updates = updates.get("root", {})
        if raw_updates or root_updates:
            console.print(
                "[green]✓[/green] Wrote metadata-only attrs "
                f"(root={len(root_updates)}, raw_video={len(raw_updates)})"
            )

    if canonical_acquisition_frame is not None:
        observed_extent = (int(source_video_width), int(source_video_height))
        declared_full_extent = (int(source_full_width), int(source_full_height))
        canonical_extent = (
            int(canonical_acquisition_frame.record.width_px),
            int(canonical_acquisition_frame.record.height_px),
        )
        if observed_extent != canonical_extent or declared_full_extent != canonical_extent:
            raise ValueError(
                "Canonical detection publication requires the decoded video and "
                "declared full-frame extent to equal the exact acquisition-owned "
                f"source-camera frame: observed={observed_extent}, "
                f"declared_full={declared_full_extent}, canonical={canonical_extent}."
            )
    
    parent_group = require_runs_parent(
        root,
        output_run_family,
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    selector_snapshot = (
        _snapshot_parent_selectors(parent_group)
        if coordinate_contract_mode == "canonical"
        else None
    )
    if coordinate_contract_mode == "artifact_unbound":
        parent_group.attrs["artifact_family_contract"] = (
            "palette.detection_artifact_family.v1"
        )
        parent_group.attrs["stage_selector_eligible"] = False
    if run_name is not None:
        run_name = str(run_name).strip()
        if not run_name or "/" in run_name or run_name in {".", ".."}:
            raise ValueError(f"Invalid detect run name: {run_name!r}")
        if run_name in parent_group:
            raise ValueError(f"{output_run_family}/{run_name} already exists")
    else:
        run_name = _next_run_name(
            parent_group,
            prefix=(
                "detect"
                if coordinate_contract_mode == "canonical"
                else "detection_artifact"
            ),
        )
    publication_attempt = _active_detection_publication_attempt()
    publication_attempt.prepare(
        root=root,
        parent_group=parent_group,
        run_name=run_name,
        selector_snapshot=selector_snapshot,
    )
    if selector_snapshot is not None:
        collisions = _selector_collisions(selector_snapshot, run_name)
        if collisions:
            raise ValueError(
                "Refusing to create a detection run whose absent name is already "
                f"referenced by parent selectors: run={run_name!r}, "
                f"selectors={list(collisions)!r}."
            )
    publication_attempt.claim_named_child()
    detect_group = parent_group.create_group(run_name)
    publication_attempt.bind_run(detect_group)
    # A normal reader must never observe a publishing candidate. Canonical
    # eligibility is flipped only after complete, fresh persisted validation and
    # parent-pointer preparation; artifact runs remain permanently ineligible.
    detect_group.attrs["stage_selector_eligible"] = False
    mark_run_started(
        detect_group,
        run_name=run_name,
        stage=(
            "detect"
            if coordinate_contract_mode == "canonical"
            else "detection_artifact"
        ),
    )
    if coordinate_contract_mode == "canonical":
        note_pending_latest(parent_group, run_name)
    console.print(
        f"Created run group: [cyan]{output_run_family}/{run_name}[/cyan]"
    )
    coordinate_checkpoints = publication_attempt.coordinate_checkpoints
    console.print(
        f"[green]✓[/green] Writing detections to {output_run_family}/{run_name}"
    )
    
    # Storage for detections
    frame_counts = np.zeros(n_frames, dtype=np.int32)
    batch_results = []
    validated_backend_result_count = 0
    validated_backend_result_orig_shapes: set[tuple[int, int]] = set()

    def accumulate_results(
        result: Any,
        global_frame_idx: int,
        *,
        orig_shape: tuple[int, int],
    ) -> None:
        """Vectorize detection accumulation for a single frame."""
        nonlocal validated_backend_result_count
        validated_backend_result_count += 1
        validated_backend_result_orig_shapes.add(
            (int(orig_shape[0]), int(orig_shape[1]))
        )
        if result.boxes is None or len(result.boxes) == 0:
            return

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores_np = result.boxes.conf.detach().cpu().numpy()
        num_detections = boxes_xyxy.shape[0]
        if num_detections == 0:
            return

        result_height, result_width = orig_shape
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5 / result_width
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5 / result_height
        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / result_width
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / result_height

        bbox_norm = np.column_stack((cx, cy, w, h)).astype(np.float64, copy=False)
        indices = np.full(num_detections, global_frame_idx, dtype=np.int32)
        scores = scores_np.astype(np.float32, copy=False)

        cls_tensor = result.boxes.cls
        if cls_tensor is None:
            class_ids = np.zeros(num_detections, dtype=np.int32)
        else:
            class_ids = cls_tensor.detach().cpu().numpy().astype(np.int32, copy=False)

        batch_results.append((indices, bbox_norm, scores, class_ids))
        frame_counts[global_frame_idx] += num_detections
    
    # Process video in batches
    frame_idx = 0
    batch_frames = []
    batch_indices = []
    
    console.print("\n[bold]Running inference...[/bold]")
    console.print(f"[cyan]Decoder: {video_reader_type}[/cyan]")
    
    # Performance tracking
    inference_times = []
    read_times = []
    stage_timings = {
        'read_decode_seconds_total': 0.0,
        'preprocess_resize_seconds_total': 0.0,
        'predict_seconds_total': 0.0,
        'postprocess_seconds_total': 0.0,
        'array_assembly_seconds_total': 0.0,
        'zarr_write_seconds_total': 0.0,
    }
    processing_start = time.time()
    pynvvc_frame_iter = None
    opencv_stream_exhausted = False
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[fps]:.1f} fps"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing frames", total=n_frames, fps=0.0)
        
        batch_count = 0
        
        if use_pynvvc:
            if pynvvc_reader is None:
                raise RuntimeError("PyNvVideoCodec reader was not initialized.")
            if effective_input_resize_dims is None:
                raise RuntimeError(f"{decode_backend_effective} requires a resolved tensor resize.")
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            dtype = torch.float16 if model_fp16 else torch.float32
            pynvvc_frame_iter = pynvvc_reader.iter_frames()

            while frame_idx < n_frames:
                max_batch_frames = min(int(batch_size), int(n_frames - frame_idx))
                processed, actual_count, read_elapsed, preprocess_elapsed = (
                    _read_and_preprocess_pynvvc_batch(
                        frame_iter=pynvvc_frame_iter,
                        max_batch_frames=max_batch_frames,
                        decode_backend_effective=decode_backend_effective,
                        source_height=pynvvc_reader.source_height,
                        device=device,
                        dtype=dtype,
                        resize_hw=effective_input_resize_dims,
                    )
                )
                stage_timings['read_decode_seconds_total'] += float(read_elapsed)
                stage_timings['preprocess_resize_seconds_total'] += float(preprocess_elapsed)
                read_times.append(read_elapsed)

                if actual_count <= 0 or processed is None:
                    break
                actual_indices = list(range(frame_idx, frame_idx + actual_count))

                inference_start = time.perf_counter()
                with torch.inference_mode():
                    results = model.predict(processed, **predict_kwargs)
                expected_orig_shape = _input_orig_shape(
                    processed[0],
                    label=f"{decode_backend_effective} model input",
                )
                results = _validate_model_batch_results(
                    results,
                    expected_orig_shapes=[expected_orig_shape] * actual_count,
                    backend=decode_backend_effective,
                )
                inference_elapsed = _record_timing(
                    stage_timings, 'predict_seconds_total', inference_start
                )
                inference_times.append(inference_elapsed)

                postprocess_start = time.perf_counter()
                for batch_i, result in enumerate(results):
                    accumulate_results(
                        result,
                        actual_indices[batch_i],
                        orig_shape=expected_orig_shape,
                    )
                _record_timing(
                    stage_timings, 'postprocess_seconds_total', postprocess_start
                )

                del results
                del processed

                frame_idx += actual_count
                batch_count += 1

                elapsed = time.time() - processing_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress.update(task, advance=actual_count, fps=current_fps)

                if batch_count % 100 == 0:
                    avg_inference = np.mean(inference_times[-100:])
                    avg_read = np.mean(read_times[-100:])
                    console.print(f"[dim]Batch {batch_count}: read={avg_read*1000:.1f}ms, "
                                f"inference={avg_inference*1000:.1f}ms, fps={current_fps:.1f}[/dim]")

                if actual_count < max_batch_frames:
                    break

        elif use_decord:
            # Double-buffered decode: prefetch next batch while current inference runs.
            batch_starts = list(range(0, n_frames, batch_size))
            prefetched = None
            
            for idx, batch_start in enumerate(batch_starts):
                batch_end = min(batch_start + batch_size, n_frames)
                indices = list(range(batch_start, batch_end))
                
                if prefetched is None:
                    read_start = time.perf_counter()
                    current_batch = vr.get_batch(indices)
                    read_elapsed = _record_timing(
                        stage_timings, 'read_decode_seconds_total', read_start
                    )
                    read_times.append(read_elapsed)
                else:
                    current_batch = prefetched

                _validate_decoded_batch_cardinality(
                    current_batch,
                    requested_indices=indices,
                    backend=decode_backend_effective,
                )
                
                next_indices = None
                prefetched = None
                if idx + 1 < len(batch_starts):
                    next_start = batch_starts[idx + 1]
                    next_end = min(next_start + batch_size, n_frames)
                    next_indices = list(range(next_start, next_end))
                    if not decord_on_gpu or torch.cuda.is_available():
                        prefetch_start = time.perf_counter()
                        prefetched = vr.get_batch(next_indices)
                        prefetch_elapsed = _record_timing(
                            stage_timings, 'read_decode_seconds_total', prefetch_start
                        )
                        read_times.append(prefetch_elapsed)
                
                if decord_on_gpu:
                    import torch.nn.functional as F
                    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    dtype = torch.float16 if model_fp16 else torch.float32
                    frames_chw = current_batch.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W] uint8
                    
                    total = frames_chw.shape[0]
                    chunk_size = total
                    results = []
                    result_orig_shapes: list[tuple[int, int]] = []
                    start = 0
                    
                    while start < total:
                        end = min(start + chunk_size, total)
                        chunk = frames_chw[start:end]
                        try:
                            preprocess_start = time.perf_counter()
                            chunk = chunk.to(device=device, dtype=dtype, non_blocking=True).contiguous(memory_format=torch.channels_last)
                            if effective_input_resize_dims:
                                chunk = F.interpolate(
                                    chunk,
                                    size=effective_input_resize_dims,
                                    mode='bilinear',
                                    align_corners=False
                                )
                            chunk = chunk.mul_(1.0 / 255.0)
                            _record_timing(
                                stage_timings,
                                'preprocess_resize_seconds_total',
                                preprocess_start,
                            )
                            
                            inference_start = time.perf_counter()
                            preds = model.predict(chunk, **predict_kwargs)
                            expected_orig_shape = _input_orig_shape(
                                chunk[0],
                                label=f"{decode_backend_effective} model input",
                            )
                            preds = _validate_model_batch_results(
                                preds,
                                expected_orig_shapes=[expected_orig_shape]
                                * (end - start),
                                backend=decode_backend_effective,
                            )
                            inference_elapsed = _record_timing(
                                stage_timings, 'predict_seconds_total', inference_start
                            )
                            inference_times.append(inference_elapsed)
                            results.extend(preds)
                            result_orig_shapes.extend(
                                [expected_orig_shape] * len(preds)
                            )
                            start = end
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if chunk_size == 1:
                                raise
                            chunk_size = max(1, chunk_size // 2)
                            continue
                        finally:
                            del chunk
                    
                    del frames_chw
                else:
                    frames_nd = current_batch.asnumpy() if hasattr(current_batch, "asnumpy") else np.asarray(current_batch)
                    preprocess_start = time.perf_counter()
                    if pre_resize_dims:
                        batch_frames_np = [
                            cv2.resize(frame, (int(pre_resize_dims[1]), int(pre_resize_dims[0]))) for frame in frames_nd
                        ]
                    else:
                        batch_frames_np = [np.asarray(frame) for frame in frames_nd]
                    _record_timing(
                        stage_timings, 'preprocess_resize_seconds_total', preprocess_start
                    )
                    del frames_nd
                
                    inference_start = time.perf_counter()
                    results = model.predict(batch_frames_np, **predict_kwargs)
                    result_orig_shapes = [
                        _input_orig_shape(
                            frame,
                            label=f"{decode_backend_effective} model input",
                        )
                        for frame in batch_frames_np
                    ]
                    results = _validate_model_batch_results(
                        results,
                        expected_orig_shapes=result_orig_shapes,
                        backend=decode_backend_effective,
                    )
                    inference_elapsed = _record_timing(
                        stage_timings, 'predict_seconds_total', inference_start
                    )
                    inference_times.append(inference_elapsed)
                
                postprocess_start = time.perf_counter()
                for batch_i, result in enumerate(results):
                    accumulate_results(
                        result,
                        indices[batch_i],
                        orig_shape=result_orig_shapes[batch_i],
                    )
                _record_timing(
                    stage_timings, 'postprocess_seconds_total', postprocess_start
                )
                
                del current_batch
                if not decord_on_gpu:
                    del batch_frames_np
                
                frame_idx += len(indices)
                batch_count += 1
                
                elapsed = time.time() - processing_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress.update(task, advance=len(indices), fps=current_fps)
                
                if batch_count % 100 == 0:
                    avg_inference = np.mean(inference_times[-100:])
                    avg_read = np.mean(read_times[-100:])
                    console.print(f"[dim]Batch {batch_count}: read={avg_read*1000:.1f}ms, "
                                f"inference={avg_inference*1000:.1f}ms, fps={current_fps:.1f}[/dim]")
            
            if prefetched is not None:
                del prefetched
        
        else:
            # OpenCV frame-by-frame processing
            for (
                batch_indices,
                batch_frames,
                read_elapsed,
                preprocess_elapsed,
                frame_idx,
            ) in _iter_opencv_rgb_batches(
                cap=cap,
                batch_size=batch_size,
                pre_resize_dims=pre_resize_dims,
            ):
                stage_timings['read_decode_seconds_total'] += float(read_elapsed)
                stage_timings['preprocess_resize_seconds_total'] += float(
                    preprocess_elapsed
                )
                read_times.append(read_elapsed)

                # Time inference
                inference_start = time.perf_counter()

                # Run inference
                results = model.predict(batch_frames, **predict_kwargs)
                result_orig_shapes = [
                    _input_orig_shape(
                        frame,
                        label=f"{decode_backend_effective} model input",
                    )
                    for frame in batch_frames
                ]
                results = _validate_model_batch_results(
                    results,
                    expected_orig_shapes=result_orig_shapes,
                    backend=decode_backend_effective,
                )

                inference_time = _record_timing(
                    stage_timings, 'predict_seconds_total', inference_start
                )
                inference_times.append(inference_time)

                # Calculate FPS
                elapsed = time.time() - processing_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0

                # Extract detections
                postprocess_start = time.perf_counter()
                for batch_i, result in enumerate(results):
                    accumulate_results(
                        result,
                        batch_indices[batch_i],
                        orig_shape=result_orig_shapes[batch_i],
                    )
                _record_timing(
                    stage_timings, 'postprocess_seconds_total', postprocess_start
                )

                batch_size_actual = len(batch_frames)
                batch_count += 1

                # Update progress
                progress.update(task, advance=batch_size_actual, fps=current_fps)

                # Print diagnostics every 100 batches
                if batch_count % 100 == 0:
                    avg_inference = np.mean(inference_times[-100:]) if len(inference_times) > 0 else 0
                    avg_read = np.mean(read_times[-100:]) if len(read_times) > 0 else 0
                    console.print(f"[dim]Batch {batch_count}: inference={avg_inference*1000:.1f}ms, "
                                f"read={avg_read*1000:.1f}ms, fps={current_fps:.1f}[/dim]")
            opencv_stream_exhausted = True
            
    decode_domain_proof = _finalize_full_decode_domain_proof(
        backend=decode_backend_effective,
        processed_frame_count=frame_idx,
        expected_frame_count=n_frames,
        pynvvc_frame_iter=pynvvc_frame_iter,
        decord_reader=vr,
        opencv_stream_exhausted=opencv_stream_exhausted,
    )
    expected_backend_result_shape = (int(inference_height), int(inference_width))
    if (
        validated_backend_result_count != int(n_frames)
        or validated_backend_result_orig_shapes != {expected_backend_result_shape}
    ):
        raise RuntimeError(
            "Validated YOLO result count/orig_shape evidence is incomplete or "
            "non-uniform: "
            f"count={validated_backend_result_count}, expected={n_frames}, "
            f"shapes={sorted(validated_backend_result_orig_shapes)!r}, "
            f"expected_shape={expected_backend_result_shape!r}."
        )

    # Decoder resources remain owned by the encompassing publication attempt
    # and are closed exactly once by its finalization boundary.
    
    total_time = time.time() - processing_start
    console.print("[green]✓[/green] Inference complete")
    console.print(f"[cyan]  Total time: {total_time:.1f}s ({total_time/60:.1f} min)[/cyan]")
    console.print(f"[cyan]  Average FPS: {n_frames/total_time:.1f}[/cyan]")
    
    if len(inference_times) > 0:
        avg_inference = np.mean(inference_times)
        console.print(f"[cyan]  Avg inference time per batch: {avg_inference*1000:.1f}ms[/cyan]")
        console.print(f"[cyan]  Avg read time per batch: {np.mean(read_times)*1000:.1f}ms[/cyan]")
    else:
        avg_inference = 0.0
    
    # Convert to arrays
    console.print("\n[bold]Saving detections to zarr...[/bold]")
    assembly_start = time.perf_counter()
    if batch_results:
        frame_indices = np.concatenate([res[0] for res in batch_results])
        bbox_coords = np.concatenate([res[1] for res in batch_results])
        scores = np.concatenate([res[2] for res in batch_results])
        class_ids = np.concatenate([res[3] for res in batch_results]).astype(np.int32, copy=False)
    else:
        frame_indices = np.empty((0,), dtype=np.int32)
        bbox_coords = np.empty((0, 4), dtype=np.float64)
        scores = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=np.int32)
    
    total_detections = frame_indices.size
    
    if total_detections > 0:
        frame_counts = np.bincount(frame_indices, minlength=n_frames).astype(np.int32, copy=False)
    else:
        frame_counts = np.zeros(n_frames, dtype=np.int32)
    key_frame_domain: str | None = None
    key_frame_mapping_sha256: str | None = None
    identity_array_name = DETECTION_ARTIFACT_ROW_ID_ARRAY
    if canonical_acquisition_frame is not None:
        recording_identity = str(
            canonical_acquisition_frame.record.recording_id
        ).strip()
        if (
            instance_key_recording_identity is not None
            and (
                type(instance_key_recording_identity) is not str
                or instance_key_recording_identity != recording_identity
            )
        ):
            raise ValueError(
                "Canonical detection instance_key_recording_identity must equal "
                "the sealed acquisition recording_id: "
                f"provided={instance_key_recording_identity!r}, "
                f"sealed={recording_identity!r}."
            )
        if not recording_identity:
            raise ValueError("Sealed acquisition recording_id must not be blank.")
        key_frame_indices = np.asarray(frame_indices, dtype=np.int64)
        key_frame_domain = "recording_parent_frame_index"
        identity_array_name = "instance_key"
        if instance_key_frame_indices is None:
            raise RuntimeError(
                "Canonical detection lacks its mechanically proven frame mapping."
            )
        dense_frame_mapping = np.asarray(instance_key_frame_indices, dtype=np.int64).reshape(-1)
        if dense_frame_mapping.shape[0] != int(n_frames):
            raise ValueError(
                "instance_key_frame_indices length does not match the source video: "
                f"{dense_frame_mapping.shape[0]} != {n_frames}."
            )
        if np.any(dense_frame_mapping < 0):
            raise ValueError("instance_key_frame_indices contains negative parent-frame values.")
        if np.unique(dense_frame_mapping).shape[0] != dense_frame_mapping.shape[0]:
            raise ValueError("instance_key_frame_indices must map run-local frames one-to-one.")
        key_frame_indices = dense_frame_mapping[np.asarray(frame_indices, dtype=np.int64)]
        key_frame_mapping_sha256 = hashlib.sha256(
            np.ascontiguousarray(dense_frame_mapping).view(np.uint8)
        ).hexdigest()
        output_row_ids = mint_detection_instance_keys(
            recording_identity=recording_identity,
            frame_indices=key_frame_indices,
            bbox_norm_coords=bbox_coords,
            class_ids=class_ids,
        )
    else:
        # Scratch transfer artifacts carry only a dense run-local row number.
        # Canonical observation identity is minted after authoritative placement.
        key_frame_indices = np.asarray(frame_indices, dtype=np.int64)
        output_row_ids = np.arange(total_detections, dtype=np.uint64)
    source_acquisition_frame_indices: np.ndarray | None = None
    bbox_img_xyxy: np.ndarray | None = None
    centers_img_xy: np.ndarray | None = None
    if canonical_acquisition_frame is not None:
        temporary_frame_evidence, temporary_frame_checkpoints = (
            _publish_detection_frame_evidence(
                root,
                detect_group,
                acquisition_frame=canonical_acquisition_frame,
            )
        )
        coordinate_checkpoints.extend(temporary_frame_checkpoints)
        temporary_checkpoint_count = len(temporary_frame_checkpoints)
        try:
            if int(frame_idx) != int(n_frames):
                raise ValueError(
                    "Canonical detection did not decode the complete acquisition "
                    f"frame domain: processed={frame_idx}, expected={n_frames}."
                )
            refreshed_acquisition = _load_full_acquisition_video_source(
                root,
                video_path=video_path,
                output_zarr=output_zarr,
                decoded_frame_count=int(n_frames),
                decoded_width=int(width),
                decoded_height=int(height),
            )
            if (
                refreshed_acquisition.record_ref
                != canonical_acquisition_frame.record_ref
                or refreshed_acquisition.record_sha256
                != canonical_acquisition_frame.record_sha256
            ):
                raise ValueError(
                    "Acquisition source authority changed during detection decode."
                )
            source_acquisition_frame_indices = np.asarray(
                key_frame_indices,
                dtype=np.int64,
            )
            bbox_img_xyxy, centers_img_xy = (
                derive_detection_source_camera_geometry(
                    bbox_coords,
                    frame_evidence=temporary_frame_evidence,
                )
            )
        except BaseException as exc:
            failures = _restore_detection_coordinate_checkpoints(
                temporary_frame_checkpoints,
                cause=exc,
            )
            if not failures and temporary_checkpoint_count:
                del coordinate_checkpoints[-temporary_checkpoint_count:]
            if failures:
                raise RuntimeError(
                    "Temporary detection frame evidence failed and checkpoint "
                    f"rollback was incomplete: {failures!r}."
                ) from exc
            raise
        temporary_publication = RuntimeError(
            "temporary detection frame evidence was used only for derivation"
        )
        failures = _restore_detection_coordinate_checkpoints(
            temporary_frame_checkpoints,
            cause=temporary_publication,
        )
        if failures:
            raise RuntimeError(
                "Temporary detection frame evidence could not be fully restored: "
                f"{failures!r}."
            ) from temporary_publication
        if temporary_checkpoint_count:
            del coordinate_checkpoints[-temporary_checkpoint_count:]
    _record_timing(stage_timings, 'array_assembly_seconds_total', assembly_start)
    
    preferred_det_chunk = max(1024, max(1, batch_size) * 8)
    det_chunk = max(1, min(max(1, total_detections), preferred_det_chunk, 16384))
    
    # Save to zarr
    zarr_write_start = time.perf_counter()
    detect_shard_write = _write_detection_output_arrays(
        detect_group,
        frame_indices=frame_indices,
        bbox_coords=bbox_coords,
        scores=scores,
        class_ids=class_ids,
        instance_keys=output_row_ids,
        frame_counts=frame_counts,
        det_chunk=det_chunk,
        detect_row_shard_rows=detect_row_shard_rows,
        detect_frame_shard_rows=int(detect_frame_shard_rows),
        source_acquisition_frame_indices=source_acquisition_frame_indices,
        bbox_img_xyxy=bbox_img_xyxy,
        centers_img_xy=centers_img_xy,
        identity_array_name=identity_array_name,
    )
    _record_timing(stage_timings, 'zarr_write_seconds_total', zarr_write_start)
    
    # Calculate statistics
    frames_with_detections = np.sum(frame_counts > 0)
    coverage_percent = (frames_with_detections / n_frames) * 100
    
    stats = {
        'total_detections': int(total_detections),
        'frames_with_detections': int(frames_with_detections),
        'percent_frames_with_detections': float(coverage_percent),
        'frames_with_zero_detections': int(np.sum(frame_counts == 0)),
        'frames_with_multiple_detections': int(np.sum(frame_counts > 1)),
        'mean_detections_per_frame': float(total_detections / n_frames),
        'mean_confidence': float(np.mean(scores)) if len(scores) > 0 else 0.0,
        'min_confidence': float(np.min(scores)) if len(scores) > 0 else 0.0,
        'max_confidence': float(np.max(scores)) if len(scores) > 0 else 0.0,
    }
    timing_summary = {
        'schema_version': 1,
        'timing_policy': 'wall_clock_no_per_batch_cuda_sync',
        'decode_backend_requested': decode_backend_requested,
        'decode_backend_effective': decode_backend_effective,
        'decode_domain_proof': decode_domain_proof,
        'pynvvc_surface_materialization': (
            PYNVVC_SURFACE_MATERIALIZATION if use_pynvvc else 'not_applicable'
        ),
        'video_reader_type': video_reader_type,
        'frames_processed': int(frame_idx),
        'batches_processed': int(len(inference_times)),
        'processing_seconds_total': float(total_time),
        'processing_fps': float(frame_idx / total_time) if total_time > 0 else 0.0,
        'read_decode_seconds_total': float(stage_timings['read_decode_seconds_total']),
        'preprocess_resize_seconds_total': float(stage_timings['preprocess_resize_seconds_total']),
        'predict_seconds_total': float(stage_timings['predict_seconds_total']),
        'postprocess_seconds_total': float(stage_timings['postprocess_seconds_total']),
        'array_assembly_seconds_total': float(stage_timings['array_assembly_seconds_total']),
        'zarr_write_seconds_total': float(stage_timings['zarr_write_seconds_total']),
        'predict_avg_batch_ms': float(avg_inference * 1000.0) if inference_times else 0.0,
        'read_decode_avg_batch_ms': float(np.mean(read_times) * 1000.0) if read_times else 0.0,
    }
    
    # Store metadata
    detection_attrs = {
        'detect_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'detection_method': 'yolo',  # 'blob' for traditional, 'yolo' for neural net
        'artifact_mutability': 'raw_immutable',
        'detection_source': 'external_video',
        'coordinate_contract_mode': coordinate_contract_mode,
        OBSERVATION_ROW_COUNT_ATTR: int(total_detections),
        'model_type': 'yolo_object_detection',
        'model_path': str(model_artifact.get('path') or model_path.absolute()),
        'model_name': Path(
            str(model_artifact.get('path') or model_path.absolute())
        ).name,
        'source_video_width': source_video_width,
        'source_video_height': source_video_height,
        'source_full_width': source_full_width,
        'source_full_height': source_full_height,
        'inference_width': int(inference_width),
        'inference_height': int(inference_height),
        'validated_backend_result_count': int(validated_backend_result_count),
        'validated_backend_result_orig_shape_hw': [
            int(inference_height),
            int(inference_width),
        ],
        'parameters': {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'max_det': max_det,
            'batch_size': batch_size,
            'resize_dims': requested_resize_dims,
            'resize_dims_source': resize_dims_source,
            'imgsz': imgsz_applied,
            'imgsz_applied': imgsz_applied,
            'imgsz_legacy_input': cli_imgsz_legacy,
            'pre_resize_dims': pre_resize_dims,
            'effective_input_resize_dims': effective_input_resize_dims,
            'tensor_resize_dims': (
                effective_input_resize_dims
                if tensor_input_path and effective_input_resize_dims is not None
                else None
            ),
            'legacy_video_resize_dims': legacy_video_resize_dims,
            'decode_backend_requested': decode_backend_requested,
            'decode_backend_effective': decode_backend_effective,
            'pynvvc_surface_materialization': (
                PYNVVC_SURFACE_MATERIALIZATION if use_pynvvc else 'not_applicable'
            ),
            'detect_row_shard_rows': (
                int(detect_row_shard_rows) if detect_row_shard_rows is not None else None
            ),
            'detect_frame_shard_rows': (
                int(detect_frame_shard_rows) if detect_row_shard_rows is not None else None
            ),
            'detect_storage_policy': (
                'default_indexed_sharding_v1'
                if detect_row_shard_rows is not None
                else 'explicit_regular_chunks_override'
            ),
            'coordinate_contract_mode': coordinate_contract_mode,
        },
        'summary_statistics': stats,
        'timing_summary': timing_summary,
        'detect_storage_layout': (
            'indexed_sharding_v1' if detect_row_shard_rows is not None else 'regular_chunks_v1'
        ),
        'detect_storage_policy': (
            'default_indexed_sharding_v1'
            if detect_row_shard_rows is not None
            else 'explicit_regular_chunks_override'
        ),
        'detect_row_shard_rows': (
            int(detect_row_shard_rows) if detect_row_shard_rows is not None else None
        ),
        'detect_frame_shard_rows': (
            int(detect_frame_shard_rows) if detect_row_shard_rows is not None else None
        ),
        'detect_shard_write': detect_shard_write,
        'git_commit': git_info.get('commit_hash', 'unknown'),
        'git_branch': git_info.get('branch', 'unknown'),
        'hostname': env_info['platform']['hostname']
    }
    if coordinate_contract_mode == "artifact_unbound":
        detection_attrs.update(
            {
                'coordinate_contract': UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
                'artifact_row_id_contract': DETECTION_ARTIFACT_ROW_ID_CONTRACT,
                'artifact_row_id_scope': 'run_local_noncanonical',
            }
        )
    else:
        if key_frame_domain is None:
            raise RuntimeError("Canonical detection lacks its instance-key frame domain.")
        detection_attrs.update(
            instance_key_attrs(
                recording_identity,
                frame_domain=key_frame_domain,
                frame_mapping_source=instance_key_frame_mapping_source,
                frame_mapping_sha256=key_frame_mapping_sha256,
            )
        )
    detect_group.attrs.update(detection_attrs)
    if source_zarr_meta:
        if source_zarr_meta.get("downsampled_resolution") is not None:
            detect_group.attrs['palette_downsampled_resolution'] = list(source_zarr_meta["downsampled_resolution"])
        if source_zarr_meta.get("has_downsampled") is not None:
            detect_group.attrs['palette_has_downsampled'] = bool(source_zarr_meta["has_downsampled"])
        if source_zarr_meta.get("has_full_resolution") is not None:
            detect_group.attrs['palette_has_full_resolution'] = bool(source_zarr_meta["has_full_resolution"])
        if source_zarr_meta.get("original_resolution") is not None:
            detect_group.attrs['palette_original_resolution'] = list(source_zarr_meta["original_resolution"])
    if source_zarr_path is not None:
        detect_group.attrs['source_zarr_path'] = str(source_zarr_path)
    detect_group.attrs['inference_duration_seconds'] = float(total_time)
    detect_group.attrs['inference_average_fps'] = float(n_frames / total_time) if total_time > 0 else 0.0
    detect_group.attrs['inference_avg_batch_ms'] = float(avg_inference * 1000.0) if inference_times else 0.0
    detect_group.attrs['inference_avg_read_ms'] = float(np.mean(read_times) * 1000.0) if read_times else 0.0
    detect_group.attrs['read_decode_seconds_total'] = timing_summary['read_decode_seconds_total']
    detect_group.attrs['preprocess_resize_seconds_total'] = timing_summary['preprocess_resize_seconds_total']
    detect_group.attrs['predict_seconds_total'] = timing_summary['predict_seconds_total']
    detect_group.attrs['postprocess_seconds_total'] = timing_summary['postprocess_seconds_total']
    detect_group.attrs['array_assembly_seconds_total'] = timing_summary['array_assembly_seconds_total']
    detect_group.attrs['zarr_write_seconds_total'] = timing_summary['zarr_write_seconds_total']
    detect_group.attrs['decode_backend_requested'] = decode_backend_requested
    detect_group.attrs['decode_backend_effective'] = decode_backend_effective
    detect_group.attrs['decode_domain_proof'] = decode_domain_proof
    detect_group.attrs['video_reader_type'] = video_reader_type

    platform_info = env_info.get("platform") or {}
    scheduler_info = platform_info.get("lsf") or platform_info.get("slurm")
    provenance_record = build_stage_provenance(
        stage="detect",
        command=" ".join(sys.argv),
        created_at_utc=str(detect_group.attrs.get("detect_timestamp_utc")),
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
            "fqdn": platform_info.get("fqdn"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "version": platform_info.get("version"),
            "machine": platform_info.get("machine"),
            "processor": platform_info.get("processor"),
            "cpu_cores": platform_info.get("cpu_cores"),
            "cpu_details": platform_info.get("cpu_details"),
            "memory": platform_info.get("memory"),
            "disk": platform_info.get("disk"),
            "python_version": platform_info.get("python_version"),
            "python_implementation": platform_info.get("python_implementation"),
        },
        scheduler=scheduler_info,
        parameters=dict(detect_group.attrs.get("parameters") or {}),
        inputs={
            "frame_source": "external",
            "source_video_path": root.attrs.get("source_video_path"),
            "source_zarr_path": detect_group.attrs.get("source_zarr_path"),
            "decode_backend_requested": decode_backend_requested,
            "decode_backend_effective": decode_backend_effective,
        },
        artifacts={
            "model_path": str(model_path.absolute()),
            "model_name": model_path.name,
            "device": "cuda" if use_gpu else "cpu",
            "gpu": env_info.get("gpu"),
        },
    )
    provenance_record["timing"] = dict(timing_summary)
    effective_run_provenance = run_provenance if run_provenance is not None else cli_provenance
    if effective_run_provenance is None:
        effective_run_provenance = build_run_provenance(
            command="fisheye.detection.detect_yolo",
            params={
                "video_path": str(video_path),
                "model_path": str(model_path.absolute()),
                "output_zarr": str(output_zarr) if output_zarr is not None else None,
                "config_path": config_path,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "max_det": max_det,
                "batch_size": batch_size,
                "resize_dims": list(resize_dims) if resize_dims is not None else None,
                "imgsz": imgsz,
                "decode_backend_requested": decode_backend_requested,
                "decode_backend_effective": decode_backend_effective,
                "use_gpu": bool(use_gpu),
                "run_name": run_name,
                "detect_row_shard_rows": detect_row_shard_rows,
                "detect_frame_shard_rows": (
                    int(detect_frame_shard_rows) if detect_row_shard_rows is not None else None
                ),
                "detect_storage_policy": (
                    "default_indexed_sharding_v1"
                    if detect_row_shard_rows is not None
                    else "explicit_regular_chunks_override"
                ),
                "coordinate_contract_mode": coordinate_contract_mode,
                **(
                    {"coordinate_contract": "canonical_v2"}
                    if coordinate_contract_mode == "canonical"
                    else {}
                ),
            },
            input_run_ids={},
            cwd=Path.cwd(),
        )
    effective_run_provenance = append_input_artifacts(effective_run_provenance, [model_artifact])
    if effective_run_provenance is not None:
        detect_group.attrs[RUN_PROVENANCE_ATTR] = dict(effective_run_provenance)
        detect_group.attrs[CLI_RUN_PROVENANCE_ATTR] = dict(effective_run_provenance)
    write_stage_provenance(detect_group, provenance_record)

    if canonical_acquisition_frame is not None:
        validate_immutable_yolo_storage(
            detect_group,
            stage="detect",
            row_shard_rows=detect_row_shard_rows,
            frame_shard_rows=detect_frame_shard_rows,
        )
    else:
        _validate_unbound_detection_artifact_storage(
            detect_group,
            source_total_frames=int(n_frames),
        )

    if canonical_acquisition_frame is not None:
        canonical_frame_evidence, frame_checkpoints = (
            _publish_detection_frame_evidence(
                root,
                detect_group,
                acquisition_frame=canonical_acquisition_frame,
            )
        )
        coordinate_checkpoints.extend(frame_checkpoints)
        observation_checkpoint = (
            capture_observation_coordinate_publication_checkpoint(
                detect_group,
                detect_group["instance_key"],
                detect_group["source_acquisition_frame_index"],
                detect_group["bbox_norm_coords"],
                detect_group["bbox_img_xyxy"],
                detect_group["centers_img_xy"],
                detect_group["class_ids"],
            )
        )
        coordinate_checkpoints.append(observation_checkpoint)
        mapping_record = _publish_detection_acquisition_mapping(
            run_group=detect_group,
            acquisition_frame=canonical_acquisition_frame,
        )
        backend_result_projection = publish_detection_backend_result_projection(
            detect_group,
            detect_group["bbox_norm_coords"],
            frame_evidence=canonical_frame_evidence,
            model_artifact=model_artifact,
        )
        instance_key_derivation = publish_detection_instance_key_derivation(
            detect_group,
            detect_group["instance_key"],
            detect_group["source_acquisition_frame_index"],
            detect_group["bbox_norm_coords"],
            detect_group["class_ids"],
            acquisition_frame=canonical_acquisition_frame,
            acquisition_mapping=mapping_record,
        )
        publish_detection_observation_cardinality(
            detect_group,
            acquisition_frame=canonical_acquisition_frame,
        )
        publish_detection_observation_geometry(
            detect_group,
            detect_group["instance_key"],
            detect_group["source_acquisition_frame_index"],
            detect_group["bbox_norm_coords"],
            detect_group["bbox_img_xyxy"],
            detect_group["centers_img_xy"],
            frame_evidence=canonical_frame_evidence,
            source_lineage_records=(
                mapping_record,
                backend_result_projection,
                instance_key_derivation,
            ),
        )
        if int(total_detections) == 0:
            _publish_empty_detection_observation_declaration(
                detect_group,
                acquisition_frame=canonical_acquisition_frame,
                decoded_frame_count=int(frame_idx),
                decode_domain_proof=decode_domain_proof,
            )
        detect_group.attrs["coordinate_contract"] = "canonical_v2"
        mark_run_complete(
            detect_group,
            parent_group=parent_group,
            run_name=run_name,
            run_provenance=effective_run_provenance,
        )
        _load_persisted_detection_observation_geometry(
            root,
            f"{DETECTION_RUN_FAMILY}/{run_name}",
            require_selector_eligible=False,
        )
    else:
        mark_run_complete(
            detect_group,
            parent_group=None,
            run_name=run_name,
            run_provenance=effective_run_provenance,
        )

    console.print("[green]✓[/green] Detection arrays validated")
    
    # Calculate storage savings
    zarr_size_mb = (total_detections * 32) / 1024 / 1024  # Rough estimate
    video_size_mb = (n_frames * width * height) / 1024 / 1024  # If we stored grayscale
    if zarr_size_mb > 0:
        storage_comparison_line = f"  Saved vs full import: ~{video_size_mb:.1f} MB ({video_size_mb/zarr_size_mb:.0f}× smaller)"
    else:
        storage_comparison_line = f"  Saved vs full import: ~{video_size_mb:.1f} MB"
    
    # Print summary
    summary_text = f"""[green]✓[/green] Inference complete!

[bold]Results:[/bold]
  Detections: {total_detections:,}
  Coverage: {coverage_percent:.1f}% ({frames_with_detections:,}/{n_frames:,} frames)
  Mean confidence: {stats['mean_confidence']:.3f}

[bold]Storage:[/bold]
  Zarr size: ~{zarr_size_mb:.1f} MB (detections only)
{storage_comparison_line}

[bold]Output:[/bold]
  {output_zarr}
  
[bold]Next steps:[/bold]
  # Refine detections
  python -m fisheye.refinement.refine_detect {output_zarr}
  
  # Run arena assignment
  python -m fisheye.tracking.arena_assignment {output_zarr}
  
  # Visualize
  python -m fisheye.visualization.detection_visualizer {output_zarr}
"""
    
    panel = Panel(
        summary_text,
        title="[bold green]Detection Summary[/bold green]",
        border_style="green"
    )
    console.print("\n")
    console.print(panel)

    # Close every decoder before the canonical commit point. The encompassing
    # guard's second close is an idempotent no-op, so no cleanup can fail after
    # selector eligibility becomes visible.
    cleanup_failures = publication_attempt.close_video_resources()
    if cleanup_failures:
        raise RuntimeError(
            "Detection output validated but video cleanup was incomplete: "
            f"{cleanup_failures!r}."
        )
    if canonical_acquisition_frame is not None:
        _publish_validated_detection_selection(
            parent_group,
            detect_group,
            run_name=run_name,
        )

    return run_name


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on video without importing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (model path in config)
  python -m fisheye.detection.detect_yolo video.mp4
  python -m fisheye.detection.detect_yolo video.mp4 --output test.zarr
  
  # Explicit model and output
  python -m fisheye.detection.detect_yolo video.mp4 --model model.pt --output test.zarr
  
  # With custom thresholds (overrides config)
  python -m fisheye.detection.detect_yolo video.mp4 --conf 0.35 --batch-size 64
  
  # Force CPU
  python -m fisheye.detection.detect_yolo video.mp4 --cpu
  
  # Then run downstream analysis
  python -m fisheye.refinement.refine_detect output.zarr
  python -m fisheye.tracking.arena_assignment output.zarr
        """
    )
    
    parser.add_argument('video_path', help='Input video file')
    parser.add_argument('--model', '--model-path', dest='model_path', default=None,
                       help='Trained YOLO model (.pt) - optional if in config')
    parser.add_argument('--output', '--output-zarr', dest='output_zarr', default=None,
                       help='Output zarr path - optional, auto-generated if not provided')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--conf', type=float, default=None, 
                       help='Confidence threshold (overrides config)')
    parser.add_argument('--iou', type=float, default=None, 
                       help='IoU threshold for NMS (overrides config)')
    parser.add_argument('--max-det', type=int, default=None, 
                       help='Max detections per frame (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Inference batch size (overrides config)')
    detect_storage_group = parser.add_mutually_exclusive_group()
    detect_storage_group.add_argument(
        '--detect-row-shard-rows',
        type=int,
        default=DEFAULT_DETECT_ROW_SHARD_ROWS,
        help=(
            'Requested outer rows for indexed-sharded detection arrays '
            f'(default: {DEFAULT_DETECT_ROW_SHARD_ROWS}).'
        ),
    )
    detect_storage_group.add_argument(
        '--no-detect-sharding',
        action='store_const',
        dest='detect_row_shard_rows',
        const=None,
        help='Use ordinary chunks for YOLO detection outputs.',
    )
    parser.add_argument(
        '--detect-frame-shard-rows',
        type=int,
        default=DEFAULT_DETECT_FRAME_SHARD_ROWS,
        help='Outer row count for frame-count arrays when detection sharding is enabled.',
    )
    parser.add_argument(
        '--resize-dims',
        nargs='+',
        type=int,
        default=None,
        help='Canonical inference size override [h w] (or one value for square); mapped to YOLO imgsz',
    )
    parser.add_argument(
        '--imgsz',
        nargs='+',
        type=int,
        default=None,
        help='Legacy alias for YOLO inference size; normalized into --resize-dims',
    )
    parser.add_argument(
        '--decode-backend',
        choices=DECODE_BACKEND_CHOICES,
        default=None,
        help=(
            "Video decode backend. Default auto prefers pynvvc_nv12_rgb "
            "when CUDA and resize are available, then falls back to Decord/OpenCV."
        ),
    )
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU inference')
    parser.add_argument(
        '--write-raw-video-metadata',
        action='store_true',
        help='Write metadata-only raw_video attrs (no frames) for registry/provenance',
    )
    parser.add_argument(
        '--overwrite-raw-video-metadata',
        action='store_true',
        help='Overwrite existing raw_video attrs when writing metadata-only import',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Optional explicit detect run group name (default: timestamped detect_<utc>).',
    )
    args = parser.parse_args()
    
    try:
        detect_yolo(
            video_path=args.video_path,
            model_path=args.model_path,
            output_zarr=args.output_zarr,
            config_path=args.config,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            resize_dims=args.resize_dims,
            imgsz=args.imgsz,
            decode_backend=args.decode_backend,
            use_gpu=not args.cpu if args.cpu else None,
            write_raw_video_metadata=args.write_raw_video_metadata,
            overwrite_raw_video_metadata=args.overwrite_raw_video_metadata,
            run_name=args.run_name,
            detect_row_shard_rows=args.detect_row_shard_rows,
            detect_frame_shard_rows=args.detect_frame_shard_rows,
        )
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
