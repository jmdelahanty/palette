"""YOLO-based keypoint detection for FishEye Zarr archives.

This mirrors :mod:`fisheye.detection.detect_keypoints_traditional` but uses a
trained Ultralytics YOLO pose model on existing ROI crops. It creates a new
``keypoints_runs`` group without overwriting prior runs and records metadata so
downstream tooling can distinguish between traditional and YOLO-derived
keypoints.
"""

from __future__ import annotations

import argparse
import copy
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import os
import sys
from queue import Full, Queue
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Mapping, Optional, Tuple, Sequence
from datetime import datetime, timezone
import time
from uuid import uuid4

import numpy as np
import torch
import zarr
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from ..registry.db import RegistryPaths
from ..registry.inline_refresh import refresh_keypoint_performance_details
from ..shared.crop_image_source import CropImageSource
from ..shared.frame_domains import FrameDomain, FrameDomainError, FrameDomains
from ..shared.zarr.crop_consumer import strict_crop_source_dimensions
from ..shared.zarr.training_crop_materialization import (
    bind_training_crop_materialization,
)
from ..shared.inference_timing import InferenceTimingProfiler
from ..shared.immutable_yolo_storage import validate_immutable_yolo_storage
from ..shared.keypoint_summary import build_frame_keypoint_counts
from ..shared.keypoint_coordinate_publication import (
    KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR,
    KEYPOINT_PUBLICATION_GENERATION_ATTR,
    KEYPOINT_PUBLICATION_OWNER_ATTR,
    KEYPOINT_PUBLICATION_POLICY_ATTR,
    _activate_validated_keypoint_coordinate_surfaces,
    _load_completed_ineligible_keypoint_coordinate_surfaces,
    capture_keypoint_coordinate_publication_checkpoint,
    derive_keypoint_coordinate_batch,
    load_persisted_keypoint_crop_source,
    model_input_batch_to_roi,
    model_input_bbox_batch_to_roi,
    prepare_keypoint_coordinate_context,
    publish_keypoint_coordinate_surfaces,
    revalidate_keypoint_coordinate_batch_context,
    require_direct_keypoint_crop_pixel_source,
    rollback_keypoint_coordinate_publication,
)
from ..shared.model_input_transform import MODEL_INPUT_TRANSFORM_CHOICES, ModelInputTransform, resolve_model_input_transform
from ..shared.provenance_attrs import build_source_crop_snapshot_attrs, build_source_roi_pixel_attrs
from ..registry.stage_complete import emit_stage_completion
from ..shared.row_lineage import (
    copy_row_lineage_arrays,
    copy_selected_crop_row_lineage_arrays,
    write_direct_source_crop_row_ids,
)
from ..shared.row_source_signature import copy_selected_row_source_signatures
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.artifact_fingerprint import fingerprint_artifact
from ..shared.pose_model_schema_binding import (
    load_pose_model_schema_binding,
    pose_schema_from_model_binding,
)
from ..shared.pose_inference_failure import (
    POSE_INFERENCE_FAILURE_SCHEMA_ID,
    POSE_INFERENCE_FAILURE_SCHEMA_VERSION,
    PoseInferenceFailureCode,
    pose_inference_failure_code_map_json,
    pose_inference_failure_histogram,
    validate_pose_inference_failure_codes,
)
from ..shared.proof_verification import (
    finish_proof_verification,
    proof_verification_operation,
    restart_proof_verification,
)
from ..shared.run_provenance import (
    CLI_RUN_PROVENANCE_ATTR,
    RUN_PROVENANCE_ATTR,
    append_input_artifacts,
    build_run_provenance,
)
from ..shared.type_conversions import normalize_attr
from ..shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from ..pose.heading import compute_heading_from_spec
from ..shared.system_metadata import get_environment_info, get_git_info
from ..pose.schema import (
    PoseSchema,
    normalize_kpt_shape,
    schema_payload_from_package,
    undirected_edge_topology,
)
from ultralytics import YOLO, __version__ as ultralytics_version

DEFAULT_POSE_SCHEMA_NAME = "traditional_v1"
TRADITIONAL_POSE_SCHEMA, TRADITIONAL_POSE_ATTR_PAYLOAD = schema_payload_from_package(
    DEFAULT_POSE_SCHEMA_NAME
)
_KEYPOINT_STEP_NAME = "keypoints"
_KEYPOINT_STATUS_SOURCE = "runtime_keypoints_detect"
_KEYPOINT_INPUT_MODES = ("numpy-list", "tensor", "auto")
_KEYPOINT_COORDINATE_CONTRACT_MODES = ("canonical", "legacy_noncanonical")
_KEYPOINT_PROGRESS_SCHEMA_ID = "palette.keypoint_inference_progress.v1"
_DISABLE_REGISTRY_WRITES_ENV = "PALETTE_DISABLE_REGISTRY_WRITES"
DEFAULT_KEYPOINT_OUTPUT_PARENT = "keypoints_runs"
KEYPOINT_OUTPUT_PARENTS = (DEFAULT_KEYPOINT_OUTPUT_PARENT, "keypoint_shard_runs")
KEYPOINT_SHARD_WRITE_SCHEMA = "palette.keypoint_double_buffered_shards.v1"
DEFAULT_KEYPOINT_ROI_SHARD_ROWS = 131_072
DEFAULT_KEYPOINT_FRAME_SHARD_ROWS = 131_072
KEYPOINT_SHARD_WRITER_QUIESCE_TIMEOUT_SECONDS = 300.0
_KEYPOINT_PARENT_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    KEYPOINT_PUBLICATION_GENERATION_ATTR,
    KEYPOINT_PUBLICATION_POLICY_ATTR,
    KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR,
)


def _normalize_keypoint_output_parent(output_parent: Optional[str]) -> str:
    parent = (output_parent or DEFAULT_KEYPOINT_OUTPUT_PARENT).strip()
    if parent not in KEYPOINT_OUTPUT_PARENTS:
        allowed = ", ".join(KEYPOINT_OUTPUT_PARENTS)
        raise ValueError(f"Unsupported keypoint output parent '{parent}'. Expected one of: {allowed}")
    return parent


def _snapshot_selected_attrs(node: Any, names: Sequence[str]) -> dict[str, tuple[bool, Any]]:
    attrs = getattr(node, "attrs", None)
    if attrs is None or not hasattr(attrs, "keys"):
        raise RuntimeError("Cannot snapshot keypoint selector attrs.")
    return {
        name: (name in attrs, copy.deepcopy(attrs.get(name)))
        for name in names
    }


def _restore_selected_attrs(
    node: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
) -> None:
    attrs = node.attrs
    failures: list[str] = []
    for name, (present, value) in snapshot.items():
        try:
            if present:
                attrs[name] = copy.deepcopy(value)
            elif name in attrs:
                del attrs[name]
            if present:
                if name not in attrs or attrs[name] != value:
                    raise RuntimeError("restored value differs from snapshot")
            elif name in attrs:
                raise RuntimeError("attribute survived rollback")
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{name}: {exc}")
    if failures:
        raise RuntimeError(f"Keypoint selector rollback was incomplete: {failures!r}.")


def _restore_owned_keypoint_selectors(
    parent: Any,
    root: Any,
    parent_snapshot: Mapping[str, tuple[bool, Any]],
    root_snapshot: Mapping[str, tuple[bool, Any]],
    *,
    run_name: str | None,
    owner_token: str | None,
) -> None:
    """Restore only selector/publication state still owned by this attempt."""

    if run_name is None or owner_token is None:
        return
    parent_attrs = parent.attrs
    lease = parent_attrs.get(KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR)
    lease_owned = (
        isinstance(lease, dict)
        and lease.get("run_path") == f"keypoints_runs/{run_name}"
        and lease.get("publication_owner") == owner_token
    )
    lease_present, lease_snapshot = parent_snapshot[
        KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR
    ]
    if (
        lease is not None
        and not lease_owned
        and (not lease_present or lease != lease_snapshot)
    ):
        return
    if lease_owned:
        base_generation = lease.get("base_generation")
        next_generation = lease.get("next_generation")
        current_generation = parent_attrs.get(
            KEYPOINT_PUBLICATION_GENERATION_ATTR,
            0,
        )
        if (
            type(base_generation) is not int
            or type(next_generation) is not int
            or next_generation != base_generation + 1
            or type(current_generation) is not int
            or current_generation not in {base_generation, next_generation}
        ):
            return

    failures: list[str] = []

    def restore_if_owned(
        attrs: Any,
        snapshot: Mapping[str, tuple[bool, Any]],
        name: str,
        owned_value: Any,
    ) -> None:
        if attrs.get(name) != owned_value:
            return
        present, value = snapshot[name]
        if present:
            attrs[name] = copy.deepcopy(value)
        elif name in attrs:
            del attrs[name]
        if present and attrs.get(name) != value:
            raise RuntimeError("restored value differs from snapshot")
        if not present and name in attrs:
            raise RuntimeError("owned selector survived rollback")

    for name in ("latest", "latest_complete", "latest_pending"):
        try:
            restore_if_owned(parent_attrs, parent_snapshot, name, run_name)
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"parent {name}: {exc}")
    try:
        restore_if_owned(
            root.attrs,
            root_snapshot,
            "current_keypoint_group_path",
            f"keypoints_runs/{run_name}",
        )
    except BaseException as exc:  # pragma: no cover - hostile store
        failures.append(f"root current_keypoint_group_path: {exc}")

    if lease_owned:
        for name in (
            KEYPOINT_PUBLICATION_GENERATION_ATTR,
            KEYPOINT_PUBLICATION_POLICY_ATTR,
            KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR,
        ):
            try:
                present, value = parent_snapshot[name]
                if present:
                    parent_attrs[name] = copy.deepcopy(value)
                elif name in parent_attrs:
                    del parent_attrs[name]
                if present and parent_attrs.get(name) != value:
                    raise RuntimeError("restored value differs from snapshot")
                if not present and name in parent_attrs:
                    raise RuntimeError("owned publication state survived rollback")
            except BaseException as exc:  # pragma: no cover - hostile store
                failures.append(f"parent {name}: {exc}")
    if failures:
        raise RuntimeError(
            f"Keypoint owned selector rollback was incomplete: {failures!r}."
        )


class _KeypointAttemptFailureBoundary:
    """Fail one newly created keypoint attempt closed across the whole writer."""

    def __init__(self) -> None:
        self.root: Any | None = None
        self.parent: Any | None = None
        self.run: Any | None = None
        self.run_name: str | None = None
        self.run_path: str | None = None
        self.parent_selector_snapshot: dict[str, tuple[bool, Any]] | None = None
        self.root_pointer_snapshot: dict[str, tuple[bool, Any]] | None = None
        self.crop_source: Any | None = None
        self.shard_writer: _AlignedKeypointShardWriter | None = None
        self.coordinate_checkpoint: Any | None = None
        self.owner_token: str | None = None
        self.finalized = False

    def prepare(self, *, root: Any, parent: Any) -> None:
        if self.root is not None or self.parent is not None:
            raise RuntimeError("A keypoint attempt cannot bind more than one run parent.")
        self.root = root
        self.parent = parent
        self.parent_selector_snapshot = _snapshot_selected_attrs(
            parent,
            _KEYPOINT_PARENT_SELECTOR_ATTRS,
        )
        self.root_pointer_snapshot = _snapshot_selected_attrs(
            root,
            ("current_keypoint_group_path",),
        )
        self.owner_token = uuid4().hex

    def bind_run(self, run: Any, run_name: str) -> None:
        if self.run is not None:
            raise RuntimeError("A keypoint attempt cannot bind more than one run.")
        if (
            self.owner_token is None
            or run.attrs.get(KEYPOINT_PUBLICATION_OWNER_ATTR) != self.owner_token
        ):
            raise RuntimeError(
                "Keypoint child did not persist its atomic publication owner."
            )
        self.run = run
        self.run_name = str(run_name)
        self.run_path = str(getattr(run, "path", "")).strip("/") or None

    def bind_crop_source(self, crop_source: Any) -> None:
        self.crop_source = crop_source

    def close_crop_source(self) -> None:
        """Close the required scientific input before publication becomes authoritative."""

        if self.crop_source is None:
            return
        self.crop_source.close()
        self.crop_source = None

    def bind_shard_writer(self, shard_writer: _AlignedKeypointShardWriter) -> None:
        if self.shard_writer is not None:
            raise RuntimeError("A keypoint attempt cannot bind more than one shard writer.")
        self.shard_writer = shard_writer

    def bind_coordinate_checkpoint(self, checkpoint: Any) -> None:
        self.coordinate_checkpoint = checkpoint

    def mark_finalized(self) -> None:
        self.finalized = True
        self.coordinate_checkpoint = None

    def fresh_parent(self) -> Any:
        if self.parent is None:
            raise RuntimeError("Keypoint publication parent is not bound.")
        path = str(getattr(self.parent, "path", "")).strip("/")
        if self.root is not None and path:
            try:
                self.parent = self.root[path]
            except BaseException as exc:
                raise RuntimeError("Keypoint publication parent disappeared.") from exc
        return self.parent

    def require_owned_run(self) -> Any:
        if (
            self.run_name is None
            or self.owner_token is None
        ):
            raise RuntimeError("Keypoint publication ownership is not bound.")
        try:
            run = self.fresh_parent()[self.run_name]
        except BaseException as exc:
            raise RuntimeError("Keypoint publication child disappeared.") from exc
        if run.attrs.get(KEYPOINT_PUBLICATION_OWNER_ATTR) != self.owner_token:
            raise RuntimeError(
                "Keypoint publication child was replaced by another owner."
            )
        self.run = run
        return run

    def fail(self, original: BaseException) -> None:
        failures: list[str] = []
        writer_quiescent = True
        if self.shard_writer is not None:
            try:
                self.shard_writer.abort()
            except BaseException as exc:  # pragma: no cover - hostile worker/store
                failures.append(f"shard writer quiescence: {exc}")
            writer_quiescent = self.shard_writer.is_quiescent
            if not writer_quiescent:
                failures.append(
                    "shard writer remains active; failed status cannot be published safely"
                )
        if self.run is not None and not self.finalized:
            try:
                run_for_failure = self.require_owned_run()
            except BaseException:
                run_for_failure = None
            if (
                writer_quiescent
                and run_for_failure is not None
                and self.coordinate_checkpoint is not None
                and run_for_failure.attrs.get("stage_selector_eligible") is not True
            ):
                try:
                    rollback_keypoint_coordinate_publication(
                        self.coordinate_checkpoint
                    )
                    # Rollback writes through fresh root-resolved handles. The
                    # previously opened attrs mapping still caches the
                    # pre-rollback publication graph and would resurrect it
                    # when failure status is written.
                    run_for_failure = self.require_owned_run()
                except BaseException as exc:  # pragma: no cover - hostile store
                    failures.append(f"coordinate publication: {exc}")
            publication_committed = (
                run_for_failure is not None
                and run_for_failure.attrs.get("stage_selector_eligible") is True
            )
            if writer_quiescent and run_for_failure is not None and not publication_committed:
                try:
                    run_for_failure.attrs["stage_selector_eligible"] = False
                    mark_run_failed(
                        run_for_failure,
                        parent_group=None,
                        run_name=self.run_name,
                        error=f"keypoint writer failed: {original}",
                    )
                    run_for_failure.attrs["stage_selector_eligible"] = False
                except BaseException as exc:  # pragma: no cover - hostile store
                    failures.append(f"run completion: {exc}")
            if (
                run_for_failure is not None
                and not publication_committed
                and self.parent_selector_snapshot is not None
                and self.root_pointer_snapshot is not None
                and self.root is not None
            ):
                try:
                    _restore_owned_keypoint_selectors(
                        self.fresh_parent(),
                        self.root,
                        self.parent_selector_snapshot,
                        self.root_pointer_snapshot,
                        run_name=self.run_name,
                        owner_token=self.owner_token,
                    )
                except BaseException as exc:  # pragma: no cover - hostile store
                    failures.append(f"owned selectors: {exc}")
        if self.crop_source is not None:
            try:
                self.close_crop_source()
            except BaseException as exc:  # pragma: no cover - hostile source
                failures.append(f"crop source close: {exc}")
        if failures:
            raise RuntimeError(
                "Keypoint attempt failed and fail-closed rollback was incomplete: "
                f"{failures!r}."
            ) from original


_ACTIVE_KEYPOINT_ATTEMPT: ContextVar[
    _KeypointAttemptFailureBoundary | None
] = ContextVar("active_keypoint_attempt", default=None)


def _fail_closed_keypoint_attempt(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        boundary = _KeypointAttemptFailureBoundary()
        token = _ACTIVE_KEYPOINT_ATTEMPT.set(boundary)
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            boundary.fail(exc)
            raise
        finally:
            _ACTIVE_KEYPOINT_ATTEMPT.reset(token)

    return wrapped


def _progress_json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _progress_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_progress_json_ready(item) for item in value]
    return str(value)


def _write_keypoint_progress_jsonl(
    progress_jsonl: Optional[Path],
    event: str,
    **payload: Any,
) -> None:
    if progress_jsonl is None:
        return
    progress_path = Path(progress_jsonl).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {
        "schema_id": _KEYPOINT_PROGRESS_SCHEMA_ID,
        "event": str(event),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_progress_json_ready(record), sort_keys=True) + "\n")


def _warn_postcommit_failure(console: Console, *, label: str, error: BaseException) -> None:
    """Report non-authoritative telemetry failures without invalidating science."""

    try:
        console.print(
            "[yellow]Warning:[/yellow] keypoint publication is complete, "
            f"but {label} failed: {error}"
        )
    except Exception:
        # A broken console must not turn a durable, freshly verified publication
        # into an apparent scientific rollback.
        pass


def _infer_roi_cache_source_tier(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    resolved = str(Path(path).expanduser().resolve())
    if resolved.startswith("/scratch/") or resolved.startswith("/tmp/"):
        return "node_scratch"
    if resolved.startswith("/groups/") or resolved.startswith("/misc/public/"):
        return "prfs_workflow_scratch"
    return "unknown"


def _current_model_device(model: YOLO, *, fallback: Optional[str]) -> Optional[str]:
    try:
        return str(next(model.model.parameters()).device)
    except (AttributeError, StopIteration):
        return fallback


def _revalidate_keypoint_model_artifact(
    model_path: Path,
    expected_artifact: Mapping[str, Any],
    *,
    checkpoint: str,
) -> None:
    """Re-hash the exact live model and reject replacement or in-place drift."""

    expected_fields = (
        "role",
        "path",
        "fingerprint_scheme",
        "sha256",
        "size_bytes",
        "mtime_ns",
    )
    if (
        not isinstance(expected_artifact, Mapping)
        or expected_artifact.get("mismatch") is True
        or "error" in expected_artifact
        or any(name not in expected_artifact for name in expected_fields)
    ):
        raise ValueError(
            "Keypoint model lacks exact pre-load path/content/size/mtime evidence."
        )
    try:
        resolved = model_path.expanduser().resolve(strict=True)
        path_stat_before = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            fd_stat_before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            fd_stat_after = os.fstat(stream.fileno())
        path_stat_after = resolved.stat()
    except OSError as exc:
        raise ValueError(
            f"Keypoint model became unreadable {checkpoint}: {exc}."
        ) from exc

    def stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    if not (
        stat_identity(path_stat_before)
        == stat_identity(fd_stat_before)
        == stat_identity(fd_stat_after)
        == stat_identity(path_stat_after)
    ):
        raise ValueError(
            f"Keypoint model changed while it was revalidated {checkpoint}."
        )
    observed = {
        "role": "keypoint_model",
        "path": str(resolved),
        "fingerprint_scheme": "content_v1",
        "sha256": digest.hexdigest(),
        "size_bytes": int(path_stat_after.st_size),
        "mtime_ns": int(path_stat_after.st_mtime_ns),
    }
    expected = {name: expected_artifact.get(name) for name in expected_fields}
    if observed != expected:
        differing = sorted(
            name for name in expected_fields if observed[name] != expected[name]
        )
        raise ValueError(
            "Keypoint model path/content/size/mtime evidence changed "
            f"{checkpoint}; differing_fields={differing!r}."
        )


def _scheduler_payload(env_info: Dict[str, Any]) -> Dict[str, Any]:
    platform_info = env_info.get("platform", {})
    gpu_info = env_info.get("gpu", {})
    scheduler: Dict[str, Any] = {
        "execution_hostname": platform_info.get("hostname"),
        "execution_fqdn": platform_info.get("fqdn"),
    }
    lsf = platform_info.get("lsf")
    if isinstance(lsf, dict):
        scheduler.update(
            {
                "scheduler": "lsf",
                "job_id": lsf.get("job_id"),
                "job_name": lsf.get("job_name"),
                "job_index": lsf.get("job_index"),
                "queue": lsf.get("queue"),
                "num_processors": lsf.get("num_processors"),
                "hosts": lsf.get("hosts"),
                "mcpu_hosts": lsf.get("mcpu_hosts"),
                "djob_hostfile": lsf.get("djob_hostfile"),
                "gpu_request": lsf.get("gpu_request"),
                "cuda_visible_devices": lsf.get("cuda_visible_devices"),
                "cuda_visible_devices_orig": lsf.get("cuda_visible_devices_orig"),
            }
        )
    elif isinstance(platform_info.get("slurm"), dict):
        slurm = platform_info["slurm"]
        scheduler.update(
            {
                "scheduler": "slurm",
                "job_id": slurm.get("job_id"),
                "job_name": slurm.get("job_name"),
                "node_list": slurm.get("node_list"),
                "num_nodes": slurm.get("num_nodes"),
                "proc_id": slurm.get("proc_id"),
                "cpus_per_task": slurm.get("cpus_per_task"),
            }
        )
    else:
        scheduler["scheduler"] = None

    if isinstance(gpu_info, dict):
        scheduler["gpu_backend"] = gpu_info.get("backend")
        scheduler["gpu_available"] = gpu_info.get("available")
        scheduler["gpu_devices"] = gpu_info.get("devices")
        scheduler["lsf_allocated_gpus"] = gpu_info.get("lsf_allocated_gpus")
        scheduler["lsf_original_gpus"] = gpu_info.get("lsf_original_gpus")
    return {key: value for key, value in scheduler.items() if value is not None}


def _prepare_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    console: Console,
) -> Tuple[zarr.Group, zarr.Group, str]:
    return _prepare_run_group_for_parent(
        root,
        run_name,
        console,
        output_parent=DEFAULT_KEYPOINT_OUTPUT_PARENT,
    )


def _prepare_run_group_for_parent(
    root: zarr.Group,
    run_name: Optional[str],
    console: Console,
    *,
    output_parent: str,
) -> Tuple[zarr.Group, zarr.Group, str]:
    output_parent = _normalize_keypoint_output_parent(output_parent)
    parent = require_runs_parent(
        root,
        output_parent,
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    boundary = _ACTIVE_KEYPOINT_ATTEMPT.get()
    if boundary is not None:
        boundary.prepare(root=root, parent=parent)
    if run_name:
        if run_name in parent:
            raise ValueError(f"{output_parent}/{run_name} already exists")
        run_group = parent.create_group(run_name)
        publication_owner = (
            boundary.owner_token if boundary is not None else uuid4().hex
        )
        run_group.attrs[KEYPOINT_PUBLICATION_OWNER_ATTR] = publication_owner
        if run_group.attrs.get(KEYPOINT_PUBLICATION_OWNER_ATTR) != publication_owner:
            raise RuntimeError("Keypoint publication owner did not persist exactly.")
        if boundary is not None:
            boundary.bind_run(run_group, run_name)
        mark_run_started(run_group, run_name=run_name, stage="keypoints")
        note_pending_latest(parent, run_name)
        console.print(f"Created run group: [cyan]{output_parent}/{run_name}[/cyan]")
        return parent, run_group, run_name
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    base_name = (
        f"keypoints_{timestamp}"
        if output_parent == DEFAULT_KEYPOINT_OUTPUT_PARENT
        else f"keypoint_shard_{timestamp}"
    )
    resolved_name = base_name
    suffix = 1
    while resolved_name in parent:
        resolved_name = f"{base_name}_{suffix:03d}"
        suffix += 1
    run_group = parent.create_group(resolved_name)
    publication_owner = (
        boundary.owner_token if boundary is not None else uuid4().hex
    )
    run_group.attrs[KEYPOINT_PUBLICATION_OWNER_ATTR] = publication_owner
    if run_group.attrs.get(KEYPOINT_PUBLICATION_OWNER_ATTR) != publication_owner:
        raise RuntimeError("Keypoint publication owner did not persist exactly.")
    if boundary is not None:
        boundary.bind_run(run_group, resolved_name)
    mark_run_started(run_group, run_name=resolved_name, stage="keypoints")
    note_pending_latest(parent, resolved_name)
    console.print(f"Created run group: [cyan]{output_parent}/{resolved_name}[/cyan]")
    return parent, run_group, resolved_name


def _resolve_registry_path(registry: Optional[Path]) -> Optional[Path]:
    if registry is not None:
        return registry.expanduser().resolve()
    inferred = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    if not inferred.exists():
        return None
    return inferred


def _registry_writes_disabled() -> bool:
    value = os.environ.get(_DISABLE_REGISTRY_WRITES_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _emit_keypoint_step_status(
    *,
    root: zarr.Group,
    zarr_path: Path,
    run_name: str,
    method: Optional[str],
    coverage_pct: Optional[float],
    details: Dict[str, object],
    console: Optional[Console],
    registry: Optional[Path],
) -> None:
    if _registry_writes_disabled():
        if console is not None:
            console.print(
                "[yellow]Registry writes disabled:[/yellow] "
                f"skipping {_KEYPOINT_STEP_NAME} step-status sync for "
                f"run {run_name!r}"
            )
        return
    registry_path = _resolve_registry_path(registry)
    if registry_path is None:
        return
    status_details = dict(details)
    status_details.update(
        refresh_keypoint_performance_details(
            root=root,
            zarr_path=zarr_path,
            run_name=run_name,
            registry_path=registry_path,
            console=console,
        )
    )
    emit_stage_completion(
        root,
        zarr_path,
        step_name=_KEYPOINT_STEP_NAME,
        status="ok",
        source=_KEYPOINT_STATUS_SOURCE,
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        details_json=status_details,
        console=console,
        registry=registry_path,
        auto_registry_from_env=False,
        trigger_run_name=run_name,
    )


def _aligned_shards(chunks: Sequence[int], shard_rows: int | None) -> tuple[int, ...] | None:
    if shard_rows is None:
        return None
    requested = int(shard_rows)
    if requested <= 0:
        raise ValueError("Keypoint shard rows must be positive.")
    inner_rows = int(chunks[0])
    outer_rows = int(((requested + inner_rows - 1) // inner_rows) * inner_rows)
    return (outer_rows, *tuple(int(value) for value in chunks[1:]))


def _shard_kwargs(chunks: Sequence[int], shard_rows: int | None) -> dict[str, object]:
    shards = _aligned_shards(chunks, shard_rows)
    return {"shards": shards} if shards is not None else {}


def _digest_keypoint_array(array: zarr.Array, *, row_step: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, int(array.shape[0]), int(row_step)):
        stop = min(start + int(row_step), int(array.shape[0]))
        values = np.ascontiguousarray(array[start:stop, ...])
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _create_output_arrays(
    group: zarr.Group,
    total_rois: int,
    chunk_hint: int,
    *,
    n_keypoints: int,
    shard_rows: int | None = None,
) -> Dict[str, zarr.Array]:
    chunk_len = min(max(chunk_hint, 1), total_rois) if total_rois > 0 else 1
    data_chunk = (chunk_len, int(n_keypoints), 2)
    scalar_chunk = (chunk_len,)

    arrays = {
        "keypoints_roi": group.create_array(
            "keypoints_roi",
            shape=(total_rois, int(n_keypoints), 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs(data_chunk, shard_rows),
        ),
        "keypoints_img": group.create_array(
            "keypoints_img",
            shape=(total_rois, int(n_keypoints), 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs(data_chunk, shard_rows),
        ),
        "keypoints_norm": group.create_array(
            "keypoints_norm",
            shape=(total_rois, int(n_keypoints), 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs(data_chunk, shard_rows),
        ),
        "heading": group.create_array(
            "heading",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs(scalar_chunk, shard_rows),
        ),
        "confidence": group.create_array(
            "confidence",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs(scalar_chunk, shard_rows),
        ),
        "keypoint_confidences": group.create_array(
            "keypoint_confidences",
            shape=(total_rois, int(n_keypoints)),
            chunks=(chunk_len, int(n_keypoints)),
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs((chunk_len, int(n_keypoints)), shard_rows),
        ),
        "detection_success": group.create_array(
            "detection_success",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
            **_shard_kwargs(scalar_chunk, shard_rows),
        ),
        "pose_failure_codes": group.create_array(
            "pose_failure_codes",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="u1",
            fill_value=np.uint8(
                PoseInferenceFailureCode.NO_POSE_DETECTION_ABOVE_THRESHOLD
            ),
            overwrite=True,
            **_shard_kwargs(scalar_chunk, shard_rows),
        ),
        "pose_bbox_xyxy_roi": group.create_array(
            "pose_bbox_xyxy_roi",
            shape=(total_rois, 4),
            chunks=(chunk_len, 4),
            dtype="f4",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs((chunk_len, 4), shard_rows),
        ),
        "pose_bbox_xyxy_img": group.create_array(
            "pose_bbox_xyxy_img",
            shape=(total_rois, 4),
            chunks=(chunk_len, 4),
            dtype="f4",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs((chunk_len, 4), shard_rows),
        ),
        "pose_bbox_xyxy_norm": group.create_array(
            "pose_bbox_xyxy_norm",
            shape=(total_rois, 4),
            chunks=(chunk_len, 4),
            dtype="f4",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs((chunk_len, 4), shard_rows),
        ),
        "heading_finite": group.create_array(
            "heading_finite",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
            **_shard_kwargs(scalar_chunk, shard_rows),
        ),
        "heading_usable": group.create_array(
            "heading_usable",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
            **_shard_kwargs(scalar_chunk, shard_rows),
        ),
        "effective_threshold": group.create_array(
            "effective_threshold",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs(scalar_chunk, shard_rows),
        ),
        "effective_se2_radius": group.create_array(
            "effective_se2_radius",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
            **_shard_kwargs(scalar_chunk, shard_rows),
        ),
    }
    return arrays


class _AlignedKeypointShardWriter:
    """Accumulate sequential YOLO batches and write each physical shard once."""

    def __init__(
        self,
        destinations: Mapping[str, zarr.Array],
        *,
        shard_rows: int,
        buffer_count: int = 2,
    ) -> None:
        if int(buffer_count) != 2:
            raise ValueError("YOLO keypoint shard writing requires exactly two buffers.")
        if not destinations:
            raise ValueError("At least one keypoint destination array is required.")
        self.destinations = dict(destinations)
        row_counts = {int(array.shape[0]) for array in self.destinations.values()}
        if len(row_counts) != 1:
            raise ValueError("All buffered keypoint arrays must share one ROI row count.")
        self.total_rows = row_counts.pop()
        self.shard_rows = int(shard_rows)
        self.buffer_rows = min(self.shard_rows, max(1, self.total_rows))
        self.buffer_count = 2
        self.buffers = [
            {
                name: np.empty(
                    (self.buffer_rows, *tuple(int(value) for value in array.shape[1:])),
                    dtype=array.dtype,
                )
                for name, array in self.destinations.items()
            }
            for _ in range(self.buffer_count)
        ]
        self._free: Queue[int] = Queue(maxsize=self.buffer_count)
        for index in range(self.buffer_count):
            self._free.put(index)
        self._flush: Queue[object] = Queue(maxsize=self.buffer_count)
        self._sentinel = object()
        self._errors: list[BaseException] = []
        self._active_index: int | None = None
        self._active_start = 0
        self._active_rows = 0
        self._next_row = 0
        self._source_digests = {
            name: hashlib.sha256() for name in self.destinations
        }
        self._write_seconds = 0.0
        self._shutdown_started = False
        self._sentinel_sent = False
        self._quiescent = False
        self._aborted = False
        self._finish_summary: dict[str, object] | None = None
        self._quiesce_timeout_seconds = (
            KEYPOINT_SHARD_WRITER_QUIESCE_TIMEOUT_SECONDS
        )
        self._worker = Thread(
            target=self._flush_worker,
            name="yolo-keypoint-shard-writer",
            daemon=True,
        )
        self._worker.start()

    def _raise_error(self) -> None:
        if self._errors:
            raise RuntimeError("YOLO keypoint shard writer failed.") from self._errors[0]

    @property
    def is_quiescent(self) -> bool:
        """Return whether the flush queue is drained and its worker has exited."""

        return bool(self._quiescent and not self._worker.is_alive())

    def _acquire(self, *, start: int) -> None:
        self._raise_error()
        self._active_index = int(self._free.get())
        self._active_start = int(start)
        self._active_rows = 0

    def _submit(self, *, timeout: float | None = None) -> None:
        if self._active_index is None or self._active_rows <= 0:
            return
        item = (self._active_index, self._active_start, self._active_rows)
        if timeout is None:
            self._flush.put(item)
        else:
            self._flush.put(item, timeout=timeout)
        self._active_index = None
        self._active_rows = 0

    def write(self, start: int, values: Mapping[str, np.ndarray]) -> None:
        if self._shutdown_started:
            raise RuntimeError("Cannot write after keypoint shard-writer shutdown began.")
        self._raise_error()
        if int(start) != self._next_row:
            raise ValueError(
                f"Keypoint batches must be sequential; expected {self._next_row}, got {start}."
            )
        if set(values) != set(self.destinations):
            raise ValueError("Keypoint batch fields do not match destination arrays.")
        arrays = {name: np.asarray(value) for name, value in values.items()}
        batch_rows = {int(value.shape[0]) for value in arrays.values()}
        if len(batch_rows) != 1:
            raise ValueError("All keypoint batch arrays must share one row count.")
        remaining = batch_rows.pop()
        source_offset = 0
        while source_offset < remaining:
            if self._active_index is None:
                self._acquire(start=self._next_row)
            assert self._active_index is not None
            take = min(
                self.buffer_rows - self._active_rows,
                remaining - source_offset,
            )
            target = slice(self._active_rows, self._active_rows + take)
            source = slice(source_offset, source_offset + take)
            for name, value in arrays.items():
                expected = tuple(int(item) for item in self.destinations[name].shape[1:])
                if tuple(int(item) for item in value.shape[1:]) != expected:
                    raise ValueError(
                        f"Keypoint field {name!r} has trailing shape {value.shape[1:]}, expected {expected}."
                    )
                np.copyto(
                    self.buffers[self._active_index][name][target],
                    value[source],
                    casting="unsafe",
                )
            self._active_rows += take
            self._next_row += take
            source_offset += take
            if self._active_rows == self.buffer_rows:
                self._submit()
        self._raise_error()

    def _flush_worker(self) -> None:
        failed = False
        while True:
            item = self._flush.get()
            try:
                if item is self._sentinel:
                    return
                index, start, row_count = (int(value) for value in item)
                if failed:
                    continue
                stop = start + row_count
                started = time.perf_counter()
                for name, destination in self.destinations.items():
                    values = np.ascontiguousarray(self.buffers[index][name][:row_count])
                    self._source_digests[name].update(values.view(np.uint8))
                    destination[start:stop, ...] = values
                self._write_seconds += float(time.perf_counter() - started)
            except BaseException as exc:  # pragma: no cover - caller observes failure
                self._errors.append(exc)
                failed = True
            finally:
                if item is not self._sentinel:
                    self._free.put(int(item[0]))
                self._flush.task_done()

    def _shutdown(self, *, submit_active: bool) -> None:
        """Signal, drain, and join exactly once before returning to the caller."""

        if self.is_quiescent:
            return
        self._shutdown_started = True
        deadline = time.monotonic() + self._quiesce_timeout_seconds
        submission_error: BaseException | None = None
        if not self._sentinel_sent:
            if submit_active and not self._aborted:
                try:
                    self._submit(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except BaseException as exc:  # pragma: no cover - hostile queue
                    submission_error = exc
            else:
                # An active buffer has never been submitted and therefore owns no
                # durable rows.  Discard it on abort rather than publishing a
                # partial logical shard after the attempt has already failed.
                self._active_index = None
                self._active_rows = 0
            try:
                self._flush.put(
                    self._sentinel,
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            except Full as exc:  # pragma: no cover - requires a hung store write
                raise RuntimeError(
                    "Keypoint shard writer could not enqueue its shutdown sentinel "
                    "before the quiescence deadline."
                ) from exc
            self._sentinel_sent = True
        remaining = max(0.0, deadline - time.monotonic())
        self._worker.join(timeout=remaining)
        if self._worker.is_alive():  # pragma: no cover - requires a hung store write
            raise RuntimeError(
                "Keypoint shard writer did not quiesce before the shutdown deadline."
            )
        if self._flush.unfinished_tasks != 0:  # pragma: no cover - hostile worker
            raise RuntimeError(
                "Keypoint shard writer exited with unfinished queued writes."
            )
        self._quiescent = True
        if submission_error is not None:
            raise RuntimeError(
                "Keypoint shard writer could not submit its final active buffer."
            ) from submission_error

    def abort(self) -> None:
        """Idempotently discard unsubmitted rows and prove no worker remains live."""

        if self.is_quiescent:
            return
        self._aborted = True
        self._shutdown(submit_active=False)
        if not self.is_quiescent:  # pragma: no cover - defensive contract check
            raise RuntimeError("Keypoint shard writer abort did not prove quiescence.")

    def finish(self) -> dict[str, object]:
        if self._finish_summary is not None:
            return dict(self._finish_summary)
        if self._aborted:
            raise RuntimeError("Cannot finish an aborted keypoint shard writer.")
        if self._errors:
            self._aborted = True
            self._shutdown(submit_active=False)
            self._raise_error()
        self._shutdown(submit_active=True)
        self._raise_error()
        if self._next_row != self.total_rows:
            raise RuntimeError(
                f"Keypoint shard writer received {self._next_row} of {self.total_rows} rows."
            )
        validation_started = time.perf_counter()
        source_hashes = {
            name: digest.hexdigest() for name, digest in self._source_digests.items()
        }
        destination_hashes = {
            name: _digest_keypoint_array(array, row_step=self.shard_rows)
            for name, array in self.destinations.items()
        }
        validation_seconds = float(time.perf_counter() - validation_started)
        if source_hashes != destination_hashes:
            raise RuntimeError(
                "YOLO keypoint shard digest mismatch: "
                f"source={source_hashes} destination={destination_hashes}"
            )
        buffer_bytes_each = int(
            sum(array.nbytes for array in self.buffers[0].values())
        )
        self._finish_summary = {
            "schema_id": KEYPOINT_SHARD_WRITE_SCHEMA,
            "status": "complete",
            "write_mode": "double_buffered_direct",
            "row_count": int(self.total_rows),
            "buffer_count": int(self.buffer_count),
            "buffer_rows": int(self.buffer_rows),
            "buffer_bytes_each": buffer_bytes_each,
            "total_buffer_bytes": int(buffer_bytes_each * self.buffer_count),
            "write_seconds": float(self._write_seconds),
            "validation_seconds": validation_seconds,
            "source_sha256_by_array": source_hashes,
            "destination_sha256_by_array": destination_hashes,
            "exact_match": True,
        }
        return dict(self._finish_summary)


def _prepare_refined_roi_overrides(
    root: zarr.Group,
    crop_group: zarr.Group,
    total_rois: int,
    roi_shape: Tuple[int, int],
    console: Console,
) -> Optional[Dict[str, Any]]:
    path = crop_group.attrs.get("refined_roi_path")
    if not path:
        return None
    if path not in root:
        console.print(
            f"[yellow]Refined ROI group '{path}' not found; continuing with original crops.[/yellow]"
        )
        return None
    group = root[path]
    required = {"detection_indices", "roi_images", "roi_coordinates_full"}
    if not required.issubset(set(group.keys())):
        console.print(
            f"[yellow]Refined ROI group '{path}' missing {required - set(group.keys())}; skipping overrides.[/yellow]"
        )
        return None

    detection_indices = group["detection_indices"][:].astype(np.int64, copy=False)
    if detection_indices.size == 0:
        return None
    if detection_indices.min(initial=0) < 0 or detection_indices.max(initial=0) >= total_rois:
        raise ValueError("Refined ROI detection indices out of range for current crop run.")

    refined_rois = group["roi_images"][:]
    if refined_rois.shape[1:3] != roi_shape:
        raise ValueError(
            f"Refined ROI shape {refined_rois.shape[1:3]} does not match crop ROI shape {roi_shape}."
        )
    refined_coords = group["roi_coordinates_full"][:]

    override_map = np.full(total_rois, -1, dtype=np.int64)
    override_map[detection_indices] = np.arange(detection_indices.size, dtype=np.int64)

    frame_indices_override = (
        group["frame_indices"][:].astype(np.int64, copy=False)
        if "frame_indices" in group
        else None
    )

    decoder = (
        group.attrs.get("video_device")
        or group.attrs.get("refined_roi_decoder")
        or crop_group.attrs.get("refined_roi_decoder")
    )
    duration = (
        group.attrs.get("duration_seconds")
        or crop_group.attrs.get("refined_roi_generation_duration_seconds")
    )

    return {
        "path": path,
        "count": detection_indices.size,
        "indices": detection_indices,
        "map": override_map,
        "rois": refined_rois,
        "coords": refined_coords,
        "frame_indices": frame_indices_override,
        "decoder": decoder,
        "duration": duration,
    }


def _compute_heading(points: np.ndarray, pose_schema: PoseSchema) -> float:
    return compute_heading_from_spec(
        pose_schema.metadata.get("heading_computation"),
        labels=pose_schema.node_names,
        points=np.asarray(points, dtype=np.float64),
    )


def _repeat_to_rgb(batch: np.ndarray) -> List[np.ndarray]:
    if batch.ndim != 3:
        raise ValueError("ROI images should have shape (N, H, W)")
    return [np.repeat(img[..., None], 3, axis=2) for img in batch]


def _normalize_input_mode(value: str) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text not in _KEYPOINT_INPUT_MODES:
        choices = ", ".join(_KEYPOINT_INPUT_MODES)
        raise ValueError(f"Invalid keypoint input mode '{value}'. Expected one of: {choices}")
    return text


def _first_attr(attrs: Any, names: Tuple[str, ...]) -> Any:
    for name in names:
        value = attrs.get(name)
        if value is not None:
            return value
    return None


def _resolve_full_image_shape(
    root: zarr.Group,
    crop_group: zarr.Group,
    *,
    crop_run_id: str | None = None,
) -> Tuple[Tuple[int, int], Optional[int]]:
    """Resolve full-frame shape for normalized keypoint coordinates.

    Modern crop-first analysis zarrs may intentionally omit raw_video/images_full.
    In that case the geometry-only crop run is the authoritative source for the
    source-video dimensions used to compute roi_coordinates_full.
    """

    total_frames_attr = (
        root.attrs.get("total_frames")
        or root.attrs.get("n_frames")
        or crop_group.attrs.get("total_frames")
    )
    total_frames: Optional[int] = int(total_frames_attr) if total_frames_attr is not None else None

    try:
        images_full = root["raw_video/images_full"]
        frame_dim, img_h, img_w = images_full.shape
        if total_frames is None:
            total_frames = int(frame_dim)
        return (int(img_h), int(img_w)), total_frames
    except KeyError:
        pass

    if crop_run_id is not None:
        strict_dimensions = strict_crop_source_dimensions(
            crop_group,
            run_id=crop_run_id,
        )
        if strict_dimensions is not None:
            strict_frames, strict_height, strict_width = strict_dimensions
            if total_frames is not None and total_frames != strict_frames:
                raise ValueError(
                    "Root frame count differs from strict crop-v2 source-pixel "
                    "authority."
                )
            return (strict_height, strict_width), strict_frames

    width_names = (
        "video_width",
        "palette_video_width",
        "source_full_width",
        "source_video_width",
        "width",
    )
    height_names = (
        "video_height",
        "palette_video_height",
        "source_full_height",
        "source_video_height",
        "height",
    )
    img_w = _first_attr(root.attrs, width_names)
    img_h = _first_attr(root.attrs, height_names)
    if img_w is None:
        img_w = _first_attr(crop_group.attrs, width_names)
    if img_h is None:
        img_h = _first_attr(crop_group.attrs, height_names)

    if img_w is None or img_h is None:
        raise ValueError(
            "Unable to determine full-resolution image dimensions. "
            "Expected raw_video/images_full, root video_width/video_height attrs, "
            "strict crop-v2 source-pixel authority, or crop-run width/height attrs."
        )
    return (int(img_h), int(img_w)), total_frames


def _resolve_crop_run_frame_count_from_domains(root: zarr.Group, crop_group: zarr.Group) -> Optional[int]:
    try:
        return int(FrameDomains(root=root, run_group=crop_group).count(FrameDomain.RUN_FRAME))
    except FrameDomainError:
        return None


def _tensor_input_blocker(
    batch: np.ndarray,
    *,
    model_input_transform: ModelInputTransform,
    model_stride: int = 32,
) -> Optional[str]:
    if batch.ndim != 3:
        return f"expected ROI batch shape (N, H, W), got {batch.shape}"
    _, height, width = batch.shape
    if height != width:
        return f"tensor mode requires square ROIs, got {height}x{width}"
    if (height, width) != model_input_transform.native_shape:
        return (
            "tensor mode requires ROI dimensions to match model input transform native shape "
            f"{model_input_transform.native_shape}, got {height}x{width}"
        )
    model_height, model_width = model_input_transform.model_shape
    if model_height != model_width:
        return f"tensor mode requires square model inputs, got {model_height}x{model_width}"
    if model_height % model_stride or model_width % model_stride:
        return (
            "tensor mode requires model input dimensions divisible by model "
            f"stride {model_stride}, got {model_height}x{model_width}"
        )
    return None


def _prepare_model_inputs(
    batch: np.ndarray,
    *,
    input_mode: str,
    model_input_transform: ModelInputTransform,
    device: Optional[str],
    model_stride: int = 32,
) -> Tuple[object, str]:
    """Prepare one ROI batch for Ultralytics prediction.

    ``numpy-list`` preserves the legacy path. ``tensor`` avoids constructing a
    Python list of RGB numpy arrays and instead supplies normalized BCHW data.
    ``auto`` uses tensor mode only when the geometry is equivalent to the legacy
    path; otherwise it falls back to the list path.
    """
    mode = _normalize_input_mode(input_mode)
    blocker = _tensor_input_blocker(
        batch,
        model_input_transform=model_input_transform,
        model_stride=model_stride,
    )
    if mode == "tensor" and blocker is not None:
        raise ValueError(f"Cannot use keypoint tensor input mode: {blocker}")
    batch = model_input_transform.apply_numpy_luma_batch(batch)
    if mode == "numpy-list" or blocker is not None:
        return _repeat_to_rgb(batch), "numpy-list"

    if not batch.flags.c_contiguous or not batch.flags.writeable:
        batch = np.array(batch, copy=True, order="C")
    scale_by_255 = not np.issubdtype(batch.dtype, np.floating)
    if not scale_by_255 and batch.size:
        scale_by_255 = bool(np.nanmax(batch) > 1.0 + np.finfo(np.float32).eps)

    tensor = torch.from_numpy(batch)
    if device:
        tensor = tensor.to(torch.device(device), non_blocking=True)
    tensor = tensor.float()
    if scale_by_255:
        tensor = tensor.div_(255.0)
    return tensor[:, None, :, :].expand(-1, 3, -1, -1).contiguous(), "tensor"


def _resolve_effective_input_mode_contract(
    requested_mode: str,
    *,
    model_input_transform: ModelInputTransform,
    model_stride: int = 32,
) -> str:
    mode = _normalize_input_mode(requested_mode)
    native_height, native_width = model_input_transform.native_shape
    shape_probe = np.empty((1, native_height, native_width), dtype=np.uint8)
    blocker = _tensor_input_blocker(
        shape_probe,
        model_input_transform=model_input_transform,
        model_stride=model_stride,
    )
    if mode == "tensor" and blocker is not None:
        raise ValueError(f"Cannot use keypoint tensor input mode: {blocker}")
    return "numpy-list" if mode == "numpy-list" or blocker is not None else "tensor"


def _require_prepared_model_input_contract(
    model_inputs: object,
    *,
    effective_mode: str,
    expected_mode: str,
    batch_count: int,
    model_input_transform: ModelInputTransform,
) -> None:
    if effective_mode != expected_mode:
        raise ValueError(
            "Prepared keypoint input mode differs from the exact persisted "
            "pre-inference coordinate context."
        )
    height, width = model_input_transform.model_shape
    if effective_mode == "numpy-list":
        if not isinstance(model_inputs, list) or len(model_inputs) != batch_count:
            raise ValueError("Prepared numpy-list input has the wrong batch cardinality.")
        for item in model_inputs:
            array = np.asarray(item)
            if array.shape != (height, width, 3) or array.dtype != np.dtype("uint8"):
                raise ValueError(
                    "Prepared numpy-list pixels differ from the exact persisted "
                    "uint8 submitted extent."
                )
            if not (
                np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 0], array[..., 2])
            ):
                raise ValueError(
                    "Prepared numpy-list pixels violate luma-repeated RGB channel semantics."
                )
        return
    if not isinstance(model_inputs, torch.Tensor) or tuple(model_inputs.shape) != (
        batch_count,
        3,
        height,
        width,
    ):
        raise ValueError(
            "Prepared tensor pixels differ from the exact persisted submitted extent."
        )
    if model_inputs.dtype != torch.float32 or not bool(torch.isfinite(model_inputs).all()):
        raise ValueError("Prepared tensor input must be finite float32.")
    if model_inputs.numel() and (
        float(model_inputs.min()) < 0.0 or float(model_inputs.max()) > 1.0
    ):
        raise ValueError("Prepared tensor input must remain inside the closed [0,1] range.")
    if not (
        torch.equal(model_inputs[:, 0], model_inputs[:, 1])
        and torch.equal(model_inputs[:, 0], model_inputs[:, 2])
    ):
        raise ValueError("Prepared tensor input violates luma-repeated RGB channel semantics.")


def _require_model_result_coordinate_contract(
    results: Sequence[Any],
    *,
    batch_count: int,
    model_input_transform: ModelInputTransform,
) -> None:
    if len(results) != batch_count:
        raise ValueError(
            "Ultralytics result cardinality differs from the exact submitted "
            f"keypoint batch: expected {batch_count}, got {len(results)}."
        )
    expected_shape = tuple(int(item) for item in model_input_transform.model_shape)
    for index, result in enumerate(results):
        raw_shape = getattr(result, "orig_shape", None)
        if (
            not isinstance(raw_shape, (tuple, list))
            or len(raw_shape) != 2
            or any(type(item) is not int for item in raw_shape)
            or tuple(raw_shape) != expected_shape
        ):
            raise ValueError(
                "Ultralytics result orig_shape differs from the persisted submitted "
                f"model-input extent for batch row {index}: expected {expected_shape}, "
                f"got {raw_shape!r}."
            )


def _select_detection(result) -> Optional[int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes is False:
        return None
    if boxes.conf is None or boxes.conf.numel() == 0:
        return 0 if boxes.xyxy.shape[0] > 0 else None
    conf = boxes.conf.detach().cpu().numpy()
    return int(conf.argmax())


def _extract_keypoint_confidences(keypoints, det_idx: int, *, n_keypoints: int = 3) -> np.ndarray:
    """Extract per-keypoint confidences for one detection.

    Returns a float array of length ``n_keypoints``. Missing/conf-unavailable
    values are left as NaN.
    """
    out = np.full(n_keypoints, np.nan, dtype=np.float64)
    kp_conf = getattr(keypoints, "conf", None)
    if kp_conf is None:
        return out
    try:
        conf_np = kp_conf[det_idx].detach().cpu().numpy()
    except Exception:
        return out
    conf_flat = np.asarray(conf_np, dtype=np.float64).reshape(-1)
    if conf_flat.size == 0:
        return out
    take = min(n_keypoints, conf_flat.size)
    out[:take] = conf_flat[:take]
    return out


def _resolve_model_kpt_shape(model: YOLO) -> Optional[tuple[int, int]]:
    """Best-effort extraction of the Ultralytics model keypoint shape."""
    model_obj = getattr(model, "model", None)
    for raw in (
        getattr(model_obj, "kpt_shape", None),
        getattr(model_obj, "args", {}).get("kpt_shape")
        if isinstance(getattr(model_obj, "args", None), dict)
        else None,
        getattr(model_obj, "yaml", {}).get("kpt_shape")
        if isinstance(getattr(model_obj, "yaml", None), dict)
        else None,
    ):
        shape = normalize_kpt_shape(raw)
        if shape is not None:
            return shape
    return None


def _resolve_model_max_stride(model: YOLO) -> int:
    """Return the exact positive maximum stride declared by the loaded model."""

    model_obj = getattr(model, "model", None)
    raw_stride = getattr(model_obj, "stride", None)
    if raw_stride is None:
        raise ValueError("Loaded keypoint model does not declare an input stride.")
    if hasattr(raw_stride, "detach"):
        raw_stride = raw_stride.detach().cpu().numpy()
    values = np.asarray(raw_stride, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Loaded keypoint model declares an invalid input stride.")
    maximum = float(np.max(values))
    rounded = int(round(maximum))
    if not np.isclose(maximum, rounded, rtol=0.0, atol=1e-9):
        raise ValueError("Loaded keypoint model stride must be an exact integer.")
    return rounded


def _normalize_torch_device(device: Optional[str]) -> Optional[str]:
    """Accept Ultralytics-style GPU ids while passing a valid torch device string."""
    if device is None:
        return None
    text = str(device).strip()
    if not text:
        return None
    if text.isdigit():
        return f"cuda:{text}"
    return text


def _extract_pose_bbox_xyxy_roi(
    boxes,
    det_idx: int,
    *,
    roi_height: int,
    roi_width: int,
) -> np.ndarray:
    out = np.full(4, np.nan, dtype=np.float32)
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return out
    try:
        bbox = xyxy[det_idx].detach().cpu().numpy()
    except Exception:
        return out

    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size < 4:
        return out

    max_x = float(max(roi_width, 0))
    max_y = float(max(roi_height, 0))
    x0 = float(np.clip(bbox[0], 0.0, max_x))
    y0 = float(np.clip(bbox[1], 0.0, max_y))
    x1 = float(np.clip(bbox[2], 0.0, max_x))
    y1 = float(np.clip(bbox[3], 0.0, max_y))
    if roi_width > 0 and x1 <= x0:
        x1 = min(max_x, x0 + 1.0)
    if roi_height > 0 and y1 <= y0:
        y1 = min(max_y, y0 + 1.0)
    if x1 <= x0 or y1 <= y0:
        return out
    out[:] = (x0, y0, x1, y1)
    return out


def _clip_xyxy_to_roi(box_xyxy: np.ndarray, *, roi_height: int, roi_width: int) -> np.ndarray:
    out = np.asarray(box_xyxy, dtype=np.float32).reshape(-1).copy()
    if out.size < 4 or not np.all(np.isfinite(out[:4])):
        return np.full(4, np.nan, dtype=np.float32)
    max_x = float(max(roi_width, 0))
    max_y = float(max(roi_height, 0))
    out[0] = np.clip(out[0], 0.0, max_x)
    out[2] = np.clip(out[2], 0.0, max_x)
    out[1] = np.clip(out[1], 0.0, max_y)
    out[3] = np.clip(out[3], 0.0, max_y)
    if roi_width > 0 and out[2] <= out[0]:
        out[2] = min(max_x, out[0] + 1.0)
    if roi_height > 0 and out[3] <= out[1]:
        out[3] = min(max_y, out[1] + 1.0)
    if out[2] <= out[0] or out[3] <= out[1]:
        return np.full(4, np.nan, dtype=np.float32)
    return out[:4].astype(np.float32, copy=False)


@_fail_closed_keypoint_attempt
@proof_verification_operation
def detect_keypoints_yolo(
    zarr_path: str,
    model_path: str,
    *,
    model_sha256: Optional[str] = None,
    expected_model_stride: Optional[int] = None,
    run_name: Optional[str] = None,
    output_parent: str = DEFAULT_KEYPOINT_OUTPUT_PARENT,
    crop_run: Optional[str] = None,
    pose_schema: Optional[str] = None,
    model_pose_schema_binding: Optional[Mapping[str, Any] | str | Path] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    model_input_size: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 1,
    verbose: bool = False,
    mask_threshold: float = 0.5,
    roi_cache_policy: str = "auto",
    roi_cache_dir: Optional[Path] = None,
    roi_cache_manifest: Optional[Path] = None,
    roi_cache_expected_archive_path: Optional[Path] = None,
    roi_work_package_manifest: Optional[Path] = None,
    source_crop_row_start: Optional[int] = None,
    source_crop_row_stop: Optional[int] = None,
    roi_cache_source_tier: Optional[str] = None,
    roi_cache_staged_to_node_scratch: bool = False,
    roi_cache_staging_details: Optional[Dict[str, Any]] = None,
    roi_live_acceleration: str = "auto",
    roi_live_gpu_chunk_frames: int = 32,
    input_mode: str = "numpy-list",
    model_input_transform_mode: str = "auto",
    coordinate_contract_mode: str = "canonical",
    require_training_materialization_binding: bool = False,
    profile_timings: bool = False,
    progress_jsonl: Optional[Path] = None,
    progress_every_batches: int = 1,
    keypoint_roi_shard_rows: Optional[int] = DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    keypoint_frame_shard_rows: int = DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    registry: Optional[Path] = None,
    console: Optional[Console] = None,
    cli_provenance: Optional[Mapping[str, Any]] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
) -> str:
    """Run YOLO pose inference and record outputs in a keypoint run parent.

    Returns the name of the created run group.
    """

    console = console or Console()
    console.rule("[bold cyan]YOLO Pose Inference[/bold cyan]")

    output_parent_name = _normalize_keypoint_output_parent(output_parent)
    if coordinate_contract_mode not in _KEYPOINT_COORDINATE_CONTRACT_MODES:
        raise ValueError(
            "Unsupported coordinate_contract_mode "
            f"{coordinate_contract_mode!r}; expected one of "
            f"{_KEYPOINT_COORDINATE_CONTRACT_MODES}."
        )
    canonical_coordinates = coordinate_contract_mode == "canonical"
    if output_parent_name == DEFAULT_KEYPOINT_OUTPUT_PARENT and not canonical_coordinates:
        raise ValueError(
            "Final keypoints_runs are canonical-only. Explicit legacy_noncanonical "
            "output is permitted only for unbound keypoint_shard_runs."
        )
    if output_parent_name != DEFAULT_KEYPOINT_OUTPUT_PARENT and canonical_coordinates:
        raise ValueError(
            "Canonical coordinate publication is available only for final "
            "keypoints_runs. Collection shards must explicitly use "
            "coordinate_contract_mode='legacy_noncanonical' and cannot self-certify."
        )
    if canonical_coordinates and roi_cache_manifest is not None:
        raise ValueError(
            "Canonical base-keypoint inference requires direct persisted crop "
            "roi_images and rejects ROI cache manifests."
        )
    if canonical_coordinates and roi_work_package_manifest is not None:
        raise ValueError(
            "Canonical base-keypoint inference rejects crop work packages; finalize "
            "an exact materialized crop first."
        )
    if roi_cache_manifest is not None and roi_work_package_manifest is not None:
        raise ValueError(
            "roi_cache_manifest and roi_work_package_manifest are mutually exclusive."
        )
    if (source_crop_row_start is None) != (source_crop_row_stop is None):
        raise ValueError(
            "source_crop_row_start and source_crop_row_stop must be provided together."
        )
    if source_crop_row_start is not None:
        if roi_work_package_manifest is not None or roi_cache_manifest is not None:
            raise ValueError(
                "Direct crop-row partitions cannot be combined with a cache or work package."
            )
        if output_parent_name != "keypoint_shard_runs":
            raise ValueError(
                "Direct crop-row partitions must write keypoint_shard_runs outputs."
            )
    if (
        roi_cache_expected_archive_path is not None
        and roi_cache_manifest is None
    ):
        raise ValueError(
            "roi_cache_expected_archive_path requires roi_cache_manifest."
        )
    if (
        roi_work_package_manifest is not None
        and output_parent_name != "keypoint_shard_runs"
    ):
        raise ValueError(
            "Crop pixel work packages may write only to keypoint_shard_runs; "
            "finalize shards before publishing a canonical keypoint run."
        )
    if require_training_materialization_binding and canonical_coordinates:
        raise ValueError(
            "A training-materialized crop may feed only non-authoritative terminal "
            "keypoint arrays. Canonical keypoint-v2 publication must finalize against "
            "the bound source crop-v2 authority."
        )
    if require_training_materialization_binding and not crop_run:
        raise ValueError(
            "Strict training materialization input requires an explicit crop_run."
        )

    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")
    training_materialization = (
        bind_training_crop_materialization(zarr_path, run_id=str(crop_run))
        if require_training_materialization_binding
        else None
    )

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    model_artifact = fingerprint_artifact(
        model_path,
        role="keypoint_model",
        registry_hash=model_sha256,
    )

    if canonical_coordinates:
        if model_pose_schema_binding is None:
            raise ValueError(
                "Canonical keypoint inference requires an explicit digest-bound "
                "model_pose_schema_binding; model keypoint identity cannot be "
                "inferred from a default pose schema or keypoint cardinality."
            )
        raw_model_pose_schema_binding = (
            load_pose_model_schema_binding(model_pose_schema_binding)
            if isinstance(model_pose_schema_binding, (str, Path))
            else dict(model_pose_schema_binding)
        )
        model_digest = model_artifact.get("sha256")
        if not isinstance(model_digest, str):
            raise ValueError(
                "Canonical keypoint inference cannot bind pose semantics without "
                "an exact model content digest."
            )
        pose_schema_obj, pose_schema_attrs, validated_model_pose_schema_binding = (
            pose_schema_from_model_binding(
                raw_model_pose_schema_binding,
                expected_model_sha256=model_digest,
            )
        )
        model_artifact = {
            **model_artifact,
            "pose_schema_binding": validated_model_pose_schema_binding,
        }
        if pose_schema is not None:
            asserted_schema, asserted_attrs = schema_payload_from_package(pose_schema)
            if (
                asserted_attrs.get("skeleton_id") != pose_schema_attrs.get("skeleton_id")
                or asserted_attrs.get("keypoint_labels")
                != pose_schema_attrs.get("keypoint_labels")
                or undirected_edge_topology(asserted_attrs.get("edges"))
                != undirected_edge_topology(pose_schema_attrs.get("edges"))
            ):
                raise ValueError(
                    "Explicit pose_schema package disagrees with the digest-bound "
                    "model pose-schema binding."
                )
            # Retain the binding-derived object as the sole publication authority.
            del asserted_schema
    else:
        pose_schema_obj, pose_schema_attrs = schema_payload_from_package(
            pose_schema or DEFAULT_POSE_SCHEMA_NAME
        )

    if expected_model_stride is not None and (
        type(expected_model_stride) is not int or expected_model_stride <= 0
    ):
        raise ValueError("Expected model stride must be a positive exact integer.")

    model = YOLO(str(model_path))
    _revalidate_keypoint_model_artifact(
        model_path,
        model_artifact,
        checkpoint="after YOLO load and before inference",
    )
    torch_device = _normalize_torch_device(device)
    if torch_device:
        model.to(torch_device)
    try:
        model_device = str(next(model.model.parameters()).device)
    except (AttributeError, StopIteration):
        model_device = torch_device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path_resolved = model_path.resolve()
    n_keypoints = int(pose_schema_obj.num_keypoints)
    model_kpt_shape = _resolve_model_kpt_shape(model)
    model_stride = _resolve_model_max_stride(model)
    if (
        expected_model_stride is not None
        and model_stride != int(expected_model_stride)
    ):
        raise ValueError(
            "Loaded keypoint model stride differs from the planned preprocessing "
            f"contract: expected {int(expected_model_stride)}, got {model_stride}."
        )
    if canonical_coordinates and model_kpt_shape is None:
        raise ValueError(
            "Canonical keypoint inference requires an explicit model kpt_shape; "
            "the model architecture cannot be bound to the selected pose schema."
        )
    if model_kpt_shape is not None and int(model_kpt_shape[0]) != n_keypoints:
        raise ValueError(
            f"Model keypoint count {int(model_kpt_shape[0])} does not match "
            f"pose schema '{pose_schema_obj.name}' keypoint count {n_keypoints}."
        )

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    if roi_work_package_manifest is not None:
        crop_source = CropImageSource.open_work_package(
            root,
            manifest_path=roi_work_package_manifest,
            zarr_path=zarr_path,
            crop_run=crop_run,
        )
    else:
        crop_source = CropImageSource.open(
            root,
            crop_run=crop_run,
            # Canonical publication must retain the archive-root identity and
            # path of crop_runs/<run>/roi_images.  Opening crop_runs as its own
            # nested store erases that ownership even when bytes are identical.
            zarr_path=None if canonical_coordinates else zarr_path,
            roi_cache_policy=roi_cache_policy,
            roi_live_acceleration=roi_live_acceleration,
            roi_live_gpu_chunk_frames=roi_live_gpu_chunk_frames,
            roi_cache_dir=roi_cache_dir,
            roi_cache_manifest=roi_cache_manifest,
            roi_cache_expected_archive_path=roi_cache_expected_archive_path,
            source_crop_row_start=source_crop_row_start,
            source_crop_row_stop=source_crop_row_stop,
            console=console,
        )
    boundary = _ACTIVE_KEYPOINT_ATTEMPT.get()
    if boundary is not None:
        boundary.bind_crop_source(crop_source)
    crop_group = crop_source.crop_group
    latest_crop = crop_source.crop_run_name
    selected_crop_rows = getattr(crop_source, "source_crop_row_ids", None)
    if training_materialization is not None:
        if (
            latest_crop != training_materialization.run_id
            or int(crop_source.total_rois) != training_materialization.row_count
            or tuple(int(value) for value in crop_source.roi_shape)
            != training_materialization.roi_shape
            or getattr(crop_source, "frame_source_kind", None) != "roi_images"
            or bool(getattr(crop_source, "roi_cache_used", False))
        ):
            raise ValueError(
                "Active keypoint pixel source differs from the strict training "
                "crop materialization binding."
            )

    canonical_crop_source = None
    canonical_selected_rows: Optional[np.ndarray] = None
    if canonical_coordinates:
        canonical_crop_path = f"crop_runs/{latest_crop}"
        if getattr(crop_group, "path", None) != canonical_crop_path:
            raise ValueError(
                "Canonical keypoint inference requires the exact selected persisted "
                f"crop rowset at {canonical_crop_path!r}."
            )
        canonical_crop_source = load_persisted_keypoint_crop_source(
            root,
            canonical_crop_path,
        )
        if selected_crop_rows is not None:
            raise ValueError(
                "Canonical base-keypoint inference requires the direct complete "
                "materialized crop rowset and rejects selected-row proxy sources."
            )
        if (
            getattr(crop_source, "storage_mode", None) != "materialized"
            or getattr(crop_source, "frame_source_kind", None) != "roi_images"
            or getattr(crop_source, "roi_read_mode", None) != "materialized_crop_run"
            or bool(getattr(crop_source, "roi_cache_used", False))
        ):
            raise ValueError(
                "Canonical base-keypoint inference requires direct root-owned "
                "materialized roi_images; caches, live/composite pixels, and work "
                "packages are unsupported."
            )
        require_direct_keypoint_crop_pixel_source(
            canonical_crop_source,
            getattr(crop_source, "_roi_images", None),
        )
        canonical_selected_rows = (
            np.arange(
                canonical_crop_source.crop_geometry.row_identity.leading_dimension,
                dtype=np.int64,
            )
            if selected_crop_rows is None
            else np.asarray(selected_crop_rows, dtype=np.int64).reshape(-1)
        )

    roi_coords = crop_source.roi_coordinates_full.copy()
    frame_indices = crop_source.frame_indices.astype(np.int64, copy=True)
    total_rois = crop_source.total_rois
    if total_rois == 0:
        console.print("[yellow]No ROIs found in crop run; nothing to process[/yellow]")
        crop_source.close()
        return ""

    roi_h, roi_w = crop_source.roi_shape
    if canonical_crop_source is not None:
        assert canonical_selected_rows is not None
        source_placement = np.asarray(
            canonical_crop_source._placement_node[:]
        )[canonical_selected_rows]
        source_frames = np.asarray(
            canonical_crop_source._rowset_node["source_acquisition_frame_index"][:],
            dtype=np.int64,
        )[canonical_selected_rows]
        expected_roi_shape = (
            int(canonical_crop_source.roi_frame.endpoint.height),
            int(canonical_crop_source.roi_frame.endpoint.width),
        )
        if (
            total_rois != int(canonical_selected_rows.shape[0])
            or (roi_h, roi_w) != expected_roi_shape
            or roi_coords.shape != source_placement[:, :2].shape
            or not np.array_equal(roi_coords, source_placement[:, :2])
            or not np.array_equal(frame_indices, source_frames)
        ):
            raise ValueError(
                "Active crop pixels/rows do not equal the exact persisted canonical "
                "crop selection, placement, ROI extent, and acquisition time mapping."
            )
        roi_coords = np.array(source_placement[:, :2], copy=True)
        frame_indices = np.array(source_frames, copy=True)
    source_detect_run = crop_group.attrs.get("source_detect_run")
    source_refined_run = crop_group.attrs.get("source_refined_run")
    override_data = (
        None
        if roi_work_package_manifest is not None
        else _prepare_refined_roi_overrides(
            root, crop_group, total_rois, (roi_h, roi_w), console
        )
    )
    override_map: Optional[np.ndarray] = None
    override_rois: Optional[np.ndarray] = None
    if override_data is not None:
        if canonical_coordinates:
            raise ValueError(
                "Refined ROI overrides are not supported by canonical base-keypoint "
                "publication; publish a new canonical crop with exact placement lineage."
            )
        indices = override_data["indices"]
        roi_coords[indices] = override_data["coords"]
        frame_override = override_data["frame_indices"]
        if frame_override is not None:
            frame_indices[indices] = frame_override.astype(frame_indices.dtype, copy=False)
        override_map = override_data["map"]
        override_rois = override_data["rois"]
        console.print(
            f"[cyan]Applying refined ROI overrides:[/cyan] {override_data['count']} detections"
        )

    imgsz = imgsz or max(roi_h, roi_w)
    submitted_model_input_size = (
        int(imgsz) if model_input_size is None else int(model_input_size)
    )
    if int(imgsz) <= 0 or submitted_model_input_size <= 0:
        raise ValueError("Model network and submitted input sizes must be positive.")
    model_input_transform = resolve_model_input_transform(
        (roi_h, roi_w),
        mode=model_input_transform_mode,
        model_hw=(submitted_model_input_size, submitted_model_input_size),
    )
    resolved_input_mode = _normalize_input_mode(input_mode)
    contracted_effective_input_mode = _resolve_effective_input_mode_contract(
        resolved_input_mode,
        model_input_transform=model_input_transform,
        model_stride=model_stride,
    )

    run_parent, run_group, resolved_run_name = _prepare_run_group_for_parent(
        root,
        run_name,
        console,
        output_parent=output_parent_name,
    )
    run_group.attrs["output_parent"] = output_parent_name
    run_group.attrs["run_group_parent"] = output_parent_name
    # Every child remains ineligible through complete-path revalidation. Normal
    # canonical runs are activated only after selectors have been written;
    # collection shards remain permanently ineligible.
    run_group.attrs["stage_selector_eligible"] = False
    if output_parent_name != DEFAULT_KEYPOINT_OUTPUT_PARENT:
        run_group.attrs["is_collection_shard"] = True
    run_group.attrs["keypoint_labels"] = list(pose_schema_obj.node_names)
    run_group.attrs["keypoint_confidence_labels"] = list(pose_schema_obj.node_names)
    run_group.attrs["skeleton_id"] = str(pose_schema_attrs["skeleton_id"])
    run_group.attrs["kpt_shape"] = list(pose_schema_attrs["kpt_shape"])
    run_group.attrs["pose_schema"] = dict(pose_schema_attrs)
    if model_kpt_shape is not None:
        run_group.attrs["model_kpt_shape"] = list(model_kpt_shape)
    arrays = _create_output_arrays(
        run_group,
        total_rois,
        chunk_hint=batch_size * 4,
        n_keypoints=n_keypoints,
        shard_rows=keypoint_roi_shard_rows,
    )

    if selected_crop_rows is not None:
        lineage_result = copy_selected_crop_row_lineage_arrays(
            run_group,
            crop_group,
            selected_crop_rows,
            use_geometry_preload_profile=True,
            shard_rows=keypoint_roi_shard_rows,
            count_shard_rows=(
                keypoint_frame_shard_rows
                if keypoint_roi_shard_rows is not None
                else None
            ),
        )
    else:
        lineage_result = copy_row_lineage_arrays(
            run_group,
            crop_group,
            total_rois=total_rois,
            use_geometry_preload_profile=True,
            shard_rows=keypoint_roi_shard_rows,
            count_shard_rows=(
                keypoint_frame_shard_rows
                if keypoint_roi_shard_rows is not None
                else None
            ),
        )
    if canonical_coordinates:
        # ``source_crop_row_ids`` on this output maps keypoint rows into the
        # selected crop rowset.  A same-named array already present on that crop
        # describes the crop's own upstream lineage and must never be inherited
        # as though it were this new relationship.  Canonical direct inference
        # consumes the complete crop in physical row order, so its exact mapping
        # is always 0..N-1.
        write_direct_source_crop_row_ids(
            run_group,
            total_rois=total_rois,
            overwrite=True,
            shard_rows=keypoint_roi_shard_rows,
        )
    elif "source_crop_row_ids" not in lineage_result.copied:
        write_direct_source_crop_row_ids(
            run_group,
            total_rois=total_rois,
            shard_rows=keypoint_roi_shard_rows,
        )
    keypoint_coordinate_context = None
    if canonical_crop_source is not None:
        assert canonical_selected_rows is not None
        row_chunks = arrays["heading"].chunks
        placement_chunks = (int(row_chunks[0]), 4)
        run_group.create_array(
            "source_acquisition_frame_index",
            data=np.asarray(source_frames, dtype="<i8"),
            chunks=row_chunks,
            overwrite=True,
            **_shard_kwargs(row_chunks, keypoint_roi_shard_rows),
        )
        run_group.create_array(
            "source_crop_xywh",
            data=np.asarray(source_placement),
            chunks=placement_chunks,
            overwrite=True,
            **_shard_kwargs(placement_chunks, keypoint_roi_shard_rows),
        )
        keypoint_coordinate_context = prepare_keypoint_coordinate_context(
            root,
            f"keypoints_runs/{resolved_run_name}",
            crop_path=f"crop_runs/{latest_crop}",
            model_input_transform=model_input_transform,
            preprocessing_input_mode=contracted_effective_input_mode,
            model_artifact=model_artifact,
        )
        # The preflight intentionally reloads through the archive root.  Rebind
        # here so a stale Zarr attrs handle cannot overwrite its new evidence.
        run_group = root[f"keypoints_runs/{resolved_run_name}"]
    if getattr(crop_source, "pixel_materialization_id", None) is not None:
        if selected_crop_rows is None:
            raise ValueError("Package-backed keypoint inference lacks selected crop rows.")
        copy_selected_row_source_signatures(
            run_group,
            crop_group,
            selected_crop_rows,
            shard_rows=int(keypoint_roi_shard_rows or DEFAULT_KEYPOINT_ROI_SHARD_ROWS),
            root=root,
        )
    if "detection_indices" not in lineage_result.copied:
        console.print("[yellow]Crop run missing 'detection_indices'; YOLO keypoint run will omit them.[/yellow]")

    if keypoint_coordinate_context is not None:
        camera_frame = (
            keypoint_coordinate_context.source.crop_geometry.source_geometry.frame_evidence.source_camera_frame
        )
        full_img_shape = (
            int(camera_frame.endpoint.height),
            int(camera_frame.endpoint.width),
        )
        total_frames = int(
            keypoint_coordinate_context.source.crop_geometry.source_geometry.frame_evidence.acquisition_frame.record.source_total_frames
        )
    else:
        full_img_shape, total_frames = _resolve_full_image_shape(
            root,
            crop_group,
            crop_run_id=latest_crop,
        )

    norm_factor = np.array([full_img_shape[1], full_img_shape[0]], dtype="f8")

    if total_frames is None:
        total_frames = _resolve_crop_run_frame_count_from_domains(root, crop_group)

    if total_frames is None:
        total_frames = int(frame_indices.max() + 1) if frame_indices.size > 0 else 0

    if "frame_counts" in lineage_result.copied:
        frame_counts_total = run_group["frame_counts"][:].astype("i4", copy=False)
    else:
        frame_counts_total = (
            np.bincount(frame_indices, minlength=total_frames).astype("i4", copy=False)
            if frame_indices.size > 0
            else np.zeros(total_frames, dtype="i4")
        )
    count_chunks = (min(len(frame_counts_total), batch_size * 4),) if frame_counts_total.size > 0 else None
    run_group.create_array(
        "n_rois",
        data=frame_counts_total,
        chunks=count_chunks,
        overwrite=True,
        **_shard_kwargs(
            count_chunks or (max(1, len(frame_counts_total)),),
            keypoint_frame_shard_rows if keypoint_roi_shard_rows is not None else None,
        ),
    )
    if "frame_counts" not in lineage_result.copied:
        run_group.create_array(
            "frame_counts",
            data=frame_counts_total,
            chunks=count_chunks,
            overwrite=True,
            **_shard_kwargs(
                count_chunks or (max(1, len(frame_counts_total)),),
                keypoint_frame_shard_rows if keypoint_roi_shard_rows is not None else None,
            ),
        )

    crop_detection_source = crop_group.get("detection_source")
    crop_source_row_count = (
        int(total_rois)
        if selected_crop_rows is None
        else int(crop_group["frame_indices"].shape[0])
    )
    if (
        crop_detection_source is not None
        and crop_detection_source.shape[0] != crop_source_row_count
    ):
        raise ValueError(
            "Crop run detection_source length "
            f"{crop_detection_source.shape[0]} does not match crop row count "
            f"{crop_source_row_count}"
        )
    scalar_chunk = arrays["heading"].chunks
    detection_source_dst = run_group.create_array(
        "detection_source",
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype="i1",
        fill_value=0,
        overwrite=True,
        **_shard_kwargs(scalar_chunk, keypoint_roi_shard_rows),
    )

    success_total = 0
    confidence_accum: List[float] = []
    timing_profiler = InferenceTimingProfiler(enabled=profile_timings)
    shard_writer = None
    if keypoint_roi_shard_rows is not None:
        shard_writer = _AlignedKeypointShardWriter(
            {**arrays, "detection_source": detection_source_dst},
            shard_rows=int(keypoint_roi_shard_rows),
            buffer_count=2,
        )
        if boundary is not None:
            boundary.bind_shard_writer(shard_writer)
    progress_jsonl_path = Path(progress_jsonl).expanduser() if progress_jsonl is not None else None
    progress_interval = max(1, int(progress_every_batches))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    start_time = time.time()
    effective_model_input_transform = (
        keypoint_coordinate_context.model_input_transform
        if keypoint_coordinate_context is not None
        else model_input_transform
    )
    effective_input_mode = contracted_effective_input_mode
    _write_keypoint_progress_jsonl(
        progress_jsonl_path,
        "start",
        zarr_path=str(zarr_path.resolve()),
        run_name=resolved_run_name,
        crop_run=latest_crop,
        total_rois=int(total_rois),
        batch_size=int(batch_size),
        pose_schema=pose_schema_obj.name,
        model_path=str(model_path_resolved),
        requested_device=device,
        normalized_torch_device=torch_device,
        initial_model_device=model_device,
        roi_read_mode=crop_source.roi_read_mode,
        roi_cache_policy=crop_source.roi_cache_policy,
        source_roi_cache_used=bool(crop_source.roi_cache_used),
        frame_source=crop_source.frame_source_kind,
        source_video_path=crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
    )

    with progress:
        task = progress.add_task("[cyan]Predicting keypoints...", total=total_rois)
        for batch_index, start in enumerate(range(0, total_rois, batch_size), start=1):
            batch_started = time.perf_counter()
            end = min(start + batch_size, total_rois)
            batch_coords = roi_coords[start:end]
            batch_count = end - start
            if keypoint_coordinate_context is not None:
                revalidate_keypoint_coordinate_batch_context(
                    keypoint_coordinate_context,
                    row_start=start,
                    row_stop=end,
                )
            with timing_profiler.time("roi_read", items=batch_count):
                batch_roi_np = crop_source.read_slice(start, end)
            if override_map is not None and override_rois is not None:
                with timing_profiler.time("roi_override_apply", items=batch_count):
                    local_map = override_map[start:end]
                    valid = local_map >= 0
                    if np.any(valid):
                        batch_roi_np[valid] = override_rois[local_map[valid]]

            with timing_profiler.time("input_prepare", items=batch_count):
                model_inputs, effective_input_mode = _prepare_model_inputs(
                    batch_roi_np,
                    input_mode=resolved_input_mode,
                    model_input_transform=effective_model_input_transform,
                    device=torch_device,
                    model_stride=model_stride,
                )
                _require_prepared_model_input_contract(
                    model_inputs,
                    effective_mode=effective_input_mode,
                    expected_mode=contracted_effective_input_mode,
                    batch_count=batch_count,
                    model_input_transform=effective_model_input_transform,
                )
            with timing_profiler.time("model_predict", items=batch_count):
                results = tuple(
                    model.predict(
                        model_inputs,
                        imgsz=imgsz,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        rect=False,
                        device=torch_device,
                        verbose=verbose,
                        stream=True,
                    )
                )
                if keypoint_coordinate_context is not None:
                    _require_model_result_coordinate_contract(
                        results,
                        batch_count=batch_count,
                        model_input_transform=effective_model_input_transform,
                    )

            batch_keypoints_roi = np.full((batch_count, n_keypoints, 2), np.nan, dtype=np.float64)
            batch_keypoints_img = np.full_like(batch_keypoints_roi, np.nan)
            batch_keypoints_norm = np.full_like(batch_keypoints_roi, np.nan)
            batch_heading = np.full(batch_count, np.nan, dtype=np.float64)
            batch_conf = np.full(batch_count, np.nan, dtype=np.float64)
            batch_keypoint_conf = np.full((batch_count, n_keypoints), np.nan, dtype=np.float64)
            batch_success = np.zeros(batch_count, dtype=bool)
            batch_failure_codes = np.full(
                batch_count,
                np.uint8(
                    PoseInferenceFailureCode.NO_POSE_DETECTION_ABOVE_THRESHOLD
                ),
                dtype=np.uint8,
            )
            batch_pose_bbox_roi = np.full((batch_count, 4), np.nan, dtype=np.float32)
            batch_pose_bbox_img = np.full_like(batch_pose_bbox_roi, np.nan)
            batch_pose_bbox_norm = np.full_like(batch_pose_bbox_roi, np.nan)

            with timing_profiler.time("result_decode", items=batch_count):
                for i, (res, top_left) in enumerate(zip(results, batch_coords)):
                    det_idx = _select_detection(res)
                    if det_idx is None:
                        continue
                    keypoints = getattr(res, "keypoints", None)
                    kp_xy = getattr(keypoints, "xy", None)
                    if kp_xy is None:
                        batch_failure_codes[i] = np.uint8(
                            PoseInferenceFailureCode.KEYPOINT_PAYLOAD_MISSING
                        )
                        continue
                    if kp_xy.ndim != 3 or kp_xy.shape[0] == 0:
                        batch_failure_codes[i] = np.uint8(
                            PoseInferenceFailureCode.KEYPOINT_PAYLOAD_EMPTY
                        )
                        continue

                    kp = kp_xy[det_idx].detach().cpu().numpy()
                    if kp.shape[0] != n_keypoints:
                        if keypoint_coordinate_context is not None:
                            raise ValueError(
                                "Canonical model result keypoint count "
                                f"{kp.shape[0]} does not match the bound pose-schema "
                                f"keypoint count {n_keypoints}."
                            )
                        if kp.shape[0] < n_keypoints:
                            batch_failure_codes[i] = np.uint8(
                                PoseInferenceFailureCode.INSUFFICIENT_KEYPOINT_COUNT
                            )
                            continue
                        kp = kp[:n_keypoints]
                    if keypoint_coordinate_context is not None:
                        kp = model_input_batch_to_roi(
                            np.asarray(kp, dtype=np.float64),
                            context=keypoint_coordinate_context,
                            output_dtype=np.float64,
                        )
                    else:
                        kp = model_input_transform.invert_points_xy(kp)

                    kp[:, 0] = np.clip(kp[:, 0], 0.0, roi_w - 1)
                    kp[:, 1] = np.clip(kp[:, 1], 0.0, roi_h - 1)

                    batch_keypoints_roi[i] = kp
                    if keypoint_coordinate_context is None:
                        top_left = np.asarray(top_left, dtype=np.float64)
                        kp_img = kp + np.array([top_left[0], top_left[1]])
                        batch_keypoints_img[i] = kp_img
                        batch_keypoints_norm[i] = kp_img / norm_factor
                    batch_heading[i] = _compute_heading(kp, pose_schema_obj)
                    batch_keypoint_conf[i] = _extract_keypoint_confidences(
                        keypoints,
                        det_idx,
                        n_keypoints=n_keypoints,
                    )

                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and boxes.conf is not None and boxes.conf.numel() > 0:
                        det_conf = float(boxes.conf[det_idx].detach().cpu())
                    else:
                        kp_conf = getattr(keypoints, "conf", None)
                        det_conf = float(kp_conf[det_idx].detach().cpu().mean()) if kp_conf is not None else 0.0
                    if boxes is not None:
                        pose_bbox_model = _extract_pose_bbox_xyxy_roi(
                            boxes,
                            det_idx,
                            roi_height=effective_model_input_transform.model_height,
                            roi_width=effective_model_input_transform.model_width,
                        )
                        if np.all(np.isfinite(pose_bbox_model)):
                            inverted_bbox = (
                                model_input_bbox_batch_to_roi(
                                    pose_bbox_model,
                                    context=keypoint_coordinate_context,
                                    output_dtype=np.float32,
                                )
                                if keypoint_coordinate_context is not None
                                else model_input_transform.invert_boxes_xyxy(
                                    pose_bbox_model
                                )
                            )
                            batch_pose_bbox_roi[i] = _clip_xyxy_to_roi(
                                inverted_bbox,
                                roi_height=roi_h,
                                roi_width=roi_w,
                            )

                    batch_conf[i] = det_conf
                    batch_success[i] = True
                    batch_failure_codes[i] = np.uint8(
                        PoseInferenceFailureCode.NONE
                    )
                    success_total += 1
                    confidence_accum.append(det_conf)

            if keypoint_coordinate_context is not None:
                derived_coordinates = derive_keypoint_coordinate_batch(
                    context=keypoint_coordinate_context,
                    row_start=start,
                    row_stop=end,
                    keypoints_roi=batch_keypoints_roi,
                    pose_bbox_xyxy_roi=batch_pose_bbox_roi,
                )
                batch_keypoints_img = derived_coordinates["keypoints_img"]
                batch_keypoints_norm = derived_coordinates["keypoints_norm"]
                batch_pose_bbox_img = derived_coordinates["pose_bbox_xyxy_img"]
                batch_pose_bbox_norm = derived_coordinates["pose_bbox_xyxy_norm"]
            else:
                bbox_offsets = np.column_stack(
                    (
                        batch_coords[:, 0],
                        batch_coords[:, 1],
                        batch_coords[:, 0],
                        batch_coords[:, 1],
                    )
                ).astype(np.float32, copy=False)
                batch_pose_bbox_img = np.asarray(
                    batch_pose_bbox_roi + bbox_offsets,
                    dtype=np.float32,
                )
                batch_pose_bbox_norm = np.asarray(
                    batch_pose_bbox_img
                    / np.tile(norm_factor, 2),
                    dtype=np.float32,
                )

            with timing_profiler.time("output_write", items=batch_count):
                if crop_detection_source is not None:
                    if selected_crop_rows is None:
                        source_chunk = crop_detection_source[start:end].astype(
                            "i1", copy=False
                        )
                    else:
                        source_chunk = np.asarray(
                            crop_detection_source[
                                selected_crop_rows[start:end]
                            ],
                            dtype="i1",
                        )
                else:
                    source_chunk = np.zeros(end - start, dtype="i1")
                heading_finite_chunk = np.isfinite(batch_heading)
                heading_usable_chunk = np.logical_and(
                    np.logical_and(batch_success, source_chunk == 0),
                    heading_finite_chunk,
                )
                batch_outputs = {
                    "keypoints_roi": batch_keypoints_roi,
                    "keypoints_img": batch_keypoints_img,
                    "keypoints_norm": batch_keypoints_norm,
                    "heading": batch_heading,
                    "confidence": batch_conf,
                    "keypoint_confidences": batch_keypoint_conf,
                    "detection_success": batch_success,
                    "pose_failure_codes": batch_failure_codes,
                    "pose_bbox_xyxy_roi": batch_pose_bbox_roi,
                    "pose_bbox_xyxy_img": batch_pose_bbox_img,
                    "pose_bbox_xyxy_norm": batch_pose_bbox_norm,
                    "effective_threshold": np.full(batch_count, np.nan, dtype=np.float64),
                    "effective_se2_radius": np.full(batch_count, np.nan, dtype=np.float64),
                    "heading_finite": heading_finite_chunk,
                    "heading_usable": heading_usable_chunk,
                    "detection_source": source_chunk,
                }
                if shard_writer is not None:
                    shard_writer.write(start, batch_outputs)
                else:
                    for name, values in batch_outputs.items():
                        destination = (
                            detection_source_dst if name == "detection_source" else arrays[name]
                        )
                        destination[start:end, ...] = values

            progress.update(task, advance=end - start)
            if batch_index % progress_interval == 0 or end >= total_rois:
                batch_seconds = time.perf_counter() - batch_started
                elapsed_seconds = time.time() - start_time
                _write_keypoint_progress_jsonl(
                    progress_jsonl_path,
                    "batch_complete",
                    zarr_path=str(zarr_path.resolve()),
                    run_name=resolved_run_name,
                    crop_run=latest_crop,
                    batch_index=int(batch_index),
                    row_start=int(start),
                    row_stop=int(end),
                    batch_rows=int(batch_count),
                    rows_written=int(end),
                    total_rois=int(total_rois),
                    percent_complete=float((end / total_rois) * 100.0) if total_rois else 100.0,
                    batch_successful=int(np.count_nonzero(batch_success)),
                    successful_so_far=int(success_total),
                    failed_so_far=int(end - success_total),
                    batch_seconds=float(batch_seconds),
                    elapsed_seconds=float(elapsed_seconds),
                    batch_rois_per_second=float(batch_count / batch_seconds) if batch_seconds > 0 else None,
                    cumulative_rois_per_second=float(end / elapsed_seconds) if elapsed_seconds > 0 else None,
                    input_mode_requested=resolved_input_mode,
                    input_mode_effective=effective_input_mode,
                )

    shard_write_summary = shard_writer.finish() if shard_writer is not None else None
    total_time = time.time() - start_time
    inference_rate = success_total / total_time if total_time > 0 else 0.0
    resolved_model_device = _current_model_device(model, fallback=model_device)

    success_rate = (success_total / total_rois * 100.0) if total_rois > 0 else 0.0
    failure_total = total_rois - success_total
    persisted_pose_success = np.asarray(arrays["detection_success"][:], dtype=bool)
    persisted_failure_codes = np.asarray(
        arrays["pose_failure_codes"][:], dtype=np.uint8
    )
    validate_pose_inference_failure_codes(
        persisted_failure_codes,
        pose_success=persisted_pose_success,
    )
    failure_code_histogram = pose_inference_failure_histogram(
        persisted_failure_codes
    )

    success_counts = build_frame_keypoint_counts(
        frame_indices,
        persisted_pose_success,
        frame_axis_len=int(frame_counts_total.shape[0]),
        keypoint_count=n_keypoints,
    )
    success_chunks = (min(len(success_counts), batch_size * 4),) if success_counts.size > 0 else None
    run_group.create_array(
        "n_keypoints",
        data=success_counts,
        chunks=success_chunks,
        overwrite=True,
        **_shard_kwargs(
            success_chunks or (max(1, len(success_counts)),),
            keypoint_frame_shard_rows if keypoint_roi_shard_rows is not None else None,
        ),
    )

    git_info = get_git_info()
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    scheduler_info = _scheduler_payload(env_info)
    crop_snapshot_attrs = build_source_crop_snapshot_attrs(
        crop_group.attrs,
        source_crop_storage_mode=crop_source.storage_mode,
    )
    crop_pixel_attrs = build_source_roi_pixel_attrs(crop_source)
    effective_roi_cache_source_tier = roi_cache_source_tier or _infer_roi_cache_source_tier(crop_source.roi_cache_path)
    roi_cache_staging_payload = dict(roi_cache_staging_details) if isinstance(roi_cache_staging_details, dict) else None
    roi_cache_staging_policy = (
        roi_cache_staging_payload.get("policy")
        if isinstance(roi_cache_staging_payload, dict)
        else None
    )
    work_package_attrs = {
        "source_crop_pixel_work_package_id": getattr(
            crop_source, "pixel_materialization_id", None
        ),
        "source_crop_pixel_work_package_manifest": (
            getattr(crop_source, "pixel_materialization_manifest", None)
        ),
        "source_crop_pixel_work_package_rows": (
            int(total_rois)
            if getattr(crop_source, "pixel_materialization_id", None) is not None
            else None
        ),
    }
    if getattr(crop_source, "pixel_materialization_id", None) is not None:
        work_package_attrs.update(
            {
                "incremental_materialization_role": "delta_replacement_rows",
                "canonical_finalization_policy": "incremental_compaction_required",
            }
        )

    model_binding_attrs: dict[str, Any] = {}
    if canonical_coordinates:
        binding_model = validated_model_pose_schema_binding["model"]
        binding_authority = validated_model_pose_schema_binding["authority"]
        model_binding_attrs = {
            "model_pose_schema_binding_schema_id": (
                validated_model_pose_schema_binding["schema_id"]
            ),
            "model_pose_schema_binding_sha256": (
                validated_model_pose_schema_binding["binding_sha256"]
            ),
            "model_pose_schema_binding_kind": (
                validated_model_pose_schema_binding["binding_kind"]
            ),
            "model_sha256": binding_model["sha256"],
            "model_training_manifest_sha256": binding_authority[
                "training_manifest_sha256"
            ],
        }
        if binding_model["registry_run_id"] is not None:
            model_binding_attrs.update(
                {
                    "model_resolution_mode": "registry",
                    "model_resolution_task": "pose",
                    "model_resolution_selected_run_id": binding_model[
                        "registry_run_id"
                    ],
                    "model_resolution_selected_set_id": binding_model[
                        "registry_set_id"
                    ],
                    "model_resolution_selected_model_path": str(
                        model_path_resolved
                    ),
                }
            )

    run_group.attrs.update({
        "method": "yolo_pose",
        "coordinate_contract_mode": coordinate_contract_mode,
        "keypoints_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path_resolved),
        "model_name": model_path.name,
        **model_binding_attrs,
        "ultralytics_version": ultralytics_version,
        "device": resolved_model_device,
        "requested_device": device,
        "normalized_torch_device": torch_device,
        "initial_model_device": model_device,
        "resolved_model_device": resolved_model_device,
        "execution_hostname": scheduler_info.get("execution_hostname"),
        "scheduler": scheduler_info.get("scheduler"),
        "scheduler_job_id": scheduler_info.get("job_id"),
        "scheduler_job_name": scheduler_info.get("job_name"),
        "scheduler_job_index": scheduler_info.get("job_index"),
        "scheduler_queue": scheduler_info.get("queue"),
        "scheduler_hosts": scheduler_info.get("hosts") or scheduler_info.get("node_list"),
        "scheduler_mcpu_hosts": scheduler_info.get("mcpu_hosts"),
        "scheduler_cuda_visible_devices": scheduler_info.get("cuda_visible_devices"),
        "scheduler_gpu_request": scheduler_info.get("gpu_request"),
        "source_crop_run": latest_crop,
        **crop_snapshot_attrs,
        **crop_pixel_attrs,
        **work_package_attrs,
        "source_roi_read_mode": crop_source.roi_read_mode,
        "roi_cache_policy": crop_source.roi_cache_policy,
        "source_roi_cache_used": bool(crop_source.roi_cache_used),
        "source_roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
        "source_roi_cache_source_tier": effective_roi_cache_source_tier,
        "source_roi_cache_staged_to_node_scratch": bool(roi_cache_staged_to_node_scratch),
        "source_roi_cache_staging_policy": roi_cache_staging_policy,
        "source_roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
        "source_roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
        "source_roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
        "source_roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
        "source_detect_run": source_detect_run or "unknown",
        "artifact_mutability": "raw_immutable",
        "keypoints_processed": total_rois,
        "success_rate": round(success_rate, 2),
        "pose_failure_code_contract": {
            "schema_id": POSE_INFERENCE_FAILURE_SCHEMA_ID,
            "schema_version": POSE_INFERENCE_FAILURE_SCHEMA_VERSION,
            "array_path": "pose_failure_codes",
            "dtype": "uint8",
            "code_map": pose_inference_failure_code_map_json(),
            "success_alignment": "code_zero_iff_detection_success_true",
        },
        "pose_failure_code_histogram": failure_code_histogram,
        "keypoint_storage_layout": (
            "indexed_sharding_v1" if keypoint_roi_shard_rows is not None else "regular_chunks_v1"
        ),
        "keypoint_storage_policy": (
            "default_indexed_sharding_v1"
            if keypoint_roi_shard_rows is not None
            else "explicit_regular_chunks_override"
        ),
        "keypoint_roi_shard_rows": (
            int(keypoint_roi_shard_rows) if keypoint_roi_shard_rows is not None else None
        ),
        "keypoint_frame_shard_rows": (
            int(keypoint_frame_shard_rows) if keypoint_roi_shard_rows is not None else None
        ),
        "keypoint_shard_write": shard_write_summary,
        "parameters": {
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "max_det": max_det,
            "imgsz": imgsz,
            "model_input_size": submitted_model_input_size,
            "model_predict_rect": False,
            "batch_size": batch_size,
            "device": resolved_model_device,
            "requested_device": device,
            "normalized_torch_device": torch_device,
            "initial_model_device": model_device,
            "resolved_model_device": resolved_model_device,
            "profile_timings": bool(profile_timings),
            "progress_jsonl": str(progress_jsonl_path) if progress_jsonl_path is not None else None,
            "progress_every_batches": int(progress_interval),
            "pose_schema": pose_schema_obj.name,
            "n_keypoints": int(n_keypoints),
            "model_kpt_shape": list(model_kpt_shape) if model_kpt_shape is not None else None,
            "roi_live_acceleration": str(roi_live_acceleration),
            "roi_live_gpu_chunk_frames": int(roi_live_gpu_chunk_frames),
            "roi_cache_manifest": str(roi_cache_manifest) if roi_cache_manifest is not None else None,
            "roi_work_package_manifest": (
                str(roi_work_package_manifest)
                if roi_work_package_manifest is not None
                else None
            ),
            "roi_cache_source_tier": effective_roi_cache_source_tier,
            "roi_cache_staged_to_node_scratch": bool(roi_cache_staged_to_node_scratch),
            "roi_cache_staging_policy": roi_cache_staging_policy,
            "input_mode_requested": resolved_input_mode,
            "input_mode_effective": effective_input_mode,
            "model_input_transform": model_input_transform.to_attrs(),
            "model_input_stride": int(model_stride),
            "expected_model_stride": expected_model_stride,
            "coordinate_contract_mode": coordinate_contract_mode,
            # Maintained for API compatibility with pipeline/batch configs.
            "mask_threshold": float(mask_threshold),
            "keypoint_roi_shard_rows": (
                int(keypoint_roi_shard_rows) if keypoint_roi_shard_rows is not None else None
            ),
            "keypoint_frame_shard_rows": (
                int(keypoint_frame_shard_rows) if keypoint_roi_shard_rows is not None else None
            ),
            "keypoint_storage_layout": (
                "indexed_sharding_v1"
                if keypoint_roi_shard_rows is not None
                else "regular_chunks_v1"
            ),
            "keypoint_storage_policy": (
                "default_indexed_sharding_v1"
                if keypoint_roi_shard_rows is not None
                else "explicit_regular_chunks_override"
            ),
        },
        "model_names": getattr(model.model, "names", None),
        "summary_statistics": {
            "total_rois": int(total_rois),
            "successful_detections": int(success_total),
            "failed_detections": int(failure_total),
            "success_rate_percent": round(success_rate, 2),
            "mean_confidence": float(np.mean(confidence_accum)) if confidence_accum else 0.0,
            "pose_failure_code_histogram": failure_code_histogram,
        },
        "git_commit": git_info.get("commit_hash", "unknown"),
        "git_branch": git_info.get("branch", "unknown"),
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "inference_duration_seconds": float(total_time),
        "inference_poses_per_second": float(inference_rate),
        "profile_timings_enabled": bool(profile_timings),
        "progress_jsonl": str(progress_jsonl_path) if progress_jsonl_path is not None else None,
        "input_mode_requested": resolved_input_mode,
        "input_mode_effective": effective_input_mode,
        "model_input_transform": model_input_transform.to_attrs(),
        "model_input_stride": int(model_stride),
        "model_input_transform_name": model_input_transform.name,
        "model_input_shape_hw": list(model_input_transform.model_shape),
        "model_network_input_shape_hw": [int(imgsz), int(imgsz)],
        "native_roi_shape_hw": list(model_input_transform.native_shape),
    })
    if timing_profiler.enabled:
        run_group.attrs["timing_profile"] = timing_profiler.summary(
            total_items=int(total_rois),
            wall_seconds=float(total_time),
            notes=[
                "roi_read measures ROI slice fetch from the active crop image source.",
                "model_predict wraps Ultralytics predict(), including model-side preprocessing and postprocessing.",
                "tensor input mode supplies normalized BCHW tensors and bypasses the legacy numpy-list RGB expansion.",
                "output_write measures Zarr writes for keypoint outputs and lineage flags.",
            ],
        )
    if crop_source.roi_cache_key:
        run_group.attrs["source_roi_cache_key"] = crop_source.roi_cache_key
    if crop_source.roi_cache_path:
        run_group.attrs["source_roi_cache_path"] = crop_source.roi_cache_path
    if training_materialization is not None:
        run_group.attrs["source_training_crop_materialization_binding"] = dict(
            training_materialization.binding
        )
        run_group.attrs["source_training_crop_materialization_binding_digest"] = (
            training_materialization.binding["payload_digest"]
        )
    if roi_cache_staging_payload is not None:
        run_group.attrs["source_roi_cache_staging"] = roi_cache_staging_payload
    if source_refined_run:
        run_group.attrs["source_refined_run"] = source_refined_run
    effective_run_provenance = run_provenance if run_provenance is not None else cli_provenance
    if effective_run_provenance is None:
        effective_run_provenance = build_run_provenance(
            command="fisheye.detection.detect_keypoints_yolo",
            params={
                "zarr_path": str(zarr_path.resolve()),
                "model_path": str(model_path_resolved),
                "run_name": run_name,
                "output_parent": output_parent_name,
                "crop_run": crop_run,
                "pose_schema": pose_schema,
                "device": device,
                "batch_size": batch_size,
                "conf": conf,
                "iou": iou,
                "max_det": max_det,
                "mask_threshold": mask_threshold,
                "roi_cache_policy": roi_cache_policy,
                "roi_cache_manifest": str(roi_cache_manifest) if roi_cache_manifest is not None else None,
                "roi_work_package_manifest": (
                    str(roi_work_package_manifest)
                    if roi_work_package_manifest is not None
                    else None
                ),
                "roi_live_acceleration": roi_live_acceleration,
                "model_input_transform_mode": model_input_transform_mode,
                "expected_model_stride": expected_model_stride,
                "coordinate_contract_mode": coordinate_contract_mode,
                "input_mode": input_mode,
                "keypoint_roi_shard_rows": keypoint_roi_shard_rows,
                "keypoint_frame_shard_rows": (
                    keypoint_frame_shard_rows if keypoint_roi_shard_rows is not None else None
                ),
                "keypoint_storage_policy": (
                    "default_indexed_sharding_v1"
                    if keypoint_roi_shard_rows is not None
                    else "explicit_regular_chunks_override"
                ),
            },
            input_run_ids={
                "crop": latest_crop,
                "refined_detect": source_refined_run,
            },
            cwd=Path.cwd(),
        )
    effective_run_provenance = append_input_artifacts(effective_run_provenance, [model_artifact])
    if effective_run_provenance is not None:
        run_group.attrs[RUN_PROVENANCE_ATTR] = dict(effective_run_provenance)
        run_group.attrs[CLI_RUN_PROVENANCE_ATTR] = dict(effective_run_provenance)
    provenance_record = build_stage_provenance(
        stage="keypoints_detect",
        command=" ".join(sys.argv),
        created_at_utc=str(run_group.attrs.get("keypoints_timestamp_utc")),
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
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        scheduler=scheduler_info,
        parameters=dict(run_group.attrs.get("parameters") or {}),
        inputs={
            "source_crop_run": latest_crop,
            **crop_snapshot_attrs,
            **crop_pixel_attrs,
            **work_package_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_cache_used": bool(crop_source.roi_cache_used),
            "roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
            "roi_cache_key": crop_source.roi_cache_key,
            "roi_cache_path": crop_source.roi_cache_path,
            "roi_cache_source_tier": effective_roi_cache_source_tier,
            "roi_cache_staged_to_node_scratch": bool(roi_cache_staged_to_node_scratch),
            "roi_cache_staging_policy": roi_cache_staging_policy,
            "roi_cache_staging": roi_cache_staging_payload,
            "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
            "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
            "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
            "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
            "input_mode_requested": resolved_input_mode,
            "input_mode_effective": effective_input_mode,
            "model_input_stride": int(model_stride),
            "source_detect_run": source_detect_run or "unknown",
            "source_refined_run": source_refined_run,
            "frame_source": crop_source.frame_source_kind,
            "source_video_path": crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
        },
        artifacts={
            "model_path": str(model_path_resolved),
            "model_name": model_path.name,
            "ultralytics_version": ultralytics_version,
            "device": resolved_model_device,
            "requested_device": device,
            "normalized_torch_device": torch_device,
            "initial_model_device": model_device,
            "resolved_model_device": resolved_model_device,
            "output_parent": output_parent_name,
            "pose_schema": pose_schema_obj.name,
            "skeleton_id": str(pose_schema_attrs["skeleton_id"]),
            "kpt_shape": list(pose_schema_attrs["kpt_shape"]),
            "model_kpt_shape": list(model_kpt_shape) if model_kpt_shape is not None else None,
            "model_input_stride": int(model_stride),
            "keypoint_storage_layout": (
                "indexed_sharding_v1"
                if keypoint_roi_shard_rows is not None
                else "regular_chunks_v1"
            ),
            "keypoint_storage_policy": (
                "default_indexed_sharding_v1"
                if keypoint_roi_shard_rows is not None
                else "explicit_regular_chunks_override"
            ),
            "keypoint_shard_write": shard_write_summary,
        },
    )
    write_stage_provenance(run_group, provenance_record)
    if override_data is not None:
        run_group.attrs["refined_roi_overrides"] = int(override_data["count"])
        run_group.attrs["refined_roi_source"] = override_data["path"]
        if override_data["decoder"]:
            run_group.attrs["refined_roi_decoder"] = override_data["decoder"]
        if override_data["duration"] is not None:
            run_group.attrs["refined_roi_generation_duration_seconds"] = float(override_data["duration"])

    validate_immutable_yolo_storage(
        run_group,
        stage="keypoints",
        row_shard_rows=keypoint_roi_shard_rows,
        frame_shard_rows=keypoint_frame_shard_rows,
    )
    expected_coordinate_contract = (
        "canonical_v2"
        if keypoint_coordinate_context is not None
        else "legacy_noncanonical_explicit_v1"
    )

    status_details: Dict[str, object] = {
        "reason": "present",
        "run_group": output_parent_name,
        "coordinate_contract_mode": coordinate_contract_mode,
        "coordinate_contract": expected_coordinate_contract,
        "source_crop_run": latest_crop,
        **crop_snapshot_attrs,
        **crop_pixel_attrs,
        **work_package_attrs,
        "source_roi_read_mode": crop_source.roi_read_mode,
        "roi_cache_policy": crop_source.roi_cache_policy,
        "roi_cache_used": bool(crop_source.roi_cache_used),
        "roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
        "roi_cache_source_tier": effective_roi_cache_source_tier,
        "roi_cache_staged_to_node_scratch": bool(roi_cache_staged_to_node_scratch),
        "roi_cache_staging_policy": roi_cache_staging_policy,
        "roi_cache_staging": roi_cache_staging_payload,
        "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
        "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
        "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
        "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
        "source_detect_run": source_detect_run or "unknown",
        "total_rois": int(total_rois),
        "successful_detections": int(success_total),
        "failed_detections": int(failure_total),
        "success_rate_percent": round(float(success_rate), 2),
        "pose_failure_code_histogram": failure_code_histogram,
        "pose_schema": pose_schema_obj.name,
        "skeleton_id": str(pose_schema_attrs["skeleton_id"]),
        "kpt_shape": list(pose_schema_attrs["kpt_shape"]),
        "model_kpt_shape": list(model_kpt_shape) if model_kpt_shape is not None else None,
        "input_mode_requested": resolved_input_mode,
        "input_mode_effective": effective_input_mode,
        "model_input_stride": int(model_stride),
        "requested_device": device,
        "normalized_torch_device": torch_device,
        "initial_model_device": model_device,
        "resolved_model_device": resolved_model_device,
        "execution_hostname": scheduler_info.get("execution_hostname"),
        "scheduler": scheduler_info.get("scheduler"),
        "scheduler_job_id": scheduler_info.get("job_id"),
        "scheduler_job_name": scheduler_info.get("job_name"),
        "scheduler_job_index": scheduler_info.get("job_index"),
        "scheduler_queue": scheduler_info.get("queue"),
        "scheduler_hosts": scheduler_info.get("hosts") or scheduler_info.get("node_list"),
        "scheduler_mcpu_hosts": scheduler_info.get("mcpu_hosts"),
        "scheduler_cuda_visible_devices": scheduler_info.get("cuda_visible_devices"),
        "scheduler_gpu_request": scheduler_info.get("gpu_request"),
    }
    if source_refined_run:
        status_details["source_refined_run"] = source_refined_run
    if override_data is not None:
        status_details["refined_roi_overrides"] = int(override_data["count"])
        status_details["refined_roi_source"] = str(override_data["path"])

    # All input-derived arrays, provenance, and status details are now durable.
    # Close the scientific pixel source before descriptors or selectors can make
    # this attempt authoritative.
    if boundary is not None:
        boundary.close_crop_source()
    else:  # pragma: no cover - the public writer is always boundary-decorated
        crop_source.close()

    _revalidate_keypoint_model_artifact(
        model_path,
        model_artifact,
        checkpoint="after inference and before publication",
    )

    run_path = f"{output_parent_name}/{resolved_run_name}"
    if keypoint_coordinate_context is not None:
        checkpoint = capture_keypoint_coordinate_publication_checkpoint(
            root,
            run_path,
            expected_publication_owner=(
                boundary.owner_token if boundary is not None else None
            ),
        )
        if boundary is not None:
            boundary.bind_coordinate_checkpoint(checkpoint)
        publish_keypoint_coordinate_surfaces(root, run_path)
        # Publication also writes through a fresh root-resolved group.  Complete
        # only through another fresh handle so descriptors cannot be lost.
        run_group = root[run_path]
    else:
        run_group.attrs["coordinate_contract"] = expected_coordinate_contract

    # Recheck every reused immutable input and newly published coordinate
    # authority while this child remains running and selector-ineligible.
    finish_proof_verification()

    mark_run_complete(
        run_group,
        # Canonical activation owns selector publication under its lease.
        # Noncanonical collection shards use the generic completion cleanup.
        parent_group=(None if keypoint_coordinate_context is not None else run_parent),
        run_name=resolved_run_name,
        run_provenance=effective_run_provenance,
    )
    if keypoint_coordinate_context is not None:
        restart_proof_verification()
        fresh_surfaces = _load_completed_ineligible_keypoint_coordinate_surfaces(
            root,
            run_path,
        )
        _activate_validated_keypoint_coordinate_surfaces(
            root,
            run_parent,
            fresh_surfaces,
            run_name=resolved_run_name,
            publication_owner_token=(
                boundary.owner_token
                if boundary is not None and boundary.owner_token is not None
                else str(run_group.attrs[KEYPOINT_PUBLICATION_OWNER_ATTR])
            ),
            parent_selector_snapshot=(
                boundary.parent_selector_snapshot
                if boundary is not None
                and boundary.parent_selector_snapshot is not None
                else _snapshot_selected_attrs(
                    run_parent,
                    _KEYPOINT_PARENT_SELECTOR_ATTRS,
                )
            ),
            root_pointer_snapshot=(
                boundary.root_pointer_snapshot
                if boundary is not None
                and boundary.root_pointer_snapshot is not None
                else _snapshot_selected_attrs(
                    root,
                    ("current_keypoint_group_path",),
                )
            ),
        )
    if boundary is not None:
        boundary.mark_finalized()

    try:
        if output_parent_name == DEFAULT_KEYPOINT_OUTPUT_PARENT:
            _emit_keypoint_step_status(
                root=root,
                zarr_path=zarr_path.resolve(),
                run_name=resolved_run_name,
                method=normalize_attr(run_group.attrs.get("method")) or "yolo_pose",
                coverage_pct=float(success_rate),
                details=status_details,
                console=console,
                registry=registry,
            )
    except Exception as exc:  # telemetry is explicitly post-commit
        _warn_postcommit_failure(
            console,
            label="registry/status telemetry",
            error=exc,
        )
    try:
        _write_keypoint_progress_jsonl(
            progress_jsonl_path,
            "complete",
            zarr_path=str(zarr_path.resolve()),
            run_name=resolved_run_name,
            crop_run=latest_crop,
            total_rois=int(total_rois),
            successful_detections=int(success_total),
            failed_detections=int(failure_total),
            success_rate_percent=round(float(success_rate), 2),
            elapsed_seconds=float(total_time),
            poses_per_second=float(inference_rate),
            resolved_model_device=resolved_model_device,
        )
    except Exception as exc:  # telemetry is explicitly post-commit
        _warn_postcommit_failure(
            console,
            label="completion progress telemetry",
            error=exc,
        )

    try:
        summary_lines = [
            "[green]✓[/green] Pose inference complete",
            "",
            f"[bold]Run:[/bold] {output_parent_name}/{resolved_run_name}",
            f"[bold]Total ROIs:[/bold] {total_rois}",
            f"[bold]Successful:[/bold] {success_total} ({success_rate:.2f}%)",
            f"[bold]Failed:[/bold] {failure_total}",
            f"[bold]Model:[/bold] {model_path_resolved}",
            f"[bold]Duration:[/bold] {total_time:.1f}s ({inference_rate:.1f} poses/s)",
        ]
        if timing_profiler.enabled:
            summary_lines.append("[bold]Timing Profile:[/bold]")
            for line in timing_profiler.render_lines(
                total_items=total_rois,
                wall_seconds=total_time,
                limit=5,
            ):
                summary_lines.append(f"[dim]{line}[/dim]")
        if override_data is not None:
            summary_lines.append(
                f"[dim]Refined ROI overrides: {override_data['count']} "
                f"from {override_data['path']}[/dim]"
            )
        completion = Panel(
            "\n".join(summary_lines),
            title="YOLO Pose Inference",
            border_style="green",
        )
        console.print("\n")
        console.print(completion)
    except Exception as exc:  # presentation is explicitly post-commit
        _warn_postcommit_failure(
            console,
            label="completion presentation",
            error=exc,
        )

    return resolved_run_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO-based keypoint inference on Palette Zarr crops")
    parser.add_argument("zarr_path", type=str, help="Path to the Palette Zarr archive")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO pose weights (.pt)")
    parser.add_argument("--run-name", help="Optional custom run name for the selected output parent")
    parser.add_argument(
        "--output-parent",
        choices=KEYPOINT_OUTPUT_PARENTS,
        default=DEFAULT_KEYPOINT_OUTPUT_PARENT,
        help=(
            "Parent group for output runs. Use keypoint_shard_runs for clipped-collection "
            "GPU shards that must not become canonical keypoint_runs."
        ),
    )
    parser.add_argument("--crop-run", help="Optional crop run override (defaults to latest)")
    parser.add_argument(
        "--pose-schema",
        default=None,
        help=(
            "Optional package-schema consistency assertion. Canonical publication "
            "takes ordered labels only from --model-pose-schema-binding; "
            f"legacy_noncanonical defaults to {DEFAULT_POSE_SCHEMA_NAME}."
        ),
    )
    parser.add_argument(
        "--model-pose-schema-binding",
        type=Path,
        default=None,
        help=(
            "Explicit digest-bound JSON mapping from this model artifact to its "
            "ordered pose schema; required for canonical direct-model inference."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    keypoint_storage_group = parser.add_mutually_exclusive_group()
    keypoint_storage_group.add_argument(
        "--keypoint-roi-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
        help=(
            "Requested outer rows for indexed-sharded ROI-domain keypoint arrays "
            f"(default: {DEFAULT_KEYPOINT_ROI_SHARD_ROWS})."
        ),
    )
    keypoint_storage_group.add_argument(
        "--no-keypoint-sharding",
        action="store_const",
        dest="keypoint_roi_shard_rows",
        const=None,
        help="Use ordinary chunks for YOLO keypoint outputs.",
    )
    parser.add_argument(
        "--keypoint-frame-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
        help=(
            "Aligned outer shard rows for frame-domain arrays when ROI sharding is enabled "
            f"(default: {DEFAULT_KEYPOINT_FRAME_SHARD_ROWS})."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Ultralytics network preprocessing size for YOLO inference.",
    )
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=None,
        help=(
            "Square pixel extent submitted to Ultralytics before its internal "
            "--imgsz preprocessing; defaults to --imgsz for compatibility."
        ),
    )
    parser.add_argument(
        "--expected-model-stride",
        type=int,
        default=None,
        help="Fail unless the loaded model declares this maximum stride.",
    )
    parser.add_argument("--device", default=None, help="Torch device string (e.g. '0' or 'cuda:0')")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=1, help="Maximum detections per ROI")
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Compatibility parameter recorded in run metadata (not used for pose decoding).",
    )
    parser.add_argument("--registry", type=Path, default=None, help="Optional registry SQLite path.")
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
    parser.add_argument("--source-crop-row-start", type=int, default=None)
    parser.add_argument("--source-crop-row-stop", type=int, default=None)
    parser.add_argument(
        "--roi-cache-expected-archive-path",
        type=Path,
        default=None,
        help=(
            "Authority archive bound by --roi-cache-manifest when inference runs "
            "against a byte-identical node-scratch archive copy."
        ),
    )
    roi_manifest_group.add_argument(
        "--roi-work-package-manifest",
        type=Path,
        default=None,
        help=(
            "Keyed subset ROI package for delta inference. Requires "
            "--output-parent keypoint_shard_runs."
        ),
    )
    parser.add_argument(
        "--roi-cache-source-tier",
        choices=("node_scratch", "prfs_workflow_scratch", "canonical_materialized", "unknown"),
        default=None,
        help="Optional provenance label for where --roi-cache-manifest was read from.",
    )
    parser.add_argument(
        "--roi-cache-staged-to-node-scratch",
        action="store_true",
        help="Stamp provenance that the effective ROI cache manifest was staged to node-local scratch before inference.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Collect per-stage timing diagnostics and store them in the output run attrs.",
    )
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file for live progress events written after each durable output batch.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=1,
        help="Write one progress JSONL event every N completed batches (default: 1).",
    )
    parser.add_argument(
        "--input-mode",
        choices=_KEYPOINT_INPUT_MODES,
        default="numpy-list",
        help=(
            "Input preparation mode for Ultralytics prediction. 'numpy-list' preserves the legacy "
            "RGB numpy-list path; 'tensor' feeds normalized BCHW tensors directly; 'auto' uses "
            "tensor mode only when ROI geometry is equivalent to the legacy path."
        ),
    )
    parser.add_argument(
        "--model-input-transform",
        choices=MODEL_INPUT_TRANSFORM_CHOICES,
        default="auto",
        help=(
            "Reversible transform from native ROI crops to the requested model input size. "
            "'auto' is identity when sizes match and centered zero-padding when --imgsz is larger."
        ),
    )
    parser.add_argument(
        "--coordinate-contract-mode",
        choices=_KEYPOINT_COORDINATE_CONTRACT_MODES,
        default="canonical",
        help=(
            "Publish a canonical-v2 coordinate graph from an exact canonical crop "
            "(default), or explicitly quarantine unsupported historical/shard output "
            "as legacy_noncanonical."
        ),
    )
    parser.add_argument(
        "--require-training-materialization-binding",
        action="store_true",
        help=(
            "Require --crop-run to be an exact self-contained training crop "
            "materialization. This is a terminal inference path; canonical v2 "
            "finalization still binds the source crop-v2 authority."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    detect_keypoints_yolo(
        zarr_path=args.zarr_path,
        model_path=args.model,
        expected_model_stride=args.expected_model_stride,
        run_name=args.run_name,
        output_parent=args.output_parent,
        crop_run=args.crop_run,
        pose_schema=args.pose_schema,
        model_pose_schema_binding=args.model_pose_schema_binding,
        batch_size=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        model_input_size=args.model_input_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=args.verbose,
        mask_threshold=args.mask_threshold,
        roi_cache_policy=args.roi_cache_policy,
        roi_cache_dir=args.roi_cache_dir,
        roi_cache_manifest=args.roi_cache_manifest,
        roi_cache_expected_archive_path=args.roi_cache_expected_archive_path,
        roi_work_package_manifest=args.roi_work_package_manifest,
        source_crop_row_start=args.source_crop_row_start,
        source_crop_row_stop=args.source_crop_row_stop,
        roi_cache_source_tier=args.roi_cache_source_tier,
        roi_cache_staged_to_node_scratch=bool(args.roi_cache_staged_to_node_scratch),
        input_mode=args.input_mode,
        model_input_transform_mode=args.model_input_transform,
        coordinate_contract_mode=args.coordinate_contract_mode,
        require_training_materialization_binding=bool(
            args.require_training_materialization_binding
        ),
        profile_timings=args.profile_timings,
        progress_jsonl=args.progress_jsonl,
        progress_every_batches=args.progress_every_batches,
        keypoint_roi_shard_rows=args.keypoint_roi_shard_rows,
        keypoint_frame_shard_rows=args.keypoint_frame_shard_rows,
        registry=args.registry,
    )


__all__ = ["detect_keypoints_yolo", "main"]


if __name__ == "__main__":
    main()
