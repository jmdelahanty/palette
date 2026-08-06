"""Node-local construction and atomic publication of sampled training bases."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    tree_inventory,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    open_zarr_group_direct,
)
from fisheye.utils.import_sampled_training_pynvvc import (
    SampledTrainingImportResult,
    import_sampled_training_pynvvc,
)


SAMPLED_TRAINING_BASE_PUBLICATION_SCHEMA_ID = (
    "palette.sampled_training_base_publication"
)
SAMPLED_TRAINING_BASE_PUBLICATION_SCHEMA_VERSION = 1
SAMPLED_TRAINING_BASE_PUBLICATION_POLICY = (
    "node_local_build_checked_copy_hidden_sibling_rename_v1"
)


def _require_node_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Sampled training scratch root not found: {resolved}"
        )
    if resolved in {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/scratch").resolve(),
    } or str(resolved).startswith(("/groups/", "/nrs/")):
        raise ValueError(
            "Sampled training scratch must be a bounded node-local directory."
        )
    return resolved


def _expected_frame_indices(
    *, source_frame_count: int, frame_step: int, skip_tail_frames: int
) -> np.ndarray:
    if source_frame_count <= 0:
        raise ValueError("source_frame_count must be positive.")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive.")
    if skip_tail_frames < 0 or skip_tail_frames >= source_frame_count:
        raise ValueError(
            "skip_tail_frames must be nonnegative and less than source_frame_count."
        )
    return np.arange(
        0,
        source_frame_count - skip_tail_frames,
        frame_step,
        dtype=np.int32,
    )


def validate_sampled_training_base(
    path: str | Path,
    *,
    source_frame_count: int,
    frame_step: int,
    skip_tail_frames: int,
) -> dict[str, Any]:
    """Validate the exact mutable base surface and its physical access units."""

    archive = Path(path).expanduser().resolve()
    root = open_zarr_group_direct(archive, mode="r")
    if str(root.attrs.get("zarr_purpose") or "") != "training":
        raise ValueError("Sampled base must declare zarr_purpose='training'.")
    raw = root["raw_video"]
    full = raw["images_full"]
    downsampled = raw["images_ds"]
    indices = raw["original_frame_indices"]
    expected = _expected_frame_indices(
        source_frame_count=int(source_frame_count),
        frame_step=int(frame_step),
        skip_tail_frames=int(skip_tail_frames),
    )
    row_count = int(expected.shape[0])
    if full.ndim != 3 or downsampled.ndim != 3:
        raise ValueError("Training images_full and images_ds must be rank three.")
    if full.dtype != np.dtype("uint8") or downsampled.dtype != np.dtype("uint8"):
        raise TypeError("Training image arrays must use exact uint8 dtype.")
    if indices.dtype != np.dtype("int32") or indices.shape != (row_count,):
        raise TypeError(
            "raw_video/original_frame_indices must be exact int32 with sampled-row shape."
        )
    if full.shape[0] != row_count or downsampled.shape[0] != row_count:
        raise ValueError("Training image arrays must share the sampled-row axis.")
    if int(full.chunks[0]) != 1 or int(downsampled.chunks[0]) != 1:
        raise ValueError(
            "Sampled base image arrays must use one-frame physical chunks."
        )
    actual = np.asarray(indices[:], dtype=np.int32)
    if not np.array_equal(actual, expected):
        raise ValueError(
            "original_frame_indices does not equal the declared sampling plan."
        )
    return {
        "valid": True,
        "row_count": row_count,
        "full_shape": [int(value) for value in full.shape],
        "full_chunks": [int(value) for value in full.chunks],
        "downsampled_shape": [int(value) for value in downsampled.shape],
        "downsampled_chunks": [int(value) for value in downsampled.chunks],
        "index_shape": [int(value) for value in indices.shape],
        "index_chunks": [int(value) for value in indices.chunks],
        "first_source_frame": int(actual[0]),
        "last_source_frame": int(actual[-1]),
    }


def _result_payload(result: SampledTrainingImportResult) -> dict[str, Any]:
    return {
        "zarr_path": str(result.zarr_path),
        "source_video_path": str(result.source_video_path),
        "imported_frame_count": int(result.imported_frame_count),
        "source_frame_count": int(result.source_frame_count),
        "frame_step": int(result.frame_step),
        "skip_tail_frames": int(result.skip_tail_frames),
        "original_resolution": [int(value) for value in result.original_resolution],
        "downsampled_resolution": (
            [int(value) for value in result.downsampled_resolution]
            if result.downsampled_resolution is not None
            else None
        ),
        "decode_backend": str(result.decode_backend),
        "duration_s": float(result.duration_s),
    }


def publish_sampled_training_base(
    *,
    destination: str | Path,
    scratch_root: str | Path,
    video_path: str | Path,
    source_frame_count: int,
    frame_step: int,
    skip_tail_frames: int,
    config_path: str | Path,
    camera_id: str,
    recording_dir: str | Path,
    h5_path: str | Path,
    gpu_id: int = 0,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Build one mutable review base locally and publish it atomically.

    The published base is intentionally not consolidated because detection and
    review stages will add groups to it.  It is also selector-ineligible and is
    not registered by this function.
    """

    target = Path(destination).expanduser().resolve()
    scratch = _require_node_local_scratch(Path(scratch_root))
    source_video = Path(video_path).expanduser().resolve()
    recording = Path(recording_dir).expanduser().resolve()
    source_h5 = Path(h5_path).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    for label, source in (
        ("video", source_video),
        ("recording", recording),
        ("H5", source_h5),
        ("config", config),
    ):
        if not source.exists():
            raise FileNotFoundError(f"Sampled training {label} input not found: {source}")
    if target.exists():
        raise FileExistsError(f"Sampled training destination exists: {target}")

    with tempfile.TemporaryDirectory(
        prefix="palette-sampled-training-base-", dir=str(scratch)
    ) as temporary:
        local_archive = Path(temporary) / "training_base.zarr"
        imported = import_sampled_training_pynvvc(
            video_path=source_video,
            zarr_path=local_archive,
            source_frame_count=int(source_frame_count),
            frame_step=int(frame_step),
            skip_tail_frames=int(skip_tail_frames),
            config_path=config,
            overwrite=False,
            camera_id=str(camera_id),
            recording_dir=recording,
            h5_path=source_h5,
            gpu_id=int(gpu_id),
            require_cuda=bool(require_cuda),
        )
        validate_sampled_training_base(
            local_archive,
            source_frame_count=int(source_frame_count),
            frame_step=int(frame_step),
            skip_tail_frames=int(skip_tail_frames),
        )
        local_root = open_zarr_group_direct(local_archive, mode="a")
        publication = {
            "schema_id": SAMPLED_TRAINING_BASE_PUBLICATION_SCHEMA_ID,
            "schema_version": SAMPLED_TRAINING_BASE_PUBLICATION_SCHEMA_VERSION,
            "policy": SAMPLED_TRAINING_BASE_PUBLICATION_POLICY,
            "publication_status": "complete",
            "training_artifact_status": "awaiting_detection_review",
            "stage_selector_eligible": False,
            "registry_activation": "deferred",
            "metadata_mode": "direct_mutable",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_provenance": build_writer_run_provenance(
                command="fisheye.shared.zarr.training_base_publication",
                params={
                    "source_frame_count": int(source_frame_count),
                    "frame_step": int(frame_step),
                    "skip_tail_frames": int(skip_tail_frames),
                    "config_path": str(config),
                    "gpu_id": int(gpu_id),
                },
                input_artifacts=[
                    {"role": "source_video", "path": str(source_video)},
                    {"role": "source_h5", "path": str(source_h5)},
                ],
            ),
        }
        local_root.attrs.update(
            {
                "training_artifact_status": "awaiting_detection_review",
                "stage_selector_eligible": False,
                "registry_activation": "deferred",
                "sampled_training_base_publication": publication,
            }
        )
        # Revalidate after lifecycle attrs are committed. Do not consolidate:
        # this artifact is explicitly mutable until review is accepted.
        validate_sampled_training_base(
            local_archive,
            source_frame_count=int(source_frame_count),
            frame_step=int(frame_step),
            skip_tail_frames=int(skip_tail_frames),
        )
        local_inventory = tree_inventory(local_archive, hash_content=True)

        target.parent.mkdir(parents=True, exist_ok=True)
        hidden = target.with_name(
            f".{target.name}.publish_tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        with archive_metadata_publication_lock(target):
            if target.exists() or hidden.exists():
                raise FileExistsError(
                    f"Sampled training publication target became occupied: {target}"
                )
            try:
                shutil.copytree(local_archive, hidden)
                hidden_inventory = tree_inventory(hidden, hash_content=True)
                if hidden_inventory != local_inventory:
                    raise RuntimeError(
                        "Sampled training physical copy differs from node-local source."
                    )
                validate_sampled_training_base(
                    hidden,
                    source_frame_count=int(source_frame_count),
                    frame_step=int(frame_step),
                    skip_tail_frames=int(skip_tail_frames),
                )
                os.replace(hidden, target)
            except Exception:
                if hidden.exists():
                    shutil.rmtree(hidden)
                raise

    final_validation = validate_sampled_training_base(
        target,
        source_frame_count=int(source_frame_count),
        frame_step=int(frame_step),
        skip_tail_frames=int(skip_tail_frames),
    )
    return {
        "schema_id": SAMPLED_TRAINING_BASE_PUBLICATION_SCHEMA_ID,
        "schema_version": SAMPLED_TRAINING_BASE_PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "destination": str(target),
        "training_artifact_status": "awaiting_detection_review",
        "stage_selector_eligible": False,
        "registry_activation": "deferred",
        "metadata_mode": "direct_mutable",
        "import": _result_payload(imported),
        "validation": final_validation,
        "physical_inventory": local_inventory.to_json(),
    }


__all__ = [
    "SAMPLED_TRAINING_BASE_PUBLICATION_POLICY",
    "SAMPLED_TRAINING_BASE_PUBLICATION_SCHEMA_ID",
    "SAMPLED_TRAINING_BASE_PUBLICATION_SCHEMA_VERSION",
    "publish_sampled_training_base",
    "validate_sampled_training_base",
]
