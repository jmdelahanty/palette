"""Atomic publication of self-contained training crop materializations."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
from typing import Any, Mapping
from collections.abc import Sequence
import uuid

import zarr

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
    tree_inventory,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.training_crop_materialization import (
    TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE,
    bind_training_crop_materialization,
    build_training_crop_materialization_binding,
)
from fisheye.shared.zarr.training_dataset_composition import (
    TRAINING_DATASET_COMPOSITION_ATTRIBUTE,
    bind_training_dataset_composition,
    build_training_dataset_composition,
    validate_training_crop_source_join,
    validate_training_detection_review_base,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
    open_zarr_group_direct,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.utils.regenerate_training_crops_pynvvc import (
    regenerate_training_crops_pynvvc,
)


TRAINING_CROP_PUBLICATION_SCHEMA_ID = (
    "palette.training_crop_materialization_publication"
)
TRAINING_CROP_PUBLICATION_SCHEMA_VERSION = 1
TRAINING_CROP_PUBLICATION_POLICY = (
    "node_local_materialize_then_atomic_selector_ineligible_import_v1"
)
TRAINING_CROP_PUBLICATION_ROLLBACK_POLICY = (
    "retain_failed_owner_bound_selector_ineligible_child_v1"
)
TRAINING_ARTIFACT_PUBLICATION_SCHEMA_ID = (
    "palette.training_crop_artifact_publication"
)
TRAINING_ARTIFACT_PUBLICATION_SCHEMA_VERSION = 1
TRAINING_DATASET_ENRICHMENT_SCHEMA_ID = "palette.training_dataset_enrichment"
TRAINING_DATASET_ENRICHMENT_SCHEMA_VERSION = 1


def _require_node_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Training crop publication scratch root not found: {resolved}"
        )
    if resolved in {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/scratch").resolve(),
    } or str(resolved).startswith(("/groups/", "/nrs/")):
        raise ValueError(
            "Training crop publication scratch must be a bounded node-local path."
        )
    return resolved


def _validate_materialized_run(path: Path) -> dict[str, Any]:
    try:
        run = zarr.open_group(str(path), mode="r", use_consolidated=False)
        binding = run.attrs.get(TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE)
        valid = (
            run.attrs.get("status") == "completed"
            and run.attrs.get("stage_selector_eligible") is False
            and isinstance(binding, Mapping)
            and dict(binding) == build_training_crop_materialization_binding(run)
        )
        return {
            "valid": bool(valid),
            "row_count": int(run["roi_images"].shape[0]),
            "roi_shape": [
                int(run["roi_images"].shape[1]),
                int(run["roi_images"].shape[2]),
            ],
            "binding_digest": (
                str(binding.get("payload_digest"))
                if isinstance(binding, Mapping)
                else None
            ),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def publish_training_crop_materialization(
    *,
    destination: str | Path,
    source_zarr: str | Path,
    source_crop_run: str,
    run_id: str,
    scratch_root: str | Path,
    video_path: str | Path | None = None,
    roi_cache_manifest: str | Path | None = None,
    copy_backend: str = "python",
    cache_copy_batch_rows: int = 1024,
    decode_mode: str = "auto",
    decode_chunk_frames: int = 1,
    source_instance_keys: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Materialize locally, atomically import, and keep the run unselected."""

    archive = Path(destination).expanduser().resolve()
    source = Path(source_zarr).expanduser().resolve()
    scratch = _require_node_local_scratch(Path(scratch_root))
    if not archive.is_dir():
        raise FileNotFoundError(f"Training Zarr not found: {archive}")
    root = open_zarr_group_direct(archive, mode="r")
    if str(root.attrs.get("zarr_purpose") or "").strip().lower() != "training":
        raise ValueError("Destination must be a training-purpose Zarr.")
    candidate = str(run_id).strip()
    if not candidate or "/" in candidate or candidate.startswith("."):
        raise ValueError("run_id must be one safe non-hidden child-group name.")
    if (video_path is None) == (roi_cache_manifest is None):
        raise ValueError(
            "Choose exactly one materialization input: video_path or roi_cache_manifest."
        )
    target_path = archive / "crop_runs" / candidate
    if target_path.exists():
        raise FileExistsError(f"Training crop run already exists: {target_path}")

    with tempfile.TemporaryDirectory(
        prefix=f"palette-training-crop-{candidate}-",
        dir=str(scratch),
    ) as temporary:
        local_archive = Path(temporary) / "training.zarr"
        local_root = zarr.open_group(str(local_archive), mode="w", zarr_format=3)
        local_root.attrs["zarr_purpose"] = "training"
        local_root.require_group("crop_runs")
        materialization = regenerate_training_crops_pynvvc(
            zarr_path=local_archive,
            source_zarr_path=source,
            source_crop_run=str(source_crop_run),
            target_crop_run=candidate,
            video_path=video_path,
            roi_cache_manifest=roi_cache_manifest,
            cache_copy_batch_rows=int(cache_copy_batch_rows),
            source_instance_keys=source_instance_keys,
            decode_mode=str(decode_mode),
            decode_chunk_frames=int(decode_chunk_frames),
            overwrite=False,
            set_latest=False,
            consolidate_metadata=True,
            dry_run=False,
        )
        local_run = local_archive / "crop_runs" / candidate
        local_group = zarr.open_group(
            str(local_run), mode="a", use_consolidated=False
        )
        mark_run_started(local_group, run_name=candidate, stage="crop")
        local_group.attrs["stage_selector_eligible"] = False
        local_group.attrs["immutable_training_materialization"] = True
        mark_run_complete(
            local_group,
            run_name=candidate,
            run_provenance=build_writer_run_provenance(
                command=(
                    "fisheye.shared.zarr.training_crop_materialization_publication"
                ),
                params={
                    "publication_schema_id": TRAINING_CROP_PUBLICATION_SCHEMA_ID,
                    "materialization_provider": materialization[
                        "materialization_provider"
                    ],
                },
                input_run_ids={"source_crop_run": str(source_crop_run)},
            ),
        )
        consolidate_metadata_capture_expected_warnings(local_archive)

        def prepare_parents(current_root: zarr.Group) -> tuple[zarr.Group, ...]:
            parent = require_runs_parent(
                current_root,
                "crop_runs",
                completion_epoch=COMPLETION_EPOCH_STRICT,
            )
            return (current_root, parent)

        def complete_run(
            _current_root: zarr.Group,
            parent: zarr.Group,
            run: zarr.Group,
        ) -> None:
            run.attrs["stage_selector_eligible"] = False
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=candidate,
                run_provenance=run.attrs.get("run_provenance"),
            )

        def verify_unselected(current_root: zarr.Group) -> None:
            parent = current_root["crop_runs"]
            for attr in ("latest", "latest_complete", "authoritative_run"):
                if parent.attrs.get(attr) == candidate:
                    raise RuntimeError(
                        f"Selector-ineligible training crop became {attr}."
                    )

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=archive,
                local_run_path=local_run,
                target_run_path=target_path,
                run_name=candidate,
                lock_suffix="training_crop_materialization_publication",
                publish_schema_id=TRAINING_CROP_PUBLICATION_SCHEMA_ID,
                policy=TRAINING_CROP_PUBLICATION_POLICY,
                rollback_policy=TRAINING_CROP_PUBLICATION_ROLLBACK_POLICY,
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=_validate_materialized_run,
            prepare_parents=prepare_parents,
            complete_run=complete_run,
            verify_pointers=verify_unselected,
            payload_metadata={
                "source_zarr": str(source),
                "source_crop_run": str(source_crop_run),
                "selector_activation": "deferred",
            },
        )

    # The child is not selectable, so a crash before this final metadata step
    # cannot expose it as production input.  Serialize the archive-wide
    # consolidated generation and then require a strict decoded binding.
    with archive_metadata_publication_lock(archive):
        consolidate_metadata_capture_expected_warnings(archive)
    bound = bind_training_crop_materialization(archive, run_id=candidate)
    return {
        "schema_id": TRAINING_CROP_PUBLICATION_SCHEMA_ID,
        "schema_version": TRAINING_CROP_PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "destination": str(archive),
        "run_id": candidate,
        "run_path": bound.run_path,
        "row_count": bound.row_count,
        "roi_shape": list(bound.roi_shape),
        "binding_digest": bound.binding["payload_digest"],
        "materialization": materialization,
        "atomic_publication": publication,
        "stage_selector_eligible": False,
    }


def enrich_sampled_training_dataset(
    *,
    destination: str | Path,
    source_zarr: str | Path,
    source_crop_run: str,
    run_id: str,
    scratch_root: str | Path,
    video_path: str | Path | None = None,
    roi_cache_manifest: str | Path | None = None,
    copy_backend: str = "python",
    cache_copy_batch_rows: int = 1024,
    decode_mode: str = "auto",
    decode_chunk_frames: int = 1,
    source_instance_keys: Sequence[int] | None = None,
    detect_run_id: str | None = None,
    refined_run_id: str | None = None,
) -> dict[str, Any]:
    """Atomically add crops only after detection review is first-class."""

    archive = Path(destination).expanduser().resolve()
    base = validate_training_detection_review_base(
        archive,
        detect_run_id=detect_run_id,
        refined_run_id=refined_run_id,
    )
    validate_training_crop_source_join(
        base,
        source_zarr=source_zarr,
        source_crop_run=source_crop_run,
        source_instance_keys=(
            tuple(int(value) for value in source_instance_keys)
            if source_instance_keys is not None
            else None
        ),
    )
    crop_publication = publish_training_crop_materialization(
        destination=archive,
        source_zarr=source_zarr,
        source_crop_run=source_crop_run,
        run_id=run_id,
        scratch_root=scratch_root,
        video_path=video_path,
        roi_cache_manifest=roi_cache_manifest,
        copy_backend=copy_backend,
        cache_copy_batch_rows=cache_copy_batch_rows,
        decode_mode=decode_mode,
        decode_chunk_frames=decode_chunk_frames,
        source_instance_keys=source_instance_keys,
    )
    _base, _crop, composition = build_training_dataset_composition(
        archive,
        crop_run_id=run_id,
        detect_run_id=base.detect_run_id,
        refined_run_id=base.refined_run_id,
        require_consolidated_crop=True,
    )
    root = open_zarr_group_direct(archive, mode="a")
    root.attrs[TRAINING_DATASET_COMPOSITION_ATTRIBUTE] = composition
    with archive_metadata_publication_lock(archive):
        consolidate_metadata_capture_expected_warnings(archive)
    bound = bind_training_dataset_composition(
        archive,
        crop_run_id=run_id,
        detect_run_id=base.detect_run_id,
        refined_run_id=base.refined_run_id,
    )
    return {
        "schema_id": TRAINING_DATASET_ENRICHMENT_SCHEMA_ID,
        "schema_version": TRAINING_DATASET_ENRICHMENT_SCHEMA_VERSION,
        "status": "complete",
        "destination": str(archive),
        "detect_run": bound.base.detect_run_id,
        "refined_detect_run": bound.base.refined_run_id,
        "crop_run": run_id,
        "crop_publication": crop_publication,
        "composition_digest": bound.binding["payload_digest"],
        "stage_selector_eligible": False,
        "registry_activation": "deferred",
    }


def create_training_crop_artifact(
    *,
    destination: str | Path,
    base_training_zarr: str | Path,
    source_zarr: str | Path,
    source_crop_run: str,
    run_id: str,
    scratch_root: str | Path,
    video_path: str | Path | None = None,
    roi_cache_manifest: str | Path | None = None,
    cache_copy_batch_rows: int = 1024,
    decode_mode: str = "auto",
    decode_chunk_frames: int = 1,
    source_instance_keys: Sequence[int] | None = None,
    copy_backend: str = "python",
    detect_run_id: str | None = None,
    refined_run_id: str | None = None,
) -> dict[str, Any]:
    """Copy a sampled detection-review Zarr, enrich it, and publish it whole."""

    target = Path(destination).expanduser().resolve()
    base_archive = Path(base_training_zarr).expanduser().resolve()
    source = Path(source_zarr).expanduser().resolve()
    scratch = _require_node_local_scratch(Path(scratch_root))
    if target.suffix != ".zarr":
        raise ValueError("Training artifact destination must end in .zarr.")
    if copy_backend != "python":
        raise ValueError(
            "Whole training-artifact publication currently supports copy_backend='python' only."
        )
    if target.exists():
        raise FileExistsError(f"Training artifact already exists: {target}")
    if not base_archive.is_dir() or base_archive.suffix != ".zarr":
        raise FileNotFoundError(
            f"Base sampled training Zarr not found: {base_archive}"
        )
    base = validate_training_detection_review_base(
        base_archive,
        detect_run_id=detect_run_id,
        refined_run_id=refined_run_id,
    )
    validate_training_crop_source_join(
        base,
        source_zarr=source,
        source_crop_run=source_crop_run,
        source_instance_keys=(
            tuple(int(value) for value in source_instance_keys)
            if source_instance_keys is not None
            else None
        ),
    )
    if (video_path is None) == (roi_cache_manifest is None):
        raise ValueError(
            "Choose exactly one materialization input: video_path or roi_cache_manifest."
        )
    candidate = str(run_id).strip()
    if not candidate or "/" in candidate or candidate.startswith("."):
        raise ValueError("run_id must be one safe non-hidden child-group name.")

    with tempfile.TemporaryDirectory(
        prefix=f"palette-training-artifact-{candidate}-",
        dir=str(scratch),
    ) as temporary:
        local_archive = Path(temporary) / target.name
        shutil.copytree(base_archive, local_archive)
        local_root = open_zarr_group_direct(local_archive, mode="a")
        local_root.attrs.update(
            {
                "zarr_purpose": "training",
                "training_artifact_status": "building",
                "stage_selector_eligible": False,
            }
        )
        local_root.require_group("crop_runs")
        materialization = regenerate_training_crops_pynvvc(
            zarr_path=local_archive,
            source_zarr_path=source,
            source_crop_run=str(source_crop_run),
            target_crop_run=candidate,
            video_path=video_path,
            roi_cache_manifest=roi_cache_manifest,
            cache_copy_batch_rows=int(cache_copy_batch_rows),
            source_instance_keys=source_instance_keys,
            decode_mode=str(decode_mode),
            decode_chunk_frames=int(decode_chunk_frames),
            overwrite=False,
            set_latest=False,
            consolidate_metadata=False,
            dry_run=False,
        )
        local_root = open_zarr_group_direct(local_archive, mode="a")
        local_root.attrs["training_artifact_status"] = "complete"
        local_root.attrs["training_artifact_publication"] = {
            "schema_id": TRAINING_ARTIFACT_PUBLICATION_SCHEMA_ID,
            "schema_version": TRAINING_ARTIFACT_PUBLICATION_SCHEMA_VERSION,
            "policy": "node_local_build_then_checked_hidden_sibling_rename_v1",
            "base_training_zarr": str(base_archive),
            "source_zarr": str(source),
            "source_crop_run": str(source_crop_run),
            "crop_run": candidate,
            "stage_selector_eligible": False,
            "registry_activation": "deferred",
        }
        _base, _crop, composition = build_training_dataset_composition(
            local_archive,
            crop_run_id=candidate,
            detect_run_id=base.detect_run_id,
            refined_run_id=base.refined_run_id,
            require_consolidated_crop=False,
        )
        local_root.attrs[TRAINING_DATASET_COMPOSITION_ATTRIBUTE] = composition
        consolidate_metadata_capture_expected_warnings(local_archive)
        bind_training_dataset_composition(
            local_archive,
            crop_run_id=candidate,
            detect_run_id=base.detect_run_id,
            refined_run_id=base.refined_run_id,
        )
        local_inventory = tree_inventory(local_archive, hash_content=True)

        target.parent.mkdir(parents=True, exist_ok=True)
        hidden = target.with_name(
            f".{target.name}.publish_tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        with archive_metadata_publication_lock(target):
            if target.exists() or hidden.exists():
                raise FileExistsError(
                    f"Training artifact publication target became occupied: {target}"
                )
            try:
                shutil.copytree(local_archive, hidden)
                hidden_inventory = tree_inventory(hidden, hash_content=True)
                if hidden_inventory != local_inventory:
                    raise RuntimeError(
                        "Training artifact physical copy differs from node-local source."
                    )
                bind_training_dataset_composition(
                    hidden,
                    crop_run_id=candidate,
                    detect_run_id=base.detect_run_id,
                    refined_run_id=base.refined_run_id,
                )
                if target.exists():
                    raise FileExistsError(
                        f"Training artifact appeared during publication: {target}"
                    )
                os.replace(hidden, target)
            except Exception:
                if hidden.exists():
                    shutil.rmtree(hidden)
                raise

    final_bound = bind_training_dataset_composition(
        target,
        crop_run_id=candidate,
        detect_run_id=base.detect_run_id,
        refined_run_id=base.refined_run_id,
    )
    return {
        "schema_id": TRAINING_ARTIFACT_PUBLICATION_SCHEMA_ID,
        "schema_version": TRAINING_ARTIFACT_PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "destination": str(target),
        "run_id": candidate,
        "run_path": final_bound.crop.run_path,
        "row_count": final_bound.crop.row_count,
        "roi_shape": list(final_bound.crop.roi_shape),
        "binding_digest": final_bound.crop.binding["payload_digest"],
        "composition_digest": final_bound.binding["payload_digest"],
        "detect_run": final_bound.base.detect_run_id,
        "refined_detect_run": final_bound.base.refined_run_id,
        "physical_inventory": local_inventory.to_json(),
        "materialization": materialization,
        "stage_selector_eligible": False,
        "registry_activation": "deferred",
    }


__all__ = [
    "TRAINING_CROP_PUBLICATION_POLICY",
    "TRAINING_CROP_PUBLICATION_ROLLBACK_POLICY",
    "TRAINING_CROP_PUBLICATION_SCHEMA_ID",
    "TRAINING_CROP_PUBLICATION_SCHEMA_VERSION",
    "TRAINING_ARTIFACT_PUBLICATION_SCHEMA_ID",
    "TRAINING_ARTIFACT_PUBLICATION_SCHEMA_VERSION",
    "TRAINING_DATASET_ENRICHMENT_SCHEMA_ID",
    "TRAINING_DATASET_ENRICHMENT_SCHEMA_VERSION",
    "create_training_crop_artifact",
    "enrich_sampled_training_dataset",
    "publish_training_crop_materialization",
]
