#!/usr/bin/env python3
"""Run canonical detection on node-local storage and atomically publish it.

The source video is intentionally decoded from its canonical acquisition path.
Only the derived ``detect_runs/<run>`` group is built on node-local storage and
copied to the authoritative recording archive after complete validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.detection.candidate_builder import build_detection_candidate
from fisheye.registry.db import RegistryPaths
from fisheye.registry.stage_complete import emit_stage_completion
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.artifact_fingerprint import fingerprint_artifact
from fisheye.shared.detection_candidate import (
    DEFAULT_DETECT_FRAME_SHARD_ROWS,
    DEFAULT_DETECT_ROW_SHARD_ROWS,
    DETECTION_CANDIDATE_BUILD_AUTHORITY_ATTR,
    node_local_detection_candidate_authority,
)
from fisheye.shared.detection_model_provenance import (
    write_detect_model_resolution_provenance,
)
from fisheye.shared.immutable_yolo_storage import validate_immutable_yolo_storage
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.import_video_metadata import (
    publish_source_camera_pixel_frame_authorities,
)
from fisheye.shared.observation_coordinate_publication import (
    _load_persisted_detection_observation_geometry,
    load_persisted_detection_observation_geometry,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.run_provenance import (
    build_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.source_video_metadata import resolve_source_video
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    reconsolidate_zarr_metadata,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    require_runs_parent,
)


SCHEMA_ID = "palette.node_local_detection_publish.v1"
PUBLISH_SCHEMA_ID = "palette.node_local_detection_atomic_publish.v1"
PUBLISH_POLICY = "node_local_complete_run_then_atomic_prfs_publication_v1"
ROLLBACK_POLICY = "retain_failed_selector_ineligible_child_v1"
_SELECTOR_ATTRS = ("latest", "latest_complete", "latest_pending")
DETECTION_CONSOLIDATION_POLICY = (
    "root_after_detection_selector_activation_direct_consolidated_verified_v1"
)
DETECTION_FAILED_VISIBILITY_REPAIR_POLICY = (
    "root_after_failed_detection_activation_rollback_verified_v1"
)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid detection run name: {value!r}")
    return name


def default_node_local_detection_scratch_root() -> Path:
    """Return a job-scoped local scratch root without touching shared storage."""

    base = Path(os.environ.get("TMPDIR") or "/tmp").expanduser().resolve()
    job_id = str(os.environ.get("LSB_JOBID") or os.getpid())
    job_index = str(os.environ.get("LSB_JOBINDEX") or "0")
    return base / "palette" / "detection_publish" / f"{job_id}_{job_index}"


def default_detection_run_name() -> str:
    """Return a collision-resistant human-readable detection run name."""

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"detect_{stamp}_{uuid.uuid4().hex[:8]}"


def _resolve_detection_run_provenance(
    *,
    supplied: Mapping[str, Any] | None,
    source_zarr: Path,
    video_path: Path,
    model_path: Path,
    model_sha256: str,
    model_run_id: str,
    model_set_id: str,
    model_created_utc: str | None,
    run_name: str,
    config_path: str | None,
    conf_threshold: float | None,
    iou_threshold: float | None,
    max_det: int | None,
    batch_size: int | None,
    resize_dims: list[int] | None,
    imgsz: list[int] | None,
    decode_backend: str | None,
    detect_row_shard_rows: int | None,
    detect_frame_shard_rows: int,
    use_gpu: bool | None,
    copy_backend: str,
) -> dict[str, Any]:
    """Return completion-grade provenance before expensive inference starts."""

    candidate = (
        dict(supplied)
        if supplied is not None
        else build_run_provenance(
            command="fisheye.utils.run_detection_local_publish",
            params={
                "source_zarr": str(source_zarr),
                "video_path": str(video_path),
                "model_path": str(model_path),
                "model_sha256": model_sha256.lower(),
                "model_run_id": model_run_id,
                "model_set_id": model_set_id,
                "model_created_utc": model_created_utc,
                "run_name": run_name,
                "config_path": config_path,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "max_det": max_det,
                "batch_size": batch_size,
                "resize_dims": resize_dims,
                "imgsz": imgsz,
                "decode_backend": decode_backend,
                "detect_row_shard_rows": detect_row_shard_rows,
                "detect_frame_shard_rows": int(detect_frame_shard_rows),
                "use_gpu": use_gpu,
                "copy_backend": copy_backend,
                "source_video_policy": "stream_canonical_prfs_video_in_place_v1",
                "output_policy": PUBLISH_POLICY,
            },
            input_run_ids={
                "model_run": model_run_id,
                "model_set": model_set_id,
            },
            cwd=Path.cwd(),
        )
    )
    validation = validate_run_provenance(candidate)
    if not validation.valid:
        raise ValueError(
            "Detection publication requires valid run provenance before "
            f"inference: {'; '.join(validation.errors)}"
        )
    return validation.normalized or candidate


def _copy_attrs(source: zarr.Group, target: zarr.Group) -> None:
    target.attrs.put(copy.deepcopy(dict(source.attrs)))


def _prepare_local_overlay(source_zarr: Path, local_zarr: Path) -> dict[str, Any]:
    """Copy only acquisition metadata required for canonical detection."""

    source = open_zarr_root(source_zarr, mode="r")
    status = load_acquisition_authority_publication_status(source)
    if (
        status.status != ACQUISITION_AUTHORITY_PUBLISHED
        or status.authority_mode != EXTERNAL_ACQUISITION_AUTHORITY_MODE
        or not status.authority_path
    ):
        raise RuntimeError(
            "Node-local detection currently requires a published external-video "
            "acquisition authority."
        )
    ownership, acquisition = load_persisted_acquisition_camera_authority(source)
    ownership.assert_verified()
    acquisition.assert_verified()
    expected_authority_path = (
        f"analysis/acquisition_camera_frames/{acquisition.record.camera_id}"
    )
    if status.authority_path != expected_authority_path:
        raise RuntimeError("Acquisition status and authority path disagree.")

    raw = source.get("raw_video")
    analysis = source.get("analysis")
    acquisition_parent = source.get("analysis/acquisition_camera_frames")
    authority = source.get(expected_authority_path)
    if not all(isinstance(node, zarr.Group) for node in (raw, analysis, acquisition_parent, authority)):
        raise RuntimeError("Canonical external acquisition metadata is incomplete.")
    assert isinstance(raw, zarr.Group)
    if tuple(raw.array_keys()):
        raise RuntimeError(
            "External-video node-local detection refuses to stage raw_video arrays."
        )

    if local_zarr.exists():
        raise FileExistsError(f"Node-local output already exists: {local_zarr}")
    local = open_zarr_root(local_zarr, mode="w")
    _copy_attrs(source, local)
    local.attrs[DETECTION_CANDIDATE_BUILD_AUTHORITY_ATTR] = (
        node_local_detection_candidate_authority()
    )
    local_raw = local.create_group("raw_video")
    _copy_attrs(raw, local_raw)
    local_analysis = local.create_group("analysis")
    _copy_attrs(analysis, local_analysis)
    local_authorities = local_analysis.create_group("acquisition_camera_frames")
    assert isinstance(acquisition_parent, zarr.Group)
    _copy_attrs(acquisition_parent, local_authorities)
    local_authority = local_authorities.create_group(acquisition.record.camera_id)
    assert isinstance(authority, zarr.Group)
    _copy_attrs(authority, local_authority)

    # Reopen and prove the copied acquisition graph rather than trusting the
    # source handles used during construction.
    reopened = open_zarr_root(local_zarr, mode="r")
    copied_status = load_acquisition_authority_publication_status(reopened)
    copied_ownership, copied_acquisition = load_persisted_acquisition_camera_authority(
        reopened,
        expected_camera_id=acquisition.record.camera_id,
    )
    if (
        copied_status != status
        or copied_ownership.record_ref != ownership.record_ref
        or copied_ownership.record_sha256 != ownership.record_sha256
        or copied_acquisition.record_ref != acquisition.record_ref
        or copied_acquisition.record_sha256 != acquisition.record_sha256
    ):
        raise RuntimeError("Node-local acquisition metadata changed during staging.")
    return {
        "authority_mode": status.authority_mode,
        "authority_path": status.authority_path,
        "camera_id": acquisition.record.camera_id,
        "recording_id": acquisition.record.recording_id,
        "source_total_frames": int(acquisition.record.source_total_frames),
        "source_width_px": int(acquisition.record.width_px),
        "source_height_px": int(acquisition.record.height_px),
        "staged_raw_video_arrays": 0,
    }


def _ensure_shared_source_camera_authorities(source_zarr: Path) -> dict[str, str]:
    """Idempotently materialize deterministic archive-wide camera frames."""

    root = open_zarr_root(source_zarr, mode="a")
    return publish_source_camera_pixel_frame_authorities(root)


def _verify_model(model_path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = str(expected_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("A canonical 64-character model SHA-256 is required.")
    artifact = fingerprint_artifact(
        model_path,
        role="detect_model",
        registry_hash=expected,
    )
    actual = str(artifact.get("sha256") or "").lower()
    if actual != expected or artifact.get("mismatch") is True:
        raise RuntimeError(
            f"Registered detect model digest mismatch: expected={expected}, actual={actual or None}."
        )
    return dict(artifact)


def _validate_run_path(
    run_path: Path,
    *,
    run_name: str,
    model_path: Path,
    model_sha256: str,
    row_shard_rows: int | None,
    frame_shard_rows: int,
) -> dict[str, Any]:
    try:
        run = open_zarr_root(run_path, mode="r")
        errors: list[str] = []
        if run.attrs.get("palette_run_completion_status") != "complete":
            errors.append("run is not complete")
        if run.attrs.get("palette_run_stage") != "detect":
            errors.append("run stage is not detect")
        if run.attrs.get("palette_run_name") != run_name:
            errors.append("run name changed")
        if run.attrs.get("coordinate_contract") != "canonical_v2":
            errors.append("coordinate contract is not canonical_v2")
        if run.attrs.get("coordinate_contract_mode") != "canonical":
            errors.append("coordinate contract mode is not canonical")
        if run.attrs.get("stage_selector_eligible") not in {True, False}:
            errors.append("selector eligibility is missing")
        if Path(str(run.attrs.get("model_path") or "")).resolve() != model_path.resolve():
            errors.append("persisted model path differs from the pinned model")
        storage = validate_immutable_yolo_storage(
            run,
            stage="detect",
            row_shard_rows=row_shard_rows,
            frame_shard_rows=(frame_shard_rows if row_shard_rows is not None else None),
            persist_report=False,
        )
        provenance = run.attrs.get("run_provenance")
        artifacts = provenance.get("input_artifacts") if isinstance(provenance, Mapping) else None
        matching_model = False
        if isinstance(artifacts, list):
            matching_model = any(
                isinstance(item, Mapping)
                and item.get("role") == "detect_model"
                and str(item.get("sha256") or "").lower() == model_sha256.lower()
                for item in artifacts
            )
        if not matching_model:
            errors.append("run provenance does not bind the pinned model digest")
        return {
            "valid": not errors,
            "run_name": run_name,
            "row_count": storage.get("row_count"),
            "frame_count": storage.get("frame_count"),
            "instance_key_unique": storage.get("instance_key_unique"),
            "errors": errors,
        }
    except Exception as exc:
        return {
            "valid": False,
            "run_name": run_name,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }


@dataclass
class _DetectionActivation:
    run_name: str
    source_zarr: Path
    snapshot: dict[str, tuple[bool, Any]] | None = None
    attempted: dict[str, Any] | None = None
    visibility_report: dict[str, Any] | None = None

    def _validate_consolidated_visibility(self) -> dict[str, Any]:
        subtree_path = f"detect_runs/{self.run_name}"
        receipt = validate_direct_consolidated_subtree(
            self.source_zarr,
            subtree_path=subtree_path,
        )
        direct = zarr.open_group(
            str(self.source_zarr),
            mode="r",
            zarr_format=3,
            use_consolidated=False,
        )
        consolidated = zarr.open_group(
            str(self.source_zarr),
            mode="r",
            zarr_format=3,
            use_consolidated=True,
        )
        direct_parent = direct["detect_runs"]
        consolidated_parent = consolidated["detect_runs"]
        for name in _SELECTOR_ATTRS:
            direct_present = name in direct_parent.attrs
            consolidated_present = name in consolidated_parent.attrs
            if direct_present != consolidated_present or (
                direct_present
                and direct_parent.attrs.get(name) != consolidated_parent.attrs.get(name)
            ):
                raise RuntimeError(
                    "Detection selector differs between direct and consolidated "
                    f"metadata: {name!r}."
                )
        for label, parent in (
            ("direct", direct_parent),
            ("consolidated", consolidated_parent),
        ):
            run = parent[self.run_name]
            if (
                parent.attrs.get("latest") != self.run_name
                or parent.attrs.get("latest_complete") != self.run_name
                or run.attrs.get("palette_run_completion_status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not True
            ):
                raise RuntimeError(
                    f"{label} detection publication is not selected, complete, "
                    "and selector eligible."
                )
        return {
            "policy": DETECTION_CONSOLIDATION_POLICY,
            "subtree_equivalence": receipt.to_json(),
            "selectors": {
                name: copy.deepcopy(direct_parent.attrs.get(name))
                for name in _SELECTOR_ATTRS
                if name in direct_parent.attrs
            },
        }

    def activate(self, _root: zarr.Group, parent: zarr.Group, run: zarr.Group) -> None:
        self.snapshot = {
            name: (name in parent.attrs, copy.deepcopy(parent.attrs.get(name)))
            for name in _SELECTOR_ATTRS
        }
        attempted: dict[str, Any] = {
            "latest_complete": self.run_name,
            "latest": self.run_name,
        }
        if parent.attrs.get("latest_pending") == self.run_name:
            attempted["latest_pending"] = None
            del parent.attrs["latest_pending"]
        parent.attrs["latest_complete"] = self.run_name
        parent.attrs["latest"] = self.run_name
        self.attempted = attempted
        # Detection activation includes the fallible final publication step:
        # root consolidation and exact direct/consolidated verification.
        run.attrs["stage_selector_eligible"] = True
        consolidation = reconsolidate_zarr_metadata(
            self.source_zarr,
            policy=DETECTION_CONSOLIDATION_POLICY,
            fail_on_error=True,
        )
        self.visibility_report = {
            **self._validate_consolidated_visibility(),
            "consolidation": consolidation,
        }

    def rollback(self) -> None:
        if self.snapshot is None or self.attempted is None:
            return
        root = open_zarr_root(self.source_zarr, mode="a")
        parent = root["detect_runs"]
        failures: list[str] = []
        for name, (present, value) in self.snapshot.items():
            try:
                attempted = self.attempted.get(name, object())
                current_present = name in parent.attrs
                current_value = parent.attrs.get(name)
                owned = (
                    (attempted is None and not current_present)
                    or (attempted is not None and current_present and current_value == attempted)
                )
                if not owned:
                    continue
                if present:
                    parent.attrs[name] = copy.deepcopy(value)
                elif name in parent.attrs:
                    del parent.attrs[name]
            except Exception as exc:  # pragma: no cover - hostile persistent store
                failures.append(f"{name}: {exc}")
        if failures:
            raise RuntimeError(f"Detection selector rollback was incomplete: {failures!r}")

    def repair_failed_visibility(self, target_path: Path) -> None:
        expected_target = self.source_zarr / "detect_runs" / self.run_name
        if target_path.resolve() != expected_target.resolve():
            raise RuntimeError(
                "Detection visibility repair received an unexpected run path."
            )
        reconsolidate_zarr_metadata(
            self.source_zarr,
            policy=DETECTION_FAILED_VISIBILITY_REPAIR_POLICY,
            fail_on_error=True,
        )
        if self.snapshot is None:
            return
        direct = zarr.open_group(
            str(self.source_zarr),
            mode="r",
            zarr_format=3,
            use_consolidated=False,
        )["detect_runs"]
        consolidated = zarr.open_group(
            str(self.source_zarr),
            mode="r",
            zarr_format=3,
            use_consolidated=True,
        )["detect_runs"]
        for name, (present, value) in self.snapshot.items():
            for label, parent in (("direct", direct), ("consolidated", consolidated)):
                if (name in parent.attrs) is not present or (
                    present and parent.attrs.get(name) != value
                ):
                    raise RuntimeError(
                        f"{label} detection selector rollback differs for {name!r}."
                    )


def _resolution_payload(
    *,
    registry_path: Path,
    recording_id: str,
    model_path: Path,
    model_sha256: str,
    model_run_id: str,
    model_set_id: str,
    model_created_utc: str | None,
) -> dict[str, Any]:
    selected = {
        "run_id": model_run_id,
        "set_id": model_set_id,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "created_utc": model_created_utc,
        "score": None,
        "selection_policy": "newest_successful_registered_detect_model_v1",
    }
    return {
        "schema_id": "palette.model_resolution.v1",
        "mode": "registry",
        "tool": "fisheye.utils.run_detection_local_publish",
        "task": "detect",
        "registry_path": str(registry_path),
        "recording_id": recording_id,
        "selected": selected,
        "candidates": [selected],
        "parameters": {
            "selection_policy": "newest_successful_registered_detect_model_v1",
            "campaign_pinned": True,
        },
        "artifacts": {"selected_model": selected},
    }


def run_detection_local_publish(
    *,
    source_zarr: str | Path,
    model_path: str | Path,
    model_sha256: str,
    model_run_id: str,
    model_set_id: str,
    run_name: str | None = None,
    scratch_root: str | Path | None = None,
    registry_path: str | Path,
    model_created_utc: str | None = None,
    video_path: str | Path | None = None,
    config_path: str | None = None,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
    max_det: int | None = None,
    batch_size: int | None = None,
    resize_dims: list[int] | None = None,
    imgsz: list[int] | None = None,
    decode_backend: str | None = None,
    detect_row_shard_rows: int | None = DEFAULT_DETECT_ROW_SHARD_ROWS,
    detect_frame_shard_rows: int = DEFAULT_DETECT_FRAME_SHARD_ROWS,
    use_gpu: bool | None = None,
    copy_backend: str = "python",
    keep_scratch: bool = False,
    model_resolution_payload: Mapping[str, Any] | None = None,
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one raw detection run locally and publish it fail-closed."""

    started = time.perf_counter()
    source = Path(source_zarr).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    scratch = Path(
        scratch_root
        if scratch_root is not None
        else default_node_local_detection_scratch_root()
    ).expanduser().resolve()
    name = _safe_name(run_name or default_detection_run_name())
    if not source.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {source}")
    if not model.is_file():
        raise FileNotFoundError(f"Detect model not found: {model}")
    if not registry.is_file():
        raise FileNotFoundError(f"Registry not found: {registry}")
    if str(scratch).startswith("/groups/") or str(scratch) == "/groups":
        raise ValueError("Node-local detection scratch must not be on /groups.")
    if detect_row_shard_rows is not None and int(detect_row_shard_rows) <= 0:
        raise ValueError("detect_row_shard_rows must be positive or None")
    if int(detect_frame_shard_rows) <= 0:
        raise ValueError("detect_frame_shard_rows must be positive")

    model_artifact = _verify_model(model, model_sha256)
    canonical_root = open_zarr_root(source, mode="r")
    resolved_video = resolve_source_video(
        canonical_root,
        zarr_path=source,
        require_exists=True,
    ).path.resolve()
    if video_path is not None and Path(video_path).expanduser().resolve() != resolved_video:
        raise ValueError(
            "Requested video path does not equal the canonical acquisition locator."
        )
    recording_id = str(canonical_root.attrs.get("recording_id") or "").strip()
    if not recording_id:
        raise RuntimeError("Canonical archive has no recording_id.")
    effective_run_provenance = _resolve_detection_run_provenance(
        supplied=run_provenance,
        source_zarr=source,
        video_path=resolved_video,
        model_path=model,
        model_sha256=model_sha256,
        model_run_id=model_run_id,
        model_set_id=model_set_id,
        model_created_utc=model_created_utc,
        run_name=name,
        config_path=config_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
        batch_size=batch_size,
        resize_dims=resize_dims,
        imgsz=imgsz,
        decode_backend=decode_backend,
        detect_row_shard_rows=detect_row_shard_rows,
        detect_frame_shard_rows=int(detect_frame_shard_rows),
        use_gpu=use_gpu,
        copy_backend=copy_backend,
    )
    target_run = source / "detect_runs" / name
    if target_run.exists():
        raise FileExistsError(f"Canonical detect run already exists: {target_run}")

    work = scratch / f"palette_detect_{name}_{uuid.uuid4().hex[:12]}"
    local_zarr = work / "analysis.zarr"
    local_run = local_zarr / "detect_runs" / name
    work.mkdir(parents=True, exist_ok=False)
    success = False
    try:
        overlay = _prepare_local_overlay(source, local_zarr)
        resolution = (
            copy.deepcopy(dict(model_resolution_payload))
            if model_resolution_payload is not None
            else _resolution_payload(
                registry_path=registry,
                recording_id=recording_id,
                model_path=model,
                model_sha256=model_sha256.lower(),
                model_run_id=model_run_id,
                model_set_id=model_set_id,
                model_created_utc=model_created_utc,
            )
        )
        detector_started = time.perf_counter()
        detected_name = build_detection_candidate(
            video_path=str(resolved_video),
            model_path=str(model),
            output_zarr=str(local_zarr),
            config_path=config_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            batch_size=batch_size,
            resize_dims=resize_dims,
            imgsz=imgsz,
            decode_backend=decode_backend,
            use_gpu=use_gpu,
            write_raw_video_metadata=False,
            overwrite_raw_video_metadata=False,
            run_name=name,
            model_sha256=model_sha256.lower(),
            detect_row_shard_rows=detect_row_shard_rows,
            detect_frame_shard_rows=int(detect_frame_shard_rows),
            run_provenance=effective_run_provenance,
        )
        if detected_name != name or not local_run.is_dir():
            raise RuntimeError("Detector returned a different or absent local run.")
        write_detect_model_resolution_provenance(
            zarr_path=local_zarr,
            run_name=name,
            payload=resolution,
        )
        detector_seconds = time.perf_counter() - detector_started

        def validator(path: Path) -> dict[str, Any]:
            return _validate_run_path(
                path,
                run_name=name,
                model_path=model,
                model_sha256=model_sha256,
                row_shard_rows=detect_row_shard_rows,
                frame_shard_rows=int(detect_frame_shard_rows),
            )
        local_validation = validator(local_run)
        if not local_validation.get("valid"):
            raise RuntimeError(f"Local detection validation failed: {local_validation}")

        shared_authorities = _ensure_shared_source_camera_authorities(source)
        activation = _DetectionActivation(name, source)

        def prepare_parents(root: zarr.Group) -> tuple[zarr.Group, ...]:
            return (
                require_runs_parent(
                    root,
                    "detect_runs",
                    completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
                ),
            )

        def after_rename(root: zarr.Group, run: zarr.Group) -> dict[str, Any]:
            proof = _load_persisted_detection_observation_geometry(
                root,
                f"detect_runs/{name}",
                require_selector_eligible=False,
            )
            return {
                "staged_coordinate_proof": {
                    "row_identity_record_ref": proof.row_identity.record_ref,
                    "row_identity_record_sha256": proof.row_identity.record_sha256,
                    "temporal_record_ref": proof.temporal_authority.record_ref,
                    "temporal_record_sha256": proof.temporal_authority.record_sha256,
                }
            }

        def complete_run(
            _root: zarr.Group,
            _parent: zarr.Group,
            run: zarr.Group,
        ) -> None:
            if (
                run.attrs.get("palette_run_completion_status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError("Published detection is not complete and staged.")

        def verify_pointers(root: zarr.Group) -> None:
            parent = root["detect_runs"]
            run = parent[name]
            if run.attrs.get("stage_selector_eligible") is not False:
                raise RuntimeError("Detection became eligible before activation.")
            collisions = [
                attr for attr in _SELECTOR_ATTRS if parent.attrs.get(attr) == name
            ]
            if collisions:
                raise RuntimeError(
                    f"Detection selectors reference the staged candidate: {collisions!r}"
                )

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=source,
                local_run_path=local_run,
                target_run_path=target_run,
                run_name=name,
                lock_suffix="detect_local_publish",
                publish_schema_id=PUBLISH_SCHEMA_ID,
                policy=PUBLISH_POLICY,
                rollback_policy=ROLLBACK_POLICY,
                content_checksum=False,
            ),
            copy_backend=copy_backend,
            validate_run=validator,
            prepare_parents=prepare_parents,
            after_rename=after_rename,
            complete_run=complete_run,
            verify_pointers=verify_pointers,
            activate_run=activation.activate,
            rollback_activation=activation.rollback,
            repair_failed_publication_visibility=(
                activation.repair_failed_visibility
            ),
            accept_persisted_activation_on_callback_error=False,
            payload_metadata={
                "source_video_policy": "stream_canonical_prfs_video_in_place_v1",
                "source_video_path": str(resolved_video),
                "node_local_workspace": str(work),
                "model": model_artifact,
                "model_resolution": resolution,
                "overlay": overlay,
                "shared_source_camera_authorities": shared_authorities,
                "detector_seconds": detector_seconds,
            },
        )
        if activation.visibility_report is None:
            raise RuntimeError(
                "Detection activation completed without consolidated visibility proof."
            )
        publication["activation_visibility"] = json_attr_safe(
            activation.visibility_report
        )

        final_root = zarr.open_group(
            str(source),
            mode="r",
            zarr_format=3,
            use_consolidated=True,
        )
        final_proof = load_persisted_detection_observation_geometry(
            final_root,
            f"detect_runs/{name}",
        )
        final_parent = final_root["detect_runs"]
        if (
            final_parent.attrs.get("latest") != name
            or final_parent.attrs.get("latest_complete") != name
        ):
            raise RuntimeError("Final detection selectors did not persist.")
        registry_updated = emit_stage_completion(
            final_root,
            source,
            step_name="detect",
            status="ok",
            source="runtime_node_local_detection_publish",
            run_name=name,
            method="yolo",
            details_json={
                "model_path": str(model),
                "model_sha256": model_sha256.lower(),
                "model_run_id": model_run_id,
                "model_set_id": model_set_id,
                "source_video_policy": "stream_canonical_prfs_video_in_place_v1",
            },
            registry=registry,
            auto_registry_from_env=False,
            invalidate_on_ok=True,
            trigger_run_name=name,
        )
        result = json_attr_safe(
            {
                "schema_id": SCHEMA_ID,
                "status": "ok",
                "source_zarr": str(source),
                "video_path": str(resolved_video),
                "source_video_policy": "stream_canonical_prfs_video_in_place_v1",
                "local_workspace": str(work),
                "local_workspace_removed": not keep_scratch,
                "run_name": name,
                "target_run_path": str(target_run),
                "model_path": str(model),
                "model_sha256": model_sha256.lower(),
                "model_run_id": model_run_id,
                "model_set_id": model_set_id,
                "registry_updated": bool(registry_updated),
                "detector_seconds": detector_seconds,
                "total_seconds": time.perf_counter() - started,
                "final_coordinate_proof": {
                    "row_identity_record_ref": final_proof.row_identity.record_ref,
                    "row_identity_record_sha256": final_proof.row_identity.record_sha256,
                    "temporal_record_ref": final_proof.temporal_authority.record_ref,
                    "temporal_record_sha256": final_proof.temporal_authority.record_sha256,
                },
                "publication": publication,
            }
        )
        success = True
        return result
    finally:
        if work.exists() and not keep_scratch and success:
            shutil.rmtree(work)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--model-set-id", required=True)
    parser.add_argument("--model-created-utc")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="Node-local scratch root (default: job-scoped path under $TMPDIR).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry SQLite path (default: Palette registry environment).",
    )
    parser.add_argument("--config")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--max-det", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resize-dims", nargs="+", type=int)
    parser.add_argument("--imgsz", nargs="+", type=int)
    parser.add_argument(
        "--decode-backend",
        choices=(
            "auto",
            "pynvvc_nv12_rgb",
            "pynvvc_luma_rgb",
            "decord_gpu",
            "decord_cpu",
            "opencv",
        ),
    )
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--detect-row-shard-rows",
        type=int,
        default=DEFAULT_DETECT_ROW_SHARD_ROWS,
    )
    storage.add_argument(
        "--no-detect-sharding",
        action="store_const",
        dest="detect_row_shard_rows",
        const=None,
    )
    parser.add_argument(
        "--detect-frame-shard-rows",
        type=int,
        default=DEFAULT_DETECT_FRAME_SHARD_ROWS,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    registry = (
        args.registry or RegistryPaths.from_env(Path.cwd()).path
    ).expanduser().resolve()
    try:
        result = run_detection_local_publish(
            source_zarr=args.zarr,
            video_path=args.video,
            model_path=args.model,
            model_sha256=args.model_sha256,
            model_run_id=args.model_run_id,
            model_set_id=args.model_set_id,
            model_created_utc=args.model_created_utc,
            run_name=args.run_name,
            scratch_root=args.scratch_root,
            registry_path=registry,
            config_path=args.config,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            resize_dims=args.resize_dims,
            imgsz=args.imgsz,
            decode_backend=args.decode_backend,
            detect_row_shard_rows=args.detect_row_shard_rows,
            detect_frame_shard_rows=args.detect_frame_shard_rows,
            use_gpu=False if args.cpu else None,
            copy_backend=args.copy_backend,
            keep_scratch=args.keep_scratch,
        )
    except Exception as exc:
        result = {
            "schema_id": SCHEMA_ID,
            "status": "failed",
            "source_zarr": str(args.zarr),
            "run_name": args.run_name,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.result_json:
            write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    if args.result_json:
        write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
