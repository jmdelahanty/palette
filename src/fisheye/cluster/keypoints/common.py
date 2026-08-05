"""Keypoint-family bindings and deterministic LSF job builders."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from fisheye.cluster.lsf import LsfDependency, LsfJob, LsfResources
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_USER_TOKEN,
    build_runtime_command,
)
from fisheye.registry.db import Registry
from fisheye.registry.model_resolution import (
    load_candidates,
    load_target_profile,
    resolve_recording_id,
)
from fisheye.shared.flat_roi_cache import (
    crop_run_name_from_manifest,
    load_flat_roi_cache_manifest,
)
from fisheye.shared.roi_pixel_contract import normalize_pixel_contract
from fisheye.shared.run_provenance import json_ready
from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_LEGACY_PROFILE,
    CROP_RUN_REFERENCE_SCHEMA_ID,
    CROP_RUN_REFERENCE_SCHEMA_VERSION,
    CROP_RUN_REFERENCE_SIGNED_PROFILE,
    authoritative_crop_roi_pixel_contract,
    build_crop_run_reference,
    strict_crop_fixed_roi_shape,
    validate_crop_run_reference,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


DEFAULT_ZEBRAFISH_MIN_ROI_SIZE = 348
DEFAULT_KEYPOINT_ROI_SHARD_ROWS = 131_072
DEFAULT_KEYPOINT_FRAME_SHARD_ROWS = 131_072


def resolve_keypoint_storage(
    *,
    roi_shard_rows: int | None,
    frame_shard_rows: int,
) -> dict[str, Any]:
    """Return requested/effective immutable keypoint storage provenance."""

    if roi_shard_rows is not None and int(roi_shard_rows) <= 0:
        raise ValueError("Keypoint ROI shard rows must be positive or None.")
    if int(frame_shard_rows) <= 0:
        raise ValueError("Keypoint frame shard rows must be positive.")
    enabled = roi_shard_rows is not None
    return {
        "requested": {
            "keypoint_roi_shard_rows": (
                int(roi_shard_rows) if roi_shard_rows is not None else None
            ),
            "keypoint_frame_shard_rows": int(frame_shard_rows),
            "no_keypoint_sharding": not enabled,
        },
        "effective": {
            "keypoint_storage_layout": (
                "indexed_sharding_v1" if enabled else "regular_chunks_v1"
            ),
            "keypoint_storage_policy": (
                "default_indexed_sharding_v1"
                if enabled
                else "explicit_regular_chunks_override"
            ),
            "keypoint_roi_shard_rows": (
                int(roi_shard_rows) if roi_shard_rows is not None else None
            ),
            "keypoint_frame_shard_rows": int(frame_shard_rows) if enabled else None,
        },
    }


def resolve_keypoint_v2_publication_storage(
    *,
    legacy_roi_shard_rows: int | None,
    legacy_frame_shard_rows: int,
) -> dict[str, Any]:
    """Describe strict v2 publication, which derives chunks from bytes and dtype.

    The row arguments remain accepted by the planner for command-line
    compatibility, but apply only to obsolete direct-writer layouts.  Terminal
    inference is deliberately unsharded; strict finalization replans every
    canonical array through the shared byte planner.
    """

    if legacy_roi_shard_rows is not None and int(legacy_roi_shard_rows) <= 0:
        raise ValueError("Legacy keypoint ROI shard rows must be positive or None.")
    if int(legacy_frame_shard_rows) <= 0:
        raise ValueError("Legacy keypoint frame shard rows must be positive.")
    return {
        "requested": {
            "legacy_keypoint_roi_shard_rows": (
                int(legacy_roi_shard_rows)
                if legacy_roi_shard_rows is not None
                else None
            ),
            "legacy_keypoint_frame_shard_rows": int(legacy_frame_shard_rows),
            "effect_on_v2_publication": "none",
        },
        "effective": {
            "keypoint_storage_layout": "shared_byte_planned_indexed_sharding_v1",
            "keypoint_storage_policy": "strict_keypoint_v2_published_http_v1",
            "storage_profile": PUBLISHED_HTTP_V1.as_manifest(),
            "chunk_derivation": "dtype_itemsize_times_per_row_shape_to_byte_budget",
            "access_classes": {
                "frame_row_offsets": "eager",
                "observation_columns": "windowed",
            },
            "terminal_inference_layout": "unsharded_non_authoritative",
            "canonical_publication_planner": (
                "fisheye.shared.zarr.keypoint_storage.plan_keypoint_storage"
            ),
        },
    }


@dataclass(frozen=True)
class KeypointRunNames:
    keypoint_run: str
    keypoint_quality_run: str
    refined_keypoint_run: str
    body_frame_run: str
    terminal_run: str

    def to_json(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PoseModelBinding:
    recording_id: str
    set_id: str
    run_id: str
    model_path: Path
    model_sha256: str
    score: float
    created_utc: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "model_path": str(self.model_path),
        }


@dataclass(frozen=True)
class FlatRoiCacheBinding:
    manifest_path: Path
    manifest_sha256: str | None
    payload_path: Path
    crop_run: str
    cache_key: str | None
    crop_signature: Any
    crop_revision: Any
    shape: tuple[int, int, int]
    total_bytes: int
    payload_sha256: str | None
    crop_run_reference: Mapping[str, Any] | None = None
    pixel_contract: Mapping[str, Any] | None = None
    availability: str = "existing"
    producer_job_key: str | None = None
    source_kind: str | None = None
    nvdec_bundle_eligible: bool = False
    nvdec_bundle_reason: str | None = None

    def __post_init__(self) -> None:
        if self.availability not in {"existing", "planned"}:
            raise ValueError(
                "Flat ROI cache availability must be 'existing' or 'planned'."
            )
        if self.availability == "existing" and not self.manifest_sha256:
            raise ValueError("An existing flat ROI cache requires a manifest digest.")
        if self.availability == "planned" and not self.producer_job_key:
            raise ValueError("A planned flat ROI cache requires a producer job key.")
        reference = self.crop_run_reference
        if reference is None:
            if (
                self.crop_signature is None
                or self.crop_signature == ""
                or self.crop_revision is None
                or self.crop_revision == ""
            ):
                raise ValueError(
                    "Flat ROI cache binding requires crop_run_reference or the "
                    "historical crop_signature/crop_revision pair."
                )
            reference = {
                "schema_id": CROP_RUN_REFERENCE_SCHEMA_ID,
                "schema_version": CROP_RUN_REFERENCE_SCHEMA_VERSION,
                "profile": CROP_RUN_REFERENCE_LEGACY_PROFILE,
                "run_id": self.crop_run,
                "crop_signature": self.crop_signature,
                "crop_revision": self.crop_revision,
            }
        validated_reference = validate_crop_run_reference(reference)
        if validated_reference["run_id"] != self.crop_run:
            raise ValueError("Flat ROI cache reference binds a different crop run.")
        object.__setattr__(self, "crop_run_reference", validated_reference)
        normalized_pixel_contract = normalize_pixel_contract(self.pixel_contract)
        if self.pixel_contract is not None and normalized_pixel_contract is None:
            raise ValueError("Flat ROI cache pixel_contract is malformed.")
        object.__setattr__(self, "pixel_contract", normalized_pixel_contract)

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "manifest_path": str(self.manifest_path),
            "payload_path": str(self.payload_path),
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class KeypointInputCapability:
    selected_source: str
    min_roi_size: int
    crop_run: str
    crop_storage_mode: str
    crop_shape: tuple[int, int, int]
    persisted_roi_images_available: bool
    persisted_roi_images_eligible: bool
    acquisition_crop_video_available: bool
    acquisition_crop_video_eligible: bool
    acquisition_crop_video_path: Path | None
    acquisition_crop_video_shape: tuple[int, int] | None
    acquisition_crop_video_probe_status: str
    flat_roi_cache_eligible: bool
    rejected_sources: Mapping[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "crop_shape": list(self.crop_shape),
            "acquisition_crop_video_path": (
                str(self.acquisition_crop_video_path)
                if self.acquisition_crop_video_path is not None
                else None
            ),
            "acquisition_crop_video_shape": (
                list(self.acquisition_crop_video_shape)
                if self.acquisition_crop_video_shape is not None
                else None
            ),
            "rejected_sources": dict(self.rejected_sources),
        }


def safe_component(value: object, *, default: str, max_length: int = 72) -> str:
    text = str(value or "").strip()
    normalized = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in text
    ).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized or default
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[: max_length - len(digest) - 1]}_{digest}"


def build_keypoint_run_names(run_label: str) -> KeypointRunNames:
    safe_label = safe_component(run_label, default="keypoints")
    return KeypointRunNames(
        keypoint_run=f"keypoints_{safe_label}",
        keypoint_quality_run=f"keypoint_quality_{safe_label}",
        refined_keypoint_run=f"refined_keypoints_{safe_label}",
        body_frame_run=f"body_frame_{safe_label}",
        terminal_run=f"keypoint_terminal_{safe_label}",
    )


def resolve_pose_model_binding(
    *,
    registry_path: Path,
    recording_id: str,
    recording_dir: Path,
    set_id: str,
    run_id: str,
) -> PoseModelBinding:
    """Resolve and validate one exact registered pose-model deployment."""

    resolved_registry = registry_path.expanduser().resolve()
    resolved_recording_dir = recording_dir.expanduser().resolve()
    registry = Registry(resolved_registry)
    try:
        recording_row = registry.conn.execute(
            "SELECT recording_path FROM recordings WHERE recording_id = ?;",
            (str(recording_id),),
        ).fetchone()
        if recording_row is None:
            raise ValueError(f"Recording id is not registered: {recording_id}")
        raw_recording_path = str(recording_row["recording_path"] or "").strip()
        if not raw_recording_path:
            raise ValueError(
                f"Recording id {recording_id!r} has no registered recording_path."
            )
        registered_path = Path(raw_recording_path).expanduser().resolve()
        if registered_path != resolved_recording_dir:
            raise ValueError(
                f"Recording path mismatch for {recording_id}: registry has "
                f"{registered_path}, manifest has {resolved_recording_dir}."
            )
        resolved_recording_id = resolve_recording_id(
            registry,
            recording_id=str(recording_id),
            recording_dir=resolved_recording_dir,
        )
        target = load_target_profile(registry, resolved_recording_id)
        candidates = load_candidates(
            registry,
            target=target,
            task="pose",
            set_id_filter=str(set_id),
            include_non_success=False,
        )
        exact = [candidate for candidate in candidates if candidate.run_id == str(run_id)]
    finally:
        registry.close()

    if len(exact) != 1:
        raise ValueError(
            f"Expected one successful pose model for set {set_id!r} and run "
            f"{run_id!r}; found {len(exact)} for recording {recording_id!r}."
        )
    selected = exact[0]
    model_path = Path(selected.model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Registered pose model artifact not found: {model_path}")
    model_sha256 = str(selected.model_sha256 or "").strip()
    if not model_sha256:
        raise ValueError(
            f"Registered pose model run {selected.run_id!r} has no model_sha256; "
            "refusing an unpinned deployment."
        )
    return PoseModelBinding(
        recording_id=resolved_recording_id,
        set_id=selected.set_id,
        run_id=selected.run_id,
        model_path=model_path,
        model_sha256=model_sha256,
        score=float(selected.weighted_score),
        created_utc=selected.created_utc,
    )


def validate_registered_analysis_zarr(
    *,
    registry_path: Path,
    recording_id: str,
    analysis_zarr: Path,
) -> None:
    resolved_zarr = analysis_zarr.expanduser().resolve()
    registry = Registry(registry_path.expanduser().resolve())
    try:
        rows = registry.conn.execute(
            """
            SELECT dataset_id
            FROM datasets
            WHERE recording_id = ?
              AND zarr_path = ?
              AND LOWER(COALESCE(zarr_use, '')) = 'analysis';
            """,
            (str(recording_id), str(resolved_zarr)),
        ).fetchall()
    finally:
        registry.close()
    if not rows:
        raise ValueError(
            f"Analysis Zarr is not registered to recording {recording_id!r}: "
            f"{resolved_zarr}"
        )


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Flat ROI cache manifest field {field!r} must be an object.")
    return value


def validate_flat_roi_cache_binding(
    *,
    manifest_path: Path,
    analysis_zarr: Path,
    crop_run: str | None,
    min_roi_size: int = DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
) -> FlatRoiCacheBinding:
    """Validate manifest identity and payload size without opening the Zarr."""

    resolved_manifest = manifest_path.expanduser().resolve()
    resolved_zarr = analysis_zarr.expanduser().resolve()
    manifest = load_flat_roi_cache_manifest(resolved_manifest)
    manifest_sha256 = hashlib.sha256(resolved_manifest.read_bytes()).hexdigest()
    manifest_crop_run = crop_run_name_from_manifest(resolved_manifest)
    requested_crop_run = str(crop_run).strip() if crop_run is not None else manifest_crop_run
    if requested_crop_run != manifest_crop_run:
        raise ValueError(
            f"Flat ROI cache crop run mismatch: manifest has {manifest_crop_run!r}, "
            f"target requests {requested_crop_run!r}."
        )
    source = _mapping(manifest.get("source"), field="source")
    source_archive = str(source.get("archive_path") or "").strip()
    if not source_archive:
        raise ValueError("Flat ROI cache manifest is missing source.archive_path.")
    if Path(source_archive).expanduser().resolve() != resolved_zarr:
        raise ValueError(
            f"Flat ROI cache archive mismatch: manifest has {source_archive!r}, "
            f"target is {resolved_zarr}."
        )

    array = _mapping(manifest.get("array"), field="array")
    raw_shape = array.get("shape")
    if not isinstance(raw_shape, list) or len(raw_shape) != 3:
        raise ValueError("Flat ROI cache manifest array.shape must be [N, H, W].")
    try:
        shape = tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Flat ROI cache manifest array.shape must contain integers."
        ) from exc
    if any(value <= 0 for value in shape):
        raise ValueError("Flat ROI cache manifest array.shape values must be positive.")
    dtype_name = str(array.get("dtype") or "")
    if dtype_name != "uint8":
        raise ValueError(
            f"Unsupported flat ROI cache dtype {dtype_name!r}. Expected uint8."
        )
    order = str(array.get("order") or "C")
    if order != "C":
        raise ValueError(
            f"Unsupported flat ROI cache order {order!r}. Expected C."
        )
    raw_payload_path = str(array.get("bin_path") or "").strip()
    if not raw_payload_path:
        raise ValueError("Flat ROI cache manifest is missing array.bin_path.")
    payload_path = Path(raw_payload_path).expanduser()
    if not payload_path.is_absolute():
        payload_path = resolved_manifest.parent / payload_path
    payload_path = payload_path.resolve()
    if not payload_path.is_file():
        raise FileNotFoundError(f"Flat ROI cache payload not found: {payload_path}")
    expected_bytes = int(shape[0]) * int(shape[1]) * int(shape[2])
    actual_bytes = payload_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            "Flat ROI cache payload size mismatch: "
            f"expected {expected_bytes} bytes, got {actual_bytes}."
        )
    cache_key = str(manifest.get("cache_key") or "").strip()
    crop_signature = source.get("crop_signature")
    crop_revision = source.get("crop_revision")
    if not cache_key:
        raise ValueError("Flat ROI cache manifest is missing cache_key.")
    raw_reference = source.get("crop_run_reference")
    if raw_reference is None:
        if (
            crop_signature is None
            or crop_signature == ""
            or crop_revision is None
            or crop_revision == ""
        ):
            raise ValueError(
                "Flat ROI cache manifest must pin source.crop_run_reference or "
                "the historical source.crop_signature/crop_revision pair."
            )
        raw_reference = {
            "schema_id": CROP_RUN_REFERENCE_SCHEMA_ID,
            "schema_version": CROP_RUN_REFERENCE_SCHEMA_VERSION,
            "profile": CROP_RUN_REFERENCE_LEGACY_PROFILE,
            "run_id": requested_crop_run,
            "crop_signature": crop_signature,
            "crop_revision": crop_revision,
        }
    crop_run_reference = validate_crop_run_reference(raw_reference)
    if crop_run_reference["run_id"] != requested_crop_run:
        raise ValueError(
            "Flat ROI cache crop_run_reference does not bind the requested run."
        )
    if crop_run_reference["profile"] in {
        CROP_RUN_REFERENCE_SIGNED_PROFILE,
        CROP_RUN_REFERENCE_LEGACY_PROFILE,
    }:
        if (
            crop_run_reference["crop_signature"] != crop_signature
            or crop_run_reference["crop_revision"] != crop_revision
        ):
            raise ValueError(
                "Flat ROI cache signed crop reference disagrees with its source fields."
            )
    if int(shape[1]) < int(min_roi_size) or int(shape[2]) < int(min_roi_size):
        raise ValueError(
            f"Flat ROI cache is {shape[2]}x{shape[1]}, but zebrafish keypoint "
            f"inference requires at least {int(min_roi_size)}x{int(min_roi_size)}."
        )
    total_bytes = int(array.get("total_bytes") or actual_bytes)
    if actual_bytes != total_bytes:
        raise ValueError(
            f"Flat ROI cache total_bytes mismatch: manifest has {total_bytes}, "
            f"payload has {actual_bytes}."
        )
    builder = manifest.get("builder")
    pixel_contract = (
        normalize_pixel_contract(builder.get("pixel_contract"))
        if isinstance(builder, Mapping)
        else None
    )
    if (
        isinstance(builder, Mapping)
        and builder.get("pixel_contract") is not None
        and pixel_contract is None
    ):
        raise ValueError("Flat ROI cache builder.pixel_contract is malformed.")
    return FlatRoiCacheBinding(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_sha256,
        payload_path=payload_path,
        crop_run=requested_crop_run,
        cache_key=cache_key,
        crop_signature=crop_signature,
        crop_revision=crop_revision,
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        total_bytes=total_bytes,
        payload_sha256=(str(array.get("sha256")) if array.get("sha256") else None),
        crop_run_reference=crop_run_reference,
        pixel_contract=pixel_contract,
        availability="existing",
        producer_job_key=None,
        source_kind=(
            str(source.get("frame_source_kind"))
            if source.get("frame_source_kind")
            else None
        ),
        nvdec_bundle_eligible=False,
        nvdec_bundle_reason="cache_already_exists",
    )


def _canonical_identity(value: object) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def _crop_run_references_match(
    live: Mapping[str, Any],
    cached: Mapping[str, Any],
) -> bool:
    if _canonical_identity(live) == _canonical_identity(cached):
        return True
    # Existing cache manifests serialized maintained acquisition sources with
    # the old legacy-named profile. Preserve those immutable caches when the
    # signed fields and exact pixel contract still match; new caches write the
    # signed-current profile.
    if (
        live.get("profile") == CROP_RUN_REFERENCE_SIGNED_PROFILE
        and cached.get("profile") == CROP_RUN_REFERENCE_LEGACY_PROFILE
    ):
        fields = ("run_id", "crop_signature", "crop_revision")
        return all(live.get(field) == cached.get(field) for field in fields)
    return False


def _crop_storage_mode(crop_group: Any) -> str:
    explicit = str(crop_group.attrs.get("crop_storage_mode") or "").strip()
    if explicit in {"materialized", "geometry_only"}:
        return explicit
    return "materialized" if "roi_images" in crop_group else "geometry_only"


def _crop_roi_shape(crop_group: Any, *, crop_run: str) -> tuple[int, int]:
    strict_shape = strict_crop_fixed_roi_shape(crop_group, run_id=crop_run)
    if strict_shape is not None:
        return strict_shape
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    roi_images = crop_group.get("roi_images")
    shape = getattr(roi_images, "shape", ())
    if len(shape) >= 3:
        return int(shape[1]), int(shape[2])
    raise ValueError("Crop run is missing a fixed roi_size and persisted roi_images shape.")


def _probe_video_shape(path: Path) -> tuple[tuple[int, int] | None, str]:
    try:
        completed = subprocess.run(
            (
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, "ffprobe_missing"
    if completed.returncode != 0:
        return None, "ffprobe_failed"
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        height = int(stream["height"])
        width = int(stream["width"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None, "ffprobe_invalid_output"
    return (height, width), "ok"


def validate_keypoint_input_dag(
    *,
    analysis_zarr: Path,
    cache: FlatRoiCacheBinding,
    min_roi_size: int = DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
) -> KeypointInputCapability:
    """Prove that a cache-backed keypoint job is runnable from the live crop DAG."""

    resolved_zarr = analysis_zarr.expanduser().resolve()
    root = open_zarr_group_direct(resolved_zarr, mode="r")
    crop_parent = root.get("crop_runs")
    if crop_parent is None or cache.crop_run not in crop_parent:
        raise ValueError(
            f"Live crop DAG is missing crop_runs/{cache.crop_run} required by the cache."
        )
    crop_group = crop_parent[cache.crop_run]
    if not is_run_complete_in_parent(crop_parent, crop_group):
        raise ValueError(f"Live crop DAG node crop_runs/{cache.crop_run} is not complete.")

    if cache.crop_run_reference is None:
        raise ValueError("Flat cache binding lacks its crop_run_reference.")
    live_reference = build_crop_run_reference(
        crop_group,
        run_id=cache.crop_run,
    )
    if not _crop_run_references_match(live_reference, cache.crop_run_reference):
        raise ValueError(
            f"Flat cache crop_run_reference does not match crop_runs/{cache.crop_run}."
        )
    required_pixel_contract = authoritative_crop_roi_pixel_contract(
        crop_group,
        run_id=cache.crop_run,
    )
    if (
        required_pixel_contract is not None
        and cache.pixel_contract != required_pixel_contract
    ):
        raise ValueError(
            "Flat cache pixel_contract does not match the authoritative crop pixel "
            f"authority for crop_runs/{cache.crop_run}."
        )

    if "roi_coordinates_full" not in crop_group:
        raise ValueError(f"crop_runs/{cache.crop_run} is missing roi_coordinates_full.")
    if "frame_indices" not in crop_group:
        raise ValueError(f"crop_runs/{cache.crop_run} is missing frame_indices lineage.")
    total_rows = int(crop_group["roi_coordinates_full"].shape[0])
    frame_rows = int(crop_group["frame_indices"].shape[0])
    roi_height, roi_width = _crop_roi_shape(
        crop_group,
        crop_run=cache.crop_run,
    )
    live_shape = (total_rows, roi_height, roi_width)
    if frame_rows != total_rows:
        raise ValueError(
            f"crop_runs/{cache.crop_run} frame_indices has {frame_rows} rows, "
            f"but roi_coordinates_full has {total_rows}."
        )
    if tuple(cache.shape) != live_shape:
        raise ValueError(
            f"Flat cache shape {cache.shape} does not match live crop DAG shape "
            f"{live_shape}."
        )
    if roi_height < int(min_roi_size) or roi_width < int(min_roi_size):
        raise ValueError(
            f"Live crop DAG node is {roi_width}x{roi_height}; zebrafish keypoint "
            f"inference requires at least {int(min_roi_size)}x{int(min_roi_size)}."
        )

    storage_mode = _crop_storage_mode(crop_group)
    roi_images = crop_group.get("roi_images")
    roi_images_shape = getattr(roi_images, "shape", ())
    persisted_available = len(roi_images_shape) >= 3
    persisted_eligible = bool(
        persisted_available
        and int(roi_images_shape[1]) >= int(min_roi_size)
        and int(roi_images_shape[2]) >= int(min_roi_size)
        and int(roi_images_shape[0]) == total_rows
    )

    source_kind = str(
        crop_group.attrs.get("source_pixels")
        or crop_group.attrs.get("roi_pixel_provider")
        or crop_group.attrs.get("source_type")
        or ""
    ).strip()
    raw_crop_video_path = str(
        crop_group.attrs.get("source_crop_video_path")
        or crop_group.attrs.get("source_video_path")
        or ""
    ).strip()
    crop_video_path: Path | None = None
    if raw_crop_video_path:
        candidate = Path(raw_crop_video_path).expanduser()
        if not candidate.is_absolute():
            recording_dir = (
                resolved_zarr.parent.parent
                if resolved_zarr.parent.name == "zarr"
                else resolved_zarr.parent
            )
            candidate = recording_dir / candidate
        crop_video_path = candidate.resolve()
    crop_video_available = bool(
        source_kind == "acquisition_crop_video"
        and crop_video_path is not None
        and crop_video_path.is_file()
    )
    crop_video_shape: tuple[int, int] | None = None
    crop_video_probe_status = "not_available"
    if crop_video_available and crop_video_path is not None:
        crop_video_shape, crop_video_probe_status = _probe_video_shape(crop_video_path)
    crop_video_eligible = bool(
        crop_video_available
        and crop_video_shape is not None
        and crop_video_shape == (roi_height, roi_width)
        and crop_video_shape[0] >= int(min_roi_size)
        and crop_video_shape[1] >= int(min_roi_size)
    )
    rejected_sources: dict[str, str] = {}
    if not persisted_eligible:
        rejected_sources["persisted_roi_images"] = (
            "missing"
            if not persisted_available
            else f"too_small_or_row_mismatch:{tuple(int(v) for v in roi_images_shape)}"
        )
    if not crop_video_eligible:
        if source_kind != "acquisition_crop_video":
            rejected_sources["acquisition_crop_video"] = "not_declared_by_crop_run"
        elif crop_video_path is None or not crop_video_path.is_file():
            rejected_sources["acquisition_crop_video"] = "missing_video_file"
        elif crop_video_probe_status != "ok" or crop_video_shape is None:
            rejected_sources["acquisition_crop_video"] = crop_video_probe_status
        elif crop_video_shape != (roi_height, roi_width):
            rejected_sources["acquisition_crop_video"] = (
                f"shape_mismatch:video={crop_video_shape},crop={(roi_height, roi_width)}"
            )
        else:
            rejected_sources["acquisition_crop_video"] = (
                f"too_small:{crop_video_shape[1]}x{crop_video_shape[0]}"
            )

    return KeypointInputCapability(
        selected_source="flat_roi_cache",
        min_roi_size=int(min_roi_size),
        crop_run=cache.crop_run,
        crop_storage_mode=storage_mode,
        crop_shape=live_shape,
        persisted_roi_images_available=persisted_available,
        persisted_roi_images_eligible=persisted_eligible,
        acquisition_crop_video_available=crop_video_available,
        acquisition_crop_video_eligible=crop_video_eligible,
        acquisition_crop_video_path=crop_video_path,
        acquisition_crop_video_shape=crop_video_shape,
        acquisition_crop_video_probe_status=crop_video_probe_status,
        flat_roi_cache_eligible=True,
        rejected_sources=rejected_sources,
    )


def build_prediction_job(
    *,
    workflow_id: str,
    target_id: str,
    recording_dir: Path,
    analysis_zarr: Path,
    registry_path: Path,
    repo: Path,
    run_root: Path,
    run_names: KeypointRunNames,
    model: PoseModelBinding,
    cache: FlatRoiCacheBinding,
    pose_schema: str,
    batch_size: int,
    device: str,
    input_mode: str,
    progress_every_batches: int,
    resources: LsfResources,
) -> LsfJob:
    safe_target = safe_component(target_id, default="target", max_length=56)
    job_key = f"predict:{target_id}"
    job_name = safe_component(
        f"kp_{workflow_id}_{safe_target}",
        default="keypoint_prediction",
        max_length=120,
    )
    scratch_stage = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "palette_keypoint_terminal"
    )
    terminal_output = run_root / "terminal" / f"{safe_target}.zarr"
    worker: list[str] = [
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.utils.run_whole_recording_keypoint_terminal",
        "--recording-id",
        model.recording_id,
        "--recording-dir",
        str(recording_dir),
        "--analysis-zarr",
        str(analysis_zarr),
        "--registry",
        str(registry_path),
        "--model-set-id",
        model.set_id,
        "--model-run-id",
        model.run_id,
        "--terminal-run-id",
        run_names.terminal_run,
        "--terminal-output",
        str(terminal_output),
        "--crop-run",
        cache.crop_run,
        "--cache-manifest",
        str(cache.manifest_path),
        "--pose-schema",
        str(pose_schema),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        str(device),
        "--scratch-root",
        scratch_stage,
        "--input-mode",
        str(input_mode),
        "--progress-jsonl",
        str(
            run_root
            / "progress"
            / f"{safe_target}.prediction.{RUNTIME_JOB_ID_TOKEN}.jsonl"
        ),
        "--progress-every-batches",
        str(int(progress_every_batches)),
    ]
    command = build_runtime_command(
        worker,
        status_path_template=(
            run_root
            / "status"
            / f"{safe_target}.prediction.{RUNTIME_JOB_ID_TOKEN}.json"
        ),
        workflow_id=workflow_id,
        family="keypoints.whole_recording",
        job_key=job_key,
        stage="keypoint_prediction",
        cwd=repo,
        environment_overrides={"PALETTE_DISABLE_REGISTRY_WRITES": "1"},
        cleanup_path_templates=(scratch_stage,),
        expected_output_templates=(
            str(terminal_output / "terminal_receipt.json"),
        ),
        python_launcher=(str(repo / "scripts" / "py"),),
    )
    return LsfJob(
        job_key=job_key,
        job_name=job_name,
        command=command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{job_name}.%J.out",
        stderr_path=run_root / "logs" / f"{job_name}.%J.err",
        metadata={
            "target_id": target_id,
            "recording_id": model.recording_id,
            "analysis_zarr": str(analysis_zarr),
            "terminal_run": run_names.terminal_run,
            "terminal_output": str(terminal_output),
            "model": model.to_json(),
            "cache": cache.to_json(),
            "publication_boundary": "strict_v2_finalizer_only",
        },
    )


def build_refinement_job(
    *,
    workflow_id: str,
    target_id: str,
    analysis_zarr: Path,
    repo: Path,
    run_root: Path,
    run_names: KeypointRunNames,
    resources: LsfResources,
    prediction_job: LsfJob,
    crop_run: str,
    recording_identity: str,
) -> LsfJob:
    safe_target = safe_component(target_id, default="target", max_length=56)
    job_key = f"refine:{target_id}"
    job_name = safe_component(
        f"kp_refine_{workflow_id}_{safe_target}",
        default="keypoint_refinement",
        max_length=120,
    )
    terminal_output = run_root / "terminal" / f"{safe_target}.zarr"
    result_json = run_root / "finalization" / f"{safe_target}.json"
    lineage_id = str(
        uuid5(
            NAMESPACE_URL,
            f"palette:{recording_identity}:{run_names.refined_keypoint_run}:lineage",
        )
    )
    snapshot_id = str(
        uuid5(
            NAMESPACE_URL,
            f"palette:{recording_identity}:{run_names.refined_keypoint_run}:snapshot",
        )
    )
    scratch_stage = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "palette_keypoint_v2_finalize"
    )
    worker: list[str] = [
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.utils.finalize_whole_recording_keypoint_v2",
        "--analysis-zarr",
        str(analysis_zarr),
        "--crop-run",
        crop_run,
        "--terminal-artifact",
        str(terminal_output),
        "--raw-run",
        run_names.keypoint_run,
        "--quality-run",
        run_names.keypoint_quality_run,
        "--refined-run",
        run_names.refined_keypoint_run,
        "--body-frame-run",
        run_names.body_frame_run,
        "--recording-identity",
        recording_identity,
        "--refined-lineage-id",
        lineage_id,
        "--refined-snapshot-id",
        snapshot_id,
        "--scratch-root",
        scratch_stage,
        "--result-json",
        str(result_json),
    ]
    command = build_runtime_command(
        worker,
        status_path_template=(
            run_root
            / "status"
            / f"{safe_target}.refinement.{RUNTIME_JOB_ID_TOKEN}.json"
        ),
        workflow_id=workflow_id,
        family="keypoints.whole_recording",
        job_key=job_key,
        stage="keypoint_refinement",
        cwd=repo,
        environment_overrides={"PALETTE_DISABLE_REGISTRY_WRITES": "1"},
        expected_output_templates=(
            str(analysis_zarr / "keypoints_runs" / run_names.keypoint_run),
            str(
                analysis_zarr / "keypoint_quality_runs" / run_names.keypoint_quality_run
            ),
            str(analysis_zarr / "refined_keypoints_runs" / run_names.refined_keypoint_run),
            str(analysis_zarr / "analysis" / "body_frame_runs" / run_names.body_frame_run),
            str(result_json),
        ),
        python_launcher=(str(repo / "scripts" / "py"),),
    )
    return LsfJob(
        job_key=job_key,
        job_name=job_name,
        command=command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{job_name}.%J.out",
        stderr_path=run_root / "logs" / f"{job_name}.%J.err",
        dependency=LsfDependency((prediction_job.job_key,)),
        metadata={
            "target_id": target_id,
            "analysis_zarr": str(analysis_zarr),
            "source_keypoint_run": run_names.keypoint_run,
            "keypoint_quality_run": run_names.keypoint_quality_run,
            "refined_keypoint_run": run_names.refined_keypoint_run,
            "body_frame_run": run_names.body_frame_run,
            "terminal_output": str(terminal_output),
            "selector_activation": False,
        },
    )


__all__ = [
    "DEFAULT_ZEBRAFISH_MIN_ROI_SIZE",
    "DEFAULT_KEYPOINT_FRAME_SHARD_ROWS",
    "DEFAULT_KEYPOINT_ROI_SHARD_ROWS",
    "FlatRoiCacheBinding",
    "KeypointInputCapability",
    "KeypointRunNames",
    "PoseModelBinding",
    "build_keypoint_run_names",
    "build_prediction_job",
    "build_refinement_job",
    "resolve_pose_model_binding",
    "resolve_keypoint_storage",
    "resolve_keypoint_v2_publication_storage",
    "safe_component",
    "validate_flat_roi_cache_binding",
    "validate_keypoint_input_dag",
    "validate_registered_analysis_zarr",
]
