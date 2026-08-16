"""Keyed, subset-only ROI pixel packages for downstream delta inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.flat_roi_cache import write_pynvvc_luma_roi_payload
from fisheye.shared.hybrid_crop_provider import validate_hybrid_crop_signed_identity
from fisheye.shared.roi_pixel_contract import (
    SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME,
    normalize_pixel_contract,
)
from fisheye.shared.row_source_signature import (
    ROW_SOURCE_SIGNATURE_ARRAY,
    ROW_SOURCE_SIGNATURE_WIDTH_BYTES,
    load_row_source_signature_spec,
)
from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_SIGNED_PROFILE,
    CROP_RUN_REFERENCE_STRICT_PROFILE,
    authoritative_crop_roi_pixel_contract,
    build_crop_run_reference,
    strict_crop_row_source_signature_spec,
)

CROP_PIXEL_WORK_PACKAGE_SCHEMA_ID = "palette.crop_pixel_work_package"
CROP_PIXEL_WORK_PACKAGE_SCHEMA_VERSION = 1
CROP_PIXEL_WORK_PACKAGE_LAYOUT = "keyed_flat_bin_v1"
CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_ID = "palette.crop_pixel_work_package.rows"
CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_VERSION = 1
PIXEL_SHA256_ARRAY = "pixel_sha256"
DEFAULT_WORK_PACKAGE_BATCH_ROWS = 256
STRICT_SOURCE_BINDING_PROFILE = "immutable_crop_run_manifest_v1"
SIGNED_SOURCE_BINDING_PROFILE = "signed_crop_run_v1"
LEGACY_SOURCE_BINDING_PROFILE = "legacy_crop_signature_revision_v1"


class CropPixelWorkPackageError(RuntimeError):
    """Raised when a pixel work package cannot be trusted or reproduced."""


class CropPixelWorkPackageArray:
    """Read-only mmap view of one subset ROI payload."""

    def __init__(self, path: Path, *, shape: tuple[int, int, int]) -> None:
        self.path = path.expanduser().resolve()
        self.shape = tuple(int(value) for value in shape)
        self.ndim = 3
        self.dtype = np.dtype(np.uint8)
        expected_bytes = int(np.prod(self.shape, dtype=np.int64))
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Crop pixel work-package payload is missing: {self.path}"
            )
        actual_bytes = int(self.path.stat().st_size)
        if actual_bytes != expected_bytes:
            raise CropPixelWorkPackageError(
                "Crop pixel work-package payload size mismatch: "
                f"expected {expected_bytes}, found {actual_bytes}."
            )
        self._memmap = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            shape=self.shape,
            order="C",
        )

    def __getitem__(self, key: object) -> np.ndarray:
        return self._memmap[key]

    def close(self) -> None:
        mmap = getattr(self._memmap, "_mmap", None)
        if mmap is not None:
            mmap.close()


@dataclass(frozen=True)
class CropPixelWorkPackage:
    manifest_path: Path
    manifest: Mapping[str, Any]
    pixels: CropPixelWorkPackageArray
    crop_row_indices: np.ndarray
    instance_keys: np.ndarray
    source_row_signatures: np.ndarray
    frame_indices: np.ndarray
    roi_coordinates_full: np.ndarray
    pixel_sha256: np.ndarray

    @property
    def package_id(self) -> str:
        return str(self.manifest["package_id"])

    @property
    def crop_run_name(self) -> str:
        return str(_require_mapping(self.manifest, "source")["crop_run_name"])

    @property
    def roi_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.pixels.shape[1:])

    @property
    def row_count(self) -> int:
        return int(self.crop_row_indices.shape[0])

    @property
    def pixel_contract(self) -> Mapping[str, Any]:
        return _require_mapping(self.manifest, "pixel_contract")

    def close(self) -> None:
        self.pixels.close()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CropPixelWorkPackageError(
            "Crop pixel work-package metadata is not strict JSON."
        ) from exc


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    def _reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    if not isinstance(payload, Mapping):
        raise CropPixelWorkPackageError("Work-package manifest root must be an object.")
    return payload


def _require_mapping(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise CropPixelWorkPackageError(
            f"Crop pixel work-package manifest is missing mapping {name!r}."
        )
    return value


def _resolve_relative(manifest_path: Path, value: object, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise CropPixelWorkPackageError(
            f"Crop pixel work-package manifest is missing {field}."
        )
    path = Path(text).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (manifest_path.parent / path).resolve()
    )


def _relative_or_absolute(path: Path, *, base: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(base.expanduser().resolve()))
    except ValueError:
        return str(resolved)


def _temporary_path(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}_{uuid.uuid4().hex}{suffix}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_pixel_rows(values: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(values, dtype=np.uint8)
    output = np.empty((rows.shape[0], hashlib.sha256().digest_size), dtype=np.uint8)
    for index, row in enumerate(rows):
        output[index] = np.frombuffer(
            hashlib.sha256(row.tobytes(order="C")).digest(),
            dtype=np.uint8,
        )
    return output


def _logical_package_id(
    *,
    crop_run_name: str,
    crop_row_indices: np.ndarray,
    instance_keys: np.ndarray,
    source_row_signatures: np.ndarray,
    pixel_sha256: np.ndarray,
    pixel_contract: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    materialization_binding: Mapping[str, Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    for text in (
        CROP_PIXEL_WORK_PACKAGE_SCHEMA_ID,
        str(CROP_PIXEL_WORK_PACKAGE_SCHEMA_VERSION),
        crop_run_name,
        _canonical_json(pixel_contract),
        _canonical_json(source_binding),
    ):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    if materialization_binding is not None:
        digest.update(b"materialization_binding\0")
        digest.update(_canonical_json(materialization_binding).encode("utf-8"))
        digest.update(b"\0")
    for values, dtype in (
        (crop_row_indices, "<i8"),
        (instance_keys, "<u8"),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes(order="C"))
    digest.update(np.asarray(source_row_signatures, dtype=np.uint8).tobytes(order="C"))
    digest.update(np.asarray(pixel_sha256, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _normalize_frame_window_binding(
    value: Mapping[str, Any],
    *,
    frame_offset: int,
    frame_count: int,
) -> dict[str, Any]:
    fields = {
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
    canonical = json.loads(_canonical_json(value))
    if not isinstance(canonical, dict) or set(canonical) != fields:
        raise CropPixelWorkPackageError(
            "Frame-window materialization binding fields are not exact."
        )
    if (
        canonical.get("schema_id") != "palette.acquisition_video_frame_window"
        or canonical.get("schema_version") != 1
        or type(canonical.get("actual_start_frame")) is not int
        or type(canonical.get("end_frame_exclusive")) is not int
        or type(canonical.get("frame_count")) is not int
        or canonical["actual_start_frame"] != int(frame_offset)
        or canonical["frame_count"] != int(frame_count)
        or canonical["end_frame_exclusive"] != int(frame_offset + frame_count)
    ):
        raise CropPixelWorkPackageError(
            "Frame-window materialization interval differs from the requested window."
        )
    for name in ("recording_identity", "camera_identity", "clip_id"):
        if not isinstance(canonical.get(name), str) or not canonical[name].strip():
            raise CropPixelWorkPackageError(
                f"Frame-window materialization binding lacks {name}."
            )
    for name in ("clip_index_document_sha256", "clip_video_sha256"):
        digest = canonical.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CropPixelWorkPackageError(
                f"Frame-window materialization binding has invalid {name}."
            )
    return canonical


def _normalize_target_rows(
    values: Sequence[int] | np.ndarray, *, total_rows: int
) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        raise CropPixelWorkPackageError(
            "Crop pixel work package requires at least one row."
        )
    if rows.min() < 0 or rows.max() >= int(total_rows):
        raise CropPixelWorkPackageError("Crop pixel work-package row is out of bounds.")
    if np.unique(rows).shape[0] != rows.shape[0]:
        raise CropPixelWorkPackageError("Crop pixel work-package rows must be unique.")
    if not np.array_equal(rows, np.sort(rows, kind="stable")):
        raise CropPixelWorkPackageError(
            "Crop pixel work-package rows must be in canonical ascending crop-row order."
        )
    return rows


def _source_binding(crop_group: Any, *, run_id: str) -> dict[str, Any]:
    source_pixels = str(
        crop_group.attrs.get("source_pixels")
        or crop_group.attrs.get("roi_pixel_provider")
        or ""
    ).strip()
    if source_pixels == SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME:
        provider_digest = str(
            crop_group.attrs.get("provider_record_sha256") or ""
        ).strip()
        validate_hybrid_crop_signed_identity(
            crop_group,
            expected_provider_record_sha256=provider_digest,
        )
    reference = build_crop_run_reference(
        crop_group,
        run_id=run_id,
        allow_unversioned_legacy=True,
    )
    if reference["profile"] == CROP_RUN_REFERENCE_STRICT_PROFILE:
        spec = strict_crop_row_source_signature_spec(crop_group, run_id=run_id)
        if spec is None:  # pragma: no cover - strict reference guarantees a manifest
            raise CropPixelWorkPackageError(
                "Strict crop work package lacks a row-signature specification."
            )
        authoritative_crop_roi_pixel_contract(crop_group, run_id=run_id)
        authority = crop_group.attrs.get("source_pixel_authority")
        if not isinstance(authority, Mapping):
            raise CropPixelWorkPackageError(
                "Strict crop work package lacks source pixel authority."
            )
        return {
            "source_binding_profile": STRICT_SOURCE_BINDING_PROFILE,
            "crop_run_reference": reference,
            "source_row_signature_spec_digest": spec.spec_digest,
            "source_pixel_authority_id": authority.get("authority_id"),
            "source_pixel_authority_manifest_digest": authority.get(
                "authority_manifest_digest"
            ),
        }

    spec = load_row_source_signature_spec(crop_group.attrs)
    binding_profile = (
        SIGNED_SOURCE_BINDING_PROFILE
        if reference["profile"] == CROP_RUN_REFERENCE_SIGNED_PROFILE
        else LEGACY_SOURCE_BINDING_PROFILE
    )
    return {
        "source_binding_profile": binding_profile,
        "source_row_signature_spec_digest": spec.spec_digest,
        "source_pixel_fingerprint": crop_group.attrs.get("source_pixel_fingerprint"),
        "source_rowset_fingerprint": crop_group.attrs.get("source_rowset_fingerprint"),
        "crop_revision": crop_group.attrs.get("crop_revision"),
        "crop_signature": crop_group.attrs.get("crop_signature"),
    }


def _manifest_source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    profile = source.get("source_binding_profile")
    if profile == STRICT_SOURCE_BINDING_PROFILE:
        fields = (
            "source_binding_profile",
            "crop_run_reference",
            "source_row_signature_spec_digest",
            "source_pixel_authority_id",
            "source_pixel_authority_manifest_digest",
        )
    elif profile in (
        None,
        SIGNED_SOURCE_BINDING_PROFILE,
        LEGACY_SOURCE_BINDING_PROFILE,
    ):
        legacy_fields = (
            "source_row_signature_spec_digest",
            "source_pixel_fingerprint",
            "source_rowset_fingerprint",
            "crop_revision",
            "crop_signature",
        )
        fields = (
            ("source_binding_profile", *legacy_fields)
            if profile is not None
            else legacy_fields
        )
    else:
        raise CropPixelWorkPackageError(
            f"Unsupported crop pixel source binding profile {profile!r}."
        )
    missing = [name for name in fields if name not in source]
    if missing:
        raise CropPixelWorkPackageError(
            "Crop pixel work-package source binding is incomplete: "
            + ", ".join(missing)
        )
    return {name: source.get(name) for name in fields}


def _write_rows_npz(
    path: Path,
    *,
    crop_row_indices: np.ndarray,
    instance_keys: np.ndarray,
    source_row_signatures: np.ndarray,
    frame_indices: np.ndarray,
    roi_coordinates_full: np.ndarray,
    pixel_sha256: np.ndarray,
) -> None:
    with path.open("wb") as handle:
        np.savez(
            handle,
            rows_schema_id=np.asarray(CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_ID),
            rows_schema_version=np.asarray(
                CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_VERSION, dtype=np.int64
            ),
            crop_row_indices=np.asarray(crop_row_indices, dtype=np.int64),
            instance_key=np.asarray(instance_keys, dtype=np.uint64),
            source_row_signature=np.asarray(source_row_signatures, dtype=np.uint8),
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            roi_coordinates_full=np.asarray(roi_coordinates_full, dtype=np.int32),
            pixel_sha256=np.asarray(pixel_sha256, dtype=np.uint8),
        )


def build_crop_pixel_work_package_from_source(
    source: Any,
    *,
    target_crop_rows: Sequence[int] | np.ndarray,
    manifest_path: str | Path,
    archive_path: str | Path,
    batch_rows: int = DEFAULT_WORK_PACKAGE_BATCH_ROWS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist only selected crop rows as one atomic keyed work package."""

    if getattr(source, "pixel_materialization_id", None):
        raise CropPixelWorkPackageError(
            "Refusing to build a work package from another work package."
        )
    crop_group = source.crop_group
    if crop_group.attrs.get("refined_roi_path"):
        raise CropPixelWorkPackageError(
            "Crop run uses legacy refined_roi_path overrides; fold them into the crop "
            "definition before building a shared pixel package."
        )
    missing = [
        name
        for name in ("instance_key", ROW_SOURCE_SIGNATURE_ARRAY, "frame_indices")
        if name not in crop_group
    ]
    if missing:
        raise CropPixelWorkPackageError(
            "Modern crop pixel work packages require crop arrays: " + ", ".join(missing)
        )
    total_rows = int(source.total_rois)
    rows = _normalize_target_rows(target_crop_rows, total_rows=total_rows)
    keys = np.asarray(crop_group["instance_key"][rows], dtype=np.uint64).reshape(-1)
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise CropPixelWorkPackageError(
            "Selected crop instance_key values are not unique."
        )
    signatures = np.asarray(
        crop_group[ROW_SOURCE_SIGNATURE_ARRAY][rows], dtype=np.uint8
    )
    if signatures.shape != (rows.shape[0], ROW_SOURCE_SIGNATURE_WIDTH_BYTES):
        raise CropPixelWorkPackageError(
            "Selected crop source signatures have invalid shape."
        )
    frame_indices = np.asarray(source.frame_indices[rows], dtype=np.int64)
    coordinates = np.asarray(source.roi_coordinates_full[rows], dtype=np.int32)
    pixel_contract = normalize_pixel_contract(source.roi_pixel_contract)
    if pixel_contract is None:
        raise CropPixelWorkPackageError("Crop source has no valid ROI pixel contract.")
    binding = _source_binding(crop_group, run_id=str(source.crop_run_name))
    required_pixel_contract = authoritative_crop_roi_pixel_contract(
        crop_group,
        run_id=str(source.crop_run_name),
    )
    if (
        required_pixel_contract is not None
        and pixel_contract != required_pixel_contract
    ):
        raise CropPixelWorkPackageError(
            "Crop work packages must be built from the authoritative "
            f"{required_pixel_contract['name']!r} pixel materialization."
        )

    manifest = Path(manifest_path).expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and not overwrite:
        existing = open_crop_pixel_work_package(
            manifest,
            expected_archive_path=archive_path,
            expected_crop_run=source.crop_run_name,
            root=source.root,
            verify_payload=True,
            verify_pixel_rows=True,
        )
        try:
            if not np.array_equal(existing.crop_row_indices, rows):
                raise CropPixelWorkPackageError(
                    "Existing crop pixel work package has different target rows."
                )
            return dict(existing.manifest)
        finally:
            existing.close()

    payload_tmp = _temporary_path(manifest, ".tmp.bin")
    rows_tmp = _temporary_path(manifest, ".tmp.npz")
    manifest_tmp = _temporary_path(manifest, ".tmp.json")
    row_count = int(rows.shape[0])
    roi_shape = tuple(int(value) for value in source.roi_shape)
    effective_batch = max(1, int(batch_rows))
    pixel_hashes = np.empty((row_count, hashlib.sha256().digest_size), dtype=np.uint8)
    payload_digest = hashlib.sha256()
    try:
        with payload_tmp.open("wb") as handle:
            for start in range(0, row_count, effective_batch):
                stop = min(start + effective_batch, row_count)
                batch = np.ascontiguousarray(
                    source.read_indices(rows[start:stop]), dtype=np.uint8
                )
                if batch.shape != (stop - start, *roi_shape):
                    raise CropPixelWorkPackageError(
                        "Crop source returned the wrong work-package batch shape."
                    )
                payload = batch.tobytes(order="C")
                handle.write(payload)
                payload_digest.update(payload)
                pixel_hashes[start:stop] = _hash_pixel_rows(batch)
        _write_rows_npz(
            rows_tmp,
            crop_row_indices=rows,
            instance_keys=keys,
            source_row_signatures=signatures,
            frame_indices=frame_indices,
            roi_coordinates_full=coordinates,
            pixel_sha256=pixel_hashes,
        )
        rows_sha256 = _sha256_file(rows_tmp)
        package_id = _logical_package_id(
            crop_run_name=str(source.crop_run_name),
            crop_row_indices=rows,
            instance_keys=keys,
            source_row_signatures=signatures,
            pixel_sha256=pixel_hashes,
            pixel_contract=pixel_contract,
            source_binding=binding,
        )
        generation_id = uuid.uuid4().hex
        artifact_stem = f"{manifest.stem}.{package_id[:16]}.{generation_id}"
        payload_path = manifest.with_name(f"{artifact_stem}.bin")
        rows_path = manifest.with_name(f"{artifact_stem}.rows.npz")
        payload_bytes = int(row_count * roi_shape[0] * roi_shape[1])
        payload = {
            "schema_id": CROP_PIXEL_WORK_PACKAGE_SCHEMA_ID,
            "schema_version": CROP_PIXEL_WORK_PACKAGE_SCHEMA_VERSION,
            "layout": CROP_PIXEL_WORK_PACKAGE_LAYOUT,
            "status": "complete",
            "package_id": package_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest),
            "source": {
                "archive_path": str(Path(archive_path).expanduser().resolve()),
                "crop_run_name": str(source.crop_run_name),
                "crop_storage_mode": str(source.storage_mode),
                **binding,
            },
            "selection": {
                "identity_mode": "instance_key",
                "ordering": "ascending_source_crop_row",
                "row_count": row_count,
                "source_crop_total_rows": total_rows,
            },
            "array": {
                "bin_path": _relative_or_absolute(payload_path, base=manifest.parent),
                "shape": [row_count, *roi_shape],
                "dtype": "uint8",
                "order": "C",
                "row_stride_bytes": int(roi_shape[0] * roi_shape[1]),
                "total_bytes": payload_bytes,
                "sha256": payload_digest.hexdigest(),
            },
            "rows": {
                "path": _relative_or_absolute(rows_path, base=manifest.parent),
                "schema_id": CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_ID,
                "schema_version": CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_VERSION,
                "sha256": rows_sha256,
            },
            "pixel_contract": pixel_contract,
            "builder": {
                "batch_rows": effective_batch,
                "source_roi_read_mode": str(source.roi_read_mode),
                "source_frame_kind": str(source.frame_source_kind),
                "source_frame_identity": source._build_frame_source_identity(),
                "semantics": "durable_shared_subset_input_not_canonical_crop_authority",
            },
        }
        manifest_tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Payload objects are generation-specific.  Publishing the manifest last
        # keeps a previously complete generation valid throughout a retry.
        os.replace(payload_tmp, payload_path)
        os.replace(rows_tmp, rows_path)
        os.replace(manifest_tmp, manifest)
    except Exception:
        for path in (payload_tmp, rows_tmp, manifest_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return payload


def build_crop_pixel_work_package_from_video_window(
    source: Any,
    *,
    target_crop_rows: Sequence[int] | np.ndarray,
    video_path: str | Path,
    source_video_frame_offset: int,
    source_video_frame_count: int,
    frame_window_binding: Mapping[str, Any],
    manifest_path: str | Path,
    archive_path: str | Path,
    batch_rows: int = 1024,
    overwrite: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a strict keyed package from one acquisition-frame video window.

    This is the maintained clipped/full-recording bridge: global crop rows and
    acquisition-frame identities remain bound to the canonical crop run while
    PyNvVideoCodec reads frame-local indices from an authenticated clip.  The
    resulting package is an immutable model-input artifact; downstream shard
    writers never reinterpret clip-local row ordinals as crop identities.
    """

    if getattr(source, "pixel_materialization_id", None):
        raise CropPixelWorkPackageError(
            "Refusing to build a video-window package from another work package."
        )
    crop_group = source.crop_group
    missing = [
        name
        for name in ("instance_key", ROW_SOURCE_SIGNATURE_ARRAY, "frame_indices")
        if name not in crop_group
    ]
    if missing:
        raise CropPixelWorkPackageError(
            "Modern video-window work packages require crop arrays: "
            + ", ".join(missing)
        )
    offset = int(source_video_frame_offset)
    count = int(source_video_frame_count)
    if offset < 0 or count <= 0:
        raise CropPixelWorkPackageError(
            "Video-window frame offset/count must be nonnegative/positive."
        )
    window_binding = _normalize_frame_window_binding(
        frame_window_binding,
        frame_offset=offset,
        frame_count=count,
    )
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Crop video window is missing: {video}")
    observed_video_sha256 = _sha256_file(video)
    if observed_video_sha256 != window_binding["clip_video_sha256"]:
        raise CropPixelWorkPackageError(
            "Crop video window digest differs from its frame-window binding."
        )

    total_rows = int(source.total_rois)
    rows = _normalize_target_rows(target_crop_rows, total_rows=total_rows)
    keys = np.asarray(crop_group["instance_key"][rows], dtype=np.uint64).reshape(-1)
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise CropPixelWorkPackageError(
            "Selected crop instance_key values are not unique."
        )
    signatures = np.asarray(
        crop_group[ROW_SOURCE_SIGNATURE_ARRAY][rows], dtype=np.uint8
    )
    if signatures.shape != (rows.shape[0], ROW_SOURCE_SIGNATURE_WIDTH_BYTES):
        raise CropPixelWorkPackageError(
            "Selected crop source signatures have invalid shape."
        )
    frame_indices = np.asarray(source.frame_indices[rows], dtype=np.int64)
    local_frame_indices = frame_indices - offset
    if np.any(local_frame_indices < 0) or np.any(local_frame_indices >= count):
        raise CropPixelWorkPackageError(
            "Selected crop rows fall outside the bound acquisition-frame window."
        )
    coordinates = np.asarray(source.roi_coordinates_full[rows], dtype=np.int32)
    binding = _source_binding(crop_group, run_id=str(source.crop_run_name))
    pixel_contract = authoritative_crop_roi_pixel_contract(
        crop_group,
        run_id=str(source.crop_run_name),
    )
    if pixel_contract is None:
        raise CropPixelWorkPackageError(
            "Video-window packages require a current authoritative crop pixel contract."
        )
    frame_shape = getattr(source, "frame_shape", None)
    if frame_shape is None or len(tuple(frame_shape)) != 2:
        raise CropPixelWorkPackageError(
            "Video-window packages require exact source video dimensions."
        )
    roi_shape = tuple(int(value) for value in source.roi_shape)
    effective_batch = max(1, int(batch_rows))
    manifest = Path(manifest_path).expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and not overwrite:
        existing = open_crop_pixel_work_package(
            manifest,
            expected_archive_path=archive_path,
            expected_crop_run=source.crop_run_name,
            root=source.root,
            verify_payload=True,
            verify_pixel_rows=True,
        )
        try:
            if (
                not np.array_equal(existing.crop_row_indices, rows)
                or existing.manifest.get("materialization_binding") != window_binding
            ):
                raise CropPixelWorkPackageError(
                    "Existing crop pixel package binds a different frame window."
                )
            return dict(existing.manifest)
        finally:
            existing.close()

    payload_tmp = _temporary_path(manifest, ".tmp.bin")
    rows_tmp = _temporary_path(manifest, ".tmp.npz")
    manifest_tmp = _temporary_path(manifest, ".tmp.json")
    row_count = int(rows.shape[0])
    pixel_hashes = np.empty((row_count, hashlib.sha256().digest_size), dtype=np.uint8)
    try:
        materialization = write_pynvvc_luma_roi_payload(
            video_path=video,
            frame_indices=local_frame_indices,
            roi_coordinates_full=coordinates,
            roi_shape=roi_shape,
            video_shape=tuple(int(value) for value in frame_shape),
            output_path=payload_tmp,
            batch_size=effective_batch,
            progress_callback=progress_callback,
        )
        pixels = CropPixelWorkPackageArray(
            payload_tmp,
            shape=(row_count, *roi_shape),
        )
        try:
            for start in range(0, row_count, effective_batch):
                stop = min(start + effective_batch, row_count)
                pixel_hashes[start:stop] = _hash_pixel_rows(
                    np.asarray(pixels[start:stop])
                )
        finally:
            pixels.close()
        _write_rows_npz(
            rows_tmp,
            crop_row_indices=rows,
            instance_keys=keys,
            source_row_signatures=signatures,
            frame_indices=frame_indices,
            roi_coordinates_full=coordinates,
            pixel_sha256=pixel_hashes,
        )
        rows_sha256 = _sha256_file(rows_tmp)
        package_id = _logical_package_id(
            crop_run_name=str(source.crop_run_name),
            crop_row_indices=rows,
            instance_keys=keys,
            source_row_signatures=signatures,
            pixel_sha256=pixel_hashes,
            pixel_contract=pixel_contract,
            source_binding=binding,
            materialization_binding=window_binding,
        )
        generation_id = uuid.uuid4().hex
        artifact_stem = f"{manifest.stem}.{package_id[:16]}.{generation_id}"
        payload_path = manifest.with_name(f"{artifact_stem}.bin")
        rows_path = manifest.with_name(f"{artifact_stem}.rows.npz")
        materialization_record = dict(materialization)
        materialization_record.pop("path", None)
        payload = {
            "schema_id": CROP_PIXEL_WORK_PACKAGE_SCHEMA_ID,
            "schema_version": CROP_PIXEL_WORK_PACKAGE_SCHEMA_VERSION,
            "layout": CROP_PIXEL_WORK_PACKAGE_LAYOUT,
            "status": "complete",
            "package_id": package_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest),
            "source": {
                "archive_path": str(Path(archive_path).expanduser().resolve()),
                "crop_run_name": str(source.crop_run_name),
                "crop_storage_mode": str(source.storage_mode),
                **binding,
            },
            "selection": {
                "identity_mode": "instance_key",
                "ordering": "ascending_source_crop_row",
                "row_count": row_count,
                "source_crop_total_rows": total_rows,
            },
            "array": {
                "bin_path": _relative_or_absolute(payload_path, base=manifest.parent),
                "shape": [row_count, *roi_shape],
                "dtype": "uint8",
                "order": "C",
                "row_stride_bytes": int(roi_shape[0] * roi_shape[1]),
                "total_bytes": int(materialization["total_bytes"]),
                "sha256": str(materialization["sha256"]),
            },
            "rows": {
                "path": _relative_or_absolute(rows_path, base=manifest.parent),
                "schema_id": CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_ID,
                "schema_version": CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_VERSION,
                "sha256": rows_sha256,
            },
            "pixel_contract": pixel_contract,
            "materialization_binding": window_binding,
            "builder": {
                "batch_rows": effective_batch,
                "decode_backend": "pynvvc_luma",
                "materialization": materialization_record,
                "semantics": (
                    "global_crop_rows_from_authenticated_acquisition_video_window_v1"
                ),
            },
        }
        manifest_tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(payload_tmp, payload_path)
        os.replace(rows_tmp, rows_path)
        os.replace(manifest_tmp, manifest)
    except Exception:
        for path in (payload_tmp, rows_tmp, manifest_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return payload


def build_crop_pixel_work_package(
    *,
    zarr_path: str | Path,
    crop_run: str,
    target_crop_rows: Sequence[int] | np.ndarray,
    manifest_path: str | Path,
    batch_rows: int = DEFAULT_WORK_PACKAGE_BATCH_ROWS,
    overwrite: bool = False,
    roi_live_acceleration: str = "auto",
) -> dict[str, Any]:
    """Open a crop source and build a keyed subset package."""

    archive = Path(zarr_path).expanduser().resolve()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    from fisheye.shared.crop_image_source import CropImageSource

    source = CropImageSource.open(
        root,
        crop_run=crop_run,
        zarr_path=archive,
        roi_cache_policy="never",
        roi_live_acceleration=roi_live_acceleration,
    )
    try:
        return build_crop_pixel_work_package_from_source(
            source,
            target_crop_rows=target_crop_rows,
            manifest_path=manifest_path,
            archive_path=archive,
            batch_rows=batch_rows,
            overwrite=overwrite,
        )
    finally:
        source.close()


def _load_rows(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            schema_id = str(np.asarray(payload["rows_schema_id"]).item())
            schema_version = int(np.asarray(payload["rows_schema_version"]).item())
            if schema_id != CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_ID:
                raise CropPixelWorkPackageError(
                    "Unsupported work-package row schema id."
                )
            if schema_version != CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_VERSION:
                raise CropPixelWorkPackageError(
                    "Unsupported work-package row schema version."
                )
            return {
                "crop_row_indices": np.asarray(
                    payload["crop_row_indices"], dtype=np.int64
                ),
                "instance_key": np.asarray(payload["instance_key"], dtype=np.uint64),
                "source_row_signature": np.asarray(
                    payload["source_row_signature"], dtype=np.uint8
                ),
                "frame_indices": np.asarray(payload["frame_indices"], dtype=np.int64),
                "roi_coordinates_full": np.asarray(
                    payload["roi_coordinates_full"], dtype=np.int32
                ),
                "pixel_sha256": np.asarray(payload["pixel_sha256"], dtype=np.uint8),
            }
    except CropPixelWorkPackageError:
        raise
    except Exception as exc:
        raise CropPixelWorkPackageError(
            f"Unable to read crop pixel work-package row index: {path}"
        ) from exc


def _validate_live_binding(
    package: CropPixelWorkPackage,
    *,
    root: Any,
) -> None:
    source = _require_mapping(package.manifest, "source")
    crop_name = str(source.get("crop_run_name") or "")
    if "crop_runs" not in root or crop_name not in root["crop_runs"]:
        raise CropPixelWorkPackageError(
            f"Bound crop run {crop_name!r} does not exist in the archive."
        )
    crop = root[f"crop_runs/{crop_name}"]
    if "instance_key" not in crop or ROW_SOURCE_SIGNATURE_ARRAY not in crop:
        raise CropPixelWorkPackageError(
            "Bound crop run lacks modern identity/signatures."
        )
    total_rows = int(crop["instance_key"].shape[0])
    rows = _normalize_target_rows(package.crop_row_indices, total_rows=total_rows)
    keys = np.asarray(crop["instance_key"][rows], dtype=np.uint64)
    signatures = np.asarray(crop[ROW_SOURCE_SIGNATURE_ARRAY][rows], dtype=np.uint8)
    if not np.array_equal(keys, package.instance_keys):
        raise CropPixelWorkPackageError(
            "Work-package keys no longer match the crop run."
        )
    if not np.array_equal(signatures, package.source_row_signatures):
        raise CropPixelWorkPackageError(
            "Work-package source signatures no longer match the crop run."
        )
    if "frame_indices" not in crop or "roi_coordinates_full" not in crop:
        raise CropPixelWorkPackageError(
            "Bound crop run lacks required geometry arrays."
        )
    if not np.array_equal(
        np.asarray(crop["frame_indices"][rows], dtype=np.int64),
        package.frame_indices,
    ):
        raise CropPixelWorkPackageError(
            "Work-package frames no longer match the crop run."
        )
    if not np.array_equal(
        np.asarray(crop["roi_coordinates_full"][rows], dtype=np.int32),
        package.roi_coordinates_full,
    ):
        raise CropPixelWorkPackageError(
            "Work-package geometry no longer matches the crop run."
        )
    binding = _source_binding(crop, run_id=crop_name)
    persisted_binding = _manifest_source_binding(source)
    for name, value in persisted_binding.items():
        if binding.get(name) != value:
            raise CropPixelWorkPackageError(
                f"Work-package source binding changed for {name!r}."
            )
    required_pixel_contract = authoritative_crop_roi_pixel_contract(
        crop,
        run_id=crop_name,
    )
    if (
        required_pixel_contract is not None
        and dict(package.pixel_contract) != required_pixel_contract
    ):
        raise CropPixelWorkPackageError(
            "Work-package pixel contract differs from its authoritative crop "
            "pixel source."
        )


def open_crop_pixel_work_package(
    manifest_path: str | Path,
    *,
    expected_archive_path: str | Path | None = None,
    expected_crop_run: str | None = None,
    root: Any | None = None,
    verify_payload: bool = True,
    verify_pixel_rows: bool = True,
) -> CropPixelWorkPackage:
    """Open and fail-closed validate one complete keyed ROI subset package."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    try:
        manifest = _load_strict_json(manifest_file)
    except Exception as exc:
        raise CropPixelWorkPackageError(
            f"Unable to read crop pixel work-package manifest: {manifest_file}"
        ) from exc
    if manifest.get("schema_id") != CROP_PIXEL_WORK_PACKAGE_SCHEMA_ID:
        raise CropPixelWorkPackageError(
            "Unsupported crop pixel work-package schema id."
        )
    if (
        int(manifest.get("schema_version", -1))
        != CROP_PIXEL_WORK_PACKAGE_SCHEMA_VERSION
    ):
        raise CropPixelWorkPackageError(
            "Unsupported crop pixel work-package schema version."
        )
    if manifest.get("layout") != CROP_PIXEL_WORK_PACKAGE_LAYOUT:
        raise CropPixelWorkPackageError("Unsupported crop pixel work-package layout.")
    if manifest.get("status") != "complete":
        raise CropPixelWorkPackageError("Crop pixel work package is not complete.")
    source = _require_mapping(manifest, "source")
    if (
        expected_archive_path is not None
        and Path(str(source.get("archive_path") or "")).expanduser().resolve()
        != Path(expected_archive_path).expanduser().resolve()
    ):
        raise CropPixelWorkPackageError("Work package is bound to a different archive.")
    if expected_crop_run is not None and str(source.get("crop_run_name") or "") != str(
        expected_crop_run
    ):
        raise CropPixelWorkPackageError(
            "Work package is bound to a different crop run."
        )
    array = _require_mapping(manifest, "array")
    shape_raw = array.get("shape")
    if not isinstance(shape_raw, list) or len(shape_raw) != 3:
        raise CropPixelWorkPackageError("Work-package array shape must be [D,H,W].")
    shape = tuple(int(value) for value in shape_raw)
    if (
        min(shape) <= 0
        or str(array.get("dtype")) != "uint8"
        or array.get("order") != "C"
    ):
        raise CropPixelWorkPackageError("Work-package array contract is invalid.")
    pixel_contract = normalize_pixel_contract(manifest.get("pixel_contract"))
    if pixel_contract is None:
        raise CropPixelWorkPackageError("Work package has no valid pixel contract.")
    materialization_binding_raw = manifest.get("materialization_binding")
    materialization_binding: dict[str, Any] | None = None
    if materialization_binding_raw is not None:
        if not isinstance(materialization_binding_raw, Mapping):
            raise CropPixelWorkPackageError(
                "Work-package materialization binding must be an object."
            )
        raw_offset = materialization_binding_raw.get("actual_start_frame")
        raw_count = materialization_binding_raw.get("frame_count")
        if type(raw_offset) is not int or type(raw_count) is not int:
            raise CropPixelWorkPackageError(
                "Work-package materialization frame offset/count must be integers."
            )
        materialization_binding = _normalize_frame_window_binding(
            materialization_binding_raw,
            frame_offset=raw_offset,
            frame_count=raw_count,
        )
    payload_path = _resolve_relative(
        manifest_file, array.get("bin_path"), field="array.bin_path"
    )
    rows_meta = _require_mapping(manifest, "rows")
    rows_path = _resolve_relative(
        manifest_file, rows_meta.get("path"), field="rows.path"
    )
    if not rows_path.is_file():
        raise FileNotFoundError(
            f"Crop pixel work-package rows are missing: {rows_path}"
        )
    if _sha256_file(rows_path) != str(rows_meta.get("sha256") or ""):
        raise CropPixelWorkPackageError("Work-package row-index digest mismatch.")
    rows = _load_rows(rows_path)
    row_count = shape[0]
    aligned_shapes = {
        "crop_row_indices": (row_count,),
        "instance_key": (row_count,),
        "source_row_signature": (row_count, ROW_SOURCE_SIGNATURE_WIDTH_BYTES),
        "frame_indices": (row_count,),
        "roi_coordinates_full": (row_count, 2),
        "pixel_sha256": (row_count, hashlib.sha256().digest_size),
    }
    for name, expected_shape in aligned_shapes.items():
        if rows[name].shape != expected_shape:
            raise CropPixelWorkPackageError(
                f"Work-package row array {name!r} has invalid shape {rows[name].shape}."
            )
    _normalize_target_rows(
        rows["crop_row_indices"],
        total_rows=int(
            _require_mapping(manifest, "selection")["source_crop_total_rows"]
        ),
    )
    if np.unique(rows["instance_key"]).shape[0] != row_count:
        raise CropPixelWorkPackageError(
            "Work-package instance_key values are not unique."
        )
    pixels = CropPixelWorkPackageArray(payload_path, shape=shape)
    try:
        if verify_payload and _sha256_file(payload_path) != str(
            array.get("sha256") or ""
        ):
            raise CropPixelWorkPackageError(
                "Work-package pixel payload digest mismatch."
            )
        package = CropPixelWorkPackage(
            manifest_path=manifest_file,
            manifest=manifest,
            pixels=pixels,
            crop_row_indices=rows["crop_row_indices"],
            instance_keys=rows["instance_key"],
            source_row_signatures=rows["source_row_signature"],
            frame_indices=rows["frame_indices"],
            roi_coordinates_full=rows["roi_coordinates_full"],
            pixel_sha256=rows["pixel_sha256"],
        )
        binding_for_id = _manifest_source_binding(source)
        expected_id = _logical_package_id(
            crop_run_name=package.crop_run_name,
            crop_row_indices=package.crop_row_indices,
            instance_keys=package.instance_keys,
            source_row_signatures=package.source_row_signatures,
            pixel_sha256=package.pixel_sha256,
            pixel_contract=package.pixel_contract,
            source_binding=binding_for_id,
            materialization_binding=materialization_binding,
        )
        if package.package_id != expected_id:
            raise CropPixelWorkPackageError(
                "Work-package logical package_id is invalid."
            )
        if verify_pixel_rows:
            for start in range(0, row_count, DEFAULT_WORK_PACKAGE_BATCH_ROWS):
                stop = min(start + DEFAULT_WORK_PACKAGE_BATCH_ROWS, row_count)
                observed = _hash_pixel_rows(np.asarray(package.pixels[start:stop]))
                if not np.array_equal(observed, package.pixel_sha256[start:stop]):
                    raise CropPixelWorkPackageError(
                        "Work-package per-row pixel digest mismatch."
                    )
        if root is not None:
            _validate_live_binding(package, root=root)
        return package
    except Exception:
        pixels.close()
        raise


def cleanup_unreferenced_crop_pixel_work_package_generations(
    manifest_path: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """List or delete generation files not referenced by the live manifest.

    The caller owns workflow-liveness checks.  This helper deliberately refuses
    to operate without a readable complete manifest and never removes its two
    referenced artifacts.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    try:
        manifest = _load_strict_json(manifest_file)
        if manifest.get("schema_id") != CROP_PIXEL_WORK_PACKAGE_SCHEMA_ID:
            raise CropPixelWorkPackageError(
                "Unsupported crop pixel work-package schema id."
            )
        array = _require_mapping(manifest, "array")
        rows = _require_mapping(manifest, "rows")
        referenced = {
            _resolve_relative(
                manifest_file, array.get("bin_path"), field="array.bin_path"
            ),
            _resolve_relative(manifest_file, rows.get("path"), field="rows.path"),
        }
    except CropPixelWorkPackageError:
        raise
    except Exception as exc:
        raise CropPixelWorkPackageError(
            "Cannot clean work-package generations without a readable live manifest."
        ) from exc

    candidates = {
        path.resolve()
        for pattern in (
            f"{manifest_file.stem}.*.bin",
            f"{manifest_file.stem}.*.rows.npz",
        )
        for path in manifest_file.parent.glob(pattern)
        if path.is_file()
    }
    unreferenced = sorted(candidates - referenced)
    unreferenced_bytes = int(sum(path.stat().st_size for path in unreferenced))
    if apply:
        for path in unreferenced:
            path.unlink()
    return {
        "manifest_path": str(manifest_file),
        "apply": bool(apply),
        "referenced_files": sorted(str(path) for path in referenced),
        "unreferenced_files": [str(path) for path in unreferenced],
        "unreferenced_file_count": int(len(unreferenced)),
        "unreferenced_bytes": unreferenced_bytes,
    }


__all__ = [
    "CROP_PIXEL_WORK_PACKAGE_SCHEMA_ID",
    "CROP_PIXEL_WORK_PACKAGE_SCHEMA_VERSION",
    "CROP_PIXEL_WORK_PACKAGE_LAYOUT",
    "CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_ID",
    "CROP_PIXEL_WORK_PACKAGE_ROWS_SCHEMA_VERSION",
    "PIXEL_SHA256_ARRAY",
    "DEFAULT_WORK_PACKAGE_BATCH_ROWS",
    "STRICT_SOURCE_BINDING_PROFILE",
    "SIGNED_SOURCE_BINDING_PROFILE",
    "LEGACY_SOURCE_BINDING_PROFILE",
    "CropPixelWorkPackageError",
    "CropPixelWorkPackageArray",
    "CropPixelWorkPackage",
    "build_crop_pixel_work_package_from_source",
    "build_crop_pixel_work_package_from_video_window",
    "build_crop_pixel_work_package",
    "open_crop_pixel_work_package",
    "cleanup_unreferenced_crop_pixel_work_package_generations",
]
