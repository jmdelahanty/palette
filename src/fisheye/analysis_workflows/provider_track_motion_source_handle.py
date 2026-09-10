"""Strict consumer boundary for one Phase 3 provider-motion publication.

The Phase 3 writer publishes immutable, selector-ineligible motion successors
under ``analysis/track_kinematics_runs/provider/<run>``.  This module binds one
caller-supplied concrete run and returns copied read-only arrays.  It never
resolves a selector, chooses a fallback, or changes the archive.

Legacy computation-v1 runs record only their caller-supplied FPS and remain an
explicit compatibility surface.  Computation-v2 runs bind the canonical
recording timing authority; the reader reopens that live clock binding and
validates the complete motion frame domain before reporting authoritative
timing.

The receipt-bound handle is a separate canary-only read profile.  It admits an
exact selector-ineligible child from direct subtree metadata, requires the
atomic publisher's persisted full-validation evidence (and native generic
payload receipts when present), and then reads only requested row slices.  It
is not a default for selector-visible publications.  The exhaustive handle
above remains the deep payload-audit path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
import uuid

import numpy as np
import zarr

from fisheye.analysis_workflows.materializers import provider_track_motion as writer
from fisheye.analysis_workflows.provider_recording_timing_authority import (
    NOMINAL_FRAME_TIME_POLICY_ID,
    PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_ID,
    PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_VERSION,
    ProviderRecordingTimingAuthority,
    ProviderRecordingTimingAuthorityError,
    load_provider_recording_timing_authority,
)
from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    _metadata_inventory,
)
from fisheye.shared.acquisition_frame_clock import (
    ACQUISITION_FRAME_CLOCK_RECORD_ATTR,
    ACQUISITION_FRAME_CLOCK_RUNS_PATH,
    ACQUISITION_FRAME_CLOCK_SCHEMA_ID,
    ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION,
    ACQUISITION_FRAME_CLOCK_SHA256_ATTR,
)
from fisheye.shared.atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
    ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
    ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
    SERIALIZATION_POLICY,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_ID = (
    "palette.provider_track_motion_source_handle"
)
PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_VERSION = 1
PROVIDER_TRACK_MOTION_RECEIPT_BOUND_HANDLE_SCHEMA_ID = (
    "palette.provider_track_motion_receipt_bound_source_handle"
)
PROVIDER_TRACK_MOTION_RECEIPT_BOUND_HANDLE_SCHEMA_VERSION = 1
PROVIDER_TRACK_MOTION_NATIVE_RECEIPT_PROFILE = "native_generic_payload_receipt_v1"
PROVIDER_TRACK_MOTION_ATOMIC_COMPATIBILITY_RECEIPT_PROFILE = (
    "atomic_full_validation_compatibility_v1"
)

_HANDLE_SEAL = object()
_RECEIPT_BOUND_HANDLE_SEAL = object()
_SELECTOR_NAMES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "latest_provider",
        "authoritative_run",
        "authoritative",
        "current",
        "default",
        "fallback",
        "selected",
    }
)
_ATOMIC_VALIDATION_FIELDS = frozenset(
    {"valid", "run_path", "status", "row_count", "track_count", "manifest_sha256"}
)
_ATOMIC_RECEIPT_FIELDS = frozenset(
    {
        "manifest_sha256",
        "source_authority_sha256",
        "tracked_input_sha256",
        "selector_ineligible",
        "schema_id",
        "publisher_contract",
        "policy",
        "serialization_policy",
        "rollback_policy",
        "published_at_utc",
        "host",
        "lsb_jobid",
        "source_zarr",
        "publication_source_run_path",
        "target_run_path",
        "publication_owner_attr",
        "publication_owner_uuid",
        "failed_public_child_policy",
        "hidden_temporary_policy",
        "copy_duration_seconds",
        "physical_copy",
        "parent_attrs_before",
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
        "parent_attrs_after",
    }
)
_ATOMIC_PHYSICAL_COPY_FIELDS = frozenset(
    {
        "backend",
        "verification",
        "file_count",
        "physical_bytes",
        "inventory_sha256",
        "content_sha256",
    }
)


class ProviderTrackMotionSourceHandleError(ValueError):
    """Raised when an exact provider-motion consumer binding is invalid."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _canonical_run_path(value: object) -> tuple[str, str]:
    if type(value) is not str:
        raise ProviderTrackMotionSourceHandleError("run_path must be one exact string.")
    prefix = f"{writer.PROVIDER_TRACK_MOTION_PARENT_PATH}/"
    if (
        not value.startswith(prefix)
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or value != value.strip()
    ):
        raise ProviderTrackMotionSourceHandleError(
            "run_path must name one exact provider/<run> path."
        )
    name = value[len(prefix) :]
    if (
        not name
        or "/" in name
        or name in {".", ".."}
        or name in _SELECTOR_NAMES
        or writer._RUN_NAME_RE.fullmatch(name) is None
    ):
        raise ProviderTrackMotionSourceHandleError(
            "run_path must name one concrete provider run, not a selector, "
            "fallback, nested path, or ambiguous name."
        )
    canonical = f"{writer.PROVIDER_TRACK_MOTION_PARENT_PATH}/{name}"
    if value != canonical:
        raise ProviderTrackMotionSourceHandleError("run_path is not canonical.")
    return canonical, name


def _node(group: Any, path: str) -> Any:
    current = group
    for component in path.split("/"):
        current = current[component]
    return current


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderTrackMotionSourceHandleError(f"{name} must be an object.")
    return value


def _require_digest(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProviderTrackMotionSourceHandleError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _require_exact_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderTrackMotionSourceHandleError(
            f"{name} must be one exact nonempty string."
        )
    return value


def _read_metadata_document(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderTrackMotionSourceHandleError(
            f"Cannot read {name} metadata at {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ProviderTrackMotionSourceHandleError(
            f"{name} metadata must be one object."
        )
    return value


def _provider_parent_attrs(archive: Path) -> Mapping[str, Any]:
    parent_path = writer.PROVIDER_TRACK_MOTION_PARENT_PATH
    direct_document = _read_metadata_document(
        archive.joinpath(*parent_path.split("/"), "zarr.json"),
        name="provider namespace direct",
    )
    direct_attrs = direct_document.get("attributes")
    if (
        direct_document.get("node_type") != "group"
        or not isinstance(direct_attrs, Mapping)
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion direct namespace metadata is not a group declaration."
        )
    selector_attrs = set(writer._SELECTOR_ATTRS).intersection(direct_attrs)
    if selector_attrs:
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion namespace contains forbidden selector attributes: "
            f"{sorted(selector_attrs)!r}."
        )
    return _freeze(direct_attrs)


def _parse_utc_timestamp(value: Any, *, name: str) -> str:
    text = _require_exact_text(value, name=name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ProviderTrackMotionSourceHandleError(
            f"{name} must be an ISO-8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProviderTrackMotionSourceHandleError(
            f"{name} must include an explicit timezone."
        )
    return text


def _manifest_array_records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("arrays")
    if not isinstance(raw, list):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion manifest has no exact array roster."
        )
    records: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion manifest array record is malformed."
            )
        path = item.get("path")
        if type(path) is not str or not path or path in records:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion manifest array identity is invalid."
            )
        records[path] = _freeze(item)
    return records


def _validate_atomic_publication_receipt(
    attrs: Mapping[str, Any],
    *,
    archive: Path,
    run_path: str,
    payload: Mapping[str, Any],
    manifest_sha256: str,
    live_parent_attrs: Mapping[str, Any],
) -> dict[str, Any]:
    raw = attrs.get("cluster_output_staging")
    if not isinstance(raw, Mapping):
        raise ProviderTrackMotionSourceHandleError(
            "Receipt-bound provider motion requires its atomic publication receipt."
        )
    receipt = _thaw(raw)
    if set(receipt) != _ATOMIC_RECEIPT_FIELDS:
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic publication receipt field set is inexact."
        )
    source_record = _require_mapping(
        payload.get("source_authority"), name="provider source authority"
    )
    tracked_record = _require_mapping(
        payload.get("tracked_input"), name="provider tracked input"
    )
    if (
        receipt.get("schema_id") != writer.PROVIDER_TRACK_MOTION_SCHEMA_ID
        or receipt.get("publisher_contract")
        != {
            "schema_id": ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
            "schema_version": ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
        }
        or receipt.get("policy") != writer.PROVIDER_TRACK_MOTION_PUBLICATION_POLICY
        or receipt.get("serialization_policy") != SERIALIZATION_POLICY
        or receipt.get("rollback_policy")
        != "retain_failed_tombstone_leave_parent_selectors_untouched"
        or receipt.get("failed_public_child_policy")
        != "retain_owner_bound_selector_ineligible_tombstone"
        or receipt.get("hidden_temporary_policy")
        != "same_parent_hidden_sibling_then_os_replace"
        or receipt.get("selector_ineligible") is not True
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("source_authority_sha256") != source_record.get("sha256")
        or receipt.get("tracked_input_sha256") != tracked_record.get("sha256")
        or receipt.get("source_zarr") != str(archive)
        or receipt.get("target_run_path")
        != str(archive.joinpath(*run_path.split("/")))
        or receipt.get("publication_owner_attr") != ATOMIC_PUBLICATION_OWNER_ATTR
        or attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
        != receipt.get("publication_owner_uuid")
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic publication receipt identity or binding is stale."
        )
    try:
        owner = str(uuid.UUID(str(receipt["publication_owner_uuid"])))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic publication owner is not a UUID."
        ) from exc
    if owner != receipt["publication_owner_uuid"]:
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic publication owner UUID is not canonical."
        )
    _parse_utc_timestamp(receipt.get("published_at_utc"), name="published_at_utc")
    _require_exact_text(receipt.get("host"), name="publication host")
    job = receipt.get("lsb_jobid")
    if job is not None:
        _require_exact_text(job, name="publication LSF job ID")
    duration = receipt.get("copy_duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic copy duration is invalid."
        )
    local_path = Path(
        _require_exact_text(
            receipt.get("publication_source_run_path"),
            name="publication source run path",
        )
    )
    if not local_path.is_absolute() or tuple(local_path.parts[-len(run_path.split("/")) :]) != tuple(
        run_path.split("/")
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic publication source names another run."
        )
    physical = receipt.get("physical_copy")
    if not isinstance(physical, Mapping) or set(physical) != _ATOMIC_PHYSICAL_COPY_FIELDS:
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic physical-copy receipt is malformed."
        )
    if (
        physical.get("backend") not in {"python", "rsync"}
        or physical.get("verification")
        not in {"sha256_all_physical_files", "rsync_checksum_dry_run"}
        or type(physical.get("file_count")) is not int
        or physical["file_count"] <= 0
        or type(physical.get("physical_bytes")) is not int
        or physical["physical_bytes"] < 0
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic physical-copy evidence is unsupported."
        )
    _require_digest(physical.get("inventory_sha256"), name="physical inventory")
    _require_digest(physical.get("content_sha256"), name="physical content")

    records = _manifest_array_records(payload)
    expected_validation = {
        "valid": True,
        "run_path": run_path,
        "status": RUN_STATUS_COMPLETE,
        "row_count": int(records["track_sample_key"]["shape"][0]),
        "track_count": int(records["track_ids"]["shape"][0]),
        "manifest_sha256": manifest_sha256,
    }
    for name in (
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
    ):
        validation = receipt.get(name)
        if (
            not isinstance(validation, Mapping)
            or set(validation) != _ATOMIC_VALIDATION_FIELDS
            or _thaw(validation) != expected_validation
        ):
            raise ProviderTrackMotionSourceHandleError(
                f"Provider atomic {name} evidence is absent or stale."
            )
    before = receipt.get("parent_attrs_before")
    after = receipt.get("parent_attrs_after")
    parent_path = writer.PROVIDER_TRACK_MOTION_PARENT_PATH
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or set(before) != {parent_path}
        or _thaw(before) != _thaw(after)
        or not isinstance(before[parent_path], Mapping)
        or set(writer._SELECTOR_ATTRS).intersection(before[parent_path])
        or _thaw(after[parent_path]) != _thaw(live_parent_attrs)
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider atomic publication did not preserve the current "
            "selector-free namespace generation."
        )
    return {
        "schema_id": ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
        "schema_version": ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
        "receipt_sha256": canonical_json_sha256(receipt),
        "publication_owner_uuid": owner,
        "physical_copy_content_sha256": physical["content_sha256"],
        "full_validation_count": 4,
    }


def _binding(
    payload: Mapping[str, Any],
    name: str,
) -> tuple[Mapping[str, Any], str]:
    value = _require_mapping(payload.get(name), name=f"provider {name} binding")
    if set(value) != {"record", "sha256"}:
        raise ProviderTrackMotionSourceHandleError(
            f"Provider {name} binding has an unexpected field set."
        )
    record = _require_mapping(value.get("record"), name=f"provider {name} record")
    digest = _require_digest(value.get("sha256"), name=f"provider {name} digest")
    if canonical_json_sha256(_thaw(record)) != digest:
        raise ProviderTrackMotionSourceHandleError(
            f"Provider {name} record digest is stale."
        )
    return _freeze(record), digest


def _temporal_authority(
    computation: Mapping[str, Any],
    *,
    archive: Path,
    use_consolidated: bool,
) -> tuple[
    Mapping[str, Any] | None,
    str | None,
    str,
    bool,
    ProviderRecordingTimingAuthority | None,
]:
    """Resolve a claimed timing binding against the live recording clock."""

    raw = computation.get("temporal_authority")
    if raw is None:
        parameters = computation.get("parameters")
        fps = parameters.get("fps") if isinstance(parameters, Mapping) else None
        if isinstance(fps, (int, float)) and not isinstance(fps, bool):
            return (
                None,
                None,
                "compatibility_caller_fps_only",
                False,
                None,
            )
        return None, None, "missing", False, None
    binding = _require_mapping(raw, name="provider temporal_authority")
    if set(binding) != {"record", "sha256"}:
        raise ProviderTrackMotionSourceHandleError(
            "Provider temporal_authority binding has an unexpected field set."
        )
    record = _require_mapping(
        binding.get("record"), name="provider temporal_authority record"
    )
    digest = _require_digest(
        binding.get("sha256"), name="provider temporal_authority digest"
    )
    if canonical_json_sha256(_thaw(record)) != digest:
        raise ProviderTrackMotionSourceHandleError(
            "Provider temporal_authority record digest is stale."
        )
    try:
        authority = load_provider_recording_timing_authority(
            archive,
            required=True,
            use_consolidated=use_consolidated,
            expected_sha256=digest,
        )
    except ProviderRecordingTimingAuthorityError as exc:
        raise ProviderTrackMotionSourceHandleError(
            f"Provider recording timing authority is stale or invalid: {exc}"
        ) from exc
    assert authority is not None
    parameters = computation.get("parameters")
    fps = parameters.get("fps") if isinstance(parameters, Mapping) else None
    if (
        _thaw(record) != _thaw(authority.record)
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or float(fps) != authority.nominal_fps
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider temporal-authority record or nominal FPS differs from the "
            "live recording timing authority."
        )
    return (
        _freeze(record),
        digest,
        "bound_live_recording_timing_authority",
        True,
        authority,
    )


def _read_array(run: Any, path: str) -> np.ndarray:
    try:
        node = _node(run, path)
    except (KeyError, ValueError, TypeError) as exc:
        raise ProviderTrackMotionSourceHandleError(
            f"Provider-motion array is missing: {path!r}."
        ) from exc
    if not isinstance(node, zarr.Array):
        raise ProviderTrackMotionSourceHandleError(
            f"Provider-motion path is not an array: {path!r}."
        )
    return _readonly(node[:])


def _validate_exact_lengths(arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    row_count = int(arrays["track_sample_key"].shape[0])
    track_count = int(arrays["track_ids"].shape[0])
    offsets = arrays["track_row_offsets"]
    if offsets.shape != (track_count + 1,):
        raise ProviderTrackMotionSourceHandleError(
            "Provider track_row_offsets length does not equal track_count + 1."
        )
    if (
        offsets.size == 0
        or int(offsets[0]) != 0
        or int(offsets[-1]) != row_count
        or np.any(offsets < 0)
        or np.any(offsets > row_count)
        or np.any(np.diff(offsets) < 0)
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider track_row_offsets are outside the exact row domain."
        )
    for path in (*writer._PIXEL_SAMPLE_ARRAYS, *writer._PHYSICAL_SAMPLE_ARRAYS):
        if path in arrays and arrays[path].shape[0] != row_count:
            raise ProviderTrackMotionSourceHandleError(
                f"Provider row array {path!r} is not aligned to track samples."
            )
    second_count = int(arrays["per_second/track_second_key"].shape[0])
    for path in (*writer._PIXEL_PER_SECOND_ARRAYS, *writer._PHYSICAL_PER_SECOND_ARRAYS):
        if path in arrays and arrays[path].shape[0] != second_count:
            raise ProviderTrackMotionSourceHandleError(
                f"Provider per-second array {path!r} is not aligned."
            )
    return row_count, track_count, second_count


def _validate_lineage_and_offsets(arrays: Mapping[str, np.ndarray]) -> None:
    row_count, _track_count, _second_count = _validate_exact_lengths(arrays)
    keys = arrays["track_sample_key"]
    if keys.shape != (row_count, 2) or np.unique(keys, axis=0).shape[0] != row_count:
        raise ProviderTrackMotionSourceHandleError(
            "Provider track_sample_key is not a unique [track, frame] row identity."
        )
    if not np.array_equal(keys[:, 1], arrays["source_acquisition_frame_index"]):
        raise ProviderTrackMotionSourceHandleError(
            "Provider track_sample_key disagrees with acquisition-frame lineage."
        )
    offsets = arrays["track_row_offsets"]
    for index, track_id in enumerate(arrays["track_ids"]):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if not np.all(keys[start:stop, 0] == track_id):
            raise ProviderTrackMotionSourceHandleError(
                "Provider track-row offsets do not delimit their declared tracks."
            )

    for path in (
        "source_provider_row_index",
        "source_position_row_index",
        "source_body_frame_row_index",
        "source_tracking_row_index",
    ):
        values = arrays[path]
        if np.any(values < 0) or not np.array_equal(
            np.sort(values), np.arange(row_count, dtype=values.dtype)
        ):
            raise ProviderTrackMotionSourceHandleError(
                f"Provider {path!r} is not an exact source-row permutation."
            )
    observation_keys = arrays["source_observation_instance_key"]
    if np.unique(observation_keys).shape[0] != row_count:
        raise ProviderTrackMotionSourceHandleError(
            "Provider source observation identities are duplicated."
        )
    if np.any(arrays["source_acquisition_frame_index"] < 0):
        raise ProviderTrackMotionSourceHandleError(
            "Provider acquisition-frame lineage contains a negative frame."
        )


def _validate_independent_validity(arrays: Mapping[str, np.ndarray]) -> None:
    for path in (
        "position_source_valid",
        "body_frame_source_valid",
        "linear_sample_valid",
        "angular_sample_valid",
        "transition_valid",
    ):
        if arrays[path].dtype != np.dtype(bool):
            raise ProviderTrackMotionSourceHandleError(
                f"Provider validity array {path!r} is not exact bool."
            )
    if np.any(arrays["linear_sample_valid"] & ~arrays["position_source_valid"]):
        raise ProviderTrackMotionSourceHandleError(
            "Provider linear validity exceeds position-source validity."
        )
    if np.any(arrays["angular_sample_valid"] & ~arrays["body_frame_source_valid"]):
        raise ProviderTrackMotionSourceHandleError(
            "Provider angular validity exceeds body-frame validity."
        )
    if "sample_valid" in arrays:
        raise ProviderTrackMotionSourceHandleError(
            "Provider motion must not synthesize or publish generic sample_valid."
        )


def _verification_digest(
    *,
    run_path: str,
    manifest_sha256: str,
    arrays: Mapping[str, np.ndarray],
    timing_status: str,
) -> str:
    return _verification_digest_from_array_digests(
        run_path=run_path,
        manifest_sha256=manifest_sha256,
        array_digests={
            path: sha256_array(value) for path, value in sorted(arrays.items())
        },
        timing_status=timing_status,
    )


def _verification_digest_from_array_digests(
    *,
    run_path: str,
    manifest_sha256: str,
    array_digests: Mapping[str, str],
    timing_status: str,
) -> str:
    return canonical_json_sha256(
        {
            "schema_id": PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_ID,
            "schema_version": PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_VERSION,
            "run_path": run_path,
            "manifest_sha256": manifest_sha256,
            "timing_status": timing_status,
            "arrays": dict(sorted(array_digests.items())),
        }
    )


@dataclass(frozen=True, init=False, eq=False)
class ProviderTrackMotionSourceHandle:
    """Immutable, verified snapshot of one exact provider-motion run."""

    analysis_zarr_path: Path
    run_path: str
    run_name: str
    provider_manifest: Mapping[str, Any] = field(repr=False)
    provider_manifest_sha256: str
    selector_eligible: bool
    source_authority_record: Mapping[str, Any] = field(repr=False)
    source_authority_sha256: str
    tracked_input_record: Mapping[str, Any] = field(repr=False)
    tracked_input_sha256: str
    physical_authority_record: Mapping[str, Any] | None = field(repr=False)
    physical_authority_sha256: str | None
    physical_authority_status: str
    computation_record: Mapping[str, Any] = field(repr=False)
    computation_sha256: str
    temporal_authority_record: Mapping[str, Any] | None = field(repr=False)
    temporal_authority_sha256: str | None
    temporal_authority_status: str
    timing_is_authoritative: bool
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    row_count: int
    track_count: int
    per_second_count: int
    verification_digest: str
    _use_consolidated: bool = field(repr=False, compare=False)
    _require_authoritative_timing: bool = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _HANDLE_SEAL:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion source handles can only be minted by the strict loader."
            )
        for name, value in values.items():
            if name == "arrays":
                value = MappingProxyType(
                    {path: _readonly(array) for path, array in value.items()}
                )
            elif name.endswith("_record") or name == "provider_manifest":
                if value is not None:
                    value = _freeze(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_verification_seal", _HANDLE_SEAL)

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Compatibility alias for the exact provider manifest snapshot."""

        return self.provider_manifest

    @property
    def manifest_sha256(self) -> str:
        return self.provider_manifest_sha256

    @property
    def provider_manifest_digest(self) -> str:
        return self.provider_manifest_sha256

    @property
    def source_path(self) -> Path:
        return self.analysis_zarr_path

    @property
    def source_authority(self) -> Mapping[str, Any]:
        return self.source_authority_record

    @property
    def tracked_input(self) -> Mapping[str, Any]:
        return self.tracked_input_record

    @property
    def computation(self) -> Mapping[str, Any]:
        return self.computation_record

    @property
    def physical_authority(self) -> Mapping[str, Any] | None:
        return self.physical_authority_record

    @property
    def temporal_authority(self) -> Mapping[str, Any] | None:
        return self.temporal_authority_record

    @property
    def track_ids(self) -> np.ndarray:
        return self.arrays["track_ids"]

    @property
    def track_row_offsets(self) -> np.ndarray:
        return self.arrays["track_row_offsets"]

    @property
    def track_sample_key(self) -> np.ndarray:
        return self.arrays["track_sample_key"]

    @property
    def source_acquisition_frame_index(self) -> np.ndarray:
        return self.arrays["source_acquisition_frame_index"]

    @property
    def source_observation_instance_key(self) -> np.ndarray:
        return self.arrays["source_observation_instance_key"]

    @property
    def source_provider_row_index(self) -> np.ndarray:
        return self.arrays["source_provider_row_index"]

    @property
    def source_position_row_index(self) -> np.ndarray:
        return self.arrays["source_position_row_index"]

    @property
    def source_body_frame_row_index(self) -> np.ndarray:
        return self.arrays["source_body_frame_row_index"]

    @property
    def source_tracking_row_index(self) -> np.ndarray:
        return self.arrays["source_tracking_row_index"]

    @property
    def time_seconds(self) -> np.ndarray:
        return self.arrays["time_seconds"]

    @property
    def delta_seconds(self) -> np.ndarray:
        return self.arrays["delta_seconds"]

    @property
    def positions_px(self) -> np.ndarray:
        return self.arrays["positions_px"]

    @property
    def positions_mm(self) -> np.ndarray | None:
        return self.arrays.get("positions_mm")

    @property
    def position_source_valid(self) -> np.ndarray:
        return self.arrays["position_source_valid"]

    @property
    def body_frame_source_valid(self) -> np.ndarray:
        return self.arrays["body_frame_source_valid"]

    @property
    def linear_sample_valid(self) -> np.ndarray:
        return self.arrays["linear_sample_valid"]

    @property
    def angular_sample_valid(self) -> np.ndarray:
        return self.arrays["angular_sample_valid"]

    @property
    def transition_valid(self) -> np.ndarray:
        return self.arrays["transition_valid"]

    @property
    def linear_sample_reason_code(self) -> np.ndarray:
        return self.arrays["linear_sample_reason_code"]

    @property
    def angular_sample_reason_code(self) -> np.ndarray:
        return self.arrays["angular_sample_reason_code"]

    @property
    def transition_reason_code(self) -> np.ndarray:
        return self.arrays["transition_reason_code"]

    def array(self, path: str) -> np.ndarray:
        """Return one copied read-only array snapshot by its exact path."""

        try:
            return self.arrays[path]
        except KeyError as exc:
            raise KeyError(f"Unknown provider-motion array {path!r}.") from exc

    def assert_current(self) -> None:
        """Reopen the same run and reject mutation or stale consolidation."""

        if self._verification_seal is not _HANDLE_SEAL:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion source handle verification seal is absent."
            )
        refreshed = load_provider_track_motion_source_handle(
            self.analysis_zarr_path,
            self.run_path,
            use_consolidated=self._use_consolidated,
            expected_manifest_sha256=self.provider_manifest_sha256,
            require_authoritative_timing=self._require_authoritative_timing,
        )
        if refreshed.verification_digest != self.verification_digest:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion source changed after the handle was sealed."
            )

    def assert_verified(self) -> None:
        self.assert_current()


@dataclass(frozen=True, init=False, eq=False)
class ReceiptBoundProviderTrackMotionSourceHandle:
    """Receipt-admitted provider metadata with bounded on-demand array access."""

    analysis_zarr_path: Path
    run_path: str
    run_name: str
    provider_manifest: Mapping[str, Any] = field(repr=False)
    provider_manifest_sha256: str
    selector_eligible: bool
    source_authority_record: Mapping[str, Any] = field(repr=False)
    source_authority_sha256: str
    tracked_input_record: Mapping[str, Any] = field(repr=False)
    tracked_input_sha256: str
    physical_authority_record: Mapping[str, Any] | None = field(repr=False)
    physical_authority_sha256: str | None
    physical_authority_status: str
    computation_record: Mapping[str, Any] = field(repr=False)
    computation_sha256: str
    temporal_authority_record: Mapping[str, Any] | None = field(repr=False)
    temporal_authority_sha256: str | None
    temporal_authority_status: str
    timing_is_authoritative: bool
    row_count: int
    track_count: int
    per_second_count: int
    receipt_profile: str
    receipt_digest: str
    atomic_publication_receipt_sha256: str
    payload_integrity_receipt_sha256: str | None
    payload_validation_receipt_sha256: str | None
    metadata_declarations_sha256: str
    verification_digest: str
    admission_digest: str
    metadata_evidence: Mapping[str, Any] = field(repr=False)
    _array_declarations: Mapping[str, Mapping[str, Any]] = field(
        repr=False, compare=False
    )
    _track_ids: np.ndarray = field(repr=False, compare=False)
    _track_row_offsets: np.ndarray = field(repr=False, compare=False)
    _run_group: Any = field(repr=False, compare=False)
    _timing_authority: ProviderRecordingTimingAuthority | None = field(
        repr=False, compare=False
    )
    _require_authoritative_timing: bool = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _RECEIPT_BOUND_HANDLE_SEAL:
            raise ProviderTrackMotionSourceHandleError(
                "Receipt-bound provider handles can only be minted by the "
                "receipt loader."
            )
        for name, value in values.items():
            if name in {"_track_ids", "_track_row_offsets"}:
                value = _readonly(value)
            elif name in {
                "provider_manifest",
                "source_authority_record",
                "tracked_input_record",
                "physical_authority_record",
                "computation_record",
                "temporal_authority_record",
                "metadata_evidence",
                "_array_declarations",
            } and value is not None:
                value = _freeze(value)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "_verification_seal", _RECEIPT_BOUND_HANDLE_SEAL
        )

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.provider_manifest

    @property
    def manifest_sha256(self) -> str:
        return self.provider_manifest_sha256

    @property
    def provider_manifest_digest(self) -> str:
        return self.provider_manifest_sha256

    @property
    def source_path(self) -> Path:
        return self.analysis_zarr_path

    @property
    def source_authority(self) -> Mapping[str, Any]:
        return self.source_authority_record

    @property
    def tracked_input(self) -> Mapping[str, Any]:
        return self.tracked_input_record

    @property
    def computation(self) -> Mapping[str, Any]:
        return self.computation_record

    @property
    def physical_authority(self) -> Mapping[str, Any] | None:
        return self.physical_authority_record

    @property
    def temporal_authority(self) -> Mapping[str, Any] | None:
        return self.temporal_authority_record

    @property
    def track_ids(self) -> np.ndarray:
        return self._track_ids

    @property
    def track_row_offsets(self) -> np.ndarray:
        return self._track_row_offsets

    @property
    def available_arrays(self) -> tuple[str, ...]:
        return tuple(sorted(self._array_declarations))

    def has_array(self, path: str) -> bool:
        return type(path) is str and path in self._array_declarations

    def array_slice(self, path: str, rows: slice | None = None) -> np.ndarray:
        """Read one declared first-axis interval under the immutable receipt."""

        if type(path) is not str or path not in self._array_declarations:
            raise KeyError(f"Unknown provider-motion array {path!r}.")
        declaration = self._array_declarations[path]
        shape = tuple(int(value) for value in declaration["shape"])
        dtype = np.dtype(declaration["dtype"])
        if not shape:
            raise ProviderTrackMotionSourceHandleError(
                f"Provider-motion array {path!r} has no first-axis slice contract."
            )
        if rows is None:
            start, stop = 0, shape[0]
        else:
            if type(rows) is not slice or rows.step not in (None, 1):
                raise ProviderTrackMotionSourceHandleError(
                    "Provider-motion lazy reads require one contiguous row slice."
                )
            start = 0 if rows.start is None else rows.start
            stop = shape[0] if rows.stop is None else rows.stop
            if (
                isinstance(start, bool)
                or isinstance(stop, bool)
                or type(start) is not int
                or type(stop) is not int
                or not (0 <= start <= stop <= shape[0])
            ):
                raise ProviderTrackMotionSourceHandleError(
                    "Provider-motion lazy row bounds leave the declared array domain."
                )
        try:
            node = _node(self._run_group, path)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderTrackMotionSourceHandleError(
                f"Provider-motion array is missing: {path!r}."
            ) from exc
        if (
            not isinstance(node, zarr.Array)
            or np.dtype(node.dtype) != dtype
            or tuple(int(value) for value in node.shape) != shape
        ):
            raise ProviderTrackMotionSourceHandleError(
                f"Provider-motion array metadata changed: {path!r}."
            )
        trailing = (slice(None),) * (len(shape) - 1)
        values = np.asarray(node[(slice(start, stop), *trailing)])
        expected_shape = (stop - start, *shape[1:])
        if values.dtype != dtype or values.shape != expected_shape:
            raise ProviderTrackMotionSourceHandleError(
                f"Provider-motion lazy read changed dtype or shape: {path!r}."
            )
        return _readonly(values)

    def array(self, path: str) -> np.ndarray:
        return self.array_slice(path)

    @property
    def source_acquisition_frame_index(self) -> np.ndarray:
        return self.array("source_acquisition_frame_index")

    @property
    def time_seconds(self) -> np.ndarray:
        return self.array("time_seconds")

    @property
    def positions_px(self) -> np.ndarray:
        return self.array("positions_px")

    @property
    def positions_mm(self) -> np.ndarray | None:
        return self.array("positions_mm") if self.has_array("positions_mm") else None

    @property
    def linear_sample_valid(self) -> np.ndarray:
        return self.array("linear_sample_valid")

    @property
    def angular_sample_valid(self) -> np.ndarray:
        return self.array("angular_sample_valid")

    @property
    def transition_valid(self) -> np.ndarray:
        return self.array("transition_valid")

    @property
    def linear_sample_reason_code(self) -> np.ndarray:
        return self.array("linear_sample_reason_code")

    @property
    def angular_sample_reason_code(self) -> np.ndarray:
        return self.array("angular_sample_reason_code")

    @property
    def transition_reason_code(self) -> np.ndarray:
        return self.array("transition_reason_code")

    def assert_current(self) -> None:
        if self._verification_seal is not _RECEIPT_BOUND_HANDLE_SEAL:
            raise ProviderTrackMotionSourceHandleError(
                "Receipt-bound provider handle verification seal is absent."
            )
        refreshed = load_receipt_bound_provider_track_motion_source_handle(
            self.analysis_zarr_path,
            self.run_path,
            expected_manifest_sha256=self.provider_manifest_sha256,
            require_authoritative_timing=self._require_authoritative_timing,
            timing_authority=self._timing_authority,
        )
        if (
            refreshed.admission_digest != self.admission_digest
            or refreshed.verification_digest != self.verification_digest
        ):
            raise ProviderTrackMotionSourceHandleError(
                "Receipt-bound provider motion changed after admission."
            )

    def assert_verified(self) -> None:
        self.assert_current()


def _receipt_bound_temporal_authority(
    computation: Mapping[str, Any],
    *,
    archive: Path,
    timing_authority: ProviderRecordingTimingAuthority | None,
) -> tuple[
    Mapping[str, Any] | None,
    str | None,
    str,
    bool,
    ProviderRecordingTimingAuthority | None,
    Mapping[str, Any] | None,
]:
    raw = computation.get("temporal_authority")
    if raw is None:
        parameters = computation.get("parameters")
        fps = parameters.get("fps") if isinstance(parameters, Mapping) else None
        status = (
            "compatibility_caller_fps_only"
            if isinstance(fps, (int, float)) and not isinstance(fps, bool)
            else "missing"
        )
        return None, None, status, False, None, None
    binding = _require_mapping(raw, name="provider temporal_authority")
    if set(binding) != {"record", "sha256"}:
        raise ProviderTrackMotionSourceHandleError(
            "Provider temporal_authority binding has an unexpected field set."
        )
    record = _require_mapping(
        binding.get("record"), name="provider temporal_authority record"
    )
    digest = _require_digest(
        binding.get("sha256"), name="provider temporal_authority digest"
    )
    parameters = computation.get("parameters")
    fps = parameters.get("fps") if isinstance(parameters, Mapping) else None
    expected_record_fields = {
        "schema_id",
        "schema_version",
        "policy_id",
        "recording_id",
        "camera_id",
        "nominal_fps",
        "frame_count",
        "acquisition_frame_clock",
        "source_video_metadata",
        "numerical_semantics",
    }
    if (
        set(record) != expected_record_fields
        or record.get("schema_id")
        != PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_ID
        or record.get("schema_version")
        != PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_VERSION
        or record.get("policy_id") != NOMINAL_FRAME_TIME_POLICY_ID
        or canonical_json_sha256(_thaw(record)) != digest
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or isinstance(record.get("nominal_fps"), bool)
        or not isinstance(record.get("nominal_fps"), (int, float))
        or float(fps) != float(record["nominal_fps"])
        or type(record.get("frame_count")) is not int
        or int(record["frame_count"]) <= 0
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider temporal-authority record or nominal FPS is invalid."
        )
    if timing_authority is not None:
        if (
            type(timing_authority) is not ProviderRecordingTimingAuthority
            or timing_authority.analysis_zarr_path != archive
            or digest != timing_authority.sha256
            or _thaw(record) != _thaw(timing_authority.record)
            or float(fps) != timing_authority.nominal_fps
        ):
            raise ProviderTrackMotionSourceHandleError(
                "Provider temporal authority differs from the prebound live clock."
            )
        evidence = {
            "profile": "prebound_exhaustive_recording_timing_authority_v1",
            "record_sha256": digest,
        }
        return (
            _freeze(record),
            digest,
            "bound_live_recording_timing_authority",
            True,
            timing_authority,
            _freeze(evidence),
        )

    clock = _require_mapping(
        record.get("acquisition_frame_clock"),
        name="provider acquisition frame clock",
    )
    if set(clock) != {"schema_id", "run_path", "record_sha256", "array_sha256"}:
        raise ProviderTrackMotionSourceHandleError(
            "Provider acquisition-frame-clock binding has an unexpected field set."
        )
    run_path = _require_exact_text(
        clock.get("run_path"), name="provider acquisition frame clock run path"
    )
    prefix = f"{ACQUISITION_FRAME_CLOCK_RUNS_PATH}/"
    run_name = run_path[len(prefix) :] if run_path.startswith(prefix) else ""
    if not run_name or "/" in run_name or run_name in _SELECTOR_NAMES:
        raise ProviderTrackMotionSourceHandleError(
            "Provider acquisition-frame-clock binding is not one exact child."
        )
    clock_digest = _require_digest(
        clock.get("record_sha256"), name="provider acquisition clock digest"
    )
    expected_array_dtypes = {
        "recording_frame_id": np.dtype("int64"),
        "parent_frame_index": np.dtype("int64"),
        "camera_timestamp_ns": np.dtype("int64"),
        "system_timestamp_ns": np.dtype("int64"),
        "camera_timestamp_valid": np.dtype("bool"),
        "system_timestamp_valid": np.dtype("bool"),
    }
    bound_arrays = _require_mapping(
        clock.get("array_sha256"), name="provider acquisition clock arrays"
    )
    if set(bound_arrays) != set(expected_array_dtypes):
        raise ProviderTrackMotionSourceHandleError(
            "Provider acquisition-frame-clock array roster is inexact."
        )
    for name in expected_array_dtypes:
        _require_digest(
            bound_arrays.get(name), name=f"provider acquisition clock {name}"
        )

    parent_document = _read_metadata_document(
        archive.joinpath(*ACQUISITION_FRAME_CLOCK_RUNS_PATH.split("/"), "zarr.json"),
        name="acquisition frame clock parent",
    )
    parent_attrs = parent_document.get("attributes")
    if (
        parent_document.get("node_type") != "group"
        or not isinstance(parent_attrs, Mapping)
        or parent_attrs.get("latest") != run_name
        or parent_attrs.get("latest_complete") != run_name
        or any(
            name in parent_attrs
            for name in ("latest_pending", "authoritative_run", "fallback")
        )
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider acquisition-frame-clock selector generation is stale or ambiguous."
        )
    run_directory = archive.joinpath(*run_path.split("/"))
    run_document = _read_metadata_document(
        run_directory / "zarr.json",
        name="acquisition frame clock run",
    )
    run_attrs = run_document.get("attributes")
    if (
        run_document.get("node_type") != "group"
        or not isinstance(run_attrs, Mapping)
        or run_attrs.get("schema_id") != ACQUISITION_FRAME_CLOCK_SCHEMA_ID
        or run_attrs.get("schema_version") != ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION
        or run_attrs.get("immutable") is not True
        or run_attrs.get("stage_selector_eligible") is not True
        or run_attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or run_attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or run_attrs.get("palette_run_name") != run_name
        or run_attrs.get(ACQUISITION_FRAME_CLOCK_SHA256_ATTR) != clock_digest
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider acquisition-frame-clock run lifecycle is invalid."
        )
    live_clock_record = _require_mapping(
        run_attrs.get(ACQUISITION_FRAME_CLOCK_RECORD_ATTR),
        name="live acquisition frame clock record",
    )
    if (
        canonical_json_sha256(_thaw(live_clock_record)) != clock_digest
        or live_clock_record.get("schema_id") != ACQUISITION_FRAME_CLOCK_SCHEMA_ID
        or live_clock_record.get("schema_version")
        != ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION
        or live_clock_record.get("camera_id") != record.get("camera_id")
        or live_clock_record.get("row_count") != record.get("frame_count")
        or _thaw(live_clock_record.get("array_sha256")) != _thaw(bound_arrays)
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider acquisition-frame-clock record differs from its bound source."
        )
    frame_count = int(record["frame_count"])
    for name, expected_dtype in expected_array_dtypes.items():
        declaration = _read_metadata_document(
            run_directory / name / "zarr.json",
            name=f"acquisition frame clock array {name}",
        )
        try:
            observed_dtype = np.dtype(declaration.get("data_type"))
        except TypeError as exc:
            raise ProviderTrackMotionSourceHandleError(
                f"Acquisition frame clock array {name!r} dtype is invalid."
            ) from exc
        if (
            declaration.get("node_type") != "array"
            or declaration.get("shape") != [frame_count]
            or observed_dtype != expected_dtype
        ):
            raise ProviderTrackMotionSourceHandleError(
                f"Acquisition frame clock array {name!r} metadata changed."
            )
    metadata_inventory = _metadata_inventory(run_directory)
    evidence = {
        "profile": "provider_bound_selected_immutable_clock_metadata_v1",
        "record_sha256": digest,
        "clock_record_sha256": clock_digest,
        "clock_parent_sha256": canonical_json_sha256(parent_document),
        "clock_metadata_inventory_sha256": metadata_inventory["inventory_sha256"],
        "historical_payload_validation": (
            "bound_by_provider_manifest_and_atomic_full_validation"
        ),
        "normal_load_clock_payload_rehash": False,
    }
    return (
        _freeze(record),
        digest,
        "bound_live_recording_timing_authority",
        True,
        None,
        _freeze(evidence),
    )


def load_receipt_bound_provider_track_motion_source_handle(
    analysis_zarr: str | Path,
    run_path: str,
    *,
    expected_manifest_sha256: str | None = None,
    require_authoritative_timing: bool = False,
    timing_authority: ProviderRecordingTimingAuthority | None = None,
) -> ReceiptBoundProviderTrackMotionSourceHandle:
    """Admit one exact canary from receipts and defer payload reads to slices.

    The direct metadata view is accepted only for the selector-ineligible
    provider namespace and is bound to the atomic publisher's historical full
    validation, exact parent snapshot, current immutable timing child, and any
    native payload receipt pair.  Selector-visible consumers must use their
    consolidated publication contract; exhaustive revalidation remains
    available through :func:`load_provider_track_motion_source_handle`.
    """

    exact_path, run_name = _canonical_run_path(run_path)
    if type(require_authoritative_timing) is not bool:
        raise ProviderTrackMotionSourceHandleError(
            "require_authoritative_timing must be the exact boolean."
        )
    if expected_manifest_sha256 is not None:
        _require_digest(
            expected_manifest_sha256, name="expected provider manifest digest"
        )
    archive = Path(analysis_zarr).expanduser().resolve()
    run_directory = archive.joinpath(*exact_path.split("/"))
    try:
        parent_attrs = _provider_parent_attrs(archive)
        run = open_zarr_root(
            run_directory,
            mode="r",
            use_consolidated=False,
        )
        metadata = writer.validate_provider_track_motion_run_metadata(
            run,
            run_directory,
            expected_run_name=run_name,
            expected_status=RUN_STATUS_COMPLETE,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        writer.ProviderTrackMotionError,
    ) as exc:
        raise ProviderTrackMotionSourceHandleError(
            f"Provider-motion receipt admission failed for {exact_path!r}: {exc}"
        ) from exc
    raw_manifest = run.attrs.get(writer.PROVIDER_TRACK_MOTION_MANIFEST_ATTR)
    if not isinstance(raw_manifest, Mapping):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion manifest is missing after metadata admission."
        )
    try:
        payload, _storage = writer._validate_manifest(
            raw_manifest,
            expected_run_name=run_name,
            expected_status=RUN_STATUS_COMPLETE,
        )
        manifest_sha256 = writer.provider_track_motion_manifest_digest(raw_manifest)
    except (KeyError, TypeError, ValueError, writer.ProviderTrackMotionError) as exc:
        raise ProviderTrackMotionSourceHandleError(str(exc)) from exc
    source_record, source_sha256 = _binding(payload, "source_authority")
    tracked_record, tracked_sha256 = _binding(payload, "tracked_input")
    computation_record, computation_sha256 = _binding(payload, "computation")
    physical_binding = _require_mapping(
        payload.get("physical_authority"), name="provider physical_authority"
    )
    physical_status = physical_binding.get("status")
    if physical_status == "bound":
        physical_value = _require_mapping(
            physical_binding.get("record"), name="provider physical authority record"
        )
        physical_sha256 = _require_digest(
            physical_binding.get("sha256"), name="provider physical authority digest"
        )
        if canonical_json_sha256(_thaw(physical_value)) != physical_sha256:
            raise ProviderTrackMotionSourceHandleError(
                "Provider physical authority record digest is stale."
            )
        physical_record = _freeze(physical_value)
    elif physical_binding == {
        "status": "omitted_explicit_pixel_only_canary",
        "record": None,
        "sha256": None,
    }:
        physical_record, physical_sha256 = None, None
    else:
        raise ProviderTrackMotionSourceHandleError(
            "Provider physical authority binding is invalid."
        )
    (
        temporal_record,
        temporal_sha256,
        timing_status,
        timing_authoritative,
        admitted_timing,
        timing_evidence,
    ) = _receipt_bound_temporal_authority(
        computation_record,
        archive=archive,
        timing_authority=timing_authority,
    )
    if require_authoritative_timing and not timing_authoritative:
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion run has no authoritative temporal authority; "
            f"status is {timing_status!r}."
        )
    atomic = _validate_atomic_publication_receipt(
        run.attrs,
        archive=archive,
        run_path=exact_path,
        payload=payload,
        manifest_sha256=manifest_sha256,
        live_parent_attrs=parent_attrs,
    )
    native = metadata.get("payload_receipt")
    if native is None:
        receipt_profile = PROVIDER_TRACK_MOTION_ATOMIC_COMPATIBILITY_RECEIPT_PROFILE
        integrity_sha256 = None
        validation_sha256 = None
    elif isinstance(native, Mapping):
        receipt_profile = PROVIDER_TRACK_MOTION_NATIVE_RECEIPT_PROFILE
        integrity_sha256 = _require_digest(
            native.get("integrity_receipt_sha256"),
            name="provider payload integrity receipt",
        )
        validation_sha256 = _require_digest(
            native.get("validation_receipt_sha256"),
            name="provider payload validation receipt",
        )
    else:  # pragma: no cover - writer validator closes this shape
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion payload receipt evidence is malformed."
        )
    records = _manifest_array_records(payload)
    try:
        track_ids = _read_array(run, "track_ids")
        offsets = _read_array(run, "track_row_offsets")
    except ProviderTrackMotionSourceHandleError:
        raise
    for path, values in (("track_ids", track_ids), ("track_row_offsets", offsets)):
        if sha256_array(values) != records[path]["sha256"]:
            raise ProviderTrackMotionSourceHandleError(
                f"Receipt-bound provider index array changed: {path!r}."
            )
    row_count = int(metadata["row_count"])
    track_count = int(metadata["track_count"])
    if (
        track_ids.shape != (track_count,)
        or np.unique(track_ids).shape[0] != track_count
        or offsets.shape != (track_count + 1,)
        or offsets.size == 0
        or int(offsets[0]) != 0
        or int(offsets[-1]) != row_count
        or np.any(offsets < 0)
        or np.any(offsets > row_count)
        or np.any(np.diff(offsets) < 0)
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Receipt-bound provider track index is invalid."
        )
    array_digests = {
        path: _require_digest(record.get("sha256"), name=f"{path} manifest digest")
        for path, record in records.items()
    }
    verification = _verification_digest_from_array_digests(
        run_path=exact_path,
        manifest_sha256=manifest_sha256,
        array_digests=array_digests,
        timing_status=timing_status,
    )
    metadata_payload = {
        "metadata_mode": "exact_selector_ineligible_direct_receipt_bound_v1",
        "direct_subtree": _metadata_inventory(run_directory),
        "provider_parent_attrs": _thaw(parent_attrs),
        "atomic_publication": atomic,
        "native_payload_receipt": _thaw(native) if native is not None else None,
        "timing_authority": (
            _thaw(timing_evidence) if timing_evidence is not None else None
        ),
    }
    admission_body = {
        "schema_id": PROVIDER_TRACK_MOTION_RECEIPT_BOUND_HANDLE_SCHEMA_ID,
        "schema_version": PROVIDER_TRACK_MOTION_RECEIPT_BOUND_HANDLE_SCHEMA_VERSION,
        "run_path": exact_path,
        "manifest_sha256": manifest_sha256,
        "receipt_profile": receipt_profile,
        "verification_digest": verification,
        "timing_status": timing_status,
        "timing_authority_sha256": temporal_sha256,
        "metadata_evidence": metadata_payload,
    }
    admission_digest = canonical_json_sha256(admission_body)
    receipt_digest = (
        validation_sha256
        if validation_sha256 is not None
        else str(atomic["receipt_sha256"])
    )
    return ReceiptBoundProviderTrackMotionSourceHandle(
        _verification_seal=_RECEIPT_BOUND_HANDLE_SEAL,
        analysis_zarr_path=archive,
        run_path=exact_path,
        run_name=run_name,
        provider_manifest=raw_manifest,
        provider_manifest_sha256=manifest_sha256,
        selector_eligible=False,
        source_authority_record=source_record,
        source_authority_sha256=source_sha256,
        tracked_input_record=tracked_record,
        tracked_input_sha256=tracked_sha256,
        physical_authority_record=physical_record,
        physical_authority_sha256=physical_sha256,
        physical_authority_status=str(physical_status),
        computation_record=computation_record,
        computation_sha256=computation_sha256,
        temporal_authority_record=temporal_record,
        temporal_authority_sha256=temporal_sha256,
        temporal_authority_status=timing_status,
        timing_is_authoritative=timing_authoritative,
        row_count=row_count,
        track_count=track_count,
        per_second_count=int(metadata["per_second_count"]),
        receipt_profile=receipt_profile,
        receipt_digest=receipt_digest,
        atomic_publication_receipt_sha256=str(atomic["receipt_sha256"]),
        payload_integrity_receipt_sha256=integrity_sha256,
        payload_validation_receipt_sha256=validation_sha256,
        metadata_declarations_sha256=str(
            metadata_payload["direct_subtree"]["inventory_sha256"]
        ),
        verification_digest=verification,
        admission_digest=admission_digest,
        metadata_evidence=metadata_payload,
        _array_declarations=records,
        _track_ids=track_ids,
        _track_row_offsets=offsets,
        _run_group=run,
        _timing_authority=admitted_timing,
        _require_authoritative_timing=require_authoritative_timing,
    )


def require_receipt_bound_provider_track_motion_source_handle(
    value: object,
) -> ReceiptBoundProviderTrackMotionSourceHandle:
    if type(value) is not ReceiptBoundProviderTrackMotionSourceHandle:
        raise ProviderTrackMotionSourceHandleError(
            "A loader-minted receipt-bound provider-motion handle is required."
        )
    value.assert_current()
    return value


def _load_once(
    archive: Path,
    run_path: str,
    run_name: str,
    *,
    use_consolidated: bool,
    expected_manifest_sha256: str | None,
    require_authoritative_timing: bool,
) -> ProviderTrackMotionSourceHandle:
    try:
        root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
        parent = root[writer.PROVIDER_TRACK_MOTION_PARENT_PATH]
        run = root[run_path]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ProviderTrackMotionSourceHandleError(
            f"Unable to open exact provider-motion run {run_path!r}: {exc}"
        ) from exc
    if not isinstance(run, zarr.Group):
        raise ProviderTrackMotionSourceHandleError(
            f"Provider-motion run {run_path!r} is not a group."
        )
    selector_attrs = set(writer._SELECTOR_ATTRS).intersection(parent.attrs)
    if selector_attrs:
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion namespace contains forbidden selector attributes: "
            f"{sorted(selector_attrs)!r}."
        )
    attrs = run.attrs
    if (
        attrs.get("schema_id") != writer.PROVIDER_TRACK_MOTION_SCHEMA_ID
        or attrs.get("schema_version") != writer.PROVIDER_TRACK_MOTION_SCHEMA_VERSION
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion run schema identity is invalid."
        )
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion run does not satisfy the complete selector-ineligible lifecycle."
        )
    raw_manifest = attrs.get(writer.PROVIDER_TRACK_MOTION_MANIFEST_ATTR)
    if not isinstance(raw_manifest, Mapping):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion manifest is missing."
        )
    try:
        writer.validate_provider_track_motion_run(
            archive,
            run_path,
            use_consolidated=use_consolidated,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        payload, receipt = writer._validate_manifest(
            raw_manifest,
            expected_run_name=run_name,
            expected_status=RUN_STATUS_COMPLETE,
        )
        manifest_sha256 = writer.provider_track_motion_manifest_digest(raw_manifest)
    except (writer.ProviderTrackMotionError, KeyError, TypeError, ValueError) as exc:
        raise ProviderTrackMotionSourceHandleError(str(exc)) from exc
    if attrs.get(writer.PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR) != manifest_sha256:
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion manifest digest attribute is stale."
        )
    if (
        attrs.get(writer.PROVIDER_TRACK_MOTION_STORAGE_PLAN_ATTR)
        != payload["physical_storage_plan"]
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion storage-plan attribute differs from its manifest."
        )
    publication = payload["publication"]
    if (
        attrs.get(writer.PROVIDER_TRACK_MOTION_PUBLICATION_ATTEMPT_ATTR)
        != publication["publication_attempt_uuid"]
    ):
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion publication attempt differs from its manifest."
        )
    source_record, source_sha256 = _binding(payload, "source_authority")
    tracked_record, tracked_sha256 = _binding(payload, "tracked_input")
    computation_record, computation_sha256 = _binding(payload, "computation")
    physical_binding = _require_mapping(
        payload["physical_authority"], name="provider physical_authority"
    )
    physical_status = physical_binding["status"]
    if physical_status == "bound":
        # The physical binding is intentionally shaped differently from the
        # ordinary record bindings: status, record, and sha256 are siblings.
        physical_value = _require_mapping(
            physical_binding.get("record"), name="provider physical authority record"
        )
        physical_sha256 = _require_digest(
            physical_binding.get("sha256"), name="provider physical authority digest"
        )
        if canonical_json_sha256(physical_value) != physical_sha256:
            raise ProviderTrackMotionSourceHandleError(
                "Provider physical authority record digest is stale."
            )
        physical_record = _freeze(physical_value)
    elif physical_binding == {
        "status": "omitted_explicit_pixel_only_canary",
        "record": None,
        "sha256": None,
    }:
        physical_record, physical_sha256 = None, None
    else:
        raise ProviderTrackMotionSourceHandleError(
            "Provider physical authority binding is invalid."
        )
    (
        temporal_record,
        temporal_sha256,
        timing_status,
        timing_authoritative,
        timing_authority,
    ) = _temporal_authority(
        computation_record,
        archive=archive,
        use_consolidated=use_consolidated,
    )
    if require_authoritative_timing and not timing_authoritative:
        raise ProviderTrackMotionSourceHandleError(
            "Provider-motion run has no authoritative temporal authority; "
            f"status is {timing_status!r}, and caller FPS is compatibility-only."
        )
    arrays = {
        entry.declaration.path: _read_array(run, entry.declaration.path)
        for entry in receipt.entries
    }
    try:
        writer._validate_arrays(arrays)
    except writer.ProviderTrackMotionError as exc:
        raise ProviderTrackMotionSourceHandleError(str(exc)) from exc
    _validate_lineage_and_offsets(arrays)
    _validate_independent_validity(arrays)
    if timing_authority is not None:
        try:
            timing_authority.validate_source_frame_indices(
                arrays["source_acquisition_frame_index"],
                name="provider-motion source acquisition frames",
            )
        except ProviderRecordingTimingAuthorityError as exc:
            raise ProviderTrackMotionSourceHandleError(str(exc)) from exc
    verification = _verification_digest(
        run_path=run_path,
        manifest_sha256=manifest_sha256,
        arrays=arrays,
        timing_status=timing_status,
    )
    return ProviderTrackMotionSourceHandle(
        analysis_zarr_path=archive,
        run_path=run_path,
        run_name=run_name,
        provider_manifest=raw_manifest,
        provider_manifest_sha256=manifest_sha256,
        selector_eligible=False,
        source_authority_record=source_record,
        source_authority_sha256=source_sha256,
        tracked_input_record=tracked_record,
        tracked_input_sha256=tracked_sha256,
        physical_authority_record=physical_record,
        physical_authority_sha256=physical_sha256,
        physical_authority_status=str(physical_status),
        computation_record=computation_record,
        computation_sha256=computation_sha256,
        temporal_authority_record=temporal_record,
        temporal_authority_sha256=temporal_sha256,
        temporal_authority_status=timing_status,
        timing_is_authoritative=timing_authoritative,
        arrays=arrays,
        row_count=_validate_exact_lengths(arrays)[0],
        track_count=_validate_exact_lengths(arrays)[1],
        per_second_count=_validate_exact_lengths(arrays)[2],
        verification_digest=verification,
        _use_consolidated=use_consolidated,
        _require_authoritative_timing=require_authoritative_timing,
        _verification_seal=_HANDLE_SEAL,
    )


def load_provider_track_motion_source_handle(
    analysis_zarr: str | Path,
    run_path: str,
    *,
    use_consolidated: bool = True,
    expected_manifest_sha256: str | None = None,
    require_authoritative_timing: bool = False,
) -> ProviderTrackMotionSourceHandle:
    """Load one exact complete provider-motion run without selector lookup."""

    exact_path, run_name = _canonical_run_path(run_path)
    if type(use_consolidated) is not bool:
        raise ProviderTrackMotionSourceHandleError(
            "use_consolidated must be the exact boolean metadata-read choice."
        )
    if type(require_authoritative_timing) is not bool:
        raise ProviderTrackMotionSourceHandleError(
            "require_authoritative_timing must be the exact boolean."
        )
    if expected_manifest_sha256 is not None:
        _require_digest(
            expected_manifest_sha256, name="expected provider manifest digest"
        )
    archive = Path(analysis_zarr).expanduser().resolve()
    snapshot = _load_once(
        archive,
        exact_path,
        run_name,
        use_consolidated=use_consolidated,
        expected_manifest_sha256=expected_manifest_sha256,
        require_authoritative_timing=require_authoritative_timing,
    )
    if use_consolidated:
        direct = _load_once(
            archive,
            exact_path,
            run_name,
            use_consolidated=False,
            expected_manifest_sha256=snapshot.provider_manifest_sha256,
            require_authoritative_timing=require_authoritative_timing,
        )
        if direct.verification_digest != snapshot.verification_digest:
            raise ProviderTrackMotionSourceHandleError(
                "Provider-motion direct metadata differs from its published consolidated generation."
            )
    return snapshot


def require_provider_track_motion_source_handle(
    value: object,
) -> ProviderTrackMotionSourceHandle:
    """Require a loader-minted, currently verified provider-motion handle."""

    if type(value) is not ProviderTrackMotionSourceHandle:
        raise ProviderTrackMotionSourceHandleError(
            "A verified ProviderTrackMotionSourceHandle is required."
        )
    value.assert_current()
    return value


__all__ = [
    "PROVIDER_TRACK_MOTION_ATOMIC_COMPATIBILITY_RECEIPT_PROFILE",
    "PROVIDER_TRACK_MOTION_NATIVE_RECEIPT_PROFILE",
    "PROVIDER_TRACK_MOTION_RECEIPT_BOUND_HANDLE_SCHEMA_ID",
    "PROVIDER_TRACK_MOTION_RECEIPT_BOUND_HANDLE_SCHEMA_VERSION",
    "PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_ID",
    "PROVIDER_TRACK_MOTION_SOURCE_HANDLE_SCHEMA_VERSION",
    "ProviderTrackMotionSourceHandle",
    "ProviderTrackMotionSourceHandleError",
    "ReceiptBoundProviderTrackMotionSourceHandle",
    "load_receipt_bound_provider_track_motion_source_handle",
    "load_provider_track_motion_source_handle",
    "require_receipt_bound_provider_track_motion_source_handle",
    "require_provider_track_motion_source_handle",
]
