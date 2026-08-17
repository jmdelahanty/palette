"""Exact immutable manifests for modern Palette tracking runs."""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import RUN_STATUS_COMPLETE


TRACKING_RUN_MANIFEST_SCHEMA_ID = "palette.tracking_run_manifest"
TRACKING_RUN_MANIFEST_SCHEMA_VERSION = 1
TRACKING_RUN_MANIFEST_ATTR = "tracking_run_manifest"
TRACKING_RUN_MANIFEST_DIGEST_ATTR = "tracking_run_manifest_sha256"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_MANDATORY_ARRAYS = frozenset(
    {
        "track_ids",
        "arena_ids",
        "frame_indices",
        "source_row_indices",
        "track_ids_present",
        "track_arena_ids",
    }
)
_OPTIONAL_ARRAYS = frozenset(
    {
        "instance_key",
        "source_refined_row_ids",
        "source_detect_row_index",
        "tracking_confidence",
        "tracking_status",
        "association_cost",
    }
)
TRACKING_RUN_ARRAYS = _MANDATORY_ARRAYS | _OPTIONAL_ARRAYS


class TrackingRunManifestError(ValueError):
    """Raised when a tracking-run manifest is incomplete or stale."""


def _run_name(value: object) -> str:
    if type(value) is not str or _RUN_NAME_RE.fullmatch(value) is None:
        raise TrackingRunManifestError("Tracking run_name is not canonical.")
    return value


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise TrackingRunManifestError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _source_record(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "source_arena_assignment_run": attrs.get("source_arena_assignment_run"),
        "source_rowset_path": attrs.get("source_rowset_path"),
        "source_rowset_fingerprint_schema_id": attrs.get(
            "source_rowset_fingerprint_schema_id"
        ),
        "source_rowset_fingerprint_schema_version": attrs.get(
            "source_rowset_fingerprint_schema_version"
        ),
        "source_rowset_fingerprint_canonicalization": attrs.get(
            "source_rowset_fingerprint_canonicalization"
        ),
        "source_rowset_fingerprint_status": attrs.get(
            "source_rowset_fingerprint_status"
        ),
        "source_rowset_row_count": attrs.get("source_rowset_row_count"),
        "source_rowset_edit_revision": attrs.get("source_rowset_edit_revision"),
        "source_rowset_instance_key_digest": attrs.get(
            "source_rowset_instance_key_digest"
        ),
        "source_rowset_fingerprint": attrs.get("source_rowset_fingerprint"),
    }


def tracking_array_records(run_group: Any) -> list[dict[str, Any]]:
    """Return sorted decoded-content records for every tracking array."""

    names = sorted(str(name) for name in run_group.array_keys())
    unknown = set(names) - TRACKING_RUN_ARRAYS
    missing = _MANDATORY_ARRAYS - set(names)
    if unknown:
        raise TrackingRunManifestError(
            f"Tracking run contains undeclared arrays: {sorted(unknown)!r}."
        )
    if missing:
        raise TrackingRunManifestError(
            f"Tracking run omits mandatory arrays: {sorted(missing)!r}."
        )
    records: list[dict[str, Any]] = []
    for name in names:
        values = np.asarray(run_group[name][:])
        records.append(
            {
                "path": name,
                "dtype": values.dtype.str,
                "shape": [int(size) for size in values.shape],
                "sha256": sha256_array(values),
            }
        )
    return records


def build_tracking_run_manifest(
    run_group: Any,
    *,
    run_name: str,
    status: str = RUN_STATUS_COMPLETE,
) -> dict[str, Any]:
    """Build a strict manifest from one fully written tracking run."""

    name = _run_name(run_name)
    if status != RUN_STATUS_COMPLETE:
        raise TrackingRunManifestError(
            "Tracking manifests are written only for complete immutable runs."
        )
    arrays = tracking_array_records(run_group)
    payload = {
        "namespace": "tracking_runs",
        "row_axis": "observation_instance",
        "run_name": name,
        "run_path": f"tracking_runs/{name}",
        "status": status,
        "stage_selector_eligible": True,
        "tracking_method": run_group.attrs.get("tracking_method"),
        "tracking_identity_mode": run_group.attrs.get("tracking_identity_mode"),
        "unassigned_track_id": run_group.attrs.get("unassigned_track_id"),
        "tracking_configuration": {
            "tracker_parameters": run_group.attrs.get("tracker_parameters"),
            "provenance": run_group.attrs.get("provenance"),
        },
        "source": _source_record(run_group.attrs),
        "arrays": arrays,
        "decoded_content_sha256": canonical_json_sha256(arrays),
    }
    return {
        "schema_id": TRACKING_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": TRACKING_RUN_MANIFEST_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def tracking_run_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Validate the envelope and return its canonical payload digest."""

    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise TrackingRunManifestError(
            "Tracking run manifest envelope is not exact."
        )
    if manifest["schema_id"] != TRACKING_RUN_MANIFEST_SCHEMA_ID:
        raise TrackingRunManifestError("Tracking run manifest schema ID differs.")
    if manifest["schema_version"] != TRACKING_RUN_MANIFEST_SCHEMA_VERSION:
        raise TrackingRunManifestError(
            "Tracking run manifest schema version differs."
        )
    payload = manifest["payload"]
    if not isinstance(payload, Mapping):
        raise TrackingRunManifestError("Tracking run manifest payload is absent.")
    digest = _sha256(manifest["payload_digest"], name="payload_digest")
    if canonical_json_sha256(payload) != digest:
        raise TrackingRunManifestError(
            "Tracking run manifest payload digest is stale."
        )
    return digest


def validate_tracking_run_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_run_name: str | None = None,
    expected_status: str = RUN_STATUS_COMPLETE,
) -> dict[str, Any]:
    """Validate exact manifest structure independently of the Zarr payload."""

    digest = tracking_run_manifest_digest(manifest)
    payload = manifest["payload"]
    if set(payload) != {
        "namespace",
        "row_axis",
        "run_name",
        "run_path",
        "status",
        "stage_selector_eligible",
        "tracking_method",
        "tracking_identity_mode",
        "unassigned_track_id",
        "tracking_configuration",
        "source",
        "arrays",
        "decoded_content_sha256",
    }:
        raise TrackingRunManifestError(
            "Tracking run manifest payload field set is not exact."
        )
    name = _run_name(payload["run_name"])
    if expected_run_name is not None and name != _run_name(expected_run_name):
        raise TrackingRunManifestError(
            "Tracking run manifest names a different run."
        )
    if payload["namespace"] != "tracking_runs" or payload["row_axis"] != (
        "observation_instance"
    ):
        raise TrackingRunManifestError(
            "Tracking run namespace or row axis differs."
        )
    if payload["run_path"] != f"tracking_runs/{name}":
        raise TrackingRunManifestError("Tracking run manifest path is stale.")
    if payload["status"] != expected_status:
        raise TrackingRunManifestError("Tracking run status differs.")
    if payload["stage_selector_eligible"] is not True:
        raise TrackingRunManifestError(
            "Modern tracking authority must be explicitly selector eligible."
        )
    if not isinstance(payload["tracking_method"], str) or not payload[
        "tracking_method"
    ]:
        raise TrackingRunManifestError("Tracking method is absent.")
    if payload["tracking_identity_mode"] not in {
        "instance_key",
        "legacy_positional",
    }:
        raise TrackingRunManifestError("Tracking identity mode is invalid.")
    if type(payload["unassigned_track_id"]) is not int:
        raise TrackingRunManifestError("Unassigned track ID is not an integer.")
    configuration = payload["tracking_configuration"]
    if not isinstance(configuration, Mapping) or set(configuration) != {
        "tracker_parameters",
        "provenance",
    }:
        raise TrackingRunManifestError(
            "Tracking configuration binding is not exact."
        )
    if not isinstance(configuration["tracker_parameters"], Mapping):
        raise TrackingRunManifestError("Tracking parameters are absent.")
    if not isinstance(configuration["provenance"], Mapping):
        raise TrackingRunManifestError("Tracking provenance is absent.")
    source = payload["source"]
    expected_source_fields = set(_source_record({}))
    if not isinstance(source, Mapping) or set(source) != expected_source_fields:
        raise TrackingRunManifestError(
            "Tracking source-lineage record is not exact."
        )
    for required in (
        "source_detect_run",
        "source_arena_assignment_run",
        "source_rowset_path",
        "source_rowset_fingerprint_status",
        "source_rowset_row_count",
    ):
        if source[required] in (None, ""):
            raise TrackingRunManifestError(
                f"Tracking source-lineage field {required!r} is absent."
            )
    arrays = payload["arrays"]
    if not isinstance(arrays, list):
        raise TrackingRunManifestError("Tracking array declarations are absent.")
    paths: list[str] = []
    for record in arrays:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "dtype",
            "shape",
            "sha256",
        }:
            raise TrackingRunManifestError(
                "Tracking array declaration is not exact."
            )
        path = record["path"]
        if path not in TRACKING_RUN_ARRAYS:
            raise TrackingRunManifestError(
                f"Tracking array path is unknown: {path!r}."
            )
        if not isinstance(record["dtype"], str) or not isinstance(
            record["shape"], list
        ):
            raise TrackingRunManifestError(
                f"Tracking array declaration {path!r} is invalid."
            )
        _sha256(record["sha256"], name=f"array {path!r}")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise TrackingRunManifestError(
            "Tracking array declarations must be sorted and unique."
        )
    if not _MANDATORY_ARRAYS.issubset(paths):
        raise TrackingRunManifestError("Tracking manifest omits mandatory arrays.")
    _sha256(payload["decoded_content_sha256"], name="decoded_content_sha256")
    if canonical_json_sha256(arrays) != payload["decoded_content_sha256"]:
        raise TrackingRunManifestError(
            "Tracking decoded-content declaration digest is stale."
        )
    return {"valid": True, "manifest_sha256": digest, "payload": payload}


__all__ = [
    "TRACKING_RUN_ARRAYS",
    "TRACKING_RUN_MANIFEST_ATTR",
    "TRACKING_RUN_MANIFEST_DIGEST_ATTR",
    "TRACKING_RUN_MANIFEST_SCHEMA_ID",
    "TRACKING_RUN_MANIFEST_SCHEMA_VERSION",
    "TrackingRunManifestError",
    "build_tracking_run_manifest",
    "tracking_array_records",
    "tracking_run_manifest_digest",
    "validate_tracking_run_manifest",
]
