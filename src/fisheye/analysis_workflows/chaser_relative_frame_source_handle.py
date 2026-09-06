"""Strict read boundary for one published chaser-relative-frame candidate.

The handle in this module is deliberately a consumer-side object.  It binds
one caller-supplied bare run name, validates the immutable publication using
the archive-root consolidated metadata generation, copies only typed numeric
arrays, and then seals those copies read-only.  It never resolves a selector,
chooses a fallback, or writes to the archive.

The on-disk layout is::

    analysis/chaser_relative_frame_runs/<run>/base/<array>
    analysis/chaser_relative_frame_runs/<run>/body/<array>   # optional

Rows are frame-major/chaser-minor.  ``reshape_frame_chaser`` is only a
zero-copy view of that declared row order; it does not infer or change the
scientific meaning of an array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    PROXY_POLICY_ID as INPUT_PROVENANCE_PROXY_POLICY_ID,
)
from fisheye.analysis_workflows.chaser_relative_frame_storage import (
    validate_chaser_input_provenance_projection_binding,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.run_provenance import validate_run_provenance
from fisheye.shared.zarr.chaser_relative_frame_schema import (
    CHASER_RELATIVE_FRAME_LAYOUT,
    CHASER_RELATIVE_FRAME_REASON_CODES,
    CHASER_RELATIVE_FRAME_SCHEMA_ID,
    CHASER_RELATIVE_FRAME_SCHEMA_V1,
    CHASER_RELATIVE_FRAME_SCHEMA_VERSION,
    ChaserRelativeFrameDimensions,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    MetadataEquivalenceReceipt,
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.shared.zarr_io import open_zarr_root


CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_ID = (
    "palette.analysis.chaser_relative_frame.source_handle"
)
CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_VERSION = 1
CHASER_RELATIVE_FRAME_RUN_PARENT_PATH = "analysis/chaser_relative_frame_runs"
CHASER_RELATIVE_FRAME_RUNS_PREFIX = f"{CHASER_RELATIVE_FRAME_RUN_PARENT_PATH}/"
CHASER_RELATIVE_FRAME_RUN_PREFIX = CHASER_RELATIVE_FRAME_RUNS_PREFIX

MANIFEST_ATTR = "chaser_relative_frame_manifest"
MANIFEST_DIGEST_ATTR = "chaser_relative_frame_manifest_sha256"
RUN_PATH_ATTR = "run_path"

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SELECTOR_ALIASES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_provider",
        "latest_any",
        "latest_materialized",
        "latest_composite",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
        "active_run",
        "active",
        "current_run",
        "current",
        "default_run",
        "default",
        "selected_run",
        "selected",
        "authoritative",
        "publication_generation",
        "publication_policy",
    }
)
_BASE_REQUIRED = frozenset(
    binding.path for binding in CHASER_RELATIVE_FRAME_SCHEMA_V1.bindings if binding.required
)
_BASE_ALL = frozenset(CHASER_RELATIVE_FRAME_SCHEMA_V1.binding_paths)
_BODY_ALL = frozenset(CHASER_RELATIVE_FRAME_SCHEMA_V1.body_extension.binding_paths)
_AUTHORITY_FIELDS = frozenset(
    {
        "recording_id",
        "source_authority_id",
        "source_digest",
        "provider_id",
        "provider_digest",
        "coordinate_authority_id",
        "scale_authority_id",
        "timing_authority_id",
        "row_axis_authority_id",
        "row_axis_authority_digest",
    }
)
_HANDLE_SEAL = object()
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ChaserRelativeFrameSourceHandleError(ValueError):
    """Raised when an exact published relative-frame candidate is invalid."""


class ChaserRelativeFrameBodyUnavailableError(
    ChaserRelativeFrameSourceHandleError
):
    """Raised when body-frame arrays are requested from a position-only run."""


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _strict_json_object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ChaserRelativeFrameSourceHandleError(
            f"{field} must be one JSON object."
        )
    try:
        import json

        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"{field} is not strict JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ChaserRelativeFrameSourceHandleError(f"{field} is not an object.")
    return decoded


def _readonly_snapshot(node: Any, *, path: str) -> np.ndarray:
    try:
        dtype = np.dtype(node.dtype)
        shape = tuple(int(value) for value in node.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Array {path!r} has invalid typed metadata: {exc}"
        ) from exc
    if dtype.hasobject or dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise ChaserRelativeFrameSourceHandleError(
            f"Array {path!r} is not a typed numeric array: {dtype.str!r}."
        )
    try:
        value = np.array(node[...], dtype=dtype, copy=True)
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Unable to read typed array {path!r}: {exc}"
        ) from exc
    if value.shape != shape:
        raise ChaserRelativeFrameSourceHandleError(
            f"Array {path!r} changed shape while being read."
        )
    value.setflags(write=False)
    return value


def _require_exact_bare_run_name(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ChaserRelativeFrameSourceHandleError(
            "run_name must be one non-empty bare run-name string."
        )
    if (
        value in {".", ".."}
        or value in _SELECTOR_ALIASES
        or "/" in value
        or "\\" in value
        or not _RUN_NAME_RE.fullmatch(value)
    ):
        raise ChaserRelativeFrameSourceHandleError(
            "run_name must be one exact concrete run name, not a selector or path."
        )
    return value


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ChaserRelativeFrameSourceHandleError(
            f"{field} must be a string-keyed mapping."
        )
    return value


def _require_nonempty_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ChaserRelativeFrameSourceHandleError(
            f"{field} must be one non-empty exact string."
        )
    return value


def _require_digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ChaserRelativeFrameSourceHandleError(
            f"{field} must be one lowercase SHA-256 digest."
        )
    return value


def _require_dimensions(manifest: Mapping[str, Any]) -> ChaserRelativeFrameDimensions:
    dimensions = _require_mapping(manifest.get("dimensions"), field="manifest.dimensions")
    required = {"n_frames", "n_chasers", "n_rows"}
    if set(dimensions) != required:
        raise ChaserRelativeFrameSourceHandleError(
            "manifest.dimensions must declare exactly n_frames, n_chasers, and n_rows."
        )
    values = {name: dimensions[name] for name in required}
    if any(type(value) is not int for value in values.values()):
        raise ChaserRelativeFrameSourceHandleError(
            "manifest dimensions must be exact JSON integers."
        )
    n_frames = int(values["n_frames"])
    n_chasers = int(values["n_chasers"])
    n_rows = int(values["n_rows"])
    if n_frames < 0 or n_chasers <= 0 or n_rows != n_frames * n_chasers:
        raise ChaserRelativeFrameSourceHandleError(
            "manifest frame/chaser dimensions are inconsistent."
        )
    return ChaserRelativeFrameDimensions(n_rows=n_rows)


def _validate_manifest_digest(
    run: Any,
    *,
    expected_recording_id: str | None,
    exact_run_path: str,
) -> tuple[dict[str, Any], str, ChaserRelativeFrameDimensions]:
    raw = run.attrs.get(MANIFEST_ATTR)
    manifest = _strict_json_object(raw, field=MANIFEST_ATTR)
    stored_digest = run.attrs.get(MANIFEST_DIGEST_ATTR)
    if type(stored_digest) is not str:
        raise ChaserRelativeFrameSourceHandleError(
            f"{MANIFEST_DIGEST_ATTR} is missing or not a string."
        )
    observed_manifest_digest = canonical_json_sha256(manifest)
    if stored_digest != observed_manifest_digest:
        raise ChaserRelativeFrameSourceHandleError(
            "Published manifest digest is stale or does not match its manifest."
        )
    payload = dict(manifest)
    payload_digest = payload.pop("payload_digest", None)
    if type(payload_digest) is not str or payload_digest != canonical_json_sha256(payload):
        raise ChaserRelativeFrameSourceHandleError(
            "Published manifest payload_digest is missing or stale."
        )
    if manifest.get("schema_id") != "palette.analysis.chaser_relative_frame.prepared_candidate":
        raise ChaserRelativeFrameSourceHandleError(
            "Published manifest has the wrong prepared-candidate schema_id."
        )
    if manifest.get("schema_version") != 1:
        raise ChaserRelativeFrameSourceHandleError(
            "Published manifest has the wrong prepared-candidate schema_version."
        )
    recording_id = _require_nonempty_text(
        manifest.get("recording_id"), field="manifest.recording_id"
    )
    if expected_recording_id is not None and recording_id != expected_recording_id:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run recording_id does not match the requested recording."
        )
    if run.attrs.get(RUN_PATH_ATTR) != exact_run_path:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run_path does not match the exact requested run."
        )
    if manifest.get("selector_eligible") is not False or manifest.get("selection") != "none":
        raise ChaserRelativeFrameSourceHandleError(
            "Published manifest is not selector-ineligible with selection=none."
        )
    if manifest.get("flatten_policy_id") != "acquisition_frame_major_chaser_axis_minor_v1":
        raise ChaserRelativeFrameSourceHandleError(
            "Published manifest does not declare the exact frame-major/chaser-minor layout."
        )
    expected_reasons = {
        str(code): reason for code, reason in CHASER_RELATIVE_FRAME_REASON_CODES.items()
    }
    if manifest.get("reason_codes") != expected_reasons:
        raise ChaserRelativeFrameSourceHandleError(
            "Published reason-code registry is missing, extra, or invalid."
        )
    dimensions = _require_dimensions(manifest)
    binding = _require_mapping(manifest.get("schema_binding"), field="manifest.schema_binding")
    body_extension_present = binding.get("body_extension_present")
    if type(body_extension_present) is not bool:
        raise ChaserRelativeFrameSourceHandleError(
            "manifest.schema_binding.body_extension_present must be an exact boolean."
        )
    expected_binding = {
        "schema_id": CHASER_RELATIVE_FRAME_SCHEMA_ID,
        "schema_version": CHASER_RELATIVE_FRAME_SCHEMA_VERSION,
        "layout": CHASER_RELATIVE_FRAME_LAYOUT,
        "body_extension_present": body_extension_present,
    }
    if binding != expected_binding:
        raise ChaserRelativeFrameSourceHandleError(
            "Published schema_binding is missing, stale, or contradictory."
        )
    if (
        run.attrs.get("schema_id") != CHASER_RELATIVE_FRAME_SCHEMA_ID
        or run.attrs.get("schema_version") != CHASER_RELATIVE_FRAME_SCHEMA_VERSION
        or run.attrs.get("layout") != CHASER_RELATIVE_FRAME_LAYOUT
    ):
        raise ChaserRelativeFrameSourceHandleError(
            "Published run schema_id, schema_version, or layout is invalid."
        )
    return manifest, observed_manifest_digest, dimensions


def _validate_completion_and_selection(run: Any, *, exact_run_path: str) -> None:
    if run.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run does not declare the completion contract."
        )
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run is not complete."
        )
    if run.attrs.get("stage_selector_eligible") is not False:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run stage_selector_eligible must be false."
        )
    if run.attrs.get("selector_eligible") is not False:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run selector_eligible must be false."
        )
    if run.attrs.get("selection") != "none":
        raise ChaserRelativeFrameSourceHandleError(
            "Published run selection must be none."
        )
    run_name = run.attrs.get("palette_run_name")
    if run_name is not None and run_name != exact_run_path.rsplit("/", 1)[1]:
        raise ChaserRelativeFrameSourceHandleError(
            "Published completion run name does not match the exact run path."
        )


def _completion_authority(
    root: Any,
    run: Any,
    *,
    exact_run_path: str,
    require_provenance_epoch: bool,
) -> dict[str, Any]:
    """Read the run completion plus its parent store epoch without payload IO."""

    parent_path = exact_run_path.rsplit("/", 1)[0]
    try:
        parent = root[parent_path]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Published run parent is unavailable for {exact_run_path!r}: {exc}"
        ) from exc
    contract = run.attrs.get(RUN_COMPLETION_CONTRACT_ATTR)
    status = run.attrs.get(RUN_COMPLETION_STATUS_ATTR)
    epoch = parent.attrs.get(COMPLETION_EPOCH_ATTR)
    if type(epoch) is not int or isinstance(epoch, bool):
        if require_provenance_epoch:
            raise ChaserRelativeFrameSourceHandleError(
                "Published run parent has no exact provenance completion epoch."
            )
        epoch = None
    if require_provenance_epoch and epoch < COMPLETION_EPOCH_REQUIRE_PROVENANCE:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run parent completion epoch is not provenance-bearing."
        )
    return {
        "contract": contract,
        "status": status,
        "epoch": epoch,
    }


def _validate_provenance(
    run: Any,
    *,
    recording_id: str,
    expected_payload_digest: str,
    run_name: str,
) -> Mapping[str, Any]:
    raw = run.attrs.get("run_provenance")
    result = validate_run_provenance(raw)
    if not result.valid:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run provenance is invalid: " + "; ".join(result.errors)
        )
    provenance = _strict_json_object(raw, field="run_provenance")
    params = _require_mapping(provenance.get("params"), field="run_provenance.params")
    if params.get("run_name") != run_name or params.get("schema_id") != CHASER_RELATIVE_FRAME_SCHEMA_ID:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run provenance does not bind the exact run and schema."
        )
    input_run_ids = _require_mapping(
        provenance.get("input_run_ids"), field="run_provenance.input_run_ids"
    )
    if input_run_ids.get("recording_id") != recording_id:
        raise ChaserRelativeFrameSourceHandleError(
            "Published provenance recording_id does not match the run."
        )
    prepared_digest = input_run_ids.get("prepared_chaser_relative_frame")
    if prepared_digest != expected_payload_digest:
        # The writer binds the prepared payload digest in input_run_ids.  This
        # catches a valid-looking provenance record copied from another run.
        raise ChaserRelativeFrameSourceHandleError(
            "Published provenance does not bind the run manifest payload digest."
        )
    return provenance


def _validate_context_and_authorities(
    manifest: Mapping[str, Any], *, recording_id: str, body_present: bool
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    context = _require_mapping(manifest.get("context"), field="manifest.context")
    required_context = {
        "fish_identity",
        "subject_identity",
        "temporal_selection",
        "chaser_occurrence",
        "acquisition_projection",
        "acquisition_projection_publication",
        "analysis_profile",
        "arena_geometry",
        "arena_to_source_camera_transform",
    }
    optional_context = {
        "controller_state",
        "body_frame_projection",
        "core_authority",
    }
    if not required_context.issubset(context) or not set(context).issubset(
        required_context | optional_context
    ):
        raise ChaserRelativeFrameSourceHandleError(
            "Published context has missing or extra authority records."
        )
    if "body_frame_projection" in context and not body_present:
        raise ChaserRelativeFrameSourceHandleError(
            "Position-only publication has an unexpected body-frame projection."
        )
    fish_identity = _require_nonempty_text(
        context.get("fish_identity"), field="manifest.context.fish_identity"
    )
    records: dict[str, Mapping[str, Any]] = {}
    for name in sorted(
        set(context)
        - {
            "fish_identity",
            "arena_geometry",
            "arena_to_source_camera_transform",
            "acquisition_projection_publication",
        }
    ):
        envelope = _require_mapping(context.get(name), field=f"manifest.context.{name}")
        if set(envelope) != {"record", "sha256"}:
            raise ChaserRelativeFrameSourceHandleError(
                f"manifest.context.{name} is not an exact record-plus-sha256 envelope."
            )
        record = _strict_json_object(envelope["record"], field=f"manifest.context.{name}.record")
        if envelope["sha256"] != canonical_json_sha256(record):
            raise ChaserRelativeFrameSourceHandleError(
                f"manifest.context.{name} digest does not match its record."
            )
        if name != "analysis_profile" and record.get("recording_id") != recording_id:
            raise ChaserRelativeFrameSourceHandleError(
                f"manifest.context.{name} recording_id does not match the run."
            )
        records[name] = record
    publication = context["acquisition_projection_publication"]
    projection = records["acquisition_projection"]
    if projection.get("policy_id") == INPUT_PROVENANCE_PROXY_POLICY_ID:
        publication_envelope = _require_mapping(
            publication,
            field="manifest.context.acquisition_projection_publication",
        )
        if set(publication_envelope) != {"record", "sha256"}:
            raise ChaserRelativeFrameSourceHandleError(
                "manifest.context.acquisition_projection_publication is not "
                "an exact record-plus-sha256 envelope."
            )
        publication_record = _strict_json_object(
            publication_envelope["record"],
            field="manifest.context.acquisition_projection_publication.record",
        )
        if publication_envelope["sha256"] != canonical_json_sha256(
            publication_record
        ):
            raise ChaserRelativeFrameSourceHandleError(
                "manifest.context.acquisition_projection_publication digest "
                "does not match its record."
            )
        try:
            validate_chaser_input_provenance_projection_binding(
                projection=projection,
                publication=publication_record,
            )
        except (TypeError, ValueError) as exc:
            raise ChaserRelativeFrameSourceHandleError(
                f"Published input-provenance proxy binding is invalid: {exc}"
            ) from exc
    elif publication is not None:
        raise ChaserRelativeFrameSourceHandleError(
            "Published context has an unexpected acquisition projection "
            "publication binding."
        )
    subject = _require_mapping(context["subject_identity"]["record"], field="subject_identity")
    if subject.get("subject_id") != fish_identity:
        raise ChaserRelativeFrameSourceHandleError(
            "manifest.context subject identity does not match fish_identity."
        )
    geometry = context["arena_geometry"]
    transform = context["arena_to_source_camera_transform"]
    if (geometry is None) != (transform is None):
        raise ChaserRelativeFrameSourceHandleError(
            "Published context has a partial arena geometry/transform binding."
        )
    for name, envelope in (
        ("arena_geometry", geometry),
        ("arena_to_source_camera_transform", transform),
    ):
        if envelope is None:
            continue
        envelope_map = _require_mapping(envelope, field=f"manifest.context.{name}")
        if set(envelope_map) != {"record", "sha256"}:
            raise ChaserRelativeFrameSourceHandleError(
                f"manifest.context.{name} is not an exact record-plus-sha256 envelope."
            )
        record = _strict_json_object(envelope_map["record"], field=f"manifest.context.{name}.record")
        if envelope_map["sha256"] != canonical_json_sha256(record):
            raise ChaserRelativeFrameSourceHandleError(
                f"manifest.context.{name} digest does not match its record."
            )
        if record.get("recording_id") != recording_id:
            raise ChaserRelativeFrameSourceHandleError(
                f"manifest.context.{name} recording_id does not match the run."
            )

    authorities = _require_mapping(
        manifest.get("source_authorities"), field="manifest.source_authorities"
    )
    expected_authority_names = {"fish_position", "chaser_position", "body_frame"}
    if set(authorities) != expected_authority_names:
        raise ChaserRelativeFrameSourceHandleError(
            "Published source authorities have missing or extra records."
        )
    for name, value in authorities.items():
        if value is None:
            if name != "body_frame" or body_present:
                raise ChaserRelativeFrameSourceHandleError(
                    f"Published source authority {name!r} is unexpectedly absent."
                )
            continue
        record = _strict_json_object(value, field=f"source_authorities.{name}")
        if set(record) != _AUTHORITY_FIELDS:
            raise ChaserRelativeFrameSourceHandleError(
                f"Published source authority {name!r} has missing or extra fields."
            )
        for field_name, field_value in record.items():
            _require_nonempty_text(field_value, field=f"source_authorities.{name}.{field_name}")
        if record["recording_id"] != recording_id:
            raise ChaserRelativeFrameSourceHandleError(
                f"Published source authority {name!r} recording_id does not match the run."
            )
    return context, authorities


def _validate_identity_registries(
    manifest: Mapping[str, Any], *, n_chasers: int, base: Mapping[str, np.ndarray]
) -> Mapping[str, Any]:
    registries = _require_mapping(
        manifest.get("identity_registries"), field="manifest.identity_registries"
    )
    if set(registries) != {"fish", "chaser", "behavior_role", "active_state"}:
        raise ChaserRelativeFrameSourceHandleError(
            "Published identity registries have missing or extra registries."
        )
    fish = _require_mapping(registries["fish"], field="identity_registries.fish")
    chaser = _require_mapping(registries["chaser"], field="identity_registries.chaser")
    roles = _require_mapping(registries["behavior_role"], field="identity_registries.behavior_role")
    active = _require_mapping(registries["active_state"], field="identity_registries.active_state")
    if set(fish) != {"1"} or not _require_nonempty_text(fish["1"], field="identity_registries.fish.1"):
        raise ChaserRelativeFrameSourceHandleError("Fish identity registry is invalid.")
    expected_chaser_keys = {str(index) for index in range(1, n_chasers + 1)}
    if set(chaser) != expected_chaser_keys:
        raise ChaserRelativeFrameSourceHandleError(
            "Chaser identity registry does not match n_chasers."
        )
    if len({_require_nonempty_text(value, field="identity_registries.chaser") for value in chaser.values()}) != n_chasers:
        raise ChaserRelativeFrameSourceHandleError("Chaser identity registry is not unique.")
    if not roles or any(
        type(key) is not str or not key.isdigit() or int(key) <= 0
        or not _require_nonempty_text(value, field="identity_registries.behavior_role")
        for key, value in roles.items()
    ):
        raise ChaserRelativeFrameSourceHandleError("Behavior-role registry is invalid.")
    if active != {"0": "inactive", "1": "active"}:
        raise ChaserRelativeFrameSourceHandleError("Active-state registry is invalid.")
    codes = np.asarray(base["chaser_identity_code"])
    expected_codes = np.tile(np.arange(1, n_chasers + 1, dtype=codes.dtype), int(codes.size / n_chasers))
    if not np.array_equal(codes, expected_codes):
        raise ChaserRelativeFrameSourceHandleError(
            "Chaser identity codes do not preserve frame-major/chaser-minor row order."
        )
    fish_codes = np.asarray(base["fish_identity_code"])
    if np.any(fish_codes != 1):
        raise ChaserRelativeFrameSourceHandleError("Fish identity codes are not stable.")
    role_codes = np.asarray(base["chaser_behavior_role_code"])
    if np.any(role_codes == 0) or np.any(~np.isin(role_codes, [int(key) for key in roles])):
        raise ChaserRelativeFrameSourceHandleError(
            "Chaser behavior-role codes are not declared by the registry."
        )
    return registries


def _validate_declarations_and_arrays(
    run: Any,
    *,
    manifest: Mapping[str, Any],
    dimensions: ChaserRelativeFrameDimensions,
    verify_content_hashes: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, bool]:
    declarations_raw = manifest.get("array_declarations")
    if not isinstance(declarations_raw, list) or not declarations_raw:
        raise ChaserRelativeFrameSourceHandleError(
            "Published manifest array_declarations must be a non-empty list."
        )
    declarations: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for declaration in declarations_raw:
        item = _require_mapping(declaration, field="array_declaration")
        if set(item) != {"path", "dtype", "shape", "content_sha256"}:
            raise ChaserRelativeFrameSourceHandleError(
                "Array declarations must contain exactly path, dtype, shape, and content_sha256."
            )
        path = _require_nonempty_text(item["path"], field="array_declaration.path")
        if path in paths or path.count("/") != 1 or path.split("/", 1)[0] not in {"base", "body"}:
            raise ChaserRelativeFrameSourceHandleError(
                f"Invalid or duplicate published array path {path!r}."
            )
        paths.append(path)
        declarations.append(item)
    expected_order = sorted(path for path in paths if path.startswith("base/")) + sorted(
        path for path in paths if path.startswith("body/")
    )
    if paths != expected_order:
        raise ChaserRelativeFrameSourceHandleError(
            "Published array declarations are reordered; canonical base/body order is required."
        )
    binding = _require_mapping(manifest["schema_binding"], field="manifest.schema_binding")
    body_present = binding["body_extension_present"] is True
    if set(paths) - {f"base/{name}" for name in _BASE_ALL} - {f"body/{name}" for name in _BODY_ALL}:
        raise ChaserRelativeFrameSourceHandleError("Published declarations contain an unknown schema array.")
    if not _BASE_REQUIRED.issubset({path.split("/", 1)[1] for path in paths if path.startswith("base/")}):
        raise ChaserRelativeFrameSourceHandleError("Published declarations omit required base arrays.")
    base_names = {path.split("/", 1)[1] for path in paths if path.startswith("base/")}
    body_names = {path.split("/", 1)[1] for path in paths if path.startswith("body/")}
    if not body_present and body_names:
        raise ChaserRelativeFrameSourceHandleError("Position-only run contains an unexpected body extension.")
    if body_present and body_names != _BODY_ALL:
        raise ChaserRelativeFrameSourceHandleError("Body extension declarations are incomplete or extra.")
    for left, right in (("trial_id", "trial_valid"), ("trial_valid", "trial_reason_code"), ("active_state_code", "active_state_valid"), ("active_state_valid", "active_state_reason_code")):
        if (left in base_names) != (right in base_names):
            raise ChaserRelativeFrameSourceHandleError(
                f"Optional base arrays {left!r} and {right!r} must be declared together."
            )
    actual_groups = {str(name) for name in run.group_keys()}
    expected_groups = {"base"} | ({"body"} if body_present else set())
    if actual_groups != expected_groups or set(str(name) for name in run.array_keys()):
        raise ChaserRelativeFrameSourceHandleError(
            "Published run has missing or extra base/body groups."
        )
    arrays: dict[str, np.ndarray] = {}
    base_arrays: dict[str, np.ndarray] = {}
    body_arrays: dict[str, np.ndarray] = {}
    for declaration, path in zip(declarations, paths):
        prefix, name = path.split("/", 1)
        group = run[prefix]
        if set(str(value) for value in group.group_keys()):
            raise ChaserRelativeFrameSourceHandleError(
                f"Published {prefix} group contains nested groups."
            )
        actual_names = {str(value) for value in group.array_keys()}
        expected_names = base_names if prefix == "base" else body_names
        if actual_names != expected_names:
            raise ChaserRelativeFrameSourceHandleError(
                f"Published {prefix} arrays do not match declarations."
            )
        value = _readonly_snapshot(group[name], path=path)
        if value.dtype.str != declaration["dtype"]:
            raise ChaserRelativeFrameSourceHandleError(
                f"Published array {path!r} dtype does not match its declaration."
            )
        if (
            not isinstance(declaration["shape"], list)
            or any(type(size) is not int for size in declaration["shape"])
            or list(value.shape) != declaration["shape"]
        ):
            raise ChaserRelativeFrameSourceHandleError(
                f"Published array {path!r} shape does not match its declaration."
            )
        _require_digest(
            declaration["content_sha256"],
            field=f"array_declaration[{path}].content_sha256",
        )
        if verify_content_hashes and array_values_sha256(value) != declaration[
            "content_sha256"
        ]:
            raise ChaserRelativeFrameSourceHandleError(
                f"Published array {path!r} content digest does not match its declaration."
            )
        arrays[path] = value
        (base_arrays if prefix == "base" else body_arrays)[name] = value
    if set(base_arrays) != base_names or (body_present and set(body_arrays) != body_names):
        raise ChaserRelativeFrameSourceHandleError("Published array declarations were not fully materialized.")
    try:
        CHASER_RELATIVE_FRAME_SCHEMA_V1.require(
            base_arrays,
            dimensions=dimensions,
            body_arrays=body_arrays if body_present else None,
        )
    except ValueError as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Published chaser-relative-frame schema validation failed: {exc}"
        ) from exc
    n_frames = int(manifest["dimensions"]["n_frames"])
    n_chasers = int(manifest["dimensions"]["n_chasers"])
    frame_ids = base_arrays["acquisition_frame_id"].reshape(n_frames, n_chasers)
    track_ids = base_arrays["track_sample_id"].reshape(n_frames, n_chasers)
    if n_frames:
        if not np.all(frame_ids == frame_ids[:, :1]) or not np.all(track_ids == track_ids[:, :1]):
            raise ChaserRelativeFrameSourceHandleError(
                "Published row order is not frame-major with frame evidence repeated across chasers."
            )
        if np.unique(frame_ids[:, 0]).size != n_frames or np.unique(track_ids[:, 0]).size != n_frames:
            raise ChaserRelativeFrameSourceHandleError(
                "Published frame or track identity is duplicated across frame rows."
            )
    return base_arrays, (body_arrays if body_present else None), body_present


def _metadata_equivalence(archive: Path, *, run_path: str) -> MetadataEquivalenceReceipt:
    try:
        return validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Archive-root consolidated metadata is missing, stale, or differs "
            f"from direct metadata for {run_path!r}: {exc}"
        ) from exc


@dataclass(frozen=True, init=False, eq=False)
class ChaserRelativeFrameSourceHandle:
    """Immutable, verified snapshot of one exact relative-frame candidate."""

    analysis_zarr_path: Path
    run_path: str
    run_name: str
    recording_id: str
    selector_eligible: bool
    selection: str
    n_frames: int
    n_chasers: int
    n_rows: int
    run_manifest: Mapping[str, Any] = field(repr=False)
    run_provenance: Mapping[str, Any] = field(repr=False)
    identity_registries: Mapping[str, Any] = field(repr=False)
    source_authorities: Mapping[str, Any] = field(repr=False)
    context: Mapping[str, Any] = field(repr=False)
    base_arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    body_arrays: Mapping[str, np.ndarray] | None = field(repr=False, compare=False)
    metadata_equivalence: Mapping[str, Any] = field(repr=False)
    verification_digest: str
    completion_authority: Mapping[str, Any] = field(repr=False)
    verification_mode: str
    receipt_digest: str | None
    _use_consolidated: bool = field(repr=False, compare=False)
    _receipt_document: Mapping[str, Any] | None = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _HANDLE_SEAL:
            raise TypeError("Chaser-relative-frame handles can only be minted by their loader.")
        for name, value in values.items():
            if name in {
                "run_manifest",
                "run_provenance",
                "identity_registries",
                "source_authorities",
                "context",
                "metadata_equivalence",
                "completion_authority",
                "_receipt_document",
            }:
                value = _freeze_json(value)
            elif name == "base_arrays":
                value = MappingProxyType({key: _readonly_snapshot(array, path=f"base/{key}") for key, array in value.items()})
            elif name == "body_arrays" and value is not None:
                value = MappingProxyType({key: _readonly_snapshot(array, path=f"body/{key}") for key, array in value.items()})
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_verification_seal", _HANDLE_SEAL)

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        values = {**{f"base/{key}": value for key, value in self.base_arrays.items()}}
        if self.body_arrays is not None:
            values.update({f"body/{key}": value for key, value in self.body_arrays.items()})
        return MappingProxyType(values)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.run_manifest

    @property
    def schema_id(self) -> str:
        return str(self.run_manifest["schema_binding"]["schema_id"])

    @property
    def schema_version(self) -> int:
        return int(self.run_manifest["schema_binding"]["schema_version"])

    @property
    def layout(self) -> str:
        return str(self.run_manifest["schema_binding"]["layout"])

    @property
    def manifest_sha256(self) -> str:
        return canonical_json_sha256(_thaw_json(self.run_manifest))

    @property
    def payload_digest(self) -> str:
        return str(self.run_manifest["payload_digest"])

    @property
    def body_available(self) -> bool:
        return self.body_arrays is not None

    @property
    def verification_authority(self) -> Mapping[str, Any]:
        """Identify whether this handle is deep-audited or receipt-backed."""

        return MappingProxyType(
            {
                "verification_mode": self.verification_mode,
                "verification_digest": self.verification_digest,
                "receipt_digest": self.receipt_digest,
            }
        )

    def array(self, path: str) -> np.ndarray:
        if type(path) is not str or path not in self.arrays:
            raise KeyError(f"Unknown chaser-relative-frame array {path!r}.")
        return self.arrays[path]

    def base_array(self, name: str) -> np.ndarray:
        if type(name) is not str or "/" in name:
            raise KeyError(f"Unknown base array {name!r}.")
        try:
            return self.base_arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown base array {name!r}.") from exc

    def body_array(self, name: str) -> np.ndarray:
        if self.body_arrays is None:
            raise ChaserRelativeFrameBodyUnavailableError(
                "This position-only chaser-relative-frame run has no body extension."
            )
        if type(name) is not str or "/" in name:
            raise KeyError(f"Unknown body array {name!r}.")
        try:
            return self.body_arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown body array {name!r}.") from exc

    def reshape_frame_chaser(self, values: np.ndarray) -> np.ndarray:
        """Return a read-only frame-by-chaser view of one flat declared array."""

        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.n_rows:
            raise ValueError(
                f"Expected a flat first axis of length {self.n_rows}; got {array.shape}."
            )
        reshaped = array.reshape((self.n_frames, self.n_chasers) + array.shape[1:])
        reshaped.setflags(write=False)
        return reshaped

    def base_frame_chaser(self, name: str) -> np.ndarray:
        return self.reshape_frame_chaser(self.base_array(name))

    def body_frame_chaser(self, name: str) -> np.ndarray:
        return self.reshape_frame_chaser(self.body_array(name))

    def assert_current(self) -> None:
        if self._verification_seal is not _HANDLE_SEAL:
            raise ChaserRelativeFrameSourceHandleError("Handle verification seal is absent.")
        if self.verification_mode == "receipt_backed":
            if self._receipt_document is None:
                raise ChaserRelativeFrameSourceHandleError(
                    "Receipt-backed handle has no sealed receipt authority."
                )
            refreshed = load_chaser_relative_frame_source_handle_from_receipt(
                self.analysis_zarr_path,
                receipt=_thaw_json(self._receipt_document),
                expected_recording_id=self.recording_id,
                use_consolidated=self._use_consolidated,
            )
        else:
            refreshed = load_chaser_relative_frame_source_handle(
                self.analysis_zarr_path,
                run_name=self.run_name,
                expected_recording_id=self.recording_id,
                use_consolidated=self._use_consolidated,
            )
        if refreshed.verification_digest != self.verification_digest:
            raise ChaserRelativeFrameSourceHandleError(
                "Published chaser-relative-frame candidate changed after the handle was sealed."
            )

    def assert_verified(self) -> None:
        self.assert_current()


def load_chaser_relative_frame_source_handle(
    analysis_zarr_path: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
    use_consolidated: bool = True,
) -> ChaserRelativeFrameSourceHandle:
    """Load one exact published candidate using consolidated metadata by default."""

    if type(use_consolidated) is not bool:
        raise ChaserRelativeFrameSourceHandleError(
            "use_consolidated must be the exact boolean metadata-read choice."
        )
    if expected_recording_id is not None:
        _require_nonempty_text(expected_recording_id, field="expected_recording_id")
    name = _require_exact_bare_run_name(run_name)
    exact_run_path = f"{CHASER_RELATIVE_FRAME_RUNS_PREFIX}{name}"
    archive = Path(analysis_zarr_path).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}")

    # This is a proof step, not a fallback: a published reader must reject a
    # missing or stale archive-root consolidated generation.
    equivalence = _metadata_equivalence(archive, run_path=exact_run_path)
    try:
        root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
        run = root[exact_run_path]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Unable to open exact published run {exact_run_path!r}: {exc}"
        ) from exc
    if not isinstance(run, zarr.Group):
        raise ChaserRelativeFrameSourceHandleError("Exact published path is not a Zarr group.")

    # Compare the root identity through both metadata views as well.  The
    # subtree proof above deliberately covers only the selected run.
    try:
        direct_root = open_zarr_root(archive, mode="r", use_consolidated=False)
        if direct_root.attrs.get("recording_id") != root.attrs.get("recording_id"):
            raise ChaserRelativeFrameSourceHandleError(
                "Archive-root consolidated recording identity is stale."
            )
        root_recording_id = root.attrs.get("recording_id")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Unable to validate archive-root recording identity: {exc}"
        ) from exc
    manifest, manifest_digest, dimensions = _validate_manifest_digest(
        run,
        expected_recording_id=expected_recording_id,
        exact_run_path=exact_run_path,
    )
    recording_id = manifest["recording_id"]
    if root_recording_id != recording_id:
        raise ChaserRelativeFrameSourceHandleError(
            "Published run recording_id does not match the analysis archive root."
        )
    _validate_completion_and_selection(run, exact_run_path=exact_run_path)
    completion_authority = _completion_authority(
        root,
        run,
        exact_run_path=exact_run_path,
        require_provenance_epoch=False,
    )
    provenance = _validate_provenance(
        run,
        recording_id=recording_id,
        expected_payload_digest=manifest["payload_digest"],
        run_name=name,
    )
    base_arrays, body_arrays, body_present = _validate_declarations_and_arrays(
        run,
        manifest=manifest,
        dimensions=dimensions,
    )
    context, authorities = _validate_context_and_authorities(
        manifest,
        recording_id=recording_id,
        body_present=body_present,
    )
    registries = _validate_identity_registries(
        manifest,
        n_chasers=int(manifest["dimensions"]["n_chasers"]),
        base=base_arrays,
    )
    equivalence_payload = equivalence.to_json()
    verification_document = {
        "schema_id": CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_ID,
        "schema_version": CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_VERSION,
        "run_path": exact_run_path,
        "recording_id": recording_id,
        "manifest_sha256": manifest_digest,
        "payload_digest": manifest["payload_digest"],
        # _validate_declarations_and_arrays has already recomputed and checked
        # every declared content digest.  Reuse those verified declarations in
        # the audit document instead of rereading and hashing every array a
        # second time.
        "arrays": {
            declaration["path"]: declaration["content_sha256"]
            for declaration in manifest["array_declarations"]
        },
        "metadata_equivalence": equivalence_payload,
        "selector_eligible": False,
        "selection": "none",
    }
    return ChaserRelativeFrameSourceHandle(
        analysis_zarr_path=archive,
        run_path=exact_run_path,
        run_name=name,
        recording_id=recording_id,
        selector_eligible=False,
        selection="none",
        n_frames=int(manifest["dimensions"]["n_frames"]),
        n_chasers=int(manifest["dimensions"]["n_chasers"]),
        n_rows=dimensions.n_rows,
        run_manifest=manifest,
        run_provenance=provenance,
        identity_registries=registries,
        source_authorities=authorities,
        context=context,
        base_arrays=base_arrays,
        body_arrays=body_arrays,
        metadata_equivalence=equivalence_payload,
        verification_digest=canonical_json_sha256(verification_document),
        completion_authority=completion_authority,
        verification_mode="deep_audit",
        receipt_digest=None,
        _use_consolidated=use_consolidated,
        _receipt_document=None,
        _verification_seal=_HANDLE_SEAL,
    )


def load_chaser_relative_frame_source_handle_from_receipt(
    analysis_zarr_path: str | Path,
    *,
    receipt: Mapping[str, Any],
    expected_recording_id: str | None = None,
    use_consolidated: bool = True,
) -> ChaserRelativeFrameSourceHandle:
    """Load one exact relative-frame run using a sealed workflow receipt.

    This is an intentionally separate path from
    :func:`load_chaser_relative_frame_source_handle`.  The ordinary loader is
    the deep-audit path and recomputes every dense array digest.  This loader
    trusts the receipt's already-deep-validated relative verification digest,
    checks current metadata and declaration bindings, and reads the scientific
    arrays once for consumers without recomputing their content hashes.
    """

    if type(use_consolidated) is not bool:
        raise ChaserRelativeFrameSourceHandleError(
            "use_consolidated must be the exact boolean metadata-read choice."
        )
    if expected_recording_id is not None:
        _require_nonempty_text(expected_recording_id, field="expected_recording_id")

    # Import lazily: the receipt builder's deep-audit implementation imports
    # this module, while the bounded envelope validator itself performs no
    # source reopening or dense-array hashing.
    from fisheye.analysis_workflows.chaser_proxy_candidate_receipt import (
        validate_chaser_proxy_candidate_receipt_for_source_load,
    )

    archive = Path(analysis_zarr_path).expanduser().resolve()
    receipt_map = validate_chaser_proxy_candidate_receipt_for_source_load(
        receipt,
        expected_analysis_zarr=archive,
        expected_recording_id=expected_recording_id,
    )
    relative_record = receipt_map["relative_frame"]
    if not isinstance(relative_record, Mapping):  # pragma: no cover - validator
        raise ChaserRelativeFrameSourceHandleError(
            "Bounded receipt relative_frame is not one object."
        )
    name = _require_exact_bare_run_name(
        str(relative_record["run_path"]).rsplit("/", 1)[-1]
    )
    exact_run_path = f"{CHASER_RELATIVE_FRAME_RUNS_PREFIX}{name}"
    if relative_record["run_path"] != exact_run_path:
        raise ChaserRelativeFrameSourceHandleError(
            "Bounded receipt relative-frame path is not the exact native run path."
        )
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}")

    equivalence = _metadata_equivalence(archive, run_path=exact_run_path)
    try:
        root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
        run = root[exact_run_path]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Unable to open exact receipt-bound run {exact_run_path!r}: {exc}"
        ) from exc
    if not isinstance(run, zarr.Group):
        raise ChaserRelativeFrameSourceHandleError(
            "Exact receipt-bound path is not a Zarr group."
        )

    try:
        direct_root = open_zarr_root(archive, mode="r", use_consolidated=False)
        if direct_root.attrs.get("recording_id") != root.attrs.get("recording_id"):
            raise ChaserRelativeFrameSourceHandleError(
                "Archive-root consolidated recording identity is stale."
            )
        root_recording_id = root.attrs.get("recording_id")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ChaserRelativeFrameSourceHandleError(
            f"Unable to validate archive-root recording identity: {exc}"
        ) from exc

    manifest, manifest_digest, dimensions = _validate_manifest_digest(
        run,
        expected_recording_id=str(receipt_map["recording_id"]),
        exact_run_path=exact_run_path,
    )
    recording_id = manifest["recording_id"]
    if root_recording_id != recording_id or manifest_digest != relative_record[
        "manifest_sha256"
    ]:
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound recording or manifest digest is stale."
        )
    if manifest["payload_digest"] != relative_record["payload_digest"]:
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound relative-frame payload digest is stale."
        )
    _validate_completion_and_selection(run, exact_run_path=exact_run_path)
    completion_authority = _completion_authority(
        root,
        run,
        exact_run_path=exact_run_path,
        require_provenance_epoch=True,
    )
    if completion_authority != dict(relative_record["completion"]):
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound completion state or epoch is stale."
        )
    provenance = _validate_provenance(
        run,
        recording_id=recording_id,
        expected_payload_digest=manifest["payload_digest"],
        run_name=name,
    )

    # This reads each scientific array once for the consumer, but deliberately
    # skips array_values_sha256.  The exact declaration table is compared to
    # the receipt after its typed metadata has been checked.
    base_arrays, body_arrays, body_present = _validate_declarations_and_arrays(
        run,
        manifest=manifest,
        dimensions=dimensions,
        verify_content_hashes=False,
    )
    if manifest["array_declarations"] != list(relative_record["array_declarations"]):
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound array declarations are stale or reordered."
        )
    if body_present != relative_record["body_extension_present"]:
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound body-extension declaration is stale."
        )
    equivalence_payload = equivalence.to_json()
    if equivalence_payload != dict(relative_record["metadata_equivalence"]):
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound direct/consolidated metadata equivalence is stale."
        )

    context, authorities = _validate_context_and_authorities(
        manifest,
        recording_id=recording_id,
        body_present=body_present,
    )
    registries = _validate_identity_registries(
        manifest,
        n_chasers=int(manifest["dimensions"]["n_chasers"]),
        base=base_arrays,
    )

    proxy_record = receipt_map["input_provenance_proxy"]
    native_record = receipt_map["native_source"]
    projection = context["acquisition_projection"]["record"]
    publication = context["acquisition_projection_publication"]["record"]
    if publication != dict(proxy_record["publication_binding"]):
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound proxy publication binding differs from the run context."
        )
    for authority_field, expected in (
        ("source_run_path", native_record["run_path"]),
        ("source_manifest_sha256", native_record["manifest_sha256"]),
        ("source_verification_digest", native_record["verification_digest"]),
    ):
        if projection.get(authority_field) != expected:
            raise ChaserRelativeFrameSourceHandleError(
                "Receipt-bound native source "
                f"{authority_field} differs from the run context."
            )
    if projection.get("policy_id") != publication.get("policy_id"):
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound proxy policy differs from the run context."
        )

    if manifest["timing_policy"] != dict(relative_record["timing_policy"]):
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound timing policy differs from the current manifest."
        )
    if (
        manifest["timing_policy"].get("timestamp_field") is not None
        or np.any(base_arrays["timestamp_valid"])
        or projection.get("physical_presentation_verified") is not False
        or projection.get("presentation_timestamp_available") is not False
        or projection.get("camera_presentation_clock_transform_available") is not False
        or projection.get("camera_exposure_reference") != "unknown"
    ):
        raise ChaserRelativeFrameSourceHandleError(
            "Receipt-bound temporal proxy caveats are no longer true."
        )

    return ChaserRelativeFrameSourceHandle(
        analysis_zarr_path=archive,
        run_path=exact_run_path,
        run_name=name,
        recording_id=recording_id,
        selector_eligible=False,
        selection="none",
        n_frames=int(manifest["dimensions"]["n_frames"]),
        n_chasers=int(manifest["dimensions"]["n_chasers"]),
        n_rows=dimensions.n_rows,
        run_manifest=manifest,
        run_provenance=provenance,
        identity_registries=registries,
        source_authorities=authorities,
        context=context,
        base_arrays=base_arrays,
        body_arrays=body_arrays,
        metadata_equivalence=equivalence_payload,
        verification_digest=str(relative_record["verification_digest"]),
        completion_authority=completion_authority,
        verification_mode="receipt_backed",
        receipt_digest=str(receipt_map["record_sha256"]),
        _use_consolidated=use_consolidated,
        _receipt_document=receipt_map,
        _verification_seal=_HANDLE_SEAL,
    )


# The longer spelling is retained as an explicit discoverable alias for call
# sites that describe the verification mode rather than the receipt source.
load_chaser_relative_frame_source_handle_receipt_backed = (
    load_chaser_relative_frame_source_handle_from_receipt
)


def require_chaser_relative_frame_source_handle(
    value: object,
) -> ChaserRelativeFrameSourceHandle:
    """Require a loader-minted, verified relative-frame handle."""

    if type(value) is not ChaserRelativeFrameSourceHandle:
        raise ChaserRelativeFrameSourceHandleError(
            "A verified ChaserRelativeFrameSourceHandle is required."
        )
    value.assert_verified()
    return value


__all__ = [
    "CHASER_RELATIVE_FRAME_RUN_PARENT_PATH",
    "CHASER_RELATIVE_FRAME_RUN_PREFIX",
    "CHASER_RELATIVE_FRAME_RUNS_PREFIX",
    "CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_ID",
    "CHASER_RELATIVE_FRAME_SOURCE_HANDLE_SCHEMA_VERSION",
    "ChaserRelativeFrameBodyUnavailableError",
    "ChaserRelativeFrameSourceHandle",
    "ChaserRelativeFrameSourceHandleError",
    "load_chaser_relative_frame_source_handle",
    "load_chaser_relative_frame_source_handle_from_receipt",
    "load_chaser_relative_frame_source_handle_receipt_backed",
    "require_chaser_relative_frame_source_handle",
]
