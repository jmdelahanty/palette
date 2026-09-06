"""Exact receipt composition for reusable recording-behavior capabilities.

The bundle is a small external JSON record.  It does not copy scientific
arrays, resolve selectors, or promote any child.  Instead it validates one
exact-chaser projection receipt, exposes the already sealed children through a
stable capability vocabulary, and records the compatible Core Behavior
sources that are transitively bound by those children.

The first schema is intentionally narrow: it requires the semantic-v2 epoch
behavior child because that child binds the exact provider-motion and
same-track swim-bout sources.  Optional gaze, subject-shape, eye-angle, tail,
and body-alignment capabilities must be represented explicitly when absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping

from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    read_chaser_relative_frame_validation_receipt,
)
from fisheye.analysis_workflows.exact_chaser_projection_receipt import (
    RECEIPT_SCHEMA_ID as PROJECTION_RECEIPT_SCHEMA_ID,
    read_exact_chaser_projection_receipt,
)
from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    read_exact_immutable_child_validation_receipt,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    ZarrMetadataEquivalenceError,
    validate_direct_consolidated_subtree,
)

BUNDLE_SCHEMA_ID = "palette.analysis.validated_recording_behavior_bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_METHOD_ID = "exact_chaser_projection_backed_recording_behavior_v1"
BUNDLE_STATUS = "complete_selector_ineligible_receipt_composition"
MAX_BUNDLE_BYTES = 262_144

CAPABILITY_STATES = frozenset(
    {
        "complete",
        "unavailable",
        "inapplicable",
        "invalid",
        "stale",
        "review_required",
    }
)
REASON_CODES_BY_STATE = MappingProxyType(
    {
        "complete": frozenset({None}),
        "unavailable": frozenset(
            {
                "missing_exact_receipt",
                "no_compatible_source",
                "not_persisted",
                "not_requested",
                "upstream_segmentation_quality",
            }
        ),
        "inapplicable": frozenset({"scientifically_inapplicable"}),
        "invalid": frozenset({"invalid_source"}),
        "stale": frozenset({"stale_source"}),
        "review_required": frozenset(
            {"review_not_accepted", "upstream_segmentation_quality"}
        ),
    }
)

BASE_SOURCE_BINDING_KEYS = (
    "fish_position_keypoint",
    "fish_position_detection",
    "chaser_observations_keypoint_projection",
    "chaser_observations_detection_projection",
    "anatomical_body_frame",
    "row_axis_timing_and_scale",
    "provider_motion",
    "canonical_swim_bouts",
    "semantic_epochs",
    "reviewed_arena_and_scale",
)
OPTIONAL_SOURCE_BINDING_KEYS = ("eye_angles",)
SOURCE_BINDING_KEYS = (*BASE_SOURCE_BINDING_KEYS, *OPTIONAL_SOURCE_BINDING_KEYS)
_SOURCE_BINDING_TYPES = MappingProxyType(
    {
        "fish_position_keypoint": "relative_frame_source_authority_v1",
        "fish_position_detection": "relative_frame_source_authority_v1",
        "chaser_observations_keypoint_projection": (
            "relative_frame_source_authority_v1"
        ),
        "chaser_observations_detection_projection": (
            "relative_frame_source_authority_v1"
        ),
        "anatomical_body_frame": "relative_frame_anatomical_body_authority_v1",
        "row_axis_timing_and_scale": "paired_relative_frame_consensus_v1",
        "provider_motion": "epoch_transitive_provider_motion_v1",
        "canonical_swim_bouts": "epoch_transitive_same_track_swim_bouts_v1",
        "semantic_epochs": "exact_child_plus_epoch_transitive_semantic_v1",
        "reviewed_arena_and_scale": "spatial_radial_consensus_v1",
        "eye_angles": "gaze_transitive_exact_eye_orientation_v1",
    }
)

BASE_SCIENTIFIC_CHILD_KEYS = (
    "semantic_epochs",
    "chaser_relative_keypoint",
    "chaser_relative_detection",
    "radial_near_field_keypoint",
    "radial_near_field_detection",
    "controller_trials",
    "generalized_bout_response",
    "escape_freeze",
    "spatial_occupancy",
    "epoch_behavior",
)
OPTIONAL_SCIENTIFIC_CHILD_KEYS = (
    "body_alignment_by_distance",
    "gaze",
)
EXTERNAL_OPTIONAL_CAPABILITY_KEYS = ("subject_shape", "eye_angles", "tail_kinematics")
CAPABILITY_KEYS = (
    "fish_position_keypoint",
    "fish_position_detection",
    "anatomical_body_frame",
    "provider_motion",
    "canonical_swim_bouts",
    "semantic_epochs",
    "reviewed_arena_and_scale",
    "chaser_relative_keypoint",
    "chaser_relative_detection",
    "radial_near_field_keypoint",
    "radial_near_field_detection",
    "controller_trials",
    "generalized_bout_response",
    "escape_freeze",
    "spatial_occupancy",
    "epoch_behavior",
    "body_alignment_by_distance",
    "gaze",
    *EXTERNAL_OPTIONAL_CAPABILITY_KEYS,
)

VALIDATION_POLICY = {
    "projection_choice": "one_exact_projection_receipt_path_and_sha256",
    "selector_resolution": False,
    "source_array_copy": False,
    "source_array_validation": "typed_consumer_rehashes_only_consumed_arrays",
    "child_metadata_validation": "current_exact_receipts_once_per_bundle_open",
    "provider_motion_metadata_validation": (
        "published_consolidated_direct_equivalence_plus_exact_epoch_bound_manifest_no_array_scan"
    ),
    "optional_capabilities": "explicit_typed_state_required_when_absent",
    "renderer_authority": False,
}
SAFETY = {
    "selector_eligible": False,
    "production_authority": False,
    "registry_update": False,
    "selector_activation": False,
    "zarr_mutation": False,
}

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SELECTOR_PARTS = frozenset(
    {
        "active",
        "active_run",
        "authoritative",
        "authoritative_run",
        "current",
        "current_run",
        "default",
        "default_run",
        "fallback",
        "latest",
        "latest_any",
        "latest_complete",
        "latest_pending",
        "selected",
        "selected_run",
    }
)

_EXACT_CHILD_TO_SCIENTIFIC = MappingProxyType(
    {
        "semantic_selection": "semantic_epochs",
        "keypoint_radial": "radial_near_field_keypoint",
        "detection_radial": "radial_near_field_detection",
        "controller": "controller_trials",
        "bout": "generalized_bout_response",
        "escape": "escape_freeze",
        "spatial_occupancy": "spatial_occupancy",
        "epoch_behavior": "epoch_behavior",
        "body_alignment_by_distance": "body_alignment_by_distance",
        "gaze": "gaze",
    }
)
_RELATIVE_TO_SCIENTIFIC = MappingProxyType(
    {
        "keypoint": "chaser_relative_keypoint",
        "detection": "chaser_relative_detection",
    }
)
_SUCCESSOR_KINDS = MappingProxyType(
    {
        "keypoint_radial": "chaser_radial_near_field",
        "detection_radial": "chaser_radial_near_field",
        "controller": "controller_chase_trials",
        "bout": "generalized_chaser_bout_response",
        "escape": "chaser_escape_freeze",
        "spatial_occupancy": "chaser_spatial_occupancy",
        "body_alignment_by_distance": "chaser_body_alignment_by_distance",
        "gaze": "chaser_gaze_tracking",
    }
)
_EXPECTED_PARENTS = MappingProxyType(
    {
        "semantic_selection": "analysis/protocol_semantic_chaser_selection_runs",
        "keypoint_radial": "analysis/chaser_radial_near_field_runs",
        "detection_radial": "analysis/chaser_radial_near_field_runs",
        "controller": "analysis/controller_chase_trial_runs",
        "bout": "analysis/generalized_chaser_bout_response_runs",
        "escape": "analysis/chaser_escape_freeze_runs",
        "spatial_occupancy": "analysis/chaser_spatial_occupancy_runs",
        "epoch_behavior": "analysis/stimulus_epoch_behavior_summary_runs",
        "body_alignment_by_distance": (
            "analysis/chaser_body_alignment_by_distance_runs"
        ),
        "gaze": "analysis/chaser_gaze_tracking_runs",
        "keypoint_relative": "analysis/chaser_relative_frame_runs",
        "detection_relative": "analysis/chaser_relative_frame_runs",
    }
)
_SHARED_RELATIVE_ARRAY_PATHS = (
    "base/acquisition_frame_id",
    "base/timestamp_ns",
    "base/timestamp_valid",
    "base/selection_member",
    "base/chaser_identity_code",
    "base/chaser_behavior_role_code",
    "base/chaser_occurrence_member",
    "base/chaser_position_xy_px",
    "base/chaser_position_valid",
)
_SHARED_TIMING_POLICY_FIELDS = (
    "policy_id",
    "frame_key_name",
    "track_sample_key_name",
    "timestamp_field",
)


class ValidatedRecordingBehaviorBundleError(ValueError):
    """The requested recording-behavior composition is not exact or compatible."""


def _fail(message: str) -> None:
    raise ValidatedRecordingBehaviorBundleError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if _DIGEST_RE.fullmatch(result) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _commit(value: object) -> str:
    result = _text(value, field="palette_commit")
    if _COMMIT_RE.fullmatch(result) is None:
        _fail("palette_commit must be one full lowercase Git object ID.")
    return result


def _exact_run_path(value: object, *, field: str, parent: str | None = None) -> str:
    result = _text(value, field=field)
    parsed = PurePosixPath(result)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or parsed.parts[0] != "analysis"
        or "\\" in result
        or any(
            part in {"", ".", ".."} or part.casefold() in _SELECTOR_PARTS
            for part in parsed.parts
        )
    ):
        _fail(f"{field} must name one exact non-selector child below analysis/.")
    normalized = parsed.as_posix()
    if parent is not None:
        prefix = f"{parent}/"
        name = normalized.removeprefix(prefix)
        if not normalized.startswith(prefix) or not name or "/" in name:
            _fail(f"{field} must name one exact child below {parent!r}.")
    return normalized


def _canonical_file(value: object, *, field: str) -> Path:
    path = Path(_text(value, field=field)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _binding_from_receipt(receipt: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "receipt_path": str(path),
        "receipt_sha256": _digest(
            receipt.get("record_sha256"), field="child receipt record_sha256"
        ),
        "run_path": _exact_run_path(receipt.get("run_path"), field="child run_path"),
        "manifest_sha256": _digest(
            receipt.get("manifest_sha256"), field="child manifest_sha256"
        ),
        "payload_digest": _digest(
            receipt.get("payload_digest"), field="child payload_digest"
        ),
    }


def _validate_child_binding_record(value: object, *, field: str) -> dict[str, str]:
    binding = _mapping(value, field=field)
    if set(binding) != {
        "receipt_path",
        "receipt_sha256",
        "run_path",
        "manifest_sha256",
        "payload_digest",
    }:
        _fail(f"{field} must be one exact child-receipt binding.")
    receipt_path = Path(
        _text(binding.get("receipt_path"), field=f"{field}.receipt_path")
    )
    if not receipt_path.is_absolute() or str(receipt_path) != str(
        receipt_path.resolve()
    ):
        _fail(f"{field}.receipt_path must be canonical and absolute.")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": _digest(
            binding.get("receipt_sha256"), field=f"{field}.receipt_sha256"
        ),
        "run_path": _exact_run_path(binding.get("run_path"), field=f"{field}.run_path"),
        "manifest_sha256": _digest(
            binding.get("manifest_sha256"), field=f"{field}.manifest_sha256"
        ),
        "payload_digest": _digest(
            binding.get("payload_digest"), field=f"{field}.payload_digest"
        ),
    }


def _validate_source_binding_record(
    value: object, *, binding_key: str
) -> dict[str, Any]:
    field = f"source_bindings.{binding_key}"
    binding = _mapping(value, field=field)
    expected_type = _SOURCE_BINDING_TYPES[binding_key]
    if binding.get("binding_type") != expected_type:
        _fail(f"{field}.binding_type is invalid.")
    expected_fields = {
        "provider_motion": {
            "binding_type",
            "source",
            "source_authority",
            "published_metadata",
            "sealed_by",
        },
    }.get(binding_key, {"binding_type", "authority", "sealed_by"})
    if binding_key in {"canonical_swim_bouts", "semantic_epochs"}:
        expected_fields = {"binding_type", "source", "sealed_by"}
    if set(binding) != expected_fields:
        _fail(f"{field} field set is inexact.")
    evidence_field = "source" if "source" in expected_fields else "authority"
    if not _mapping(binding.get(evidence_field), field=f"{field}.{evidence_field}"):
        _fail(f"{field}.{evidence_field} must not be empty.")
    if "source_authority" in expected_fields and not _mapping(
        binding.get("source_authority"), field=f"{field}.source_authority"
    ):
        _fail(f"{field}.source_authority must not be empty.")
    if "published_metadata" in expected_fields:
        published = _mapping(
            binding.get("published_metadata"), field=f"{field}.published_metadata"
        )
        if set(published) != {
            "schema_id",
            "schema_version",
            "subtree_path",
            "node_count",
            "group_count",
            "array_count",
            "declarations_sha256",
        }:
            _fail(f"{field}.published_metadata field set is inexact.")
        node_count = published.get("node_count")
        group_count = published.get("group_count")
        array_count = published.get("array_count")
        source = _mapping(binding.get("source"), field=f"{field}.source")
        if (
            published.get("schema_id") != "palette.zarr.metadata_equivalence"
            or published.get("schema_version") != 1
            or published.get("subtree_path") != source.get("run_path")
            or type(node_count) is not int
            or type(group_count) is not int
            or type(array_count) is not int
            or min(node_count, group_count, array_count) < 0
            or group_count + array_count != node_count
        ):
            _fail(f"{field}.published_metadata identity or counts are invalid.")
        _digest(
            published.get("declarations_sha256"),
            field=f"{field}.published_metadata.declarations_sha256",
        )
    sealed_by = binding.get("sealed_by")
    if binding_key == "row_axis_timing_and_scale":
        sealed = _mapping(sealed_by, field=f"{field}.sealed_by")
        if set(sealed) != {"keypoint", "detection"}:
            _fail(f"{field}.sealed_by roster is inexact.")
        for role in ("keypoint", "detection"):
            _validate_child_binding_record(
                sealed[role], field=f"{field}.sealed_by.{role}"
            )
    elif binding_key == "reviewed_arena_and_scale":
        sealed = _mapping(sealed_by, field=f"{field}.sealed_by")
        if set(sealed) != {
            "spatial_occupancy",
            "keypoint_radial",
            "detection_radial",
        }:
            _fail(f"{field}.sealed_by roster is inexact.")
        for child_key in (
            "spatial_occupancy",
            "keypoint_radial",
            "detection_radial",
        ):
            _validate_child_binding_record(
                sealed[child_key], field=f"{field}.sealed_by.{child_key}"
            )
    else:
        _validate_child_binding_record(sealed_by, field=f"{field}.sealed_by")
    return _plain(binding)


def _require_binding_equal(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, field: str
) -> None:
    if _plain(observed) != _plain(expected):
        _fail(f"{field} differs from its exact projection-receipt binding.")


def _read_child_receipts(
    projection: Mapping[str, Any],
    *,
    archive: Path,
    recording_id: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    exact: dict[str, Mapping[str, Any]] = {}
    relative: dict[str, Mapping[str, Any]] = {}
    for key, raw_binding in _mapping(
        projection.get("exact_children"), field="projection exact_children"
    ).items():
        binding = _mapping(raw_binding, field=f"exact_children.{key}")
        path = _canonical_file(
            binding.get("receipt_path"), field=f"exact_children.{key}.receipt_path"
        )
        receipt = read_exact_immutable_child_validation_receipt(
            path,
            expected_analysis_zarr=archive,
            expected_recording_id=recording_id,
            expected_run_path=binding.get("run_path"),
        )
        _require_binding_equal(
            _binding_from_receipt(receipt, path),
            binding,
            field=f"exact child {key!r}",
        )
        exact[str(key)] = receipt
    for key, raw_binding in _mapping(
        projection.get("relative_frame_children"),
        field="projection relative_frame_children",
    ).items():
        binding = _mapping(raw_binding, field=f"relative_frame_children.{key}")
        path = _canonical_file(
            binding.get("receipt_path"),
            field=f"relative_frame_children.{key}.receipt_path",
        )
        expected_path = _exact_run_path(
            binding.get("run_path"), field=f"relative_frame_children.{key}.run_path"
        )
        receipt = read_chaser_relative_frame_validation_receipt(
            path,
            expected_analysis_zarr=archive,
            expected_recording_id=recording_id,
            expected_run_name=PurePosixPath(expected_path).name,
            expected_manifest_sha256=binding.get("manifest_sha256"),
        )
        _require_binding_equal(
            _binding_from_receipt(receipt, path),
            binding,
            field=f"relative-frame child {key!r}",
        )
        relative[str(key)] = receipt
    return exact, relative


def _scientific_manifest(
    receipt: Mapping[str, Any], *, child_key: str
) -> Mapping[str, Any]:
    outer = _mapping(receipt.get("manifest"), field=f"{child_key} manifest")
    expected_kind = _SUCCESSOR_KINDS.get(child_key)
    if expected_kind is None:
        return outer
    if (
        outer.get("schema_id") != "palette.analysis.composable_chaser_successor.run"
        or outer.get("schema_version") != 1
        or outer.get("successor_kind") != expected_kind
        or outer.get("run_path") != receipt.get("run_path")
        or outer.get("recording_id") != receipt.get("recording_id")
        or outer.get("selector_eligible") is not False
        or outer.get("production_authority") is not False
    ):
        _fail(f"Exact child {child_key!r} has the wrong publication identity.")
    scientific = _mapping(
        outer.get("scientific_manifest"),
        field=f"{child_key} scientific manifest",
    )
    unsigned = {
        key: _plain(value)
        for key, value in scientific.items()
        if key != "payload_digest"
    }
    if _digest(
        scientific.get("payload_digest"),
        field=f"{child_key} scientific payload_digest",
    ) != canonical_json_sha256(unsigned) or outer.get(
        "scientific_payload_sha256"
    ) != scientific.get(
        "payload_digest"
    ):
        _fail(f"Exact child {child_key!r} scientific payload binding is stale.")
    return scientific


def _simple_binding(binding: Mapping[str, Any], *, field: str) -> dict[str, str]:
    return {
        "run_path": _exact_run_path(binding.get("run_path"), field=f"{field}.run_path"),
        "manifest_sha256": _digest(
            binding.get("manifest_sha256"), field=f"{field}.manifest_sha256"
        ),
    }


def _proof(policy_id: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    plain_evidence = _plain(evidence)
    return {
        "status": "proved",
        "policy_id": policy_id,
        "evidence": plain_evidence,
        "evidence_sha256": canonical_json_sha256(plain_evidence),
    }


def _relative_evidence(
    receipt: Mapping[str, Any], *, role: str, receipt_path: Path
) -> dict[str, Any]:
    manifest = _mapping(receipt.get("run_manifest"), field=f"{role} relative manifest")
    expected_parent = _EXPECTED_PARENTS[f"{role}_relative"]
    _exact_run_path(
        receipt.get("run_path"),
        field=f"{role} relative run_path",
        parent=expected_parent,
    )
    dimensions = _mapping(manifest.get("dimensions"), field=f"{role} dimensions")
    authorities = _mapping(
        manifest.get("source_authorities"), field=f"{role} source authorities"
    )
    fish = _mapping(authorities.get("fish_position"), field=f"{role} fish authority")
    chaser = _mapping(
        authorities.get("chaser_position"), field=f"{role} chaser authority"
    )
    for name, authority in (("fish", fish), ("chaser", chaser)):
        if authority.get("recording_id") != receipt.get("recording_id"):
            _fail(f"{role} relative {name} authority belongs to another recording.")
        for digest_field in (
            "source_digest",
            "provider_digest",
            "row_axis_authority_digest",
        ):
            _text(
                authority.get(digest_field),
                field=f"{role} {name} authority {digest_field}",
            )
    body = authorities.get("body_frame")
    if role == "keypoint":
        body = _mapping(body, field="keypoint body-frame authority")
        if body.get("recording_id") != receipt.get("recording_id"):
            _fail("Keypoint body-frame authority belongs to another recording.")
        schema = _mapping(manifest.get("schema_binding"), field="relative schema")
        if schema.get("body_extension_present") is not True:
            _fail("Keypoint relative child lacks its anatomical body extension.")
    elif body is not None:
        body = _mapping(body, field="detection body-frame authority")
    declarations = receipt.get("array_declarations")
    if not isinstance(declarations, list):
        _fail(f"{role} relative receipt lacks its sealed array declarations.")
    by_path = {
        item.get("path"): item
        for item in declarations
        if isinstance(item, Mapping) and type(item.get("path")) is str
    }
    if len(by_path) != len(declarations):
        _fail(f"{role} relative array declarations are duplicated or malformed.")
    shared_declarations: dict[str, Any] = {}
    for path in _SHARED_RELATIVE_ARRAY_PATHS:
        declaration = _mapping(
            by_path.get(path), field=f"{role} relative declaration {path}"
        )
        if set(declaration) != {"path", "dtype", "shape", "content_sha256"}:
            _fail(f"{role} relative declaration {path!r} is inexact.")
        _text(declaration.get("dtype"), field=f"{role} {path} dtype")
        shape = declaration.get("shape")
        if not isinstance(shape, list) or any(
            type(value) is not int or value < 0 for value in shape
        ):
            _fail(f"{role} relative declaration {path!r} shape is invalid.")
        _digest(
            declaration.get("content_sha256"),
            field=f"{role} {path} content_sha256",
        )
        shared_declarations[path] = _plain(declaration)
    return {
        "binding": _binding_from_receipt(receipt, receipt_path),
        "manifest": manifest,
        "dimensions": dimensions,
        "fish": fish,
        "chaser": chaser,
        "body": body,
        "coordinate": _mapping(
            manifest.get("coordinate_policy"), field=f"{role} coordinate policy"
        ),
        "scale": _mapping(manifest.get("scale_policy"), field=f"{role} scale policy"),
        "timing": _mapping(
            manifest.get("timing_policy"), field=f"{role} timing policy"
        ),
        "context": _mapping(manifest.get("context"), field=f"{role} context"),
        "registries": _mapping(
            manifest.get("identity_registries"), field=f"{role} identity registries"
        ),
        "shared_array_declarations": shared_declarations,
    }


def _paired_relative_proof(
    keypoint: Mapping[str, Any], detection: Mapping[str, Any]
) -> dict[str, Any]:
    def context_record(source: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        envelope = _mapping(
            source["context"].get(name), field=f"relative context {name}"
        )
        if set(envelope) != {"record", "sha256"}:
            _fail(f"Relative context {name!r} envelope is inexact.")
        record = _mapping(
            envelope.get("record"), field=f"relative context {name} record"
        )
        if _digest(
            envelope.get("sha256"), field=f"relative context {name} sha256"
        ) != canonical_json_sha256(_plain(record)):
            _fail(f"Relative context {name!r} digest is stale.")
        return record

    for field in ("dimensions", "coordinate", "scale", "shared_array_declarations"):
        if _plain(keypoint[field]) != _plain(detection[field]):
            _fail(f"Keypoint and detection relative frames differ at {field!r}.")
    keypoint_timing = keypoint["timing"]
    detection_timing = detection["timing"]
    for field in _SHARED_TIMING_POLICY_FIELDS:
        if keypoint_timing.get(field) != detection_timing.get(field):
            _fail(
                "Keypoint and detection relative frames have incompatible timing "
                f"semantics at {field!r}."
            )
    for role in ("fish", "chaser"):
        left = keypoint[role]
        right = detection[role]
        for field in (
            "recording_id",
            "coordinate_authority_id",
            "scale_authority_id",
        ):
            if left.get(field) != right.get(field):
                _fail(
                    "Keypoint and detection relative frames have incompatible "
                    f"{role} {field}."
                )
    for field in ("chaser", "behavior_role"):
        if _plain(keypoint["registries"].get(field)) != _plain(
            detection["registries"].get(field)
        ):
            _fail(
                "Keypoint and detection relative frames disagree on chaser "
                f"{field!r} identity."
            )
    keypoint_occurrence = context_record(keypoint, "chaser_occurrence")
    detection_occurrence = context_record(detection, "chaser_occurrence")
    if _plain(keypoint_occurrence) != _plain(detection_occurrence):
        _fail(
            "Keypoint and detection relative frames disagree on exact chaser "
            "occurrence evidence."
        )
    keypoint_selection = context_record(keypoint, "temporal_selection")
    detection_selection = context_record(detection, "temporal_selection")
    provider_axis_fields = {
        "row_axis_authority_id",
        "row_axis_authority_sha256",
    }
    keypoint_selection_semantics = {
        key: _plain(value)
        for key, value in keypoint_selection.items()
        if key not in provider_axis_fields
    }
    detection_selection_semantics = {
        key: _plain(value)
        for key, value in detection_selection.items()
        if key not in provider_axis_fields
    }
    if keypoint_selection_semantics != detection_selection_semantics:
        _fail(
            "Keypoint and detection relative frames disagree on temporal-selection "
            "semantics."
        )
    return {
        "dimensions": _plain(keypoint["dimensions"]),
        "coordinate_policy": _plain(keypoint["coordinate"]),
        "scale_policy": _plain(keypoint["scale"]),
        "shared_timing_semantics": {
            field: keypoint_timing.get(field) for field in _SHARED_TIMING_POLICY_FIELDS
        },
        "shared_array_declarations": _plain(keypoint["shared_array_declarations"]),
        "temporal_selection_semantics": keypoint_selection_semantics,
        "chaser_occurrence": _plain(keypoint_occurrence),
        "provider_projection_authorities": {
            "keypoint": {
                "timing_authority_id": keypoint["fish"].get("timing_authority_id"),
                "row_axis_authority_id": keypoint["fish"].get("row_axis_authority_id"),
                "row_axis_authority_digest": keypoint["fish"].get(
                    "row_axis_authority_digest"
                ),
                "temporal_selection_row_axis_authority_id": (
                    keypoint_selection.get("row_axis_authority_id")
                ),
                "temporal_selection_row_axis_authority_sha256": (
                    keypoint_selection.get("row_axis_authority_sha256")
                ),
            },
            "detection": {
                "timing_authority_id": detection["fish"].get("timing_authority_id"),
                "row_axis_authority_id": detection["fish"].get("row_axis_authority_id"),
                "row_axis_authority_digest": detection["fish"].get(
                    "row_axis_authority_digest"
                ),
                "temporal_selection_row_axis_authority_id": (
                    detection_selection.get("row_axis_authority_id")
                ),
                "temporal_selection_row_axis_authority_sha256": (
                    detection_selection.get("row_axis_authority_sha256")
                ),
            },
        },
    }


def _provider_motion_manifest(
    archive: Path,
    source: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    run_path = _exact_run_path(
        source.get("run_path"),
        field="provider_motion.run_path",
        parent="analysis/track_kinematics_runs/provider",
    )
    try:
        published_metadata = validate_direct_consolidated_subtree(
            archive, subtree_path=run_path
        ).to_json()
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        ZarrMetadataEquivalenceError,
    ) as exc:
        raise ValidatedRecordingBehaviorBundleError(
            "Provider-motion published metadata is absent, stale, or "
            f"inconsistent: {exc}"
        ) from exc
    metadata_path = archive / run_path / "zarr.json"
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedRecordingBehaviorBundleError(
            f"Cannot read exact provider-motion metadata {metadata_path}: {exc}"
        ) from exc
    attrs = document.get("attributes") if isinstance(document, Mapping) else None
    if (
        not isinstance(document, Mapping)
        or document.get("node_type") != "group"
        or not isinstance(attrs, Mapping)
    ):
        _fail("Provider-motion zarr.json is not one exact group metadata record.")
    manifest = _mapping(
        attrs.get("provider_track_motion_manifest"),
        field="provider-motion manifest",
    )
    if set(manifest) != {"schema_id", "schema_version", "payload", "payload_digest"}:
        _fail("Provider-motion manifest envelope is inexact.")
    payload = _mapping(manifest.get("payload"), field="provider-motion payload")
    payload_digest = _digest(
        manifest.get("payload_digest"), field="provider-motion payload_digest"
    )
    if (
        manifest.get("schema_id") != "palette.provider_track_motion_run_manifest"
        or manifest.get("schema_version") != 1
        or canonical_json_sha256(_plain(payload)) != payload_digest
        or payload_digest != source.get("manifest_sha256")
        or attrs.get("provider_track_motion_manifest_sha256") != payload_digest
        or payload.get("run_path") != run_path
        or payload.get("status") != "complete"
        or payload.get("stage_selector_eligible") is not False
        or attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not False
    ):
        _fail("Provider-motion exact manifest identity or lifecycle is invalid.")
    authority = _mapping(
        payload.get("source_authority"), field="provider-motion source authority"
    )
    record = _mapping(
        authority.get("record"), field="provider-motion source authority record"
    )
    authority_sha = _digest(
        authority.get("sha256"), field="provider-motion source authority sha256"
    )
    if canonical_json_sha256(_plain(record)) != authority_sha:
        _fail("Provider-motion source-authority digest is stale.")
    return manifest, authority, published_metadata


def _epoch_and_motion_proof(
    *,
    archive: Path,
    epoch: Mapping[str, Any],
    semantic_binding: Mapping[str, Any],
    keypoint_relative: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    sources = _mapping(epoch.get("sources"), field="epoch-behavior sources")
    semantic = _mapping(
        sources.get("protocol_semantic_selection"),
        field="epoch-behavior semantic selection",
    )
    if _simple_binding(semantic, field="epoch semantic") != dict(semantic_binding):
        _fail("Epoch behavior binds another protocol-semantic selection.")
    motion = _mapping(sources.get("provider_motion"), field="provider motion")
    bouts = _mapping(sources.get("swim_bouts"), field="canonical swim bouts")
    motion_path = _exact_run_path(
        motion.get("run_path"),
        field="provider_motion.run_path",
        parent="analysis/track_kinematics_runs/provider",
    )
    bout_path = _exact_run_path(
        bouts.get("run_path"),
        field="swim_bouts.run_path",
        parent="analysis/swim_bout_runs",
    )
    motion_manifest_sha = _digest(
        motion.get("manifest_sha256"), field="provider_motion.manifest_sha256"
    )
    motion_verification = _digest(
        motion.get("verification_digest"), field="provider_motion.verification_digest"
    )
    bout_lineage = _digest(bouts.get("lineage_hash"), field="swim_bouts.lineage_hash")
    if (
        type(motion.get("track_id")) is not int
        or motion.get("track_id") < 0
        or motion.get("track_id") != bouts.get("track_id")
        or motion.get("track_row_start") != bouts.get("track_row_start")
        or motion.get("track_row_stop") != bouts.get("track_row_stop")
        or bouts.get("source_track_motion_manifest_sha256") != motion_manifest_sha
        or bouts.get("source_track_motion_verification_digest") != motion_verification
    ):
        _fail("Provider motion and canonical swim bouts are not one exact track.")
    _digest(bouts.get("frame_axis_sha256"), field="swim_bouts.frame_axis_sha256")
    _digest(bouts.get("sha256"), field="swim_bouts.sha256")
    motion_manifest, motion_authority, published_metadata = _provider_motion_manifest(
        archive, motion
    )
    authority_record = _mapping(
        motion_authority.get("record"), field="motion authority record"
    )
    position_source = _mapping(
        authority_record.get("position_source"),
        field="motion authority position source",
    )
    body_source = _mapping(
        authority_record.get("body_frame_source"),
        field="motion authority body-frame source",
    )
    body_authority = _mapping(
        keypoint_relative.get("body"), field="keypoint body authority"
    )
    fish_authority = _mapping(
        keypoint_relative.get("fish"), field="keypoint fish authority"
    )
    if (
        authority_record.get("analysis_zarr_path") != str(archive)
        or motion_authority.get("sha256") != body_authority.get("provider_digest")
        or position_source.get("run_path") != fish_authority.get("source_authority_id")
        or position_source.get("manifest_sha256") != fish_authority.get("source_digest")
        or body_source.get("run_path") != body_authority.get("source_authority_id")
        or body_source.get("manifest_sha256") != body_authority.get("source_digest")
    ):
        _fail(
            "Provider motion and keypoint relative geometry do not share the exact "
            "position/body-frame authority."
        )
    evidence = {
        "provider_motion": _plain(motion),
        "canonical_swim_bouts": _plain(bouts),
        "provider_motion_source_authority_sha256": motion_authority["sha256"],
        "motion_manifest_payload_digest": motion_manifest["payload_digest"],
        "published_metadata": _plain(published_metadata),
        "shared_track_id": motion["track_id"],
        "shared_track_row_start": motion.get("track_row_start"),
        "shared_track_row_stop": motion.get("track_row_stop"),
        "shared_body_frame_run_path": body_source["run_path"],
        "shared_body_frame_manifest_sha256": body_source["manifest_sha256"],
    }
    return evidence, motion, bouts, motion_authority, published_metadata


def _spatial_radial_proof(
    *,
    scientific: Mapping[str, Mapping[str, Any]],
    exact_bindings: Mapping[str, Mapping[str, Any]],
    relative_bindings: Mapping[str, Mapping[str, Any]],
    relatives: Mapping[str, Mapping[str, Any]],
    semantic_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    spatial = scientific["spatial_occupancy"]
    spatial_sources = _mapping(spatial.get("sources"), field="spatial sources")
    if _simple_binding(
        _mapping(
            spatial_sources.get("protocol_semantic_selection"),
            field="spatial semantic selection",
        ),
        field="spatial semantic selection",
    ) != dict(semantic_binding):
        _fail("Spatial occupancy binds another semantic selection.")
    providers = spatial_sources.get("position_providers")
    if not isinstance(providers, list) or len(providers) != 2:
        _fail("Spatial occupancy must bind exactly keypoint and detection providers.")
    if [
        item.get("provider_role") for item in providers if isinstance(item, Mapping)
    ] != [
        "keypoint",
        "detection",
    ]:
        _fail("Spatial occupancy provider roles are not keypoint then detection.")
    geometry = _mapping(
        spatial_sources.get("arena_geometry_and_scale"),
        field="spatial arena geometry and scale",
    )
    radial_keys = ("keypoint_radial", "detection_radial")
    roles = ("keypoint", "detection")
    first_epoch_records: Any = None
    first_arena: Any = None
    for role, radial_key, provider in zip(roles, radial_keys, providers, strict=True):
        provider = _mapping(provider, field=f"spatial {role} provider")
        expected_relative = {
            "run_path": relative_bindings[role]["run_path"],
            "manifest_sha256": relative_bindings[role]["manifest_sha256"],
        }
        if (
            _simple_binding(
                _mapping(
                    provider.get("relative_frame"), field=f"spatial {role} relative"
                ),
                field=f"spatial {role} relative",
            )
            != expected_relative
        ):
            _fail(f"Spatial {role} provider binds another relative-frame child.")
        expected_radial = {
            "run_path": exact_bindings[radial_key]["run_path"],
            "manifest_sha256": exact_bindings[radial_key]["manifest_sha256"],
        }
        if (
            _simple_binding(
                _mapping(
                    provider.get("radial_near_field"), field=f"spatial {role} radial"
                ),
                field=f"spatial {role} radial",
            )
            != expected_radial
        ):
            _fail(f"Spatial {role} provider binds another radial child.")
        fish_authority = relatives[role]["fish"]
        if (
            provider.get("provider_id") != fish_authority.get("provider_id")
            or provider.get("provider_digest") != fish_authority.get("provider_digest")
            or _plain(provider.get("fish_position_authority")) != _plain(fish_authority)
        ):
            _fail(f"Spatial {role} provider authority differs from its relative frame.")
        radial = scientific[radial_key]
        radial_sources = _mapping(radial.get("sources"), field=f"{role} radial sources")
        if (
            _simple_binding(
                _mapping(
                    radial_sources.get("relative_frame"),
                    field=f"{role} radial relative",
                ),
                field=f"{role} radial relative",
            )
            != expected_relative
            or _simple_binding(
                _mapping(
                    radial_sources.get("protocol_semantic_selection"),
                    field=f"{role} radial semantic",
                ),
                field=f"{role} radial semantic",
            )
            != dict(semantic_binding)
            or _plain(radial_sources.get("fish_position")) != _plain(fish_authority)
            or _plain(radial_sources.get("arena_geometry_and_scale"))
            != _plain(geometry)
        ):
            _fail(f"{role.capitalize()} radial sources differ from the spatial bundle.")
        position_provider = _mapping(
            radial.get("position_provider"), field=f"{role} radial position provider"
        )
        if (
            position_provider.get("status") != "first_class_explicit_authority"
            or position_provider.get("provider_id") != provider.get("provider_id")
            or position_provider.get("provider_digest")
            != provider.get("provider_digest")
        ):
            _fail(f"{role.capitalize()} radial provider identity is stale.")
        epoch_records = _plain(radial.get("epoch_records"))
        arena = _plain(radial.get("arena"))
        if first_epoch_records is None:
            first_epoch_records = epoch_records
            first_arena = arena
        elif epoch_records != first_epoch_records or arena != first_arena:
            _fail("Paired radial successors use different epochs or arena geometry.")
    if _plain(spatial.get("epoch_records")) != first_epoch_records:
        _fail("Spatial occupancy and radial successors use different epoch records.")
    return (
        {
            "semantic_selection": dict(semantic_binding),
            "arena_geometry_and_scale": _plain(geometry),
            "provider_roles": ["keypoint", "detection"],
            "keypoint_relative": {
                "run_path": relative_bindings["keypoint"]["run_path"],
                "manifest_sha256": relative_bindings["keypoint"]["manifest_sha256"],
            },
            "detection_relative": {
                "run_path": relative_bindings["detection"]["run_path"],
                "manifest_sha256": relative_bindings["detection"]["manifest_sha256"],
            },
            "keypoint_radial": {
                "run_path": exact_bindings["keypoint_radial"]["run_path"],
                "manifest_sha256": exact_bindings["keypoint_radial"]["manifest_sha256"],
            },
            "detection_radial": {
                "run_path": exact_bindings["detection_radial"]["run_path"],
                "manifest_sha256": exact_bindings["detection_radial"][
                    "manifest_sha256"
                ],
            },
        },
        geometry,
    )


@dataclass(frozen=True)
class BoundExactChaserExtensionSources:
    """One fully validated exact-chaser projection without a core selection.

    This is the shared receipt-resolution boundary used by both the historical
    recording-behavior bundle and a core-roster-backed composite.  It deliberately
    stops before interpreting motion, body-frame, or bout authority: those belong
    to the consuming bundle profile rather than to the chaser receipt grammar.
    """

    projection_path: Path
    projection: Mapping[str, Any]
    analysis_zarr: Path
    recording_id: str
    exact_receipts: Mapping[str, Mapping[str, Any]]
    relative_receipts: Mapping[str, Mapping[str, Any]]
    exact_bindings: Mapping[str, Mapping[str, Any]]
    relative_bindings: Mapping[str, Mapping[str, Any]]
    scientific_child_bindings: Mapping[str, Mapping[str, Any]]
    scientific_manifests: Mapping[str, Mapping[str, Any]]
    semantic_binding: Mapping[str, Any]
    relative_evidence: Mapping[str, Mapping[str, Any]]
    relative_axis: Mapping[str, Any]
    spatial_evidence: Mapping[str, Any]
    reviewed_geometry: Mapping[str, Any]


def bind_exact_chaser_extension_sources(
    projection_receipt_path: str | Path,
    *,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
) -> BoundExactChaserExtensionSources:
    """Resolve one exact chaser receipt through the shared strict grammar.

    Child receipts and their current metadata generations are validated exactly
    once here.  No selector is consulted and no scientific array is copied.
    """

    projection_path = Path(projection_receipt_path).expanduser().resolve()
    projection = read_exact_chaser_projection_receipt(
        projection_path,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
        validate_current_metadata=False,
        validate_child_receipts=False,
    )
    archive = Path(str(projection["analysis_zarr"])).resolve()
    recording_id = str(projection["recording_id"])
    exact, relative = _read_child_receipts(
        projection, archive=archive, recording_id=recording_id
    )
    exact_bindings = {
        key: _binding_from_receipt(
            receipt, Path(str(projection["exact_children"][key]["receipt_path"]))
        )
        for key, receipt in exact.items()
    }
    relative_bindings = {
        key: _binding_from_receipt(
            receipt,
            Path(str(projection["relative_frame_children"][key]["receipt_path"])),
        )
        for key, receipt in relative.items()
    }
    for key, binding in exact_bindings.items():
        parent = _EXPECTED_PARENTS.get(key)
        if parent is not None:
            _exact_run_path(binding["run_path"], field=f"{key} run_path", parent=parent)
    for key, binding in relative_bindings.items():
        _exact_run_path(
            binding["run_path"],
            field=f"{key} relative run_path",
            parent=_EXPECTED_PARENTS[f"{key}_relative"],
        )
    scientific = {
        key: _scientific_manifest(receipt, child_key=key)
        for key, receipt in exact.items()
    }
    semantic_binding = {
        "run_path": exact_bindings["semantic_selection"]["run_path"],
        "manifest_sha256": exact_bindings["semantic_selection"]["manifest_sha256"],
    }
    relatives = {
        role: _relative_evidence(
            relative[role],
            role=role,
            receipt_path=Path(
                str(projection["relative_frame_children"][role]["receipt_path"])
            ),
        )
        for role in ("keypoint", "detection")
    }
    relative_axis = _paired_relative_proof(
        relatives["keypoint"], relatives["detection"]
    )
    spatial_evidence, geometry = _spatial_radial_proof(
        scientific=scientific,
        exact_bindings=exact_bindings,
        relative_bindings=relative_bindings,
        relatives=relatives,
        semantic_binding=semantic_binding,
    )
    scientific_children = {
        _EXACT_CHILD_TO_SCIENTIFIC[key]: binding
        for key, binding in exact_bindings.items()
    }
    scientific_children.update(
        {
            _RELATIVE_TO_SCIENTIFIC[key]: binding
            for key, binding in relative_bindings.items()
        }
    )
    if set(BASE_SCIENTIFIC_CHILD_KEYS).difference(scientific_children):
        _fail("Projection receipt lacks a required base scientific child.")
    return BoundExactChaserExtensionSources(
        projection_path=projection_path,
        projection=projection,
        analysis_zarr=archive,
        recording_id=recording_id,
        exact_receipts=MappingProxyType(exact),
        relative_receipts=MappingProxyType(relative),
        exact_bindings=MappingProxyType(exact_bindings),
        relative_bindings=MappingProxyType(relative_bindings),
        scientific_child_bindings=MappingProxyType(scientific_children),
        scientific_manifests=MappingProxyType(scientific),
        semantic_binding=MappingProxyType(semantic_binding),
        relative_evidence=MappingProxyType(relatives),
        relative_axis=MappingProxyType(relative_axis),
        spatial_evidence=MappingProxyType(spatial_evidence),
        reviewed_geometry=geometry,
    )


def _chain_proofs(
    *,
    scientific: Mapping[str, Mapping[str, Any]],
    exact_bindings: Mapping[str, Mapping[str, Any]],
    relative_bindings: Mapping[str, Mapping[str, Any]],
    semantic_binding: Mapping[str, Any],
    motion: Mapping[str, Any],
    bouts: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    keypoint_relative = {
        "run_path": relative_bindings["keypoint"]["run_path"],
        "manifest_sha256": relative_bindings["keypoint"]["manifest_sha256"],
    }
    controller = scientific["controller"]
    if _simple_binding(
        _mapping(
            controller.get("source_relative_frame"),
            field="controller relative frame",
        ),
        field="controller relative frame",
    ) != keypoint_relative or _simple_binding(
        _mapping(
            controller.get("semantic_selection"),
            field="controller semantic selection",
        ),
        field="controller semantic selection",
    ) != dict(
        semantic_binding
    ):
        _fail("Controller-trial child binds another relative or semantic source.")
    controller_payload = _digest(
        controller.get("payload_digest"), field="controller payload_digest"
    )
    bout = scientific["bout"]
    bout_sources = _mapping(bout.get("sources"), field="bout-response sources")
    bout_motion = _mapping(bout_sources.get("motion"), field="bout motion source")
    bout_swim = _mapping(bout_sources.get("swim_bouts"), field="bout swim source")
    if (
        _simple_binding(
            _mapping(bout_sources.get("relative_frame"), field="bout relative frame"),
            field="bout relative frame",
        )
        != keypoint_relative
        or bout_motion.get("run_path") != motion.get("run_path")
        or bout_motion.get("manifest_sha256") != motion.get("manifest_sha256")
        or bout_swim.get("run_path") != bouts.get("run_path")
        or bout_swim.get("lineage_sha256") != bouts.get("lineage_hash")
        or bout_sources.get("semantic_selection_manifest_sha256")
        != semantic_binding["manifest_sha256"]
        or bout_sources.get("controller_trial_payload_sha256") != controller_payload
    ):
        _fail("Generalized bout-response dependency chain is stale or mixed.")
    bout_payload = _digest(bout.get("payload_digest"), field="bout payload_digest")
    escape = scientific["escape"]
    escape_sources = _mapping(escape.get("sources"), field="escape sources")
    escape_motion = _mapping(escape_sources.get("motion"), field="escape motion source")
    if (
        escape_motion.get("run_path") != motion.get("run_path")
        or escape_motion.get("manifest_sha256") != motion.get("manifest_sha256")
        or _plain(escape_motion.get("relative_frame_projection"))
        != _plain(bout_motion.get("relative_frame_projection"))
        or escape_sources.get("controller_trial_payload_sha256") != controller_payload
        or escape_sources.get("bout_response_payload_sha256") != bout_payload
    ):
        _fail("Escape/freeze dependency chain is stale or mixed.")
    proofs = {
        "controller_trial_chain": _proof(
            "exact_keypoint_relative_and_semantic_selection_v1",
            {
                "relative_frame": keypoint_relative,
                "semantic_selection": dict(semantic_binding),
                "controller_payload_digest": controller_payload,
            },
        ),
        "generalized_bout_response_chain": _proof(
            "exact_motion_bout_relative_semantic_controller_chain_v1",
            {
                "relative_frame": keypoint_relative,
                "provider_motion": {
                    "run_path": motion["run_path"],
                    "manifest_sha256": motion["manifest_sha256"],
                },
                "swim_bouts": {
                    "run_path": bouts["run_path"],
                    "lineage_hash": bouts["lineage_hash"],
                },
                "semantic_selection_manifest_sha256": semantic_binding[
                    "manifest_sha256"
                ],
                "controller_payload_digest": controller_payload,
                "bout_payload_digest": bout_payload,
            },
        ),
        "escape_freeze_chain": _proof(
            "exact_controller_bout_motion_projection_chain_v1",
            {
                "controller_payload_digest": controller_payload,
                "bout_payload_digest": bout_payload,
                "motion_projection_sha256": canonical_json_sha256(
                    _plain(bout_motion.get("relative_frame_projection"))
                ),
                "escape_payload_digest": escape["payload_digest"],
            },
        ),
    }
    if "body_alignment_by_distance" in scientific:
        alignment = scientific["body_alignment_by_distance"]
        alignment_sources = _mapping(
            alignment.get("sources"), field="body-alignment sources"
        )
        if _simple_binding(
            _mapping(
                alignment_sources.get("relative_frame"),
                field="body-alignment relative frame",
            ),
            field="body-alignment relative frame",
        ) != keypoint_relative or _simple_binding(
            _mapping(
                alignment_sources.get("protocol_semantic_selection"),
                field="body-alignment semantic selection",
            ),
            field="body-alignment semantic selection",
        ) != dict(
            semantic_binding
        ):
            _fail("Body-alignment child binds another relative or semantic source.")
        proofs["body_alignment_chain"] = _proof(
            "exact_anatomical_relative_and_semantic_selection_v1",
            {
                "relative_frame": keypoint_relative,
                "semantic_selection": dict(semantic_binding),
            },
        )
    if "gaze" in scientific:
        gaze = scientific["gaze"]
        gaze_sources = _mapping(gaze.get("sources"), field="gaze sources")
        radial = _mapping(
            gaze_sources.get("radial_near_field_geometry_authority"),
            field="gaze radial authority",
        )
        expected_radial = exact_bindings["keypoint_radial"]
        eye = _mapping(gaze_sources.get("eye_orientation"), field="gaze eye source")
        _exact_run_path(eye.get("run_path"), field="gaze eye run_path")
        _digest(eye.get("manifest_sha256"), field="gaze eye manifest_sha256")
        convention = _digest(
            eye.get("convention_receipt_sha256"),
            field="gaze convention receipt sha256",
        )
        if (
            _simple_binding(
                _mapping(
                    gaze_sources.get("relative_frame"), field="gaze relative frame"
                ),
                field="gaze relative frame",
            )
            != keypoint_relative
            or gaze_sources.get("semantic_selection_manifest_sha256")
            != semantic_binding["manifest_sha256"]
            or radial.get("run_path") != expected_radial["run_path"]
            or radial.get("manifest_sha256") != expected_radial["manifest_sha256"]
            or _plain(radial.get("arena_geometry_and_scale")) != _plain(geometry)
        ):
            _fail(
                "Gaze child binds another relative, semantic, radial, or arena source."
            )
        proofs["gaze_chain"] = _proof(
            "exact_eye_convention_relative_radial_semantic_chain_v1",
            {
                "eye_orientation": _plain(eye),
                "accepted_convention_receipt_sha256": convention,
                "relative_frame": keypoint_relative,
                "radial_near_field": {
                    "run_path": expected_radial["run_path"],
                    "manifest_sha256": expected_radial["manifest_sha256"],
                },
                "semantic_selection_manifest_sha256": semantic_binding[
                    "manifest_sha256"
                ],
            },
        )
    return proofs


def _normalize_disposition(value: object, *, capability: str) -> dict[str, Any]:
    record = _mapping(value, field=f"capability disposition {capability!r}")
    if set(record) != {"state", "reason_code", "detail"}:
        _fail(
            f"Capability disposition {capability!r} must contain exactly state, "
            "reason_code, and detail."
        )
    state = record.get("state")
    if state not in CAPABILITY_STATES or state == "complete":
        _fail(
            f"Absent capability {capability!r} requires one non-complete typed state."
        )
    reason = record.get("reason_code")
    if reason not in REASON_CODES_BY_STATE[state]:
        _fail(f"Capability {capability!r} has an invalid reason for state {state!r}.")
    detail = record.get("detail")
    if detail is not None:
        detail = _text(detail, field=f"capability {capability} detail")
        if len(detail.encode("utf-8")) > 512:
            _fail(f"Capability {capability!r} detail exceeds 512 UTF-8 bytes.")
    return {
        "state": state,
        "reason_code": reason,
        "detail": detail,
        "binding_scope": None,
        "binding_key": None,
    }


def _complete_capability(scope: str, key: str) -> dict[str, Any]:
    return {
        "state": "complete",
        "reason_code": None,
        "detail": None,
        "binding_scope": scope,
        "binding_key": key,
    }


def _resolve_bundle_content(
    projection_receipt_path: str | Path,
    *,
    absent_capability_dispositions: Mapping[str, Mapping[str, Any]],
    expected_analysis_zarr: str | Path | None,
    expected_recording_id: str | None,
) -> dict[str, Any]:
    extension = bind_exact_chaser_extension_sources(
        projection_receipt_path,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
    )
    projection_path = extension.projection_path
    projection = extension.projection
    archive = extension.analysis_zarr
    recording_id = extension.recording_id
    exact = extension.exact_receipts
    if "epoch_behavior" not in exact:
        _fail(
            "Validated recording-behavior schema v1 requires an exact "
            "epoch_behavior child to bind provider motion and swim bouts."
        )
    exact_bindings = extension.exact_bindings
    relative_bindings = extension.relative_bindings
    scientific = extension.scientific_manifests
    semantic_binding = extension.semantic_binding
    relatives = extension.relative_evidence
    relative_axis = extension.relative_axis
    spatial_evidence = extension.spatial_evidence
    geometry = extension.reviewed_geometry
    epoch = scientific["epoch_behavior"]
    (
        epoch_evidence,
        motion,
        bouts,
        motion_authority,
        motion_published_metadata,
    ) = _epoch_and_motion_proof(
        archive=archive,
        epoch=epoch,
        semantic_binding=semantic_binding,
        keypoint_relative=relatives["keypoint"],
    )
    chain_proofs = _chain_proofs(
        scientific=scientific,
        exact_bindings=exact_bindings,
        relative_bindings=relative_bindings,
        semantic_binding=semantic_binding,
        motion=motion,
        bouts=bouts,
        geometry=geometry,
    )

    scientific_children = extension.scientific_child_bindings

    source_bindings = {
        "fish_position_keypoint": {
            "binding_type": "relative_frame_source_authority_v1",
            "authority": _plain(relatives["keypoint"]["fish"]),
            "sealed_by": relative_bindings["keypoint"],
        },
        "fish_position_detection": {
            "binding_type": "relative_frame_source_authority_v1",
            "authority": _plain(relatives["detection"]["fish"]),
            "sealed_by": relative_bindings["detection"],
        },
        "chaser_observations_keypoint_projection": {
            "binding_type": "relative_frame_source_authority_v1",
            "authority": _plain(relatives["keypoint"]["chaser"]),
            "sealed_by": relative_bindings["keypoint"],
        },
        "chaser_observations_detection_projection": {
            "binding_type": "relative_frame_source_authority_v1",
            "authority": _plain(relatives["detection"]["chaser"]),
            "sealed_by": relative_bindings["detection"],
        },
        "anatomical_body_frame": {
            "binding_type": "relative_frame_anatomical_body_authority_v1",
            "authority": _plain(relatives["keypoint"]["body"]),
            "sealed_by": relative_bindings["keypoint"],
        },
        "row_axis_timing_and_scale": {
            "binding_type": "paired_relative_frame_consensus_v1",
            "authority": relative_axis,
            "sealed_by": {
                "keypoint": relative_bindings["keypoint"],
                "detection": relative_bindings["detection"],
            },
        },
        "provider_motion": {
            "binding_type": "epoch_transitive_provider_motion_v1",
            "source": _plain(motion),
            "source_authority": _plain(motion_authority),
            "published_metadata": _plain(motion_published_metadata),
            "sealed_by": exact_bindings["epoch_behavior"],
        },
        "canonical_swim_bouts": {
            "binding_type": "epoch_transitive_same_track_swim_bouts_v1",
            "source": _plain(bouts),
            "sealed_by": exact_bindings["epoch_behavior"],
        },
        "semantic_epochs": {
            "binding_type": "exact_child_plus_epoch_transitive_semantic_v1",
            "source": _plain(epoch["sources"]["protocol_semantic_selection"]),
            "sealed_by": exact_bindings["semantic_selection"],
        },
        "reviewed_arena_and_scale": {
            "binding_type": "spatial_radial_consensus_v1",
            "authority": _plain(geometry),
            "sealed_by": {
                "spatial_occupancy": exact_bindings["spatial_occupancy"],
                "keypoint_radial": exact_bindings["keypoint_radial"],
                "detection_radial": exact_bindings["detection_radial"],
            },
        },
    }
    if "gaze" in scientific:
        gaze_sources = _mapping(scientific["gaze"].get("sources"), field="gaze sources")
        eye_source = _mapping(
            gaze_sources.get("eye_orientation"), field="gaze eye orientation"
        )
        source_bindings["eye_angles"] = {
            "binding_type": "gaze_transitive_exact_eye_orientation_v1",
            "authority": _plain(eye_source),
            "sealed_by": exact_bindings["gaze"],
        }

    capability_bindings = {
        "fish_position_keypoint": ("source_bindings", "fish_position_keypoint"),
        "fish_position_detection": ("source_bindings", "fish_position_detection"),
        "anatomical_body_frame": ("source_bindings", "anatomical_body_frame"),
        "provider_motion": ("source_bindings", "provider_motion"),
        "canonical_swim_bouts": ("source_bindings", "canonical_swim_bouts"),
        "semantic_epochs": ("scientific_child_bindings", "semantic_epochs"),
        "reviewed_arena_and_scale": (
            "source_bindings",
            "reviewed_arena_and_scale",
        ),
        **{
            key: ("scientific_child_bindings", key)
            for key in scientific_children
            if key != "semantic_epochs"
        },
    }
    if "eye_angles" in source_bindings:
        capability_bindings["eye_angles"] = ("source_bindings", "eye_angles")
    capabilities = {
        key: _complete_capability(*binding)
        for key, binding in capability_bindings.items()
    }
    missing_optional = {
        *(
            key
            for key in EXTERNAL_OPTIONAL_CAPABILITY_KEYS
            if key not in capability_bindings
        ),
        *(
            key
            for key in OPTIONAL_SCIENTIFIC_CHILD_KEYS
            if key not in scientific_children
        ),
    }
    if set(absent_capability_dispositions) != missing_optional:
        missing = sorted(missing_optional.difference(absent_capability_dispositions))
        extra = sorted(set(absent_capability_dispositions).difference(missing_optional))
        _fail(
            "Absent capability dispositions are inexact; "
            f"missing={missing!r}, unexpected={extra!r}."
        )
    for key in sorted(missing_optional):
        capabilities[key] = _normalize_disposition(
            absent_capability_dispositions[key], capability=key
        )
    if set(capabilities) != set(CAPABILITY_KEYS):
        _fail("Resolved capability roster is incomplete or unexpected.")

    proofs = {
        "projection_receipt_closed_roster": _proof(
            "exact_projection_receipt_self_digest_and_closed_children_v1",
            {
                "projection_receipt_path": str(projection_path),
                "projection_receipt_sha256": projection["record_sha256"],
                "projection_schema_version": projection["schema_version"],
                "exact_child_keys": sorted(exact),
                "relative_child_keys": sorted(extension.relative_receipts),
            },
        ),
        "paired_relative_frame_axis": _proof(
            "exact_keypoint_detection_axis_timing_scale_consensus_v1",
            relative_axis,
        ),
        "spatial_radial_provider_composition": _proof(
            "exact_two_provider_spatial_radial_semantic_arena_consensus_v1",
            spatial_evidence,
        ),
        "provider_motion_swim_bout_body_frame_composition": _proof(
            "exact_epoch_motion_same_track_bouts_and_body_frame_lineage_v1",
            epoch_evidence,
        ),
        **chain_proofs,
    }
    return _plain(
        {
            "analysis_zarr": str(archive),
            "recording_id": recording_id,
            "projection_receipt": {
                "receipt_path": str(projection_path),
                "receipt_sha256": projection["record_sha256"],
                "schema_id": PROJECTION_RECEIPT_SCHEMA_ID,
                "schema_version": projection["schema_version"],
            },
            "source_bindings": source_bindings,
            "scientific_child_bindings": scientific_children,
            "capabilities": capabilities,
            "compatibility_proofs": proofs,
            "validation_policy": VALIDATION_POLICY,
            "safety": SAFETY,
        }
    )


def build_validated_recording_behavior_bundle(
    projection_receipt_path: str | Path,
    *,
    absent_capability_dispositions: Mapping[str, Mapping[str, Any]],
    palette_commit: str,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
) -> dict[str, Any]:
    """Build one immutable read-only composition from exact receipt evidence."""

    content = _resolve_bundle_content(
        projection_receipt_path,
        absent_capability_dispositions=absent_capability_dispositions,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
    )
    body = {
        "schema_id": BUNDLE_SCHEMA_ID,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "method_id": BUNDLE_METHOD_ID,
        "status": BUNDLE_STATUS,
        **content,
        "software_authority": {
            "repository": "palette",
            "commit": _commit(palette_commit),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidatedRecordingBehaviorBundleError(
            f"Bundle is not strict JSON: {exc}"
        ) from exc
    if len(encoded) > MAX_BUNDLE_BYTES:
        _fail(
            "Validated recording-behavior bundle exceeds its 256-KiB metadata " "limit."
        )
    return {**body, "record_sha256": canonical_json_sha256(body)}


def _validate_capability_record(value: object, *, capability: str) -> dict[str, Any]:
    record = _mapping(value, field=f"capabilities.{capability}")
    required = {"state", "reason_code", "detail", "binding_scope", "binding_key"}
    if set(record) != required:
        _fail(f"Capability {capability!r} field set is inexact.")
    state = record.get("state")
    if state not in CAPABILITY_STATES:
        _fail(f"Capability {capability!r} state is invalid.")
    reason = record.get("reason_code")
    if reason not in REASON_CODES_BY_STATE[state]:
        _fail(f"Capability {capability!r} reason is invalid for state {state!r}.")
    detail = record.get("detail")
    if detail is not None:
        detail = _text(detail, field=f"capabilities.{capability}.detail")
        if len(detail.encode("utf-8")) > 512:
            _fail(f"Capability {capability!r} detail exceeds 512 UTF-8 bytes.")
    if state == "complete":
        if (
            record.get("binding_scope")
            not in {
                "source_bindings",
                "scientific_child_bindings",
            }
            or record.get("binding_key") != capability
        ):
            _fail(f"Complete capability {capability!r} lacks one exact binding.")
        if detail is not None:
            _fail(f"Complete capability {capability!r} cannot carry detail.")
    elif (
        record.get("binding_scope") is not None or record.get("binding_key") is not None
    ):
        _fail(f"Non-complete capability {capability!r} must not name a binding.")
    return {
        "state": state,
        "reason_code": reason,
        "detail": detail,
        "binding_scope": record.get("binding_scope"),
        "binding_key": record.get("binding_key"),
    }


def validate_validated_recording_behavior_bundle(
    bundle: object,
    *,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
    validate_current_sources: bool = True,
) -> Mapping[str, Any]:
    """Validate the bundle envelope and, by default, every current source binding."""

    if not isinstance(bundle, Mapping):
        _fail("Validated recording-behavior bundle must be one object.")
    value = _plain(bundle)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidatedRecordingBehaviorBundleError(
            f"Bundle is not strict JSON: {exc}"
        ) from exc
    if len(encoded) > MAX_BUNDLE_BYTES + 128:
        _fail("Validated recording-behavior bundle exceeds its metadata limit.")
    persisted = value.pop("record_sha256", None)
    if _digest(persisted, field="record_sha256") != canonical_json_sha256(value):
        _fail("Validated recording-behavior bundle digest is stale.")
    required = {
        "schema_id",
        "schema_version",
        "method_id",
        "status",
        "analysis_zarr",
        "recording_id",
        "projection_receipt",
        "source_bindings",
        "scientific_child_bindings",
        "capabilities",
        "compatibility_proofs",
        "validation_policy",
        "safety",
        "software_authority",
        "created_at_utc",
    }
    if set(value) != required:
        _fail("Validated recording-behavior bundle field set is inexact.")
    if (
        value.get("schema_id") != BUNDLE_SCHEMA_ID
        or value.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or value.get("method_id") != BUNDLE_METHOD_ID
        or value.get("status") != BUNDLE_STATUS
        or value.get("validation_policy") != VALIDATION_POLICY
        or value.get("safety") != SAFETY
    ):
        _fail("Bundle identity, validation policy, or safety contract is invalid.")
    archive = Path(_text(value.get("analysis_zarr"), field="analysis_zarr")).resolve()
    if str(archive) != value.get("analysis_zarr"):
        _fail("Bundle analysis_zarr path is not canonical and absolute.")
    recording_id = _text(value.get("recording_id"), field="recording_id")
    if (
        expected_analysis_zarr is not None
        and archive != Path(expected_analysis_zarr).expanduser().resolve()
    ):
        _fail("Bundle names another analysis archive.")
    if expected_recording_id is not None and recording_id != _text(
        expected_recording_id, field="expected_recording_id"
    ):
        _fail("Bundle names another recording.")
    projection = _mapping(value.get("projection_receipt"), field="projection_receipt")
    if set(projection) != {
        "receipt_path",
        "receipt_sha256",
        "schema_id",
        "schema_version",
    }:
        _fail("Bundle projection receipt binding is inexact.")
    projection_path = Path(
        _text(
            projection.get("receipt_path"),
            field="projection_receipt.receipt_path",
        )
    )
    if not projection_path.is_absolute() or str(projection_path) != str(
        projection_path.resolve()
    ):
        _fail("Bundle projection receipt path must be canonical and absolute.")
    _digest(
        projection.get("receipt_sha256"),
        field="projection_receipt.receipt_sha256",
    )
    if (
        projection.get("schema_id") != PROJECTION_RECEIPT_SCHEMA_ID
        or type(projection.get("schema_version")) is not int
        or projection.get("schema_version") < 1
    ):
        _fail("Bundle projection receipt schema identity is invalid.")
    source_bindings = _mapping(value.get("source_bindings"), field="source_bindings")
    if not set(BASE_SOURCE_BINDING_KEYS).issubset(source_bindings) or not set(
        source_bindings
    ).issubset(SOURCE_BINDING_KEYS):
        _fail("Bundle source-binding roster is inexact.")
    validated_source_bindings = {
        key: _validate_source_binding_record(source_bindings[key], binding_key=key)
        for key in source_bindings
    }
    scientific = _mapping(
        value.get("scientific_child_bindings"), field="scientific_child_bindings"
    )
    capabilities = _mapping(value.get("capabilities"), field="capabilities")
    if set(capabilities) != set(CAPABILITY_KEYS):
        _fail("Bundle capability roster is inexact.")
    validated_capabilities = {
        key: _validate_capability_record(capabilities[key], capability=key)
        for key in CAPABILITY_KEYS
    }
    for key, capability in validated_capabilities.items():
        if capability["state"] != "complete":
            continue
        scope = capability["binding_scope"]
        binding_key = capability["binding_key"]
        bindings = (
            validated_source_bindings if scope == "source_bindings" else scientific
        )
        if binding_key not in bindings:
            _fail(f"Complete capability {key!r} references an absent binding.")
    if not set(BASE_SCIENTIFIC_CHILD_KEYS).issubset(scientific):
        _fail("Bundle lacks a required base scientific-child binding.")
    allowed_scientific = set(BASE_SCIENTIFIC_CHILD_KEYS).union(
        OPTIONAL_SCIENTIFIC_CHILD_KEYS
    )
    if not set(scientific).issubset(allowed_scientific):
        _fail("Bundle contains an unexpected scientific-child binding.")
    for key, binding in scientific.items():
        _validate_child_binding_record(
            binding, field=f"scientific_child_bindings.{key}"
        )
    proofs = _mapping(value.get("compatibility_proofs"), field="compatibility_proofs")
    if not proofs:
        _fail("Bundle lacks compatibility proofs.")
    for name, raw in proofs.items():
        proof = _mapping(raw, field=f"compatibility_proofs.{name}")
        if set(proof) != {"status", "policy_id", "evidence", "evidence_sha256"}:
            _fail(f"Compatibility proof {name!r} field set is inexact.")
        evidence = _mapping(
            proof.get("evidence"), field=f"compatibility_proofs.{name}.evidence"
        )
        if (
            proof.get("status") != "proved"
            or _text(
                proof.get("policy_id"),
                field=f"compatibility_proofs.{name}.policy_id",
            )
            != proof.get("policy_id")
            or _digest(
                proof.get("evidence_sha256"),
                field=f"compatibility_proofs.{name}.evidence_sha256",
            )
            != canonical_json_sha256(_plain(evidence))
        ):
            _fail(f"Compatibility proof {name!r} is stale or invalid.")
    software = _mapping(value.get("software_authority"), field="software_authority")
    if (
        set(software) != {"repository", "commit"}
        or software.get("repository") != "palette"
        or _commit(software.get("commit")) != software.get("commit")
    ):
        _fail("Bundle software authority is invalid.")
    _text(value.get("created_at_utc"), field="created_at_utc")

    if validate_current_sources:
        dispositions = {
            key: {
                "state": capability["state"],
                "reason_code": capability["reason_code"],
                "detail": capability["detail"],
            }
            for key, capability in validated_capabilities.items()
            if capability["state"] != "complete"
        }
        current = _resolve_bundle_content(
            projection["receipt_path"],
            absent_capability_dispositions=dispositions,
            expected_analysis_zarr=archive,
            expected_recording_id=recording_id,
        )
        for field in (
            "analysis_zarr",
            "recording_id",
            "projection_receipt",
            "source_bindings",
            "scientific_child_bindings",
            "capabilities",
            "compatibility_proofs",
            "validation_policy",
            "safety",
        ):
            if _plain(value[field]) != _plain(current[field]):
                _fail(f"Current exact source composition changed at {field!r}.")
    return _freeze({**value, "record_sha256": persisted})


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedRecordingBehaviorBundleError(
            f"Cannot read strict bundle JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"Bundle JSON is not one object: {path}")
    return value


def read_validated_recording_behavior_bundle(
    path: str | Path, **expected: Any
) -> Mapping[str, Any]:
    return validate_validated_recording_behavior_bundle(
        _read_object(Path(path).expanduser().resolve()), **expected
    )


def ensure_validated_recording_behavior_bundle(
    projection_receipt_path: str | Path,
    *,
    absent_capability_dispositions: Mapping[str, Mapping[str, Any]],
    palette_commit: str,
    output_json: str | Path,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
) -> dict[str, Any]:
    """Create the external bundle or exactly validate and reuse its generation."""

    output = Path(output_json).expanduser().resolve()
    if output.exists():
        current = read_validated_recording_behavior_bundle(
            output,
            expected_analysis_zarr=expected_analysis_zarr,
            expected_recording_id=expected_recording_id,
        )
        requested_projection = _canonical_file(
            str(projection_receipt_path), field="projection_receipt_path"
        )
        if current["projection_receipt"]["receipt_path"] != str(requested_projection):
            _fail("Existing bundle belongs to another projection receipt.")
        if current["software_authority"]["commit"] != _commit(palette_commit):
            _fail("Existing bundle belongs to another Palette commit.")
        expected_dispositions = {
            key: {
                "state": value["state"],
                "reason_code": value["reason_code"],
                "detail": value["detail"],
            }
            for key, value in current["capabilities"].items()
            if value["state"] != "complete"
        }
        if _plain(expected_dispositions) != _plain(absent_capability_dispositions):
            _fail("Existing bundle has another absent-capability disposition.")
        return {**_plain(current), "bundle_path": str(output), "mode": "reused_exact"}
    bundle = build_validated_recording_behavior_bundle(
        projection_receipt_path,
        absent_capability_dispositions=absent_capability_dispositions,
        palette_commit=palette_commit,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
    )
    write_json_atomic(output, bundle, overwrite=False)
    return {**bundle, "bundle_path": str(output), "mode": "created"}


__all__ = [
    "BoundExactChaserExtensionSources",
    "BUNDLE_METHOD_ID",
    "BUNDLE_SCHEMA_ID",
    "BUNDLE_SCHEMA_VERSION",
    "CAPABILITY_KEYS",
    "CAPABILITY_STATES",
    "MAX_BUNDLE_BYTES",
    "REASON_CODES_BY_STATE",
    "ValidatedRecordingBehaviorBundleError",
    "bind_exact_chaser_extension_sources",
    "build_validated_recording_behavior_bundle",
    "ensure_validated_recording_behavior_bundle",
    "read_validated_recording_behavior_bundle",
    "validate_validated_recording_behavior_bundle",
]
