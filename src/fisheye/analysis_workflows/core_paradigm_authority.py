"""Shared core-roster lineage projection for paradigm descendants.

This module does not select an authority. It projects the already validated
core binding carried by one chaser-relative publication into a small sealed
dependency record that every descendant can retain and compare.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from fisheye.analytics_exports.kinematics_samples import (
    CORE_MOTION_SOURCE_SURFACE_PROFILE_ID,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CROSS_GRAIN_JOIN_AUTHORITY,
    KINEMATICS_SAMPLES_CAPABILITY,
    SUBJECT_BODY_FRAME_CAPABILITY,
    SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

CORE_PARADIGM_DEPENDENCY_SCHEMA_ID = (
    "palette.core_behavior.paradigm_relative_frame_dependency"
)
CORE_PARADIGM_DEPENDENCY_SCHEMA_VERSION = 1
_DEPENDENCY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "core_authority_roster_sha256",
        "core_authority_consumption_receipt_sha256",
        "selected_track_id",
        "source_relative_frame_run_path",
        "source_relative_frame_manifest_sha256",
        "source_core_authority_binding_sha256",
        "core_motion_source_binding_sha256",
        "core_subject_body_frame_source_binding_sha256",
        "fallback",
        "record_sha256",
    }
)
_CORE_BINDING_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "core_authority_roster_sha256",
        "core_authority_consumption_receipt",
        "core_motion",
        "core_subject_body_frame",
        "chaser_source",
        "fish_pixel_projection",
        "core_motion_facts_repeated",
        "fallback",
    }
)
_CORE_MOTION_FIELDS = frozenset(
    {
        "run_path",
        "source_manifest_sha256",
        "source_binding_sha256",
        "track_id",
        "row_axis_sha256",
    }
)
_CORE_BODY_FRAME_FIELDS = frozenset(
    {
        "run_path",
        "publication_manifest_sha256",
        "source_binding_sha256",
        "row_identity_sha256",
        "body_frame_record_sha256",
        "projection_record_sha256",
    }
)
_CORE_CONSUMPTION_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "consumer_id",
        "recording_id",
        "analysis_zarr",
        "core_authority_roster_sha256",
        "required_capabilities",
        "capability_binding_digests",
        "selected_track_id",
        "record_sha256",
    }
)
_CAPABILITY_DIGEST_FIELDS = frozenset(
    {
        "profile_id",
        "source_binding_sha256",
        "projection_contract_sha256",
        "join_authority_sha256",
    }
)
_CHASER_SOURCE_FIELDS = frozenset(
    {
        "run_path",
        "manifest_sha256",
        "verification_digest",
        "consumed_authority",
        "fish_position_authority",
        "body_frame_authority",
    }
)
_FISH_PIXEL_PROJECTION_FIELDS = frozenset(
    {"source", "formula", "physical_authority_sha256"}
)
_CORE_ANALYSIS_PROFILE_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "profile_id",
        "core_authority_roster_sha256",
        "source_chaser_profile_sha256",
        "body_frame",
    }
)
_REQUIRED_CAPABILITIES = frozenset(
    {
        CROSS_GRAIN_JOIN_AUTHORITY,
        KINEMATICS_SAMPLES_CAPABILITY,
        SUBJECT_BODY_FRAME_CAPABILITY,
    }
)
_CAPABILITY_PROFILES = {
    CROSS_GRAIN_JOIN_AUTHORITY: "cross_grain_join_authority_v1",
    KINEMATICS_SAMPLES_CAPABILITY: CORE_MOTION_SOURCE_SURFACE_PROFILE_ID,
    SUBJECT_BODY_FRAME_CAPABILITY: SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID,
}


class CoreParadigmAuthorityError(ValueError):
    """A paradigm source does not retain one exact selected core roster."""


def _fail(message: str) -> None:
    raise CoreParadigmAuthorityError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one mapping.")
    return value


def validate_core_paradigm_dependency(value: object) -> Mapping[str, Any]:
    """Validate one sealed downstream roster identity without source reads."""

    record = _plain(_mapping(value, field="core paradigm dependency"))
    if set(record) != _DEPENDENCY_FIELDS:
        _fail("Core paradigm dependency field set is not exact.")
    digest = _digest(
        record.get("record_sha256"),
        field="core paradigm dependency digest",
    )
    body = {key: item for key, item in record.items() if key != "record_sha256"}
    if canonical_json_sha256(body) != digest:
        _fail("Core paradigm dependency digest is stale.")
    if (
        record.get("schema_id") != CORE_PARADIGM_DEPENDENCY_SCHEMA_ID
        or record.get("schema_version") != CORE_PARADIGM_DEPENDENCY_SCHEMA_VERSION
        or record.get("fallback") != "prohibited"
    ):
        _fail("Core paradigm dependency schema or fallback policy is invalid.")
    for field_name in (
        "recording_id",
        "source_relative_frame_run_path",
    ):
        if type(record.get(field_name)) is not str or not record[field_name]:
            _fail(f"Core paradigm dependency {field_name!r} is invalid.")
    for field_name in (
        "core_authority_roster_sha256",
        "core_authority_consumption_receipt_sha256",
        "source_relative_frame_manifest_sha256",
        "source_core_authority_binding_sha256",
        "core_motion_source_binding_sha256",
        "core_subject_body_frame_source_binding_sha256",
    ):
        _digest(record.get(field_name), field=f"core paradigm {field_name}")
    if (
        type(record.get("selected_track_id")) is not int
        or record["selected_track_id"] < 0
    ):
        _fail("Core paradigm selected track identity is invalid.")
    return MappingProxyType(record)


def validate_core_paradigm_source_dependency(
    value: object | None,
    *,
    recording_id: str,
    source_relative_frame_run_path: str,
    source_relative_frame_manifest_sha256: str,
) -> Mapping[str, Any] | None:
    """Validate an optional dependency against the exact relative-frame source."""

    if value is None:
        return None
    dependency = validate_core_paradigm_dependency(value)
    if (
        dependency["recording_id"] != recording_id
        or dependency["source_relative_frame_run_path"]
        != source_relative_frame_run_path
        or dependency["source_relative_frame_manifest_sha256"]
        != source_relative_frame_manifest_sha256
    ):
        _fail("Core paradigm dependency names another relative-frame source.")
    return dependency


def core_paradigm_dependency_from_relative_frame(
    value: object,
    *,
    required: bool = False,
) -> Mapping[str, Any] | None:
    """Project an existing relative-frame core binding into descendant lineage."""

    if type(required) is not bool:
        raise TypeError("required must be the exact boolean.")
    from .chaser_relative_frame_source_handle import ChaserRelativeFrameSourceHandle
    from .chaser_relative_frame_validation_receipt import (
        ChaserRelativeFrameTargetedSourceHandle,
    )

    if type(value) not in {
        ChaserRelativeFrameSourceHandle,
        ChaserRelativeFrameTargetedSourceHandle,
    }:
        raise TypeError("value must be one strict chaser-relative source handle.")
    handle = value
    handle.assert_current()
    context = (
        handle.context
        if type(handle) is ChaserRelativeFrameSourceHandle
        else handle.run_manifest.get("context")
    )
    context = _mapping(context, field="relative-frame manifest context")
    raw_envelope = context.get("core_authority")
    if raw_envelope is None:
        if required:
            _fail("Chaser-relative source has no selected core-authority binding.")
        return None
    envelope = _plain(_mapping(raw_envelope, field="relative core-authority envelope"))
    if set(envelope) != {"record", "sha256"}:
        _fail("Relative core-authority envelope field set is not exact.")
    binding = _plain(
        _mapping(envelope.get("record"), field="relative core-authority binding")
    )
    binding_sha256 = _digest(
        envelope.get("sha256"),
        field="relative core-authority binding digest",
    )
    if canonical_json_sha256(binding) != binding_sha256:
        _fail("Relative core-authority binding digest is stale.")
    if (
        set(binding) != _CORE_BINDING_FIELDS
        or binding.get("schema_id")
        != "palette.chaser_relative_frame.core_authority_binding"
        or binding.get("schema_version") != 1
        or binding.get("recording_id") != handle.recording_id
        or binding.get("fallback") != "prohibited"
        or binding.get("core_motion_facts_repeated") is not False
    ):
        _fail("Relative core-authority binding contract is invalid.")
    roster_sha256 = _digest(
        binding.get("core_authority_roster_sha256"),
        field="selected core-authority roster digest",
    )
    receipt = _plain(
        _mapping(
            binding.get("core_authority_consumption_receipt"),
            field="relative core-authority consumption receipt",
        )
    )
    receipt_body = {
        key: item for key, item in receipt.items() if key != "record_sha256"
    }
    receipt_sha256 = _digest(
        receipt.get("record_sha256"),
        field="relative core-authority consumption receipt digest",
    )
    required_capabilities = receipt.get("required_capabilities")
    capability_bindings = receipt.get("capability_binding_digests")
    if (
        set(receipt) != _CORE_CONSUMPTION_FIELDS
        or receipt.get("schema_id")
        != "palette.core_behavior.authority_consumption_receipt"
        or receipt.get("schema_version") != 1
        or receipt.get("consumer_id") != "palette.chaser.core_relative_frame.v1"
        or canonical_json_sha256(receipt_body) != receipt_sha256
        or receipt.get("recording_id") != handle.recording_id
        or receipt.get("core_authority_roster_sha256") != roster_sha256
        or type(receipt.get("analysis_zarr")) is not str
        or not receipt["analysis_zarr"]
        or not isinstance(required_capabilities, list)
        or tuple(required_capabilities) != tuple(sorted(_REQUIRED_CAPABILITIES))
        or not isinstance(capability_bindings, Mapping)
        or set(capability_bindings) != _REQUIRED_CAPABILITIES
    ):
        _fail("Relative core-authority consumption receipt is incomplete or stale.")
    if (
        Path(receipt["analysis_zarr"]).expanduser().resolve()
        != Path(handle.analysis_zarr_path).expanduser().resolve()
    ):
        _fail("Relative core-authority receipt belongs to another analysis Zarr.")
    for capability, raw_digest_record in capability_bindings.items():
        digest_record = _plain(
            _mapping(
                raw_digest_record,
                field=f"core capability digest {capability!r}",
            )
        )
        if set(digest_record) != _CAPABILITY_DIGEST_FIELDS:
            _fail("Core capability-binding digest field set is not exact.")
        capability_profile = _text(
            digest_record.get("profile_id"),
            field=f"core capability profile {capability!r}",
        )
        if capability_profile != _CAPABILITY_PROFILES[capability]:
            _fail(f"Core capability {capability!r} names another source profile.")
        for field_name in ("source_binding_sha256", "join_authority_sha256"):
            _digest(
                digest_record.get(field_name),
                field=f"core capability {capability!r} {field_name}",
            )
        projection_digest = digest_record.get("projection_contract_sha256")
        if capability == "cross_grain_join_authority":
            if projection_digest is not None:
                _fail("Cross-grain join capability cannot carry a projection digest.")
        else:
            _digest(
                projection_digest,
                field=f"core capability {capability!r} projection digest",
            )
    join_binding = capability_bindings[CROSS_GRAIN_JOIN_AUTHORITY]
    join_sha256 = join_binding["source_binding_sha256"]
    if (
        join_binding["join_authority_sha256"] != join_sha256
        or capability_bindings[KINEMATICS_SAMPLES_CAPABILITY]["join_authority_sha256"]
        != join_sha256
        or capability_bindings[SUBJECT_BODY_FRAME_CAPABILITY]["join_authority_sha256"]
        != join_sha256
    ):
        _fail("Core capability bindings do not share one cross-grain join authority.")
    motion = _plain(_mapping(binding.get("core_motion"), field="core motion binding"))
    body_frame = _plain(
        _mapping(
            binding.get("core_subject_body_frame"),
            field="core subject body-frame binding",
        )
    )
    track_id = receipt.get("selected_track_id")
    if (
        set(motion) != _CORE_MOTION_FIELDS
        or set(body_frame) != _CORE_BODY_FRAME_FIELDS
        or type(track_id) is not int
        or track_id < 0
        or motion.get("track_id") != track_id
    ):
        _fail("Relative core motion and consumption receipt track identities differ.")
    for field_name in ("run_path",):
        _text(motion.get(field_name), field=f"core motion {field_name}")
        _text(body_frame.get(field_name), field=f"core body frame {field_name}")
    for field_name in (
        "source_manifest_sha256",
        "source_binding_sha256",
        "row_axis_sha256",
    ):
        _digest(motion.get(field_name), field=f"core motion {field_name}")
    for field_name in (
        "publication_manifest_sha256",
        "source_binding_sha256",
        "row_identity_sha256",
        "body_frame_record_sha256",
        "projection_record_sha256",
    ):
        _digest(body_frame.get(field_name), field=f"core body frame {field_name}")
    motion_binding_sha256 = _digest(
        motion.get("source_binding_sha256"),
        field="core motion source-binding digest",
    )
    body_binding_sha256 = _digest(
        body_frame.get("source_binding_sha256"),
        field="core body-frame source-binding digest",
    )
    if (
        capability_bindings[KINEMATICS_SAMPLES_CAPABILITY]["source_binding_sha256"]
        != motion_binding_sha256
        or capability_bindings[SUBJECT_BODY_FRAME_CAPABILITY]["source_binding_sha256"]
        != body_binding_sha256
    ):
        _fail("Relative core source bindings differ from the consumption receipt.")
    chaser_source = _plain(
        _mapping(binding.get("chaser_source"), field="chaser source binding")
    )
    if (
        set(chaser_source) != _CHASER_SOURCE_FIELDS
        or chaser_source.get("consumed_authority") != "chaser_position"
        or chaser_source.get("fish_position_authority")
        != "not_used_core_roster_selected_instead"
        or chaser_source.get("body_frame_authority")
        != "not_used_core_roster_selected_instead"
    ):
        _fail("Relative chaser-source binding contract is invalid.")
    _text(chaser_source.get("run_path"), field="relative chaser source path")
    _digest(chaser_source.get("manifest_sha256"), field="chaser source manifest")
    _digest(
        chaser_source.get("verification_digest"),
        field="chaser source verification digest",
    )
    fish_projection = _plain(
        _mapping(binding.get("fish_pixel_projection"), field="fish pixel projection")
    )
    if (
        set(fish_projection) != _FISH_PIXEL_PROJECTION_FIELDS
        or fish_projection.get("source") != "core_positions_mm"
        or fish_projection.get("formula") != "positions_mm * pixels_per_mm"
    ):
        _fail("Core fish pixel-projection contract is invalid.")
    _digest(
        fish_projection.get("physical_authority_sha256"),
        field="fish pixel physical-authority digest",
    )
    profile_envelope = _plain(
        _mapping(context.get("analysis_profile"), field="analysis profile")
    )
    if set(profile_envelope) != {"record", "sha256"}:
        _fail("Relative analysis-profile envelope field set is not exact.")
    profile = _plain(
        _mapping(profile_envelope.get("record"), field="analysis profile record")
    )
    profile_sha256 = _digest(
        profile_envelope.get("sha256"), field="relative analysis-profile digest"
    )
    if (
        canonical_json_sha256(profile) != profile_sha256
        or set(profile) != _CORE_ANALYSIS_PROFILE_FIELDS
        or profile.get("schema_id")
        != "palette.chaser_relative_frame.core_analysis_profile"
        or profile.get("schema_version") != 1
        or profile.get("recording_id") != handle.recording_id
        or profile.get("profile_id") != "core_roster_chaser_relative_frame_v1"
        or profile.get("core_authority_roster_sha256") != roster_sha256
    ):
        _fail("Relative analysis profile names another core-authority roster.")
    _digest(
        profile.get("source_chaser_profile_sha256"),
        field="source chaser analysis-profile digest",
    )
    if profile.get("body_frame") != "core_roster_selected_subject_body_frame":
        _fail("Relative analysis profile does not select the core body frame.")
    dependency_body = {
        "schema_id": CORE_PARADIGM_DEPENDENCY_SCHEMA_ID,
        "schema_version": CORE_PARADIGM_DEPENDENCY_SCHEMA_VERSION,
        "recording_id": handle.recording_id,
        "core_authority_roster_sha256": roster_sha256,
        "core_authority_consumption_receipt_sha256": receipt_sha256,
        "selected_track_id": track_id,
        "source_relative_frame_run_path": handle.run_path,
        "source_relative_frame_manifest_sha256": handle.manifest_sha256,
        "source_core_authority_binding_sha256": binding_sha256,
        "core_motion_source_binding_sha256": motion_binding_sha256,
        "core_subject_body_frame_source_binding_sha256": body_binding_sha256,
        "fallback": "prohibited",
    }
    return validate_core_paradigm_dependency(
        {
            **dependency_body,
            "record_sha256": canonical_json_sha256(dependency_body),
        }
    )


__all__ = [
    "CORE_PARADIGM_DEPENDENCY_SCHEMA_ID",
    "CORE_PARADIGM_DEPENDENCY_SCHEMA_VERSION",
    "CoreParadigmAuthorityError",
    "core_paradigm_dependency_from_relative_frame",
    "validate_core_paradigm_dependency",
    "validate_core_paradigm_source_dependency",
]
