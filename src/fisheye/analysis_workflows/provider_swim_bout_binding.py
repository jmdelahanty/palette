"""Exact provider-motion to selector-ineligible swim-bout binding validation.

This boundary validates the one maintained provider-motion swim-bout branch.
It never resolves a selector and never reopens the provider's sealed upstream
position, body-frame, tracking, keypoint, or mask lineage.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.swim_bout_frame_axis import (
    build_frame_axis_contract,
    canonical_frame_axis_sha256,
)
from fisheye.analysis.track_kinematics_io import (
    TRACK_KINEMATICS_PUBLICATION_PROFILE_SELECTOR_INELIGIBLE_CANARY_V1,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.metadata import resolve_persisted_artifact_fps
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

PROVIDER_SWIM_BOUT_BINDING_SCHEMA_ID = (
    "palette.selector_ineligible_swim_bout_binding.v1"
)
PROVIDER_SWIM_BOUT_READ_AUTHORITY_SCHEMA_ID = (
    "palette.provider_track_motion_read_authority"
)
PROVIDER_SWIM_BOUT_READ_AUTHORITY_SCHEMA_VERSION = 1
PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_LEGACY_COMPATIBILITY_V1 = (
    "legacy_compatibility_v1"
)
PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1 = "current_strict_v1"
PROVIDER_SWIM_BOUT_VALIDATION_PROFILES = frozenset(
    {
        PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_LEGACY_COMPATIBILITY_V1,
        PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1,
    }
)
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ProviderSwimBoutBindingError(ValueError):
    """Raised when a swim-bout candidate does not bind one exact provider track."""


def _digest(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProviderSwimBoutBindingError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _whole_track_rows(provider: Any, *, track_id: int) -> slice:
    ids = np.asarray(provider.track_ids, dtype=np.int64).reshape(-1)
    offsets = np.asarray(provider.track_row_offsets, dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(ids == int(track_id))
    if matches.size != 1:
        raise ProviderSwimBoutBindingError(
            "Provider-motion track identity did not resolve exactly once."
        )
    if offsets.shape != (ids.size + 1,):
        raise ProviderSwimBoutBindingError(
            "Provider-motion track offsets do not bind the track axis."
        )
    index = int(matches[0])
    start, stop = int(offsets[index]), int(offsets[index + 1])
    frame_count = np.asarray(provider.source_acquisition_frame_index).reshape(-1).size
    if start < 0 or stop < start or stop > frame_count:
        raise ProviderSwimBoutBindingError(
            "Provider-motion track offsets exceed the verified frame domain."
        )
    return slice(start, stop)


def _normalize_rows(
    provider: Any,
    *,
    track_id: int,
    rows: slice | None,
) -> slice:
    expected = _whole_track_rows(provider, track_id=track_id)
    if rows is None:
        return expected
    if (
        type(rows) is not slice
        or rows.step not in (None, 1)
        or rows.start != expected.start
        or rows.stop != expected.stop
    ):
        raise ProviderSwimBoutBindingError(
            "Swim-bout binding must cover the whole provider track row slice."
        )
    return expected


def provider_swim_bout_content_sha256(tables: Any) -> str:
    """Hash the maintained normalized swim-bout payload for metadata-view parity."""

    arrays = {
        "bouts": tables.bouts,
        "peak_events": tables.peak_events,
        "inter_bout_intervals": tables.inter_bout_intervals,
        "inter_bout_interval_histogram": tables.inter_bout_interval_histogram,
        "global_metrics": tables.global_metrics,
        "trials": tables.trials,
        "bout_points": tables.bout_points,
        **{f"series/{name}": value for name, value in tables.series.items()},
    }
    return canonical_json_sha256(
        {
            "run_path": tables.run_path,
            "candidate_id": int(tables.candidate.candidate_id),
            "signal_id": int(tables.signal.signal_id),
            "lineage_hash": tables.run_attrs.get("lineage_hash"),
            "arrays": {
                name: array_values_sha256(np.asarray(value))
                for name, value in sorted(arrays.items())
            },
        }
    )


def validate_provider_swim_bout_binding(
    tables: Any,
    *,
    provider: Any,
    track_id: int,
    rows: slice | None = None,
    recording_root: Any | None = None,
    validation_profile: str = (
        PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_LEGACY_COMPATIBILITY_V1
    ),
) -> tuple[dict[str, Any], str, str]:
    """Validate and describe one exact provider-motion swim-bout dependency.

    The default profile preserves the original provider epoch-summary contract:
    it validates the exact source, track slice, manifest, provider verification,
    and frame content while accepting historical computation-v1 motion without
    authoritative recording timing.  New workflow execution must opt into the
    current strict profile, which requires the complete read-authority, timing,
    publication-profile, frame-contract, and FPS bindings.
    """

    if validation_profile not in PROVIDER_SWIM_BOUT_VALIDATION_PROFILES:
        raise ProviderSwimBoutBindingError(
            "validation_profile must name one supported provider swim-bout "
            f"contract; got {validation_profile!r}."
        )
    strict = (
        validation_profile == PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1
    )

    if isinstance(track_id, bool) or not isinstance(track_id, (int, np.integer)):
        raise ProviderSwimBoutBindingError("track_id must be one exact integer.")
    selected_track_id = int(track_id)
    selected_rows = _normalize_rows(
        provider,
        track_id=selected_track_id,
        rows=rows,
    )
    start, stop = int(selected_rows.start or 0), int(selected_rows.stop or 0)

    provider_run_name = str(provider.run_name)
    provider_run_path = str(provider.run_path)
    if strict and (
        _RUN_NAME_RE.fullmatch(provider_run_name) is None
        or provider_run_path
        != f"analysis/track_kinematics_runs/provider/{provider_run_name}"
    ):
        raise ProviderSwimBoutBindingError(
            "Provider-motion handle has a noncanonical exact run path."
        )
    manifest_sha256 = provider.provider_manifest_sha256
    verification_digest = provider.verification_digest
    physical_sha256: str | None = None
    if strict:
        manifest_sha256 = _digest(
            manifest_sha256,
            label="provider-motion manifest binding",
        )
        verification_digest = _digest(
            verification_digest,
            label="provider-motion verification binding",
        )
        physical_sha256 = _digest(
            provider.physical_authority_sha256,
            label="provider-motion physical authority binding",
        )
        if (
            provider.temporal_authority_status
            != "bound_live_recording_timing_authority"
            or provider.timing_is_authoritative is not True
        ):
            raise ProviderSwimBoutBindingError(
                "Provider-motion timing is not authoritative for swim-bout "
                "execution."
            )

    attrs = dict(tables.run_attrs)
    if attrs.get("source_track_kinematics_scope") != "provider":
        raise ProviderSwimBoutBindingError(
            "Swim-bout candidate is not bound to provider motion."
        )
    if attrs.get("source_track_kinematics_run") != provider_run_name:
        raise ProviderSwimBoutBindingError(
            "Swim-bout and provider-motion run identities disagree."
        )
    if strict and (
        attrs.get("source_track_kinematics_publication_profile_id")
        != TRACK_KINEMATICS_PUBLICATION_PROFILE_SELECTOR_INELIGIBLE_CANARY_V1
    ):
        raise ProviderSwimBoutBindingError(
            "Swim-bout provider-motion publication profile is invalid."
        )
    try:
        observed_track_id = int(attrs.get("track_id", -1))
    except (TypeError, ValueError) as exc:
        raise ProviderSwimBoutBindingError(
            "Swim-bout track identity is malformed."
        ) from exc
    if observed_track_id != selected_track_id:
        raise ProviderSwimBoutBindingError(
            "Swim-bout and provider-motion track identities disagree."
        )
    if attrs.get("source_track_motion_manifest_sha256") != manifest_sha256:
        raise ProviderSwimBoutBindingError(
            "Swim-bout provider-motion manifest binding is stale."
        )

    authority = attrs.get("source_track_motion_authority")
    if not isinstance(authority, Mapping):
        raise ProviderSwimBoutBindingError(
            "Swim-bout provider read authority is absent."
        )
    authority_root = f"/{provider_run_path}"
    compatibility_authority = {
        "track_id": selected_track_id,
        "track_row_start": start,
        "track_row_stop": stop,
        "motion_manifest_sha256": manifest_sha256,
        "provider_verification_digest": verification_digest,
    }
    if strict:
        expected_authority = {
            "schema_id": PROVIDER_SWIM_BOUT_READ_AUTHORITY_SCHEMA_ID,
            "schema_version": PROVIDER_SWIM_BOUT_READ_AUTHORITY_SCHEMA_VERSION,
            "run_ref": authority_root,
            "track_ref": authority_root,
            **compatibility_authority,
            "motion_manifest_ref": (f"{authority_root}@provider_track_motion_manifest"),
            "positions_px_ref": f"{authority_root}/positions_px",
            "positions_mm_ref": f"{authority_root}/positions_mm",
            "track_sample_key_ref": f"{authority_root}/track_sample_key",
            "source_acquisition_frame_index_ref": (
                f"{authority_root}/source_acquisition_frame_index"
            ),
            "physical_authority_sha256": physical_sha256,
            "temporal_authority_status": ("bound_live_recording_timing_authority"),
            "timing_is_authoritative": True,
        }
        if dict(authority) != expected_authority:
            differing = sorted(
                key
                for key in set(authority).union(expected_authority)
                if authority.get(key) != expected_authority.get(key)
            )
            label = differing[0] if differing else "field set"
            raise ProviderSwimBoutBindingError(
                f"Swim-bout provider read authority differs at {label!r}."
            )
    else:
        for key, expected in compatibility_authority.items():
            if authority.get(key) != expected:
                raise ProviderSwimBoutBindingError(
                    f"Swim-bout provider read authority differs at {key!r}."
                )

    frames = np.asarray(
        provider.source_acquisition_frame_index[selected_rows],
        dtype=np.int64,
    )
    frame_contract = attrs.get("frame_axis_contract")
    if not isinstance(frame_contract, Mapping):
        raise ProviderSwimBoutBindingError("Swim-bout frame-axis contract is absent.")
    frame_sha256 = canonical_frame_axis_sha256(frames)
    if not strict:
        if frame_contract.get("content_sha256") != frame_sha256:
            raise ProviderSwimBoutBindingError(
                "Swim-bout frame axis differs from the selected provider track."
            )
    else:
        try:
            expected_frame_contract = build_frame_axis_contract(
                frames,
                authoritative_path=(
                    f"{provider_run_path}/source_acquisition_frame_index"
                ),
                source_track_kinematics_run=provider_run_name,
                track_id=selected_track_id,
                source_track_motion_manifest_sha256=manifest_sha256,
                storage_mode=str(frame_contract.get("storage_mode")),
                authoritative_dtype=frame_contract.get("authoritative_dtype", "int64"),
            )
        except (TypeError, ValueError) as exc:
            raise ProviderSwimBoutBindingError(
                f"Swim-bout frame-axis contract is invalid: {exc}"
            ) from exc
        if dict(frame_contract) != expected_frame_contract:
            raise ProviderSwimBoutBindingError(
                "Swim-bout frame axis differs from the selected provider track."
            )
        frame_contract_sha256 = canonical_json_sha256(dict(frame_contract))
        if attrs.get("frame_axis_contract_sha256") != frame_contract_sha256:
            raise ProviderSwimBoutBindingError(
                "Swim-bout frame-axis contract digest is stale."
            )

        if recording_root is None:
            raise ProviderSwimBoutBindingError(
                "Current strict swim-bout validation requires the recording root."
            )
        temporal_authority = provider.temporal_authority_record
        provider_fps = (
            temporal_authority.get("nominal_fps")
            if isinstance(temporal_authority, Mapping)
            else None
        )
        try:
            bout_fps = resolve_persisted_artifact_fps(
                recording_root,
                attrs,
                artifact_name=str(tables.run_path),
            )
        except ValueError as exc:
            raise ProviderSwimBoutBindingError(
                f"Swim-bout FPS binding is invalid: {exc}"
            ) from exc
        if (
            isinstance(provider_fps, bool)
            or isinstance(bout_fps, bool)
            or not isinstance(provider_fps, (int, float))
            or not isinstance(bout_fps, (int, float))
            or not math.isclose(
                float(provider_fps),
                float(bout_fps),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ProviderSwimBoutBindingError(
                "Swim-bout and provider-motion FPS disagree."
            )

    run_name = str(tables.run_name)
    if strict and (
        _RUN_NAME_RE.fullmatch(run_name) is None
        or tables.run_path != f"analysis/swim_bout_runs/{run_name}"
    ):
        raise ProviderSwimBoutBindingError(
            "Swim-bout tables have a noncanonical exact run path."
        )
    raw_lineage_hash = attrs.get("lineage_hash")
    if strict:
        lineage_hash = _digest(raw_lineage_hash, label="swim-bout lineage")
    elif type(raw_lineage_hash) is not str or len(raw_lineage_hash) != 64:
        raise ProviderSwimBoutBindingError(
            "Swim-bout candidate lacks its exact lineage digest."
        )
    else:
        lineage_hash = raw_lineage_hash
    binding = {
        "schema_id": PROVIDER_SWIM_BOUT_BINDING_SCHEMA_ID,
        "run_name": run_name,
        "run_path": tables.run_path,
        "lineage_hash": lineage_hash,
        "frame_axis_sha256": frame_sha256,
        "source_track_motion_manifest_sha256": manifest_sha256,
        "source_track_motion_verification_digest": verification_digest,
        "track_id": selected_track_id,
        "track_row_start": start,
        "track_row_stop": stop,
        "default_candidate_id": int(tables.candidate.candidate_id),
        "default_signal_id": int(tables.signal.signal_id),
        "default_signal_level": str(tables.signal.speed_level),
    }
    binding["sha256"] = canonical_json_sha256(binding)
    return binding, lineage_hash, frame_sha256


__all__ = [
    "PROVIDER_SWIM_BOUT_BINDING_SCHEMA_ID",
    "PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1",
    "PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_LEGACY_COMPATIBILITY_V1",
    "PROVIDER_SWIM_BOUT_VALIDATION_PROFILES",
    "ProviderSwimBoutBindingError",
    "provider_swim_bout_content_sha256",
    "validate_provider_swim_bout_binding",
]
