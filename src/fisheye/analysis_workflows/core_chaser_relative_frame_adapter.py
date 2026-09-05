"""Rebind chaser position facts to one selected core-motion authority.

The input chaser-relative publication is used only as a strict carrier of
chaser position, identity, occurrence, trial, and controller evidence.  Its
historical fish-position and body-frame authorities are deliberately ignored.
The adapter feeds the existing relative-frame computation, schema, and atomic
publisher; it does not introduce another scientific run or export surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CROSS_GRAIN_JOIN_AUTHORITY,
    KINEMATICS_SAMPLES_CAPABILITY,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .chaser_relative_frame import (
    AcquisitionFrameKeys,
    ChaserObservations,
    ChaserRelativeFrameInput,
    CoordinatePolicy,
    ProviderSourceAuthority,
    ScalePolicy,
    TimingPolicy,
    compute_chaser_relative_frame,
)
from .chaser_relative_frame_source_handle import (
    ChaserRelativeFrameSourceHandle,
    require_chaser_relative_frame_source_handle,
)
from .chaser_relative_frame_storage import (
    ChaserRelativeFramePublicationContext,
    PreparedChaserRelativeFrame,
    prepare_chaser_relative_frame,
)
from .core_motion_source_handle import (
    CoreMotionTrackSourceHandle,
    require_core_motion_track_source_handle,
)
from .provider_recording_timing_authority import (
    ProviderRecordingTimingAuthorityError,
    load_provider_recording_timing_authority,
)

CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_ID = "palette.chaser.core_relative_frame_adapter"
CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_VERSION = 1
CORE_CHASER_RELATIVE_CONSUMER_ID = "palette.chaser.core_relative_frame.v1"
CORE_CHASER_RELATIVE_PROFILE_ID = "core_roster_chaser_relative_frame_v1"
CORE_CHASER_ROW_AXIS_POLICY_ID = "core_track_to_chaser_frame_exact_join_v1"
CORE_CHASER_TEMPORAL_SELECTION_ID = "core_track_chaser_frame_intersection_v1"


class CoreChaserRelativeFrameAdapterError(ValueError):
    """The selected core track cannot safely replace historical fish authority."""


def _fail(message: str) -> None:
    raise CoreChaserRelativeFrameAdapterError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be one exact object.")
    return value


def _context_record(
    context: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    value = context.get(name)
    if value is None:
        if required:
            _fail(f"Chaser source context lacks {name!r}.")
        return None
    envelope = _mapping(value, label=f"chaser context {name}")
    if set(envelope) != {"record", "sha256"}:
        _fail(f"Chaser context {name!r} is not a record-plus-digest envelope.")
    record = _plain(_mapping(envelope.get("record"), label=f"{name} record"))
    if canonical_json_sha256(record) != envelope.get("sha256"):
        _fail(f"Chaser context {name!r} digest is stale.")
    return record


def _same_across_chasers(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 2:
        _fail(f"{label} must have frame and chaser axes.")
    if array.shape[0] and not np.all(array == array[:, :1]):
        _fail(f"{label} differs across chasers for one acquisition frame.")
    return np.array(array[:, 0], copy=True)


def _source_authority(
    record: Mapping[str, Any],
    *,
    recording_id: str,
    coordinate_authority_id: str,
    scale_authority_id: str,
    timing_authority_id: str,
    row_axis_authority_id: str,
    row_axis_authority_digest: str,
) -> ProviderSourceAuthority:
    if record.get("recording_id") != recording_id:
        _fail("Chaser position authority belongs to another recording.")
    expected = {
        "coordinate_authority_id": coordinate_authority_id,
        "scale_authority_id": scale_authority_id,
        "timing_authority_id": timing_authority_id,
    }
    for field_name, expected_value in expected.items():
        if record.get(field_name) != expected_value:
            _fail(f"Chaser position authority differs at {field_name!r}.")
    return ProviderSourceAuthority(
        recording_id=recording_id,
        source_authority_id=str(record["source_authority_id"]),
        source_digest=str(record["source_digest"]),
        provider_id=str(record["provider_id"]),
        provider_digest=str(record["provider_digest"]),
        coordinate_authority_id=coordinate_authority_id,
        scale_authority_id=scale_authority_id,
        timing_authority_id=timing_authority_id,
        row_axis_authority_id=row_axis_authority_id,
        row_axis_authority_digest=row_axis_authority_digest,
    )


@dataclass(frozen=True)
class PreparedCoreChaserRelativeFrame:
    """Existing-schema relative-frame payload plus its core authority evidence."""

    prepared: PreparedChaserRelativeFrame
    core_authority_record: Mapping[str, Any]
    core_authority_record_sha256: str
    source_chaser_run_path: str
    source_chaser_manifest_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_ID,
            "schema_version": CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_VERSION,
            "core_authority_record_sha256": self.core_authority_record_sha256,
            "source_chaser_run_path": self.source_chaser_run_path,
            "source_chaser_manifest_sha256": self.source_chaser_manifest_sha256,
            "prepared_manifest_sha256": self.prepared.payload_digest,
            "publication_surface": "analysis/chaser_relative_frame_runs",
            "selector_eligible": False,
            "selection": "none",
        }


def prepare_core_chaser_relative_frame(
    core_motion: CoreMotionTrackSourceHandle,
    chaser_source: ChaserRelativeFrameSourceHandle,
) -> PreparedCoreChaserRelativeFrame:
    """Prepare a core-bound candidate for the existing relative-frame publisher."""

    core = require_core_motion_track_source_handle(core_motion)
    chaser = require_chaser_relative_frame_source_handle(chaser_source)
    if chaser.selector_eligible is not False:
        _fail("Core rebinding requires one exact selector-ineligible chaser source.")
    if core.analysis_zarr_path != chaser.analysis_zarr_path:
        _fail("Core and chaser sources do not belong to one analysis Zarr.")
    if core.recording_id != chaser.recording_id:
        _fail("Core and chaser sources name different recordings.")

    receipt = _mapping(core.consumption_receipt, label="core consumption receipt")
    if (
        receipt.get("consumer_id") != CORE_CHASER_RELATIVE_CONSUMER_ID
        or receipt.get("selected_track_id") != core.track_id
        or set(receipt.get("required_capabilities", ()))
        != {CROSS_GRAIN_JOIN_AUTHORITY, KINEMATICS_SAMPLES_CAPABILITY}
    ):
        _fail("Core track was not admitted for this exact relative-frame consumer.")
    roster = _mapping(core.core_authority_roster, label="core authority roster")
    join = _mapping(
        roster.get("cross_grain_join_authority"),
        label="core cross-grain join authority",
    )
    manifest = _mapping(chaser.run_manifest, label="chaser source manifest")
    coordinate = _mapping(
        manifest.get("coordinate_policy"), label="chaser coordinate policy"
    )
    scale = _mapping(manifest.get("scale_policy"), label="chaser scale policy")
    timing_policy_source = _mapping(
        manifest.get("timing_policy"), label="chaser timing policy"
    )
    coordinate_id = str(coordinate.get("coordinate_authority_id"))
    scale_id = str(scale.get("scale_authority_id"))
    timing_id = str(timing_policy_source.get("timing_authority_id"))
    if (
        coordinate.get("coordinate_frame") != "source_camera_continuous_pixel_xy"
        or coordinate.get("origin") != "top_left"
        or coordinate.get("x_axis_direction") != "right"
        or coordinate.get("y_axis_direction") != "down"
        or coordinate_id != join.get("acquisition_camera_frame_ref")
    ):
        _fail("Core and chaser sources do not share one source-camera authority.")
    pixels_per_mm = scale.get("pixels_per_unit")
    if (
        scale.get("unit") != "mm"
        or isinstance(pixels_per_mm, bool)
        or not isinstance(pixels_per_mm, (int, float))
        or not math.isfinite(float(pixels_per_mm))
        or float(pixels_per_mm) <= 0
    ):
        _fail("Chaser source lacks one valid pixels-per-millimeter authority.")
    position_surface = _mapping(
        core.selected_surfaces.get("positions_mm"),
        label="core physical-position surface",
    )
    if position_surface.get("physical_authority_sha256") != scale.get("scale_digest"):
        _fail("Core and chaser positions bind different physical-scale authority.")

    source_context = _mapping(chaser.context, label="chaser source context")
    controller = _context_record(source_context, "controller_state")
    assert controller is not None
    session_timing = _mapping(
        controller.get("session_timestamp_authority"),
        label="chaser session timestamp authority",
    )
    try:
        timing = load_provider_recording_timing_authority(
            core.analysis_zarr_path,
            required=True,
            use_consolidated=True,
            expected_sha256=session_timing.get("recording_timing_authority_sha256"),
        )
    except (ProviderRecordingTimingAuthorityError, TypeError, ValueError) as exc:
        raise CoreChaserRelativeFrameAdapterError(
            f"Chaser timing cannot be bound to the core frame domain: {exc}"
        ) from exc
    if timing is None:  # pragma: no cover - required=True contract
        _fail("Chaser timing authority is absent.")
    if (
        timing.recording_id != core.recording_id
        or timing.camera_id != join.get("camera_id")
        or timing.frame_count != join.get("source_total_frames")
        or timing.source_video_metadata_sha256
        != join.get("source_video_metadata_sha256")
    ):
        _fail("Core and chaser sources bind different acquisition timing authority.")

    frames = _same_across_chasers(
        chaser.base_frame_chaser("acquisition_frame_id"),
        label="chaser acquisition frame",
    ).astype(np.int64, copy=False)
    core_frames = np.asarray(core.frame_indices, dtype=np.int64)
    if core_frames.shape != (core.sample_count,) or (
        core_frames.size > 1 and np.any(np.diff(core_frames) <= 0)
    ):
        _fail("Core track frames are not one unique increasing identity axis.")
    positions = np.searchsorted(core_frames, frames)
    if np.any(positions >= core_frames.size) or not np.array_equal(
        core_frames[positions], frames
    ):
        _fail("Every chaser frame must resolve one exact selected core-track row.")
    core_rows = positions.astype(np.int64, copy=False)

    core_positions_mm = np.asarray(core.positions_mm, dtype=np.float64)
    core_sample_valid = np.asarray(core.array("sample_valid"), dtype=bool)
    core_position_finite = np.asarray(core.array("position_finite"), dtype=bool)
    if (
        core_positions_mm.shape != (core.sample_count, 2)
        or core_sample_valid.shape != (core.sample_count,)
        or core_position_finite.shape != (core.sample_count,)
    ):
        _fail("Core position arrays do not share the selected track row axis.")
    fish_mm = core_positions_mm[core_rows]
    fish_valid = (
        core_sample_valid[core_rows]
        & core_position_finite[core_rows]
        & np.isfinite(fish_mm).all(axis=1)
    )
    fish_px = fish_mm * float(pixels_per_mm)
    fish_px = np.array(fish_px, copy=True)
    fish_px[~fish_valid] = np.nan

    timestamp_ns = _same_across_chasers(
        chaser.base_frame_chaser("timestamp_ns"),
        label="chaser session timestamp",
    ).astype(np.int64, copy=False)
    timestamp_valid = _same_across_chasers(
        chaser.base_frame_chaser("timestamp_valid"),
        label="chaser timestamp validity",
    ).astype(bool, copy=False)
    if not np.all(timestamp_valid):
        _fail("Core-bound chaser frames require exact timestamps for every row.")

    row_axis_body = {
        "schema_id": "palette.chaser_relative_frame.core_row_axis",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "policy_id": CORE_CHASER_ROW_AXIS_POLICY_ID,
        "core_authority_roster_sha256": core.core_authority_roster_sha256,
        "core_motion_run_path": core.run_path,
        "core_motion_manifest_sha256": core.source_manifest_sha256,
        "track_id": core.track_id,
        "acquisition_frame_sha256": array_values_sha256(frames),
        "core_track_sample_index_sha256": array_values_sha256(core_rows),
        "source_chaser_run_path": chaser.run_path,
        "source_chaser_manifest_sha256": chaser.manifest_sha256,
        "cardinality": "one_chaser_frame_to_one_selected_core_track_sample",
        "interpolation": "prohibited",
        "fallback": "prohibited",
    }
    row_axis_digest = canonical_json_sha256(row_axis_body)
    row_axis_id = f"{core.run_path}/tracks/id_{core.track_id}/track_sample_key"
    frame_keys = AcquisitionFrameKeys(
        recording_id=core.recording_id,
        acquisition_frame_id=frames,
        track_sample_id=core_rows,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
        timestamp_ns=timestamp_ns,
    )

    chaser_authority_record = _mapping(
        chaser.source_authorities.get("chaser_position"),
        label="chaser position authority",
    )
    chaser_authority = _source_authority(
        chaser_authority_record,
        recording_id=core.recording_id,
        coordinate_authority_id=coordinate_id,
        scale_authority_id=scale_id,
        timing_authority_id=timing_id,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
    )
    fish_authority = ProviderSourceAuthority(
        recording_id=core.recording_id,
        source_authority_id=core.run_path,
        source_digest=core.source_manifest_sha256,
        provider_id="core_authority_roster",
        provider_digest=core.source_binding_sha256,
        coordinate_authority_id=coordinate_id,
        scale_authority_id=scale_id,
        timing_authority_id=timing_id,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
    )
    registries = _mapping(chaser.identity_registries, label="chaser registries")
    identity_registry = _mapping(
        registries.get("chaser"), label="chaser identity registry"
    )
    role_registry = _mapping(
        registries.get("behavior_role"), label="chaser role registry"
    )
    identities = tuple(
        str(identity_registry[str(index)]) for index in range(1, chaser.n_chasers + 1)
    )
    role_codes = np.asarray(chaser.base_frame_chaser("chaser_behavior_role_code"))
    try:
        roles = np.asarray(
            [role_registry[str(int(code))] for code in role_codes.reshape(-1)],
            dtype=np.dtypes.StringDType(),
        ).reshape(role_codes.shape)
    except KeyError as exc:
        raise CoreChaserRelativeFrameAdapterError(
            "Chaser role codes are not closed by the source registry."
        ) from exc
    selection = _same_across_chasers(
        chaser.base_frame_chaser("selection_member"),
        label="chaser selection membership",
    ).astype(bool, copy=False)
    occurrence = np.asarray(
        chaser.base_frame_chaser("chaser_occurrence_member"), dtype=bool
    )
    chaser_px = np.asarray(
        chaser.base_frame_chaser("chaser_position_xy_px"), dtype=np.float64
    )
    chaser_valid = np.asarray(
        chaser.base_frame_chaser("chaser_position_valid"), dtype=bool
    )
    chaser_rows = np.asarray(
        chaser.base_frame_chaser("chaser_source_row_id"), dtype=np.int64
    )
    trial_ids = np.asarray(chaser.base_frame_chaser("trial_id"), dtype=np.int64)
    active = (
        np.asarray(chaser.base_frame_chaser("active_state_code"), dtype=np.uint8) == 1
    )
    expected_pair_shape = (chaser.n_frames, chaser.n_chasers)
    if (
        chaser_px.shape != (*expected_pair_shape, 2)
        or chaser_valid.shape != expected_pair_shape
        or chaser_rows.shape != expected_pair_shape
        or occurrence.shape != expected_pair_shape
        or trial_ids.shape != expected_pair_shape
        or active.shape != expected_pair_shape
    ):
        _fail("Chaser facts do not share the declared frame/chaser axes.")

    coordinate_policy = CoordinatePolicy(
        coordinate_authority_id=coordinate_id,
        coordinate_frame="source_camera_continuous_pixel_xy",
        policy_id=str(coordinate.get("policy_id")),
    )
    scale_policy = ScalePolicy(
        scale_authority_id=scale_id,
        scale_digest=str(scale.get("scale_digest")),
        pixels_per_unit=float(pixels_per_mm),
        unit="mm",
        policy_id=str(scale.get("policy_id")),
    )
    timing_policy = TimingPolicy(
        timing_authority_id=timing_id,
        timing_digest=str(timing_policy_source.get("timing_digest")),
        recording_id=core.recording_id,
        timestamp_field=str(timing_policy_source.get("timestamp_field")),
        policy_id=str(timing_policy_source.get("policy_id")),
    )
    inputs = ChaserRelativeFrameInput(
        frame_keys=frame_keys,
        fish_xy=fish_px,
        fish_valid=fish_valid,
        fish_source_row_index=core_rows,
        fish_authority=fish_authority,
        chasers=ChaserObservations(
            identities=identities,
            behavior_roles=roles,
            xy=chaser_px,
            valid=chaser_valid,
            source_row_index=chaser_rows,
            authority=chaser_authority,
            trial_ids=trial_ids,
            active=active,
        ),
        selection_membership=selection,
        occurrence_membership=occurrence,
        coordinate_policy=coordinate_policy,
        scale_policy=scale_policy,
        timing_policy=timing_policy,
        body_frame=None,
    )
    result = compute_chaser_relative_frame(inputs)

    core_authority_record = {
        "schema_id": "palette.chaser_relative_frame.core_authority_binding",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "core_authority_roster_sha256": core.core_authority_roster_sha256,
        "core_authority_consumption_receipt": _plain(receipt),
        "core_motion": {
            "run_path": core.run_path,
            "source_manifest_sha256": core.source_manifest_sha256,
            "source_binding_sha256": core.source_binding_sha256,
            "track_id": core.track_id,
            "row_axis_sha256": row_axis_digest,
        },
        "chaser_source": {
            "run_path": chaser.run_path,
            "manifest_sha256": chaser.manifest_sha256,
            "verification_digest": chaser.verification_digest,
            "consumed_authority": "chaser_position",
            "fish_position_authority": "not_used_core_roster_selected_instead",
            "body_frame_authority": "not_used_core_roster_selected_instead",
        },
        "fish_pixel_projection": {
            "source": "core_positions_mm",
            "formula": "positions_mm * pixels_per_mm",
            "physical_authority_sha256": scale.get("scale_digest"),
        },
        "core_motion_facts_repeated": False,
        "fallback": "prohibited",
    }
    temporal_record = {
        "schema_id": "palette.chaser_relative_frame.core_temporal_selection",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "selection_id": CORE_CHASER_TEMPORAL_SELECTION_ID,
        "row_axis_authority_id": row_axis_id,
        "row_axis_authority_sha256": row_axis_digest,
        "recording_timing_authority_sha256": timing.sha256,
        "selected_frame_count": chaser.n_frames,
        "selection": "source_chaser_frames_exactly_present_in_selected_core_track",
        "fallback": "prohibited",
    }
    analysis_profile = {
        "schema_id": "palette.chaser_relative_frame.core_analysis_profile",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "profile_id": CORE_CHASER_RELATIVE_PROFILE_ID,
        "core_authority_roster_sha256": core.core_authority_roster_sha256,
        "source_chaser_profile_sha256": canonical_json_sha256(
            _plain(_context_record(source_context, "analysis_profile"))
        ),
        "body_frame": "unavailable_not_requested",
    }
    subject = _context_record(source_context, "subject_identity")
    occurrence_record = _context_record(source_context, "chaser_occurrence")
    acquisition = _context_record(source_context, "acquisition_projection")
    assert (
        subject is not None
        and occurrence_record is not None
        and acquisition is not None
    )
    publication = _context_record(
        source_context,
        "acquisition_projection_publication",
        required=False,
    )
    arena = _context_record(source_context, "arena_geometry", required=False)
    transform = _context_record(
        source_context,
        "arena_to_source_camera_transform",
        required=False,
    )
    publication_context = ChaserRelativeFramePublicationContext(
        fish_identity=str(source_context["fish_identity"]),
        subject_identity_record=subject,
        temporal_selection_record=temporal_record,
        chaser_occurrence_record=occurrence_record,
        acquisition_projection_record=acquisition,
        acquisition_projection_publication_record=publication,
        controller_state_record=controller,
        analysis_profile_record=analysis_profile,
        core_authority_record=core_authority_record,
        arena_geometry_record=arena,
        arena_to_source_camera_transform_record=transform,
    )
    prepared = prepare_chaser_relative_frame(result, context=publication_context)
    core_sha = canonical_json_sha256(core_authority_record)
    if prepared.manifest["context"]["core_authority"]["sha256"] != core_sha:
        _fail("Prepared relative frame lost its core authority binding.")
    return PreparedCoreChaserRelativeFrame(
        prepared=prepared,
        core_authority_record=MappingProxyType(core_authority_record),
        core_authority_record_sha256=core_sha,
        source_chaser_run_path=chaser.run_path,
        source_chaser_manifest_sha256=chaser.manifest_sha256,
    )


__all__ = [
    "CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_ID",
    "CORE_CHASER_RELATIVE_CONSUMER_ID",
    "CORE_CHASER_RELATIVE_PROFILE_ID",
    "CoreChaserRelativeFrameAdapterError",
    "PreparedCoreChaserRelativeFrame",
    "prepare_core_chaser_relative_frame",
]
