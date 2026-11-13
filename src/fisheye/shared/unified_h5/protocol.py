"""Native paths, existing protocol semantic/execution contract owners."""

import math

import numpy as np

from fisheye.shared.protocol_execution_contract import validate_protocol_execution_index
from fisheye.shared.protocol_semantic_contract import (
    PRE_V4_STIMULUS_RENDER_RATE_HZ,
    validate_protocol_semantic_snapshot,
)

from .common import require, text
from .hdf5_types import dataset_bytes
from .schema import read_json


def validate_protocol(
    h5,
    stimulus_angle_convention: str | None = None,
    admits_heading_split: bool | None = None,
):
    """Validate the native protocol snapshot and execution index.

    ``stimulus_angle_convention`` is the admitted binding's
    (``validate_capacity_preflight(h5)["stimulus_angle_convention"]``); it
    selects the protocol contract (semantic v2 or recipes v3/v4 / trial index v3
    for ``atan2_negative_y_x_degrees``). The recorded rates must agree with the
    identity's render rate (``validate_render_rate``). When omitted it is resolved from the
    file's capacity preflight binding, and a file without one is legacy
    (its admission revision predates capacity preflights). Likewise
    ``admits_heading_split`` is the binding's (revision 6, phototaxis_v1);
    a caller that passes the convention without it admits no reactive step.
    """
    if stimulus_angle_convention is None:
        binding = _binding(h5)
        stimulus_angle_convention = binding["stimulus_angle_convention"]
        if admits_heading_split is None:
            admits_heading_split = binding["admits_heading_split"]
    elif admits_heading_split is None:
        admits_heading_split = False
    authored, executed = h5["/protocol/authored"], h5["/protocol/executed"]
    for group in (authored, executed):
        version = group.attrs.get_id("schema_version")
        require(
            version.shape == () and version.dtype == np.dtype("<i4"),
            "native_protocol_version_type",
        )
    require(
        authored.attrs.get("schema_id") == "citrus.protocol.snapshot"
        and authored.attrs.get("contract_status") == "valid",
        "native_protocol_snapshot_status",
    )
    # Parse first with the bounded duplicate-key/nonfinite parser, even for v1.
    for path in (
        "protocol_definition_json",
        "protocol_semantic_json",
        "protocol_trial_index_json",
    ):
        read_json(h5, authored.name + "/" + path)
    read_json(h5, executed.name + "/execution_index_json")

    def value(group, name):
        return text(dataset_bytes(group[name]), name)

    snapshot = validate_protocol_semantic_snapshot(
        semantic_hash=value(authored, "protocol_semantic_hash"),
        semantic_json=value(authored, "protocol_semantic_json"),
        trial_index_json=value(authored, "protocol_trial_index_json"),
        trial_index_hash=value(authored, "protocol_trial_index_hash"),
        snapshot_schema_version=int(authored.attrs["schema_version"]),
        snapshot_policy_id=text(authored.attrs["policy_id"], "protocol_policy_id"),
        stimulus_angle_convention=stimulus_angle_convention,
        admits_heading_split=admits_heading_split,
    )
    execution = validate_protocol_execution_index(
        execution_json=value(executed, "execution_index_json"),
        execution_hash=value(executed, "execution_index_hash"),
        snapshot=snapshot,
    )
    validate_render_rate(h5, snapshot.stimulus_render_rate_hz)
    require(
        executed.attrs.get("schema_id") == "citrus.protocol.execution_index"
        and executed.attrs.get("schema_version") == 1
        and executed.attrs.get("policy_id")
        == "citrus.protocol.execution_index.half_open_stimulus_frames.v1"
        and executed.attrs.get("status") == execution.status,
        "native_execution_attributes_mismatch",
    )
    return snapshot, execution


SESSION = "/metadata/session"
DISPLAY_TIMING = "/timing/display_submissions"
RATE_ATTRIBUTES = ("stimulus_frame_rate_nominal_hz", "display_submission_rate_target_hz")
RENDER_RATE_SOURCE = "rig_file_projection_path_stimulus_render_rate_hz"
REFRESH_TOLERANCE_HZ = 0.05


def _rate_attribute(attrs, name: str) -> float | None:
    if name not in attrs:
        return None
    raw = attrs.get_id(name)
    require(
        raw.shape == () and raw.dtype == np.dtype("<f8"),
        f"native_render_rate_attribute_type:{name}",
    )
    value = float(attrs[name])
    require(math.isfinite(value) and value > 0, f"native_render_rate_attribute_value:{name}")
    return value


def validate_render_rate(h5, identity_rate_hz: int | None) -> int:
    """The H5's recorded rates equal the protocol identity's render rate.

    Semantic recipe v4 (pairing revision 3) hashes the render rate; every
    earlier identity ran at 120 Hz. Citrus writes all of these from one
    per-process value, so any disagreement is refused: a 240 Hz recording can
    never be read as a 120 Hz one. A v4 identity requires both session rate
    attributes; older files are checked where they carry them. A declared
    refresh (rig file ``display_mode``) must match the reported refresh and be
    a whole multiple of the render rate, each within 0.05 Hz.
    """

    rate = PRE_V4_STIMULUS_RENDER_RATE_HZ if identity_rate_hz is None else identity_rate_hz
    session = h5[SESSION].attrs
    for name in RATE_ATTRIBUTES:
        value = _rate_attribute(session, name)
        if identity_rate_hz is not None:
            require(value is not None, f"native_render_rate_attribute_missing:{name}")
        require(value is None or value == rate, f"native_render_rate_mismatch:{name}")
    if DISPLAY_TIMING in h5:
        value = _rate_attribute(h5[DISPLAY_TIMING].attrs, "display_submission_rate_target_hz")
        require(
            value is None or value == rate,
            "native_render_rate_mismatch:display_submissions",
        )
    declared = _rate_attribute(session, "display_refresh_rate_declared_hz")
    source = session.get("stimulus_render_rate_source")
    if declared is None:
        require(source is None, "native_render_rate_source_without_declared_refresh")
        return rate
    require(source == RENDER_RATE_SOURCE, "native_render_rate_source")
    reported = _rate_attribute(session, "display_refresh_rate_reported_hz")
    require(
        reported is None or abs(reported - declared) <= REFRESH_TOLERANCE_HZ,
        "native_display_refresh_mismatch",
    )
    multiple = round(declared / rate)
    require(
        multiple >= 1 and abs(declared - multiple * rate) <= REFRESH_TOLERANCE_HZ,
        "native_render_rate_incompatible_with_refresh",
    )
    return rate


def _binding(h5) -> dict:
    # Imported here: correspondence imports this package's schema helpers.
    from fisheye.shared.heading_convention import LEGACY_CANVAS_ANGLE

    from .correspondence import PREFLIGHT, validate_capacity_preflight

    if PREFLIGHT not in h5:
        return {"stimulus_angle_convention": LEGACY_CANVAS_ANGLE, "admits_heading_split": False}
    return validate_capacity_preflight(h5)
