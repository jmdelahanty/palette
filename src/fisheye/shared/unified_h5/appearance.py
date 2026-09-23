"""Step-local appearance ownership and the pinned float32/RGBA8 audit witness."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .common import (
    KeyIndex,
    canonical_json,
    contract_errors,
    digest,
    exact_keys,
    require,
    text,
    uint64,
)
from .hdf5_types import iter_blocks
from .integrity import internal_nodes
from .rows import index_table
from .schema import APPEARANCE, APPEARANCE_MANIFEST, describe_table, read_json
from .vendor import object_appearance_reference as oracle


@dataclass(frozen=True)
class AppearanceSummary:
    row_count: int
    authored_owner_count: int


def _profile_identity(value):
    exact_keys(value, oracle.PROFILE_FIELDS | {"profile_digest"}, "authored_appearance")
    profile = {key: deepcopy(value[key]) for key in oracle.PROFILE_FIELDS}
    oracle.validate_profile(profile)
    identity = deepcopy(profile)
    for name in (
        "baseline_contrast",
        "peak_contrast",
        "baseline_chromatic_strength",
        "peak_chromatic_strength",
    ):
        scaled = float(np.float32(profile[name])) / 0.000001
        rounded = int(np.floor(abs(scaled) + 0.5)) * (-1 if scaled < 0 else 1)
        identity[name] = {"scale": "1e-6", "unit": "scalar", "value": rounded}
    require(
        value["profile_digest"] == digest(canonical_json(identity)),
        "appearance_profile_digest_mismatch",
    )
    return profile


def _owners_and_intervals(h5):
    authored = read_json(h5, "/protocol/authored/protocol_trial_index_json")
    execution = read_json(h5, "/protocol/executed/execution_index_json")
    require(
        execution["schema_id"] == "citrus.protocol.execution_index"
        and execution["policy_id"]
        == "citrus.protocol.execution_index.half_open_stimulus_frames.v1"
        and execution["authoritative_interval_axis"] == "stimulus_frame_num"
        and execution["status"] in ("complete", "interrupted"),
        "appearance_execution_contract",
    )
    owners = {}
    require(
        type(authored["steps"]) is list and type(execution["steps"]) is list,
        "appearance_steps_malformed",
    )
    for ordinal, step in enumerate(authored["steps"]):
        require(
            uint64(step["step_index"], "authored_step") == ordinal,
            "appearance_authored_step_order",
        )
        chaser_indices = set()
        for chaser in step.get("features", {}).get("chasers", []):
            chaser_index = uint64(chaser["chaser_index"], "chaser_index")
            require(
                chaser_index not in chaser_indices,
                "appearance_authored_step_chaser_duplicate",
            )
            chaser_indices.add(chaser_index)
            value = chaser.get("appearance")
            if value is None:
                continue
            require(
                value["schema_id"] == oracle.PROFILE_SCHEMA_ID
                and uint64(value["schema_version"], "appearance_profile_schema_version")
                == 1,
                "appearance_profile_identity_schema",
            )
            identity = (
                ordinal,
                text(value["scope_id"], "scope_id"),
                text(value["profile_digest"], "profile_digest"),
                value["schema_version"],
            )
            require(identity not in owners, "appearance_authored_step_owner_duplicate")
            owners[identity] = (chaser, value)
    intervals, seen = [], set()
    for step in execution["steps"]:
        ordinal = uint64(step["step_index"], "executed_step")
        require(
            ordinal < len(authored["steps"]) and ordinal not in seen,
            "appearance_executed_step_duplicate_or_unknown",
        )
        seen.add(ordinal)
        interval = step["interval"]
        start = uint64(interval["start_stimulus_frame_inclusive"], "interval_start")
        end = uint64(interval["end_stimulus_frame_exclusive"], "interval_end")
        require(start <= end, "appearance_interval_reversed")
        if start < end:
            intervals.append((start, end, ordinal))
    intervals.sort()
    require(
        all(first[1] <= second[0] for first, second in zip(intervals, intervals[1:])),
        "appearance_intervals_overlap",
    )
    return owners, intervals


def _verify_witness(h5, row, chaser, profile, index):
    enum_values = {
        "contrast_model_id": ("appearance_contrast_models", profile["contrast_model"]),
        "calculation_domain_id": (
            "appearance_calculation_domains",
            profile["calculation_domain"],
        ),
        "reference_policy_id": ("appearance_reference_policies", profile["reference"]),
        "alpha_policy_id": ("appearance_alpha_policies", profile["alpha_policy"]),
        "modulation_mode_id": (
            "appearance_modulation_modes",
            profile["modulation"]["mode"],
        ),
        "modulation_curve_id": (
            "appearance_modulation_curves",
            profile["modulation"]["curve"],
        ),
        "invalid_drive_policy_id": (
            "appearance_invalid_drive_policies",
            profile["modulation"]["invalid_drive_policy"],
        ),
        "invalid_reference_policy_id": (
            "appearance_invalid_reference_policies",
            profile["invalid_reference_policy"],
        ),
    }
    for field, (enum, expected) in enum_values.items():
        path = "/definitions/enums/" + enum
        ordinal = index.lookup(path, (int(row[field]),))
        require(
            ordinal is not None and text(h5[path][ordinal]["name"], enum) == expected,
            f"appearance_enum_profile_mismatch:{field}",
        )
    rgba8 = chaser["color"]["rgba8"]
    require(
        type(rgba8) is list
        and len(rgba8) == 4
        and all(type(value) is int and 0 <= value <= 255 for value in rgba8),
        "appearance_authored_rgba8",
    )
    request = {
        "schema_id": oracle.REQUEST_SCHEMA_ID,
        "schema_version": 1,
        "appearance_profile": profile,
        "drive": {
            "value": float(row["drive_value"]),
            "valid": bool(row["drive_valid"]),
        },
        "local_underlay_rgba": list(map(float, row["reference_rgba"])),
        "authored_target_rgba": [
            float(np.float32(value) * np.float32(1 / 255)) for value in rgba8
        ],
    }
    result = oracle.evaluate_request(request)["result"]
    stored = oracle.stored_h5_witness_v1(result, bool(row["sample_scope_visible"]))
    for name in (
        "effective_drive",
        "requested_contrast",
        "requested_chromatic_strength",
        "reference_code_luminance",
        "requested_code_luminance",
    ):
        require(
            abs(float(row[name]) - result[name]) <= 1e-6,
            f"appearance_oracle_float:{name}",
        )
    for name in (
        "invalid_drive_fallback_used",
        "reference_valid",
        "authored_chromaticity_valid",
        "clipped",
    ):
        require(bool(row[name]) == result[name], f"appearance_oracle_validity:{name}")
    require(
        bool(row["realized_valid"]) == stored["realized_valid"],
        "appearance_stored_validity",
    )
    require(
        np.allclose(row["realized_rgba"], stored["realized_rgba"], atol=1e-6, rtol=0)
        and abs(
            float(row["realized_code_luminance"]) - stored["realized_code_luminance"]
        )
        <= 1e-6,
        "appearance_stored_rgba_luminance",
    )
    require(
        np.rint(row["realized_rgba"] * 255).astype(int).tolist()
        == result["realized_rgba8"],
        "appearance_stored_rgba8",
    )


def _joined_rows(h5):
    internal_nodes(h5)
    describe_table(h5, APPEARANCE)
    manifest = read_json(h5, APPEARANCE_MANIFEST, canonical=True)
    owners, intervals = _owners_and_intervals(h5)
    starts = [value[0] for value in intervals]
    with KeyIndex() as index:
        for path in ("/frames/stimulus", "/components/chaser/states"):
            describe_table(h5, path)
            index_table(
                h5[path],
                index,
                path,
                reason=f"appearance_dependency_key_duplicate:{path}",
            )
        for name in h5["/definitions/enums"]:
            if name.startswith("appearance_"):
                path = "/definitions/enums/" + name
                describe_table(h5, path)
                index_table(h5[path], index, path, reason="appearance_enum_duplicate")
        for start, block in iter_blocks(h5[APPEARANCE]):
            for offset, row in enumerate(block):
                frame, scope = int(row["stimulus_frame_num"]), text(
                    row["scope_id"], "scope_id"
                )
                index.add(
                    "appearance",
                    (frame, scope),
                    start + offset,
                    reason="appearance_runtime_key_duplicate",
                )
                require(
                    index.lookup("/frames/stimulus", (frame,)) is not None,
                    "appearance_frame_unresolved",
                )
                interval_index = bisect_right(starts, frame) - 1
                require(
                    interval_index >= 0 and frame < intervals[interval_index][1],
                    "appearance_runtime_step_unresolved",
                )
                step = intervals[interval_index][2]
                owner = owners.get(
                    (
                        step,
                        scope,
                        text(row["profile_digest"], "profile_digest"),
                        int(row["profile_schema_version"]),
                    )
                )
                require(owner is not None, "appearance_runtime_profile_wrong_step")
                chaser, profile = owner
                require(
                    index.lookup(
                        "/components/chaser/states", (chaser["chaser_index"], frame)
                    )
                    is not None,
                    "appearance_runtime_chaser_state_unresolved",
                )
                require(
                    text(row["renderer_adapter_id"], "renderer_adapter_id")
                    == manifest["adapters"][0]["renderer_adapter_id"],
                    "appearance_renderer_adapter_mismatch",
                )
                require(
                    text(
                        row["reference_sample_policy_id"], "reference_sample_policy_id"
                    )
                    == "object_center_v1",
                    "appearance_reference_sample_policy",
                )
                yield row, chaser, profile, index, len(owners)


@contract_errors
def validate_appearance_component(h5) -> AppearanceSummary:
    """Validate scoped table/reference/ownership joins, not full replay admission.

    The producer's component-only identity witnesses intentionally do not carry
    a complete authored profile or global protocol seal. Artifact admission
    separately requires the complete protocol and numerical witness below.
    """
    count, owners = 0, 0
    for _, _, _, _, owners in _joined_rows(h5):
        count += 1
    return AppearanceSummary(count, owners)


@contract_errors
def validate_appearance_witness(h5) -> AppearanceSummary:
    count, owners = 0, 0
    for row, chaser, value, index, owners in _joined_rows(h5):
        profile = _profile_identity(value)
        _verify_witness(h5, row, chaser, profile, index)
        count += 1
    return AppearanceSummary(count, owners)
