"""Same-file keyed correspondence; sealed lineage is not upstream acceptance."""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.stimulus_coordinate_contract import (
    _v6_validate_orange_source_record,
)

from .common import (
    MAX_ROWS,
    KeyIndex,
    canonical_json,
    contract_errors,
    digest,
    exact_keys,
    require,
    same_json,
    text,
    uint64,
)
from .hdf5_types import dataset_bytes, iter_blocks
from .integrity import internal_nodes
from .rows import index_table
from .schema import describe_table, read_json

BINDING = "/correspondence/acquisition/binding_json"
INPUT = "/correspondence/acquisition/raw_live_identity_provenance_json"
SOURCE = "/correspondence/acquisition/source_semantic_record_json"
FRAMES = "/frames/stimulus"
FRAME_SOURCES = "/correspondence/frames/sources"


@dataclass(frozen=True)
class CorrespondenceSummary:
    mapped_frame_count: int
    component_rows: dict[str, int]
    recording_id: str
    camera_serial: str


def _binding(h5):
    binding = read_json(h5, BINDING, canonical=True)
    exact_keys(
        binding,
        (
            "schema_id",
            "schema_version",
            "recording_id",
            "recording_identity_token",
            "acquisition_camera_id",
            "camera_serial",
            "shaman_numeric_camera_id",
        ),
        "acquisition_binding",
    )
    require(
        binding["schema_id"] == "citrus.experimental_h5.acquisition_binding"
        and type(binding["schema_version"]) is int
        and binding["schema_version"] == 1,
        "acquisition_binding_schema",
    )
    for name in ("recording_id", "camera_serial", "acquisition_camera_id"):
        text(binding[name], name)
    require(
        uint64(binding["shaman_numeric_camera_id"], "shaman_numeric_camera_id") < 2**32,
        "shaman_numeric_camera_id_overflow",
    )
    token = {
        "canonicalization": "canonical_json_utf8_sort_keys_compact_v1",
        "recording_id": binding["recording_id"],
        "schema_id": "orange.shaman_v2.recording_identity",
        "schema_version": 1,
        "scope": "recording_session",
    }
    require(
        binding["recording_identity_token"] == digest(canonical_json(token)),
        "recording_identity_token_mismatch",
    )
    source = read_json(h5, SOURCE)
    require(
        uint64(source["schema_version"], "source_schema_version") == 1,
        "source_schema_version",
    )
    total = _v6_validate_orange_source_record(
        source,
        recording_id=binding["recording_id"],
        camera_serial=binding["camera_serial"],
    )
    for stream in source["camera_streams"].values():
        for values, name in (
            (stream["producer_identity"], "index_base"),
            (stream["destination_identity"], "index_base"),
            (stream["conversion"], "constant"),
        ):
            uint64(values[name], "source_mapping:" + name)
        for name in (
            "first_recording_frame_id",
            "last_recording_frame_id",
            "metadata_row_count",
            "total_acquisitions",
            "gap_count",
        ):
            uint64(stream["coverage"][name], "source_coverage:" + name)
    require(total <= 2**63, "acquisition_index_int64_overflow")
    return binding, total


def _receipt(h5, total, components):
    receipt = read_json(h5, "/correspondence/receipt_json", canonical=True)
    exact_keys(
        receipt,
        (
            "schema_id",
            "schema_version",
            "status",
            "reason",
            "scope",
            "binding_ref",
            "binding_sha256",
            "input_ref",
            "input_sha256",
            "source_record_ref",
            "source_record_declared_sha256",
            "source_record_observed_sha256",
            "mapping_validation",
            "dependencies",
            "outputs",
        ),
        "correspondence_receipt",
    )
    require(
        receipt["schema_id"] == "citrus.experimental_h5.correspondence_receipt"
        and type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["status"] == "complete"
        and receipt["reason"] == ""
        and receipt["scope"]
        == "validated_index_correspondence_not_complete_experiment_or_geometry_admission",
        "correspondence_receipt_incomplete",
    )
    for name, path in (
        ("binding", BINDING),
        ("input", INPUT),
        ("source_record", SOURCE),
    ):
        require(receipt[name + "_ref"] == path, f"correspondence_reference:{name}")
        actual = digest(dataset_bytes(h5[path]))
        names = (
            ("source_record_declared_sha256", "source_record_observed_sha256")
            if name == "source_record"
            else (name + "_sha256",)
        )
        require(
            all(receipt[key] == actual for key in names),
            f"correspondence_digest:{name}",
        )
    require(
        receipt["mapping_validation"]
        == {
            "reason": "",
            "status": "valid",
            "total_acquisitions": total,
            "validator": "orange_acquisition_index_mapping_v1",
        },
        "correspondence_mapping_validation",
    )
    for key, paths in (
        (
            "dependencies",
            [FRAMES] + [f"/components/{name}/states" for name in components],
        ),
        (
            "outputs",
            [FRAME_SOURCES]
            + [f"/correspondence/{name}/sources" for name in components],
        ),
    ):
        values = receipt[key]
        require(
            type(values) is list and len(values) == len(paths),
            f"correspondence_descriptor_count:{key}",
        )
        observed = {
            value.get("table", {}).get("path"): value
            for value in values
            if type(value) is dict
        }
        require(set(observed) == set(paths), f"correspondence_descriptor_paths:{key}")
        for path in paths:
            require(
                same_json(observed[path], describe_table(h5, path)),
                f"correspondence_descriptor_mismatch:{path}",
            )


@contract_errors
def validate_component_correspondence(h5) -> CorrespondenceSummary:
    internal_nodes(h5)
    binding, total = _binding(h5)
    inputs = read_json(h5, INPUT, canonical=True)
    exact_keys(
        inputs,
        (
            "schema_id",
            "schema_version",
            "state_coverage",
            "upstream_failure_reason",
            "frames",
            "states",
        ),
        "correspondence_input",
    )
    require(
        inputs["schema_id"] == "citrus.experimental_h5.correspondence_input"
        and type(inputs["schema_version"]) is int
        and inputs["schema_version"] == 1
        and inputs["state_coverage"] == "all_canonical_rows_of_selected_components"
        and inputs["upstream_failure_reason"] == ""
        and type(inputs["frames"]) is list
        and type(inputs["states"]) is dict,
        "correspondence_input_contract",
    )
    components = tuple(inputs["states"])
    require(
        all(
            name in ("chaser", "independent_motion_grid", "moving_grating")
            for name in components
        ),
        "unsupported_correspondence_component",
    )
    require(
        len(inputs["frames"]) + sum(len(value) for value in inputs["states"].values())
        <= MAX_ROWS,
        "correspondence_row_budget",
    )
    _receipt(h5, total, components)
    counts = {}
    with KeyIndex() as index:
        index_table(h5[FRAMES], index, FRAMES, reason="canonical_frame_duplicate")
        for position, row in enumerate(inputs["frames"]):
            exact_keys(row, ("stimulus_frame_num", "recording_frame_id"), "input_frame")
            frame = uint64(row["stimulus_frame_num"], "stimulus_frame_num")
            recording = uint64(row["recording_frame_id"], "recording_frame_id")
            require(
                index.lookup(FRAMES, (frame,)) is not None and 0 < recording <= total,
                "input_frame_unresolved",
            )
            index.add(
                "input_frames", (frame,), position, reason="input_frame_duplicate"
            )
        require(
            h5[FRAME_SOURCES].shape == (len(inputs["frames"]),), "mapped_frame_coverage"
        )
        for _, block in iter_blocks(h5[FRAME_SOURCES]):
            for row in block:
                frame = int(row["stimulus_frame_num"])
                position = index.lookup("input_frames", (frame,))
                require(position is not None, "mapped_frame_unresolved")
                index.add(
                    "mapped_frames", (frame,), position, reason="mapped_frame_duplicate"
                )
                recording = inputs["frames"][position]["recording_frame_id"]
                require(
                    int(row["source_recording_frame_id"]) == recording
                    and row["source_recording_frame_valid"]
                    == row["source_acquisition_frame_valid"]
                    == 1
                    and int(row["source_acquisition_frame_index"]) == recording - 1,
                    "mapped_frame_identity_mismatch",
                )
        for component in components:
            states = h5[f"/components/{component}/states"]
            mapped = h5[f"/correspondence/{component}/sources"]
            evidence = inputs["states"][component]
            require(
                type(evidence) is list
                and states.shape == mapped.shape == (len(evidence),),
                "selected_state_coverage",
            )
            fields = states.attrs["key_fields"].split(",")
            index_table(
                states, index, component, reason="canonical_state_key_duplicate"
            )
            for position, expected in enumerate(evidence):
                exact_keys(
                    expected,
                    (
                        "key",
                        "source_recording_frame_id",
                        "target_recording_frame_id",
                        "target_recording_frame_valid",
                    ),
                    "input_state",
                )
                require(
                    type(expected["key"]) is list
                    and len(expected["key"]) == len(fields),
                    "input_state_key_malformed",
                )
                key = tuple(
                    uint64(value, "input_state_key") for value in expected["key"]
                )
                ordinal = index.lookup(component, key)
                require(ordinal is not None, "input_state_key_unresolved")
                index.add(
                    component + ":inputs",
                    (ordinal,),
                    position,
                    reason="input_state_key_duplicate",
                )
            for _, block in iter_blocks(mapped):
                for row in block:
                    ordinal = int(row["state_row_index"])
                    position = index.lookup(component + ":inputs", (ordinal,))
                    require(position is not None, "mapped_state_unresolved")
                    index.add(
                        component + ":mapped",
                        (ordinal,),
                        position,
                        reason="mapped_state_duplicate",
                    )
                    expected = evidence[position]
                    state = states[ordinal]
                    frame_position = index.lookup(
                        "input_frames", (int(state["stimulus_frame_num"]),)
                    )
                    require(frame_position is not None, "state_source_frame_unresolved")
                    recording = uint64(
                        expected["source_recording_frame_id"],
                        "state_source_recording_frame_id",
                    )
                    require(
                        recording
                        == inputs["frames"][frame_position]["recording_frame_id"]
                        and row["source_acquisition_frame_valid"] == 1
                        and int(row["source_acquisition_frame_index"]) == recording - 1,
                        "state_current_source_mismatch",
                    )
                    valid = uint64(
                        expected["target_recording_frame_valid"], "target_valid"
                    )
                    target = uint64(
                        expected["target_recording_frame_id"],
                        "target_recording_frame_id",
                    )
                    require(
                        valid in (0, 1)
                        and (0 < target <= total if valid else target == 0),
                        "invalid_target_source",
                    )
                    require(
                        int(row["target_source_acquisition_frame_valid"]) == valid
                        and int(row["target_source_acquisition_frame_index"])
                        == (target - 1 if valid else 0),
                        "state_held_target_mismatch",
                    )
            counts[component] = len(evidence)
    return CorrespondenceSummary(
        len(inputs["frames"]), counts, binding["recording_id"], binding["camera_serial"]
    )
