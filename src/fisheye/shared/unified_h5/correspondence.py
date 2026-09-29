"""Same-file keyed correspondence; sealed lineage is not upstream acceptance."""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.stimulus_coordinate_contract import (
    _v6_validate_orange_source_record,
)

from .common import (
    BLOCK_BYTES,
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
from .schema import (
    CHASER_STATES,
    contract,
    describe_internal_dataset,
    describe_table,
    read_json,
    table_version,
    validate_catalog_table,
)

BINDING = "/correspondence/acquisition/binding_json"
INPUT = "/correspondence/acquisition/raw_live_identity_provenance_json"
SOURCE = "/correspondence/acquisition/source_semantic_record_json"
FRAMES = "/frames/stimulus"
FRAME_SOURCES = "/correspondence/frames/sources"
CHASER_SOURCES = "/correspondence/chaser/sources"
RECEIPT = "/correspondence/receipt_json"
LIVE_FRAMES = "/correspondence/acquisition/live_frame_identities"
LIVE_CHASERS = "/correspondence/acquisition/live_chaser_identities"
PREFLIGHT = "/metadata/admission/capacity_preflight_json"


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


@dataclass(frozen=True)
class AcquisitionBinding:
    """Validated Orange acquisition identity bound into one unified H5.

    ``acquisition_session_id`` is the binding's ``recording_id``: the Orange
    ``recording_session`` shared by every camera recording started together.
    It is not the per-Arena Citrus ``/metadata/session@session_uuid``.
    """

    acquisition_session_id: str
    camera_serial: str
    acquisition_camera_id: str


@contract_errors
def read_acquisition_binding(h5) -> AcquisitionBinding:
    binding, _total = _binding(h5)
    return AcquisitionBinding(
        acquisition_session_id=binding["recording_id"],
        camera_serial=binding["camera_serial"],
        acquisition_camera_id=binding["acquisition_camera_id"],
    )


def _schema_check(document, schema_name: str, label: str) -> None:
    from jsonschema import Draft202012Validator

    errors = list(Draft202012Validator(contract(schema_name)).iter_errors(document))
    if errors:
        first = min(errors, key=lambda error: list(error.absolute_path))
        path = "/".join(str(part) for part in first.absolute_path)
        require(False, f"{label}_schema:{path}:{first.validator}")


def _check_refs(h5, receipt) -> None:
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


def _receipt(h5, total, components):
    receipt = read_json(h5, RECEIPT, canonical=True)
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
    _check_refs(h5, receipt)
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
    version = inputs.get("schema_version")
    chaser_version = table_version(h5, CHASER_STATES)
    # One revision per file: correspondence v2 goes with chaser v2 (admission
    # v2); v1 with v1 (the earlier revision, kept until fixtures are regenerated).
    require(
        type(version) is int
        and version in (1, 2)
        and chaser_version in (None, version),
        "correspondence_revision_mismatch",
    )
    if version == 2:
        return _validate_v2(h5, binding, total, inputs)
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


class _RowReader:
    """Random row access through one bounded, cached first-axis block."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.rows = max(1, BLOCK_BYTES // max(1, dataset.dtype.itemsize))
        self.start = 0
        self.block = dataset[0:0]

    def __getitem__(self, index: int):
        if not self.start <= index < self.start + len(self.block):
            self.start = index - index % self.rows
            self.block = self.dataset[self.start : self.start + self.rows]
        return self.block[index - self.start]


def _preflight(h5) -> None:
    """Capacity-preflight evidence under the exact frozen producer policy."""

    document = read_json(h5, PREFLIGHT, canonical=True)
    _schema_check(
        document, "experimental_h5_capacity_preflight_v1.schema.json", "capacity_preflight"
    )
    pin = contract("unified_h5_admission_v2.json")["producer_policy_pin"]["file_sha256"]
    require(
        "sha256:" + document["admission_policy_sha256"] == pin,
        "capacity_preflight_producer_policy",
    )


def _validate_v2(h5, binding, total, inputs) -> CorrespondenceSummary:
    """Correspondence v2: streamed same-H5 live identities (admission v2)."""

    _schema_check(
        inputs, "experimental_h5_correspondence_input_v2.schema.json", "correspondence_input"
    )
    _preflight(h5)
    validate_catalog_table(h5, LIVE_FRAMES)
    validate_catalog_table(h5, LIVE_CHASERS)
    frames_live, chasers_live = h5[LIVE_FRAMES], h5[LIVE_CHASERS]
    frame_count, chaser_count = frames_live.shape[0], chasers_live.shape[0]
    require(
        inputs["frame_rows"] == inputs["expected_frame_rows"] == frame_count
        and inputs["chaser_rows"] == inputs["expected_chaser_rows"] == chaser_count
        and inputs["dropped_batches"] == 0
        and inputs["write_errors"] == 0
        and inputs["upstream_failure_reason"] == "",
        "correspondence_input_incomplete",
    )
    has_chaser = CHASER_STATES in h5
    require(has_chaser or chaser_count == 0, "live_chaser_without_canonical_table")

    receipt = read_json(h5, RECEIPT, canonical=True)
    _schema_check(
        receipt,
        "experimental_h5_correspondence_receipt_v2.schema.json",
        "correspondence_receipt",
    )
    require(
        receipt["status"] == "complete" and receipt["reason"] == "",
        "correspondence_receipt_incomplete",
    )
    _check_refs(h5, receipt)
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
    chaser_paths = [CHASER_STATES] if has_chaser else []
    expected = {
        "dependencies": {path: describe_table(h5, path) for path in [FRAMES] + chaser_paths}
        | {
            path: describe_internal_dataset(h5[path], "correspondence_input")
            for path in (LIVE_FRAMES, LIVE_CHASERS)
        },
        # Frame-only runs have no derived chaser sources.
        "outputs": {
            path: describe_table(h5, path)
            for path in [FRAME_SOURCES] + ([CHASER_SOURCES] if has_chaser else [])
        },
    }
    for key, descriptors in expected.items():
        values = receipt[key]
        observed = {
            (value.get("table") or {}).get("path") or value.get("path"): value
            for value in values
        }
        require(
            len(values) == len(observed) and set(observed) == set(descriptors),
            f"correspondence_descriptor_paths:{key}",
        )
        for path, descriptor in descriptors.items():
            require(
                same_json(observed[path], descriptor),
                f"correspondence_descriptor_mismatch:{path}",
            )

    # Live frames match /frames/stimulus row for row, strictly increasing, with a
    # current recording identity in [1, total].
    canonical_frames = h5[FRAMES]
    require(canonical_frames.shape == (frame_count,), "live_frame_canonical_coverage")
    previous = None
    for start, block in iter_blocks(frames_live):
        canonical = canonical_frames[start : start + len(block)]
        require(
            (block["stimulus_frame_num"] == canonical["stimulus_frame_num"]).all(),
            "live_frame_canonical_ordinal",
        )
        numbers = block["stimulus_frame_num"].astype("u8")
        require(
            (numbers[1:] > numbers[:-1]).all()
            and (previous is None or len(numbers) == 0 or numbers[0] > previous),
            "live_frame_order",
        )
        if len(numbers):
            previous = int(numbers[-1])
        recordings = block["recording_frame_id"]
        require(
            ((recordings >= 1) & (recordings <= total)).all(),
            "live_frame_recording_identity",
        )

    counts = {}
    if has_chaser:
        states = h5[CHASER_STATES]
        require(states.shape == (chaser_count,), "live_chaser_canonical_coverage")
        frames = _RowReader(frames_live)
        frame_position, last_frame, per_frame = 0, None, 0
        with KeyIndex() as index:
            for start, block in iter_blocks(chasers_live):
                canonical = states[start : start + len(block)]
                require(
                    (block["stimulus_frame_num"] == canonical["stimulus_frame_num"]).all()
                    and (block["chaser_index"] == canonical["chaser_index"]).all()
                    and (block["chaser_index"] >= 0).all(),
                    "live_chaser_canonical_ordinal",
                )
                for offset, row in enumerate(block):
                    frame = int(row["stimulus_frame_num"])
                    require(last_frame is None or frame >= last_frame, "live_chaser_order")
                    per_frame = per_frame + 1 if frame == last_frame else 1
                    require(per_frame <= 4096, "live_chaser_per_frame_budget")
                    last_frame = frame
                    index.add(
                        "live_chasers",
                        (frame, int(row["chaser_index"])),
                        start + offset,
                        reason="live_chaser_duplicate",
                    )
                    # Merge-join onto the strictly increasing live frames.
                    while (
                        frame_position < frame_count
                        and int(frames[frame_position]["stimulus_frame_num"]) < frame
                    ):
                        frame_position += 1
                    require(
                        frame_position < frame_count
                        and int(frames[frame_position]["stimulus_frame_num"]) == frame,
                        "live_chaser_frame_unresolved",
                    )
                    require(
                        int(row["source_recording_frame_id"])
                        == int(frames[frame_position]["recording_frame_id"]),
                        "live_chaser_current_source_mismatch",
                    )
                    valid = int(row["target_recording_frame_valid"])
                    target = int(row["target_recording_frame_id"])
                    require(
                        (0 < target <= total) if valid else target == 0,
                        "live_chaser_target_identity",
                    )
            live = _RowReader(chasers_live)
            require(h5[CHASER_SOURCES].shape == (chaser_count,), "mapped_state_coverage")
            for _, block in iter_blocks(h5[CHASER_SOURCES]):
                for row in block:
                    ordinal = int(row["state_row_index"])
                    require(0 <= ordinal < chaser_count, "mapped_state_unresolved")
                    index.add(
                        "mapped_chasers", (ordinal,), ordinal, reason="mapped_state_duplicate"
                    )
                    source = live[ordinal]
                    recording = int(source["source_recording_frame_id"])
                    valid = int(source["target_recording_frame_valid"])
                    target = int(source["target_recording_frame_id"])
                    require(
                        row["source_acquisition_frame_valid"] == 1
                        and int(row["source_acquisition_frame_index"]) == recording - 1,
                        "state_current_source_mismatch",
                    )
                    require(
                        int(row["target_source_acquisition_frame_valid"]) == valid
                        and int(row["target_source_acquisition_frame_index"])
                        == (target - 1 if valid else 0),
                        "state_held_target_mismatch",
                    )
        counts["chaser"] = chaser_count
    else:
        require(CHASER_SOURCES not in h5, "derived_chaser_sources_without_states")

    require(h5[FRAME_SOURCES].shape == (frame_count,), "mapped_frame_coverage")
    frames = _RowReader(frames_live)
    with KeyIndex() as index:
        for start, block in iter_blocks(frames_live):
            for offset, row in enumerate(block):
                index.add(
                    "live_frames",
                    (int(row["stimulus_frame_num"]),),
                    start + offset,
                    reason="input_frame_duplicate",
                )
        for _, block in iter_blocks(h5[FRAME_SOURCES]):
            for row in block:
                frame = int(row["stimulus_frame_num"])
                position = index.lookup("live_frames", (frame,))
                require(position is not None, "mapped_frame_unresolved")
                index.add("mapped_frames", (frame,), position, reason="mapped_frame_duplicate")
                recording = int(frames[position]["recording_frame_id"])
                require(
                    int(row["source_recording_frame_id"]) == recording
                    and row["source_recording_frame_valid"]
                    == row["source_acquisition_frame_valid"]
                    == 1
                    and int(row["source_acquisition_frame_index"]) == recording - 1,
                    "mapped_frame_identity_mismatch",
                )
    return CorrespondenceSummary(
        frame_count, counts, binding["recording_id"], binding["camera_serial"]
    )
