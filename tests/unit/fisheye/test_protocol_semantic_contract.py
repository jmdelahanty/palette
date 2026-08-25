from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import h5py
import numpy as np
import pytest
import zarr

from fisheye.analysis.import_stimulus_to_zarr import (
    _bind_protocol_semantic_steps,
    _materialize_protocol_execution_index,
    _materialize_protocol_semantic_snapshot,
    _materialize_stimulus_steps,
    _protocol_semantic_storage_state,
    _validate_protocol_frame_correspondence_proxy,
)
from fisheye.shared.protocol_semantic_contract import (
    ProtocolSemanticContractError,
    TRIAL_INDEX_INTEGRITY_PRODUCER,
    read_materialized_protocol_semantic_snapshot,
    read_protocol_semantic_snapshot,
    validate_protocol_semantic_snapshot,
)
from fisheye.shared.protocol_execution_contract import (
    ProtocolExecutionContractError,
    read_materialized_protocol_execution_index,
    read_protocol_execution_index,
    validate_protocol_execution_index,
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _documents(
    modes: tuple[str, ...],
) -> tuple[str, str, str]:
    mode_ids = {"SOLID_BLACK": 4, "CHASER": 12}
    durations = {"SOLID_BLACK": 300.0, "CHASER": 1500.0}
    families = {"SOLID_BLACK": "solid_color", "CHASER": "chaser"}
    semantic_steps = []
    trial_steps = []
    for index, mode in enumerate(modes):
        duration_s = durations[mode]
        parameters = {"color_type_id": 0} if mode == "SOLID_BLACK" else {}
        features = (
            {
                "color_name": "black",
                "resolved_color": {
                    "color_space": "srgb",
                    "rgba8": [0, 0, 0, 255],
                },
            }
            if mode == "SOLID_BLACK"
            else {}
        )
        semantic_steps.append(
            {
                "duration": {
                    "scale": "1e-3",
                    "unit": "s",
                    "value": int(duration_s * 1000),
                },
                "parameters": parameters,
                "post_stimulus_iti": {"scale": "1e-3", "unit": "s", "value": 0},
                "stimulus_mode_id": mode_ids[mode],
            }
        )
        trial_steps.append(
            {
                "duration_s": duration_s,
                "features": features,
                "index_status": "detailed",
                "post_stimulus_iti_s": 0.0,
                "step_index": index,
                "stimulus_family": families[mode],
                "stimulus_mode": mode,
                "stimulus_mode_id": mode_ids[mode],
            }
        )
    semantic_json = _json(
        {
            "identity": {"iti_stimulus_mode_id": 99, "steps": semantic_steps},
            "normalization_policy": "citrus.protocol.semantic.v1",
            "schema_id": "citrus.protocol.semantic",
            "schema_version": 1,
        }
    )
    semantic_hash = "sha256:" + sha256(semantic_json.encode("utf-8")).hexdigest()
    trial_json = _json(
        {
            "normalization_policy": "citrus.protocol.trial_index.v1",
            "protocol_semantic_hash": semantic_hash,
            "schema_id": "citrus.protocol.trial_index",
            "schema_version": 1,
            "steps": trial_steps,
        }
    )
    return semantic_hash, semantic_json, trial_json


def _snapshot(modes: tuple[str, ...]):
    semantic_hash, semantic_json, trial_json = _documents(modes)
    return validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_json,
    )


def _snapshot_v2(modes: tuple[str, ...]):
    semantic_hash, semantic_json, trial_json = _documents(modes)
    trial = json.loads(trial_json)
    trial["schema_version"] = 2
    trial["normalization_policy"] = "citrus.protocol.trial_index.v2"
    trial_json = _json(trial)
    trial_hash = "sha256:" + sha256(trial_json.encode("utf-8")).hexdigest()
    return validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_json,
        trial_index_hash=trial_hash,
        snapshot_schema_version=2,
        snapshot_policy_id="citrus.protocol.snapshot.v2",
    )


def _execution_document(snapshot, *, status: str = "complete") -> tuple[str, str]:
    steps = []
    frame = 10
    for identity in snapshot.steps:
        start = frame
        end = frame + 10
        interval = {
            "start_stimulus_frame_inclusive": start,
            "end_stimulus_frame_exclusive": end,
            "first_camera_frame_id": 100 + start,
            "last_camera_frame_id": 100 + end - 1,
        }
        step = {
            "step_index": identity.step_index,
            "stimulus_mode_id": identity.stimulus_mode_id,
            "completion_status": "completed",
            "end_reason": "completed",
            "interval": interval,
        }
        if identity.stimulus_mode_id == 12:
            pre_end = start + 2
            training_end = end - 2
            step["chaser_phases"] = {
                "chaser_pre": {
                    **interval,
                    "end_stimulus_frame_exclusive": pre_end,
                    "last_camera_frame_id": 100 + pre_end,
                },
                "chaser_training": {
                    **interval,
                    "start_stimulus_frame_inclusive": pre_end,
                    "end_stimulus_frame_exclusive": training_end,
                    "first_camera_frame_id": 100 + pre_end,
                    "last_camera_frame_id": 100 + training_end,
                },
                "chaser_post": {
                    **interval,
                    "start_stimulus_frame_inclusive": training_end,
                    "first_camera_frame_id": 100 + training_end,
                },
            }
        steps.append(step)
        frame = end
    payload = {
        "authoritative_interval_axis": "stimulus_frame_num",
        "camera_frame_role": "correspondence_only",
        "chaser_repositioning_ownership": (
            "before_chaser_post_start_belongs_to_training;at_or_after_belongs_to_post"
        ),
        "policy_id": "citrus.protocol.execution_index.half_open_stimulus_frames.v1",
        "protocol_trial_index_hash": snapshot.trial_index_sha256,
        "schema_id": "citrus.protocol.execution_index",
        "schema_version": 1,
        "status": status,
        "steps": steps,
    }
    text = _json(payload)
    return text, "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def _seed_steps(run: zarr.Group, snapshot) -> zarr.Group:
    steps = run.create_group("steps")
    for identity in snapshot.steps:
        step = steps.create_group(f"step_{identity.step_index}")
        step.attrs.update(
            {
                "step_index": identity.step_index,
                "stimulus_mode_id": identity.stimulus_mode_id,
                "stimulus_mode": identity.stimulus_mode,
                "duration_s": identity.duration_s,
            }
        )
    return steps


def test_valid_one_and_two_step_recipes_have_distinct_exact_identity() -> None:
    one = _snapshot(("CHASER",))
    two = _snapshot(("SOLID_BLACK", "CHASER"))

    assert one.semantic_hash != two.semantic_hash
    assert one.recipe_label == "CHASER"
    assert two.recipe_label == "SOLID_BLACK -> CHASER"
    assert one.steps[0].display_context == "chaser"
    assert two.steps[0].display_context == "solid_black"
    assert two.mode_sequence == ("SOLID_BLACK", "CHASER")


def test_snapshot_v2_requires_and_preserves_producer_trial_index_hash() -> None:
    snapshot = _snapshot_v2(("SOLID_BLACK", "CHASER"))

    assert snapshot.snapshot_schema_version == 2
    assert snapshot.trial_index_schema_version == 2
    assert snapshot.trial_index_integrity_status == TRIAL_INDEX_INTEGRITY_PRODUCER

    with pytest.raises(ProtocolSemanticContractError, match="requires"):
        validate_protocol_semantic_snapshot(
            semantic_hash=snapshot.semantic_hash,
            semantic_json=snapshot.semantic_json,
            trial_index_json=snapshot.trial_index_json,
            snapshot_schema_version=2,
            snapshot_policy_id="citrus.protocol.snapshot.v2",
        )


def test_snapshot_v2_and_execution_index_read_exact_h5_contract(tmp_path: Path) -> None:
    snapshot = _snapshot_v2(("SOLID_BLACK", "CHASER"))
    execution_json, execution_hash = _execution_document(snapshot)
    path = tmp_path / "protocol-v2.h5"
    with h5py.File(path, "w") as h5:
        protocol = h5.create_group("protocol_snapshot")
        protocol.attrs.update(
            {
                "schema_id": "citrus.protocol.snapshot",
                "schema_version": 2,
                "policy_id": "citrus.protocol.snapshot.v2",
                "contract_status": "valid",
            }
        )
        protocol.create_dataset("protocol_semantic_hash", data=snapshot.semantic_hash)
        protocol.create_dataset("protocol_semantic_json", data=snapshot.semantic_json)
        protocol.create_dataset("protocol_trial_index_json", data=snapshot.trial_index_json)
        protocol.create_dataset("protocol_trial_index_hash", data=snapshot.trial_index_sha256)
        execution = h5.create_group("protocol_execution")
        execution.attrs.update(
            {
                "schema_id": "citrus.protocol.execution_index",
                "schema_version": 1,
                "policy_id": (
                    "citrus.protocol.execution_index.half_open_stimulus_frames.v1"
                ),
                "status": "complete",
            }
        )
        execution.create_dataset("execution_index_json", data=execution_json)
        execution.create_dataset("execution_index_hash", data=execution_hash)

    with h5py.File(path, "r") as h5:
        reloaded_snapshot = read_protocol_semantic_snapshot(h5)
        assert reloaded_snapshot is not None
        reloaded_execution = read_protocol_execution_index(
            h5,
            snapshot=reloaded_snapshot,
        )

    assert reloaded_execution.status == "complete"
    assert reloaded_execution.steps[1].chaser_phases is not None
    assert (
        reloaded_execution.steps[1]
        .chaser_phases["chaser_training"]
        .start_stimulus_frame_inclusive
        == 22
    )


def test_execution_index_rejects_camera_axis_authority() -> None:
    snapshot = _snapshot_v2(("CHASER",))
    execution_json, execution_hash = _execution_document(snapshot)
    payload = json.loads(execution_json)
    payload["authoritative_interval_axis"] = "camera_frame_id"
    mutated = _json(payload)
    mutated_hash = "sha256:" + sha256(mutated.encode("utf-8")).hexdigest()

    with pytest.raises(ProtocolExecutionContractError, match="authoritative_interval_axis"):
        validate_protocol_execution_index(
            execution_json=mutated,
            execution_hash=mutated_hash,
            snapshot=snapshot,
        )


def test_semantic_byte_digest_mismatch_fails_closed() -> None:
    semantic_hash, semantic_json, trial_json = _documents(("CHASER",))

    with pytest.raises(ProtocolSemanticContractError, match="bytes do not match"):
        validate_protocol_semantic_snapshot(
            semantic_hash=semantic_hash,
            semantic_json=semantic_json + " ",
            trial_index_json=trial_json,
        )


def test_h5_contract_distinguishes_legacy_absence_from_partial(tmp_path: Path) -> None:
    path = tmp_path / "protocol.h5"
    with h5py.File(path, "w") as h5:
        assert read_protocol_semantic_snapshot(h5) is None
        group = h5.create_group("protocol_snapshot")
        group.create_dataset("protocol_semantic_hash", data=b"sha256:" + b"0" * 64)
        with pytest.raises(ProtocolSemanticContractError, match="partial"):
            read_protocol_semantic_snapshot(h5)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda trial: trial.update({"protocol_semantic_hash": "sha256:" + "0" * 64}), "must be"),
        (lambda trial: trial["steps"][0].update({"step_index": False}), "exact integer"),
        (lambda trial: trial["steps"][0].update({"stimulus_mode": "SOLID_BLACK"}), "stimulus_mode"),
        (lambda trial: trial["steps"][0].update({"duration_s": "1500"}), "JSON number"),
    ],
)
def test_trial_index_binding_and_types_fail_closed(mutation, match: str) -> None:
    semantic_hash, semantic_json, trial_json = _documents(("CHASER",))
    trial = json.loads(trial_json)
    mutation(trial)

    with pytest.raises(ProtocolSemanticContractError, match=match):
        validate_protocol_semantic_snapshot(
            semantic_hash=semantic_hash,
            semantic_json=semantic_json,
            trial_index_json=_json(trial),
        )


@pytest.mark.parametrize(
    "rgba8",
    ([255, 0, 0, 255], [0, 0, 0]),
)
def test_contradictory_or_malformed_black_evidence_fails_closed(rgba8) -> None:
    semantic_hash, semantic_json, trial_json = _documents(
        ("SOLID_BLACK", "CHASER")
    )
    trial = json.loads(trial_json)
    trial["steps"][0]["features"]["resolved_color"]["rgba8"] = rgba8

    with pytest.raises(ProtocolSemanticContractError):
        validate_protocol_semantic_snapshot(
            semantic_hash=semantic_hash,
            semantic_json=semantic_json,
            trial_index_json=_json(trial),
        )


def test_snapshot_payloads_are_deeply_immutable() -> None:
    snapshot = _snapshot(("CHASER",))
    identity = snapshot.semantic_payload["identity"]

    assert isinstance(identity, MappingProxyType)
    assert isinstance(identity["steps"], tuple)
    with pytest.raises(TypeError):
        identity["new"] = "value"


def test_snapshot_materialization_round_trips_exact_arrays_and_step_bindings(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(("SOLID_BLACK", "CHASER"))
    run = zarr.open_group(str(tmp_path / "run.zarr"), mode="w")
    _seed_steps(run, snapshot)

    _bind_protocol_semantic_steps(run, snapshot)
    _materialize_protocol_semantic_snapshot(run, snapshot)

    stored = run["protocol_semantic_snapshot"]
    assert "protocol_semantic_json" not in stored.attrs
    assert "protocol_trial_index_json" not in stored.attrs
    assert np.asarray(stored["protocol_semantic_json_utf8"][:]).tobytes() == (
        snapshot.semantic_json.encode("utf-8")
    )
    assert np.asarray(stored["protocol_trial_index_json_utf8"][:]).tobytes() == (
        snapshot.trial_index_json.encode("utf-8")
    )
    assert run["steps"]["step_0"].attrs["display_context"] == "solid_black"
    assert run["steps"]["step_1"].attrs["display_context"] == "chaser"
    assert _protocol_semantic_storage_state(run, snapshot) == "verified"


def test_v2_materialization_uses_execution_stimulus_axis_and_blocks_acquisition(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot_v2(("SOLID_BLACK", "CHASER"))
    execution_json, execution_hash = _execution_document(snapshot)
    execution = validate_protocol_execution_index(
        execution_json=execution_json,
        execution_hash=execution_hash,
        snapshot=snapshot,
    )
    run = zarr.open_group(str(tmp_path / "run-v2.zarr"), mode="w")
    h5_path = tmp_path / "source-v2.h5"
    with h5py.File(h5_path, "w") as h5:
        _materialize_stimulus_steps(
            run,
            h5=h5,
            events_data=_events(),
            protocol=_authored_protocol(),
            arena_config={},
            metadata=np.asarray([], dtype=np.dtype([])),
            protocol_semantic_snapshot=snapshot,
            protocol_execution_index=execution,
            console=None,
        )
    _materialize_protocol_semantic_snapshot(run, snapshot)
    correspondence = np.asarray(
        [(frame, 1000 + frame // 2) for frame in range(10, 30)],
        dtype=np.dtype(
            [
                ("stimulus_frame_num", np.uint64),
                ("triggering_camera_frame_id", np.uint64),
            ]
        ),
    )
    _materialize_protocol_execution_index(
        run,
        execution,
        frame_metadata=correspondence,
    )

    step = run["steps/step_1"]
    assert "start_camera_frame" not in step.attrs
    assert step.attrs["start_stimulus_frame_inclusive"] == 20
    assert step.attrs["end_stimulus_frame_exclusive"] == 30
    assert step.attrs["camera_frame_role"] == "correspondence_only"
    assert (
        step["execution_phases/chaser_training"].attrs[
            "start_stimulus_frame_inclusive"
        ]
        == 22
    )
    assert run.attrs["protocol_selector_eligibility"] == (
        "blocked_missing_sealed_stimulus_to_acquisition_mapping"
    )
    proxy = run["protocol_execution/frame_correspondence_proxy"]
    assert proxy.attrs["mapping_class"] == "sealed_derived_correspondence_proxy"
    assert proxy.attrs["selector_eligible"] is False
    assert proxy.attrs["coverage_status"] == "complete"
    assert np.asarray(proxy["chaser_phase_id"][:]).tolist()[-10:] == (
        [0, 0, 1, 1, 1, 1, 1, 1, 2, 2]
    )
    _validate_protocol_frame_correspondence_proxy(
        run["protocol_execution"],
        execution=execution,
        frame_metadata=correspondence,
    )
    reloaded_execution = read_materialized_protocol_execution_index(
        run,
        snapshot=snapshot,
    )
    assert reloaded_execution.execution_hash == execution.execution_hash
    assert reloaded_execution.execution_json == execution.execution_json
    proxy["camera_frame_id_correspondence"][0] = 9999
    with pytest.raises(RuntimeError, match="exact source H5 rows"):
        _validate_protocol_frame_correspondence_proxy(
            run["protocol_execution"],
            execution=execution,
            frame_metadata=correspondence,
        )
    reloaded = read_materialized_protocol_semantic_snapshot(run)
    assert reloaded.trial_index_integrity_status == TRIAL_INDEX_INTEGRITY_PRODUCER


def test_binding_validates_all_steps_before_writing_semantic_attrs(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(("SOLID_BLACK", "CHASER"))
    run = zarr.open_group(str(tmp_path / "run.zarr"), mode="w")
    steps = _seed_steps(run, snapshot)
    steps["step_1"].attrs["duration_s"] = 1.0

    with pytest.raises(ValueError, match="does not match"):
        _bind_protocol_semantic_steps(run, snapshot)

    assert "protocol_semantic_status" not in steps.attrs
    assert "protocol_semantic_status" not in steps["step_0"].attrs


def _events(
    *,
    duplicate_start: bool = False,
    wrong_end_mode: bool = False,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("event_name", "S32"),
            ("current_step_index", np.int32),
            ("stimulus_mode_id", np.int32),
            ("camera_frame_id", np.int64),
        ]
    )
    rows = [
        (b"STEP_START", 0, 4, 862),
        (b"STEP_END", 0, 12 if wrong_end_mode else 4, 30863),
        (b"STEP_START", 1, 12, 30864),
        (b"STEP_END", 1, 12, 180864),
    ]
    if duplicate_start:
        rows.insert(1, (b"STEP_START", 0, 4, 900))
    return np.asarray(rows, dtype=dtype)


def _authored_protocol() -> dict[str, object]:
    return {
        "steps": [
            {
                "name": "locator text is not identity",
                "stimulus_mode_str": "SOLID_BLACK",
                "duration_seconds": 300.0,
                "parameters": {"color_type": "Black"},
            },
            {
                "name": "another locator",
                "stimulus_mode_str": "CHASER",
                "duration_seconds": 1500.0,
                "parameters": {},
            },
        ]
    }


def test_materialized_steps_bind_exact_recipe_without_using_names(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(("SOLID_BLACK", "CHASER"))
    run = zarr.open_group(str(tmp_path / "run.zarr"), mode="w")
    h5_path = tmp_path / "source.h5"
    with h5py.File(h5_path, "w") as h5:
        _materialize_stimulus_steps(
            run,
            h5=h5,
            events_data=_events(),
            protocol=_authored_protocol(),
            arena_config={},
            metadata=np.asarray([], dtype=np.dtype([])),
            protocol_semantic_snapshot=snapshot,
            console=None,
        )

    assert run["steps"]["step_0"].attrs["step_name"] == (
        "locator text is not identity"
    )
    assert run["steps"]["step_0"].attrs["display_context"] == "solid_black"
    assert run["steps"]["step_1"].attrs["protocol_semantic_step_index"] == 1


def test_duplicate_modern_step_events_fail_before_writing_steps(tmp_path: Path) -> None:
    snapshot = _snapshot(("SOLID_BLACK", "CHASER"))
    run = zarr.open_group(str(tmp_path / "run.zarr"), mode="w")
    h5_path = tmp_path / "source.h5"
    with h5py.File(h5_path, "w") as h5:
        with pytest.raises(ValueError, match="duplicate boundaries"):
            _materialize_stimulus_steps(
                run,
                h5=h5,
                events_data=_events(duplicate_start=True),
                protocol=_authored_protocol(),
                arena_config={},
                metadata=np.asarray([], dtype=np.dtype([])),
                protocol_semantic_snapshot=snapshot,
                console=None,
            )

    assert "steps" not in run


def test_modern_step_end_mode_mismatch_fails_before_writing_steps(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(("SOLID_BLACK", "CHASER"))
    run = zarr.open_group(str(tmp_path / "run.zarr"), mode="w")
    h5_path = tmp_path / "source.h5"
    with h5py.File(h5_path, "w") as h5:
        with pytest.raises(ValueError, match="STEP_START/STEP_END mode IDs"):
            _materialize_stimulus_steps(
                run,
                h5=h5,
                events_data=_events(wrong_end_mode=True),
                protocol=_authored_protocol(),
                arena_config={},
                metadata=np.asarray([], dtype=np.dtype([])),
                protocol_semantic_snapshot=snapshot,
                console=None,
            )

    assert "steps" not in run
