"""Orange physical-registration masks in the Citrus H5 loader.

Citrus copies Orange's camera entries into the H5 scope verbatim and embeds the
exact contract, so the H5 path applies the folder loader's rule: a camera's mask
is its physical registration when that is selected_resolved, else its daily one.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

from fisheye.analysis_workflows.materializers import (
    arena_geometry_candidates as candidates,
)
from fisheye.shared.recording_geometry import (
    REGISTRATION_SOURCE_DAILY,
    REGISTRATION_SOURCE_PHYSICAL,
    CitrusRegistrationStatus,
    GeometryLoadPolicy,
    MaskGeometryStatus,
    _physical_cameras,
    _physical_selection,
    load_registered_dish_masks_from_citrus_h5,
)
from tests.unit.fisheye.test_arena_geometry_candidates import _bound_mask
from tests.unit.fisheye.test_recording_geometry import (
    GATE_FIXTURES,
    _json_bytes,
    _rim_entry,
    _sha,
    _write_h5,
)
from tests.unit.fisheye.test_recording_geometry_physical import (
    CAPTURED,
    SERIAL,
    VALID_UNTIL,
    _freshness,
)

REAL_CONTRACT = GATE_FIXTURES / "valid" / "contract_v3_quality_gate_selection.json"


def _scope_for(contract: dict, contract_bytes: bytes) -> dict:
    """The scope Citrus writes (session_logger.cpp BuildH5Scope): cameras verbatim."""

    return {
        "schema_id": "citrus.session.orange_recording_geometry_contract_scope",
        "schema_version": 3,
        "scope_status": "resolved",
        "source_contract": {
            "sha256": _sha(contract_bytes),
            "schema_version": contract["schema_version"],
        },
        "target": {
            "rig_id": "omnifin0",
            "canvas_name": "shadow",
            "arena_id": "arena_1",
            "associated_camera_ids": [SERIAL],
        },
        "cameras": {SERIAL: copy.deepcopy(contract["cameras"][SERIAL])},
    }


def _write_recording_group(h5: h5py.File, contract_bytes: bytes, scope: dict) -> None:
    if "recording_geometry_contract" in h5:
        del h5["recording_geometry_contract"]
    contract = json.loads(contract_bytes)
    recording = h5.create_group("recording_geometry_contract")
    recording.attrs["capture_status"] = "embedded_verified"
    recording.attrs["checksum_verified"] = 1
    recording.attrs["schema_id"] = "orange.recording.geometry_contract"
    recording.attrs["schema_version"] = contract["schema_version"]
    dataset = recording.create_dataset("contract_json", data=contract_bytes)
    dataset.attrs["checksum_sha256"] = _sha(contract_bytes)
    scope_bytes = _json_bytes(scope)
    dataset = recording.create_dataset("h5_scope_json", data=scope_bytes)
    dataset.attrs["checksum_sha256"] = _sha(scope_bytes)
    dataset.attrs["source_contract_sha256"] = _sha(contract_bytes)


def _write_physical_only_h5(path: Path, mutate=None) -> dict:
    """Orange's own physical-only v3 contract, scoped by Citrus. No runtime group:
    Citrus applied no daily registration."""

    contract = json.loads(REAL_CONTRACT.read_bytes())
    if mutate is not None:
        mutate(contract)
    contract_bytes = _json_bytes(contract, newline=True)
    with h5py.File(path, "w") as h5:
        _write_recording_group(h5, contract_bytes, _scope_for(contract, contract_bytes))
    return contract


def _physical_camera() -> dict:
    observation = b'{"artifact_id":"dishrim-physical-2010093"}\n'
    entry = _rim_entry(source_sha=_sha(observation), rim_version=3)
    entry["artifact_id"] = "dishrim-physical-2010093"
    entry["active_in_orange_live_detection_pipeline"] = True
    entry["orange_live_detection_pipeline_mode"] = "centroid_gate"
    return {
        "camera_serial": SERIAL,
        "mode": "selected_physical_registration",
        "status": "selected_resolved",
        "artifact_id": entry["artifact_id"],
        "observation": {"sha256": _sha(observation)},
        "active_pointer": {
            "sha256": "sha256:" + "e" * 64,
            "snapshot": {"schema_version": 2},
        },
        "freshness": _freshness(),
        "recording_snapshot_entry": entry,
    }


def _write_daily_and_physical_h5(path: Path, mutate=None, mutate_scope=None) -> None:
    """A Citrus daily H5 (runtime applied) whose Orange contract also selects a
    physical registration for the same camera, a different observation."""

    _write_h5(path, contract_version=3, scope_version=3)
    with h5py.File(path, "a") as h5:
        scope = json.loads(h5["recording_geometry_contract/h5_scope_json"][()])
    camera = _physical_camera()
    contract = {
        "schema_id": "orange.recording.geometry_contract",
        "schema_version": 3,
        "status": "resolved",
        "captured_at_utc": CAPTURED,
        "physical_registration_geometry": {
            "schema_id": "orange.recording.physical_registration_geometry",
            "schema_version": 1,
            "status": "selected_resolved",
            "captured_at_utc": CAPTURED,
            "cameras": {SERIAL: camera},
        },
        "cameras": {
            SERIAL: {
                "arena_id": "arena_1",
                "camera_serial": SERIAL,
                "physical_registration": copy.deepcopy(camera),
            }
        },
    }
    # Orange chose the physical entry, so the daily one carries no live-use marks.
    daily_entry = scope["cameras"][SERIAL]["daily_registration_geometry"][
        "recording_snapshot_entry"
    ]
    daily_entry.pop("active_in_orange_live_detection_pipeline", None)
    daily_entry.pop("orange_live_detection_pipeline_mode", None)
    scope["cameras"][SERIAL]["physical_registration"] = copy.deepcopy(camera)
    if mutate is not None:
        mutate(contract)
    if mutate_scope is not None:
        mutate_scope(scope)
    contract_bytes = _json_bytes(contract, newline=True)
    scope["source_contract"]["sha256"] = _sha(contract_bytes)
    with h5py.File(path, "a") as h5:
        _write_recording_group(h5, contract_bytes, scope)


def _both(mutate):
    def apply(contract: dict) -> None:
        mutate(contract["physical_registration_geometry"]["cameras"][SERIAL])
        mutate(contract["cameras"][SERIAL]["physical_registration"])

    return apply


def _refused(path: Path, message: str) -> None:
    result = load_registered_dish_masks_from_citrus_h5(path)
    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert result.masks == {}
    assert message in result.issues[0].message, result.issues


@pytest.mark.parametrize(
    "policy", [GeometryLoadPolicy.IF_AVAILABLE, GeometryLoadPolicy.REQUIRED]
)
def test_physical_only_orange_contract_loads_from_the_h5(
    tmp_path: Path, policy
) -> None:
    path = tmp_path / "experiment.h5"
    contract = _write_physical_only_h5(path)

    result = load_registered_dish_masks_from_citrus_h5(path, policy=policy)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID, result.issues
    assert result.issues == ()
    (mask,) = result.masks.values()
    selection = _physical_selection(
        contract, SERIAL, _physical_cameras(contract)[SERIAL]
    )
    assert mask.source_kind == "citrus_h5"
    assert mask.registration_source == REGISTRATION_SOURCE_PHYSICAL
    assert mask.artifact_id == selection.entry["artifact_id"]
    assert mask.registration_id == mask.artifact_id
    assert (
        mask.registration_sha256
        == selection.observation_sha256
        == mask.source_observation_sha256
    )
    assert mask.selection_record_sha256 == selection.active_pointer_sha256
    assert mask.source_valid_until_utc == "2026-11-05T01:00:16Z"
    # Citrus does not apply Orange's physical registration: nothing to reconcile.
    assert mask.citrus_registration_status is CitrusRegistrationStatus.MISSING
    assert mask.selected_daily_registration_applied_by_citrus is None
    assert mask.h5_scope_sha256 is not None
    assert mask.key.arena_id == "arena_1"


def test_physical_selection_is_chosen_over_the_applied_daily(tmp_path: Path) -> None:
    path = tmp_path / "experiment.h5"
    _write_daily_and_physical_h5(path)

    result = load_registered_dish_masks_from_citrus_h5(
        path, policy=GeometryLoadPolicy.REQUIRED
    )

    assert result.mask_geometry_status is MaskGeometryStatus.VALID, result.issues
    (mask,) = result.masks.values()
    assert mask.registration_source == REGISTRATION_SOURCE_PHYSICAL
    assert mask.artifact_id == "dishrim-physical-2010093"
    assert mask.selection_record_sha256 == "sha256:" + "e" * 64
    assert mask.source_valid_until_utc == VALID_UNTIL


def test_unselected_physical_camera_leaves_the_daily_mask(tmp_path: Path) -> None:
    # Cam2010093 on 2026-11-15: the physical pointer expired, so Orange used
    # the daily registration, and Citrus's exact-match claim still applies.
    path = tmp_path / "experiment.h5"

    def expire(camera: dict) -> None:
        camera["status"] = "invalid_selected"
        camera["freshness"]["state"] = "expired"
        camera["recording_snapshot_entry"].pop(
            "active_in_orange_live_detection_pipeline"
        )
        camera["recording_snapshot_entry"].pop("orange_live_detection_pipeline_mode")

    _write_daily_and_physical_h5(
        path,
        mutate=_both(expire),
        mutate_scope=lambda scope: expire(
            scope["cameras"][SERIAL]["physical_registration"]
        ),
    )

    result = load_registered_dish_masks_from_citrus_h5(
        path, policy=GeometryLoadPolicy.REQUIRED
    )

    assert result.mask_geometry_status is MaskGeometryStatus.VALID, result.issues
    (mask,) = result.masks.values()
    assert mask.registration_source == REGISTRATION_SOURCE_DAILY
    assert mask.registration_id == "dailyreg-1"
    assert (
        mask.citrus_registration_status is CitrusRegistrationStatus.EXACT_MATCH_APPLIED
    )


def test_daily_h5_without_a_physical_block_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "experiment.h5"
    _write_h5(path, contract_version=3, scope_version=3)
    (mask,) = load_registered_dish_masks_from_citrus_h5(path).masks.values()
    assert mask.registration_source == REGISTRATION_SOURCE_DAILY
    assert mask.selection_record_sha256 is None
    assert (
        mask.citrus_registration_status is CitrusRegistrationStatus.EXACT_MATCH_APPLIED
    )


def test_scope_copy_must_equal_the_embedded_contract(tmp_path: Path) -> None:
    path = tmp_path / "experiment.h5"
    _write_daily_and_physical_h5(
        path,
        mutate_scope=lambda scope: scope["cameras"][SERIAL][
            "physical_registration"
        ].__setitem__("artifact_id", "dishrim-other"),
    )
    _refused(path, "physical_registration differs from the embedded contract")


@pytest.mark.parametrize(
    "mark",
    ["active_in_orange_live_detection_pipeline", "orange_live_detection_pipeline_mode"],
)
def test_daily_entry_may_not_carry_live_use_when_physical_is_chosen(
    tmp_path: Path, mark: str
) -> None:
    path = tmp_path / "experiment.h5"
    value = (
        True if mark == "active_in_orange_live_detection_pipeline" else "centroid_gate"
    )
    _write_daily_and_physical_h5(
        path,
        mutate_scope=lambda scope: scope["cameras"][SERIAL][
            "daily_registration_geometry"
        ]["recording_snapshot_entry"].__setitem__(mark, value),
    )
    _refused(path, "physical chosen")


def test_unselected_physical_entry_with_live_use_is_refused(tmp_path: Path) -> None:
    # Enforcement correction: the folder loader already refused this.
    path = tmp_path / "experiment.h5"

    def unselect(camera: dict) -> None:
        camera["status"] = "invalid_selected"

    _write_daily_and_physical_h5(
        path,
        mutate=_both(unselect),
        mutate_scope=lambda scope: unselect(
            scope["cameras"][SERIAL]["physical_registration"]
        ),
    )
    _refused(path, "Unselected physical entry")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            _both(lambda cam: cam["freshness"].__setitem__("state", "expired")),
            "not fresh",
        ),
        (
            lambda c: (
                c.__setitem__("captured_at_utc", VALID_UNTIL)
                or c["physical_registration_geometry"].__setitem__(
                    "captured_at_utc", VALID_UNTIL
                )
                or _both(
                    lambda cam: cam["freshness"].__setitem__(
                        "evaluated_at_utc", VALID_UNTIL
                    )
                )(c)
            ),
            "outside its window",
        ),
        (
            lambda c: c["cameras"][SERIAL].__setitem__("physical_registration", {}),
            "physical_registration differs",
        ),
        (
            lambda c: c["physical_registration_geometry"].__setitem__(
                "schema_version", 2
            ),
            "physical_registration_geometry",
        ),
    ],
)
def test_physical_selection_rules_apply_to_the_h5(
    tmp_path: Path, mutate, message: str
) -> None:
    path = tmp_path / "experiment.h5"
    _write_physical_only_h5(path, mutate)
    _refused(path, message)


def test_physical_entry_arena_must_be_the_h5_target(tmp_path: Path) -> None:
    path = tmp_path / "experiment.h5"
    _write_physical_only_h5(
        path,
        _both(
            lambda cam: cam["recording_snapshot_entry"].__setitem__(
                "arena_id", "arena_2"
            )
        ),
    )
    _refused(path, "arena differs from the H5 target arena")


def test_h5_physical_mask_writes_candidate_record_v2(tmp_path: Path) -> None:
    path = tmp_path / "experiment.h5"
    _write_physical_only_h5(path)
    (mask,) = load_registered_dish_masks_from_citrus_h5(
        path, policy=GeometryLoadPolicy.REQUIRED
    ).masks.values()
    record = candidates.build_acquisition_geometry_candidate_record(
        replace(_bound_mask(), mask=mask), recovery_binding=None
    )
    assert record["schema_version"] == 2
    source = record["acquisition_source"]
    assert source["source_kind"] == "citrus_h5"
    assert source["registration_source"] == REGISTRATION_SOURCE_PHYSICAL
    assert source["citrus_registration_status"] == "missing"
    assert source["selected_daily_registration_applied_by_citrus"] is None
    candidates.validate_arena_geometry_candidate_record(record)
