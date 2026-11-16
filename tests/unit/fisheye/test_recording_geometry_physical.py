"""Orange physical-registration masks in the recording-folder loader (dish_rim_registration_v4)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fisheye.shared.recording_geometry import (
    REGISTRATION_SOURCE_DAILY,
    REGISTRATION_SOURCE_PHYSICAL,
    MaskGeometryStatus,
    RecordingGeometryError,
    _physical_cameras,
    _physical_selection,
    load_registered_dish_masks_from_recording_folder,
)
from tests.unit.fisheye.test_recording_geometry import (
    GATE_FIXTURES,
    _json_bytes,
    _rim_entry,
    _sha,
    _write_folder_bundle,
)

SERIAL = "2010093"
CAPTURED = "2026-11-04T02:00:00Z"
FIT_CREATED = "2026-11-04T01:00:16Z"
VALID_UNTIL = "2026-11-05T01:00:16Z"


def _freshness(**changes: object) -> dict[str, object]:
    freshness: dict[str, object] = {
        "rule": "orange_physical_registration_freshness_24h_v1",
        "selection_policy": "explicit_operator_selection",
        "fit_created_utc": FIT_CREATED,
        "valid_from_utc": FIT_CREATED,
        "valid_until_utc": VALID_UNTIL,
        "evaluated_at_utc": CAPTURED,
        "state": "fresh",
    }
    freshness.update(changes)
    return freshness


def _write_physical_bundle(
    root: Path,
    *,
    daily_selected: bool = True,
    freshness: dict[str, object] | None = None,
    pointer_snapshot: dict[str, object] | None = None,
    captured: str = CAPTURED,
) -> None:
    """A daily bundle plus a selected_resolved physical camera for Cam2010093.

    The physical observation is a different artifact from the daily one, as
    Orange allows, and the snapshot's calibrations name it, as Orange writes.
    """

    _write_folder_bundle(root, contract_version=2)
    assets = root / "recording_geometry_assets"
    prefix = f"cameras/Cam{SERIAL}/physical_registration/"
    observation_bytes = b'{"artifact_id":"dishrim-physical-2010093"}\n'
    pointer_bytes = b'{"schema_version":2,"selection":"physical"}\n'
    (assets / prefix).mkdir(parents=True)
    (assets / (prefix + "observation.json")).write_bytes(observation_bytes)
    (assets / (prefix + "active_pointer.json")).write_bytes(pointer_bytes)
    manifest = json.loads((assets / "manifest.json").read_bytes())
    for name, role, payload in (
        ("observation.json", "physical_registration_observation", observation_bytes),
        ("active_pointer.json", "physical_registration_active_pointer", pointer_bytes),
    ):
        manifest["files"].append(
            {
                "relative_path": prefix + name,
                "role": role,
                "required": True,
                "size_bytes": len(payload),
                "sha256": _sha(payload),
            }
        )
    manifest["materialized_file_count"] = len(manifest["files"])
    manifest_bytes = _json_bytes(manifest)
    (assets / "manifest.json").write_bytes(manifest_bytes)

    entry = _rim_entry(source_sha=_sha(observation_bytes), rim_version=3)
    entry["artifact_id"] = "dishrim-physical-2010093"
    entry["active_in_orange_live_detection_pipeline"] = False
    entry["orange_live_detection_pipeline_mode"] = "off"
    camera = {
        "camera_serial": SERIAL,
        "mode": "selected_physical_registration",
        "status": "selected_resolved",
        "artifact_id": entry["artifact_id"],
        "observation": {"sha256": _sha(observation_bytes)},
        "active_pointer": {
            "sha256": _sha(pointer_bytes),
            "snapshot": pointer_snapshot or {"schema_version": 2},
        },
        "freshness": freshness or _freshness(),
        "recording_snapshot_entry": entry,
    }
    contract_path = root / "recording_geometry_contract.json"
    contract = json.loads(contract_path.read_bytes())
    contract["captured_at_utc"] = captured
    contract["physical_registration_geometry"] = {
        "schema_id": "orange.recording.physical_registration_geometry",
        "schema_version": 1,
        "status": "selected_resolved",
        "captured_at_utc": captured,
        "cameras": {SERIAL: camera},
    }
    contract["cameras"] = {
        SERIAL: {
            "arena_id": "arena_1",
            "camera_serial": SERIAL,
            "physical_registration": copy.deepcopy(camera),
        }
    }
    contract["materialized_assets"]["sha256"] = _sha(manifest_bytes)
    if not daily_selected:
        contract["daily_registration_geometry"] = {"mode": "not_selected", "status": "not_performed"}
    contract_bytes = _json_bytes(contract, newline=True)
    contract_path.write_bytes(contract_bytes)

    snapshot_path = root / "recording_snapshot.json"
    snapshot = json.loads(snapshot_path.read_bytes())
    snapshot["calibrations"][SERIAL]["dish_top_rim_observation"] = {
        "artifact_id": entry["artifact_id"],
        "sha256": _sha(observation_bytes),
    }
    snapshot["recording_geometry_contract"]["sha256"] = _sha(contract_bytes)
    snapshot_path.write_bytes(_json_bytes(snapshot))


def _rewrite(root: Path, mutate) -> None:
    contract_path = root / "recording_geometry_contract.json"
    contract = json.loads(contract_path.read_bytes())
    mutate(contract)
    payload = _json_bytes(contract, newline=True)
    contract_path.write_bytes(payload)
    snapshot_path = root / "recording_snapshot.json"
    snapshot = json.loads(snapshot_path.read_bytes())
    snapshot["recording_geometry_contract"]["sha256"] = _sha(payload)
    snapshot_path.write_bytes(_json_bytes(snapshot))


def _both_copies(mutate):
    def apply(contract: dict) -> None:
        mutate(contract["physical_registration_geometry"]["cameras"][SERIAL])
        mutate(contract["cameras"][SERIAL]["physical_registration"])

    return apply


@pytest.mark.parametrize("daily_selected", [True, False], ids=["with_daily", "recording_only"])
def test_physical_selection_is_chosen_over_daily(tmp_path: Path, daily_selected: bool) -> None:
    _write_physical_bundle(tmp_path, daily_selected=daily_selected)

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID, result.issues
    (mask,) = result.masks.values()
    assert mask.registration_source == REGISTRATION_SOURCE_PHYSICAL
    assert mask.artifact_id == "dishrim-physical-2010093"
    assert mask.registration_id == "dishrim-physical-2010093"
    assert mask.registration_sha256 == mask.source_observation_sha256
    assert mask.selection_record_sha256 is not None
    assert mask.source_valid_until_utc == VALID_UNTIL
    assert mask.key.arena_id == "arena_1"
    assert result.enclosing_selection_status == "selected_resolved"


def test_daily_masks_keep_their_registration_source(tmp_path: Path) -> None:
    _write_folder_bundle(tmp_path)
    (mask,) = load_registered_dish_masks_from_recording_folder(tmp_path).masks.values()
    assert mask.registration_source == REGISTRATION_SOURCE_DAILY
    assert mask.selection_record_sha256 is None
    assert mask.registration_id == "dailyreg-1"


def test_unselected_physical_camera_leaves_the_daily_mask(tmp_path: Path) -> None:
    # Orange's 2026-11-15 state: the physical pointer expired, so Orange falls
    # back to the daily registration and so does Palette.
    _write_physical_bundle(tmp_path)

    def expire(camera: dict) -> None:
        camera["status"] = "invalid_selected"
        camera["freshness"]["state"] = "expired"
        camera["recording_snapshot_entry"].pop("orange_live_detection_pipeline_mode")

    _rewrite(tmp_path, _both_copies(expire))
    snapshot_path = tmp_path / "recording_snapshot.json"
    snapshot = json.loads(snapshot_path.read_bytes())
    contract = json.loads((tmp_path / "recording_geometry_contract.json").read_bytes())
    daily_entry = contract["daily_registration_geometry"]["cameras"][SERIAL]["recording_snapshot_entry"]
    snapshot["calibrations"][SERIAL]["dish_top_rim_observation"] = {
        "artifact_id": daily_entry["artifact_id"],
        "sha256": daily_entry["source"]["sha256"],
    }
    snapshot_path.write_bytes(_json_bytes(snapshot))

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID, result.issues
    (mask,) = result.masks.values()
    assert mask.registration_source == REGISTRATION_SOURCE_DAILY


def _refused(tmp_path: Path, message: str) -> None:
    result = load_registered_dish_masks_from_recording_folder(tmp_path)
    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert result.masks == {}
    assert message in result.issues[0].message


@pytest.mark.parametrize(
    ("freshness", "captured", "message"),
    [
        # Captured before the window opened (Orange 6ea52cfc now refuses this too).
        (_freshness(evaluated_at_utc="2026-11-04T01:00:15Z"), "2026-11-04T01:00:15Z", "outside its window"),
        # valid_until is exclusive.
        (_freshness(evaluated_at_utc=VALID_UNTIL), VALID_UNTIL, "outside its window"),
        (_freshness(valid_until_utc="2026-11-05T01:00:17Z"), CAPTURED, "not exactly 24 h"),
        (_freshness(state="expired"), CAPTURED, "not fresh"),
        (_freshness(evaluated_at_utc="2026-11-04T02:00:01Z"), CAPTURED, "not evaluated at the recording's captured_at_utc"),
        (_freshness(valid_from_utc="2026-11-04T01:30:00Z", valid_until_utc="2026-11-05T01:30:00Z"), CAPTURED, "does not start at the fit's creation"),
        (_freshness(rule="orange_physical_registration_freshness_48h_v1"), CAPTURED, "freshness rule"),
        (_freshness(selection_policy="automatic"), CAPTURED, "selection_policy"),
        (_freshness(valid_from_utc="2026-11-04T01:00:16.5Z"), CAPTURED, "whole-second UTC"),
    ],
)
def test_physical_freshness_window_is_enforced(
    tmp_path: Path, freshness: dict, captured: str, message: str
) -> None:
    _write_physical_bundle(tmp_path, freshness=freshness, captured=captured)
    _refused(tmp_path, message)


def test_reuse_window_starts_at_the_reselection(tmp_path: Path) -> None:
    reselected = "2026-11-04T01:30:00Z"
    freshness = _freshness(
        selection_policy="explicit_operator_reuse",
        valid_from_utc=reselected,
        valid_until_utc="2026-11-05T01:30:00Z",
    )
    snapshot = {
        "schema_version": 2,
        "selected_at_utc": reselected,
        "freshness": {"valid_from_utc": reselected},
    }
    _write_physical_bundle(tmp_path, freshness=freshness, pointer_snapshot=snapshot)
    result = load_registered_dish_masks_from_recording_folder(tmp_path)
    assert result.mask_geometry_status is MaskGeometryStatus.VALID, result.issues

    snapshot["selected_at_utc"] = "2026-11-04T01:29:59Z"
    _write_physical_bundle(tmp_path / "moved", freshness=freshness, pointer_snapshot=snapshot)
    _refused(tmp_path / "moved", "reuse window does not start at its re-selection")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda c: c["cameras"][SERIAL]["physical_registration"].__setitem__("status", "invalid_selected"),
            "differs between cameras.<serial>.physical_registration",
        ),
        (
            _both_copies(lambda cam: cam.__setitem__("mode", "not_selected")),
            "not selected_physical_registration",
        ),
        (
            _both_copies(lambda cam: cam["observation"].__setitem__("sha256", "sha256:" + "0" * 64)),
            "observation sha256 differs",
        ),
        (
            _both_copies(lambda cam: cam["active_pointer"].__setitem__("sha256", "sha256:" + "0" * 64)),
            "active_pointer.json checksum",
        ),
        (
            _both_copies(lambda cam: cam.__setitem__("artifact_id", "dishrim-other")),
            "artifact_id differs",
        ),
        (
            lambda c: c["physical_registration_geometry"].__setitem__("captured_at_utc", "2026-11-04T02:00:01Z"),
            "not evaluated at the recording's captured_at_utc",
        ),
        (
            lambda c: c["physical_registration_geometry"].__setitem__("schema_version", 2),
            "physical_registration_geometry",
        ),
        (
            lambda c: c["cameras"][SERIAL].__setitem__("arena_id", "arena_2"),
            "arena_id differs",
        ),
    ],
)
def test_physical_selection_must_bind(tmp_path: Path, mutate, message: str) -> None:
    _write_physical_bundle(tmp_path)
    _rewrite(tmp_path, mutate)
    _refused(tmp_path, message)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda c: c["daily_registration_geometry"]["cameras"][SERIAL]["recording_snapshot_entry"].__setitem__(
                "active_in_orange_live_detection_pipeline", True
            ),
            "physical chosen",
        ),
        (
            lambda c: c["daily_registration_geometry"]["cameras"][SERIAL]["recording_snapshot_entry"].__setitem__(
                "orange_live_detection_pipeline_mode", "centroid_gate"
            ),
            "physical chosen",
        ),
    ],
)
def test_only_the_chosen_entry_carries_live_use(tmp_path: Path, mutate, message: str) -> None:
    _write_physical_bundle(tmp_path)
    _rewrite(tmp_path, mutate)
    _refused(tmp_path, message)


def test_unselected_physical_entry_with_live_use_is_refused(tmp_path: Path) -> None:
    _write_physical_bundle(tmp_path)

    def unselect(camera: dict) -> None:
        camera["status"] = "invalid_selected"
        camera["recording_snapshot_entry"]["active_in_orange_live_detection_pipeline"] = True

    _rewrite(tmp_path, _both_copies(unselect))
    _refused(tmp_path, "Unselected physical entry")


def test_snapshot_must_name_the_physical_observation(tmp_path: Path) -> None:
    _write_physical_bundle(tmp_path)
    snapshot_path = tmp_path / "recording_snapshot.json"
    snapshot = json.loads(snapshot_path.read_bytes())
    snapshot["calibrations"][SERIAL]["dish_top_rim_observation"]["artifact_id"] = "dishrim-2010093"
    snapshot_path.write_bytes(_json_bytes(snapshot))
    _refused(tmp_path, "rim artifact_id disagrees")


def test_materialized_physical_observation_must_match(tmp_path: Path) -> None:
    _write_physical_bundle(tmp_path)
    (
        tmp_path / "recording_geometry_assets" / f"cameras/Cam{SERIAL}/physical_registration/observation.json"
    ).write_bytes(b'{"artifact_id":"dishrim-physical-2010094"}\n')  # same size, one byte changed
    _refused(tmp_path, "checksum mismatch")


@pytest.mark.parametrize(
    "name", ["contract_v3_operator_v3_observation.json", "contract_v3_quality_gate_selection.json"]
)
def test_contract_fixtures_pass_the_physical_selection_rules(name: str) -> None:
    contract = json.loads((GATE_FIXTURES / "valid" / name).read_bytes())
    cameras = _physical_cameras(contract)
    selection = _physical_selection(contract, SERIAL, cameras[SERIAL])
    assert selection.valid_until_utc == "2026-11-05T01:00:16Z"
    assert selection.arena_id == "arena_1"


@pytest.mark.parametrize(
    "name", ["contract_camera_copy_freshness_state_unknown.json", "contract_freshness_state_unknown.json"]
)
def test_contract_fixtures_with_unknown_freshness_are_refused(name: str) -> None:
    contract = json.loads((GATE_FIXTURES / "invalid" / name).read_bytes())
    cameras = _physical_cameras(contract)
    with pytest.raises(RecordingGeometryError):
        _physical_selection(contract, SERIAL, cameras[SERIAL])
