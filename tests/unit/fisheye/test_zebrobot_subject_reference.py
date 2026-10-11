"""Orange's sealed Zebrobot subject references, validated and resolved at intake."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import zarr

from fisheye.shared import zebrobot_subject_reference as zsr
from fisheye.shared.subject_metadata import publish_subject_metadata, resolve_subject_metadata

UUID = "28c29cc6-a1ef-4382-9dc5-414fbb445d92"
BASE = "http://delahantyj-ws1.hhmi.org"
COLLECTED = {
    "schema_id": "orange.recording_subject_reference",
    "schema_version": 1,
    "status": "collected",
    "reason": "",
    "zebrobot": {
        "base_url": BASE,
        "endpoint": "/dishes/19220_1/citrus-snapshot",
        "fish_endpoint": "/dishes/19220_1/fish",
        "api_schema_version": 2,
        "queried_at_utc": "2026-10-05T19:30:00Z",
        "http_status": 200,
        "error": None,
    },
    "dish": {"dish_id": "19220_1", "dish_uuid": UUID, "revision": 1, "updated_at": "2026-10-02 16:28:20"},
    "dish_fish": [],
    "dish_fish_lookup": {"status": "complete", "http_status": 200, "error": None},
    "subject_count": None,
}
NOT_COLLECTED = {
    "schema_id": "orange.recording_subject_reference", "schema_version": 1,
    "status": "not_collected", "reason": "no_dish_declared",
    "zebrobot": None, "dish": None, "dish_fish": [], "dish_fish_lookup": None,
    "subject_count": None,
}
SNAPSHOT = {
    "schema_version": 2, "dish_id": "19220_1", "dish_uuid": UUID, "revision": 1,
    "updated_at": "2026-10-02 16:28:20", "cross_id": "19220", "genotype": "Tg(gfap:b-ARK)",
    "dof": "20260928", "dpf": 7, "fish_count": 32, "species": "Danio rerio", "sex": "unknown",
    "line_strain": "Tg(gfap:b-ARK)",
    "parents": [{"identifier": "8130_M13_A2", "sex": "F"}, {"identifier": "6728_M19_E5", "sex": "M"}],
}


def _fetch(snapshot=SNAPSHOT, fish=(), status=200):
    def fetch(url):
        if url.endswith("/citrus-snapshot"):
            return status, copy.deepcopy(snapshot)
        if url.endswith("/fish"):
            return 200, {"items": [dict(item) for item in fish]}
        raise AssertionError(url)
    return fetch


def test_reference_schema_accepts_the_three_declared_states():
    lookup_failed = copy.deepcopy(COLLECTED)
    lookup_failed.update(
        status="lookup_failed", reason="dish_lookup_transport_failure", dish=None,
        dish_fish_lookup={"status": "not_attempted", "http_status": None, "error": None},
    )
    lookup_failed["zebrobot"].update(
        http_status=None, error={"kind": "transport", "detail_error": None, "message": "timeout"},
    )
    for reference in (COLLECTED, NOT_COLLECTED, lookup_failed):
        assert zsr.validate_subject_reference(reference, "CAM-1")["status"] == reference["status"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(reason="should be empty"),
        lambda r: r["dish"].update(dish_uuid="NOT-A-UUID"),
        lambda r: r["dish"].update(revision=0),
        lambda r: r.update(extra="field"),
        lambda r: r["zebrobot"].update(api_schema_version=1),
        lambda r: r.update(
            dish_fish_lookup={"status": "failed", "http_status": 503, "error": None},
            dish_fish=[{"fish_id": "f1", "revision": 1, "updated_at": "x"}],
        ),
        lambda r: r.update(subject_count=0),
    ],
)
def test_malformed_references_are_refused(mutate):
    reference = copy.deepcopy(COLLECTED)
    mutate(reference)
    with pytest.raises(zsr.SubjectReferenceError):
        zsr.validate_subject_reference(reference, "CAM-1")


def test_every_camera_must_declare_a_reference():
    with pytest.raises(zsr.SubjectReferenceError, match="every camera"):
        zsr.validate_subject_references({"CAM-1": COLLECTED}, ["CAM-1", "CAM-2"])


def test_unchanged_dish_resolves_with_recording_date_dpf():
    resolved = zsr.resolve_subject_reference(COLLECTED, camera="CAM-1", fetch=_fetch())
    metadata = resolved.metadata
    assert resolved.status == "collected"
    assert metadata["dish_changed_since_recording"] is False
    assert metadata["dpf_at_recording"] == 7  # 2026-10-05 recording, dof 2026-09-28
    assert metadata["cross_fields_resolved_at"] == "intake"
    assert "fish_ids" not in metadata and metadata["dish_fish_ids"] == []


def test_changed_dish_and_fish_are_flagged_not_hidden():
    reference = copy.deepcopy(COLLECTED)
    reference["dish_fish"] = [{"fish_id": "f1", "revision": 1, "updated_at": "2026-10-02 16:28:20"}]
    resolved = zsr.resolve_subject_reference(
        reference, camera="CAM-1",
        fetch=_fetch(snapshot={**SNAPSHOT, "revision": 3}, fish=[{"fish_id": "f1", "revision": 2}]),
    )
    assert resolved.metadata["dish_changed_since_recording"] is True
    assert resolved.metadata["dish_revision_at_intake"] == 3
    assert resolved.metadata["dish_fish_changed_since_recording"] == ["f1"]


def test_a_recreated_dish_is_refused():
    with pytest.raises(zsr.SubjectReferenceError, match="re-created"):
        zsr.resolve_subject_reference(
            COLLECTED, camera="CAM-1",
            fetch=_fetch(snapshot={**SNAPSHOT, "dish_uuid": "11111111-1111-4111-8111-111111111111"}),
        )


def test_a_dish_missing_at_intake_is_refused():
    with pytest.raises(zsr.SubjectReferenceError, match="not served"):
        zsr.resolve_subject_reference(COLLECTED, camera="CAM-1", fetch=_fetch(status=404))


def test_unreachable_zebrobot_is_retryable_not_absent():
    with pytest.raises(zsr.ZebrobotUnavailable):
        zsr.http_fetch("http://127.0.0.1:9/dishes/x/citrus-snapshot", timeout=1)


def test_declared_absence_passes_through_without_fetching():
    resolved = zsr.resolve_subject_reference(
        NOT_COLLECTED, camera="CAM-1", fetch=lambda url: pytest.fail("fetched")
    )
    assert (resolved.status, resolved.reason, resolved.metadata) == (
        "not_collected", "no_dish_declared", None,
    )


def _recording(tmp_path: Path, reference) -> object:
    from fisheye.utils import import_recording_analysis as importer

    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "recording_manifest.json").write_text(
        json.dumps({"camera_id": "CAM-1", "zebrobot_subject_reference": reference})
    )
    zarr_path = recording / "zarr" / "rec_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    return importer, importer.RecordingAnalysisPlan(
        recording_dir=recording, h5_path=None, cam_video=None, zarr_path=zarr_path
    )


def test_importer_publishes_recording_only_subject_metadata(tmp_path):
    importer, plan = _recording(tmp_path, COLLECTED)
    result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
    assert result["published"] is True
    root = zarr.open_group(str(plan.zarr_path), mode="r")
    resolved = resolve_subject_metadata(root)
    assert resolved.metadata["dish_uuid"] == UUID
    assert resolved.subject_ids == ()  # dish fish are not recorded subjects
    assert result["experiment_setup"] is False
    assert root.attrs["experiment_setup_status"] == "subject_count_not_declared"


def test_declared_subject_count_also_publishes_experiment_setup(tmp_path):
    importer, plan = _recording(tmp_path, {**COLLECTED, "subject_count": 3})
    result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
    assert result["experiment_setup"] is True
    assert result["expected_subject_count"] == 3


def test_importer_records_declared_absence(tmp_path):
    importer, plan = _recording(tmp_path, NOT_COLLECTED)
    assert importer.import_zebrobot_subject_reference(plan)["status"] == "not_collected"
    attrs = zarr.open_group(str(plan.zarr_path), mode="r").attrs
    assert attrs["orange_subject_reference_status"] == "not_collected"
    assert attrs["orange_subject_reference_reason"] == "no_dish_declared"


def test_importer_cross_checks_h5_dish_identity(tmp_path):
    importer, plan = _recording(tmp_path, COLLECTED)
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    publish_subject_metadata(
        root, {"dish_uuid": "11111111-1111-4111-8111-111111111111", "subject_count": 1},
        source_artifact={"kind": "test_h5"},
    )
    with pytest.raises(ValueError, match="zebrobot_dish_mismatch"):
        importer.import_zebrobot_subject_reference(plan, fetch=_fetch())


def test_organizer_carries_each_cameras_reference(tmp_path):
    from fisheye.utils import organize_transfer_recordings as organizer
    from tests.unit.fisheye.test_transfer_recording_organization import FIXTURES, _resign

    source = Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))
    session_path = source / "recording_session.json"
    session = json.loads(session_path.read_text())
    session["subject_references"] = {camera: COLLECTED for camera in session["cameras"]}
    session_path.write_text(json.dumps(session))
    _resign(source)
    plan = organizer.build_transfer_organization_plan(source, destination_root=tmp_path / "out")
    assert all(p["zebrobot_subject_reference"] == COLLECTED for p in plan["parents"])
    manifest = organizer._parent_manifest(plan, plan["parents"][0])
    assert manifest["zebrobot_subject_reference"]["dish"]["dish_uuid"] == UUID

    del session["subject_references"][session["cameras"][0]]
    session_path.write_text(json.dumps(session))
    _resign(source)
    with pytest.raises(ValueError, match="every camera"):
        organizer.build_transfer_organization_plan(source, destination_root=tmp_path / "out2")


def _v2(**build):
    reference = copy.deepcopy(COLLECTED)
    reference["schema_version"] = 2
    reference["zebrobot"].update(
        {
            "service_commit": "509a3eb8", "service_commit_dirty": False,
            "consumer_schema_sha256": zsr.MZB_PIN["consumer_openapi_sha256"], **build,
        }
    )
    return reference


def test_v2_records_the_serving_build_against_the_pin():
    source = zsr.resolve_subject_reference(_v2(), camera="CAM-1", fetch=_fetch()).source
    assert source["reference_schema_version"] == 2
    assert source["zebrobot_service_commit"] == "509a3eb8"
    assert source["zebrobot_consumer_schema_matches_pin"] is True

    drifted = zsr.resolve_subject_reference(
        _v2(consumer_schema_sha256="0" * 64), camera="CAM-1", fetch=_fetch()
    ).source
    assert drifted["zebrobot_consumer_schema_matches_pin"] is False

    unread = zsr.resolve_subject_reference(
        _v2(service_commit=None, service_commit_dirty=None, consumer_schema_sha256=None),
        camera="CAM-1", fetch=_fetch(),
    ).source
    assert unread["zebrobot_consumer_schema_matches_pin"] is None


def test_v1_reference_carries_no_build_fields():
    source = zsr.resolve_subject_reference(COLLECTED, camera="CAM-1", fetch=_fetch()).source
    assert source["reference_schema_version"] == 1
    assert "zebrobot_consumer_schema_matches_pin" not in source


@pytest.mark.parametrize(
    "reference",
    [
        {**copy.deepcopy(COLLECTED), "schema_version": 2},  # v2 without build fields
        _v2() | {"schema_version": 1},  # v1 with build fields
        {**copy.deepcopy(COLLECTED), "schema_version": 3},
    ],
)
def test_versions_do_not_accept_each_others_shapes(reference):
    with pytest.raises(zsr.SubjectReferenceError):
        zsr.validate_subject_reference(reference, "CAM-1")


def test_a_session_cannot_mix_reference_versions():
    with pytest.raises(zsr.SubjectReferenceError, match="mixes schema versions"):
        zsr.validate_subject_references({"CAM-1": COLLECTED, "CAM-2": _v2()}, ["CAM-1", "CAM-2"])


@pytest.mark.parametrize("h5_dish, ok", [("19220_1", True), ("other_dish", False)])
def test_cross_check_falls_back_to_dish_id_without_h5_uuid(tmp_path, h5_dish, ok):
    # A Citrus lookup that failed records the declared dish_id but no dish_uuid.
    importer, plan = _recording(tmp_path, COLLECTED)
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    publish_subject_metadata(
        root, {"dish_id": h5_dish, "subject_count": 1}, source_artifact={"kind": "test_h5"}
    )
    if ok:
        result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
        assert result["cross_checked"] == "dish_id" and result["published"] is False
    else:
        with pytest.raises(ValueError, match="zebrobot_dish_mismatch: Orange declared dish_id"):
            importer.import_zebrobot_subject_reference(plan, fetch=_fetch())


def test_orange_absence_is_recorded_only_when_nothing_else_is(tmp_path):
    importer, plan = _recording(tmp_path, NOT_COLLECTED)
    importer.import_zebrobot_subject_reference(plan)
    root = zarr.open_group(str(plan.zarr_path), mode="r")
    assert resolve_subject_metadata(root).subject == {
        "subject_lookup": {"status": "not_collected", "reason": "no_dish_declared"}}

    (tmp_path / "bound").mkdir()
    importer, plan = _recording(tmp_path / "bound", NOT_COLLECTED)
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    h5 = publish_subject_metadata(root, {"dish_id": "19220_1", "subject_count": 1},
                                  source_artifact={"kind": "test_h5"}, translator="h5_attributes")
    importer.import_zebrobot_subject_reference(plan)
    assert resolve_subject_metadata(root).run_name == h5.run_name  # the H5 record stays


def test_orange_supplies_the_dish_when_the_h5_declared_none(tmp_path):
    importer, plan = _recording(tmp_path, {**COLLECTED, "subject_count": 1})
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    publish_subject_metadata(
        root, {"subject_lookup_status": "not_collected", "subject_lookup_reason": "no_dish_declared"},
        source_artifact={"kind": "test_h5"}, translator="declared_absence",
    )
    result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
    assert result["published"] is True
    subject = resolve_subject_metadata(root).subject
    assert subject["dish_uuid"] == UUID and subject["dpf_at_acquisition"] == 7
    assert subject["subject_lookup"] == {"status": "collected", "reason": ""}


@pytest.mark.parametrize("revision, refused", [(1, True), (2, False)])
def test_h5_and_zebrobot_must_agree_on_sex_unless_the_dish_was_edited(tmp_path, revision, refused):
    importer, plan = _recording(tmp_path, COLLECTED)
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    publish_subject_metadata(
        root, {"dish_id": "19220_1", "dish_uuid": UUID, "sex": "M", "subject_count": 1},
        source_artifact={"kind": "test_h5"}, translator="h5_attributes",
    )
    fetch = _fetch(snapshot={**SNAPSHOT, "revision": revision})  # served sex "unknown"
    if refused:
        with pytest.raises(ValueError, match="zebrobot_subject_mismatch: sex H5='M'"):
            importer.import_zebrobot_subject_reference(plan, fetch=fetch)
    else:
        result = importer.import_zebrobot_subject_reference(plan, fetch=fetch)
        assert result["biology_differs_after_dish_edit"] == ["sex"]


@pytest.mark.parametrize("orange_count", [None, 1])
def test_h5_ids_without_a_dish_stay_authoritative_over_orange(tmp_path, orange_count):
    # A legacy / pre-contract H5 declares subject_count and ids but no dish.
    # The H5 record wins: its ids are kept and the setup still resolves.
    from fisheye.shared.experiment_setup import resolve_experiment_setup

    importer, plan = _recording(tmp_path, {**COLLECTED, "subject_count": orange_count})
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    h5 = importer._publish_subject_and_setup(
        root, {"subject_ids": ["fish-a", "fish-b"], "subject_count": 2},
        source_artifact={"kind": "test_h5", "count_field": "subject_count"},
        translator="h5_attributes",
    )
    result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
    assert result["published"] is False
    root = zarr.open_group(str(plan.zarr_path), mode="r")
    subject = resolve_subject_metadata(root, allow_legacy=False)
    assert subject.run_name == h5["subject_metadata_run"]
    assert subject.subject_ids == ("fish-a", "fish-b")
    assert len(list(root["analysis/subject_metadata_runs"].group_keys())) == 1
    setup = resolve_experiment_setup(root, allow_legacy=False)
    assert setup.run_name == h5["run_name"] and setup.expected_subject_count == 2


def test_orange_fills_in_when_the_h5_declared_no_subject_record(tmp_path):
    from fisheye.shared.experiment_setup import resolve_experiment_setup

    importer, plan = _recording(tmp_path, {**COLLECTED, "subject_count": 1})
    result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
    assert result["published"] is True
    root = zarr.open_group(str(plan.zarr_path), mode="r")
    assert resolve_subject_metadata(root, allow_legacy=False).subject["dish_uuid"] == UUID
    assert resolve_experiment_setup(root, allow_legacy=False).expected_subject_count == 1


def test_h5_dish_record_is_kept_over_orange(tmp_path):
    importer, plan = _recording(tmp_path, {**COLLECTED, "subject_count": 1})
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    h5 = importer._publish_subject_and_setup(
        root, {"dish_id": "19220_1", "dish_uuid": UUID, "subject_id": "fish-a", "subject_count": 1},
        source_artifact={"kind": "test_h5", "count_field": "subject_count"},
        translator="h5_attributes",
    )
    result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
    assert result["published"] is False and result["cross_checked"] == "dish_uuid"
    root = zarr.open_group(str(plan.zarr_path), mode="r")
    assert resolve_subject_metadata(root, allow_legacy=False).run_name == h5["subject_metadata_run"]


H5_OTHER_DISH = {"dish_id": "19220_2", "dish_uuid": "11111111-1111-4111-8111-111111111111",
                 "subject_count": 1}


def _with_start_snapshot(plan, reference, *, configured, tamper=False):
    """Seal a recording_snapshot_start.json into the parent's transfer mapping."""

    snapshot = {
        "session": {
            "subject_references": {"CAM-1": reference},
            "effective_configuration": {"app_config": {
                "status": "loaded", "path": "/rig/app/default.json", "sha256": "a" * 64,
                "size_bytes": 1, "error": None,
                "contents": {"recording": {"subject_references": configured}},
            }},
        }
    }
    relative = "raw/acquisition/recording_snapshot_start.json"
    path = plan.recording_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(snapshot).encode()
    path.write_bytes(data)
    sealed_sha = hashlib.sha256(data).hexdigest()
    if tamper:
        path.write_bytes(data + b" ")
    manifest_path = plan.recording_dir / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_transfer"] = {"source_to_parent_files": [{
        "relative_path": relative,
        "source": {"path": "recording_snapshot_start.json", "sha256": sealed_sha,
                   "size_bytes": len(data)},
    }]}
    manifest_path.write_text(json.dumps(manifest))


def _h5_dish_recording(tmp_path, reference):
    importer, plan = _recording(tmp_path, reference)
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    publish_subject_metadata(root, H5_OTHER_DISH, source_artifact={"kind": "test_h5"})
    return importer, plan


DEFAULT_ONLY = {"schema_version": 1, "default": {"dish_id": "19220_1", "subject_count": 1}}


def test_v2_app_config_default_dish_disagreement_is_recorded_not_refused(tmp_path):
    importer, plan = _h5_dish_recording(tmp_path, _v2())
    _with_start_snapshot(plan, _v2(), configured=DEFAULT_ONLY)
    result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
    assert result["published"] is False
    assert result["orange_reference_origin"] == "app_config_default"
    disagreement = result["dish_disagreement"]
    assert disagreement["h5"] == H5_OTHER_DISH["dish_uuid"] and disagreement["orange"] == UUID
    attrs = zarr.open_group(str(plan.zarr_path), mode="r").attrs
    assert attrs["orange_subject_reference_dish_disagreement"] == disagreement
    assert attrs["orange_subject_reference_origin"]["orange_app_config_sha256"] == "a" * 64
    # The H5 record stays the subject authority.
    root = zarr.open_group(str(plan.zarr_path), mode="r")
    assert resolve_subject_metadata(root).subject["dish_uuid"] == H5_OTHER_DISH["dish_uuid"]


def test_v2_app_config_default_that_agrees_records_no_disagreement(tmp_path):
    importer, plan = _recording(tmp_path, _v2())
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    publish_subject_metadata(root, {"dish_uuid": UUID, "subject_count": 1},
                             source_artifact={"kind": "test_h5"})
    _with_start_snapshot(plan, _v2(), configured=DEFAULT_ONLY)
    result = importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
    assert result["cross_checked"] == "dish_uuid" and "dish_disagreement" not in result
    assert "orange_subject_reference_dish_disagreement" not in zarr.open_group(
        str(plan.zarr_path), mode="r").attrs


@pytest.mark.parametrize(
    "case",
    [
        "per_camera_entry",      # an operator entry for this camera is an observation
        "v1_reference",          # only v2 had the default-only GUI
        "default_names_other",   # the default does not explain the reference
        "snapshot_reference_differs",
        "config_not_loaded",
        "no_start_snapshot",
    ],
)
def test_dish_disagreement_still_refuses_unless_default_is_proven(tmp_path, case):
    reference = COLLECTED if case == "v1_reference" else _v2()
    importer, plan = _h5_dish_recording(tmp_path, reference)
    configured = copy.deepcopy(DEFAULT_ONLY)
    snapshot_reference = reference
    if case == "per_camera_entry":
        configured["cameras"] = {"CAM-1": {"dish_id": "19220_1", "subject_count": 1}}
    elif case == "default_names_other":
        configured["default"]["dish_id"] = "19220_9"
    elif case == "snapshot_reference_differs":
        snapshot_reference = {**reference, "subject_count": 2}
    if case != "no_start_snapshot":
        _with_start_snapshot(plan, snapshot_reference, configured=configured)
    if case == "config_not_loaded":
        manifest = json.loads((plan.recording_dir / "recording_manifest.json").read_text())
        item = manifest["source_transfer"]["source_to_parent_files"][0]
        path = plan.recording_dir / item["relative_path"]
        snapshot = json.loads(path.read_text())
        snapshot["session"]["effective_configuration"]["app_config"]["status"] = "missing"
        data = json.dumps(snapshot).encode()
        path.write_bytes(data)
        item["source"].update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        (plan.recording_dir / "recording_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="zebrobot_dish_mismatch"):
        importer.import_zebrobot_subject_reference(plan, fetch=_fetch())


def test_a_start_snapshot_that_differs_from_the_seal_refuses(tmp_path):
    importer, plan = _h5_dish_recording(tmp_path, _v2())
    _with_start_snapshot(plan, _v2(), configured=DEFAULT_ONLY, tamper=True)
    with pytest.raises(ValueError, match="differs from the sealed inventory"):
        importer.import_zebrobot_subject_reference(plan, fetch=_fetch())
