"""Closed-inventory organization must account for every staged component."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import h5py
import pytest

from fisheye.shared import recording_transfer_snapshot as transfer
from fisheye.utils import organize_transfer_recordings as organizer

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"
CONTEXT = {
    "recording_type": "behavior",
    "recording_subtype": "free",
    "behavior_mode": "free",
}


def _source(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))


def _bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _resign(root: Path) -> None:
    snapshot = transfer.build_snapshot(root, transfer.MARKER_NAME, destination=True)
    data = transfer.canonical_bytes(snapshot)
    (root / transfer.SNAPSHOT_PATH).write_bytes(data)
    path = root / transfer.MARKER_NAME
    marker = json.loads(path.read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    marker["snapshot_id"] = "sha256:" + digest
    marker["snapshot"].update(size_bytes=len(data), sha256=digest)
    marker["recording_payload_kind"] = snapshot["recording_payload_kind"]
    path.write_bytes(transfer.canonical_bytes(marker))


def _plan(source: Path, destination: Path, **context):
    return organizer.build_transfer_organization_plan(
        source, destination_root=destination, **{**CONTEXT, **context}
    )


def test_plan_covers_every_file_and_does_not_write(tmp_path):
    source = _source(tmp_path)
    opaque = source / "unfamiliar_session_artifact.bin"
    opaque.write_bytes(b"not disposable staging debris")
    _resign(source)
    before = _bytes(source)
    destination = tmp_path / "recordings"
    plan = _plan(source, destination)
    assert not destination.exists()
    assert _bytes(source) == before
    assert plan["status"] == "organization_planned"
    assert plan["staging_finalized"] is False
    assert plan["import_complete"] is False
    assert {item["source"]["path"] for item in plan["files"]} == set(before)
    assert len(plan["files"]) == len(before)
    assert len(plan["parents"]) == 2
    assert {parent["identity"]["camera_id"] for parent in plan["parents"]} == {
        "02010093",
        "02010094",
    }
    by_camera = {p["identity"]["camera_id"]: p for p in plan["parents"]}
    for item in plan["files"]:
        assert item["destinations"]
        assert item["source"] == transfer.file_ref(source, item["source"]["path"])
        for target in item["destinations"]:
            relative = target["relative_path"]
            assert not Path(relative).is_absolute()
            assert relative.endswith(item["source"]["path"])
        if item["role"] == "camera_output":
            assert len(item["destinations"]) == 1
            owner = by_camera[item["camera_id"]]["identity"]["recording_id"]
            assert item["destinations"][0]["recording_id"] == owner
        else:
            assert len(item["destinations"]) == 2
    paths = [
        (target["recording_id"], target["relative_path"])
        for item in plan["files"]
        for target in item["destinations"]
    ]
    assert len(paths) == len(set(paths))


def test_transfer_namespace_control_files_are_preserved_and_retired(
    tmp_path, monkeypatch
):
    source = _source(tmp_path)
    (source / "_citrus_transfer/transfer.lock").write_bytes(b"")
    (source / "_citrus_transfer/producer_diagnostic.json").write_bytes(
        b'{"note":"original"}'
    )
    before = _bytes(source)
    plan = _plan(source, tmp_path / "recordings")
    assert {item["source"]["path"] for item in plan["files"]} == set(before)
    organizer.prepare_transfer_parent_recordings(plan)
    monkeypatch.setattr(
        organizer,
        "_verify_parent_imports",
        lambda *a, **k: {"stub": "admission-mechanics-only"},
    )
    organizer.finalize_transfer_staging(plan)
    assert not list(source.iterdir())
    for parent in plan["parents"]:
        root = Path(parent["destination_dir"])
        for relative in (
            "_citrus_transfer/transfer.lock",
            "_citrus_transfer/producer_diagnostic.json",
        ):
            assert (root / "raw/acquisition" / relative).read_bytes() == before[
                relative
            ]


@pytest.mark.parametrize("field", list(CONTEXT))
@pytest.mark.parametrize("value", [None, "", "invented", 17])
def test_context_is_explicit_and_validated(tmp_path, field, value):
    source = _source(tmp_path)
    destination = tmp_path / "recordings"
    with pytest.raises(ValueError, match="manifest context"):
        _plan(source, destination, **{field: value})
    assert not destination.exists()


@pytest.mark.parametrize("target", ["source", "child", "ancestor", "symlink"])
def test_output_scope_refuses_source_overlap_and_symlinks(tmp_path, target):
    source = _source(tmp_path)
    destination = {
        "source": source,
        "child": source / "recordings",
        "ancestor": tmp_path,
        "symlink": tmp_path / "link" / "recordings",
    }[target]
    if target == "symlink":
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    before = _bytes(source)
    with pytest.raises(ValueError):
        _plan(source, destination)
    assert _bytes(source) == before


def test_failed_optional_proof_is_refused_not_discarded(tmp_path):
    source = Path(
        shutil.copytree(FIXTURES / "failed_optional_proof", tmp_path / "source")
    )
    with pytest.raises(ValueError, match="frame_identity_proof.*failed"):
        _plan(source, tmp_path / "recordings")


@pytest.mark.parametrize(
    "camera,session",
    [
        ("02010093", "fixture-session"),
        ("2010093", "fixture-session"),
        ("02010093", "different-session"),
        (None, "fixture-session"),
    ],
)
def test_h5_mapping_uses_exact_context_not_filename(tmp_path, camera, session):
    source = _source(tmp_path)
    h5_path = source / "unhelpful_filename.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["session_uuid"] = session
        if camera is not None:
            h5.attrs["camera_id"] = camera
        h5.attrs["rig_id"] = "synthetic-rig"
    _resign(source)
    if camera != "02010093" or session != "fixture-session":
        with pytest.raises(ValueError, match="H5"):
            _plan(source, tmp_path / "recordings")
    else:
        plan = _plan(source, tmp_path / "recordings")
        item = next(
            item for item in plan["files"] if item["source"]["path"] == h5_path.name
        )
        assert item["role"] == "camera_h5"
        assert len(item["destinations"]) == 1
        parent = next(
            p for p in plan["parents"] if p["identity"]["camera_id"] == camera
        )
        assert parent["h5_relative_path"] == "raw/acquisition/unhelpful_filename.h5"
        assert parent["producer_context"]["rig_id"] == "synthetic-rig"


def test_duplicate_h5_for_one_parent_refuses_ambiguity(tmp_path):
    source = _source(tmp_path)
    for name in ("one.h5", "two.hdf5"):
        with h5py.File(source / name, "w") as h5:
            h5.attrs["camera_id"] = "02010093"
            h5.attrs["session_uuid"] = "fixture-session"
    _resign(source)
    with pytest.raises(ValueError, match="multiple H5"):
        _plan(source, tmp_path / "recordings")


def test_plan_digest_binds_every_destination_and_context(tmp_path):
    source = _source(tmp_path)
    first = _plan(source, tmp_path / "recordings")
    assert first == _plan(source, tmp_path / "recordings")
    assert first["plan_sha256"] != _plan(source, tmp_path / "elsewhere")["plan_sha256"]
    changed = _plan(
        source,
        tmp_path / "recordings",
        recording_subtype="embedded",
        behavior_mode="embedded",
    )
    assert changed["snapshot_id"] == first["snapshot_id"]
    assert changed["plan_sha256"] != first["plan_sha256"]


@pytest.mark.parametrize("tampered", [False, True])
def test_existing_geometry_bundle_owner_validates_and_preserves_all_members(
    tmp_path, tampered
):
    from tests.unit.fisheye.test_recording_geometry import _write_folder_bundle

    source = _source(tmp_path)
    _write_folder_bundle(source)
    if tampered:
        next(
            (source / "recording_geometry_assets").rglob("observation.json")
        ).write_bytes(b"changed")
    _resign(source)
    if tampered:
        with pytest.raises(ValueError):
            _plan(source, tmp_path / "recordings")
    else:
        plan = _plan(source, tmp_path / "recordings")
        geometry = [item for item in plan["files"] if item["role"] == "geometry_bundle"]
        assert geometry
        assert all(
            target["relative_path"].startswith("raw/recording_geometry_bundle/")
            for item in geometry
            for target in item["destinations"]
        )
        assert all(
            parent["recording_geometry_bundle"]["verification_status"] == "verified"
            for parent in plan["parents"]
        )


def test_materialize_preserves_all_bytes_and_replays_owned_destinations(tmp_path):
    source = _source(tmp_path)
    before = _bytes(source)
    plan = _plan(source, tmp_path / "recordings")
    first = organizer.materialize_transfer_organization(plan)
    assert first["status"] == "materialized"
    assert first["staging_finalized"] is False
    assert _bytes(source) == before
    for item in plan["files"]:
        for target in item["destinations"]:
            path = (
                Path(plan["destination_root"])
                / target["recording_id"]
                / target["relative_path"]
            )
            assert path.read_bytes() == before[item["source"]["path"]]
    assert organizer.materialize_transfer_organization(plan) == first


@pytest.mark.parametrize(
    "mutation", ["source", "plan", "unowned", "parent_replaced", "target_changed"]
)
def test_materialization_refuses_conflicting_evidence_and_never_retires_source(
    tmp_path, mutation
):
    source = _source(tmp_path)
    plan = _plan(source, tmp_path / "recordings")
    parent = Path(plan["parents"][0]["destination_dir"])
    if mutation in {"parent_replaced", "target_changed"}:
        organizer.materialize_transfer_organization(plan)
        if mutation == "parent_replaced":
            parent.rename(parent.with_name(parent.name + ".prior"))
            parent.mkdir()
        else:
            target = next(item["destinations"][0] for item in plan["files"])
            path = (
                Path(plan["destination_root"])
                / target["recording_id"]
                / target["relative_path"]
            )
            path.unlink()
            path.write_bytes(b"wrong destination bytes")
    elif mutation == "unowned":
        parent.mkdir(parents=True)
        (parent / "someone_elses_work").write_bytes(b"preserve")
    elif mutation == "source":
        (source / "unexpected.bin").write_bytes(b"not in frozen inventory")
    else:
        plan["files"][0]["destinations"][0]["relative_path"] = "../../escape"
    before = _bytes(source)
    with pytest.raises((ValueError, FileExistsError)):
        organizer.materialize_transfer_organization(plan)
    assert _bytes(source) == before


def test_partial_materialization_retry_keeps_every_source(tmp_path, monkeypatch):
    source = _source(tmp_path)
    before = _bytes(source)
    plan = _plan(source, tmp_path / "recordings")
    real = organizer._materialize_file
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected partial materialization")
        return real(*args, **kwargs)

    monkeypatch.setattr(organizer, "_materialize_file", interrupted)
    with pytest.raises(OSError, match="injected"):
        organizer.materialize_transfer_organization(plan)
    assert _bytes(source) == before
    monkeypatch.setattr(organizer, "_materialize_file", real)
    assert organizer.materialize_transfer_organization(plan)["status"] == "materialized"
    assert _bytes(source) == before


def test_cross_device_fallback_is_bounded_and_byte_verified(tmp_path, monkeypatch):
    import errno

    source = _source(tmp_path)
    plan = _plan(source, tmp_path / "recordings")
    real = organizer.os.link

    def cross_device(src, dst, **kwargs):
        if Path(src).is_relative_to(source):
            raise OSError(errno.EXDEV, "test separate filesystems")
        return real(src, dst, **kwargs)

    monkeypatch.setattr(organizer.os, "link", cross_device)
    assert organizer.materialize_transfer_organization(plan)["status"] == "materialized"
    assert not list(Path(plan["destination_root"]).rglob("*.intake-partial"))


def test_prepare_parent_manifests_are_exact_and_retry_is_byte_stable(tmp_path):
    from fisheye.shared.source_recording_identity import load_source_recording_identity

    source = _source(tmp_path)
    before = _bytes(source)
    plan = _plan(source, tmp_path / "recordings")
    result = organizer.prepare_transfer_parent_recordings(plan, batch_rows=1)
    assert result["import_complete"] is False
    assert result["staging_finalized"] is False
    manifests = []
    for parent in plan["parents"]:
        directory = Path(parent["destination_dir"])
        manifest, identity = load_source_recording_identity(
            directory / "recording_manifest.json"
        )
        assert identity.recording_id == parent["identity"]["recording_id"]
        assert manifest["source_layout"] == "rolling_clips"
        assert "video_streams" not in manifest
        assert "h5_relative_path" not in manifest
        assert manifest["rolling_clip_streams"]["output_kinds"] == ["crop", "full"]
        assert (
            manifest["source_transfer"]["organization_plan_sha256"]
            == plan["plan_sha256"]
        )
        manifests.append(_bytes(directory))
    assert _bytes(source) == before
    organizer.prepare_transfer_parent_recordings(plan, batch_rows=1)
    assert [
        _bytes(Path(parent["destination_dir"])) for parent in plan["parents"]
    ] == manifests


def test_original_ptp_summary_uses_existing_clock_locator_without_rewriting(tmp_path):
    source = _source(tmp_path)
    summary = (
        b'{"sync": {"camera_sync_enabled": false}, "opaque": "preserve whitespace"}\n'
    )
    (source / "ptp_sync_summary.json").write_bytes(summary)
    _resign(source)
    plan = _plan(source, tmp_path / "recordings")
    organizer.prepare_transfer_parent_recordings(plan)
    for parent in plan["parents"]:
        directory = Path(parent["destination_dir"])
        assert (directory / "raw/ptp_sync_summary.json").read_bytes() == summary
        assert not (directory / "raw/acquisition/ptp_sync_summary.json").exists()
        manifest = json.loads((directory / "recording_manifest.json").read_bytes())
        assert "raw/ptp_sync_summary.json" in manifest["files"]["raw"]
    assert (source / "ptp_sync_summary.json").read_bytes() == summary


def test_prepare_manifest_failure_is_recoverable_without_source_retirement(
    tmp_path, monkeypatch
):
    source = _source(tmp_path)
    before = _bytes(source)
    plan = _plan(source, tmp_path / "recordings")
    real = organizer.write_json_atomic

    def fail(path, *args, **kwargs):
        if path.name == "recording_manifest.json":
            raise OSError("injected manifest publication failure")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(organizer, "write_json_atomic", fail)
    with pytest.raises(OSError, match="injected"):
        organizer.prepare_transfer_parent_recordings(plan)
    assert _bytes(source) == before
    monkeypatch.setattr(organizer, "write_json_atomic", real)
    organizer.prepare_transfer_parent_recordings(plan)
    assert _bytes(source) == before


def test_prepare_refuses_an_incomplete_index_and_preserves_recovery_evidence(tmp_path):
    source = _source(tmp_path)
    plan = _plan(source, tmp_path / "recordings")
    organizer.materialize_transfer_organization(plan)
    index = Path(plan["parents"][0]["destination_dir"]) / organizer.INDEX_DIRECTORY
    index.mkdir(parents=True)
    (index / "recording_frame_index.parquet").write_bytes(b"interrupted index")
    before = _bytes(source)
    with pytest.raises((ValueError, FileNotFoundError), match="index"):
        organizer.prepare_transfer_parent_recordings(plan)
    assert _bytes(source) == before
    assert (
        index / "recording_frame_index.parquet"
    ).read_bytes() == b"interrupted index"


def _prepared_with_admission_stub(tmp_path, monkeypatch):
    """Exercise retirement mechanics only; real admission is covered separately."""
    source = _source(tmp_path)
    plan = _plan(source, tmp_path / "recordings")
    organizer.prepare_transfer_parent_recordings(plan)
    monkeypatch.setattr(
        organizer,
        "_verify_parent_imports",
        lambda *args, **kwargs: {"fixture": "admission_stub_not_authority_evidence"},
    )
    return source, plan


def test_finalize_refuses_without_actual_import_receipts(tmp_path):
    source = _source(tmp_path)
    plan = _plan(source, tmp_path / "recordings")
    organizer.prepare_transfer_parent_recordings(plan)
    before = _bytes(source)
    with pytest.raises(Exception):
        organizer.finalize_transfer_staging(plan)
    assert _bytes(source) == before


def test_verified_retirement_empties_only_source_and_preserves_every_destination(
    tmp_path, monkeypatch
):
    source, plan = _prepared_with_admission_stub(tmp_path, monkeypatch)
    before = _bytes(source)
    result = organizer.finalize_transfer_staging(plan)
    assert result["status"] == "complete"
    assert result["staging_finalized"] is True
    assert source.is_dir() and not list(source.iterdir())
    for item in plan["files"]:
        for target in item["destinations"]:
            path = (
                Path(plan["destination_root"])
                / target["recording_id"]
                / target["relative_path"]
            )
            assert path.read_bytes() == before[item["source"]["path"]]
    assert organizer.finalize_transfer_staging(plan) == result


@pytest.mark.parametrize(
    "failure",
    [
        "new_source_file",
        "changed_source_file",
        "changed_destination",
        "parent_replaced",
        "admission_failed",
    ],
)
def test_failed_retirement_gate_keeps_all_remaining_staging_bytes(
    tmp_path, monkeypatch, failure
):
    source, plan = _prepared_with_admission_stub(tmp_path, monkeypatch)
    if failure == "new_source_file":
        (source / "new_component").write_bytes(b"unassigned")
    elif failure == "changed_source_file":
        path = source / plan["files"][0]["source"]["path"]
        path.unlink()
        path.write_bytes(b"different generation")
    elif failure == "changed_destination":
        target = plan["files"][0]["destinations"][0]
        path = (
            Path(plan["destination_root"])
            / target["recording_id"]
            / target["relative_path"]
        )
        path.unlink()
        path.write_bytes(b"different destination")
    elif failure == "parent_replaced":
        path = Path(plan["parents"][0]["destination_dir"])
        path.rename(path.with_name(path.name + ".preserved"))
        path.mkdir()
    else:

        def fail(*args, **kwargs):
            raise ValueError("admission failed")

        monkeypatch.setattr(organizer, "_verify_parent_imports", fail)
    before = _bytes(source)
    with pytest.raises((ValueError, OSError)):
        organizer.finalize_transfer_staging(plan)
    assert _bytes(source) == before


def test_partial_retirement_replays_exact_journal_without_snapshot_reconstruction(
    tmp_path, monkeypatch
):
    source, plan = _prepared_with_admission_stub(tmp_path, monkeypatch)
    real = organizer._retire_source_file
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected retirement interruption")
        return real(*args, **kwargs)

    monkeypatch.setattr(organizer, "_retire_source_file", fail)
    with pytest.raises(OSError, match="injected"):
        organizer.finalize_transfer_staging(plan)
    assert len(_bytes(source)) == len(plan["files"]) - 3
    monkeypatch.setattr(organizer, "_retire_source_file", real)
    assert organizer.finalize_transfer_staging(plan)["staging_finalized"] is True
    assert not list(source.iterdir())


@pytest.mark.parametrize(
    "failure", ["file_flush", "directory_flush", "retiring_journal_flush"]
)
def test_durability_barrier_failure_precedes_every_source_unlink(
    tmp_path, monkeypatch, failure
):
    source, plan = _prepared_with_admission_stub(tmp_path, monkeypatch)
    before = _bytes(source)
    real_file = organizer._fsync_regular_file
    real_directory = organizer._fsync_directory

    def fail_file(path):
        if failure == "file_flush":
            raise OSError("injected durable file flush failure")
        return real_file(path)

    def fail_directory(path):
        if failure == "directory_flush" and path == Path(
            plan["parents"][0]["destination_dir"]
        ):
            raise OSError("injected durable directory flush failure")
        if failure == "retiring_journal_flush" and path == organizer._state_directory(
            plan
        ):
            state = json.loads((path / "organization_state.json").read_bytes())
            if state["status"] == "retiring":
                raise OSError("injected retiring journal flush failure")
        return real_directory(path)

    monkeypatch.setattr(organizer, "_fsync_regular_file", fail_file)
    monkeypatch.setattr(organizer, "_fsync_directory", fail_directory)
    monkeypatch.setattr(
        organizer,
        "_retire_source_file",
        lambda *a, **k: pytest.fail("unlink before durability gate"),
    )
    with pytest.raises(OSError, match="injected"):
        organizer.finalize_transfer_staging(plan)
    assert _bytes(source) == before


def test_all_durable_copies_and_initial_journal_precede_first_unlink(
    tmp_path, monkeypatch
):
    source, plan = _prepared_with_admission_stub(tmp_path, monkeypatch)
    flushed_files, flushed_dirs = set(), set()
    real_file, real_directory = (
        organizer._fsync_regular_file,
        organizer._fsync_directory,
    )
    real_retire = organizer._retire_source_file
    retired = 0

    def file_flush(path):
        real_file(path)
        flushed_files.add(path)

    def directory_flush(path):
        real_directory(path)
        flushed_dirs.add(path)

    def retire(*args, **kwargs):
        nonlocal retired
        if retired == 0:
            expected = {
                Path(plan["destination_root"]) / t["recording_id"] / t["relative_path"]
                for item in plan["files"]
                for t in item["destinations"]
            }
            assert expected <= flushed_files
            assert organizer._state_directory(plan) in flushed_dirs
            assert Path(plan["destination_root"]) in flushed_dirs
            assert (
                json.loads(
                    (
                        organizer._state_directory(plan) / "organization_state.json"
                    ).read_bytes()
                )["status"]
                == "retiring"
            )
        retired += 1
        return real_retire(*args, **kwargs)

    monkeypatch.setattr(organizer, "_fsync_regular_file", file_flush)
    monkeypatch.setattr(organizer, "_fsync_directory", directory_flush)
    monkeypatch.setattr(organizer, "_retire_source_file", retire)
    organizer.finalize_transfer_staging(plan)
    assert retired == len(plan["files"])
    assert not list(source.iterdir())
