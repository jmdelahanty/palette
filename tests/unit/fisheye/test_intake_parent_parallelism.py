"""Per-parent verification scope and parallel parents in one delivery.

A parent's frame-index builder rechecks its own organized copies; the whole
delivery is still verified before the builders run and after they finish, so
a tampered copy of another parent is refused by preparation, not by the
unrelated builder. Parents are independent, so ``parent_workers`` of them
may be indexed and imported at the same time without changing a written byte
or the plan order of the results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import threading
import time

import pytest

from fisheye.intake import importing
from fisheye.shared import recording_transfer_snapshot as transfer
from fisheye.utils import build_transfer_parent_frame_index as indexer
from fisheye.utils import organize_transfer_recordings as organizer
from fisheye.utils import run_citrus_session_import as runner

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"
CAMERAS = ("02010093", "02010094")


@pytest.fixture(autouse=True)
def _placeholder_media_sync_assessment(monkeypatch):
    from fisheye.diagnostics.video import container

    monkeypatch.setattr(
        container,
        "check_hevc_keyframe_flags",
        lambda path, **_: {
            "schema_id": "palette.video.sync_sample_assessment.v1",
            "codec": "h264",
            "container_inspection_status": "ok",
            "sync_sample_proof": "container_declared",
            "message": "placeholder media",
        },
    )


@pytest.fixture(autouse=True)
def _files_have_settled(monkeypatch):
    """Fixture files were written moments ago; the job's clock says they settled."""

    monkeypatch.setattr(transfer, "_now_ns", lambda: time.time_ns() + 3600 * 10**9)


@pytest.fixture(autouse=True)
def _admission_stub(monkeypatch):
    monkeypatch.setattr(
        organizer,
        "_verify_parent_imports",
        lambda *args, **kwargs: {"fixture": "admission_stub_not_authority_evidence"},
    )


def _materialized(tmp_path: Path, name: str = "rolling"):
    source = Path(shutil.copytree(FIXTURES / name, tmp_path / "staging"))
    plan = organizer.build_transfer_organization_plan(
        source, destination_root=tmp_path / "recordings"
    )
    organizer.materialize_transfer_organization(plan)
    return source, plan


def _parent(plan: dict, camera_id: str) -> dict:
    [parent] = [p for p in plan["parents"] if p["identity"]["camera_id"] == camera_id]
    return parent


def _copy_of(plan: dict, camera_id: str, suffix: str) -> Path:
    recording_id = _parent(plan, camera_id)["identity"]["recording_id"]
    return next(
        organizer._parent_directory(plan, recording_id) / target["relative_path"]
        for item in plan["files"]
        if item["source"]["path"].endswith(suffix)
        for target in item["destinations"]
        if target["recording_id"] == recording_id
    )


def _flip_a_byte(path: Path) -> None:
    """Change one byte of this copy only: the hard link to staging is broken first."""

    mode = path.stat().st_mode
    data = bytearray(path.read_bytes())
    data[-2] ^= 0x01
    path.unlink()
    path.write_bytes(bytes(data))
    os.chmod(path, mode)


# ------------------------------------------------------- verification scope


def test_a_builder_rechecks_only_its_own_parent_and_preparation_still_refuses(tmp_path):
    source, plan = _materialized(tmp_path)
    other_only = _copy_of(plan, CAMERAS[1], f"Cam{CAMERAS[1]}_full.csv")
    _flip_a_byte(other_only)

    root, mapping = organizer.resolve_materialized_parent_sources(plan, camera_id=CAMERAS[0])
    assert root == Path(_parent(plan, CAMERAS[0])["destination_dir"])
    assert all(not path.startswith("Cam" + CAMERAS[1]) for path in mapping.values() if "Cam" in path)
    with pytest.raises(ValueError, match="differs"):
        organizer.resolve_materialized_parent_sources(plan, camera_id=CAMERAS[1])
    # The whole delivery is checked by preparation, whichever parent went first.
    with pytest.raises(ValueError, match="differs"):
        organizer.prepare_transfer_parent_recordings(plan)
    with pytest.raises(ValueError, match="differs"):
        organizer.finalize_transfer_staging(plan)
    assert source.is_dir() and any(source.iterdir())  # nothing retired


def test_a_shared_files_tampered_copy_is_pinned_to_its_parent(tmp_path):
    _, plan = _materialized(tmp_path)
    shared = "recording_session.json"
    copies = {camera: _copy_of(plan, camera, shared) for camera in CAMERAS}
    assert copies[CAMERAS[0]] != copies[CAMERAS[1]]
    assert os.stat(copies[CAMERAS[0]]).st_ino == os.stat(copies[CAMERAS[1]]).st_ino  # hard links

    # Break the link for one parent only, then change that parent's copy.
    data = copies[CAMERAS[1]].read_bytes()
    copies[CAMERAS[1]].unlink()
    copies[CAMERAS[1]].write_bytes(data)
    organizer.resolve_materialized_parent_sources(plan, camera_id=CAMERAS[1])  # same bytes: fine
    _flip_a_byte(copies[CAMERAS[1]])
    assert copies[CAMERAS[0]].read_bytes() == data  # the other parent's copy is untouched
    organizer.resolve_materialized_parent_sources(plan, camera_id=CAMERAS[0])
    with pytest.raises(ValueError, match="differs"):
        organizer.resolve_materialized_parent_sources(plan, camera_id=CAMERAS[1])
    with pytest.raises(ValueError, match="differs"):
        organizer._verify_materialized_files(
            plan, organizer.strict_json(organizer._state_directory(plan) / "organization_state.json")
        )


def test_the_scoped_check_covers_every_copy_the_builder_will_read(tmp_path):
    _, plan = _materialized(tmp_path)
    for parent in plan["parents"]:
        root, mapping = organizer.resolve_materialized_parent_sources(
            plan, camera_id=parent["identity"]["camera_id"]
        )
        recording_id = parent["identity"]["recording_id"]
        planned = {
            target["relative_path"]
            for item in plan["files"]
            for target in item["destinations"]
            if target["recording_id"] == recording_id
        }
        assert set(mapping.values()) == planned
        assert all((root / relative).is_file() for relative in planned)


# ---------------------------------------------------------- parallel parents


# Fields that bind the delivery's own location or clock, so they differ between
# two copies of the fixture regardless of how many workers prepared them.
_LOCATION_BOUND = {
    "organization_plan_sha256", "plan_sha256", "parquet_sha256",
    "recording_clip_index_sha256", "generated_at_utc", "duration_seconds", "host",
    "sha256",  # of a projected manifest / plan reference whose bytes hold paths
}


def _relocated(value, tmp_root: Path):
    if isinstance(value, dict):
        return {k: _relocated(v, tmp_root) for k, v in value.items() if k not in _LOCATION_BOUND}
    if isinstance(value, list):
        return [_relocated(v, tmp_root) for v in value]
    if isinstance(value, str):
        return value.replace(str(tmp_root), "<tmp>")
    return value


def _prepared_outputs(plan: dict, tmp_root: Path) -> dict[str, object]:
    """Everything preparation publishes, with the delivery's location abstracted."""

    import pyarrow.parquet as pq

    outputs: dict[str, object] = {}
    for parent in plan["parents"]:
        directory = Path(parent["destination_dir"])
        index = directory / "derived/recording_frame_index"
        table = pq.read_table(index / "recording_frame_index.parquet")
        outputs[f"{directory.name}/frame_index"] = (
            table.schema.to_string(),
            _relocated(table.to_pylist(), tmp_root),
        )
        for name in (
            index / "recording_clip_index.json",
            index / "recording_frame_index_manifest.json",
            directory / "derived/video_sync_assessment.json",
            directory / "recording_manifest.json",
            *sorted(index.glob("clip_*_projection.json")),
        ):
            outputs[f"{directory.name}/{name.name}"] = _relocated(
                json.loads(name.read_bytes()), tmp_root
            )
    assert len(outputs) > 4 * len(plan["parents"])
    return outputs


def test_parallel_preparation_writes_the_sequential_bytes(tmp_path, monkeypatch):
    sequential_root = tmp_path / "one"
    parallel_root = tmp_path / "two"
    sequential_root.mkdir()
    parallel_root.mkdir()

    _, plan_one = _materialized(sequential_root)
    organizer.prepare_transfer_parent_recordings(plan_one, parent_workers=1)

    _, plan_two = _materialized(parallel_root)
    barrier = threading.Barrier(len(plan_two["parents"]), timeout=30)
    threads: list[str] = []
    real_builder = indexer.build_transfer_parent_frame_index

    def builder_meets_its_peers(*args, **kwargs):
        threads.append(threading.current_thread().name)
        barrier.wait()  # every parent is being built at the same time
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(indexer, "build_transfer_parent_frame_index", builder_meets_its_peers)
    with transfer.digest_memo() as memo:
        state = organizer.prepare_transfer_parent_recordings(plan_two, parent_workers=len(plan_two["parents"]))
    assert state["parent_manifests_prepared"] is True
    assert len(threads) == len(plan_two["parents"]) and len(set(threads)) == len(threads)
    assert all(name.startswith("palette-parent") for name in threads)
    assert memo.hits > 0  # the workers shared the job's memo

    assert _prepared_outputs(plan_two, parallel_root) == _prepared_outputs(plan_one, sequential_root)
    # And the parallel delivery finishes like the sequential one.
    assert organizer.finalize_transfer_staging(plan_two)["status"] == "complete"
    assert organizer.finalize_transfer_staging(plan_one)["status"] == "complete"


def test_one_worker_runs_inline_on_the_callers_thread(tmp_path, monkeypatch):
    _, plan = _materialized(tmp_path)
    threads: list[str] = []
    real_builder = indexer.build_transfer_parent_frame_index

    def noting(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(indexer, "build_transfer_parent_frame_index", noting)
    organizer.prepare_transfer_parent_recordings(plan)  # default: one worker
    assert threads == [threading.current_thread().name] * len(plan["parents"])


def test_a_failing_parent_refuses_preparation_after_its_peers_finished(tmp_path, monkeypatch):
    _, plan = _materialized(tmp_path)
    finished: list[str] = []
    real_builder = indexer.build_transfer_parent_frame_index

    def one_fails(recording_dir, *, camera_id, **kwargs):
        if camera_id == CAMERAS[0]:
            time.sleep(0.05)
            raise OSError("injected storage failure")
        result = real_builder(recording_dir, camera_id=camera_id, **kwargs)
        time.sleep(0.2)
        finished.append(camera_id)
        return result

    monkeypatch.setattr(indexer, "build_transfer_parent_frame_index", one_fails)
    with pytest.raises(OSError, match="injected"):
        organizer.prepare_transfer_parent_recordings(plan, parent_workers=2)
    assert finished == [CAMERAS[1]]  # the peer ran to completion before the refusal
    state = organizer.strict_json(organizer._state_directory(plan) / "organization_state.json")
    assert not state.get("parent_manifests_prepared")
    # No parent manifest was published: the manifests follow every builder.
    for parent in plan["parents"]:
        assert not (Path(parent["destination_dir"]) / "recording_manifest.json").exists()


def test_parents_import_concurrently_in_plan_order_with_the_lock_lent_to_all(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fisheye.utils import import_organized_recordings_analysis as batch
    from fisheye.utils import import_recording_analysis as importer

    lock = tmp_path / "lease"
    lock.write_text("")
    descriptor = os.open(lock, os.O_RDWR)
    parents = [
        {"destination_dir": str(tmp_path / f"rec{i}"),
         "identity": {"recording_id": f"r{i}", "camera_id": f"c{i}"}}
        for i in range(3)
    ]
    plan = {"parents": parents}

    def plans(dirs, **_):
        [directory] = dirs
        return [SimpleNamespace(
            zarr_path=Path(directory) / "zarr" / f"{Path(directory).name}_analysis.zarr",
            status="ok", reason=None, recording_dir=Path(directory), h5_path=None,
            cam_video=None, recording_layout="rolling_clips",
        )]

    monkeypatch.setattr(batch, "build_plans", plans)
    barrier = threading.Barrier(2, timeout=30)
    seen: list[tuple[str, tuple, str]] = []

    def import_owner(plan, options):
        name = Path(plan.recording_dir).name
        seen.append((name, importer._stimulus_import_lease_fds(), threading.current_thread().name))
        if name in ("rec0", "rec2"):
            barrier.wait()  # the first and the third parent import at the same time
        if name == "rec1":
            return importer.RecordingImportResult(ok=False, failed_step="import_stimulus", returncode=3)
        return importer.RecordingImportResult(ok=True)

    monkeypatch.setattr(importer, "process_recording_import", import_owner)
    try:
        results = importing.import_parents(plan, recording_only=True, lease_fd=descriptor, parent_workers=2)
    finally:
        os.close(descriptor)
    assert [r.recording_id for r in results] == ["r0", "r1", "r2"]  # plan order, not finish order
    assert [r.outcome for r in results] == ["imported", "failed", "imported"]
    assert results[1].failed_step == "import_stimulus" and results[1].returncode == 3
    assert {lent for _, lent, _ in seen} == {(descriptor,)}  # every child gets the lock
    assert len({thread for _, _, thread in seen}) == 2
    assert importing.LEASE_FD_ENV not in os.environ


def test_import_parents_default_is_sequential_and_unchanged(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fisheye.utils import import_organized_recordings_analysis as batch
    from fisheye.utils import import_recording_analysis as importer

    lock = tmp_path / "lease"
    lock.write_text("")
    descriptor = os.open(lock, os.O_RDWR)
    plan = {"parents": [
        {"destination_dir": str(tmp_path / f"rec{i}"),
         "identity": {"recording_id": f"r{i}", "camera_id": f"c{i}"}}
        for i in range(2)
    ]}
    monkeypatch.setattr(batch, "build_plans", lambda dirs, **_: [SimpleNamespace(
        zarr_path=Path(dirs[0]) / "zarr" / f"{Path(dirs[0]).name}_analysis.zarr", status="ok",
        reason=None, recording_dir=Path(dirs[0]), h5_path=None, cam_video=None,
        recording_layout="rolling_clips")])
    threads: list[str] = []

    def import_owner(plan, options):
        threads.append(threading.current_thread().name)
        return importer.RecordingImportResult(ok=True)

    monkeypatch.setattr(importer, "process_recording_import", import_owner)
    try:
        results = importing.import_parents(plan, recording_only=True, lease_fd=descriptor)
    finally:
        os.close(descriptor)
    assert [r.outcome for r in results] == ["imported", "imported"]
    assert threads == [threading.current_thread().name] * 2


# ------------------------------------------------------ the job records both


def test_the_status_records_the_execution_resources_and_the_memo(tmp_path, monkeypatch):
    source = Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))
    workers_seen: list[int] = []

    def failed(plan, *, recording_only, lease_fd, parent_workers=1):
        workers_seen.append(parent_workers)
        return [
            importing.ParentImportResult(
                recording_id=parent["identity"]["recording_id"],
                camera_id=parent["identity"]["camera_id"],
                recording_dir=parent["destination_dir"],
                zarr_path="unused", outcome="failed",
                failed_step="ensure_analysis_archive", error="fixture failure",
            )
            for parent in plan["parents"]
        ]

    monkeypatch.setattr(importing, "import_parents", failed)
    arguments = [str(source), "--dest-root", str(tmp_path / "recordings"), "--run-dir", str(tmp_path / "run"), "--apply"]
    assert runner.main(arguments) == 1
    status = json.loads((tmp_path / "run/citrus_session_import.status.json").read_bytes())
    assert workers_seen == [importing.DEFAULT_PARENT_WORKERS]
    execution = status["execution"]
    assert execution["parent_workers"] == importing.DEFAULT_PARENT_WORKERS
    memo = execution["digest_memo"]
    assert memo["misses"] > 0  # the delivery was hashed in this job
    assert all(isinstance(memo[key], int) for key in ("hits", "misses", "unsettled", "unsettled_files"))
    assert set(memo["frame_maps"]) == {"hits", "misses", "unsettled"}
    assert memo["frame_maps"]["misses"] > 0 and memo["frame_maps"]["hits"] > 0
    assert isinstance(memo["unsettled_sample"], list)

    shutil.rmtree(tmp_path / "run")
    assert runner.main([*arguments, "--parent-workers", "1"]) == 1
    status = json.loads((tmp_path / "run/citrus_session_import.status.json").read_bytes())
    assert workers_seen[-1] == 1 and status["execution"]["parent_workers"] == 1


@pytest.mark.parametrize("workers", [0, -1])
def test_fewer_than_one_worker_is_refused_before_any_write(tmp_path, workers):
    source = Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))
    with pytest.raises(importing.IntakeRefused, match="parent_workers"):
        importing.import_delivery(
            "0" * 64, tmp_path / "run", destination_root=tmp_path / "recordings",
            session_dir=source, parent_workers=workers,
        )
    assert not (tmp_path / "run").exists() and not (tmp_path / "recordings").exists()
