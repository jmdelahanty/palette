"""fisheye.intake: the five entry points, their claims, exit codes and probes.

The acceptance gate (docs/design/2026-10-07-intake-single-writer,
"Enforcement" 2) runs a sealed transfer through the real organizer, the real
recording import owner, the real identity authority and the real shadow
gateway into a temporary registry, then replays every step. The sealed
delivery is ``encoded_rolling``: the pinned Citrus transfer of real tiny H264
clips. The placeholder-media ``rolling`` fixture cannot pass the real importer
(its crop CSV has no ``has_detection``), so it is used only where no import
has to complete: claims, refusals, discovery and the CLI contract.

Only the producing-code identity is stubbed (the importer refuses a dirty
checkout), exactly as the other real-import tests do.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys

import pytest

from fisheye.intake import (
    EXIT_HELD,
    EXIT_REFUSED,
    IntakeHeld,
    IntakeRefused,
    RegistryWriter,
    discover,
    import_delivery,
    probe_import,
    probe_register,
    register_delivery,
)
from fisheye.intake.delivery import (
    CANONICAL_REGISTRY,
    IMPORT_LOCK_KIND,
    REGISTER_LOCK_KIND,
    claim,
)
from fisheye.intake.registration import is_writer_host
from fisheye.registry.db import Registry
from fisheye.shared.recording_transfer_snapshot import MARKER_NAME
from fisheye.utils import organize_transfer_recordings as organizer

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests/fixtures/recording_transfer_v2"
LAUNCHER = REPO / "scripts/submit_citrus_session_import_bsub.sh"
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
HOST = socket.gethostname()

needs_media_tools = pytest.mark.skipif(
    shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None,
    reason="the real import probes the encoded fixture with ffprobe",
)


# ---------------------------------------------------------------- helpers


def _stub_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    from fisheye.shared import run_provenance
    from fisheye.utils import import_recording_analysis as importer

    identity = lambda **_: {"git_sha": GIT_SHA, "git_dirty": False}  # noqa: E731
    monkeypatch.setattr(importer, "git_identity", identity)
    monkeypatch.setattr(run_provenance, "git_identity", identity)


def _stub_environment(tmp_path: Path, sha: str = GIT_SHA) -> dict[str, str]:
    """Subprocess environment whose importer sees a clean producing commit."""

    stub = tmp_path / f"checkout_identity_stub_{sha[:8]}"
    stub.mkdir(exist_ok=True)
    (stub / "sitecustomize.py").write_text(
        "identity = lambda **_: {'git_sha': %r, 'git_dirty': False}\n"
        "from fisheye.shared import run_provenance\n"
        "run_provenance.git_identity = identity\n"
        "import fisheye.utils.import_recording_analysis as importer\n"
        "importer.git_identity = identity\n" % sha
    )
    return dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{stub}")


def _delivery(tmp_path: Path, name: str = "encoded_rolling") -> tuple[Path, str]:
    session = Path(shutil.copytree(FIXTURES / name, tmp_path / "staging" / name))
    marker = json.loads((session / MARKER_NAME).read_bytes())
    return session, marker["snapshot_id"].removeprefix("sha256:")


def _destination(tmp_path: Path) -> Path:
    # Current-source receipts require the store to be a "recordings" directory.
    return tmp_path / "recordings"


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry" / "registry.sqlite"
    path.parent.mkdir(exist_ok=True)
    Registry(path).close()
    return path


def _writer(tmp_path: Path, registry: Path, host: str = HOST) -> RegistryWriter:
    return RegistryWriter(
        registry=registry,
        writer_host=host,
        writer_lock_path=tmp_path / "writer.lock",
        shadow_temp_root=tmp_path / "shadows",
        shadow_backup_dir=tmp_path / "backups",
    )


def _tree(root: Path) -> dict[str, str | None]:
    """Every path under root with its content digest.

    Directories and intake lock files map to None: a lock file's content is
    only the latest holder's informational record, rewritten by every claim.
    """

    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() and not path.name.endswith(".lock")
            else None
        )
        for path in sorted(root.rglob("*"))
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(registry: Path) -> dict[str, int]:
    with sqlite3.connect(f"{registry.as_uri()}?mode=ro", uri=True) as connection:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("datasets", "recordings", "recording_import_receipt_bindings")
        }


def _cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(REPO / "scripts/py"), "-m", "fisheye.intake", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        env=env,
        timeout=600,
    )


def _import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _stub_checkout(monkeypatch)
    session, sha = _delivery(tmp_path)
    destination = _destination(tmp_path)
    result = import_delivery(
        sha, tmp_path / "runs" / "attempt-1", destination_root=destination, session_dir=session
    )
    assert result.verdict, result
    return session, sha, destination, result


@pytest.fixture
def placeholder_media(monkeypatch):
    """The ``rolling`` fixture's MP4s are text; the sync-sample check is stubbed."""

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


# ---------------------------------------------------------------- the gate


@needs_media_tools
def test_end_to_end_gate_job_script_register_probes_and_replay(tmp_path: Path, monkeypatch) -> None:
    _stub_checkout(monkeypatch)  # registration binds the receipt's producing commit
    session, sha = _delivery(tmp_path)
    destination = _destination(tmp_path)
    registry = _registry(tmp_path)
    env = _stub_environment(tmp_path)

    # 1. import_delivery through the generated LSF job script.
    launch = subprocess.run(
        ["bash", str(LAUNCHER), "--session-dir", str(session), "--marker-key", "e" * 64,
         "--log-dir", str(tmp_path / "logs"), "--run-id", "gate", "--dest-root", str(destination),
         "--dry-run"],
        check=False, text=True, capture_output=True,
    )
    assert launch.returncode == 0, launch.stderr
    [job_script] = (tmp_path / "logs").glob("citrus_import_*/run_citrus_session_import.sh")
    job = subprocess.run(
        ["bash", str(job_script)], check=False, text=True, capture_output=True,
        env=dict(env, LSB_JOBID="777"), timeout=600,
    )
    payload_err = next(job_script.parent.glob("*.777.*.payload.err")).read_text()
    assert job.returncode == 0, job.stdout + payload_err
    [workflow_dir] = job_script.parent.glob("workflow-777-*")
    status = json.loads((workflow_dir / "citrus_session_import.status.json").read_text())
    assert status["status"] == "complete" and status["staging_finalized"] is True
    assert status["registry"] is None
    assert [parent["outcome"] for parent in status["parents"]] == ["imported", "imported"]
    assert not list(session.iterdir())  # staging retired

    # Imported but not registered: discovery still lists it, for registration.
    found = discover(tmp_path / "staging", destination, registry=registry).to_json()
    [target] = found["targets"]
    assert (target["snapshot_sha"], target["state"], target["admission_mode"]) == (
        sha, "complete", "workstation",
    )
    assert target["import_recorded"] is True and target["register_recorded"] is False

    # 2. register_delivery into the temporary registry; 3. both probes true.
    imported = probe_import(sha, destination_root=destination)
    assert imported.verdict
    assert imported.evidence_digest == status["probe_import"]["evidence_digest"]
    assert imported.producer_git_sha == status["probe_import"]["producer_git_sha"] == GIT_SHA
    assert target["producer_git_sha"] == GIT_SHA  # read cheaply from the receipt files
    assert sorted(imported.zarr_paths) == sorted(status["zarr_paths"])
    rows_before = _rows(registry)
    registered = register_delivery(
        sha, writer=_writer(tmp_path, registry), destination_root=destination, allow_synthetic=True
    )
    assert registered.verdict and len(registered.bindings) == 2
    assert _rows(registry)["recording_import_receipt_bindings"] == (
        rows_before["recording_import_receipt_bindings"] + 2
    )
    [backup] = (tmp_path / "registry" / ".palette-registry-backups").iterdir()  # one publication
    assert ".before-recording-imports-" in backup.name
    assert probe_register(sha, destination_root=destination, registry=registry) == registered
    assert discover(tmp_path / "staging", destination, registry=registry).targets == ()

    # 4. Replay every step: no change, no duplicate rows, identical digests.
    recordings = _tree(destination)
    registry_bytes = _sha(registry)
    rows = _rows(registry)
    backups = _tree(tmp_path / "registry")

    replay = _cli("import-delivery", sha, "--run-dir", str(tmp_path / "runs" / "replay"),
                  "--destination-root", str(destination), "--json", env=env)
    assert replay.returncode == 0, replay.stderr
    replayed = json.loads(replay.stdout)  # stdout is exactly one JSON document
    assert replayed["schema"] == "palette.intake.probe_import.v1"
    assert replayed["verdict"] is True
    assert replayed["evidence_digest"] == imported.evidence_digest
    assert replayed["producer_git_sha"] == GIT_SHA

    again = _cli(
        "register-delivery", sha, "--destination-root", str(destination),
        "--registry", str(registry), "--writer-host", HOST,
        "--writer-lock-path", str(tmp_path / "writer.lock"),
        "--shadow-temp-root", str(tmp_path / "shadows"),
        "--shadow-backup-dir", str(tmp_path / "backups"),
        "--allow-synthetic-isolated-registry", "--json", env=env,
    )
    assert again.returncode == 0, again.stderr
    reregistered = json.loads(again.stdout)
    assert reregistered["schema"] == "palette.intake.probe_register.v1"
    assert reregistered["evidence_digest"] == registered.evidence_digest
    assert reregistered["bindings"] == [dict(row) for row in registered.bindings]

    for name in ("probe-import", "probe-register"):
        extra = ["--registry", str(registry)] if name == "probe-register" else []
        probe = _cli(name, sha, "--destination-root", str(destination), *extra, "--json")
        assert probe.returncode == 0, probe.stderr
        document = json.loads(probe.stdout)
        expected = imported if name == "probe-import" else registered
        assert document["verdict"] is True
        assert document["evidence_digest"] == expected.evidence_digest
        assert document["receipt_sha256s"] == list(expected.receipt_sha256s)

    assert _tree(destination) == recordings
    assert _sha(registry) == registry_bytes
    assert _rows(registry) == rows
    assert _tree(tmp_path / "registry") == backups  # no new backup: nothing published

    # The register claim is honoured: a held delivery is attached, not re-run.
    plan = json.loads((workflow_dir / "organization_plan.json").read_text())
    with claim(plan, kind=REGISTER_LOCK_KIND):
        with pytest.raises(IntakeHeld) as held:
            register_delivery(sha, writer=_writer(tmp_path, registry), destination_root=destination,
                              allow_synthetic=True)
    assert held.value.holder["pid"] == os.getpid()
    assert _sha(registry) == registry_bytes


@needs_media_tools
def test_resume_from_retiring_with_the_marker_already_gone(tmp_path, monkeypatch) -> None:
    _stub_checkout(monkeypatch)
    session, sha = _delivery(tmp_path)
    destination = _destination(tmp_path)
    original = organizer._retire_source_file

    def dies_after_the_marker(source, expected, signature):
        original(source, expected, signature)
        if source.name == MARKER_NAME:
            raise OSError("node lost after retiring the marker")

    monkeypatch.setattr(organizer, "_retire_source_file", dies_after_the_marker)
    with pytest.raises(OSError, match="node lost"):
        import_delivery(sha, tmp_path / "runs" / "attempt-1", destination_root=destination,
                        session_dir=session)
    monkeypatch.setattr(organizer, "_retire_source_file", original)
    assert not (session / MARKER_NAME).exists()
    assert any(session.rglob("*.mp4"))  # retirement is only half done

    [target] = discover(tmp_path / "staging", destination).targets
    assert (target.state, target.has_plan, target.marker_path) == ("retiring", True, None)
    assert not probe_import(sha, destination_root=destination).verdict

    # No marker, no session, no explicit plan: the module loads state["plan"].
    resumed = import_delivery(sha, tmp_path / "runs" / "attempt-2", destination_root=destination)
    assert resumed.verdict and resumed.state == "complete"
    assert not list(session.iterdir())
    assert probe_import(sha, destination_root=destination) == resumed


def test_second_concurrent_attempt_exits_75_and_leaves_no_new_files(tmp_path, monkeypatch) -> None:
    from fisheye.utils import run_citrus_session_import as compatibility

    session, sha = _delivery(tmp_path, "rolling")
    destination = _destination(tmp_path)
    plan = organizer.build_transfer_organization_plan(session, destination_root=destination)
    with claim(plan, kind=IMPORT_LOCK_KIND):
        before = _tree(tmp_path)
        attempt = _cli("import-delivery", sha, "--run-dir", str(tmp_path / "runs" / "second"),
                       "--session-dir", str(session), "--destination-root", str(destination),
                       "--json")
        assert attempt.returncode == EXIT_HELD, attempt.stderr
        held = json.loads(attempt.stdout)
        assert held["schema"] == "palette.intake.held.v1"
        assert held["holder"]["pid"] == os.getpid() and held["holder"]["host"] == HOST
        assert held["lock_path"].endswith(f"{sha}{IMPORT_LOCK_KIND}")
        # The compatibility command the LSF job script runs answers the same.
        assert compatibility.main(
            [str(session), "--apply", "--dest-root", str(destination),
             "--run-dir", str(tmp_path / "runs" / "third")]
        ) == EXIT_HELD
        assert _tree(tmp_path) == before  # no run dir, no status, no state


def test_in_flight_job_mode_delivery_is_reported_not_resumed(
    tmp_path, monkeypatch, placeholder_media
) -> None:
    session, sha = _delivery(tmp_path, "rolling")
    destination = _destination(tmp_path)
    plan = organizer.build_transfer_organization_plan(session, destination_root=destination)
    # A delivery whose first attempt ran under the retired "job registers" mode.
    organizer.prepare_transfer_parent_recordings(plan, registry_path=tmp_path / "old.sqlite")

    [target] = discover(tmp_path / "staging", destination).targets
    assert (target.state, target.admission_mode, target.legacy_mode) == ("materialized", "job", True)
    before = _tree(tmp_path)
    attempt = _cli("import-delivery", sha, "--run-dir", str(tmp_path / "runs" / "resume"),
                   "--destination-root", str(destination), "--json")
    assert attempt.returncode == EXIT_REFUSED, attempt.stderr
    refusal = json.loads(attempt.stdout)
    assert refusal["error"] == "legacy_job_mode_delivery"
    assert "retired job-mode registration" in refusal["message"]
    assert "operator path" in refusal["message"]
    assert refusal["admission_contract"]["registry_path"] == str((tmp_path / "old.sqlite").resolve())
    with pytest.raises(IntakeRefused, match="retired job-mode"):
        register_delivery(sha, writer=_writer(tmp_path, _registry(tmp_path)),
                          destination_root=destination, allow_synthetic=True)
    with pytest.raises(IntakeRefused, match="retired job-mode"):
        import_delivery(sha, tmp_path / "runs" / "in-process", destination_root=destination)
    assert {k: v for k, v in _tree(tmp_path).items() if not k.startswith("registry")} == before


@needs_media_tools
def test_register_failure_on_a_later_camera_publishes_nothing(tmp_path, monkeypatch) -> None:
    from fisheye.registry import db as registry_db

    _session, sha, destination, imported = _import(tmp_path, monkeypatch)
    registry = _registry(tmp_path)
    before = _sha(registry)
    real = registry_db.Registry.synchronize_recording_import
    seen = []

    def second_camera_fails(self, *, zarr_path, receipt, decided_by):
        seen.append(zarr_path)
        if len(seen) == 2:
            raise RuntimeError("registry write failed on the second camera")
        return real(self, zarr_path=zarr_path, receipt=receipt, decided_by=decided_by)

    monkeypatch.setattr(registry_db.Registry, "synchronize_recording_import", second_camera_fails)
    with pytest.raises(RuntimeError, match="second camera"):
        register_delivery(sha, writer=_writer(tmp_path, registry), destination_root=destination,
                          allow_synthetic=True)
    assert len(seen) == 2
    assert _sha(registry) == before  # the first camera's row never reached the registry
    assert not probe_register(sha, destination_root=destination, registry=registry).verdict

    monkeypatch.setattr(registry_db.Registry, "synchronize_recording_import", real)
    retried = register_delivery(sha, writer=_writer(tmp_path, registry),
                                destination_root=destination, allow_synthetic=True)
    assert retried.verdict and list(retried.receipt_sha256s) == list(imported.receipt_sha256s)


@needs_media_tools
def test_synthetic_origin_is_refused_and_a_missing_receipt_is_not_done(tmp_path, monkeypatch) -> None:
    _session, sha, destination, imported = _import(tmp_path, monkeypatch)
    registry = _registry(tmp_path)
    before = _sha(registry)
    refused = _cli(
        "register-delivery", sha, "--destination-root", str(destination),
        "--registry", str(registry), "--writer-host", HOST,
        "--writer-lock-path", str(tmp_path / "writer.lock"),
        "--shadow-temp-root", str(tmp_path / "shadows"),
        "--shadow-backup-dir", str(tmp_path / "backups"),
    )
    assert refused.returncode == EXIT_REFUSED, refused.stderr
    error = json.loads(refused.stdout)
    assert error["schema"] == "palette.intake.error.v1" and "synthetic" in error["error"]
    assert _sha(registry) == before
    assert not (registry.parent / ".palette-registry-backups").exists()
    # The isolated-registry allowance can never reach the canonical registry.
    with pytest.raises(IntakeRefused, match="canonical registry"):
        register_delivery(sha, writer=_writer(tmp_path, CANONICAL_REGISTRY),
                          destination_root=destination, allow_synthetic=True)

    # Evidence, not state, decides "done": a lost receipt makes the probe false.
    zarr_path, receipt = imported.zarr_paths[0], imported.receipt_sha256s[0]
    (Path(zarr_path) / ".imports" / f"{receipt}.json").unlink()
    assert not probe_import(sha, destination_root=destination).verdict
    with pytest.raises(Exception):
        import_delivery(sha, tmp_path / "runs" / "replay", destination_root=destination)


def test_import_parents_lends_the_workflow_lock_to_the_stimulus_child(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from fisheye.intake import importing
    from fisheye.utils import import_organized_recordings_analysis as batch
    from fisheye.utils import import_recording_analysis as importer

    lock = tmp_path / "lease"
    lock.write_text("")
    descriptor = os.open(lock, os.O_RDWR)
    plan = {
        "parents": [{"destination_dir": str(tmp_path / "rec"),
                     "identity": {"recording_id": "r", "camera_id": "c"}}],
    }
    zarr_path = tmp_path / "rec" / "zarr" / "rec_analysis.zarr"
    monkeypatch.setattr(
        batch, "build_plans",
        lambda dirs, **_: [SimpleNamespace(
            zarr_path=zarr_path, status="ok", reason=None, recording_dir=tmp_path / "rec",
            h5_path=None, cam_video=None, recording_layout="rolling_clips")],
    )
    lent = []

    def import_owner(plan, options):
        lent.append(importer._stimulus_import_lease_fds())
        return importer.RecordingImportResult(ok=True)

    monkeypatch.setattr(importer, "process_recording_import", import_owner)
    try:
        [result] = importing.import_parents(plan, recording_only=True, lease_fd=descriptor)
    finally:
        os.close(descriptor)
    assert lent == [(descriptor,)]
    assert result.outcome == "imported" and result.zarr_path == str(zarr_path)
    assert importing.LEASE_FD_ENV not in os.environ  # lent only for the import


# ---------------------------------------------------------------- CLI contract


def test_cli_usage_errors_are_never_exit_1(tmp_path) -> None:
    for argv in (["import-delivery"], ["no-such-step"], [], ["probe-import"]):
        result = _cli(*argv)
        assert result.returncode == 2, (argv, result.returncode, result.stderr)
        assert result.stdout == ""


def test_cli_probe_shapes_and_false_verdicts(tmp_path) -> None:
    destination = _destination(tmp_path)
    destination.mkdir()
    sha = "a" * 64
    for name, extra in (("probe-import", []), ("probe-register", ["--registry", str(tmp_path / "r.sqlite")])):
        result = _cli(name, sha, "--destination-root", str(destination), *extra, "--json")
        assert result.returncode == 1, result.stderr
        document = json.loads(result.stdout)
        assert document["schema"] == f"palette.intake.{name.replace('-', '_')}.v1"
        assert document["verdict"] is False and document["evidence_digest"] is None
        assert document["zarr_paths"] == [] and document["receipt_sha256s"] == []
        assert ("bindings" in document) == (name == "probe-register")
    # A malformed sha is a usage error (2), so probes only ever exit 0/1/2.
    bad = _cli("probe-import", "not-a-sha", "--destination-root", str(destination))
    assert bad.returncode == 2 and bad.stdout == ""


def test_cli_register_off_the_writer_host_is_refused(tmp_path) -> None:
    result = _cli(
        "register-delivery", "a" * 64, "--destination-root", str(tmp_path),
        "--registry", str(tmp_path / "r.sqlite"), "--writer-host", "some-other-writer.example.org",
        "--writer-lock-path", str(tmp_path / "w.lock"), "--shadow-temp-root", str(tmp_path / "s"),
        "--shadow-backup-dir", str(tmp_path / "b"),
    )
    assert result.returncode == EXIT_REFUSED
    assert "is not the registry writer" in json.loads(result.stdout)["error"]
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_cli_discover_reports_markers_states_and_refusals(tmp_path) -> None:
    session, sha = _delivery(tmp_path, "rolling")
    legacy = tmp_path / "staging" / "legacy"
    legacy.mkdir()
    (legacy / MARKER_NAME).write_text(json.dumps({"schema_id": "citrus.transfer_completion_marker.v1"}))
    broken = tmp_path / "staging" / "broken"
    broken.mkdir()
    (broken / MARKER_NAME).write_text("{not json")
    before = _tree(tmp_path)
    result = _cli("discover", "--staging-dir", str(tmp_path / "staging"),
                  "--destination-root", str(_destination(tmp_path)), "--json")
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["schema"] == "palette.intake.discover.v1"
    assert document["probe_depth"] == "recorded"
    [target] = document["targets"]
    assert target == {
        "snapshot_sha": sha, "state": "marker", "admission_mode": "unset", "legacy_mode": False,
        "has_plan": False, "session_dir": str(session), "marker_path": str(session / MARKER_NAME),
        "import_recorded": False, "register_recorded": False, "zarr_paths": [],
        "producer_git_sha": None,
    }
    assert document["registry"] == str(CANONICAL_REGISTRY)  # the default
    assert document["registry_error"] is None
    assert document["legacy_markers"] == [str(legacy / MARKER_NAME)]
    assert [item["path"] for item in document["refused_markers"]] == [str(broken / MARKER_NAME)]
    assert _tree(tmp_path) == before  # discovery writes nothing
    missing = _cli("discover", "--staging-dir", str(tmp_path / "absent"),
                   "--destination-root", str(tmp_path))
    assert missing.returncode == EXIT_REFUSED


def test_discovery_never_reads_v1_or_runner_state(tmp_path) -> None:
    _session, sha = _delivery(tmp_path, "rolling")
    # Poller state of both generations and a runner sentinel are not evidence.
    for name in (".processing_state", ".processing_state_v2"):
        (tmp_path / "staging" / name).mkdir()
        (tmp_path / "staging" / name / f"{'f' * 64}.registered").write_text("{}")
    sentinel = tmp_path / "flow" / "intake" / sha / "register.done.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("{}")
    [target] = discover(tmp_path / "staging", _destination(tmp_path)).targets
    assert target.snapshot_sha == sha and target.state == "marker"


@pytest.mark.parametrize(
    "configured,current,same",
    [
        ("delahantyj-ws1.hhmi.org", "delahantyj-ws1.hhmi.org", True),
        ("delahantyj-ws1.hhmi.org", "delahantyj-ws1", True),
        ("delahantyj-ws1", "DELAHANTYJ-WS1.hhmi.org.", True),
        ("delahantyj-ws1.hhmi.org", "delahantyj-ws1.other.org", False),
        ("delahantyj-ws1", "delahantyj-ws2", False),
        ("", "delahantyj-ws1", False),
    ],
)
def test_writer_host_identity_is_normalized_in_one_place(configured, current, same) -> None:
    assert is_writer_host(configured, current) is same


def test_python_entry_points_match_the_runner_contract() -> None:
    import inspect

    assert list(inspect.signature(discover).parameters)[:2] == ["staging_dir", "destination_root"]
    assert list(inspect.signature(import_delivery).parameters)[:3] == [
        "snapshot_sha", "run_dir", "resume_plan",
    ]
    assert list(inspect.signature(register_delivery).parameters)[0] == "snapshot_sha"
    assert list(inspect.signature(probe_import).parameters)[0] == "snapshot_sha"
    assert list(inspect.signature(probe_register).parameters)[0] == "snapshot_sha"
    assert sys.modules["fisheye.intake"].__all__  # importable without side effects



# ---------------------------------------------------------------- review fixes


OTHER_SHA = "f" * 40


@needs_media_tools
def test_registrar_at_another_commit_is_refused_before_any_backup(tmp_path, monkeypatch) -> None:
    """B1: a doomed registration is refused (65) and copies no backup, ever."""

    from fisheye.intake.outcomes import RegistrarCommitMismatch, exit_code_for
    from fisheye.shared import run_provenance

    _session, sha, destination, imported = _import(tmp_path, monkeypatch)
    registry = _registry(tmp_path)
    before = _sha(registry)
    monkeypatch.setattr(
        run_provenance, "git_identity", lambda **_: {"git_sha": OTHER_SHA, "git_dirty": False}
    )
    for _ in range(3):
        with pytest.raises(RegistrarCommitMismatch) as refused:
            register_delivery(sha, writer=_writer(tmp_path, registry),
                              destination_root=destination, allow_synthetic=True)
        assert exit_code_for(refused.value) == EXIT_REFUSED
        assert refused.value.code == "registrar_commit_mismatch"
        assert refused.value.details["receipt_producer_git_sha"] == GIT_SHA
        assert refused.value.details["registrar_git_sha"] == OTHER_SHA
    assert not (registry.parent / ".palette-registry-backups").exists()
    assert _sha(registry) == before

    # The CLI carries the needed commit as machine-readable fields.
    result = _cli(
        "register-delivery", sha, "--destination-root", str(destination),
        "--registry", str(registry), "--writer-host", HOST,
        "--writer-lock-path", str(tmp_path / "writer.lock"),
        "--shadow-temp-root", str(tmp_path / "shadows"),
        "--shadow-backup-dir", str(tmp_path / "backups"),
        "--allow-synthetic-isolated-registry",
        env=_stub_environment(tmp_path, OTHER_SHA),
    )
    assert result.returncode == EXIT_REFUSED, result.stderr
    error = json.loads(result.stdout)
    assert error["error"] == "registrar_commit_mismatch"
    assert error["receipt_producer_git_sha"] == GIT_SHA
    assert error["registrar_git_sha"] == OTHER_SHA
    assert "producer commit" in error["message"]
    assert not (registry.parent / ".palette-registry-backups").exists()

    # Defence in depth: the gateway itself refuses before copying a backup.
    from fisheye.registry.shadow_publish import (
        RegistryProducerCommitMismatch,
        shadow_synchronize_recording_imports,
    )
    from fisheye.shared.recording_import_receipt import (
        RecordingImportReceipt,
        recording_import_receipt_path,
    )

    imports = [
        (Path(z), RecordingImportReceipt.from_path(recording_import_receipt_path(Path(z), r)))
        for z, r in zip(imported.zarr_paths, imported.receipt_sha256s)
    ]
    with pytest.raises(RegistryProducerCommitMismatch):
        shadow_synchronize_recording_imports(
            canonical_registry=registry, imports=imports, decided_by="pytest"
        )
    assert not (registry.parent / ".palette-registry-backups").exists()


@needs_media_tools
def test_receipts_that_disagree_on_their_producer_are_refused(tmp_path, monkeypatch) -> None:
    from fisheye.intake import probes

    _session, sha, destination, imported = _import(tmp_path, monkeypatch)
    first = imported.zarr_paths[0]
    monkeypatch.setattr(
        probes, "receipt_producer_git_sha",
        lambda zarr, receipt: GIT_SHA if str(zarr) == first else OTHER_SHA,
    )
    probe = probe_import(sha, destination_root=destination)
    assert not probe.verdict and "disagree" in probe.reason
    with pytest.raises(IntakeRefused) as refused:
        import_delivery(sha, tmp_path / "runs" / "replay", destination_root=destination)
    assert refused.value.code == "receipt_producer_commits_disagree"


@needs_media_tools
def test_deterministic_organizer_violations_are_refused_and_lock_loss_retried(
    tmp_path, monkeypatch
) -> None:
    """S2b: a tampered parent manifest under the lock is 65; lock loss stays 1."""

    from fisheye.intake.outcomes import exit_code_for
    from fisheye.shared.recording_transfer_snapshot import TransferSnapshotError

    _stub_checkout(monkeypatch)
    session, sha = _delivery(tmp_path)
    destination = _destination(tmp_path)
    original = organizer._retire_source_file

    def dies_after_the_marker(source, expected, signature):
        original(source, expected, signature)
        if source.name == MARKER_NAME:
            raise OSError("node lost")

    monkeypatch.setattr(organizer, "_retire_source_file", dies_after_the_marker)
    with pytest.raises(OSError):
        import_delivery(sha, tmp_path / "runs" / "a1", destination_root=destination, session_dir=session)
    monkeypatch.setattr(organizer, "_retire_source_file", original)

    real_finalize = organizer.finalize_transfer_staging

    def lock_lost(*args, **kwargs):
        raise TransferSnapshotError("coordinator lock ownership lost")

    monkeypatch.setattr(organizer, "finalize_transfer_staging", lock_lost)
    with pytest.raises(TransferSnapshotError) as transient:
        import_delivery(sha, tmp_path / "runs" / "a2", destination_root=destination)
    assert exit_code_for(transient.value) == 1
    monkeypatch.setattr(organizer, "finalize_transfer_staging", real_finalize)

    [manifest] = list(destination.glob("*/recording_manifest.json"))[:1]
    manifest.write_text(manifest.read_text().replace("{", '{"tampered": true, ', 1))
    with pytest.raises(IntakeRefused) as refused:
        import_delivery(sha, tmp_path / "runs" / "a3", destination_root=destination)
    assert refused.value.code == "intake_invariant_violation"
    assert "parent manifest changed before retirement" in str(refused.value)


@pytest.mark.parametrize(
    "step,returncode,expected",
    [
        ("recording_import_preflight", None, EXIT_REFUSED),
        ("preflight_gate", None, EXIT_REFUSED),
        ("import_stimulus_to_zarr", 2, EXIT_REFUSED),
        ("import_stimulus_to_zarr", 1, 1),  # a dead child: cannot tell from I/O
        ("ensure_analysis_archive", None, 1),
    ],
)
def test_parent_import_refusals_are_classified(
    tmp_path, monkeypatch, placeholder_media, step, returncode, expected
) -> None:
    from fisheye.intake import importing
    from fisheye.intake.outcomes import exit_code_for

    session, sha = _delivery(tmp_path, "rolling")

    def failed(plan, *, recording_only, lease_fd):
        return [
            importing.ParentImportResult(
                recording_id=p["identity"]["recording_id"], camera_id=p["identity"]["camera_id"],
                recording_dir=p["destination_dir"], zarr_path="z", outcome="failed",
                failed_step=step, error="fixture", returncode=returncode,
            )
            for p in plan["parents"]
        ]

    monkeypatch.setattr(importing, "import_parents", failed)
    with pytest.raises(Exception) as exc:
        import_delivery(sha, tmp_path / "run", destination_root=_destination(tmp_path),
                        session_dir=session)
    assert exit_code_for(exc.value) == expected


def test_probes_answer_only_true_or_false(tmp_path, monkeypatch) -> None:
    """S2a: malformed state is a false verdict; an I/O error propagates (1)."""

    from fisheye.intake import delivery

    sha = "a" * 64
    destination = _destination(tmp_path)
    state_dir = destination / ".transfer_intake" / sha
    state_dir.mkdir(parents=True)
    (state_dir / "organization_state.json").write_text("{not json")
    probe = probe_import(sha, destination_root=destination)
    assert not probe.verdict and "malformed" in probe.reason
    result = _cli("probe-import", sha, "--destination-root", str(destination))
    assert result.returncode == 1 and json.loads(result.stdout)["verdict"] is False

    def eio(path):
        raise OSError(5, "Input/output error", str(path))

    monkeypatch.setattr(delivery, "strict_json", eio)
    with pytest.raises(OSError):
        probe_import(sha, destination_root=destination)


@needs_media_tools
def test_discover_reports_an_unreadable_registry_as_unknown_not_unregistered(
    tmp_path, monkeypatch
) -> None:
    """S4 / runner 4: registry_error, and register_recorded null, never false."""

    _session, sha, destination, _imported = _import(tmp_path, monkeypatch)
    for registry in (tmp_path / "absent.sqlite", tmp_path / "garbage.sqlite"):
        if registry.name == "garbage.sqlite":
            registry.write_bytes(b"not a database")
        found = discover(tmp_path / "staging", destination, registry=registry)
        assert found.registry_error
        [target] = found.targets
        assert target.state == "complete" and target.register_recorded is None
        assert target.producer_git_sha == GIT_SHA
        assert found.to_json()["registry_error"] == found.registry_error


def test_compatibility_entry_finds_a_retired_delivery_without_its_marker(
    tmp_path, placeholder_media
) -> None:
    """S5: a missing marker resolves through durable state, else refuses (65)."""

    from argparse import Namespace

    from fisheye.utils import citrus_transfer_parent_workflow as workflow

    session, sha = _delivery(tmp_path, "rolling")
    destination = _destination(tmp_path)
    args = Namespace(session_dir=session, dest_root=destination, resume_transfer_plan=None)
    plan = organizer.build_transfer_organization_plan(session, destination_root=destination)
    (session / MARKER_NAME).unlink()
    with pytest.raises(IntakeRefused, match="--resume-transfer-plan"):
        workflow._snapshot_sha(args)
    with organizer._organization_state(plan):
        pass  # the organizer reserves the delivery's durable state
    assert workflow._snapshot_sha(args) == f"sha256:{sha}"


def _cli_code(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    code = "import os, signal, sys, fisheye.intake.__main__ as m\n" + body + (
        "sys.exit(m.main(['discover', '--staging-dir', %r, '--registry', %r]))\n"
        % (str(tmp_path), str(tmp_path / "r.sqlite"))
    )
    return subprocess.run([str(REPO / "scripts/py"), "-c", code], capture_output=True,
                          text=True, timeout=120)


@pytest.mark.parametrize(
    "body",
    [
        "def work(args):\n    raise SystemExit(0)\nm._run = work\n",
        "def work(args):\n    raise KeyboardInterrupt()\nm._run = work\n",
        "def work(args):\n    os.kill(os.getpid(), signal.SIGTERM)\nm._run = work\n",
    ],
    ids=["systemexit", "keyboardinterrupt", "sigterm"],
)
def test_cli_interrupted_work_still_emits_json_and_retries(tmp_path, body) -> None:
    """N2."""

    result = _cli_code(tmp_path, body)
    assert result.returncode == 1, result.stderr
    document = json.loads(result.stdout)
    assert document["schema"] == "palette.intake.error.v1"
    assert document["error"].startswith("interrupted")


def test_cli_buffered_original_stdout_never_follows_the_json(tmp_path) -> None:
    """N1."""

    result = _cli_code(
        tmp_path,
        "def work(args):\n"
        "    sys.__stdout__.write('library chatter\\n')\n"
        "    return 0, {'ok': True}\n"
        "m._run = work\n",
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"ok": True}
    assert "library chatter" in result.stderr


def test_canonical_registry_identity_is_by_inode(tmp_path) -> None:
    """N4: another path to the same file is the same registry."""

    from fisheye.intake.delivery import same_file

    original = tmp_path / "registry.sqlite"
    original.write_bytes(b"x")
    linked = tmp_path / "elsewhere" / "alias.sqlite"
    linked.parent.mkdir()
    os.link(original, linked)
    assert same_file(linked, original)
    assert not same_file(tmp_path / "missing.sqlite", original)
    other = tmp_path / "copy.sqlite"
    other.write_bytes(b"x")
    assert not same_file(other, original)
