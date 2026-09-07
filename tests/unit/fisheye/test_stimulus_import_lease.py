"""Opt-in workflow-lease inheritance at the actual stimulus subprocess boundary."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from fisheye.utils import import_recording_analysis as importer

LEASE_ENV = "PALETTE_RECORDING_IMPORT_LEASE_FD"


def _plan_and_options(tmp_path: Path):
    return (
        importer.RecordingAnalysisPlan(
            recording_dir=tmp_path,
            h5_path=tmp_path / "protocol.h5",
            cam_video=tmp_path / "full.mp4",
            zarr_path=tmp_path / "analysis.zarr",
        ),
        importer.RecordingImportOptions(
            import_video_metadata=True,
            video_metadata_overwrite=False,
            import_stimulus=True,
            stimulus_always=False,
            stimulus_run_name="selected-run",
            stimulus_overwrite=True,
            stimulus_quiet=True,
            stimulus_metadata_and_calibration_only=True,
        ),
    )


def test_no_lease_preserves_exact_legacy_command_and_subprocess_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(LEASE_ENV, raising=False)
    plan, options = _plan_and_options(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(importer.subprocess, "run", run)
    ok, returncode, command = importer.run_stimulus_import(plan, options)
    assert (ok, returncode) == (True, 0)
    assert command == [
        sys.executable,
        "-m",
        "fisheye.analysis.import_stimulus_to_zarr",
        str(plan.h5_path),
        str(plan.zarr_path),
        "--run-name",
        "selected-run",
        "--overwrite",
        "--quiet",
        "--metadata-and-calibration-only",
    ]
    assert calls == [(command, {"check": False})]


@pytest.mark.parametrize("outcome", [0, 7, "raise"])
def test_declared_regular_lease_is_forwarded_without_close_or_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: int | str
) -> None:
    plan, options = _plan_and_options(tmp_path)
    lock_path = tmp_path / "workflow.lock"
    with lock_path.open("a+b") as lease, lock_path.open("a+b") as retry:
        descriptor = lease.fileno()
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv(LEASE_ENV, str(descriptor))

        def run(_command, **kwargs):
            assert kwargs == {"check": False, "pass_fds": (descriptor,)}
            if outcome == "raise":
                raise OSError("synthetic child launch failure")
            return SimpleNamespace(returncode=outcome)

        monkeypatch.setattr(importer.subprocess, "run", run)
        if outcome == "raise":
            with pytest.raises(OSError, match="synthetic child launch failure"):
                importer.run_stimulus_import(plan, options)
        else:
            ok, returncode, _command = importer.run_stimulus_import(plan, options)
            assert (ok, returncode) == (outcome == 0, outcome)
        os.fstat(descriptor)
        with pytest.raises(BlockingIOError):
            fcntl.flock(retry.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    "value",
    ["", "-1", "+1", " 1", "1 ", "1.0", "abc", "١", "²", "1\n", "9" * 40],
)
def test_invalid_declared_lease_refuses_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    plan, options = _plan_and_options(tmp_path)
    monkeypatch.setenv(LEASE_ENV, value)

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid declared lease reached a subprocess")

    monkeypatch.setattr(importer.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match=LEASE_ENV):
        importer.run_stimulus_import(plan, options)


@pytest.mark.parametrize("kind", ["closed", "directory", "pipe"])
def test_unusable_declared_lease_refuses_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    plan, options = _plan_and_options(tmp_path)
    descriptors = []
    if kind == "pipe":
        descriptors.extend(os.pipe())
        selected = descriptors[0]
    elif kind == "directory":
        selected = os.open(tmp_path, os.O_RDONLY)
        descriptors.append(selected)
    else:
        selected = os.open(tmp_path / "closed.lock", os.O_CREAT | os.O_RDWR, 0o600)
        os.close(selected)
    monkeypatch.setenv(LEASE_ENV, str(selected))

    def forbidden(*_args, **_kwargs):
        pytest.fail("unusable declared lease reached a subprocess")

    monkeypatch.setattr(importer.subprocess, "run", forbidden)
    try:
        with pytest.raises(ValueError, match=LEASE_ENV):
            importer.run_stimulus_import(plan, options)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _wait_until(predicate, *, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for the disposable subprocess")
        time.sleep(0.01)


def test_actual_stimulus_leaf_keeps_lease_after_batch_sigkill(tmp_path: Path) -> None:
    """Use the unpatched producer call with only its child module made harmless."""
    module_root = tmp_path / "harmless_modules"
    package = module_root / "fisheye" / "analysis"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "import_stimulus_to_zarr.py").write_text(
        "import json, os, pathlib, sys, time\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "fd = int(os.environ['PALETTE_RECORDING_IMPORT_LEASE_FD'])\n"
        "try:\n"
        "    info = os.fstat(fd)\n"
        "    identity = [info.st_dev, info.st_ino]\n"
        "except OSError:\n"
        "    identity = None\n"
        "(root / 'ready.json.tmp').write_text(json.dumps({'pid': os.getpid(), 'identity': identity}))\n"
        "(root / 'ready.json.tmp').replace(root / 'ready.json')\n"
        "deadline = time.monotonic() + 20\n"
        "while not (root / 'release').exists():\n"
        "    if time.monotonic() >= deadline:\n"
        "        raise SystemExit(8)\n"
        "    time.sleep(0.01)\n"
        "(root / 'harmless-write-complete').write_text('one writer finished')\n"
    )
    batch_script = tmp_path / "batch.py"
    batch_script.write_text(
        "import os, pathlib, sys\n"
        "from fisheye.utils.import_recording_analysis import (\n"
        "    RecordingAnalysisPlan, RecordingImportOptions, run_stimulus_import)\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "os.environ['PYTHONPATH'] = str(root / 'harmless_modules')\n"
        "plan = RecordingAnalysisPlan(root, root, None, root / 'unused.zarr')\n"
        "options = RecordingImportOptions(True, False, True, False, None, False, True)\n"
        "ok, code, command = run_stimulus_import(plan, options)\n"
        "raise SystemExit(code)\n"
    )
    lock_path = tmp_path / "workflow.lock"
    ready_path = tmp_path / "ready.json"
    lease = lock_path.open("a+b")
    descriptor = lease.fileno()
    identity = os.fstat(descriptor)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = {**os.environ, LEASE_ENV: str(descriptor)}
    repo = Path(__file__).resolve().parents[3]
    batch = subprocess.Popen(
        [str(repo / "scripts" / "py"), str(batch_script), str(tmp_path)],
        env=environment,
        pass_fds=(descriptor,),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _wait_until(ready_path.exists)
        child = json.loads(ready_path.read_text())
        lease.close()
        batch.kill()
        assert batch.wait(timeout=5) == -signal.SIGKILL
        os.kill(child["pid"], 0)
        with lock_path.open("a+b") as retry:
            with pytest.raises(BlockingIOError):
                fcntl.flock(retry.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert child["identity"] == [identity.st_dev, identity.st_ino]
            (tmp_path / "release").touch()

            def retry_acquires() -> bool:
                try:
                    fcntl.flock(retry.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    return False

            _wait_until(retry_acquires)
            assert (
                tmp_path / "harmless-write-complete"
            ).read_text() == "one writer finished"
    finally:
        lease.close()
        try:
            os.killpg(batch.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        batch.wait(timeout=5)
        if batch.stderr is not None:
            batch.stderr.close()
