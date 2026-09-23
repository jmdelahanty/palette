"""A surviving import writer must retain its supervisor's exclusive lease."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from fisheye.utils import organize_transfer_recordings as organizer

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"
REPOSITORY = Path(__file__).resolve().parents[3]

# Exercise the real public workflow, preparation, command runner and lease.
# Replace only the importer command's payload, never its descriptor handling.
SUPERVISOR = (
    r"""
import json
from pathlib import Path
import sys
from fisheye.utils import citrus_transfer_parent_workflow as workflow
from fisheye.utils import run_citrus_session_import as runner

worker = """
    + repr(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "print(os.getpid(), flush=True)\n"
        "release = Path(sys.argv[1])\n"
        "deadline = time.monotonic() + 20\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
    )
    + r"""
actual_run_command = runner._run_command

def harmless_writer(command, *, name, run_dir, **kwargs):
    return actual_run_command(
        [sys.executable, "-c", worker, sys.argv[2]],
        name=name, run_dir=run_dir, **kwargs,
    )

workflow._run_command = harmless_writer
raise SystemExit(runner.main(json.loads(sys.argv[1])))
"""
)


def _wait_for_writer(supervisor: subprocess.Popen, output: Path) -> int:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if output.is_file() and output.read_text().strip():
            return int(output.read_text().strip())
        if supervisor.poll() is not None:
            _, errors = supervisor.communicate(timeout=1)
            pytest.fail(f"supervisor exited before starting its writer: {errors}")
        time.sleep(0.02)
    pytest.fail("supervisor did not start its harmless writer")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux flock/SIGKILL regression")
def test_writer_retains_workflow_lease_after_supervisor_sigkill(tmp_path):
    source = Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))
    destination = tmp_path / "recordings"
    run_dir = tmp_path / "run"
    release = tmp_path / "release-writer"
    plan = organizer.build_transfer_organization_plan(
        source,
        destination_root=destination,
        recording_type="behavior",
        recording_subtype="free",
        behavior_mode="free",
    )
    arguments = [
        str(source),
        "--transfer-v2",
        "--recording-only",
        "--apply",
        "--recording-type",
        "behavior",
        "--recording-subtype",
        "free",
        "--behavior-mode",
        "free",
        "--dest-root",
        str(destination),
        "--run-dir",
        str(run_dir),
    ]
    supervisor = subprocess.Popen(
        [
            str(REPOSITORY / "scripts/py"),
            "-c",
            SUPERVISOR,
            json.dumps(arguments),
            str(release),
        ],
        cwd=REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    writer_pid = None
    try:
        writer_pid = _wait_for_writer(supervisor, run_dir / "import_parents.stdout.txt")
        supervisor.kill()
        assert supervisor.wait(timeout=5) == -signal.SIGKILL
        os.kill(writer_pid, 0)  # The original writer has not exited with its parent.

        with pytest.raises(BlockingIOError):
            with organizer.transfer_parent_workflow_lock(plan):
                pytest.fail("retry admitted while the orphaned writer is still live")

        release.touch()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with organizer.transfer_parent_workflow_lock(plan):
                    break
            except BlockingIOError:
                time.sleep(0.02)
        else:
            pytest.fail("lease remained held after its writer was released")
    finally:
        release.touch(exist_ok=True)
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=5)
        if writer_pid is not None:
            try:
                os.kill(writer_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        supervisor.communicate(timeout=5)
