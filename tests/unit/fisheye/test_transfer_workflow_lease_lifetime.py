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

# Exercise the real public workflow, preparation and lease. The import owner
# runs in-process (review F1); its one child writer, the stimulus importer, is
# lent the lease through PALETTE_RECORDING_IMPORT_LEASE_FD. Replace only that
# child's payload with a harmless writer spawned the owner's way.
SUPERVISOR = (
    r"""
import json
from pathlib import Path
import subprocess
import sys
from fisheye.utils import import_recording_analysis as importer
from fisheye.utils import run_citrus_session_import as runner

worker = """
    + repr(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "Path(sys.argv[2]).write_text(str(os.getpid()))\n"
        "release = Path(sys.argv[1])\n"
        "deadline = time.monotonic() + 20\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
    )
    + r"""

def harmless_stimulus_writer(plan, options, **_):
    subprocess.run(
        [sys.executable, "-c", worker, sys.argv[2], sys.argv[3]],
        check=False,
        pass_fds=importer._stimulus_import_lease_fds(),
    )
    return importer.RecordingImportResult(ok=False, failed_step="stimulus", error="fixture")

importer.process_recording_import = harmless_stimulus_writer
# The pinned transfer fixture's MP4s are text placeholders (see its README);
# the real sync-sample check runs against real H264 in the packaged canary.
from fisheye.diagnostics.video import container
container.check_hevc_keyframe_flags = lambda path, **_: {
    "container_inspection_status": "ok",
    "sync_sample_proof": "container_declared",
    "message": "placeholder media",
}
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
    )
    arguments = [
        str(source),
        "--apply",
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
            str(tmp_path / "writer.pid"),
        ],
        cwd=REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    writer_pid = None
    try:
        writer_pid = _wait_for_writer(supervisor, tmp_path / "writer.pid")
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
