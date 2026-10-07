"""The generated LSF job script runs the intake workflow and leaves its status where readers look."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "submit_citrus_session_import_bsub.sh"
FIXTURE = REPO / "tests" / "fixtures" / "recording_transfer_v2" / "rolling"
KEY = "c" * 64


def test_job_script_writes_the_workflow_status_json(tmp_path: Path) -> None:
    session = Path(shutil.copytree(FIXTURE, tmp_path / "staging" / "session"))
    logs = tmp_path / "logs"
    launch = subprocess.run(
        ["bash", str(SCRIPT), "--session-dir", str(session), "--marker-key", KEY,
         "--log-dir", str(logs), "--run-id", "e2e", "--dest-root", str(tmp_path / "recordings"),
         "--no-register", "--dry-run"],
        check=False, text=True, capture_output=True,
    )
    assert launch.returncode == 0, launch.stderr
    job_script = next(logs.glob("citrus_import_*/run_citrus_session_import.sh"))

    job = subprocess.run(
        ["bash", str(job_script)], check=False, text=True, capture_output=True,
        env=dict(os.environ, LSB_JOBID="777"), timeout=600,
    )
    run_dir = job_script.parent
    status_json = run_dir / "workflow-777" / "citrus_session_import.status.json"
    # The fixture's media are placeholders, so the import may fail; what must
    # hold is that the workflow ran (not refused its run dir) and wrote status.
    assert status_json.is_file(), job.stdout + job.stderr
    status = json.loads(status_json.read_text())
    assert status["schema_id"] == "palette.citrus_transfer_parent_intake.status.v1"
    assert "overlaps workflow outputs" not in str(status.get("error"))
    status_txt = next(run_dir.glob("*.777.status.txt")).read_text()
    assert f"status_json={status_json}" in status_txt
