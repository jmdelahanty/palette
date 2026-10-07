"""The Citrus import launcher submits each delivery (marker key) at most once."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "submit_citrus_session_import_bsub.sh"
KEY = "a" * 64


def _fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _setup(tmp_path: Path, *, bsub_output: str, bjobs_output: str = "") -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "bsub_calls"
    _fake(bin_dir, "bsub", f'printf "%s\\n" "$*" >>{calls}\nprintf "%s\\n" "{bsub_output}"\n')
    bjobs_reply = tmp_path / "bjobs_reply"
    bjobs_reply.write_text(bjobs_output)
    _fake(bin_dir, "bjobs", f"cat {bjobs_reply}\n")
    (tmp_path / "session").mkdir()
    return {"bin": bin_dir, "calls": calls, "bjobs": bjobs_reply}


def _launch(tmp_path: Path, fakes: dict, run_id: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", str(SCRIPT), "--session-dir", str(tmp_path / "session"),
         "--marker-key", KEY, "--log-dir", str(tmp_path / "logs"),
         "--run-id", run_id, "--no-register"],
        check=False, text=True, capture_output=True, env=env,
    )


def _bsub_calls(fakes: dict) -> list[str]:
    return fakes["calls"].read_text().splitlines() if fakes["calls"].exists() else []


def test_retry_after_success_reuses_the_job(tmp_path: Path) -> None:
    fakes = _setup(tmp_path, bsub_output="Job <4242> is submitted to default queue <normal>.")
    first = _launch(tmp_path, fakes, "attempt-1")
    second = _launch(tmp_path, fakes, "attempt-2")

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert len(_bsub_calls(fakes)) == 1
    assert f"-J citrus_import_{KEY}" in _bsub_calls(fakes)[0]
    assert "already_submitted=1" in second.stdout and "job_id=4242" in second.stdout


def test_ambiguous_failure_then_retry_finds_the_accepted_job(tmp_path: Path) -> None:
    # LSF accepted the job but its reply was unparseable: the launcher fails,
    # and the caller retries. The retry must find the job, not submit again.
    fakes = _setup(tmp_path, bsub_output="garbled", bjobs_output="")
    first = _launch(tmp_path, fakes, "attempt-1")
    assert first.returncode != 0
    fakes["bjobs"].write_text("5151 PEND\n")  # LSF had in fact accepted it

    second = _launch(tmp_path, fakes, "attempt-2")
    assert second.returncode == 0, second.stderr
    assert len(_bsub_calls(fakes)) == 1
    assert "job_id=5151" in second.stdout
    record = tmp_path / "logs" / "by_marker" / f"{KEY}.job"
    assert "job_id=5151" in record.read_text()


def test_a_genuinely_failed_submission_is_retried(tmp_path: Path) -> None:
    fakes = _setup(tmp_path, bsub_output="garbled", bjobs_output="")  # LSF has no such job
    assert _launch(tmp_path, fakes, "attempt-1").returncode != 0
    assert _launch(tmp_path, fakes, "attempt-2").returncode != 0
    assert len(_bsub_calls(fakes)) == 2  # nothing was accepted, so retrying is correct


def test_a_finished_job_with_the_name_does_not_block_a_resubmission(tmp_path: Path) -> None:
    # bjobs -a still lists EXIT/DONE jobs for LSF's clean period. After an
    # operator clears the poller state and the by_marker record to retry a
    # failed import, a finished job must not count as "already submitted".
    for finished in ("EXIT", "DONE"):
        case = tmp_path / finished
        case.mkdir()
        fakes = _setup(case, bsub_output="Job <6262> is submitted to default queue <normal>.",
                       bjobs_output=f"5151 {finished}\n")
        result = _launch(case, fakes, "retry")
        assert result.returncode == 0, result.stderr
        assert len(_bsub_calls(fakes)) == 1
        assert "already_submitted=1" not in result.stdout and "job_id=6262" in result.stdout


@pytest.mark.parametrize("live", ["PEND", "RUN", "PSUSP", "USUSP", "SSUSP"])
def test_a_live_job_with_the_name_is_the_existing_submission(tmp_path: Path, live: str) -> None:
    fakes = _setup(tmp_path, bsub_output="garbled", bjobs_output=f"5151 EXIT\n7373 {live}\n")
    result = _launch(tmp_path, fakes, "retry")
    assert result.returncode == 0, result.stderr
    assert _bsub_calls(fakes) == [] and "job_id=7373" in result.stdout
