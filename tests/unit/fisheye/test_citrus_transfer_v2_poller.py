"""Citrus transfer-v2 staging poller: detection, refusal, dedupe, dry-run."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from fisheye.utils import citrus_transfer_v2_poller as poller

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "recording_transfer_v2" / "rolling"


def _config(tmp_path: Path, **overrides) -> dict:
    config = {
        "staging_dir": str(tmp_path / "staging"),
        "state_dir": str(tmp_path / "state"),
        "log_dir": str(tmp_path / "logs"),
        "submit": {"transport": "ssh", "host": "submit-host", "repo": "/groups/palette"},
    }
    config.update(overrides)
    (tmp_path / "state").mkdir(exist_ok=True)
    path = tmp_path / "poller.json"
    path.write_text(json.dumps(config))
    return poller.load_config(path)


def _session(tmp_path: Path, name: str = "session_a") -> Path:
    dest = tmp_path / "staging" / name
    shutil.copytree(FIXTURE, dest)
    return dest


class FakeRunner:
    def __init__(self, returncode: int = 0):
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, command):
        self.calls.append(command)
        return subprocess.CompletedProcess(command, self.returncode, "job_id=42\n", "")


def test_v2_marker_submits_launcher_without_operator_context(tmp_path):
    session = _session(tmp_path)
    runner = FakeRunner()
    assert poller.poll(_config(tmp_path), dry_run=False, runner=runner) == 0
    [command] = runner.calls
    assert command[:2] == ["ssh", "submit-host"]
    remote = command[2]
    assert remote.startswith("cd /groups/palette && scripts/submit_citrus_session_import_bsub.sh")
    assert f"--session-dir {session}" in remote
    for flag in ("--recording-type", "--recording-subtype", "--behavior-mode", "--recording-only"):
        assert flag not in remote
    [submitted] = (tmp_path / "state").glob("*.submitted")
    assert submitted.read_text() == "job_id=42\n"


def test_v1_marker_ignored(tmp_path, capsys):
    session = tmp_path / "staging" / "legacy"
    session.mkdir(parents=True)
    (session / poller.MARKER_NAME).write_text(json.dumps(
        {"schema_id": "citrus.transfer_completion_marker.v1", "status": "transfer_complete"}))
    runner = FakeRunner()
    assert poller.poll(_config(tmp_path), dry_run=False, runner=runner) == 0
    assert runner.calls == []
    assert "legacy v1 marker ignored" in capsys.readouterr().out
    assert list((tmp_path / "state").iterdir()) == []


@pytest.mark.parametrize(
    "stale", ["recording_type", "recording_subtype", "behavior_mode", "recording_only"]
)
def test_config_that_still_declares_context_refuses(tmp_path, stale):
    with pytest.raises(poller.PollerRefusal, match=stale):
        _config(tmp_path, **{stale: "free"})


def test_main_refuses_incomplete_config(tmp_path):
    (tmp_path / "poller.json").write_text(json.dumps({"staging_dir": str(tmp_path)}))
    assert poller.main(["--config", str(tmp_path / "poller.json")]) == 2


def test_marker_for_the_old_consumer_profile_is_refused(tmp_path, capsys):
    session = _session(tmp_path)
    marker_path = session / poller.MARKER_NAME
    marker = json.loads(marker_path.read_text())
    marker["required_consumer_profile"] = "parent_recording_intake_v1"
    marker_path.write_text(json.dumps(marker))
    runner = FakeRunner()
    poller.poll(_config(tmp_path), dry_run=False, runner=runner)
    assert runner.calls == []
    assert "not a complete transfer-v2 marker" in capsys.readouterr().out


def test_duplicate_run_does_not_resubmit_but_changed_marker_does(tmp_path):
    session = _session(tmp_path)
    config, runner = _config(tmp_path), FakeRunner()
    poller.poll(config, dry_run=False, runner=runner)
    poller.poll(config, dry_run=False, runner=runner)
    assert len(runner.calls) == 1
    marker_path = session / poller.MARKER_NAME
    marker = json.loads(marker_path.read_text())
    marker["delivery"]["attempt_id"] = "1" * 32
    marker_path.write_text(json.dumps(marker))
    poller.poll(config, dry_run=False, runner=runner)
    assert len(runner.calls) == 2
    assert runner.calls[0] != runner.calls[1]  # new marker key


def test_failed_submission_releases_claim_for_retry(tmp_path):
    _session(tmp_path)
    config = _config(tmp_path)
    assert poller.poll(config, dry_run=False, runner=FakeRunner(returncode=1)) == 1
    assert list((tmp_path / "state").glob("*.claimed")) == []
    runner = FakeRunner()
    poller.poll(config, dry_run=False, runner=runner)
    assert len(runner.calls) == 1


@pytest.mark.parametrize("mutate", [
    lambda p: p.write_text("{not json"),
    lambda p: p.write_text(json.dumps({**json.loads(p.read_text()), "schema_id": "citrus.other"})),
    lambda p: (p.parent / poller.SNAPSHOT_PATH).write_text("{}"),
    lambda p: (p.parent / poller.SNAPSHOT_PATH).unlink(),
])
def test_malformed_marker_refused(tmp_path, capsys, mutate):
    session = _session(tmp_path)
    mutate(session / poller.MARKER_NAME)
    runner = FakeRunner()
    poller.poll(_config(tmp_path), dry_run=False, runner=runner)
    assert runner.calls == []
    assert "refused marker=" in capsys.readouterr().out
    assert list((tmp_path / "state").iterdir()) == []


def test_dry_run_has_no_side_effects(tmp_path, capsys):
    _session(tmp_path)
    config = _config(tmp_path)
    (tmp_path / "poller.json").write_text(json.dumps(config))
    before = sorted(p for p in tmp_path.rglob("*"))
    runner = FakeRunner()
    assert poller.poll(config, dry_run=True, runner=runner) == 0
    assert poller.main(["--config", str(tmp_path / "poller.json"), "--dry-run"]) == 0
    assert runner.calls == []
    assert sorted(p for p in tmp_path.rglob("*")) == before
    assert "dry-run: would submit ssh submit-host" in capsys.readouterr().out


def test_local_transport_builds_bash_command(tmp_path):
    config = _config(tmp_path, submit={"transport": "local", "repo": "/r"})
    command = poller.build_command(config, Path("/s"), "k")
    assert command[:2] == ["bash", "-c"] and command[2].startswith("cd /r && scripts/")
