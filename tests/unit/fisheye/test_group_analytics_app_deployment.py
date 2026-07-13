from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pixi_exposes_one_lightweight_public_app_task() -> None:
    config = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))

    assert config["workspace"]["platforms"] == ["linux-64"]
    assert config["tasks"] == {
        "app": {
            "cmd": "bash scripts/run_group_analytics_app.sh",
            "description": "Start the read-only Palette analytics Marimo application.",
        }
    }
    assert "pyarrow" not in config["dependencies"]
    assert config["dependencies"]["polars"] == "==1.40.0"


def test_fileglancer_service_delegates_to_the_pixi_app() -> None:
    manifest = yaml.safe_load((REPO_ROOT / "runnables.yaml").read_text(encoding="utf-8"))
    runnable = manifest["runnables"][0]

    assert manifest["requirements"] == ["pixi>=0.40"]
    assert runnable["id"] == "app"
    assert runnable["type"] == "service"
    assert runnable["command"] == "pixi run app --"
    assert runnable["working_dir"] == "repo"
    assert runnable["auto_url"] is True
    assert runnable["service_url_suffix"] == "/?access_token=${FG_SERVICE_TOKEN}"
    assert runnable["parameters"][0]["flag"] == "--export-root"
    assert runnable["parameters"][0]["type"] == "directory"
    assert runnable["parameters"][0]["required"] is True


def _run_launcher(tmp_path: Path, *, extra_env: dict[str, str] | None = None) -> list[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_marimo = fake_bin / "marimo"
    fake_marimo.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_marimo.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TMPDIR": str(tmp_path / "tmp"),
    }
    env.pop("FG_SERVICE_PORT", None)
    env.pop("FG_SERVICE_TOKEN", None)
    env.pop("PALETTE_ANALYTICS_APP_HOST", None)
    env.pop("PALETTE_ANALYTICS_APP_PORT", None)
    env.pop("PALETTE_ANALYTICS_APP_TOKEN", None)
    env.update(extra_env or {})
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_group_analytics_app.sh"),
            "--",
            "--export-root",
            "/shared/analytics",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def test_launcher_uses_safe_local_defaults_and_forwards_notebook_arguments(tmp_path: Path) -> None:
    arguments = _run_launcher(tmp_path)

    assert arguments[:7] == [
        "run",
        "--headless",
        "--host",
        "127.0.0.1",
        "--port",
        "2718",
        "--no-token",
    ]
    assert arguments[-3:] == ["--", "--export-root", "/shared/analytics"]


def test_launcher_uses_fileglancer_service_endpoint_and_token(tmp_path: Path) -> None:
    arguments = _run_launcher(
        tmp_path,
        extra_env={"FG_SERVICE_PORT": "31876", "FG_SERVICE_TOKEN": "service-secret"},
    )

    assert arguments[2:6] == ["--host", "0.0.0.0", "--port", "31876"]
    assert arguments[6:8] == ["--token-password", "service-secret"]
    assert "--no-token" not in arguments
