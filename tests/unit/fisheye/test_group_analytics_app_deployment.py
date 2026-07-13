from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pixi_exposes_separate_group_and_recording_app_tasks() -> None:
    config = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))

    assert config["workspace"]["platforms"] == ["linux-64"]
    assert set(config["tasks"]) == {"app", "recording-app"}
    assert config["tasks"]["app"]["cmd"] == "bash scripts/run_group_analytics_app.sh"
    assert config["tasks"]["recording-app"]["cmd"] == (
        "bash scripts/run_recording_explorer_app.sh"
    )
    assert config["environments"]["recording"]["features"] == ["recording"]
    assert set(config["feature"]["recording"]["dependencies"]) == {
        "matplotlib-base",
        "pyyaml",
        "rich",
        "scipy",
        "zarr",
    }
    assert "pyarrow" not in config["dependencies"]
    assert "pyarrow" not in config["feature"]["recording"]["dependencies"]
    assert config["dependencies"]["polars"] == "==1.40.0"


def test_fileglancer_service_delegates_to_the_pixi_app() -> None:
    manifest = yaml.safe_load((REPO_ROOT / "runnables.yaml").read_text(encoding="utf-8"))
    runnables = {item["id"]: item for item in manifest["runnables"]}
    runnable = runnables["app"]

    assert manifest["requirements"] == ["pixi>=0.40"]
    assert set(runnables) == {"app", "recording-app"}
    assert runnable["id"] == "app"
    assert runnable["type"] == "service"
    assert runnable["command"] == "pixi run app --"
    assert runnable["working_dir"] == "repo"
    assert runnable["auto_url"] is True
    assert runnable["service_url_suffix"] == "/?access_token=${FG_SERVICE_TOKEN}"
    assert runnable["parameters"][0]["flag"] == "--export-root"
    assert runnable["parameters"][0]["type"] == "directory"
    assert runnable["parameters"][0]["required"] is True

    recording = runnables["recording-app"]
    assert recording["type"] == "service"
    assert recording["command"] == "pixi run -e recording recording-app --"
    assert recording["working_dir"] == "repo"
    assert recording["auto_url"] is True
    assert recording["service_url_suffix"] == "/?access_token=${FG_SERVICE_TOKEN}"
    assert recording["parameters"] == [
        {
            "flag": "--zarr-path",
            "name": "Recording Analysis Zarr",
            "type": "directory",
            "description": (
                "One Palette analysis Zarr directory. Direct FileGlancer launches show "
                "only this recording unless collection arguments are supplied separately."
            ),
            "required": True,
            "exists": True,
        }
    ]


def _run_launcher(
    tmp_path: Path,
    *,
    script_name: str = "run_group_analytics_app.sh",
    notebook_args: tuple[str, ...] = ("--export-root", "/shared/analytics"),
    extra_env: dict[str, str] | None = None,
) -> list[str]:
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
    env.pop("PALETTE_RECORDING_APP_HOST", None)
    env.pop("PALETTE_RECORDING_APP_PORT", None)
    env.pop("PALETTE_RECORDING_APP_TOKEN", None)
    env.update(extra_env or {})
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / f"scripts/{script_name}"),
            "--",
            *notebook_args,
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


def test_recording_launcher_uses_recording_defaults_and_forwards_zarr(tmp_path: Path) -> None:
    arguments = _run_launcher(
        tmp_path,
        script_name="run_recording_explorer_app.sh",
        notebook_args=("--zarr-path", "/shared/recording_analysis.zarr"),
    )

    assert arguments[:7] == [
        "run",
        "--headless",
        "--host",
        "127.0.0.1",
        "--port",
        "2720",
        "--no-token",
    ]
    assert arguments[-3:] == [
        "--",
        "--zarr-path",
        "/shared/recording_analysis.zarr",
    ]


def test_recording_launcher_uses_fileglancer_endpoint_and_token(tmp_path: Path) -> None:
    arguments = _run_launcher(
        tmp_path,
        script_name="run_recording_explorer_app.sh",
        notebook_args=("--zarr-path", "/shared/recording_analysis.zarr"),
        extra_env={"FG_SERVICE_PORT": "31877", "FG_SERVICE_TOKEN": "service-secret"},
    )

    assert arguments[2:6] == ["--host", "0.0.0.0", "--port", "31877"]
    assert arguments[6:8] == ["--token-password", "service-secret"]
    assert "--no-token" not in arguments
