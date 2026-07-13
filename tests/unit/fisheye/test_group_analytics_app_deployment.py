from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pixi_exposes_group_recording_and_editable_workspace_tasks() -> None:
    config = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))

    assert config["workspace"]["platforms"] == ["linux-64"]
    assert set(config["tasks"]) == {"app", "recording-app", "recording-workspace"}
    assert config["tasks"]["app"]["cmd"] == "bash scripts/run_group_analytics_app.sh"
    assert config["tasks"]["recording-app"]["cmd"] == (
        "bash scripts/run_recording_explorer_app.sh"
    )
    assert config["tasks"]["recording-workspace"]["cmd"] == (
        "bash scripts/run_recording_exploration_workspace.sh"
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
    assert set(runnables) == {"app", "recording-app", "recording-workspace"}
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

    workspace = runnables["recording-workspace"]
    assert workspace["type"] == "service"
    assert workspace["command"] == "pixi run -e recording recording-workspace --"
    assert workspace["working_dir"] == "repo"
    assert workspace["auto_url"] is True
    assert workspace["service_url_suffix"] == "/?access_token=${FG_SERVICE_TOKEN}"
    assert "Bubblewrap" in workspace["description"]
    assert "read-only" in workspace["description"]
    assert workspace["parameters"] == [
        {
            "flag": "--zarr-path",
            "name": "Recording Analysis Zarr",
            "type": "directory",
            "description": (
                "Exactly one Palette analysis Zarr. It is exposed inside the editable "
                "workspace as the read-only /data/recording.zarr mount."
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
    env.pop("PALETTE_RECORDING_WORKSPACE_HOST", None)
    env.pop("PALETTE_RECORDING_WORKSPACE_PORT", None)
    env.pop("PALETTE_RECORDING_WORKSPACE_TOKEN", None)
    env.pop("PALETTE_RECORDING_WORKSPACE_ROOT", None)
    env.pop("PALETTE_RECORDING_WORKSPACE_PYTHON", None)
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


def _run_workspace_launcher(
    tmp_path: Path,
    *,
    extra_args: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bwrap = fake_bin / "bwrap"
    fake_bwrap.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_bwrap.chmod(0o755)

    fake_python = tmp_path / "fake-pixi" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir()
    workspace_root = tmp_path / "workspaces"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PALETTE_RECORDING_WORKSPACE_PYTHON": str(fake_python),
        "PALETTE_RECORDING_WORKSPACE_ROOT": str(workspace_root),
    }
    for name in (
        "FG_SERVICE_PORT",
        "FG_SERVICE_TOKEN",
        "PALETTE_RECORDING_WORKSPACE_HOST",
        "PALETTE_RECORDING_WORKSPACE_PORT",
        "PALETTE_RECORDING_WORKSPACE_TOKEN",
    ):
        env.pop(name, None)
    env.update(extra_env or {})
    return subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_recording_exploration_workspace.sh"),
            "--",
            "--zarr-path",
            str(zarr_path),
            *extra_args,
        ],
        cwd=REPO_ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def _mount_pairs(arguments: list[str], option: str) -> list[tuple[str, str]]:
    return [
        (arguments[index + 1], arguments[index + 2])
        for index, value in enumerate(arguments[:-2])
        if value == option
    ]


def test_recording_workspace_mounts_only_the_dataset_and_code_read_only(
    tmp_path: Path,
) -> None:
    completed = _run_workspace_launcher(tmp_path)
    arguments = completed.stdout.splitlines()

    assert arguments[:2] == ["--die-with-parent", "--new-session"]
    assert "--clearenv" in arguments
    read_only = _mount_pairs(arguments, "--ro-bind")
    writable = _mount_pairs(arguments, "--bind")
    zarr_path = str((tmp_path / "recording_analysis.zarr").resolve())
    assert (zarr_path, "/data/recording.zarr") in read_only
    assert (str(REPO_ROOT), str(REPO_ROOT)) in read_only
    assert {destination for _, destination in writable} == {"/tmp", "/workspace"}
    assert all(source != zarr_path for source, _ in writable)

    command_index = arguments.index("-m")
    assert arguments[command_index : command_index + 3] == ["-m", "marimo", "edit"]
    assert "--headless" in arguments[command_index:]
    assert "--skip-update-check" in arguments[command_index:]
    assert "--no-token" in arguments[command_index:]
    notebook_args = arguments[arguments.index("--zarr-path", command_index) :]
    assert notebook_args[:4] == [
        "--zarr-path",
        "/data/recording.zarr",
        "--workspace",
        "true",
    ]

    notebook_copies = list((tmp_path / "workspaces").glob("*/palette_recording_workspace.py"))
    assert len(notebook_copies) == 1
    notebook_source = notebook_copies[0].read_text(encoding="utf-8")
    assert "@app.cell(hide_code=True)" in notebook_source
    assert "exploration = recording_workspace if workspace_mode else None" in notebook_source


def test_recording_workspace_uses_fileglancer_authentication(tmp_path: Path) -> None:
    completed = _run_workspace_launcher(
        tmp_path,
        extra_env={"FG_SERVICE_PORT": "31878", "FG_SERVICE_TOKEN": "service-secret"},
    )
    arguments = completed.stdout.splitlines()

    assert ["--host", "0.0.0.0"] == arguments[
        arguments.index("--host") : arguments.index("--host") + 2
    ]
    assert ["--port", "31878"] == arguments[
        arguments.index("--port") : arguments.index("--port") + 2
    ]
    assert ["--token-password", "service-secret"] == arguments[
        arguments.index("--token-password") : arguments.index("--token-password") + 2
    ]
    assert "--no-token" not in arguments


def test_recording_workspace_rejects_collection_paths(tmp_path: Path) -> None:
    completed = _run_workspace_launcher(
        tmp_path,
        extra_args=("--registry", "/shared/palette.sqlite"),
        check=False,
    )

    assert completed.returncode == 2
    assert "unavailable in the single-recording read-only workspace" in completed.stderr
