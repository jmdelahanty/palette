from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_import_palette_does_not_import_textual() -> None:
    result = _run_python(
        """
        import json
        import sys

        import fisheye.cli.palette  # noqa: F401

        textual_modules = sorted(
            name for name in sys.modules if name == "textual" or name.startswith("textual.")
        )
        print(json.dumps(textual_modules))
        raise SystemExit(1 if textual_modules else 0)
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip()) == []


def test_lazy_cli_exports_resolve_tui_functions() -> None:
    pytest.importorskip("textual")

    result = _run_python(
        """
        import json

        from fisheye.cli import (
            pick_config_path,
            pick_video_path,
            pick_zarr_path,
            run_interactive_launcher,
        )

        print(json.dumps({
            "run_interactive_launcher": callable(run_interactive_launcher),
            "pick_zarr_path": callable(pick_zarr_path),
            "pick_video_path": callable(pick_video_path),
            "pick_config_path": callable(pick_config_path),
        }, sort_keys=True))
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip()) == {
        "pick_config_path": True,
        "pick_video_path": True,
        "pick_zarr_path": True,
        "run_interactive_launcher": True,
    }
