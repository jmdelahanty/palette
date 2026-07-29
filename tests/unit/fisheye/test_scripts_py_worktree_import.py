from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_scripts_py_imports_fisheye_from_its_own_worktree() -> None:
    result = subprocess.run(
        [
            str(REPO_ROOT / "scripts" / "py"),
            "-c",
            (
                "import pathlib, fisheye; "
                "print(pathlib.Path(fisheye.__file__).resolve())"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert Path(result.stdout.strip()) == (
        REPO_ROOT / "src" / "fisheye" / "__init__.py"
    ).resolve()
