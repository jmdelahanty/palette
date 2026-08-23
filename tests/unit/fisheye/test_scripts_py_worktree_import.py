from __future__ import annotations

import os
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


def test_scripts_py_uses_account_home_when_scheduler_replaces_home(
    tmp_path: Path,
) -> None:
    account_home = tmp_path / "account_home"
    python_bin = (
        account_home / "miniforge3" / "envs" / "palette-py311" / "bin" / "python"
    )
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        "#!/usr/bin/env bash\nprintf 'account-home:%s\\n' \"$1\"\n",
        encoding="utf-8",
    )
    python_bin.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    getent = fake_bin / "getent"
    getent.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'palette:*:1:1:Palette:{account_home}:/bin/bash\\n'\n",
        encoding="utf-8",
    )
    getent.chmod(0o755)
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "lsf_temporary_home"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    environment.pop("PALETTE_PYTHON", None)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "py"), "sentinel"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == "account-home:sentinel"
