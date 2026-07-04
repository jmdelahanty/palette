"""Compatibility re-export for system metadata helpers.

The implementation lives in :mod:`fisheye.shared.system_metadata` so shared
modules can record provenance without importing upward from ``utils``.
"""

import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fisheye.shared import system_metadata as _system_metadata
from fisheye.shared.system_metadata import *  # noqa: F401,F403

_which = _system_metadata._which
_run = _system_metadata._run
_find_git_root = _system_metadata._find_git_root
_to_jsonable = _system_metadata._to_jsonable
_serialize_args = _system_metadata._serialize_args


def build_invocation_record(
    *,
    tool: str,
    args: Optional[Any] = None,
    argv: Optional[List[str]] = None,
    include_git: bool = True,
    include_environment: bool = True,
) -> Dict[str, Any]:
    """Compatibility wrapper preserving the historic ``utils.system`` patch seam."""
    argv_tokens = [str(token) for token in (list(argv) if argv is not None else list(sys.argv[1:]))]
    entrypoint = str(sys.argv[0]) if sys.argv else ""
    command_tokens = [entrypoint, *argv_tokens] if entrypoint else argv_tokens
    platform_info = get_platform_info(collect_ip=False)

    record: Dict[str, Any] = {
        "tool": tool,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "cwd": str(Path.cwd()),
        "entrypoint": entrypoint,
        "argv": argv_tokens,
        "command": " ".join(shlex.quote(token) for token in command_tokens),
        "args": _serialize_args(args),
        "platform": {
            "hostname": platform_info.get("hostname"),
            "username": platform_info.get("username"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
        },
    }
    if include_git:
        record["git"] = _to_jsonable(get_git_info())
    if include_environment:
        record["environment"] = _to_jsonable(get_environment_summary())
    return record


def get_environment_info(
    include_all_packages: bool = False,
    disk_path: Optional[str] = None,
    collect_ip: bool = False,
    capture_env_vars: bool = True,
) -> Dict[str, Any]:
    """Compatibility wrapper preserving the historic ``utils.system`` patch seam."""
    env_info = {
        "git": get_git_info(),
        "platform": get_platform_info(collect_ip=collect_ip, disk_path=disk_path),
        "gpu": get_gpu_info(),
        "environment": get_environment_summary(),
    }

    if capture_env_vars:
        env_info["env_vars"] = get_critical_env_vars()

    if include_all_packages:
        env_info["all_packages"] = get_software_versions()

    return env_info
