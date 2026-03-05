from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional


def resolve_recording_roots(paths: Optional[Iterable[Path]]) -> List[Path]:
    if paths:
        return list(paths)
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def resolve_log_dir(
    arg_log_dir: Optional[Path],
    roots: Iterable[Path],
    *,
    log_subdir: str,
) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / log_subdir
    for root in roots:
        expanded = root.expanduser()
        if expanded.exists() and expanded.is_dir():
            return expanded / "logs" / log_subdir
    return Path.cwd() / "logs" / log_subdir
