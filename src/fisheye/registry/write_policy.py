"""Process-level registry write policy.

``PALETTE_DISABLE_REGISTRY_WRITES`` is the established switch that cluster
launchers export so a job never mutates the canonical registry SQLite (the
registry is written only on the writer host, through the shadow gateway).
Several stages read it through private copies of this check
(``detection.detect_keypoints_yolo``, ``refinement.refine_keypoints``,
``tracking.crop``, ``analysis_workflows.registry_finalize``); this module is
the public owner new callers should use. Migrating those copies is separate,
behavior-preserving work.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

REGISTRY_WRITES_DISABLED_ENV = "PALETTE_DISABLE_REGISTRY_WRITES"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def registry_writes_disabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return True when this process has been told not to write the registry."""
    env = os.environ if environ is None else environ
    value = env.get(REGISTRY_WRITES_DISABLED_ENV)
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES


def running_inside_lsf_job(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return True when the process runs inside an LSF job allocation."""
    env = os.environ if environ is None else environ
    return bool(str(env.get("LSB_JOBID") or "").strip())


__all__ = [
    "REGISTRY_WRITES_DISABLED_ENV",
    "registry_writes_disabled",
    "running_inside_lsf_job",
]
