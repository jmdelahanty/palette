from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.run_provenance import fisheye_version
from fisheye.shared.run_provenance import git_identity
from fisheye.shared.run_provenance import json_ready
from fisheye.shared.run_provenance import sha256_payload
from fisheye.shared.run_provenance import stable_json


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2
EXIT_USAGE = 3

SCHEMA = "palette.cli.workflow_oracle.v1"


def build_envelope(
    *,
    command: str,
    status: str,
    reason_code: str,
    recording: str | None = None,
    dataset_id: str | None = None,
    zarr_path: str | Path | None = None,
    run: str | None = None,
    artifacts: Sequence[Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    next_hints: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "command": str(command),
        "status": str(status),
        "reason_code": str(reason_code),
        "recording": recording,
        "dataset_id": dataset_id,
        "zarr_path": str(zarr_path) if zarr_path is not None else None,
        "run": run,
        "artifacts": json_ready(list(artifacts or ([] if zarr_path is None else [zarr_path]))),
        "metrics": json_ready(dict(metrics or {})),
        "next_hints": [str(item) for item in (next_hints or [])],
        "provenance": json_ready(dict(provenance or {})),
    }
    for key, value in extra.items():
        payload[str(key)] = json_ready(value)
    return payload


def exit_code_for_status(status: str) -> int:
    if status in {"ok", "dry_run"}:
        return EXIT_OK
    if status == "blocked":
        return EXIT_BLOCKED
    return EXIT_FAILED


def print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True))
