"""The runner config: one JSON file, validated before any step runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

CONFIG_SCHEMA = "palette.flow.intake_config.v1"
V1_STATE_DIR_NAMES = frozenset({".processing_state", ".processing_logs"})
_HOST = re.compile(r"[A-Za-z0-9_.@-]+")
_WALLTIME = re.compile(r"\d+:\d{2}")


class FlowConfigError(ValueError):
    """The runner config is unusable; nothing was run."""


@dataclass(frozen=True)
class LsfSettings:
    submit_host: str
    queue: str | None
    ncores: int
    mem_gb: int
    walltime: str
    max_wait_s: int
    poll_s: int
    heartbeat_stale_s: int
    bjobs_min_interval_s: int


@dataclass(frozen=True)
class IntakeFlowConfig:
    flow_root: Path
    staging_dir: Path
    destination_root: Path
    registry: Path
    registrar_config: Path
    ops_deployment: Path
    deployments_root: Path
    lsf_repo: Path
    lsf: LsfSettings

    @property
    def intake_root(self) -> Path:
        return self.flow_root / "intake"

    @property
    def lsf_state_dir(self) -> Path:
        return self.flow_root / "lsf"

    def delivery_dir(self, snapshot_sha: str) -> Path:
        return self.intake_root / snapshot_sha


def _path(raw: Mapping[str, Any], key: str) -> Path:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise FlowConfigError(f"config needs {key!r} as an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FlowConfigError(f"{key!r} must be absolute: {value}")
    return path


def _int(raw: Mapping[str, Any], key: str, default: int, minimum: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FlowConfigError(f"{key!r} must be an integer >= {minimum}")
    return value


def _reject_v1_state(name: str, path: Path) -> None:
    if V1_STATE_DIR_NAMES.intersection(path.parts):
        raise FlowConfigError(f"{name} must never be inside v1 intake state: {path}")


def parse_config(raw: Mapping[str, Any]) -> IntakeFlowConfig:
    if raw.get("schema") != CONFIG_SCHEMA:
        raise FlowConfigError(f"config schema must be {CONFIG_SCHEMA!r}")
    lsf_raw = raw.get("lsf")
    if not isinstance(lsf_raw, Mapping):
        raise FlowConfigError("config needs an 'lsf' object")
    host = lsf_raw.get("submit_host")
    if not isinstance(host, str) or not _HOST.fullmatch(host):
        raise FlowConfigError("lsf.submit_host must be a plain host alias")
    queue = lsf_raw.get("queue")
    if queue is not None and (not isinstance(queue, str) or not _HOST.fullmatch(queue)):
        raise FlowConfigError("lsf.queue must be a plain queue name or null")
    walltime = lsf_raw.get("walltime", "4:00")
    if not isinstance(walltime, str) or not _WALLTIME.fullmatch(walltime):
        raise FlowConfigError("lsf.walltime must look like H:MM")
    lsf = LsfSettings(
        submit_host=host,
        queue=queue,
        ncores=_int(lsf_raw, "ncores", 1, 1),
        mem_gb=_int(lsf_raw, "mem_gb", 16, 1),
        walltime=walltime,
        max_wait_s=_int(lsf_raw, "max_wait_s", 6 * 3600, 60),
        poll_s=_int(lsf_raw, "poll_s", 60, 5),
        heartbeat_stale_s=_int(lsf_raw, "heartbeat_stale_s", 300, 60),
        # The login-node budget (Jeremy, 2026-10-07): never below 5 minutes.
        bjobs_min_interval_s=_int(lsf_raw, "bjobs_min_interval_s", 300, 300),
    )
    config = IntakeFlowConfig(
        flow_root=_path(raw, "flow_root"),
        staging_dir=_path(raw, "staging_dir"),
        destination_root=_path(raw, "destination_root"),
        registry=_path(raw, "registry"),
        registrar_config=_path(raw, "registrar_config"),
        ops_deployment=_path(raw, "ops_deployment"),
        deployments_root=_path(raw, "deployments_root"),
        lsf_repo=_path(raw, "lsf_repo"),
        lsf=lsf,
    )
    _reject_v1_state("flow_root", config.flow_root)
    if config.flow_root == config.staging_dir or config.staging_dir in config.flow_root.parents:
        # Discovery walks staging; runner state there would be scanned as deliveries.
        raise FlowConfigError("flow_root must not be inside staging_dir")
    return config


def load_config(path: str | Path) -> IntakeFlowConfig:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise FlowConfigError(f"unreadable runner config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FlowConfigError("runner config must be a JSON object")
    return parse_config(raw)


__all__ = [
    "CONFIG_SCHEMA",
    "FlowConfigError",
    "IntakeFlowConfig",
    "LsfSettings",
    "load_config",
    "parse_config",
]
