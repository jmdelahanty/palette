"""Explicit run-resolution modes for Palette Zarr run parents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence
import sqlite3

from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.shared.zarr_run_completion import (
    AUTHORITATIVE_RUN_ATTR,
    is_run_complete_in_parent,
    resolve_authoritative_run_name,
    resolve_latest_complete_run_name,
)


class RunResolution(str, Enum):
    """Named meanings of "current run"."""

    AUTHORITATIVE = "authoritative"
    LATEST_COMPLETE = "latest_complete"
    INVENTORY_LATEST = "inventory_latest"
    SOURCE_MATCH = "source_match"


@dataclass(frozen=True)
class RunResolutionResult:
    """Resolved run plus the provenance of that resolution decision."""

    run_name: str | None
    mode: RunResolution
    resolution_source: str
    parent_path: str | None = None
    run_group: Any | None = None
    fallback_used: bool = False
    registry_path: Path | None = None
    dataset_id: str | None = None
    recording_id: str | None = None
    stage: str | None = None
    source_attr: str | None = None
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.run_name is not None


def _normalize_text(value: Any) -> str | None:
    normalized = normalize_attr(value)
    if normalized is None:
        return None
    return normalized or None


def _get_child_group(parent_group: Any, run_name: str | None) -> Any | None:
    name = _normalize_text(run_name)
    if name is None:
        return None
    current = parent_group
    for part in [piece for piece in name.split("/") if piece]:
        try:
            if part not in current:
                return None
            current = current[part]
        except Exception:
            return None
    return current if hasattr(current, "attrs") else None


def _completed_child(parent_group: Any, run_name: str | None, *, legacy_default: bool | None) -> Any | None:
    child = _get_child_group(parent_group, run_name)
    if child is None:
        return None
    if not is_run_complete_in_parent(parent_group, child, legacy_default=legacy_default):
        return None
    return child


def _open_registry_connection(registry: Path | str | sqlite3.Connection) -> tuple[sqlite3.Connection, bool, Path | None]:
    if isinstance(registry, sqlite3.Connection):
        registry.row_factory = sqlite3.Row
        return registry, False, None
    path = Path(registry).expanduser().resolve()
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn, True, path


def _resolve_inventory_latest(
    *,
    registry: Path | str | sqlite3.Connection | None,
    dataset_id: str | None,
    recording_id: str | None,
    stage: str | None,
) -> RunResolutionResult:
    if registry is None:
        raise ValueError("INVENTORY_LATEST requires a registry path or sqlite connection.")
    stage_name = _normalize_text(stage)
    if stage_name is None:
        raise ValueError("INVENTORY_LATEST requires a stage/step name.")
    dataset = _normalize_text(dataset_id)
    recording = _normalize_text(recording_id)
    if dataset is None and recording is None:
        raise ValueError("INVENTORY_LATEST requires dataset_id or recording_id.")

    conn, should_close, registry_path = _open_registry_connection(registry)
    try:
        rows = conn.execute(
            """
            SELECT dataset_id, recording_id, step_name, run_name, updated_utc
            FROM recording_step_status_latest
            WHERE step_name = ?
              AND (
                    (? IS NOT NULL AND dataset_id = ?)
                 OR (? IS NOT NULL AND recording_id = ?)
              )
            ORDER BY updated_utc DESC
            LIMIT 1;
            """,
            (stage_name, dataset, dataset, recording, recording),
        ).fetchall()
    finally:
        if should_close:
            conn.close()

    if not rows:
        return RunResolutionResult(
            run_name=None,
            mode=RunResolution.INVENTORY_LATEST,
            resolution_source="inventory_latest_missing",
            registry_path=registry_path,
            dataset_id=dataset,
            recording_id=recording,
            stage=stage_name,
            reason="registry has no latest row for stage",
        )
    row = rows[0]
    return RunResolutionResult(
        run_name=_normalize_text(row["run_name"]),
        mode=RunResolution.INVENTORY_LATEST,
        resolution_source="inventory_latest",
        registry_path=registry_path,
        dataset_id=_normalize_text(row["dataset_id"]),
        recording_id=_normalize_text(row["recording_id"]),
        stage=_normalize_text(row["step_name"]),
    )


def _source_run_name(
    source_group: Any | None,
    *,
    source_attr: str | None,
    source_attrs: Sequence[str],
) -> tuple[str | None, str | None]:
    if source_group is None:
        raise ValueError("SOURCE_MATCH requires source_group with lineage attrs.")
    attrs = getattr(source_group, "attrs", {})
    candidates = (source_attr,) if source_attr else tuple(source_attrs)
    for attr in candidates:
        if not attr:
            continue
        value = _normalize_text(attrs.get(str(attr)))
        if value is not None:
            return value, str(attr)
    names = ", ".join(str(value) for value in candidates if value)
    raise ValueError(f"SOURCE_MATCH could not find any source run attr: {names}")


def resolve_run(
    parent_group: Any | None,
    mode: RunResolution | str = RunResolution.AUTHORITATIVE,
    *,
    parent_path: str | None = None,
    explicit_run: str | None = None,
    root: Any | None = None,
    latest_attr: str = "latest",
    legacy_default: bool | None = None,
    registry: Path | str | sqlite3.Connection | None = None,
    dataset_id: str | None = None,
    recording_id: str | None = None,
    stage: str | None = None,
    source_group: Any | None = None,
    source_attr: str | None = None,
    source_attrs: Sequence[str] = ("source_run", "source_detect_run", "source_crop_run"),
    run_label: str = "Run",
) -> RunResolutionResult:
    """Resolve a run name according to an explicit resolution mode.

    ``explicit_run`` bypasses mode dispatch and validates that the named child
    exists under ``parent_group``. It is intended for accessor calls where the
    user has already chosen a run.
    """

    resolution = RunResolution(mode)
    if explicit_run is not None:
        if parent_group is None:
            raise ValueError("explicit run resolution requires parent_group.")
        name = _normalize_text(explicit_run)
        if name is None:
            raise ValueError("explicit run name must be non-empty.")
        child = _get_child_group(parent_group, name)
        if child is None:
            raise ValueError(f"{run_label} {name!r} not found under {parent_path or '(parent)'}.")
        return RunResolutionResult(
            run_name=name,
            mode=resolution,
            resolution_source="explicit",
            parent_path=parent_path,
            run_group=child,
        )

    if resolution is RunResolution.INVENTORY_LATEST:
        return _resolve_inventory_latest(
            registry=registry,
            dataset_id=dataset_id,
            recording_id=recording_id,
            stage=stage,
        )

    if resolution is RunResolution.SOURCE_MATCH:
        name, matched_attr = _source_run_name(
            source_group,
            source_attr=source_attr,
            source_attrs=source_attrs,
        )
        if root is not None and parent_path is not None:
            run_group, resolved_name = resolve_zarr_run(
                root,
                parent_path,
                name,
                fallback_to_latest=False,
                run_label=run_label,
            )
            return RunResolutionResult(
                run_name=resolved_name,
                mode=resolution,
                resolution_source="source_match",
                parent_path=parent_path,
                run_group=run_group,
                source_attr=matched_attr,
            )
        if parent_group is None:
            raise ValueError("SOURCE_MATCH requires parent_group or root plus parent_path.")
        child = _get_child_group(parent_group, name)
        if child is None:
            raise ValueError(f"{run_label} {name!r} from {matched_attr} not found under {parent_path or '(parent)'}.")
        return RunResolutionResult(
            run_name=name,
            mode=resolution,
            resolution_source="source_match",
            parent_path=parent_path,
            run_group=child,
            source_attr=matched_attr,
        )

    if parent_group is None:
        raise ValueError(f"{resolution.value} resolution requires parent_group.")

    if resolution is RunResolution.AUTHORITATIVE:
        name = resolve_authoritative_run_name(
            parent_group,
            latest_attr=latest_attr,
            legacy_default=legacy_default,
        )
        authoritative = _normalize_text(parent_group.attrs.get(AUTHORITATIVE_RUN_ATTR))
        source = "authoritative" if authoritative is not None and name == authoritative else "latest_complete"
        return RunResolutionResult(
            run_name=name,
            mode=resolution,
            resolution_source=source if name is not None else "authoritative_missing",
            parent_path=parent_path,
            run_group=_completed_child(parent_group, name, legacy_default=legacy_default),
            fallback_used=authoritative is None and name is not None,
        )

    if resolution is RunResolution.LATEST_COMPLETE:
        name = resolve_latest_complete_run_name(
            parent_group,
            latest_attr=latest_attr,
            legacy_default=legacy_default,
        )
        return RunResolutionResult(
            run_name=name,
            mode=resolution,
            resolution_source="latest_complete" if name is not None else "latest_complete_missing",
            parent_path=parent_path,
            run_group=_completed_child(parent_group, name, legacy_default=legacy_default),
        )

    raise ValueError(f"Unsupported run resolution mode: {resolution.value}")
