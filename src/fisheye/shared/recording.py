"""Read-only Recording accessor for Palette Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import sqlite3

import numpy as np

from fisheye.registry.db import RegistryPaths
from fisheye.shared.frame_domains import FrameDomains
from fisheye.shared.mask_store import open_mask_store
from fisheye.shared.recording_artifact_inventory import build_recording_artifact_inventory
from fisheye.shared.run_resolution import RunResolution, RunResolutionResult, resolve_run
from fisheye.shared.stage_run_groups import stage_run_parent_paths
from fisheye.shared.zarr_helpers import (
    open_zarr_group_direct,
    read_zarr_array_mapping,
    zarr_attrs_dict,
    zarr_child_group,
)


@dataclass(frozen=True)
class RecordingRef:
    """Resolved recording identity for a read-only accessor."""

    zarr_path: Path
    dataset_id: str | None = None
    recording_id: str | None = None
    zarr_use: str | None = None
    status: str | None = None
    registry_path: Path | None = None


@dataclass(frozen=True)
class RecordingRead:
    """Data returned by a read-only accessor method."""

    kind: str
    run_name: str
    parent_path: str
    resolution: RunResolutionResult
    attrs: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    masks: np.ndarray | None = None
    mask_labels: tuple[str, ...] = ()
    mask_encoding: str | None = None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_zarr_path(path: Path) -> bool:
    return path.exists() and (
        path.suffix == ".zarr"
        or (path / "zarr.json").is_file()
        or (path / ".zgroup").is_file()
    )


def _resolve_explicit_zarr_path(raw: str | Path) -> tuple[Path | None, list[str]]:
    path = Path(raw).expanduser()
    if not path.exists():
        return None, []
    if _is_zarr_path(path):
        return path.resolve(), []
    if not path.is_dir():
        return None, [f"path is not a zarr directory: {path}"]
    candidates = sorted(candidate for candidate in path.iterdir() if _is_zarr_path(candidate))
    if not candidates:
        return None, [f"directory contains no zarr stores: {path}"]
    training = [candidate for candidate in candidates if candidate.name.endswith("_training.zarr")]
    if len(training) == 1:
        return training[0].resolve(), []
    if len(candidates) == 1:
        return candidates[0].resolve(), []
    names = ", ".join(candidate.name for candidate in candidates[:8])
    return None, [f"directory contains multiple zarr stores; pass one explicitly: {names}"]


def _open_readonly_registry(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _resolve_registry_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    return RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()


def _resolve_from_registry(recording: str, registry_path: Path) -> RecordingRef | None:
    with _open_readonly_registry(registry_path) as conn:
        rows = conn.execute(
            """
            SELECT dataset_id, recording_id, zarr_path, zarr_use, status
            FROM datasets
            WHERE dataset_id = ?
               OR recording_id = ?
               OR zarr_path = ?
            ORDER BY
                CASE WHEN status = 'active' THEN 0 ELSE 1 END,
                CASE WHEN zarr_use = 'training' THEN 0 ELSE 1 END,
                dataset_id
            LIMIT 2;
            """,
            (recording, recording, recording),
        ).fetchall()
    if not rows:
        return None
    if len(rows) > 1:
        active_training = [
            row
            for row in rows
            if str(row["status"] or "") == "active" and str(row["zarr_use"] or "") == "training"
        ]
        row = active_training[0] if len(active_training) == 1 else rows[0]
    else:
        row = rows[0]
    return RecordingRef(
        dataset_id=_normalize_text(row["dataset_id"]),
        recording_id=_normalize_text(row["recording_id"]),
        zarr_path=Path(str(row["zarr_path"])).expanduser().resolve(),
        zarr_use=_normalize_text(row["zarr_use"]),
        status=_normalize_text(row["status"]),
        registry_path=registry_path,
    )


def _resolve_recording_ref(path_or_id: str | Path, registry: Path | str | None) -> RecordingRef:
    raw = str(path_or_id)
    explicit, explicit_notes = _resolve_explicit_zarr_path(raw)
    registry_path = _resolve_registry_path(registry)
    if explicit is not None:
        if registry_path.exists():
            resolved = _resolve_from_registry(str(explicit), registry_path)
            if resolved is not None:
                return resolved
        return RecordingRef(zarr_path=explicit)
    if Path(raw).expanduser().exists():
        notes = "; ".join(explicit_notes) if explicit_notes else "path exists but is not a zarr store"
        raise ValueError(notes)
    if not registry_path.exists():
        raise ValueError(
            f"registry not found: {registry_path}; pass an explicit zarr path or registry=..."
        )
    resolved = _resolve_from_registry(raw, registry_path)
    if resolved is None:
        raise ValueError(f"recording not found in registry: {raw}")
    return resolved


def _array_prefix(parent_path: str, run_name: str) -> str:
    return f"{parent_path.strip('/')}/{run_name.strip('/')}"


class Recording:
    """Read-only handle for a Palette recording/training Zarr store."""

    def __init__(self, ref: RecordingRef) -> None:
        self.ref = ref
        self.path = ref.zarr_path
        self.root = open_zarr_group_direct(ref.zarr_path, mode="r")

    def _candidate_parent_paths(self, stage: str) -> tuple[str, ...]:
        paths = stage_run_parent_paths(stage)
        if not paths:
            raise ValueError(f"stage {stage!r} has no known run-parent mapping")
        return paths

    def _find_parent(self, stage: str, *, run_name: str | None = None) -> tuple[str, Any]:
        first_existing: tuple[str, Any] | None = None
        for parent_path in self._candidate_parent_paths(stage):
            parent = zarr_child_group(self.root, parent_path)
            if parent is None:
                continue
            if first_existing is None:
                first_existing = (parent_path, parent)
            if run_name is None:
                return parent_path, parent
            if zarr_child_group(parent, run_name) is not None:
                return parent_path, parent
        if first_existing is not None:
            return first_existing
        candidates = ", ".join(self._candidate_parent_paths(stage))
        raise ValueError(f"no run parent found for stage {stage!r}; tried: {candidates}")

    def _resolve_stage_run(
        self,
        stage: str,
        *,
        run: str | None,
        resolution: RunResolution | str,
    ) -> tuple[str, Any, RunResolutionResult]:
        mode = RunResolution(resolution)
        if mode is RunResolution.INVENTORY_LATEST and run is None:
            inventory = resolve_run(
                None,
                mode,
                registry=self.ref.registry_path,
                dataset_id=self.ref.dataset_id,
                recording_id=self.ref.recording_id,
                stage=stage,
            )
            if inventory.run_name is None:
                raise ValueError(inventory.reason or f"registry has no latest run for stage {stage!r}")
            parent_path, parent = self._find_parent(stage, run_name=inventory.run_name)
            child = zarr_child_group(parent, inventory.run_name)
            if child is None:
                raise ValueError(
                    f"registry latest run {inventory.run_name!r} for stage {stage!r} "
                    f"was not found under {parent_path}"
                )
            return parent_path, child, replace(inventory, parent_path=parent_path, run_group=child)

        parent_path, parent = self._find_parent(stage, run_name=run)
        resolved = resolve_run(
            parent,
            mode,
            parent_path=parent_path,
            explicit_run=run,
            legacy_default=None,
            run_label=f"{stage} run",
        )
        if resolved.run_name is None or resolved.run_group is None:
            raise ValueError(f"could not resolve {stage!r} run using {mode.value}")
        return parent_path, resolved.run_group, resolved

    def _read_arrays(
        self,
        *,
        stage: str,
        kind: str,
        run: str | None,
        resolution: RunResolution | str,
        array_names: Sequence[str] | None = None,
    ) -> RecordingRead:
        parent_path, run_group, resolved = self._resolve_stage_run(
            stage,
            run=run,
            resolution=resolution,
        )
        run_name = resolved.run_name or ""
        return RecordingRead(
            kind=kind,
            run_name=run_name,
            parent_path=parent_path,
            resolution=resolved,
            attrs=zarr_attrs_dict(run_group),
            arrays=read_zarr_array_mapping(
                run_group,
                physical_prefix=_array_prefix(parent_path, run_name),
                array_names=array_names,
            ),
        )

    def detections(
        self,
        *,
        run: str | None = None,
        resolution: RunResolution | str = RunResolution.AUTHORITATIVE,
        stage: str = "detect",
        array_names: Sequence[str] | None = None,
    ) -> RecordingRead:
        """Read a detection run, resolving authoritative-first by default."""

        return self._read_arrays(
            stage=stage,
            kind="detections",
            run=run,
            resolution=resolution,
            array_names=array_names,
        )

    def keypoints(
        self,
        *,
        run: str | None = None,
        resolution: RunResolution | str = RunResolution.AUTHORITATIVE,
        stage: str = "keypoints",
        array_names: Sequence[str] | None = None,
    ) -> RecordingRead:
        """Read a keypoint run, resolving authoritative-first by default."""

        return self._read_arrays(
            stage=stage,
            kind="keypoints",
            run=run,
            resolution=resolution,
            array_names=array_names,
        )

    def subject_masks(
        self,
        *,
        run: str | None = None,
        resolution: RunResolution | str = RunResolution.AUTHORITATIVE,
        stage: str = "refined_subject_masks",
        rows: Sequence[int] | np.ndarray | slice | int | None = None,
        channels: Sequence[int | str] | int | str | slice | None = None,
        prefer: str = "auto",
    ) -> RecordingRead:
        """Read subject masks as dense arrays without materializing caches."""

        parent_path, run_group, resolved = self._resolve_stage_run(
            stage,
            run=run,
            resolution=resolution,
        )
        run_name = resolved.run_name or ""
        mask_store = open_mask_store(
            run_group,
            source_path=_array_prefix(parent_path, run_name),
            prefer=prefer,
        )
        _channel_indices, labels = mask_store.resolve_channels(channels)
        masks = mask_store.read_dense(rows=rows, channels=channels)
        return RecordingRead(
            kind="subject_masks",
            run_name=run_name,
            parent_path=parent_path,
            resolution=resolved,
            attrs=zarr_attrs_dict(run_group),
            arrays={},
            masks=masks,
            mask_labels=labels,
            mask_encoding=mask_store.encoding,
        )

    def frame_domains(
        self,
        *,
        stage: str | None = None,
        run: str | None = None,
        resolution: RunResolution | str = RunResolution.AUTHORITATIVE,
    ) -> FrameDomains:
        """Return the explicit frame-domain resolver for this recording or run."""

        if stage is None:
            return FrameDomains(root=self.root)
        parent_path, run_group, resolved = self._resolve_stage_run(
            stage,
            run=run,
            resolution=resolution,
        )
        return FrameDomains(
            root=self.root,
            stage=stage,
            run_group=run_group,
            parent_path=parent_path,
            run_name=resolved.run_name,
            resolution=resolved,
        )

    def artifact_inventory(self) -> dict[str, Any]:
        """Build the read-only artifact inventory spanning all run families."""

        return build_recording_artifact_inventory(self.root, zarr_path=self.path)


def open_recording(path_or_id: str | Path, *, registry: Path | str | None = None) -> Recording:
    """Open a Palette recording/training Zarr read-only."""

    return Recording(_resolve_recording_ref(path_or_id, registry))
