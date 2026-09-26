"""Version lineage of mask-Apply tail successors and the work each version holds.

A mask Apply publishes a new review version whose keypoint and mask runs are
copies of the source runs *at the moment the version was created*
(``palette_run_started_at_utc``). The source runs remain in the archive, so a
later edit applied to a source run is not part of any newer version. This
module derives that lineage from the archive and finds such stranded edits.

Lineage is read only from each run's sealed ``source_bindings.mask_apply_refresh``
proof (``source_pose_run`` / ``source_mask_run``); nothing here writes Zarr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

KEYPOINT_FAMILY = "refined_keypoints_runs"
MASK_FAMILY = "refined_subject_masks_runs"
_PROOF_SOURCE_KEY = {
    KEYPOINT_FAMILY: "source_pose_run",
    MASK_FAMILY: "source_mask_run",
}
_WORKFLOW_FAMILY = {"keypoints": KEYPOINT_FAMILY, "subject_mask_component": MASK_FAMILY}
_MANUAL_ORIGIN = 3


@dataclass(frozen=True)
class FamilyLineage:
    """Parent links between runs of one family in one archive."""

    family: str
    parent: Mapping[str, str | None]
    started_at: Mapping[str, str]

    def children(self, run: str) -> list[str]:
        return sorted(name for name, parent in self.parent.items() if parent == run)

    def component(self, run: str) -> frozenset[str]:
        """Every run connected to ``run`` through successor links."""

        if run not in self.parent:
            return frozenset({run})
        seen = {run}
        frontier = [run]
        while frontier:
            current = frontier.pop()
            linked = [*self.children(current)]
            parent = self.parent.get(current)
            if parent is not None:
                linked.append(parent)
            for name in linked:
                if name not in seen:
                    seen.add(name)
                    frontier.append(name)
        return frozenset(seen)

    def path_to_root(self, run: str) -> list[str]:
        """``run`` followed by its ancestors, nearest first."""

        path = [run]
        while True:
            parent = self.parent.get(path[-1])
            if parent is None or parent in path:
                return path
            path.append(parent)

    def newest_leaf(self, runs: Iterable[str]) -> str:
        """The most recently created run in ``runs`` that has no successor."""

        leaves = [name for name in runs if not self.children(name)]
        if not leaves:
            raise ValueError("Lineage has no leaf version")
        return max(leaves, key=lambda name: (self.started_at.get(name, ""), name))

    def snapshot_cutoffs(self, run: str) -> dict[str, str | None]:
        """For ``run`` and each ancestor, the time after which its edits are not in ``run``.

        ``None`` means every edit made in that run is present in ``run`` (the
        run itself). An ancestor's edits are present only if applied before its
        child on the path was created.
        """

        path = self.path_to_root(run)
        cutoffs: dict[str, str | None] = {run: None}
        for child, ancestor in zip(path, path[1:]):
            cutoffs[ancestor] = self.started_at.get(child)
        return cutoffs


def _run_name(path: object) -> str | None:
    text = str(path or "").strip().strip("/")
    return text.split("/")[-1] if text else None


def read_family_lineage(root: object, family: str) -> FamilyLineage:
    parent: dict[str, str | None] = {}
    started: dict[str, str] = {}
    group = root.get(family) if hasattr(root, "get") else None
    if group is None:
        return FamilyLineage(family, parent, started)
    source_key = _PROOF_SOURCE_KEY[family]
    for name, run in group.groups():
        attrs = run.attrs
        bindings = attrs.get("source_bindings") or {}
        proof = bindings.get("mask_apply_refresh") if isinstance(bindings, Mapping) else None
        parent[str(name)] = _run_name(proof.get(source_key)) if isinstance(proof, Mapping) else None
        started[str(name)] = str(attrs.get("palette_run_started_at_utc") or "")
    return FamilyLineage(family, parent, started)


def newer_version_of(archive: str | Path, keypoint_run: str) -> str | None:
    """The newest successor of ``keypoint_run`` in ``archive``, or None if it is current."""

    import zarr

    if not keypoint_run or not Path(archive).exists():
        return None
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    lineage = read_family_lineage(root, KEYPOINT_FAMILY)
    if not keypoint_run or not lineage.children(keypoint_run):
        return None
    descendants = [
        name
        for name in lineage.component(keypoint_run)
        if keypoint_run in lineage.path_to_root(name) and name != keypoint_run
    ]
    return lineage.newest_leaf(descendants) if descendants else None


def task_family(task: Mapping[str, object]) -> str | None:
    return _WORKFLOW_FAMILY.get(str(task.get("workflow_kind") or ""))


def task_run(task: Mapping[str, object]) -> str | None:
    scope = task.get("scope") or {}
    run = scope.get("refined_run") if isinstance(scope, Mapping) else None
    return str(run or task.get("run_name") or "").strip() or None


def task_archive(task: Mapping[str, object]) -> Path | None:
    scope = task.get("scope") or {}
    path = scope.get("zarr_path") if isinstance(scope, Mapping) else None
    return Path(str(path)).resolve() if path else None


def lineage_tasks(
    tasks: Sequence[Mapping[str, object]],
    *,
    archive: Path,
    family: str,
    component: frozenset[str],
) -> list[Mapping[str, object]]:
    """Tasks of ``family`` whose run in ``archive`` belongs to ``component``."""

    archive = Path(archive).resolve()
    return [
        task
        for task in tasks
        if task_family(task) == family
        and task_archive(task) == archive
        and task_run(task) in component
    ]


@dataclass
class StrandedRow:
    """One row whose latest applied manual edit is not in the target version."""

    roi_idx: int
    source_run: str
    source_task_id: str
    checkpoint_ids: list[str]
    applied_at_utc: str
    disposition: str  # carry | already_present | target_edit_newer | not_carryable
    manual_keypoints: list[int] = field(default_factory=list)
    points: list[list[float]] | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "roi_idx": self.roi_idx,
            "source_run": self.source_run,
            "source_task_id": self.source_task_id,
            "checkpoint_ids": list(self.checkpoint_ids),
            "applied_at_utc": self.applied_at_utc,
            "disposition": self.disposition,
            "manual_keypoints": list(self.manual_keypoints),
            "points": self.points,
            "detail": self.detail,
        }


def _run_path(run: str) -> str:
    return f"{KEYPOINT_FAMILY}/{run}"


def stranded_keypoint_rows(
    *,
    root: object,
    checkpoints: Sequence[Mapping[str, object]],
    lineage: FamilyLineage,
    target_run: str,
) -> list[StrandedRow]:
    """Rows of ``target_run`` missing a newer manual edit applied elsewhere in its lineage.

    ``checkpoints`` are the applied keypoint checkpoints of the recording
    (store rows with ``target_run_path``, ``roi_idx``, ``applied_at_utc``).
    The latest applied edit of each row wins. An edit counts as present in
    ``target_run`` if it was applied to ``target_run`` or to an ancestor before
    that ancestor was snapshotted. Only manually set landmarks are carried;
    automatic (mask-derived) points in the target come from its newer masks.
    """

    component = lineage.component(target_run)
    cutoffs = lineage.snapshot_cutoffs(target_run)
    present: dict[int, str] = {}
    candidates: dict[int, list[Mapping[str, object]]] = {}
    for checkpoint in checkpoints:
        if str(checkpoint.get("state") or "") != "applied":
            continue
        run = _run_name(checkpoint.get("target_run_path"))
        if run not in component:
            continue
        roi = int(checkpoint["roi_idx"])
        applied = str(checkpoint.get("applied_at_utc") or "")
        if run in cutoffs and (cutoffs[run] is None or applied < str(cutoffs[run])):
            if applied > present.get(roi, ""):
                present[roi] = applied
        else:
            candidates.setdefault(roi, []).append(checkpoint)
    if not candidates:
        return []
    target = root[_run_path(target_run)]
    target_points = np.asarray(target["keypoints_roi"][:], dtype=np.float64)
    target_manual = np.asarray(target["keypoint_manual_edit"][:], dtype=bool)
    height = width = None
    source_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[StrandedRow] = []
    for roi in sorted(candidates):
        latest_run_edits = sorted(
            candidates[roi], key=lambda item: str(item.get("applied_at_utc") or "")
        )
        latest = latest_run_edits[-1]
        source_run = str(_run_name(latest.get("target_run_path")))
        same_run = [
            item for item in latest_run_edits
            if _run_name(item.get("target_run_path")) == source_run
        ]
        applied = str(latest.get("applied_at_utc") or "")
        row = StrandedRow(
            roi_idx=roi,
            source_run=source_run,
            source_task_id=str(latest.get("task_id") or ""),
            checkpoint_ids=[str(item.get("checkpoint_id") or "") for item in same_run],
            applied_at_utc=applied,
            disposition="carry",
        )
        rows.append(row)
        if present.get(roi, "") > applied:
            row.disposition = "target_edit_newer"
            continue
        if source_run not in source_arrays:
            group = root[_run_path(source_run)]
            source_arrays[source_run] = (
                np.asarray(group["keypoints_roi"][:], dtype=np.float64),
                np.asarray(group["keypoint_manual_edit"][:], dtype=bool),
            )
        source_points, source_manual = source_arrays[source_run]
        if source_points.shape != target_points.shape:
            row.disposition = "not_carryable"
            row.detail = "row layout differs between versions"
            continue
        manual = source_manual[roi]
        row.manual_keypoints = [int(i) for i in np.flatnonzero(manual)]
        carried = target_points[roi].copy()
        carried[manual] = source_points[roi][manual]
        row.points = [[float(x), float(y)] for x, y in carried]
        if not manual.any():
            row.disposition = "already_present"
            row.detail = "latest edit set no manual landmarks"
            continue
        same = np.allclose(carried, target_points[roi], equal_nan=True)
        if same and target_manual[roi][manual].all():
            row.disposition = "already_present"
            continue
        if not np.isfinite(carried).all():
            row.disposition = "not_carryable"
            row.detail = "a landmark would be missing (manual clear or failed derivation)"
            row.points = None
            continue
    return rows
