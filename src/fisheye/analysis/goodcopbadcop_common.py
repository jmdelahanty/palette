#!/usr/bin/env python
"""Shared cohort resolution and zarr navigation for the GoodCopBadCop analyses.

These helpers back the durable `analyze_goodcopbadcop_*` and `plot_goodcopbadcop_*`
scripts. Centralising them fixes one real bug: the registry can hold *duplicate*
`datasets` rows for the same recording (a legacy un-suffixed dataset_id with no
provenance plus the content-hash-suffixed row). A naive
`SELECT ... FROM dataset_context_current` therefore returns some recordings twice,
which would silently double-count those fish in every cohort statistic.
`resolve_cohort()` dedupes by `recording_id` so each fish is counted once.

Registry path: `$PALETTE_REGISTRY_PATH` if set, else the canonical live registry on
`/groups`. The `/nvme1/palette_registry.sqlite` copy is STALE -- it is missing the
June 21 and July 2 sessions and still carries the retired May duplicate rows, so
querying it silently resolves only the 12 June-14 recordings.
Figures: `$PALETTE_RECORDINGS_ROOT/figures` (default `/nvme1/recordings/figures`) --
committed scripts, out-of-repo figures.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np

from fisheye.analysis.cra_primary_endpoint import resolve_object_roles_from_protocol_payload
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadSnapshot,
    load_chaser_distance_run,
)
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track

# Canonical epoch keys used across the analyses.
EPOCHS = ("pre", "chase", "post")

# The live registry on /groups (matches DEFAULT_REGISTRY in fisheye.cluster.*). The
# /nvme1 copy is stale (June-14 only, plus retired May duplicate rows).
CANONICAL_REGISTRY = "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"


def registry_db() -> str:
    """Path to the registry SQLite. `$PALETTE_REGISTRY_PATH` wins; else the canonical
    /groups registry (NOT the stale /nvme1 copy)."""
    return os.environ.get("PALETTE_REGISTRY_PATH", CANONICAL_REGISTRY)


def figures_dir() -> Path:
    d = Path(os.environ.get("PALETTE_RECORDINGS_ROOT", "/nvme1/recordings")) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_cohort(pattern: str = "%GoodCopBadCop%", *, require_on_disk: bool = True):
    """Return [(recording_id, zarr_path)] for active analysis zarrs, deduped by recording.

    - Dedupes by `recording_id` (see module docstring) -- one row per fish.
    - When `require_on_disk`, drops rows whose `zarr_path` does not resolve, so a stale
      registry pointer (e.g. an un-repointed /nvme1 path) is skipped rather than crashing.
    - Sorted by recording_id for stable, reproducible ordering.
    """
    c = sqlite3.connect(registry_db())
    c.row_factory = sqlite3.Row
    try:
        rows = c.execute(
            "SELECT recording_id, zarr_path FROM dataset_context_current "
            "WHERE recording_id LIKE ? AND zarr_use='analysis' AND dataset_status='active' "
            "ORDER BY recording_id",
            (pattern,),
        ).fetchall()
    finally:
        c.close()
    seen: dict[str, str] = {}
    for r in rows:
        rid, zp = r["recording_id"], r["zarr_path"]
        if require_on_disk and not Path(zp).is_dir():
            continue
        # First reachable row per recording wins; identical duplicates collapse harmlessly.
        seen.setdefault(rid, zp)
    return sorted(seen.items())


def nav(node, parts):
    """Stepwise group navigation. Full-path string indexing is flaky on these stores."""
    for p in parts:
        keys = list(node.group_keys())
        if p not in keys:
            raise KeyError((p, keys[:6]))
        node = node[p]
    return node


def latest(group):
    """The lexicographically-last child group (runs are timestamp/id keyed)."""
    return group[sorted(group.group_keys())[-1]]


def open_distance_run(
    root,
    *,
    run_name: str = "latest",
) -> ChaserDistanceReadSnapshot:
    """Load one exact, verified canonical chaser-distance snapshot.

    The historical helper returned the lexicographically last raw Zarr child.
    Normal future-recording analyses must instead cross the typed publication
    boundary.  Standalone legacy inspection scripts that still index the result
    as a Zarr group now fail visibly rather than silently guessing a frame.
    """

    return load_chaser_distance_run(root, run_name=str(run_name))


def load_epochs(root) -> dict:
    """Parse `stimulus_epoch_runs` windows -> {canonical_label: (start_frame, end_frame)}.

    Canonicalisation: any label containing 'pre' -> 'pre', 'post' -> 'post', and
    'train'/'chase' -> 'chase'. On these recordings all three windows are present, so
    this matches the per-analysis parsing the scratch scripts used.
    """
    w = latest(nav(root, ["analysis", "stimulus_epoch_runs"]))["windows"]
    starts = np.asarray(w["start_frame"][:])
    ends = np.asarray(w["end_frame"][:])
    labels = [x.tobytes().decode("utf-8", "ignore").strip("\x00") for x in np.asarray(w["label_bytes"][:])]
    ep: dict[str, tuple[int, int]] = {}
    for s, e, lab in zip(starts, ends, labels):
        low = lab.lower()
        if "pre" in low:
            key = "pre"
        elif "post" in low:
            key = "post"
        else:
            key = "chase"  # 'training' / 'chase'
        ep[key] = (int(s), int(e))
    return ep


def role_name(o) -> str:
    """Canonical role ('aggressive'/'inert') across the legacy and behavior-profile APIs.

    The chaser roles resolver was generalized (commit c0f5e158): it now returns
    ChaserQuadrantRole (`.behavior_class`, `.chaser_index`) instead of the legacy object
    (`.object_role`, `.object_index`). These accessors work with either.
    """
    return o.object_role if hasattr(o, "object_role") else o.behavior_class


def role_index(o) -> int:
    return o.object_index if hasattr(o, "object_index") else o.chaser_index


def resolve_object_roles(root) -> dict:
    """{role: chaser/object index} from the recording's stimulus protocol payload."""
    stim_par = nav(root, ["analysis", "stimulus_runs"])
    stim = next(stim_par[k] for k in stim_par.group_keys() if "protocol_json" in dict(stim_par[k].attrs))
    payload = json.loads(str(stim.attrs["protocol_json"]))
    return {role_name(o): role_index(o) for o in resolve_object_roles_from_protocol_payload(payload)}


_DENSE_TRACK_FIELDS = {
    "speed_raw_mm": "raw",
    "speed_filtered_mm": "filtered",
    "speed_smoothed_mm": "smoothed",
    "speed_averaged_mm": "averaged",
}


def load_dense_kinematics(
    root,
    total_frames: int,
    fields=("speed_smoothed_mm",),
    *,
    track_kinematics_run: str = "latest",
    track_id: int = 0,
):
    """Dense per-acquisition-frame arrays from one verified offline track run.

    Returns (dict field->dense array of shape (total_frames,), sample_valid dense bool).
    Immobility/speed metrics MUST use `speed_smoothed_mm` -- raw centroid speed has a
    ~1.6 mm/s noise floor that makes sub-threshold "stillness" measure tracking jitter.

    The legacy-looking field labels are controlled logical metric names, not Zarr
    paths.  Physical discovery, selector eligibility, coordinate binding, and
    payload verification are delegated to the strict track-motion reader.
    """
    frame_count = int(total_frames)
    if frame_count < 0:
        raise ValueError("total_frames must be nonnegative.")
    requested_fields = tuple(dict.fromkeys(str(field) for field in fields))
    unsupported = tuple(
        field for field in requested_fields if field not in _DENSE_TRACK_FIELDS
    )
    if unsupported:
        raise ValueError(
            "Unsupported verified dense-track field(s): "
            + ", ".join(unsupported)
        )
    required_levels = tuple(
        dict.fromkeys(_DENSE_TRACK_FIELDS[field] for field in requested_fields)
    )
    track = load_track_kinematics_track(
        root,
        run_name=str(track_kinematics_run),
        scope="offline",
        track_id=int(track_id),
        required_speed_levels=required_levels,
    )
    if track.source_acquisition_frame_index is None:
        raise ValueError(
            f"Verified track motion {track.track_path} has no acquisition-frame identity."
        )
    fi = np.asarray(track.source_acquisition_frame_index, dtype=np.int64).reshape(-1)
    if np.any(fi < 0) or np.any(fi >= frame_count):
        raise ValueError(
            f"Verified acquisition-frame identities in {track.track_path} exceed "
            f"the declared recording extent [0, {frame_count})."
        )
    if np.unique(fi).shape[0] != fi.shape[0]:
        raise ValueError(
            f"Verified track motion {track.track_path} repeats acquisition-frame identities."
        )
    if track.sample_valid is None:
        raise ValueError(
            f"Verified track motion {track.track_path} has no sample_valid surface."
        )
    sample_valid = np.asarray(track.sample_valid, dtype=bool).reshape(-1)
    if sample_valid.shape != fi.shape:
        raise ValueError(
            f"Verified sample_valid length {sample_valid.shape[0]} does not match "
            f"acquisition-frame length {fi.shape[0]}."
        )
    out = {}
    for field in requested_fields:
        level = _DENSE_TRACK_FIELDS[field]
        values = np.asarray(track.speed_mm_by_level[level], dtype=np.float64).reshape(-1)
        if values.shape != fi.shape:
            raise ValueError(
                f"Verified {field} length {values.shape[0]} does not match "
                f"acquisition-frame length {fi.shape[0]}."
            )
        dense = np.full(frame_count, np.nan)
        dense[fi] = values
        out[field] = dense
    valid = np.zeros(frame_count, bool)
    valid[fi] = sample_valid
    return out, valid
