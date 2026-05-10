"""Runtime cascade invalidation for recording step status.

When a pipeline step produces new output (status="ok"), all downstream steps
become potentially stale.  This module marks them ``"missing"`` in the registry
so that dashboards reflect staleness immediately rather than waiting for the
next maintenance backfill.

The runtime cascade is best-effort.  The maintenance backfill in
``maintenance.py`` remains the authoritative full-reconciliation path that reads
zarr and checks source-run matches.

**Scope boundary**: This cascade fires only when a *new run* is produced (full
inference or refinement).  Manual corrections to individual data points within
an existing run use the granular stale-marking system in
``fisheye.shared.keypoint_stale`` instead.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, FrozenSet, Optional, Set

from .stage_catalog import canonical_stage_id, invalidation_map
from .status_ledger import upsert_recording_step_status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency graph: step  →  frozenset of its *direct* dependents.
#
# When step X completes with a new run, every step reachable from X in this
# graph should be marked ``"missing"`` because it was produced from the
# previous version of X (or an ancestor of X).
# ---------------------------------------------------------------------------
STEP_DEPENDENTS: Dict[str, FrozenSet[str]] = {
    step_name: frozenset(dependents)
    for step_name, dependents in invalidation_map().items()
}

_CASCADE_SOURCE_PREFIX = "runtime_cascade_invalidation"

# Statuses that already indicate the step needs (re-)work.  Writing
# ``"missing"`` over these would just add noise to the history table.
_SKIP_STATUSES = frozenset({"missing", "absent", "na"})


def get_transitive_dependents(step_name: str) -> FrozenSet[str]:
    """Return all steps transitively downstream of *step_name* (excluding itself)."""
    try:
        resolved_step_name = canonical_stage_id(step_name)
    except KeyError:
        resolved_step_name = step_name
    visited: Set[str] = set()
    frontier = list(STEP_DEPENDENTS.get(resolved_step_name, frozenset()))
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(STEP_DEPENDENTS.get(current, frozenset()))
    return frozenset(visited)


def invalidate_downstream_steps(
    registry: object,
    *,
    dataset_id: str,
    step_name: str,
    source: str,
    recording_id: Optional[str] = None,
    trigger_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Mark all transitive dependents of *step_name* as ``"missing"``.

    Parameters
    ----------
    registry
        Open :class:`Registry` connection (caller owns open/close lifecycle).
    dataset_id
        The dataset whose downstream steps are being invalidated.
    step_name
        The step that just completed successfully.
    source
        The source tag of the original upsert (e.g.
        ``"runtime_predict_detections"``).  Used to build the cascade source
        string for traceability.
    recording_id
        Passed through to :func:`upsert_recording_step_status` so it resolves
        correctly without an extra DB lookup.
    trigger_run_name
        The ``run_name`` of the triggering step, recorded in ``details_json``
        for provenance.

    Returns
    -------
    dict
        Summary with keys ``steps_invalidated``, ``steps_skipped``, and
        ``errors``.
    """
    try:
        resolved_step_name = canonical_stage_id(step_name)
    except KeyError:
        resolved_step_name = step_name

    downstream = get_transitive_dependents(resolved_step_name)
    if not downstream:
        return {"steps_invalidated": [], "steps_skipped": [], "errors": []}

    cascade_source = f"{_CASCADE_SOURCE_PREFIX}:{source}"
    details_payload: Dict[str, Any] = {
        "cascade_trigger_step": resolved_step_name,
    }
    if trigger_run_name is not None:
        details_payload["cascade_trigger_run"] = trigger_run_name
    details_text = json.dumps(details_payload, sort_keys=True)

    result: Dict[str, Any] = {
        "steps_invalidated": [],
        "steps_skipped": [],
        "errors": [],
    }

    for dep_step in sorted(downstream):
        try:
            existing = registry.conn.execute(
                "SELECT status FROM recording_step_status"
                " WHERE dataset_id = ? AND step_name = ?;",
                (dataset_id, dep_step),
            ).fetchone()

            if existing is not None:
                current_status = str(existing["status"]).lower()
                if current_status in _SKIP_STATUSES:
                    result["steps_skipped"].append(dep_step)
                    continue

            upsert_recording_step_status(
                registry,
                dataset_id=dataset_id,
                recording_id=recording_id,
                step_name=dep_step,
                status="missing",
                run_name=None,
                method=None,
                coverage_pct=None,
                details_json=details_text,
                source=cascade_source,
            )
            result["steps_invalidated"].append(dep_step)
        except Exception as exc:
            logger.warning(
                "cascade invalidation failed for step %r on dataset %r: %s",
                dep_step,
                dataset_id,
                exc,
            )
            result["errors"].append(f"{dep_step}: {exc}")

    if result["steps_invalidated"]:
        logger.info(
            "cascade from %s on %s: invalidated %s, skipped %s",
            step_name,
            dataset_id,
            result["steps_invalidated"],
            result["steps_skipped"],
        )

    return result


__all__ = [
    "STEP_DEPENDENTS",
    "get_transitive_dependents",
    "invalidate_downstream_steps",
]
