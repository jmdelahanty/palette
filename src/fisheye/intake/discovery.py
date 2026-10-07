"""Which deliveries intake still has work for.

Targets come from two durable sources only: live sealed transfer-v2 markers
under the staging directory, and the organizer's intake states under
``<destination_root>/.transfer_intake``. Staging (marker included) is
retired during finalization, so a marker-only scan would orphan half-retired
and imported-but-unregistered deliveries. Runner sentinels and the poller's
``.processing_state*`` files are never read.

Discovery runs every cron tick, so it is deliberately cheap: it reads small
JSON files and, for retired deliveries, one read-only registry query for the
recorded receipt bindings. It never re-verifies receipts or live surfaces;
that is the probes' job. Its ``import_recorded``/``register_recorded`` flags
are therefore "evidence recorded", not the probe verdicts that define done.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from fisheye.intake.delivery import (
    DURABLE_STATES,
    STATE_DIRECTORY,
    STATE_FILE,
    admission_mode,
    load_durable_state,
)
from fisheye.intake.outcomes import IntakeRefused
from fisheye.shared.recording_transfer_snapshot import (
    CONSUMER_PROFILE,
    MARKER_NAME,
    MARKER_SCHEMAS,
    SNAPSHOT_PATH,
)

DISCOVER_SCHEMA = "palette.intake.discover.v1"
LEGACY_MARKER_SCHEMA = "citrus.transfer_completion_marker.v1"


class MarkerRefusal(ValueError):
    """A marker that is not a complete transfer-v2 delivery."""


def check_marker(marker_path: Path) -> dict | None:
    """Return a v2/v3 marker, None for a legacy v1 marker; raise if malformed."""

    marker_bytes = marker_path.read_bytes()
    try:
        marker = json.loads(marker_bytes)
    except ValueError as exc:
        raise MarkerRefusal(f"invalid JSON: {exc}") from exc
    if not isinstance(marker, dict):
        raise MarkerRefusal("marker is not a JSON object")
    if marker.get("schema_id") == LEGACY_MARKER_SCHEMA:
        return None
    snapshot = marker.get("snapshot")
    if (
        marker.get("schema_id") not in MARKER_SCHEMAS
        or marker.get("schema_version") != MARKER_SCHEMAS[marker["schema_id"]][0]
        or marker.get("status") != "transfer_complete"
        or marker.get("required_consumer_profile") != CONSUMER_PROFILE
        or not isinstance(snapshot, dict)
        or snapshot.get("path") != SNAPSHOT_PATH
        or marker.get("recording_payload_kind") not in ("citrus_h5", "video_only")
    ):
        raise MarkerRefusal(
            f"not a complete transfer-v2 marker (schema_id={marker.get('schema_id')!r})"
        )
    snapshot_file = marker_path.parent / SNAPSHOT_PATH
    if not snapshot_file.is_file():
        raise MarkerRefusal("snapshot file missing")
    digest = hashlib.sha256(snapshot_file.read_bytes()).hexdigest()
    if digest != snapshot.get("sha256") or marker.get("snapshot_id") != f"sha256:{digest}":
        raise MarkerRefusal("snapshot bytes do not match marker binding")
    marker["_marker_sha256"] = hashlib.sha256(marker_bytes).hexdigest()
    return marker


@dataclass(frozen=True)
class Target:
    snapshot_sha: str
    state: str  # "marker" or a durable organizer status
    admission_mode: str
    legacy_mode: bool
    has_plan: bool
    session_dir: str | None
    marker_path: str | None
    import_recorded: bool
    register_recorded: bool | None
    zarr_paths: tuple[str, ...]
    producer_git_sha: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot_sha": self.snapshot_sha,
            "state": self.state,
            "admission_mode": self.admission_mode,
            "legacy_mode": self.legacy_mode,
            "has_plan": self.has_plan,
            "session_dir": self.session_dir,
            "marker_path": self.marker_path,
            "import_recorded": self.import_recorded,
            "register_recorded": self.register_recorded,
            "zarr_paths": list(self.zarr_paths),
            "producer_git_sha": self.producer_git_sha,
        }


@dataclass(frozen=True)
class Discovery:
    staging_dir: str
    destination_root: str
    registry: str | None
    targets: tuple[Target, ...]
    refused_markers: tuple[dict, ...]
    legacy_markers: tuple[str, ...]
    registry_error: str | None = None
    unreadable_states: tuple[dict, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": DISCOVER_SCHEMA,
            "staging_dir": self.staging_dir,
            "destination_root": self.destination_root,
            "registry": self.registry,
            "registry_error": self.registry_error,
            "probe_depth": "recorded",
            "targets": [target.to_json() for target in self.targets],
            "refused_markers": list(self.refused_markers),
            "legacy_markers": list(self.legacy_markers),
            "unreadable_states": list(self.unreadable_states),
        }


def _recorded_bindings(registry: Path, receipts: Iterable[str]) -> tuple[set[str] | None, str | None]:
    """Receipt digests the registry binds, read-only, or (None, why) if unreadable.

    No receipts to look up means no registry access at all.
    """

    wanted = sorted(set(receipts))
    if not wanted:
        return set(), None
    path = Path(registry)
    if not path.is_file():
        return None, f"registry not found: {path}"
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return None, f"registry unreadable: {exc}"
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT receipt_sha256 FROM recording_import_receipt_bindings "
            f"WHERE receipt_sha256 IN ({','.join('?' * len(wanted))})",
            wanted,
        ).fetchall()
    except sqlite3.Error as exc:
        return None, f"registry unreadable: {exc}"
    finally:
        connection.close()
    return {str(row[0]) for row in rows}, None


def _recorded_producer(pairs: list[tuple[str, str]]) -> str | None:
    """Producer commit from the receipt files (no verification), if one."""

    from fisheye.intake.probes import receipt_producer_git_sha

    try:
        shas = {receipt_producer_git_sha(zarr, receipt) for zarr, receipt in pairs}
    except Exception:
        return None
    return shas.pop() if len(shas) == 1 else None


def _durable_target(
    sha: str, state: dict, registry: Path | None, bound: set[str] | None
) -> Target | None:
    from fisheye.shared.recording_import_receipt import recording_import_receipt_path
    from fisheye.utils.organize_transfer_recordings import parent_zarr_paths

    plan = state["plan"]
    status = state.get("status")
    zarr_paths = tuple(str(path) for path in parent_zarr_paths(plan))
    receipts = state.get("import_receipts") or {}
    pairs: list[tuple[str, str]] = []
    try:
        pairs = [
            (zarr, receipts[parent["identity"]["recording_id"]])
            for parent, zarr in zip(plan["parents"], zarr_paths)
        ]
        receipt_files = all(
            recording_import_receipt_path(Path(zarr), receipt).is_file() for zarr, receipt in pairs
        )
    except (KeyError, ValueError):
        pairs, receipt_files = [], False
    import_recorded = status == "complete" and receipt_files
    # None (unknown) without a registry or when it could not be read: a
    # transient read failure never makes a delivery look unregistered.
    register_recorded = None
    if registry is not None and bound is not None:
        register_recorded = bool(import_recorded and set(receipts.values()) <= bound)
    if status == "complete" and register_recorded:
        return None  # retired and registered: nothing left to do
    mode = admission_mode(state)
    return Target(
        snapshot_sha=sha,
        state=str(status),
        admission_mode=mode,
        legacy_mode=mode == "job",
        has_plan=True,
        session_dir=plan.get("source_dir"),
        marker_path=None,
        import_recorded=import_recorded,
        register_recorded=register_recorded,
        zarr_paths=zarr_paths,
        producer_git_sha=_recorded_producer(pairs) if receipt_files and pairs else None,
    )


def discover(
    staging_dir: Path, destination_root: Path, *, registry: Path | None = None
) -> Discovery:
    """List every delivery with intake work left, from durable evidence only.

    Live sealed markers give fresh deliveries (state ``marker``). Durable
    states ``reserved``/``materialized``/``retiring`` are resumable; a
    ``complete`` delivery is listed only while its receipts are not all bound
    in ``registry`` (always listed when no registry is given or it cannot be
    read; then ``registry_error`` says why and ``register_recorded`` is
    null, unknown, never false). Each target
    carries its stored admission mode; a ``job`` mode delivery is flagged
    ``legacy_mode`` so the caller reports it rather than resuming it.
    """

    staging = Path(staging_dir)
    destination = Path(destination_root)
    if not staging.is_dir():
        raise IntakeRefused(f"staging directory does not exist: {staging}")
    targets: dict[str, Target] = {}
    refused: list[dict] = []
    legacy: list[str] = []
    unreadable: list[dict] = []

    states: dict[str, dict] = {}
    intake_root = destination / STATE_DIRECTORY
    if intake_root.is_dir():
        for state_file in sorted(intake_root.glob(f"*/{STATE_FILE}")):
            sha = state_file.parent.name
            try:
                state = load_durable_state(destination, sha)
            except IntakeRefused as exc:
                unreadable.append({"path": str(state_file), "kind": "malformed", "reason": str(exc)})
                continue
            except OSError as exc:  # one bad state never aborts discovery
                unreadable.append({"path": str(state_file), "kind": "unreadable", "reason": str(exc)})
                continue
            if state is not None and state.get("status") in DURABLE_STATES:
                states[sha] = state
    bound, registry_error = None, None
    if registry is not None:
        bound, registry_error = _recorded_bindings(
            Path(registry),
            (receipt for state in states.values() for receipt in (state.get("import_receipts") or {}).values()),
        )
    for sha, state in states.items():
        target = _durable_target(sha, state, Path(registry) if registry is not None else None, bound)
        if target is not None:
            targets[sha] = target

    for marker_path in sorted(staging.rglob(MARKER_NAME)):
        try:
            marker = check_marker(marker_path)
        except (MarkerRefusal, OSError) as exc:
            refused.append({"path": str(marker_path), "reason": str(exc)})
            continue
        if marker is None:
            legacy.append(str(marker_path))
            continue
        sha = marker["snapshot_id"].removeprefix("sha256:")
        if sha in states:
            current = targets.get(sha)
            if current is not None:
                targets[sha] = Target(**{**current.__dict__, "marker_path": str(marker_path)})
            continue
        targets[sha] = Target(
            snapshot_sha=sha,
            state="marker",
            admission_mode="unset",
            legacy_mode=False,
            has_plan=False,
            session_dir=str(marker_path.parent),
            marker_path=str(marker_path),
            import_recorded=False,
            register_recorded=False if registry is not None and registry_error is None else None,
            zarr_paths=(),
        )
    return Discovery(
        staging_dir=str(staging),
        destination_root=str(destination),
        registry=str(registry) if registry is not None else None,
        targets=tuple(targets[sha] for sha in sorted(targets)),
        refused_markers=tuple(refused),
        legacy_markers=tuple(legacy),
        registry_error=registry_error,
        unreadable_states=tuple(unreadable),
    )


__all__ = [
    "DISCOVER_SCHEMA",
    "Discovery",
    "MarkerRefusal",
    "Target",
    "check_marker",
    "discover",
]
