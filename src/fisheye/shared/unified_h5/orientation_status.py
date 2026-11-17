"""Per-recording orientation status, derived and never stamped (palette#377 §5, §10).

One resolver answers whether a Citrus recording's camera -> canvas handedness
is verified. Nothing is written: no existing run is mutated, and a consumer
that needs the status asks this module (Jeremy, 2026-11-15).

A recording is ``verified`` for the ``camera_to_canvas`` link on exactly one of
two bases, else ``unverified``:

1. ``citrus_declared_orientation_check_v1``: its sealed
   ``/geometry/calibration/declared_orientation`` record passes Palette's full
   check (``declared_orientation.validate_declared_orientation``: schema, rules
   R1-R4, the determinant recomputed from the active homography with
   ``w(p) > 0``, Palette's own xrandr parse, the rig pin, camera and arena).
   Jeremy, 2026-11-17: this counts as verified.
2. An orientation evidence record (``palette_orientation_evidence_v1.json``):
   a measured asymmetric-pattern check that lists the recording by Citrus
   session, arena, camera and the digest of its active homography. The 11-15
   omnifin0 check covers only the 10-17/18 phototaxis sessions (Jeremy,
   2026-11-17).

Either way, camera-to-dish parity (whether the camera image is mirrored
relative to the fish) is still declared by the rig file, not measured; every
result says so in ``camera_to_dish_parity``.

The resolver is total: a refused declared-orientation record, a mismatching
evidence entry or missing attributes give ``unverified`` with a reason; they
never raise. A recording that admission refused would also be refused there.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from . import declared_orientation
from .common import UnifiedH5ContractError

SCHEMA_ID = "palette.orientation_status"
SCHEMA_VERSION = 1
VERIFIED = "verified"
UNVERIFIED = "unverified"
CAMERA_TO_CANVAS = "camera_to_canvas"
BASIS_CITRUS_CHECK = "citrus_declared_orientation_check_v1"
CAMERA_TO_DISH_DECLARED = "declared"

EVIDENCE_FILE = "palette_orientation_evidence_v1.json"
EVIDENCE_SCHEMA_ID = "palette.orientation_evidence"
EVIDENCE_SCHEMA_VERSION = 1


def _evidence_bytes() -> bytes:
    return (
        resource_files("fisheye.shared")
        .joinpath("contracts")
        .joinpath(EVIDENCE_FILE)
        .read_bytes()
    )


@lru_cache(maxsize=1)
def evidence() -> tuple[str, Mapping[str, Any]]:
    """The evidence file's sha256 and its session_uuid -> entry index.

    Malformed evidence is a packaging defect, so it raises.
    """

    payload = _evidence_bytes()
    document = json.loads(payload)
    if (
        document.get("schema_id") != EVIDENCE_SCHEMA_ID
        or document.get("schema_version") != EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{EVIDENCE_FILE} is not {EVIDENCE_SCHEMA_ID} v{EVIDENCE_SCHEMA_VERSION}."
        )
    index: dict[str, Mapping[str, Any]] = {}
    for evidence_id, record in document["evidence"].items():
        if record.get("verified_link") != CAMERA_TO_CANVAS:
            raise ValueError(
                f"Evidence {evidence_id} does not verify {CAMERA_TO_CANVAS}."
            )
        for entry in record["recordings"]:
            session_uuid = entry["session_uuid"]
            if session_uuid in index:
                raise ValueError(
                    f"Session {session_uuid} is listed twice in {EVIDENCE_FILE}."
                )
            index[session_uuid] = {
                "evidence_id": evidence_id,
                "rig_id": record["rig_id"],
                **entry,
            }
    return "sha256:" + hashlib.sha256(payload).hexdigest(), index


def homography_sha256(h5: h5py.File, camera_serial: str) -> str:
    """sha256 of the active 3x3 homography as little-endian float64, C order."""

    matrix = np.asarray(
        h5[f"/geometry/calibration/{camera_serial}/homography_matrix"][()], dtype="<f8"
    )
    if matrix.shape != (3, 3):
        raise UnifiedH5ContractError("orientation_status_homography_shape")
    return (
        "sha256:" + hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    )


def _text(value: Any) -> str | None:
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return value if isinstance(value, str) else None


def _status(identity: Mapping[str, Any], status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        **identity,
        "orientation_status": status,
        "verified_link": CAMERA_TO_CANVAS if status == VERIFIED else None,
        "basis": None,
        "evidence_sha256": None,
        "reason": None,
        "camera_to_dish_parity": CAMERA_TO_DISH_DECLARED,
        **fields,
    }


def _identity(h5: h5py.File) -> dict[str, Any]:
    session = (
        h5[declared_orientation.SESSION].attrs
        if declared_orientation.SESSION in h5
        else {}
    )
    inputs = (
        h5[declared_orientation.INPUT].attrs if declared_orientation.INPUT in h5 else {}
    )
    return {
        "rig_id": _text(session.get("rig_id")),
        "session_uuid": _text(session.get("session_uuid")),
        "arena_id": _text(session.get("arena_id")),
        "camera_serial": _text(inputs.get("camera_serial")),
    }


def resolve_orientation_status_h5(h5: h5py.File) -> dict[str, Any]:
    """The orientation status of one open Citrus unified H5."""

    identity = _identity(h5)
    try:
        has_group = declared_orientation.GROUP in h5
        if has_group or declared_orientation.required_by_cutoff(h5):
            record = declared_orientation.validate_declared_orientation(h5)
            return _status(
                identity,
                VERIFIED,
                basis=BASIS_CITRUS_CHECK,
                recorded_space_parity_basis={
                    name: record[name]
                    for name in (
                        "viewing_side",
                        "camera_optical_mirror_count",
                        "camera_reverse_x",
                        "camera_reverse_y",
                    )
                },
            )
    except (UnifiedH5ContractError, KeyError, ValueError, TypeError) as exc:
        return _status(
            identity, UNVERIFIED, reason=f"declared_orientation_refused:{exc}"
        )

    evidence_sha, index = evidence()
    entry = index.get(identity["session_uuid"]) if identity["session_uuid"] else None
    if entry is None:
        return _status(identity, UNVERIFIED, reason="no_orientation_evidence")
    for name in ("rig_id", "arena_id", "camera_serial"):
        if identity[name] != entry[name]:
            return _status(
                identity, UNVERIFIED, reason=f"orientation_evidence_mismatch:{name}"
            )
    try:
        digest = homography_sha256(h5, identity["camera_serial"])
    except (UnifiedH5ContractError, KeyError, ValueError, TypeError):
        return _status(
            identity,
            UNVERIFIED,
            reason="orientation_evidence_mismatch:homography_unreadable",
        )
    if digest != entry["homography_matrix_sha256"]:
        return _status(
            identity,
            UNVERIFIED,
            reason="orientation_evidence_mismatch:homography_matrix_sha256",
        )
    return _status(
        identity,
        VERIFIED,
        basis=entry["evidence_id"],
        evidence_sha256=evidence_sha,
    )


def resolve_orientation_status(source_h5: str | Path) -> dict[str, Any]:
    """The orientation status of the Citrus unified H5 at ``source_h5`` (read-only)."""

    with h5py.File(Path(source_h5), "r") as h5:
        return resolve_orientation_status_h5(h5)


__all__ = [
    "BASIS_CITRUS_CHECK",
    "CAMERA_TO_CANVAS",
    "SCHEMA_ID",
    "UNVERIFIED",
    "VERIFIED",
    "evidence",
    "homography_sha256",
    "resolve_orientation_status",
    "resolve_orientation_status_h5",
]
