"""A recording camera's dish design, from Orange's recording geometry contract.

Unified (v2) recordings carry no ``dish_design`` in their manifest. Orange
records it per camera in ``raw/recording_geometry_bundle/recording_geometry_contract.json``
(agent-contracts orange-palette/dish_rim_registration_v4):

- ``cameras.<serial>.tank_design.tank_design_id`` with status ``referenced``;
- every rim observation under that camera repeats it as
  ``physical_geometry.dish_design_id``.

``recording_dish_design`` returns that value, or None when the recording has
no geometry contract or the camera's contract names no dish design. It refuses
an unsupported contract version, a malformed tank design, rim observations
that disagree with the tank design, and a caller-supplied value that disagrees
with the contract. It never guesses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fisheye.shared.recording_geometry import (
    GEOMETRY_CONTRACT_VERSIONS,
    RECORDING_GEOMETRY_CONTRACT_SCHEMA_ID,
    RecordingGeometryError,
)

GEOMETRY_CONTRACT_RELATIVE = Path("raw/recording_geometry_bundle/recording_geometry_contract.json")


def _dish_design_ids(node: Any, found: set) -> set:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "dish_design_id":
                found.add(value)
            else:
                _dish_design_ids(value, found)
    elif isinstance(node, list):
        for value in node:
            _dish_design_ids(value, found)
    return found


def recording_dish_design(
    recording_dir: Path | str | None,
    camera_id: Optional[str],
    *,
    declared: Optional[str] = None,
) -> Optional[str]:
    """The contract's dish design for ``camera_id``, else ``declared``.

    ``declared`` is a value from another on-disk source (manifest, attrs). It
    is returned unchanged when there is no contract value, and must agree with
    the contract when there is one.
    """

    path = Path(recording_dir) / GEOMETRY_CONTRACT_RELATIVE if recording_dir else None
    if path is None or not camera_id or not path.is_file():
        return declared
    label = str(GEOMETRY_CONTRACT_RELATIVE)
    try:
        contract = json.loads(path.read_bytes())
    except ValueError as exc:
        raise RecordingGeometryError(f"{label} is not JSON: {exc}") from exc
    version = contract.get("schema_version") if isinstance(contract, dict) else None
    if (
        not isinstance(contract, dict)
        or contract.get("schema_id") != RECORDING_GEOMETRY_CONTRACT_SCHEMA_ID
        or type(version) is not int
        or version not in GEOMETRY_CONTRACT_VERSIONS
    ):
        raise RecordingGeometryError(
            f"{label} must be {RECORDING_GEOMETRY_CONTRACT_SCHEMA_ID!r} version "
            f"{' or '.join(str(v) for v in sorted(GEOMETRY_CONTRACT_VERSIONS))}."
        )
    cameras = contract.get("cameras")
    camera = cameras.get(str(camera_id)) if isinstance(cameras, dict) else None
    if not isinstance(camera, dict):
        return declared
    tank = camera.get("tank_design")
    observed = _dish_design_ids(camera, set())
    if tank is None:
        if observed:
            raise RecordingGeometryError(
                f"{label} camera {camera_id} has rim dish_design_id {sorted(map(str, observed))} "
                "but no tank_design."
            )
        return declared
    design = tank.get("tank_design_id") if isinstance(tank, dict) else None
    if not isinstance(tank, dict) or tank.get("status") != "referenced" or not (
        isinstance(design, str) and design.strip() == design and design
    ):
        raise RecordingGeometryError(
            f"{label} camera {camera_id} tank_design must be status 'referenced' "
            f"with a tank_design_id: {tank!r}"
        )
    if observed - {design}:
        raise RecordingGeometryError(
            f"{label} camera {camera_id} rim dish_design_id {sorted(map(str, observed))} "
            f"disagrees with tank_design_id {design!r}."
        )
    if declared is not None and declared != design:
        raise RecordingGeometryError(
            f"dish_design {declared!r} disagrees with {label} camera {camera_id} "
            f"tank_design_id {design!r}."
        )
    return design


__all__ = ["GEOMETRY_CONTRACT_RELATIVE", "recording_dish_design"]
