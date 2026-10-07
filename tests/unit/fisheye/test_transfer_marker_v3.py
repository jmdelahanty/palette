"""Sealer 2.0.0 completion marker v3: accepted with v2, sealer kept as provenance."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    TransferSnapshotError,
    canonical_bytes,
    verify_transfer_snapshot,
)
from fisheye.utils import citrus_transfer_v2_poller as poller
from fisheye.utils import organize_transfer_recordings as organizer

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "recording_transfer_v2" / "rolling"
SEALER = {"package": "citrus-recording-transfer", "version": "2.0.0"}


def _bundle(tmp_path: Path, *, v3: bool = True, sealer: dict | None = SEALER) -> Path:
    root = Path(shutil.copytree(FIXTURE, tmp_path / "staging" / "session"))
    marker = json.loads((root / MARKER_NAME).read_bytes())
    if v3:
        marker.update(schema_id="citrus.transfer_completion_marker.v3", schema_version=3)
        if sealer is not None:
            marker["sealer"] = sealer
    (root / MARKER_NAME).write_bytes(canonical_bytes(marker))
    return root


def test_v3_marker_is_verified_and_its_sealer_reaches_the_parent_manifest(tmp_path):
    root = _bundle(tmp_path)
    assert verify_transfer_snapshot(root).sealer == SEALER
    plan = organizer.build_transfer_organization_plan(root, destination_root=tmp_path / "out")
    assert plan["transfer_sealer"] == SEALER
    manifest = organizer._parent_manifest(plan, plan["parents"][0])
    assert manifest["source_transfer"]["sealer"] == SEALER


def test_v2_marker_still_verifies_and_plans_carry_no_sealer(tmp_path):
    root = _bundle(tmp_path, v3=False)
    assert verify_transfer_snapshot(root).sealer is None
    plan = organizer.build_transfer_organization_plan(root, destination_root=tmp_path / "out")
    assert "transfer_sealer" not in plan
    assert "sealer" not in organizer._parent_manifest(plan, plan["parents"][0])["source_transfer"]


@pytest.mark.parametrize(
    "sealer", [None, {"package": "citrus-recording-transfer", "version": "2.0"},
               {"package": "someone-else", "version": "2.0.0"}],
)
def test_v3_marker_needs_a_well_formed_sealer(tmp_path, sealer):
    with pytest.raises(TransferSnapshotError, match="marker_v3 violates the transfer-v2 envelope schema"):
        verify_transfer_snapshot(_bundle(tmp_path, sealer=sealer))


def test_v2_marker_may_not_carry_a_sealer(tmp_path):
    root = _bundle(tmp_path, v3=False)
    marker = json.loads((root / MARKER_NAME).read_bytes())
    marker["sealer"] = SEALER
    (root / MARKER_NAME).write_bytes(canonical_bytes(marker))
    with pytest.raises(TransferSnapshotError, match="marker violates"):
        verify_transfer_snapshot(root)


def test_poller_accepts_v3_markers(tmp_path):
    root = _bundle(tmp_path)
    assert poller.check_marker(root / MARKER_NAME)["schema_version"] == 3
