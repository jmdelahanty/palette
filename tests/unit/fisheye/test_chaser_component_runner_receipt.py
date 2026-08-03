from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_component_publication import (
    ChaserComponentContract,
    component_record_sha256,
    persist_chaser_component_manifest,
)
from fisheye.analysis_workflows import chaser_component_receipt as mod


def _snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        run_name="base",
        run_path="analysis/chaser_distance_runs/base",
        publication_seal_ref=(
            "/analysis/chaser_distance_runs/base@chaser_distance_publication_seal"
        ),
        publication_seal_sha256="a" * 64,
        surface_manifest_ref=(
            "/analysis/chaser_distance_runs/base@chaser_distance_surface_manifest"
        ),
        surface_manifest_sha256="b" * 64,
        row_identity_ref="/analysis/chaser_distance_runs/base@row_identity_contract",
        row_identity_sha256="c" * 64,
        authority_record=lambda: {
            "schema_id": "palette.chaser_distance_read_authority",
            "schema_version": 1,
            "run_ref": "/analysis/chaser_distance_runs/base",
        },
    )


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3, use_consolidated=False)
    component = root.require_group(
        "analysis/chaser_distance_runs/base/egocentric_bearing/candidate"
    )
    component.create_array(
        "heading_deg",
        data=np.asarray([1.5, 2.5], dtype=np.float32),
        chunks=(2,),
    )
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="egocentric_bearing/candidate",
        contract=ChaserComponentContract(
            component_family="egocentric_bearing",
            component_name="candidate",
            semantic_schema_id="palette.chaser_egocentric_bearing",
            semantic_schema_version=1,
            method_id="egocentric_bearing",
            method_version="1",
            parameters={},
            source_authorities={"track": {"manifest_sha256": "d" * 64}},
        ),
    )
    component.attrs["stage_selector_eligible"] = False
    return path


def test_runner_receipt_binds_exact_ineligible_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _archive(tmp_path)
    monkeypatch.setattr(mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())

    receipt = mod.build_chaser_component_runner_receipt(
        path,
        chaser_distance_run="base",
        component_requests=(("egocentric_bearing", "candidate"),),
    )

    assert receipt["status"] == "complete"
    assert receipt["requested_component_count"] == 1
    row = receipt["components"][0]
    assert row["validation"]["valid"] is True
    assert row["authority_mode"] == "explicit_dependency_handle"
    assert row["selector_eligible"] is False
    assert row["dependency_handle"]["component_name"] == "candidate"
    body = {key: value for key, value in receipt.items() if key != "record_sha256"}
    assert receipt["record_sha256"] == component_record_sha256(body)
    mod.validate_chaser_component_runner_receipt(
        receipt,
        expected_zarr_path=path,
    )


def test_runner_receipt_rejects_duplicate_or_missing_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _archive(tmp_path)
    monkeypatch.setattr(mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())

    with pytest.raises(ValueError, match="duplicate"):
        mod.build_chaser_component_runner_receipt(
            path,
            chaser_distance_run="base",
            component_requests=(
                ("egocentric_bearing", "candidate"),
                ("egocentric_bearing", "candidate"),
            ),
        )
    with pytest.raises(ValueError, match="unavailable"):
        mod.build_chaser_component_runner_receipt(
            path,
            chaser_distance_run="base",
            component_requests=(("egocentric_bearing", "missing"),),
        )


def test_component_request_is_exact() -> None:
    assert mod.parse_component_request("egocentric_bearing=candidate") == (
        "egocentric_bearing",
        "candidate",
    )
    with pytest.raises(ValueError, match="FAMILY=NAME"):
        mod.parse_component_request("candidate")
    with pytest.raises(ValueError, match="controlled"):
        mod.parse_component_request("family=bad/name")


def test_runner_receipt_rejects_rehashed_nested_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _archive(tmp_path)
    monkeypatch.setattr(mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())
    receipt = mod.build_chaser_component_runner_receipt(
        path,
        chaser_distance_run="base",
        component_requests=(("egocentric_bearing", "candidate"),),
    )
    tampered = copy.deepcopy(receipt)
    tampered["components"][0]["validation"]["payload_array_count"] = 99
    body = {key: value for key, value in tampered.items() if key != "record_sha256"}
    tampered["record_sha256"] = component_record_sha256(body)

    with pytest.raises(ValueError, match="receipt or payload changed"):
        mod.validate_chaser_component_runner_receipt(tampered)
