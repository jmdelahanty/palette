from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_component_publication import (
    COMPONENT_MANIFEST_ATTR,
    COMPONENT_MANIFEST_DIGEST_ATTR,
    COMPONENT_SELECTOR_ATTR,
    COMPONENT_SELECTOR_DIGEST_ATTR,
    ChaserComponentContract,
    persist_chaser_component_manifest,
    validate_chaser_component_selector,
)
from fisheye.analysis_workflows.materializers import chaser_component as mod
from fisheye.analysis_workflows.materializers import atomic_run_publisher


def _snapshot():
    return SimpleNamespace(
        run_path="analysis/chaser_distance_runs/canonical",
        publication_seal_ref=(
            "/analysis/chaser_distance_runs/canonical@chaser_distance_publication_seal"
        ),
        publication_seal_sha256="a" * 64,
        surface_manifest_ref=(
            "/analysis/chaser_distance_runs/canonical@chaser_distance_surface_manifest"
        ),
        surface_manifest_sha256="b" * 64,
        row_identity_ref=(
            "/analysis/chaser_distance_runs/canonical@row_identity_contract"
        ),
        row_identity_sha256="c" * 64,
        authority_record=lambda: {
            "schema_id": "palette.chaser_distance_read_authority",
            "schema_version": 1,
            "run_ref": "/analysis/chaser_distance_runs/canonical",
        },
    )


def _contract():
    return ChaserComponentContract(
        component_family="chaser_escape_events",
        component_name="escape_v2",
        semantic_schema_id="palette.chaser_escape_events",
        semantic_schema_version=2,
        method_id="palette.chaser_escape_event_detector",
        method_version="2.1.0",
        parameters={"threshold_mm": 4.0},
        source_authorities={"bout_manifest_sha256": "d" * 64},
    )


def _fixture(tmp_path: Path):
    source = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=3, use_consolidated=False)
    root.require_group("analysis/chaser_distance_runs/canonical")

    local = tmp_path / "escape_component.zarr"
    component = zarr.open_group(
        str(local), mode="w", zarr_format=3, use_consolidated=False
    )
    component.attrs["coordinate_space"] = "arena_relative_canvas_px"
    component.create_array(
        "frame_index",
        data=np.asarray([2, 7], dtype=np.int64),
        chunks=(2,),
    )
    events = component.require_group("events")
    events.attrs["row_axis"] = "event"
    distances = events.create_array(
        "distance_mm",
        data=np.asarray([1.25, 3.5], dtype=np.float32),
        chunks=(2,),
    )
    distances.attrs["units"] = "mm"
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    request = mod.ChaserComponentPublishRequest(
        source_zarr=source,
        local_component_path=local,
        base_run_name="canonical",
        base_run_path="analysis/chaser_distance_runs/canonical",
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    return source, local, request


def test_atomic_component_publication_commits_selector_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, local, request = _fixture(tmp_path)
    monkeypatch.setattr(mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())

    receipt = mod.publish_sealed_chaser_component(
        replace(request, activate_selector=True)
    )

    root = zarr.open_group(str(source), mode="r", zarr_format=3, use_consolidated=False)
    parent = root["analysis/chaser_distance_runs/canonical/chaser_escape_events"]
    component = parent["escape_v2"]
    selector = validate_chaser_component_selector(
        parent,
        component=component,
        snapshot=_snapshot(),
        expected_family="chaser_escape_events",
    )
    assert selector["selected_component"] == "escape_v2"
    assert component.attrs["palette_run_completion_status"] == "complete"
    assert component.attrs["stage_selector_eligible"] is True
    assert "latest" not in parent.attrs
    assert receipt["component_manifest_sha256"] == selector["component_manifest_sha256"]
    assert local.is_dir()

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        mod.publish_sealed_chaser_component(
            replace(request, activate_selector=True)
        )


def test_activation_failure_restores_selector_and_retains_failed_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, _local, request = _fixture(tmp_path)
    monkeypatch.setattr(mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())

    root = zarr.open_group(str(source), mode="a", zarr_format=3, use_consolidated=False)
    parent = root["analysis/chaser_distance_runs/canonical"].require_group(
        "chaser_escape_events"
    )
    previous_selector = {"legacy": "still authoritative"}
    parent.attrs[COMPONENT_SELECTOR_ATTR] = previous_selector
    parent.attrs[COMPONENT_SELECTOR_DIGEST_ATTR] = "e" * 64

    real_persist = mod.persist_chaser_component_selector

    def fail_after_selector(*args, **kwargs):
        real_persist(*args, **kwargs)
        raise RuntimeError("injected post-selector failure")

    monkeypatch.setattr(mod, "persist_chaser_component_selector", fail_after_selector)

    with pytest.raises(RuntimeError, match="injected post-selector failure"):
        mod.publish_sealed_chaser_component(
            replace(request, activate_selector=True)
        )

    check = zarr.open_group(
        str(source), mode="r", zarr_format=3, use_consolidated=False
    )
    parent = check["analysis/chaser_distance_runs/canonical/chaser_escape_events"]
    failed = parent["escape_v2"]
    assert parent.attrs[COMPONENT_SELECTOR_ATTR] == previous_selector
    assert parent.attrs[COMPONENT_SELECTOR_DIGEST_ATTR] == "e" * 64
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["atomic_publication_tombstone"]["public_path_retained"] is True


def test_default_publication_is_ineligible_and_preserves_stale_legacy_pointers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, _local, request = _fixture(tmp_path)
    monkeypatch.setattr(mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())
    root = zarr.open_group(str(source), mode="a", zarr_format=3, use_consolidated=False)
    parent = root["analysis/chaser_distance_runs/canonical"].require_group(
        "chaser_escape_events"
    )
    parent.attrs["latest"] = "stale_legacy_child"
    parent.attrs["latest_complete"] = "stale_legacy_child"

    receipt = mod.publish_sealed_chaser_component(request)

    check = zarr.open_group(
        str(source), mode="r", zarr_format=3, use_consolidated=False
    )
    parent = check["analysis/chaser_distance_runs/canonical/chaser_escape_events"]
    component = parent["escape_v2"]
    assert component.attrs["stage_selector_eligible"] is False
    assert component.attrs["palette_run_completion_status"] == "complete"
    assert parent.attrs["latest"] == "stale_legacy_child"
    assert parent.attrs["latest_complete"] == "stale_legacy_child"
    assert COMPONENT_SELECTOR_ATTR not in parent.attrs
    assert COMPONENT_SELECTOR_DIGEST_ATTR not in parent.attrs
    assert receipt["selector_activation_requested"] is False
    assert receipt["component_family"] == "chaser_escape_events"
    assert receipt["component_name"] == "escape_v2"
    assert receipt["final_validation"]["valid"] is True

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        mod.publish_sealed_chaser_component(request)


def test_unsealed_local_component_fails_before_destination_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, local, request = _fixture(tmp_path)
    monkeypatch.setattr(mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())
    component = zarr.open_group(
        str(local), mode="a", zarr_format=3, use_consolidated=False
    )
    del component.attrs[COMPONENT_MANIFEST_ATTR]
    del component.attrs[COMPONENT_MANIFEST_DIGEST_ATTR]

    with pytest.raises(ValueError, match="manifest"):
        mod.publish_sealed_chaser_component(request)

    check = zarr.open_group(
        str(source), mode="r", zarr_format=3, use_consolidated=False
    )
    base = check["analysis/chaser_distance_runs/canonical"]
    assert "chaser_escape_events" not in base


def test_pre_rename_interruption_can_retry_without_stale_pointer_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, _local, request = _fixture(tmp_path)
    monkeypatch.setattr(mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())
    root = zarr.open_group(str(source), mode="a", zarr_format=3, use_consolidated=False)
    parent = root["analysis/chaser_distance_runs/canonical"].require_group(
        "chaser_escape_events"
    )
    parent.attrs["latest"] = "old"
    parent.attrs["latest_complete"] = "old"

    real_copy = atomic_run_publisher._copy_tree
    calls = 0

    def interrupt_once(source_path, target_path, *, backend):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected pre-rename interruption")
        return real_copy(source_path, target_path, backend=backend)

    monkeypatch.setattr(atomic_run_publisher, "_copy_tree", interrupt_once)
    with pytest.raises(RuntimeError, match="injected pre-rename interruption"):
        mod.publish_sealed_chaser_component(request)

    interrupted = zarr.open_group(
        str(source), mode="r", zarr_format=3, use_consolidated=False
    )["analysis/chaser_distance_runs/canonical/chaser_escape_events"]
    assert interrupted.attrs["latest"] == "old"
    assert interrupted.attrs["latest_complete"] == "old"
    assert "escape_v2" not in interrupted

    receipt = mod.publish_sealed_chaser_component(request)
    assert receipt["final_validation"]["valid"] is True
    retried = zarr.open_group(
        str(source), mode="r", zarr_format=3, use_consolidated=False
    )["analysis/chaser_distance_runs/canonical/chaser_escape_events"]
    assert retried.attrs["latest"] == "old"
    assert retried.attrs["latest_complete"] == "old"
    assert retried["escape_v2"].attrs["stage_selector_eligible"] is False
