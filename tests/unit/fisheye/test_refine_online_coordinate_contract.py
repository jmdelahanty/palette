from __future__ import annotations

from copy import deepcopy
from datetime import datetime as RealDatetime
from io import StringIO

import numpy as np
import pytest
from rich.console import Console
import zarr

from fisheye.refinement import refine_online_detect as mod
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
    COORDINATE_DESCRIPTOR_ATTR,
    COORDINATE_DESCRIPTOR_DIGEST_SUFFIX,
    canonical_coordinate_descriptor_v2_digest,
)
from fisheye.shared.coordinate_identity import (
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_ARRAY_REF,
    STIMULUS_STATE_KEY_MODE,
    load_bound_row_identity_contract,
)
from fisheye.shared.stimulus_coordinate_contract import (
    SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
)
from tests.unit.fisheye.test_chaser_metrics_loader import _canonical_root


def _console() -> Console:
    return Console(file=StringIO(), force_terminal=False)


def _canonical_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    multi_chaser: bool = False,
) -> tuple[zarr.Group, zarr.Group]:
    root, chaser = _canonical_root(multi_chaser=multi_chaser)
    monkeypatch.setattr(mod.zarr, "open", lambda *args, **kwargs: root)
    monkeypatch.setattr(mod, "get_git_info", lambda: {})
    monkeypatch.setattr(mod, "get_environment_info", lambda: {"platform": {}})
    return root, chaser


def _run_refinement(root: zarr.Group, *, chaser_index: int = 0) -> tuple[str, zarr.Group]:
    run_name = mod.refine_online_positions(
        "ignored.zarr",
        chaser_index=chaser_index,
        window_length=3,
        polyorder=1,
        displacement_threshold=10_000.0,
        max_gap=2,
        console=_console(),
        created_at_utc="2026-07-19T12:00:00+00:00",
    )
    return run_name, root[mod.REFINED_ONLINE_GROUP][run_name]


def _replace_descriptor(node: zarr.Array, payload: dict[str, object]) -> None:
    node.attrs[COORDINATE_DESCRIPTOR_ATTR] = payload
    node.attrs[f"{COORDINATE_DESCRIPTOR_ATTR}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"] = (
        canonical_coordinate_descriptor_v2_digest(payload)
    )


def test_load_uses_exact_child_surface_and_stimulus_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)

    loaded = mod.load_online_positions("ignored.zarr", console=_console())

    np.testing.assert_array_equal(loaded.camera_frame_ids, [10, 11, 12])
    np.testing.assert_array_equal(
        loaded.source_acquisition_frame_index,
        [0, 1, 2],
    )
    np.testing.assert_array_equal(loaded.source_row_indices, [0, 1, 2])
    np.testing.assert_array_equal(loaded.stimulus_state_key, [0, 1, 2])
    assert loaded.stimulus_state_key_components == ("stimulus_frame_num",)
    np.testing.assert_allclose(
        loaded.positions,
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
    )
    assert loaded.source_descriptor.profile_id == mod.CANONICAL_ARENA_PROFILE
    assert (
        loaded.source_descriptor.source_camera_overlay.status
        == CANONICAL_OVERLAY_REQUIRES_TRANSFORM
    )
    assert loaded.source_descriptor.source_camera_overlay.transform_refs
    loaded.bound_source_handoff.assert_verified()
    assert mod.REFINED_ONLINE_GROUP not in root


def test_load_selects_chaser_by_stimulus_key_not_external_camera_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _ = _canonical_fixture(monkeypatch, multi_chaser=True)

    loaded = mod.load_online_positions(
        "ignored.zarr",
        chaser_index=1,
        console=_console(),
    )

    np.testing.assert_array_equal(
        loaded.stimulus_state_key,
        [[1, 0], [1, 1], [1, 2]],
    )
    np.testing.assert_array_equal(loaded.source_row_indices, [1, 3, 5])
    np.testing.assert_array_equal(
        loaded.source_acquisition_frame_index,
        [0, 1, 2],
    )
    np.testing.assert_array_equal(loaded.camera_frame_ids, [10, 11, 12])
    np.testing.assert_allclose(
        loaded.positions,
        [[101.0, 104.0], [102.0, 105.0], [103.0, 106.0]],
    )


def test_smoothing_never_crosses_nonconsecutive_acquisition_frames() -> None:
    positions = np.asarray(
        [[0.0, 0.0], [100.0, 100.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    acquisition_frames = np.asarray([100, 103, 104], dtype=np.int64)

    smoothed, mask = mod.smooth_positions(
        positions,
        acquisition_frames,
        np.ones(3, dtype=bool),
        window_length=3,
        polyorder=1,
    )

    np.testing.assert_array_equal(smoothed, positions)
    np.testing.assert_array_equal(mask, [True, True, True])


def test_stimulus_rows_are_ordered_by_acquisition_not_external_camera_id() -> None:
    rows, acquisition, camera = mod._select_stimulus_rows_for_refinement(
        source_keys=np.asarray(
            [[0, 100], [1, 100], [0, 101], [1, 101], [0, 102], [1, 102]],
            dtype=np.int64,
        ),
        components=("chaser_index", "stimulus_frame_num"),
        source_acquisition_frame_index=np.asarray(
            [8, 9, 4, 5, 6, 7],
            dtype=np.int64,
        ),
        camera_frame_ids=np.asarray(
            [500, 42, 499, 42, 498, 42],
            dtype=np.int64,
        ),
        chaser_index=1,
    )

    np.testing.assert_array_equal(rows, [3, 5, 1])
    np.testing.assert_array_equal(acquisition, [5, 7, 9])
    # Duplicate, non-ordering external IDs remain exact provenance.
    np.testing.assert_array_equal(camera, [42, 42, 42])


def test_duplicate_acquisition_time_fails_even_when_camera_ids_are_unique() -> None:
    with pytest.raises(
        mod.CanonicalOnlineRefinementError,
        match="one-to-one",
    ):
        mod._select_stimulus_rows_for_refinement(
            source_keys=np.asarray([100, 101], dtype=np.int64),
            components=("stimulus_frame_num",),
            source_acquisition_frame_index=np.asarray([4, 4], dtype=np.int64),
            camera_frame_ids=np.asarray([900, 901], dtype=np.int64),
            chaser_index=0,
        )


def test_normal_path_rejects_noncanonical_archive_without_legacy_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    monkeypatch.setattr(mod.zarr, "open", lambda *args, **kwargs: root)

    with pytest.raises(
        mod.CanonicalOnlineRefinementError,
        match="lacks canonical analysis/stimulus_runs",
    ):
        mod.load_online_positions("ignored.zarr", console=_console())

    assert not hasattr(mod, "resolve_online_coordinate_descriptor")


def test_refinement_publishes_stimulus_identity_temporal_authority_and_v2_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    parent = root.create_group(mod.REFINED_ONLINE_GROUP)
    parent.create_group("prior")
    parent.attrs["latest"] = "prior"
    original_private_loader = mod._load_bound_refined_online_coordinate_evidence
    observed_latest: list[object] = []
    observed_status: list[object] = []
    observed_eligibility: list[object] = []

    def record_call(args):
        observed_latest.append(
            args[0][mod.REFINED_ONLINE_GROUP].attrs.get("latest")
        )
        observed_status.append(args[1].attrs.get("publication_status"))
        observed_eligibility.append(
            args[1].attrs.get("stage_selector_eligible")
        )

    def recording_private_loader(*args, **kwargs):
        record_call(args)
        return original_private_loader(*args, **kwargs)

    monkeypatch.setattr(
        mod,
        "_load_bound_refined_online_coordinate_evidence",
        recording_private_loader,
    )

    run_name, run = _run_refinement(root)

    assert observed_latest == ["prior", "prior", "prior"]
    assert observed_status == ["staging", "complete", "complete"]
    assert observed_eligibility == [False, False, False]
    assert root[mod.REFINED_ONLINE_GROUP].attrs["latest"] == run_name
    assert root[mod.REFINED_ONLINE_GROUP].attrs[mod.RUN_LATEST_COMPLETE_ATTR] == run_name
    assert mod.RUN_LATEST_PENDING_ATTR not in root[mod.REFINED_ONLINE_GROUP].attrs
    assert run.attrs["publication_status"] == "complete"
    assert run.attrs[mod.RUN_COMPLETION_STATUS_ATTR] == mod.RUN_STATUS_COMPLETE
    assert run.attrs[mod.RUN_NAME_ATTR] == run_name
    assert run.attrs[mod.RUN_STAGE_ATTR] == "refine_online_detect"
    assert run.attrs["stage_selector_eligible"] is True
    assert "run_provenance" in run.attrs
    assert run.attrs["coordinate_contract_epoch"] == mod.COORDINATE_CONTRACT_EPOCH
    assert "instance_key" not in run
    assert "camera_frame_ids" not in run["filtered"]
    assert "camera_frame_ids" not in run["interpolated"]
    for legacy_attr in (
        "coordinate_space",
        "texture_to_camera_scale",
        "pixels_per_mm_projector",
        "legacy_space_label",
    ):
        assert legacy_attr not in run.attrs

    output_key = run[STIMULUS_STATE_KEY_ARRAY_REF]
    output_identity = load_bound_row_identity_contract(run, output_key)
    assert output_identity.contract.domain == STIMULUS_STATE_DOMAIN
    assert output_identity.contract.mode == STIMULUS_STATE_KEY_MODE
    np.testing.assert_array_equal(output_key[:], [0, 1, 2])
    np.testing.assert_array_equal(run["camera_frame_ids"][:], [10, 11, 12])
    np.testing.assert_array_equal(
        run[SOURCE_ACQUISITION_FRAME_INDEX_ARRAY][:],
        [0, 1, 2],
    )
    np.testing.assert_array_equal(run["source_row_indices"][:], [0, 1, 2])

    evidence = mod.load_bound_refined_online_coordinate_evidence(
        root,
        run,
    )
    filtered = evidence.descriptor_for("filtered").descriptor
    interpolated = evidence.descriptor_for("interpolated").descriptor
    assert filtered == interpolated
    assert filtered.source_camera_overlay.status == (
        CANONICAL_OVERLAY_REQUIRES_TRANSFORM
    )
    assert len(filtered.source_camera_overlay.transform_refs) == 2
    assert filtered.row_identity.record_ref == (
        f"/{run.path}@row_identity_contract"
    )
    assert evidence.source_temporal_authority.record_ref == (
        f"/{run.path}@source_row_temporal_authority"
    )
    np.testing.assert_array_equal(
        evidence.source_acquisition_frame_index,
        [0, 1, 2],
    )
    mapping = evidence.source_mapping.record
    assert mapping["row_identity_preserved_during_interpolation"] is True
    assert mapping["source_row_identity_contract_ref"].endswith(
        "/chaser_states@row_identity_contract"
    )
    assert mapping["source_transform_chain"] == [
        item.to_dict()
        for item in filtered.source_camera_overlay.transform_refs
    ]
    evidence.assert_verified()
    run.attrs["stage_selector_eligible"] = False
    with pytest.raises(
        mod.CanonicalOnlineRefinementError,
        match="canonical lifecycle identity",
    ):
        mod.load_bound_refined_online_coordinate_evidence(root, run)


def test_completion_rejects_identity_acquisition_and_surface_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    _, run = _run_refinement(root)

    original_key = np.asarray(run[STIMULUS_STATE_KEY_ARRAY_REF][:]).copy()
    run[STIMULUS_STATE_KEY_ARRAY_REF][0] = 999
    with pytest.raises(mod.CanonicalOnlineRefinementError, match="identity is invalid"):
        mod.load_bound_refined_online_coordinate_evidence(root, run)
    run[STIMULUS_STATE_KEY_ARRAY_REF][:] = original_key

    acquisition = run[SOURCE_ACQUISITION_FRAME_INDEX_ARRAY]
    original_acquisition = np.asarray(acquisition[:]).copy()
    acquisition[1] = 2
    with pytest.raises(
        mod.CanonicalOnlineRefinementError,
        match="acquisition-frame uniqueness|temporal authority",
    ):
        mod.load_bound_refined_online_coordinate_evidence(root, run)
    acquisition[:] = original_acquisition

    position = run["filtered/positions_px"]
    original_position = float(position[0, 0])
    position[0, 0] = original_position + 1.0
    with pytest.raises(
        mod.CanonicalOnlineRefinementError,
        match="manifest is incomplete or stale",
    ):
        mod.load_bound_refined_online_coordinate_evidence(root, run)


def test_completion_rejects_wrong_transform_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    _, run = _run_refinement(root)
    node = run["filtered/positions_px"]
    payload = deepcopy(node.attrs[COORDINATE_DESCRIPTOR_ATTR])
    overlay = payload["source_camera_overlay"]
    overlay["transform_refs"] = list(reversed(overlay["transform_refs"]))
    _replace_descriptor(node, payload)

    with pytest.raises(
        mod.CanonicalOnlineRefinementError,
        match="descriptor cannot be rebound exactly",
    ):
        mod.load_bound_refined_online_coordinate_evidence(root, run)


@pytest.mark.parametrize("mutation", ["missing_lineage", "unsupported_space"])
def test_completion_rejects_missing_lineage_or_unsupported_space(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    _, run = _run_refinement(root)
    node = run["filtered/positions_px"]
    payload = deepcopy(node.attrs[COORDINATE_DESCRIPTOR_ATTR])
    if mutation == "missing_lineage":
        payload["lineage_refs"] = payload["lineage_refs"][:-1]
    else:
        payload["profile_id"] = "stimulus_canvas_px.top_left_y_down.v1"
        payload["space_id"] = "stimulus_canvas_px"
        payload["origin"] = "top_left"
    _replace_descriptor(node, payload)

    with pytest.raises(mod.CanonicalOnlineRefinementError):
        mod.load_bound_refined_online_coordinate_evidence(root, run)


def test_failed_staging_is_deleted_and_prior_latest_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    parent = root.create_group(mod.REFINED_ONLINE_GROUP)
    parent.create_group("prior")
    parent.attrs["latest"] = "prior"
    parent.attrs[mod.RUN_LATEST_COMPLETE_ATTR] = "prior-complete"
    parent.attrs[mod.RUN_LATEST_PENDING_ATTR] = "prior-pending"

    def fail_validation(*args, **kwargs):
        raise RuntimeError("injected completion validation failure")

    monkeypatch.setattr(mod, "_validate_refined_online_run", fail_validation)

    with pytest.raises(RuntimeError, match="injected completion validation failure"):
        _run_refinement(root)

    assert list(parent.keys()) == ["prior"]
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs[mod.RUN_LATEST_COMPLETE_ATTR] == "prior-complete"
    assert parent.attrs[mod.RUN_LATEST_PENDING_ATTR] == "prior-pending"


def test_failed_fresh_complete_load_rolls_back_run_and_all_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    parent = root.create_group(mod.REFINED_ONLINE_GROUP)
    parent.create_group("prior")
    parent.attrs["latest"] = "prior"
    parent.attrs[mod.RUN_LATEST_COMPLETE_ATTR] = "prior"
    parent.attrs[mod.RUN_LATEST_PENDING_ATTR] = "older-pending"
    original_private_loader = mod._load_bound_refined_online_coordinate_evidence

    def fail_complete_load(*args, **kwargs):
        if kwargs.get("require_complete") is True:
            raise RuntimeError("injected fresh complete-load failure")
        return original_private_loader(*args, **kwargs)

    monkeypatch.setattr(
        mod,
        "_load_bound_refined_online_coordinate_evidence",
        fail_complete_load,
    )

    with pytest.raises(RuntimeError, match="injected fresh complete-load failure"):
        _run_refinement(root)

    assert list(parent.keys()) == ["prior"]
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs[mod.RUN_LATEST_COMPLETE_ATTR] == "prior"
    assert parent.attrs[mod.RUN_LATEST_PENDING_ATTR] == "older-pending"


def test_public_coordinate_loaders_cannot_read_staging_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    _, run = _run_refinement(root)

    with pytest.raises(TypeError, match="require_complete"):
        mod.load_bound_refined_online_coordinate_evidence(
            root,
            run,
            require_complete=False,
        )
    with pytest.raises(TypeError, match="require_complete"):
        mod.validate_refined_online_run(
            root,
            run,
            require_complete=False,
        )


def test_keyboard_interrupt_during_fresh_load_rolls_back_run_and_all_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    parent = root.create_group(mod.REFINED_ONLINE_GROUP)
    parent.create_group("prior")
    parent.attrs["latest"] = "prior"
    parent.attrs[mod.RUN_LATEST_COMPLETE_ATTR] = "prior-complete"
    parent.attrs[mod.RUN_LATEST_PENDING_ATTR] = "prior-pending"

    original_private_loader = mod._load_bound_refined_online_coordinate_evidence

    def interrupt_complete_load(*args, **kwargs):
        if kwargs.get("require_complete") is True:
            raise KeyboardInterrupt("injected fresh complete-load interruption")
        return original_private_loader(*args, **kwargs)

    monkeypatch.setattr(
        mod,
        "_load_bound_refined_online_coordinate_evidence",
        interrupt_complete_load,
    )

    with pytest.raises(
        KeyboardInterrupt,
        match="injected fresh complete-load interruption",
    ):
        _run_refinement(root)

    assert list(parent.keys()) == ["prior"]
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs[mod.RUN_LATEST_COMPLETE_ATTR] == "prior-complete"
    assert parent.attrs[mod.RUN_LATEST_PENDING_ATTR] == "prior-pending"


def test_interrupt_between_selector_updates_and_eligibility_restores_prior_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    parent = root.create_group(mod.REFINED_ONLINE_GROUP)
    parent.create_group("prior")
    parent.attrs["latest"] = "prior"
    parent.attrs[mod.RUN_LATEST_COMPLETE_ATTR] = "prior-complete"
    parent.attrs[mod.RUN_LATEST_PENDING_ATTR] = "prior-pending"

    def interrupt_activation(refined_runs, refined_group, *, run_name):
        assert refined_group.attrs["stage_selector_eligible"] is False
        refined_runs.attrs[mod.RUN_LATEST_COMPLETE_ATTR] = run_name
        refined_runs.attrs["latest"] = run_name
        raise SystemExit("injected pre-eligibility interruption")

    monkeypatch.setattr(
        mod,
        "_activate_refined_online_run",
        interrupt_activation,
    )

    with pytest.raises(SystemExit, match="injected pre-eligibility interruption"):
        _run_refinement(root)

    assert list(parent.keys()) == ["prior"]
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs[mod.RUN_LATEST_COMPLETE_ATTR] == "prior-complete"
    assert parent.attrs[mod.RUN_LATEST_PENDING_ATTR] == "prior-pending"


def test_run_name_collision_never_deletes_a_preexisting_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)

    class FixedDatetime:
        @classmethod
        def now(cls, tz=None):
            value = RealDatetime(2026, 7, 19, 12, 34, 56)
            return value if tz is None else value.replace(tzinfo=tz)

    monkeypatch.setattr(mod, "datetime", FixedDatetime)
    parent = root.create_group(mod.REFINED_ONLINE_GROUP)
    colliding_name = "refined_online_2026-07-19_12-34-56"
    existing = parent.create_group(colliding_name)
    existing.attrs["sentinel"] = "preserve"
    parent.attrs["latest"] = colliding_name

    with pytest.raises(Exception):
        _run_refinement(root)

    preserved = root[mod.REFINED_ONLINE_GROUP][colliding_name]
    assert preserved.attrs["sentinel"] == "preserve"
    assert parent.attrs["latest"] == colliding_name


def test_source_authority_tamper_fails_before_output_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    arena = root["analysis/stimulus_runs/stim_1/calibration/arena_geometry"]
    arena.attrs["arena_region_width_px"] = 999

    with pytest.raises(
        mod.CanonicalOnlineRefinementError,
        match="evidence is invalid|authority cannot be verified exactly",
    ):
        mod.refine_online_positions("ignored.zarr", console=_console())

    assert mod.REFINED_ONLINE_GROUP not in root


def test_completion_rejects_processing_lineage_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_fixture(monkeypatch)
    _, run = _run_refinement(root)

    processing = deepcopy(run.attrs[mod.PROCESSING_RECORD_ATTR])
    processing["parameters"]["max_gap"] += 1
    run.attrs[mod.PROCESSING_RECORD_ATTR] = processing
    run.attrs[mod.PROCESSING_RECORD_DIGEST_ATTR] = mod._canonical_mapping_digest(
        processing
    )

    with pytest.raises(
        mod.CanonicalOnlineRefinementError,
        match="processing record is stale or inconsistent",
    ):
        mod.load_bound_refined_online_coordinate_evidence(root, run)
