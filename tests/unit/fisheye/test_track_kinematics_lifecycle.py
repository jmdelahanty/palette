from __future__ import annotations

import copy
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import zarr

from fisheye.analysis import track_kinematics as mod
from fisheye.analysis.chaser_metrics_loader import ChaserMetricsBundle


def _new_run(tmp_path, *, name: str = "candidate"):
    root = zarr.open_group(
        str(tmp_path / f"{name}.zarr"),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    run_name, run = mod.ensure_track_kinematics_run_group(
        root,
        name,
        run_type="offline",
    )
    parent = root["analysis/track_kinematics_runs"]
    # This unit test exercises selection ordering, not provenance enforcement.
    parent.attrs["palette_completion_epoch"] = 1
    return root, parent, parent["offline"], run_name, run


def test_track_completion_validates_complete_while_ineligible_then_selects_last(
    monkeypatch,
    tmp_path,
) -> None:
    root, parent, offline, run_name, run = _new_run(tmp_path)
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
        }
    )
    offline.attrs["latest"] = "previous"
    observed: dict[str, object] = {}
    order: list[str] = []

    def seal_motion(
        authoritative_root,
        sealed_run,
        *,
        expected_publication_owner_uuid,
    ):
        assert authoritative_root is root
        assert sealed_run.path == run.path
        assert sealed_run.attrs["palette_run_completion_status"] == "complete"
        assert sealed_run.attrs["stage_selector_eligible"] is False
        assert expected_publication_owner_uuid == sealed_run.attrs[
            mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
        ]
        assert parent.attrs["latest"] == "offline/previous"
        order.append("seal")
        return SimpleNamespace(
            tracks=(object(),),
            assert_verified=lambda: order.append("verify"),
        )

    monkeypatch.setattr(
        mod,
        "_seal_and_load_track_motion_run_before_selection",
        seal_motion,
    )

    def validate_complete(fresh_run):
        assert fresh_run.path == run.path
        order.append("validate")
        observed.update(
            {
                "status": fresh_run.attrs["palette_run_completion_status"],
                "eligible": fresh_run.attrs["stage_selector_eligible"],
                "latest": parent.attrs["latest"],
                "latest_complete": parent.attrs["latest_complete"],
                "latest_offline": parent.attrs["latest_offline"],
                "type_latest": offline.attrs["latest"],
            }
        )
        return {"valid": True}

    mod.mark_track_kinematics_run_complete(
        root,
        run,
        run_name=run_name,
        run_type="offline",
        publication_owner_uuid=run.attrs[
            mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
        ],
        validate_complete_run=validate_complete,
    )

    assert observed == {
        "status": "complete",
        "eligible": False,
        "latest": "offline/previous",
        "latest_complete": "offline/previous",
        "latest_offline": "previous",
        "type_latest": "previous",
    }
    assert order == [
        "seal",
        "validate",
        "verify",
        "verify",
        "validate",
        "seal",
        "verify",
    ]
    fresh_parent = root["analysis/track_kinematics_runs"]
    fresh_offline = fresh_parent["offline"]
    fresh_run = fresh_offline[run_name]
    assert fresh_parent.attrs["latest"] == f"offline/{run_name}"
    assert fresh_parent.attrs["latest_complete"] == f"offline/{run_name}"
    assert fresh_parent.attrs["latest_offline"] == run_name
    assert fresh_offline.attrs["latest"] == run_name
    assert fresh_run.attrs["stage_selector_eligible"] is True
    assert str(
        uuid.UUID(
            fresh_run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]
        )
    ) == fresh_run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]
    selector_owner = fresh_parent.attrs[
        mod.TRACK_KINEMATICS_SELECTOR_OWNER_ATTR
    ]
    assert selector_owner["owner_uuid"] == fresh_run.attrs[
        mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
    ]


def test_deferred_track_commit_rebinds_and_preserves_post_receipt_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root, _parent, _offline, run_name, run = _new_run(
        tmp_path,
        name="deferred_rebind",
    )
    owner = run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]

    def seal_motion(
        authoritative_root,
        sealed_run,
        *,
        expected_publication_owner_uuid,
    ):
        assert expected_publication_owner_uuid == owner
        assert sealed_run.attrs["stage_selector_eligible"] is False
        assert sealed_run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR] == (
            owner
        )
        assert authoritative_root.attrs is not None
        return SimpleNamespace(
            tracks=(object(),),
            manifest_sha256="a" * 64,
            assert_verified=lambda: None,
        )

    monkeypatch.setattr(
        mod,
        "_seal_and_load_track_motion_run_before_selection",
        seal_motion,
    )
    activation = mod.mark_track_kinematics_run_complete(
        root,
        run,
        run_name=run_name,
        run_type="offline",
        publication_owner_uuid=owner,
        validate_complete_run=lambda _fresh_run: {"valid": True},
        defer_selector_eligibility=True,
    )
    assert callable(activation)

    fresh_root = zarr.open_group(
        str(tmp_path / "deferred_rebind.zarr"),
        mode="a",
        zarr_format=3,
        use_consolidated=False,
    )
    fresh_parent = fresh_root["analysis/track_kinematics_runs/offline"]
    fresh_run = fresh_parent[run_name]
    final_payload = {"final_validation": {"valid": True}, "sentinel": "keep"}
    fresh_run.attrs["cluster_output_staging"] = final_payload

    activation(
        fresh_root,
        fresh_parent,
        fresh_run,
        validate_fresh_complete_run=lambda _fresh_run: {"valid": True},
        expected_cluster_output_staging=final_payload,
    )

    reloaded = zarr.open_group(
        str(tmp_path / "deferred_rebind.zarr"),
        mode="r",
        zarr_format=3,
        use_consolidated=False,
    )[f"analysis/track_kinematics_runs/offline/{run_name}"]
    assert reloaded.attrs["stage_selector_eligible"] is True
    assert reloaded.attrs["cluster_output_staging"] == final_payload


def test_track_completion_keyboard_interrupt_restores_exact_selectors(
    monkeypatch,
    tmp_path,
) -> None:
    root, parent, offline, run_name, run = _new_run(tmp_path)
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
            "latest_pending": "unrelated_pending",
        }
    )
    offline.attrs["latest"] = "previous"
    parent_snapshot = copy.deepcopy(dict(parent.attrs))
    offline_snapshot = copy.deepcopy(dict(offline.attrs))

    def interrupt_seal(
        authoritative_root,
        sealed_run,
        *,
        expected_publication_owner_uuid,
    ):
        assert authoritative_root is root
        assert sealed_run.path == run.path
        assert sealed_run.attrs["palette_run_completion_status"] == "complete"
        assert sealed_run.attrs["stage_selector_eligible"] is False
        assert parent.attrs["latest"] == "offline/previous"
        assert expected_publication_owner_uuid == sealed_run.attrs[
            mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
        ]
        raise KeyboardInterrupt("injected motion-seal interrupt")

    monkeypatch.setattr(
        mod,
        "_seal_and_load_track_motion_run_before_selection",
        interrupt_seal,
    )

    with pytest.raises(KeyboardInterrupt, match="injected motion-seal interrupt"):
        mod.mark_track_kinematics_run_complete(
            root,
            run,
            run_name=run_name,
            run_type="offline",
            publication_owner_uuid=run.attrs[
                mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
            ],
            validate_complete_run=lambda _fresh_run: pytest.fail(
                "validation callback must not run after motion-seal interrupt"
            ),
        )

    fresh_parent = root["analysis/track_kinematics_runs"]
    fresh_offline = fresh_parent["offline"]
    fresh_run = fresh_offline[run_name]
    assert dict(fresh_parent.attrs) == parent_snapshot
    assert dict(fresh_offline.attrs) == offline_snapshot
    assert fresh_run.attrs["palette_run_completion_status"] == "failed"
    assert fresh_run.attrs["stage_selector_eligible"] is False


def test_track_completion_does_not_mutate_replacement_child(
    monkeypatch,
    tmp_path,
) -> None:
    root, parent, offline, run_name, run = _new_run(
        tmp_path,
        name="replacement",
    )
    original_owner = run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
        }
    )
    offline.attrs["latest"] = "previous"
    sealed = SimpleNamespace(tracks=(object(),), assert_verified=lambda: None)
    monkeypatch.setattr(
        mod,
        "_seal_and_load_track_motion_run_before_selection",
        lambda _root, _run, **_kwargs: sealed,
    )
    replacement_owner = str(uuid.uuid4())

    def replace_then_validate(fresh_run):
        assert fresh_run.attrs[
            mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
        ] == original_owner
        del offline[run_name]
        replacement = offline.create_group(
            run_name,
            attributes={
                mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR: replacement_owner,
                "stage_selector_eligible": False,
                "palette_run_completion_status": "running",
            },
        )
        assert replacement.attrs[
            mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
        ] != original_owner
        return {"valid": True}

    with pytest.raises(RuntimeError, match="replaced|ownership"):
        mod.mark_track_kinematics_run_complete(
            root,
            run,
            run_name=run_name,
            run_type="offline",
            publication_owner_uuid=original_owner,
            validate_complete_run=replace_then_validate,
        )

    replacement = root[
        f"analysis/track_kinematics_runs/offline/{run_name}"
    ]
    assert replacement.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR] == (
        replacement_owner
    )
    assert replacement.attrs["palette_run_completion_status"] == "running"
    assert replacement.attrs["stage_selector_eligible"] is False
    assert parent.attrs["latest"] == "offline/previous"
    assert offline.attrs["latest"] == "previous"


def test_track_failure_preserves_concurrent_selector_publication(
    monkeypatch,
    tmp_path,
) -> None:
    root, parent, offline, run_name, run = _new_run(
        tmp_path,
        name="concurrent",
    )
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
        }
    )
    offline.attrs["latest"] = "previous"
    monkeypatch.setattr(
        mod,
        "_seal_and_load_track_motion_run_before_selection",
        lambda _root, _run, **_kwargs: SimpleNamespace(
            tracks=(object(),),
            assert_verified=lambda: None,
        ),
    )

    def concurrent_then_interrupt(_fresh_run):
        parent.attrs["latest"] = "offline/concurrent_winner"
        parent.attrs["latest_complete"] = "offline/concurrent_winner"
        parent.attrs["latest_offline"] = "concurrent_winner"
        offline.attrs["latest"] = "concurrent_winner"
        raise KeyboardInterrupt("concurrent publication won")

    with pytest.raises(KeyboardInterrupt, match="concurrent publication won"):
        mod.mark_track_kinematics_run_complete(
            root,
            run,
            run_name=run_name,
            run_type="offline",
            publication_owner_uuid=run.attrs[
                mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
            ],
            validate_complete_run=concurrent_then_interrupt,
        )

    assert parent.attrs["latest"] == "offline/concurrent_winner"
    assert parent.attrs["latest_complete"] == "offline/concurrent_winner"
    assert parent.attrs["latest_offline"] == "concurrent_winner"
    assert offline.attrs["latest"] == "concurrent_winner"
    failed = offline[run_name]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False


def test_track_interrupt_after_pointer_writes_restores_owned_selectors(
    monkeypatch,
    tmp_path,
) -> None:
    root, parent, offline, run_name, run = _new_run(
        tmp_path,
        name="post_pointer_interrupt",
    )
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
            "latest_pending": f"offline/{run_name}",
        }
    )
    offline.attrs["latest"] = "previous"
    parent_snapshot = copy.deepcopy(dict(parent.attrs))
    offline_snapshot = copy.deepcopy(dict(offline.attrs))
    verification_count = 0

    def verify() -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            fresh_parent = root["analysis/track_kinematics_runs"]
            fresh_offline = fresh_parent["offline"]
            assert fresh_parent.attrs["latest"] == f"offline/{run_name}"
            assert fresh_offline.attrs["latest"] == run_name
            raise KeyboardInterrupt("interrupt after track pointer writes")

    monkeypatch.setattr(
        mod,
        "_seal_and_load_track_motion_run_before_selection",
        lambda _root, _run, **_kwargs: SimpleNamespace(
            tracks=(object(),),
            assert_verified=verify,
        ),
    )
    owner = run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]

    with pytest.raises(KeyboardInterrupt, match="after track pointer writes"):
        mod.mark_track_kinematics_run_complete(
            root,
            run,
            run_name=run_name,
            run_type="offline",
            publication_owner_uuid=owner,
            validate_complete_run=lambda _fresh_run: {"valid": True},
        )

    assert dict(parent.attrs) == parent_snapshot
    assert dict(offline.attrs) == offline_snapshot
    assert offline[run_name].attrs["palette_run_completion_status"] == "failed"
    assert offline[run_name].attrs["stage_selector_eligible"] is False


def test_track_selector_owner_takeover_prevents_stale_pointer_rollback(
    monkeypatch,
    tmp_path,
) -> None:
    root, parent, offline, run_name, run = _new_run(
        tmp_path,
        name="selector_owner_takeover",
    )
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
        }
    )
    offline.attrs["latest"] = "previous"
    verification_count = 0
    winner_owner = str(uuid.uuid4())
    winner = "concurrent_winner"

    def verify() -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            fresh_parent = root["analysis/track_kinematics_runs"]
            fresh_offline = fresh_parent["offline"]
            fresh_parent.attrs[mod.TRACK_KINEMATICS_SELECTOR_OWNER_ATTR] = (
                mod._track_selector_owner_record(
                    owner_uuid=winner_owner,
                    qualified_name=f"offline/{winner}",
                )
            )
            fresh_parent.attrs["latest"] = f"offline/{winner}"
            fresh_parent.attrs["latest_complete"] = f"offline/{winner}"
            fresh_parent.attrs["latest_offline"] = winner
            fresh_offline.attrs["latest"] = winner
            raise KeyboardInterrupt("concurrent selector owner took over")

    monkeypatch.setattr(
        mod,
        "_seal_and_load_track_motion_run_before_selection",
        lambda _root, _run, **_kwargs: SimpleNamespace(
            tracks=(object(),),
            assert_verified=verify,
        ),
    )
    owner = run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]

    with pytest.raises(KeyboardInterrupt, match="selector owner took over"):
        mod.mark_track_kinematics_run_complete(
            root,
            run,
            run_name=run_name,
            run_type="offline",
            publication_owner_uuid=owner,
            validate_complete_run=lambda _fresh_run: {"valid": True},
        )

    fresh_parent = root["analysis/track_kinematics_runs"]
    fresh_offline = fresh_parent["offline"]
    assert fresh_parent.attrs["latest"] == f"offline/{winner}"
    assert fresh_parent.attrs["latest_complete"] == f"offline/{winner}"
    assert fresh_parent.attrs["latest_offline"] == winner
    assert fresh_offline.attrs["latest"] == winner
    assert fresh_parent.attrs[mod.TRACK_KINEMATICS_SELECTOR_OWNER_ATTR][
        "owner_uuid"
    ] == winner_owner


def test_deferred_track_rollback_stops_on_mid_rollback_owner_takeover(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root, parent, offline, run_name, run = _new_run(
        tmp_path,
        name="rollback_owner_takeover",
    )
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
        }
    )
    offline.attrs["latest"] = "previous"
    monkeypatch.setattr(
        mod,
        "_seal_and_load_track_motion_run_before_selection",
        lambda _root, _run, **_kwargs: SimpleNamespace(
            tracks=(object(),),
            manifest_sha256="a" * 64,
            assert_verified=lambda: None,
        ),
    )
    owner = run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]
    activation = mod.mark_track_kinematics_run_complete(
        root,
        run,
        run_name=run_name,
        run_type="offline",
        publication_owner_uuid=owner,
        validate_complete_run=lambda _fresh_run: {"valid": True},
        defer_selector_eligibility=True,
    )
    assert isinstance(
        activation,
        mod.DeferredTrackKinematicsSelectorActivation,
    )

    original_restore = mod._restore_track_selector_value
    winner = "winner_after_rollback_started"
    winner_owner = str(uuid.uuid4())
    restore_calls = 0

    def takeover_after_first_restore(attrs, name, previous):
        nonlocal restore_calls
        original_restore(attrs, name, previous)
        restore_calls += 1
        if restore_calls != 1:
            return
        fresh_parent = root["analysis/track_kinematics_runs"]
        fresh_offline = fresh_parent["offline"]
        fresh_parent.attrs.update(
            {
                mod.TRACK_KINEMATICS_SELECTOR_OWNER_ATTR: (
                    mod._track_selector_owner_record(
                        owner_uuid=winner_owner,
                        qualified_name=f"offline/{winner}",
                    )
                ),
                "latest": f"offline/{winner}",
                "latest_complete": f"offline/{winner}",
                "latest_offline": winner,
            }
        )
        fresh_offline.attrs["latest"] = winner

    monkeypatch.setattr(
        mod,
        "_restore_track_selector_value",
        takeover_after_first_restore,
    )
    mod.rollback_deferred_track_kinematics_selector_activation(activation)

    fresh_parent = root["analysis/track_kinematics_runs"]
    fresh_offline = fresh_parent["offline"]
    assert restore_calls == 1
    assert fresh_parent.attrs[mod.TRACK_KINEMATICS_SELECTOR_OWNER_ATTR][
        "owner_uuid"
    ] == winner_owner
    assert fresh_parent.attrs["latest"] == f"offline/{winner}"
    assert fresh_parent.attrs["latest_complete"] == f"offline/{winner}"
    assert fresh_parent.attrs["latest_offline"] == winner
    assert fresh_offline.attrs["latest"] == winner


def test_track_overwrite_rejects_complete_and_selected_runs(tmp_path) -> None:
    root, parent, _offline, run_name, run = _new_run(tmp_path, name="complete")
    run.attrs["palette_run_completion_status"] = "complete"
    with pytest.raises(ValueError, match="complete or selected"):
        mod.ensure_track_kinematics_run_group(
            root,
            run_name,
            run_type="offline",
            overwrite=True,
        )

    root2, parent2, _offline2, selected_name, selected = _new_run(
        tmp_path,
        name="selected",
    )
    selected.attrs["palette_run_completion_status"] = "failed"
    parent2.attrs["latest_offline"] = selected_name
    with pytest.raises(ValueError, match="complete or selected"):
        mod.ensure_track_kinematics_run_group(
            root2,
            selected_name,
            run_type="offline",
            overwrite=True,
        )


def test_track_overwrite_never_reuses_failed_public_child(tmp_path) -> None:
    root, _parent, offline, run_name, run = _new_run(
        tmp_path,
        name="failed_public_tombstone",
    )
    run.attrs.update(
        {
            "palette_run_completion_status": "failed",
            "stage_selector_eligible": False,
            "sentinel": "preserve exact failed child",
        }
    )
    owner = run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]

    with pytest.raises(ValueError, match="immutable tombstones"):
        mod.ensure_track_kinematics_run_group(
            root,
            run_name,
            run_type="offline",
            overwrite=True,
        )

    preserved = offline[run_name]
    assert preserved.attrs["sentinel"] == "preserve exact failed child"
    assert preserved.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR] == owner


def test_track_creation_persist_then_raise_retains_owned_failed_tombstone(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(
        str(tmp_path / "create-ack-failure.zarr"),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    real_create_group = zarr.Group.create_group

    def persist_then_raise(group, name, *args, **kwargs):
        created = real_create_group(group, name, *args, **kwargs)
        if (
            group.path == "analysis/track_kinematics_runs/offline"
            and name == "ack_failed"
        ):
            raise RuntimeError("injected create acknowledgement failure")
        return created

    monkeypatch.setattr(zarr.Group, "create_group", persist_then_raise)

    with pytest.raises(RuntimeError, match="create acknowledgement failure"):
        mod.ensure_track_kinematics_run_group(
            root,
            "ack_failed",
            run_type="offline",
        )

    failed = root["analysis/track_kinematics_runs/offline/ack_failed"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR]
    tombstone = failed.attrs[
        mod.TRACK_KINEMATICS_PUBLICATION_TOMBSTONE_ATTR
    ]
    assert tombstone["public_path_retained"] is True
    assert tombstone["retry_policy"] == "new_immutable_run_name_required"


def test_legacy_chaser_geometry_is_omitted_from_track_run_root(tmp_path) -> None:
    root = zarr.open_group(
        str(tmp_path / "chaser-omission.zarr"),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    run = root.create_group("run")
    bundle = ChaserMetricsBundle(
        camera_frame_ids=np.asarray([10, 11], dtype=np.int64),
        stimulus_frame_nums=np.asarray([0, 1], dtype=np.int64),
        timestamp_ns=np.asarray([100, 200], dtype=np.int64),
        trial_state=np.asarray([1, 1], dtype=np.int16),
        metadata_mask=None,
        online={},
        offline={
            "distance_px": np.asarray([2.0, 3.0], dtype=np.float64),
            "distance_mm": np.asarray([0.2, 0.3], dtype=np.float64),
            "fish_centroid_px": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "chaser_position_px": np.asarray([[5.0, 6.0], [7.0, 8.0]]),
            "angle_unsigned_deg": np.asarray([20.0, 30.0]),
            "has_offline": np.asarray([True, True]),
        },
        provenance={"metrics_run": "legacy", "stimulus_run": "stim", "chaser_index": 0},
    )

    metadata = mod._persist_chaser_metrics_to_run(
        run,
        bundle,
        fps=100.0,
        smooth_seconds=0.05,
        distance_interp_seconds=0.2,
    )

    assert metadata["coordinate_geometry_status"] == (
        "omitted_untyped_legacy_chaser_metrics_v1"
    )
    assert metadata["omitted_coordinate_fields"] == [
        "chaser_position_px",
        "distance_mm",
        "distance_px",
        "fish_centroid_px",
    ]
    assert "angle_unsigned_deg" in run
    assert "has_offline" in run
    assert not any(
        name.endswith("_px") or name.endswith("_mm")
        for name in run.array_keys()
    )
    mod._validate_no_run_root_coordinate_arrays(run)

    run.create_array(
        "distance_to_target_mm",
        data=np.asarray([0.2, 0.3], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="unsupported untyped"):
        mod._validate_no_run_root_coordinate_arrays(run)
