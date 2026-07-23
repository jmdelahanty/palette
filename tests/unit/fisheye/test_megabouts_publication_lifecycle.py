from __future__ import annotations

import uuid

import numpy as np
import pytest
import zarr

from fisheye.analysis import megabouts_classifier as module
from fisheye.analysis.megabouts_classifier import (
    BOUT_CLASSIFICATION_PARENT_PUBLICATION_LEASE_ATTR,
    BOUT_CLASSIFICATION_PUBLICATION_GENERATION_ATTR,
    BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR,
    BOUT_CLASSIFICATION_PUBLICATION_POLICY,
    BOUT_CLASSIFICATION_PUBLICATION_POLICY_ATTR,
    BOUT_CLASSIFICATION_PUBLICATION_TOMBSTONE_ATTR,
    MegaboutsClassificationResult,
    write_megabouts_classification_run,
)
from fisheye.analysis.megabouts_classifier_inputs import (
    MegaboutsClassifierInputPack,
)
from fisheye.shared import selector_activation as selector_activation_module


def _pack() -> MegaboutsClassifierInputPack:
    window_frames = 4
    return MegaboutsClassifierInputPack(
        tail_array=np.zeros((1, 10, window_frames), dtype=np.float32),
        traj_array=np.zeros((1, 3, window_frames), dtype=np.float32),
        tail_valid=np.ones((1, window_frames), dtype=bool),
        traj_valid=np.ones((1, window_frames), dtype=bool),
        traj_reference_valid=np.ones((1,), dtype=bool),
        source_bout_id=np.asarray([11], dtype=np.int64),
        source_start_frame=np.asarray([0], dtype=np.int64),
        source_end_frame=np.asarray([3], dtype=np.int64),
        window_start_frame=np.asarray([0], dtype=np.int64),
        window_end_frame=np.asarray([3], dtype=np.int64),
        tail_valid_fraction=np.asarray([1.0], dtype=np.float32),
        traj_valid_fraction=np.asarray([1.0], dtype=np.float32),
        max_consecutive_tail_invalid=np.asarray([0], dtype=np.int32),
        max_consecutive_traj_invalid=np.asarray([0], dtype=np.int32),
        valid_bout=np.asarray([True], dtype=bool),
        failure_reason=np.asarray(["ok"], dtype=object),
        source_refs={
            "tail_angle_rad": "analysis/tail_posture_view_runs/posture/tail_angle_rad",
            "tail_valid": "analysis/tail_posture_view_runs/posture/valid",
            "positions_mm": "analysis/track_kinematics_runs/offline/tk/tracks/id_0/positions_mm",
            "heading": "analysis/track_kinematics_runs/offline/tk/tracks/id_0/smoothed_heading_radians",
            "sample_valid": "analysis/track_kinematics_runs/offline/tk/tracks/id_0/sample_valid",
            "swim_bout_level": "analysis/swim_bout_runs/bouts/speed_filtered",
        },
        parameters={
            "fps": 60.0,
            "bout_duration_s": window_frames / 60.0,
            "classifier_input_mode": module.PALETTE_PREPARED_INPUT_MODE,
            "megabouts_preprocessing": False,
            "megabouts_segmentation": False,
            "traj_alignment": "onset_translation_rotation",
            "traj_reference_index": 0,
            "min_tail_valid_fraction": 0.9,
            "min_traj_valid_fraction": 0.9,
            "max_consecutive_invalid_frames": 1,
            "requires_traj_reference_valid": True,
        },
    )


def _result() -> MegaboutsClassificationResult:
    return MegaboutsClassificationResult(
        classified_indices=np.asarray([0], dtype=np.int64),
        classif_results={
            "cat": np.asarray([2], dtype=np.int32),
            "subcat": np.asarray([12], dtype=np.int32),
            "sign": np.asarray([-1], dtype=np.int32),
            "proba": np.asarray([0.875], dtype=np.float32),
            "first_half_beat": np.asarray([1], dtype=np.int32),
        },
        runtime=None,
    )


def _write(root: zarr.Group, run_name: str) -> str:
    return write_megabouts_classification_run(
        root,
        run_name=run_name,
        pack=_pack(),
        result=_result(),
    )


def _assert_prior_remains_selected(parent: zarr.Group) -> None:
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs["latest_complete"] == "prior"
    assert parent.attrs[BOUT_CLASSIFICATION_PUBLICATION_GENERATION_ATTR] == 1


def _assert_failed_tombstone(
    parent: zarr.Group,
    run_name: str,
    *,
    failure_text: str,
) -> zarr.Group:
    failed = parent[run_name]
    owner = str(
        uuid.UUID(failed.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR])
    )
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert "palette_run_completed_at_utc" not in failed.attrs
    tombstone = failed.attrs[BOUT_CLASSIFICATION_PUBLICATION_TOMBSTONE_ATTR]
    assert set(tombstone) == {
        "schema_id",
        "schema_version",
        "failed_at_utc",
        "publication_owner_uuid",
        "run_name",
        "run_path",
        "public_path_retained",
        "selector_eligible",
        "retry_policy",
        "failure_type",
        "failure",
    }
    assert tombstone == {
        "schema_id": "palette.bout_classification_publication_tombstone",
        "schema_version": 1,
        "failed_at_utc": failed.attrs["palette_run_failed_at_utc"],
        "publication_owner_uuid": owner,
        "run_name": run_name,
        "run_path": f"analysis/bout_classification_runs/{run_name}",
        "public_path_retained": True,
        "selector_eligible": False,
        "retry_policy": "new_immutable_run_name_required",
        "failure_type": "RuntimeError",
        "failure": failure_text,
    }
    assert parent.attrs.get("latest") != run_name
    assert parent.attrs.get("latest_complete") != run_name
    return failed


def test_successful_publication_is_owner_guarded_and_selector_eligible() -> None:
    root = zarr.group()

    assert _write(root, "classification_001") == "classification_001"

    parent = root["analysis/bout_classification_runs"]
    run = parent["classification_001"]
    owner = str(uuid.UUID(run.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR]))
    assert run.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR] == owner
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is True
    assert parent.attrs["latest"] == "classification_001"
    assert parent.attrs["latest_complete"] == "classification_001"
    assert parent.attrs[BOUT_CLASSIFICATION_PUBLICATION_GENERATION_ATTR] == 1
    assert (
        parent.attrs[BOUT_CLASSIFICATION_PUBLICATION_POLICY_ATTR]
        == BOUT_CLASSIFICATION_PUBLICATION_POLICY
    )
    lease = parent.attrs[BOUT_CLASSIFICATION_PARENT_PUBLICATION_LEASE_ATTR]
    assert lease["publication_owner"] == owner
    assert lease["run_name"] == "classification_001"


def test_successor_advances_generation_without_mutating_prior_run() -> None:
    root = zarr.group()
    _write(root, "classification_001")
    prior = root["analysis/bout_classification_runs/classification_001"]
    prior_owner = prior.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR]

    _write(root, "classification_002")

    parent = root["analysis/bout_classification_runs"]
    assert parent.attrs["latest"] == "classification_002"
    assert parent.attrs["latest_complete"] == "classification_002"
    assert parent.attrs[BOUT_CLASSIFICATION_PUBLICATION_GENERATION_ATTR] == 2
    assert prior.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR] == prior_owner
    assert prior.attrs["stage_selector_eligible"] is True
    assert prior.attrs["palette_run_completion_status"] == "complete"


def test_overwrite_never_deletes_or_reuses_a_public_run_name() -> None:
    root = zarr.group()
    _write(root, "classification_001")
    run = root["analysis/bout_classification_runs/classification_001"]
    owner = run.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR]
    category_before = np.asarray(run["per_bout/category_id"][:]).copy()

    with pytest.raises(ValueError, match="cannot replace an immutable public run"):
        write_megabouts_classification_run(
            root,
            run_name="classification_001",
            pack=_pack(),
            result=_result(),
            overwrite=True,
        )

    same_run = root["analysis/bout_classification_runs/classification_001"]
    assert same_run.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR] == owner
    assert same_run.attrs["stage_selector_eligible"] is True
    np.testing.assert_array_equal(
        same_run["per_bout/category_id"][:],
        category_before,
    )


def test_payload_failure_retains_owned_ineligible_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    _write(root, "prior")
    parent = root["analysis/bout_classification_runs"]

    def _fail_payload(*_args, **_kwargs):
        candidate = root["analysis/bout_classification_runs/failed"]
        uuid.UUID(candidate.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR])
        assert candidate.attrs["stage_selector_eligible"] is False
        assert candidate.attrs["palette_run_completion_status"] == "running"
        raise RuntimeError("injected payload failure")

    monkeypatch.setattr(
        module,
        "_populate_megabouts_classification_run",
        _fail_payload,
    )

    with pytest.raises(RuntimeError, match="injected payload failure"):
        _write(root, "failed")

    failed = _assert_failed_tombstone(
        parent,
        "failed",
        failure_text="injected payload failure",
    )
    failed_owner = failed.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR]
    _assert_prior_remains_selected(parent)
    with pytest.raises(ValueError, match="immutable public run"):
        write_megabouts_classification_run(
            root,
            run_name="failed",
            pack=_pack(),
            result=_result(),
            overwrite=True,
        )
    assert parent["failed"].attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR] == failed_owner


def test_activation_failure_retains_complete_payload_as_failed_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    _write(root, "prior")
    parent = root["analysis/bout_classification_runs"]

    def _fail_activation(*_args, **_kwargs):
        raise RuntimeError("injected activation failure")

    monkeypatch.setattr(
        module,
        "_activate_megabouts_classification_run",
        _fail_activation,
    )

    with pytest.raises(RuntimeError, match="injected activation failure"):
        _write(root, "failed_activation")

    failed = _assert_failed_tombstone(
        parent,
        "failed_activation",
        failure_text="injected activation failure",
    )
    assert "per_bout" in failed
    _assert_prior_remains_selected(parent)


def test_owner_takeover_cannot_publish_under_a_replacement_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    _write(root, "prior")
    parent = root["analysis/bout_classification_runs"]
    original_populate = module._populate_megabouts_classification_run
    replacement_owner = str(uuid.uuid4())

    def _populate_then_take_owner(*args, **kwargs):
        result = original_populate(*args, **kwargs)
        kwargs["run_group"].attrs[
            BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR
        ] = replacement_owner
        return result

    monkeypatch.setattr(
        module,
        "_populate_megabouts_classification_run",
        _populate_then_take_owner,
    )

    with pytest.raises(RuntimeError, match="failure cleanup was incomplete"):
        _write(root, "owner_takeover")

    candidate = parent["owner_takeover"]
    assert candidate.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR] == (
        replacement_owner
    )
    assert candidate.attrs["stage_selector_eligible"] is False
    _assert_prior_remains_selected(parent)


def test_delete_recreate_before_activation_rejects_foreign_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    _write(root, "prior")
    parent = root["analysis/bout_classification_runs"]
    original_activate = module._activate_megabouts_classification_run
    replacement_owner = str(uuid.uuid4())

    def _replace_then_activate(
        root,
        parent,
        stale_run,
        *,
        run_name,
        expected_publication_owner_uuid,
    ):
        assert stale_run.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR] == (
            expected_publication_owner_uuid
        )
        del parent[run_name]
        replacement = parent.create_group(
            run_name,
            attributes={
                BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR: replacement_owner,
                "palette_run_completion_status": "complete",
                "stage_selector_eligible": False,
                "sentinel": "foreign-successor",
            },
        )
        original_activate(
            root,
            parent,
            stale_run,
            run_name=run_name,
            expected_publication_owner_uuid=expected_publication_owner_uuid,
        )
        return replacement

    monkeypatch.setattr(
        module,
        "_activate_megabouts_classification_run",
        _replace_then_activate,
    )

    with pytest.raises(RuntimeError, match="failure cleanup was incomplete") as exc:
        _write(root, "pre_activation_replacement")

    assert "expected publication owner" in str(exc.value.__cause__)
    successor = parent["pre_activation_replacement"]
    assert successor.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR] == (
        replacement_owner
    )
    assert successor.attrs["stage_selector_eligible"] is False
    assert successor.attrs["sentinel"] == "foreign-successor"
    _assert_prior_remains_selected(parent)


def test_persist_then_raise_creation_is_recovered_as_an_exact_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    _write(root, "prior")
    parent = root["analysis/bout_classification_runs"]
    original_create_group = zarr.Group.create_group

    def _persist_then_raise(self, name, *args, **kwargs):
        child = original_create_group(self, name, *args, **kwargs)
        if (
            getattr(self, "path", None) == "analysis/bout_classification_runs"
            and name == "persisted_then_failed"
        ):
            raise RuntimeError("injected create acknowledgement failure")
        return child

    monkeypatch.setattr(zarr.Group, "create_group", _persist_then_raise)

    with pytest.raises(
        RuntimeError,
        match="injected create acknowledgement failure",
    ):
        _write(root, "persisted_then_failed")

    failed = _assert_failed_tombstone(
        parent,
        "persisted_then_failed",
        failure_text="injected create acknowledgement failure",
    )
    assert not tuple(failed.array_keys())
    assert not tuple(failed.group_keys())
    _assert_prior_remains_selected(parent)


def test_field_array_attr_mutation_invalidates_activation_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    _write(root, "prior")
    parent = root["analysis/bout_classification_runs"]
    real_activate = selector_activation_module.activate_selector_eligible_run
    real_write_attr = selector_activation_module.write_activation_attr
    state = {"mutated": False}

    def _activate_with_array_attr_mutation(*args, **kwargs):
        candidate = args[2]
        lease_attr = kwargs["lease_attr"]

        def _hostile_writer(attrs, name, value):
            real_write_attr(attrs, name, value)
            if name == lease_attr and not state["mutated"]:
                state["mutated"] = True
                candidate["per_bout/category_id"].attrs[
                    "adversarial_mutation"
                ] = "changed"

        return real_activate(*args, **kwargs, attr_writer=_hostile_writer)

    monkeypatch.setattr(
        module,
        "activate_selector_eligible_run",
        _activate_with_array_attr_mutation,
    )

    with pytest.raises(RuntimeError, match="activation lost exact ownership"):
        _write(root, "proof_mutation")

    assert state["mutated"] is True
    failed = _assert_failed_tombstone(
        parent,
        "proof_mutation",
        failure_text=(
            "Bout-classification activation lost exact ownership: Candidate "
            "publication changed after lease acquisition.."
        ),
    )
    assert (
        failed["per_bout/category_id"].attrs["adversarial_mutation"]
        == "changed"
    )
    _assert_prior_remains_selected(parent)
