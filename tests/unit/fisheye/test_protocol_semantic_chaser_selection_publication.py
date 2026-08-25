from __future__ import annotations

import copy
from pathlib import Path

import pytest
import zarr

from fisheye.analysis_workflows.composable_epoch_selection_adapter import (
    EpochRoleBinding,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
    STEP_END_INCLUSIVE,
    STANDALONE_SOLID_BLACK_ROLE,
    compile_protocol_semantic_chaser_selections,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    RUNS_PARENT_PATH,
    ProtocolSemanticChaserSelectionPublicationError,
    ProtocolSemanticChaserSelectionSourceHandle,
    build_protocol_semantic_chaser_selection_publication_plan,
    load_protocol_semantic_chaser_selection_source_handle,
    publish_protocol_semantic_chaser_selection_run,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.utils.materialize_provider_chaser_position_suite_canary import (
    ProviderChaserPositionSuiteCanaryError,
    _epoch_specs,
)
from fisheye.utils.materialize_protocol_semantic_chaser_selection import (
    plan_protocol_semantic_chaser_selection_run,
)
from tests.unit.fisheye.test_protocol_semantic_chaser_selection import (
    _chaser_bindings,
    _evidence,
    _semantic_selection,
)


def _plan(
    tmp_path: Path,
    *,
    two_step: bool = False,
    include_standalone_window: bool = False,
    run_name: str = "semantic-selection-v1",
):
    if two_step:
        archive, selection, evidence = _semantic_selection(
            tmp_path,
            modes=("SOLID_BLACK", "CHASER"),
            step_bounds=((0, 9), (10, 39)),
            include_standalone_window=include_standalone_window,
        )
    else:
        archive, selection, evidence = _semantic_selection(tmp_path)
    offset = 1 if include_standalone_window else 0
    bindings = _chaser_bindings(selection, offset=offset)
    if include_standalone_window:
        bindings[STANDALONE_SOLID_BLACK_ROLE] = EpochRoleBinding.by_window_id(
            selection.intervals[0].window_id
        )
    selections = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        protocol_evidence=evidence,
        role_bindings=bindings,
    )
    plan = build_protocol_semantic_chaser_selection_publication_plan(
        archive,
        selections=selections,
        source_selection=selection,
        run_name=run_name,
    )
    return archive, selection, selections, plan


def test_dry_run_binds_exact_hierarchy_without_writing(tmp_path: Path) -> None:
    archive, _selection, selections, plan = _plan(tmp_path)

    record = plan.to_json()

    assert record["status"] == "dry_run_plan"
    assert record["roles"] == list(CHASER_WINDOW_ROLES)
    assert record["role_count"] == 3
    assert record["selector_eligible"] is False
    assert record["production_authority"] is False
    assert record["registry_update"] is False
    assert plan.manifest["selection_identity_sha256"] == (
        selections.identity_sha256
    )
    assert not (archive / plan.run_path).exists()


def test_operator_plan_loads_all_authorities_from_exact_run_names(
    tmp_path: Path,
) -> None:
    archive, selection, _selections, _plan_value = _plan(tmp_path)

    plan = plan_protocol_semantic_chaser_selection_run(
        archive,
        source_epoch_selection_run=selection.run_name,
        expected_source_epoch_manifest_sha256=selection.run_manifest_digest,
        run_name="semantic-selection-operator-v1",
        chaser_pre_window_id=0,
        chaser_training_window_id=1,
        chaser_post_window_id=2,
    )

    assert plan.to_json()["roles"] == list(CHASER_WINDOW_ROLES)
    assert plan.manifest["source_epoch_selection"][
        "selection_sha256"
    ] == selection.selection_digest
    assert plan.to_json()["target_exists"] is False


def test_publication_rejects_unproven_producer_declared_end_policy(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(
        tmp_path,
        step_end_interval_semantics=STEP_END_INCLUSIVE,
    )
    selections = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        protocol_evidence=evidence,
        role_bindings=_chaser_bindings(selection),
    )

    with pytest.raises(
        ProtocolSemanticChaserSelectionPublicationError,
        match="not publishable until their acquisition authority",
    ):
        build_protocol_semantic_chaser_selection_publication_plan(
            archive,
            selections=selections,
            source_selection=selection,
            run_name="unproven-inclusive-policy",
        )


def test_publication_round_trips_loader_minted_position_epochs(
    tmp_path: Path,
) -> None:
    archive, _selection, selections, plan = _plan(tmp_path)

    result = publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-publication",
    )
    handle = load_protocol_semantic_chaser_selection_source_handle(
        archive,
        run_name=plan.run_name,
        expected_recording_id="recording_1",
    )

    assert result["status"] == "published_selector_ineligible"
    assert result["selector_eligible"] is False
    assert result["selection"] == "none"
    assert result["production_authority"] is False
    assert result["registry_update"] is False
    assert handle.selection_identity_sha256 == selections.identity_sha256
    assert handle.standalone_solid_black_status.startswith("not_applicable_")
    epochs = handle.position_suite_epochs()
    assert tuple(epoch.analysis_role for epoch in epochs) == CHASER_WINDOW_ROLES
    assert [(epoch.start_frame, epoch.end_frame) for epoch in epochs] == [
        (0, 10),
        (10, 20),
        (20, 29),
    ]
    assert all(not values.flags.writeable for values in handle.arrays.values())
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent = root[RUNS_PARENT_PATH]
    assert not set(parent.attrs).intersection(
        {"latest", "latest_complete", "selected", "current"}
    )


def test_two_step_publication_preserves_real_baseline_but_position_epochs_do_not_alias_it(
    tmp_path: Path,
) -> None:
    archive, _selection, _selections, plan = _plan(
        tmp_path,
        two_step=True,
        include_standalone_window=True,
    )
    publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-two-step",
    )
    handle = load_protocol_semantic_chaser_selection_source_handle(
        archive,
        run_name=plan.run_name,
    )

    assert handle.standalone_solid_black_status == "selected"
    assert tuple(handle.role_records) == (
        STANDALONE_SOLID_BLACK_ROLE,
        *CHASER_WINDOW_ROLES,
    )
    assert tuple(
        epoch.analysis_role for epoch in handle.position_suite_epochs()
    ) == CHASER_WINDOW_ROLES


def test_array_tamper_is_rejected_without_relying_on_dense_rehash(
    tmp_path: Path,
) -> None:
    archive, _selection, _selections, plan = _plan(tmp_path)
    publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-tamper",
    )
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    run = root[plan.run_path]
    run["selected_end_frame_exclusive"][2] = 28
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(
        ProtocolSemanticChaserSelectionPublicationError,
        match="array .* is stale",
    ):
        load_protocol_semantic_chaser_selection_source_handle(
            archive,
            run_name=plan.run_name,
        )


def test_rehashed_role_swap_is_rejected_against_current_exact_source(
    tmp_path: Path,
) -> None:
    archive, _selection, _selections, plan = _plan(tmp_path)
    publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-rehash",
    )
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    run = root[plan.run_path]
    manifest = copy.deepcopy(dict(run.attrs[MANIFEST_ATTR]))
    manifest["role_records"][2]["selected_end_frame_exclusive"] = 28
    manifest["payload_digest"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "payload_digest"}
    )
    run.attrs[MANIFEST_ATTR] = manifest
    run.attrs[MANIFEST_DIGEST_ATTR] = canonical_json_sha256(manifest)
    run["selected_end_frame_exclusive"][2] = 28
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(
        ProtocolSemanticChaserSelectionPublicationError,
        match="role records differ from current recompilation",
    ):
        load_protocol_semantic_chaser_selection_source_handle(
            archive,
            run_name=plan.run_name,
        )


def test_source_fingerprint_drift_invalidates_published_handle(
    tmp_path: Path,
) -> None:
    archive, _selection, _selections, plan = _plan(tmp_path)
    publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-source",
    )
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root["analysis/stimulus_runs/stimulus_1"].attrs["tamper_marker"] = True
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(
        ProtocolSemanticChaserSelectionPublicationError,
        match="fingerprint differs",
    ):
        load_protocol_semantic_chaser_selection_source_handle(
            archive,
            run_name=plan.run_name,
        )


def test_rehashed_unknown_role_is_rejected(tmp_path: Path) -> None:
    archive, _selection, _selections, plan = _plan(tmp_path)
    publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-unknown-role",
    )
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    run = root[plan.run_path]
    manifest = copy.deepcopy(dict(run.attrs[MANIFEST_ATTR]))
    manifest["role_records"][0]["role"] = "unknown_chaser_phase"
    manifest["role_order"][0] = "unknown_chaser_phase"
    manifest["payload_digest"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "payload_digest"}
    )
    run.attrs[MANIFEST_ATTR] = manifest
    run.attrs[MANIFEST_DIGEST_ATTR] = canonical_json_sha256(manifest)
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(
        ProtocolSemanticChaserSelectionPublicationError,
        match="inexact role set or order",
    ):
        load_protocol_semantic_chaser_selection_source_handle(
            archive,
            run_name=plan.run_name,
        )


def test_root_timeline_metadata_drift_invalidates_published_handle(
    tmp_path: Path,
) -> None:
    archive, _selection, _selections, plan = _plan(tmp_path)
    publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-timeline",
    )
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    source_video = dict(root.attrs["source_video_metadata"])
    source_video["semantic_tamper_marker"] = True
    root.attrs["source_video_metadata"] = source_video
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(
        ProtocolSemanticChaserSelectionPublicationError,
        match="source epoch authority changed|timeline evidence changed",
    ):
        load_protocol_semantic_chaser_selection_source_handle(
            archive,
            run_name=plan.run_name,
        )


def test_handles_cannot_be_fabricated_and_runs_cannot_be_replaced(
    tmp_path: Path,
) -> None:
    archive, selection, selections, plan = _plan(tmp_path)
    with pytest.raises(TypeError, match="strict loader"):
        ProtocolSemanticChaserSelectionSourceHandle()
    publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-first",
    )
    with pytest.raises(FileExistsError, match="replace existing run"):
        build_protocol_semantic_chaser_selection_publication_plan(
            archive,
            selections=selections,
            source_selection=selection,
            run_name=plan.run_name,
        )


def test_position_suite_epochs_require_the_exact_semantic_source(
    tmp_path: Path,
) -> None:
    archive, selection, _selections, plan = _plan(tmp_path)
    publish_protocol_semantic_chaser_selection_run(
        plan,
        scratch_root=tmp_path / "scratch-position-binding",
    )
    handle = load_protocol_semantic_chaser_selection_source_handle(
        archive,
        run_name=plan.run_name,
    )

    epochs = _epoch_specs(selection, (), semantic_selection=handle)

    assert [epoch.analysis_role for epoch in epochs] == [
        "chaser_pre",
        "chaser_training",
        "chaser_post",
    ]
    assert [epoch.start_frame for epoch in epochs] == [0, 10, 20]
    assert [epoch.end_frame for epoch in epochs] == [10, 20, 29]
    with pytest.raises(
        ProviderChaserPositionSuiteCanaryError,
        match="cannot be combined",
    ):
        _epoch_specs(
            selection,
            (("legacy_alias", 0),),
            semantic_selection=handle,
        )
