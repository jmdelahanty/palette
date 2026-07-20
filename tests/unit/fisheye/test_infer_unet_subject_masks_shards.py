from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import pytest
import zarr

from fisheye.segmentation import infer_unet_subject_masks as mod
import fisheye.shared.subject_mask_coordinate_publication as publication_module
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR, mark_run_complete


def _provenance(output_parent: str) -> dict[str, object]:
    return build_writer_run_provenance(
        command="test_infer_unet_subject_masks_shards",
        params={"output_parent": output_parent},
        input_run_ids={"crop": "crop_001"},
    )


def test_prepare_subject_mask_shard_run_does_not_touch_canonical_parent(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="w")
    canonical_parent = root.require_group("subject_mask_runs")
    canonical_parent.attrs["latest"] = "canonical_subject_masks"
    canonical_parent.attrs["latest_complete"] = "canonical_subject_masks"
    canonical_parent.create_group("canonical_subject_masks")
    root.attrs["current_subject_mask_group_path"] = "subject_mask_runs/canonical_subject_masks"

    shard, run_name = mod._prepare_run_group(
        root,
        run_name="subject_mask_shard_clip_000001",
        overwrite=False,
        output_parent=mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT,
    )
    shard.attrs.update(
        mod._shard_attrs_from_args(
            mod._build_arg_parser().parse_args(
                [
                    "archive.zarr",
                    "checkpoint.pt",
                    "--output-parent",
                    "subject_mask_shard_runs",
                    "--source-collection-id",
                    "collection_001",
                    "--source-clip-id",
                    "clip_000001",
                    "--source-clip-index",
                    "1",
                    "--source-roi-cache-alias-manifest",
                    "/tmp/cache_alias.json",
                    "--source-roi-cache-row-index-path",
                    "/tmp/cache_rows.parquet",
                ]
            ),
            output_parent=mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT,
        )
    )
    mark_run_complete(
        shard,
        parent_group=root[mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT],
        run_name=run_name,
        run_provenance=_provenance(mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT),
    )

    assert root["subject_mask_runs"].attrs["latest"] == "canonical_subject_masks"
    assert root["subject_mask_runs"].attrs["latest_complete"] == "canonical_subject_masks"
    assert root.attrs["current_subject_mask_group_path"] == "subject_mask_runs/canonical_subject_masks"
    assert "subject_mask_shard_runs" in root
    assert "latest" not in root["subject_mask_shard_runs"].attrs
    assert "latest_complete" not in root["subject_mask_shard_runs"].attrs
    assert shard.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert shard.attrs["is_collection_shard"] is True
    assert shard.attrs["stage_selector_eligible"] is False
    assert shard.attrs["canonical_selector_publication"] == "suppressed_for_collection_shard"
    assert shard.attrs["source_collection_id"] == "collection_001"
    assert shard.attrs["source_clip_id"] == "clip_000001"
    assert shard.attrs["source_clip_index"] == 1
    assert shard.attrs["source_roi_cache_alias_manifest"] == "/tmp/cache_alias.json"
    assert shard.attrs["source_roi_cache_row_index_path"] == "/tmp/cache_rows.parquet"


def test_prepare_default_subject_mask_run_stays_ineligible_until_activation(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="w")

    run, run_name = mod._prepare_run_group(
        root,
        run_name="subject_masks_full",
        overwrite=False,
    )
    mark_run_complete(
        run,
        parent_group=root["subject_mask_runs"],
        run_name=run_name,
        run_provenance=_provenance(mod.SUBJECT_MASK_CANONICAL_OUTPUT_PARENT),
    )

    assert "subject_mask_shard_runs" not in root
    assert "latest" not in root["subject_mask_runs"].attrs
    assert "latest_complete" not in root["subject_mask_runs"].attrs
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert run.attrs["stage_selector_eligible"] is False


def test_prepare_refuses_overwrite_of_complete_subject_mask_run(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="w")
    run, run_name = mod._prepare_run_group(
        root,
        run_name="subject_masks_full",
        overwrite=False,
    )
    mark_run_complete(
        run,
        parent_group=root["subject_mask_runs"],
        run_name=run_name,
        run_provenance=_provenance(mod.SUBJECT_MASK_CANONICAL_OUTPUT_PARENT),
    )

    with pytest.raises(ValueError, match="Refusing to overwrite.*complete"):
        mod._prepare_run_group(
            root,
            run_name=run_name,
            overwrite=True,
        )


@pytest.mark.parametrize(
    "selector",
    ("latest", "latest_complete", "latest_pending", "authoritative_run"),
)
def test_prepare_refuses_overwrite_of_selected_subject_mask_run(
    tmp_path: Path,
    selector: str,
) -> None:
    root = zarr.open_group(str(tmp_path / f"{selector}.zarr"), mode="w")
    _run, run_name = mod._prepare_run_group(
        root,
        run_name="selected_attempt",
        overwrite=False,
    )
    if selector != "latest_pending":
        del root["subject_mask_runs"].attrs["latest_pending"]
    root["subject_mask_runs"].attrs[selector] = run_name

    with pytest.raises(ValueError, match=f"selected by {selector}"):
        mod._prepare_run_group(
            root,
            run_name=run_name,
            overwrite=True,
        )


def test_subject_mask_attempt_boundary_catches_keyboardinterrupt_and_restores_selectors(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "interrupt.zarr"), mode="w")
    parent = root.require_group("subject_mask_runs")
    parent.attrs["latest"] = "previous"
    parent.attrs["latest_complete"] = "previous"
    observed: dict[str, str] = {}

    @mod._fail_closed_subject_mask_attempt
    def interrupted_attempt() -> None:
        run, run_name = mod._prepare_run_group(
            root,
            run_name="interrupted",
            overwrite=False,
        )
        observed["run_name"] = run_name
        run.attrs["payload_started"] = True
        parent.attrs["latest"] = run_name
        raise KeyboardInterrupt("synthetic interruption")

    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        interrupted_attempt()

    parent = root["subject_mask_runs"]
    failed = parent[observed["run_name"]]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert parent.attrs["latest"] == "previous"
    assert parent.attrs["latest_complete"] == "previous"
    assert "latest_pending" not in parent.attrs


def test_subject_mask_activation_eligibility_is_literal_final_persisted_action() -> None:
    source = textwrap.dedent(
        inspect.getsource(
            publication_module._activate_validated_subject_mask_coordinate_surfaces
        )
    )
    function = ast.parse(source).body[0]
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    final = function.body[-1]
    assert isinstance(final, ast.Assign)
    target = final.targets[0]
    assert isinstance(target, ast.Subscript)
    assert isinstance(target.slice, ast.Constant)
    assert target.slice.value == "stage_selector_eligible"
    assert isinstance(final.value, ast.Constant)
    assert final.value.value is True


def test_subject_mask_writer_orders_publication_completion_activation() -> None:
    source = inspect.getsource(mod.main.__wrapped__)
    publication = source.index("publish_subject_mask_coordinate_surfaces(")
    completion = source.index("mark_run_complete(", publication)
    activation = source.index(
        "_activate_validated_subject_mask_coordinate_surfaces(",
        completion,
    )
    assert publication < completion < activation
    assert "_load_completed_ineligible_subject_mask_coordinate_surfaces(" not in source

    activation_source = inspect.getsource(
        publication_module._activate_validated_subject_mask_coordinate_surfaces
    )
    validation = activation_source.index(
        "_load_completed_ineligible_subject_mask_coordinate_surfaces("
    )
    eligibility = activation_source.rindex(
        'activation_run.attrs["stage_selector_eligible"] = True'
    )
    assert validation < eligibility


def test_subject_mask_child_creation_persists_atomic_owner_and_ineligible_sentinel() -> None:
    source = inspect.getsource(mod._prepare_run_group)
    function = ast.parse(textwrap.dedent(source)).body[0]
    create_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_group"
    ]
    assert len(create_calls) == 1
    attributes = next(
        keyword.value
        for keyword in create_calls[0].keywords
        if keyword.arg == "attributes"
    )
    assert isinstance(attributes, ast.Name)
    sentinel_assignments = [
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == attributes.id
            for target in node.targets
        )
    ]
    assert len(sentinel_assignments) == 1
    payload = sentinel_assignments[0].value
    assert isinstance(payload, ast.Dict)
    keys = {
        key.value: value
        for key, value in zip(payload.keys, payload.values, strict=True)
        if isinstance(key, ast.Constant)
    }
    assert isinstance(keys["stage_selector_eligible"], ast.Constant)
    assert keys["stage_selector_eligible"].value is False
    assert any(
        isinstance(key, ast.Name)
        and key.id == "SUBJECT_MASK_PUBLICATION_OWNER_ATTR"
        for key in payload.keys
    )


def test_prepare_rejects_stale_selector_naming_missing_child(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "stale.zarr"), mode="w")
    parent = root.require_group("subject_mask_runs")
    parent.attrs["latest"] = "missing_attempt"

    with pytest.raises(ValueError, match="stale selector"):
        mod._prepare_run_group(
            root,
            run_name="missing_attempt",
            overwrite=False,
        )
    assert "missing_attempt" not in root["subject_mask_runs"]


def test_prepare_checks_foreign_pending_before_overwrite_deletion(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "pending-before-delete.zarr"), mode="w")
    _run, run_name = mod._prepare_run_group(
        root,
        run_name="keep_running",
        overwrite=False,
    )
    parent = root["subject_mask_runs"]
    parent.attrs["latest_pending"] = "another_attempt"

    with pytest.raises(ValueError, match="already owned"):
        mod._prepare_run_group(
            root,
            run_name=run_name,
            overwrite=True,
        )
    assert run_name in root["subject_mask_runs"]


def test_failure_boundary_does_not_mark_replacement_child_failed(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "replacement.zarr"), mode="w")
    observed: dict[str, str] = {}

    @mod._fail_closed_subject_mask_attempt
    def replaced_attempt() -> None:
        _run, run_name = mod._prepare_run_group(
            root,
            run_name="replaced",
            overwrite=False,
        )
        parent = root["subject_mask_runs"]
        del parent[run_name]
        replacement_owner = "f" * 32
        replacement = parent.create_group(
            run_name,
            attributes={
                "stage_selector_eligible": False,
                mod.SUBJECT_MASK_PUBLICATION_OWNER_ATTR: replacement_owner,
            },
        )
        observed["owner"] = replacement_owner
        replacement.attrs["replacement_marker"] = "keep"
        raise KeyboardInterrupt("synthetic replacement")

    with pytest.raises(KeyboardInterrupt, match="synthetic replacement"):
        replaced_attempt()

    replacement = root["subject_mask_runs/replaced"]
    assert replacement.attrs[mod.SUBJECT_MASK_PUBLICATION_OWNER_ATTR] == observed["owner"]
    assert replacement.attrs["replacement_marker"] == "keep"
    assert replacement.attrs.get(RUN_COMPLETION_STATUS_ATTR) != "failed"


def test_failure_boundary_refuses_to_rollback_an_eligible_publication(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "eligible.zarr"), mode="w")

    @mod._fail_closed_subject_mask_attempt
    def committed_attempt() -> None:
        run, run_name = mod._prepare_run_group(
            root,
            run_name="committed",
            overwrite=False,
        )
        mark_run_complete(
            run,
            parent_group=None,
            run_name=run_name,
            run_provenance=_provenance(mod.SUBJECT_MASK_CANONICAL_OUTPUT_PARENT),
        )
        run.attrs["stage_selector_eligible"] = True
        raise KeyboardInterrupt("synthetic post-commit interruption")

    with pytest.raises(KeyboardInterrupt, match="post-commit"):
        committed_attempt()

    run = root["subject_mask_runs/committed"]
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert run.attrs["stage_selector_eligible"] is True
    assert root["subject_mask_runs"].attrs["latest_pending"] == "committed"


def test_failure_rollback_restores_only_attempt_owned_selectors(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "concurrent.zarr"), mode="w")
    parent = root.require_group("subject_mask_runs")
    parent.attrs["latest"] = "previous"
    parent.attrs["latest_complete"] = "previous"
    parent.attrs["authoritative_run"] = "approved"
    provenance = {"approved_by": "reviewer", "note": "keep"}
    parent.attrs["authoritative_run_provenance"] = provenance

    @mod._fail_closed_subject_mask_attempt
    def interrupted_attempt() -> None:
        _run, run_name = mod._prepare_run_group(
            root,
            run_name="interrupted",
            overwrite=False,
        )
        active_parent = root["subject_mask_runs"]
        active_parent.attrs["latest"] = run_name
        active_parent.attrs["latest_complete"] = "concurrent_complete"
        raise KeyboardInterrupt("synthetic concurrent mutation")

    with pytest.raises(KeyboardInterrupt, match="synthetic concurrent mutation"):
        interrupted_attempt()

    parent = root["subject_mask_runs"]
    assert parent.attrs["latest"] == "previous"
    assert parent.attrs["latest_complete"] == "concurrent_complete"
    assert parent.attrs["authoritative_run"] == "approved"
    assert parent.attrs["authoritative_run_provenance"] == provenance
    assert "latest_pending" not in parent.attrs


def test_failure_boundary_rolls_back_owned_publication_epoch_and_selectors(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "owned-epoch.zarr"), mode="w")
    parent = root.require_group("subject_mask_runs")
    parent.attrs["latest"] = "previous"
    parent.attrs["latest_complete"] = "previous"

    @mod._fail_closed_subject_mask_attempt
    def interrupted_activation() -> None:
        run, run_name = mod._prepare_run_group(
            root,
            run_name="interrupted",
            overwrite=False,
        )
        owner = run.attrs[mod.SUBJECT_MASK_PUBLICATION_OWNER_ATTR]
        active_parent = root["subject_mask_runs"]
        active_parent.attrs["latest"] = run_name
        active_parent.attrs["latest_complete"] = run_name
        active_parent.attrs[mod.SUBJECT_MASK_PUBLICATION_POLICY_ATTR] = (
            publication_module._SUBJECT_MASK_PUBLICATION_POLICY
        )
        active_parent.attrs[mod.SUBJECT_MASK_PUBLICATION_GENERATION_ATTR] = 1
        active_parent.attrs[mod.SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR] = {
            "schema_id": "palette.subject_mask_publication_lease",
            "schema_version": 1,
            "policy": publication_module._SUBJECT_MASK_PUBLICATION_POLICY,
            "run_path": f"subject_mask_runs/{run_name}",
            "publication_owner": owner,
            "base_generation": 0,
            "next_generation": 1,
        }
        raise KeyboardInterrupt("synthetic activation interruption")

    with pytest.raises(KeyboardInterrupt, match="activation interruption"):
        interrupted_activation()

    parent = root["subject_mask_runs"]
    assert parent.attrs["latest"] == "previous"
    assert parent.attrs["latest_complete"] == "previous"
    assert "latest_pending" not in parent.attrs
    assert mod.SUBJECT_MASK_PUBLICATION_GENERATION_ATTR not in parent.attrs
    assert mod.SUBJECT_MASK_PUBLICATION_POLICY_ATTR not in parent.attrs
    assert mod.SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR not in parent.attrs


def test_failure_boundary_reports_foreign_lease_instead_of_silently_leaving_selectors(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "foreign-lease.zarr"), mode="w")
    parent = root.require_group("subject_mask_runs")
    parent.attrs["latest"] = "previous"

    @mod._fail_closed_subject_mask_attempt
    def interrupted_by_foreign_lease() -> None:
        _run, run_name = mod._prepare_run_group(
            root,
            run_name="interrupted",
            overwrite=False,
        )
        active_parent = root["subject_mask_runs"]
        active_parent.attrs["latest"] = run_name
        active_parent.attrs[mod.SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR] = {
            "schema_id": "palette.subject_mask_publication_lease",
            "schema_version": 1,
            "policy": publication_module._SUBJECT_MASK_PUBLICATION_POLICY,
            "run_path": "subject_mask_runs/concurrent",
            "publication_owner": "f" * 32,
            "base_generation": 0,
            "next_generation": 1,
        }
        raise KeyboardInterrupt("synthetic concurrent lease replacement")

    with pytest.raises(
        RuntimeError,
        match="fail-closed rollback was incomplete.*owned parent selectors",
    ):
        interrupted_by_foreign_lease()

    parent = root["subject_mask_runs"]
    assert parent.attrs["latest"] == "interrupted"
    assert (
        parent.attrs[mod.SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR][
            "publication_owner"
        ]
        == "f" * 32
    )
