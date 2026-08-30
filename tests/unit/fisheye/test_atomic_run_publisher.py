from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import threading
import uuid

import numpy as np
import pytest
import zarr

from fisheye.shared import atomic_run_publisher as mod
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)


OWNER_ATTR = "publication_owner_uuid"
SELECTOR_OWNER_ATTR = "selector_publication_owner"


def _publication_fixture(tmp_path: Path):
    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w", use_consolidated=False)
    parent = root.require_group("analysis").require_group("runs")
    parent.attrs["latest"] = "previous"
    parent.attrs["publication_generation"] = 0

    local = tmp_path / "local.zarr"
    local_run = zarr.open_group(str(local), mode="w", use_consolidated=False)
    owner = str(uuid.uuid4())
    local_run.attrs.update(
        {
            OWNER_ATTR: owner,
            "stage_selector_eligible": False,
            "palette_run_completion_status": "running",
        }
    )
    local_run.create_array(
        "values",
        data=np.asarray([1, 2], dtype=np.int16),
        chunks=(2,),
    )
    target = source / "analysis" / "runs" / "candidate"
    spec = mod.AtomicRunPublishSpec(
        source_zarr=source,
        local_run_path=local,
        target_run_path=target,
        run_name="candidate",
        lock_suffix="atomic-test",
        publish_schema_id="palette.test_atomic_publish",
        policy="unit_test",
        rollback_policy="retain_failed_public_tombstone_and_restore_owned_attrs",
        publication_owner_attr=OWNER_ATTR,
        selector_owner_attr=SELECTOR_OWNER_ATTR,
        selector_generation_attr="publication_generation",
        owned_parent_attr_names=(
            ("latest", "publication_generation", SELECTOR_OWNER_ATTR),
        ),
    )

    def validate(path: Path):
        zarr.open_group(str(path), mode="r", use_consolidated=False)
        return {"valid": True}

    def prepare(open_root):
        return (open_root["analysis/runs"],)

    return source, target, spec, owner, validate, prepare


def test_atomic_copy_integrity_defaults_fail_safe() -> None:
    assert mod.DEFAULT_COPY_INTEGRITY_POLICY == "content_checksum_required_v1"
    assert mod.DEFAULT_COPY_CONTENT_CHECKSUM is True


def test_atomic_publisher_can_keep_operational_receipt_outside_sealed_run(
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)
    sealed = replace(spec, persist_run_receipt=False)

    result = mod.atomic_publish_run_group(
        sealed,
        copy_backend="python",
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=lambda _root, _parent, run: run.attrs.update(
            {"palette_run_completion_status": "complete"}
        ),
        verify_pointers=lambda _root: None,
    )

    assert result["final_validation"]["valid"] is True
    run = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert "cluster_output_staging" not in run.attrs
    assert run.attrs["palette_run_completion_status"] == "complete"


def test_atomic_temporary_paths_are_host_job_and_attempt_unique(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, _target, first, _owner, _validate, _prepare = _publication_fixture(
        tmp_path
    )
    second = replace(first, publication_attempt_uuid=uuid.uuid4().hex)
    monkeypatch.setattr(mod.socket, "gethostname", lambda: "compute/node 7")
    monkeypatch.setenv("LSB_JOBID", "job/123")

    assert first.temporary_path != second.temporary_path
    assert first.temporary_path.parent == first.target_run_path.parent
    assert first.temporary_path.name.startswith(
        ".candidate.publish_tmp.compute_node_7.job_123."
    )
    assert first.publication_attempt_uuid in first.temporary_path.name
    assert second.publication_attempt_uuid in second.temporary_path.name


def test_atomic_temporary_path_rejects_non_uuid_attempt_token(
    tmp_path: Path,
) -> None:
    _source, _target, spec, _owner, _validate, _prepare = _publication_fixture(
        tmp_path
    )

    with pytest.raises(ValueError, match="publication_attempt_uuid"):
        _ = replace(spec, publication_attempt_uuid="../escape").temporary_path


def test_atomic_publisher_checks_renamed_owner_before_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, target, spec, owner, validate, prepare = _publication_fixture(tmp_path)
    real_replace = os.replace
    callbacks: list[str] = []

    def swap_owner_after_rename(source_path, target_path):
        real_replace(source_path, target_path)
        if Path(target_path) == target:
            run = zarr.open_group(
                str(target),
                mode="a",
                use_consolidated=False,
            )
            run.attrs[OWNER_ATTR] = str(uuid.uuid4())

    monkeypatch.setattr(mod.os, "replace", swap_owner_after_rename)

    with pytest.raises(RuntimeError, match="changed publication owner"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            after_rename=lambda *_args: callbacks.append("after_rename"),
            complete_run=lambda *_args: callbacks.append("complete"),
            verify_pointers=lambda *_args: callbacks.append("verify"),
        )

    assert callbacks == []
    replacement = zarr.open_group(
        str(target),
        mode="r",
        use_consolidated=False,
    )
    assert replacement.attrs[OWNER_ATTR] != owner
    assert replacement.attrs["stage_selector_eligible"] is False
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert root["analysis/runs"].attrs["latest"] == "previous"


def test_atomic_publisher_rejects_competing_selector_rollback_authorities(
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)

    with pytest.raises(ValueError, match="mutually exclusive"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=lambda *_args: None,
            verify_pointers=lambda *_args: None,
            rollback_activation=lambda: None,
        )

    assert not target.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert root["analysis/runs"].attrs["latest"] == "previous"


def test_atomic_publisher_retains_owned_tombstone_when_replace_renames_then_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(
        tmp_path
    )
    real_replace = os.replace

    def rename_then_interrupt(source_path, target_path):
        real_replace(source_path, target_path)
        if Path(target_path) == target:
            raise KeyboardInterrupt("injected interrupt after successful rename")

    monkeypatch.setattr(mod.os, "replace", rename_then_interrupt)

    with pytest.raises(KeyboardInterrupt, match="after successful rename"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=lambda *_args: None,
            verify_pointers=lambda *_args: None,
        )

    assert target.exists()
    assert not spec.temporary_path.exists()
    failed = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert failed.attrs[OWNER_ATTR] == _owner
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_completion_status"] == "failed"
    tombstone = failed.attrs[mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR]
    assert tombstone["public_path_retained"] is True
    assert tombstone["retry_policy"] == "new_immutable_run_name_required"
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert root["analysis/runs"].attrs["latest"] == "previous"


def test_atomic_rollback_retains_failed_public_tombstone_and_restores_owned_selectors(
    tmp_path: Path,
) -> None:
    source, target, spec, owner, validate, prepare = _publication_fixture(tmp_path)

    def fail_after_selector_write(open_root, _run, _physical_copy):
        parent = open_root["analysis/runs"]
        parent.attrs[SELECTOR_OWNER_ATTR] = {
            "owner_uuid": owner,
            "base_generation": 0,
            "next_generation": 1,
        }
        parent.attrs["latest"] = "candidate"
        parent.attrs["publication_generation"] = 1
        raise KeyboardInterrupt("injected callback interrupt")

    with pytest.raises(KeyboardInterrupt, match="injected callback interrupt"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            after_rename=fail_after_selector_write,
            complete_run=lambda *_args: None,
            verify_pointers=lambda *_args: None,
        )

    failed = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR in failed.attrs
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/runs"]
    assert parent.attrs["latest"] == "previous"
    assert parent.attrs["publication_generation"] == 0
    assert SELECTOR_OWNER_ATTR not in parent.attrs


def test_owned_selector_restore_stops_before_clobbering_successor_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = str(uuid.uuid4())
    successor = str(uuid.uuid4())
    attempted_lease = {
        "owner_uuid": owner,
        "base_generation": 0,
        "next_generation": 1,
    }

    class TakeoverAttrs(dict):
        def __setitem__(self, name, value):
            super().__setitem__(name, value)
            if name == "first" and value == "before-first":
                dict.__setitem__(
                    self,
                    SELECTOR_OWNER_ATTR,
                    {
                        "owner_uuid": successor,
                        "base_generation": 1,
                        "next_generation": 2,
                    },
                )
                dict.__setitem__(self, "publication_generation", 2)

    class Group:
        attrs = TakeoverAttrs(
            {
                "first": "candidate",
                "second": "candidate",
                "publication_generation": 1,
                SELECTOR_OWNER_ATTR: attempted_lease,
            }
        )

    group = Group()
    monkeypatch.setattr(mod, "_resolve_group", lambda _root, _path: group)
    receipt = mod._capture_owned_parent_rollback_receipt(
        object(),
        (
            (
                "analysis/runs",
                {
                    "first": "before-first",
                    "second": "before-second",
                    "publication_generation": 0,
                },
            ),
        ),
        (
            (
                "first",
                "second",
                "publication_generation",
                SELECTOR_OWNER_ATTR,
            ),
        ),
        lease_parent_path="analysis/runs",
        lease_attr=SELECTOR_OWNER_ATTR,
        generation_attr="publication_generation",
        publication_owner=owner,
    )
    assert receipt is not None

    mod._restore_owned_parent_attrs(object(), receipt)

    assert group.attrs["first"] == "before-first"
    assert group.attrs["second"] == "candidate"
    assert group.attrs["publication_generation"] == 2
    assert group.attrs[SELECTOR_OWNER_ATTR]["owner_uuid"] == successor


def test_unleased_parent_state_is_never_reconstructed_during_failure(
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)
    unleased = replace(
        spec,
        publication_owner_attr=None,
        selector_owner_attr=None,
        selector_generation_attr=None,
        owned_parent_attr_names=(),
    )

    def fail_after_unowned_pointer_write(open_root, _run, _physical_copy):
        open_root["analysis/runs"].attrs["latest"] = "candidate"
        raise RuntimeError("injected unleased callback failure")

    with pytest.raises(RuntimeError, match="injected unleased callback failure"):
        mod.atomic_publish_run_group(
            unleased,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            after_rename=fail_after_unowned_pointer_write,
            complete_run=lambda *_args: None,
            verify_pointers=lambda *_args: None,
        )

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert root["analysis/runs"].attrs["latest"] == "candidate"
    failed = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert failed.attrs[mod.ATOMIC_PUBLICATION_OWNER_ATTR]
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_completion_status"] == "failed"


def test_atomic_activation_callback_is_absolute_final_metadata_commit(
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)
    events: list[str] = []
    verified_copies: list[dict[str, object]] = []

    def after_rename(_root, _run, physical_copy):
        verified_copies.append(dict(physical_copy))

    def complete(_root, parent, run_group):
        events.append("complete")
        run_group.attrs["palette_run_completion_status"] = "complete"
        run_group.attrs["stage_selector_eligible"] = False
        parent.attrs["latest_complete"] = "candidate"
        parent.attrs["latest"] = "candidate"

    def verify(open_root):
        events.append("verify")
        parent = open_root["analysis/runs"]
        assert parent.attrs["latest_complete"] == "candidate"
        assert parent.attrs["latest"] == "candidate"

    def activate(_root, _parent, run_group):
        events.append("activate")
        payload = dict(run_group.attrs["cluster_output_staging"])
        assert payload["final_validation"]["valid"] is True
        assert "parent_attrs_after" in payload
        assert run_group.attrs["stage_selector_eligible"] is False
        run_group.attrs["stage_selector_eligible"] = True

    result = mod.atomic_publish_run_group(
        spec,
        copy_backend="python",
        validate_run=validate,
        prepare_parents=prepare,
        after_rename=after_rename,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=activate,
    )

    assert events == ["complete", "verify", "activate"]
    assert result["final_validation"]["valid"] is True
    assert result["physical_copy"]["verification"] == "sha256_all_physical_files"
    assert result["physical_copy"]["content_sha256"]
    assert verified_copies == [result["physical_copy"]]
    telemetry = result["runtime_telemetry"]
    assert telemetry["identity_policy"] == (
        "report_only_excluded_from_scientific_identity_and_payload_digests"
    )
    assert [phase["name"] for phase in telemetry["phases"]][-3:] == [
        "activation_preflight",
        "selector_activation",
        "publication_lock_release",
    ]
    published = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert published.attrs["stage_selector_eligible"] is True
    assert "runtime_telemetry" not in published.attrs["cluster_output_staging"]


def test_atomic_activation_accepts_persisted_then_interrupted_final_write(
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)

    def complete(_root, parent, run_group):
        run_group.attrs["palette_run_completion_status"] = "complete"
        parent.attrs["latest_complete"] = "candidate"
        parent.attrs["latest"] = "candidate"

    def activate(_root, _parent, run_group):
        run_group.attrs["stage_selector_eligible"] = True
        raise KeyboardInterrupt("injected failure after persisted final write")

    result = mod.atomic_publish_run_group(
        spec,
        copy_backend="python",
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=lambda _root: None,
        activate_run=activate,
    )

    assert result["final_validation"]["valid"] is True
    committed = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert committed.attrs["palette_run_completion_status"] == "complete"
    assert committed.attrs["stage_selector_eligible"] is True
    assert mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR not in committed.attrs


def test_atomic_activation_rejects_foreign_same_name_replacement(
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)
    foreign_owner = str(uuid.uuid4())
    prepare_calls = 0
    activation_calls: list[str] = []

    def hostile_prepare(root):
        nonlocal prepare_calls
        prepare_calls += 1
        parent = root["analysis/runs"]
        if prepare_calls == 4:
            del parent["candidate"]
            parent.create_group(
                "candidate",
                attributes={
                    OWNER_ATTR: foreign_owner,
                    "palette_run_completion_status": "complete",
                    "stage_selector_eligible": False,
                    "sentinel": "foreign replacement",
                },
            )
        return prepare(root)

    def complete(_root, parent, run_group):
        run_group.attrs["palette_run_completion_status"] = "complete"
        parent.attrs["latest_complete"] = "candidate"
        parent.attrs["latest"] = "candidate"

    with pytest.raises(RuntimeError, match="lost its exact owned"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=hostile_prepare,
            complete_run=complete,
            verify_pointers=lambda _root: None,
            activate_run=lambda *_args: activation_calls.append("activate"),
        )

    assert activation_calls == []
    replacement = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert replacement.attrs[OWNER_ATTR] == foreign_owner
    assert replacement.attrs["sentinel"] == "foreign replacement"
    assert replacement.attrs["stage_selector_eligible"] is False
    assert mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR not in replacement.attrs


def test_atomic_tombstone_cleanup_stops_on_same_name_takeover(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)
    foreign_owner = str(uuid.uuid4())
    real_fresh = mod._fresh_owned_public_target
    fresh_calls = 0

    def takeover_before_next_cleanup(*args, **kwargs):
        nonlocal fresh_calls
        fresh_calls += 1
        if fresh_calls == 2:
            root = zarr.open_group(str(source), mode="a", use_consolidated=False)
            parent = root["analysis/runs"]
            del parent["candidate"]
            parent.create_group(
                "candidate",
                attributes={
                    OWNER_ATTR: foreign_owner,
                    "palette_run_completion_status": "complete",
                    "stage_selector_eligible": True,
                    "sentinel": "takeover must survive",
                },
            )
        return real_fresh(*args, **kwargs)

    monkeypatch.setattr(
        mod,
        "_fresh_owned_public_target",
        takeover_before_next_cleanup,
    )

    with pytest.raises(RuntimeError, match="injected publication failure"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("injected publication failure")
            ),
            verify_pointers=lambda _root: None,
        )

    replacement = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert replacement.attrs[OWNER_ATTR] == foreign_owner
    assert replacement.attrs["sentinel"] == "takeover must survive"
    assert replacement.attrs["palette_run_completion_status"] == "complete"
    assert replacement.attrs["stage_selector_eligible"] is True
    assert mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR not in replacement.attrs


def test_atomic_failed_tombstone_removes_persisted_completion_timestamp(
    tmp_path: Path,
) -> None:
    _source, target, spec, _owner, validate, prepare = _publication_fixture(
        tmp_path
    )

    def fail_after_completion(_root, _parent, run_group):
        run_group.attrs["palette_run_completion_status"] = "complete"
        run_group.attrs["palette_run_completed_at_utc"] = (
            "2026-07-20T00:00:00+00:00"
        )
        raise RuntimeError("injected failure after completion metadata")

    with pytest.raises(RuntimeError, match="after completion metadata"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=fail_after_completion,
            verify_pointers=lambda _root: None,
        )

    failed = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert "palette_run_completed_at_utc" not in failed.attrs
    assert mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR in failed.attrs


def test_atomic_publishers_share_one_archive_lock_and_consolidate_both_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, target, first_spec, _owner, validate, prepare = _publication_fixture(
        tmp_path
    )
    second_spec = replace(
        first_spec,
        target_run_path=target.with_name("candidate_two"),
        run_name="candidate_two",
        lock_suffix="different-stage-label",
    )
    assert first_spec.lock_path == second_spec.lock_path
    assert first_spec.lock_path.name.endswith(
        f".{mod.ARCHIVE_PUBLICATION_LOCK_SUFFIX}.lock"
    )

    first_holds_lock = threading.Event()
    release_first = threading.Event()
    second_entered_locked_body = threading.Event()
    real_locked = mod._atomic_publish_locked

    def observe_locked_body(spec, **kwargs):  # type: ignore[no-untyped-def]
        if spec.run_name == "candidate_two":
            second_entered_locked_body.set()
        return real_locked(spec, **kwargs)

    monkeypatch.setattr(mod, "_atomic_publish_locked", observe_locked_body)

    def complete(_root, _parent, run_group):  # type: ignore[no-untyped-def]
        run_group.attrs["palette_run_completion_status"] = "complete"
        run_group.attrs["stage_selector_eligible"] = False

    def first_after_rename(  # type: ignore[no-untyped-def]
        _root, _run, _physical_copy
    ):
        first_holds_lock.set()
        assert release_first.wait(timeout=5)

    def consolidate(_root, _parent, _run):  # type: ignore[no-untyped-def]
        consolidate_metadata_capture_expected_warnings(source)

    def publish(spec, *, after_rename=None):  # type: ignore[no-untyped-def]
        return mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=lambda _root: None,
            after_rename=after_rename,
            activate_run=consolidate,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            publish,
            first_spec,
            after_rename=first_after_rename,
        )
        assert first_holds_lock.wait(timeout=5)
        second = executor.submit(publish, second_spec)
        assert not second_entered_locked_body.wait(timeout=0.25)
        release_first.set()
        first.result(timeout=10)
        second.result(timeout=10)

    assert second_entered_locked_body.is_set()
    consolidated = zarr.open_group(
        str(source), mode="r", use_consolidated=True
    )
    parent = consolidated["analysis/runs"]
    assert parent["candidate"].attrs["palette_run_completion_status"] == "complete"
    assert parent["candidate_two"].attrs[
        "palette_run_completion_status"
    ] == "complete"


def test_atomic_failure_visibility_repair_reconsolidates_tombstone(
    tmp_path: Path,
) -> None:
    source, target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)
    repairs: list[Path] = []

    def complete(_root, _parent, run_group):  # type: ignore[no-untyped-def]
        run_group.attrs["palette_run_completion_status"] = "complete"
        run_group.attrs["stage_selector_eligible"] = False

    def consolidate_then_fail(_root, _parent, _run):  # type: ignore[no-untyped-def]
        consolidate_metadata_capture_expected_warnings(source)
        raise RuntimeError("injected failure after consolidation")

    def repair(path: Path) -> None:
        repairs.append(path)
        consolidate_metadata_capture_expected_warnings(source)

    with pytest.raises(RuntimeError, match="failure after consolidation"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=lambda _root: None,
            activate_run=consolidate_then_fail,
            repair_failed_publication_visibility=repair,
        )

    assert repairs == [target]
    direct = zarr.open_group(str(source), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(source), mode="r", use_consolidated=True)
    for root in (direct, consolidated):
        failed = root["analysis/runs/candidate"]
        assert failed.attrs["stage_selector_eligible"] is False
        assert failed.attrs["palette_run_completion_status"] == "failed"
        assert mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR in failed.attrs


def test_atomic_failure_visibility_rejects_noop_repair(
    tmp_path: Path,
) -> None:
    source, _target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)

    def complete(_root, _parent, run_group):  # type: ignore[no-untyped-def]
        run_group.attrs["palette_run_completion_status"] = "complete"
        run_group.attrs["stage_selector_eligible"] = False

    def consolidate_then_fail(_root, _parent, _run):  # type: ignore[no-untyped-def]
        consolidate_metadata_capture_expected_warnings(source)
        raise RuntimeError("injected failure after consolidation")

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=lambda _root: None,
            activate_run=consolidate_then_fail,
            repair_failed_publication_visibility=lambda _path: None,
        )


def test_atomic_failure_visibility_rejects_rewritten_tombstone(
    tmp_path: Path,
) -> None:
    source, _target, spec, _owner, validate, prepare = _publication_fixture(tmp_path)

    def complete(_root, _parent, run_group):  # type: ignore[no-untyped-def]
        run_group.attrs["palette_run_completion_status"] = "complete"
        run_group.attrs["stage_selector_eligible"] = False

    def consolidate_then_fail(_root, _parent, _run):  # type: ignore[no-untyped-def]
        consolidate_metadata_capture_expected_warnings(source)
        raise RuntimeError("injected failure after consolidation")

    def rewrite_then_consolidate(path: Path) -> None:
        failed = zarr.open_group(str(path), mode="r+", use_consolidated=False)
        receipt = dict(failed.attrs[mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR])
        receipt["failure_type"] = "forged"
        failed.attrs[mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR] = receipt
        consolidate_metadata_capture_expected_warnings(source)

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        mod.atomic_publish_run_group(
            spec,
            copy_backend="python",
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=lambda _root: None,
            activate_run=consolidate_then_fail,
            repair_failed_publication_visibility=rewrite_then_consolidate,
        )
