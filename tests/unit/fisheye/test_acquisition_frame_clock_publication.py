"""Publication enforcement with deterministic in-memory storage failures."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.provider_recording_timing_authority import (
    load_provider_recording_timing_authority,
)
from fisheye.shared import acquisition_frame_clock as mod
from tests.unit.fisheye.test_provider_recording_timing_authority import _clock_archive


class _Attrs(dict):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def __setitem__(self, key, value):
        before = self.group.root.on_before_write
        if before is not None:
            before(self.group, key, value)
        self.group.root.events.append((self.group.path, key, copy.deepcopy(value)))
        super().__setitem__(key, copy.deepcopy(value))
        hook = self.group.root.on_write
        if hook is not None:
            hook(self.group, key, value)

    def update(self, values):
        for key, value in values.items():
            self[key] = value


class _Group:
    def __init__(self, path="", *, root=None, attributes=None):
        self.path = path
        self.root = self if root is None else root
        if root is None:
            self.nodes = {}
            self.events = []
            self.on_write = None
            self.on_before_write = None
            self._coordinate_archive_token = object()
        else:
            self._coordinate_archive_token = root._coordinate_archive_token
        self.attrs = _Attrs(self)
        self.root.nodes[path] = self
        if attributes is not None:
            self.attrs.update(attributes)

    def _path(self, name):
        return "/".join(part for part in (self.path, str(name).strip("/")) if part)

    def __getitem__(self, name):
        return self.root.nodes[self._path(name)]

    def __contains__(self, name):
        return self._path(name) in self.root.nodes

    def get(self, name, default=None):
        return self.root.nodes.get(self._path(name), default)

    def require_group(self, name):
        if name not in self:
            return self.create_group(name)
        return self[name]

    def create_group(self, name, *, attributes=None):
        if name in self:
            raise ValueError("public name is occupied")
        return _Group(self._path(name), root=self.root, attributes=attributes)

    def group_keys(self):
        prefix = self.path + "/"
        return [
            path[len(prefix) :]
            for path, node in self.root.nodes.items()
            if isinstance(node, _Group)
            and path.startswith(prefix)
            and "/" not in path[len(prefix) :]
        ]

    def keys(self):
        return self.group_keys()


@pytest.fixture
def clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "clock.csv"
    path.write_text(
        "recording_frame_id,timestamp,timestamp_sys\n"
        "1,1000000000,2000000000\n2,1010000000,2010000000\n",
        encoding="utf-8",
    )
    source = mod._load_csv_source(path, recording_dir=tmp_path, camera_id="2010093")
    root = _Group()

    def store(run, name, values, **_kwargs):
        root.nodes[run._path(name)] = np.array(values, copy=True)

    monkeypatch.setattr(mod, "store_array", store)
    return root, source


def _changed(source):
    return replace(source, camera_timestamp_ns=source.camera_timestamp_ns + 7)


def test_public_source_digest_matches_real_publication_without_mutation(clock):
    root, source = clock
    before = {name: np.array(getattr(source, name), copy=True) for name in mod._ARRAY_NAMES}
    digest = mod.acquisition_frame_clock_source_sha256(source)
    published = mod.publish_acquisition_frame_clock(root, source)
    assert digest == published.record_sha256
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(source, name), expected)


def test_public_source_digest_refuses_invalid_source(clock):
    _, source = clock
    invalid = replace(source, parent_frame_index=np.array([0, 0], dtype=np.int64))
    with pytest.raises(mod.AcquisitionFrameClockError):
        mod.acquisition_frame_clock_source_sha256(invalid)


def _parent(root):
    return root[mod.ACQUISITION_FRAME_CLOCK_RUNS_PATH]


def _assert_failed(root, name):
    run = _parent(root)[name]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert "palette_run_completed_at_utc" not in run.attrs
    tombstone = run.attrs["acquisition_frame_clock_publication_tombstone"]
    assert tombstone["schema_version"] == 1
    assert tombstone["retry_requires_new_run_name"] is True
    assert (
        tombstone["owner_uuid"]
        == run.attrs["acquisition_frame_clock_publication_owner_uuid"]
    )


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("failure", ["write", "corruption", "loadback"])
def test_payload_failure_preserves_selection_and_retry_retains_tombstone(
    clock,
    monkeypatch: pytest.MonkeyPatch,
    prior: bool,
    failure: str,
):
    root, source = clock
    previous = mod.publish_acquisition_frame_clock(root, source) if prior else None
    root_before = copy.deepcopy(dict(root.attrs))
    parent_before = copy.deepcopy(dict(_parent(root).attrs)) if prior else None
    source = _changed(source)
    original = mod.store_array

    class Unreadable:
        def __getitem__(self, key):
            raise OSError("injected loadback failure")

    def broken(run, name, values, **kwargs):
        original(run, name, values, **kwargs)
        if name == "camera_timestamp_ns":
            if failure == "write":
                raise OSError("injected array write failure")
            if failure == "corruption":
                root.nodes[run._path(name)][0] += 1
            if failure == "loadback":
                root.nodes[run._path(name)] = Unreadable()

    monkeypatch.setattr(mod, "store_array", broken)
    with pytest.raises((OSError, mod.AcquisitionFrameClockError)):
        mod.publish_acquisition_frame_clock(root, source)

    assert dict(root.attrs) == root_before
    parent = _parent(root)
    if prior:
        assert dict(parent.attrs) == parent_before
        assert mod.resolve_acquisition_frame_clock(root) == previous
    else:
        assert "latest" not in parent.attrs
        assert "latest_complete" not in parent.attrs
    failed = next(
        name
        for name in parent.group_keys()
        if previous is None or name != previous.run_name
    )
    _assert_failed(root, failed)
    tombstone_before = copy.deepcopy(dict(parent[failed].attrs))
    monkeypatch.setattr(mod, "store_array", original)
    retried = mod.publish_acquisition_frame_clock(root, source)
    assert retried.run_name != failed
    assert dict(parent[failed].attrs) == tombstone_before
    assert mod.resolve_acquisition_frame_clock(root) == retried
    events_before = list(root.events)
    assert mod.publish_acquisition_frame_clock(root, source) == retried
    assert root.events == events_before


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "key",
    [
        "palette_run_completion_status",
        "latest_complete",
        "latest",
        "acquisition_frame_clock_publication_policy",
        "acquisition_frame_clock_publication_generation",
        "acquisition_frame_clock_publication_lease",
        "acquisition_frame_clock_ref",
        "acquisition_frame_clock_sha256",
        "acquisition_frame_clock_camera_id",
        "acquisition_frame_clock_row_count",
        "acquisition_frame_clock_available",
        "acquisition_frame_clock_status",
    ],
)
def test_metadata_write_then_raise_restores_previous_selection(clock, key, prior):
    root, source = clock
    previous = mod.publish_acquisition_frame_clock(root, source) if prior else None
    root_before = copy.deepcopy(dict(root.attrs))
    parent_before = copy.deepcopy(dict(_parent(root).attrs)) if prior else None
    fired = False

    def fail(group, name, value):
        nonlocal fired
        if (
            not fired
            and name == key
            and (name != "palette_run_completion_status" or value == "complete")
        ):
            if (
                name.startswith("acquisition_frame_clock_")
                and group.path
                and "_publication_" not in name
            ):
                return
            fired = True
            raise OSError("injected persisted metadata failure")

    root.on_write = fail
    with pytest.raises(OSError, match="injected"):
        mod.publish_acquisition_frame_clock(root, _changed(source))
    root.on_write = None
    assert fired
    assert dict(root.attrs) == root_before
    if prior:
        assert dict(_parent(root).attrs) == parent_before
        assert mod.resolve_acquisition_frame_clock(root) == previous
    else:
        assert "latest" not in _parent(root).attrs
        assert "latest_complete" not in _parent(root).attrs
    failed = next(
        name
        for name in _parent(root).group_keys()
        if previous is None or name != previous.run_name
    )
    _assert_failed(root, failed)


def test_eligibility_is_first_created_false_and_literal_last_write(clock):
    root, source = clock
    resolved = mod.publish_acquisition_frame_clock(root, source)
    run_events = [event for event in root.events if event[0] == resolved.group_path]
    assert run_events[0][1] == "acquisition_frame_clock_publication_owner_uuid"
    assert run_events[1][1:] == ("stage_selector_eligible", False)
    assert root.events[-1] == (resolved.group_path, "stage_selector_eligible", True)
    assert mod.resolve_acquisition_frame_clock(root) == resolved


def test_persisted_final_eligibility_error_is_success(clock):
    root, source = clock

    def fail(group, name, value):
        if name == "stage_selector_eligible" and value is True:
            raise OSError("commit persisted but acknowledgment lost")

    root.on_write = fail
    resolved = mod.publish_acquisition_frame_clock(root, source)
    root.on_write = None
    assert mod.resolve_acquisition_frame_clock(root) == resolved
    assert (
        "acquisition_frame_clock_publication_tombstone"
        not in root[resolved.group_path].attrs
    )


def test_same_path_foreign_replacement_is_not_failed_or_selected(clock, monkeypatch):
    root, source = clock
    previous = mod.publish_acquisition_frame_clock(root, source)
    original = mod.store_array
    replacement = None

    def replace_owner(run, name, values, **kwargs):
        nonlocal replacement
        original(run, name, values, **kwargs)
        if name == mod._ARRAY_NAMES[-1]:
            replacement = _Group(
                run.path,
                root=root,
                attributes={
                    "acquisition_frame_clock_publication_owner_uuid": "foreign-owner",
                    "stage_selector_eligible": False,
                    "foreign_payload": "preserve",
                },
            )

    monkeypatch.setattr(mod, "store_array", replace_owner)
    with pytest.raises(mod.AcquisitionFrameClockError, match="owner"):
        mod.publish_acquisition_frame_clock(root, _changed(source))
    assert dict(replacement.attrs) == {
        "acquisition_frame_clock_publication_owner_uuid": "foreign-owner",
        "stage_selector_eligible": False,
        "foreign_payload": "preserve",
    }
    assert mod.resolve_acquisition_frame_clock(root) == previous


def test_republishing_old_source_never_mutates_old_public_child(clock):
    root, source = clock
    first = mod.publish_acquisition_frame_clock(root, source)
    before = copy.deepcopy(dict(root[first.group_path].attrs))
    mod.publish_acquisition_frame_clock(root, _changed(source))
    replay = mod.publish_acquisition_frame_clock(root, source)
    assert replay.record_sha256 == first.record_sha256
    assert replay.run_name != first.run_name
    assert dict(root[first.group_path].attrs) == before
    assert mod.resolve_acquisition_frame_clock(root) == replay


def test_final_eligibility_failure_before_persist_restores_prior(clock):
    root, source = clock
    previous = mod.publish_acquisition_frame_clock(root, source)
    root_before = copy.deepcopy(dict(root.attrs))
    parent_before = copy.deepcopy(dict(_parent(root).attrs))

    def refuse(group, name, value):
        if name == "stage_selector_eligible" and value is True:
            raise OSError("final commit refused")

    root.on_before_write = refuse
    with pytest.raises(OSError, match="final commit refused"):
        mod.publish_acquisition_frame_clock(root, _changed(source))
    root.on_before_write = None
    assert dict(root.attrs) == root_before
    assert dict(_parent(root).attrs) == parent_before
    assert mod.resolve_acquisition_frame_clock(root) == previous


def test_foreign_root_and_selector_takeover_survive_rollback(clock):
    root, source = clock
    mod.publish_acquisition_frame_clock(root, source)
    fired = False

    def takeover(group, name, value):
        nonlocal fired
        if not fired and group.path == "" and name == "acquisition_frame_clock_ref":
            fired = True
            root.attrs["acquisition_frame_clock_ref"] = "foreign-clock"
            root.attrs["acquisition_frame_clock_sha256"] = "foreign-digest"
            _parent(root).attrs["latest"] = "foreign-run"
            _parent(root).attrs["latest_complete"] = "foreign-run"
            _parent(root).attrs[mod._PUBLICATION_LEASE_ATTR] = {"owner_uuid": "foreign"}
            raise OSError("foreign publication takeover")

    root.on_write = takeover
    with pytest.raises(OSError, match="foreign publication takeover"):
        mod.publish_acquisition_frame_clock(root, _changed(source))
    root.on_write = None
    assert root.attrs["acquisition_frame_clock_ref"] == "foreign-clock"
    assert root.attrs["acquisition_frame_clock_sha256"] == "foreign-digest"
    assert _parent(root).attrs["latest"] == "foreign-run"
    assert _parent(root).attrs["latest_complete"] == "foreign-run"
    assert _parent(root).attrs[mod._PUBLICATION_LEASE_ATTR] == {"owner_uuid": "foreign"}


@pytest.mark.parametrize(
    "phase", ["palette_run_started_at_utc", "palette_run_completed_at_utc"]
)
def test_shared_lifecycle_does_not_mutate_replacement_after_owner_loss(clock, phase):
    root, source = clock
    previous = mod.publish_acquisition_frame_clock(root, source)
    replacement = None

    def takeover(group, name, value):
        nonlocal replacement
        if replacement is None and name == phase:
            replacement = _Group(
                group.path,
                root=root,
                attributes={
                    mod._PUBLICATION_OWNER_ATTR: "foreign-owner",
                    "stage_selector_eligible": False,
                    "foreign_payload": "preserve",
                },
            )

    root.on_write = takeover
    with pytest.raises(mod.AcquisitionFrameClockError, match="owner"):
        mod.publish_acquisition_frame_clock(root, _changed(source))
    root.on_write = None
    assert dict(replacement.attrs) == {
        mod._PUBLICATION_OWNER_ATTR: "foreign-owner",
        "stage_selector_eligible": False,
        "foreign_payload": "preserve",
    }
    assert mod.resolve_acquisition_frame_clock(root) == previous


def test_selected_legacy_clock_replay_is_read_only(clock):
    root, source = clock
    prior = mod.publish_acquisition_frame_clock(root, source)
    # Compatibility is read-only: historical clocks did not carry owner epochs.
    del root[prior.group_path].attrs[mod._PUBLICATION_OWNER_ATTR]
    for key in (
        mod._PUBLICATION_POLICY_ATTR,
        mod._PUBLICATION_GENERATION_ATTR,
        mod._PUBLICATION_LEASE_ATTR,
    ):
        del _parent(root).attrs[key]
    events_before = list(root.events)
    assert mod.publish_acquisition_frame_clock(root, source) == prior
    assert root.events == events_before
    updated = mod.publish_acquisition_frame_clock(root, _changed(source))
    assert mod.resolve_acquisition_frame_clock(root) == updated


@pytest.mark.parametrize(
    "defect", ["tampered", "ineligible", "selector_mismatch", "root_mismatch"]
)
def test_replay_refuses_inconsistent_selected_clock_without_writes(clock, defect):
    root, source = clock
    prior = mod.publish_acquisition_frame_clock(root, source)
    if defect == "tampered":
        root[f"{prior.group_path}/camera_timestamp_ns"][0] += 1
    elif defect == "ineligible":
        root[prior.group_path].attrs["stage_selector_eligible"] = False
    elif defect == "selector_mismatch":
        _parent(root).attrs["latest"] = "missing"
    else:
        root.attrs["acquisition_frame_clock_sha256"] = "wrong"
    events_before = list(root.events)
    with pytest.raises(mod.AcquisitionFrameClockError):
        mod.publish_acquisition_frame_clock(root, source)
    assert root.events == events_before


def test_candidate_cannot_be_read_while_root_is_being_bound(clock):
    root, source = clock
    refused = []

    def read_during_publication(group, name, value):
        if group.path == "" and name.startswith("acquisition_frame_clock_"):
            with pytest.raises(mod.AcquisitionFrameClockError):
                mod.resolve_acquisition_frame_clock(root)
            refused.append(name)

    root.on_write = read_during_publication
    result = mod.publish_acquisition_frame_clock(root, source)
    root.on_write = None
    assert len(refused) == 6
    assert mod.resolve_acquisition_frame_clock(root) == result


@pytest.mark.parametrize("failure", ["corruption", "root_binding", "final_commit"])
def test_real_import_failure_retry_and_consolidated_timing_consumer(
    tmp_path, monkeypatch, failure
):
    archive = _clock_archive(tmp_path)
    prior = load_provider_recording_timing_authority(archive)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root_before = dict(root.attrs)
    parent_before = dict(_parent(root).attrs)
    recording = tmp_path / "recording_clock_source"
    video = recording / "cams" / "Cam2010093_recording.mp4"
    source = mod.load_acquisition_frame_clock_source(
        recording, camera_id="2010093", video_path=video
    )
    source = _changed(source)
    original_store = mod.store_array
    attrs_class = type(root.attrs)
    original_update = attrs_class.update
    original_set = attrs_class.__setitem__
    fired = False

    def corrupt(run, name, values, **kwargs):
        original_store(run, name, values, **kwargs)
        if name == "camera_timestamp_ns":
            run[name][0] = int(values[0]) + 1

    def update_then_raise(attrs, values):
        nonlocal fired
        original_update(attrs, values)
        if not fired and "acquisition_frame_clock_ref" in values:
            fired = True
            raise OSError("real root binding persisted then raised")

    def refuse_commit(attrs, name, value):
        if name == "stage_selector_eligible" and value is True:
            raise OSError("real eligibility commit refused")
        original_set(attrs, name, value)

    with monkeypatch.context() as fault:
        if failure == "corruption":
            fault.setattr(mod, "store_array", corrupt)
        elif failure == "root_binding":
            fault.setattr(attrs_class, "update", update_then_raise)
        else:
            fault.setattr(attrs_class, "__setitem__", refuse_commit)
        with pytest.raises((OSError, mod.AcquisitionFrameClockError)):
            mod.publish_acquisition_frame_clock(root, source)

    # Reopen after rollback; never repair a cached mutable handle by rewriting it.
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    assert dict(root.attrs) == root_before
    assert dict(_parent(root).attrs) == parent_before
    current = load_provider_recording_timing_authority(archive, use_consolidated=False)
    assert current.sha256 == prior.sha256
    assert load_provider_recording_timing_authority(archive).sha256 == prior.sha256
    failed_name = next(
        name
        for name in _parent(root).group_keys()
        if root[f"{mod.ACQUISITION_FRAME_CLOCK_RUNS_PATH}/{name}"].attrs[
            "palette_run_completion_status"
        ]
        == "failed"
    )
    _assert_failed(root, failed_name)
    failed_attrs = dict(_parent(root)[failed_name].attrs)
    retry = mod.publish_acquisition_frame_clock(root, source)
    assert retry.run_name != failed_name
    assert dict(_parent(root)[failed_name].attrs) == failed_attrs
    zarr.consolidate_metadata(str(archive))
    published = load_provider_recording_timing_authority(archive)
    assert published.record["acquisition_frame_clock"]["run_path"] == retry.group_path
    assert (
        published.record["acquisition_frame_clock"]["record_sha256"]
        == retry.record_sha256
    )
    published.assert_current()
    consolidated_root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    assert mod.resolve_acquisition_frame_clock(consolidated_root) == retry


def test_publisher_refuses_consolidated_mutation_handle(tmp_path):
    archive = _clock_archive(tmp_path)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=True)
    with pytest.raises(mod.AcquisitionFrameClockError, match="use_consolidated=False"):
        mod.publish_acquisition_frame_clock(root, None)


def test_lost_first_root_binding_write_restores_other_owned_values(clock):
    root, source = clock
    prior = mod.publish_acquisition_frame_clock(root, source)
    root_before = copy.deepcopy(dict(root.attrs))
    fired = False

    def lose_ref(group, name, value):
        nonlocal fired
        if not fired and not group.path and name == "acquisition_frame_clock_ref":
            fired = True
            dict.__setitem__(root.attrs, name, prior.group_path)

    root.on_write = lose_ref
    with pytest.raises(mod.AcquisitionFrameClockError, match="did not persist"):
        mod.publish_acquisition_frame_clock(root, _changed(source))
    root.on_write = None
    assert dict(root.attrs) == root_before
    assert mod.resolve_acquisition_frame_clock(root) == prior


def test_payload_tampering_after_root_binding_is_refused(clock):
    root, source = clock
    prior = mod.publish_acquisition_frame_clock(root, source)
    root_before = copy.deepcopy(dict(root.attrs))

    def tamper(group, name, value):
        if not group.path and name == "acquisition_frame_clock_status":
            candidate_path = root.attrs["acquisition_frame_clock_ref"]
            root[f"{candidate_path}/camera_timestamp_ns"][0] += 1

    root.on_write = tamper
    with pytest.raises(mod.AcquisitionFrameClockError, match="bound digest"):
        mod.publish_acquisition_frame_clock(root, _changed(source))
    root.on_write = None
    assert dict(root.attrs) == root_before
    assert mod.resolve_acquisition_frame_clock(root) == prior
