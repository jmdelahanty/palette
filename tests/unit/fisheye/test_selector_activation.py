from __future__ import annotations

import uuid
from typing import Any

import pytest

from fisheye.shared import selector_activation as mod


class _FakeGroup:
    def __init__(
        self,
        path: str = "",
        *,
        registry: dict[str, "_FakeGroup"] | None = None,
    ) -> None:
        self.path = path.strip("/")
        self.attrs: dict[str, Any] = {}
        self._registry = registry if registry is not None else {}
        self._registry[self.path] = self

    def require_group(self, path: str) -> "_FakeGroup":
        relative = str(path).strip("/")
        full = relative if not self.path else f"{self.path}/{relative}"
        current = ""
        for part in full.split("/"):
            current = part if not current else f"{current}/{part}"
            if current not in self._registry:
                _FakeGroup(current, registry=self._registry)
        return self._registry[full]

    def __getitem__(self, path: str) -> "_FakeGroup":
        relative = str(path).strip("/")
        full = relative if not self.path or "/" in relative else f"{self.path}/{relative}"
        return self._registry[full]

    def get(self, path: str, default: Any = None) -> Any:
        try:
            return self[path]
        except KeyError:
            return default


@pytest.fixture(autouse=True)
def _fake_archive_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    archive = object()
    monkeypatch.setattr(mod, "archive_identity", lambda _group: archive)
    monkeypatch.setattr(mod, "canonical_node_path", lambda group: group.path)


def _candidate(
    root: _FakeGroup,
    name: str,
) -> tuple[_FakeGroup, _FakeGroup, str]:
    parent = root.require_group("analysis/example_runs")
    run = parent.require_group(name)
    owner = str(uuid.uuid4())
    run.attrs.update(
        {
            "publication_owner_uuid": owner,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "proof": "sealed",
        }
    )
    return parent, run, owner


def _activate(
    root: _FakeGroup,
    parent: _FakeGroup,
    run: _FakeGroup,
    name: str,
    *,
    attr_writer=mod.write_activation_attr,
    expected_owner_uuid: str | None = None,
) -> None:
    owner_uuid = (
        str(run.attrs["publication_owner_uuid"])
        if expected_owner_uuid is None
        else expected_owner_uuid
    )
    mod.activate_selector_eligible_run(
        root,
        parent,
        run,
        parent_path="analysis/example_runs",
        run_path=f"analysis/example_runs/{name}",
        run_name=name,
        owner_attr="publication_owner_uuid",
        expected_owner_uuid=owner_uuid,
        policy_attr="publication_policy",
        generation_attr="publication_generation",
        lease_attr="publication_lease",
        policy="owner_generation_guarded_selectors_then_eligibility_v1",
        lease_schema_id="fixture.publication_lease",
        proof_loader=lambda: (
            run.attrs["proof"],
            run.attrs["palette_run_completion_status"],
            run.attrs["stage_selector_eligible"],
        ),
        attr_writer=attr_writer,
    )


def test_activation_commits_eligibility_last_and_advances_generations() -> None:
    root = _FakeGroup()
    parent, first, first_owner = _candidate(root, "first")

    _activate(root, parent, first, "first")

    assert first.attrs["stage_selector_eligible"] is True
    assert parent.attrs["latest"] == "first"
    assert parent.attrs["latest_complete"] == "first"
    assert parent.attrs["publication_generation"] == 1
    assert parent.attrs["publication_lease"]["owner_uuid"] == first_owner

    _parent, second, second_owner = _candidate(root, "second")
    _activate(root, parent, second, "second")

    assert second.attrs["stage_selector_eligible"] is True
    assert parent.attrs["publication_generation"] == 2
    assert parent.attrs["publication_lease"]["owner_uuid"] == second_owner


def test_activation_rejects_same_path_replacement_with_foreign_owner() -> None:
    root = _FakeGroup()
    parent, stale_run, original_owner = _candidate(root, "candidate")
    parent.attrs["latest"] = "prior"
    parent.attrs["latest_complete"] = "prior"

    del root._registry[stale_run.path]
    replacement = _FakeGroup(stale_run.path, registry=root._registry)
    replacement.attrs.update(
        {
            "publication_owner_uuid": str(uuid.uuid4()),
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "proof": "sealed",
        }
    )

    with pytest.raises(
        mod.SelectorActivationError,
        match="expected publication owner",
    ):
        _activate(
            root,
            parent,
            stale_run,
            "candidate",
            expected_owner_uuid=original_owner,
        )

    assert parent.attrs["latest"] == "prior"
    assert parent.attrs["latest_complete"] == "prior"
    assert "publication_generation" not in parent.attrs
    assert replacement.attrs["stage_selector_eligible"] is False
    assert stale_run.attrs["stage_selector_eligible"] is False


def test_activation_preserves_alien_parent_mutation_and_disarms_candidate() -> None:
    root = _FakeGroup()
    parent, run, _owner = _candidate(root, "candidate")
    parent.attrs["latest"] = "prior"
    parent.attrs["latest_complete"] = "prior"
    injected = False

    def hostile_write(attrs, name, value):
        nonlocal injected
        mod.write_activation_attr(attrs, name, value)
        if name == "latest_complete" and not injected:
            injected = True
            parent.attrs["latest"] = "alien"

    with pytest.raises(mod.SelectorActivationError, match="Concurrent parent mutation"):
        _activate(
            root,
            parent,
            run,
            "candidate",
            attr_writer=hostile_write,
        )

    assert parent.attrs["latest"] == "alien"
    assert parent.attrs["latest_complete"] == "prior"
    assert "publication_generation" not in parent.attrs
    assert "publication_policy" not in parent.attrs
    assert "publication_lease" not in parent.attrs
    assert run.attrs["stage_selector_eligible"] is False


@pytest.mark.parametrize("takeover_attr", ["publication_lease", "publication_generation"])
def test_activation_restores_own_partial_selector_after_epoch_takeover(
    takeover_attr: str,
) -> None:
    root = _FakeGroup()
    parent, run, _owner = _candidate(root, "candidate")
    parent.attrs["latest"] = "prior"
    parent.attrs["latest_complete"] = "prior"
    alien = (
        {
            "schema_id": "alien.lease",
            "schema_version": 1,
            "owner_uuid": str(uuid.uuid4()),
        }
        if takeover_attr == "publication_lease"
        else 99
    )

    def hostile_write(attrs, name, value):
        mod.write_activation_attr(attrs, name, value)
        if name == "latest_complete":
            parent.attrs[takeover_attr] = alien

    with pytest.raises(mod.SelectorActivationError, match="Concurrent parent mutation"):
        _activate(
            root,
            parent,
            run,
            "candidate",
            attr_writer=hostile_write,
        )

    assert parent.attrs["latest_complete"] == "prior"
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs[takeover_attr] == alien
    assert run.attrs["stage_selector_eligible"] is False


def test_activation_rejects_an_uncommitted_prior_generation() -> None:
    root = _FakeGroup()
    parent, first, _owner = _candidate(root, "first")
    _activate(root, parent, first, "first")
    first.attrs["stage_selector_eligible"] = False
    _parent, second, _second_owner = _candidate(root, "second")

    with pytest.raises(
        mod.SelectorActivationError,
        match="uncommitted child",
    ):
        _activate(root, parent, second, "second")

    assert parent.attrs["latest"] == "first"
    assert parent.attrs["latest_complete"] == "first"
    assert second.attrs["stage_selector_eligible"] is False


def test_activation_rejects_a_malformed_prior_lease() -> None:
    root = _FakeGroup()
    parent, first, _owner = _candidate(root, "first")
    _activate(root, parent, first, "first")
    malformed = dict(parent.attrs["publication_lease"])
    malformed["schema_id"] = "alien.schema"
    parent.attrs["publication_lease"] = malformed
    _parent, second, _second_owner = _candidate(root, "second")

    with pytest.raises(
        mod.SelectorActivationError,
        match="not one exact committed epoch",
    ):
        _activate(root, parent, second, "second")

    assert parent.attrs["publication_lease"] == malformed
    assert parent.attrs["latest"] == "first"
    assert second.attrs["stage_selector_eligible"] is False


def test_final_write_then_raise_is_committed_despite_newer_parent_epoch() -> None:
    root = _FakeGroup()
    parent, run, _owner = _candidate(root, "candidate")
    successor_owner = str(uuid.uuid4())

    def write_then_supersede(attrs, name, value):
        mod.write_activation_attr(attrs, name, value)
        if name != "stage_selector_eligible":
            return
        successor = parent.require_group("successor")
        successor.attrs.update(
            {
                "publication_owner_uuid": successor_owner,
                "palette_run_completion_status": "complete",
                "stage_selector_eligible": True,
                "proof": "sealed",
            }
        )
        parent.attrs.update(
            {
                "latest_complete": "successor",
                "latest": "successor",
                "publication_policy": (
                    "owner_generation_guarded_selectors_then_eligibility_v1"
                ),
                "publication_generation": 2,
                "publication_lease": {
                    "schema_id": "fixture.publication_lease",
                    "schema_version": 1,
                    "policy": (
                        "owner_generation_guarded_selectors_then_eligibility_v1"
                    ),
                    "owner_uuid": successor_owner,
                    "publication_owner": successor_owner,
                    "run_path": "analysis/example_runs/successor",
                    "run_name": "successor",
                    "base_generation": 1,
                    "next_generation": 2,
                    "selector_attrs": ["latest_complete", "latest"],
                },
            }
        )
        raise RuntimeError("persisted final write interrupted after supersession")

    _activate(
        root,
        parent,
        run,
        "candidate",
        attr_writer=write_then_supersede,
    )

    assert run.attrs["stage_selector_eligible"] is True
    assert parent.attrs["latest"] == "successor"
    assert parent.attrs["latest_complete"] == "successor"
    assert parent.attrs["publication_generation"] == 2


def test_deferred_commit_rechecks_parent_guards_and_rolls_back_own_epoch() -> None:
    root = _FakeGroup()
    parent, run, _owner = _candidate(root, "candidate")
    parent.attrs["latest"] = "prior"
    parent.attrs["latest_complete"] = "prior"
    activation = mod.activate_selector_eligible_run(
        root,
        parent,
        run,
        parent_path="analysis/example_runs",
        run_path="analysis/example_runs/candidate",
        run_name="candidate",
        owner_attr="publication_owner_uuid",
        expected_owner_uuid=_owner,
        policy_attr="publication_policy",
        generation_attr="publication_generation",
        lease_attr="publication_lease",
        policy="owner_generation_guarded_selectors_then_eligibility_v1",
        lease_schema_id="fixture.publication_lease",
        proof_loader=lambda: (
            run.attrs["proof"],
            run.attrs["palette_run_completion_status"],
            run.attrs["stage_selector_eligible"],
        ),
        defer_eligibility=True,
    )
    assert isinstance(activation, mod.DeferredSelectorActivation)
    assert run.attrs["stage_selector_eligible"] is False
    parent.attrs["latest_pending"] = "alien-pending"

    with pytest.raises(
        mod.SelectorActivationError,
        match="Concurrent parent mutation",
    ):
        mod.commit_deferred_selector_activation(
            activation,
            root=root,
            parent_group=parent,
            run_group=run,
            proof_loader=lambda: (
                run.attrs["proof"],
                run.attrs["palette_run_completion_status"],
                run.attrs["stage_selector_eligible"],
            ),
        )

    assert parent.attrs["latest"] == "prior"
    assert parent.attrs["latest_complete"] == "prior"
    assert parent.attrs["latest_pending"] == "alien-pending"
    assert "publication_generation" not in parent.attrs
    assert "publication_policy" not in parent.attrs
    assert "publication_lease" not in parent.attrs
    assert run.attrs["stage_selector_eligible"] is False


def test_deferred_commit_rebinds_to_fresh_handles_and_preserves_new_attrs() -> None:
    stale_root = _FakeGroup()
    stale_parent, stale_run, owner = _candidate(stale_root, "candidate")
    activation = mod.activate_selector_eligible_run(
        stale_root,
        stale_parent,
        stale_run,
        parent_path="analysis/example_runs",
        run_path="analysis/example_runs/candidate",
        run_name="candidate",
        owner_attr="publication_owner_uuid",
        expected_owner_uuid=owner,
        policy_attr="publication_policy",
        generation_attr="publication_generation",
        lease_attr="publication_lease",
        policy="owner_generation_guarded_selectors_then_eligibility_v1",
        lease_schema_id="fixture.publication_lease",
        proof_loader=lambda: (
            stale_run.attrs["proof"],
            stale_run.attrs["palette_run_completion_status"],
            stale_run.attrs["stage_selector_eligible"],
        ),
        defer_eligibility=True,
    )
    assert isinstance(activation, mod.DeferredSelectorActivation)

    fresh_root = _FakeGroup()
    fresh_parent = fresh_root.require_group("analysis/example_runs")
    fresh_parent.attrs.update(stale_parent.attrs)
    fresh_run = fresh_parent.require_group("candidate")
    fresh_run.attrs.update(stale_run.attrs)
    fresh_run.attrs["metadata_written_after_receipt"] = {
        "final_validation": {"valid": True}
    }

    mod.commit_deferred_selector_activation(
        activation,
        root=fresh_root,
        parent_group=fresh_parent,
        run_group=fresh_run,
        proof_loader=lambda: (
            fresh_run.attrs["proof"],
            fresh_run.attrs["palette_run_completion_status"],
            fresh_run.attrs["stage_selector_eligible"],
        ),
    )

    assert fresh_run.attrs["stage_selector_eligible"] is True
    assert fresh_run.attrs["metadata_written_after_receipt"] == {
        "final_validation": {"valid": True}
    }
    assert stale_run.attrs["stage_selector_eligible"] is False
