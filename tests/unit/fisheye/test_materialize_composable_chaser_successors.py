from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fisheye.utils import materialize_composable_chaser_successors as operator


@dataclass(frozen=True)
class _Prepared:
    label: str

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "payload_digest": (self.label[0] * 64),
            "dimensions": {"rows": 3},
        }


def _handle(label: str) -> SimpleNamespace:
    return SimpleNamespace(
        recording_id="recording-1",
        run_name=f"{label}-v1",
        run_path=f"analysis/{label}_runs/{label}-v1",
        manifest_sha256=(label[0] * 64),
    )


def _install_ready_fakes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setattr(
        operator,
        "load_chaser_relative_frame_source_handle",
        lambda *args, **kwargs: _handle("relative"),
    )
    monkeypatch.setattr(
        operator,
        "load_protocol_semantic_chaser_selection_source_handle",
        lambda *args, **kwargs: _handle("semantic"),
    )
    monkeypatch.setattr(
        operator,
        "load_provider_track_motion_source_handle",
        lambda *args, **kwargs: _handle("motion"),
    )
    monkeypatch.setattr(
        operator,
        "load_eye_gaze_source_handle",
        lambda *args, **kwargs: _handle("eye"),
    )
    monkeypatch.setattr(
        operator,
        "load_composable_chaser_successor_source_handle",
        lambda *args, **kwargs: _handle("radial"),
    )
    prepared_order: list[str] = []

    def prepared(label: str):
        def build(*args: Any, **kwargs: Any) -> _Prepared:
            prepared_order.append(label)
            return _Prepared(label)

        return build

    monkeypatch.setattr(
        operator,
        "prepare_controller_trial_successor_from_handles",
        prepared("controller"),
    )
    monkeypatch.setattr(
        operator,
        "prepare_generalized_bout_response_successor_from_handles",
        prepared("bout"),
    )
    monkeypatch.setattr(
        operator,
        "prepare_escape_freeze_successor_from_handles",
        prepared("escape"),
    )
    monkeypatch.setattr(
        operator,
        "prepare_gaze_tracking_successor_from_handles",
        prepared("gaze"),
    )

    kind_by_label = {
        "controller": "controller_chase_trials",
        "bout": "generalized_chaser_bout_response",
        "escape": "chaser_escape_freeze",
        "gaze": "chaser_gaze_tracking",
    }

    def plan(_archive: Path, *, run_name: str, prepared: _Prepared):
        kind = kind_by_label[prepared.label]
        return SimpleNamespace(
            successor_kind=kind,
            run_path=f"analysis/{kind}_runs/{run_name}",
        )

    monkeypatch.setattr(
        operator,
        "build_composable_chaser_successor_publication_plan",
        plan,
    )
    return prepared_order


def _run(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "run_name": "operator_trial_v1",
        "relative_frame_run": "relative-v1",
        "semantic_selection_run": "semantic-v1",
        "provider_motion_run_path": "analysis/motion_runs/motion-v1",
        "swim_bout_run_name": "bouts-v1",
        "track_id": 0,
        "expected_recording_id": "recording-1",
        "eye_run_name": "eye-v1",
        "eye_convention_receipt": {"review": "accepted"},
        "radial_run_name": "radial-v1",
    }
    values.update(overrides)
    return operator.run_composable_chaser_successors(tmp_path, **values)


def test_dry_run_prepares_all_modules_in_dependency_order_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_order = _install_ready_fakes(monkeypatch)
    publish_calls: list[str] = []
    monkeypatch.setattr(
        operator,
        "publish_composable_chaser_successor_run",
        lambda plan, **kwargs: publish_calls.append(plan.successor_kind),
    )

    result = _run(tmp_path)

    assert result["status"] == "planned_no_writes"
    assert prepared_order == ["controller", "bout", "escape", "gaze"]
    assert [row["status"] for row in result["modules"]] == ["planned"] * 4
    assert publish_calls == []
    assert result["selector_eligible"] is False
    assert result["production_authority"] is False


def test_escape_request_expands_and_publishes_exact_dependency_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_order = _install_ready_fakes(monkeypatch)
    publish_calls: list[str] = []

    def publish(plan: Any, **kwargs: Any) -> dict[str, Any]:
        publish_calls.append(plan.successor_kind)
        return {
            "status": "published_selector_ineligible",
            "selector_eligible": False,
        }

    monkeypatch.setattr(
        operator,
        "publish_composable_chaser_successor_run",
        publish,
    )

    result = _run(
        tmp_path,
        modules=(operator.ESCAPE_FREEZE,),
        apply=True,
        eye_run_name=None,
        eye_convention_receipt=None,
    )

    assert result["status"] == "published_selector_ineligible"
    assert prepared_order == ["controller", "bout", "escape"]
    assert publish_calls == [
        "controller_chase_trials",
        "generalized_chaser_bout_response",
        "chaser_escape_freeze",
    ]
    assert [row["explicitly_requested"] for row in result["modules"]] == [
        False,
        False,
        True,
    ]


def test_missing_reviewed_eye_source_blocks_only_gaze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ready_fakes(monkeypatch)
    published: list[str] = []

    def publish(plan: Any, **kwargs: Any) -> dict[str, Any]:
        published.append(plan.successor_kind)
        return {"status": "published_selector_ineligible"}

    monkeypatch.setattr(
        operator,
        "publish_composable_chaser_successor_run",
        publish,
    )

    result = _run(
        tmp_path,
        apply=True,
        eye_run_name=None,
        eye_convention_receipt=None,
    )

    assert result["status"] == "published_partial"
    assert published == [
        "controller_chase_trials",
        "generalized_chaser_bout_response",
        "chaser_escape_freeze",
    ]
    gaze = result["modules"][-1]
    assert gaze["status"] == "blocked"
    assert gaze["blocking_sources"] == ["eye_gaze"]
    assert result["sources"]["eye_gaze"]["reason_code"] == (
        "reviewed_eye_source_not_supplied"
    )


def test_semantic_loader_failure_is_structured_and_blocks_all_dependents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ready_fakes(monkeypatch)
    monkeypatch.setattr(
        operator,
        "load_protocol_semantic_chaser_selection_source_handle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("materialized semantic authority unavailable")
        ),
    )

    result = _run(tmp_path)

    assert result["status"] == "blocked_no_products"
    assert result["sources"]["semantic_selection"]["status"] == "blocked"
    assert (
        "semantic authority unavailable"
        in result["sources"]["semantic_selection"]["error"]["message"]
    )
    assert [row["reason_code"] for row in result["modules"]] == [
        "source_handle_unavailable",
        "dependency_unavailable",
        "dependency_unavailable",
        "source_handle_unavailable",
    ]


def test_core_mode_binds_one_roster_motion_and_bout_without_legacy_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_order = _install_ready_fakes(monkeypatch)
    roster = {"record_sha256": "a" * 64}
    bound = object()
    core_handle = SimpleNamespace(
        canonical_bout_source=SimpleNamespace(
            binding={
                "run_name": "core-bouts-v1",
                "run_path": "analysis/swim_bout_runs/core-bouts-v1",
                "payload_sha256": "b" * 64,
            }
        ),
        core_authority_roster_sha256="a" * 64,
    )
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        operator,
        "bind_core_motion_and_bouts_from_roster",
        lambda value: bound if value is roster else pytest.fail("wrong roster"),
    )

    def bind_handle(value: object, **kwargs: object) -> object:
        calls.append((value, kwargs))
        return core_handle

    monkeypatch.setattr(operator, "bind_core_motion_track_source_handle", bind_handle)
    monkeypatch.setattr(
        operator,
        "load_provider_track_motion_source_handle",
        lambda *args, **kwargs: pytest.fail("legacy motion fallback must not run"),
    )

    result = _run(
        tmp_path,
        modules=(operator.ESCAPE_FREEZE,),
        provider_motion_run_path=None,
        swim_bout_run_name=None,
        core_authority_roster=roster,
        core_track_id=7,
        eye_run_name=None,
        eye_convention_receipt=None,
    )

    assert result["status"] == "planned_no_writes"
    assert prepared_order == ["controller", "bout", "escape"]
    assert len(calls) == 1
    assert calls[0][0] is bound
    assert calls[0][1]["track_id"] == 7
    assert result["sources"]["core_motion"]["status"] == "ready"
    assert result["sources"]["swim_bouts"]["run_name"] == "core-bouts-v1"
    assert "provider_motion" not in result["sources"]


def test_core_mode_rejects_independent_provider_inputs(tmp_path: Path) -> None:
    with pytest.raises(
        operator.ComposableChaserSuccessorOperatorError,
        match="cannot accept provider-motion or independent bout",
    ):
        _run(
            tmp_path,
            core_authority_roster={"record_sha256": "a" * 64},
            core_track_id=7,
        )
