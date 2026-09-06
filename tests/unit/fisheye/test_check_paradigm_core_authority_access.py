from __future__ import annotations

from pathlib import Path

from scripts import check_paradigm_core_authority_access as mod


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_ratchet_accepts_shared_roster_planning_and_execution(tmp_path: Path) -> None:
    planner = tmp_path / "planner.py"
    _write(
        planner,
        "def plan(roster):\n"
        "    binding = _core_authority_plan_binding(roster)\n"
        "    track = selected_core_track_id_from_roster(roster)\n"
        "    return binding, track\n"
        "def run(task, entry):\n"
        "    _revalidate_core_bundle_selection(task, entry)\n"
        "    validate_core_paradigm_source_dependency(receipt)\n"
        "    _existing_complete_output(path, "
        "expected_core_authority_roster_sha256=digest)\n"
        "    return ('--core-authority-roster', "
        "'--expected-core-authority-roster-sha256', '--core-track-id')\n",
    )

    assert mod.collect_paradigm_core_authority_violations(planner) == []


def test_ratchet_rejects_ordered_pair_and_legacy_cli_selection(tmp_path: Path) -> None:
    planner = tmp_path / "planner.py"
    _write(
        planner,
        "MOTION_BOUT_PAIRS = (('motion', 'bouts'),)\n"
        "def _resolve_motion_bouts(root):\n"
        "    motion = load_provider_track_motion_source_handle(root)\n"
        "    return motion, '--provider-motion-run-path', '--swim-bout-run-name'\n"
        "def execute():\n"
        "    return launch('--track-id', track_id=0)\n",
    )

    reasons = {
        item.reason for item in mod.collect_paradigm_core_authority_violations(planner)
    }

    assert "uses retired core-source selector 'MOTION_BOUT_PAIRS'" in reasons
    assert "defines retired core-source selector '_resolve_motion_bouts'" in reasons
    assert (
        "calls independent core-source resolver "
        "'load_provider_track_motion_source_handle'"
    ) in reasons
    assert "hard-codes implicit core track zero instead of roster selection" in reasons
    assert "invokes retired independent core-source argument '--track-id'" in reasons
    assert any(
        "does not invoke shared core-authority boundary" in item for item in reasons
    )


def test_repository_maintained_chaser_planner_passes_ratchet() -> None:
    assert mod.check_paradigm_core_authority_access() == 0
