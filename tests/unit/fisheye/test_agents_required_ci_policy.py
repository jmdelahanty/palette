from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
AGENT_INSTRUCTIONS_PATH = REPOSITORY_ROOT / "AGENTS.md"
POLICY_HEADING = "## Required CI and Integration Rule"
POLICY_MARKER = "<!-- required-ci-integration-contract:v2 -->"


def _required_ci_policy() -> str:
    text = AGENT_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    assert text.count(POLICY_HEADING) == 1
    section = text.split(POLICY_HEADING, maxsplit=1)[1]
    section = section.split("\n## ", maxsplit=1)[0]
    return " ".join(section.split())


def test_required_ci_policy_blocks_merge_integration_and_promotion() -> None:
    policy = _required_ci_policy()

    assert POLICY_MARKER in policy
    assert "must not be merged" in policy
    assert "integrated into another merge candidate" in policy
    assert "fast-forwarded into the shared `/groups` checkout" in policy
    assert "used to activate a production selector/publication" in policy
    assert "every required CI check" in policy
    assert "completed successfully" in policy


def test_required_ci_policy_distinguishes_incomplete_handoff() -> None:
    policy = _required_ci_policy()

    assert "explicitly incomplete work" in policy
    assert "every failing or unrun check" in policy
    assert "not authorization to merge, integrate, promote" in policy
    assert "reported as not merge-ready" in policy


def test_required_ci_policy_defines_fail_closed_same_tree_integration() -> None:
    policy = _required_ci_policy()

    assert "successful full CI run on the exact integrated commit" in policy
    assert "successful `main-integration` same-tree gate" in policy
    assert "first parent is an ancestor" in policy
    assert "landed tree is identical" in policy
    assert "attested tested PR tree" in policy
    assert "active strict required-check ruleset has no bypass actors" in policy
    assert "requires a successful manual full CI run" in policy
