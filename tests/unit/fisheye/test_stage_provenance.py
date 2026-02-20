from __future__ import annotations

import json
from typing import Any

from fisheye.shared.stage_provenance import (
    STAGE_PROVENANCE_CONTRACT_NAME,
    STAGE_PROVENANCE_CONTRACT_VERSION,
    build_stage_provenance,
    get_stage_contract,
    get_stage_git,
    get_stage_provenance,
    write_stage_provenance,
)


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}


def test_get_stage_provenance_adds_contract_defaults_without_mutation() -> None:
    attrs = {
        "provenance": {
            "stage": "refine_eye_masks",
            "parameters": {"min_circularity": 0.77},
            "inputs": {"eye_masks_run": "eye_masks_001"},
        }
    }
    original = dict(attrs["provenance"])

    payload = get_stage_provenance(attrs)

    assert payload["contract"]["name"] == STAGE_PROVENANCE_CONTRACT_NAME
    assert payload["contract"]["version"] == STAGE_PROVENANCE_CONTRACT_VERSION
    assert payload["stage"] == "refine_eye_masks"
    assert attrs["provenance"] == original
    assert "contract" not in attrs["provenance"]


def test_get_stage_provenance_parses_json_string_payload() -> None:
    attrs = {
        "provenance": json.dumps(
            {
                "stage": "refine_detect",
                "contract": {"name": "custom_contract", "version": "2"},
            }
        )
    }
    payload = get_stage_provenance(attrs)
    assert payload["contract"]["name"] == "custom_contract"
    assert payload["contract"]["version"] == 2


def test_get_stage_contract_defaults_when_missing() -> None:
    contract = get_stage_contract({})
    assert contract == {
        "name": STAGE_PROVENANCE_CONTRACT_NAME,
        "version": STAGE_PROVENANCE_CONTRACT_VERSION,
    }


def test_get_stage_git_prefers_provenance_payload() -> None:
    attrs = {
        "provenance": {
            "git": {
                "commit": "a" * 40,
                "branch": "main",
                "is_dirty": True,
            }
        },
        "git_commit": "b" * 40,
        "git_branch": "legacy_branch",
    }
    payload = get_stage_git(attrs)
    assert payload["commit"] == "a" * 40
    assert payload["short"] == "a" * 8
    assert payload["branch"] == "main"
    assert payload["is_dirty"] is True


def test_get_stage_git_falls_back_to_legacy_top_level_attrs() -> None:
    attrs = {
        "provenance": {"git": {"commit": "unknown", "branch": ""}},
        "git_commit_hash": "c" * 40,
        "git_branch": "release",
    }
    payload = get_stage_git(attrs)
    assert payload["commit"] == "c" * 40
    assert payload["short"] == "c" * 8
    assert payload["branch"] == "release"


def test_build_stage_provenance_writes_required_fields() -> None:
    payload = build_stage_provenance(
        stage="refine_keypoints",
        created_at_utc="2026-02-20T00:00:00+00:00",
        parameters={"confidence_threshold": 0.2},
        inputs={"keypoints_run": "keypoints_001"},
        command="scripts/py -m fisheye.refinement.refine_keypoints ...",
        git={"commit_hash": "d" * 40, "branch": "feature/refine"},
    )
    assert payload["contract"]["name"] == STAGE_PROVENANCE_CONTRACT_NAME
    assert payload["contract"]["version"] == STAGE_PROVENANCE_CONTRACT_VERSION
    assert payload["stage"] == "refine_keypoints"
    assert payload["parameters"]["confidence_threshold"] == 0.2
    assert payload["inputs"]["keypoints_run"] == "keypoints_001"
    assert payload["git"]["commit"] == "d" * 40
    assert payload["git"]["branch"] == "feature/refine"


def test_write_stage_provenance_writes_top_level_git_by_default() -> None:
    group = _FakeGroup()
    payload = build_stage_provenance(
        stage="refine_detect",
        created_at_utc="2026-02-20T00:00:00+00:00",
        parameters={"max_gap": 15},
        inputs={"detect_run": "detect_001"},
        git={"commit": "e" * 40, "branch": "main"},
    )

    write_stage_provenance(group, payload)

    assert group.attrs["provenance"]["contract"]["name"] == STAGE_PROVENANCE_CONTRACT_NAME
    assert group.attrs["git_commit"] == "e" * 40
    assert group.attrs["git_branch"] == "main"


def test_write_stage_provenance_can_skip_top_level_git_mirror() -> None:
    group = _FakeGroup()
    payload = build_stage_provenance(
        stage="refine_eye_masks",
        created_at_utc="2026-02-20T00:00:00+00:00",
        parameters={"min_circularity": 0.8},
        inputs={"eye_masks_run": "eye_masks_002"},
        git={"commit": "f" * 40, "branch": "topic"},
    )

    write_stage_provenance(group, payload, include_top_level_git=False)

    assert "provenance" in group.attrs
    assert "git_commit" not in group.attrs
    assert "git_branch" not in group.attrs
