from __future__ import annotations

import pytest

from fisheye.visualization.chaser_provider_display import (
    POSITION_PROVIDER_DISPLAY_POLICY_ID,
    paired_position_provider_display_bindings,
    position_provider_display_binding,
)


def test_explicit_provider_roles_map_to_compact_labels_and_retain_exact_ids() -> None:
    keypoint_id = "opaque-keypoint-authority:" + "a" * 96
    detection_id = "opaque-detection-authority:" + "b" * 96

    bindings = paired_position_provider_display_bindings(
        provider_ids=(keypoint_id, detection_id),
        provider_roles=("keypoint", "detection"),
    )

    assert [binding.role for binding in bindings] == ["keypoint", "detection"]
    assert [binding.display_label for binding in bindings] == [
        "Keypoint-derived position",
        "Detection-derived position",
    ]
    assert [binding.compact_label for binding in bindings] == [
        "Keypoint",
        "Detection",
    ]
    assert [binding.provider_id for binding in bindings] == [
        keypoint_id,
        detection_id,
    ]
    assert bindings[0].provenance_record() == {
        "policy_id": POSITION_PROVIDER_DISPLAY_POLICY_ID,
        "provider_role": "keypoint",
        "display_label": "Keypoint-derived position",
        "compact_label": "Keypoint",
        "provider_id": keypoint_id,
        "role_source": "explicit_validated_caller_contract",
        "provider_id_parsing": "prohibited",
    }


@pytest.mark.parametrize("role", ["", "model", "Keypoint", "unknown"])
def test_provider_display_rejects_unknown_or_noncanonical_role(role: str) -> None:
    with pytest.raises(ValueError, match="provider role"):
        position_provider_display_binding(
            provider_role=role,
            provider_id="opaque-authority",
        )


def test_paired_provider_display_rejects_wrong_order_or_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="keypoint/detection order"):
        paired_position_provider_display_bindings(
            provider_ids=("detection-id", "keypoint-id"),
            provider_roles=("detection", "keypoint"),
        )
    with pytest.raises(ValueError, match="distinct"):
        paired_position_provider_display_bindings(
            provider_ids=("same-id", "same-id"),
            provider_roles=("keypoint", "detection"),
        )
