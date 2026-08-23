from __future__ import annotations

from fisheye.analysis_workflows.provider_chaser_position_suite_publication import (
    publish_provider_chaser_position_suite_run,
)
from fisheye.utils.provider_chaser_position_suite_readiness import (
    build_readiness_receipt,
)
from tests.unit.fisheye.test_provider_chaser_position_suite_publication import _plan


def test_readiness_separates_complete_candidate_from_production_authority(
    tmp_path,
) -> None:
    archive, _report, plan = _plan(tmp_path)
    publish_provider_chaser_position_suite_run(plan, scratch_root=tmp_path / "scratch")

    receipt = build_readiness_receipt(
        archive,
        run_name=plan.run_name,
        expected_recording_id="recording-fixture",
    )
    payload = receipt["payload"]

    assert receipt["status"] == "candidate_complete_production_blocked"
    assert len(receipt["payload_digest"]) == 64
    assert payload["scientific_candidate_state"] == "complete"
    assert payload["scientific_scope"] == "position_only"
    assert payload["source_verification_mode"] == "bounded_publication"
    assert payload["direct_consolidated_metadata_equivalent"] is True
    assert payload["selector_eligible"] is False
    assert payload["selection"] == "none"
    assert payload["production_authority"] is False
    assert payload["production_readiness"] == "blocked"
    assert payload["registry_projection_eligible"] is False
    assert payload["registry_update"] is False
    assert payload["production_blockers"] == [
        "required_ci_not_bound",
        "production_selector_not_activated",
    ]
