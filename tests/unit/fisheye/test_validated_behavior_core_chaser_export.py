from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

import fisheye.analytics_exports.validated_behavior_core_chaser_adapters as subject
from fisheye.analytics_exports.validated_behavior_cohort import (
    ValidatedBehaviorBatchSource,
)


def test_composite_router_reuses_one_context_and_rewrites_enclosing_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []

    class FakeContext:
        constructions = 0

        def __init__(self, *_args: object) -> None:
            type(self).constructions += 1

        def core_plan(self) -> dict[str, str]:
            return {"route": "core"}

        def core_bundle_member(self) -> dict[str, str]:
            return {"route": "core-bundle"}

        def chaser_bundle_member(self) -> dict[str, str]:
            return {"route": "chaser-bundle"}

        @property
        def provenance(self) -> dict[str, str]:
            return {
                "membership_member_sha256": "selected-membership",
                "bundle_set_member_sha256": "selected-bundle-member",
                "bundle_record_sha256": "selected-composite",
            }

    def core_rows(
        plan: Mapping[str, Any],
        _membership: Mapping[str, Any],
        bundle: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], None]:
        calls.append(("core", plan, bundle))
        return (
            [
                {
                    "membership_member_sha256": "core-placeholder",
                    "bundle_set_member_sha256": "core-placeholder",
                    "bundle_record_sha256": "core-report",
                    "value": 1,
                }
            ],
            None,
        )

    def body_rows(
        plan: Mapping[str, Any],
        _membership: Mapping[str, Any],
        bundle: Mapping[str, Any],
    ) -> ValidatedBehaviorBatchSource:
        calls.append(("chaser", plan, bundle))
        return ValidatedBehaviorBatchSource(
            batches=iter(
                [
                    {
                        "membership_member_sha256": ["legacy", "legacy"],
                        "bundle_set_member_sha256": ["legacy", "legacy"],
                        "bundle_record_sha256": ["legacy", "legacy"],
                        "body_source_row_id": [7, -1],
                        "body_source_row_valid": [True, False],
                    }
                ]
            ),
            zero_row_reason=None,
        )

    monkeypatch.setattr(subject, "_CompositeRoutingContext", FakeContext)
    monkeypatch.setattr(subject, "CORE_BEHAVIOR_TABLE_SPECS", {"core_rows": object()})
    monkeypatch.setattr(
        subject,
        "CORE_CHASER_EXTENSION_TABLE_SPECS",
        {"body_relative_samples": object()},
    )
    monkeypatch.setattr(
        subject, "build_core_behavior_row_extractors", lambda: {"core_rows": core_rows}
    )
    monkeypatch.setattr(
        subject,
        "build_phase_c_compact_row_extractors",
        lambda: {"body_relative_samples": body_rows},
    )
    monkeypatch.setattr(subject, "build_phase_b_dense_row_extractors", lambda: {})

    extractors = subject.build_core_chaser_row_extractors()
    plan = {"plan_sha256": "plan", "route": "composite"}
    membership = {"member_sha256": "membership"}
    bundle = {"member_sha256": "bundle"}
    core_result, reason = extractors["core_rows"](plan, membership, bundle)
    body_result = extractors["body_relative_samples"](plan, membership, bundle)
    body_batch = next(iter(body_result.batches))

    assert reason is None
    assert core_result[0]["value"] == 1
    assert core_result[0]["bundle_record_sha256"] == "selected-composite"
    assert body_batch["core_subject_shape_row_index"] == [7, None]
    assert body_batch["membership_member_sha256"] == [
        "selected-membership",
        "selected-membership",
    ]
    assert calls == [
        ("core", {"route": "core"}, {"route": "core-bundle"}),
        ("chaser", plan, {"route": "chaser-bundle"}),
    ]
    assert FakeContext.constructions == 1


def test_composite_router_rejects_duplicate_extension_projectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projector = lambda *_args: ([], None)
    monkeypatch.setattr(subject, "CORE_BEHAVIOR_TABLE_SPECS", {})
    monkeypatch.setattr(
        subject,
        "CORE_CHASER_EXTENSION_TABLE_SPECS",
        {"duplicate": object()},
    )
    monkeypatch.setattr(subject, "build_core_behavior_row_extractors", lambda: {})
    monkeypatch.setattr(
        subject,
        "build_phase_c_compact_row_extractors",
        lambda: {"duplicate": projector},
    )
    monkeypatch.setattr(
        subject,
        "build_phase_b_dense_row_extractors",
        lambda: {"duplicate": projector},
    )

    with pytest.raises(
        subject.CoreChaserExportAdapterError,
        match="exactly one installed projector",
    ):
        subject.build_core_chaser_row_extractors()
