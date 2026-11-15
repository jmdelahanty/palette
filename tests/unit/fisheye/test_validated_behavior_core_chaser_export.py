from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import fisheye.analysis_workflows.core_chaser_composite_bundle as composite_bundle_module
import fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication as semantic_selection_module
import fisheye.analytics_exports.validated_behavior_adapters as compact_adapters
import fisheye.analytics_exports.validated_behavior_core_chaser_adapters as subject
from fisheye.analysis_workflows.core_chaser_composite_bundle import (
    CORE_CHASER_BUNDLE_ADAPTER_ID,
)
from fisheye.analytics_exports.validated_behavior_cohort import (
    ValidatedBehaviorBatchSource,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

_SEMANTIC_ROLES = ("chaser_pre", "chaser_training", "chaser_post")


def _semantic_manifest() -> dict[str, Any]:
    role_records = [
        {
            "role": role,
            "source_window_id": index,
            "source_label": ("pre_event", "training_event", "post_event")[index],
            "selected_start_frame": (10, 20, 30)[index],
            "selected_end_frame_exclusive": (20, 30, 40)[index],
            "source_interval_sha256": str(index + 1) * 64,
            "protocol_semantic_step_index": 0,
            "terminal_frame_excluded_pending_step_end_contract": False,
        }
        for index, role in enumerate(_SEMANTIC_ROLES)
    ]
    selection_identity = {
        "protocol_semantic_hash": "sha256:" + "a" * 64,
        "palette_computed_trial_index_sha256": "sha256:" + "b" * 64,
        "trial_index_integrity_status": "producer_asserted",
        "standalone_solid_black_status": "not_applicable_protocol_has_no_standalone_solid_black",
    }
    body = {
        "schema_id": "palette.protocol_semantic_chaser_selection_run",
        "schema_version": 1,
        "recording_id": "recording-1",
        "run_name": "semantic-exact-v1",
        "run_path": (
            "analysis/protocol_semantic_chaser_selection_runs/semantic-exact-v1"
        ),
        "role_order": list(_SEMANTIC_ROLES),
        "role_records": role_records,
        "selection_identity": selection_identity,
        "selection_identity_sha256": canonical_json_sha256(selection_identity),
        "source_epoch_selection": {"selection_sha256": "c" * 64},
        "protocol_evidence": {"step_end_interval_semantics": "producer_half_open_v1"},
        "selector_eligible": False,
        "production_authority": False,
    }
    return {**body, "payload_digest": canonical_json_sha256(body)}


def _semantic_handle(
    manifest: Mapping[str, Any],
    *,
    analysis_zarr: Path = Path("/tmp/fixture-analysis.zarr"),
    receipt_path: Path = Path("/tmp/fixture-semantic-receipt.json"),
    receipt_sha256: str = "d" * 64,
) -> Any:
    return semantic_selection_module.ProtocolSemanticChaserSelectionSourceHandle(
        analysis_zarr=analysis_zarr,
        run_name=manifest["run_name"],
        run_path=manifest["run_path"],
        recording_id=manifest["recording_id"],
        manifest=manifest,
        arrays={},
        metadata_equivalence={
            "verification_mode": "exact_immutable_child_receipt_v1",
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha256,
        },
        _seal=semantic_selection_module._SOURCE_HANDLE_SEAL,  # noqa: SLF001
    )


def _composite_semantic_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundled_manifest: Mapping[str, Any] | None = None,
    handle_manifest: Mapping[str, Any] | None = None,
    handle_receipt_sha256: str = "d" * 64,
) -> tuple[
    Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], list[dict[str, Any]]
]:
    bundled = deepcopy(dict(bundled_manifest or _semantic_manifest()))
    current = deepcopy(dict(handle_manifest or bundled))
    bundle_path = tmp_path / "composite_bundle.json"
    bundle_path.write_text("{}\n", encoding="utf-8")
    receipt_path = (tmp_path / "semantic_receipt.json").resolve()
    file_sha256 = "f" * 64
    record_sha256 = "9" * 64
    seal = {
        "run_path": bundled["run_path"],
        "manifest_sha256": canonical_json_sha256(bundled),
        "payload_digest": bundled["payload_digest"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": "d" * 64,
    }
    composite = {
        "schema_id": "palette.analysis.core_chaser_composite_bundle",
        "schema_version": 1,
        "method_id": "one_core_roster_plus_exact_chaser_extension_v1",
        "status": "complete_selector_ineligible_receipt_composition",
        "record_sha256": record_sha256,
        "recording_id": "recording-1",
        "analysis_zarr": str((tmp_path / "analysis.zarr").resolve()),
        "source_bindings": {
            "semantic_epochs": {
                "binding_type": "exact_protocol_semantic_selection_v1",
                "source": bundled,
                "sealed_by": seal,
            }
        },
        "scientific_child_bindings": {
            "semantic_epochs": {
                "run_path": bundled["run_path"],
                "manifest_sha256": seal["manifest_sha256"],
                "payload_digest": seal["payload_digest"],
                "receipt_path": seal["receipt_path"],
                "receipt_sha256": seal["receipt_sha256"],
            }
        },
        "internal_capabilities": {
            "semantic_epochs": {
                "state": "complete",
                "reason_code": None,
                "detail": None,
                "binding_scope": "scientific_child_bindings",
                "binding_key": "semantic_epochs",
            }
        },
    }
    bundle_binding = {
        "adapter_id": CORE_CHASER_BUNDLE_ADAPTER_ID,
        "path": str(bundle_path.resolve()),
        "file_sha256": file_sha256,
        "record_sha256": record_sha256,
        "schema_id": composite["schema_id"],
        "schema_version": composite["schema_version"],
        "method_id": composite["method_id"],
        "status": composite["status"],
        "binding_inventory_sha256": "8" * 64,
    }
    plan = {"plan_sha256": "7" * 64, "export_run_id": "export-1"}
    membership = {
        "recording_id": composite["recording_id"],
        "analysis_zarr": composite["analysis_zarr"],
        "member_sha256": "6" * 64,
    }
    member = {
        "recording_id": composite["recording_id"],
        "analysis_zarr": composite["analysis_zarr"],
        "bundle_state": "complete",
        "bundle": bundle_binding,
        "capabilities": {},
        "member_sha256": "5" * 64,
    }
    calls: list[dict[str, Any]] = []

    def read_composite(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        return composite

    def load_semantic(*_args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return _semantic_handle(
            current,
            analysis_zarr=Path(composite["analysis_zarr"]),
            receipt_path=receipt_path,
            receipt_sha256=handle_receipt_sha256,
        )

    monkeypatch.setattr(subject, "sha256_file", lambda _path: file_sha256)
    monkeypatch.setattr(compact_adapters, "sha256_file", lambda _path: file_sha256)
    monkeypatch.setattr(subject, "read_core_chaser_composite_bundle", read_composite)
    monkeypatch.setattr(
        composite_bundle_module,
        "read_core_chaser_composite_bundle",
        read_composite,
    )
    monkeypatch.setattr(
        semantic_selection_module,
        "load_protocol_semantic_chaser_selection_source_handle",
        load_semantic,
    )
    return plan, membership, member, calls


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


def test_composite_router_projects_semantic_epochs_from_exact_sealed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, membership, member, calls = _composite_semantic_fixture(tmp_path, monkeypatch)

    rows, reason = subject.build_core_chaser_row_extractors()["semantic_epochs"](
        plan, membership, member
    )

    assert reason is None
    assert [row["epoch_window_id"] for row in rows] == [0, 1, 2]
    assert [row["analysis_role"] for row in rows] == list(_SEMANTIC_ROLES)
    assert [row["start_frame"] for row in rows] == [10, 20, 30]
    assert [row["end_frame_exclusive"] for row in rows] == [20, 30, 40]
    assert {row["protocol_semantic_step_ref"] for row in rows} == {
        "protocol_semantic_snapshot@recipe.steps[0]"
    }
    assert {row["membership_member_sha256"] for row in rows} == {"6" * 64}
    assert {row["bundle_set_member_sha256"] for row in rows} == {"5" * 64}
    assert {row["bundle_record_sha256"] for row in rows} == {"9" * 64}
    assert calls == [
        {
            "run_name": "semantic-exact-v1",
            "expected_recording_id": "recording-1",
            "direct_validation_receipt": (tmp_path / "semantic_receipt.json").resolve(),
        }
    ]


def test_composite_semantic_projection_rejects_substituted_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _semantic_manifest()
    substituted = deepcopy(current)
    substituted["role_records"][1]["selected_start_frame"] = 21
    body = {key: value for key, value in substituted.items() if key != "payload_digest"}
    substituted["payload_digest"] = canonical_json_sha256(body)
    plan, membership, member, _ = _composite_semantic_fixture(
        tmp_path,
        monkeypatch,
        bundled_manifest=substituted,
        handle_manifest=current,
    )

    with pytest.raises(
        subject.CoreChaserExportAdapterError,
        match="differs from the sealed composite source",
    ):
        subject.build_core_chaser_row_extractors()["semantic_epochs"](
            plan, membership, member
        )


def test_composite_semantic_projection_rejects_wrong_receipt_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, membership, member, _ = _composite_semantic_fixture(
        tmp_path,
        monkeypatch,
        handle_receipt_sha256="e" * 64,
    )

    with pytest.raises(
        subject.CoreChaserExportAdapterError,
        match="receipt differs from the composite seal",
    ):
        subject.build_core_chaser_row_extractors()["semantic_epochs"](
            plan, membership, member
        )


def test_composite_semantic_projection_rejects_mixed_binding_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, membership, member, _ = _composite_semantic_fixture(tmp_path, monkeypatch)
    original = subject.read_core_chaser_composite_bundle

    def mixed(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        bundle = deepcopy(dict(original(*args, **kwargs)))
        bundle["source_bindings"]["semantic_epochs"][
            "binding_type"
        ] = "unsupported_mixed_semantic_v1"
        return bundle

    monkeypatch.setattr(subject, "read_core_chaser_composite_bundle", mixed)
    monkeypatch.setattr(
        composite_bundle_module,
        "read_core_chaser_composite_bundle",
        mixed,
    )

    with pytest.raises(
        subject.CoreChaserExportAdapterError,
        match="unsupported semantic-epoch source binding",
    ):
        subject.build_core_chaser_row_extractors()["semantic_epochs"](
            plan, membership, member
        )


def test_phase_c_semantic_epoch_projection_remains_byte_for_byte_compatible() -> None:
    manifest = _semantic_manifest()
    source = _semantic_handle(manifest).source_binding()
    context = object.__new__(compact_adapters._RecordingContext)  # noqa: SLF001
    context.bundle = {
        "source_bindings": {
            "semantic_epochs": {
                "binding_type": "exact_child_plus_epoch_transitive_semantic_v1",
                "source": source,
            }
        }
    }
    common = {
        "export_run_id": "export-legacy",
        "recording_id": "recording-1",
        "membership_member_sha256": "1" * 64,
        "bundle_set_member_sha256": "2" * 64,
        "bundle_record_sha256": "3" * 64,
        "source_child_key": "semantic_epochs",
        "source_run_path": manifest["run_path"],
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "source_payload_sha256": manifest["payload_digest"],
        "source_receipt_sha256": "4" * 64,
    }
    context.child_common = lambda key: {**common, "source_child_key": key}

    rows, reason = compact_adapters._semantic_epochs(context)  # noqa: SLF001

    assert reason is None
    assert rows == [
        {
            **common,
            "epoch_window_id": index,
            "analysis_role": role,
            "source_label": ("pre_event", "training_event", "post_event")[index],
            "start_frame": (10, 20, 30)[index],
            "end_frame_exclusive": (20, 30, 40)[index],
            "source_interval_sha256": str(index + 1) * 64,
            "protocol_semantic_hash": "sha256:" + "a" * 64,
            "protocol_semantic_step_index": 0,
            "protocol_semantic_step_ref": (
                "protocol_semantic_snapshot@recipe.steps[0]"
            ),
            "terminal_frame_excluded_pending_step_end_contract": False,
            "selection_identity_sha256": manifest["selection_identity_sha256"],
            "source_epoch_selection_sha256": "c" * 64,
            "step_end_interval_semantics": "producer_half_open_v1",
            "trial_index_integrity_status": "producer_asserted",
        }
        for index, role in enumerate(_SEMANTIC_ROLES)
    ]
