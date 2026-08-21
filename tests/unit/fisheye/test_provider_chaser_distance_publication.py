from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.analysis_workflows import (
    chaser_proxy_candidate_receipt as receipt_module,
    chaser_relative_frame_source_handle as relative_handle_module,
    provider_chaser_distance_publication as publication_module,
    provider_chaser_distance_successor as successor_module,
)
from fisheye.analysis_workflows.provider_chaser_distance_publication import (
    build_provider_chaser_distance_publication_plan,
    deep_audit_provider_chaser_distance_run,
    load_provider_chaser_distance_source_handle,
    publish_provider_chaser_distance_run,
)
from fisheye.utils import materialize_provider_chaser_distance as cli_module
from tests.unit.fisheye.test_chaser_relative_frame_source_handle import (
    _publish_proxy_bound,
)
from tests.unit.fisheye.test_chaser_receipt_backed_source_handle import (
    _receipt_for,
)


def _plan(tmp_path: Path, *, run_name: str = "provider-v1"):
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    receipt = _receipt_for(archive, tmp_path)
    return archive, receipt, build_provider_chaser_distance_publication_plan(
        archive,
        receipt=receipt,
        run_name=run_name,
        expected_recording_id="recording-1",
    )


def test_dry_run_plan_is_selector_ineligible_and_does_not_create_target(
    tmp_path: Path,
) -> None:
    archive, _receipt, plan = _plan(tmp_path)

    result = plan.to_json()

    assert result["status"] == "dry_run_plan"
    assert result["selector_eligible"] is False
    assert result["selection"] == "none"
    assert result["production_authority"] is False
    assert result["registry_update"] is False
    assert result["target_exists"] is False
    assert not (archive / plan.run_path).exists()


def test_receipt_backed_publication_does_not_rehash_upstream_dense_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    receipt = _receipt_for(archive, tmp_path)

    def unexpected_hash(_value: object) -> str:
        raise AssertionError("ordinary publication must not rehash upstream arrays")

    monkeypatch.setattr(relative_handle_module, "array_values_sha256", unexpected_hash)
    monkeypatch.setattr(
        receipt_module,
        "validate_chaser_proxy_candidate_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ordinary publication called the deep receipt audit")
        ),
    )

    plan = build_provider_chaser_distance_publication_plan(
        archive,
        receipt=receipt,
        run_name="provider-v1",
        expected_recording_id="recording-1",
    )
    assert plan.to_json()["upstream_dense_hash_recomputation"] is False


def test_output_declarations_are_hashed_once_and_not_by_publisher_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    receipt = _receipt_for(archive, tmp_path)
    calls: list[int] = []
    original = successor_module.array_values_sha256

    def counted_hash(value: object) -> str:
        calls.append(1)
        return original(value)

    monkeypatch.setattr(successor_module, "array_values_sha256", counted_hash)
    plan = build_provider_chaser_distance_publication_plan(
        archive,
        receipt=receipt,
        run_name="provider-v1",
        expected_recording_id="recording-1",
    )
    assert len(calls) == len(plan.prepared.arrays)

    result = publish_provider_chaser_distance_run(plan, scratch_root=tmp_path / "scratch")

    assert len(calls) == len(plan.prepared.arrays)
    assert result["selector_eligible"] is False
    assert result["selection"] == "none"
    assert result["registry_update"] is False


def test_published_reader_is_bounded_and_deep_audit_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, receipt, plan = _plan(tmp_path)
    publish_provider_chaser_distance_run(plan, scratch_root=tmp_path / "scratch")

    def unexpected_hash(_value: object) -> str:
        raise AssertionError("bounded persistent reader must not hash arrays")

    monkeypatch.setattr(publication_module, "array_values_sha256", unexpected_hash)
    bounded = load_provider_chaser_distance_source_handle(
        archive,
        run_name=plan.run_name,
        expected_recording_id="recording-1",
    )
    assert bounded.verification_mode == "bounded_publication"
    assert bounded.n_rows == plan.prepared.dimensions.n_rows
    assert all(not value.flags.writeable for value in bounded.arrays.values())
    assert bounded.source_receipt_sha256 == receipt["record_sha256"]
    assert dict(bounded.source_receipt) == receipt

    with pytest.raises(AssertionError, match="must not hash"):
        deep_audit_provider_chaser_distance_run(
            archive,
            run_name=plan.run_name,
            expected_recording_id="recording-1",
        )


def test_publisher_rejects_selector_like_run_names(tmp_path: Path) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    receipt = _receipt_for(archive, tmp_path)

    with pytest.raises(ValueError, match="concrete bare run name"):
        build_provider_chaser_distance_publication_plan(
            archive,
            receipt=receipt,
            run_name="latest",
            expected_recording_id="recording-1",
        )


def test_cli_is_dry_run_by_default_and_returns_a_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    receipt = _receipt_for(archive, tmp_path)
    receipt_path = tmp_path / "candidate-chain-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert cli_module.main(
        [
            str(archive),
            "--receipt-json",
            str(receipt_path),
            "--run-name",
            "provider-v1",
            "--expected-recording-id",
            "recording-1",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "dry_run_plan"
    assert result["target_exists"] is False
    assert not (archive / "analysis/provider_chaser_distance_runs/provider-v1").exists()
