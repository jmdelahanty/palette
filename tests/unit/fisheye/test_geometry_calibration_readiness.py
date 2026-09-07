"""Pure/filesystem readiness tests; fixtures are not calibration evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from fisheye.cluster import arena_geometry_campaign as campaign
from fisheye.analysis_workflows.materializers import arena_geometry_candidates
from fisheye.cohorts.registry import compute_manifest_sha256, validate_frozen_cohort
from fisheye.cohorts.spec import canonical_sha256
from fisheye.shared.arena_geometry_auto_policy import build_geometry_auto_policy
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_arena_geometry_auto_policy import (
    _applicability,
    _evidence,
    _policy,
)
from tests.unit.fisheye.test_arena_geometry_candidates import (
    _bound_mask,
    _recovery_binding,
)


def _file(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _cohort(tmp_path, role, ids):
    query = {"recording_ids": ids}
    manifest = {
        "schema_id": "palette.frozen_cohort_manifest",
        "schema_version": 1,
        "manifest_canonicalization": "json_sorted_keys_no_manifest_sha256_v1",
        "created_utc": "2026-08-12T12:00:00Z",
        "cohort_id": role,
        "cohort_name": role,
        "cohort_query": query,
        "cohort_query_sha256": canonical_sha256(query),
        "registry": {"query_snapshot_sha256": "a" * 64, "access_mode": "read_only"},
        "selection_policy": {"include_every_match": True, "limit": None},
        "member_count": len(ids),
        "members": [
            {
                "dataset_id": key + "-dataset",
                "recording_id": key,
                "zarr_path": str(tmp_path / (key + ".zarr")),
                "zarr_origin": "source",
                "zarr_use": "analysis",
                "dataset_status": "active",
                "rig_id": "omnifin0",
                "arena_id": "arena_1",
                "camera_id": "2010093",
                "recording_started_utc": "2026-08-10T12:00:00Z"
                if role == "derivation"
                else "2026-08-11T12:00:00Z",
            }
            for key in ids
        ],
        "selection_summary": {"included_count": len(ids), "blocked_count": 0},
    }
    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)
    assert not validate_frozen_cohort(manifest)
    return _file(tmp_path, role + "_cohort.json", manifest), manifest


def _inputs(tmp_path):
    derivation, derivation_payload = _cohort(tmp_path, "derivation", ["d1", "d2"])
    validation, validation_payload = _cohort(tmp_path, "validation", ["v1", "v2"])
    rows = []
    for index, key in enumerate(("d1", "d2", "v1", "v2"), start=1):
        evidence = _evidence()
        registration = ("d" if key[0] == "d" else "e") * 64
        bound = _bound_mask()
        candidate = (
            arena_geometry_candidates.build_acquisition_geometry_candidate_record(
                replace(
                    bound,
                    mask=replace(
                        bound.mask, registration_sha256="sha256:" + registration
                    ),
                ),
                recovery_binding=_recovery_binding(),
            )
        )
        evidence["source_bindings"]["detection_source_signature"] = str(index) * 64
        evidence["source_bindings"]["acquisition_candidate_record_sha256"] = (
            arena_geometry_candidates._payload_sha256(candidate)
        )
        evidence["source_bindings"]["acquisition_observation_sha256"] = candidate[
            "acquisition_source"
        ]["source_observation_sha256"].removeprefix("sha256:")
        evidence["source_bindings"]["coordinate_binding_sha256"] = (
            canonical_json_sha256(candidate["coordinate_binding"])
        )
        evidence["acquisition_gate"] = deepcopy(
            candidate["valid_detection_region"]["geometry"]
        )
        rows.append(
            {
                "recording_id": key,
                "parent_recording_id": key,
                "applicability": _applicability(),
                "registration_sha256": registration,
                "scientific_recipe_sha256": "c" * 64,
                "expected_source_bindings": evidence["source_bindings"],
                "shadow_evidence_reference": _file(
                    tmp_path, key + "_metrics.json", evidence
                ),
                "historical_review_reference": None,
                "acquisition_candidate_reference": _file(
                    tmp_path, key + "_candidate.json", candidate
                ),
            }
        )
    arguments = {
        "derivation_cohort": derivation,
        "validation_cohort": validation,
        "recording_evidence": rows,
        "scientific_recipe_sha256": "c" * 64,
        "expected_camera_registration_cells": {
            role: [
                {"applicability": _applicability(), "registration_sha256": digest * 64}
            ]
            for role, digest in (("derivation", "d"), ("validation", "e"))
        },
        "required_negative_controls": [
            {"control_id": "not_configured", "kind": "real"},
            {"control_id": "wrong_coordinate", "kind": "injected"},
        ],
    }
    return arguments, derivation_payload, validation_payload


def _chronology(tmp_path, plan, *, prior_tuning=False):
    base_policy = _policy()
    calibration = dict(base_policy["calibration"])
    calibration["frozen_at_utc"] = "2026-08-21T12:00:00Z"
    calibration["derivation_manifest_sha256"] = plan["cohorts"]["derivation"][
        "manifest_sha256"
    ]
    derivation_ref = _file(
        tmp_path,
        "threshold_derivation.json",
        {
            "source_cohort_manifest_sha256": calibration["derivation_manifest_sha256"],
            "notes": "explicit synthetic test derivation artifact",
        },
    )
    calibration["threshold_derivation_sha256"] = derivation_ref["sha256"]
    policy = build_geometry_auto_policy(
        thresholds=base_policy["thresholds"], calibration=calibration
    )
    policy_ref = _file(tmp_path, "policy.json", policy)
    started = _file(
        tmp_path,
        "validation_started.json",
        {
            "at": "2026-08-22T12:00:00Z",
            "subject": plan["cohorts"]["validation"]["manifest_sha256"],
        },
    )
    events = [
        {
            "kind": "threshold_freeze",
            "reference": policy_ref,
            "timestamp_path": ["calibration", "frozen_at_utc"],
            "subject_digest_path": ["digest"],
        },
        {
            "kind": "validation_access",
            "reference": started,
            "timestamp_path": ["at"],
            "subject_digest_path": ["subject"],
        },
    ]
    if prior_tuning:
        tuned = _file(
            tmp_path,
            "prior_tuning.json",
            {
                "at": "2026-08-20T12:00:00Z",
                "subject": plan["cohorts"]["validation"]["manifest_sha256"],
            },
        )
        events.append(
            {
                "kind": "threshold_tuning",
                "reference": tuned,
                "timestamp_path": ["at"],
                "subject_digest_path": ["subject"],
            }
        )
    return [
        {
            "policy_reference": policy_ref,
            "threshold_derivation_reference": derivation_ref,
        }
    ], events


def test_freeze_never_opens_validation_metric_or_review_paths(tmp_path, monkeypatch):
    arguments, _derivation, _validation = _inputs(tmp_path)
    blocked_paths = {
        row["shadow_evidence_reference"]["path"]
        for row in arguments["recording_evidence"]
        if row["recording_id"].startswith("v")
    }
    forbidden_review = str(tmp_path / "v1_review.json")
    arguments["recording_evidence"][2]["historical_review_reference"] = {
        "path": forbidden_review,
        "sha256": "b" * 64,
    }
    blocked_paths.add(forbidden_review)
    original = Path.open
    opened = []

    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        assert str(path) not in blocked_paths, (
            "holdout evidence was opened during derivation freeze"
        )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    readiness = campaign.inspect_geometry_calibration_readiness(plan)
    assert plan["status"] == "planned"
    assert readiness["phase"] == "derivation"
    assert readiness["validation_metrics_opened"] is False
    assert readiness["coverage"]["derivation"]["metric_references_valid"] == 2
    assert readiness["coverage"]["validation"]["metric_references_valid"] == 0
    assert not (set(opened) & blocked_paths)


def test_frozen_membership_is_deterministic_complete_and_not_independent_recordings(
    tmp_path,
):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    arguments["recording_evidence"].reverse()
    assert plan == campaign.build_geometry_calibration_plan(**arguments)
    assert plan["cohorts"]["derivation"]["recording_ids"] == ["d1", "d2"]
    assert plan["coverage"]["derivation"]["expected_recordings"] == 2
    assert (
        plan["coverage"]["derivation"]["expected_camera_registration_cell_count"] == 1
    )
    assert plan["scientific_acceptance_created"] is False
    assert plan["selection_performed"] is False


@pytest.mark.parametrize("field", ["canvas_name", "arena_id", "camera_serial"])
def test_wrong_applicability_refuses_before_any_holdout_read(tmp_path, field):
    arguments, _d, _v = _inputs(tmp_path)
    arguments["recording_evidence"][0]["applicability"][field] = "wrong"
    with pytest.raises(ValueError, match="camera.registration cell|frozen cohort"):
        campaign.build_geometry_calibration_plan(**arguments)


def test_wrong_recipe_and_clip_pseudoreplication_refuse(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    arguments["recording_evidence"][0]["scientific_recipe_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="recipe"):
        campaign.build_geometry_calibration_plan(**arguments)
    arguments["recording_evidence"][0]["scientific_recipe_sha256"] = "c" * 64
    arguments["recording_evidence"][1]["parent_recording_id"] = "d1"
    with pytest.raises(ValueError, match="clip|parent"):
        campaign.build_geometry_calibration_plan(**arguments)


def test_missing_member_metrics_reviews_and_controls_remain_visible(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    arguments["recording_evidence"].pop(1)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    result = campaign.inspect_geometry_calibration_readiness(plan)
    assert result["coverage"]["derivation"]["expected_recordings"] == 2
    assert result["coverage"]["derivation"]["missing_inventory"] == 1
    assert result["coverage"]["derivation"]["metric_references_valid"] == 1
    assert result["coverage"]["derivation"]["policy_adjudication_validated"] == 0
    assert result["negative_controls"]["missing_result_ids"] == [
        "not_configured",
        "wrong_coordinate",
    ]
    assert result["promotion_readiness"] == "not_established"


def test_validation_phase_requires_recorded_freeze_before_access_and_never_claims_fresh(
    tmp_path,
):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    result = campaign.inspect_geometry_calibration_readiness(plan, phase="validation")
    assert result["validation_metrics_opened"] is False
    assert "frozen_policy_references_missing" in result["reason_codes"]
    policies, events = _chronology(tmp_path, plan)
    result = campaign.inspect_geometry_calibration_readiness(
        plan, phase="validation", policy_references=policies, chronology_events=events
    )
    assert result["validation_metrics_opened"] is True
    assert result["coverage"]["validation"]["metric_references_valid"] == 2
    assert result["declared_chronology_consistent"] is True
    assert result["holdout_freshness"] == "unknown"
    assert result["promotion_readiness"] == "not_established"


def test_known_tuning_forces_fresh_holdout_without_peeking(tmp_path, monkeypatch):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, events = _chronology(tmp_path, plan, prior_tuning=True)
    result = campaign.inspect_geometry_calibration_readiness(
        plan, phase="validation", policy_references=policies, chronology_events=events
    )
    assert result["validation_metrics_opened"] is False
    assert result["holdout_freshness"] == "fresh_holdout_required"
    assert "holdout_used_for_threshold_tuning" in result["reason_codes"]


def test_stale_cohort_or_changed_evidence_refuses_readiness(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    path = Path(arguments["recording_evidence"][0]["shadow_evidence_reference"]["path"])
    path.write_text("{}", encoding="utf-8")
    result = campaign.inspect_geometry_calibration_readiness(plan)
    assert result["coverage"]["derivation"]["metrics_invalid"] == 1
    Path(arguments["derivation_cohort"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        campaign.inspect_geometry_calibration_readiness(plan)


def test_boolean_untouched_claim_cannot_unlock_validation(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, _events = _chronology(tmp_path, plan)
    with pytest.raises(ValueError):
        campaign.inspect_geometry_calibration_readiness(
            plan,
            phase="validation",
            policy_references=policies,
            chronology_events=[{"untouched": True}],
        )


def test_oq5_profile_does_not_silently_accept_an_incomplete_cohort(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    with pytest.raises(ValueError, match="OQ5"):
        campaign.build_geometry_calibration_plan(
            **arguments, cohort_profile="goodbatbadbat_oq5_2026_08_12"
        )


def test_rehashed_plan_cannot_relabel_one_canvas_or_hide_a_member(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    original = campaign.build_geometry_calibration_plan(**arguments)
    for change in ("canvas", "membership"):
        plan = deepcopy(original)
        if change == "canvas":
            plan["recordings"][0]["evidence"]["applicability"]["canvas_name"] = "other"
        else:
            plan["recordings"].pop()
        plan["plan_sha256"] = canonical_json_sha256(
            {k: v for k, v in plan.items() if k != "plan_sha256"}
        )
        with pytest.raises(ValueError):
            campaign.inspect_geometry_calibration_readiness(plan)


def test_wrong_recording_metric_binding_is_visible_as_invalid(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    arguments["recording_evidence"][0]["shadow_evidence_reference"] = arguments[
        "recording_evidence"
    ][1]["shadow_evidence_reference"]
    plan = campaign.build_geometry_calibration_plan(**arguments)
    result = campaign.inspect_geometry_calibration_readiness(plan)
    assert result["coverage"]["derivation"]["metric_references_valid"] == 1
    assert result["coverage"]["derivation"]["metrics_invalid"] == 1


def test_recorded_inspection_before_threshold_freeze_keeps_holdout_locked(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, events = _chronology(tmp_path, plan)
    old = _file(
        tmp_path,
        "known_prior_inspection.json",
        {
            "at": "2026-08-15T12:00:00Z",
            "subject": plan["cohorts"]["validation"]["manifest_sha256"],
        },
    )
    events.append(
        {
            "kind": "aggregate_inspection",
            "reference": old,
            "timestamp_path": ["at"],
            "subject_digest_path": ["subject"],
        }
    )
    result = campaign.inspect_geometry_calibration_readiness(
        plan, phase="validation", policy_references=policies, chronology_events=events
    )
    assert result["validation_metrics_opened"] is False
    assert result["holdout_freshness"] == "unknown"
    assert (
        "thresholds_not_frozen_before_recorded_holdout_access" in result["reason_codes"]
    )


def test_existing_campaign_read_only_calibration_cli(tmp_path, capsys):
    arguments, _d, _v = _inputs(tmp_path)
    inputs = _file(tmp_path, "inputs.json", arguments)
    assert campaign.main(["calibration", "freeze", "--input-json", inputs["path"]]) == 0
    plan = json.loads(capsys.readouterr().out)
    frozen = _file(tmp_path, "plan.json", plan)
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert (
        campaign.main(["calibration", "readiness", "--input-json", frozen["path"]]) == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "readiness"
    assert result["coverage"]["derivation"]["accounted_recordings"] == 2
    assert result["coverage"]["validation"]["accounted_recordings"] == 2
    assert result["selection_performed"] is False
    assert before == {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }


def test_required_metric_file_missing_does_not_disappear_from_denominator(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    Path(
        arguments["recording_evidence"][0]["shadow_evidence_reference"]["path"]
    ).unlink()
    result = campaign.inspect_geometry_calibration_readiness(plan)
    assert result["coverage"]["derivation"]["metrics_missing"] == 1
    assert result["coverage"]["derivation"]["accounted_recordings"] == 2


def test_full_oq5_profile_preserves_64_members_and_twelve_cells_per_role(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    arguments["recording_evidence"] = []
    arguments["cohort_profile"] = "goodbatbadbat_oq5_2026_08_12"
    for role, count, offset in (("derivation", 36, 0), ("validation", 28, 3)):
        reference, manifest = _cohort(
            tmp_path, role, [f"{role}-{index:02d}" for index in range(count)]
        )
        cells = []
        for camera in range(4):
            applicability = _applicability()
            applicability["camera_serial"] = str(2010093 + camera)
            applicability["arena_id"] = f"arena_{camera + 1}"
            for registration in range(3):
                cells.append(
                    {
                        "applicability": dict(applicability),
                        "registration_sha256": str(offset + registration + 1) * 64,
                    }
                )
        for index, member in enumerate(manifest["members"]):
            cell = cells[index % len(cells)]
            member["camera_id"] = cell["applicability"]["camera_serial"]
            member["arena_id"] = cell["applicability"]["arena_id"]
            arguments["recording_evidence"].append(
                {
                    "recording_id": member["recording_id"],
                    "parent_recording_id": member["recording_id"],
                    **cell,
                    "scientific_recipe_sha256": "c" * 64,
                    "expected_source_bindings": None,
                    "shadow_evidence_reference": None,
                    "historical_review_reference": None,
                    "acquisition_candidate_reference": None,
                }
            )
        manifest["manifest_sha256"] = compute_manifest_sha256(manifest)
        arguments[role + "_cohort"] = _file(
            tmp_path, Path(reference["path"]).name, manifest
        )
        arguments["expected_camera_registration_cells"][role] = cells
    plan = campaign.build_geometry_calibration_plan(**arguments)
    result = campaign.inspect_geometry_calibration_readiness(plan)
    assert len(plan["recordings"]) == 64
    for role, count in (("derivation", 36), ("validation", 28)):
        assert result["coverage"][role]["expected_recordings"] == count
        assert result["coverage"][role]["accounted_recordings"] == count
        assert len(result["coverage"][role]["camera_registration_cells"]) == 12
    assert result["coverage"]["derivation"]["metrics_missing"] == 36
    assert result["coverage"]["validation"]["locked"] == 28


def test_policy_derivation_artifact_and_recipe_cannot_be_rebound(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, _events = _chronology(tmp_path, plan)
    policies[0]["threshold_derivation_reference"] = _file(
        tmp_path, "other_derivation.json", {"wrong": "derivation"}
    )
    with pytest.raises(ValueError, match="derivation cohort/artifact"):
        campaign.inspect_geometry_calibration_readiness(
            plan, policy_references=policies
        )
    policies, _events = _chronology(tmp_path, plan)
    path = Path(policies[0]["policy_reference"]["path"])
    base_policy = json.loads(path.read_text(encoding="utf-8"))
    base_policy["calibration"]["scientific_recipe_sha256"] = "f" * 64
    policy = build_geometry_auto_policy(
        thresholds=base_policy["thresholds"], calibration=base_policy["calibration"]
    )
    policies[0]["policy_reference"] = _file(tmp_path, path.name, policy)
    with pytest.raises(ValueError, match="wrong scientific recipe"):
        campaign.inspect_geometry_calibration_readiness(
            plan, policy_references=policies
        )


def test_cohorts_cannot_overlap_or_reuse_registration_snapshot(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    arguments["validation_cohort"] = arguments["derivation_cohort"]
    with pytest.raises(ValueError, match="membership overlaps"):
        campaign.build_geometry_calibration_plan(**arguments)
    arguments, _d, _v = _inputs(tmp_path)
    arguments["expected_camera_registration_cells"]["validation"][0][
        "registration_sha256"
    ] = "d" * 64
    with pytest.raises(ValueError, match="independent registration"):
        campaign.build_geometry_calibration_plan(**arguments)


def test_derivation_does_not_open_chronology_catalog_or_control_evidence(
    tmp_path, monkeypatch
):
    arguments, _d, _v = _inputs(tmp_path)
    forbidden = {"path": str(tmp_path / "forbidden.json"), "sha256": "a" * 64}
    arguments["diagnostic_catalog_reference"] = forbidden
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert str(path) != forbidden["path"]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    result = campaign.inspect_geometry_calibration_readiness(
        plan,
        chronology_events=[{"reference": forbidden}],
        negative_control_references={"not_configured": forbidden},
    )
    assert result["recorded_chronology"] == []
    assert result["diagnostic_catalog_reference"]["sha256"] == forbidden["sha256"]
    assert result["negative_controls"]["records"][0]["status"] == (
        "result_reference_frozen_not_opened"
    )


def test_present_control_files_and_numeric_pass_do_not_invent_adjudication(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, events = _chronology(tmp_path, plan)
    controls = {
        control["control_id"]: _file(
            tmp_path,
            control["control_id"] + "_result.json",
            {"control_id": control["control_id"], "passed": True},
        )
        for control in arguments["required_negative_controls"]
    }
    result = campaign.inspect_geometry_calibration_readiness(
        plan,
        phase="validation",
        policy_references=policies,
        chronology_events=events,
        negative_control_references=controls,
    )
    assert result["negative_controls"]["missing_result_ids"] == []
    assert result["negative_controls"]["validated_outcome_count"] == 0
    assert result["coverage"]["validation"]["policy_adjudication_validated"] == 0
    assert all(row["shadow_thresholds_satisfied"] for row in result["recordings"])
    assert result["false_automatic_pass_count"] is None
    assert result["promotion_readiness"] == "not_established"


def test_real_comparison_reference_remains_distinct_from_policy_adjudication(
    tmp_path, monkeypatch
):
    from fisheye.analysis_workflows.materializers import arena_geometry_comparison
    from tests.unit.fisheye.test_arena_geometry_comparison import (
        _archive_with_candidates,
        _write_nested_detection_source,
    )

    arguments, _d, _v = _inputs(tmp_path)
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    comparison = arena_geometry_comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
        detect_source_group_path=_write_nested_detection_source(archive),
    ).comparison_record
    item = arguments["recording_evidence"][0]
    item["expected_source_bindings"]["acquisition_candidate_record_sha256"] = (
        comparison["candidate_bindings"]["acquisition"]["candidate_record_sha256"]
    )
    metrics = _evidence()
    metrics["source_bindings"] = deepcopy(item["expected_source_bindings"])
    item["shadow_evidence_reference"] = _file(tmp_path, "d1_metrics.json", metrics)
    item["historical_review_reference"] = _file(tmp_path, "comparison.json", comparison)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    result = campaign.inspect_geometry_calibration_readiness(plan)
    assert result["coverage"]["derivation"]["historical_review_references_valid"] == 1
    assert result["coverage"]["derivation"]["policy_adjudication_validated"] == 0
    assert result["recordings"][0]["historical_review_status"] == (
        "comparison_present_not_policy_specific_adjudication"
    )


@pytest.mark.parametrize("alias_kind", ["direct", "copy", "symlink", "hardlink"])
def test_known_holdout_cannot_be_read_as_derivation_artifact(
    tmp_path, monkeypatch, alias_kind
):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, _events = _chronology(tmp_path, plan)
    holdout = deepcopy(arguments["recording_evidence"][2]["shadow_evidence_reference"])
    alias = tmp_path / "aliased_derivation.json"
    if alias_kind == "copy":
        shutil.copyfile(holdout["path"], alias)
    elif alias_kind == "symlink":
        alias.symlink_to(holdout["path"])
    elif alias_kind == "hardlink":
        alias.hardlink_to(holdout["path"])
    if alias_kind != "direct":
        holdout["path"] = str(alias)
    if alias_kind in {"symlink", "hardlink"}:
        # A deliberately false digest must not defeat physical alias checks;
        # rejection after opening and hashing would already expose the file.
        holdout["sha256"] = "f" * 64
    policy = campaign._read_calibration_reference(policies[0]["policy_reference"])
    calibration = dict(policy["calibration"])
    calibration["threshold_derivation_sha256"] = holdout["sha256"]
    policies[0] = {
        "policy_reference": _file(
            tmp_path,
            "alias_policy.json",
            build_geometry_auto_policy(
                thresholds=policy["thresholds"], calibration=calibration
            ),
        ),
        "threshold_derivation_reference": holdout,
    }
    original = Path.open
    opened = []

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    with pytest.raises(ValueError, match="holdout|artifact role"):
        campaign.inspect_geometry_calibration_readiness(
            plan, policy_references=policies
        )
    assert holdout["path"] not in opened


def test_locked_validation_cannot_read_compound_holdout_control_file(
    tmp_path, monkeypatch
):
    arguments, _d, _v = _inputs(tmp_path)
    holdout = campaign._read_calibration_reference(
        arguments["recording_evidence"][2]["shadow_evidence_reference"]
    )
    bundle = _file(
        tmp_path,
        "holdout_and_control.json",
        {"metrics": holdout, "control": {"control_id": "not_configured"}},
    )
    arguments["recording_evidence"][2]["shadow_evidence_reference"] = {
        **bundle,
        "json_path": ["metrics"],
    }
    plan = campaign.build_geometry_calibration_plan(**arguments)
    original = Path.open
    opened = []

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    result = campaign.inspect_geometry_calibration_readiness(
        plan,
        phase="validation",
        negative_control_references={
            "not_configured": {**bundle, "json_path": ["control"]}
        },
    )
    assert bundle["path"] not in opened
    assert result["validation_metrics_opened"] is False


def test_future_freeze_and_access_cannot_unlock_current_holdout(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, events = _chronology(tmp_path, plan)
    policy = campaign._read_calibration_reference(policies[0]["policy_reference"])
    calibration = dict(policy["calibration"])
    calibration["frozen_at_utc"] = "2099-01-01T00:00:00Z"
    future = _file(
        tmp_path,
        "future_policy.json",
        build_geometry_auto_policy(
            thresholds=policy["thresholds"], calibration=calibration
        ),
    )
    policies[0]["policy_reference"] = future
    events[0]["reference"] = future
    events[1]["reference"] = _file(
        tmp_path,
        "future_access.json",
        {
            "at": "2099-01-02T00:00:00Z",
            "subject": plan["cohorts"]["validation"]["manifest_sha256"],
        },
    )
    with pytest.raises(ValueError, match="future"):
        campaign.inspect_geometry_calibration_readiness(
            plan,
            phase="validation",
            policy_references=policies,
            chronology_events=events,
        )


def test_candidate_registration_cannot_be_replaced_with_independent_cell_labels(
    tmp_path,
):
    from fisheye.analysis_workflows.materializers import arena_geometry_candidates
    from tests.unit.fisheye.test_arena_geometry_candidates import (
        _bound_mask,
        _recovery_binding,
    )

    arguments, _d, _v = _inputs(tmp_path)
    candidate = arena_geometry_candidates.build_acquisition_geometry_candidate_record(
        _bound_mask(), recovery_binding=_recovery_binding()
    )
    candidate_digest = arena_geometry_candidates._payload_sha256(candidate)
    for row in arguments["recording_evidence"]:
        metrics = campaign._read_calibration_reference(row["shadow_evidence_reference"])
        metrics["source_bindings"]["acquisition_candidate_record_sha256"] = (
            candidate_digest
        )
        row["expected_source_bindings"] = deepcopy(metrics["source_bindings"])
        row["shadow_evidence_reference"] = _file(
            tmp_path, row["recording_id"] + "_metrics.json", metrics
        )
    plan = campaign.build_geometry_calibration_plan(**arguments)
    result = campaign.inspect_geometry_calibration_readiness(plan)
    assert result["coverage"]["derivation"]["metrics_invalid"] == 2


def test_exact_candidate_reference_still_rejects_wrong_registration_cell(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    candidate = arena_geometry_candidates.build_acquisition_geometry_candidate_record(
        _bound_mask(), recovery_binding=_recovery_binding()
    )
    row = arguments["recording_evidence"][0]
    metrics = campaign._read_calibration_reference(row["shadow_evidence_reference"])
    metrics["source_bindings"]["acquisition_candidate_record_sha256"] = (
        arena_geometry_candidates._payload_sha256(candidate)
    )
    row["expected_source_bindings"] = deepcopy(metrics["source_bindings"])
    row["acquisition_candidate_reference"] = _file(
        tmp_path, "wrong_registration_candidate.json", candidate
    )
    row["shadow_evidence_reference"] = _file(
        tmp_path, "wrong_registration_metrics.json", metrics
    )
    result = campaign.inspect_geometry_calibration_readiness(
        campaign.build_geometry_calibration_plan(**arguments)
    )
    assert result["coverage"]["derivation"]["metrics_invalid"] == 1
    assert any(
        "Candidate registration" in reason
        for reason in result["recordings"][0]["reason_codes"]
    )


def test_valid_candidate_and_metric_refs_do_not_prove_recording_attachment(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    result = campaign.inspect_geometry_calibration_readiness(
        campaign.build_geometry_calibration_plan(**arguments)
    )
    assert result["coverage"]["derivation"]["metric_references_valid"] == 2
    assert result["coverage"]["derivation"]["member_source_bindings_validated"] == 0
    assert (
        result["coverage"]["derivation"][
            "independent_camera_registration_cells_validated"
        ]
        == 0
    )
    assert all(
        row["member_source_provenance_status"] == "not_independently_verified"
        for row in result["recordings"]
    )
    assert (
        "member_source_provenance_not_independently_verified" in result["reason_codes"]
    )


def test_missing_candidate_reference_is_visible_and_metrics_are_not_counted_valid(
    tmp_path,
):
    arguments, _d, _v = _inputs(tmp_path)
    arguments["recording_evidence"][0]["acquisition_candidate_reference"] = None
    result = campaign.inspect_geometry_calibration_readiness(
        campaign.build_geometry_calibration_plan(**arguments)
    )
    assert result["coverage"]["derivation"]["metric_references_valid"] == 1
    assert result["coverage"]["derivation"]["metrics_missing"] == 1
    assert result["coverage"]["derivation"]["accounted_recordings"] == 2


def test_future_access_without_future_policy_is_rejected(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, events = _chronology(tmp_path, plan)
    events[1]["reference"] = _file(
        tmp_path,
        "future_access_only.json",
        {
            "at": "2099-01-02T00:00:00Z",
            "subject": plan["cohorts"]["validation"]["manifest_sha256"],
        },
    )
    with pytest.raises(ValueError, match="future chronology"):
        campaign.inspect_geometry_calibration_readiness(
            plan,
            phase="validation",
            policy_references=policies,
            chronology_events=events,
        )


@pytest.mark.parametrize("metadata_role", ["policy", "chronology", "cohort", "cli"])
def test_compound_holdout_cannot_be_read_as_metadata(
    tmp_path, monkeypatch, metadata_role
):
    arguments, derivation, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, events = _chronology(tmp_path, plan)
    payload = campaign._read_calibration_reference(policies[0]["policy_reference"])
    metrics = campaign._read_calibration_reference(
        arguments["recording_evidence"][2]["shadow_evidence_reference"]
    )
    bundle = _file(
        tmp_path,
        "compound_metadata_and_holdout.json",
        {
            "policy": payload,
            "metrics": metrics,
            "cohort": derivation,
            "event": {
                "at": "2026-08-22T12:00:00Z",
                "subject": plan["cohorts"]["validation"]["manifest_sha256"],
            },
        },
    )
    arguments["recording_evidence"][2]["shadow_evidence_reference"] = {
        **bundle,
        "json_path": ["metrics"],
    }
    if metadata_role == "cohort":
        arguments["derivation_cohort"] = {**bundle, "json_path": ["cohort"]}
    else:
        plan = campaign.build_geometry_calibration_plan(**arguments)
    if metadata_role == "policy":
        policies[0]["policy_reference"] = {**bundle, "json_path": ["policy"]}
    elif metadata_role == "chronology":
        events[1]["reference"] = {**bundle, "json_path": ["event"]}
    if metadata_role == "cli":
        plan_ref = _file(tmp_path, "cli_plan.json", plan)
    original = Path.open
    opened = []

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    with pytest.raises(ValueError, match="holdout|artifact role"):
        if metadata_role == "cohort":
            campaign.build_geometry_calibration_plan(**arguments)
        elif metadata_role == "cli":
            campaign.main(
                [
                    "calibration",
                    "readiness",
                    "--input-json",
                    plan_ref["path"],
                    "--readiness-inputs",
                    bundle["path"],
                    "--readiness-inputs-sha256",
                    bundle["sha256"],
                ]
            )
        else:
            campaign.inspect_geometry_calibration_readiness(
                plan,
                phase="validation",
                policy_references=policies,
                chronology_events=events,
            )
    assert bundle["path"] not in opened


def test_authorized_compound_control_open_is_reported_as_holdout_read(tmp_path):
    arguments, _d, _v = _inputs(tmp_path)
    metrics = campaign._read_calibration_reference(
        arguments["recording_evidence"][2]["shadow_evidence_reference"]
    )
    bundle = _file(
        tmp_path,
        "authorized_compound_control.json",
        {"metrics": metrics, "control": {"control_id": "not_configured"}},
    )
    arguments["recording_evidence"][2]["shadow_evidence_reference"] = {
        **bundle,
        "json_path": ["metrics"],
    }
    # A missing expected binding prevents a direct metric read, but an explicitly
    # authorized later control read still opens the whole holdout-bearing file.
    arguments["recording_evidence"][2]["expected_source_bindings"] = None
    arguments["recording_evidence"][3]["shadow_evidence_reference"] = None
    plan = campaign.build_geometry_calibration_plan(**arguments)
    policies, events = _chronology(tmp_path, plan)
    result = campaign.inspect_geometry_calibration_readiness(
        plan,
        phase="validation",
        policy_references=policies,
        chronology_events=events,
        negative_control_references={
            "not_configured": {**bundle, "json_path": ["control"]}
        },
    )
    assert result["coverage"]["validation"]["metric_references_valid"] == 0
    assert result["validation_metrics_opened"] is True
    assert result["validation_evidence_opened"] is True


def test_cli_requires_exact_metadata_digest_before_opening_holdout_copy(
    tmp_path, monkeypatch
):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    plan_ref = _file(tmp_path, "plan_for_cli_copy.json", plan)
    holdout = arguments["recording_evidence"][2]["shadow_evidence_reference"]
    copied = tmp_path / "copied_holdout_as_cli_metadata.json"
    shutil.copyfile(holdout["path"], copied)
    original = Path.open
    opened = []

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    command = [
        "calibration",
        "readiness",
        "--input-json",
        plan_ref["path"],
        "--readiness-inputs",
        str(copied),
    ]
    with pytest.raises(SystemExit):
        campaign.main(command)
    assert str(copied) not in opened
    with pytest.raises(ValueError, match="holdout|artifact role"):
        campaign.main([*command, "--readiness-inputs-sha256", holdout["sha256"]])
    assert str(copied) not in opened


def test_cli_bound_nonholdout_metadata_preserves_valid_validation_path(
    tmp_path, capsys
):
    arguments, _d, _v = _inputs(tmp_path)
    plan = campaign.build_geometry_calibration_plan(**arguments)
    plan_ref = _file(tmp_path, "positive_cli_plan.json", plan)
    policies, events = _chronology(tmp_path, plan)
    inputs = _file(
        tmp_path,
        "positive_readiness_inputs.json",
        {
            "policy_references": policies,
            "chronology_events": events,
        },
    )
    assert (
        campaign.main(
            [
                "calibration",
                "readiness",
                "--input-json",
                plan_ref["path"],
                "--phase",
                "validation",
                "--readiness-inputs",
                inputs["path"],
                "--readiness-inputs-sha256",
                inputs["sha256"],
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["validation_metrics_opened"] is True
    assert result["coverage"]["validation"]["metric_references_valid"] == 2
    assert result["promotion_readiness"] == "not_established"
