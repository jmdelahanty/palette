"""Motion DAG admission through real in-memory writers and strict readers.

Metadata discovery reads temporary zarr.json files; the numerical producer/seal
and runtime reader use the existing deterministic coordinate-publication fixture
and its supplied position authority. Archive opening is adapted to that fixture.
The strict motion loader and the DAG verifier are never patched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.analysis import track_kinematics as motion_writer
from fisheye.analysis_workflows import runtime_verification as mod
from fisheye.analysis_workflows.execution_profiles import (
    PRODUCTION_EXECUTION_PROFILE_ID,
    SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID,
)
from fisheye.shared.json_safety import json_attr_safe
from tests.unit.fisheye.test_track_kinematics_coordinate_contract import _WritableGroup
from tests.unit.fisheye.test_track_motion_publication import (
    _clone_motion_run_template,
    _motion_run_template,
)


@pytest.fixture(
    scope="module", params=(False, True), ids=("raw_keypoints", "refined_keypoints")
)
def motion_template(request):
    with pytest.MonkeyPatch.context() as patch:
        template = _motion_run_template(patch, refined_keypoints=request.param)
    return template, request.param


@pytest.fixture
def motion_output(motion_template, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    template, refined = motion_template
    root, run, track, _sealed, _physical = _clone_motion_run_template(
        template,
        monkeypatch,
    )
    # The shared fixture's root supports component lookup. Give the filesystem
    # adapter Zarr's slash lookup without changing any nodes or authority seals.
    runtime_root = _WritableGroup(
        path=root.path,
        archive_token=root._coordinate_archive_token,
    )
    runtime_root.attrs = root.attrs
    runtime_root.children = root.children
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: runtime_root)
    return tmp_path / "recording.zarr", runtime_root, run, track, refined


def _publish_metadata(archive: Path, run, *, canary: bool = False, root=None) -> None:
    run.attrs["stage_selector_eligible"] = not canary
    if canary:
        run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_PROFILE_ATTR] = (
            mod.TRACK_KINEMATICS_PUBLICATION_PROFILE_SELECTOR_INELIGIBLE_CANARY_V1
        )
        # The actual publisher declares the profile before sealing. Merely
        # adding it to an already sealed fixture would correctly look tampered.
        assert root is not None
        motion_writer._seal_and_load_track_motion_run_before_selection(root, run)
    parents = Path(run.path).parents
    for relative in (*reversed(parents), Path(run.path)):
        path = archive / relative
        path.mkdir(parents=True, exist_ok=True)
        attrs = run.attrs if relative == Path(run.path) else {}
        (path / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": json_attr_safe(attrs),
                }
            ),
            encoding="utf-8",
        )


def _verify(output, dependencies, *, canary=False, reuse=False):
    archive, _root, run, _track, _refined = output
    return mod.verify_persisted_stage_output(
        archive,
        "track_kinematics",
        requested_run=run.path.rsplit("/", 1)[1],
        dependency_runs=dependencies,
        session=mod.RuntimeVerificationSession(archive) if reuse else None,
        execution_profile_id=(
            SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID
            if canary
            else PRODUCTION_EXECUTION_PROFILE_ID
        ),
    )


@pytest.mark.parametrize("canary", (False, True), ids=("production", "canary"))
@pytest.mark.parametrize("reuse", (False, True), ids=("post_output", "reuse"))
def test_motion_verifier_accepts_exact_dependency_bindings(
    motion_output, canary, reuse
):
    archive, root, run, _track, refined = motion_output
    _publish_metadata(archive, run, canary=canary, root=root)
    dependencies = {
        "tracks": "trk_1",
        "refined_keypoints": "refined/kp_1" if refined else "kp_1",
    }

    result = _verify(motion_output, dependencies, canary=canary, reuse=reuse)

    assert result.available, result.reason
    assert result.run_name == run.path.rsplit("/", 1)[1]


@pytest.mark.parametrize("reuse", (False, True), ids=("post_output", "reuse"))
@pytest.mark.parametrize(
    ("dependency", "selection"),
    (
        ("tracks", "another_tracking_run"),
        ("refined_keypoints", "another_keypoint_run"),
        ("tracks", "latest"),
        ("refined_keypoints", "latest"),
        ("tracks", ""),
    ),
)
def test_motion_verifier_rejects_conflicting_or_unresolved_plan_dependencies(
    motion_output, dependency, selection, reuse
):
    archive, _root, run, _track, _refined = motion_output
    _publish_metadata(archive, run)
    # The artifact is internally valid. Only the workflow's selected dependency
    # conflicts, which its normal full-motion reader cannot detect by itself.
    result = _verify(motion_output, {dependency: selection}, reuse=reuse)

    assert not result.available
    assert "differs from the plan" in result.reason


def test_motion_verifier_does_not_equate_raw_and_refined_run_names(motion_output):
    archive, _root, run, _track, refined = motion_output
    _publish_metadata(archive, run)

    result = _verify(
        motion_output,
        {"refined_keypoints": "kp_1" if refined else "refined/kp_1"},
    )

    assert not result.available
    assert "differs from the plan" in result.reason


@pytest.mark.parametrize("dependency", ("tracks", "refined_keypoints"))
def test_canary_motion_reuse_rejects_conflicting_plan_dependency(
    motion_output, dependency
):
    archive, root, run, _track, _refined = motion_output
    _publish_metadata(archive, run, canary=True, root=root)

    result = _verify(
        motion_output, {dependency: "another_run"}, canary=True, reuse=True
    )

    assert not result.available
    assert "differs from the plan" in result.reason


def test_motion_verifier_accepts_exact_source_paths(motion_output):
    archive, _root, run, _track, _refined = motion_output
    _publish_metadata(archive, run)
    sources = run.attrs["source_refs"]

    result = _verify(
        motion_output,
        {
            "tracks": sources["source_tracking_path"],
            "refined_keypoints": sources["source_keypoint_path"],
        },
    )

    assert result.available, result.reason


@pytest.mark.parametrize("canary", (False, True), ids=("production", "canary"))
def test_motion_reuse_preserves_sealed_branch_without_declared_ancestors(
    motion_output, canary
):
    archive, root, run, _track, _refined = motion_output
    _publish_metadata(archive, run, canary=canary, root=root)

    result = _verify(motion_output, {}, canary=canary, reuse=True)

    assert result.available, result.reason


@pytest.mark.parametrize(
    "mutation", ("numerical_payload", "manifest", "source_payload", "incomplete")
)
def test_motion_verifier_preserves_strict_publication_refusal(motion_output, mutation):
    archive, root, run, track, refined = motion_output
    _publish_metadata(archive, run)
    if mutation == "numerical_payload":
        track["positions_px"].data[0, 0] += 1.0
    elif mutation == "manifest":
        run.attrs[motion_writer.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR] = (
            "0" * 64
        )
    elif mutation == "source_payload":
        family = "refined_keypoints_runs" if refined else "keypoints_runs"
        root[f"{family}/kp_1/heading"].data[0] += 1.0
    else:
        run.attrs["palette_run_completion_status"] = "failed"
    _publish_metadata(archive, run)

    result = _verify(motion_output, {})

    assert not result.available
