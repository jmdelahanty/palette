"""Metadata-file-only availability checks for analysis workflow planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from fisheye.analysis_workflows.storage_contract_catalog import (
    DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS,
)
from fisheye.registry.stage_catalog import canonical_stage_id
from fisheye.shared.keypoint_motion_authority import (
    KeypointMotionAuthorityError,
    keypoint_lineage_from_attributes,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent_attrs,
    is_run_selector_eligible_attrs,
    resolve_latest_complete_run_name_from_attrs,
)


STAGE_RUN_PARENTS: Mapping[str, tuple[str, ...]] = {
    # ``refined_keypoints`` is the workflow's curated keypoint-authority
    # dependency.  Its resolver below accepts either the refined member of an
    # active keypoint bundle or an explicitly selected canonical raw run (the
    # clipped importer uses the latter for canonical passthrough).
    "refined_keypoints": ("refined_keypoints_runs", "keypoints_runs"),
    "refined_subject_masks": ("refined_subject_masks_runs",),
    "tracks": ("tracking_runs",),
    **DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS,
}
TRACK_KINEMATICS_VISUALIZATION_STAGE = "track_kinematics_visualization"
TRACK_KINEMATICS_PARENT = "analysis/track_kinematics_runs/offline"
TRACK_KINEMATICS_VISUALIZATION_PARENT = (
    "analysis/track_kinematics_visualization_runs/offline"
)
TRACK_KINEMATICS_INTERACTIVE_ARTIFACT = (
    "visualizations/track_kinematics_summary_track_0_interactive"
)
TRACK_KINEMATICS_INTERACTIVE_RENDERER = "palette-track-kinematics-summary-v1"
KEYPOINT_BUNDLE_AUTHORITY_ATTR = "keypoint_bundle_authority"
KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR = (
    "keypoint_bundle_authority_generation"
)
KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR = "keypoint_bundle_authority_lease"
SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR = "subject_mask_authority"
SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR = (
    "subject_mask_authority_generation"
)
SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR = "subject_mask_authority_lease"
SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR = (
    "subject_mask_bundle_selector_eligible"
)


@dataclass(frozen=True)
class StageAvailability:
    stage_id: str
    available: bool
    artifact_path: str | None = None
    run_name: str | None = None
    reason: str = ""
    completion_status: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "available": self.available,
            "artifact_path": self.artifact_path,
            "run_name": self.run_name,
            "reason": self.reason,
            "completion_status": self.completion_status,
        }


def _attrs(path: Path) -> dict[str, object]:
    metadata_path = path / "zarr.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    raw = payload.get("attributes") if isinstance(payload, dict) else None
    return dict(raw) if isinstance(raw, dict) else {}


def _safe_run_name(value: str) -> str:
    run_name = str(value).strip().strip("/")
    pure = PurePosixPath(run_name)
    if not run_name or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe workflow run selection: {value!r}")
    return run_name


def _completion_status(attrs: Mapping[str, object]) -> str | None:
    for key in (
        "palette_run_completion_status",
        "status",
        "run_status",
        "completion_status",
    ):
        value = attrs.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return None


def stage_run_relative_path(stage_id: str, run_name: str) -> str:
    """Return the sole registered run path for an executable canonical stage."""

    canonical = canonical_stage_id(stage_id)
    if canonical == TRACK_KINEMATICS_VISUALIZATION_STAGE:
        return f"{TRACK_KINEMATICS_VISUALIZATION_PARENT}/{_safe_run_name(run_name)}"
    parents = STAGE_RUN_PARENTS.get(canonical)
    if not parents:
        raise KeyError(f"no run parent is registered for stage {canonical!r}")
    if len(parents) != 1:
        raise ValueError(
            f"stage {canonical!r} has ambiguous run parents; execution must select one"
        )
    return f"{parents[0]}/{_safe_run_name(run_name)}"


def _metadata_child_names(parent: Path) -> tuple[str, ...]:
    try:
        return tuple(
            child.name
            for child in parent.iterdir()
            if child.is_dir() and (child / "zarr.json").is_file()
        )
    except OSError:
        return ()


def _resolve_metadata_run_name(
    parent: Path,
    parent_attrs: Mapping[str, object],
    requested_run: str | None,
    *,
    parent_relative_path: str,
) -> tuple[str | None, str | None]:
    """Resolve one strict maintained run through the shared lifecycle contract."""

    requested = "" if requested_run is None else str(requested_run).strip().strip("/")
    parent_prefix = str(parent_relative_path).strip().strip("/") + "/"
    if requested.startswith(parent_prefix):
        requested = requested[len(parent_prefix) :]
    if requested and requested.lower() != "latest":
        run_name = _safe_run_name(requested)
        run_attrs = _attrs(parent / run_name)
        if not (parent / run_name / "zarr.json").is_file():
            return run_name, "selected run metadata is missing"
        if not is_run_complete_in_parent_attrs(
            parent_attrs,
            run_attrs,
            legacy_default=False,
        ):
            return run_name, "selected run is not complete"
        if not is_run_selector_eligible_attrs(run_attrs):
            return run_name, "selected run is not selector-eligible"
        return run_name, None

    run_name = resolve_latest_complete_run_name_from_attrs(
        parent_attrs=parent_attrs,
        child_names=_metadata_child_names(parent),
        child_attrs=lambda name: (
            _attrs(parent / name)
            if (parent / name / "zarr.json").is_file()
            else None
        ),
        legacy_default=False,
    )
    if run_name is None:
        return None, (
            "no stable complete selector-eligible run is selected; "
            "selector activation may be in progress"
        )
    return run_name, None


def _available_selected_run(
    root: Path,
    *,
    stage_id: str,
    relative_parent: str,
    run_name: str,
    encoded_run_name: str,
    reason: str,
    allow_selector_ineligible: bool = False,
) -> StageAvailability:
    """Validate one already-authorized exact run using metadata files only."""

    parent = root / relative_parent
    run_path = parent / run_name
    relative_run_path = f"{relative_parent}/{run_name}"
    if not (parent / "zarr.json").is_file() or not (
        run_path / "zarr.json"
    ).is_file():
        return StageAvailability(
            stage_id=stage_id,
            available=False,
            artifact_path=relative_run_path,
            run_name=encoded_run_name,
            reason="selected authority member metadata is missing",
        )
    parent_attrs = _attrs(parent)
    run_attrs = _attrs(run_path)
    if not is_run_complete_in_parent_attrs(
        parent_attrs,
        run_attrs,
        legacy_default=False,
    ):
        return StageAvailability(
            stage_id=stage_id,
            available=False,
            artifact_path=relative_run_path,
            run_name=encoded_run_name,
            reason="selected authority member is not complete",
            completion_status=_completion_status(run_attrs),
        )
    if not allow_selector_ineligible and not is_run_selector_eligible_attrs(
        run_attrs
    ):
        return StageAvailability(
            stage_id=stage_id,
            available=False,
            artifact_path=relative_run_path,
            run_name=encoded_run_name,
            reason="selected authority member is not selector-eligible",
            completion_status=_completion_status(run_attrs),
        )
    return StageAvailability(
        stage_id=stage_id,
        available=True,
        artifact_path=relative_run_path,
        run_name=encoded_run_name,
        reason=reason,
        completion_status=_completion_status(run_attrs),
    )


def _active_keypoint_bundle_refined_path(root: Path) -> str | None:
    attrs = _attrs(root)
    if KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR in attrs:
        return None
    authority = attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR)
    generation = attrs.get(KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR)
    if (
        not isinstance(authority, Mapping)
        or authority.get("schema_id") != "palette.keypoint.bundle_authority"
        or authority.get("schema_version") != 1
        or type(generation) is not int
        or generation <= 0
        or authority.get("generation") != generation
    ):
        return None
    members = authority.get("members")
    refined = (
        members.get("refined_keypoints") if isinstance(members, Mapping) else None
    )
    run_path = refined.get("run_path") if isinstance(refined, Mapping) else None
    if (
        not isinstance(run_path, str)
        or not run_path.startswith("refined_keypoints_runs/")
        or len(PurePosixPath(run_path).parts) != 2
    ):
        return None
    return run_path


def _keypoint_authority_availability(
    root: Path,
    *,
    requested_run: str | None,
) -> StageAvailability:
    """Resolve refined-bundle or clipped canonical-passthrough keypoints."""

    requested = str(requested_run or "latest").strip().strip("/")
    root_attrs = _attrs(root)
    authority_present = KEYPOINT_BUNDLE_AUTHORITY_ATTR in root_attrs
    if KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR in root_attrs:
        return StageAvailability(
            stage_id="refined_keypoints",
            available=False,
            reason="keypoint bundle activation lease is present",
        )
    active_refined_path = _active_keypoint_bundle_refined_path(root)
    if authority_present and active_refined_path is None:
        return StageAvailability(
            stage_id="refined_keypoints",
            available=False,
            reason="keypoint bundle authority is malformed or incomplete",
        )
    if requested.lower() == "latest" and active_refined_path is not None:
        run_name = PurePosixPath(active_refined_path).name
        return _available_selected_run(
            root,
            stage_id="refined_keypoints",
            relative_parent="refined_keypoints_runs",
            run_name=run_name,
            encoded_run_name=f"refined/{run_name}",
            reason="active keypoint-bundle refined authority is available",
            allow_selector_ineligible=True,
        )

    explicit_family: str | None = None
    explicit_name = requested
    if requested.startswith("refined/"):
        explicit_family = "refined_keypoints_runs"
        explicit_name = requested.split("/", 1)[1]
    elif requested.startswith("refined_keypoints_runs/"):
        explicit_family = "refined_keypoints_runs"
        explicit_name = requested.split("/", 1)[1]
    elif requested.startswith("keypoints_runs/"):
        explicit_family = "keypoints_runs"
        explicit_name = requested.split("/", 1)[1]

    families = (
        (explicit_family,)
        if explicit_family is not None
        else ("refined_keypoints_runs", "keypoints_runs")
    )
    for family in families:
        assert family is not None
        parent = root / family
        if not (parent / "zarr.json").is_file():
            continue
        if requested.lower() == "latest":
            name, error = _resolve_metadata_run_name(
                parent,
                _attrs(parent),
                "latest",
                parent_relative_path=family,
            )
            if name is None or error is not None:
                continue
        else:
            name = _safe_run_name(explicit_name)
            member_path = f"{family}/{name}"
            allow_ineligible = member_path == active_refined_path
            result = _available_selected_run(
                root,
                stage_id="refined_keypoints",
                relative_parent=family,
                run_name=name,
                encoded_run_name=(f"refined/{name}" if family.startswith("refined_") else name),
                reason=(
                    "active keypoint-bundle refined authority is available"
                    if allow_ineligible
                    else "selected canonical keypoint authority is available"
                ),
                allow_selector_ineligible=allow_ineligible,
            )
            if result.available or explicit_family is not None:
                return result
            continue
        return _available_selected_run(
            root,
            stage_id="refined_keypoints",
            relative_parent=family,
            run_name=name,
            encoded_run_name=(f"refined/{name}" if family.startswith("refined_") else name),
            reason=(
                "persisted refined keypoint authority is available"
                if family.startswith("refined_")
                else "persisted canonical keypoint passthrough authority is available"
            ),
        )
    return StageAvailability(
        stage_id="refined_keypoints",
        available=False,
        reason=(
            "no active keypoint bundle, refined selector, or canonical raw "
            "keypoint selector is available"
        ),
    )


def _keypoint_crop_lineage(
    root: Path,
    *,
    selected_keypoint_run: str,
) -> tuple[str, int | None, str | None] | StageAvailability:
    """Resolve the exact crop rowset named by one selected keypoint authority."""

    keypoints = _keypoint_authority_availability(
        root,
        requested_run=selected_keypoint_run,
    )
    if not keypoints.available or not keypoints.artifact_path:
        return StageAvailability(
            stage_id="tracks",
            available=False,
            reason=(
                "selected keypoint dependency is unavailable: "
                f"{keypoints.reason}"
            ),
        )
    keypoint_path = root / keypoints.artifact_path
    keypoint_attrs = _attrs(keypoint_path)
    family, keypoint_name = keypoints.artifact_path.split("/", 1)
    raw_attrs: Mapping[str, object] | None = None
    if (
        family == "refined_keypoints_runs"
        and not isinstance(keypoint_attrs.get("run_manifest"), Mapping)
    ):
        raw_name = str(keypoint_attrs.get("source_keypoints_run") or "").strip()
        if not raw_name or "/" in raw_name:
            return StageAvailability(
                stage_id="tracks",
                available=False,
                artifact_path=keypoints.artifact_path,
                run_name=keypoints.run_name,
                reason="selected refined keypoint authority lacks its exact base run",
            )
        raw_parent = root / "keypoints_runs"
        raw_path = raw_parent / raw_name
        if not (raw_parent / "zarr.json").is_file() or not (
            raw_path / "zarr.json"
        ).is_file():
            return StageAvailability(
                stage_id="tracks",
                available=False,
                artifact_path=f"keypoints_runs/{raw_name}",
                run_name=keypoints.run_name,
                reason="selected refined keypoint base-run metadata is missing",
            )
        raw_attrs = _attrs(raw_path)
        if not is_run_complete_in_parent_attrs(
            _attrs(raw_parent),
            raw_attrs,
            legacy_default=False,
        ):
            return StageAvailability(
                stage_id="tracks",
                available=False,
                artifact_path=f"keypoints_runs/{raw_name}",
                run_name=keypoints.run_name,
                reason="selected refined keypoint base run is incomplete",
            )
    try:
        lineage = keypoint_lineage_from_attributes(
            family=family,
            run_name=keypoint_name,
            attrs=keypoint_attrs,
            raw_attrs=raw_attrs,
        )
    except (KeypointMotionAuthorityError, TypeError, ValueError) as exc:
        return StageAvailability(
            stage_id="tracks",
            available=False,
            artifact_path=keypoints.artifact_path,
            run_name=keypoints.run_name,
            reason=f"selected keypoint lineage is invalid: {exc}",
        )

    raw_parent = root / "keypoints_runs"
    raw_path = root / lineage.base_run_path
    if not (raw_parent / "zarr.json").is_file() or not (
        raw_path / "zarr.json"
    ).is_file():
        return StageAvailability(
            stage_id="tracks",
            available=False,
            artifact_path=lineage.base_run_path,
            run_name=keypoints.run_name,
            reason="selected keypoint base-run metadata is missing",
        )
    persisted_raw_attrs = _attrs(raw_path)
    if not is_run_complete_in_parent_attrs(
        _attrs(raw_parent),
        persisted_raw_attrs,
        legacy_default=False,
    ):
        return StageAvailability(
            stage_id="tracks",
            available=False,
            artifact_path=lineage.base_run_path,
            run_name=keypoints.run_name,
            reason="selected keypoint base run is incomplete",
        )
    if lineage.base_manifest_digest is not None:
        raw_manifest = persisted_raw_attrs.get("run_manifest")
        try:
            raw_digest = (
                canonical_json_sha256(raw_manifest)
                if isinstance(raw_manifest, Mapping)
                else None
            )
        except (TypeError, ValueError):
            raw_digest = None
        if raw_digest != lineage.base_manifest_digest:
            return StageAvailability(
                stage_id="tracks",
                available=False,
                artifact_path=lineage.base_run_path,
                run_name=keypoints.run_name,
                reason="selected keypoint base manifest differs from its binding",
            )

    crop_name = lineage.crop_run
    crop_path = root / "crop_runs" / crop_name
    if not (crop_path / "zarr.json").is_file():
        return StageAvailability(
            stage_id="tracks",
            available=False,
            artifact_path=f"crop_runs/{crop_name}",
            run_name=keypoints.run_name,
            reason="selected keypoint source-crop metadata is missing",
        )
    crop_attrs = _attrs(crop_path)
    if lineage.crop_manifest_digest is not None:
        crop_manifest = crop_attrs.get("run_manifest")
        try:
            crop_digest = (
                canonical_json_sha256(crop_manifest)
                if isinstance(crop_manifest, Mapping)
                else None
            )
        except (TypeError, ValueError):
            crop_digest = None
        if crop_digest != lineage.crop_manifest_digest:
            return StageAvailability(
                stage_id="tracks",
                available=False,
                artifact_path=lineage.crop_path,
                run_name=keypoints.run_name,
                reason="selected keypoint crop manifest differs from its binding",
            )
    refined_name = str(
        crop_attrs.get("source_refined_run")
        or crop_attrs.get("source_refined_detect_run")
        or ""
    ).strip()
    if not refined_name:
        source = crop_attrs.get("source_refined_snapshot")
        if not isinstance(source, Mapping):
            manifest = crop_attrs.get("run_manifest")
            payload = manifest.get("payload") if isinstance(manifest, Mapping) else None
            source = (
                payload.get("source_refined_snapshot")
                if isinstance(payload, Mapping)
                else None
            )
        refined_name = (
            str(source.get("run_id") or "").strip()
            if isinstance(source, Mapping)
            else ""
        )
    if refined_name and "/" in refined_name:
        return StageAvailability(
            stage_id="tracks",
            available=False,
            artifact_path=f"crop_runs/{crop_name}",
            run_name=keypoints.run_name,
            reason="selected keypoint source-crop refined lineage is malformed",
        )
    return lineage.crop_path, lineage.row_count, refined_name or None


def _tracking_authority_availability(
    root: Path,
    *,
    requested_run: str | None,
    selected_keypoint_run: str,
) -> StageAvailability:
    """Require selected tracks to bind the selected keypoint crop authority."""

    selected = discover_stage_availability(
        root,
        "tracks",
        requested_run=requested_run,
    )
    if not selected.available or not selected.artifact_path:
        return selected
    lineage = _keypoint_crop_lineage(
        root,
        selected_keypoint_run=selected_keypoint_run,
    )
    if isinstance(lineage, StageAvailability):
        return lineage
    expected_path, expected_rows, expected_refined = lineage
    tracking_attrs = _attrs(root / selected.artifact_path)
    actual_path = str(tracking_attrs.get("source_rowset_path") or "").strip().strip("/")
    actual_refined = str(tracking_attrs.get("source_refined_run") or "").strip()
    actual_rows_value = tracking_attrs.get("source_rowset_row_count")
    actual_rows = (
        int(actual_rows_value)
        if type(actual_rows_value) is int and actual_rows_value >= 0
        else None
    )
    mismatches: list[str] = []
    if actual_path != expected_path:
        mismatches.append(
            f"source_rowset_path={actual_path!r}, expected {expected_path!r}"
        )
    if expected_refined is not None and actual_refined != expected_refined:
        mismatches.append(
            f"source_refined_run={actual_refined!r}, expected {expected_refined!r}"
        )
    if (
        expected_rows is not None
        and actual_rows is not None
        and actual_rows != expected_rows
    ):
        mismatches.append(
            f"source_rowset_row_count={actual_rows}, expected {expected_rows}"
        )
    if mismatches:
        return StageAvailability(
            stage_id="tracks",
            available=False,
            artifact_path=selected.artifact_path,
            run_name=selected.run_name,
            reason=(
                "selected tracking authority does not match the selected "
                "keypoint crop lineage: " + "; ".join(mismatches)
            ),
            completion_status=selected.completion_status,
        )
    return StageAvailability(
        stage_id="tracks",
        available=True,
        artifact_path=selected.artifact_path,
        run_name=selected.run_name,
        reason="persisted tracking authority matches the selected keypoint crop lineage",
        completion_status=selected.completion_status,
    )


def _active_subject_mask_bundle_selection(
    root: Path,
) -> tuple[str, str] | None:
    attrs = _attrs(root)
    if SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR in attrs:
        return None
    authority = attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR)
    generation = attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR)
    if (
        not isinstance(authority, Mapping)
        or authority.get("schema_id") != "palette.subject_mask.bundle_authority"
        or authority.get("schema_version") != 1
        or type(generation) is not int
        or generation <= 0
        or authority.get("generation") != generation
    ):
        return None
    bundle_id = authority.get("bundle_id")
    bundle_path = authority.get("bundle_path")
    members = authority.get("members")
    refined = members.get("refined") if isinstance(members, Mapping) else None
    refined_path = refined.get("run_path") if isinstance(refined, Mapping) else None
    if (
        not isinstance(bundle_id, str)
        or bundle_path != f"subject_mask_bundle_runs/{bundle_id}"
        or not isinstance(refined_path, str)
        or not refined_path.startswith("refined_subject_masks_runs/")
        or len(PurePosixPath(refined_path).parts) != 2
    ):
        return None
    return bundle_id, refined_path


def _subject_mask_authority_availability(
    root: Path,
    *,
    requested_run: str | None,
) -> StageAvailability | None:
    root_attrs = _attrs(root)
    authority_present = SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR in root_attrs
    if SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR in root_attrs:
        return StageAvailability(
            stage_id="refined_subject_masks",
            available=False,
            reason="subject-mask bundle activation lease is present",
        )
    active = _active_subject_mask_bundle_selection(root)
    if authority_present and active is None:
        return StageAvailability(
            stage_id="refined_subject_masks",
            available=False,
            reason="subject-mask bundle authority is malformed or incomplete",
        )
    if active is None:
        return None
    bundle_id, refined_path = active
    requested = str(requested_run or "latest").strip().strip("/")
    refined_name = PurePosixPath(refined_path).name
    accepted = {
        "latest",
        bundle_id,
        f"bundle/{bundle_id}",
        refined_name,
        refined_path,
    }
    if requested not in accepted:
        return None
    bundle_attrs = _attrs(root / "subject_mask_bundle_runs" / bundle_id)
    refined_attrs = _attrs(root / refined_path)
    if (
        bundle_attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR) is not True
        or refined_attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR) is not True
    ):
        return StageAvailability(
            stage_id="refined_subject_masks",
            available=False,
            artifact_path=f"subject_mask_bundle_runs/{bundle_id}",
            run_name=f"bundle/{bundle_id}",
            reason="active subject-mask bundle readiness metadata is incomplete",
        )
    member = _available_selected_run(
        root,
        stage_id="refined_subject_masks",
        relative_parent="refined_subject_masks_runs",
        run_name=refined_name,
        encoded_run_name=f"bundle/{bundle_id}",
        reason="active subject-mask bundle authority is available",
        allow_selector_ineligible=True,
    )
    if not member.available:
        return member
    return StageAvailability(
        stage_id="refined_subject_masks",
        available=True,
        artifact_path=f"subject_mask_bundle_runs/{bundle_id}",
        run_name=f"bundle/{bundle_id}",
        reason="active subject-mask bundle authority is available",
        completion_status=member.completion_status,
    )


def _track_kinematics_visualization_availability(
    root: Path,
    *,
    requested_run: str | None,
    dependency_runs: Mapping[str, str] | None,
) -> StageAvailability:
    """Resolve the explorer contract embedded in one offline kinematics run."""

    parent = root / TRACK_KINEMATICS_PARENT
    if not (parent / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            reason="persisted track-kinematics run parent is missing",
        )
    parent_attrs = _attrs(parent)
    run_name, selection_error = _resolve_metadata_run_name(
        parent,
        parent_attrs,
        requested_run,
        parent_relative_path=TRACK_KINEMATICS_PARENT,
    )
    if run_name is None or selection_error is not None:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=(
                f"{TRACK_KINEMATICS_PARENT}/{run_name}"
                if run_name
                else TRACK_KINEMATICS_PARENT
            ),
            run_name=run_name,
            reason=selection_error or "track-kinematics run selection failed",
        )

    run_path = parent / run_name
    run_relative_path = f"{TRACK_KINEMATICS_PARENT}/{run_name}"
    if not (run_path / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=run_relative_path,
            run_name=run_name,
            reason="selected track-kinematics run metadata is missing",
        )
    run_attrs = _attrs(run_path)
    status = _completion_status(run_attrs)

    visualization_parent_relative = stage_run_relative_path(
        TRACK_KINEMATICS_VISUALIZATION_STAGE, run_name
    ) + "/tracks/id_0"
    visualization_parent = root / visualization_parent_relative
    if not (visualization_parent / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=visualization_parent_relative,
            run_name=run_name,
            reason="sibling track-kinematics visualization parent is missing",
            completion_status=status,
        )
    visualization_parent_attrs = _attrs(visualization_parent)
    render_name, render_selection_error = _resolve_metadata_run_name(
        visualization_parent,
        visualization_parent_attrs,
        None,
        parent_relative_path=visualization_parent_relative,
    )
    if render_name is None or render_selection_error is not None:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=visualization_parent_relative,
            run_name=run_name,
            reason=render_selection_error or "visualization render selection failed",
            completion_status=status,
        )
    render_relative_path = f"{visualization_parent_relative}/{render_name}"
    render_path = root / render_relative_path
    render_attrs = _attrs(render_path)
    render_status = _completion_status(render_attrs)
    artifact_relative_path = (
        f"{render_relative_path}/{TRACK_KINEMATICS_INTERACTIVE_ARTIFACT}"
    )
    artifact_path = root / artifact_relative_path
    if not (artifact_path / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive track-kinematics contract is missing",
            completion_status=status,
        )
    if not (artifact_path / "spec_json" / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive track-kinematics contract lacks spec_json",
            completion_status=status,
        )
    artifact_attrs = _attrs(artifact_path)
    renderer = str(artifact_attrs.get("renderer") or "").strip()
    if renderer != TRACK_KINEMATICS_INTERACTIVE_RENDERER:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason=(
                "interactive track-kinematics contract has unsupported renderer "
                f"{renderer!r}"
            ),
            completion_status=status,
        )
    motion_authority = artifact_attrs.get("track_motion_authority")
    motion_authority = (
        dict(motion_authority) if isinstance(motion_authority, Mapping) else {}
    )
    expected_run_ref = f"/{run_relative_path}"
    expected_track_ref = f"{expected_run_ref}/tracks/id_0"
    if (
        motion_authority.get("run_ref") != expected_run_ref
        or motion_authority.get("track_ref") != expected_track_ref
        or motion_authority.get("track_id") != 0
        or not str(motion_authority.get("motion_manifest_sha256") or "").strip()
        or not str(
            motion_authority.get("positions_px_coordinate_descriptor_sha256")
            or ""
        ).strip()
        or render_attrs.get("source_track_motion_authority") != motion_authority
        or render_attrs.get("track_id") != 0
    ):
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive contract lacks exact track-motion authority",
            completion_status=render_status,
        )
    expected_runs = dict(dependency_runs or {})
    source_runs = artifact_attrs.get("source_runs")
    source_runs = dict(source_runs) if isinstance(source_runs, Mapping) else {}
    expected_track_run = expected_runs.get("track_kinematics")
    persisted_track_run = str(source_runs.get("track_kinematics") or "").strip()
    if expected_track_run and persisted_track_run not in {
        expected_track_run,
        f"offline/{expected_track_run}",
    }:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive contract track-kinematics lineage does not match",
            completion_status=status,
        )
    parameters = artifact_attrs.get("parameters")
    parameters = dict(parameters) if isinstance(parameters, Mapping) else {}
    expected_swim_bout_run = expected_runs.get("swim_bouts")
    persisted_swim_bout_run = str(parameters.get("swim_bout_run") or "").strip()
    if expected_swim_bout_run and persisted_swim_bout_run != expected_swim_bout_run:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive contract swim-bout lineage does not match",
            completion_status=status,
        )
    return StageAvailability(
        stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
        available=True,
        artifact_path=artifact_relative_path,
        run_name=run_name,
        reason="persisted interactive track-kinematics contract is available",
        completion_status=render_status,
    )


def discover_stage_availability(
    zarr_path: str | Path,
    stage_id: str,
    *,
    requested_run: str | None = None,
    dependency_runs: Mapping[str, str] | None = None,
) -> StageAvailability:
    """Resolve one persisted run using direct ``zarr.json`` reads only."""

    canonical = canonical_stage_id(stage_id)
    root = Path(zarr_path)
    if canonical == "refined_keypoints":
        return _keypoint_authority_availability(
            root,
            requested_run=requested_run,
        )
    if canonical == "refined_subject_masks":
        bundle_status = _subject_mask_authority_availability(
            root,
            requested_run=requested_run,
        )
        if bundle_status is not None:
            return bundle_status
    if canonical == "tracks" and dependency_runs:
        selected_keypoints = dependency_runs.get("refined_keypoints")
        if selected_keypoints:
            return _tracking_authority_availability(
                root,
                requested_run=requested_run,
                selected_keypoint_run=selected_keypoints,
            )
    if canonical == TRACK_KINEMATICS_VISUALIZATION_STAGE:
        return _track_kinematics_visualization_availability(
            root,
            requested_run=requested_run,
            dependency_runs=dependency_runs,
        )
    parents = STAGE_RUN_PARENTS.get(canonical)
    if not parents:
        return StageAvailability(
            stage_id=canonical,
            available=False,
            reason="no metadata-only availability resolver is registered",
        )
    for relative_parent in parents:
        parent = root / relative_parent
        if not (parent / "zarr.json").is_file():
            continue
        parent_attrs = _attrs(parent)
        run_name, selection_error = _resolve_metadata_run_name(
            parent,
            parent_attrs,
            requested_run,
            parent_relative_path=relative_parent,
        )
        if run_name is None or selection_error is not None:
            return StageAvailability(
                stage_id=canonical,
                available=False,
                artifact_path=(
                    f"{relative_parent}/{run_name}"
                    if run_name
                    else relative_parent
                ),
                run_name=run_name,
                reason=selection_error or "run selection failed",
            )
        run_path = parent / run_name
        relative_run_path = f"{relative_parent}/{run_name}"
        if not (run_path / "zarr.json").is_file():
            return StageAvailability(
                stage_id=canonical,
                available=False,
                artifact_path=relative_run_path,
                run_name=run_name,
                reason="selected run metadata is missing",
            )
        run_attrs = _attrs(run_path)
        status = _completion_status(run_attrs)
        return StageAvailability(
            stage_id=canonical,
            available=True,
            artifact_path=relative_run_path,
            run_name=run_name,
            reason="persisted complete selector-eligible run is available",
            completion_status=status,
        )
    return StageAvailability(
        stage_id=canonical,
        available=False,
        reason="persisted run parent is missing",
    )


__all__ = [
    "STAGE_RUN_PARENTS",
    "StageAvailability",
    "discover_stage_availability",
    "stage_run_relative_path",
]
