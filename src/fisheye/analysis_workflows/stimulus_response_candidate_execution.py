"""Exact compact-v3 suite and authority binding for stimulus-response execution."""

from __future__ import annotations

from itertools import combinations
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.stimulus_epoch_schema import (
    stimulus_group_logical_fingerprint,
)
from fisheye.analysis.stimulus_response_storage import (
    STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
    build_stimulus_response_storage_receipt,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.stimulus_coordinate_contract import canonical_mapping_digest
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr.stimulus_response_schema import (
    KNOWN_BUNDLES,
    STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR,
    STIMULUS_RESPONSE_LAYOUT,
    STIMULUS_RESPONSE_SCHEMA_ID,
    STIMULUS_RESPONSE_SCHEMA_VERSION,
    stimulus_response_array_declarations,
    stimulus_response_array_manifest,
    validate_stimulus_response_v3_run,
)

STIMULUS_RESPONSE_EXECUTION_FAMILY_ID = "stimulus_response"
STIMULUS_RESPONSE_EXECUTION_PROFILE_ID = STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID
STIMULUS_RESPONSE_SOURCE_IDENTITY_SCHEMA_ID = (
    "palette.stimulus_response_execution_source_identity"
)
STIMULUS_RESPONSE_SOURCE_IDENTITY_SCHEMA_VERSION = 1
STIMULUS_RESPONSE_SOURCE_STAGING_MODE = "archive_snapshot_copy_v1"
STIMULUS_RESPONSE_SOURCE_COMPATIBILITY_ROLE = (
    "explicit_complete_eligible_compact_v3_benchmark_authority_nonproduction"
)

_RUN_PARENT = "analysis/stimulus_response_runs"
_ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
_SCIENTIFIC_PARAMETER_FIELDS = frozenset(
    {
        "moving_threshold_mm_s",
        "camera_to_projector_offset_deg",
        "bin_size_s",
        "follow_threshold",
        "follow_window_s",
        "omr_enabled",
        "omr_projection_deadzone",
        "omr_projection_speed_deadzone_mm_s",
        "omr_window_s",
        "omr_early_window_s",
        "center_threshold_mm",
        "concentric_radial_singularity_epsilon_mm",
        "escape_speed_threshold_mm_s",
        "escape_window_s",
        "loom_pre_onset_s",
        "loom_post_onset_s",
        "loom_bin_size_s",
    }
)
STIMULUS_RESPONSE_INVOCATION_FIELDS = frozenset(
    {
        "source_track_kinematics_scope",
        "source_track_kinematics_run",
        "source_track_motion_manifest_sha256",
        "source_stimulus_run",
        "source_stimulus_logical_tree_sha256",
        "source_stimulus_coordinate_lineage_sha256",
        "source_bout_mode",
        "source_swim_bout_run",
        "source_swim_bout_logical_tree_sha256",
        "scientific_parameters",
        "execution_backend",
        "source_staging_mode",
        "storage_profile_id",
        "copy_backend",
        "keep_scratch",
        "check_capacity",
    }
)


def _attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _array(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _require_run_path(value: object, *, parent: str, label: str) -> str:
    if type(value) is not str or not value.startswith(f"{parent}/"):
        raise ValueError(f"{label} must be one explicit child of {parent}")
    if value.count("/") != parent.count("/") + 1:
        raise ValueError(f"{label} must be one immediate immutable run child")
    child = value.rsplit("/", 1)[1]
    if (
        not child
        or child in _ALIASES
        or child in {".", ".."}
        or child != child.strip()
        or "\\" in child
        or any(character.isspace() for character in child)
    ):
        raise ValueError(f"{label} has an unsafe run name")
    return value


def _require_run_name(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string")
    _require_run_path(
        f"analysis/placeholder/{value}",
        parent="analysis/placeholder",
        label=label,
    )
    return value


def _finite_number(value: object, *, label: str, positive: bool = False) -> float:
    if type(value) not in {int, float} or not np.isfinite(float(value)):
        raise ValueError(f"{label} must be one finite number")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def require_stimulus_response_invocation_parameters(
    parameters: object,
) -> Mapping[str, Any]:
    """Validate the proposed shared ``stimulus_response_v1`` grammar."""

    if not isinstance(parameters, Mapping) or set(parameters) != set(
        STIMULUS_RESPONSE_INVOCATION_FIELDS
    ):
        raise ValueError("stimulus-response invocation parameter field set differs")
    if parameters["source_track_kinematics_scope"] != "offline":
        raise ValueError("stimulus-response track scope must be offline")
    track_run = parameters["source_track_kinematics_run"]
    stimulus_run = parameters["source_stimulus_run"]
    for value, label in ((track_run, "track run"), (stimulus_run, "stimulus run")):
        _require_run_name(value, label=label)
    for field in (
        "source_track_motion_manifest_sha256",
        "source_stimulus_logical_tree_sha256",
        "source_stimulus_coordinate_lineage_sha256",
    ):
        _require_sha256(parameters[field], label=field)
    mode = parameters["source_bout_mode"]
    if mode == "disabled":
        if (
            parameters["source_swim_bout_run"] is not None
            or parameters["source_swim_bout_logical_tree_sha256"] is not None
        ):
            raise ValueError("disabled bout mode requires null bout authority")
    elif mode == "explicit":
        bout_run = parameters["source_swim_bout_run"]
        _require_run_name(bout_run, label="swim-bout run")
        _require_sha256(
            parameters["source_swim_bout_logical_tree_sha256"],
            label="source_swim_bout_logical_tree_sha256",
        )
    else:
        raise ValueError("source_bout_mode must be disabled or explicit")
    scientific = parameters["scientific_parameters"]
    if not isinstance(scientific, Mapping) or set(scientific) != set(
        _SCIENTIFIC_PARAMETER_FIELDS
    ):
        raise ValueError("stimulus-response scientific parameter field set differs")
    for field in _SCIENTIFIC_PARAMETER_FIELDS - {
        "omr_enabled",
        "omr_window_s",
        "omr_early_window_s",
    }:
        _finite_number(
            scientific[field],
            label=field,
            positive=field
            in {
                "bin_size_s",
                "follow_window_s",
                "center_threshold_mm",
                "concentric_radial_singularity_epsilon_mm",
                "escape_speed_threshold_mm_s",
                "escape_window_s",
                "loom_pre_onset_s",
                "loom_post_onset_s",
                "loom_bin_size_s",
            },
        )
    if type(scientific["omr_enabled"]) is not bool:
        raise TypeError("omr_enabled must be an exact bool")
    for field in ("omr_window_s", "omr_early_window_s"):
        values = scientific[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} must be one nonempty JSON list")
        for index, item in enumerate(values):
            _finite_number(item, label=f"{field}[{index}]", positive=True)
    if parameters["execution_backend"] != "dask_threads_per_step_v1":
        raise ValueError("stimulus-response execution backend differs")
    if parameters["source_staging_mode"] != STIMULUS_RESPONSE_SOURCE_STAGING_MODE:
        raise ValueError("stimulus-response source staging mode differs")
    if parameters["storage_profile_id"] != STIMULUS_RESPONSE_EXECUTION_PROFILE_ID:
        raise ValueError("stimulus-response storage profile differs")
    if parameters["copy_backend"] not in {"python", "rsync"}:
        raise ValueError("copy_backend must be python or rsync")
    for field in ("keep_scratch", "check_capacity"):
        if type(parameters[field]) is not bool:
            raise TypeError(f"{field} must be an exact bool")
    canonical_json_bytes(parameters)
    return parameters


def stimulus_response_writer_arguments(
    parameters: Mapping[str, Any],
) -> tuple[str, ...]:
    """Translate one validated invocation into the complete scientific CLI."""

    parsed = require_stimulus_response_invocation_parameters(parameters)
    scientific = parsed["scientific_parameters"]
    result = [
        "--track-kinematics-type",
        "offline",
        "--track-kinematics-run",
        str(parsed["source_track_kinematics_run"]),
        "--stimulus-run",
        str(parsed["source_stimulus_run"]),
    ]
    if parsed["source_bout_mode"] == "disabled":
        result.append("--no-bouts")
    else:
        result.extend(("--bout-run", str(parsed["source_swim_bout_run"])))
    option_by_field = {
        "moving_threshold_mm_s": "--moving-threshold-mm-s",
        "camera_to_projector_offset_deg": "--camera-to-projector-offset-deg",
        "bin_size_s": "--bin-size-s",
        "follow_threshold": "--follow-threshold",
        "follow_window_s": "--follow-window-s",
        "omr_projection_deadzone": "--omr-projection-deadzone",
        "omr_projection_speed_deadzone_mm_s": ("--omr-projection-speed-deadzone-mm-s"),
        "center_threshold_mm": "--center-threshold-mm",
        "concentric_radial_singularity_epsilon_mm": (
            "--concentric-radial-singularity-epsilon-mm"
        ),
        "escape_speed_threshold_mm_s": "--escape-speed-threshold-mm-s",
        "escape_window_s": "--escape-window-s",
        "loom_pre_onset_s": "--loom-pre-onset-s",
        "loom_post_onset_s": "--loom-post-onset-s",
        "loom_bin_size_s": "--loom-bin-size-s",
    }
    for field, option in option_by_field.items():
        result.extend((option, str(scientific[field])))
    if not scientific["omr_enabled"]:
        result.append("--no-omr")
    for field, option in (
        ("omr_window_s", "--omr-window-s"),
        ("omr_early_window_s", "--omr-early-window-s"),
    ):
        for value in scientific[field]:
            result.extend((option, str(value)))
    return tuple(result)


def stimulus_response_bundles(source_group: Any) -> tuple[str, ...]:
    value = source_group.attrs.get("stimulus_response_v3_bundles")
    if not isinstance(value, (tuple, list)) or any(
        type(item) is not str for item in value
    ):
        raise ValueError("stimulus-response source lacks an exact bundle list")
    bundles = tuple(sorted(set(value)))
    if list(value) != list(bundles) or set(bundles) - set(KNOWN_BUNDLES):
        raise ValueError("stimulus-response source bundle list is not canonical")
    return bundles


def stimulus_response_arrays(source_group: Any) -> dict[str, Any]:
    bundles = stimulus_response_bundles(source_group)
    return {
        declaration.path: _array(source_group, declaration.path)
        for declaration in stimulus_response_array_declarations(
            bundles=bundles,
            byte_planner_adopted=False,
        )
    }


def compute_stimulus_response_logical_hashes(
    source_group: Any,
) -> dict[str, str]:
    """Hash every exact decoded compact-v3 array including dtype and shape."""

    hashes: dict[str, str] = {}
    for path, array in stimulus_response_arrays(source_group).items():
        values = np.ascontiguousarray(array[...])
        digest = hashlib.sha256()
        digest.update(str(np.dtype(array.dtype)).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        hashes[path] = digest.hexdigest()
    return hashes


def _validated_coordinate_lineage(source_refs: Mapping[str, Any]) -> dict[str, Any]:
    lineage = source_refs.get("stimulus_coordinate_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("stimulus-response source lacks coordinate lineage")
    result = dict(lineage)
    digest = result.pop("record_sha256", None)
    if digest != canonical_mapping_digest(result):
        raise ValueError("stimulus-response coordinate lineage digest differs")
    result["record_sha256"] = digest
    return result


def _source_authorities(
    root: Any,
    source_group: Any,
) -> dict[str, Any]:
    attrs = _attrs(source_group)
    refs = attrs.get("source_refs")
    if not isinstance(refs, Mapping):
        raise ValueError("stimulus-response source_refs must be one mapping")
    track_path = _require_run_path(
        refs.get("source_track_kinematics_run"),
        parent="analysis/track_kinematics_runs/offline",
        label="source track-kinematics path",
    )
    stimulus_path = _require_run_path(
        refs.get("source_stimulus_run"),
        parent="analysis/stimulus_runs",
        label="source stimulus path",
    )
    upstream = refs.get("upstream_lineage")
    if not isinstance(upstream, Mapping):
        raise ValueError("stimulus-response source lacks upstream track lineage")
    track_manifest_sha256 = _require_sha256(
        upstream.get("source_track_motion_manifest_sha256"),
        label="source track-motion manifest",
    )
    if upstream.get("source_track_motion_run_ref") != f"/{track_path}":
        raise ValueError("stimulus-response track lineage path differs")
    track = root[track_path]
    if track.attrs.get("track_motion_publication_manifest_sha256") != (
        track_manifest_sha256
    ):
        raise ValueError("live track-motion authority differs from source lineage")
    coordinate = _validated_coordinate_lineage(refs)
    if coordinate.get("source_stimulus_run_ref") != f"/{stimulus_path}":
        raise ValueError("stimulus coordinate lineage selects another run")
    stimulus_sha256 = stimulus_group_logical_fingerprint(root[stimulus_path])

    bout_path = refs.get("source_bout_run")
    if bout_path is None:
        bout_identity_sha256 = None
    else:
        bout_path = _require_run_path(
            bout_path,
            parent="analysis/swim_bout_runs",
            label="source swim-bout path",
        )
        bout_identity_sha256 = stimulus_group_logical_fingerprint(root[bout_path])
    return {
        "track_run_path": track_path,
        "track_motion_manifest_sha256": track_manifest_sha256,
        "stimulus_run_path": stimulus_path,
        "stimulus_logical_tree_sha256": stimulus_sha256,
        "stimulus_coordinate_lineage": json_attr_safe(coordinate),
        "stimulus_coordinate_lineage_sha256": str(coordinate["record_sha256"]),
        "swim_bout_run_path": bout_path,
        "swim_bout_logical_tree_sha256": bout_identity_sha256,
    }


def build_stimulus_response_source_identity(
    root: Any,
    *,
    source_run_path: str,
) -> dict[str, Any]:
    """Bind one compatible compact-v3 source and all scientific authorities."""

    path = _require_run_path(
        source_run_path,
        parent=_RUN_PARENT,
        label="stimulus-response source path",
    )
    source = root[path]
    errors = validate_stimulus_response_v3_run(source)
    if errors:
        raise ValueError("invalid stimulus-response source: " + "; ".join(errors))
    attrs = _attrs(source)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not True
        or attrs.get("schema_id") != STIMULUS_RESPONSE_SCHEMA_ID
        or attrs.get("schema_version") != STIMULUS_RESPONSE_SCHEMA_VERSION
        or attrs.get("layout") != STIMULUS_RESPONSE_LAYOUT
    ):
        raise ValueError(
            "stimulus-response benchmark compatibility source is not exact, "
            "complete, and explicitly eligible"
        )
    bundles = stimulus_response_bundles(source)
    expected_manifest = stimulus_response_array_manifest(
        bundles=bundles,
        byte_planner_adopted=False,
    )
    if attrs.get(STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR) != expected_manifest:
        raise ValueError("stimulus-response source array manifest differs")
    parameters = attrs.get("parameters")
    source_refs = attrs.get("source_refs")
    if not isinstance(parameters, Mapping) or not isinstance(source_refs, Mapping):
        raise ValueError("stimulus-response source parameters or refs are absent")
    authorities = _source_authorities(root, source)
    logical_hashes = compute_stimulus_response_logical_hashes(source)
    document = {
        "schema_id": STIMULUS_RESPONSE_SOURCE_IDENTITY_SCHEMA_ID,
        "schema_version": STIMULUS_RESPONSE_SOURCE_IDENTITY_SCHEMA_VERSION,
        "compatibility_role": STIMULUS_RESPONSE_SOURCE_COMPATIBILITY_ROLE,
        "source_run_path": path,
        "source_schema_id": attrs["schema_id"],
        "source_schema_version": attrs["schema_version"],
        "source_layout": attrs["layout"],
        "source_method_version": attrs.get("method_version"),
        "bundles": list(bundles),
        "parameters": json_attr_safe(dict(parameters)),
        "source_refs": json_attr_safe(dict(source_refs)),
        "source_authorities": authorities,
        "source_array_manifest": expected_manifest,
        "source_array_logical_hashes": logical_hashes,
        "source_array_logical_manifest_sha256": canonical_json_sha256(logical_hashes),
    }
    canonical_json_bytes(document)
    return document


def build_stimulus_response_coordinate_evidence(
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the exact input authorities into shared bound-derivative evidence."""

    if source_identity.get("schema_id") != STIMULUS_RESPONSE_SOURCE_IDENTITY_SCHEMA_ID:
        raise ValueError("stimulus-response source identity schema differs")
    authorities = source_identity.get("source_authorities")
    if not isinstance(authorities, Mapping):
        raise ValueError("stimulus-response source authorities are absent")
    records = [
        {
            "role": "source_track_motion_publication",
            "sha256": _require_sha256(
                authorities.get("track_motion_manifest_sha256"),
                label="track-motion authority",
            ),
        },
        {
            "role": "source_stimulus_coordinate_lineage",
            "sha256": _require_sha256(
                authorities.get("stimulus_coordinate_lineage_sha256"),
                label="stimulus coordinate authority",
            ),
        },
        {
            "role": "source_stimulus_logical_tree",
            "sha256": _require_sha256(
                authorities.get("stimulus_logical_tree_sha256"),
                label="stimulus logical tree",
            ),
        },
    ]
    bout_digest = authorities.get("swim_bout_logical_tree_sha256")
    if bout_digest is not None:
        records.append(
            {
                "role": "source_swim_bout_logical_tree",
                "sha256": _require_sha256(
                    bout_digest,
                    label="swim-bout logical tree",
                ),
            }
        )
    records.sort(key=lambda record: str(record["role"]))
    validation = {
        "schema_id": "palette.stimulus_response_bound_authority_validation",
        "schema_version": 1,
        "source_identity_sha256": canonical_json_sha256(source_identity),
        "source_authority_digests": records,
    }
    digest = canonical_json_sha256(validation)
    return {
        "role": "bound_derivative",
        "status": "verified_bound_source",
        "source_authority_digests": records,
        "published_authority_sha256": None,
        "published_authority_ref": None,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": (f"{__name__}:build_stimulus_response_coordinate_evidence"),
        "validation_receipt_sha256": digest,
        "coordinate_gate_passed": True,
    }


def require_stimulus_response_candidate_scientific_binding(
    candidate_group: Any,
    *,
    source_identity: Mapping[str, Any],
) -> None:
    """Require recomputation to preserve every scientific input and parameter."""

    if source_identity.get("schema_id") != STIMULUS_RESPONSE_SOURCE_IDENTITY_SCHEMA_ID:
        raise ValueError("stimulus-response source identity schema differs")
    source_parameters = source_identity.get("parameters")
    source_refs = source_identity.get("source_refs")
    authorities = source_identity.get("source_authorities")
    if not all(
        isinstance(value, Mapping)
        for value in (source_parameters, source_refs, authorities)
    ):
        raise ValueError("stimulus-response source identity is incomplete")
    candidate_attrs = _attrs(candidate_group)
    candidate_parameters = candidate_attrs.get("parameters")
    expected_parameters = dict(source_parameters)
    expected_parameters["layout"] = STIMULUS_RESPONSE_LAYOUT
    expected_parameters["storage_profile_id"] = STIMULUS_RESPONSE_EXECUTION_PROFILE_ID
    if candidate_parameters != expected_parameters:
        raise ValueError("stimulus-response candidate parameters differ from source")
    if candidate_attrs.get("source_refs") != dict(source_refs):
        raise ValueError("stimulus-response candidate source refs differ from source")

    track_path = str(authorities["track_run_path"])
    stimulus_path = str(authorities["stimulus_run_path"])
    bout_path = authorities["swim_bout_run_path"]
    if (
        candidate_attrs.get("source_track_kinematics_type") != "offline"
        or candidate_attrs.get("source_track_kinematics_run")
        != track_path.rsplit("/", 1)[1]
        or candidate_attrs.get("source_stimulus_run") != stimulus_path.rsplit("/", 1)[1]
        or candidate_attrs.get("source_bout_run")
        != (None if bout_path is None else str(bout_path).rsplit("/", 1)[1])
    ):
        raise ValueError("stimulus-response candidate source aliases differ")


def build_stimulus_response_execution_suite(
    source_group: Any,
    *,
    seed: int = 31,
    repetitions: int = 5,
) -> dict[str, object]:
    """Build one exact compact-v3 byte-planned candidate suite."""

    bundles = stimulus_response_bundles(source_group)
    receipt = build_stimulus_response_storage_receipt(
        arrays_by_path=stimulus_response_arrays(source_group),
        bundles=bundles,
        profile=get_storage_profile(STIMULUS_RESPONSE_EXECUTION_PROFILE_ID),
    )
    suite = build_analysis_benchmark_suite(
        family_id=STIMULUS_RESPONSE_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id="explicit_stimulus_response_compact_v3_run",
            dimensions=receipt.dimensions,
            description=(
                "Exact compatible compact-v3 response tables recomputed as one "
                "selector-ineligible byte-planned compact-v3 candidate."
            ),
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_stimulus_response_execution_suite(
        STIMULUS_RESPONSE_EXECUTION_FAMILY_ID,
        suite,
    )
    return suite


def _infer_bundles(paths: set[str]) -> tuple[str, ...]:
    known = sorted(KNOWN_BUNDLES)
    matches: list[tuple[str, ...]] = []
    for count in range(len(known) + 1):
        for selected in combinations(known, count):
            expected = {
                item.path
                for item in stimulus_response_array_declarations(
                    bundles=selected,
                    byte_planner_adopted=True,
                )
            }
            if expected == paths:
                matches.append(selected)
    if len(matches) != 1:
        raise ValueError("stimulus-response suite does not identify one bundle set")
    return matches[0]


def require_stimulus_response_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Require the exact live compact-v3 byte-planned declaration replay."""

    if stage_id != STIMULUS_RESPONSE_EXECUTION_FAMILY_ID:
        raise ValueError("stimulus-response suite validator owns only its family")
    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if payload["family_id"] != stage_id:
        raise ValueError("stimulus-response suite family differs")
    receipt = payload["storage_plan_receipt"]["payload"]
    if receipt["storage_profile"]["profile_id"] != (
        STIMULUS_RESPONSE_EXECUTION_PROFILE_ID
    ):
        raise ValueError("stimulus-response suite profile differs")
    records = receipt.get("arrays")
    if not isinstance(records, list) or not records:
        raise ValueError("stimulus-response suite has no arrays")
    paths: set[str] = set()
    observed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("stimulus-response suite array record is invalid")
        facts = record.get("observed_facts")
        declaration = record.get("declaration")
        if not isinstance(facts, Mapping) or not isinstance(declaration, Mapping):
            raise ValueError("stimulus-response suite lacks facts or declaration")
        path = facts.get("path")
        if type(path) is not str or path in paths:
            raise ValueError("stimulus-response suite path is invalid or repeated")
        paths.add(path)
        if declaration.get("path") != path:
            raise ValueError("stimulus-response suite declaration path differs")
        observed[path] = declaration
    bundles = _infer_bundles(paths)
    expected = {
        declaration.path: declaration.as_manifest()
        for declaration in stimulus_response_array_declarations(
            bundles=bundles,
            byte_planner_adopted=True,
        )
    }
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError("stimulus-response suite declarations differ")


__all__ = [
    "STIMULUS_RESPONSE_EXECUTION_FAMILY_ID",
    "STIMULUS_RESPONSE_EXECUTION_PROFILE_ID",
    "STIMULUS_RESPONSE_INVOCATION_FIELDS",
    "STIMULUS_RESPONSE_SOURCE_IDENTITY_SCHEMA_ID",
    "STIMULUS_RESPONSE_SOURCE_IDENTITY_SCHEMA_VERSION",
    "STIMULUS_RESPONSE_SOURCE_COMPATIBILITY_ROLE",
    "STIMULUS_RESPONSE_SOURCE_STAGING_MODE",
    "build_stimulus_response_coordinate_evidence",
    "build_stimulus_response_execution_suite",
    "build_stimulus_response_source_identity",
    "compute_stimulus_response_logical_hashes",
    "require_stimulus_response_execution_suite",
    "require_stimulus_response_invocation_parameters",
    "require_stimulus_response_candidate_scientific_binding",
    "stimulus_response_arrays",
    "stimulus_response_bundles",
    "stimulus_response_writer_arguments",
]
