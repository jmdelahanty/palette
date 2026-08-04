"""Family-local suite validation for the diagnostic track-flat candidate."""

from __future__ import annotations

import re
from typing import Any, Mapping

from fisheye.analysis.track_kinematics_schema import (
    TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS,
    build_track_kinematics_flat_lineage_declarations,
)
from fisheye.analysis.track_kinematics_storage import (
    build_flat_candidate_declarations,
    build_flat_candidate_storage_receipt,
)
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes
from fisheye.shared.zarr.storage_profiles import get_storage_profile

_TRACK_PATH = re.compile(r"^tracks/id_([0-9]+)/(.+)$")
_PHYSICAL_RELATIVE_PATHS = frozenset(
    declaration.relative_path
    for declaration in TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS
)


def build_track_flat_execution_suite(
    source_group: Any,
    *,
    storage_profile_id: str,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    """Build the exact no-physical suite accepted by the typed runner."""

    declarations = build_flat_candidate_declarations(source_group)
    physical_paths = [
        declaration.path
        for declaration in declarations
        if declaration.path.endswith("/positions_mm")
    ]
    if physical_paths:
        raise ValueError("track_flat_v1 explicitly excludes the physical track bundle")
    receipt = build_flat_candidate_storage_receipt(
        source_group,
        profile=get_storage_profile(storage_profile_id),
    )
    suite = build_analysis_benchmark_suite(
        family_id="track_kinematics",
        scale=AnalysisBenchmarkScale(
            scale_id="explicit_track_run",
            dimensions=receipt.dimensions,
            description=(
                "Exact explicit offline track-v1 run projected into primitive flat-v2 "
                "lineage arrays; physical coordinate peers are excluded."
            ),
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_track_flat_execution_suite("track_kinematics", suite)
    return suite


def _records(
    benchmark_suite: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    payload = benchmark_suite["payload"]
    records = payload["storage_plan_receipt"]["payload"]["arrays"]
    if not isinstance(records, list) or not records:
        raise ValueError("track-flat suite must plan one nonempty array set")
    result: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("track-flat suite contains a non-object array record")
        facts = record.get("observed_facts")
        declaration = record.get("declaration")
        if not isinstance(facts, Mapping) or not isinstance(declaration, Mapping):
            raise ValueError("track-flat suite array lacks facts or declaration")
        if (
            type(facts.get("path")) is not str
            or not isinstance(facts.get("shape"), list)
            or type(facts.get("dtype")) is not str
        ):
            raise ValueError("track-flat suite array facts are not canonical")
        result.append(record)
    return tuple(result)


def require_track_flat_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Require one exact no-physical flat-v2 track projection and byte plan."""

    if stage_id != "track_kinematics":
        raise ValueError("track-flat suite validator owns only track_kinematics")
    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if payload["family_id"] != stage_id:
        raise ValueError("track-flat suite family differs from track_kinematics")

    records = _records(benchmark_suite)
    declarations: list[Mapping[str, Any]] = []
    track_ids: set[int] = set()
    physical_paths: list[str] = []
    facts_by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        facts = record["observed_facts"]
        path = str(facts["path"])
        if path in facts_by_path:
            raise ValueError(f"track-flat suite repeats array path {path!r}")
        facts_by_path[path] = facts
        declarations.append(record["declaration"])
        match = _TRACK_PATH.fullmatch(path)
        if match is not None:
            track_id = int(match.group(1))
            track_ids.add(track_id)
            if match.group(2) in _PHYSICAL_RELATIVE_PATHS:
                physical_paths.append(path)

    if physical_paths:
        raise ValueError(
            "track_flat_v1 explicitly excludes physical bundle arrays: "
            f"{sorted(physical_paths)!r}"
        )
    if not track_ids:
        raise ValueError("track-flat suite contains no track IDs")
    if "track_ids" not in facts_by_path:
        raise ValueError("track-flat suite lacks the track_ids inventory")
    track_ids_facts = facts_by_path["track_ids"]
    if track_ids_facts["dtype"] != "int32" or track_ids_facts["shape"] != [
        len(track_ids)
    ]:
        raise ValueError("track-flat track_ids facts differ from its inventory")
    include_arena = "track_arena_ids" in facts_by_path
    if include_arena:
        arena = facts_by_path["track_arena_ids"]
        if arena["dtype"] != "int32" or arena["shape"] != [len(track_ids)]:
            raise ValueError("track-flat arena inventory facts differ")

    expected = [
        declaration.as_manifest()
        for declaration in build_track_kinematics_flat_lineage_declarations(
            track_ids=tuple(sorted(track_ids)),
            include_physical=False,
            include_arena_inventory=include_arena,
        )
    ]
    observed_by_path = {
        str(declaration["path"]): declaration for declaration in declarations
    }
    expected_by_path = {
        str(declaration["path"]): declaration for declaration in expected
    }
    if canonical_json_bytes(observed_by_path) != canonical_json_bytes(expected_by_path):
        raise ValueError(
            "track-flat suite declarations differ from the live flat-v2 projection"
        )


__all__ = [
    "build_track_flat_execution_suite",
    "require_track_flat_execution_suite",
]
