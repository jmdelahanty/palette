"""Audit derived analysis runs for stale or unverifiable upstream lineage.

This utility is intentionally read-only. It checks run-level source references
recorded on ``analysis/*_runs`` groups against the source groups currently
present in the same Palette Zarr archive.
"""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr_paths
import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fisheye.shared.run_lineage_fingerprint import normalize_lineage_value
from fisheye.shared.zarr_io import open_zarr_root


@dataclass(frozen=True)
class RunParentSpec:
    family: str
    parent_path: tuple[str, ...]


@dataclass
class DirectJsonNode:
    """Minimal Zarr-like node loaded directly from metadata files."""

    attrs: dict[str, Any]


@dataclass(frozen=True)
class SourceRef:
    key: str
    path: str
    expected_fingerprint: str | None = None


@dataclass(frozen=True)
class SourceAudit:
    key: str
    path: str
    status: str
    message: str
    expected_fingerprint: str | None = None
    actual_fingerprint: str | None = None
    actual_fingerprint_status: str | None = None
    latest_parent_path: str | None = None
    latest_run_id: str | None = None
    referenced_run_id: str | None = None


@dataclass(frozen=True)
class RunAudit:
    zarr_path: str
    run_family: str
    run_id: str
    run_path: str
    status: str
    sources: list[SourceAudit] = field(default_factory=list)
    explicit_stale_attrs: list[str] = field(default_factory=list)
    message: str = ""


RUN_PARENT_SPECS: tuple[RunParentSpec, ...] = (
    RunParentSpec("track_kinematics_run", ("analysis", "track_kinematics_runs", "offline")),
    RunParentSpec("track_kinematics_run", ("analysis", "track_kinematics_runs", "online")),
    RunParentSpec("swim_bout_run", ("analysis", "swim_bout_runs")),
    RunParentSpec("bout_kinematics_run", ("analysis", "bout_kinematics_runs")),
    RunParentSpec("stimulus_run", ("analysis", "stimulus_runs")),
    RunParentSpec("stimulus_response_run", ("analysis", "stimulus_response_runs")),
    RunParentSpec("eye_angle_run", ("analysis", "eye_angle_runs")),
    RunParentSpec("subject_shape_run", ("analysis", "subject_shape_runs")),
    RunParentSpec("tail_kinematics_run", ("analysis", "tail_kinematics_runs")),
    RunParentSpec("tail_posture_view_run", ("analysis", "tail_posture_view_runs")),
    RunParentSpec("bout_classification_run", ("analysis", "bout_classification_runs")),
)

SOURCE_RUN_ATTR_CANDIDATE_PARENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "source_detect_run": (("detect_runs",),),
    "source_refined_run": (("refined_detect_runs",),),
    "source_refined_detect_run": (("refined_detect_runs",),),
    "source_crop_run": (("crop_runs",),),
    "source_keypoint_run": (("refined_keypoints_runs",), ("keypoints_runs",)),
    "source_keypoints_run": (("refined_keypoints_runs",), ("keypoints_runs",)),
    "source_refined_keypoints_run": (("refined_keypoints_runs",),),
    "source_subject_mask_run": (("subject_mask_runs",),),
    "source_refined_subject_masks_run": (("refined_subject_masks_runs",),),
    "source_refined_eye_run": (("refined_eye_masks_runs",),),
    "source_eye_masks_run": (("eye_masks_runs",),),
    "source_tracking_run": (("tracking_runs",),),
    "source_track_kinematics_run": (
        ("analysis", "track_kinematics_runs", "offline"),
        ("analysis", "track_kinematics_runs", "online"),
    ),
    "source_swim_bout_run": (("analysis", "swim_bout_runs"),),
    "source_bout_kinematics_run": (("analysis", "bout_kinematics_runs"),),
    "source_stimulus_run": (("analysis", "stimulus_runs"),),
    "source_stimulus_response_run": (("analysis", "stimulus_response_runs"),),
    "source_eye_angle_run": (("analysis", "eye_angle_runs"),),
    "source_subject_shape_run": (("analysis", "subject_shape_runs"),),
    "source_tail_kinematics_run": (("analysis", "tail_kinematics_runs"),),
    "source_tail_posture_view_run": (("analysis", "tail_posture_view_runs"),),
    "source_bout_classification_run": (("analysis", "bout_classification_runs"),),
}

STALE_SOURCE_STATUSES = {
    "stale",
    "missing_source",
    "source_explicit_stale",
}
WARNING_SOURCE_STATUSES = {
    "source_not_latest",
    "unverifiable_missing_actual_fingerprint",
    "unverifiable_missing_expected_fingerprint",
}


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _coerce_mapping(value: Any) -> dict[str, Any]:
    value = _parse_jsonish(value)
    return dict(value) if isinstance(value, Mapping) else {}


def _attrs_dict(group: Any) -> dict[str, Any]:
    attrs = getattr(group, "attrs", {})
    try:
        return {str(key): _parse_jsonish(value) for key, value in attrs.items()}
    except AttributeError:
        return {}


def _get_nested_group(root: Any, path: Sequence[str]) -> Any | None:
    group = root
    for part in path:
        try:
            if part not in group:
                return None
            group = group[part]
        except Exception:
            return None
    return group


def _group_names(group: Any) -> list[str]:
    try:
        return sorted(str(name) for name in group.group_keys())
    except Exception:
        try:
            return sorted(str(name) for name in group.keys())
        except Exception:
            return []


def _normalize_internal_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace("\\", "/")
    if not text:
        return None
    marker = ".zarr/"
    if marker in text:
        text = text.split(marker, 1)[1]
    elif text.startswith("/"):
        # Absolute filesystem inputs are external provenance, not same-archive
        # Zarr paths. Without an explicit ".zarr/<internal>" suffix there is
        # no internal group/array to resolve.
        return None
    if "?" in text:
        # Compact layouts may record logical table selections such as
        # "tables/bouts?candidate_id=0&signal_id=4". The table path is the
        # resolvable Zarr node; the query string is row-selection provenance.
        text = text.split("?", 1)[0]
    text = text.lstrip("/")
    if not text or text.startswith(("http://", "https://", "s3://")):
        return None
    if "/" not in text:
        return None
    return "/".join(part for part in text.split("/") if part)


def _source_path_from_run_attr(root: Any, attr_name: str, run_id: Any) -> str | None:
    if not isinstance(run_id, str) or not run_id.strip():
        return None
    normalized = _normalize_internal_path(run_id)
    if normalized is not None:
        return normalized
    parents = SOURCE_RUN_ATTR_CANDIDATE_PARENTS.get(attr_name)
    if not parents:
        return None
    clean_run_id = run_id.strip()
    candidate_paths = ["/".join((*parent, clean_run_id)) for parent in parents]
    for candidate in candidate_paths:
        if _resolve_internal_path(root, candidate) is not None:
            return candidate
    return candidate_paths[0] if candidate_paths else None


def _extract_fingerprint(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in (
            "fingerprint",
            "source_fingerprint",
            "source_lineage_hash",
            "lineage_hash",
        ):
            fingerprint = _extract_fingerprint(value.get(key))
            if fingerprint:
                return fingerprint
    return None


def _source_ref_from_mapping(key: str, value: Mapping[str, Any]) -> SourceRef | None:
    path = None
    for path_key in ("path", "run_path", "zarr_path", "source_path", "array_path"):
        path = _normalize_internal_path(value.get(path_key))
        if path:
            break
    if path is None:
        return None
    return SourceRef(key=key, path=path, expected_fingerprint=_extract_fingerprint(value))


def _fingerprint_from_source_fingerprints(
    source_fingerprints: Mapping[str, Any],
    *,
    key: str,
    path: str,
) -> str | None:
    candidates = (
        key,
        path,
        f"{key}_fingerprint",
        Path(path).name,
        f"{Path(path).name}_fingerprint",
    )
    for candidate in candidates:
        fingerprint = _extract_fingerprint(source_fingerprints.get(candidate))
        if fingerprint:
            return fingerprint
    return None


def discover_source_refs(root: Any, run_group: Any) -> list[SourceRef]:
    """Discover same-archive source references from run attrs.

    The resolver is deliberately conservative: it uses explicit paths first and
    falls back to common ``source_*_run`` attrs only when they can be mapped to a
    known stage family.
    """

    attrs = _attrs_dict(run_group)
    source_refs = _coerce_mapping(attrs.get("source_refs"))
    source_fingerprints = _coerce_mapping(attrs.get("source_fingerprints"))
    refs: list[SourceRef] = []
    seen: set[tuple[str, str]] = set()

    def add(ref: SourceRef) -> None:
        key = (ref.key, ref.path)
        if key in seen:
            return
        seen.add(key)
        refs.append(ref)

    for raw_key, raw_value in source_refs.items():
        key = str(raw_key)
        value = _parse_jsonish(raw_value)
        ref: SourceRef | None = None
        if isinstance(value, Mapping):
            ref = _source_ref_from_mapping(key, value)
        else:
            path = _normalize_internal_path(value)
            if path is None and key in SOURCE_RUN_ATTR_CANDIDATE_PARENTS:
                path = _source_path_from_run_attr(root, key, value)
            if path:
                ref = SourceRef(
                    key=key,
                    path=path,
                    expected_fingerprint=_fingerprint_from_source_fingerprints(
                        source_fingerprints,
                        key=key,
                        path=path,
                    ),
                )
        if ref is not None and ref.expected_fingerprint is None:
            ref = SourceRef(
                key=ref.key,
                path=ref.path,
                expected_fingerprint=_fingerprint_from_source_fingerprints(
                    source_fingerprints,
                    key=ref.key,
                    path=ref.path,
                ),
            )
        if ref is not None:
            add(ref)

    for key, value in attrs.items():
        if key in {"source_refs", "source_fingerprints"}:
            continue
        if key.endswith("_path") and key.startswith("source_"):
            path = _normalize_internal_path(value)
        elif key in SOURCE_RUN_ATTR_CANDIDATE_PARENTS:
            path = _source_path_from_run_attr(root, key, value)
        else:
            path = None
        if path:
            add(
                SourceRef(
                    key=key,
                    path=path,
                    expected_fingerprint=_fingerprint_from_source_fingerprints(
                        source_fingerprints,
                        key=key,
                        path=path,
                    ),
                )
            )
    return refs


def _direct_json_node(archive_path: Path | None, path: str) -> DirectJsonNode | None:
    if archive_path is None:
        return None
    node_path = Path(archive_path).joinpath(*(part for part in path.split("/") if part))
    for metadata_name, attrs_key in (("zarr.json", "attributes"), (".zattrs", None)):
        metadata_path = node_path / metadata_name
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        attrs = payload.get(attrs_key) if attrs_key is not None else payload
        if isinstance(attrs, Mapping):
            return DirectJsonNode(
                attrs={str(key): _parse_jsonish(value) for key, value in attrs.items()}
            )
        return DirectJsonNode(attrs={})
    return None


def _resolve_internal_path(
    root: Any,
    path: str,
    *,
    archive_path: Path | None = None,
) -> Any | None:
    node = _get_nested_group(root, tuple(part for part in path.split("/") if part))
    if node is not None:
        return node
    return _direct_json_node(archive_path, path)


def _source_actual_fingerprint(group: Any) -> tuple[str | None, str | None]:
    attrs = _attrs_dict(group)
    for key in ("lineage_hash", "source_lineage_hash", "source_fingerprint"):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), str(attrs.get("fingerprint_status") or "")
    return None, str(attrs.get("fingerprint_status") or "")


def _explicit_stale_attrs(group: Any) -> list[str]:
    stale_keys: list[str] = []
    attrs = _attrs_dict(group)
    for key, value in attrs.items():
        parsed = _parse_jsonish(value)
        if isinstance(parsed, Mapping):
            state = str(parsed.get("state") or parsed.get("lifecycle_state") or "").strip().lower()
            if key.endswith("_stale") and state == "stale":
                stale_keys.append(key)
        elif key in {"stale", "quality_stale"} and bool(parsed):
            stale_keys.append(key)
        elif key in {"lifecycle_state", "stale_state"} and str(parsed).strip().lower() == "stale":
            stale_keys.append(key)
    return sorted(stale_keys)


def _latest_reference(root: Any, path: str) -> tuple[str, str, str] | None:
    parts = [part for part in path.split("/") if part]
    group = root
    latest_hit: tuple[str, str, str] | None = None
    traversed: list[str] = []
    for part in parts:
        attrs = _attrs_dict(group)
        latest = attrs.get("latest")
        if isinstance(latest, str) and latest.strip():
            latest_hit = ("/".join(traversed), latest.strip(), part)
        try:
            if part not in group:
                break
            group = group[part]
            traversed.append(part)
        except Exception:
            break
    return latest_hit


def audit_source_ref(
    root: Any,
    ref: SourceRef,
    *,
    zarr_path: Path | None = None,
    require_latest_sources: bool = False,
) -> SourceAudit:
    source_group = _resolve_internal_path(root, ref.path, archive_path=zarr_path)
    latest = _latest_reference(root, ref.path)
    latest_parent_path = latest[0] if latest else None
    latest_run_id = latest[1] if latest else None
    referenced_run_id = latest[2] if latest else None
    if source_group is None:
        return SourceAudit(
            key=ref.key,
            path=ref.path,
            status="missing_source",
            message="source path does not resolve in archive",
            expected_fingerprint=ref.expected_fingerprint,
            latest_parent_path=latest_parent_path,
            latest_run_id=latest_run_id,
            referenced_run_id=referenced_run_id,
        )
    source_stale_attrs = _explicit_stale_attrs(source_group)
    actual, actual_status = _source_actual_fingerprint(source_group)
    if source_stale_attrs:
        return SourceAudit(
            key=ref.key,
            path=ref.path,
            status="source_explicit_stale",
            message=f"source carries explicit stale attrs: {', '.join(source_stale_attrs)}",
            expected_fingerprint=ref.expected_fingerprint,
            actual_fingerprint=actual,
            actual_fingerprint_status=actual_status or None,
            latest_parent_path=latest_parent_path,
            latest_run_id=latest_run_id,
            referenced_run_id=referenced_run_id,
        )
    not_latest = latest_run_id is not None and referenced_run_id is not None and latest_run_id != referenced_run_id
    if ref.expected_fingerprint and actual and ref.expected_fingerprint != actual:
        return SourceAudit(
            key=ref.key,
            path=ref.path,
            status="stale",
            message="stored source fingerprint differs from current source fingerprint",
            expected_fingerprint=ref.expected_fingerprint,
            actual_fingerprint=actual,
            actual_fingerprint_status=actual_status or None,
            latest_parent_path=latest_parent_path,
            latest_run_id=latest_run_id,
            referenced_run_id=referenced_run_id,
        )
    if not_latest and require_latest_sources:
        status = "stale"
        message = "source does not match parent latest selection"
    elif not_latest:
        status = "source_not_latest"
        message = "source does not match parent latest selection"
    elif not ref.expected_fingerprint:
        status = "unverifiable_missing_expected_fingerprint"
        message = "run does not record an expected fingerprint for this source"
    elif not actual:
        status = "unverifiable_missing_actual_fingerprint"
        message = "source does not expose a run-level fingerprint"
    else:
        status = "fresh"
        message = "source fingerprint matches"
    return SourceAudit(
        key=ref.key,
        path=ref.path,
        status=status,
        message=message,
        expected_fingerprint=ref.expected_fingerprint,
        actual_fingerprint=actual,
        actual_fingerprint_status=actual_status or None,
        latest_parent_path=latest_parent_path,
        latest_run_id=latest_run_id,
        referenced_run_id=referenced_run_id,
    )


def _run_status(
    sources: Sequence[SourceAudit],
    *,
    explicit_stale_attrs: Sequence[str],
    require_latest_sources: bool,
) -> str:
    if explicit_stale_attrs:
        return "stale"
    stale_statuses = set(STALE_SOURCE_STATUSES)
    if require_latest_sources:
        stale_statuses.add("source_not_latest")
    if any(source.status in stale_statuses for source in sources):
        return "stale"
    if any(source.status in WARNING_SOURCE_STATUSES for source in sources):
        return "warning"
    if not sources:
        return "no_sources"
    return "fresh"


def audit_run(
    root: Any,
    zarr_path: Path,
    *,
    run_family: str,
    run_id: str,
    run_path: str,
    run_group: Any,
    require_latest_sources: bool = False,
) -> RunAudit:
    explicit_stale_attrs = _explicit_stale_attrs(run_group)
    refs = discover_source_refs(root, run_group)
    sources = [
        audit_source_ref(
            root,
            ref,
            zarr_path=zarr_path,
            require_latest_sources=require_latest_sources,
        )
        for ref in refs
    ]
    status = _run_status(
        sources,
        explicit_stale_attrs=explicit_stale_attrs,
        require_latest_sources=require_latest_sources,
    )
    return RunAudit(
        zarr_path=str(zarr_path),
        run_family=run_family,
        run_id=run_id,
        run_path=run_path,
        status=status,
        sources=sources,
        explicit_stale_attrs=explicit_stale_attrs,
    )


def audit_zarr_analysis_staleness(
    zarr_path: Path,
    *,
    run_families: set[str] | None = None,
    require_latest_sources: bool = False,
) -> list[RunAudit]:
    root = open_zarr_root(zarr_path, mode="r")
    results: list[RunAudit] = []
    for spec in RUN_PARENT_SPECS:
        if run_families is not None and spec.family not in run_families:
            continue
        parent = _get_nested_group(root, spec.parent_path)
        if parent is None:
            continue
        for run_id in _group_names(parent):
            run_path = "/".join((*spec.parent_path, run_id))
            try:
                run_group = parent[run_id]
                results.append(
                    audit_run(
                        root,
                        zarr_path,
                        run_family=spec.family,
                        run_id=run_id,
                        run_path=run_path,
                        run_group=run_group,
                        require_latest_sources=require_latest_sources,
                    )
                )
            except Exception as exc:
                results.append(
                    RunAudit(
                        zarr_path=str(zarr_path),
                        run_family=spec.family,
                        run_id=run_id,
                        run_path=run_path,
                        status="error",
                        message=str(exc),
                    )
                )
    return results


def _json_ready(results: Sequence[RunAudit]) -> list[dict[str, Any]]:
    return [normalize_lineage_value(asdict(result)) for result in results]


def _status_counts(results: Sequence[RunAudit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _should_fail(results: Sequence[RunAudit], fail_on: str) -> bool:
    if fail_on == "never":
        return False
    if fail_on == "stale":
        return any(result.status in {"stale", "error"} for result in results)
    if fail_on == "warning":
        return any(result.status in {"stale", "warning", "error"} for result in results)
    if fail_on == "any_issue":
        return any(result.status != "fresh" for result in results)
    raise ValueError(f"unknown fail_on policy: {fail_on}")


def _print_text_report(results: Sequence[RunAudit]) -> None:
    by_archive: dict[str, list[RunAudit]] = {}
    for result in results:
        by_archive.setdefault(result.zarr_path, []).append(result)
    for archive, archive_results in by_archive.items():
        counts = _status_counts(archive_results)
        counts_text = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
        print(f"{archive}: {len(archive_results)} analysis runs audited ({counts_text})")
        for result in archive_results:
            if result.status == "fresh":
                continue
            print(f"  [{result.status}] {result.run_path}")
            if result.explicit_stale_attrs:
                print(f"    run stale attrs: {', '.join(result.explicit_stale_attrs)}")
            if result.message:
                print(f"    {result.message}")
            for source in result.sources:
                if source.status == "fresh":
                    continue
                print(f"    - {source.key}: {source.status}: {source.message}")
                print(f"      path: {source.path}")
                if source.expected_fingerprint or source.actual_fingerprint:
                    print(f"      expected: {source.expected_fingerprint or '<missing>'}")
                    print(f"      actual:   {source.actual_fingerprint or '<missing>'}")
                if source.latest_run_id and source.referenced_run_id:
                    print(
                        "      latest: "
                        f"{source.latest_run_id} at {source.latest_parent_path or '<root>'}; "
                        f"referenced: {source.referenced_run_id}"
                    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit for stale Palette derived-analysis source lineage.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr paths or directories containing Zarrs")
    parser.add_argument("--recursive", action="store_true", help="recursively discover *.zarr under directories")
    parser.add_argument(
        "--run-family",
        action="append",
        choices=sorted({spec.family for spec in RUN_PARENT_SPECS}),
        help="limit to one run family; may be repeated",
    )
    parser.add_argument(
        "--require-latest-sources",
        action="store_true",
        help="treat source refs that do not point at their parent latest run as stale",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument(
        "--fail-on",
        choices=("stale", "warning", "any_issue", "never"),
        default="stale",
        help="exit nonzero for stale/error runs, warnings, any non-fresh run, or never",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_families = set(args.run_family) if args.run_family else None
    all_results: list[RunAudit] = []
    for zarr_path in _iter_zarr_paths(args.paths, recursive=bool(args.recursive)):
        try:
            all_results.extend(
                audit_zarr_analysis_staleness(
                    zarr_path,
                    run_families=run_families,
                    require_latest_sources=bool(args.require_latest_sources),
                )
            )
        except Exception as exc:
            all_results.append(
                RunAudit(
                    zarr_path=str(zarr_path),
                    run_family="",
                    run_id="",
                    run_path="",
                    status="error",
                    message=str(exc),
                )
            )
    if args.json:
        print(json.dumps(_json_ready(all_results), indent=2, allow_nan=False, sort_keys=True))
    else:
        _print_text_report(all_results)
    return 1 if _should_fail(all_results, str(args.fail_on)) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
