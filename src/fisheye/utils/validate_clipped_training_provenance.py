"""Validate and repair clipped training Zarr provenance metadata.

The checks in this module focus on provenance surfaces that make clipped
training archives trustworthy as future training sources:

* root-level clipped-training identity and source-frame sidecar pointers;
* sampled ``source_frame_index.parquet`` shape and row alignment;
* PyNvVideoCodec luma crop-run pixel-contract metadata;
* downstream keypoint/mask run references to those luma crop runs.

Repair mode is intentionally metadata-only. It updates stale attrs whose
correct value can be derived from already-present top-level run attrs.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import zarr

from fisheye.shared.roi_pixel_contract import (
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    ROI_IMAGE_REPRESENTATION,
    normalize_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.utils.create_clipped_training_zarr import (
    SOURCE_FRAME_INDEX_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,
)


MODULE_NAME = "fisheye.utils.validate_clipped_training_provenance"
REGEN_CROP_MODULE = "fisheye.utils.regenerate_training_crops_pynvvc"

REQUIRED_SOURCE_FRAME_INDEX_COLUMNS = {
    "sample_index",
    "session_id",
    "recording_id",
    "camera_serial",
    "parent_frame_index",
    "recording_frame_id",
    "clip_index",
    "clip_id",
    "clip_local_frame_index",
    "timestamp",
    "timestamp_sys",
    "video_path",
    "metadata_path",
    "keyframe_path",
    "clip_manifest_path",
    "source_recording_frame_index_path",
}

RUN_GROUPS_WITH_SOURCE_CROP = (
    "keypoints_runs",
    "refined_keypoints_runs",
    "eye_masks_runs",
    "refined_eye_masks_runs",
    "subject_mask_runs",
    "refined_subject_masks_runs",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _contract_name(value: Any) -> str | None:
    contract = normalize_pixel_contract(value)
    if contract is None:
        return None
    name = contract.get("name")
    return str(name) if name is not None else None


def _relative_path(base: Path, maybe_relative: Any) -> Path | None:
    text = str(maybe_relative or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    path: str | None = None
    repairable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "repairable": self.repairable,
        }


@dataclass(frozen=True)
class PlannedRepair:
    target_path: str
    attr_path: tuple[str, ...]
    old_value: Any
    new_value: Any
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_path": self.target_path,
            "attr_path": list(self.attr_path),
            "old_value": _json_safe(self.old_value),
            "new_value": _json_safe(self.new_value),
            "reason": self.reason,
        }


class _Collector:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.repairs: list[PlannedRepair] = []

    def fail(self, code: str, message: str, *, path: str | None = None) -> None:
        self.findings.append(Finding("error", code, message, path=path))

    def warn(self, code: str, message: str, *, path: str | None = None) -> None:
        self.findings.append(Finding("warning", code, message, path=path))

    def repair(
        self,
        *,
        target_path: str,
        attr_path: tuple[str, ...],
        old_value: Any,
        new_value: Any,
        reason: str,
        code: str,
    ) -> None:
        self.findings.append(
            Finding(
                "warning",
                code,
                f"{target_path}:{'.'.join(attr_path)} is stale and can be repaired.",
                path=target_path,
                repairable=True,
            )
        )
        self.repairs.append(
            PlannedRepair(
                target_path=target_path,
                attr_path=attr_path,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
            )
        )


def _get_group(root: zarr.Group, path: str) -> zarr.Group:
    if not path:
        return root
    current: Any = root
    for part in path.split("/"):
        current = current[part]
    return current


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _nested_set(mapping: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    current: MutableMapping[str, Any] = mapping
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def _apply_repair(root: zarr.Group, repair: PlannedRepair) -> bool:
    group = _get_group(root, repair.target_path)
    attr_path = repair.attr_path
    if not attr_path:
        return False
    if len(attr_path) == 1:
        current = group.attrs.get(attr_path[0])
        if current == repair.new_value:
            return False
        group.attrs[attr_path[0]] = repair.new_value
        return True

    top_key = attr_path[0]
    payload = deepcopy(group.attrs.get(top_key) or {})
    if not isinstance(payload, MutableMapping):
        payload = {}
    current = _nested_get(payload, attr_path[1:])
    if current == repair.new_value:
        return False
    _nested_set(payload, attr_path[1:], repair.new_value)
    group.attrs[top_key] = payload
    return True


def _check_root_attrs(root: zarr.Group, zarr_path: Path, collector: _Collector) -> Path | None:
    attrs = root.attrs
    if attrs.get("zarr_purpose") != "training":
        collector.fail(
            "root_not_training",
            f"Expected root zarr_purpose='training', got {attrs.get('zarr_purpose')!r}.",
            path=".",
        )
    if attrs.get("training_schema_version") != TRAINING_SCHEMA_VERSION:
        collector.fail(
            "root_not_clipped_training_schema",
            "Expected clipped training schema "
            f"{TRAINING_SCHEMA_VERSION!r}, got {attrs.get('training_schema_version')!r}.",
            path=".",
        )
    if attrs.get("source_layout") != "rolling_clips":
        collector.fail(
            "root_not_rolling_clips",
            f"Expected source_layout='rolling_clips', got {attrs.get('source_layout')!r}.",
            path=".",
        )
    if attrs.get("source_frame_index_schema") != SOURCE_FRAME_INDEX_SCHEMA_VERSION:
        collector.fail(
            "bad_source_frame_index_schema",
            "Expected source_frame_index_schema "
            f"{SOURCE_FRAME_INDEX_SCHEMA_VERSION!r}, got {attrs.get('source_frame_index_schema')!r}.",
            path=".",
        )

    source_index_path = _relative_path(zarr_path, attrs.get("source_frame_index_path"))
    if source_index_path is None:
        collector.fail("missing_source_frame_index_path", "Missing root source_frame_index_path.", path=".")
    elif not source_index_path.exists():
        collector.fail(
            "source_frame_index_missing",
            f"source_frame_index_path does not exist: {source_index_path}",
            path=".",
        )

    recording_index_path = _relative_path(Path("/"), attrs.get("source_recording_frame_index_path"))
    if recording_index_path is not None and not recording_index_path.exists():
        collector.warn(
            "source_recording_frame_index_missing",
            f"source_recording_frame_index_path does not exist locally: {recording_index_path}",
            path=".",
        )
    return source_index_path


def _check_source_frame_index(
    root: zarr.Group,
    zarr_path: Path,
    source_index_path: Path | None,
    collector: _Collector,
    *,
    check_source_paths: bool,
    max_source_path_checks: int,
) -> dict[str, Any]:
    if source_index_path is None or not source_index_path.exists():
        return {"checked": False}

    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment dependency
        collector.fail("pyarrow_unavailable", f"pyarrow import failed: {exc}", path=str(source_index_path))
        return {"checked": False}

    try:
        table = pq.read_table(source_index_path).combine_chunks()
    except Exception as exc:
        collector.fail(
            "source_frame_index_unreadable",
            f"Could not read source_frame_index.parquet: {exc}",
            path=str(source_index_path),
        )
        return {"checked": False}

    columns = set(table.column_names)
    missing = sorted(REQUIRED_SOURCE_FRAME_INDEX_COLUMNS - columns)
    if missing:
        collector.fail(
            "source_frame_index_missing_columns",
            f"source_frame_index.parquet missing required columns: {missing}",
            path=str(source_index_path),
        )

    row_count = int(table.num_rows)
    raw = root.get("raw_video")
    if raw is None:
        collector.fail("missing_raw_video_group", "Missing raw_video group.", path="raw_video")
    else:
        if "images_full" in raw and int(raw["images_full"].shape[0]) != row_count:
            collector.fail(
                "images_full_row_count_mismatch",
                f"raw_video/images_full rows {raw['images_full'].shape[0]} != source_frame_index rows {row_count}.",
                path="raw_video/images_full",
            )
        if "original_frame_indices" not in raw:
            collector.fail(
                "missing_original_frame_indices",
                "Missing raw_video/original_frame_indices.",
                path="raw_video/original_frame_indices",
            )
        else:
            original = np.asarray(raw["original_frame_indices"][:], dtype=np.int64)
            if int(original.shape[0]) != row_count:
                collector.fail(
                    "original_frame_indices_row_count_mismatch",
                    f"raw_video/original_frame_indices rows {original.shape[0]} != source_frame_index rows {row_count}.",
                    path="raw_video/original_frame_indices",
                )
            elif "parent_frame_index" in columns:
                parent = np.asarray(table["parent_frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
                if not np.array_equal(original, parent):
                    collector.fail(
                        "original_frame_indices_parent_mismatch",
                        "raw_video/original_frame_indices must equal source_frame_index.parent_frame_index.",
                        path="raw_video/original_frame_indices",
                    )

    if "sample_index" in columns:
        sample_index = np.asarray(table["sample_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
        expected = np.arange(row_count, dtype=np.int64)
        if not np.array_equal(sample_index, expected):
            collector.fail(
                "sample_index_not_dense",
                "source_frame_index.sample_index must be dense zero-based row order.",
                path=str(source_index_path),
            )

    source_path_checks = {"enabled": bool(check_source_paths), "checked": 0, "missing": 0}
    if check_source_paths and "video_path" in columns:
        video_paths = pc.unique(table["video_path"]).to_pylist()
        for video_path in video_paths[: max(0, int(max_source_path_checks))]:
            source_path_checks["checked"] += 1
            if not Path(str(video_path)).expanduser().exists():
                source_path_checks["missing"] += 1
                collector.warn(
                    "source_video_path_missing",
                    f"source_frame_index video_path does not exist locally: {video_path}",
                    path=str(source_index_path),
                )

    return {
        "checked": True,
        "path": str(source_index_path),
        "rows": row_count,
        "columns": table.column_names,
        "source_path_checks": source_path_checks,
    }


def _is_luma_crop_run(crop: zarr.Group) -> bool:
    attrs = crop.attrs
    return (
        attrs.get("generated_by") == REGEN_CROP_MODULE
        or attrs.get("decode_backend") == "pynvvc_luma"
        or _contract_name(attrs.get("roi_pixel_contract")) == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
    )


def _plan_attr_contract_repair(
    collector: _Collector,
    *,
    group: zarr.Group,
    target_path: str,
    attr_path: tuple[str, ...],
    current: Any,
    reason: str,
    code: str,
) -> None:
    expected = orange_mono_pynvvc_luma_pixel_contract()
    if _contract_name(current) == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
        return
    collector.repair(
        target_path=target_path,
        attr_path=attr_path,
        old_value=current,
        new_value=expected,
        reason=reason,
        code=code,
    )


def _check_luma_crop_run(crop: zarr.Group, crop_name: str, collector: _Collector) -> None:
    target_path = f"crop_runs/{crop_name}"
    attrs = crop.attrs
    if attrs.get("crop_storage_mode") != "materialized":
        collector.fail(
            "luma_crop_not_materialized",
            f"{target_path} should be materialized, got {attrs.get('crop_storage_mode')!r}.",
            path=target_path,
        )
    if attrs.get("roi_image_representation") != ROI_IMAGE_REPRESENTATION:
        collector.fail(
            "luma_crop_bad_image_representation",
            f"{target_path} has roi_image_representation={attrs.get('roi_image_representation')!r}.",
            path=target_path,
        )
    if _contract_name(attrs.get("roi_pixel_contract")) != ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
        collector.fail(
            "luma_crop_bad_top_level_contract",
            f"{target_path} top-level roi_pixel_contract is not {ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME}.",
            path=target_path,
        )
    summary = attrs.get("summary_statistics")
    if isinstance(summary, Mapping):
        if summary.get("pixel_contract_name") != ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
            collector.warn(
                "luma_crop_summary_contract_mismatch",
                f"{target_path} summary_statistics.pixel_contract_name is stale.",
                path=target_path,
            )

    provenance = attrs.get("provenance")
    if isinstance(provenance, Mapping):
        params = provenance.get("parameters")
        if isinstance(params, Mapping):
            _plan_attr_contract_repair(
                collector,
                group=crop,
                target_path=target_path,
                attr_path=("provenance", "parameters", "roi_pixel_contract"),
                current=params.get("roi_pixel_contract"),
                reason="PyNvVideoCodec luma crop run has stale nested provenance.parameters.roi_pixel_contract.",
                code="repair_luma_crop_nested_parameter_contract",
            )
            if params.get("roi_image_representation") not in {None, ROI_IMAGE_REPRESENTATION}:
                collector.repair(
                    target_path=target_path,
                    attr_path=("provenance", "parameters", "roi_image_representation"),
                    old_value=params.get("roi_image_representation"),
                    new_value=ROI_IMAGE_REPRESENTATION,
                    reason="PyNvVideoCodec luma crop run has stale nested provenance.parameters.roi_image_representation.",
                    code="repair_luma_crop_nested_parameter_representation",
                )

    if "source_frame_indices" not in crop:
        collector.fail("missing_crop_source_frame_indices", "Luma crop run is missing source_frame_indices.", path=target_path)
    if attrs.get("source_layout") == "rolling_clips":
        for name in ("source_clip_indices", "source_clip_local_frame_indices"):
            if name not in crop:
                collector.fail(
                    f"missing_crop_{name}",
                    f"Luma crop run sourced from rolling clips is missing {name}.",
                    path=f"{target_path}/{name}",
                )


def _check_crop_runs(root: zarr.Group, collector: _Collector) -> set[str]:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        collector.fail("missing_crop_runs", "Missing crop_runs group.", path="crop_runs")
        return set()
    luma_crop_runs: set[str] = set()
    for crop_name in sorted(str(name) for name in crop_parent.group_keys()):
        crop = crop_parent[crop_name]
        if not _is_luma_crop_run(crop):
            continue
        luma_crop_runs.add(crop_name)
        _check_luma_crop_run(crop, crop_name, collector)
    if not luma_crop_runs:
        collector.warn("no_luma_crop_runs", "No PyNvVideoCodec luma crop runs found.", path="crop_runs")
    return luma_crop_runs


def _check_downstream_run(
    run: zarr.Group,
    *,
    run_path: str,
    luma_crop_runs: set[str],
    collector: _Collector,
) -> None:
    attrs = run.attrs
    source_crop_run = attrs.get("source_crop_run")
    if source_crop_run not in luma_crop_runs:
        return

    if attrs.get("source_roi_image_representation") != ROI_IMAGE_REPRESENTATION:
        collector.fail(
            "downstream_bad_source_roi_representation",
            f"{run_path} source_roi_image_representation is not {ROI_IMAGE_REPRESENTATION}.",
            path=run_path,
        )
    if attrs.get("source_roi_pixel_contract_name") != ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
        collector.fail(
            "downstream_bad_source_roi_contract_name",
            f"{run_path} source_roi_pixel_contract_name is not {ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME}.",
            path=run_path,
        )
    if _contract_name(attrs.get("source_roi_pixel_contract")) != ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
        collector.fail(
            "downstream_bad_source_roi_contract",
            f"{run_path} source_roi_pixel_contract is not {ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME}.",
            path=run_path,
        )

    provenance = attrs.get("provenance")
    if isinstance(provenance, Mapping):
        inputs = provenance.get("inputs")
        if isinstance(inputs, Mapping):
            if _contract_name(inputs.get("source_roi_pixel_contract")) != ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
                collector.repair(
                    target_path=run_path,
                    attr_path=("provenance", "inputs", "source_roi_pixel_contract"),
                    old_value=inputs.get("source_roi_pixel_contract"),
                    new_value=orange_mono_pynvvc_luma_pixel_contract(),
                    reason="Downstream run references a luma crop but nested provenance input contract is stale.",
                    code="repair_downstream_nested_source_roi_contract",
                )
            if inputs.get("source_roi_image_representation") not in {None, ROI_IMAGE_REPRESENTATION}:
                collector.repair(
                    target_path=run_path,
                    attr_path=("provenance", "inputs", "source_roi_image_representation"),
                    old_value=inputs.get("source_roi_image_representation"),
                    new_value=ROI_IMAGE_REPRESENTATION,
                    reason="Downstream run references a luma crop but nested provenance input representation is stale.",
                    code="repair_downstream_nested_source_roi_representation",
                )


def _check_downstream_runs(root: zarr.Group, luma_crop_runs: set[str], collector: _Collector) -> int:
    checked = 0
    for family in RUN_GROUPS_WITH_SOURCE_CROP:
        parent = root.get(family)
        if parent is None:
            continue
        for run_name in sorted(str(name) for name in parent.group_keys()):
            checked += 1
            _check_downstream_run(
                parent[run_name],
                run_path=f"{family}/{run_name}",
                luma_crop_runs=luma_crop_runs,
                collector=collector,
            )
    return checked


def validate_clipped_training_provenance(
    zarr_path: str | Path,
    *,
    apply: bool = False,
    check_source_paths: bool = False,
    max_source_path_checks: int = 50,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    mode = "a" if apply else "r"
    root = zarr.open_group(str(archive_path), mode=mode, use_consolidated=False)

    collector = _Collector()
    source_index_path = _check_root_attrs(root, archive_path, collector)
    source_index_summary = _check_source_frame_index(
        root,
        archive_path,
        source_index_path,
        collector,
        check_source_paths=check_source_paths,
        max_source_path_checks=max_source_path_checks,
    )
    luma_crop_runs = _check_crop_runs(root, collector)
    downstream_runs_checked = _check_downstream_runs(root, luma_crop_runs, collector)

    applied_repairs = 0
    if apply:
        for repair in collector.repairs:
            if _apply_repair(root, repair):
                applied_repairs += 1
        if applied_repairs:
            root.attrs["clipped_training_provenance_repair"] = {
                "repaired_at_utc": _utc_now(),
                "repaired_by": MODULE_NAME,
                "applied_repairs": int(applied_repairs),
            }

    error_count = sum(1 for finding in collector.findings if finding.severity == "error")
    repair_count = len(collector.repairs)
    if error_count:
        status = "failed"
    elif repair_count and not apply:
        status = "needs_repair"
    else:
        status = "ok"

    return {
        "schema_version": "palette.clipped_training_provenance_validation.v1",
        "status": status,
        "zarr_path": str(archive_path),
        "apply": bool(apply),
        "error_count": int(error_count),
        "warning_count": int(sum(1 for finding in collector.findings if finding.severity == "warning")),
        "planned_repairs": int(repair_count),
        "applied_repairs": int(applied_repairs),
        "luma_crop_runs": sorted(luma_crop_runs),
        "downstream_runs_checked": int(downstream_runs_checked),
        "source_frame_index": source_index_summary,
        "findings": [finding.as_dict() for finding in collector.findings],
        "repairs": [repair.as_dict() for repair in collector.repairs],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--apply", action="store_true", help="Write metadata-only provenance repairs.")
    parser.add_argument(
        "--check-source-paths",
        action="store_true",
        help="Check that sampled source video paths still exist locally.",
    )
    parser.add_argument(
        "--max-source-path-checks",
        type=int,
        default=50,
        help="Maximum distinct source video paths to check when --check-source-paths is set.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"status: {report['status']}")
    print(f"zarr_path: {report['zarr_path']}")
    print(f"errors: {report['error_count']}")
    print(f"warnings: {report['warning_count']}")
    print(f"planned_repairs: {report['planned_repairs']}")
    print(f"applied_repairs: {report['applied_repairs']}")
    print(f"luma_crop_runs: {', '.join(report['luma_crop_runs']) or 'none'}")
    for finding in report["findings"]:
        print(
            f"- {finding['severity']} {finding['code']}: {finding['message']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_clipped_training_provenance(
        args.zarr_path,
        apply=bool(args.apply),
        check_source_paths=bool(args.check_source_paths),
        max_source_path_checks=int(args.max_source_path_checks),
    )
    text = json.dumps(_json_safe(report), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        _print_summary(report)
    return 0 if report["status"] in {"ok", "needs_repair"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
