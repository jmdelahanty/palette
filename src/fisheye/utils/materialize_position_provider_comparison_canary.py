"""Materialize and compare explicitly named selector-ineligible providers.

The task is retryable because every output name is frozen.  Existing outputs
are reused only after strict validation.  No ``latest`` selector is resolved or
updated, and no estimator is promoted by this utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.analysis_workflows.materializers.provider_position_comparison import (
    plan_provider_position_comparison_run,
    publish_provider_position_comparison_run,
)
from fisheye.analysis_workflows.materializers.subject_position import (
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    load_subject_position_source_handle,
)
from fisheye.shared.anatomy_profile import load_anatomy_profile
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.subject_position_detection_source import (
    load_persisted_selector_ineligible_detection_position_source,
)
from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
)
from fisheye.shared.subject_position_keypoint_source import (
    KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY,
    KeypointPositionSourcePolicy,
    load_bound_keypoint_position_source,
)
from fisheye.shared.subject_position_mask_source import (
    REFINED_SUBJECT_MASK_SOURCE_KIND,
    SUBJECT_MASK_POSITION_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY,
    load_subject_mask_position_source_for_estimator,
)
from fisheye.shared.subject_position_preparation import (
    prepare_subject_position_input,
)


TASK_SCHEMA_ID = "palette.position_provider_comparison_canary_task"
TASK_SCHEMA_VERSION = 1
RESULT_SCHEMA_ID = "palette.position_provider_comparison_canary_result"
RESULT_SCHEMA_VERSION = 1

DETECTION_SOURCE_KIND = "selector_ineligible_canonical_detection_v3"
KEYPOINT_SOURCE_KIND = "coordinate_successor_keypoint"
REFINED_MASK_SOURCE_KIND = "coordinate_successor_refined_subject_mask"
_ESTIMATORS_BY_SOURCE_KIND = {
    DETECTION_SOURCE_KIND: {DETECTION_BBOX_CENTROID_ESTIMATOR_ID},
    KEYPOINT_SOURCE_KIND: {KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID},
    REFINED_MASK_SOURCE_KIND: {
        MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
        SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
    },
}


class PositionProviderCanaryError(ValueError):
    """Raised when an explicit canary task is incomplete or inconsistent."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PositionProviderCanaryError(f"{label} must be one JSON object.")
    return dict(value)


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PositionProviderCanaryError(f"{label} must be a nonempty exact string.")
    return value


def _run_name(value: Any, *, label: str) -> str:
    result = _text(value, label=label)
    if "/" in result or "\\" in result or result in {".", ".."}:
        raise PositionProviderCanaryError(f"{label} must be one bare run name.")
    return result


def load_task(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    task = _mapping(json.loads(source.read_text(encoding="utf-8")), label="task")
    if (
        task.get("schema_id") != TASK_SCHEMA_ID
        or task.get("schema_version") != TASK_SCHEMA_VERSION
    ):
        raise PositionProviderCanaryError("Task schema identity is unsupported.")
    recording_id = _text(task.get("recording_id"), label="recording_id")
    archive = Path(_text(task.get("analysis_zarr"), label="analysis_zarr")).resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    if archive.name != f"{recording_id}_analysis.zarr":
        raise PositionProviderCanaryError(
            "Analysis Zarr basename differs from recording_id."
        )
    task["recording_id"] = recording_id
    task["analysis_zarr"] = str(archive)
    profile = Path(
        _text(task.get("anatomy_profile"), label="anatomy_profile")
    ).resolve()
    if not profile.is_file():
        raise FileNotFoundError(f"Anatomy profile does not exist: {profile}")
    task["anatomy_profile"] = str(profile)
    task["software"] = _mapping(task.get("software"), label="software")
    if not task["software"]:
        raise PositionProviderCanaryError("software must not be empty.")
    raw_providers = task.get("providers")
    if not isinstance(raw_providers, list) or len(raw_providers) < 2:
        raise PositionProviderCanaryError(
            "providers must contain at least two entries."
        )
    providers: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_providers):
        provider = _mapping(raw, label=f"providers[{index}]")
        provider["provider_id"] = _text(
            provider.get("provider_id"), label=f"providers[{index}].provider_id"
        )
        provider["estimator_id"] = _text(
            provider.get("estimator_id"), label=f"providers[{index}].estimator_id"
        )
        provider["position_run_name"] = _run_name(
            provider.get("position_run_name"),
            label=f"providers[{index}].position_run_name",
        )
        source_record = _mapping(
            provider.get("source"), label=f"providers[{index}].source"
        )
        source_kind = _text(
            source_record.get("kind"), label=f"providers[{index}].source.kind"
        )
        if source_kind not in _ESTIMATORS_BY_SOURCE_KIND:
            raise PositionProviderCanaryError(
                f"providers[{index}].source.kind is unsupported."
            )
        if provider["estimator_id"] not in _ESTIMATORS_BY_SOURCE_KIND[source_kind]:
            raise PositionProviderCanaryError(
                f"providers[{index}] estimator is incompatible with its source kind."
            )
        source_record["kind"] = source_kind
        source_record["run_path"] = _text(
            source_record.get("run_path"),
            label=f"providers[{index}].source.run_path",
        )
        if source_kind in {KEYPOINT_SOURCE_KIND, REFINED_MASK_SOURCE_KIND}:
            source_record["binding_id"] = _text(
                source_record.get("binding_id"),
                label=f"providers[{index}].source.binding_id",
            )
        provider["source"] = source_record
        providers.append(provider)
    ids = [item["provider_id"] for item in providers]
    outputs = [item["position_run_name"] for item in providers]
    if len(set(ids)) != len(ids) or len(set(outputs)) != len(outputs):
        raise PositionProviderCanaryError(
            "provider_id and position_run_name values must each be unique."
        )
    task["providers"] = providers
    task["comparison_run_name"] = _run_name(
        task.get("comparison_run_name"), label="comparison_run_name"
    )
    return task


def _load_source(task: Mapping[str, Any], provider: Mapping[str, Any]) -> Any:
    archive = Path(str(task["analysis_zarr"]))
    source = provider["source"]
    kind = source["kind"]
    if kind == DETECTION_SOURCE_KIND:
        return load_persisted_selector_ineligible_detection_position_source(
            archive, source["run_path"]
        )
    profile = load_anatomy_profile(task["anatomy_profile"])
    if kind == KEYPOINT_SOURCE_KIND:
        return load_bound_keypoint_position_source(
            archive,
            run_path=source["run_path"],
            policy=KeypointPositionSourcePolicy(
                anatomy_profile=profile,
                binding_id=source["binding_id"],
                authority_mode=KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY,
            ),
        )
    if kind == REFINED_MASK_SOURCE_KIND:
        return load_subject_mask_position_source_for_estimator(
            archive,
            run_path=source["run_path"],
            source_kind=REFINED_SUBJECT_MASK_SOURCE_KIND,
            anatomy_profile=profile,
            binding_id=source["binding_id"],
            estimator_id=provider["estimator_id"],
            authority_mode=(
                SUBJECT_MASK_POSITION_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY
            ),
        )
    raise AssertionError("validated source kind was not dispatched")


def _materialize_position(
    task: Mapping[str, Any],
    provider: Mapping[str, Any],
    *,
    scratch_root: Path,
) -> tuple[Any, dict[str, Any]]:
    archive = Path(str(task["analysis_zarr"]))
    name = provider["position_run_name"]
    run_path = f"analysis/subject_position_runs/observation/{name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        handle = load_subject_position_source_handle(
            archive,
            run_path,
            expected_selector_eligible=False,
            use_consolidated=True,
        )
        if handle.estimator_record["estimator_id"] != provider["estimator_id"]:
            raise PositionProviderCanaryError(
                f"Existing position run {run_path!r} has another estimator."
            )
        return handle, {
            "status": "reused",
            "run_path": run_path,
            "manifest_sha256": handle.manifest_sha256,
        }
    bound = _load_source(task, provider)
    prepared = prepare_subject_position_input(
        bound,
        estimator_id=provider["estimator_id"],
        software_record=task["software"],
    )
    plan = plan_subject_position_run(
        archive,
        prepared,
        run_name=name,
        scratch_root=scratch_root / provider["provider_id"],
    )
    publication = publish_subject_position_run(plan, keep_scratch=False)
    handle = load_subject_position_source_handle(
        archive,
        run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    return handle, {
        "status": "published",
        "run_path": run_path,
        "manifest_sha256": handle.manifest_sha256,
        "publication": publication["publication"],
    }


def materialize_task(
    task: Mapping[str, Any],
    *,
    scratch_root: str | Path,
) -> dict[str, Any]:
    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    handles: list[tuple[str, Any]] = []
    position_results: dict[str, Any] = {}
    for provider in task["providers"]:
        handle, result = _materialize_position(
            task, provider, scratch_root=scratch / "positions"
        )
        handles.append((provider["provider_id"], handle))
        position_results[provider["provider_id"]] = result

    archive = Path(str(task["analysis_zarr"]))
    comparison_name = task["comparison_run_name"]
    comparison_path = f"analysis/provider_position_comparison_runs/{comparison_name}"
    comparison_target = archive.joinpath(*comparison_path.split("/"))
    if comparison_target.exists():
        raise FileExistsError(
            "Comparison target already exists; immutable retries require a new "
            f"comparison_run_name: {comparison_target}"
        )
    plan = plan_provider_position_comparison_run(
        archive,
        handles,
        run_name=comparison_name,
        scratch_root=scratch / "comparison",
        software_record=task["software"],
    )
    comparison = publish_provider_position_comparison_run(plan, keep_scratch=False)
    return json_attr_safe(
        {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "recording_id": task["recording_id"],
            "analysis_zarr": task["analysis_zarr"],
            "positions": position_results,
            "comparison": comparison,
            "selector_eligible": False,
            "selection": "none",
            "promotion": "not_performed",
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    task = load_task(args.task_json)
    if not args.apply:
        result = {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "ready",
            "recording_id": task["recording_id"],
            "analysis_zarr": task["analysis_zarr"],
            "provider_ids": [item["provider_id"] for item in task["providers"]],
            "comparison_run_name": task["comparison_run_name"],
            "writes": False,
        }
    else:
        result = materialize_task(task, scratch_root=args.scratch_root)
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
