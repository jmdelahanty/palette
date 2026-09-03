"""Build and atomically publish paradigm-neutral recording distributions."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from fisheye.analysis_workflows.recording_behavior_distribution_publication import (
    build_recording_behavior_distribution_publication_plan,
    materialize_recording_behavior_distribution_locally,
    publish_recording_behavior_distribution_candidate,
)
from fisheye.analysis_workflows.recording_behavior_distribution_workflow import (
    prepare_recording_behavior_distribution,
)
from fisheye.analysis_workflows.recording_distribution_scope_adapters import (
    NamedSessionTimeBracket,
    named_session_time_distribution_scopes,
    protocol_semantic_distribution_scopes,
)
from fisheye.analysis_workflows.recording_distribution_timebase_adapter import (
    load_bundle_recording_session_timebase,
)
from fisheye.analysis_workflows.validated_recording_behavior_source import (
    ValidatedRecordingBehaviorSource,
)
from fisheye.group_statistics.recording_behavior_distribution_specs import (
    recording_distribution_metric_specs_for_families,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    whole_session_scope,
)
from fisheye.shared.json_safety import json_attr_safe


SCOPE_MODES = ("whole_session", "protocol_semantic", "named_session_time")


class RecordingBehaviorDistributionMaterializerError(ValueError):
    """A requested recording-distribution materialization is invalid."""


def _load_time_brackets(path: Path) -> tuple[NamedSessionTimeBracket, ...]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingBehaviorDistributionMaterializerError(
            f"Unable to read named time brackets: {exc}"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise RecordingBehaviorDistributionMaterializerError(
            "Named time-bracket JSON must be one nonempty list."
        )
    result = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or set(raw) != {
            "scope_id",
            "scope_label",
            "start_timestamp_ns_session",
            "end_timestamp_ns_session_exclusive",
        }:
            raise RecordingBehaviorDistributionMaterializerError(
                f"Time bracket {index} has an inexact field set."
            )
        if (
            type(raw["scope_id"]) is not str
            or type(raw["scope_label"]) is not str
            or type(raw["start_timestamp_ns_session"]) is not int
            or type(raw["end_timestamp_ns_session_exclusive"]) is not int
        ):
            raise RecordingBehaviorDistributionMaterializerError(
                f"Time bracket {index} has inexact JSON field types."
            )
        try:
            result.append(
                NamedSessionTimeBracket(
                    scope_id=raw["scope_id"],
                    scope_label=raw["scope_label"],
                    start_timestamp_ns_session=raw[
                        "start_timestamp_ns_session"
                    ],
                    end_timestamp_ns_session_exclusive=raw[
                        "end_timestamp_ns_session_exclusive"
                    ],
                )
            )
        except (TypeError, ValueError) as exc:
            raise RecordingBehaviorDistributionMaterializerError(
                f"Time bracket {index} is invalid: {exc}"
            ) from exc
    return tuple(result)


def _scope_request(
    source: ValidatedRecordingBehaviorSource,
    *,
    scope_mode: str,
    time_brackets_json: str | Path | None,
) -> tuple[Any, Any]:
    if scope_mode == "whole_session":
        if time_brackets_json is not None:
            raise RecordingBehaviorDistributionMaterializerError(
                "Whole-session mode cannot consume a time-bracket file."
            )
        return (whole_session_scope(),), None
    if scope_mode == "protocol_semantic":
        if time_brackets_json is not None:
            raise RecordingBehaviorDistributionMaterializerError(
                "Protocol-semantic mode cannot consume a time-bracket file."
            )
        return protocol_semantic_distribution_scopes(source), None
    if scope_mode == "named_session_time":
        if time_brackets_json is None:
            raise RecordingBehaviorDistributionMaterializerError(
                "Named-session-time mode requires --time-brackets-json."
            )
        timebase = load_bundle_recording_session_timebase(source)
        scopes = named_session_time_distribution_scopes(
            _load_time_brackets(Path(time_brackets_json)),
            timebase_binding=timebase.binding,
        )
        return scopes, timebase
    raise RecordingBehaviorDistributionMaterializerError(
        f"Unsupported scope mode {scope_mode!r}."
    )


def materialize_recording_behavior_distributions(
    bundle_path: str | Path,
    *,
    run_name: str,
    scratch_root: str | Path,
    scope_mode: str = "protocol_semantic",
    time_brackets_json: str | Path | None = None,
    metric_families: Sequence[str] = (),
    include_chaser_distance: bool = True,
    chaser_provider_roles: Sequence[str] = ("keypoint", "detection"),
    require_all_metrics: bool = False,
    copy_backend: str = "python",
    apply: bool = False,
    keep_scratch: bool = False,
) -> Mapping[str, Any]:
    """Prepare one exact component and optionally publish it atomically."""

    for field, value in (
        ("include_chaser_distance", include_chaser_distance),
        ("require_all_metrics", require_all_metrics),
        ("apply", apply),
        ("keep_scratch", keep_scratch),
    ):
        if type(value) is not bool:
            raise RecordingBehaviorDistributionMaterializerError(
                f"{field} must be the exact boolean."
            )
    source = ValidatedRecordingBehaviorSource(bundle_path)
    scopes, timebase = _scope_request(
        source,
        scope_mode=scope_mode,
        time_brackets_json=time_brackets_json,
    )
    specs = recording_distribution_metric_specs_for_families(metric_families)
    if not include_chaser_distance:
        specs = tuple(
            spec for spec in specs if spec.source_surface != "chaser_relative_samples"
        )
    prepared = prepare_recording_behavior_distribution(
        source,
        distribution_run_id=run_name,
        scopes=scopes,
        metric_specs=specs,
        session_timebase=timebase,
        chaser_provider_roles=chaser_provider_roles,
        require_all_metrics=require_all_metrics,
    )
    scratch = Path(scratch_root).expanduser().resolve()
    if apply:
        scratch.mkdir(parents=True, exist_ok=True)
    plan = build_recording_behavior_distribution_publication_plan(
        source.analysis_zarr,
        scratch_root=scratch,
        prepared=prepared,
    )
    payload: dict[str, Any] = {
        "status": "planned" if not apply else "running",
        "analysis_zarr": str(source.analysis_zarr),
        "recording_id": source.recording_id,
        "bundle_path": str(source.bundle_path),
        "bundle_record_sha256": source.bundle_sha256,
        "run_name": run_name,
        "run_path": plan.run_path,
        "scope_mode": scope_mode,
        "scope_ids": [scope.scope_id for scope in scopes],
        "metric_ids": [row["metric_id"] for row in prepared.result.metric_registry],
        "omitted_metrics": [dict(row) for row in prepared.omitted_metrics],
        "support_row_count": len(prepared.result.support),
        "sparse_bin_row_count": len(prepared.result.sparse_bins),
        "result_record_sha256": plan.result_record_sha256,
        "selector_attrs_before": dict(plan.selector_attrs_before),
        "scratch_root": str(scratch),
        "local_zarr": str(plan.local_zarr),
        "apply": apply,
    }
    if not apply:
        return json_attr_safe(payload)
    succeeded = False
    try:
        local = materialize_recording_behavior_distribution_locally(
            plan, prepared=prepared
        )
        publication = publish_recording_behavior_distribution_candidate(
            plan, copy_backend=copy_backend
        )
        payload.update(
            status="complete",
            local_validation=dict(local),
            publication=dict(publication),
        )
        succeeded = True
        return json_attr_safe(payload)
    finally:
        if succeeded and not keep_scratch and plan.local_zarr.exists():
            shutil.rmtree(plan.local_zarr)


def _default_scratch() -> Path:
    user = os.environ.get("USER") or getpass.getuser() or "unknown"
    job = os.environ.get("LSB_JOBID") or "manual"
    root = Path("/scratch") / user
    if root.is_dir() and os.access(root, os.W_OK | os.X_OK):
        return root / job / "palette_recording_behavior_distributions"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_recording_behavior_distributions_{job}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--scope-mode", choices=SCOPE_MODES, default="protocol_semantic")
    parser.add_argument("--time-brackets-json", type=Path)
    parser.add_argument("--metric-family", action="append", default=[])
    parser.add_argument(
        "--chaser-provider",
        action="append",
        choices=("keypoint", "detection"),
        dest="chaser_providers",
    )
    parser.add_argument("--exclude-chaser-distance", action="store_true")
    parser.add_argument("--require-all-metrics", action="store_true")
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = materialize_recording_behavior_distributions(
        args.bundle_path,
        run_name=args.run_name,
        scratch_root=args.scratch_root or _default_scratch(),
        scope_mode=args.scope_mode,
        time_brackets_json=args.time_brackets_json,
        metric_families=args.metric_family,
        include_chaser_distance=not args.exclude_chaser_distance,
        chaser_provider_roles=args.chaser_providers or ("keypoint", "detection"),
        require_all_metrics=args.require_all_metrics,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
    )
    print(json.dumps(payload, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RecordingBehaviorDistributionMaterializerError",
    "SCOPE_MODES",
    "materialize_recording_behavior_distributions",
]
