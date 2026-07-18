"""CLI for dataset report discovery and planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.registry.db import Registry, RegistryPaths

from .catalog import PROVIDERS, VISUALIZATIONS
from .execution import execute_report_plan, execution_result_to_dict
from .export import MATERIALIZATION_POLICIES, export_report_bundle
from .manifest import report_plan_json, report_plan_to_dict
from .montage import build_semantic_visualization_montages
from .montage_report import publish_semantic_montage_report
from .planner import build_report_plan
from .report_registry import (
    check_report_manifest,
    index_report_manifest,
    query_indexed_reports,
    report_output_dir,
    resolve_analytics_export_binding,
    validate_report_id,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan composable Palette dataset reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan",
        help="Inspect applicable analyses and visualizations without writing anything.",
    )
    _add_selection_arguments(plan)
    plan.add_argument("--compact", action="store_true", help="Emit compact JSON.")

    apply = subparsers.add_parser(
        "apply",
        help="Explicitly run allowlisted missing-analysis or rendering actions.",
    )
    _add_selection_arguments(apply)
    apply.add_argument("--render-missing", action="store_true")
    apply.add_argument("--apply-analysis", action="store_true")
    apply.add_argument(
        "--refresh-contract-mismatches",
        action="store_true",
        help="Re-render historical artifacts whose contracts do not match.",
    )
    apply.add_argument(
        "--visualization-id",
        action="append",
        choices=sorted(VISUALIZATIONS),
        default=[],
    )
    apply.add_argument("--overwrite-analysis", action="store_true")
    apply.add_argument("--continue-on-error", action="store_true")
    apply.add_argument("--compact", action="store_true", help="Emit compact JSON.")

    montage = subparsers.add_parser(
        "montage",
        help="Compose contract-safe cohort montages by semantic visualization ID.",
    )
    _add_selection_arguments(montage)
    montage.add_argument(
        "--visualization-id",
        action="append",
        choices=sorted(VISUALIZATIONS),
        required=True,
    )
    montage.add_argument("--output-dir", type=Path, required=True)
    montage.add_argument("--columns", type=int, default=4)
    montage.add_argument("--tile-width", type=int, default=600)
    montage.add_argument("--max-image-height", type=int, default=480)
    montage.add_argument(
        "--allow-nonready",
        action="store_true",
        help="Render labeled placeholders for missing or non-contracted tiles.",
    )
    montage.add_argument("--overwrite", action="store_true")
    montage.add_argument("--compact", action="store_true", help="Emit compact JSON.")

    export = subparsers.add_parser(
        "export",
        help="Write an immutable reference or portable-copy report bundle.",
    )
    _add_selection_arguments(export)
    export.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Immutable report directory. Omit with --analytics-export-run-id to use "
            "the canonical reports/export_run_id=.../report_id=... layout."
        ),
    )
    export.add_argument("--analytics-export-run-id")
    export.add_argument("--report-id")
    export.add_argument(
        "--index-registry",
        action="store_true",
        help="Index the completed bound report in the registry.",
    )
    export.add_argument(
        "--materialization",
        choices=MATERIALIZATION_POLICIES,
        default="reference",
    )
    export.add_argument(
        "--visualization-id",
        action="append",
        choices=sorted(VISUALIZATIONS),
        default=[],
    )
    export.add_argument(
        "--allow-nonready",
        action="store_true",
        help="Record non-ready items instead of rejecting the export.",
    )
    export.add_argument("--source-collection-manifest", type=Path)
    export.add_argument("--compact", action="store_true", help="Emit compact JSON.")

    publish_montage = subparsers.add_parser(
        "publish-montage-report",
        help=(
            "Copy a completed semantic montage set into an immutable analytics "
            "report bundle."
        ),
    )
    publish_montage.add_argument("--registry", type=Path)
    publish_montage.add_argument("--semantic-manifest", type=Path, required=True)
    publish_montage.add_argument("--analytics-export-run-id", required=True)
    publish_montage.add_argument("--report-id", required=True)
    publish_montage.add_argument(
        "--index-registry",
        action="store_true",
        help="Index the completed montage report under its analytics export.",
    )
    publish_montage.add_argument(
        "--compact", action="store_true", help="Emit compact JSON."
    )

    index_report = subparsers.add_parser(
        "index-report",
        help="Verify and index an immutable report manifest.",
    )
    index_report.add_argument("--registry", type=Path)
    index_report.add_argument("--manifest", type=Path, required=True)
    index_report.add_argument("--status", default="active")
    index_report.add_argument("--compact", action="store_true")

    query_reports = subparsers.add_parser(
        "query-reports",
        help="Query indexed report bundles without opening recording Zarrs.",
    )
    query_reports.add_argument("--registry", type=Path)
    query_reports.add_argument("--export-run-id")
    query_reports.add_argument("--report-id")
    query_reports.add_argument("--visualization-id", choices=sorted(VISUALIZATIONS))
    query_reports.add_argument("--status", default="active")
    query_reports.add_argument("--latest", action="store_true")
    query_reports.add_argument("--limit", type=int, default=100)
    query_reports.add_argument(
        "--format", choices=("table", "json", "path"), default="table"
    )

    check_report = subparsers.add_parser(
        "check-report",
        help="Verify a report manifest, its analytics parent, and optional files.",
    )
    check_report.add_argument("--manifest", type=Path, required=True)
    check_report.add_argument("--check-files", action="store_true")
    check_report.add_argument("--compact", action="store_true")

    listing = subparsers.add_parser("list", help="List registered providers and plots.")
    listing.add_argument(
        "--kind",
        choices=("providers", "visualizations", "all"),
        default="all",
    )
    return parser


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--protocol-name")
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--recording-id-contains")
    parser.add_argument("--path-contains")
    parser.add_argument("--zarr-use", default="analysis")
    parser.add_argument("--status", default="active")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-recordings", action="store_true")
    parser.add_argument("--provider", action="append", choices=sorted(PROVIDERS), default=[])
    parser.add_argument("--include-not-applicable", action="store_true")


def _print_catalog(kind: str) -> None:
    if kind in {"providers", "all"}:
        for provider in PROVIDERS.values():
            modes = ",".join(provider.stimulus_modes) or "always"
            print(f"provider\t{provider.provider_id}\t{provider.label}\tmodes={modes}")
    if kind in {"visualizations", "all"}:
        for visualization in VISUALIZATIONS.values():
            print(
                "\t".join(
                    (
                        "visualization",
                        visualization.visualization_id,
                        visualization.provider_id,
                        visualization.analysis_family_id,
                        visualization.entity_scope.value,
                        visualization.visualization_contract_id or "uncontracted",
                    )
                )
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "list":
        _print_catalog(str(args.kind))
        return 0

    if args.command == "check-report":
        payload = check_report_manifest(
            args.manifest,
            check_files=bool(args.check_files),
        )
        print(
            json.dumps(
                payload,
                indent=None if args.compact else 2,
                sort_keys=True,
                separators=(",", ":") if args.compact else None,
            )
        )
        return 0 if payload["ok"] else 1

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    if args.command == "publish-montage-report":
        report_id = validate_report_id(str(args.report_id))
        binding = resolve_analytics_export_binding(
            registry_path,
            str(args.analytics_export_run_id),
        )
        output_dir = report_output_dir(binding, report_id)
        payload = publish_semantic_montage_report(
            semantic_manifest_path=args.semantic_manifest,
            output_dir=output_dir,
            report_id=report_id,
            analytics_export=binding.to_dict(),
        )
        if args.index_registry:
            registry = Registry(registry_path)
            try:
                indexed_export_run_id, indexed_report_id = index_report_manifest(
                    registry,
                    Path(payload["manifest_path"]),
                )
            finally:
                registry.close()
            payload["registry_index"] = {
                "export_run_id": indexed_export_run_id,
                "report_id": indexed_report_id,
            }
        print(
            json.dumps(
                payload,
                indent=None if args.compact else 2,
                sort_keys=True,
                separators=(",", ":") if args.compact else None,
            )
        )
        return 0
    if args.command == "index-report":
        registry = Registry(registry_path)
        try:
            export_run_id, report_id = index_report_manifest(
                registry,
                args.manifest,
                status=str(args.status),
            )
        finally:
            registry.close()
        payload = {"export_run_id": export_run_id, "report_id": report_id}
        print(
            json.dumps(
                payload,
                indent=None if args.compact else 2,
                sort_keys=True,
                separators=(",", ":") if args.compact else None,
            )
        )
        return 0
    if args.command == "query-reports":
        rows = query_indexed_reports(
            registry_path,
            export_run_id=args.export_run_id,
            report_id=args.report_id,
            visualization_id=args.visualization_id,
            status=str(args.status),
            latest=bool(args.latest),
            limit=int(args.limit),
        )
        if args.format == "json":
            print(json.dumps(rows, sort_keys=True))
        elif args.format == "path":
            for row in rows:
                print(row["report_manifest_path"])
        else:
            headers = (
                "export_run_id",
                "report_id",
                "status",
                "visualization_count",
                "artifact_count",
                "nonready_count",
                "report_manifest_path",
            )
            print("\t".join(headers))
            for row in rows:
                print(
                    "\t".join(
                        "" if row.get(key) is None else str(row[key]) for key in headers
                    )
                )
        return 0

    requested_providers = list(args.provider)
    if args.command in {"montage", "export"}:
        requested_providers.extend(
            VISUALIZATIONS[visualization_id].provider_id
            for visualization_id in args.visualization_id
        )
    plan_kwargs = dict(
        registry_path=registry_path,
        protocol_name=args.protocol_name,
        recording_ids=args.recording_id,
        recording_id_contains=args.recording_id_contains,
        path_contains=args.path_contains,
        zarr_use=str(args.zarr_use),
        status=str(args.status),
        limit=args.limit,
        all_recordings=bool(args.all_recordings),
        provider_ids=tuple(dict.fromkeys(requested_providers)),
        include_not_applicable=bool(args.include_not_applicable),
    )
    plan = build_report_plan(**plan_kwargs)
    if args.command == "montage":
        payload = build_semantic_visualization_montages(
            plan=plan,
            output_dir=args.output_dir,
            visualization_ids=tuple(dict.fromkeys(args.visualization_id)),
            columns=int(args.columns),
            tile_width=int(args.tile_width),
            max_image_height=int(args.max_image_height),
            fail_on_nonready=not bool(args.allow_nonready),
            overwrite=bool(args.overwrite),
        )
        print(
            json.dumps(
                payload,
                indent=None if args.compact else 2,
                sort_keys=True,
                separators=(",", ":") if args.compact else None,
            )
        )
        return 0
    if args.command == "export":
        binding = None
        output_dir = args.output_dir
        if args.analytics_export_run_id:
            if not args.report_id:
                raise SystemExit(
                    "--report-id is required with --analytics-export-run-id"
                )
            report_id = validate_report_id(str(args.report_id))
            binding = resolve_analytics_export_binding(
                registry_path,
                str(args.analytics_export_run_id),
            )
            canonical_output = report_output_dir(binding, report_id)
            if output_dir is not None and output_dir.expanduser().resolve() != canonical_output:
                raise SystemExit(
                    "--output-dir must match the canonical analytics report path: "
                    f"{canonical_output}"
                )
            output_dir = canonical_output
        elif args.report_id:
            validate_report_id(str(args.report_id))
        if output_dir is None:
            raise SystemExit(
                "--output-dir is required unless --analytics-export-run-id is provided"
            )
        if args.index_registry and binding is None:
            raise SystemExit("--index-registry requires --analytics-export-run-id")
        payload = export_report_bundle(
            plan=plan,
            output_dir=output_dir,
            materialization_policy=str(args.materialization),
            visualization_ids=tuple(dict.fromkeys(args.visualization_id)),
            fail_on_nonready=not bool(args.allow_nonready),
            source_collection_manifest=args.source_collection_manifest,
            report_id=args.report_id,
            analytics_export=binding.to_dict() if binding is not None else None,
        )
        if args.index_registry:
            registry = Registry(registry_path)
            try:
                indexed_export_run_id, indexed_report_id = index_report_manifest(
                    registry,
                    Path(payload["manifest_path"]),
                )
            finally:
                registry.close()
            payload["registry_index"] = {
                "export_run_id": indexed_export_run_id,
                "report_id": indexed_report_id,
            }
        print(
            json.dumps(
                payload,
                indent=None if args.compact else 2,
                sort_keys=True,
                separators=(",", ":") if args.compact else None,
            )
        )
        return 0
    if args.command == "apply":
        results = execute_report_plan(
            plan,
            render_missing=bool(args.render_missing),
            apply_analysis=bool(args.apply_analysis),
            refresh_contract_mismatches=bool(args.refresh_contract_mismatches),
            visualization_ids=args.visualization_id,
            overwrite_analysis=bool(args.overwrite_analysis),
            continue_on_error=bool(args.continue_on_error),
        )
        after = build_report_plan(**plan_kwargs)
        payload = {
            "schema_id": "palette.dataset_report_execution.v1",
            "before": report_plan_to_dict(plan),
            "results": [execution_result_to_dict(result) for result in results],
            "after": report_plan_to_dict(after),
        }
        print(
            json.dumps(
                payload,
                indent=None if args.compact else 2,
                sort_keys=True,
                separators=(",", ":") if args.compact else None,
            )
        )
        return 0 if all(result.status != "failed" for result in results) else 1
    print(report_plan_json(plan, pretty=not bool(args.compact)))
    return 0
