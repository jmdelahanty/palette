"""Plan or publish one explicit chaser input-provenance proxy run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    select_chaser_input_provenance_proxy,
)
from fisheye.analysis_workflows.chaser_input_provenance_proxy_storage import (
    prepare_chaser_input_provenance_proxy,
)
from fisheye.analysis_workflows.materializers.chaser_input_provenance_proxy import (
    build_chaser_input_provenance_proxy_materialization_plan,
    materialize_chaser_input_provenance_proxy,
)
from fisheye.analysis_workflows.provider_chaser_stimulus_source_handle import (
    load_provider_chaser_stimulus_source_handle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one selector-ineligible controller input-provenance "
            "proxy from an exact native provider-chaser candidate."
        )
    )
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--source-run-name", required=True)
    parser.add_argument("--output-run-name", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish the named immutable candidate; otherwise emit a dry-run plan.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    archive = args.analysis_zarr.expanduser().resolve()
    source = load_provider_chaser_stimulus_source_handle(
        archive,
        run_name=args.source_run_name,
        expected_recording_id=args.expected_recording_id,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        use_consolidated=True,
    )
    result = select_chaser_input_provenance_proxy(source)
    prepared = prepare_chaser_input_provenance_proxy(result)
    if args.apply:
        publication = materialize_chaser_input_provenance_proxy(
            archive,
            prepared=prepared,
            scratch_root=args.scratch_root,
            run_name=args.output_run_name,
            copy_backend=args.copy_backend,
        )
        return {
            "schema_id": "palette.chaser_input_provenance_proxy_cli_result",
            "schema_version": 1,
            "status": "published_selector_ineligible",
            "source_run_path": source.run_path,
            "source_manifest_sha256": source.manifest_sha256,
            "acquisition_projection_record_sha256": (
                result.acquisition_projection_record_sha256
            ),
            "prepared_manifest_sha256": prepared.payload_digest,
            "publication": publication,
        }
    plan = build_chaser_input_provenance_proxy_materialization_plan(
        archive,
        scratch_root=args.scratch_root,
        run_name=args.output_run_name,
        prepared=prepared,
    )
    return {
        "schema_id": "palette.chaser_input_provenance_proxy_cli_result",
        "schema_version": 1,
        "status": "planned_no_writes",
        "source_run_path": source.run_path,
        "source_manifest_sha256": source.manifest_sha256,
        "acquisition_projection_record_sha256": (
            result.acquisition_projection_record_sha256
        ),
        "prepared_manifest_sha256": prepared.payload_digest,
        "plan": plan.to_json(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
