"""Guarded, dependency-aware cohort release submission.

The workstation performs only a read-only registry query and writes the frozen
selection/submission records. All analysis, collection binding, export,
statistics, montage, and report validation work is submitted to LSF workers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

from fisheye.cohorts.registry import (
    CohortSelectionError,
    build_cohort_plan,
    coverage_report,
    freeze_cohort,
)
from fisheye.cohorts.spec import (
    CohortSpec,
    CohortSpecError,
    DatasetSelector,
    DpfSelector,
    PrerequisiteSelector,
    ProtocolSelector,
    SubjectSelector,
    load_cohort_spec,
)
from fisheye.analytics_exports.chaser_authority import (
    load_chaser_export_authority_set,
)


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_REGISTRY = Path(
    "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
)
DEFAULT_OUTPUT_ROOT = Path("/groups/johnson/johnsonlab/palette_analytics")
DEFAULT_PALETTE_REPO = Path("/groups/johnson/johnsonlab/jeremy/gitrepos/palette")
DEFAULT_PROTOCOL_PROFILE = Path(
    "src/fisheye/analysis/profiles/chaser_event_windows_v1.yaml"
)
DEFAULT_ANALYSIS_PROFILE = Path("src/fisheye/analysis/profiles/chaser_behavior_v1.yaml")


class CohortReleaseError(RuntimeError):
    """Raised when a release cannot be rendered or submitted safely."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _job_id(output: str) -> str:
    matches = re.findall(r"^job_id=([1-9][0-9]*)$", output, flags=re.MULTILINE)
    if not matches:
        raise CohortReleaseError("submission wrapper did not report a numeric job_id")
    return matches[-1]


def _run_wrapper(command: Sequence[str]) -> str:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise CohortReleaseError(
            f"submission wrapper failed ({result.returncode}): {' '.join(command)}"
        )
    return result.stdout


def _spec_from_args(args: argparse.Namespace) -> CohortSpec:
    if args.spec is not None:
        return load_cohort_spec(args.spec)
    if not args.cohort_id or not args.cohort_name:
        raise CohortSpecError("direct selectors require --cohort-id and --cohort-name")
    if not (args.stimulus_mode or args.protocol_name or args.protocol_hash):
        raise CohortSpecError(
            "direct selectors require at least one --stimulus-mode, --protocol-name, "
            "or --protocol-hash"
        )
    return CohortSpec(
        cohort_id=str(args.cohort_id),
        cohort_name=str(args.cohort_name),
        purpose=str(args.purpose),
        dataset=DatasetSelector(),
        protocol=ProtocolSelector(
            stimulus_modes_any=tuple(
                dict.fromkeys(
                    str(value).strip().upper() for value in args.stimulus_mode
                )
            ),
            protocol_names_any=tuple(
                dict.fromkeys(str(value).strip() for value in args.protocol_name)
            ),
            protocol_hashes_any=tuple(
                dict.fromkeys(
                    str(value).strip().lower() for value in args.protocol_hash
                )
            ),
        ),
        subjects=SubjectSelector(
            dpf=DpfSelector(
                values=tuple(dict.fromkeys(args.dpf)),
                minimum=args.dpf_min,
                maximum=args.dpf_max,
            ),
            line_strains_any=tuple(dict.fromkeys(args.strain)),
            genotypes_any=tuple(dict.fromkeys(args.genotype)),
            cross_ids_any=tuple(dict.fromkeys(args.cross_id)),
            match_policy=str(args.subject_match_policy),
        ),
        prerequisites=PrerequisiteSelector(
            required_steps_ok=tuple(
                dict.fromkeys(
                    str(value).strip().lower() for value in args.require_step_ok
                )
            )
        ),
        missing_selected_metadata=str(args.missing_selected_metadata),
    )


def _validate_direct_spec(spec: CohortSpec) -> CohortSpec:
    # Round-trip through the public validator so direct CLI values receive the
    # same hash and validation semantics as YAML/JSON specifications.
    return CohortSpec.from_mapping(spec.to_mapping())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a typed registry cohort and render or submit its LSF release DAG."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", type=Path, help="Versioned cohort query YAML/JSON.")
    source.add_argument(
        "--direct-selectors",
        action="store_true",
        help="Build the query from the typed selector flags below.",
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--cohort-id")
    parser.add_argument("--cohort-name")
    parser.add_argument(
        "--purpose", default="registry-defined chaser analytics cohort release"
    )
    parser.add_argument("--stimulus-mode", action="append", default=[])
    parser.add_argument("--protocol-name", action="append", default=[])
    parser.add_argument("--protocol-hash", action="append", default=[])
    parser.add_argument("--dpf", type=int, action="append", default=[])
    parser.add_argument("--dpf-min", type=int)
    parser.add_argument("--dpf-max", type=int)
    parser.add_argument("--strain", action="append", default=[])
    parser.add_argument("--genotype", action="append", default=[])
    parser.add_argument("--cross-id", action="append", default=[])
    parser.add_argument(
        "--subject-match-policy",
        choices=("unambiguous_recording", "any_subject", "all_subjects"),
        default="unambiguous_recording",
    )
    parser.add_argument(
        "--missing-selected-metadata", choices=("error", "exclude"), default="error"
    )
    parser.add_argument("--require-step-ok", action="append", default=[])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--palette-repo", type=Path, default=DEFAULT_PALETTE_REPO)
    parser.add_argument(
        "--chaser-authority-manifest",
        type=Path,
        required=True,
        help="Immutable exact chaser export authority-set JSON.",
    )
    parser.add_argument("--chaser-authority-sha256")
    parser.add_argument("--submit-host", default="login1-citrus-poller")
    parser.add_argument(
        "--protocol-profile", type=Path, default=DEFAULT_PROTOCOL_PROFILE
    )
    parser.add_argument(
        "--analysis-profile", type=Path, default=DEFAULT_ANALYSIS_PROFILE
    )
    parser.add_argument("--preset", default="chaser_v1")
    parser.add_argument("--queue")
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=16)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--export-ncores", type=int, default=4)
    parser.add_argument("--export-mem-gb", type=int, default=16)
    parser.add_argument("--include-baseline-samples", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--allow-nonready-report", action="store_true")
    parser.add_argument("--report-id")
    parser.add_argument("--visualization-id", action="append", default=[])
    parser.add_argument("--submit", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not SAFE_ID.fullmatch(str(args.release_id)):
        raise CohortReleaseError(f"unsafe --release-id: {args.release_id}")
    if args.report_id and not SAFE_ID.fullmatch(str(args.report_id)):
        raise CohortReleaseError(f"unsafe --report-id: {args.report_id}")
    for name in ("ncores", "mem_gb", "max_active", "export_ncores", "export_mem_gb"):
        if int(getattr(args, name)) < 1:
            raise CohortReleaseError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.dpf_min is not None
        and args.dpf_max is not None
        and args.dpf_min > args.dpf_max
    ):
        raise CohortReleaseError("--dpf-min cannot exceed --dpf-max")
    for value in args.protocol_hash:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value)):
            raise CohortReleaseError(
                "--protocol-hash must be a 64-character SHA-256 hex value"
            )
    if args.chaser_authority_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(args.chaser_authority_sha256)
    ):
        raise CohortReleaseError(
            "--chaser-authority-sha256 must be a lowercase SHA-256 digest"
        )


def _submission_preflight(script_repo: Path, cluster_repo: Path) -> str:
    if _git(script_repo, "status", "--porcelain"):
        raise CohortReleaseError(
            "submission requires a clean workstation checkout so the frozen cohort code is committed"
        )
    local_commit = _git(script_repo, "rev-parse", "HEAD")
    cluster_commit = _git(cluster_repo, "rev-parse", "HEAD")
    if local_commit != cluster_commit:
        raise CohortReleaseError(
            "submission requires the workstation and cluster-visible Palette checkouts "
            f"at the same commit (workstation={local_commit}, cluster={cluster_commit})"
        )
    return local_commit


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    release_record_path: Path | None = None
    release_record: dict[str, Any] | None = None
    try:
        _validate_args(args)
        spec = _validate_direct_spec(_spec_from_args(args))
        if "CHASER" not in spec.protocol.stimulus_modes_any:
            raise CohortReleaseError(
                "release v1 executes chaser analytics and therefore requires CHASER in "
                "protocol.stimulus_modes_any"
            )
        script_repo = Path(__file__).resolve().parents[3]
        registry = args.registry.expanduser().resolve()
        output_root = args.output_root.expanduser().resolve()
        cluster_repo = args.palette_repo.expanduser().resolve()
        if not registry.is_file():
            raise CohortReleaseError(f"registry not found: {registry}")
        if not cluster_repo.is_dir():
            raise CohortReleaseError(
                f"cluster-visible Palette checkout not found: {cluster_repo}"
            )

        release_dir = output_root / "logs" / "releases" / str(args.release_id)
        if release_dir.exists():
            raise CohortReleaseError(f"release directory already exists: {release_dir}")
        release_dir.mkdir(parents=True)
        query_path = release_dir / "cohort_query.json"
        plan_path = release_dir / "cohort_plan.json"
        coverage_path = release_dir / "metadata_coverage.json"
        cohort_path = release_dir / "frozen_cohort_manifest.json"
        zarr_list = release_dir / "zarr_paths.txt"
        release_record_path = release_dir / "release_submission.json"
        _write_json(query_path, spec.to_mapping())

        plan = build_cohort_plan(registry, spec)
        coverage = coverage_report(registry, spec, plan=plan)
        _write_json(plan_path, plan)
        _write_json(coverage_path, coverage)
        cohort = freeze_cohort(plan)
        _write_json(cohort_path, cohort)
        zarr_list.write_text(
            "".join(f"{member['zarr_path']}\n" for member in cohort["members"]),
            encoding="utf-8",
        )
        authority = load_chaser_export_authority_set(
            args.chaser_authority_manifest,
            expected_file_sha256=args.chaser_authority_sha256,
        )
        cohort_sources = {
            str(Path(member["zarr_path"]).expanduser().resolve())
            for member in cohort["members"]
        }
        authority_sources = set(authority.sources_by_path)
        if authority_sources != cohort_sources:
            raise CohortReleaseError(
                "chaser authority source set must exactly match the frozen cohort "
                f"(missing={sorted(cohort_sources - authority_sources)!r}, "
                f"unexpected={sorted(authority_sources - cohort_sources)!r})"
            )

        palette_commit = _git(cluster_repo, "rev-parse", "HEAD")
        if args.submit:
            palette_commit = _submission_preflight(script_repo, cluster_repo)

        collection_manifest = (
            output_root
            / "v1"
            / "manifests"
            / "collections"
            / f"{args.release_id}.manifest.json"
        )
        export_manifest = (
            output_root / "v1" / "manifests" / f"export_run_id={args.release_id}.json"
        )
        report_id = args.report_id or f"{args.release_id}_recording_montages"
        release_record = {
            "schema_id": "palette.cohort_release_submission",
            "schema_version": 1,
            "release_id": args.release_id,
            "mode": "submit" if args.submit else "render_only",
            "created_utc": _utc_now(),
            "palette_commit": palette_commit,
            "registry_path": str(registry),
            "cohort_query_path": str(query_path),
            "cohort_query_sha256": spec.sha256,
            "cohort_plan_path": str(plan_path),
            "metadata_coverage_path": str(coverage_path),
            "frozen_cohort_manifest_path": str(cohort_path),
            "frozen_cohort_manifest_sha256": cohort["manifest_sha256"],
            "zarr_list_path": str(zarr_list),
            "zarr_list_sha256": _sha256_file(zarr_list),
            "chaser_authority_manifest_path": str(authority.path),
            "chaser_authority_file_sha256": authority.file_sha256,
            "chaser_authority_record_sha256": authority.record["record_sha256"],
            "member_count": cohort["member_count"],
            "expected_artifacts": {
                "collection_manifest": str(collection_manifest),
                "analytics_export_manifest": str(export_manifest),
                "statistics_export_run_id": f"{args.release_id}_stats",
                "report_id": None if args.skip_report else report_id,
            },
            "stages": [],
            "status": "rendering" if not args.submit else "submitting",
        }
        _write_json(release_record_path, release_record)

        jobs_root = release_dir / "jobs"
        jobs_root.mkdir()
        placeholder = 900000

        def run_stage(name: str, command: list[str]) -> str:
            nonlocal placeholder
            output = _run_wrapper(command)
            if args.submit:
                stage_job_id = _job_id(output)
            else:
                placeholder += 1
                stage_job_id = str(placeholder)
            release_record["stages"].append(
                {
                    "name": name,
                    "job_id": stage_job_id,
                    "job_id_kind": "lsf" if args.submit else "render_placeholder",
                    "command": command,
                }
            )
            _write_json(release_record_path, release_record)
            return stage_job_id

        analytics_cmd = [
            "bash",
            str(script_repo / "scripts" / "submit_chaser_analytics_bsub.sh"),
            "--zarr-list",
            str(zarr_list),
            "--palette-repo",
            str(cluster_repo),
            "--submit-host",
            str(args.submit_host),
            "--protocol-profile",
            str(args.protocol_profile),
            "--analysis-profile",
            str(args.analysis_profile),
            "--preset",
            str(args.preset),
            "--run-id",
            str(args.release_id),
            "--log-dir",
            str(jobs_root),
            "--ncores",
            str(args.ncores),
            "--mem-gb",
            str(args.mem_gb),
            "--max-active",
            str(args.max_active),
        ]
        if args.queue:
            analytics_cmd.extend(["--queue", str(args.queue)])
        if not args.submit:
            analytics_cmd.append("--dry-run")
        analytics_job = run_stage("recording_analytics", analytics_cmd)

        collection_cmd = [
            "bash",
            str(
                script_repo / "scripts" / "submit_registry_collection_manifest_bsub.sh"
            ),
            "--collection-id",
            str(args.release_id),
            "--collection-name",
            spec.cohort_name,
            "--zarr-list",
            str(zarr_list),
            "--profile",
            "chaser",
            "--output",
            str(collection_manifest),
            "--output-root",
            str(output_root),
            "--palette-repo",
            str(cluster_repo),
            "--submit-host",
            str(args.submit_host),
            "--dependency-done",
            analytics_job,
        ]
        if args.queue:
            collection_cmd.extend(["--queue", str(args.queue)])
        if args.submit:
            collection_cmd.append("--submit")
        collection_job = run_stage("collection_binding", collection_cmd)

        export_cmd = [
            "bash",
            str(script_repo / "scripts" / "submit_analytics_export_bsub.sh"),
            "--collection-manifest",
            str(collection_manifest),
            "--chaser-authority-manifest",
            str(authority.path),
            "--chaser-authority-sha256",
            authority.file_sha256,
            "--export-run-id",
            str(args.release_id),
            "--output-root",
            str(output_root),
            "--palette-repo",
            str(cluster_repo),
            "--submit-host",
            str(args.submit_host),
            "--registry",
            str(registry),
            "--index-registry",
            "--log-dir",
            str(jobs_root),
            "--ncores",
            str(args.export_ncores),
            "--mem-gb",
            str(args.export_mem_gb),
            "--dependency-done",
            collection_job,
        ]
        if args.include_baseline_samples:
            export_cmd.append("--include-baseline-samples")
        if args.queue:
            export_cmd.extend(["--queue", str(args.queue)])
        if args.submit:
            export_cmd.append("--submit")
        export_job = run_stage("analytics_export_and_statistics", export_cmd)

        if not args.skip_report:
            report_cmd = [
                "bash",
                str(script_repo / "scripts" / "submit_cohort_report_bsub.sh"),
                "--cohort-manifest",
                str(cohort_path),
                "--export-run-id",
                str(args.release_id),
                "--report-id",
                str(report_id),
                "--registry",
                str(registry),
                "--output-root",
                str(output_root),
                "--palette-repo",
                str(cluster_repo),
                "--submit-host",
                str(args.submit_host),
                "--log-dir",
                str(jobs_root),
                "--dependency-done",
                export_job,
            ]
            for visualization_id in args.visualization_id:
                report_cmd.extend(["--visualization-id", str(visualization_id)])
            if args.allow_nonready_report:
                report_cmd.append("--allow-nonready")
            if args.queue:
                report_cmd.extend(["--queue", str(args.queue)])
            if args.submit:
                report_cmd.append("--submit")
            run_stage("semantic_montages_and_report", report_cmd)

        release_record["status"] = "submitted" if args.submit else "rendered"
        release_record["completed_utc"] = _utc_now()
        _write_json(release_record_path, release_record)
        print(f"release_dir={release_dir}")
        print(f"release_submission={release_record_path}")
        print(f"frozen_cohort_manifest={cohort_path}")
        print(f"member_count={cohort['member_count']}")
        return 0
    except (
        CohortReleaseError,
        CohortSelectionError,
        CohortSpecError,
        FileExistsError,
        FileNotFoundError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        if release_record_path is not None and release_record is not None:
            release_record["status"] = "failed"
            release_record["failed_utc"] = _utc_now()
            release_record["error"] = str(exc)
            _write_json(release_record_path, release_record)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
