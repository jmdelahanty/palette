"""Submit a clipped detect/refine workflow plan to LSF.

The input is the dry-run JSON emitted by
``fisheye.utils.plan_clipped_detect_refine_workflow``. This submitter keeps the
cluster workflow conservative:

* dry-run is the default;
* actual submission refuses more than one work unit unless ``--allow-multiple``
  is provided;
* GPU work is limited to the detect artifact stage;
* import, validation, detect-quality, and refine stages are CPU jobs chained by
  LSF ``done(<jobid>)`` dependencies;
* one final CPU job validates the completed clip collection before shared
  collection metadata is written.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now_compact as _utc_now_compact
from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SUBMISSION_SCHEMA = "palette.clipped_detect_refine_bsub_submission.v1"
PLAN_SCHEMA = "palette.clipped_detect_refine_workflow_plan.v1"


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _safe_component(value: Any, *, fallback: str = "item") -> str:
    text = str(value or "").strip()
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return label or fallback


def _parse_bsub_job_id(output: str) -> str:
    match = re.search(r"Job <(\d+)>", output)
    if not match:
        raise RuntimeError(f"Could not parse LSF job id from bsub output:\n{output}")
    return match.group(1)


def _command(parts: Sequence[Any]) -> str:
    return shlex.join(str(part) for part in parts if part is not None and str(part) != "")


def _load_plan(plan_path: Path) -> dict[str, Any]:
    payload = _read_json(plan_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Plan JSON must be an object: {plan_path}")
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(
            f"Expected {PLAN_SCHEMA}, got {payload.get('schema_version')!r}: {plan_path}"
        )
    work_units = payload.get("work_units")
    if not isinstance(work_units, list) or not work_units:
        raise ValueError(f"Plan has no work_units: {plan_path}")
    return payload


def _group_path_exists(zarr_path: Path, path: str) -> bool:
    normalized = str(path or "").strip().strip("/")
    if not normalized:
        return False
    group_path = zarr_path / normalized
    return (group_path / "zarr.json").exists() or (group_path / ".zgroup").exists()


def _target_preflight(
    plan: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    *,
    include_finalizer: bool,
) -> dict[str, Any]:
    zarr_value = plan.get("analysis_zarr")
    if not zarr_value:
        return {
            "status": "unchecked",
            "reason": "plan has no analysis_zarr",
            "collisions": [],
            "checked_paths": [],
        }
    zarr_path = Path(str(zarr_value)).expanduser().resolve()
    if not zarr_path.exists():
        return {
            "status": "unchecked",
            "reason": f"analysis_zarr does not exist: {zarr_path}",
            "analysis_zarr": str(zarr_path),
            "collisions": [],
            "checked_paths": [],
        }

    checked_paths: list[dict[str, str]] = []
    collisions: list[dict[str, str]] = []
    for unit in units:
        work_unit_id = str(unit.get("work_unit_id") or "")
        paths = unit.get("zarr_paths", {})
        run_names = unit.get("run_names", {})
        detect_path = str(paths.get("detect_target_group_path") or "")
        refined_path = str(paths.get("refined_group_path") or "")
        quality_path = ""
        quality_run = str(run_names.get("detect_quality") or "")
        if detect_path and quality_run:
            quality_path = f"{detect_path}/quality_reports/{quality_run}"
        for kind, path in (
            ("detect_target", detect_path),
            ("detect_quality_target", quality_path),
            ("refined_target", refined_path),
        ):
            if not path:
                continue
            record = {"work_unit_id": work_unit_id, "kind": kind, "path": path}
            checked_paths.append(record)
            if _group_path_exists(zarr_path, path):
                collisions.append(record)

    if include_finalizer:
        collection_id = str(plan.get("workflow_id") or "clipped_detect_refine")
        finalizer_path = f"experiment_index/finalized_runs/{collection_id}"
        record = {"work_unit_id": "recording_collection", "kind": "finalized_collection", "path": finalizer_path}
        checked_paths.append(record)
        if _group_path_exists(zarr_path, finalizer_path):
            collisions.append(record)

    return {
        "status": "blocked" if collisions else "ok",
        "analysis_zarr": str(zarr_path),
        "collisions": collisions,
        "checked_paths": checked_paths,
    }


def _filter_work_units(
    plan: Mapping[str, Any],
    *,
    work_unit_ids: Optional[Sequence[str]],
    clip_ids: Optional[Sequence[str]],
    limit: Optional[int],
) -> list[dict[str, Any]]:
    selected_ids = {str(value) for value in work_unit_ids} if work_unit_ids else None
    selected_clips = {str(value) for value in clip_ids} if clip_ids else None
    units: list[dict[str, Any]] = []
    for raw_unit in plan["work_units"]:
        if not isinstance(raw_unit, dict):
            raise ValueError("Plan work_unit entry is not an object")
        if selected_ids is not None and str(raw_unit.get("work_unit_id")) not in selected_ids:
            continue
        if selected_clips is not None and str(raw_unit.get("clip_id")) not in selected_clips:
            continue
        units.append(raw_unit)
    if limit is not None:
        units = units[: int(limit)]
    if not units:
        raise ValueError("No plan work units matched the requested filters")
    return units


def _job_cache_shell(prefix: str) -> str:
    return f"""scratch_user="${{USER:-$(id -un)}}"
if [[ -n "${{LSB_JOBID:-}}" && -d "/scratch/${{scratch_user}}" ]]; then
  export PALETTE_JOB_CACHE="/scratch/${{scratch_user}}/${{LSB_JOBID}}/palette_cache"
else
  export PALETTE_JOB_CACHE="${{TMPDIR:-/tmp}}/{prefix}_${{JOB_ID}}/palette_cache"
fi
export MPLBACKEND=Agg
mkdir -p "$PALETTE_JOB_CACHE" "$RUN_DIR"
"""


def _write_stage_script(
    path: Path,
    *,
    repo: Path,
    run_dir: Path,
    work_unit: Mapping[str, Any],
    stage: str,
    command: str,
    status_path: Path,
    extra_env: Optional[Mapping[str, str]] = None,
) -> None:
    work_unit_id = str(work_unit["work_unit_id"])
    clip_id = str(work_unit.get("clip_id") or "")
    camera_serial = str(work_unit.get("camera_serial") or "")
    detect_run = str(work_unit.get("run_names", {}).get("detect") or "")
    quality_run = str(work_unit.get("run_names", {}).get("detect_quality") or "")
    refined_run = str(work_unit.get("run_names", {}).get("refined_detect") or "")
    env_lines = []
    for key, value in (extra_env or {}).items():
        env_lines.append(f"export {key}={shlex.quote(str(value))}")
    env_block = "\n".join(env_lines)
    content = f"""#!/usr/bin/env bash
set -euo pipefail

cd {shlex.quote(str(repo))}

RUN_DIR={shlex.quote(str(run_dir))}
JOB_ID="${{LSB_JOBID:-manual_{stage}}}"
STATUS_JSON_TEMPLATE={shlex.quote(str(status_path))}
STATUS_JSON="${{STATUS_JSON_TEMPLATE//<jobid>/$JOB_ID}}"
WORK_UNIT_ID={shlex.quote(work_unit_id)}
CLIP_ID={shlex.quote(clip_id)}
CAMERA_SERIAL={shlex.quote(camera_serial)}
DETECT_RUN={shlex.quote(detect_run)}
QUALITY_RUN={shlex.quote(quality_run)}
REFINED_RUN={shlex.quote(refined_run)}
export WORK_UNIT_ID CLIP_ID CAMERA_SERIAL DETECT_RUN QUALITY_RUN REFINED_RUN
{env_block}

{_job_cache_shell(f"palette_{stage}")}

echo "repo=$(pwd)"
echo "host=$(hostname)"
echo "job_id=$JOB_ID"
echo "stage={stage}"
echo "work_unit_id=$WORK_UNIT_ID"
echo "clip_id=$CLIP_ID"
echo "camera_serial=$CAMERA_SERIAL"
echo "palette_job_cache=$PALETTE_JOB_CACHE"
echo "status_json=$STATUS_JSON"

stage_start_ns=""
write_stage_status() {{
  local status_value="$1"
  local exit_code="$2"
  local stage_end_ns
  local stage_seconds=""
  stage_end_ns="$(date +%s%N)"
  if [[ -n "${{stage_start_ns:-}}" ]]; then
    stage_seconds="$(awk -v s="$stage_start_ns" -v e="$stage_end_ns" 'BEGIN {{ printf "%.6f", (e - s) / 1000000000 }}')"
  fi
  export STAGE_STATUS="$status_value"
  export STAGE_EXIT_CODE="$exit_code"
  export STAGE_SECONDS="$stage_seconds"
  scripts/py - "$STATUS_JSON" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
seconds = os.environ.get("STAGE_SECONDS") or None
payload = {{
    "status": os.environ.get("STAGE_STATUS"),
    "schema_version": 1,
    "stage": "{stage}",
    "work_unit_id": os.environ.get("WORK_UNIT_ID"),
    "clip_id": os.environ.get("CLIP_ID"),
    "camera_serial": os.environ.get("CAMERA_SERIAL"),
    "detect_run": os.environ.get("DETECT_RUN"),
    "quality_run": os.environ.get("QUALITY_RUN"),
    "refined_run": os.environ.get("REFINED_RUN"),
    "exit_code": int(os.environ.get("STAGE_EXIT_CODE", "0")),
    "job_id": os.environ.get("LSB_JOBID"),
    "queue": os.environ.get("LSB_QUEUE"),
    "slots": os.environ.get("LSB_DJOB_NUMPROC"),
    "host": socket.gethostname(),
    "started_at_utc": os.environ.get("STAGE_STARTED_AT_UTC"),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage_seconds": float(seconds) if seconds is not None else None,
    "palette_job_cache": os.environ.get("PALETTE_JOB_CACHE"),
}}
status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(f"status_json={{status_path}}")
PY
}}

trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_stage_status failed "$rc" || true; fi' EXIT
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stage_start_ns="$(date +%s%N)"
export STAGE_STARTED_AT_UTC="$started_at"
{command}
write_stage_status ok 0
trap - EXIT
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _write_finalizer_script(
    path: Path,
    *,
    repo: Path,
    run_dir: Path,
    command: str,
    status_path: Path,
) -> None:
    content = f"""#!/usr/bin/env bash
set -euo pipefail

cd {shlex.quote(str(repo))}

RUN_DIR={shlex.quote(str(run_dir))}
JOB_ID="${{LSB_JOBID:-manual_finalize_recording_collection}}"
STATUS_JSON_TEMPLATE={shlex.quote(str(status_path))}
STATUS_JSON="${{STATUS_JSON_TEMPLATE//<jobid>/$JOB_ID}}"
export WORK_UNIT_ID="recording_collection"
export CLIP_ID=""
export CAMERA_SERIAL=""
export DETECT_RUN=""
export QUALITY_RUN=""
export REFINED_RUN=""

{_job_cache_shell("palette_finalize_recording_collection")}

echo "repo=$(pwd)"
echo "host=$(hostname)"
echo "job_id=$JOB_ID"
echo "stage=finalize_recording_collection"
echo "palette_job_cache=$PALETTE_JOB_CACHE"
echo "status_json=$STATUS_JSON"

stage_start_ns=""
write_stage_status() {{
  local status_value="$1"
  local exit_code="$2"
  local stage_end_ns
  local stage_seconds=""
  stage_end_ns="$(date +%s%N)"
  if [[ -n "${{stage_start_ns:-}}" ]]; then
    stage_seconds="$(awk -v s="$stage_start_ns" -v e="$stage_end_ns" 'BEGIN {{ printf "%.6f", (e - s) / 1000000000 }}')"
  fi
  export STAGE_STATUS="$status_value"
  export STAGE_EXIT_CODE="$exit_code"
  export STAGE_SECONDS="$stage_seconds"
  scripts/py - "$STATUS_JSON" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
seconds = os.environ.get("STAGE_SECONDS") or None
payload = {{
    "status": os.environ.get("STAGE_STATUS"),
    "schema_version": 1,
    "stage": "finalize_recording_collection",
    "exit_code": int(os.environ.get("STAGE_EXIT_CODE", "0")),
    "job_id": os.environ.get("LSB_JOBID"),
    "queue": os.environ.get("LSB_QUEUE"),
    "slots": os.environ.get("LSB_DJOB_NUMPROC"),
    "host": socket.gethostname(),
    "started_at_utc": os.environ.get("STAGE_STARTED_AT_UTC"),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage_seconds": float(seconds) if seconds is not None else None,
    "palette_job_cache": os.environ.get("PALETTE_JOB_CACHE"),
}}
status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(f"status_json={{status_path}}")
PY
}}

trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_stage_status failed "$rc" || true; fi' EXIT
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stage_start_ns="$(date +%s%N)"
export STAGE_STARTED_AT_UTC="$started_at"
{command}
write_stage_status ok 0
trap - EXIT
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _bsub_args(
    *,
    job_name: str,
    queue: str,
    ncores: int,
    mem_gb: int,
    walltime: str,
    stdout_path: Path,
    stderr_path: Path,
    dependency: Optional[str] = None,
) -> list[str]:
    args = [
        "-J",
        job_name,
        "-n",
        str(ncores),
        "-W",
        walltime,
        "-R",
        f"rusage[mem={mem_gb}G]",
        "-oo",
        str(stdout_path),
        "-eo",
        str(stderr_path),
    ]
    if queue:
        args.extend(["-q", queue])
    if dependency:
        args.extend(["-w", dependency])
    return args


def _submit_bsub(args: Sequence[str], script: Path, *, cwd: Path) -> tuple[str, str]:
    command = ["bsub", *args, "bash", str(script)]
    result = subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return _parse_bsub_job_id(result.stdout), result.stdout


def _run_detect_submit(command: str, *, repo: Path) -> tuple[str, str]:
    result = subprocess.run(
        shlex.split(command),
        cwd=str(repo),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return _parse_bsub_job_id(result.stdout), result.stdout


def _materialized_tarball_path(unit: Mapping[str, Any], detect_jobid: str) -> Path:
    expected = str(unit["artifact_paths"]["expected_tarball"])
    if "<detect_jobid>" not in expected:
        raise ValueError(f"expected_tarball does not include <detect_jobid>: {expected}")
    return Path(expected.replace("<detect_jobid>", detect_jobid))


def build_submission_bundle(
    plan_path: str | Path,
    *,
    repo: str | Path = ".",
    run_dir: str | Path | None = None,
    work_unit_ids: Optional[Sequence[str]] = None,
    clip_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    cpu_queue: str = "short",
    import_ncores: int = 2,
    import_mem_gb: int = 16,
    import_walltime: str = "1:00",
    validate_ncores: int = 2,
    validate_mem_gb: int = 16,
    validate_walltime: str = "1:00",
    quality_ncores: int = 2,
    quality_mem_gb: int = 16,
    quality_walltime: str = "1:00",
    refine_ncores: int = 4,
    refine_mem_gb: int = 32,
    refine_walltime: str = "1:00",
    finalizer_ncores: int = 2,
    finalizer_mem_gb: int = 16,
    finalizer_walltime: str = "1:00",
    include_finalizer: bool = True,
    allow_existing_outputs: bool = False,
    submit: bool = False,
    allow_multiple: bool = False,
) -> dict[str, Any]:
    plan_file = Path(plan_path).expanduser().resolve()
    plan = _load_plan(plan_file)
    units = _filter_work_units(
        plan,
        work_unit_ids=work_unit_ids,
        clip_ids=clip_ids,
        limit=limit,
    )
    if submit and len(units) > 1 and not allow_multiple:
        raise ValueError(
            "Refusing to submit more than one clip-camera without --allow-multiple; "
            "use --limit 1 or --work-unit-id for smoke testing."
        )
    target_preflight = _target_preflight(plan, units, include_finalizer=include_finalizer)
    if (
        submit
        and target_preflight.get("status") == "blocked"
        and not allow_existing_outputs
    ):
        collisions = target_preflight.get("collisions", [])
        raise ValueError(
            "Refusing to submit because planned output targets already exist; "
            "use a new workflow id or pass --allow-existing-outputs. "
            f"First collisions: {collisions[:5]}"
        )

    repo_path = Path(repo).expanduser().resolve()
    workflow_id = str(plan.get("workflow_id") or "clipped_detect_refine")
    workflow_component = _safe_component(workflow_id, fallback="workflow")
    if run_dir is None:
        run_root = repo_path / "runs" / "cluster" / "clipped_detect_refine_bsub"
        bundle_dir = run_root / f"{workflow_component}_{_utc_now_compact()}"
    else:
        bundle_dir = Path(run_dir).expanduser().resolve()
    if bundle_dir.exists():
        raise FileExistsError(f"Submission run dir already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)

    work_items: list[dict[str, Any]] = []
    finalizer_dependencies: list[str] = []
    for unit in units:
        work_unit_id = str(unit["work_unit_id"])
        safe_unit = _safe_component(work_unit_id, fallback="work_unit")
        unit_dir = bundle_dir / safe_unit
        scripts_dir = unit_dir / "scripts"
        logs_dir = unit_dir / "logs"
        status_dir = unit_dir / "status"
        scripts_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        status_dir.mkdir(parents=True)

        detect_jobid = f"<detect_jobid:{safe_unit}>"
        detect_submit_output = None
        if submit:
            detect_jobid, detect_submit_output = _run_detect_submit(
                str(unit["commands"]["detect_submit"]),
                repo=repo_path,
            )
        tarball_path = _materialized_tarball_path(unit, detect_jobid)

        stage_specs = [
            (
                "import_detect",
                _command(
                    [
                        "scripts/py",
                        "-m",
                        "fisheye.utils.import_run_group_artifact",
                        tarball_path,
                        "--use-intended-target",
                        "--apply",
                    ]
                ),
                import_ncores,
                import_mem_gb,
                import_walltime,
                f"done({detect_jobid})",
            ),
            (
                "validate_detect",
                str(unit["commands"]["validate_detect"]),
                validate_ncores,
                validate_mem_gb,
                validate_walltime,
                None,
            ),
            (
                "detect_quality",
                str(unit["commands"]["detect_quality"]),
                quality_ncores,
                quality_mem_gb,
                quality_walltime,
                None,
            ),
            (
                "refine_detect",
                str(unit["commands"]["refine_detect"]),
                refine_ncores,
                refine_mem_gb,
                refine_walltime,
                None,
            ),
            (
                "validate_refined_detect",
                str(unit["commands"]["validate_refined_detect"]),
                validate_ncores,
                validate_mem_gb,
                validate_walltime,
                None,
            ),
        ]

        stage_records: list[dict[str, Any]] = []
        previous_jobid: Optional[str] = detect_jobid
        for stage, command, ncores, mem_gb, walltime, dependency in stage_specs:
            dependency_expr = dependency or (f"done({previous_jobid})" if previous_jobid else None)
            status_path = status_dir / f"{stage}.<jobid>.json"
            script_path = scripts_dir / f"run_{stage}.sh"
            _write_stage_script(
                script_path,
                repo=repo_path,
                run_dir=unit_dir,
                work_unit=unit,
                stage=stage,
                command=command,
                status_path=status_path,
            )
            job_name = _safe_component(f"{stage}_{unit.get('clip_id')}_cam{unit.get('camera_serial')}", fallback=stage)
            stdout_path = logs_dir / f"{stage}_%J.out"
            stderr_path = logs_dir / f"{stage}_%J.err"
            bsub_args = _bsub_args(
                job_name=job_name,
                queue=cpu_queue,
                ncores=ncores,
                mem_gb=mem_gb,
                walltime=walltime,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                dependency=dependency_expr,
            )
            bsub_command = _command(["bsub", *bsub_args, "bash", script_path])
            jobid = f"<{stage}_jobid:{safe_unit}>"
            submit_output = None
            if submit:
                jobid, submit_output = _submit_bsub(bsub_args, script_path, cwd=repo_path)
            actual_status_path = Path(str(status_path).replace("<jobid>", jobid))
            previous_jobid = jobid
            stage_records.append(
                {
                    "stage": stage,
                    "job_id": jobid,
                    "dependency": dependency_expr,
                    "script": str(script_path),
                    "status_json": str(actual_status_path),
                    "stdout": str(stdout_path).replace("%J", jobid),
                    "stderr": str(stderr_path).replace("%J", jobid),
                    "command": command,
                    "bsub_command": bsub_command,
                    "submit_output": submit_output,
                }
            )

        work_items.append(
            {
                "work_unit_id": work_unit_id,
                "clip_id": unit.get("clip_id"),
                "camera_serial": unit.get("camera_serial"),
                "detect_submit_command": unit["commands"]["detect_submit"],
                "detect_job_id": detect_jobid,
                "detect_submit_output": detect_submit_output,
                "detect_artifact_tarball": str(tarball_path),
                "unit_dir": str(unit_dir),
                "stages": stage_records,
            }
        )
        if stage_records:
            finalizer_dependencies.append(str(stage_records[-1]["job_id"]))

    manifest = {
        "status": "submitted" if submit else "dry_run",
        "schema_version": SUBMISSION_SCHEMA,
        "created_at_utc": _utc_now(),
        "plan_path": str(plan_file),
        "plan_schema_version": plan.get("schema_version"),
        "workflow_id": workflow_id,
        "repo": str(repo_path),
        "run_dir": str(bundle_dir),
        "work_unit_count": len(work_items),
        "submit": bool(submit),
        "allow_multiple": bool(allow_multiple),
        "allow_existing_outputs": bool(allow_existing_outputs),
        "target_preflight": target_preflight,
        "work_items": work_items,
    }
    manifest_path = bundle_dir / "submission_manifest.json"
    finalizer_record = None
    if include_finalizer:
        finalizer_dir = bundle_dir / "finalize_recording_collection"
        finalizer_scripts = finalizer_dir / "scripts"
        finalizer_logs = finalizer_dir / "logs"
        finalizer_status = finalizer_dir / "status"
        finalizer_scripts.mkdir(parents=True)
        finalizer_logs.mkdir(parents=True)
        finalizer_status.mkdir(parents=True)
        dependency_expr = " && ".join(f"done({jobid})" for jobid in finalizer_dependencies)
        finalizer_command = _command(
            [
                "scripts/py",
                "-m",
                "fisheye.utils.finalize_clipped_detect_refine_workflow",
                plan_file,
                "--submission-manifest",
                manifest_path,
                "--apply",
            ]
        )
        finalizer_script = finalizer_scripts / "run_finalize_recording_collection.sh"
        finalizer_status_path = finalizer_status / "finalize_recording_collection.<jobid>.json"
        _write_finalizer_script(
            finalizer_script,
            repo=repo_path,
            run_dir=finalizer_dir,
            command=finalizer_command,
            status_path=finalizer_status_path,
        )
        stdout_path = finalizer_logs / "finalize_recording_collection_%J.out"
        stderr_path = finalizer_logs / "finalize_recording_collection_%J.err"
        bsub_args = _bsub_args(
            job_name=_safe_component(f"finalize_{workflow_component}", fallback="finalize_recording_collection"),
            queue=cpu_queue,
            ncores=finalizer_ncores,
            mem_gb=finalizer_mem_gb,
            walltime=finalizer_walltime,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            dependency=dependency_expr,
        )
        bsub_command = _command(["bsub", *bsub_args, "bash", finalizer_script])
        finalizer_jobid = "<finalize_recording_collection_jobid>"
        finalizer_submit_output = None
        finalizer_record = {
            "stage": "finalize_recording_collection",
            "job_id": finalizer_jobid,
            "dependency": dependency_expr,
            "script": str(finalizer_script),
            "status_json": str(finalizer_status_path).replace("<jobid>", finalizer_jobid),
            "stdout": str(stdout_path).replace("%J", finalizer_jobid),
            "stderr": str(stderr_path).replace("%J", finalizer_jobid),
            "command": finalizer_command,
            "bsub_command": bsub_command,
            "submit_output": finalizer_submit_output,
        }
        manifest["finalizer"] = finalizer_record
        _write_json(manifest_path, manifest)
        if submit:
            finalizer_jobid, finalizer_submit_output = _submit_bsub(
                bsub_args,
                finalizer_script,
                cwd=repo_path,
            )
            finalizer_record.update(
                {
                    "job_id": finalizer_jobid,
                    "status_json": str(finalizer_status_path).replace("<jobid>", finalizer_jobid),
                    "stdout": str(stdout_path).replace("%J", finalizer_jobid),
                    "stderr": str(stderr_path).replace("%J", finalizer_jobid),
                    "submit_output": finalizer_submit_output,
                }
            )
            manifest["finalizer"] = finalizer_record

    if finalizer_record is None:
        manifest["finalizer"] = None
    _write_json(manifest_path, manifest)
    manifest["submission_manifest"] = str(manifest_path)
    _write_json(manifest_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit or dry-run a clipped detect/refine plan as LSF dependency chains."
    )
    parser.add_argument("plan_json", type=Path, help="Plan JSON from plan_clipped_detect_refine_workflow")
    parser.add_argument("--repo", type=Path, default=Path("."), help="Palette repo path (default: cwd)")
    parser.add_argument("--run-dir", type=Path, default=None, help="Submission bundle directory")
    parser.add_argument("--work-unit-id", action="append", default=None, help="Submit only selected work unit id(s)")
    parser.add_argument("--clip-id", action="append", default=None, help="Submit only selected clip id(s)")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected work units")
    parser.add_argument("--cpu-queue", default="short")
    parser.add_argument("--import-ncores", type=int, default=2)
    parser.add_argument("--import-mem-gb", type=int, default=16)
    parser.add_argument("--import-walltime", default="1:00")
    parser.add_argument("--validate-ncores", type=int, default=2)
    parser.add_argument("--validate-mem-gb", type=int, default=16)
    parser.add_argument("--validate-walltime", default="1:00")
    parser.add_argument("--quality-ncores", type=int, default=2)
    parser.add_argument("--quality-mem-gb", type=int, default=16)
    parser.add_argument("--quality-walltime", default="1:00")
    parser.add_argument("--refine-ncores", type=int, default=4)
    parser.add_argument("--refine-mem-gb", type=int, default=32)
    parser.add_argument("--refine-walltime", default="1:00")
    parser.add_argument("--finalizer-ncores", type=int, default=2)
    parser.add_argument("--finalizer-mem-gb", type=int, default=16)
    parser.add_argument("--finalizer-walltime", default="1:00")
    parser.add_argument("--no-finalizer", action="store_true", help="Do not create/submit the final collection job")
    parser.add_argument(
        "--allow-existing-outputs",
        action="store_true",
        help="Allow submission even when planned target groups or final collection already exist.",
    )
    parser.add_argument("--submit", action="store_true", help="Actually call bsub; default is dry-run only")
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Allow --submit for more than one work unit. Without this, submit mode is limited to one unit.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional copy of submission manifest")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_submission_bundle(
            args.plan_json,
            repo=args.repo,
            run_dir=args.run_dir,
            work_unit_ids=args.work_unit_id,
            clip_ids=args.clip_id,
            limit=args.limit,
            cpu_queue=args.cpu_queue,
            import_ncores=args.import_ncores,
            import_mem_gb=args.import_mem_gb,
            import_walltime=args.import_walltime,
            validate_ncores=args.validate_ncores,
            validate_mem_gb=args.validate_mem_gb,
            validate_walltime=args.validate_walltime,
            quality_ncores=args.quality_ncores,
            quality_mem_gb=args.quality_mem_gb,
            quality_walltime=args.quality_walltime,
            refine_ncores=args.refine_ncores,
            refine_mem_gb=args.refine_mem_gb,
            refine_walltime=args.refine_walltime,
            finalizer_ncores=args.finalizer_ncores,
            finalizer_mem_gb=args.finalizer_mem_gb,
            finalizer_walltime=args.finalizer_walltime,
            include_finalizer=not args.no_finalizer,
            allow_existing_outputs=args.allow_existing_outputs,
            submit=args.submit,
            allow_multiple=args.allow_multiple,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
