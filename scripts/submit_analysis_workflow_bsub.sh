#!/usr/bin/env bash
set -euo pipefail
umask 0002

ZARR_PATH=""
EXECUTION_ID=""
CONFIG=""
EXPORT_ROOT=""
SCRATCH_ROOT=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
QUEUE=""
NCORES=4
MEM_GB=32
WALLTIME="24:00"
SUBMIT=0
SUBMIT_REGISTRY_FINALIZER=1
REGISTRY_FINALIZER_MEM_GB=8
REGISTRY_FINALIZER_WALLTIME="1:00"
KINEMATICS_SAMPLE_RATE_HZ=""
ACTIVITY_SPATIAL_BIN_SIZE_S=""
TARGETS=()
STAGE_RUNS=()
OUTPUT_RUNS=()
EXPORT_RUNS=()
FORCE_STAGES=()

usage() {
  cat <<'USAGE'
Usage: submit_analysis_workflow_bsub.sh --zarr PATH --execution-id ID --target NODE [options]

Render or submit one fail-closed LSF job for selected analysis-workflow DAG
nodes. The cluster job verifies the exact Palette commit, executes nodes
serially in dependency order, and verifies each run complete before advancing.

Required:
  --zarr PATH                   Palette analysis Zarr on shared storage
  --execution-id ID            Safe immutable execution identifier
  --target NODE                Executable analysis or implemented export target; repeatable

Options:
  --stage-run STAGE=RUN        Pin an existing dependency run; repeatable
  --output-run STAGE=RUN       Override a generated output run; repeatable
  --export-run NODE=RUN        Override an immutable export run; repeatable
  --export-root PATH           Immutable Parquet publication root for export targets
  --scratch-root PATH          Node-local export scratch override; defaults below TMPDIR
  --force-stage STAGE          Recompute an otherwise reusable stage; repeatable
  --config PATH                Shared custom workflow profile; packaged core profile by default
  --kinematics-sample-rate-hz HZ
  --activity-spatial-bin-size-s S
  --palette-repo PATH          Cluster-visible Palette checkout
  --registry PATH              Shared Palette registry
  --no-registry-finalizer      Leave receipt unapplied; workers still never write SQLite
  --registry-finalizer-mem-gb N
  --registry-finalizer-walltime H:MM
  --submit-host HOST           SSH poller used when bsub is unavailable locally
  --log-dir PATH               Submission root; defaults beside the recording
  --queue NAME                 LSF queue; default is the cluster default
  --ncores N                   CPU slots and Dask workers (default: 4)
  --mem-gb N                   LSF memory request per slot (default: 32)
  --walltime H:MM              LSF walltime (default: 24:00)
  --submit                     Render and submit; default reserves a render-only directory
  -h, --help                   Show this help

The exact kinematics_samples, activity_spatial_summaries, eye_traces, and
tail_traces exports are implemented and remain opt-in. Other export targets
fail closed until their immutable streaming adapters exist. Track, bout,
subject-shape, tail, and eye kinematics use dedicated node-local, sharded,
atomic-publication materializers.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --execution-id) EXECUTION_ID="$2"; shift 2;;
    --target) TARGETS+=("$2"); shift 2;;
    --stage-run) STAGE_RUNS+=("$2"); shift 2;;
    --output-run) OUTPUT_RUNS+=("$2"); shift 2;;
    --export-run) EXPORT_RUNS+=("$2"); shift 2;;
    --export-root) EXPORT_ROOT="$2"; shift 2;;
    --scratch-root) SCRATCH_ROOT="$2"; shift 2;;
    --force-stage) FORCE_STAGES+=("$2"); shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --kinematics-sample-rate-hz) KINEMATICS_SAMPLE_RATE_HZ="$2"; shift 2;;
    --activity-spatial-bin-size-s) ACTIVITY_SPATIAL_BIN_SIZE_S="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --no-registry-finalizer) SUBMIT_REGISTRY_FINALIZER=0; shift;;
    --registry-finalizer-mem-gb) REGISTRY_FINALIZER_MEM_GB="$2"; shift 2;;
    --registry-finalizer-walltime) REGISTRY_FINALIZER_WALLTIME="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "Unknown argument: $1";;
  esac
done

[[ -n "$ZARR_PATH" ]] || fail "--zarr is required"
[[ -n "$EXECUTION_ID" ]] || fail "--execution-id is required"
(( ${#TARGETS[@]} > 0 )) || fail "at least one --target is required"
[[ "$EXECUTION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  fail "unsafe --execution-id: $EXECUTION_ID"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be a positive integer"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be a positive integer"
[[ "$REGISTRY_FINALIZER_MEM_GB" =~ ^[1-9][0-9]*$ ]] || \
  fail "--registry-finalizer-mem-gb must be a positive integer"
[[ -z "$SCRATCH_ROOT" || -n "$EXPORT_ROOT" ]] || \
  fail "--scratch-root requires --export-root"
(( ${#EXPORT_RUNS[@]} == 0 )) || [[ -n "$EXPORT_ROOT" ]] || \
  fail "--export-run requires --export-root"
[[ -z "$EXPORT_ROOT" || "$EXPORT_ROOT" == /* ]] || \
  fail "--export-root must be an absolute cluster-visible path"
[[ -z "$SCRATCH_ROOT" || "$SCRATCH_ROOT" == /* ]] || \
  fail "--scratch-root must be an absolute node-local path"
for target in "${TARGETS[@]}"; do
  case "$target" in
    kinematics_samples|activity_spatial_summaries|eye_traces|tail_traces)
      [[ -n "$EXPORT_ROOT" ]] || fail "the $target target requires --export-root"
      ;;
  esac
done
[[ -f "$ZARR_PATH/zarr.json" ]] || fail "analysis Zarr metadata not found: $ZARR_PATH"
[[ -e "$PALETTE_REPO/.git" ]] || fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/utils/execute_analysis_workflow.py" ]] || \
  fail "Palette checkout does not contain the analysis-workflow executor"
[[ -f "$PALETTE_REPO/src/fisheye/analysis_workflows/registry_finalize.py" ]] || \
  fail "Palette checkout does not contain the analysis registry finalizer"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || \
  fail "Palette checkout must be clean before rendering an execution job: $PALETTE_REPO"
if [[ -n "$CONFIG" ]]; then
  [[ -f "$CONFIG" ]] || fail "workflow config not found: $CONFIG"
fi

ZARR_PARENT="$(dirname -- "$ZARR_PATH")"
RECORDING_DIR="$(dirname -- "$ZARR_PARENT")"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${RECORDING_DIR}/.processing_logs/analysis_workflows"
fi
SCRATCH_ROOT_REQUEST_DISPLAY="not_requested"
if [[ -n "$EXPORT_ROOT" ]]; then
  SCRATCH_ROOT_REQUEST_DISPLAY="${SCRATCH_ROOT:-<lsf-tmpdir-default>}"
fi
SAFE_ZARR_NAME="$(basename -- "$ZARR_PATH" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${LOG_DIR}/${EXECUTION_ID}_${SAFE_ZARR_NAME}"
[[ ! -e "$RUN_DIR" ]] || fail "execution directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_analysis_workflow.sh"
REPORT_PATH="${RUN_DIR}/execution_report.json"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"
RUNTIME_ENVIRONMENT_FILE="${RUN_DIR}/runtime_environment.txt"
RESOURCE_SUMMARY_FILE="${RUN_DIR}/resource_telemetry_summary.json"
RESOURCE_SAMPLES_FILE="${RUN_DIR}/resource_telemetry_samples.jsonl"
RESOURCE_STDOUT_FILE="${RUN_DIR}/workflow_stdout.log"
REGISTRY_FINALIZER_SCRIPT="${RUN_DIR}/run_registry_finalizer.sh"
REGISTRY_FINALIZER_REPORT="${RUN_DIR}/registry_finalizer.json"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_zarr="$(printf '%q' "$ZARR_PATH")"
q_execution_id="$(printf '%q' "$EXECUTION_ID")"
q_expected_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_report="$(printf '%q' "$REPORT_PATH")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_runtime_environment="$(printf '%q' "$RUNTIME_ENVIRONMENT_FILE")"
q_resource_summary="$(printf '%q' "$RESOURCE_SUMMARY_FILE")"
q_resource_samples="$(printf '%q' "$RESOURCE_SAMPLES_FILE")"
q_resource_stdout="$(printf '%q' "$RESOURCE_STDOUT_FILE")"
q_registry="$(printf '%q' "$REGISTRY")"
q_registry_finalizer_report="$(printf '%q' "$REGISTRY_FINALIZER_REPORT")"
q_requested_queue="$(printf '%q' "${QUEUE:-<cluster-default>}")"
q_export_root="$(printf '%q' "$EXPORT_ROOT")"
q_scratch_root="$(printf '%q' "$SCRATCH_ROOT")"
quote_array() {
  local rendered="" value quoted
  for value in "$@"; do
    printf -v quoted '%q' "$value"
    rendered+="${quoted} "
  done
  printf '%s' "$rendered"
}
q_targets="$(quote_array "${TARGETS[@]}")"
q_stage_runs="$(quote_array "${STAGE_RUNS[@]}")"
q_output_runs="$(quote_array "${OUTPUT_RUNS[@]}")"
q_export_runs="$(quote_array "${EXPORT_RUNS[@]}")"
q_force_stages="$(quote_array "${FORCE_STAGES[@]}")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
ZARR_PATH=${q_zarr}
EXECUTION_ID=${q_execution_id}
EXPECTED_COMMIT=${q_expected_commit}
REPORT_PATH=${q_report}
STATUS_FILE=${q_status}
RUNTIME_ENVIRONMENT_FILE=${q_runtime_environment}
RESOURCE_SUMMARY_FILE=${q_resource_summary}
RESOURCE_SAMPLES_FILE=${q_resource_samples}
RESOURCE_STDOUT_FILE=${q_resource_stdout}
REQUESTED_QUEUE=${q_requested_queue}
NCORES=${NCORES}
REQUESTED_MEM_GB_PER_SLOT=${MEM_GB}
REQUESTED_WALLTIME=${WALLTIME}
EXPORT_ROOT=${q_export_root}
REQUESTED_SCRATCH_ROOT=${q_scratch_root}
TARGETS=(${q_targets})
STAGE_RUNS=(${q_stage_runs})
OUTPUT_RUNS=(${q_output_runs})
EXPORT_RUNS=(${q_export_runs})
FORCE_STAGES=(${q_force_stages})

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi
if [[ -n "\$(git status --porcelain)" ]]; then
  printf 'Refusing dirty Palette checkout on the execution host.\n' >&2
  exit 2
fi

# The palette-py311 environment may contain an editable installation from a
# different checkout.  Pin imports to the same cluster-visible source tree
# whose Git commit was verified above, then prove that the imported package
# actually resolved there before doing any scientific work.
export PYTHONPATH="\${PALETTE_REPO}/src:\${PALETTE_REPO}\${PYTHONPATH:+:\${PYTHONPATH}}"
EXPECTED_FISHEYE_DIR="\$(cd "\${PALETTE_REPO}/src/fisheye" && pwd -P)"
ACTUAL_FISHEYE_FILE="\$(scripts/py -c '
from pathlib import Path
import fisheye
print(Path(fisheye.__file__).resolve())
')"
case "\${ACTUAL_FISHEYE_FILE}" in
  "\${EXPECTED_FISHEYE_DIR}"/*) ;;
  *)
    printf 'Palette source mismatch: expected fisheye below %s, imported %s\n' \
      "\${EXPECTED_FISHEYE_DIR}" "\${ACTUAL_FISHEYE_FILE}" >&2
    exit 2
    ;;
esac
[[ -n "\${LSB_JOBID:-}" ]] || {
  printf 'Refusing analysis execution outside an LSF allocation.\n' >&2
  exit 2
}

SCRATCH_ROOT=""
SCRATCH_ROOT_SOURCE="not_requested"
if [[ -n "\${EXPORT_ROOT}" ]]; then
  if [[ -n "\${REQUESTED_SCRATCH_ROOT}" ]]; then
    SCRATCH_ROOT="\${REQUESTED_SCRATCH_ROOT}"
    SCRATCH_ROOT_SOURCE="explicit"
  else
    TASK_SCRATCH_BASE="\${TMPDIR:-/tmp}"
    SCRATCH_ROOT="\${TASK_SCRATCH_BASE%/}/palette_analysis_exports/\${EXECUTION_ID}"
    SCRATCH_ROOT_SOURCE="lsf_tmpdir_default"
  fi
  mkdir -p "\${SCRATCH_ROOT}"
fi

EXECUTION_HOST="\$(hostname)"
EXECUTION_HOST_FQDN="\$(hostname -f 2>/dev/null || hostname)"
EFFECTIVE_QUEUE="\${LSB_QUEUE:-unknown}"
LSF_EXECUTION_HOSTS="\${LSB_HOSTS:-\${EXECUTION_HOST}}"
ALLOCATED_SLOTS="\${LSB_DJOB_NUMPROC:-\${NCORES}}"
CPU_MODEL=""
if command -v lscpu >/dev/null 2>&1; then
  CPU_MODEL="\$(LC_ALL=C lscpu 2>/dev/null | awk -F: '
    /^Model name:/ {
      value=\$2
      sub(/^[[:space:]]+/, "", value)
      print value
      exit
    }
  ')"
fi
if [[ -z "\${CPU_MODEL}" && -r /proc/cpuinfo ]]; then
  CPU_MODEL="\$(awk -F: '
    /^(model name|Hardware)[[:space:]]*:/ {
      value=\$2
      sub(/^[[:space:]]+/, "", value)
      print value
      exit
    }
  ' /proc/cpuinfo)"
fi
CPU_MODEL="\${CPU_MODEL:-unknown}"
CPU_LOGICAL_COUNT="\$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
CPU_LOGICAL_COUNT="\${CPU_LOGICAL_COUNT:-unknown}"

runtime_environment_tmp="\${RUNTIME_ENVIRONMENT_FILE}.tmp.\$\$"
{
  printf 'schema_id=palette.analysis_workflow_runtime_environment.v1\n'
  printf 'captured_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'requested_queue=%s\n' "\${REQUESTED_QUEUE}"
  printf 'effective_queue=%s\n' "\${EFFECTIVE_QUEUE}"
  printf 'execution_host=%s\n' "\${EXECUTION_HOST}"
  printf 'execution_host_fqdn=%s\n' "\${EXECUTION_HOST_FQDN}"
  printf 'lsf_execution_hosts=%s\n' "\${LSF_EXECUTION_HOSTS}"
  printf 'requested_ncores=%s\n' "\${NCORES}"
  printf 'allocated_slots=%s\n' "\${ALLOCATED_SLOTS}"
  printf 'requested_mem_gb_per_slot=%s\n' "\${REQUESTED_MEM_GB_PER_SLOT}"
  printf 'requested_walltime=%s\n' "\${REQUESTED_WALLTIME}"
  printf 'cpu_model=%s\n' "\${CPU_MODEL}"
  printf 'cpu_architecture=%s\n' "\$(uname -m)"
  printf 'cpu_logical_count=%s\n' "\${CPU_LOGICAL_COUNT}"
  printf 'kernel_release=%s\n' "\$(uname -r)"
  printf 'fisheye_source_file=%s\n' "\${ACTUAL_FISHEYE_FILE}"
  printf 'export_root=%s\n' "\${EXPORT_ROOT:-not_requested}"
  printf 'scratch_root=%s\n' "\${SCRATCH_ROOT:-not_requested}"
  printf 'scratch_root_source=%s\n' "\${SCRATCH_ROOT_SOURCE}"
} >"\${runtime_environment_tmp}"
mv "\${runtime_environment_tmp}" "\${RUNTIME_ENVIRONMENT_FILE}"

# Multi-process analysis stages own CPU parallelism at the process level.
# Keep BLAS/OpenMP/OpenCV-adjacent native pools from multiplying that count.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Registry mutations are owned by the dependent one-slot finalizer.  Every
# stage subprocess inherits this fail-closed guard.
export PALETTE_DISABLE_REGISTRY_WRITES=1

cmd=(
  scripts/py -m fisheye.utils.execute_analysis_workflow
  "\${ZARR_PATH}"
  --execution-id "\${EXECUTION_ID}"
  --num-workers "\${NCORES}"
  --report "\${REPORT_PATH}"
  --apply
)
for value in "\${TARGETS[@]}"; do cmd+=(--target "\${value}"); done
for value in "\${STAGE_RUNS[@]}"; do cmd+=(--stage-run "\${value}"); done
for value in "\${OUTPUT_RUNS[@]}"; do cmd+=(--output-run "\${value}"); done
for value in "\${EXPORT_RUNS[@]}"; do cmd+=(--export-run "\${value}"); done
for value in "\${FORCE_STAGES[@]}"; do cmd+=(--force-stage "\${value}"); done
if [[ -n "\${EXPORT_ROOT}" ]]; then
  cmd+=(--export-root "\${EXPORT_ROOT}" --scratch-root "\${SCRATCH_ROOT}")
fi
JOBSCRIPT

if [[ -n "$CONFIG" ]]; then
  printf 'cmd+=(--config %q)\n' "$CONFIG" >>"$JOB_SCRIPT"
fi
if [[ -n "$KINEMATICS_SAMPLE_RATE_HZ" ]]; then
  printf 'cmd+=(--kinematics-sample-rate-hz %q)\n' \
    "$KINEMATICS_SAMPLE_RATE_HZ" >>"$JOB_SCRIPT"
fi
if [[ -n "$ACTIVITY_SPATIAL_BIN_SIZE_S" ]]; then
  printf 'cmd+=(--activity-spatial-bin-size-s %q)\n' \
    "$ACTIVITY_SPATIAL_BIN_SIZE_S" >>"$JOB_SCRIPT"
fi

cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'

printf 'workflow_command='; printf '%q ' "${cmd[@]}"; printf '\n'
telemetry_cmd=(
  scripts/py -m fisheye.diagnostics.run_with_resource_telemetry
  --summary-json "${RESOURCE_SUMMARY_FILE}"
  --samples-jsonl "${RESOURCE_SAMPLES_FILE}"
  --stdout-log "${RESOURCE_STDOUT_FILE}"
  --requested-workers "${NCORES}"
  --allocated-slots "${ALLOCATED_SLOTS}"
  --sample-interval-seconds 2.0
  --
  "${cmd[@]}"
)
printf 'telemetry_command='; printf '%q ' "${telemetry_cmd[@]}"; printf '\n'
set +e
"${telemetry_cmd[@]}"
payload_rc=$?
set -e

status_tmp="${STATUS_FILE}.tmp.$$"
{
  printf 'status=%s\n' "$([[ "$payload_rc" == "0" ]] && printf complete || printf failed)"
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "${EXECUTION_HOST}"
  printf 'host_fqdn=%s\n' "${EXECUTION_HOST_FQDN}"
  printf 'requested_queue=%s\n' "${REQUESTED_QUEUE}"
  printf 'effective_queue=%s\n' "${EFFECTIVE_QUEUE}"
  printf 'cpu_model=%s\n' "${CPU_MODEL}"
  printf 'allocated_slots=%s\n' "${ALLOCATED_SLOTS}"
  printf 'job_id=%s\n' "${LSB_JOBID}"
  printf 'palette_commit=%s\n' "${ACTUAL_COMMIT}"
  printf 'zarr_path=%s\n' "${ZARR_PATH}"
  printf 'execution_id=%s\n' "${EXECUTION_ID}"
  printf 'export_root=%s\n' "${EXPORT_ROOT:-not_requested}"
  printf 'scratch_root=%s\n' "${SCRATCH_ROOT:-not_requested}"
  printf 'scratch_root_source=%s\n' "${SCRATCH_ROOT_SOURCE}"
  printf 'payload_returncode=%s\n' "${payload_rc}"
  printf 'execution_report=%s\n' "${REPORT_PATH}"
  printf 'runtime_environment=%s\n' "${RUNTIME_ENVIRONMENT_FILE}"
  printf 'resource_telemetry_summary=%s\n' "${RESOURCE_SUMMARY_FILE}"
  printf 'resource_telemetry_samples=%s\n' "${RESOURCE_SAMPLES_FILE}"
  printf 'workflow_stdout=%s\n' "${RESOURCE_STDOUT_FILE}"
} >"${status_tmp}"
mv "${status_tmp}" "${STATUS_FILE}"
exit "${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

cat >"$REGISTRY_FINALIZER_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
EXPECTED_COMMIT=${q_expected_commit}
REPORT_PATH=${q_report}
REGISTRY=${q_registry}
FINALIZER_REPORT=${q_registry_finalizer_report}

cd "\${PALETTE_REPO}"
[[ "\$(git rev-parse HEAD)" == "\${EXPECTED_COMMIT}" ]] || {
  printf 'Palette commit mismatch in registry finalizer.\n' >&2
  exit 2
}
[[ -z "\$(git status --porcelain)" ]] || {
  printf 'Refusing dirty Palette checkout in registry finalizer.\n' >&2
  exit 2
}
unset PALETTE_DISABLE_REGISTRY_WRITES
export PYTHONPATH="\${PALETTE_REPO}/src:\${PALETTE_REPO}\${PYTHONPATH:+:\${PYTHONPATH}}"
scripts/py -m fisheye.analysis_workflows.registry_finalize \
  --execution-report "\${REPORT_PATH}" \
  --registry "\${REGISTRY}" \
  --apply \
  --output-json "\${FINALIZER_REPORT}"
JOBSCRIPT
chmod +x "$REGISTRY_FINALIZER_SCRIPT"

BSUB_ARGS=(
  -J "analysis_workflow_${EXECUTION_ID}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "span[hosts=1]"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'zarr_path=%s\n' "$ZARR_PATH"
printf 'execution_id=%s\n' "$EXECUTION_ID"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'execution_report=%s\n' "$REPORT_PATH"
printf 'runtime_environment=%s\n' "$RUNTIME_ENVIRONMENT_FILE"
printf 'resource_telemetry_summary=%s\n' "$RESOURCE_SUMMARY_FILE"
printf 'resource_telemetry_samples=%s\n' "$RESOURCE_SAMPLES_FILE"
printf 'workflow_stdout=%s\n' "$RESOURCE_STDOUT_FILE"
printf 'export_root=%s\n' "${EXPORT_ROOT:-not_requested}"
printf 'scratch_root_request=%s\n' "$SCRATCH_ROOT_REQUEST_DISPLAY"
printf 'registry_write_mode=deferred_to_serial_finalizer\n'
printf 'registry=%s\n' "$REGISTRY"
printf 'registry_finalizer=%s\n' \
  "$([[ "$SUBMIT_REGISTRY_FINALIZER" == "1" ]] && printf enabled || printf disabled)"
printf 'registry_finalizer_script=%s\n' "$REGISTRY_FINALIZER_SCRIPT"
printf 'registry_finalizer_report=%s\n' "$REGISTRY_FINALIZER_REPORT"
printf 'requested_queue=%s\n' "${QUEUE:-<cluster-default>}"
printf 'requested_resources=ncores:%s mem_gb_per_slot:%s walltime:%s\n' \
  "$NCORES" "$MEM_GB" "$WALLTIME"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub is unavailable and --submit-host is empty"
    submit_mode="ssh_bsub"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse an LSF job ID"
  submission_tmp="${SUBMISSION_FILE}.tmp.$$"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'requested_queue=%s\n' "${QUEUE:-<cluster-default>}"
    printf 'requested_ncores=%s\n' "$NCORES"
    printf 'requested_mem_gb_per_slot=%s\n' "$MEM_GB"
    printf 'requested_walltime=%s\n' "$WALLTIME"
    printf 'export_root=%s\n' "${EXPORT_ROOT:-not_requested}"
    printf 'scratch_root_request=%s\n' "$SCRATCH_ROOT_REQUEST_DISPLAY"
    printf 'job_id=%s\n' "$job_id"
    printf 'job_script=%s\n' "$JOB_SCRIPT"
    printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
    printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
    printf 'status_file=%s\n' "$STATUS_FILE"
    printf 'execution_report=%s\n' "$REPORT_PATH"
    printf 'runtime_environment=%s\n' "$RUNTIME_ENVIRONMENT_FILE"
    printf 'resource_telemetry_summary=%s\n' "$RESOURCE_SUMMARY_FILE"
    printf 'resource_telemetry_samples=%s\n' "$RESOURCE_SAMPLES_FILE"
    printf 'workflow_stdout=%s\n' "$RESOURCE_STDOUT_FILE"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$submission_tmp"
  mv "$submission_tmp" "$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
  printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
  printf 'status_file=%s\n' "$STATUS_FILE"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"

  if [[ "$SUBMIT_REGISTRY_FINALIZER" == "1" ]]; then
    FINALIZER_BSUB_ARGS=(
      -J "analysis_registry_${EXECUTION_ID}"
      -n 1
      -W "$REGISTRY_FINALIZER_WALLTIME"
      -R "span[hosts=1]"
      -R "rusage[mem=${REGISTRY_FINALIZER_MEM_GB}G]"
      -w "done(${job_id})"
      -oo "${RUN_DIR}/registry_finalizer_%J.out"
      -eo "${RUN_DIR}/registry_finalizer_%J.err"
    )
    if [[ -n "$QUEUE" ]]; then FINALIZER_BSUB_ARGS+=(-q "$QUEUE"); fi
    FINALIZER_BSUB_COMMAND=(
      bsub "${FINALIZER_BSUB_ARGS[@]}" bash "$REGISTRY_FINALIZER_SCRIPT"
    )
    if command -v bsub >/dev/null 2>&1; then
      finalizer_output="$("${FINALIZER_BSUB_COMMAND[@]}")"
    else
      printf -v remote_finalizer_command '%q ' "${FINALIZER_BSUB_COMMAND[@]}"
      finalizer_output="$(ssh "$SUBMIT_HOST" "$remote_finalizer_command")"
    fi
    printf '%s\n' "$finalizer_output"
    finalizer_job_id="$(printf '%s\n' "$finalizer_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
    [[ -n "$finalizer_job_id" ]] || fail "could not parse registry finalizer job ID"
    {
      printf 'registry_write_mode=deferred_to_serial_finalizer\n'
      printf 'registry=%s\n' "$REGISTRY"
      printf 'registry_finalizer_job_id=%s\n' "$finalizer_job_id"
      printf 'registry_finalizer_dependency=done(%s)\n' "$job_id"
      printf 'registry_finalizer_report=%s\n' "$REGISTRY_FINALIZER_REPORT"
    } >>"$SUBMISSION_FILE"
    printf 'registry_finalizer_job_id=%s\n' "$finalizer_job_id"
    printf 'registry_finalizer_report=%s\n' "$REGISTRY_FINALIZER_REPORT"
  fi
fi
