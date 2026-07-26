#!/usr/bin/env bash
set -euo pipefail
umask 0002

ZARR=""
VIDEO=""
PALETTE_CANDIDATE_RUN=""
ACQUISITION_CANDIDATE_RUN=""
DETECT_RUN=""
AUDIT_ID=""
OUTPUT_ROOT="/groups/johnson/johnsonlab/jeremy/diagnostics/arena_geometry_detection_gate_audits"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
QUEUE="gpu_l4"
NCORES=4
MEM_GB=16
WALLTIME="1:00"
GPU_RESOURCE="num=1:mode=shared:j_exclusive=no"
MAX_SAMPLES=8
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_arena_geometry_detection_gate_audit_bsub.sh \
  --zarr PATH --video PATH \
  --palette-candidate-run RUN --acquisition-candidate-run RUN \
  --detect-run RUN --audit-id ID [options]

Compare two exact pointerless arena-geometry candidates against one exact raw
detection run and render deterministic disagreement samples through PyNvVC.
The job is diagnostic-only: it does not select a candidate, gate detections,
modify the Zarr, or update the registry.

Options:
  --output-root PATH       Shared diagnostic root
  --palette-repo PATH      Clean cluster-visible Palette checkout
  --submit-host HOST       Citrus SSH poller when bsub is unavailable locally
  --queue NAME             LSF queue (default: gpu_l4)
  --ncores N               CPU slots (default: 4)
  --mem-gb N               Approximate total memory request (default: 16)
  --walltime H:MM          Walltime (default: 1:00)
  --gpu-resource SPEC      Raw LSF -gpu specification
  --max-samples N          Review images per exclusive category (default: 8)
  --submit                 Submit; otherwise render the exact command only
  -h, --help               Show this help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR="$2"; shift 2;;
    --video) VIDEO="$2"; shift 2;;
    --palette-candidate-run) PALETTE_CANDIDATE_RUN="$2"; shift 2;;
    --acquisition-candidate-run) ACQUISITION_CANDIDATE_RUN="$2"; shift 2;;
    --detect-run) DETECT_RUN="$2"; shift 2;;
    --audit-id) AUDIT_ID="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --gpu-resource) GPU_RESOURCE="$2"; shift 2;;
    --max-samples) MAX_SAMPLES="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -d "$ZARR" ]] || fail "analysis Zarr not found: $ZARR"
[[ -f "$VIDEO" ]] || fail "video not found: $VIDEO"
[[ -n "$PALETTE_CANDIDATE_RUN" ]] || fail "--palette-candidate-run is required"
[[ -n "$ACQUISITION_CANDIDATE_RUN" ]] || fail "--acquisition-candidate-run is required"
[[ -n "$DETECT_RUN" ]] || fail "--detect-run is required"
[[ "$AUDIT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe --audit-id"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]] || fail "--max-samples must be positive"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/diagnostics/audit_arena_geometry_detection_gates.py" ]] || \
  fail "Palette checkout lacks the geometry gate audit"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || \
  fail "Palette checkout must be clean"

RUN_DIR="${OUTPUT_ROOT}/${AUDIT_ID}"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"
JOB_SCRIPT="${RUN_DIR}/run_audit.sh"
[[ ! -e "$RUN_DIR" ]] || fail "refusing existing audit directory: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
MEM_MB_PER_SLOT=$(( (MEM_GB * 1024 + NCORES - 1) / NCORES ))
q_repo="$(printf '%q' "$PALETTE_REPO")"
q_zarr="$(printf '%q' "$ZARR")"
q_video="$(printf '%q' "$VIDEO")"
q_palette="$(printf '%q' "$PALETTE_CANDIDATE_RUN")"
q_acquisition="$(printf '%q' "$ACQUISITION_CANDIDATE_RUN")"
q_detect="$(printf '%q' "$DETECT_RUN")"
q_artifacts="$(printf '%q' "$ARTIFACT_DIR")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
ZARR=${q_zarr}
VIDEO=${q_video}
PALETTE_CANDIDATE_RUN=${q_palette}
ACQUISITION_CANDIDATE_RUN=${q_acquisition}
DETECT_RUN=${q_detect}
ARTIFACT_DIR=${q_artifacts}
STATUS_FILE=${q_status}
EXPECTED_COMMIT=${q_expected}

[[ -n "\${LSB_JOBID:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
[[ "\${ACTUAL_COMMIT}" == "\${EXPECTED_COMMIT}" ]] || {
  printf 'Palette commit mismatch: expected=%s actual=%s\n' "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
}
[[ -z "\$(git status --porcelain)" ]] || { printf 'Refusing dirty Palette checkout.\n' >&2; exit 2; }
export PYTHONPYCACHEPREFIX="\${TMPDIR:-/tmp}/palette-geometry-gate-audit-pycache-\${LSB_JOBID}"
export OMP_NUM_THREADS=${NCORES}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cmd=(
  scripts/py -m fisheye.diagnostics.audit_arena_geometry_detection_gates
  --zarr "\${ZARR}"
  --video "\${VIDEO}"
  --palette-candidate-run "\${PALETTE_CANDIDATE_RUN}"
  --acquisition-candidate-run "\${ACQUISITION_CANDIDATE_RUN}"
  --detect-run "\${DETECT_RUN}"
  --output-dir "\${ARTIFACT_DIR}"
  --gpu-id 0
  --max-review-samples-per-category ${MAX_SAMPLES}
)
printf 'audit_command='; printf '%q ' "\${cmd[@]}"; printf '\n'
set +e
"\${cmd[@]}"
payload_rc=\$?
set -e

status_tmp="\${STATUS_FILE}.tmp.\$\$"
{
  if [[ "\${payload_rc}" == "0" ]]; then printf 'status=complete\n'; else printf 'status=failed\n'; fi
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'artifact_dir=%s\n' "\${ARTIFACT_DIR}"
  printf 'audit_report=%s\n' "\${ARTIFACT_DIR}/audit_report.json"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
rm -rf -- "\${PYTHONPYCACHEPREFIX}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "arena_geometry_gate_audit_${AUDIT_ID}"
  -q "$QUEUE"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "span[hosts=1] rusage[mem=${MEM_MB_PER_SLOT}]"
  -gpu "$GPU_RESOURCE"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'audit_id=%s\n' "$AUDIT_ID"
printf 'memory_request_mb_per_slot=%s\n' "$MEM_MB_PER_SLOT"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'artifact_dir=%s\n' "$ARTIFACT_DIR"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and submit host empty"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_mode="ssh_bsub"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse an LSF job ID"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'job_script=%s\n' "$JOB_SCRIPT"
    printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
    printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
    printf 'status_file=%s\n' "$STATUS_FILE"
    printf 'artifact_dir=%s\n' "$ARTIFACT_DIR"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
fi
