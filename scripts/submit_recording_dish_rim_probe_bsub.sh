#!/usr/bin/env bash
set -euo pipefail
umask 0002

VIDEO=""
SUMMARY=""
PROBE_ID=""
ACQUISITION_OBSERVATION=""
OUTPUT_ROOT="/groups/johnson/johnsonlab/jeremy/diagnostics/recording_dish_rim_probes"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
QUEUE="gpu_l4"
NCORES=8
MEM_GB=32
WALLTIME="1:00"
GPU_RESOURCE="num=1:mode=shared:j_exclusive=no"
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_recording_dish_rim_probe_bsub.sh \
  --video PATH --summary PATH --probe-id ID [options]

Render or submit one diagnostic-only, three-window dish-rim probe. The job
decodes through PyNvVideoCodec on an LSF GPU worker and writes review PNGs plus
immutable JSON reports. It never opens an analysis Zarr or registry.

Required:
  --video PATH                   Native/full-frame Orange MP4
  --summary PATH                 Matching Orange external summary JSON
  --probe-id ID                  Immutable diagnostic identifier

Options:
  --acquisition-observation P    Reveal-only Orange observation JSON; the blind
                                Palette fit is frozen before this file is read
  --output-root PATH             Diagnostic root on shared storage
  --palette-repo PATH            Clean cluster-visible Palette checkout
  --submit-host HOST             Citrus SSH poller if bsub is unavailable locally
  --queue NAME                   LSF queue (default: gpu_l4)
  --ncores N                     CPU slots (default: 8)
  --mem-gb N                     Approximate total memory request (default: 32)
  --walltime H:MM                Walltime (default: 1:00)
  --gpu-resource SPEC            Raw LSF -gpu spec
  --submit                       Submit; otherwise render only
  -h, --help                     Show this help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --summary) SUMMARY="$2"; shift 2;;
    --probe-id) PROBE_ID="$2"; shift 2;;
    --acquisition-observation) ACQUISITION_OBSERVATION="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --gpu-resource) GPU_RESOURCE="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -f "$VIDEO" ]] || fail "video not found: $VIDEO"
[[ -f "$SUMMARY" ]] || fail "summary not found: $SUMMARY"
[[ -n "$PROBE_ID" ]] || fail "--probe-id is required"
[[ "$PROBE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe --probe-id"
[[ -z "$ACQUISITION_OBSERVATION" || -f "$ACQUISITION_OBSERVATION" ]] || \
  fail "acquisition observation not found: $ACQUISITION_OBSERVATION"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/diagnostics/probe_recording_dish_rim_fit.py" ]] || \
  fail "Palette checkout lacks the dish-rim probe"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || \
  fail "Palette checkout must be clean"

RUN_DIR="${OUTPUT_ROOT}/${PROBE_ID}"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"
JOB_SCRIPT="${RUN_DIR}/run_probe.sh"
[[ ! -e "$RUN_DIR" ]] || fail "refusing existing probe directory: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
MEM_MB_PER_SLOT=$(( (MEM_GB * 1024 + NCORES - 1) / NCORES ))
q_repo="$(printf '%q' "$PALETTE_REPO")"
q_video="$(printf '%q' "$VIDEO")"
q_summary="$(printf '%q' "$SUMMARY")"
q_observation="$(printf '%q' "$ACQUISITION_OBSERVATION")"
q_artifacts="$(printf '%q' "$ARTIFACT_DIR")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
VIDEO=${q_video}
SUMMARY=${q_summary}
ACQUISITION_OBSERVATION=${q_observation}
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
export PYTHONPYCACHEPREFIX="\${TMPDIR:-/tmp}/palette-dish-rim-probe-pycache-\${LSB_JOBID}"
export OMP_NUM_THREADS=${NCORES}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cmd=(
  scripts/py -m fisheye.diagnostics.probe_recording_dish_rim_fit
  --video "\${VIDEO}"
  --summary "\${SUMMARY}"
  --output-dir "\${ARTIFACT_DIR}"
  --gpu-id 0
)
if [[ -n "\${ACQUISITION_OBSERVATION}" ]]; then
  cmd+=(--acquisition-observation "\${ACQUISITION_OBSERVATION}")
fi
printf 'probe_command='; printf '%q ' "\${cmd[@]}"; printf '\n'
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
  printf 'video=%s\n' "\${VIDEO}"
  printf 'artifact_dir=%s\n' "\${ARTIFACT_DIR}"
  printf 'fit_report=%s\n' "\${ARTIFACT_DIR}/fit_report.json"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
rm -rf -- "\${PYTHONPYCACHEPREFIX}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "dish_rim_probe_${PROBE_ID}"
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
printf 'probe_id=%s\n' "$PROBE_ID"
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
    submit_mode="ssh_bsub"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
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
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
  printf 'status_file=%s\n' "$STATUS_FILE"
fi
