#!/usr/bin/env bash
set -euo pipefail
umask 0002

REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
OUTPUT_ROOT="${PALETTE_ANALYTICS_EXPORT_ROOT:-/groups/johnson/johnsonlab/palette_analytics}"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
COLLECTION_ID=""
COLLECTION_NAME=""
STIMULUS_MODE=""
ZARR_LIST=""
PROFILE=""
OUTPUT=""
ZARR_USE="analysis"
DATASET_STATUS="active"
STORAGE_TIER="shared_groups"
QUEUE=""
MEM_GB=8
WALLTIME="1:00"
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_registry_collection_manifest_bsub.sh [required options] [options]

Render or submit one CPU LSF job that queries normalized stimulus metadata from
the registry, resolves per-recording Zarr run families, writes an immutable
virtual collection manifest, and validates its canonical hash.

Required:
  --collection-id ID          Unique collection identifier
  --collection-name NAME      Human-readable collection name

Select exactly one source:
  --stimulus-mode MODE        Normalized registry stimulus mode, for example CHASER
  --zarr-list PATH            Frozen newline-delimited analysis-Zarr selection

Options:
  --registry PATH             Shared Palette registry
  --output PATH               Manifest destination (default: shared manifests root)
  --output-root PATH          Shared analytics root
  --profile ID                Builder profile (CHASER defaults to chaser)
  --zarr-use VALUE            Registry zarr_use filter (default: analysis)
  --dataset-status VALUE      Registry status filter (default: active)
  --storage-tier VALUE        Manifest locator tier (default: shared_groups)
  --palette-repo PATH         Cluster-visible Palette checkout
  --submit-host HOST          SSH submission host if bsub is unavailable locally
  --queue NAME
  --mem-gb N                  Default: 8
  --walltime H:MM             Default: 1:00
  --submit                    Submit; otherwise render only
  -h, --help

The job refuses an existing destination and checks that the shared checkout is
still at the exact commit captured during submission.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collection-id) COLLECTION_ID="$2"; shift 2;;
    --collection-name) COLLECTION_NAME="$2"; shift 2;;
    --stimulus-mode) STIMULUS_MODE="$2"; shift 2;;
    --zarr-list) ZARR_LIST="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --zarr-use) ZARR_USE="$2"; shift 2;;
    --dataset-status) DATASET_STATUS="$2"; shift 2;;
    --storage-tier) STORAGE_TIER="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$COLLECTION_ID" ]] || fail "--collection-id is required"
[[ "$COLLECTION_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe --collection-id: $COLLECTION_ID"
[[ -n "$COLLECTION_NAME" ]] || fail "--collection-name is required"
if [[ -n "$STIMULUS_MODE" && -n "$ZARR_LIST" ]]; then
  fail "select exactly one source: --stimulus-mode or --zarr-list"
fi
if [[ -z "$STIMULUS_MODE" && -z "$ZARR_LIST" ]]; then
  fail "select exactly one source: --stimulus-mode or --zarr-list"
fi
if [[ -n "$STIMULUS_MODE" ]]; then
  [[ "$STIMULUS_MODE" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe --stimulus-mode: $STIMULUS_MODE"
  [[ -f "$REGISTRY" ]] || fail "registry not found: $REGISTRY"
  SOURCE_MODE="registry_stimulus_mode"
else
  [[ -f "$ZARR_LIST" ]] || fail "Zarr list not found: $ZARR_LIST"
  SOURCE_MODE="zarr_list"
fi
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
git -C "$PALETTE_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${OUTPUT_ROOT}/v1/manifests/collections/${COLLECTION_ID}.manifest.json"
fi
RUN_DIR="${OUTPUT_ROOT}/logs/lsf/collection_manifest_${COLLECTION_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
[[ ! -e "$OUTPUT" ]] || fail "manifest already exists: $OUTPUT"
mkdir -p "$RUN_DIR"
if [[ "$SOURCE_MODE" == "zarr_list" ]]; then
  ZARR_LIST_SNAPSHOT="$RUN_DIR/zarr_paths.txt"
  cp "$ZARR_LIST" "$ZARR_LIST_SNAPSHOT"
else
  ZARR_LIST_SNAPSHOT=""
fi

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_collection_manifest.sh"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_registry="$(printf '%q' "$REGISTRY")"
q_output="$(printf '%q' "$OUTPUT")"
q_collection_id="$(printf '%q' "$COLLECTION_ID")"
q_collection_name="$(printf '%q' "$COLLECTION_NAME")"
q_stimulus_mode="$(printf '%q' "$STIMULUS_MODE")"
q_zarr_list="$(printf '%q' "$ZARR_LIST_SNAPSHOT")"
q_source_mode="$(printf '%q' "$SOURCE_MODE")"
q_profile="$(printf '%q' "$PROFILE")"
q_zarr_use="$(printf '%q' "$ZARR_USE")"
q_dataset_status="$(printf '%q' "$DATASET_STATUS")"
q_storage_tier="$(printf '%q' "$STORAGE_TIER")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_status="$(printf '%q' "$STATUS_FILE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
REGISTRY=${q_registry}
OUTPUT=${q_output}
COLLECTION_ID=${q_collection_id}
COLLECTION_NAME=${q_collection_name}
STIMULUS_MODE=${q_stimulus_mode}
ZARR_LIST=${q_zarr_list}
SOURCE_MODE=${q_source_mode}
PROFILE=${q_profile}
ZARR_USE=${q_zarr_use}
DATASET_STATUS=${q_dataset_status}
STORAGE_TIER=${q_storage_tier}
EXPECTED_COMMIT=${q_commit}
STATUS_FILE=${q_status}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi

mkdir -p "\$(dirname -- "\${OUTPUT}")"
cmd=(
  scripts/py -m fisheye.utils.build_virtual_collection_manifest
  --collection-id "\${COLLECTION_ID}"
  --collection-name "\${COLLECTION_NAME}"
  --zarr-use "\${ZARR_USE}"
  --dataset-status "\${DATASET_STATUS}"
  --storage-tier "\${STORAGE_TIER}"
  --output "\${OUTPUT}"
)
if [[ "\${SOURCE_MODE}" == "registry_stimulus_mode" ]]; then
  cmd+=(--registry "\${REGISTRY}" --stimulus-mode "\${STIMULUS_MODE}")
else
  mapfile -t zarr_paths < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "\${ZARR_LIST}")
  if [[ "\${#zarr_paths[@]}" == "0" ]]; then
    printf 'Zarr list contains no paths: %s\n' "\${ZARR_LIST}" >&2
    exit 2
  fi
  cmd+=("\${zarr_paths[@]}")
fi
if [[ -n "\${PROFILE}" ]]; then
  cmd+=(--profile "\${PROFILE}")
fi
printf 'command='; printf '%q ' "\${cmd[@]}"; printf '\n'
"\${cmd[@]}"
scripts/py -m fisheye.utils.virtual_collection_manifest \
  validate "\${OUTPUT}" --check-hash
MANIFEST_SHA256="\$(scripts/py -m fisheye.utils.virtual_collection_manifest hash "\${OUTPUT}")"

{
  printf 'status=complete\n'
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'collection_id=%s\n' "\${COLLECTION_ID}"
  printf 'source_mode=%s\n' "\${SOURCE_MODE}"
  if [[ -n "\${ZARR_LIST}" ]]; then
    printf 'zarr_list=%s\n' "\${ZARR_LIST}"
    printf 'zarr_list_sha256=%s\n' "\$(sha256sum "\${ZARR_LIST}" | awk '{print \$1}')"
  fi
  printf 'manifest=%s\n' "\${OUTPUT}"
  printf 'manifest_sha256=%s\n' "\${MANIFEST_SHA256}"
} >"\${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "collection_manifest_${COLLECTION_ID}"
  -n 1
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'collection_id=%s\n' "$COLLECTION_ID"
printf 'source_mode=%s\n' "$SOURCE_MODE"
printf 'stimulus_mode=%s\n' "$STIMULUS_MODE"
if [[ -n "$ZARR_LIST_SNAPSHOT" ]]; then printf 'zarr_list=%s\n' "$ZARR_LIST_SNAPSHOT"; fi
printf 'manifest=%s\n' "$OUTPUT"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and --submit-host is empty"
    submit_mode="ssh_bsub"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse LSF job ID"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
    printf 'manifest=%s\n' "$OUTPUT"
    printf 'status_file=%s\n' "$STATUS_FILE"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
