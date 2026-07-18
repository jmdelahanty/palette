#!/usr/bin/env bash
set -euo pipefail
umask 0002

COHORT_MANIFEST=""
EXPORT_RUN_ID=""
REPORT_ID=""
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
OUTPUT_ROOT="${PALETTE_ANALYTICS_EXPORT_ROOT:-/groups/johnson/johnsonlab/palette_analytics}"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
QUEUE=""
MEM_GB=12
WALLTIME="2:00"
SUBMIT=0
ALLOW_NONREADY=0
VISUALIZATION_IDS=()
DEPENDENCY_DONE=()

usage() {
  cat <<'USAGE'
Usage: submit_cohort_report_bsub.sh --cohort-manifest PATH --export-run-id ID --report-id ID [options]

Render or submit a cohort report job. The worker selects exactly the recording
IDs frozen in the cohort manifest, builds semantic PNG montages, publishes an
immutable report bound to the analytics export, indexes it serially, and checks
the completed report and files.

Required:
  --cohort-manifest PATH       Frozen palette.frozen_cohort_manifest
  --export-run-id ID           Completed or dependency-produced analytics export
  --report-id ID               Immutable report identifier

Options:
  --visualization-id ID        Semantic visualization; may repeat. Defaults to
                               the three recording-level chaser visualizations.
  --allow-nonready             Render labeled placeholders instead of failing.
  --registry PATH
  --output-root PATH
  --palette-repo PATH
  --submit-host HOST
  --log-dir PATH               Default: <output-root>/logs/lsf
  --dependency-done JOBID      Submit after this LSF job succeeds. May repeat.
  --queue NAME
  --mem-gb N                   Default: 12
  --walltime H:MM              Default: 2:00
  --submit                     Submit; otherwise render only.
  -h, --help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohort-manifest) COHORT_MANIFEST="$2"; shift 2;;
    --export-run-id) EXPORT_RUN_ID="$2"; shift 2;;
    --report-id) REPORT_ID="$2"; shift 2;;
    --visualization-id) VISUALIZATION_IDS+=("$2"); shift 2;;
    --allow-nonready) ALLOW_NONREADY=1; shift;;
    --registry) REGISTRY="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --dependency-done) DEPENDENCY_DONE+=("$2"); shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$COHORT_MANIFEST" ]] || fail "--cohort-manifest is required"
[[ -f "$COHORT_MANIFEST" ]] || fail "cohort manifest not found: $COHORT_MANIFEST"
[[ -n "$EXPORT_RUN_ID" ]] || fail "--export-run-id is required"
[[ "$EXPORT_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe --export-run-id: $EXPORT_RUN_ID"
[[ -n "$REPORT_ID" ]] || fail "--report-id is required"
[[ "$REPORT_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe --report-id: $REPORT_ID"
[[ -f "$REGISTRY" ]] || fail "registry not found: $REGISTRY"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
for dependency_job_id in "${DEPENDENCY_DONE[@]}"; do
  [[ "$dependency_job_id" =~ ^[1-9][0-9]*$ ]] || \
    fail "--dependency-done must be a positive numeric LSF job ID: $dependency_job_id"
done
git -C "$PALETTE_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"

if [[ "${#VISUALIZATION_IDS[@]}" == "0" ]]; then
  VISUALIZATION_IDS=(
    stimulus.chaser.distance_trace
    stimulus.chaser.distance_distribution
    stimulus.chaser.egocentric_bearing
  )
fi

SCRIPT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$SCRIPT_REPO/src" "$SCRIPT_REPO/scripts/py" -m fisheye.cohorts \
  validate "$COHORT_MANIFEST" --check-hash >/dev/null

if [[ -z "$LOG_DIR" ]]; then LOG_DIR="${OUTPUT_ROOT}/logs/lsf"; fi
RUN_DIR="${LOG_DIR}/cohort_report_${EXPORT_RUN_ID}_${REPORT_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_cohort_report.sh"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"
VISUALIZATION_FILE="${RUN_DIR}/visualization_ids.txt"
printf '%s\n' "${VISUALIZATION_IDS[@]}" >"$VISUALIZATION_FILE"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_registry="$(printf '%q' "$REGISTRY")"
q_cohort="$(printf '%q' "$COHORT_MANIFEST")"
q_output_root="$(printf '%q' "$OUTPUT_ROOT")"
q_export_id="$(printf '%q' "$EXPORT_RUN_ID")"
q_report_id="$(printf '%q' "$REPORT_ID")"
q_run_dir="$(printf '%q' "$RUN_DIR")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_visualizations="$(printf '%q' "$VISUALIZATION_FILE")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
REGISTRY=${q_registry}
COHORT_MANIFEST=${q_cohort}
OUTPUT_ROOT=${q_output_root}
EXPORT_RUN_ID=${q_export_id}
REPORT_ID=${q_report_id}
RUN_DIR=${q_run_dir}
STATUS_FILE=${q_status}
VISUALIZATION_FILE=${q_visualizations}
EXPECTED_COMMIT=${q_commit}
ALLOW_NONREADY=${ALLOW_NONREADY}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi
scripts/py -m fisheye.cohorts validate "\${COHORT_MANIFEST}" --check-hash
EXPORT_MANIFEST="\${OUTPUT_ROOT}/v1/manifests/export_run_id=\${EXPORT_RUN_ID}.json"
if [[ ! -f "\${EXPORT_MANIFEST}" ]]; then
  printf 'Analytics export manifest is unavailable after dependencies completed: %s\n' \
    "\${EXPORT_MANIFEST}" >&2
  exit 2
fi

scripts/py - "\${REGISTRY}" "\${COHORT_MANIFEST}" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

registry_path = Path(sys.argv[1]).expanduser().resolve()
cohort_path = Path(sys.argv[2]).expanduser().resolve()
payload = json.loads(cohort_path.read_text(encoding="utf-8"))
connection = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
connection.execute("PRAGMA query_only = ON")
connection.execute("BEGIN")
try:
    for member in payload["members"]:
        rows = connection.execute(
            """
            SELECT dataset_id, recording_id, zarr_path, dataset_status, zarr_use
            FROM dataset_context_current
            WHERE dataset_id = ?
            """,
            (member["dataset_id"],),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                f"Frozen dataset {member['dataset_id']!r} resolves to {len(rows)} current rows"
            )
        row = rows[0]
        expected = (
            str(member["recording_id"]),
            str(Path(member["zarr_path"]).expanduser().resolve(strict=False)),
            str(member["dataset_status"]),
            str(member["zarr_use"]),
        )
        actual = (
            str(row["recording_id"]),
            str(Path(row["zarr_path"]).expanduser().resolve(strict=False)),
            str(row["dataset_status"]),
            str(row["zarr_use"]),
        )
        if actual != expected:
            raise ValueError(
                f"Frozen dataset binding changed for {member['dataset_id']!r}: "
                f"expected={expected!r}, current={actual!r}"
            )
finally:
    connection.close()
PY

mapfile -t recording_ids < <(scripts/py - "\${COHORT_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for member in payload["members"]:
    print(member["recording_id"])
PY
)
if [[ "\${#recording_ids[@]}" == "0" ]]; then
  printf 'Frozen cohort has no recording IDs.\n' >&2
  exit 2
fi
recording_args=()
for recording_id in "\${recording_ids[@]}"; do
  recording_args+=(--recording-id "\${recording_id}")
done
visualization_args=()
while IFS= read -r visualization_id; do
  [[ -n "\${visualization_id}" ]] || continue
  visualization_args+=(--visualization-id "\${visualization_id}")
done <"\${VISUALIZATION_FILE}"

SEMANTIC_DIR="\${RUN_DIR}/semantic_montages"
montage_cmd=(
  scripts/py -m fisheye.reporting montage
  --registry "\${REGISTRY}"
  "\${recording_args[@]}"
  "\${visualization_args[@]}"
  --output-dir "\${SEMANTIC_DIR}"
)
if [[ "\${ALLOW_NONREADY}" == "1" ]]; then montage_cmd+=(--allow-nonready); fi
printf 'montage_command='; printf '%q ' "\${montage_cmd[@]}"; printf '\n'
"\${montage_cmd[@]}" >"\${RUN_DIR}/montage_result.json"

scripts/py -m fisheye.reporting publish-montage-report \
  --registry "\${REGISTRY}" \
  --semantic-manifest "\${SEMANTIC_DIR}/semantic_montage_manifest.json" \
  --analytics-export-run-id "\${EXPORT_RUN_ID}" \
  --report-id "\${REPORT_ID}" \
  --index-registry \
  >"\${RUN_DIR}/publish_result.json"

REPORT_MANIFEST="\$(scripts/py - "\${RUN_DIR}/publish_result.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["manifest_path"])
PY
)"
scripts/py -m fisheye.reporting check-report \
  --manifest "\${REPORT_MANIFEST}" --check-files \
  >"\${RUN_DIR}/report_check.json"

{
  printf 'status=complete\n'
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'cohort_manifest=%s\n' "\${COHORT_MANIFEST}"
  printf 'export_run_id=%s\n' "\${EXPORT_RUN_ID}"
  printf 'report_manifest=%s\n' "\${REPORT_MANIFEST}"
} >"\${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "cohort_report_${REPORT_ID}"
  -n 1
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ "${#DEPENDENCY_DONE[@]}" -gt 0 ]]; then
  dependency_expression=""
  for dependency_job_id in "${DEPENDENCY_DONE[@]}"; do
    if [[ -n "$dependency_expression" ]]; then dependency_expression+=" && "; fi
    dependency_expression+="done(${dependency_job_id})"
  done
  BSUB_ARGS+=(-w "$dependency_expression")
fi
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'cohort_manifest=%s\n' "$COHORT_MANIFEST"
printf 'export_run_id=%s\n' "$EXPORT_RUN_ID"
printf 'report_id=%s\n' "$REPORT_ID"
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
    printf 'cohort_manifest=%s\n' "$COHORT_MANIFEST"
    printf 'export_run_id=%s\n' "$EXPORT_RUN_ID"
    printf 'report_id=%s\n' "$REPORT_ID"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
