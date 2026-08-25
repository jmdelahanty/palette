#!/usr/bin/env bash
set -euo pipefail
umask 0002

REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SOURCE_REPO="${PALETTE_SOURCE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
OUTPUT_ROOT="${PALETTE_REGISTRY_PROJECTION_REFRESH_ROOT:-/groups/johnson/johnsonlab/jeremy/registries/audits/registry_zarr_projection_refresh_bsub}"
RUN_ID=""
QUEUE=""
MEM_GB=8
WALLTIME="1:00"
APPLY=0
SUBMIT=0
REFRESH_DETECT_QUALITY=1
REFRESH_DETECT_PERFORMANCE=0
REFRESH_KEYPOINT_PERFORMANCE=0
ZARR_PATHS=()

usage() {
  cat <<'USAGE'
Usage: submit_registry_zarr_projection_refresh_bsub.sh --run-id ID --zarr-path PATH... [options]

Render or submit one sequential CPU job that refreshes selected registry
projections from analysis Zarrs. The login host only submits bsub; all Zarr
reads and registry writes occur inside the LSF allocation.

Required:
  --run-id ID                 Unique run identifier
  --zarr-path PATH            Analysis Zarr in scope; repeatable

Options:
  --apply                     Back up and update the registry (default: dry-run)
  --refresh-performance       Refresh both detect and keypoint performance
  --refresh-detect-performance
                              Refresh detect_performance only
  --refresh-keypoint-performance
                              Refresh keypoint_performance only
  --skip-detect-quality       Do not refresh detect_quality
  --registry PATH             Shared Palette registry
  --palette-repo PATH         Cluster-visible Palette checkout
  --source-repo PATH          Local checkout supplying bundled projection code
  --output-root PATH          Logs, backup, and path manifest root
  --submit-host HOST          SSH host used when bsub is unavailable locally
  --queue NAME
  --mem-gb N                  Default: 8
  --walltime H:MM             Default: 1:00
  --submit                    Submit; otherwise render only
  -h, --help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2;;
    --zarr-path) ZARR_PATHS+=("$2"); shift 2;;
    --apply) APPLY=1; shift;;
    --refresh-performance) REFRESH_DETECT_PERFORMANCE=1; REFRESH_KEYPOINT_PERFORMANCE=1; shift;;
    --refresh-detect-performance) REFRESH_DETECT_PERFORMANCE=1; shift;;
    --refresh-keypoint-performance) REFRESH_KEYPOINT_PERFORMANCE=1; shift;;
    --skip-detect-quality) REFRESH_DETECT_QUALITY=0; shift;;
    --registry) REGISTRY="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --source-repo) SOURCE_REPO="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$RUN_ID" ]] || fail "--run-id is required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe --run-id: $RUN_ID"
[[ "${#ZARR_PATHS[@]}" -gt 0 ]] || fail "provide at least one --zarr-path"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
if [[ "$APPLY" == "1" ]]; then
  fail "--apply is disabled until this maintenance path uses the shared registry shadow-publication gateway"
fi
[[ -f "$REGISTRY" ]] || fail "registry not found: $REGISTRY"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable: $PALETTE_REPO"
SOURCE_MAINTENANCE="$SOURCE_REPO/src/fisheye/registry/maintenance.py"
SOURCE_STAGE_CATALOG="$SOURCE_REPO/src/fisheye/registry/stage_catalog.py"
SOURCE_DETECT_PERFORMANCE="$SOURCE_REPO/src/fisheye/registry/extractors/detect_performance.py"
SOURCE_KEYPOINT_PERFORMANCE="$SOURCE_REPO/src/fisheye/registry/extractors/keypoint_performance.py"
[[ -f "$SOURCE_MAINTENANCE" ]] || fail "local maintenance module not found: $SOURCE_MAINTENANCE"
[[ -f "$SOURCE_STAGE_CATALOG" ]] || fail "local stage catalog not found: $SOURCE_STAGE_CATALOG"
[[ -f "$SOURCE_DETECT_PERFORMANCE" ]] || fail "local detect-performance extractor not found: $SOURCE_DETECT_PERFORMANCE"
[[ -f "$SOURCE_KEYPOINT_PERFORMANCE" ]] || fail "local keypoint-performance extractor not found: $SOURCE_KEYPOINT_PERFORMANCE"
for path in "${ZARR_PATHS[@]}"; do
  [[ -f "$path/zarr.json" ]] || fail "not a Zarr v3 root: $path"
done

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"
PATHS_FILE="$RUN_DIR/zarr_paths.txt"
JOB_SCRIPT="$RUN_DIR/run_registry_zarr_projection_refresh.sh"
STATUS_FILE="$RUN_DIR/status.txt"
SUBMISSION_FILE="$RUN_DIR/submission.txt"
BACKUP="$RUN_DIR/registry.before_refresh.sqlite"
MAINTENANCE_LOG="$RUN_DIR/maintenance.log"
BUNDLED_MAINTENANCE="$RUN_DIR/maintenance.py"
BUNDLED_STAGE_CATALOG="$RUN_DIR/stage_catalog.py"
BUNDLED_DETECT_PERFORMANCE="$RUN_DIR/detect_performance.py"
BUNDLED_KEYPOINT_PERFORMANCE="$RUN_DIR/keypoint_performance.py"
BUNDLED_LOADER="$RUN_DIR/run_bundled_maintenance.py"
printf '%s\n' "${ZARR_PATHS[@]}" >"$PATHS_FILE"
cp "$SOURCE_MAINTENANCE" "$BUNDLED_MAINTENANCE"
cp "$SOURCE_STAGE_CATALOG" "$BUNDLED_STAGE_CATALOG"
cp "$SOURCE_DETECT_PERFORMANCE" "$BUNDLED_DETECT_PERFORMANCE"
cp "$SOURCE_KEYPOINT_PERFORMANCE" "$BUNDLED_KEYPOINT_PERFORMANCE"

cat >"$BUNDLED_LOADER" <<'PY'
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(
            "usage: loader STAGE_CATALOG DETECT_PERFORMANCE "
            "KEYPOINT_PERFORMANCE MAINTENANCE [maintenance args...]"
        )
    import fisheye.registry as registry_package
    import fisheye.registry.extractors as extractors_package

    stage_catalog = _load("fisheye.registry.stage_catalog", Path(sys.argv[1]))
    setattr(registry_package, "stage_catalog", stage_catalog)
    detect_performance = _load(
        "fisheye.registry.extractors.detect_performance",
        Path(sys.argv[2]),
    )
    setattr(extractors_package, "detect_performance", detect_performance)
    keypoint_performance = _load(
        "fisheye.registry.extractors.keypoint_performance",
        Path(sys.argv[3]),
    )
    setattr(extractors_package, "keypoint_performance", keypoint_performance)
    maintenance = _load("fisheye.registry.maintenance", Path(sys.argv[4]))
    setattr(registry_package, "maintenance", maintenance)
    maintenance.main(sys.argv[5:])


if __name__ == "__main__":
    main()
PY

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
EXPECTED_SOURCE_SHA256="$(sha256sum \
  "$BUNDLED_MAINTENANCE" \
  "$BUNDLED_STAGE_CATALOG" \
  "$BUNDLED_DETECT_PERFORMANCE" \
  "$BUNDLED_KEYPOINT_PERFORMANCE" | awk '{print $1}' | sha256sum | awk '{print $1}')"
q_repo="$(printf '%q' "$PALETTE_REPO")"
q_registry="$(printf '%q' "$REGISTRY")"
q_paths="$(printf '%q' "$PATHS_FILE")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_backup="$(printf '%q' "$BACKUP")"
q_maintenance_log="$(printf '%q' "$MAINTENANCE_LOG")"
q_bundled_maintenance="$(printf '%q' "$BUNDLED_MAINTENANCE")"
q_bundled_stage_catalog="$(printf '%q' "$BUNDLED_STAGE_CATALOG")"
q_bundled_detect_performance="$(printf '%q' "$BUNDLED_DETECT_PERFORMANCE")"
q_bundled_keypoint_performance="$(printf '%q' "$BUNDLED_KEYPOINT_PERFORMANCE")"
q_bundled_loader="$(printf '%q' "$BUNDLED_LOADER")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_source_sha="$(printf '%q' "$EXPECTED_SOURCE_SHA256")"
q_expected_zarr_count="$(printf '%q' "${#ZARR_PATHS[@]}")"
q_refresh_detect_quality="$(printf '%q' "$REFRESH_DETECT_QUALITY")"
q_refresh_detect_performance="$(printf '%q' "$REFRESH_DETECT_PERFORMANCE")"
q_refresh_keypoint_performance="$(printf '%q' "$REFRESH_KEYPOINT_PERFORMANCE")"
if [[ "$APPLY" == "1" ]]; then
  q_operation="$(printf '%q' apply)"
  q_backup_status="$q_backup"
else
  q_operation="$(printf '%q' dry-run)"
  q_backup_status="$(printf '%q' none)"
fi

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
REGISTRY=${q_registry}
PATHS_FILE=${q_paths}
STATUS_FILE=${q_status}
BACKUP=${q_backup}
MAINTENANCE_LOG=${q_maintenance_log}
BUNDLED_MAINTENANCE=${q_bundled_maintenance}
BUNDLED_STAGE_CATALOG=${q_bundled_stage_catalog}
BUNDLED_DETECT_PERFORMANCE=${q_bundled_detect_performance}
BUNDLED_KEYPOINT_PERFORMANCE=${q_bundled_keypoint_performance}
BUNDLED_LOADER=${q_bundled_loader}
EXPECTED_COMMIT=${q_commit}
EXPECTED_SOURCE_SHA256=${q_source_sha}
EXPECTED_ZARR_COUNT=${q_expected_zarr_count}
REFRESH_DETECT_QUALITY=${q_refresh_detect_quality}
REFRESH_DETECT_PERFORMANCE=${q_refresh_detect_performance}
REFRESH_KEYPOINT_PERFORMANCE=${q_refresh_keypoint_performance}
OPERATION=${q_operation}
BACKUP_STATUS=${q_backup_status}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
ACTUAL_SOURCE_SHA256="\$(sha256sum \
  "\${BUNDLED_MAINTENANCE}" \
  "\${BUNDLED_STAGE_CATALOG}" \
  "\${BUNDLED_DETECT_PERFORMANCE}" \
  "\${BUNDLED_KEYPOINT_PERFORMANCE}" | awk '{print \$1}' | sha256sum | awk '{print \$1}')"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi
if [[ "\${ACTUAL_SOURCE_SHA256}" != "\${EXPECTED_SOURCE_SHA256}" ]]; then
  printf 'Projection source hash mismatch: expected %s, found %s\n' \
    "\${EXPECTED_SOURCE_SHA256}" "\${ACTUAL_SOURCE_SHA256}" >&2
  exit 2
fi
mapfile -t ZARR_PATHS <"\${PATHS_FILE}"
cmd=(
  scripts/py "\${BUNDLED_LOADER}"
  "\${BUNDLED_STAGE_CATALOG}"
  "\${BUNDLED_DETECT_PERFORMANCE}"
  "\${BUNDLED_KEYPOINT_PERFORMANCE}"
  "\${BUNDLED_MAINTENANCE}"
  --registry "\${REGISTRY}"
  --backfill-recording-step-status
  --recording-step-zarr-use analysis
)
if [[ "\${REFRESH_DETECT_QUALITY}" == "1" ]]; then
  cmd+=(--refresh-detect-quality)
fi
if [[ "\${REFRESH_DETECT_PERFORMANCE}" == "1" ]]; then
  cmd+=(--refresh-detect-performance)
fi
if [[ "\${REFRESH_KEYPOINT_PERFORMANCE}" == "1" ]]; then
  cmd+=(--refresh-keypoint-performance)
fi
JOBSCRIPT

if [[ "$APPLY" == "1" ]]; then
  cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
[[ ! -e "${BACKUP}" ]] || {
  printf 'Backup already exists: %s\n' "${BACKUP}" >&2
  exit 2
}
scripts/py - "${REGISTRY}" "${BACKUP}" <<'PY'
import sqlite3
import sys

source_path, backup_path = sys.argv[1:]
with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as source:
    with sqlite3.connect(backup_path) as backup:
        source.backup(backup)
print(f"registry_backup={backup_path}")
PY
JOBSCRIPT
else
  printf 'cmd+=(--dry-run)\n' >>"$JOB_SCRIPT"
fi

cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
cmd+=("${ZARR_PATHS[@]}")
printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
"${cmd[@]}" 2>&1 | tee "${MAINTENANCE_LOG}"
if [[ "${REFRESH_DETECT_QUALITY}" == "1" ]]; then
  grep -Fq \
    "Detect quality refresh: scanned=${EXPECTED_ZARR_COUNT} missing=0 errors=0 no_quality=0" \
    "${MAINTENANCE_LOG}"
fi
if [[ "${REFRESH_DETECT_PERFORMANCE}" == "1" ]]; then
  grep -Fq \
    "Detect performance refresh: scope=source-analysis-only scanned=${EXPECTED_ZARR_COUNT} missing=0 errors=0 no_performance=0" \
    "${MAINTENANCE_LOG}"
fi
if [[ "${REFRESH_KEYPOINT_PERFORMANCE}" == "1" ]]; then
  grep -Fq \
    "Keypoint performance refresh: scope=source-analysis-only scanned=${EXPECTED_ZARR_COUNT} missing=0 errors=0 no_performance=0" \
    "${MAINTENANCE_LOG}"
fi
grep -Eq \
  "^Recording step status backfill: scanned=[0-9]+ in_scope=${EXPECTED_ZARR_COUNT} missing_zarr=0 errors=0 " \
  "${MAINTENANCE_LOG}"

scripts/py - "${REGISTRY}" <<'PY'
import sqlite3
import sys

registry_path = sys.argv[1]
with sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True) as conn:
    foreign_key_issues = conn.execute("PRAGMA foreign_key_check;").fetchall()
    integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
if foreign_key_issues:
    raise SystemExit(f"foreign_key_check failed: {foreign_key_issues[:5]}")
if integrity != "ok":
    raise SystemExit(f"integrity_check failed: {integrity}")
print("sqlite_foreign_key_check=ok")
print("sqlite_integrity_check=ok")
PY

{
  printf 'status=complete\n'
  printf 'operation=%s\n' "${OPERATION}"
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'job_id=%s\n' "${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "${ACTUAL_COMMIT}"
  printf 'source_sha256=%s\n' "${ACTUAL_SOURCE_SHA256}"
  printf 'registry=%s\n' "${REGISTRY}"
  printf 'backup=%s\n' "${BACKUP_STATUS}"
} >"${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "registry_projection_${RUN_ID}"
  -n 1
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G] span[hosts=1]"
  -oo "$RUN_DIR/%J.out"
  -eo "$RUN_DIR/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'operation=%s\n' "$([[ "$APPLY" == "1" ]] && printf apply || printf dry-run)"
printf 'zarr_count=%s\n' "${#ZARR_PATHS[@]}"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'source_sha256=%s\n' "$EXPECTED_SOURCE_SHA256"
printf 'backup=%s\n' "$([[ "$APPLY" == "1" ]] && printf '%s' "$BACKUP" || printf none)"
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
    printf 'source_sha256=%s\n' "$EXPECTED_SOURCE_SHA256"
    printf 'backup=%s\n' "$([[ "$APPLY" == "1" ]] && printf '%s' "$BACKUP" || printf none)"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
