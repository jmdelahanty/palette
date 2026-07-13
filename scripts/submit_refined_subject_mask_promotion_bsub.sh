#!/usr/bin/env bash
set -euo pipefail

SOURCE_ZARR=""
SOURCE_RUN=""
TARGET_ZARR=""
EVIDENCE_JSON=""
REGISTRY=""
EXPECTED_ROWS=""
RUN_ID="refined_subject_mask_promotion_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/refined_subject_mask_promotion"
QUEUE="local"
WALLTIME="2:00"
MEM_GB=16
DRY_RUN=0
RESUME_EXISTING=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_refined_subject_mask_promotion_bsub.sh [options]

Submit a copy-only refined-subject-mask promotion. The compute job copies a
completed run to a hidden canonical path, validates the entire logical array
surface and tree inventory, atomically publishes it, then updates parent
pointers and registry projections. It performs no mask computation.

Required:
  --source-zarr PATH
  --source-run RUN
  --target-zarr PATH
  --expected-rows N
  --registry PATH

Options:
  --evidence-json PATH    Corrected validation evidence required for a canary.
  --run-id ID
  --log-root PATH
  --queue NAME            Default: local.
  --walltime H:MM         Default: 2:00.
  --mem-gb N              Default: 16.
  --resume-existing       Validate/reconcile an already copied target.
  --dry-run               Print bsub command without submitting.
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-zarr) SOURCE_ZARR="$2"; shift 2;;
    --source-run) SOURCE_RUN="$2"; shift 2;;
    --target-zarr) TARGET_ZARR="$2"; shift 2;;
    --expected-rows) EXPECTED_ROWS="$2"; shift 2;;
    --evidence-json) EVIDENCE_JSON="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --resume-existing) RESUME_EXISTING=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

for required in SOURCE_ZARR SOURCE_RUN TARGET_ZARR EXPECTED_ROWS REGISTRY; do
  if [[ -z "${!required}" ]]; then
    echo "Missing required option for $required." >&2
    usage
    exit 2
  fi
done
if [[ ! "$EXPECTED_ROWS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--expected-rows must be a positive integer." >&2
  exit 2
fi
if [[ ! "$MEM_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "--mem-gb must be a positive integer." >&2
  exit 2
fi

SOURCE_ZARR="${SOURCE_ZARR%/}"
TARGET_ZARR="${TARGET_ZARR%/}"
SOURCE_PATH="$SOURCE_ZARR/refined_subject_masks_runs/$SOURCE_RUN"
TARGET_PATH="$TARGET_ZARR/refined_subject_masks_runs/$SOURCE_RUN"
for path in \
  "$SOURCE_ZARR/zarr.json" \
  "$SOURCE_PATH/zarr.json" \
  "$TARGET_ZARR/zarr.json" \
  "$TARGET_ZARR/refined_subject_masks_runs/zarr.json" \
  "$REGISTRY"; do
  if [[ ! -e "$path" ]]; then
    echo "Required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ -n "$EVIDENCE_JSON" && ! -f "$EVIDENCE_JSON" ]]; then
  echo "Evidence JSON does not exist: $EVIDENCE_JSON" >&2
  exit 2
fi
if [[ -e "$TARGET_PATH" && "$RESUME_EXISTING" != "1" ]]; then
  echo "Canonical target already exists: $TARGET_PATH" >&2
  exit 2
fi

RUN_DIR="${LOG_ROOT%/}/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

COMMAND=(
  "$REPO_ROOT/scripts/py" -m fisheye.utils.promote_refined_subject_mask_run
  --source-zarr "$SOURCE_ZARR"
  --source-run "$SOURCE_RUN"
  --target-zarr "$TARGET_ZARR"
  --expected-rows "$EXPECTED_ROWS"
  --registry "$REGISTRY"
  --apply
  --output-json "$RUN_DIR/promotion.json"
)
if [[ -n "$EVIDENCE_JSON" ]]; then
  COMMAND+=(--evidence-json "$EVIDENCE_JSON")
fi
if [[ "$RESUME_EXISTING" == "1" ]]; then
  COMMAND+=(--resume-existing)
fi

BSUB=(
  bsub
  -J "refined_sm_promote"
  -q "$QUEUE"
  -n 1
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "$RUN_DIR/%J.out"
  -eo "$RUN_DIR/%J.err"
)

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'source_path=%s\n' "$SOURCE_PATH"
printf 'target_path=%s\n' "$TARGET_PATH"
printf 'operation=validated_copy_only_no_mask_recomputation\n'
printf 'command='
printf ' %q' "${BSUB[@]}" "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
"${BSUB[@]}" "${COMMAND[@]}"
