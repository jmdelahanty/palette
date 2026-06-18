#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
CONFIG="configs/fisheye/yolo_detect_config.yaml"
PATH_CONTAINS=""
LOG_DIR=""
RUN_ID=""
MODEL=""
DETECT_SET_ID=""
DETECT_REQUIRE_UNIQUE=0
DETECT_INCLUDE_NON_SUCCESS=0
DETECT_TOP_K=5
REQUIRE_TUNING=0
OVERWRITE=0

DETECT_QUEUE="gpu_l4"
DETECT_NCORES=8
DETECT_MEM_GB=120
DETECT_GPUS=1
DETECT_WALLTIME="2:00"
DETECT_DECODE_BACKEND="pynvvc_nv12_rgb"
DETECT_BATCH_SIZE=16
DETECT_CONF=""
DETECT_IOU=""
DETECT_MAX_DET=""
DETECT_RESIZE_DIMS=("640" "640")
ARTIFACT_LATEST_POLICY="set_latest_explicit"

POST_QUEUE="short"
POST_NCORES=4
POST_MEM_GB=16
POST_WALLTIME="1:00"
QUALITY_THRESHOLD=100.0
QUALITY_THRESHOLD_MODE="scaled"
QUALITY_THRESHOLD_REFERENCE_WIDTH=640.0
REFINE_SAVE_VISUALS=0

SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_detect_artifact_quality_refine_bsub.sh [--model PATH] [options]

Submit a registry-discovered per-recording artifact workflow:

  detect artifact on node scratch -> import into canonical zarr -> validate
  imported run -> detect_quality -> refined_detect

The script plans by default. Pass --submit to call bsub.

Discovery options:
  --root PATH                    Recording root (default: /groups/johnson/johnsonlab/jeremy/recordings)
  --registry PATH                Registry sqlite path (default: $PALETTE_REGISTRY_PATH or /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite)
  --path-contains STR            Registry zarr_path substring filter
  --config PATH                  Detect/refine config path
  --log-dir PATH                 Submission run root (default: <root>/logs/detect_artifact_quality_refine_bsub)
  --run-id ID                    Stable workflow id
  --require-tuning               Skip zarrs without detection_tuning
  --overwrite                    Plan zarrs even if detect is already ok

Detect artifact options:
  --model PATH                   Explicit detect model path. If omitted, resolve
                                  a detect model from the registry per recording.
                                  Resolved/explicit paths must be readable on
                                  the submit host.
  --detect-set-id ID             Optional detect set filter for registry model resolution
  --detect-require-unique        Fail if top registry model candidates tie
  --detect-include-non-success   Allow non-success training runs as model candidates
  --detect-top-k N               Candidate provenance depth (default: 5)
  --detect-queue NAME            LSF queue for detect artifact jobs (default: gpu_l4)
  --detect-ncores N              Cores per detect job (default: 8)
  --detect-mem-gb N              Memory per detect job in GB (default: 120)
  --detect-gpus N                GPUs per detect job (default: 1)
  --detect-walltime HH:MM        Walltime per detect job (default: 2:00)
  --detect-decode-backend NAME   Backend passed to run_detection_artifact
                                  (default: pynvvc_nv12_rgb)
  --detect-batch-size N          Inference batch size (default: 16)
  --detect-conf FLOAT            Optional confidence threshold
  --detect-iou FLOAT             Optional IoU threshold
  --detect-max-det N             Optional max detections per frame
  --detect-resize-dims H W       Canonical inference size (default: 640 640)
  --artifact-latest-policy NAME  Manifest latest policy (default: set_latest_explicit)

Postprocess options:
  --post-queue NAME              LSF queue for import/quality/refine jobs (default: short)
  --post-ncores N                Cores per postprocess job (default: 4)
  --post-mem-gb N                Memory per postprocess job in GB (default: 16)
  --post-walltime HH:MM          Walltime per postprocess job (default: 1:00)
  --quality-threshold VALUE      Jump threshold (default: 100.0)
  --quality-threshold-mode MODE  scaled, pixels, or normalized (default: scaled)
  --quality-threshold-reference-width VALUE
                                  Reference width for scaled threshold (default: 640.0)
  --refine-save-visuals          Ask refine_detect_batch to save visuals

Execution:
  --submit                       Actually submit jobs. Without this, dry-run only.
  -h, --help                     Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --require-tuning) REQUIRE_TUNING=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --model) MODEL="$2"; shift 2;;
    --detect-set-id) DETECT_SET_ID="$2"; shift 2;;
    --detect-require-unique) DETECT_REQUIRE_UNIQUE=1; shift;;
    --detect-include-non-success) DETECT_INCLUDE_NON_SUCCESS=1; shift;;
    --detect-top-k) DETECT_TOP_K="$2"; shift 2;;
    --detect-queue) DETECT_QUEUE="$2"; shift 2;;
    --detect-ncores) DETECT_NCORES="$2"; shift 2;;
    --detect-mem-gb) DETECT_MEM_GB="$2"; shift 2;;
    --detect-gpus) DETECT_GPUS="$2"; shift 2;;
    --detect-walltime) DETECT_WALLTIME="$2"; shift 2;;
    --detect-decode-backend) DETECT_DECODE_BACKEND="$2"; shift 2;;
    --detect-batch-size) DETECT_BATCH_SIZE="$2"; shift 2;;
    --detect-conf) DETECT_CONF="$2"; shift 2;;
    --detect-iou) DETECT_IOU="$2"; shift 2;;
    --detect-max-det) DETECT_MAX_DET="$2"; shift 2;;
    --detect-resize-dims) DETECT_RESIZE_DIMS=("$2" "$3"); shift 3;;
    --artifact-latest-policy) ARTIFACT_LATEST_POLICY="$2"; shift 2;;
    --post-queue) POST_QUEUE="$2"; shift 2;;
    --post-ncores) POST_NCORES="$2"; shift 2;;
    --post-mem-gb) POST_MEM_GB="$2"; shift 2;;
    --post-walltime) POST_WALLTIME="$2"; shift 2;;
    --quality-threshold) QUALITY_THRESHOLD="$2"; shift 2;;
    --quality-threshold-mode) QUALITY_THRESHOLD_MODE="$2"; shift 2;;
    --quality-threshold-reference-width) QUALITY_THRESHOLD_REFERENCE_WIDTH="$2"; shift 2;;
    --refine-save-visuals) REFINE_SAVE_VISUALS=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/palette_lsf.sh"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT}/logs/detect_artifact_quality_refine_bsub"
fi

RUN_DIR="${LOG_DIR}/detect_artifact_quality_refine_${RUN_ID}"
ARTIFACT_OUTPUT_DIR="${RUN_DIR}/artifacts"
POSTPROCESS_DIR="${RUN_DIR}/postprocess"
QUALITY_LOG_DIR="${RUN_DIR}/detect_quality_logs"
REFINE_LOG_DIR="${RUN_DIR}/refine_detect_logs"
RAW_PLAN_JSONL="${RUN_DIR}/detect_plan.raw.jsonl"
TARGETS_JSONL="${RUN_DIR}/targets.jsonl"
TARGETS_TSV="${RUN_DIR}/targets.tsv"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or remove the existing run directory." >&2
  exit 2
fi
mkdir -p "$RUN_DIR" "$ARTIFACT_OUTPUT_DIR" "$POSTPROCESS_DIR" "$QUALITY_LOG_DIR" "$REFINE_LOG_DIR"
export PALETTE_JOB_CACHE="${RUN_DIR}/palette_job_cache"
export YOLO_CONFIG_DIR="${PALETTE_JOB_CACHE}/ultralytics"
export MPLCONFIGDIR="${PALETTE_JOB_CACHE}/matplotlib"
mkdir -p "$PALETTE_JOB_CACHE" "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"

PLAN_CMD=(
  scripts/py -m fisheye.utils.run_detections_batch
  "$ROOT"
  --source registry
  --registry "$REGISTRY"
  --dry-run
  --json
  --resolve-models
  --no-log
  --config "$CONFIG"
  --decode-backend "$DETECT_DECODE_BACKEND"
  --batch-size "$DETECT_BATCH_SIZE"
)
if [[ -n "$PATH_CONTAINS" ]]; then PLAN_CMD+=(--path-contains "$PATH_CONTAINS"); fi
if [[ "$REQUIRE_TUNING" == "1" ]]; then PLAN_CMD+=(--require-tuning); fi
if [[ "$OVERWRITE" == "1" ]]; then PLAN_CMD+=(--overwrite); fi
if [[ -n "$MODEL" ]]; then PLAN_CMD+=(--model "$MODEL"); fi
if [[ -n "$DETECT_SET_ID" ]]; then PLAN_CMD+=(--set-id "$DETECT_SET_ID"); fi
if [[ "$DETECT_REQUIRE_UNIQUE" == "1" ]]; then PLAN_CMD+=(--require-unique); fi
if [[ "$DETECT_INCLUDE_NON_SUCCESS" == "1" ]]; then PLAN_CMD+=(--include-non-success); fi
if [[ -n "$DETECT_TOP_K" ]]; then PLAN_CMD+=(--top-k "$DETECT_TOP_K"); fi
if [[ -n "$DETECT_CONF" ]]; then PLAN_CMD+=(--conf "$DETECT_CONF"); fi
if [[ -n "$DETECT_IOU" ]]; then PLAN_CMD+=(--iou "$DETECT_IOU"); fi
if [[ -n "$DETECT_MAX_DET" ]]; then PLAN_CMD+=(--max-det "$DETECT_MAX_DET"); fi
if [[ "${#DETECT_RESIZE_DIMS[@]}" -gt 0 ]]; then PLAN_CMD+=(--resize-dims "${DETECT_RESIZE_DIMS[@]}"); fi

echo "Planning target recordings..."
"${PLAN_CMD[@]}" > "$RAW_PLAN_JSONL"

scripts/py - "$RAW_PLAN_JSONL" "$RUN_DIR" "$RUN_ID" <<'PY'
import json
import re
import sys
from pathlib import Path

raw_plan = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
run_id = sys.argv[3]


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("_") or "target"


plans = []
ignored = []
for line_no, line in enumerate(raw_plan.read_text(encoding="utf-8").splitlines(), start=1):
    text = line.strip()
    if not text:
        continue
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        ignored.append({"line_no": line_no, "text": text})
        continue
    if not isinstance(payload, dict) or "zarr" not in payload:
        ignored.append({"line_no": line_no, "text": text})
        continue
    plans.append(payload)

targets = []
for plan in plans:
    if plan.get("status") != "ok":
        continue
    video = plan.get("video")
    zarr = plan.get("zarr")
    model = plan.get("selected_model")
    if not isinstance(video, str) or not video or not isinstance(zarr, str) or not zarr:
        continue
    if not isinstance(model, str) or not model:
        raise SystemExit(
            f"Runnable plan is missing selected_model for zarr={zarr!r}; "
            "pass --model or fix registry model resolution."
        )
    if not Path(model).is_file():
        raise SystemExit(
            f"Selected model path is not readable for zarr={zarr!r}: {model}. "
            "Pass --model with a cluster-visible weights path or refresh the registry model path."
        )
    index = len(targets) + 1
    zarr_stem = Path(zarr).name
    if zarr_stem.endswith(".zarr"):
        zarr_stem = zarr_stem[:-5]
    if zarr_stem.endswith("_analysis"):
        zarr_stem = zarr_stem[:-9]
    label_base = safe_token(zarr_stem)
    run_id_token = safe_token(run_id)
    target = {
        "index": index,
        "recording": plan.get("recording"),
        "zarr": zarr,
        "video": video,
        "model": model,
        "selected_run_id": plan.get("selected_run_id"),
        "selected_set_id": plan.get("selected_set_id"),
        "detect_run_name": f"detect_{run_id_token}_{index:04d}",
        "run_label": f"{index:04d}_{label_base}",
        "safe_label": safe_token(f"{index:04d}_{label_base}"),
        "artifact_run_id": f"{run_id_token}_{index:04d}",
    }
    targets.append(target)

(run_dir / "targets.jsonl").write_text(
    "".join(json.dumps(target, sort_keys=True) + "\n" for target in targets),
    encoding="utf-8",
)
(run_dir / "targets.tsv").write_text(
    "".join(
        "\t".join(
            [
                f"{target['index']:04d}",
                str(target["zarr"]),
                str(target["video"]),
                str(target["model"]),
                str(target["detect_run_name"]),
                str(target["run_label"]),
                str(target["safe_label"]),
                str(target["artifact_run_id"]),
            ]
        )
        + "\n"
        for target in targets
    ),
    encoding="utf-8",
)
(run_dir / "plan_ignored_lines.jsonl").write_text(
    "".join(json.dumps(item, sort_keys=True) + "\n" for item in ignored),
    encoding="utf-8",
)
counts: dict[str, int] = {}
for plan in plans:
    status = str(plan.get("status") or "unknown")
    counts[status] = counts.get(status, 0) + 1
summary = {
    "schema_version": 1,
    "workflow": "detect_artifact_quality_refine_bsub",
    "run_id": run_id,
    "planned_rows": len(plans),
    "target_count": len(targets),
    "counts_by_status": counts,
    "ignored_non_json_lines": len(ignored),
}
(run_dir / "manifest_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

target_count="$(scripts/py - "$RUN_DIR/manifest_summary.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["target_count"])
PY
)"

if [[ "$target_count" == "0" ]]; then
  echo "No runnable target zarrs discovered."
  echo "Raw plan: $RAW_PLAN_JSONL"
  echo "Summary: ${RUN_DIR}/manifest_summary.json"
  exit 0
fi

SUBMISSIONS_TSV="${RUN_DIR}/submissions.tsv"
printf 'index\tzarr\tvideo\tmodel\tdetect_run\tquality_run\tdetect_jobid\tpost_jobid\tartifact_tarball\n' > "$SUBMISSIONS_TSV"

while IFS=$'\t' read -r TARGET_INDEX ZARR VIDEO TARGET_MODEL DETECT_RUN_NAME RUN_LABEL SAFE_LABEL ARTIFACT_RUN_ID; do
  [[ -z "$TARGET_INDEX" ]] && continue

  ARTIFACT_CMD=(
    "${SCRIPT_DIR}/submit_detect_artifact_bsub.sh"
    --zarr "$ZARR"
    --video "$VIDEO"
    --model "$TARGET_MODEL"
    --output-dir "$ARTIFACT_OUTPUT_DIR"
    --config "$CONFIG"
    --queue "$DETECT_QUEUE"
    --ncores "$DETECT_NCORES"
    --mem-gb "$DETECT_MEM_GB"
    --gpus "$DETECT_GPUS"
    --walltime "$DETECT_WALLTIME"
    --decode-backend "$DETECT_DECODE_BACKEND"
    --batch-size "$DETECT_BATCH_SIZE"
    --run-id "$ARTIFACT_RUN_ID"
    --run-label "$RUN_LABEL"
    --workflow-id "$RUN_ID"
    --detect-run-name "$DETECT_RUN_NAME"
    --latest-policy "$ARTIFACT_LATEST_POLICY"
  )
  if [[ -n "$DETECT_CONF" ]]; then ARTIFACT_CMD+=(--conf "$DETECT_CONF"); fi
  if [[ -n "$DETECT_IOU" ]]; then ARTIFACT_CMD+=(--iou "$DETECT_IOU"); fi
  if [[ -n "$DETECT_MAX_DET" ]]; then ARTIFACT_CMD+=(--max-det "$DETECT_MAX_DET"); fi
  if [[ "${#DETECT_RESIZE_DIMS[@]}" -gt 0 ]]; then ARTIFACT_CMD+=(--resize-dims "${DETECT_RESIZE_DIMS[@]}"); fi
  if [[ "$SUBMIT" != "1" ]]; then ARTIFACT_CMD+=(--dry-run); fi

  ARTIFACT_SUBMIT_LOG="${RUN_DIR}/artifact_submit_${TARGET_INDEX}.log"
  echo "Planning detect artifact ${TARGET_INDEX}/${target_count}: $ZARR"
  "${ARTIFACT_CMD[@]}" 2>&1 | tee "$ARTIFACT_SUBMIT_LOG"

  if [[ "$SUBMIT" == "1" ]]; then
    detect_jobid="$(palette_lsf_extract_jobid "$ARTIFACT_SUBMIT_LOG")"
    if [[ -z "$detect_jobid" ]]; then
      echo "Could not parse detect artifact job id from $ARTIFACT_SUBMIT_LOG" >&2
      exit 2
    fi
  else
    detect_jobid="<detect_jobid_${TARGET_INDEX}>"
  fi

  ARTIFACT_RUN_DIR="${ARTIFACT_OUTPUT_DIR}/detect_artifact_${ARTIFACT_RUN_ID}_${SAFE_LABEL}"
  ARTIFACT_TARBALL="${ARTIFACT_RUN_DIR}/${SAFE_LABEL}.${detect_jobid}.tar.gz"
  QUALITY_RUN_NAME="detect_quality_${ARTIFACT_RUN_ID}"
  TARGET_POST_DIR="${POSTPROCESS_DIR}/${TARGET_INDEX}_${SAFE_LABEL}"
  mkdir -p "$TARGET_POST_DIR"

  ZARR_Q="$(printf '%q' "$ZARR")"
  TARBALL_Q="$(printf '%q' "$ARTIFACT_TARBALL")"
  DETECT_RUN_Q="$(printf '%q' "$DETECT_RUN_NAME")"
  QUALITY_RUN_Q="$(printf '%q' "$QUALITY_RUN_NAME")"
  TARGET_POST_DIR_Q="$(printf '%q' "$TARGET_POST_DIR")"
  QUALITY_LOG_DIR_Q="$(printf '%q' "$QUALITY_LOG_DIR")"
  REFINE_LOG_DIR_Q="$(printf '%q' "$REFINE_LOG_DIR")"
  CONFIG_Q="$(printf '%q' "$CONFIG")"
  REPO_ROOT_Q="$(printf '%q' "$REPO_ROOT")"

  POST_SCRIPT="${TARGET_POST_DIR}/run_import_quality_refine.sh"
  cat > "$POST_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd ${REPO_ROOT_Q}
ZARR=${ZARR_Q}
ARTIFACT_TARBALL=${TARBALL_Q}
DETECT_RUN_NAME=${DETECT_RUN_Q}
QUALITY_RUN_NAME=${QUALITY_RUN_Q}
TARGET_POST_DIR=${TARGET_POST_DIR_Q}
QUALITY_LOG_DIR=${QUALITY_LOG_DIR_Q}
REFINE_LOG_DIR=${REFINE_LOG_DIR_Q}
CONFIG=${CONFIG_Q}

mkdir -p "\$TARGET_POST_DIR" "\$QUALITY_LOG_DIR" "\$REFINE_LOG_DIR"

scripts/py -m fisheye.utils.import_run_group_artifact "\$ARTIFACT_TARBALL" \\
  --target-zarr "\$ZARR" \\
  --apply \\
  > "\$TARGET_POST_DIR/import_run_group_artifact.json"

scripts/py -m fisheye.utils.validate_imported_run_group "\$ZARR" \\
  --target-group-path "detect_runs/\$DETECT_RUN_NAME" \\
  > "\$TARGET_POST_DIR/validate_imported_run_group.json"

scripts/py -m fisheye.utils.detect_quality_batch "\$ZARR" \\
  --apply \\
  --json \\
  --detect-run "\$DETECT_RUN_NAME" \\
  --quality-run-name "\$QUALITY_RUN_NAME" \\
  --threshold "$QUALITY_THRESHOLD" \\
  --threshold-mode "$QUALITY_THRESHOLD_MODE" \\
  --threshold-reference-width "$QUALITY_THRESHOLD_REFERENCE_WIDTH" \\
  --log-dir "\$QUALITY_LOG_DIR" \\
  > "\$TARGET_POST_DIR/detect_quality.stdout.jsonl"

refine_cmd=(
  scripts/py -m fisheye.utils.refine_detect_batch "\$ZARR"
  --apply
  --zarr-use analysis
  --detect-run "\$DETECT_RUN_NAME"
  --quality-run "\$QUALITY_RUN_NAME"
  --config "\$CONFIG"
  --log-dir "\$REFINE_LOG_DIR"
)
if [[ "$REFINE_SAVE_VISUALS" == "1" ]]; then
  refine_cmd+=(--save-visuals)
fi
"\${refine_cmd[@]}" > "\$TARGET_POST_DIR/refine_detect.stdout.txt"
JOBSCRIPT
  chmod +x "$POST_SCRIPT"

  POST_BSUB_ARGS=(
    bsub
    -J "detect_post_${TARGET_INDEX}_${RUN_ID}"
    -n "$POST_NCORES"
    -R "rusage[mem=${POST_MEM_GB}G]"
    -W "$POST_WALLTIME"
    -oo "${TARGET_POST_DIR}/%J.out"
    -eo "${TARGET_POST_DIR}/%J.err"
    -w "done(${detect_jobid})"
  )
  if [[ -n "$POST_QUEUE" ]]; then POST_BSUB_ARGS+=(-q "$POST_QUEUE"); fi
  POST_BSUB_ARGS+=(bash "$POST_SCRIPT")

  POST_SUBMIT_LOG="${TARGET_POST_DIR}/postprocess_submit.log"
  echo "Planning import/quality/refine postprocess for ${TARGET_INDEX}"
  palette_lsf_submit_or_print "$SUBMIT" "$POST_SUBMIT_LOG" "${POST_BSUB_ARGS[@]}"

  if [[ "$SUBMIT" == "1" ]]; then
    post_jobid="$(palette_lsf_extract_jobid "$POST_SUBMIT_LOG")"
    if [[ -z "$post_jobid" ]]; then
      echo "Could not parse postprocess job id from $POST_SUBMIT_LOG" >&2
      exit 2
    fi
  else
    post_jobid="<postprocess_jobid_${TARGET_INDEX}>"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$TARGET_INDEX" "$ZARR" "$VIDEO" "$TARGET_MODEL" "$DETECT_RUN_NAME" "$QUALITY_RUN_NAME" "$detect_jobid" "$post_jobid" "$ARTIFACT_TARBALL" \
    >> "$SUBMISSIONS_TSV"
done < "$TARGETS_TSV"

cat > "${RUN_DIR}/submission_summary.txt" <<SUMMARY
run_id=$RUN_ID
run_dir=$RUN_DIR
root=$ROOT
registry=$REGISTRY
path_contains=${PATH_CONTAINS:-<none>}
target_count=$target_count
model=${MODEL:-<registry resolution>}
config=$CONFIG
artifact_output_dir=$ARTIFACT_OUTPUT_DIR
postprocess_dir=$POSTPROCESS_DIR
raw_plan_jsonl=$RAW_PLAN_JSONL
targets_jsonl=$TARGETS_JSONL
submissions_tsv=$SUBMISSIONS_TSV
SUMMARY

echo "Run dir: $RUN_DIR"
echo "Targets: $target_count"
echo "Targets JSONL: $TARGETS_JSONL"
echo "Submissions: $SUBMISSIONS_TSV"
echo "Summary: ${RUN_DIR}/submission_summary.txt"
if [[ "$SUBMIT" != "1" ]]; then
  echo "Dry run only; pass --submit to submit the per-recording artifact chains."
fi
