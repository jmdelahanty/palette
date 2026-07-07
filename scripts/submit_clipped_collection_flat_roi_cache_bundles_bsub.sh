#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
COLLECTION_ID=""
RECORDING_FRAME_INDEX=""
CLIP_IDS=()
WORK_UNIT_IDS=()
ALL_CLIPS=0
PUBLIC_CACHE_ROOT="/misc/public/palette_cache"
PUBLIC_CACHE_DIR_ROOT=""
LOG_DIR=""
RUN_ID_PREFIX=""
RUN_LABEL_PREFIX=""
QUEUE="gpu_l4"
NCORES=8
MEM_GB=64
GPUS=1
GPU_RESOURCE=""
WALLTIME="4:00"
ROI_SIZE=()
LIMIT_ROWS=""
GPU_CHUNK_FRAMES=32
CLIPS_PER_JOB=4
MAX_WORKERS=4
START_BUNDLE_INDEX=0
LIMIT_BUNDLES=""
PROGRESS_INTERVAL_S=30
PROGRESS_EVERY_BATCHES=0
SHA256=0
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_clipped_collection_flat_roi_cache_bundles_bsub.sh --zarr PATH --collection-id ID (--all-clips | --clip-id ID...) [options]

Resolve a clipped finalized collection, split clips into bundles, and submit one
bundle flat-cache job per group. Each bundle job runs
submit_clipped_collection_flat_roi_cache_bundle_bsub.sh, which launches multiple
clip-cache workers inside one GPU allocation.

Required:
  --zarr PATH                       Analysis Zarr archive
  --collection-id ID                Finalized clipped refined-detect collection id
  --all-clips                       Schedule every clip in the finalized collection
  --clip-id ID                      Schedule only this clip; repeatable

Selection:
  --recording-frame-index PATH      Override recording_frame_index.parquet path
  --work-unit-id ID                 Restrict selection to matching work_unit_id; repeatable

Bundle layout:
  --clips-per-job N                 Clip IDs per submitted bundle job (default: 4)
  --max-workers N                   Concurrent child builders per bundle job (default: 4)
  --start-bundle-index N            Skip bundles before this index (default: 0)
  --limit-bundles N                 Submit at most N bundles after start index

Cache options:
  --public-cache-root PATH          Shared cache root (default: /misc/public/palette_cache)
  --public-cache-dir-root PATH      Root for per-bundle publish dirs. Default:
                                    <public-cache-root>/<collection>/roi_cache_bundles/<run-id-prefix>
  --roi-size H W                    ROI size in Palette order; default from archive policy
  --limit-rows N                    Debug/smoke limit per child clip
  --gpu-chunk-frames N              Sequential PyNv decode batch size per child (default: 32)
  --sha256                          Record payload sha256 in child manifests
  --overwrite                       Overwrite existing published child cache files

LSF options:
  --queue NAME                      LSF queue (default: gpu_l4)
  --ncores N                        CPU slots per bundle job (default: 8)
  --mem-gb N                        Memory request in GB per bundle job (default: 64)
  --gpus N                          GPU count; 0 omits -gpu (default: 1)
  --gpu-resource STRING             Raw LSF -gpu resource string; overrides --gpus
  --walltime H:MM                   Wall time per bundle job (default: 4:00)

Logging:
  --log-dir PATH                    Log/output directory for child bundle submissions
  --run-id-prefix ID                Stable prefix; default is UTC timestamp
  --run-label-prefix LABEL          Bundle label prefix; default is collection id
  --progress-interval-s SECONDS     Progress interval passed to each child builder
  --progress-every-batches N        Emit progress every N batches per child; 0 disables

General:
  --dry-run                         Print child submit commands; do not submit
  -h, --help                        Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --collection-id) COLLECTION_ID="$2"; shift 2;;
    --recording-frame-index) RECORDING_FRAME_INDEX="$2"; shift 2;;
    --clip-id) CLIP_IDS+=("$2"); shift 2;;
    --all-clips) ALL_CLIPS=1; shift;;
    --work-unit-id) WORK_UNIT_IDS+=("$2"); shift 2;;
    --clips-per-job) CLIPS_PER_JOB="$2"; shift 2;;
    --max-workers) MAX_WORKERS="$2"; shift 2;;
    --start-bundle-index) START_BUNDLE_INDEX="$2"; shift 2;;
    --limit-bundles) LIMIT_BUNDLES="$2"; shift 2;;
    --public-cache-root) PUBLIC_CACHE_ROOT="$2"; shift 2;;
    --public-cache-dir-root) PUBLIC_CACHE_DIR_ROOT="$2"; shift 2;;
    --roi-size) ROI_SIZE=("$2" "$3"); shift 3;;
    --limit-rows) LIMIT_ROWS="$2"; shift 2;;
    --gpu-chunk-frames) GPU_CHUNK_FRAMES="$2"; shift 2;;
    --sha256) SHA256=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --gpu-resource) GPU_RESOURCE="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2;;
    --run-label-prefix) RUN_LABEL_PREFIX="$2"; shift 2;;
    --progress-interval-s) PROGRESS_INTERVAL_S="$2"; shift 2;;
    --progress-every-batches) PROGRESS_EVERY_BATCHES="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR_PATH" || -z "$COLLECTION_ID" ]]; then
  echo "Missing required --zarr PATH or --collection-id ID" >&2
  usage
  exit 2
fi
if [[ "$ALL_CLIPS" == "1" && "${#CLIP_IDS[@]}" -gt 0 ]]; then
  echo "Use either --all-clips or explicit --clip-id values, not both." >&2
  exit 2
fi
if [[ "$ALL_CLIPS" != "1" && "${#CLIP_IDS[@]}" -eq 0 ]]; then
  echo "Provide at least one --clip-id, or pass --all-clips." >&2
  exit 2
fi
if [[ "$CLIPS_PER_JOB" -lt 1 ]]; then
  echo "--clips-per-job must be >= 1" >&2
  exit 2
fi
if [[ "$MAX_WORKERS" -lt 1 ]]; then
  echo "--max-workers must be >= 1" >&2
  exit 2
fi
if [[ "$START_BUNDLE_INDEX" -lt 0 ]]; then
  echo "--start-bundle-index must be >= 0" >&2
  exit 2
fi
if [[ -n "$LIMIT_BUNDLES" && "$LIMIT_BUNDLES" -lt 1 ]]; then
  echo "--limit-bundles must be >= 1 when provided" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -d "$ZARR_PATH" ]] || { echo "Zarr path not found: $ZARR_PATH" >&2; exit 2; }
  if [[ -n "$RECORDING_FRAME_INDEX" ]]; then
    [[ -f "$RECORDING_FRAME_INDEX" ]] || { echo "Recording frame index not found: $RECORDING_FRAME_INDEX" >&2; exit 2; }
  fi
fi

if [[ "$ALL_CLIPS" == "1" ]]; then
  mapfile -t CLIP_IDS < <(
    scripts/py - "$ZARR_PATH" "$COLLECTION_ID" "$(IFS=,; echo "${WORK_UNIT_IDS[*]}")" <<'PY'
import sys
from collections.abc import Mapping
from pathlib import Path

import zarr

zarr_path, collection_id, work_units_csv = sys.argv[1:]
work_filter = {item for item in work_units_csv.split(",") if item}

def open_root(path: Path):
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")

root = open_root(Path(zarr_path))
collection = root["experiment_index"]["finalized_runs"][collection_id]
selected_runs = collection.attrs.get("selected_runs", [])
clip_ids: list[str] = []
for row in selected_runs:
    if not isinstance(row, Mapping):
        continue
    clip_id = str(row.get("clip_id") or "").strip()
    work_unit_id = str(row.get("work_unit_id") or "").strip()
    if not clip_id:
        continue
    if work_filter and work_unit_id not in work_filter:
        continue
    if clip_id not in clip_ids:
        clip_ids.append(clip_id)
for clip_id in sorted(clip_ids):
    print(clip_id)
PY
  )
  if [[ "${#CLIP_IDS[@]}" -eq 0 ]]; then
    echo "--all-clips resolved no clip ids for collection: $COLLECTION_ID" >&2
    exit 2
  fi
fi

if [[ -z "$RUN_ID_PREFIX" ]]; then
  RUN_ID_PREFIX="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_LABEL_PREFIX" ]]; then
  RUN_LABEL_PREFIX="$COLLECTION_ID"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/diagnostics/clipped_collection_flat_roi_cache_bundle_bsub"
fi

SAFE_COLLECTION_ID="$(printf '%s' "$COLLECTION_ID" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_RUN_ID_PREFIX="$(printf '%s' "$RUN_ID_PREFIX" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_LABEL_PREFIX="$(printf '%s' "$RUN_LABEL_PREFIX" | tr -c 'A-Za-z0-9_.-' '_')"

if [[ -z "$PUBLIC_CACHE_DIR_ROOT" ]]; then
  PUBLIC_CACHE_DIR_ROOT="${PUBLIC_CACHE_ROOT}/${SAFE_COLLECTION_ID}/roi_cache_bundles/${SAFE_RUN_ID_PREFIX}"
fi

total_clips="${#CLIP_IDS[@]}"
total_bundles=$(( (total_clips + CLIPS_PER_JOB - 1) / CLIPS_PER_JOB ))
submitted=0

echo "Resolved clips: $total_clips"
echo "Total bundles: $total_bundles"
echo "Clips per job: $CLIPS_PER_JOB"
echo "Max workers per job: $MAX_WORKERS"
echo "Start bundle index: $START_BUNDLE_INDEX"
echo "Limit bundles: ${LIMIT_BUNDLES:-none}"
echo "Public cache dir root: $PUBLIC_CACHE_DIR_ROOT"
echo "Log dir: $LOG_DIR"

for ((bundle_index = START_BUNDLE_INDEX; bundle_index < total_bundles; bundle_index++)); do
  if [[ -n "$LIMIT_BUNDLES" && "$submitted" -ge "$LIMIT_BUNDLES" ]]; then
    break
  fi
  offset=$(( bundle_index * CLIPS_PER_JOB ))
  bundle_clips=( "${CLIP_IDS[@]:offset:CLIPS_PER_JOB}" )
  if [[ "${#bundle_clips[@]}" -eq 0 ]]; then
    continue
  fi
  bundle_suffix="$(printf 'b%04d' "$bundle_index")"
  bundle_run_id="${SAFE_RUN_ID_PREFIX}_${bundle_suffix}"
  bundle_label="${SAFE_LABEL_PREFIX}_${bundle_suffix}"
  bundle_public_dir="${PUBLIC_CACHE_DIR_ROOT}/${bundle_label}"

  args=(
    --zarr "$ZARR_PATH"
    --collection-id "$COLLECTION_ID"
    --public-cache-dir "$bundle_public_dir"
    --log-dir "$LOG_DIR"
    --run-id "$bundle_run_id"
    --run-label "$bundle_label"
    --queue "$QUEUE"
    --ncores "$NCORES"
    --mem-gb "$MEM_GB"
    --gpus "$GPUS"
    --max-workers "$MAX_WORKERS"
    --walltime "$WALLTIME"
    --gpu-chunk-frames "$GPU_CHUNK_FRAMES"
    --progress-interval-s "$PROGRESS_INTERVAL_S"
    --progress-every-batches "$PROGRESS_EVERY_BATCHES"
  )
  if [[ -n "$RECORDING_FRAME_INDEX" ]]; then args+=(--recording-frame-index "$RECORDING_FRAME_INDEX"); fi
  if [[ -n "$GPU_RESOURCE" ]]; then args+=(--gpu-resource "$GPU_RESOURCE"); fi
  for clip_id in "${bundle_clips[@]}"; do args+=(--clip-id "$clip_id"); done
  for work_unit_id in "${WORK_UNIT_IDS[@]}"; do args+=(--work-unit-id "$work_unit_id"); done
  if [[ "${#ROI_SIZE[@]}" -gt 0 ]]; then args+=(--roi-size "${ROI_SIZE[@]}"); fi
  if [[ -n "$LIMIT_ROWS" ]]; then args+=(--limit-rows "$LIMIT_ROWS"); fi
  if [[ "$SHA256" == "1" ]]; then args+=(--sha256); fi
  if [[ "$OVERWRITE" == "1" ]]; then args+=(--overwrite); fi
  if [[ "$DRY_RUN" == "1" ]]; then args+=(--dry-run); fi

  echo
  echo "Submitting bundle $bundle_index/${total_bundles}: ${bundle_clips[*]}"
  echo "Bundle public dir: $bundle_public_dir"
  scripts/submit_clipped_collection_flat_roi_cache_bundle_bsub.sh "${args[@]}"
  submitted=$(( submitted + 1 ))
done

echo
echo "Submitted bundles: $submitted"
