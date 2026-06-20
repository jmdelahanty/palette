#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
MAX_ACTIVE=4
QUEUE=""
NCORES=8
MEM_GB=48
GPUS=1
SPLIT_FINALIZATION_JOB=1
FINALIZE_QUEUE=""
FINALIZE_NCORES=""
FINALIZE_MEM_GB=""
FINALIZE_MAX_ACTIVE=""
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
LOG_DIR=""
RUN_ID_OVERRIDE=""
RUN_LABEL_OVERRIDE=""
DRY_RUN=0
SOURCE="registry"
RIG_ID=""
ARENA_ID=""
CAMERA_ID_FILTER=""
PATH_CONTAINS=""
DEVICE=""
BATCH_SIZE_SM=128
MASK_PROBS_DTYPE="uint8"
MASK_PROBS_CHUNK_ROIS=32
OUTPUT_QUEUE_SIZE=2
PROFILE_TIMINGS=0
MODEL_COVERAGE_CLASS="dense_all_components"
MODEL_COMPONENT_COVERAGE_KEY="body+eyes+swim_bladder"
MODEL_LABEL_SCHEMA_ID="subject_v1_union"
MODEL_TOP_K=5
MODEL_REQUIRE_UNIQUE=0
MODEL_INCLUDE_NON_SUCCESS=0
FINALIZE_CHUNK_SIZE=256
FINALIZE_EXECUTION_BACKEND="process_shards"
FINALIZE_SCHEDULER="processes"
FINALIZE_NUM_WORKERS="auto"
FINALIZE_POSTCOMPUTE_BACKEND="process_shards"
FINALIZE_POSTCOMPUTE_CHUNK_SIZE=""
FINALIZE_POSTCOMPUTE_NUM_WORKERS="auto"
METRIC_LEVEL="cheap"
MASK_STORAGE="dense_uint8"
MASK_RLE_VALIDATION_MODE="invariants"
WRITE_EYE_GEOMETRY=1
WRITE_COMPONENT_CONTOURS=1
RETAIN_SOURCE_SEEDS=0
FORCE_INFERENCE=0
FORCE_FINALIZATION=0
OVERWRITE=0
ROI_CACHE_ROOT=""
ROI_CACHE_MANIFEST=""
REQUIRE_ROI_CACHE=1
STAGE_ROI_CACHE_TO_SCRATCH=1
ROI_CACHE_STAGING_DIR=""
ROI_CACHE_POLICY="never"
ROI_LIVE_ACCELERATION="auto"
ROI_LIVE_GPU_CHUNK_FRAMES=32
STAGE_OUTPUT_TO_SCRATCH=1
OUTPUT_STAGING_DIR=""
KEEP_STAGED_OUTPUT=0
STAGE_FINALIZATION_INPUT_TO_SCRATCH=1
HANDOFF_PACKAGE_DIR="${PALETTE_SUBJECT_MASK_HANDOFF_DIR:-/nrs/ahrens/palette_staging/subject_mask_run_packages}"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_batches_bsub.sh [options]

Submits one LSF array task per selected recording. By default, inference runs in
a GPU array job, publishes subject_mask_runs to the recording zarr, and a
dependent CPU array job finalizes refined_subject_masks. Each job may stage
temporary cache/output data to node-local scratch and deletes it via an EXIT
trap.

Options:
  --root PATH               Root recordings directory (default: /groups/johnson/johnsonlab/jeremy/recordings)
  --max-active N            Max concurrent recording jobs (default: 4)
  --queue NAME              LSF queue name
  --ncores N                Cores per job (default: 8)
  --mem-gb N                Memory per job in GB (default: 48)
  --gpus N                  GPUs per job (default: 1); when >0 defaults --device 0
  --single-job-finalization Run finalization inside the GPU job instead of a dependent CPU job
  --finalize-queue NAME     CPU finalization queue (default: cluster default)
  --finalize-ncores N       CPU finalization cores (default: --ncores)
  --finalize-mem-gb N       CPU finalization memory in GB (default: --mem-gb)
  --finalize-max-active N   Max concurrent finalization jobs (default: --max-active)
  --registry PATH           Registry sqlite path (default: $PALETTE_REGISTRY_PATH or PRFS registry)
  --source filesystem|registry
                            Discovery source (default: registry)
  --rig-id ID               Filter by rig_id (registry source only)
  --arena-id ID             Filter by arena_id (registry source only)
  --camera-id-filter ID     Filter by camera_id (registry source only)
  --path-contains STR       Filter zarr_path by substring
  --roi-cache-root PATH     Directory containing *.flat_roi_cache.json manifests; resolved per recording
  --roi-cache-manifest PATH Explicit flat_bin_v1 ROI cache manifest; allowed only for one selected recording
  --allow-missing-roi-cache Do not fail planning if a selected recording has no cache manifest
  --no-stage-roi-cache-to-scratch
                            Read cache directly from its source instead of staging to node-local scratch
  --roi-cache-staging-dir PATH
                            Override local staging base (default: /scratch/$USER/$LSB_JOBID/...)
  --no-stage-output-to-scratch
                            Write subject-mask outputs directly to PRFS instead of staging locally
  --output-staging-dir PATH Override output staging base (default: /scratch/$USER/$LSB_JOBID/...)
  --keep-staged-output      Keep local staged output zarr after successful publish
  --no-stage-finalization-input-to-scratch
                            In split mode, symlink the published subject-mask run instead of copying it local
  --handoff-package-dir PATH
                            Shared non-backed-up package dir for split finalization handoff
                            (default: $PALETTE_SUBJECT_MASK_HANDOFF_DIR or /nrs/ahrens/palette_staging/subject_mask_run_packages)
  --device DEVICE           Torch device override (default: 0 when --gpus > 0, else cuda:0)
  --batch-size-sm N         Subject-mask inference batch size (default: 128)
  --mask-probs-dtype DTYPE  uint8|float16 (default: uint8)
  --mask-probs-chunk-rois N Chunk length for mask_probs_roi (default: 32)
  --output-queue-size N     Async output queue size (default: 2)
  --profile-timings         Persist per-stage subject-mask inference timing diagnostics
  --model-coverage-class X  Registry model coverage_class (default: dense_all_components)
  --model-component-coverage-key X
                            Registry model component key (default: body+eyes+swim_bladder)
  --model-label-schema-id X Registry model label schema (default: subject_v1_union)
  --model-top-k N           Candidate provenance depth (default: 5)
  --model-require-unique    Fail if model resolution top score ties
  --model-include-non-success
                            Include non-success registry model rows
  --finalize-chunk-size N   Refined finalizer chunk size (default: 256)
  --finalize-execution-backend NAME
                            serial_driver|dask_worker_chunks|process_shards (default: process_shards)
  --finalize-num-workers N|auto
                            Refined finalizer worker count (default: auto => --ncores)
  --finalize-postcompute-backend NAME
                            serial|process_shards for eye geometry/contours (default: process_shards)
  --finalize-postcompute-chunk-size N
                            Rows per postcompute shard (default: finalizer uses --finalize-chunk-size)
  --finalize-postcompute-num-workers N|auto
                            Postcompute worker count (default: finalizer uses --finalize-num-workers)
  --metric-level LEVEL      cheap|full (default: cheap)
  --mask-storage MODE       dense_uint8|dense_and_rle|rle_v1 for refined masks
                            (default: dense_uint8)
  --mask-rle-validation-mode MODE
                            full|invariants|none after writing compact mask_rle
                            (default: invariants for production batch runs)
  --no-write-eye-geometry   Do not ask finalizer to write eye geometry
  --no-write-component-contours
                            Do not ask finalizer to write component contours
  --retain-source-seeds     Retain dense source_seed_masks_roi debug arrays during finalization
  --force-inference         Run inference even if subject_mask_runs exists
  --force-finalization      Run finalization even if refined_subject_masks_runs exists
  --overwrite               Pass overwrite through to child stages
  --log-dir PATH            Submission logs (default: <root>/logs/run_subject_mask_batch/bsub_submissions)
  --run-id ID               Stable run id for deterministic reruns
  --run-label LABEL         Run label passed to the Python batch driver
  --dry-run                 Print manifests + commands; do not submit
  -h, --help                Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --max-active) MAX_ACTIVE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --single-job-finalization) SPLIT_FINALIZATION_JOB=0; shift;;
    --finalize-queue) FINALIZE_QUEUE="$2"; shift 2;;
    --finalize-ncores) FINALIZE_NCORES="$2"; shift 2;;
    --finalize-mem-gb) FINALIZE_MEM_GB="$2"; shift 2;;
    --finalize-max-active) FINALIZE_MAX_ACTIVE="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --rig-id) RIG_ID="$2"; shift 2;;
    --arena-id) ARENA_ID="$2"; shift 2;;
    --camera-id-filter) CAMERA_ID_FILTER="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    --roi-cache-root) ROI_CACHE_ROOT="$2"; shift 2;;
    --roi-cache-manifest) ROI_CACHE_MANIFEST="$2"; shift 2;;
    --allow-missing-roi-cache) REQUIRE_ROI_CACHE=0; shift;;
    --no-stage-roi-cache-to-scratch) STAGE_ROI_CACHE_TO_SCRATCH=0; shift;;
    --roi-cache-staging-dir) ROI_CACHE_STAGING_DIR="$2"; shift 2;;
    --no-stage-output-to-scratch) STAGE_OUTPUT_TO_SCRATCH=0; shift;;
    --output-staging-dir) OUTPUT_STAGING_DIR="$2"; shift 2;;
    --keep-staged-output) KEEP_STAGED_OUTPUT=1; shift;;
    --no-stage-finalization-input-to-scratch) STAGE_FINALIZATION_INPUT_TO_SCRATCH=0; shift;;
    --handoff-package-dir) HANDOFF_PACKAGE_DIR="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --batch-size-sm) BATCH_SIZE_SM="$2"; shift 2;;
    --mask-probs-dtype) MASK_PROBS_DTYPE="$2"; shift 2;;
    --mask-probs-chunk-rois) MASK_PROBS_CHUNK_ROIS="$2"; shift 2;;
    --output-queue-size) OUTPUT_QUEUE_SIZE="$2"; shift 2;;
    --profile-timings) PROFILE_TIMINGS=1; shift;;
    --model-coverage-class) MODEL_COVERAGE_CLASS="$2"; shift 2;;
    --model-component-coverage-key) MODEL_COMPONENT_COVERAGE_KEY="$2"; shift 2;;
    --model-label-schema-id) MODEL_LABEL_SCHEMA_ID="$2"; shift 2;;
    --model-top-k) MODEL_TOP_K="$2"; shift 2;;
    --model-require-unique) MODEL_REQUIRE_UNIQUE=1; shift;;
    --model-include-non-success) MODEL_INCLUDE_NON_SUCCESS=1; shift;;
    --finalize-chunk-size) FINALIZE_CHUNK_SIZE="$2"; shift 2;;
    --finalize-execution-backend) FINALIZE_EXECUTION_BACKEND="$2"; shift 2;;
    --finalize-num-workers) FINALIZE_NUM_WORKERS="$2"; shift 2;;
    --finalize-postcompute-backend) FINALIZE_POSTCOMPUTE_BACKEND="$2"; shift 2;;
    --finalize-postcompute-chunk-size) FINALIZE_POSTCOMPUTE_CHUNK_SIZE="$2"; shift 2;;
    --finalize-postcompute-num-workers) FINALIZE_POSTCOMPUTE_NUM_WORKERS="$2"; shift 2;;
    --metric-level) METRIC_LEVEL="$2"; shift 2;;
    --mask-storage) MASK_STORAGE="$2"; shift 2;;
    --mask-rle-validation-mode) MASK_RLE_VALIDATION_MODE="$2"; shift 2;;
    --no-write-eye-geometry) WRITE_EYE_GEOMETRY=0; shift;;
    --no-write-component-contours) WRITE_COMPONENT_CONTOURS=0; shift;;
    --retain-source-seeds) RETAIN_SOURCE_SEEDS=1; shift;;
    --force-inference) FORCE_INFERENCE=1; shift;;
    --force-finalization) FORCE_FINALIZATION=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID_OVERRIDE="$2"; shift 2;;
    --run-label) RUN_LABEL_OVERRIDE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ "$SOURCE" != "filesystem" && "$SOURCE" != "registry" ]]; then
  echo "--source must be filesystem or registry." >&2
  exit 2
fi
if [[ "$FINALIZE_EXECUTION_BACKEND" != "serial_driver" && "$FINALIZE_EXECUTION_BACKEND" != "dask_worker_chunks" && "$FINALIZE_EXECUTION_BACKEND" != "process_shards" ]]; then
  echo "--finalize-execution-backend must be serial_driver, dask_worker_chunks, or process_shards." >&2
  exit 2
fi
if [[ "$FINALIZE_POSTCOMPUTE_BACKEND" != "serial" && "$FINALIZE_POSTCOMPUTE_BACKEND" != "process_shards" ]]; then
  echo "--finalize-postcompute-backend must be serial or process_shards." >&2
  exit 2
fi
if [[ "$MASK_STORAGE" != "dense_uint8" && "$MASK_STORAGE" != "dense_and_rle" && "$MASK_STORAGE" != "rle_v1" ]]; then
  echo "--mask-storage must be dense_uint8, dense_and_rle, or rle_v1." >&2
  exit 2
fi
if [[ "$MASK_RLE_VALIDATION_MODE" != "full" && "$MASK_RLE_VALIDATION_MODE" != "invariants" && "$MASK_RLE_VALIDATION_MODE" != "none" ]]; then
  echo "--mask-rle-validation-mode must be full, invariants, or none." >&2
  exit 2
fi
if [[ "$STAGE_ROI_CACHE_TO_SCRATCH" == "1" && "$REQUIRE_ROI_CACHE" != "1" ]]; then
  echo "--no-stage-roi-cache-to-scratch is required when --allow-missing-roi-cache is used." >&2
  exit 2
fi
if [[ "$STAGE_ROI_CACHE_TO_SCRATCH" == "1" && -z "$ROI_CACHE_ROOT" && -z "$ROI_CACHE_MANIFEST" ]]; then
  echo "Staging requires --roi-cache-root or --roi-cache-manifest." >&2
  exit 2
fi
if [[ "$GPUS" != "0" && -z "$DEVICE" ]]; then
  DEVICE="0"
fi
if [[ -z "$DEVICE" ]]; then
  DEVICE="cuda:0"
fi
if [[ "$FINALIZE_NUM_WORKERS" == "auto" || -z "$FINALIZE_NUM_WORKERS" ]]; then
  FINALIZE_NUM_WORKERS="$NCORES"
fi
if ! [[ "$NCORES" =~ ^[0-9]+$ ]] || [[ "$NCORES" -lt 1 ]]; then
  echo "--ncores must be a positive integer." >&2
  exit 2
fi
if [[ -z "$FINALIZE_NCORES" ]]; then
  FINALIZE_NCORES="$NCORES"
fi
if [[ -z "$FINALIZE_MEM_GB" ]]; then
  FINALIZE_MEM_GB="$MEM_GB"
fi
if [[ -z "$FINALIZE_MAX_ACTIVE" ]]; then
  FINALIZE_MAX_ACTIVE="$MAX_ACTIVE"
fi
if ! [[ "$FINALIZE_NCORES" =~ ^[0-9]+$ ]] || [[ "$FINALIZE_NCORES" -lt 1 ]]; then
  echo "--finalize-ncores must be a positive integer." >&2
  exit 2
fi
if ! [[ "$FINALIZE_NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$FINALIZE_NUM_WORKERS" -lt 1 ]]; then
  echo "--finalize-num-workers must be a positive integer or auto." >&2
  exit 2
fi
if [[ -n "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE" ]]; then
  if ! [[ "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE" =~ ^[0-9]+$ ]] || [[ "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE" -lt 1 ]]; then
    echo "--finalize-postcompute-chunk-size must be a positive integer." >&2
    exit 2
  fi
fi
if [[ "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" != "auto" && -n "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" ]]; then
  if ! [[ "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" -lt 1 ]]; then
    echo "--finalize-postcompute-num-workers must be a positive integer or auto." >&2
    exit 2
  fi
fi
if [[ "$SPLIT_FINALIZATION_JOB" == "1" && "$FINALIZE_NUM_WORKERS" == "$NCORES" && "$FINALIZE_NCORES" != "$NCORES" ]]; then
  FINALIZE_NUM_WORKERS="$FINALIZE_NCORES"
fi
if [[ "$FINALIZE_NUM_WORKERS" -gt "$FINALIZE_NCORES" ]]; then
  echo "Warning: --finalize-num-workers (${FINALIZE_NUM_WORKERS}) exceeds finalization cores (${FINALIZE_NCORES}); this can oversubscribe the LSF allocation." >&2
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT}/logs/run_subject_mask_batch/bsub_submissions"
fi
if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -n "$RUN_LABEL_OVERRIDE" ]]; then
  RUN_LABEL="$RUN_LABEL_OVERRIDE"
else
  RUN_LABEL="subject_masks_${RUN_ID}"
fi
RUN_DIR="${LOG_DIR}/sm_${RUN_ID}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or remove the existing run directory." >&2
  exit 2
fi
if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  if [[ "$DRY_RUN" == "1" ]]; then
    FALLBACK_LOG_DIR="${TMPDIR:-/tmp}/palette/run_subject_mask_batch/bsub_submissions"
    RUN_DIR="${FALLBACK_LOG_DIR}/sm_${RUN_ID}"
    if [[ -e "$RUN_DIR" ]]; then
      echo "Fallback run directory already exists: $RUN_DIR" >&2
      exit 2
    fi
    mkdir -p "$RUN_DIR"
    echo "Warning: cannot write under ${LOG_DIR}; using fallback ${FALLBACK_LOG_DIR}" >&2
  else
    echo "Cannot create run directory: $RUN_DIR" >&2
    echo "Use --log-dir to choose a writable location." >&2
    exit 2
  fi
fi

if [[ "$SPLIT_FINALIZATION_JOB" == "1" ]]; then
  DISCOVER_WORKFLOW_STAGE="inference"
else
  DISCOVER_WORKFLOW_STAGE="all"
fi
DISCOVER_ARGS=(--source "$SOURCE" --emit-paths --registry "$REGISTRY" --workflow-stage "$DISCOVER_WORKFLOW_STAGE")
if [[ "$FORCE_INFERENCE" == "1" ]]; then DISCOVER_ARGS+=(--force-inference); fi
if [[ "$FORCE_FINALIZATION" == "1" ]]; then DISCOVER_ARGS+=(--force-finalization); fi
if [[ "$OVERWRITE" == "1" ]]; then DISCOVER_ARGS+=(--overwrite); fi
if [[ -n "$RIG_ID" ]]; then DISCOVER_ARGS+=(--rig-id "$RIG_ID"); fi
if [[ -n "$ARENA_ID" ]]; then DISCOVER_ARGS+=(--arena-id "$ARENA_ID"); fi
if [[ -n "$CAMERA_ID_FILTER" ]]; then DISCOVER_ARGS+=(--camera-id-filter "$CAMERA_ID_FILTER"); fi
if [[ -n "$PATH_CONTAINS" ]]; then DISCOVER_ARGS+=(--path-contains "$PATH_CONTAINS"); fi
DISCOVER_ARGS+=("$ROOT")

scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline "${DISCOVER_ARGS[@]}" > "${RUN_DIR}/discovered_paths.txt"

scripts/py - "${RUN_DIR}/discovered_paths.txt" "$RUN_DIR" "$ROI_CACHE_ROOT" "$ROI_CACHE_MANIFEST" "$REQUIRE_ROI_CACHE" <<'PY'
import json
import sys
from pathlib import Path

paths_file = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
cache_root_arg = sys.argv[3]
explicit_manifest_arg = sys.argv[4]
require_cache = bool(int(sys.argv[5]))

zarr_paths = [Path(line.strip()) for line in paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]
cache_root = Path(cache_root_arg).expanduser().resolve() if cache_root_arg else None
explicit_manifest = Path(explicit_manifest_arg).expanduser().resolve() if explicit_manifest_arg else None

if explicit_manifest is not None and len(zarr_paths) != 1:
    raise SystemExit("--roi-cache-manifest is only valid when exactly one recording is selected.")

def read_attrs(group_path: Path) -> dict:
    try:
        payload = json.loads((group_path / "zarr.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}

def child_groups(group_path: Path) -> list[str]:
    if not group_path.is_dir():
        return []
    return sorted(path.name for path in group_path.iterdir() if path.is_dir())

def latest_crop_run(zarr_path: Path) -> str | None:
    parent = zarr_path / "crop_runs"
    attrs = read_attrs(parent)
    for key in ("latest_any", "latest", "latest_materialized"):
        value = attrs.get(key)
        if isinstance(value, str) and value and (parent / value).is_dir():
            return value
    children = child_groups(parent)
    return children[-1] if children else None

def safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value)) or "cache"

def resolve_manifest(zarr_path: Path) -> Path | None:
    if explicit_manifest is not None:
        return explicit_manifest
    if cache_root is None:
        return None
    crop_run = latest_crop_run(zarr_path)
    archive = safe_component(zarr_path.stem)
    patterns: list[str] = []
    if crop_run:
        patterns.append(f"{archive}__{safe_component(crop_run)}__*.flat_roi_cache.json")
    patterns.append(f"{archive}.flat_roi_cache.json")
    patterns.append(f"{archive}__*.flat_roi_cache.json")
    matches: list[Path] = []
    for pattern in patterns:
        # cache_root should point directly at the roi_cache directory. Avoid
        # recursive PRFS walks here; they can be extremely slow on large trees.
        matches = sorted(cache_root.glob(pattern), key=lambda item: (item.stat().st_mtime_ns, str(item)))
        if matches:
            break
    return matches[-1] if matches else None

run_dir.mkdir(parents=True, exist_ok=True)
target_rows: list[tuple[Path, Path | None]] = []
missing: list[str] = []
for zarr_path in zarr_paths:
    manifest = resolve_manifest(zarr_path)
    if manifest is None and require_cache:
        missing.append(str(zarr_path))
        continue
    target_rows.append((zarr_path, manifest))

if missing:
    raise SystemExit(
        "Missing flat ROI cache manifest for selected recordings:\n"
        + "\n".join(f"  {item}" for item in missing)
    )

(run_dir / "recordings.txt").write_text(
    "\n".join(str(path) for path, _manifest in target_rows) + ("\n" if target_rows else ""),
    encoding="utf-8",
)
(run_dir / "targets.tsv").write_text(
    "\n".join(f"{path}\t{manifest or ''}" for path, manifest in target_rows) + ("\n" if target_rows else ""),
    encoding="utf-8",
)
for idx, (path, manifest) in enumerate(target_rows, start=1):
    (run_dir / f"target_{idx:04d}.tsv").write_text(f"{path}\t{manifest or ''}\n", encoding="utf-8")

(run_dir / "manifest_summary.json").write_text(
    json.dumps(
        {
            "analysis_zarr_count": len(target_rows),
            "one_job_per_recording": True,
            "cache_root": str(cache_root) if cache_root else None,
            "explicit_manifest": str(explicit_manifest) if explicit_manifest else None,
            "require_roi_cache": require_cache,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

job_count=$(find "$RUN_DIR" -maxdepth 1 -name 'target_*.tsv' | wc -l | tr -d ' ')
if [[ "$job_count" == "0" ]]; then
  echo "No subject-mask targets selected."
  exit 0
fi
analysis_count=$(wc -l < "$RUN_DIR/recordings.txt" | tr -d ' ')

SUBJECT_ARGS=(
  --apply
  --registry "$REGISTRY"
  --run-label "$RUN_LABEL"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE_SM"
  --mask-probs-dtype "$MASK_PROBS_DTYPE"
  --mask-probs-chunk-rois "$MASK_PROBS_CHUNK_ROIS"
  --output-queue-size "$OUTPUT_QUEUE_SIZE"
  --model-coverage-class "$MODEL_COVERAGE_CLASS"
  --model-component-coverage-key "$MODEL_COMPONENT_COVERAGE_KEY"
  --model-label-schema-id "$MODEL_LABEL_SCHEMA_ID"
  --metric-level "$METRIC_LEVEL"
  --mask-storage "$MASK_STORAGE"
  --mask-rle-validation-mode "$MASK_RLE_VALIDATION_MODE"
  --finalize-chunk-size "$FINALIZE_CHUNK_SIZE"
  --finalize-execution-backend "$FINALIZE_EXECUTION_BACKEND"
  --finalize-scheduler "$FINALIZE_SCHEDULER"
  --finalize-num-workers "$FINALIZE_NUM_WORKERS"
  --finalize-postcompute-backend "$FINALIZE_POSTCOMPUTE_BACKEND"
  --roi-cache-policy "$ROI_CACHE_POLICY"
  --roi-live-acceleration "$ROI_LIVE_ACCELERATION"
  --roi-live-gpu-chunk-frames "$ROI_LIVE_GPU_CHUNK_FRAMES"
  --progress-dir "$RUN_DIR/progress"
)
if [[ -n "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE" ]]; then SUBJECT_ARGS+=(--finalize-postcompute-chunk-size "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE"); fi
if [[ "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" != "auto" && -n "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" ]]; then SUBJECT_ARGS+=(--finalize-postcompute-num-workers "$FINALIZE_POSTCOMPUTE_NUM_WORKERS"); fi
if [[ "$MODEL_REQUIRE_UNIQUE" == "1" ]]; then SUBJECT_ARGS+=(--model-require-unique); fi
if [[ "$MODEL_INCLUDE_NON_SUCCESS" == "1" ]]; then SUBJECT_ARGS+=(--model-include-non-success); fi
if [[ "$PROFILE_TIMINGS" == "1" ]]; then SUBJECT_ARGS+=(--profile-timings); fi
if [[ "$WRITE_EYE_GEOMETRY" == "1" ]]; then SUBJECT_ARGS+=(--write-eye-geometry); else SUBJECT_ARGS+=(--no-write-eye-geometry); fi
if [[ "$WRITE_COMPONENT_CONTOURS" == "1" ]]; then SUBJECT_ARGS+=(--write-component-contours); else SUBJECT_ARGS+=(--no-write-component-contours); fi
if [[ "$RETAIN_SOURCE_SEEDS" == "1" ]]; then SUBJECT_ARGS+=(--retain-source-seeds); fi
if [[ "$STAGE_OUTPUT_TO_SCRATCH" == "1" ]]; then SUBJECT_ARGS+=(--stage-output-to-scratch); fi
if [[ -n "$OUTPUT_STAGING_DIR" ]]; then SUBJECT_ARGS+=(--output-staging-dir "$OUTPUT_STAGING_DIR"); fi
if [[ "$KEEP_STAGED_OUTPUT" == "1" ]]; then SUBJECT_ARGS+=(--keep-staged-output); fi
if [[ "$STAGE_FINALIZATION_INPUT_TO_SCRATCH" == "1" ]]; then SUBJECT_ARGS+=(--stage-finalization-input-to-scratch); fi
if [[ -n "$HANDOFF_PACKAGE_DIR" ]]; then SUBJECT_ARGS+=(--handoff-package-dir "$HANDOFF_PACKAGE_DIR"); fi
if [[ "$FORCE_INFERENCE" == "1" ]]; then SUBJECT_ARGS+=(--force-inference); fi
if [[ "$FORCE_FINALIZATION" == "1" ]]; then SUBJECT_ARGS+=(--force-finalization); fi
if [[ "$OVERWRITE" == "1" ]]; then SUBJECT_ARGS+=(--overwrite); fi

{
  echo "SUBJECT_ARGS=("
  printf '  %q\n' "${SUBJECT_ARGS[@]}"
  echo ")"
  printf 'RUN_LABEL=%q\n' "$RUN_LABEL"
  printf 'STAGE_ROI_CACHE_TO_SCRATCH=%q\n' "$STAGE_ROI_CACHE_TO_SCRATCH"
  printf 'ROI_CACHE_STAGING_DIR=%q\n' "$ROI_CACHE_STAGING_DIR"
} > "${RUN_DIR}/subject_args.sh"

JOB_SCRIPT="${RUN_DIR}/run_one_recording.sh"
cat > "$JOB_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
WORKFLOW_STAGE="${2:-all}"
if [[ -z "${LSB_JOBINDEX:-}" ]]; then
  echo "LSB_JOBINDEX not set; are you running under bsub array?" >&2
  exit 2
fi

source "${RUN_DIR}/subject_args.sh"

THREADS_PER_PROCESS="${THREADS_PER_PROCESS:-1}"
export OMP_NUM_THREADS="$THREADS_PER_PROCESS"
export MKL_NUM_THREADS="$THREADS_PER_PROCESS"
export OPENBLAS_NUM_THREADS="$THREADS_PER_PROCESS"
export TBB_NUM_THREADS="$THREADS_PER_PROCESS"
export OPENMP_NUM_THREADS="$THREADS_PER_PROCESS"
export NUM_MKL_THREADS="$THREADS_PER_PROCESS"
export NUMEXPR_NUM_THREADS="$THREADS_PER_PROCESS"

TARGET_FILE="${RUN_DIR}/target_$(printf '%04d' "${LSB_JOBINDEX}").tsv"
if [[ ! -f "$TARGET_FILE" ]]; then
  echo "Missing target file: $TARGET_FILE" >&2
  exit 2
fi

line="$(head -n 1 "$TARGET_FILE")"
zarr_path="${line%%$'\t'*}"
manifest_path=""
if [[ "$line" == *$'\t'* ]]; then
  manifest_path="${line#*$'\t'}"
fi

if [[ -z "$zarr_path" ]]; then
  echo "Empty zarr path in $TARGET_FILE" >&2
  exit 2
fi

if [[ "$WORKFLOW_STAGE" == "finalization" ]]; then
  expected_subject_run="subject_masks_unet_registry_${RUN_LABEL}"
  expected_subject_marker="${zarr_path}/subject_mask_runs/${expected_subject_run}/zarr.json"
  for attempt in $(seq 1 60); do
    if [[ -f "$expected_subject_marker" ]]; then
      break
    fi
    if [[ "$attempt" == "60" ]]; then
      echo "Subject-mask run is not visible for finalization: $expected_subject_marker" >&2
      exit 3
    fi
    sleep 10
  done
fi

LOCAL_CACHE_DIR=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$LOCAL_CACHE_DIR" && -d "$LOCAL_CACHE_DIR" ]]; then
    echo "Cleaning staged ROI cache: $LOCAL_CACHE_DIR"
    rm -rf "$LOCAL_CACHE_DIR"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

effective_manifest=""
if [[ "$WORKFLOW_STAGE" != "finalization" && -n "$manifest_path" ]]; then
  if [[ "${STAGE_ROI_CACHE_TO_SCRATCH}" == "1" ]]; then
    user_name="${USER:-unknown}"
    job_id="${LSB_JOBID:-manual}"
    job_index="${LSB_JOBINDEX:-0}"
    if [[ -n "${ROI_CACHE_STAGING_DIR}" ]]; then
      LOCAL_CACHE_DIR="${ROI_CACHE_STAGING_DIR}/${job_id}_${job_index}"
    elif [[ -d "/scratch/${user_name}" ]]; then
      LOCAL_CACHE_DIR="/scratch/${user_name}/${job_id}/palette_subject_mask_cache_${job_index}"
    else
      LOCAL_CACHE_DIR="${TMPDIR:-/tmp}/palette_subject_mask_cache_${job_id}_${job_index}"
    fi
    echo "Staging ROI cache for ${zarr_path}"
    echo "  source manifest: ${manifest_path}"
    echo "  local dir: ${LOCAL_CACHE_DIR}"
    effective_manifest="$(
      scripts/py - "$manifest_path" "$LOCAL_CACHE_DIR" <<'PY'
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

source_manifest = Path(sys.argv[1]).expanduser().resolve()
target_dir = Path(sys.argv[2]).expanduser().resolve()
target_dir.mkdir(parents=True, exist_ok=True)

payload = json.loads(source_manifest.read_text(encoding="utf-8"))
array = payload.get("array")
if not isinstance(array, dict):
    raise SystemExit(f"Flat ROI cache manifest is missing array metadata: {source_manifest}")
raw_bin = str(array.get("bin_path") or "")
if not raw_bin:
    raise SystemExit(f"Flat ROI cache manifest is missing array.bin_path: {source_manifest}")
source_bin = Path(raw_bin).expanduser()
if not source_bin.is_absolute():
    source_bin = source_manifest.parent / source_bin
source_bin = source_bin.resolve()
if not source_bin.exists():
    raise SystemExit(f"Flat ROI cache payload not found: {source_bin}")

local_manifest = target_dir / source_manifest.name
local_bin = target_dir / source_bin.name
if local_manifest == source_manifest or local_bin == source_bin:
    raise SystemExit("Refusing to stage ROI cache into the source cache directory.")

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

started_at = now()
started = time.perf_counter()
tmp_bin = local_bin.with_name(f"{local_bin.name}.tmp.{os.getpid()}")
shutil.copy2(source_bin, tmp_bin)
os.replace(tmp_bin, local_bin)
duration = time.perf_counter() - started
size = local_bin.stat().st_size

staged = dict(payload)
staged_array = dict(array)
staged_array["bin_path"] = local_bin.name
staged["array"] = staged_array
staged["manifest_path"] = str(local_manifest)
staged["staging"] = {
    "schema": "palette_roi_cache_staging_v1",
    "policy": "node_scratch_staged_flat_cache",
    "staged": True,
    "requested_manifest_path": str(source_manifest),
    "effective_manifest_path": str(local_manifest),
    "source_bin_path": str(source_bin),
    "effective_bin_path": str(local_bin),
    "staging_dir": str(target_dir),
    "host": socket.gethostname(),
    "lsb_jobid": os.environ.get("LSB_JOBID"),
    "lsb_jobindex": os.environ.get("LSB_JOBINDEX"),
    "copy": {
        "started_at_utc": started_at,
        "finished_at_utc": now(),
        "duration_seconds": duration,
        "size_bytes": size,
        "mib_per_second": (size / (1024 * 1024)) / duration if duration > 0 else None,
    },
}
tmp_manifest = local_manifest.with_name(f"{local_manifest.name}.tmp.{os.getpid()}")
tmp_manifest.write_text(json.dumps(staged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp_manifest, local_manifest)
print(local_manifest)
PY
    )"
  else
    effective_manifest="$manifest_path"
  fi
fi

report_stem="$(basename "$zarr_path" .zarr)"
mkdir -p "${RUN_DIR}/reports"
cmd=(scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline "$zarr_path" "${SUBJECT_ARGS[@]}"
  --workflow-stage "$WORKFLOW_STAGE"
  --json-report "${RUN_DIR}/reports/${report_stem}_${WORKFLOW_STAGE}.json"
  --markdown-report "${RUN_DIR}/reports/${report_stem}_${WORKFLOW_STAGE}.md")
if [[ -n "$effective_manifest" ]]; then
  cmd+=(--roi-cache-manifest "$effective_manifest")
fi

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "job_index=${LSB_JOBINDEX:-}"
echo "workflow_stage=${WORKFLOW_STAGE}"
echo "threads_per_process=${THREADS_PER_PROCESS}"
echo "zarr_path=${zarr_path}"
echo "roi_cache_manifest=${manifest_path}"
echo "effective_roi_cache_manifest=${effective_manifest}"
printf '+ %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

INFERENCE_BSUB_ARGS=(-J "sm_infer[1-${job_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/infer_%J_%I.out" -eo "${RUN_DIR}/infer_%J_%I.err")
if [[ -n "$QUEUE" ]]; then
  INFERENCE_BSUB_ARGS+=(-q "$QUEUE")
fi
if [[ "$GPUS" != "0" ]]; then
  INFERENCE_BSUB_ARGS+=(-gpu "num=${GPUS}")
fi

FULL_BSUB_ARGS=(-J "sm_batch[1-${job_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/%J_%I.out" -eo "${RUN_DIR}/%J_%I.err")
if [[ -n "$QUEUE" ]]; then
  FULL_BSUB_ARGS+=(-q "$QUEUE")
fi
if [[ "$GPUS" != "0" ]]; then
  FULL_BSUB_ARGS+=(-gpu "num=${GPUS}")
fi

FINALIZE_BSUB_ARGS=(-J "sm_finalize[1-${job_count}]%${FINALIZE_MAX_ACTIVE}" -n "$FINALIZE_NCORES" -R "rusage[mem=${FINALIZE_MEM_GB}G]" -oo "${RUN_DIR}/finalize_%J_%I.out" -eo "${RUN_DIR}/finalize_%J_%I.err")
if [[ -n "$FINALIZE_QUEUE" ]]; then
  FINALIZE_BSUB_ARGS+=(-q "$FINALIZE_QUEUE")
fi

format_bsub_cmd() {
  local mode="$1"
  shift
  local cmd="bsub"
  local arg
  for arg in "$@"; do
    cmd+=" $(printf '%q' "$arg")"
  done
  cmd+=" bash $(printf '%q' "$JOB_SCRIPT") $(printf '%q' "$RUN_DIR") $(printf '%q' "$mode")"
  printf '%s\n' "$cmd"
}

FULL_BSUB_CMD="$(format_bsub_cmd all "${FULL_BSUB_ARGS[@]}")"
INFERENCE_BSUB_CMD="$(format_bsub_cmd inference "${INFERENCE_BSUB_ARGS[@]}")"
FINALIZE_BSUB_CMD_TEMPLATE="$(format_bsub_cmd finalization "${FINALIZE_BSUB_ARGS[@]}" -w "done(<inference_job_id>)")"

echo "Run dir: $RUN_DIR"
echo "Source: $SOURCE"
echo "Root: $ROOT"
echo "Registry: $REGISTRY"
echo "Analysis zarrs: $analysis_count"
echo "Jobs: $job_count (one recording per job)"
echo "Max active: $MAX_ACTIVE"
echo "Queue: ${QUEUE:-<default>}"
echo "Resources: ncores=$NCORES mem_gb=$MEM_GB gpus=$GPUS device=$DEVICE"
echo "Split finalization job: $SPLIT_FINALIZATION_JOB"
echo "Finalizer: chunk_size=$FINALIZE_CHUNK_SIZE workers=$FINALIZE_NUM_WORKERS backend=$FINALIZE_EXECUTION_BACKEND scheduler=$FINALIZE_SCHEDULER"
echo "Refined mask storage: $MASK_STORAGE"
echo "Finalizer resources: queue=${FINALIZE_QUEUE:-<default>} ncores=$FINALIZE_NCORES mem_gb=$FINALIZE_MEM_GB max_active=$FINALIZE_MAX_ACTIVE"
echo "ROI cache root: ${ROI_CACHE_ROOT:-<none>}"
echo "Stage ROI cache to scratch: $STAGE_ROI_CACHE_TO_SCRATCH"
echo "Stage output to scratch: $STAGE_OUTPUT_TO_SCRATCH"
echo "Stage finalization input to scratch: $STAGE_FINALIZATION_INPUT_TO_SCRATCH"
echo "Handoff package dir: ${HANDOFF_PACKAGE_DIR:-<none>}"
echo "Manifest file: $RUN_DIR/targets.tsv"
if [[ "$SPLIT_FINALIZATION_JOB" == "1" ]]; then
  echo "Inference submit command: $INFERENCE_BSUB_CMD"
  echo "Finalization submit command template: $FINALIZE_BSUB_CMD_TEMPLATE"
else
  echo "Submit command: $FULL_BSUB_CMD"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster?" >&2
  exit 2
fi

if [[ "$SPLIT_FINALIZATION_JOB" == "1" ]]; then
  inference_submit_output="$(bsub "${INFERENCE_BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR" inference)"
  echo "$inference_submit_output"
  inference_job_id="$(printf '%s\n' "$inference_submit_output" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  if [[ -z "$inference_job_id" ]]; then
    echo "Could not parse inference job id from bsub output." >&2
    exit 2
  fi
  finalization_dependency="done(${inference_job_id})"
  finalization_submit_output="$(bsub "${FINALIZE_BSUB_ARGS[@]}" -w "$finalization_dependency" bash "$JOB_SCRIPT" "$RUN_DIR" finalization)"
  echo "$finalization_submit_output"
else
  bsub "${FULL_BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR" all
fi
