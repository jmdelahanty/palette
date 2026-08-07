#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
PATH_CONTAINS=""
LIMIT=0
CONFIG="configs/fisheye/import_local.yaml"
TARGET_SAMPLED_FRAMES="200"
FRAME_STEP=""
SKIP_TAIL_FRAMES="200"
DECODE_BACKEND="pynvvc-luma"
GPU_ID=0
OVERWRITE=0
REGISTER=1
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
IMPORT_STIMULUS=1
STIMULUS_QUIET=1
STIMULUS_ALWAYS=0
STIMULUS_RUN_NAME=""
STIMULUS_OVERWRITE=0
INCLUDE_ACQUISITION_CROP_VIDEO=0
ACQUISITION_CROP_RUN_PREFIX="crop_acquisition_crop_video_training"
OVERWRITE_ACQUISITION_CROP_RUN=0

QUEUE="gpu_l4"
GPU_SPEC="num=1:mode=exclusive_process"
NCORES=2
MEM_GB=16
WALLTIME="2:00"
MAX_ACTIVE=4

LOG_DIR=""
RUN_ID=""
REPO_DIR=""
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_import_recordings_training_bsub.sh [options]

Submit sampled training-Zarr imports to LSF, one recording per array task.
The task command uses fisheye.utils.import_recordings_training and defaults to
the PyNvVC luma decode path.

Discovery/import:
  --root PATH               Recording root (default: /groups/.../recordings)
  --path-contains STR       Keep recordings whose path contains this substring
  --limit N                 Limit planned recordings after filtering; 0 means no limit
  --config PATH             Import config YAML (default: configs/fisheye/import_local.yaml)
  --target-sampled-frames N Target sampled frames per recording (default: 200)
  --frame-step N            Fixed frame step instead of target-sampled-frames
  --skip-tail-frames N      Frames to skip at EOF (default: 200)
  --decode-backend NAME     Must be pynvvc-luma; retained for command compatibility
  --gpu-id N                PyNvVC GPU id visible inside the job (default: 0)
  --overwrite               Retired; exits with an error. Publish a new version.

Registry/stimulus:
  --registry PATH           Registry sqlite path
  --no-register             Do not register created training Zarrs
  --no-import-stimulus      Do not mirror stimulus into the training Zarr
  --stimulus-always         Import stimulus even if already present
  --stimulus-run-name NAME  Optional stimulus run name
  --stimulus-overwrite      Overwrite an existing stimulus run name
  --no-stimulus-quiet       Do not suppress verbose stimulus import output
  --include-acquisition-crop-video
                            Append sampled acquisition crop-video frames into
                            crop_runs/<run> in the same *_training.zarr
  --acquisition-crop-run-prefix NAME
                            Prefix for generated crop-video crop run names
  --overwrite-acquisition-crop-run
                            Overwrite generated crop-video crop run if present

LSF resources:
  --queue NAME              LSF queue (default: gpu_l4)
  --gpu SPEC                LSF GPU resource spec (default: num=1:mode=exclusive_process)
  --no-gpu                  Do not request an LSF GPU resource
  --ncores N                CPU slots per task (default: 2)
  --mem-gb N                Memory resource in GB (default: 16)
  --walltime H:MM           Wall time per task (default: 2:00)
  --max-active N            Max concurrent array tasks (default: 4)

General:
  --log-dir PATH            Submission log dir (default: <root>/logs/import_recordings_training/bsub_submissions)
  --run-id ID               Stable run id; default UTC timestamp
  --palette-repo PATH       Immutable Palette checkout visible to compute nodes
                            (default: current directory); its full commit is
                            recorded and verified by every task
  --repo-dir PATH           Compatibility alias for --palette-repo
  --submit                  Actually call bsub. Without this, dry-run only.
  -h, --help                Show this message

Example smoke:
  scripts/submit_import_recordings_training_bsub.sh \
    --path-contains GoodCopBadCop \
    --target-sampled-frames 200 \
    --limit 1 \
    --submit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --target-sampled-frames) TARGET_SAMPLED_FRAMES="$2"; shift 2;;
    --frame-step) FRAME_STEP="$2"; shift 2;;
    --skip-tail-frames) SKIP_TAIL_FRAMES="$2"; shift 2;;
    --decode-backend) DECODE_BACKEND="$2"; shift 2;;
    --gpu-id) GPU_ID="$2"; shift 2;;
    --overwrite) OVERWRITE=1; shift;;
    --registry) REGISTRY="$2"; shift 2;;
    --no-register) REGISTER=0; shift;;
    --no-import-stimulus) IMPORT_STIMULUS=0; shift;;
    --stimulus-always) STIMULUS_ALWAYS=1; shift;;
    --stimulus-run-name) STIMULUS_RUN_NAME="$2"; shift 2;;
    --stimulus-overwrite) STIMULUS_OVERWRITE=1; shift;;
    --no-stimulus-quiet) STIMULUS_QUIET=0; shift;;
    --include-acquisition-crop-video) INCLUDE_ACQUISITION_CROP_VIDEO=1; shift;;
    --acquisition-crop-run-prefix) ACQUISITION_CROP_RUN_PREFIX="$2"; shift 2;;
    --overwrite-acquisition-crop-run) OVERWRITE_ACQUISITION_CROP_RUN=1; shift;;
    --queue) QUEUE="$2"; shift 2;;
    --gpu) GPU_SPEC="$2"; shift 2;;
    --no-gpu) GPU_SPEC=""; shift;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --max-active) MAX_ACTIVE="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --palette-repo|--repo-dir) REPO_DIR="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ "$OVERWRITE" == "1" ]]; then
  echo "--overwrite is retired; publish a new versioned training artifact instead" >&2
  exit 2
fi

if [[ "$DECODE_BACKEND" != "pynvvc-luma" ]]; then
  echo "Unsupported import backend: $DECODE_BACKEND (only pynvvc-luma is supported)" >&2
  exit 2
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT%/}/logs/import_recordings_training/bsub_submissions"
fi
if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$(pwd)"
fi
REPO_DIR="$(realpath -- "$REPO_DIR")"
[[ -x "$REPO_DIR/scripts/py" ]] || {
  echo "Palette repository lacks executable scripts/py: $REPO_DIR" >&2
  exit 2
}
PALETTE_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
[[ -z "$(git -C "$REPO_DIR" status --porcelain --untracked-files=all)" ]] || {
  echo "Palette repository must be clean for cluster submission: $REPO_DIR" >&2
  exit 2
}

RUN_DIR="${LOG_DIR%/}/import_training_${RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

TARGETS_FILE="${RUN_DIR}/recording_dirs.txt"
PLAN_JSON="${RUN_DIR}/plan.json"
scripts/py - "$ROOT" "$PATH_CONTAINS" "$LIMIT" "$CONFIG" "$TARGET_SAMPLED_FRAMES" "$FRAME_STEP" \
  "$SKIP_TAIL_FRAMES" "$DECODE_BACKEND" "$OVERWRITE" "$TARGETS_FILE" "$PLAN_JSON" <<'PY'
import json
import sys
from pathlib import Path

from fisheye.utils import import_recordings_training as mod

(
    root,
    path_contains,
    limit,
    config,
    target_sampled_frames,
    frame_step,
    skip_tail_frames,
    decode_backend,
    overwrite,
    targets_file,
    plan_json,
) = sys.argv[1:]

root_path = Path(root).expanduser().resolve()
limit_value = int(limit)
requested_frame_step = int(frame_step) if frame_step else None
target_value = int(target_sampled_frames) if target_sampled_frames else None
skip_tail = int(skip_tail_frames)
plans = mod._build_plans(
    root_path,
    recursive=False,
    skip_existing=(overwrite != "1"),
    check_stimulus=False,
    requested_frame_step=requested_frame_step,
    target_sampled_frames=target_value,
    skip_tail_frames=skip_tail,
    path_contains=path_contains or None,
    limit=limit_value if limit_value > 0 else None,
    require_source_frame_count=decode_backend == mod.DECODE_BACKEND_PYNVVC_LUMA,
)

ok_plans = [plan for plan in plans if plan.status == "ok"]
targets = [str(plan.recording_dir) for plan in ok_plans]
Path(targets_file).write_text("\n".join(targets) + ("\n" if targets else ""), encoding="utf-8")

payload = {
    "root": str(root_path),
    "path_contains": path_contains or None,
    "limit": limit_value,
    "config": config,
    "decode_backend": decode_backend,
    "target_sampled_frames": target_value,
    "frame_step": requested_frame_step,
    "skip_tail_frames": skip_tail,
    "overwrite": overwrite == "1",
    "planned_count": len(plans),
    "target_count": len(ok_plans),
    "status_counts": {
        status: sum(1 for plan in plans if plan.status == status)
        for status in sorted({plan.status for plan in plans})
    },
    "plans": [
        {
            "recording_dir": str(plan.recording_dir),
            "h5_path": str(plan.h5_path),
            "camera_id": plan.camera_id,
            "cam_video": str(plan.cam_video) if plan.cam_video else None,
            "zarr_path": str(plan.zarr_path),
            "status": plan.status,
            "reason": plan.reason,
            "frame_step": plan.frame_step,
            "source_frame_count": plan.source_frame_count,
            "target_sampled_frames": plan.target_sampled_frames,
            "estimated_sampled_frames": plan.estimated_sampled_frames,
            "frame_count_source": plan.frame_count_source,
            "existing_frame_step": plan.existing_frame_step,
            "frame_step_mismatch": plan.frame_step_mismatch,
        }
        for plan in plans
    ],
}
Path(plan_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

target_count="$(wc -l < "$TARGETS_FILE" | tr -d ' ')"
if [[ "$target_count" == "0" ]]; then
  echo "No importable recording targets found."
  echo "Plan: $PLAN_JSON"
  exit 0
fi

IMPORT_ARGS=(--config "$CONFIG" --decode-backend "$DECODE_BACKEND" --skip-tail-frames "$SKIP_TAIL_FRAMES" --gpu-id "$GPU_ID")
if [[ -n "$FRAME_STEP" ]]; then
  IMPORT_ARGS+=(--frame-step "$FRAME_STEP")
else
  IMPORT_ARGS+=(--target-sampled-frames "$TARGET_SAMPLED_FRAMES")
fi
if [[ "$OVERWRITE" == "1" ]]; then IMPORT_ARGS+=(--overwrite); fi
if [[ "$REGISTER" == "1" ]]; then IMPORT_ARGS+=(--register --registry "$REGISTRY"); fi
if [[ "$IMPORT_STIMULUS" == "1" ]]; then IMPORT_ARGS+=(--import-stimulus); fi
if [[ "$STIMULUS_QUIET" == "1" ]]; then IMPORT_ARGS+=(--stimulus-quiet); fi
if [[ "$STIMULUS_ALWAYS" == "1" ]]; then IMPORT_ARGS+=(--stimulus-always); fi
if [[ -n "$STIMULUS_RUN_NAME" ]]; then IMPORT_ARGS+=(--stimulus-run-name "$STIMULUS_RUN_NAME"); fi
if [[ "$STIMULUS_OVERWRITE" == "1" ]]; then IMPORT_ARGS+=(--stimulus-overwrite); fi
if [[ "$INCLUDE_ACQUISITION_CROP_VIDEO" == "1" ]]; then IMPORT_ARGS+=(--include-acquisition-crop-video --acquisition-crop-run-prefix "$ACQUISITION_CROP_RUN_PREFIX"); fi
if [[ "$OVERWRITE_ACQUISITION_CROP_RUN" == "1" ]]; then IMPORT_ARGS+=(--overwrite-acquisition-crop-run); fi
printf -v IMPORT_ARGS_SHELL '%q ' "${IMPORT_ARGS[@]}"

JOB_SCRIPT="${RUN_DIR}/run_import_training_task.sh"
REPO_DIR_Q="$(printf '%q' "$REPO_DIR")"
PALETTE_COMMIT_Q="$(printf '%q' "$PALETTE_COMMIT")"
ROOT_Q="$(printf '%q' "$ROOT")"
cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="\$1"
TARGETS_FILE="\${RUN_DIR}/recording_dirs.txt"
if [[ -z "\${LSB_JOBINDEX:-}" ]]; then
  echo "LSB_JOBINDEX not set; are you running under bsub array?" >&2
  exit 2
fi
recording_dir="\$(sed -n "\${LSB_JOBINDEX}p" "\$TARGETS_FILE")"
if [[ -z "\$recording_dir" ]]; then
  echo "No recording dir for array index \${LSB_JOBINDEX}" >&2
  exit 2
fi

cd ${REPO_DIR_Q}
expected_palette_commit=${PALETTE_COMMIT_Q}
actual_palette_commit="\$(git rev-parse HEAD)"
if [[ "\$actual_palette_commit" != "\$expected_palette_commit" ]]; then
  echo "Palette commit mismatch: expected \$expected_palette_commit, got \$actual_palette_commit" >&2
  exit 2
fi
echo "repo=\$(pwd)"
echo "palette_commit=\$actual_palette_commit"
echo "host=\$(hostname)"
echo "job_id=\${LSB_JOBID:-unknown}"
echo "job_index=\${LSB_JOBINDEX}"
echo "recording_dir=\$recording_dir"
echo "started_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"

task_log_dir="\${RUN_DIR}/task_logs/\$(printf '%04d' "\${LSB_JOBINDEX}")"
mkdir -p "\$task_log_dir"

node_scratch="\${TMPDIR:?LSF TMPDIR is required for atomic base publication}/palette-sampled-training-base-\${LSB_JOBID:-unknown}-\${LSB_JOBINDEX}"
mkdir -p "\$node_scratch"

scripts/py -m fisheye.utils.import_recordings_training ${ROOT_Q} \
  --path-contains "\$recording_dir" \
  --limit 1 \
  --log-dir "\$task_log_dir" \
  --scratch-root "\$node_scratch" \
  ${IMPORT_ARGS_SHELL}--apply

echo "finished_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

scripts/py - "$RUN_DIR/submission_manifest.json" "$ROOT" "$PATH_CONTAINS" "$LIMIT" "$target_count" \
  "$QUEUE" "$GPU_SPEC" "$NCORES" "$MEM_GB" "$WALLTIME" "$MAX_ACTIVE" "$CONFIG" \
  "$TARGET_SAMPLED_FRAMES" "$FRAME_STEP" "$SKIP_TAIL_FRAMES" "$DECODE_BACKEND" "$GPU_ID" \
  "$OVERWRITE" "$REGISTER" "$REGISTRY" "$IMPORT_STIMULUS" "$STIMULUS_QUIET" "$JOB_SCRIPT" \
  "$TARGETS_FILE" "$PLAN_JSON" "$REPO_DIR" "$INCLUDE_ACQUISITION_CROP_VIDEO" \
  "$ACQUISITION_CROP_RUN_PREFIX" "$OVERWRITE_ACQUISITION_CROP_RUN" \
  "$PALETTE_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    root,
    path_contains,
    limit,
    target_count,
    queue,
    gpu_spec,
    ncores,
    mem_gb,
    walltime,
    max_active,
    config,
    target_sampled_frames,
    frame_step,
    skip_tail_frames,
    decode_backend,
    gpu_id,
    overwrite,
    register,
    registry,
    import_stimulus,
    stimulus_quiet,
    job_script,
    targets_file,
    plan_json,
    repo_dir,
    include_acquisition_crop_video,
    acquisition_crop_run_prefix,
    overwrite_acquisition_crop_run,
    palette_commit,
) = sys.argv[1:]

payload = {
    "schema": "palette.import_recordings_training_bsub_submission.v1",
    "root": root,
    "path_contains": path_contains or None,
    "limit": int(limit),
    "target_count": int(target_count),
    "queue": queue,
    "gpu_spec": gpu_spec or None,
    "ncores": int(ncores),
    "mem_gb": int(mem_gb),
    "walltime": walltime,
    "max_active": int(max_active),
    "config": config,
    "target_sampled_frames": int(target_sampled_frames) if target_sampled_frames else None,
    "frame_step": int(frame_step) if frame_step else None,
    "skip_tail_frames": int(skip_tail_frames),
    "decode_backend": decode_backend,
    "gpu_id": int(gpu_id),
    "overwrite": overwrite == "1",
    "register": register == "1",
    "registry": registry if register == "1" else None,
    "import_stimulus": import_stimulus == "1",
    "stimulus_quiet": stimulus_quiet == "1",
    "include_acquisition_crop_video": include_acquisition_crop_video == "1",
    "acquisition_crop_run_prefix": acquisition_crop_run_prefix,
    "overwrite_acquisition_crop_run": overwrite_acquisition_crop_run == "1",
    "atomic_base_publication": True,
    "job_script": job_script,
    "targets_file": targets_file,
    "plan_json": plan_json,
    "repo_dir": repo_dir,
    "palette_repo": repo_dir,
    "palette_commit": palette_commit,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

BSUB_ARGS=(
  -J "import_training[1-${target_count}]%${MAX_ACTIVE}"
  -q "$QUEUE"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J_%I.out"
  -eo "${RUN_DIR}/%J_%I.err"
)
if [[ -n "$GPU_SPEC" ]]; then
  BSUB_ARGS+=(-gpu "$GPU_SPEC")
fi

BSUB_CMD="bsub"
for arg in "${BSUB_ARGS[@]}"; do
  BSUB_CMD+=" $(printf '%q' "$arg")"
done
BSUB_CMD+=" bash"
BSUB_CMD+=" $(printf '%q' "$JOB_SCRIPT")"
BSUB_CMD+=" $(printf '%q' "$RUN_DIR")"

printf -v IMPORT_CMD 'scripts/py -m fisheye.utils.import_recordings_training %q --path-contains <recording_dir> --limit 1 --log-dir <task_log_dir> %s--apply' "$ROOT" "$IMPORT_ARGS_SHELL"

echo "Run dir: $RUN_DIR"
echo "Targets: $target_count"
echo "Target list: $TARGETS_FILE"
echo "Plan: $PLAN_JSON"
echo "Repo dir: $REPO_DIR"
echo "Palette commit: $PALETTE_COMMIT"
echo "Registry: $([[ "$REGISTER" == "1" ]] && echo "$REGISTRY" || echo "<disabled>")"
echo "Queue: $QUEUE"
echo "GPU: ${GPU_SPEC:-<none>}"
echo "Resources: ncores=$NCORES mem_gb=$MEM_GB walltime=$WALLTIME"
echo "Max active: $MAX_ACTIVE"
echo "Decode backend: $DECODE_BACKEND"
echo "Include acquisition crop video: $([[ "$INCLUDE_ACQUISITION_CROP_VIDEO" == "1" ]] && echo "$ACQUISITION_CROP_RUN_PREFIX" || echo "<disabled>")"
echo "Atomic base publication: node-local checked publish (required)"
echo "Per-target command: $IMPORT_CMD"
echo "Submit command: $BSUB_CMD"

if [[ "$SUBMIT" != "1" ]]; then
  echo "Dry run only; pass --submit to submit."
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Run this on an LSF login node." >&2
  exit 2
fi

bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
