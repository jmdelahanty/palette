#!/usr/bin/env bash
set -euo pipefail

ROOTS=()
ZARRS=()
ZARR_LIST=""
ROOTS_FROM_REPORT=""
PATH_CONTAINS=""
SOURCE="filesystem"
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
ZARR_USE="analysis"
ZARR_ORIGIN=""
RIG_ID=""
ARENA_ID=""
CAMERA_ID=""
PROTOCOL_NAME=""
REQUIRE_STEPS_OK=()
DEPENDENCY_DONE=()

QUEUE=""
NCORES=4
MEM_GB=16
MAX_ACTIVE=8
SUBMIT_REGISTRY_FINALIZER=1
REGISTRY_FINALIZER_MEM_GB=8
REGISTRY_FINALIZER_WALLTIME="1:00"
LOG_DIR="/groups/johnson/johnsonlab/jeremy/recordings/logs/chaser_analytics/bsub_submissions"
RUN_ID_OVERRIDE=""
DRY_RUN=0
PROTOCOL_PROFILE="src/fisheye/analysis/profiles/chaser_event_windows_v1.yaml"
CHASER_ANALYSIS_PROFILE="src/fisheye/analysis/profiles/chaser_behavior_v1.yaml"

RUN_MOVEMENT=1
RUN_STIMULUS_EPOCH=1
RUN_DETECTION_OCCUPANCY=1
RUN_CHASER_DISTANCE=1
RUN_QUADRANT_OCCUPANCY=1
RUN_NEAR_FIELD_OCCUPANCY=1
RUN_EPOCH_BEHAVIOR=1
RUN_EGOCENTRIC=1
RUN_ESCAPE_FREEZE=1
RUN_EYE_ANGLES=1
RUN_GAZE_TRACKING=1

OVERWRITE=0
NO_PNG=0
NO_INTERACTIVE_SPEC=0

TRACK_RUN="track_kinematics_chaser_v1_20260717"
SWIM_BOUT_RUN="swim_bouts_chaser_v1_20260717"
BOUT_KINEMATICS_RUN="bout_kinematics_chaser_v1_20260717"
TRACK_RUN_EXPLICIT=0
SWIM_BOUT_RUN_EXPLICIT=0
BOUT_KINEMATICS_RUN_EXPLICIT=0
INCLUDE_EYE_GAZE=1

EPOCH_RUN="stimulus_epochs_chaser_v1_20260717"
OCCUPANCY_RUN="detection_occupancy_chaser_v1_20260717"
CHASER_DISTANCE_RUN="chaser_distance_chaser_v1_20260717"
EPOCH_RUN_EXPLICIT=0
CHASER_DISTANCE_RUN_EXPLICIT=0
QUADRANT_OCCUPANCY_COMPONENT="quadrant_occupancy_v1_20260717"
NEAR_FIELD_OCCUPANCY_COMPONENT="near_field_occupancy_v1_20260717"
EPOCH_BEHAVIOR_COMPONENT="epoch_behavior_v1_20260717"
EGOCENTRIC_COMPONENT="egocentric_bearing_v1"
ESCAPE_FREEZE_COMPONENT="escape_freeze_chaser_{chaser_index}_v1_20260717"
EYE_ANGLE_OUTPUT_RUN="eye_angles_chaser_gaze_v1_20260714"
EYE_ANGLE_RUN="latest"
GAZE_EGOCENTRIC_COMPONENT="latest"
GAZE_TRACKING_COMPONENT="chaser_gaze_tracking_v1_20260714"

OCCUPANCY_BIN_SIZE=128
OCCUPANCY_SMOOTH_SIGMA=1.0
CHASER_THRESHOLD_MM=20.0
CHASER_DISTRIBUTION_BIN_WIDTH_MM=2.0
TRACK_ID=0
# Optional physical track-kinematics override for epoch speed/path summaries.
# Empty means follow the persisted swim-bout detector signal's physical source
# level (commonly speed_exponential derived from speed_filtered).
SPEED_LEVEL=""
EGOCENTRIC_DISTANCE_BIN_WIDTH_MM=""
EGOCENTRIC_BEARING_BIN_WIDTH_DEG=""
ESCAPE_CHASER_INDEX="all_applicable"
ESCAPE_TRIGGER_RADIUS_MM=""
ESCAPE_PATH_THRESHOLD_MM=""

apply_preset() {
  case "$1" in
    chaser_v1)
      TRACK_RUN="track_kinematics_chaser_v1_20260717"
      SWIM_BOUT_RUN="swim_bouts_chaser_v1_20260717"
      BOUT_KINEMATICS_RUN="bout_kinematics_chaser_v1_20260717"
      EPOCH_RUN="stimulus_epochs_chaser_v1_20260717"
      OCCUPANCY_RUN="detection_occupancy_chaser_v1_20260717"
      CHASER_DISTANCE_RUN="chaser_distance_chaser_v1_20260717"
      QUADRANT_OCCUPANCY_COMPONENT="quadrant_occupancy_v1_20260717"
      NEAR_FIELD_OCCUPANCY_COMPONENT="near_field_occupancy_v1_20260717"
      EPOCH_BEHAVIOR_COMPONENT="epoch_behavior_v1_20260717"
      EGOCENTRIC_COMPONENT="egocentric_bearing_v1_20260717"
      ESCAPE_FREEZE_COMPONENT="escape_freeze_chaser_{chaser_index}_v1_20260717"
      ;;
    goodcopbadcop)
      TRACK_RUN="tk_hyst4_low2_latch_s005"
      SWIM_BOUT_RUN="bouts_tk_hyst4_low2_latch_s005_peak_event_exp_tau025_prom4_dist010_w098"
      BOUT_KINEMATICS_RUN="bk_tk_hyst4_low2_latch_s005_peak_event_prom4_w098_interbout"
      EPOCH_RUN="goodcopbadcop_stimulus_epochs_v1_20260617"
      OCCUPANCY_RUN="goodcopbadcop_detection_occupancy_v1_20260617"
      CHASER_DISTANCE_RUN="goodcopbadcop_chaser_distance_v1_20260617"
      QUADRANT_OCCUPANCY_COMPONENT="quadrant_occupancy_v1_20260717"
      NEAR_FIELD_OCCUPANCY_COMPONENT="near_field_occupancy_v1_20260717"
      EPOCH_BEHAVIOR_COMPONENT="epoch_behavior_v1_20260717"
      EGOCENTRIC_COMPONENT="egocentric_bearing_v1"
      ESCAPE_FREEZE_COMPONENT="escape_freeze_chaser_{chaser_index}_v1_20260717"
      ;;
    redscare)
      TRACK_RUN="tk_hyst4_low2_latch_s005_redscare_v1_20260708"
      SWIM_BOUT_RUN="bouts_tk_hyst4_low2_latch_s005_peak_event_exp_tau025_prom4_dist010_w098_redscare_v1_20260708"
      BOUT_KINEMATICS_RUN="bk_tk_hyst4_low2_latch_s005_peak_event_prom4_w098_interbout_redscare_v1_20260708"
      EPOCH_RUN="stimulus_epochs_redscare_v1_20260708"
      OCCUPANCY_RUN="detection_occupancy_redscare_v1_20260708"
      CHASER_DISTANCE_RUN="chaser_distance_redscare_v1_20260708"
      QUADRANT_OCCUPANCY_COMPONENT="quadrant_occupancy_redscare_v1_20260708"
      NEAR_FIELD_OCCUPANCY_COMPONENT="near_field_occupancy_redscare_v1_20260708"
      EPOCH_BEHAVIOR_COMPONENT="kinematics_bouts_v1"
      EGOCENTRIC_COMPONENT="egocentric_bearing_redscare_v1_20260708"
      ESCAPE_FREEZE_COMPONENT="escape_freeze_chaser_{chaser_index}_redscare_v1_20260708"
      ;;
    *)
      echo "Unknown --preset: $1" >&2
      echo "Known presets: chaser_v1, goodcopbadcop, redscare" >&2
      exit 2
      ;;
  esac
}

usage() {
  cat <<'USAGE'
Usage: submit_chaser_analytics_bsub.sh [options]

Submit one CPU LSF array task per analysis zarr. Each task runs the selected
generic chaser modules serially, avoiding concurrent writes to the same store.

Target selection, choose at least one, or use --source registry:
  --zarr PATH                 Add one explicit analysis zarr. May be repeated.
  --zarr-list PATH            Newline-delimited analysis-zarr list.
  --roots-from-report PATH    JSON report with results/plans rows containing zarr_path.
  --root PATH                 Recursively discover *_analysis.zarr under PATH. May repeat.
  --path-contains STR         Optional substring filter on discovered/listed zarr paths.
  --source {filesystem,registry}
                              Discovery source for --root targets (default: filesystem).
                              Registry mode requires normalized chaser metadata and
                              snapshots matching rows before bsub.
  --registry PATH             Registry sqlite path for --source registry.
  --zarr-use NAME             Registry zarr_use filter (default: analysis).
  --zarr-origin NAME          Registry origin filter, usually source or derived.
  --rig-id ID                 Registry rig_id filter.
  --arena-id ID               Registry arena_id filter.
  --camera-id ID              Registry camera_id filter.
  --protocol-name NAME        Exact, case-insensitive registry protocol filter.
  --require-step-ok STEP      Registry step-status filter. May be repeated.
  --dependency-done JOBID     Submit only after an LSF job completes successfully.
                              May be repeated.
  --palette-repo PATH         Clean cluster-visible Palette checkout.
  --submit-host HOST          Citrus SSH poller when bsub is unavailable locally.
  --protocol-profile PATH     Versioned protocol adapter/window profile.
  --analysis-profile PATH     Versioned generic chaser-analysis profile.

LSF options:
  --queue NAME                LSF queue. If omitted, bsub default queue is used.
  --ncores N                  Cores per array task (default: 4).
  --mem-gb N                  Memory per task in GB (default: 16).
  --max-active N              LSF array concurrency, %[N] (default: 8).
  --no-registry-finalizer     Leave completion receipts unapplied; array workers
                              still never write SQLite directly.
  --registry-finalizer-mem-gb N
  --registry-finalizer-walltime H:MM
  --log-dir PATH              Submission run dir parent.
  --run-id ID                 Stable run id. Defaults to UTC timestamp.
  --dry-run                   Write manifest/job script and print bsub command; do not submit.

Authoritative input resolution:
  When a producer stage is skipped and its corresponding --*-run selector was
  not explicitly supplied, each task resolves that input from the recording's
  authoritative latest-complete pointer. Explicit selectors always win. This
  permits registry cohorts whose completed prerequisite run names differ.

Stage toggles:
  --skip-movement             Skip arena assignment, track kinematics, swim bouts, bout kinematics.
  --skip-stimulus-epoch       Skip stimulus epoch materialization.
  --skip-detection-occupancy  Skip detection occupancy materialization.
  --skip-chaser-distance      Skip chaser-distance materialization.
  --skip-quadrant-occupancy   Skip chaser quadrant occupancy.
  --skip-near-field-occupancy Skip chaser near-field occupancy.
  --skip-epoch-behavior       Skip epoch behavior/bout distribution summary.
  --skip-egocentric           Skip fish-centered egocentric chaser bearing.
  --skip-escape-freeze        Skip per-applicable-chaser escape/freeze summaries.
  --run-eye-angles            Run/reuse the DAG eye-angle target (default).
  --skip-eye-angles           Skip the separate DAG eye-angle target.
  --run-gaze-tracking         Validate conventions and write chaser gaze tracking (default).
  --skip-gaze-tracking        Skip chaser-relative gaze tracking.
  --eye-and-gaze-only         Disable other modules; run eye angles then gaze tracking.
  --gaze-only                 Disable other modules; validate/use an existing eye run.

Run names and components:
  --preset NAME               Apply chaser_v1, goodcopbadcop, or redscare.
                              chaser_v1 is the generic default.
                              Options are parsed in order; later explicit names win.
  --track-run NAME
  --swim-bout-run NAME
  --bout-kinematics-run NAME
  --epoch-run NAME
  --occupancy-run NAME
  --chaser-distance-run NAME
  --quadrant-occupancy-component NAME
  --near-field-occupancy-component NAME
  --epoch-behavior-component NAME
  --egocentric-component NAME
  --escape-freeze-component NAME
  --eye-angle-output-run NAME Output run used when the DAG must create eye angles.
  --eye-angle-run NAME        Eye run validated/used for gaze (default: latest).
  --gaze-egocentric-component NAME
  --gaze-tracking-component NAME

Parameters:
  --overwrite                 Overwrite existing runs/components where supported.
  --no-png                    Skip PNG artifacts where supported.
  --no-interactive-spec       Skip interactive specs where supported.
  --include-eye-gaze          Include eye-gaze bout metrics (default).
  --no-include-eye-gaze       Explicitly omit eye-gaze bout metrics for compatibility.
  --occupancy-bin-size N
  --occupancy-smooth-sigma X
  --chaser-threshold-mm X
  --chaser-distribution-bin-width-mm X
  --track-id N
  --speed-level NAME          Physical track speed for epoch summaries
                              (default: detector signal's persisted physical source;
                              does not select the bout detector signal).
  --egocentric-distance-bin-width-mm X
  --egocentric-bearing-bin-width-deg X
  --escape-chaser-index N     Selected chaser, or all_applicable (default).
  --escape-trigger-radius-mm X
  --escape-path-threshold-mm X
  -h, --help                  Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARRS+=("$2"); shift 2;;
    --zarr-list) ZARR_LIST="$2"; shift 2;;
    --roots-from-report) ROOTS_FROM_REPORT="$2"; shift 2;;
    --root) ROOTS+=("$2"); shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --zarr-use) ZARR_USE="$2"; shift 2;;
    --zarr-origin) ZARR_ORIGIN="$2"; shift 2;;
    --rig-id) RIG_ID="$2"; shift 2;;
    --arena-id) ARENA_ID="$2"; shift 2;;
    --camera-id) CAMERA_ID="$2"; shift 2;;
    --protocol-name) PROTOCOL_NAME="$2"; shift 2;;
    --require-step-ok) REQUIRE_STEPS_OK+=("$2"); shift 2;;
    --dependency-done) DEPENDENCY_DONE+=("$2"); shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --protocol-profile) PROTOCOL_PROFILE="$2"; shift 2;;
    --analysis-profile) CHASER_ANALYSIS_PROFILE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --max-active) MAX_ACTIVE="$2"; shift 2;;
    --no-registry-finalizer) SUBMIT_REGISTRY_FINALIZER=0; shift;;
    --registry-finalizer-mem-gb) REGISTRY_FINALIZER_MEM_GB="$2"; shift 2;;
    --registry-finalizer-walltime) REGISTRY_FINALIZER_WALLTIME="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID_OVERRIDE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --skip-movement) RUN_MOVEMENT=0; shift;;
    --skip-stimulus-epoch) RUN_STIMULUS_EPOCH=0; shift;;
    --skip-detection-occupancy) RUN_DETECTION_OCCUPANCY=0; shift;;
    --skip-chaser-distance) RUN_CHASER_DISTANCE=0; shift;;
    --skip-quadrant-occupancy) RUN_QUADRANT_OCCUPANCY=0; shift;;
    --skip-near-field-occupancy) RUN_NEAR_FIELD_OCCUPANCY=0; shift;;
    --skip-epoch-behavior) RUN_EPOCH_BEHAVIOR=0; shift;;
    --skip-egocentric) RUN_EGOCENTRIC=0; shift;;
    --skip-escape-freeze) RUN_ESCAPE_FREEZE=0; shift;;
    --run-eye-angles) RUN_EYE_ANGLES=1; shift;;
    --skip-eye-angles) RUN_EYE_ANGLES=0; shift;;
    --run-gaze-tracking) RUN_GAZE_TRACKING=1; shift;;
    --skip-gaze-tracking) RUN_GAZE_TRACKING=0; shift;;
    --eye-and-gaze-only)
      RUN_MOVEMENT=0; RUN_STIMULUS_EPOCH=0; RUN_DETECTION_OCCUPANCY=0
      RUN_CHASER_DISTANCE=0; RUN_QUADRANT_OCCUPANCY=0; RUN_NEAR_FIELD_OCCUPANCY=0
      RUN_EPOCH_BEHAVIOR=0; RUN_EGOCENTRIC=0; RUN_ESCAPE_FREEZE=0
      RUN_EYE_ANGLES=1; RUN_GAZE_TRACKING=1; shift;;
    --gaze-only)
      RUN_MOVEMENT=0; RUN_STIMULUS_EPOCH=0; RUN_DETECTION_OCCUPANCY=0
      RUN_CHASER_DISTANCE=0; RUN_QUADRANT_OCCUPANCY=0; RUN_NEAR_FIELD_OCCUPANCY=0
      RUN_EPOCH_BEHAVIOR=0; RUN_EGOCENTRIC=0; RUN_ESCAPE_FREEZE=0
      RUN_EYE_ANGLES=0; RUN_GAZE_TRACKING=1; shift;;
    --preset) apply_preset "$2"; shift 2;;
    --track-run) TRACK_RUN="$2"; TRACK_RUN_EXPLICIT=1; shift 2;;
    --swim-bout-run) SWIM_BOUT_RUN="$2"; SWIM_BOUT_RUN_EXPLICIT=1; shift 2;;
    --bout-kinematics-run) BOUT_KINEMATICS_RUN="$2"; BOUT_KINEMATICS_RUN_EXPLICIT=1; shift 2;;
    --epoch-run) EPOCH_RUN="$2"; EPOCH_RUN_EXPLICIT=1; shift 2;;
    --occupancy-run) OCCUPANCY_RUN="$2"; shift 2;;
    --chaser-distance-run) CHASER_DISTANCE_RUN="$2"; CHASER_DISTANCE_RUN_EXPLICIT=1; shift 2;;
    --quadrant-occupancy-component) QUADRANT_OCCUPANCY_COMPONENT="$2"; shift 2;;
    --near-field-occupancy-component) NEAR_FIELD_OCCUPANCY_COMPONENT="$2"; shift 2;;
    --epoch-behavior-component) EPOCH_BEHAVIOR_COMPONENT="$2"; shift 2;;
    --egocentric-component) EGOCENTRIC_COMPONENT="$2"; shift 2;;
    --escape-freeze-component) ESCAPE_FREEZE_COMPONENT="$2"; shift 2;;
    --eye-angle-output-run) EYE_ANGLE_OUTPUT_RUN="$2"; shift 2;;
    --eye-angle-run) EYE_ANGLE_RUN="$2"; shift 2;;
    --gaze-egocentric-component) GAZE_EGOCENTRIC_COMPONENT="$2"; shift 2;;
    --gaze-tracking-component) GAZE_TRACKING_COMPONENT="$2"; shift 2;;
    --overwrite) OVERWRITE=1; shift;;
    --no-png) NO_PNG=1; shift;;
    --no-interactive-spec) NO_INTERACTIVE_SPEC=1; shift;;
    --include-eye-gaze) INCLUDE_EYE_GAZE=1; shift;;
    --no-include-eye-gaze) INCLUDE_EYE_GAZE=0; shift;;
    --occupancy-bin-size) OCCUPANCY_BIN_SIZE="$2"; shift 2;;
    --occupancy-smooth-sigma) OCCUPANCY_SMOOTH_SIGMA="$2"; shift 2;;
    --chaser-threshold-mm) CHASER_THRESHOLD_MM="$2"; shift 2;;
    --chaser-distribution-bin-width-mm) CHASER_DISTRIBUTION_BIN_WIDTH_MM="$2"; shift 2;;
    --track-id) TRACK_ID="$2"; shift 2;;
    --speed-level) SPEED_LEVEL="$2"; shift 2;;
    --egocentric-distance-bin-width-mm) EGOCENTRIC_DISTANCE_BIN_WIDTH_MM="$2"; shift 2;;
    --egocentric-bearing-bin-width-deg) EGOCENTRIC_BEARING_BIN_WIDTH_DEG="$2"; shift 2;;
    --escape-chaser-index) ESCAPE_CHASER_INDEX="$2"; shift 2;;
    --escape-trigger-radius-mm) ESCAPE_TRIGGER_RADIUS_MM="$2"; shift 2;;
    --escape-path-threshold-mm) ESCAPE_PATH_THRESHOLD_MM="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

TRACK_RUN_SELECTION="configured_output"
SWIM_BOUT_RUN_SELECTION="configured_output"
BOUT_KINEMATICS_RUN_SELECTION="configured_output"
EPOCH_RUN_SELECTION="configured_output"
CHASER_DISTANCE_RUN_SELECTION="configured_output"

if [[ "$RUN_MOVEMENT" == "0" ]]; then
  TRACK_RUN_SELECTION="explicit_reuse"
  SWIM_BOUT_RUN_SELECTION="explicit_reuse"
  BOUT_KINEMATICS_RUN_SELECTION="explicit_reuse"
  if [[ "$TRACK_RUN_EXPLICIT" == "0" ]]; then
    TRACK_RUN="latest"
    TRACK_RUN_SELECTION="authoritative_latest_complete"
  fi
  if [[ "$SWIM_BOUT_RUN_EXPLICIT" == "0" ]]; then
    SWIM_BOUT_RUN="latest"
    SWIM_BOUT_RUN_SELECTION="authoritative_latest_complete"
  fi
  if [[ "$BOUT_KINEMATICS_RUN_EXPLICIT" == "0" ]]; then
    BOUT_KINEMATICS_RUN="latest"
    BOUT_KINEMATICS_RUN_SELECTION="authoritative_latest_complete"
  fi
fi
if [[ "$RUN_STIMULUS_EPOCH" == "0" ]]; then
  EPOCH_RUN_SELECTION="explicit_reuse"
  if [[ "$EPOCH_RUN_EXPLICIT" == "0" ]]; then
    EPOCH_RUN="latest"
    EPOCH_RUN_SELECTION="authoritative_latest_complete"
  fi
fi
if [[ "$RUN_CHASER_DISTANCE" == "0" ]]; then
  CHASER_DISTANCE_RUN_SELECTION="explicit_reuse"
  if [[ "$CHASER_DISTANCE_RUN_EXPLICIT" == "0" ]]; then
    CHASER_DISTANCE_RUN="latest"
    CHASER_DISTANCE_RUN_SELECTION="authoritative_latest_complete"
  fi
fi

if [[ "$SOURCE" != "filesystem" && "$SOURCE" != "registry" ]]; then
  echo "--source must be filesystem or registry, got: $SOURCE" >&2
  exit 2
fi
if [[ ! "$REGISTRY_FINALIZER_MEM_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "--registry-finalizer-mem-gb must be a positive integer" >&2
  exit 2
fi
for dependency_job_id in "${DEPENDENCY_DONE[@]}"; do
  if [[ ! "$dependency_job_id" =~ ^[1-9][0-9]*$ ]]; then
    echo "--dependency-done must be a positive numeric LSF job ID: $dependency_job_id" >&2
    exit 2
  fi
done

if [[ ! -d "$PALETTE_REPO/.git" || ! -x "$PALETTE_REPO/scripts/py" ]]; then
  echo "Cluster-visible Palette checkout is unavailable: $PALETTE_REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$PALETTE_REPO" status --porcelain)" ]]; then
  echo "Cluster-visible Palette checkout must be clean: $PALETTE_REPO" >&2
  exit 2
fi
EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"

if [[ "$PROTOCOL_PROFILE" = /* ]]; then
  PROTOCOL_PROFILE_RESOLVED="$PROTOCOL_PROFILE"
else
  PROTOCOL_PROFILE_RESOLVED="$PALETTE_REPO/$PROTOCOL_PROFILE"
fi
if [[ "$CHASER_ANALYSIS_PROFILE" = /* ]]; then
  CHASER_ANALYSIS_PROFILE_RESOLVED="$CHASER_ANALYSIS_PROFILE"
else
  CHASER_ANALYSIS_PROFILE_RESOLVED="$PALETTE_REPO/$CHASER_ANALYSIS_PROFILE"
fi
if [[ ! -f "$PROTOCOL_PROFILE_RESOLVED" ]]; then
  echo "Protocol profile does not exist: $PROTOCOL_PROFILE_RESOLVED" >&2
  exit 2
fi
if [[ ! -f "$CHASER_ANALYSIS_PROFILE_RESOLVED" ]]; then
  echo "Chaser analysis profile does not exist: $CHASER_ANALYSIS_PROFILE_RESOLVED" >&2
  exit 2
fi

if [[ "${#ZARRS[@]}" -eq 0 && -z "$ZARR_LIST" && -z "$ROOTS_FROM_REPORT" && "${#ROOTS[@]}" -eq 0 && "$SOURCE" != "registry" ]]; then
  echo "No targets provided. Pass --zarr, --zarr-list, --roots-from-report, or --root." >&2
  exit 2
fi

if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
RUN_DIR="${LOG_DIR}/chaser_analytics_${RUN_ID}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or remove the existing run directory." >&2
  exit 2
fi
if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  if [[ "$DRY_RUN" == "1" ]]; then
    FALLBACK_LOG_DIR="${TMPDIR:-/tmp}/palette/chaser_analytics/bsub_submissions"
    RUN_DIR="${FALLBACK_LOG_DIR}/chaser_analytics_${RUN_ID}"
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

mkdir -p "$RUN_DIR/json" "$RUN_DIR/status"

env PYTHONPATH="$PALETTE_REPO/src" "$PALETTE_REPO/scripts/py" - \
  "$PROTOCOL_PROFILE_RESOLVED" \
  "$CHASER_ANALYSIS_PROFILE_RESOLVED" \
  "$RUN_DIR/profiles.json" <<'PY'
import json
import sys
from pathlib import Path

from fisheye.analysis.chaser_profiles import (
    load_chaser_analysis_profile,
    load_chaser_protocol_profile,
)

protocol = load_chaser_protocol_profile(Path(sys.argv[1]))
analysis = load_chaser_analysis_profile(Path(sys.argv[2]))
payload = {
    "protocol_profile": protocol.to_dict(),
    "protocol_profile_sha256": protocol.sha256,
    "analysis_profile": analysis.to_dict(),
    "analysis_profile_sha256": analysis.sha256,
}
Path(sys.argv[3]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

TARGET_ARGS=()
TARGET_ARGS+=(--source "$SOURCE")
if [[ "$SOURCE" == "registry" ]]; then
  TARGET_ARGS+=(--registry "$REGISTRY" --zarr-use "$ZARR_USE" --require-chaser-metadata)
  if [[ -n "$ZARR_ORIGIN" ]]; then TARGET_ARGS+=(--zarr-origin "$ZARR_ORIGIN"); fi
  if [[ -n "$RIG_ID" ]]; then TARGET_ARGS+=(--rig-id "$RIG_ID"); fi
  if [[ -n "$ARENA_ID" ]]; then TARGET_ARGS+=(--arena-id "$ARENA_ID"); fi
  if [[ -n "$CAMERA_ID" ]]; then TARGET_ARGS+=(--camera-id "$CAMERA_ID"); fi
  if [[ -n "$PROTOCOL_NAME" ]]; then TARGET_ARGS+=(--protocol-name "$PROTOCOL_NAME"); fi
  for step in "${REQUIRE_STEPS_OK[@]}"; do
    TARGET_ARGS+=(--require-step-ok "$step")
  done
fi
for zarr_path in "${ZARRS[@]}"; do
  TARGET_ARGS+=(--zarr "$zarr_path")
done
if [[ -n "$ZARR_LIST" ]]; then
  TARGET_ARGS+=(--zarr-list "$ZARR_LIST")
fi
if [[ -n "$ROOTS_FROM_REPORT" ]]; then
  TARGET_ARGS+=(--roots-from-report "$ROOTS_FROM_REPORT")
fi
for root in "${ROOTS[@]}"; do
  TARGET_ARGS+=(--root "$root")
done
if [[ -n "$PATH_CONTAINS" ]]; then
  TARGET_ARGS+=(--path-contains "$PATH_CONTAINS")
fi

scripts/py - "$RUN_DIR/recordings.txt" "${TARGET_ARGS[@]}" <<'PY'
import argparse
import json
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--source", choices=["filesystem", "registry"], default="filesystem")
parser.add_argument("--registry", type=Path)
parser.add_argument("--zarr-use", default="analysis")
parser.add_argument("--zarr-origin")
parser.add_argument("--rig-id")
parser.add_argument("--arena-id")
parser.add_argument("--camera-id")
parser.add_argument("--protocol-name")
parser.add_argument("--require-chaser-metadata", action="store_true")
parser.add_argument("--require-step-ok", action="append", default=[])
parser.add_argument("--zarr", action="append", default=[])
parser.add_argument("--zarr-list", type=Path)
parser.add_argument("--roots-from-report", type=Path)
parser.add_argument("--root", action="append", default=[])
parser.add_argument("--path-contains", default="")
args = parser.parse_args()

paths: list[Path] = []
for raw in args.zarr:
    paths.append(Path(raw).expanduser())

if args.zarr_list:
    for line in args.zarr_list.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            paths.append(Path(text).expanduser())

if args.roots_from_report:
    payload = json.loads(args.roots_from_report.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        rows = payload.get("plans")
    if not isinstance(rows, list):
        raise ValueError(
            f"{args.roots_from_report} does not contain a results or plans list."
        )
    for row in rows:
        if isinstance(row, dict) and row.get("zarr_path"):
            paths.append(Path(str(row["zarr_path"])).expanduser())

root_paths = [Path(raw_root).expanduser() for raw_root in args.root]

if args.source == "registry":
    if args.registry is None:
        raise ValueError("--source registry requires --registry PATH")
    from fisheye.shared.zarr_discovery import discover_registry_zarrs

    paths.extend(
        discover_registry_zarrs(
            registry_path=args.registry,
            scope_paths=root_paths,
            zarr_use=args.zarr_use,
            zarr_origin=args.zarr_origin,
            rig_id=args.rig_id,
            arena_id=args.arena_id,
            camera_id=args.camera_id,
            protocol_name=args.protocol_name,
            require_chaser_metadata=args.require_chaser_metadata,
            path_contains=args.path_contains or None,
            require_steps_ok=args.require_step_ok or None,
            zarr_suffix="_analysis.zarr" if args.zarr_use == "analysis" else ".zarr",
        )
    )
else:
    for root in root_paths:
        candidates = [root] if root.name.endswith("_analysis.zarr") else sorted(root.rglob("*_analysis.zarr"))
        paths.extend(candidates)

seen: set[str] = set()
selected: list[str] = []
for path in paths:
    text = str(path)
    if args.path_contains and args.path_contains not in text:
        continue
    if not text.endswith(".zarr"):
        continue
    try:
        key = str(path.resolve())
    except OSError:
        key = text
    if key in seen:
        continue
    seen.add(key)
    selected.append(key)

args.output.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
summary = {
    "source": args.source,
    "registry": str(args.registry) if args.registry is not None else None,
    "zarr_use": args.zarr_use,
    "zarr_origin": args.zarr_origin,
    "rig_id": args.rig_id,
    "arena_id": args.arena_id,
    "camera_id": args.camera_id,
    "protocol_name": args.protocol_name,
    "require_chaser_metadata": args.require_chaser_metadata,
    "root_count": len(root_paths),
    "require_steps_ok": args.require_step_ok,
    "target_count": len(selected),
    "path_contains": args.path_contains,
    "targets": selected,
}
(args.output.parent / "manifest_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

target_count=$(wc -l < "$RUN_DIR/recordings.txt" | tr -d ' ')
if [[ "$target_count" == "0" ]]; then
  echo "No analysis zarr targets selected."
  exit 0
fi

CONFIG_FILE="$RUN_DIR/config.env"
: > "$CONFIG_FILE"
write_var() {
  printf '%s=%q\n' "$1" "$2" >> "$CONFIG_FILE"
}

write_var REPO_ROOT "$PALETTE_REPO"
write_var EXPECTED_COMMIT "$EXPECTED_COMMIT"
write_var PROTOCOL_PROFILE "$PROTOCOL_PROFILE_RESOLVED"
write_var CHASER_ANALYSIS_PROFILE "$CHASER_ANALYSIS_PROFILE_RESOLVED"
write_var RUN_MOVEMENT "$RUN_MOVEMENT"
write_var RUN_STIMULUS_EPOCH "$RUN_STIMULUS_EPOCH"
write_var RUN_DETECTION_OCCUPANCY "$RUN_DETECTION_OCCUPANCY"
write_var RUN_CHASER_DISTANCE "$RUN_CHASER_DISTANCE"
write_var RUN_QUADRANT_OCCUPANCY "$RUN_QUADRANT_OCCUPANCY"
write_var RUN_NEAR_FIELD_OCCUPANCY "$RUN_NEAR_FIELD_OCCUPANCY"
write_var RUN_EPOCH_BEHAVIOR "$RUN_EPOCH_BEHAVIOR"
write_var RUN_EGOCENTRIC "$RUN_EGOCENTRIC"
write_var RUN_ESCAPE_FREEZE "$RUN_ESCAPE_FREEZE"
write_var RUN_EYE_ANGLES "$RUN_EYE_ANGLES"
write_var RUN_GAZE_TRACKING "$RUN_GAZE_TRACKING"
write_var OVERWRITE "$OVERWRITE"
write_var NO_PNG "$NO_PNG"
write_var NO_INTERACTIVE_SPEC "$NO_INTERACTIVE_SPEC"
write_var TRACK_RUN "$TRACK_RUN"
write_var SWIM_BOUT_RUN "$SWIM_BOUT_RUN"
write_var BOUT_KINEMATICS_RUN "$BOUT_KINEMATICS_RUN"
write_var TRACK_RUN_SELECTION "$TRACK_RUN_SELECTION"
write_var SWIM_BOUT_RUN_SELECTION "$SWIM_BOUT_RUN_SELECTION"
write_var BOUT_KINEMATICS_RUN_SELECTION "$BOUT_KINEMATICS_RUN_SELECTION"
write_var INCLUDE_EYE_GAZE "$INCLUDE_EYE_GAZE"
write_var EPOCH_RUN "$EPOCH_RUN"
write_var EPOCH_RUN_SELECTION "$EPOCH_RUN_SELECTION"
write_var OCCUPANCY_RUN "$OCCUPANCY_RUN"
write_var CHASER_DISTANCE_RUN "$CHASER_DISTANCE_RUN"
write_var CHASER_DISTANCE_RUN_SELECTION "$CHASER_DISTANCE_RUN_SELECTION"
write_var QUADRANT_OCCUPANCY_COMPONENT "$QUADRANT_OCCUPANCY_COMPONENT"
write_var NEAR_FIELD_OCCUPANCY_COMPONENT "$NEAR_FIELD_OCCUPANCY_COMPONENT"
write_var EPOCH_BEHAVIOR_COMPONENT "$EPOCH_BEHAVIOR_COMPONENT"
write_var EGOCENTRIC_COMPONENT "$EGOCENTRIC_COMPONENT"
write_var ESCAPE_FREEZE_COMPONENT "$ESCAPE_FREEZE_COMPONENT"
write_var EYE_ANGLE_OUTPUT_RUN "$EYE_ANGLE_OUTPUT_RUN"
write_var EYE_ANGLE_RUN "$EYE_ANGLE_RUN"
write_var GAZE_EGOCENTRIC_COMPONENT "$GAZE_EGOCENTRIC_COMPONENT"
write_var GAZE_TRACKING_COMPONENT "$GAZE_TRACKING_COMPONENT"
write_var OCCUPANCY_BIN_SIZE "$OCCUPANCY_BIN_SIZE"
write_var OCCUPANCY_SMOOTH_SIGMA "$OCCUPANCY_SMOOTH_SIGMA"
write_var CHASER_THRESHOLD_MM "$CHASER_THRESHOLD_MM"
write_var CHASER_DISTRIBUTION_BIN_WIDTH_MM "$CHASER_DISTRIBUTION_BIN_WIDTH_MM"
write_var TRACK_ID "$TRACK_ID"
write_var SPEED_LEVEL "$SPEED_LEVEL"
write_var EGOCENTRIC_DISTANCE_BIN_WIDTH_MM "$EGOCENTRIC_DISTANCE_BIN_WIDTH_MM"
write_var EGOCENTRIC_BEARING_BIN_WIDTH_DEG "$EGOCENTRIC_BEARING_BIN_WIDTH_DEG"
write_var ESCAPE_CHASER_INDEX "$ESCAPE_CHASER_INDEX"
write_var ESCAPE_TRIGGER_RADIUS_MM "$ESCAPE_TRIGGER_RADIUS_MM"
write_var ESCAPE_PATH_THRESHOLD_MM "$ESCAPE_PATH_THRESHOLD_MM"

JOB_SCRIPT="$RUN_DIR/run_one_zarr.sh"
cat > "$JOB_SCRIPT" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
source "$RUN_DIR/config.env"

index="${LSB_JOBINDEX:-1}"
zarr_path="$(sed -n "${index}p" "$RUN_DIR/recordings.txt")"
if [[ -z "$zarr_path" ]]; then
  echo "No zarr path for LSB_JOBINDEX=${index}" >&2
  exit 2
fi

safe_name="$(basename "$zarr_path" | tr -c 'A-Za-z0-9._-' '_')"
json_dir="$RUN_DIR/json/${index}_${safe_name}"
mkdir -p "$json_dir" "$RUN_DIR/status" "$RUN_DIR/matplotlib/${index}"
export MPLCONFIGDIR="$RUN_DIR/matplotlib/${index}"
# Array elements publish Zarr artifacts only.  The dependent one-slot
# finalizer owns every SQLite mutation for this campaign.
export PALETTE_DISABLE_REGISTRY_WRITES=1

cd "$REPO_ROOT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "Palette commit mismatch on execution host" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Palette checkout became dirty on execution host" >&2
  exit 2
fi
py="$REPO_ROOT/scripts/py"

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then overwrite_args+=(--overwrite); fi
png_args=()
if [[ "$NO_PNG" == "1" ]]; then png_args+=(--no-png); fi
interactive_args=()
if [[ "$NO_INTERACTIVE_SPEC" == "1" ]]; then interactive_args+=(--no-interactive-spec); fi

write_status() {
  local state="$1"
  "$py" - "$RUN_DIR/status/${index}_${safe_name}.json" "$state" "$index" "$zarr_path" \
    "$EXPECTED_COMMIT" "$RUN_MOVEMENT" "$TRACK_RUN" "$RUN_EYE_ANGLES" \
    "$EYE_ANGLE_RUN" "$EYE_ANGLE_OUTPUT_RUN" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_id": "palette.chaser_analytics_target_receipt.v1",
    "schema_version": 1,
    "status": sys.argv[2],
    "registry_write_mode": "deferred_to_serial_finalizer",
    "array_index": int(sys.argv[3]),
    "zarr_path": sys.argv[4],
    "palette_commit": sys.argv[5],
    "requested_publications": {
        "track_kinematics": {
            "requested": sys.argv[6] == "1",
            "run_name": f"offline/{sys.argv[7]}",
        },
        "eye_angles": {
            "requested": sys.argv[8] == "1",
            "input_selector": sys.argv[9],
            "output_run_name": sys.argv[10],
        },
    },
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_json_step() {
  local step="$1"
  shift
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${step}: ${zarr_path}"
  if ! "$@" > "$json_dir/${step}.json"; then
    write_status failed
    return 1
  fi
}

run_log_step() {
  local step="$1"
  shift
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${step}: ${zarr_path}"
  if ! "$@" > "$json_dir/${step}.log" 2>&1; then
    write_status failed
    return 1
  fi
}

trap 'write_status failed' ERR
write_status started

if [[ "$RUN_MOVEMENT" == "1" ]]; then
  movement_args=()
  if [[ "$INCLUDE_EYE_GAZE" == "0" ]]; then
    movement_args+=(--no-include-eye-gaze)
  fi
  run_log_step movement_bout "$py" -m fisheye.utils.run_movement_bout_batch_pipeline \
    "$zarr_path" \
    --apply \
    --jobs 1 \
    --track-run "$TRACK_RUN" \
    --swim-bout-run "$SWIM_BOUT_RUN" \
    --bout-kinematics-run "$BOUT_KINEMATICS_RUN" \
    "${movement_args[@]}" \
    "${overwrite_args[@]}" \
    --json-report "$json_dir/movement_bout.json" \
    --markdown-report "$json_dir/movement_bout.md"
fi

if [[ "$RUN_STIMULUS_EPOCH" == "1" ]]; then
  run_json_step stimulus_epoch "$py" -m fisheye.analysis.stimulus_epoch_runs \
    "$zarr_path" \
    --run-name "$EPOCH_RUN" \
    --protocol-profile "$PROTOCOL_PROFILE" \
    --apply \
    "${overwrite_args[@]}" \
    --json
fi

if [[ "$RUN_DETECTION_OCCUPANCY" == "1" ]]; then
  run_json_step detection_occupancy "$py" -m fisheye.analysis.detection_occupancy_runs \
    "$zarr_path" \
    --run-name "$OCCUPANCY_RUN" \
    --stimulus-epoch-run "$EPOCH_RUN" \
    --source active \
    --bin-size "$OCCUPANCY_BIN_SIZE" \
    --smooth-sigma "$OCCUPANCY_SMOOTH_SIGMA" \
    --apply \
    "${overwrite_args[@]}" \
    "${png_args[@]}" \
    --json
fi

if [[ "$RUN_CHASER_DISTANCE" == "1" ]]; then
  run_json_step chaser_distance "$py" -m fisheye.analysis.chaser_distance_runs \
    "$zarr_path" \
    --run-name "$CHASER_DISTANCE_RUN" \
    --stimulus-epoch-run "$EPOCH_RUN" \
    --source active \
    --threshold-mm "$CHASER_THRESHOLD_MM" \
    --distribution-bin-width-mm "$CHASER_DISTRIBUTION_BIN_WIDTH_MM" \
    --apply \
    "${overwrite_args[@]}" \
    "${png_args[@]}" \
    "${interactive_args[@]}" \
    --json
fi

if [[ "$RUN_QUADRANT_OCCUPANCY" == "1" ]]; then
  primary_interactive_args=("${interactive_args[@]}")
  if [[ "$NO_INTERACTIVE_SPEC" == "1" ]]; then
    primary_interactive_args+=(--no-run-level-interactive-spec)
  fi
  run_json_step quadrant_occupancy "$py" -m fisheye.analysis.chaser_quadrant_occupancy \
    "$zarr_path" \
    --chaser-distance-run "$CHASER_DISTANCE_RUN" \
    --protocol-profile "$PROTOCOL_PROFILE" \
    --component-name "$QUADRANT_OCCUPANCY_COMPONENT" \
    --apply \
    "${overwrite_args[@]}" \
    "${png_args[@]}" \
    "${primary_interactive_args[@]}" \
    --json
fi

if [[ "$RUN_NEAR_FIELD_OCCUPANCY" == "1" ]]; then
  run_json_step near_field_occupancy "$py" -m fisheye.analysis.chaser_near_field_occupancy \
    "$zarr_path" \
    --chaser-distance-run "$CHASER_DISTANCE_RUN" \
    --quadrant-occupancy-component "$QUADRANT_OCCUPANCY_COMPONENT" \
    --component-name "$NEAR_FIELD_OCCUPANCY_COMPONENT" \
    --apply \
    "${overwrite_args[@]}" \
    "${png_args[@]}" \
    "${interactive_args[@]}" \
    --json
fi

if [[ "$RUN_EPOCH_BEHAVIOR" == "1" ]]; then
  epoch_speed_args=()
  if [[ -n "$SPEED_LEVEL" ]]; then
    epoch_speed_args+=(--speed-level "$SPEED_LEVEL")
  fi
  run_log_step epoch_behavior "$py" -m fisheye.analysis.chaser_epoch_behavior_summary \
    "$zarr_path" \
    --chaser-distance-run "$CHASER_DISTANCE_RUN" \
    --component-name "$EPOCH_BEHAVIOR_COMPONENT" \
    --swim-bout-run "$SWIM_BOUT_RUN" \
    --track-kinematics-run "$TRACK_RUN" \
    --track-kinematics-scope offline \
    --track-id "$TRACK_ID" \
    "${epoch_speed_args[@]}" \
    "${overwrite_args[@]}"
fi

if [[ "$RUN_EGOCENTRIC" == "1" ]]; then
  egocentric_args=()
  if [[ -n "$EGOCENTRIC_DISTANCE_BIN_WIDTH_MM" ]]; then
    egocentric_args+=(--distance-bin-width-mm "$EGOCENTRIC_DISTANCE_BIN_WIDTH_MM")
  fi
  if [[ -n "$EGOCENTRIC_BEARING_BIN_WIDTH_DEG" ]]; then
    egocentric_args+=(--bearing-bin-width-deg "$EGOCENTRIC_BEARING_BIN_WIDTH_DEG")
  fi
  run_json_step egocentric "$py" -m fisheye.analysis.chaser_egocentric_bearing \
    "$zarr_path" \
    --chaser-distance-run "$CHASER_DISTANCE_RUN" \
    --track-kinematics-run "$TRACK_RUN" \
    --track-scope offline \
    --track-id "$TRACK_ID" \
    --component-name "$EGOCENTRIC_COMPONENT" \
    --apply \
    "${overwrite_args[@]}" \
    "${png_args[@]}" \
    "${interactive_args[@]}" \
    "${egocentric_args[@]}" \
    --json
fi

if [[ "$RUN_EYE_ANGLES" == "1" ]]; then
  if "$py" - "$zarr_path" "$EYE_ANGLE_RUN" <<'PY'
import sys
from pathlib import Path

from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    resolve_authoritative_run_name,
)

root = open_zarr_root(Path(sys.argv[1]), mode="r")
parent = root.get("analysis/eye_angle_runs")
requested = sys.argv[2]
name = (
    resolve_authoritative_run_name(parent)
    if parent is not None and requested == "latest"
    else requested
)
complete = bool(
    parent is not None
    and name
    and name in parent
    and is_run_complete_in_parent(parent, parent[name])
)
if complete:
    print(f"reuse_complete_eye_angle_run={name}")
raise SystemExit(0 if complete else 1)
PY
  then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eye_angles: reusing complete run"
  else
    run_log_step eye_angles "$py" -m fisheye.utils.execute_analysis_workflow \
      "$zarr_path" \
      --execution-id "${GAZE_TRACKING_COMPONENT}_${index}" \
      --num-workers "${LSB_DJOB_NUMPROC:-1}" \
      --report "$json_dir/eye_angle_workflow_report.json" \
      --apply \
      --target eye_angles \
      --output-run "eye_angles=${EYE_ANGLE_OUTPUT_RUN}"
  fi
fi

if [[ "$RUN_GAZE_TRACKING" == "1" ]]; then
  run_log_step gaze_convention_validation "$py" -m fisheye.analysis.gaze_convention_validation \
    "$zarr_path" \
    --eye-angle-run "$EYE_ANGLE_RUN" \
    --windows 12 \
    --rows-per-window 256 \
    --review-png "$json_dir/gaze_convention_review.png" \
    --review-panels 12 \
    --json-output "$json_dir/gaze_convention_validation.json" \
    --fail-on-error

  run_json_step chaser_gaze_tracking "$py" -m fisheye.analysis.chaser_gaze_tracking \
    "$zarr_path" \
    --chaser-distance-run "$CHASER_DISTANCE_RUN" \
    --egocentric-component "$GAZE_EGOCENTRIC_COMPONENT" \
    --eye-angle-run "$EYE_ANGLE_RUN" \
    --component-name "$GAZE_TRACKING_COMPONENT" \
    --apply \
    "${overwrite_args[@]}" \
    "${png_args[@]}"
fi

if [[ "$RUN_ESCAPE_FREEZE" == "1" ]]; then
  escape_args=()
  if [[ -n "$ESCAPE_TRIGGER_RADIUS_MM" ]]; then
    escape_args+=(--trigger-radius-mm "$ESCAPE_TRIGGER_RADIUS_MM")
  fi
  if [[ -n "$ESCAPE_PATH_THRESHOLD_MM" ]]; then
    escape_args+=(--escape-path-threshold-mm "$ESCAPE_PATH_THRESHOLD_MM")
  fi
  escape_indices=()
  if [[ "$ESCAPE_CHASER_INDEX" == "all_applicable" ]]; then
    mapfile -t escape_indices < <("$py" - "$zarr_path" "$CHASER_DISTANCE_RUN" <<'PY'
import sys
from pathlib import Path

import numpy as np

from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name

root = open_zarr_root(Path(sys.argv[1]), mode="r")
parent = root.get("analysis/chaser_distance_runs")
if parent is None:
    raise ValueError("Archive has no analysis/chaser_distance_runs group.")
requested = str(sys.argv[2]).strip()
resolved = requested
if not resolved or resolved == "latest":
    resolved = resolve_authoritative_run_name(parent) or ""
if not resolved or resolved not in parent:
    raise ValueError(
        "No authoritative chaser-distance run found for escape/freeze discovery."
    )
chasers = parent[resolved]["chasers"]
indices = np.asarray(chasers["chaser_index"][:], dtype=np.int64).reshape(-1)
if "behavior_class_label_bytes" not in chasers:
    raise ValueError("all_applicable escape/freeze selection requires persisted behavior classes")
labels = [
    decode_null_terminated_text(row).strip()
    for row in np.asarray(chasers["behavior_class_label_bytes"][:])
]
for index, label in zip(indices, labels):
    if label == "aggressive":
        print(int(index))
PY
    )
  else
    escape_indices+=("$ESCAPE_CHASER_INDEX")
  fi
  for escape_chaser_index in "${escape_indices[@]}"; do
    escape_component="${ESCAPE_FREEZE_COMPONENT//\{chaser_index\}/$escape_chaser_index}"
    run_json_step "escape_freeze_chaser_${escape_chaser_index}" \
      "$py" -m fisheye.analysis.chaser_escape_freeze_summary \
      "$zarr_path" \
      --chaser-distance-run "$CHASER_DISTANCE_RUN" \
      --component-name "$escape_component" \
      --chaser-index "$escape_chaser_index" \
      --apply \
      "${overwrite_args[@]}" \
      "${png_args[@]}" \
      "${escape_args[@]}"
  done
fi

write_status complete
JOB
chmod +x "$JOB_SCRIPT"

FINALIZER_SCRIPT="$RUN_DIR/run_registry_finalizer.sh"
FINALIZER_REPORT="$RUN_DIR/registry_finalizer.json"
DERIVED_STAGE_SELECTORS=()
if [[ "$RUN_MOVEMENT" == "1" ]]; then
  DERIVED_STAGE_SELECTORS+=("track_kinematics=offline/${TRACK_RUN}")
fi
if [[ "$RUN_EYE_ANGLES" == "1" ]]; then
  DERIVED_STAGE_SELECTORS+=("eye_angles=latest")
fi

repo_q="$(printf '%q' "$PALETTE_REPO")"
commit_q="$(printf '%q' "$EXPECTED_COMMIT")"
targets_q="$(printf '%q' "$RUN_DIR/recordings.txt")"
status_q="$(printf '%q' "$RUN_DIR/status")"
registry_q="$(printf '%q' "$REGISTRY")"
report_q="$(printf '%q' "$FINALIZER_REPORT")"
stage_args=""
for selector in "${DERIVED_STAGE_SELECTORS[@]}"; do
  printf -v selector_q '%q' "$selector"
  stage_args+=" --stage-selector ${selector_q}"
done
cat >"$FINALIZER_SCRIPT" <<JOB
#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd ${repo_q}
[[ "\$(git rev-parse HEAD)" == ${commit_q} ]] || {
  printf 'Palette commit mismatch in chaser registry finalizer.\n' >&2
  exit 2
}
[[ -z "\$(git status --porcelain)" ]] || {
  printf 'Refusing dirty Palette checkout in chaser registry finalizer.\n' >&2
  exit 2
}
unset PALETTE_DISABLE_REGISTRY_WRITES
export PYTHONPATH=${repo_q}/src:${repo_q}\${PYTHONPATH:+:\${PYTHONPATH}}
scripts/py -m fisheye.analysis_workflows.registry_finalize \
  --target-list ${targets_q} \
  --status-dir ${status_q}${stage_args} \
  --registry ${registry_q} \
  --apply \
  --output-json ${report_q}
JOB
chmod +x "$FINALIZER_SCRIPT"

BSUB_ARGS=(-J "chaser_analytics[1-${target_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/%J_%I.out" -eo "${RUN_DIR}/%J_%I.err")
if [[ "${#DEPENDENCY_DONE[@]}" -gt 0 ]]; then
  dependency_expression=""
  for dependency_job_id in "${DEPENDENCY_DONE[@]}"; do
    if [[ -n "$dependency_expression" ]]; then dependency_expression+=" && "; fi
    dependency_expression+="done(${dependency_job_id})"
  done
  BSUB_ARGS+=(-w "$dependency_expression")
fi
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS=(-q "$QUEUE" "${BSUB_ARGS[@]}")
fi

BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR")

echo "run_dir: $RUN_DIR"
echo "target_count: $target_count"
echo "resources: ncores=$NCORES mem_gb=$MEM_GB max_active=$MAX_ACTIVE queue=${QUEUE:-<default>}"
echo "job_script: $JOB_SCRIPT"
echo "registry_write_mode: deferred_to_serial_finalizer"
echo "registry_finalizer: $([[ "$SUBMIT_REGISTRY_FINALIZER" == "1" ]] && echo enabled || echo disabled)"
echo "registry_finalizer_script: $FINALIZER_SCRIPT"
echo "registry_finalizer_report: $FINALIZER_REPORT"
printf 'bsub_command:'
printf ' %q' "${BSUB_COMMAND[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry_run: no jobs submitted"
  exit 0
fi

if command -v bsub >/dev/null 2>&1; then
  submit_output="$("${BSUB_COMMAND[@]}")"
else
  if [[ -z "$SUBMIT_HOST" ]]; then
    echo "bsub is unavailable and --submit-host is empty" >&2
    exit 2
  fi
  printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
  submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
fi
printf '%s\n' "$submit_output"
job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
if [[ -z "$job_id" ]]; then
  echo "Could not parse an LSF job ID from submission output" >&2
  exit 2
fi
printf 'job_id=%s\n' "$job_id"

if [[ "$SUBMIT_REGISTRY_FINALIZER" == "1" && "${#DERIVED_STAGE_SELECTORS[@]}" -gt 0 ]]; then
  FINALIZER_BSUB_ARGS=(
    -J "chaser_registry_${RUN_ID}"
    -n 1
    -W "$REGISTRY_FINALIZER_WALLTIME"
    -R "span[hosts=1]"
    -R "rusage[mem=${REGISTRY_FINALIZER_MEM_GB}G]"
    -w "done(${job_id})"
    -oo "$RUN_DIR/registry_finalizer_%J.out"
    -eo "$RUN_DIR/registry_finalizer_%J.err"
  )
  if [[ -n "$QUEUE" ]]; then FINALIZER_BSUB_ARGS+=(-q "$QUEUE"); fi
  FINALIZER_COMMAND=(bsub "${FINALIZER_BSUB_ARGS[@]}" bash "$FINALIZER_SCRIPT")
  if command -v bsub >/dev/null 2>&1; then
    finalizer_output="$("${FINALIZER_COMMAND[@]}")"
  else
    printf -v remote_finalizer_command '%q ' "${FINALIZER_COMMAND[@]}"
    finalizer_output="$(ssh "$SUBMIT_HOST" "$remote_finalizer_command")"
  fi
  printf '%s\n' "$finalizer_output"
  finalizer_job_id="$(printf '%s\n' "$finalizer_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$finalizer_job_id" ]] || {
    echo "Could not parse registry finalizer LSF job ID" >&2
    exit 2
  }
  printf 'registry_finalizer_job_id=%s\n' "$finalizer_job_id"
  printf 'registry_finalizer_dependency=done(%s)\n' "$job_id"
  printf 'registry_finalizer_report=%s\n' "$FINALIZER_REPORT"
elif [[ "${#DERIVED_STAGE_SELECTORS[@]}" -gt 0 ]]; then
  echo "Registry writes were deferred. Run the rendered finalizer after the array succeeds:"
  echo "  bash $(printf '%q' "$FINALIZER_SCRIPT")"
fi
