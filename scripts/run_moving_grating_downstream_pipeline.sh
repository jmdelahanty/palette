#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_moving_grating_downstream_pipeline.sh [options] [analysis.zarr]

Runs Palette's canonical downstream analysis chain for a moving-grating
recording that already has detections/crops/refined keypoints:

  dish_mask -> arena_assignment/tracking -> track_kinematics -> swim_bout_runs -> stimulus_response

The default target is the current moving-grating canary. By default this script
prints the commands without mutating the Zarr. Pass --apply to write outputs.

Options:
  --apply                         Execute commands. Default: dry run.
  --overwrite                     Pass --overwrite to downstream writers when
                                  fixed run names already exist.
  --no-bouts                      Skip swim-bout detection and run
                                  stimulus_response with --no-bouts.
  --keypoint-run RUN              Keypoint run for track_kinematics.
                                  Default: refined/refined_keypoints_2026-03-03_23-45-34
  --crop-run RUN                  Crop rowset used for arena assignment and
                                  keypoint-aligned motion.
                                  Default: crop_2026-02-10_21-05-18
  --stimulus-run RUN              Stimulus run for stimulus_response.
                                  Default: stimulus_20260209_084518
  --track-run RUN                 Output offline track_kinematics run.
                                  Default: tk_hyst4_low2_latch_s005
  --bout-run RUN                  Output swim_bout_runs name.
                                  Default: bouts_tk_hyst4_low2_latch_s005_filtered_thr0p01
  --stimulus-response-run RUN     Output stimulus_response_runs name.
                                  Default: stimulus_response_tk_hyst4_low2_latch_s005_omr_canary
  --track-id N                    Track id for bout detection. Default: 0.
  --hysteresis-high-px PX         track_kinematics high threshold. Default: 4.
  --hysteresis-low-px PX          track_kinematics low threshold. Default: 2.
  --hysteresis-min-frames N       track_kinematics exit duration. Default: 3.
  --hysteresis-band-policy MODE   reset|latch. Default: latch.
  --smooth-seconds S              track_kinematics smoothing window. Default: 0.05.
  --smoothing-alignment MODE      centered|causal. Default: causal.
  --bout-threshold-mm-s V         swim-bout threshold. Default: 0.01.
  --bout-default-level LEVEL      Downstream bout level. Default: filtered.
  --moving-threshold-mm-s V       stimulus_response moving threshold. Default: 2.0.
  --camera-to-projector-offset-deg DEG
                                  Angular correction into camera-space degrees.
                                  Default: 180.0 for the current inverted-
                                  projector moving-grating canary.
  --bin-size-s S                  Stimulus-response temporal bin size. Default: 1.0.
  --no-visualizations             Do not write stimulus_response review plot
                                  artifacts. Default: write artifacts.
  -h, --help                      Show this help.

Example:
  scripts/run_moving_grating_downstream_pipeline.sh --apply
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_ZARR="/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr"

ZARR_PATH="$DEFAULT_ZARR"
APPLY=0
OVERWRITE=0
WITH_BOUTS=1
WRITE_VISUALIZATIONS=1

KEYPOINT_RUN="refined/refined_keypoints_2026-03-03_23-45-34"
CROP_RUN="crop_2026-02-10_21-05-18"
STIMULUS_RUN="stimulus_20260209_084518"
TRACK_RUN="tk_hyst4_low2_latch_s005"
BOUT_RUN="bouts_tk_hyst4_low2_latch_s005_filtered_thr0p01"
STIMULUS_RESPONSE_RUN="stimulus_response_tk_hyst4_low2_latch_s005_omr_canary"
TRACK_ID=0

HYSTERESIS_HIGH_PX=4
HYSTERESIS_LOW_PX=2
HYSTERESIS_MIN_FRAMES=3
HYSTERESIS_BAND_POLICY="latch"
SMOOTH_SECONDS=0.05
SMOOTHING_ALIGNMENT="causal"
BOUT_THRESHOLD_MM_S=0.01
BOUT_DEFAULT_LEVEL="filtered"
MOVING_THRESHOLD_MM_S=2.0
CAMERA_TO_PROJECTOR_OFFSET_DEG=180.0
BIN_SIZE_S=1.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-bouts)
      WITH_BOUTS=0
      shift
      ;;
    --keypoint-run)
      KEYPOINT_RUN="${2:-}"
      shift 2
      ;;
    --crop-run)
      CROP_RUN="${2:-}"
      shift 2
      ;;
    --stimulus-run)
      STIMULUS_RUN="${2:-}"
      shift 2
      ;;
    --track-run)
      TRACK_RUN="${2:-}"
      shift 2
      ;;
    --bout-run)
      BOUT_RUN="${2:-}"
      shift 2
      ;;
    --stimulus-response-run)
      STIMULUS_RESPONSE_RUN="${2:-}"
      shift 2
      ;;
    --track-id)
      TRACK_ID="${2:-}"
      shift 2
      ;;
    --hysteresis-high-px)
      HYSTERESIS_HIGH_PX="${2:-}"
      shift 2
      ;;
    --hysteresis-low-px)
      HYSTERESIS_LOW_PX="${2:-}"
      shift 2
      ;;
    --hysteresis-min-frames)
      HYSTERESIS_MIN_FRAMES="${2:-}"
      shift 2
      ;;
    --hysteresis-band-policy)
      HYSTERESIS_BAND_POLICY="${2:-}"
      shift 2
      ;;
    --smooth-seconds)
      SMOOTH_SECONDS="${2:-}"
      shift 2
      ;;
    --smoothing-alignment)
      SMOOTHING_ALIGNMENT="${2:-}"
      shift 2
      ;;
    --bout-threshold-mm-s)
      BOUT_THRESHOLD_MM_S="${2:-}"
      shift 2
      ;;
    --bout-default-level)
      BOUT_DEFAULT_LEVEL="${2:-}"
      shift 2
      ;;
    --moving-threshold-mm-s)
      MOVING_THRESHOLD_MM_S="${2:-}"
      shift 2
      ;;
    --camera-to-projector-offset-deg)
      CAMERA_TO_PROJECTOR_OFFSET_DEG="${2:-}"
      shift 2
      ;;
    --bin-size-s)
      BIN_SIZE_S="${2:-}"
      shift 2
      ;;
    --no-visualizations)
      WRITE_VISUALIZATIONS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
    *)
      ZARR_PATH="$1"
      shift
      ;;
  esac
done

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

if [[ ! -d "$ZARR_PATH" ]]; then
  echo "Analysis Zarr not found: $ZARR_PATH" >&2
  exit 2
fi

if [[ "$SMOOTHING_ALIGNMENT" != "centered" && "$SMOOTHING_ALIGNMENT" != "causal" ]]; then
  echo "--smoothing-alignment must be centered or causal (got: $SMOOTHING_ALIGNMENT)" >&2
  exit 2
fi

if [[ "$HYSTERESIS_BAND_POLICY" != "reset" && "$HYSTERESIS_BAND_POLICY" != "latch" ]]; then
  echo "--hysteresis-band-policy must be reset or latch (got: $HYSTERESIS_BAND_POLICY)" >&2
  exit 2
fi

section() {
  printf '\n=== %s ===\n' "$1"
}

print_cmd() {
  printf '  %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$APPLY" -eq 1 ]]; then
    "$@"
  fi
}

section "Moving-Grating Downstream Pipeline"
echo "zarr_path=$ZARR_PATH"
echo "apply=$APPLY"
echo "keypoint_run=$KEYPOINT_RUN"
echo "crop_run=$CROP_RUN"
echo "stimulus_run=$STIMULUS_RUN"
echo "track_run=$TRACK_RUN"
echo "hysteresis_band_policy=$HYSTERESIS_BAND_POLICY"
echo "with_bouts=$WITH_BOUTS"
if [[ "$WITH_BOUTS" -eq 1 ]]; then
  echo "bout_run=$BOUT_RUN"
  echo "bout_default_level=$BOUT_DEFAULT_LEVEL"
fi
echo "stimulus_response_run=$STIMULUS_RESPONSE_RUN"
echo "camera_to_projector_offset_deg=$CAMERA_TO_PROJECTOR_OFFSET_DEG"
echo "write_visualizations=$WRITE_VISUALIZATIONS"

if [[ "$APPLY" -ne 1 ]]; then
  echo
  echo "Dry run only. Re-run with --apply to write analysis outputs."
fi

section "Arena Assignment / Tracking"
run_cmd scripts/py -m fisheye.tracking.arena_assignment "$ZARR_PATH" \
  --source-rowset "crop_runs/$CROP_RUN"

section "Track Kinematics"
run_cmd scripts/py -m fisheye.analysis.track_kinematics "$ZARR_PATH" \
  --offline-only \
  --keypoint-run "$KEYPOINT_RUN" \
  --hysteresis-high-px "$HYSTERESIS_HIGH_PX" \
  --hysteresis-low-px "$HYSTERESIS_LOW_PX" \
  --hysteresis-min-frames "$HYSTERESIS_MIN_FRAMES" \
  --hysteresis-band-policy "$HYSTERESIS_BAND_POLICY" \
  --smooth-seconds "$SMOOTH_SECONDS" \
  --smoothing-alignment "$SMOOTHING_ALIGNMENT" \
  --offline-run-name "$TRACK_RUN"

if [[ "$WITH_BOUTS" -eq 1 ]]; then
  section "Swim-Bout Candidates"
  BOUT_ARGS=(
    scripts/py -m fisheye.analysis.detect_bouts_multi_level "$ZARR_PATH"
    --track-kinematics-run "$TRACK_RUN"
    --track-id "$TRACK_ID"
    --run-name "$BOUT_RUN"
    --threshold-mm "$BOUT_THRESHOLD_MM_S"
    --default-level "$BOUT_DEFAULT_LEVEL"
    --boundary-mode threshold
  )
  if [[ "$OVERWRITE" -eq 1 ]]; then
    BOUT_ARGS+=(--overwrite)
  fi
  run_cmd "${BOUT_ARGS[@]}"
fi

section "Stimulus Response"
STIM_ARGS=(
  scripts/py -m fisheye.analysis.stimulus_response "$ZARR_PATH"
  --track-kinematics-type offline
  --track-kinematics-run "$TRACK_RUN"
  --stimulus-run "$STIMULUS_RUN"
  --moving-threshold-mm-s "$MOVING_THRESHOLD_MM_S"
  --camera-to-projector-offset-deg "$CAMERA_TO_PROJECTOR_OFFSET_DEG"
  --bin-size-s "$BIN_SIZE_S"
  --run-name "$STIMULUS_RESPONSE_RUN"
)
if [[ "$WITH_BOUTS" -eq 1 ]]; then
  STIM_ARGS+=(--bout-run "$BOUT_RUN")
else
  STIM_ARGS+=(--no-bouts)
fi
if [[ "$OVERWRITE" -eq 1 ]]; then
  STIM_ARGS+=(--overwrite)
fi
if [[ "$WRITE_VISUALIZATIONS" -eq 1 ]]; then
  STIM_ARGS+=(--write-zarr-artifacts)
fi
run_cmd "${STIM_ARGS[@]}"

if [[ "$APPLY" -eq 1 ]]; then
  section "Outputs"
  echo "track_kinematics=analysis/track_kinematics_runs/offline/$TRACK_RUN"
  if [[ "$WITH_BOUTS" -eq 1 ]]; then
    echo "swim_bouts=analysis/swim_bout_runs/$BOUT_RUN"
  fi
  echo "stimulus_response=analysis/stimulus_response_runs/$STIMULUS_RESPONSE_RUN"
fi
