#!/usr/bin/env bash
set -euo pipefail

repo=/tmp/palette-pose-head-crops-20260914
run_root=/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/pose_head_merged_v1/pose_head_192_traditional_v1_reuse_v001/runs
run_name=yolo11n_pose_192_100e_20260914
run_dir="$run_root/$run_name"
log_path="$run_root/$run_name.log"
status_path="$run_root/$run_name.status"
pid_path="$run_root/$run_name.pid"

exec >"$log_path" 2>&1
trap 'rc=$?; printf "exit_code=%s\nfinished_utc=%s\n" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$status_path"' EXIT
printf '%s\n' "$$" >"$pid_path"
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

export PALETTE_PYTHON=/home/delahantyj@hhmi.org/miniconda3/envs/palette-py311-gpu/bin/python
cd "$repo"

scripts/py -m fisheye.training.train_pose \
  experimental_training/yolo11n_pose_100e_v001.yaml \
  --run-name "$run_name" \
  --project "$run_root" \
  --no-log-registry \
  --export-onnx \
  --onnx-opset 17 \
  --onnx-batch 1

scripts/py experimental_training/finalize_pose_onnx.py \
  --run-dir "$run_dir" \
  --config "$run_dir/inputs/yolo11n_pose_100e_v001.yaml" \
  --expected-epochs 100

printf 'validated_bundle=%s\n' "$run_dir/model_sources/pose/$run_name"
