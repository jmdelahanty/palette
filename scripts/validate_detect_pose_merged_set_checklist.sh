#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/validate_detect_pose_merged_set_checklist.sh [options]

Runs operator validation for detect-vs-pose merged-set checklist items:
1) Training-purpose merged datasets are excluded from pose source selection.
2) Pose runs appear in `check_training_registry --view models`.
3) Detect and pose merged exports are deterministic for identical manifest+seed.

Options:
  --registry PATH              Registry SQLite path
                               (default: /nvme1/palette_registry.sqlite)
  --target-contains TEXT       Substring used for preflight dataset selection
                               (default: 2026-01-28T21-47-47Z_arena_1_DefaultScreen_training.zarr)
  --split SPEC                 Split spec for merged exports (default: 0.8/0.2)
  --seed N                     Seed for merged exports (default: 123)
  --tmp-root DIR               Temp parent directory (default: /tmp)
  -h, --help                   Show this help
EOF
}

REGISTRY="/nvme1/palette_registry.sqlite"
TARGET_CONTAINS="2026-01-28T21-47-47Z_arena_1_DefaultScreen_training.zarr"
SPLIT_SPEC="0.8/0.2"
SEED="123"
TMP_ROOT="/tmp"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --target-contains)
      TARGET_CONTAINS="$2"
      shift 2
      ;;
    --split)
      SPLIT_SPEC="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --tmp-root)
      TMP_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$REGISTRY" ]]; then
  echo "Registry not found: $REGISTRY" >&2
  exit 2
fi

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "ripgrep (rg) is required for log extraction." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="${TMP_ROOT%/}/prov_checklist_validate_${STAMP}"
mkdir -p "$TMP_DIR"

section() {
  printf '\n=== %s ===\n' "$1"
}

echo "TMP_DIR=$TMP_DIR"
echo "REGISTRY=$REGISTRY"
echo "TARGET_CONTAINS=$TARGET_CONTAINS"
echo "SPLIT_SPEC=$SPLIT_SPEC"
echo "SEED=$SEED"

section "Merged Dataset Count In Registry"
scripts/py -c "import sqlite3; c=sqlite3.connect('$REGISTRY'); print(c.execute(\"select count(*) from datasets where lower(zarr_path) like '%_merged.zarr%'\").fetchone()[0]); c.close()" \
  >"$TMP_DIR/merged_dataset_count.txt"
cat "$TMP_DIR/merged_dataset_count.txt"

section "Pose Prefilter Exclusion Check"
set +e
scripts/py -m fisheye.utils.prepare_pose_training_from_registry \
  --registry "$REGISTRY" \
  --path-contains "_merged.zarr" \
  --dry-run >"$TMP_DIR/pose_prefilter_merged.txt" 2>&1
PREFILTER_EXIT=$?
set -e
echo "exit_code=$PREFILTER_EXIT"
cat "$TMP_DIR/pose_prefilter_merged.txt"

section "Models View Pose/Keypoint Hit Check"
scripts/py -m fisheye.utils.check_training_registry \
  --registry "$REGISTRY" \
  --view models \
  --no-rich \
  --limit 200 >"$TMP_DIR/models.txt"
rg -n "pose|_pose_|keypoint" "$TMP_DIR/models.txt" >"$TMP_DIR/models_pose_hits.txt" || true
if [[ -s "$TMP_DIR/models_pose_hits.txt" ]]; then
  cat "$TMP_DIR/models_pose_hits.txt"
else
  echo "(no pose/keypoint rows matched via regex)"
fi

section "Detect Preflight"
scripts/py -m fisheye.utils.prepare_detect_training_from_registry \
  --registry "$REGISTRY" \
  --path-contains "$TARGET_CONTAINS" \
  --source-type manual \
  --input-format gray \
  --out-config "$TMP_DIR/detect.yaml" \
  --out-manifest "$TMP_DIR/detect.manifest.json" \
  >"$TMP_DIR/detect_preflight.txt" 2>&1
tail -n 40 "$TMP_DIR/detect_preflight.txt"

section "Pose Preflight"
scripts/py -m fisheye.utils.prepare_pose_training_from_registry \
  --registry "$REGISTRY" \
  --path-contains "$TARGET_CONTAINS" \
  --source-type filtered \
  --input-format gray \
  --keypoint-run latest_traditional \
  --out-config "$TMP_DIR/pose.yaml" \
  --out-manifest "$TMP_DIR/pose.manifest.json" \
  >"$TMP_DIR/pose_preflight.txt" 2>&1
tail -n 40 "$TMP_DIR/pose_preflight.txt"

section "Detect Determinism Export A"
scripts/py -m fisheye.utils.export_detect_training_zarr \
  --manifest "$TMP_DIR/detect.manifest.json" \
  --merge \
  --split "$SPLIT_SPEC" \
  --seed "$SEED" \
  --out-zarr "$TMP_DIR/detect_A.zarr" \
  --out-dir "$TMP_DIR/detect_A" \
  --overwrite \
  >"$TMP_DIR/detect_export_A.txt" 2>&1
tail -n 30 "$TMP_DIR/detect_export_A.txt"

section "Detect Determinism Export B"
scripts/py -m fisheye.utils.export_detect_training_zarr \
  --manifest "$TMP_DIR/detect.manifest.json" \
  --merge \
  --split "$SPLIT_SPEC" \
  --seed "$SEED" \
  --out-zarr "$TMP_DIR/detect_B.zarr" \
  --out-dir "$TMP_DIR/detect_B" \
  --overwrite \
  >"$TMP_DIR/detect_export_B.txt" 2>&1
tail -n 30 "$TMP_DIR/detect_export_B.txt"

section "Pose Determinism Export A"
scripts/py -m fisheye.utils.export_keypoint_training_zarr \
  --manifest "$TMP_DIR/pose.manifest.json" \
  --merge \
  --split "$SPLIT_SPEC" \
  --seed "$SEED" \
  --out-zarr "$TMP_DIR/pose_A.zarr" \
  --out-dir "$TMP_DIR/pose_A" \
  --overwrite \
  >"$TMP_DIR/pose_export_A.txt" 2>&1
tail -n 30 "$TMP_DIR/pose_export_A.txt"

section "Pose Determinism Export B"
scripts/py -m fisheye.utils.export_keypoint_training_zarr \
  --manifest "$TMP_DIR/pose.manifest.json" \
  --merge \
  --split "$SPLIT_SPEC" \
  --seed "$SEED" \
  --out-zarr "$TMP_DIR/pose_B.zarr" \
  --out-dir "$TMP_DIR/pose_B" \
  --overwrite \
  >"$TMP_DIR/pose_export_B.txt" 2>&1
tail -n 30 "$TMP_DIR/pose_export_B.txt"

section "Split Fingerprint Comparison"
TMP_DIR="$TMP_DIR" scripts/py - <<'PY' > "$TMP_DIR/split_determinism.txt"
import hashlib
import os
from pathlib import Path

import numpy as np
import zarr

tmp_dir = Path(os.environ["TMP_DIR"])

def fingerprint(zarr_path: Path):
    root = zarr.open_group(str(zarr_path), mode="r")
    out = {}
    for name in ("splits/train_indices", "splits/val_indices", "splits/test_indices"):
        try:
            arr = np.asarray(root[name], dtype=np.int64)
            out[name] = {
                "shape": tuple(arr.shape),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            }
        except Exception:
            out[name] = None
    return out

detect_a = fingerprint(tmp_dir / "detect_A.zarr")
detect_b = fingerprint(tmp_dir / "detect_B.zarr")
pose_a = fingerprint(tmp_dir / "pose_A.zarr")
pose_b = fingerprint(tmp_dir / "pose_B.zarr")

print(f"detect_split_match={detect_a == detect_b}")
print(f"pose_split_match={pose_a == pose_b}")
print(f"detect_A={detect_a}")
print(f"detect_B={detect_b}")
print(f"pose_A={pose_a}")
print(f"pose_B={pose_b}")
PY
cat "$TMP_DIR/split_determinism.txt"

PREFILTER_PASS=0
MODELS_PASS=0
DETECT_SPLIT_PASS=0
POSE_SPLIT_PASS=0

if rg -q "Skipped [0-9]+ training-artifact dataset\\(s\\)" "$TMP_DIR/pose_prefilter_merged.txt"; then
  PREFILTER_PASS=1
fi
if [[ -s "$TMP_DIR/models_pose_hits.txt" ]]; then
  MODELS_PASS=1
fi
if rg -q "^detect_split_match=True$" "$TMP_DIR/split_determinism.txt"; then
  DETECT_SPLIT_PASS=1
fi
if rg -q "^pose_split_match=True$" "$TMP_DIR/split_determinism.txt"; then
  POSE_SPLIT_PASS=1
fi

section "Checklist-Oriented Summary"
echo "prefilter_excludes_training_artifacts=$PREFILTER_PASS"
echo "models_view_has_pose_rows=$MODELS_PASS"
echo "detect_split_deterministic=$DETECT_SPLIT_PASS"
echo "pose_split_deterministic=$POSE_SPLIT_PASS"
echo "artifact_dir=$TMP_DIR"
